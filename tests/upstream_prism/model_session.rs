//! Bounded, safe process transport for Rust-owned model and vocabulary tests.
//!
//! Requests run on persistent native worker lanes. Rust controls each operation,
//! callback decision and assertion; the bridge contains no test schedules.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use super::engine::Engine;
use super::{Result, invalid};

const MAX_FRAME: u64 = 16 * 1024 * 1024;
const MAX_STREAM: u64 = 64 * 1024 * 1024;
const MAX_STDERR: u64 = 1024 * 1024;
// Real-eval retains one model/speculation scope across all24 source prompts;
// its default96-token AR and verify schedules require at most~33k primitives.
const MAX_REQUESTS: u64 = 65536;
const MAX_CALLBACKS: usize = 65536;
const OPERATION_TIMEOUT: Duration = Duration::from_secs(90);
const SESSION_TIMEOUT: Duration = Duration::from_secs(240);
const REAL_EVAL_SESSION_TIMEOUT: Duration = Duration::from_mins(15);
const SAVE_LOAD_SESSION_TIMEOUT: Duration = Duration::from_mins(10);
const THREAD_SAFETY_SESSION_TIMEOUT: Duration = Duration::from_mins(30);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(10);
const QUANTIZATION_OPERATION_TIMEOUT: Duration = Duration::from_secs(3600);
const QUANTIZATION_SESSION_TIMEOUT: Duration = Duration::from_hours(4);

fn operation_timeout(quantization: bool, operation: &Value) -> Duration {
    if quantization && operation.get("Op").and_then(Value::as_str) == Some("qb-trace") {
        QUANTIZATION_OPERATION_TIMEOUT
    } else {
        OPERATION_TIMEOUT
    }
}

fn session_timeout(name: &str, quantization: bool, main_ci: bool) -> Duration {
    match (name, quantization, main_ci) {
        ("quantize-stats", true, true) => QUANTIZATION_SESSION_TIMEOUT,
        // The source owns one model/speculation scope across all 24 prompts.
        // Native ARM64 evidence reached prompt 13 at the ordinary 240s limit.
        // Keep every per-operation and transport bound while allowing the
        // complete source schedule to finish on the hosted reference runner.
        ("dspark-real-eval", false, true) => REAL_EVAL_SESSION_TIMEOUT,
        // These sources retain one live model/state or all concurrent contexts.
        // Hosted ARM64 reached 240s after 10 of the final 16 save/load decodes;
        // the eight thread workers had generated only 20-25 of 128 tokens each.
        // Keep both complete schedules and the ordinary per-operation bounds.
        ("tests/test-save-load-state.cpp", false, true) => SAVE_LOAD_SESSION_TIMEOUT,
        ("tests/test-thread-safety.cpp", false, true) => THREAD_SAFETY_SESSION_TIMEOUT,
        _ => SESSION_TIMEOUT,
    }
}

type Callback = fn(f32) -> bool;
pub(super) type TensorData = dyn Fn(&Value) -> Result<Value> + Send + Sync;
pub(super) type SamplerEvent = dyn Fn(&Value) -> Result<Value> + Send + Sync;
type Response = std::result::Result<Reply, String>;

pub(super) struct Reply {
    pub(super) value: Value,
    /// Raw IEEE-754 progress bits and the exact Rust callback decisions.
    pub(super) progress: Vec<(u32, bool)>,
}

struct Pending {
    sender: mpsc::SyncSender<Response>,
    progress: Option<Callback>,
    tensor_data: Option<Arc<TensorData>>,
    sampler_event: Option<Arc<SamplerEvent>>,
    callback_error: Option<String>,
    observations: Vec<(u32, bool)>,
    started: Instant,
    timeout: Duration,
}

struct Writer {
    stdin: Option<ChildStdin>,
    log: File,
    bytes: u64,
}

struct State {
    child: Mutex<Child>,
    writer: Mutex<Writer>,
    pending: Mutex<HashMap<u64, Pending>>,
    fatal: Mutex<Option<String>>,
    next_id: AtomicU64,
    closing_at: Mutex<Option<Instant>>,
    done: AtomicBool,
    started: Instant,
    session_timeout: Duration,
    quantization: bool,
}

// All transport mutations are short and contain no user callbacks. A poisoned
// lock is still recovered for mandatory kill/reap and pending-request cleanup.
fn lock<T>(value: &Mutex<T>) -> MutexGuard<'_, T> {
    value
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl State {
    fn fail(&self, message: impl Into<String>) {
        let message = message.into();
        {
            let mut fatal = lock(&self.fatal);
            if fatal.is_none() {
                *fatal = Some(message.clone());
            }
        }
        let pending = std::mem::take(&mut *lock(&self.pending));
        for (_, request) in pending {
            let _ = request.sender.send(Err(message.clone()));
        }
        // Never acquire writer here: killing the child must also interrupt a
        // pipe write that is blocked because the native reader stopped.
        let _ = lock(&self.child).kill();
    }

    fn check(&self) -> Result<()> {
        if let Some(error) = &*lock(&self.fatal) {
            return Err(invalid(error.clone()));
        }
        if self.started.elapsed() >= self.session_timeout {
            return Err(invalid("native session exceeded its lifetime"));
        }
        Ok(())
    }

    fn write(&self, value: &Value) -> Result<()> {
        self.check()?;
        let mut bytes = serde_json::to_vec(value)?;
        bytes.push(b'\n');
        if u64::try_from(bytes.len())? > MAX_FRAME {
            return Err(invalid("native session request frame exceeds bound"));
        }
        let mut writer = lock(&self.writer);
        writer.bytes = writer
            .bytes
            .checked_add(u64::try_from(bytes.len())?)
            .filter(|bytes| *bytes <= MAX_STREAM)
            .ok_or_else(|| invalid("native session request stream exceeds bound"))?;
        writer.log.write_all(&bytes)?;
        writer.log.flush()?;
        let stdin = writer
            .stdin
            .as_mut()
            .ok_or_else(|| invalid("native session input is closed"))?;
        stdin.write_all(&bytes)?;
        stdin.flush()?;
        drop(writer);
        Ok(())
    }

    fn watchdog(&self) {
        while !self.done.load(Ordering::Acquire) {
            let expired_operation = lock(&self.pending)
                .values()
                .any(|request| request.started.elapsed() >= request.timeout);
            let expired_shutdown =
                lock(&self.closing_at).is_some_and(|started| started.elapsed() >= SHUTDOWN_TIMEOUT);
            if self.started.elapsed() >= self.session_timeout
                || expired_operation
                || expired_shutdown
            {
                self.fail("native session operation, shutdown or lifetime timeout");
                return;
            }
            thread::sleep(Duration::from_millis(25));
        }
    }
}

#[derive(Clone)]
pub(super) struct Client {
    state: Arc<State>,
}

impl Client {
    pub(super) fn call(
        &self,
        worker: u32,
        operation: Value,
        progress: Option<Callback>,
    ) -> Result<Reply> {
        self.call_with_handlers(worker, operation, progress, None)
    }

    pub(super) fn call_with_handlers(
        &self,
        worker: u32,
        operation: Value,
        progress: Option<Callback>,
        tensor_data: Option<Arc<TensorData>>,
    ) -> Result<Reply> {
        self.call_with_event_handlers(worker, operation, progress, tensor_data, None)
    }

    pub(super) fn call_with_event_handlers(
        &self,
        worker: u32,
        operation: Value,
        progress: Option<Callback>,
        tensor_data: Option<Arc<TensorData>>,
        sampler_event: Option<Arc<SamplerEvent>>,
    ) -> Result<Reply> {
        self.state.check()?;
        if worker >= 16 || lock(&self.state.closing_at).is_some() {
            return Err(invalid(
                "native session worker is outside range or session is closing",
            ));
        }
        let id = self.state.next_id.fetch_add(1, Ordering::Relaxed);
        if id > MAX_REQUESTS {
            self.state
                .fail("native session request count exceeds bound");
            return Err(invalid("native session request count exceeds bound"));
        }
        let (sender, receiver) = mpsc::sync_channel(1);
        let timeout = operation_timeout(self.state.quantization, &operation);
        let mut pending = lock(&self.state.pending);
        if lock(&self.state.closing_at).is_some() {
            return Err(invalid("native session is closing"));
        }
        pending.insert(
            id,
            Pending {
                sender,
                progress,
                tensor_data,
                sampler_event,
                callback_error: None,
                observations: Vec::new(),
                started: Instant::now(),
                timeout,
            },
        );
        drop(pending);
        let mut request = json!({"Id":id,"Worker":worker,"Operation":null});
        request["Operation"] = operation;
        if let Err(error) = self.state.write(&request) {
            self.state
                .fail(format!("native session request failed: {error}"));
            return Err(error);
        }
        match receiver.recv_timeout(timeout) {
            Ok(Ok(reply)) => Ok(reply),
            Ok(Err(error)) => Err(invalid(error)),
            Err(error) => {
                self.state.fail(format!(
                    "native session response timeout/disconnection: {error}"
                ));
                Err(invalid("native session response timeout/disconnection"))
            }
        }
    }
}

fn exact_keys(value: &Value, names: &[&str]) -> Result<()> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid("native session frame must be an object"))?;
    if object.len() != names.len() || names.iter().any(|name| !object.contains_key(*name)) {
        return Err(invalid(
            "native session frame has unknown or missing fields",
        ));
    }
    Ok(())
}

fn unsigned(value: &Value, name: &str) -> Result<u64> {
    value
        .get(name)
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid(format!("native session {name} must be unsigned")))
}

fn callback(state: &State, value: &Value, seen: &mut HashSet<u64>) -> Result<()> {
    let id = unsigned(value, "CallbackId")?;
    if seen.len() >= MAX_CALLBACKS || !seen.insert(id) {
        return Err(invalid("native session callback identity/count mismatch"));
    }
    let request_id = unsigned(value, "RequestId")?;
    if value.get("TensorSpec").is_some() {
        return object_callback(state, value, id, request_id, "TensorSpec");
    }
    if value.get("SamplerEvent").is_some() {
        return object_callback(state, value, id, request_id, "SamplerEvent");
    }
    exact_keys(value, &["CallbackId", "RequestId", "ProgressBits"])?;
    let bits = u32::try_from(unsigned(value, "ProgressBits")?)?;
    let function = lock(&state.pending)
        .get(&request_id)
        .and_then(|request| request.progress)
        .ok_or_else(|| invalid("native session callback has no pending Rust handler"))?;
    let decision = std::panic::catch_unwind(|| function(f32::from_bits(bits)))
        .map_err(|_| invalid("Rust model progress callback panicked"))?;
    lock(&state.pending)
        .get_mut(&request_id)
        .ok_or_else(|| invalid("native callback request disappeared"))?
        .observations
        .push((bits, decision));
    state.write(&json!({"CallbackId":id,"Return":decision}))
}

fn object_callback(
    state: &State,
    value: &Value,
    id: u64,
    request_id: u64,
    field: &str,
) -> Result<()> {
    exact_keys(value, &["CallbackId", "RequestId", field])?;
    let function = lock(&state.pending)
        .get(&request_id)
        .and_then(|request| match field {
            "TensorSpec" => request.tensor_data.clone(),
            "SamplerEvent" => request.sampler_event.clone(),
            _ => None,
        })
        .ok_or_else(|| {
            invalid(format!(
                "native {field} callback has no pending Rust handler"
            ))
        })?;
    let observation =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| function(&value[field])))
            .map_err(|_| invalid(format!("Rust {field} callback panicked")))?;
    match observation {
        Ok(result) => {
            let mut reply = json!({"CallbackId":id,"Result":null});
            reply["Result"] = result;
            state.write(&reply)
        }
        Err(error) => {
            let error: String = error.to_string().chars().take(4096).collect();
            lock(&state.pending)
                .get_mut(&request_id)
                .ok_or_else(|| invalid("native callback request disappeared"))?
                .callback_error = Some(error.clone());
            // Let production cancel and release partial model state. The final
            // native response is still correlated, then rejected with this cause.
            state.write(&json!({"CallbackId":id,"Error":error}))
        }
    }
}

fn dispatch(state: &State, value: &Value, callbacks: &mut HashSet<u64>) -> Result<()> {
    if value.get("CallbackId").is_some() {
        return callback(state, value, callbacks);
    }
    if value.get("Shutdown") == Some(&Value::Bool(true)) {
        exact_keys(value, &["Shutdown"])?;
        if lock(&state.closing_at).is_none() || !lock(&state.pending).is_empty() {
            return Err(invalid("unexpected native shutdown acknowledgment"));
        }
        return Ok(());
    }
    let id = unsigned(value, "Id")?;
    let response = if value.get("Result").is_some() {
        exact_keys(value, &["Id", "Result"])?;
        Ok(value["Result"].clone())
    } else {
        exact_keys(value, &["Id", "Error"])?;
        Err(value["Error"]
            .as_str()
            .ok_or_else(|| invalid("native session error must be text"))?
            .to_owned())
    };
    let request = lock(&state.pending).remove(&id).ok_or_else(|| {
        invalid("native session returned an unknown or duplicate request identity")
    })?;
    let response = request.callback_error.map_or(response, Err);
    request
        .sender
        .send(response.map(|value| Reply {
            value,
            progress: request.observations,
        }))
        .map_err(|_| invalid("native session response receiver disappeared"))?;
    Ok(())
}

fn read_stdout(state: &State, output: impl Read, mut log: File) -> Result<()> {
    let mut reader = BufReader::new(output);
    let mut total = 0_u64;
    let mut callbacks = HashSet::new();
    loop {
        let mut frame = Vec::new();
        let count = reader
            .by_ref()
            .take(MAX_FRAME + 1)
            .read_until(b'\n', &mut frame)?;
        if count == 0 {
            if lock(&state.closing_at).is_some() && lock(&state.pending).is_empty() {
                return Ok(());
            }
            return Err(invalid("native session ended before graceful shutdown"));
        }
        total = total
            .checked_add(u64::try_from(count)?)
            .filter(|bytes| *bytes <= MAX_STREAM)
            .ok_or_else(|| invalid("native session stdout exceeds stream bound"))?;
        if u64::try_from(count)? > MAX_FRAME || frame.last() != Some(&b'\n') {
            return Err(invalid(
                "native session stdout frame exceeds bound or is truncated",
            ));
        }
        log.write_all(&frame)?;
        log.flush()?;
        dispatch(state, &serde_json::from_slice(&frame)?, &mut callbacks)?;
    }
}

fn read_stderr(state: &State, mut stderr: impl Read, mut log: File) -> Result<()> {
    let mut total = 0_u64;
    let mut bytes = [0; 16384];
    loop {
        let count = stderr.read(&mut bytes)?;
        if count == 0 {
            return Ok(());
        }
        total = total
            .checked_add(u64::try_from(count)?)
            .filter(|bytes| *bytes <= MAX_STDERR)
            .ok_or_else(|| invalid("native session stderr exceeds bound"))?;
        log.write_all(&bytes[..count])?;
        log.flush()?;
        state.check()?;
    }
}

pub(super) struct ModelSession {
    state: Arc<State>,
    readers: Vec<JoinHandle<()>>,
    directory: PathBuf,
    finished: bool,
}

impl ModelSession {
    pub(super) fn evidence_directory(&self) -> &Path {
        &self.directory
    }
    pub(super) fn open(engine: &Engine, name: &str) -> Result<Option<Self>> {
        if name == "quantize-stats" {
            return Self::open_quantization(engine);
        }
        Self::open_with_environment(engine, name, &[])
    }

    /// Only the complete statistics source on main CI has a longer lifetime.
    /// Its caller-supplied chunk continuations retain the loaded model; other
    /// operations, ordinary sessions and every transport bound stay unchanged.
    pub(super) fn open_quantization(engine: &Engine) -> Result<Option<Self>> {
        let quantization = std::env::var("GITHUB_ACTIONS").as_deref() == Ok("true")
            && std::env::var("GITHUB_REF").as_deref() == Ok("refs/heads/main");
        Self::open_scoped(engine, "quantize-stats", &[], quantization)
    }

    pub(super) fn open_with_environment(
        engine: &Engine,
        name: &str,
        environment: &[(String, String)],
    ) -> Result<Option<Self>> {
        Self::open_scoped(engine, name, environment, false)
    }

    fn open_scoped(
        engine: &Engine,
        name: &str,
        environment: &[(String, String)],
        quantization: bool,
    ) -> Result<Option<Self>> {
        let Some(spec) = engine.model_session_spec(name)? else {
            return Ok(None);
        };
        validate_environment(environment)?;
        let request_log = File::create(spec.evidence_directory.join("requests.jsonl"))?;
        let response_log = File::create(spec.evidence_directory.join("responses.jsonl"))?;
        let error_log = File::create(spec.evidence_directory.join("stderr.log"))?;
        let mut child = Command::new(&spec.executable)
            .arg("--model-session")
            .envs(environment.iter().map(|(key, value)| (key, value)))
            .current_dir(&spec.evidence_directory)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        // std::process guarantees these handles for Stdio::piped. Keep explicit
        // cleanup for an unexpected platform failure instead of leaking a child.
        let (Some(stdin), Some(stdout), Some(stderr)) =
            (child.stdin.take(), child.stdout.take(), child.stderr.take())
        else {
            let _ = child.kill();
            let _ = child.wait();
            return Err(invalid("native session pipe creation failed"));
        };
        let state = Arc::new(State {
            child: Mutex::new(child),
            writer: Mutex::new(Writer {
                stdin: Some(stdin),
                log: request_log,
                bytes: 0,
            }),
            pending: Mutex::new(HashMap::new()),
            fatal: Mutex::new(None),
            next_id: AtomicU64::new(1),
            closing_at: Mutex::new(None),
            done: AtomicBool::new(false),
            started: Instant::now(),
            session_timeout: session_timeout(
                name,
                quantization,
                std::env::var("GITHUB_ACTIONS").as_deref() == Ok("true")
                    && std::env::var("GITHUB_REF").as_deref() == Ok("refs/heads/main"),
            ),
            quantization,
        });
        let mut readers = Vec::new();
        let output_state = Arc::clone(&state);
        readers.push(thread::spawn(move || {
            if let Err(error) = read_stdout(&output_state, stdout, response_log) {
                output_state.fail(format!("native session stdout: {error}"));
            }
        }));
        let error_state = Arc::clone(&state);
        readers.push(thread::spawn(move || {
            if let Err(error) = read_stderr(&error_state, stderr, error_log) {
                error_state.fail(format!("native session stderr: {error}"));
            }
        }));
        let watchdog_state = Arc::clone(&state);
        readers.push(thread::spawn(move || watchdog_state.watchdog()));
        Ok(Some(Self {
            state,
            readers,
            directory: spec.evidence_directory,
            finished: false,
        }))
    }

    pub(super) fn client(&self) -> Client {
        Client {
            state: Arc::clone(&self.state),
        }
    }

    fn wait_for_exit(&self) -> Result<ExitStatus> {
        let started = Instant::now();
        loop {
            let status = lock(&self.state.child).try_wait()?;
            if let Some(status) = status {
                return Ok(status);
            }
            if started.elapsed() >= SHUTDOWN_TIMEOUT {
                self.state
                    .fail("native session did not exit after shutdown");
                return Ok(lock(&self.state.child).wait()?);
            }
            thread::sleep(Duration::from_millis(10));
        }
    }

    fn join(&mut self) -> Result<()> {
        self.state.done.store(true, Ordering::Release);
        lock(&self.state.writer).stdin.take();
        for reader in self.readers.drain(..) {
            reader
                .join()
                .map_err(|_| invalid("native session transport thread panicked"))?;
        }
        Ok(())
    }

    pub(super) fn finish(mut self) -> Result<()> {
        self.state.check()?;
        let pending = lock(&self.state.pending);
        if !pending.is_empty() {
            return Err(invalid(
                "native session cannot finish with pending requests",
            ));
        }
        *lock(&self.state.closing_at) = Some(Instant::now());
        drop(pending);
        if let Err(error) = self.state.write(&json!({"Shutdown":true})) {
            self.state
                .fail(format!("native shutdown write failed: {error}"));
        }
        let status = self.wait_for_exit()?;
        self.join()?;
        self.finished = true;
        let fatal = lock(&self.state.fatal).clone();
        fs::write(
            self.evidence_directory().join("session.json"),
            serde_json::to_vec_pretty(&json!({
            "SchemaVersion":1,"ExitCode":status.code(),"Requests":self.state.next_id.load(Ordering::Relaxed) - 1,
                "ElapsedMilliseconds":self.state.started.elapsed().as_millis(),"Error":fatal,
                "QuantizationScope":self.state.quantization,"SessionTimeoutSeconds":self.state.session_timeout.as_secs(),
            }))?,
        )?;
        if let Some(error) = fatal {
            return Err(invalid(error));
        }
        if !status.success() {
            return Err(invalid(format!("native session exited with {status}")));
        }
        Ok(())
    }
}

fn validate_environment(environment: &[(String, String)]) -> Result<()> {
    let mut names = HashSet::new();
    for (name, value) in environment {
        if !names.insert(name) {
            return Err(invalid("duplicate native test environment variable"));
        }
        match name.as_str() {
            "LLAMA_CACHE" => {
                let path = Path::new(value);
                if !path.is_absolute() || !path.is_dir() || value.contains('%') {
                    return Err(invalid(
                        "LLAMA_CACHE requires an existing expanded absolute evidence directory",
                    ));
                }
            }
            "MODEL_ENDPOINT" => {
                let port = value
                    .trim_end_matches('/')
                    .strip_prefix("http://127.0.0.1:")
                    .ok_or_else(|| {
                        invalid("MODEL_ENDPOINT must be the explicit local IPv4 fixture server")
                    })?
                    .parse::<u16>()?;
                if port == 0 {
                    return Err(invalid("MODEL_ENDPOINT requires a bound nonzero port"));
                }
            }
            _ => {
                return Err(invalid(
                    "native test environment variable is not allowlisted",
                ));
            }
        }
    }
    Ok(())
}

impl Drop for ModelSession {
    fn drop(&mut self) {
        if !self.finished {
            self.state
                .fail("native session dropped before successful finish");
            let _ = lock(&self.state.child).wait();
            let _ = self.join();
        }
    }
}

#[cfg(test)]
mod quantization_deadline_tests {
    use super::*;

    #[test]
    fn only_scoped_codec_trace_receives_long_deadline() -> Result<()> {
        for (scope, op, expected) in [
            (false, "qb-trace", OPERATION_TIMEOUT),
            (true, "qb-trace", QUANTIZATION_OPERATION_TIMEOUT),
            (true, "qb-call", OPERATION_TIMEOUT),
            (true, "model-load", OPERATION_TIMEOUT),
            (false, "decode", OPERATION_TIMEOUT),
        ] {
            if operation_timeout(scope, &json!({"Op":op})) != expected {
                return Err(invalid("quantization deadline escaped its exact scope"));
            }
        }
        Ok(())
    }

    #[test]
    fn complete_source_lifetimes_are_scoped_to_main_ci() -> Result<()> {
        for (name, quantization, main_ci, expected) in [
            ("dspark-real-eval", false, true, REAL_EVAL_SESSION_TIMEOUT),
            ("dspark-real-eval", false, false, SESSION_TIMEOUT),
            ("dspark-forward", false, true, SESSION_TIMEOUT),
            (
                "tests/test-save-load-state.cpp",
                false,
                true,
                SAVE_LOAD_SESSION_TIMEOUT,
            ),
            (
                "tests/test-thread-safety.cpp",
                false,
                true,
                THREAD_SAFETY_SESSION_TIMEOUT,
            ),
            (
                "tests/test-save-load-state.cpp",
                false,
                false,
                SESSION_TIMEOUT,
            ),
            (
                "tests/test-thread-safety.cpp",
                false,
                false,
                SESSION_TIMEOUT,
            ),
            (
                "tests/test-save-load-state.cpp",
                true,
                true,
                SESSION_TIMEOUT,
            ),
            ("tests/test-thread-safety.cpp", true, true, SESSION_TIMEOUT),
            ("test-save-load-state", false, true, SESSION_TIMEOUT),
            (
                "tests/test-save-load-state.cpp-extra",
                false,
                true,
                SESSION_TIMEOUT,
            ),
            (
                "tests/test-state-restore-fragmented.cpp",
                false,
                true,
                SESSION_TIMEOUT,
            ),
            ("ordinary", false, true, SESSION_TIMEOUT),
            ("quantize-stats", true, true, QUANTIZATION_SESSION_TIMEOUT),
            ("quantize-stats", true, false, SESSION_TIMEOUT),
        ] {
            if session_timeout(name, quantization, main_ci) != expected {
                return Err(invalid("source lifetime escaped its exact CI scope"));
            }
        }
        Ok(())
    }
}
