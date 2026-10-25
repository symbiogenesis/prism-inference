// SPDX-License-Identifier: MIT
//! A bounded pipe transport reuses a process, never a geometry's native state.

use super::super::super::engine::Engine;
use super::super::super::{Result, invalid};
use super::{MAX_ACTION_FILE, MAX_ACTIONS, MAX_FILE_BYTES, MAX_OBSERVATION, Session};
use serde::Serialize;
use serde_json::{Value, json};
use std::cell::RefCell;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

#[derive(Serialize)]
#[serde(rename_all = "PascalCase")]
struct Request<'a, T> {
    operation: &'a T,
    sequence: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "PascalCase")]
struct Actions<'a> {
    actions: &'a [Value],
    op: &'static str,
}

fn lock<T>(value: &Mutex<T>) -> MutexGuard<'_, T> {
    value
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn read_frame(input: &mut impl Read) -> Result<Value> {
    let mut header = [0; 4];
    input.read_exact(&mut header)?;
    let length = u32::from_le_bytes(header) as usize;
    if length == 0 || length as u64 > MAX_OBSERVATION {
        return Err(invalid("backend pipe response exceeds its frame bound"));
    }
    let mut bytes = vec![0; length];
    input.read_exact(&mut bytes)?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn response(mut value: Value, sequence: u64) -> Result<Value> {
    if value.as_object().is_none_or(|object| object.len() != 2)
        || value["Sequence"] != sequence
        || value.get("Result").is_none()
    {
        return Err(invalid("backend pipe response identity/framing differs"));
    }
    value
        .as_object_mut()
        .and_then(|object| object.remove("Result"))
        .ok_or_else(|| invalid("backend pipe missing result"))
}

struct Monitor {
    child: Mutex<Child>,
    deadline: Mutex<Option<Instant>>,
    done: AtomicBool,
    fatal: Mutex<Option<String>>,
}

impl Monitor {
    fn fail(&self, message: impl Into<String>) {
        let mut fatal = lock(&self.fatal);
        if fatal.is_none() {
            *fatal = Some(message.into());
        }
        drop(fatal);
        let _ = lock(&self.child).kill();
    }
}

struct Worker {
    monitor: Arc<Monitor>,
    input: Option<ChildStdin>,
    output: Option<mpsc::Receiver<std::result::Result<Value, String>>>,
    readers: Vec<JoinHandle<()>>,
    sequence: u64,
    poisoned: bool,
    finished: bool,
    directory: PathBuf,
    case_deadline: Option<Instant>,
}

impl Worker {
    fn open(engine: &Engine) -> Result<Option<Self>> {
        let Some(spec) = engine.model_session_spec("backend-pipe")? else {
            return Ok(None);
        };
        let errors = File::create(spec.evidence_directory.join("stderr.log"))?;
        let mut child = Command::new(spec.executable)
            .arg("--backend-session")
            .current_dir(&spec.evidence_directory)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let (Some(input), Some(output), Some(mut stderr)) =
            (child.stdin.take(), child.stdout.take(), child.stderr.take())
        else {
            let _ = child.kill();
            let _ = child.wait();
            return Err(invalid("backend pipe creation failed"));
        };
        let monitor = Arc::new(Monitor {
            child: Mutex::new(child),
            deadline: Mutex::new(None),
            done: AtomicBool::new(false),
            fatal: Mutex::new(None),
        });
        let (sender, receiver) = mpsc::sync_channel(1);
        let mut readers = Vec::new();
        readers.push(thread::spawn(move || {
            let mut output = BufReader::new(output);
            loop {
                let result = read_frame(&mut output).map_err(|error| error.to_string());
                let failed = result.is_err();
                if sender.send(result).is_err() || failed {
                    break;
                }
            }
        }));
        let error_monitor = Arc::clone(&monitor);
        readers.push(thread::spawn(move || {
            let mut errors = BufWriter::new(errors);
            let mut bytes = [0; 16_384];
            let mut total = 0_usize;
            let result: std::io::Result<()> = (|| {
                loop {
                    let count = stderr.read(&mut bytes)?;
                    if count == 0 {
                        return errors.flush();
                    }
                    total += count;
                    // One partition now shares a process; retain a bounded log
                    // instead of allocating a separate 1MiB log per geometry.
                    if total > 64 * 1024 * 1024 {
                        return Err(std::io::Error::other("backend stderr exceeds64MiB"));
                    }
                    errors.write_all(&bytes[..count])?;
                }
            })();
            if let Err(error) = result {
                error_monitor.fail(format!("backend stderr capture failed: {error}"));
            }
        }));
        let watchdog = Arc::clone(&monitor);
        readers.push(thread::spawn(move || {
            while !watchdog.done.load(Ordering::Acquire) {
                if lock(&watchdog.deadline).is_some_and(|deadline| Instant::now() >= deadline) {
                    watchdog.fail("backend pipe operation/case deadline expired");
                    return;
                }
                thread::sleep(Duration::from_millis(25));
            }
        }));
        Ok(Some(Self {
            monitor,
            input: Some(input),
            output: Some(receiver),
            readers,
            sequence: 0,
            poisoned: false,
            finished: false,
            directory: spec.evidence_directory,
            case_deadline: None,
        }))
    }

    fn call(&mut self, operation: &impl Serialize, deadline: Instant) -> Result<Value> {
        if self.poisoned {
            return Err(invalid("backend pipe is poisoned by an earlier failure"));
        }
        *lock(&self.monitor.deadline) = Some(deadline);
        let result = (|| {
            if deadline <= Instant::now() {
                return Err(invalid("backend pipe deadline already expired"));
            }
            if let Some(error) = &*lock(&self.monitor.fatal) {
                return Err(invalid(error.clone()));
            }
            let sequence = self.sequence + 1;
            let request = Request {
                operation,
                sequence,
            };
            let bytes = serde_json::to_vec(&request)?;
            if bytes.is_empty() || bytes.len() > MAX_ACTION_FILE {
                return Err(invalid("backend pipe request exceeds128MiB"));
            }
            self.sequence = sequence;
            let input = self
                .input
                .as_mut()
                .ok_or_else(|| invalid("backend pipe input is closed"))?;
            input.write_all(&u32::try_from(bytes.len())?.to_le_bytes())?;
            input.write_all(&bytes)?;
            input.flush()?;
            let value = self
                .output
                .as_ref()
                .ok_or_else(|| invalid("backend pipe output is closed"))?
                .recv_timeout(deadline.saturating_duration_since(Instant::now()))
                .map_err(|_| invalid("backend pipe response timeout/disconnection"))?
                .map_err(invalid)?;
            response(value, self.sequence)
        })();
        *lock(&self.monitor.deadline) = self.case_deadline;
        if result.is_err() {
            self.poison("backend pipe request/response failed");
        }
        result
    }

    fn poison(&mut self, message: &str) {
        self.poisoned = true;
        self.monitor.fail(message);
        // Complete native termination before the driver hashes or prunes the
        // failed geometry's files. Killing alone can return before Windows has
        // closed every native file handle and completed outstanding writes.
        let _ = lock(&self.monitor.child).wait();
    }

    fn finish(&mut self) -> Result<()> {
        self.call(
            &json!({"Op":"shutdown"}),
            Instant::now() + Duration::from_secs(10),
        )?;
        self.input.take();
        let started = Instant::now();
        let status = loop {
            let exited = lock(&self.monitor.child).try_wait()?;
            if let Some(status) = exited {
                break status;
            }
            if started.elapsed() > Duration::from_secs(10) {
                return Err(invalid("backend pipe did not exit after shutdown"));
            }
            thread::sleep(Duration::from_millis(10));
        };
        self.monitor.done.store(true, Ordering::Release);
        self.output.take();
        for reader in self.readers.drain(..) {
            reader
                .join()
                .map_err(|_| invalid("backend pipe reader panicked"))?;
        }
        if let Some(error) = &*lock(&self.monitor.fatal) {
            return Err(invalid(error.clone()));
        }
        self.finished = true;
        fs::write(
            self.directory.join("session.json"),
            serde_json::to_vec_pretty(&json!({
                "SchemaVersion":1,"Requests":self.sequence,"ExitCode":status.code(),"FreshStatePerGeometry":true
            }))?,
        )?;
        if !status.success() {
            return Err(invalid("backend pipe process exited unsuccessfully"));
        }
        Ok(())
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        if !self.finished {
            self.monitor.done.store(true, Ordering::Release);
            let _ = lock(&self.monitor.child).kill();
            self.input.take();
            self.output.take();
            let status = lock(&self.monitor.child).wait();
            // A failed native process cannot send a second queued frame.
            for reader in self.readers.drain(..) {
                let _ = reader.join();
            }
            let _ = fs::write(self.directory.join("session.json"), serde_json::to_vec_pretty(&json!({
                "SchemaVersion":1,"Requests":self.sequence,"ExitCode":status.ok().and_then(|status|status.code()),
                "FreshStatePerGeometry":true,"Poisoned":self.poisoned,"Error":*lock(&self.monitor.fatal)
            })).unwrap_or_default());
        }
    }
}

pub(in super::super) struct Case {
    worker: Rc<RefCell<Worker>>,
    pub(super) directory: PathBuf,
    deadline: Instant,
    total_actions: usize,
    requests: BufWriter<File>,
    responses: BufWriter<File>,
}

impl Case {
    pub(super) fn exchange(&mut self, actions: &[Value]) -> Result<Vec<Value>> {
        let result = self.exchange_checked(actions);
        if result.is_err() {
            self.worker
                .borrow_mut()
                .poison("backend action/observation exchange failed");
        }
        result
    }

    fn exchange_checked(&mut self, actions: &[Value]) -> Result<Vec<Value>> {
        self.total_actions = self
            .total_actions
            .checked_add(actions.len() + 1)
            .filter(|total| *total <= MAX_ACTIONS)
            .ok_or_else(|| invalid("backend explicit action extent exceeds its bound"))?;
        if actions.is_empty() {
            return Err(invalid("backend empty action batch"));
        }
        serde_json::to_writer(&mut self.requests, &actions)?;
        self.requests.write_all(b"\n")?;
        self.requests.flush()?;
        let value = self.worker.borrow_mut().call(
            &Actions {
                actions,
                op: "actions",
            },
            self.deadline,
        )?;
        serde_json::to_writer(&mut self.responses, &value)?;
        self.responses.write_all(b"\n")?;
        self.responses.flush()?;
        if value.as_object().is_none_or(|object| object.len() != 1) {
            return Err(invalid("backend pipe observation fields differ"));
        }
        let Value::Object(mut object) = value else {
            return Err(invalid("backend pipe observation must be an object"));
        };
        match object.remove("Trace") {
            Some(Value::Array(trace)) if trace.len() == actions.len() => Ok(trace),
            _ => Err(invalid("backend action/observation count differs")),
        }
    }
}

pub(in super::super) struct Partition {
    worker: Option<Rc<RefCell<Worker>>>,
}

impl Partition {
    pub(in super::super) fn open(engine: &Engine) -> Result<Self> {
        Ok(Self {
            worker: Worker::open(engine)?.map(|worker| Rc::new(RefCell::new(worker))),
        })
    }

    pub(in super::super) fn with_session<T>(
        &self,
        directory: &Path,
        controller: impl FnOnce(&mut Session) -> Result<T>,
    ) -> Result<Option<T>> {
        let directory = fs::canonicalize(directory)?;
        if !directory.is_absolute() || !directory.is_dir() {
            return Err(invalid(
                "backend session directory must exist and be absolute",
            ));
        }
        let timeout = if std::env::var("GITHUB_ACTIONS").as_deref() == Ok("true")
            && std::env::var("GITHUB_REF").as_deref() == Ok("refs/heads/main")
        {
            Duration::from_secs(3600)
        } else {
            Duration::from_secs(60)
        };
        let deadline = Instant::now() + timeout;
        let Some(worker) = &self.worker else {
            // model_session_spec returns None for the Rust target, whose
            // production backend lives in this process. A missing pipe is
            // therefore not an unsupported production graph API.
            let mut session = Session::Rust(super::super::rust::GraphSession::open(
                &directory,
                usize::try_from(MAX_FILE_BYTES)?,
                deadline,
            )?);
            return controller(&mut session).map(Some);
        };
        worker.borrow_mut().case_deadline = Some(deadline);
        worker.borrow_mut().call(
            &json!({"Op":"open","Directory":directory,"MaxFileBytes":MAX_FILE_BYTES}),
            deadline,
        )?;
        let mut session = Session::Pipe(Case {
            worker: Rc::clone(worker),
            requests: BufWriter::new(File::create(directory.join("actions.jsonl"))?),
            responses: BufWriter::new(File::create(directory.join("observations.jsonl"))?),
            directory,
            deadline,
            // Retain the file protocol's initial and per-exchange checkpoint
            // slots in the cumulative bound, although no files are exchanged.
            total_actions: 1,
        });
        let result = controller(&mut session);
        let completion = worker.borrow_mut().call(&json!({"Op":"close"}), deadline);
        {
            let mut worker = worker.borrow_mut();
            worker.case_deadline = None;
            *lock(&worker.monitor.deadline) = None;
        }
        let result = result?;
        if completion? != json!({}) {
            worker
                .borrow_mut()
                .poison("backend pipe close acknowledgement differs");
            return Err(invalid("backend pipe close acknowledgement differs"));
        }
        Ok(Some(result))
    }

    pub(in super::super) fn healthy(&self) -> bool {
        self.worker.as_ref().is_none_or(|worker| {
            let worker = worker.borrow();
            !worker.poisoned && lock(&worker.monitor.fatal).is_none()
        })
    }

    pub(in super::super) fn finish(&self) -> Result<()> {
        if let Some(worker) = &self.worker {
            worker.borrow_mut().finish()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::{
        evaluate,
        graph::Plan,
        host::Production,
        planner,
        prepare::{Mode, prepare},
    };
    use super::*;

    fn native_rejects_frame(bridge: &Path, directory: &Path, frame: &[u8]) -> Result<()> {
        let mut child = Command::new(bridge)
            .arg("--backend-session")
            .current_dir(directory)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(File::create(directory.join("malformed-frame.log"))?)
            .spawn()?;
        if let Some(mut input) = child.stdin.take() {
            let _ = input.write_all(frame);
        }
        let started = Instant::now();
        loop {
            if let Some(status) = child.try_wait()? {
                assert!(
                    !status.success(),
                    "malformed/EOF backend transport input must exit unsuccessfully"
                );
                return Ok(());
            }
            if started.elapsed() > Duration::from_secs(10) {
                let _ = child.kill();
                let _ = child.wait();
                return Err(invalid(
                    "malformed backend frame did not terminate the native process",
                ));
            }
            thread::sleep(Duration::from_millis(10));
        }
    }

    fn numerical_fixture(session: &mut Session) -> Result<(Vec<Value>, Vec<u8>)> {
        let left = session.file("left.bin")?;
        let right = session.file("right.bin")?;
        let output = session.file("output.bin")?;
        let values = (0..32)
            .map(|index| (index as f32 - 16.0) * 0.25)
            .collect::<Vec<_>>();
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        fs::write(&left, &bytes)?;
        fs::write(&right, &bytes)?;
        let mut observations = session.exchange(vec![
            json!({"Op":"context-init","Id":0,"MemoryBytes":1_048_576,"NoAlloc":true}),
            json!({"Op":"backend-init","Id":0,"Device":"CPU","Threads":1,"UseRef":false}),
            json!({"Op":"call","Function":"ggml_new_tensor_1d","Arguments":[{"Context":0},{"I32":0},{"I64":32}],"Result":{"Tensor":0}}),
            json!({"Op":"call","Function":"ggml_new_tensor_1d","Arguments":[{"Context":0},{"I32":0},{"I64":32}],"Result":{"Tensor":1}}),
            json!({"Op":"call","Function":"ggml_set_name","Arguments":[{"Tensor":0},{"String":"left"}],"Result":{"Tensor":9}}),
            json!({"Op":"call","Function":"ggml_set_name","Arguments":[{"Tensor":9},{"String":"left"}],"Result":{"Tensor":3}}),
            json!({"Op":"call","Function":"ggml_add","Arguments":[{"Context":0},{"Tensor":0},{"Tensor":1}],"Result":{"Tensor":2}}),
            json!({"Op":"call","Function":"ggml_new_graph","Arguments":[{"Context":0}],"Result":{"Graph":0}}),
            json!({"Op":"call","Function":"ggml_build_forward_expand","Arguments":[{"Graph":0},{"Tensor":2}],"Result":null}),
            json!({"Op":"allocate","Context":0,"Backend":0,"Result":{"Buffer":0}}),
            json!({"Op":"tensor-snapshot","Tensor":9}),
            json!({"Op":"tensor-snapshot","Tensor":2}),
            json!({"Op":"context-snapshot","Context":0}),
            json!({"Op":"graph-snapshot","Graph":0}),
        ])?;
        let tensors = observations
            .iter()
            .find_map(|value| value.get("Tensors"))
            .ok_or_else(|| invalid("focused tensor snapshot missing"))?;
        assert_eq!(tensors[0]["Aliases"], json!([0, 3, 9]));
        assert_eq!(tensors[0]["Id"], 0);
        assert_eq!(tensors[0]["Name"], "left");
        let single = observations
            .iter()
            .filter_map(|value| value.get("Tensor"))
            .collect::<Vec<_>>();
        assert_eq!(single, vec![&tensors[0], &tensors[2]]);
        observations.extend(session.exchange(vec![
            json!({"Op":"tensor-set","Tensor":0,"File":left,"FileOffset":0,"TensorOffset":0,"Bytes":bytes.len()}),
            json!({"Op":"tensor-set","Tensor":1,"File":right,"FileOffset":0,"TensorOffset":0,"Bytes":bytes.len()}),
            json!({"Op":"call","Function":"ggml_backend_graph_compute","Arguments":[{"Backend":0},{"Graph":0}],"Result":{"Scalar":0}}),
            json!({"Op":"scalar","Scalar":0,"Release":true}),
            json!({"Op":"tensor-get","Tensor":2,"File":output,"FileOffset":0,"TensorOffset":0,"Bytes":bytes.len()}),
        ])?);
        let output = fs::read(output)?;
        let expected = values
            .iter()
            .flat_map(|value| (value * 2.0).to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(output, expected);
        Ok((observations, output))
    }

    fn source_fixture(
        session: &mut Session,
        index: usize,
    ) -> Result<(Value, Value, Vec<Value>, Vec<u32>)> {
        let (program, geometries, _) = planner::extent(Mode::Eval)?;
        let geometry = &geometries[index];
        let mut host = Production::new(
            Plan::new(session),
            program,
            index as u64,
            1,
            Mode::Eval.code(),
        );
        let mut prepared = prepare(&mut host, geometry, Mode::Eval, None)?;
        let receipt = planner::receipt(&host, &prepared, geometry, index)?;
        assert!(
            prepared.initialize,
            "focused equivalence case must execute numerical graphs"
        );
        evaluate::allocate(&mut host, &mut prepared, geometry, Mode::Eval, 0)?;
        let result = evaluate::evaluate(&mut host, &mut prepared, geometry)?;
        assert_eq!(result["Status"], "pass");
        Ok((
            receipt,
            result,
            host.inputs.receipts,
            host.inputs.ledger.seeds,
        ))
    }

    fn native_failure_guards(engine: &Engine, bridge: &Path, root: &Path) -> Result<()> {
        let mut expired =
            Worker::open(engine)?.ok_or_else(|| invalid("native backend worker unavailable"))?;
        assert!(
            expired
                .call(&json!({"Op":"shutdown"}), Instant::now())
                .is_err()
        );
        assert!(expired.poisoned);
        drop(expired);
        for frame in [
            Vec::new(), // EOF, including a partial header/payload, is fatal.
            vec![1, 0],
            vec![0, 0, 0, 0],
            (u32::try_from(MAX_ACTION_FILE)? + 1).to_le_bytes().to_vec(),
            vec![5, 0, 0, 0, b'{'],
        ] {
            native_rejects_frame(bridge, root, &frame)?;
        }
        for invalid in [
            json!({"Sequence":2,"Operation":{"Op":"shutdown"}}),
            json!({"Sequence":1,"Operation":{"Op":"close"}}),
            json!({"Sequence":1,"Operation":{"Op":"actions","Actions":[]}}),
        ] {
            let payload = serde_json::to_vec(&invalid)?;
            let mut frame = u32::try_from(payload.len())?.to_le_bytes().to_vec();
            frame.extend(payload);
            native_rejects_frame(bridge, root, &frame)?;
        }
        Ok(())
    }

    #[test]
    fn rust_partition_matches_file_protocol_and_resets_production_state() -> Result<()> {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let base = std::env::temp_dir();
        if !base.is_absolute() {
            return Err(invalid("Rust backend test temp directory must be absolute"));
        }
        let root = base.join(format!(
            "prism-rust-partition-{}-{stamp}",
            std::process::id()
        ));
        fs::create_dir(&root)?;
        let engine = Engine::new(
            super::super::super::super::engine::Target::Rust,
            None,
            None,
            Some(root.clone()),
            None,
            None,
        )?;
        let old = engine.evidence_path("file-numerical")?;
        let expected = super::super::with_session(&engine, &old, numerical_fixture)?
            .ok_or_else(|| invalid("Rust file production API unavailable"))?;
        let partition = Partition::open(&engine)?;
        for _ in 0..3 {
            let directory = engine.evidence_path("direct-numerical")?;
            let actual = partition
                .with_session(&directory, numerical_fixture)?
                .ok_or_else(|| invalid("Rust partition must execute its production API"))?;
            assert_eq!(actual, expected);
        }
        let (_, geometries, _) = planner::extent(Mode::Eval)?;
        for class in ["test_bin_bcast", "test_get_rows"] {
            let index = geometries
                .iter()
                .position(|geometry| geometry.class == class)
                .ok_or_else(|| invalid("focused source geometry missing"))?;
            let old = engine.evidence_path("file-source")?;
            let expected = super::super::with_session(&engine, &old, |session| {
                source_fixture(session, index)
            })?
            .ok_or_else(|| invalid("Rust file source API unavailable"))?;
            let directory = engine.evidence_path("direct-source")?;
            let actual = partition
                .with_session(&directory, |session| source_fixture(session, index))?
                .ok_or_else(|| invalid("Rust partition source API unavailable"))?;
            assert_eq!(actual, expected);
            if class == "test_get_rows" {
                assert!(actual.0["CrtDraws"].as_u64().is_some_and(|draws| draws > 0));
            }
        }

        let directory = engine.evidence_path("poisoned-source")?;
        assert_eq!(
            partition.with_session(&directory, |session| {
                assert!(
                    session
                        .exchange(vec![json!({"Op":"invalid-operation"})])
                        .is_err()
                );
                assert!(
                    session
                        .exchange(vec![json!({"Op":"context-init","Id":0})])
                        .is_err()
                );
                Ok(true)
            })?,
            Some(true)
        );
        let directory = engine.evidence_path("after-poisoned-source")?;
        assert!(
            partition
                .with_session(&directory, numerical_fixture)?
                .is_some()
        );
        partition.finish()?;
        drop(partition);
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    #[ignore = "requires the checked native backend tools; explicitly required by the backend tools CI job"]
    fn native_file_and_pipe_observations_match_and_geometry_state_is_fresh() -> Result<()> {
        let bridge = PathBuf::from(std::env::var("BONSAI_BACKEND_TRANSPORT_BRIDGE")?);
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let base = std::env::var_os("BONSAI_BACKEND_TRANSPORT_EVIDENCE")
            .map_or_else(std::env::temp_dir, PathBuf::from);
        if !base.is_absolute() || !base.is_dir() {
            return Err(invalid(
                "backend transport evidence parent must be an existing absolute directory",
            ));
        }
        let root = base.join(format!(
            "prism-pipe-equivalence-{}-{stamp}",
            std::process::id()
        ));
        fs::create_dir(&root)?;
        let engine = Engine::new(
            super::super::super::super::engine::Target::Cpp,
            None,
            Some(bridge.clone()),
            Some(root.clone()),
            None,
            None,
        )?;
        let old = engine.evidence_path("old-numerical")?;
        let expected = super::super::with_session(&engine, &old, numerical_fixture)?
            .ok_or_else(|| invalid("native file API unavailable"))?;
        let partition = Partition::open(&engine)?;
        for _ in 0..3 {
            let directory = engine.evidence_path("pipe-numerical")?;
            let actual = partition
                .with_session(&directory, numerical_fixture)?
                .ok_or_else(|| invalid("native pipe API unavailable"))?;
            assert_eq!(actual, expected);
            assert_eq!(
                planner::hash(&serde_json::to_vec(&actual)?),
                planner::hash(&serde_json::to_vec(&expected)?)
            );
        }
        let (_, geometries, _) = planner::extent(Mode::Eval)?;
        for class in ["test_bin_bcast", "test_get_rows", "test_cpy"] {
            let index = geometries
                .iter()
                .position(|geometry| geometry.class == class)
                .ok_or_else(|| invalid(format!("missing focused geometry {class}")))?;
            let old = engine.evidence_path("old-source")?;
            let expected = super::super::with_session(&engine, &old, |session| {
                source_fixture(session, index)
            })?
            .ok_or_else(|| invalid("native file source API unavailable"))?;
            let directory = engine.evidence_path("pipe-source")?;
            let actual = partition
                .with_session(&directory, |session| source_fixture(session, index))?
                .ok_or_else(|| invalid("native pipe source API unavailable"))?;
            assert_eq!(
                actual, expected,
                "full geometry observations differ for {class}"
            );
        }
        partition.finish()?;
        drop(partition);
        let poisoned = Partition::open(&engine)?;
        let directory = engine.evidence_path("bad-action")?;
        assert!(
            poisoned
                .with_session(&directory, |session| session
                    .exchange(vec![json!({"Op":"invalid-operation"})]))
                .is_err()
        );
        assert!(!poisoned.healthy());
        assert!(
            lock(
                &poisoned
                    .worker
                    .as_ref()
                    .ok_or_else(|| invalid("native worker missing"))?
                    .borrow()
                    .monitor
                    .child
            )
            .try_wait()?
            .is_some(),
            "failed native worker must be reaped before evidence retention"
        );
        let directory = engine.evidence_path("after-bad-action")?;
        assert!(
            poisoned
                .with_session(&directory, numerical_fixture)
                .is_err()
        );
        drop(poisoned);
        native_failure_guards(&engine, &bridge, &root)?;
        // Keep failed runs for inspection; remove only this uniquely owned tree
        // after every comparison and native process shutdown has succeeded.
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn bounded_frames_reject_truncation_oversize_and_wrong_identity() -> Result<()> {
        let actions = vec![
            json!({"Op":"call","Arguments":[{"Tensor":7},{"I64Array":[1,2,3]}],"Result":{"Tensor":8}}),
        ];
        let operation = Actions {
            actions: &actions,
            op: "actions",
        };
        let borrowed = Request {
            operation: &operation,
            sequence: 7,
        };
        let owned = json!({"Sequence":7,"Operation":{"Op":"actions","Actions":actions}});
        assert_eq!(serde_json::to_vec(&borrowed)?, serde_json::to_vec(&owned)?);
        let value = json!({"Sequence":7,"Result":{"Trace":[{}]}});
        let mut bytes = serde_json::to_vec(&value)?;
        let mut frame = u32::try_from(bytes.len())?.to_le_bytes().to_vec();
        frame.append(&mut bytes);
        assert_eq!(read_frame(&mut frame.as_slice())?, value);
        assert!(read_frame(&mut &frame[..frame.len() - 1]).is_err());
        assert!(read_frame(&mut [0_u8; 4].as_slice()).is_err());
        assert!(
            read_frame(
                &mut (u32::try_from(MAX_OBSERVATION)? + 1)
                    .to_le_bytes()
                    .as_slice()
            )
            .is_err()
        );
        assert!(response(value.clone(), 6).is_err());
        assert_eq!(response(value.clone(), 7)?, value["Result"]);
        assert!(response(json!({"Sequence":7,"Result":{},"Unexpected":true}), 7).is_err());
        Ok(())
    }
}
