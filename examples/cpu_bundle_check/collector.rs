use bonsai_native_hash::{Hasher, hash_bytes};
use bonsai_prism_runtime::{
    Result,
    kernels::{MatvecBackend, RowParallelism},
    model::{lora::SWIFT_ADAPTER_SHA256, swift::Representation},
    runtime::{Cancellation, Completion, Finish, GenerationPhase, GenerationTimings, PromptReuse},
    swift_session::{SwiftSession, TextEvent},
};
use serde_json::{Value, json};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    process::ExitCode,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const MAX_INPUT_BYTES: u64 = 64 * 1024;
const MAX_EXECUTABLE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_OUTPUT_BYTES: usize = 64 * 1024;
const MAX_PROMPT_TOKENS: usize = 128;
const RECEIPT_RESERVE: Duration = Duration::from_secs(2);
const LEGACY_F16_SOURCE: &str = "055c7430cb4771dfddb24ddaa30494af6ad182da";

struct Selection {
    model: PathBuf,
    adapter: PathBuf,
    evidence: PathBuf,
    messages: PathBuf,
    output_tokens: usize,
    source_commit: String,
    profile: Profile,
    deadline_unix_milliseconds: u64,
    deadline: Instant,
    generation_deadline: Instant,
}

#[derive(Default)]
struct CapturedOutput {
    text: String,
    terminal: Option<Completion>,
    first_text_delta_milliseconds: Option<f64>,
}

#[derive(Default)]
struct Observation {
    stage: &'static str,
    model_loaded: bool,
    input_sha256: Option<String>,
    executable_sha256: Option<String>,
    effective_row_workers: Option<usize>,
    observed_adapter_sha256: Option<String>,
    prompt_tokens: Option<usize>,
    load_milliseconds: Option<f64>,
    preparation_milliseconds: Option<f64>,
    request_milliseconds: Option<f64>,
    timings: GenerationTimings,
    completion: Option<Completion>,
    output: CapturedOutput,
}

fn reject(message: &str) -> bonsai_prism_runtime::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message).into()
}

fn check(deadline: Instant) -> Result<()> {
    if Instant::now() >= deadline {
        return Err(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            "the original CPU comparison arm deadline expired",
        )
        .into());
    }
    Ok(())
}

fn absolute_path(value: &str) -> Result<PathBuf> {
    let path = PathBuf::from(value);
    if !path.is_absolute()
        || value.contains(['%', '\0'])
        || path
            .components()
            .any(|component| matches!(component, Component::ParentDir | Component::CurDir))
    {
        return Err(reject(
            "expanded absolute paths without parent components are required",
        ));
    }
    Ok(path)
}

fn bounded_deadline(value: &str, started: Instant, now: SystemTime) -> Result<(u64, Instant)> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(reject("an absolute Unix-millisecond deadline is required"));
    }
    let milliseconds = value.parse::<u64>()?;
    let absolute = UNIX_EPOCH
        .checked_add(Duration::from_millis(milliseconds))
        .ok_or_else(|| reject("deadline exceeds the supported clock range"))?;
    let remaining = absolute
        .duration_since(now)
        .map_err(|_| reject("the original deadline already expired"))?;
    if !(Duration::from_secs(5)..=Duration::from_mins(4)).contains(&remaining) {
        return Err(reject(
            "the original deadline must leave five through 240 seconds",
        ));
    }
    Ok((milliseconds, started + remaining))
}

fn selection(arguments: &[String], started: Instant, now: SystemTime) -> Result<Selection> {
    if arguments.len() < 7 {
        return Err(reject(USAGE));
    }
    let profile = parse_profile(&arguments[7..])?;
    let output_tokens = arguments[4].parse::<usize>()?;
    let source_commit = &arguments[6];
    if !(1..=32).contains(&output_tokens)
        || source_commit.len() != 40
        || !source_commit
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(reject("invalid bounded CPU comparison selection"));
    }
    if cfg!(cpu_bundle_legacy_f16) && source_commit != LEGACY_F16_SOURCE {
        return Err(reject(
            "legacy FP16 construction is restricted to the reviewed baseline source",
        ));
    }
    let (deadline_unix_milliseconds, deadline) = bounded_deadline(&arguments[5], started, now)?;
    let generation_deadline = deadline
        .checked_sub(RECEIPT_RESERVE)
        .ok_or_else(|| reject("deadline cannot reserve receipt time"))?;
    Ok(Selection {
        model: absolute_path(&arguments[0])?,
        adapter: absolute_path(&arguments[1])?,
        evidence: absolute_path(&arguments[2])?,
        messages: absolute_path(&arguments[3])?,
        output_tokens,
        source_commit: source_commit.clone(),
        profile,
        deadline_unix_milliseconds,
        deadline,
        generation_deadline,
    })
}

fn hex(digest: impl IntoIterator<Item = u8>) -> String {
    digest
        .into_iter()
        .flat_map(|byte| [byte >> 4, byte & 15])
        .map(|nibble| char::from(b"0123456789abcdef"[usize::from(nibble)]))
        .collect()
}

fn read_input(path: &Path, deadline: Instant) -> Result<(Vec<u8>, String)> {
    check(deadline)?;
    let file = File::open(path)?;
    if !file.metadata()?.is_file() {
        return Err(reject("messages must be a regular file"));
    }
    let mut bytes = Vec::new();
    file.take(MAX_INPUT_BYTES + 1).read_to_end(&mut bytes)?;
    check(deadline)?;
    if bytes.len() > usize::try_from(MAX_INPUT_BYTES)? {
        return Err(reject("public messages exceed the fixed byte bound"));
    }
    let digest = hex(hash_bytes(&bytes)?);
    Ok((bytes, digest))
}

fn executable_hash(deadline: Instant) -> Result<String> {
    let mut file = File::open(std::env::current_exe()?)?;
    if file.metadata()?.len() > MAX_EXECUTABLE_BYTES {
        return Err(reject("collector executable exceeds its fixed byte bound"));
    }
    let mut digest = Hasher::new()?;
    let mut bytes = 0_u64;
    let mut buffer = [0_u8; 8192];
    loop {
        check(deadline)?;
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        bytes += u64::try_from(count)?;
        if bytes > MAX_EXECUTABLE_BYTES {
            return Err(reject("collector executable exceeds its fixed byte bound"));
        }
        digest.update(&buffer[..count])?;
    }
    Ok(hex(digest.finish()?))
}

impl CapturedOutput {
    fn push_observed(&mut self, event: &TextEvent<'_>, elapsed: Duration) -> Result<()> {
        let nonempty = matches!(event, TextEvent::Delta(text) if !text.is_empty());
        self.push(event)?;
        if nonempty && self.first_text_delta_milliseconds.is_none() {
            self.first_text_delta_milliseconds = Some(elapsed.as_secs_f64() * 1000.0);
        }
        Ok(())
    }

    fn push(&mut self, event: &TextEvent<'_>) -> Result<()> {
        if self.terminal.is_some() {
            return Err(reject("output arrived after terminal accounting"));
        }
        match event {
            TextEvent::Delta(text) => {
                if self
                    .text
                    .len()
                    .checked_add(text.len())
                    .is_none_or(|bytes| bytes > MAX_OUTPUT_BYTES)
                {
                    return Err(reject("output exceeds the fixed retained-text bound"));
                }
                self.text.try_reserve(text.len())?;
                self.text.push_str(text);
            }
            TextEvent::Complete(completion) => self.terminal = Some(*completion),
        }
        Ok(())
    }
}

fn complete_request(
    session: &mut SwiftSession,
    selection: &Selection,
    messages: &Value,
    observation: &mut Observation,
) -> Result<()> {
    observation.stage = "request-preparation";
    let started = Instant::now();
    let prepared =
        session.prepare_with_settings(messages, selection.output_tokens, false, Some(1.3));
    observation.preparation_milliseconds = Some(started.elapsed().as_secs_f64() * 1000.0);
    let prepared = prepared?;
    observation.prompt_tokens = Some(prepared.prompt_tokens());
    if !(1..=MAX_PROMPT_TOKENS).contains(&prepared.prompt_tokens()) {
        return Err(reject("rendered prompt exceeds the fixed comparison bound"));
    }
    check(selection.generation_deadline)?;
    observation.stage = "session-generation";
    let mut output = CapturedOutput::default();
    let started = Instant::now();
    let completion = session.complete_with_timings(
        &prepared,
        &Cancellation::default(),
        selection.generation_deadline,
        &mut observation.timings,
        |event| output.push_observed(&event, started.elapsed()),
    );
    observation.request_milliseconds = Some(started.elapsed().as_secs_f64() * 1000.0);
    observation.output = output;
    observation.completion = Some(completion?);
    validate_completion(observation, selection.output_tokens)
}

fn execute(selection: &Selection, observation: &mut Observation) -> Result<()> {
    if !cfg!(all(
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc"
    )) {
        return Err(reject("CPU collection requires native Windows ARM64"));
    }
    observation.stage = "input-intake";
    let (bytes, digest) = read_input(&selection.messages, selection.generation_deadline)?;
    observation.input_sha256 = Some(digest);
    let messages: Value = serde_json::from_slice(&bytes)?;
    observation.stage = "executable-identity";
    observation.executable_sha256 = Some(executable_hash(selection.generation_deadline)?);
    observation.stage = "session-loading";
    let started = Instant::now();
    let loaded = load_session(selection);
    observation.load_milliseconds = Some(started.elapsed().as_secs_f64() * 1000.0);
    let mut session = loaded?;
    observation.model_loaded = true;
    observation.effective_row_workers = Some(session.row_workers());
    observation.observed_adapter_sha256 = session.adapter_identity().map(str::to_owned);
    // This direct source-bound session never installs a DSP/GPU executor.
    // Scalar fallback and optional NEON instructions are both CPU execution.
    session.set_projection_executor(None);
    if session.row_workers() != 4
        || session.requested_backend() != MatvecBackend::ExperimentalNeon
        || session.prompt_reuse() != PromptReuse::Disabled
        || session.adapter_identity() != Some(SWIFT_ADAPTER_SHA256)
        || !profile_matches(selection, &session)
    {
        return Err(reject(
            "constructed CPU session disagrees with the fixed profile",
        ));
    }
    check(selection.generation_deadline)?;
    complete_request(&mut session, selection, &messages, observation)?;
    observation.stage = "session-drop";
    drop(session);
    check(selection.generation_deadline)?;
    observation.stage = "complete";
    Ok(())
}

fn validate_completion(observation: &Observation, output_tokens: usize) -> Result<()> {
    let completion = observation
        .completion
        .ok_or_else(|| reject("session did not return successful accounting"))?;
    if observation.output.terminal != Some(completion)
        || Some(completion.prompt_tokens) != observation.prompt_tokens
        || completion.reused_tokens != 0
        || completion.generated_tokens == 0
        || completion.generated_tokens > output_tokens
        || observation.output.text.is_empty()
        || !observation.timings.entered
        || !observation.timings.succeeded
        || observation.timings.prompt_tokens != completion.prompt_tokens
        || observation.timings.reused_tokens != Some(0)
        || observation.timings.committed_generated_tokens != Some(completion.generated_tokens)
    {
        return Err(reject(
            "successful terminal accounting and nonempty output are required",
        ));
    }
    Ok(())
}

const fn finish_name(finish: Finish) -> &'static str {
    match finish {
        Finish::Stop => "stop",
        Finish::Length => "length",
    }
}

fn phase_records(timings: &GenerationTimings) -> Vec<Value> {
    GenerationPhase::ALL
        .into_iter()
        .map(|phase| {
            let timing = timings.phase(phase);
            json!({
                "phase":phase.name(), "milliseconds":timing.elapsed.as_secs_f64() * 1000.0,
                "attempted_calls":timing.attempted_calls, "completed_calls":timing.completed_calls,
                "attempted_tokens":timing.attempted_tokens,
                "completed_tokens_lower_bound":timing.completed_tokens_lower_bound,
                "partial_progress_unknown":timing.partial_progress_unknown,
            })
        })
        .collect()
}

fn identity_record(selection: &Selection, observation: &Observation) -> Value {
    let mut identity = json!({
        "schema_version":1, "kind":"swift-cpu-bundle-arm",
        "source_commit_declaration":selection.source_commit,
        "source_binding":"requires independent hosted build/executable/collector-source receipt verification",
        "executable_sha256":observation.executable_sha256, "input_sha256":observation.input_sha256,
        "model_sha256":Representation::Pq2.model_sha256(), "model_bytes":Representation::Pq2.model_bytes(),
        "weight_representation":"PQ2_0", "adapter_sha256":SWIFT_ADAPTER_SHA256,
        "observed_adapter_sha256":observation.observed_adapter_sha256,
        "adapter_scale_bits":format!("{:08x}", 1.3_f32.to_bits()),
        "row_workers":4, "effective_row_workers":observation.effective_row_workers,
        "backend_requested":"ExperimentalNeon", "execution_device":"cpu",
        "execution_basis":"direct-session-without-projection-executor",
        "backend_statistics":null, "backend_statistics_scope":"not exposed by the common SwiftSession API; per-operation NEON versus scalar is unknown",
        "kv_cache":"f16", "legacy_f16_build":cfg!(cpu_bundle_legacy_f16),
        "prompt_reuse":"disabled", "seed":42, "thinking_enabled":false,
        "sampler":"production SwiftSampler and model-declared stops",
        "maximum_prompt_tokens":MAX_PROMPT_TOKENS, "output_tokens_reserved":selection.output_tokens,
        "original_deadline_unix_milliseconds":selection.deadline_unix_milliseconds,
        "receipt_reserve_milliseconds":RECEIPT_RESERVE.as_millis(),
    });
    extend_identity(selection, &mut identity);
    identity
}

fn record(
    selection: &Selection,
    observation: &Observation,
    succeeded: bool,
    elapsed: Duration,
) -> Result<Value> {
    let mut record = identity_record(selection, observation);
    record["capture_status"] = json!(if succeeded { "complete" } else { "incomplete" });
    record["arm_succeeded"] = json!(succeeded);
    record["stage"] = json!(observation.stage);
    record["model_loaded"] = json!(observation.model_loaded);
    record["session_succeeded"] = json!(observation.completion.is_some());
    record["terminal_delivered"] = json!(observation.output.terminal.is_some());
    record["terminal_accounting_matches"] = json!(
        observation.completion.is_some() && observation.completion == observation.output.terminal
    );
    record["prompt_tokens"] = json!(observation.prompt_tokens);
    record["reused_tokens"] = json!(observation.completion.map(|value| value.reused_tokens));
    record["generated_tokens"] = json!(observation.completion.map(|value| value.generated_tokens));
    record["finish"] = json!(
        observation
            .completion
            .map(|value| finish_name(value.finish))
    );
    record["output_text"] = json!(
        observation
            .request_milliseconds
            .map(|_| &observation.output.text)
    );
    record["output_bytes"] = json!(
        observation
            .request_milliseconds
            .map(|_| observation.output.text.len())
    );
    record["output_sha256"] = json!(
        observation
            .request_milliseconds
            .map(|_| hash_bytes(observation.output.text.as_bytes()).map(hex))
            .transpose()?
    );
    record["output_committed"] = json!(succeeded);
    record["load_milliseconds"] = json!(observation.load_milliseconds);
    record["preparation_milliseconds"] = json!(observation.preparation_milliseconds);
    record["request_milliseconds"] = json!(observation.request_milliseconds);
    record["first_text_delta_milliseconds"] =
        json!(observation.output.first_text_delta_milliseconds);
    record["first_text_delta_scope"] = json!(
        "first accepted nonempty UTF-8 text callback measured from SwiftSession completion entry; includes prefill and sampling; not a model token timestamp"
    );
    record["generation_phases"] = json!(phase_records(&observation.timings));
    record["generation_phases_scope"] = json!(
        "runtime brackets include validation/allocation; sampling counts draws and callback counts provisional delivery units; completed_tokens_lower_bound is phase-specific, not generated output; phase sums exclude other request overhead"
    );
    record["runtime_milliseconds"] = json!(
        observation
            .timings
            .entered
            .then_some(observation.timings.total.as_secs_f64() * 1000.0)
    );
    record["elapsed_milliseconds"] = json!(elapsed.as_secs_f64() * 1000.0);
    record["timing_scope"] = json!(
        "one direct SwiftSession request; elapsed includes input/hash/load/preparation/request/session-drop, excludes final receipt write and external owner cleanup"
    );
    record["qualification_scope"] = json!(
        "no independent full-state, quality, sustained-use, X1 speedup or promotion verdict; successful bounded external supervision remains required"
    );
    Ok(record)
}

fn write_record(path: &Path, record: &Value) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    serde_json::to_writer_pretty(&mut file, record)?;
    file.write_all(b"\n")?;
    file.flush()?;
    Ok(())
}

fn capture(
    selection: &Selection,
    started: Instant,
    execute: impl FnOnce(&Selection, &mut Observation) -> Result<()>,
) -> Result<()> {
    if !selection.evidence.is_dir() || fs::read_dir(&selection.evidence)?.next().is_some() {
        return Err(reject(
            "each arm requires a new existing empty evidence directory",
        ));
    }
    let mut observation = Observation {
        stage: "started",
        ..Observation::default()
    };
    let (start_name, final_name) = receipt_names();
    write_record(
        &selection.evidence.join(start_name),
        &record(selection, &observation, false, started.elapsed())?,
    )?;
    let result =
        execute(selection, &mut observation).and_then(|()| check(selection.generation_deadline));
    let mut final_record = record(selection, &observation, result.is_ok(), started.elapsed())?;
    final_record["error"] = json!(result.as_ref().err().map(ToString::to_string));
    write_record(&selection.evidence.join(final_name), &final_record)?;
    check(selection.deadline)?;
    result
}

fn run(arguments: &[String]) -> Result<()> {
    if arguments.len() == 1 && arguments[0] == "--help" {
        std::io::stdout()
            .lock()
            .write_all(format!("{USAGE}\n").as_bytes())?;
        return Ok(());
    }
    let started = Instant::now();
    let selection = selection(arguments, started, SystemTime::now())?;
    capture(&selection, started, execute)
}

fn main() -> ExitCode {
    match run(&std::env::args().skip(1).collect::<Vec<_>>()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            let _ = serde_json::to_writer(
                std::io::stderr().lock(),
                &json!({"event":"cpu-bundle-error", "error":error.to_string()}),
            );
            ExitCode::FAILURE
        }
    }
}
