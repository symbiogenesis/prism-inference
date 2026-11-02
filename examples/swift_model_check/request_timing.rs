//! One bounded production-session request, with value-free core attribution.

use super::{AdapterSelection, Representation, evidence_backend, reject};
use bonsai_prism_runtime::{
    Result,
    kernels::{MatvecBackend, RowParallelism},
    model::lora::SWIFT_ADAPTER_SHA256,
    runtime::{Cancellation, Completion, GenerationPhase, GenerationTimings},
    swift_session::SwiftSession,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::{Read, Write},
    path::Path,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const MAX_INPUT_BYTES: u64 = 64 * 1024;
const MAX_EXECUTABLE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_PROMPT_TOKENS: usize = 128;

struct Selection<'a> {
    model: &'a Path,
    messages: &'a Path,
    backend: MatvecBackend,
    representation: Representation,
    adapter: Option<&'a AdapterSelection>,
    row_workers: usize,
    output_tokens: usize,
    source_commit: &'a str,
    deadline_unix_milliseconds: u64,
    deadline: Instant,
}

#[derive(Default)]
struct Observation {
    stage: &'static str,
    model_loaded: bool,
    request_prepared: bool,
    effective_row_workers: Option<usize>,
    input_sha256: Option<String>,
    executable_sha256: Option<String>,
    load_milliseconds: Option<f64>,
    preparation_milliseconds: Option<f64>,
    session_milliseconds: Option<f64>,
    timings: GenerationTimings,
    completion: Option<Completion>,
}

fn bounded_deadline(value: &str, started: Instant, now: SystemTime) -> Result<(u64, Instant)> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(reject(
            "an absolute deadline in Unix milliseconds is required",
        ));
    }
    let milliseconds = value.parse::<u64>()?;
    let absolute = UNIX_EPOCH
        .checked_add(Duration::from_millis(milliseconds))
        .ok_or_else(|| reject("absolute deadline exceeds the supported clock range"))?;
    let remaining = absolute
        .duration_since(now)
        .map_err(|_| reject("request deadline already expired"))?;
    if remaining < Duration::from_secs(1) || remaining > Duration::from_secs(240) {
        return Err(reject(
            "original deadline must leave from one through 240 seconds",
        ));
    }
    Ok((milliseconds, started + remaining))
}

fn check(deadline: Instant) -> Result<()> {
    if Instant::now() >= deadline {
        return Err(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            "original request deadline expired",
        )
        .into());
    }
    Ok(())
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
    let input = File::open(path)?;
    if !input.metadata()?.is_file() {
        return Err(reject("messages input must be a regular file"));
    }
    let mut bytes = Vec::new();
    input.take(MAX_INPUT_BYTES + 1).read_to_end(&mut bytes)?;
    check(deadline)?;
    if bytes.len() > usize::try_from(MAX_INPUT_BYTES)? {
        return Err(reject("messages input exceeds its byte bound"));
    }
    // The exact same captured bytes are hashed before parsing; no reread race.
    let digest = hex(Sha256::digest(&bytes));
    Ok((bytes, digest))
}

fn executable_hash(deadline: Instant) -> Result<String> {
    let mut file = File::open(std::env::current_exe()?)?;
    if file.metadata()?.len() > MAX_EXECUTABLE_BYTES {
        return Err(reject("diagnostic executable exceeds its byte bound"));
    }
    let mut digest = Sha256::new();
    let mut bytes = 0_u64;
    let mut buffer = [0_u8; 8_192];
    loop {
        check(deadline)?;
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        bytes += u64::try_from(count)?;
        if bytes > MAX_EXECUTABLE_BYTES {
            return Err(reject("diagnostic executable exceeds its byte bound"));
        }
        digest.update(&buffer[..count]);
    }
    Ok(hex(digest.finalize()))
}

fn execute(selection: &Selection<'_>, observation: &mut Observation) -> Result<()> {
    observation.stage = "input-intake";
    let (bytes, digest) = read_input(selection.messages, selection.deadline)?;
    observation.input_sha256 = Some(digest);
    let messages: Value = serde_json::from_slice(&bytes)?;
    observation.stage = "executable-identity";
    observation.executable_sha256 = Some(executable_hash(selection.deadline)?);
    observation.stage = "session-loading";
    check(selection.deadline)?;
    let parallelism = RowParallelism::new(selection.row_workers)?;
    let started = Instant::now();
    let loaded = SwiftSession::load_profile_with_row_parallelism(
        selection.model,
        selection.representation,
        selection.adapter.map(|value| value.path.as_path()),
        selection.backend,
        42,
        parallelism,
    );
    observation.load_milliseconds = Some(started.elapsed().as_secs_f64() * 1000.0);
    let mut session = loaded?;
    observation.model_loaded = true;
    observation.effective_row_workers = Some(session.row_workers());
    check(selection.deadline)?;
    observation.stage = "request-preparation";
    let started = Instant::now();
    let prepared = session.prepare_with_settings(
        &messages,
        selection.output_tokens,
        false,
        selection.adapter.map(|value| value.scale),
    );
    observation.preparation_milliseconds = Some(started.elapsed().as_secs_f64() * 1000.0);
    let prepared = prepared?;
    if prepared.prompt_tokens() > MAX_PROMPT_TOKENS {
        return Err(reject(
            "rendered prompt exceeds bounded diagnostic token count",
        ));
    }
    observation.request_prepared = true;
    check(selection.deadline)?;
    observation.stage = "session-generation";
    let started = Instant::now();
    let completion = session.complete_with_timings(
        &prepared,
        &Cancellation::default(),
        selection.deadline,
        &mut observation.timings,
        |_| Ok(()),
    );
    observation.session_milliseconds = Some(started.elapsed().as_secs_f64() * 1000.0);
    observation.completion = Some(completion?);
    observation.stage = "complete";
    Ok(())
}

fn phase_records(timings: &GenerationTimings) -> Vec<Value> {
    GenerationPhase::ALL
        .into_iter()
        .map(|phase| {
            let value = timings.phase(phase);
            json!({
                "phase":phase.name(), "milliseconds":value.elapsed.as_secs_f64()*1000.0,
                "attempted_calls":value.attempted_calls, "completed_calls":value.completed_calls,
                "attempted_tokens":value.attempted_tokens,
                "completed_tokens_lower_bound":value.completed_tokens_lower_bound,
                "partial_progress_unknown":value.partial_progress_unknown,
            })
        })
        .collect()
}

fn identity_record(selection: &Selection<'_>, observation: &Observation) -> Value {
    json!({
        "schema_version":1, "kind":"swift-source-request-timing",
        "capture_status":if observation.stage == "started" { "in-progress" } else { "returned" },
        "scope":"one production SwiftSession request with an in-process no-op text consumer; no HTTP or service qualification",
        "source_commit_declaration":selection.source_commit, "executable_sha256":observation.executable_sha256,
        "source_binding":"declared commit must independently match the hosted build and executable artifact receipt",
        "package_version":env!("CARGO_PKG_VERSION"), "input_bytes_sha256":observation.input_sha256,
        "weight_representation":selection.representation.name(), "model_sha256":selection.representation.model_sha256(),
        "selected_adapter_sha256":selection.adapter.map(|_| SWIFT_ADAPTER_SHA256),
        "adapter_scale_bits":selection.adapter.map(|value| format!("{:08x}", value.scale.to_bits())),
        "configured_row_workers":selection.row_workers, "effective_row_workers":observation.effective_row_workers,
        "backend_requested":format!("{:?}", selection.backend), "seed":42, "thinking_enabled":false,
        "sampler":"production SwiftSampler and model-declared native stops", "prompt_reuse":"disabled",
        "maximum_prompt_tokens":MAX_PROMPT_TOKENS, "output_tokens_reserved":selection.output_tokens,
        "original_deadline_unix_milliseconds":selection.deadline_unix_milliseconds,
    })
}

fn record(
    selection: &Selection<'_>,
    observation: &Observation,
    succeeded: bool,
    elapsed: Duration,
) -> Value {
    // Keep each macro expansion bounded while preserving the flat receipt schema.
    let mut record = identity_record(selection, observation);
    record["elapsed_milliseconds"] = json!(elapsed.as_secs_f64() * 1000.0);
    record["stage"] = json!(observation.stage);
    record["model_loaded"] = json!(observation.model_loaded);
    record["request_prepared"] = json!(observation.request_prepared);
    record["diagnostic_succeeded"] = json!((observation.stage != "started").then_some(succeeded));
    record["session_succeeded"] = json!(
        observation
            .session_milliseconds
            .map(|_| observation.completion.is_some())
    );
    record["session_milliseconds"] = json!(observation.session_milliseconds);
    record["load_milliseconds"] = json!(observation.load_milliseconds);
    record["preparation_milliseconds"] = json!(observation.preparation_milliseconds);
    record["runtime_entered"] = json!(observation.timings.entered);
    record["runtime_succeeded"] = json!(observation.timings.succeeded);
    record["runtime_milliseconds"] = json!(observation.timings.total.as_secs_f64() * 1000.0);
    record["prompt_tokens"] = json!(
        observation
            .timings
            .entered
            .then_some(observation.timings.prompt_tokens)
    );
    record["reused_tokens"] = json!(observation.timings.reused_tokens);
    record["runtime_committed_generated_tokens"] =
        json!(observation.timings.committed_generated_tokens);
    record["session_committed_generated_tokens"] =
        json!(observation.completion.map(|value| value.generated_tokens));
    record["phases"] = json!(phase_records(&observation.timings));
    record["counter_scope"] = json!(
        "successful callback units remain provisional; failed block/callback progress is unknown beyond earlier completed lower bounds"
    );
    record["timing_scope"] = json!(
        "runtime operation brackets include checks and allocation; callback includes UTF-8 processing/no-op consumer; no network backpressure or isolated-kernel claim"
    );
    record["cleanup_scope"] = json!(
        "elapsed includes returned session drop but excludes final receipt write and external owner cleanup; original aggregate owner clock and cleanup reserve remain mandatory"
    );
    record
}

#[allow(
    clippy::redundant_pub_crate,
    reason = "Private example subcommand used only by its parent dispatcher."
)]
pub(super) fn run(
    model: &Path,
    arguments: &[String],
    representation: Representation,
    adapter: Option<&AdapterSelection>,
    row_workers: usize,
) -> Result<()> {
    let started = Instant::now();
    let wall = SystemTime::now();
    if arguments.len() != 6 {
        return Err(reject(
            "request-timing requires evidence, backend, messages, output bound, original deadline and source commit",
        ));
    }
    let (evidence, backend) = evidence_backend(arguments)?;
    let messages = Path::new(&arguments[2]);
    let output_tokens = arguments[3].parse::<usize>()?;
    let (deadline_unix_milliseconds, deadline) = bounded_deadline(&arguments[4], started, wall)?;
    let source_commit = arguments[5].as_str();
    if !messages.is_absolute()
        || !(1..=16).contains(&output_tokens)
        || source_commit.len() != 40
        || !source_commit
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(reject("invalid bounded request timing selection"));
    }
    let selection = Selection {
        model,
        messages,
        backend,
        representation,
        adapter,
        row_workers,
        output_tokens,
        source_commit,
        deadline_unix_milliseconds,
        deadline,
    };
    capture(evidence, &selection, started, execute)
}

fn capture(
    evidence: &Path,
    selection: &Selection<'_>,
    started: Instant,
    execute: impl FnOnce(&Selection<'_>, &mut Observation) -> Result<()>,
) -> Result<()> {
    let mut started_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(evidence.join("request-timing-started.json"))?;
    let mut final_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(evidence.join("request-timing.json"))?;
    let mut observation = Observation {
        stage: "started",
        ..Observation::default()
    };
    serde_json::to_writer_pretty(
        &mut started_file,
        &record(selection, &observation, false, started.elapsed()),
    )?;
    started_file.flush()?;
    let result = execute(selection, &mut observation).and_then(|()| {
        // Session destruction remains inside the original allowance as well.
        // Receipt writing and external containment cleanup use the owner's
        // separately retained reserve; they never create another model budget.
        observation.stage = "original-deadline-check";
        check(selection.deadline)?;
        observation.stage = "complete";
        Ok(())
    });
    serde_json::to_writer_pretty(
        &mut final_file,
        &record(selection, &observation, result.is_ok(), started.elapsed()),
    )?;
    final_file.flush()?;
    result.map_err(|_| {
        reject("request timing diagnostic failed; inspect the retained fixed-stage receipt")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn selection() -> Selection<'static> {
        Selection {
            model: Path::new("private-model"),
            messages: Path::new("private-prompt"),
            backend: MatvecBackend::Scalar,
            representation: Representation::CurrentRepackedPtq1,
            adapter: None,
            row_workers: 4,
            output_tokens: 1,
            source_commit: "0000000000000000000000000000000000000000",
            deadline_unix_milliseconds: 1,
            deadline: Instant::now() + Duration::from_secs(60),
        }
    }

    #[test]
    fn request_timing_requires_original_bounded_absolute_deadline() -> Result<()> {
        let started = Instant::now();
        let now = UNIX_EPOCH + Duration::from_secs(1000);
        for invalid in [
            "",
            "-1",
            "1000000",
            "1000999",
            "1240001",
            "18446744073709551615",
            "999999999999999999999",
        ] {
            assert!(bounded_deadline(invalid, started, now).is_err());
        }
        let (value, deadline) = bounded_deadline("1123000", started, now)?;
        assert_eq!(value, 1_123_000);
        assert_eq!(deadline.duration_since(started), Duration::from_secs(123));
        Ok(())
    }

    #[test]
    fn unentered_failure_record_has_no_inferred_token_counts_or_input_text() {
        let selection = selection();
        let observation = Observation {
            stage: "session-loading",
            ..Observation::default()
        };
        let record = record(&selection, &observation, false, Duration::ZERO);
        assert_eq!(record["runtime_entered"], false);
        assert!(record["prompt_tokens"].is_null());
        assert!(record["runtime_committed_generated_tokens"].is_null());
        assert!(record["session_committed_generated_tokens"].is_null());
        assert!(!record.to_string().contains("private-"));
    }

    #[test]
    fn completed_session_stays_distinct_from_failed_aggregate_deadline() {
        let observation = Observation {
            stage: "original-deadline-check",
            session_milliseconds: Some(1.0),
            completion: Some(Completion {
                prompt_tokens: 2,
                reused_tokens: 0,
                generated_tokens: 1,
                finish: bonsai_prism_runtime::runtime::Finish::Length,
            }),
            ..Observation::default()
        };
        let record = record(&selection(), &observation, false, Duration::ZERO);
        assert_eq!(record["session_succeeded"], true);
        assert_eq!(record["diagnostic_succeeded"], false);
        assert_eq!(record["session_committed_generated_tokens"], 1);
    }

    #[test]
    fn request_timing_retains_failure_receipt_and_never_overwrites_prior_capture() -> Result<()> {
        let directory = std::env::temp_dir().join(format!(
            "Prism-Request-Timing-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos(),
        ));
        std::fs::create_dir(&directory)?;
        let result = capture(
            &directory,
            &selection(),
            Instant::now(),
            |_, observation| {
                observation.stage = "session-loading";
                Err(reject("private-error-value"))
            },
        );
        assert!(result.is_err());
        let started_path = directory.join("request-timing-started.json");
        let final_path = directory.join("request-timing.json");
        let initial: Value = serde_json::from_slice(&std::fs::read(&started_path)?)?;
        assert_eq!(initial["capture_status"], "in-progress");
        assert!(initial["diagnostic_succeeded"].is_null());
        let retained = std::fs::read(&final_path)?;
        let finished: Value = serde_json::from_slice(&retained)?;
        assert_eq!(finished["stage"], "session-loading");
        assert_eq!(finished["diagnostic_succeeded"], false);
        assert!(finished["session_succeeded"].is_null());
        assert!(!finished.to_string().contains("private-"));
        let called = std::cell::Cell::new(false);
        assert!(
            capture(&directory, &selection(), Instant::now(), |_, _| {
                called.set(true);
                Ok(())
            })
            .is_err()
        );
        assert!(!called.get());
        assert_eq!(std::fs::read(&final_path)?, retained);
        std::fs::remove_file(started_path)?;
        std::fs::remove_file(final_path)?;
        std::fs::remove_dir(directory)?;
        Ok(())
    }
}
