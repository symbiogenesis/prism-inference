#![forbid(unsafe_code)]
//! One fixed factual smoke arm, externally supervised with an original deadline.
//!
//! Standard versus absent adapter is an ablation ceiling, not a quality verdict.

use std::{
    fmt::Write as _,
    fs::{File, OpenOptions},
    io::{Read as _, Write as _},
    path::{Path, PathBuf},
    process::ExitCode,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use bonsai_native_hash::Hasher;
use bonsai_prism_runtime::{
    Result,
    graph::KvCachePolicy,
    kernels::{MatvecBackend, RowParallelism},
    model::{lora::SWIFT_ADAPTER_SHA256, swift::Representation},
    runtime::{Cancellation, Finish, GenerationPhase, GenerationTimings, PromptReuse},
    swift_session::{SwiftSession, TextEvent},
};
use serde::Deserialize;
use serde_json::{Value, json};

const OUTPUT_TOKENS: usize = 8;
const CASES: [(&str, &str, &str); 4] = [
    (
        "capital",
        "What is the capital of France? Answer with only the city name.",
        "paris",
    ),
    (
        "arithmetic",
        "What is 2 + 2? Answer with only the numeral.",
        "4",
    ),
    (
        "chemistry",
        "What is the chemical formula of water? Use plain ASCII. Answer with only the formula.",
        "h2o",
    ),
    (
        "planet",
        "What is the largest planet in our Solar System? Answer with only its name.",
        "jupiter",
    ),
];

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Request {
    model_path: PathBuf,
    adapter_path: Option<PathBuf>,
    condition: String,
    source_commit: String,
    deadline_unix_milliseconds: u64,
    output_path: PathBuf,
}

fn reject(message: &str) -> bonsai_prism_runtime::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message).into()
}

fn hash_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Hasher::new()?;
    let mut buffer = [0_u8; 8192];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count])?;
    }
    let mut digest = String::with_capacity(64);
    for byte in hash.finish()? {
        write!(digest, "{byte:02x}")?;
    }
    Ok(digest)
}

fn normalized(text: &str) -> String {
    text.trim()
        .trim_end_matches(['.', '!'])
        .to_ascii_lowercase()
}

fn check(deadline: Instant) -> Result<()> {
    if Instant::now() >= deadline {
        return Err(reject("The original factual-smoke deadline expired."));
    }
    Ok(())
}

fn timing_record(timings: &GenerationTimings) -> Value {
    json!({"entered":timings.entered, "succeeded":timings.succeeded,
        "totalMilliseconds":timings.total.as_secs_f64() * 1000.0,
        "decodeMilliseconds":timings.phase(GenerationPhase::DecodeEvaluation).elapsed.as_secs_f64() * 1000.0,
        "prefillMilliseconds":(timings.phase(GenerationPhase::PrefillIngestion).elapsed.as_secs_f64()
            + timings.phase(GenerationPhase::PrefillEvaluation).elapsed.as_secs_f64()) * 1000.0,
        "committedGeneratedTokens":timings.committed_generated_tokens})
}

#[expect(
    clippy::too_many_lines,
    reason = "One fixed arm retains loading, all four answers and cleanup in one transaction."
)]
fn execute(request: &Request, deadline: Instant, record: &mut Value) -> Result<()> {
    record["stage"] = json!("executable-identity");
    record["executableSha256"] = json!(hash_file(&std::env::current_exe()?)?);
    check(deadline)?;
    record["stage"] = json!("model-load");
    let started = Instant::now();
    let session = SwiftSession::load_profile_with_context(
        &request.model_path,
        Representation::Pq2,
        request.adapter_path.as_deref(),
        MatvecBackend::ExperimentalNeon,
        42,
        RowParallelism::new(4)?,
        KvCachePolicy::Rotated4,
        256,
    );
    record["loadMilliseconds"] = json!(started.elapsed().as_secs_f64() * 1000.0);
    let mut session = session?;
    session.set_projection_executor(None);
    if session.row_workers() != 4
        || session.prompt_reuse() != PromptReuse::Disabled
        || session.adapter_identity() != request.adapter_path.as_ref().map(|_| SWIFT_ADAPTER_SHA256)
    {
        return Err(reject(
            "The factual smoke session disagrees with its fixed profile.",
        ));
    }
    record["modelLoaded"] = json!(true);
    record["graphIdentity"] = json!(session.decoder_identity());
    for (id, prompt, expected) in CASES {
        check(deadline)?;
        record["stage"] = json!(id);
        session.reset()?;
        let messages = json!([{"role":"user", "content":prompt}]);
        let started = Instant::now();
        let prepared = session.prepare_with_settings(
            &messages,
            OUTPUT_TOKENS,
            false,
            request.adapter_path.as_ref().map(|_| 1.3),
        )?;
        if prepared.prompt_tokens() > 128 {
            return Err(reject("The fixed prompt exceeded 128 tokens."));
        }
        let preparation = started.elapsed().as_secs_f64() * 1000.0;
        let mut text = String::new();
        let mut terminal = None;
        let mut timings = GenerationTimings::default();
        let started = Instant::now();
        let completed = session.complete_with_timings(
            &prepared,
            &Cancellation::default(),
            deadline,
            &mut timings,
            |event| {
                match event {
                    TextEvent::Delta(delta) if text.len() + delta.len() <= 4096 => {
                        text.push_str(delta);
                    }
                    TextEvent::Delta(_) => {
                        return Err(reject("Factual answer exceeded its text bound."));
                    }
                    TextEvent::Complete(value) => terminal = Some(value),
                }
                Ok(())
            },
        );
        let elapsed = started.elapsed().as_secs_f64() * 1000.0;
        let row = match completed {
            Ok(completion) => {
                let complete = terminal == Some(completion) && completion.finish == Finish::Stop;
                let passed = complete && normalized(&text) == expected;
                json!({"id":id, "prompt":prompt, "expected":expected, "text":text,
                    "complete":complete, "passed":passed,
                    "finish":if completion.finish == Finish::Stop {"stop"} else {"length"},
                    "promptTokens":completion.prompt_tokens, "generatedTokens":completion.generated_tokens,
                    "reusedTokens":completion.reused_tokens, "preparationMilliseconds":preparation,
                    "requestMilliseconds":elapsed, "timings":timing_record(&timings), "error":null})
            }
            Err(error) => json!({"id":id, "prompt":prompt, "expected":expected, "text":text,
                "complete":false, "passed":false, "preparationMilliseconds":preparation,
                "requestMilliseconds":elapsed, "timings":timing_record(&timings), "error":error.to_string()}),
        };
        let failed = !row["error"].is_null();
        record["cases"]
            .as_array_mut()
            .ok_or_else(|| reject("Missing cases."))?
            .push(row);
        if failed {
            return Err(reject(
                "A factual smoke request failed; this arm will not continue.",
            ));
        }
    }
    record["graphIdentity"] = json!(session.decoder_identity());
    let memory = session.state_memory()?;
    record["stateMemory"] = json!({"position":memory.position,
        "contextTokens":memory.context_tokens, "kvPayloadBytes":memory.kv_payload_bytes,
        "recurrentPayloadBytes":memory.recurrent_payload_bytes,
        "liveStateStorageBytes":memory.live_state_storage_bytes});
    drop(session);
    check(deadline)?;
    record["stage"] = json!("complete");
    record["armCompleted"] = json!(true);
    Ok(())
}

fn run() -> Result<()> {
    let started = Instant::now();
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    if arguments.len() != 1 {
        return Err(reject("Usage: swift_sanity_check <absolute-request.json>"));
    }
    let input = PathBuf::from(&arguments[0]);
    if !input.is_absolute() {
        return Err(reject("Request must be an absolute path."));
    }
    let mut bytes = Vec::new();
    File::open(input)?.take(65_537).read_to_end(&mut bytes)?;
    if bytes.len() > 65_536 {
        return Err(reject("Request exceeds 64 KiB."));
    }
    let request: Request = serde_json::from_slice(&bytes)?;
    if !request.model_path.is_absolute()
        || !request.output_path.is_absolute()
        || request
            .adapter_path
            .as_ref()
            .is_some_and(|path| !path.is_absolute())
        || !matches!(request.condition.as_str(), "standard" | "none")
        || (request.condition == "standard") != request.adapter_path.is_some()
        || request.source_commit.len() != 40
        || !request
            .source_commit
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(reject(
            "Invalid fixed factual smoke profile or source declaration.",
        ));
    }
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?;
    let remaining = Duration::from_millis(request.deadline_unix_milliseconds)
        .checked_sub(now)
        .ok_or_else(|| reject("Original deadline already expired."))?;
    if !(Duration::from_secs(5)..=Duration::from_secs(120)).contains(&remaining) {
        return Err(reject(
            "Original arm deadline must leave 5 through 120 seconds.",
        ));
    }
    let deadline = Instant::now()
        .checked_add(remaining)
        .and_then(|deadline| deadline.checked_sub(Duration::from_secs(2)))
        .ok_or_else(|| reject("The factual-smoke deadline overflowed."))?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&request.output_path)?;
    let mut record = json!({"schemaVersion":1, "kind":"swift-fixed-factual-smoke",
        "sourceCommitDeclaration":request.source_commit, "condition":request.condition,
        "modelSha256":Representation::Pq2.model_sha256(), "weightRepresentation":"PQ2_0",
        "adapterSha256":request.adapter_path.as_ref().map(|_| SWIFT_ADAPTER_SHA256),
        "adapterScaleBits":if request.adapter_path.is_some() {"3fa66666"} else {"00000000"},
        "rowWorkers":4, "backend":"experimental-neon-cpu", "kvCache":"rsq4",
        "contextTokens":256, "outputTokensPerCase":OUTPUT_TOKENS, "seed":42,
        "thinking":false, "promptReuse":"disabled", "cases":[], "armCompleted":false,
        "modelLoaded":false, "stage":"admitted", "error":null,
        "limit":"Four factual answers only; no broader quality, statistical performance or promotion verdict."});
    let result = execute(&request, deadline, &mut record);
    if let Err(error) = &result {
        record["error"] = json!(error.to_string());
    }
    record["elapsedMilliseconds"] = json!(started.elapsed().as_secs_f64() * 1000.0);
    serde_json::to_writer(&mut output, &record)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    result
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            let _ = writeln!(std::io::stderr().lock(), "{error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::normalized;

    #[test]
    fn factual_gate_rejects_explanations_and_wrong_answers() {
        assert_eq!(normalized(" Paris.\n"), "paris");
        assert_eq!(normalized("H2O"), "h2o");
        assert_ne!(normalized("Paris is the capital."), "paris");
        assert_ne!(normalized("Saturn"), "jupiter");
    }
}
