//! Bounded layer attribution on the unmodified full-token graph path.

use super::{
    AdapterSelection, Representation, evidence_backend, load_graph, reject, request_timing,
    valid_tokens,
};
use bonsai_prism_runtime::{
    Decoder, Result,
    graph::{OperationTiming, TokenTimings},
    model::lora::SWIFT_ADAPTER_SHA256,
};
use serde_json::{Value, json};
use std::{
    fs::OpenOptions,
    io::Write,
    path::Path,
    time::{Instant, SystemTime},
};

fn operation(timing: OperationTiming) -> Value {
    json!({"attempted":timing.attempted,"completed":timing.completed,
        "milliseconds":timing.attempted.then_some(timing.elapsed.as_secs_f64()*1000.0)})
}

fn token(timing: &TokenTimings) -> Value {
    let layers: Vec<_> = timing.layers().iter().enumerate().map(|(layer, timing)| {
        json!({"layer":layer,"attention":operation(timing.attention),"ffn":operation(timing.ffn)})
    }).collect();
    json!({"entered":timing.entered,"committed":timing.succeeded,
        "evaluationMilliseconds":timing.entered.then_some(timing.elapsed.as_secs_f64()*1000.0),
        "layers":layers})
}

#[derive(Default)]
struct Observation {
    stage: &'static str,
    input_sha256: Option<String>,
    executable_sha256: Option<String>,
    graph_identity: Option<String>,
    row_workers: Option<usize>,
    load_milliseconds: Option<f64>,
    requested_tokens: Option<usize>,
    tokens: Vec<Value>,
}

struct Selection<'a> {
    model: &'a Path,
    arguments: &'a [String],
    representation: Representation,
    adapter: Option<&'a AdapterSelection>,
    row_workers: usize,
    deadline_unix_milliseconds: u64,
    deadline: Instant,
}

fn record(
    selection: &Selection<'_>,
    observation: &Observation,
    success: Option<bool>,
    started: Instant,
) -> Value {
    json!({"schemaVersion":1,"kind":"experimental-ffn-layer-timing-v1",
        "captureStatus":if success.is_none() {"in-progress"} else {"returned"},
        "diagnosticSucceeded":success,"stage":observation.stage,
        "declaredSourceCommit":selection.arguments[4],"sourceAuthenticated":false,
        "executableSha256":observation.executable_sha256,
        "inputSha256":observation.input_sha256,
        "modelSha256":selection.representation.model_sha256(),
        "weightRepresentation":selection.representation.name(),
        "adapterSha256":selection.adapter.map(|_| SWIFT_ADAPTER_SHA256),
        "adapterScaleBits":format!("{:08x}",selection.adapter.map_or(0.0, |adapter| adapter.scale).to_bits()),
        "graphIdentity":observation.graph_identity,"backendRequested":selection.arguments[1],
        "requestedRowWorkers":selection.row_workers,"effectiveRowWorkers":observation.row_workers,
        "originalDeadlineUnixMilliseconds":selection.deadline_unix_milliseconds,
        "elapsedMilliseconds":started.elapsed().as_secs_f64()*1000.0,
        "loadMilliseconds":observation.load_milliseconds,
        "requestedTokens":observation.requested_tokens,"tokens":observation.tokens,
        "workload":"fixed supplied IDs; full logits on every token; no sampling, tiling or narrowed ingestion",
        "attentionScope":"input normalization and full/recurrent attention, excluding attention residual add",
        "ffnScope":"attention residual add, post-attention normalization, actual FFN/fusion/standard adapter and FFN residual add",
        "counterScope":"completed layer brackets are provisional; only token committed=true establishes graph-state commit",
        "excludedFromLayerBrackets":"embedding, vocabulary output, graph-state commit, loading and receipt I/O",
        "powerObservations":"external owner required; unknown in this helper",
        "cleanupScope":"model drop charged before final deadline check; final receipt and external process cleanup require original owner reserve",
        "performanceComparisonEligible":false,"processMemoryQualified":false,"answerQualityQualified":false})
}

fn execute(selection: &Selection<'_>, observation: &mut Observation) -> Result<()> {
    observation.stage = "input-intake";
    let (bytes, hash) =
        request_timing::read_input(Path::new(&selection.arguments[2]), selection.deadline)?;
    observation.input_sha256 = Some(hash);
    let tokens: Vec<u32> = serde_json::from_slice(&bytes)?;
    if !valid_tokens(&tokens) || tokens.len() > 16 {
        return Err(reject("ffn-timing requires 1..16 valid supplied token IDs"));
    }
    observation.requested_tokens = Some(tokens.len());
    observation.stage = "executable-identity";
    observation.executable_sha256 = Some(request_timing::executable_hash(selection.deadline)?);
    observation.stage = "model-loading";
    request_timing::check(selection.deadline)?;
    let (_, backend) = evidence_backend(selection.arguments)?;
    let started = Instant::now();
    let mut graph = load_graph(
        selection.model,
        backend,
        selection.representation,
        selection.adapter,
        selection.row_workers,
    )?;
    observation.load_milliseconds = Some(started.elapsed().as_secs_f64() * 1000.0);
    observation.graph_identity = Some(graph.identity().to_owned());
    observation.row_workers = Some(graph.row_workers());
    observation.stage = "token-evaluation";
    let mut logits = Vec::new();
    let mut timings = TokenTimings::default();
    for input in tokens {
        request_timing::check(selection.deadline)?;
        let result = graph.evaluate_with_timings_into(input, &mut logits, &mut timings);
        observation.tokens.push(token(&timings));
        result?;
        request_timing::check(selection.deadline)?;
    }
    drop(graph);
    observation.stage = "original-deadline-check";
    request_timing::check(selection.deadline)?;
    observation.stage = "complete";
    Ok(())
}

#[expect(
    clippy::redundant_pub_crate,
    reason = "The parent example exclusively owns this command entry point."
)]
pub(super) fn run(
    model: &Path,
    arguments: &[String],
    representation: Representation,
    adapter: Option<&AdapterSelection>,
    row_workers: usize,
) -> Result<()> {
    let started = Instant::now();
    let now = SystemTime::now();
    if arguments.len() != 5
        || !Path::new(&arguments[2]).is_absolute()
        || arguments[4].len() != 40
        || !arguments[4]
            .bytes()
            .all(|value| value.is_ascii_digit() || (b'a'..=b'f').contains(&value))
    {
        return Err(reject(
            "ffn-timing requires evidence, backend, absolute token input, original deadline and source commit",
        ));
    }
    let (evidence, _) = evidence_backend(arguments)?;
    let (deadline_unix_milliseconds, deadline) =
        request_timing::bounded_deadline(&arguments[3], started, now)?;
    let selection = Selection {
        model,
        arguments,
        representation,
        adapter,
        row_workers,
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
    let mut initial = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(evidence.join("ffn-timing-started.json"))?;
    let mut terminal = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(evidence.join("ffn-timing.json"))?;
    let mut observation = Observation {
        stage: "started",
        ..Observation::default()
    };
    serde_json::to_writer_pretty(
        &mut initial,
        &record(selection, &observation, None, started),
    )?;
    initial.flush()?;
    let result = execute(selection, &mut observation);
    serde_json::to_writer_pretty(
        &mut terminal,
        &record(selection, &observation, Some(result.is_ok()), started),
    )?;
    terminal.flush()?;
    result.map_err(|_| reject("FFN attribution failed; inspect the retained fixed-stage receipt"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ffn_timing_retains_failed_stage_without_input_values_and_never_overwrites() -> Result<()> {
        let directory = std::env::temp_dir().join(format!(
            "Prism-Ffn-Timing-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        std::fs::create_dir(&directory)?;
        let arguments = vec![
            "private-evidence".into(),
            "scalar".into(),
            "private-input".into(),
            "1".into(),
            "0".repeat(40),
        ];
        let selection = Selection {
            model: Path::new("private-model"),
            arguments: &arguments,
            representation: Representation::Pq2,
            adapter: None,
            row_workers: 1,
            deadline_unix_milliseconds: 1,
            deadline: Instant::now(),
        };
        assert!(
            capture(&directory, &selection, Instant::now(), |_, observation| {
                observation.stage = "model-loading";
                Err(reject("private failure detail"))
            })
            .is_err()
        );
        let path = directory.join("ffn-timing.json");
        let bytes = std::fs::read(&path)?;
        let record: Value = serde_json::from_slice(&bytes)?;
        assert_eq!(record["stage"], "model-loading");
        assert_eq!(record["diagnosticSucceeded"], false);
        assert!(record["requestedTokens"].is_null());
        assert!(!record.to_string().contains("private-"));
        assert!(
            capture(&directory, &selection, Instant::now(), |_, _| Err(reject(
                "must not run"
            )))
            .is_err()
        );
        assert_eq!(std::fs::read(&path)?, bytes);
        std::fs::remove_file(path)?;
        std::fs::remove_file(directory.join("ffn-timing-started.json"))?;
        std::fs::remove_dir(directory)?;
        Ok(())
    }
}
