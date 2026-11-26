#![forbid(unsafe_code)]
//! Bounded, explicit actual-model load, forward and rollback correctness evidence.

#[path = "swift_model_check/ffn_timing.rs"]
mod ffn_timing;
#[path = "swift_model_check/request_timing.rs"]
mod request_timing;
#[path = "swift_model_check/rollback.rs"]
mod rollback;

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use bonsai_native_hash::Hasher;
use bonsai_prism_runtime::graph::{KvCachePolicy, Qwen35, StateComponent};
use bonsai_prism_runtime::kernels::{MAX_ROW_WORKERS, MatvecBackend, RowParallelism};
use bonsai_prism_runtime::model::lora::{MultiRankAdapter, MultiRankPolicy};
use bonsai_prism_runtime::model::swift::{self, Representation};
use bonsai_prism_runtime::runtime::greedy;
use bonsai_prism_runtime::token_input::read_tokens;
use bonsai_prism_runtime::{Decoder, Result};
use serde_json::{Value, json};

const DEFAULT_MODEL_BUDGET: usize = 8 * 1024 * 1024 * 1024;
const NARIQ_MODEL_BUDGET: usize = 12 * 1024 * 1024 * 1024;

const fn model_budget_for(representation: Representation) -> usize {
    if representation.is_nariq() {
        NARIQ_MODEL_BUDGET
    } else {
        DEFAULT_MODEL_BUDGET
    }
}

const MAX_GREEDY_STEPS: usize = 16;
const MAX_TRAJECTORY_EVIDENCE_BYTES: usize = 8 * 1024 * 1024 * 1024;
const USAGE: &str = concat!(
    "Usage: swift_model_check <inspect|load> <absolute-model>\n",
    "       swift_model_check forward <absolute-model> <absolute-evidence-directory> <scalar|auto|neon> <id,id,...>\n",
    "       swift_model_check trajectory <absolute-model> <absolute-evidence-directory> <scalar|auto|neon> <absolute-prompt-ids.json> <greedy-steps>\n",
    "       swift_model_check rollback <absolute-model> <absolute-evidence-directory> <scalar|auto|neon> <id,id,...>\n",
    "       swift_model_check request-timing <absolute-model> <absolute-evidence-directory> <scalar|auto|neon> <absolute-messages.json> <output-1..16> <absolute-deadline-unix-ms> <source-commit>\n",
    "       swift_model_check ffn-timing <absolute-model> <absolute-evidence-directory> <scalar|auto|neon> <absolute-token-ids.json> <absolute-deadline-unix-ms> <source-commit>\n",
    "Prefix any mode with --bonsai-2-5 <pq2|ptq1> (Ternary Bonsai 2.5 27B; pq2 is default) or --nariq <2.5bpw|3.0bpw|2.0bpw> (opt-in NariQ trellis artifact; mutually exclusive prefixes).\n",
    "For load/forward/trajectory/rollback/request-timing/ffn-timing, --row-workers <1..18> follows any representation prefix and precedes any adapter prefix. Default: 1; larger counts require a parallel-rows build.\n",
    "For load/forward/trajectory/rollback/request-timing/ffn-timing, optionally prefix --multi-rank-adapter <absolute-gguf>[:<scale-0..2>] (default scale 0.0; NpuFriendly policy).\n",
    "forward permits 1..16 exact supplied token IDs and retains its per-token snapshots.\n",
    "trajectory reads a bounded JSON u32 array, reserving all 1..16 greedy steps in the 8192-token context.\n",
    "It snapshots the final prompt token and each fed-back greedy token, including special IDs; model stops do not end this fixed-step diagnostic.\n",
    "Evidence directories must already exist. Files are create-new; trajectory snapshot payload is capped at 8 GiB.\n",
    "rollback permits 1..16 supplied IDs, cancels after one evaluated provisional token and compares full state/tokens/logits before reset plus fresh-recompute continuation; at most 2 GiB of snapshots.\n",
    "Its explicit capture-stop token 0 and raw greedy tokens do not qualify model stops, production sampling or private HTTP cancellation.\n",
    "Evaluation milliseconds exclude model loading, snapshots, greedy selection and output; they do not establish benchmark or service qualification.\n",
    "request-timing uses production Swift sampling/stops with seed 42, thinking disabled, at most 128 rendered prompt tokens and a no-op text consumer.\n",
    "Its original absolute deadline must leave 1..240 seconds at entry, covering intake, loading and generation; the external owner must separately reserve cleanup and enforce the aggregate limit.\n",
    "request-timing writes create-new started/final receipts, including returned failures, with input/executable hashes and a declared source commit requiring external build-receipt verification. It retains no prompt/output text or token IDs.\n",
    "ffn-timing attributes attention/FFN at each layer for 1..16 fixed full-token evaluations with logits, not narrowed/tiled prefill or sampled answers. Same worker/adapter prefixes and original deadline rules apply. It writes value-free started/final receipts; external containment, power observations and source-bound prebuilt helpers remain required.\n",
);

fn reject(message: &str) -> bonsai_prism_runtime::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message).into()
}

fn emit(record: &Value) -> Result<()> {
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(&mut stdout, record)?;
    stdout.write_all(b"\n")?;
    stdout.flush()?;
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            let _ = emit(&json!({"event":"error","error":error.to_string()}));
            ExitCode::FAILURE
        }
    }
}

enum ParsedCommand {
    Help,
    Inspect {
        path: PathBuf,
        representation: Representation,
    },
    Mode {
        mode: String,
        path: PathBuf,
        tail: Vec<String>,
        representation: Representation,
        multi_rank_adapter: Option<MultiRankAdapterSelection>,
        row_workers: usize,
    },
}

fn parse_cli(mut arguments: Vec<String>) -> Result<ParsedCommand> {
    let representation = select_representation(&mut arguments)?;
    let row_workers = select_row_workers(&mut arguments)?;
    let multi_rank_adapter = select_multi_rank_adapter(&mut arguments)?;
    if arguments.len() == 1 && arguments[0] == "--help" {
        return Ok(ParsedCommand::Help);
    }
    if !(2..=8).contains(&arguments.len()) {
        return Err(reject(USAGE));
    }
    let mode = arguments[0].clone();
    let path = PathBuf::from(&arguments[1]);
    if !path.is_absolute()
        || !matches!(
            mode.as_str(),
            "inspect"
                | "load"
                | "forward"
                | "trajectory"
                | "rollback"
                | "request-timing"
                | "ffn-timing"
        )
    {
        return Err(reject(
            "an explicit mode and absolute model path are required",
        ));
    }
    if mode == "inspect" {
        if arguments.len() != 2 || multi_rank_adapter.is_some() || row_workers.is_some() {
            return Err(reject("inspect takes only the model path"));
        }
        return Ok(ParsedCommand::Inspect {
            path,
            representation,
        });
    }
    let tail: Vec<String> = arguments.drain(2..).collect();
    validate_mode_tail(&mode, &tail)?;
    Ok(ParsedCommand::Mode {
        mode,
        path,
        tail,
        representation,
        multi_rank_adapter,
        row_workers: row_workers.unwrap_or(1),
    })
}

fn validate_mode_tail(mode: &str, tail: &[String]) -> Result<()> {
    match mode {
        "load" if !tail.is_empty() => Err(reject("load takes only the model path")),
        "forward" if tail.len() != 3 => Err(reject(
            "forward requires output directory, backend and token IDs",
        )),
        "trajectory" if tail.len() != 4 => Err(reject(
            "trajectory requires output directory, backend, prompt-ID file and greedy steps",
        )),
        "rollback" if tail.len() != 3 => Err(reject(
            "rollback requires output directory, backend and 1..16 token IDs",
        )),
        "request-timing" if tail.len() != 6 => Err(reject(
            "request-timing requires evidence, backend, messages, output bound, original deadline and source commit",
        )),
        "ffn-timing" if tail.len() != 5 => Err(reject(
            "ffn-timing requires evidence, backend, absolute token input, original deadline and source commit",
        )),
        _ => Ok(()),
    }
}

fn run() -> Result<()> {
    match parse_cli(std::env::args().skip(1).collect())? {
        ParsedCommand::Help => {
            std::io::stdout().lock().write_all(USAGE.as_bytes())?;
            Ok(())
        }
        ParsedCommand::Inspect {
            path,
            representation,
        } => report(
            "header-verified",
            swift::inspect_representation(&path, model_budget_for(representation), representation)?,
            representation,
        ),
        ParsedCommand::Mode {
            mode,
            path,
            tail,
            representation,
            multi_rank_adapter,
            row_workers,
        } => dispatch_mode(
            &mode,
            &path,
            &tail,
            representation,
            multi_rank_adapter.as_ref(),
            row_workers,
        ),
    }
}

fn dispatch_mode(
    mode: &str,
    path: &Path,
    tail: &[String],
    representation: Representation,
    multi_rank_adapter: Option<&MultiRankAdapterSelection>,
    row_workers: usize,
) -> Result<()> {
    if mode == "load" {
        if !tail.is_empty() {
            return Err(reject("load takes only the model path"));
        }
        let (graph, _) = load_graph_with_adapters(
            path,
            MatvecBackend::Auto,
            representation,
            multi_rank_adapter,
            row_workers,
        )?;
        return emit(
            &json!({"event":"graph-ready","identity":graph.identity(),"vocabulary":graph.vocab_size(),"initial_checkpoint_bytes":graph.checkpoint_bytes(),"maximum_live_state_payload_bytes":swift::STATE_PAYLOAD_BYTES,"full_attention_blocks":16,"recurrent_blocks":48,"row_workers":graph.row_workers(),"parallel_row_operations":graph.backend_statistics().parallel_row_operations}),
        );
    }
    match mode {
        "forward" => forward(path, tail, representation, multi_rank_adapter, row_workers),
        "trajectory" => trajectory(path, tail, representation, multi_rank_adapter, row_workers),
        "rollback" => rollback::run(path, tail, representation, multi_rank_adapter, row_workers),
        "request-timing" => {
            request_timing::run(path, tail, representation, multi_rank_adapter, row_workers)
        }
        "ffn-timing" => {
            ffn_timing::run(path, tail, representation, multi_rank_adapter, row_workers)
        }
        _ => Err(reject("unsupported evaluation mode")),
    }
}

fn select_row_workers(arguments: &mut Vec<String>) -> Result<Option<usize>> {
    if arguments
        .first()
        .is_none_or(|argument| argument != "--row-workers")
    {
        return Ok(None);
    }
    let value = arguments
        .get(1)
        .ok_or_else(|| reject("row workers require one bounded integer"))?;
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(reject("row workers require one bounded integer"));
    }
    let workers = value.parse::<usize>()?;
    if !(1..=MAX_ROW_WORKERS).contains(&workers) {
        return Err(reject("row workers must be from 1 through 18"));
    }
    arguments.drain(..2);
    if arguments
        .first()
        .is_some_and(|argument| argument == "--row-workers")
    {
        return Err(reject("duplicate row worker selection"));
    }
    Ok(Some(workers))
}

struct MultiRankAdapterSelection {
    path: PathBuf,
    scale: f32,
}

struct MultiRankMetadata {
    sha256: String,
    max_rank: usize,
    target_count: usize,
    scale: f32,
}

fn split_multi_rank_path_and_scale(raw: &str) -> Result<(PathBuf, f32)> {
    let (path_text, scale) = if let Some((prefix, suffix)) = raw.rsplit_once(':')
        && !suffix.contains(['\\', '/'])
    {
        if suffix.is_empty() {
            return Err(reject("multi-rank adapter scale after ':' cannot be empty"));
        }
        (prefix, suffix.parse::<f32>()?)
    } else {
        (raw, 0.0_f32)
    };
    let path = PathBuf::from(path_text);
    if !path.is_absolute()
        || path_text.contains(['%', '\0'])
        || path.components().any(|part| {
            matches!(
                part,
                std::path::Component::ParentDir | std::path::Component::CurDir
            )
        })
        || !scale.is_finite()
        || !(0.0..=2.0).contains(&scale)
    {
        return Err(reject(
            "multi-rank adapter requires an expanded absolute path and finite scale in 0..2",
        ));
    }
    Ok((path, scale))
}

fn select_multi_rank_adapter(
    arguments: &mut Vec<String>,
) -> Result<Option<MultiRankAdapterSelection>> {
    if arguments
        .first()
        .is_none_or(|argument| argument != "--multi-rank-adapter")
    {
        return Ok(None);
    }
    if arguments.len() < 2 {
        return Err(reject(
            "multi-rank adapter requires an absolute path and optional ':scale'",
        ));
    }
    let (path, scale) = split_multi_rank_path_and_scale(&arguments[1])?;
    arguments.drain(..2);
    if arguments
        .first()
        .is_some_and(|argument| argument == "--multi-rank-adapter")
    {
        return Err(reject("duplicate multi-rank adapter selection"));
    }
    Ok(Some(MultiRankAdapterSelection { path, scale }))
}

fn hash_adapter_file(path: &Path) -> Result<(u64, String)> {
    let mut file = std::fs::File::open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() == 0 {
        return Err(reject(
            "multi-rank adapter must be a non-empty regular file",
        ));
    }
    let mut hasher = Hasher::new()?;
    let mut buffer = vec![0_u8; 65_536];
    let mut read_bytes = 0_u64;
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read])?;
        read_bytes = read_bytes
            .checked_add(u64::try_from(read)?)
            .ok_or_else(|| reject("multi-rank adapter byte length overflow"))?;
    }
    if read_bytes != metadata.len() {
        return Err(reject(
            "multi-rank adapter byte length changed during hashing",
        ));
    }
    Ok((read_bytes, hex::encode(hasher.finish()?)))
}

fn select_representation(arguments: &mut Vec<String>) -> Result<Representation> {
    let mut selected = None;
    while let Some(first) = arguments.first().map(String::as_str) {
        let next = match first {
            "--nariq" => {
                let value = arguments
                    .get(1)
                    .map(String::as_str)
                    .ok_or_else(|| reject("--nariq requires 2.5bpw, 3.0bpw, or 2.0bpw"))?;
                let representation = match value {
                    "2.5bpw" => Representation::NariQ2_5Bpw,
                    "3.0bpw" => Representation::NariQ3_0Bpw,
                    "2.0bpw" => Representation::NariQ2_0Bpw,
                    _ => {
                        return Err(reject(
                            "unsupported --nariq variant; expected 2.5bpw, 3.0bpw, or 2.0bpw",
                        ));
                    }
                };
                arguments.drain(..2);
                representation
            }
            "--bonsai-2-5" => {
                let value = arguments
                    .get(1)
                    .map(String::as_str)
                    .ok_or_else(|| reject("--bonsai-2-5 requires pq2 or ptq1"))?;
                let representation = match value {
                    "pq2" => Representation::Bonsai25Pq2,
                    "ptq1" => Representation::Bonsai25Ptq1,
                    _ => {
                        return Err(reject(
                            "unsupported --bonsai-2-5 variant; expected pq2 or ptq1",
                        ));
                    }
                };
                arguments.drain(..2);
                representation
            }
            _ => break,
        };
        if selected.is_some() {
            return Err(reject(
                "conflicting or duplicate model representation prefix",
            ));
        }
        selected = Some(next);
    }
    Ok(selected.unwrap_or(Representation::Bonsai25Pq2))
}

fn evidence_backend(arguments: &[String]) -> Result<(&Path, MatvecBackend)> {
    let evidence = Path::new(&arguments[0]);
    if !evidence.is_absolute() || !evidence.is_dir() {
        return Err(reject("evidence directory must exist and be absolute"));
    }
    let backend = match arguments[1].as_str() {
        "scalar" => MatvecBackend::Scalar,
        "auto" => MatvecBackend::Auto,
        "neon" => MatvecBackend::ExperimentalNeon,
        _ => return Err(reject("invalid backend")),
    };
    Ok((evidence, backend))
}

fn valid_tokens(tokens: &[u32]) -> bool {
    !tokens.is_empty()
        && tokens
            .iter()
            .all(|&token| usize::try_from(token).is_ok_and(|id| id < swift::VOCAB_SIZE))
}

fn load_graph_with_adapters(
    path: &Path,
    backend: MatvecBackend,
    representation: Representation,
    multi_rank: Option<&MultiRankAdapterSelection>,
    row_workers: usize,
) -> Result<(Qwen35, Option<MultiRankMetadata>)> {
    // Reject unavailable worker counts before model I/O.
    let parallelism = RowParallelism::new(row_workers)?;
    let (model, receipt) =
        swift::load_representation(path, model_budget_for(representation), representation)?;
    report("weights-loaded", receipt, representation)?;
    let (multi_adapter, multi_metadata) = if let Some(selection) = multi_rank {
        let (bytes, sha256) = hash_adapter_file(&selection.path)?;
        let loaded = MultiRankAdapter::load_verified(
            &model,
            &selection.path,
            bytes,
            &sha256,
            512 * 1024 * 1024,
            MultiRankPolicy::NpuFriendly,
        )?;
        let metadata = MultiRankMetadata {
            sha256,
            max_rank: loaded.max_rank(),
            target_count: loaded.target_count(),
            scale: selection.scale,
        };
        emit(&json!({
            "event": "multi-rank-adapter-loaded",
            "weight_representation": representation.name(),
            "model_sha256": representation.model_sha256(),
            "multi_rank_adapter_sha256": metadata.sha256,
            "multi_rank_target_pairs": metadata.target_count,
            "multi_rank_max_rank": metadata.max_rank,
            "multi_rank_scale": metadata.scale,
            "multi_rank_scale_bits": format!("{:08x}", metadata.scale.to_bits()),
        }))?;
        (Some((loaded, selection.scale)), Some(metadata))
    } else {
        (None, None)
    };
    let mut graph = Qwen35::with_adapters_and_kv(
        model,
        swift::CONTEXT_TOKENS,
        1024 * 1024 * 1024,
        backend,
        multi_adapter,
        KvCachePolicy::F16,
    )?;
    graph.set_row_parallelism(parallelism);
    Ok((graph, multi_metadata))
}

fn forward(
    path: &Path,
    arguments: &[String],
    representation: Representation,
    multi_rank_adapter: Option<&MultiRankAdapterSelection>,
    row_workers: usize,
) -> Result<()> {
    if arguments.len() != 3 {
        return Err(reject(
            "forward requires output directory, backend and token IDs",
        ));
    }
    let (evidence, backend) = evidence_backend(arguments)?;
    let tokens = arguments[2]
        .split(',')
        .map(str::parse::<u32>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if !valid_tokens(&tokens) || tokens.len() > 16 {
        return Err(reject("forward requires 1..16 valid token IDs"));
    }
    let (mut graph, multi_metadata) = load_graph_with_adapters(
        path,
        backend,
        representation,
        multi_rank_adapter,
        row_workers,
    )?;
    let mut logits = Vec::new();
    let mut remaining = usize::MAX;
    for (position, token) in tokens.into_iter().enumerate() {
        let started = Instant::now();
        graph.evaluate_into(token, &mut logits)?;
        let evaluation_milliseconds = started.elapsed().as_secs_f64() * 1000.0;
        let output = snapshot(&graph, &logits, evidence, position, &mut remaining)?;
        let mut record = json!({
            "event": "forward-step",
            "position": position,
            "input_token": token,
            "greedy_token": greedy(&logits)?,
            "logits": output,
            "logit_count": logits.len(),
            "checkpoint_bytes": graph.checkpoint_bytes(),
            "identity": graph.identity(),
            "backend_requested": arguments[1],
            "row_workers": graph.row_workers(),
            "parallel_row_operations": graph.backend_statistics().parallel_row_operations,
            "evaluation_milliseconds": evaluation_milliseconds,
        });
        if let Some(meta) = &multi_metadata {
            record["multiRankAdapterSha256"] = json!(meta.sha256);
            record["multiRankAdapterScaleBits"] = json!(format!("{:08x}", meta.scale.to_bits()));
            record["multiRankAdapterMaxRank"] = json!(meta.max_rank);
        }
        emit(&record)?;
    }
    Ok(())
}

fn attach_trajectory_multi_rank(record: &mut Value, multi_metadata: Option<&MultiRankMetadata>) {
    if let Some(meta) = multi_metadata {
        record["multi_rank_adapter_sha256"] = json!(meta.sha256);
        record["multi_rank_max_rank"] = json!(meta.max_rank);
        record["multi_rank_scale_bits"] = json!(format!("{:08x}", meta.scale.to_bits()));
    }
}

fn trajectory(
    path: &Path,
    arguments: &[String],
    representation: Representation,
    multi_rank_adapter: Option<&MultiRankAdapterSelection>,
    row_workers: usize,
) -> Result<()> {
    if arguments.len() != 4 {
        return Err(reject(
            "trajectory requires output directory, backend, prompt-ID file and greedy steps",
        ));
    }
    let (evidence, backend) = evidence_backend(arguments)?;
    let steps = arguments[3].parse::<usize>()?;
    if !(1..=MAX_GREEDY_STEPS).contains(&steps) {
        return Err(reject("trajectory requires 1..16 greedy steps"));
    }
    let input = Path::new(&arguments[2]);
    if !input.is_absolute() {
        return Err(reject("trajectory requires an absolute prompt-ID file"));
    }
    let mut input = std::fs::File::open(input)?;
    if !input.metadata()?.is_file() {
        return Err(reject("trajectory requires a regular prompt-ID file"));
    }
    let prompt = read_tokens(&mut input, swift::CONTEXT_TOKENS - steps)?;
    if !valid_tokens(&prompt) {
        return Err(reject("trajectory requires nonempty valid Swift token IDs"));
    }
    let (&last, intermediate) = prompt
        .split_last()
        .ok_or_else(|| reject("trajectory requires a prompt"))?;
    emit(
        &json!({"event":"trajectory-start","prompt_token_ids":prompt,"prompt_tokens":prompt.len(),"greedy_steps":steps,"context_tokens":swift::CONTEXT_TOKENS,"maximum_snapshot_payload_bytes":MAX_TRAJECTORY_EVIDENCE_BYTES,"snapshot_count":steps+1,"stop_policy":"fixed-step raw greedy; special IDs are included"}),
    )?;
    let (mut graph, multi_metadata) = load_graph_with_adapters(
        path,
        backend,
        representation,
        multi_rank_adapter,
        row_workers,
    )?;
    for &token in intermediate {
        graph.ingest(token)?;
    }
    let mut logits = Vec::new();
    let mut remaining = MAX_TRAJECTORY_EVIDENCE_BYTES;
    let mut generated = Vec::with_capacity(steps);
    let mut token = last;
    for step in 0..=steps {
        let started = Instant::now();
        graph.evaluate_into(token, &mut logits)?;
        let evaluation_milliseconds = started.elapsed().as_secs_f64() * 1000.0;
        let position = prompt.len() - 1 + step;
        let output = snapshot(&graph, &logits, evidence, position, &mut remaining)?;
        let next = greedy(&logits)?;
        let mut step_record = json!({"event":"trajectory-step","step":step,"position":position,"kind":if step==0 {"prefix"} else {"generated"},"input_token":token,"greedy_token":next,"logits":output,"logit_count":logits.len(),"checkpoint_bytes":graph.checkpoint_bytes(),"identity":graph.identity(),"backend_requested":arguments[1],"snapshot_payload_bytes_written":MAX_TRAJECTORY_EVIDENCE_BYTES-remaining,"row_workers":graph.row_workers(),"parallel_row_operations":graph.backend_statistics().parallel_row_operations,"evaluation_milliseconds":evaluation_milliseconds});
        attach_trajectory_multi_rank(&mut step_record, multi_metadata.as_ref());
        emit(&step_record)?;
        if step < steps {
            generated.push(next);
            token = next;
        }
    }
    let mut complete_record = json!({"event":"trajectory-complete","prompt_token_ids":prompt,"generated_token_ids":generated,"prompt_tokens":prompt.len(),"generated_tokens":generated.len(),"snapshot_payload_bytes_written":MAX_TRAJECTORY_EVIDENCE_BYTES-remaining});
    attach_trajectory_multi_rank(&mut complete_record, multi_metadata.as_ref());
    emit(&complete_record)
}

fn hash_f32_le(values: &[f32]) -> Result<(usize, String)> {
    let bytes = values
        .len()
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| reject("logit snapshot byte size overflow"))?;
    let mut hash = Hasher::new()?;
    for value in values {
        hash.update(&value.to_le_bytes())?;
    }
    let sha256 = hex::encode(hash.finish()?);
    Ok((bytes, sha256))
}

fn snapshot_artifact_metadata(position: usize, logits: &[f32]) -> Result<Value> {
    let (bytes, sha256) = hash_f32_le(logits)?;
    Ok(json!({
        "role": "logits-snapshot",
        "name": format!("logits-{position}.f32le"),
        "bytes": bytes,
        "sha256": sha256,
    }))
}

fn snapshot(
    graph: &Qwen35,
    logits: &[f32],
    evidence: &Path,
    position: usize,
    remaining: &mut usize,
) -> Result<Value> {
    let output = evidence.join(format!("logits-{position}.f32le"));
    write_values(&output, logits, remaining)?;
    let states = evidence.join(format!("state-{position}"));
    std::fs::create_dir(&states)?;
    dump_state(graph, &states, remaining)?;
    snapshot_artifact_metadata(position, logits)
}

fn write_values(path: &Path, values: &[f32], remaining: &mut usize) -> Result<()> {
    let bytes = values
        .len()
        .checked_mul(size_of::<f32>())
        .filter(|&bytes| bytes <= *remaining)
        .ok_or_else(|| reject("snapshot payload exceeds the evidence byte allowance"))?;
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    *remaining -= bytes;
    let mut file = std::io::BufWriter::with_capacity(65_536, file);
    for value in values {
        file.write_all(&value.to_le_bytes())?;
    }
    file.flush()?;
    Ok(())
}

fn dump_state(graph: &Qwen35, states: &Path, remaining: &mut usize) -> Result<()> {
    graph.visit_state(&mut |block, component, values| {
        let label = match component {
            StateComponent::Key(token) => format!("key-{token}"),
            StateComponent::Value(token) => format!("value-{token}"),
            StateComponent::Convolution => "convolution".to_owned(),
            StateComponent::Recurrent => "recurrent".to_owned(),
        };
        write_values(
            &states.join(format!("block-{block}-{label}.f32le")),
            values,
            remaining,
        )
    })
}

fn report(event: &str, receipt: swift::LoadReceipt, representation: Representation) -> Result<()> {
    let record = json!({"event":event,"weight_representation":representation.name(),"model_sha256":representation.model_sha256(),"model_bytes":representation.model_bytes(),"header_charge_bytes":receipt.header_charge_bytes,"tensor_payload_bytes":receipt.tensor_payload_bytes,"retained_storage_bytes":receipt.retained_storage_bytes,"largest_tensor_conversion_copy_bytes":receipt.largest_tensor_bytes,"tensor_counts_f32_bf16_packed":receipt.tensor_counts,"model_storage_budget_bytes":model_budget_for(representation),"memory_claim":"retained storage accounting; process working set and peak commit are separate"});
    emit(&record)
}

#[cfg(test)]
mod tests {
    use super::{
        MultiRankMetadata, ParsedCommand, Representation, Result, attach_trajectory_multi_rank,
        model_budget_for, parse_cli, select_multi_rank_adapter, select_representation,
        select_row_workers, write_values,
    };

    #[test]
    fn row_worker_prefix_is_bounded_and_composes_with_exact_profile_selection() -> Result<()> {
        let mut default = vec!["forward".into(), "model.gguf".into()];
        assert_eq!(select_row_workers(&mut default)?, None);
        assert_eq!(default, ["forward", "model.gguf"]);
        let path = std::env::temp_dir().join("daisyrose.gguf");
        for ptq1 in [false, true] {
            for workers in [1_usize, 2, 16, 18] {
                let mut arguments = Vec::new();
                if ptq1 {
                    arguments.extend(["--bonsai-2-5".into(), "ptq1".into()]);
                }
                arguments.extend([
                    "--row-workers".into(),
                    workers.to_string(),
                    "--multi-rank-adapter".into(),
                    format!("{}:1.3", path.to_string_lossy()),
                    "forward".into(),
                ]);
                assert_eq!(
                    select_representation(&mut arguments)?,
                    if ptq1 {
                        Representation::Bonsai25Ptq1
                    } else {
                        Representation::Bonsai25Pq2
                    }
                );
                assert_eq!(select_row_workers(&mut arguments)?, Some(workers));
                let adapter =
                    select_multi_rank_adapter(&mut arguments)?.ok_or("missing multi-rank")?;
                assert_eq!(adapter.path, path);
                assert_eq!(adapter.scale.to_bits(), 1.3_f32.to_bits());
                assert_eq!(arguments, ["forward"]);
            }
        }
        for value in ["", "0", "19", "+2", "-1", "2.0", "184467440737095516160"] {
            let mut arguments = vec!["--row-workers".into(), value.into(), "forward".into()];
            assert!(select_row_workers(&mut arguments).is_err());
        }
        let mut missing = vec!["--row-workers".into()];
        assert!(select_row_workers(&mut missing).is_err());
        let mut duplicate = vec![
            "--row-workers".into(),
            "2".into(),
            "--row-workers".into(),
            "1".into(),
        ];
        assert!(select_row_workers(&mut duplicate).is_err());
        Ok(())
    }

    #[cfg(not(feature = "parallel-rows"))]
    #[test]
    fn row_worker_feature_rejection_precedes_model_file_access() {
        let rejected = super::load_graph_with_adapters(
            std::path::Path::new("missing-relative-model.gguf"),
            super::MatvecBackend::Auto,
            Representation::Bonsai25Pq2,
            None,
            2,
        );
        assert!(
            matches!(rejected, Err(error) if error.to_string() == "multiple row workers require parallel-rows")
        );
    }

    #[test]
    fn ptq1_correctness_capture_requires_explicit_prefix_and_keeps_source_default() -> Result<()> {
        let mut source = vec!["inspect".to_owned(), "model.gguf".to_owned()];
        assert_eq!(
            select_representation(&mut source)?,
            Representation::Bonsai25Pq2
        );
        assert_eq!(source, ["inspect", "model.gguf"]);

        let mut exported = vec![
            "--bonsai-2-5".to_owned(),
            "ptq1".to_owned(),
            "forward".to_owned(),
            "export.gguf".to_owned(),
        ];
        assert_eq!(
            select_representation(&mut exported)?,
            Representation::Bonsai25Ptq1
        );
        assert_eq!(exported, ["forward", "export.gguf"]);

        let mut misplaced = vec![
            "inspect".to_owned(),
            "--bonsai-2-5".to_owned(),
            "ptq1".to_owned(),
        ];
        assert_eq!(
            select_representation(&mut misplaced)?,
            Representation::Bonsai25Pq2
        );
        assert_eq!(misplaced, ["inspect", "--bonsai-2-5", "ptq1"]);
        Ok(())
    }

    #[test]
    fn snapshot_output_preserves_existing_evidence_and_enforces_payload_budget() -> Result<()> {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(std::io::Error::other)?
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "PrismSwiftQualifier-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&directory)?;
        let output = directory.join("values.f32le");
        let values = [1.0, -0.5];
        let mut remaining = 7;
        assert!(write_values(&output, &values, &mut remaining).is_err());
        assert!(!output.exists());
        assert_eq!(remaining, 7);

        remaining = 8;
        write_values(&output, &values, &mut remaining)?;
        assert_eq!(remaining, 0);
        let expected = [0, 0, 128, 63, 0, 0, 0, 191];
        assert_eq!(std::fs::read(&output)?, expected);

        remaining = 16;
        assert!(write_values(&output, &[2.0], &mut remaining).is_err());
        assert_eq!(remaining, 16);
        assert_eq!(std::fs::read(&output)?, expected);
        std::fs::remove_file(output)?;
        std::fs::remove_dir(directory)?;
        Ok(())
    }

    #[test]
    fn snapshot_artifact_metadata_omits_paths_and_records_sha256() -> Result<()> {
        let values = [1.0_f32, -0.5_f32];
        let metadata = super::snapshot_artifact_metadata(7, &values)?;
        assert_eq!(metadata["role"], "logits-snapshot");
        assert_eq!(metadata["name"], "logits-7.f32le");
        assert_eq!(metadata["bytes"], 8);
        let expected_bytes = [0_u8, 0, 128, 63, 0, 0, 0, 191];
        let expected_sha256 = hex::encode(bonsai_native_hash::hash_bytes(&expected_bytes)?);
        assert_eq!(metadata["sha256"], expected_sha256);
        let serialized = serde_json::to_string(&metadata)?;
        assert!(!serialized.contains('\\'));
        assert!(!serialized.contains('/'));
        assert!(!serialized.contains("C:"));
        Ok(())
    }

    #[test]
    fn multi_rank_adapter_selection_parses_optional_scale_and_drive_letters() -> Result<()> {
        let path = std::env::temp_dir().join("daisyrose-lora.gguf");
        let path_str = path.to_string_lossy().into_owned();

        let mut default_args = vec![
            "--multi-rank-adapter".to_owned(),
            path_str.clone(),
            "forward".to_owned(),
        ];
        let default_sel =
            select_multi_rank_adapter(&mut default_args)?.ok_or("missing default multi-rank")?;
        assert_eq!(default_sel.path, path);
        assert_eq!(default_sel.scale.to_bits(), 0.0_f32.to_bits());
        assert_eq!(default_args, ["forward"]);

        let mut scaled_args = vec![
            "--multi-rank-adapter".to_owned(),
            format!("{path_str}:0.5"),
            "rollback".to_owned(),
        ];
        let scaled_sel =
            select_multi_rank_adapter(&mut scaled_args)?.ok_or("missing scaled multi-rank")?;
        assert_eq!(scaled_sel.path, path);
        assert_eq!(scaled_sel.scale.to_bits(), 0.5_f32.to_bits());
        assert_eq!(scaled_args, ["rollback"]);

        for bad in [
            "relative.gguf",
            "relative.gguf:0.5",
            &format!("{path_str}:"),
            &format!("{path_str}:NaN"),
            &format!("{path_str}:-0.1"),
            &format!("{path_str}:2.1"),
            "%TEMP%\\daisyrose.gguf:0.5",
        ] {
            let mut bad_args = vec!["--multi-rank-adapter".to_owned(), bad.to_owned()];
            assert!(
                select_multi_rank_adapter(&mut bad_args).is_err(),
                "expected rejection for {bad}"
            );
        }

        let mut duplicate = vec![
            "--multi-rank-adapter".to_owned(),
            path_str.clone(),
            "--multi-rank-adapter".to_owned(),
            path_str,
        ];
        assert!(select_multi_rank_adapter(&mut duplicate).is_err());
        Ok(())
    }

    #[test]
    fn nariq_representation_parsing_and_budget() -> Result<()> {
        assert_eq!(
            model_budget_for(Representation::Bonsai25Pq2),
            8 * 1024 * 1024 * 1024
        );
        assert_eq!(
            model_budget_for(Representation::Bonsai25Ptq1),
            8 * 1024 * 1024 * 1024
        );
        for (flag, expected) in [
            ("2.5bpw", Representation::NariQ2_5Bpw),
            ("3.0bpw", Representation::NariQ3_0Bpw),
            ("2.0bpw", Representation::NariQ2_0Bpw),
        ] {
            let mut args = vec!["--nariq".to_owned(), flag.to_owned(), "load".to_owned()];
            assert_eq!(select_representation(&mut args)?, expected);
            assert_eq!(args, ["load"]);
            assert_eq!(model_budget_for(expected), 12 * 1024 * 1024 * 1024);
        }

        for bad in [
            vec!["--nariq".to_owned()],
            vec!["--nariq".to_owned(), "4.0bpw".to_owned()],
            vec![
                "--nariq".to_owned(),
                "2.5bpw".to_owned(),
                "--bonsai-2-5".to_owned(),
                "ptq1".to_owned(),
            ],
            vec![
                "--bonsai-2-5".to_owned(),
                "ptq1".to_owned(),
                "--nariq".to_owned(),
                "3.0bpw".to_owned(),
            ],
            vec![
                "--nariq".to_owned(),
                "2.5bpw".to_owned(),
                "--nariq".to_owned(),
                "2.0bpw".to_owned(),
            ],
        ] {
            let mut args = bad;
            assert!(select_representation(&mut args).is_err());
        }
        Ok(())
    }

    #[test]
    fn nariq_load_admits_multi_rank_adapter() -> Result<()> {
        let model = std::env::temp_dir()
            .join("nariq-3.0bpw.gguf")
            .to_string_lossy()
            .into_owned();
        let multi_rank = std::env::temp_dir()
            .join("daisyrose.gguf")
            .to_string_lossy()
            .into_owned();

        let parsed = parse_cli(vec![
            "--nariq".into(),
            "3.0bpw".into(),
            "--multi-rank-adapter".into(),
            format!("{multi_rank}:0.5"),
            "load".into(),
            model,
        ])?;
        match parsed {
            ParsedCommand::Mode {
                mode,
                representation,
                multi_rank_adapter,
                ..
            } => {
                assert_eq!(mode, "load");
                assert_eq!(representation, Representation::NariQ3_0Bpw);
                let mr = multi_rank_adapter.ok_or("missing multi-rank adapter")?;
                assert_eq!(mr.scale.to_bits(), 0.5_f32.to_bits());
            }
            _ => return Err(super::reject("expected parsed load command")),
        }
        Ok(())
    }

    #[test]
    fn trajectory_and_request_timing_parse_nariq_and_multi_rank_adapter() -> Result<()> {
        let base = std::env::temp_dir();
        let model = base.join("model.gguf").to_string_lossy().into_owned();
        let evidence = base.join("evidence").to_string_lossy().into_owned();
        let input = base.join("input.json").to_string_lossy().into_owned();
        let mr = base.join("daisyrose.gguf").to_string_lossy().into_owned();
        let commit = "0".repeat(40);

        let traj = parse_cli(vec![
            "--nariq".into(),
            "2.5bpw".into(),
            "--row-workers".into(),
            "4".into(),
            "--multi-rank-adapter".into(),
            format!("{mr}:0.5"),
            "trajectory".into(),
            model.clone(),
            evidence.clone(),
            "auto".into(),
            input.clone(),
            "4".into(),
        ])?;
        match traj {
            ParsedCommand::Mode {
                mode,
                tail,
                representation,
                multi_rank_adapter,
                row_workers,
                ..
            } => {
                assert_eq!(mode, "trajectory");
                assert_eq!(tail.len(), 4);
                assert_eq!(representation, Representation::NariQ2_5Bpw);
                assert_eq!(row_workers, 4);
                assert_eq!(
                    multi_rank_adapter.ok_or("missing mr")?.scale.to_bits(),
                    0.5_f32.to_bits()
                );
            }
            _ => return Err(super::reject("expected trajectory command")),
        }

        let req = parse_cli(vec![
            "--nariq".into(),
            "2.0bpw".into(),
            "--multi-rank-adapter".into(),
            mr,
            "request-timing".into(),
            model,
            evidence,
            "neon".into(),
            input,
            "8".into(),
            "1700000000000".into(),
            commit,
        ])?;
        match req {
            ParsedCommand::Mode {
                mode,
                tail,
                representation,
                multi_rank_adapter,
                ..
            } => {
                assert_eq!(mode, "request-timing");
                assert_eq!(tail.len(), 6);
                assert_eq!(representation, Representation::NariQ2_0Bpw);
                assert_eq!(
                    multi_rank_adapter.ok_or("missing mr")?.scale.to_bits(),
                    0.0_f32.to_bits()
                );
            }
            _ => return Err(super::reject("expected request-timing command")),
        }
        Ok(())
    }

    #[test]
    fn ffn_timing_and_trajectory_receipt_support_multi_rank_adapter() -> Result<()> {
        let base = std::env::temp_dir();
        let model = base.join("model.gguf").to_string_lossy().into_owned();
        let evidence = base.join("evidence").to_string_lossy().into_owned();
        let input = base.join("input.json").to_string_lossy().into_owned();
        let mr = base.join("daisyrose.gguf").to_string_lossy().into_owned();
        let commit = "0".repeat(40);

        let ffn = parse_cli(vec![
            "--bonsai-2-5".into(),
            "ptq1".into(),
            "--row-workers".into(),
            "2".into(),
            "--multi-rank-adapter".into(),
            format!("{mr}:1.0"),
            "ffn-timing".into(),
            model.clone(),
            evidence.clone(),
            "scalar".into(),
            input,
            "1700000000000".into(),
            commit,
        ])?;
        match ffn {
            ParsedCommand::Mode {
                mode,
                tail,
                representation,
                multi_rank_adapter,
                row_workers,
                ..
            } => {
                assert_eq!(mode, "ffn-timing");
                assert_eq!(tail.len(), 5);
                assert_eq!(representation, Representation::Bonsai25Ptq1);
                assert_eq!(row_workers, 2);
                assert_eq!(
                    multi_rank_adapter.ok_or("missing mr")?.scale.to_bits(),
                    1.0_f32.to_bits()
                );
            }
            _ => return Err(super::reject("expected ffn-timing command")),
        }

        // Verify trajectory receipt multi-rank attachment helper.
        let mut receipt = serde_json::json!({"event": "trajectory-complete"});
        attach_trajectory_multi_rank(
            &mut receipt,
            Some(&MultiRankMetadata {
                sha256: "c".repeat(64),
                max_rank: 6,
                target_count: 12,
                scale: 0.5,
            }),
        );
        assert_eq!(receipt["multi_rank_adapter_sha256"], "c".repeat(64));
        assert_eq!(receipt["multi_rank_max_rank"], 6);
        assert_eq!(
            receipt["multi_rank_scale_bits"],
            format!("{:08x}", 0.5_f32.to_bits())
        );

        // Missing tail argument is rejected for timing/trajectory modes.
        assert!(
            parse_cli(vec![
                "--nariq".into(),
                "2.5bpw".into(),
                "trajectory".into(),
                model.clone(),
                evidence.clone(),
                "auto".into(),
            ])
            .is_err()
        );
        assert!(
            parse_cli(vec![
                "request-timing".into(),
                model.clone(),
                evidence.clone(),
                "auto".into(),
            ])
            .is_err()
        );
        assert!(parse_cli(vec!["ffn-timing".into(), model, evidence, "auto".into()]).is_err());
        Ok(())
    }

    #[test]
    fn bonsai25_cli_supports_multi_rank_and_rejects_conflicts() -> Result<()> {
        for (flag, expected_repr) in [
            ("pq2", Representation::Bonsai25Pq2),
            ("ptq1", Representation::Bonsai25Ptq1),
        ] {
            let mut args = vec![
                "--bonsai-2-5".to_owned(),
                flag.to_owned(),
                "load".to_owned(),
            ];
            assert_eq!(select_representation(&mut args)?, expected_repr);
            assert_eq!(args, ["load"]);
            assert_eq!(model_budget_for(expected_repr), 8 * 1024 * 1024 * 1024);

            let mr_path = std::env::temp_dir()
                .join("daisyrose.gguf")
                .to_string_lossy()
                .into_owned();
            let parsed = parse_cli(vec![
                "--bonsai-2-5".into(),
                flag.into(),
                "--multi-rank-adapter".into(),
                format!("{mr_path}:0.5"),
                "load".into(),
                std::env::temp_dir()
                    .join("bonsai25.gguf")
                    .to_string_lossy()
                    .into_owned(),
            ])?;
            match parsed {
                ParsedCommand::Mode {
                    representation,
                    multi_rank_adapter,
                    ..
                } => {
                    assert_eq!(representation, expected_repr);
                    assert!(multi_rank_adapter.is_some());
                }
                _ => return Err(super::reject("expected Bonsai 2.5 parsed load command")),
            }
        }
        for bad in [
            vec!["--bonsai-2-5".to_owned()],
            vec!["--bonsai-2-5".to_owned(), "q4_0".to_owned()],
            vec![
                "--bonsai-2-5".to_owned(),
                "pq2".to_owned(),
                "--nariq".to_owned(),
                "2.5bpw".to_owned(),
            ],
            vec![
                "--bonsai-2-5".to_owned(),
                "pq2".to_owned(),
                "--bonsai-2-5".to_owned(),
                "ptq1".to_owned(),
            ],
        ] {
            let mut args = bad;
            assert!(select_representation(&mut args).is_err());
        }
        Ok(())
    }
}
