#![forbid(unsafe_code)]
//! Bounded, explicit actual-model load and forward correctness evidence.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use bonsai_prism_runtime::graph::{Qwen35, StateComponent};
use bonsai_prism_runtime::kernels::MatvecBackend;
use bonsai_prism_runtime::model::swift::{self, Representation};
use bonsai_prism_runtime::runtime::greedy;
use bonsai_prism_runtime::token_input::read_tokens;
use bonsai_prism_runtime::{Decoder, Result};
use serde_json::json;

const MODEL_BUDGET: usize = 8 * 1024 * 1024 * 1024;
const MAX_GREEDY_STEPS: usize = 16;
const MAX_TRAJECTORY_EVIDENCE_BYTES: usize = 8 * 1024 * 1024 * 1024;
const USAGE: &str = concat!(
    "Usage: swift_model_check <inspect|load> <absolute-model>\n",
    "       swift_model_check forward <absolute-model> <absolute-evidence-directory> <scalar|auto|neon> <id,id,...>\n",
    "       swift_model_check trajectory <absolute-model> <absolute-evidence-directory> <scalar|auto|neon> <absolute-prompt-ids.json> <greedy-steps>\n",
    "Prefix any mode with --use-repacked-ptq1 to select only the verified current-weight PTQ1 export. PQ2 remains the default.\n",
    "forward permits 1..16 exact supplied token IDs and retains its per-token snapshots.\n",
    "trajectory reads a bounded JSON u32 array, reserving all 1..16 greedy steps in the 8192-token context.\n",
    "It snapshots the final prompt token and each fed-back greedy token, including special IDs; model stops do not end this fixed-step diagnostic.\n",
    "Evidence directories must already exist. Files are create-new; trajectory snapshot payload is capped at 8 GiB.\n",
    "This is correctness evidence, without timing, sampling or service qualification.\n",
);

fn reject(message: &str) -> bonsai_prism_runtime::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message).into()
}

fn emit(record: &serde_json::Value) -> Result<()> {
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

fn run() -> Result<()> {
    let mut arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let representation = select_representation(&mut arguments);
    if arguments.len() == 1 && arguments[0] == "--help" {
        std::io::stdout().lock().write_all(USAGE.as_bytes())?;
        return Ok(());
    }
    if !(2..=6).contains(&arguments.len()) {
        return Err(reject(USAGE));
    }
    let mode = &arguments[0];
    let path = PathBuf::from(&arguments[1]);
    if !path.is_absolute()
        || !matches!(mode.as_str(), "inspect" | "load" | "forward" | "trajectory")
    {
        return Err(reject(
            "an explicit mode and absolute model path are required",
        ));
    }
    if mode == "inspect" {
        if arguments.len() != 2 {
            return Err(reject("inspect takes only the model path"));
        }
        return report(
            "header-verified",
            swift::inspect_representation(&path, MODEL_BUDGET, representation)?,
            representation,
        );
    }
    if mode == "load" {
        if arguments.len() != 2 {
            return Err(reject("load takes only the model path"));
        }
        let (model, receipt) = swift::load_representation(&path, MODEL_BUDGET, representation)?;
        report("weights-loaded", receipt, representation)?;
        let graph = Qwen35::with_state_budget(
            model,
            swift::CONTEXT_TOKENS,
            1024 * 1024 * 1024,
            MatvecBackend::Auto,
        )?;
        return emit(
            &json!({"event":"graph-ready","identity":graph.identity(),"vocabulary":graph.vocab_size(),"initial_checkpoint_bytes":graph.checkpoint_bytes(),"maximum_live_state_payload_bytes":swift::STATE_PAYLOAD_BYTES,"full_attention_blocks":16,"recurrent_blocks":48}),
        );
    }
    match mode.as_str() {
        "forward" => forward(&path, &arguments[2..], representation),
        "trajectory" => trajectory(&path, &arguments[2..], representation),
        _ => Err(reject("unsupported evaluation mode")),
    }
}

fn select_representation(arguments: &mut Vec<String>) -> Representation {
    if arguments
        .first()
        .is_some_and(|argument| argument == "--use-repacked-ptq1")
    {
        arguments.remove(0);
        Representation::CurrentRepackedPtq1
    } else {
        Representation::Pq2
    }
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

fn load_graph(
    path: &Path,
    backend: MatvecBackend,
    representation: Representation,
) -> Result<Qwen35> {
    let (model, receipt) = swift::load_representation(path, MODEL_BUDGET, representation)?;
    report("weights-loaded", receipt, representation)?;
    Qwen35::with_state_budget(model, swift::CONTEXT_TOKENS, 1024 * 1024 * 1024, backend)
}

fn forward(path: &Path, arguments: &[String], representation: Representation) -> Result<()> {
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
    let mut graph = load_graph(path, backend, representation)?;
    let mut logits = Vec::new();
    let mut remaining = usize::MAX;
    for (position, token) in tokens.into_iter().enumerate() {
        graph.evaluate_into(token, &mut logits)?;
        let output = snapshot(&graph, &logits, evidence, position, &mut remaining)?;
        emit(
            &json!({"event":"forward-step","position":position,"input_token":token,"greedy_token":greedy(&logits)?,"logits":output,"logit_count":logits.len(),"checkpoint_bytes":graph.checkpoint_bytes(),"identity":graph.identity(),"backend_requested":arguments[1]}),
        )?;
    }
    Ok(())
}

fn trajectory(path: &Path, arguments: &[String], representation: Representation) -> Result<()> {
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
    let mut graph = load_graph(path, backend, representation)?;
    for &token in intermediate {
        graph.ingest(token)?;
    }
    let mut logits = Vec::new();
    let mut remaining = MAX_TRAJECTORY_EVIDENCE_BYTES;
    let mut generated = Vec::with_capacity(steps);
    let mut token = last;
    for step in 0..=steps {
        graph.evaluate_into(token, &mut logits)?;
        let position = prompt.len() - 1 + step;
        let output = snapshot(&graph, &logits, evidence, position, &mut remaining)?;
        let next = greedy(&logits)?;
        emit(
            &json!({"event":"trajectory-step","step":step,"position":position,"kind":if step==0 {"prefix"} else {"generated"},"input_token":token,"greedy_token":next,"logits":output,"logit_count":logits.len(),"checkpoint_bytes":graph.checkpoint_bytes(),"identity":graph.identity(),"backend_requested":arguments[1],"snapshot_payload_bytes_written":MAX_TRAJECTORY_EVIDENCE_BYTES-remaining}),
        )?;
        if step < steps {
            generated.push(next);
            token = next;
        }
    }
    emit(
        &json!({"event":"trajectory-complete","prompt_token_ids":prompt,"generated_token_ids":generated,"prompt_tokens":prompt.len(),"generated_tokens":generated.len(),"snapshot_payload_bytes_written":MAX_TRAJECTORY_EVIDENCE_BYTES-remaining}),
    )
}

fn snapshot(
    graph: &Qwen35,
    logits: &[f32],
    evidence: &Path,
    position: usize,
    remaining: &mut usize,
) -> Result<PathBuf> {
    let output = evidence.join(format!("logits-{position}.f32le"));
    write_values(&output, logits, remaining)?;
    let states = evidence.join(format!("state-{position}"));
    std::fs::create_dir(&states)?;
    dump_state(graph, &states, remaining)?;
    Ok(output)
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
    let mut record = json!({"event":event,"weight_representation":representation.name(),"model_sha256":representation.model_sha256(),"model_bytes":representation.model_bytes(),"header_charge_bytes":receipt.header_charge_bytes,"tensor_payload_bytes":receipt.tensor_payload_bytes,"retained_storage_bytes":receipt.retained_storage_bytes,"largest_tensor_conversion_copy_bytes":receipt.largest_tensor_bytes,"tensor_counts_f32_bf16_packed":receipt.tensor_counts,"model_storage_budget_bytes":MODEL_BUDGET,"memory_claim":"retained storage accounting; process working set and peak commit are separate"});
    // Retain the original PQ2 evidence key without falsely naming PTQ1 counts PQ2.
    if representation == Representation::Pq2 {
        record["tensor_counts_f32_bf16_pq2"] = json!(receipt.tensor_counts);
    }
    emit(&record)
}

#[cfg(test)]
mod tests {
    use super::{Representation, Result, select_representation, write_values};

    #[test]
    fn ptq1_correctness_capture_requires_explicit_prefix_and_keeps_source_default() {
        let mut source = vec!["inspect".to_owned(), "model.gguf".to_owned()];
        assert_eq!(select_representation(&mut source), Representation::Pq2);
        assert_eq!(source, ["inspect", "model.gguf"]);

        let mut exported = vec![
            "--use-repacked-ptq1".to_owned(),
            "forward".to_owned(),
            "export.gguf".to_owned(),
        ];
        assert_eq!(
            select_representation(&mut exported),
            Representation::CurrentRepackedPtq1
        );
        assert_eq!(exported, ["forward", "export.gguf"]);

        let mut misplaced = vec!["inspect".to_owned(), "--use-repacked-ptq1".to_owned()];
        assert_eq!(select_representation(&mut misplaced), Representation::Pq2);
        assert_eq!(misplaced, ["inspect", "--use-repacked-ptq1"]);
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
}
