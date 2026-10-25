#![forbid(unsafe_code)]
//! Bounded, explicit actual-model load and forward correctness evidence.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use bonsai_prism_runtime::graph::{Qwen35, StateComponent};
use bonsai_prism_runtime::kernels::MatvecBackend;
use bonsai_prism_runtime::model::swift;
use bonsai_prism_runtime::runtime::greedy;
use bonsai_prism_runtime::{Decoder, Result};
use serde_json::json;

const MODEL_BUDGET: usize = 8 * 1024 * 1024 * 1024;

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
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    if !(2..=5).contains(&arguments.len()) {
        return Err(reject(
            "Usage: swift_model_check <inspect|load|forward> <absolute-model> [absolute-evidence-directory scalar|auto|neon id,id,...]; forward permits 1..16 exact tokens",
        ));
    }
    let mode = &arguments[0];
    let path = PathBuf::from(&arguments[1]);
    if !path.is_absolute() || !matches!(mode.as_str(), "inspect" | "load" | "forward") {
        return Err(reject(
            "an explicit mode and absolute model path are required",
        ));
    }
    if mode == "inspect" {
        if arguments.len() != 2 {
            return Err(reject("inspect takes only the model path"));
        }
        return report("header-verified", swift::inspect(&path, MODEL_BUDGET)?);
    }
    if mode == "load" {
        if arguments.len() != 2 {
            return Err(reject("load takes only the model path"));
        }
        let (model, receipt) = swift::load(&path, MODEL_BUDGET)?;
        report("weights-loaded", receipt)?;
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
    if arguments.len() != 5 {
        return Err(reject(
            "forward requires output directory, backend and token IDs",
        ));
    }
    let evidence = Path::new(&arguments[2]);
    if !evidence.is_absolute() || !evidence.is_dir() {
        return Err(reject("evidence directory must exist and be absolute"));
    }
    let backend = match arguments[3].as_str() {
        "scalar" => MatvecBackend::Scalar,
        "auto" => MatvecBackend::Auto,
        "neon" => MatvecBackend::ExperimentalNeon,
        _ => return Err(reject("invalid backend")),
    };
    let tokens = arguments[4]
        .split(',')
        .map(str::parse::<u32>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if tokens.is_empty()
        || tokens.len() > 16
        || tokens
            .iter()
            .any(|&token| usize::try_from(token).map_or(true, |id| id >= swift::VOCAB_SIZE))
    {
        return Err(reject("forward requires 1..16 valid token IDs"));
    }
    let (model, receipt) = swift::load(&path, MODEL_BUDGET)?;
    report("weights-loaded", receipt)?;
    let mut graph =
        Qwen35::with_state_budget(model, swift::CONTEXT_TOKENS, 1024 * 1024 * 1024, backend)?;
    let mut logits = Vec::new();
    for (position, token) in tokens.into_iter().enumerate() {
        graph.evaluate_into(token, &mut logits)?;
        let output = evidence.join(format!("logits-{position}.f32le"));
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&output)?;
        let mut file = std::io::BufWriter::with_capacity(65_536, file);
        for value in &logits {
            file.write_all(&value.to_le_bytes())?;
        }
        file.flush()?;
        let states = evidence.join(format!("state-{position}"));
        std::fs::create_dir(&states)?;
        dump_state(&graph, &states)?;
        emit(
            &json!({"event":"forward-step","position":position,"input_token":token,"greedy_token":greedy(&logits)?,"logits":output,"logit_count":logits.len(),"checkpoint_bytes":graph.checkpoint_bytes(),"identity":graph.identity(),"backend_requested":arguments[3]}),
        )?;
    }
    Ok(())
}

fn dump_state(graph: &Qwen35, states: &Path) -> Result<()> {
    graph.visit_state(&mut |block, component, values| {
        let label = match component {
            StateComponent::Key(token) => format!("key-{token}"),
            StateComponent::Value(token) => format!("value-{token}"),
            StateComponent::Convolution => "convolution".to_owned(),
            StateComponent::Recurrent => "recurrent".to_owned(),
        };
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(states.join(format!("block-{block}-{label}.f32le")))?;
        let mut file = std::io::BufWriter::with_capacity(65_536, file);
        for value in values {
            file.write_all(&value.to_le_bytes())?;
        }
        file.flush()?;
        Ok(())
    })
}

fn report(event: &str, receipt: swift::LoadReceipt) -> Result<()> {
    emit(
        &json!({"event":event,"model_sha256":swift::MODEL_SHA256,"model_bytes":swift::MODEL_BYTES,"header_charge_bytes":receipt.header_charge_bytes,"tensor_payload_bytes":receipt.tensor_payload_bytes,"retained_storage_bytes":receipt.retained_storage_bytes,"largest_tensor_conversion_copy_bytes":receipt.largest_tensor_bytes,"tensor_counts_f32_bf16_pq2":receipt.tensor_counts,"model_storage_budget_bytes":MODEL_BUDGET,"memory_claim":"retained storage accounting; process working set and peak commit are separate"}),
    )
}
