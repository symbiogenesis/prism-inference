#![forbid(unsafe_code)]
#![cfg(feature = "cli")]
//! Swift-only native prompt-byte and tokenizer captures at pinned Prism 73bfe6c.
//!
//! The fixture template was extracted after full pinned artifact verification.
//! Native template bytes came from the unchanged pinned Jinja/JSON/Unicode
//! sources, compiled for Windows ARM64 with MSVC 19.51. Native token IDs came
//! from the pinned archive's `llama-tokenize --ids --no-bos --no-escape` with
//! exact binary prompt input, special parsing enabled and offline mode. The
//! fixture records source/executable identities; these are correctness captures,
//! never throughput measurements. Regeneration requires both native controls.

use std::path::Path;

use bonsai_prism_runtime::Result;
use bonsai_prism_runtime::chat::ChatTemplate;
use bonsai_prism_runtime::model::ModelMetadata;
use bonsai_prism_runtime::swift_chat::SwiftChatSession;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const TEMPLATE: &str = include_str!("fixtures/swift-chat-template.jinja");
const CAPTURE: &str = include_str!("fixtures/swift-prompt-parity.json");

fn capture() -> Result<Value> {
    serde_json::from_str(CAPTURE).map_err(Into::into)
}

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .into_iter()
        .flat_map(|byte| [byte >> 4, byte & 15])
        .map(|nibble| char::from(b"0123456789abcdef"[usize::from(nibble)]))
        .collect()
}

#[test]
fn swift_pinned_template_matches_native_bytes_with_reasoning_history() -> Result<()> {
    let fixture = capture()?;
    assert_eq!(digest(TEMPLATE.as_bytes()), fixture["template_sha256"]);
    let template = ChatTemplate::new(TEMPLATE, "<|endoftext|>", "<|im_end|>")?;
    for case in fixture["cases"].as_array().ok_or("missing native cases")? {
        let rendered = template.render_json(
            case["messages"].clone(),
            true,
            &json!({"enable_thinking":true,"reasoning_effort":"xhigh","preserve_thinking":true}),
        )?;
        assert_eq!(rendered, case["rendered"]);
        assert!(rendered.ends_with("<|im_start|>assistant\n<think>\n"));
    }
    let history = fixture["cases"][1]["rendered"]
        .as_str()
        .ok_or("missing history")?;
    assert!(history.contains("Consider a small tree.\nKeep this supplied thought."));
    assert!(history.contains("café 🌱.<|im_end|>"));
    assert!(history.contains("user\nUse that name in one line.<|im_end|>"));
    Ok(())
}

#[test]
fn swift_chat_rejects_foreign_model_and_unverified_template_before_tokenizer_build() -> Result<()> {
    let fixture = capture()?;
    let mut metadata = ModelMetadata {
        metadata: bonsai_gguf::Metadata::new(),
        vocab_size: 248_320,
        identity: "0".repeat(64),
    };
    assert!(SwiftChatSession::from_metadata(&metadata).is_err());
    metadata.identity = fixture["model_sha256"]
        .as_str()
        .ok_or("missing model hash")?
        .into();
    metadata.metadata.insert(
        "general.architecture".into(),
        bonsai_gguf::MetadataValue::String("qwen35".into()),
    );
    metadata.metadata.insert(
        "tokenizer.chat_template".into(),
        bonsai_gguf::MetadataValue::String(format!("{TEMPLATE}\n")),
    );
    let error = SwiftChatSession::from_metadata(&metadata)
        .err()
        .ok_or("changed template accepted")?;
    assert!(error.to_string().contains("template identity mismatch"));
    Ok(())
}

#[test]
#[ignore = "requires the pinned artifact at BONSAI_SWIFT_MODEL_PATH; focused device qualification"]
fn swift_actual_artifact_matches_native_prompt_bytes_ids_and_budget() -> Result<()> {
    let model_path = std::env::var("BONSAI_SWIFT_MODEL_PATH")?;
    let session = SwiftChatSession::load(Path::new(&model_path), 128 * 1024 * 1024)?;
    let fixture = capture()?;
    assert_eq!(session.identity(), fixture["model_sha256"]);
    assert_eq!(session.template_sha256(), fixture["template_sha256"]);
    assert_eq!(
        session.stop_tokens(),
        [248_044, 248_046, 248_063, 248_064, 248_065]
    );
    let generated = session.generation(4096)?;
    assert_eq!(generated.max_tokens, 4096);
    assert_eq!(
        generated.stop_tokens,
        [248_046, 248_044, 248_063, 248_064, 248_065]
    );
    assert!(session.generation(0).is_err());
    assert!(session.generation(4097).is_err());
    for case in fixture["cases"].as_array().ok_or("missing native cases")? {
        let prompt = session.prepare(&case["messages"], 4096)?;
        assert_eq!(prompt.rendered, case["rendered"]);
        assert_eq!(
            serde_json::to_value(&prompt.tokenization.token_ids)?,
            case["token_ids"]
        );
        assert_eq!(prompt.tokenization.context, 8192);
        assert_eq!(prompt.tokenization.reserved_output_tokens, 4096);
        assert_eq!(
            serde_json::to_value(prompt.tokenization.prompt_token_count())?,
            case["prompt_token_count"]
        );
    }
    assert!(
        session
            .prepare(&json!([{"role":"user","content":"q ".repeat(5000)}]), 4096)
            .is_err()
    );
    assert!(
        session
            .prepare(&fixture["cases"][0]["messages"], 4096)
            .is_ok()
    );
    Ok(())
}
