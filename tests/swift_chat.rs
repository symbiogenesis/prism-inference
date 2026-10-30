//! Swift-only native prompt-byte and tokenizer captures at pinned Prism 73bfe6c.
//!
//! The fixture template was extracted after full pinned artifact verification.
//! Native template bytes came from the unchanged pinned Jinja/JSON/Unicode
//! sources, compiled for Windows ARM64 with MSVC 19.51. Native token IDs came
//! from the pinned archive's `llama-tokenize --ids --no-bos --no-escape` with
//! exact binary prompt input, special parsing enabled and offline mode. The
//! fixture records source/executable identities; these are correctness captures,
//! never throughput measurements. Regeneration requires both native controls.
#![forbid(unsafe_code)]
#![cfg(feature = "cli")]

use std::path::Path;

use bonsai_prism_runtime::Result;
use bonsai_prism_runtime::chat::ChatTemplate;
use bonsai_prism_runtime::model::ModelMetadata;
use bonsai_prism_runtime::model::swift::{self, Representation};
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
fn swift_pinned_template_disables_reasoning_without_losing_supplied_history() -> Result<()> {
    let fixture = capture()?;
    let template = ChatTemplate::new(TEMPLATE, "<|endoftext|>", "<|im_end|>")?;
    for case in fixture["cases"].as_array().ok_or("missing native cases")? {
        let rendered = template.render_json(
            case["messages"].clone(),
            true,
            &json!({"enable_thinking":false,"reasoning_effort":"xhigh","preserve_thinking":true}),
        )?;
        assert!(rendered.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        assert!(!rendered.contains("Reasoning effort is set to xhigh."));
        if case["messages"]
            .as_array()
            .is_some_and(|messages| messages.len() > 1)
        {
            assert!(rendered.contains("Consider a small tree.\nKeep this supplied thought."));
        }
    }
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

fn synthetic_swift_metadata() -> Result<ModelMetadata> {
    use bonsai_gguf::{Metadata, MetadataArray, MetadataValue};

    // A bounded tokenizer-only fixture: 256 normal byte pieces, pinned control
    // spellings and unused entries retain the exact declared vocabulary size.
    let mut tokens = vec![String::new(); swift::VOCAB_SIZE];
    let mut types = vec![5; swift::VOCAB_SIZE];
    let mut next = 256_u32;
    for byte in u8::MIN..=u8::MAX {
        let scalar = if (b'!'..=b'~').contains(&byte)
            || (0xa1..=0xac).contains(&byte)
            || (0xae..=0xff).contains(&byte)
        {
            u32::from(byte)
        } else {
            let scalar = next;
            next += 1;
            scalar
        };
        tokens[usize::from(byte)] = char::from_u32(scalar)
            .ok_or("invalid fixture byte alphabet")?
            .to_string();
        types[usize::from(byte)] = 1;
    }
    for (id, spelling) in [
        (248_044, "<|endoftext|>"),
        (248_045, "<|im_start|>"),
        (248_046, "<|im_end|>"),
        (248_063, "<|fim_pad|>"),
        (248_064, "<|repo_name|>"),
        (248_065, "<|file_sep|>"),
    ] {
        tokens[id] = spelling.into();
        types[id] = 3;
    }
    Ok(ModelMetadata {
        identity: swift::MODEL_SHA256.into(),
        vocab_size: swift::VOCAB_SIZE,
        metadata: Metadata::from([
            ("general.architecture".into(), MetadataValue::from("qwen35")),
            ("tokenizer.ggml.model".into(), MetadataValue::from("gpt2")),
            ("tokenizer.ggml.pre".into(), MetadataValue::from("qwen35")),
            (
                "tokenizer.chat_template".into(),
                MetadataValue::from(TEMPLATE),
            ),
            (
                "tokenizer.ggml.tokens".into(),
                MetadataValue::Array(MetadataArray::String(tokens)),
            ),
            (
                "tokenizer.ggml.token_type".into(),
                MetadataValue::Array(MetadataArray::I32(types)),
            ),
            (
                "tokenizer.ggml.merges".into(),
                MetadataValue::Array(MetadataArray::String(Vec::new())),
            ),
            (
                "tokenizer.ggml.bos_token_id".into(),
                MetadataValue::U32(248_044),
            ),
            (
                "tokenizer.ggml.eos_token_id".into(),
                MetadataValue::U32(248_046),
            ),
            (
                "tokenizer.ggml.padding_token_id".into(),
                MetadataValue::U32(248_044),
            ),
            (
                "tokenizer.ggml.add_bos_token".into(),
                MetadataValue::Bool(false),
            ),
        ]),
    })
}

#[test]
fn swift_chat_current_weight_selection_preserves_exact_template_tokens_and_default_identity()
-> Result<()> {
    use bonsai_gguf::MetadataValue;

    let mut model = synthetic_swift_metadata()?;
    let pq2 = SwiftChatSession::from_metadata(&model)?;
    assert!(
        SwiftChatSession::from_metadata_representation(&model, Representation::CurrentRepackedPtq1)
            .is_err()
    );
    model.identity = swift::REPACKED_PTQ1_SHA256.into();
    assert!(SwiftChatSession::from_metadata(&model).is_err());
    let ptq1 = SwiftChatSession::from_metadata_representation(
        &model,
        Representation::CurrentRepackedPtq1,
    )?;
    assert_ne!(pq2.identity(), ptq1.identity());
    assert_eq!(ptq1.identity(), swift::REPACKED_PTQ1_SHA256);
    assert_eq!(pq2.template_sha256(), ptq1.template_sha256());
    assert_eq!(pq2.stop_tokens(), ptq1.stop_tokens());
    for thinking in [false, true] {
        let messages = json!([{"role":"user", "content":"Name a tree."}]);
        assert_eq!(
            pq2.prepare_with_thinking(&messages, 4096, thinking)?,
            ptq1.prepare_with_thinking(&messages, 4096, thinking)?
        );
    }
    model.identity = "older-published-PTQ1-checkpoint".into();
    assert!(
        SwiftChatSession::from_metadata_representation(&model, Representation::CurrentRepackedPtq1)
            .is_err()
    );
    model.identity = swift::REPACKED_PTQ1_SHA256.into();
    model.metadata.insert(
        "tokenizer.chat_template".into(),
        MetadataValue::from(format!("{TEMPLATE}\n")),
    );
    assert!(
        SwiftChatSession::from_metadata_representation(&model, Representation::CurrentRepackedPtq1)
            .is_err()
    );
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
        let disabled = session.prepare_with_thinking(&case["messages"], 4096, false)?;
        assert!(
            disabled
                .rendered
                .ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        );
        assert!(
            !disabled
                .rendered
                .contains("Reasoning effort is set to xhigh.")
        );
        assert_ne!(
            disabled.tokenization.token_ids,
            prompt.tokenization.token_ids
        );
        assert_eq!(disabled.tokenization.reserved_output_tokens, 4096);
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
