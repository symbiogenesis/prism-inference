#![forbid(unsafe_code)]
//! Shared strict vocabulary validation across independent and composed callers.

use bonsai_gguf::{Metadata, MetadataValue};
use bonsai_prism_runtime::Result;
use bonsai_prism_runtime::bpe::Bpe;
use bonsai_prism_runtime::special_input::SpecialInput;
use bonsai_prism_runtime::tokenizer::Tokenizer;
use bonsai_prism_runtime::vocabulary::{MAX_PIECE_BYTES, MAX_VOCABULARY_TOKENS, Vocabulary};

fn metadata(piece: &str, kind: i32) -> Metadata {
    Metadata::from([
        ("tokenizer.ggml.model".to_owned(), "gpt2".into()),
        ("tokenizer.ggml.pre".to_owned(), "qwen35".into()),
        ("tokenizer.ggml.tokens".to_owned(), vec!["a", piece].into()),
        (
            "tokenizer.ggml.token_type".to_owned(),
            vec![1_i32, kind].into(),
        ),
        (
            "tokenizer.ggml.merges".to_owned(),
            Vec::<String>::new().into(),
        ),
    ])
}

fn assert_rejected(values: &Metadata, count: usize) {
    for (constructor, result) in [
        (
            "Vocabulary",
            Vocabulary::from_metadata(values, count).map(|_| ()),
        ),
        ("Bpe", Bpe::from_metadata(values, count).map(|_| ())),
        (
            "SpecialInput",
            SpecialInput::from_metadata(values, count).map(|_| ()),
        ),
        (
            "Tokenizer",
            Tokenizer::from_metadata(values, count).map(|_| ()),
        ),
    ] {
        assert!(
            result.is_err(),
            "{constructor} accepted count={count}, metadata={values:?}"
        );
    }
}

#[test]
fn every_constructor_requires_exact_metadata_types_and_graph_counts() {
    let valid = metadata("b", 1);
    for count in [0, 1, 3, MAX_VOCABULARY_TOKENS + 1, usize::MAX] {
        assert_rejected(&valid, count);
    }
    for (key, value) in [
        ("tokenizer.ggml.model", "llama".into()),
        ("tokenizer.ggml.model", MetadataValue::U32(1)),
        ("tokenizer.ggml.pre", "qwen2".into()),
        ("tokenizer.ggml.pre", MetadataValue::Bool(true)),
        (
            "tokenizer.ggml.tokens",
            MetadataValue::String("a".to_owned()),
        ),
        ("tokenizer.ggml.tokens", vec![1_i32, 2].into()),
        ("tokenizer.ggml.tokens", Vec::<String>::new().into()),
        ("tokenizer.ggml.token_type", vec![1_u32, 1].into()),
        ("tokenizer.ggml.token_type", vec![1_i64, 1].into()),
        ("tokenizer.ggml.token_type", vec![1_f32, 1.0].into()),
        ("tokenizer.ggml.token_type", vec![true, true].into()),
        ("tokenizer.ggml.token_type", vec![1_i32].into()),
        ("tokenizer.ggml.token_type", vec![1_i32, 1, 1].into()),
        ("tokenizer.ggml.token_type", Vec::<i32>::new().into()),
    ] {
        let mut values = valid.clone();
        values.insert(key.to_owned(), value);
        assert_rejected(&values, 2);
    }
    for key in [
        "tokenizer.ggml.model",
        "tokenizer.ggml.pre",
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.token_type",
    ] {
        let mut values = valid.clone();
        values.remove(key);
        assert_rejected(&values, 2);
    }
}

#[test]
fn every_constructor_validates_pieces_outside_its_own_token_types() -> Result<()> {
    for (piece, kind) in [
        ("b", 1),
        ("", 2),
        ("<control>", 3),
        ("<user>", 4),
        ("", 5),
        ("<0x41>", 6),
    ] {
        let values = metadata(piece, kind);
        Vocabulary::from_metadata(&values, 2)?;
        Bpe::from_metadata(&values, 2)?;
        SpecialInput::from_metadata(&values, 2)?;
        Tokenizer::from_metadata(&values, 2)?;
    }
    for piece in ["", " ", "\n", "\0", "λ", "🦀", "Ŕ"] {
        assert_rejected(&metadata(piece, 1), 2);
    }
    for piece in [
        "", "<0x0>", "<0x100>", "<0xGG>", "<0xλ>", "<0XFF>", "x<0xFF>", "<0xFF>x", "<0x041>",
    ] {
        assert_rejected(&metadata(piece, 6), 2);
    }
    for kind in [0, 7, -1, i32::MAX] {
        assert_rejected(&metadata("b", kind), 2);
    }
    for kind in [3, 4] {
        assert_rejected(&metadata("", kind), 2);
    }
    Ok(())
}

#[test]
fn input_constructors_enforce_the_decoded_piece_limit_without_output_storage() {
    let piece = "b".repeat(MAX_PIECE_BYTES + 1);
    let values = metadata(&piece, 1);
    for result in [
        Bpe::from_metadata(&values, 2).map(|_| ()),
        SpecialInput::from_metadata(&values, 2).map(|_| ()),
        Tokenizer::from_metadata(&values, 2).map(|_| ()),
    ] {
        assert!(result.is_err_and(|error| error.to_string().contains("decoded byte limit")));
    }
}
