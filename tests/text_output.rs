#![forbid(unsafe_code)]
//! Focused incremental UTF-8 display, bounds, and request-reuse regressions.

use bonsai_gguf::Metadata;
use bonsai_prism_runtime::{
    Result,
    text_output::{MAX_TEXT_OUTPUT_BYTES, Utf8Output},
    vocabulary::Vocabulary,
};

fn append(output: &mut Utf8Output, bytes: &[u8], rendered: &mut String) -> Result<()> {
    rendered.push_str(&output.push(bytes)?);
    assert!(output.pending_bytes().len() <= 3);
    assert_eq!(output.push(&[])?, "");
    Ok(())
}

fn assert_all_chunkings(bytes: &[u8]) -> Result<()> {
    let expected = String::from_utf8_lossy(bytes);
    for mask in 0..(1_usize << bytes.len().saturating_sub(1)) {
        let mut output = Utf8Output::new();
        let mut rendered = String::new();
        let mut start = 0;
        for end in 1..bytes.len() {
            if mask & (1 << (end - 1)) != 0 {
                append(&mut output, &bytes[start..end], &mut rendered)?;
                start = end;
            }
        }
        append(&mut output, &bytes[start..], &mut rendered)?;
        rendered.push_str(output.finish());
        assert_eq!(rendered, expected, "bytes {bytes:?}, partition {mask}");
        assert_eq!(output.pending_bytes(), [0_u8; 0]);
        assert_eq!(output.finish(), "");
    }
    Ok(())
}

#[test]
fn common_text_survives_every_single_split_and_bytewise_input() -> Result<()> {
    for text in ["", "a café\n", "🌱 bonsai", "\0中文", "a\u{0301} \r\n\t"] {
        let bytes = text.as_bytes();
        for split in 0..=bytes.len() {
            let mut output = Utf8Output::default();
            let mut rendered = String::new();
            append(&mut output, &bytes[..split], &mut rendered)?;
            append(&mut output, &bytes[split..], &mut rendered)?;
            rendered.push_str(output.finish());
            assert_eq!(rendered, text);
            assert_eq!(output.pending_bytes(), [0_u8; 0]);
        }
        let mut output = Utf8Output::new();
        let mut rendered = String::new();
        for byte in bytes {
            append(&mut output, &[*byte], &mut rendered)?;
        }
        rendered.push_str(output.finish());
        assert_eq!(rendered, text);
        assert_eq!(output.pending_bytes(), [0_u8; 0]);
    }
    Ok(())
}

#[test]
fn malformed_and_truncated_sequences_match_standard_lossy_display_in_all_partitions() -> Result<()>
{
    for bytes in [
        &[][..],
        &[0xff, b'a', 0xc3, b'b'][..],
        &[0xf0, 0x9f, 0x8c][..],
        &[0xed, 0xa0, 0x80][..],
        &[0xc0, 0xaf][..],
        &[0xe2, 0x82, b'!', 0x80][..],
        &[0xf4, 0x90, 0x80, 0x80][..],
        &[0xf0, 0x8f, 0xbf, 0xbf][..],
        &[0xe0, 0x9f, 0xbf][..],
        &[0xf0, 0x9f, 0xf0, 0x9f, 0x8c, 0xb1][..],
        &[0xc3, 0xa9, 0xc3, 0xa9, 0xe4, 0xb8, 0xad][..],
        &[0xe2, 0x82, 0xac, 0xe2, 0x82, 0xac, 0xc3][..],
        &[0x80, 0xbf, 0xfe, 0xff, 0xc2, 0xdf, 0xe0, 0xf4][..],
    ] {
        assert_all_chunkings(bytes)?;
    }
    Ok(())
}

#[test]
fn every_byte_and_utf8_boundary_value_survives_arbitrary_chunking() -> Result<()> {
    for byte in u8::MIN..=u8::MAX {
        for bytes in [
            [0xf0, 0x9f, byte, 0x80, b'a'],
            [0xe0, byte, 0xbf, 0xc2, 0xa0],
            [0xed, byte, 0x80, 0xc3, 0xa9],
            [0xf4, byte, 0x80, 0x80, byte],
        ] {
            assert_all_chunkings(&bytes)?;
        }
    }
    Ok(())
}

#[test]
fn bounded_push_rejects_without_losing_pending_bytes_and_accepts_exact_limit() -> Result<()> {
    let mut output = Utf8Output::new();
    assert_eq!(output.push(&[0xf0, 0x9f, 0x8c])?, "");
    assert_eq!(output.pending_bytes(), &[0xf0, 0x9f, 0x8c]);
    let oversized = vec![b'a'; MAX_TEXT_OUTPUT_BYTES + 1];
    assert!(output.push(&oversized).is_err());
    assert_eq!(output.pending_bytes(), &[0xf0, 0x9f, 0x8c]);
    assert_eq!(output.push(&[0xb1])?, "🌱");
    assert_eq!(output.finish(), "");

    let bounded = &oversized[..MAX_TEXT_OUTPUT_BYTES];
    assert_eq!(output.push(bounded)?.as_bytes(), bounded);
    assert_eq!(output.pending_bytes(), [0_u8; 0]);
    let malformed = vec![0xff; MAX_TEXT_OUTPUT_BYTES];
    let rendered = output.push(&malformed)?;
    assert_eq!(rendered.len(), 3 * MAX_TEXT_OUTPUT_BYTES);
    assert!(rendered.chars().all(|character| character == '\u{fffd}'));
    assert_eq!(output.finish(), "");
    Ok(())
}

#[test]
fn reset_and_finish_make_request_reuse_explicit() -> Result<()> {
    let mut output = Utf8Output::new();
    assert_eq!(output.finish(), "");
    for suffix in [&[0xc3][..], &[0xe2, 0x82][..], &[0xf0, 0x9f, 0x8c][..]] {
        assert_eq!(output.push(suffix)?, "");
        assert_eq!(output.pending_bytes(), suffix);
        assert_eq!(output.push(&[])?, "");
        assert_eq!(output.pending_bytes(), suffix);
        assert_eq!(output.finish(), "\u{fffd}");
        assert_eq!(output.finish(), "");
        assert_eq!(output.push(b"new request")?, "new request");

        assert_eq!(output.push(suffix)?, "");
        output.reset();
        assert_eq!(output.pending_bytes(), [0_u8; 0]);
        assert_eq!(output.finish(), "");
        assert_eq!(output.push(&[0x80])?, "\u{fffd}");
        assert_eq!(output.push(b"after cancellation")?, "after cancellation");
    }
    Ok(())
}

#[test]
fn vocabulary_composition_preserves_lossless_bytes_and_literal_control_spellings() -> Result<()> {
    let metadata = Metadata::from([
        ("tokenizer.ggml.model".to_owned(), "gpt2".into()),
        ("tokenizer.ggml.pre".to_owned(), "qwen35".into()),
        (
            "tokenizer.ggml.tokens".to_owned(),
            [
                "hello",
                "<0x20>",
                "<0xF0>",
                "<0x9F>",
                "<0x8C>",
                "<0xB1>",
                "<|im_end|>",
                "\0",
                "<0xFF>",
            ]
            .map(str::to_owned)
            .to_vec()
            .into(),
        ),
        (
            "tokenizer.ggml.token_type".to_owned(),
            vec![1_i32, 6, 6, 6, 6, 6, 3, 4, 6].into(),
        ),
    ]);
    let vocabulary = Vocabulary::from_metadata(&metadata, 9)?;
    let mut output = Utf8Output::new();
    let mut lossless = Vec::new();
    let mut rendered = String::new();
    for token in 0..9 {
        let piece = vocabulary.piece(token)?;
        lossless.extend_from_slice(piece);
        append(&mut output, piece, &mut rendered)?;
    }
    rendered.push_str(output.finish());
    assert_eq!(lossless, b"hello \xf0\x9f\x8c\xb1<|im_end|>\0\xff");
    assert_eq!(rendered, "hello 🌱<|im_end|>\0\u{fffd}");
    assert_eq!(rendered, String::from_utf8_lossy(&lossless));
    assert!(vocabulary.is_control(6)?);
    Ok(())
}
