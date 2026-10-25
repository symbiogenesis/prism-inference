#![forbid(unsafe_code)]
//! Focused ranked BPE regressions for already pretokenized byte pieces.

use bonsai_gguf::{Metadata, MetadataValue};
use bonsai_prism_runtime::Result;
use bonsai_prism_runtime::bpe::{
    Bpe, MAX_BPE_INPUT_BYTES, MAX_BPE_INPUT_PIECES, MAX_BPE_MERGE_BYTES, MAX_BPE_MERGES,
    MAX_BPE_PIECE_BYTES,
};
use bonsai_prism_runtime::vocabulary::{MAX_PIECE_BYTES, Vocabulary};

fn metadata(tokens: &[(&str, i32)], merges: &[&str]) -> Metadata {
    Metadata::from([
        ("tokenizer.ggml.model".to_owned(), "gpt2".into()),
        ("tokenizer.ggml.pre".to_owned(), "qwen35".into()),
        (
            "tokenizer.ggml.tokens".to_owned(),
            tokens
                .iter()
                .map(|(text, _)| *text)
                .collect::<Vec<_>>()
                .into(),
        ),
        (
            "tokenizer.ggml.token_type".to_owned(),
            tokens
                .iter()
                .map(|(_, kind)| *kind)
                .collect::<Vec<_>>()
                .into(),
        ),
        ("tokenizer.ggml.merges".to_owned(), merges.to_vec().into()),
    ])
}

// Independent forward GPT-2 alphabet used to build complete fallback fixtures.
fn byte_alphabet() -> Result<Vec<String>> {
    let printable = (b'!'..=b'~')
        .chain(0xa1..=0xac)
        .chain(0xae..=0xff)
        .collect::<Vec<_>>();
    let mut next = 256_u32;
    (u8::MIN..=u8::MAX)
        .map(|byte| {
            let scalar = if printable.contains(&byte) {
                u32::from(byte)
            } else {
                let scalar = next;
                next += 1;
                scalar
            };
            char::from_u32(scalar)
                .map(|character| character.to_string())
                .ok_or_else(|| "invalid independent byte alphabet".into())
        })
        .collect()
}

fn complete_metadata() -> Result<Metadata> {
    let alphabet = byte_alphabet()?;
    let tokens = alphabet
        .iter()
        .map(|text| (text.as_str(), 1))
        .collect::<Vec<_>>();
    Ok(metadata(&tokens, &[]))
}

fn decode(vocabulary: &Vocabulary, tokens: &[u32]) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    for &token in tokens {
        bytes.extend_from_slice(vocabulary.piece(token)?);
    }
    Ok(bytes)
}

#[test]
fn array_rank_wins_over_longest_vocabulary_piece() -> Result<()> {
    let tokens = [
        ("a", 1),
        ("b", 1),
        ("c", 1),
        ("ab", 1),
        ("bc", 1),
        ("abc", 1),
    ];
    let bpe = Bpe::from_metadata(&metadata(&tokens, &["b c", "a b", "ab c"]), tokens.len())?;
    assert_eq!(bpe.encode_piece(b"abc", 3)?, [0, 4]);
    let reordered = Bpe::from_metadata(&metadata(&tokens, &["a b", "b c", "ab c"]), tokens.len())?;
    assert_eq!(reordered.encode_piece(b"abc", 1)?, [5]);
    Ok(())
}

#[test]
fn equal_rank_overlaps_merge_leftmost_and_refresh_neighbor_candidates() -> Result<()> {
    let repeated = metadata(&[("a", 1), ("aa", 1), ("aaa", 1)], &["a a", "aa a"]);
    let bpe = Bpe::from_metadata(&repeated, 3)?;
    assert_eq!(bpe.encode_piece(b"aaa", 1)?, [2]);
    assert_eq!(bpe.encode_piece(b"aaaaa", 2)?, [1, 2]);
    let tokens = [
        ("a", 1),
        ("b", 1),
        ("c", 1),
        ("ab", 1),
        ("bc", 1),
        ("abc", 1),
    ];
    let refreshed = Bpe::from_metadata(&metadata(&tokens, &["ab c", "a b", "b c"]), tokens.len())?;
    assert_eq!(refreshed.encode_piece(b"abc", 1)?, [5]);
    Ok(())
}

#[test]
fn pretokenizer_boundaries_are_never_crossed() -> Result<()> {
    let values = metadata(&[("a", 1), ("b", 1), ("ab", 1)], &["a b"]);
    let bpe = Bpe::from_metadata(&values, 3)?;
    assert_eq!(bpe.encode_piece(b"ab", 1)?, [2]);
    assert_eq!(bpe.encode_pieces(&[b"a", b"", b"b"], 2)?, [0, 1]);
    assert!(bpe.encode_pieces(&[b"a", b"b"], 1).is_err());
    Ok(())
}

#[test]
fn merged_pieces_share_the_aggregate_token_allowance() -> Result<()> {
    let values = metadata(&[("a", 1), ("b", 1), ("ab", 1)], &["a b"]);
    let bpe = Bpe::from_metadata(&values, 3)?;
    let pieces: &[&[u8]] = &[b"", b"ab", b"a", b"", b"b", b"ab", b""];
    assert_eq!(bpe.encode_pieces(pieces, 4)?, [2, 0, 1, 2]);
    assert!(bpe.encode_pieces(pieces, 3).is_err());
    // A failed request leaves subsequent encodings independent.
    assert_eq!(bpe.encode_pieces(pieces, usize::MAX)?, [2, 0, 1, 2]);
    assert!(bpe.encode_pieces(&[b"ab", b"x"], 4).is_err());
    assert_eq!(bpe.encode_pieces(&[b"ab", b""], 1)?, [2]);
    Ok(())
}

#[test]
fn successive_unequal_pieces_keep_nodes_and_merge_candidates_independent() -> Result<()> {
    let tokens = [
        ("a", 1),
        ("aa", 1),
        ("aaa", 1),
        ("b", 1),
        ("ab", 1),
        ("aab", 1),
    ];
    let bpe = Bpe::from_metadata(
        &metadata(&tokens, &["a a", "aa a", "a b", "aa b"]),
        tokens.len(),
    )?;
    // Repeated leftmost ties leave stale candidates; neither those candidates
    // nor old links may affect the next shorter, longer or empty piece.
    let pieces: &[&[u8]] = &[b"aaaaa", b"", b"a", b"aab", b"b", b"aaa", b"ab"];
    let expected = [1, 2, 0, 5, 3, 2, 4];
    assert_eq!(bpe.encode_pieces(pieces, expected.len())?, expected);
    assert!(bpe.encode_pieces(pieces, expected.len() - 1).is_err());
    assert!(bpe.encode_pieces(&[b"aaaaa", b"aax"], 10).is_err());
    assert_eq!(bpe.encode_pieces(pieces, usize::MAX)?, expected);
    Ok(())
}

#[test]
fn all_bytes_and_unicode_round_trip_without_merges() -> Result<()> {
    let values = complete_metadata()?;
    let vocabulary = Vocabulary::from_metadata(&values, 256)?;
    let bpe = Bpe::from_metadata(&values, 256)?;
    let bytes = (u8::MIN..=u8::MAX).collect::<Vec<_>>();
    let encoded = bpe.encode_piece(&bytes, 256)?;
    assert_eq!(encoded, (0_u32..256).collect::<Vec<_>>());
    assert_eq!(decode(&vocabulary, &encoded)?, bytes);
    let unicode = " λ🦀\n\t\0".as_bytes();
    assert_eq!(
        decode(&vocabulary, &bpe.encode_piece(unicode, unicode.len())?)?,
        unicode
    );
    Ok(())
}

#[test]
fn serialized_non_ascii_byte_operands_merge_and_decode_losslessly() -> Result<()> {
    let tokens = [("Ġ", 1), ("Ã", 1), ("©", 1), ("Ã©", 1), ("ĠÃ©", 1)];
    let values = metadata(&tokens, &["Ã ©", "Ġ Ã©"]);
    let bpe = Bpe::from_metadata(&values, tokens.len())?;
    let vocabulary = Vocabulary::from_metadata(&values, tokens.len())?;
    let encoded = bpe.encode_piece(" é".as_bytes(), 1)?;
    assert_eq!(encoded, [4]);
    assert_eq!(decode(&vocabulary, &encoded)?, " é".as_bytes());
    Ok(())
}

#[test]
fn special_looking_bytes_never_recognize_or_inject_special_ids() -> Result<()> {
    let mut alphabet = byte_alphabet()?;
    alphabet.extend([
        "<|im_end|>".to_owned(),
        "<|im_end|>".to_owned(),
        "<|user|>".to_owned(),
    ]);
    let tokens = alphabet
        .iter()
        .enumerate()
        .map(|(id, text)| {
            (
                text.as_str(),
                if id == 257 {
                    3
                } else if id == 258 {
                    4
                } else {
                    1
                },
            )
        })
        .collect::<Vec<_>>();
    let values = metadata(&tokens, &[]);
    let bpe = Bpe::from_metadata(&values, tokens.len())?;
    let vocabulary = Vocabulary::from_metadata(&values, tokens.len())?;
    let bytes = b"<|im_end|><|user|>";
    let encoded = bpe.encode_piece(bytes, bytes.len())?;
    assert!(encoded.iter().all(|&id| id < 256));
    assert_eq!(decode(&vocabulary, &encoded)?, bytes);
    Ok(())
}

#[test]
fn missing_normal_byte_fallback_fails_even_if_a_longer_or_byte_type_token_exists() -> Result<()> {
    let bpe = Bpe::from_metadata(&metadata(&[("ab", 1), ("<0x61>", 6), ("b", 1)], &[]), 3)?;
    assert!(bpe.encode_piece(b"a", 1).is_err());
    assert!(bpe.encode_piece(b"ab", 1).is_err());
    assert_eq!(bpe.encode_piece(b"b", 1)?, [2]);
    assert_eq!(bpe.encode_piece(b"", 0)?, [0_u32; 0]);
    assert!(bpe.encode_piece(b"b", 0).is_err());
    Ok(())
}

#[test]
fn merge_metadata_has_strict_type_and_required_presence() {
    let valid = metadata(&[("a", 1)], &[]);
    for value in [MetadataValue::String("a a".to_owned()), vec![1_i32].into()] {
        let mut values = valid.clone();
        values.insert("tokenizer.ggml.merges".to_owned(), value);
        assert!(Bpe::from_metadata(&values, 1).is_err());
    }
    let mut missing = valid;
    missing.remove("tokenizer.ggml.merges");
    assert!(Bpe::from_metadata(&missing, 1).is_err());
}

#[test]
fn malformed_duplicate_and_non_normal_rules_fail_closed() {
    let tokens = [("a", 1), ("b", 1), ("ab", 1)];
    for merge in [
        "", "ab", " a", "a ", "a  b", "a b c", "a\tb", "a x", "x b", "ab a",
    ] {
        assert!(Bpe::from_metadata(&metadata(&tokens, &[merge]), 3).is_err());
    }
    assert!(Bpe::from_metadata(&metadata(&tokens, &["a b", "a b"]), 3).is_err());
    assert!(Bpe::from_metadata(&metadata(&[("a", 1), ("a", 1)], &[]), 2).is_err());
    for kind in [2, 3, 4, 5, 6] {
        let special = if kind == 6 { "<0x61>" } else { "a" };
        let values = metadata(&[(special, kind), ("b", 1), ("ab", 1)], &["a b"]);
        assert!(Bpe::from_metadata(&values, 3).is_err());
    }
    let special_result = metadata(&[("a", 1), ("b", 1), ("ab", 3)], &["a b"]);
    assert!(Bpe::from_metadata(&special_result, 3).is_err());
}

#[test]
fn merge_count_and_aggregate_serialized_bytes_are_bounded_before_parsing() {
    let mut values = metadata(&[("a", 1)], &[]);
    values.insert(
        "tokenizer.ggml.merges".to_owned(),
        vec![String::new(); MAX_BPE_MERGES + 1].into(),
    );
    assert!(
        Bpe::from_metadata(&values, 1)
            .is_err_and(|error| error.to_string().contains("merge count"))
    );
    values.insert(
        "tokenizer.ggml.merges".to_owned(),
        vec!["a".repeat(MAX_PIECE_BYTES); MAX_BPE_MERGE_BYTES / MAX_PIECE_BYTES + 1].into(),
    );
    assert!(Bpe::from_metadata(&values, 1).is_err_and(|error| {
        error
            .to_string()
            .contains("merges exceed serialized byte limit")
    }));
    values.insert(
        "tokenizer.ggml.merges".to_owned(),
        vec!["a".repeat(2 * MAX_PIECE_BYTES + 2)].into(),
    );
    assert!(
        Bpe::from_metadata(&values, 1)
            .is_err_and(|error| error.to_string().contains("serialized rule byte limit"))
    );
}

#[test]
fn input_work_and_final_output_limits_are_enforced() -> Result<()> {
    let bpe = Bpe::from_metadata(&metadata(&[("a", 1), ("aa", 1)], &["a a"]), 2)?;
    let at_limit = vec![b'a'; MAX_BPE_PIECE_BYTES];
    assert_eq!(
        bpe.encode_piece(&at_limit, usize::MAX)?.len(),
        MAX_BPE_PIECE_BYTES / 2
    );
    assert!(
        bpe.encode_piece(&at_limit, MAX_BPE_PIECE_BYTES / 2 - 1)
            .is_err()
    );
    assert!(
        bpe.encode_piece(&vec![b'a'; MAX_BPE_PIECE_BYTES + 1], usize::MAX)
            .is_err()
    );
    let too_many = vec![b"".as_slice(); MAX_BPE_INPUT_PIECES + 1];
    assert!(bpe.encode_pieces(&too_many, usize::MAX).is_err());
    let too_large = vec![at_limit.as_slice(); MAX_BPE_INPUT_BYTES / MAX_BPE_PIECE_BYTES + 1];
    assert!(bpe.encode_pieces(&too_large, usize::MAX).is_err());
    assert_eq!(bpe.encode_pieces(&[], 0)?, [0_u32; 0]);
    Ok(())
}
