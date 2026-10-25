#![forbid(unsafe_code)]
//! Focused pinned Qwen35 ordinary-text boundary and resource-limit regressions.

use bonsai_gguf::{Metadata, MetadataValue};
use bonsai_prism_runtime::Result;
use bonsai_prism_runtime::bpe::{MAX_BPE_INPUT_BYTES, MAX_BPE_INPUT_PIECES, MAX_BPE_PIECE_BYTES};
use bonsai_prism_runtime::pretokenizer::Qwen35PreTokenizer;

fn metadata() -> Metadata {
    Metadata::from([
        ("tokenizer.ggml.model".to_owned(), "gpt2".into()),
        ("tokenizer.ggml.pre".to_owned(), "qwen35".into()),
    ])
}

#[test]
fn contractions_keep_the_pinned_branch_order_and_ascii_case_policy() -> Result<()> {
    let splitter = Qwen35PreTokenizer::from_metadata(&metadata())?;
    assert_eq!(
        splitter.split("can't I'VE We're")?,
        ["can", "'t", " I", "'VE", " We", "'re"]
    );
    assert_eq!(
        splitter.split("'s'S't'T'm'M'd'D're'RE've'VE'll'LL")?,
        [
            "'s", "'S", "'t", "'T", "'m", "'M", "'d", "'D", "'re", "'RE", "'ve", "'VE", "'ll",
            "'LL"
        ]
    );
    assert_eq!(splitter.split("'ſtail")?, ["'ſtail"]);
    assert_eq!(splitter.split(" 'LL")?, [" '", "LL"]);
    Ok(())
}

#[test]
fn letters_marks_and_single_unicode_numbers_follow_pinned_categories() -> Result<()> {
    let splitter = Qwen35PreTokenizer::from_metadata(&metadata())?;
    assert_eq!(
        splitter.split("hello世界 e\u{301} \u{301}abc 12²Ⅷ٣")?,
        [
            "hello世界",
            " e\u{301}",
            " \u{301}abc",
            " ",
            "1",
            "2",
            "²",
            "Ⅷ",
            "٣"
        ]
    );
    assert_eq!(splitter.split("\u{301}\u{342}A")?, ["\u{301}\u{342}A"]);
    assert_eq!(splitter.split("\u{903}\u{20dd}A")?, ["\u{903}\u{20dd}A"]);
    assert_eq!(splitter.split("🦀字 !é")?, ["🦀字", " !", "é"]);
    // U+1C89 is unassigned in the pinned table, although later Unicode versions
    // classify it as a Cyrillic letter. Platform Unicode upgrades cannot alter
    // the runtime's token boundaries.
    assert_eq!(splitter.split("A\u{1c89}!")?, ["A", "\u{1c89}!"]);
    Ok(())
}

#[test]
fn punctuation_consumes_crlf_and_preserves_every_utf8_byte() -> Result<()> {
    let splitter = Qwen35PreTokenizer::from_metadata(&metadata())?;
    for (input, expected) in [
        ("hi!!!\r\n\nnext", vec!["hi", "!!!\r\n\n", "next"]),
        (" !!!\r\n", vec![" !!!\r\n"]),
        ("a\r\nb", vec!["a", "\r\n", "b"]),
        ("\0🦀‍🚀\n", vec!["\0🦀‍🚀\n"]),
        ("<|im_start|>\n", vec!["<|", "im", "_start", "|>\n"]),
    ] {
        let pieces = splitter.split(input)?;
        assert_eq!(pieces, expected);
        assert_eq!(pieces.concat(), input);
        let mut offset = 0;
        for piece in pieces {
            assert_ne!(piece, "");
            assert_eq!(piece.as_ptr(), input[offset..].as_ptr());
            offset += piece.len();
        }
    }
    Ok(())
}

#[test]
fn whitespace_backtracks_one_scalar_and_stops_after_the_last_crlf() -> Result<()> {
    let splitter = Qwen35PreTokenizer::from_metadata(&metadata())?;
    for (input, expected) in [
        ("  word", vec![" ", " word"]),
        ("\t\tword", vec!["\t", "\tword"]),
        (" \u{3000}word", vec![" ", "\u{3000}word"]),
        ("\t\u{3000}7", vec!["\t", "\u{3000}", "7"]),
        (" \n \t\r\n  word", vec![" \n \t\r\n", " ", " word"]),
        ("word \t\u{3000}", vec!["word", " \t\u{3000}"]),
        (" \u{2028}word", vec![" ", "\u{2028}word"]),
        ("\r\n\u{2028}", vec!["\r\n", "\u{2028}"]),
    ] {
        assert_eq!(splitter.split(input)?, expected, "input: {input:?}");
    }
    Ok(())
}

#[test]
fn every_pinned_whitespace_scalar_uses_the_whitespace_branch() -> Result<()> {
    let splitter = Qwen35PreTokenizer::from_metadata(&metadata())?;
    for scalar in [
        0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x20, 0x85, 0xa0, 0x1680, 0x2000, 0x2001, 0x2002, 0x2003,
        0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a, 0x2028, 0x2029, 0x202f, 0x205f,
        0x3000,
    ] {
        let character = char::from_u32(scalar).ok_or("invalid whitespace fixture")?;
        let input = format!("{character}{character}7");
        let first = character.to_string();
        if matches!(character, '\r' | '\n') {
            assert_eq!(splitter.split(&input)?, [&input[..2], "7"]);
        } else {
            assert_eq!(
                splitter.split(&input)?,
                [first.as_str(), first.as_str(), "7"]
            );
        }
    }
    // Control characters excluded from the pinned whitespace set remain ordinary
    // punctuation bytes, even if a platform's whitespace predicate accepts them.
    assert_eq!(
        splitter.split("\u{1c}\u{1d}\u{1e}\u{1f}7")?,
        ["\u{1c}\u{1d}\u{1e}\u{1f}", "7"]
    );
    Ok(())
}

#[test]
fn empty_input_and_exact_piece_limits_are_supported() -> Result<()> {
    let splitter = Qwen35PreTokenizer::from_metadata(&metadata())?;
    assert_eq!(splitter.split("")?, Vec::<&str>::new());
    for character in ['a', '!', ' ', 'é'] {
        let input = character
            .to_string()
            .repeat(MAX_BPE_PIECE_BYTES / character.len_utf8());
        assert_eq!(splitter.split(&input)?, [input.as_str()]);
        let oversized = format!("{input}{character}");
        assert!(splitter.split(&oversized).is_err());
    }
    Ok(())
}

#[test]
fn input_byte_and_piece_count_limits_fail_without_invented_boundaries() -> Result<()> {
    let splitter = Qwen35PreTokenizer::from_metadata(&metadata())?;
    let exact = "1".repeat(MAX_BPE_INPUT_PIECES);
    assert_eq!(splitter.split(&exact)?.len(), MAX_BPE_INPUT_PIECES);
    assert!(splitter.split(&format!("{exact}1")).is_err());
    let chunk = format!("{}1", "a".repeat(MAX_BPE_PIECE_BYTES - 1));
    let full = chunk.repeat(MAX_BPE_INPUT_BYTES / chunk.len());
    assert_eq!(full.len(), MAX_BPE_INPUT_BYTES);
    assert_eq!(splitter.split(&full)?.concat(), full);
    assert!(splitter.split(&format!("{full}a")).is_err());
    Ok(())
}

#[test]
fn checked_iterator_borrows_boundaries_and_fuses_after_limit_errors() -> Result<()> {
    let splitter = Qwen35PreTokenizer::from_metadata(&metadata())?;
    let input = "can't  é1\r\n🦀";
    let mut pieces = splitter.pieces(input)?;
    let mut offset = 0;
    for expected in ["can", "'t", " ", " é", "1", "\r\n", "🦀"] {
        let piece = pieces.next().ok_or("missing expected boundary")??;
        assert_eq!(piece, expected);
        assert_eq!(piece.as_ptr(), input[offset..].as_ptr());
        offset += piece.len();
    }
    assert_eq!(offset, input.len());
    assert!(pieces.next().is_none());
    assert!(pieces.next().is_none());

    // A valid prefix is available before a later oversized match is rejected.
    let oversized_piece = "1".to_owned() + &"a".repeat(MAX_BPE_PIECE_BYTES + 1);
    let mut pieces = splitter.pieces(&oversized_piece)?;
    assert_eq!(pieces.next().ok_or("missing valid prefix")??, "1");
    assert!(pieces.next().is_some_and(|piece| {
        piece.is_err_and(|error| error.to_string().contains("piece exceeds byte limit"))
    }));
    assert!(pieces.next().is_none());

    let too_many = "1".repeat(MAX_BPE_INPUT_PIECES + 1);
    let mut pieces = splitter.pieces(&too_many)?;
    for _ in 0..MAX_BPE_INPUT_PIECES {
        assert_eq!(pieces.next().ok_or("missing permitted piece")??, "1");
    }
    assert!(pieces.next().is_some_and(|piece| {
        piece.is_err_and(|error| error.to_string().contains("input exceeds piece limit"))
    }));
    assert!(pieces.next().is_none());
    assert!(
        splitter
            .pieces(&"1".repeat(MAX_BPE_INPUT_BYTES + 1))
            .is_err()
    );
    Ok(())
}

#[test]
fn missing_mistyped_and_other_tokenizer_metadata_is_rejected() {
    for key in ["tokenizer.ggml.model", "tokenizer.ggml.pre"] {
        let mut values = metadata();
        values.remove(key);
        assert!(Qwen35PreTokenizer::from_metadata(&values).is_err());
        for value in [
            MetadataValue::Bool(true),
            MetadataValue::U32(1),
            "other".into(),
        ] {
            values.insert(key.to_owned(), value);
            assert!(Qwen35PreTokenizer::from_metadata(&values).is_err());
        }
    }
}
