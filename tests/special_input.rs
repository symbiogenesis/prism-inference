#![forbid(unsafe_code)]
//! Focused explicit-special recognition, overlap, UTF-8 and resource regressions.

use bonsai_gguf::Metadata;
use bonsai_prism_runtime::Result;
use bonsai_prism_runtime::special_input::{
    InputSegment, MAX_SPECIAL_INPUT_BYTES, MAX_SPECIAL_INPUT_SEGMENTS, MAX_SPECIAL_SPELLING_BYTES,
    MAX_SPECIAL_TOKENS, SpecialInput, SpecialInputPolicy,
};

fn metadata(tokens: &[(&str, i32)]) -> Metadata {
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
    ])
}

#[test]
fn recognition_requires_explicit_parse_and_declared_special_types() -> Result<()> {
    let records = [
        ("<ctrl>", 3),
        ("<user>", 4),
        ("<normal>", 1),
        ("<unknown>", 2),
        ("<unused>", 5),
        ("<0x61>", 6),
        ("<ctrl>", 1),
    ];
    let input = SpecialInput::from_metadata(&metadata(&records), records.len())?;
    let text = "<ctrl><user><normal><unknown><unused><0x61>";
    assert_eq!(
        input.split(text, SpecialInputPolicy::Literal, 1)?,
        [InputSegment::Text(text)]
    );
    assert_eq!(
        input.split(text, SpecialInputPolicy::Parse, 3)?,
        [
            InputSegment::Special(0),
            InputSegment::Special(1),
            InputSegment::Text("<normal><unknown><unused><0x61>"),
        ]
    );
    Ok(())
}

#[test]
fn parse_chooses_earliest_start_then_longest_match() -> Result<()> {
    let records = [("ab", 3), ("abc", 4), ("bcd", 3), ("bc", 4)];
    let input = SpecialInput::from_metadata(&metadata(&records), records.len())?;
    assert_eq!(
        input.split("xabcabdbcd!", SpecialInputPolicy::Parse, 7)?,
        [
            InputSegment::Text("x"),
            InputSegment::Special(1),
            InputSegment::Special(0),
            InputSegment::Text("d"),
            InputSegment::Special(2),
            InputSegment::Text("!"),
        ]
    );
    // "bcd" is longer, but its later start cannot displace "ab".
    assert_eq!(
        input.split("abcd", SpecialInputPolicy::Parse, 2)?,
        [InputSegment::Special(1), InputSegment::Text("d")]
    );
    let earliest = SpecialInput::from_metadata(&metadata(&[("ab", 3), ("bcd", 4)]), 2)?;
    assert_eq!(
        earliest.split("abcd", SpecialInputPolicy::Parse, 2)?,
        [InputSegment::Special(0), InputSegment::Text("cd")]
    );
    Ok(())
}

#[test]
fn adjacent_and_unicode_specials_preserve_borrowed_text_ranges() -> Result<()> {
    let records = [("λ", 3), ("λ🦀", 4), ("終", 4), ("\0", 3)];
    let input = SpecialInput::from_metadata(&metadata(&records), records.len())?;
    let text = "préλ🦀終\0尾λ";
    let segments = input.split(text, SpecialInputPolicy::Parse, 6)?;
    assert_eq!(
        segments,
        [
            InputSegment::Text("pré"),
            InputSegment::Special(1),
            InputSegment::Special(2),
            InputSegment::Special(3),
            InputSegment::Text("尾"),
            InputSegment::Special(0),
        ]
    );
    let InputSegment::Text(first) = segments[0] else {
        return Err("missing initial ordinary range".into());
    };
    assert_eq!(first.as_ptr(), text.as_ptr());
    Ok(())
}

#[test]
fn empty_input_and_a_vocabulary_without_specials_insert_nothing() -> Result<()> {
    let input = SpecialInput::from_metadata(&metadata(&[("a", 1)]), 1)?;
    for policy in [SpecialInputPolicy::Literal, SpecialInputPolicy::Parse] {
        assert_eq!(input.split("", policy, 0)?, []);
        assert_eq!(
            input.split("<|im_start|>λ", policy, 1)?,
            [InputSegment::Text("<|im_start|>λ")]
        );
        assert!(input.split("a", policy, 0).is_err());
    }
    // Insertion flags belong to a separate contract and cannot insert tokens.
    let mut values = metadata(&[("<bos>", 3)]);
    values.insert("tokenizer.ggml.add_bos_token".to_owned(), true.into());
    values.insert("tokenizer.ggml.add_eos_token".to_owned(), true.into());
    let input = SpecialInput::from_metadata(&values, 1)?;
    assert_eq!(input.split("", SpecialInputPolicy::Parse, 0)?, []);
    assert_eq!(
        input.split("hello", SpecialInputPolicy::Parse, 1)?,
        [InputSegment::Text("hello")]
    );
    Ok(())
}

#[test]
fn overlapping_failure_links_agree_with_a_simple_independent_oracle() -> Result<()> {
    let spellings = ["ab", "bab", "bc", "bca", "c", "caa", "aaab", "aab"];
    let records = spellings.map(|text| (text, 3));
    let input = SpecialInput::from_metadata(&metadata(&records), records.len())?;
    // Exhaustive short input catches inherited outputs, failed longer matches,
    // earliest-start selection and longest-at-start without copying the trie.
    for length in 0..=7_u32 {
        for mut value in 0..3_usize.pow(length) {
            let mut text = String::new();
            for _ in 0..length {
                text.push(['a', 'b', 'c'][value % 3]);
                value /= 3;
            }
            let mut expected = Vec::new();
            let mut position = 0;
            let mut text_start = 0;
            while position < text.len() {
                let matched = spellings
                    .iter()
                    .enumerate()
                    .filter(|(_, spelling)| text[position..].starts_with(**spelling))
                    .max_by_key(|(_, spelling)| spelling.len());
                if let Some((token, spelling)) = matched {
                    if text_start < position {
                        expected.push(InputSegment::Text(&text[text_start..position]));
                    }
                    expected.push(InputSegment::Special(u32::try_from(token)?));
                    position += spelling.len();
                    text_start = position;
                } else {
                    position += 1;
                }
            }
            if text_start < text.len() {
                expected.push(InputSegment::Text(&text[text_start..]));
            }
            assert_eq!(
                input.split(&text, SpecialInputPolicy::Parse, usize::MAX)?,
                expected,
                "input {text:?}"
            );
        }
    }
    Ok(())
}

#[test]
fn ambiguous_special_spellings_are_rejected() {
    for kinds in [[3, 3], [4, 4], [3, 4], [4, 3]] {
        let records = [("<same>", kinds[0]), ("<same>", kinds[1])];
        assert!(SpecialInput::from_metadata(&metadata(&records), records.len()).is_err());
    }
}

#[test]
fn spelling_and_special_count_limits_are_enforced() -> Result<()> {
    let at_limit = "λ".repeat(MAX_SPECIAL_SPELLING_BYTES / 2);
    let input = SpecialInput::from_metadata(&metadata(&[(&at_limit, 4)]), 1)?;
    assert_eq!(
        input.split(&at_limit, SpecialInputPolicy::Parse, 1)?,
        [InputSegment::Special(0)]
    );
    let over_limit = "a".repeat(MAX_SPECIAL_SPELLING_BYTES + 1);
    assert!(SpecialInput::from_metadata(&metadata(&[(&over_limit, 3)]), 1).is_err());

    let spellings = (0..=MAX_SPECIAL_TOKENS)
        .map(|index| format!("<special:{index}>"))
        .collect::<Vec<_>>();
    let records = spellings
        .iter()
        .map(|spelling| (spelling.as_str(), 3))
        .collect::<Vec<_>>();
    let input = SpecialInput::from_metadata(
        &metadata(&records[..MAX_SPECIAL_TOKENS]),
        MAX_SPECIAL_TOKENS,
    )?;
    assert_eq!(
        input.split(
            &spellings[MAX_SPECIAL_TOKENS - 1],
            SpecialInputPolicy::Parse,
            1
        )?,
        [InputSegment::Special(u32::try_from(
            MAX_SPECIAL_TOKENS - 1
        )?)]
    );
    assert!(SpecialInput::from_metadata(&metadata(&records), records.len()).is_err());
    Ok(())
}

#[test]
fn input_and_output_are_bounded_even_with_unlimited_caller_limits() -> Result<()> {
    let input = SpecialInput::from_metadata(&metadata(&[("x", 3)]), 1)?;
    let at_limit = "a".repeat(MAX_SPECIAL_INPUT_BYTES);
    let over_limit = "a".repeat(MAX_SPECIAL_INPUT_BYTES + 1);
    for policy in [SpecialInputPolicy::Literal, SpecialInputPolicy::Parse] {
        assert_eq!(
            input.split(&at_limit, policy, 1)?,
            [InputSegment::Text(&at_limit)]
        );
        assert!(input.split(&over_limit, policy, usize::MAX).is_err());
        assert!(input.split("x", policy, 0).is_err());
    }
    assert!(input.split("axb", SpecialInputPolicy::Parse, 2).is_err());
    assert_eq!(
        input.split("axb", SpecialInputPolicy::Parse, 3)?,
        [
            InputSegment::Text("a"),
            InputSegment::Special(0),
            InputSegment::Text("b"),
        ]
    );
    let many = "x".repeat(MAX_SPECIAL_INPUT_SEGMENTS);
    assert_eq!(
        input
            .split(&many, SpecialInputPolicy::Parse, usize::MAX)?
            .len(),
        MAX_SPECIAL_INPUT_SEGMENTS
    );
    let too_many = "x".repeat(MAX_SPECIAL_INPUT_SEGMENTS + 1);
    assert!(
        input
            .split(&too_many, SpecialInputPolicy::Parse, usize::MAX)
            .is_err()
    );
    Ok(())
}
