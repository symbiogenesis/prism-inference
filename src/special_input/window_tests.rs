//! Independent matching and allocation bounds for request-local reverse windows.

use super::{
    InputSegment, MATCH_WINDOW_BYTES, MAX_SPECIAL_INPUT_BYTES, MAX_SPECIAL_INPUT_SEGMENTS,
    MAX_SPECIAL_TOKENS, NO_MATCH, SpecialInput, SpecialInputPolicy,
};
use crate::Result;

fn oracle<'a>(text: &'a str, spellings: &[(&str, u32)]) -> Vec<InputSegment<'a>> {
    let mut output = Vec::new();
    let mut position = 0;
    let mut text_start = 0;
    while position < text.len() {
        let matched = spellings
            .iter()
            .filter(|entry| text[position..].starts_with(entry.0))
            .max_by_key(|entry| entry.0.len());
        if let Some((spelling, token)) = matched {
            if text_start < position {
                output.push(InputSegment::Text(&text[text_start..position]));
            }
            output.push(InputSegment::Special(*token));
            position += spelling.len();
            text_start = position;
        } else if let Some(character) = text[position..].chars().next() {
            position += character.len_utf8();
        }
    }
    if text_start < text.len() {
        output.push(InputSegment::Text(&text[text_start..]));
    }
    output
}

fn recognizer(spellings: &[(&str, u32)]) -> Result<SpecialInput> {
    SpecialInput::build(
        spellings
            .iter()
            .map(|(spelling, token)| (spelling.as_bytes(), *token))
            .collect(),
    )
}

#[test]
fn windowed_specials_match_independent_oracle_at_all_spelling_boundary_offsets() -> Result<()> {
    let longest = format!("{}bc", "a".repeat(254));
    let shorter = "a".repeat(254);
    let spellings = [
        (longest.as_str(), 1),
        (shorter.as_str(), 2),
        ("bc", 3),
        ("bca", 4),
        ("caa", 5),
        ("\u{3bb}\u{1f980}", 6),
        ("\u{1f980}", 7),
        ("\0", 8),
    ];
    let input = recognizer(&spellings)?;
    for offset in 0..=256 {
        let prefix = "q".repeat(MATCH_WINDOW_BYTES - offset);
        let text = format!("{prefix}{longest}bcabca\u{3bb}\u{1f980}\0tail");
        let expected = oracle(&text, &spellings);
        assert_eq!(
            input.split(&text, SpecialInputPolicy::Parse, expected.len())?,
            expected,
            "boundary offset {offset}"
        );
        assert!(
            input
                .split(&text, SpecialInputPolicy::Parse, expected.len() - 1)
                .is_err()
        );
    }
    for offset in 0..8 {
        // The owned window can end inside either ordinary or special UTF-8.
        let prefix = "\u{e9}".repeat(MATCH_WINDOW_BYTES / 2 - offset);
        let text = format!("{prefix}x\u{3bb}\u{1f980}\0{prefix}\u{3bb}\u{1f980}");
        assert_eq!(
            input.split(&text, SpecialInputPolicy::Parse, usize::MAX)?,
            oracle(&text, &spellings)
        );
    }
    Ok(())
}

#[test]
fn windowed_specials_keep_fixed_scratch_and_grow_only_emitted_segments() -> Result<()> {
    assert_eq!(size_of::<[u16; MATCH_WINDOW_BYTES]>(), 8192);
    let input = recognizer(&[("<stop>", 1)])?;
    let text = "a".repeat(MAX_SPECIAL_INPUT_BYTES);
    let segments = input.split(&text, SpecialInputPolicy::Parse, usize::MAX)?;
    assert_eq!(segments, [InputSegment::Text(&text)]);
    assert!(segments.capacity() < 32);
    let InputSegment::Text(borrowed) = segments[0] else {
        return Err("missing borrowed ordinary segment".into());
    };
    assert_eq!(borrowed.as_ptr(), text.as_ptr());
    let repeated = "<stop>".repeat(MAX_SPECIAL_INPUT_SEGMENTS);
    let segments = input.split(&repeated, SpecialInputPolicy::Parse, usize::MAX)?;
    assert_eq!(segments.len(), MAX_SPECIAL_INPUT_SEGMENTS);
    assert_eq!(segments.first(), Some(&InputSegment::Special(1)));
    assert_eq!(segments.last(), Some(&InputSegment::Special(1)));
    let rejected = format!("{repeated}<stop>");
    assert!(
        input
            .split(&rejected, SpecialInputPolicy::Parse, usize::MAX)
            .is_err()
    );
    assert!(
        input
            .split(&format!("{text}a"), SpecialInputPolicy::Parse, usize::MAX)
            .is_err()
    );
    assert_eq!(
        input.split("a<stop>b", SpecialInputPolicy::Parse, 3)?,
        [
            InputSegment::Text("a"),
            InputSegment::Special(1),
            InputSegment::Text("b"),
        ]
    );
    Ok(())
}

#[test]
fn windowed_specials_retain_maximum_special_index_and_literal_limits() -> Result<()> {
    let spellings = (0..MAX_SPECIAL_TOKENS)
        .map(|index| format!("<{index}>"))
        .collect::<Vec<_>>();
    let records = spellings
        .iter()
        .enumerate()
        .map(|(index, spelling)| Ok((spelling.as_str(), u32::try_from(index)?)))
        .collect::<Result<Vec<_>>>()?;
    let input = recognizer(&records)?;
    assert!(input.nodes.iter().any(|node| node.best == 4095));
    assert!(input.nodes.iter().any(|node| node.best == NO_MATCH));
    let text = format!("{}<4095><0><4094>", "x".repeat(MATCH_WINDOW_BYTES - 2));
    assert_eq!(
        input.split(&text, SpecialInputPolicy::Parse, 4)?,
        oracle(&text, &records)
    );
    assert_eq!(
        input.split(&text, SpecialInputPolicy::Literal, 1)?,
        [InputSegment::Text(&text)]
    );
    for policy in [SpecialInputPolicy::Parse, SpecialInputPolicy::Literal] {
        assert!(input.split(&text, policy, 0).is_err());
        assert_eq!(input.split("", policy, 0)?.len(), 0);
    }
    Ok(())
}
