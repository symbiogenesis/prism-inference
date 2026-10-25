//! Streaming byte extents and source-compatible Unicode parser primitives.

use super::{Arena, ParseResult, Parser, Range, Result, invalid};

pub(super) enum Utf8 {
    Point(u32, usize),
    Incomplete,
    Invalid,
}
pub(super) fn point(bytes: &[u8], offset: usize) -> Utf8 {
    let Some(&first) = bytes.get(offset) else {
        return Utf8::Incomplete;
    };
    let (length, mask) = match first {
        0..=0x7f => (1, 0x7f),
        0xc0..=0xdf => (2, 0x1f),
        0xe0..=0xef => (3, 0x0f),
        0xf0..=0xf7 => (4, 7),
        _ => return Utf8::Invalid,
    };
    if offset + length > bytes.len() {
        return Utf8::Incomplete;
    }
    let mut result = u32::from(first & mask);
    for &value in &bytes[offset + 1..offset + length] {
        if value & 0xc0 != 0x80 {
            return Utf8::Invalid;
        }
        result = (result << 6) | u32::from(value & 0x3f);
    }
    Utf8::Point(result, length)
}

pub(super) fn classes(pattern: &str) -> Result<(Vec<Range>, bool)> {
    let content = pattern.strip_prefix('[').unwrap_or(pattern);
    let content = content.strip_suffix(']').unwrap_or(content);
    let (negated, content) = content
        .strip_prefix('^')
        .map_or((false, content), |content| (true, content));
    let bytes = content.as_bytes();
    let parse = |offset: usize| -> Result<(u32, usize)> {
        let value = *bytes
            .get(offset)
            .ok_or_else(|| invalid("PEG class character missing"))?;
        if value != b'\\' || offset + 1 == bytes.len() {
            return Ok((u32::from(value), 1));
        }
        let next = bytes[offset + 1];
        let length = match next {
            b'x' => 2,
            b'u' => 4,
            b'U' => 8,
            _ => 0,
        };
        if length > 0 {
            if let Some(hex) = bytes.get(offset + 2..offset + 2 + length)
                && hex.iter().all(u8::is_ascii_hexdigit)
            {
                return Ok((
                    u32::from_str_radix(std::str::from_utf8(hex)?, 16)?,
                    length + 2,
                ));
            }
            return Ok((u32::from(next), 2));
        }
        Ok((
            u32::from(match next {
                b'n' => b'\n',
                b't' => b'\t',
                b'r' => b'\r',
                _ => next,
            }),
            2,
        ))
    };
    let mut ranges = Vec::new();
    let mut offset = 0;
    while offset < bytes.len() {
        let (start, length) = parse(offset)?;
        offset += length;
        if offset + 1 < bytes.len() && bytes[offset] == b'-' {
            let (end, length) = parse(offset + 1)?;
            ranges.push(Range { start, end });
            offset += length + 1;
        } else {
            ranges.push(Range { start, end: start });
        }
    }
    Ok((ranges, negated))
}

impl Arena {
    /// Parses bounded arbitrary bytes with source flags (bit zero is lenient).
    ///
    /// # Errors
    /// Rejects unsupported flag bits, excessive input, recursion or work budgets.
    pub fn parse(&self, bytes: &[u8], flags: u8) -> Result<ParseResult> {
        if bytes.len() > 65536 || flags > 3 {
            return Err(invalid("PEG input bytes/flags outside bound"));
        }
        self.parse_at(self.root, bytes, flags & 1 != 0, 0, 0, &mut 0)
    }
    #[allow(
        clippy::too_many_lines,
        reason = "Keep parser variant streaming/extent transitions next to each other"
    )]
    pub(super) fn parse_at(
        &self,
        id: usize,
        bytes: &[u8],
        lenient: bool,
        start: usize,
        depth: usize,
        steps: &mut usize,
    ) -> Result<ParseResult> {
        *steps += 1;
        if depth > 256 || *steps > 1_000_000 {
            return Err(invalid("PEG recursion/work budget exceeded"));
        }
        let result = |kind, end| ParseResult::new(kind, start, end);
        let child = |id, start, steps: &mut usize| {
            self.parse_at(id, bytes, lenient, start, depth + 1, steps)
        };
        Ok(
            match self
                .parsers
                .get(id)
                .ok_or_else(|| invalid("PEG parser ID invalid"))?
            {
                Parser::Epsilon => result(1, start),
                Parser::Start => result(u8::from(start == 0), start),
                Parser::End => result(u8::from(start >= bytes.len()), start),
                Parser::Literal { literal } => {
                    let mut offset = start;
                    let mut kind = 1;
                    for value in literal.bytes() {
                        if offset == bytes.len() {
                            kind = if lenient { 2 } else { 0 };
                            break;
                        }
                        if bytes[offset] != value {
                            kind = 0;
                            break;
                        }
                        offset += 1;
                    }
                    result(kind, if kind == 0 { start } else { offset })
                }
                Parser::Sequence { children } => {
                    let mut offset = start;
                    let mut kind = 1;
                    for &id in children {
                        let matched = child(id, offset, steps)?;
                        offset = matched.end;
                        if matched.result_type != 1 {
                            kind = matched.result_type;
                            break;
                        }
                    }
                    result(kind, offset)
                }
                Parser::Choice { children } => {
                    let mut matched = result(0, start);
                    for &id in children {
                        let trial = child(id, start, steps)?;
                        if trial.result_type != 0 {
                            matched = trial;
                            break;
                        }
                    }
                    matched
                }
                Parser::Repetition {
                    child: id,
                    min_count,
                    max_count,
                } => {
                    let mut offset = start;
                    let mut count = 0;
                    let mut incomplete = None;
                    while (*max_count < 0 || count < *max_count) && offset < bytes.len() {
                        let matched = child(*id, offset, steps)?;
                        if matched.result_type == 2 {
                            incomplete = Some(matched.end);
                            break;
                        }
                        if matched.result_type == 0 || matched.end == offset {
                            break;
                        }
                        offset = matched.end;
                        count += 1;
                    }
                    incomplete.map_or_else(
                        || {
                            if count < *min_count {
                                result(
                                    if offset >= bytes.len() && lenient {
                                        2
                                    } else {
                                        0
                                    },
                                    offset,
                                )
                            } else {
                                result(1, offset)
                            }
                        },
                        |end| result(2, end),
                    )
                }
                Parser::And { child: id } => result(child(*id, start, steps)?.result_type, start),
                Parser::Not { child: id } => result(
                    match child(*id, start, steps)?.result_type {
                        1 => 0,
                        2 => 2,
                        _ => 1,
                    },
                    start,
                ),
                Parser::Any => match point(bytes, start) {
                    Utf8::Point(_, length) => result(1, start + length),
                    Utf8::Incomplete => result(if lenient { 2 } else { 0 }, start),
                    Utf8::Invalid => result(0, start),
                },
                Parser::Space => {
                    let mut offset = start;
                    while bytes.get(offset).is_some_and(u8::is_ascii_whitespace) {
                        offset += 1;
                    }
                    result(1, offset)
                }
                Parser::Chars {
                    ranges,
                    negated,
                    min_count,
                    max_count,
                    ..
                } => {
                    let mut offset = start;
                    let mut count = 0;
                    let mut early = None;
                    while *max_count < 0 || count < *max_count {
                        match point(bytes, offset) {
                            Utf8::Incomplete => {
                                early = Some(if count >= *min_count {
                                    result(1, offset)
                                } else if lenient {
                                    result(2, offset)
                                } else {
                                    result(0, start)
                                });
                                break;
                            }
                            Utf8::Invalid => {
                                early = Some(if count >= *min_count {
                                    result(1, offset)
                                } else {
                                    result(0, start)
                                });
                                break;
                            }
                            Utf8::Point(value, length) => {
                                let contains = ranges
                                    .iter()
                                    .any(|range| range.start <= value && value <= range.end);
                                if contains == *negated {
                                    break;
                                }
                                offset += length;
                                count += 1;
                            }
                        }
                    }
                    early.unwrap_or_else(|| {
                        if count < *min_count {
                            result(
                                if offset >= bytes.len() && lenient {
                                    2
                                } else {
                                    0
                                },
                                offset,
                            )
                        } else {
                            result(1, offset)
                        }
                    })
                }
                Parser::String { delimiter } => {
                    let delimiter = delimiter
                        .as_bytes()
                        .first()
                        .copied()
                        .ok_or_else(|| invalid("PEG string delimiter empty"))?;
                    let mut offset = start;
                    let mut matched = None;
                    while offset < bytes.len() {
                        let value = bytes[offset];
                        if value == delimiter {
                            matched = Some(result(1, offset));
                            break;
                        }
                        if value == b'\\' {
                            let save = offset;
                            offset += 1;
                            if offset == bytes.len() {
                                matched = Some(result(
                                    if lenient { 2 } else { 0 },
                                    if lenient { save } else { start },
                                ));
                                break;
                            }
                            let escape = bytes[offset];
                            if escape == delimiter
                                || matches!(escape, b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't')
                            {
                                offset += 1;
                            } else if escape == b'u' {
                                offset += 1;
                                for _ in 0..4 {
                                    if offset == bytes.len() {
                                        matched = Some(result(
                                            if lenient { 2 } else { 0 },
                                            if lenient { save } else { start },
                                        ));
                                        break;
                                    }
                                    if !bytes[offset].is_ascii_hexdigit() {
                                        matched = Some(result(0, start));
                                        break;
                                    }
                                    offset += 1;
                                }
                                if matched.is_some() {
                                    break;
                                }
                            } else {
                                matched = Some(result(0, start));
                                break;
                            }
                        } else {
                            match point(bytes, offset) {
                                Utf8::Point(_, length) => offset += length,
                                Utf8::Incomplete => {
                                    matched = Some(result(
                                        if lenient { 2 } else { 0 },
                                        if lenient { offset } else { start },
                                    ));
                                    break;
                                }
                                Utf8::Invalid => {
                                    matched = Some(result(0, start));
                                    break;
                                }
                            }
                        }
                    }
                    matched.unwrap_or_else(|| result(if lenient { 2 } else { 0 }, offset))
                }
                Parser::Until { delimiters } => {
                    let mut offset = start;
                    let mut matched = None;
                    while offset < bytes.len() {
                        match point(bytes, offset) {
                            Utf8::Incomplete => {
                                matched = Some(result(
                                    if lenient { 2 } else { 0 },
                                    if lenient { offset } else { start },
                                ));
                                break;
                            }
                            Utf8::Invalid => {
                                matched = Some(result(0, start));
                                break;
                            }
                            Utf8::Point(_, length) => {
                                if delimiters.iter().any(|delimiter| {
                                    !delimiter.is_empty()
                                        && (bytes[offset..].starts_with(delimiter.as_bytes())
                                            || delimiter.as_bytes().starts_with(&bytes[offset..]))
                                }) {
                                    matched = Some(result(1, offset));
                                    break;
                                }
                                offset += length;
                            }
                        }
                    }
                    matched.unwrap_or_else(|| result(if lenient { 2 } else { 1 }, offset))
                }
                Parser::Rule { child: id, .. }
                | Parser::Atomic { child: id }
                | Parser::Tag { child: id, .. }
                | Parser::Gbnf { child: id, .. }
                | Parser::Ac { child: id, .. }
                | Parser::Schema { child: id, .. } => child(*id, start, steps)?,
                Parser::Ref { name } => child(
                    *self
                        .rules
                        .get(name)
                        .ok_or_else(|| invalid("PEG rule reference missing"))?,
                    start,
                    steps,
                )?,
            },
        )
    }
}
