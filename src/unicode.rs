//! Bounded UTF-8 regular-expression splitting and GPT byte encoding.

use regex::RegexBuilder;

use crate::{Result, invalid};

/// Encodes the pinned runtime's integer codepoint domain into raw UTF-8 bytes.
///
/// The upstream primitive admits surrogate values as three raw bytes. The
/// returned buffer therefore deliberately does not promise Rust string validity.
///
/// # Errors
/// Rejects values above `0x0010_ffff`.
pub fn codepoint_to_utf8(point: u32) -> Result<Vec<u8>> {
    let values = match point {
        0..=0x7f => vec![point],
        0x80..=0x7ff => vec![0xc0 | (point >> 6), 0x80 | (point & 0x3f)],
        0x800..=0xffff => vec![
            0xe0 | (point >> 12),
            0x80 | ((point >> 6) & 0x3f),
            0x80 | (point & 0x3f),
        ],
        0x10000..=0x0010_ffff => vec![
            0xf0 | (point >> 18),
            0x80 | ((point >> 12) & 0x3f),
            0x80 | ((point >> 6) & 0x3f),
            0x80 | (point & 0x3f),
        ],
        _ => return Err(invalid("invalid codepoint")),
    };
    values
        .into_iter()
        .map(|value| u8::try_from(value).map_err(Into::into))
        .collect()
}

/// Decodes raw bytes with the pinned production bulk-decoder semantics.
///
/// Malformed prefixes produce one replacement codepoint per rejected byte.
/// As in the source primitive, structurally complete two-to-four-byte sequences
/// retain their integer value, including overlong and surrogate encodings.
/// Strict Rust UTF-8 validation is a separate caller policy.
///
/// # Errors
/// Rejects input above 1 MiB and allocation failures.
pub fn codepoints_from_utf8(bytes: &[u8]) -> Result<Vec<u32>> {
    if bytes.len() > 1024 * 1024 {
        return Err(invalid("Unicode byte input exceeds bound"));
    }
    let mut output = Vec::new();
    output.try_reserve_exact(bytes.len())?;
    let mut offset = 0;
    while offset < bytes.len() {
        let first = bytes[offset];
        let (length, mask) = match first {
            0..=0x7f => (1, 0x7f),
            0xc0..=0xdf => (2, 0x1f),
            0xe0..=0xef => (3, 0x0f),
            0xf0..=0xf7 => (4, 7),
            _ => (0, 0),
        };
        if length == 0
            || offset + length > bytes.len()
            || bytes[offset + 1..offset + length]
                .iter()
                .any(|value| value & 0xc0 != 0x80)
        {
            output.push(0xfffd);
            offset += 1;
            continue;
        }
        let mut point = u32::from(first & mask);
        for value in &bytes[offset + 1..offset + length] {
            point = (point << 6) | u32::from(value & 0x3f);
        }
        output.push(point);
        offset += length;
    }
    Ok(output)
}

/// Splits text sequentially with each expression, retaining unmatched spans.
///
/// Expressions use the linear-time Rust regex syntax, including Unicode
/// categories. Each later expression subdivides the existing pieces without
/// recombining them. Optional GPT byte encoding maps every original UTF-8 byte
/// to its reversible vocabulary character. No normalization is performed.
///
/// # Errors
/// Rejects unsupported regex syntax and bounded input/expression/output limits.
pub fn regex_split(text: &str, expressions: &[&str], byte_encoding: bool) -> Result<Vec<String>> {
    if text.len() > 65536
        || expressions.is_empty()
        || expressions.len() > 16
        || expressions.iter().any(|expression| expression.len() > 8192)
    {
        return Err(invalid("Unicode split input or expression bound exceeded"));
    }
    let mut pieces = if text.is_empty() {
        Vec::new()
    } else {
        vec![text]
    };
    for expression in expressions {
        let regex = RegexBuilder::new(expression)
            .size_limit(8 * 1024 * 1024)
            .build()?;
        let mut next = Vec::new();
        for piece in pieces {
            let mut end = 0;
            for matched in regex.find_iter(piece) {
                if matched.start() > end {
                    next.push(&piece[end..matched.start()]);
                }
                if !matched.is_empty() {
                    next.push(matched.as_str());
                }
                end = matched.end();
            }
            if end < piece.len() {
                next.push(&piece[end..]);
            }
            if next.len() > 65536 {
                return Err(invalid("Unicode split output piece bound exceeded"));
            }
        }
        pieces = next;
    }
    if byte_encoding {
        let mut missing = 256_u32;
        let mut table = ['\0'; 256];
        for byte in 0..=u8::MAX {
            let codepoint =
                if (b'!'..=b'~').contains(&byte) || (0xa1..=0xac).contains(&byte) || byte >= 0xae {
                    u32::from(byte)
                } else {
                    let codepoint = missing;
                    missing += 1;
                    codepoint
                };
            table[usize::from(byte)] =
                char::from_u32(codepoint).ok_or_else(|| invalid("invalid byte encoding scalar"))?;
        }
        Ok(pieces
            .into_iter()
            .map(|piece| piece.bytes().map(|byte| table[usize::from(byte)]).collect())
            .collect())
    } else {
        Ok(pieces.into_iter().map(str::to_owned).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_utf8_domain_and_malformed_prefixes_match_the_source() -> Result<()> {
        for point in [
            0,
            0x7f,
            0x80,
            0x7ff,
            0x800,
            0xd800,
            0xdfff,
            0xffff,
            0x10000,
            0x0010_ffff,
        ] {
            assert_eq!(codepoints_from_utf8(&codepoint_to_utf8(point)?)?, [point]);
        }
        assert!(codepoint_to_utf8(0x0011_0000).is_err());
        assert_eq!(
            codepoints_from_utf8(&[0xff, 0xe2, b'x', 0x80])?,
            [0xfffd, 0xfffd, u32::from(b'x'), 0xfffd]
        );
        assert_eq!(codepoints_from_utf8(&[0xc0, 0x80])?, [0]);
        Ok(())
    }
}
