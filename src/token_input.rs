//! Bounded exact raw token-ID reading for file and service input streams.

use std::io::{self, Read};

use crate::{Result, invalid, tokenization::MAX_TOKENIZATION_CONTEXT};

/// Largest serialized raw-token input, independent of its admitted token count.
pub const MAX_TOKEN_INPUT_BYTES: usize = 1024 * 1024;

/// Read one strict JSON array of unsigned 32-bit token IDs through EOF.
///
/// IDs remain in their original order, including repeated IDs and zero. No
/// tokenization, vocabulary validation, template or special-token insertion is
/// performed. An empty array is valid for counting; drivers may separately
/// require a nonempty prompt. Only JSON whitespace and decimal JSON integers
/// are accepted, so negative numbers, fractions, exponents and leading zeroes
/// reject. The complete input must be exactly one array.
///
/// `token_limit` must be from 1 through [`MAX_TOKENIZATION_CONTEXT`]. Allocation
/// is fallible and bounded independently of a reader's size hint. At most
/// [`MAX_TOKEN_INPUT_BYTES`] plus one overflow byte is consumed, even for an
/// infinite stream. The entire serialized input is bounded before parsing.
///
/// This synchronous operation has no timeout; callers must arrange stream
/// cancellation or deadlines. I/O errors, including interruption, return once
/// without retrying or returning partial IDs. Retry requires a replaced or
/// rewound reader when the complete input is needed again.
///
/// # Errors
/// Rejects invalid limits, malformed/oversized input, excessive token counts,
/// allocation failures and I/O errors. Errors retain no input bytes or
/// reader-provided messages; I/O error kinds are preserved with a fixed message.
pub fn read_tokens(reader: &mut (impl Read + ?Sized), token_limit: usize) -> Result<Vec<u32>> {
    if !(1..=MAX_TOKENIZATION_CONTEXT).contains(&token_limit) {
        return Err(invalid("invalid raw input token limit"));
    }
    let maximum_read = MAX_TOKEN_INPUT_BYTES + 1;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(maximum_read)?;
    let mut buffer = [0_u8; 8192];
    loop {
        let available = buffer.len().min(maximum_read - bytes.len());
        let count = reader
            .read(&mut buffer[..available])
            .map_err(|error| io::Error::new(error.kind(), "failed to read raw token input"))?;
        if count == 0 {
            break;
        }
        bytes.extend_from_slice(&buffer[..count]);
        if bytes.len() > MAX_TOKEN_INPUT_BYTES {
            return Err(invalid("raw token input exceeds bounded byte limit"));
        }
    }
    parse_tokens(&bytes, token_limit)
}

fn skip_whitespace(bytes: &mut &[u8]) {
    while matches!(bytes.first(), Some(b' ' | b'\t' | b'\r' | b'\n')) {
        *bytes = &bytes[1..];
    }
}

fn parse_tokens(mut bytes: &[u8], token_limit: usize) -> Result<Vec<u32>> {
    skip_whitespace(&mut bytes);
    if bytes.first() != Some(&b'[') {
        return Err(invalid("raw token input must be a JSON array of u32 IDs"));
    }
    bytes = &bytes[1..];
    skip_whitespace(&mut bytes);
    let mut tokens = Vec::new();
    if bytes.first() != Some(&b']') {
        // Every ID needs at least one digit and an intervening separator.
        // Reserve a conservative serialized-size bound without unbounded growth.
        tokens.try_reserve_exact(token_limit.min(bytes.len().div_ceil(2)))?;
        loop {
            if tokens.len() == token_limit {
                return Err(invalid("raw input exceeds token limit"));
            }
            let token = parse_id(&mut bytes)?;
            tokens.push(token);
            skip_whitespace(&mut bytes);
            match bytes.first() {
                Some(b']') => break,
                Some(b',') => {
                    bytes = &bytes[1..];
                    skip_whitespace(&mut bytes);
                }
                _ => return Err(invalid("raw token input must be a JSON array of u32 IDs")),
            }
        }
    }
    bytes = &bytes[1..];
    skip_whitespace(&mut bytes);
    if !bytes.is_empty() {
        return Err(invalid("raw token input must be a JSON array of u32 IDs"));
    }
    Ok(tokens)
}

fn parse_id(bytes: &mut &[u8]) -> Result<u32> {
    let Some(first @ b'0'..=b'9') = bytes.first().copied() else {
        return Err(invalid("raw token input must be a JSON array of u32 IDs"));
    };
    *bytes = &bytes[1..];
    let mut token = u32::from(first - b'0');
    if first != b'0' {
        while let Some(digit @ b'0'..=b'9') = bytes.first().copied() {
            token = token
                .checked_mul(10)
                .and_then(|value| value.checked_add(u32::from(digit - b'0')))
                .ok_or_else(|| invalid("raw token input must be a JSON array of u32 IDs"))?;
            *bytes = &bytes[1..];
        }
    }
    Ok(token)
}
