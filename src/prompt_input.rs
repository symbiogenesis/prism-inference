//! Bounded exact UTF-8 prompt reading for file and service input streams.

use std::io::{self, Read};

use crate::{Result, bpe::MAX_BPE_INPUT_BYTES, invalid};

/// Reads one nonempty prompt through EOF, preserving every valid UTF-8 byte.
///
/// The fixed byte limit is [`MAX_BPE_INPUT_BYTES`], shared with tokenization.
/// Newlines, NUL bytes, a UTF-8 BOM and literal special-token spellings remain
/// unchanged. No trimming, normalization, token insertion or template is applied.
/// At most the limit plus one byte is consumed, including on oversized input.
/// Allocation is fallible and bounded independently of the reader's size hint.
///
/// This synchronous operation has no timeout; callers must arrange any stream
/// cancellation or deadline. Every I/O error, including interruption, returns
/// immediately without retrying or returning a partial prompt. Retrying starts
/// at the reader's current position, so callers must replace or rewind the input
/// when they need the complete prompt again.
///
/// # Errors
/// Rejects empty, oversized or malformed UTF-8 input, allocation failures and
/// I/O errors. I/O error kinds are preserved with a fixed message; errors do not
/// retain prompt bytes or reader-provided messages that might contain them.
pub fn read_prompt(reader: &mut (impl Read + ?Sized)) -> Result<String> {
    let maximum_read = MAX_BPE_INPUT_BYTES + 1;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(maximum_read)?;
    let mut buffer = [0_u8; 8192];
    loop {
        let available = buffer.len().min(maximum_read - bytes.len());
        let count = reader
            .read(&mut buffer[..available])
            .map_err(|error| io::Error::new(error.kind(), "failed to read prompt input"))?;
        if count == 0 {
            break;
        }
        bytes.extend_from_slice(&buffer[..count]);
        if bytes.len() > MAX_BPE_INPUT_BYTES {
            return Err(invalid("prompt input exceeds bounded byte limit"));
        }
    }
    if bytes.is_empty() {
        return Err(invalid("prompt input must not be empty"));
    }
    String::from_utf8(bytes).map_err(|_| invalid("prompt input is not valid UTF-8"))
}
