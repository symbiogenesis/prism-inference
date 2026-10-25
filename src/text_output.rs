//! Bounded incremental UTF-8 display for caller-owned token bytes.
//!
//! Feed pieces from [`crate::vocabulary::Vocabulary::piece`] in order and retain
//! those original pieces separately when lossless output is required. The
//! returned display chunks can contain replacement characters, so they cannot
//! reconstruct the original bytes. Control spellings and NUL bytes pass through
//! unchanged; this module applies no token, chat-template, or display policy.

use crate::{Result, invalid};

/// Maximum raw bytes accepted by one [`Utf8Output::push`] call.
///
/// This admits every validated vocabulary piece. Larger caller-owned streams
/// must be split into bounded chunks; the display is independent of chunking.
pub const MAX_TEXT_OUTPUT_BYTES: usize = crate::vocabulary::MAX_PIECE_BYTES;

/// Incremental lossy UTF-8 display with at most three retained input bytes.
///
/// Concatenating every successful [`Self::push`] result and the final
/// [`Self::finish`] result exactly matches [`String::from_utf8_lossy`] applied
/// to the concatenated input. Incomplete characters wait for subsequent input;
/// malformed sequences produce the standard replacement character. No complete
/// input chunk or output allocation is retained by this value.
///
/// Call [`Self::finish`] at a successful end of stream, or [`Self::reset`] when
/// abandoning provisional output after cancellation or starting a new request.
/// Both operations leave an empty decoder that can be reused immediately.
#[derive(Clone, Debug, Default)]
pub struct Utf8Output {
    pending: [u8; 3],
    pending_len: usize,
}

impl Utf8Output {
    /// Creates an empty display decoder without allocating.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pending: [0; 3],
            pending_len: 0,
        }
    }

    /// Displays a bounded byte chunk, deferring only an incomplete UTF-8 suffix.
    ///
    /// Input is borrowed, never changed, and need not remain alive after this
    /// call. The returned string is independently owned. Its length is at most
    /// three times the input length plus nine bytes for the retained suffix.
    /// Empty input leaves any pending suffix intact.
    ///
    /// # Errors
    /// Rejects input exceeding [`MAX_TEXT_OUTPUT_BYTES`] and output-allocation
    /// failures. Every error leaves the pending suffix unchanged, so callers
    /// may retry with smaller input or reset before starting another stream.
    pub fn push(&mut self, bytes: &[u8]) -> Result<String> {
        if bytes.len() > MAX_TEXT_OUTPUT_BYTES {
            return Err(invalid("text output chunk exceeds bounded byte limit"));
        }
        let mut output = String::new();
        output.try_reserve(bytes.len())?;
        let mut pending = self.pending;
        let mut pending_len = self.pending_len;
        let mut remaining = bytes;
        while pending_len != 0 && !remaining.is_empty() {
            // Four bytes suffice to decide the first pending scalar without
            // copying or retaining an entire vocabulary piece.
            let mut bridge = [0; 4];
            bridge[..pending_len].copy_from_slice(&pending[..pending_len]);
            let count = (bridge.len() - pending_len).min(remaining.len());
            bridge[pending_len..pending_len + count].copy_from_slice(&remaining[..count]);
            let suffix = append_complete(&bridge[..pending_len + count], &mut output)?;
            pending_len = suffix.len();
            pending[..pending_len].copy_from_slice(suffix);
            remaining = &remaining[count..];
        }
        if pending_len == 0 {
            let suffix = append_complete(remaining, &mut output)?;
            pending_len = suffix.len();
            pending[..pending_len].copy_from_slice(suffix);
        }
        self.pending = pending;
        self.pending_len = pending_len;
        Ok(output)
    }

    /// Flushes a truncated suffix as one replacement character and resets.
    ///
    /// Returns an empty string when no suffix remains. Repeated calls are safe
    /// and allocation-free; the decoder accepts a new stream after this call.
    #[must_use]
    pub const fn finish(&mut self) -> &'static str {
        let truncated = self.pending_len != 0;
        self.reset();
        if truncated { "\u{fffd}" } else { "" }
    }

    /// Discards provisional suffix bytes without producing display output.
    ///
    /// Use this when cancelling a request so its incomplete character cannot
    /// combine with bytes from the next request. This does not retract display
    /// chunks already returned to the caller.
    pub const fn reset(&mut self) {
        self.pending = [0; 3];
        self.pending_len = 0;
    }

    /// Borrows the incomplete suffix, always at most three input bytes.
    #[must_use]
    pub fn pending_bytes(&self) -> &[u8] {
        &self.pending[..self.pending_len]
    }
}

fn append_complete<'a>(mut bytes: &'a [u8], output: &mut String) -> Result<&'a [u8]> {
    while !bytes.is_empty() {
        match std::str::from_utf8(bytes) {
            Ok(text) => {
                output.try_reserve(text.len())?;
                output.push_str(text);
                return Ok(&[]);
            }
            Err(error) => {
                let (valid, suffix) = bytes.split_at(error.valid_up_to());
                let text = std::str::from_utf8(valid)?;
                output.try_reserve(text.len())?;
                output.push_str(text);
                if let Some(invalid_bytes) = error.error_len() {
                    output.try_reserve('\u{fffd}'.len_utf8())?;
                    output.push('\u{fffd}');
                    bytes = &suffix[invalid_bytes..];
                } else {
                    return Ok(suffix);
                }
            }
        }
    }
    Ok(bytes)
}
