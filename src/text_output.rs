//! Bounded incremental UTF-8 display for caller-owned token bytes.
//!
//! Feed pieces from [`crate::vocabulary::Vocabulary::piece`] in order and retain
//! those original pieces separately when lossless output is required. The
//! returned display chunks can contain replacement characters, so they cannot
//! reconstruct the original bytes. Control spellings and NUL bytes pass through
//! unchanged; this module applies no token, chat-template, or display policy.

use std::borrow::Cow;

use crate::{Result, invalid};

/// Maximum raw bytes accepted by one [`Utf8Output::push`] call.
///
/// This admits every validated vocabulary piece. Larger caller-owned streams
/// must be split into bounded chunks; the display is independent of chunking.
pub const MAX_TEXT_OUTPUT_BYTES: usize = crate::vocabulary::MAX_PIECE_BYTES;

/// Incremental UTF-8 decoder whose display pieces may borrow the caller's input.
pub trait LendingText {
    /// A decoded piece tied to the input borrow.
    type Piece<'a>: AsRef<str>
    where
        Self: 'a;

    /// Decode one bounded input byte chunk and lend its display text.
    ///
    /// # Errors
    /// Propagates input-bound or output-allocation errors from the decoder.
    fn next_text<'a>(&'a mut self, bytes: &'a [u8]) -> Result<Self::Piece<'a>>;
}

/// Decode one bounded byte chunk and pass its borrowed or owned piece to a visitor.
///
/// The source, input and visitor need only support this call's borrow. A visitor
/// may retain a borrowed piece while that borrow remains valid.
///
/// # Errors
/// Propagates decoder or visitor errors.
pub fn push_with<'a, T: LendingText>(
    decoder: &'a mut T,
    bytes: &'a [u8],
    visit: impl FnOnce(T::Piece<'a>) -> Result<()>,
) -> Result<()> {
    let piece = decoder.next_text(bytes)?;
    visit(piece)
}

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

impl LendingText for Utf8Output {
    type Piece<'a> = Cow<'a, str>;

    fn next_text<'a>(&'a mut self, bytes: &'a [u8]) -> Result<Self::Piece<'a>> {
        self.push_borrowed(bytes)
    }
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

    /// Displays a bounded byte chunk, borrowing valid UTF-8 directly when no
    /// incomplete prefix is pending.
    ///
    /// When `pending_len == 0` and `bytes` is valid UTF-8 (or valid UTF-8 up to
    /// an incomplete trailing scalar suffix), returns [`Cow::Borrowed`] from
    /// `bytes` without allocating. Falls back to [`Self::push`] only when bridging
    /// a retained suffix or substituting replacement characters.
    ///
    /// # Errors
    /// Rejects input exceeding [`MAX_TEXT_OUTPUT_BYTES`] and output-allocation
    /// failures while leaving any pending suffix unchanged.
    pub fn push_borrowed<'a>(&mut self, bytes: &'a [u8]) -> Result<Cow<'a, str>> {
        if bytes.len() > MAX_TEXT_OUTPUT_BYTES {
            return Err(invalid("text output chunk exceeds bounded byte limit"));
        }
        if self.pending_len == 0 {
            match std::str::from_utf8(bytes) {
                Ok(text) => return Ok(Cow::Borrowed(text)),
                Err(error) if error.error_len().is_none() => {
                    let (valid, suffix) = bytes.split_at(error.valid_up_to());
                    let text = std::str::from_utf8(valid)?;
                    self.pending_len = suffix.len();
                    self.pending[..self.pending_len].copy_from_slice(suffix);
                    return Ok(Cow::Borrowed(text));
                }
                Err(_) => {}
            }
        }
        self.push(bytes).map(Cow::Owned)
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

#[cfg(test)]
mod tests {
    use super::{Cow, LendingText, MAX_TEXT_OUTPUT_BYTES, Utf8Output, push_with};
    use crate::{Result, invalid};

    #[test]
    fn visitor_can_retain_a_piece_from_a_nonstatic_source() -> Result<()> {
        struct BorrowedText<'a>(&'a str);

        impl LendingText for BorrowedText<'_> {
            type Piece<'a>
                = &'a str
            where
                Self: 'a;

            fn next_text<'a>(&'a mut self, _bytes: &'a [u8]) -> Result<Self::Piece<'a>> {
                Ok(self.0)
            }
        }

        let text = String::from("borrowed source");
        let mut decoder = BorrowedText(&text);
        let mut retained = None;
        push_with(&mut decoder, b"", |piece| {
            retained = Some(piece);
            Ok(())
        })?;
        assert_eq!(retained, Some(text.as_str()));
        push_with(&mut decoder, b"", |piece| {
            assert_eq!(piece, text);
            Ok(())
        })
    }

    #[test]
    fn visitor_errors_propagate_and_decoder_errors_skip_the_visitor() -> Result<()> {
        let mut decoder = Utf8Output::new();
        let _ = decoder.push_borrowed(b"\xe9\x9b")?;
        let oversized = vec![b'a'; MAX_TEXT_OUTPUT_BYTES + 1];
        let mut visited = false;
        assert!(
            push_with(&mut decoder, &oversized, |_| {
                visited = true;
                Ok(())
            })
            .is_err()
        );
        assert!(!visited);
        assert_eq!(decoder.pending_bytes(), b"\xe9\x9b");
        let error = push_with(&mut decoder, b"\xa8", |_| Err(invalid("visitor rejected")));
        assert_eq!(
            error.map_err(|error| error.to_string()),
            Err("visitor rejected".to_owned())
        );
        assert_eq!(decoder.pending_bytes(), b"");
        Ok(())
    }

    #[test]
    fn push_borrowed_lends_valid_prefixes_and_owns_bridged_or_lossy_chunks() -> Result<()> {
        let mut decoder = Utf8Output::new();
        let plain = decoder.push_borrowed(b"hello")?;
        assert!(matches!(plain, Cow::Borrowed("hello")));

        // Split "雨" (E9 9B A8) across two pieces: prefix "Rain " + E9 9B, then A8 + "!".
        let prefix = decoder.push_borrowed(b"Rain \xe9\x9b")?;
        assert!(matches!(prefix, Cow::Borrowed("Rain ")));
        assert_eq!(decoder.pending_bytes(), b"\xe9\x9b");

        let mut bridged = String::new();
        push_with(&mut decoder, b"\xa8!", |piece| {
            assert!(matches!(piece, Cow::Owned(_)));
            bridged.push_str(piece.as_ref());
            Ok(())
        })?;
        assert_eq!(bridged, "雨!");
        assert_eq!(decoder.pending_bytes(), b"");
        assert_eq!(decoder.finish(), "");
        Ok(())
    }
}
