//! Bounded, byte-preserving output pieces for the pinned GPT-2/Qwen35 vocabulary.
//!
//! This module only reverses the serialized byte alphabet. It does not tokenize
//! prompts, apply merges or chat templates, or decide which generated IDs stop a
//! request. A piece can end partway through a UTF-8 sequence; callers must retain
//! those bytes across token boundaries before interpreting text.

use bonsai_gguf::{Metadata, MetadataArray};

use crate::{Result, invalid};

/// Maximum admitted output vocabulary, independent of untrusted metadata counts.
pub const MAX_VOCABULARY_TOKENS: usize = 1_000_000;
/// Maximum retained decoded piece bytes, excluding the bounded entry table.
pub const MAX_VOCABULARY_BYTES: usize = 64 * 1024 * 1024;
/// Maximum decoded bytes in one output piece.
pub const MAX_PIECE_BYTES: usize = 1024 * 1024;

#[derive(Clone, Copy, Debug)]
enum TokenType {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl TokenType {
    fn from_i32(value: i32) -> Result<Self> {
        match value {
            1 => Ok(Self::Normal),
            2 => Ok(Self::Unknown),
            3 => Ok(Self::Control),
            4 => Ok(Self::UserDefined),
            5 => Ok(Self::Unused),
            6 => Ok(Self::Byte),
            _ => Err(invalid(format!(
                "unsupported vocabulary token type {value}"
            ))),
        }
    }
}

#[derive(Debug)]
struct Entry {
    end: usize,
    kind: TokenType,
}

/// Validated output pieces with one bounded, contiguous byte allocation.
///
/// Normal pieces use the inverse GPT-2 byte alphabet. Control and user-defined
/// pieces retain literal UTF-8 bytes; byte pieces use strict `<0xHH>` notation.
/// Unknown and unused IDs are retained but cannot be rendered. Control pieces
/// are never silently omitted: [`Self::is_control`] lets the caller explicitly
/// choose its display policy without guessing from a token's spelling.
#[derive(Debug)]
pub struct Vocabulary {
    entries: Vec<Entry>,
    bytes: Vec<u8>,
}

/// Fully checked metadata, without allocating decoded output pieces.
///
/// The private slices have equal, bounded lengths and every renderable piece
/// obeys the same spelling and decoded-size contract as [`Vocabulary`].
pub(crate) struct ValidatedVocabulary<'a> {
    tokens: &'a [String],
    types: &'a [i32],
    decoded_bytes: usize,
}

impl<'a> ValidatedVocabulary<'a> {
    pub(crate) fn from_metadata(metadata: &'a Metadata, vocab_size: usize) -> Result<Self> {
        for (key, expected) in [
            ("tokenizer.ggml.model", "gpt2"),
            ("tokenizer.ggml.pre", "qwen35"),
        ] {
            if metadata.get(key).and_then(|value| value.as_str()) != Some(expected) {
                return Err(invalid(format!("vocabulary requires {key}={expected}")));
            }
        }
        if vocab_size == 0 || vocab_size > MAX_VOCABULARY_TOKENS {
            return Err(invalid("vocabulary size exceeds supported bounds"));
        }
        let Some(MetadataArray::String(tokens)) = metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|value| value.as_array())
        else {
            return Err(invalid("vocabulary requires a STRING token array"));
        };
        let Some(MetadataArray::I32(types)) = metadata
            .get("tokenizer.ggml.token_type")
            .and_then(|value| value.as_array())
        else {
            return Err(invalid("vocabulary requires an INT32 token type array"));
        };
        if tokens.len() != vocab_size || types.len() != vocab_size {
            return Err(invalid("vocabulary counts do not match graph output size"));
        }
        let mut required = 0_usize;
        for (token, &kind) in tokens.iter().zip(types) {
            let length = piece_length(token, TokenType::from_i32(kind)?)?;
            required = required
                .checked_add(length)
                .filter(|&bytes| bytes <= MAX_VOCABULARY_BYTES)
                .ok_or_else(|| invalid("vocabulary exceeds decoded byte limit"))?;
        }
        Ok(Self {
            tokens,
            types,
            decoded_bytes: required,
        })
    }

    pub(crate) const fn len(&self) -> usize {
        self.tokens.len()
    }

    pub(crate) fn entries(&self) -> impl ExactSizeIterator<Item = (&'a str, i32)> {
        self.tokens
            .iter()
            .map(String::as_str)
            .zip(self.types.iter().copied())
    }
}

impl Vocabulary {
    /// Validates the pinned `gpt2` / `qwen35` output vocabulary against the graph.
    ///
    /// Requires string tokens and GGUF `INT32` token types with exactly
    /// `vocab_size` elements each. Construction validates every renderable piece
    /// before reserving decoded storage. At most [`MAX_VOCABULARY_TOKENS`]
    /// entries, [`MAX_VOCABULARY_BYTES`] decoded bytes, and [`MAX_PIECE_BYTES`]
    /// bytes per piece are admitted. The entry table additionally retains at
    /// most `MAX_VOCABULARY_TOKENS * size_of::<Entry>()` bytes.
    ///
    /// # Errors
    /// Rejects unsupported tokenizer metadata, graph/count mismatches, invalid
    /// token types, malformed byte alphabet or byte notation, empty renderable
    /// pieces, resource limits, and allocation failures.
    pub fn from_metadata(metadata: &Metadata, vocab_size: usize) -> Result<Self> {
        let vocabulary = ValidatedVocabulary::from_metadata(metadata, vocab_size)?;
        let mut entries = Vec::new();
        entries.try_reserve_exact(vocabulary.len())?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(vocabulary.decoded_bytes)?;
        for (token, kind) in vocabulary.entries() {
            let kind = TokenType::from_i32(kind)?;
            match kind {
                TokenType::Normal => {
                    for character in token.chars() {
                        bytes.push(decode_character(character)?);
                    }
                }
                TokenType::Control | TokenType::UserDefined => {
                    bytes.extend_from_slice(token.as_bytes());
                }
                TokenType::Byte => bytes.push(decode_byte(token)?),
                TokenType::Unknown | TokenType::Unused => {}
            }
            entries.push(Entry {
                end: bytes.len(),
                kind,
            });
        }
        Ok(Self { entries, bytes })
    }

    /// Number of output IDs, exactly matching the validated graph output size.
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.entries.len()
    }

    /// Borrows an output piece without interpreting UTF-8 or omitting controls.
    ///
    /// # Errors
    /// Rejects out-of-range, unknown and unused IDs. A returned byte slice can
    /// contain any byte, including NUL or an incomplete UTF-8 sequence.
    pub fn piece(&self, id: u32) -> Result<&[u8]> {
        let index = usize::try_from(id)?;
        let entry = self.entry(index)?;
        if matches!(entry.kind, TokenType::Unknown | TokenType::Unused) {
            return Err(invalid(format!("vocabulary token {id} has no text piece")));
        }
        let start = index
            .checked_sub(1)
            .map_or(0, |previous| self.entries[previous].end);
        Ok(&self.bytes[start..entry.end])
    }

    /// Reports the declared control type, without inspecting the token text.
    ///
    /// # Errors
    /// Rejects IDs outside the validated output vocabulary.
    pub fn is_control(&self, id: u32) -> Result<bool> {
        Ok(matches!(
            self.entry(usize::try_from(id)?)?.kind,
            TokenType::Control
        ))
    }

    fn entry(&self, index: usize) -> Result<&Entry> {
        self.entries
            .get(index)
            .ok_or_else(|| invalid("output token ID exceeds validated vocabulary"))
    }
}

fn piece_length(token: &str, kind: TokenType) -> Result<usize> {
    let length = match kind {
        TokenType::Normal => {
            let mut length = 0_usize;
            for character in token.chars() {
                decode_character(character)?;
                length += 1;
                if length > MAX_PIECE_BYTES {
                    return Err(invalid("vocabulary piece exceeds decoded byte limit"));
                }
            }
            length
        }
        TokenType::Control | TokenType::UserDefined => token.len(),
        TokenType::Byte => {
            decode_byte(token)?;
            1
        }
        TokenType::Unknown | TokenType::Unused => return Ok(0),
    };
    if length == 0 || length > MAX_PIECE_BYTES {
        return Err(invalid("vocabulary piece has invalid decoded length"));
    }
    Ok(length)
}

pub(crate) fn decode_character(character: char) -> Result<u8> {
    let scalar = u32::from(character);
    let byte = match scalar {
        33..=126 | 161..=172 | 174..=255 => scalar,
        256..=288 => scalar - 256,
        289..=322 => scalar - 162,
        323 => 173,
        _ => return Err(invalid("normal token contains a non-GPT-2 byte character")),
    };
    Ok(u8::try_from(byte)?)
}

fn decode_byte(token: &str) -> Result<u8> {
    let Some(hex) = token
        .strip_prefix("<0x")
        .and_then(|value| value.strip_suffix('>'))
        .filter(|value| value.len() == 2 && value.bytes().all(|byte| byte.is_ascii_hexdigit()))
    else {
        return Err(invalid("byte token requires exact <0xHH> notation"));
    };
    Ok(u8::from_str_radix(hex, 16)?)
}
