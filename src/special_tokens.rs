//! Checked model-declared special token IDs and generation stop tokens.
//!
//! This contract reads explicit GGUF declarations. It does not infer special
//! tokens from vocabulary spellings, encode text, or apply a chat template.
//! [`crate::tokenizer::Tokenizer::encode_with_special_tokens`] can explicitly
//! apply its insertion declarations to a complete text input.

use bonsai_gguf::{Metadata, MetadataValue};

use crate::{Result, invalid, runtime::Generation};

const EOS: &str = "tokenizer.ggml.eos_token_id";
const BOS: &str = "tokenizer.ggml.bos_token_id";
const PADDING: &str = "tokenizer.ggml.padding_token_id";
const EOT: &str = "tokenizer.ggml.eot_token_id";
const EOM: &str = "tokenizer.ggml.eom_token_id";
const ADD_BOS: &str = "tokenizer.ggml.add_bos_token";
const ADD_EOS: &str = "tokenizer.ggml.add_eos_token";

/// Validated GGUF declarations for special tokens and model generation stops.
///
/// IDs must use the GGUF `UINT32` type and lie within the decoder vocabulary.
/// EOS is required. Explicit EOT and EOM declarations join EOS in the stop set;
/// aliases are valid and are deduplicated without inventing additional stops.
/// Optional insertion flags preserve absence rather than choosing a tokenizer
/// default. This type does not insert BOS/EOS into caller-provided token input.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpecialTokens {
    vocab_size: usize,
    eos_token: u32,
    bos_token: Option<u32>,
    padding_token: Option<u32>,
    eot_token: Option<u32>,
    eom_token: Option<u32>,
    add_bos_token: Option<bool>,
    add_eos_token: Option<bool>,
    stops: [u32; 3],
    stop_count: usize,
}

impl SpecialTokens {
    /// Read the explicit special-token contract for this decoder vocabulary.
    ///
    /// EOS, optional BOS/padding/EOT/EOM, and optional BOS/EOS insertion flags
    /// are checked even when a caller does not use a particular declaration.
    /// Other tokenizer metadata is outside this contract and is not inspected.
    ///
    /// # Errors
    /// Rejects an empty vocabulary, missing EOS, an incorrectly typed ID or
    /// flag, an out-of-vocabulary ID, or enabled BOS insertion without BOS.
    pub fn from_metadata(metadata: &Metadata, vocab_size: usize) -> Result<Self> {
        if vocab_size == 0 {
            return Err(invalid("special tokens require a nonempty vocabulary"));
        }
        let end_of_sequence = token(metadata, EOS, vocab_size)?
            .ok_or_else(|| invalid(format!("required special token metadata {EOS} is missing")))?;
        let beginning = token(metadata, BOS, vocab_size)?;
        let padding = token(metadata, PADDING, vocab_size)?;
        let end_of_turn = token(metadata, EOT, vocab_size)?;
        let end_of_message = token(metadata, EOM, vocab_size)?;
        let insert_beginning = flag(metadata, ADD_BOS)?;
        let insert_end = flag(metadata, ADD_EOS)?;
        if insert_beginning == Some(true) && beginning.is_none() {
            return Err(invalid(format!("{ADD_BOS} requires {BOS}")));
        }

        let mut stops = [end_of_sequence; 3];
        let mut stop_count = 1;
        for token in [end_of_turn, end_of_message].into_iter().flatten() {
            if !stops[..stop_count].contains(&token) {
                stops[stop_count] = token;
                stop_count += 1;
            }
        }
        Ok(Self {
            vocab_size,
            eos_token: end_of_sequence,
            bos_token: beginning,
            padding_token: padding,
            eot_token: end_of_turn,
            eom_token: end_of_message,
            add_bos_token: insert_beginning,
            add_eos_token: insert_end,
            stops,
            stop_count,
        })
    }

    /// Vocabulary size against which every declared ID was validated.
    ///
    /// Equal sizes establish matching bounds, without proving model identity.
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Required model-declared end-of-sequence token.
    #[must_use]
    pub const fn eos_token(&self) -> u32 {
        self.eos_token
    }

    /// Optional model-declared beginning-of-sequence token.
    #[must_use]
    pub const fn bos_token(&self) -> Option<u32> {
        self.bos_token
    }

    /// Optional model-declared padding token.
    #[must_use]
    pub const fn padding_token(&self) -> Option<u32> {
        self.padding_token
    }

    /// Optional explicitly declared end-of-turn token.
    #[must_use]
    pub const fn eot_token(&self) -> Option<u32> {
        self.eot_token
    }

    /// Optional explicitly declared end-of-message token.
    #[must_use]
    pub const fn eom_token(&self) -> Option<u32> {
        self.eom_token
    }

    /// Optional tokenizer BOS insertion declaration, without an inferred default.
    #[must_use]
    pub const fn add_bos_token(&self) -> Option<bool> {
        self.add_bos_token
    }

    /// Optional tokenizer EOS insertion declaration, without an inferred default.
    ///
    /// A false insertion flag does not remove EOS from generation stops.
    #[must_use]
    pub const fn add_eos_token(&self) -> Option<bool> {
        self.add_eos_token
    }

    /// Deduplicated explicit generation stops, ordered EOS, then EOT, then EOM.
    #[must_use]
    pub fn stop_tokens(&self) -> &[u32] {
        &self.stops[..self.stop_count]
    }

    /// Build bounded generation options from model and caller stop declarations.
    ///
    /// Stops retain EOS, EOT, EOM order, followed by distinct caller IDs in their
    /// original order. BOS/EOS insertion flags do not change the stop policy.
    /// The caller's slice is unchanged. This helper does not insert input tokens
    /// or apply a chat template, and raw [`Generation`] construction remains
    /// available for callers that intentionally choose their own stop policy.
    ///
    /// `max_tokens` must be from one through 262,143: the runtime's maximum
    /// 262,144-token context must also contain a nonempty prompt. The runtime
    /// still validates the actual prompt and its configured context capacity.
    /// These vocabulary bounds must match the runtime's decoder vocabulary.
    ///
    /// # Errors
    /// Rejects a zero or excessive generation bound, an out-of-vocabulary caller
    /// ID, more than 256 distinct combined stops, or an allocation failure.
    pub fn generation(&self, max_tokens: usize, caller_stops: &[u32]) -> Result<Generation> {
        if max_tokens == 0 || max_tokens >= 262_144 {
            return Err(invalid("invalid native generation token bound"));
        }
        let mut stop_tokens = Vec::new();
        stop_tokens.try_reserve_exact(256)?;
        for &token in self.stop_tokens().iter().chain(caller_stops) {
            if usize::try_from(token).map_or(true, |token| token >= self.vocab_size) {
                return Err(invalid("generation stop is outside the decoder vocabulary"));
            }
            if !stop_tokens.contains(&token) {
                if stop_tokens.len() == 256 {
                    return Err(invalid("generation requires at most 256 distinct stops"));
                }
                stop_tokens.push(token);
            }
        }
        Ok(Generation {
            max_tokens,
            stop_tokens,
        })
    }
}

fn token(metadata: &Metadata, key: &str, vocab_size: usize) -> Result<Option<u32>> {
    match metadata.get(key) {
        None => Ok(None),
        Some(MetadataValue::U32(token)) => {
            if usize::try_from(*token).map_or(true, |token| token >= vocab_size) {
                return Err(invalid(format!("{key} is outside the decoder vocabulary")));
            }
            Ok(Some(*token))
        }
        Some(_) => Err(invalid(format!("{key} must have GGUF UINT32 type"))),
    }
}

fn flag(metadata: &Metadata, key: &str) -> Result<Option<bool>> {
    match metadata.get(key) {
        None => Ok(None),
        Some(MetadataValue::Bool(value)) => Ok(Some(*value)),
        Some(_) => Err(invalid(format!("{key} must have GGUF BOOL type"))),
    }
}
