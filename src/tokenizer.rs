//! Bounded plain-text encoding for the experimental GPT-2/Qwen35 reference.
//!
//! This composes explicit special-token handling, ordinary-text pretokenization
//! and ranked byte-pair merges. It never applies a chat template; callers
//! supply the complete text they intend to encode. [`Tokenizer::encode`] leaves
//! BOS/EOS insertion disabled, while [`Tokenizer::encode_with_special_tokens`]
//! explicitly opts into validated model declarations.

use bonsai_gguf::Metadata;

use crate::bpe::{Bpe, BpeScratch, MAX_BPE_INPUT_BYTES, MAX_BPE_INPUT_PIECES};
use crate::pretokenizer::Qwen35PreTokenizer;
use crate::special_input::{InputSegment, SpecialInput, SpecialInputPolicy};
use crate::special_tokens::SpecialTokens;
use crate::{Result, invalid};

pub mod gguf;

/// Validated, bounded plain-text encoder without implicit template or insertion.
///
/// Special spellings become token IDs only with [`SpecialInputPolicy::Parse`].
/// Literal mode sends every byte through ordinary pretokenization and BPE.
/// [`Self::encode`] does not act on model-declared insertion flags. Call
/// [`Self::encode_with_special_tokens`] to explicitly apply them to complete
/// input text. Neither method applies a chat template.
#[derive(Debug)]
pub struct Tokenizer {
    vocab_size: usize,
    bpe: Bpe,
    pretokenizer: Qwen35PreTokenizer,
    specials: SpecialInput,
}

impl Tokenizer {
    /// Validate vocabulary, ordered merges and the exact tokenizer identity.
    ///
    /// # Errors
    /// Rejects malformed or unsupported tokenizer metadata, inconsistent graph
    /// vocabulary size, resource limits and allocation failures.
    pub fn from_metadata(metadata: &Metadata, vocab_size: usize) -> Result<Self> {
        let (bpe, vocabulary) = Bpe::from_metadata_with_vocabulary(metadata, vocab_size)?;
        Ok(Self {
            vocab_size,
            bpe,
            pretokenizer: Qwen35PreTokenizer::from_metadata(metadata)?,
            specials: SpecialInput::from_vocabulary(&vocabulary)?,
        })
    }

    /// Encode complete UTF-8 text with an explicit special-token policy.
    ///
    /// At most 1 MiB of input and 65,536 combined special tokens and ordinary
    /// pretokenizer pieces are admitted across the entire request. Ordinary
    /// pieces additionally obey BPE's 64 KiB piece limit. Merges never cross
    /// pretokenizer or recognized special-token boundaries. Empty input returns
    /// no IDs, including with a zero output allowance.
    ///
    /// # Errors
    /// Rejects input/piece/output bounds, missing normal byte tokens and
    /// allocation failures. Errors return no partial token output.
    pub fn encode(
        &self,
        text: &str,
        max_tokens: usize,
        policy: SpecialInputPolicy,
    ) -> Result<Vec<u32>> {
        if text.len() > MAX_BPE_INPUT_BYTES {
            return Err(invalid("tokenizer input exceeds byte limit"));
        }
        let segments = self.specials.split(text, policy, MAX_BPE_INPUT_PIECES)?;
        let mut output = Vec::new();
        output.try_reserve_exact(text.len().min(max_tokens))?;
        let mut scratch = BpeScratch::default();
        let mut piece_count = 0_usize;
        for segment in segments {
            match segment {
                InputSegment::Special(token) => {
                    charge_pieces(&mut piece_count, 1)?;
                    if output.len() == max_tokens {
                        return Err(invalid("tokenizer output exceeds token limit"));
                    }
                    output.push(token);
                }
                InputSegment::Text(text) => {
                    for piece in self.pretokenizer.pieces(text)? {
                        let piece = piece?;
                        charge_pieces(&mut piece_count, 1)?;
                        self.bpe.encode_piece_into(
                            piece.as_bytes(),
                            max_tokens - output.len(),
                            &mut output,
                            &mut scratch,
                        )?;
                    }
                }
            }
        }
        Ok(output)
    }

    /// Encode complete text and explicitly apply model-declared BOS/EOS flags.
    ///
    /// Only flags equal to `Some(true)` insert their declared token. Missing or
    /// false flags insert nothing. BOS precedes the encoded text and EOS follows
    /// it, including for empty input. Parsed special tokens in the text remain
    /// present even when they duplicate an inserted boundary token. The combined
    /// output, including insertions, must fit `max_tokens` exactly.
    ///
    /// `special_tokens` must come from the same model metadata as this tokenizer.
    /// This method verifies matching vocabulary sizes, which checks the ID bounds
    /// but does not establish vocabulary or model identity. Ordinary text and
    /// special-token parsing retain [`Self::encode`]'s input and piece limits.
    ///
    /// # Errors
    /// Rejects mismatched vocabulary sizes, insufficient output allowance, and
    /// all encoding or allocation errors. Errors return no partial output and
    /// leave the tokenizer reusable.
    pub fn encode_with_special_tokens(
        &self,
        text: &str,
        max_tokens: usize,
        policy: SpecialInputPolicy,
        special_tokens: &SpecialTokens,
    ) -> Result<Vec<u32>> {
        if self.vocab_size != special_tokens.vocab_size() {
            return Err(invalid(
                "special tokens and tokenizer vocabulary sizes differ",
            ));
        }
        let beginning = if special_tokens.add_bos_token() == Some(true) {
            special_tokens.bos_token()
        } else {
            None
        };
        let end =
            (special_tokens.add_eos_token() == Some(true)).then_some(special_tokens.eos_token());
        let insertion_count = usize::from(beginning.is_some()) + usize::from(end.is_some());
        let text_limit = max_tokens
            .checked_sub(insertion_count)
            .ok_or_else(|| invalid("tokenizer insertions exceed token limit"))?;
        let mut output = self.encode(text, text_limit, policy)?;
        output.try_reserve_exact(insertion_count)?;
        if let Some(token) = beginning {
            output.insert(0, token);
        }
        if let Some(token) = end {
            output.push(token);
        }
        Ok(output)
    }
}

fn charge_pieces(total: &mut usize, count: usize) -> Result<()> {
    *total = total
        .checked_add(count)
        .filter(|&value| value <= MAX_BPE_INPUT_PIECES)
        .ok_or_else(|| invalid("tokenizer total piece count exceeds limit"))?;
    Ok(())
}
