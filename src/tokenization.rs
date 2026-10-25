//! Reusable artifact-verified plain-text tokenization and context accounting.
//!
//! Loading verifies one complete model artifact, constructs the tokenizer and
//! special-token declarations from that same metadata, then drops the metadata
//! and file handle. Requests reuse only the validated tokenizer. No weights,
//! executable graph, chat template or broker service are constructed here.

use std::path::Path;

use crate::model::ModelMetadata;
use crate::special_input::SpecialInputPolicy;
use crate::special_tokens::SpecialTokens;
use crate::tokenizer::Tokenizer;
use crate::{Result, invalid};

/// Largest context admitted by the native text/token input contract.
pub const MAX_TOKENIZATION_CONTEXT: usize = 262_144;

/// Explicit context size and output allowance for a complete prompt.
///
/// Output reservation is accounting only; tokenization never generates tokens.
/// At least one context slot must remain available for prompt input, although
/// empty text and empty raw input remain valid token-count requests.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenBudget {
    /// Total prompt-plus-output context allowance, from 1 to 262,144 tokens.
    pub context: usize,
    /// Context slots reserved for later output, strictly less than `context`.
    pub reserved_output_tokens: usize,
}

impl TokenBudget {
    /// Validate the budget and return the number of available prompt slots.
    ///
    /// # Errors
    /// Rejects a zero or oversized context and output reservation that leaves
    /// no prompt slot. Validation uses no input text and performs no encoding.
    pub fn prompt_token_limit(self) -> Result<usize> {
        if !(1..=MAX_TOKENIZATION_CONTEXT).contains(&self.context)
            || self.reserved_output_tokens >= self.context
        {
            return Err(invalid(
                "invalid tokenization context or output reservation",
            ));
        }
        Ok(self.context - self.reserved_output_tokens)
    }
}

/// Explicit handling of model-declared BOS/EOS insertion flags for text input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpecialTokenInsertion {
    /// Insert no boundary tokens, independently of model declarations.
    None,
    /// Apply the verified model's explicit BOS/EOS insertion declarations.
    Model,
}

/// Complete text-tokenization policy, with no implicit special-token behavior.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TextTokenizationPolicy {
    /// Whether declared special spellings are literal text or parsed token IDs.
    pub special_input: SpecialInputPolicy,
    /// Whether to insert model-declared boundary tokens around complete text.
    pub insert_special: SpecialTokenInsertion,
    /// Shared prompt/context/output accounting contract.
    pub budget: TokenBudget,
}

/// Exact admitted prompt IDs and their context/output accounting.
///
/// Insertion, when requested, is included in `token_ids` and the prompt count.
/// Raw-ID requests preserve the caller's sequence without parsing or insertion.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizationResult {
    /// Complete prompt token sequence, in original encoding or raw-input order.
    pub token_ids: Vec<u32>,
    /// Total context allowance used to validate this prompt.
    pub context: usize,
    /// Context slots reserved for output when this prompt was admitted.
    pub reserved_output_tokens: usize,
}

impl TokenizationResult {
    /// Number of prompt tokens, including explicitly requested insertions.
    #[must_use]
    pub const fn prompt_token_count(&self) -> usize {
        self.token_ids.len()
    }
}

/// Validated tokenizer and special declarations from one verified artifact.
///
/// The retained session owns no model metadata, weight tensors or file handle.
/// Encoding takes `&self`, so a failed request leaves the session reusable.
/// Artifact verification establishes identity, not full-model tokenizer parity
/// or inference validity; only the supported `gpt2` / `qwen35` contract is used.
#[derive(Debug)]
pub struct TokenizationSession {
    identity: String,
    vocab_size: usize,
    tokenizer: Tokenizer,
    special_tokens: SpecialTokens,
}

impl TokenizationSession {
    /// Verify an artifact once and prepare its bounded plain-text tokenizer.
    ///
    /// Both tokenizer and special declarations are built from the same verified
    /// metadata, including when later requests disable parsing and insertion.
    /// `memory_limit` bounds retained GGUF header storage while loading; it is
    /// not a process-memory or retained-tokenizer allocation budget. Tokenizer
    /// construction enforces its own documented resource limits.
    ///
    /// # Errors
    /// Rejects artifact identity, I/O, header-budget and malformed/unsupported
    /// tokenizer or special-token declarations. Does not load executable weights.
    pub fn load(
        path: &Path,
        expected_bytes: u64,
        expected_sha256: &str,
        memory_limit: usize,
    ) -> Result<Self> {
        let model = ModelMetadata::load(path, expected_bytes, expected_sha256, memory_limit)?;
        Self::from_metadata(&model)
    }

    /// Prepare tokenization from metadata whose artifact was already verified.
    ///
    /// This avoids hashing or opening the model again when a caller also owns
    /// its executable weights. The caller retains responsibility for the
    /// provenance of the supplied metadata and identity; this constructor
    /// validates the tokenizer and special declarations, not artifact bytes.
    ///
    /// # Errors
    /// Rejects malformed or unsupported tokenizer/special metadata and bounded
    /// tokenizer allocation failures.
    pub fn from_metadata(model: &ModelMetadata) -> Result<Self> {
        Self::from_parts(&model.metadata, model.vocab_size, &model.identity)
    }

    pub(crate) fn from_parts(
        metadata: &bonsai_gguf::Metadata,
        vocab_size: usize,
        identity: &str,
    ) -> Result<Self> {
        let special_tokens = SpecialTokens::from_metadata(metadata, vocab_size)?;
        let tokenizer = Tokenizer::from_metadata(metadata, vocab_size)?;
        Ok(Self {
            identity: identity.to_owned(),
            vocab_size,
            tokenizer,
            special_tokens,
        })
    }

    /// Lowercase SHA-256 identity verified at session construction.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Verified embedding vocabulary size used for all token-ID validation.
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Validated explicit special declarations from this session's artifact.
    #[must_use]
    pub const fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    /// Encode exact caller-provided text under an explicit complete policy.
    ///
    /// The context/output budget is validated before inspecting or encoding
    /// text. No template, normalization or hidden prefix/suffix is applied.
    /// Output includes requested insertions and must fit the remaining prompt
    /// allowance. Empty text is supported, including model-declared insertions.
    ///
    /// # Errors
    /// Rejects invalid budgets, tokenizer resource/input/output limits and
    /// allocation errors. Errors contain no partial token result.
    pub fn encode(&self, text: &str, policy: TextTokenizationPolicy) -> Result<TokenizationResult> {
        let limit = policy.budget.prompt_token_limit()?;
        let tokens = match policy.insert_special {
            SpecialTokenInsertion::None => self.tokenizer.encode(text, limit, policy.special_input),
            SpecialTokenInsertion::Model => self.tokenizer.encode_with_special_tokens(
                text,
                limit,
                policy.special_input,
                &self.special_tokens,
            ),
        }?;
        Ok(Self::result(tokens, policy.budget))
    }

    /// Validate and copy exact raw IDs using the same context accounting.
    ///
    /// No special parsing or insertion applies to raw IDs. Empty input is valid
    /// for counting; a later generation request may independently require a
    /// nonempty prompt. Bounds and every ID are checked before copying input.
    ///
    /// # Errors
    /// Rejects invalid budgets, excessive prompt length, out-of-vocabulary IDs
    /// and allocation failures. Errors contain no partial token result.
    pub fn check_tokens(&self, tokens: &[u32], budget: TokenBudget) -> Result<TokenizationResult> {
        self.validate_tokens(tokens, budget)?;
        let mut output = Vec::new();
        output.try_reserve_exact(tokens.len())?;
        output.extend_from_slice(tokens);
        Ok(Self::result(output, budget))
    }

    /// Validate borrowed raw IDs without allocating or copying their storage.
    ///
    /// Uses the same context and vocabulary checks as [`Self::check_tokens`].
    /// No text parsing or boundary insertion applies, and empty input is valid.
    /// Callers can count or serialize their original IDs after this succeeds.
    ///
    /// # Errors
    /// Rejects invalid budgets, excessive prompt length and out-of-vocabulary
    /// IDs. No input values or partial token results are included in errors.
    pub fn validate_tokens(&self, tokens: &[u32], budget: TokenBudget) -> Result<()> {
        let limit = budget.prompt_token_limit()?;
        if tokens.len() > limit {
            return Err(invalid("raw input exceeds token limit"));
        }
        if tokens
            .iter()
            .any(|&token| usize::try_from(token).map_or(true, |token| token >= self.vocab_size))
        {
            return Err(invalid("input token is outside the model vocabulary"));
        }
        Ok(())
    }

    const fn result(token_ids: Vec<u32>, budget: TokenBudget) -> TokenizationResult {
        TokenizationResult {
            token_ids,
            context: budget.context,
            reserved_output_tokens: budget.reserved_output_tokens,
        }
    }
}
