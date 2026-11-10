//! Native Rust CPU reference execution and bounded token conversation ownership.
//!
//! This experimental engine uses the pinned Prism numerical contract. It has no
//! C/C++ inference dependency. Full-model parity, chat-template integration and
//! target-device performance qualification remain separate acceptance gates.

#![forbid(unsafe_code)]

pub mod backend;
pub mod bpe;
#[cfg(feature = "cli")]
pub mod capabilities;
#[cfg(feature = "cli")]
pub mod chat;
pub mod codec;
pub mod grammar;
pub mod graph;
#[cfg(feature = "cli")]
pub mod jinja;
pub mod kernels;
pub mod logging;
pub mod media;
pub mod model;
pub mod optimization;
#[cfg(feature = "cli")]
pub mod peg;
pub mod pretokenizer;
pub mod prompt_input;
#[cfg(feature = "cli")]
pub mod resolution;
pub mod runtime;
pub mod sampling;
#[cfg(feature = "cli")]
pub mod schema;
pub mod special_input;
pub mod special_tokens;
pub mod speculative;
#[cfg(feature = "cli")]
pub mod swift_chat;
#[cfg(feature = "cli")]
pub mod swift_client;
#[cfg(feature = "cli")]
pub mod swift_session;
pub mod text_output;
pub mod text_session;
#[cfg(feature = "cli")]
pub mod token_client;
pub mod token_input;
#[cfg(feature = "cli")]
pub mod token_worker;
pub mod tokenization;
pub mod tokenizer;
pub mod unicode;
pub mod vocabulary;

/// A rejected model, numerical contract, resource limit, or I/O operation.
pub type Error = Box<dyn std::error::Error + Send + Sync>;
/// A checked native runtime operation.
pub type Result<T> = std::result::Result<T, Error>;

/// An encoded token sequence exceeds its explicit caller-supplied output capacity.
///
/// This value-free marker distinguishes token output accounting from malformed
/// tokenizers, byte/piece safety limits and allocation failures. A session may
/// classify it as context rejection only when that capacity is derived from its
/// validated prompt-plus-output context budget. Existing boxed results retain
/// the concrete type for checked downcasting; display text is not a protocol.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenLimitExceeded;

impl std::fmt::Display for TokenLimitExceeded {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("token output exceeds token limit")
    }
}

impl std::error::Error for TokenLimitExceeded {}

/// Maximum number of state-only prompt tokens admitted in one prefill block.
/// This bounds scheduling granularity, without implying a batched graph kernel.
pub const MAX_PREFILL_BLOCK_TOKENS: usize = 256;

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()).into()
}

/// Incremental token decoder with complete, independently owned checkpoints.
///
/// An evaluation must include exactly one token in the state and return the next
/// token logits. Errors may mutate state: [`runtime::Runtime`] restores its
/// transaction checkpoint before allowing another request. A checkpoint includes
/// all recurrent/convolution/KV state and position, never references mutable
/// decoder storage, and is valid only for its originating decoder identity.
pub trait Decoder {
    /// Complete state snapshot, including the current sequence position.
    type Checkpoint;

    /// Stable model, layout and cache-contract identity.
    fn identity(&self) -> &str;
    /// Number of model output tokens.
    fn vocab_size(&self) -> usize;
    /// Conservative maximum bytes allocated when capturing the current state.
    fn checkpoint_bytes(&self) -> usize;
    /// Check the maximum remaining forward work before starting new inference.
    ///
    /// `ingest_tokens` excludes output projections; `evaluate_tokens` includes
    /// logits and the complete reserved output length, even if a stop token may
    /// finish early. Implementations with finite device sessions must reject
    /// work that cannot fit their remaining operation budget. This check does
    /// not submit work, change sequence state or replenish a device lifetime.
    /// CPU decoders have no additional operation limit by default.
    ///
    /// # Errors
    /// Rejects exhausted resources, arithmetic overflow or a closed decoder.
    fn admit_work(&self, _ingest_tokens: usize, _evaluate_tokens: usize) -> Result<()> {
        Ok(())
    }
    /// Clear all sequence state without changing model identity.
    ///
    /// # Errors
    /// Returns an error if resetting cannot establish an empty sequence.
    fn reset(&mut self) -> Result<()>;
    /// Include one token in the sequence and compute the next-token logits.
    ///
    /// # Errors
    /// Rejects invalid tokens, exhausted context, numerical and allocation errors.
    fn evaluate(&mut self, token: u32) -> Result<Vec<f32>>;
    /// Include one token and replace a caller-owned reusable logits buffer.
    ///
    /// The caller must finish consuming the previous output before this call.
    /// On success the buffer contains exactly [`Self::vocab_size`] finite logits;
    /// on error it is empty. Implementations may retain independent graph scratch,
    /// but checkpoints must never borrow either scratch or this output storage.
    /// The compatibility implementation copies the owned evaluation result;
    /// decoders can override it to write directly into the retained allocation.
    ///
    /// # Errors
    /// Returns evaluation or output-allocation errors. Decoder state may have
    /// changed on failure and requires transaction restoration before reuse.
    fn evaluate_into(&mut self, token: u32, output: &mut Vec<f32>) -> Result<()> {
        output.clear();
        let logits = self.evaluate(token)?;
        if logits.len() != self.vocab_size() || logits.iter().any(|value| !value.is_finite()) {
            return Err(invalid("native decoder returned invalid logits"));
        }
        output.try_reserve_exact(logits.len())?;
        output.extend_from_slice(&logits);
        Ok(())
    }
    /// Include one token and borrow its checked finite output from reusable storage.
    ///
    /// The default independently checks the complete output of
    /// [`Self::evaluate_into`], including custom implementations of that method.
    /// Native decoders that already validate their complete output may return
    /// the same immutable proof without repeating the vocabulary traversal.
    /// The view borrows only `output`, preventing its mutation until the last
    /// use of the view; checkpoints remain independent of that allocation.
    /// The runtime still checks the view's storage identity, complete extent
    /// and current vocabulary size before publishing the token.
    ///
    /// # Errors
    /// Returns evaluation errors or rejects wrong-size, empty, oversized or
    /// nonfinite output. The compatibility implementation empties `output` on
    /// any error. Decoder state may require transaction restoration.
    fn evaluate_finite_into<'a>(
        &mut self,
        token: u32,
        output: &'a mut Vec<f32>,
    ) -> Result<sampling::FiniteLogits<'a>> {
        if let Err(error) = self.evaluate_into(token, output) {
            output.clear();
            return Err(error);
        }
        if output.len() != self.vocab_size()
            || output.is_empty()
            || output.len() > 1_048_576
            || !kernels::is_finite_slice(output)
        {
            output.clear();
            return Err(invalid("native decoder returned invalid logits"));
        }
        sampling::FiniteLogits::from_validated(output)
    }
    /// Include one token in sequence state without retaining next-token logits.
    ///
    /// Success advances exactly the same recurrent/convolution/KV state and
    /// position as [`Self::evaluate_into`]. An override may skip output-only
    /// normalization and vocabulary projection, but must still validate the
    /// token, context bounds and every computed state value. Checkpoints must
    /// remain complete and independent of scratch storage.
    ///
    /// The compatibility implementation calls [`Self::evaluate_into`] and
    /// discards its output only after independently checking that it contains
    /// exactly [`Self::vocab_size`] finite logits. Thus an invalid output from
    /// an existing decoder remains an error even when no logits are needed.
    ///
    /// # Errors
    /// Rejects invalid tokens, exhausted context, numerical and allocation
    /// errors. State may have changed on failure and requires transaction
    /// restoration before reuse.
    fn ingest(&mut self, token: u32) -> Result<()> {
        let mut logits = Vec::new();
        self.evaluate_into(token, &mut logits)?;
        if logits.len() != self.vocab_size() || logits.iter().any(|value| !value.is_finite()) {
            return Err(invalid("native decoder returned invalid logits"));
        }
        Ok(())
    }
    /// Include a bounded, nonempty causal prompt block without retaining logits.
    ///
    /// The reference implementation calls [`Self::ingest`] in original token
    /// order, preserving its exact numerical operations. It checks `check`
    /// before and after every token, including the last, so cancellation and
    /// request deadlines remain cooperative at token boundaries. This method
    /// establishes a scheduling boundary; it does not implement tiled matmul,
    /// parallel attention or chunkwise recurrent kernels.
    ///
    /// Overrides must preserve complete state/position equivalence to sequential
    /// ingestion, honor checks between tokens and reject empty, oversized or
    /// invalid-token blocks before changing decoder state. An error may leave a
    /// partial block in the decoder; the owning runtime restores the whole
    /// request checkpoint before allowing reuse.
    ///
    /// # Errors
    /// Rejects blocks beyond [`MAX_PREFILL_BLOCK_TOKENS`], invalid tokens,
    /// cancellation/deadline checks, and any underlying ingestion error.
    fn ingest_block(&mut self, tokens: &[u32], check: &dyn Fn() -> Result<()>) -> Result<()> {
        if tokens.is_empty()
            || tokens.len() > MAX_PREFILL_BLOCK_TOKENS
            || tokens
                .iter()
                .any(|&token| !usize::try_from(token).is_ok_and(|value| value < self.vocab_size()))
        {
            return Err(invalid("invalid native prefill block bounds or tokens"));
        }
        for &token in tokens {
            check()?;
            self.ingest(token)?;
            check()?;
        }
        Ok(())
    }
    /// Capture every mutable component needed for exact prefix restoration.
    ///
    /// # Errors
    /// Returns an error if the snapshot cannot be allocated.
    fn checkpoint(&self) -> Result<Self::Checkpoint>;
    /// Restore a snapshot from this exact model and cache contract.
    ///
    /// # Errors
    /// Rejects foreign or malformed snapshots and allocation failures.
    fn restore(&mut self, checkpoint: &Self::Checkpoint) -> Result<()>;
}
