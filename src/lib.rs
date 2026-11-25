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
pub mod kv_quant;
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
pub mod vision;
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
    /// Include a bounded, nonempty causal prompt block and compute finite
    /// next-token logits only for its final token in reusable storage.
    ///
    /// The reference implementation splits the block into an optional leading
    /// prefix passed to [`Self::ingest_block`] followed by [`Self::evaluate_into`]
    /// on the last token, honoring `check` around both stages and clearing
    /// `output` on any error. Overrides must preserve complete state/position
    /// equivalence to [`Self::ingest_block`] plus [`Self::evaluate_into`].
    ///
    /// # Errors
    /// Rejects empty or oversized blocks beyond [`MAX_PREFILL_BLOCK_TOKENS`],
    /// invalid tokens, cancellation/deadline checks, and underlying evaluation
    /// errors. On error `output` is empty and state may require restoration.
    fn ingest_block_and_evaluate_last_into(
        &mut self,
        tokens: &[u32],
        output: &mut Vec<f32>,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        if tokens.is_empty()
            || tokens.len() > MAX_PREFILL_BLOCK_TOKENS
            || tokens
                .iter()
                .any(|&token| !usize::try_from(token).is_ok_and(|value| value < self.vocab_size()))
        {
            output.clear();
            return Err(invalid("invalid native prefill block bounds or tokens"));
        }
        let Some((&last, prefix)) = tokens.split_last() else {
            output.clear();
            return Err(invalid("invalid native prefill block bounds or tokens"));
        };
        if !prefix.is_empty()
            && let Err(error) = self.ingest_block(prefix, check)
        {
            output.clear();
            return Err(error);
        }
        if let Err(error) = check() {
            output.clear();
            return Err(error);
        }
        if let Err(error) = self.evaluate_into(last, output) {
            output.clear();
            return Err(error);
        }
        if let Err(error) = check() {
            output.clear();
            return Err(error);
        }
        Ok(())
    }
    /// Include a bounded, nonempty causal prompt block and borrow checked
    /// finite logits for its final token from reusable storage.
    ///
    /// The default validates the complete output produced by
    /// [`Self::ingest_block_and_evaluate_last_into`], clearing `output` on any
    /// error.
    ///
    /// # Errors
    /// Returns ingestion/evaluation errors or rejects wrong-size, empty,
    /// oversized or nonfinite output. On error `output` is empty and state may
    /// require restoration.
    fn ingest_block_and_evaluate_last_finite_into<'a>(
        &mut self,
        tokens: &[u32],
        output: &'a mut Vec<f32>,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<sampling::FiniteLogits<'a>> {
        if let Err(error) = self.ingest_block_and_evaluate_last_into(tokens, output, check) {
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
    /// Maximum token count supported by an explicitly selected, weight-sharing
    /// evaluation path, or zero when that path is unavailable.
    ///
    /// Ordinary decoders retain zero. A nonzero value does not grant new memory
    /// admission: the model owner must reserve all per-position logits and graph
    /// scratch before enabling this capability.
    fn batched_evaluation_capacity(&self) -> usize {
        0
    }
    /// Include a bounded causal block and return next-token logits at every
    /// position using the explicitly admitted weight-sharing implementation.
    ///
    /// On success the final decoder state matches sequential evaluation under
    /// its declared arithmetic contract. Every row must contain exactly the
    /// vocabulary's finite logits. On error `output` is empty and decoder state
    /// may be provisional; the owner must restore its whole-block checkpoint.
    /// The default refuses without changing state, rather than disguising
    /// sequential evaluation as a batched capability.
    ///
    /// # Errors
    /// Rejects unavailable capability, invalid bounds/tokens, cancellation,
    /// context/resource exhaustion, allocation and numerical failures.
    fn evaluate_batch_into(
        &mut self,
        _tokens: &[u32],
        output: &mut Vec<Vec<f32>>,
        _check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        output.clear();
        Err(invalid("batched decoder evaluation is not selected"))
    }
    /// Include a spatially gridded sequence of pre-projected vision embeddings
    /// in decoder state using interleaved 3D `MRoPE` coordinates.
    ///
    /// Implementations that do not support multimodal vision residuals refuse
    /// without mutating decoder state and clear `logits` if supplied.
    ///
    /// # Errors
    /// Rejects unsupported decoders, invalid grid dimensions, mismatched or
    /// nonfinite embeddings, exhausted context, cancellation checks, and
    /// underlying decoder failures.
    fn evaluate_image_chunk(
        &mut self,
        _embeddings: &[f32],
        _grid_width: usize,
        _grid_height: usize,
        logits: Option<&mut Vec<f32>>,
        _check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        if let Some(buffer) = logits {
            buffer.clear();
        }
        Err(invalid(
            "multimodal image-chunk evaluation is not supported by this decoder",
        ))
    }
    /// Capture every mutable component needed for exact prefix restoration.
    ///
    /// # Errors
    /// Returns an error if the snapshot cannot be allocated.
    fn checkpoint(&self) -> Result<Self::Checkpoint>;
    /// Capture every mutable component into an existing snapshot buffer,
    /// reusing compatible allocations when possible.
    ///
    /// # Errors
    /// Returns an error if the snapshot cannot be updated or allocated.
    fn checkpoint_into(&self, target: &mut Self::Checkpoint) -> Result<()> {
        *target = self.checkpoint()?;
        Ok(())
    }
    /// Restore a snapshot from this exact model and cache contract.
    ///
    /// # Errors
    /// Rejects foreign or malformed snapshots and allocation failures.
    fn restore(&mut self, checkpoint: &Self::Checkpoint) -> Result<()>;

    /// Return whether this decoder can fold accepted-prefix recurrent replay into the next block.
    #[must_use]
    fn supports_folded_verification(&self) -> bool {
        false
    }

    /// Return whether folded verification state currently awaits commit, flush, or discard.
    #[must_use]
    fn has_pending_folded_verification(&self) -> bool {
        false
    }

    /// Evaluate a speculative verification block while folding `carried_tokens` replay ahead of `tokens`.
    ///
    /// # Errors
    /// Rejects invalid state transitions, cancellation, or underlying decoder failures.
    fn evaluate_verification_block_folded(
        &mut self,
        carried_tokens: &[u32],
        tokens: &[u32],
        output: &mut Vec<Vec<f32>>,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        if !carried_tokens.is_empty() {
            self.evaluate_batch_into(carried_tokens, output, check)?;
        }
        self.evaluate_batch_into(tokens, output, check)
    }

    /// Commit `kept` accepted draft tokens from the active folded verification block (`kept <= block_len`).
    ///
    /// # Errors
    /// Rejects invalid commit bounds or missing pending folded state.
    fn commit_folded_verification(&mut self, kept: usize) -> Result<()> {
        let _ = kept;
        Ok(())
    }

    /// Eagerly flush any deferred folded replay state so recurrent layers reach exact serial parity.
    ///
    /// # Errors
    /// Rejects pending uncommitted verification state or replay execution failures.
    fn flush_folded_verification(&mut self, check: &dyn Fn() -> Result<()>) -> Result<()> {
        let _ = check;
        Ok(())
    }

    /// Discard any uncommitted or deferred folded verification buffers on rollback or error.
    fn discard_folded_verification(&mut self) {}
}
