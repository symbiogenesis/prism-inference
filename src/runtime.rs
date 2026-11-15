//! Transactional token generation, bounded snapshots and exact edited-prefix reuse.

use crate::{
    Decoder, MAX_PREFILL_BLOCK_TOKENS, Result, invalid,
    sampling::{FiniteLogits, Greedy, Sampler},
    speculative::{
        MAX_SPECULATIVE_DRAFT_TOKENS, SpeculativeStats, SuffixDraftConfig, SuffixDrafter,
        VerifiedRound,
    },
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Instant;

mod sampled_speculation;
mod timing;
mod verification;
use timing::measure;
pub use timing::{GenerationPhase, GenerationTimings, PhaseTiming};

const MAX_RECYCLED_CHECKPOINTS: usize = 2;
const MAX_RECYCLED_BATCH_LOGITS: usize = MAX_SPECULATIVE_DRAFT_TOKENS;

struct GenerationEffects<'a, S, E> {
    sampler: &'a mut S,
    emit: E,
    timings: Option<&'a mut GenerationTimings>,
    terminal_state_only: bool,
}

/// Cancellation shared with the request owner; checked between decoder steps.
#[derive(Clone, Default)]
pub struct Cancellation(Arc<AtomicBool>);

impl Cancellation {
    /// Request cancellation; an executing scalar operation completes first.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    /// Whether this request has been cancelled.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }

    fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "native generation cancelled",
            )
            .into());
        }
        Ok(())
    }
}

/// Hard bounds for one token conversation and its independently owned snapshots.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Maximum prompt plus generated sequence length; no eviction or shifting.
    pub context_tokens: usize,
    /// Number of completed prefixes retained, from zero through eight.
    pub checkpoints: usize,
    /// Total snapshot storage, including the request's rollback snapshot.
    pub checkpoint_bytes: usize,
}

/// Generation bounds and exact stop tokens, independent of sampling policy.
#[derive(Clone, Debug)]
pub struct Generation {
    /// Maximum sampled tokens, including a sampled stop token.
    pub max_tokens: usize,
    /// Exact token IDs that finish generation without entering conversation state.
    pub stop_tokens: Vec<u32>,
}

/// Bounded scheduling for intermediate, uncached prompt tokens.
///
/// Blocks execute causally through [`Decoder::ingest_block`]; the reference
/// fallback retains exact sequential arithmetic. The final prompt token always
/// computes logits separately. This does not select optimized graph kernels or
/// change decode scheduling. Existing generation methods use one-token blocks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Prefill {
    /// Maximum state-only tokens per block, from one through
    /// [`MAX_PREFILL_BLOCK_TOKENS`]. Invalid sizes reject the request before any
    /// decoder operation or checkpoint eviction.
    pub block_tokens: usize,
}

impl Default for Prefill {
    fn default() -> Self {
        Self { block_tokens: 1 }
    }
}

/// Why a successfully completed request stopped.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Finish {
    /// The sampled next token was in the supplied stop set.
    Stop,
    /// The requested maximum number of sampled tokens was reached.
    Length,
}

/// Accounting for a successful request; no text or private prompt is retained here.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Completion {
    /// Exact original prompt length.
    pub prompt_tokens: usize,
    /// Tokens restored or reused without evaluation.
    pub reused_tokens: usize,
    /// Newly emitted tokens; excludes the stop token.
    pub generated_tokens: usize,
    /// Stop-token or length completion.
    pub finish: Finish,
}

/// Explicit request-boundary policy for reusing the latest committed live state.
///
/// This does not add retained snapshots or admit approximate prefix matches.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PromptReuse {
    /// Start every request from empty model state.
    #[default]
    Disabled,
    /// Permit exact live-prefix reuse at an unchanged profile and scale bits.
    Latest,
    /// Permit exact live-prefix reuse and also retain the latest prompt-end checkpoint.
    PromptEndAndLatest,
}

/// A contiguous range of placeholder tokens in a prompt backed by one encoded image.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MediaSpan {
    /// Start token index in the linearized prompt token slice.
    pub offset: usize,
    /// Number of placeholder tokens covered by this image chunk (`grid_width * grid_height`).
    pub len: usize,
    /// Merged patch grid width in tokens per row.
    pub grid_width: usize,
    /// Merged patch grid height in rows.
    pub grid_height: usize,
    /// SHA-256 digest of the canonical image bytes for prefix-reuse equality.
    pub content_sha256: [u8; 32],
}

/// Borrowed image chunk paired with its flattened hidden-state embeddings (`len * d_model`).
#[derive(Clone, Copy, Debug)]
pub struct MultimodalChunk<'a> {
    /// Placeholder token span and image identity in the linearized prompt.
    pub span: MediaSpan,
    /// Row-major `F32` embeddings produced by the vision projector.
    pub embeddings: &'a [f32],
}

struct Snapshot<S> {
    state: S,
    tokens: Vec<u32>,
    media_spans: Vec<MediaSpan>,
    logits: Arc<Vec<f32>>,
    bytes: usize,
}

struct PromptEndCapture<'a, S> {
    reserved: usize,
    slot: &'a mut Option<Snapshot<S>>,
}

/// Reconciled maximal reusable prefix boundary across token history, paged KV
/// alignment, recurrent `GDN`/convolution state availability, and identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HybridCheckpointBoundary {
    /// Maximal shared prefix length safe to reuse without truncating recurrent state.
    pub position: usize,
    /// Common token-prefix length matched between prompt and retained history.
    pub matched_tokens: usize,
    /// Maximal KV cache prefix after optional page-size alignment.
    pub kv_bound: usize,
    /// Whether recurrent state availability or page alignment clamped the boundary.
    pub clamped_by_recurrent: bool,
    /// Whether representation and adapter identities matched.
    pub identity_matched: bool,
}

impl HybridCheckpointBoundary {
    /// Reconcile a candidate prefix state against requested prompt tokens and cache bounds.
    ///
    /// When `kv_page_alignment` is `Some(page)` with `page > 1`, full-attention KV
    /// alignment floors down to multiples of `page`; passing `None` permits exact
    /// unaligned KV tails. When `has_recurrent_layers` is true, recurrent `GDN` and
    /// convolution buffers are non-truncatable backwards, so an exact committed
    /// `recurrent_position` (`<= kv_bound`) is required to reuse any nonempty prefix.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "Keep token history, identity, paged KV alignment, and recurrent boundary explicit."
    )]
    pub fn reconcile(
        prompt: &[u32],
        history_tokens: &[u32],
        live_identity: &str,
        candidate_identity: &str,
        kv_tokens: usize,
        kv_page_alignment: Option<usize>,
        has_recurrent_layers: bool,
        recurrent_position: Option<usize>,
    ) -> Self {
        if live_identity != candidate_identity {
            return Self {
                position: 0,
                matched_tokens: 0,
                kv_bound: 0,
                clamped_by_recurrent: false,
                identity_matched: false,
            };
        }
        let matched_tokens = prompt
            .iter()
            .zip(history_tokens)
            .take_while(|(a, b)| a == b)
            .count();
        let raw_kv = matched_tokens.min(kv_tokens);
        let kv_bound = match kv_page_alignment {
            Some(page) if page > 1 => (raw_kv / page) * page,
            _ => raw_kv,
        };
        if !has_recurrent_layers {
            return Self {
                position: kv_bound,
                matched_tokens,
                kv_bound,
                clamped_by_recurrent: false,
                identity_matched: true,
            };
        }
        let Some(rec_pos) = recurrent_position else {
            return Self {
                position: 0,
                matched_tokens,
                kv_bound,
                clamped_by_recurrent: kv_bound > 0,
                identity_matched: true,
            };
        };
        if rec_pos <= kv_bound {
            Self {
                position: rec_pos,
                matched_tokens,
                kv_bound,
                clamped_by_recurrent: rec_pos < kv_bound,
                identity_matched: true,
            }
        } else {
            // Recurrent buffers at `rec_pos` cannot be rolled backwards to a
            // shorter matched or page-aligned KV boundary (`kv_bound`).
            Self {
                position: 0,
                matched_tokens,
                kv_bound,
                clamped_by_recurrent: true,
                identity_matched: true,
            }
        }
    }
}

/// Hard output bounds for one bounded speculative draft verification call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VerificationLimits<'a> {
    /// Maximum new tokens that may be committed during this verification call.
    pub max_new_tokens: usize,
    /// Exact stop tokens (such as `EOS` or turn-end tokens) that halt emission immediately.
    pub stop_tokens: &'a [u32],
}

/// Witness counters for buffer reallocations and reused scratch preparations in [`Runtime`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReallocationWitness {
    /// Reservations or buffer allocations that increased element capacity (`new_cap > old_cap`).
    pub growth_reallocations: u64,
    /// Detached buffer rebuilds matching prior capacity when no recycled slot was free.
    pub same_size_reallocations: u64,
    /// Decode steps, batch rows, or checkpoints that reused an existing buffer capacity.
    pub reused_preparations: u64,
}

/// Single-owner native conversation with rollback on cancellation or failure.
///
/// Edited prompts reuse only an exact saved token prefix. Recurrent state is
/// never truncated as though it were a KV-only cache. A failed stream callback
/// rolls back the entire request, including prefill; already emitted output is
/// provisional until this call succeeds. Failed restoration latches the runtime
/// closed until [`Self::reset`] succeeds. Snapshots share immutable logits; a
/// subsequent evaluation detaches without copying and reuses its output buffer.
/// No task or background work is spawned.
pub struct Runtime<D: Decoder> {
    decoder: D,
    limits: Limits,
    tokens: Vec<u32>,
    media_spans: Vec<MediaSpan>,
    // Empty means invalid; a valid decoder vocabulary is always nonempty.
    // Captures/restores share immutable output. Mutation detaches without copying.
    logits: Arc<Vec<f32>>,
    checkpoints: Vec<Snapshot<D::Checkpoint>>,
    prompt_end_checkpoint: Option<Snapshot<D::Checkpoint>>,
    retain_prompt_end: bool,
    recycled_checkpoints: Vec<D::Checkpoint>,
    batch_logits: Vec<Vec<f32>>,
    reallocation_witness: ReallocationWitness,
    faulted: bool,
}

impl Runtime<crate::graph::Qwen35> {
    /// Prepare one sealed profile request under an explicit prompt-reuse policy.
    ///
    /// Only bit-identical scales preserve the live sequence. Every scale change,
    /// including signed zero and change-back, clears state and renews checkpoint
    /// provenance. Unmatched prompts are still recomputed transactionally by
    /// generation. This operation never changes caller-owned sampler draws.
    ///
    /// # Errors
    /// Rejects mismatched/invalid scales and faulted admission before mutation;
    /// reset failures keep admission closed until successful explicit reset.
    pub fn prepare_profile_request(
        &mut self,
        scale: Option<f32>,
        reuse: PromptReuse,
    ) -> Result<()> {
        let current = self.decoder.adapter_scale();
        match (current, scale) {
            (None, None) => {}
            (Some(_), Some(scale)) => {
                crate::model::lora::RankOneAdapter::validate_scale(scale)?;
            }
            _ => return Err(invalid("request scale does not match the loaded adapter")),
        }
        if self.faulted {
            return Err(invalid(
                "native runtime requires reset after failed restoration",
            ));
        }
        if matches!(reuse, PromptReuse::Latest | PromptReuse::PromptEndAndLatest)
            && current.map(f32::to_bits) == scale.map(f32::to_bits)
        {
            if reuse == PromptReuse::PromptEndAndLatest {
                self.retain_prompt_end = true;
            }
            return Ok(());
        }
        if reuse == PromptReuse::PromptEndAndLatest {
            self.retain_prompt_end = true;
        }
        if let Some(scale) = scale {
            self.reset_with_adapter_scale(scale)
        } else {
            self.reset()
        }
    }

    /// Clear complete conversation state before changing a loaded adapter scale.
    ///
    /// This narrow operation exposes no mutable decoder access. All committed
    /// tokens, logits and saved prefixes are discarded only after the decoder
    /// renews its empty-state identity and checkpoint provenance. Sampler state
    /// remains caller-owned and is not changed by this operation.
    ///
    /// # Errors
    /// Invalid/nonfinite scales and graphs without an adapter reject before any
    /// mutation. Later failure closes admission until successful explicit reset.
    pub fn set_projection_executor(
        &mut self,
        executor: Option<Box<dyn crate::kernels::Pq2ProjectionExecutor + Send>>,
    ) {
        self.decoder.set_projection_executor(executor);
    }

    /// Clear complete conversation state before changing a loaded adapter scale.
    ///
    /// This narrow operation exposes no mutable decoder access. All committed
    /// tokens, logits and saved prefixes are discarded only after the decoder
    /// renews its empty-state identity and checkpoint provenance. Sampler state
    /// remains caller-owned and is not changed by this operation.
    ///
    /// # Errors
    /// Invalid/nonfinite scales and graphs without an adapter reject before any
    /// mutation. Later failure closes admission until successful explicit reset.
    pub fn reset_with_adapter_scale(&mut self, scale: f32) -> Result<()> {
        crate::model::lora::RankOneAdapter::validate_scale(scale)?;
        if self.decoder.adapter_scale().is_none() {
            return Err(invalid("adapter scale requires a loaded rank-one adapter"));
        }
        self.faulted = true;
        self.decoder.reset_with_adapter_scale(scale)?;
        self.checkpoints.clear();
        self.prompt_end_checkpoint = None;
        self.recycled_checkpoints.clear();
        self.tokens.clear();
        self.media_spans.clear();
        mutable_logits(&mut self.logits).clear();
        self.faulted = false;
        Ok(())
    }
}

impl<D: Decoder> Runtime<D> {
    /// Establish an empty conversation and validate all fixed ownership bounds.
    ///
    /// # Errors
    /// Rejects invalid limits, impossible vocabulary sizes, allocation failures,
    /// or a decoder that cannot reset to an empty sequence.
    pub fn new(mut decoder: D, limits: Limits) -> Result<Self> {
        if limits.context_tokens == 0
            || limits.context_tokens > 262_144
            || limits.checkpoints > 8
            || limits.checkpoint_bytes == 0
            || decoder.vocab_size() == 0
            || decoder.vocab_size() > 1_048_576
            || decoder.identity().is_empty()
        {
            return Err(invalid(
                "invalid native conversation limits or decoder identity",
            ));
        }
        decoder.reset()?;
        let mut tokens = Vec::new();
        tokens.try_reserve_exact(limits.context_tokens)?;
        let mut checkpoints = Vec::new();
        checkpoints.try_reserve_exact(limits.checkpoints)?;
        let mut recycled_checkpoints = Vec::new();
        recycled_checkpoints.try_reserve_exact(MAX_RECYCLED_CHECKPOINTS)?;
        let mut batch_logits = Vec::new();
        batch_logits.try_reserve_exact(MAX_RECYCLED_BATCH_LOGITS)?;
        Ok(Self {
            decoder,
            limits,
            tokens,
            media_spans: Vec::new(),
            logits: Arc::new(Vec::new()),
            checkpoints,
            prompt_end_checkpoint: None,
            retain_prompt_end: false,
            recycled_checkpoints,
            batch_logits,
            reallocation_witness: ReallocationWitness::default(),
            faulted: false,
        })
    }

    /// Runtime reallocation and recycled-scratch preparation witnesses.
    #[must_use]
    pub const fn reallocation_witness(&self) -> ReallocationWitness {
        self.reallocation_witness
    }

    /// Committed prompt plus generated tokens; excludes sampled stop tokens.
    #[must_use]
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Committed multimodal image spans aligned with [`Self::tokens`].
    #[must_use]
    pub fn media_spans(&self) -> &[MediaSpan] {
        &self.media_spans
    }

    /// Inspect the decoder's identity and diagnostics without changing ownership.
    ///
    /// Diagnostics describe the backend's latest operation, which may have been
    /// provisional or failed. Consult [`Self::tokens`] for committed history and
    /// [`Self::is_faulted`] before interpreting the backend as reusable state.
    #[must_use]
    pub const fn decoder(&self) -> &D {
        &self.decoder
    }

    /// Whether failed backend restoration or reset requires a successful reset.
    #[must_use]
    pub const fn is_faulted(&self) -> bool {
        self.faulted
    }

    /// Retained complete-prefix snapshot count.
    #[must_use]
    pub const fn checkpoint_count(&self) -> usize {
        self.checkpoints.len()
    }

    /// Enable or disable retention of a dedicated reusable prompt-end checkpoint.
    ///
    /// Disabling clears and recycles any previously retained prompt-end checkpoint.
    pub fn set_retain_prompt_end(&mut self, retain: bool) {
        self.retain_prompt_end = retain;
        if !retain {
            let previous = self.prompt_end_checkpoint.take();
            self.recycle_optional_snapshot(previous);
        }
    }

    /// Whether dedicated prompt-end checkpoint retention is enabled.
    #[must_use]
    pub const fn retain_prompt_end(&self) -> bool {
        self.retain_prompt_end
    }

    /// Whether a prompt-end checkpoint is currently retained.
    #[must_use]
    pub const fn has_prompt_end_checkpoint(&self) -> bool {
        self.prompt_end_checkpoint.is_some()
    }

    /// Compute the maximal reusable prefix boundary across live state and saved checkpoints.
    ///
    /// Inspects active conversation tokens (when current logits are valid), saved prefix
    /// checkpoints, and any retained prompt-end checkpoint.
    #[must_use]
    pub fn hybrid_boundary(
        &self,
        prompt: &[u32],
        has_recurrent_layers: bool,
        kv_page_alignment: Option<usize>,
    ) -> HybridCheckpointBoundary {
        let identity = self.decoder.identity();
        let mut best = if !self.tokens.is_empty() && !self.logits.is_empty() {
            HybridCheckpointBoundary::reconcile(
                prompt,
                &self.tokens,
                identity,
                identity,
                self.tokens.len(),
                kv_page_alignment,
                has_recurrent_layers,
                has_recurrent_layers.then_some(self.tokens.len()),
            )
        } else {
            HybridCheckpointBoundary::reconcile(
                prompt,
                &[],
                identity,
                identity,
                0,
                kv_page_alignment,
                has_recurrent_layers,
                None,
            )
        };
        for snapshot in self.checkpoints.iter().chain(&self.prompt_end_checkpoint) {
            let candidate = HybridCheckpointBoundary::reconcile(
                prompt,
                &snapshot.tokens,
                identity,
                identity,
                snapshot.tokens.len(),
                kv_page_alignment,
                has_recurrent_layers,
                has_recurrent_layers.then_some(snapshot.tokens.len()),
            );
            if candidate.position > best.position {
                best = candidate;
            }
        }
        best
    }

    /// Drop saved prefixes and clear all model state, recovering a failure latch.
    ///
    /// # Errors
    /// A reset failure leaves admission closed and returns the decoder error.
    /// Committed history and saved prefixes are cleared only after success;
    /// backend state remains unavailable until a subsequent reset succeeds.
    pub fn reset(&mut self) -> Result<()> {
        self.faulted = true;
        self.decoder.reset()?;
        while let Some(snapshot) = self.checkpoints.pop() {
            self.recycle_snapshot(snapshot);
        }
        if let Some(snapshot) = self.prompt_end_checkpoint.take() {
            self.recycle_snapshot(snapshot);
        }
        self.tokens.clear();
        self.media_spans.clear();
        mutable_logits(&mut self.logits).clear();
        self.faulted = false;
        Ok(())
    }

    /// Evaluate a nonempty prompt and stream greedy token IDs synchronously.
    ///
    /// The callback supplies backpressure and may return an error to cancel a
    /// disconnected consumer. Context and token validation precede execution.
    /// Cancellation is checked before/after every decoder step and callback.
    ///
    /// # Errors
    /// Rejects invalid tokens/options, resource exhaustion, cancellation,
    /// nonfinite/wrong-size logits, callback errors and decoder failures. State
    /// returns to the previous committed prefix unless restoration itself fails.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        options: &Generation,
        cancellation: &Cancellation,
        emit: impl FnMut(u32) -> Result<()>,
    ) -> Result<Completion> {
        self.generate_with_prefill(prompt, options, Prefill::default(), cancellation, emit)
    }

    /// Generate with bounded, causal scheduling of uncached prompt blocks.
    ///
    /// The entire request, including all completed and partial blocks, shares
    /// one rollback checkpoint. Cancellation is checked between tokens inside
    /// each block. There is no intermediate commit or background work.
    ///
    /// # Errors
    /// Returns the errors of [`Self::generate`], and rejects an invalid prefill
    /// block size before capturing a checkpoint or changing decoder state.
    pub fn generate_with_prefill(
        &mut self,
        prompt: &[u32],
        options: &Generation,
        prefill: Prefill,
        cancellation: &Cancellation,
        emit: impl FnMut(u32) -> Result<()>,
    ) -> Result<Completion> {
        self.generate_checked(prompt, options, prefill, &|| cancellation.check(), emit)
    }

    /// Generate synchronously before an absolute monotonic request deadline.
    ///
    /// An optional deadline includes prefill, generation, callbacks and snapshot work.
    /// Passing None keeps cancellation and all non-time bounds without a total cutoff.
    /// Expiry is cooperative: a running decoder operation or callback completes
    /// before the next check detects it. Recovery is allowed to finish after the
    /// deadline. Emitted output remains provisional until this call succeeds.
    /// Cancellation takes precedence when both cancellation and expiry occur.
    ///
    /// # Errors
    /// Returns an [`std::io::ErrorKind::TimedOut`] error on expiry, or the same
    /// errors as [`Self::generate`]. An expired request is rejected before any
    /// decoder operation. Admitted failures restore the previous committed
    /// state, unless restoration fails and closes admission until reset.
    pub fn generate_until(
        &mut self,
        prompt: &[u32],
        options: &Generation,
        cancellation: &Cancellation,
        deadline: impl Into<Option<Instant>>,
        emit: impl FnMut(u32) -> Result<()>,
    ) -> Result<Completion> {
        let deadline = deadline.into();
        self.generate_until_with_prefill(
            prompt,
            options,
            Prefill::default(),
            cancellation,
            deadline,
            emit,
        )
    }

    /// Generate bounded causal prompt blocks before a monotonic deadline.
    ///
    /// Deadline and cancellation checks occur between tokens within blocks,
    /// with the same whole-request rollback and provisional output contract as
    /// [`Self::generate_until`]. Recovery may finish after the deadline.
    ///
    /// # Errors
    /// Returns the errors of [`Self::generate_until`] and rejects invalid
    /// prefill block sizes before any decoder operation.
    pub fn generate_until_with_prefill(
        &mut self,
        prompt: &[u32],
        options: &Generation,
        prefill: Prefill,
        cancellation: &Cancellation,
        deadline: impl Into<Option<Instant>>,
        emit: impl FnMut(u32) -> Result<()>,
    ) -> Result<Completion> {
        let deadline = deadline.into();
        self.generate_checked(
            prompt,
            options,
            prefill,
            &|| check_deadline(cancellation, deadline, Instant::now()),
            emit,
        )
    }

    /// Generate with a caller-owned sampler and bounded causal prompt blocks.
    ///
    /// A successful request commits the sampler's state, including draws for a
    /// stop token. Any returned error restores its original state alongside the
    /// decoder, even if decoder restoration fails. The sampler's [`Clone`]
    /// implementation must independently snapshot all state changed by sampling;
    /// shared mutable RNG state cannot provide transactional recovery.
    /// External effects performed by a sampler cannot be rolled back.
    /// Cancellation is checked immediately before and after every sample.
    /// Emitted output remains provisional until the whole request succeeds.
    ///
    /// # Errors
    /// Returns the errors of [`Self::generate_with_prefill`], sampler failures,
    /// or an out-of-vocabulary sampled token.
    pub fn generate_sampled<S: Sampler + Clone>(
        &mut self,
        prompt: &[u32],
        options: &Generation,
        prefill: Prefill,
        cancellation: &Cancellation,
        sampler: &mut S,
        emit: impl FnMut(u32) -> Result<()>,
    ) -> Result<Completion> {
        self.generate_sampled_checked(
            prompt,
            options,
            prefill,
            &|| cancellation.check(),
            sampler,
            emit,
        )
    }

    /// Generate with a caller-owned sampler before a monotonic deadline.
    ///
    /// Uses the sampling and recovery contract of [`Self::generate_sampled`].
    /// Deadline checks surround sampling as well as decoder and callback work;
    /// a running operation completes before expiry is observed. Cancellation
    /// takes precedence over expiry, and recovery may finish after the deadline.
    ///
    /// # Errors
    /// Returns the errors of [`Self::generate_sampled`] or a timed-out error.
    #[expect(
        clippy::too_many_arguments,
        reason = "explicit sampling extends the existing bounded deadline API"
    )]
    pub fn generate_sampled_until<S: Sampler + Clone>(
        &mut self,
        prompt: &[u32],
        options: &Generation,
        prefill: Prefill,
        cancellation: &Cancellation,
        deadline: impl Into<Option<Instant>>,
        sampler: &mut S,
        emit: impl FnMut(u32) -> Result<()>,
    ) -> Result<Completion> {
        let deadline = deadline.into();
        self.generate_sampled_checked(
            prompt,
            options,
            prefill,
            &|| check_deadline(cancellation, deadline, Instant::now()),
            sampler,
            emit,
        )
    }

    /// Generate with opt-in, value-free runtime phase and token-work attribution.
    ///
    /// Uses exactly the transaction, sampling, cancellation and callback contract
    /// of [`Self::generate_sampled_until`]. Replaces `timings` for this attempt,
    /// including returned failures. Failed operation counts are lower bounds;
    /// callback success remains provisional until the full call succeeds.
    /// Ordinary generation entry points perform no attribution clock reads.
    ///
    /// # Errors
    /// Returns the same errors as [`Self::generate_sampled_until`].
    #[expect(
        clippy::too_many_arguments,
        reason = "Optional attribution extends the existing explicit sampled deadline API."
    )]
    pub fn generate_sampled_until_with_timings<S: Sampler + Clone>(
        &mut self,
        prompt: &[u32],
        options: &Generation,
        prefill: Prefill,
        cancellation: &Cancellation,
        deadline: impl Into<Option<Instant>>,
        sampler: &mut S,
        timings: &mut GenerationTimings,
        emit: impl FnMut(u32) -> Result<()>,
    ) -> Result<Completion> {
        let deadline = deadline.into();
        self.generate_sampled_observed(
            prompt,
            &[],
            options,
            prefill,
            &|| check_deadline(cancellation, deadline, Instant::now()),
            GenerationEffects {
                sampler,
                emit,
                timings: Some(timings),
                terminal_state_only: false,
            },
        )
    }

    /// Generate with multimodal image chunks inserted at their exact token spans.
    ///
    /// Uses the same sampling, prefix-reuse, cancellation and rollback contract as
    /// [`Self::generate_sampled_until`].
    ///
    /// # Errors
    /// Rejects invalid chunks, overlaps, nonfinite embeddings, or any error
    /// returned by [`Self::generate_sampled_until`].
    #[expect(
        clippy::too_many_arguments,
        reason = "Multimodal sampling adds chunk payloads to the explicit deadline API."
    )]
    pub fn generate_multimodal_sampled_until<S: Sampler + Clone>(
        &mut self,
        prompt: &[u32],
        chunks: &[MultimodalChunk<'_>],
        options: &Generation,
        prefill: Prefill,
        cancellation: &Cancellation,
        deadline: impl Into<Option<Instant>>,
        sampler: &mut S,
        emit: impl FnMut(u32) -> Result<()>,
    ) -> Result<Completion> {
        let deadline = deadline.into();
        self.generate_sampled_observed(
            prompt,
            chunks,
            options,
            prefill,
            &|| check_deadline(cancellation, deadline, Instant::now()),
            GenerationEffects {
                sampler,
                emit,
                timings: None,
                terminal_state_only: false,
            },
        )
    }

    /// Generate with multimodal image chunks and fixed-size runtime phase timings.
    ///
    /// # Errors
    /// Returns the errors of [`Self::generate_multimodal_sampled_until`].
    #[expect(
        clippy::too_many_arguments,
        reason = "Optional attribution extends the explicit multimodal deadline API."
    )]
    pub fn generate_multimodal_sampled_until_with_timings<S: Sampler + Clone>(
        &mut self,
        prompt: &[u32],
        chunks: &[MultimodalChunk<'_>],
        options: &Generation,
        prefill: Prefill,
        cancellation: &Cancellation,
        deadline: impl Into<Option<Instant>>,
        sampler: &mut S,
        timings: &mut GenerationTimings,
        emit: impl FnMut(u32) -> Result<()>,
    ) -> Result<Completion> {
        let deadline = deadline.into();
        self.generate_sampled_observed(
            prompt,
            chunks,
            options,
            prefill,
            &|| check_deadline(cancellation, deadline, Instant::now()),
            GenerationEffects {
                sampler,
                emit,
                timings: Some(timings),
                terminal_state_only: false,
            },
        )
    }

    /// Finish a reset-only multimodal request without retaining prefix checkpoints or terminal logits.
    #[expect(
        clippy::too_many_arguments,
        reason = "The reset-only multimodal service preserves deadline, sampler and timing attribution arguments."
    )]
    #[cfg(any(feature = "cli", test))]
    pub(crate) fn generate_multimodal_sampled_until_for_reset<S: Sampler + Clone>(
        &mut self,
        prompt: &[u32],
        chunks: &[MultimodalChunk<'_>],
        options: &Generation,
        prefill: Prefill,
        cancellation: &Cancellation,
        deadline: impl Into<Option<Instant>>,
        sampler: &mut S,
        timings: Option<&mut GenerationTimings>,
        emit: impl FnMut(u32) -> Result<()>,
    ) -> Result<Completion> {
        let deadline = deadline.into();
        if self.limits.checkpoints != 0 {
            return Err(invalid(
                "reset-only generation cannot retain prefix checkpoints",
            ));
        }
        self.generate_sampled_observed(
            prompt,
            chunks,
            options,
            prefill,
            &|| check_deadline(cancellation, deadline, Instant::now()),
            GenerationEffects {
                sampler,
                emit,
                timings,
                terminal_state_only: true,
            },
        )
    }

    /// Finish a reset-only service request without unused terminal logits.
    ///
    /// This is deliberately unavailable through the public generation API. The
    /// service must reset before its next request, and cannot retain prefix
    /// snapshots. Empty terminal logits also force fresh preparation if another
    /// internal caller accidentally tries to continue this runtime directly.
    #[expect(
        clippy::too_many_arguments,
        reason = "The reset-only service keeps the existing deadline, sampler and optional attribution arguments."
    )]
    #[cfg(any(feature = "cli", test))]
    pub(crate) fn generate_sampled_until_for_reset<S: Sampler + Clone>(
        &mut self,
        prompt: &[u32],
        options: &Generation,
        prefill: Prefill,
        cancellation: &Cancellation,
        deadline: impl Into<Option<Instant>>,
        sampler: &mut S,
        timings: Option<&mut GenerationTimings>,
        emit: impl FnMut(u32) -> Result<()>,
    ) -> Result<Completion> {
        let deadline = deadline.into();
        if self.limits.checkpoints != 0 {
            return Err(invalid(
                "reset-only generation cannot retain prefix checkpoints",
            ));
        }
        self.generate_sampled_observed(
            prompt,
            &[],
            options,
            prefill,
            &|| check_deadline(cancellation, deadline, Instant::now()),
            GenerationEffects {
                sampler,
                emit,
                timings,
                terminal_state_only: true,
            },
        )
    }

    fn generate_checked(
        &mut self,
        prompt: &[u32],
        options: &Generation,
        prefill: Prefill,
        check: &impl Fn() -> Result<()>,
        emit: impl FnMut(u32) -> Result<()>,
    ) -> Result<Completion> {
        self.generate_sampled_checked(prompt, options, prefill, check, &mut Greedy, emit)
    }

    fn generate_sampled_checked<S: Sampler + Clone>(
        &mut self,
        prompt: &[u32],
        options: &Generation,
        prefill: Prefill,
        check: &impl Fn() -> Result<()>,
        sampler: &mut S,
        emit: impl FnMut(u32) -> Result<()>,
    ) -> Result<Completion> {
        self.generate_sampled_observed(
            prompt,
            &[],
            options,
            prefill,
            check,
            GenerationEffects {
                sampler,
                emit,
                timings: None,
                terminal_state_only: false,
            },
        )
    }

    fn generate_sampled_observed<S: Sampler + Clone>(
        &mut self,
        prompt: &[u32],
        chunks: &[MultimodalChunk<'_>],
        options: &Generation,
        prefill: Prefill,
        check: &impl Fn() -> Result<()>,
        mut effects: GenerationEffects<'_, S, impl FnMut(u32) -> Result<()>>,
    ) -> Result<Completion> {
        let started = effects.timings.as_deref_mut().map(|timings| {
            *timings = GenerationTimings::started(prompt.len());
            Instant::now()
        });
        let checkpoint = measure(
            &mut effects.timings,
            GenerationPhase::SamplerSnapshot,
            0,
            || Ok(effects.sampler.clone()),
        )?;
        let result =
            self.generate_transaction(prompt, chunks, options, prefill, check, &mut effects);
        if result.is_err() {
            measure(
                &mut effects.timings,
                GenerationPhase::SamplerRestore,
                0,
                || {
                    *effects.sampler = checkpoint;
                    Ok(())
                },
            )?;
        }
        if let (Some(timings), Some(started)) = (effects.timings, started) {
            timings.total = started.elapsed();
            timings.succeeded = result.is_ok();
            timings.committed_generated_tokens =
                result.as_ref().ok().map(|value| value.generated_tokens);
        }
        result
    }

    fn generate_transaction(
        &mut self,
        prompt: &[u32],
        chunks: &[MultimodalChunk<'_>],
        options: &Generation,
        prefill: Prefill,
        check: &impl Fn() -> Result<()>,
        effects: &mut GenerationEffects<'_, impl Sampler, impl FnMut(u32) -> Result<()>>,
    ) -> Result<Completion> {
        if self.faulted {
            return Err(invalid(
                "native runtime requires reset after failed restoration",
            ));
        }
        check()?;
        self.validate_request(prompt, options, prefill)?;
        validate_multimodal_chunks(prompt, chunks)?;
        check()?;
        let rollback = measure(
            &mut effects.timings,
            GenerationPhase::RollbackSnapshot,
            0,
            || self.capture_transient(0, check),
        )?;
        let mut prompt_end = None;
        let result = (|| {
            check()?;
            let mut prompt_end_capture = PromptEndCapture {
                reserved: rollback.bytes,
                slot: &mut prompt_end,
            };
            let completion = self.generate_inner(
                prompt,
                chunks,
                options,
                prefill,
                check,
                effects,
                &mut prompt_end_capture,
            )?;
            check()?;
            // A failed optional capture still permits success, but deadline
            // expiry during capture must roll back the completed generation.
            let snapshot = if self.limits.checkpoints == 0 {
                None
            } else {
                let snapshot_reserved = rollback
                    .bytes
                    .saturating_add(prompt_end.as_ref().map_or(0, |s| s.bytes));
                let snapshot = measure(
                    &mut effects.timings,
                    GenerationPhase::CompletionSnapshot,
                    0,
                    || self.capture_reusable(snapshot_reserved, check),
                )
                .ok();
                check()?;
                snapshot
            };
            Ok((completion, snapshot))
        })();
        match result {
            Ok((completion, snapshot)) => {
                // Snapshots are an optional reuse optimization. A budget miss
                // does not invalidate an otherwise completed generation.
                self.commit_prompt_end(prompt_end);
                self.commit_reusable_snapshot(snapshot);
                self.recycle_snapshot(rollback);
                Ok(completion)
            }
            Err(error) => {
                self.faulted = true;
                measure(
                    &mut effects.timings,
                    GenerationPhase::RollbackRestore,
                    0,
                    || {
                        let Snapshot {
                            state,
                            tokens,
                            media_spans,
                            logits,
                            ..
                        } = rollback;
                        let restored = self.decoder.restore(&state);
                        // History describes the committed request even if backend
                        // recovery fails and admission remains closed.
                        self.tokens = tokens;
                        self.media_spans = media_spans;
                        let previous_logits = std::mem::replace(&mut self.logits, logits);
                        restored?;
                        self.recycle_checkpoint(state);
                        self.recycle_logits(previous_logits);
                        Ok(())
                    },
                )?;
                self.recycle_optional_snapshot(prompt_end);
                self.faulted = false;
                Err(error)
            }
        }
    }

    fn validate_request(
        &self,
        prompt: &[u32],
        options: &Generation,
        prefill: Prefill,
    ) -> Result<()> {
        let length = prompt
            .len()
            .checked_add(options.max_tokens)
            .ok_or_else(|| invalid("native context length overflow"))?;
        if prompt.is_empty()
            || options.max_tokens == 0
            || prefill.block_tokens == 0
            || prefill.block_tokens > MAX_PREFILL_BLOCK_TOKENS
            || length > self.limits.context_tokens
            || options.stop_tokens.len() > 256
            || prompt.iter().chain(&options.stop_tokens).any(|&token| {
                !usize::try_from(token).is_ok_and(|value| value < self.decoder.vocab_size())
            })
        {
            return Err(invalid(
                "invalid tokens or native context/generation/prefill bounds",
            ));
        }
        Ok(())
    }

    fn snapshot_bytes(&self) -> Result<usize> {
        // Charge every shared payload in full, plus its Vec and Arc metadata.
        // Sharing removes copies without lowering the conservative cache bound.
        let logits_capacity = if self.logits.is_empty() {
            0
        } else {
            self.logits.capacity()
        };
        let logits_bytes = logits_capacity
            .checked_mul(size_of::<f32>())
            .and_then(|bytes| bytes.checked_add(size_of::<Vec<f32>>()))
            .and_then(|bytes| bytes.checked_add(2 * size_of::<usize>()))
            .ok_or_else(|| invalid("native logits checkpoint byte count overflow"))?;
        let media_bytes = self
            .media_spans
            .len()
            .checked_mul(size_of::<MediaSpan>())
            .ok_or_else(|| invalid("native media span checkpoint byte count overflow"))?;
        self.tokens
            .len()
            .checked_mul(size_of::<u32>())
            .and_then(|size| size.checked_add(media_bytes))
            .and_then(|size| size.checked_add(logits_bytes))
            .and_then(|size| size.checked_add(self.decoder.checkpoint_bytes()))
            .and_then(|size| size.checked_add(size_of::<Snapshot<D::Checkpoint>>()))
            .ok_or_else(|| invalid("native checkpoint byte count overflow"))
    }

    fn retained_reusable_bytes(&self) -> Result<usize> {
        let mut total = self
            .prompt_end_checkpoint
            .as_ref()
            .map_or(0, |snapshot| snapshot.bytes);
        for checkpoint in &self.checkpoints {
            total = total
                .checked_add(checkpoint.bytes)
                .ok_or_else(|| invalid("native checkpoint byte count overflow"))?;
        }
        Ok(total)
    }

    fn finish_capture(
        &mut self,
        bytes: usize,
        check: &impl Fn() -> Result<()>,
    ) -> Result<Snapshot<D::Checkpoint>> {
        let tokens = copy(&self.tokens)?;
        let media_spans = copy(&self.media_spans)?;
        // Invalid/empty output needs no retained backing storage. Keep its
        // dormant allocation unique so reset-only requests can reuse it.
        let logits = if self.logits.is_empty() {
            Arc::new(Vec::new())
        } else {
            Arc::clone(&self.logits)
        };
        check()?;
        let state = if let Some(mut recycled) = self.recycled_checkpoints.pop() {
            self.decoder.checkpoint_into(&mut recycled)?;
            self.reallocation_witness.reused_preparations = self
                .reallocation_witness
                .reused_preparations
                .saturating_add(1);
            recycled
        } else {
            let state = self.decoder.checkpoint()?;
            self.reallocation_witness.growth_reallocations = self
                .reallocation_witness
                .growth_reallocations
                .saturating_add(1);
            state
        };
        Ok(Snapshot {
            state,
            tokens,
            media_spans,
            logits,
            bytes,
        })
    }

    fn capture_transient(
        &mut self,
        reserved: usize,
        check: &impl Fn() -> Result<()>,
    ) -> Result<Snapshot<D::Checkpoint>> {
        let bytes = self.snapshot_bytes()?;
        let retained = self.retained_reusable_bytes()?;
        if reserved
            .checked_add(retained)
            .and_then(|total| total.checked_add(bytes))
            .is_none_or(|total| total > self.limits.checkpoint_bytes)
        {
            return Err(invalid("native transient checkpoint exceeds byte budget"));
        }
        self.finish_capture(bytes, check)
    }

    fn capture_reusable(
        &mut self,
        reserved: usize,
        check: &impl Fn() -> Result<()>,
    ) -> Result<Snapshot<D::Checkpoint>> {
        let bytes = self.snapshot_bytes()?;
        let prompt_end_bytes = self
            .prompt_end_checkpoint
            .as_ref()
            .map_or(0, |snapshot| snapshot.bytes);
        let static_reserved = reserved.saturating_add(prompt_end_bytes);
        if bytes > self.limits.checkpoint_bytes.saturating_sub(static_reserved) {
            return Err(invalid("native rollback checkpoint exceeds byte budget"));
        }
        while self
            .checkpoints
            .iter()
            .try_fold(static_reserved, |total, checkpoint| {
                total.checked_add(checkpoint.bytes)
            })
            .and_then(|size| size.checked_add(bytes))
            .is_none_or(|total| total > self.limits.checkpoint_bytes)
        {
            if self.checkpoints.is_empty() {
                return Err(invalid("native checkpoint budget exhausted"));
            }
            let evicted = self.checkpoints.remove(0);
            self.recycle_snapshot(evicted);
        }
        self.finish_capture(bytes, check)
    }

    #[cfg(test)]
    fn capture(
        &mut self,
        reserved: usize,
        check: &impl Fn() -> Result<()>,
    ) -> Result<Snapshot<D::Checkpoint>> {
        self.capture_reusable(reserved, check)
    }

    fn maybe_capture_prompt_end(
        &mut self,
        prompt: &[u32],
        capture: &mut PromptEndCapture<'_, D::Checkpoint>,
        check: &impl Fn() -> Result<()>,
    ) -> Result<()> {
        self.maybe_capture_prompt_end_with_spans(prompt, &[], capture, check)
    }

    fn maybe_capture_prompt_end_with_spans(
        &mut self,
        prompt: &[u32],
        spans: &[MediaSpan],
        capture: &mut PromptEndCapture<'_, D::Checkpoint>,
        check: &impl Fn() -> Result<()>,
    ) -> Result<()> {
        if !self.retain_prompt_end
            || self
                .prompt_end_checkpoint
                .as_ref()
                .is_some_and(|saved| saved.tokens == prompt && saved.media_spans == spans)
        {
            return Ok(());
        }
        let candidate = self.capture_transient(capture.reserved, check).ok();
        check()?;
        *capture.slot = candidate;
        Ok(())
    }

    fn commit_prompt_end(&mut self, candidate: Option<Snapshot<D::Checkpoint>>) {
        if let Some(snapshot) = candidate
            && let Some(previous) = self.prompt_end_checkpoint.replace(snapshot)
        {
            self.recycle_snapshot(previous);
        }
    }

    fn commit_reusable_snapshot(&mut self, snapshot: Option<Snapshot<D::Checkpoint>>) {
        if let Some(snapshot) = snapshot {
            // Count pressure alone must not discard a usable prefix
            // when its optional replacement could not be captured.
            if self.checkpoints.len() == self.limits.checkpoints {
                let evicted = self.checkpoints.remove(0);
                self.recycle_snapshot(evicted);
            }
            self.checkpoints.push(snapshot);
        }
    }

    fn recycle_optional_snapshot(&mut self, snapshot: Option<Snapshot<D::Checkpoint>>) {
        if let Some(snapshot) = snapshot {
            self.recycle_snapshot(snapshot);
        }
    }

    fn recycle_checkpoint(&mut self, state: D::Checkpoint) {
        if self.recycled_checkpoints.len() < MAX_RECYCLED_CHECKPOINTS {
            self.recycled_checkpoints.push(state);
        }
    }

    fn recycle_logits(&mut self, logits: Arc<Vec<f32>>) {
        if let Ok(mut values) = Arc::try_unwrap(logits)
            && values.capacity() > 0
            && self.batch_logits.len() < MAX_RECYCLED_BATCH_LOGITS
        {
            values.clear();
            self.batch_logits.push(values);
        }
    }

    fn recycle_snapshot(&mut self, snapshot: Snapshot<D::Checkpoint>) {
        self.recycle_checkpoint(snapshot.state);
        self.recycle_logits(snapshot.logits);
    }

    fn take_batch_rows(&mut self, count: usize) -> Result<Vec<Vec<f32>>> {
        let mut rows = Vec::new();
        rows.try_reserve_exact(count)?;
        let available = self.batch_logits.len().min(count);
        let start = self.batch_logits.len() - available;
        rows.extend(self.batch_logits.drain(start..));
        Ok(rows)
    }

    fn recycle_batch_rows(&mut self, rows: Vec<Vec<f32>>) {
        for mut row in rows {
            if row.capacity() > 0 && self.batch_logits.len() < MAX_RECYCLED_BATCH_LOGITS {
                row.clear();
                self.batch_logits.push(row);
            }
        }
    }

    fn record_batch_row_witness(&mut self, before_caps: &[usize], rows: &[Vec<f32>]) {
        for (idx, row) in rows.iter().enumerate() {
            let before = before_caps.get(idx).copied().unwrap_or(0);
            let after = row.capacity();
            if after > before {
                self.reallocation_witness.growth_reallocations = self
                    .reallocation_witness
                    .growth_reallocations
                    .saturating_add(1);
            } else if after > 0 {
                self.reallocation_witness.reused_preparations = self
                    .reallocation_witness
                    .reused_preparations
                    .saturating_add(1);
            }
        }
    }

    fn prepare(&mut self, prompt: &[u32], check: &impl Fn() -> Result<()>) -> Result<usize> {
        self.prepare_with_spans(prompt, &[], check)
    }

    fn prepare_with_spans(
        &mut self,
        prompt: &[u32],
        incoming_spans: &[MediaSpan],
        check: &impl Fn() -> Result<()>,
    ) -> Result<usize> {
        enum CandidateSlot {
            Prefix(usize),
            PromptEnd,
        }

        if !self.tokens.is_empty()
            && prefix_matches_media(prompt, incoming_spans, &self.tokens, &self.media_spans)
            && !self.logits.is_empty()
        {
            return Ok(self.tokens.len());
        }
        let best_cached = self
            .checkpoints
            .iter()
            .enumerate()
            .filter(|(_, snapshot)| {
                prefix_matches_media(
                    prompt,
                    incoming_spans,
                    &snapshot.tokens,
                    &snapshot.media_spans,
                )
            })
            .max_by_key(|(_, snapshot)| snapshot.tokens.len());
        let best_prompt_end = self.prompt_end_checkpoint.as_ref().filter(|snapshot| {
            prefix_matches_media(
                prompt,
                incoming_spans,
                &snapshot.tokens,
                &snapshot.media_spans,
            )
        });
        let best = match (best_cached, best_prompt_end) {
            (Some((index, cached)), Some(end)) => {
                if cached.tokens.len() >= end.tokens.len() {
                    Some(CandidateSlot::Prefix(index))
                } else {
                    Some(CandidateSlot::PromptEnd)
                }
            }
            (Some((index, _)), None) => Some(CandidateSlot::Prefix(index)),
            (None, Some(_)) => Some(CandidateSlot::PromptEnd),
            (None, None) => None,
        };
        let snapshot = match best {
            Some(CandidateSlot::Prefix(index)) => Some(&self.checkpoints[index]),
            Some(CandidateSlot::PromptEnd) => self.prompt_end_checkpoint.as_ref(),
            None => None,
        };
        if let Some(snapshot) = snapshot {
            let tokens_cap_before = self.tokens.capacity();
            self.tokens
                .try_reserve_exact(snapshot.tokens.len().saturating_sub(self.tokens.len()))?;
            if self.tokens.capacity() > tokens_cap_before {
                self.reallocation_witness.growth_reallocations = self
                    .reallocation_witness
                    .growth_reallocations
                    .saturating_add(1);
            }
            check()?;
            self.decoder.restore(&snapshot.state)?;
            check()?;
            self.tokens.clear();
            self.tokens.extend_from_slice(&snapshot.tokens);
            self.media_spans.clear();
            self.media_spans.extend_from_slice(&snapshot.media_spans);
            let previous_logits = std::mem::replace(&mut self.logits, Arc::clone(&snapshot.logits));
            self.recycle_logits(previous_logits);
        } else {
            check()?;
            self.decoder.reset()?;
            check()?;
            self.tokens.clear();
            self.media_spans.clear();
            mutable_logits(&mut self.logits).clear();
        }
        Ok(self.tokens.len())
    }

    fn evaluate(&mut self, token: u32, check: &impl Fn() -> Result<()>) -> Result<()> {
        check()?;
        // Reserve before mutating decoder state; rollback is still available if
        // a decoder allocation or arithmetic operation subsequently fails.
        let tokens_cap_before = self.tokens.capacity();
        self.tokens.try_reserve(1)?;
        if self.tokens.capacity() > tokens_cap_before {
            self.reallocation_witness.growth_reallocations = self
                .reallocation_witness
                .growth_reallocations
                .saturating_add(1);
        }
        let shared = Arc::strong_count(&self.logits) > 1;
        let reused_from_pool = shared && !self.batch_logits.is_empty();
        let prev_logits_cap = self.logits.capacity();
        let output = mutable_logits_with_scratch(&mut self.logits, &mut self.batch_logits);
        let output_cap_before = output.capacity();
        output.clear();
        check()?;
        let result = self
            .decoder
            .evaluate_finite_into(token, output)
            .map(|logits| (logits.as_slice().as_ptr(), logits.as_slice().len()))
            .and_then(|(pointer, length)| {
                // A public custom decoder can return a checked view of other
                // storage. Admit only a proof covering this complete output.
                if length != self.decoder.vocab_size()
                    || length != output.len()
                    || !std::ptr::eq(pointer, output.as_ptr())
                {
                    return Err(invalid("native decoder returned invalid logits"));
                }
                check()
            });
        let output_cap_after = output.capacity();
        if shared && !reused_from_pool && prev_logits_cap > 0 && output_cap_after == prev_logits_cap
        {
            self.reallocation_witness.same_size_reallocations = self
                .reallocation_witness
                .same_size_reallocations
                .saturating_add(1);
        } else if output_cap_after > output_cap_before {
            self.reallocation_witness.growth_reallocations = self
                .reallocation_witness
                .growth_reallocations
                .saturating_add(1);
        } else if output_cap_after > 0 {
            self.reallocation_witness.reused_preparations = self
                .reallocation_witness
                .reused_preparations
                .saturating_add(1);
        }
        if let Err(error) = result {
            output.clear();
            return Err(error);
        }
        self.tokens.push(token);
        Ok(())
    }

    fn ingest_block(&mut self, tokens: &[u32], check: &impl Fn() -> Result<()>) -> Result<()> {
        check()?;
        let tokens_cap_before = self.tokens.capacity();
        self.tokens.try_reserve(tokens.len())?;
        if self.tokens.capacity() > tokens_cap_before {
            self.reallocation_witness.growth_reallocations = self
                .reallocation_witness
                .growth_reallocations
                .saturating_add(1);
        }
        // Preserve the reusable allocation, but invalidate logits for the old
        // prefix before advancing state without producing a new distribution.
        mutable_logits(&mut self.logits).clear();
        check()?;
        self.decoder.ingest_block(tokens, check)?;
        check()?;
        self.tokens.extend_from_slice(tokens);
        Ok(())
    }

    fn evaluate_image_chunk(
        &mut self,
        prompt_slice: &[u32],
        chunk: &MultimodalChunk<'_>,
        want_logits: bool,
        check: &impl Fn() -> Result<()>,
    ) -> Result<()> {
        check()?;
        let tokens_cap_before = self.tokens.capacity();
        self.tokens.try_reserve(chunk.span.len)?;
        if self.tokens.capacity() > tokens_cap_before {
            self.reallocation_witness.growth_reallocations = self
                .reallocation_witness
                .growth_reallocations
                .saturating_add(1);
        }
        self.media_spans.try_reserve(1)?;
        if want_logits {
            let shared = Arc::strong_count(&self.logits) > 1;
            let reused_from_pool = shared && !self.batch_logits.is_empty();
            let prev_logits_cap = self.logits.capacity();
            let output = mutable_logits_with_scratch(&mut self.logits, &mut self.batch_logits);
            let output_cap_before = output.capacity();
            output.clear();
            check()?;
            let result = self
                .decoder
                .evaluate_image_chunk(
                    chunk.embeddings,
                    chunk.span.grid_width,
                    chunk.span.grid_height,
                    Some(output),
                    check,
                )
                .and_then(|()| {
                    if output.len() != self.decoder.vocab_size()
                        || !output.iter().all(|value| value.is_finite())
                    {
                        return Err(invalid(
                            "native decoder returned invalid image-chunk logits",
                        ));
                    }
                    check()
                });
            let output_cap_after = output.capacity();
            if shared
                && !reused_from_pool
                && prev_logits_cap > 0
                && output_cap_after == prev_logits_cap
            {
                self.reallocation_witness.same_size_reallocations = self
                    .reallocation_witness
                    .same_size_reallocations
                    .saturating_add(1);
            } else if output_cap_after > output_cap_before {
                self.reallocation_witness.growth_reallocations = self
                    .reallocation_witness
                    .growth_reallocations
                    .saturating_add(1);
            } else if output_cap_after > 0 {
                self.reallocation_witness.reused_preparations = self
                    .reallocation_witness
                    .reused_preparations
                    .saturating_add(1);
            }
            if let Err(error) = result {
                output.clear();
                return Err(error);
            }
        } else {
            mutable_logits(&mut self.logits).clear();
            check()?;
            self.decoder.evaluate_image_chunk(
                chunk.embeddings,
                chunk.span.grid_width,
                chunk.span.grid_height,
                None,
                check,
            )?;
            check()?;
        }
        self.tokens.extend_from_slice(prompt_slice);
        self.media_spans.push(chunk.span);
        Ok(())
    }

    fn prefill_remaining(
        &mut self,
        prompt: &[u32],
        reused: usize,
        chunks: &[MultimodalChunk<'_>],
        prefill: Prefill,
        timings: &mut Option<&mut GenerationTimings>,
        check: &impl Fn() -> Result<()>,
    ) -> Result<()> {
        let mut cursor = reused;
        for chunk in chunks.iter().filter(|chunk| chunk.span.offset >= reused) {
            for block in prompt[cursor..chunk.span.offset].chunks(prefill.block_tokens) {
                measure(
                    timings,
                    GenerationPhase::PrefillIngestion,
                    block.len(),
                    || self.ingest_block(block, check),
                )?;
            }
            let chunk_end = chunk.span.offset + chunk.span.len;
            let want_logits = chunk_end == prompt.len();
            let phase = if want_logits {
                GenerationPhase::PrefillEvaluation
            } else {
                GenerationPhase::PrefillIngestion
            };
            measure(timings, phase, chunk.span.len, || {
                self.evaluate_image_chunk(
                    &prompt[chunk.span.offset..chunk_end],
                    chunk,
                    want_logits,
                    check,
                )
            })?;
            cursor = chunk_end;
        }
        if let Some((&last, prefix)) = prompt[cursor..].split_last() {
            for block in prefix.chunks(prefill.block_tokens) {
                measure(
                    timings,
                    GenerationPhase::PrefillIngestion,
                    block.len(),
                    || self.ingest_block(block, check),
                )?;
            }
            measure(timings, GenerationPhase::PrefillEvaluation, 1, || {
                self.evaluate(last, check)
            })?;
        }
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Text and multimodal execution share one guarded transaction sequence."
    )]
    fn generate_inner(
        &mut self,
        prompt: &[u32],
        chunks: &[MultimodalChunk<'_>],
        options: &Generation,
        prefill: Prefill,
        check: &impl Fn() -> Result<()>,
        effects: &mut GenerationEffects<'_, impl Sampler, impl FnMut(u32) -> Result<()>>,
        prompt_end: &mut PromptEndCapture<'_, D::Checkpoint>,
    ) -> Result<Completion> {
        check()?;
        let incoming_spans: Vec<MediaSpan> = chunks.iter().map(|chunk| chunk.span).collect();
        let reused = measure(
            &mut effects.timings,
            GenerationPhase::PrefixPreparation,
            0,
            || self.prepare_with_spans(prompt, &incoming_spans, check),
        )?;
        if let Some(timings) = effects.timings.as_deref_mut() {
            timings.reused_tokens = Some(reused);
        }
        check()?;
        let remaining = prompt.len() - reused;
        let evaluations = options
            .max_tokens
            .checked_add(usize::from(remaining > 0))
            .ok_or_else(|| invalid("native forward work count overflow"))?;
        self.decoder
            .admit_work(remaining.saturating_sub(1), evaluations)?;
        check()?;
        self.prefill_remaining(prompt, reused, chunks, prefill, &mut effects.timings, check)?;
        self.maybe_capture_prompt_end_with_spans(prompt, &incoming_spans, prompt_end, check)?;
        let mut completion = Completion {
            prompt_tokens: prompt.len(),
            reused_tokens: reused,
            generated_tokens: 0,
            finish: Finish::Length,
        };
        for index in 0..options.max_tokens {
            check()?;
            let token = measure(&mut effects.timings, GenerationPhase::Sampling, 1, || {
                effects
                    .sampler
                    .sample_finite(FiniteLogits::from_validated(&self.logits)?)
            })?;
            check()?;
            if !usize::try_from(token).is_ok_and(|value| value < self.decoder.vocab_size()) {
                return Err(invalid(
                    "native sampler returned an out-of-vocabulary token",
                ));
            }
            if options.stop_tokens.contains(&token) {
                completion.finish = Finish::Stop;
                break;
            }
            if effects.terminal_state_only && index + 1 == options.max_tokens {
                // No subsequent sample can consume this token's logits, and
                // the reset-only owner cannot reuse the finished prefix. Keep
                // complete state and invalidate old logits before advancing.
                measure(
                    &mut effects.timings,
                    GenerationPhase::TerminalIngestion,
                    1,
                    || self.ingest_block(&[token], check),
                )?;
            } else {
                measure(
                    &mut effects.timings,
                    GenerationPhase::DecodeEvaluation,
                    1,
                    || self.evaluate(token, check),
                )?;
            }
            check()?;
            measure(&mut effects.timings, GenerationPhase::Callback, 1, || {
                (effects.emit)(token)
            })?;
            check()?;
            completion.generated_tokens += 1;
        }
        Ok(completion)
    }

    /// Generate greedy tokens with training-free prompt/history suffix speculation
    /// (`SuffixDecoding`) and exact target verification.
    ///
    /// Preserves bit-identical greedy output tokens, stop-token semantics, and
    /// whole-request rollback on cancellation or error.
    ///
    /// # Errors
    /// Rejects invalid suffix draft configs or any [`Self::generate_with_prefill`] error.
    pub fn generate_speculative(
        &mut self,
        prompt: &[u32],
        options: &Generation,
        prefill: Prefill,
        suffix_config: SuffixDraftConfig,
        cancellation: &Cancellation,
        emit: impl FnMut(u32) -> Result<()>,
    ) -> Result<(Completion, SpeculativeStats)> {
        self.generate_speculative_checked(
            prompt,
            options,
            prefill,
            suffix_config,
            &|| cancellation.check(),
            emit,
        )
    }

    /// Generate greedy tokens with prompt/history suffix speculation before a deadline.
    ///
    /// # Errors
    /// Rejects invalid suffix draft configs, deadline expiry, or any
    /// [`Self::generate_until_with_prefill`] error.
    #[expect(
        clippy::too_many_arguments,
        reason = "Explicit suffix draft configuration extends the bounded deadline API."
    )]
    pub fn generate_speculative_until(
        &mut self,
        prompt: &[u32],
        options: &Generation,
        prefill: Prefill,
        suffix_config: SuffixDraftConfig,
        cancellation: &Cancellation,
        deadline: impl Into<Option<Instant>>,
        emit: impl FnMut(u32) -> Result<()>,
    ) -> Result<(Completion, SpeculativeStats)> {
        let deadline = deadline.into();
        self.generate_speculative_checked(
            prompt,
            options,
            prefill,
            suffix_config,
            &|| check_deadline(cancellation, deadline, Instant::now()),
            emit,
        )
    }

    /// Verify one bounded block of draft continuation tokens against the target decoder.
    ///
    /// Preserves [`Self::verify_draft_block`] transactionally while enforcing
    /// `limits.max_new_tokens` and `limits.stop_tokens` so verification never
    /// emits or commits surplus bonus/correction tokens beyond the stop or length limit.
    ///
    /// # Errors
    /// Rejects invalid runtime state, invalid draft/limit bounds, context overflow,
    /// cancellation, or decoder checkpoint/restore failures.
    pub fn verify_draft_block_bounded(
        &mut self,
        draft: &[u32],
        limits: VerificationLimits<'_>,
        cancellation: &Cancellation,
    ) -> Result<(Vec<u32>, SpeculativeStats)> {
        self.verify_draft_block_inner(draft, limits, cancellation)
    }

    /// Speculatively advance and verify a short draft chain (`1..=8` tokens) against
    /// the current committed state, rolling back the decoder checkpoint on the first
    /// rejected draft token and committing all accepted tokens plus one bonus/correction token.
    ///
    /// # Errors
    /// Rejects faulted runtimes, empty committed state, invalid draft lengths or token IDs,
    /// context overflow, cancellation, or decoder checkpoint/restore failures.
    pub fn verify_draft_block(
        &mut self,
        draft: &[u32],
        cancellation: &Cancellation,
    ) -> Result<(Vec<u32>, SpeculativeStats)> {
        self.verify_draft_block_inner(
            draft,
            VerificationLimits {
                max_new_tokens: usize::MAX,
                stop_tokens: &[],
            },
            cancellation,
        )
    }

    fn verify_draft_block_inner(
        &mut self,
        draft: &[u32],
        limits: VerificationLimits<'_>,
        cancellation: &Cancellation,
    ) -> Result<(Vec<u32>, SpeculativeStats)> {
        if self.faulted {
            return Err(invalid(
                "native runtime requires reset after failed restoration",
            ));
        }
        cancellation.check()?;
        if self.tokens.is_empty() || self.logits.is_empty() {
            return Err(invalid(
                "speculative block verification requires a prepared nonempty prefix",
            ));
        }
        if draft.is_empty()
            || draft.len() > MAX_SPECULATIVE_DRAFT_TOKENS
            || limits.stop_tokens.len() > 256
            || draft.iter().any(|&token| {
                !usize::try_from(token).is_ok_and(|value| value < self.decoder.vocab_size())
            })
        {
            return Err(invalid("invalid speculative draft block bounds or tokens"));
        }
        if limits.max_new_tokens == 0 {
            return Ok((Vec::new(), SpeculativeStats::default()));
        }
        let effective_draft = &draft[..draft.len().min(limits.max_new_tokens)];
        let allow_bonus = effective_draft.len() < limits.max_new_tokens;
        let max_advance = effective_draft
            .len()
            .checked_add(usize::from(allow_bonus))
            .ok_or_else(|| invalid("speculative context length overflow"))?;
        if self
            .tokens
            .len()
            .checked_add(max_advance)
            .is_none_or(|len| len > self.limits.context_tokens)
        {
            return Err(invalid(
                "speculative draft block exceeds context token limit",
            ));
        }
        let batched = effective_draft.len() > 1
            && effective_draft.len() <= self.decoder.batched_evaluation_capacity();
        // A rejection pays for every proposed token plus replay of the accepted
        // prefix and its correction. Device work is never refunded by restore.
        let evaluations = if batched {
            effective_draft
                .len()
                .checked_mul(2)
                .and_then(|tokens| tokens.checked_add(usize::from(allow_bonus)))
                .ok_or_else(|| invalid("speculative verification work overflow"))?
        } else {
            max_advance
        };
        self.decoder.admit_work(0, evaluations)?;
        let check = || cancellation.check();
        let rollback = self.capture_transient(0, &check)?;
        let result = if batched {
            self.verify_batched_draft(effective_draft, limits, &rollback, &check)
        } else {
            self.verify_sequential_draft(effective_draft, limits, allow_bonus, max_advance, &check)
        };
        match result {
            Ok((committed, stats)) => {
                self.recycle_snapshot(rollback);
                Ok((committed, stats))
            }
            Err(error) => {
                if self.faulted {
                    // A failed accepted-prefix restore already closed admission.
                    // Do not retry that restore or resume from provisional state.
                    return Err(error);
                }
                self.faulted = true;
                let Snapshot {
                    state,
                    tokens,
                    media_spans,
                    logits,
                    ..
                } = rollback;
                let restored = self.decoder.restore(&state);
                self.tokens = tokens;
                self.media_spans = media_spans;
                let previous_logits = std::mem::replace(&mut self.logits, logits);
                restored?;
                self.faulted = false;
                self.recycle_checkpoint(state);
                self.recycle_logits(previous_logits);
                Err(error)
            }
        }
    }

    fn verify_sequential_draft(
        &mut self,
        effective_draft: &[u32],
        limits: VerificationLimits<'_>,
        allow_bonus: bool,
        max_advance: usize,
        check: &impl Fn() -> Result<()>,
    ) -> Result<(Vec<u32>, SpeculativeStats)> {
        let mut stats = SpeculativeStats::default();
        let mut committed = Vec::new();
        committed.try_reserve_exact(max_advance)?;
        let mut accepted = 0_usize;
        let mut verified = 0_usize;
        let mut target_evals = 0_usize;
        let mut finished_early = false;
        let mut rolled_back = false;
        let mut correction = 0_u64;
        let mut bonus_emitted = 0_u64;
        for &candidate in effective_draft {
            check()?;
            let expected = greedy_finite(FiniteLogits::from_validated(&self.logits)?)?;
            verified += 1;
            if limits.stop_tokens.contains(&expected) {
                finished_early = true;
                break;
            }
            if candidate == expected {
                self.evaluate(candidate, check)?;
                target_evals += 1;
                accepted += 1;
                committed.push(candidate);
            } else {
                // Candidate diverged from greedy target verification before mutating
                // the committed GDN + KV state: skip the rejected forward pass and
                // per-step GDN checkpoint clone, and advance directly with the exact
                // target correction token.
                rolled_back = true;
                correction = 1;
                self.evaluate(expected, check)?;
                target_evals += 1;
                committed.push(expected);
                finished_early = true;
                break;
            }
        }
        if !finished_early && allow_bonus {
            // All draft tokens were accepted and token budget permits one bonus token.
            check()?;
            let bonus = greedy_finite(FiniteLogits::from_validated(&self.logits)?)?;
            if !limits.stop_tokens.contains(&bonus) {
                self.evaluate(bonus, check)?;
                target_evals += 1;
                bonus_emitted = 1;
                committed.push(bonus);
            }
        }
        stats.record_verified_round(VerifiedRound {
            proposed: effective_draft.len(),
            accepted,
            verified,
            target_evaluations: target_evals,
            rolled_back,
            bonus_tokens: bonus_emitted,
            correction_tokens: correction,
            fallback_tokens: 0,
            replay_tokens: 0,
        });
        Ok((committed, stats))
    }

    fn generate_speculative_checked(
        &mut self,
        prompt: &[u32],
        options: &Generation,
        prefill: Prefill,
        suffix_config: SuffixDraftConfig,
        check: &impl Fn() -> Result<()>,
        mut emit: impl FnMut(u32) -> Result<()>,
    ) -> Result<(Completion, SpeculativeStats)> {
        if self.faulted {
            return Err(invalid(
                "native runtime requires reset after failed restoration",
            ));
        }
        suffix_config.validate()?;
        check()?;
        self.validate_request(prompt, options, prefill)?;
        check()?;
        let rollback = self.capture_transient(0, check)?;
        let mut prompt_end = None;
        let result = (|| {
            check()?;
            let mut prompt_end_capture = PromptEndCapture {
                reserved: rollback.bytes,
                slot: &mut prompt_end,
            };
            let (completion, stats) = self.generate_speculative_inner(
                prompt,
                options,
                prefill,
                suffix_config,
                check,
                &mut emit,
                &mut prompt_end_capture,
            )?;
            check()?;
            let snapshot = if self.limits.checkpoints == 0 {
                None
            } else {
                let snapshot_reserved = rollback
                    .bytes
                    .saturating_add(prompt_end.as_ref().map_or(0, |s| s.bytes));
                let snapshot = self.capture_reusable(snapshot_reserved, check).ok();
                check()?;
                snapshot
            };
            Ok((completion, stats, snapshot))
        })();
        match result {
            Ok((completion, stats, snapshot)) => {
                self.commit_prompt_end(prompt_end);
                self.commit_reusable_snapshot(snapshot);
                self.recycle_snapshot(rollback);
                Ok((completion, stats))
            }
            Err(error) => {
                self.faulted = true;
                let Snapshot {
                    state,
                    tokens,
                    media_spans,
                    logits,
                    ..
                } = rollback;
                let restored = self.decoder.restore(&state);
                self.tokens = tokens;
                self.media_spans = media_spans;
                let previous_logits = std::mem::replace(&mut self.logits, logits);
                restored?;
                self.faulted = false;
                self.recycle_checkpoint(state);
                self.recycle_logits(previous_logits);
                self.recycle_optional_snapshot(prompt_end);
                Err(error)
            }
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Keep suffix configuration and transactional prompt-end capture explicit."
    )]
    fn generate_speculative_inner(
        &mut self,
        prompt: &[u32],
        options: &Generation,
        prefill: Prefill,
        suffix_config: SuffixDraftConfig,
        check: &impl Fn() -> Result<()>,
        emit: &mut impl FnMut(u32) -> Result<()>,
        prompt_end: &mut PromptEndCapture<'_, D::Checkpoint>,
    ) -> Result<(Completion, SpeculativeStats)> {
        check()?;
        let reused = self.prepare(prompt, check)?;
        check()?;
        let remaining = prompt.len() - reused;
        let evaluations = options
            .max_tokens
            .checked_add(usize::from(remaining > 0))
            .ok_or_else(|| invalid("native forward work count overflow"))?;
        self.decoder
            .admit_work(remaining.saturating_sub(1), evaluations)?;
        check()?;
        if let Some((&last, prefix)) = prompt[reused..].split_last() {
            for block in prefix.chunks(prefill.block_tokens) {
                self.ingest_block(block, check)?;
            }
            self.evaluate(last, check)?;
        }
        self.maybe_capture_prompt_end(prompt, prompt_end, check)?;
        let mut completion = Completion {
            prompt_tokens: prompt.len(),
            reused_tokens: reused,
            generated_tokens: 0,
            finish: Finish::Length,
        };
        let mut stats = SpeculativeStats::default();
        let mut draft_storage = [0; MAX_SPECULATIVE_DRAFT_TOKENS];
        while completion.generated_tokens < options.max_tokens {
            check()?;
            let remaining_budget = options.max_tokens - completion.generated_tokens;
            let draft_len = SuffixDrafter::propose_into(
                &self.tokens,
                suffix_config,
                remaining_budget,
                &mut draft_storage,
            )?;
            let draft = &draft_storage[..draft_len];
            if self.verify_greedy_draft_round(
                draft,
                options,
                check,
                emit,
                &mut completion,
                &mut stats,
            )? {
                break;
            }
        }
        Ok((completion, stats))
    }

    fn verify_greedy_draft_round(
        &mut self,
        draft: &[u32],
        options: &Generation,
        check: &impl Fn() -> Result<()>,
        emit: &mut impl FnMut(u32) -> Result<()>,
        completion: &mut Completion,
        stats: &mut SpeculativeStats,
    ) -> Result<bool> {
        let mut stopped = false;
        let mut correction = None;
        let mut accepted = 0_usize;
        let mut verified = 0_usize;
        let mut evals = 0_usize;
        for &candidate in draft {
            check()?;
            let target_token = greedy_finite(FiniteLogits::from_validated(&self.logits)?)?;
            check()?;
            verified += 1;
            if target_token != candidate {
                // No evaluation or callback occurs before correction,
                // so these same validated logits still select this token.
                // Avoid scanning the complete vocabulary a second time.
                correction = Some(target_token);
                break;
            }
            accepted += 1;
            if options.stop_tokens.contains(&target_token) {
                completion.finish = Finish::Stop;
                stopped = true;
                break;
            }
            self.evaluate(target_token, check)?;
            evals += 1;
            check()?;
            emit(target_token)?;
            check()?;
            completion.generated_tokens += 1;
        }
        let record_terminal = |stats: &mut SpeculativeStats| {
            if !draft.is_empty() {
                stats.record_verified_round(VerifiedRound {
                    proposed: draft.len(),
                    accepted,
                    verified,
                    target_evaluations: evals,
                    rolled_back: false,
                    bonus_tokens: 0,
                    correction_tokens: 0,
                    fallback_tokens: 0,
                    replay_tokens: 0,
                });
            }
        };
        if stopped || completion.generated_tokens >= options.max_tokens {
            record_terminal(stats);
            return Ok(true);
        }
        check()?;
        let token = correction.map_or_else(
            || greedy_finite(FiniteLogits::from_validated(&self.logits)?),
            Ok,
        )?;
        check()?;
        if options.stop_tokens.contains(&token) {
            record_terminal(stats);
            completion.finish = Finish::Stop;
            return Ok(true);
        }
        self.evaluate(token, check)?;
        evals += 1;
        if draft.is_empty() {
            stats.record_verified_round(VerifiedRound {
                proposed: 0,
                accepted: 0,
                verified: 0,
                target_evaluations: 1,
                rolled_back: false,
                bonus_tokens: 0,
                correction_tokens: 0,
                fallback_tokens: 1,
                replay_tokens: 0,
            });
        } else {
            stats.record_verified_round(VerifiedRound {
                proposed: draft.len(),
                accepted,
                verified,
                target_evaluations: evals,
                rolled_back: false,
                bonus_tokens: u64::from(correction.is_none()),
                correction_tokens: u64::from(correction.is_some()),
                fallback_tokens: 0,
                replay_tokens: 0,
            });
        }
        check()?;
        emit(token)?;
        check()?;
        completion.generated_tokens += 1;
        Ok(false)
    }
}

fn prefix_matches_media(
    prompt: &[u32],
    incoming_spans: &[MediaSpan],
    prefix_tokens: &[u32],
    prefix_media: &[MediaSpan],
) -> bool {
    prompt.starts_with(prefix_tokens)
        && incoming_spans
            .iter()
            .take_while(|span| span.offset < prefix_tokens.len())
            .copied()
            .eq(prefix_media.iter().copied())
}

fn validate_multimodal_chunks(prompt: &[u32], chunks: &[MultimodalChunk<'_>]) -> Result<()> {
    let mut prev_end = 0_usize;
    for chunk in chunks {
        let span = chunk.span;
        if span.grid_width == 0
            || span.grid_height == 0
            || span.grid_width.checked_mul(span.grid_height) != Some(span.len)
            || span.offset < prev_end
        {
            return Err(invalid("invalid or overlapping multimodal span"));
        }
        let Some(end) = span.offset.checked_add(span.len) else {
            return Err(invalid("multimodal span offset overflow"));
        };
        if end > prompt.len() {
            return Err(invalid("multimodal span extends beyond prompt tokens"));
        }
        if chunk.embeddings.is_empty()
            || !chunk.embeddings.len().is_multiple_of(span.len)
            || !chunk.embeddings.iter().all(|value| value.is_finite())
        {
            return Err(invalid("invalid or nonfinite multimodal embeddings"));
        }
        prev_end = end;
    }
    Ok(())
}

fn check_deadline(
    cancellation: &Cancellation,
    deadline: impl Into<Option<Instant>>,
    now: Instant,
) -> Result<()> {
    cancellation.check()?;
    if deadline.into().is_some_and(|deadline| now >= deadline) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            "native generation deadline exceeded",
        )
        .into());
    }
    Ok(())
}

// A snapshot owns only an immutable Arc. Before writing a new distribution,
// detach shared output without cloning values that will immediately be cleared.
// The fresh allocation then remains unique and reusable across decoder steps.
fn mutable_logits_with_scratch<'a>(
    logits: &'a mut Arc<Vec<f32>>,
    scratch: &mut Vec<Vec<f32>>,
) -> &'a mut Vec<f32> {
    if let Some(values) = Arc::get_mut(logits) {
        return values;
    }
    let next = scratch.pop().unwrap_or_default();
    *logits = Arc::new(next);
    Arc::make_mut(logits)
}

fn mutable_logits(logits: &mut Arc<Vec<f32>>) -> &mut Vec<f32> {
    if let Some(values) = Arc::get_mut(logits) {
        return values;
    }
    *logits = Arc::new(Vec::new());
    Arc::make_mut(logits)
}

fn copy<T: Copy>(source: &[T]) -> Result<Vec<T>> {
    let mut output = Vec::new();
    output.try_reserve_exact(source.len())?;
    output.extend_from_slice(source);
    Ok(output)
}

/// Select the greatest finite logit, using the lowest token ID to break ties.
///
/// # Errors
/// Rejects empty, oversized or nonfinite logits.
pub fn greedy(logits: &[f32]) -> Result<u32> {
    if logits.is_empty() || logits.len() > 1_048_576 {
        return Err(invalid("greedy sampling requires nonempty finite logits"));
    }
    Ok(u32::try_from(greedy_inner::<true>(logits)?)?)
}

pub(crate) fn greedy_finite(logits: FiniteLogits<'_>) -> Result<u32> {
    Ok(u32::try_from(greedy_inner::<false>(logits.as_slice())?)?)
}

fn greedy_inner<const VALIDATE: bool>(logits: &[f32]) -> Result<usize> {
    let mut best = 0;
    let mut maximum = logits[0];
    if VALIDATE && !maximum.is_finite() {
        return Err(invalid("greedy sampling requires nonempty finite logits"));
    }
    let (blocks, remainder) = logits.as_chunks::<16>();
    for (block_index, block) in blocks.iter().enumerate() {
        let mut lanes = crate::sampling::selection::candidate_lanes::<VALIDATE>(block, maximum);
        let base = block_index * 16;
        while lanes != 0 {
            let offset = usize::try_from(lanes.trailing_zeros())?;
            lanes &= lanes - 1;
            let value = block[offset];
            // Validate every element even after finding an unbeatable maximum.
            // A strict comparison retains the lowest token ID for all ties,
            // including equal positive/negative zero values.
            if VALIDATE && !value.is_finite() {
                return Err(invalid("greedy sampling requires nonempty finite logits"));
            }
            if value > maximum {
                best = base + offset;
                maximum = value;
            }
        }
    }
    let base = blocks.len() * 16;
    for (offset, &value) in remainder.iter().enumerate() {
        if VALIDATE && !value.is_finite() {
            return Err(invalid("greedy sampling requires nonempty finite logits"));
        }
        if value > maximum {
            best = base + offset;
            maximum = value;
        }
    }
    Ok(best)
}

#[cfg(test)]
mod terminal_tests;

#[cfg(test)]
mod logits_tests;

#[cfg(test)]
mod deadline_tests {
    use super::*;
    use crate::speculative::MAX_SUFFIX_MATCH_TOKENS;
    use std::{
        cell::{Cell, RefCell},
        io::ErrorKind,
        rc::Rc,
        time::Duration,
    };

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum Operation {
        Reset,
        Restore,
        Checkpoint,
        Ingest(u32),
        Evaluate(u32),
        Sample,
        Emit(u32),
    }

    struct Clock {
        now: Cell<Instant>,
        deadline: Instant,
        expire_on: Cell<Option<Operation>>,
        operations: RefCell<Vec<Operation>>,
        restore_fail: Cell<bool>,
    }

    impl Clock {
        fn record(&self, operation: Operation) {
            self.operations.borrow_mut().push(operation);
            if self.expire_on.get() == Some(operation) {
                self.expire_on.set(None);
                self.now.set(self.deadline);
            }
        }

        fn check(&self) -> Result<()> {
            check_deadline(&Cancellation::default(), self.deadline, self.now.get())
        }
    }

    struct TimedDecoder {
        tokens: Vec<u32>,
        clock: Rc<Clock>,
    }

    impl Decoder for TimedDecoder {
        type Checkpoint = Vec<u32>;

        fn identity(&self) -> &'static str {
            "deadline-fixture/v1"
        }
        fn vocab_size(&self) -> usize {
            8
        }
        fn checkpoint_bytes(&self) -> usize {
            self.tokens.len() * 4
        }
        fn reset(&mut self) -> Result<()> {
            self.tokens.clear();
            self.clock.record(Operation::Reset);
            Ok(())
        }
        fn ingest(&mut self, token: u32) -> Result<()> {
            self.tokens.push(token);
            self.clock.record(Operation::Ingest(token));
            Ok(())
        }
        fn evaluate(&mut self, token: u32) -> Result<Vec<f32>> {
            self.tokens.push(token);
            self.clock.record(Operation::Evaluate(token));
            let mut logits = vec![-1.0; 8];
            logits[usize::try_from(self.tokens.iter().sum::<u32>() % 8)?] = 1.0;
            Ok(logits)
        }
        fn checkpoint(&self) -> Result<Self::Checkpoint> {
            self.clock.record(Operation::Checkpoint);
            Ok(self.tokens.clone())
        }
        fn restore(&mut self, checkpoint: &Self::Checkpoint) -> Result<()> {
            self.clock.record(Operation::Restore);
            if self.clock.restore_fail.get() {
                return Err(std::io::Error::other("injected deadline restoration failure").into());
            }
            self.tokens.clone_from(checkpoint);
            Ok(())
        }
    }

    const LIMITS: Limits = Limits {
        context_tokens: 32,
        checkpoints: 4,
        checkpoint_bytes: 16_384,
    };

    const OPTIONS: Generation = Generation {
        max_tokens: 1,
        stop_tokens: vec![],
    };

    #[derive(Clone)]
    struct TimedSampler {
        draws: usize,
        clock: Rc<Clock>,
    }

    impl Sampler for TimedSampler {
        fn sample(&mut self, logits: &[f32]) -> Result<u32> {
            self.draws += 1;
            self.clock.record(Operation::Sample);
            greedy(logits)
        }
    }

    fn seeded() -> Result<(Runtime<TimedDecoder>, Rc<Clock>)> {
        let now = Instant::now();
        let clock = Rc::new(Clock {
            now: Cell::new(now),
            deadline: now + Duration::from_secs(1),
            expire_on: Cell::new(None),
            operations: RefCell::new(Vec::new()),
            restore_fail: Cell::new(false),
        });
        let mut runtime = Runtime::new(
            TimedDecoder {
                tokens: Vec::new(),
                clock: clock.clone(),
            },
            LIMITS,
        )?;
        runtime.generate(&[1], &OPTIONS, &Cancellation::default(), |_| Ok(()))?;
        clock.operations.borrow_mut().clear();
        Ok((runtime, clock))
    }

    fn assert_timeout(result: Result<Completion>) -> Result<()> {
        let Err(error) = result else {
            return Err(invalid("deadline fixture unexpectedly completed"));
        };
        assert_eq!(
            error
                .downcast_ref::<std::io::Error>()
                .map(std::io::Error::kind),
            Some(ErrorKind::TimedOut)
        );
        Ok(())
    }

    fn assert_continues(runtime: &mut Runtime<TimedDecoder>) -> Result<()> {
        let committed = runtime.tokens().to_vec();
        let completion =
            runtime.generate(&committed, &OPTIONS, &Cancellation::default(), |_| Ok(()))?;
        assert_eq!(completion.reused_tokens, committed.len());
        let expected = committed.iter().sum::<u32>() % 8;
        assert_eq!(runtime.tokens().last(), Some(&expected));
        Ok(())
    }

    #[test]
    fn finite_greedy_matches_checked_selection_across_tiles_and_ties() -> Result<()> {
        for length in [1_usize, 15, 16, 17, 31, 32, 33, 257, 1_024] {
            for round in 0..6 {
                let mut logits = vec![0.0; length];
                for (index, value) in logits.iter_mut().enumerate() {
                    *value = match round {
                        0 => -0.0,
                        1 => {
                            if index.is_multiple_of(2) {
                                -0.0
                            } else {
                                0.0
                            }
                        }
                        2 => f32::MIN,
                        3 => f32::MAX,
                        4 => f32::from_bits(u32::try_from(index)?),
                        _ => f32::from(u8::try_from(index % 7)?) - 3.0,
                    };
                }
                let view = FiniteLogits::new(&logits)?;
                assert_eq!(greedy_finite(view)?, greedy(&logits)?);
                assert_eq!(Greedy.sample_finite(view)?, Greedy.sample(&logits)?);
                for index in 0..length {
                    logits[index] = f32::MAX;
                    assert_eq!(
                        greedy_finite(FiniteLogits::new(&logits)?)?,
                        greedy(&logits)?
                    );
                    logits[index] = 0.0;
                }
            }
        }
        Ok(())
    }

    #[test]
    fn deadline_backend_expiry_rolls_back_prefill_and_generation() -> Result<()> {
        for operation in [
            Operation::Checkpoint,
            Operation::Reset,
            Operation::Ingest(2),
            Operation::Evaluate(3),
            Operation::Evaluate(5),
        ] {
            let (mut runtime, clock) = seeded()?;
            clock.expire_on.set(Some(operation));
            assert_timeout(runtime.generate_checked(
                &[2, 3],
                &OPTIONS,
                Prefill::default(),
                &|| clock.check(),
                |_| Err(invalid("expired decoder output must not emit")),
            ))?;
            assert!(clock.operations.borrow().contains(&operation));
            assert_eq!(clock.operations.borrow().last(), Some(&Operation::Restore));
            assert_eq!(runtime.tokens(), &[1, 1]);
            assert!(!runtime.is_faulted());
            assert_continues(&mut runtime)?;
        }
        Ok(())
    }

    #[test]
    fn deadline_expiry_during_prefix_restore_rolls_back() -> Result<()> {
        let (mut runtime, clock) = seeded()?;
        runtime.generate(&[1, 1, 2], &OPTIONS, &Cancellation::default(), |_| Ok(()))?;
        clock.operations.borrow_mut().clear();
        clock.expire_on.set(Some(Operation::Restore));
        assert_timeout(runtime.generate_checked(
            &[1, 1, 3],
            &OPTIONS,
            Prefill::default(),
            &|| clock.check(),
            |_| Err(invalid("expired prefix restore must not emit")),
        ))?;
        assert_eq!(
            *clock.operations.borrow(),
            [
                Operation::Checkpoint,
                Operation::Restore,
                Operation::Restore
            ]
        );
        assert_eq!(runtime.tokens(), &[1, 1, 2, 4]);
        assert!(!runtime.is_faulted());
        assert_continues(&mut runtime)
    }

    #[test]
    fn deadline_expiry_during_emit_or_optional_snapshot_rolls_back() -> Result<()> {
        for operation in [Operation::Emit(5), Operation::Checkpoint] {
            let (mut runtime, clock) = seeded()?;
            let mut emitted = Vec::new();
            assert_timeout(runtime.generate_checked(
                &[2, 3],
                &OPTIONS,
                Prefill::default(),
                &|| clock.check(),
                |token| {
                    emitted.push(token);
                    // Arm after the mandatory rollback capture so Checkpoint
                    // targets the optional completed-prefix capture exclusively.
                    clock.expire_on.set(Some(operation));
                    clock.record(Operation::Emit(token));
                    Ok(())
                },
            ))?;
            assert_eq!(emitted, [5]);
            assert_eq!(clock.operations.borrow().last(), Some(&Operation::Restore));
            assert_eq!(runtime.tokens(), &[1, 1]);
            assert_eq!(runtime.checkpoint_count(), 1);
            assert!(!runtime.is_faulted());
            assert_continues(&mut runtime)?;
        }
        Ok(())
    }

    #[test]
    fn deadline_restoration_failure_latches_admission_until_reset() -> Result<()> {
        let (mut runtime, clock) = seeded()?;
        clock.restore_fail.set(true);
        clock.expire_on.set(Some(Operation::Ingest(2)));
        let result = runtime.generate_checked(
            &[2, 3],
            &OPTIONS,
            Prefill::default(),
            &|| clock.check(),
            |_| Ok(()),
        );
        let Err(error) = result else {
            return Err(invalid(
                "failed deadline restoration unexpectedly completed",
            ));
        };
        assert!(
            error
                .to_string()
                .contains("injected deadline restoration failure")
        );
        assert!(runtime.is_faulted());
        assert_eq!(runtime.tokens(), &[1, 1]);
        clock.operations.borrow_mut().clear();
        assert!(
            runtime
                .generate(&[1, 1], &OPTIONS, &Cancellation::default(), |_| Ok(()))
                .is_err()
        );
        assert!(clock.operations.borrow().is_empty());
        clock.restore_fail.set(false);
        runtime.reset()?;
        runtime.generate(&[3], &OPTIONS, &Cancellation::default(), |_| Ok(()))?;
        assert_eq!(runtime.tokens(), &[3, 3]);
        assert!(!runtime.is_faulted());
        Ok(())
    }

    #[test]
    fn deadline_prefill_expiry_inside_blocks_restores_the_whole_request() -> Result<()> {
        for operation in [
            Operation::Ingest(3),
            Operation::Ingest(7),
            Operation::Evaluate(0),
        ] {
            let (mut runtime, clock) = seeded()?;
            clock.expire_on.set(Some(operation));
            assert_timeout(runtime.generate_checked(
                &[2, 3, 4, 5, 6, 7, 0],
                &OPTIONS,
                Prefill { block_tokens: 2 },
                &|| clock.check(),
                |_| Err(invalid("expired prefill must not emit")),
            ))?;
            assert!(clock.operations.borrow().contains(&operation));
            assert_eq!(clock.operations.borrow().last(), Some(&Operation::Restore));
            assert_eq!(runtime.tokens(), &[1, 1]);
            assert_eq!(runtime.decoder().tokens, [1, 1]);
            assert!(!runtime.is_faulted());
            assert_continues(&mut runtime)?;
        }
        Ok(())
    }

    #[test]
    fn sampled_deadline_during_sampling_delivery_or_snapshot_restores_caller_rng() -> Result<()> {
        for operation in [Operation::Sample, Operation::Emit(5), Operation::Checkpoint] {
            let (mut runtime, clock) = seeded()?;
            let initial_now = clock.now.get();
            let mut sampler = TimedSampler {
                draws: 7,
                clock: clock.clone(),
            };
            if operation == Operation::Sample {
                clock.expire_on.set(Some(operation));
            }
            let mut emitted = Vec::new();
            assert_timeout(runtime.generate_sampled_checked(
                &[2, 3],
                &OPTIONS,
                Prefill::default(),
                &|| clock.check(),
                &mut sampler,
                |token| {
                    emitted.push(token);
                    clock.expire_on.set(Some(operation));
                    clock.record(Operation::Emit(token));
                    Ok(())
                },
            ))?;
            assert_eq!(sampler.draws, 7);
            assert_eq!(runtime.tokens(), &[1, 1]);
            assert_eq!(runtime.decoder().tokens, [1, 1]);
            assert!(!runtime.is_faulted());
            assert_eq!(emitted.len(), usize::from(operation != Operation::Sample));
            clock.now.set(initial_now);
            clock.expire_on.set(None);
            let completion = runtime.generate_sampled_checked(
                &[2, 3],
                &OPTIONS,
                Prefill::default(),
                &|| clock.check(),
                &mut sampler,
                |token| {
                    assert_eq!(token, 5);
                    Ok(())
                },
            )?;
            assert_eq!(completion.generated_tokens, 1);
            assert_eq!(sampler.draws, 8);
        }
        Ok(())
    }

    #[test]
    fn sampled_deadline_after_stop_draw_or_failed_recovery_restores_caller_rng() -> Result<()> {
        for fail_restore in [false, true] {
            let (mut runtime, clock) = seeded()?;
            let mut sampler = TimedSampler {
                draws: 7,
                clock: clock.clone(),
            };
            clock.restore_fail.set(fail_restore);
            clock.expire_on.set(Some(Operation::Sample));
            assert!(
                runtime
                    .generate_sampled_checked(
                        &[2, 3],
                        &Generation {
                            max_tokens: 1,
                            stop_tokens: vec![5],
                        },
                        Prefill::default(),
                        &|| clock.check(),
                        &mut sampler,
                        |_| Err(invalid("expired stop sample must not emit")),
                    )
                    .is_err()
            );
            assert_eq!(sampler.draws, 7);
            assert_eq!(runtime.tokens(), &[1, 1]);
            assert_eq!(runtime.is_faulted(), fail_restore);
            assert!(!clock.operations.borrow().contains(&Operation::Evaluate(5)));
        }
        Ok(())
    }

    #[test]
    fn speculative_suffix_generation_and_block_verifier_preserve_exact_greedy_state() -> Result<()>
    {
        let now = Instant::now();
        let make_runtime = || -> Result<(Runtime<TimedDecoder>, Rc<Clock>)> {
            let clock = Rc::new(Clock {
                now: Cell::new(now),
                deadline: now + Duration::from_secs(10),
                expire_on: Cell::new(None),
                operations: RefCell::new(Vec::new()),
                restore_fail: Cell::new(false),
            });
            let runtime = Runtime::new(
                TimedDecoder {
                    tokens: Vec::new(),
                    clock: clock.clone(),
                },
                LIMITS,
            )?;
            Ok((runtime, clock))
        };

        // Prompt [0, 0, 0, 0, 0] causes TimedDecoder (sum % 8) to emit 0 repeatedly,
        // so SuffixDrafter matches [0, 0] in history and proposes [0, 0] drafts that are accepted!
        let prompt = [0_u32, 0, 0, 0, 0];
        let gen_opts = Generation {
            max_tokens: 6,
            stop_tokens: vec![],
        };
        let (mut baseline_rt, _) = make_runtime()?;
        let mut baseline_out = Vec::new();
        let baseline_comp =
            baseline_rt.generate(&prompt, &gen_opts, &Cancellation::default(), |t| {
                baseline_out.push(t);
                Ok(())
            })?;

        let (mut spec_rt, _) = make_runtime()?;
        let mut spec_out = Vec::new();
        let (spec_comp, spec_stats) = spec_rt.generate_speculative(
            &prompt,
            &gen_opts,
            Prefill::default(),
            SuffixDraftConfig::new(2, 8, 3)?,
            &Cancellation::default(),
            |t| {
                spec_out.push(t);
                Ok(())
            },
        )?;
        assert_eq!(spec_comp, baseline_comp);
        assert_eq!(spec_out, baseline_out);
        assert_eq!(spec_rt.tokens(), baseline_rt.tokens());
        assert!(spec_stats.draft_rounds > 0);
        assert!(spec_stats.draft_tokens_accepted > 0);

        // Now test verify_draft_block with a draft chain where token 0 is accepted and token 1 (7 != 0) is rejected:
        let (rendered, verify_stats) =
            spec_rt.verify_draft_block(&[0, 7, 0], &Cancellation::default())?;
        assert_eq!(rendered, vec![0, 0]);
        assert_eq!(verify_stats.draft_tokens_proposed, 3);
        assert_eq!(verify_stats.draft_tokens_accepted, 1);
        assert_eq!(verify_stats.rollback_restores, 1);
        Ok(())
    }

    #[test]
    fn speculative_suffix_buffered_rounds_restore_state_after_cancel_and_delivery_failure()
    -> Result<()> {
        for cancel in [false, true] {
            let (mut runtime, _) = seeded()?;
            let before_tokens = runtime.tokens.clone();
            let before_logits = runtime
                .logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>();
            let cancellation = Cancellation::default();
            let mut emitted = 0;
            let result = runtime.generate_speculative(
                &[0, 0, 0, 0, 0],
                &Generation {
                    max_tokens: 6,
                    stop_tokens: vec![],
                },
                Prefill::default(),
                SuffixDraftConfig::new(1, MAX_SUFFIX_MATCH_TOKENS, 1)?,
                &cancellation,
                |_| {
                    emitted += 1;
                    if emitted == 3 {
                        if cancel {
                            cancellation.cancel();
                        } else {
                            return Err(invalid("injected speculative delivery failure"));
                        }
                    }
                    Ok(())
                },
            );
            assert!(result.is_err());
            assert_eq!(emitted, 3);
            assert_eq!(runtime.tokens, before_tokens);
            assert_eq!(runtime.decoder.tokens, before_tokens);
            assert_eq!(
                runtime
                    .logits
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                before_logits
            );
            assert!(!runtime.is_faulted());
            assert_continues(&mut runtime)?;
        }
        Ok(())
    }

    #[test]
    fn speculative_rejected_suffix_correction_matches_independent_greedy_requests() -> Result<()> {
        // The first prompt rejects its first proposed token. The second accepts
        // historical token 1, then rejects historical token 0 against target 2.
        // Later zero runs exercise accepted chains and ordinary bonus tokens.
        for (prompt, expected) in [
            (&[1, 2, 1, 2][..], &[6, 4, 0, 0, 0, 0][..]),
            (&[0, 0, 1, 0, 0][..], &[1, 2, 4, 0, 0, 0][..]),
        ] {
            for stop in [None, Some(expected[0]), Some(expected[1]), Some(0)] {
                for max_tokens in [1, 2, expected.len()] {
                    let options = Generation {
                        max_tokens,
                        stop_tokens: stop.into_iter().collect(),
                    };
                    let (mut ordinary, ordinary_clock) = seeded()?;
                    let (mut speculative, speculative_clock) = seeded()?;
                    let mut ordinary_output = Vec::new();
                    let ordinary_completion =
                        ordinary.generate(prompt, &options, &Cancellation::default(), |token| {
                            ordinary_output.push(token);
                            Ok(())
                        })?;
                    let mut speculative_output = Vec::new();
                    let (speculative_completion, stats) = speculative.generate_speculative(
                        prompt,
                        &options,
                        Prefill::default(),
                        SuffixDraftConfig::new(2, 8, 3)?,
                        &Cancellation::default(),
                        |token| {
                            speculative_output.push(token);
                            Ok(())
                        },
                    )?;
                    let expected_output = expected[..max_tokens]
                        .iter()
                        .copied()
                        .take_while(|token| Some(*token) != stop)
                        .collect::<Vec<_>>();
                    assert_eq!(ordinary_output, expected_output);
                    assert_eq!(speculative_output, expected_output);
                    assert_eq!(speculative_completion, ordinary_completion);
                    assert_eq!(speculative.tokens(), ordinary.tokens());
                    assert_eq!(speculative.decoder.tokens, ordinary.decoder.tokens);
                    assert_eq!(
                        speculative
                            .logits
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>(),
                        ordinary
                            .logits
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(
                        *speculative_clock.operations.borrow(),
                        *ordinary_clock.operations.borrow()
                    );
                    assert!(stats.draft_rounds > 0);
                    assert_eq!(stats.rollback_restores, 0);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn speculative_rejected_suffix_keeps_both_correction_cancellation_boundaries() -> Result<()> {
        // After the final prompt evaluation, checks 1..=4 cover publication,
        // the round, the first candidate and its greedy selection. Checks 5
        // and 6 are the boundaries before and after correction selection.
        for fail_at in [5, 6] {
            let (mut runtime, clock) = seeded()?;
            let before_tokens = runtime.tokens.clone();
            let before_logits = runtime
                .logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>();
            let checks_after_prompt = Cell::new(0);
            let check = || {
                if clock.operations.borrow().last() == Some(&Operation::Evaluate(2)) {
                    let count = checks_after_prompt.get() + 1;
                    checks_after_prompt.set(count);
                    if count == fail_at {
                        return Err(std::io::Error::new(
                            ErrorKind::Interrupted,
                            "injected correction cancellation",
                        )
                        .into());
                    }
                }
                Ok(())
            };
            let result = runtime.generate_speculative_checked(
                &[1, 2, 1, 2],
                &Generation {
                    max_tokens: 6,
                    stop_tokens: vec![],
                },
                Prefill::default(),
                SuffixDraftConfig::new(2, 8, 3)?,
                &check,
                |_| Err(invalid("cancelled correction must not emit")),
            );
            let Err(error) = result else {
                return Err(invalid("correction cancellation unexpectedly completed"));
            };
            assert_eq!(
                error
                    .downcast_ref::<std::io::Error>()
                    .map(std::io::Error::kind),
                Some(ErrorKind::Interrupted)
            );
            assert_eq!(checks_after_prompt.get(), fail_at);
            assert!(!clock.operations.borrow().contains(&Operation::Evaluate(6)));
            assert_eq!(clock.operations.borrow().last(), Some(&Operation::Restore));
            assert_eq!(runtime.tokens, before_tokens);
            assert_eq!(runtime.decoder.tokens, before_tokens);
            assert_eq!(
                runtime
                    .logits
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                before_logits
            );
            assert!(!runtime.is_faulted());
            assert_continues(&mut runtime)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod optional_deadline_tests {
    use super::{Cancellation, check_deadline};
    use std::time::{Duration, Instant};

    #[test]
    fn uncapped_generation_checks_cancellation_without_an_elapsed_cutoff() {
        let cancel = Cancellation::default();
        let start = Instant::now();
        let later = start + Duration::from_mins(11);
        assert!(check_deadline(&cancel, None, later).is_ok());
        let expired = check_deadline(&cancel, start + Duration::from_mins(10), later);
        assert!(expired.is_err_and(|error| {
            error
                .downcast_ref::<std::io::Error>()
                .is_some_and(|error| error.kind() == std::io::ErrorKind::TimedOut)
        }));
        cancel.cancel();
        let cancelled = check_deadline(&cancel, None, later);
        assert!(cancelled.is_err_and(|error| {
            error
                .downcast_ref::<std::io::Error>()
                .is_some_and(|error| error.kind() == std::io::ErrorKind::Interrupted)
        }));
    }
}

#[cfg(test)]
mod recycling_tests {
    use super::{
        Arc, Cancellation, Decoder, Generation, Limits, MAX_RECYCLED_BATCH_LOGITS,
        MAX_RECYCLED_CHECKPOINTS, Prefill, Result, Runtime, SuffixDraftConfig,
    };
    use std::cell::{Cell, RefCell};

    struct RecycledDecoder {
        tokens: Vec<u32>,
        checkpoint_into_calls: Cell<usize>,
        checkpoint_reused_count: Cell<usize>,
        batch_row_initial_caps: RefCell<Vec<usize>>,
        vocab: usize,
    }

    impl RecycledDecoder {
        fn new() -> Self {
            Self {
                tokens: Vec::new(),
                checkpoint_into_calls: Cell::new(0),
                checkpoint_reused_count: Cell::new(0),
                batch_row_initial_caps: RefCell::new(Vec::new()),
                vocab: 8,
            }
        }

        fn fill_logits(tokens: &[u32], vocab: usize, output: &mut Vec<f32>) -> Result<()> {
            output.clear();
            output.try_reserve_exact(vocab)?;
            output.resize(vocab, 0.0);
            let last = tokens.last().copied().unwrap_or(0);
            let next = usize::try_from((last + 1) % u32::try_from(vocab)?)?;
            output[next] = 10.0;
            Ok(())
        }
    }

    impl Decoder for RecycledDecoder {
        type Checkpoint = Vec<u32>;

        fn identity(&self) -> &'static str {
            "recycled-checkpoint-test/v1"
        }

        fn vocab_size(&self) -> usize {
            self.vocab
        }

        fn checkpoint_bytes(&self) -> usize {
            self.tokens.len() * size_of::<u32>()
        }

        fn reset(&mut self) -> Result<()> {
            self.tokens.clear();
            Ok(())
        }

        fn evaluate(&mut self, token: u32) -> Result<Vec<f32>> {
            let mut output = Vec::new();
            self.evaluate_into(token, &mut output)?;
            Ok(output)
        }

        fn evaluate_into(&mut self, token: u32, output: &mut Vec<f32>) -> Result<()> {
            self.tokens.push(token);
            Self::fill_logits(&self.tokens, self.vocab, output)
        }

        fn evaluate_batch_into(
            &mut self,
            tokens: &[u32],
            output: &mut Vec<Vec<f32>>,
            check: &dyn Fn() -> Result<()>,
        ) -> Result<()> {
            output.truncate(tokens.len());
            while output.len() < tokens.len() {
                output.push(Vec::new());
            }
            for (index, &token) in tokens.iter().enumerate() {
                check()?;
                self.batch_row_initial_caps
                    .borrow_mut()
                    .push(output[index].capacity());
                self.evaluate_into(token, &mut output[index])?;
            }
            Ok(())
        }

        fn batched_evaluation_capacity(&self) -> usize {
            8
        }

        fn checkpoint(&self) -> Result<Self::Checkpoint> {
            let mut copy = Vec::new();
            copy.try_reserve_exact(self.tokens.len().max(16))?;
            copy.extend_from_slice(&self.tokens);
            Ok(copy)
        }

        fn checkpoint_into(&self, target: &mut Self::Checkpoint) -> Result<()> {
            self.checkpoint_into_calls
                .set(self.checkpoint_into_calls.get() + 1);
            let before_ptr = target.as_ptr();
            let before_cap = target.capacity();
            if self.tokens.len() > before_cap {
                target.try_reserve_exact(self.tokens.len() - target.len())?;
            }
            target.clear();
            target.extend_from_slice(&self.tokens);
            if before_cap > 0 && std::ptr::eq(target.as_ptr(), before_ptr) {
                self.checkpoint_reused_count
                    .set(self.checkpoint_reused_count.get() + 1);
            }
            Ok(())
        }

        fn restore(&mut self, checkpoint: &Self::Checkpoint) -> Result<()> {
            self.tokens.clear();
            self.tokens.extend_from_slice(checkpoint);
            Ok(())
        }
    }

    #[test]
    fn recycled_checkpoints_capped_at_two_and_unique_logits_reused_without_mutating_shared_arcs()
    -> Result<()> {
        let mut runtime = Runtime::new(
            RecycledDecoder::new(),
            Limits {
                context_tokens: 64,
                checkpoints: 2,
                checkpoint_bytes: 128 * 1_024,
            },
        )?;
        let cancellation = Cancellation::default();
        let options = Generation {
            max_tokens: 2,
            stop_tokens: vec![],
        };

        for seed in 0_u32..5 {
            runtime.generate(&[seed, seed + 1], &options, &cancellation, |_| Ok(()))?;
            assert!(runtime.recycled_checkpoints.len() <= MAX_RECYCLED_CHECKPOINTS);
        }
        assert!(runtime.decoder.checkpoint_into_calls.get() > 0);
        assert!(runtime.decoder.checkpoint_reused_count.get() > 0);

        let pre_cancel_tokens = runtime.tokens().to_vec();
        let cancelled = Cancellation::default();
        cancelled.cancel();
        assert!(
            runtime
                .generate(&[9, 10], &options, &cancelled, |_| Ok(()))
                .is_err()
        );
        assert_eq!(runtime.tokens(), pre_cancel_tokens);
        assert!(runtime.recycled_checkpoints.len() <= MAX_RECYCLED_CHECKPOINTS);

        runtime.reset()?;
        let spec_options = Generation {
            max_tokens: 3,
            stop_tokens: vec![],
        };
        let suffix_config = SuffixDraftConfig::default();
        runtime.generate_speculative(
            &[1, 2, 3, 4, 1, 2],
            &spec_options,
            Prefill::default(),
            suffix_config,
            &cancellation,
            |_| Ok(()),
        )?;
        let retained_shared = Arc::clone(&runtime.logits);
        let retained_bits: Vec<u32> = retained_shared.iter().map(|v| v.to_bits()).collect();
        assert_ne!(retained_bits, Vec::<u32>::new());

        runtime.generate_speculative(
            &[2, 3, 4, 5, 2, 3],
            &spec_options,
            Prefill::default(),
            suffix_config,
            &cancellation,
            |_| Ok(()),
        )?;
        let _ = runtime.verify_draft_block(&[1, 2], &cancellation)?;
        let _ = runtime.verify_draft_block(&[1, 2], &cancellation)?;
        assert!(runtime.batch_logits.len() <= MAX_RECYCLED_BATCH_LOGITS);
        let actual_retained_bits: Vec<u32> = retained_shared.iter().map(|v| v.to_bits()).collect();
        assert_eq!(actual_retained_bits, retained_bits);
        assert!(
            runtime
                .decoder
                .batch_row_initial_caps
                .borrow()
                .iter()
                .any(|&cap| cap > 0),
            "subsequent speculative verification must reuse recycled batch logit rows"
        );
        Ok(())
    }
}

#[cfg(test)]
mod boundary_and_checkpoint_tests {
    use super::{
        Cancellation, Decoder, Generation, HybridCheckpointBoundary, Limits, Result, Runtime,
        VerificationLimits,
    };
    use crate::speculative::SpeculativeStats;
    use std::cell::Cell;

    struct BoundaryDecoder {
        tokens: Vec<u32>,
        identity: &'static str,
        vocab: usize,
    }

    impl BoundaryDecoder {
        fn new() -> Self {
            Self {
                tokens: Vec::new(),
                identity: "boundary-test-decoder/v1",
                vocab: 16,
            }
        }
    }

    impl Decoder for BoundaryDecoder {
        type Checkpoint = Vec<u32>;

        fn identity(&self) -> &str {
            self.identity
        }

        fn vocab_size(&self) -> usize {
            self.vocab
        }

        fn checkpoint_bytes(&self) -> usize {
            self.tokens.len() * size_of::<u32>()
        }

        fn reset(&mut self) -> Result<()> {
            self.tokens.clear();
            Ok(())
        }

        fn evaluate(&mut self, token: u32) -> Result<Vec<f32>> {
            let mut output = Vec::new();
            self.evaluate_into(token, &mut output)?;
            Ok(output)
        }

        fn evaluate_into(&mut self, token: u32, output: &mut Vec<f32>) -> Result<()> {
            self.tokens.push(token);
            output.clear();
            output.resize(self.vocab, 0.0);
            let last = self.tokens.last().copied().unwrap_or(0);
            let next = usize::try_from((last + 1) % u32::try_from(self.vocab)?)?;
            output[next] = 10.0;
            Ok(())
        }

        fn evaluate_batch_into(
            &mut self,
            tokens: &[u32],
            output: &mut Vec<Vec<f32>>,
            check: &dyn Fn() -> Result<()>,
        ) -> Result<()> {
            output.truncate(tokens.len());
            while output.len() < tokens.len() {
                output.push(Vec::new());
            }
            for (index, &token) in tokens.iter().enumerate() {
                check()?;
                self.evaluate_into(token, &mut output[index])?;
            }
            Ok(())
        }

        fn batched_evaluation_capacity(&self) -> usize {
            8
        }

        fn checkpoint(&self) -> Result<Self::Checkpoint> {
            Ok(self.tokens.clone())
        }

        fn restore(&mut self, checkpoint: &Self::Checkpoint) -> Result<()> {
            self.tokens.clear();
            self.tokens.extend_from_slice(checkpoint);
            Ok(())
        }
    }

    #[test]
    fn transient_capture_never_evicts_reusable_prefix() -> Result<()> {
        let mut runtime = Runtime::new(
            BoundaryDecoder::new(),
            Limits {
                context_tokens: 64,
                checkpoints: 1,
                checkpoint_bytes: 128 * 1_024,
            },
        )?;
        let cancellation = Cancellation::default();
        let options = Generation {
            max_tokens: 2,
            stop_tokens: vec![],
        };
        runtime.generate(&[1, 2], &options, &cancellation, |_| Ok(()))?;
        assert_eq!(runtime.checkpoint_count(), 1);
        let saved_bytes = runtime.checkpoints[0].bytes;
        let saved_tokens = runtime.checkpoints[0].tokens.clone();

        // Batched draft verification uses a transient checkpoint that must never
        // displace or evict saved reusable prefixes.
        let (committed, stats) = runtime.verify_draft_block(&[5, 6], &cancellation)?;
        assert_eq!(committed, [5, 6, 7]);
        assert_eq!(stats.draft_tokens_accepted, 2);
        assert_eq!(runtime.checkpoint_count(), 1);
        assert_eq!(runtime.checkpoints[0].tokens, saved_tokens);

        // Tight budget leaving room for the existing reusable entry + transient
        // fails transient capture without evicting `checkpoints[0]`.
        let needed_transient = runtime.snapshot_bytes()?;
        runtime.limits.checkpoint_bytes = saved_bytes + needed_transient - 1;
        assert!(runtime.capture_transient(0, &|| Ok(())).is_err());
        assert_eq!(runtime.checkpoint_count(), 1);
        assert_eq!(runtime.checkpoints[0].tokens, saved_tokens);

        runtime.limits.checkpoint_bytes = saved_bytes + needed_transient;
        let transient = runtime.capture_transient(0, &|| Ok(()))?;
        assert_eq!(runtime.checkpoint_count(), 1);
        assert_eq!(runtime.checkpoints[0].tokens, saved_tokens);
        runtime.recycle_snapshot(transient);
        Ok(())
    }

    #[test]
    fn prompt_end_checkpoint_survives_assistant_decode() -> Result<()> {
        let mut runtime = Runtime::new(
            BoundaryDecoder::new(),
            Limits {
                context_tokens: 64,
                checkpoints: 0,
                checkpoint_bytes: 128 * 1_024,
            },
        )?;
        runtime.set_retain_prompt_end(true);
        assert!(runtime.retain_prompt_end());
        assert!(!runtime.has_prompt_end_checkpoint());

        let cancellation = Cancellation::default();
        let options = Generation {
            max_tokens: 3,
            stop_tokens: vec![],
        };
        let first = runtime.generate(&[1, 2, 3, 4], &options, &cancellation, |_| Ok(()))?;
        assert_eq!(first.reused_tokens, 0);
        assert_eq!(runtime.tokens(), [1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(runtime.checkpoint_count(), 0);
        assert!(runtime.has_prompt_end_checkpoint());

        // Failed follow-up request after assistant decode must preserve the prior
        // prompt-end checkpoint transactionally.
        let emitted = Cell::new(0_usize);
        let failed = runtime.generate(&[9, 10], &options, &cancellation, |_| {
            emitted.set(emitted.get() + 1);
            Err(crate::invalid("abort after prompt prefill"))
        });
        assert!(failed.is_err());
        assert_eq!(runtime.tokens(), [1, 2, 3, 4, 5, 6, 7]);

        // Editing/regenerating immediately after the first prompt still hits the
        // dedicated prompt-end prefix even though assistant tokens advanced past it.
        let regenerated =
            runtime.generate(&[1, 2, 3, 4, 0], &options, &cancellation, |_| Ok(()))?;
        assert_eq!(regenerated.reused_tokens, 4);
        assert_eq!(runtime.tokens(), [1, 2, 3, 4, 0, 1, 2, 3]);
        Ok(())
    }

    #[test]
    fn hybrid_checkpoint_boundary_reconciles_page_and_recurrent_limits() {
        let prompt: Vec<u32> = (0..130).collect();
        let history: Vec<u32> = (0..130).collect();
        let identity = "qwen35-hybrid-f16/v1";

        // 1. Attention hits all 130 tokens and KV aligns down to 128 (page 64),
        // while recurrent state is stuck at unaligned position 130: cannot
        // roll recurrent state backwards, so boundary clamps safely to 0.
        let clamped_unaligned = HybridCheckpointBoundary::reconcile(
            &prompt,
            &history,
            identity,
            identity,
            130,
            Some(64),
            true,
            Some(130),
        );
        assert_eq!(clamped_unaligned.matched_tokens, 130);
        assert_eq!(clamped_unaligned.kv_bound, 128);
        assert_eq!(clamped_unaligned.position, 0);
        assert!(clamped_unaligned.clamped_by_recurrent);
        assert!(clamped_unaligned.identity_matched);

        // 2. When exact unaligned KV tail is permitted (`None`), the exact hybrid
        // boundary at 130 is reusable.
        let unaligned_tail_ok = HybridCheckpointBoundary::reconcile(
            &prompt,
            &history,
            identity,
            identity,
            130,
            None,
            true,
            Some(130),
        );
        assert_eq!(unaligned_tail_ok.position, 130);
        assert_eq!(unaligned_tail_ok.kv_bound, 130);
        assert!(!unaligned_tail_ok.clamped_by_recurrent);

        // 3. Aligned recurrent checkpoint at 64 reconciles below KV page bound 128.
        let aligned_recurrent = HybridCheckpointBoundary::reconcile(
            &prompt,
            &history,
            identity,
            identity,
            130,
            Some(64),
            true,
            Some(64),
        );
        assert_eq!(aligned_recurrent.kv_bound, 128);
        assert_eq!(aligned_recurrent.position, 64);
        assert!(aligned_recurrent.clamped_by_recurrent);

        // 4. Representation/adapter identity mismatch rejects any reuse.
        let mismatched = HybridCheckpointBoundary::reconcile(
            &prompt,
            &history,
            identity,
            "qwen35-hybrid-f16/lora:0x3f800000",
            130,
            None,
            true,
            Some(130),
        );
        assert_eq!(mismatched.position, 0);
        assert!(!mismatched.identity_matched);

        // 5. Missing recurrent state availability clamps hybrid reuse even with
        // full KV page hits; pure KV (`has_recurrent_layers = false`) reuses 128.
        let no_recurrent_state = HybridCheckpointBoundary::reconcile(
            &prompt,
            &history,
            identity,
            identity,
            130,
            Some(64),
            true,
            None,
        );
        assert_eq!(no_recurrent_state.position, 0);
        assert!(no_recurrent_state.clamped_by_recurrent);

        let pure_kv = HybridCheckpointBoundary::reconcile(
            &prompt,
            &history,
            identity,
            identity,
            130,
            Some(64),
            false,
            None,
        );
        assert_eq!(pure_kv.position, 128);
        assert!(!pure_kv.clamped_by_recurrent);
    }

    #[test]
    fn runtime_hybrid_boundary_reconciles_live_checkpoint() -> Result<()> {
        let mut runtime = Runtime::new(
            BoundaryDecoder::new(),
            Limits {
                context_tokens: 256,
                checkpoints: 2,
                checkpoint_bytes: 128 * 1_024,
            },
        )?;
        let cancellation = Cancellation::default();
        let page_prompt: Vec<u32> = (0_u32..63).map(|value| value % 16).collect();
        let tail_prompt: Vec<u32> = (0_u32..70).map(|value| value % 16).collect();
        runtime.generate(
            &page_prompt,
            &Generation {
                max_tokens: 1,
                stop_tokens: vec![],
            },
            &cancellation,
            |_| Ok(()),
        )?;
        let extra = runtime.hybrid_boundary(&tail_prompt, true, Some(64));
        assert_eq!(extra.position, 64);
        Ok(())
    }

    struct BatchedGreedyControl {
        tokens: Vec<u32>,
        next_stop_on_one: bool,
    }

    impl Decoder for BatchedGreedyControl {
        type Checkpoint = Vec<u32>;

        fn identity(&self) -> &'static str {
            "bounded-verify-control"
        }
        fn vocab_size(&self) -> usize {
            3
        }
        fn checkpoint_bytes(&self) -> usize {
            128
        }
        fn reset(&mut self) -> Result<()> {
            self.tokens.clear();
            Ok(())
        }
        fn evaluate(&mut self, token: u32) -> Result<Vec<f32>> {
            self.tokens.push(token);
            if self.next_stop_on_one {
                if token == 1 {
                    // Next greedy token after 1 is stop token 2.
                    Ok(vec![0.0, 0.0, 5.0])
                } else if self.tokens.len() >= 2 {
                    // Prompt+1 generation done; next greedy token is 1.
                    Ok(vec![0.0, 5.0, 0.0])
                } else {
                    Ok(vec![5.0, 0.0, 0.0])
                }
            } else {
                Ok(vec![5.0, 0.0, 0.0])
            }
        }
        fn evaluate_batch_into(
            &mut self,
            tokens: &[u32],
            output: &mut Vec<Vec<f32>>,
            check: &dyn Fn() -> Result<()>,
        ) -> Result<()> {
            output.truncate(tokens.len());
            while output.len() < tokens.len() {
                output.push(Vec::new());
            }
            for (index, &token) in tokens.iter().enumerate() {
                check()?;
                let logits = self.evaluate(token)?;
                output[index].clear();
                output[index].extend_from_slice(&logits);
            }
            Ok(())
        }
        fn batched_evaluation_capacity(&self) -> usize {
            8
        }
        fn checkpoint(&self) -> Result<Self::Checkpoint> {
            Ok(self.tokens.clone())
        }
        fn restore(&mut self, state: &Self::Checkpoint) -> Result<()> {
            self.tokens.clone_from(state);
            Ok(())
        }
    }

    #[test]
    fn verify_draft_block_bounded_stops_cleanly_at_stop_token_and_limit() -> Result<()> {
        let limits = Limits {
            context_tokens: 32,
            checkpoints: 0,
            checkpoint_bytes: 4096,
        };
        let cancel = Cancellation::default();

        // 1. max_new_tokens == 0 returns empty without touching decoder state.
        let mut runtime = Runtime::new(
            BatchedGreedyControl {
                tokens: Vec::new(),
                next_stop_on_one: false,
            },
            limits,
        )?;
        runtime.generate(
            &[0],
            &Generation {
                max_tokens: 1,
                stop_tokens: Vec::new(),
            },
            &cancel,
            |_| Ok(()),
        )?;
        let before_zero = runtime.tokens.clone();
        let (zero_tokens, zero_stats) = runtime.verify_draft_block_bounded(
            &[0, 0],
            VerificationLimits {
                max_new_tokens: 0,
                stop_tokens: &[2],
            },
            &cancel,
        )?;
        assert_eq!(zero_tokens, Vec::<u32>::new());
        assert_eq!(zero_stats, SpeculativeStats::default());
        assert_eq!(runtime.tokens, before_zero);
        assert_eq!(runtime.decoder.tokens, before_zero);

        // 2. max_new_tokens == draft.len() caps emission without adding a surplus bonus token.
        let (capped, capped_stats) = runtime.verify_draft_block_bounded(
            &[0, 0],
            VerificationLimits {
                max_new_tokens: 2,
                stop_tokens: &[2],
            },
            &cancel,
        )?;
        assert_eq!(capped, [0, 0]);
        assert_eq!(capped_stats.draft_tokens_accepted, 2);
        assert_eq!(capped_stats.bonus_tokens, 0);
        assert_eq!(runtime.decoder.tokens, runtime.tokens);

        // 3. Stop token encountered inside a batched draft rolls back surplus draft tokens.
        let mut stop_runtime = Runtime::new(
            BatchedGreedyControl {
                tokens: Vec::new(),
                next_stop_on_one: true,
            },
            limits,
        )?;
        stop_runtime.generate(
            &[0],
            &Generation {
                max_tokens: 1,
                stop_tokens: Vec::new(),
            },
            &cancel,
            |_| Ok(()),
        )?;
        // State is [0, 0]; draft [1, 2] accepts 1, then stops at 2 without evaluating/emitting 2.
        let (stopped, stop_stats) = stop_runtime.verify_draft_block_bounded(
            &[1, 2],
            VerificationLimits {
                max_new_tokens: 4,
                stop_tokens: &[2],
            },
            &cancel,
        )?;
        assert_eq!(stopped, [1]);
        assert_eq!(stop_stats.draft_tokens_accepted, 1);
        assert_eq!(stop_stats.rollback_restores, 1);
        assert_eq!(stop_stats.correction_tokens, 0);
        assert_eq!(stop_stats.replay_steps, 1);
        assert_eq!(stop_stats.replay_tokens, 1);
        assert_eq!(stop_runtime.decoder.tokens, stop_runtime.tokens);
        assert_eq!(stop_runtime.tokens, [0, 0, 1]);
        Ok(())
    }

    #[test]
    fn reallocation_witness_stops_growing_and_counts_reuse_after_warmup() -> Result<()> {
        let mut runtime = Runtime::new(
            BoundaryDecoder::new(),
            Limits {
                context_tokens: 32,
                checkpoints: 2,
                checkpoint_bytes: 8192,
            },
        )?;
        let cancel = Cancellation::default();
        runtime.generate(
            &[1, 2],
            &Generation {
                max_tokens: 1,
                stop_tokens: Vec::new(),
            },
            &cancel,
            |_| Ok(()),
        )?;
        // Warm up checkpoint & batch logits pools on initial transient verification rounds.
        let _ = runtime.verify_draft_block(&[0, 1], &cancel)?;
        let _ = runtime.verify_draft_block(&[0, 1], &cancel)?;
        let warmup = runtime.reallocation_witness();
        assert!(warmup.growth_reallocations > 0);

        for _ in 0..3 {
            let _ = runtime.verify_draft_block(&[0, 1], &cancel)?;
        }
        let steady = runtime.reallocation_witness();
        assert_eq!(steady.growth_reallocations, warmup.growth_reallocations);
        assert_eq!(
            steady.same_size_reallocations,
            warmup.same_size_reallocations
        );
        assert!(steady.reused_preparations > warmup.reused_preparations);
        Ok(())
    }
}

#[cfg(test)]
mod multimodal_tests {
    use super::*;

    #[derive(Clone, Debug, Eq, PartialEq)]
    enum Op {
        TextIngest(Vec<u32>),
        TextEval(u32),
        ImageChunk {
            width: usize,
            height: usize,
            want_logits: bool,
        },
    }

    struct MultimodalMockDecoder {
        tokens: Vec<u32>,
        ops: Vec<Op>,
        fail_on_image_calls: Option<usize>,
        image_calls: usize,
    }

    impl MultimodalMockDecoder {
        fn new() -> Self {
            Self {
                tokens: Vec::new(),
                ops: Vec::new(),
                fail_on_image_calls: None,
                image_calls: 0,
            }
        }
    }

    #[derive(Clone)]
    struct MockCheckpoint {
        tokens: Vec<u32>,
        image_calls: usize,
    }

    impl Decoder for MultimodalMockDecoder {
        type Checkpoint = MockCheckpoint;

        fn identity(&self) -> &'static str {
            "multimodal-mock/v1"
        }

        fn vocab_size(&self) -> usize {
            16
        }

        fn checkpoint_bytes(&self) -> usize {
            self.tokens.len() * 4
        }

        fn reset(&mut self) -> Result<()> {
            self.tokens.clear();
            Ok(())
        }

        fn ingest_block(&mut self, tokens: &[u32], check: &dyn Fn() -> Result<()>) -> Result<()> {
            check()?;
            self.tokens.extend_from_slice(tokens);
            self.ops.push(Op::TextIngest(tokens.to_vec()));
            Ok(())
        }

        fn evaluate(&mut self, token: u32) -> Result<Vec<f32>> {
            self.tokens.push(token);
            self.ops.push(Op::TextEval(token));
            let mut logits = vec![-1.0; 16];
            logits[usize::try_from((token + 1) % 16)?] = 1.0;
            Ok(logits)
        }

        fn evaluate_image_chunk(
            &mut self,
            embeddings: &[f32],
            grid_width: usize,
            grid_height: usize,
            logits: Option<&mut Vec<f32>>,
            check: &dyn Fn() -> Result<()>,
        ) -> Result<()> {
            check()?;
            self.image_calls += 1;
            if self.fail_on_image_calls == Some(self.image_calls) {
                return Err(invalid("injected image-chunk failure"));
            }
            let count = grid_width * grid_height;
            assert_eq!(embeddings.len(), count * 2);
            self.tokens.extend(std::iter::repeat_n(9_u32, count));
            self.ops.push(Op::ImageChunk {
                width: grid_width,
                height: grid_height,
                want_logits: logits.is_some(),
            });
            if let Some(output) = logits {
                output.clear();
                output.resize(16, -1.0);
                output[3] = 2.0;
            }
            Ok(())
        }

        fn checkpoint(&self) -> Result<Self::Checkpoint> {
            Ok(MockCheckpoint {
                tokens: self.tokens.clone(),
                image_calls: self.image_calls,
            })
        }

        fn restore(&mut self, checkpoint: &Self::Checkpoint) -> Result<()> {
            self.tokens.clone_from(&checkpoint.tokens);
            self.image_calls = checkpoint.image_calls;
            Ok(())
        }
    }

    fn make_chunk(offset: usize, w: usize, h: usize, hash_byte: u8) -> (MediaSpan, Vec<f32>) {
        let len = w * h;
        (
            MediaSpan {
                offset,
                len,
                grid_width: w,
                grid_height: h,
                content_sha256: [hash_byte; 32],
            },
            vec![0.25_f32; len * 2],
        )
    }

    #[test]
    fn multimodal_prefill_and_prefix_reuse_track_media_spans() -> Result<()> {
        let limits = Limits {
            context_tokens: 64,
            checkpoints: 2,
            checkpoint_bytes: 32_768,
        };
        let mut runtime = Runtime::new(MultimodalMockDecoder::new(), limits)?;
        let cancel = Cancellation::default();
        let options = Generation {
            max_tokens: 1,
            stop_tokens: Vec::new(),
        };
        let mut sampler = Greedy;

        let (span1, emb1) = make_chunk(2, 2, 2, 0xAA);
        let prompt1 = [1, 2, 9, 9, 9, 9, 4];
        let mut out = Vec::new();
        let first = runtime.generate_multimodal_sampled_until(
            &prompt1,
            &[MultimodalChunk {
                span: span1,
                embeddings: &emb1,
            }],
            &options,
            Prefill { block_tokens: 4 },
            &cancel,
            None,
            &mut sampler,
            |tok| {
                out.push(tok);
                Ok(())
            },
        )?;
        assert_eq!(out, [5]);
        assert_eq!(first.reused_tokens, 0);
        assert_eq!(runtime.media_spans(), [span1]);
        assert_eq!(
            runtime.decoder().ops,
            vec![
                Op::TextIngest(vec![1, 2]),
                Op::ImageChunk {
                    width: 2,
                    height: 2,
                    want_logits: false
                },
                Op::TextEval(4),
                Op::TextEval(5),
            ]
        );

        let mut prompt2 = prompt1.to_vec();
        prompt2.extend_from_slice(&[5, 6]);
        let before_ops = runtime.decoder().ops.len();
        let second = runtime.generate_multimodal_sampled_until(
            &prompt2,
            &[MultimodalChunk {
                span: span1,
                embeddings: &emb1,
            }],
            &options,
            Prefill::default(),
            &cancel,
            None,
            &mut sampler,
            |_| Ok(()),
        )?;
        assert_eq!(second.reused_tokens, 8);
        assert_eq!(
            &runtime.decoder().ops[before_ops..],
            &[Op::TextEval(6), Op::TextEval(7)]
        );

        let (span_changed, emb_changed) = make_chunk(2, 2, 2, 0xBB);
        let third = runtime.generate_multimodal_sampled_until(
            &prompt2,
            &[MultimodalChunk {
                span: span_changed,
                embeddings: &emb_changed,
            }],
            &options,
            Prefill::default(),
            &cancel,
            None,
            &mut sampler,
            |_| Ok(()),
        )?;
        assert_eq!(third.reused_tokens, 0);
        assert_eq!(runtime.media_spans(), [span_changed]);

        let text_only = runtime.generate_sampled_until(
            &prompt2,
            &options,
            Prefill::default(),
            &cancel,
            None,
            &mut sampler,
            |_| Ok(()),
        )?;
        assert_eq!(text_only.reused_tokens, 0);
        assert_eq!(runtime.media_spans(), []);
        Ok(())
    }

    #[test]
    fn multimodal_rollback_and_reset_only_preserve_invariants() -> Result<()> {
        let mut runtime = Runtime::new(
            MultimodalMockDecoder::new(),
            Limits {
                context_tokens: 64,
                checkpoints: 0,
                checkpoint_bytes: 32_768,
            },
        )?;
        let cancel = Cancellation::default();
        let options = Generation {
            max_tokens: 1,
            stop_tokens: Vec::new(),
        };
        let mut sampler = Greedy;

        let (span_end, emb_end) = make_chunk(1, 2, 1, 0xCC);
        runtime.decoder.fail_on_image_calls = Some(1);
        let err = runtime.generate_multimodal_sampled_until_for_reset(
            &[1, 9, 9],
            &[MultimodalChunk {
                span: span_end,
                embeddings: &emb_end,
            }],
            &options,
            Prefill::default(),
            &cancel,
            None,
            &mut sampler,
            None,
            |_| Ok(()),
        );
        assert!(err.is_err());
        assert_eq!(runtime.media_spans(), []);

        runtime.decoder.fail_on_image_calls = None;
        let mut terminal_out = Vec::new();
        runtime.generate_multimodal_sampled_until_for_reset(
            &[1, 9, 9],
            &[MultimodalChunk {
                span: span_end,
                embeddings: &emb_end,
            }],
            &options,
            Prefill::default(),
            &cancel,
            None,
            &mut sampler,
            None,
            |tok| {
                terminal_out.push(tok);
                Ok(())
            },
        )?;
        assert_eq!(terminal_out, [3]);
        assert_eq!(runtime.media_spans(), [span_end]);
        Ok(())
    }
}
