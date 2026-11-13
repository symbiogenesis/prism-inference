//! One pinned Swift conversation joining exact chat input and transactional text output.
//!
//! Readiness verifies identity and executable graph construction. Full-model
//! correctness and device qualification require separately retained comparisons.

use std::{path::Path, sync::Arc, time::Instant};

use serde_json::Value;

use crate::{
    Decoder, Result,
    graph::{KvCachePolicy, Qwen35},
    invalid,
    kernels::{MatvecBackend, RowParallelism},
    model::{
        lora::RankOneAdapter,
        swift::{self, LoadReceipt, Representation},
    },
    runtime::{Cancellation, Completion, GenerationTimings, Limits, Prefill, PromptReuse, Runtime},
    sampling::SwiftSampler,
    speculative::{SpeculativeStats, SuffixDraftConfig},
    swift_chat::SwiftChatSession,
    text_output::Utf8Output,
    vocabulary::Vocabulary,
};

/// Maximum retained model storage admitted by this Swift session.
pub const MODEL_MEMORY_BYTES: usize = 8 * 1024 * 1024 * 1024;
/// Development-default Swift attention cache; explicit FP16 remains available.
///
/// This approximate K4/V4 policy does not establish full-model quality or device
/// qualification. Generic graph constructors retain the FP16 reference policy.
pub const DEFAULT_KV_CACHE: KvCachePolicy = KvCachePolicy::Rotated4;
/// Maximum live selected-profile KV and F32 recurrent/convolution payload.
pub const STATE_MEMORY_BYTES: usize = 1024 * 1024 * 1024;
/// Maximum complete transactional rollback snapshot storage.
pub const CHECKPOINT_MEMORY_BYTES: usize = 1024 * 1024 * 1024;
/// Conservative fixed model/runtime allowance, excluding context growth.
pub const NON_CONTEXT_MEMORY_BYTES: u64 = 11 * 1024 * 1024 * 1024;
/// Live and rollback FP16 KV, bounded attention scores and context bookkeeping
/// charged for each selected context token.
///
/// Quantized caches retain this conservative FP16 capacity charge. Bookkeeping
/// includes token histories and checkpoint/page metadata. Fixed workspace and
/// recurrent state use the non-context allowance. The shared launcher charge
/// reserves 64-token, 24-head F32 scores for publisher microbatch attention.
/// Native ordinary block prefill is sequential and retains 24 score rows;
/// optional tiled-prefill APIs still need their separate reservation review.
/// This source estimate is not a measured peak or a memory qualification.
pub const CONTEXT_TOKEN_MEMORY_BYTES: u64 = 2 * 65_536 + 64 * 24 * 4 + 128;
/// Maximum context fitting the existing graph state bound at FP16 precision.
///
/// Preserve the graph's 4 GiB cap and the pinned 156,893,184-byte recurrent
/// payload. The extra scratch charge and 256-token rounding leave metadata room.
pub const MAX_CONTEXT_TOKENS: usize =
    (crate::graph::MAX_STATE_BYTES - 156_893_184) / (65_536 + 96 + 128) / 256 * 256;

/// Validate a launcher's selected context and conservative process allowance.
///
/// This never queries memory or creates a Job. The owner enforces that process
/// allowance and independently retains a 4 GiB physical host reserve.
///
/// # Errors
/// Rejects a context outside the model/state bounds or an insufficient allowance.
pub fn validate_context_capacity(context_tokens: usize, memory_limit_bytes: u64) -> Result<()> {
    let required = u64::try_from(context_tokens)?
        .checked_mul(CONTEXT_TOKEN_MEMORY_BYTES)
        .and_then(|bytes| bytes.checked_add(NON_CONTEXT_MEMORY_BYTES))
        .ok_or_else(|| invalid("Swift context memory allowance overflow"))?;
    if context_tokens == 0
        || context_tokens > MAX_CONTEXT_TOKENS
        || context_tokens > swift::TRAINING_CONTEXT_TOKENS
        || memory_limit_bytes < required
    {
        return Err(invalid("Swift context memory allowance rejected"));
    }
    Ok(())
}

/// Explicit exact CPU prompt tiles and their separately charged scratch reserve.
///
/// Only FP16 KV is supported. This selects the existing sequential-GDN tile
/// arithmetic, not the separately identified triangular GDN approximation.
/// Owners must add the full reserve to their process allowance before loading.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CpuPrefill {
    tokens: usize,
    scratch_bytes: usize,
}

impl CpuPrefill {
    /// Construct an explicit tile selection without loading any model bytes.
    ///
    /// The graph independently checks its geometry-derived minimum reservation
    /// during loading. A larger caller reserve retains its full admission charge.
    ///
    /// # Errors
    /// Rejects tile sizes other than 8, 16 or 32 and an empty scratch reserve.
    pub fn new(tokens: usize, scratch_bytes: usize) -> Result<Self> {
        if !matches!(tokens, 8 | 16 | 32) || scratch_bytes == 0 {
            return Err(invalid(
                "invalid CPU tiled prefill selection or scratch reserve",
            ));
        }
        Ok(Self {
            tokens,
            scratch_bytes,
        })
    }

    /// Maximum prompt tokens sharing one layer-major projection tile.
    #[must_use]
    pub const fn tokens(self) -> usize {
        self.tokens
    }

    /// Additional process-memory allowance reserved by the owning launcher.
    #[must_use]
    pub const fn scratch_bytes(self) -> usize {
        self.scratch_bytes
    }
}

/// Validate context plus the entire explicitly selected CPU tile reservation.
///
/// This performs no artifact I/O or allocation. The process owner separately
/// enforces the supplied allowance and retains the existing physical host reserve.
///
/// # Errors
/// Rejects invalid context capacity, insufficient memory and reservation overflow.
pub fn validate_prefill_capacity(
    context_tokens: usize,
    memory_limit_bytes: u64,
    prefill: Option<CpuPrefill>,
) -> Result<()> {
    let reserve = prefill.map_or(0, CpuPrefill::scratch_bytes);
    let available = memory_limit_bytes
        .checked_sub(u64::try_from(reserve)?)
        .ok_or_else(|| invalid("Swift CPU prefill memory allowance rejected"))?;
    validate_context_capacity(context_tokens, available)
}

/// Largest complete displayed output retained or delivered by one request.
pub const MAX_OUTPUT_BYTES: usize = 4 * 1024 * 1024;

/// Exact rendered and counted input, sealed by its originating verified session.
pub struct PreparedChat {
    owner: Arc<()>,
    adapter_scale: Option<f32>,
    tokens: Vec<u32>,
    output_tokens: usize,
    enable_thinking: bool,
}

impl PreparedChat {
    /// Exact correction scale reserved by this request, when an adapter is loaded.
    #[must_use]
    pub const fn adapter_scale(&self) -> Option<f32> {
        self.adapter_scale
    }

    /// Exact number of tokens in the complete rendered prompt.
    #[must_use]
    pub const fn prompt_tokens(&self) -> usize {
        self.tokens.len()
    }

    /// Complete output allowance reserved before model execution.
    #[must_use]
    pub const fn output_tokens(&self) -> usize {
        self.output_tokens
    }

    /// Whether the exact prepared prompt starts an open reasoning block.
    #[must_use]
    pub const fn thinking_enabled(&self) -> bool {
        self.enable_thinking
    }
}

/// A provisional text event; callers commit displayed output only on completion.
#[derive(Debug)]
pub enum TextEvent<'a> {
    /// Incrementally decoded UTF-8, including a possible final replacement suffix.
    Delta(&'a str),
    /// Terminal accounting. Failure to deliver this event closes admission.
    Complete(Completion),
}

/// Single-owner verified Swift runtime, tokenizer, output vocabulary and sampler.
///
/// Mutable access serializes generation and reset. Requests start model state
/// from empty by default; exact latest-state reuse requires explicit selection.
/// Sampler draws persist until reset;
/// cancelled or failed generation restores the pre-request sampler. A failed
/// terminal delivery closes admission until an explicit successful reset.
pub struct SwiftSession {
    owner: Arc<()>,
    representation: Representation,
    kv_cache: KvCachePolicy,
    cpu_prefill: Option<CpuPrefill>,
    adapter_identity: Option<String>,
    runtime: Runtime<Qwen35>,
    chat: SwiftChatSession,
    vocabulary: Vocabulary,
    sampler: SwiftSampler,
    seed: u32,
    delivery_fault: bool,
    prompt_reuse: PromptReuse,
    receipt: LoadReceipt,
}

impl SwiftSession {
    /// Load the pinned immutable artifact and establish bounded executable state.
    ///
    /// Attention uses the development-default K4/V4 cache. Use
    /// [`Self::load_profile_with_kv_cache`] for explicit FP16 comparison/recovery.
    ///
    /// The separate storage/state/checkpoint caps exclude allocator overhead,
    /// bounded graph scratch and temporary loader copies. They do not replace
    /// the existing process-memory admission and watchdog policies.
    ///
    /// # Errors
    /// Rejects artifact/metadata/inventory drift, graph errors or allocation bounds.
    pub fn load(path: &Path, backend: MatvecBackend, seed: u32) -> Result<Self> {
        Self::load_with_row_parallelism(path, backend, seed, RowParallelism::default())
    }

    /// Loads the pinned session with an explicitly constructed row-worker pool.
    ///
    /// Default loading remains serial. The pool is shared by packed projections
    /// and survives reset; existing transaction/cancellation behavior is retained.
    ///
    /// # Errors
    /// Returns the same loading, identity and state errors as [`Self::load`].
    pub fn load_with_row_parallelism(
        path: &Path,
        backend: MatvecBackend,
        seed: u32,
        parallelism: RowParallelism,
    ) -> Result<Self> {
        Self::load_profile_with_row_parallelism(
            path,
            Representation::Pq2,
            None,
            backend,
            seed,
            parallelism,
        )
    }

    /// Load one exact Swift representation and an optional approved rank-one adapter.
    ///
    /// Selection is explicit and immutable for this session. Only the approved
    /// adapter artifact is admitted. This executable path does not qualify model
    /// arithmetic, authoring basis, device resources, quality or release readiness.
    /// The attention cache uses [`DEFAULT_KV_CACHE`].
    ///
    /// # Errors
    /// Rejects artifact, representation, adapter, graph and storage-budget drift.
    pub fn load_profile_with_row_parallelism(
        path: &Path,
        representation: Representation,
        adapter_path: Option<&Path>,
        backend: MatvecBackend,
        seed: u32,
        parallelism: RowParallelism,
    ) -> Result<Self> {
        Self::load_profile_with_kv_cache(
            path,
            representation,
            adapter_path,
            backend,
            seed,
            parallelism,
            DEFAULT_KV_CACHE,
        )
    }

    /// Load an explicit, immutable KV precision profile with optional approved `LoRA`.
    ///
    /// Rotated profiles change numerical cache semantics and remain experimental.
    /// Context, output, model, checkpoint and process-admission limits are unchanged.
    /// Ordinary loading APIs use K4/V4. Pass [`KvCachePolicy::F16`] explicitly for
    /// comparison/recovery; selection is never a request-body control or fallback.
    ///
    /// # Errors
    /// Rejects unsupported cache geometry and the same loading errors as
    /// [`Self::load_profile_with_row_parallelism`].
    pub fn load_profile_with_kv_cache(
        path: &Path,
        representation: Representation,
        adapter_path: Option<&Path>,
        backend: MatvecBackend,
        seed: u32,
        parallelism: RowParallelism,
        kv_cache: KvCachePolicy,
    ) -> Result<Self> {
        Self::load_profile_with_context(
            path,
            representation,
            adapter_path,
            backend,
            seed,
            parallelism,
            kv_cache,
            swift::CONTEXT_TOKENS,
        )
    }

    /// Load one immutable context shared by tokenizer, decoder and runtime.
    ///
    /// Fixed callers retain the 8,192-token default through existing loading
    /// methods. A launcher may select a larger or smaller context from current
    /// memory and must enforce its separate process and host-reserve allowance.
    /// The graph's existing 4 GiB live-state ceiling is never increased.
    ///
    /// # Errors
    /// Rejects a context beyond the conservative FP16 state capacity and the
    /// artifact, adapter, cache and allocation errors of other loading methods.
    #[expect(
        clippy::too_many_arguments,
        reason = "The explicit context extends the established seven-argument immutable profile loader."
    )]
    pub fn load_profile_with_context(
        path: &Path,
        representation: Representation,
        adapter_path: Option<&Path>,
        backend: MatvecBackend,
        seed: u32,
        parallelism: RowParallelism,
        kv_cache: KvCachePolicy,
        context_tokens: usize,
    ) -> Result<Self> {
        Self::load_profile_with_prefill(
            path,
            representation,
            adapter_path,
            backend,
            seed,
            parallelism,
            kv_cache,
            context_tokens,
            None,
        )
    }

    /// Load an explicitly selected exact CPU tiled-prefill profile.
    ///
    /// Existing loaders keep ordinary prompt ingestion. An explicit tile requires
    /// FP16 KV; incompatible precision is rejected before artifact I/O. The model
    /// owner must charge [`CpuPrefill::scratch_bytes`] in addition to ordinary
    /// context/process admission, using [`validate_prefill_capacity`]. Graph
    /// construction independently rejects an insufficient geometry-specific reserve.
    /// Reset, adapter changes and exact latest-prefix reuse retain this selection.
    /// No sampling, decode, triangular-GDN or automatic accelerator policy changes.
    ///
    /// # Errors
    /// Rejects incompatible precision or scratch and the errors of
    /// [`Self::load_profile_with_context`].
    #[expect(
        clippy::too_many_arguments,
        reason = "The explicit tile policy extends the compatible immutable profile loader."
    )]
    pub fn load_profile_with_prefill(
        path: &Path,
        representation: Representation,
        adapter_path: Option<&Path>,
        backend: MatvecBackend,
        seed: u32,
        parallelism: RowParallelism,
        kv_cache: KvCachePolicy,
        context_tokens: usize,
        cpu_prefill: Option<CpuPrefill>,
    ) -> Result<Self> {
        validate_context_capacity(context_tokens, u64::MAX)?;
        if cpu_prefill.is_some() && kv_cache != KvCachePolicy::F16 {
            return Err(invalid("CPU tiled prefill requires explicit FP16 KV"));
        }
        let state_budget = if context_tokens == swift::CONTEXT_TOKENS {
            STATE_MEMORY_BYTES
        } else {
            crate::graph::MAX_STATE_BYTES
        };
        let checkpoint_budget = CHECKPOINT_MEMORY_BYTES.max(state_budget);
        let sampler = SwiftSampler::new(seed)?;
        let (model, receipt) =
            swift::load_representation(path, MODEL_MEMORY_BYTES, representation)?;
        let chat = SwiftChatSession::from_model_representation(&model, representation)?
            .with_context_tokens(context_tokens)?;
        let vocabulary = Vocabulary::from_metadata(&model.metadata, swift::VOCAB_SIZE)?;
        let adapter = adapter_path
            .map(|path| RankOneAdapter::load_swift_representation(path, &model, representation))
            .transpose()?;
        let adapter_identity = adapter
            .as_ref()
            .map(|_| crate::model::lora::SWIFT_ADAPTER_SHA256.to_owned());
        let mut decoder = if let Some(adapter) = adapter {
            Qwen35::with_rank_one_adapter_and_kv(
                model,
                context_tokens,
                state_budget,
                backend,
                adapter,
                1.3,
                kv_cache,
            )?
        } else {
            Qwen35::with_state_budget_and_kv(
                model,
                context_tokens,
                state_budget,
                backend,
                kv_cache,
            )?
        };
        decoder.set_row_parallelism(parallelism);
        if let Some(prefill) = cpu_prefill {
            decoder = decoder.with_tiled_prefill(prefill.tokens, prefill.scratch_bytes)?;
        }
        let runtime = Runtime::new(
            decoder,
            Limits {
                context_tokens,
                checkpoints: 0,
                checkpoint_bytes: checkpoint_budget,
            },
        )?;
        Ok(Self {
            owner: Arc::new(()),
            representation,
            kv_cache,
            cpu_prefill,
            adapter_identity,
            runtime,
            chat,
            vocabulary,
            sampler,
            seed,
            delivery_fault: false,
            prompt_reuse: PromptReuse::Disabled,
            receipt,
        })
    }

    /// Immutable capacity shared by exact input admission and executable state.
    #[must_use]
    pub const fn context_tokens(&self) -> usize {
        self.chat.context_tokens()
    }

    /// Exact immutable model representation selected for this session.
    #[must_use]
    pub const fn representation(&self) -> Representation {
        self.representation
    }

    /// Exact immutable KV-cache precision profile selected at session construction.
    #[must_use]
    pub const fn kv_cache_policy(&self) -> KvCachePolicy {
        self.kv_cache
    }

    /// Immutable exact CPU prompt-tile policy, absent for ordinary ingestion.
    #[must_use]
    pub const fn cpu_prefill(&self) -> Option<CpuPrefill> {
        self.cpu_prefill
    }

    /// Observe live state capacities and payload without copying cache contents.
    ///
    /// This is not complete process memory or a peak observation. See
    /// [`crate::graph::StateMemory`] for the storage scope and exclusions.
    ///
    /// # Errors
    /// Returns an error if checked accounting arithmetic exceeds address space.
    pub fn state_memory(&self) -> Result<crate::graph::StateMemory> {
        self.runtime.decoder().state_memory()
    }

    /// Current complete graph identity, including an explicitly changed adapter scale.
    #[must_use]
    pub fn decoder_identity(&self) -> &str {
        self.runtime.decoder().identity()
    }

    /// Exact approved adapter artifact SHA-256, when one was explicitly loaded.
    #[must_use]
    pub fn adapter_identity(&self) -> Option<&str> {
        self.adapter_identity.as_deref()
    }

    /// Verified loading and bounded storage accounting for this model.
    #[must_use]
    pub const fn receipt(&self) -> &LoadReceipt {
        &self.receipt
    }

    /// Explicit reproducible seed restored by a successful reset.
    #[must_use]
    pub const fn seed(&self) -> u32 {
        self.seed
    }

    /// Requested CPU dispatch policy; individual projections may fall back.
    #[must_use]
    pub const fn requested_backend(&self) -> MatvecBackend {
        self.runtime.decoder().requested_backend()
    }

    /// Effective persistent projection-worker budget installed in this session.
    ///
    /// Small or floating projections can still execute serially. This identity
    /// does not count executed parallel operations or establish device benefit.
    #[must_use]
    pub const fn row_workers(&self) -> usize {
        self.runtime.decoder().row_workers()
    }

    /// Cloned row-parallelism handle for external projection executors.
    #[must_use]
    pub fn clone_row_parallelism(&self) -> RowParallelism {
        self.runtime.decoder().clone_row_parallelism()
    }

    /// Look up a named projection tensor in the underlying decoder graph.
    #[must_use]
    pub fn projection_tensor(&self, name: &str) -> Option<&crate::model::Tensor> {
        self.runtime.decoder().projection_tensor(name)
    }

    /// Install or clear an optional persistent PQ2 projection executor.
    pub fn set_projection_executor(
        &mut self,
        executor: Option<Box<dyn crate::kernels::Pq2ProjectionExecutor + Send>>,
    ) {
        self.runtime.set_projection_executor(executor);
    }

    /// Explicit request-boundary policy; fresh recomputation remains the default.
    #[must_use]
    pub const fn prompt_reuse(&self) -> PromptReuse {
        self.prompt_reuse
    }

    /// Select exact latest-state reuse, clearing live state when policy changes.
    ///
    /// No extra prefix snapshots are retained. This experimental policy requires
    /// independent full-model, lifecycle and device qualification. Changing it
    /// preserves sampler progression and the per-session prepared-prompt seal.
    ///
    /// # Errors
    /// Closed admission requires explicit reset; reset failure keeps it closed.
    pub fn set_prompt_reuse(&mut self, reuse: PromptReuse) -> Result<()> {
        if self.is_faulted() {
            return Err(invalid("Swift session requires reset"));
        }
        if self.prompt_reuse != reuse {
            self.runtime.reset()?;
            self.prompt_reuse = reuse;
        }
        Ok(())
    }

    /// Whether reset is required before another completion can start.
    #[must_use]
    pub const fn is_faulted(&self) -> bool {
        self.delivery_fault || self.runtime.is_faulted()
    }

    /// Render and encode once, reserving the entire output allowance.
    ///
    /// # Errors
    /// Rejects malformed history, output/context overflow and closed admission.
    pub fn prepare(&self, messages: &Value, output_tokens: usize) -> Result<PreparedChat> {
        self.prepare_with_thinking(messages, output_tokens, true)
    }

    /// Render and encode once with the selected thinking mode.
    ///
    /// The prepared prompt retains its mode for matching response parsing.
    ///
    /// # Errors
    /// Returns the same admission and prompt errors as [`Self::prepare`].
    pub fn prepare_with_thinking(
        &self,
        messages: &Value,
        output_tokens: usize,
        enable_thinking: bool,
    ) -> Result<PreparedChat> {
        self.prepare_with_settings(
            messages,
            output_tokens,
            enable_thinking,
            self.adapter_identity.as_ref().map(|_| 1.3),
        )
    }

    /// Seal exact prompt tokens, thinking mode and an explicit adapter scale.
    ///
    /// A loaded adapter requires `Some` finite scale in 0..2; an adapter-free
    /// session requires `None`. The prompt belongs only to this session, across
    /// explicit resets; it cannot be completed by another session or profile.
    ///
    /// # Errors
    /// Rejects closed admission, mismatched settings and all normal prompt bounds.
    pub fn prepare_with_settings(
        &self,
        messages: &Value,
        output_tokens: usize,
        enable_thinking: bool,
        adapter_scale: Option<f32>,
    ) -> Result<PreparedChat> {
        validate_adapter_scale(self.adapter_identity.is_some(), adapter_scale)?;
        if self.is_faulted() {
            return Err(invalid("Swift session requires reset"));
        }
        let prompt = self
            .chat
            .prepare_with_thinking(messages, output_tokens, enable_thinking)?;
        Ok(PreparedChat {
            owner: self.owner.clone(),
            adapter_scale,
            tokens: prompt.tokenization.token_ids,
            output_tokens,
            enable_thinking,
        })
    }

    /// Clear model and display state and restore the initial seeded sampler.
    ///
    /// # Errors
    /// Failed reset keeps admission closed until a subsequent reset succeeds.
    pub fn reset(&mut self) -> Result<()> {
        self.delivery_fault = true;
        self.runtime.reset()?;
        self.sampler = SwiftSampler::new(self.seed)?;
        self.delivery_fault = false;
        Ok(())
    }

    /// Generate sampled text under the complete output cap and absolute deadline.
    ///
    /// Callbacks provide backpressure and must report failed delivery. Deadlines
    /// and cancellation are cooperative between synchronous decoder operations.
    /// Provisional deltas are abandoned when the method fails. Failure during
    /// terminal UTF-8/accounting delivery latches admission closed until reset.
    ///
    /// # Errors
    /// Rejects closed admission, cancellation, expiry, output limits and I/O or
    /// model failures. Generation failures transactionally restore model/sampler.
    pub fn complete(
        &mut self,
        prompt: &PreparedChat,
        cancellation: &Cancellation,
        deadline: Instant,
        emit: impl FnMut(TextEvent<'_>) -> Result<()>,
    ) -> Result<Completion> {
        self.complete_observed(prompt, cancellation, deadline, None, emit)
    }

    /// Complete with optional fixed-size runtime attribution and unchanged sampling.
    ///
    /// The record is cleared before session admission. `entered` remains false
    /// if generation is rejected before reaching the runtime. Runtime success
    /// does not establish later terminal text delivery: callers must retain the
    /// result of this session call separately. Runtime callback time includes
    /// vocabulary/UTF-8 processing and the supplied provisional text consumer;
    /// rendering, profile reset and final text/accounting delivery are excluded.
    ///
    /// # Errors
    /// Returns the errors of [`Self::complete`] with the same recovery semantics.
    pub fn complete_with_timings(
        &mut self,
        prompt: &PreparedChat,
        cancellation: &Cancellation,
        deadline: Instant,
        timings: &mut GenerationTimings,
        emit: impl FnMut(TextEvent<'_>) -> Result<()>,
    ) -> Result<Completion> {
        *timings = GenerationTimings::default();
        self.complete_observed(prompt, cancellation, deadline, Some(timings), emit)
    }

    /// Generate greedy text with prompt/history suffix speculation (`SuffixDecoding`)
    /// under the complete output cap and absolute deadline.
    ///
    /// # Errors
    /// Rejects invalid suffix draft configs or any [`Self::complete`] error.
    pub fn complete_speculative(
        &mut self,
        prompt: &PreparedChat,
        suffix_config: SuffixDraftConfig,
        cancellation: &Cancellation,
        deadline: Instant,
        mut emit: impl FnMut(TextEvent<'_>) -> Result<()>,
    ) -> Result<(Completion, SpeculativeStats)> {
        suffix_config.validate()?;
        if self.is_faulted() {
            return Err(invalid("Swift session requires reset"));
        }
        validate_prompt_owner(&self.owner, prompt)?;
        validate_adapter_scale(self.adapter_identity.is_some(), prompt.adapter_scale)?;
        check_request(cancellation, deadline)?;
        self.runtime
            .prepare_profile_request(prompt.adapter_scale, self.prompt_reuse)?;
        let options = self.chat.generation(prompt.output_tokens)?;
        let mut display = Utf8Output::new();
        let mut output_bytes = 0usize;
        let consume = |token| {
            crate::text_output::push_with(&mut display, self.vocabulary.piece(token)?, |piece| {
                let text = piece.as_ref();
                charge_output(&mut output_bytes, text.len())?;
                if !text.is_empty() {
                    emit(TextEvent::Delta(text))?;
                }
                Ok(())
            })
        };
        let (completion, stats) = self.runtime.generate_speculative_until(
            &prompt.tokens,
            &options,
            Prefill { block_tokens: 64 },
            suffix_config,
            cancellation,
            deadline,
            consume,
        )?;
        self.delivery_fault = true;
        let suffix = display.finish();
        charge_output(&mut output_bytes, suffix.len())?;
        if !suffix.is_empty() {
            emit(TextEvent::Delta(suffix))?;
        }
        check_request(cancellation, deadline)?;
        emit(TextEvent::Complete(completion))?;
        check_request(cancellation, deadline)?;
        self.delivery_fault = false;
        Ok((completion, stats))
    }

    fn complete_observed(
        &mut self,
        prompt: &PreparedChat,
        cancellation: &Cancellation,
        deadline: Instant,
        timings: Option<&mut GenerationTimings>,
        mut emit: impl FnMut(TextEvent<'_>) -> Result<()>,
    ) -> Result<Completion> {
        if self.is_faulted() {
            return Err(invalid("Swift session requires reset"));
        }
        validate_prompt_owner(&self.owner, prompt)?;
        validate_adapter_scale(self.adapter_identity.is_some(), prompt.adapter_scale)?;
        check_request(cancellation, deadline)?;
        // Exact reuse is opt-in and never crosses a bitwise scale change.
        // The runtime recomputes unmatched histories under whole-request rollback.
        self.runtime
            .prepare_profile_request(prompt.adapter_scale, self.prompt_reuse)?;
        let options = self.chat.generation(prompt.output_tokens)?;
        let mut display = Utf8Output::new();
        let mut output_bytes = 0usize;
        let consume = |token| {
            crate::text_output::push_with(&mut display, self.vocabulary.piece(token)?, |piece| {
                let text = piece.as_ref();
                charge_output(&mut output_bytes, text.len())?;
                if !text.is_empty() {
                    emit(TextEvent::Delta(text))?;
                }
                Ok(())
            })
        };
        let completion = if self.prompt_reuse == PromptReuse::Disabled {
            self.runtime.generate_sampled_until_for_reset(
                &prompt.tokens,
                &options,
                Prefill { block_tokens: 64 },
                cancellation,
                deadline,
                &mut self.sampler,
                timings,
                consume,
            )?
        } else if let Some(timings) = timings {
            self.runtime.generate_sampled_until_with_timings(
                &prompt.tokens,
                &options,
                Prefill { block_tokens: 64 },
                cancellation,
                deadline,
                &mut self.sampler,
                timings,
                consume,
            )?
        } else {
            self.runtime.generate_sampled_until(
                &prompt.tokens,
                &options,
                Prefill { block_tokens: 64 },
                cancellation,
                deadline,
                &mut self.sampler,
                consume,
            )?
        };
        self.delivery_fault = true;
        let suffix = display.finish();
        charge_output(&mut output_bytes, suffix.len())?;
        if !suffix.is_empty() {
            emit(TextEvent::Delta(suffix))?;
        }
        check_request(cancellation, deadline)?;
        emit(TextEvent::Complete(completion))?;
        check_request(cancellation, deadline)?;
        self.delivery_fault = false;
        Ok(completion)
    }
}

fn validate_prompt_owner(owner: &Arc<()>, prompt: &PreparedChat) -> Result<()> {
    if Arc::ptr_eq(owner, &prompt.owner) {
        Ok(())
    } else {
        Err(invalid("prepared Swift prompt belongs to another session"))
    }
}

fn validate_adapter_scale(loaded: bool, scale: Option<f32>) -> Result<()> {
    match (loaded, scale) {
        (false, None) => Ok(()),
        (true, Some(scale)) => RankOneAdapter::validate_scale(scale),
        _ => Err(invalid(
            "Swift adapter scale does not match the loaded profile",
        )),
    }
}

fn charge_output(total: &mut usize, bytes: usize) -> Result<()> {
    *total = total
        .checked_add(bytes)
        .filter(|&n| n <= MAX_OUTPUT_BYTES)
        .ok_or_else(|| invalid("Swift output exceeds bounded byte allowance"))?;
    Ok(())
}

fn check_request(cancellation: &Cancellation, deadline: Instant) -> Result<()> {
    if cancellation.is_cancelled() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Interrupted,
            "Swift request cancelled",
        )
        .into());
    }
    if Instant::now() >= deadline {
        return Err(
            std::io::Error::new(std::io::ErrorKind::TimedOut, "Swift request expired").into(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        CONTEXT_TOKEN_MEMORY_BYTES, CpuPrefill, MAX_CONTEXT_TOKENS, NON_CONTEXT_MEMORY_BYTES,
        PreparedChat, SwiftSession, validate_adapter_scale, validate_context_capacity,
        validate_prefill_capacity, validate_prompt_owner,
    };
    use crate::{
        Result,
        graph::KvCachePolicy,
        kernels::{MatvecBackend, RowParallelism},
        model::swift::Representation,
    };
    use std::{path::Path, sync::Arc};

    #[test]
    fn cpu_prefill_requires_fp16_before_artifact_io_and_charges_full_reserve() -> Result<()> {
        let reserve = 64 * 1024 * 1024;
        let context = 8_192;
        let base = NON_CONTEXT_MEMORY_BYTES + u64::try_from(context)? * CONTEXT_TOKEN_MEMORY_BYTES;
        validate_prefill_capacity(context, base, None)?;
        for tokens in [8, 16, 32] {
            let selected = CpuPrefill::new(tokens, reserve)?;
            assert_eq!(selected.tokens(), tokens);
            assert_eq!(selected.scratch_bytes(), reserve);
            validate_prefill_capacity(context, base + u64::try_from(reserve)?, Some(selected))?;
            assert!(
                validate_prefill_capacity(
                    context,
                    base + u64::try_from(reserve)? - 1,
                    Some(selected)
                )
                .is_err()
            );
            for cache in [KvCachePolicy::Rotated4, KvCachePolicy::Rotated4Keys3Values] {
                let error = SwiftSession::load_profile_with_prefill(
                    Path::new("THIS_MODEL_MUST_NOT_BE_OPENED.gguf"),
                    Representation::Pq2,
                    None,
                    MatvecBackend::Scalar,
                    42,
                    RowParallelism::default(),
                    cache,
                    context,
                    Some(selected),
                )
                .err()
                .ok_or("incompatible cache was accepted")?;
                assert_eq!(
                    error.to_string(),
                    "CPU tiled prefill requires explicit FP16 KV"
                );
            }
        }
        for tokens in [0, 1, 4, 7, 9, 64, usize::MAX] {
            assert!(CpuPrefill::new(tokens, reserve).is_err());
        }
        assert!(CpuPrefill::new(8, 0).is_err());
        let selected = CpuPrefill::new(8, reserve)?;
        assert!(validate_prefill_capacity(context, 1, Some(selected)).is_err());
        Ok(())
    }

    #[test]
    fn context_capacity_checks_memory_and_preserves_graph_state_ceiling() -> Result<()> {
        for tokens in [1, 4_352, 8_192, 32_768, MAX_CONTEXT_TOKENS] {
            let memory =
                NON_CONTEXT_MEMORY_BYTES + u64::try_from(tokens)? * CONTEXT_TOKEN_MEMORY_BYTES;
            validate_context_capacity(tokens, memory)?;
            assert!(validate_context_capacity(tokens, memory - 1).is_err());
        }
        assert!(validate_context_capacity(0, u64::MAX).is_err());
        assert!(validate_context_capacity(MAX_CONTEXT_TOKENS + 1, u64::MAX).is_err());
        assert!(validate_context_capacity(usize::MAX, u64::MAX).is_err());
        assert_eq!(MAX_CONTEXT_TOKENS, 62_720);
        Ok(())
    }

    #[test]
    fn prepared_chat_seals_session_and_exact_adapter_settings() -> Result<()> {
        let owner = Arc::new(());
        let prompt = PreparedChat {
            owner: owner.clone(),
            adapter_scale: Some(-0.0),
            tokens: vec![1, 2],
            output_tokens: 3,
            enable_thinking: false,
        };
        validate_prompt_owner(&owner, &prompt)?;
        assert!(validate_prompt_owner(&Arc::new(()), &prompt).is_err());
        assert_eq!(prompt.prompt_tokens(), 2);
        assert_eq!(prompt.output_tokens(), 3);
        assert!(!prompt.thinking_enabled());
        assert_eq!(
            prompt.adapter_scale().map(f32::to_bits),
            Some((-0.0_f32).to_bits())
        );
        assert!(validate_adapter_scale(false, None).is_ok());
        assert!(validate_adapter_scale(true, None).is_err());
        for scale in [0.0, -0.0, 1.3, 2.0] {
            validate_adapter_scale(true, Some(scale))?;
            assert!(validate_adapter_scale(false, Some(scale)).is_err());
        }
        for scale in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.01, 2.01] {
            assert!(validate_adapter_scale(true, Some(scale)).is_err());
        }
        Ok(())
    }
}
