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
        lora::{AdapterStorage, EditedAdapter, RankOneAdapter},
        swift::{self, LoadReceipt, MmprojLoadReceipt, Representation},
    },
    runtime::{
        Cancellation, Completion, GenerationTimings, Limits, MediaSpan, MultimodalChunk, Prefill,
        PromptReuse, Runtime,
    },
    sampling::SwiftSampler,
    speculative::{SpeculativeStats, SuffixDraftConfig},
    swift_chat::{
        ContentPart, ImageChunkPlan, MultimodalContent, MultimodalMessage, SwiftChatSession,
    },
    text_output::Utf8Output,
    vision::{EncodedImage, SwiftVisionEncoder, VisionConfig},
    vocabulary::Vocabulary,
};

/// Maximum retained model storage admitted by this Swift session.
pub const MODEL_MEMORY_BYTES: usize = 8 * 1024 * 1024 * 1024;
/// Conservative host commit reserve for the verified Qwen3-VL vision projector.
pub const MMPROJ_MEMORY_BYTES: usize = 768 * 1024 * 1024;
/// Default maximum visual placeholder tokens admitted per input image.
pub const DEFAULT_MAX_IMAGE_TOKENS: usize = 1_024;
/// Smallest selectable per-image token budget.
pub const MIN_IMAGE_TOKENS: usize = 64;
/// Largest selectable per-image token budget.
pub const MAX_IMAGE_TOKENS: usize = 4_096;
/// Maximum number of images permitted in a single conversation prompt.
pub const MAX_PROMPT_IMAGES: usize = 24;
const MAX_CONTENT_PARTS: usize = 128;
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
/// Native sequential FP16 live/rollback KV, attention scores and bookkeeping.
/// Publisher microbatch scratch is a separate execution profile.
pub const CONTEXT_TOKEN_MEMORY_BYTES: u64 = 2 * 65_536 + 24 * 4 + 128;
/// Pinned Swift F32 recurrent/convolution payload, unchanged by KV precision.
pub const RECURRENT_MEMORY_BYTES: usize = 156_893_184;
/// Maximum context fitting the existing graph state bound at FP16 precision.
///
/// Preserve the graph's 4 GiB cap and the pinned 156,893,184-byte recurrent
/// payload. The extra scratch charge and 256-token rounding leave metadata room.
pub const MAX_CONTEXT_TOKENS: usize = max_context_tokens(KvCachePolicy::F16);

/// Selected immutable cache payload per token across all 16 attention layers.
/// Packed records include one F32 scale for every key and value head.
#[must_use]
pub const fn kv_token_memory_bytes(kv_cache: KvCachePolicy) -> usize {
    match kv_cache {
        KvCachePolicy::F16 => 65_536,
        KvCachePolicy::Rotated4 | KvCachePolicy::Rotated4FittedValues => 16_896,
        KvCachePolicy::Rotated4Keys3Values => 14_848,
    }
}

/// Complete per-token native process allowance for two independently live caches.
///
/// The 96-byte score allowance covers 24 sequential query heads. The 128-byte
/// metadata allowance covers token histories, both page tables and Arc headers;
/// pages contain 64 tokens. Fixed codec/transform and prepared-row storage remains
/// in the unchanged 11 GiB allowance, along with recurrence and runtime headroom.
/// Explicit tiles and speculative recovery are additionally charged in full.
#[must_use]
pub const fn context_token_memory_bytes(kv_cache: KvCachePolicy) -> u64 {
    let kv_bytes = match kv_cache {
        KvCachePolicy::F16 => 65_536,
        KvCachePolicy::Rotated4 | KvCachePolicy::Rotated4FittedValues => 16_896,
        KvCachePolicy::Rotated4Keys3Values => 14_848,
    };
    2 * kv_bytes + 96 + 128
}

/// Geometry-derived state/checkpoint ceiling for one explicitly selected Swift cache.
///
/// K4/V4 admits the full model context with its narrow verified-geometry graph
/// constructor. Generic graphs, FP16, fitted-value K4/V4 and K4/V3 retain their
/// 4 GiB ceiling.
/// This is source accounting, not measured memory or quality qualification.
#[must_use]
pub const fn state_limit_bytes(kv_cache: KvCachePolicy) -> usize {
    match kv_cache {
        KvCachePolicy::Rotated4 => crate::graph::SWIFT_RSQ4_FULL_STATE_BYTES,
        KvCachePolicy::F16
        | KvCachePolicy::Rotated4FittedValues
        | KvCachePolicy::Rotated4Keys3Values => crate::graph::MAX_STATE_BYTES,
    }
}

/// Largest page-aligned context admitted by this immutable Swift cache/state scope.
/// This source allowance does not establish measured memory or long-range quality.
#[must_use]
pub const fn max_context_tokens(kv_cache: KvCachePolicy) -> usize {
    let state_max = (state_limit_bytes(kv_cache) - RECURRENT_MEMORY_BYTES)
        / (kv_token_memory_bytes(kv_cache) + 96 + 128)
        / 256
        * 256;
    if state_max < swift::TRAINING_CONTEXT_TOKENS {
        state_max
    } else {
        swift::TRAINING_CONTEXT_TOKENS
    }
}

/// Validate a launcher's selected context and conservative process allowance.
///
/// This never queries memory or creates a Job. The owner enforces that process
/// allowance and independently retains a 4 GiB physical host reserve.
///
/// # Errors
/// Rejects a context outside the model/state bounds or an insufficient allowance.
pub fn validate_context_capacity(context_tokens: usize, memory_limit_bytes: u64) -> Result<()> {
    validate_context_capacity_with_kv(context_tokens, memory_limit_bytes, KvCachePolicy::F16)
}

/// Validate selected-cache process and state accounting before artifact I/O.
///
/// # Errors
/// Rejects a context above the selected cache/model bound or an insufficient allowance.
pub fn validate_context_capacity_with_kv(
    context_tokens: usize,
    memory_limit_bytes: u64,
    kv_cache: KvCachePolicy,
) -> Result<()> {
    let required = u64::try_from(context_tokens)?
        .checked_mul(context_token_memory_bytes(kv_cache))
        .and_then(|bytes| bytes.checked_add(NON_CONTEXT_MEMORY_BYTES))
        .ok_or_else(|| invalid("Swift context memory allowance overflow"))?;
    if context_tokens == 0
        || context_tokens > max_context_tokens(kv_cache)
        || context_tokens > swift::TRAINING_CONTEXT_TOKENS
        || memory_limit_bytes < required
    {
        return Err(invalid("Swift context memory allowance rejected"));
    }
    Ok(())
}

/// Maximum number of reusable branch prefix checkpoints admitted by the runtime.
pub const MAX_REUSABLE_CHECKPOINTS: usize = 8;

/// Explicit exact CPU prompt tiles and their separately charged scratch reserve.
///
/// The selected KV policy is retained. By default this uses sequential-GDN tile
/// arithmetic; calling [`Self::with_chunkwise_gdn`] selects the separately
/// identified FP32 triangular GDN chunk algorithm on the FP16 KV profile.
/// Owners must add the full reserve to their process allowance before loading.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CpuPrefill {
    tokens: usize,
    scratch_bytes: usize,
    chunkwise_gdn: bool,
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
            chunkwise_gdn: false,
        })
    }

    /// Enable the separately identified FP32 triangular GDN chunk prefill algorithm.
    #[must_use]
    pub const fn with_chunkwise_gdn(mut self) -> Self {
        self.chunkwise_gdn = true;
        self
    }

    /// Whether the FP32 triangular chunkwise GDN prefill algorithm is enabled.
    #[must_use]
    pub const fn chunkwise_gdn(self) -> bool {
        self.chunkwise_gdn
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

/// Explicit sparse multi-branch prefix checkpoint slots and optional byte reserve.
///
/// Retaining more than one reusable branch slot or enabling prompt-end capture
/// charges `checkpoints * selected_state_budget` (or a larger explicit caller
/// reserve) against the process memory allowance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BranchCheckpoints {
    checkpoints: usize,
    checkpoint_bytes: Option<usize>,
}

impl Default for BranchCheckpoints {
    fn default() -> Self {
        Self {
            checkpoints: 1,
            checkpoint_bytes: None,
        }
    }
}

impl BranchCheckpoints {
    /// Construct a validated reusable checkpoint slot count (`1..=8`).
    ///
    /// # Errors
    /// Rejects counts outside `1..=MAX_REUSABLE_CHECKPOINTS`.
    pub fn new(checkpoints: usize) -> Result<Self> {
        if !(1..=MAX_REUSABLE_CHECKPOINTS).contains(&checkpoints) {
            return Err(invalid("branch checkpoint count must be from 1 through 8"));
        }
        Ok(Self {
            checkpoints,
            checkpoint_bytes: None,
        })
    }

    /// Construct a validated reusable checkpoint slot count (`1..=8`) paired
    /// with an explicit caller-charged checkpoint byte reserve.
    ///
    /// # Errors
    /// Rejects counts outside `1..=MAX_REUSABLE_CHECKPOINTS` or a zero byte reserve.
    pub fn with_checkpoint_bytes(checkpoints: usize, checkpoint_bytes: usize) -> Result<Self> {
        if !(1..=MAX_REUSABLE_CHECKPOINTS).contains(&checkpoints) || checkpoint_bytes == 0 {
            return Err(invalid("invalid branch checkpoint count or byte reserve"));
        }
        Ok(Self {
            checkpoints,
            checkpoint_bytes: Some(checkpoint_bytes),
        })
    }

    /// Configured reusable checkpoint slot count (`1..=8`).
    #[must_use]
    pub const fn checkpoints(self) -> usize {
        self.checkpoints
    }

    /// Explicit caller-supplied checkpoint byte reserve, when separately configured.
    #[must_use]
    pub const fn checkpoint_bytes(self) -> Option<usize> {
        self.checkpoint_bytes
    }
}

/// Explicit sampled CPU suffix proposals with independently admitted recovery storage.
///
/// Deterministic history proposals use the existing Swift target sampler once
/// per selected token. The graph retains its selected KV precision; this does
/// not make rotated caches equivalent to FP16 or qualify model performance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CpuSpeculation {
    tokens: usize,
    verification_scratch_bytes: usize,
    checkpoint_bytes: usize,
}

impl CpuSpeculation {
    /// Bind a short suffix bound to additional logits and checkpoint reserves.
    ///
    /// The complete caller reserves are charged, including unused headroom.
    /// Graph loading separately checks its logits minimum. Checkpoint reserve
    /// must cover a second complete prefix beside the request rollback.
    ///
    /// # Errors
    /// Rejects bounds outside 1 through 8 or an empty/overflowing reserve.
    pub fn new(
        tokens: usize,
        verification_scratch_bytes: usize,
        checkpoint_bytes: usize,
    ) -> Result<Self> {
        if !(1..=crate::speculative::MAX_SPECULATIVE_DRAFT_TOKENS).contains(&tokens)
            || verification_scratch_bytes == 0
            || checkpoint_bytes == 0
            || verification_scratch_bytes
                .checked_add(checkpoint_bytes)
                .is_none()
        {
            return Err(invalid("invalid CPU speculation selection or reserves"));
        }
        Ok(Self {
            tokens,
            verification_scratch_bytes,
            checkpoint_bytes,
        })
    }

    /// Maximum deterministic suffix positions proposed in one target round.
    #[must_use]
    pub const fn tokens(self) -> usize {
        self.tokens
    }

    /// Additional process allowance reserved for complete per-position logits.
    #[must_use]
    pub const fn verification_scratch_bytes(self) -> usize {
        self.verification_scratch_bytes
    }

    /// Additional complete-prefix recovery allowance, separate from request rollback.
    #[must_use]
    pub const fn checkpoint_bytes(self) -> usize {
        self.checkpoint_bytes
    }

    fn suffix_config(self) -> Result<SuffixDraftConfig> {
        SuffixDraftConfig::new(2, crate::speculative::MAX_SUFFIX_MATCH_TOKENS, self.tokens)
    }
}

fn selected_state_budget(context_tokens: usize, kv_cache: KvCachePolicy) -> Result<usize> {
    validate_context_capacity_with_kv(context_tokens, u64::MAX, kv_cache)?;
    let bytes = context_tokens
        .checked_mul(kv_token_memory_bytes(kv_cache) + 96 + 128)
        .and_then(|bytes| bytes.checked_add(RECURRENT_MEMORY_BYTES))
        .ok_or_else(|| invalid("Swift selected state budget overflow"))?;
    // Preserve fixed graph/checkpoint/logits overhead for short contexts. At
    // larger capacities the per-token allowance also covers the page metadata.
    Ok(bytes.max(STATE_MEMORY_BYTES))
}

/// Validate all explicitly selected CPU execution reserves before artifact I/O.
///
/// The extra checkpoint reserve covers a per-round prefix while the request
/// rollback remains live. Owners must still enforce the process allowance and
/// retain their existing physical host reserve.
///
/// # Errors
/// Rejects missing tiles, insufficient checkpoint/process reserves or overflow.
pub fn validate_execution_capacity(
    context_tokens: usize,
    memory_limit_bytes: u64,
    prefill: Option<CpuPrefill>,
    speculation: Option<CpuSpeculation>,
) -> Result<()> {
    validate_execution_capacity_with_kv(
        context_tokens,
        memory_limit_bytes,
        prefill,
        speculation,
        KvCachePolicy::F16,
    )
}

fn branch_checkpoint_charge(
    context_tokens: usize,
    kv_cache: KvCachePolicy,
    prompt_reuse: PromptReuse,
    branch_checkpoints: BranchCheckpoints,
) -> Result<usize> {
    if !(1..=MAX_REUSABLE_CHECKPOINTS).contains(&branch_checkpoints.checkpoints) {
        return Err(invalid("branch checkpoint count must be from 1 through 8"));
    }
    if prompt_reuse == PromptReuse::Disabled {
        if branch_checkpoints.checkpoints > 1 || branch_checkpoints.checkpoint_bytes.is_some() {
            return Err(invalid(
                "branch checkpoints require prompt reuse to be enabled",
            ));
        }
        return Ok(0);
    }
    let state_budget = selected_state_budget(context_tokens, kv_cache)?;
    let minimum = branch_checkpoints
        .checkpoints
        .checked_mul(state_budget)
        .ok_or_else(|| invalid("Swift branch checkpoint reservation overflow"))?;
    if let Some(explicit_bytes) = branch_checkpoints.checkpoint_bytes {
        if explicit_bytes < minimum {
            return Err(invalid(
                "branch checkpoint reserve is below selected state budget",
            ));
        }
        return Ok(explicit_bytes);
    }
    if branch_checkpoints.checkpoints > 1 || prompt_reuse == PromptReuse::PromptEndAndLatest {
        return Ok(minimum);
    }
    Ok(0)
}

/// Validate selected-cache context, execution reserves, and sparse branch checkpoints.
///
/// Retaining more than one reusable branch checkpoint slot or selecting
/// [`PromptReuse::PromptEndAndLatest`] charges `slots * selected_state_budget`
/// (or a larger caller-supplied reserve) against `memory_limit_bytes`.
///
/// # Errors
/// Rejects invalid pairing, undersized reserves, capacity overflow or insufficient allowance.
pub fn validate_execution_capacity_with_reuse(
    context_tokens: usize,
    memory_limit_bytes: u64,
    prefill: Option<CpuPrefill>,
    speculation: Option<CpuSpeculation>,
    kv_cache: KvCachePolicy,
    prompt_reuse: PromptReuse,
    branch_checkpoints: BranchCheckpoints,
) -> Result<()> {
    let branch_charge =
        branch_checkpoint_charge(context_tokens, kv_cache, prompt_reuse, branch_checkpoints)?;
    let available = memory_limit_bytes
        .checked_sub(u64::try_from(branch_charge)?)
        .ok_or_else(|| invalid("Swift branch checkpoint memory allowance rejected"))?;
    validate_execution_capacity_with_kv(context_tokens, available, prefill, speculation, kv_cache)
}

/// Validate selected-cache context plus complete tile and speculative reserves.
///
/// # Errors
/// Rejects context/state limits, missing tiles, incompatible chunkwise GDN,
/// insufficient reserves or overflow.
pub fn validate_execution_capacity_with_kv(
    context_tokens: usize,
    memory_limit_bytes: u64,
    prefill: Option<CpuPrefill>,
    speculation: Option<CpuSpeculation>,
    kv_cache: KvCachePolicy,
) -> Result<()> {
    if prefill.is_some_and(CpuPrefill::chunkwise_gdn) && speculation.is_some() {
        return Err(invalid(
            "chunkwise GDN prefill is incompatible with CPU speculation",
        ));
    }
    let additional = if let Some(speculation) = speculation {
        if prefill.is_none()
            || speculation.checkpoint_bytes < selected_state_budget(context_tokens, kv_cache)?
        {
            return Err(invalid(
                "CPU speculation requires tiles and a complete prefix reserve",
            ));
        }
        speculation
            .verification_scratch_bytes
            .checked_add(speculation.checkpoint_bytes)
            .ok_or_else(|| invalid("Swift CPU execution reserve overflow"))?
    } else {
        0
    };
    let available = memory_limit_bytes
        .checked_sub(u64::try_from(additional)?)
        .ok_or_else(|| invalid("Swift CPU speculation memory allowance rejected"))?;
    validate_prefill_capacity_with_kv(context_tokens, available, prefill, kv_cache)
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
    validate_prefill_capacity_with_kv(
        context_tokens,
        memory_limit_bytes,
        prefill,
        KvCachePolicy::F16,
    )
}

/// Validate selected-cache context and the entire additional tile reservation.
///
/// # Errors
/// Rejects invalid context capacity, non-FP16 chunkwise GDN, insufficient memory
/// and reservation overflow.
pub fn validate_prefill_capacity_with_kv(
    context_tokens: usize,
    memory_limit_bytes: u64,
    prefill: Option<CpuPrefill>,
    kv_cache: KvCachePolicy,
) -> Result<()> {
    if prefill.is_some_and(CpuPrefill::chunkwise_gdn) && kv_cache != KvCachePolicy::F16 {
        return Err(invalid(
            "chunkwise GDN prefill requires the separate FP16 KV cache profile",
        ));
    }
    let reserve = prefill.map_or(0, CpuPrefill::scratch_bytes);
    let available = memory_limit_bytes
        .checked_sub(u64::try_from(reserve)?)
        .ok_or_else(|| invalid("Swift CPU prefill memory allowance rejected"))?;
    validate_context_capacity_with_kv(context_tokens, available, kv_cache)
}

/// Explicit breakdown of host process commit and external accelerator memory.
///
/// Windows Job-object commit covers pinned host weights, live and checkpointed
/// state, logits/replay/tile scratch, and fixed runtime headroom; external
/// accelerator allocations (`accelerator_external_bytes`) are separated so
/// callers never equate host commit limits with total device footprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SwiftExecutionMemoryLedger {
    /// Pinned host model weights allowance.
    pub host_weights_bytes: u64,
    /// Live `F32` recurrent/convolution state across all layers.
    pub live_recurrent_bytes: u64,
    /// Live attention `KV` cache bytes for the configured context length.
    pub live_kv_bytes: u64,
    /// Retained and rollback checkpoint `F32` recurrent/convolution bytes.
    pub checkpoint_recurrent_bytes: u64,
    /// Retained and rollback checkpoint `KV` cache bytes.
    pub checkpoint_kv_bytes: u64,
    /// Bounded speculative-verification or step logits scratch bytes.
    pub verification_logits_bytes: u64,
    /// Context bookkeeping, attention score, and replay history scratch bytes.
    pub replay_scratch_bytes: u64,
    /// Explicit `CPU` tiled-prefill scratch allowance.
    pub tile_scratch_bytes: u64,
    /// External accelerator/GPU memory outside the host Job commit limit.
    pub accelerator_external_bytes: u64,
    /// Fixed codec/transform/allocator fragmentation and runtime headroom.
    pub fragmentation_margin_bytes: u64,
}

impl SwiftExecutionMemoryLedger {
    /// Constructs an itemized memory ledger for an immutable Swift profile.
    ///
    /// # Errors
    /// Rejects invalid context/KV geometry and `u64` arithmetic overflow.
    pub fn for_swift_profile(
        context_tokens: usize,
        kv_cache: KvCachePolicy,
        cpu_prefill: Option<CpuPrefill>,
        cpu_speculation: Option<CpuSpeculation>,
        checkpoint_slots: usize,
        accelerator_external_bytes: u64,
    ) -> Result<Self> {
        validate_context_capacity_with_kv(context_tokens, u64::MAX, kv_cache)?;
        let context_u64 = u64::try_from(context_tokens)?;
        let slots_u64 = u64::try_from(checkpoint_slots)?;
        let host_weights_bytes = u64::try_from(MODEL_MEMORY_BYTES)?;
        let live_recurrent_bytes = u64::try_from(RECURRENT_MEMORY_BYTES)?;
        let live_kv_bytes = context_u64
            .checked_mul(u64::try_from(kv_token_memory_bytes(kv_cache))?)
            .ok_or_else(|| invalid("Swift ledger live KV overflow"))?;
        let spec_checkpoint_bytes =
            u64::try_from(cpu_speculation.map_or(0, CpuSpeculation::checkpoint_bytes))?;
        let spec_recurrent_bytes = spec_checkpoint_bytes.min(live_recurrent_bytes);
        let spec_kv_bytes = spec_checkpoint_bytes.saturating_sub(spec_recurrent_bytes);
        let checkpoint_recurrent_bytes = slots_u64
            .checked_mul(live_recurrent_bytes)
            .and_then(|bytes| bytes.checked_add(spec_recurrent_bytes))
            .ok_or_else(|| invalid("Swift ledger checkpoint recurrent overflow"))?;
        let checkpoint_kv_bytes = slots_u64
            .checked_mul(live_kv_bytes)
            .and_then(|bytes| bytes.checked_add(spec_kv_bytes))
            .ok_or_else(|| invalid("Swift ledger checkpoint KV overflow"))?;
        let verification_logits_bytes = if let Some(speculation) = cpu_speculation {
            u64::try_from(speculation.verification_scratch_bytes())?
        } else {
            u64::try_from(
                swift::VOCAB_SIZE
                    .checked_mul(4)
                    .ok_or_else(|| invalid("Swift ledger logits scratch overflow"))?,
            )?
        };
        let replay_scratch_bytes = context_u64
            .checked_mul(96 + 128)
            .ok_or_else(|| invalid("Swift ledger replay scratch overflow"))?;
        let tile_scratch_bytes = u64::try_from(cpu_prefill.map_or(0, CpuPrefill::scratch_bytes))?;
        let fixed_host_baseline = u64::try_from(MODEL_MEMORY_BYTES)?
            .checked_add(u64::try_from(CHECKPOINT_MEMORY_BYTES)?)
            .ok_or_else(|| invalid("Swift ledger baseline overflow"))?;
        let fragmentation_margin_bytes =
            NON_CONTEXT_MEMORY_BYTES.saturating_sub(fixed_host_baseline);
        Ok(Self {
            host_weights_bytes,
            live_recurrent_bytes,
            live_kv_bytes,
            checkpoint_recurrent_bytes,
            checkpoint_kv_bytes,
            verification_logits_bytes,
            replay_scratch_bytes,
            tile_scratch_bytes,
            accelerator_external_bytes,
            fragmentation_margin_bytes,
        })
    }

    /// Host commit total excluding `accelerator_external_bytes`.
    ///
    /// Returns `None` on `u64` overflow.
    #[must_use]
    pub const fn job_commit_total_bytes(&self) -> Option<u64> {
        let Some(total) = self
            .host_weights_bytes
            .checked_add(self.live_recurrent_bytes)
        else {
            return None;
        };
        let Some(total) = total.checked_add(self.live_kv_bytes) else {
            return None;
        };
        let Some(total) = total.checked_add(self.checkpoint_recurrent_bytes) else {
            return None;
        };
        let Some(total) = total.checked_add(self.checkpoint_kv_bytes) else {
            return None;
        };
        let Some(total) = total.checked_add(self.verification_logits_bytes) else {
            return None;
        };
        let Some(total) = total.checked_add(self.replay_scratch_bytes) else {
            return None;
        };
        let Some(total) = total.checked_add(self.tile_scratch_bytes) else {
            return None;
        };
        total.checked_add(self.fragmentation_margin_bytes)
    }

    /// Total estimated footprint including `accelerator_external_bytes`.
    ///
    /// Returns `None` on `u64` overflow.
    #[must_use]
    pub const fn total_with_external_bytes(&self) -> Option<u64> {
        let Some(host_commit) = self.job_commit_total_bytes() else {
            return None;
        };
        host_commit.checked_add(self.accelerator_external_bytes)
    }
}

/// Largest complete displayed output retained or delivered by one request.
pub const MAX_OUTPUT_BYTES: usize = 4 * 1024 * 1024;

struct PreparedMediaChunk {
    span: MediaSpan,
    encoded: EncodedImage,
}

/// Exact rendered and counted input, sealed by its originating verified session.
pub struct PreparedChat {
    owner: Arc<()>,
    adapter_scale: Option<f32>,
    tokens: Vec<u32>,
    media_chunks: Vec<PreparedMediaChunk>,
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

    /// Number of encoded visual image chunks included in the prepared prompt.
    #[must_use]
    pub const fn image_count(&self) -> usize {
        self.media_chunks.len()
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
    cpu_speculation: Option<CpuSpeculation>,
    adapter_identity: Option<String>,
    adapter_storage: Option<AdapterStorage>,
    runtime: Runtime<Qwen35>,
    chat: SwiftChatSession,
    vocabulary: Vocabulary,
    sampler: SwiftSampler,
    seed: u32,
    delivery_fault: bool,
    prompt_reuse: PromptReuse,
    branch_checkpoints: BranchCheckpoints,
    runtime_checkpoints: usize,
    last_speculative_stats: Option<SpeculativeStats>,
    vision: Option<Arc<SwiftVisionEncoder>>,
    mmproj_receipt: Option<MmprojLoadReceipt>,
    max_image_tokens: usize,
    receipt: LoadReceipt,
}

#[derive(Clone, Copy)]
enum AdapterSelection<'a> {
    Approved(Option<&'a Path>),
    Edited(EditedAdapter<'a>),
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
    /// Generic/FP16 graphs retain 4 GiB. Explicit full-context Swift K4/V4
    /// uses its geometry-derived 4,644,798,464-byte state/checkpoint ceiling.
    ///
    /// # Errors
    /// Rejects a context beyond its selected cache/state capacity and the
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
    /// Existing loaders keep ordinary prompt ingestion. An explicit tile retains
    /// the selected KV precision. The model owner must charge [`CpuPrefill::scratch_bytes`] in addition to ordinary
    /// context/process admission, using [`validate_prefill_capacity`]. Graph
    /// construction independently rejects an insufficient geometry-specific reserve.
    /// Reset, adapter changes and exact latest-prefix reuse retain this selection.
    /// No sampling, decode, triangular-GDN or automatic accelerator policy changes.
    ///
    /// # Errors
    /// Rejects insufficient scratch and the errors of
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
        Self::load_profile_with_execution(
            path,
            representation,
            adapter_path,
            backend,
            seed,
            parallelism,
            kv_cache,
            context_tokens,
            cpu_prefill,
            None,
        )
    }

    /// Load an immutable, explicitly admitted CPU tile and sampled suffix profile.
    ///
    /// Existing loaders select no speculation. Both declared reserves are added
    /// to ordinary admission: per-position logits and a second full prefix
    /// checkpoint beside the request rollback. Invalid pairing and insufficient
    /// recovery reserves reject before artifact I/O. The exact graph independently
    /// validates its logits reservation before enabling target tiles.
    ///
    /// # Errors
    /// Returns execution admission, artifact, graph and allocation failures.
    #[expect(
        clippy::too_many_arguments,
        reason = "Explicit immutable execution extends the compatible profile loader."
    )]
    pub fn load_profile_with_execution(
        path: &Path,
        representation: Representation,
        adapter_path: Option<&Path>,
        backend: MatvecBackend,
        seed: u32,
        parallelism: RowParallelism,
        kv_cache: KvCachePolicy,
        context_tokens: usize,
        cpu_prefill: Option<CpuPrefill>,
        cpu_speculation: Option<CpuSpeculation>,
    ) -> Result<Self> {
        Self::load_profile_with_execution_and_reuse(
            path,
            representation,
            adapter_path,
            backend,
            seed,
            parallelism,
            kv_cache,
            context_tokens,
            cpu_prefill,
            cpu_speculation,
            PromptReuse::Disabled,
            BranchCheckpoints::default(),
        )
    }

    /// Load an immutable CPU/accelerator profile with explicit prompt-reuse and
    /// sparse multi-branch checkpoint retention (`1..=8` reusable slots).
    ///
    /// Invalid combinations (including chunkwise GDN outside FP16 or alongside
    /// speculation, and branch checkpoints when reuse is disabled) reject before
    /// any artifact I/O.
    ///
    /// # Errors
    /// Returns execution/reuse admission, artifact, graph and allocation failures.
    #[expect(
        clippy::too_many_arguments,
        reason = "Explicit prompt reuse and branch checkpoint slots extend the immutable execution loader."
    )]
    pub fn load_profile_with_execution_and_reuse(
        path: &Path,
        representation: Representation,
        adapter_path: Option<&Path>,
        backend: MatvecBackend,
        seed: u32,
        parallelism: RowParallelism,
        kv_cache: KvCachePolicy,
        context_tokens: usize,
        cpu_prefill: Option<CpuPrefill>,
        cpu_speculation: Option<CpuSpeculation>,
        prompt_reuse: PromptReuse,
        branch_checkpoints: BranchCheckpoints,
    ) -> Result<Self> {
        Self::load_selected_profile(
            path,
            representation,
            AdapterSelection::Approved(adapter_path),
            backend,
            seed,
            parallelism,
            kv_cache,
            context_tokens,
            cpu_prefill,
            cpu_speculation,
            prompt_reuse,
            branch_checkpoints,
        )
    }

    /// Load one explicitly bound edited adapter with experimental sparse A dots.
    ///
    /// The model, tokenizer, context, KV cache and transaction contracts remain
    /// those of the exact selected Swift artifact. Adapter authenticity is the
    /// caller's candidate hash, not the released Orca approval. Sparse arithmetic
    /// has its own graph identity. No HTTP or ordinary loading path calls this
    /// research entry point; it selects no tiled prefill or speculation.
    ///
    /// # Errors
    /// Rejects candidate identity/inventory, unprofitable sparse factors and the
    /// model, context, allocation and execution errors of ordinary loading.
    #[expect(
        clippy::too_many_arguments,
        reason = "The explicit candidate replaces the approved adapter in the established immutable context loader."
    )]
    pub fn load_edited_profile_with_context(
        path: &Path,
        representation: Representation,
        adapter: EditedAdapter<'_>,
        backend: MatvecBackend,
        seed: u32,
        parallelism: RowParallelism,
        kv_cache: KvCachePolicy,
        context_tokens: usize,
    ) -> Result<Self> {
        Self::load_selected_profile(
            path,
            representation,
            AdapterSelection::Edited(adapter),
            backend,
            seed,
            parallelism,
            kv_cache,
            context_tokens,
            None,
            None,
            PromptReuse::Disabled,
            BranchCheckpoints::default(),
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Both explicit adapter selections share one immutable execution loader."
    )]
    fn load_selected_profile(
        path: &Path,
        representation: Representation,
        adapter_selection: AdapterSelection<'_>,
        backend: MatvecBackend,
        seed: u32,
        parallelism: RowParallelism,
        kv_cache: KvCachePolicy,
        context_tokens: usize,
        cpu_prefill: Option<CpuPrefill>,
        cpu_speculation: Option<CpuSpeculation>,
        prompt_reuse: PromptReuse,
        branch_checkpoints: BranchCheckpoints,
    ) -> Result<Self> {
        if let AdapterSelection::Edited(candidate) = &adapter_selection {
            candidate.validate_identity()?;
        }
        validate_execution_capacity_with_reuse(
            context_tokens,
            u64::MAX,
            cpu_prefill,
            cpu_speculation,
            kv_cache,
            prompt_reuse,
            branch_checkpoints,
        )?;
        let state_budget = selected_state_budget(context_tokens, kv_cache)?;
        let branch_budget =
            branch_checkpoint_charge(context_tokens, kv_cache, prompt_reuse, branch_checkpoints)?;
        let checkpoint_budget = CHECKPOINT_MEMORY_BYTES
            .max(state_budget)
            .checked_add(cpu_speculation.map_or(0, CpuSpeculation::checkpoint_bytes))
            .and_then(|bytes| bytes.checked_add(branch_budget))
            .ok_or_else(|| invalid("Swift checkpoint reservation overflow"))?;
        let sampler = SwiftSampler::new(seed)?;
        let (model, receipt) =
            swift::load_representation(path, MODEL_MEMORY_BYTES, representation)?;
        let chat = SwiftChatSession::from_model_representation(&model, representation)?
            .with_context_tokens(context_tokens)?;
        let vocabulary = Vocabulary::from_metadata(&model.metadata, swift::VOCAB_SIZE)?;
        let adapter = match adapter_selection {
            AdapterSelection::Approved(path) => path
                .map(|path| RankOneAdapter::load_swift_representation(path, &model, representation))
                .transpose()?,
            AdapterSelection::Edited(candidate) => Some(RankOneAdapter::load_edited_swift(
                candidate,
                &model,
                representation,
            )?),
        };
        let adapter_identity = adapter
            .as_ref()
            .map(|adapter| adapter.identity().to_owned());
        let adapter_storage = adapter.as_ref().map(RankOneAdapter::storage);
        let mut decoder = Qwen35::with_swift_state_budget_and_kv(
            model,
            context_tokens,
            state_budget,
            backend,
            kv_cache,
            adapter,
        )?;
        decoder.set_row_parallelism(parallelism);
        if let Some(prefill) = cpu_prefill {
            decoder = decoder.with_tiled_prefill(prefill.tokens, prefill.scratch_bytes)?;
            if prefill.chunkwise_gdn() {
                decoder = decoder.with_chunkwise_gdn_prefill()?;
            }
        }
        if let Some(speculation) = cpu_speculation {
            decoder = decoder.with_tiled_verification(speculation.verification_scratch_bytes)?;
        }
        let runtime_checkpoints = if prompt_reuse == PromptReuse::Disabled {
            0
        } else {
            branch_checkpoints.checkpoints()
        };
        let mut runtime = Runtime::new(
            decoder,
            Limits {
                context_tokens,
                checkpoints: runtime_checkpoints,
                checkpoint_bytes: checkpoint_budget,
            },
        )?;
        runtime.set_retain_prompt_end(prompt_reuse == PromptReuse::PromptEndAndLatest);
        Ok(Self {
            owner: Arc::new(()),
            representation,
            kv_cache,
            cpu_prefill,
            cpu_speculation,
            adapter_identity,
            adapter_storage,
            runtime,
            chat,
            vocabulary,
            sampler,
            seed,
            delivery_fault: false,
            prompt_reuse,
            branch_checkpoints,
            runtime_checkpoints,
            last_speculative_stats: None,
            vision: None,
            mmproj_receipt: None,
            max_image_tokens: DEFAULT_MAX_IMAGE_TOKENS,
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

    /// Immutable sampled CPU suffix policy, absent for ordinary generation.
    #[must_use]
    pub const fn cpu_speculation(&self) -> Option<CpuSpeculation> {
        self.cpu_speculation
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

    /// Exact verified adapter artifact SHA-256, when one was explicitly loaded.
    #[must_use]
    pub fn adapter_identity(&self) -> Option<&str> {
        self.adapter_identity.as_deref()
    }

    /// Retained adapter factor payload; excludes process memory and overhead.
    #[must_use]
    pub const fn adapter_storage(&self) -> Option<AdapterStorage> {
        self.adapter_storage
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

    /// Installed projection executor identity, absent for ordinary CPU execution.
    ///
    /// This describes installation, not device execution or qualification.
    #[must_use]
    pub fn projection_executor_identity(&self) -> Option<&str> {
        self.runtime.decoder().projection_executor_identity()
    }

    /// Explicit request-boundary policy; fresh recomputation remains the default.
    #[must_use]
    pub const fn prompt_reuse(&self) -> PromptReuse {
        self.prompt_reuse
    }

    /// Configured sparse multi-branch prefix checkpoint slot count (`1..=8`).
    #[must_use]
    pub const fn prompt_reuse_checkpoints(&self) -> usize {
        self.branch_checkpoints.checkpoints()
    }

    /// Configured sparse multi-branch prefix checkpoint selection and optional reserve.
    #[must_use]
    pub const fn branch_checkpoints(&self) -> BranchCheckpoints {
        self.branch_checkpoints
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
            self.last_speculative_stats = None;
            self.runtime
                .set_retain_prompt_end(reuse == PromptReuse::PromptEndAndLatest);
        }
        Ok(())
    }

    /// Whether reset is required before another completion can start.
    #[must_use]
    pub const fn is_faulted(&self) -> bool {
        self.delivery_fault || self.runtime.is_faulted()
    }

    /// Speculative decoding counters recorded by the most recent speculative completion.
    ///
    /// Returns `None` after session load, reset, or non-speculative completion.
    #[must_use]
    pub const fn last_speculative_stats(&self) -> Option<&SpeculativeStats> {
        self.last_speculative_stats.as_ref()
    }

    /// Load and verify the pinned Qwen3-VL `mmproj` projector for multimodal prompts.
    ///
    /// # Errors
    /// Rejects closed admission, out-of-range image token budgets, hash/metadata
    /// drift, or storage budget exhaustion before enabling vision encoding.
    pub fn load_vision(
        &mut self,
        mmproj_path: &Path,
        max_image_tokens: usize,
    ) -> Result<&MmprojLoadReceipt> {
        if self.is_faulted() {
            return Err(invalid("Swift session requires reset"));
        }
        validate_vision_token_budget(max_image_tokens)?;
        let (encoder, receipt) = SwiftVisionEncoder::load(
            mmproj_path,
            MMPROJ_MEMORY_BYTES,
            self.requested_backend(),
            self.clone_row_parallelism(),
        )?;
        self.vision = Some(Arc::new(encoder));
        self.max_image_tokens = max_image_tokens;
        Ok(self.mmproj_receipt.insert(receipt))
    }

    /// Whether a verified vision projector is loaded on this session.
    #[must_use]
    pub const fn has_vision(&self) -> bool {
        self.vision.is_some()
    }

    /// Verified storage accounting receipt for the loaded `mmproj` artifact.
    #[must_use]
    pub const fn mmproj_receipt(&self) -> Option<&MmprojLoadReceipt> {
        self.mmproj_receipt.as_ref()
    }

    /// Hyperparameters verified from the loaded `mmproj` artifact, when present.
    #[must_use]
    pub fn vision_config(&self) -> Option<&VisionConfig> {
        self.vision.as_deref().map(SwiftVisionEncoder::config)
    }

    /// Maximum vision placeholder token budget configured per input image.
    #[must_use]
    pub const fn max_image_tokens(&self) -> usize {
        self.max_image_tokens
    }

    /// Itemized host process commit and external accelerator memory ledger.
    ///
    /// # Errors
    /// Rejects invalid or overflowing capacity calculations.
    pub fn execution_memory_ledger(
        &self,
        accelerator_external_bytes: u64,
    ) -> Result<SwiftExecutionMemoryLedger> {
        let checkpoint_slots = match self.prompt_reuse {
            PromptReuse::Disabled => 1,
            PromptReuse::Latest => 1 + self.branch_checkpoints.checkpoints(),
            PromptReuse::PromptEndAndLatest => 2 + self.branch_checkpoints.checkpoints(),
        };
        SwiftExecutionMemoryLedger::for_swift_profile(
            self.context_tokens(),
            self.kv_cache,
            self.cpu_prefill,
            self.cpu_speculation,
            checkpoint_slots,
            accelerator_external_bytes,
        )
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
        if messages_have_array_content(messages) {
            return self.prepare_multimodal_with_settings(
                messages,
                output_tokens,
                enable_thinking,
                adapter_scale,
            );
        }
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
            media_chunks: Vec::new(),
            output_tokens,
            enable_thinking,
        })
    }

    /// Encode image attachments, splice vision patch spans and seal exact prompt tokens.
    ///
    /// # Errors
    /// Rejects closed admission, missing vision projector, malformed parts/URLs,
    /// image decode errors, mismatched settings and context overflow.
    pub fn prepare_multimodal_with_settings(
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
        let vision = self
            .vision
            .as_ref()
            .ok_or_else(|| invalid("Swift vision projector is not loaded"))?;
        let raw_messages = parse_multimodal_messages(messages)?;
        validate_multimodal_budget(&raw_messages, output_tokens, self.context_tokens())?;
        let encoded_images =
            encode_parsed_images(&raw_messages, vision.as_ref(), self.max_image_tokens)?;
        let mut part_lists = Vec::with_capacity(raw_messages.len());
        for message in &raw_messages {
            let mut parts = Vec::with_capacity(message.parts.len());
            for part in &message.parts {
                match part {
                    ParsedPart::Text(text) => parts.push(ContentPart::Text(text.as_str())),
                    ParsedPart::ImageUrl { image_index, .. } => {
                        let encoded = &encoded_images[*image_index];
                        parts.push(ContentPart::Image(ImageChunkPlan {
                            grid_width: encoded.grid_width,
                            grid_height: encoded.grid_height,
                        }));
                    }
                }
            }
            part_lists.push(parts);
        }
        let mm_messages: Vec<MultimodalMessage<'_>> = raw_messages
            .iter()
            .zip(part_lists.iter())
            .map(|(message, parts)| MultimodalMessage {
                role: message.role.as_str(),
                content: message.text.as_deref().map_or(
                    MultimodalContent::Parts(parts.as_slice()),
                    MultimodalContent::Text,
                ),
                reasoning_content: message.reasoning_content.as_deref(),
            })
            .collect();
        let prepared = self.chat.prepare_multimodal_with_thinking(
            &mm_messages,
            output_tokens,
            enable_thinking,
        )?;
        if prepared.image_spans.len() != encoded_images.len() {
            return Err(invalid("multimodal image span count mismatch"));
        }
        let media_chunks = prepared
            .image_spans
            .into_iter()
            .zip(encoded_images)
            .map(|(span, encoded)| PreparedMediaChunk {
                span: MediaSpan {
                    offset: span.token_offset,
                    len: span.token_count,
                    grid_width: span.grid_width,
                    grid_height: span.grid_height,
                    content_sha256: encoded.sha256,
                },
                encoded,
            })
            .collect();
        Ok(PreparedChat {
            owner: self.owner.clone(),
            adapter_scale,
            tokens: prepared.tokenization.token_ids,
            media_chunks,
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
        self.last_speculative_stats = None;
        self.runtime.reset()?;
        self.sampler = SwiftSampler::new(self.seed)?;
        self.delivery_fault = false;
        Ok(())
    }

    /// Generate sampled text under the complete output cap and optional deadline.
    ///
    /// None leaves ordinary generation uncapped in elapsed time; cancellation,
    /// context/output bounds and transactional recovery still apply.
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
        deadline: impl Into<Option<Instant>>,
        emit: impl FnMut(TextEvent<'_>) -> Result<()>,
    ) -> Result<Completion> {
        self.complete_observed(prompt, cancellation, deadline.into(), None, emit)
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
        deadline: impl Into<Option<Instant>>,
        timings: &mut GenerationTimings,
        emit: impl FnMut(TextEvent<'_>) -> Result<()>,
    ) -> Result<Completion> {
        *timings = GenerationTimings::default();
        self.complete_observed(prompt, cancellation, deadline.into(), Some(timings), emit)
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
        deadline: impl Into<Option<Instant>>,
        mut emit: impl FnMut(TextEvent<'_>) -> Result<()>,
    ) -> Result<(Completion, SpeculativeStats)> {
        let deadline = deadline.into();
        self.last_speculative_stats = None;
        suffix_config.validate()?;
        if !prompt.media_chunks.is_empty() {
            return Err(invalid(
                "CPU suffix speculation does not support multimodal image prompts",
            ));
        }
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
            emit_decoded_token(
                &self.vocabulary,
                &mut display,
                &mut output_bytes,
                token,
                &mut emit,
            )
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
        self.last_speculative_stats = Some(stats);
        Ok((completion, stats))
    }

    fn complete_observed(
        &mut self,
        prompt: &PreparedChat,
        cancellation: &Cancellation,
        deadline: Option<Instant>,
        timings: Option<&mut GenerationTimings>,
        mut emit: impl FnMut(TextEvent<'_>) -> Result<()>,
    ) -> Result<Completion> {
        let is_multimodal = !prompt.media_chunks.is_empty();
        if timings.is_some() && self.cpu_speculation.is_some() && !is_multimodal {
            return Err(invalid(
                "runtime phase timings are unavailable for CPU speculation",
            ));
        }
        self.last_speculative_stats = None;
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
            emit_decoded_token(
                &self.vocabulary,
                &mut display,
                &mut output_bytes,
                token,
                &mut emit,
            )
        };
        let mut recorded_stats = None;
        let completion = if is_multimodal {
            execute_multimodal_completion(
                &mut self.runtime,
                self.prompt_reuse == PromptReuse::Disabled && self.runtime_checkpoints == 0,
                &mut self.sampler,
                prompt,
                &options,
                cancellation,
                deadline,
                timings,
                consume,
            )?
        } else if let Some(speculation) = self.cpu_speculation {
            let (completion, stats) = self.runtime.generate_sampled_speculative_until(
                &prompt.tokens,
                &options,
                Prefill { block_tokens: 64 },
                speculation.suffix_config()?,
                cancellation,
                deadline,
                &mut self.sampler,
                consume,
            )?;
            recorded_stats = Some(stats);
            completion
        } else if self.prompt_reuse == PromptReuse::Disabled && self.runtime_checkpoints == 0 {
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
        self.last_speculative_stats = recorded_stats;
        Ok(completion)
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Borrowing disjoint session fields keeps the token output closure free of mutable session overlap."
)]
fn execute_multimodal_completion(
    runtime: &mut Runtime<Qwen35>,
    reset_only: bool,
    sampler: &mut SwiftSampler,
    prompt: &PreparedChat,
    options: &crate::runtime::Generation,
    cancellation: &Cancellation,
    deadline: Option<Instant>,
    timings: Option<&mut GenerationTimings>,
    consume: impl FnMut(u32) -> Result<()>,
) -> Result<Completion> {
    let chunks: Vec<MultimodalChunk<'_>> = prompt
        .media_chunks
        .iter()
        .map(|chunk| MultimodalChunk {
            span: chunk.span,
            embeddings: &chunk.encoded.embeddings,
        })
        .collect();
    if reset_only {
        runtime.generate_multimodal_sampled_until_for_reset(
            &prompt.tokens,
            &chunks,
            options,
            Prefill { block_tokens: 64 },
            cancellation,
            deadline,
            sampler,
            timings,
            consume,
        )
    } else if let Some(timings) = timings {
        runtime.generate_multimodal_sampled_until_with_timings(
            &prompt.tokens,
            &chunks,
            options,
            Prefill { block_tokens: 64 },
            cancellation,
            deadline,
            sampler,
            timings,
            consume,
        )
    } else {
        runtime.generate_multimodal_sampled_until(
            &prompt.tokens,
            &chunks,
            options,
            Prefill { block_tokens: 64 },
            cancellation,
            deadline,
            sampler,
            consume,
        )
    }
}

fn messages_have_array_content(messages: &Value) -> bool {
    messages.as_array().is_some_and(|list| {
        list.iter()
            .any(|message| message.get("content").is_some_and(Value::is_array))
    })
}

fn validate_vision_token_budget(max_image_tokens: usize) -> Result<()> {
    if !(MIN_IMAGE_TOKENS..=MAX_IMAGE_TOKENS).contains(&max_image_tokens) {
        return Err(invalid(
            "Swift image token budget must be from 64 through 4096",
        ));
    }
    Ok(())
}

fn validate_multimodal_budget(
    messages: &[ParsedMessage],
    output_tokens: usize,
    context_tokens: usize,
) -> Result<()> {
    if output_tokens == 0 || output_tokens > swift::MAX_OUTPUT_TOKENS {
        return Err(invalid(
            "Swift multimodal output reservation exceeds supported bounds",
        ));
    }
    let image_count = messages
        .iter()
        .flat_map(|message| &message.parts)
        .filter(|part| matches!(part, ParsedPart::ImageUrl { .. }))
        .count();
    // Each image emits at least `<|vision_start|><|image_pad|><|vision_end|>` plus prompt framing.
    let min_image_tokens = image_count.saturating_mul(3);
    if output_tokens.saturating_add(min_image_tokens) >= context_tokens {
        return Err(invalid(
            "Swift multimodal prompt reservation exceeds context budget",
        ));
    }
    Ok(())
}

enum ParsedPart {
    Text(String),
    ImageUrl { url: String, image_index: usize },
}

struct ParsedMessage {
    role: String,
    text: Option<String>,
    parts: Vec<ParsedPart>,
    reasoning_content: Option<String>,
}

fn parse_multimodal_part(part: &Value, image_count: &mut usize) -> Result<ParsedPart> {
    let object = part
        .as_object()
        .ok_or_else(|| invalid("Swift multimodal content part must be an object"))?;
    let kind = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("Swift multimodal part requires type"))?;
    match kind {
        "text" => {
            if object
                .keys()
                .any(|key| !matches!(key.as_str(), "type" | "text"))
            {
                return Err(invalid("unsupported Swift multimodal text part field"));
            }
            let text = object
                .get("text")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("Swift multimodal text part requires text"))?;
            Ok(ParsedPart::Text(text.to_owned()))
        }
        "image_url" => {
            if object
                .keys()
                .any(|key| !matches!(key.as_str(), "type" | "image_url"))
            {
                return Err(invalid("unsupported Swift multimodal image_url part field"));
            }
            let image_obj = object
                .get("image_url")
                .and_then(Value::as_object)
                .ok_or_else(|| invalid("Swift image_url part requires image_url object"))?;
            if image_obj
                .keys()
                .any(|key| !matches!(key.as_str(), "url" | "detail"))
            {
                return Err(invalid("unsupported Swift image_url object field"));
            }
            if let Some(detail) = image_obj.get("detail")
                && !detail
                    .as_str()
                    .is_some_and(|value| matches!(value, "auto" | "low" | "high"))
            {
                return Err(invalid("invalid Swift image_url detail value"));
            }
            let url = image_obj
                .get("url")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("Swift image_url requires url string"))?;
            if *image_count >= MAX_PROMPT_IMAGES {
                return Err(invalid("Swift prompt image count exceeds supported bounds"));
            }
            let image_index = *image_count;
            *image_count += 1;
            Ok(ParsedPart::ImageUrl {
                url: url.to_owned(),
                image_index,
            })
        }
        _ => Err(invalid("unsupported Swift multimodal part type")),
    }
}

fn parse_multimodal_messages(messages: &Value) -> Result<Vec<ParsedMessage>> {
    let array = messages
        .as_array()
        .ok_or_else(|| invalid("Swift messages must be an array"))?;
    if array.is_empty() || array.len() > crate::swift_chat::MAX_MESSAGES {
        return Err(invalid("Swift message count exceeds supported bounds"));
    }
    let mut parsed = Vec::with_capacity(array.len());
    let mut image_count = 0_usize;
    for (index, message) in array.iter().enumerate() {
        let object = message
            .as_object()
            .ok_or_else(|| invalid("Swift message must be an object"))?;
        if object
            .keys()
            .any(|key| !matches!(key.as_str(), "role" | "content" | "reasoning_content"))
        {
            return Err(invalid("unsupported Swift message field"));
        }
        let role = object
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("Swift message role must be text"))?;
        if !(matches!(role, "user" | "assistant") || role == "system" && index == 0) {
            return Err(invalid("unsupported Swift message role"));
        }
        let reasoning_content = if let Some(reasoning) = object.get("reasoning_content") {
            if role != "assistant" {
                return Err(invalid(
                    "Swift reasoning history requires an assistant turn",
                ));
            }
            Some(
                reasoning
                    .as_str()
                    .ok_or_else(|| invalid("Swift reasoning history must be text"))?
                    .to_owned(),
            )
        } else {
            None
        };
        let (text, parts) = match object.get("content") {
            Some(Value::String(text)) => (Some(text.clone()), Vec::new()),
            Some(Value::Array(raw_parts)) => {
                if role != "user" {
                    return Err(invalid(
                        "Swift multimodal content parts are only allowed on user turns",
                    ));
                }
                if raw_parts.is_empty() || raw_parts.len() > MAX_CONTENT_PARTS {
                    return Err(invalid(
                        "Swift multimodal content part count exceeds supported bounds",
                    ));
                }
                let mut parts = Vec::with_capacity(raw_parts.len());
                for part in raw_parts {
                    parts.push(parse_multimodal_part(part, &mut image_count)?);
                }
                (None, parts)
            }
            _ => return Err(invalid("Swift message content must be text or part array")),
        };
        parsed.push(ParsedMessage {
            role: role.to_owned(),
            text,
            parts,
            reasoning_content,
        });
    }
    Ok(parsed)
}

fn encode_parsed_images(
    messages: &[ParsedMessage],
    vision: &SwiftVisionEncoder,
    max_image_tokens: usize,
) -> Result<Vec<EncodedImage>> {
    let mut encoded = Vec::new();
    for message in messages {
        for part in &message.parts {
            if let ParsedPart::ImageUrl { url, .. } = part {
                encoded.push(vision.encode_data_url(url, max_image_tokens)?);
            }
        }
    }
    Ok(encoded)
}

fn emit_decoded_token(
    vocabulary: &Vocabulary,
    display: &mut Utf8Output,
    output_bytes: &mut usize,
    token: u32,
    emit: &mut impl FnMut(TextEvent<'_>) -> Result<()>,
) -> Result<()> {
    if vocabulary.is_control(token)? {
        let pending = display.flush_incomplete();
        if !pending.is_empty() {
            charge_output(output_bytes, pending.len())?;
            emit(TextEvent::Delta(pending.as_ref()))?;
        }
    }
    crate::text_output::push_with(display, vocabulary.piece(token)?, |piece| {
        let text = piece.as_ref();
        charge_output(output_bytes, text.len())?;
        if !text.is_empty() {
            emit(TextEvent::Delta(text))?;
        }
        Ok(())
    })
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

fn check_request(cancellation: &Cancellation, deadline: Option<Instant>) -> Result<()> {
    if cancellation.is_cancelled() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Interrupted,
            "Swift request cancelled",
        )
        .into());
    }
    if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
        return Err(
            std::io::Error::new(std::io::ErrorKind::TimedOut, "Swift request expired").into(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        CONTEXT_TOKEN_MEMORY_BYTES, CpuPrefill, CpuSpeculation, MAX_CONTEXT_TOKENS,
        NON_CONTEXT_MEMORY_BYTES, PreparedChat, STATE_MEMORY_BYTES, SwiftExecutionMemoryLedger,
        SwiftSession, TextEvent, context_token_memory_bytes, emit_decoded_token,
        kv_token_memory_bytes, max_context_tokens, selected_state_budget, state_limit_bytes,
        validate_adapter_scale, validate_context_capacity, validate_context_capacity_with_kv,
        validate_execution_capacity, validate_execution_capacity_with_kv,
        validate_prefill_capacity, validate_prompt_owner,
    };
    use crate::{
        Result,
        graph::{KvCachePolicy, MAX_STATE_BYTES},
        kernels::{MatvecBackend, RowParallelism},
        model::swift::Representation,
        text_output::Utf8Output,
        vocabulary::Vocabulary,
    };
    use bonsai_gguf::Metadata;
    use std::{path::Path, sync::Arc};

    #[test]
    fn cpu_prefill_charges_full_reserve_and_preserves_boundaries() -> Result<()> {
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
    fn cpu_speculation_charges_logits_and_second_prefix_before_artifact_io() -> Result<()> {
        let context = 8_192;
        let tile = CpuPrefill::new(8, 64 * 1024 * 1024)?;
        let scratch = 16 * 1024 * 1024;
        let reserve = STATE_MEMORY_BYTES + 4096;
        let base = NON_CONTEXT_MEMORY_BYTES + u64::try_from(context)? * CONTEXT_TOKEN_MEMORY_BYTES;
        for tokens in [1, 4, 8] {
            let selection = CpuSpeculation::new(tokens, scratch, reserve)?;
            assert_eq!(selection.tokens(), tokens);
            assert_eq!(selection.verification_scratch_bytes(), scratch);
            assert_eq!(selection.checkpoint_bytes(), reserve);
            let required = base + u64::try_from(tile.scratch_bytes() + scratch + reserve)?;
            validate_execution_capacity(context, required, Some(tile), Some(selection))?;
            assert!(
                validate_execution_capacity(context, required - 1, Some(tile), Some(selection))
                    .is_err()
            );
            assert!(validate_execution_capacity(context, required, None, Some(selection)).is_err());
        }
        Ok(())
    }

    #[test]
    fn cpu_speculation_rejects_invalid_profile_before_artifact_io() -> Result<()> {
        let context = 8_192;
        let tile = CpuPrefill::new(8, 64 * 1024 * 1024)?;
        let scratch = 16 * 1024 * 1024;
        let reserve = STATE_MEMORY_BYTES + 4096;
        for tokens in [0, 9, usize::MAX] {
            assert!(CpuSpeculation::new(tokens, scratch, reserve).is_err());
        }
        for (scratch, checkpoint) in [(0, reserve), (scratch, 0), (usize::MAX, 1)] {
            assert!(CpuSpeculation::new(4, scratch, checkpoint).is_err());
        }
        let undersized = CpuSpeculation::new(4, scratch, STATE_MEMORY_BYTES - 1)?;
        let selected = CpuSpeculation::new(4, scratch, reserve)?;
        for (prefill, speculation) in [(None, selected), (Some(tile), undersized)] {
            let error = SwiftSession::load_profile_with_execution(
                Path::new("THIS_MODEL_MUST_NOT_BE_OPENED.gguf"),
                Representation::Pq2,
                None,
                MatvecBackend::Scalar,
                42,
                RowParallelism::default(),
                KvCachePolicy::Rotated4,
                context,
                prefill,
                Some(speculation),
            )
            .err()
            .ok_or("invalid execution selection reached artifact loading")?;
            assert_eq!(
                error.to_string(),
                "CPU speculation requires tiles and a complete prefix reserve"
            );
        }
        assert!(validate_execution_capacity(32_768, u64::MAX, Some(tile), Some(selected)).is_err());
        let excessive = CpuSpeculation::new(4, 1, usize::MAX - 1)?;
        assert!(
            validate_execution_capacity(context, u64::MAX, Some(tile), Some(excessive)).is_err()
        );
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
    fn selected_cache_capacity_covers_packed_geometry_and_staged_windows() -> Result<()> {
        for (cache, ceiling) in [
            (KvCachePolicy::F16, 62_720),
            (KvCachePolicy::Rotated4, 262_144),
            (KvCachePolicy::Rotated4FittedValues, 241_664),
            (KvCachePolicy::Rotated4Keys3Values, 262_144),
        ] {
            // The graph codec independently computes packed records and scales.
            assert_eq!(
                kv_token_memory_bytes(cache),
                cache.layer_payload_bytes(256, 4, 1)? * 16
            );
            assert_eq!(max_context_tokens(cache), ceiling);
            for context in [4_352, 8_192, ceiling] {
                let memory = NON_CONTEXT_MEMORY_BYTES
                    + u64::try_from(context)? * context_token_memory_bytes(cache);
                validate_context_capacity_with_kv(context, memory, cache)?;
                assert!(validate_context_capacity_with_kv(context, memory - 1, cache).is_err());
                let state = selected_state_budget(context, cache)?;
                assert!(state >= STATE_MEMORY_BYTES);
                assert!(state <= state_limit_bytes(cache));
            }
            assert!(validate_context_capacity_with_kv(0, u64::MAX, cache).is_err());
            assert!(validate_context_capacity_with_kv(ceiling + 1, u64::MAX, cache).is_err());
            assert!(validate_context_capacity_with_kv(usize::MAX, u64::MAX, cache).is_err());
        }
        for context in [65_536, 131_072] {
            let memory = NON_CONTEXT_MEMORY_BYTES + u64::try_from(context)? * 34_016;
            validate_context_capacity_with_kv(context, memory, KvCachePolicy::Rotated4)?;
            assert!(
                validate_context_capacity_with_kv(context, u64::MAX, KvCachePolicy::F16).is_err()
            );
        }
        let packed_8k_memory = NON_CONTEXT_MEMORY_BYTES + 8_192 * 34_016;
        validate_context_capacity_with_kv(8_192, packed_8k_memory, KvCachePolicy::Rotated4)?;
        assert!(
            validate_context_capacity_with_kv(8_192, packed_8k_memory, KvCachePolicy::F16,)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn selected_cache_speculation_reserves_full_scaled_checkpoint() -> Result<()> {
        for context in [131_072, 262_144] {
            let cache = KvCachePolicy::Rotated4;
            let tile = CpuPrefill::new(8, 64 * 1024 * 1024)?;
            let checkpoint = selected_state_budget(context, cache)?;
            let scratch = 16 * 1024 * 1024;
            let selection = CpuSpeculation::new(4, scratch, checkpoint)?;
            let required = NON_CONTEXT_MEMORY_BYTES
                + u64::try_from(context)? * context_token_memory_bytes(cache)
                + u64::try_from(tile.scratch_bytes() + scratch + checkpoint)?;
            validate_execution_capacity_with_kv(
                context,
                required,
                Some(tile),
                Some(selection),
                cache,
            )?;
            assert!(
                validate_execution_capacity_with_kv(
                    context,
                    required - 1,
                    Some(tile),
                    Some(selection),
                    cache,
                )
                .is_err()
            );
            let undersized = CpuSpeculation::new(4, scratch, checkpoint - 1)?;
            assert!(
                validate_execution_capacity_with_kv(
                    context,
                    u64::MAX,
                    Some(tile),
                    Some(undersized),
                    cache,
                )
                .is_err()
            );
            assert!(selected_state_budget(context, cache)? <= state_limit_bytes(cache));
        }
        Ok(())
    }

    #[test]
    fn prepared_chat_seals_session_and_exact_adapter_settings() -> Result<()> {
        let owner = Arc::new(());
        let prompt = PreparedChat {
            owner: owner.clone(),
            adapter_scale: Some(-0.0),
            tokens: vec![1, 2],
            media_chunks: Vec::new(),
            output_tokens: 3,
            enable_thinking: false,
        };
        validate_prompt_owner(&owner, &prompt)?;
        assert!(validate_prompt_owner(&Arc::new(()), &prompt).is_err());
        assert_eq!(prompt.prompt_tokens(), 2);
        assert_eq!(prompt.image_count(), 0);
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

    #[test]
    fn execution_memory_ledger_separates_job_commit_from_external_bytes() -> Result<()> {
        assert_eq!(
            state_limit_bytes(KvCachePolicy::Rotated4FittedValues),
            MAX_STATE_BYTES
        );
        let base = SwiftExecutionMemoryLedger::for_swift_profile(
            8_192,
            KvCachePolicy::Rotated4,
            None,
            None,
            1,
            0,
        )?;
        let external = 2_u64 * 1024 * 1024 * 1024;
        let with_external = SwiftExecutionMemoryLedger::for_swift_profile(
            8_192,
            KvCachePolicy::Rotated4,
            None,
            None,
            1,
            external,
        )?;
        assert_eq!(
            base.job_commit_total_bytes(),
            with_external.job_commit_total_bytes()
        );
        assert_eq!(
            with_external.total_with_external_bytes(),
            base.job_commit_total_bytes()
                .and_then(|bytes| bytes.checked_add(external))
        );

        let tile = CpuPrefill::new(16, 64 * 1024 * 1024)?;
        let spec = CpuSpeculation::new(4, 16 * 1024 * 1024, STATE_MEMORY_BYTES + 4096)?;
        let expanded = SwiftExecutionMemoryLedger::for_swift_profile(
            8_192,
            KvCachePolicy::Rotated4,
            Some(tile),
            Some(spec),
            3,
            external,
        )?;
        assert_eq!(expanded.tile_scratch_bytes, 64 * 1024 * 1024);
        assert_eq!(expanded.verification_logits_bytes, 16 * 1024 * 1024);
        assert!(expanded.checkpoint_recurrent_bytes > base.checkpoint_recurrent_bytes);
        assert!(expanded.checkpoint_kv_bytes > base.checkpoint_kv_bytes);
        assert!(expanded.job_commit_total_bytes() > base.job_commit_total_bytes());
        Ok(())
    }

    #[test]
    fn control_token_boundary_flushes_incomplete_utf8_sequence() -> Result<()> {
        let metadata = Metadata::from([
            ("tokenizer.ggml.model".to_owned(), "gpt2".into()),
            ("tokenizer.ggml.pre".to_owned(), "qwen35".into()),
            (
                "tokenizer.ggml.tokens".to_owned(),
                vec![
                    "<0xF0>",
                    "<0x9F>",
                    "<|im_end|>",
                    "<0xE9>",
                    "<0x9B>",
                    "<0xA8>",
                ]
                .into(),
            ),
            (
                "tokenizer.ggml.token_type".to_owned(),
                vec![6_i32, 6, 3, 6, 6, 6].into(),
            ),
        ]);
        let vocabulary = Vocabulary::from_metadata(&metadata, 6)?;

        let mut bridged = Utf8Output::new();
        let mut bridged_bytes = 0usize;
        let mut bridged_chunks = Vec::new();
        for token in [3_u32, 4, 5] {
            emit_decoded_token(
                &vocabulary,
                &mut bridged,
                &mut bridged_bytes,
                token,
                &mut |event| {
                    if let TextEvent::Delta(chunk) = event {
                        bridged_chunks.push(chunk.to_owned());
                    }
                    Ok(())
                },
            )?;
        }
        assert_eq!(bridged_chunks.concat(), "雨");
        assert_eq!(bridged.pending_bytes(), b"");

        let mut interrupted = Utf8Output::new();
        let mut interrupted_bytes = 0usize;
        let mut chunks = Vec::new();
        for token in [0_u32, 1, 2] {
            emit_decoded_token(
                &vocabulary,
                &mut interrupted,
                &mut interrupted_bytes,
                token,
                &mut |event| {
                    if let TextEvent::Delta(chunk) = event {
                        chunks.push(chunk.to_owned());
                    }
                    Ok(())
                },
            )?;
        }
        assert_eq!(chunks, vec!["\u{fffd}".to_owned(), "<|im_end|>".to_owned()]);
        assert_eq!(interrupted.pending_bytes(), b"");
        assert_eq!(interrupted_bytes, "\u{fffd}<|im_end|>".len());
        Ok(())
    }

    #[test]
    fn vision_token_budget_and_mmproj_reserve_enforce_bounds() -> Result<()> {
        assert!(super::MMPROJ_MEMORY_BYTES >= usize::try_from(crate::model::swift::MMPROJ_BYTES)?);
        assert!(super::validate_vision_token_budget(63).is_err());
        assert!(super::validate_vision_token_budget(4_097).is_err());
        super::validate_vision_token_budget(super::MIN_IMAGE_TOKENS)?;
        super::validate_vision_token_budget(super::DEFAULT_MAX_IMAGE_TOKENS)?;
        super::validate_vision_token_budget(super::MAX_IMAGE_TOKENS)?;
        Ok(())
    }

    #[test]
    fn multimodal_message_parser_enforces_user_roles_parts_and_image_cap() -> Result<()> {
        let valid = serde_json::json!([
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": [
                {"type": "text", "text": "Describe:"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": "high"}}
            ]}
        ]);
        assert!(super::messages_have_array_content(&valid));
        let parsed = super::parse_multimodal_messages(&valid)?;
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[1].parts.len(), 2);

        let assistant_parts = serde_json::json!([
            {"role": "assistant", "content": [{"type": "text", "text": "nope"}]},
            {"role": "user", "content": "ok"}
        ]);
        assert!(super::parse_multimodal_messages(&assistant_parts).is_err());

        let empty_parts = serde_json::json!([
            {"role": "user", "content": []}
        ]);
        assert!(super::parse_multimodal_messages(&empty_parts).is_err());

        let bad_detail = serde_json::json!([
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": "ultra"}}
            ]}
        ]);
        assert!(super::parse_multimodal_messages(&bad_detail).is_err());

        let mut too_many = Vec::new();
        for _ in 0..=super::MAX_PROMPT_IMAGES {
            too_many.push(serde_json::json!({
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,AAAA"}
            }));
        }
        let overflow = serde_json::json!([{"role": "user", "content": too_many}]);
        assert!(super::parse_multimodal_messages(&overflow).is_err());

        // Reject invalid output reservations or impossible multimodal context budgets early.
        assert!(super::validate_multimodal_budget(&parsed, 0, 1024).is_err());
        assert!(
            super::validate_multimodal_budget(&parsed, super::swift::MAX_OUTPUT_TOKENS + 1, 65536)
                .is_err()
        );
        assert!(super::validate_multimodal_budget(&parsed, 1021, 1024).is_err());
        super::validate_multimodal_budget(&parsed, 64, 1024)?;
        Ok(())
    }

    #[test]
    fn chunkwise_gdn_requires_fp16_without_speculation_before_artifact_io() -> Result<()> {
        let context = 8_192;
        let reserve = 64 * 1024 * 1024;
        let sequential = CpuPrefill::new(16, reserve)?;
        assert!(!sequential.chunkwise_gdn());
        let chunkwise = sequential.with_chunkwise_gdn();
        assert!(chunkwise.chunkwise_gdn());
        assert_eq!(chunkwise.tokens(), 16);
        assert_eq!(chunkwise.scratch_bytes(), reserve);

        let f16_required = NON_CONTEXT_MEMORY_BYTES
            + u64::try_from(context)? * context_token_memory_bytes(KvCachePolicy::F16)
            + u64::try_from(reserve)?;
        super::validate_prefill_capacity_with_kv(
            context,
            f16_required,
            Some(chunkwise),
            KvCachePolicy::F16,
        )?;
        validate_execution_capacity_with_kv(
            context,
            f16_required,
            Some(chunkwise),
            None,
            KvCachePolicy::F16,
        )?;

        for rotated in [
            KvCachePolicy::Rotated4,
            KvCachePolicy::Rotated4FittedValues,
            KvCachePolicy::Rotated4Keys3Values,
        ] {
            assert!(
                super::validate_prefill_capacity_with_kv(
                    context,
                    u64::MAX,
                    Some(chunkwise),
                    rotated
                )
                .is_err()
            );
            assert!(
                validate_execution_capacity_with_kv(
                    context,
                    u64::MAX,
                    Some(chunkwise),
                    None,
                    rotated
                )
                .is_err()
            );
            let error = SwiftSession::load_profile_with_execution(
                Path::new("THIS_MODEL_MUST_NOT_BE_OPENED.gguf"),
                Representation::Pq2,
                None,
                MatvecBackend::Scalar,
                42,
                RowParallelism::default(),
                rotated,
                context,
                Some(chunkwise),
                None,
            )
            .err()
            .ok_or("non-FP16 chunkwise GDN reached artifact loading")?;
            assert_eq!(
                error.to_string(),
                "chunkwise GDN prefill requires the separate FP16 KV cache profile"
            );
        }

        let speculation = CpuSpeculation::new(4, 16 * 1024 * 1024, STATE_MEMORY_BYTES)?;
        assert!(
            validate_execution_capacity_with_kv(
                context,
                u64::MAX,
                Some(chunkwise),
                Some(speculation),
                KvCachePolicy::F16,
            )
            .is_err()
        );
        let error = SwiftSession::load_profile_with_execution(
            Path::new("THIS_MODEL_MUST_NOT_BE_OPENED.gguf"),
            Representation::Pq2,
            None,
            MatvecBackend::Scalar,
            42,
            RowParallelism::default(),
            KvCachePolicy::F16,
            context,
            Some(chunkwise),
            Some(speculation),
        )
        .err()
        .ok_or("speculative chunkwise GDN reached artifact loading")?;
        assert_eq!(
            error.to_string(),
            "chunkwise GDN prefill is incompatible with CPU speculation"
        );
        Ok(())
    }

    #[test]
    fn branch_checkpoints_charge_explicit_state_budgets_and_reject_invalid_bounds() -> Result<()> {
        use super::{
            BranchCheckpoints, MAX_REUSABLE_CHECKPOINTS, validate_execution_capacity_with_reuse,
        };
        use crate::runtime::PromptReuse;

        assert_eq!(MAX_REUSABLE_CHECKPOINTS, 8);
        assert_eq!(BranchCheckpoints::default().checkpoints(), 1);
        assert_eq!(BranchCheckpoints::default().checkpoint_bytes(), None);
        for invalid_slots in [0, MAX_REUSABLE_CHECKPOINTS + 1, usize::MAX] {
            assert!(BranchCheckpoints::new(invalid_slots).is_err());
            assert!(BranchCheckpoints::with_checkpoint_bytes(invalid_slots, 1).is_err());
        }
        assert!(BranchCheckpoints::with_checkpoint_bytes(1, 0).is_err());

        let context = 8_192;
        let cache = KvCachePolicy::Rotated4;
        let base = NON_CONTEXT_MEMORY_BYTES
            + u64::try_from(context)? * context_token_memory_bytes(cache);
        let state_budget = u64::try_from(selected_state_budget(context, cache)?)?;

        // Default checkpoints=1 with Disabled or Latest adds no extra process reserve.
        for reuse in [PromptReuse::Disabled, PromptReuse::Latest] {
            validate_execution_capacity_with_reuse(
                context,
                base,
                None,
                None,
                cache,
                reuse,
                BranchCheckpoints::default(),
            )?;
        }

        // Disabled rejects slots > 1 or explicit checkpoint reserves before artifact I/O.
        for bad in [
            BranchCheckpoints::new(2)?,
            BranchCheckpoints::with_checkpoint_bytes(1, selected_state_budget(context, cache)?)?,
        ] {
            assert!(
                validate_execution_capacity_with_reuse(
                    context,
                    u64::MAX,
                    None,
                    None,
                    cache,
                    PromptReuse::Disabled,
                    bad,
                )
                .is_err()
            );
            assert!(
                SwiftSession::load_profile_with_execution_and_reuse(
                    Path::new("THIS_MODEL_MUST_NOT_BE_OPENED.gguf"),
                    Representation::Pq2,
                    None,
                    MatvecBackend::Scalar,
                    42,
                    RowParallelism::default(),
                    cache,
                    context,
                    None,
                    None,
                    PromptReuse::Disabled,
                    bad,
                )
                .is_err()
            );
        }

        // PromptEndAndLatest with slots=1 charges 1 * selected_state_budget.
        let prompt_end_required = base + state_budget;
        validate_execution_capacity_with_reuse(
            context,
            prompt_end_required,
            None,
            None,
            cache,
            PromptReuse::PromptEndAndLatest,
            BranchCheckpoints::default(),
        )?;
        assert!(
            validate_execution_capacity_with_reuse(
                context,
                prompt_end_required - 1,
                None,
                None,
                cache,
                PromptReuse::PromptEndAndLatest,
                BranchCheckpoints::default(),
            )
            .is_err()
        );

        // Multi-branch slots (2..=8) charge slots * selected_state_budget for both reuse modes.
        for slots in 2..=MAX_REUSABLE_CHECKPOINTS {
            let branches = BranchCheckpoints::new(slots)?;
            let required = base + u64::try_from(slots)? * state_budget;
            for reuse in [PromptReuse::Latest, PromptReuse::PromptEndAndLatest] {
                validate_execution_capacity_with_reuse(
                    context, required, None, None, cache, reuse, branches,
                )?;
                assert!(
                    validate_execution_capacity_with_reuse(
                        context,
                        required - 1,
                        None,
                        None,
                        cache,
                        reuse,
                        branches,
                    )
                    .is_err()
                );
            }
        }

        // Explicit byte reserve rejects undersized values and charges the full caller reserve.
        let min_four = 4 * selected_state_budget(context, cache)?;
        let undersized = BranchCheckpoints::with_checkpoint_bytes(4, min_four - 1)?;
        assert!(
            validate_execution_capacity_with_reuse(
                context,
                u64::MAX,
                None,
                None,
                cache,
                PromptReuse::Latest,
                undersized,
            )
            .is_err()
        );
        let extra = min_four + 64 * 1024 * 1024;
        let explicit = BranchCheckpoints::with_checkpoint_bytes(4, extra)?;
        let required_explicit = base + u64::try_from(extra)?;
        validate_execution_capacity_with_reuse(
            context,
            required_explicit,
            None,
            None,
            cache,
            PromptReuse::Latest,
            explicit,
        )?;
        assert!(
            validate_execution_capacity_with_reuse(
                context,
                required_explicit - 1,
                None,
                None,
                cache,
                PromptReuse::Latest,
                explicit,
            )
            .is_err()
        );
        Ok(())
    }
}
