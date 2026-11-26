//! Batched multi-layer GPU graph executor and TDR chunk policy for Qualcomm Adreno Vulkan.
//!
//! Keeps token activations, residual streams, `F32` recurrent GDN/conv state, and `FP16`/`F32`
//! KV caches resident on the GPU across complete hybrid `Qwen35` recurrent and full-attention
//! layers, recording whole layers and multi-layer chunks into [`CommandBatchRecorder`] under an
//! explicit [`TdrChunkPolicy`] instead of synchronizing after every projection or operator.
#![forbid(unsafe_code)]

use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
    path::{Path, PathBuf},
};

use bonsai_native_hash::Hasher;
use bonsai_prism_runtime::{
    Decoder, MAX_PREFILL_BLOCK_TOKENS, Result as RuntimeResult,
    kernels::{row_into, validate_tensor},
    model::{
        Model, Tensor,
        lora::{MultiRankAdapter, MultiRankPolicy},
        swift::{BONSAI_2_5_PQ2_SHA256, BONSAI_2_5_PTQ1_SHA256},
    },
    sampling::FiniteLogits,
};
use bonsai_ternary::half_to_float;

use crate::{
    AdrenoAttentionPath, AttentionDescriptor, AttentionStrategy, Error, ProjectionKernelPath,
    attention_cache_key,
    device::{
        AdrenoVulkanDevice, BufferId, BufferSpan, CommandBatchRecorder,
        MAX_DESCRIPTOR_SETS_PER_SUBMIT, PipelineId, WeightArena,
    },
    executor::{
        BatchTokenPath, KIND_PQ2, KIND_PTQ1, KIND_Q8_0, MIN_SCRATCH_BYTES,
        PreparedProjectionShaders, PreparedShaderSpec, ProjectionPipelines, Q8_0_WORDS_PER_BLOCK,
        Q8K_4PLANE_WORDS_PER_HALF, attention_push_constants, checked_grid, ensure_finite_f32,
        f32_slice_bytes, hadamard_norm_bits, repack_coalesced_into, u32_slice_bytes,
        validate_matrix, validate_weight_scales,
    },
    hash_bytes, hash_str, hash_u32, hash_u64, hash_usize, select_attention_plan, spirv,
};

const WORKGROUP_X: u32 = 64;
const MAX_GRAPH_LAYERS: usize = 512;
const MAX_GRAPH_WIDTH: usize = 1_048_576;
const MIN_NORM_OR_BIAS_BYTES: u64 = 16;

/// Explicit Windows WDDM Timeout Detection and Recovery (`TDR`) chunking policy.
///
/// Controls how many transformer layers are recorded into each batched Vulkan command buffer
/// submission (`layers_per_submit`) and the maximum compute dispatches allowed per individual
/// queue submit (`max_stages_per_submit`), trading per-submit host latency against GPU preemption
/// margins under Windows' 2-second WDDM timeout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TdrChunkPolicy {
    /// Number of consecutive transformer layers recorded per GPU submit (`1..=64`, default `8`).
    pub layers_per_submit: usize,
    /// Maximum number of compute stages packed into any single GPU submit (`1..=4096`).
    pub max_stages_per_submit: usize,
}

impl TdrChunkPolicy {
    /// Default chunk size (`8` layers/submit reduces a 64-layer token from ~256+ submits to `<= 8`).
    pub const DEFAULT_LAYERS_PER_SUBMIT: usize = 8;
    /// Upper bound on layers bundled in one chunk submit.
    pub const MAX_LAYERS_PER_SUBMIT: usize = 64;
    /// Default stage cap per command-buffer submit safely below pool/descriptor limits.
    pub const DEFAULT_MAX_STAGES_PER_SUBMIT: usize = 2048;

    /// Constructs and validates an explicit [`TdrChunkPolicy`].
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] if `layers_per_submit` or `max_stages_per_submit` is zero
    /// or exceeds crate bounds.
    pub fn new(layers_per_submit: usize, max_stages_per_submit: usize) -> Result<Self, Error> {
        let policy = Self {
            layers_per_submit,
            max_stages_per_submit,
        };
        policy.validate()?;
        Ok(policy)
    }

    /// Validates that chunking parameters are non-zero and bounded.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] on out-of-range `layers_per_submit` or
    /// `max_stages_per_submit`.
    pub fn validate(&self) -> Result<(), Error> {
        if self.layers_per_submit == 0 || self.layers_per_submit > Self::MAX_LAYERS_PER_SUBMIT {
            return Err(Error::InvalidInput(format!(
                "layers_per_submit must be in 1..={}, got {}",
                Self::MAX_LAYERS_PER_SUBMIT,
                self.layers_per_submit
            )));
        }
        if self.max_stages_per_submit == 0
            || self.max_stages_per_submit > MAX_DESCRIPTOR_SETS_PER_SUBMIT
        {
            return Err(Error::InvalidInput(format!(
                "max_stages_per_submit must be in 1..={MAX_DESCRIPTOR_SETS_PER_SUBMIT}, got {}",
                self.max_stages_per_submit
            )));
        }
        Ok(())
    }

    /// Partitions `0..total_layers` into contiguous submit chunks bounded by `layers_per_submit`.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] if `self` fails validation.
    pub fn layer_chunk_ranges(&self, total_layers: usize) -> Result<Vec<Range<usize>>, Error> {
        self.layer_chunk_ranges_with_limit(total_layers, self.layers_per_submit)
    }

    /// Returns the effective layers per submit for a batch of `tokens` tokens.
    ///
    /// Single-token decode (`tokens == 1`) under the default policy raises chunking to
    /// [`Self::MAX_LAYERS_PER_SUBMIT`] (`64`) to minimize per-token `vkQueueSubmit` overhead
    /// while remaining bounded by `max_stages_per_submit` and WDDM TDR margins. Explicit
    /// non-default policies (such as unit-test chunk overrides) are preserved verbatim.
    #[must_use]
    pub const fn effective_layers_per_submit(&self, tokens: usize) -> usize {
        if tokens == 1 && self.layers_per_submit == Self::DEFAULT_LAYERS_PER_SUBMIT {
            Self::MAX_LAYERS_PER_SUBMIT
        } else {
            self.layers_per_submit
        }
    }

    fn layer_chunk_ranges_for_tokens(
        &self,
        total_layers: usize,
        tokens: usize,
    ) -> Result<Vec<Range<usize>>, Error> {
        self.layer_chunk_ranges_with_limit(total_layers, self.effective_layers_per_submit(tokens))
    }

    fn layer_chunk_ranges_with_limit(
        &self,
        total_layers: usize,
        per_submit: usize,
    ) -> Result<Vec<Range<usize>>, Error> {
        self.validate()?;
        let mut ranges = Vec::with_capacity(total_layers.div_ceil(per_submit));
        let mut start = 0_usize;
        while start < total_layers {
            let end = start.saturating_add(per_submit).min(total_layers);
            ranges.push(start..end);
            start = end;
        }
        Ok(ranges)
    }

    /// Returns the expected number of layer-chunk submissions for `total_layers` assuming each
    /// chunk fits within `max_stages_per_submit`.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] if `self` fails validation.
    pub fn expected_submits_for_layers(&self, total_layers: usize) -> Result<usize, Error> {
        Ok(self.layer_chunk_ranges(total_layers)?.len())
    }
}

impl Default for TdrChunkPolicy {
    fn default() -> Self {
        Self {
            layers_per_submit: Self::DEFAULT_LAYERS_PER_SUBMIT,
            max_stages_per_submit: Self::DEFAULT_MAX_STAGES_PER_SUBMIT,
        }
    }
}

/// GPU KV-cache storage representation for full-attention layers.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum VulkanKvPrecision {
    /// Packed IEEE binary16 words (`2 × f16` per `u32`), consumed directly by `#40` attention via `UnpackHalf2x16`.
    #[default]
    Fp16,
    /// Uncompressed `F32` storage holding binary16-rounded values directly consumable by `#40`.
    Fp32,
}

impl VulkanKvPrecision {
    /// Returns the canonical contract identifier of this KV storage format.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::Fp32 => "fp32",
        }
    }
}

/// Multi-rank `LoRA` adapter input (`F16`/`F32`) for [`VulkanBatchedGraphExecutor`].
///
/// Accepts a GGUF adapter [`Model`] validated through [`MultiRankAdapter::from_model`] under
/// `policy` and uploads all `LoRA` `A` and `B` matrices into GPU resident memory.
#[derive(Debug)]
pub struct VulkanMultiRankAdapterSpec<'a> {
    /// Loaded GGUF adapter [`Model`] conforming to [`MultiRankAdapter`].
    pub adapter: &'a Model,
    /// Target admission policy enforced by [`MultiRankAdapter::from_model`].
    pub policy: MultiRankPolicy,
    /// Initial adapter scale multiplier (`0.0..=2.0`, finite; `0.0` bypasses execution).
    pub scale: f32,
}

/// Optional GPU-resident multi-rank `LoRA` adapters attached at executor construction.
#[derive(Debug, Default)]
pub struct VulkanBatchedAdapters<'a> {
    /// Optional stackable multi-rank `LoRA` adapter (`F16`/`F32`).
    pub multi_rank: Option<VulkanMultiRankAdapterSpec<'a>>,
}

/// Configuration for constructing [`VulkanBatchedGraphExecutor`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VulkanBatchedConfig {
    /// Chunking policy keeping each GPU submit comfortably inside Windows WDDM TDR bounds.
    pub tdr_policy: TdrChunkPolicy,
    /// Full-attention GPU KV-cache storage precision.
    pub kv_precision: VulkanKvPrecision,
    /// Attention execution strategy passed to the `#40` tile planner.
    pub attention_strategy: AttentionStrategy,
    /// Optional forced projection kernel path (`None` chooses device preferred path).
    pub kernel_path: Option<ProjectionKernelPath>,
    /// Optional override context length (`None` uses `min(train_context, 8192)`).
    pub context_limit: Option<usize>,
    /// Optional active layer count override (`None` loads all non-MTP model layers).
    pub max_layer_count: Option<usize>,
    /// Advisory device-memory budget for GPU scratch buffers in bytes.
    pub scratch_budget_bytes: u64,
    /// Optional explicit override for Prism `PinnedSwift` vs `Source` integer rounding/FMA contract.
    /// When `None`, auto-detects from verified Swift model SHA-256 identities.
    pub pinned_arithmetic: Option<bool>,
    /// Maximum prompt tokens evaluated together per batched GPU prefill tile (`1..=256`, default `32`).
    ///
    /// Setting `1` forces single-token evaluation inside [`Decoder::ingest_block`]. Larger values
    /// amortize weight reads across tokens while remaining bounded by `scratch_budget_bytes` and
    /// device `maxStorageBufferRange` limits.
    pub prefill_tile_tokens: usize,
    /// Optional disk file path for persisting driver `VkPipelineCache` binaries across runs.
    ///
    /// When `Some`, [`VulkanBatchedGraphExecutor::open_with_adapters`] attempts to seed the
    /// Vulkan pipeline cache from this file and exports updated cache data after compilation.
    pub pipeline_cache_path: Option<PathBuf>,
}

impl Default for VulkanBatchedConfig {
    fn default() -> Self {
        Self {
            tdr_policy: TdrChunkPolicy::default(),
            kv_precision: VulkanKvPrecision::Fp16,
            attention_strategy: AttentionStrategy::Auto,
            kernel_path: None,
            context_limit: None,
            max_layer_count: None,
            scratch_budget_bytes: 256 * 1024 * 1024,
            pinned_arithmetic: None,
            prefill_tile_tokens: 32,
            pipeline_cache_path: None,
        }
    }
}

/// Compiled SPIR-V 1.6 compute pipelines used by [`VulkanBatchedGraphExecutor`].
///
/// Bundles `#39` projection pipelines, `#40` `F32` and direct packed-`FP16` attention pipelines,
/// `LoRA` accumulation pipelines, and whole-layer intra-GPU transformation/state pipelines alongside
/// a deterministic digest over all emitted SPIR-V words.
#[derive(Debug)]
pub struct VulkanLayerPipelines {
    proj: ProjectionPipelines,
    attn_qk_scores: PipelineId,
    attn_qk_scores_f16kv: PipelineId,
    attn_softmax_f16: PipelineId,
    attn_pv_accum: PipelineId,
    attn_pv_accum_f16kv: PipelineId,
    attn_fused_f16prob: PipelineId,
    attn_fused_f16prob_f16kv: PipelineId,
    attn_fused_gqa_prefill_f16kv: PipelineId,
    rms_norm: PipelineId,
    add_rms_norm: PipelineId,
    rope: PipelineId,
    rope_batched: PipelineId,
    mul_sigmoid: PipelineId,
    conv1d_silu: PipelineId,
    conv1d_silu_offset: PipelineId,
    conv1d_silu_batched_inplace: PipelineId,
    l2_norm: PipelineId,
    l2_norm_batched_strided: PipelineId,
    gdn_step: PipelineId,
    gdn_step_batched_token: PipelineId,
    gdn_scan_batched: PipelineId,
    gdn_gates: PipelineId,
    gdn_gates_batched: PipelineId,
    deinterleave_q_gate: PipelineId,
    deinterleave_q_gate_batched: PipelineId,
    round_f16: PipelineId,
    kv_append_f32: PipelineId,
    kv_append_batched_f32: PipelineId,
    kv_append_f16: PipelineId,
    kv_append_batched_f16: PipelineId,
    quant_q8_0_source: PipelineId,
    quant_q8_0_source_batched: PipelineId,
    quant_q8_0_pinned: PipelineId,
    quant_q8_0_pinned_batched: PipelineId,
    quant_q8_0_pinned_native: PipelineId,
    quant_q8_0_pinned_native_batched: PipelineId,
    fused_hadamard1024_q8k_pinned: PipelineId,
    fused_hadamard1024_q8k_unpinned: PipelineId,
    fused_hadamard1024_q8_0_source: PipelineId,
    fused_hadamard1024_q8_0_pinned: PipelineId,
    fused_hadamard1024_q8_0_pinned_native: PipelineId,
    hadamard_batched: PipelineId,
    f32_matvec: PipelineId,
    f32_gemm_batched: PipelineId,
    lora_b_accum: PipelineId,
    lora_b_accum_batched: PipelineId,
    mul_f32_elemwise: PipelineId,
    mul_f32_elemwise_batched: PipelineId,
    permute_ssm_heads: PipelineId,
    permute_ssm_heads_batched: PipelineId,
    copy_slice: PipelineId,
    copy_slice_offset: PipelineId,
    use_sdot: bool,
    attn_qk_digest: String,
    attn_qk_f16kv_digest: String,
    attn_softmax_digest: String,
    attn_pv_digest: String,
    attn_pv_f16kv_digest: String,
    attn_fused_digest: String,
    attn_fused_f16kv_digest: String,
    attn_fused_gqa_prefill_f16kv_digest: String,
    shader_digests_sha256: String,
}

struct PreparedLayerShaders {
    use_sdot: bool,
    use_wave64: bool,
    proj: PreparedProjectionShaders,
    modules: [PreparedShaderSpec; 52],
    shader_digests_sha256: String,
}

impl VulkanLayerPipelines {
    /// Emits and digests every layer, projection, `LoRA`, and attention SPIR-V 1.6 module without
    /// yet opening a Vulkan device or invoking QGLC compilation.
    fn prepare(use_sdot: bool, use_wave64_shuffle: bool) -> Result<PreparedLayerShaders, Error> {
        Self::prepare_with_subgroup(use_sdot, use_wave64_shuffle)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Emits and digests all whole-layer, projection, LoRA, batched prefill, and F32/packed-FP16 attention SPIR-V modules sequentially."
    )]
    fn prepare_with_subgroup(
        use_sdot: bool,
        use_wave64_shuffle: bool,
    ) -> Result<PreparedLayerShaders, Error> {
        let proj = ProjectionPipelines::prepare_with_subgroup(use_sdot, use_wave64_shuffle)?;
        let mut hasher = Hasher::new().map_err(|err| Error::InvalidInput(err.to_string()))?;
        hash_bytes(&mut hasher, b"vulkan-layer-pipelines-v6:")?;
        hash_u32(&mut hasher, u32::from(use_sdot))?;
        hash_u32(&mut hasher, u32::from(use_wave64_shuffle))?;
        hash_bytes(&mut hasher, b"proj_bundle:")?;
        hash_bytes(&mut hasher, proj.digest_sha256.as_bytes())?;

        let mut make_spec = |tag: &'static [u8],
                             words: Vec<u32>,
                             num_buffers: usize,
                             push_u32s: usize|
         -> Result<PreparedShaderSpec, Error> {
            let digest = spirv::shader_digest(&words)?;
            hash_bytes(&mut hasher, tag)?;
            hash_bytes(&mut hasher, digest.as_bytes())?;
            Ok(PreparedShaderSpec {
                tag,
                words,
                num_buffers,
                push_u32s,
                digest,
            })
        };

        let modules = [
            make_spec(b"attn_qk_scores", spirv::attn_qk_scores(), 3, 8)?,
            make_spec(b"attn_qk_scores_f16kv", spirv::attn_qk_scores_f16kv(), 3, 8)?,
            make_spec(
                b"attn_softmax_f16",
                spirv::attn_softmax_f16_with_subgroup(use_wave64_shuffle),
                2,
                8,
            )?,
            make_spec(b"attn_pv_accum", spirv::attn_pv_accum(), 3, 8)?,
            make_spec(b"attn_pv_accum_f16kv", spirv::attn_pv_accum_f16kv(), 3, 8)?,
            make_spec(
                b"attn_fused_f16prob",
                spirv::attn_fused_f16prob_with_subgroup(use_wave64_shuffle),
                4,
                8,
            )?,
            make_spec(
                b"attn_fused_f16prob_f16kv",
                spirv::attn_fused_f16prob_f16kv_with_subgroup(use_wave64_shuffle),
                4,
                8,
            )?,
            make_spec(
                b"attn_fused_gqa_prefill_f16kv",
                spirv::attn_fused_gqa_prefill_f16kv_with_subgroup(use_wave64_shuffle),
                4,
                8,
            )?,
            make_spec(
                b"rms_norm_f32",
                spirv::rms_norm_f32_with_subgroup(use_wave64_shuffle),
                3,
                2,
            )?,
            make_spec(
                b"add_rms_norm_f32",
                spirv::add_rms_norm_f32_with_subgroup(use_wave64_shuffle),
                4,
                2,
            )?,
            make_spec(b"rope_f32", spirv::rope_f32(), 2, 4)?,
            make_spec(b"rope_batched_f32", spirv::rope_batched_f32(), 2, 5)?,
            make_spec(b"mul_sigmoid_f32", spirv::mul_sigmoid_f32(), 2, 1)?,
            make_spec(b"conv1d_silu_f32", spirv::conv1d_silu_f32(), 4, 2)?,
            make_spec(
                b"conv1d_silu_offset_f32",
                spirv::conv1d_silu_offset_f32(),
                4,
                3,
            )?,
            make_spec(
                b"conv1d_silu_batched_inplace_f32",
                spirv::conv1d_silu_batched_inplace_f32(),
                3,
                3,
            )?,
            make_spec(
                b"l2_norm_f32",
                spirv::l2_norm_f32_with_subgroup(use_wave64_shuffle),
                1,
                3,
            )?,
            make_spec(
                b"l2_norm_batched_strided_f32",
                spirv::l2_norm_batched_strided_f32_with_subgroup(use_wave64_shuffle),
                1,
                5,
            )?,
            make_spec(
                b"gdn_step_f32",
                spirv::gdn_step_f32_with_subgroup(use_wave64_shuffle),
                7,
                4,
            )?,
            make_spec(
                b"gdn_step_batched_token_f32",
                spirv::gdn_step_batched_token_f32_with_subgroup(use_wave64_shuffle),
                5,
                5,
            )?,
            make_spec(
                b"gdn_scan_batched_f32",
                spirv::gdn_scan_batched_f32_with_subgroup(use_wave64_shuffle),
                5,
                5,
            )?,
            make_spec(b"gdn_gates_f32", spirv::gdn_gates_f32(), 4, 1)?,
            make_spec(
                b"gdn_gates_batched_f32",
                spirv::gdn_gates_batched_f32(),
                4,
                2,
            )?,
            make_spec(
                b"deinterleave_q_gate_f32",
                spirv::deinterleave_q_gate_f32(),
                3,
                2,
            )?,
            make_spec(
                b"deinterleave_q_gate_batched_f32",
                spirv::deinterleave_q_gate_batched_f32(),
                3,
                3,
            )?,
            make_spec(b"round_f16_f32", spirv::round_f16_f32(), 1, 1)?,
            make_spec(b"kv_append_f32", spirv::kv_append_f32(), 2, 2)?,
            make_spec(
                b"kv_append_batched_f32",
                spirv::kv_append_batched_f32(),
                2,
                3,
            )?,
            make_spec(b"kv_append_f16", spirv::kv_append_f16(), 2, 2)?,
            make_spec(
                b"kv_append_batched_f16",
                spirv::kv_append_batched_f16(),
                2,
                3,
            )?,
            make_spec(
                b"quant_q8_0_source",
                spirv::quantize_f32_to_q8_0(false, false),
                2,
                1,
            )?,
            make_spec(
                b"quant_q8_0_source_batched",
                spirv::quantize_f32_to_q8_0_batched(false, false),
                2,
                2,
            )?,
            make_spec(
                b"quant_q8_0_pinned",
                spirv::quantize_f32_to_q8_0(true, false),
                2,
                1,
            )?,
            make_spec(
                b"quant_q8_0_pinned_batched",
                spirv::quantize_f32_to_q8_0_batched(true, false),
                2,
                2,
            )?,
            make_spec(
                b"quant_q8_0_pinned_native",
                spirv::quantize_f32_to_q8_0(true, true),
                2,
                1,
            )?,
            make_spec(
                b"quant_q8_0_pinned_native_batched",
                spirv::quantize_f32_to_q8_0_batched(true, true),
                2,
                2,
            )?,
            make_spec(
                b"fused_hadamard1024_q8k_pinned",
                spirv::fused_basis_hadamard1024_q8k_4plane_with_subgroup(true, use_wave64_shuffle),
                3,
                7,
            )?,
            make_spec(
                b"fused_hadamard1024_q8k_unpinned",
                spirv::fused_basis_hadamard1024_q8k_4plane_with_subgroup(false, use_wave64_shuffle),
                3,
                7,
            )?,
            make_spec(
                b"fused_hadamard1024_q8_0_source",
                spirv::fused_basis_hadamard1024_q8_0_with_subgroup(
                    false,
                    false,
                    use_wave64_shuffle,
                ),
                3,
                7,
            )?,
            make_spec(
                b"fused_hadamard1024_q8_0_pinned",
                spirv::fused_basis_hadamard1024_q8_0_with_subgroup(true, false, use_wave64_shuffle),
                3,
                7,
            )?,
            make_spec(
                b"fused_hadamard1024_q8_0_pinned_native",
                spirv::fused_basis_hadamard1024_q8_0_with_subgroup(true, true, use_wave64_shuffle),
                3,
                7,
            )?,
            make_spec(
                b"ffn_hadamard_batched_f32",
                spirv::ffn_hadamard_batched_f32(),
                2,
                4,
            )?,
            make_spec(
                b"f32_matvec",
                spirv::f32_matvec_with_subgroup(use_wave64_shuffle),
                3,
                2,
            )?,
            make_spec(
                b"f32_gemm_batched",
                spirv::f32_gemm_batched_with_subgroup(use_wave64_shuffle),
                3,
                3,
            )?,
            make_spec(b"lora_b_accum_f32", spirv::lora_b_accum_f32(), 3, 3)?,
            make_spec(
                b"lora_b_accum_batched_f32",
                spirv::lora_b_accum_batched_f32(),
                3,
                4,
            )?,
            make_spec(b"mul_f32_elemwise", spirv::mul_f32_elemwise(), 3, 1)?,
            make_spec(
                b"mul_f32_elemwise_batched",
                spirv::mul_f32_elemwise_batched(),
                3,
                2,
            )?,
            make_spec(
                b"permute_ssm_heads_f32",
                spirv::permute_ssm_heads_f32(),
                2,
                3,
            )?,
            make_spec(
                b"permute_ssm_heads_batched_f32",
                spirv::permute_ssm_heads_batched_f32(),
                2,
                4,
            )?,
            make_spec(b"copy_slice_f32", spirv::copy_slice_f32(), 2, 2)?,
            make_spec(
                b"copy_slice_offset_f32",
                spirv::copy_slice_offset_f32(),
                2,
                3,
            )?,
        ];

        let digest_bytes = hasher
            .finish()
            .map_err(|err| Error::InvalidInput(err.to_string()))?;
        let shader_digests_sha256 = hex::encode(digest_bytes);

        Ok(PreparedLayerShaders {
            use_sdot,
            use_wave64: use_wave64_shuffle,
            proj,
            modules,
            shader_digests_sha256,
        })
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Compiles all prepared projection, layer, LoRA, batched prefill, and attention SPIR-V pipelines sequentially."
    )]
    fn compile_prepared(
        device: &mut AdrenoVulkanDevice,
        prepared: PreparedLayerShaders,
    ) -> Result<Self, Error> {
        let proj = ProjectionPipelines::compile_prepared(device, prepared.proj)?;
        let mut ids = Vec::with_capacity(prepared.modules.len());
        let mut digests = Vec::with_capacity(8);
        for (idx, spec) in prepared.modules.into_iter().enumerate() {
            let _ = spec.tag;
            if idx < 8 {
                digests.push(spec.digest);
            }
            ids.push(device.create_compute_pipeline(
                &spec.words,
                spec.num_buffers,
                spec.push_u32s,
            )?);
        }
        let [
            attn_qk_digest,
            attn_qk_f16kv_digest,
            attn_softmax_digest,
            attn_pv_digest,
            attn_pv_f16kv_digest,
            attn_fused_digest,
            attn_fused_f16kv_digest,
            attn_fused_gqa_prefill_f16kv_digest,
        ]: [String; 8] = digests
            .try_into()
            .map_err(|_| Error::InvalidInput("attention digest count mismatch".to_owned()))?;
        let [
            attn_qk_scores,
            attn_qk_scores_f16kv,
            attn_softmax_f16,
            attn_pv_accum,
            attn_pv_accum_f16kv,
            attn_fused_f16prob,
            attn_fused_f16prob_f16kv,
            attn_fused_gqa_prefill_f16kv,
            rms_norm,
            add_rms_norm,
            rope,
            rope_batched,
            mul_sigmoid,
            conv1d_silu,
            conv1d_silu_offset,
            conv1d_silu_batched_inplace,
            l2_norm,
            l2_norm_batched_strided,
            gdn_step,
            gdn_step_batched_token,
            gdn_scan_batched,
            gdn_gates,
            gdn_gates_batched,
            deinterleave_q_gate,
            deinterleave_q_gate_batched,
            round_f16,
            kv_append_f32,
            kv_append_batched_f32,
            kv_append_f16,
            kv_append_batched_f16,
            quant_q8_0_source,
            quant_q8_0_source_batched,
            quant_q8_0_pinned,
            quant_q8_0_pinned_batched,
            quant_q8_0_pinned_native,
            quant_q8_0_pinned_native_batched,
            fused_hadamard1024_q8k_pinned,
            fused_hadamard1024_q8k_unpinned,
            fused_hadamard1024_q8_0_source,
            fused_hadamard1024_q8_0_pinned,
            fused_hadamard1024_q8_0_pinned_native,
            hadamard_batched,
            f32_matvec,
            f32_gemm_batched,
            lora_b_accum,
            lora_b_accum_batched,
            mul_f32_elemwise,
            mul_f32_elemwise_batched,
            permute_ssm_heads,
            permute_ssm_heads_batched,
            copy_slice,
            copy_slice_offset,
        ]: [PipelineId; 52] = ids
            .try_into()
            .map_err(|_| Error::InvalidInput("layer pipeline count mismatch".to_owned()))?;

        Ok(Self {
            proj,
            attn_qk_scores,
            attn_qk_scores_f16kv,
            attn_softmax_f16,
            attn_pv_accum,
            attn_pv_accum_f16kv,
            attn_fused_f16prob,
            attn_fused_f16prob_f16kv,
            attn_fused_gqa_prefill_f16kv,
            rms_norm,
            add_rms_norm,
            rope,
            rope_batched,
            mul_sigmoid,
            conv1d_silu,
            conv1d_silu_offset,
            conv1d_silu_batched_inplace,
            l2_norm,
            l2_norm_batched_strided,
            gdn_step,
            gdn_step_batched_token,
            gdn_scan_batched,
            gdn_gates,
            gdn_gates_batched,
            deinterleave_q_gate,
            deinterleave_q_gate_batched,
            round_f16,
            kv_append_f32,
            kv_append_batched_f32,
            kv_append_f16,
            kv_append_batched_f16,
            quant_q8_0_source,
            quant_q8_0_source_batched,
            quant_q8_0_pinned,
            quant_q8_0_pinned_batched,
            quant_q8_0_pinned_native,
            quant_q8_0_pinned_native_batched,
            fused_hadamard1024_q8k_pinned,
            fused_hadamard1024_q8k_unpinned,
            fused_hadamard1024_q8_0_source,
            fused_hadamard1024_q8_0_pinned,
            fused_hadamard1024_q8_0_pinned_native,
            hadamard_batched,
            f32_matvec,
            f32_gemm_batched,
            lora_b_accum,
            lora_b_accum_batched,
            mul_f32_elemwise,
            mul_f32_elemwise_batched,
            permute_ssm_heads,
            permute_ssm_heads_batched,
            copy_slice,
            copy_slice_offset,
            use_sdot: prepared.use_sdot,
            attn_qk_digest,
            attn_qk_f16kv_digest,
            attn_softmax_digest,
            attn_pv_digest,
            attn_pv_f16kv_digest,
            attn_fused_digest,
            attn_fused_f16kv_digest,
            attn_fused_gqa_prefill_f16kv_digest,
            shader_digests_sha256: prepared.shader_digests_sha256,
        })
    }

    /// Compiles every layer, projection, `LoRA`, and attention SPIR-V 1.6 pipeline on `device`.
    ///
    /// # Errors
    /// Returns [`Error::Unsupported`], [`Error::Vulkan`], or [`Error::InvalidInput`] if pipeline
    /// creation or digest calculation fails.
    pub fn compile(device: &mut AdrenoVulkanDevice, use_sdot: bool) -> Result<Self, Error> {
        let use_wave64 = device.capabilities().wave64_subgroup_shuffle;
        let prepared = Self::prepare(use_sdot, use_wave64)?;
        Self::compile_prepared(device, prepared)
    }

    /// Returns whether this compiled bundle uses hardware `VK_KHR_shader_integer_dot_product` (`OpSDot`).
    #[must_use]
    pub const fn uses_sdot(&self) -> bool {
        self.use_sdot
    }

    /// Returns the deterministic SHA-256 digest over all compiled SPIR-V shader modules.
    #[must_use]
    pub fn shader_digests_sha256(&self) -> &str {
        &self.shader_digests_sha256
    }
}

const KV_PREFIX_HISTORY_LIMIT: usize = 64;

#[derive(Clone, Debug, PartialEq, Eq)]
struct KvPrefixTracker {
    current_gen: u64,
    high_water: usize,
    valid_prefixes: Vec<(u64, usize)>,
}

impl KvPrefixTracker {
    const fn new() -> Self {
        Self {
            current_gen: 1,
            high_water: 0,
            valid_prefixes: Vec::new(),
        }
    }

    fn invalidate(&mut self) {
        self.current_gen = self.current_gen.wrapping_add(1).max(1);
        self.high_water = 0;
        self.valid_prefixes.clear();
    }

    fn on_reset(&mut self) {
        self.invalidate();
    }

    fn on_full_kv_restore(&mut self, restored_position: usize) {
        self.current_gen = self.current_gen.wrapping_add(1).max(1);
        self.high_water = restored_position;
        self.valid_prefixes.clear();
    }

    fn begin_write(&mut self, write_start: usize, token_count: usize) {
        let write_end = write_start.saturating_add(token_count);
        if write_start < self.high_water {
            for (_, intact_limit) in &mut self.valid_prefixes {
                *intact_limit = (*intact_limit).min(write_start);
            }
            self.valid_prefixes.retain(|&(_, limit)| limit > 0);
            if write_start > 0 {
                if self.valid_prefixes.len() >= KV_PREFIX_HISTORY_LIMIT {
                    self.valid_prefixes.remove(0);
                }
                self.valid_prefixes.push((self.current_gen, write_start));
            }
            self.current_gen = self.current_gen.wrapping_add(1).max(1);
            self.high_water = write_end;
        } else {
            self.high_water = self.high_water.max(write_end);
        }
    }

    fn intact_prefix_for(&self, target_gen: u64, watermark: usize) -> usize {
        if target_gen == 0 {
            return 0;
        }
        if target_gen == self.current_gen {
            return self.high_water.min(watermark);
        }
        self.valid_prefixes
            .iter()
            .rev()
            .find(|&&(entry_gen, _)| entry_gen == target_gen)
            .map_or(0, |&(_, limit)| limit.min(watermark))
    }

    fn can_skip_kv_restore(
        &self,
        state_consistent: bool,
        checkpoint_gen: u64,
        checkpoint_watermark: usize,
        checkpoint_position: usize,
    ) -> bool {
        if checkpoint_position == 0 {
            return state_consistent;
        }
        state_consistent
            && self.intact_prefix_for(checkpoint_gen, checkpoint_watermark) >= checkpoint_position
    }
}

/// Checkpoint snapshot of GPU-resident recurrent GDN/conv state and attention KV cache.
#[derive(Clone, Debug, PartialEq)]
pub struct VulkanBatchedCheckpoint {
    executor_identity_sha256: String,
    provenance_nonce: u64,
    position: usize,
    mrope_positions: [usize; 4],
    kv_generation: u64,
    kv_prefix_watermark: usize,
    layers: Vec<LayerStateCheckpoint>,
}

#[derive(Clone, Debug, PartialEq)]
enum LayerStateCheckpoint {
    Full {
        precision: VulkanKvPrecision,
        width: usize,
        keys_words: Vec<u32>,
        values_words: Vec<u32>,
    },
    Recurrent {
        convolution: Vec<f32>,
        matrix: Vec<f32>,
    },
}

impl VulkanBatchedCheckpoint {
    /// Returns the committed token position captured in this checkpoint.
    #[must_use]
    pub const fn position(&self) -> usize {
        self.position
    }

    /// Returns the executor KV cache lineage generation active when this checkpoint was taken.
    #[must_use]
    pub const fn kv_generation(&self) -> u64 {
        self.kv_generation
    }

    /// Returns the contiguous intact KV prefix watermark (in tokens) recorded by this checkpoint.
    #[must_use]
    pub const fn kv_prefix_watermark(&self) -> usize {
        self.kv_prefix_watermark
    }

    /// Returns the number of transformer layers captured in this checkpoint.
    #[must_use]
    pub const fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Returns the provenance identity SHA-256 of the executor that produced this checkpoint.
    #[must_use]
    pub fn executor_identity_sha256(&self) -> &str {
        &self.executor_identity_sha256
    }

    /// Computes a deterministic SHA-256 fingerprint over this checkpoint's position and state.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] if hashing or integer conversion fails.
    pub fn identity_sha256(&self) -> Result<String, Error> {
        let mut hasher = Hasher::new().map_err(|err| Error::InvalidInput(err.to_string()))?;
        hash_str(
            &mut hasher,
            b"vk-batched-checkpoint-v1:",
            &self.executor_identity_sha256,
        )?;
        hash_u64(&mut hasher, self.provenance_nonce)?;
        hash_usize(&mut hasher, self.position)?;
        for pos in self.mrope_positions {
            hash_usize(&mut hasher, pos)?;
        }
        hash_usize(&mut hasher, self.layers.len())?;
        for layer in &self.layers {
            match layer {
                LayerStateCheckpoint::Full {
                    precision,
                    width,
                    keys_words,
                    values_words,
                } => {
                    hash_str(&mut hasher, b"full:", precision.name())?;
                    hash_usize(&mut hasher, *width)?;
                    hash_usize(&mut hasher, keys_words.len())?;
                    for &w in keys_words {
                        hash_u32(&mut hasher, w)?;
                    }
                    hash_usize(&mut hasher, values_words.len())?;
                    for &w in values_words {
                        hash_u32(&mut hasher, w)?;
                    }
                }
                LayerStateCheckpoint::Recurrent {
                    convolution,
                    matrix,
                } => {
                    hash_bytes(&mut hasher, b"rec:")?;
                    hash_usize(&mut hasher, convolution.len())?;
                    for &v in convolution {
                        hash_u32(&mut hasher, v.to_bits())?;
                    }
                    hash_usize(&mut hasher, matrix.len())?;
                    for &v in matrix {
                        hash_u32(&mut hasher, v.to_bits())?;
                    }
                }
            }
        }
        let digest = hasher
            .finish()
            .map_err(|err| Error::InvalidInput(err.to_string()))?;
        Ok(hex::encode(digest))
    }
}

#[derive(Clone, Copy, Default)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "Tracks disjoint scratch buffer usage across sibling basis subgroups inside one projection group."
)]
struct ProjectionGroupSync {
    src_barrier_emitted: bool,
    used_q8k: bool,
    used_q8_0: bool,
    used_perm_tmp: bool,
    used_sign_tmp: bool,
    used_transformed: bool,
}

const MAX_READ_PIPELINE_CACHE_FILE_BYTES: u64 = 64 * 1024 * 1024 + 4096;

fn read_pipeline_cache_file(path: &Path) -> Option<Vec<u8>> {
    let meta = std::fs::metadata(path).ok()?;
    if !meta.is_file() || meta.len() == 0 || meta.len() > MAX_READ_PIPELINE_CACHE_FILE_BYTES {
        return None;
    }
    std::fs::read(path).ok()
}

fn write_pipeline_cache_file(path: &Path, bytes: &[u8]) {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
        && std::fs::create_dir_all(parent).is_err()
    {
        return;
    }
    let tmp_path = path.with_extension(format!("tmp.{}", std::process::id()));
    if std::fs::write(&tmp_path, bytes).is_ok() {
        if std::fs::rename(&tmp_path, path).is_ok() {
            return;
        }
        let _ = std::fs::remove_file(&tmp_path);
    }
    let _ = std::fs::write(path, bytes);
}

#[derive(Clone, Copy, Debug)]
struct GraphGeometry {
    embedding: usize,
    feed_forward: usize,
    vocabulary: usize,
    heads: usize,
    kv_heads: usize,
    head: usize,
    rope: usize,
    rope_sections: [usize; 4],
    rope_base: f32,
    epsilon: f32,
    ssm_head: usize,
    ssm_keys: usize,
    ssm_values: usize,
    convolution: usize,
    context_limit: usize,
}

impl GraphGeometry {
    const fn key_width(&self) -> usize {
        self.ssm_head * self.ssm_keys
    }

    const fn value_width(&self) -> usize {
        self.ssm_head * self.ssm_values
    }

    const fn convolution_width(&self) -> usize {
        2 * self.key_width() + self.value_width()
    }

    const fn kv_width(&self) -> usize {
        self.kv_heads * self.head
    }

    const fn q_width(&self) -> usize {
        self.heads * self.head
    }
}

#[derive(Default)]
struct GraphTransforms {
    block: usize,
    weights: BTreeSet<String>,
    inverse_embedding: bool,
    grouped_values: bool,
    sign_buffers: BTreeMap<usize, BufferId>,
}

impl GraphTransforms {
    fn basis(&self, name: &str) -> (bool, bool) {
        let rotated = self.weights.contains(name);
        (
            rotated,
            rotated && self.grouped_values && name.ends_with(".ssm_out.weight"),
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WeightEncoding {
    Pq2Q8k,
    Pq2Q8_0,
    Ptq1Q8_0,
    Q8_0,
    F32,
}

#[derive(Clone, Copy, Debug)]
struct GpuLoraFactor {
    a_span: BufferSpan,
    b_span: BufferSpan,
    in_features: usize,
    out_features: usize,
    rank: usize,
    scaling: f32,
}

#[derive(Default)]
struct UploadedAdapters {
    multi_rank_linear: BTreeMap<String, GpuLoraFactor>,
    max_rank: usize,
    content_sha256: String,
    multi_rank_scale: Option<f32>,
}

#[derive(Clone, Copy, Debug)]
struct GpuMatrixWeight {
    span: BufferSpan,
    encoding: WeightEncoding,
    cols: usize,
    rows: usize,
    rotated: bool,
    grouped_ssm: bool,
    multi_rank: Option<GpuLoraFactor>,
}

impl GpuMatrixWeight {
    const fn basis(&self) -> (bool, bool) {
        (self.rotated, self.grouped_ssm)
    }
}

enum GpuLayerAttention {
    Recurrent {
        qkv: GpuMatrixWeight,
        gate: GpuMatrixWeight,
        beta: GpuMatrixWeight,
        alpha: GpuMatrixWeight,
        convolution_w: BufferId,
        time_bias: BufferId,
        ssm_a: BufferId,
        norm_w: BufferId,
        output: GpuMatrixWeight,
        conv_state: BufferId,
        conv_state_elems: usize,
        matrix_state: BufferId,
        matrix_state_elems: usize,
    },
    Full {
        query: GpuMatrixWeight,
        key: GpuMatrixWeight,
        value: GpuMatrixWeight,
        output: GpuMatrixWeight,
        query_norm_w: BufferId,
        key_norm_w: BufferId,
        k_cache: BufferId,
        v_cache: BufferId,
    },
}

struct GpuLayer {
    attn_norm_w: BufferId,
    post_norm_w: BufferId,
    attention: GpuLayerAttention,
    ffn_gate: GpuMatrixWeight,
    ffn_up: GpuMatrixWeight,
    ffn_down: GpuMatrixWeight,
}

struct GpuGraphScratch {
    hidden: BufferId,
    hidden_staging: [BufferId; 2],
    norm: BufferId,
    delta: BufferId,
    transformed: BufferId,
    perm_tmp: BufferId,
    sign_tmp: BufferId,
    act_q8k: BufferId,
    act_q8_0: BufferId,
    proj_a: BufferId,
    proj_b: BufferId,
    proj_c: BufferId,
    proj_d: BufferId,
    proj_e: BufferId,
    gdn_k: BufferId,
    gdn_v: BufferId,
    rope_freqs: BufferId,
    rope_freqs_staging: [BufferId; 2],
    lora_ranks: BufferId,
    #[expect(
        dead_code,
        reason = "Reserved per-bank staging slots aligned with tile scratch layout."
    )]
    lora_ranks_staging: [BufferId; 2],
    attn_scores: BufferId,
    attn_probs: BufferId,
    attn_score_bytes: u64,
    allocated_tile_tokens: usize,
    logits: BufferId,
    dummy_state: BufferId,
}

struct BufferCleanupGuard<'a> {
    device: &'a mut AdrenoVulkanDevice,
    arena: WeightArena,
    buffers: Vec<BufferId>,
    active: bool,
}

impl<'a> BufferCleanupGuard<'a> {
    fn new(device: &'a mut AdrenoVulkanDevice) -> Self {
        let arena = WeightArena::new(device);
        Self {
            device,
            arena,
            buffers: Vec::new(),
            active: true,
        }
    }

    fn alloc_storage(&mut self, bytes: u64) -> Result<BufferId, Error> {
        let id = self.device.create_storage_buffer(bytes)?;
        self.buffers.push(id);
        Ok(id)
    }

    fn alloc_readback_storage(&mut self, bytes: u64) -> Result<BufferId, Error> {
        let id = self.device.create_readback_storage_buffer(bytes)?;
        self.buffers.push(id);
        Ok(id)
    }

    fn alloc_f32_buffer(&mut self, data: &[f32]) -> Result<BufferId, Error> {
        let byte_len = f32_slice_bytes(data.len())?.max(MIN_NORM_OR_BIAS_BYTES);
        let id = self.alloc_storage(byte_len)?;
        self.device.write_buffer_f32(id, data)?;
        Ok(id)
    }

    fn defuse(mut self) -> (WeightArena, Vec<BufferId>) {
        self.active = false;
        let arena = std::mem::take(&mut self.arena);
        let buffers = std::mem::take(&mut self.buffers);
        (arena, buffers)
    }
}

impl Drop for BufferCleanupGuard<'_> {
    fn drop(&mut self) {
        if self.active {
            for &id in &self.buffers {
                self.device.free_buffer(id);
            }
            self.arena.free_all(self.device);
        }
    }
}

/// Whole-layer and multi-layer chunked GPU graph executor for hybrid `Qwen35` models on Adreno.
///
/// Weights, layer normalization vectors, `F32` recurrent GDN/conv state, `FP16`/`F32` KV caches,
/// and reusable layer activation scratch live persistently in GPU memory so multi-layer chunks
/// execute end-to-end without host activation round-trips between operators or layers.
pub struct VulkanBatchedGraphExecutor {
    device: AdrenoVulkanDevice,
    pipelines: VulkanLayerPipelines,
    kernel_path: ProjectionKernelPath,
    config: VulkanBatchedConfig,
    resolved_context_limit: usize,
    resolved_pinned_arithmetic: bool,
    geometry: GraphGeometry,
    transforms: GraphTransforms,
    embedding: Tensor,
    arena: WeightArena,
    owned_buffers: Vec<BufferId>,
    layers: Vec<GpuLayer>,
    output_norm_w: BufferId,
    output_weight: GpuMatrixWeight,
    scratch: GpuGraphScratch,
    batch_recorder: CommandBatchRecorder,
    host_embed_scratch: Vec<f32>,
    host_embed_tile_scratch: Vec<f32>,
    host_rope_freqs: Vec<f32>,
    staging_bank_in_flight: [bool; 2],
    next_staging_bank: usize,
    multi_rank_scale: Option<f32>,
    base_provenance_sha256: String,
    provenance_nonce: u64,
    kv_tracker: KvPrefixTracker,
    kv_fast_restores: u64,
    kv_full_restores: u64,
    position: usize,
    mrope_positions: [usize; 4],
    submits_completed: u64,
    state_consistent: bool,
    identity_sha256: String,
}

impl Drop for VulkanBatchedGraphExecutor {
    fn drop(&mut self) {
        for &id in &self.owned_buffers {
            self.device.free_buffer(id);
        }
        self.arena.free_all(&mut self.device);
    }
}

impl VulkanBatchedGraphExecutor {
    /// Opens the native Adreno Vulkan device and uploads `model` onto the GPU with `config`.
    ///
    /// # Errors
    /// Returns [`Error::Unsupported`], [`Error::InvalidInput`], [`Error::ResourceLimit`],
    /// [`Error::DeviceFault`], or [`Error::Vulkan`].
    pub fn open(model: &Model, config: VulkanBatchedConfig) -> Result<Self, Error> {
        Self::open_with_adapters(model, config, VulkanBatchedAdapters::default())
    }

    /// Opens the native Adreno Vulkan device and uploads `model` together with GPU-resident
    /// `MultiRankAdapter` weights.
    ///
    /// # Errors
    /// Returns [`Error::Unsupported`], [`Error::InvalidInput`], [`Error::ResourceLimit`],
    /// [`Error::DeviceFault`], or [`Error::Vulkan`].
    pub fn open_with_adapters(
        model: &Model,
        config: VulkanBatchedConfig,
        adapters: VulkanBatchedAdapters<'_>,
    ) -> Result<Self, Error> {
        let Some(cache_path) = config.pipeline_cache_path.clone() else {
            let device = AdrenoVulkanDevice::open()?;
            return Self::from_device_with_adapters(device, model, config, adapters);
        };

        config.tdr_policy.validate()?;
        let use_sdot_hint = !matches!(
            config.kernel_path,
            Some(ProjectionKernelPath::PortableIntegerFallback)
        );
        let prepared = VulkanLayerPipelines::prepare(use_sdot_hint, true)?;
        let cached_blob = read_pipeline_cache_file(&cache_path);
        let (device, loaded_from_cache) = match cached_blob.as_deref() {
            Some(blob) => match AdrenoVulkanDevice::open_with_pipeline_cache_data(
                Some(blob),
                &prepared.shader_digests_sha256,
            ) {
                Ok(dev) => (dev, true),
                Err(Error::InvalidInput(_)) => (AdrenoVulkanDevice::open()?, false),
                Err(err) => return Err(err),
            },
            None => (AdrenoVulkanDevice::open()?, false),
        };

        let executor = Self::from_device_with_adapters_prepared(
            device,
            model,
            config,
            adapters,
            Some(prepared),
        )?;
        if (!loaded_from_cache
            || executor.pipelines.uses_sdot() != use_sdot_hint
            || !executor.device.capabilities().wave64_subgroup_shuffle)
            && let Ok(exported) = executor
                .device
                .export_pipeline_cache_data(executor.pipelines.shader_digests_sha256())
        {
            write_pipeline_cache_file(&cache_path, &exported);
        }
        Ok(executor)
    }

    /// Uploads `model` onto an already-opened [`AdrenoVulkanDevice`] with `config`.
    ///
    /// # Errors
    /// Returns [`Error::Unsupported`], [`Error::InvalidInput`], [`Error::ResourceLimit`],
    /// [`Error::DeviceFault`], or [`Error::Vulkan`].
    pub fn from_device(
        device: AdrenoVulkanDevice,
        model: &Model,
        config: VulkanBatchedConfig,
    ) -> Result<Self, Error> {
        Self::from_device_with_adapters(device, model, config, VulkanBatchedAdapters::default())
    }

    /// Uploads `model` and GPU-resident `LoRA` adapters onto an already-opened [`AdrenoVulkanDevice`].
    ///
    /// # Errors
    /// Returns [`Error::Unsupported`], [`Error::InvalidInput`], [`Error::ResourceLimit`],
    /// [`Error::DeviceFault`], or [`Error::Vulkan`].
    pub fn from_device_with_adapters(
        device: AdrenoVulkanDevice,
        model: &Model,
        config: VulkanBatchedConfig,
        adapters: VulkanBatchedAdapters<'_>,
    ) -> Result<Self, Error> {
        Self::from_device_with_adapters_prepared(device, model, config, adapters, None)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Coordinates single-pass model/adapter validation, GPU weight upload, state/scratch allocation, and deterministic identity hashing."
    )]
    fn from_device_with_adapters_prepared(
        mut device: AdrenoVulkanDevice,
        model: &Model,
        config: VulkanBatchedConfig,
        adapters: VulkanBatchedAdapters<'_>,
        prepared_shaders: Option<PreparedLayerShaders>,
    ) -> Result<Self, Error> {
        config.tdr_policy.validate()?;
        if config.prefill_tile_tokens == 0 || config.prefill_tile_tokens > MAX_PREFILL_BLOCK_TOKENS
        {
            return Err(Error::InvalidInput(format!(
                "prefill_tile_tokens must be in 1..={MAX_PREFILL_BLOCK_TOKENS}, got {}",
                config.prefill_tile_tokens
            )));
        }
        let preferred_path = if device.capabilities().supports_integer_dot_4x8() {
            ProjectionKernelPath::IntegerDotPacked4x8
        } else {
            ProjectionKernelPath::PortableIntegerFallback
        };
        let kernel_path = config.kernel_path.unwrap_or(preferred_path);
        let use_sdot = matches!(kernel_path, ProjectionKernelPath::IntegerDotPacked4x8);
        if use_sdot && !device.capabilities().supports_integer_dot_4x8() {
            return Err(Error::Unsupported);
        }

        let (geometry, layer_count, recurrent_flags) = parse_model_geometry(model, &config)?;
        let resolved_context_limit = geometry.context_limit;
        let resolved_pinned_arithmetic = config.pinned_arithmetic.unwrap_or(matches!(
            model.identity.as_str(),
            BONSAI_2_5_PQ2_SHA256 | BONSAI_2_5_PTQ1_SHA256
        ));

        let embedding = lookup_tensor(model, "token_embd.weight", &[geometry.embedding, 0])?;
        if embedding.dimensions.get(1).copied() != Some(geometry.vocabulary) {
            return Err(Error::InvalidInput(
                "token_embd.weight vocabulary extent mismatch".to_owned(),
            ));
        }
        validate_model_tensors_subset(model, &geometry, layer_count, &recurrent_flags)?;

        let use_wave64 = device.capabilities().wave64_subgroup_shuffle;
        let pipelines = match prepared_shaders {
            Some(prepared)
                if prepared.use_sdot == use_sdot && prepared.use_wave64 == use_wave64 =>
            {
                VulkanLayerPipelines::compile_prepared(&mut device, prepared)?
            }
            _ => VulkanLayerPipelines::compile(&mut device, use_sdot)?,
        };
        let min_align =
            usize::try_from(device.min_storage_buffer_offset_alignment().max(1)).unwrap_or(256);
        let max_storage_range = device
            .capabilities()
            .compute_limits
            .max_storage_buffer_range;
        let max_wg = device
            .capabilities()
            .compute_limits
            .max_compute_work_group_count;

        let mut guard = BufferCleanupGuard::new(&mut device);
        let transforms = parse_and_upload_transforms(model, &geometry, &mut guard)?;
        let has_separate_output = model.tensors.contains_key("output.weight");
        let uploaded_adapters = parse_and_upload_adapters(
            model,
            &geometry,
            &recurrent_flags,
            has_separate_output,
            adapters,
            &mut guard,
        )?;

        let mut repack_scratch = Vec::<u32>::new();
        let mut row_scratch = Vec::<f32>::new();
        let mut layers = Vec::with_capacity(layer_count);

        for (layer_idx, &is_recurrent) in recurrent_flags.iter().enumerate() {
            let prefix = format!("blk.{layer_idx}.");
            let attn_norm_w = load_and_upload_vector(
                model,
                &format!("{prefix}attn_norm.weight"),
                geometry.embedding,
                &mut row_scratch,
                &mut guard,
            )?;
            let post_norm_w = load_and_upload_vector(
                model,
                &format!("{prefix}post_attention_norm.weight"),
                geometry.embedding,
                &mut row_scratch,
                &mut guard,
            )?;

            let attention = if is_recurrent {
                upload_recurrent_attention(
                    model,
                    &prefix,
                    &geometry,
                    &transforms,
                    &uploaded_adapters,
                    &mut repack_scratch,
                    &mut row_scratch,
                    &mut guard,
                )?
            } else {
                upload_full_attention(
                    model,
                    &prefix,
                    &geometry,
                    config.kv_precision,
                    &transforms,
                    &uploaded_adapters,
                    &mut repack_scratch,
                    &mut row_scratch,
                    &mut guard,
                )?
            };

            let ffn_gate = upload_matrix_by_name(
                model,
                &format!("{prefix}ffn_gate.weight"),
                geometry.embedding,
                geometry.feed_forward,
                &transforms,
                &uploaded_adapters,
                &mut repack_scratch,
                &mut row_scratch,
                &mut guard,
            )?;
            let ffn_up = upload_matrix_by_name(
                model,
                &format!("{prefix}ffn_up.weight"),
                geometry.embedding,
                geometry.feed_forward,
                &transforms,
                &uploaded_adapters,
                &mut repack_scratch,
                &mut row_scratch,
                &mut guard,
            )?;
            let ffn_down = upload_matrix_by_name(
                model,
                &format!("{prefix}ffn_down.weight"),
                geometry.feed_forward,
                geometry.embedding,
                &transforms,
                &uploaded_adapters,
                &mut repack_scratch,
                &mut row_scratch,
                &mut guard,
            )?;

            layers.push(GpuLayer {
                attn_norm_w,
                post_norm_w,
                attention,
                ffn_gate,
                ffn_up,
                ffn_down,
            });
        }

        let output_norm_w = load_and_upload_vector(
            model,
            "output_norm.weight",
            geometry.embedding,
            &mut row_scratch,
            &mut guard,
        )?;
        let output_tensor = if has_separate_output {
            lookup_tensor(
                model,
                "output.weight",
                &[geometry.embedding, geometry.vocabulary],
            )?
        } else {
            embedding.clone()
        };
        let output_weight = upload_matrix_tensor(
            &output_tensor,
            &transforms,
            &uploaded_adapters,
            &mut repack_scratch,
            &mut row_scratch,
            &mut guard,
        )?;

        let scratch = allocate_scratch(
            &geometry,
            uploaded_adapters.max_rank,
            config.prefill_tile_tokens,
            config.scratch_budget_bytes,
            max_storage_range,
            max_wg,
            &mut guard,
        )?;

        let (arena, owned_buffers) = guard.defuse();
        let base_provenance_sha256 = compute_base_provenance(
            &device,
            model,
            &config,
            resolved_context_limit,
            resolved_pinned_arithmetic,
            kernel_path,
            layer_count,
            &pipelines,
            &uploaded_adapters.content_sha256,
        )?;
        let identity_sha256 =
            compute_executor_identity(&base_provenance_sha256, uploaded_adapters.multi_rank_scale)?;

        let _ = min_align;
        let alloc_tile = scratch.allocated_tile_tokens;
        let host_rope_freqs = vec![0.0_f32; alloc_tile * geometry.rope];
        let host_embed_tile_scratch = Vec::with_capacity(alloc_tile * geometry.embedding);
        Ok(Self {
            device,
            pipelines,
            kernel_path,
            config,
            resolved_context_limit,
            resolved_pinned_arithmetic,
            geometry,
            transforms,
            embedding,
            arena,
            owned_buffers,
            layers,
            output_norm_w,
            output_weight,
            scratch,
            batch_recorder: CommandBatchRecorder::with_capacity(256),
            host_embed_scratch: Vec::with_capacity(geometry.embedding),
            host_embed_tile_scratch,
            host_rope_freqs,
            staging_bank_in_flight: [false; 2],
            next_staging_bank: 0,
            multi_rank_scale: uploaded_adapters.multi_rank_scale,
            base_provenance_sha256,
            provenance_nonce: 0,
            kv_tracker: KvPrefixTracker::new(),
            kv_fast_restores: 0,
            kv_full_restores: 0,
            position: 0,
            mrope_positions: [0; 4],
            submits_completed: 0,
            state_consistent: true,
            identity_sha256,
        })
    }

    /// Evaluates one token forward pass on GPU and returns `[vocabulary]` output logits.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`], [`Error::ResourceLimit`], [`Error::DeviceFault`],
    /// [`Error::Unsupported`], or [`Error::Vulkan`].
    pub fn evaluate_token(&mut self, token: u32) -> Result<Vec<f32>, Error> {
        let mut logits = Vec::new();
        self.evaluate_token_into(token, &mut logits)?;
        Ok(logits)
    }

    /// Evaluates one token forward pass on GPU, writing `[vocabulary]` logits into `logits`.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`], [`Error::ResourceLimit`], [`Error::DeviceFault`],
    /// [`Error::Unsupported`], or [`Error::Vulkan`].
    pub fn evaluate_token_into(&mut self, token: u32, logits: &mut Vec<f32>) -> Result<(), Error> {
        logits.clear();
        let res = self.step_token_impl(token, Some(logits));
        if res.is_err() {
            logits.clear();
        }
        res
    }

    /// Advances model recurrent/KV state on GPU by ingesting one prompt token without reading logits.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`], [`Error::ResourceLimit`], [`Error::DeviceFault`],
    /// [`Error::Unsupported`], or [`Error::Vulkan`].
    pub fn ingest_token(&mut self, token: u32) -> Result<(), Error> {
        self.step_token_impl(token, None)
    }

    /// Advances model recurrent/KV state on GPU across `tokens` in batched GPU tiles without
    /// reading logits.
    ///
    /// Tiles up to [`VulkanBatchedConfig::prefill_tile_tokens`] (bounded by scratch budget and
    /// device limits) evaluate together in a single GPU sweep over layer weights, using 16-token /
    /// 4-token coalesced GEMM kernels for quantized/F32 projections and `#40` multi-query causal
    /// attention, overlapping double-buffered host staging across consecutive mid-prefill tiles.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] if `tokens` is empty, contains an out-of-vocabulary token,
    /// or exceeds `context_limit`, or [`Error::ResourceLimit`]/[`Error::DeviceFault`]/[`Error::Vulkan`]
    /// on GPU failure.
    pub fn ingest_tokens_batched(&mut self, tokens: &[u32]) -> Result<(), Error> {
        self.ingest_tokens_batched_with_check(tokens, &mut || Ok(()))
    }

    /// Advances model recurrent/KV state on GPU across `tokens` in batched GPU tiles and writes
    /// `[vocabulary]` logits for the final token `tokens[tokens.len() - 1]` into `logits`.
    ///
    /// Mid-prefill tiles evaluate without reading logits and overlap host-upload staging without
    /// draining between tiles. On the final tile, layers `0..last_layer` run batched across all
    /// `t_len` tokens while `last_layer` updates recurrent/KV state for `0..t_len` and computes
    /// attention output, FFN, `output_norm`, and `lm_head` only for slice `t_len - 1`.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] if `tokens` is empty, contains an out-of-vocabulary token,
    /// or exceeds `context_limit`, or [`Error::ResourceLimit`]/[`Error::DeviceFault`]/[`Error::Vulkan`]
    /// on GPU failure. Clears `logits` on error.
    pub fn ingest_block_and_evaluate_last_into(
        &mut self,
        tokens: &[u32],
        logits: &mut Vec<f32>,
    ) -> Result<(), Error> {
        self.ingest_block_and_evaluate_last_with_check_into(tokens, logits, &mut || Ok(()))
    }

    /// Advances model recurrent/KV state on GPU across `tokens` in batched GPU tiles with
    /// cooperative cancellation/deadline checks between tiles and TDR chunks, writing
    /// `[vocabulary]` logits for the final token into `logits`.
    ///
    /// # Errors
    /// Returns [`Error::Cancelled`] when `check` rejects, [`Error::InvalidInput`] if `tokens` is
    /// empty, contains an out-of-vocabulary token, or exceeds `context_limit`, or
    /// [`Error::ResourceLimit`]/[`Error::DeviceFault`]/[`Error::Vulkan`] on GPU failure. Clears
    /// `logits` on error.
    pub fn ingest_block_and_evaluate_last_with_check_into(
        &mut self,
        tokens: &[u32],
        logits: &mut Vec<f32>,
        check: &mut impl FnMut() -> Result<(), Error>,
    ) -> Result<(), Error> {
        logits.clear();
        let res = self.ingest_tokens_and_maybe_eval_last_with_check(tokens, Some(logits), check);
        if res.is_err() {
            logits.clear();
        }
        res
    }

    /// Resets all GPU recurrent GDN/conv state and position counters cleanly back to position `0`.
    ///
    /// # Errors
    /// Returns [`Error::DeviceFault`] or [`Error::Vulkan`] if GPU buffer zeroing fails.
    pub fn reset(&mut self) -> Result<(), Error> {
        if self.device.is_faulted() {
            return Err(Error::DeviceFault(
                "Vulkan batched executor faulted and cannot reset".to_owned(),
            ));
        }
        self.device.ensure_pipelined_idle()?;
        self.staging_bank_in_flight = [false; 2];
        self.next_staging_bank = 0;
        for layer in &self.layers {
            match &layer.attention {
                GpuLayerAttention::Recurrent {
                    conv_state,
                    conv_state_elems,
                    matrix_state,
                    matrix_state_elems,
                    ..
                } => {
                    if *conv_state_elems > 0 {
                        let bytes = self.device.buffer_size_bytes(*conv_state)?;
                        self.device.fill_buffer_zero(*conv_state, 0, bytes)?;
                    }
                    if *matrix_state_elems > 0 {
                        let bytes = self.device.buffer_size_bytes(*matrix_state)?;
                        self.device.fill_buffer_zero(*matrix_state, 0, bytes)?;
                    }
                }
                GpuLayerAttention::Full {
                    k_cache, v_cache, ..
                } => {
                    let k_bytes = self.device.buffer_size_bytes(*k_cache)?;
                    let v_bytes = self.device.buffer_size_bytes(*v_cache)?;
                    self.device.fill_buffer_zero(*k_cache, 0, k_bytes)?;
                    self.device.fill_buffer_zero(*v_cache, 0, v_bytes)?;
                }
            }
        }
        self.position = 0;
        self.mrope_positions = [0; 4];
        self.kv_tracker.on_reset();
        self.state_consistent = true;
        Ok(())
    }

    /// Returns the active multi-rank `LoRA` adapter scale, or `None` when none is loaded.
    #[must_use]
    pub const fn multi_rank_scale(&self) -> Option<f32> {
        self.multi_rank_scale
    }

    /// Dynamically updates the multi-rank `LoRA` scale (`0.0..=2.0`) and invalidates executor
    /// identity and prior checkpoints without freeing or re-uploading GPU weights.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] if no multi-rank adapter is loaded, `scale` is outside
    /// finite `0.0..=2.0`, or identity hashing fails.
    pub fn set_multi_rank_scale(&mut self, scale: f32) -> Result<(), Error> {
        if self.multi_rank_scale.is_none() {
            return Err(Error::InvalidInput(
                "multi-rank adapter scale update requires a loaded multi-rank adapter".to_owned(),
            ));
        }
        self.set_adapter_scales(Some(scale))
    }

    /// Updates the loaded multi-rank `LoRA` scale and invalidates identity and prior checkpoints.
    ///
    /// # Errors
    /// Rejects scale mismatches against the loaded adapter, scales outside finite `0.0..=2.0`,
    /// or identity hashing failures without mutating existing scales.
    pub fn set_adapter_scales(&mut self, multi_rank_scale: Option<f32>) -> Result<(), Error> {
        let resolved_mr = self.validate_adapter_scale_update(multi_rank_scale)?;
        let new_identity = compute_executor_identity(&self.base_provenance_sha256, resolved_mr)?;
        self.multi_rank_scale = resolved_mr;
        self.identity_sha256 = new_identity;
        self.provenance_nonce = self.provenance_nonce.wrapping_add(1);
        Ok(())
    }

    /// Resets all GPU recurrent/KV state to position `0` and updates the multi-rank `LoRA` scale.
    ///
    /// # Errors
    /// Rejects missing adapters, invalid scales, or Vulkan reset failures.
    pub fn reset_with_multi_rank_scale(&mut self, scale: f32) -> Result<(), Error> {
        if self.multi_rank_scale.is_none() {
            return Err(Error::InvalidInput(
                "multi-rank adapter scale update requires a loaded multi-rank adapter".to_owned(),
            ));
        }
        self.reset_with_adapter_scales(Some(scale))
    }

    /// Atomically resets all GPU recurrent/KV state to position `0`, updates the loaded adapter
    /// scale, and invalidates prior checkpoints.
    ///
    /// # Errors
    /// Rejects scale mismatches against loaded adapters, scales outside finite `0.0..=2.0`,
    /// identity hashing failures, or Vulkan reset errors.
    pub fn reset_with_adapter_scales(
        &mut self,
        multi_rank_scale: Option<f32>,
    ) -> Result<(), Error> {
        let resolved_mr = self.validate_adapter_scale_update(multi_rank_scale)?;
        let new_identity = compute_executor_identity(&self.base_provenance_sha256, resolved_mr)?;
        self.reset()?;
        self.multi_rank_scale = resolved_mr;
        self.identity_sha256 = new_identity;
        self.provenance_nonce = self.provenance_nonce.wrapping_add(1);
        Ok(())
    }

    fn validate_adapter_scale_update(
        &self,
        multi_rank_scale: Option<f32>,
    ) -> Result<Option<f32>, Error> {
        if self.multi_rank_scale.is_none() {
            return Err(Error::InvalidInput(
                "adapter scale update requires a loaded multi-rank LoRA adapter".to_owned(),
            ));
        }
        match (self.multi_rank_scale, multi_rank_scale) {
            (Some(_), Some(scale)) => {
                MultiRankAdapter::validate_scale(scale)
                    .map_err(|err| Error::InvalidInput(err.to_string()))?;
                Ok(Some(MultiRankAdapter::normalize_scale(scale)))
            }
            (Some(_), None) => Err(Error::InvalidInput(
                "multi-rank adapter requires a multi-rank scale".to_owned(),
            )),
            (None, None) => Ok(None),
            (None, Some(_)) => Err(Error::InvalidInput(
                "multi-rank scale provided without a loaded multi-rank adapter".to_owned(),
            )),
        }
    }

    /// Captures a host-side snapshot of all mutable GPU recurrent and KV-cache state.
    ///
    /// # Errors
    /// Returns [`Error::DeviceFault`], [`Error::InvalidInput`], or [`Error::Vulkan`].
    pub fn checkpoint(&self) -> Result<VulkanBatchedCheckpoint, Error> {
        let mut snapshot = VulkanBatchedCheckpoint {
            executor_identity_sha256: String::new(),
            provenance_nonce: 0,
            position: 0,
            mrope_positions: [0; 4],
            kv_generation: 0,
            kv_prefix_watermark: 0,
            layers: Vec::new(),
        };
        self.checkpoint_into(&mut snapshot)?;
        Ok(snapshot)
    }

    /// Captures a host-side snapshot of all mutable GPU state into `target`, reusing compatible
    /// host allocations when possible.
    ///
    /// # Errors
    /// Returns [`Error::DeviceFault`], [`Error::InvalidInput`], or [`Error::Vulkan`].
    pub fn checkpoint_into(&self, target: &mut VulkanBatchedCheckpoint) -> Result<(), Error> {
        if self.device.is_faulted() || !self.state_consistent {
            return Err(Error::DeviceFault(
                "cannot checkpoint faulted or inconsistent Vulkan graph executor".to_owned(),
            ));
        }
        self.device.ensure_pipelined_idle()?;
        let kv_width = self.geometry.kv_width();
        let reuse = target.layers.len() == self.layers.len();
        let same_lineage = reuse
            && target.executor_identity_sha256 == self.identity_sha256
            && target.provenance_nonce == self.provenance_nonce;
        let reusable_tokens = if same_lineage {
            self.kv_tracker
                .intact_prefix_for(target.kv_generation, target.kv_prefix_watermark)
                .min(target.position)
                .min(self.position)
        } else {
            0
        };
        if !reuse {
            target.layers.clear();
            target.layers.reserve(self.layers.len());
        }

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            match &layer.attention {
                GpuLayerAttention::Recurrent {
                    conv_state,
                    conv_state_elems,
                    matrix_state,
                    matrix_state_elems,
                    ..
                } => {
                    if reuse
                        && let Some(LayerStateCheckpoint::Recurrent {
                            convolution,
                            matrix,
                        }) = target.layers.get_mut(layer_idx)
                    {
                        convolution.resize(*conv_state_elems, 0.0_f32);
                        if *conv_state_elems > 0 {
                            self.device.read_buffer_f32(*conv_state, convolution)?;
                        }
                        matrix.resize(*matrix_state_elems, 0.0_f32);
                        if *matrix_state_elems > 0 {
                            self.device.read_buffer_f32(*matrix_state, matrix)?;
                        }
                    } else {
                        let convolution = if *conv_state_elems > 0 {
                            let mut buf = vec![0.0_f32; *conv_state_elems];
                            self.device.read_buffer_f32(*conv_state, &mut buf)?;
                            buf
                        } else {
                            Vec::new()
                        };
                        let matrix = if *matrix_state_elems > 0 {
                            let mut buf = vec![0.0_f32; *matrix_state_elems];
                            self.device.read_buffer_f32(*matrix_state, &mut buf)?;
                            buf
                        } else {
                            Vec::new()
                        };
                        let snap = LayerStateCheckpoint::Recurrent {
                            convolution,
                            matrix,
                        };
                        if reuse {
                            target.layers[layer_idx] = snap;
                        } else {
                            target.layers.push(snap);
                        }
                    }
                }
                GpuLayerAttention::Full {
                    k_cache, v_cache, ..
                } => {
                    self.read_full_kv_checkpoint_layer(
                        *k_cache,
                        *v_cache,
                        layer_idx,
                        kv_width,
                        reusable_tokens,
                        reuse,
                        target,
                    )?;
                }
            }
        }

        target.executor_identity_sha256.clear();
        target
            .executor_identity_sha256
            .push_str(&self.identity_sha256);
        target.provenance_nonce = self.provenance_nonce;
        target.position = self.position;
        target.mrope_positions = self.mrope_positions;
        target.kv_generation = self.kv_tracker.current_gen;
        target.kv_prefix_watermark = self.position;
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Incrementally downloads only uncaptured `[reusable_tokens..position)` KV cache slices when lineage is intact."
    )]
    fn read_full_kv_checkpoint_layer(
        &self,
        k_cache: BufferId,
        v_cache: BufferId,
        layer_idx: usize,
        kv_width: usize,
        reusable_tokens: usize,
        reuse: bool,
        target: &mut VulkanBatchedCheckpoint,
    ) -> Result<(), Error> {
        let words_per_tok = match self.config.kv_precision {
            VulkanKvPrecision::Fp16 => kv_width / 2,
            VulkanKvPrecision::Fp32 => kv_width,
        };
        let total_words = self.position.saturating_mul(words_per_tok);
        let expected_prev_words = target.position.saturating_mul(words_per_tok);
        let reuse_layer = reuse
            && matches!(
                target.layers.get(layer_idx),
                Some(LayerStateCheckpoint::Full {
                    precision,
                    width,
                    keys_words,
                    values_words,
                }) if *precision == self.config.kv_precision
                    && *width == kv_width
                    && keys_words.len() == expected_prev_words
                    && values_words.len() == expected_prev_words
            );
        let start_tokens = if reuse_layer {
            reusable_tokens.min(self.position)
        } else {
            0
        };
        let start_words = start_tokens.saturating_mul(words_per_tok);
        let delta_words = total_words.saturating_sub(start_words);
        let start_bytes = u32_slice_bytes(start_words)?;
        let delta_bytes = usize::try_from(u32_slice_bytes(delta_words)?)
            .map_err(|_| Error::ResourceLimit("KV checkpoint byte length overflow".to_owned()))?;

        let mut raw_k = Vec::new();
        let mut raw_v = Vec::new();
        if delta_bytes > 0 {
            raw_k.resize(delta_bytes, 0_u8);
            raw_v.resize(delta_bytes, 0_u8);
            self.device
                .read_buffer_bytes_at(k_cache, start_bytes, &mut raw_k)?;
            self.device
                .read_buffer_bytes_at(v_cache, start_bytes, &mut raw_v)?;
        }

        if reuse
            && let Some(LayerStateCheckpoint::Full {
                precision,
                width,
                keys_words,
                values_words,
            }) = target.layers.get_mut(layer_idx)
        {
            *precision = self.config.kv_precision;
            *width = kv_width;
            keys_words.resize(total_words, 0_u32);
            values_words.resize(total_words, 0_u32);
            for (dst, chunk) in keys_words[start_words..]
                .iter_mut()
                .zip(raw_k.as_chunks::<4>().0)
            {
                *dst = u32::from_le_bytes(*chunk);
            }
            for (dst, chunk) in values_words[start_words..]
                .iter_mut()
                .zip(raw_v.as_chunks::<4>().0)
            {
                *dst = u32::from_le_bytes(*chunk);
            }
        } else {
            let mut keys_words = vec![0_u32; total_words];
            let mut values_words = vec![0_u32; total_words];
            for (dst, chunk) in keys_words.iter_mut().zip(raw_k.as_chunks::<4>().0) {
                *dst = u32::from_le_bytes(*chunk);
            }
            for (dst, chunk) in values_words.iter_mut().zip(raw_v.as_chunks::<4>().0) {
                *dst = u32::from_le_bytes(*chunk);
            }
            let snap = LayerStateCheckpoint::Full {
                precision: self.config.kv_precision,
                width: kv_width,
                keys_words,
                values_words,
            };
            if reuse {
                target.layers[layer_idx] = snap;
            } else {
                target.layers.push(snap);
            }
        }
        Ok(())
    }

    fn validate_checkpoint_for_restore(
        &self,
        checkpoint: &VulkanBatchedCheckpoint,
    ) -> Result<(), Error> {
        #[expect(
            clippy::suspicious_operation_groupings,
            reason = "Checkpoint stores the target executor digest in `executor_identity_sha256` while executor holds `identity_sha256`."
        )]
        let provenance_matches = checkpoint.executor_identity_sha256 == self.identity_sha256
            && checkpoint.provenance_nonce == self.provenance_nonce;
        if !provenance_matches {
            return Err(Error::InvalidInput(
                "checkpoint executor identity does not match this VulkanBatchedGraphExecutor"
                    .to_owned(),
            ));
        }
        if checkpoint.position > self.resolved_context_limit {
            return Err(Error::InvalidInput(format!(
                "checkpoint position {} exceeds context limit {}",
                checkpoint.position, self.resolved_context_limit
            )));
        }
        if checkpoint.layers.len() != self.layers.len() {
            return Err(Error::InvalidInput(
                "checkpoint layer count mismatch".to_owned(),
            ));
        }
        let kv_width = self.geometry.kv_width();
        for (layer, snap) in self.layers.iter().zip(&checkpoint.layers) {
            match (&layer.attention, snap) {
                (
                    GpuLayerAttention::Recurrent {
                        conv_state_elems,
                        matrix_state_elems,
                        ..
                    },
                    LayerStateCheckpoint::Recurrent {
                        convolution,
                        matrix,
                    },
                ) => {
                    if convolution.len() != *conv_state_elems || matrix.len() != *matrix_state_elems
                    {
                        return Err(Error::InvalidInput(
                            "checkpoint recurrent state size mismatch".to_owned(),
                        ));
                    }
                    ensure_finite_f32(
                        convolution,
                        "checkpoint conv state contains non-finite value",
                    )?;
                    ensure_finite_f32(matrix, "checkpoint matrix state contains non-finite value")?;
                }
                (
                    GpuLayerAttention::Full { .. },
                    LayerStateCheckpoint::Full {
                        precision,
                        width,
                        keys_words,
                        values_words,
                    },
                ) => {
                    if *precision != self.config.kv_precision || *width != kv_width {
                        return Err(Error::InvalidInput(
                            "checkpoint KV precision or width mismatch".to_owned(),
                        ));
                    }
                    let words_per_tok = match precision {
                        VulkanKvPrecision::Fp16 => kv_width / 2,
                        VulkanKvPrecision::Fp32 => kv_width,
                    };
                    let expected_words = checkpoint.position.saturating_mul(words_per_tok);
                    if keys_words.len() != expected_words || values_words.len() != expected_words {
                        return Err(Error::InvalidInput(
                            "checkpoint KV word count mismatch for position".to_owned(),
                        ));
                    }
                }
                _ => {
                    return Err(Error::InvalidInput(
                        "checkpoint layer kind mismatch".to_owned(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Restores GPU state from a checkpoint produced by this executor.
    ///
    /// When the executor's live device KV cache prefix up to `checkpoint.position` has not been
    /// overwritten since the checkpoint's KV lineage generation, this performs an append-only
    /// cursor rollback that uploads only recurrent GDN/conv states and preserves GPU KV caches
    /// without host->GPU KV copying. Whenever earlier KV positions have been overwritten by a
    /// diverging branch (or the state is inconsistent), it falls back to full KV cache restoration.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`], [`Error::DeviceFault`], or [`Error::Vulkan`] if the
    /// checkpoint belongs to a different model/executor configuration or fails validation.
    pub fn restore(&mut self, checkpoint: &VulkanBatchedCheckpoint) -> Result<(), Error> {
        if self.device.is_faulted() {
            return Err(Error::DeviceFault(
                "Vulkan batched executor faulted and cannot restore".to_owned(),
            ));
        }
        self.device.ensure_pipelined_idle()?;
        self.staging_bank_in_flight = [false; 2];
        self.next_staging_bank = 0;
        self.validate_checkpoint_for_restore(checkpoint)?;

        let can_skip_kv = self.kv_tracker.can_skip_kv_restore(
            self.state_consistent,
            checkpoint.kv_generation,
            checkpoint.kv_prefix_watermark,
            checkpoint.position,
        );

        for (layer, snap) in self.layers.iter().zip(&checkpoint.layers) {
            match (&layer.attention, snap) {
                (
                    GpuLayerAttention::Recurrent {
                        conv_state,
                        conv_state_elems,
                        matrix_state,
                        matrix_state_elems,
                        ..
                    },
                    LayerStateCheckpoint::Recurrent {
                        convolution,
                        matrix,
                    },
                ) => {
                    if *conv_state_elems > 0 {
                        self.device.write_buffer_f32(*conv_state, convolution)?;
                    }
                    if *matrix_state_elems > 0 {
                        self.device.write_buffer_f32(*matrix_state, matrix)?;
                    }
                }
                (
                    GpuLayerAttention::Full {
                        k_cache, v_cache, ..
                    },
                    LayerStateCheckpoint::Full {
                        keys_words,
                        values_words,
                        ..
                    },
                ) if !can_skip_kv => {
                    let k_bytes = self.device.buffer_size_bytes(*k_cache)?;
                    let v_bytes = self.device.buffer_size_bytes(*v_cache)?;
                    self.device.fill_buffer_zero(*k_cache, 0, k_bytes)?;
                    self.device.fill_buffer_zero(*v_cache, 0, v_bytes)?;
                    if !keys_words.is_empty() {
                        self.device.write_buffer_u32_at(*k_cache, 0, keys_words)?;
                        self.device.write_buffer_u32_at(*v_cache, 0, values_words)?;
                    }
                }
                _ => {}
            }
        }
        if can_skip_kv {
            // Live `[0..checkpoint.position)` KV slots remain intact on GPU; keep `high_water`
            // intact so any later write below `high_water` narrows `valid_prefixes` accurately.
            self.kv_fast_restores = self.kv_fast_restores.saturating_add(1);
        } else {
            self.kv_tracker.on_full_kv_restore(checkpoint.position);
            self.kv_full_restores = self.kv_full_restores.saturating_add(1);
        }
        self.position = checkpoint.position;
        self.mrope_positions = checkpoint.mrope_positions;
        self.state_consistent = true;
        Ok(())
    }

    /// Returns the active GPU KV cache lineage generation counter.
    #[must_use]
    pub const fn kv_generation(&self) -> u64 {
        self.kv_tracker.current_gen
    }

    /// Returns the contiguous intact token watermark recorded for the current GPU KV lineage.
    #[must_use]
    pub const fn kv_prefix_watermark(&self) -> usize {
        self.kv_tracker.high_water
    }

    /// Returns how many `restore` calls reused the live GPU KV prefix without copying KV buffers.
    #[must_use]
    pub const fn kv_fast_restores(&self) -> u64 {
        self.kv_fast_restores
    }

    /// Returns how many `restore` calls fell back to full GPU KV buffer upload.
    #[must_use]
    pub const fn kv_full_restores(&self) -> u64 {
        self.kv_full_restores
    }

    /// Returns the total number of Vulkan queue submissions executed across this executor's life.
    #[must_use]
    pub const fn submits_completed(&self) -> u64 {
        self.submits_completed
    }

    /// Returns the active TDR chunking policy.
    #[must_use]
    pub const fn tdr_policy(&self) -> TdrChunkPolicy {
        self.config.tdr_policy
    }

    /// Returns the active integer projection kernel path (`IntegerDotPacked4x8` or `PortableIntegerFallback`).
    #[must_use]
    pub const fn kernel_path(&self) -> ProjectionKernelPath {
        self.kernel_path
    }

    /// Returns the current committed token sequence position.
    #[must_use]
    pub const fn position(&self) -> usize {
        self.position
    }

    /// Returns the model output vocabulary extent.
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.geometry.vocabulary
    }

    /// Returns the resolved context limit in tokens.
    #[must_use]
    pub const fn context_limit(&self) -> usize {
        self.resolved_context_limit
    }

    /// Returns the number of active transformer layers loaded onto GPU.
    #[must_use]
    pub const fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Returns whether this executor uses `PinnedSwift` rounding/FMA arithmetic.
    #[must_use]
    pub const fn uses_pinned_arithmetic(&self) -> bool {
        self.resolved_pinned_arithmetic
    }

    /// Returns the deterministic SHA-256 identity binding model, TDR policy, shaders, adapters, and GPU.
    #[must_use]
    pub fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }

    /// Returns a reference to the compiled GPU layer pipelines.
    #[must_use]
    pub const fn pipelines(&self) -> &VulkanLayerPipelines {
        &self.pipelines
    }

    /// Returns `true` when the underlying Adreno Vulkan device has latched a terminal fault.
    #[must_use]
    pub const fn is_faulted(&self) -> bool {
        self.device.is_faulted()
    }

    fn step_token_impl(
        &mut self,
        token: u32,
        logits_out: Option<&mut Vec<f32>>,
    ) -> Result<(), Error> {
        self.validate_execution_ready(&[token])?;
        self.step_tile_pipelined_impl(&[token], logits_out, true, &mut || Ok(()))
    }

    fn validate_execution_ready(&self, tokens: &[u32]) -> Result<(), Error> {
        if self.device.is_faulted() {
            return Err(Error::DeviceFault(
                "Vulkan batched executor faulted and requires terminal teardown".to_owned(),
            ));
        }
        if !self.state_consistent {
            return Err(Error::InvalidInput(
                "executor state is inconsistent after failed evaluation and requires reset or restore"
                    .to_owned(),
            ));
        }
        if tokens.is_empty() {
            return Err(Error::InvalidInput(
                "at least one token is required".to_owned(),
            ));
        }
        for &token in tokens {
            let token_idx = usize::try_from(token)
                .map_err(|_| Error::InvalidInput("token index overflow".to_owned()))?;
            if token_idx >= self.geometry.vocabulary {
                return Err(Error::InvalidInput(format!(
                    "token {token} out of vocabulary bounds {}",
                    self.geometry.vocabulary
                )));
            }
        }
        if self
            .position
            .checked_add(tokens.len())
            .is_none_or(|end| end > self.resolved_context_limit)
        {
            return Err(Error::InvalidInput(format!(
                "context limit {} exhausted for {} tokens at position {}",
                self.resolved_context_limit,
                tokens.len(),
                self.position
            )));
        }
        Ok(())
    }

    fn drain_pipelined_session(&mut self) -> Result<(), Error> {
        if !self.device.has_unwaited_slots() {
            self.staging_bank_in_flight = [false; 2];
            return Ok(());
        }
        let res = self.device.finish_pipelined_session();
        self.staging_bank_in_flight = [false; 2];
        if res.is_err() {
            self.device.abort_pipelined_session();
            self.state_consistent = false;
            self.kv_tracker.invalidate();
        }
        res
    }

    fn ingest_tokens_batched_with_check(
        &mut self,
        tokens: &[u32],
        check: &mut impl FnMut() -> Result<(), Error>,
    ) -> Result<(), Error> {
        self.ingest_tokens_and_maybe_eval_last_with_check(tokens, None, check)
    }

    fn ingest_tokens_and_maybe_eval_last_with_check(
        &mut self,
        tokens: &[u32],
        mut logits_out: Option<&mut Vec<f32>>,
        check: &mut impl FnMut() -> Result<(), Error>,
    ) -> Result<(), Error> {
        self.validate_execution_ready(tokens)?;
        let tile_cap = self.scratch.allocated_tile_tokens.max(1);
        let total_tiles = tokens.len().div_ceil(tile_cap);
        for (tile_idx, tile) in tokens.chunks(tile_cap).enumerate() {
            if let Err(err) = check() {
                self.drain_pipelined_session()?;
                return Err(err);
            }
            let is_last_tile = tile_idx + 1 == total_tiles;
            let tile_logits = if is_last_tile {
                logits_out.take()
            } else {
                None
            };
            let drain_at_tile_end = is_last_tile || tile_logits.is_some();
            self.step_tile_pipelined_impl(tile, tile_logits, drain_at_tile_end, check)?;
            if let Err(err) = check() {
                self.drain_pipelined_session()?;
                return Err(err);
            }
        }
        self.drain_pipelined_session()
    }

    fn write_tile_embeddings(&mut self, tokens: &[u32], hidden_dst: BufferId) -> Result<(), Error> {
        if tokens.len() == 1 {
            let token_idx = usize::try_from(tokens[0])
                .map_err(|_| Error::InvalidInput("token index overflow".to_owned()))?;
            row_into(&self.embedding, token_idx, &mut self.host_embed_scratch)
                .map_err(|err| Error::InvalidInput(err.to_string()))?;
            ensure_finite_f32(
                &self.host_embed_scratch,
                "token embedding contains non-finite value",
            )?;
            self.device
                .write_buffer_f32(hidden_dst, &self.host_embed_scratch)
        } else {
            self.host_embed_tile_scratch.clear();
            for &token in tokens {
                let token_idx = usize::try_from(token)
                    .map_err(|_| Error::InvalidInput("token index overflow".to_owned()))?;
                row_into(&self.embedding, token_idx, &mut self.host_embed_scratch)
                    .map_err(|err| Error::InvalidInput(err.to_string()))?;
                ensure_finite_f32(
                    &self.host_embed_scratch,
                    "token embedding contains non-finite value",
                )?;
                self.host_embed_tile_scratch
                    .extend_from_slice(&self.host_embed_scratch);
            }
            self.device
                .write_buffer_f32(hidden_dst, &self.host_embed_tile_scratch)
        }
    }

    fn prepare_tile_host_inputs(&mut self, tokens: &[u32]) -> Result<PreparedTileUpload, Error> {
        if !self.device.has_unwaited_slots() {
            self.staging_bank_in_flight = [false; 2];
        }
        let staged_bank = if self.device.has_unwaited_slots() {
            let bank = self.next_staging_bank;
            self.next_staging_bank ^= 1;
            if self.staging_bank_in_flight[bank] {
                self.drain_pipelined_session()?;
                None
            } else {
                Some(bank)
            }
        } else {
            None
        };

        let (hidden_dst, rope_dst) = if let Some(bank) = staged_bank {
            (
                self.scratch.hidden_staging[bank],
                self.scratch.rope_freqs_staging[bank],
            )
        } else {
            (self.scratch.hidden, self.scratch.rope_freqs)
        };

        self.write_tile_embeddings(tokens, hidden_dst)?;

        let has_full_layers = self
            .layers
            .iter()
            .any(|layer| matches!(layer.attention, GpuLayerAttention::Full { .. }));
        if has_full_layers {
            let rope = self.geometry.rope;
            let rope_end = tokens.len() * rope;
            for (t, dst_slice) in self.host_rope_freqs[..rope_end]
                .chunks_exact_mut(rope)
                .enumerate()
            {
                let tok_mrope = [
                    self.mrope_positions[0] + t,
                    self.mrope_positions[1] + t,
                    self.mrope_positions[2] + t,
                    self.mrope_positions[3],
                ];
                fill_rope_frequencies(&self.geometry, tok_mrope, dst_slice)?;
            }
            self.device
                .write_buffer_f32(rope_dst, &self.host_rope_freqs[..rope_end])?;
        }

        Ok(PreparedTileUpload {
            staged_bank,
            has_full_layers,
        })
    }

    fn step_tile_pipelined_impl(
        &mut self,
        tokens: &[u32],
        logits_out: Option<&mut Vec<f32>>,
        drain_at_tile_end: bool,
        check: &mut impl FnMut() -> Result<(), Error>,
    ) -> Result<(), Error> {
        let t_len = tokens.len();
        if let Some(dst) = logits_out {
            self.run_tile_gpu_pipelined_into(tokens, Some(dst), drain_at_tile_end, check)?;
            if let Err(err) = ensure_finite_f32(dst, "GPU batched graph produced non-finite logits")
            {
                dst.clear();
                self.state_consistent = false;
                self.kv_tracker.invalidate();
                return Err(err);
            }
        } else {
            self.run_tile_gpu_pipelined_into(tokens, None, drain_at_tile_end, check)?;
        }
        self.commit_tile_position(t_len);
        Ok(())
    }

    fn commit_tile_position(&mut self, t_len: usize) {
        self.position += t_len;
        for slot in &mut self.mrope_positions[..3] {
            *slot += t_len;
        }
    }

    fn run_tile_gpu_into(
        &mut self,
        tokens: &[u32],
        logits_out: Option<&mut Vec<f32>>,
        check: &mut impl FnMut() -> Result<(), Error>,
    ) -> Result<(), Error> {
        self.run_tile_gpu_pipelined_into(tokens, logits_out, true, check)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Coordinates host-staged tile upload, layered TDR command chunks, fused final-token logits, and optional queue drain."
    )]
    fn run_tile_gpu_pipelined_into(
        &mut self,
        tokens: &[u32],
        logits_out: Option<&mut Vec<f32>>,
        drain_at_tile_end: bool,
        check: &mut impl FnMut() -> Result<(), Error>,
    ) -> Result<(), Error> {
        let t_len = tokens.len();
        let prepared = self.prepare_tile_host_inputs(tokens)?;

        let chunks = self
            .config
            .tdr_policy
            .layer_chunk_ranges_for_tokens(self.layers.len(), t_len)?;
        let need_logits = logits_out.is_some();
        let total_layers = self.layers.len();
        let min_align = usize::try_from(self.device.min_storage_buffer_offset_alignment().max(1))
            .unwrap_or(256);

        let had_unwaited_before = self.device.has_unwaited_slots();
        let mut mutated_gpu = false;
        let run_result = (|| -> Result<(), Error> {
            let record_tile_prologue = |this: &mut Self| -> Result<(), Error> {
                if let Some(bank) = prepared.staged_bank {
                    let total_embed_u32 =
                        u32::try_from(t_len * this.geometry.embedding).map_err(|_| {
                            Error::ResourceLimit("staged embedding exceeds u32".to_owned())
                        })?;
                    this.record_copy_slice(
                        this.scratch.hidden_staging[bank],
                        0,
                        this.scratch.hidden,
                        total_embed_u32,
                        true,
                    )?;
                    if prepared.has_full_layers {
                        let total_rope_u32 =
                            u32::try_from(t_len * this.geometry.rope).map_err(|_| {
                                Error::ResourceLimit("staged rope exceeds u32".to_owned())
                            })?;
                        this.record_copy_slice(
                            this.scratch.rope_freqs_staging[bank],
                            0,
                            this.scratch.rope_freqs,
                            total_rope_u32,
                            false,
                        )?;
                    }
                    this.staging_bank_in_flight[bank] = true;
                }
                if this.transforms.inverse_embedding {
                    this.record_inverse_embedding_transform_tokens(t_len)?;
                }
                Ok(())
            };

            if chunks.is_empty() {
                if need_logits || prepared.staged_bank.is_some() {
                    self.batch_recorder.clear();
                    record_tile_prologue(self)?;
                    if need_logits {
                        self.record_final_logits(t_len)?;
                    }
                    if !self.batch_recorder.is_empty() {
                        mutated_gpu = true;
                        let submits = self.device.submit_pipelined_batches_with_max_stages(
                            &self.batch_recorder,
                            self.config.tdr_policy.max_stages_per_submit,
                            drain_at_tile_end,
                        )?;
                        self.submits_completed = self.submits_completed.saturating_add(submits);
                        if !self.device.has_unwaited_slots() {
                            self.staging_bank_in_flight = [false; 2];
                        }
                    }
                }
            } else {
                for (chunk_idx, range) in chunks.iter().enumerate() {
                    if chunk_idx > 0 {
                        check()?;
                    }
                    self.batch_recorder.clear();
                    if chunk_idx == 0 {
                        record_tile_prologue(self)?;
                    }
                    for layer_idx in range.clone() {
                        let mode = if layer_idx + 1 == total_layers {
                            if !need_logits {
                                FinalLayerMode::IngestOnly
                            } else if t_len > 1 {
                                FinalLayerMode::LastTokenLogits
                            } else {
                                FinalLayerMode::FullAllTokens
                            }
                        } else {
                            FinalLayerMode::FullAllTokens
                        };
                        self.record_layer_tokens(layer_idx, t_len, mode, min_align)?;
                    }
                    let is_final_chunk = chunk_idx + 1 == chunks.len();
                    if is_final_chunk && need_logits {
                        self.record_final_logits(t_len)?;
                    }
                    if !self.batch_recorder.is_empty() {
                        if !mutated_gpu {
                            self.kv_tracker.begin_write(self.position, t_len);
                            mutated_gpu = true;
                        }
                        let fence_this_chunk = drain_at_tile_end && is_final_chunk;
                        let submits = self.device.submit_pipelined_batches_with_max_stages(
                            &self.batch_recorder,
                            self.config.tdr_policy.max_stages_per_submit,
                            fence_this_chunk,
                        )?;
                        self.submits_completed = self.submits_completed.saturating_add(submits);
                        if !self.device.has_unwaited_slots() {
                            self.staging_bank_in_flight = [false; 2];
                        }
                    }
                }
            }
            if drain_at_tile_end || need_logits {
                self.device.finish_pipelined_session()?;
                self.staging_bank_in_flight = [false; 2];
            }

            if let Some(dst) = logits_out {
                dst.resize(self.geometry.vocabulary, 0.0_f32);
                self.device.read_buffer_f32(self.scratch.logits, dst)?;
            }
            Ok(())
        })();

        if run_result.is_err() {
            self.device.abort_pipelined_session();
            self.staging_bank_in_flight = [false; 2];
            if mutated_gpu || had_unwaited_before {
                self.state_consistent = false;
                self.kv_tracker.invalidate();
            }
            return run_result;
        }

        Ok(())
    }

    fn record_inverse_embedding_transform_tokens(&mut self, tokens: usize) -> Result<(), Error> {
        let len = self.geometry.embedding;
        let len_u32 = u32::try_from(len)
            .map_err(|_| Error::ResourceLimit("embedding length exceeds u32".to_owned()))?;
        let tokens_u32 = u32::try_from(tokens)
            .map_err(|_| Error::ResourceLimit("token count exceeds u32".to_owned()))?;
        let total_len_u32 = u32::try_from(tokens * len)
            .map_err(|_| Error::ResourceLimit("batched embedding length exceeds u32".to_owned()))?;

        if self.transforms.block >= 2 && len.is_multiple_of(self.transforms.block) {
            let block_u32 = u32::try_from(self.transforms.block)
                .map_err(|_| Error::InvalidInput("Hadamard block exceeds u32".to_owned()))?;
            let blk_cnt = u32::try_from(len / self.transforms.block)
                .map_err(|_| Error::ResourceLimit("Hadamard grid exceeds u32".to_owned()))?;
            let norm_bits = hadamard_norm_bits(block_u32);
            if tokens == 1 {
                self.batch_recorder.push_barrier(
                    self.pipelines.proj.hadamard,
                    &[
                        BufferSpan::whole(self.scratch.hidden),
                        BufferSpan::whole(self.scratch.transformed),
                    ],
                    &[block_u32, norm_bits],
                    checked_grid(&self.device, [blk_cnt, 1, 1])?,
                );
            } else {
                self.batch_recorder.push_barrier(
                    self.pipelines.hadamard_batched,
                    &[
                        BufferSpan::whole(self.scratch.hidden),
                        BufferSpan::whole(self.scratch.transformed),
                    ],
                    &[block_u32, norm_bits, blk_cnt, tokens_u32],
                    checked_grid(&self.device, [blk_cnt, tokens_u32, 1])?,
                );
            }
            if let Some(&sign_buf) = self.transforms.sign_buffers.get(&len) {
                self.record_mul_sign_tokens(
                    self.scratch.transformed,
                    sign_buf,
                    self.scratch.hidden,
                    len_u32,
                    tokens_u32,
                )?;
            } else {
                let copy_grid =
                    checked_grid(&self.device, [total_len_u32.div_ceil(WORKGROUP_X), 1, 1])?;
                self.batch_recorder.push_barrier(
                    self.pipelines.copy_slice,
                    &[
                        BufferSpan::whole(self.scratch.transformed),
                        BufferSpan::whole(self.scratch.hidden),
                    ],
                    &[0, total_len_u32],
                    copy_grid,
                );
            }
        } else if let Some(&sign_buf) = self.transforms.sign_buffers.get(&len) {
            self.record_mul_sign_tokens(
                self.scratch.hidden,
                sign_buf,
                self.scratch.transformed,
                len_u32,
                tokens_u32,
            )?;
            let copy_grid =
                checked_grid(&self.device, [total_len_u32.div_ceil(WORKGROUP_X), 1, 1])?;
            self.batch_recorder.push_barrier(
                self.pipelines.copy_slice,
                &[
                    BufferSpan::whole(self.scratch.transformed),
                    BufferSpan::whole(self.scratch.hidden),
                ],
                &[0, total_len_u32],
                copy_grid,
            );
        }
        Ok(())
    }

    fn record_mul_sign_tokens(
        &mut self,
        in_buf: BufferId,
        sign_buf: BufferId,
        out_buf: BufferId,
        len_u32: u32,
        tokens_u32: u32,
    ) -> Result<(), Error> {
        self.record_mul_sign_tokens_with_barrier(
            in_buf, sign_buf, out_buf, len_u32, tokens_u32, true,
        )
    }

    fn record_mul_sign_tokens_with_barrier(
        &mut self,
        in_buf: BufferId,
        sign_buf: BufferId,
        out_buf: BufferId,
        len_u32: u32,
        tokens_u32: u32,
        barrier: bool,
    ) -> Result<(), Error> {
        if tokens_u32 == 1 {
            let elem_grid = checked_grid(&self.device, [len_u32.div_ceil(WORKGROUP_X), 1, 1])?;
            self.batch_recorder.push(
                self.pipelines.mul_f32_elemwise,
                &[
                    BufferSpan::whole(in_buf),
                    BufferSpan::whole(sign_buf),
                    BufferSpan::whole(out_buf),
                ],
                &[len_u32],
                elem_grid,
                barrier,
            );
        } else {
            let elem_grid =
                checked_grid(&self.device, [len_u32.div_ceil(WORKGROUP_X), tokens_u32, 1])?;
            self.batch_recorder.push(
                self.pipelines.mul_f32_elemwise_batched,
                &[
                    BufferSpan::whole(in_buf),
                    BufferSpan::whole(sign_buf),
                    BufferSpan::whole(out_buf),
                ],
                &[len_u32, tokens_u32],
                elem_grid,
                barrier,
            );
        }
        Ok(())
    }

    fn record_copy_slice(
        &mut self,
        src: BufferId,
        src_offset: u32,
        dst: BufferId,
        len: u32,
        barrier: bool,
    ) -> Result<(), Error> {
        let grid = checked_grid(&self.device, [len.div_ceil(WORKGROUP_X), 1, 1])?;
        self.batch_recorder.push(
            self.pipelines.copy_slice,
            &[BufferSpan::whole(src), BufferSpan::whole(dst)],
            &[src_offset, len],
            grid,
            barrier,
        );
        Ok(())
    }

    fn record_last_token_recurrent_slice(
        &mut self,
        tokens: usize,
        gdn_out_buf: BufferId,
        val_width: usize,
    ) -> Result<(), Error> {
        let embed = self.geometry.embedding;
        let last = tokens - 1;
        let last_embed_off = u32::try_from(last * embed).map_err(|_| {
            Error::ResourceLimit("last token embedding offset exceeds u32".to_owned())
        })?;
        let embed_u32 = u32::try_from(embed)
            .map_err(|_| Error::ResourceLimit("embedding exceeds u32".to_owned()))?;
        let last_val_off = u32::try_from(last * val_width)
            .map_err(|_| Error::ResourceLimit("last token value offset exceeds u32".to_owned()))?;
        let val_u32 = u32::try_from(val_width)
            .map_err(|_| Error::ResourceLimit("val_width exceeds u32".to_owned()))?;

        // `proj_a`, `proj_c`, `proj_d` are unused after GDN scan and disjoint from `gdn_out_buf` (`proj_e`).
        self.record_copy_slice(
            self.scratch.norm,
            last_embed_off,
            self.scratch.proj_a,
            embed_u32,
            true,
        )?;
        self.record_copy_slice(
            self.scratch.hidden,
            last_embed_off,
            self.scratch.proj_c,
            embed_u32,
            false,
        )?;
        self.record_copy_slice(
            gdn_out_buf,
            last_val_off,
            self.scratch.proj_d,
            val_u32,
            false,
        )?;

        self.record_copy_slice(self.scratch.proj_a, 0, self.scratch.norm, embed_u32, true)?;
        self.record_copy_slice(
            self.scratch.proj_c,
            0,
            self.scratch.hidden,
            embed_u32,
            false,
        )?;
        self.record_copy_slice(self.scratch.proj_d, 0, gdn_out_buf, val_u32, false)
    }

    fn record_last_token_full_slice(&mut self, tokens: usize) -> Result<(), Error> {
        let embed = self.geometry.embedding;
        let rope = self.geometry.rope;
        let last = tokens - 1;
        let last_embed_off = u32::try_from(last * embed).map_err(|_| {
            Error::ResourceLimit("last token embedding offset exceeds u32".to_owned())
        })?;
        let embed_u32 = u32::try_from(embed)
            .map_err(|_| Error::ResourceLimit("embedding exceeds u32".to_owned()))?;
        let last_rope_off = u32::try_from(last * rope)
            .map_err(|_| Error::ResourceLimit("last token rope offset exceeds u32".to_owned()))?;
        let rope_u32 =
            u32::try_from(rope).map_err(|_| Error::ResourceLimit("rope exceeds u32".to_owned()))?;

        // `proj_b`, `proj_c`, `proj_e` are dead after K/V append and before Q-only evaluation.
        self.record_copy_slice(
            self.scratch.norm,
            last_embed_off,
            self.scratch.proj_b,
            embed_u32,
            true,
        )?;
        self.record_copy_slice(
            self.scratch.hidden,
            last_embed_off,
            self.scratch.proj_c,
            embed_u32,
            false,
        )?;
        self.record_copy_slice(
            self.scratch.rope_freqs,
            last_rope_off,
            self.scratch.proj_e,
            rope_u32,
            false,
        )?;

        self.record_copy_slice(self.scratch.proj_b, 0, self.scratch.norm, embed_u32, true)?;
        self.record_copy_slice(
            self.scratch.proj_c,
            0,
            self.scratch.hidden,
            embed_u32,
            false,
        )?;
        self.record_copy_slice(
            self.scratch.proj_e,
            0,
            self.scratch.rope_freqs,
            rope_u32,
            false,
        )
    }

    fn record_layer_tokens(
        &mut self,
        layer_idx: usize,
        tokens: usize,
        mode: FinalLayerMode,
        min_align: usize,
    ) -> Result<(), Error> {
        let embed_u32 = u32::try_from(self.geometry.embedding)
            .map_err(|_| Error::ResourceLimit("embedding exceeds u32".to_owned()))?;
        let tokens_u32 = u32::try_from(tokens)
            .map_err(|_| Error::ResourceLimit("tokens exceed u32".to_owned()))?;
        let eps_bits = self.geometry.epsilon.to_bits();
        let attn_norm_w = self.layers[layer_idx].attn_norm_w;
        let post_norm_w = self.layers[layer_idx].post_norm_w;

        if layer_idx == 0 {
            self.batch_recorder.push_barrier(
                self.pipelines.rms_norm,
                &[
                    BufferSpan::whole(self.scratch.hidden),
                    BufferSpan::whole(attn_norm_w),
                    BufferSpan::whole(self.scratch.norm),
                ],
                &[embed_u32, eps_bits],
                checked_grid(&self.device, [1, tokens_u32, 1])?,
            );
        } else {
            self.batch_recorder.push_barrier(
                self.pipelines.add_rms_norm,
                &[
                    BufferSpan::whole(self.scratch.hidden),
                    BufferSpan::whole(self.scratch.delta),
                    BufferSpan::whole(attn_norm_w),
                    BufferSpan::whole(self.scratch.norm),
                ],
                &[embed_u32, eps_bits],
                checked_grid(&self.device, [1, tokens_u32, 1])?,
            );
        }

        let is_recurrent = matches!(
            self.layers[layer_idx].attention,
            GpuLayerAttention::Recurrent { .. }
        );
        if is_recurrent {
            self.record_recurrent_attention_tokens(layer_idx, tokens, mode, min_align)?;
        } else {
            self.record_full_attention_tokens(layer_idx, tokens, mode, min_align)?;
        }

        if mode == FinalLayerMode::IngestOnly {
            return Ok(());
        }

        let post_tokens = if mode == FinalLayerMode::LastTokenLogits {
            1
        } else {
            tokens
        };
        let post_tokens_u32 = u32::try_from(post_tokens)
            .map_err(|_| Error::ResourceLimit("post_tokens exceed u32".to_owned()))?;
        self.batch_recorder.push_barrier(
            self.pipelines.add_rms_norm,
            &[
                BufferSpan::whole(self.scratch.hidden),
                BufferSpan::whole(self.scratch.delta),
                BufferSpan::whole(post_norm_w),
                BufferSpan::whole(self.scratch.norm),
            ],
            &[embed_u32, eps_bits],
            checked_grid(&self.device, [1, post_tokens_u32, 1])?,
        );

        self.record_ffn_tokens(layer_idx, post_tokens)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Encodes GPU-resident GDN recurrent layer stages preserving exact Qwen35 conv1d+SiLU, ClampedL2, decay/beta, delta-rule, and gated norm sequence for single, batched, and final-token-logits tiles."
    )]
    fn record_recurrent_attention_tokens(
        &mut self,
        layer_idx: usize,
        tokens: usize,
        mode: FinalLayerMode,
        min_align: usize,
    ) -> Result<(), Error> {
        let (
            qkv,
            gate,
            beta,
            alpha,
            convolution_w,
            time_bias,
            ssm_a,
            norm_w,
            output_w,
            conv_state,
            conv_state_elems,
            matrix_state,
        ) = match &self.layers[layer_idx].attention {
            GpuLayerAttention::Recurrent {
                qkv,
                gate,
                beta,
                alpha,
                convolution_w,
                time_bias,
                ssm_a,
                norm_w,
                output,
                conv_state,
                conv_state_elems,
                matrix_state,
                ..
            } => (
                *qkv,
                *gate,
                *beta,
                *alpha,
                *convolution_w,
                *time_bias,
                *ssm_a,
                *norm_w,
                *output,
                *conv_state,
                *conv_state_elems,
                *matrix_state,
            ),
            GpuLayerAttention::Full { .. } => {
                return Err(Error::InvalidInput("expected recurrent layer".to_owned()));
            }
        };

        let proj_qkv_buf = self.scratch.proj_a;
        let proj_gate_buf = self.scratch.proj_b;
        let proj_beta_buf = self.scratch.proj_c;
        let proj_alpha_buf = self.scratch.proj_d;

        if mode == FinalLayerMode::FullAllTokens {
            self.record_projection_group_tokens(
                self.scratch.norm,
                &[
                    (qkv, proj_qkv_buf),
                    (gate, proj_gate_buf),
                    (beta, proj_beta_buf),
                    (alpha, proj_alpha_buf),
                ],
                tokens,
            )?;
        } else {
            self.record_projection_group_tokens(
                self.scratch.norm,
                &[
                    (qkv, proj_qkv_buf),
                    (beta, proj_beta_buf),
                    (alpha, proj_alpha_buf),
                ],
                tokens,
            )?;
        }

        let channels = self.geometry.convolution_width();
        let channels_u32 = u32::try_from(channels)
            .map_err(|_| Error::ResourceLimit("conv channels exceed u32".to_owned()))?;
        let kernel_u32 = u32::try_from(self.geometry.convolution)
            .map_err(|_| Error::ResourceLimit("conv kernel exceeds u32".to_owned()))?;
        let key_width = self.geometry.key_width();
        let val_width = self.geometry.value_width();
        let v_heads_u32 = u32::try_from(self.geometry.ssm_values)
            .map_err(|_| Error::ResourceLimit("ssm_values exceeds u32".to_owned()))?;
        let k_heads_u32 = u32::try_from(self.geometry.ssm_keys)
            .map_err(|_| Error::ResourceLimit("ssm_keys exceeds u32".to_owned()))?;
        let ssm_head_u32 = u32::try_from(self.geometry.ssm_head)
            .map_err(|_| Error::ResourceLimit("ssm_head exceeds u32".to_owned()))?;
        let qk_heads_u32 = u32::try_from((2 * key_width) / self.geometry.ssm_head)
            .map_err(|_| Error::ResourceLimit("qk_heads exceeds u32".to_owned()))?;
        let tokens_u32 = u32::try_from(tokens)
            .map_err(|_| Error::ResourceLimit("tokens exceed u32".to_owned()))?;
        let dim_f32 = u16::try_from(self.geometry.ssm_head).map_or(1.0_f32, f32::from);
        let gdn_scale_bits = (1.0_f32 / dim_f32.sqrt()).to_bits();

        let active_conv_state = if conv_state_elems > 0 {
            conv_state
        } else {
            self.scratch.dummy_state
        };

        let (gdn_out_buf, gdn_normed_buf) = if self.geometry.convolution <= 8
            && ((tokens == 1 && self.geometry.ssm_head <= 128)
                || (tokens > 1 && self.geometry.ssm_head <= 256))
        {
            let batched_out_buf = self.scratch.proj_e;
            let conv_grid = checked_grid(&self.device, [channels_u32.div_ceil(WORKGROUP_X), 1, 1])?;
            let gates_grid = checked_grid(
                &self.device,
                [v_heads_u32.div_ceil(WORKGROUP_X), tokens_u32, 1],
            )?;
            let l2_grid = checked_grid(&self.device, [1, qk_heads_u32, tokens_u32])?;
            let gdn_grid = checked_grid(&self.device, [ssm_head_u32, v_heads_u32, 1])?;

            self.batch_recorder.push_barrier(
                self.pipelines.conv1d_silu_batched_inplace,
                &[
                    BufferSpan::whole(proj_qkv_buf),
                    BufferSpan::whole(active_conv_state),
                    BufferSpan::whole(convolution_w),
                ],
                &[channels_u32, kernel_u32, tokens_u32],
                conv_grid,
            );
            self.batch_recorder.push_concurrent(
                self.pipelines.gdn_gates_batched,
                &[
                    BufferSpan::whole(proj_alpha_buf),
                    BufferSpan::whole(proj_beta_buf),
                    BufferSpan::whole(time_bias),
                    BufferSpan::whole(ssm_a),
                ],
                &[v_heads_u32, tokens_u32],
                gates_grid,
            );
            self.batch_recorder.push_barrier(
                self.pipelines.l2_norm_batched_strided,
                &[BufferSpan::whole(proj_qkv_buf)],
                &[
                    qk_heads_u32,
                    ssm_head_u32,
                    self.geometry.epsilon.to_bits(),
                    channels_u32,
                    tokens_u32,
                ],
                l2_grid,
            );
            self.batch_recorder.push_barrier(
                self.pipelines.gdn_scan_batched,
                &[
                    BufferSpan::whole(matrix_state),
                    BufferSpan::whole(proj_qkv_buf),
                    BufferSpan::whole(proj_alpha_buf),
                    BufferSpan::whole(proj_beta_buf),
                    BufferSpan::whole(batched_out_buf),
                ],
                &[
                    v_heads_u32,
                    k_heads_u32,
                    ssm_head_u32,
                    gdn_scale_bits,
                    tokens_u32,
                ],
                gdn_grid,
            );
            (batched_out_buf, self.scratch.perm_tmp)
        } else if tokens == 1 {
            let conv_out_buf = self.scratch.proj_e;
            self.batch_recorder.push_barrier(
                self.pipelines.conv1d_silu,
                &[
                    BufferSpan::whole(proj_qkv_buf),
                    BufferSpan::whole(active_conv_state),
                    BufferSpan::whole(convolution_w),
                    BufferSpan::whole(conv_out_buf),
                ],
                &[channels_u32, kernel_u32],
                checked_grid(&self.device, [channels_u32.div_ceil(WORKGROUP_X), 1, 1])?,
            );
            self.batch_recorder.push_concurrent(
                self.pipelines.gdn_gates,
                &[
                    BufferSpan::whole(proj_alpha_buf),
                    BufferSpan::whole(proj_beta_buf),
                    BufferSpan::whole(time_bias),
                    BufferSpan::whole(ssm_a),
                ],
                &[v_heads_u32],
                checked_grid(&self.device, [v_heads_u32.div_ceil(WORKGROUP_X), 1, 1])?,
            );
            self.batch_recorder.push_barrier(
                self.pipelines.l2_norm,
                &[BufferSpan::whole(conv_out_buf)],
                &[qk_heads_u32, ssm_head_u32, self.geometry.epsilon.to_bits()],
                checked_grid(&self.device, [1, qk_heads_u32, 1])?,
            );

            let (q_span, k_span, v_span) =
                self.resolve_gdn_qkv_spans(conv_out_buf, key_width, val_width, min_align)?;
            let out_buf = proj_qkv_buf;
            self.batch_recorder.push_barrier(
                self.pipelines.gdn_step,
                &[
                    BufferSpan::whole(matrix_state),
                    q_span,
                    k_span,
                    v_span,
                    BufferSpan::whole(proj_alpha_buf),
                    BufferSpan::whole(proj_beta_buf),
                    BufferSpan::whole(out_buf),
                ],
                &[v_heads_u32, k_heads_u32, ssm_head_u32, gdn_scale_bits],
                checked_grid(&self.device, [ssm_head_u32, v_heads_u32, 1])?,
            );
            (out_buf, conv_out_buf)
        } else {
            // `perm_tmp` holds single-token [q, k, v] during the serial recurrence while `proj_e` accumulates `[T, val_width]`.
            let conv_tmp_buf = self.scratch.perm_tmp;
            let batched_out_buf = self.scratch.proj_e;
            let conv_grid = checked_grid(&self.device, [channels_u32.div_ceil(WORKGROUP_X), 1, 1])?;
            let l2_grid = checked_grid(&self.device, [1, qk_heads_u32, 1])?;
            let gdn_grid = checked_grid(&self.device, [ssm_head_u32, v_heads_u32, 1])?;

            for t in 0..tokens {
                let qkv_off = u32::try_from(t * channels)
                    .map_err(|_| Error::ResourceLimit("GDN qkv offset exceeds u32".to_owned()))?;
                let tok_u32 = u32::try_from(t)
                    .map_err(|_| Error::ResourceLimit("GDN token index exceeds u32".to_owned()))?;
                self.batch_recorder.push_barrier(
                    self.pipelines.conv1d_silu_offset,
                    &[
                        BufferSpan::whole(proj_qkv_buf),
                        BufferSpan::whole(active_conv_state),
                        BufferSpan::whole(convolution_w),
                        BufferSpan::whole(conv_tmp_buf),
                    ],
                    &[channels_u32, kernel_u32, qkv_off],
                    conv_grid,
                );
                if t == 0 {
                    self.batch_recorder.push_concurrent(
                        self.pipelines.gdn_gates_batched,
                        &[
                            BufferSpan::whole(proj_alpha_buf),
                            BufferSpan::whole(proj_beta_buf),
                            BufferSpan::whole(time_bias),
                            BufferSpan::whole(ssm_a),
                        ],
                        &[v_heads_u32, tokens_u32],
                        checked_grid(
                            &self.device,
                            [v_heads_u32.div_ceil(WORKGROUP_X), tokens_u32, 1],
                        )?,
                    );
                }
                self.batch_recorder.push_barrier(
                    self.pipelines.l2_norm,
                    &[BufferSpan::whole(conv_tmp_buf)],
                    &[qk_heads_u32, ssm_head_u32, self.geometry.epsilon.to_bits()],
                    l2_grid,
                );
                self.batch_recorder.push_barrier(
                    self.pipelines.gdn_step_batched_token,
                    &[
                        BufferSpan::whole(matrix_state),
                        BufferSpan::whole(conv_tmp_buf),
                        BufferSpan::whole(proj_alpha_buf),
                        BufferSpan::whole(proj_beta_buf),
                        BufferSpan::whole(batched_out_buf),
                    ],
                    &[
                        v_heads_u32,
                        k_heads_u32,
                        ssm_head_u32,
                        gdn_scale_bits,
                        tok_u32,
                    ],
                    gdn_grid,
                );
            }
            (batched_out_buf, self.scratch.perm_tmp)
        };

        if mode == FinalLayerMode::IngestOnly {
            return Ok(());
        }

        let out_tokens = if mode == FinalLayerMode::LastTokenLogits && tokens > 1 {
            self.record_last_token_recurrent_slice(tokens, gdn_out_buf, val_width)?;
            self.record_projection_group_tokens(self.scratch.norm, &[(gate, proj_gate_buf)], 1)?;
            1
        } else {
            tokens
        };

        let norm_rows = u32::try_from(out_tokens * self.geometry.ssm_values)
            .map_err(|_| Error::ResourceLimit("GDN norm rows exceed u32".to_owned()))?;
        self.batch_recorder.push_barrier(
            self.pipelines.rms_norm,
            &[
                BufferSpan::whole(gdn_out_buf),
                BufferSpan::whole(norm_w),
                BufferSpan::whole(gdn_normed_buf),
            ],
            &[ssm_head_u32, self.geometry.epsilon.to_bits()],
            checked_grid(&self.device, [1, norm_rows, 1])?,
        );

        let total_val_u32 = u32::try_from(out_tokens * val_width)
            .map_err(|_| Error::ResourceLimit("batched value_width exceeds u32".to_owned()))?;
        let gated_buf = gdn_out_buf;
        let swiglu_pipe = if self.resolved_pinned_arithmetic {
            self.pipelines.proj.swiglu_pinned
        } else {
            self.pipelines.proj.swiglu_unpinned
        };
        self.batch_recorder.push_barrier(
            swiglu_pipe,
            &[
                BufferSpan::whole(proj_gate_buf),
                BufferSpan::whole(gdn_normed_buf),
                BufferSpan::whole(gated_buf),
            ],
            &[total_val_u32],
            checked_grid(&self.device, [total_val_u32.div_ceil(WORKGROUP_X), 1, 1])?,
        );

        self.record_projection_group_tokens(
            gated_buf,
            &[(output_w, self.scratch.delta)],
            out_tokens,
        )
    }

    fn resolve_gdn_qkv_spans(
        &mut self,
        conv_out_buf: BufferId,
        key_width: usize,
        val_width: usize,
        min_align: usize,
    ) -> Result<(BufferSpan, BufferSpan, BufferSpan), Error> {
        let q_bytes = f32_slice_bytes(key_width)?;
        let v_bytes = f32_slice_bytes(val_width)?;
        let conv_whole = BufferSpan::whole(conv_out_buf);
        let q_span = BufferSpan::subspan(conv_whole, 0, q_bytes)?;
        let k_byte_off = q_bytes;
        let v_byte_off = k_byte_off
            .checked_add(q_bytes)
            .ok_or_else(|| Error::ResourceLimit("GDN qkv byte offset overflow".to_owned()))?;
        let k_usize_off = usize::try_from(k_byte_off)
            .map_err(|_| Error::ResourceLimit("k offset overflow".to_owned()))?;
        let v_usize_off = usize::try_from(v_byte_off)
            .map_err(|_| Error::ResourceLimit("v offset overflow".to_owned()))?;

        let (k_span, v_span) =
            if k_usize_off.is_multiple_of(min_align) && v_usize_off.is_multiple_of(min_align) {
                (
                    BufferSpan::subspan(conv_whole, k_byte_off, q_bytes)?,
                    BufferSpan::subspan(conv_whole, v_byte_off, v_bytes)?,
                )
            } else {
                let key_u32 = u32::try_from(key_width)
                    .map_err(|_| Error::ResourceLimit("key_width exceeds u32".to_owned()))?;
                let val_u32 = u32::try_from(val_width)
                    .map_err(|_| Error::ResourceLimit("value_width exceeds u32".to_owned()))?;
                let two_key_u32 = u32::try_from(2 * key_width)
                    .map_err(|_| Error::ResourceLimit("2*key_width exceeds u32".to_owned()))?;
                self.batch_recorder.push_barrier(
                    self.pipelines.copy_slice,
                    &[
                        BufferSpan::whole(conv_out_buf),
                        BufferSpan::whole(self.scratch.gdn_k),
                    ],
                    &[key_u32, key_u32],
                    checked_grid(&self.device, [key_u32.div_ceil(WORKGROUP_X), 1, 1])?,
                );
                self.batch_recorder.push_concurrent(
                    self.pipelines.copy_slice,
                    &[
                        BufferSpan::whole(conv_out_buf),
                        BufferSpan::whole(self.scratch.gdn_v),
                    ],
                    &[two_key_u32, val_u32],
                    checked_grid(&self.device, [val_u32.div_ceil(WORKGROUP_X), 1, 1])?,
                );
                (
                    BufferSpan::whole(self.scratch.gdn_k),
                    BufferSpan::whole(self.scratch.gdn_v),
                )
            };
        Ok((q_span, k_span, v_span))
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Encodes GPU-resident full-attention layer preserving exact Q/Gate deinterleave, Q/K RMSNorm+RoPE(f16), V(f16), KV append, #40 causal attention (with budget sub-tiling), sigmoid gate, and output projection."
    )]
    fn record_full_attention_tokens(
        &mut self,
        layer_idx: usize,
        tokens: usize,
        mode: FinalLayerMode,
        min_align: usize,
    ) -> Result<(), Error> {
        let (query_w, key_w, value_w, output_w, query_norm_w, key_norm_w, k_cache, v_cache) =
            match &self.layers[layer_idx].attention {
                GpuLayerAttention::Full {
                    query,
                    key,
                    value,
                    output,
                    query_norm_w,
                    key_norm_w,
                    k_cache,
                    v_cache,
                } => (
                    *query,
                    *key,
                    *value,
                    *output,
                    *query_norm_w,
                    *key_norm_w,
                    *k_cache,
                    *v_cache,
                ),
                GpuLayerAttention::Recurrent { .. } => {
                    return Err(Error::InvalidInput(
                        "expected full attention layer".to_owned(),
                    ));
                }
            };

        let q_gate_buf = self.scratch.proj_a;
        let key_buf = self.scratch.proj_b;
        let val_buf = self.scratch.proj_c;
        let query_buf = self.scratch.proj_d;
        let gate_buf = self.scratch.proj_e;

        let num_heads_u32 = u32::try_from(self.geometry.heads)
            .map_err(|_| Error::ResourceLimit("heads exceeds u32".to_owned()))?;
        let kv_heads_u32 = u32::try_from(self.geometry.kv_heads)
            .map_err(|_| Error::ResourceLimit("kv_heads exceeds u32".to_owned()))?;
        let head_dim_u32 = u32::try_from(self.geometry.head)
            .map_err(|_| Error::ResourceLimit("head exceeds u32".to_owned()))?;
        let rope_u32 = u32::try_from(self.geometry.rope)
            .map_err(|_| Error::ResourceLimit("rope exceeds u32".to_owned()))?;
        let kv_width = self.geometry.kv_width();
        let kv_width_u32 = u32::try_from(kv_width)
            .map_err(|_| Error::ResourceLimit("kv_width exceeds u32".to_owned()))?;
        let tokens_u32 = u32::try_from(tokens)
            .map_err(|_| Error::ResourceLimit("tokens exceed u32".to_owned()))?;
        let total_kv_u32 = u32::try_from(tokens * kv_width)
            .map_err(|_| Error::ResourceLimit("batched kv_width exceeds u32".to_owned()))?;
        let pos_u32 = u32::try_from(self.position)
            .map_err(|_| Error::ResourceLimit("position exceeds u32".to_owned()))?;
        let eps_bits = self.geometry.epsilon.to_bits();
        let q_width = self.geometry.q_width();
        let q_width_u32 = u32::try_from(q_width)
            .map_err(|_| Error::ResourceLimit("q_width exceeds u32".to_owned()))?;

        let (q_tokens, q_past_offset) = if mode == FinalLayerMode::FullAllTokens {
            self.record_projection_group_tokens(
                self.scratch.norm,
                &[(query_w, q_gate_buf), (key_w, key_buf), (value_w, val_buf)],
                tokens,
            )?;
            self.batch_recorder.push_barrier(
                self.pipelines.rms_norm,
                &[
                    BufferSpan::whole(key_buf),
                    BufferSpan::whole(key_norm_w),
                    BufferSpan::whole(key_buf),
                ],
                &[head_dim_u32, eps_bits],
                checked_grid(&self.device, [1, tokens_u32 * kv_heads_u32, 1])?,
            );
            if self.config.kv_precision == VulkanKvPrecision::Fp32 {
                self.batch_recorder.push_concurrent(
                    self.pipelines.round_f16,
                    &[BufferSpan::whole(val_buf)],
                    &[total_kv_u32],
                    checked_grid(&self.device, [total_kv_u32.div_ceil(WORKGROUP_X), 1, 1])?,
                );
            }
            self.record_rope_tokens(
                key_buf,
                kv_heads_u32,
                head_dim_u32,
                rope_u32,
                tokens_u32,
                true,
            )?;
            self.record_kv_append_tokens(
                key_buf,
                val_buf,
                k_cache,
                v_cache,
                kv_width_u32,
                pos_u32,
                tokens_u32,
            )?;
            (tokens, 0)
        } else {
            self.record_projection_group_tokens(
                self.scratch.norm,
                &[(key_w, key_buf), (value_w, val_buf)],
                tokens,
            )?;
            self.batch_recorder.push_barrier(
                self.pipelines.rms_norm,
                &[
                    BufferSpan::whole(key_buf),
                    BufferSpan::whole(key_norm_w),
                    BufferSpan::whole(key_buf),
                ],
                &[head_dim_u32, eps_bits],
                checked_grid(&self.device, [1, tokens_u32 * kv_heads_u32, 1])?,
            );
            if self.config.kv_precision == VulkanKvPrecision::Fp32 {
                self.batch_recorder.push_concurrent(
                    self.pipelines.round_f16,
                    &[BufferSpan::whole(val_buf)],
                    &[total_kv_u32],
                    checked_grid(&self.device, [total_kv_u32.div_ceil(WORKGROUP_X), 1, 1])?,
                );
            }
            self.record_rope_tokens(
                key_buf,
                kv_heads_u32,
                head_dim_u32,
                rope_u32,
                tokens_u32,
                true,
            )?;
            self.record_kv_append_tokens(
                key_buf,
                val_buf,
                k_cache,
                v_cache,
                kv_width_u32,
                pos_u32,
                tokens_u32,
            )?;
            if mode == FinalLayerMode::IngestOnly {
                return Ok(());
            }
            self.record_last_token_full_slice(tokens)?;
            self.record_projection_group_tokens(self.scratch.norm, &[(query_w, q_gate_buf)], 1)?;
            (1, tokens - 1)
        };

        let q_tokens_u32 = u32::try_from(q_tokens)
            .map_err(|_| Error::ResourceLimit("q_tokens exceed u32".to_owned()))?;
        if q_tokens == 1 {
            self.batch_recorder.push_barrier(
                self.pipelines.deinterleave_q_gate,
                &[
                    BufferSpan::whole(q_gate_buf),
                    BufferSpan::whole(query_buf),
                    BufferSpan::whole(gate_buf),
                ],
                &[num_heads_u32, head_dim_u32],
                checked_grid(&self.device, [q_width_u32.div_ceil(WORKGROUP_X), 1, 1])?,
            );
        } else {
            self.batch_recorder.push_barrier(
                self.pipelines.deinterleave_q_gate_batched,
                &[
                    BufferSpan::whole(q_gate_buf),
                    BufferSpan::whole(query_buf),
                    BufferSpan::whole(gate_buf),
                ],
                &[num_heads_u32, head_dim_u32, q_tokens_u32],
                checked_grid(
                    &self.device,
                    [q_width_u32.div_ceil(WORKGROUP_X), q_tokens_u32, 1],
                )?,
            );
        }

        self.batch_recorder.push_barrier(
            self.pipelines.rms_norm,
            &[
                BufferSpan::whole(query_buf),
                BufferSpan::whole(query_norm_w),
                BufferSpan::whole(query_buf),
            ],
            &[head_dim_u32, eps_bits],
            checked_grid(&self.device, [1, q_tokens_u32 * num_heads_u32, 1])?,
        );

        self.record_rope_tokens(
            query_buf,
            num_heads_u32,
            head_dim_u32,
            rope_u32,
            q_tokens_u32,
            true,
        )?;

        let attended_buf = q_gate_buf;
        self.record_full_attention_causal(
            query_buf,
            k_cache,
            v_cache,
            attended_buf,
            q_tokens,
            q_past_offset,
            q_width,
            min_align,
        )?;

        let total_q_u32 = u32::try_from(q_tokens * q_width)
            .map_err(|_| Error::ResourceLimit("batched q_width exceeds u32".to_owned()))?;
        self.batch_recorder.push_barrier(
            self.pipelines.mul_sigmoid,
            &[BufferSpan::whole(attended_buf), BufferSpan::whole(gate_buf)],
            &[total_q_u32],
            checked_grid(&self.device, [total_q_u32.div_ceil(WORKGROUP_X), 1, 1])?,
        );

        self.record_projection_group_tokens(
            attended_buf,
            &[(output_w, self.scratch.delta)],
            q_tokens,
        )
    }

    fn record_rope_tokens(
        &mut self,
        vec_buf: BufferId,
        heads_u32: u32,
        head_dim_u32: u32,
        rope_u32: u32,
        tokens_u32: u32,
        barrier: bool,
    ) -> Result<(), Error> {
        let bufs = [
            BufferSpan::whole(vec_buf),
            BufferSpan::whole(self.scratch.rope_freqs),
        ];
        if tokens_u32 == 1 {
            let grid = checked_grid(
                &self.device,
                [head_dim_u32.div_ceil(WORKGROUP_X), heads_u32, 1],
            )?;
            self.batch_recorder.push(
                self.pipelines.rope,
                &bufs,
                &[heads_u32, head_dim_u32, rope_u32, 1],
                grid,
                barrier,
            );
        } else {
            let grid = checked_grid(
                &self.device,
                [head_dim_u32.div_ceil(WORKGROUP_X), heads_u32, tokens_u32],
            )?;
            self.batch_recorder.push(
                self.pipelines.rope_batched,
                &bufs,
                &[heads_u32, head_dim_u32, rope_u32, 1, tokens_u32],
                grid,
                barrier,
            );
        }
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Dispatches #40 causal attention across full or sub-tiled query chunks within scratch and TDR MAC limits."
    )]
    fn record_full_attention_causal(
        &mut self,
        query_buf: BufferId,
        k_cache: BufferId,
        v_cache: BufferId,
        attended_buf: BufferId,
        tokens: usize,
        past_token_offset: usize,
        q_width: usize,
        min_align: usize,
    ) -> Result<(), Error> {
        let mut q_start = 0_usize;
        while q_start < tokens {
            let (chosen_sub_q, desc, plan) =
                self.select_attention_sub_chunk(past_token_offset + q_start, tokens - q_start)?;
            self.dispatch_attention_sub_chunk(
                query_buf,
                k_cache,
                v_cache,
                attended_buf,
                q_start,
                chosen_sub_q,
                tokens,
                q_width,
                min_align,
                &desc,
                plan.path,
            )?;
            q_start += chosen_sub_q;
        }
        Ok(())
    }

    fn select_attention_sub_chunk(
        &self,
        q_start: usize,
        max_sub_q: usize,
    ) -> Result<(usize, AttentionDescriptor, crate::AttentionTilePlan), Error> {
        let num_heads_u32 = u32::try_from(self.geometry.heads)
            .map_err(|_| Error::ResourceLimit("heads exceeds u32".to_owned()))?;
        let kv_heads_u32 = u32::try_from(self.geometry.kv_heads)
            .map_err(|_| Error::ResourceLimit("kv_heads exceeds u32".to_owned()))?;
        let head_dim_u32 = u32::try_from(self.geometry.head)
            .map_err(|_| Error::ResourceLimit("head exceeds u32".to_owned()))?;
        let head_f32 = u16::try_from(self.geometry.head).map_or(1.0_f32, f32::from);
        let scale = 1.0_f32 / head_f32.sqrt();
        let effective_score_budget = self
            .config
            .scratch_budget_bytes
            .min(self.scratch.attn_score_bytes.saturating_mul(2));

        let mut sub_q = max_sub_q;
        loop {
            let sub_q_u32 = u32::try_from(sub_q)
                .map_err(|_| Error::ResourceLimit("sub_q exceeds u32".to_owned()))?;
            let sub_past = self.position + q_start;
            let sub_past_u32 = u32::try_from(sub_past)
                .map_err(|_| Error::ResourceLimit("past_len exceeds u32".to_owned()))?;
            let sub_total_u32 = u32::try_from(sub_past + sub_q)
                .map_err(|_| Error::ResourceLimit("total_seq exceeds u32".to_owned()))?;
            let desc = AttentionDescriptor {
                query_len: sub_q_u32,
                total_seq: sub_total_u32,
                past_len: sub_past_u32,
                num_heads: num_heads_u32,
                num_kv_heads: kv_heads_u32,
                head_dim: head_dim_u32,
                scale,
                causal: true,
            };
            match select_attention_plan(
                &desc,
                self.device.capabilities(),
                self.config.attention_strategy,
                effective_score_budget,
            ) {
                Ok(plan) => return Ok((sub_q, desc, plan)),
                Err(_) if sub_q > 1 => {
                    sub_q /= 2;
                }
                Err(err) => return Err(err),
            }
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Resolves direct, aligned-subspan, or GPU-copied Q/output spans for one causal attention sub-chunk."
    )]
    fn prepare_attention_chunk_spans(
        &mut self,
        query_buf: BufferId,
        attended_buf: BufferId,
        q_start: usize,
        chosen_sub_q: usize,
        tokens: usize,
        q_width: usize,
        min_align: usize,
    ) -> Result<AttentionChunkSpans, Error> {
        if q_start == 0 && chosen_sub_q == tokens {
            return Ok(AttentionChunkSpans {
                q_in: BufferSpan::whole(query_buf),
                o_out: BufferSpan::whole(attended_buf),
                copy_back: None,
            });
        }
        let byte_off = f32_slice_bytes(q_start * q_width)?;
        let slice_bytes = f32_slice_bytes(chosen_sub_q * q_width)?;
        let aligned = usize::try_from(byte_off).is_ok_and(|off| off.is_multiple_of(min_align));
        if aligned {
            return Ok(AttentionChunkSpans {
                q_in: BufferSpan::subspan(BufferSpan::whole(query_buf), byte_off, slice_bytes)?,
                o_out: BufferSpan::subspan(BufferSpan::whole(attended_buf), byte_off, slice_bytes)?,
                copy_back: None,
            });
        }
        let elem_off = u32::try_from(q_start * q_width)
            .map_err(|_| Error::ResourceLimit("attention slice offset exceeds u32".to_owned()))?;
        let slice_len = u32::try_from(chosen_sub_q * q_width)
            .map_err(|_| Error::ResourceLimit("attention slice length exceeds u32".to_owned()))?;
        self.batch_recorder.push_barrier(
            self.pipelines.copy_slice_offset,
            &[
                BufferSpan::whole(query_buf),
                BufferSpan::whole(self.scratch.transformed),
            ],
            &[elem_off, 0, slice_len],
            checked_grid(&self.device, [slice_len.div_ceil(WORKGROUP_X), 1, 1])?,
        );
        Ok(AttentionChunkSpans {
            q_in: BufferSpan::whole(self.scratch.transformed),
            o_out: BufferSpan::whole(self.scratch.sign_tmp),
            copy_back: Some((elem_off, slice_len)),
        })
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Dispatches #40 fused or 3-stage causal attention and optional slice copy-back for one query sub-chunk."
    )]
    fn dispatch_attention_sub_chunk(
        &mut self,
        query_buf: BufferId,
        k_cache: BufferId,
        v_cache: BufferId,
        attended_buf: BufferId,
        q_start: usize,
        chosen_sub_q: usize,
        tokens: usize,
        q_width: usize,
        min_align: usize,
        desc: &AttentionDescriptor,
        path: AdrenoAttentionPath,
    ) -> Result<(), Error> {
        let k_attn_span = BufferSpan::whole(k_cache);
        let v_attn_span = BufferSpan::whole(v_cache);
        let (fused_pipe, qk_pipe, pv_pipe) = match self.config.kv_precision {
            VulkanKvPrecision::Fp16 => (
                self.pipelines.attn_fused_f16prob_f16kv,
                self.pipelines.attn_qk_scores_f16kv,
                self.pipelines.attn_pv_accum_f16kv,
            ),
            VulkanKvPrecision::Fp32 => (
                self.pipelines.attn_fused_f16prob,
                self.pipelines.attn_qk_scores,
                self.pipelines.attn_pv_accum,
            ),
        };

        let spans = self.prepare_attention_chunk_spans(
            query_buf,
            attended_buf,
            q_start,
            chosen_sub_q,
            tokens,
            q_width,
            min_align,
        )?;
        let q_in = spans.q_in;
        let o_out = spans.o_out;
        let copy_back = spans.copy_back;
        let attn_push = attention_push_constants(desc);
        match path {
            AdrenoAttentionPath::Fused => {
                let choice = select_fused_attention_dispatch(
                    desc,
                    self.config.kv_precision,
                    self.device
                        .capabilities()
                        .compute_limits
                        .max_compute_shared_memory_size,
                );
                let pipe = match choice.kind {
                    FusedAttentionKernelKind::GqaGrouped6PrefillF16Kv => {
                        self.pipelines.attn_fused_gqa_prefill_f16kv
                    }
                    FusedAttentionKernelKind::PerHead => fused_pipe,
                };
                self.batch_recorder.push_barrier(
                    pipe,
                    &[q_in, k_attn_span, v_attn_span, o_out],
                    &attn_push,
                    checked_grid(&self.device, choice.grid)?,
                );
            }
            AdrenoAttentionPath::Decomposed3Stage => {
                self.dispatch_decomposed_3stage_attention(
                    qk_pipe,
                    pv_pipe,
                    q_in,
                    k_attn_span,
                    v_attn_span,
                    o_out,
                    &attn_push,
                    desc,
                )?;
            }
        }

        if let Some((elem_off, slice_len)) = copy_back {
            self.batch_recorder.push_barrier(
                self.pipelines.copy_slice_offset,
                &[
                    BufferSpan::whole(self.scratch.sign_tmp),
                    BufferSpan::whole(attended_buf),
                ],
                &[0, elem_off, slice_len],
                checked_grid(&self.device, [slice_len.div_ceil(WORKGROUP_X), 1, 1])?,
            );
        }
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Records the 3-stage QK^T -> FP16-rounded Softmax -> PV causal attention sequence for one query sub-chunk."
    )]
    fn dispatch_decomposed_3stage_attention(
        &mut self,
        qk_pipe: PipelineId,
        pv_pipe: PipelineId,
        q_in: BufferSpan,
        k_attn_span: BufferSpan,
        v_attn_span: BufferSpan,
        o_out: BufferSpan,
        attn_push: &[u32; 8],
        desc: &AttentionDescriptor,
    ) -> Result<(), Error> {
        let num_heads_u32 = desc.num_heads;
        let head_dim_u32 = desc.head_dim;
        self.batch_recorder.push_barrier(
            qk_pipe,
            &[
                q_in,
                k_attn_span,
                BufferSpan::whole(self.scratch.attn_scores),
            ],
            attn_push,
            checked_grid(
                &self.device,
                [
                    desc.total_seq.div_ceil(WORKGROUP_X),
                    num_heads_u32,
                    desc.query_len,
                ],
            )?,
        );
        self.batch_recorder.push_barrier(
            self.pipelines.attn_softmax_f16,
            &[
                BufferSpan::whole(self.scratch.attn_scores),
                BufferSpan::whole(self.scratch.attn_probs),
            ],
            attn_push,
            checked_grid(&self.device, [1, num_heads_u32, desc.query_len])?,
        );
        self.batch_recorder.push_barrier(
            pv_pipe,
            &[
                BufferSpan::whole(self.scratch.attn_probs),
                v_attn_span,
                o_out,
            ],
            attn_push,
            checked_grid(
                &self.device,
                [
                    head_dim_u32.div_ceil(WORKGROUP_X),
                    num_heads_u32,
                    desc.query_len,
                ],
            )?,
        );
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Appends single-token or batched K/V rows to device FP16/F32 caches concurrently."
    )]
    fn record_kv_append_tokens(
        &mut self,
        key_buf: BufferId,
        val_buf: BufferId,
        k_cache: BufferId,
        v_cache: BufferId,
        kv_width_u32: u32,
        pos_u32: u32,
        tokens_u32: u32,
    ) -> Result<(), Error> {
        match (self.config.kv_precision, tokens_u32 == 1) {
            (VulkanKvPrecision::Fp16, true) => {
                let half_w = kv_width_u32 / 2;
                let grid = checked_grid(&self.device, [half_w.div_ceil(WORKGROUP_X), 1, 1])?;
                self.batch_recorder.push_barrier(
                    self.pipelines.kv_append_f16,
                    &[BufferSpan::whole(key_buf), BufferSpan::whole(k_cache)],
                    &[kv_width_u32, pos_u32],
                    grid,
                );
                self.batch_recorder.push_concurrent(
                    self.pipelines.kv_append_f16,
                    &[BufferSpan::whole(val_buf), BufferSpan::whole(v_cache)],
                    &[kv_width_u32, pos_u32],
                    grid,
                );
            }
            (VulkanKvPrecision::Fp16, false) => {
                let half_w = kv_width_u32 / 2;
                let grid =
                    checked_grid(&self.device, [half_w.div_ceil(WORKGROUP_X), tokens_u32, 1])?;
                self.batch_recorder.push_barrier(
                    self.pipelines.kv_append_batched_f16,
                    &[BufferSpan::whole(key_buf), BufferSpan::whole(k_cache)],
                    &[kv_width_u32, pos_u32, tokens_u32],
                    grid,
                );
                self.batch_recorder.push_concurrent(
                    self.pipelines.kv_append_batched_f16,
                    &[BufferSpan::whole(val_buf), BufferSpan::whole(v_cache)],
                    &[kv_width_u32, pos_u32, tokens_u32],
                    grid,
                );
            }
            (VulkanKvPrecision::Fp32, true) => {
                let grid = checked_grid(&self.device, [kv_width_u32.div_ceil(WORKGROUP_X), 1, 1])?;
                self.batch_recorder.push_barrier(
                    self.pipelines.kv_append_f32,
                    &[BufferSpan::whole(key_buf), BufferSpan::whole(k_cache)],
                    &[kv_width_u32, pos_u32],
                    grid,
                );
                self.batch_recorder.push_concurrent(
                    self.pipelines.kv_append_f32,
                    &[BufferSpan::whole(val_buf), BufferSpan::whole(v_cache)],
                    &[kv_width_u32, pos_u32],
                    grid,
                );
            }
            (VulkanKvPrecision::Fp32, false) => {
                let grid = checked_grid(
                    &self.device,
                    [kv_width_u32.div_ceil(WORKGROUP_X), tokens_u32, 1],
                )?;
                self.batch_recorder.push_barrier(
                    self.pipelines.kv_append_batched_f32,
                    &[BufferSpan::whole(key_buf), BufferSpan::whole(k_cache)],
                    &[kv_width_u32, pos_u32, tokens_u32],
                    grid,
                );
                self.batch_recorder.push_concurrent(
                    self.pipelines.kv_append_batched_f32,
                    &[BufferSpan::whole(val_buf), BufferSpan::whole(v_cache)],
                    &[kv_width_u32, pos_u32, tokens_u32],
                    grid,
                );
            }
        }
        Ok(())
    }

    fn record_ffn_tokens(&mut self, layer_idx: usize, tokens: usize) -> Result<(), Error> {
        let ffn_gate = self.layers[layer_idx].ffn_gate;
        let ffn_up = self.layers[layer_idx].ffn_up;
        let ffn_down = self.layers[layer_idx].ffn_down;

        let gate_out_buf = self.scratch.proj_a;
        let up_out_buf = self.scratch.proj_b;
        let mid_buf = self.scratch.proj_c;

        self.record_projection_group_tokens(
            self.scratch.norm,
            &[(ffn_gate, gate_out_buf), (ffn_up, up_out_buf)],
            tokens,
        )?;

        let total_inter_u32 = u32::try_from(tokens * self.geometry.feed_forward)
            .map_err(|_| Error::ResourceLimit("batched feed_forward exceeds u32".to_owned()))?;
        let swiglu_pipe = if self.resolved_pinned_arithmetic {
            self.pipelines.proj.swiglu_pinned
        } else {
            self.pipelines.proj.swiglu_unpinned
        };
        self.batch_recorder.push_barrier(
            swiglu_pipe,
            &[
                BufferSpan::whole(gate_out_buf),
                BufferSpan::whole(up_out_buf),
                BufferSpan::whole(mid_buf),
            ],
            &[total_inter_u32],
            checked_grid(&self.device, [total_inter_u32.div_ceil(WORKGROUP_X), 1, 1])?,
        );

        self.record_projection_group_tokens(mid_buf, &[(ffn_down, self.scratch.delta)], tokens)?;
        Ok(())
    }

    fn record_final_logits(&mut self, tile_tokens: usize) -> Result<(), Error> {
        let embed_u32 = u32::try_from(self.geometry.embedding)
            .map_err(|_| Error::ResourceLimit("embedding exceeds u32".to_owned()))?;
        if self.layers.is_empty() {
            let hidden_src = if tile_tokens > 1 {
                let last_off =
                    u32::try_from((tile_tokens - 1) * self.geometry.embedding).map_err(|_| {
                        Error::ResourceLimit("last hidden offset exceeds u32".to_owned())
                    })?;
                self.record_copy_slice(
                    self.scratch.hidden,
                    last_off,
                    self.scratch.proj_a,
                    embed_u32,
                    true,
                )?;
                self.scratch.proj_a
            } else {
                self.scratch.hidden
            };
            self.batch_recorder.push_barrier(
                self.pipelines.rms_norm,
                &[
                    BufferSpan::whole(hidden_src),
                    BufferSpan::whole(self.output_norm_w),
                    BufferSpan::whole(self.scratch.norm),
                ],
                &[embed_u32, self.geometry.epsilon.to_bits()],
                checked_grid(&self.device, [1, 1, 1])?,
            );
        } else {
            self.batch_recorder.push_barrier(
                self.pipelines.add_rms_norm,
                &[
                    BufferSpan::whole(self.scratch.hidden),
                    BufferSpan::whole(self.scratch.delta),
                    BufferSpan::whole(self.output_norm_w),
                    BufferSpan::whole(self.scratch.norm),
                ],
                &[embed_u32, self.geometry.epsilon.to_bits()],
                checked_grid(&self.device, [1, 1, 1])?,
            );
        }
        self.record_projection_group_tokens(
            self.scratch.norm,
            &[(self.output_weight, self.scratch.logits)],
            1,
        )
    }

    fn record_projection_group_tokens(
        &mut self,
        src_buf: BufferId,
        targets: &[(GpuMatrixWeight, BufferId)],
        tokens: usize,
    ) -> Result<(), Error> {
        if targets.is_empty() {
            return Ok(());
        }
        // Group sibling projections by `(rotated, grouped_ssm)` basis, emitting unrotated
        // siblings first so `src_buf` stays read-only across both basis groups.
        let mut bases: [(bool, bool); 3] = [(false, false); 3];
        let mut base_count = 0_usize;
        for (weight, _) in targets {
            let b = weight.basis();
            if !bases[..base_count].contains(&b) {
                bases[base_count] = b;
                base_count += 1;
            }
        }
        if base_count > 1 {
            bases[..base_count].sort_unstable();
        }

        let mut sync = ProjectionGroupSync::default();
        for &basis in &bases[..base_count] {
            let group: Vec<(GpuMatrixWeight, BufferId)> = targets
                .iter()
                .copied()
                .filter(|(w, _)| w.basis() == basis)
                .collect();
            self.record_basis_projection_subgroup_tokens(
                src_buf, basis, &group, tokens, &mut sync,
            )?;
        }
        self.record_projection_group_adapters_tokens(src_buf, targets, tokens)
    }

    fn record_projection_group_adapters_tokens(
        &mut self,
        src_buf: BufferId,
        targets: &[(GpuMatrixWeight, BufferId)],
        tokens: usize,
    ) -> Result<(), Error> {
        let Some(scale) = self.multi_rank_scale.filter(|&s| s != 0.0) else {
            return Ok(());
        };
        for &(weight, out_buf) in targets {
            if let Some(lora) = weight.multi_rank {
                self.record_linear_lora_tokens(src_buf, out_buf, lora, scale, tokens)?;
            }
        }
        Ok(())
    }

    fn record_linear_lora_tokens(
        &mut self,
        src_buf: BufferId,
        out_buf: BufferId,
        lora: GpuLoraFactor,
        user_scale: f32,
        tokens: usize,
    ) -> Result<(), Error> {
        let effective_scale = user_scale * lora.scaling;
        if !effective_scale.is_finite() {
            return Err(Error::InvalidInput(
                "non-finite effective GPU LoRA scale".to_owned(),
            ));
        }
        let in_u32 = u32::try_from(lora.in_features)
            .map_err(|_| Error::ResourceLimit("LoRA in_features exceeds u32".to_owned()))?;
        let out_u32 = u32::try_from(lora.out_features)
            .map_err(|_| Error::ResourceLimit("LoRA out_features exceeds u32".to_owned()))?;
        let rank_u32 = u32::try_from(lora.rank)
            .map_err(|_| Error::ResourceLimit("LoRA rank exceeds u32".to_owned()))?;
        let tokens_u32 = u32::try_from(tokens)
            .map_err(|_| Error::ResourceLimit("tokens exceed u32".to_owned()))?;

        // `src_buf` holds unrotated original activations across the whole projection group.
        if tokens == 1 {
            self.batch_recorder.push_barrier(
                self.pipelines.f32_matvec,
                &[
                    lora.a_span,
                    BufferSpan::whole(src_buf),
                    BufferSpan::whole(self.scratch.lora_ranks),
                ],
                &[in_u32, rank_u32],
                checked_grid(&self.device, [rank_u32, 1, 1])?,
            );
            self.batch_recorder.push_barrier(
                self.pipelines.lora_b_accum,
                &[
                    lora.b_span,
                    BufferSpan::whole(self.scratch.lora_ranks),
                    BufferSpan::whole(out_buf),
                ],
                &[out_u32, rank_u32, effective_scale.to_bits()],
                checked_grid(&self.device, [out_u32.div_ceil(WORKGROUP_X), 1, 1])?,
            );
        } else {
            let tok_tiles = tokens_u32.div_ceil(4);
            self.batch_recorder.push_barrier(
                self.pipelines.f32_gemm_batched,
                &[
                    lora.a_span,
                    BufferSpan::whole(src_buf),
                    BufferSpan::whole(self.scratch.lora_ranks),
                ],
                &[in_u32, rank_u32, tokens_u32],
                checked_grid(&self.device, [rank_u32, tok_tiles, 1])?,
            );
            self.batch_recorder.push_barrier(
                self.pipelines.lora_b_accum_batched,
                &[
                    lora.b_span,
                    BufferSpan::whole(self.scratch.lora_ranks),
                    BufferSpan::whole(out_buf),
                ],
                &[out_u32, rank_u32, tokens_u32, effective_scale.to_bits()],
                checked_grid(&self.device, [out_u32.div_ceil(WORKGROUP_X), tok_tiles, 1])?,
            );
        }
        Ok(())
    }

    fn can_fuse_hadamard_1024(
        &self,
        basis: (bool, bool),
        cols: usize,
        group: &[(GpuMatrixWeight, BufferId)],
    ) -> bool {
        let (rotated, grouped_ssm) = basis;
        if !rotated
            || self.transforms.block != 1024
            || !cols.is_multiple_of(1024)
            || group.iter().any(|(w, _)| w.encoding == WeightEncoding::F32)
        {
            return false;
        }
        if grouped_ssm {
            let kh = self.geometry.ssm_keys;
            let rep = self.geometry.ssm_values.checked_div(kh).unwrap_or(0);
            let hd = self.geometry.ssm_head;
            return kh > 0 && rep > 0 && hd > 0 && kh * rep * hd == cols;
        }
        true
    }

    fn fused_hadamard1024_push(
        &self,
        cols: usize,
        grouped_ssm: bool,
    ) -> Result<(BufferId, [u32; 7]), Error> {
        let cols_u32 = u32::try_from(cols)
            .map_err(|_| Error::ResourceLimit("transform width exceeds u32".to_owned()))?;
        let (sign_buf, has_signs) = if let Some(&sb) = self.transforms.sign_buffers.get(&cols) {
            (sb, 1_u32)
        } else {
            (self.scratch.dummy_state, 0_u32)
        };
        let (use_permute, kh, rep, hd) = if grouped_ssm {
            let kh = u32::try_from(self.geometry.ssm_keys)
                .map_err(|_| Error::ResourceLimit("ssm_keys exceeds u32".to_owned()))?;
            let rep = u32::try_from(self.geometry.ssm_values / self.geometry.ssm_keys)
                .map_err(|_| Error::ResourceLimit("repetitions exceeds u32".to_owned()))?;
            let hd = u32::try_from(self.geometry.ssm_head)
                .map_err(|_| Error::ResourceLimit("ssm_head exceeds u32".to_owned()))?;
            (1_u32, kh, rep, hd)
        } else {
            (0_u32, 1_u32, 1_u32, 1_u32)
        };
        Ok((
            sign_buf,
            [
                cols_u32,
                hadamard_norm_bits(1024),
                has_signs,
                use_permute,
                kh,
                rep,
                hd,
            ],
        ))
    }

    fn record_basis_projection_subgroup_tokens(
        &mut self,
        src_buf: BufferId,
        basis: (bool, bool),
        group: &[(GpuMatrixWeight, BufferId)],
        tokens: usize,
        sync: &mut ProjectionGroupSync,
    ) -> Result<(), Error> {
        let cols = group[0].0.cols;
        let fused_h1024 = self.can_fuse_hadamard_1024(basis, cols, group);
        let act_buf = if fused_h1024 {
            src_buf
        } else {
            self.apply_forward_basis_transform_tokens(src_buf, cols, basis, tokens, sync)?
        };

        let needs_q8k = group
            .iter()
            .any(|(w, _)| w.encoding == WeightEncoding::Pq2Q8k);
        let needs_q8_0_source_or_pinned = group
            .iter()
            .any(|(w, _)| matches!(w.encoding, WeightEncoding::Pq2Q8_0 | WeightEncoding::Q8_0));
        let needs_ptq1_q8_0 = group
            .iter()
            .any(|(w, _)| w.encoding == WeightEncoding::Ptq1Q8_0);

        if fused_h1024 {
            return self.dispatch_fused_hadamard1024_projection_group(
                src_buf,
                cols,
                basis.1,
                group,
                tokens,
                needs_q8k,
                needs_q8_0_source_or_pinned && needs_ptq1_q8_0 && self.resolved_pinned_arithmetic,
                sync,
            );
        }

        if needs_q8_0_source_or_pinned && needs_ptq1_q8_0 && self.resolved_pinned_arithmetic {
            let q8_0_pq2: Vec<(GpuMatrixWeight, BufferId)> = group
                .iter()
                .copied()
                .filter(|(w, _)| w.encoding != WeightEncoding::Ptq1Q8_0)
                .collect();
            let q8_0_ptq1: Vec<(GpuMatrixWeight, BufferId)> = group
                .iter()
                .copied()
                .filter(|(w, _)| w.encoding == WeightEncoding::Ptq1Q8_0)
                .collect();
            self.dispatch_prepared_projection_group_tokens(
                src_buf, act_buf, cols, &q8_0_pq2, tokens, sync,
            )?;
            return self.dispatch_prepared_projection_group_tokens(
                src_buf, act_buf, cols, &q8_0_ptq1, tokens, sync,
            );
        }

        self.dispatch_prepared_projection_group_tokens(src_buf, act_buf, cols, group, tokens, sync)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Encodes fused 1024-element basis-transform+quantization dispatch without mutating `src_buf` for LoRA."
    )]
    fn dispatch_fused_hadamard1024_projection_group(
        &mut self,
        src_buf: BufferId,
        cols: usize,
        grouped_ssm: bool,
        group: &[(GpuMatrixWeight, BufferId)],
        tokens: usize,
        needs_q8k: bool,
        split_pinned_ptq1: bool,
        sync: &mut ProjectionGroupSync,
    ) -> Result<(), Error> {
        let tokens_u32 = u32::try_from(tokens)
            .map_err(|_| Error::ResourceLimit("tokens exceed u32".to_owned()))?;
        let blocks1024 = u32::try_from(cols / 1024)
            .map_err(|_| Error::ResourceLimit("Hadamard1024 grid exceeds u32".to_owned()))?;
        let grid = checked_grid(&self.device, [blocks1024, tokens_u32, 1])?;
        let (sign_buf, push) = self.fused_hadamard1024_push(cols, grouped_ssm)?;

        if needs_q8k {
            let q8k_pipe = if self.resolved_pinned_arithmetic {
                self.pipelines.fused_hadamard1024_q8k_pinned
            } else {
                self.pipelines.fused_hadamard1024_q8k_unpinned
            };
            let barrier = !sync.src_barrier_emitted || sync.used_q8k;
            self.batch_recorder.push(
                q8k_pipe,
                &[
                    BufferSpan::whole(src_buf),
                    BufferSpan::whole(sign_buf),
                    BufferSpan::whole(self.scratch.act_q8k),
                ],
                &push,
                grid,
                barrier,
            );
            sync.src_barrier_emitted = true;
            sync.used_q8k = true;
        }

        if split_pinned_ptq1 {
            let q8_0_pq2: Vec<(GpuMatrixWeight, BufferId)> = group
                .iter()
                .copied()
                .filter(|(w, _)| w.encoding != WeightEncoding::Ptq1Q8_0)
                .collect();
            let q8_0_ptq1: Vec<(GpuMatrixWeight, BufferId)> = group
                .iter()
                .copied()
                .filter(|(w, _)| w.encoding == WeightEncoding::Ptq1Q8_0)
                .collect();
            let barrier = !sync.src_barrier_emitted || sync.used_q8_0;
            self.batch_recorder.push(
                self.pipelines.fused_hadamard1024_q8_0_pinned,
                &[
                    BufferSpan::whole(src_buf),
                    BufferSpan::whole(sign_buf),
                    BufferSpan::whole(self.scratch.act_q8_0),
                ],
                &push,
                grid,
                barrier,
            );
            sync.src_barrier_emitted = true;
            sync.used_q8_0 = true;
            self.dispatch_projection_weights(src_buf, cols, &q8_0_pq2, tokens)?;
            self.batch_recorder.push_barrier(
                self.pipelines.fused_hadamard1024_q8_0_pinned_native,
                &[
                    BufferSpan::whole(src_buf),
                    BufferSpan::whole(sign_buf),
                    BufferSpan::whole(self.scratch.act_q8_0),
                ],
                &push,
                grid,
            );
            return self.dispatch_projection_weights(src_buf, cols, &q8_0_ptq1, tokens);
        }

        let q8_kind = group.iter().find_map(|(w, _)| match w.encoding {
            WeightEncoding::Pq2Q8_0 => Some(WeightEncoding::Pq2Q8_0),
            WeightEncoding::Ptq1Q8_0 => Some(WeightEncoding::Ptq1Q8_0),
            WeightEncoding::Q8_0 => Some(WeightEncoding::Q8_0),
            WeightEncoding::Pq2Q8k | WeightEncoding::F32 => None,
        });
        if let Some(kind) = q8_kind {
            let pipe = match (self.resolved_pinned_arithmetic, kind) {
                (false, _) => self.pipelines.fused_hadamard1024_q8_0_source,
                (true, WeightEncoding::Ptq1Q8_0) => {
                    self.pipelines.fused_hadamard1024_q8_0_pinned_native
                }
                (true, _) => self.pipelines.fused_hadamard1024_q8_0_pinned,
            };
            let barrier = !sync.src_barrier_emitted || sync.used_q8_0;
            self.batch_recorder.push(
                pipe,
                &[
                    BufferSpan::whole(src_buf),
                    BufferSpan::whole(sign_buf),
                    BufferSpan::whole(self.scratch.act_q8_0),
                ],
                &push,
                grid,
                barrier,
            );
            sync.src_barrier_emitted = true;
            sync.used_q8_0 = true;
        }
        self.dispatch_projection_weights(src_buf, cols, group, tokens)
    }

    fn apply_forward_basis_transform_tokens(
        &mut self,
        src_buf: BufferId,
        cols: usize,
        basis: (bool, bool),
        tokens: usize,
        sync: &mut ProjectionGroupSync,
    ) -> Result<BufferId, Error> {
        let (rotated, grouped_ssm) = basis;
        if !rotated {
            return Ok(src_buf);
        }
        let cols_u32 = u32::try_from(cols)
            .map_err(|_| Error::ResourceLimit("transform width exceeds u32".to_owned()))?;
        let tokens_u32 = u32::try_from(tokens)
            .map_err(|_| Error::ResourceLimit("tokens exceed u32".to_owned()))?;
        let mut cur = src_buf;

        if grouped_ssm {
            let rep = u32::try_from(self.geometry.ssm_values / self.geometry.ssm_keys)
                .map_err(|_| Error::ResourceLimit("repetitions exceeds u32".to_owned()))?;
            let kh = u32::try_from(self.geometry.ssm_keys)
                .map_err(|_| Error::ResourceLimit("ssm_keys exceeds u32".to_owned()))?;
            let hd = u32::try_from(self.geometry.ssm_head)
                .map_err(|_| Error::ResourceLimit("ssm_head exceeds u32".to_owned()))?;
            let barrier = !sync.src_barrier_emitted || sync.used_perm_tmp;
            if tokens == 1 {
                self.batch_recorder.push(
                    self.pipelines.permute_ssm_heads,
                    &[
                        BufferSpan::whole(cur),
                        BufferSpan::whole(self.scratch.perm_tmp),
                    ],
                    &[kh, rep, hd],
                    checked_grid(&self.device, [cols_u32.div_ceil(WORKGROUP_X), 1, 1])?,
                    barrier,
                );
            } else {
                self.batch_recorder.push(
                    self.pipelines.permute_ssm_heads_batched,
                    &[
                        BufferSpan::whole(cur),
                        BufferSpan::whole(self.scratch.perm_tmp),
                    ],
                    &[kh, rep, hd, tokens_u32],
                    checked_grid(
                        &self.device,
                        [cols_u32.div_ceil(WORKGROUP_X), tokens_u32, 1],
                    )?,
                    barrier,
                );
            }
            sync.src_barrier_emitted = true;
            sync.used_perm_tmp = true;
            cur = self.scratch.perm_tmp;
        }

        if let Some(&sign_buf) = self.transforms.sign_buffers.get(&cols) {
            let barrier = cur != src_buf || !sync.src_barrier_emitted || sync.used_sign_tmp;
            self.record_mul_sign_tokens_with_barrier(
                cur,
                sign_buf,
                self.scratch.sign_tmp,
                cols_u32,
                tokens_u32,
                barrier,
            )?;
            sync.src_barrier_emitted = true;
            sync.used_sign_tmp = true;
            cur = self.scratch.sign_tmp;
        }

        if self.transforms.block >= 2 && cols.is_multiple_of(self.transforms.block) {
            let block_u32 = u32::try_from(self.transforms.block)
                .map_err(|_| Error::InvalidInput("Hadamard block exceeds u32".to_owned()))?;
            let blk_cnt = u32::try_from(cols / self.transforms.block)
                .map_err(|_| Error::ResourceLimit("Hadamard grid exceeds u32".to_owned()))?;
            let barrier = cur != src_buf || !sync.src_barrier_emitted || sync.used_transformed;
            if tokens == 1 {
                self.batch_recorder.push(
                    self.pipelines.proj.hadamard,
                    &[
                        BufferSpan::whole(cur),
                        BufferSpan::whole(self.scratch.transformed),
                    ],
                    &[block_u32, hadamard_norm_bits(block_u32)],
                    checked_grid(&self.device, [blk_cnt, 1, 1])?,
                    barrier,
                );
            } else {
                self.batch_recorder.push(
                    self.pipelines.hadamard_batched,
                    &[
                        BufferSpan::whole(cur),
                        BufferSpan::whole(self.scratch.transformed),
                    ],
                    &[
                        block_u32,
                        hadamard_norm_bits(block_u32),
                        blk_cnt,
                        tokens_u32,
                    ],
                    checked_grid(&self.device, [blk_cnt, tokens_u32, 1])?,
                    barrier,
                );
            }
            sync.src_barrier_emitted = true;
            sync.used_transformed = true;
            cur = self.scratch.transformed;
        }

        Ok(cur)
    }

    fn dispatch_prepared_projection_group_tokens(
        &mut self,
        src_buf: BufferId,
        act_buf: BufferId,
        cols: usize,
        group: &[(GpuMatrixWeight, BufferId)],
        tokens: usize,
        sync: &mut ProjectionGroupSync,
    ) -> Result<(), Error> {
        if group.is_empty() {
            return Ok(());
        }
        let cols_u32 = u32::try_from(cols)
            .map_err(|_| Error::ResourceLimit("projection cols exceed u32".to_owned()))?;
        let tokens_u32 = u32::try_from(tokens)
            .map_err(|_| Error::ResourceLimit("tokens exceed u32".to_owned()))?;
        let multi_token = tokens > 1;

        let needs_q8k = group
            .iter()
            .any(|(w, _)| w.encoding == WeightEncoding::Pq2Q8k);
        let q8_0_kind = group.iter().find_map(|(w, _)| match w.encoding {
            WeightEncoding::Pq2Q8_0 => Some(WeightEncoding::Pq2Q8_0),
            WeightEncoding::Ptq1Q8_0 => Some(WeightEncoding::Ptq1Q8_0),
            WeightEncoding::Q8_0 => Some(WeightEncoding::Q8_0),
            WeightEncoding::Pq2Q8k | WeightEncoding::F32 => None,
        });

        let quant_emitted = if needs_q8k {
            let q8k_pipe = if self.resolved_pinned_arithmetic {
                self.pipelines.proj.quant_q8k_pinned
            } else {
                self.pipelines.proj.quant_q8k_unpinned
            };
            let barrier = act_buf != src_buf || !sync.src_barrier_emitted || sync.used_q8k;
            self.batch_recorder.push(
                q8k_pipe,
                &[
                    BufferSpan::whole(act_buf),
                    BufferSpan::whole(self.scratch.act_q8k),
                ],
                &[cols_u32, tokens_u32],
                checked_grid(&self.device, [cols_u32 / 256, tokens_u32, 1])?,
                barrier,
            );
            sync.src_barrier_emitted = true;
            sync.used_q8k = true;
            true
        } else {
            false
        };

        if let Some(q8_kind) = q8_0_kind {
            let q8_blocks = cols_u32 / 32;
            let q8_bufs = [
                BufferSpan::whole(act_buf),
                BufferSpan::whole(self.scratch.act_q8_0),
            ];
            let barrier = if act_buf == src_buf {
                !sync.src_barrier_emitted || sync.used_q8_0
            } else {
                !quant_emitted
            };
            if multi_token {
                let q8_pipe = match (self.resolved_pinned_arithmetic, q8_kind) {
                    (false, _) => self.pipelines.quant_q8_0_source_batched,
                    (true, WeightEncoding::Ptq1Q8_0) => {
                        self.pipelines.quant_q8_0_pinned_native_batched
                    }
                    (true, _) => self.pipelines.quant_q8_0_pinned_batched,
                };
                self.batch_recorder.push(
                    q8_pipe,
                    &q8_bufs,
                    &[q8_blocks, tokens_u32],
                    checked_grid(
                        &self.device,
                        [q8_blocks.div_ceil(WORKGROUP_X), tokens_u32, 1],
                    )?,
                    barrier,
                );
            } else {
                let q8_pipe = match (self.resolved_pinned_arithmetic, q8_kind) {
                    (false, _) => self.pipelines.quant_q8_0_source,
                    (true, WeightEncoding::Ptq1Q8_0) => self.pipelines.quant_q8_0_pinned_native,
                    (true, _) => self.pipelines.quant_q8_0_pinned,
                };
                self.batch_recorder.push(
                    q8_pipe,
                    &q8_bufs,
                    &[q8_blocks],
                    checked_grid(&self.device, [q8_blocks.div_ceil(WORKGROUP_X), 1, 1])?,
                    barrier,
                );
            }
            sync.src_barrier_emitted = true;
            sync.used_q8_0 = true;
        } else if !quant_emitted && act_buf == src_buf {
            let first_barrier = !sync.src_barrier_emitted;
            sync.src_barrier_emitted = true;
            return self.dispatch_projection_weights_with_first_barrier(
                act_buf,
                cols,
                group,
                tokens,
                first_barrier,
            );
        }

        self.dispatch_projection_weights(act_buf, cols, group, tokens)
    }

    fn dispatch_projection_weights(
        &mut self,
        f32_act_buf: BufferId,
        cols: usize,
        group: &[(GpuMatrixWeight, BufferId)],
        tokens: usize,
    ) -> Result<(), Error> {
        self.dispatch_projection_weights_with_first_barrier(f32_act_buf, cols, group, tokens, true)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Dispatches all weight-encoded matrix projections (`Pq2Q8k`, `Pq2Q8_0`, `Ptq1Q8_0`, `Q8_0`, `F32`) from already-quantized or float activation scratch."
    )]
    fn dispatch_projection_weights_with_first_barrier(
        &mut self,
        f32_act_buf: BufferId,
        cols: usize,
        group: &[(GpuMatrixWeight, BufferId)],
        tokens: usize,
        first_barrier: bool,
    ) -> Result<(), Error> {
        if group.is_empty() {
            return Ok(());
        }
        let cols_u32 = u32::try_from(cols)
            .map_err(|_| Error::ResourceLimit("projection cols exceed u32".to_owned()))?;
        let tokens_u32 = u32::try_from(tokens)
            .map_err(|_| Error::ResourceLimit("tokens exceed u32".to_owned()))?;
        let multi_token = tokens > 1;
        let batch_path = BatchTokenPath::for_tokens(&self.device, tokens);
        let tok_tiles_16y = batch_path.grid_y(tokens_u32);
        let tile_y_4 = batch_path.grid_y_four_tok_cap(tokens_u32);

        for (idx, (weight, out_buf)) in group.iter().enumerate() {
            let rows_u32 = u32::try_from(weight.rows)
                .map_err(|_| Error::ResourceLimit("projection rows exceed u32".to_owned()))?;
            let barrier_before = idx == 0 && first_barrier;
            match weight.encoding {
                WeightEncoding::Pq2Q8k => {
                    let blocks_per_row = cols / 128;
                    let blocks_u32 = u32::try_from(blocks_per_row).map_err(|_| {
                        Error::ResourceLimit("blocks_per_row exceed u32".to_owned())
                    })?;
                    let act_words = u32::try_from(blocks_per_row * Q8K_4PLANE_WORDS_PER_HALF)
                        .map_err(|_| Error::ResourceLimit("act_words exceed u32".to_owned()))?;
                    let pipe = self.pipelines.proj.select_pq2_q8k(
                        self.resolved_pinned_arithmetic,
                        batch_path,
                        false,
                    );
                    self.batch_recorder.push(
                        pipe,
                        &[
                            weight.span,
                            BufferSpan::whole(self.scratch.act_q8k),
                            BufferSpan::whole(*out_buf),
                        ],
                        &[rows_u32, blocks_u32, tokens_u32, act_words],
                        checked_grid(
                            &self.device,
                            [rows_u32.div_ceil(WORKGROUP_X), tok_tiles_16y, 1],
                        )?,
                        barrier_before,
                    );
                }
                WeightEncoding::Pq2Q8_0 | WeightEncoding::Ptq1Q8_0 | WeightEncoding::Q8_0 => {
                    let blocks_per_row = if weight.encoding == WeightEncoding::Q8_0 {
                        cols / 32
                    } else {
                        cols / 128
                    };
                    let blocks_u32 = u32::try_from(blocks_per_row).map_err(|_| {
                        Error::ResourceLimit("blocks_per_row exceed u32".to_owned())
                    })?;
                    let act_words = u32::try_from((cols / 32) * Q8_0_WORDS_PER_BLOCK)
                        .map_err(|_| Error::ResourceLimit("act_words exceed u32".to_owned()))?;
                    let (pipe, tok_tiles_y) = match weight.encoding {
                        WeightEncoding::Pq2Q8_0 => {
                            (self.pipelines.proj.select_pq2_q8_0(batch_path), tile_y_4)
                        }
                        WeightEncoding::Ptq1Q8_0 => (
                            self.pipelines
                                .proj
                                .select_ptq1_q8_0(self.resolved_pinned_arithmetic, batch_path),
                            tok_tiles_16y,
                        ),
                        WeightEncoding::Q8_0 => (
                            self.pipelines.proj.select_q8_0_q8_0(batch_path),
                            tok_tiles_16y,
                        ),
                        WeightEncoding::Pq2Q8k | WeightEncoding::F32 => {
                            return Err(Error::InvalidInput("unexpected encoding".to_owned()));
                        }
                    };
                    let tile_rows = rows_u32.div_ceil(WORKGROUP_X);
                    self.batch_recorder.push(
                        pipe,
                        &[
                            weight.span,
                            BufferSpan::whole(self.scratch.act_q8_0),
                            BufferSpan::whole(*out_buf),
                        ],
                        &[rows_u32, blocks_u32, tokens_u32, act_words],
                        checked_grid(&self.device, [tile_rows, tok_tiles_y, 1])?,
                        barrier_before,
                    );
                }
                WeightEncoding::F32 => {
                    if multi_token {
                        self.batch_recorder.push(
                            self.pipelines.f32_gemm_batched,
                            &[
                                weight.span,
                                BufferSpan::whole(f32_act_buf),
                                BufferSpan::whole(*out_buf),
                            ],
                            &[cols_u32, rows_u32, tokens_u32],
                            checked_grid(&self.device, [rows_u32, tile_y_4, 1])?,
                            barrier_before,
                        );
                    } else {
                        self.batch_recorder.push(
                            self.pipelines.f32_matvec,
                            &[
                                weight.span,
                                BufferSpan::whole(f32_act_buf),
                                BufferSpan::whole(*out_buf),
                            ],
                            &[cols_u32, rows_u32],
                            checked_grid(&self.device, [rows_u32, 1, 1])?,
                            barrier_before,
                        );
                    }
                }
            }
        }
        Ok(())
    }
}

impl Decoder for VulkanBatchedGraphExecutor {
    type Checkpoint = VulkanBatchedCheckpoint;

    fn identity(&self) -> &str {
        &self.identity_sha256
    }

    fn vocab_size(&self) -> usize {
        self.geometry.vocabulary
    }

    fn checkpoint_bytes(&self) -> usize {
        let words_per_tok = match self.config.kv_precision {
            VulkanKvPrecision::Fp16 => self.geometry.kv_width() / 2,
            VulkanKvPrecision::Fp32 => self.geometry.kv_width(),
        };
        let kv_bytes_per_layer = self
            .position
            .saturating_mul(words_per_tok)
            .saturating_mul(2 * size_of::<u32>());
        let mut payload_bytes = 0_usize;
        for layer in &self.layers {
            payload_bytes = payload_bytes.saturating_add(match &layer.attention {
                GpuLayerAttention::Recurrent {
                    conv_state_elems,
                    matrix_state_elems,
                    ..
                } => conv_state_elems
                    .saturating_add(*matrix_state_elems)
                    .saturating_mul(size_of::<f32>()),
                GpuLayerAttention::Full { .. } => kv_bytes_per_layer,
            });
        }
        size_of::<VulkanBatchedCheckpoint>()
            .saturating_add(self.identity_sha256.capacity())
            .saturating_add(
                self.layers
                    .len()
                    .saturating_mul(size_of::<LayerStateCheckpoint>()),
            )
            .saturating_add(payload_bytes)
    }

    fn admit_work(&self, ingest_tokens: usize, evaluate_tokens: usize) -> RuntimeResult<()> {
        if self.device.is_faulted() {
            return Err(Error::DeviceFault(
                "Vulkan batched executor faulted and requires terminal teardown".to_owned(),
            )
            .into_runtime_error());
        }
        if !self.state_consistent {
            return Err(Error::InvalidInput(
                "executor state is inconsistent after failed evaluation and requires reset or restore"
                    .to_owned(),
            )
            .into_runtime_error());
        }
        ingest_tokens.checked_add(evaluate_tokens).ok_or_else(|| {
            Error::ResourceLimit("requested token work count overflow".to_owned())
                .into_runtime_error()
        })?;
        Ok(())
    }

    fn reset(&mut self) -> RuntimeResult<()> {
        Self::reset(self).map_err(Error::into_runtime_error)
    }

    fn evaluate(&mut self, token: u32) -> RuntimeResult<Vec<f32>> {
        self.evaluate_token(token)
            .map_err(Error::into_runtime_error)
    }

    fn evaluate_into(&mut self, token: u32, output: &mut Vec<f32>) -> RuntimeResult<()> {
        self.evaluate_token_into(token, output)
            .map_err(Error::into_runtime_error)
    }

    fn evaluate_finite_into<'a>(
        &mut self,
        token: u32,
        output: &'a mut Vec<f32>,
    ) -> RuntimeResult<FiniteLogits<'a>> {
        output.clear();
        self.validate_execution_ready(&[token])
            .map_err(Error::into_runtime_error)?;
        if let Err(err) = self.run_tile_gpu_into(&[token], Some(output), &mut || Ok(())) {
            output.clear();
            return Err(err.into_runtime_error());
        }
        if output.len() != self.geometry.vocabulary || output.is_empty() || output.len() > 1_048_576
        {
            output.clear();
            self.state_consistent = false;
            self.kv_tracker.invalidate();
            return Err(Error::InvalidInput(
                "GPU returned invalid logit vocabulary extent".to_owned(),
            )
            .into_runtime_error());
        }
        let prev_consistent = self.state_consistent;
        self.state_consistent = false;
        let finite = match FiniteLogits::new(output.as_slice()) {
            Ok(finite) => finite,
            Err(err) => {
                self.kv_tracker.invalidate();
                return Err(err);
            }
        };
        self.state_consistent = prev_consistent;
        self.commit_tile_position(1);
        Ok(finite)
    }

    fn ingest(&mut self, token: u32) -> RuntimeResult<()> {
        self.ingest_token(token).map_err(Error::into_runtime_error)
    }

    fn ingest_block(
        &mut self,
        tokens: &[u32],
        check: &dyn Fn() -> RuntimeResult<()>,
    ) -> RuntimeResult<()> {
        if tokens.is_empty()
            || tokens.len() > MAX_PREFILL_BLOCK_TOKENS
            || tokens
                .iter()
                .any(|&token| !usize::try_from(token).is_ok_and(|value| value < self.vocab_size()))
        {
            return Err(Error::InvalidInput(
                "invalid native prefill block bounds or tokens".to_owned(),
            )
            .into_runtime_error());
        }
        let mut pending_check_err: Option<bonsai_prism_runtime::Error> = None;
        let mut wrapped_check = || -> Result<(), Error> {
            if let Err(err) = check() {
                pending_check_err = Some(err);
                return Err(Error::Cancelled);
            }
            Ok(())
        };
        let res = self.ingest_tokens_batched_with_check(tokens, &mut wrapped_check);
        res.map_err(|err| {
            pending_check_err
                .take()
                .unwrap_or_else(|| err.into_runtime_error())
        })
    }

    fn checkpoint(&self) -> RuntimeResult<Self::Checkpoint> {
        Self::checkpoint(self).map_err(Error::into_runtime_error)
    }

    fn checkpoint_into(&self, target: &mut Self::Checkpoint) -> RuntimeResult<()> {
        Self::checkpoint_into(self, target).map_err(Error::into_runtime_error)
    }

    fn restore(&mut self, checkpoint: &Self::Checkpoint) -> RuntimeResult<()> {
        Self::restore(self, checkpoint).map_err(Error::into_runtime_error)
    }
}

fn meta_str<'a>(model: &'a Model, key: &str) -> Result<&'a str, Error> {
    model
        .metadata
        .get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidInput(format!("required string metadata missing: {key}")))
}

fn meta_usize(model: &Model, key: &str) -> Result<usize, Error> {
    let val = model
        .metadata
        .get(key)
        .ok_or_else(|| Error::InvalidInput(format!("required integer metadata missing: {key}")))?;
    let raw = val.as_u64().ok_or_else(|| {
        Error::InvalidInput(format!("metadata '{key}' is not an unsigned integer"))
    })?;
    usize::try_from(raw)
        .map_err(|_| Error::ResourceLimit(format!("metadata '{key}' overflows usize")))
}

fn meta_optional_usize(model: &Model, key: &str, default: usize) -> Result<usize, Error> {
    if model.metadata.contains_key(key) {
        meta_usize(model, key)
    } else {
        Ok(default)
    }
}

fn meta_f32(model: &Model, key: &str) -> Result<f32, Error> {
    let Some(val) = model.metadata.get(key) else {
        return Err(Error::InvalidInput(format!(
            "required float metadata missing: {key}"
        )));
    };
    let Some(raw) = val.as_f64() else {
        return Err(Error::InvalidInput(format!(
            "required float metadata missing: {key}"
        )));
    };
    let f = f64_to_f32_ieee(raw);
    if !f.is_finite() {
        return Err(Error::InvalidInput(format!(
            "metadata float '{key}' is non-finite"
        )));
    }
    Ok(f)
}

fn meta_optional_bool(model: &Model, key: &str, default: bool) -> Result<bool, Error> {
    model.metadata.get(key).map_or(Ok(default), |v| {
        v.as_bool()
            .ok_or_else(|| Error::InvalidInput(format!("metadata '{key}' is not boolean")))
    })
}

#[expect(
    clippy::too_many_lines,
    reason = "Parses and validates all Qwen35 architecture dimensions, RoPE sections, and layer recurrence flags together."
)]
fn parse_model_geometry(
    model: &Model,
    config: &VulkanBatchedConfig,
) -> Result<(GraphGeometry, usize, Vec<bool>), Error> {
    if meta_str(model, "general.architecture")? != "qwen35" {
        return Err(Error::InvalidInput(
            "VulkanBatchedGraphExecutor supports general.architecture=qwen35 only".to_owned(),
        ));
    }
    let block_count = meta_usize(model, "qwen35.block_count")?;
    let mtp = meta_optional_usize(model, "qwen35.nextn_predict_layers", 0)?;
    if block_count == 0 || block_count > MAX_GRAPH_LAYERS || mtp >= block_count {
        return Err(Error::InvalidInput(
            "invalid Qwen35 block or MTP count".to_owned(),
        ));
    }
    let full_layer_count = block_count - mtp;
    let layer_count = match config.max_layer_count {
        Some(n) if n > 0 && n <= full_layer_count => n,
        Some(n) => {
            return Err(Error::InvalidInput(format!(
                "max_layer_count {n} outside 1..={full_layer_count}"
            )));
        }
        None => full_layer_count,
    };

    let train_context = meta_usize(model, "qwen35.context_length")?;
    let context_limit = config
        .context_limit
        .unwrap_or_else(|| train_context.min(8192));
    if context_limit == 0 || context_limit > train_context || context_limit > 1_048_576 {
        return Err(Error::InvalidInput(format!(
            "context_limit {context_limit} outside 1..={train_context}"
        )));
    }

    let embedding = meta_usize(model, "qwen35.embedding_length")?;
    let feed_forward = meta_usize(model, "qwen35.feed_forward_length")?;
    let heads = meta_usize(model, "qwen35.attention.head_count")?;
    let kv_heads = meta_usize(model, "qwen35.attention.head_count_kv")?;
    let head = meta_usize(model, "qwen35.attention.key_length")?;
    let value_length = meta_usize(model, "qwen35.attention.value_length")?;
    let rope = meta_usize(model, "qwen35.rope.dimension_count")?;
    let rope_base = meta_f32(model, "qwen35.rope.freq_base")?;
    let epsilon = meta_f32(model, "qwen35.attention.layer_norm_rms_epsilon")?;
    let ssm_head = meta_usize(model, "qwen35.ssm.state_size")?;
    let ssm_keys = meta_usize(model, "qwen35.ssm.group_count")?;
    let ssm_values = meta_usize(model, "qwen35.ssm.time_step_rank")?;
    let ssm_inner = meta_usize(model, "qwen35.ssm.inner_size")?;
    let convolution = meta_usize(model, "qwen35.ssm.conv_kernel")?;

    let token_embd = model
        .tensors
        .get("token_embd.weight")
        .ok_or_else(|| Error::InvalidInput("missing token_embd.weight".to_owned()))?;
    let vocabulary = token_embd.dimensions.get(1).copied().unwrap_or(0);

    let sizes = [
        embedding,
        feed_forward,
        heads,
        kv_heads,
        head,
        rope,
        ssm_head,
        ssm_keys,
        ssm_values,
        convolution,
        vocabulary,
    ];
    if sizes.contains(&0)
        || sizes[..10].iter().any(|&v| v > 65_535)
        || vocabulary > 2_097_152
        || !heads.is_multiple_of(kv_heads)
        || !ssm_values.is_multiple_of(ssm_keys)
        || rope > head
        || !rope.is_multiple_of(2)
        || rope_base <= 0.0
        || epsilon <= 0.0
        || value_length != head
        || ssm_inner != ssm_head * ssm_values
        || heads * head > MAX_GRAPH_WIDTH
        || (2 * ssm_head * ssm_keys + ssm_inner) > MAX_GRAPH_WIDTH
    {
        return Err(Error::InvalidInput(
            "unsupported or invalid Qwen35 geometry".to_owned(),
        ));
    }

    if config.kv_precision == VulkanKvPrecision::Fp16 && !(kv_heads * head).is_multiple_of(2) {
        return Err(Error::InvalidInput(
            "VulkanKvPrecision::Fp16 requires even kv_heads * head_dim".to_owned(),
        ));
    }

    let sec_arr = model
        .metadata
        .get("qwen35.rope.dimension_sections")
        .and_then(|v| v.as_array())
        .ok_or_else(|| Error::InvalidInput("missing qwen35.rope.dimension_sections".to_owned()))?;
    if sec_arr.len() != 4 {
        return Err(Error::InvalidInput(
            "qwen35.rope.dimension_sections must have length 4".to_owned(),
        ));
    }
    let mut rope_sections = [0_usize; 4];
    let mut sec_sum = 0_usize;
    for (idx, item) in sec_arr.iter().enumerate() {
        let v = item
            .as_u64()
            .and_then(|u| usize::try_from(u).ok())
            .ok_or_else(|| Error::InvalidInput("invalid RoPE section integer".to_owned()))?;
        rope_sections[idx] = v;
        sec_sum = sec_sum
            .checked_add(v)
            .ok_or_else(|| Error::InvalidInput("RoPE section sum overflow".to_owned()))?;
    }
    if sec_sum != rope / 2 {
        return Err(Error::InvalidInput(
            "RoPE dimension sections must sum to rope_dim / 2".to_owned(),
        ));
    }

    let full_interval = meta_optional_usize(model, "qwen35.full_attention_interval", 4)?;
    if full_interval == 0 {
        return Err(Error::InvalidInput(
            "qwen35.full_attention_interval must be positive".to_owned(),
        ));
    }
    let recurrent_flags =
        if let Some(flags_val) = model.metadata.get("qwen35.attention.recurrent_layers") {
            let arr = flags_val.as_array().ok_or_else(|| {
                Error::InvalidInput("qwen35.attention.recurrent_layers must be an array".to_owned())
            })?;
            if arr.len() != block_count {
                return Err(Error::InvalidInput(
                    "qwen35.attention.recurrent_layers length mismatch".to_owned(),
                ));
            }
            let mut flags = Vec::with_capacity(layer_count);
            for item in arr.iter().take(layer_count) {
                flags.push(item.as_bool().ok_or_else(|| {
                    Error::InvalidInput(
                        "qwen35.attention.recurrent_layers must contain bools".to_owned(),
                    )
                })?);
            }
            flags
        } else {
            (0..layer_count)
                .map(|idx| !(idx + 1).is_multiple_of(full_interval))
                .collect()
        };

    let geometry = GraphGeometry {
        embedding,
        feed_forward,
        vocabulary,
        heads,
        kv_heads,
        head,
        rope,
        rope_sections,
        rope_base,
        epsilon,
        ssm_head,
        ssm_keys,
        ssm_values,
        convolution,
        context_limit,
    };
    Ok((geometry, layer_count, recurrent_flags))
}

#[expect(
    clippy::too_many_lines,
    reason = "Reads and validates Hadamard rotation parameters, weight list, and explicit signs in one coherent pass."
)]
fn parse_and_upload_transforms(
    model: &Model,
    geometry: &GraphGeometry,
    guard: &mut BufferCleanupGuard<'_>,
) -> Result<GraphTransforms, Error> {
    if !model.metadata.contains_key("prism.hadamard.version") {
        if model
            .metadata
            .keys()
            .any(|key| key.starts_with("prism.hadamard."))
        {
            return Err(Error::InvalidInput(
                "Hadamard metadata requires prism.hadamard.version".to_owned(),
            ));
        }
        return Ok(GraphTransforms::default());
    }

    let version = meta_usize(model, "prism.hadamard.version")?;
    let tied = meta_optional_bool(model, "prism.hadamard.tied_output", false)?;
    if !matches!(version, 1 | 2) || (version == 2) != tied {
        return Err(Error::InvalidInput(
            "Hadamard version 2 requires tied_output; version 1 forbids it".to_owned(),
        ));
    }
    let block = meta_usize(model, "prism.hadamard.block_size")?;
    if !block.is_power_of_two() || !(2..=1024).contains(&block) {
        return Err(Error::InvalidInput(format!(
            "GPU Hadamard block size {block} must be a power of 2 in 2..=1024"
        )));
    }
    if meta_str(model, "prism.hadamard.transform")? != "normalized-sylvester-walsh-hadamard"
        || meta_str(model, "prism.hadamard.axis")? != "input-last-dimension"
    {
        return Err(Error::InvalidInput(
            "unsupported Hadamard transform or axis".to_owned(),
        ));
    }

    let mut weights = BTreeSet::new();
    let w_arr = model
        .metadata
        .get("prism.hadamard.weight_names")
        .and_then(|v| v.as_array())
        .ok_or_else(|| Error::InvalidInput("missing prism.hadamard.weight_names".to_owned()))?;
    for entry in w_arr.iter() {
        let name = entry.as_str().ok_or_else(|| {
            Error::InvalidInput("Hadamard weight names must be strings".to_owned())
        })?;
        weights.insert(name.to_owned());
    }
    if tied {
        weights.insert("token_embd.weight".to_owned());
    }

    let mut inverse_embedding = false;
    if let Some(inv_arr) = model
        .metadata
        .get("prism.hadamard.inverse_weight_names")
        .and_then(|v| v.as_array())
    {
        for entry in inv_arr.iter() {
            if entry.as_str() == Some("token_embd.weight") {
                inverse_embedding = true;
            }
        }
    }

    let mut sign_buffers = BTreeMap::new();
    match meta_str(model, "prism.hadamard.sign_mode")? {
        "identity" => {}
        "explicit" => {
            let widths = model
                .metadata
                .get("prism.hadamard.sign_widths")
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    Error::InvalidInput("missing prism.hadamard.sign_widths".to_owned())
                })?;
            let values = model
                .metadata
                .get("prism.hadamard.sign_values")
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    Error::InvalidInput("missing prism.hadamard.sign_values".to_owned())
                })?;
            let mut offset = 0_usize;
            for width_val in widths.iter() {
                let width = width_val
                    .as_u64()
                    .and_then(|u| usize::try_from(u).ok())
                    .ok_or_else(|| Error::InvalidInput("invalid sign width".to_owned()))?;
                let end = offset
                    .checked_add(width)
                    .ok_or_else(|| Error::InvalidInput("sign length overflow".to_owned()))?;
                if width == 0 || !width.is_multiple_of(block) || end > values.len() {
                    return Err(Error::InvalidInput(
                        "invalid Hadamard sign width".to_owned(),
                    ));
                }
                let mut vec = Vec::with_capacity(width);
                for idx in offset..end {
                    let Some(val) = values.get(idx) else {
                        return Err(Error::InvalidInput(
                            "Hadamard signs must be +1 or -1".to_owned(),
                        ));
                    };
                    let s = match val.as_i64() {
                        Some(-1) => -1.0_f32,
                        Some(1) => 1.0_f32,
                        _ => {
                            return Err(Error::InvalidInput(
                                "Hadamard signs must be +1 or -1".to_owned(),
                            ));
                        }
                    };
                    vec.push(s);
                }
                let buf = guard.alloc_f32_buffer(&vec)?;
                if sign_buffers.insert(width, buf).is_some() {
                    return Err(Error::InvalidInput(
                        "duplicate Hadamard sign width".to_owned(),
                    ));
                }
                offset = end;
            }
        }
        _ => {
            return Err(Error::InvalidInput(
                "unsupported Hadamard sign mode".to_owned(),
            ));
        }
    }

    let grouped_values = meta_optional_bool(model, "prism.hadamard.gdn_v_grouped", false)?;
    if grouped_values && !geometry.ssm_values.is_multiple_of(geometry.ssm_keys) {
        return Err(Error::InvalidInput(
            "grouped GDN transform has incompatible head geometry".to_owned(),
        ));
    }

    Ok(GraphTransforms {
        block,
        weights,
        inverse_embedding,
        grouped_values,
        sign_buffers,
    })
}

fn validate_model_tensors_subset(
    model: &Model,
    geometry: &GraphGeometry,
    layer_count: usize,
    recurrent_flags: &[bool],
) -> Result<(), Error> {
    let required_common = ["token_embd.weight", "output_norm.weight"];
    for name in required_common {
        let t = model
            .tensors
            .get(name)
            .ok_or_else(|| Error::InvalidInput(format!("missing tensor '{name}'")))?;
        validate_tensor(t).map_err(|e| Error::InvalidInput(e.to_string()))?;
    }
    if let Some(out) = model.tensors.get("output.weight") {
        validate_tensor(out).map_err(|e| Error::InvalidInput(e.to_string()))?;
    }

    for (layer_idx, &is_recurrent) in recurrent_flags.iter().enumerate() {
        let prefix = format!("blk.{layer_idx}.");
        let mut names = vec![
            format!("{prefix}attn_norm.weight"),
            format!("{prefix}post_attention_norm.weight"),
            format!("{prefix}ffn_gate.weight"),
            format!("{prefix}ffn_up.weight"),
            format!("{prefix}ffn_down.weight"),
        ];
        if is_recurrent {
            names.extend([
                format!("{prefix}attn_qkv.weight"),
                format!("{prefix}attn_gate.weight"),
                format!("{prefix}ssm_beta.weight"),
                format!("{prefix}ssm_alpha.weight"),
                format!("{prefix}ssm_conv1d.weight"),
                format!("{prefix}ssm_dt.bias"),
                format!("{prefix}ssm_a"),
                format!("{prefix}ssm_norm.weight"),
                format!("{prefix}ssm_out.weight"),
            ]);
        } else {
            names.extend([
                format!("{prefix}attn_q.weight"),
                format!("{prefix}attn_k.weight"),
                format!("{prefix}attn_v.weight"),
                format!("{prefix}attn_output.weight"),
                format!("{prefix}attn_q_norm.weight"),
                format!("{prefix}attn_k_norm.weight"),
            ]);
        }
        for name in names {
            let t = model
                .tensors
                .get(&name)
                .ok_or_else(|| Error::InvalidInput(format!("missing tensor '{name}'")))?;
            validate_tensor(t).map_err(|e| Error::InvalidInput(e.to_string()))?;
        }
    }
    let _ = (geometry, layer_count);
    Ok(())
}

fn lookup_tensor(model: &Model, name: &str, expected_dims: &[usize]) -> Result<Tensor, Error> {
    let t = model
        .tensors
        .get(name)
        .ok_or_else(|| Error::InvalidInput(format!("missing tensor '{name}'")))?;
    if t.dimensions.len() != expected_dims.len()
        || t.dimensions
            .iter()
            .zip(expected_dims)
            .any(|(&actual, &expected)| actual == 0 || (expected != 0 && actual != expected))
    {
        return Err(Error::InvalidInput(format!(
            "tensor '{name}' shape mismatch: got {:?}, expected {expected_dims:?}",
            t.dimensions
        )));
    }
    Ok(t.clone())
}

fn load_and_upload_vector(
    model: &Model,
    name: &str,
    width: usize,
    row_scratch: &mut Vec<f32>,
    guard: &mut BufferCleanupGuard<'_>,
) -> Result<BufferId, Error> {
    let tensor = lookup_tensor(model, name, &[width])?;
    row_into(&tensor, 0, row_scratch).map_err(|err| Error::InvalidInput(err.to_string()))?;
    ensure_finite_f32(
        row_scratch,
        &format!("vector '{name}' contains non-finite value"),
    )?;
    guard.alloc_f32_buffer(row_scratch)
}

fn parse_blk_target(target: &str) -> Option<(usize, &str)> {
    let rest = target.strip_prefix("blk.")?;
    let (layer_str, suffix) = rest.split_once('.')?;
    if layer_str.is_empty() || !layer_str.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let layer_idx = layer_str.parse::<usize>().ok()?;
    Some((layer_idx, suffix))
}

fn is_live_multi_rank_target(target: &str, recurrent_flags: &[bool]) -> bool {
    let Some((layer_idx, suffix)) = parse_blk_target(target) else {
        return false;
    };
    let Some(&is_recurrent) = recurrent_flags.get(layer_idx) else {
        return false;
    };
    match suffix {
        "ffn_down.weight" | "ffn_gate.weight" | "ffn_up.weight" => true,
        "attn_qkv.weight" | "attn_gate.weight" | "ssm_beta.weight" | "ssm_alpha.weight" => {
            is_recurrent
        }
        "attn_q.weight" | "attn_k.weight" | "attn_v.weight" | "attn_output.weight" => !is_recurrent,
        _ => false,
    }
}

fn decode_lora_words(tensor: &Tensor) -> Result<Vec<u32>, Error> {
    match tensor.kind {
        0 => {
            if !tensor.data.len().is_multiple_of(4) {
                return Err(Error::InvalidInput(format!(
                    "F32 LoRA tensor '{}' byte length is not a multiple of 4",
                    tensor.name
                )));
            }
            let mut words = Vec::with_capacity(tensor.data.len() / 4);
            for chunk in tensor.data.as_chunks::<4>().0 {
                let v = f32::from_le_bytes(*chunk);
                if !v.is_finite() {
                    return Err(Error::InvalidInput(format!(
                        "F32 LoRA tensor '{}' contains non-finite value",
                        tensor.name
                    )));
                }
                words.push(v.to_bits());
            }
            Ok(words)
        }
        1 => {
            if !tensor.data.len().is_multiple_of(2) {
                return Err(Error::InvalidInput(format!(
                    "F16 LoRA tensor '{}' byte length is not even",
                    tensor.name
                )));
            }
            let mut words = Vec::with_capacity(tensor.data.len() / 2);
            for chunk in tensor.data.as_chunks::<2>().0 {
                let v = half_to_float(u16::from_le_bytes(*chunk));
                if !v.is_finite() || v.abs() >= 65_520.0 {
                    return Err(Error::InvalidInput(format!(
                        "F16 LoRA tensor '{}' contains non-finite or overflowed value",
                        tensor.name
                    )));
                }
                words.push(v.to_bits());
            }
            Ok(words)
        }
        kind => Err(Error::InvalidInput(format!(
            "unsupported LoRA tensor '{}' encoding {kind}",
            tensor.name
        ))),
    }
}

fn upload_arena_words(
    words: &[u32],
    guard: &mut BufferCleanupGuard<'_>,
) -> Result<BufferSpan, Error> {
    let bytes = u32_slice_bytes(words.len())?;
    let span = guard.arena.alloc(guard.device, bytes)?;
    guard
        .device
        .write_buffer_u32_at(span.buffer, span.offset, words)?;
    Ok(span)
}

struct PreparedLinearLoraFactor {
    a_words: Vec<u32>,
    b_words: Vec<u32>,
    in_features: usize,
    out_features: usize,
    rank: usize,
    scaling: f32,
}

#[derive(Default)]
struct PreparedAdapters {
    multi_rank_linear: BTreeMap<String, PreparedLinearLoraFactor>,
    max_rank: usize,
    content_sha256: String,
    multi_rank_scale: Option<f32>,
}

fn prepare_adapters_upload(
    base_model: &Model,
    _geometry: &GraphGeometry,
    recurrent_flags: &[bool],
    _has_separate_output: bool,
    adapters: VulkanBatchedAdapters<'_>,
) -> Result<PreparedAdapters, Error> {
    let mut prepared = PreparedAdapters::default();
    let mut hasher = Hasher::new().map_err(|err| Error::InvalidInput(err.to_string()))?;
    hash_bytes(&mut hasher, b"vk-batched-adapters-v1:")?;

    if let Some(spec) = adapters.multi_rank {
        MultiRankAdapter::validate_scale(spec.scale)
            .map_err(|err| Error::InvalidInput(err.to_string()))?;
        let mr_scale = MultiRankAdapter::normalize_scale(spec.scale);
        let validated = MultiRankAdapter::from_model(base_model, spec.adapter, spec.policy)
            .map_err(|err| Error::InvalidInput(err.to_string()))?;
        let alpha = validated.alpha();
        let policy_tag = match spec.policy {
            MultiRankPolicy::NpuFriendly => "npu-friendly",
            MultiRankPolicy::AnyLinear => "any-linear",
        };
        hash_str(&mut hasher, b"mr-id:", validated.identity())?;
        hash_str(&mut hasher, b"mr-pol:", policy_tag)?;
        hash_u32(&mut hasher, alpha.to_bits())?;
        hash_usize(&mut hasher, validated.target_count())?;

        for target in validated.targets() {
            if !is_live_multi_rank_target(target, recurrent_flags) {
                return Err(Error::InvalidInput(
                    "multi-rank adapter target is outside the supported live GPU graph".to_owned(),
                ));
            }
            let a_tensor = spec
                .adapter
                .tensors
                .get(&format!("{target}.lora_a"))
                .ok_or_else(|| Error::InvalidInput("multi-rank lora_a missing".to_owned()))?;
            let b_tensor = spec
                .adapter
                .tensors
                .get(&format!("{target}.lora_b"))
                .ok_or_else(|| Error::InvalidInput("multi-rank lora_b missing".to_owned()))?;
            let [in_features, rank] = a_tensor.dimensions.as_slice() else {
                return Err(Error::InvalidInput(
                    "multi-rank lora_a must be rank 2".to_owned(),
                ));
            };
            let [b_rank, out_features] = b_tensor.dimensions.as_slice() else {
                return Err(Error::InvalidInput(
                    "multi-rank lora_b must be rank 2".to_owned(),
                ));
            };
            if *b_rank != *rank {
                return Err(Error::InvalidInput(
                    "multi-rank lora_a/lora_b rank mismatch".to_owned(),
                ));
            }
            let rank_u16 = u16::try_from(*rank).map_err(|_| {
                Error::ResourceLimit("multi-rank adapter rank exceeds u16".to_owned())
            })?;
            let scaling = alpha / f32::from(rank_u16);
            let a_words = decode_lora_words(a_tensor)?;
            let b_words = decode_lora_words(b_tensor)?;
            hash_str(&mut hasher, b"mr-tgt:", target)?;
            hash_usize(&mut hasher, *in_features)?;
            hash_usize(&mut hasher, *out_features)?;
            hash_usize(&mut hasher, *rank)?;
            hash_u32(&mut hasher, a_tensor.kind)?;
            for &w in &a_words {
                hash_u32(&mut hasher, w)?;
            }
            for &w in &b_words {
                hash_u32(&mut hasher, w)?;
            }
            prepared.multi_rank_linear.insert(
                target.to_owned(),
                PreparedLinearLoraFactor {
                    a_words,
                    b_words,
                    in_features: *in_features,
                    out_features: *out_features,
                    rank: *rank,
                    scaling,
                },
            );
            prepared.max_rank = prepared.max_rank.max(*rank);
        }
        prepared.multi_rank_scale = Some(mr_scale);
    } else {
        hash_bytes(&mut hasher, b"mr-none")?;
    }

    let digest = hasher
        .finish()
        .map_err(|err| Error::InvalidInput(err.to_string()))?;
    prepared.content_sha256 = hex::encode(digest);
    Ok(prepared)
}

fn parse_and_upload_adapters(
    base_model: &Model,
    geometry: &GraphGeometry,
    recurrent_flags: &[bool],
    has_separate_output: bool,
    adapters: VulkanBatchedAdapters<'_>,
    guard: &mut BufferCleanupGuard<'_>,
) -> Result<UploadedAdapters, Error> {
    let prepared = prepare_adapters_upload(
        base_model,
        geometry,
        recurrent_flags,
        has_separate_output,
        adapters,
    )?;
    let mut uploaded = UploadedAdapters {
        max_rank: prepared.max_rank,
        content_sha256: prepared.content_sha256,
        multi_rank_scale: prepared.multi_rank_scale,
        ..UploadedAdapters::default()
    };
    for (target, factor) in prepared.multi_rank_linear {
        let a_span = upload_arena_words(&factor.a_words, guard)?;
        let b_span = upload_arena_words(&factor.b_words, guard)?;
        uploaded.multi_rank_linear.insert(
            target,
            GpuLoraFactor {
                a_span,
                b_span,
                in_features: factor.in_features,
                out_features: factor.out_features,
                rank: factor.rank,
                scaling: factor.scaling,
            },
        );
    }
    Ok(uploaded)
}

#[expect(
    clippy::too_many_arguments,
    reason = "Shares reusable host repack and row scratch buffers across all layer matrix uploads."
)]
fn upload_matrix_by_name(
    model: &Model,
    name: &str,
    cols: usize,
    rows: usize,
    transforms: &GraphTransforms,
    adapters: &UploadedAdapters,
    repack_scratch: &mut Vec<u32>,
    row_scratch: &mut Vec<f32>,
    guard: &mut BufferCleanupGuard<'_>,
) -> Result<GpuMatrixWeight, Error> {
    let tensor = lookup_tensor(model, name, &[cols, rows])?;
    upload_matrix_tensor(
        &tensor,
        transforms,
        adapters,
        repack_scratch,
        row_scratch,
        guard,
    )
}

fn upload_matrix_tensor(
    tensor: &Tensor,
    transforms: &GraphTransforms,
    adapters: &UploadedAdapters,
    repack_scratch: &mut Vec<u32>,
    row_scratch: &mut Vec<f32>,
    guard: &mut BufferCleanupGuard<'_>,
) -> Result<GpuMatrixWeight, Error> {
    let [cols, rows] = tensor.dimensions.as_slice() else {
        return Err(Error::InvalidInput(format!(
            "matrix '{}' must be rank 2, got {:?}",
            tensor.name, tensor.dimensions
        )));
    };
    let (cols, rows) = (*cols, *rows);
    let (rotated, grouped_ssm) = transforms.basis(&tensor.name);

    let (encoding, word_len) = match tensor.kind {
        KIND_PQ2 if cols.is_multiple_of(128) => {
            validate_matrix(tensor, KIND_PQ2, 128)?;
            validate_weight_scales(KIND_PQ2, &tensor.data)?;
            repack_coalesced_into(KIND_PQ2, &tensor.data, cols, rows, repack_scratch)?;
            let enc = if cols.is_multiple_of(256) {
                WeightEncoding::Pq2Q8k
            } else {
                WeightEncoding::Pq2Q8_0
            };
            (enc, repack_scratch.len())
        }
        KIND_PTQ1 if cols.is_multiple_of(128) => {
            validate_matrix(tensor, KIND_PTQ1, 128)?;
            validate_weight_scales(KIND_PTQ1, &tensor.data)?;
            repack_coalesced_into(KIND_PTQ1, &tensor.data, cols, rows, repack_scratch)?;
            (WeightEncoding::Ptq1Q8_0, repack_scratch.len())
        }
        KIND_Q8_0 if cols.is_multiple_of(32) => {
            validate_matrix(tensor, KIND_Q8_0, 32)?;
            validate_weight_scales(KIND_Q8_0, &tensor.data)?;
            repack_coalesced_into(KIND_Q8_0, &tensor.data, cols, rows, repack_scratch)?;
            (WeightEncoding::Q8_0, repack_scratch.len())
        }
        _ => {
            let total = cols
                .checked_mul(rows)
                .ok_or_else(|| Error::ResourceLimit("F32 matrix element overflow".to_owned()))?;
            repack_scratch.clear();
            repack_scratch.reserve(total);
            for r in 0..rows {
                row_into(tensor, r, row_scratch)
                    .map_err(|err| Error::InvalidInput(err.to_string()))?;
                ensure_finite_f32(
                    row_scratch,
                    &format!("matrix '{}' contains non-finite value", tensor.name),
                )?;
                repack_scratch.extend(row_scratch.iter().copied().map(f32::to_bits));
            }
            (WeightEncoding::F32, repack_scratch.len())
        }
    };

    let bytes = u32_slice_bytes(word_len)?;
    let span = guard.arena.alloc(guard.device, bytes)?;
    guard
        .device
        .write_buffer_u32_at(span.buffer, span.offset, repack_scratch)?;
    let multi_rank = adapters
        .multi_rank_linear
        .get(&tensor.name)
        .copied()
        .filter(|lora| lora.in_features == cols && lora.out_features == rows);
    Ok(GpuMatrixWeight {
        span,
        encoding,
        cols,
        rows,
        rotated,
        grouped_ssm,
        multi_rank,
    })
}

#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "Uploads all tensors and initial zeroed GPU state buffers for one recurrent GDN layer."
)]
fn upload_recurrent_attention(
    model: &Model,
    prefix: &str,
    geometry: &GraphGeometry,
    transforms: &GraphTransforms,
    adapters: &UploadedAdapters,
    repack_scratch: &mut Vec<u32>,
    row_scratch: &mut Vec<f32>,
    guard: &mut BufferCleanupGuard<'_>,
) -> Result<GpuLayerAttention, Error> {
    let channels = geometry.convolution_width();
    let qkv = upload_matrix_by_name(
        model,
        &format!("{prefix}attn_qkv.weight"),
        geometry.embedding,
        channels,
        transforms,
        adapters,
        repack_scratch,
        row_scratch,
        guard,
    )?;
    let gate = upload_matrix_by_name(
        model,
        &format!("{prefix}attn_gate.weight"),
        geometry.embedding,
        geometry.value_width(),
        transforms,
        adapters,
        repack_scratch,
        row_scratch,
        guard,
    )?;
    let beta = upload_matrix_by_name(
        model,
        &format!("{prefix}ssm_beta.weight"),
        geometry.embedding,
        geometry.ssm_values,
        transforms,
        adapters,
        repack_scratch,
        row_scratch,
        guard,
    )?;
    let alpha = upload_matrix_by_name(
        model,
        &format!("{prefix}ssm_alpha.weight"),
        geometry.embedding,
        geometry.ssm_values,
        transforms,
        adapters,
        repack_scratch,
        row_scratch,
        guard,
    )?;

    let conv_tensor = lookup_tensor(
        model,
        &format!("{prefix}ssm_conv1d.weight"),
        &[geometry.convolution, channels],
    )?;
    let mut conv_flat = Vec::with_capacity(geometry.convolution * channels);
    for ch in 0..channels {
        row_into(&conv_tensor, ch, row_scratch)
            .map_err(|err| Error::InvalidInput(err.to_string()))?;
        ensure_finite_f32(row_scratch, "conv1d weight contains non-finite value")?;
        conv_flat.extend_from_slice(row_scratch);
    }
    let convolution_w = guard.alloc_f32_buffer(&conv_flat)?;

    let time_bias = load_and_upload_vector(
        model,
        &format!("{prefix}ssm_dt.bias"),
        geometry.ssm_values,
        row_scratch,
        guard,
    )?;

    let ssm_a_tensor = lookup_tensor(model, &format!("{prefix}ssm_a"), &[geometry.ssm_values])?;
    row_into(&ssm_a_tensor, 0, row_scratch).map_err(|err| Error::InvalidInput(err.to_string()))?;
    if row_scratch.iter().any(|&v| !v.is_finite() || v > 0.0) {
        return Err(Error::InvalidInput(
            "recurrent decay ssm_a must be finite and nonpositive".to_owned(),
        ));
    }
    let ssm_a = guard.alloc_f32_buffer(row_scratch)?;

    let norm_w = load_and_upload_vector(
        model,
        &format!("{prefix}ssm_norm.weight"),
        geometry.ssm_head,
        row_scratch,
        guard,
    )?;
    let output = upload_matrix_by_name(
        model,
        &format!("{prefix}ssm_out.weight"),
        geometry.value_width(),
        geometry.embedding,
        transforms,
        adapters,
        repack_scratch,
        row_scratch,
        guard,
    )?;

    let conv_state_elems = (geometry.convolution - 1) * channels;
    let conv_state = if conv_state_elems > 0 {
        let conv_bytes = f32_slice_bytes(conv_state_elems)?;
        let id = guard.alloc_readback_storage(conv_bytes)?;
        guard.device.fill_buffer_zero(id, 0, conv_bytes)?;
        id
    } else {
        guard.alloc_storage(MIN_NORM_OR_BIAS_BYTES)?
    };

    let matrix_state_elems = geometry.ssm_head * geometry.ssm_head * geometry.ssm_values;
    let matrix_bytes = f32_slice_bytes(matrix_state_elems)?;
    let matrix_state = guard.alloc_readback_storage(matrix_bytes)?;
    guard
        .device
        .fill_buffer_zero(matrix_state, 0, matrix_bytes)?;

    Ok(GpuLayerAttention::Recurrent {
        qkv,
        gate,
        beta,
        alpha,
        convolution_w,
        time_bias,
        ssm_a,
        norm_w,
        output,
        conv_state,
        conv_state_elems,
        matrix_state,
        matrix_state_elems,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "Uploads all tensors and allocates GPU KV caches for one full-attention layer."
)]
fn upload_full_attention(
    model: &Model,
    prefix: &str,
    geometry: &GraphGeometry,
    kv_precision: VulkanKvPrecision,
    transforms: &GraphTransforms,
    adapters: &UploadedAdapters,
    repack_scratch: &mut Vec<u32>,
    row_scratch: &mut Vec<f32>,
    guard: &mut BufferCleanupGuard<'_>,
) -> Result<GpuLayerAttention, Error> {
    let query = upload_matrix_by_name(
        model,
        &format!("{prefix}attn_q.weight"),
        geometry.embedding,
        geometry.head * geometry.heads * 2,
        transforms,
        adapters,
        repack_scratch,
        row_scratch,
        guard,
    )?;
    let key = upload_matrix_by_name(
        model,
        &format!("{prefix}attn_k.weight"),
        geometry.embedding,
        geometry.kv_width(),
        transforms,
        adapters,
        repack_scratch,
        row_scratch,
        guard,
    )?;
    let value = upload_matrix_by_name(
        model,
        &format!("{prefix}attn_v.weight"),
        geometry.embedding,
        geometry.kv_width(),
        transforms,
        adapters,
        repack_scratch,
        row_scratch,
        guard,
    )?;
    let output = upload_matrix_by_name(
        model,
        &format!("{prefix}attn_output.weight"),
        geometry.q_width(),
        geometry.embedding,
        transforms,
        adapters,
        repack_scratch,
        row_scratch,
        guard,
    )?;
    let query_norm_w = load_and_upload_vector(
        model,
        &format!("{prefix}attn_q_norm.weight"),
        geometry.head,
        row_scratch,
        guard,
    )?;
    let key_norm_w = load_and_upload_vector(
        model,
        &format!("{prefix}attn_k_norm.weight"),
        geometry.head,
        row_scratch,
        guard,
    )?;

    let kv_elems = geometry
        .context_limit
        .checked_mul(geometry.kv_width())
        .ok_or_else(|| Error::ResourceLimit("KV cache size overflow".to_owned()))?;
    let cache_bytes = match kv_precision {
        VulkanKvPrecision::Fp16 => u32_slice_bytes(kv_elems / 2)?,
        VulkanKvPrecision::Fp32 => f32_slice_bytes(kv_elems)?,
    }
    .max(MIN_NORM_OR_BIAS_BYTES);
    let k_cache = guard.alloc_storage(cache_bytes)?;
    let v_cache = guard.alloc_storage(cache_bytes)?;
    guard.device.fill_buffer_zero(k_cache, 0, cache_bytes)?;
    guard.device.fill_buffer_zero(v_cache, 0, cache_bytes)?;

    Ok(GpuLayerAttention::Full {
        query,
        key,
        value,
        output,
        query_norm_w,
        key_norm_w,
        k_cache,
        v_cache,
    })
}

struct PreparedTileUpload {
    staged_bank: Option<usize>,
    has_full_layers: bool,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum FinalLayerMode {
    FullAllTokens,
    IngestOnly,
    LastTokenLogits,
}

struct AttentionChunkSpans {
    q_in: BufferSpan,
    o_out: BufferSpan,
    copy_back: Option<(u32, u32)>,
}

struct TileScratchPlan {
    allocated_tile_tokens: usize,
    score_bytes: u64,
    sizes: [u64; 27],
}

fn compute_max_wg_tile_tokens(
    geometry: &GraphGeometry,
    max_wg: [u32; 3],
    requested: usize,
) -> usize {
    let wg_x = usize::try_from(max_wg[0]).unwrap_or(65_535);
    let wg_y = usize::try_from(max_wg[1]).unwrap_or(65_535);
    let wg_z = usize::try_from(max_wg[2]).unwrap_or(65_535);
    let wg_block = usize::try_from(WORKGROUP_X).unwrap_or(64);
    let max_heads = [geometry.heads, geometry.kv_heads, geometry.ssm_values, 1]
        .into_iter()
        .max()
        .unwrap_or(1);
    let max_1d_per_tok = [
        geometry.embedding,
        geometry.feed_forward,
        geometry.value_width(),
        geometry.kv_width(),
        geometry.q_width(),
        1,
    ]
    .into_iter()
    .max()
    .unwrap_or(1);

    let cap_y = wg_y / max_heads;
    let cap_1d = wg_x.saturating_mul(wg_block) / max_1d_per_tok;
    requested.min(cap_y).min(wg_z).min(cap_1d).max(1)
}

struct TileScratchInputs {
    embedding: usize,
    max_f32_work: usize,
    act_q8k_words_per_tok: usize,
    act_q8_0_words_per_tok: usize,
    rank_per_tok: usize,
    rope: usize,
    score_per_tok: usize,
    key_scratch_bytes: u64,
    val_scratch_bytes: u64,
    logits_bytes: u64,
}

fn build_tile_scratch_plan(
    inputs: &TileScratchInputs,
    tile_tokens: usize,
    score_tokens: usize,
) -> Result<TileScratchPlan, Error> {
    let hidden_bytes = f32_slice_bytes(tile_tokens * inputs.embedding)?.max(MIN_SCRATCH_BYTES);
    let work_bytes = f32_slice_bytes(tile_tokens * inputs.max_f32_work)?.max(MIN_SCRATCH_BYTES);
    let act_q8k_bytes =
        u32_slice_bytes(tile_tokens * inputs.act_q8k_words_per_tok)?.max(MIN_SCRATCH_BYTES);
    let act_q8_0_bytes =
        u32_slice_bytes(tile_tokens * inputs.act_q8_0_words_per_tok)?.max(MIN_SCRATCH_BYTES);
    let lora_ranks_bytes =
        f32_slice_bytes(tile_tokens * inputs.rank_per_tok)?.max(MIN_SCRATCH_BYTES);
    let rope_bytes = f32_slice_bytes(tile_tokens * inputs.rope)?.max(MIN_SCRATCH_BYTES);
    let score_elems = score_tokens
        .checked_mul(inputs.score_per_tok)
        .ok_or_else(|| Error::ResourceLimit("attention score buffer overflow".to_owned()))?;
    let score_bytes = f32_slice_bytes(score_elems)?.max(MIN_SCRATCH_BYTES);

    let sizes = [
        hidden_bytes,
        hidden_bytes,
        hidden_bytes,
        work_bytes,
        work_bytes,
        work_bytes,
        act_q8k_bytes,
        act_q8_0_bytes,
        work_bytes,
        work_bytes,
        work_bytes,
        work_bytes,
        work_bytes,
        inputs.key_scratch_bytes,
        inputs.val_scratch_bytes,
        rope_bytes,
        lora_ranks_bytes,
        score_bytes,
        score_bytes,
        inputs.logits_bytes,
        MIN_NORM_OR_BIAS_BYTES,
        hidden_bytes,
        hidden_bytes,
        rope_bytes,
        rope_bytes,
        lora_ranks_bytes,
        lora_ranks_bytes,
    ];
    Ok(TileScratchPlan {
        allocated_tile_tokens: tile_tokens,
        score_bytes,
        sizes,
    })
}

fn check_tile_scratch_limits(
    plan: &TileScratchPlan,
    scratch_budget_bytes: u64,
    max_storage_range: u32,
) -> Result<(), Error> {
    let range_u64 = u64::from(max_storage_range);
    let total_scratch: u64 = plan.sizes.iter().try_fold(0_u64, |acc, &sz| {
        if sz > range_u64 {
            return Err(Error::ResourceLimit(format!(
                "scratch buffer size {sz} exceeds maxStorageBufferRange {max_storage_range}"
            )));
        }
        acc.checked_add(sz)
            .ok_or_else(|| Error::ResourceLimit("total scratch size overflow".to_owned()))
    })?;
    if scratch_budget_bytes > 0 && total_scratch > scratch_budget_bytes {
        return Err(Error::ResourceLimit(format!(
            "total GPU graph scratch {total_scratch} exceeds budget {scratch_budget_bytes}"
        )));
    }
    Ok(())
}

fn plan_tile_scratch(
    geometry: &GraphGeometry,
    max_adapter_rank: usize,
    prefill_tile_tokens: usize,
    scratch_budget_bytes: u64,
    max_storage_range: u32,
    max_wg: [u32; 3],
) -> Result<TileScratchPlan, Error> {
    let max_proj_in = [
        geometry.embedding,
        geometry.feed_forward,
        geometry.q_width(),
        geometry.value_width(),
    ]
    .into_iter()
    .max()
    .unwrap_or(geometry.embedding);

    let max_proj_out = [
        geometry.embedding,
        geometry.feed_forward,
        2 * geometry.q_width(),
        geometry.convolution_width(),
        geometry.value_width(),
        geometry.kv_width(),
        geometry.ssm_values,
    ]
    .into_iter()
    .max()
    .unwrap_or(geometry.embedding);

    let max_f32_work = max_proj_in.max(max_proj_out);
    let score_per_tok = geometry
        .context_limit
        .checked_mul(geometry.heads)
        .ok_or_else(|| Error::ResourceLimit("attention score buffer overflow".to_owned()))?;
    let inputs = TileScratchInputs {
        embedding: geometry.embedding,
        max_f32_work,
        act_q8k_words_per_tok: max_proj_in.div_ceil(128) * Q8K_4PLANE_WORDS_PER_HALF,
        act_q8_0_words_per_tok: max_proj_in.div_ceil(32) * Q8_0_WORDS_PER_BLOCK,
        rank_per_tok: max_adapter_rank.max(1),
        rope: geometry.rope,
        score_per_tok,
        key_scratch_bytes: f32_slice_bytes(geometry.key_width())?.max(MIN_SCRATCH_BYTES),
        val_scratch_bytes: f32_slice_bytes(geometry.value_width())?.max(MIN_SCRATCH_BYTES),
        logits_bytes: f32_slice_bytes(geometry.vocabulary)?.max(MIN_SCRATCH_BYTES),
    };

    let mut tile_cap = compute_max_wg_tile_tokens(geometry, max_wg, prefill_tile_tokens);
    loop {
        let candidate = build_tile_scratch_plan(&inputs, tile_cap, 1)?;
        match check_tile_scratch_limits(&candidate, scratch_budget_bytes, max_storage_range) {
            Ok(()) => break,
            Err(_) if tile_cap > 1 => {
                tile_cap -= 1;
            }
            Err(err) => return Err(err),
        }
    }

    let mut score_tokens = tile_cap;
    let chosen = loop {
        if let Ok(candidate) = build_tile_scratch_plan(&inputs, tile_cap, score_tokens)
            && check_tile_scratch_limits(&candidate, scratch_budget_bytes, max_storage_range)
                .is_ok()
        {
            break candidate;
        }
        score_tokens -= 1;
    };
    Ok(chosen)
}

fn allocate_scratch(
    geometry: &GraphGeometry,
    max_adapter_rank: usize,
    prefill_tile_tokens: usize,
    scratch_budget_bytes: u64,
    max_storage_range: u32,
    max_wg: [u32; 3],
    guard: &mut BufferCleanupGuard<'_>,
) -> Result<GpuGraphScratch, Error> {
    let plan = plan_tile_scratch(
        geometry,
        max_adapter_rank,
        prefill_tile_tokens,
        scratch_budget_bytes,
        max_storage_range,
        max_wg,
    )?;
    let sizes = plan.sizes;

    Ok(GpuGraphScratch {
        hidden: guard.alloc_storage(sizes[0])?,
        hidden_staging: [
            guard.alloc_storage(sizes[21])?,
            guard.alloc_storage(sizes[22])?,
        ],
        norm: guard.alloc_storage(sizes[1])?,
        delta: guard.alloc_storage(sizes[2])?,
        transformed: guard.alloc_storage(sizes[3])?,
        perm_tmp: guard.alloc_storage(sizes[4])?,
        sign_tmp: guard.alloc_storage(sizes[5])?,
        act_q8k: guard.alloc_storage(sizes[6])?,
        act_q8_0: guard.alloc_storage(sizes[7])?,
        proj_a: guard.alloc_storage(sizes[8])?,
        proj_b: guard.alloc_storage(sizes[9])?,
        proj_c: guard.alloc_storage(sizes[10])?,
        proj_d: guard.alloc_storage(sizes[11])?,
        proj_e: guard.alloc_storage(sizes[12])?,
        gdn_k: guard.alloc_storage(sizes[13])?,
        gdn_v: guard.alloc_storage(sizes[14])?,
        rope_freqs: guard.alloc_storage(sizes[15])?,
        rope_freqs_staging: [
            guard.alloc_storage(sizes[23])?,
            guard.alloc_storage(sizes[24])?,
        ],
        lora_ranks: guard.alloc_storage(sizes[16])?,
        lora_ranks_staging: [
            guard.alloc_storage(sizes[25])?,
            guard.alloc_storage(sizes[26])?,
        ],
        attn_scores: guard.alloc_storage(sizes[17])?,
        attn_probs: guard.alloc_storage(sizes[18])?,
        attn_score_bytes: plan.score_bytes,
        allocated_tile_tokens: plan.allocated_tile_tokens,
        logits: guard.alloc_readback_storage(sizes[19])?,
        dummy_state: guard.alloc_storage(sizes[20])?,
    })
}

fn fill_rope_frequencies(
    geometry: &GraphGeometry,
    mrope_positions: [usize; 4],
    dst: &mut [f32],
) -> Result<(), Error> {
    let half = geometry.rope / 2;
    if dst.len() != geometry.rope {
        return Err(Error::InvalidInput(
            "rope frequency buffer length mismatch".to_owned(),
        ));
    }
    let rope_f64 = f64::from(
        u32::try_from(geometry.rope)
            .map_err(|_| Error::ResourceLimit("rope_dim exceeds u32".to_owned()))?,
    );
    let base_f64 = f64::from(geometry.rope_base);
    for pair in 0..half {
        let sector = pair % 3;
        let axis_idx = if pair < 3 * geometry.rope_sections[sector] {
            sector
        } else {
            3
        };
        let pos_u32 = u32::try_from(mrope_positions[axis_idx])
            .map_err(|_| Error::ResourceLimit("mrope position exceeds u32".to_owned()))?;
        let exp = f64::from(
            u32::try_from(pair * 2)
                .map_err(|_| Error::ResourceLimit("rope pair index exceeds u32".to_owned()))?,
        ) / rope_f64;
        let inv_freq = 1.0_f64 / base_f64.powf(exp);
        let angle = f64::from(pos_u32) * inv_freq;
        dst[pair * 2] = f64_to_f32_ieee(angle.cos());
        dst[pair * 2 + 1] = f64_to_f32_ieee(angle.sin());
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "Binds all static provenance inputs (device, model, chunk policy, kernel path, shaders, adapters) into one SHA-256."
)]
fn compute_base_provenance(
    device: &AdrenoVulkanDevice,
    model: &Model,
    config: &VulkanBatchedConfig,
    resolved_context_limit: usize,
    resolved_pinned_arithmetic: bool,
    kernel_path: ProjectionKernelPath,
    layer_count: usize,
    pipelines: &VulkanLayerPipelines,
    adapter_content_sha256: &str,
) -> Result<String, Error> {
    let mut hasher = Hasher::new().map_err(|err| Error::InvalidInput(err.to_string()))?;
    hash_str(&mut hasher, b"vk-batched-graph-v2:", &model.identity)?;
    hash_str(&mut hasher, b"dev:", &device.capabilities().identity_sha256)?;
    hash_str(&mut hasher, b"shaders:", pipelines.shader_digests_sha256())?;
    hash_str(&mut hasher, b"path:", kernel_path.as_tag())?;
    hash_str(&mut hasher, b"kv:", config.kv_precision.name())?;
    let strat_tag = match config.attention_strategy {
        AttentionStrategy::Auto => "auto",
        AttentionStrategy::ForceFused => "force-fused",
        AttentionStrategy::ForceDecomposed => "force-decomposed",
    };
    hash_str(&mut hasher, b"attn_strat:", strat_tag)?;
    hash_usize(&mut hasher, config.tdr_policy.layers_per_submit)?;
    hash_usize(&mut hasher, config.tdr_policy.max_stages_per_submit)?;
    hash_usize(&mut hasher, resolved_context_limit)?;
    hash_usize(&mut hasher, layer_count)?;
    hash_u32(&mut hasher, u32::from(resolved_pinned_arithmetic))?;
    hash_u64(&mut hasher, config.scratch_budget_bytes)?;
    hash_usize(&mut hasher, config.prefill_tile_tokens)?;
    hash_str(&mut hasher, b"adapters:", adapter_content_sha256)?;

    // Also include sample attention cache key to tie planner geometry/limits cleanly.
    let sample_desc = AttentionDescriptor {
        query_len: 1,
        total_seq: 1,
        past_len: 0,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: 2,
        scale: 1.0,
        causal: true,
    };
    let sample_plan = select_attention_plan(
        &sample_desc,
        device.capabilities(),
        config.attention_strategy,
        config.scratch_budget_bytes,
    )?;
    let (qk_digest, pv_digest, fused_digest) = match config.kv_precision {
        VulkanKvPrecision::Fp16 => (
            &pipelines.attn_qk_f16kv_digest,
            &pipelines.attn_pv_f16kv_digest,
            &pipelines.attn_fused_f16kv_digest,
        ),
        VulkanKvPrecision::Fp32 => (
            &pipelines.attn_qk_digest,
            &pipelines.attn_pv_digest,
            &pipelines.attn_fused_digest,
        ),
    };
    let sample_key = attention_cache_key(
        &sample_desc,
        &sample_plan,
        device.capabilities(),
        qk_digest,
        &pipelines.attn_softmax_digest,
        pv_digest,
        fused_digest,
    )?;
    hash_str(&mut hasher, b"attn_sample:", &sample_key)?;
    hash_str(
        &mut hasher,
        b"attn_gqa_prefill_f16kv:",
        &pipelines.attn_fused_gqa_prefill_f16kv_digest,
    )?;

    let digest = hasher
        .finish()
        .map_err(|err| Error::InvalidInput(err.to_string()))?;
    Ok(hex::encode(digest))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FusedAttentionKernelKind {
    PerHead,
    GqaGrouped6PrefillF16Kv,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FusedAttentionDispatchChoice {
    kind: FusedAttentionKernelKind,
    grid: [u32; 3],
}

const fn select_fused_attention_dispatch(
    desc: &AttentionDescriptor,
    kv_precision: VulkanKvPrecision,
    max_compute_shared_memory_size: u32,
) -> FusedAttentionDispatchChoice {
    let gqa6_matches = desc.query_len > 1
        && matches!(kv_precision, VulkanKvPrecision::Fp16)
        && desc.num_kv_heads > 0
        && desc.num_heads.is_multiple_of(desc.num_kv_heads)
        && (desc.num_heads / desc.num_kv_heads) == spirv::ATTN_FUSED_GQA6_RATIO
        && desc.total_seq <= spirv::ATTN_FUSED_GQA6_MAX_TOTAL_SEQ
        && (max_compute_shared_memory_size / 2) >= spirv::ATTN_FUSED_GQA6_SHARED_MEMORY_BYTES;
    if gqa6_matches {
        FusedAttentionDispatchChoice {
            kind: FusedAttentionKernelKind::GqaGrouped6PrefillF16Kv,
            grid: [1, desc.num_kv_heads, desc.query_len],
        }
    } else {
        FusedAttentionDispatchChoice {
            kind: FusedAttentionKernelKind::PerHead,
            grid: [1, desc.num_heads, desc.query_len],
        }
    }
}

fn compute_executor_identity(
    base_provenance_sha256: &str,
    multi_rank_scale: Option<f32>,
) -> Result<String, Error> {
    let mut hasher = Hasher::new().map_err(|err| Error::InvalidInput(err.to_string()))?;
    hash_str(
        &mut hasher,
        b"vk-batched-exec-id-v1:",
        base_provenance_sha256,
    )?;
    match multi_rank_scale {
        Some(scale) => {
            hash_bytes(&mut hasher, &[1])?;
            hash_u32(&mut hasher, scale.to_bits())?;
        }
        None => hash_bytes(&mut hasher, &[0])?,
    }
    let digest = hasher
        .finish()
        .map_err(|err| Error::InvalidInput(err.to_string()))?;
    Ok(hex::encode(digest))
}

fn f64_to_f32_ieee(value: f64) -> f32 {
    if value == 0.0 {
        return if value.is_sign_negative() {
            -0.0_f32
        } else {
            0.0_f32
        };
    }
    if value.is_nan() {
        return f32::NAN;
    }
    if value.is_infinite() {
        return if value.is_sign_negative() {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
    }
    let bits = value.to_bits();
    let sign = u32::try_from((bits >> 63) & 1).unwrap_or(0) << 31;
    let exp64 = i32::try_from((bits >> 52) & 0x7FF).unwrap_or(0);
    let mant64 = bits & 0x000F_FFFF_FFFF_FFFF;
    let unbiased = exp64 - 1023;
    if unbiased > 127 {
        return f32::from_bits(sign | 0x7F80_0000);
    }
    if unbiased >= -126 {
        let exp32 = u32::try_from(unbiased + 127).unwrap_or(0);
        let mant_hi = u32::try_from(mant64 >> 29).unwrap_or(0);
        let rem = mant64 & ((1_u64 << 29) - 1);
        let half = 1_u64 << 28;
        let round_up = rem > half || (rem == half && (mant_hi & 1) != 0);
        let raw = sign | (exp32 << 23) | mant_hi;
        return f32::from_bits(if round_up { raw + 1 } else { raw });
    }
    let shift = -126 - unbiased;
    if shift > 24 {
        return f32::from_bits(sign);
    }
    let full_mant = (1_u64 << 52) | mant64;
    let total_shift = u32::try_from(29 + shift).unwrap_or(53);
    let sub_mant = u32::try_from(full_mant >> total_shift).unwrap_or(0);
    let rem_mask = (1_u64 << total_shift) - 1;
    let rem = full_mant & rem_mask;
    let half = 1_u64 << (total_shift - 1);
    let round_up = rem > half || (rem == half && (sub_mant & 1) != 0);
    let raw = sign | sub_mant;
    f32::from_bits(if round_up { raw + 1 } else { raw })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tdr_chunk_policy_validates_bounds_and_partitions_layers() -> Result<(), Error> {
        let default_policy = TdrChunkPolicy::default();
        assert_eq!(default_policy.layers_per_submit, 8);
        assert_eq!(default_policy.max_stages_per_submit, 2048);
        assert_eq!(default_policy.expected_submits_for_layers(64)?, 8);
        assert_eq!(default_policy.expected_submits_for_layers(65)?, 9);
        assert_eq!(default_policy.expected_submits_for_layers(0)?, 0);

        let ranges = default_policy.layer_chunk_ranges(19)?;
        assert_eq!(ranges, vec![0..8, 8..16, 16..19]);

        assert!(TdrChunkPolicy::new(0, 1024).is_err());
        assert!(TdrChunkPolicy::new(TdrChunkPolicy::MAX_LAYERS_PER_SUBMIT + 1, 1024).is_err());
        assert!(TdrChunkPolicy::new(8, 0).is_err());
        assert!(TdrChunkPolicy::new(8, MAX_DESCRIPTOR_SETS_PER_SUBMIT + 1).is_err());
        Ok(())
    }

    #[test]
    fn f64_to_f32_ieee_matches_exact_bit_cases() {
        for sample in [
            0.0_f64,
            -0.0_f64,
            1e-6_f64,
            10_000.0_f64,
            -1.234_567_89_f64,
            1e-40_f64,
            1e40_f64,
        ] {
            #[expect(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "Test oracle verifies bit-exact equivalence against hardware IEEE f64->f32 cast."
            )]
            let oracle = sample as f32;
            assert_eq!(f64_to_f32_ieee(sample).to_bits(), oracle.to_bits());
        }
        assert!(f64_to_f32_ieee(f64::NAN).is_nan());
        assert_eq!(
            f64_to_f32_ieee(f64::INFINITY).to_bits(),
            f32::INFINITY.to_bits()
        );
        assert_eq!(
            f64_to_f32_ieee(f64::NEG_INFINITY).to_bits(),
            f32::NEG_INFINITY.to_bits()
        );
    }

    #[test]
    fn batched_checkpoint_identity_is_deterministic_and_sensitive_to_state() -> Result<(), Error> {
        let make = |pos: usize, val: f32, nonce: u64, kv_gen: u64| VulkanBatchedCheckpoint {
            executor_identity_sha256: "exec-abc".to_owned(),
            provenance_nonce: nonce,
            position: pos,
            mrope_positions: [pos, pos, pos, 0],
            kv_generation: kv_gen,
            kv_prefix_watermark: pos,
            layers: vec![
                LayerStateCheckpoint::Recurrent {
                    convolution: vec![val, -val],
                    matrix: vec![val; 4],
                },
                LayerStateCheckpoint::Full {
                    precision: VulkanKvPrecision::Fp16,
                    width: 4,
                    keys_words: vec![0x3c00_3c00; pos * 2],
                    values_words: vec![0x4000_4000; pos * 2],
                },
            ],
        };
        let snap_base = make(1, 0.25, 0, 1);
        let snap_repeat = make(1, 0.25, 0, 9);
        let snap_pos2 = make(2, 0.25, 0, 1);
        let snap_val_half = make(1, 0.50, 0, 1);
        let snap_layer_one = make(1, 0.25, 1, 1);
        assert_eq!(snap_base.identity_sha256()?, snap_repeat.identity_sha256()?);
        assert_eq!(snap_repeat.kv_generation(), 9);
        assert_eq!(snap_repeat.kv_prefix_watermark(), 1);
        assert_ne!(snap_base.identity_sha256()?, snap_pos2.identity_sha256()?);
        assert_ne!(
            snap_base.identity_sha256()?,
            snap_val_half.identity_sha256()?
        );
        assert_ne!(
            snap_base.identity_sha256()?,
            snap_layer_one.identity_sha256()?
        );
        Ok(())
    }

    #[test]
    fn kv_prefix_tracker_handles_append_rollback_and_branch_overwrites() {
        let mut tracker = KvPrefixTracker::new();
        assert_eq!(tracker.current_gen, 1);
        assert_eq!(tracker.high_water, 0);

        // Prompt prefill [0..100) + checkpoint C100.
        tracker.begin_write(0, 100);
        let c100_gen = tracker.current_gen;
        let c100_wm = tracker.high_water;
        assert_eq!((c100_gen, c100_wm), (1, 100));

        // Branch A generates [100..150) + mid checkpoint.
        tracker.begin_write(100, 20);
        let branch_a_gen = tracker.current_gen;
        let branch_a_wm = tracker.high_water;
        tracker.begin_write(120, 30);
        assert_eq!(tracker.current_gen, 1);
        assert_eq!(tracker.high_water, 150);
        assert!(tracker.can_skip_kv_restore(true, c100_gen, c100_wm, 100));
        assert!(tracker.can_skip_kv_restore(true, branch_a_gen, branch_a_wm, 120));

        // Append-only rollback to C100 leaves C100 prefix intact without full KV copy.
        assert!(tracker.can_skip_kv_restore(true, c100_gen, c100_wm, 100));

        // Branch B writes at [100..140): C100 [0..100) remains intact across turns,
        // while Branch A [0..120) had [100..120) overwritten and MUST fall back to full KV restore!
        tracker.begin_write(100, 40);
        assert_eq!(tracker.current_gen, 2);
        assert_eq!(tracker.high_water, 140);
        assert!(tracker.can_skip_kv_restore(true, c100_gen, c100_wm, 100));
        assert!(!tracker.can_skip_kv_restore(true, branch_a_gen, branch_a_wm, 120));
        assert_eq!(tracker.intact_prefix_for(branch_a_gen, branch_a_wm), 100);

        // Turn 3 overwrites [100..130) again from C100: C100 STILL fast-restores!
        let next_gen = tracker.current_gen;
        let next_wm = 130_usize;
        assert!(tracker.can_skip_kv_restore(true, next_gen, next_wm, 130));
        tracker.begin_write(100, 30);
        assert!(tracker.can_skip_kv_restore(true, c100_gen, c100_wm, 100));
        assert!(!tracker.can_skip_kv_restore(true, next_gen, next_wm, 130));

        // Deep rollback writing at [50..80) invalidates C100 [50..100) cleanly.
        tracker.begin_write(50, 30);
        assert!(!tracker.can_skip_kv_restore(true, c100_gen, c100_wm, 100));
        assert_eq!(tracker.intact_prefix_for(c100_gen, c100_wm), 50);

        // Full KV restore & inconsistent state force full fallback for stale checkpoints.
        assert!(!tracker.can_skip_kv_restore(false, tracker.current_gen, tracker.high_water, 80));
        assert!(!tracker.can_skip_kv_restore(false, c100_gen, 0, 0));
        tracker.on_full_kv_restore(120);
        assert!(!tracker.can_skip_kv_restore(true, c100_gen, c100_wm, 50));
        tracker.on_reset();
        assert_eq!(tracker.high_water, 0);
        assert_eq!(tracker.intact_prefix_for(c100_gen, c100_wm), 0);
    }

    #[test]
    fn executor_identity_changes_with_adapter_scales() -> Result<(), Error> {
        let base = "base-prov-123";
        let id_none = compute_executor_identity(base, None)?;
        let id_mr_1 = compute_executor_identity(base, Some(1.0))?;
        let id_mr_0 = compute_executor_identity(base, Some(0.0))?;
        let id_mr_half = compute_executor_identity(base, Some(0.5))?;
        assert_ne!(id_none, id_mr_1);
        assert_ne!(id_mr_1, id_mr_0);
        assert_ne!(id_mr_1, id_mr_half);
        assert_eq!(id_mr_1, compute_executor_identity(base, Some(1.0))?);
        Ok(())
    }

    #[test]
    fn tile_scratch_plan_respects_budget_and_falls_back_cleanly() -> Result<(), Error> {
        let geometry = GraphGeometry {
            embedding: 512,
            feed_forward: 1024,
            vocabulary: 4096,
            heads: 16,
            kv_heads: 4,
            head: 64,
            rope: 64,
            rope_sections: [8, 8, 8, 0],
            rope_base: 10_000.0,
            epsilon: 1e-6,
            ssm_head: 64,
            ssm_keys: 4,
            ssm_values: 8,
            convolution: 4,
            context_limit: 2048,
        };
        let wg = [65_535, 65_535, 65_535];
        let full = plan_tile_scratch(&geometry, 8, 16, 64 * 1024 * 1024, u32::MAX, wg)?;
        assert_eq!(full.allocated_tile_tokens, 16);
        assert_eq!(full.score_bytes, 16 * 2048 * 16 * 4);

        // Constrain budget so full 16-token non-score buffers fit (~1 MiB) while score scratch
        // must step down below 16 tokens without failing initialization.
        let one_tok_scores = 2048_u64 * 16 * 4;
        let non_score_16: u64 = full
            .sizes
            .iter()
            .enumerate()
            .filter(|&(idx, _)| idx != 17 && idx != 18)
            .map(|(_, &sz)| sz)
            .sum();
        let tight_budget = non_score_16 + 4 * one_tok_scores;
        let sub_score = plan_tile_scratch(&geometry, 8, 16, tight_budget, u32::MAX, wg)?;
        assert_eq!(sub_score.allocated_tile_tokens, 16);
        assert_eq!(sub_score.score_bytes, 2 * one_tok_scores);

        // Extremely tight budget shrinks tile_cap toward 1 cleanly.
        let plan_1tok = plan_tile_scratch(&geometry, 8, 1, 0, u32::MAX, wg)?;
        let min_needed: u64 = plan_1tok.sizes.iter().sum();
        let shrunk = plan_tile_scratch(&geometry, 8, 16, min_needed, u32::MAX, wg)?;
        assert_eq!(shrunk.allocated_tile_tokens, 1);
        assert!(plan_tile_scratch(&geometry, 8, 16, min_needed - 1, u32::MAX, wg).is_err());
        Ok(())
    }

    fn test_heterogeneous_f32_tensor(name: &str, dims: &[usize], val: f32) -> Tensor {
        let count: usize = dims.iter().product();
        let bytes: Vec<u8> = vec![val; count]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        Tensor {
            name: name.to_owned(),
            dimensions: dims.to_vec(),
            kind: 0,
            data: std::sync::Arc::<[u8]>::from(bytes).into(),
        }
    }

    fn test_heterogeneous_f16_tensor(name: &str, dims: &[usize]) -> Tensor {
        let count: usize = dims.iter().product();
        let mut bytes = Vec::with_capacity(count * 2);
        for i in 0..count {
            let half: u16 = if i % 2 == 0 { 0x3000 } else { 0xb000 };
            bytes.extend_from_slice(&half.to_le_bytes());
        }
        Tensor {
            name: name.to_owned(),
            dimensions: dims.to_vec(),
            kind: 1,
            data: std::sync::Arc::<[u8]>::from(bytes).into(),
        }
    }

    fn test_heterogeneous_base_model(geometry: &GraphGeometry) -> Model {
        let mut base_tensors = BTreeMap::new();
        for layer in 0..4_usize {
            base_tensors.insert(
                format!("blk.{layer}.ffn_down.weight"),
                test_heterogeneous_f32_tensor(
                    &format!("blk.{layer}.ffn_down.weight"),
                    &[geometry.feed_forward, geometry.embedding],
                    0.125,
                ),
            );
            if layer < 3 {
                base_tensors.insert(
                    format!("blk.{layer}.attn_qkv.weight"),
                    test_heterogeneous_f32_tensor(
                        &format!("blk.{layer}.attn_qkv.weight"),
                        &[geometry.embedding, geometry.convolution_width()],
                        0.125,
                    ),
                );
            } else {
                for (suffix, in_f, out_f) in [
                    ("attn_q.weight", geometry.embedding, 2 * geometry.q_width()),
                    ("attn_k.weight", geometry.embedding, geometry.kv_width()),
                    ("attn_v.weight", geometry.embedding, geometry.kv_width()),
                    ("attn_output.weight", geometry.q_width(), geometry.embedding),
                ] {
                    let name = format!("blk.{layer}.{suffix}");
                    base_tensors.insert(
                        name.clone(),
                        test_heterogeneous_f32_tensor(&name, &[in_f, out_f], 0.125),
                    );
                }
            }
        }
        Model {
            metadata: BTreeMap::from([(
                "general.architecture".to_owned(),
                bonsai_gguf::MetadataValue::from("qwen35"),
            )]),
            tensors: base_tensors,
            identity: "vk-heterogeneous-mr-base".to_owned(),
        }
    }

    fn test_heterogeneous_mr_adapter(blk3_rank: usize) -> Model {
        use bonsai_gguf::MetadataValue;

        let mut tensors = BTreeMap::new();
        let specs = [
            ("blk.0.ffn_down.weight", 32, 16, 2),
            // blk.1.ffn_down.weight intentionally omitted!
            ("blk.2.attn_qkv.weight", 16, 32, 3),
            ("blk.3.ffn_down.weight", 32, 16, blk3_rank),
            ("blk.3.attn_q.weight", 16, 64, 4),
            ("blk.3.attn_k.weight", 16, 16, 4),
            ("blk.3.attn_v.weight", 16, 16, 4),
            ("blk.3.attn_output.weight", 32, 16, 4),
        ];
        for (target, in_f, out_f, rank) in specs {
            tensors.insert(
                format!("{target}.lora_a"),
                test_heterogeneous_f16_tensor(&format!("{target}.lora_a"), &[in_f, rank]),
            );
            tensors.insert(
                format!("{target}.lora_b"),
                test_heterogeneous_f16_tensor(&format!("{target}.lora_b"), &[rank, out_f]),
            );
        }
        Model {
            metadata: BTreeMap::from([
                (
                    "general.architecture".to_owned(),
                    MetadataValue::from("qwen35"),
                ),
                ("general.type".to_owned(), MetadataValue::from("adapter")),
                ("adapter.type".to_owned(), MetadataValue::from("lora")),
                (
                    "adapter.lora.alpha".to_owned(),
                    MetadataValue::from(6.0_f32),
                ),
                (
                    "adapter.lora_alpha".to_owned(),
                    MetadataValue::from(6.0_f32),
                ),
            ]),
            tensors,
            identity: "vk-mr-heterogeneous".to_owned(),
        }
    }

    #[test]
    fn sparse_heterogeneous_multi_rank_adapter_prepares_per_site_ranks_and_scratch_layout()
    -> Result<(), Error> {
        let geometry = GraphGeometry {
            embedding: 16,
            feed_forward: 32,
            vocabulary: 8,
            heads: 2,
            kv_heads: 1,
            head: 16,
            rope: 8,
            rope_sections: [4, 0, 0, 0],
            rope_base: 10_000.0,
            epsilon: 1e-6,
            ssm_head: 8,
            ssm_keys: 1,
            ssm_values: 2,
            convolution: 2,
            context_limit: 32,
        };
        let base_model = test_heterogeneous_base_model(&geometry);
        let mr_model = test_heterogeneous_mr_adapter(6);
        let recurrent_flags = [true, true, true, false];
        let prepare = |mr: &Model| {
            prepare_adapters_upload(
                &base_model,
                &geometry,
                &recurrent_flags,
                true,
                VulkanBatchedAdapters {
                    multi_rank: Some(VulkanMultiRankAdapterSpec {
                        adapter: mr,
                        policy: MultiRankPolicy::NpuFriendly,
                        scale: 0.5,
                    }),
                },
            )
        };
        let prepared = prepare(&mr_model)?;
        assert_eq!(prepared.max_rank, 6);
        assert_eq!(prepared.multi_rank_linear.len(), 7);
        assert!(
            !prepared
                .multi_rank_linear
                .contains_key("blk.1.ffn_down.weight")
        );

        for (name, exp_rank, exp_scaling, exp_a, exp_b) in [
            ("blk.0.ffn_down.weight", 2_usize, 3.0_f32, 32 * 2, 2 * 16),
            ("blk.2.attn_qkv.weight", 3, 2.0, 16 * 3, 3 * 32),
            ("blk.3.ffn_down.weight", 6, 1.0, 32 * 6, 6 * 16),
            ("blk.3.attn_q.weight", 4, 1.5, 16 * 4, 4 * 64),
        ] {
            let factor = &prepared.multi_rank_linear[name];
            assert_eq!(factor.rank, exp_rank);
            assert_eq!(factor.scaling.to_bits(), exp_scaling.to_bits());
            assert_eq!(factor.a_words.len(), exp_a);
            assert_eq!(factor.b_words.len(), exp_b);
        }

        let plan_small = plan_tile_scratch(
            &geometry,
            prepared.max_rank,
            16,
            64 * 1024 * 1024,
            u32::MAX,
            [65_535, 65_535, 65_535],
        )?;
        assert_eq!(plan_small.sizes[16], MIN_SCRATCH_BYTES);
        let plan_large = plan_tile_scratch(
            &geometry,
            prepared.max_rank,
            256,
            64 * 1024 * 1024,
            u32::MAX,
            [65_535, 65_535, 65_535],
        )?;
        assert_eq!(plan_large.allocated_tile_tokens, 256);
        assert_eq!(plan_large.sizes[16], 256 * 6 * 4);

        let mr_alt = test_heterogeneous_mr_adapter(4);
        let prepared_alt = prepare(&mr_alt)?;
        assert_eq!(prepared_alt.max_rank, 4);
        assert_ne!(prepared.content_sha256, prepared_alt.content_sha256);
        let plan_alt = plan_tile_scratch(
            &geometry,
            prepared_alt.max_rank,
            256,
            64 * 1024 * 1024,
            u32::MAX,
            [65_535, 65_535, 65_535],
        )?;
        assert_eq!(plan_alt.sizes[16], (256 * 4 * 4).max(MIN_SCRATCH_BYTES));
        Ok(())
    }

    #[test]
    fn default_prefill_tile32_and_double_buffered_staging_plan() -> Result<(), Error> {
        let cfg = VulkanBatchedConfig::default();
        assert_eq!(cfg.prefill_tile_tokens, 32);

        let geometry = GraphGeometry {
            embedding: 512,
            feed_forward: 1024,
            vocabulary: 4096,
            heads: 16,
            kv_heads: 4,
            head: 64,
            rope: 64,
            rope_sections: [8, 8, 8, 0],
            rope_base: 10_000.0,
            epsilon: 1e-6,
            ssm_head: 64,
            ssm_keys: 4,
            ssm_values: 8,
            convolution: 4,
            context_limit: 2048,
        };
        let plan = plan_tile_scratch(
            &geometry,
            8,
            cfg.prefill_tile_tokens,
            64 * 1024 * 1024,
            u32::MAX,
            [65_535, 65_535, 65_535],
        )?;
        assert_eq!(plan.allocated_tile_tokens, 32);
        let expected_hidden = (32_u64 * 512 * 4).max(MIN_SCRATCH_BYTES);
        let expected_rope = (32_u64 * 64 * 4).max(MIN_SCRATCH_BYTES);
        let expected_lora = (32_u64 * 8 * 4).max(MIN_SCRATCH_BYTES);
        assert_eq!(plan.sizes[0], expected_hidden);
        assert_eq!(plan.sizes[21], expected_hidden);
        assert_eq!(plan.sizes[22], expected_hidden);
        assert_eq!(plan.sizes[23], expected_rope);
        assert_eq!(plan.sizes[24], expected_rope);
        assert_eq!(plan.sizes[25], expected_lora);
        assert_eq!(plan.sizes[26], expected_lora);
        Ok(())
    }

    #[test]
    fn fused_attention_dispatch_uses_gqa6_for_prefill_and_falls_back_cleanly() {
        let base_desc = AttentionDescriptor {
            query_len: 32,
            total_seq: 128,
            past_len: 96,
            num_heads: 24,
            num_kv_heads: 4,
            head_dim: 256,
            scale: 0.0625,
            causal: true,
        };
        let shared_32k = 32_768_u32;

        // Prefill with t_len > 1, 6:1 GQA, Fp16 KV, and 32 KiB shared memory routes to GQA6.
        let gqa6 = select_fused_attention_dispatch(&base_desc, VulkanKvPrecision::Fp16, shared_32k);
        assert_eq!(
            gqa6,
            FusedAttentionDispatchChoice {
                kind: FusedAttentionKernelKind::GqaGrouped6PrefillF16Kv,
                grid: [1, 4, 32],
            }
        );

        // Single-token decode keeps existing per-head fused shader even for 6:1 GQA + Fp16 KV.
        let decode_desc = AttentionDescriptor {
            query_len: 1,
            total_seq: 97,
            ..base_desc
        };
        assert_eq!(
            select_fused_attention_dispatch(&decode_desc, VulkanKvPrecision::Fp16, shared_32k),
            FusedAttentionDispatchChoice {
                kind: FusedAttentionKernelKind::PerHead,
                grid: [1, 24, 1],
            }
        );

        // Non-6:1 GQA ratio (4:1 and non-divisible) falls back cleanly to per-head fused shader.
        let gqa4_desc = AttentionDescriptor {
            num_heads: 16,
            num_kv_heads: 4,
            ..base_desc
        };
        assert_eq!(
            select_fused_attention_dispatch(&gqa4_desc, VulkanKvPrecision::Fp16, shared_32k),
            FusedAttentionDispatchChoice {
                kind: FusedAttentionKernelKind::PerHead,
                grid: [1, 16, 32],
            }
        );
        let odd_desc = AttentionDescriptor {
            num_heads: 13,
            num_kv_heads: 2,
            ..base_desc
        };
        assert_eq!(
            select_fused_attention_dispatch(&odd_desc, VulkanKvPrecision::Fp16, shared_32k),
            FusedAttentionDispatchChoice {
                kind: FusedAttentionKernelKind::PerHead,
                grid: [1, 13, 32],
            }
        );

        // Fp32 KV precision and tight shared memory fall back cleanly to per-head fused shader.
        assert_eq!(
            select_fused_attention_dispatch(&base_desc, VulkanKvPrecision::Fp32, shared_32k),
            FusedAttentionDispatchChoice {
                kind: FusedAttentionKernelKind::PerHead,
                grid: [1, 24, 32],
            }
        );
        assert_eq!(
            select_fused_attention_dispatch(
                &base_desc,
                VulkanKvPrecision::Fp16,
                spirv::ATTN_FUSED_GQA6_SHARED_MEMORY_BYTES * 2 - 4,
            ),
            FusedAttentionDispatchChoice {
                kind: FusedAttentionKernelKind::PerHead,
                grid: [1, 24, 32],
            }
        );
    }

    #[test]
    fn tdr_policy_raises_default_decode_layers_per_submit_to_64() -> Result<(), Error> {
        let default_policy = TdrChunkPolicy::default();
        assert_eq!(default_policy.effective_layers_per_submit(1), 64);
        assert_eq!(default_policy.effective_layers_per_submit(32), 8);
        assert_eq!(
            default_policy.layer_chunk_ranges_for_tokens(36, 1)?,
            vec![0..36]
        );
        assert_eq!(
            default_policy.layer_chunk_ranges_for_tokens(36, 32)?,
            vec![0..8, 8..16, 16..24, 24..32, 32..36]
        );

        let explicit_one = TdrChunkPolicy::new(1, 1024)?;
        assert_eq!(explicit_one.effective_layers_per_submit(1), 1);
        assert_eq!(
            explicit_one.layer_chunk_ranges_for_tokens(3, 1)?,
            vec![0..1, 1..2, 2..3]
        );
        Ok(())
    }
}
