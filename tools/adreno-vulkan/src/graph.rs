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
};

use bonsai_native_hash::Hasher;
use bonsai_prism_runtime::{
    kernels::{row_into, validate_tensor},
    model::{
        Model, Tensor,
        swift::{MODEL_SHA256, REPACKED_PTQ1_SHA256},
    },
};

use crate::{
    AdrenoAttentionPath, AttentionDescriptor, AttentionStrategy, Error, ProjectionKernelPath,
    attention_cache_key,
    device::{
        AdrenoVulkanDevice, BufferId, BufferSpan, CommandBatchRecorder,
        MAX_DESCRIPTOR_SETS_PER_SUBMIT, PipelineId, WeightArena,
    },
    executor::{
        KIND_PQ2, KIND_PTQ1, MIN_SCRATCH_BYTES, ProjectionPipelines, Q8_0_WORDS_PER_BLOCK,
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
        self.validate()?;
        let mut ranges = Vec::with_capacity(total_layers.div_ceil(self.layers_per_submit));
        let mut start = 0_usize;
        while start < total_layers {
            let end = start
                .saturating_add(self.layers_per_submit)
                .min(total_layers);
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
    /// Packed IEEE binary16 words (`2 × f16` per `u32`), unpacked to `F32` before `#40` attention.
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
        }
    }
}

/// Compiled SPIR-V 1.6 compute pipelines used by [`VulkanBatchedGraphExecutor`].
///
/// Bundles `#39` projection pipelines, `#40` attention pipelines, and whole-layer intra-GPU
/// transformation/state pipelines alongside a deterministic digest over all emitted SPIR-V words.
#[derive(Debug)]
pub struct VulkanLayerPipelines {
    proj: ProjectionPipelines,
    attn_qk_scores: PipelineId,
    attn_softmax_f16: PipelineId,
    attn_pv_accum: PipelineId,
    attn_fused_f16prob: PipelineId,
    rms_norm: PipelineId,
    add_rms_norm: PipelineId,
    residual_add: PipelineId,
    rope: PipelineId,
    mul_sigmoid: PipelineId,
    conv1d_silu: PipelineId,
    l2_norm: PipelineId,
    gdn_step: PipelineId,
    gdn_gates: PipelineId,
    deinterleave_q_gate: PipelineId,
    round_f16: PipelineId,
    kv_append_f32: PipelineId,
    kv_append_f16: PipelineId,
    kv_unpack_f16: PipelineId,
    quant_q8_0_source: PipelineId,
    quant_q8_0_pinned: PipelineId,
    quant_q8_0_pinned_native: PipelineId,
    f32_matvec: PipelineId,
    mul_f32_elemwise: PipelineId,
    permute_ssm_heads: PipelineId,
    copy_slice: PipelineId,
    use_sdot: bool,
    attn_qk_digest: String,
    attn_softmax_digest: String,
    attn_pv_digest: String,
    attn_fused_digest: String,
    shader_digests_sha256: String,
}

impl VulkanLayerPipelines {
    /// Compiles every layer, projection, and attention SPIR-V 1.6 pipeline on `device`.
    ///
    /// # Errors
    /// Returns [`Error::Unsupported`], [`Error::Vulkan`], or [`Error::InvalidInput`] if pipeline
    /// creation or digest calculation fails.
    #[expect(
        clippy::too_many_lines,
        reason = "Compiles and digests all 25 whole-layer, projection, and attention SPIR-V pipelines sequentially."
    )]
    pub fn compile(device: &mut AdrenoVulkanDevice, use_sdot: bool) -> Result<Self, Error> {
        let proj = ProjectionPipelines::compile(device, use_sdot)?;
        let mut hasher = Hasher::new().map_err(|err| Error::InvalidInput(err.to_string()))?;
        hash_bytes(&mut hasher, b"vulkan-layer-pipelines-v1:")?;
        hash_u32(&mut hasher, u32::from(use_sdot))?;

        let mut make_pipe = |tag: &[u8],
                             words: Vec<u32>,
                             num_buffers: usize,
                             push_u32s: usize|
         -> Result<(PipelineId, String), Error> {
            let digest = spirv::shader_digest(&words)?;
            hash_bytes(&mut hasher, tag)?;
            hash_bytes(&mut hasher, digest.as_bytes())?;
            let pipe = device.create_compute_pipeline(&words, num_buffers, push_u32s)?;
            Ok((pipe, digest))
        };

        let (attn_qk_scores, attn_qk_digest) =
            make_pipe(b"attn_qk_scores", spirv::attn_qk_scores(), 3, 8)?;
        let (attn_softmax_f16, attn_softmax_digest) =
            make_pipe(b"attn_softmax_f16", spirv::attn_softmax_f16(), 2, 8)?;
        let (attn_pv_accum, attn_pv_digest) =
            make_pipe(b"attn_pv_accum", spirv::attn_pv_accum(), 3, 8)?;
        let (attn_fused_f16prob, attn_fused_digest) =
            make_pipe(b"attn_fused_f16prob", spirv::attn_fused_f16prob(), 4, 8)?;

        let (rms_norm, _) = make_pipe(b"rms_norm_f32", spirv::rms_norm_f32(), 3, 2)?;
        let (add_rms_norm, _) = make_pipe(b"add_rms_norm_f32", spirv::add_rms_norm_f32(), 4, 2)?;
        let (residual_add, _) = make_pipe(b"residual_add_f32", spirv::residual_add_f32(), 2, 1)?;
        let (rope, _) = make_pipe(b"rope_f32", spirv::rope_f32(), 2, 4)?;
        let (mul_sigmoid, _) = make_pipe(b"mul_sigmoid_f32", spirv::mul_sigmoid_f32(), 2, 1)?;
        let (conv1d_silu, _) = make_pipe(b"conv1d_silu_f32", spirv::conv1d_silu_f32(), 4, 2)?;
        let (l2_norm, _) = make_pipe(b"l2_norm_f32", spirv::l2_norm_f32(), 1, 3)?;
        let (gdn_step, _) = make_pipe(b"gdn_step_f32", spirv::gdn_step_f32(), 7, 4)?;
        let (gdn_gates, _) = make_pipe(b"gdn_gates_f32", spirv::gdn_gates_f32(), 4, 1)?;
        let (deinterleave_q_gate, _) = make_pipe(
            b"deinterleave_q_gate_f32",
            spirv::deinterleave_q_gate_f32(),
            3,
            2,
        )?;
        let (round_f16, _) = make_pipe(b"round_f16_f32", spirv::round_f16_f32(), 1, 1)?;
        let (kv_append_f32, _) = make_pipe(b"kv_append_f32", spirv::kv_append_f32(), 2, 2)?;
        let (kv_append_f16, _) = make_pipe(b"kv_append_f16", spirv::kv_append_f16(), 2, 2)?;
        let (kv_unpack_f16, _) =
            make_pipe(b"kv_unpack_f16_to_f32", spirv::kv_unpack_f16_to_f32(), 2, 1)?;
        let (quant_q8_0_source, _) = make_pipe(
            b"quant_q8_0_source",
            spirv::quantize_f32_to_q8_0(false, false),
            2,
            1,
        )?;
        let (quant_q8_0_pinned, _) = make_pipe(
            b"quant_q8_0_pinned",
            spirv::quantize_f32_to_q8_0(true, false),
            2,
            1,
        )?;
        let (quant_q8_0_pinned_native, _) = make_pipe(
            b"quant_q8_0_pinned_native",
            spirv::quantize_f32_to_q8_0(true, true),
            2,
            1,
        )?;
        let (f32_matvec, _) = make_pipe(b"f32_matvec", spirv::f32_matvec(), 3, 2)?;
        let (mul_f32_elemwise, _) =
            make_pipe(b"mul_f32_elemwise", spirv::mul_f32_elemwise(), 3, 1)?;
        let (permute_ssm_heads, _) = make_pipe(
            b"permute_ssm_heads_f32",
            spirv::permute_ssm_heads_f32(),
            2,
            3,
        )?;
        let (copy_slice, _) = make_pipe(b"copy_slice_f32", spirv::copy_slice_f32(), 2, 2)?;

        let digest_bytes = hasher
            .finish()
            .map_err(|err| Error::InvalidInput(err.to_string()))?;
        let shader_digests_sha256 = hex::encode(digest_bytes);

        Ok(Self {
            proj,
            attn_qk_scores,
            attn_softmax_f16,
            attn_pv_accum,
            attn_fused_f16prob,
            rms_norm,
            add_rms_norm,
            residual_add,
            rope,
            mul_sigmoid,
            conv1d_silu,
            l2_norm,
            gdn_step,
            gdn_gates,
            deinterleave_q_gate,
            round_f16,
            kv_append_f32,
            kv_append_f16,
            kv_unpack_f16,
            quant_q8_0_source,
            quant_q8_0_pinned,
            quant_q8_0_pinned_native,
            f32_matvec,
            mul_f32_elemwise,
            permute_ssm_heads,
            copy_slice,
            use_sdot,
            attn_qk_digest,
            attn_softmax_digest,
            attn_pv_digest,
            attn_fused_digest,
            shader_digests_sha256,
        })
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

/// Checkpoint snapshot of GPU-resident recurrent GDN/conv state and attention KV cache.
#[derive(Clone, Debug, PartialEq)]
pub struct VulkanBatchedCheckpoint {
    executor_identity_sha256: String,
    position: usize,
    mrope_positions: [usize; 4],
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
    F32,
}

#[derive(Clone, Copy, Debug)]
struct GpuMatrixWeight {
    span: BufferSpan,
    encoding: WeightEncoding,
    cols: usize,
    rows: usize,
    rotated: bool,
    grouped_ssm: bool,
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
    k_unpacked: BufferId,
    v_unpacked: BufferId,
    attn_scores: BufferId,
    attn_probs: BufferId,
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
    host_rope_freqs: Vec<f32>,
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
        let device = AdrenoVulkanDevice::open()?;
        Self::from_device(device, model, config)
    }

    /// Uploads `model` onto an already-opened [`AdrenoVulkanDevice`] with `config`.
    ///
    /// # Errors
    /// Returns [`Error::Unsupported`], [`Error::InvalidInput`], [`Error::ResourceLimit`],
    /// [`Error::DeviceFault`], or [`Error::Vulkan`].
    #[expect(
        clippy::too_many_lines,
        reason = "Coordinates single-pass model metadata parsing, GPU weight upload, state/scratch allocation, and deterministic identity hashing."
    )]
    pub fn from_device(
        mut device: AdrenoVulkanDevice,
        model: &Model,
        config: VulkanBatchedConfig,
    ) -> Result<Self, Error> {
        config.tdr_policy.validate()?;
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
            MODEL_SHA256 | REPACKED_PTQ1_SHA256
        ));

        let embedding = lookup_tensor(model, "token_embd.weight", &[geometry.embedding, 0])?;
        if embedding.dimensions.get(1).copied() != Some(geometry.vocabulary) {
            return Err(Error::InvalidInput(
                "token_embd.weight vocabulary extent mismatch".to_owned(),
            ));
        }
        validate_model_tensors_subset(model, &geometry, layer_count, &recurrent_flags)?;

        let pipelines = VulkanLayerPipelines::compile(&mut device, use_sdot)?;
        let min_align =
            usize::try_from(device.min_storage_buffer_offset_alignment().max(1)).unwrap_or(256);
        let max_storage_range = device
            .capabilities()
            .compute_limits
            .max_storage_buffer_range;

        let mut guard = BufferCleanupGuard::new(&mut device);
        let transforms = parse_and_upload_transforms(model, &geometry, &mut guard)?;

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
        let output_tensor = if model.tensors.contains_key("output.weight") {
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
            &mut repack_scratch,
            &mut row_scratch,
            &mut guard,
        )?;

        let scratch = allocate_scratch(
            &geometry,
            config.scratch_budget_bytes,
            max_storage_range,
            &mut guard,
        )?;

        let (arena, owned_buffers) = guard.defuse();
        let identity_sha256 = compute_executor_identity(
            &device,
            model,
            &config,
            resolved_context_limit,
            resolved_pinned_arithmetic,
            kernel_path,
            layer_count,
            &pipelines,
        )?;

        let _ = min_align;
        let host_rope_freqs = vec![0.0_f32; geometry.rope];
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
            host_rope_freqs,
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
        self.step_token_impl(token, Some(logits))
    }

    /// Advances model recurrent/KV state on GPU by ingesting one prompt token without reading logits.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`], [`Error::ResourceLimit`], [`Error::DeviceFault`],
    /// [`Error::Unsupported`], or [`Error::Vulkan`].
    pub fn ingest_token(&mut self, token: u32) -> Result<(), Error> {
        self.step_token_impl(token, None)
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
        self.state_consistent = true;
        Ok(())
    }

    /// Captures a host-side snapshot of all mutable GPU recurrent and KV-cache state.
    ///
    /// # Errors
    /// Returns [`Error::DeviceFault`], [`Error::InvalidInput`], or [`Error::Vulkan`].
    pub fn checkpoint(&self) -> Result<VulkanBatchedCheckpoint, Error> {
        if self.device.is_faulted() || !self.state_consistent {
            return Err(Error::DeviceFault(
                "cannot checkpoint faulted or inconsistent Vulkan graph executor".to_owned(),
            ));
        }
        let kv_width = self.geometry.kv_width();
        let mut layer_states = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let snap = match &layer.attention {
                GpuLayerAttention::Recurrent {
                    conv_state,
                    conv_state_elems,
                    matrix_state,
                    matrix_state_elems,
                    ..
                } => {
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
                    LayerStateCheckpoint::Recurrent {
                        convolution,
                        matrix,
                    }
                }
                GpuLayerAttention::Full {
                    k_cache, v_cache, ..
                } => {
                    let words_per_tok = match self.config.kv_precision {
                        VulkanKvPrecision::Fp16 => kv_width / 2,
                        VulkanKvPrecision::Fp32 => kv_width,
                    };
                    let total_words = self.position.saturating_mul(words_per_tok);
                    let total_bytes =
                        usize::try_from(u32_slice_bytes(total_words)?).map_err(|_| {
                            Error::ResourceLimit("KV checkpoint byte length overflow".to_owned())
                        })?;
                    let mut keys_words = vec![0_u32; total_words];
                    let mut values_words = vec![0_u32; total_words];
                    if total_bytes > 0 {
                        let mut raw_k = vec![0_u8; total_bytes];
                        let mut raw_v = vec![0_u8; total_bytes];
                        self.device.read_buffer_bytes_at(*k_cache, 0, &mut raw_k)?;
                        self.device.read_buffer_bytes_at(*v_cache, 0, &mut raw_v)?;
                        for (dst, chunk) in keys_words.iter_mut().zip(raw_k.as_chunks::<4>().0) {
                            *dst = u32::from_le_bytes(*chunk);
                        }
                        for (dst, chunk) in values_words.iter_mut().zip(raw_v.as_chunks::<4>().0) {
                            *dst = u32::from_le_bytes(*chunk);
                        }
                    }
                    LayerStateCheckpoint::Full {
                        precision: self.config.kv_precision,
                        width: kv_width,
                        keys_words,
                        values_words,
                    }
                }
            };
            layer_states.push(snap);
        }
        Ok(VulkanBatchedCheckpoint {
            executor_identity_sha256: self.identity_sha256.clone(),
            position: self.position,
            mrope_positions: self.mrope_positions,
            layers: layer_states,
        })
    }

    /// Restores GPU recurrent and KV-cache state from a checkpoint produced by this executor.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`], [`Error::DeviceFault`], or [`Error::Vulkan`] if the
    /// checkpoint belongs to a different model/executor configuration or fails validation.
    #[expect(
        clippy::too_many_lines,
        reason = "Validates checkpoint provenance/shapes before overwriting recurrent or KV GPU state."
    )]
    pub fn restore(&mut self, checkpoint: &VulkanBatchedCheckpoint) -> Result<(), Error> {
        if self.device.is_faulted() {
            return Err(Error::DeviceFault(
                "Vulkan batched executor faulted and cannot restore".to_owned(),
            ));
        }
        if checkpoint.executor_identity_sha256 != self.identity_sha256 {
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
                    let k_bytes = self.device.buffer_size_bytes(*k_cache)?;
                    let v_bytes = self.device.buffer_size_bytes(*v_cache)?;
                    self.device.fill_buffer_zero(*k_cache, 0, k_bytes)?;
                    self.device.fill_buffer_zero(*v_cache, 0, v_bytes)?;
                    if !keys_words.is_empty() {
                        self.device.write_buffer_u32_at(*k_cache, 0, keys_words)?;
                        self.device.write_buffer_u32_at(*v_cache, 0, values_words)?;
                    }
                }
                _ => {
                    return Err(Error::InvalidInput(
                        "checkpoint layer kind mismatch".to_owned(),
                    ));
                }
            }
        }
        self.position = checkpoint.position;
        self.mrope_positions = checkpoint.mrope_positions;
        self.state_consistent = true;
        Ok(())
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

    /// Returns the deterministic SHA-256 identity binding model, TDR policy, shaders, and GPU.
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
        let token_idx = usize::try_from(token)
            .map_err(|_| Error::InvalidInput("token index overflow".to_owned()))?;
        if token_idx >= self.geometry.vocabulary {
            return Err(Error::InvalidInput(format!(
                "token {token} out of vocabulary bounds {}",
                self.geometry.vocabulary
            )));
        }
        if self.position >= self.resolved_context_limit {
            return Err(Error::InvalidInput(format!(
                "context limit {} exhausted at position {}",
                self.resolved_context_limit, self.position
            )));
        }

        row_into(&self.embedding, token_idx, &mut self.host_embed_scratch)
            .map_err(|err| Error::InvalidInput(err.to_string()))?;
        ensure_finite_f32(
            &self.host_embed_scratch,
            "token embedding contains non-finite value",
        )?;
        self.device
            .write_buffer_f32(self.scratch.hidden, &self.host_embed_scratch)?;

        let has_full_layers = self
            .layers
            .iter()
            .any(|layer| matches!(layer.attention, GpuLayerAttention::Full { .. }));
        if has_full_layers {
            fill_rope_frequencies(
                &self.geometry,
                self.mrope_positions,
                &mut self.host_rope_freqs,
            )?;
            self.device
                .write_buffer_f32(self.scratch.rope_freqs, &self.host_rope_freqs)?;
        }

        let chunks = self
            .config
            .tdr_policy
            .layer_chunk_ranges(self.layers.len())?;
        let need_logits = logits_out.is_some();
        let total_layers = self.layers.len();
        let min_align = usize::try_from(self.device.min_storage_buffer_offset_alignment().max(1))
            .unwrap_or(256);

        let run_result = (|| -> Result<(), Error> {
            for (chunk_idx, range) in chunks.iter().enumerate() {
                self.batch_recorder.clear();
                if chunk_idx == 0 && self.transforms.inverse_embedding {
                    self.record_inverse_embedding_transform()?;
                }
                for layer_idx in range.clone() {
                    let narrow_final = !need_logits && layer_idx + 1 == total_layers;
                    self.record_layer(layer_idx, narrow_final, min_align)?;
                }
                if chunk_idx + 1 == chunks.len() && need_logits {
                    self.record_final_logits()?;
                }
                if !self.batch_recorder.is_empty() {
                    let submits = self.device.submit_compute_batches_with_max_stages(
                        &self.batch_recorder,
                        self.config.tdr_policy.max_stages_per_submit,
                    )?;
                    self.submits_completed = self.submits_completed.saturating_add(submits);
                }
            }

            if let Some(dst) = logits_out {
                dst.resize(self.geometry.vocabulary, 0.0_f32);
                self.device.read_buffer_f32(self.scratch.logits, dst)?;
                ensure_finite_f32(dst, "GPU batched graph produced non-finite logits")?;
            }
            Ok(())
        })();

        if run_result.is_err() {
            if !chunks.is_empty() {
                self.state_consistent = false;
            }
            return run_result;
        }

        self.position += 1;
        for slot in &mut self.mrope_positions[..3] {
            *slot += 1;
        }
        Ok(())
    }

    fn record_inverse_embedding_transform(&mut self) -> Result<(), Error> {
        let len = self.geometry.embedding;
        if self.transforms.block >= 2 && len.is_multiple_of(self.transforms.block) {
            let block_u32 = u32::try_from(self.transforms.block)
                .map_err(|_| Error::InvalidInput("Hadamard block exceeds u32".to_owned()))?;
            let blk_cnt = u32::try_from(len / self.transforms.block)
                .map_err(|_| Error::ResourceLimit("Hadamard grid exceeds u32".to_owned()))?;
            let norm_bits = hadamard_norm_bits(block_u32);
            let grid = checked_grid(&self.device, [blk_cnt, 1, 1])?;
            self.batch_recorder.push_barrier(
                self.pipelines.proj.hadamard,
                &[
                    BufferSpan::whole(self.scratch.hidden),
                    BufferSpan::whole(self.scratch.transformed),
                ],
                &[block_u32, norm_bits],
                grid,
            );
            if let Some(&sign_buf) = self.transforms.sign_buffers.get(&len) {
                let len_u32 = u32::try_from(len)
                    .map_err(|_| Error::ResourceLimit("embedding length exceeds u32".to_owned()))?;
                let elem_grid = checked_grid(&self.device, [len_u32.div_ceil(WORKGROUP_X), 1, 1])?;
                self.batch_recorder.push_barrier(
                    self.pipelines.mul_f32_elemwise,
                    &[
                        BufferSpan::whole(self.scratch.transformed),
                        BufferSpan::whole(sign_buf),
                        BufferSpan::whole(self.scratch.hidden),
                    ],
                    &[len_u32],
                    elem_grid,
                );
            } else {
                let len_u32 = u32::try_from(len)
                    .map_err(|_| Error::ResourceLimit("embedding length exceeds u32".to_owned()))?;
                let copy_grid = checked_grid(&self.device, [len_u32.div_ceil(WORKGROUP_X), 1, 1])?;
                self.batch_recorder.push_barrier(
                    self.pipelines.copy_slice,
                    &[
                        BufferSpan::whole(self.scratch.transformed),
                        BufferSpan::whole(self.scratch.hidden),
                    ],
                    &[0, len_u32],
                    copy_grid,
                );
            }
        } else if let Some(&sign_buf) = self.transforms.sign_buffers.get(&len) {
            let len_u32 = u32::try_from(len)
                .map_err(|_| Error::ResourceLimit("embedding length exceeds u32".to_owned()))?;
            let elem_grid = checked_grid(&self.device, [len_u32.div_ceil(WORKGROUP_X), 1, 1])?;
            self.batch_recorder.push_barrier(
                self.pipelines.mul_f32_elemwise,
                &[
                    BufferSpan::whole(self.scratch.hidden),
                    BufferSpan::whole(sign_buf),
                    BufferSpan::whole(self.scratch.transformed),
                ],
                &[len_u32],
                elem_grid,
            );
            self.batch_recorder.push_barrier(
                self.pipelines.copy_slice,
                &[
                    BufferSpan::whole(self.scratch.transformed),
                    BufferSpan::whole(self.scratch.hidden),
                ],
                &[0, len_u32],
                elem_grid,
            );
        }
        Ok(())
    }

    fn record_layer(
        &mut self,
        layer_idx: usize,
        narrow_final: bool,
        min_align: usize,
    ) -> Result<(), Error> {
        let embed_u32 = u32::try_from(self.geometry.embedding)
            .map_err(|_| Error::ResourceLimit("embedding exceeds u32".to_owned()))?;
        let eps_bits = self.geometry.epsilon.to_bits();
        let attn_norm_w = self.layers[layer_idx].attn_norm_w;
        let post_norm_w = self.layers[layer_idx].post_norm_w;

        self.batch_recorder.push_barrier(
            self.pipelines.rms_norm,
            &[
                BufferSpan::whole(self.scratch.hidden),
                BufferSpan::whole(attn_norm_w),
                BufferSpan::whole(self.scratch.norm),
            ],
            &[embed_u32, eps_bits],
            checked_grid(&self.device, [1, 1, 1])?,
        );

        let is_recurrent = matches!(
            self.layers[layer_idx].attention,
            GpuLayerAttention::Recurrent { .. }
        );
        if is_recurrent {
            self.record_recurrent_attention(layer_idx, narrow_final, min_align)?;
        } else {
            self.record_full_attention(layer_idx, narrow_final)?;
        }

        if narrow_final {
            return Ok(());
        }

        self.batch_recorder.push_barrier(
            self.pipelines.add_rms_norm,
            &[
                BufferSpan::whole(self.scratch.hidden),
                BufferSpan::whole(self.scratch.delta),
                BufferSpan::whole(post_norm_w),
                BufferSpan::whole(self.scratch.norm),
            ],
            &[embed_u32, eps_bits],
            checked_grid(&self.device, [1, 1, 1])?,
        );

        self.record_ffn(layer_idx)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Encodes GPU-resident GDN recurrent layer stages preserving exact Qwen35 conv1d+SiLU, ClampedL2, decay/beta, delta-rule, and gated norm sequence."
    )]
    fn record_recurrent_attention(
        &mut self,
        layer_idx: usize,
        narrow_final: bool,
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
        let conv_out_buf = self.scratch.proj_e;

        if narrow_final {
            self.record_projection_group(
                self.scratch.norm,
                &[
                    (qkv, proj_qkv_buf),
                    (beta, proj_beta_buf),
                    (alpha, proj_alpha_buf),
                ],
            )?;
        } else {
            self.record_projection_group(
                self.scratch.norm,
                &[
                    (qkv, proj_qkv_buf),
                    (gate, proj_gate_buf),
                    (beta, proj_beta_buf),
                    (alpha, proj_alpha_buf),
                ],
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

        let active_conv_state = if conv_state_elems > 0 {
            conv_state
        } else {
            self.scratch.dummy_state
        };

        // conv1d+SiLU writes `conv_out` and updates `conv_state`; `gdn_gates` writes `alpha`/`beta`
        // concurrently because it touches disjoint projection outputs!
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

        // `ClampedL2` across [q (k_heads), k (k_heads)] in the prefix of `conv_out`.
        self.batch_recorder.push_barrier(
            self.pipelines.l2_norm,
            &[BufferSpan::whole(conv_out_buf)],
            &[qk_heads_u32, ssm_head_u32, self.geometry.epsilon.to_bits()],
            checked_grid(&self.device, [1, qk_heads_u32, 1])?,
        );

        // Extract `k` and `v` as aligned storage buffer spans (`subspan` if aligned, else GPU copy).
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

        let dim_f32 = u16::try_from(self.geometry.ssm_head).map_or(1.0_f32, f32::from);
        let gdn_scale_bits = (1.0_f32 / dim_f32.sqrt()).to_bits();
        let gdn_out_buf = proj_qkv_buf;
        self.batch_recorder.push_barrier(
            self.pipelines.gdn_step,
            &[
                BufferSpan::whole(matrix_state),
                q_span,
                k_span,
                v_span,
                BufferSpan::whole(proj_alpha_buf),
                BufferSpan::whole(proj_beta_buf),
                BufferSpan::whole(gdn_out_buf),
            ],
            &[v_heads_u32, k_heads_u32, ssm_head_u32, gdn_scale_bits],
            checked_grid(&self.device, [ssm_head_u32, v_heads_u32, 1])?,
        );

        if narrow_final {
            return Ok(());
        }

        // Per-head RMSNorm on `gdn_out` -> `conv_out_buf` (reused as normalized GDN output).
        let gdn_normed_buf = conv_out_buf;
        self.batch_recorder.push_barrier(
            self.pipelines.rms_norm,
            &[
                BufferSpan::whole(gdn_out_buf),
                BufferSpan::whole(norm_w),
                BufferSpan::whole(gdn_normed_buf),
            ],
            &[ssm_head_u32, self.geometry.epsilon.to_bits()],
            checked_grid(&self.device, [1, v_heads_u32, 1])?,
        );

        // Recurrent output gating uses `silu(gate) * gdn_normed` via `ffn_swiglu_f32`.
        let val_u32 = u32::try_from(val_width)
            .map_err(|_| Error::ResourceLimit("value_width exceeds u32".to_owned()))?;
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
            &[val_u32],
            checked_grid(&self.device, [val_u32.div_ceil(WORKGROUP_X), 1, 1])?,
        );

        self.record_projection_group(gated_buf, &[(output_w, self.scratch.delta)])
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Encodes GPU-resident full-attention layer preserving exact Q/Gate deinterleave, Q/K RMSNorm+RoPE(f16), V(f16), KV append/unpack, #40 attention, sigmoid gate, and output projection."
    )]
    fn record_full_attention(&mut self, layer_idx: usize, narrow_final: bool) -> Result<(), Error> {
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
        let pos_u32 = u32::try_from(self.position)
            .map_err(|_| Error::ResourceLimit("position exceeds u32".to_owned()))?;
        let eps_bits = self.geometry.epsilon.to_bits();
        let total_seq = self.position + 1;
        let total_seq_u32 = u32::try_from(total_seq)
            .map_err(|_| Error::ResourceLimit("total_seq exceeds u32".to_owned()))?;

        if narrow_final {
            self.record_projection_group(
                self.scratch.norm,
                &[(key_w, key_buf), (value_w, val_buf)],
            )?;
            self.batch_recorder.push_barrier(
                self.pipelines.rms_norm,
                &[
                    BufferSpan::whole(key_buf),
                    BufferSpan::whole(key_norm_w),
                    BufferSpan::whole(key_buf),
                ],
                &[head_dim_u32, eps_bits],
                checked_grid(&self.device, [1, kv_heads_u32, 1])?,
            );
            self.batch_recorder.push_concurrent(
                self.pipelines.round_f16,
                &[BufferSpan::whole(val_buf)],
                &[kv_width_u32],
                checked_grid(&self.device, [kv_width_u32.div_ceil(WORKGROUP_X), 1, 1])?,
            );
            self.batch_recorder.push_barrier(
                self.pipelines.rope,
                &[
                    BufferSpan::whole(key_buf),
                    BufferSpan::whole(self.scratch.rope_freqs),
                ],
                &[kv_heads_u32, head_dim_u32, rope_u32, 1],
                checked_grid(
                    &self.device,
                    [head_dim_u32.div_ceil(WORKGROUP_X), kv_heads_u32, 1],
                )?,
            );
            self.record_kv_append(key_buf, val_buf, k_cache, v_cache, kv_width_u32, pos_u32)?;
            return Ok(());
        }

        self.record_projection_group(
            self.scratch.norm,
            &[(query_w, q_gate_buf), (key_w, key_buf), (value_w, val_buf)],
        )?;

        let q_width_u32 = u32::try_from(self.geometry.q_width())
            .map_err(|_| Error::ResourceLimit("q_width exceeds u32".to_owned()))?;
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

        // Normalize Q, normalize K, and binary16-round V concurrently across disjoint buffers!
        self.batch_recorder.push_barrier(
            self.pipelines.rms_norm,
            &[
                BufferSpan::whole(query_buf),
                BufferSpan::whole(query_norm_w),
                BufferSpan::whole(query_buf),
            ],
            &[head_dim_u32, eps_bits],
            checked_grid(&self.device, [1, num_heads_u32, 1])?,
        );
        self.batch_recorder.push_concurrent(
            self.pipelines.rms_norm,
            &[
                BufferSpan::whole(key_buf),
                BufferSpan::whole(key_norm_w),
                BufferSpan::whole(key_buf),
            ],
            &[head_dim_u32, eps_bits],
            checked_grid(&self.device, [1, kv_heads_u32, 1])?,
        );
        self.batch_recorder.push_concurrent(
            self.pipelines.round_f16,
            &[BufferSpan::whole(val_buf)],
            &[kv_width_u32],
            checked_grid(&self.device, [kv_width_u32.div_ceil(WORKGROUP_X), 1, 1])?,
        );

        // Apply RoPE with F16 rounding concurrently to normalized Q and K!
        self.batch_recorder.push_barrier(
            self.pipelines.rope,
            &[
                BufferSpan::whole(query_buf),
                BufferSpan::whole(self.scratch.rope_freqs),
            ],
            &[num_heads_u32, head_dim_u32, rope_u32, 1],
            checked_grid(
                &self.device,
                [head_dim_u32.div_ceil(WORKGROUP_X), num_heads_u32, 1],
            )?,
        );
        self.batch_recorder.push_concurrent(
            self.pipelines.rope,
            &[
                BufferSpan::whole(key_buf),
                BufferSpan::whole(self.scratch.rope_freqs),
            ],
            &[kv_heads_u32, head_dim_u32, rope_u32, 1],
            checked_grid(
                &self.device,
                [head_dim_u32.div_ceil(WORKGROUP_X), kv_heads_u32, 1],
            )?,
        );

        self.record_kv_append(key_buf, val_buf, k_cache, v_cache, kv_width_u32, pos_u32)?;

        let (k_attn_span, v_attn_span) = match self.config.kv_precision {
            VulkanKvPrecision::Fp16 => {
                let unpack_words = total_seq.checked_mul(kv_width / 2).ok_or_else(|| {
                    Error::ResourceLimit("KV unpack word count overflow".to_owned())
                })?;
                let unpack_u32 = u32::try_from(unpack_words)
                    .map_err(|_| Error::ResourceLimit("KV unpack words exceed u32".to_owned()))?;
                let grid = checked_grid(&self.device, [unpack_u32.div_ceil(WORKGROUP_X), 1, 1])?;
                self.batch_recorder.push_barrier(
                    self.pipelines.kv_unpack_f16,
                    &[
                        BufferSpan::whole(k_cache),
                        BufferSpan::whole(self.scratch.k_unpacked),
                    ],
                    &[unpack_u32],
                    grid,
                );
                self.batch_recorder.push_concurrent(
                    self.pipelines.kv_unpack_f16,
                    &[
                        BufferSpan::whole(v_cache),
                        BufferSpan::whole(self.scratch.v_unpacked),
                    ],
                    &[unpack_u32],
                    grid,
                );
                (
                    BufferSpan::whole(self.scratch.k_unpacked),
                    BufferSpan::whole(self.scratch.v_unpacked),
                )
            }
            VulkanKvPrecision::Fp32 => (BufferSpan::whole(k_cache), BufferSpan::whole(v_cache)),
        };

        let head_f32 = u16::try_from(self.geometry.head).map_or(1.0_f32, f32::from);
        let desc = AttentionDescriptor {
            query_len: 1,
            total_seq: total_seq_u32,
            past_len: pos_u32,
            num_heads: num_heads_u32,
            num_kv_heads: kv_heads_u32,
            head_dim: head_dim_u32,
            scale: 1.0_f32 / head_f32.sqrt(),
            causal: true,
        };
        let plan = select_attention_plan(
            &desc,
            self.device.capabilities(),
            self.config.attention_strategy,
            self.config.scratch_budget_bytes,
        )?;
        let attn_push = attention_push_constants(&desc);
        let attended_buf = q_gate_buf;

        match plan.path {
            AdrenoAttentionPath::Fused => {
                self.batch_recorder.push_barrier(
                    self.pipelines.attn_fused_f16prob,
                    &[
                        BufferSpan::whole(query_buf),
                        k_attn_span,
                        v_attn_span,
                        BufferSpan::whole(attended_buf),
                    ],
                    &attn_push,
                    checked_grid(&self.device, [1, num_heads_u32, 1])?,
                );
            }
            AdrenoAttentionPath::Decomposed3Stage => {
                self.batch_recorder.push_barrier(
                    self.pipelines.attn_qk_scores,
                    &[
                        BufferSpan::whole(query_buf),
                        k_attn_span,
                        BufferSpan::whole(self.scratch.attn_scores),
                    ],
                    &attn_push,
                    checked_grid(
                        &self.device,
                        [total_seq_u32.div_ceil(WORKGROUP_X), num_heads_u32, 1],
                    )?,
                );
                self.batch_recorder.push_barrier(
                    self.pipelines.attn_softmax_f16,
                    &[
                        BufferSpan::whole(self.scratch.attn_scores),
                        BufferSpan::whole(self.scratch.attn_probs),
                    ],
                    &attn_push,
                    checked_grid(&self.device, [1, num_heads_u32, 1])?,
                );
                self.batch_recorder.push_barrier(
                    self.pipelines.attn_pv_accum,
                    &[
                        BufferSpan::whole(self.scratch.attn_probs),
                        v_attn_span,
                        BufferSpan::whole(attended_buf),
                    ],
                    &attn_push,
                    checked_grid(
                        &self.device,
                        [head_dim_u32.div_ceil(WORKGROUP_X), num_heads_u32, 1],
                    )?,
                );
            }
        }

        // Apply `attended *= sigmoid(gate)` in place, then project `attn_output`.
        self.batch_recorder.push_barrier(
            self.pipelines.mul_sigmoid,
            &[BufferSpan::whole(attended_buf), BufferSpan::whole(gate_buf)],
            &[q_width_u32],
            checked_grid(&self.device, [q_width_u32.div_ceil(WORKGROUP_X), 1, 1])?,
        );

        self.record_projection_group(attended_buf, &[(output_w, self.scratch.delta)])
    }

    fn record_kv_append(
        &mut self,
        key_buf: BufferId,
        val_buf: BufferId,
        k_cache: BufferId,
        v_cache: BufferId,
        kv_width_u32: u32,
        pos_u32: u32,
    ) -> Result<(), Error> {
        match self.config.kv_precision {
            VulkanKvPrecision::Fp16 => {
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
            VulkanKvPrecision::Fp32 => {
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
        }
        Ok(())
    }

    fn record_ffn(&mut self, layer_idx: usize) -> Result<(), Error> {
        let ffn_gate = self.layers[layer_idx].ffn_gate;
        let ffn_up = self.layers[layer_idx].ffn_up;
        let ffn_down = self.layers[layer_idx].ffn_down;

        let gate_out_buf = self.scratch.proj_a;
        let up_out_buf = self.scratch.proj_b;
        let mid_buf = self.scratch.proj_c;

        self.record_projection_group(
            self.scratch.norm,
            &[(ffn_gate, gate_out_buf), (ffn_up, up_out_buf)],
        )?;

        let inter_u32 = u32::try_from(self.geometry.feed_forward)
            .map_err(|_| Error::ResourceLimit("feed_forward exceeds u32".to_owned()))?;
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
            &[inter_u32],
            checked_grid(&self.device, [inter_u32.div_ceil(WORKGROUP_X), 1, 1])?,
        );

        self.record_projection_group(mid_buf, &[(ffn_down, self.scratch.delta)])?;

        let embed_u32 = u32::try_from(self.geometry.embedding)
            .map_err(|_| Error::ResourceLimit("embedding exceeds u32".to_owned()))?;
        self.batch_recorder.push_barrier(
            self.pipelines.residual_add,
            &[
                BufferSpan::whole(self.scratch.hidden),
                BufferSpan::whole(self.scratch.delta),
            ],
            &[embed_u32],
            checked_grid(&self.device, [embed_u32.div_ceil(WORKGROUP_X), 1, 1])?,
        );
        Ok(())
    }

    fn record_final_logits(&mut self) -> Result<(), Error> {
        let embed_u32 = u32::try_from(self.geometry.embedding)
            .map_err(|_| Error::ResourceLimit("embedding exceeds u32".to_owned()))?;
        self.batch_recorder.push_barrier(
            self.pipelines.rms_norm,
            &[
                BufferSpan::whole(self.scratch.hidden),
                BufferSpan::whole(self.output_norm_w),
                BufferSpan::whole(self.scratch.norm),
            ],
            &[embed_u32, self.geometry.epsilon.to_bits()],
            checked_grid(&self.device, [1, 1, 1])?,
        );
        self.record_projection_group(
            self.scratch.norm,
            &[(self.output_weight, self.scratch.logits)],
        )
    }

    fn record_projection_group(
        &mut self,
        src_buf: BufferId,
        targets: &[(GpuMatrixWeight, BufferId)],
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

        for &basis in &bases[..base_count] {
            let group: Vec<(GpuMatrixWeight, BufferId)> = targets
                .iter()
                .copied()
                .filter(|(w, _)| w.basis() == basis)
                .collect();
            self.record_basis_projection_subgroup(src_buf, basis, &group)?;
        }
        Ok(())
    }

    fn record_basis_projection_subgroup(
        &mut self,
        src_buf: BufferId,
        basis: (bool, bool),
        group: &[(GpuMatrixWeight, BufferId)],
    ) -> Result<(), Error> {
        let cols = group[0].0.cols;
        let act_buf = self.apply_forward_basis_transform(src_buf, cols, basis)?;

        let needs_q8k = group
            .iter()
            .any(|(w, _)| w.encoding == WeightEncoding::Pq2Q8k);
        let needs_q8_0_source_or_pinned = group
            .iter()
            .any(|(w, _)| w.encoding == WeightEncoding::Pq2Q8_0);
        let needs_ptq1_q8_0 = group
            .iter()
            .any(|(w, _)| w.encoding == WeightEncoding::Ptq1Q8_0);
        if needs_q8_0_source_or_pinned && needs_ptq1_q8_0 && self.resolved_pinned_arithmetic {
            // Execute Q8_0 variants sequentially if a mixed sibling group needs both standard and native scales.
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
            self.dispatch_prepared_projection_group(act_buf, cols, &q8_0_pq2)?;
            return self.dispatch_prepared_projection_group(act_buf, cols, &q8_0_ptq1);
        }

        let _ = (needs_q8k, needs_q8_0_source_or_pinned, needs_ptq1_q8_0);
        self.dispatch_prepared_projection_group(act_buf, cols, group)
    }

    fn apply_forward_basis_transform(
        &mut self,
        src_buf: BufferId,
        cols: usize,
        basis: (bool, bool),
    ) -> Result<BufferId, Error> {
        let (rotated, grouped_ssm) = basis;
        if !rotated {
            return Ok(src_buf);
        }
        let cols_u32 = u32::try_from(cols)
            .map_err(|_| Error::ResourceLimit("transform width exceeds u32".to_owned()))?;
        let mut cur = src_buf;

        if grouped_ssm {
            let rep = u32::try_from(self.geometry.ssm_values / self.geometry.ssm_keys)
                .map_err(|_| Error::ResourceLimit("repetitions exceeds u32".to_owned()))?;
            let kh = u32::try_from(self.geometry.ssm_keys)
                .map_err(|_| Error::ResourceLimit("ssm_keys exceeds u32".to_owned()))?;
            let hd = u32::try_from(self.geometry.ssm_head)
                .map_err(|_| Error::ResourceLimit("ssm_head exceeds u32".to_owned()))?;
            self.batch_recorder.push_barrier(
                self.pipelines.permute_ssm_heads,
                &[
                    BufferSpan::whole(cur),
                    BufferSpan::whole(self.scratch.perm_tmp),
                ],
                &[kh, rep, hd],
                checked_grid(&self.device, [cols_u32.div_ceil(WORKGROUP_X), 1, 1])?,
            );
            cur = self.scratch.perm_tmp;
        }

        if let Some(&sign_buf) = self.transforms.sign_buffers.get(&cols) {
            self.batch_recorder.push_barrier(
                self.pipelines.mul_f32_elemwise,
                &[
                    BufferSpan::whole(cur),
                    BufferSpan::whole(sign_buf),
                    BufferSpan::whole(self.scratch.sign_tmp),
                ],
                &[cols_u32],
                checked_grid(&self.device, [cols_u32.div_ceil(WORKGROUP_X), 1, 1])?,
            );
            cur = self.scratch.sign_tmp;
        }

        if self.transforms.block >= 2 && cols.is_multiple_of(self.transforms.block) {
            let block_u32 = u32::try_from(self.transforms.block)
                .map_err(|_| Error::InvalidInput("Hadamard block exceeds u32".to_owned()))?;
            let blk_cnt = u32::try_from(cols / self.transforms.block)
                .map_err(|_| Error::ResourceLimit("Hadamard grid exceeds u32".to_owned()))?;
            self.batch_recorder.push_barrier(
                self.pipelines.proj.hadamard,
                &[
                    BufferSpan::whole(cur),
                    BufferSpan::whole(self.scratch.transformed),
                ],
                &[block_u32, hadamard_norm_bits(block_u32)],
                checked_grid(&self.device, [blk_cnt, 1, 1])?,
            );
            cur = self.scratch.transformed;
        }

        Ok(cur)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Quantizes shared activation once (Q8K and/or Q8_0) and dispatches sibling projections cleanly."
    )]
    fn dispatch_prepared_projection_group(
        &mut self,
        act_buf: BufferId,
        cols: usize,
        group: &[(GpuMatrixWeight, BufferId)],
    ) -> Result<(), Error> {
        if group.is_empty() {
            return Ok(());
        }
        let cols_u32 = u32::try_from(cols)
            .map_err(|_| Error::ResourceLimit("projection cols exceed u32".to_owned()))?;
        let needs_q8k = group
            .iter()
            .any(|(w, _)| w.encoding == WeightEncoding::Pq2Q8k);
        let q8_0_kind = group.iter().find_map(|(w, _)| match w.encoding {
            WeightEncoding::Pq2Q8_0 => Some(WeightEncoding::Pq2Q8_0),
            WeightEncoding::Ptq1Q8_0 => Some(WeightEncoding::Ptq1Q8_0),
            WeightEncoding::Pq2Q8k | WeightEncoding::F32 => None,
        });

        let quant_emitted = if needs_q8k {
            let q8k_pipe = if self.resolved_pinned_arithmetic {
                self.pipelines.proj.quant_q8k_pinned
            } else {
                self.pipelines.proj.quant_q8k_unpinned
            };
            self.batch_recorder.push_barrier(
                q8k_pipe,
                &[
                    BufferSpan::whole(act_buf),
                    BufferSpan::whole(self.scratch.act_q8k),
                ],
                &[cols_u32, 1],
                checked_grid(&self.device, [cols_u32 / 256, 1, 1])?,
            );
            true
        } else {
            false
        };

        if let Some(q8_kind) = q8_0_kind {
            let q8_pipe = match (self.resolved_pinned_arithmetic, q8_kind) {
                (false, _) => self.pipelines.quant_q8_0_source,
                (true, WeightEncoding::Ptq1Q8_0) => self.pipelines.quant_q8_0_pinned_native,
                (true, _) => self.pipelines.quant_q8_0_pinned,
            };
            let q8_blocks = cols_u32 / 32;
            self.batch_recorder.push(
                q8_pipe,
                &[
                    BufferSpan::whole(act_buf),
                    BufferSpan::whole(self.scratch.act_q8_0),
                ],
                &[q8_blocks],
                checked_grid(&self.device, [q8_blocks.div_ceil(WORKGROUP_X), 1, 1])?,
                !quant_emitted,
            );
        }
        for (idx, (weight, out_buf)) in group.iter().enumerate() {
            let rows_u32 = u32::try_from(weight.rows)
                .map_err(|_| Error::ResourceLimit("projection rows exceed u32".to_owned()))?;
            let barrier_before = idx == 0;
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
                        false,
                        false,
                    );
                    self.batch_recorder.push(
                        pipe,
                        &[
                            weight.span,
                            BufferSpan::whole(self.scratch.act_q8k),
                            BufferSpan::whole(*out_buf),
                        ],
                        &[rows_u32, blocks_u32, 1, act_words],
                        checked_grid(&self.device, [rows_u32.div_ceil(WORKGROUP_X), 1, 1])?,
                        barrier_before,
                    );
                }
                WeightEncoding::Pq2Q8_0 | WeightEncoding::Ptq1Q8_0 => {
                    let blocks_per_row = cols / 128;
                    let blocks_u32 = u32::try_from(blocks_per_row).map_err(|_| {
                        Error::ResourceLimit("blocks_per_row exceed u32".to_owned())
                    })?;
                    let act_words = u32::try_from((cols / 32) * Q8_0_WORDS_PER_BLOCK)
                        .map_err(|_| Error::ResourceLimit("act_words exceed u32".to_owned()))?;
                    let pipe = match weight.encoding {
                        WeightEncoding::Pq2Q8_0 => self.pipelines.proj.pq2_q8_0,
                        WeightEncoding::Ptq1Q8_0 if self.resolved_pinned_arithmetic => {
                            self.pipelines.proj.ptq1_q8_0_pinned
                        }
                        WeightEncoding::Ptq1Q8_0 => self.pipelines.proj.ptq1_q8_0,
                        WeightEncoding::Pq2Q8k | WeightEncoding::F32 => {
                            return Err(Error::InvalidInput("unexpected encoding".to_owned()));
                        }
                    };
                    self.batch_recorder.push(
                        pipe,
                        &[
                            weight.span,
                            BufferSpan::whole(self.scratch.act_q8_0),
                            BufferSpan::whole(*out_buf),
                        ],
                        &[rows_u32, blocks_u32, 1, act_words],
                        checked_grid(&self.device, [rows_u32.div_ceil(WORKGROUP_X), 1, 1])?,
                        barrier_before,
                    );
                }
                WeightEncoding::F32 => {
                    self.batch_recorder.push(
                        self.pipelines.f32_matvec,
                        &[
                            weight.span,
                            BufferSpan::whole(act_buf),
                            BufferSpan::whole(*out_buf),
                        ],
                        &[cols_u32, rows_u32],
                        checked_grid(&self.device, [rows_u32, 1, 1])?,
                        barrier_before,
                    );
                }
            }
        }
        Ok(())
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
    repack_scratch: &mut Vec<u32>,
    row_scratch: &mut Vec<f32>,
    guard: &mut BufferCleanupGuard<'_>,
) -> Result<GpuMatrixWeight, Error> {
    let tensor = lookup_tensor(model, name, &[cols, rows])?;
    upload_matrix_tensor(&tensor, transforms, repack_scratch, row_scratch, guard)
}

fn upload_matrix_tensor(
    tensor: &Tensor,
    transforms: &GraphTransforms,
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
    Ok(GpuMatrixWeight {
        span,
        encoding,
        cols,
        rows,
        rotated,
        grouped_ssm,
    })
}

#[expect(
    clippy::too_many_lines,
    reason = "Uploads all tensors and initial zeroed GPU state buffers for one recurrent GDN layer."
)]
fn upload_recurrent_attention(
    model: &Model,
    prefix: &str,
    geometry: &GraphGeometry,
    transforms: &GraphTransforms,
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
        repack_scratch,
        row_scratch,
        guard,
    )?;

    let conv_state_elems = (geometry.convolution - 1) * channels;
    let conv_state = if conv_state_elems > 0 {
        let conv_bytes = f32_slice_bytes(conv_state_elems)?;
        let id = guard.alloc_storage(conv_bytes)?;
        guard.device.fill_buffer_zero(id, 0, conv_bytes)?;
        id
    } else {
        guard.alloc_storage(MIN_NORM_OR_BIAS_BYTES)?
    };

    let matrix_state_elems = geometry.ssm_head * geometry.ssm_head * geometry.ssm_values;
    let matrix_bytes = f32_slice_bytes(matrix_state_elems)?;
    let matrix_state = guard.alloc_storage(matrix_bytes)?;
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

#[expect(
    clippy::too_many_lines,
    reason = "Computes exact scratch sizes, checks total against scratch_budget_bytes and maxStorageBufferRange, and allocates reusable buffers."
)]
fn allocate_scratch(
    geometry: &GraphGeometry,
    scratch_budget_bytes: u64,
    max_storage_range: u32,
    guard: &mut BufferCleanupGuard<'_>,
) -> Result<GpuGraphScratch, Error> {
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
    let hidden_bytes = f32_slice_bytes(geometry.embedding)?.max(MIN_SCRATCH_BYTES);
    let work_bytes = f32_slice_bytes(max_f32_work)?.max(MIN_SCRATCH_BYTES);
    let act_q8k_words = max_proj_in.div_ceil(128) * Q8K_4PLANE_WORDS_PER_HALF;
    let act_q8k_bytes = u32_slice_bytes(act_q8k_words)?.max(MIN_SCRATCH_BYTES);
    let act_q8_0_words = max_proj_in.div_ceil(32) * Q8_0_WORDS_PER_BLOCK;
    let act_q8_0_bytes = u32_slice_bytes(act_q8_0_words)?.max(MIN_SCRATCH_BYTES);

    let kv_unpack_elems = geometry
        .context_limit
        .checked_mul(geometry.kv_width())
        .ok_or_else(|| Error::ResourceLimit("KV unpack buffer overflow".to_owned()))?;
    let kv_unpack_bytes = f32_slice_bytes(kv_unpack_elems)?.max(MIN_SCRATCH_BYTES);
    let score_elems = geometry
        .context_limit
        .checked_mul(geometry.heads)
        .ok_or_else(|| Error::ResourceLimit("attention score buffer overflow".to_owned()))?;
    let score_bytes = f32_slice_bytes(score_elems)?.max(MIN_SCRATCH_BYTES);
    let logits_bytes = f32_slice_bytes(geometry.vocabulary)?.max(MIN_SCRATCH_BYTES);
    let rope_bytes = f32_slice_bytes(geometry.rope)?.max(MIN_SCRATCH_BYTES);
    let key_scratch_bytes = f32_slice_bytes(geometry.key_width())?.max(MIN_SCRATCH_BYTES);
    let val_scratch_bytes = f32_slice_bytes(geometry.value_width())?.max(MIN_SCRATCH_BYTES);

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
        key_scratch_bytes,
        val_scratch_bytes,
        rope_bytes,
        kv_unpack_bytes,
        kv_unpack_bytes,
        score_bytes,
        score_bytes,
        logits_bytes,
        MIN_NORM_OR_BIAS_BYTES,
    ];
    let total_scratch: u64 = sizes.iter().try_fold(0_u64, |acc, &sz| {
        if sz > u64::from(max_storage_range) {
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

    Ok(GpuGraphScratch {
        hidden: guard.alloc_storage(sizes[0])?,
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
        k_unpacked: guard.alloc_storage(sizes[16])?,
        v_unpacked: guard.alloc_storage(sizes[17])?,
        attn_scores: guard.alloc_storage(sizes[18])?,
        attn_probs: guard.alloc_storage(sizes[19])?,
        logits: guard.alloc_storage(sizes[20])?,
        dummy_state: guard.alloc_storage(sizes[21])?,
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
    let [s0, s1, s2, _] = geometry.rope_sections;
    for pair in 0..half {
        let axis_idx = if mrope_positions[0] == mrope_positions[1]
            && mrope_positions[1] == mrope_positions[2]
        {
            0
        } else {
            let p3 = pair / 3;
            let rem = pair % 3;
            match rem {
                1 if p3 < s1 => 1,
                2 if p3 < s2 => 2,
                _ => {
                    let _ = s0;
                    0
                }
            }
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
    reason = "Binds all static provenance inputs (device, model, chunk policy, kernel path, shaders) into one SHA-256."
)]
fn compute_executor_identity(
    device: &AdrenoVulkanDevice,
    model: &Model,
    config: &VulkanBatchedConfig,
    resolved_context_limit: usize,
    resolved_pinned_arithmetic: bool,
    kernel_path: ProjectionKernelPath,
    layer_count: usize,
    pipelines: &VulkanLayerPipelines,
) -> Result<String, Error> {
    let mut hasher = Hasher::new().map_err(|err| Error::InvalidInput(err.to_string()))?;
    hash_str(&mut hasher, b"vk-batched-graph-v1:", &model.identity)?;
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
    let sample_key = attention_cache_key(
        &sample_desc,
        &sample_plan,
        device.capabilities(),
        &pipelines.attn_qk_digest,
        &pipelines.attn_softmax_digest,
        &pipelines.attn_pv_digest,
        &pipelines.attn_fused_digest,
    )?;
    hash_str(&mut hasher, b"attn_sample:", &sample_key)?;

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
        let make = |pos: usize, val: f32| VulkanBatchedCheckpoint {
            executor_identity_sha256: "exec-abc".to_owned(),
            position: pos,
            mrope_positions: [pos, pos, pos, 0],
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
        let a = make(1, 0.25);
        let b = make(1, 0.25);
        let c = make(2, 0.25);
        let d = make(1, 0.50);
        assert_eq!(a.identity_sha256()?, b.identity_sha256()?);
        assert_ne!(a.identity_sha256()?, c.identity_sha256()?);
        assert_ne!(a.identity_sha256()?, d.identity_sha256()?);
        Ok(())
    }
}
