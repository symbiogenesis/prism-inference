//! Native Adreno Vulkan projection (`#39`) and attention (`#40`) executors.
//!
//! Implements [`Pq2ProjectionExecutor`] with coalesced 64-row `PQ2_0`/`PTQ1_0` SPIR-V 1.6 kernels,
//! on-GPU `F32 -> 4-plane Q8_K` quantization, GPU `SwiGLU`, Sylvester `Hadamard`, rank-1 `LoRA` dot
//! side-branches, `WeightArena` residency, persistent scratch buffers, and multi-token GEMM, plus
//! [`VulkanAttentionExecutor`] supporting single-dispatch fused and 3-stage decomposed attention.
#![forbid(unsafe_code)]

use std::{borrow::Borrow, cmp::Ordering, collections::BTreeMap};

use bonsai_native_hash::Hasher;
use bonsai_prism_runtime::{
    kernels::{Pq2ProjectionExecutor, Q8Block},
    model::{Tensor, TensorBytes, WeakTensorBytes},
};
use bonsai_ternary::Q8KBlock;

use crate::{
    AdrenoAttentionPath, AttentionDescriptor, AttentionExecutionWitness, AttentionOutput,
    AttentionStageOutputs, AttentionStrategy, Error, ProjectionKernelPath,
    VulkanDeviceCapabilities, attention_cache_key,
    device::{AdrenoVulkanDevice, BufferId, BufferSpan, ComputeSpanStage, PipelineId, WeightArena},
    hash_bytes, hash_u32, select_attention_plan, spirv,
};

pub(crate) const KIND_Q8_0: u32 = 8;
pub(crate) const KIND_PQ2: u32 = 142;
pub(crate) const KIND_PTQ1: u32 = 143;
const PQ2_BLOCK_BYTES: usize = 34;
const PTQ1_BLOCK_BYTES: usize = 28;
const Q8_0_BLOCK_BYTES: usize = 34;
const PQ2_COALESCED_TILE_WORDS: usize = 544;
const PTQ1_COALESCED_TILE_WORDS: usize = 448;
const Q8_0_COALESCED_TILE_WORDS: usize = 544;
const Q8K_WORDS_PER_BLOCK: usize = 65;
pub(crate) const Q8K_4PLANE_WORDS_PER_HALF: usize = 36;
pub(crate) const Q8_0_WORDS_PER_BLOCK: usize = 12;
const DEFAULT_OPERATION_CAPACITY: u64 = 1_u64 << 40;
const ARENA_BUDGET_THRESHOLD_BYTES: usize = 256 * 1024 * 1024;
pub(crate) const MIN_SCRATCH_BYTES: u64 = 4096;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct WeightCacheKey {
    name: String,
    kind: u32,
    cols: usize,
    rows: usize,
}

trait WeightKey {
    fn parts(&self) -> (&str, u32, usize, usize);
}

impl WeightKey for WeightCacheKey {
    fn parts(&self) -> (&str, u32, usize, usize) {
        (&self.name, self.kind, self.cols, self.rows)
    }
}

struct WeightCacheLookup<'a> {
    name: &'a str,
    kind: u32,
    cols: usize,
    rows: usize,
}

impl WeightKey for WeightCacheLookup<'_> {
    fn parts(&self) -> (&str, u32, usize, usize) {
        (self.name, self.kind, self.cols, self.rows)
    }
}

impl WeightCacheLookup<'_> {
    const fn as_key(&self) -> &(dyn WeightKey + '_) {
        self
    }

    fn to_owned_key(&self) -> WeightCacheKey {
        WeightCacheKey {
            name: self.name.to_owned(),
            kind: self.kind,
            cols: self.cols,
            rows: self.rows,
        }
    }
}

impl<'a> Borrow<dyn WeightKey + 'a> for WeightCacheKey {
    fn borrow(&self) -> &(dyn WeightKey + 'a) {
        self
    }
}

impl PartialEq for dyn WeightKey + '_ {
    fn eq(&self, other: &Self) -> bool {
        self.parts() == other.parts()
    }
}

impl Eq for dyn WeightKey + '_ {}

impl PartialOrd for dyn WeightKey + '_ {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for dyn WeightKey + '_ {
    fn cmp(&self, other: &Self) -> Ordering {
        self.parts().cmp(&other.parts())
    }
}

#[derive(Clone, Copy)]
enum ResidentStorage {
    Arena(BufferSpan),
    Standalone(BufferId),
}

impl ResidentStorage {
    const fn span(self) -> BufferSpan {
        match self {
            Self::Arena(span) => span,
            Self::Standalone(id) => BufferSpan::whole(id),
        }
    }
}

struct CachedWeight {
    storage: ResidentStorage,
    bytes: usize,
    weak_source: WeakTensorBytes,
    last_used: u64,
}

#[derive(Clone, Copy)]
enum WeightLease {
    Cached(BufferSpan),
    Transient(BufferId),
}

impl WeightLease {
    const fn span(self) -> BufferSpan {
        match self {
            Self::Cached(span) => span,
            Self::Transient(id) => BufferSpan::whole(id),
        }
    }

    const fn buffer_id(self) -> BufferId {
        match self {
            Self::Cached(span) => span.buffer,
            Self::Transient(id) => id,
        }
    }
}

#[derive(Clone, Copy, Default)]
struct ScratchBuf {
    id: Option<BufferId>,
    cap_bytes: u64,
}

impl ScratchBuf {
    fn free(&mut self, device: &mut AdrenoVulkanDevice) {
        if let Some(id) = self.id.take() {
            device.free_buffer(id);
        }
        self.cap_bytes = 0;
    }
}

#[derive(Default)]
struct ProjectionScratchSet {
    act_u32: ScratchBuf,
    f32_in: ScratchBuf,
    q8k_4p: ScratchBuf,
    residual: ScratchBuf,
    out_a: ScratchBuf,
    out_b: ScratchBuf,
    out_c: ScratchBuf,
    mid: ScratchBuf,
    mid_rot: ScratchBuf,
    q8k_mid: ScratchBuf,
    lora_a: ScratchBuf,
    lora_scalar: ScratchBuf,
}

impl ProjectionScratchSet {
    fn free_all(&mut self, device: &mut AdrenoVulkanDevice) {
        self.act_u32.free(device);
        self.f32_in.free(device);
        self.q8k_4p.free(device);
        self.residual.free(device);
        self.out_a.free(device);
        self.out_b.free(device);
        self.out_c.free(device);
        self.mid.free(device);
        self.mid_rot.free(device);
        self.q8k_mid.free(device);
        self.lora_a.free(device);
        self.lora_scalar.free(device);
    }
}

#[derive(Clone, Copy)]
struct FfnChainScratchIds {
    f32_in: BufferId,
    q8k_in: BufferId,
    out_a: BufferId,
    out_b: BufferId,
    mid: BufferId,
    mid_rot: BufferId,
    q8k_mid: BufferId,
    out_c: BufferId,
}

#[derive(Default)]
struct AttentionScratchSet {
    q: ScratchBuf,
    k: ScratchBuf,
    v: ScratchBuf,
    scores: ScratchBuf,
    probs: ScratchBuf,
    out: ScratchBuf,
}

impl AttentionScratchSet {
    fn free_all(&mut self, device: &mut AdrenoVulkanDevice) {
        self.q.free(device);
        self.k.free(device);
        self.v.free(device);
        self.scores.free(device);
        self.probs.free(device);
        self.out.free(device);
    }
}

#[derive(Debug)]
pub(crate) struct ProjectionPipelines {
    pub(crate) pq2_q8k_fused: PipelineId,
    pub(crate) pq2_q8k_unfused: PipelineId,
    pub(crate) pq2_q8k_fused_res: PipelineId,
    pub(crate) pq2_q8k_unfused_res: PipelineId,
    pub(crate) pq2_q8k_4tok_fused: PipelineId,
    pub(crate) pq2_q8k_4tok_unfused: PipelineId,
    pub(crate) pq2_q8k_4tok_fused_res: PipelineId,
    pub(crate) pq2_q8k_4tok_unfused_res: PipelineId,
    pub(crate) pq2_q8k_16tok_fused: PipelineId,
    pub(crate) pq2_q8k_16tok_unfused: PipelineId,
    pub(crate) pq2_q8k_16tok_fused_res: PipelineId,
    pub(crate) pq2_q8k_16tok_unfused_res: PipelineId,
    pub(crate) pq2_q8k_32tok_fused: PipelineId,
    pub(crate) pq2_q8k_32tok_unfused: PipelineId,
    pub(crate) pq2_q8k_32tok_fused_res: PipelineId,
    pub(crate) pq2_q8k_32tok_unfused_res: PipelineId,
    pub(crate) pq2_q8k_splitk_fused: PipelineId,
    pub(crate) pq2_q8k_splitk_unfused: PipelineId,
    pub(crate) pq2_q8k_splitk_fused_res: PipelineId,
    pub(crate) pq2_q8k_splitk_unfused_res: PipelineId,
    pub(crate) pq2_q8_0: PipelineId,
    pub(crate) pq2_q8_0_4tok: PipelineId,
    pub(crate) pq2_q8_0_32tok: PipelineId,
    pub(crate) pq2_q8_0_splitk: PipelineId,
    pub(crate) ptq1_q8_0: PipelineId,
    pub(crate) ptq1_q8_0_pinned: PipelineId,
    pub(crate) ptq1_q8_0_4tok: PipelineId,
    pub(crate) ptq1_q8_0_4tok_pinned: PipelineId,
    pub(crate) ptq1_q8_0_16tok: PipelineId,
    pub(crate) ptq1_q8_0_16tok_pinned: PipelineId,
    pub(crate) ptq1_q8_0_32tok: PipelineId,
    pub(crate) ptq1_q8_0_32tok_pinned: PipelineId,
    pub(crate) ptq1_q8_0_splitk: PipelineId,
    pub(crate) ptq1_q8_0_splitk_pinned: PipelineId,
    pub(crate) q8_0_q8_0: PipelineId,
    pub(crate) q8_0_q8_0_4tok: PipelineId,
    pub(crate) q8_0_q8_0_16tok: PipelineId,
    pub(crate) q8_0_q8_0_32tok: PipelineId,
    pub(crate) q8_0_q8_0_splitk: PipelineId,
    pub(crate) ptq1_q8k: PipelineId,
    pub(crate) quant_q8k_pinned: PipelineId,
    pub(crate) quant_q8k_unpinned: PipelineId,
    pub(crate) swiglu_pinned: PipelineId,
    pub(crate) swiglu_unpinned: PipelineId,
    pub(crate) hadamard: PipelineId,
    pub(crate) dot_scalar: PipelineId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BatchTokenPath {
    Single,
    Four,
    Sixteen,
    ThirtyTwo,
}

impl BatchTokenPath {
    pub(crate) const fn for_tokens(device: &AdrenoVulkanDevice, tokens: usize) -> Self {
        if tokens >= 17 && supports_32tok_gemm(device) {
            Self::ThirtyTwo
        } else if tokens >= 8 && supports_16tok_gemm(device) {
            Self::Sixteen
        } else if tokens > 1 {
            Self::Four
        } else {
            Self::Single
        }
    }

    pub(crate) const fn grid_y(self, tokens_u32: u32) -> u32 {
        match self {
            Self::Single => tokens_u32,
            Self::Four => tokens_u32.div_ceil(4),
            Self::Sixteen => tokens_u32.div_ceil(16),
            Self::ThirtyTwo => tokens_u32.div_ceil(32),
        }
    }

    pub(crate) const fn gemm_grid(self, row_tiles: u32, tokens_u32: u32) -> [u32; 3] {
        let tok_tiles = self.grid_y(tokens_u32);
        match self {
            Self::ThirtyTwo => [tok_tiles, row_tiles, 1],
            Self::Single | Self::Four | Self::Sixteen => [row_tiles, tok_tiles, 1],
        }
    }

    pub(crate) const fn grid_y_four_tok_cap(self, tokens_u32: u32) -> u32 {
        match self {
            Self::Single => tokens_u32,
            Self::Four | Self::Sixteen | Self::ThirtyTwo => tokens_u32.div_ceil(4),
        }
    }
}

pub(crate) const fn supports_16tok_gemm(device: &AdrenoVulkanDevice) -> bool {
    let limits = &device.capabilities().compute_limits;
    limits.max_compute_work_group_invocations >= 512
        && limits.max_compute_work_group_size[0] >= 64
        && limits.max_compute_work_group_size[1] >= 8
        && limits.max_compute_shared_memory_size >= 4_864
}

pub(crate) const fn supports_32tok_gemm(device: &AdrenoVulkanDevice) -> bool {
    let limits = &device.capabilities().compute_limits;
    limits.max_compute_work_group_invocations >= 512
        && limits.max_compute_work_group_size[0] >= 64
        && limits.max_compute_work_group_size[1] >= 8
        && limits.max_compute_shared_memory_size >= 16_640
}

pub(crate) const fn supports_splitk_gemv(device: &AdrenoVulkanDevice) -> bool {
    let limits = &device.capabilities().compute_limits;
    limits.max_compute_work_group_invocations >= 256
        && limits.max_compute_work_group_size[0] >= 64
        && limits.max_compute_work_group_size[1] >= 4
        && limits.max_compute_shared_memory_size >= 1_024
}

pub(crate) const fn should_use_decode_splitk(
    device: &AdrenoVulkanDevice,
    tokens: usize,
    cols: usize,
    rows: usize,
    blocks_per_row: usize,
) -> bool {
    tokens == 1
        && cols > 1024
        && supports_splitk_gemv(device)
        && ((rows <= 4096 && blocks_per_row >= 8) || (rows <= 12_288 && blocks_per_row >= 32))
}

pub(crate) struct PreparedShaderSpec {
    pub(crate) tag: &'static [u8],
    pub(crate) words: Vec<u32>,
    pub(crate) num_buffers: usize,
    pub(crate) push_u32s: usize,
    pub(crate) digest: String,
}

pub(crate) struct PreparedProjectionShaders {
    pub(crate) digest_sha256: String,
    modules: [PreparedShaderSpec; 46],
}

impl ProjectionPipelines {
    #[expect(
        clippy::too_many_lines,
        reason = "Emits and digests all 1-tok, Split-K, 4-tok, 16-tok, 32-tok shared-memory GEMM, quantization, and helper SPIR-V modules sequentially."
    )]
    pub(crate) fn prepare_with_sdot_acc_sat(
        use_sdot: bool,
        use_sdot_acc_sat: bool,
        use_wave64_shuffle: bool,
    ) -> Result<PreparedProjectionShaders, Error> {
        let mut hasher = Hasher::new().map_err(|err| Error::InvalidInput(err.to_string()))?;
        hash_bytes(&mut hasher, b"projection-pipelines-v7:")?;
        hash_u32(&mut hasher, u32::from(use_sdot))?;
        hash_u32(&mut hasher, u32::from(use_sdot_acc_sat))?;
        hash_u32(&mut hasher, u32::from(use_wave64_shuffle))?;
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
            make_spec(
                b"pq2_q8k_fused",
                spirv::pq2_q8k_gemv_coalesced_with_acc_sat(use_sdot, use_sdot_acc_sat, true, false),
                3,
                4,
            )?,
            make_spec(
                b"pq2_q8k_unfused",
                spirv::pq2_q8k_gemv_coalesced_with_acc_sat(
                    use_sdot,
                    use_sdot_acc_sat,
                    false,
                    false,
                ),
                3,
                4,
            )?,
            make_spec(
                b"pq2_q8k_fused_res",
                spirv::pq2_q8k_gemv_coalesced_with_acc_sat(use_sdot, use_sdot_acc_sat, true, true),
                4,
                4,
            )?,
            make_spec(
                b"pq2_q8k_unfused_res",
                spirv::pq2_q8k_gemv_coalesced_with_acc_sat(use_sdot, use_sdot_acc_sat, false, true),
                4,
                4,
            )?,
            make_spec(
                b"pq2_q8k_4tok_fused",
                spirv::pq2_q8k_gemm_4tok_coalesced_with_acc_sat(
                    use_sdot,
                    use_sdot_acc_sat,
                    true,
                    false,
                ),
                3,
                4,
            )?,
            make_spec(
                b"pq2_q8k_4tok_unfused",
                spirv::pq2_q8k_gemm_4tok_coalesced_with_acc_sat(
                    use_sdot,
                    use_sdot_acc_sat,
                    false,
                    false,
                ),
                3,
                4,
            )?,
            make_spec(
                b"pq2_q8k_4tok_fused_res",
                spirv::pq2_q8k_gemm_4tok_coalesced_with_acc_sat(
                    use_sdot,
                    use_sdot_acc_sat,
                    true,
                    true,
                ),
                4,
                4,
            )?,
            make_spec(
                b"pq2_q8k_4tok_unfused_res",
                spirv::pq2_q8k_gemm_4tok_coalesced_with_acc_sat(
                    use_sdot,
                    use_sdot_acc_sat,
                    false,
                    true,
                ),
                4,
                4,
            )?,
            make_spec(
                b"pq2_q8k_16tok_fused",
                spirv::pq2_q8k_gemm_16tok_coalesced_with_acc_sat(
                    use_sdot,
                    use_sdot_acc_sat,
                    true,
                    false,
                ),
                3,
                4,
            )?,
            make_spec(
                b"pq2_q8k_16tok_unfused",
                spirv::pq2_q8k_gemm_16tok_coalesced_with_acc_sat(
                    use_sdot,
                    use_sdot_acc_sat,
                    false,
                    false,
                ),
                3,
                4,
            )?,
            make_spec(
                b"pq2_q8k_16tok_fused_res",
                spirv::pq2_q8k_gemm_16tok_coalesced_with_acc_sat(
                    use_sdot,
                    use_sdot_acc_sat,
                    true,
                    true,
                ),
                4,
                4,
            )?,
            make_spec(
                b"pq2_q8k_16tok_unfused_res",
                spirv::pq2_q8k_gemm_16tok_coalesced_with_acc_sat(
                    use_sdot,
                    use_sdot_acc_sat,
                    false,
                    true,
                ),
                4,
                4,
            )?,
            make_spec(
                b"pq2_q8k_32tok_fused",
                spirv::pq2_q8k_gemm_32tok_coalesced_with_acc_sat(
                    use_sdot,
                    use_sdot_acc_sat,
                    true,
                    false,
                ),
                3,
                4,
            )?,
            make_spec(
                b"pq2_q8k_32tok_unfused",
                spirv::pq2_q8k_gemm_32tok_coalesced_with_acc_sat(
                    use_sdot,
                    use_sdot_acc_sat,
                    false,
                    false,
                ),
                3,
                4,
            )?,
            make_spec(
                b"pq2_q8k_32tok_fused_res",
                spirv::pq2_q8k_gemm_32tok_coalesced_with_acc_sat(
                    use_sdot,
                    use_sdot_acc_sat,
                    true,
                    true,
                ),
                4,
                4,
            )?,
            make_spec(
                b"pq2_q8k_32tok_unfused_res",
                spirv::pq2_q8k_gemm_32tok_coalesced_with_acc_sat(
                    use_sdot,
                    use_sdot_acc_sat,
                    false,
                    true,
                ),
                4,
                4,
            )?,
            make_spec(
                b"pq2_q8k_splitk_fused",
                spirv::pq2_q8k_gemv_splitk_with_acc_sat(use_sdot, use_sdot_acc_sat, true, false),
                3,
                4,
            )?,
            make_spec(
                b"pq2_q8k_splitk_unfused",
                spirv::pq2_q8k_gemv_splitk_with_acc_sat(use_sdot, use_sdot_acc_sat, false, false),
                3,
                4,
            )?,
            make_spec(
                b"pq2_q8k_splitk_fused_res",
                spirv::pq2_q8k_gemv_splitk_with_acc_sat(use_sdot, use_sdot_acc_sat, true, true),
                4,
                4,
            )?,
            make_spec(
                b"pq2_q8k_splitk_unfused_res",
                spirv::pq2_q8k_gemv_splitk_with_acc_sat(use_sdot, use_sdot_acc_sat, false, true),
                4,
                4,
            )?,
            make_spec(
                b"pq2_q8_0",
                spirv::pq2_q8_0_gemv_coalesced_with_acc_sat(use_sdot, use_sdot_acc_sat),
                3,
                4,
            )?,
            make_spec(
                b"pq2_q8_0_4tok",
                spirv::pq2_q8_0_gemm_4tok_coalesced_with_acc_sat(use_sdot, use_sdot_acc_sat),
                3,
                4,
            )?,
            make_spec(
                b"pq2_q8_0_32tok",
                spirv::pq2_q8_0_gemm_32tok_coalesced_with_acc_sat(use_sdot, use_sdot_acc_sat),
                3,
                4,
            )?,
            make_spec(
                b"pq2_q8_0_splitk",
                spirv::pq2_q8_0_gemv_splitk_with_acc_sat(use_sdot, use_sdot_acc_sat),
                3,
                4,
            )?,
            make_spec(
                b"ptq1_q8_0",
                spirv::ptq1_q8_0_gemv_coalesced_with_acc_sat(use_sdot, use_sdot_acc_sat, false),
                3,
                4,
            )?,
            make_spec(
                b"ptq1_q8_0_pinned",
                spirv::ptq1_q8_0_gemv_coalesced_with_acc_sat(use_sdot, use_sdot_acc_sat, true),
                3,
                4,
            )?,
            make_spec(
                b"ptq1_q8_0_4tok",
                spirv::ptq1_q8_0_gemm_4tok_coalesced_with_acc_sat(
                    use_sdot,
                    use_sdot_acc_sat,
                    false,
                ),
                3,
                4,
            )?,
            make_spec(
                b"ptq1_q8_0_4tok_pinned",
                spirv::ptq1_q8_0_gemm_4tok_coalesced_with_acc_sat(use_sdot, use_sdot_acc_sat, true),
                3,
                4,
            )?,
            make_spec(
                b"ptq1_q8_0_16tok",
                spirv::ptq1_q8_0_gemm_16tok_coalesced_with_acc_sat(
                    use_sdot,
                    use_sdot_acc_sat,
                    false,
                ),
                3,
                4,
            )?,
            make_spec(
                b"ptq1_q8_0_16tok_pinned",
                spirv::ptq1_q8_0_gemm_16tok_coalesced_with_acc_sat(
                    use_sdot,
                    use_sdot_acc_sat,
                    true,
                ),
                3,
                4,
            )?,
            make_spec(
                b"ptq1_q8_0_32tok",
                spirv::ptq1_q8_0_gemm_32tok_coalesced_with_acc_sat(
                    use_sdot,
                    use_sdot_acc_sat,
                    false,
                ),
                3,
                4,
            )?,
            make_spec(
                b"ptq1_q8_0_32tok_pinned",
                spirv::ptq1_q8_0_gemm_32tok_coalesced_with_acc_sat(
                    use_sdot,
                    use_sdot_acc_sat,
                    true,
                ),
                3,
                4,
            )?,
            make_spec(
                b"ptq1_q8_0_splitk",
                spirv::ptq1_q8_0_gemv_splitk_with_acc_sat(use_sdot, use_sdot_acc_sat, false),
                3,
                4,
            )?,
            make_spec(
                b"ptq1_q8_0_splitk_pinned",
                spirv::ptq1_q8_0_gemv_splitk_with_acc_sat(use_sdot, use_sdot_acc_sat, true),
                3,
                4,
            )?,
            make_spec(
                b"q8_0_q8_0",
                spirv::q8_0_q8_0_gemv_coalesced_with_acc_sat(use_sdot, use_sdot_acc_sat),
                3,
                4,
            )?,
            make_spec(
                b"q8_0_q8_0_4tok",
                spirv::q8_0_q8_0_gemm_4tok_coalesced_with_acc_sat(use_sdot, use_sdot_acc_sat),
                3,
                4,
            )?,
            make_spec(
                b"q8_0_q8_0_16tok",
                spirv::q8_0_q8_0_gemm_16tok_coalesced_with_acc_sat(use_sdot, use_sdot_acc_sat),
                3,
                4,
            )?,
            make_spec(
                b"q8_0_q8_0_32tok",
                spirv::q8_0_q8_0_gemm_32tok_coalesced_with_acc_sat(use_sdot, use_sdot_acc_sat),
                3,
                4,
            )?,
            make_spec(
                b"q8_0_q8_0_splitk",
                spirv::q8_0_q8_0_gemv_splitk_with_acc_sat(use_sdot, use_sdot_acc_sat),
                3,
                4,
            )?,
            make_spec(
                b"ptq1_q8k",
                spirv::ptq1_q8k_gemv_coalesced_with_acc_sat(use_sdot, use_sdot_acc_sat),
                3,
                4,
            )?,
            make_spec(
                b"quant_q8k_pinned",
                spirv::quantize_f32_to_q8k_4plane_with_subgroup(true, use_wave64_shuffle),
                2,
                2,
            )?,
            make_spec(
                b"quant_q8k_unpinned",
                spirv::quantize_f32_to_q8k_4plane_with_subgroup(false, use_wave64_shuffle),
                2,
                2,
            )?,
            make_spec(b"swiglu_pinned", spirv::ffn_swiglu_f32(true), 3, 1)?,
            make_spec(b"swiglu_unpinned", spirv::ffn_swiglu_f32(false), 3, 1)?,
            make_spec(b"hadamard", spirv::ffn_hadamard_f32(), 2, 2)?,
            make_spec(b"dot_scalar", spirv::dot_f32_scalar(), 3, 1)?,
        ];
        let digest_bytes = hasher
            .finish()
            .map_err(|err| Error::InvalidInput(err.to_string()))?;

        Ok(PreparedProjectionShaders {
            digest_sha256: hex::encode(digest_bytes),
            modules,
        })
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Compiles all 46 prepared projection SPIR-V modules into Vulkan compute pipelines."
    )]
    pub(crate) fn compile_prepared(
        device: &mut AdrenoVulkanDevice,
        prepared: PreparedProjectionShaders,
    ) -> Result<Self, Error> {
        let mut ids = Vec::with_capacity(prepared.modules.len());
        for spec in prepared.modules {
            let _ = spec.tag;
            let _ = spec.digest;
            ids.push(device.create_compute_pipeline(
                &spec.words,
                spec.num_buffers,
                spec.push_u32s,
            )?);
        }
        let [
            pq2_q8k_fused,
            pq2_q8k_unfused,
            pq2_q8k_fused_res,
            pq2_q8k_unfused_res,
            pq2_q8k_4tok_fused,
            pq2_q8k_4tok_unfused,
            pq2_q8k_4tok_fused_res,
            pq2_q8k_4tok_unfused_res,
            pq2_q8k_16tok_fused,
            pq2_q8k_16tok_unfused,
            pq2_q8k_16tok_fused_res,
            pq2_q8k_16tok_unfused_res,
            pq2_q8k_32tok_fused,
            pq2_q8k_32tok_unfused,
            pq2_q8k_32tok_fused_res,
            pq2_q8k_32tok_unfused_res,
            pq2_q8k_splitk_fused,
            pq2_q8k_splitk_unfused,
            pq2_q8k_splitk_fused_res,
            pq2_q8k_splitk_unfused_res,
            pq2_q8_0,
            pq2_q8_0_4tok,
            pq2_q8_0_32tok,
            pq2_q8_0_splitk,
            ptq1_q8_0,
            ptq1_q8_0_pinned,
            ptq1_q8_0_4tok,
            ptq1_q8_0_4tok_pinned,
            ptq1_q8_0_16tok,
            ptq1_q8_0_16tok_pinned,
            ptq1_q8_0_32tok,
            ptq1_q8_0_32tok_pinned,
            ptq1_q8_0_splitk,
            ptq1_q8_0_splitk_pinned,
            q8_0_q8_0,
            q8_0_q8_0_4tok,
            q8_0_q8_0_16tok,
            q8_0_q8_0_32tok,
            q8_0_q8_0_splitk,
            ptq1_q8k,
            quant_q8k_pinned,
            quant_q8k_unpinned,
            swiglu_pinned,
            swiglu_unpinned,
            hadamard,
            dot_scalar,
        ]: [PipelineId; 46] = ids
            .try_into()
            .map_err(|_| Error::InvalidInput("projection pipeline count mismatch".to_owned()))?;

        Ok(Self {
            pq2_q8k_fused,
            pq2_q8k_unfused,
            pq2_q8k_fused_res,
            pq2_q8k_unfused_res,
            pq2_q8k_4tok_fused,
            pq2_q8k_4tok_unfused,
            pq2_q8k_4tok_fused_res,
            pq2_q8k_4tok_unfused_res,
            pq2_q8k_16tok_fused,
            pq2_q8k_16tok_unfused,
            pq2_q8k_16tok_fused_res,
            pq2_q8k_16tok_unfused_res,
            pq2_q8k_32tok_fused,
            pq2_q8k_32tok_unfused,
            pq2_q8k_32tok_fused_res,
            pq2_q8k_32tok_unfused_res,
            pq2_q8k_splitk_fused,
            pq2_q8k_splitk_unfused,
            pq2_q8k_splitk_fused_res,
            pq2_q8k_splitk_unfused_res,
            pq2_q8_0,
            pq2_q8_0_4tok,
            pq2_q8_0_32tok,
            pq2_q8_0_splitk,
            ptq1_q8_0,
            ptq1_q8_0_pinned,
            ptq1_q8_0_4tok,
            ptq1_q8_0_4tok_pinned,
            ptq1_q8_0_16tok,
            ptq1_q8_0_16tok_pinned,
            ptq1_q8_0_32tok,
            ptq1_q8_0_32tok_pinned,
            ptq1_q8_0_splitk,
            ptq1_q8_0_splitk_pinned,
            q8_0_q8_0,
            q8_0_q8_0_4tok,
            q8_0_q8_0_16tok,
            q8_0_q8_0_32tok,
            q8_0_q8_0_splitk,
            ptq1_q8k,
            quant_q8k_pinned,
            quant_q8k_unpinned,
            swiglu_pinned,
            swiglu_unpinned,
            hadamard,
            dot_scalar,
        })
    }

    pub(crate) const fn select_pq2_q8k(
        &self,
        fused: bool,
        path: BatchTokenPath,
        has_residual: bool,
    ) -> PipelineId {
        match (path, fused, has_residual) {
            (BatchTokenPath::Single, true, false) => self.pq2_q8k_fused,
            (BatchTokenPath::Single, false, false) => self.pq2_q8k_unfused,
            (BatchTokenPath::Single, true, true) => self.pq2_q8k_fused_res,
            (BatchTokenPath::Single, false, true) => self.pq2_q8k_unfused_res,
            (BatchTokenPath::Four, true, false) => self.pq2_q8k_4tok_fused,
            (BatchTokenPath::Four, false, false) => self.pq2_q8k_4tok_unfused,
            (BatchTokenPath::Four, true, true) => self.pq2_q8k_4tok_fused_res,
            (BatchTokenPath::Four, false, true) => self.pq2_q8k_4tok_unfused_res,
            (BatchTokenPath::Sixteen, true, false) => self.pq2_q8k_16tok_fused,
            (BatchTokenPath::Sixteen, false, false) => self.pq2_q8k_16tok_unfused,
            (BatchTokenPath::Sixteen, true, true) => self.pq2_q8k_16tok_fused_res,
            (BatchTokenPath::Sixteen, false, true) => self.pq2_q8k_16tok_unfused_res,
            (BatchTokenPath::ThirtyTwo, true, false) => self.pq2_q8k_32tok_fused,
            (BatchTokenPath::ThirtyTwo, false, false) => self.pq2_q8k_32tok_unfused,
            (BatchTokenPath::ThirtyTwo, true, true) => self.pq2_q8k_32tok_fused_res,
            (BatchTokenPath::ThirtyTwo, false, true) => self.pq2_q8k_32tok_unfused_res,
        }
    }

    pub(crate) const fn select_pq2_q8k_decode_splitk(
        &self,
        fused: bool,
        has_residual: bool,
    ) -> PipelineId {
        match (fused, has_residual) {
            (true, false) => self.pq2_q8k_splitk_fused,
            (false, false) => self.pq2_q8k_splitk_unfused,
            (true, true) => self.pq2_q8k_splitk_fused_res,
            (false, true) => self.pq2_q8k_splitk_unfused_res,
        }
    }

    pub(crate) const fn select_pq2_q8_0(&self, path: BatchTokenPath) -> PipelineId {
        match path {
            BatchTokenPath::Single => self.pq2_q8_0,
            BatchTokenPath::Four | BatchTokenPath::Sixteen => self.pq2_q8_0_4tok,
            BatchTokenPath::ThirtyTwo => self.pq2_q8_0_32tok,
        }
    }

    pub(crate) const fn select_pq2_q8_0_decode_splitk(&self) -> PipelineId {
        self.pq2_q8_0_splitk
    }

    pub(crate) const fn select_ptq1_q8_0(&self, pinned: bool, path: BatchTokenPath) -> PipelineId {
        match (path, pinned) {
            (BatchTokenPath::Single, false) => self.ptq1_q8_0,
            (BatchTokenPath::Single, true) => self.ptq1_q8_0_pinned,
            (BatchTokenPath::Four, false) => self.ptq1_q8_0_4tok,
            (BatchTokenPath::Four, true) => self.ptq1_q8_0_4tok_pinned,
            (BatchTokenPath::Sixteen, false) => self.ptq1_q8_0_16tok,
            (BatchTokenPath::Sixteen, true) => self.ptq1_q8_0_16tok_pinned,
            (BatchTokenPath::ThirtyTwo, false) => self.ptq1_q8_0_32tok,
            (BatchTokenPath::ThirtyTwo, true) => self.ptq1_q8_0_32tok_pinned,
        }
    }

    pub(crate) const fn select_ptq1_q8_0_decode_splitk(&self, pinned: bool) -> PipelineId {
        if pinned {
            self.ptq1_q8_0_splitk_pinned
        } else {
            self.ptq1_q8_0_splitk
        }
    }

    pub(crate) const fn select_q8_0_q8_0(&self, path: BatchTokenPath) -> PipelineId {
        match path {
            BatchTokenPath::Single => self.q8_0_q8_0,
            BatchTokenPath::Four => self.q8_0_q8_0_4tok,
            BatchTokenPath::Sixteen => self.q8_0_q8_0_16tok,
            BatchTokenPath::ThirtyTwo => self.q8_0_q8_0_32tok,
        }
    }

    pub(crate) const fn select_q8_0_q8_0_decode_splitk(&self) -> PipelineId {
        self.q8_0_q8_0_splitk
    }
}

/// Scoped GPU projection executor (`#39`) backed by coalesced native Adreno SPIR-V 1.6 shaders.
///
/// Supports integer `Q8_K` / `Q8_0` projections, direct GPU `F32 -> 4-plane Q8_K` quantization,
/// multi-token GEMM tiles, fused GPU FFN `gate/up + SwiGLU + Hadamard + down + residual`, and
/// rank-1 `LoRA` A dot side-branches with persistent scratch reuse and `WeightArena` residency.
pub struct VulkanPq2Executor {
    device: AdrenoVulkanDevice,
    kernel_path: ProjectionKernelPath,
    identity: String,
    pipes: ProjectionPipelines,
    weight_arena: WeightArena,
    weight_cache: BTreeMap<WeightCacheKey, CachedWeight>,
    cache_access_counter: u64,
    cached_bytes: usize,
    max_cached_weight_bytes: usize,
    scratch: ProjectionScratchSet,
    act_pack_scratch: Vec<u32>,
    f32_pack_scratch: Vec<f32>,
    repack_scratch: Vec<u32>,
}

impl VulkanPq2Executor {
    /// Opens the native Adreno Vulkan device, auto-selecting [`ProjectionKernelPath::IntegerDotPacked4x8`]
    /// when `VK_KHR_shader_integer_dot_product` (`4x8BitPackedSignedAccelerated`) is enabled, or
    /// [`ProjectionKernelPath::PortableIntegerFallback`] otherwise.
    ///
    /// # Errors
    /// Propagates device discovery, loader, or SPIR-V pipeline creation errors.
    pub fn open_default(max_cached_weight_bytes: usize) -> Result<Self, Error> {
        let device = AdrenoVulkanDevice::open()?;
        let path = if device.integer_dot_product_enabled()
            && device.capabilities().supports_integer_dot_4x8()
        {
            ProjectionKernelPath::IntegerDotPacked4x8
        } else {
            ProjectionKernelPath::PortableIntegerFallback
        };
        Self::from_device(device, path, max_cached_weight_bytes)
    }

    /// Opens the native Adreno Vulkan device requiring a specific [`ProjectionKernelPath`].
    ///
    /// # Errors
    /// Returns [`Error::Unsupported`] if [`ProjectionKernelPath::IntegerDotPacked4x8`] is requested
    /// on a device lacking `VK_KHR_shader_integer_dot_product`, or propagates device/pipeline errors.
    pub fn open_with_path(
        path: ProjectionKernelPath,
        max_cached_weight_bytes: usize,
    ) -> Result<Self, Error> {
        let device = AdrenoVulkanDevice::open()?;
        if path == ProjectionKernelPath::IntegerDotPacked4x8
            && !device.integer_dot_product_enabled()
        {
            return Err(Error::Unsupported);
        }
        Self::from_device(device, path, max_cached_weight_bytes)
    }

    fn from_device(
        mut device: AdrenoVulkanDevice,
        kernel_path: ProjectionKernelPath,
        max_cached_weight_bytes: usize,
    ) -> Result<Self, Error> {
        let use_sdot = spirv::path_uses_sdot(kernel_path);
        let use_sdot_acc_sat = use_sdot && device.integer_dot_product_acc_sat_enabled();
        let use_wave64 = device.capabilities().wave64_subgroup_shuffle;
        let prepared =
            ProjectionPipelines::prepare_with_sdot_acc_sat(use_sdot, use_sdot_acc_sat, use_wave64)?;
        let pipes = ProjectionPipelines::compile_prepared(&mut device, prepared)?;
        let weight_arena = WeightArena::new(&device);
        let caps = device.capabilities();
        let short_hash = &caps.identity_sha256[..16.min(caps.identity_sha256.len())];
        let identity = format!(
            "adreno-vulkan-v2:{}:{}:{}",
            caps.family.as_tag(),
            short_hash,
            kernel_path.as_tag()
        );
        if identity.is_empty()
            || identity.len() > 256
            || !identity.bytes().all(|byte| byte.is_ascii_graphic())
        {
            return Err(Error::InvalidInput(
                "constructed executor identity is invalid".to_owned(),
            ));
        }

        Ok(Self {
            device,
            kernel_path,
            identity,
            pipes,
            weight_arena,
            weight_cache: BTreeMap::new(),
            cache_access_counter: 0,
            cached_bytes: 0,
            max_cached_weight_bytes,
            scratch: ProjectionScratchSet::default(),
            act_pack_scratch: Vec::new(),
            f32_pack_scratch: Vec::new(),
            repack_scratch: Vec::new(),
        })
    }

    /// Returns the discovered Qualcomm Adreno device capability snapshot.
    #[must_use]
    pub const fn capabilities(&self) -> &VulkanDeviceCapabilities {
        self.device.capabilities()
    }

    /// Returns the active [`ProjectionKernelPath`].
    #[must_use]
    pub const fn kernel_path(&self) -> ProjectionKernelPath {
        self.kernel_path
    }

    /// Returns the currently resident cached GPU weight payload size in bytes.
    #[must_use]
    pub const fn cached_weight_bytes(&self) -> usize {
        self.cached_bytes
    }

    /// Returns `true` if a device timeout or fault has latched terminal teardown.
    #[must_use]
    pub const fn is_faulted(&self) -> bool {
        self.device.is_faulted()
    }

    /// Explicitly releases all resident cached GPU weight buffers and returns freed bytes.
    pub fn release_resident_weights(&mut self) -> usize {
        let released = self.cached_bytes;
        let entries = std::mem::take(&mut self.weight_cache);
        for (_, cached) in entries {
            if let ResidentStorage::Standalone(id) = cached.storage {
                self.device.free_buffer(id);
            }
        }
        self.weight_arena.free_all(&mut self.device);
        self.cached_bytes = 0;
        released
    }

    /// Repacks and uploads all supported `PQ2_0`/`PTQ1_0`/`Q8_0` tensors into GPU resident storage ahead
    /// of execution, returning the number of tensors prepared or refreshed.
    ///
    /// # Errors
    /// Propagates weight validation or Vulkan upload/allocation errors.
    pub fn prewarm_tensors(&mut self, tensors: &[&Tensor]) -> Result<usize, Error> {
        let mut warmed = 0_usize;
        for &tensor in tensors {
            let valid = match tensor.kind {
                KIND_PQ2 => validate_matrix(tensor, KIND_PQ2, 128).ok(),
                KIND_PTQ1 => validate_matrix(tensor, KIND_PTQ1, 128).ok(),
                KIND_Q8_0 => validate_matrix(tensor, KIND_Q8_0, 32).ok(),
                _ => None,
            };
            let Some((cols, rows)) = valid else {
                continue;
            };
            let lease = self.acquire_weight_buffer(tensor, cols, rows, &[])?;
            match lease {
                WeightLease::Cached(_) => warmed += 1,
                WeightLease::Transient(id) => self.device.free_buffer(id),
            }
        }
        Ok(warmed)
    }

    /// Projects a `PTQ1_0` (`kind == 143`) weight matrix directly against prepared `Q8_K` blocks.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`], [`Error::ResourceLimit`], [`Error::DeviceFault`], or [`Error::Vulkan`].
    pub fn project_ptq1_q8k(
        &mut self,
        tensor: &Tensor,
        activation: &[Q8KBlock],
        output: &mut Vec<f32>,
    ) -> Result<(), Error> {
        output.clear();
        let (cols, rows) = validate_matrix(tensor, KIND_PTQ1, 128)?;
        let blocks_per_row = cols / 128;
        let expected_q8k = blocks_per_row.div_ceil(2);
        if activation.len() != expected_q8k {
            return Err(Error::InvalidInput(format!(
                "PTQ1 x Q8K expected {expected_q8k} activation blocks, got {}",
                activation.len()
            )));
        }
        pack_q8k_blocks_into(activation, &mut self.act_pack_scratch)?;
        let act_words_per_token = u32::try_from(self.act_pack_scratch.len())
            .map_err(|_| Error::ResourceLimit("activation word count overflow".to_owned()))?;
        let words = std::mem::take(&mut self.act_pack_scratch);
        let res = self.dispatch_single_word_gemv(
            tensor,
            cols,
            rows,
            blocks_per_row,
            1,
            act_words_per_token,
            &words,
            self.pipes.ptq1_q8k,
            output,
        );
        self.act_pack_scratch = words;
        res
    }

    /// Batched multi-token `PQ2_0` × `Q8_K` projection writing `[tokens, rows]` into `output`.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`], [`Error::ResourceLimit`], [`Error::DeviceFault`], or [`Error::Vulkan`].
    pub fn project_pq2_q8k_tokens(
        &mut self,
        tensor: &Tensor,
        activations: &[Q8KBlock],
        tokens: usize,
        fused: bool,
        output: &mut Vec<f32>,
    ) -> Result<(), Error> {
        output.clear();
        if tokens == 0 {
            return Err(Error::InvalidInput(
                "multi-token projection requires tokens >= 1".to_owned(),
            ));
        }
        let (cols, rows) = validate_matrix(tensor, KIND_PQ2, 256)?;
        let blocks_per_row = cols / 128;
        let q8k_per_token = cols / 256;
        let expected = q8k_per_token
            .checked_mul(tokens)
            .ok_or_else(|| Error::InvalidInput("activation token count overflow".to_owned()))?;
        if activations.len() != expected {
            return Err(Error::InvalidInput(format!(
                "PQ2 x Q8K multi-token expected {expected} blocks, got {}",
                activations.len()
            )));
        }
        pack_q8k_blocks_4plane_into(activations, &mut self.act_pack_scratch)?;
        let act_words_per_token = u32::try_from(blocks_per_row * Q8K_4PLANE_WORDS_PER_HALF)
            .map_err(|_| Error::ResourceLimit("activation word count overflow".to_owned()))?;
        let path = BatchTokenPath::for_tokens(&self.device, tokens);
        let pipe = self.pipes.select_pq2_q8k(fused, path, false);
        let words = std::mem::take(&mut self.act_pack_scratch);
        let res = self.dispatch_single_word_gemv(
            tensor,
            cols,
            rows,
            blocks_per_row,
            tokens,
            act_words_per_token,
            &words,
            pipe,
            output,
        );
        self.act_pack_scratch = words;
        res
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "Centralizes single-projection weight lease, scratch activation/output upload, and finite verification."
    )]
    fn dispatch_single_word_gemv(
        &mut self,
        tensor: &Tensor,
        cols: usize,
        rows: usize,
        blocks_per_row: usize,
        tokens: usize,
        act_words_per_token: u32,
        act_words: &[u32],
        pipeline: PipelineId,
        output: &mut Vec<f32>,
    ) -> Result<(), Error> {
        self.ensure_healthy()?;
        let (rows_u32, blocks_u32, tokens_u32) = checked_gemv_dims(rows, blocks_per_row, tokens)?;
        let total_outputs = rows
            .checked_mul(tokens)
            .ok_or_else(|| Error::ResourceLimit("projection output count overflow".to_owned()))?;
        let act_bytes = u32_slice_bytes(act_words.len())?;
        let out_bytes = f32_slice_bytes(total_outputs)?;
        let raw_grid = if is_32tok_pipeline(&self.pipes, pipeline) {
            [tokens_u32.div_ceil(32), rows_u32.div_ceil(64), 1]
        } else {
            let grid_y = if is_16tok_pipeline(&self.pipes, pipeline) {
                tokens_u32.div_ceil(16)
            } else if is_4tok_pipeline(&self.pipes, pipeline) {
                tokens_u32.div_ceil(4)
            } else {
                tokens_u32
            };
            [rows_u32.div_ceil(64), grid_y, 1]
        };
        let grid = checked_grid(&self.device, raw_grid)?;

        let act_buf = ensure_scratch(&mut self.device, &mut self.scratch.act_u32, act_bytes)?;
        let out_buf = ensure_scratch(&mut self.device, &mut self.scratch.out_a, out_bytes)?;
        self.device.write_buffer_u32(act_buf, act_words)?;

        let lease = self.acquire_weight_buffer(tensor, cols, rows, &[])?;
        let push = [rows_u32, blocks_u32, tokens_u32, act_words_per_token];
        let bufs = [
            lease.span(),
            BufferSpan::whole(act_buf),
            BufferSpan::whole(out_buf),
        ];
        let stage = ComputeSpanStage {
            pipeline,
            buffers: &bufs,
            push_u32s: &push,
            workgroups: grid,
        };
        let run_result = (|| -> Result<(), Error> {
            self.device.submit_compute_span_stages(&[stage])?;
            output.resize(total_outputs, 0.0_f32);
            self.device.read_buffer_f32(out_buf, output)?;
            ensure_finite_f32(output, "GPU projection produced non-finite output")
        })();
        self.release_transient_lease(lease);
        if run_result.is_err() {
            output.clear();
        }
        run_result
    }

    fn ensure_healthy(&self) -> Result<(), Error> {
        if self.is_faulted() {
            Err(Error::DeviceFault(
                "Vulkan executor faulted and requires terminal teardown".to_owned(),
            ))
        } else {
            Ok(())
        }
    }

    fn release_transient_lease(&mut self, lease: WeightLease) {
        if let WeightLease::Transient(id) = lease {
            self.device.free_buffer(id);
        }
    }

    fn release_transient_leases(&mut self, leases: &[WeightLease]) {
        for &lease in leases {
            self.release_transient_lease(lease);
        }
    }

    fn acquire_weight_buffer(
        &mut self,
        tensor: &Tensor,
        cols: usize,
        rows: usize,
        protected: &[BufferId],
    ) -> Result<WeightLease, Error> {
        let lookup = WeightCacheLookup {
            name: &tensor.name,
            kind: tensor.kind,
            cols,
            rows,
        };
        let mut stale_protected_standalone = false;
        if let Some(cached) = self.weight_cache.get_mut(lookup.as_key()) {
            if cached
                .weak_source
                .upgrade()
                .is_some_and(|live| TensorBytes::ptr_eq(&live, &tensor.data))
            {
                self.cache_access_counter += 1;
                cached.last_used = self.cache_access_counter;
                return Ok(WeightLease::Cached(cached.storage.span()));
            }
            let stale_buf = cached.storage.span().buffer;
            if protected.contains(&stale_buf) {
                stale_protected_standalone =
                    matches!(cached.storage, ResidentStorage::Standalone(_));
            } else if let Some(removed) = self.weight_cache.remove(lookup.as_key()) {
                self.cached_bytes = self.cached_bytes.saturating_sub(removed.bytes);
                if let ResidentStorage::Standalone(id) = removed.storage {
                    self.device.free_buffer(id);
                }
            }
        }

        validate_weight_scales(tensor.kind, &tensor.data)?;
        repack_coalesced_into(
            tensor.kind,
            &tensor.data,
            cols,
            rows,
            &mut self.repack_scratch,
        )?;
        let upload_bytes = u32_slice_bytes_usize(self.repack_scratch.len())?;
        let upload_u64 = u64::try_from(upload_bytes)
            .map_err(|_| Error::ResourceLimit("weight byte length overflow".to_owned()))?;
        let max_range = u64::from(
            self.device
                .capabilities()
                .compute_limits
                .max_storage_buffer_range,
        );
        if upload_u64 > max_range {
            return Err(Error::ResourceLimit(format!(
                "weight tensor '{}' ({upload_u64} B) exceeds maxStorageBufferRange ({max_range} B)",
                tensor.name
            )));
        }

        if stale_protected_standalone
            || self.max_cached_weight_bytes == 0
            || upload_bytes > self.max_cached_weight_bytes
        {
            return self.upload_standalone_transient(upload_u64);
        }

        if self.max_cached_weight_bytes >= ARENA_BUDGET_THRESHOLD_BYTES
            && self.cached_bytes + upload_bytes <= self.max_cached_weight_bytes
        {
            let span = self.weight_arena.alloc(&mut self.device, upload_u64)?;
            self.device
                .write_buffer_u32_at(span.buffer, span.offset, &self.repack_scratch)?;
            self.record_cached(
                lookup.to_owned_key(),
                ResidentStorage::Arena(span),
                upload_bytes,
                &tensor.data,
            );
            return Ok(WeightLease::Cached(span));
        }

        if !self.evict_standalone_cache_for_budget(upload_bytes, protected) {
            return self.upload_standalone_transient(upload_u64);
        }

        let buf = self.device.create_storage_buffer(upload_u64)?;
        if let Err(err) = self.device.write_buffer_u32(buf, &self.repack_scratch) {
            self.device.free_buffer(buf);
            return Err(err);
        }
        self.record_cached(
            lookup.to_owned_key(),
            ResidentStorage::Standalone(buf),
            upload_bytes,
            &tensor.data,
        );
        Ok(WeightLease::Cached(BufferSpan::whole(buf)))
    }

    fn evict_standalone_cache_for_budget(
        &mut self,
        upload_bytes: usize,
        protected: &[BufferId],
    ) -> bool {
        evict_standalone_cache_entries(
            &mut self.weight_cache,
            &mut self.cached_bytes,
            self.max_cached_weight_bytes,
            upload_bytes,
            protected,
            |id| self.device.free_buffer(id),
        )
    }

    fn upload_standalone_transient(&mut self, upload_u64: u64) -> Result<WeightLease, Error> {
        let buf = self.device.create_storage_buffer(upload_u64)?;
        if let Err(err) = self.device.write_buffer_u32(buf, &self.repack_scratch) {
            self.device.free_buffer(buf);
            return Err(err);
        }
        Ok(WeightLease::Transient(buf))
    }

    fn record_cached(
        &mut self,
        key: WeightCacheKey,
        storage: ResidentStorage,
        bytes: usize,
        source: &TensorBytes,
    ) {
        record_cached_entry(
            &mut self.weight_cache,
            &mut self.cached_bytes,
            &mut self.cache_access_counter,
            key,
            storage,
            bytes,
            source,
            |id| self.device.free_buffer(id),
        );
    }

    fn acquire_three_weights(
        &mut self,
        t0: &Tensor,
        t1: Option<&Tensor>,
        t2: Option<&Tensor>,
    ) -> Result<[Option<WeightLease>; 3], Error> {
        let ((c0, r0), dims1, dims2) = validate_three_pq2_matrices(t0, t1, t2)?;

        let l0 = self.acquire_weight_buffer(t0, c0, r0, &[])?;
        let mut leases = [Some(l0), None, None];
        if let (Some(t1), Some((c1, r1))) = (t1, dims1) {
            let prot = [l0.buffer_id()];
            match self.acquire_weight_buffer(t1, c1, r1, &prot) {
                Ok(l1) => leases[1] = Some(l1),
                Err(err) => {
                    self.release_transient_lease(l0);
                    return Err(err);
                }
            }
        }
        if let (Some(t2), Some((c2, r2))) = (t2, dims2) {
            let l1_buf = leases[1].map_or_else(|| l0.buffer_id(), WeightLease::buffer_id);
            let prot = [l0.buffer_id(), l1_buf];
            match self.acquire_weight_buffer(t2, c2, r2, &prot) {
                Ok(l2) => leases[2] = Some(l2),
                Err(err) => {
                    self.release_transient_lease(l0);
                    if let Some(l1) = leases[1] {
                        self.release_transient_lease(l1);
                    }
                    return Err(err);
                }
            }
        }
        Ok(leases)
    }

    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "Centralizes shared-input multi-token F32 projection group submission on Vulkan."
    )]
    fn execute_f32_group_batched(
        &mut self,
        tensors: &[&Tensor],
        flat_activations: &[f32],
        tokens: usize,
        pinned: bool,
        cols: usize,
        read_into: &mut [&mut Vec<f32>],
    ) -> Result<(), Error> {
        self.ensure_healthy()?;
        ensure_finite_f32(
            flat_activations,
            "GPU F32 projection input contains non-finite value",
        )?;
        let t_u32 = u32::try_from(tokens)
            .map_err(|_| Error::ResourceLimit("token count exceeds u32".to_owned()))?;
        let cols_u32 = u32::try_from(cols)
            .map_err(|_| Error::ResourceLimit("column count exceeds u32".to_owned()))?;
        let blocks_per_row = cols / 128;
        let blocks_u32 = u32::try_from(blocks_per_row)
            .map_err(|_| Error::ResourceLimit("block count exceeds u32".to_owned()))?;
        let act_words_per_token = u32::try_from(blocks_per_row * Q8K_4PLANE_WORDS_PER_HALF)
            .map_err(|_| Error::ResourceLimit("activation words exceed u32".to_owned()))?;

        let in_bytes = f32_slice_bytes(flat_activations.len())?;
        let q8k_bytes = u32_slice_bytes(tokens * blocks_per_row * Q8K_4PLANE_WORDS_PER_HALF)?;
        let f32_in_buf = ensure_scratch(&mut self.device, &mut self.scratch.f32_in, in_bytes)?;
        let q8k_buf = ensure_scratch(&mut self.device, &mut self.scratch.q8k_4p, q8k_bytes)?;
        self.device.write_buffer_f32(f32_in_buf, flat_activations)?;

        let mut rows_list = [0_usize; 3];
        let mut out_bufs = [BufferSpan::whole(f32_in_buf); 3];
        let out_slots = [
            f32_slice_bytes(tensors[0].dimensions[1] * tokens)?,
            tensors
                .get(1)
                .map_or(Ok(0), |t| f32_slice_bytes(t.dimensions[1] * tokens))?,
            tensors
                .get(2)
                .map_or(Ok(0), |t| f32_slice_bytes(t.dimensions[1] * tokens))?,
        ];
        let b0 = ensure_scratch(&mut self.device, &mut self.scratch.out_a, out_slots[0])?;
        out_bufs[0] = BufferSpan::whole(b0);
        if tensors.len() >= 2 {
            let b1 = ensure_scratch(&mut self.device, &mut self.scratch.out_b, out_slots[1])?;
            out_bufs[1] = BufferSpan::whole(b1);
        }
        if tensors.len() >= 3 {
            let b2 = ensure_scratch(&mut self.device, &mut self.scratch.out_c, out_slots[2])?;
            out_bufs[2] = BufferSpan::whole(b2);
        }

        let leases = self.acquire_three_weights(
            tensors[0],
            tensors.get(1).copied(),
            tensors.get(2).copied(),
        )?;
        let run_result = (|| -> Result<(), Error> {
            let quant_pipe = if pinned {
                self.pipes.quant_q8k_pinned
            } else {
                self.pipes.quant_q8k_unpinned
            };
            let quant_push = [cols_u32, t_u32];
            let quant_grid = checked_grid(&self.device, [cols_u32 / 256, t_u32, 1])?;
            let quant_spans = [BufferSpan::whole(f32_in_buf), BufferSpan::whole(q8k_buf)];

            let mut gemm_bufs = [[BufferSpan::whole(q8k_buf); 3]; 3];
            let mut gemm_pushes = [[0_u32; 4]; 3];
            let mut gemm_grids = [[1_u32; 3]; 3];
            let mut gemm_pipes = [quant_pipe; 3];
            let path = BatchTokenPath::for_tokens(&self.device, tokens);
            let gemm_pipe = self.pipes.select_pq2_q8k(pinned, path, false);

            for (idx, &tensor) in tensors.iter().enumerate() {
                let rows = tensor.dimensions[1];
                rows_list[idx] = rows;
                let rows_u32 = u32::try_from(rows)
                    .map_err(|_| Error::ResourceLimit("row count exceeds u32".to_owned()))?;
                let lease = leases[idx]
                    .ok_or_else(|| Error::InvalidInput("missing weight lease".to_owned()))?;
                gemm_pipes[idx] = gemm_pipe;
                gemm_bufs[idx] = [lease.span(), BufferSpan::whole(q8k_buf), out_bufs[idx]];
                gemm_pushes[idx] = [rows_u32, blocks_u32, t_u32, act_words_per_token];
                gemm_grids[idx] =
                    checked_grid(&self.device, path.gemm_grid(rows_u32.div_ceil(64), t_u32))?;
            }

            let mut stages = [ComputeSpanStage {
                pipeline: quant_pipe,
                buffers: &quant_spans,
                push_u32s: &quant_push,
                workgroups: quant_grid,
            }; 4];
            for idx in 0..tensors.len() {
                stages[1 + idx] = ComputeSpanStage {
                    pipeline: gemm_pipes[idx],
                    buffers: &gemm_bufs[idx],
                    push_u32s: &gemm_pushes[idx],
                    workgroups: gemm_grids[idx],
                };
            }
            self.device
                .submit_compute_span_stages(&stages[..=tensors.len()])?;

            for (idx, dst) in read_into.iter_mut().enumerate() {
                let count = tokens * rows_list[idx];
                dst.resize(count, 0.0_f32);
                self.device.read_buffer_f32(out_bufs[idx].buffer, dst)?;
                ensure_finite_f32(dst, "GPU group projection produced non-finite output")?;
            }
            Ok(())
        })();

        for l in leases.into_iter().flatten() {
            self.release_transient_lease(l);
        }
        run_result
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Executes complete FFN SwiGLU scratch setup and 4-stage Vulkan submission together."
    )]
    fn execute_ffn_swiglu_batched(
        &mut self,
        gate: &Tensor,
        up: &Tensor,
        flat_activations: &[f32],
        tokens: usize,
        pinned: bool,
        flat_output: &mut Vec<f32>,
    ) -> Result<(), Error> {
        self.ensure_healthy()?;
        ensure_finite_f32(
            flat_activations,
            "GPU FFN SwiGLU input contains non-finite value",
        )?;
        let (cols, inter_rows) = validate_matrix(gate, KIND_PQ2, 256)?;
        let t_u32 = u32::try_from(tokens)
            .map_err(|_| Error::ResourceLimit("token count exceeds u32".to_owned()))?;
        let cols_u32 = u32::try_from(cols)
            .map_err(|_| Error::ResourceLimit("column count exceeds u32".to_owned()))?;
        let rows_u32 = u32::try_from(inter_rows)
            .map_err(|_| Error::ResourceLimit("row count exceeds u32".to_owned()))?;
        let blocks_per_row = cols / 128;
        let blocks_u32 = u32::try_from(blocks_per_row)
            .map_err(|_| Error::ResourceLimit("block count exceeds u32".to_owned()))?;
        let act_words_per_token = u32::try_from(blocks_per_row * Q8K_4PLANE_WORDS_PER_HALF)
            .map_err(|_| Error::ResourceLimit("activation words exceed u32".to_owned()))?;
        let mid_elems = tokens * inter_rows;
        let mid_elems_u32 = u32::try_from(mid_elems)
            .map_err(|_| Error::ResourceLimit("SwiGLU count exceeds u32".to_owned()))?;

        let f32_in_buf = ensure_scratch(
            &mut self.device,
            &mut self.scratch.f32_in,
            f32_slice_bytes(flat_activations.len())?,
        )?;
        let q8k_buf = ensure_scratch(
            &mut self.device,
            &mut self.scratch.q8k_4p,
            u32_slice_bytes(tokens * blocks_per_row * Q8K_4PLANE_WORDS_PER_HALF)?,
        )?;
        let out_a = ensure_scratch(
            &mut self.device,
            &mut self.scratch.out_a,
            f32_slice_bytes(mid_elems)?,
        )?;
        let out_b = ensure_scratch(
            &mut self.device,
            &mut self.scratch.out_b,
            f32_slice_bytes(mid_elems)?,
        )?;
        let mid_buf = ensure_scratch(
            &mut self.device,
            &mut self.scratch.mid,
            f32_slice_bytes(mid_elems)?,
        )?;
        self.device.write_buffer_f32(f32_in_buf, flat_activations)?;

        let leases = self.acquire_three_weights(gate, Some(up), None)?;
        let w_gate =
            leases[0].ok_or_else(|| Error::InvalidInput("missing gate lease".to_owned()))?;
        let w_up = leases[1].ok_or_else(|| Error::InvalidInput("missing up lease".to_owned()))?;
        let run_result = (|| -> Result<(), Error> {
            let quant_pipe = if pinned {
                self.pipes.quant_q8k_pinned
            } else {
                self.pipes.quant_q8k_unpinned
            };
            let swiglu_pipe = if pinned {
                self.pipes.swiglu_pinned
            } else {
                self.pipes.swiglu_unpinned
            };
            let path = BatchTokenPath::for_tokens(&self.device, tokens);
            let gemm_pipe = self.pipes.select_pq2_q8k(pinned, path, false);
            let quant_spans = [BufferSpan::whole(f32_in_buf), BufferSpan::whole(q8k_buf)];
            let gate_spans = [
                w_gate.span(),
                BufferSpan::whole(q8k_buf),
                BufferSpan::whole(out_a),
            ];
            let up_spans = [
                w_up.span(),
                BufferSpan::whole(q8k_buf),
                BufferSpan::whole(out_b),
            ];
            let swiglu_spans = [
                BufferSpan::whole(out_a),
                BufferSpan::whole(out_b),
                BufferSpan::whole(mid_buf),
            ];
            let gemm_push = [rows_u32, blocks_u32, t_u32, act_words_per_token];
            let gemm_grid =
                checked_grid(&self.device, path.gemm_grid(rows_u32.div_ceil(64), t_u32))?;
            let quant_push = [cols_u32, t_u32];
            let swiglu_push = [mid_elems_u32];
            let stages = [
                ComputeSpanStage {
                    pipeline: quant_pipe,
                    buffers: &quant_spans,
                    push_u32s: &quant_push,
                    workgroups: checked_grid(&self.device, [cols_u32 / 256, t_u32, 1])?,
                },
                ComputeSpanStage {
                    pipeline: gemm_pipe,
                    buffers: &gate_spans,
                    push_u32s: &gemm_push,
                    workgroups: gemm_grid,
                },
                ComputeSpanStage {
                    pipeline: gemm_pipe,
                    buffers: &up_spans,
                    push_u32s: &gemm_push,
                    workgroups: gemm_grid,
                },
                ComputeSpanStage {
                    pipeline: swiglu_pipe,
                    buffers: &swiglu_spans,
                    push_u32s: &swiglu_push,
                    workgroups: checked_grid(&self.device, [mid_elems_u32.div_ceil(64), 1, 1])?,
                },
            ];
            self.device.submit_compute_span_stages(&stages)?;
            flat_output.resize(mid_elems, 0.0_f32);
            self.device.read_buffer_f32(mid_buf, flat_output)?;
            ensure_finite_f32(flat_output, "GPU FFN SwiGLU produced non-finite output")
        })();
        self.release_transient_leases(&[w_gate, w_up]);
        run_result
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "Executes complete fused FFN chain (with optional Hadamard, residual, and LoRA A dot) in one Vulkan submission."
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "Validates inputs, prepares scratch buffers, acquires leases, and dispatches the fused FFN chain."
    )]
    fn execute_ffn_chain_batched(
        &mut self,
        tensors: [&Tensor; 3],
        flat_activations: &[f32],
        flat_residuals: Option<&[f32]>,
        hadamard_block_size: Option<u32>,
        lora_a: Option<&[f32]>,
        tokens: usize,
        pinned: bool,
        flat_output: &mut Vec<f32>,
    ) -> Result<f32, Error> {
        self.ensure_healthy()?;
        ensure_finite_f32(
            flat_activations,
            "GPU FFN chain input contains non-finite value",
        )?;
        if let Some(res) = flat_residuals {
            ensure_finite_f32(res, "GPU FFN chain residual contains non-finite value")?;
        }
        if let Some(la) = lora_a {
            ensure_finite_f32(la, "GPU FFN chain LoRA A contains non-finite value")?;
        }
        let [gate, up, down] = tensors;
        let (in_cols, inter_rows) = validate_matrix(gate, KIND_PQ2, 256)?;
        let hadamard_blk = parse_hadamard_block(hadamard_block_size, inter_rows)?;
        let out_total = tokens * in_cols;
        let mid_total = tokens * inter_rows;
        let in_blocks = in_cols / 128;
        let mid_blocks = inter_rows / 128;

        let f32_in_buf = ensure_scratch(
            &mut self.device,
            &mut self.scratch.f32_in,
            f32_slice_bytes(flat_activations.len())?,
        )?;
        let q8k_in_buf = ensure_scratch(
            &mut self.device,
            &mut self.scratch.q8k_4p,
            u32_slice_bytes(tokens * in_blocks * Q8K_4PLANE_WORDS_PER_HALF)?,
        )?;
        let out_a = ensure_scratch(
            &mut self.device,
            &mut self.scratch.out_a,
            f32_slice_bytes(mid_total)?,
        )?;
        let out_b = ensure_scratch(
            &mut self.device,
            &mut self.scratch.out_b,
            f32_slice_bytes(mid_total)?,
        )?;
        let mid_buf = ensure_scratch(
            &mut self.device,
            &mut self.scratch.mid,
            f32_slice_bytes(mid_total)?,
        )?;
        let mid_rot_buf = ensure_scratch(
            &mut self.device,
            &mut self.scratch.mid_rot,
            f32_slice_bytes(mid_total)?,
        )?;
        let q8k_mid_buf = ensure_scratch(
            &mut self.device,
            &mut self.scratch.q8k_mid,
            u32_slice_bytes(tokens * mid_blocks * Q8K_4PLANE_WORDS_PER_HALF)?,
        )?;
        let out_c = ensure_scratch(
            &mut self.device,
            &mut self.scratch.out_c,
            f32_slice_bytes(out_total)?,
        )?;
        self.device.write_buffer_f32(f32_in_buf, flat_activations)?;

        let res_buf = if let Some(res) = flat_residuals {
            let b = ensure_scratch(
                &mut self.device,
                &mut self.scratch.residual,
                f32_slice_bytes(out_total)?,
            )?;
            self.device.write_buffer_f32(b, res)?;
            Some(b)
        } else {
            None
        };
        let lora_bufs = if let Some(la) = lora_a {
            let b = ensure_scratch(
                &mut self.device,
                &mut self.scratch.lora_a,
                f32_slice_bytes(inter_rows)?,
            )?;
            let s = ensure_scratch(&mut self.device, &mut self.scratch.lora_scalar, 4)?;
            self.device.write_buffer_f32(b, la)?;
            Some((b, s))
        } else {
            None
        };

        let leases = self.acquire_three_weights(gate, Some(up), Some(down))?;
        let w_gate =
            leases[0].ok_or_else(|| Error::InvalidInput("missing gate lease".to_owned()))?;
        let w_up = leases[1].ok_or_else(|| Error::InvalidInput("missing up lease".to_owned()))?;
        let w_down =
            leases[2].ok_or_else(|| Error::InvalidInput("missing down lease".to_owned()))?;

        let bufs = FfnChainScratchIds {
            f32_in: f32_in_buf,
            q8k_in: q8k_in_buf,
            out_a,
            out_b,
            mid: mid_buf,
            mid_rot: mid_rot_buf,
            q8k_mid: q8k_mid_buf,
            out_c,
        };
        let run_result = self.submit_ffn_chain_stages(
            bufs,
            w_gate,
            w_up,
            w_down,
            in_cols,
            inter_rows,
            tokens,
            pinned,
            hadamard_blk,
            res_buf,
            lora_bufs,
            flat_output,
        );
        self.release_transient_leases(&[w_gate, w_up, w_down]);
        run_result
    }

    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "Encodes and submits the 6-8 stage FFN chain command buffer with a single fence wait."
    )]
    fn submit_ffn_chain_stages(
        &mut self,
        bufs: FfnChainScratchIds,
        w_gate: WeightLease,
        w_up: WeightLease,
        w_down: WeightLease,
        in_cols: usize,
        inter_rows: usize,
        tokens: usize,
        pinned: bool,
        hadamard_blk: Option<u32>,
        res_buf: Option<BufferId>,
        lora_bufs: Option<(BufferId, BufferId)>,
        flat_output: &mut Vec<f32>,
    ) -> Result<f32, Error> {
        let t_u32 = u32::try_from(tokens)
            .map_err(|_| Error::ResourceLimit("token count exceeds u32".to_owned()))?;
        let in_cols_u32 = u32::try_from(in_cols)
            .map_err(|_| Error::ResourceLimit("in_cols exceeds u32".to_owned()))?;
        let inter_u32 = u32::try_from(inter_rows)
            .map_err(|_| Error::ResourceLimit("inter_rows exceeds u32".to_owned()))?;
        let in_blocks_u32 = u32::try_from(in_cols / 128)
            .map_err(|_| Error::ResourceLimit("in_blocks exceeds u32".to_owned()))?;
        let mid_blocks_u32 = u32::try_from(inter_rows / 128)
            .map_err(|_| Error::ResourceLimit("mid_blocks exceeds u32".to_owned()))?;
        let in_act_words = u32::try_from((in_cols / 128) * Q8K_4PLANE_WORDS_PER_HALF)
            .map_err(|_| Error::ResourceLimit("in_act_words exceeds u32".to_owned()))?;
        let mid_act_words = u32::try_from((inter_rows / 128) * Q8K_4PLANE_WORDS_PER_HALF)
            .map_err(|_| Error::ResourceLimit("mid_act_words exceeds u32".to_owned()))?;
        let mid_total = tokens * inter_rows;
        let mid_total_u32 = u32::try_from(mid_total)
            .map_err(|_| Error::ResourceLimit("mid_total exceeds u32".to_owned()))?;

        let f32_in_span = BufferSpan::whole(bufs.f32_in);
        let q8k_in_span = BufferSpan::whole(bufs.q8k_in);
        let gate_out_span = BufferSpan::whole(bufs.out_a);
        let up_out_span = BufferSpan::whole(bufs.out_b);
        let mid_span = BufferSpan::whole(bufs.mid);
        let mid_rot_span = BufferSpan::whole(bufs.mid_rot);
        let q8k_mid_span = BufferSpan::whole(bufs.q8k_mid);
        let out_c_buf = bufs.out_c;
        let final_out_span = BufferSpan::whole(out_c_buf);

        let quant_pipe = if pinned {
            self.pipes.quant_q8k_pinned
        } else {
            self.pipes.quant_q8k_unpinned
        };
        let swiglu_pipe = if pinned {
            self.pipes.swiglu_pinned
        } else {
            self.pipes.swiglu_unpinned
        };
        let path = BatchTokenPath::for_tokens(&self.device, tokens);
        let up_pipe =
            if should_use_decode_splitk(&self.device, tokens, in_cols, inter_rows, in_cols / 128) {
                self.pipes.select_pq2_q8k_decode_splitk(pinned, false)
            } else {
                self.pipes.select_pq2_q8k(pinned, path, false)
            };
        let down_pipe = if should_use_decode_splitk(
            &self.device,
            tokens,
            inter_rows,
            in_cols,
            inter_rows / 128,
        ) {
            self.pipes
                .select_pq2_q8k_decode_splitk(pinned, res_buf.is_some())
        } else {
            self.pipes.select_pq2_q8k(pinned, path, res_buf.is_some())
        };

        let quant_in_spans = [f32_in_span, q8k_in_span];
        let quant_in_push = [in_cols_u32, t_u32];
        let gate_spans = [w_gate.span(), q8k_in_span, gate_out_span];
        let up_spans = [w_up.span(), q8k_in_span, up_out_span];
        let up_push = [inter_u32, in_blocks_u32, t_u32, in_act_words];
        let up_grid = checked_grid(&self.device, path.gemm_grid(inter_u32.div_ceil(64), t_u32))?;
        let swiglu_spans = [gate_out_span, up_out_span, mid_span];
        let swiglu_push = [mid_total_u32];

        let mut lora_spans = [mid_span; 3];
        let lora_push = [inter_u32];
        if let Some((la, ls)) = lora_bufs {
            lora_spans = [mid_span, BufferSpan::whole(la), BufferSpan::whole(ls)];
        }

        let hadamard_spans = [mid_span, mid_rot_span];
        let hadamard_blk_u = hadamard_blk.unwrap_or(1);
        let hadamard_push = [hadamard_blk_u, hadamard_norm_bits(hadamard_blk_u)];
        let hadamard_grid = if hadamard_blk.is_some() {
            let blk_us = usize::try_from(hadamard_blk_u).unwrap_or(1);
            let count = u32::try_from(mid_total / blk_us)
                .map_err(|_| Error::ResourceLimit("hadamard grid exceeds u32".to_owned()))?;
            checked_grid(&self.device, [count, 1, 1])?
        } else {
            [1, 1, 1]
        };

        let mid_active_span = if hadamard_blk.is_some() {
            mid_rot_span
        } else {
            mid_span
        };
        let quant_mid_spans = [mid_active_span, q8k_mid_span];
        let quant_mid_push = [inter_u32, t_u32];

        let down_spans_3 = [w_down.span(), q8k_mid_span, final_out_span];
        let mut down_spans_4 = [w_down.span(), q8k_mid_span, final_out_span, final_out_span];
        let down_spans: &[BufferSpan] = res_buf.map_or(&down_spans_3, |rb| {
            down_spans_4[3] = BufferSpan::whole(rb);
            &down_spans_4
        });
        let down_push = [in_cols_u32, mid_blocks_u32, t_u32, mid_act_words];
        let down_grid = checked_grid(
            &self.device,
            path.gemm_grid(in_cols_u32.div_ceil(64), t_u32),
        )?;

        let mut stages: Vec<ComputeSpanStage<'_>> = Vec::with_capacity(8);
        stages.push(ComputeSpanStage {
            pipeline: quant_pipe,
            buffers: &quant_in_spans,
            push_u32s: &quant_in_push,
            workgroups: checked_grid(&self.device, [in_cols_u32 / 256, t_u32, 1])?,
        });
        stages.push(ComputeSpanStage {
            pipeline: up_pipe,
            buffers: &gate_spans,
            push_u32s: &up_push,
            workgroups: up_grid,
        });
        stages.push(ComputeSpanStage {
            pipeline: up_pipe,
            buffers: &up_spans,
            push_u32s: &up_push,
            workgroups: up_grid,
        });
        stages.push(ComputeSpanStage {
            pipeline: swiglu_pipe,
            buffers: &swiglu_spans,
            push_u32s: &swiglu_push,
            workgroups: checked_grid(&self.device, [mid_total_u32.div_ceil(64), 1, 1])?,
        });
        if lora_bufs.is_some() {
            stages.push(ComputeSpanStage {
                pipeline: self.pipes.dot_scalar,
                buffers: &lora_spans,
                push_u32s: &lora_push,
                workgroups: [1, 1, 1],
            });
        }
        if hadamard_blk.is_some() {
            stages.push(ComputeSpanStage {
                pipeline: self.pipes.hadamard,
                buffers: &hadamard_spans,
                push_u32s: &hadamard_push,
                workgroups: hadamard_grid,
            });
        }
        stages.push(ComputeSpanStage {
            pipeline: quant_pipe,
            buffers: &quant_mid_spans,
            push_u32s: &quant_mid_push,
            workgroups: checked_grid(&self.device, [inter_u32 / 256, t_u32, 1])?,
        });
        stages.push(ComputeSpanStage {
            pipeline: down_pipe,
            buffers: down_spans,
            push_u32s: &down_push,
            workgroups: down_grid,
        });

        self.device.submit_compute_span_stages(&stages)?;

        flat_output.resize(tokens * in_cols, 0.0_f32);
        self.device.read_buffer_f32(out_c_buf, flat_output)?;
        ensure_finite_f32(flat_output, "GPU FFN chain produced non-finite output")?;

        if let Some((_, ls)) = lora_bufs {
            let mut scalar = [0.0_f32; 1];
            self.device.read_buffer_f32(ls, &mut scalar)?;
            if !scalar[0].is_finite() {
                return Err(Error::InvalidInput(
                    "GPU FFN chain LoRA dot produced non-finite scalar".to_owned(),
                ));
            }
            Ok(scalar[0])
        } else {
            Ok(0.0)
        }
    }
}

impl Drop for VulkanPq2Executor {
    fn drop(&mut self) {
        self.scratch.free_all(&mut self.device);
        let _ = self.release_resident_weights();
    }
}

fn validate_ffn_chain_tensors(
    gate: &Tensor,
    up: &Tensor,
    down: &Tensor,
    require_256: usize,
) -> Result<(usize, usize), Error> {
    let (gate_cols, gate_rows) = validate_matrix(gate, KIND_PQ2, require_256)?;
    let (up_cols, up_rows) = validate_matrix(up, KIND_PQ2, require_256)?;
    let (down_cols, down_rows) = validate_matrix(down, KIND_PQ2, require_256)?;
    if gate_cols != up_cols
        || gate_rows != up_rows
        || down_cols != gate_rows
        || down_rows != gate_cols
    {
        return Err(Error::InvalidInput(
            "mismatched FFN chain tensor dimensions".to_owned(),
        ));
    }
    Ok((gate_cols, gate_rows))
}

fn pack_flat_tokens(
    activations: &[Vec<f32>],
    cols: usize,
    dst: &mut Vec<f32>,
) -> Result<(), Error> {
    dst.clear();
    let total = activations
        .len()
        .checked_mul(cols)
        .ok_or_else(|| Error::ResourceLimit("token activation buffer overflow".to_owned()))?;
    dst.reserve(total);
    for act in activations {
        if act.len() != cols {
            return Err(Error::InvalidInput(format!(
                "F32 token activation expected {cols} elements, got {}",
                act.len()
            )));
        }
        dst.extend_from_slice(act);
    }
    Ok(())
}

fn scatter_flat_tokens(flat: &[f32], stride: usize, outputs: &mut [Vec<f32>]) {
    for (dst, chunk) in outputs.iter_mut().zip(flat.chunks_exact(stride)) {
        dst.clear();
        dst.extend_from_slice(chunk);
    }
}

impl Pq2ProjectionExecutor for VulkanPq2Executor {
    fn identity(&self) -> &str {
        &self.identity
    }

    fn operation_capacity(&self) -> Option<u64> {
        if self.is_faulted() {
            Some(0)
        } else {
            Some(DEFAULT_OPERATION_CAPACITY)
        }
    }

    fn operation_count(&self, tensor: &Tensor) -> Option<u64> {
        if !(self.supports(tensor) || self.supports_q8(tensor)) {
            return None;
        }
        let cols = u64::try_from(*tensor.dimensions.first()?).ok()?;
        let rows = u64::try_from(*tensor.dimensions.get(1)?).ok()?;
        cols.checked_mul(rows)
    }

    fn supports(&self, tensor: &Tensor) -> bool {
        !self.is_faulted() && validate_matrix(tensor, KIND_PQ2, 256).is_ok()
    }

    fn supports_tile_tokens(&self) -> bool {
        !self.is_faulted()
    }

    fn project(
        &mut self,
        tensor: &Tensor,
        activation: &[Q8KBlock],
        fused: bool,
        output: &mut Vec<f32>,
    ) -> bonsai_prism_runtime::Result<()> {
        output.clear();
        let (cols, rows) =
            validate_matrix(tensor, KIND_PQ2, 256).map_err(Error::into_runtime_error)?;
        let blocks_per_row = cols / 128;
        let expected_q8k = cols / 256;
        if activation.len() != expected_q8k {
            return Err(Error::InvalidInput(format!(
                "PQ2 x Q8K expected {expected_q8k} activation blocks, got {}",
                activation.len()
            ))
            .into_runtime_error());
        }
        pack_q8k_blocks_4plane_into(activation, &mut self.act_pack_scratch)
            .map_err(Error::into_runtime_error)?;
        let act_words_per_token = u32::try_from(self.act_pack_scratch.len()).map_err(|_| {
            Error::ResourceLimit("activation word count overflow".to_owned()).into_runtime_error()
        })?;
        let pipeline = self
            .pipes
            .select_pq2_q8k(fused, BatchTokenPath::Single, false);
        let words = std::mem::take(&mut self.act_pack_scratch);
        let res = self.dispatch_single_word_gemv(
            tensor,
            cols,
            rows,
            blocks_per_row,
            1,
            act_words_per_token,
            &words,
            pipeline,
            output,
        );
        self.act_pack_scratch = words;
        res.map_err(Error::into_runtime_error)
    }

    fn supports_q8(&self, tensor: &Tensor) -> bool {
        !self.is_faulted()
            && (validate_matrix(tensor, KIND_PQ2, 128).is_ok()
                || validate_matrix(tensor, KIND_PTQ1, 128).is_ok()
                || validate_matrix(tensor, KIND_Q8_0, 32).is_ok())
    }

    fn project_q8(
        &mut self,
        tensor: &Tensor,
        activation: &[Q8Block],
        pinned: bool,
        output: &mut Vec<f32>,
    ) -> bonsai_prism_runtime::Result<()> {
        output.clear();
        let (cols, rows) = match tensor.kind {
            KIND_PQ2 => validate_matrix(tensor, KIND_PQ2, 128),
            KIND_PTQ1 => validate_matrix(tensor, KIND_PTQ1, 128),
            KIND_Q8_0 => validate_matrix(tensor, KIND_Q8_0, 32),
            other => Err(Error::InvalidInput(format!(
                "unsupported Q8 projection tensor kind {other}"
            ))),
        }
        .map_err(Error::into_runtime_error)?;

        let blocks_per_row = if tensor.kind == KIND_Q8_0 {
            cols / 32
        } else {
            cols / 128
        };
        let expected_q8 = cols / 32;
        if activation.len() != expected_q8 {
            return Err(Error::InvalidInput(format!(
                "Q8 projection expected {expected_q8} activation blocks, got {}",
                activation.len()
            ))
            .into_runtime_error());
        }
        pack_q8_blocks_into(activation, &mut self.act_pack_scratch)
            .map_err(Error::into_runtime_error)?;
        let act_words_per_token = u32::try_from(self.act_pack_scratch.len()).map_err(|_| {
            Error::ResourceLimit("activation word count overflow".to_owned()).into_runtime_error()
        })?;
        let pipeline = match tensor.kind {
            KIND_PQ2 => self.pipes.pq2_q8_0,
            KIND_PTQ1 if pinned => self.pipes.ptq1_q8_0_pinned,
            KIND_PTQ1 => self.pipes.ptq1_q8_0,
            KIND_Q8_0 => self.pipes.q8_0_q8_0,
            other => {
                return Err(Error::InvalidInput(format!(
                    "unsupported Q8 projection tensor kind {other}"
                ))
                .into_runtime_error());
            }
        };
        let words = std::mem::take(&mut self.act_pack_scratch);
        let res = self.dispatch_single_word_gemv(
            tensor,
            cols,
            rows,
            blocks_per_row,
            1,
            act_words_per_token,
            &words,
            pipeline,
            output,
        );
        self.act_pack_scratch = words;
        res.map_err(Error::into_runtime_error)
    }

    fn supports_f32(&self, tensor: &Tensor) -> bool {
        self.supports(tensor)
    }

    fn project_f32(
        &mut self,
        tensor: &Tensor,
        activation: &[f32],
        pinned: bool,
        output: &mut Vec<f32>,
    ) -> bonsai_prism_runtime::Result<()> {
        output.clear();
        let (cols, _) =
            validate_matrix(tensor, KIND_PQ2, 256).map_err(Error::into_runtime_error)?;
        if activation.len() != cols {
            return Err(Error::InvalidInput(format!(
                "F32 projection expected {cols} inputs, got {}",
                activation.len()
            ))
            .into_runtime_error());
        }
        self.execute_f32_group_batched(&[tensor], activation, 1, pinned, cols, &mut [output])
            .map_err(Error::into_runtime_error)
    }

    fn supports_f32_group(&self, tensors: &[&Tensor]) -> bool {
        if self.is_faulted() || !(1..=3).contains(&tensors.len()) {
            return false;
        }
        let first_cols = tensors[0].dimensions.first().copied();
        tensors
            .iter()
            .all(|t| self.supports_f32(t) && t.dimensions.first().copied() == first_cols)
    }

    fn project_f32_group(
        &mut self,
        tensors: &[&Tensor],
        activation: &[f32],
        pinned: bool,
        outputs: &mut [&mut Vec<f32>],
    ) -> bonsai_prism_runtime::Result<()> {
        for out in outputs.iter_mut() {
            out.clear();
        }
        if tensors.len() != outputs.len() || !self.supports_f32_group(tensors) {
            return Err(
                Error::InvalidInput("unsupported F32 projection group".to_owned())
                    .into_runtime_error(),
            );
        }
        let cols = tensors[0].dimensions[0];
        if activation.len() != cols {
            return Err(Error::InvalidInput(format!(
                "F32 group projection expected {cols} inputs, got {}",
                activation.len()
            ))
            .into_runtime_error());
        }
        self.execute_f32_group_batched(tensors, activation, 1, pinned, cols, outputs)
            .map_err(Error::into_runtime_error)
    }

    fn project_f32_tokens(
        &mut self,
        tensor: &Tensor,
        activations: &[Vec<f32>],
        pinned: bool,
        outputs: &mut [Vec<f32>],
    ) -> bonsai_prism_runtime::Result<()> {
        if activations.len() != outputs.len() {
            return Err(
                Error::InvalidInput("mismatched F32 token output count".to_owned())
                    .into_runtime_error(),
            );
        }
        for out in outputs.iter_mut() {
            out.clear();
        }
        if activations.is_empty() {
            return Ok(());
        }
        let (cols, rows) =
            validate_matrix(tensor, KIND_PQ2, 256).map_err(Error::into_runtime_error)?;
        let mut flat_in = std::mem::take(&mut self.f32_pack_scratch);
        let mut flat_out = Vec::new();
        let res = (|| {
            pack_flat_tokens(activations, cols, &mut flat_in)?;
            self.execute_f32_group_batched(
                &[tensor],
                &flat_in,
                activations.len(),
                pinned,
                cols,
                &mut [&mut flat_out],
            )
        })();
        self.f32_pack_scratch = flat_in;
        res.map_err(Error::into_runtime_error)?;
        scatter_flat_tokens(&flat_out, rows, outputs);
        Ok(())
    }

    fn project_f32_pair_tokens(
        &mut self,
        tensors: [&Tensor; 2],
        activations: &[Vec<f32>],
        pinned: bool,
        outputs_a: &mut [Vec<f32>],
        outputs_b: &mut [Vec<f32>],
    ) -> bonsai_prism_runtime::Result<()> {
        if activations.len() != outputs_a.len()
            || activations.len() != outputs_b.len()
            || !self.supports_f32_group(&tensors)
        {
            return Err(
                Error::InvalidInput("invalid F32 pair token projection inputs".to_owned())
                    .into_runtime_error(),
            );
        }
        for out in outputs_a.iter_mut().chain(outputs_b.iter_mut()) {
            out.clear();
        }
        if activations.is_empty() {
            return Ok(());
        }
        let cols = tensors[0].dimensions[0];
        let mut flat_in = std::mem::take(&mut self.f32_pack_scratch);
        let (mut flat_a, mut flat_b) = (Vec::new(), Vec::new());
        let res = (|| {
            pack_flat_tokens(activations, cols, &mut flat_in)?;
            self.execute_f32_group_batched(
                &tensors,
                &flat_in,
                activations.len(),
                pinned,
                cols,
                &mut [&mut flat_a, &mut flat_b],
            )
        })();
        self.f32_pack_scratch = flat_in;
        res.map_err(Error::into_runtime_error)?;
        scatter_flat_tokens(&flat_a, tensors[0].dimensions[1], outputs_a);
        scatter_flat_tokens(&flat_b, tensors[1].dimensions[1], outputs_b);
        Ok(())
    }

    fn project_f32_triple_tokens(
        &mut self,
        tensors: [&Tensor; 3],
        activations: &[Vec<f32>],
        pinned: bool,
        outputs_a: &mut [Vec<f32>],
        outputs_b: &mut [Vec<f32>],
        outputs_c: &mut [Vec<f32>],
    ) -> bonsai_prism_runtime::Result<()> {
        if activations.len() != outputs_a.len()
            || activations.len() != outputs_b.len()
            || activations.len() != outputs_c.len()
            || !self.supports_f32_group(&tensors)
        {
            return Err(Error::InvalidInput(
                "invalid F32 triple token projection inputs".to_owned(),
            )
            .into_runtime_error());
        }
        for out in outputs_a
            .iter_mut()
            .chain(outputs_b.iter_mut())
            .chain(outputs_c.iter_mut())
        {
            out.clear();
        }
        if activations.is_empty() {
            return Ok(());
        }
        let cols = tensors[0].dimensions[0];
        let mut flat_in = std::mem::take(&mut self.f32_pack_scratch);
        let (mut flat_a, mut flat_b, mut flat_c) = (Vec::new(), Vec::new(), Vec::new());
        let res = (|| {
            pack_flat_tokens(activations, cols, &mut flat_in)?;
            self.execute_f32_group_batched(
                &tensors,
                &flat_in,
                activations.len(),
                pinned,
                cols,
                &mut [&mut flat_a, &mut flat_b, &mut flat_c],
            )
        })();
        self.f32_pack_scratch = flat_in;
        res.map_err(Error::into_runtime_error)?;
        scatter_flat_tokens(&flat_a, tensors[0].dimensions[1], outputs_a);
        scatter_flat_tokens(&flat_b, tensors[1].dimensions[1], outputs_b);
        scatter_flat_tokens(&flat_c, tensors[2].dimensions[1], outputs_c);
        Ok(())
    }

    fn supports_f32_ffn_swiglu(&self, gate: &Tensor, up: &Tensor) -> bool {
        !self.is_faulted()
            && self.supports_f32(gate)
            && self.supports_f32(up)
            && gate.dimensions == up.dimensions
    }

    fn project_f32_ffn_swiglu(
        &mut self,
        gate: &Tensor,
        up: &Tensor,
        activation: &[f32],
        pinned: bool,
        output: &mut Vec<f32>,
    ) -> bonsai_prism_runtime::Result<()> {
        output.clear();
        if !self.supports_f32_ffn_swiglu(gate, up) || activation.len() != gate.dimensions[0] {
            return Err(
                Error::InvalidInput("unsupported FFN SwiGLU inputs".to_owned())
                    .into_runtime_error(),
            );
        }
        self.execute_ffn_swiglu_batched(gate, up, activation, 1, pinned, output)
            .map_err(Error::into_runtime_error)
    }

    fn project_f32_ffn_swiglu_tokens(
        &mut self,
        gate: &Tensor,
        up: &Tensor,
        activations: &[Vec<f32>],
        pinned: bool,
        outputs: &mut [Vec<f32>],
    ) -> bonsai_prism_runtime::Result<()> {
        if activations.len() != outputs.len() || !self.supports_f32_ffn_swiglu(gate, up) {
            return Err(Error::InvalidInput(
                "invalid FFN SwiGLU token projection inputs".to_owned(),
            )
            .into_runtime_error());
        }
        for out in outputs.iter_mut() {
            out.clear();
        }
        if activations.is_empty() {
            return Ok(());
        }
        let cols = gate.dimensions[0];
        let rows = gate.dimensions[1];
        let mut flat_in = std::mem::take(&mut self.f32_pack_scratch);
        let mut flat_out = Vec::new();
        let res = (|| {
            pack_flat_tokens(activations, cols, &mut flat_in)?;
            self.execute_ffn_swiglu_batched(
                gate,
                up,
                &flat_in,
                activations.len(),
                pinned,
                &mut flat_out,
            )
        })();
        self.f32_pack_scratch = flat_in;
        res.map_err(Error::into_runtime_error)?;
        scatter_flat_tokens(&flat_out, rows, outputs);
        Ok(())
    }

    fn supports_f32_ffn_chain(&self, [gate, up, down]: [&Tensor; 3]) -> bool {
        !self.is_faulted() && validate_ffn_chain_tensors(gate, up, down, 256).is_ok()
    }

    fn project_f32_ffn_chain(
        &mut self,
        tensors: [&Tensor; 3],
        activation: &[f32],
        residual: Option<&[f32]>,
        hadamard_block_size: Option<u32>,
        pinned: bool,
        output: &mut Vec<f32>,
    ) -> bonsai_prism_runtime::Result<()> {
        output.clear();
        let [gate, up, down] = tensors;
        let (in_cols, _) =
            validate_ffn_chain_tensors(gate, up, down, 256).map_err(Error::into_runtime_error)?;
        if activation.len() != in_cols || residual.is_some_and(|r| r.len() != in_cols) {
            return Err(
                Error::InvalidInput("FFN chain input/residual length mismatch".to_owned())
                    .into_runtime_error(),
            );
        }
        let _ = self
            .execute_ffn_chain_batched(
                tensors,
                activation,
                residual,
                hadamard_block_size,
                None,
                1,
                pinned,
                output,
            )
            .map_err(Error::into_runtime_error)?;
        Ok(())
    }

    fn project_f32_ffn_chain_tokens(
        &mut self,
        tensors: [&Tensor; 3],
        activations: &[Vec<f32>],
        residuals: Option<&[Vec<f32>]>,
        hadamard_block_size: Option<u32>,
        pinned: bool,
        outputs: &mut [Vec<f32>],
    ) -> bonsai_prism_runtime::Result<()> {
        if activations.len() != outputs.len()
            || residuals.is_some_and(|r| r.len() != activations.len())
        {
            return Err(
                Error::InvalidInput("mismatched FFN chain token slices".to_owned())
                    .into_runtime_error(),
            );
        }
        for out in outputs.iter_mut() {
            out.clear();
        }
        if activations.is_empty() {
            return Ok(());
        }
        let [gate, up, down] = tensors;
        let (in_cols, _) =
            validate_ffn_chain_tensors(gate, up, down, 256).map_err(Error::into_runtime_error)?;
        let mut flat_in = std::mem::take(&mut self.f32_pack_scratch);
        let mut flat_res = Vec::new();
        let mut flat_out = Vec::new();
        let res = (|| {
            pack_flat_tokens(activations, in_cols, &mut flat_in)?;
            let res_slice = if let Some(res_vecs) = residuals {
                pack_flat_tokens(res_vecs, in_cols, &mut flat_res)?;
                Some(flat_res.as_slice())
            } else {
                None
            };
            let _ = self.execute_ffn_chain_batched(
                tensors,
                &flat_in,
                res_slice,
                hadamard_block_size,
                None,
                activations.len(),
                pinned,
                &mut flat_out,
            )?;
            Ok(())
        })();
        self.f32_pack_scratch = flat_in;
        res.map_err(Error::into_runtime_error)?;
        scatter_flat_tokens(&flat_out, in_cols, outputs);
        Ok(())
    }

    fn supports_f32_ffn_chain_lora(&self, tensors: [&Tensor; 3]) -> bool {
        self.supports_f32_ffn_chain(tensors)
    }

    fn project_f32_ffn_chain_lora(
        &mut self,
        tensors: [&Tensor; 3],
        activation: &[f32],
        residual: Option<&[f32]>,
        hadamard_block_size: Option<u32>,
        lora_a: &[f32],
        pinned: bool,
        output: &mut Vec<f32>,
    ) -> bonsai_prism_runtime::Result<f32> {
        output.clear();
        let [gate, up, down] = tensors;
        let (in_cols, inter_rows) =
            validate_ffn_chain_tensors(gate, up, down, 256).map_err(Error::into_runtime_error)?;
        if activation.len() != in_cols
            || residual.is_some_and(|r| r.len() != in_cols)
            || lora_a.len() != inter_rows
        {
            return Err(Error::InvalidInput(
                "FFN chain LoRA input/residual/lora_a length mismatch".to_owned(),
            )
            .into_runtime_error());
        }
        self.execute_ffn_chain_batched(
            tensors,
            activation,
            residual,
            hadamard_block_size,
            Some(lora_a),
            1,
            pinned,
            output,
        )
        .map_err(Error::into_runtime_error)
    }
}

/// Adreno Vulkan attention executor (`#40`) implementing fused and 3-stage decomposed kernels
/// with explicit pipeline barriers, binary16 probability rounding, cancellation, and scratch reuse.
pub struct VulkanAttentionExecutor {
    device: AdrenoVulkanDevice,
    scratch_budget_bytes: u64,
    pipe_qk: PipelineId,
    pipe_softmax: PipelineId,
    pipe_pv: PipelineId,
    pipe_fused: PipelineId,
    qk_digest: String,
    softmax_digest: String,
    pv_digest: String,
    fused_digest: String,
    scratch: AttentionScratchSet,
}

impl VulkanAttentionExecutor {
    /// Opens the native Adreno Vulkan device and compiles `#40` fused & decomposed pipelines.
    ///
    /// # Errors
    /// Propagates loader, device selection, or SPIR-V compilation errors.
    pub fn open(scratch_budget_bytes: u64) -> Result<Self, Error> {
        let mut device = AdrenoVulkanDevice::open()?;
        let use_wave64 = device.capabilities().wave64_subgroup_shuffle;
        let qk_spv = spirv::attn_qk_scores();
        let softmax_spv = spirv::attn_softmax_f16_with_subgroup(use_wave64);
        let pv_spv = spirv::attn_pv_accum();
        let fused_spv = spirv::attn_fused_f16prob_with_subgroup(use_wave64);

        let qk_digest = spirv::shader_digest(&qk_spv)?;
        let softmax_digest = spirv::shader_digest(&softmax_spv)?;
        let pv_digest = spirv::shader_digest(&pv_spv)?;
        let fused_digest = spirv::shader_digest(&fused_spv)?;

        let pipe_qk = device.create_compute_pipeline(&qk_spv, 3, 8)?;
        let pipe_softmax = device.create_compute_pipeline(&softmax_spv, 2, 8)?;
        let pipe_pv = device.create_compute_pipeline(&pv_spv, 3, 8)?;
        let pipe_fused = device.create_compute_pipeline(&fused_spv, 4, 8)?;

        Ok(Self {
            device,
            scratch_budget_bytes,
            pipe_qk,
            pipe_softmax,
            pipe_pv,
            pipe_fused,
            qk_digest,
            softmax_digest,
            pv_digest,
            fused_digest,
            scratch: AttentionScratchSet::default(),
        })
    }

    /// Returns the discovered Qualcomm Adreno capability snapshot.
    #[must_use]
    pub const fn capabilities(&self) -> &VulkanDeviceCapabilities {
        self.device.capabilities()
    }

    /// Executes `#40` attention using [`select_attention_plan`], reusing persistent scratch buffers.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`], [`Error::ResourceLimit`], [`Error::Cancelled`],
    /// [`Error::DeviceFault`], or [`Error::Vulkan`].
    #[expect(
        clippy::too_many_arguments,
        reason = "Explicitly separates Q/K/V slices, descriptor, strategy, witness capture, and cancellation hook."
    )]
    pub fn execute<F>(
        &mut self,
        desc: &AttentionDescriptor,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        strategy: AttentionStrategy,
        capture_stages: bool,
        mut check_cancelled: F,
    ) -> Result<AttentionOutput, Error>
    where
        F: FnMut() -> bool,
    {
        let (q_elems, _kv_elems, score_elems) = validate_attention_inputs(desc, q, k, v)?;
        let plan = select_attention_plan(
            desc,
            self.device.capabilities(),
            strategy,
            self.scratch_budget_bytes,
        )?;
        let cache_key = attention_cache_key(
            desc,
            &plan,
            self.device.capabilities(),
            &self.qk_digest,
            &self.softmax_digest,
            &self.pv_digest,
            &self.fused_digest,
        )?;
        if check_cancelled() {
            return Err(Error::Cancelled);
        }

        let q_buf = upload_f32_scratch(&mut self.device, &mut self.scratch.q, q)?;
        let k_buf = upload_f32_scratch(&mut self.device, &mut self.scratch.k, k)?;
        let v_buf = upload_f32_scratch(&mut self.device, &mut self.scratch.v, v)?;
        let out_bytes = f32_slice_bytes(q_elems)?;
        let out_buf = ensure_scratch(&mut self.device, &mut self.scratch.out, out_bytes)?;
        let push = attention_push_constants(desc);

        match plan.path {
            AdrenoAttentionPath::Fused => self.execute_fused(
                plan, cache_key, q_buf, k_buf, v_buf, out_buf, &push, q_elems,
            ),
            AdrenoAttentionPath::Decomposed3Stage => self.execute_decomposed_staged(
                plan,
                cache_key,
                q_buf,
                k_buf,
                v_buf,
                out_buf,
                &push,
                q_elems,
                score_elems,
                capture_stages,
                &mut check_cancelled,
            ),
        }
    }

    /// Executes 3-stage decomposed attention in one command buffer with explicit barriers.
    ///
    /// # Errors
    /// Propagates validation, planning, or Vulkan execution errors.
    pub fn execute_pipelined_decomposed(
        &mut self,
        desc: &AttentionDescriptor,
        q: &[f32],
        k: &[f32],
        v: &[f32],
    ) -> Result<AttentionOutput, Error> {
        let (q_elems, _kv_elems, score_elems) = validate_attention_inputs(desc, q, k, v)?;
        let plan = select_attention_plan(
            desc,
            self.device.capabilities(),
            AttentionStrategy::ForceDecomposed,
            self.scratch_budget_bytes,
        )?;
        let cache_key = attention_cache_key(
            desc,
            &plan,
            self.device.capabilities(),
            &self.qk_digest,
            &self.softmax_digest,
            &self.pv_digest,
            &self.fused_digest,
        )?;

        let q_buf = upload_f32_scratch(&mut self.device, &mut self.scratch.q, q)?;
        let k_buf = upload_f32_scratch(&mut self.device, &mut self.scratch.k, k)?;
        let v_buf = upload_f32_scratch(&mut self.device, &mut self.scratch.v, v)?;
        let scores_buf = ensure_scratch(
            &mut self.device,
            &mut self.scratch.scores,
            f32_slice_bytes(score_elems)?,
        )?;
        let probs_buf = ensure_scratch(
            &mut self.device,
            &mut self.scratch.probs,
            f32_slice_bytes(score_elems)?,
        )?;
        let out_buf = ensure_scratch(
            &mut self.device,
            &mut self.scratch.out,
            f32_slice_bytes(q_elems)?,
        )?;

        let push = attention_push_constants(desc);
        let qk_spans = [
            BufferSpan::whole(q_buf),
            BufferSpan::whole(k_buf),
            BufferSpan::whole(scores_buf),
        ];
        let sm_spans = [BufferSpan::whole(scores_buf), BufferSpan::whole(probs_buf)];
        let pv_spans = [
            BufferSpan::whole(probs_buf),
            BufferSpan::whole(v_buf),
            BufferSpan::whole(out_buf),
        ];
        let stages = [
            ComputeSpanStage {
                pipeline: self.pipe_qk,
                buffers: &qk_spans,
                push_u32s: &push,
                workgroups: plan.qk_workgroups,
            },
            ComputeSpanStage {
                pipeline: self.pipe_softmax,
                buffers: &sm_spans,
                push_u32s: &push,
                workgroups: plan.softmax_workgroups,
            },
            ComputeSpanStage {
                pipeline: self.pipe_pv,
                buffers: &pv_spans,
                push_u32s: &push,
                workgroups: plan.pv_workgroups,
            },
        ];

        self.device.submit_compute_span_stages(&stages)?;
        let mut output = vec![0.0_f32; q_elems];
        self.device.read_buffer_f32(out_buf, &mut output)?;
        ensure_finite_f32(
            &output,
            "GPU pipelined attention produced non-finite output",
        )?;
        Ok(AttentionOutput {
            output,
            witness: self.witness(plan, cache_key, 3),
            stages: None,
        })
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "Encapsulates fused submission, readback, finite verification, and witness emission."
    )]
    fn execute_fused(
        &mut self,
        plan: crate::AttentionTilePlan,
        cache_key: String,
        q_buf: BufferId,
        k_buf: BufferId,
        v_buf: BufferId,
        out_buf: BufferId,
        push: &[u32; 8],
        q_elems: usize,
    ) -> Result<AttentionOutput, Error> {
        self.device.submit_compute(
            self.pipe_fused,
            &[q_buf, k_buf, v_buf, out_buf],
            push,
            plan.qk_workgroups,
        )?;
        let mut output = vec![0.0_f32; q_elems];
        self.device.read_buffer_f32(out_buf, &mut output)?;
        ensure_finite_f32(&output, "GPU fused attention produced non-finite output")?;
        Ok(AttentionOutput {
            output,
            witness: self.witness(plan, cache_key, 1),
            stages: None,
        })
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "Coordinates 3-stage dispatches, cancellation checkpoints, and optional stage capture."
    )]
    fn execute_decomposed_staged<F>(
        &mut self,
        plan: crate::AttentionTilePlan,
        cache_key: String,
        q_buf: BufferId,
        k_buf: BufferId,
        v_buf: BufferId,
        out_buf: BufferId,
        push: &[u32; 8],
        q_elems: usize,
        score_elems: usize,
        capture_stages: bool,
        check_cancelled: &mut F,
    ) -> Result<AttentionOutput, Error>
    where
        F: FnMut() -> bool,
    {
        let scratch_bytes = f32_slice_bytes(score_elems)?;
        let scores_buf = ensure_scratch(&mut self.device, &mut self.scratch.scores, scratch_bytes)?;
        let probs_buf = ensure_scratch(&mut self.device, &mut self.scratch.probs, scratch_bytes)?;

        self.device.submit_compute(
            self.pipe_qk,
            &[q_buf, k_buf, scores_buf],
            push,
            plan.qk_workgroups,
        )?;
        if check_cancelled() {
            return Err(Error::Cancelled);
        }

        self.device.submit_compute(
            self.pipe_softmax,
            &[scores_buf, probs_buf],
            push,
            plan.softmax_workgroups,
        )?;
        if check_cancelled() {
            return Err(Error::Cancelled);
        }

        self.device.submit_compute(
            self.pipe_pv,
            &[probs_buf, v_buf, out_buf],
            push,
            plan.pv_workgroups,
        )?;

        let mut output = vec![0.0_f32; q_elems];
        self.device.read_buffer_f32(out_buf, &mut output)?;
        ensure_finite_f32(
            &output,
            "GPU decomposed attention produced non-finite output",
        )?;

        let stages = if capture_stages {
            let mut scores = vec![0.0_f32; score_elems];
            let mut rounded_probs = vec![0.0_f32; score_elems];
            self.device.read_buffer_f32(scores_buf, &mut scores)?;
            self.device.read_buffer_f32(probs_buf, &mut rounded_probs)?;
            ensure_finite_f32(&scores, "GPU attention scores produced non-finite values")?;
            ensure_finite_f32(
                &rounded_probs,
                "GPU attention probs produced non-finite values",
            )?;
            Some(AttentionStageOutputs {
                scores,
                rounded_probs,
            })
        } else {
            None
        };

        Ok(AttentionOutput {
            output,
            witness: self.witness(plan, cache_key, 3),
            stages,
        })
    }

    fn witness(
        &self,
        plan: crate::AttentionTilePlan,
        cache_key: String,
        stage_count: u32,
    ) -> AttentionExecutionWitness {
        AttentionExecutionWitness {
            plan,
            cache_key,
            stage_count,
            qk_shader_digest: self.qk_digest.clone(),
            softmax_shader_digest: self.softmax_digest.clone(),
            pv_shader_digest: self.pv_digest.clone(),
            fused_shader_digest: self.fused_digest.clone(),
        }
    }
}

impl Drop for VulkanAttentionExecutor {
    fn drop(&mut self) {
        self.scratch.free_all(&mut self.device);
    }
}

fn is_4tok_pipeline(pipes: &ProjectionPipelines, pipe: PipelineId) -> bool {
    pipe == pipes.pq2_q8k_4tok_fused
        || pipe == pipes.pq2_q8k_4tok_unfused
        || pipe == pipes.pq2_q8k_4tok_fused_res
        || pipe == pipes.pq2_q8k_4tok_unfused_res
        || pipe == pipes.pq2_q8_0_4tok
        || pipe == pipes.ptq1_q8_0_4tok
        || pipe == pipes.ptq1_q8_0_4tok_pinned
        || pipe == pipes.q8_0_q8_0_4tok
}

fn is_32tok_pipeline(pipes: &ProjectionPipelines, pipe: PipelineId) -> bool {
    pipe == pipes.pq2_q8k_32tok_fused
        || pipe == pipes.pq2_q8k_32tok_unfused
        || pipe == pipes.pq2_q8k_32tok_fused_res
        || pipe == pipes.pq2_q8k_32tok_unfused_res
        || pipe == pipes.pq2_q8_0_32tok
        || pipe == pipes.ptq1_q8_0_32tok
        || pipe == pipes.ptq1_q8_0_32tok_pinned
        || pipe == pipes.q8_0_q8_0_32tok
}

fn is_16tok_pipeline(pipes: &ProjectionPipelines, pipe: PipelineId) -> bool {
    pipe == pipes.pq2_q8k_16tok_fused
        || pipe == pipes.pq2_q8k_16tok_unfused
        || pipe == pipes.pq2_q8k_16tok_fused_res
        || pipe == pipes.pq2_q8k_16tok_unfused_res
        || pipe == pipes.ptq1_q8_0_16tok
        || pipe == pipes.ptq1_q8_0_16tok_pinned
        || pipe == pipes.q8_0_q8_0_16tok
}

fn ensure_scratch(
    device: &mut AdrenoVulkanDevice,
    slot: &mut ScratchBuf,
    needed_bytes: u64,
) -> Result<BufferId, Error> {
    let req = needed_bytes.max(MIN_SCRATCH_BYTES);
    let max_range = u64::from(
        device
            .capabilities()
            .compute_limits
            .max_storage_buffer_range,
    );
    if req > max_range {
        return Err(Error::ResourceLimit(format!(
            "scratch buffer ({req} B) exceeds maxStorageBufferRange ({max_range} B)"
        )));
    }
    if let Some(id) = slot.id {
        if slot.cap_bytes >= req {
            return Ok(id);
        }
        device.free_buffer(id);
        slot.id = None;
        slot.cap_bytes = 0;
    }
    let id = device.create_storage_buffer(req)?;
    slot.id = Some(id);
    slot.cap_bytes = req;
    Ok(id)
}

fn upload_f32_scratch(
    device: &mut AdrenoVulkanDevice,
    slot: &mut ScratchBuf,
    values: &[f32],
) -> Result<BufferId, Error> {
    let bytes = f32_slice_bytes(values.len())?;
    let id = ensure_scratch(device, slot, bytes)?;
    device.write_buffer_f32(id, values)?;
    Ok(id)
}

fn checked_gemv_dims(
    rows: usize,
    blocks_per_row: usize,
    tokens: usize,
) -> Result<(u32, u32, u32), Error> {
    let rows_u32 = u32::try_from(rows)
        .map_err(|_| Error::ResourceLimit("projection row count exceeds u32".to_owned()))?;
    let blocks_u32 = u32::try_from(blocks_per_row)
        .map_err(|_| Error::ResourceLimit("projection block count exceeds u32".to_owned()))?;
    let tokens_u32 = u32::try_from(tokens)
        .map_err(|_| Error::ResourceLimit("projection token count exceeds u32".to_owned()))?;
    Ok((rows_u32, blocks_u32, tokens_u32))
}

pub(crate) fn checked_grid(device: &AdrenoVulkanDevice, grid: [u32; 3]) -> Result<[u32; 3], Error> {
    let max_wg = device
        .capabilities()
        .compute_limits
        .max_compute_work_group_count;
    if grid.contains(&0) || grid[0] > max_wg[0] || grid[1] > max_wg[1] || grid[2] > max_wg[2] {
        return Err(Error::ResourceLimit(
            "projection workgroup count out of maxComputeWorkGroupCount bounds".to_owned(),
        ));
    }
    Ok(grid)
}

fn u32_slice_bytes_usize(word_count: usize) -> Result<usize, Error> {
    word_count
        .checked_mul(4)
        .ok_or_else(|| Error::ResourceLimit("u32 byte count overflow".to_owned()))
}

pub(crate) fn u32_slice_bytes(word_count: usize) -> Result<u64, Error> {
    u64::try_from(word_count)
        .ok()
        .and_then(|w| w.checked_mul(4))
        .ok_or_else(|| Error::ResourceLimit("u32 byte length overflow".to_owned()))
}

pub(crate) fn f32_slice_bytes(elem_count: usize) -> Result<u64, Error> {
    u64::try_from(elem_count)
        .ok()
        .and_then(|e| e.checked_mul(4))
        .ok_or_else(|| Error::ResourceLimit("f32 byte length overflow".to_owned()))
}

pub(crate) fn ensure_finite_f32(values: &[f32], msg: &str) -> Result<(), Error> {
    if values.iter().any(|val| !val.is_finite()) {
        return Err(Error::InvalidInput(msg.to_owned()));
    }
    Ok(())
}

pub(crate) fn parse_hadamard_block(block: Option<u32>, cols: usize) -> Result<Option<u32>, Error> {
    match block {
        None | Some(1) => Ok(None),
        Some(b)
            if b.is_power_of_two()
                && (2..=1024).contains(&b)
                && usize::try_from(b).is_ok_and(|bu| cols.is_multiple_of(bu)) =>
        {
            Ok(Some(b))
        }
        Some(b) => Err(Error::InvalidInput(format!(
            "unsupported GPU Hadamard block size {b} for {cols} columns"
        ))),
    }
}

pub(crate) fn hadamard_norm_bits(block_size: u32) -> u32 {
    let f = u16::try_from(block_size).map_or(1.0_f32, f32::from);
    (1.0_f32 / f.sqrt()).to_bits()
}

pub(crate) const fn attention_push_constants(desc: &AttentionDescriptor) -> [u32; 8] {
    [
        desc.query_len,
        desc.total_seq,
        desc.past_len,
        desc.num_heads,
        desc.num_kv_heads,
        desc.head_dim,
        desc.scale.to_bits(),
        if desc.causal { 1 } else { 0 },
    ]
}

fn validate_attention_inputs(
    desc: &AttentionDescriptor,
    q: &[f32],
    k: &[f32],
    v: &[f32],
) -> Result<(usize, usize, usize), Error> {
    let (q_elems, kv_elems, score_elems) = desc.validate()?;
    if q.len() != q_elems || k.len() != kv_elems || v.len() != kv_elems {
        return Err(Error::InvalidInput(format!(
            "mismatched Q/K/V lengths: got ({}, {}, {}), expected ({q_elems}, {kv_elems}, {kv_elems})",
            q.len(),
            k.len(),
            v.len()
        )));
    }
    ensure_finite_f32(q, "attention Q contains non-finite f32 values")?;
    ensure_finite_f32(k, "attention K contains non-finite f32 values")?;
    ensure_finite_f32(v, "attention V contains non-finite f32 values")?;
    Ok((q_elems, kv_elems, score_elems))
}

pub(crate) fn validate_matrix(
    tensor: &Tensor,
    expected_kind: u32,
    col_alignment: usize,
) -> Result<(usize, usize), Error> {
    if tensor.kind != expected_kind {
        return Err(Error::InvalidInput(format!(
            "tensor '{}' has kind {}, expected {expected_kind}",
            tensor.name, tensor.kind
        )));
    }
    let [cols, rows] = tensor.dimensions.as_slice() else {
        return Err(Error::InvalidInput(format!(
            "tensor '{}' must be rank 2, got {:?}",
            tensor.name, tensor.dimensions
        )));
    };
    let (cols, rows) = (*cols, *rows);
    if cols == 0 || rows == 0 || !cols.is_multiple_of(col_alignment) {
        return Err(Error::InvalidInput(format!(
            "tensor '{}' has invalid extents [{cols}, {rows}] for alignment {col_alignment}",
            tensor.name
        )));
    }
    let (block_cols, block_bytes) = match expected_kind {
        KIND_PQ2 => (128_usize, PQ2_BLOCK_BYTES),
        KIND_PTQ1 => (128_usize, PTQ1_BLOCK_BYTES),
        KIND_Q8_0 => (32_usize, Q8_0_BLOCK_BYTES),
        other => {
            return Err(Error::InvalidInput(format!(
                "unsupported matrix kind {other}"
            )));
        }
    };
    if !cols.is_multiple_of(block_cols) {
        return Err(Error::InvalidInput(format!(
            "tensor '{}' has invalid extents [{cols}, {rows}] for block alignment {block_cols}",
            tensor.name
        )));
    }
    let expected_bytes = (cols / block_cols)
        .checked_mul(block_bytes)
        .and_then(|row_bytes| row_bytes.checked_mul(rows))
        .ok_or_else(|| Error::ResourceLimit("tensor byte size overflow".to_owned()))?;
    if tensor.data.len() != expected_bytes {
        return Err(Error::InvalidInput(format!(
            "tensor '{}' byte length {} != expected {expected_bytes}",
            tensor.name,
            tensor.data.len()
        )));
    }
    Ok((cols, rows))
}

pub(crate) fn validate_weight_scales(kind: u32, data: &[u8]) -> Result<(), Error> {
    match kind {
        KIND_PQ2 => {
            for block in data.as_chunks::<PQ2_BLOCK_BYTES>().0 {
                let bits = u16::from_le_bytes([block[0], block[1]]);
                check_f16_scale(bits)?;
            }
        }
        KIND_Q8_0 => {
            for block in data.as_chunks::<Q8_0_BLOCK_BYTES>().0 {
                let bits = u16::from_le_bytes([block[0], block[1]]);
                check_f16_scale(bits)?;
            }
        }
        KIND_PTQ1 => {
            for block in data.as_chunks::<PTQ1_BLOCK_BYTES>().0 {
                let bits = u16::from_le_bytes([block[26], block[27]]);
                check_f16_scale(bits)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn check_f16_scale(bits: u16) -> Result<(), Error> {
    if (bits & 0x7C00) == 0x7C00 {
        return Err(Error::InvalidInput(
            "quantized weight block has non-finite f16 scale".to_owned(),
        ));
    }
    Ok(())
}

pub(crate) fn repack_coalesced_into(
    kind: u32,
    raw: &[u8],
    cols: usize,
    rows: usize,
    dst: &mut Vec<u32>,
) -> Result<(), Error> {
    match kind {
        KIND_PQ2 => {
            repack_pq2_coalesced_64(raw, cols, rows, dst)?;
            Ok(())
        }
        KIND_PTQ1 => {
            repack_ptq1_coalesced_64(raw, cols, rows, dst)?;
            Ok(())
        }
        KIND_Q8_0 => {
            repack_q8_0_coalesced_64(raw, cols, rows, dst)?;
            Ok(())
        }
        other => Err(Error::InvalidInput(format!(
            "unsupported coalesced weight kind {other}"
        ))),
    }
}

fn repack_scale_and_32b_coalesced_64(
    raw: &[u8],
    cols: usize,
    rows: usize,
    block_cols: usize,
    tile_words: usize,
    dst: &mut Vec<u32>,
) -> Result<(), Error> {
    if cols == 0 || rows == 0 || !cols.is_multiple_of(block_cols) {
        return Err(Error::InvalidInput(format!(
            "invalid coalesced matrix dimensions [{cols}, {rows}] for block {block_cols}"
        )));
    }
    let blocks_per_row = cols / block_cols;
    let row_bytes = blocks_per_row
        .checked_mul(Q8_0_BLOCK_BYTES)
        .ok_or_else(|| Error::ResourceLimit("coalesced row byte count overflow".to_owned()))?;
    let expected_raw_bytes = rows
        .checked_mul(row_bytes)
        .ok_or_else(|| Error::ResourceLimit("coalesced raw byte count overflow".to_owned()))?;
    if raw.len() != expected_raw_bytes {
        return Err(Error::InvalidInput(format!(
            "coalesced raw byte length {} != expected {expected_raw_bytes}",
            raw.len()
        )));
    }
    let row_groups = rows.div_ceil(64);
    let total_words = row_groups
        .checked_mul(blocks_per_row)
        .and_then(|t| t.checked_mul(tile_words))
        .ok_or_else(|| Error::ResourceLimit("coalesced weight word count overflow".to_owned()))?;
    dst.clear();
    dst.resize(total_words, 0);

    for (rg, rg_words) in dst
        .chunks_exact_mut(blocks_per_row * tile_words)
        .enumerate()
    {
        let r_base = rg * 64;
        for (blk, tile) in rg_words.chunks_exact_mut(tile_words).enumerate() {
            let blk_off = blk * Q8_0_BLOCK_BYTES;
            for (pair, scale_slot) in tile[..32].iter_mut().enumerate() {
                let r_ev = r_base + pair * 2;
                let r_od = r_ev + 1;
                let s_ev = if r_ev < rows {
                    let b = &raw[r_ev * row_bytes + blk_off..];
                    u16::from_le_bytes([b[0], b[1]])
                } else {
                    0
                };
                let s_od = if r_od < rows {
                    let b = &raw[r_od * row_bytes + blk_off..];
                    u16::from_le_bytes([b[0], b[1]])
                } else {
                    0
                };
                *scale_slot = u32::from(s_ev) | (u32::from(s_od) << 16);
            }
            for lane in 0..64 {
                let r = r_base + lane;
                if r >= rows {
                    continue;
                }
                let block = &raw[r * row_bytes + blk_off..][..Q8_0_BLOCK_BYTES];
                for half_j in 0..2 {
                    for c in 0..4 {
                        let word_j = half_j * 4 + c;
                        let c_off = 2 + word_j * 4;
                        let word = u32::from_le_bytes([
                            block[c_off],
                            block[c_off + 1],
                            block[c_off + 2],
                            block[c_off + 3],
                        ]);
                        tile[32 + (half_j * 64 + lane) * 4 + c] = word;
                    }
                }
            }
        }
    }
    Ok(())
}

fn repack_pq2_coalesced_64(
    raw: &[u8],
    cols: usize,
    rows: usize,
    dst: &mut Vec<u32>,
) -> Result<(), Error> {
    repack_scale_and_32b_coalesced_64(raw, cols, rows, 128, PQ2_COALESCED_TILE_WORDS, dst)
}

fn repack_q8_0_coalesced_64(
    raw: &[u8],
    cols: usize,
    rows: usize,
    dst: &mut Vec<u32>,
) -> Result<(), Error> {
    repack_scale_and_32b_coalesced_64(raw, cols, rows, 32, Q8_0_COALESCED_TILE_WORDS, dst)
}

fn repack_ptq1_coalesced_64(
    raw: &[u8],
    cols: usize,
    rows: usize,
    dst: &mut Vec<u32>,
) -> Result<(), Error> {
    let blocks_per_row = cols / 128;
    let row_bytes = blocks_per_row * PTQ1_BLOCK_BYTES;
    let row_groups = rows.div_ceil(64);
    let total_words = row_groups
        .checked_mul(blocks_per_row)
        .and_then(|t| t.checked_mul(PTQ1_COALESCED_TILE_WORDS))
        .ok_or_else(|| Error::ResourceLimit("coalesced PTQ1 word count overflow".to_owned()))?;
    dst.clear();
    dst.resize(total_words, 0);

    for (rg, rg_words) in dst
        .chunks_exact_mut(blocks_per_row * PTQ1_COALESCED_TILE_WORDS)
        .enumerate()
    {
        let r_base = rg * 64;
        for (blk, tile) in rg_words
            .as_chunks_mut::<PTQ1_COALESCED_TILE_WORDS>()
            .0
            .iter_mut()
            .enumerate()
        {
            let blk_off = blk * PTQ1_BLOCK_BYTES;
            for lane in 0..64 {
                let r = r_base + lane;
                if r >= rows {
                    continue;
                }
                let block = &raw[r * row_bytes + blk_off..][..PTQ1_BLOCK_BYTES];
                for (w_idx, chunk) in block.as_chunks::<4>().0.iter().enumerate() {
                    tile[w_idx * 64 + lane] = u32::from_le_bytes(*chunk);
                }
            }
        }
    }
    Ok(())
}

fn pack_q8k_blocks_4plane_into(blocks: &[Q8KBlock], out: &mut Vec<u32>) -> Result<(), Error> {
    out.clear();
    out.reserve(blocks.len() * 2 * Q8K_4PLANE_WORDS_PER_HALF);
    for block in blocks {
        if !block.scale.is_finite() {
            return Err(Error::InvalidInput(
                "Q8_K activation block has non-finite scale".to_owned(),
            ));
        }
        let scale_bits = block.scale.to_bits();
        for sub in block.values.as_chunks::<128>().0 {
            let half_sum: i32 = sub.iter().map(|&v| i32::from(v)).sum();
            out.push(scale_bits);
            out.push(half_sum.cast_unsigned());
            out.push(0);
            out.push(0);
            for chunk16 in sub.as_chunks::<16>().0 {
                for digit in 0..4 {
                    out.push(u32::from_le_bytes([
                        chunk16[digit].cast_unsigned(),
                        chunk16[digit + 4].cast_unsigned(),
                        chunk16[digit + 8].cast_unsigned(),
                        chunk16[digit + 12].cast_unsigned(),
                    ]));
                }
            }
        }
    }
    Ok(())
}

fn pack_q8k_blocks_into(blocks: &[Q8KBlock], out: &mut Vec<u32>) -> Result<(), Error> {
    out.clear();
    out.reserve(blocks.len() * Q8K_WORDS_PER_BLOCK);
    for block in blocks {
        if !block.scale.is_finite() {
            return Err(Error::InvalidInput(
                "Q8_K activation block has non-finite scale".to_owned(),
            ));
        }
        out.push(block.scale.to_bits());
        for chunk in block.values.as_chunks::<4>().0 {
            out.push(u32::from_le_bytes([
                chunk[0].cast_unsigned(),
                chunk[1].cast_unsigned(),
                chunk[2].cast_unsigned(),
                chunk[3].cast_unsigned(),
            ]));
        }
    }
    Ok(())
}

fn pack_q8_blocks_into(blocks: &[Q8Block], out: &mut Vec<u32>) -> Result<(), Error> {
    out.clear();
    out.reserve(blocks.len() * Q8_0_WORDS_PER_BLOCK);
    for block in blocks {
        if !block.scale.is_finite() {
            return Err(Error::InvalidInput(
                "Q8_0 activation block has non-finite scale".to_owned(),
            ));
        }
        out.push(block.scale.to_bits());
        out.push(0);
        out.push(0);
        out.push(0);
        for chunk in block.values.as_chunks::<4>().0 {
            out.push(u32::from_le_bytes([
                chunk[0].cast_unsigned(),
                chunk[1].cast_unsigned(),
                chunk[2].cast_unsigned(),
                chunk[3].cast_unsigned(),
            ]));
        }
    }
    Ok(())
}

fn evict_standalone_cache_entries(
    weight_cache: &mut BTreeMap<WeightCacheKey, CachedWeight>,
    cached_bytes: &mut usize,
    max_cached_weight_bytes: usize,
    upload_bytes: usize,
    protected: &[BufferId],
    mut free_buf: impl FnMut(BufferId),
) -> bool {
    while *cached_bytes + upload_bytes > max_cached_weight_bytes {
        let Some(evict_key) = weight_cache
            .iter()
            .filter(|(_, entry)| match entry.storage {
                ResidentStorage::Standalone(id) => !protected.contains(&id),
                ResidentStorage::Arena(_) => false,
            })
            .min_by_key(|(_, entry)| entry.last_used)
            .map(|(k, _)| k.clone())
        else {
            return false;
        };
        if let Some(removed) = weight_cache.remove(&evict_key) {
            *cached_bytes = cached_bytes.saturating_sub(removed.bytes);
            if let ResidentStorage::Standalone(id) = removed.storage {
                free_buf(id);
            }
        }
    }
    true
}

#[expect(
    clippy::too_many_arguments,
    reason = "Pure cache insertion bookkeeping decoupled from AdrenoVulkanDevice for unit tests."
)]
fn record_cached_entry(
    weight_cache: &mut BTreeMap<WeightCacheKey, CachedWeight>,
    cached_bytes: &mut usize,
    cache_access_counter: &mut u64,
    key: WeightCacheKey,
    storage: ResidentStorage,
    bytes: usize,
    source: &TensorBytes,
    mut free_buf: impl FnMut(BufferId),
) {
    *cache_access_counter += 1;
    if let Some(prev) = weight_cache.insert(
        key,
        CachedWeight {
            storage,
            bytes,
            weak_source: TensorBytes::downgrade(source),
            last_used: *cache_access_counter,
        },
    ) {
        *cached_bytes = cached_bytes.saturating_sub(prev.bytes);
        if let ResidentStorage::Standalone(id) = prev.storage {
            free_buf(id);
        }
    }
    *cached_bytes += bytes;
}

type ThreeMatrixDims = ((usize, usize), Option<(usize, usize)>, Option<(usize, usize)>);

fn validate_three_pq2_matrices(
    t0: &Tensor,
    t1: Option<&Tensor>,
    t2: Option<&Tensor>,
) -> Result<ThreeMatrixDims, Error> {
    let dims0 = validate_matrix(t0, KIND_PQ2, 256)?;
    let dims1 = t1.map(|t| validate_matrix(t, KIND_PQ2, 256)).transpose()?;
    let dims2 = t2.map(|t| validate_matrix(t, KIND_PQ2, 256)).transpose()?;
    Ok((dims0, dims1, dims2))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    fn test_pq2_tensor(name: &str, cols: usize, rows: usize) -> Tensor {
        let blocks_per_row = cols / 128;
        let mut raw = vec![0_u8; rows * blocks_per_row * PQ2_BLOCK_BYTES];
        for blk in raw.as_chunks_mut::<PQ2_BLOCK_BYTES>().0 {
            blk[0] = 0x00;
            blk[1] = 0x3C;
        }
        Tensor {
            name: name.to_owned(),
            dimensions: vec![cols, rows],
            kind: KIND_PQ2,
            data: Arc::<[u8]>::from(raw).into(),
        }
    }

    #[test]
    fn weight_cache_bookkeeping_evicts_lru_skips_protected_and_replaces_entries() {
        struct TestCache {
            cache: BTreeMap<WeightCacheKey, CachedWeight>,
            cached_bytes: usize,
            counter: u64,
            freed: Vec<BufferId>,
        }
        impl TestCache {
            fn insert(&mut self, key: WeightCacheKey, st: ResidentStorage, bytes: usize, src: &Tensor) {
                let freed = &mut self.freed;
                record_cached_entry(
                    &mut self.cache,
                    &mut self.cached_bytes,
                    &mut self.counter,
                    key,
                    st,
                    bytes,
                    &src.data,
                    |id| freed.push(id),
                );
            }
            fn evict(&mut self, max: usize, upload: usize, prot: &[BufferId]) -> bool {
                let freed = &mut self.freed;
                evict_standalone_cache_entries(
                    &mut self.cache,
                    &mut self.cached_bytes,
                    max,
                    upload,
                    prot,
                    |id| freed.push(id),
                )
            }
        }
        let mut tc = TestCache {
            cache: BTreeMap::new(),
            cached_bytes: 0,
            counter: 0,
            freed: Vec::new(),
        };
        let arena_t = test_pq2_tensor("arena", 256, 32);
        let prot_t = test_pq2_tensor("prot", 256, 32);
        let lru_t = test_pq2_tensor("lru", 256, 32);
        let newer_t = test_pq2_tensor("newer", 256, 32);
        let key = |name: &str| WeightCacheKey {
            name: name.to_owned(),
            kind: KIND_PQ2,
            cols: 256,
            rows: 32,
        };
        tc.insert(key("arena"), ResidentStorage::Arena(BufferSpan::whole(BufferId(1))), 400, &arena_t);
        tc.insert(key("prot"), ResidentStorage::Standalone(BufferId(2)), 300, &prot_t);
        tc.insert(key("lru"), ResidentStorage::Standalone(BufferId(3)), 200, &lru_t);
        tc.insert(key("newer"), ResidentStorage::Standalone(BufferId(4)), 200, &newer_t);
        assert_eq!(tc.cached_bytes, 1100);

        tc.insert(key("newer"), ResidentStorage::Standalone(BufferId(5)), 250, &newer_t);
        assert_eq!(tc.freed, vec![BufferId(4)]);
        assert_eq!(tc.cached_bytes, 1150);

        assert!(tc.evict(1200, 250, &[BufferId(2)]));
        assert_eq!(tc.freed, vec![BufferId(4), BufferId(3)]);
        assert_eq!(tc.cached_bytes, 950);

        assert!(!tc.evict(800, 250, &[BufferId(2)]));
        assert_eq!(tc.freed, vec![BufferId(4), BufferId(3), BufferId(5)]);
        assert_eq!(tc.cached_bytes, 700);
        assert!(tc.cache.contains_key(&key("arena")) && tc.cache.contains_key(&key("prot")));

        tc.insert(key("arena"), ResidentStorage::Arena(BufferSpan::whole(BufferId(1))), 300, &arena_t);
        assert_eq!(tc.freed.len(), 3);
        assert_eq!(tc.cached_bytes, 600);
    }

    #[test]
    fn three_matrix_validation_rejects_bad_mates_before_acquisition() -> Result<(), Error> {
        let valid0 = test_pq2_tensor("t0", 256, 32);
        let mut bad1 = test_pq2_tensor("bad1", 256, 32);
        bad1.kind = KIND_Q8_0;
        assert!(validate_three_pq2_matrices(&valid0, Some(&bad1), None).is_err());
        assert!(validate_three_pq2_matrices(&valid0, Some(&valid0), Some(&bad1)).is_err());
        assert_eq!(
            validate_three_pq2_matrices(&valid0, Some(&valid0), Some(&valid0))?,
            ((256, 32), Some((256, 32)), Some((256, 32)))
        );
        Ok(())
    }

    #[test]
    fn repack_q8_0_coalesced_64_packs_scales_and_words_with_tail_zero_pad() -> Result<(), Error> {
        let cols = 64_usize; // 2 blocks per row
        let rows = 67_usize; // 2 row groups (64 + 3)
        let blocks_per_row = cols / 32;
        let row_bytes = blocks_per_row * Q8_0_BLOCK_BYTES;
        let mut raw = vec![0_u8; rows * row_bytes];

        for r in 0..rows {
            for blk in 0..blocks_per_row {
                let off = r * row_bytes + blk * Q8_0_BLOCK_BYTES;
                let scale_bits = u16::try_from(0x3C00 + r + blk * 7).unwrap_or(0x3C00);
                raw[off..off + 2].copy_from_slice(&scale_bits.to_le_bytes());
                for i in 0..32 {
                    let raw_i32 = i32::try_from((r * 5 + blk * 11 + i) % 251).unwrap_or(0) - 125;
                    let byte = i8::try_from(raw_i32).unwrap_or(0).cast_unsigned();
                    raw[off + 2 + i] = byte;
                }
            }
        }

        validate_weight_scales(KIND_Q8_0, &raw)?;
        let tensor = Tensor {
            name: "test.q8_0".to_owned(),
            dimensions: vec![cols, rows],
            kind: KIND_Q8_0,
            data: Arc::<[u8]>::from(raw.clone()).into(),
        };
        assert_eq!(validate_matrix(&tensor, KIND_Q8_0, 32)?, (cols, rows));

        let mut dst = Vec::new();
        repack_coalesced_into(KIND_Q8_0, &raw, cols, rows, &mut dst)?;
        assert_eq!(dst.len(), 2 * blocks_per_row * Q8_0_COALESCED_TILE_WORDS);

        for r in 0..rows {
            let rg = r / 64;
            let lane = r % 64;
            for blk in 0..blocks_per_row {
                let tile_base = (rg * blocks_per_row + blk) * Q8_0_COALESCED_TILE_WORDS;
                let pair_word = dst[tile_base + (lane / 2)];
                let scale_u16 = if lane.is_multiple_of(2) {
                    u16::try_from(pair_word & 0xFFFF).unwrap_or(0)
                } else {
                    u16::try_from(pair_word >> 16).unwrap_or(0)
                };
                let raw_off = r * row_bytes + blk * Q8_0_BLOCK_BYTES;
                assert_eq!(
                    scale_u16,
                    u16::from_le_bytes([raw[raw_off], raw[raw_off + 1]])
                );
                for word_j in 0..8 {
                    let half_j = word_j / 4;
                    let c = word_j % 4;
                    let c_off = raw_off + 2 + word_j * 4;
                    let expected_word = u32::from_le_bytes([
                        raw[c_off],
                        raw[c_off + 1],
                        raw[c_off + 2],
                        raw[c_off + 3],
                    ]);
                    assert_eq!(
                        dst[tile_base + 32 + (half_j * 64 + lane) * 4 + c],
                        expected_word
                    );
                }
            }
        }

        // Tail rows in the second row group (`lane >= 3`) must remain zero-padded.
        for blk in 0..blocks_per_row {
            let tile_base = (blocks_per_row + blk) * Q8_0_COALESCED_TILE_WORDS;
            let pair1 = dst[tile_base + 1];
            assert_eq!(pair1 >> 16, 0);
            for pair in 2..32 {
                assert_eq!(dst[tile_base + pair], 0);
            }
            for half_j in 0..2 {
                for lane in 3..64 {
                    for c in 0..4 {
                        assert_eq!(dst[tile_base + 32 + (half_j * 64 + lane) * 4 + c], 0);
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn validate_q8_0_matrix_and_scales_reject_invalid_inputs() {
        let mut raw = vec![0_u8; Q8_0_BLOCK_BYTES];
        raw[0..2].copy_from_slice(&0x7C00_u16.to_le_bytes()); // +Inf f16 scale
        assert!(validate_weight_scales(KIND_Q8_0, &raw).is_err());

        raw[0..2].copy_from_slice(&0x7E00_u16.to_le_bytes()); // NaN f16 scale
        assert!(validate_weight_scales(KIND_Q8_0, &raw).is_err());

        let bad_cols = Tensor {
            name: "bad.cols".to_owned(),
            dimensions: vec![48, 2],
            kind: KIND_Q8_0,
            data: Arc::<[u8]>::from(vec![0_u8; 4 * Q8_0_BLOCK_BYTES]).into(),
        };
        assert!(validate_matrix(&bad_cols, KIND_Q8_0, 32).is_err());

        let bad_len = Tensor {
            name: "bad.len".to_owned(),
            dimensions: vec![32, 2],
            kind: KIND_Q8_0,
            data: Arc::<[u8]>::from(vec![0_u8; Q8_0_BLOCK_BYTES]).into(),
        };
        assert!(validate_matrix(&bad_len, KIND_Q8_0, 32).is_err());
    }

    #[test]
    #[ignore = "Requires live Qualcomm Adreno Vulkan hardware."]
    #[expect(
        clippy::too_many_lines,
        reason = "End-to-end live Adreno parity check across all three 32-token double-buffered GEMMs."
    )]
    fn gemm32_pq2_q8_0_and_ptq1_match_1tok_at_t32() -> Result<(), Error> {
        let mut dev = AdrenoVulkanDevice::open()?;
        let sdot = dev.capabilities().supports_integer_dot_4x8();
        let cols = 1024_usize;
        let rows = 131_usize;
        let tokens = 32_usize;
        let pq2_bpr = cols / 128;
        let q8_bpr = cols / 32;

        let make_coal = |bpr: usize| -> Vec<u32> {
            let rg = rows.div_ceil(64);
            let mut out = vec![0_u32; rg * bpr * 544];
            for (i, w) in out.iter_mut().enumerate() {
                let in_tile = i % 544;
                if in_tile < 32 {
                    *w = 0x3800_3400; // two small finite f16 scales (0.25, 0.5)
                } else {
                    *w = u32::try_from((i * 17 + 3) & 0xFFFF_FFFF).unwrap_or(0);
                }
            }
            out
        };
        let w_544 = make_coal(q8_bpr);
        let w_ptq1 = {
            let rg = rows.div_ceil(64);
            let mut out = vec![0_u32; rg * pq2_bpr * 448];
            for (i, w) in out.iter_mut().enumerate() {
                let in_tile = i % 448;
                if in_tile >= 384 {
                    *w = 0x3400_0010;
                } else {
                    *w = u32::try_from((i * 13 + 1) & 0x0FFF_0FFF).unwrap_or(0);
                }
            }
            out
        };
        let act_q8k: Vec<u32> = (0..tokens * pq2_bpr * 34)
            .map(|i| {
                if i % 34 == 0 {
                    0.125_f32.to_bits()
                } else if i % 34 == 1 {
                    12_i32.cast_unsigned()
                } else {
                    0x0102_0304_u32.wrapping_add(u32::try_from(i).unwrap_or(0))
                }
            })
            .collect();
        let act_q8_0: Vec<u32> = (0..tokens * q8_bpr * 9)
            .map(|i| {
                if i % 9 == 0 {
                    0.125_f32.to_bits()
                } else {
                    0x01FF_02FE_u32.wrapping_add(u32::try_from(i).unwrap_or(0))
                }
            })
            .collect();

        let b_w544 = dev.create_storage_buffer(u32_slice_bytes(w_544.len())?)?;
        let b_ptq1 = dev.create_storage_buffer(u32_slice_bytes(w_ptq1.len())?)?;
        let buf_act_q8k = dev.create_storage_buffer(u32_slice_bytes(act_q8k.len())?)?;
        let buf_act_q8_0 = dev.create_storage_buffer(u32_slice_bytes(act_q8_0.len())?)?;
        let b_ref = dev.create_storage_buffer(f32_slice_bytes(tokens * rows)?)?;
        let b_32 = dev.create_storage_buffer(f32_slice_bytes(tokens * rows)?)?;
        dev.write_buffer_u32(b_w544, &w_544)?;
        dev.write_buffer_u32(b_ptq1, &w_ptq1)?;
        dev.write_buffer_u32(buf_act_q8k, &act_q8k)?;
        dev.write_buffer_u32(buf_act_q8_0, &act_q8_0)?;

        let rows_u32 = u32::try_from(rows).unwrap_or(0);
        let tokens_u32 = u32::try_from(tokens).unwrap_or(0);
        let tile_x = rows_u32.div_ceil(64);
        let mut out_ref = vec![0.0_f32; tokens * rows];
        let mut out_32 = vec![0.0_f32; tokens * rows];

        for use_sdot in if sdot { vec![false, true] } else { vec![false] } {
            // 1. PQ2 x Q8K
            let p_ref = dev.create_compute_pipeline(
                &spirv::pq2_q8k_gemv_coalesced(use_sdot, true, false),
                3,
                4,
            )?;
            let p_32 = dev.create_compute_pipeline(
                &spirv::pq2_q8k_gemm_32tok_coalesced(use_sdot, true, false),
                3,
                4,
            )?;
            let push_k = [
                rows_u32,
                u32::try_from(pq2_bpr).unwrap_or(0),
                tokens_u32,
                u32::try_from(pq2_bpr * 34).unwrap_or(0),
            ];
            dev.submit_compute(
                p_ref,
                &[b_w544, buf_act_q8k, b_ref],
                &push_k,
                [tile_x, tokens_u32, 1],
            )?;
            dev.submit_compute(p_32, &[b_w544, buf_act_q8k, b_32], &push_k, [1, tile_x, 1])?;
            dev.read_buffer_f32(b_ref, &mut out_ref)?;
            dev.read_buffer_f32(b_32, &mut out_32)?;
            for (i, (&a, &b)) in out_32.iter().zip(&out_ref).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "pq2_q8k T=32 sdot={use_sdot} idx={i}"
                );
            }

            // 2. Q8_0 x Q8_0
            let q_ref =
                dev.create_compute_pipeline(&spirv::q8_0_q8_0_gemv_coalesced(use_sdot), 3, 4)?;
            let q_32 = dev.create_compute_pipeline(
                &spirv::q8_0_q8_0_gemm_32tok_coalesced(use_sdot),
                3,
                4,
            )?;
            let push_8 = [
                rows_u32,
                u32::try_from(q8_bpr).unwrap_or(0),
                tokens_u32,
                u32::try_from(q8_bpr * 9).unwrap_or(0),
            ];
            dev.submit_compute(
                q_ref,
                &[b_w544, buf_act_q8_0, b_ref],
                &push_8,
                [tile_x, tokens_u32, 1],
            )?;
            dev.submit_compute(q_32, &[b_w544, buf_act_q8_0, b_32], &push_8, [1, tile_x, 1])?;
            dev.read_buffer_f32(b_ref, &mut out_ref)?;
            dev.read_buffer_f32(b_32, &mut out_32)?;
            for (i, (&a, &b)) in out_32.iter().zip(&out_ref).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "q8_0 T=32 sdot={use_sdot} idx={i}"
                );
            }

            // 3. PTQ1 x Q8_0
            for pinned in [false, true] {
                let t_ref = dev.create_compute_pipeline(
                    &spirv::ptq1_q8_0_gemv_coalesced(use_sdot, pinned),
                    3,
                    4,
                )?;
                let t_32 = dev.create_compute_pipeline(
                    &spirv::ptq1_q8_0_gemm_32tok_coalesced(use_sdot, pinned),
                    3,
                    4,
                )?;
                let push_t = [
                    rows_u32,
                    u32::try_from(pq2_bpr).unwrap_or(0),
                    tokens_u32,
                    u32::try_from(pq2_bpr * 36).unwrap_or(0),
                ];
                dev.submit_compute(
                    t_ref,
                    &[b_ptq1, buf_act_q8_0, b_ref],
                    &push_t,
                    [tile_x, tokens_u32, 1],
                )?;
                dev.submit_compute(t_32, &[b_ptq1, buf_act_q8_0, b_32], &push_t, [1, tile_x, 1])?;
                dev.read_buffer_f32(b_ref, &mut out_ref)?;
                dev.read_buffer_f32(b_32, &mut out_32)?;
                for (i, (&a, &b)) in out_32.iter().zip(&out_ref).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "ptq1 T=32 sdot={use_sdot} pinned={pinned} idx={i}"
                    );
                }
            }

            // 4. PQ2 x Q8_0
            let pq_q8_ref =
                dev.create_compute_pipeline(&spirv::pq2_q8_0_gemv_coalesced(use_sdot), 3, 4)?;
            let pq_q8_32 =
                dev.create_compute_pipeline(&spirv::pq2_q8_0_gemm_32tok_coalesced(use_sdot), 3, 4)?;
            let push_pq8 = [
                rows_u32,
                u32::try_from(pq2_bpr).unwrap_or(0),
                tokens_u32,
                u32::try_from(pq2_bpr * 48).unwrap_or(0),
            ];
            dev.submit_compute(
                pq_q8_ref,
                &[b_w544, buf_act_q8_0, b_ref],
                &push_pq8,
                [tile_x, tokens_u32, 1],
            )?;
            dev.submit_compute(
                pq_q8_32,
                &[b_w544, buf_act_q8_0, b_32],
                &push_pq8,
                [1, tile_x, 1],
            )?;
            dev.read_buffer_f32(b_ref, &mut out_ref)?;
            dev.read_buffer_f32(b_32, &mut out_32)?;
            for (i, (&a, &b)) in out_32.iter().zip(&out_ref).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "pq2_q8_0 T=32 sdot={use_sdot} idx={i}"
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "Requires live Qualcomm Adreno Vulkan hardware."]
    #[expect(
        clippy::too_many_lines,
        reason = "End-to-end live Adreno parity check across Split-K decode GEMVs."
    )]
    fn splitk_decode_gemvs_compile_and_match_1tok_within_1e4() -> Result<(), Error> {
        let mut dev = AdrenoVulkanDevice::open()?;
        if !supports_splitk_gemv(&dev) {
            return Ok(());
        }
        let sdot = dev.capabilities().supports_integer_dot_4x8();
        let cols = 2048_usize;
        let rows = 131_usize;
        let pq2_bpr = cols / 128;
        let q8_bpr = cols / 32;
        let rg = rows.div_ceil(64);

        let w_544: Vec<u32> = (0..rg * q8_bpr * 544)
            .map(|i| {
                if i % 544 < 32 {
                    0x3400_3200
                } else {
                    u32::try_from((i * 17 + 3) & 0xFFFF_FFFF).unwrap_or(0)
                }
            })
            .collect();
        let w_ptq1: Vec<u32> = (0..rg * pq2_bpr * 448)
            .map(|i| {
                if i % 448 >= 384 {
                    0x3400_0010
                } else {
                    u32::try_from((i * 13 + 1) & 0x0FFF_0FFF).unwrap_or(0)
                }
            })
            .collect();
        let act_q8k: Vec<u32> = (0..pq2_bpr * 36)
            .map(|i| {
                if i % 36 == 0 {
                    0.0625_f32.to_bits()
                } else if i % 36 == 1 {
                    12_i32.cast_unsigned()
                } else if i % 36 < 4 {
                    0
                } else {
                    0x0102_0304_u32.wrapping_add(u32::try_from(i).unwrap_or(0))
                }
            })
            .collect();
        let act_q8_0: Vec<u32> = (0..q8_bpr * 12)
            .map(|i| {
                if i % 12 == 0 {
                    0.0625_f32.to_bits()
                } else if i % 12 < 4 {
                    0
                } else {
                    0x01FF_02FE_u32.wrapping_add(u32::try_from(i).unwrap_or(0))
                }
            })
            .collect();

        let b_w544 = dev.create_storage_buffer(u32_slice_bytes(w_544.len())?)?;
        let b_ptq1 = dev.create_storage_buffer(u32_slice_bytes(w_ptq1.len())?)?;
        let buf_act_q8k = dev.create_storage_buffer(u32_slice_bytes(act_q8k.len())?)?;
        let buf_act_q8_0 = dev.create_storage_buffer(u32_slice_bytes(act_q8_0.len())?)?;
        let b_ref = dev.create_storage_buffer(f32_slice_bytes(rows)?)?;
        let buf_splitk = dev.create_storage_buffer(f32_slice_bytes(rows)?)?;
        dev.write_buffer_u32(b_w544, &w_544)?;
        dev.write_buffer_u32(b_ptq1, &w_ptq1)?;
        dev.write_buffer_u32(buf_act_q8k, &act_q8k)?;
        dev.write_buffer_u32(buf_act_q8_0, &act_q8_0)?;

        let rows_u32 = u32::try_from(rows).unwrap_or(0);
        let tile_x = rows_u32.div_ceil(64);
        let mut out_ref = vec![0.0_f32; rows];
        let mut out_sk = vec![0.0_f32; rows];

        for use_sdot in if sdot { vec![false, true] } else { vec![false] } {
            let push_k = [
                rows_u32,
                u32::try_from(pq2_bpr).unwrap_or(0),
                1,
                u32::try_from(pq2_bpr * 36).unwrap_or(0),
            ];
            let p_ref = dev.create_compute_pipeline(
                &spirv::pq2_q8k_gemv_coalesced(use_sdot, true, false),
                3,
                4,
            )?;
            let p_sk = dev.create_compute_pipeline(
                &spirv::pq2_q8k_gemv_splitk(use_sdot, true, false),
                3,
                4,
            )?;
            dev.submit_compute(
                p_ref,
                &[b_w544, buf_act_q8k, b_ref],
                &push_k,
                [tile_x, 1, 1],
            )?;
            dev.submit_compute(
                p_sk,
                &[b_w544, buf_act_q8k, buf_splitk],
                &push_k,
                [tile_x, 1, 1],
            )?;
            dev.read_buffer_f32(b_ref, &mut out_ref)?;
            dev.read_buffer_f32(buf_splitk, &mut out_sk)?;
            for (i, (&a, &b)) in out_sk.iter().zip(&out_ref).enumerate() {
                assert!((a - b).abs() <= 1e-4, "pq2_q8k splitk idx={i} {a} vs {b}");
            }

            let push_8 = [
                rows_u32,
                u32::try_from(q8_bpr).unwrap_or(0),
                1,
                u32::try_from(q8_bpr * 12).unwrap_or(0),
            ];
            let q_ref =
                dev.create_compute_pipeline(&spirv::q8_0_q8_0_gemv_coalesced(use_sdot), 3, 4)?;
            let q_sk =
                dev.create_compute_pipeline(&spirv::q8_0_q8_0_gemv_splitk(use_sdot), 3, 4)?;
            dev.submit_compute(
                q_ref,
                &[b_w544, buf_act_q8_0, b_ref],
                &push_8,
                [tile_x, 1, 1],
            )?;
            dev.submit_compute(
                q_sk,
                &[b_w544, buf_act_q8_0, buf_splitk],
                &push_8,
                [tile_x, 1, 1],
            )?;
            dev.read_buffer_f32(b_ref, &mut out_ref)?;
            dev.read_buffer_f32(buf_splitk, &mut out_sk)?;
            for (i, (&a, &b)) in out_sk.iter().zip(&out_ref).enumerate() {
                assert!((a - b).abs() <= 1e-4, "q8_0 splitk idx={i} {a} vs {b}");
            }

            let push_t = [
                rows_u32,
                u32::try_from(pq2_bpr).unwrap_or(0),
                1,
                u32::try_from(pq2_bpr * 48).unwrap_or(0),
            ];
            let t_ref = dev.create_compute_pipeline(
                &spirv::ptq1_q8_0_gemv_coalesced(use_sdot, true),
                3,
                4,
            )?;
            let t_sk =
                dev.create_compute_pipeline(&spirv::ptq1_q8_0_gemv_splitk(use_sdot, true), 3, 4)?;
            dev.submit_compute(
                t_ref,
                &[b_ptq1, buf_act_q8_0, b_ref],
                &push_t,
                [tile_x, 1, 1],
            )?;
            dev.submit_compute(
                t_sk,
                &[b_ptq1, buf_act_q8_0, buf_splitk],
                &push_t,
                [tile_x, 1, 1],
            )?;
            dev.read_buffer_f32(b_ref, &mut out_ref)?;
            dev.read_buffer_f32(buf_splitk, &mut out_sk)?;
            for (i, (&a, &b)) in out_sk.iter().zip(&out_ref).enumerate() {
                assert!((a - b).abs() <= 1e-4, "ptq1 splitk idx={i} {a} vs {b}");
            }
        }
        Ok(())
    }
}
