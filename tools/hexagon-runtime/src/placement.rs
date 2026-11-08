//! Immutable explicit placement for a bounded partial-graph experiment.

use std::{fmt::Write, sync::Arc};

use bonsai_native_hash::Hasher;
use bonsai_prism_runtime::{
    kernels::{Pq2ProjectionExecutor, Q8Block},
    model::{Model, Tensor},
};
use bonsai_ternary::Q8KBlock;

use crate::{Result, invalid};

const MAX_SELECTIONS: usize = 1024;

#[derive(Clone, Copy, Eq, PartialEq)]
enum Route {
    F32,
    Q8K,
    Q8,
}

struct Selected {
    tensor: Tensor,
    route: Route,
    operations: Option<u64>,
}

/// An explicit, immutable subset of a borrowed projection executor's placement.
///
/// Build this from the verified model before moving that model into its graph.
/// Selected tensors retain their immutable serialized storage; matching requires
/// the same storage, name, type and dimensions. All other projections use the
/// graph's existing CPU path before submission. Unknown, duplicate and unsupported
/// selections are errors rather than silent CPU-only experiments.
///
/// The graph's ordinary whole-request operation admission counts only these
/// selected projections. No session capacity, lifetime or resource limit is
/// enlarged. This supports bounded partial-placement qualification; it establishes
/// neither device qualification nor efficient complete-model NPU inference.
pub struct PlacedProjectionExecutor<'a> {
    executor: &'a mut dyn Pq2ProjectionExecutor,
    selected: Vec<Selected>,
    identity: String,
    f32_groups: bool,
    faulted: bool,
}

impl<'a> PlacedProjectionExecutor<'a> {
    /// Freeze a nonempty set of at most 1024 uniquely named model projections.
    ///
    /// The placement identity binds the inner executor, verified model identity,
    /// sorted tensor names, encodings, geometry and exact operation counts.
    /// Serialized model data is shared, not rehashed or copied. The caller must
    /// separately account for retained model storage, selection metadata and all
    /// executor/native resources. No submission occurs during construction.
    ///
    /// # Errors
    /// Rejects unknown, duplicate or unsupported selections, malformed identities,
    /// missing nonzero counts for a bounded executor, and allocation failures.
    pub fn new(
        executor: &'a mut dyn Pq2ProjectionExecutor,
        model: &Model,
        names: &[&str],
    ) -> Result<Self> {
        if names.is_empty()
            || names.len() > MAX_SELECTIONS
            || !valid_identity(executor.identity())
            || !valid_identity(&model.identity)
        {
            return Err(invalid("invalid explicit projection placement"));
        }
        let mut selected = Vec::new();
        selected.try_reserve_exact(names.len())?;
        let bounded = executor.operation_capacity().is_some();
        for &name in names {
            let tensor = model
                .tensors
                .get(name)
                .filter(|tensor| tensor.name == name && tensor.dimensions.len() == 2)
                .ok_or_else(|| invalid("unknown projection placement tensor"))?;
            let route = match (
                tensor.kind,
                tensor.dimensions[0].is_multiple_of(128),
                tensor.dimensions[0].is_multiple_of(256),
            ) {
                (142, true, _) if executor.supports_f32(tensor) => Route::F32,
                (142, _, true) if executor.supports(tensor) => Route::Q8K,
                (142, _, false) | (143, _, _) if executor.supports_q8(tensor) => Route::Q8,
                _ => return Err(invalid("unsupported explicit projection placement tensor")),
            };
            let operations = executor.operation_count(tensor);
            if bounded && operations.is_none_or(|count| count == 0) {
                return Err(invalid(
                    "explicit placement requires bounded operation counts",
                ));
            }
            let mut dimensions = Vec::new();
            dimensions.try_reserve_exact(tensor.dimensions.len())?;
            dimensions.extend_from_slice(&tensor.dimensions);
            let mut name = String::new();
            name.try_reserve_exact(tensor.name.len())?;
            name.push_str(&tensor.name);
            selected.push(Selected {
                tensor: Tensor {
                    name,
                    dimensions,
                    kind: tensor.kind,
                    data: Arc::clone(&tensor.data),
                },
                route,
                operations,
            });
        }
        selected.sort_unstable_by(|left, right| left.tensor.name.cmp(&right.tensor.name));
        if selected
            .windows(2)
            .any(|pair| pair[0].tensor.name == pair[1].tensor.name)
        {
            return Err(invalid("duplicate explicit projection placement tensor"));
        }
        let mut digest = Hasher::new()?;
        digest.update(b"prism-explicit-projection-placement-v1\0")?;
        hash_bytes(&mut digest, executor.identity().as_bytes())?;
        hash_bytes(&mut digest, model.identity.as_bytes())?;
        for selection in &selected {
            let tensor = &selection.tensor;
            hash_bytes(&mut digest, tensor.name.as_bytes())?;
            digest.update(&tensor.kind.to_le_bytes())?;
            for &dimension in &tensor.dimensions {
                digest.update(&u64::try_from(dimension)?.to_le_bytes())?;
            }
            digest.update(&[u8::from(selection.operations.is_some())])?;
            digest.update(&selection.operations.unwrap_or(0).to_le_bytes())?;
        }
        let mut identity = String::new();
        identity.try_reserve_exact(96)?;
        identity.push_str("hexagon-explicit-placement-v1:");
        for byte in digest.finish()? {
            write!(&mut identity, "{byte:02x}")?;
        }
        Ok(Self {
            executor,
            selected,
            identity,
            f32_groups: false,
            faulted: false,
        })
    }

    /// Enable or disable batched `F32` sibling-group delegation when all group
    /// members belong to this explicit placement.
    #[must_use]
    pub const fn with_f32_groups(mut self, enabled: bool) -> Self {
        self.f32_groups = enabled;
        self
    }

    /// Whether any delegated operation failed and requires terminal teardown.
    #[must_use]
    pub const fn is_faulted(&self) -> bool {
        self.faulted
    }

    fn selection(&self, tensor: &Tensor) -> Option<&Selected> {
        let index = self
            .selected
            .binary_search_by(|selection| selection.tensor.name.cmp(&tensor.name))
            .ok()?;
        let selection = &self.selected[index];
        (Arc::ptr_eq(&selection.tensor.data, &tensor.data)
            && selection.tensor.kind == tensor.kind
            && selection.tensor.dimensions == tensor.dimensions)
            .then_some(selection)
    }
}

fn valid_identity(identity: &str) -> bool {
    !identity.is_empty()
        && identity.len() <= 256
        && identity.bytes().all(|byte| byte.is_ascii_graphic())
}

fn hash_bytes(digest: &mut Hasher, bytes: &[u8]) -> Result<()> {
    digest.update(&u64::try_from(bytes.len())?.to_le_bytes())?;
    digest.update(bytes)?;
    Ok(())
}

impl Pq2ProjectionExecutor for PlacedProjectionExecutor<'_> {
    fn identity(&self) -> &str {
        &self.identity
    }

    fn operation_capacity(&self) -> Option<u64> {
        if self.faulted {
            Some(0)
        } else {
            self.executor.operation_capacity()
        }
    }

    fn operation_count(&self, tensor: &Tensor) -> Option<u64> {
        self.selection(tensor)
            .and_then(|selection| selection.operations)
    }

    fn supports(&self, tensor: &Tensor) -> bool {
        self.selection(tensor)
            .is_some_and(|selection| selection.route == Route::Q8K)
    }

    fn supports_q8(&self, tensor: &Tensor) -> bool {
        self.selection(tensor)
            .is_some_and(|selection| selection.route == Route::Q8)
    }

    fn supports_f32(&self, tensor: &Tensor) -> bool {
        self.selection(tensor)
            .is_some_and(|selection| selection.route == Route::F32)
    }

    fn project(
        &mut self,
        tensor: &Tensor,
        activation: &[Q8KBlock],
        fused: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        output.clear();
        if self.faulted || !self.supports(tensor) {
            return Err(invalid("unavailable explicit Q8_K projection placement"));
        }
        let result = self.executor.project(tensor, activation, fused, output);
        if result.is_err() {
            self.faulted = true;
            output.clear();
        }
        result
    }

    fn project_q8(
        &mut self,
        tensor: &Tensor,
        activation: &[Q8Block],
        pinned: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        output.clear();
        if self.faulted || !self.supports_q8(tensor) {
            return Err(invalid("unavailable explicit Q8_0 projection placement"));
        }
        let result = self.executor.project_q8(tensor, activation, pinned, output);
        if result.is_err() {
            self.faulted = true;
            output.clear();
        }
        result
    }

    fn project_f32(
        &mut self,
        tensor: &Tensor,
        activation: &[f32],
        pinned: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        if self.faulted || !self.supports_f32(tensor) {
            output.clear();
            return Err(invalid("unavailable explicit F32 projection placement"));
        }
        let result = self
            .executor
            .project_f32(tensor, activation, pinned, output);
        if result.is_err() {
            self.faulted = true;
            output.clear();
        }
        result
    }

    fn supports_f32_group(&self, tensors: &[&Tensor]) -> bool {
        self.f32_groups
            && !self.faulted
            && !tensors.is_empty()
            && tensors.iter().all(|&tensor| self.supports_f32(tensor))
            && self.executor.supports_f32_group(tensors)
    }

    fn project_f32_group(
        &mut self,
        tensors: &[&Tensor],
        activation: &[f32],
        pinned: bool,
        outputs: &mut [&mut Vec<f32>],
    ) -> Result<()> {
        if !self.supports_f32_group(tensors) {
            for output in outputs.iter_mut() {
                output.clear();
            }
            return Err(invalid(
                "unavailable explicit F32 group projection placement",
            ));
        }
        let result = self
            .executor
            .project_f32_group(tensors, activation, pinned, outputs);
        if result.is_err() {
            self.faulted = true;
            for output in outputs.iter_mut() {
                output.clear();
            }
        }
        result
    }

    fn project_f32_tokens(
        &mut self,
        tensor: &Tensor,
        activations: &[Vec<f32>],
        pinned: bool,
        outputs: &mut [Vec<f32>],
    ) -> Result<()> {
        if self.faulted || !self.supports_f32(tensor) {
            for output in outputs.iter_mut() {
                output.clear();
            }
            return Err(invalid("unavailable explicit F32 projection placement"));
        }
        let result = self
            .executor
            .project_f32_tokens(tensor, activations, pinned, outputs);
        if result.is_err() {
            self.faulted = true;
            for output in outputs.iter_mut() {
                output.clear();
            }
        }
        result
    }

    fn project_f32_pair_tokens(
        &mut self,
        tensors: [&Tensor; 2],
        activations: &[Vec<f32>],
        pinned: bool,
        outputs_a: &mut [Vec<f32>],
        outputs_b: &mut [Vec<f32>],
    ) -> Result<()> {
        if !self.supports_f32_group(&tensors) {
            for output in outputs_a.iter_mut() {
                output.clear();
            }
            for output in outputs_b.iter_mut() {
                output.clear();
            }
            return Err(invalid(
                "unavailable explicit F32 group projection placement",
            ));
        }
        let result = self.executor.project_f32_pair_tokens(
            tensors,
            activations,
            pinned,
            outputs_a,
            outputs_b,
        );
        if result.is_err() {
            self.faulted = true;
            for output in outputs_a.iter_mut() {
                output.clear();
            }
            for output in outputs_b.iter_mut() {
                output.clear();
            }
        }
        result
    }

    fn project_f32_triple_tokens(
        &mut self,
        tensors: [&Tensor; 3],
        activations: &[Vec<f32>],
        pinned: bool,
        outputs_a: &mut [Vec<f32>],
        outputs_b: &mut [Vec<f32>],
        outputs_c: &mut [Vec<f32>],
    ) -> Result<()> {
        if !self.supports_f32_group(&tensors) {
            for output in outputs_a.iter_mut() {
                output.clear();
            }
            for output in outputs_b.iter_mut() {
                output.clear();
            }
            for output in outputs_c.iter_mut() {
                output.clear();
            }
            return Err(invalid(
                "unavailable explicit F32 group projection placement",
            ));
        }
        let result = self.executor.project_f32_triple_tokens(
            tensors,
            activations,
            pinned,
            outputs_a,
            outputs_b,
            outputs_c,
        );
        if result.is_err() {
            self.faulted = true;
            for output in outputs_a.iter_mut() {
                output.clear();
            }
            for output in outputs_b.iter_mut() {
                output.clear();
            }
            for output in outputs_c.iter_mut() {
                output.clear();
            }
        }
        result
    }

    fn supports_f32_ffn_swiglu(&self, gate: &Tensor, up: &Tensor) -> bool {
        self.f32_groups
            && !self.faulted
            && self.supports_f32(gate)
            && self.supports_f32(up)
            && self.executor.supports_f32_ffn_swiglu(gate, up)
    }

    fn project_f32_ffn_swiglu(
        &mut self,
        gate: &Tensor,
        up: &Tensor,
        activation: &[f32],
        pinned: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        if !self.supports_f32_ffn_swiglu(gate, up) {
            output.clear();
            return Err(invalid(
                "unavailable explicit F32 FFN SwiGLU projection placement",
            ));
        }
        let result = self
            .executor
            .project_f32_ffn_swiglu(gate, up, activation, pinned, output);
        if result.is_err() {
            self.faulted = true;
            output.clear();
        }
        result
    }

    fn project_f32_ffn_swiglu_tokens(
        &mut self,
        gate: &Tensor,
        up: &Tensor,
        activations: &[Vec<f32>],
        pinned: bool,
        outputs: &mut [Vec<f32>],
    ) -> Result<()> {
        if !self.supports_f32_ffn_swiglu(gate, up) {
            for output in outputs.iter_mut() {
                output.clear();
            }
            return Err(invalid(
                "unavailable explicit F32 FFN SwiGLU projection placement",
            ));
        }
        let result =
            self.executor
                .project_f32_ffn_swiglu_tokens(gate, up, activations, pinned, outputs);
        if result.is_err() {
            self.faulted = true;
            for output in outputs.iter_mut() {
                output.clear();
            }
        }
        result
    }
}

/// One double-buffered pipeline step in a 2-stage Hexagon NPU + CPU prefill schedule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HybridPrefillStep {
    /// Microbatch index executing on Stage 0 (Hexagon NPU `blk.0..npu_layers`), when active.
    pub npu_microbatch: Option<usize>,
    /// Microbatch index executing on Stage 1 (Oryon CPU `blk.npu_layers..total_layers`), when active.
    pub cpu_microbatch: Option<usize>,
    /// Token range `[start..end)` of the active NPU microbatch (or `0..0` when idle).
    pub npu_token_range: std::ops::Range<usize>,
    /// Token range `[start..end)` of the active CPU microbatch (or `0..0` when idle).
    pub cpu_token_range: std::ops::Range<usize>,
}

/// Two-stage double-buffered chunked prefill schedule overlapping Hexagon NPU
/// (`blk.0..npu_layers`) on microbatch `m + 1` with Oryon CPU (`blk.npu_layers..total_layers`)
/// on microbatch `m`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HybridPrefillSchedule {
    /// Layer range assigned to Stage 0 (Hexagon NPU, e.g., `0..27`).
    pub npu_layer_range: std::ops::Range<usize>,
    /// Layer range assigned to Stage 1 (Oryon CPU / Adreno GPU, e.g., `27..64`).
    pub cpu_layer_range: std::ops::Range<usize>,
    /// Maximum tokens per microbatch (`1..=64`, typically `8`).
    pub microbatch_tokens: usize,
    /// Total number of microbatches covering `prompt_tokens`.
    pub microbatch_count: usize,
    /// Ordered pipeline steps (`microbatch_count + 1` steps when `microbatch_count >= 1`).
    pub steps: Vec<HybridPrefillStep>,
}

/// Plan a 2-stage double-buffered Hexagon NPU + Oryon CPU chunked prefill pipeline.
///
/// Stage 0 (`0..npu_layers`) runs microbatch `m + 1` on the Hexagon NPU while
/// Stage 1 (`npu_layers..total_layers`) runs microbatch `m` on the Oryon CPU.
///
/// # Errors
/// Rejects zero layers, `npu_layers >= total_layers`, zero prompt tokens or
/// `microbatch_tokens` outside `1..=64`.
pub fn plan_hybrid_prefill_pipeline(
    total_layers: usize,
    npu_layers: usize,
    prompt_tokens: usize,
    microbatch_tokens: usize,
) -> Result<HybridPrefillSchedule> {
    if total_layers < 2
        || npu_layers == 0
        || npu_layers >= total_layers
        || prompt_tokens == 0
        || microbatch_tokens == 0
        || microbatch_tokens > 64
    {
        return Err(invalid(
            "invalid hybrid NPU+CPU prefill pipeline parameters",
        ));
    }
    let microbatch_count = prompt_tokens.div_ceil(microbatch_tokens);
    let step_count = microbatch_count + 1;
    let mut steps = Vec::new();
    steps
        .try_reserve_exact(step_count)
        .map_err(|_| invalid("hybrid prefill schedule allocation failed"))?;
    let token_range = |mb: usize| {
        let start = mb * microbatch_tokens;
        let end = (start + microbatch_tokens).min(prompt_tokens);
        start..end
    };
    for step in 0..step_count {
        let npu_microbatch = (step < microbatch_count).then_some(step);
        let cpu_microbatch = step.checked_sub(1).filter(|&mb| mb < microbatch_count);
        steps.push(HybridPrefillStep {
            npu_microbatch,
            cpu_microbatch,
            npu_token_range: npu_microbatch.map_or(0..0, token_range),
            cpu_token_range: cpu_microbatch.map_or(0..0, token_range),
        });
    }
    Ok(HybridPrefillSchedule {
        npu_layer_range: 0..npu_layers,
        cpu_layer_range: npu_layers..total_layers,
        microbatch_tokens,
        microbatch_count,
        steps,
    })
}

/// Target compute unit for a projection group in a 3-processor heterogeneous placement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AcceleratorTarget {
    /// Hexagon v81 NPU (`libggml-htp-v81.so` signed HVX `HTP_TYPE_Q2_0` + `HTP_OP_HADAMARD`).
    HexagonNpu,
    /// Adreno X2-90 / X1-85 `OpenCL` `Q2_0` + local-memory `fwht.cl` (`1024`-element block).
    AdrenoGpuOpenCl,
    /// Adreno X2-90 / X1-85 Vulkan `Q2_0` + shared-memory `hadamard.comp` (`up to 8192`).
    AdrenoGpuVulkan,
    /// Oryon CPU (ARMv8.2-A `dotprod` / `i8mm` + NEON recurrent GDN, KV cache, and `LoRA` adapters).
    OryonCpu,
}

/// Adreno GPU `Q2_0` / `FWHT` kernel family selected for a projection stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdrenoQ2KernelKind {
    /// Batch-1 decode 4-row GEMV (`mul_mv_q2_0_f32_4x_flat.cl`, 64-lane subgroup shuffles).
    DecodeGemv4xFlat,
    /// Batch-1 decode 8-row GEMV (`N_DST = 8` for `M >= 16384` `ffn_gate`/`ffn_up`).
    DecodeGemv8xFlat,
    /// Multi-token prefill 4-column GEMV (`gemv_noshuffle.cl`).
    PrefillGemvNoShuffle4Col,
    /// `OpenCL` local-memory Fast Walsh-Hadamard transform (`fwht.cl`, `32..=1024`).
    FwhtLocalMem1024,
    /// Vulkan shared-memory Fast Walsh-Hadamard transform (`hadamard.comp`, `32..=8192`).
    FwhtVulkan8192,
}

/// Per-layer assignment in a 3-processor heterogeneous placement plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeterogeneousLayerAssignment {
    /// Zero-based decoder layer index (`0..total_layers`).
    pub layer_index: usize,
    /// Whether this layer is a full-attention layer (`7` `Q2_0` projections) or recurrent GDN (`6` `Q2_0` projections).
    pub is_full_attention: bool,
    /// Compute processor assigned to this layer's `Q2_0` linear projections and `FWHT` basis transforms.
    pub projection_target: AcceleratorTarget,
    /// Number of `Q2_0` linear projections in this layer (`6` or `7`).
    pub q2_projection_count: usize,
    /// Primary Adreno decode kernel when `projection_target` is an Adreno GPU backend.
    pub adreno_decode_kernel: Option<AdrenoQ2KernelKind>,
    /// Primary Adreno prefill kernel when `projection_target` is an Adreno GPU backend.
    pub adreno_prefill_kernel: Option<AdrenoQ2KernelKind>,
    /// Primary Adreno `FWHT` kernel when `projection_target` is an Adreno GPU backend.
    pub adreno_fwht_kernel: Option<AdrenoQ2KernelKind>,
}

/// Complete 3-processor heterogeneous placement plan across Hexagon NPU (`blk.0..npu_layers`),
/// Adreno GPU (`blk.npu_layers..npu_layers + gpu_layers`), and Oryon CPU.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeterogeneousPlacementPlan {
    /// Stable hardware-hashed placement identity (`heterogeneous-3proc-placement-v1:<sha256>`).
    pub identity: String,
    /// Total decoder layers in the model.
    pub total_layers: usize,
    /// Number of layers assigned to Hexagon NPU (`blk.0..npu_layers`).
    pub npu_layers: usize,
    /// Number of layers assigned to Adreno GPU (`blk.npu_layers..npu_layers + gpu_layers`).
    pub gpu_layers: usize,
    /// Number of layers whose projections remain on Oryon CPU.
    pub cpu_layers: usize,
    /// Total `Q2_0` projections assigned to Hexagon NPU (`168` for `blk.0..=26` on Qwen3.5-27B).
    pub npu_q2_projections: usize,
    /// Total `Q2_0` projections assigned to Adreno GPU (`232` for `blk.27..=63` on Qwen3.5-27B).
    pub gpu_q2_projections: usize,
    /// Total `Q2_0` projections remaining on Oryon CPU.
    pub cpu_q2_projections: usize,
    /// Per-layer placement descriptors.
    pub layers: Vec<HeterogeneousLayerAssignment>,
}

/// Build a validated 3-processor heterogeneous placement plan across Hexagon NPU,
/// Adreno GPU (`OpenCL` or `Vulkan` `Q2_0` + `FWHT`), and Oryon CPU.
///
/// # Errors
/// Rejects zero `total_layers`, `full_attention_interval == 0`, `npu_layers + gpu_layers > total_layers`,
/// or non-GPU `gpu_backend` when `gpu_layers > 0`.
pub fn plan_three_processor_placement(
    total_layers: usize,
    npu_layers: usize,
    gpu_layers: usize,
    full_attention_interval: usize,
    gpu_backend: AcceleratorTarget,
) -> Result<HeterogeneousPlacementPlan> {
    if total_layers == 0
        || full_attention_interval == 0
        || npu_layers
            .checked_add(gpu_layers)
            .is_none_or(|sum| sum > total_layers)
        || (gpu_layers > 0
            && !matches!(
                gpu_backend,
                AcceleratorTarget::AdrenoGpuOpenCl | AcceleratorTarget::AdrenoGpuVulkan
            ))
    {
        return Err(invalid(
            "invalid 3-processor heterogeneous placement parameters",
        ));
    }
    let cpu_layers = total_layers - (npu_layers + gpu_layers);
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(total_layers)
        .map_err(|_| invalid("heterogeneous placement layer allocation failed"))?;
    let mut npu_q2_projections = 0_usize;
    let mut gpu_q2_projections = 0_usize;
    let mut cpu_q2_projections = 0_usize;
    let mut digest = Hasher::new()?;
    digest.update(b"prism-heterogeneous-3proc-placement-v1\0")?;

    for layer_index in 0..total_layers {
        let is_full_attention = (layer_index + 1).is_multiple_of(full_attention_interval);
        let q2_projection_count = if is_full_attention { 7 } else { 6 };
        let (projection_target, adreno_decode_kernel, adreno_prefill_kernel, adreno_fwht_kernel) =
            if layer_index < npu_layers {
                npu_q2_projections += q2_projection_count;
                (AcceleratorTarget::HexagonNpu, None, None, None)
            } else if layer_index < npu_layers + gpu_layers {
                gpu_q2_projections += q2_projection_count;
                let fwht = if gpu_backend == AcceleratorTarget::AdrenoGpuVulkan {
                    AdrenoQ2KernelKind::FwhtVulkan8192
                } else {
                    AdrenoQ2KernelKind::FwhtLocalMem1024
                };
                (
                    gpu_backend,
                    Some(AdrenoQ2KernelKind::DecodeGemv8xFlat),
                    Some(AdrenoQ2KernelKind::PrefillGemvNoShuffle4Col),
                    Some(fwht),
                )
            } else {
                cpu_q2_projections += q2_projection_count;
                (AcceleratorTarget::OryonCpu, None, None, None)
            };
        let target_tag: u8 = match projection_target {
            AcceleratorTarget::HexagonNpu => 1,
            AcceleratorTarget::AdrenoGpuOpenCl => 2,
            AcceleratorTarget::AdrenoGpuVulkan => 3,
            AcceleratorTarget::OryonCpu => 4,
        };
        digest.update(&u64::try_from(layer_index)?.to_le_bytes())?;
        digest.update(&[target_tag, u8::from(is_full_attention)])?;
        layers.push(HeterogeneousLayerAssignment {
            layer_index,
            is_full_attention,
            projection_target,
            q2_projection_count,
            adreno_decode_kernel,
            adreno_prefill_kernel,
            adreno_fwht_kernel,
        });
    }

    let mut identity = String::new();
    identity.try_reserve_exact(103)?;
    identity.push_str("heterogeneous-3proc-placement-v1:");
    for byte in digest.finish()? {
        write!(&mut identity, "{byte:02x}")?;
    }

    Ok(HeterogeneousPlacementPlan {
        identity,
        total_layers,
        npu_layers,
        gpu_layers,
        cpu_layers,
        npu_q2_projections,
        gpu_q2_projections,
        cpu_q2_projections,
        layers,
    })
}

#[cfg(test)]
mod tests;
