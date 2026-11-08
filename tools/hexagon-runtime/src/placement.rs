//! Immutable explicit placement for a bounded partial-graph experiment.

use std::{fmt::Write, sync::Arc};

use bonsai_prism_runtime::{
    kernels::{Pq2ProjectionExecutor, Q8Block},
    model::{Model, Tensor},
};
use bonsai_ternary::Q8KBlock;
use sha2::{Digest, Sha256};

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
        let mut digest = Sha256::new();
        digest.update(b"prism-explicit-projection-placement-v1\0");
        hash_bytes(&mut digest, executor.identity().as_bytes())?;
        hash_bytes(&mut digest, model.identity.as_bytes())?;
        for selection in &selected {
            let tensor = &selection.tensor;
            hash_bytes(&mut digest, tensor.name.as_bytes())?;
            digest.update(tensor.kind.to_le_bytes());
            for &dimension in &tensor.dimensions {
                digest.update(u64::try_from(dimension)?.to_le_bytes());
            }
            digest.update([u8::from(selection.operations.is_some())]);
            digest.update(selection.operations.unwrap_or(0).to_le_bytes());
        }
        let mut identity = String::new();
        identity.try_reserve_exact(96)?;
        identity.push_str("hexagon-explicit-placement-v1:");
        for byte in digest.finalize() {
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

fn hash_bytes(digest: &mut Sha256, bytes: &[u8]) -> Result<()> {
    digest.update(u64::try_from(bytes.len())?.to_le_bytes());
    digest.update(bytes);
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
        return Err(invalid("invalid hybrid NPU+CPU prefill pipeline parameters"));
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

#[cfg(test)]
mod tests;
