//! Composite NPU + GPU projection executor and hybrid prefill coordinator.

use bonsai_native_hash::Hasher;
use bonsai_prism_runtime::{
    kernels::{Pq2ProjectionExecutor, Q8Block},
    model::{Tensor, TensorBytes},
};
use bonsai_ternary::Q8KBlock;

use crate::{
    HeterogeneousPlacementPlan, HybridPrefillSchedule, HybridPrefillStep, Result, invalid,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RouteTarget {
    Npu,
    Gpu,
}

/// Telemetry counters and latched fault state for [`CompositeProjectionExecutor`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CompositeTelemetry {
    /// Completed projections routed to the Hexagon NPU child executor.
    pub npu_projections: u64,
    /// Completed projections routed to the Adreno GPU child executor.
    pub gpu_projections: u64,
    /// Whether either child executor has faulted; latches `true` with no CPU fallback.
    pub faulted: bool,
}

const fn valid_component_identity(identity: &str) -> bool {
    !identity.is_empty() && identity.len() <= 256 && identity.is_ascii()
}

fn layer_index(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("blk.")?;
    let (digits, _) = rest.split_once('.')?;
    if digits.is_empty()
        || !digits.bytes().all(|byte| byte.is_ascii_digit())
        || (digits.len() > 1 && digits.starts_with('0'))
    {
        return None;
    }
    digits.parse::<usize>().ok()
}

struct CompositeCore {
    plan_npu_layers: usize,
    plan_total_layers: usize,
    identity: String,
    telemetry: CompositeTelemetry,
}

impl CompositeCore {
    fn new(
        plan: &HeterogeneousPlacementPlan,
        npu_identity: &str,
        gpu_identity: &str,
    ) -> Result<Self> {
        if plan.cpu_layers != 0
            || plan.total_layers == 0
            || plan.npu_layers + plan.gpu_layers != plan.total_layers
            || plan.layers.len() != plan.total_layers
            || !valid_component_identity(&plan.identity)
            || !valid_component_identity(npu_identity)
            || !valid_component_identity(gpu_identity)
        {
            return Err(invalid("invalid composite projection placement"));
        }
        let mut digest = Hasher::new()?;
        digest.update(b"prism-composite-projection-v1\0")?;
        digest.update(&u64::try_from(plan.npu_layers)?.to_le_bytes())?;
        digest.update(&u64::try_from(plan.total_layers)?.to_le_bytes())?;
        for component in [plan.identity.as_str(), npu_identity, gpu_identity] {
            digest.update(&u64::try_from(component.len())?.to_le_bytes())?;
            digest.update(component.as_bytes())?;
        }
        let identity = crate::finish_sha256_prefixed_hex(digest, "composite-projection-v1:")?;
        Ok(Self {
            plan_npu_layers: plan.npu_layers,
            plan_total_layers: plan.total_layers,
            identity,
            telemetry: CompositeTelemetry::default(),
        })
    }

    fn route(&self, name: &str) -> Option<RouteTarget> {
        let layer = layer_index(name)?;
        if layer < self.plan_npu_layers {
            Some(RouteTarget::Npu)
        } else if layer < self.plan_total_layers {
            Some(RouteTarget::Gpu)
        } else {
            None
        }
    }

    fn group_target(&self, tensors: &[&Tensor]) -> Option<RouteTarget> {
        if self.telemetry.faulted {
            return None;
        }
        let first = tensors.first()?;
        let target = self.route(&first.name)?;
        if tensors
            .iter()
            .skip(1)
            .all(|tensor| self.route(&tensor.name) == Some(target))
        {
            Some(target)
        } else {
            None
        }
    }

    fn single_target(&self, name: &str) -> Option<RouteTarget> {
        if self.telemetry.faulted {
            return None;
        }
        self.route(name)
    }

    const fn record_success(&mut self, target: RouteTarget, count: u64) {
        match target {
            RouteTarget::Npu => self.telemetry.npu_projections += count,
            RouteTarget::Gpu => self.telemetry.gpu_projections += count,
        }
    }

    fn finish_dispatch<T>(
        &mut self,
        target: RouteTarget,
        count: u64,
        result: Result<T>,
    ) -> Result<T> {
        match result {
            Ok(value) => {
                self.record_success(target, count);
                Ok(value)
            }
            Err(error) => {
                self.telemetry.faulted = true;
                Err(error)
            }
        }
    }
}

fn clear_vec_slice(outputs: &mut [Vec<f32>]) {
    for output in outputs.iter_mut() {
        output.clear();
    }
}

/// Strict two-processor [`Pq2ProjectionExecutor`] dispatching prefix layers to a borrowed
/// NPU executor and overflow layers to a borrowed GPU executor according to a
/// [`HeterogeneousPlacementPlan`].
pub type CompositeProjectionExecutor<'a> = OwnedCompositeProjectionExecutor<
    &'a mut dyn Pq2ProjectionExecutor,
    &'a mut dyn Pq2ProjectionExecutor,
>;

/// Owned strict two-processor [`Pq2ProjectionExecutor`] dispatching prefix layers
/// to an owned NPU executor and overflow layers to an owned GPU executor.
pub struct OwnedCompositeProjectionExecutor<N, G> {
    core: CompositeCore,
    npu: N,
    gpu: G,
}

impl<N: Pq2ProjectionExecutor, G: Pq2ProjectionExecutor> OwnedCompositeProjectionExecutor<N, G> {
    /// Construct an owned composite NPU + GPU executor from an explicit placement plan.
    ///
    /// # Errors
    /// Rejects plans with non-zero CPU layers, empty/inconsistent layer tables, or
    /// invalid child executor identities.
    pub fn new(plan: &HeterogeneousPlacementPlan, npu: N, gpu: G) -> Result<Self> {
        let core = CompositeCore::new(plan, npu.identity(), gpu.identity())?;
        Ok(Self { core, npu, gpu })
    }

    /// Snapshot cumulative NPU/GPU projection counts and latched fault status.
    #[must_use]
    pub const fn telemetry(&self) -> CompositeTelemetry {
        self.core.telemetry
    }
}

impl<N: Pq2ProjectionExecutor, G: Pq2ProjectionExecutor> Pq2ProjectionExecutor
    for OwnedCompositeProjectionExecutor<N, G>
{
    fn identity(&self) -> &str {
        &self.core.identity
    }

    fn operation_capacity(&self) -> Option<u64> {
        if self.core.telemetry.faulted {
            Some(0)
        } else {
            self.npu
                .operation_capacity()
                .zip(self.gpu.operation_capacity())
                .and_then(|(left, right)| left.checked_add(right))
        }
    }

    fn operation_count(&self, tensor: &Tensor) -> Option<u64> {
        match self.core.single_target(&tensor.name)? {
            RouteTarget::Npu => self.npu.operation_count(tensor),
            RouteTarget::Gpu => self.gpu.operation_count(tensor),
        }
    }

    fn supports(&self, tensor: &Tensor) -> bool {
        match self.core.single_target(&tensor.name) {
            Some(RouteTarget::Npu) => self.npu.supports(tensor),
            Some(RouteTarget::Gpu) => self.gpu.supports(tensor),
            None => false,
        }
    }

    fn supports_q8(&self, tensor: &Tensor) -> bool {
        match self.core.single_target(&tensor.name) {
            Some(RouteTarget::Npu) => self.npu.supports_q8(tensor),
            Some(RouteTarget::Gpu) => self.gpu.supports_q8(tensor),
            None => false,
        }
    }

    fn supports_f32(&self, tensor: &Tensor) -> bool {
        match self.core.single_target(&tensor.name) {
            Some(RouteTarget::Npu) => self.npu.supports_f32(tensor),
            Some(RouteTarget::Gpu) => self.gpu.supports_f32(tensor),
            None => false,
        }
    }

    fn supports_f32_group(&self, tensors: &[&Tensor]) -> bool {
        match self.core.group_target(tensors) {
            Some(RouteTarget::Npu) => self.npu.supports_f32_group(tensors),
            Some(RouteTarget::Gpu) => self.gpu.supports_f32_group(tensors),
            None => false,
        }
    }

    fn project(
        &mut self,
        tensor: &Tensor,
        activation: &[Q8KBlock],
        fused: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        output.clear();
        let target = self
            .core
            .single_target(&tensor.name)
            .filter(|&target| match target {
                RouteTarget::Npu => self.npu.supports(tensor),
                RouteTarget::Gpu => self.gpu.supports(tensor),
            })
            .ok_or_else(|| invalid("unavailable composite Q8_K projection placement"))?;
        let result = match target {
            RouteTarget::Npu => self.npu.project(tensor, activation, fused, output),
            RouteTarget::Gpu => self.gpu.project(tensor, activation, fused, output),
        };
        if result.is_err() {
            output.clear();
        }
        self.core.finish_dispatch(target, 1, result)
    }

    fn project_q8(
        &mut self,
        tensor: &Tensor,
        activation: &[Q8Block],
        pinned: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        output.clear();
        let target = self
            .core
            .single_target(&tensor.name)
            .filter(|&target| match target {
                RouteTarget::Npu => self.npu.supports_q8(tensor),
                RouteTarget::Gpu => self.gpu.supports_q8(tensor),
            })
            .ok_or_else(|| invalid("unavailable composite Q8_0 projection placement"))?;
        let result = match target {
            RouteTarget::Npu => self.npu.project_q8(tensor, activation, pinned, output),
            RouteTarget::Gpu => self.gpu.project_q8(tensor, activation, pinned, output),
        };
        if result.is_err() {
            output.clear();
        }
        self.core.finish_dispatch(target, 1, result)
    }

    fn project_f32(
        &mut self,
        tensor: &Tensor,
        activation: &[f32],
        pinned: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        output.clear();
        let target = self
            .core
            .single_target(&tensor.name)
            .filter(|&target| match target {
                RouteTarget::Npu => self.npu.supports_f32(tensor),
                RouteTarget::Gpu => self.gpu.supports_f32(tensor),
            })
            .ok_or_else(|| invalid("unavailable composite F32 projection placement"))?;
        let result = match target {
            RouteTarget::Npu => self.npu.project_f32(tensor, activation, pinned, output),
            RouteTarget::Gpu => self.gpu.project_f32(tensor, activation, pinned, output),
        };
        if result.is_err() {
            output.clear();
        }
        self.core.finish_dispatch(target, 1, result)
    }

    fn project_f32_group(
        &mut self,
        tensors: &[&Tensor],
        activation: &[f32],
        pinned: bool,
        outputs: &mut [&mut Vec<f32>],
    ) -> Result<()> {
        for output in outputs.iter_mut() {
            output.clear();
        }
        let target = self
            .core
            .group_target(tensors)
            .filter(|&target| match target {
                RouteTarget::Npu => self.npu.supports_f32_group(tensors),
                RouteTarget::Gpu => self.gpu.supports_f32_group(tensors),
            })
            .ok_or_else(|| invalid("unavailable composite F32 group projection placement"))?;
        let count = u64::try_from(tensors.len())?;
        let result = match target {
            RouteTarget::Npu => self
                .npu
                .project_f32_group(tensors, activation, pinned, outputs),
            RouteTarget::Gpu => self
                .gpu
                .project_f32_group(tensors, activation, pinned, outputs),
        };
        if result.is_err() {
            for output in outputs.iter_mut() {
                output.clear();
            }
        }
        self.core.finish_dispatch(target, count, result)
    }

    fn project_f32_tokens(
        &mut self,
        tensor: &Tensor,
        activations: &[Vec<f32>],
        pinned: bool,
        outputs: &mut [Vec<f32>],
    ) -> Result<()> {
        clear_vec_slice(outputs);
        if activations.len() != outputs.len() {
            return Err(invalid("mismatched F32 token projection outputs"));
        }
        let target = self
            .core
            .single_target(&tensor.name)
            .filter(|&target| match target {
                RouteTarget::Npu => self.npu.supports_f32(tensor),
                RouteTarget::Gpu => self.gpu.supports_f32(tensor),
            })
            .ok_or_else(|| invalid("unavailable composite F32 token projection placement"))?;
        let count = u64::try_from(activations.len())?;
        let result = match target {
            RouteTarget::Npu => self
                .npu
                .project_f32_tokens(tensor, activations, pinned, outputs),
            RouteTarget::Gpu => self
                .gpu
                .project_f32_tokens(tensor, activations, pinned, outputs),
        };
        if result.is_err() {
            clear_vec_slice(outputs);
        }
        self.core.finish_dispatch(target, count, result)
    }

    fn project_f32_pair_tokens(
        &mut self,
        tensors: [&Tensor; 2],
        activations: &[Vec<f32>],
        pinned: bool,
        outputs_a: &mut [Vec<f32>],
        outputs_b: &mut [Vec<f32>],
    ) -> Result<()> {
        clear_vec_slice(outputs_a);
        clear_vec_slice(outputs_b);
        if activations.len() != outputs_a.len() || activations.len() != outputs_b.len() {
            return Err(invalid("mismatched F32 pair token projection outputs"));
        }
        let target = self
            .core
            .group_target(&tensors)
            .filter(|&target| match target {
                RouteTarget::Npu => self.npu.supports_f32_group(&tensors),
                RouteTarget::Gpu => self.gpu.supports_f32_group(&tensors),
            })
            .ok_or_else(|| invalid("unavailable composite F32 pair token projection placement"))?;
        let count = u64::try_from(activations.len().saturating_mul(2))?;
        let result = match target {
            RouteTarget::Npu => {
                self.npu
                    .project_f32_pair_tokens(tensors, activations, pinned, outputs_a, outputs_b)
            }
            RouteTarget::Gpu => {
                self.gpu
                    .project_f32_pair_tokens(tensors, activations, pinned, outputs_a, outputs_b)
            }
        };
        if result.is_err() {
            clear_vec_slice(outputs_a);
            clear_vec_slice(outputs_b);
        }
        self.core.finish_dispatch(target, count, result)
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
        clear_vec_slice(outputs_a);
        clear_vec_slice(outputs_b);
        clear_vec_slice(outputs_c);
        if activations.len() != outputs_a.len()
            || activations.len() != outputs_b.len()
            || activations.len() != outputs_c.len()
        {
            return Err(invalid("mismatched F32 triple token projection outputs"));
        }
        let target = self
            .core
            .group_target(&tensors)
            .filter(|&target| match target {
                RouteTarget::Npu => self.npu.supports_f32_group(&tensors),
                RouteTarget::Gpu => self.gpu.supports_f32_group(&tensors),
            })
            .ok_or_else(|| {
                invalid("unavailable composite F32 triple token projection placement")
            })?;
        let count = u64::try_from(activations.len().saturating_mul(3))?;
        let result = match target {
            RouteTarget::Npu => self.npu.project_f32_triple_tokens(
                tensors,
                activations,
                pinned,
                outputs_a,
                outputs_b,
                outputs_c,
            ),
            RouteTarget::Gpu => self.gpu.project_f32_triple_tokens(
                tensors,
                activations,
                pinned,
                outputs_a,
                outputs_b,
                outputs_c,
            ),
        };
        if result.is_err() {
            clear_vec_slice(outputs_a);
            clear_vec_slice(outputs_b);
            clear_vec_slice(outputs_c);
        }
        self.core.finish_dispatch(target, count, result)
    }

    fn supports_f32_ffn_swiglu(&self, gate: &Tensor, up: &Tensor) -> bool {
        match self.core.group_target(&[gate, up]) {
            Some(RouteTarget::Npu) => self.npu.supports_f32_ffn_swiglu(gate, up),
            Some(RouteTarget::Gpu) => self.gpu.supports_f32_ffn_swiglu(gate, up),
            None => false,
        }
    }

    fn project_f32_ffn_swiglu(
        &mut self,
        gate: &Tensor,
        up: &Tensor,
        activation: &[f32],
        pinned: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        output.clear();
        let target = self
            .core
            .group_target(&[gate, up])
            .filter(|&target| match target {
                RouteTarget::Npu => self.npu.supports_f32_ffn_swiglu(gate, up),
                RouteTarget::Gpu => self.gpu.supports_f32_ffn_swiglu(gate, up),
            })
            .ok_or_else(|| invalid("unavailable composite F32 FFN SwiGLU placement"))?;
        let result = match target {
            RouteTarget::Npu => self
                .npu
                .project_f32_ffn_swiglu(gate, up, activation, pinned, output),
            RouteTarget::Gpu => self
                .gpu
                .project_f32_ffn_swiglu(gate, up, activation, pinned, output),
        };
        if result.is_err() {
            output.clear();
        }
        self.core.finish_dispatch(target, 2, result)
    }

    fn project_f32_ffn_swiglu_tokens(
        &mut self,
        gate: &Tensor,
        up: &Tensor,
        activations: &[Vec<f32>],
        pinned: bool,
        outputs: &mut [Vec<f32>],
    ) -> Result<()> {
        clear_vec_slice(outputs);
        if activations.len() != outputs.len() {
            return Err(invalid(
                "mismatched F32 FFN SwiGLU token projection outputs",
            ));
        }
        let target = self
            .core
            .group_target(&[gate, up])
            .filter(|&target| match target {
                RouteTarget::Npu => self.npu.supports_f32_ffn_swiglu(gate, up),
                RouteTarget::Gpu => self.gpu.supports_f32_ffn_swiglu(gate, up),
            })
            .ok_or_else(|| invalid("unavailable composite F32 FFN SwiGLU token placement"))?;
        let count = u64::try_from(activations.len().saturating_mul(2))?;
        let result = match target {
            RouteTarget::Npu => {
                self.npu
                    .project_f32_ffn_swiglu_tokens(gate, up, activations, pinned, outputs)
            }
            RouteTarget::Gpu => {
                self.gpu
                    .project_f32_ffn_swiglu_tokens(gate, up, activations, pinned, outputs)
            }
        };
        if result.is_err() {
            clear_vec_slice(outputs);
        }
        self.core.finish_dispatch(target, count, result)
    }

    fn supports_f32_ffn_chain(&self, tensors: [&Tensor; 3]) -> bool {
        match self.core.group_target(&tensors) {
            Some(RouteTarget::Npu) => self.npu.supports_f32_ffn_chain(tensors),
            Some(RouteTarget::Gpu) => self.gpu.supports_f32_ffn_chain(tensors),
            None => false,
        }
    }

    fn supports_tile_tokens(&self) -> bool {
        !self.core.telemetry.faulted
            && (self.core.plan_npu_layers == 0 || self.npu.supports_tile_tokens())
            && (self.core.plan_npu_layers == self.core.plan_total_layers
                || self.gpu.supports_tile_tokens())
    }

    fn release_host_weight_binding(
        &mut self,
        previous_tensor: &Tensor,
        released_data: &TensorBytes,
    ) -> bool {
        match self.core.single_target(&previous_tensor.name) {
            Some(RouteTarget::Npu) => self
                .npu
                .release_host_weight_binding(previous_tensor, released_data),
            Some(RouteTarget::Gpu) => self
                .gpu
                .release_host_weight_binding(previous_tensor, released_data),
            None => false,
        }
    }

    fn has_released_host_weight(&self, tensor: &Tensor) -> bool {
        match self.core.single_target(&tensor.name) {
            Some(RouteTarget::Npu) => self.npu.has_released_host_weight(tensor),
            Some(RouteTarget::Gpu) => self.gpu.has_released_host_weight(tensor),
            None => false,
        }
    }

    fn project_f32_ffn_chain(
        &mut self,
        tensors: [&Tensor; 3],
        activation: &[f32],
        residual: Option<&[f32]>,
        hadamard_block_size: Option<u32>,
        pinned: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        output.clear();
        let target = self
            .core
            .group_target(&tensors)
            .filter(|&target| match target {
                RouteTarget::Npu => self.npu.supports_f32_ffn_chain(tensors),
                RouteTarget::Gpu => self.gpu.supports_f32_ffn_chain(tensors),
            })
            .ok_or_else(|| invalid("unavailable composite F32 FFN chain placement"))?;
        let result = match target {
            RouteTarget::Npu => self.npu.project_f32_ffn_chain(
                tensors,
                activation,
                residual,
                hadamard_block_size,
                pinned,
                output,
            ),
            RouteTarget::Gpu => self.gpu.project_f32_ffn_chain(
                tensors,
                activation,
                residual,
                hadamard_block_size,
                pinned,
                output,
            ),
        };
        if result.is_err() {
            output.clear();
        }
        self.core.finish_dispatch(target, 3, result)
    }

    fn supports_f32_ffn_chain_lora(&self, tensors: [&Tensor; 3]) -> bool {
        match self.core.group_target(&tensors) {
            Some(RouteTarget::Npu) => self.npu.supports_f32_ffn_chain_lora(tensors),
            Some(RouteTarget::Gpu) => self.gpu.supports_f32_ffn_chain_lora(tensors),
            None => false,
        }
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
    ) -> Result<f32> {
        output.clear();
        let target = self
            .core
            .group_target(&tensors)
            .filter(|&target| match target {
                RouteTarget::Npu => self.npu.supports_f32_ffn_chain_lora(tensors),
                RouteTarget::Gpu => self.gpu.supports_f32_ffn_chain_lora(tensors),
            })
            .ok_or_else(|| invalid("unavailable composite F32 FFN chain LoRA placement"))?;
        let result = match target {
            RouteTarget::Npu => self.npu.project_f32_ffn_chain_lora(
                tensors,
                activation,
                residual,
                hadamard_block_size,
                lora_a,
                pinned,
                output,
            ),
            RouteTarget::Gpu => self.gpu.project_f32_ffn_chain_lora(
                tensors,
                activation,
                residual,
                hadamard_block_size,
                lora_a,
                pinned,
                output,
            ),
        };
        if result.is_err() {
            output.clear();
        }
        self.core.finish_dispatch(target, 3, result)
    }

    fn project_f32_ffn_chain_tokens(
        &mut self,
        tensors: [&Tensor; 3],
        activations: &[Vec<f32>],
        residuals: Option<&[Vec<f32>]>,
        hadamard_block_size: Option<u32>,
        pinned: bool,
        outputs: &mut [Vec<f32>],
    ) -> Result<()> {
        clear_vec_slice(outputs);
        if activations.len() != outputs.len()
            || residuals.is_some_and(|res| res.len() != activations.len())
        {
            return Err(invalid("mismatched F32 FFN chain token projection outputs"));
        }
        let target = self
            .core
            .group_target(&tensors)
            .filter(|&target| match target {
                RouteTarget::Npu => self.npu.supports_f32_ffn_chain(tensors),
                RouteTarget::Gpu => self.gpu.supports_f32_ffn_chain(tensors),
            })
            .ok_or_else(|| invalid("unavailable composite F32 FFN chain token placement"))?;
        let count = u64::try_from(activations.len().saturating_mul(3))?;
        let result = match target {
            RouteTarget::Npu => self.npu.project_f32_ffn_chain_tokens(
                tensors,
                activations,
                residuals,
                hadamard_block_size,
                pinned,
                outputs,
            ),
            RouteTarget::Gpu => self.gpu.project_f32_ffn_chain_tokens(
                tensors,
                activations,
                residuals,
                hadamard_block_size,
                pinned,
                outputs,
            ),
        };
        if result.is_err() {
            clear_vec_slice(outputs);
        }
        self.core.finish_dispatch(target, count, result)
    }
}

/// Ordered state tracker advancing through a [`HybridPrefillSchedule`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HybridPrefillCoordinator {
    schedule: HybridPrefillSchedule,
    completed: usize,
    npu_steps: usize,
    gpu_steps: usize,
}

impl HybridPrefillCoordinator {
    /// Start coordinating the supplied double-buffered NPU + secondary prefill schedule.
    #[must_use]
    pub const fn new(schedule: HybridPrefillSchedule) -> Self {
        Self {
            schedule,
            completed: 0,
            npu_steps: 0,
            gpu_steps: 0,
        }
    }

    /// Number of schedule steps completed so far.
    #[must_use]
    pub const fn completed_steps(&self) -> usize {
        self.completed
    }

    /// Completed microbatch invocations on Stage 0 (Hexagon NPU).
    #[must_use]
    pub const fn npu_steps_completed(&self) -> usize {
        self.npu_steps
    }

    /// Completed microbatch invocations on Stage 1 (Adreno GPU / Oryon CPU overflow).
    #[must_use]
    pub const fn gpu_steps_completed(&self) -> usize {
        self.gpu_steps
    }

    /// Whether every step in the schedule has completed.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.completed == self.schedule.steps.len()
    }

    /// Advance the pipeline by one verified step index without skipping or replaying steps.
    ///
    /// # Errors
    /// Rejects out-of-order indices and calls after the final scheduled step.
    pub fn record_next_step(&mut self, expected_step_index: usize) -> Result<HybridPrefillStep> {
        if expected_step_index != self.completed || self.completed >= self.schedule.steps.len() {
            return Err(invalid("out-of-order hybrid prefill step"));
        }
        let step = self.schedule.steps[self.completed].clone();
        if step.npu_microbatch.is_some() {
            self.npu_steps += 1;
        }
        if step.cpu_microbatch.is_some() {
            self.gpu_steps += 1;
        }
        self.completed += 1;
        Ok(step)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{
        AcceleratorTarget, plan_hybrid_prefill_pipeline, plan_x1_v73_three_processor_placement,
    };

    struct MockExecutor {
        id: &'static str,
        capacity: Option<u64>,
        fail_on: Option<&'static str>,
        tile_tokens: bool,
        calls: usize,
    }

    impl MockExecutor {
        fn check_fail(&self, tensors: &[&Tensor], message: &str) -> Result<()> {
            if tensors
                .iter()
                .any(|tensor| self.fail_on == Some(tensor.name.as_str()))
            {
                return Err(invalid(message));
            }
            Ok(())
        }
    }

    impl Pq2ProjectionExecutor for MockExecutor {
        fn identity(&self) -> &str {
            self.id
        }

        fn operation_capacity(&self) -> Option<u64> {
            self.capacity
        }

        fn operation_count(&self, _tensor: &Tensor) -> Option<u64> {
            Some(1)
        }

        fn supports(&self, _tensor: &Tensor) -> bool {
            true
        }

        fn supports_q8(&self, _tensor: &Tensor) -> bool {
            true
        }

        fn supports_f32(&self, _tensor: &Tensor) -> bool {
            true
        }

        fn supports_f32_group(&self, tensors: &[&Tensor]) -> bool {
            !tensors.is_empty()
        }

        fn supports_tile_tokens(&self) -> bool {
            self.tile_tokens
        }

        fn project(
            &mut self,
            tensor: &Tensor,
            _activation: &[Q8KBlock],
            _fused: bool,
            output: &mut Vec<f32>,
        ) -> Result<()> {
            self.check_fail(&[tensor], "injected child failure")?;
            self.calls += 1;
            output.push(1.0);
            Ok(())
        }

        fn project_q8(
            &mut self,
            tensor: &Tensor,
            _activation: &[Q8Block],
            _pinned: bool,
            output: &mut Vec<f32>,
        ) -> Result<()> {
            self.check_fail(&[tensor], "injected child failure")?;
            self.calls += 1;
            output.push(2.0);
            Ok(())
        }

        fn project_f32(
            &mut self,
            tensor: &Tensor,
            _activation: &[f32],
            _pinned: bool,
            output: &mut Vec<f32>,
        ) -> Result<()> {
            self.check_fail(&[tensor], "injected child failure")?;
            self.calls += 1;
            output.push(3.0);
            Ok(())
        }

        fn project_f32_group(
            &mut self,
            tensors: &[&Tensor],
            _activation: &[f32],
            _pinned: bool,
            outputs: &mut [&mut Vec<f32>],
        ) -> Result<()> {
            self.check_fail(tensors, "injected group failure")?;
            for output in outputs.iter_mut() {
                output.push(4.0);
            }
            self.calls += tensors.len();
            Ok(())
        }

        fn project_f32_tokens(
            &mut self,
            tensor: &Tensor,
            activations: &[Vec<f32>],
            _pinned: bool,
            outputs: &mut [Vec<f32>],
        ) -> Result<()> {
            self.check_fail(&[tensor], "injected token failure")?;
            for output in outputs.iter_mut() {
                output.push(5.0);
            }
            self.calls += activations.len();
            Ok(())
        }

        fn project_f32_pair_tokens(
            &mut self,
            tensors: [&Tensor; 2],
            activations: &[Vec<f32>],
            _pinned: bool,
            outputs_a: &mut [Vec<f32>],
            outputs_b: &mut [Vec<f32>],
        ) -> Result<()> {
            self.check_fail(&tensors, "injected pair token failure")?;
            for (out_a, out_b) in outputs_a.iter_mut().zip(outputs_b.iter_mut()) {
                out_a.push(6.0);
                out_b.push(6.5);
            }
            self.calls += activations.len() * 2;
            Ok(())
        }

        fn project_f32_triple_tokens(
            &mut self,
            tensors: [&Tensor; 3],
            activations: &[Vec<f32>],
            _pinned: bool,
            outputs_a: &mut [Vec<f32>],
            outputs_b: &mut [Vec<f32>],
            outputs_c: &mut [Vec<f32>],
        ) -> Result<()> {
            self.check_fail(&tensors, "injected triple token failure")?;
            for ((out_a, out_b), out_c) in outputs_a
                .iter_mut()
                .zip(outputs_b.iter_mut())
                .zip(outputs_c.iter_mut())
            {
                out_a.push(7.0);
                out_b.push(7.5);
                out_c.push(8.0);
            }
            self.calls += activations.len() * 3;
            Ok(())
        }

        fn supports_f32_ffn_swiglu(&self, _gate: &Tensor, _up: &Tensor) -> bool {
            true
        }

        fn project_f32_ffn_swiglu(
            &mut self,
            gate: &Tensor,
            up: &Tensor,
            _activation: &[f32],
            _pinned: bool,
            output: &mut Vec<f32>,
        ) -> Result<()> {
            self.check_fail(&[gate, up], "injected swiglu failure")?;
            output.push(9.0);
            self.calls += 2;
            Ok(())
        }

        fn project_f32_ffn_swiglu_tokens(
            &mut self,
            gate: &Tensor,
            up: &Tensor,
            activations: &[Vec<f32>],
            _pinned: bool,
            outputs: &mut [Vec<f32>],
        ) -> Result<()> {
            self.check_fail(&[gate, up], "injected swiglu token failure")?;
            for output in outputs.iter_mut() {
                output.push(9.5);
            }
            self.calls += activations.len() * 2;
            Ok(())
        }

        fn supports_f32_ffn_chain(&self, _tensors: [&Tensor; 3]) -> bool {
            true
        }

        fn project_f32_ffn_chain(
            &mut self,
            tensors: [&Tensor; 3],
            _activation: &[f32],
            _residual: Option<&[f32]>,
            _hadamard_block_size: Option<u32>,
            _pinned: bool,
            output: &mut Vec<f32>,
        ) -> Result<()> {
            self.check_fail(&tensors, "injected ffn chain failure")?;
            output.push(10.0);
            self.calls += 3;
            Ok(())
        }

        fn project_f32_ffn_chain_tokens(
            &mut self,
            tensors: [&Tensor; 3],
            activations: &[Vec<f32>],
            _residuals: Option<&[Vec<f32>]>,
            _hadamard_block_size: Option<u32>,
            _pinned: bool,
            outputs: &mut [Vec<f32>],
        ) -> Result<()> {
            self.check_fail(&tensors, "injected ffn chain token failure")?;
            for output in outputs.iter_mut() {
                output.push(10.5);
            }
            self.calls += activations.len() * 3;
            Ok(())
        }

        fn supports_f32_ffn_chain_lora(&self, _tensors: [&Tensor; 3]) -> bool {
            true
        }

        fn project_f32_ffn_chain_lora(
            &mut self,
            tensors: [&Tensor; 3],
            _activation: &[f32],
            _residual: Option<&[f32]>,
            _hadamard_block_size: Option<u32>,
            lora_a: &[f32],
            _pinned: bool,
            output: &mut Vec<f32>,
        ) -> Result<f32> {
            self.check_fail(&tensors, "injected ffn chain lora failure")?;
            output.push(11.0);
            self.calls += 3;
            Ok(lora_a.first().copied().unwrap_or(1.25))
        }
    }

    fn tensor(name: &str) -> Tensor {
        Tensor {
            name: name.to_owned(),
            dimensions: vec![128, 4],
            kind: 142,
            data: Arc::<[u8]>::from([]).into(),
        }
    }

    #[test]
    fn x1_v73_placement_assigns_27_npu_and_37_adreno_layers() -> Result<()> {
        assert_eq!(crate::X1_V73_MAX_NPU_LAYERS, 27);
        let plan =
            plan_x1_v73_three_processor_placement(64, 4, AcceleratorTarget::AdrenoGpuOpenCl)?;
        assert_eq!(plan.npu_layers, 27);
        assert_eq!(plan.gpu_layers, 37);
        assert_eq!(plan.cpu_layers, 0);
        assert_eq!(plan.npu_q2_projections, 168);
        assert_eq!(plan.gpu_q2_projections, 232);
        assert_eq!(plan.cpu_q2_projections, 0);
        assert!(plan_x1_v73_three_processor_placement(16, 4, AcceleratorTarget::OryonCpu).is_err());
        Ok(())
    }

    #[test]
    fn composite_executor_routes_groups_tracks_telemetry_and_latches_fault() -> Result<()> {
        let plan =
            plan_x1_v73_three_processor_placement(64, 4, AcceleratorTarget::AdrenoGpuOpenCl)?;
        let mut npu = MockExecutor {
            id: "mock-htp-v73",
            capacity: Some(100),
            fail_on: None,
            tile_tokens: false,
            calls: 0,
        };
        let mut gpu = MockExecutor {
            id: "mock-adreno-x1-85",
            capacity: Some(200),
            fail_on: Some("blk.27.ffn_down.weight"),
            tile_tokens: true,
            calls: 0,
        };
        let mut composite = CompositeProjectionExecutor::new(&plan, &mut npu, &mut gpu)?;
        assert_eq!(composite.operation_capacity(), Some(300));
        assert!(!composite.supports_tile_tokens());

        let t0 = tensor("blk.0.attn_qkv.weight");
        let t26 = tensor("blk.26.ffn_up.weight");
        let t27_ok = tensor("blk.27.ffn_up.weight");
        let t63 = tensor("blk.63.ffn_gate.weight");
        let t27_fail = tensor("blk.27.ffn_down.weight");
        let out_of_range = tensor("blk.64.ffn_up.weight");

        let mut out = Vec::new();
        composite.project_f32(&t0, &[], true, &mut out)?;
        assert_eq!(out, [3.0]);
        composite.project_f32(&t27_ok, &[], true, &mut out)?;
        assert_eq!(out, [3.0]);
        assert!(!composite.supports_f32_group(&[&t26, &t27_ok]));
        assert!(composite.supports_f32_group(&[&t27_ok, &t63]));
        assert!(!composite.supports(&out_of_range));

        let mut g0 = Vec::new();
        let mut g1 = Vec::new();
        composite.project_f32_group(&[&t27_ok, &t63], &[], true, &mut [&mut g0, &mut g1])?;
        assert_eq!((g0, g1), (vec![4.0], vec![4.0]));
        assert_eq!(composite.telemetry().npu_projections, 1);
        assert_eq!(composite.telemetry().gpu_projections, 3);
        assert!(!composite.telemetry().faulted);

        assert!(
            composite
                .project_f32(&t27_fail, &[], true, &mut out)
                .is_err()
        );
        assert_eq!(out, [0.0_f32; 0]);
        assert!(composite.telemetry().faulted);
        assert_eq!(composite.operation_capacity(), Some(0));
        assert!(!composite.supports_f32(&t0));
        assert!(composite.project_f32(&t0, &[], true, &mut out).is_err());
        Ok(())
    }

    #[test]
    fn composite_forwards_tile_token_projections() -> Result<()> {
        let plan =
            plan_x1_v73_three_processor_placement(64, 4, AcceleratorTarget::AdrenoGpuVulkan)?;
        let mut npu = MockExecutor {
            id: "mock-htp",
            capacity: Some(100),
            fail_on: None,
            tile_tokens: true,
            calls: 0,
        };
        let mut gpu = MockExecutor {
            id: "mock-adreno",
            capacity: Some(200),
            fail_on: Some("blk.30.ffn_down.weight"),
            tile_tokens: true,
            calls: 0,
        };
        let mut composite = CompositeProjectionExecutor::new(&plan, &mut npu, &mut gpu)?;
        assert!(composite.supports_tile_tokens());

        let npu_gate = tensor("blk.3.ffn_gate.weight");
        let gpu_gate = tensor("blk.29.ffn_gate.weight");
        let gpu_up = tensor("blk.29.ffn_up.weight");
        let gpu_down = tensor("blk.29.ffn_down.weight");

        let acts = vec![vec![1.0], vec![2.0]];
        let mut out_a = vec![Vec::new(); 2];
        let mut out_b = vec![Vec::new(); 2];
        let mut out_c = vec![Vec::new(); 2];
        composite.project_f32_tokens(&npu_gate, &acts, true, &mut out_a)?;
        assert_eq!(out_a, [vec![5.0], vec![5.0]]);
        composite.project_f32_pair_tokens(
            [&gpu_gate, &gpu_up],
            &acts,
            true,
            &mut out_a,
            &mut out_b,
        )?;
        assert_eq!(out_a, [vec![6.0], vec![6.0]]);
        assert_eq!(out_b, [vec![6.5], vec![6.5]]);
        composite.project_f32_triple_tokens(
            [&gpu_gate, &gpu_up, &gpu_down],
            &acts,
            true,
            &mut out_a,
            &mut out_b,
            &mut out_c,
        )?;
        assert_eq!(out_c, [vec![8.0], vec![8.0]]);
        assert_eq!(composite.telemetry().npu_projections, 2);
        assert_eq!(composite.telemetry().gpu_projections, 10);
        Ok(())
    }

    #[test]
    fn composite_forwards_ffn_swiglu_chain_and_lora_routes() -> Result<()> {
        let plan =
            plan_x1_v73_three_processor_placement(64, 4, AcceleratorTarget::AdrenoGpuVulkan)?;
        let mut npu = MockExecutor {
            id: "mock-htp",
            capacity: Some(100),
            fail_on: None,
            tile_tokens: true,
            calls: 0,
        };
        let mut gpu = MockExecutor {
            id: "mock-adreno",
            capacity: Some(200),
            fail_on: Some("blk.30.ffn_down.weight"),
            tile_tokens: true,
            calls: 0,
        };
        let mut composite = CompositeProjectionExecutor::new(&plan, &mut npu, &mut gpu)?;

        let npu_gate = tensor("blk.3.ffn_gate.weight");
        let npu_up = tensor("blk.3.ffn_up.weight");
        let npu_down = tensor("blk.3.ffn_down.weight");
        let gpu_gate = tensor("blk.29.ffn_gate.weight");
        let gpu_up = tensor("blk.29.ffn_up.weight");
        let gpu_down = tensor("blk.29.ffn_down.weight");
        let gpu_fail_down = tensor("blk.30.ffn_down.weight");

        assert!(composite.supports_f32_ffn_swiglu(&npu_gate, &npu_up));
        assert!(!composite.supports_f32_ffn_swiglu(&npu_gate, &gpu_up));
        assert!(composite.supports_f32_ffn_chain([&gpu_gate, &gpu_up, &gpu_down]));
        assert!(!composite.supports_f32_ffn_chain([&npu_gate, &gpu_up, &gpu_down]));
        assert!(composite.supports_f32_ffn_chain_lora([&gpu_gate, &gpu_up, &gpu_down]));

        let acts = vec![vec![1.0], vec![2.0]];
        let mut out_a = vec![Vec::new(); 2];
        let mut single = Vec::new();
        composite.project_f32_ffn_swiglu(&npu_gate, &npu_up, &[], true, &mut single)?;
        assert_eq!(single, [9.0]);
        composite.project_f32_ffn_swiglu_tokens(&gpu_gate, &gpu_up, &acts, true, &mut out_a)?;
        assert_eq!(out_a, [vec![9.5], vec![9.5]]);

        composite.project_f32_ffn_chain(
            [&npu_gate, &npu_up, &npu_down],
            &[],
            None,
            Some(128),
            true,
            &mut single,
        )?;
        assert_eq!(single, [10.0]);
        composite.project_f32_ffn_chain_tokens(
            [&gpu_gate, &gpu_up, &gpu_down],
            &acts,
            None,
            Some(128),
            true,
            &mut out_a,
        )?;
        assert_eq!(out_a, [vec![10.5], vec![10.5]]);

        let dot = composite.project_f32_ffn_chain_lora(
            [&gpu_gate, &gpu_up, &gpu_down],
            &[],
            None,
            Some(128),
            &[2.5],
            true,
            &mut single,
        )?;
        assert_eq!(dot, 2.5);
        assert_eq!(single, [11.0]);
        assert_eq!(composite.telemetry().npu_projections, 5);
        assert_eq!(composite.telemetry().gpu_projections, 13);
        assert!(!composite.telemetry().faulted);

        let gpu_fail_gate = tensor("blk.30.ffn_gate.weight");
        let gpu_fail_up = tensor("blk.30.ffn_up.weight");
        assert!(
            composite
                .project_f32_ffn_chain_lora(
                    [&gpu_fail_gate, &gpu_fail_up, &gpu_fail_down],
                    &[],
                    None,
                    Some(128),
                    &[2.5],
                    true,
                    &mut single,
                )
                .is_err()
        );
        assert_eq!(single, [0.0_f32; 0]);
        assert!(composite.telemetry().faulted);
        assert!(!composite.supports_tile_tokens());
        assert!(!composite.supports_f32_ffn_chain([&npu_gate, &npu_up, &npu_down]));
        Ok(())
    }

    #[test]
    fn owned_composite_executor_forwards_and_tracks_telemetry() -> Result<()> {
        let plan =
            plan_x1_v73_three_processor_placement(64, 4, AcceleratorTarget::AdrenoGpuVulkan)?;
        let npu = MockExecutor {
            id: "mock-htp-owned",
            capacity: Some(50),
            fail_on: None,
            tile_tokens: true,
            calls: 0,
        };
        let gpu = MockExecutor {
            id: "mock-adreno-owned",
            capacity: Some(75),
            fail_on: None,
            tile_tokens: true,
            calls: 0,
        };
        let mut owned = OwnedCompositeProjectionExecutor::new(&plan, npu, gpu)?;
        assert_eq!(owned.operation_capacity(), Some(125));
        assert!(owned.supports_tile_tokens());

        let npu_gate = tensor("blk.1.ffn_gate.weight");
        let gpu_gate = tensor("blk.40.ffn_gate.weight");
        let gpu_up = tensor("blk.40.ffn_up.weight");
        let gpu_down = tensor("blk.40.ffn_down.weight");
        let mut out = Vec::new();
        owned.project_f32(&npu_gate, &[], true, &mut out)?;
        assert_eq!(out, [3.0]);
        let dot = owned.project_f32_ffn_chain_lora(
            [&gpu_gate, &gpu_up, &gpu_down],
            &[],
            None,
            None,
            &[3.75],
            true,
            &mut out,
        )?;
        assert_eq!(dot, 3.75);
        assert_eq!(out, [11.0]);
        assert_eq!(owned.telemetry().npu_projections, 1);
        assert_eq!(owned.telemetry().gpu_projections, 3);
        Ok(())
    }

    #[test]
    fn hybrid_prefill_coordinator_advances_in_order_and_rejects_skips() -> Result<()> {
        let schedule = plan_hybrid_prefill_pipeline(64, 27, 20, 8)?;
        let mut coordinator = HybridPrefillCoordinator::new(schedule);
        assert!(!coordinator.is_complete());
        assert!(coordinator.record_next_step(1).is_err());
        for step in 0..4 {
            let _ = coordinator.record_next_step(step)?;
        }
        assert!(coordinator.is_complete());
        assert_eq!(coordinator.completed_steps(), 4);
        assert_eq!(coordinator.npu_steps_completed(), 3);
        assert_eq!(coordinator.gpu_steps_completed(), 3);
        assert!(coordinator.record_next_step(4).is_err());
        Ok(())
    }

    struct CompositePtq1Submitter;

    impl hexagon_native_rpc::NativeQ2Submitter for CompositePtq1Submitter {
        fn submit(
            &mut self,
            columns: usize,
            rows: usize,
            weights: &[u8],
            activation: &[f32],
            output: &mut [f32],
        ) -> std::result::Result<(), hexagon_native_rpc::Error> {
            assert!(columns >= 128 && columns.is_multiple_of(128));
            assert!(rows >= 32 && rows.is_multiple_of(32));
            assert_eq!(weights.len(), rows * (columns / 128) * 36);
            assert_eq!(activation.len(), columns);
            output.fill(9.25);
            Ok(())
        }
    }

    fn tensor_with_dims(name: &str, kind: u32, columns: usize, rows: usize) -> Tensor {
        let block_bytes = if kind == 143 { 28 } else { 34 };
        let mut block = vec![0_u8; block_bytes];
        if kind == 143 {
            block[26..28].copy_from_slice(&0x3C00_u16.to_le_bytes());
        } else {
            block[0..2].copy_from_slice(&0x3C00_u16.to_le_bytes());
        }
        let blocks = rows * (columns / 128);
        let mut data = Vec::with_capacity(blocks * block_bytes);
        for _ in 0..blocks {
            data.extend_from_slice(&block);
        }
        Tensor {
            name: name.to_owned(),
            dimensions: vec![columns, rows],
            kind,
            data: Arc::<[u8]>::from(data).into(),
        }
    }

    #[test]
    fn ptq1_and_pq2_composite_routing_with_x1_v73_and_native_q2_npu() -> Result<()> {
        let plan =
            plan_x1_v73_three_processor_placement(64, 4, AcceleratorTarget::AdrenoGpuOpenCl)?;
        let limits = hexagon_native_rpc::NativeQ2ProjectionLimits::new(
            256,
            32,
            16,
            std::time::Duration::from_secs(4),
        )
        .map_err(|_| invalid("limits"))?;
        let mut npu = crate::NativeQ2Executor::new(CompositePtq1Submitter, limits, 1024 * 1024);
        let mut gpu = MockExecutor {
            id: "adreno-ptq1-gpu",
            capacity: Some(128),
            fail_on: None,
            tile_tokens: false,
            calls: 0,
        };
        let mut composite = CompositeProjectionExecutor::new(&plan, &mut npu, &mut gpu)?;

        let npu_ptq1 = tensor_with_dims("blk.0.ffn_up.weight", 143, 256, 32);
        let npu_pq2 = tensor_with_dims("blk.1.ffn_gate.weight", 142, 256, 32);
        let gpu_ptq1 = tensor_with_dims("blk.27.ffn_up.weight", 143, 256, 32);
        assert!(composite.supports_f32(&npu_ptq1));
        assert!(composite.supports_f32(&npu_pq2));
        assert!(composite.supports_f32(&gpu_ptq1));

        let mut out = Vec::new();
        composite.project_f32(&npu_ptq1, &vec![1.0_f32; 256], true, &mut out)?;
        assert_eq!(out, [9.25; 32]);
        composite.project_f32(&npu_pq2, &vec![1.0_f32; 256], true, &mut out)?;
        assert_eq!(out, [9.25; 32]);
        composite.project_f32(&gpu_ptq1, &vec![1.0_f32; 256], true, &mut out)?;
        assert_eq!(out, [3.0]);
        assert_eq!(composite.telemetry().npu_projections, 2);
        assert_eq!(composite.telemetry().gpu_projections, 1);
        assert!(!composite.telemetry().faulted);
        Ok(())
    }

    fn build_vtcm_ptq1_test_model() -> bonsai_prism_runtime::model::Model {
        let mut tensors = std::collections::BTreeMap::new();
        for layer_idx in 0..4_usize {
            let kind = if layer_idx.is_multiple_of(2) {
                143
            } else {
                142
            };
            let suffixes: &[&str] = if (layer_idx + 1).is_multiple_of(4) {
                &[
                    "ffn_up.weight",
                    "ffn_gate.weight",
                    "ffn_down.weight",
                    "attn_q.weight",
                    "attn_k.weight",
                    "attn_v.weight",
                    "attn_output.weight",
                ]
            } else {
                &[
                    "ffn_up.weight",
                    "ffn_gate.weight",
                    "ffn_down.weight",
                    "attn_qkv.weight",
                    "attn_gate.weight",
                    "ssm_out.weight",
                ]
            };
            for &suffix in suffixes {
                let name = format!("blk.{layer_idx}.{suffix}");
                let t = tensor_with_dims(&name, kind, 256, 32);
                tensors.insert(name, t);
            }
        }
        bonsai_prism_runtime::model::Model {
            identity: "composite-vtcm-ptq1-model".into(),
            metadata: std::collections::BTreeMap::new(),
            tensors,
        }
    }

    #[test]
    fn owned_composite_routes_vtcm_planned_ptq1_and_overflow_layers() -> Result<()> {
        let model = build_vtcm_ptq1_test_model();
        // 13 * 2304 = 29_952 bytes fits blk.0 (6 * 2304 = 13_824) and blk.1 (6 * 2304 = 13_824 -> 27_648 total),
        // leaving 2_304 < 13_824 so blk.2..4 overflow to GPU cleanly.
        let vtcm_plan =
            crate::plan_native_q2_layers_for_budget(&model, 73, 29_952, 1, 4 * 1024 * 1024)?;
        assert_eq!(vtcm_plan.admitted_layers, 2);
        assert_eq!(vtcm_plan.npu_projections.len(), 12);
        assert_eq!(vtcm_plan.overflow_projections.len(), 13);

        let het_plan = crate::plan_three_processor_placement(
            vtcm_plan.total_layers,
            vtcm_plan.admitted_layers,
            vtcm_plan.total_layers - vtcm_plan.admitted_layers,
            4,
            AcceleratorTarget::AdrenoGpuOpenCl,
        )?;
        let limits = hexagon_native_rpc::NativeQ2ProjectionLimits::new(
            256,
            32,
            16,
            std::time::Duration::from_secs(4),
        )
        .map_err(|_| invalid("limits"))?;
        let npu_names: Vec<&str> = vtcm_plan
            .npu_projections
            .iter()
            .map(String::as_str)
            .collect();
        let mut npu = crate::NativeQ2Executor::new(CompositePtq1Submitter, limits, 1024 * 1024);
        let mut placed_npu = crate::PlacedProjectionExecutor::new(&mut npu, &model, &npu_names)?;
        assert!(placed_npu.supports_f32(&model.tensors["blk.0.ffn_up.weight"]));
        assert!(placed_npu.supports_f32(&model.tensors["blk.1.ffn_gate.weight"]));
        assert!(!placed_npu.supports_f32(&model.tensors["blk.2.ffn_up.weight"]));
        let mut gpu = MockExecutor {
            id: "adreno-overflow",
            capacity: Some(64),
            fail_on: None,
            tile_tokens: false,
            calls: 0,
        };
        let mut composite = CompositeProjectionExecutor::new(&het_plan, &mut placed_npu, &mut gpu)?;
        let mut out = Vec::new();
        composite.project_f32(
            &model.tensors["blk.0.ffn_up.weight"],
            &vec![1.0_f32; 256],
            true,
            &mut out,
        )?;
        assert_eq!(out, [9.25; 32]);
        composite.project_f32(
            &model.tensors["blk.1.ffn_gate.weight"],
            &vec![1.0_f32; 256],
            true,
            &mut out,
        )?;
        assert_eq!(out, [9.25; 32]);
        composite.project_f32(
            &model.tensors["blk.2.ffn_up.weight"],
            &vec![1.0_f32; 256],
            true,
            &mut out,
        )?;
        assert_eq!(out, [3.0]);
        assert_eq!(composite.telemetry().npu_projections, 2);
        assert_eq!(composite.telemetry().gpu_projections, 1);

        let hexagon_child =
            crate::NativeQ2Executor::new(CompositePtq1Submitter, limits, 1024 * 1024);
        let adreno_child = MockExecutor {
            id: "adreno-owned-overflow",
            capacity: Some(64),
            fail_on: None,
            tile_tokens: false,
            calls: 0,
        };
        let mut owned =
            OwnedCompositeProjectionExecutor::new(&het_plan, hexagon_child, adreno_child)?;
        owned.project_f32(
            &model.tensors["blk.0.ffn_up.weight"],
            &vec![1.0_f32; 256],
            true,
            &mut out,
        )?;
        assert_eq!(out, [9.25; 32]);
        assert_eq!(owned.telemetry().npu_projections, 1);
        Ok(())
    }

    struct CompositeResidentSubmitter {
        registry: hexagon_native_rpc::NativeQ2WeightRegistry,
    }

    impl hexagon_native_rpc::NativeQ2Submitter for CompositeResidentSubmitter {
        fn residency_capacity_bytes(&self) -> usize {
            self.registry.remaining_bytes()
        }

        fn prepare_weights(
            &mut self,
            columns: usize,
            rows: usize,
            weights: &[u8],
        ) -> std::result::Result<
            Option<hexagon_native_rpc::NativeQ2WeightHandle>,
            hexagon_native_rpc::Error,
        > {
            self.registry.prepare(columns, rows, weights)
        }

        fn submit(
            &mut self,
            _columns: usize,
            _rows: usize,
            _weights: &[u8],
            _activation: &[f32],
            _output: &mut [f32],
        ) -> std::result::Result<(), hexagon_native_rpc::Error> {
            Err(hexagon_native_rpc::Error::Contract)
        }

        fn submit_resident(
            &mut self,
            handle: &hexagon_native_rpc::NativeQ2WeightHandle,
            activation: &[f32],
            output: &mut [f32],
        ) -> std::result::Result<(), hexagon_native_rpc::Error> {
            self.registry.range(handle)?;
            if activation.len() != handle.columns() || output.len() != handle.rows() {
                return Err(hexagon_native_rpc::Error::Contract);
            }
            output.fill(9.25);
            Ok(())
        }
    }

    #[test]
    fn owned_composite_forwards_host_weight_release_only_to_resident_npu() -> Result<()> {
        let model = build_vtcm_ptq1_test_model();
        let vtcm_plan =
            crate::plan_native_q2_layers_for_budget(&model, 73, 29_952, 1, 4 * 1024 * 1024)?;
        let het_plan = crate::plan_three_processor_placement(
            vtcm_plan.total_layers,
            vtcm_plan.admitted_layers,
            vtcm_plan.total_layers - vtcm_plan.admitted_layers,
            4,
            AcceleratorTarget::AdrenoGpuVulkan,
        )?;
        let limits = hexagon_native_rpc::NativeQ2ProjectionLimits::new(
            256,
            32,
            16,
            std::time::Duration::from_secs(4),
        )
        .map_err(|_| invalid("limits"))?
        .with_resident_weights(65_536)
        .map_err(|_| invalid("residency"))?;
        let submitter = CompositeResidentSubmitter {
            registry: hexagon_native_rpc::NativeQ2WeightRegistry::new(65_536)
                .map_err(|_| invalid("registry"))?,
        };
        let mut npu =
            crate::NativeQ2Executor::new(submitter, limits, 1024 * 1024).with_resident_weights(4096)?;
        let mut npu_up = model.tensors["blk.0.ffn_up.weight"].clone();
        let mut gpu_up = model.tensors["blk.2.ffn_up.weight"].clone();
        npu.select_and_prewarm_tensors(&[npu_up.clone()])?;

        let gpu = MockExecutor {
            id: "adreno-no-host-release",
            capacity: Some(64),
            fail_on: None,
            tile_tokens: false,
            calls: 0,
        };
        let mut owned = OwnedCompositeProjectionExecutor::new(&het_plan, npu, gpu)?;

        let orig_npu_data = npu_up.data.clone();
        let before_release_refs = TensorBytes::strong_count(&orig_npu_data);
        assert_eq!(before_release_refs, 5);
        let npu_token = TensorBytes::default();
        assert!(owned.release_host_weight_binding(&npu_up, &npu_token));
        npu_up.data = npu_token;
        assert_eq!(
            TensorBytes::strong_count(&orig_npu_data),
            before_release_refs - 3
        );
        assert!(owned.has_released_host_weight(&npu_up));

        // MockExecutor on GPU does not opt in to host-weight release, leaving overflow intact.
        let gpu_token = TensorBytes::default();
        assert!(!owned.release_host_weight_binding(&gpu_up, &gpu_token));
        assert!(!owned.has_released_host_weight(&gpu_up));
        gpu_up.data = TensorBytes::default();
        assert!(!owned.has_released_host_weight(&gpu_up));

        let mut out = Vec::new();
        owned.project_f32(&npu_up, &vec![1.0_f32; 256], true, &mut out)?;
        assert_eq!(out, [9.25; 32]);
        assert_eq!(owned.telemetry().npu_projections, 1);
        Ok(())
    }
}
