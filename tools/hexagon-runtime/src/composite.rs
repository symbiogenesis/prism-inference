//! Composite NPU + GPU projection executor and hybrid prefill coordinator.

use std::fmt::Write;

use bonsai_native_hash::Hasher;
use bonsai_prism_runtime::{
    kernels::{Pq2ProjectionExecutor, Q8Block},
    model::Tensor,
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

/// Strict two-processor [`Pq2ProjectionExecutor`] dispatching prefix layers to NPU
/// and overflow layers to GPU according to a [`HeterogeneousPlacementPlan`].
pub struct CompositeProjectionExecutor<'a> {
    npu: &'a mut dyn Pq2ProjectionExecutor,
    gpu: &'a mut dyn Pq2ProjectionExecutor,
    plan_npu_layers: usize,
    plan_total_layers: usize,
    identity: String,
    telemetry: CompositeTelemetry,
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

impl<'a> CompositeProjectionExecutor<'a> {
    /// Construct a composite NPU + GPU executor from an explicit placement plan.
    ///
    /// # Errors
    /// Rejects plans with non-zero CPU layers, empty/inconsistent layer tables, or
    /// invalid child executor identities.
    pub fn new(
        plan: &HeterogeneousPlacementPlan,
        npu: &'a mut dyn Pq2ProjectionExecutor,
        gpu: &'a mut dyn Pq2ProjectionExecutor,
    ) -> Result<Self> {
        if plan.cpu_layers != 0
            || plan.total_layers == 0
            || plan.npu_layers + plan.gpu_layers != plan.total_layers
            || plan.layers.len() != plan.total_layers
            || !valid_component_identity(&plan.identity)
            || !valid_component_identity(npu.identity())
            || !valid_component_identity(gpu.identity())
        {
            return Err(invalid("invalid composite projection placement"));
        }
        let mut digest = Hasher::new()?;
        digest.update(b"prism-composite-projection-v1\0")?;
        digest.update(&u64::try_from(plan.npu_layers)?.to_le_bytes())?;
        digest.update(&u64::try_from(plan.total_layers)?.to_le_bytes())?;
        for component in [plan.identity.as_str(), npu.identity(), gpu.identity()] {
            digest.update(&u64::try_from(component.len())?.to_le_bytes())?;
            digest.update(component.as_bytes())?;
        }
        let mut identity = String::new();
        identity.try_reserve_exact(88)?;
        identity.push_str("composite-projection-v1:");
        for byte in digest.finish()? {
            write!(&mut identity, "{byte:02x}")?;
        }
        Ok(Self {
            npu,
            gpu,
            plan_npu_layers: plan.npu_layers,
            plan_total_layers: plan.total_layers,
            identity,
            telemetry: CompositeTelemetry::default(),
        })
    }

    /// Snapshot cumulative NPU/GPU projection counts and latched fault status.
    #[must_use]
    pub const fn telemetry(&self) -> CompositeTelemetry {
        self.telemetry
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

    fn finish_dispatch(
        &mut self,
        target: RouteTarget,
        count: u64,
        result: Result<()>,
    ) -> Result<()> {
        match result {
            Ok(()) => {
                match target {
                    RouteTarget::Npu => self.telemetry.npu_projections += count,
                    RouteTarget::Gpu => self.telemetry.gpu_projections += count,
                }
                Ok(())
            }
            Err(error) => {
                self.telemetry.faulted = true;
                Err(error)
            }
        }
    }
}

impl Pq2ProjectionExecutor for CompositeProjectionExecutor<'_> {
    fn identity(&self) -> &str {
        &self.identity
    }

    fn operation_capacity(&self) -> Option<u64> {
        if self.telemetry.faulted {
            Some(0)
        } else {
            self.npu
                .operation_capacity()
                .zip(self.gpu.operation_capacity())
                .and_then(|(left, right)| left.checked_add(right))
        }
    }

    fn operation_count(&self, tensor: &Tensor) -> Option<u64> {
        if self.telemetry.faulted {
            return None;
        }
        match self.route(&tensor.name)? {
            RouteTarget::Npu => self.npu.operation_count(tensor),
            RouteTarget::Gpu => self.gpu.operation_count(tensor),
        }
    }

    fn supports(&self, tensor: &Tensor) -> bool {
        if self.telemetry.faulted {
            return false;
        }
        match self.route(&tensor.name) {
            Some(RouteTarget::Npu) => self.npu.supports(tensor),
            Some(RouteTarget::Gpu) => self.gpu.supports(tensor),
            None => false,
        }
    }

    fn supports_q8(&self, tensor: &Tensor) -> bool {
        if self.telemetry.faulted {
            return false;
        }
        match self.route(&tensor.name) {
            Some(RouteTarget::Npu) => self.npu.supports_q8(tensor),
            Some(RouteTarget::Gpu) => self.gpu.supports_q8(tensor),
            None => false,
        }
    }

    fn supports_f32(&self, tensor: &Tensor) -> bool {
        if self.telemetry.faulted {
            return false;
        }
        match self.route(&tensor.name) {
            Some(RouteTarget::Npu) => self.npu.supports_f32(tensor),
            Some(RouteTarget::Gpu) => self.gpu.supports_f32(tensor),
            None => false,
        }
    }

    fn supports_f32_group(&self, tensors: &[&Tensor]) -> bool {
        match self.group_target(tensors) {
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
        if self.telemetry.faulted || !self.supports(tensor) {
            return Err(invalid("unavailable composite Q8_K projection placement"));
        }
        let target = self
            .route(&tensor.name)
            .ok_or_else(|| invalid("missing composite layer target"))?;
        let result = match target {
            RouteTarget::Npu => self.npu.project(tensor, activation, fused, output),
            RouteTarget::Gpu => self.gpu.project(tensor, activation, fused, output),
        };
        if result.is_err() {
            output.clear();
        }
        self.finish_dispatch(target, 1, result)
    }

    fn project_q8(
        &mut self,
        tensor: &Tensor,
        activation: &[Q8Block],
        pinned: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        output.clear();
        if self.telemetry.faulted || !self.supports_q8(tensor) {
            return Err(invalid("unavailable composite Q8_0 projection placement"));
        }
        let target = self
            .route(&tensor.name)
            .ok_or_else(|| invalid("missing composite layer target"))?;
        let result = match target {
            RouteTarget::Npu => self.npu.project_q8(tensor, activation, pinned, output),
            RouteTarget::Gpu => self.gpu.project_q8(tensor, activation, pinned, output),
        };
        if result.is_err() {
            output.clear();
        }
        self.finish_dispatch(target, 1, result)
    }

    fn project_f32(
        &mut self,
        tensor: &Tensor,
        activation: &[f32],
        pinned: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        output.clear();
        if self.telemetry.faulted || !self.supports_f32(tensor) {
            return Err(invalid("unavailable composite F32 projection placement"));
        }
        let target = self
            .route(&tensor.name)
            .ok_or_else(|| invalid("missing composite layer target"))?;
        let result = match target {
            RouteTarget::Npu => self.npu.project_f32(tensor, activation, pinned, output),
            RouteTarget::Gpu => self.gpu.project_f32(tensor, activation, pinned, output),
        };
        if result.is_err() {
            output.clear();
        }
        self.finish_dispatch(target, 1, result)
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
        if !self.supports_f32_group(tensors) {
            return Err(invalid(
                "unavailable composite F32 group projection placement",
            ));
        }
        let target = self
            .group_target(tensors)
            .ok_or_else(|| invalid("missing composite group target"))?;
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
        self.finish_dispatch(target, count, result)
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
        calls: usize,
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

        fn project(
            &mut self,
            tensor: &Tensor,
            _activation: &[Q8KBlock],
            _fused: bool,
            output: &mut Vec<f32>,
        ) -> Result<()> {
            if self.fail_on == Some(tensor.name.as_str()) {
                return Err(invalid("injected child failure"));
            }
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
            if self.fail_on == Some(tensor.name.as_str()) {
                return Err(invalid("injected child failure"));
            }
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
            if self.fail_on == Some(tensor.name.as_str()) {
                return Err(invalid("injected child failure"));
            }
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
            for (tensor, output) in tensors.iter().zip(outputs.iter_mut()) {
                if self.fail_on == Some(tensor.name.as_str()) {
                    return Err(invalid("injected group failure"));
                }
                output.push(4.0);
            }
            self.calls += tensors.len();
            Ok(())
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
            calls: 0,
        };
        let mut gpu = MockExecutor {
            id: "mock-adreno-x1-85",
            capacity: Some(200),
            fail_on: Some("blk.27.ffn_down.weight"),
            calls: 0,
        };
        let mut composite = CompositeProjectionExecutor::new(&plan, &mut npu, &mut gpu)?;
        assert_eq!(composite.operation_capacity(), Some(300));

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
}
