//! Capability rejection before any fused/group submission or output publication.

use std::{cell::Cell, sync::Arc};

use super::{Pq2ProjectionExecutor, Q8KBlock};
use crate::{Result, invalid, kernels::MatvecScratch, model::Tensor};

#[derive(Default)]
struct RefusingExecutor {
    calls: usize,
    capability_queries: Cell<usize>,
}

impl Pq2ProjectionExecutor for RefusingExecutor {
    fn identity(&self) -> &'static str {
        "refuses-f32-arithmetic"
    }

    fn supports(&self, _tensor: &Tensor) -> bool {
        false
    }

    fn supports_f32_group(&self, _tensors: &[&Tensor]) -> bool {
        self.capability_queries
            .set(self.capability_queries.get() + 1);
        false
    }

    fn project(
        &mut self,
        _tensor: &Tensor,
        _activation: &[Q8KBlock],
        _fused: bool,
        _output: &mut Vec<f32>,
    ) -> Result<()> {
        self.calls += 1;
        Err(invalid("unexpected quantized submission"))
    }

    fn project_f32_group(
        &mut self,
        _tensors: &[&Tensor],
        _activation: &[f32],
        _pinned: bool,
        _outputs: &mut [&mut Vec<f32>],
    ) -> Result<()> {
        self.calls += 1;
        Err(invalid("unexpected F32 group"))
    }

    fn project_f32_ffn_swiglu(
        &mut self,
        _gate: &Tensor,
        _up: &Tensor,
        _activation: &[f32],
        _pinned: bool,
        _output: &mut Vec<f32>,
    ) -> Result<()> {
        self.calls += 1;
        Err(invalid("unexpected F32 SwiGLU fusion"))
    }

    fn project_f32_ffn_chain(
        &mut self,
        _tensors: [&Tensor; 3],
        _activation: &[f32],
        _residual: Option<&[f32]>,
        _hadamard_block_size: Option<u32>,
        _pinned: bool,
        _output: &mut Vec<f32>,
    ) -> Result<()> {
        self.calls += 1;
        Err(invalid("unexpected F32 FFN chain"))
    }

    fn project_f32_ffn_chain_lora(
        &mut self,
        _tensors: [&Tensor; 3],
        _activation: &[f32],
        _residual: Option<&[f32]>,
        _hadamard_block_size: Option<u32>,
        _lora_a: &[f32],
        _pinned: bool,
        _output: &mut Vec<f32>,
    ) -> Result<f32> {
        self.calls += 1;
        Err(invalid("unexpected F32 FFN chain LoRA"))
    }
}

fn weight(name: &str) -> Tensor {
    let mut block = [0_u8; 34];
    block[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
    Tensor {
        name: name.into(),
        dimensions: vec![128, 128],
        kind: 142,
        data: Arc::<[u8]>::from(block.repeat(128)).into(),
    }
}

#[test]
fn projection_precision_capabilities_gate_fused_and_group_callbacks() -> Result<()> {
    let gate = weight("gate");
    let up = weight("up");
    let down = weight("down");
    // Both values are finite F32: one exceeds F16 range and one distinguishes
    // F32 from F16 rounding. Unsupported precision must reject before callbacks.
    let input: Vec<_> = (0..128)
        .map(|index| {
            if index % 2 == 0 {
                131_072.0
            } else {
                f32::from_bits(0x3f80_0001)
            }
        })
        .collect();
    let mut scratch = MatvecScratch::default();
    let mut prepared = scratch.prepare_f32_executor(&gate, &input)?;
    let mut executor = RefusingExecutor::default();
    let mut output = vec![17.0];
    assert!(
        prepared
            .matvec_ffn_chain_with_executor(
                [&gate, &up, &down],
                None,
                None,
                &mut output,
                &mut executor
            )
            .is_err()
    );
    assert_eq!(output, Vec::<f32>::new());
    output.push(17.0);
    let lora_a = vec![0.25_f32; 128];
    assert!(
        prepared
            .matvec_ffn_chain_lora_with_executor(
                [&gate, &up, &down],
                None,
                None,
                &lora_a,
                &mut output,
                &mut executor
            )
            .is_err()
    );
    assert_eq!(output, Vec::<f32>::new());
    output.push(17.0);
    assert!(
        prepared
            .matvec_ffn_swiglu_with_executor(&gate, &up, &mut output, &mut executor)
            .is_err()
    );
    assert_eq!(output, Vec::<f32>::new());
    let mut other = vec![23.0];
    output.push(17.0);
    assert!(
        prepared
            .matvec_group_with_executor(
                &[&gate, &up],
                &mut [&mut output, &mut other],
                &mut executor
            )
            .is_err()
    );
    assert_eq!(output, Vec::<f32>::new());
    assert_eq!(other, Vec::<f32>::new());
    assert_eq!(executor.calls, 0);
    assert_eq!(executor.capability_queries.get(), 1);
    Ok(())
}

#[test]
fn projection_precision_validates_group_geometry_before_capability_callback() -> Result<()> {
    let valid = weight("valid");
    let malformed = Tensor {
        dimensions: Vec::new(),
        ..weight("malformed")
    };
    let mut scratch = MatvecScratch::default();
    let input = vec![0.0; 128];
    let mut prepared = scratch.prepare_f32_executor(&valid, &input)?;
    let mut executor = RefusingExecutor::default();
    let mut output = vec![17.0];
    let mut other = vec![23.0];
    assert!(
        prepared
            .matvec_group_with_executor(
                &[&valid, &malformed],
                &mut [&mut output, &mut other],
                &mut executor
            )
            .is_err()
    );
    assert_eq!(output, Vec::<f32>::new());
    assert_eq!(other, Vec::<f32>::new());
    assert_eq!(executor.capability_queries.get(), 0);
    assert_eq!(executor.calls, 0);
    Ok(())
}

struct ChainLoraExecutor {
    dot: f32,
    output_val: f32,
    calls: usize,
}

impl Pq2ProjectionExecutor for ChainLoraExecutor {
    fn identity(&self) -> &'static str {
        "chain-lora-test"
    }

    fn supports(&self, _tensor: &Tensor) -> bool {
        false
    }

    fn project(
        &mut self,
        _tensor: &Tensor,
        _activation: &[Q8KBlock],
        _fused: bool,
        _output: &mut Vec<f32>,
    ) -> Result<()> {
        Err(invalid("unexpected quantized submission"))
    }

    fn supports_f32_ffn_chain_lora(&self, tensors: [&Tensor; 3]) -> bool {
        tensors.iter().all(|tensor| tensor.kind == 142)
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
        self.calls += 1;
        assert_eq!(lora_a.len(), tensors[2].dimensions[0]);
        output.clear();
        output.resize(tensors[2].dimensions[1], self.output_val);
        Ok(self.dot)
    }
}

#[test]
fn ffn_chain_lora_validates_lora_a_and_rejects_nonfinite_dot_before_publication() -> Result<()> {
    let gate = weight("gate");
    let up = weight("up");
    let down = weight("down");
    let input = vec![1.0_f32; 128];
    let mut scratch = MatvecScratch::default();
    let mut prepared = scratch.prepare_f32_executor(&gate, &input)?;
    let mut exec = ChainLoraExecutor {
        dot: 3.5,
        output_val: 2.0,
        calls: 0,
    };
    let mut output = Vec::new();
    let dot = prepared.matvec_ffn_chain_lora_with_executor(
        [&gate, &up, &down],
        None,
        None,
        &vec![0.5_f32; 128],
        &mut output,
        &mut exec,
    )?;
    assert_eq!(dot, 3.5);
    assert_eq!(output, vec![2.0_f32; 128]);
    assert_eq!(exec.calls, 1);

    // Mismatched or nonfinite lora_a refuses before calling executor.
    assert!(
        prepared
            .matvec_ffn_chain_lora_with_executor(
                [&gate, &up, &down],
                None,
                None,
                &vec![0.5_f32; 127],
                &mut output,
                &mut exec,
            )
            .is_err()
    );
    assert_eq!(output, Vec::<f32>::new());
    let mut nonfinite_a = vec![0.5_f32; 128];
    nonfinite_a[17] = f32::NAN;
    assert!(
        prepared
            .matvec_ffn_chain_lora_with_executor(
                [&gate, &up, &down],
                None,
                None,
                &nonfinite_a,
                &mut output,
                &mut exec,
            )
            .is_err()
    );
    assert_eq!(exec.calls, 1);

    // Nonfinite scalar dot clears output and returns error.
    exec.dot = f32::INFINITY;
    assert!(
        prepared
            .matvec_ffn_chain_lora_with_executor(
                [&gate, &up, &down],
                None,
                None,
                &vec![0.5_f32; 128],
                &mut output,
                &mut exec,
            )
            .is_err()
    );
    assert_eq!(output, Vec::<f32>::new());
    assert_eq!(exec.calls, 2);
    Ok(())
}

#[derive(Default)]
struct FullForwardingExecutor {
    calls: usize,
}

impl Pq2ProjectionExecutor for FullForwardingExecutor {
    fn identity(&self) -> &'static str {
        "full-forwarding-test"
    }

    fn operation_capacity(&self) -> Option<u64> {
        Some(42)
    }

    fn operation_count(&self, _tensor: &Tensor) -> Option<u64> {
        Some(3)
    }

    fn supports(&self, _tensor: &Tensor) -> bool {
        true
    }

    fn project(
        &mut self,
        tensor: &Tensor,
        _activation: &[Q8KBlock],
        _fused: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        self.calls += 1;
        output.clear();
        output.resize(tensor.dimensions[1], 1.0);
        Ok(())
    }

    fn supports_q8(&self, _tensor: &Tensor) -> bool {
        true
    }

    fn project_q8(
        &mut self,
        tensor: &Tensor,
        _activation: &[super::Q8Block],
        _pinned: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        self.calls += 1;
        output.clear();
        output.resize(tensor.dimensions[1], 2.0);
        Ok(())
    }

    fn supports_f32(&self, _tensor: &Tensor) -> bool {
        true
    }

    fn project_f32(
        &mut self,
        tensor: &Tensor,
        _activation: &[f32],
        _pinned: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        self.calls += 1;
        output.clear();
        output.resize(tensor.dimensions[1], 3.0);
        Ok(())
    }

    fn supports_f32_group(&self, _tensors: &[&Tensor]) -> bool {
        true
    }

    fn project_f32_group(
        &mut self,
        tensors: &[&Tensor],
        _activation: &[f32],
        _pinned: bool,
        outputs: &mut [&mut Vec<f32>],
    ) -> Result<()> {
        self.calls += 1;
        for (tensor, output) in tensors.iter().zip(outputs.iter_mut()) {
            output.clear();
            output.resize(tensor.dimensions[1], 4.0);
        }
        Ok(())
    }

    fn project_f32_tokens(
        &mut self,
        tensor: &Tensor,
        activations: &[Vec<f32>],
        _pinned: bool,
        outputs: &mut [Vec<f32>],
    ) -> Result<()> {
        self.calls += 1;
        assert_eq!(activations.len(), outputs.len());
        for output in outputs.iter_mut() {
            output.clear();
            output.resize(tensor.dimensions[1], 5.0);
        }
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
        self.calls += 1;
        assert_eq!(activations.len(), outputs_a.len());
        assert_eq!(activations.len(), outputs_b.len());
        for (out_a, out_b) in outputs_a.iter_mut().zip(outputs_b.iter_mut()) {
            out_a.clear();
            out_a.resize(tensors[0].dimensions[1], 6.0);
            out_b.clear();
            out_b.resize(tensors[1].dimensions[1], 6.5);
        }
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
        self.calls += 1;
        assert_eq!(activations.len(), outputs_a.len());
        assert_eq!(activations.len(), outputs_b.len());
        assert_eq!(activations.len(), outputs_c.len());
        for ((out_a, out_b), out_c) in outputs_a
            .iter_mut()
            .zip(outputs_b.iter_mut())
            .zip(outputs_c.iter_mut())
        {
            out_a.clear();
            out_a.resize(tensors[0].dimensions[1], 7.0);
            out_b.clear();
            out_b.resize(tensors[1].dimensions[1], 7.5);
            out_c.clear();
            out_c.resize(tensors[2].dimensions[1], 8.0);
        }
        Ok(())
    }

    fn supports_f32_ffn_swiglu(&self, _gate: &Tensor, _up: &Tensor) -> bool {
        true
    }

    fn project_f32_ffn_swiglu(
        &mut self,
        gate: &Tensor,
        _up: &Tensor,
        _activation: &[f32],
        _pinned: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        self.calls += 1;
        output.clear();
        output.resize(gate.dimensions[1], 9.0);
        Ok(())
    }

    fn project_f32_ffn_swiglu_tokens(
        &mut self,
        gate: &Tensor,
        _up: &Tensor,
        activations: &[Vec<f32>],
        _pinned: bool,
        outputs: &mut [Vec<f32>],
    ) -> Result<()> {
        self.calls += 1;
        assert_eq!(activations.len(), outputs.len());
        for output in outputs.iter_mut() {
            output.clear();
            output.resize(gate.dimensions[1], 9.5);
        }
        Ok(())
    }

    fn supports_f32_ffn_chain(&self, _tensors: [&Tensor; 3]) -> bool {
        true
    }

    fn supports_tile_tokens(&self) -> bool {
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
        self.calls += 1;
        output.clear();
        output.resize(tensors[2].dimensions[1], 10.0);
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
        _lora_a: &[f32],
        _pinned: bool,
        output: &mut Vec<f32>,
    ) -> Result<f32> {
        self.calls += 1;
        output.clear();
        output.resize(tensors[2].dimensions[1], 11.0);
        Ok(11.5)
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
        self.calls += 1;
        assert_eq!(activations.len(), outputs.len());
        for output in outputs.iter_mut() {
            output.clear();
            output.resize(tensors[2].dimensions[1], 12.0);
        }
        Ok(())
    }
}

fn exercise_forwarded_executor<E: Pq2ProjectionExecutor>(
    mut exec: E,
    gate: &Tensor,
    up: &Tensor,
    down: &Tensor,
    input: &[f32],
) -> Result<E> {
    assert_eq!(exec.identity(), "full-forwarding-test");
    assert_eq!(exec.operation_capacity(), Some(42));
    assert_eq!(exec.operation_count(gate), Some(3));
    assert!(exec.supports(gate));
    assert!(exec.supports_q8(gate));
    assert!(exec.supports_f32(gate));
    assert!(exec.supports_f32_group(&[gate, up]));
    assert!(exec.supports_f32_ffn_swiglu(gate, up));
    assert!(exec.supports_f32_ffn_chain([gate, up, down]));
    assert!(exec.supports_tile_tokens());
    assert!(exec.supports_f32_ffn_chain_lora([gate, up, down]));

    let q8k = [Q8KBlock {
        scale: 1.0,
        values: [0; 256],
        sums: [0; 16],
    }];
    let q8 = [super::Q8Block {
        scale: 1.0,
        sum: 0.0,
        values: [0; 32],
    }];
    let mut output = Vec::new();
    let mut other = Vec::new();
    exec.project(gate, &q8k, false, &mut output)?;
    assert_eq!(output[0], 1.0);
    exec.project_q8(gate, &q8, false, &mut output)?;
    assert_eq!(output[0], 2.0);
    exec.project_f32(gate, input, false, &mut output)?;
    assert_eq!(output[0], 3.0);
    exec.project_f32_group(&[gate, up], input, false, &mut [&mut output, &mut other])?;
    assert_eq!((output[0], other[0]), (4.0, 4.0));

    let acts = vec![input.to_vec(), input.to_vec()];
    let mut out_a = vec![Vec::new(); 2];
    let mut out_b = vec![Vec::new(); 2];
    let mut out_c = vec![Vec::new(); 2];
    exec.project_f32_tokens(gate, &acts, false, &mut out_a)?;
    assert_eq!(out_a[1][0], 5.0);
    exec.project_f32_pair_tokens([gate, up], &acts, false, &mut out_a, &mut out_b)?;
    assert_eq!((out_a[0][0], out_b[1][0]), (6.0, 6.5));
    exec.project_f32_triple_tokens(
        [gate, up, down],
        &acts,
        false,
        &mut out_a,
        &mut out_b,
        &mut out_c,
    )?;
    assert_eq!((out_a[0][0], out_b[0][0], out_c[1][0]), (7.0, 7.5, 8.0));
    exec.project_f32_ffn_swiglu(gate, up, input, false, &mut output)?;
    assert_eq!(output[0], 9.0);
    exec.project_f32_ffn_swiglu_tokens(gate, up, &acts, false, &mut out_a)?;
    assert_eq!(out_a[1][0], 9.5);
    exec.project_f32_ffn_chain([gate, up, down], input, None, None, false, &mut output)?;
    assert_eq!(output[0], 10.0);
    let dot = exec.project_f32_ffn_chain_lora(
        [gate, up, down],
        input,
        None,
        None,
        input,
        false,
        &mut output,
    )?;
    assert_eq!((dot, output[0]), (11.5, 11.0));
    exec.project_f32_ffn_chain_tokens([gate, up, down], &acts, None, None, false, &mut out_a)?;
    assert_eq!(out_a[1][0], 12.0);
    Ok(exec)
}

#[test]
fn forwarding_impls_preserve_all_executor_methods_and_matvec_compatibility() -> Result<()> {
    let gate = weight("gate");
    let up = weight("up");
    let down = weight("down");
    let input = vec![1.0_f32; 128];

    let mut concrete = FullForwardingExecutor::default();
    let _ = exercise_forwarded_executor(&mut concrete, &gate, &up, &down, &input)?;
    assert_eq!(concrete.calls, 12);

    let dyn_ref: &mut dyn Pq2ProjectionExecutor = &mut concrete;
    let _ = exercise_forwarded_executor(dyn_ref, &gate, &up, &down, &input)?;
    assert_eq!(concrete.calls, 24);

    let boxed = Box::new(FullForwardingExecutor::default());
    let mut boxed = exercise_forwarded_executor(boxed, &gate, &up, &down, &input)?;
    assert_eq!(boxed.calls, 12);

    let boxed_dyn: Box<dyn Pq2ProjectionExecutor + Send> = Box::new(concrete);
    let mut boxed_dyn = exercise_forwarded_executor(boxed_dyn, &gate, &up, &down, &input)?;
    assert_eq!(boxed_dyn.identity(), "full-forwarding-test");

    let mut scratch = MatvecScratch::default();
    let mut prepared = scratch.prepare_f32_executor(&gate, &input)?;
    let mut output = Vec::new();
    let mut other = Vec::new();
    prepared.matvec_with_executor(&gate, &mut output, &mut boxed)?;
    assert_eq!(output, vec![3.0_f32; 128]);
    prepared.matvec_group_with_executor(
        &[&gate, &up],
        &mut [&mut output, &mut other],
        &mut boxed_dyn,
    )?;
    assert_eq!(output, vec![4.0_f32; 128]);
    assert_eq!(other, vec![4.0_f32; 128]);
    prepared.matvec_ffn_swiglu_with_executor(&gate, &up, &mut output, &mut boxed)?;
    assert_eq!(output, vec![9.0_f32; 128]);
    prepared.matvec_ffn_chain_with_executor(
        [&gate, &up, &down],
        None,
        None,
        &mut output,
        &mut boxed,
    )?;
    assert_eq!(output, vec![10.0_f32; 128]);
    let dot = prepared.matvec_ffn_chain_lora_with_executor(
        [&gate, &up, &down],
        None,
        None,
        &input,
        &mut output,
        &mut boxed_dyn,
    )?;
    assert_eq!(dot, 11.5);
    assert_eq!(output, vec![11.0_f32; 128]);
    Ok(())
}
