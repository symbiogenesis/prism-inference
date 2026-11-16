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
