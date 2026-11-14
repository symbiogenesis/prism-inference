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
}

fn weight(name: &str) -> Tensor {
    let mut block = [0_u8; 34];
    block[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
    Tensor {
        name: name.into(),
        dimensions: vec![128, 128],
        kind: 142,
        data: Arc::from(block.repeat(128)),
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
    assert!(output.is_empty());
    output.push(17.0);
    assert!(
        prepared
            .matvec_ffn_swiglu_with_executor(&gate, &up, &mut output, &mut executor)
            .is_err()
    );
    assert!(output.is_empty());
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
    assert!(output.is_empty());
    assert!(other.is_empty());
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
    assert!(output.is_empty());
    assert!(other.is_empty());
    assert_eq!(executor.capability_queries.get(), 0);
    assert_eq!(executor.calls, 0);
    Ok(())
}
