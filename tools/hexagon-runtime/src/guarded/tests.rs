//! Independent scalar decoding of the proposed signed operation's inputs.

use std::{sync::Arc, time::Duration};

use bonsai_prism_runtime::kernels::{MatvecBackend, MatvecScratch, matvec_into};
use hexagon_native_rpc::Error;

use super::*;

#[derive(Default)]
struct CpuSubmitter {
    calls: usize,
    fail: bool,
    partial: bool,
}

#[allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "All fixture integer dots are bounded by 16384 and exactly representable as F32."
)]
impl GuardedQ2Submitter for CpuSubmitter {
    fn submit(
        &mut self,
        weights: &[u8],
        activation: &[[f32; 256]; 2],
        output: &mut [f32],
    ) -> std::result::Result<(), Error> {
        self.calls += 1;
        if self.fail {
            return Err(Error::Contract);
        }
        let rows = output.len() / 2;
        assert_eq!(weights.len(), rows * 72);
        for tile in weights.as_chunks::<576>().0 {
            for scale in tile[512..].as_chunks::<2>().0 {
                assert_eq!(*scale, [0, 0x3c]);
            }
        }
        for limb in 0..2 {
            for row in 0..rows {
                if self.partial && row == rows - 1 {
                    continue;
                }
                let mut dot = 0_i32;
                for (lane, &value) in activation[limb].iter().enumerate() {
                    let local = lane % 64;
                    let tile = (row / 32 * 4 + lane / 64) * 576;
                    let byte = weights[tile + local / 16 * 128 + row % 32 * 4 + local % 4];
                    let symbol = (byte >> (local / 4 % 4 * 2)) & 3;
                    assert!(value.is_finite() && value.fract() == 0.0 && value.abs() <= 127.0);
                    dot += (i32::from(symbol) - 1) * value as i32;
                }
                output[limb * rows + row] = dot as f32;
            }
        }
        Ok(())
    }
}

fn fixture(columns: usize, rows: usize) -> Tensor {
    let mut bytes = vec![0; columns / 128 * rows * 34];
    for (index, block) in bytes.as_chunks_mut::<34>().0.iter_mut().enumerate() {
        block[..2].copy_from_slice(&[0x3c01_u16, 0x3787, 0x8000, 1][index % 4].to_le_bytes());
        for (lane, byte) in block[2..].iter_mut().enumerate() {
            *byte = (index * 53 + lane * 37).to_le_bytes()[0];
        }
    }
    Tensor {
        name: "hexagon-control".into(),
        kind: 142,
        dimensions: vec![columns, rows],
        data: Arc::from(bytes),
    }
}

#[test]
fn guarded_executor_preserves_prepared_codes_and_original_reduction_bits() -> Result<()> {
    for (columns, rows) in [(256, 1), (768, 35), (512, 64)] {
        let tensor = fixture(columns, rows);
        let input: Vec<_> = (0..columns)
            .map(|lane| [127.0, -128.0, 0.1, -0.0, 53.25][lane % 5])
            .collect();
        for backend in [MatvecBackend::Scalar, MatvecBackend::Auto] {
            let scratch = || MatvecScratch::with_backend(backend);
            let mut control_scratch = scratch();
            let mut expected = Vec::new();
            matvec_into(&tensor, &input, &mut expected, &mut control_scratch)?;
            for tile_rows in [32, 64] {
                let limits = Q2ProjectionLimits::new(tile_rows, 100, Duration::from_secs(10))?;
                let mut submitter = CpuSubmitter::default();
                let mut executor = GuardedQ2Executor::new(&mut submitter, limits, 1 << 20);
                let mut prepared_scratch = scratch();
                let mut prepared = prepared_scratch.prepare(&tensor, &input)?;
                let mut actual = Vec::new();
                prepared.matvec_with_executor(&tensor, &mut actual, &mut executor)?;
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
                assert_eq!(executor.statistics().projections_completed, 1);
                assert_eq!(
                    executor.statistics().operations_completed,
                    u64::try_from(columns / 128 * rows.div_ceil(tile_rows))?
                );
                assert!(!executor.is_faulted());
            }
        }
    }
    Ok(())
}

#[test]
fn guarded_executor_refusal_and_unwritten_output_are_terminal_without_replay() -> Result<()> {
    let tensor = fixture(256, 1);
    for partial in [false, true] {
        let mut submitter = CpuSubmitter {
            fail: !partial,
            partial,
            ..Default::default()
        };
        let limits = Q2ProjectionLimits::new(32, 10, Duration::from_secs(10))?;
        let mut executor = GuardedQ2Executor::new(&mut submitter, limits, 1 << 20);
        let mut scratch = MatvecScratch::default();
        let mut prepared = scratch.prepare(&tensor, &[1.0; 256])?;
        let mut output = vec![42.0];
        assert!(
            prepared
                .matvec_with_executor(&tensor, &mut output, &mut executor)
                .is_err()
        );
        assert_eq!(output, Vec::<f32>::new());
        assert!(executor.is_faulted());
        assert!(
            prepared
                .matvec_with_executor(&tensor, &mut output, &mut executor)
                .is_err()
        );
        assert_eq!(executor.statistics().operations_attempted, 1);
        assert_eq!(executor.statistics().operations_completed, 0);
    }
    Ok(())
}

#[test]
fn guarded_executor_memory_refusal_selects_cpu_before_any_submit() -> Result<()> {
    let tensor = fixture(256, 35);
    let mut submitter = CpuSubmitter::default();
    let limits = Q2ProjectionLimits::new(32, 10, Duration::from_secs(10))?;
    let mut executor = GuardedQ2Executor::new(&mut submitter, limits, 2048);
    let mut scratch = MatvecScratch::default();
    let mut output = Vec::new();
    scratch
        .prepare(&tensor, &[1.0; 256])?
        .matvec_with_executor(&tensor, &mut output, &mut executor)?;
    assert_eq!(output.len(), 35);
    assert_eq!(executor.statistics(), ProjectionStatistics::default());
    Ok(())
}
