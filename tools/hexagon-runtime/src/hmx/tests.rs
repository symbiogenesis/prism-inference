//! Independent scalar tile decoding and original-code reduction controls.

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
    reason = "Fixture dots are exact small integers and plane inputs are explicitly binary."
)]
impl HmxQ2Submitter for CpuSubmitter {
    fn submit(
        &mut self,
        weights: &[u8],
        activation: &[[f32; 128]; 8],
        output: &mut [f32],
    ) -> std::result::Result<(), Error> {
        self.calls += 1;
        if self.fail {
            return Err(Error::Contract);
        }
        let rows = output.len() / 8;
        assert_eq!(weights.len(), rows * 36);
        for (plane, codes) in activation.iter().enumerate() {
            for row in 0..rows {
                if self.partial && row == rows - 1 {
                    continue;
                }
                let mut dot = 0_i32;
                for (lane, &value) in codes.iter().enumerate() {
                    assert!([0.0_f32.to_bits(), 1.0_f32.to_bits()].contains(&value.to_bits()));
                    let local = lane % 64;
                    let tile = (row / 32 * 2 + lane / 64) * 576;
                    assert_eq!(
                        &weights[tile + 512 + row % 32 * 2..tile + 514 + row % 32 * 2],
                        &[0, 0x3c]
                    );
                    let byte = weights[tile + local / 16 * 128 + row % 32 * 4 + local % 4];
                    let symbol = (byte >> (local / 4 % 4 * 2)) & 3;
                    dot +=
                        (i32::from(symbol) - 1) * i32::from(value.to_bits() == 1.0_f32.to_bits());
                }
                output[plane * rows + row] = dot as f32;
            }
        }
        Ok(())
    }
}

fn fixture(columns: usize, rows: usize) -> Tensor {
    let mut data = vec![0_u8; columns / 128 * rows * 34];
    for (index, block) in data.as_chunks_mut::<34>().0.iter_mut().enumerate() {
        block[..2].copy_from_slice(&[0x3c01_u16, 0x3787, 0x8000, 1][index % 4].to_le_bytes());
        for (lane, byte) in block[2..].iter_mut().enumerate() {
            *byte = (index * 53 + lane * 37).to_le_bytes()[0];
        }
    }
    Tensor {
        name: "hmx-control".into(),
        kind: 142,
        dimensions: vec![columns, rows],
        data: Arc::from(data),
    }
}

#[test]
fn hmx_executor_preserves_preparation_fma_and_all_row_tiles() -> Result<()> {
    for (columns, rows) in [(256, 1), (768, 35), (512, 64)] {
        let tensor = fixture(columns, rows);
        let input: Vec<_> = (0..columns)
            .map(|lane| [127.0, -128.0, 0.1, -0.0, 53.25][lane % 5])
            .collect();
        for backend in [MatvecBackend::Scalar, MatvecBackend::Auto] {
            let scratch = || MatvecScratch::with_backend(backend);
            let mut expected = Vec::new();
            matvec_into(&tensor, &input, &mut expected, &mut scratch())?;
            for tile_rows in [32, 64] {
                let limits = HmxQ2ProjectionLimits::new(tile_rows, 100, Duration::from_secs(10))?;
                let mut submitter = CpuSubmitter::default();
                let mut executor = HmxQ2Executor::new(&mut submitter, limits, 1 << 20);
                let mut scratch = scratch();
                let mut actual = Vec::new();
                scratch.prepare(&tensor, &input)?.matvec_with_executor(
                    &tensor,
                    &mut actual,
                    &mut executor,
                )?;
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
                assert_eq!(executor.statistics().projections_completed, 1);
                assert_eq!(
                    executor.statistics().operations_completed,
                    u64::try_from(columns / 128 * rows.div_ceil(tile_rows))?
                );
            }
        }
    }
    Ok(())
}

#[test]
fn hmx_executor_full_i8_planes_preserve_signed_reconstruction() {
    for code in i8::MIN..=i8::MAX {
        let planes = activation_planes(&[code; 128]);
        let reconstructed: i16 = planes
            .iter()
            .enumerate()
            .map(|(bit, plane)| {
                let coefficient = if bit == 7 { -128 } else { 1 << bit };
                coefficient * i16::from(plane[0].to_bits() == 1.0_f32.to_bits())
            })
            .sum();
        assert_eq!(reconstructed, i16::from(code));
    }
}

#[test]
fn hmx_executor_retains_independent_original_fma_golden() -> Result<()> {
    let mut weights = [0xaa_u8; 68];
    weights[..2].copy_from_slice(&0x3c01_u16.to_le_bytes());
    weights[34..36].copy_from_slice(&0x3787_u16.to_le_bytes());
    let tensor = Tensor {
        name: "fma-golden".into(),
        kind: 142,
        dimensions: vec![256, 1],
        data: Arc::from(weights),
    };
    let mut codes = [3_i8; 256];
    codes[128..].fill(-13);
    let activation = [Q8KBlock {
        scale: f32::from_bits(0x3da0_0001),
        values: codes,
        sums: [0; 16],
    }];
    for (fused, expected) in [(true, 0xc1f9_0b01), (false, 0xc1f9_0b02)] {
        let limits = HmxQ2ProjectionLimits::new(32, 2, Duration::from_secs(10))?;
        let mut submitter = CpuSubmitter::default();
        let mut executor = HmxQ2Executor::new(&mut submitter, limits, 1 << 20);
        let mut output = Vec::new();
        executor.project(&tensor, &activation, fused, &mut output)?;
        assert_eq!(output[0].to_bits(), expected);
    }
    Ok(())
}

#[test]
fn hmx_executor_refusal_and_unwritten_output_close_without_replay() -> Result<()> {
    for partial in [false, true] {
        let tensor = fixture(256, 1);
        let limits = HmxQ2ProjectionLimits::new(32, 10, Duration::from_secs(10))?;
        let mut submitter = CpuSubmitter {
            fail: !partial,
            partial,
            ..Default::default()
        };
        let mut executor = HmxQ2Executor::new(&mut submitter, limits, 1 << 20);
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
