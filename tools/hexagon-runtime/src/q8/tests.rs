//! Independent tile decoding and retained native DOTPROD result controls.

use std::{sync::Arc, time::Duration};

use bonsai_prism_runtime::kernels::{MatvecScratch, matvec};
use bonsai_ternary::half_to_float;
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
    reason = "Independent fixture plane dots are exactly representable small integers."
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

fn limits() -> Result<HmxQ2ProjectionLimits> {
    Ok(HmxQ2ProjectionLimits::new(
        32,
        1000,
        Duration::from_secs(10),
    )?)
}

fn fixture(kind: u32, columns: usize, rows: usize) -> Tensor {
    let width = if kind == PTQ1 { 28 } else { 34 };
    let mut data = vec![0_u8; columns / 128 * rows * width];
    for (index, block) in data.chunks_exact_mut(width).enumerate() {
        for (lane, byte) in block.iter_mut().enumerate() {
            *byte = (index * 53 + lane * 37).to_le_bytes()[0];
        }
        let scale_offset = if kind == PTQ1 { 26 } else { 0 };
        block[scale_offset..scale_offset + 2]
            .copy_from_slice(&[0x3c01_u16, 0x3787, 0x8000, 1][index % 4].to_le_bytes());
    }
    Tensor {
        name: "q8-hmx-control".into(),
        dimensions: vec![columns, rows],
        kind,
        data: Arc::from(data),
    }
}

#[test]
fn q8_hmx_preserves_prepared_groups_alias_bytes_and_partial_row_tiles() -> Result<()> {
    for (kind, columns, rows) in [(PTQ1, 256, 1), (PTQ1, 384, 19), (PQ2, 384, 19)] {
        let tensor = fixture(kind, columns, rows);
        let input: Vec<_> = (0..columns)
            .map(|lane| [127.0, -128.0, 0.1, -0.0, 53.25][lane % 5])
            .collect();
        let expected = matvec(&tensor, &input)?;
        let mut submitter = CpuSubmitter::default();
        let mut executor = HmxQ8Executor::new(&mut submitter, limits()?, 1 << 20);
        let mut scratch = MatvecScratch::default();
        let mut output = Vec::new();
        scratch.prepare(&tensor, &input)?.matvec_with_executor(
            &tensor,
            &mut output,
            &mut executor,
        )?;
        assert_eq!(
            output
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        let operations = u64::try_from(columns / 32 * rows.div_ceil(8))?;
        assert_eq!(executor.operation_count(&tensor), Some(operations));
        assert_eq!(executor.statistics().operations_completed, operations);
        assert_eq!(executor.operation_capacity(), Some(1000 - operations));
    }
    Ok(())
}

fn take<'a>(bytes: &mut &'a [u8], count: usize) -> Result<&'a [u8]> {
    let (first, rest) = bytes
        .split_at_checked(count)
        .ok_or("truncated native fixture")?;
    *bytes = rest;
    Ok(first)
}

#[test]
fn q8_hmx_matches_independent_pinned_native_dotprod_capture() -> Result<()> {
    let mut capture =
        include_bytes!("../../../../tests/fixtures/swift-ptq1-dot-parity.bin").as_slice();
    for _ in 0..2 {
        let columns = usize::try_from(u32::from_le_bytes(take(&mut capture, 4)?.try_into()?))?;
        let expected = u32::from_le_bytes(take(&mut capture, 4)?.try_into()?);
        let tensor = Tensor {
            name: "retained-native-dotprod".into(),
            dimensions: vec![columns, 1],
            kind: PTQ1,
            data: Arc::from(take(&mut capture, columns / 128 * 28)?),
        };
        let activation = take(&mut capture, columns / 32 * 34)?
            .as_chunks::<34>()
            .0
            .iter()
            .map(|block| Q8Block {
                scale: half_to_float(u16::from_le_bytes([block[0], block[1]])),
                sum: 0.0,
                values: std::array::from_fn(|lane| i8::from_ne_bytes([block[2 + lane]])),
            })
            .collect::<Vec<_>>();
        let mut submitter = CpuSubmitter::default();
        let mut executor = HmxQ8Executor::new(&mut submitter, limits()?, 1 << 20);
        let mut output = Vec::new();
        executor.project_q8(&tensor, &activation, true, &mut output)?;
        assert_eq!(output[0].to_bits(), expected);
    }
    assert_eq!(capture, b"");
    Ok(())
}

#[test]
fn q8_hmx_failure_and_unwritten_output_close_without_replay() -> Result<()> {
    let tensor = fixture(PTQ1, 128, 1);
    for partial in [false, true] {
        let mut submitter = CpuSubmitter {
            fail: !partial,
            partial,
            ..Default::default()
        };
        let mut executor = HmxQ8Executor::new(&mut submitter, limits()?, 1 << 20);
        let mut scratch = MatvecScratch::default();
        let mut prepared = scratch.prepare(&tensor, &[1.0; 128])?;
        let mut output = vec![42.0];
        for _ in 0..2 {
            assert!(
                prepared
                    .matvec_with_executor(&tensor, &mut output, &mut executor)
                    .is_err()
            );
            assert_eq!(output, Vec::<f32>::new());
            assert!(executor.is_faulted());
        }
        assert_eq!(executor.statistics().operations_attempted, 1);
        assert_eq!(executor.statistics().operations_completed, 0);
    }
    Ok(())
}

#[test]
fn q8_hmx_memory_refusal_retains_cpu_before_submission() -> Result<()> {
    let tensor = fixture(PTQ1, 128, 1);
    let mut submitter = CpuSubmitter::default();
    let mut executor = HmxQ8Executor::new(&mut submitter, limits()?, 1);
    let mut scratch = MatvecScratch::default();
    let mut output = Vec::new();
    scratch
        .prepare(&tensor, &[1.0; 128])?
        .matvec_with_executor(&tensor, &mut output, &mut executor)?;
    assert_eq!(executor.statistics(), ProjectionStatistics::default());
    assert_eq!(output.len(), 1);
    Ok(())
}
