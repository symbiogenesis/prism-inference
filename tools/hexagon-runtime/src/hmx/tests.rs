//! Independent scalar tile decoding and original-code reduction controls.

use std::{sync::Arc, time::Duration};

use bonsai_prism_runtime::kernels::{MatvecBackend, MatvecScratch, matvec_into};
use hexagon_native_rpc::{Error, HmxQ2WeightHandle, HmxQ2WeightRegistry};

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
        let mut resident_submitter = ResidentSubmitter::new(4096)?;
        let mut resident_executor = HmxQ2Executor::new(&mut resident_submitter, limits, 1 << 20)
            .with_resident_weights(16_384)?;
        resident_executor.project(&tensor, &activation, fused, &mut output)?;
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

struct ResidentSubmitter {
    oracle: CpuSubmitter,
    registry: HmxQ2WeightRegistry,
    arena: Vec<u8>,
    preparations: usize,
    resident_calls: usize,
    streaming_calls: usize,
    refuse_preparation: Option<usize>,
}

impl ResidentSubmitter {
    fn new(bytes: usize) -> Result<Self> {
        Ok(Self {
            oracle: CpuSubmitter::default(),
            registry: HmxQ2WeightRegistry::new(bytes)?,
            arena: vec![0; bytes],
            preparations: 0,
            resident_calls: 0,
            streaming_calls: 0,
            refuse_preparation: None,
        })
    }
}

impl HmxQ2Submitter for ResidentSubmitter {
    fn residency_capacity_bytes(&self) -> usize {
        self.registry.remaining_bytes()
    }

    fn prepare_weights(
        &mut self,
        weights: &[u8],
    ) -> std::result::Result<Option<HmxQ2WeightHandle>, Error> {
        self.preparations += 1;
        if self.refuse_preparation == Some(self.preparations) {
            return Ok(None);
        }
        let handle = self.registry.prepare(weights)?;
        if let Some(handle) = &handle {
            self.arena[self.registry.range(handle)?].copy_from_slice(weights);
        }
        Ok(handle)
    }

    fn submit_resident(
        &mut self,
        handle: &HmxQ2WeightHandle,
        activation: &[[f32; 128]; 8],
        output: &mut [f32],
    ) -> std::result::Result<(), Error> {
        self.resident_calls += 1;
        self.oracle.submit(
            &self.arena[self.registry.range(handle)?],
            activation,
            output,
        )
    }

    fn submit(
        &mut self,
        weights: &[u8],
        activation: &[[f32; 128]; 8],
        output: &mut [f32],
    ) -> std::result::Result<(), Error> {
        self.streaming_calls += 1;
        self.oracle.submit(weights, activation, output)
    }
}

#[test]
fn hmx_resident_reuse_preserves_complete_original_outputs_and_operation_counts() -> Result<()> {
    for (columns, rows) in [(256, 1), (768, 35), (512, 64)] {
        let tensor = fixture(columns, rows);
        for backend in [MatvecBackend::Scalar, MatvecBackend::Auto] {
            for tile_rows in [32, 64] {
                let operations = columns / 128 * rows.div_ceil(tile_rows);
                let packed_bytes = columns / 128 * rows.next_multiple_of(32) * 36;
                let limits = HmxQ2ProjectionLimits::new(tile_rows, 100, Duration::from_secs(10))?;
                let mut submitter = ResidentSubmitter::new(packed_bytes)?;
                let mut executor = HmxQ2Executor::new(&mut submitter, limits, 1 << 20)
                    .with_resident_weights(16_384)?;
                let eligibility = executor.supports(&tensor);
                for seed in [0, 3] {
                    let input: Vec<_> = (0..columns)
                        .map(|lane| [127.0, -128.0, 0.1, -0.0, 53.25][(lane + seed) % 5])
                        .collect();
                    let mut expected = Vec::new();
                    matvec_into(
                        &tensor,
                        &input,
                        &mut expected,
                        &mut MatvecScratch::with_backend(backend),
                    )?;
                    let mut actual = Vec::new();
                    MatvecScratch::with_backend(backend)
                        .prepare(&tensor, &input)?
                        .matvec_with_executor(&tensor.clone(), &mut actual, &mut executor)?;
                    assert_eq!(
                        actual
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>(),
                        expected
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(executor.supports(&tensor), eligibility);
                }
                assert_eq!(executor.residency_statistics().tensors_prepared, 1);
                assert_eq!(executor.residency_statistics().cache_hits, 1);
                assert_eq!(
                    executor
                        .residency_statistics()
                        .weight_payload_bytes_prepared,
                    u64::try_from(packed_bytes)?
                );
                assert!(executor.residency_statistics().metadata_payload_bytes <= 16_384);
                assert_eq!(
                    executor.statistics().operations_completed,
                    u64::try_from(operations * 2)?
                );
                drop(executor);
                assert_eq!(submitter.preparations, operations);
                assert_eq!(submitter.resident_calls, operations * 2);
                assert_eq!(submitter.streaming_calls, 0);
            }
        }
    }
    Ok(())
}

#[test]
fn hmx_resident_source_identity_retains_arc_and_checks_geometry() -> Result<()> {
    let tensor = fixture(256, 2);
    let limits = HmxQ2ProjectionLimits::new(32, 100, Duration::from_secs(10))?;
    let mut submitter = ResidentSubmitter::new(32_768)?;
    let mut executor =
        HmxQ2Executor::new(&mut submitter, limits, 1 << 20).with_resident_weights(16_384)?;
    let mut changed_data = tensor.clone();
    let mut changed_geometry = tensor.clone();
    changed_geometry.dimensions = vec![512, 1];
    let compare = |current: &Tensor, executor: &mut HmxQ2Executor<'_>| -> Result<()> {
        let input = vec![1.0; current.dimensions[0]];
        let mut expected = Vec::new();
        matvec_into(
            current,
            &input,
            &mut expected,
            &mut MatvecScratch::default(),
        )?;
        let mut actual = Vec::new();
        MatvecScratch::default()
            .prepare(current, &input)?
            .matvec_with_executor(current, &mut actual, executor)?;
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        Ok(())
    };
    compare(&tensor, &mut executor)?;
    // Cache retention forces copy-on-write; the original allocation cannot be
    // mutated or reused behind a resident handle.
    Arc::make_mut(&mut changed_data.data)[2] ^= 0xff;
    compare(&changed_data, &mut executor)?;
    compare(&changed_geometry, &mut executor)?;
    assert_eq!(executor.residency_statistics().tensors_prepared, 3);
    assert_eq!(executor.residency_statistics().cache_hits, 0);
    Ok(())
}

#[test]
fn hmx_resident_capacity_refusal_streams_whole_projection_before_preparation() -> Result<()> {
    // Refuse either insufficient native bytes or insufficient host handle slots
    // before any preparation, even when at least one operation would fit.
    for (native_bytes, metadata_bytes) in [(1152, 16_384), (16_384, 128)] {
        let tensor = fixture(256, 35);
        let limits = HmxQ2ProjectionLimits::new(32, 100, Duration::from_secs(10))?;
        let mut submitter = ResidentSubmitter::new(native_bytes)?;
        let mut executor = HmxQ2Executor::new(&mut submitter, limits, 1 << 20)
            .with_resident_weights(metadata_bytes)?;
        let mut actual = Vec::new();
        MatvecScratch::default()
            .prepare(&tensor, &[1.0; 256])?
            .matvec_with_executor(&tensor, &mut actual, &mut executor)?;
        assert_eq!(executor.residency_statistics().tensors_prepared, 0);
        assert!(!executor.is_faulted());
        drop(executor);
        assert_eq!(submitter.preparations, 0);
        assert_eq!(submitter.resident_calls, 0);
        assert_eq!(submitter.streaming_calls, 4);
    }
    Ok(())
}

#[test]
fn hmx_resident_incomplete_preparation_faults_before_any_kernel_or_replay() -> Result<()> {
    let tensor = fixture(256, 35);
    let limits = HmxQ2ProjectionLimits::new(32, 100, Duration::from_secs(10))?;
    let mut submitter = ResidentSubmitter::new(16_384)?;
    submitter.refuse_preparation = Some(2);
    let mut executor =
        HmxQ2Executor::new(&mut submitter, limits, 1 << 20).with_resident_weights(16_384)?;
    let mut output = vec![42.0];
    let mut scratch = MatvecScratch::default();
    let mut prepared = scratch.prepare(&tensor, &[1.0; 256])?;
    assert!(
        prepared
            .matvec_with_executor(&tensor, &mut output, &mut executor)
            .is_err()
    );
    assert_eq!(output, Vec::<f32>::new());
    assert!(executor.is_faulted());
    assert_eq!(executor.statistics().operations_attempted, 0);
    assert_eq!(executor.residency_statistics().tensors_prepared, 0);
    assert!(
        prepared
            .matvec_with_executor(&tensor, &mut output, &mut executor)
            .is_err()
    );
    drop(executor);
    assert_eq!(submitter.preparations, 2);
    assert_eq!(submitter.resident_calls, 0);
    assert_eq!(submitter.streaming_calls, 0);
    Ok(())
}

#[test]
fn hmx_resident_unwritten_output_and_native_refusal_close_without_replay() -> Result<()> {
    for partial in [false, true] {
        let tensor = fixture(256, 1);
        let limits = HmxQ2ProjectionLimits::new(32, 10, Duration::from_secs(10))?;
        let mut submitter = ResidentSubmitter::new(4096)?;
        submitter.oracle.fail = !partial;
        submitter.oracle.partial = partial;
        let mut executor =
            HmxQ2Executor::new(&mut submitter, limits, 1 << 20).with_resident_weights(16_384)?;
        let mut output = vec![42.0];
        let mut scratch = MatvecScratch::default();
        let mut prepared = scratch.prepare(&tensor, &[1.0; 256])?;
        assert!(
            prepared
                .matvec_with_executor(&tensor, &mut output, &mut executor)
                .is_err()
        );
        assert_eq!(output, Vec::<f32>::new());
        assert!(executor.is_faulted());
        assert_eq!(executor.statistics().operations_attempted, 1);
        assert_eq!(executor.statistics().operations_completed, 0);
        assert!(
            prepared
                .matvec_with_executor(&tensor, &mut output, &mut executor)
                .is_err()
        );
        drop(executor);
        assert_eq!(submitter.resident_calls, 1);
        assert_eq!(submitter.streaming_calls, 0);
    }
    Ok(())
}
