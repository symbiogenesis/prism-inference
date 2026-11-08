//! Scoped PQ2 activation layout for baseline NEON's packed consumption order.
//!
//! Only activation bytes are rearranged. Original `Q8_K` blocks remain available
//! to other formats and external executors; weights never expand or repack.

use bonsai_ternary::Q8KBlock;

use crate::{Result, invalid};

/// One original 128-weight block's exact scale and reordered activation bytes.
pub(super) struct Activation {
    pub(super) scale: f32,
    pub(super) values: [i8; 128],
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    pub(super) widened: [i16; 128],
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    pub(super) sum: i32,
}

/// Prepare four digit planes for each consecutive 16-byte weight load.
pub(super) fn prepare(input: &[Q8KBlock], output: &mut Vec<Activation>) -> Result<()> {
    super::fill_into(output, |output| {
        output.try_reserve_exact(
            input
                .len()
                .checked_mul(2)
                .ok_or_else(|| invalid("packed activation extent overflow"))?,
        )?;
        for block in input {
            let scale = super::finite(block.scale)?;
            for source in block.values.as_chunks::<128>().0 {
                #[cfg(all(
                    feature = "experimental-neon",
                    target_arch = "aarch64",
                    target_os = "windows",
                    target_env = "msvc",
                    target_feature = "neon"
                ))]
                {
                    let (values, widened, sum) = super::neon::prepare_pq2_activations(source);
                    output.push(Activation {
                        scale,
                        values,
                        widened,
                        sum,
                    });
                }
                #[cfg(not(all(
                    feature = "experimental-neon",
                    target_arch = "aarch64",
                    target_os = "windows",
                    target_env = "msvc",
                    target_feature = "neon"
                )))]
                {
                    let values = std::array::from_fn(|index| {
                        source[(index / 64) * 64 + (index % 16) * 4 + (index / 16) % 4]
                    });
                    output.push(Activation { scale, values });
                }
            }
        }
        Ok(())
    })
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
#[allow(
    clippy::suboptimal_flops,
    reason = "Preserve original per-128 scalar multiply/add or pinned FMA rounding."
)]
pub(super) fn dot_row<const FUSED: bool>(bytes: &[u8], input: &[Activation]) -> Result<f32> {
    if input.is_empty() || bytes.len() / 34 != input.len() || !bytes.len().is_multiple_of(34) {
        return Err(invalid("incompatible packed consumption activation extent"));
    }
    let mut output = 0.0;
    for (block, activation) in bytes.as_chunks::<34>().0.iter().zip(input) {
        let d = super::scale(block)? * activation.scale;
        let sum = super::integer_float(super::neon::dot_consumption_order_prewidened(
            block[2..].try_into()?,
            &activation.values,
            &activation.widened,
            activation.sum,
        ));
        output = if FUSED {
            d.mul_add(sum, output)
        } else {
            output + d * sum
        };
    }
    super::finite(output)
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
#[allow(
    clippy::suboptimal_flops,
    reason = "Preserve original per-128 scalar multiply/add or pinned FMA rounding."
)]
pub(super) fn dot_four_rows<const FUSED: bool>(
    rows: [&[u8]; 4],
    input: &[Activation],
) -> Result<[f32; 4]> {
    if input.is_empty()
        || rows
            .iter()
            .any(|bytes| bytes.len() / 34 != input.len() || !bytes.len().is_multiple_of(34))
    {
        return Err(invalid("incompatible packed consumption activation extent"));
    }
    if let Some(result) = super::neon::dot_four_rows_consumption::<FUSED>(rows, input) {
        return result;
    }
    let mut outputs = [0.0_f32; 4];
    let c0 = rows[0].as_chunks::<34>().0;
    let c1 = rows[1].as_chunks::<34>().0;
    let c2 = rows[2].as_chunks::<34>().0;
    let c3 = rows[3].as_chunks::<34>().0;
    for ((((b0, b1), b2), b3), activation) in c0.iter().zip(c1).zip(c2).zip(c3).zip(input) {
        let act_scale = activation.scale;
        let ds = [
            super::scale(b0)? * act_scale,
            super::scale(b1)? * act_scale,
            super::scale(b2)? * act_scale,
            super::scale(b3)? * act_scale,
        ];
        let sums = super::neon::dot_consumption_order_four(
            [
                b0[2..].try_into()?,
                b1[2..].try_into()?,
                b2[2..].try_into()?,
                b3[2..].try_into()?,
            ],
            &activation.values,
        );
        for lane in 0..4 {
            let sum = super::integer_float(sums[lane]);
            outputs[lane] = if FUSED {
                ds[lane].mul_add(sum, outputs[lane])
            } else {
                outputs[lane] + ds[lane] * sum
            };
        }
    }
    Ok([
        super::finite(outputs[0])?,
        super::finite(outputs[1])?,
        super::finite(outputs[2])?,
        super::finite(outputs[3])?,
    ])
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
pub(super) fn dot_eight_rows<const FUSED: bool>(
    rows: [&[u8]; 8],
    input: &[Activation],
) -> Result<[f32; 8]> {
    if input.is_empty()
        || rows
            .iter()
            .any(|bytes| bytes.len() / 34 != input.len() || !bytes.len().is_multiple_of(34))
    {
        return Err(invalid("incompatible packed consumption activation extent"));
    }
    if let Some(result) = super::neon::dot_eight_rows_consumption::<FUSED>(rows, input) {
        return result;
    }
    let lo = dot_four_rows::<FUSED>([rows[0], rows[1], rows[2], rows[3]], input)?;
    let hi = dot_four_rows::<FUSED>([rows[4], rows[5], rows[6], rows[7]], input)?;
    Ok([lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]])
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bonsai_ternary::Q8KBlock;

    use super::prepare;
    #[cfg(feature = "parallel-rows")]
    use crate::kernels::RowParallelism;
    use crate::{
        Result,
        kernels::{MatvecBackend, MatvecScratch, matvec_into},
        model::Tensor,
    };

    #[test]
    fn packed_consumption_preparation_preserves_each_lane_scale_and_source() -> Result<()> {
        let input = [Q8KBlock {
            scale: f32::from_bits(0x3eaa_aaab),
            sums: [123; 16],
            values: std::array::from_fn(|index| i8::from_ne_bytes([index.to_le_bytes()[0]])),
        }];
        let original = input[0].values;
        let mut output = Vec::new();
        prepare(&input, &mut output)?;
        assert_eq!(output.len(), 2);
        for (half, activation) in output.iter().enumerate() {
            assert_eq!(activation.scale.to_bits(), input[0].scale.to_bits());
            for load in 0..2 {
                for byte in 0..16 {
                    for digit in 0..4 {
                        assert_eq!(
                            activation.values[load * 64 + digit * 16 + byte],
                            original[half * 128 + load * 64 + byte * 4 + digit]
                        );
                    }
                }
            }
        }
        assert_eq!(input[0].values, original);
        assert_eq!(input[0].sums, [123; 16]);
        Ok(())
    }

    #[test]
    fn packed_consumption_preparation_replaces_previous_data_and_clears_failures() -> Result<()> {
        let mut input = [Q8KBlock {
            scale: -0.0,
            sums: [0; 16],
            values: [-128; 256],
        }];
        let mut output = Vec::new();
        prepare(&input, &mut output)?;
        assert_eq!(output[0].scale.to_bits(), (-0.0_f32).to_bits());
        let capacity = output.capacity();
        input[0].values.fill(127);
        input[0].scale = 0.5;
        prepare(&input, &mut output)?;
        assert_eq!(output.len(), 2);
        assert!(output.iter().all(|block| block.values == [127; 128]));
        assert_eq!(output.capacity(), capacity);
        input[0].scale = f32::INFINITY;
        assert!(prepare(&input, &mut output).is_err());
        assert!(output.is_empty());
        assert_eq!(output.capacity(), capacity);
        prepare(&[], &mut output)?;
        assert!(output.is_empty());
        Ok(())
    }

    fn weights(columns: usize, rows: usize) -> Tensor {
        let blocks = columns / 128;
        let mut data = Vec::new();
        for index in 0..blocks * rows {
            let scale = [0x0000_u16, 0x8000, 0x3555, 0x3c01, 0xb787][index % 5];
            data.extend_from_slice(&scale.to_le_bytes());
            data.extend((0..32_usize).map(|byte| {
                (byte + index).to_le_bytes()[0]
                    .wrapping_mul(197)
                    .wrapping_add(13)
            }));
        }
        Tensor {
            name: "packed-consumption-control".to_owned(),
            dimensions: vec![columns, rows],
            kind: super::super::PQ2,
            data: Arc::from(data),
        }
    }

    #[test]
    fn packed_consumption_dispatch_matches_scalar_and_reuses_only_the_current_scope() -> Result<()>
    {
        for columns in [256, 512, 5120] {
            let tensor = weights(columns, 7);
            let mut input: Vec<_> = (0..columns)
                .map(|index| [0.0, -0.0, 1.0, -1.0, 1.5, 127.0, -128.0][index % 7])
                .collect();
            for pinned in [false, true] {
                let make_scratch = |backend| {
                    if pinned {
                        MatvecScratch::with_pinned_backend(backend)
                    } else {
                        MatvecScratch::with_backend(backend)
                    }
                };
                let mut reference = make_scratch(MatvecBackend::Scalar);
                let mut actual = make_scratch(MatvecBackend::ExperimentalNeon);
                let (mut expected, mut output) = (Vec::new(), Vec::new());
                for preparations in 1..=2 {
                    matvec_into(&tensor, &input, &mut expected, &mut reference)?;
                    {
                        let mut prepared = actual.prepare(&tensor, &input)?;
                        prepared.matvec_into(&tensor, &mut output)?;
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
                        let storage = prepared.scratch.pq2_consumption.as_ptr();
                        let count = prepared.scratch.pq2_consumption.len();
                        assert_eq!(
                            count,
                            if matches!(
                                prepared.scratch.selected_backend(),
                                MatvecBackend::ExperimentalNeon
                            ) {
                                columns / 128
                            } else {
                                0
                            }
                        );
                        let original = prepared.scratch.q8k[0].values;
                        prepared.matvec_into(&tensor, &mut output)?;
                        assert_eq!(prepared.scratch.pq2_consumption.as_ptr(), storage);
                        assert_eq!(prepared.scratch.pq2_consumption.len(), count);
                        assert_eq!(prepared.scratch.q8k[0].values, original);
                        assert_eq!(prepared.scratch.preparation_count(), preparations);
                    }
                    // Mutate the same allocation after the guard ends: neither
                    // its address nor retained scratch can authorize reuse.
                    input.rotate_left(1);
                    input[0] = -63.5;
                    assert!(actual.prepare(&tensor, &[1.0]).is_err());
                    assert!(actual.pq2_consumption.is_empty());
                }
            }
        }
        Ok(())
    }

    #[test]
    fn packed_consumption_failures_clear_output_and_leave_scalar_storage_unprepared() -> Result<()>
    {
        let tensor = weights(256, 3);
        let mut input = vec![1.0; 256];
        let mut scratch = MatvecScratch::with_backend(MatvecBackend::Scalar);
        let mut output = vec![99.0];
        matvec_into(&tensor, &input, &mut output, &mut scratch)?;
        assert!(scratch.pq2_consumption.is_empty());
        let mut bad = tensor.clone();
        let mut bytes = bad.data.to_vec();
        bytes[68..70].copy_from_slice(&0x7c00_u16.to_le_bytes());
        bad.data = Arc::from(bytes);
        for backend in [MatvecBackend::Scalar, MatvecBackend::ExperimentalNeon] {
            scratch = MatvecScratch::with_backend(backend);
            assert!(matvec_into(&bad, &input, &mut output, &mut scratch).is_err());
            assert_eq!(output, [0.0_f32; 0]);
            assert!(!scratch.diagnostics().succeeded);
            matvec_into(&tensor, &input, &mut output, &mut scratch)?;
            assert_eq!(output.len(), 3);
            input[0] = f32::NAN;
            assert!(matvec_into(&tensor, &input, &mut output, &mut scratch).is_err());
            assert_eq!(output, [0.0_f32; 0]);
            assert!(scratch.pq2_consumption.is_empty());
            input[0] = 1.0;
        }
        Ok(())
    }

    #[cfg(feature = "parallel-rows")]
    #[test]
    fn packed_consumption_parallel_rows_match_scalar_for_partial_worker_tiles() -> Result<()> {
        for rows in [1, 3, 4, 5, 7, 8, 9, 17, 25] {
            let tensor = weights(512, rows);
            let input: Vec<_> = (0..512)
                .map(|index| [1.5, -127.0, 0.0, -0.0, 63.5][index % 5])
                .collect();
            let mut scalar = MatvecScratch::with_pinned_backend(MatvecBackend::Scalar);
            let mut parallel = MatvecScratch::with_pinned_backend(MatvecBackend::ExperimentalNeon);
            parallel.set_row_parallelism(RowParallelism::new(4)?);
            let (mut expected, mut actual) = (Vec::new(), Vec::new());
            matvec_into(&tensor, &input, &mut expected, &mut scalar)?;
            matvec_into(&tensor, &input, &mut actual, &mut parallel)?;
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
            assert_eq!(
                parallel.statistics().parallel_row_operations,
                u64::from(rows > 1)
            );
        }
        Ok(())
    }
}

#[cfg(all(
    test,
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
mod native_tests {
    use bonsai_ternary::Q8KBlock;

    use super::{dot_eight_rows, dot_four_rows, dot_row, prepare};
    use crate::Result;

    #[test]
    fn packed_consumption_preserves_independent_pinned_fma_rounding_control() -> Result<()> {
        // Frozen native arithmetic control from kernels' original PQ2 test.
        // Its pinned FMA differs by one bit from multiply-then-add.
        let mut weights = [0xaa; 68];
        weights[..2].copy_from_slice(&0x3c01_u16.to_le_bytes());
        weights[34..36].copy_from_slice(&0x3787_u16.to_le_bytes());
        let mut values = [3; 256];
        values[128..].fill(-13);
        let mut sums = [48; 16];
        sums[8..].fill(-208);
        let input = [Q8KBlock {
            scale: f32::from_bits(0x3da0_0001),
            values,
            sums,
        }];
        let mut prepared = Vec::new();
        prepare(&input, &mut prepared)?;
        assert_eq!(dot_row::<true>(&weights, &prepared)?.to_bits(), 0xc1f9_0b01);
        assert_eq!(
            dot_row::<false>(&weights, &prepared)?.to_bits(),
            0xc1f9_0b02
        );
        assert_eq!(
            dot_four_rows::<true>([&weights; 4], &prepared)?.map(f32::to_bits),
            [0xc1f9_0b01; 4]
        );
        assert_eq!(
            dot_four_rows::<false>([&weights; 4], &prepared)?.map(f32::to_bits),
            [0xc1f9_0b02; 4]
        );
        assert_eq!(
            dot_eight_rows::<true>([&weights; 8], &prepared)?.map(f32::to_bits),
            [0xc1f9_0b01; 8]
        );
        assert_eq!(
            dot_eight_rows::<false>([&weights; 8], &prepared)?.map(f32::to_bits),
            [0xc1f9_0b02; 8]
        );
        assert!(dot_row::<true>(&weights[..34], &prepared).is_err());
        assert!(dot_row::<true>(&weights[..67], &prepared).is_err());
        assert!(dot_row::<true>(&[], &[]).is_err());
        assert!(dot_four_rows::<true>([&weights[..34]; 4], &prepared).is_err());
        assert!(dot_eight_rows::<true>([&weights[..34]; 8], &prepared).is_err());
        weights[..2].copy_from_slice(&0x7c00_u16.to_le_bytes());
        assert!(dot_row::<true>(&weights, &prepared).is_err());
        assert!(dot_four_rows::<true>([&weights; 4], &prepared).is_err());
        assert!(dot_eight_rows::<true>([&weights; 8], &prepared).is_err());
        Ok(())
    }
}
