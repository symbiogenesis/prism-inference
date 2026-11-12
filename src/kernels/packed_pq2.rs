//! Scoped PQ2 activation layout for baseline NEON's packed consumption order.
//!
//! Only activation bytes are rearranged. Original `Q8_K` blocks remain available
//! to other formats and external executors; weights never expand or repack.

use bonsai_ternary::Q8KBlock;

use crate::{Result, invalid};

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
pub(super) use bonsai_neon_dot::Pq2Activation as Activation;

#[cfg(not(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
)))]
/// One original 128-weight block's exact scale and reordered activation bytes.
pub(super) struct Activation {
    pub(super) scale: f32,
    pub(super) values: [i8; 128],
}

/// Byte planes shared by every backend, with wide planes only for baseline NEON.
#[derive(Default)]
pub(super) struct Activations {
    blocks: Vec<Activation>,
    pub(super) widened: Vec<[i16; 128]>,
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    pub(super) dotprod: Option<bonsai_neon_dot::Pq2Dotprod>,
}

impl std::ops::Deref for Activations {
    type Target = [Activation];

    fn deref(&self) -> &Self::Target {
        &self.blocks
    }
}

impl Activations {
    pub(super) fn clear(&mut self) {
        self.blocks.clear();
        self.widened.clear();
        #[cfg(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc",
            target_feature = "neon"
        ))]
        {
            self.dotprod = None;
        }
    }

    #[cfg(test)]
    pub(super) const fn capacity(&self) -> usize {
        self.blocks.capacity()
    }
}

/// Prepare four digit planes for each consecutive 16-byte weight load.
pub(super) fn prepare(input: &[Q8KBlock], output: &mut Activations) -> Result<()> {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    let dotprod = bonsai_neon_dot::Pq2Dotprod::detect();
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    let widened = dotprod.is_none();
    #[cfg(not(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    )))]
    let widened = false;
    prepare_with_layout(input, output, widened)?;
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    {
        output.dotprod = dotprod;
    }
    Ok(())
}

fn prepare_with_layout(input: &[Q8KBlock], output: &mut Activations, widened: bool) -> Result<()> {
    output.clear();
    let result = (|| {
        let count = input
            .len()
            .checked_mul(2)
            .ok_or_else(|| invalid("packed activation extent overflow"))?;
        output.blocks.try_reserve_exact(count)?;
        if widened {
            output.widened.try_reserve_exact(count)?;
        }
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
                    let wide = if widened {
                        output.widened.push([0; 128]);
                        output.widened.last_mut()
                    } else {
                        None
                    };
                    let (values, sum) = super::neon::prepare_pq2_activations(source, wide);
                    output.blocks.push(Activation { scale, values, sum });
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
                    if widened {
                        output.widened.push(values.map(i16::from));
                    }
                    output.blocks.push(Activation { scale, values });
                }
            }
        }
        Ok(())
    })();
    if result.is_err() {
        output.clear();
    }
    result
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
#[expect(
    clippy::suboptimal_flops,
    reason = "Preserve original per-128 scalar multiply/add or pinned FMA rounding."
)]
pub(super) fn dot_row<const FUSED: bool>(bytes: &[u8], input: &Activations) -> Result<f32> {
    if input.is_empty() || bytes.len() / 34 != input.len() || !bytes.len().is_multiple_of(34) {
        return Err(invalid("incompatible packed consumption activation extent"));
    }
    if let Some(dotprod) = input.dotprod {
        return dotprod
            .dot_scaled_rows::<1, FUSED>([bytes], input)
            .ok_or_else(|| invalid("nonfinite tensor or activation value"))
            .map(|output| output[0]);
    }
    let mut output = 0.0;
    for (index, (block, activation)) in bytes
        .as_chunks::<34>()
        .0
        .iter()
        .zip(input.iter())
        .enumerate()
    {
        let d = super::scale(block)? * activation.scale;
        let sum = super::integer_float(super::neon::dot_consumption_order_prewidened(
            block[2..].try_into()?,
            &activation.values,
            input.widened.get(index),
            activation.sum,
            input.dotprod,
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
#[expect(
    clippy::suboptimal_flops,
    reason = "Preserve original per-128 scalar multiply/add or pinned FMA rounding."
)]
pub(super) fn dot_four_rows<const FUSED: bool>(
    rows: [&[u8]; 4],
    input: &Activations,
) -> Result<[f32; 4]> {
    if input.is_empty()
        || rows
            .iter()
            .any(|bytes| bytes.len() / 34 != input.len() || !bytes.len().is_multiple_of(34))
    {
        return Err(invalid("incompatible packed consumption activation extent"));
    }
    if let Some(dotprod) = input.dotprod {
        return dotprod
            .dot_scaled_rows::<4, FUSED>(rows, input)
            .ok_or_else(|| invalid("nonfinite tensor or activation value"));
    }
    if let Some(result) = super::neon::dot_four_rows_consumption::<FUSED>(rows, input) {
        return result;
    }
    let mut outputs = [0.0_f32; 4];
    let c0 = rows[0].as_chunks::<34>().0;
    let c1 = rows[1].as_chunks::<34>().0;
    let c2 = rows[2].as_chunks::<34>().0;
    let c3 = rows[3].as_chunks::<34>().0;
    for ((((b0, b1), b2), b3), activation) in c0.iter().zip(c1).zip(c2).zip(c3).zip(input.iter()) {
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
    input: &Activations,
) -> Result<[f32; 8]> {
    if input.is_empty()
        || rows
            .iter()
            .any(|bytes| bytes.len() / 34 != input.len() || !bytes.len().is_multiple_of(34))
    {
        return Err(invalid("incompatible packed consumption activation extent"));
    }
    if let Some(dotprod) = input.dotprod {
        return dotprod
            .dot_scaled_rows::<8, FUSED>(rows, input)
            .ok_or_else(|| invalid("nonfinite tensor or activation value"));
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
    fn x1_pq2_compact_preparation_preserves_baseline_planes_and_clears_both_layouts() -> Result<()>
    {
        let mut input = [Q8KBlock {
            scale: f32::from_bits(0x3eaa_aaab),
            sums: [123; 16],
            values: std::array::from_fn(|index| i8::from_ne_bytes([index.to_le_bytes()[0]])),
        }];
        let mut compact = super::Activations::default();
        let mut baseline = super::Activations::default();
        super::prepare_with_layout(&input, &mut compact, false)?;
        super::prepare_with_layout(&input, &mut baseline, true)?;
        assert_eq!(compact.len(), 2);
        assert_eq!(compact.widened.capacity(), 0);
        assert_eq!(baseline.widened.len(), 2);
        for (index, (actual, wide)) in compact.iter().zip(&baseline.widened).enumerate() {
            assert_eq!(actual.scale.to_bits(), baseline[index].scale.to_bits());
            assert_eq!(actual.values, baseline[index].values);
            for lane in 0..128 {
                let source = index * 128 + (lane / 64) * 64 + (lane % 16) * 4 + (lane / 16) % 4;
                assert_eq!(actual.values[lane], input[0].values[source]);
                assert_eq!(wide[lane], i16::from(input[0].values[source]));
            }
        }
        // Native compact blocks have scale + bytes + exact integer correction,
        // independent of whether this host can execute the optional ISA.
        assert!(size_of::<super::Activation>() <= 136);
        input[0].scale = f32::NAN;
        assert!(super::prepare_with_layout(&input, &mut baseline, true).is_err());
        assert!(baseline.is_empty());
        assert_eq!(baseline.widened.len(), 0);
        assert!(super::prepare_with_layout(&input, &mut compact, false).is_err());
        assert!(compact.is_empty());
        assert_eq!(compact.widened.len(), 0);
        Ok(())
    }

    #[test]
    fn packed_consumption_preparation_preserves_each_lane_scale_and_source() -> Result<()> {
        let input = [Q8KBlock {
            scale: f32::from_bits(0x3eaa_aaab),
            sums: [123; 16],
            values: std::array::from_fn(|index| i8::from_ne_bytes([index.to_le_bytes()[0]])),
        }];
        let original = input[0].values;
        let mut output = super::Activations::default();
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
        let mut output = super::Activations::default();
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
    fn x1_pq2_compact_prefill_matches_independent_scalar_rows() -> Result<()> {
        let tensor = weights(512, 9);
        for count in [1, 8, 9, 16, 32] {
            let inputs: Vec<Vec<f32>> = (0..count)
                .map(|token| {
                    (0..512)
                        .map(|lane| [0.0, -0.0, 127.0, -128.0, 63.5, -3.25][(lane + token) % 6])
                        .collect()
                })
                .collect();
            for pinned in [false, true] {
                let mut tile = crate::kernels::prefill::ProjectionTile::with_backend(
                    MatvecBackend::ExperimentalNeon,
                );
                #[cfg(feature = "parallel-rows")]
                tile.set_row_parallelism(RowParallelism::new(4)?);
                let mut outputs = vec![Vec::new(); count];
                tile.project(&tensor, &inputs, &mut outputs, pinned, &|| Ok(()))?;
                let mut reference = if pinned {
                    MatvecScratch::with_pinned_backend(MatvecBackend::Scalar)
                } else {
                    MatvecScratch::with_backend(MatvecBackend::Scalar)
                };
                for (input, output) in inputs.iter().zip(outputs) {
                    let mut expected = Vec::new();
                    matvec_into(&tensor, input, &mut expected, &mut reference)?;
                    assert_eq!(
                        output.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                        expected.into_iter().map(f32::to_bits).collect::<Vec<_>>()
                    );
                }
            }
        }
        Ok(())
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
    fn x1_pq2_prepared_dispatch_matches_forced_baseline_and_clears_on_failure() -> Result<()> {
        let mut input = [Q8KBlock {
            scale: f32::from_bits(0x3eaa_aaab),
            values: std::array::from_fn(|index| i8::from_ne_bytes([index.to_le_bytes()[0]])),
            sums: [0; 16],
        }];
        let mut detected = super::Activations::default();
        let mut baseline = super::Activations::default();
        prepare(&input, &mut detected)?;
        super::prepare_with_layout(&input, &mut baseline, true)?;
        assert_eq!(detected.dotprod.is_some(), bonsai_neon_dot::has_dotprod());
        assert!(baseline.dotprod.is_none());
        let rows: [[u8; 68]; 8] = std::array::from_fn(|row| {
            let mut bytes = std::array::from_fn(|index: usize| {
                index.to_le_bytes()[0]
                    .wrapping_mul(73)
                    .wrapping_add(row.to_le_bytes()[0])
            });
            bytes[..2].copy_from_slice(&0x3555_u16.to_le_bytes());
            bytes[34..36].copy_from_slice(&0xb787_u16.to_le_bytes());
            bytes
        });
        let refs: [&[u8]; 8] = std::array::from_fn(|row| rows[row].as_slice());
        for (pinned, expected) in [
            (true, dot_eight_rows::<true>(refs, &baseline)?),
            (false, dot_eight_rows::<false>(refs, &baseline)?),
        ] {
            let actual = if pinned {
                dot_eight_rows::<true>(refs, &detected)?
            } else {
                dot_eight_rows::<false>(refs, &detected)?
            };
            assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
            let (single, four) = if pinned {
                (
                    dot_row::<true>(refs[0], &detected)?,
                    dot_four_rows::<true>([refs[0], refs[1], refs[2], refs[3]], &detected)?,
                )
            } else {
                (
                    dot_row::<false>(refs[0], &detected)?,
                    dot_four_rows::<false>([refs[0], refs[1], refs[2], refs[3]], &detected)?,
                )
            };
            assert_eq!(single.to_bits(), expected[0].to_bits());
            assert_eq!(
                four.map(f32::to_bits),
                [expected[0], expected[1], expected[2], expected[3]].map(f32::to_bits)
            );
        }
        let mut malformed = rows;
        malformed[7][34..36].copy_from_slice(&0x7c01_u16.to_le_bytes());
        let refs: [&[u8]; 8] = std::array::from_fn(|row| malformed[row].as_slice());
        assert!(dot_eight_rows::<true>(refs, &detected).is_err());
        assert!(dot_eight_rows::<false>(refs, &detected).is_err());
        assert!(dot_row::<true>(refs[7], &detected).is_err());
        assert!(dot_four_rows::<true>([refs[4], refs[5], refs[6], refs[7]], &detected).is_err());
        detected.clear();
        assert!(detected.dotprod.is_none());
        prepare(&input, &mut detected)?;
        input[0].scale = f32::NAN;
        assert!(prepare(&input, &mut detected).is_err());
        assert!(detected.is_empty());
        assert!(detected.dotprod.is_none());
        Ok(())
    }

    #[test]
    fn x1_packed_consumption_preserves_independent_pinned_fma_rounding_control() -> Result<()> {
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
        let mut prepared = super::Activations::default();
        prepare(&input, &mut prepared)?;
        assert_eq!(prepared.widened.is_empty(), bonsai_neon_dot::has_dotprod());
        if bonsai_neon_dot::has_dotprod() {
            assert_eq!(prepared.widened.capacity(), 0);
        }
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
        assert!(dot_row::<true>(&[], &super::Activations::default()).is_err());
        assert!(dot_four_rows::<true>([&weights[..34]; 4], &prepared).is_err());
        assert!(dot_eight_rows::<true>([&weights[..34]; 8], &prepared).is_err());
        weights[..2].copy_from_slice(&0x7c00_u16.to_le_bytes());
        assert!(dot_row::<true>(&weights, &prepared).is_err());
        assert!(dot_four_rows::<true>([&weights; 4], &prepared).is_err());
        assert!(dot_eight_rows::<true>([&weights; 8], &prepared).is_err());
        Ok(())
    }
}
