//! Complete-row arithmetic shared by the safe detected boundary and controls.

#![expect(
    clippy::redundant_pub_crate,
    reason = "Parent-only checks and portable controls remain inside the safe scaled-row boundary."
)]

/// Exact prepared activation for one original 128-column PQ2 block.
///
/// Byte planes follow two 64-column halves, each with four 16-byte digit
/// planes. `sum` is the sum of all signed bytes used for the unsigned-code
/// correction. The runtime retains its existing 136-byte block layout.
pub struct Pq2Activation {
    /// Original `Q8_K` scale, shared by both halves of its quantizer block.
    pub scale: f32,
    /// Original signed codes rearranged into consumption order.
    pub values: [i8; 128],
    /// Exact signed-byte sum subtracted from each unsigned-code dot.
    pub sum: i32,
}

pub(super) fn valid<const ROWS: usize>(rows: [&[u8]; ROWS], activation: &[Pq2Activation]) -> bool {
    valid_rows(rows, activation.len()) && activation.iter().all(|block| block.scale.is_finite())
}

pub(super) fn valid_rows<const ROWS: usize>(rows: [&[u8]; ROWS], count: usize) -> bool {
    matches!(ROWS, 1 | 4 | 8 | 16)
        && count != 0
        && count
            .checked_mul(34)
            .is_some_and(|bytes| rows.iter().all(|row| row.len() == bytes))
}

#[cfg(any(test, not(target_arch = "aarch64")))]
pub(super) fn portable<const ROWS: usize, const FUSED: bool>(
    rows: [&[u8]; ROWS],
    activation: &[Pq2Activation],
) -> Option<[f32; ROWS]> {
    if !valid(rows, activation) {
        return None;
    }
    portable_validated::<ROWS, FUSED>(rows, activation)
}

#[cfg(any(test, not(target_arch = "aarch64")))]
#[expect(
    clippy::suboptimal_flops,
    reason = "The nonfused contract requires separate product and addition rounding."
)]
pub(super) fn portable_validated<const ROWS: usize, const FUSED: bool>(
    rows: [&[u8]; ROWS],
    activation: &[Pq2Activation],
) -> Option<[f32; ROWS]> {
    let mut output = [0.0_f32; ROWS];
    for (row, output) in rows.into_iter().zip(&mut output) {
        for (bytes, activation) in row.as_chunks::<34>().0.iter().zip(activation) {
            let scale = super::half::scalar(u16::from_le_bytes([bytes[0], bytes[1]]));
            if !scale.is_finite() {
                return None;
            }
            let mut dot = 0_i32;
            for (half, bytes) in bytes[2..].as_chunks::<16>().0.iter().enumerate() {
                for (index, &byte) in bytes.iter().enumerate() {
                    for digit in 0..4 {
                        dot += i32::from((byte >> (digit * 2)) & 3)
                            * i32::from(activation.values[half * 64 + digit * 16 + index]);
                    }
                }
            }
            // Public callers supply the correction; wrapping matches SIMD
            // subtraction even for an inconsistent but memory-safe input.
            let dot = dot.wrapping_sub(activation.sum);
            #[expect(
                clippy::cast_precision_loss,
                clippy::as_conversions,
                reason = "The exact PQ2 dot bound fits binary32; arbitrary safe caller corrections retain the existing integer-to-float contract."
            )]
            let dot = dot as f32;
            let scale = scale * activation.scale;
            *output = if FUSED {
                scale.mul_add(dot, *output)
            } else {
                *output + scale * dot
            };
        }
        if !output.is_finite() {
            return None;
        }
    }
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::{Pq2Activation, portable};

    #[test]
    fn x1_scoped_pq2_matches_reference_and_revalidates_replaced_activations() {
        #[cfg(target_arch = "aarch64")]
        let Some(detected) = crate::Pq2Dotprod::detect() else {
            return;
        };
        // This test-only capability selects exclusively portable arithmetic on
        // non-ARM targets. It never reaches a target-feature instruction.
        #[cfg(not(target_arch = "aarch64"))]
        let detected = crate::Pq2Dotprod { _detected: () };
        let mut activation = [
            Pq2Activation {
                scale: f32::from_bits(0x3da0_0001),
                values: [3; 128],
                sum: 384,
            },
            Pq2Activation {
                scale: f32::from_bits(0x3da0_0001),
                values: [-13; 128],
                sum: -1664,
            },
        ];
        let mut row = [0xaa; 68];
        row[..2].copy_from_slice(&0x3c01_u16.to_le_bytes());
        row[34..36].copy_from_slice(&0x3787_u16.to_le_bytes());
        {
            let prepared = detected.prepare_scaled(&activation);
            assert!(prepared.is_some());
            if let Some(prepared) = prepared {
                for _ in 0..3 {
                    assert_eq!(
                        prepared
                            .dot_scaled_rows::<1, true>([&row])
                            .map(|v| v.map(f32::to_bits)),
                        Some([0xc1f9_0b01])
                    );
                    assert_eq!(
                        prepared
                            .dot_scaled_rows::<4, false>([&row; 4])
                            .map(|v| v.map(f32::to_bits)),
                        Some([0xc1f9_0b02; 4])
                    );
                    assert_eq!(
                        prepared
                            .dot_scaled_rows::<8, true>([&row; 8])
                            .map(|v| v.map(f32::to_bits)),
                        portable::<8, true>([&row; 8], &activation).map(|v| v.map(f32::to_bits))
                    );
                }
                assert!(prepared.dot_scaled_rows::<2, true>([&row; 2]).is_none());
                assert!(
                    prepared
                        .dot_scaled_rows::<8, true>([&row[..67]; 8])
                        .is_none()
                );
                row[34..36].copy_from_slice(&0x7c01_u16.to_le_bytes());
                assert!(prepared.dot_scaled_rows::<8, true>([&row; 8]).is_none());
            }
        }
        activation[1].scale = f32::NAN;
        assert!(detected.prepare_scaled(&activation).is_none());
        assert!(detected.prepare_scaled(&[]).is_none());
        activation[1].scale = -0.125;
        row[34..36].copy_from_slice(&0x3787_u16.to_le_bytes());
        let expected = portable::<8, true>([&row; 8], &activation).map(|v| v.map(f32::to_bits));
        assert_eq!(
            detected
                .prepare_scaled(&activation)
                .and_then(|scope| scope.dot_scaled_rows::<8, true>([&row; 8]))
                .map(|v| v.map(f32::to_bits)),
            expected
        );
    }

    #[test]
    fn x1_scaled_pq2_rows_preserve_independent_fma_control_and_reject_bad_inputs() {
        let activation = [
            Pq2Activation {
                scale: f32::from_bits(0x3da0_0001),
                values: [3; 128],
                sum: 384,
            },
            Pq2Activation {
                scale: f32::from_bits(0x3da0_0001),
                values: [-13; 128],
                sum: -1664,
            },
        ];
        let mut row = [0xaa; 68];
        row[..2].copy_from_slice(&0x3c01_u16.to_le_bytes());
        row[34..36].copy_from_slice(&0x3787_u16.to_le_bytes());
        assert_eq!(
            portable::<1, true>([&row], &activation).map(|v| v.map(f32::to_bits)),
            Some([0xc1f9_0b01])
        );
        assert_eq!(
            portable::<1, false>([&row], &activation).map(|v| v.map(f32::to_bits)),
            Some([0xc1f9_0b02])
        );
        if let Some(detected) = crate::Pq2Dotprod::detect() {
            assert_eq!(
                detected
                    .dot_scaled_rows::<1, true>([&row], &activation)
                    .map(|v| v.map(f32::to_bits)),
                Some([0xc1f9_0b01])
            );
            assert_eq!(
                detected
                    .dot_scaled_rows::<4, false>([&row; 4], &activation)
                    .map(|v| v.map(f32::to_bits)),
                Some([0xc1f9_0b02; 4])
            );
            assert_eq!(
                detected
                    .dot_scaled_rows::<8, true>([&row; 8], &activation)
                    .map(|v| v.map(f32::to_bits)),
                Some([0xc1f9_0b01; 8])
            );
            assert!(
                detected
                    .dot_scaled_rows::<1, true>([&row[..67]], &activation)
                    .is_none()
            );
            assert!(
                detected
                    .dot_scaled_rows::<2, true>([&row; 2], &activation)
                    .is_none()
            );
            row[34..36].copy_from_slice(&0x7c00_u16.to_le_bytes());
            assert!(
                detected
                    .dot_scaled_rows::<8, true>([&row; 8], &activation)
                    .is_none()
            );
        }
        assert!(portable::<1, true>([&[]], &[]).is_none());
        assert!(portable::<1, true>([&row[..34]], &activation).is_none());
    }

    #[test]
    fn x1_scaled_pq2_rows_match_all_symbols_extremes_and_scale_orders() {
        for count in [1, 2, 3, 40] {
            let activation: Vec<_> = (0..count)
                .map(|block: usize| {
                    let values: [i8; 128] = std::array::from_fn(|lane: usize| {
                        i8::from_ne_bytes([lane.to_le_bytes()[0]
                            .wrapping_mul(71)
                            .wrapping_add(block.to_le_bytes()[0])])
                    });
                    Pq2Activation {
                        scale: [0.0, -0.0, 0.125, -0.375][block % 4],
                        sum: values.iter().map(|&v| i32::from(v)).sum(),
                        values,
                    }
                })
                .collect();
            let rows: [Vec<u8>; 8] = std::array::from_fn(|row: usize| {
                let mut bytes: Vec<u8> = (0..count * 34)
                    .map(|index: usize| {
                        index.to_le_bytes()[0]
                            .wrapping_mul(73)
                            .wrapping_add(row.to_le_bytes()[0])
                    })
                    .collect();
                for (index, block) in bytes.as_chunks_mut::<34>().0.iter_mut().enumerate() {
                    block[..2].copy_from_slice(
                        &[0x0001_u16, 0x8000, 0x3555, 0xb787][(index + row) % 4].to_le_bytes(),
                    );
                }
                bytes
            });
            let refs = rows.each_ref().map(Vec::as_slice);
            let expected_fused =
                portable::<8, true>(refs, &activation).map(|v| v.map(f32::to_bits));
            let expected_separate =
                portable::<8, false>(refs, &activation).map(|v| v.map(f32::to_bits));
            if let Some(detected) = crate::Pq2Dotprod::detect() {
                assert_eq!(
                    detected
                        .dot_scaled_rows::<8, true>(refs, &activation)
                        .map(|v| v.map(f32::to_bits)),
                    expected_fused
                );
                assert_eq!(
                    detected
                        .dot_scaled_rows::<8, false>(refs, &activation)
                        .map(|v| v.map(f32::to_bits)),
                    expected_separate
                );
                for (row, &bytes) in refs.iter().enumerate() {
                    assert_eq!(
                        detected
                            .dot_scaled_rows::<1, true>([bytes], &activation)
                            .map(|v| v[0].to_bits()),
                        expected_fused.map(|v| v[row])
                    );
                }
            }
        }
    }

    #[test]
    fn x1_scaled_pq2_rows_preserve_16_row_parity_and_errors() {
        let mut raw_values = [[0_i8; 128]; 2];
        for (block_idx, block) in raw_values.iter_mut().enumerate() {
            for (elem_idx, slot) in block.iter_mut().enumerate() {
                let raw = i8::try_from((block_idx * 43 + elem_idx * 17) % 199).unwrap_or(0);
                *slot = raw.wrapping_sub(99);
            }
        }
        let activation = [
            Pq2Activation {
                values: raw_values[0],
                scale: 0.125,
                sum: raw_values[0].iter().map(|&v| i32::from(v)).sum(),
            },
            Pq2Activation {
                values: raw_values[1],
                scale: -0.0625,
                sum: raw_values[1].iter().map(|&v| i32::from(v)).sum(),
            },
        ];
        let scale_pool: [u16; 16] = [
            0x3c00, 0xbc00, 0x3800, 0xb800, 0x4000, 0xc000, 0x3400, 0xb400, 0x8000, 0x0000, 0x3e00,
            0xbe00, 0x4200, 0xc200, 0x3600, 0xb600,
        ];
        let mut rows = [[0_u8; 68]; 16];
        for (row_idx, row) in rows.iter_mut().enumerate() {
            for (block_idx, scale_bits) in [scale_pool[row_idx], scale_pool[(row_idx + 7) % 16]]
                .into_iter()
                .enumerate()
            {
                let start = block_idx * 34;
                let le = scale_bits.to_le_bytes();
                row[start] = le[0];
                row[start + 1] = le[1];
                for byte_idx in 0..32 {
                    row[start + 2 + byte_idx] =
                        u8::try_from((row_idx * 29 + block_idx * 53 + byte_idx * 13 + 3) % 256)
                            .unwrap_or(0);
                }
            }
        }
        let refs: [&[u8]; 16] = std::array::from_fn(|row| rows[row].as_slice());
        let expected_fused = portable::<16, true>(refs, &activation);
        let expected_unfused = portable::<16, false>(refs, &activation);
        assert!(expected_fused.is_some());
        assert!(expected_unfused.is_some());

        if let Some(detected) = crate::Pq2Dotprod::detect() {
            assert_eq!(
                detected
                    .dot_scaled_rows::<16, true>(refs, &activation)
                    .map(|values| values.map(f32::to_bits)),
                expected_fused.map(|values| values.map(f32::to_bits))
            );
            assert_eq!(
                detected
                    .dot_scaled_rows::<16, false>(refs, &activation)
                    .map(|values| values.map(f32::to_bits)),
                expected_unfused.map(|values| values.map(f32::to_bits))
            );
            assert_eq!(
                detected
                    .prepare_scaled(&activation)
                    .and_then(|prepared| prepared.dot_scaled_rows::<16, true>(refs))
                    .map(|values| values.map(f32::to_bits)),
                expected_fused.map(|values| values.map(f32::to_bits))
            );
            for (row, &bytes) in refs.iter().enumerate() {
                assert_eq!(
                    detected
                        .dot_scaled_rows::<1, true>([bytes], &activation)
                        .map(|value| value[0].to_bits()),
                    expected_fused.map(|values| values[row].to_bits())
                );
            }

            let mut short_refs = refs;
            short_refs[15] = &rows[15][..67];
            assert!(
                detected
                    .dot_scaled_rows::<16, true>(short_refs, &activation)
                    .is_none()
            );

            let mut bad_scale_rows = rows;
            let nan_bits = 0x7e00_u16.to_le_bytes();
            bad_scale_rows[11][34] = nan_bits[0];
            bad_scale_rows[11][35] = nan_bits[1];
            let bad_refs: [&[u8]; 16] = std::array::from_fn(|row| bad_scale_rows[row].as_slice());
            assert!(
                detected
                    .dot_scaled_rows::<16, true>(bad_refs, &activation)
                    .is_none()
            );

            let mut inf_act = activation;
            inf_act[1].scale = f32::INFINITY;
            assert!(
                detected
                    .dot_scaled_rows::<16, true>(refs, &inf_act)
                    .is_none()
            );
        }
    }
}
