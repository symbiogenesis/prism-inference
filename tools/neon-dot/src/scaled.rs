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
    /// Original Q8_K scale, shared by both halves of its quantizer block.
    pub scale: f32,
    /// Original signed codes rearranged into consumption order.
    pub values: [i8; 128],
    /// Exact signed-byte sum subtracted from each unsigned-code dot.
    pub sum: i32,
}

pub(super) fn valid<const ROWS: usize>(rows: [&[u8]; ROWS], activation: &[Pq2Activation]) -> bool {
    matches!(ROWS, 1 | 4 | 8)
        && !activation.is_empty()
        && activation
            .len()
            .checked_mul(34)
            .is_some_and(|bytes| rows.iter().all(|row| row.len() == bytes))
        && activation.iter().all(|block| block.scale.is_finite())
}

#[cfg(any(test, not(target_arch = "aarch64")))]
#[expect(
    clippy::suboptimal_flops,
    reason = "The nonfused contract requires separate product and addition rounding."
)]
pub(super) fn portable<const ROWS: usize, const FUSED: bool>(
    rows: [&[u8]; ROWS],
    activation: &[Pq2Activation],
) -> Option<[f32; ROWS]> {
    if !valid(rows, activation) {
        return None;
    }
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
}
