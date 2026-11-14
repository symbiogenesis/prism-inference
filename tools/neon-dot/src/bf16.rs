//! Ordered binary64 reductions across independent BF16 projection rows.

#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_endian = "little"
))]
#[allow(
    unsafe_code,
    reason = "Baseline-NEON value arithmetic stays behind the checked borrowed row boundary."
)]
#[path = "bf16_aarch64.rs"]
mod aarch64;

/// Multiplies four or eight BF16 rows by one shared binary32 activation row.
///
/// Each product rounds to binary32 before its binary64 addition in original
/// column order, starting from negative zero. Native baseline NEON uses the
/// two binary64 lanes for independent rows; columns are never reassociated.
/// The final result rounds once to binary32. Other targets retain scalar
/// ordered arithmetic. No input is copied, retained or expanded into a matrix.
///
/// Returns `None` for unsupported row counts, empty/mismatched extents,
/// nonfinite weights or results. Failure exposes no partial output.
#[inline]
#[must_use]
pub fn dot_bf16_rows<const ROWS: usize>(rows: [&[u8]; ROWS], input: &[f32]) -> Option<[f32; ROWS]> {
    if !matches!(ROWS, 4 | 8)
        || input.is_empty()
        || input
            .len()
            .checked_mul(2)
            .is_none_or(|bytes| rows.iter().any(|row| row.len() != bytes))
    {
        return None;
    }
    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    ))]
    {
        aarch64::dot_rows(rows, input)
    }
    #[cfg(not(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    )))]
    {
        portable(rows, input)
    }
}

#[cfg(any(
    test,
    not(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    ))
))]
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "The original dot contract rounds its ordered binary64 sum to binary32 once."
)]
fn portable<const ROWS: usize>(rows: [&[u8]; ROWS], input: &[f32]) -> Option<[f32; ROWS]> {
    let mut output = [0.0; ROWS];
    for (row, target) in rows.into_iter().zip(&mut output) {
        let mut sum = -0.0_f64;
        for (weight, &activation) in row.as_chunks::<2>().0.iter().zip(input) {
            let weight = f32::from_bits(u32::from(u16::from_le_bytes(*weight)) << 16);
            if !weight.is_finite() {
                return None;
            }
            sum += f64::from(weight * activation);
        }
        *target = sum as f32;
        if !target.is_finite() {
            return None;
        }
    }
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::{dot_bf16_rows, portable};

    fn rows<const ROWS: usize>(count: usize) -> [Vec<u8>; ROWS] {
        std::array::from_fn(|row: usize| {
            (0..count)
                .flat_map(|column: usize| {
                    let bits = [0x0000_u16, 0x8000, 0x0001, 0x8001, 0x3f80, 0xbf87, 0x4401];
                    bits[(row * 3 + column) % bits.len()].to_le_bytes()
                })
                .collect()
        })
    }

    fn compare<const ROWS: usize>(count: usize) {
        let rows = rows::<ROWS>(count);
        let input: Vec<_> = (0..count)
            .map(|column| [0.0, -0.0, f32::from_bits(1), -0.125, 1.375, -2.5][column % 6])
            .collect();
        let refs = rows.each_ref().map(Vec::as_slice);
        let expected = portable(refs, &input).map(|values| values.map(f32::to_bits));
        assert!(expected.is_some());
        assert_eq!(
            dot_bf16_rows(refs, &input).map(|values| values.map(f32::to_bits)),
            expected,
            "rows {ROWS}, columns {count}"
        );
    }

    #[test]
    fn x1_bf16_paired_rows_match_ordered_scalar_at_blocks_and_tails() {
        for count in [1, 3, 7, 8, 9, 15, 16, 17, 127, 256, 5120] {
            compare::<4>(count);
            compare::<8>(count);
        }
    }

    #[test]
    fn x1_bf16_paired_rows_preserve_order_and_signed_zero() {
        // Ordered f64 additions discard the 1 between +/-2^64. A grouped
        // reduction or independently accumulated column lanes would yield 4.
        for (count, offset) in [(8, 0), (9, 0), (17, 6)] {
            let mut weights = vec![0_u16; count];
            weights[offset..offset + 4].copy_from_slice(&[0x5f80, 0x3f80, 0xdf80, 0x4040]);
            let row: Vec<_> = weights.into_iter().flat_map(u16::to_le_bytes).collect();
            let input = vec![1.0; count];
            assert_eq!(
                dot_bf16_rows::<4>([&row; 4], &input).map(|v| v.map(f32::to_bits)),
                Some([3.0_f32.to_bits(); 4])
            );
            assert_eq!(
                dot_bf16_rows::<8>([&row; 8], &input).map(|v| v.map(f32::to_bits)),
                Some([3.0_f32.to_bits(); 8])
            );
        }
        for count in [1, 7, 8, 9, 17] {
            let row = vec![0; count * 2];
            let input = vec![-1.0; count];
            assert_eq!(
                dot_bf16_rows::<8>([&row; 8], &input).map(|v| v.map(f32::to_bits)),
                Some([(-0.0_f32).to_bits(); 8])
            );
        }
    }

    #[test]
    fn x1_bf16_paired_rows_reject_geometry_and_late_numeric_failures_then_recover() {
        let mut rows = rows::<8>(17);
        let input = [1.0; 17];
        assert!(dot_bf16_rows::<0>([], &input).is_none());
        assert!(dot_bf16_rows::<1>([&rows[0]], &input).is_none());
        assert!(dot_bf16_rows::<4>([&[]; 4], &[]).is_none());
        let refs = rows.each_ref().map(Vec::as_slice);
        let mut short = refs;
        short[7] = &short[7][..33];
        assert!(dot_bf16_rows(short, &input).is_none());
        for row in 0..8 {
            for column in [0, 7, 8, 15, 16] {
                for bits in [0x7f80_u16, 0xff80, 0x7f81, 0x7fc0] {
                    let offset = column * 2;
                    let original = [rows[row][offset], rows[row][offset + 1]];
                    rows[row][offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
                    let refs = rows.each_ref().map(Vec::as_slice);
                    assert!(dot_bf16_rows(refs, &input).is_none());
                    assert!(portable(refs, &input).is_none());
                    rows[row][offset..offset + 2].copy_from_slice(&original);
                    assert!(dot_bf16_rows(rows.each_ref().map(Vec::as_slice), &input).is_some());
                }
            }
        }
        let maximum: Vec<_> = [0x7f7f_u16; 17]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect();
        assert!(dot_bf16_rows::<8>([&maximum; 8], &[2.0; 17]).is_none());
        let zero = vec![0; 34];
        assert!(dot_bf16_rows::<4>([&zero; 4], &[f32::INFINITY; 17]).is_none());
    }
}
