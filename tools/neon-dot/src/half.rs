//! Exact binary16 widening and GGML-compatible narrowing with baseline-NEON fast paths.

#![expect(
    clippy::redundant_pub_crate,
    reason = "Parent-only fallback helper remains inside the safe binary16 boundary."
)]

#[cfg(any(not(all(target_arch = "aarch64", target_feature = "neon")), test))]
use bonsai_ternary::checked_float_to_half;
use bonsai_ternary::{float_to_half, half_to_float};

/// Widens four IEEE binary16 bit patterns without changing NaN payload bits.
///
/// Native `AArch64` NEON uses `FCVTL` for finite inputs; this conversion does not
/// require the optional full FP16 arithmetic extension. Exceptional arrays use
/// integer decoding, preserving signaling NaNs without submitting them to a
/// floating-point conversion. Other targets use that same portable decoding.
#[inline]
#[must_use]
pub fn widen_f16_four(bits: &[u16; 4]) -> [f32; 4] {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        super::aarch64::widen_f16_four(*bits)
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        bits.map(half_to_float)
    }
}

/// Widens eight IEEE binary16 bit patterns in their original lane order.
///
/// Uses baseline `FCVTL`/`FCVTL2` for finite `AArch64` NEON inputs, with the same
/// exact exceptional and portable behavior as [`widen_f16_four`].
#[inline]
#[must_use]
pub fn widen_f16_eight(bits: &[u16; 8]) -> [[f32; 4]; 2] {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        super::aarch64::widen_f16_eight(bits)
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        portable_eight(bits)
    }
}

#[inline]
pub(super) fn portable_eight(bits: &[u16; 8]) -> [[f32; 4]; 2] {
    std::array::from_fn(|group| std::array::from_fn(|lane| half_to_float(bits[group * 4 + lane])))
}

/// Widens `src.len().min(dst.len())` IEEE binary16 words into `dst`, using baseline `FCVTL`/`FCVTL2`
/// chunks when NEON is enabled and preserving exceptional/NaN bit patterns via `bonsai_ternary`.
#[inline]
pub fn widen_f16_slice(src: &[u16], dst: &mut [f32]) {
    let count = src.len().min(dst.len());
    let (src_chunks, src_tail) = src[..count].as_chunks::<8>();
    let (dst_chunks, dst_tail) = dst[..count].as_chunks_mut::<8>();
    for (chunk, out_chunk) in src_chunks.iter().zip(dst_chunks.iter_mut()) {
        let [lo, hi] = widen_f16_eight(chunk);
        out_chunk[..4].copy_from_slice(&lo);
        out_chunk[4..].copy_from_slice(&hi);
    }
    let (src_four, src_rem) = src_tail.as_chunks::<4>();
    let (dst_four, dst_rem) = dst_tail.as_chunks_mut::<4>();
    if let (Some(chunk), Some(out_chunk)) = (src_four.first(), dst_four.first_mut()) {
        *out_chunk = widen_f16_four(chunk);
    }
    for (&bits, slot) in src_rem.iter().zip(dst_rem.iter_mut()) {
        *slot = half_to_float(bits);
    }
}

/// Narrows four `f32` values to IEEE binary16 bit patterns matching GGML rounding.
///
/// Native `AArch64` NEON uses hardware `FCVTN` when all four lanes are non-NaN, falling back per
/// chunk when a NaN is present so canonical quiet NaN bits match GGML scalar rounding.
#[inline]
#[must_use]
pub fn narrow_f32_four(values: [f32; 4]) -> [u16; 4] {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        super::aarch64::narrow_f32_four(values)
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        values.map(float_to_half)
    }
}

/// Narrows eight `f32` values to IEEE binary16 bit patterns matching GGML rounding.
///
/// Native `AArch64` NEON uses hardware `FCVTN`/`FCVTN2` when all eight lanes are non-NaN.
#[inline]
#[must_use]
pub fn narrow_f32_eight(values: &[f32; 8]) -> [u16; 8] {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        super::aarch64::narrow_f32_eight(values)
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        values.map(float_to_half)
    }
}

/// Narrows `src.len().min(dst.len())` `f32` values into `dst` matching GGML rounding.
#[inline]
pub fn narrow_f32_slice(src: &[f32], dst: &mut [u16]) {
    let count = src.len().min(dst.len());
    let (src_chunks, src_tail) = src[..count].as_chunks::<8>();
    let (dst_chunks, dst_tail) = dst[..count].as_chunks_mut::<8>();
    for (chunk, out_chunk) in src_chunks.iter().zip(dst_chunks.iter_mut()) {
        *out_chunk = narrow_f32_eight(chunk);
    }
    let (src_four, src_rem) = src_tail.as_chunks::<4>();
    let (dst_four, dst_rem) = dst_tail.as_chunks_mut::<4>();
    if let (Some(chunk), Some(out_chunk)) = (src_four.first(), dst_four.first_mut()) {
        *out_chunk = narrow_f32_four(*chunk);
    }
    for (&value, slot) in src_rem.iter().zip(dst_rem.iter_mut()) {
        *slot = float_to_half(value);
    }
}

/// Narrows finite `f32` values into IEEE binary16 bits (`dst.len() == src.len()`), rejecting
/// length mismatches, nonfinite inputs, and magnitudes that round to infinity (`|x| >= 65520.0`).
#[inline]
#[must_use]
pub fn narrow_f32_slice_checked(src: &[f32], dst: &mut [u16]) -> bool {
    if src.len() != dst.len() {
        return false;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        super::aarch64::narrow_f32_slice_checked(src, dst)
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        for (&value, slot) in src.iter().zip(dst.iter_mut()) {
            let Some(bits) = checked_float_to_half(value) else {
                return false;
            };
            *slot = bits;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::{
        checked_float_to_half, float_to_half, half_to_float, narrow_f32_eight, narrow_f32_four,
        narrow_f32_slice, narrow_f32_slice_checked, widen_f16_eight, widen_f16_four,
        widen_f16_slice,
    };

    // Independent normalized-significand construction, including subnormals.
    fn reference(bits: u16) -> u32 {
        let sign = u32::from(bits & 0x8000) << 16;
        let magnitude = u32::from(bits & 0x7fff);
        match magnitude {
            0 => sign,
            1..=0x03ff => {
                let leading = magnitude.leading_zeros() - 21;
                sign | ((113 - leading) << 23) | ((magnitude << (13 + leading)) & 0x007f_ffff)
            }
            0x0400..=0x7bff => sign | ((magnitude + 0x1_c000) << 13),
            _ => sign | 0x7f80_0000 | ((magnitude & 0x03ff) << 13),
        }
    }

    #[test]
    fn exhaustive_half_widening_preserves_every_encoding_in_each_lane() {
        for bits in 0..=u16::MAX {
            let four =
                std::array::from_fn(|lane: usize| bits.wrapping_add(lane.to_le_bytes()[0].into()));
            assert_eq!(widen_f16_four(&four).map(f32::to_bits), four.map(reference));
            let eight = std::array::from_fn(|lane: usize| {
                bits.wrapping_add(u16::from(lane.to_le_bytes()[0]) * 8191)
            });
            let actual = widen_f16_eight(&eight).map(|row| row.map(f32::to_bits));
            assert_eq!(
                actual,
                std::array::from_fn(|group| {
                    std::array::from_fn(|lane| reference(eight[group * 4 + lane]))
                })
            );
            if (bits & 0x7fff) < 0x7c00 {
                let four_vals = widen_f16_four(&four);
                assert_eq!(narrow_f32_four(four_vals), four_vals.map(float_to_half));
                let [low, high] = widen_f16_eight(&[
                    bits,
                    bits ^ 0x8000,
                    0x0000,
                    0x8000,
                    bits & 0x7bff,
                    0x7bff,
                    0xfbff,
                    0x0001,
                ]);
                let eight_vals = [
                    low[0], low[1], low[2], low[3], high[0], high[1], high[2], high[3],
                ];
                assert_eq!(narrow_f32_eight(&eight_vals), eight_vals.map(float_to_half));
            }
        }
    }

    #[test]
    fn narrowing_matches_ggml_rounding_at_boundaries() {
        let edges = [
            0.0_f32,
            -0.0_f32,
            f32::from_bits(0x3300_0000),
            f32::from_bits(0x3300_0001),
            f32::from_bits(0x3380_0000),
            f32::from_bits(0x3380_0001),
            f32::from_bits(0x387f_f000),
            f32::from_bits(0x387f_ffff),
            1.0 + 1.0 / 2048.0,
            1.0 + 3.0 / 2048.0,
            65_504.0,
            -65_504.0,
            65_519.0,
            65_520.0,
            -65_520.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            -f32::NAN,
            f32::from_bits(0x7f80_0001),
            f32::from_bits(0xffff_ffff),
        ];
        for start in 0..edges.len() {
            let four = std::array::from_fn(|idx| edges[(start + idx) % edges.len()]);
            assert_eq!(narrow_f32_four(four), four.map(float_to_half));
            let eight = std::array::from_fn(|idx| edges[(start + idx) % edges.len()]);
            assert_eq!(narrow_f32_eight(&eight), eight.map(float_to_half));
        }
    }

    #[test]
    fn slice_helpers_cover_all_remainders_and_checked_overflow() {
        let mut odd = [0_u16; 1];
        assert!(!narrow_f32_slice_checked(&[1.0, 2.0], &mut odd));
        for len in 0..=23_usize {
            let src: Vec<f32> = (0..len)
                .map(|i| half_to_float(u16::try_from((i * 199 + 1) & 0x7bff).unwrap_or(0)))
                .collect();
            let mut narrowed = vec![0_u16; len];
            narrow_f32_slice(&src, &mut narrowed);
            assert_eq!(
                narrowed,
                src.iter().copied().map(float_to_half).collect::<Vec<_>>()
            );

            let mut checked = vec![0_u16; len];
            assert!(narrow_f32_slice_checked(&src, &mut checked));
            assert_eq!(checked, narrowed);

            let mut widened = vec![0.0_f32; len];
            widen_f16_slice(&checked, &mut widened);
            assert_eq!(
                widened
                    .iter()
                    .copied()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
                src.iter().copied().map(f32::to_bits).collect::<Vec<_>>()
            );

            for bad in [65_520.0_f32, -65_520.0, f32::INFINITY, f32::NAN] {
                for bad_idx in 0..len {
                    let mut with_bad = src.clone();
                    with_bad[bad_idx] = bad;
                    assert_eq!(checked_float_to_half(bad), None);
                    assert!(!narrow_f32_slice_checked(&with_bad, &mut checked));
                }
            }
            if len > 0 {
                let mut edge = src;
                edge[len - 1] = f32::from_bits(0x477f_efff);
                assert!(narrow_f32_slice_checked(&edge, &mut checked));
                assert_eq!(checked[len - 1], 0x7bff);
            }
        }
        // Preserve signaling NaN payload bits across widen_f16_slice.
        let special = [
            0x7c01_u16, 0xfc01, 0x7e00, 0x0001, 0x8000, 0x3c00, 0xbc00, 0x7bff, 0x7c01,
        ];
        let mut widened = vec![0.0_f32; special.len()];
        widen_f16_slice(&special, &mut widened);
        assert_eq!(
            widened
                .iter()
                .copied()
                .map(f32::to_bits)
                .collect::<Vec<_>>(),
            special.map(reference)
        );
    }
}
