//! Exact binary16 widening with a baseline-NEON finite fast path.

#![expect(
    clippy::redundant_pub_crate,
    reason = "Parent-only fallback helpers remain inside the safe binary16 boundary."
)]

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
        bits.map(scalar)
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
    std::array::from_fn(|group| std::array::from_fn(|lane| scalar(bits[group * 4 + lane])))
}

/// Converts one `f32` value to its IEEE binary16 bit pattern using GGML ties-to-even rounding.
///
/// Preserves signed zero, subnormals, overflows to infinity, and maps NaNs to canonical quiet NaNs
/// with the original sign bit intact.
#[inline]
#[must_use]
pub fn scalar_half_bits(value: f32) -> u16 {
    let mut base = (value.abs() * f32::from_bits(0x7780_0000)) * f32::from_bits(0x0880_0000);
    let bits = value.to_bits();
    let doubled = bits.wrapping_add(bits);
    let exponent_bias = (doubled & 0xff00_0000).max(0x7100_0000);
    base += f32::from_bits((exponent_bias >> 1) + 0x0780_0000);
    let rounded = base.to_bits();
    let magnitude = if doubled > 0xff00_0000 {
        0x7e00
    } else {
        ((rounded >> 13) & 0x7c00) + (rounded & 0x0fff)
    };
    #[expect(
        clippy::cast_possible_truncation,
        clippy::as_conversions,
        reason = "The converted binary16 sign and magnitude fit in sixteen bits."
    )]
    {
        (((bits >> 16) & 0x8000) | magnitude) as u16
    }
}

/// Narrows one finite `f32` below the binary16 overflow threshold, returning `None` on overflow or NaN.
#[inline]
pub(super) fn scalar_narrow_f32_to_f16(value: f32) -> Option<u16> {
    if (value.to_bits() & 0x7fff_ffff) >= 0x477f_f000 {
        return None;
    }
    Some(scalar_half_bits(value))
}

/// Narrows four `f32` values to IEEE binary16 bit patterns matching GGML rounding.
///
/// Native `AArch64` NEON uses hardware `FCVTN` when all four lanes are non-NaN, falling back per
/// chunk when a NaN is present so canonical quiet NaN bits match [`scalar_half_bits`].
#[inline]
#[must_use]
pub fn narrow_f32_four(values: [f32; 4]) -> [u16; 4] {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        super::aarch64::narrow_f32_four(values)
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        values.map(scalar_half_bits)
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
        values.map(scalar_half_bits)
    }
}

#[inline]
pub(super) fn portable_narrow_slice(src: &[f32], dst: &mut [u16]) -> bool {
    for (&value, slot) in src.iter().zip(dst.iter_mut()) {
        let Some(bits) = scalar_narrow_f32_to_f16(value) else {
            return false;
        };
        *slot = bits;
    }
    true
}

/// Narrows finite `f32` elements into IEEE binary16 bits, rejecting nonfinite or overflowing lanes.
///
/// Native `AArch64` NEON processes 8-element chunks via `FCVTN`/`FCVTN2`. Returns `false` if
/// slice lengths differ or any element is `NaN`, infinite, or `|value| >= 65520.0`.
#[must_use]
pub fn narrow_f32_slice_to_f16(src: &[f32], dst: &mut [u16]) -> bool {
    if src.len() != dst.len() {
        return false;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        super::aarch64::narrow_f32_slice_to_f16(src, dst)
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        portable_narrow_slice(src, dst)
    }
}

#[inline]
pub(super) fn scalar(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let mut mantissa = u32::from(bits & 0x03ff);
    let mut exponent = u32::from((bits >> 10) & 0x1f);
    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        exponent = 113;
        while mantissa & 0x0400 == 0 {
            mantissa <<= 1;
            exponent -= 1;
        }
        mantissa &= 0x03ff;
    } else if exponent == 31 {
        exponent = 255;
    } else {
        exponent += 112;
    }
    f32::from_bits(sign | (exponent << 23) | (mantissa << 13))
}

#[cfg(test)]
mod tests {
    use super::{
        narrow_f32_eight, narrow_f32_four, narrow_f32_slice_to_f16, scalar_half_bits,
        widen_f16_eight, widen_f16_four,
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
                assert_eq!(narrow_f32_four(four_vals), four_vals.map(scalar_half_bits));
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
                assert_eq!(
                    narrow_f32_eight(&eight_vals),
                    eight_vals.map(scalar_half_bits)
                );
            }
        }
    }

    #[test]
    fn narrowing_matches_ggml_rounding_and_slice_checks_at_boundaries() {
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
            assert_eq!(narrow_f32_four(four), four.map(scalar_half_bits));
            let eight = std::array::from_fn(|idx| edges[(start + idx) % edges.len()]);
            assert_eq!(narrow_f32_eight(&eight), eight.map(scalar_half_bits));
        }

        let mut dst = vec![0_u16; 19];
        let src: Vec<f32> = (0..19)
            .map(|idx| f32::from(u8::try_from(idx).unwrap_or(0)) - 9.5)
            .collect();
        assert!(narrow_f32_slice_to_f16(&src, &mut dst));
        for (&s_val, &d_val) in src.iter().zip(&dst) {
            assert_eq!(d_val, scalar_half_bits(s_val));
        }
        assert!(!narrow_f32_slice_to_f16(&src, &mut dst[..18]));
        for bad_val in [
            65_520.0_f32,
            -65_520.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ] {
            let mut bad_src = src.clone();
            bad_src[11] = bad_val;
            assert!(!narrow_f32_slice_to_f16(&bad_src, &mut dst));
            assert!(!narrow_f32_slice_to_f16(&[bad_val], &mut [0_u16; 1]));
        }
    }
}
