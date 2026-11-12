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
    use super::{widen_f16_eight, widen_f16_four};

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
        }
    }
}
