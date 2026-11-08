//! Native 256-element ternary codecs, `TL1` nibble lookup tables and packed `Q8_K` dot products.
//! Ported from pinned ggml-quants.c and ggml-cpu/quants.c; upstream MIT LICENSE.

#![allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::suboptimal_flops,
    reason = "Pinned ternary encoding uses bounded integer casts and exact-zero scale branches."
)]

use super::{Q8KBlock, finite, half_bits, scale};
use crate::{Result, invalid};
use bonsai_ternary::half_to_float;

pub(super) const fn block_bytes(kind: u32) -> Option<usize> {
    match kind {
        34 => Some(54),
        35 => Some(66),
        _ => None,
    }
}

/// Compile-time lookup table mapping each `TQ1_0` base-3 packed byte (`0..=255`)
/// to its five decoded ternary weights in `{-1, 0, +1}`.
const TQ1_BYTE_LUT: [[i8; 5]; 256] = {
    let mut lut = [[0_i8; 5]; 256];
    let powers = [1_u8, 3, 9, 27, 81];
    let mut byte = 0_usize;
    while byte < 256 {
        let b = byte as u8;
        let mut exp = 0_usize;
        while exp < 5 {
            lut[byte][exp] = ((b.wrapping_mul(powers[exp]) as i16) * 3 / 256 - 1) as i8;
            exp += 1;
        }
        byte += 1;
    }
    lut
};

/// Compile-time lookup table mapping each `TQ2_0` 2-bit packed byte (`0..=255`)
/// to its four decoded signed digits `(code & 3) - 1`.
const TQ2_BYTE_LUT: [[i8; 4]; 256] = {
    let mut lut = [[0_i8; 4]; 256];
    let mut byte = 0_usize;
    while byte < 256 {
        let b = byte as u8;
        lut[byte][0] = ((b & 3) as i8) - 1;
        lut[byte][1] = (((b >> 2) & 3) as i8) - 1;
        lut[byte][2] = (((b >> 4) & 3) as i8) - 1;
        lut[byte][3] = (((b >> 6) & 3) as i8) - 1;
        byte += 1;
    }
    lut
};

/// `TL1`-style 16-entry INT16 activation-pair lookup table for 128-element
/// 2-bit ternary / `PQ2` weight blocks.
///
/// Precomputes `(c0 - 1) * a0 + (c1 - 1) * a1` for every 4-bit nibble
/// `c0 | (c1 << 2)` (`0..16`), replacing 128 bit-extractions and multiplications
/// per block with 64 table lookups (2 nibble lookups per packed weight byte).
pub struct Tl1BlockLut {
    low_nibbles: [[i16; 16]; 32],
    high_nibbles: [[i16; 16]; 32],
}

impl Tl1BlockLut {
    /// Build a `TL1` nibble lookup table for a 128-element `Q8` activation slice
    /// indexed in contiguous 4-weights-per-byte order (`PQ2` / `PTQ1` unpacked order).
    #[must_use]
    pub fn from_contiguous_activations(activations: &[i8; 128]) -> Self {
        let mut low_nibbles = [[0_i16; 16]; 32];
        let mut high_nibbles = [[0_i16; 16]; 32];
        for (byte_idx, chunk) in activations.as_chunks::<4>().0.iter().enumerate() {
            let a0 = i16::from(chunk[0]);
            let a1 = i16::from(chunk[1]);
            let a2 = i16::from(chunk[2]);
            let a3 = i16::from(chunk[3]);
            for nibble in 0_u8..16 {
                let w0 = i16::from(nibble & 3) - 1;
                let w1 = i16::from((nibble >> 2) & 3) - 1;
                let idx = usize::from(nibble);
                low_nibbles[byte_idx][idx] = w0 * a0 + w1 * a1;
                high_nibbles[byte_idx][idx] = w0 * a2 + w1 * a3;
            }
        }
        Self {
            low_nibbles,
            high_nibbles,
        }
    }

    /// Build a `TL1` nibble lookup table for a 128-element `Q8` activation slice
    /// indexed in 32-stride plane order (`TQ2_0` half-block order).
    #[must_use]
    pub fn from_strided32_activations(activations: &[i8; 128]) -> Self {
        let mut low_nibbles = [[0_i16; 16]; 32];
        let mut high_nibbles = [[0_i16; 16]; 32];
        for byte_idx in 0..32 {
            let a0 = i16::from(activations[byte_idx]);
            let a1 = i16::from(activations[byte_idx + 32]);
            let a2 = i16::from(activations[byte_idx + 64]);
            let a3 = i16::from(activations[byte_idx + 96]);
            for nibble in 0_u8..16 {
                let w0 = i16::from(nibble & 3) - 1;
                let w1 = i16::from((nibble >> 2) & 3) - 1;
                let idx = usize::from(nibble);
                low_nibbles[byte_idx][idx] = w0 * a0 + w1 * a1;
                high_nibbles[byte_idx][idx] = w0 * a2 + w1 * a3;
            }
        }
        Self {
            low_nibbles,
            high_nibbles,
        }
    }

    /// Compute the exact 128-element integer dot product against a 32-byte packed
    /// 2-bit weight block using 64 `TL1` nibble lookups.
    #[inline]
    #[must_use]
    pub fn dot(&self, packed: &[u8; 32]) -> i32 {
        let mut sum = 0_i32;
        for (byte_idx, &byte) in packed.iter().enumerate() {
            let lo = usize::from(byte & 0x0f);
            let hi = usize::from(byte >> 4);
            sum += i32::from(self.low_nibbles[byte_idx][lo])
                + i32::from(self.high_nibbles[byte_idx][hi]);
        }
        sum
    }
}

fn codes(kind: u32, block: &[u8]) -> [i16; 256] {
    let mut out = [0_i16; 256];
    if kind == 35 {
        for half in 0..2 {
            let byte_base = half * 32;
            let elem_base = half * 128;
            for col in 0..32 {
                let digits = TQ2_BYTE_LUT[usize::from(block[byte_base + col])];
                out[elem_base + col] = i16::from(digits[0]);
                out[elem_base + col + 32] = i16::from(digits[1]);
                out[elem_base + col + 64] = i16::from(digits[2]);
                out[elem_base + col + 96] = i16::from(digits[3]);
            }
        }
    } else {
        for col in 0..32 {
            let digits = TQ1_BYTE_LUT[usize::from(block[col])];
            out[col] = i16::from(digits[0]);
            out[col + 32] = i16::from(digits[1]);
            out[col + 64] = i16::from(digits[2]);
            out[col + 96] = i16::from(digits[3]);
            out[col + 128] = i16::from(digits[4]);
        }
        for col in 0..16 {
            let digits = TQ1_BYTE_LUT[usize::from(block[32 + col])];
            out[160 + col] = i16::from(digits[0]);
            out[176 + col] = i16::from(digits[1]);
            out[192 + col] = i16::from(digits[2]);
            out[208 + col] = i16::from(digits[3]);
            out[224 + col] = i16::from(digits[4]);
        }
        for col in 0..4 {
            let digits = TQ1_BYTE_LUT[usize::from(block[48 + col])];
            out[240 + col] = i16::from(digits[0]);
            out[244 + col] = i16::from(digits[1]);
            out[248 + col] = i16::from(digits[2]);
            out[252 + col] = i16::from(digits[3]);
        }
    }
    out
}

#[inline]
fn dot_tq1_block(block: &[u8], values: &[i8; 256]) -> i32 {
    let mut sum = 0_i32;
    for col in 0..32 {
        let digits = TQ1_BYTE_LUT[usize::from(block[col])];
        sum += i32::from(digits[0]) * i32::from(values[col])
            + i32::from(digits[1]) * i32::from(values[col + 32])
            + i32::from(digits[2]) * i32::from(values[col + 64])
            + i32::from(digits[3]) * i32::from(values[col + 96])
            + i32::from(digits[4]) * i32::from(values[col + 128]);
    }
    for col in 0..16 {
        let digits = TQ1_BYTE_LUT[usize::from(block[32 + col])];
        sum += i32::from(digits[0]) * i32::from(values[160 + col])
            + i32::from(digits[1]) * i32::from(values[176 + col])
            + i32::from(digits[2]) * i32::from(values[192 + col])
            + i32::from(digits[3]) * i32::from(values[208 + col])
            + i32::from(digits[4]) * i32::from(values[224 + col]);
    }
    for col in 0..4 {
        let digits = TQ1_BYTE_LUT[usize::from(block[48 + col])];
        sum += i32::from(digits[0]) * i32::from(values[240 + col])
            + i32::from(digits[1]) * i32::from(values[244 + col])
            + i32::from(digits[2]) * i32::from(values[248 + col])
            + i32::from(digits[3]) * i32::from(values[252 + col]);
    }
    sum
}

#[inline]
fn dot_tq2_block(block: &[u8], values: &[i8; 256]) -> i32 {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Ok(packed) = <&[u8; 64]>::try_from(&block[..64]) {
        return super::neon::dot_tq2_block(packed, values);
    }
    let mut sum = 0_i32;
    for half in 0..2 {
        let byte_base = half * 32;
        let elem_base = half * 128;
        for col in 0..32 {
            let digits = TQ2_BYTE_LUT[usize::from(block[byte_base + col])];
            sum += i32::from(digits[0]) * i32::from(values[elem_base + col])
                + i32::from(digits[1]) * i32::from(values[elem_base + col + 32])
                + i32::from(digits[2]) * i32::from(values[elem_base + col + 64])
                + i32::from(digits[3]) * i32::from(values[elem_base + col + 96]);
        }
    }
    sum
}

pub(super) fn validate(kind: u32, bytes: &[u8]) -> Result<()> {
    let size = block_bytes(kind).ok_or_else(|| invalid("unknown TQ type"))?;
    if !bytes.len().is_multiple_of(size) {
        return Err(invalid("partial TQ block"));
    }
    for block in bytes.chunks_exact(size) {
        scale(&block[size - 2..])?;
    }
    Ok(())
}

pub(super) fn decode(kind: u32, bytes: &[u8], output: &mut Vec<f32>) -> Result<()> {
    validate(kind, bytes)?;
    let size = block_bytes(kind).ok_or_else(|| invalid("unknown TQ type"))?;
    output.try_reserve_exact(bytes.len() / size * 256)?;
    for block in bytes.chunks_exact(size) {
        let d = scale(&block[size - 2..])?;
        for code in codes(kind, block) {
            output.push(finite(f32::from(code) * d)?);
        }
    }
    Ok(())
}

pub(super) fn dot(kind: u32, weights: &[u8], input: &[Q8KBlock]) -> Result<f32> {
    let size = block_bytes(kind).ok_or_else(|| invalid("unknown TQ type"))?;
    if input.len().checked_mul(size) != Some(weights.len()) {
        return Err(invalid("TQ packed dot extents differ"));
    }
    let mut sum = 0.0_f32;
    for (block, input) in weights.chunks_exact(size).zip(input) {
        let integer = if kind == 35 {
            dot_tq2_block(block, &input.values)
        } else {
            dot_tq1_block(block, &input.values)
        };
        let d = scale(&block[size - 2..])?;
        sum += integer as f32 * (d * finite(input.scale)?);
    }
    finite(sum)
}

pub(super) fn quantize(kind: u32, input: &[f32]) -> Result<Vec<u8>> {
    let size = block_bytes(kind).ok_or_else(|| invalid("unknown TQ type"))?;
    if input.is_empty()
        || !input.len().is_multiple_of(256)
        || input.iter().any(|value| !value.is_finite())
    {
        return Err(invalid("TQ quantizer needs complete finite blocks"));
    }
    let mut output = Vec::new();
    output.try_reserve_exact(input.len() / 256 * size)?;
    output.resize(input.len() / 256 * size, 0);
    for (input, output) in input
        .as_chunks::<256>()
        .0
        .iter()
        .zip(output.chunks_exact_mut(size))
    {
        let d = input
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f32, f32::max);
        let bits = half_bits(d);
        finite(half_to_float(bits))?;
        output[size - 2..].copy_from_slice(&bits.to_le_bytes());
        let inverse = if d == 0.0 { 0.0 } else { finite(1.0 / d)? };
        let levels = std::array::from_fn::<_, 256, _>(|index| {
            ((input[index] * inverse).round() as i16 + 1) as u8
        });
        if kind == 35 {
            for (index, &level) in levels.iter().enumerate() {
                output[index / 128 * 32 + index % 32] |= level << (2 * (index % 128 / 32));
            }
        } else {
            for (byte, destination) in output[..52].iter_mut().enumerate() {
                let (first, stride, count) = if byte < 32 {
                    (byte, 32, 5)
                } else if byte < 48 {
                    (160 + byte - 32, 16, 5)
                } else {
                    (240 + byte - 48, 4, 4)
                };
                let mut code = 0_u16;
                for digit in 0..count {
                    code = code * 3 + u16::from(levels[first + digit * stride]);
                }
                if count == 4 {
                    code *= 3;
                }
                *destination = (code * 256).div_ceil(243) as u8;
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tl1_block_lut_matches_contiguous_and_strided32_ternary_dots() {
        let mut packed = [0_u8; 32];
        for (idx, b) in packed.iter_mut().enumerate() {
            *b = ((idx * 73 + 19) & 0xff) as u8;
        }
        let mut activations = [0_i8; 128];
        for (idx, a) in activations.iter_mut().enumerate() {
            *a = (((idx * 37 + 11) % 251) as i16 - 125) as i8;
        }
        let contiguous_lut = Tl1BlockLut::from_contiguous_activations(&activations);
        let expected_contiguous: i32 = activations
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                let weight = i32::from((packed[index / 4] >> (2 * (index % 4))) & 3) - 1;
                weight * i32::from(value)
            })
            .sum();
        assert_eq!(contiguous_lut.dot(&packed), expected_contiguous);

        let strided_lut = Tl1BlockLut::from_strided32_activations(&activations);
        let expected_strided: i32 = (0..128)
            .map(|index| {
                let weight = i32::from((packed[index % 32] >> (2 * (index / 32))) & 3) - 1;
                weight * i32::from(activations[index])
            })
            .sum();
        assert_eq!(strided_lut.dot(&packed), expected_strided);
    }
}
