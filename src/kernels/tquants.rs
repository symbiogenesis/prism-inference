//! Native 256-element ternary codecs and packed `Q8_K` dot products.
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

fn codes(kind: u32, block: &[u8]) -> [i16; 256] {
    std::array::from_fn(|index| {
        if kind == 35 {
            i16::from((block[index / 128 * 32 + index % 32] >> (2 * (index % 128 / 32))) & 3) - 1
        } else {
            let (byte, exponent) = if index < 160 {
                (index % 32, index / 32)
            } else if index < 240 {
                (32 + (index - 160) % 16, (index - 160) / 16)
            } else {
                (48 + (index - 240) % 4, (index - 240) / 4)
            };
            let power = [1_u8, 3, 9, 27, 81][exponent];
            i16::from(block[byte].wrapping_mul(power)) * 3 / 256 - 1
        }
    })
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
        let integer: i32 = codes(kind, block)
            .iter()
            .zip(input.values)
            .map(|(&weight, value)| i32::from(weight) * i32::from(value))
            .sum();
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
