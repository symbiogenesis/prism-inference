//! Pinned nonlinear IQ4 codecs with packed integer activation dots.
//!
//! Adapted from ggml-quants.c and ggml-cpu/quants.c (MIT; upstream LICENSE).
//! The native `IQ4_NL` reference encoder reads uninitialized nibbles for blocks
//! below its 1e-15 cutoff. This safe implementation uses canonical zero nibbles
//! for that zero-scale payload; numerical zero semantics are preserved.

#![allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::suboptimal_flops,
    clippy::similar_names,
    clippy::suspicious_operation_groupings,
    reason = "Pinned nonlinear fitting uses explicit bit fields, exact-zero branches and source scalar arithmetic association."
)]

use super::{Q8Block, Q8KBlock, finite, half_bits, scale};
use crate::{Result, invalid};
use bonsai_ternary::half_to_float;

const VALUES: [i16; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

pub(super) const fn layout(kind: u32) -> Option<(usize, usize)> {
    match kind {
        20 => Some((32, 18)),
        23 => Some((256, 136)),
        _ => None,
    }
}

fn best_index(value: f32) -> u8 {
    if value <= f32::from(VALUES[0]) {
        return 0;
    }
    if value >= f32::from(VALUES[15]) {
        return 15;
    }
    let mut lower = 0;
    let mut upper = 15;
    while upper - lower > 1 {
        let middle = usize::midpoint(lower, upper);
        if value < f32::from(VALUES[middle]) {
            upper = middle;
        } else {
            lower = middle;
        }
    }
    if value - f32::from(VALUES[upper - 1]) < f32::from(VALUES[upper]) - value {
        (upper - 1) as u8
    } else {
        upper as u8
    }
}

fn nearest(value: f32) -> Result<i32> {
    if !value.is_finite() || value.abs() > 4_194_303.0 {
        return Err(invalid("IQ4 scale conversion exceeds native range"));
    }
    Ok(((value + 12_582_912.0).to_bits() & 0x007f_ffff).cast_signed() - 0x0040_0000)
}
fn half(value: f32) -> Result<u16> {
    finite(value)?;
    let bits = half_bits(value);
    finite(half_to_float(bits))?;
    Ok(bits)
}

fn subscale(bytes: &[u8], group: usize) -> i16 {
    let high = u16::from_le_bytes([bytes[2], bytes[3]]);
    i16::from((bytes[4 + group / 2] >> (4 * (group % 2))) & 15)
        + ((high >> (2 * group)) & 3).cast_signed() * 16
        - 32
}

pub(super) fn validate(kind: u32, bytes: &[u8]) -> Result<()> {
    let (_, size) = layout(kind).ok_or_else(|| invalid("unknown IQ4 type"))?;
    if !bytes.len().is_multiple_of(size) {
        return Err(invalid("partial IQ4 block"));
    }
    for block in bytes.chunks_exact(size) {
        scale(block)?;
    }
    Ok(())
}

pub(super) fn decode(kind: u32, bytes: &[u8], output: &mut Vec<f32>) -> Result<()> {
    validate(kind, bytes)?;
    let (count, size) = layout(kind).ok_or_else(|| invalid("unknown IQ4 type"))?;
    output.try_reserve_exact(bytes.len() / size * count)?;
    for block in bytes.chunks_exact(size) {
        let d = scale(block)?;
        let offset = if kind == 20 { 2 } else { 8 };
        for group in 0..count / 32 {
            let d = if kind == 20 {
                d
            } else {
                d * f32::from(subscale(block, group))
            };
            for index in 0..32 {
                let code = (block[offset + group * 16 + index % 16] >> (4 * (index / 16))) & 15;
                output.push(finite(d * f32::from(VALUES[usize::from(code)]))?);
            }
        }
    }
    Ok(())
}

pub(super) fn dot_q8(weights: &[u8], input: &[Q8Block]) -> Result<f32> {
    if input.len().checked_mul(18) != Some(weights.len()) {
        return Err(invalid("IQ4_NL dot extents differ"));
    }
    let mut sum = 0.0_f32;
    for (block, input) in weights.as_chunks::<18>().0.iter().zip(input) {
        let mut integer = 0_i32;
        for index in 0..32 {
            let code = (block[2 + index % 16] >> (4 * (index / 16))) & 15;
            integer += i32::from(input.values[index]) * i32::from(VALUES[usize::from(code)]);
        }
        sum += (finite(input.scale)? * scale(block)?) * integer as f32;
    }
    finite(sum)
}

pub(super) fn dot_q8k(weights: &[u8], input: &[Q8KBlock]) -> Result<f32> {
    if input.len().checked_mul(136) != Some(weights.len()) {
        return Err(invalid("IQ4_XS dot extents differ"));
    }
    let mut sum = 0.0_f32;
    for (block, input) in weights.as_chunks::<136>().0.iter().zip(input) {
        let d = scale(block)? * finite(input.scale)?;
        for group in 0..8 {
            let mut integer = 0_i32;
            for index in 0..32 {
                let code = (block[8 + group * 16 + index % 16] >> (4 * (index / 16))) & 15;
                integer += i32::from(input.values[group * 32 + index])
                    * i32::from(VALUES[usize::from(code)]);
            }
            sum += (d * f32::from(subscale(block, group))) * integer as f32;
        }
    }
    finite(sum)
}

#[allow(
    clippy::too_many_lines,
    reason = "Keep native local fit, search, super-block fit and bit packing in source order."
)]
pub(super) fn quantize(kind: u32, input: &[f32], importance: Option<&[f32]>) -> Result<Vec<u8>> {
    let (count, size) = layout(kind).ok_or_else(|| invalid("unknown IQ4 type"))?;
    if input.is_empty()
        || !input.len().is_multiple_of(count)
        || input.iter().any(|value| !value.is_finite())
        || importance.is_some_and(|weights| {
            weights.len() != input.len()
                || weights
                    .iter()
                    .any(|weight| !weight.is_finite() || *weight < 0.0)
        })
    {
        return Err(invalid(
            "IQ4 quantizer needs complete finite blocks and nonnegative importance",
        ));
    }
    let mut output = Vec::new();
    output.try_reserve_exact(input.len() / count * size)?;
    output.resize(input.len() / count * size, 0);
    let tries = if kind == 20 && importance.is_none() {
        -1_i32
    } else {
        7
    };
    for (block, (input, output)) in input
        .chunks_exact(count)
        .zip(output.chunks_exact_mut(size))
        .enumerate()
    {
        let mut sigma2 = 0.0_f32;
        for &value in input {
            sigma2 += value * value;
        }
        sigma2 *= 2.0 / count as f32;
        let importance = importance.map(|weights| &weights[block * count..(block + 1) * count]);
        let mut levels = vec![0_u8; count];
        let mut scales = [0.0_f32; 8];
        let mut maximum_scale = 0.0_f32;
        for (group, values) in input.as_chunks::<32>().0.iter().enumerate() {
            let weights = std::array::from_fn::<_, 32, _>(|index| {
                importance.map_or(values[index] * values[index], |weights| {
                    weights[group * 32 + index] * (sigma2 + values[index] * values[index]).sqrt()
                })
            });
            let mut maximum = 0.0_f32;
            for &value in values {
                if value.abs() > maximum.abs() {
                    maximum = value;
                }
            }
            if maximum.abs() < 1.0e-15 {
                continue;
            }
            let initial = if tries > 0 {
                -maximum / f32::from(VALUES[0])
            } else {
                maximum / f32::from(VALUES[0])
            };
            let inverse = 1.0 / initial;
            let mut sum_qx = 0.0_f32;
            let mut sum_q2 = 0.0_f32;
            for index in 0..32 {
                let level = best_index(inverse * values[index]);
                levels[group * 32 + index] = level;
                let q = f32::from(VALUES[usize::from(level)]);
                sum_qx += weights[index] * q * values[index];
                sum_q2 += weights[index] * q * q;
            }
            let mut d = if sum_q2 > 0.0 { sum_qx / sum_q2 } else { 0.0 };
            let mut best = d * sum_qx;
            for trial in -tries..=tries {
                let inverse = (trial as f32 + f32::from(VALUES[0])) / maximum;
                sum_qx = 0.0;
                sum_q2 = 0.0;
                for index in 0..32 {
                    let q = f32::from(VALUES[usize::from(best_index(inverse * values[index]))]);
                    sum_qx += weights[index] * q * values[index];
                    sum_q2 += weights[index] * q * q;
                }
                if sum_q2 > 0.0 && sum_qx * sum_qx > best * sum_q2 {
                    d = sum_qx / sum_q2;
                    best = d * sum_qx;
                }
            }
            scales[group] = finite(d)?;
            if d.abs() > maximum_scale.abs() {
                maximum_scale = d;
            }
        }
        if kind == 23 {
            let d = -maximum_scale / 32.0;
            output[..2].copy_from_slice(&half(d)?.to_le_bytes());
            let inverse = if d == 0.0 { 0.0 } else { 1.0 / d };
            let mut high = 0_u16;
            for group in 0..8 {
                let level = nearest(inverse * scales[group])?.clamp(-32, 31);
                let local = d * level as f32;
                let inverse = if local == 0.0 { 0.0 } else { 1.0 / local };
                for index in 0..32 {
                    levels[group * 32 + index] = best_index(inverse * input[group * 32 + index]);
                }
                let packed = (level + 32) as u8;
                output[4 + group / 2] |= (packed & 15) << (4 * (group % 2));
                high |= u16::from(packed >> 4) << (2 * group);
            }
            output[2..4].copy_from_slice(&high.to_le_bytes());
        } else {
            output[..2].copy_from_slice(&half(scales[0])?.to_le_bytes());
            if tries > 0 {
                let inverse = if scales[0] == 0.0 {
                    0.0
                } else {
                    1.0 / scales[0]
                };
                for (level, &value) in levels.iter_mut().zip(input) {
                    *level = best_index(inverse * value);
                }
            }
        }
        let offset = if kind == 20 { 2 } else { 8 };
        for (index, &level) in levels.iter().enumerate() {
            output[offset + index / 32 * 16 + index % 16] |= level << (4 * (index % 32 / 16));
        }
    }
    Ok(output)
}
