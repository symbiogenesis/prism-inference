//! Pinned microscaled and NVIDIA FP4 reference codecs and packed `Q8_0` dots.
//! Ported from ggml-quants.c, ggml-impl.h and ggml-cpu/quants.c (upstream MIT).

#![allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::suboptimal_flops,
    reason = "Pinned FP4 bit encodings and native multiply/add association are explicit format arithmetic."
)]

use super::{Q8Block, finite};
use crate::{Result, invalid};

const VALUES: [i16; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

pub(super) const fn layout(kind: u32) -> Option<(usize, usize)> {
    match kind {
        39 => Some((32, 17)),
        40 => Some((64, 36)),
        _ => None,
    }
}

fn exponent_half(value: u8) -> f32 {
    f32::from_bits(if value < 2 {
        0x0020_0000 << value
    } else {
        (u32::from(value) - 1) << 23
    })
}
fn ue4m3_half(value: u8) -> f32 {
    if matches!(value, 0 | 0x7f) {
        return 0.0;
    }
    let exponent = i32::from((value >> 3) & 15);
    let mantissa = f32::from(value & 7);
    if exponent == 0 {
        mantissa * 2.0_f32.powi(-9) * 0.5
    } else {
        (1.0 + mantissa / 8.0) * 2.0_f32.powi(exponent - 7) * 0.5
    }
}

fn ue4m3(mut value: f32) -> u8 {
    if value <= 0.0 {
        return 0;
    }
    if value > 448.0 {
        value = 448.0;
    }
    let bits = value.to_bits();
    let exponent = ((bits >> 23) & 255).cast_signed() - 127;
    let mantissa = (bits >> 20) & 7;
    let mut encoded_exponent = exponent + 7;
    if encoded_exponent <= 0 {
        let mantissa = (value * 512.0 + 0.5) as i32;
        return if mantissa < 1 {
            0
        } else {
            mantissa.min(7) as u8
        };
    }
    if encoded_exponent >= 15 {
        return 0x7e;
    }
    let mut encoded_mantissa = mantissa + ((bits >> 19) & 1);
    if encoded_mantissa > 7 {
        encoded_mantissa = 0;
        encoded_exponent += 1;
        if encoded_exponent >= 15 {
            return 0x7e;
        }
    }
    ((encoded_exponent as u32) << 3 | encoded_mantissa) as u8
}

fn best_index(value: f32, scale: f32) -> u8 {
    let mut best = 0;
    let mut error = (f32::from(VALUES[0]) * scale - value).abs();
    for (index, &code) in VALUES.iter().enumerate().skip(1) {
        let candidate = (f32::from(code) * scale - value).abs();
        if candidate < error {
            best = index as u8;
            error = candidate;
        }
    }
    best
}

pub(super) fn validate(kind: u32, bytes: &[u8]) -> Result<()> {
    let (_, size) = layout(kind).ok_or_else(|| invalid("unknown FP4 type"))?;
    if !bytes.len().is_multiple_of(size) {
        return Err(invalid("partial FP4 block"));
    }
    Ok(())
}

pub(super) fn decode(kind: u32, bytes: &[u8], output: &mut Vec<f32>) -> Result<()> {
    validate(kind, bytes)?;
    let (count, size) = layout(kind).ok_or_else(|| invalid("unknown FP4 type"))?;
    output.try_reserve_exact(bytes.len() / size * count)?;
    let group_size = if kind == 39 { 32 } else { 16 };
    let offset = if kind == 39 { 1 } else { 4 };
    for block in bytes.chunks_exact(size) {
        for group in 0..count / group_size {
            let d = if kind == 39 {
                exponent_half(block[0])
            } else {
                ue4m3_half(block[group])
            };
            for index in 0..group_size {
                let code = (block[offset + group * group_size / 2 + index % (group_size / 2)]
                    >> (4 * (index / (group_size / 2))))
                    & 15;
                output.push(finite(f32::from(VALUES[usize::from(code)]) * d)?);
            }
        }
    }
    Ok(())
}

pub(super) fn dot(kind: u32, weights: &[u8], input: &[Q8Block]) -> Result<f32> {
    let (count, size) = layout(kind).ok_or_else(|| invalid("unknown FP4 type"))?;
    if weights.len() / size * (count / 32) != input.len() || !weights.len().is_multiple_of(size) {
        return Err(invalid("FP4 packed dot extents differ"));
    }
    let mut sum = 0.0_f32;
    let group_size = if kind == 39 { 32 } else { 16 };
    let offset = if kind == 39 { 1 } else { 4 };
    for (block_index, block) in weights.chunks_exact(size).enumerate() {
        for group in 0..count / group_size {
            let input = &input[block_index * (count / 32) + group * group_size / 32];
            let input_offset = group * group_size % 32;
            let d = if kind == 39 {
                exponent_half(block[0])
            } else {
                ue4m3_half(block[group])
            };
            let mut integer = 0_i32;
            for index in 0..group_size {
                let code = (block[offset + group * group_size / 2 + index % (group_size / 2)]
                    >> (4 * (index / (group_size / 2))))
                    & 15;
                integer += i32::from(input.values[input_offset + index])
                    * i32::from(VALUES[usize::from(code)]);
            }
            sum += (finite(input.scale)? * d) * integer as f32;
        }
    }
    finite(sum)
}

pub(super) fn quantize(kind: u32, input: &[f32]) -> Result<Vec<u8>> {
    let (count, size) = layout(kind).ok_or_else(|| invalid("unknown FP4 type"))?;
    if input.is_empty()
        || !input.len().is_multiple_of(count)
        || input.iter().any(|value| !value.is_finite())
    {
        return Err(invalid("FP4 quantizer needs complete finite blocks"));
    }
    let mut output = Vec::new();
    output.try_reserve_exact(input.len() / count * size)?;
    output.resize(input.len() / count * size, 0);
    let group_size = if kind == 39 { 32 } else { 16 };
    let offset = if kind == 39 { 1 } else { 4 };
    for (input, output) in input.chunks_exact(count).zip(output.chunks_exact_mut(size)) {
        for (group, values) in input.chunks_exact(group_size).enumerate() {
            let maximum = values
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f32, f32::max);
            let d = if kind == 39 {
                let exponent = if maximum > 0.0 {
                    (maximum.log2().floor() as i32 - 2 + 127) as u8
                } else {
                    0
                };
                output[0] = exponent;
                exponent_half(exponent)
            } else {
                let encoded = ue4m3(maximum / 6.0);
                output[group] = encoded;
                ue4m3_half(encoded)
            };
            for (index, &value) in values.iter().enumerate() {
                output[offset + group * group_size / 2 + index % (group_size / 2)] |=
                    best_index(value, d) << (4 * (index / (group_size / 2)));
            }
        }
    }
    Ok(output)
}
