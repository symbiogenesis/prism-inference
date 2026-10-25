//! Safe K-block codecs and native `Q8_K` integer dot products.
//!
//! Ported from the pinned ggml-quants.c and ggml-cpu/quants.c generic kernels.
//! Upstream MIT attribution: tools/prism/upstream/LICENSE. Packed matrices remain
//! packed; only one 256-element integer block is unpacked at a time.

#![allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::suboptimal_flops,
    reason = "Pinned quantizer bit fields and scalar floating association are explicit ABI operations."
)]

use super::{Q8KBlock, finite, scale};
use crate::{Result, invalid};
mod encode;

pub(super) fn quantize(kind: u32, input: &[f32]) -> Result<Vec<u8>> {
    encode::quantize(kind, input, None)
}

pub(super) fn quantize_weighted(kind: u32, input: &[f32], importance: &[f32]) -> Result<Vec<u8>> {
    encode::quantize(kind, input, Some(importance))
}

pub(super) const fn block_bytes(kind: u32) -> Option<usize> {
    match kind {
        10 => Some(84),
        11 => Some(110),
        12 => Some(144),
        13 => Some(176),
        14 => Some(210),
        _ => None,
    }
}

fn scale_min(group: usize, packed: &[u8]) -> (i16, i16) {
    if group < 4 {
        (
            i16::from(packed[group] & 63),
            i16::from(packed[group + 4] & 63),
        )
    } else {
        (
            i16::from((packed[group + 4] & 15) | ((packed[group - 4] >> 6) << 4)),
            i16::from((packed[group + 4] >> 4) | ((packed[group] >> 6) << 4)),
        )
    }
}

struct Block {
    values: [i16; 256],
    scales: [i16; 16],
    minima: [i16; 16],
    d: f32,
    minimum: f32,
    group: usize,
}

fn unpack(kind: u32, bytes: &[u8]) -> Result<Block> {
    if block_bytes(kind) != Some(bytes.len()) {
        return Err(invalid("K quant block extent differs"));
    }
    let mut block = Block {
        values: [0; 256],
        scales: [0; 16],
        minima: [0; 16],
        d: 0.0,
        minimum: 0.0,
        group: 16,
    };
    match kind {
        10 => {
            block.d = scale(&bytes[80..82])?;
            block.minimum = scale(&bytes[82..84])?;
            for group in 0..16 {
                block.scales[group] = i16::from(bytes[group] & 15);
                block.minima[group] = i16::from(bytes[group] >> 4);
            }
            for (index, value) in block.values.iter_mut().enumerate() {
                *value = i16::from(
                    (bytes[16 + index / 128 * 32 + index % 32] >> (2 * (index % 128 / 32))) & 3,
                );
            }
        }
        11 => {
            block.d = scale(&bytes[108..110])?;
            for group in 0..16 {
                block.scales[group] = i16::from(
                    ((bytes[96 + group % 8] >> (4 * (group / 8))) & 15)
                        | (((bytes[104 + group % 4] >> (2 * (group / 4))) & 3) << 4),
                ) - 32;
            }
            for (index, value) in block.values.iter_mut().enumerate() {
                *value = i16::from(
                    (bytes[32 + index / 128 * 32 + index % 32] >> (2 * (index % 128 / 32))) & 3,
                ) - if bytes[index % 32] & (1 << (index / 32)) == 0 {
                    4
                } else {
                    0
                };
            }
        }
        12 | 13 => {
            block.group = 32;
            block.d = scale(bytes)?;
            block.minimum = scale(&bytes[2..4])?;
            for group in 0..8 {
                (block.scales[group], block.minima[group]) = scale_min(group, &bytes[4..16]);
            }
            let offset = if kind == 12 { 16 } else { 48 };
            for (index, value) in block.values.iter_mut().enumerate() {
                *value = i16::from(
                    (bytes[offset + index / 64 * 32 + index % 32] >> (4 * (index % 64 / 32))) & 15,
                );
                if kind == 13 && bytes[16 + index % 32] & (1 << (index / 32)) != 0 {
                    *value += 16;
                }
            }
        }
        14 => {
            block.d = scale(&bytes[208..210])?;
            for group in 0..16 {
                block.scales[group] = i16::from(i8::from_ne_bytes([bytes[192 + group]]));
            }
            for (index, value) in block.values.iter_mut().enumerate() {
                let part = index % 128 / 32;
                let low =
                    (bytes[index / 128 * 64 + part % 2 * 32 + index % 32] >> (4 * (part / 2))) & 15;
                let high = (bytes[128 + index / 128 * 32 + index % 32] >> (2 * part)) & 3;
                *value = i16::from(low | (high << 4)) - 32;
            }
        }
        _ => return Err(invalid("unknown K quant kind")),
    }
    Ok(block)
}

pub(super) fn validate(kind: u32, bytes: &[u8]) -> Result<()> {
    unpack(kind, bytes)?;
    Ok(())
}

pub(super) fn decode(kind: u32, bytes: &[u8], output: &mut Vec<f32>) -> Result<()> {
    let size = block_bytes(kind).ok_or_else(|| invalid("unknown K quant kind"))?;
    if !bytes.len().is_multiple_of(size) {
        return Err(invalid("K quant row has partial block"));
    }
    output.try_reserve_exact(bytes.len() / size * 256)?;
    for bytes in bytes.chunks_exact(size) {
        let block = unpack(kind, bytes)?;
        for index in 0..256 {
            let group = index / block.group;
            output.push(finite(
                (block.d * f32::from(block.scales[group])) * f32::from(block.values[index])
                    - block.minimum * f32::from(block.minima[group]),
            )?);
        }
    }
    Ok(())
}

pub(super) fn dot(kind: u32, weights: &[u8], input: &[Q8KBlock]) -> Result<f32> {
    let size = block_bytes(kind).ok_or_else(|| invalid("unknown K quant kind"))?;
    if input.len().checked_mul(size) != Some(weights.len()) {
        return Err(invalid("K quant dot block counts differ"));
    }
    let mut sum = 0.0_f32;
    let mut lanes = [0.0_f32; 8];
    for (weights, input) in weights.chunks_exact(size).zip(input) {
        let block = unpack(kind, weights)?;
        let activation_scale = finite(input.scale)?;
        let mut integers = [0_i32; 8];
        let mut minimum_sum = 0_i32;
        for index in 0..256 {
            integers[index % 8] += i32::from(block.scales[index / block.group])
                * i32::from(block.values[index])
                * i32::from(input.values[index]);
        }
        for group in 0..16 {
            minimum_sum +=
                i32::from(input.sums[group]) * i32::from(block.minima[group * 16 / block.group]);
        }
        if kind == 10 {
            let integer_sum: i32 = integers.into_iter().sum();
            sum += (activation_scale * block.d) * integer_sum as f32
                - (activation_scale * block.minimum) * minimum_sum as f32;
        } else {
            let d = block.d * activation_scale;
            for (lane, integer) in lanes.iter_mut().zip(integers) {
                *lane += d * integer as f32;
            }
            if matches!(kind, 12 | 13) {
                sum -= (block.minimum * activation_scale) * minimum_sum as f32;
            }
        }
    }
    if kind != 10 {
        for lane in lanes {
            sum += lane;
        }
    }
    finite(sum)
}
