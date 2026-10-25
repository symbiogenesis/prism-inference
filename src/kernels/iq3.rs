//! Safe Prism IQ3 packed decoding and native scalar `Q8_K` activation dots.
//!
//! Ported from PrismML-Eng/llama.cpp `73bfe6cc485554822f80c473a3c0c4f25e80f41b`,
//! ggml/src/ggml-quants.c and ggml/src/ggml-cpu/quants.c (MIT; root NOTICE).
//! The integer codebooks retain the original ggml-common.h values. Reference
//! and importance encoders preserve the pinned native grid-fitting contracts.

#![allow(
    clippy::cast_precision_loss,
    clippy::as_conversions,
    clippy::suboptimal_flops,
    reason = "Native packed-dot integer bounds fit i32 and preserve scalar floating association."
)]

use super::{Q8KBlock, finite, scale};
use crate::{Result, invalid};
#[path = "iq3_tables.rs"]
mod tables;

pub(super) fn quantize(kind: u32, input: &[f32], importance: Option<&[f32]>) -> Result<Vec<u8>> {
    match kind {
        18 => {
            // The decoder's optimized magnitudes identify the same native
            // odd-level grid by floor(magnitude/8), including its top value 62.
            let points =
                tables::XXS.map(|word| word.to_le_bytes().map(|value| 2 * (value / 8) + 1));
            super::iqfit::quantize(kind, input, importance, &points)
        }
        21 => {
            let points = tables::SMALL.map(u32::to_le_bytes);
            super::iqfit::quantize(kind, input, importance, &points)
        }
        _ => Err(invalid("unknown IQ3 quantizer")),
    }
}

pub(super) const fn layout(kind: u32) -> Option<(usize, usize)> {
    match kind {
        18 => Some((256, 98)),
        21 => Some((256, 110)),
        _ => None,
    }
}

pub(super) fn validate(kind: u32, bytes: &[u8]) -> Result<()> {
    let (_, size) = layout(kind).ok_or_else(|| invalid("unknown IQ3 type"))?;
    if !bytes.len().is_multiple_of(size) {
        return Err(invalid("partial IQ3 block"));
    }
    for block in bytes.chunks_exact(size) {
        scale(block)?;
    }
    Ok(())
}

fn unpack(kind: u32, bytes: &[u8]) -> ([i16; 256], [i16; 8]) {
    let mut values = [0_i16; 256];
    let mut scales = [0_i16; 8];
    for (group, subscale) in scales.iter_mut().enumerate() {
        let auxiliary = if kind == 18 {
            u32::from_le_bytes(std::array::from_fn(|index| bytes[66 + group * 4 + index]))
        } else {
            0
        };
        *subscale = if kind == 18 {
            2 * i16::from(auxiliary.to_le_bytes()[3] >> 4) + 1
        } else {
            2 * i16::from((bytes[106 + group / 2] >> (4 * (group % 2))) & 15) + 1
        };
        for octet in 0..4 {
            let signs = if kind == 18 {
                let low = (auxiliary >> (7 * octet)).to_le_bytes()[0] & 127;
                low | (u8::from(low.count_ones() % 2 != 0) << 7)
            } else {
                bytes[74 + group * 4 + octet]
            };
            for half in 0..2 {
                let low = usize::from(bytes[2 + group * 8 + octet * 2 + half]);
                let grid = if kind == 18 {
                    tables::XXS[low]
                } else {
                    let high = usize::from((bytes[66 + group] >> (octet * 2 + half)) & 1);
                    tables::SMALL[low | high << 8]
                };
                for (index, magnitude) in grid.to_le_bytes().into_iter().enumerate() {
                    let sign = if signs & (1 << (half * 4 + index)) == 0 {
                        1
                    } else {
                        -1
                    };
                    values[group * 32 + octet * 8 + half * 4 + index] = i16::from(magnitude) * sign;
                }
            }
        }
    }
    (values, scales)
}

pub(super) fn decode(kind: u32, bytes: &[u8], output: &mut Vec<f32>) -> Result<()> {
    validate(kind, bytes)?;
    let (_, size) = layout(kind).ok_or_else(|| invalid("unknown IQ3 type"))?;
    let count = (bytes.len() / size)
        .checked_mul(256)
        .ok_or_else(|| invalid("IQ3 decoded extent overflow"))?;
    output.try_reserve_exact(count)?;
    for block in bytes.chunks_exact(size) {
        let d = scale(block)?;
        let (values, scales) = unpack(kind, block);
        for (group, values) in values.as_chunks::<32>().0.iter().enumerate() {
            let d = if kind == 18 {
                d * (0.5 + f32::from((scales[group] - 1) / 2)) * 0.5
            } else {
                d * f32::from(scales[group])
            };
            for &value in values {
                output.push(finite(d * f32::from(value))?);
            }
        }
    }
    Ok(())
}

pub(super) fn dot(kind: u32, bytes: &[u8], input: &[Q8KBlock]) -> Result<f32> {
    validate(kind, bytes)?;
    let (_, size) = layout(kind).ok_or_else(|| invalid("unknown IQ3 type"))?;
    if input.len().checked_mul(size) != Some(bytes.len()) {
        return Err(invalid("IQ3 dot extents differ"));
    }
    let mut sum = 0.0_f32;
    for (block, activation) in bytes.chunks_exact(size).zip(input) {
        let (values, scales) = unpack(kind, block);
        let mut integer = 0_i32;
        for (group, (&subscale, values)) in
            scales.iter().zip(values.as_chunks::<32>().0).enumerate()
        {
            let mut local = 0_i32;
            for (index, &value) in values.iter().enumerate() {
                local += i32::from(value) * i32::from(activation.values[group * 32 + index]);
            }
            integer += local * i32::from(subscale);
        }
        let d = scale(block)? * finite(activation.scale)?;
        sum += d * integer as f32;
    }
    finite(if kind == 18 { 0.25 * sum } else { sum })
}
