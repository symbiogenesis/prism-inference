//! Safe importance-codebook IQ2 decoders and native packed `Q8_K` dots.
//!
//! Adapted from PrismML-Eng/llama.cpp `73bfe6cc485554822f80c473a3c0c4f25e80f41b`,
//! `ggml/src/ggml-quants.c`, `ggml/src/ggml-cpu/quants.c`, and the tables in
//! `ggml/src/ggml-common.h` (MIT; see NOTICE). All fields are read from bytes;
//! no alignment, host endianness, or decoded matrix is assumed. `IQ2_S` also
//! exposes the pinned reference and importance fit; `IQ2_XXS/XS` do not.

#![allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::suboptimal_flops,
    reason = "The native dot rounds its bounded integer block sum once and preserves scalar multiply/add association."
)]

use super::{Q8KBlock, finite, scale};
use crate::{Result, invalid};

#[path = "iq2_tables.rs"]
mod tables;

pub(super) fn quantize(input: &[f32], importance: Option<&[f32]>) -> Result<Vec<u8>> {
    // Native encoder grid uses odd levels 1,3,5. The decode codebook's
    // optimized magnitudes 8,25,43 identify exactly those same grid indices.
    let points = tables::IQ2S.map(|word| word.to_le_bytes().map(|value| 2 * (value / 16) + 1));
    super::iqfit::quantize(22, input, importance, &points)
}

pub(super) const fn layout(kind: u32) -> Option<(usize, usize)> {
    match kind {
        16 => Some((256, 66)),
        17 => Some((256, 74)),
        22 => Some((256, 82)),
        _ => None,
    }
}

pub(super) fn validate(kind: u32, bytes: &[u8]) -> Result<()> {
    let (_, size) = layout(kind).ok_or_else(|| invalid("unknown IQ2 type"))?;
    if !bytes.len().is_multiple_of(size) {
        return Err(invalid("partial IQ2 block"));
    }
    for block in bytes.chunks_exact(size) {
        scale(block)?;
    }
    Ok(())
}

const fn sign_pattern(code: u8) -> u8 {
    // Native ksigns_iq2xs stores the seven supplied bits and an eighth bit
    // chosen so each complete sign pattern has even parity.
    code | if code.count_ones().is_multiple_of(2) {
        0
    } else {
        128
    }
}

fn group(kind: u32, block: &[u8], index: usize) -> ([u8; 8], u8, u8) {
    let group32 = index / 4;
    let local = index % 4;
    match kind {
        16 => {
            let offset = 2 + group32 * 8;
            let high = u32::from_le_bytes([
                block[offset + 4],
                block[offset + 5],
                block[offset + 6],
                block[offset + 7],
            ]);
            (
                tables::IQ2XXS[usize::from(block[offset + local])].to_le_bytes(),
                sign_pattern((high >> (7 * local)).to_le_bytes()[0] & 127),
                block[offset + 7] >> 4,
            )
        }
        17 => {
            let offset = 2 + index * 2;
            let code = u16::from_le_bytes([block[offset], block[offset + 1]]);
            (
                tables::IQ2XS[usize::from(code & 511)].to_le_bytes(),
                sign_pattern(block[offset + 1] >> 1),
                (block[66 + group32] >> (4 * (local / 2))) & 15,
            )
        }
        _ => {
            let code = usize::from(block[2 + index])
                | (usize::from((block[66 + group32] >> (2 * local)) & 3) << 8);
            (
                tables::IQ2S[code].to_le_bytes(),
                block[34 + index],
                (block[74 + group32] >> (4 * (local / 2))) & 15,
            )
        }
    }
}

pub(super) fn decode(kind: u32, bytes: &[u8], output: &mut Vec<f32>) -> Result<()> {
    validate(kind, bytes)?;
    let (_, size) = layout(kind).ok_or_else(|| invalid("unknown IQ2 type"))?;
    let count = (bytes.len() / size)
        .checked_mul(256)
        .ok_or_else(|| invalid("IQ2 decoded extent overflow"))?;
    output.try_reserve_exact(count)?;
    for block in bytes.chunks_exact(size) {
        let d = scale(block)?;
        for index in 0..32 {
            let (grid, signs, local) = group(kind, block, index);
            let d = d * (0.5 + f32::from(local)) * 0.25;
            for (lane, value) in grid.into_iter().enumerate() {
                let value = d * f32::from(value);
                output.push(finite(if signs & (1 << lane) == 0 {
                    value
                } else {
                    -value
                })?);
            }
        }
    }
    Ok(())
}

pub(super) fn dot(kind: u32, bytes: &[u8], input: &[Q8KBlock]) -> Result<f32> {
    validate(kind, bytes)?;
    let (_, size) = layout(kind).ok_or_else(|| invalid("unknown IQ2 type"))?;
    if input.len().checked_mul(size) != Some(bytes.len()) {
        return Err(invalid("IQ2 dot extents differ"));
    }
    let mut sum = 0.0_f32;
    for (block, input) in bytes.chunks_exact(size).zip(input) {
        let d = scale(block)? * finite(input.scale)?;
        let mut integer = 0_i32;
        for index in 0..32 {
            let (grid, signs, local) = group(kind, block, index);
            let mut group_sum = 0_i32;
            for (lane, value) in grid.into_iter().enumerate() {
                let product = i32::from(value) * i32::from(input.values[index * 8 + lane]);
                group_sum += if signs & (1 << lane) == 0 {
                    product
                } else {
                    -product
                };
            }
            integer += group_sum * (2 * i32::from(local) + 1);
        }
        sum += d * integer as f32;
    }
    // Native IQ2 kernels apply this factor after accumulating all blocks.
    finite(0.125 * sum)
}
