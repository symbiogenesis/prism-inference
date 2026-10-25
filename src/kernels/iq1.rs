// Copyright (c) 2023-2026 The ggml authors
// SPDX-License-Identifier: MIT
//! Pinned `IQ1_S`/`IQ1_M` decoders and native `Q8_K` activation dots.
//!
//! Adapted from PrismML-Eng/llama.cpp at
//! `73bfe6cc485554822f80c473a3c0c4f25e80f41b`, ggml-common.h,
//! ggml-quants.c and ggml-cpu/quants.c. Reference encoding is unavailable.

#![allow(
    clippy::cast_precision_loss,
    clippy::as_conversions,
    clippy::suboptimal_flops,
    reason = "Pinned integer totals are bounded and native scalar arithmetic keeps multiply/add association."
)]

use super::{Q8KBlock, finite};
use crate::{Result, invalid};
use bonsai_ternary::half_to_float;

include!("iq1_tables.rs");

pub(super) const fn layout(kind: u32) -> Option<(usize, usize)> {
    match kind {
        19 => Some((256, 50)),
        29 => Some((256, 56)),
        _ => None,
    }
}

const fn word(block: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([block[offset], block[offset + 1]])
}

fn base_scale(kind: u32, block: &[u8]) -> Result<f32> {
    let bits = if kind == 19 {
        word(block, 0)
    } else {
        (word(block, 48) >> 12)
            | ((word(block, 50) >> 8) & 0x00f0)
            | ((word(block, 52) >> 4) & 0x0f00)
            | (word(block, 54) & 0xf000)
    };
    finite(half_to_float(bits))
}

fn subscale(kind: u32, block: &[u8], group: usize, half: usize) -> i32 {
    let bits = if kind == 19 {
        (word(block, 34 + 2 * group) >> 12) & 7
    } else {
        (word(block, 48 + 2 * (group / 2)) >> (6 * (group % 2) + 3 * half)) & 7
    };
    2 * i32::from(bits) + 1
}

fn grid(kind: u32, block: &[u8], group: usize, part: usize) -> ([i8; 8], i32) {
    let (low, high, negative) = if kind == 19 {
        let high = word(block, 34 + 2 * group);
        (
            block[2 + group * 4 + part],
            usize::from((high >> (3 * part)) & 7),
            high & 0x8000 != 0,
        )
    } else {
        let high = block[32 + group * 2 + part / 2] >> (4 * (part % 2));
        (
            block[group * 4 + part],
            usize::from(high & 7),
            high & 8 != 0,
        )
    };
    let code = usize::from(low) | (high << 8);
    (
        GRID[code]
            .to_le_bytes()
            .map(|byte| i8::from_ne_bytes([byte])),
        if negative { -1 } else { 1 },
    )
}

pub(super) fn validate(kind: u32, bytes: &[u8]) -> Result<()> {
    let (_, size) = layout(kind).ok_or_else(|| invalid("unknown IQ1 type"))?;
    if !bytes.len().is_multiple_of(size) {
        return Err(invalid("partial IQ1 block"));
    }
    for block in bytes.chunks_exact(size) {
        base_scale(kind, block)?;
    }
    Ok(())
}

pub(super) fn decode(kind: u32, bytes: &[u8], output: &mut Vec<f32>) -> Result<()> {
    validate(kind, bytes)?;
    let (_, size) = layout(kind).ok_or_else(|| invalid("unknown IQ1 type"))?;
    let count = (bytes.len() / size)
        .checked_mul(256)
        .ok_or_else(|| invalid("IQ1 decoded extent overflow"))?;
    output.try_reserve_exact(count)?;
    for block in bytes.chunks_exact(size) {
        let scale = base_scale(kind, block)?;
        for group in 0..8 {
            for part in 0..4 {
                let scale = scale * subscale(kind, block, group, part / 2) as f32;
                let (grid, shift) = grid(kind, block, group, part);
                let shift = 0.125 * shift as f32;
                for value in grid {
                    output.push(finite(scale * (f32::from(value) + shift))?);
                }
            }
        }
    }
    Ok(())
}

pub(super) fn dot_q8k(kind: u32, bytes: &[u8], input: &[Q8KBlock]) -> Result<f32> {
    validate(kind, bytes)?;
    let (_, size) = layout(kind).ok_or_else(|| invalid("unknown IQ1 type"))?;
    if input.len().checked_mul(size) != Some(bytes.len()) {
        return Err(invalid("IQ1 packed dot extents differ"));
    }
    let mut result = 0.0_f32;
    for (block, activation) in bytes.chunks_exact(size).zip(input) {
        let mut integer = 0_i32;
        let mut shift_integer = 0_i32;
        for group in 0..8 {
            let mut halves = [0_i32; 2];
            let mut shift_halves = [0_i32; 2];
            for part in 0..4 {
                let (grid, shift) = grid(kind, block, group, part);
                let mut local = 0_i32;
                let mut local_sum = 0_i32;
                for (index, value) in grid.into_iter().enumerate() {
                    let activation = i32::from(activation.values[group * 32 + part * 8 + index]);
                    local += activation * i32::from(value);
                    local_sum += activation;
                }
                halves[part / 2] += local;
                shift_halves[part / 2] += local_sum * shift;
            }
            if kind == 19 {
                let scale = subscale(kind, block, group, 0);
                let (_, shift) = grid(kind, block, group, 0);
                integer += scale * (halves[0] + halves[1]);
                // IQ1_S consumes the supplied native Q8_K sums. They are not
                // interchangeable with recomputing sums from activation bytes.
                let supplied_sum = i32::from(activation.sums[group * 2])
                    + i32::from(activation.sums[group * 2 + 1]);
                shift_integer += scale * shift * supplied_sum;
            } else {
                for half in 0..2 {
                    let scale = subscale(kind, block, group, half);
                    integer += scale * halves[half];
                    shift_integer += scale * shift_halves[half];
                }
            }
        }
        result += (base_scale(kind, block)? * finite(activation.scale)?)
            * (integer as f32 + 0.125 * shift_integer as f32);
    }
    finite(result)
}
