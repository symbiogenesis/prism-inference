//! Prism `Q1_0`/`Q2_0` packed codecs and native `Q8_0` activation dots.
//!
//! Decoder adaptation: Camelid `1829a7e73775eb829efe8fabf46a6680897a975b`,
//! `src/tensor/mod.rs::decode_q1_0_tensor` and `decode_q2_0_tensor`
//! (MIT, Copyright (c) 2026 Tim Toole; see NOTICE).
//! Layouts, reference encoders and scalar dot arithmetic are checked against
//! PrismML-Eng/llama.cpp `73bfe6cc485554822f80c473a3c0c4f25e80f41b`,
//! `ggml/src/ggml-{common.h,quants.c}` and `ggml/src/ggml-cpu/quants.c`
//! (MIT; upstream LICENSE). Type 42 retains Prism's 64-element contract.

#![allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::suboptimal_flops,
    reason = "Native codecs use bounded integer accumulators and preserve scalar multiply/add association."
)]

use super::{Q8Block, finite, half_bits, scale};
use crate::{Result, invalid};
use bonsai_ternary::half_to_float;

pub(super) const fn layout(kind: u32) -> Option<(usize, usize)> {
    match kind {
        41 => Some((128, 18)),
        42 => Some((64, 18)),
        _ => None,
    }
}

pub(super) fn validate(kind: u32, bytes: &[u8]) -> Result<()> {
    layout(kind).ok_or_else(|| invalid("unknown sign quantization type"))?;
    if !bytes.len().is_multiple_of(18) {
        return Err(invalid("partial Q1_0/Q2_0 block"));
    }
    for block in bytes.as_chunks::<18>().0 {
        scale(block)?;
    }
    Ok(())
}

fn value(kind: u32, block: &[u8], index: usize) -> i16 {
    if kind == 41 {
        i16::from((block[2 + index / 8] >> (index % 8)) & 1) * 2 - 1
    } else {
        i16::from((block[2 + index / 4] >> (2 * (index % 4))) & 3) - 1
    }
}

pub(super) fn decode(kind: u32, bytes: &[u8], output: &mut Vec<f32>) -> Result<()> {
    validate(kind, bytes)?;
    let (count, _) = layout(kind).ok_or_else(|| invalid("unknown sign quantization type"))?;
    output.try_reserve_exact(bytes.len() / 18 * count)?;
    for block in bytes.as_chunks::<18>().0 {
        let d = scale(block)?;
        for index in 0..count {
            // Q1's direct +/- scale retains the native sign of a zero scale.
            output.push(finite(if kind == 41 {
                if value(kind, block, index) > 0 { d } else { -d }
            } else {
                f32::from(value(kind, block, index)) * d
            })?);
        }
    }
    Ok(())
}

pub(super) fn quantize(kind: u32, input: &[f32]) -> Result<Vec<u8>> {
    let (count, _) = layout(kind).ok_or_else(|| invalid("unknown sign quantization type"))?;
    if input.is_empty()
        || !input.len().is_multiple_of(count)
        || input.iter().any(|value| !value.is_finite())
    {
        return Err(invalid(
            "Q1_0/Q2_0 quantization needs complete finite blocks",
        ));
    }
    let mut output = Vec::new();
    output.try_reserve_exact(input.len() / count * 18)?;
    for values in input.chunks_exact(count) {
        let d = if kind == 41 {
            let mut sum = 0.0_f32;
            for &value in values {
                sum += value.abs();
            }
            finite(sum / 128.0)?
        } else {
            values.iter().fold(0.0_f32, |d, value| d.max(value.abs()))
        };
        let half = half_bits(d);
        finite(half_to_float(half))?;
        let mut block = [0_u8; 18];
        block[..2].copy_from_slice(&half.to_le_bytes());
        if kind == 41 {
            for (index, &value) in values.iter().enumerate() {
                if value >= 0.0 {
                    block[2 + index / 8] |= 1 << (index % 8);
                }
            }
        } else {
            let inverse = finite(if d > 0.0 { 1.0 / d } else { 0.0 })?;
            for (index, &value) in values.iter().enumerate() {
                // The native encoder normalizes with the unrounded scale.
                // Finite input and inverse can still overflow their product.
                let rounded = finite(value * inverse)?.round();
                let code = if rounded < -0.5 {
                    0
                } else if rounded < 0.5 {
                    1
                } else {
                    2
                };
                block[2 + index / 4] |= code << (2 * (index % 4));
            }
        }
        output.extend(block);
    }
    Ok(output)
}

pub(super) fn dot(kind: u32, bytes: &[u8], input: &[Q8Block]) -> Result<f32> {
    validate(kind, bytes)?;
    let (count, _) = layout(kind).ok_or_else(|| invalid("unknown sign quantization type"))?;
    let groups = count / 32;
    if (bytes.len() / 18).checked_mul(groups) != Some(input.len()) {
        return Err(invalid("Q1_0/Q2_0 dot extents differ"));
    }
    let mut sum = 0.0_f32;
    for (block, activations) in bytes
        .as_chunks::<18>()
        .0
        .iter()
        .zip(input.chunks_exact(groups))
    {
        let mut local = 0.0_f32;
        for (group, activation) in activations.iter().enumerate() {
            let mut integer = 0_i32;
            for (index, &activation) in activation.values.iter().enumerate() {
                integer +=
                    i32::from(value(kind, block, group * 32 + index)) * i32::from(activation);
            }
            local += finite(activation.scale)? * integer as f32;
        }
        sum += scale(block)? * local;
    }
    finite(sum)
}
