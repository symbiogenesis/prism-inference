// Copyright (c) 2023-2026 The ggml/llama.cpp authors
// SPDX-License-Identifier: MIT
//! Native legacy affine and five-bit row codecs, without decoded matrix storage.
#![allow(
    clippy::suboptimal_flops,
    clippy::suspicious_operation_groupings,
    reason = "Preserve the pinned scalar quantizer's separate binary32 arithmetic."
)]

use super::{Q4_1, Q5_0, Q5_1, Q8_1, Q8Block, finite, half_bits, integer_float, quantized, scale};
use crate::{Result, invalid};

fn half(value: f32) -> Result<[u8; 2]> {
    let bits = half_bits(finite(value)?);
    finite(bonsai_ternary::half_to_float(bits))?;
    Ok(bits.to_le_bytes())
}

fn codes(kind: u32, block: &[u8]) -> [i16; 32] {
    let affine = matches!(kind, Q4_1 | Q5_1);
    let five = kind != Q4_1;
    let offset = if affine { 4 } else { 2 };
    let high = if five {
        &block[offset..offset + 4]
    } else {
        &[]
    };
    let low = &block[offset + if five { 4 } else { 0 }..];
    std::array::from_fn(|index| {
        let mut value = i16::from((low[index % 16] >> (4 * (index / 16))) & 15);
        if five {
            value |= i16::from((high[index / 8] >> (index % 8)) & 1) << 4;
        }
        value - if affine { 0 } else { 16 }
    })
}

pub(super) fn decode(kind: u32, block: &[u8], output: &mut Vec<f32>) -> Result<()> {
    let d = scale(block)?;
    let minimum = if matches!(kind, Q4_1 | Q5_1) {
        scale(&block[2..])?
    } else {
        0.0
    };
    if kind == Q8_1 {
        scale(&block[2..])?;
        output.extend(
            block[4..]
                .iter()
                .map(|&byte| f32::from(i8::from_ne_bytes([byte])) * d),
        );
    } else {
        output.extend(codes(kind, block).map(|code| {
            let value = f32::from(code) * d;
            if kind == Q5_0 { value } else { value + minimum }
        }));
    }
    Ok(())
}

fn pack(kind: u32, d: f32, minimum: f32, values: &[u8; 32], output: &mut Vec<u8>) -> Result<()> {
    output.extend(half(d)?);
    if matches!(kind, Q4_1 | Q5_1) {
        output.extend(half(minimum)?);
    }
    if kind != Q4_1 {
        let mut high = [0_u8; 4];
        for (index, value) in values.iter().enumerate() {
            high[index / 8] |= ((value >> 4) & 1) << (index % 8);
        }
        output.extend(high);
    }
    output.extend((0..16).map(|index| (values[index] & 15) | ((values[index + 16] & 15) << 4)));
    Ok(())
}

pub(super) fn encode(kind: u32, input: &[f32], output: &mut Vec<u8>) -> Result<()> {
    for values in input.as_chunks::<32>().0 {
        if kind == Q8_1 {
            let maximum = values
                .iter()
                .fold(0.0_f32, |maximum, value| maximum.max(value.abs()));
            let d = maximum / 127.0;
            let inverse = if d == 0.0 { 0.0 } else { finite(1.0 / d)? };
            let mut packed = [0_i8; 32];
            for (target, &value) in packed.iter_mut().zip(values) {
                *target = quantized((value * inverse).round())?;
            }
            let sum: i32 = packed.iter().map(|&value| i32::from(value)).sum();
            output.extend(half(d)?);
            output.extend(half(integer_float(sum) * d)?);
            output.extend(packed.iter().flat_map(|value| value.to_le_bytes()));
            continue;
        }
        let affine = kind != Q5_0;
        let mut minimum = f32::MAX;
        let mut maximum = if affine { -f32::MAX } else { 0.0 };
        for &value in values {
            if value < minimum {
                minimum = value;
            }
            if (affine && value > maximum) || (!affine && value.abs() > maximum.abs()) {
                maximum = value;
            }
        }
        let d = if affine {
            (maximum - minimum) / if kind == Q4_1 { 15.0 } else { 31.0 }
        } else {
            maximum / -16.0
        };
        let inverse = if d == 0.0 { 0.0 } else { finite(1.0 / d)? };
        let mut packed = [0; 32];
        for (code, &value) in packed.iter_mut().zip(values) {
            let shifted = if affine {
                (value - minimum) * inverse + 0.5
            } else {
                value * inverse + 16.5
            };
            *code = u8::try_from(quantized(shifted.trunc().min(if kind == Q4_1 {
                15.0
            } else {
                31.0
            }))?)?;
        }
        pack(kind, d, minimum, &packed, output)?;
    }
    Ok(())
}

// make_qkx3_quants with the production legacy affine row parameters.
fn affine(input: &[f32; 32], weights: &[f32; 32], limit: u8) -> Result<(f32, f32, [u8; 32])> {
    let mut minimum = input[0];
    let mut maximum = input[0];
    let mut sum_weight = weights[0];
    let mut sum_input = weights[0] * input[0];
    for index in 1..32 {
        if input[index] < minimum {
            minimum = input[index];
        }
        if input[index] > maximum {
            maximum = input[index];
        }
        sum_weight += weights[index];
        sum_input += weights[index] * input[index];
    }
    if minimum > 0.0 {
        minimum = 0.0;
    }
    if maximum <= minimum {
        return Ok((0.0, minimum, [0; 32]));
    }
    let inverse = finite(f32::from(limit) / (maximum - minimum))?;
    let mut d = finite(1.0 / inverse)?;
    let assign = |inverse: f32, minimum: f32, values: &mut [u8; 32]| -> Result<()> {
        for (code, &value) in values.iter_mut().zip(input) {
            *code = u8::try_from(quantized(
                (inverse * (value - minimum))
                    .round_ties_even()
                    .clamp(0.0, f32::from(limit)),
            )?)?;
        }
        Ok(())
    };
    let mut best_codes = [0; 32];
    assign(inverse, minimum, &mut best_codes)?;
    let error = |d: f32, m: f32, values: &[u8; 32]| -> f32 {
        let mut sum = 0.0;
        for index in 0..32 {
            let diff = d * f32::from(values[index]) + m - input[index];
            sum += weights[index] * (diff * diff);
        }
        sum
    };
    let mut best_error = finite(error(d, minimum, &best_codes))?;
    for step in 0_u8..=36 {
        let inverse =
            finite((-0.9 + 0.05 * f32::from(step) + f32::from(limit)) / (maximum - minimum))?;
        let mut candidate = [0; 32];
        assign(inverse, minimum, &mut candidate)?;
        let mut sum_code = 0.0;
        let mut sum_squared = 0.0;
        let mut sum_product = 0.0;
        for index in 0..32 {
            let code = f32::from(candidate[index]);
            let weight = weights[index];
            sum_code += weight * code;
            sum_squared += weight * code * code;
            sum_product += weight * code * input[index];
        }
        let determinant = finite(sum_weight * sum_squared - sum_code * sum_code)?;
        if determinant > 0.0 {
            let mut candidate_d =
                finite((sum_weight * sum_product - sum_input * sum_code) / determinant)?;
            let mut candidate_m =
                finite((sum_squared * sum_input - sum_code * sum_product) / determinant)?;
            if candidate_m > 0.0 {
                candidate_m = 0.0;
                candidate_d = finite(sum_product / sum_squared)?;
            }
            let candidate_error = finite(error(candidate_d, candidate_m, &candidate))?;
            if candidate_error < best_error {
                best_error = candidate_error;
                best_codes = candidate;
                d = candidate_d;
                minimum = candidate_m;
            }
        }
    }
    Ok((d, minimum, best_codes))
}

pub(super) fn weighted(kind: u32, input: &[f32], importance: &[f32]) -> Result<Vec<u8>> {
    let mut sum_squared = 0.0;
    for &value in input {
        sum_squared += value * value;
    }
    let sigma = finite(sum_squared / super::bounded_count_float(u32::try_from(input.len())?))?;
    let mut output = Vec::new();
    output.try_reserve_exact(input.len() / 32 * super::layout(kind)?.1)?;
    for (values, importance) in input
        .as_chunks::<32>()
        .0
        .iter()
        .zip(importance.as_chunks::<32>().0)
    {
        let mut weights = [0.0; 32];
        for index in 0..32 {
            weights[index] =
                finite(importance[index] * (sigma + values[index] * values[index]).sqrt())?;
        }
        if kind == Q5_0 {
            let (bits, values) = super::weighted_symmetric_block(values, &weights, 16)?;
            pack(
                kind,
                bonsai_ternary::half_to_float(bits),
                0.0,
                &values,
                &mut output,
            )?;
        } else {
            let (d, minimum, values) =
                affine(values, &weights, if kind == Q4_1 { 15 } else { 31 })?;
            pack(kind, d, minimum, &values, &mut output)?;
        }
    }
    Ok(output)
}

pub(super) fn dot(kind: u32, bytes: &[u8], input: &[Q8Block]) -> Result<f32> {
    let (_, size) = super::layout(kind)?;
    if !matches!(kind, Q4_1 | Q5_0 | Q5_1) {
        return Err(invalid("legacy dot format is not supported"));
    }
    let mut result = 0.0;
    for (block, activation) in bytes.chunks_exact(size).zip(input) {
        let integer = super::dot_integer(&codes(kind, block), &activation.values);
        let mut value = (scale(block)? * activation.scale) * integer;
        if kind != Q5_0 {
            value += scale(&block[2..])? * finite(activation.sum)?;
        }
        result += value;
    }
    finite(result)
}
