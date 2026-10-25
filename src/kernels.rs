//! Safe reference kernels and automatic baseline NEON packed integer arithmetic.
//!
//! The reference basis is PrismML-Eng/llama.cpp revision
//! `73bfe6cc485554822f80c473a3c0c4f25e80f41b`, `ggml/src/ggml-quants.c`
//! (SHA256 `46dd353f30e7f1c42c7d82d88d91fad64dbe5d6e680b234de25355d9fb482d13`)
//! and `ggml/src/ggml-cpu/quants.c`
//! (SHA256 `b8dd3a7c7da1ff702908d5b71c7c7791eb11e2481f6ce3d0ca91e4705f33bb65`).
//! These reference semantics do not establish full-model bit parity with Prism.
//! PQ2 uses signed `Q8_K` activations for rows divisible by 256, retaining the
//! pinned `Q8_0` fallback for odd numbers of 128-element groups. PTQ1 uses
//! `Q8_0` activations in groups of 32. Neither path expands a matrix to floats.

use bonsai_ternary::{Q8KBlock, decode_ptq_trits, half_to_float};

use crate::model::Tensor;
use crate::{Result, invalid};

mod fp4;
mod iq1;
mod iq2;
mod iq3;
mod iqfit;
mod iquants;
mod kquants;
mod legacy;
mod signquants;
mod tquants;

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
mod neon;

const F32: u32 = 0;
const F16: u32 = 1;
const Q4_0: u32 = 2;
const Q4_1: u32 = 3;
const Q5_0: u32 = 6;
const Q5_1: u32 = 7;
const Q8_0: u32 = 8;
const Q8_1: u32 = 9;
const Q8_K: u32 = 15;
const BF16: u32 = 30;
const PQ2: u32 = 142;
const PTQ1: u32 = 143;

fn layout(kind: u32) -> Result<(usize, usize)> {
    if let Some(layout) = fp4::layout(kind)
        .or_else(|| iquants::layout(kind))
        .or_else(|| signquants::layout(kind))
        .or_else(|| iq1::layout(kind))
        .or_else(|| iq2::layout(kind))
        .or_else(|| iq3::layout(kind))
    {
        return Ok(layout);
    }
    if let Some(bytes) = tquants::block_bytes(kind) {
        return Ok((256, bytes));
    }
    match kind {
        F32 => Ok((1, 4)),
        F16 | BF16 => Ok((1, 2)),
        Q4_0 => Ok((32, 18)),
        Q4_1 => Ok((32, 20)),
        Q5_0 => Ok((32, 22)),
        Q5_1 => Ok((32, 24)),
        Q8_0 => Ok((32, 34)),
        Q8_1 => Ok((32, 36)),
        Q8_K => Ok((256, 292)),
        10..=14 => Ok((
            256,
            kquants::block_bytes(kind).ok_or_else(|| invalid("unknown K quantization type"))?,
        )),
        PQ2 => Ok((128, 34)),
        PTQ1 => Ok((128, 28)),
        _ => Err(invalid(format!("unsupported runtime tensor type {kind}"))),
    }
}

fn dimensions(tensor: &Tensor) -> Result<(usize, usize, usize)> {
    let Some(&columns) = tensor.dimensions.first() else {
        return Err(invalid("tensor requires at least one dimension"));
    };
    let rows = tensor.dimensions[1..]
        .iter()
        .try_fold(1_usize, |count, &dimension| count.checked_mul(dimension))
        .ok_or_else(|| invalid("tensor row count overflow"))?;
    let (group, bytes) = layout(tensor.kind)?;
    if columns == 0 || rows == 0 || !columns.is_multiple_of(group) {
        return Err(invalid(
            "tensor dimensions do not contain complete nonempty blocks",
        ));
    }
    let row_bytes = (columns / group)
        .checked_mul(bytes)
        .ok_or_else(|| invalid("tensor row byte length overflow"))?;
    if row_bytes.checked_mul(rows) != Some(tensor.data.len()) {
        return Err(invalid("tensor data length does not match its dimensions"));
    }
    Ok((columns, rows, row_bytes))
}

fn finite(value: f32) -> Result<f32> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid("nonfinite tensor or activation value"))
    }
}

fn scale(bytes: &[u8]) -> Result<f32> {
    finite(half_to_float(u16::from_le_bytes([bytes[0], bytes[1]])))
}

fn fill_into<T>(output: &mut Vec<T>, fill: impl FnOnce(&mut Vec<T>) -> Result<()>) -> Result<()> {
    output.clear();
    let result = fill(output);
    if result.is_err() {
        output.clear();
    }
    result
}

fn collect_into<T>(
    values: impl ExactSizeIterator<Item = Result<T>>,
    output: &mut Vec<T>,
) -> Result<()> {
    fill_into(output, |output| {
        output.try_reserve_exact(values.len())?;
        for value in values {
            output.push(value?);
        }
        Ok(())
    })
}

fn pq2_values(block: &[u8]) -> [i16; 128] {
    std::array::from_fn(|index| i16::from((block[2 + index / 4] >> (2 * (index % 4))) & 3) - 1)
}

fn ptq1_values(block: &[u8]) -> Result<[i16; 128]> {
    Ok(decode_ptq_trits(<&[u8; 28]>::try_from(block)?))
}

/// Validates a supported tensor without retaining decoded rows or matrices.
///
/// # Errors
/// Rejects unsupported kinds, malformed dimensions and byte extents, nonfinite
/// scalar values or scales. Every native PTQ1 trit byte is decodable.
pub fn validate_tensor(tensor: &Tensor) -> Result<()> {
    dimensions(tensor)?;
    match tensor.kind {
        16 | 17 | 22 => return iq2::validate(tensor.kind, &tensor.data),
        18 | 21 => return iq3::validate(tensor.kind, &tensor.data),
        19 | 29 => return iq1::validate(tensor.kind, &tensor.data),
        20 | 23 => return iquants::validate(tensor.kind, &tensor.data),
        34 | 35 => return tquants::validate(tensor.kind, &tensor.data),
        39 | 40 => return fp4::validate(tensor.kind, &tensor.data),
        41 | 42 => return signquants::validate(tensor.kind, &tensor.data),
        _ => {}
    }
    match tensor.kind {
        F32 => {
            for chunk in tensor.data.as_chunks::<4>().0 {
                finite(f32::from_le_bytes(*chunk))?;
            }
        }
        F16 | BF16 => {
            for chunk in tensor.data.as_chunks::<2>().0 {
                let bits = u16::from_le_bytes(*chunk);
                finite(if tensor.kind == F16 {
                    half_to_float(bits)
                } else {
                    f32::from_bits(u32::from(bits) << 16)
                })?;
            }
        }
        PTQ1 => {
            for block in tensor.data.as_chunks::<28>().0 {
                scale(&block[26..])?;
                ptq1_values(block)?;
            }
        }
        Q8_K => {
            for block in tensor.data.as_chunks::<292>().0 {
                finite(f32::from_le_bytes(block[..4].try_into()?))?;
            }
        }
        10..=14 => {
            for block in tensor.data.chunks_exact(layout(tensor.kind)?.1) {
                kquants::validate(tensor.kind, block)?;
            }
        }
        _ => {
            let (_, block_bytes) = layout(tensor.kind)?;
            for block in tensor.data.chunks_exact(block_bytes) {
                scale(block)?;
                if matches!(tensor.kind, Q4_1 | Q5_1 | Q8_1) {
                    scale(&block[2..])?;
                }
            }
        }
    }
    Ok(())
}

/// Decodes a single contiguous row, including row zero of a vector tensor.
///
/// Only the selected row is expanded; the packed tensor remains immutable.
///
/// # Errors
/// Rejects unsupported kinds, incomplete blocks,
/// nonfinite values, malformed byte extents, and an out-of-range row.
pub fn row(tensor: &Tensor, index: usize) -> Result<Vec<f32>> {
    let mut output = Vec::new();
    row_into(tensor, index, &mut output)?;
    Ok(output)
}

/// Decodes a single row into reusable caller-owned storage.
///
/// Existing elements are discarded. Capacity is retained, including on error;
/// the output is empty whenever decoding fails. Only the selected row expands.
///
/// # Errors
/// Rejects the same malformed tensors and row indices as [`row`], and returns
/// an error if the output cannot be allocated.
pub fn row_into(tensor: &Tensor, index: usize, output: &mut Vec<f32>) -> Result<()> {
    fill_into(output, |output| row_impl(tensor, index, output))
}

fn row_impl(tensor: &Tensor, index: usize, output: &mut Vec<f32>) -> Result<()> {
    let (columns, rows, row_bytes) = dimensions(tensor)?;
    if index >= rows {
        return Err(invalid("tensor row index out of bounds"));
    }
    let bytes = &tensor.data[index * row_bytes..(index + 1) * row_bytes];
    decode_row_impl(tensor.kind, columns, bytes, output)
}

/// Decodes one complete contiguous row from borrowed packed bytes.
///
/// Byte alignment is unrestricted. The packed input is never copied into an
/// owned tensor. Existing output elements are discarded, capacity is retained,
/// and the output is empty on every error, including invalid layout or extent.
///
/// # Errors
/// Rejects unsupported kinds, empty or incomplete blocks, byte extent mismatch,
/// nonfinite stored values or scales, and output allocation failures.
pub fn decode_row_into(kind: u32, count: usize, bytes: &[u8], output: &mut Vec<f32>) -> Result<()> {
    fill_into(output, |output| {
        let (group, block_bytes) = layout(kind)?;
        if count == 0
            || !count.is_multiple_of(group)
            || (count / group).checked_mul(block_bytes) != Some(bytes.len())
        {
            return Err(invalid("packed row byte extent mismatch"));
        }
        decode_row_impl(kind, count, bytes, output)
    })
}

fn decode_row_impl(kind: u32, columns: usize, bytes: &[u8], output: &mut Vec<f32>) -> Result<()> {
    output.try_reserve_exact(columns)?;
    match kind {
        16 | 17 | 22 => return iq2::decode(kind, bytes, output),
        18 | 21 => return iq3::decode(kind, bytes, output),
        19 | 29 => return iq1::decode(kind, bytes, output),
        20 | 23 => return iquants::decode(kind, bytes, output),
        34 | 35 => return tquants::decode(kind, bytes, output),
        39 | 40 => return fp4::decode(kind, bytes, output),
        41 | 42 => return signquants::decode(kind, bytes, output),
        _ => {}
    }
    if matches!(kind, 10..=14) {
        return kquants::decode(kind, bytes, output);
    }
    match kind {
        F32 => {
            for chunk in bytes.as_chunks::<4>().0 {
                output.push(finite(f32::from_le_bytes(*chunk))?);
            }
        }
        F16 | BF16 => {
            for chunk in bytes.as_chunks::<2>().0 {
                let bits = u16::from_le_bytes(*chunk);
                output.push(finite(if kind == F16 {
                    half_to_float(bits)
                } else {
                    f32::from_bits(u32::from(bits) << 16)
                })?);
            }
        }
        PQ2 | PTQ1 => {
            let (_, block_bytes) = layout(kind)?;
            for block in bytes.chunks_exact(block_bytes) {
                let (values, d) = if kind == PQ2 {
                    (pq2_values(block), scale(block)?)
                } else {
                    (ptq1_values(block)?, scale(&block[26..])?)
                };
                output.extend(values.map(|value| f32::from(value) * d));
            }
        }
        Q4_0 => {
            for block in bytes.as_chunks::<18>().0 {
                let d = scale(block)?;
                for shift in [0, 4] {
                    output.extend(
                        block[2..]
                            .iter()
                            .map(|&byte| f32::from(i16::from((byte >> shift) & 15) - 8) * d),
                    );
                }
            }
        }
        Q8_0 => {
            for block in bytes.as_chunks::<34>().0 {
                let d = scale(block)?;
                output.extend(
                    block[2..]
                        .iter()
                        .map(|&byte| f32::from(i8::from_ne_bytes([byte])) * d),
                );
            }
        }
        Q8_K => {
            for block in bytes.as_chunks::<292>().0 {
                let d = finite(f32::from_le_bytes(block[..4].try_into()?))?;
                output.extend(
                    block[4..260]
                        .iter()
                        .map(|&byte| f32::from(i8::from_ne_bytes([byte])) * d),
                );
            }
        }
        Q4_1 | Q5_0 | Q5_1 | Q8_1 => {
            for block in bytes.chunks_exact(layout(kind)?.1) {
                legacy::decode(kind, block, output)?;
            }
        }
        _ => return Err(invalid("unsupported row decoder")),
    }
    Ok(())
}

// Bounded casts implement the native float quantizers, with checked finite
// inputs and an explicit integer range check before conversion.
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "The finite rounded activation is range checked before conversion to i8."
)]
fn quantized(value: f32) -> Result<i8> {
    if value.is_finite() && (-127.0..=127.0).contains(&value) {
        Ok(value as i8)
    } else {
        Err(invalid(
            "activation quantization exceeded its finite i8 range",
        ))
    }
}

fn q8k_into(input: &[f32], output: &mut Vec<Q8KBlock>) -> Result<()> {
    if !input.len().is_multiple_of(256) {
        return Err(invalid(
            "Q8_K activations require a multiple of 256 columns",
        ));
    }
    collect_into(
        input.as_chunks::<256>().0.iter().map(|chunk| {
            let mut maximum = 0.0_f32;
            for &value in chunk {
                finite(value)?;
                if value.abs() > maximum.abs() {
                    maximum = value;
                }
            }
            let mut block = Q8KBlock {
                scale: 0.0,
                values: [0; 256],
                sums: [0; 16],
            };
            if maximum == 0.0 {
                return Ok(block);
            }
            let inverse_scale = -127.0 / maximum;
            if !inverse_scale.is_finite() {
                return Err(invalid("Q8_K activation scale overflow"));
            }
            block.scale = 1.0 / inverse_scale;
            for (target, &value) in block.values.iter_mut().zip(chunk) {
                // nearest_int in the pinned C reference is ties-to-even.
                *target = quantized((value * inverse_scale).round_ties_even().min(127.0))?;
            }
            for (sum, values) in block.sums.iter_mut().zip(block.values.as_chunks::<16>().0) {
                *sum = values.iter().map(|&value| i16::from(value)).sum();
            }
            Ok(block)
        }),
        output,
    )
}

struct Q8Block {
    scale: f32,
    sum: f32,
    values: [i8; 32],
}

fn q8_into(input: &[f32], output: &mut Vec<Q8Block>) -> Result<()> {
    if !input.len().is_multiple_of(32) {
        return Err(invalid("Q8_0 activations require a multiple of 32 columns"));
    }
    collect_into(
        input.as_chunks::<32>().0.iter().map(|chunk| {
            let mut maximum = 0.0_f32;
            for &value in chunk {
                maximum = maximum.max(finite(value)?.abs());
            }
            let d = maximum / 127.0;
            let inverse_scale = if d == 0.0 { 0.0 } else { 1.0 / d };
            let mut block = Q8Block {
                scale: round_f16(d),
                sum: 0.0,
                values: [0; 32],
            };
            finite(block.scale)?;
            for (target, &value) in block.values.iter_mut().zip(chunk) {
                *target = quantized((value * inverse_scale).round())?;
            }
            let sum = block.values.iter().map(|&value| i32::from(value)).sum();
            block.sum = round_f16(integer_float(sum) * d);
            Ok(block)
        }),
        output,
    )
}

fn quantize_ternary(input: &[f32]) -> Result<[u8; 34]> {
    let maximum = input.iter().try_fold(0.0_f32, |maximum, &value| {
        Ok::<_, crate::Error>(maximum.max(finite(value)?.abs()))
    })?;
    let inverse = if maximum > 0.0 {
        finite(1.0 / maximum)?
    } else {
        0.0
    };
    let scale = half_bits(maximum);
    finite(half_to_float(scale))?;
    let mut block = [0; 34];
    block[..2].copy_from_slice(&scale.to_le_bytes());
    for (index, &value) in input.iter().enumerate() {
        let code = u8::try_from(i16::from(quantized((value * inverse).round())?) + 1)?;
        block[2 + index / 4] |= code.min(3) << (2 * (index % 4));
    }
    Ok(block)
}

/// Encodes one or more complete native blocks using deterministic reference quantizers.
///
/// Supported kinds include F32, F16, BF16, the legacy Q4/Q5/Q8 formats,
/// `Q8_K`, `Q1_0`, `Q2_0`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, PQ2 and PTQ1.
/// Binary16 and bfloat16 use ties-to-even; ternary and `Q8_0` integer rounding
/// uses ties away from zero. PQ2/PTQ1 share identical scales and ternary values.
/// Zero scales retain each native format's original signed-zero representation.
///
/// # Errors
/// Rejects unsupported kinds, empty or incomplete blocks, nonfinite inputs,
/// unrepresentable packed scales, inverse-scale overflow and allocation errors.
#[allow(
    clippy::suboptimal_flops,
    reason = "Preserve native reference quantizer multiply-then-add rounding."
)]
pub fn quantize_row(kind: u32, input: &[f32]) -> Result<Vec<u8>> {
    match kind {
        10..=14 => return kquants::quantize(kind, input),
        18 | 21 => return iq3::quantize(kind, input, None),
        22 => return iq2::quantize(input, None),
        20 | 23 => return iquants::quantize(kind, input, None),
        34 | 35 => return tquants::quantize(kind, input),
        39 | 40 => return fp4::quantize(kind, input),
        41 | 42 => return signquants::quantize(kind, input),
        _ => {}
    }
    let (group, bytes) = layout(kind)?;
    if input.is_empty()
        || !input.len().is_multiple_of(group)
        || input.iter().any(|value| !value.is_finite())
    {
        return Err(invalid(
            "quantizer requires complete nonempty finite blocks",
        ));
    }
    let size = (input.len() / group)
        .checked_mul(bytes)
        .ok_or_else(|| invalid("quantizer byte extent overflow"))?;
    let mut output = Vec::new();
    output.try_reserve_exact(size)?;
    match kind {
        F32 => output.extend(input.iter().flat_map(|value| value.to_le_bytes())),
        F16 | BF16 => {
            for &value in input {
                let bits = if kind == F16 {
                    let bits = half_bits(value);
                    finite(half_to_float(bits))?;
                    bits
                } else {
                    let rounded = round_bf16(value);
                    finite(rounded)?;
                    u16::try_from(rounded.to_bits() >> 16)?
                };
                output.extend(bits.to_le_bytes());
            }
        }
        PQ2 | PTQ1 => {
            for values in input.as_chunks::<128>().0 {
                let mut block = quantize_ternary(values)?;
                if kind == PQ2 {
                    output.extend(block);
                } else {
                    // Canonical conversion validates a nonzero scale; runtime
                    // quantization also defines zero-scale blocks. Pack trits
                    // under a neutral scale, then retain the original scale.
                    let scale = [block[0], block[1]];
                    block[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
                    let mut packed = bonsai_ternary::pack_ptq(&block)?;
                    packed[26..].copy_from_slice(&scale);
                    output.extend(packed);
                }
            }
        }
        Q8_0 => {
            let mut blocks = Vec::new();
            q8_into(input, &mut blocks)?;
            for block in blocks {
                output.extend(half_bits(block.scale).to_le_bytes());
                output.extend(block.values.iter().flat_map(|value| value.to_le_bytes()));
            }
        }
        Q8_K => {
            let mut blocks = Vec::new();
            q8k_into(input, &mut blocks)?;
            for block in blocks {
                output.extend(block.scale.to_le_bytes());
                output.extend(block.values.iter().flat_map(|value| value.to_le_bytes()));
                output.extend(block.sums.iter().flat_map(|sum| sum.to_le_bytes()));
            }
        }
        Q4_0 => {
            for values in input.as_chunks::<32>().0 {
                let mut maximum = 0.0_f32;
                for &value in values {
                    if value.abs() > maximum.abs() {
                        maximum = value;
                    }
                }
                let d = maximum / -8.0;
                let inverse = if d == 0.0 { 0.0 } else { finite(1.0 / d)? };
                let scale = half_bits(d);
                finite(half_to_float(scale))?;
                output.extend(scale.to_le_bytes());
                for index in 0..16 {
                    let low = u8::try_from(quantized(
                        (values[index] * inverse + 8.5).trunc().min(15.0),
                    )?)?;
                    let high = u8::try_from(quantized(
                        (values[index + 16] * inverse + 8.5).trunc().min(15.0),
                    )?)?;
                    output.push(low | (high << 4));
                }
            }
        }
        Q4_1 | Q5_0 | Q5_1 | Q8_1 => legacy::encode(kind, input, &mut output)?,
        _ => return Err(invalid("unsupported reference row quantizer")),
    }
    Ok(output)
}

#[allow(
    clippy::suboptimal_flops,
    clippy::suspicious_operation_groupings,
    reason = "Preserve weighted native quantizer search arithmetic association."
)]
fn weighted_symmetric_block(
    input: &[f32; 32],
    weights: &[f32; 32],
    limit: i8,
) -> Result<(u16, [u8; 32])> {
    let mut maximum = 0.0_f32;
    for &value in input {
        if value.abs() > maximum.abs() {
            maximum = value;
        }
    }
    if maximum.abs() < 1e-15 {
        return Ok((0, [0; 32]));
    }
    let mut codes = [0_u8; 32];
    let bound = f32::from(limit);
    let mut inverse = finite(-bound / maximum)?;
    let sums = |inverse: f32| -> Result<(f32, f32)> {
        let mut weighted_values = 0.0;
        let mut weighted_codes = 0.0;
        for (&value, &weight) in input.iter().zip(weights) {
            let code = f32::from(quantized(
                (inverse * value)
                    .round_ties_even()
                    .clamp(-bound, bound - 1.0),
            )?);
            weighted_values += weight * value * code;
            weighted_codes += weight * code * code;
        }
        Ok((finite(weighted_values)?, finite(weighted_codes)?))
    };
    let assign = |codes: &mut [u8; 32], inverse: f32| -> Result<()> {
        for (code, &value) in codes.iter_mut().zip(input) {
            *code = u8::try_from(
                i16::from(limit)
                    + i16::from(quantized(
                        (inverse * value)
                            .round_ties_even()
                            .clamp(-bound, bound - 1.0),
                    )?),
            )?;
        }
        Ok(())
    };
    assign(&mut codes, inverse)?;
    let (values, codes_squared) = sums(inverse)?;
    let mut scale = if codes_squared == 0.0 {
        0.0
    } else {
        values / codes_squared
    };
    let mut best = scale * values;
    for step in -9_i16..=9 {
        if step == 0 {
            continue;
        }
        inverse = finite(-(bound + 0.1 * f32::from(step)) / maximum)?;
        let (values, codes_squared) = sums(inverse)?;
        if codes_squared > 0.0 && values * values > best * codes_squared {
            assign(&mut codes, inverse)?;
            scale = values / codes_squared;
            best = scale * values;
        }
    }
    let scale = half_bits(finite(scale)?);
    finite(half_to_float(scale))?;
    Ok((scale, codes))
}

/// Encodes one row with an explicit per-element importance vector.
///
/// `Q4_0` uses the native weighted scale search, including row-wide variance.
/// Scalar formats, `Q8_0`, PQ2 and PTQ1 retain their native behavior in which
/// importance does not alter the output. The same row importance vector may
/// be reused by the caller for subsequent rows.
/// `IQ2_S`, `IQ3_XXS` and `IQ3_S` retain their codebook-specific native fits.
///
/// # Errors
/// Rejects unequal or empty extents, negative/nonfinite importance, malformed
/// blocks, nonfinite inputs, numerical overflow and unsupported kinds.
#[allow(
    clippy::suboptimal_flops,
    reason = "Preserve source row-wide sum of squares and per-element weighting association."
)]
pub fn quantize_row_weighted(kind: u32, input: &[f32], importance: &[f32]) -> Result<Vec<u8>> {
    if matches!(kind, 18 | 21) {
        return iq3::quantize(kind, input, Some(importance));
    }
    if kind == 22 {
        return iq2::quantize(input, Some(importance));
    }
    if matches!(kind, 20 | 23) {
        return iquants::quantize(kind, input, Some(importance));
    }
    if matches!(kind, 10..=14) {
        return kquants::quantize_weighted(kind, input, importance);
    }
    if input.is_empty()
        || input.len() != importance.len()
        || importance
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(invalid(
            "quantization importance must be finite, nonnegative and match the row",
        ));
    }
    if !matches!(kind, Q4_0 | Q4_1 | Q5_0 | Q5_1) {
        return quantize_row(kind, input);
    }
    if !input.len().is_multiple_of(32) || input.iter().any(|value| !value.is_finite()) {
        return Err(invalid(
            "weighted legacy quantization requires complete finite blocks",
        ));
    }
    if kind != Q4_0 {
        return legacy::weighted(kind, input, importance);
    }
    let mut sum_squared = 0.0;
    for &value in input {
        sum_squared += value * value;
    }
    let count = u32::try_from(input.len())?;
    let sigma = finite(sum_squared / bounded_count_float(count))?;
    let mut output = Vec::new();
    output.try_reserve_exact(input.len() / 32 * 18)?;
    for (values, importance) in input
        .as_chunks::<32>()
        .0
        .iter()
        .zip(importance.as_chunks::<32>().0)
    {
        let mut weights = [0.0; 32];
        for (index, weight) in weights.iter_mut().enumerate() {
            *weight = finite(importance[index] * (sigma + values[index] * values[index]).sqrt())?;
        }
        let (scale, codes) = weighted_symmetric_block(values, &weights, 8)?;
        output.extend(scale.to_le_bytes());
        output.extend((0..16).map(|index| codes[index] | (codes[index + 16] << 4)));
    }
    Ok(output)
}

#[allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Native row variance divides by a float-converted bounded element count."
)]
const fn bounded_count_float(count: u32) -> f32 {
    count as f32
}

#[allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Bounded dot sums and power-of-two transform lengths are exactly representable."
)]
const fn integer_float(value: i32) -> f32 {
    // Packed dot sums and power-of-two Hadamard lengths are exactly
    // representable in f32 for the supported dimensions.
    value as f32
}

fn dot_integer(weights: &[i16], activations: &[i8]) -> f32 {
    integer_float(
        weights
            .iter()
            .zip(activations)
            .map(|(&weight, &activation)| i32::from(weight) * i32::from(activation))
            .sum(),
    )
}

#[allow(
    clippy::suboptimal_flops,
    reason = "Preserve scalar reference multiply-then-add rounding."
)]
fn dot_pq2(bytes: &[u8], input: &[Q8KBlock]) -> Result<f32> {
    let mut output = 0.0;
    for (index, block) in bytes.as_chunks::<34>().0.iter().enumerate() {
        let activation = &input[index / 2];
        let start = (index % 2) * 128;
        output += (scale(block)? * activation.scale)
            * dot_integer(&pq2_values(block), &activation.values[start..start + 128]);
    }
    finite(output)
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
#[allow(
    clippy::suboptimal_flops,
    reason = "Preserve scalar reference multiply-then-add rounding."
)]
fn dot_pq2_neon(bytes: &[u8], input: &[Q8KBlock]) -> Result<f32> {
    let mut output = 0.0;
    for (index, block) in bytes.as_chunks::<34>().0.iter().enumerate() {
        let activation = &input[index / 2];
        let start = (index % 2) * 128;
        output += (scale(block)? * activation.scale)
            * integer_float(neon::dot(
                block[2..].try_into()?,
                activation.values[start..start + 128].try_into()?,
            ));
    }
    finite(output)
}

#[allow(
    clippy::suboptimal_flops,
    reason = "Preserve scalar reference multiply-then-add rounding."
)]
fn dot_q8_groups(kind: u32, bytes: &[u8], input: &[Q8Block]) -> Result<f32> {
    let (_, block_bytes) = layout(kind)?;
    let mut output = 0.0;
    for (index, block) in bytes.chunks_exact(block_bytes).enumerate() {
        let (weights, d) = if kind == PQ2 {
            (pq2_values(block), scale(block)?)
        } else {
            (ptq1_values(block)?, scale(&block[26..])?)
        };
        let mut sum = 0.0;
        for (values, activation) in weights
            .as_chunks::<32>()
            .0
            .iter()
            .zip(&input[index * 4..index * 4 + 4])
        {
            sum += activation.scale * dot_integer(values, &activation.values);
        }
        output += d * sum;
    }
    finite(output)
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
#[allow(
    clippy::suboptimal_flops,
    reason = "Preserve scalar reference multiply-then-add rounding."
)]
fn dot_q8_groups_neon(kind: u32, bytes: &[u8], input: &[Q8Block]) -> Result<f32> {
    let (_, block_bytes) = layout(kind)?;
    let mut output = 0.0;
    for (index, block) in bytes.chunks_exact(block_bytes).enumerate() {
        let activations = <&[Q8Block; 4]>::try_from(&input[index * 4..index * 4 + 4])?;
        let values = activations.each_ref().map(|activation| &activation.values);
        let (sums, d) = if kind == PQ2 {
            (
                neon::dot_groups(block[2..].try_into()?, values),
                scale(block)?,
            )
        } else {
            (
                neon::dot_ptq1_groups(block.try_into()?, values)?,
                scale(&block[26..])?,
            )
        };
        let mut sum = 0.0;
        for (integer, activation) in sums.into_iter().zip(activations) {
            sum += activation.scale * integer_float(integer);
        }
        output += d * sum;
    }
    finite(output)
}

#[allow(
    clippy::suboptimal_flops,
    reason = "Preserve scalar reference multiply-then-add rounding."
)]
fn dot_additional(kind: u32, bytes: &[u8], q8: &[Q8Block], q8k: &[Q8KBlock]) -> Result<f32> {
    match kind {
        16 | 17 | 22 => iq2::dot(kind, bytes, q8k),
        18 | 21 => iq3::dot(kind, bytes, q8k),
        19 | 29 => iq1::dot_q8k(kind, bytes, q8k),
        20 => iquants::dot_q8(bytes, q8),
        23 => iquants::dot_q8k(bytes, q8k),
        34 | 35 => tquants::dot(kind, bytes, q8k),
        39 | 40 => fp4::dot(kind, bytes, q8),
        41 | 42 => signquants::dot(kind, bytes, q8),
        _ => Err(invalid("unknown additional packed dot format")),
    }
}

#[allow(
    clippy::suboptimal_flops,
    reason = "Preserve scalar reference multiply-then-add rounding."
)]
fn dot_legacy(kind: u32, bytes: &[u8], input: &[Q8Block]) -> Result<f32> {
    if matches!(kind, Q4_1 | Q5_0 | Q5_1) {
        return legacy::dot(kind, bytes, input);
    }
    let (_, block_bytes) = layout(kind)?;
    let mut output = 0.0;
    for (block, activation) in bytes.chunks_exact(block_bytes).zip(input) {
        let values: [i16; 32] = std::array::from_fn(|index| {
            if kind == Q4_0 {
                i16::from((block[2 + index % 16] >> (4 * (index / 16))) & 15) - 8
            } else {
                i16::from(i8::from_ne_bytes([block[2 + index]]))
            }
        });
        output += (scale(block)? * activation.scale) * dot_integer(&values, &activation.values);
    }
    finite(output)
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "The scalar reference rounds the accumulated binary64 sum to binary32."
)]
fn float_dot(weights: &[f32], input: &[f32]) -> f32 {
    // The non-SIMD ggml reference sums rounded f32 products in f64.
    weights
        .iter()
        .zip(input)
        .map(|(&a, &b)| f64::from(a * b))
        .sum::<f64>() as f32
}

/// Computes a scalar dot directly from native packed weight and activation bytes.
///
/// Activations use `Q8_K` for PQ2, `Q8_0` for PTQ1/`Q1_0`/`Q2_0`/`Q4_0`/`Q8_0`,
/// and the weight's own scalar format for F32/F16/BF16. Supplied integer activations
/// and scales are consumed exactly; no float activation quantization occurs.
///
/// # Errors
/// Rejects unsupported kinds, empty or incomplete blocks, mismatched byte
/// extents, nonfinite stored values/scales, nonfinite accumulated results and
/// temporary activation allocation failures.
pub fn packed_dot(kind: u32, count: usize, weights: &[u8], activations: &[u8]) -> Result<f32> {
    if matches!(kind, Q8_K | Q8_1) {
        return Err(invalid(
            "native auxiliary Q8 format has no packed weight dot entry point",
        ));
    }
    let (group, block_bytes) = layout(kind)?;
    if count == 0
        || !count.is_multiple_of(group)
        || count
            .checked_div(group)
            .and_then(|blocks| blocks.checked_mul(block_bytes))
            != Some(weights.len())
    {
        return Err(invalid("packed dot weight extent mismatch"));
    }
    let (activation_group, activation_bytes) = match kind {
        PQ2 | 10..=14 | 16..=19 | 21..=23 | 29 | 34 | 35 => (256, 292),
        PTQ1 | Q4_0 | Q5_0 | Q8_0 | 20 | 39..=42 => (32, 34),
        Q4_1 | Q5_1 => (32, 36),
        _ => (group, block_bytes),
    };
    if !count.is_multiple_of(activation_group)
        || (count / activation_group).checked_mul(activation_bytes) != Some(activations.len())
    {
        return Err(invalid("packed dot activation extent mismatch"));
    }
    match kind {
        PQ2 | 10..=14 | 16..=19 | 21..=23 | 29 | 34 | 35 => {
            let mut input = Vec::new();
            collect_into(
                activations.as_chunks::<292>().0.iter().map(|block| {
                    Ok(Q8KBlock {
                        scale: finite(f32::from_le_bytes(block[..4].try_into()?))?,
                        values: std::array::from_fn(|index| i8::from_ne_bytes([block[4 + index]])),
                        sums: std::array::from_fn(|index| {
                            i16::from_le_bytes([block[260 + 2 * index], block[261 + 2 * index]])
                        }),
                    })
                }),
                &mut input,
            )?;
            if kind == PQ2 {
                dot_pq2(weights, &input)
            } else if matches!(kind, 16..=19 | 21..=23 | 29 | 34 | 35) {
                dot_additional(kind, weights, &[], &input)
            } else {
                kquants::dot(kind, weights, &input)
            }
        }
        PTQ1 | Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | 20 | 39..=42 => {
            let affine = matches!(kind, Q4_1 | Q5_1);
            let offset = if affine { 4 } else { 2 };
            let mut input = Vec::new();
            collect_into(
                activations.chunks_exact(activation_bytes).map(|block| {
                    Ok(Q8Block {
                        scale: scale(block)?,
                        sum: if affine { scale(&block[2..])? } else { 0.0 },
                        values: std::array::from_fn(|index| {
                            i8::from_ne_bytes([block[offset + index]])
                        }),
                    })
                }),
                &mut input,
            )?;
            if matches!(kind, 20 | 39..=42) {
                dot_additional(kind, weights, &input, &[])
            } else if kind == PTQ1 {
                dot_q8_groups(kind, weights, &input)
            } else {
                dot_legacy(kind, weights, &input)
            }
        }
        _ => {
            let decode = |data: &[u8]| -> Result<Vec<f32>> {
                let mut values = Vec::new();
                decode_row_into(kind, count, data, &mut values)?;
                Ok(values)
            };
            finite(float_dot(&decode(weights)?, &decode(activations)?))
        }
    }
}

/// Reusable activation and decoded-row storage for [`matvec_into`].
///
/// Construct with [`Default::default`]. Storage grows fallibly to accommodate
/// each call and retains its capacity across tensors, activation formats and
/// errors. Scratch contains no tensor-sized decoded matrix or model state.
pub struct MatvecScratch {
    q8k: Vec<Q8KBlock>,
    q8: Vec<Q8Block>,
    activations: Vec<f32>,
    row: Vec<f32>,
    diagnostics: MatvecDiagnostics,
    statistics: MatvecStatistics,
    #[cfg(test)]
    preparations: usize,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum ActivationContract {
    Q8K,
    Q8,
    F32,
    F16,
    BF16,
}

impl ActivationContract {
    const fn for_tensor(tensor: &Tensor, columns: usize) -> Self {
        match tensor.kind {
            10..=14 | 16..=19 | 21..=23 | 29 | 34 | 35 => Self::Q8K,
            PQ2 if columns.is_multiple_of(256) => Self::Q8K,
            PQ2 | PTQ1 | Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | 20 | 39..=42 => Self::Q8,
            F16 => Self::F16,
            BF16 => Self::BF16,
            _ => Self::F32,
        }
    }
}

/// One scoped activation preparation shared by compatible matrix operations.
///
/// Created by [`MatvecScratch::prepare`]. The guard borrows both its exact input
/// and scratch until dropped, preventing input mutation or scratch reuse between
/// sibling projections. No pointer/content cache survives the scope. Tensor
/// weights must already use the same input basis, signs and permutation.
pub struct PreparedMatvec<'a> {
    input: &'a [f32],
    contract: ActivationContract,
    scratch: &'a mut MatvecScratch,
}

/// Packed matrix kernel selection, independent of model and activation precision.
///
/// Automatic selection uses the compiled baseline NEON implementation on native
/// Windows ARM64 MSVC for complete PQ2 and PTQ1 blocks. Other targets and formats
/// use scalar. Select [`Self::Scalar`] to retain the reference implementation.
/// No optional CPU extension is enabled or inferred by this selection.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MatvecBackend {
    /// Selects the best available packed kernel for the build, target and format.
    #[default]
    Auto,
    /// Scalar reference implementation and unconditional fallback.
    Scalar,
    /// Explicit baseline NEON request, requiring the `experimental-neon` feature.
    ///
    /// Exact synthetic parity does not qualify whole-model performance or parity.
    ExperimentalNeon,
}

/// Why the selected matrix operation uses scalar arithmetic.
///
/// Reasons have a fixed precedence: an explicit scalar request, a tensor kind
/// without a NEON kernel, a disabled feature, then an unsupported build target.
/// This reports dispatch policy and compile-time eligibility, not CPU probing,
/// whole-model qualification or measured performance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatvecFallbackReason {
    /// The caller explicitly requested the scalar backend.
    ScalarRequested,
    /// This supported tensor kind has no NEON matrix kernel.
    TensorKindUnsupported,
    /// The build omitted the `experimental-neon` feature.
    FeatureDisabled,
    /// The build is not Windows ARM64 MSVC with baseline NEON enabled.
    TargetUnsupported,
}

/// Dispatch and completion information for the most recent matrix attempt.
///
/// A new preparation or prepared matrix attempt resets this record, including
/// on validation failure. Earlier successful operations are not retained.
/// Neither a selected backend nor build eligibility establishes model parity
/// or performance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MatvecDiagnostics {
    /// Configured policy, preserved across preparations and failures.
    pub requested_backend: MatvecBackend,
    /// Actual row backend selected; scalar until dispatch occurs.
    pub selected_backend: MatvecBackend,
    /// Whether this build includes the supported baseline NEON implementation.
    pub neon_build_eligible: bool,
    /// Scalar selection reason, absent before dispatch and for NEON selection.
    pub fallback_reason: Option<MatvecFallbackReason>,
    /// A row backend was selected after dimension, activation and output allocation checks.
    ///
    /// Row validation or arithmetic may still fail after selection.
    pub dispatched: bool,
    /// The most recent matrix attempt returned success with complete output.
    ///
    /// Always false before dispatch and after any failed attempt.
    pub succeeded: bool,
}

impl MatvecDiagnostics {
    const fn new(requested_backend: MatvecBackend) -> Self {
        Self {
            requested_backend,
            selected_backend: MatvecBackend::Scalar,
            neon_build_eligible: cfg!(all(
                feature = "experimental-neon",
                target_arch = "aarch64",
                target_os = "windows",
                target_env = "msvc",
                target_feature = "neon"
            )),
            fallback_reason: None,
            dispatched: false,
            succeeded: false,
        }
    }

    const fn select(mut self, supports_neon: bool) -> Self {
        self.fallback_reason = if matches!(self.requested_backend, MatvecBackend::Scalar) {
            Some(MatvecFallbackReason::ScalarRequested)
        } else if !supports_neon {
            Some(MatvecFallbackReason::TensorKindUnsupported)
        } else if !cfg!(feature = "experimental-neon") {
            Some(MatvecFallbackReason::FeatureDisabled)
        } else if !self.neon_build_eligible {
            Some(MatvecFallbackReason::TargetUnsupported)
        } else {
            None
        };
        self.selected_backend = if self.fallback_reason.is_none() {
            MatvecBackend::ExperimentalNeon
        } else {
            MatvecBackend::Scalar
        };
        self.dispatched = true;
        self
    }
}

/// Aggregate matrix work observed by one scratch owner over its lifetime.
///
/// Counts include provisional and failed work and survive reset or checkpoint
/// restoration of an owning graph. They describe dispatch, not committed tokens
/// or performance. Counters saturate at [`u64::MAX`] and retain no input data.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MatvecStatistics {
    /// Activation preparations attempted, including unused prepared guards.
    pub preparations: u64,
    /// Preparations rejected before a matrix could be attempted.
    pub preparation_failures: u64,
    /// Matrix attempts using successfully prepared activations.
    pub matrix_attempts: u64,
    /// Matrix attempts returning complete successful output.
    pub matrix_successes: u64,
    /// Matrix attempts rejected before selecting a row backend.
    pub rejected_before_dispatch: u64,
    /// Matrix attempts dispatched to scalar, including later row failures.
    pub scalar_operations: u64,
    /// Matrix attempts dispatched to NEON, including later row failures.
    pub neon_operations: u64,
    /// Scalar dispatches explicitly requested by the caller.
    pub scalar_requested: u64,
    /// Scalar dispatches for tensor kinds without a NEON kernel.
    pub tensor_kind_unsupported: u64,
    /// Scalar dispatches because the NEON feature was disabled.
    pub feature_disabled: u64,
    /// Scalar dispatches because the build target was ineligible.
    pub target_unsupported: u64,
}

impl MatvecStatistics {
    const fn record(&mut self, diagnostics: MatvecDiagnostics) {
        self.matrix_attempts = self.matrix_attempts.saturating_add(1);
        if diagnostics.succeeded {
            self.matrix_successes = self.matrix_successes.saturating_add(1);
        }
        if !diagnostics.dispatched {
            self.rejected_before_dispatch = self.rejected_before_dispatch.saturating_add(1);
            return;
        }
        if matches!(
            diagnostics.selected_backend,
            MatvecBackend::ExperimentalNeon
        ) {
            self.neon_operations = self.neon_operations.saturating_add(1);
        } else {
            self.scalar_operations = self.scalar_operations.saturating_add(1);
        }
        match diagnostics.fallback_reason {
            Some(MatvecFallbackReason::ScalarRequested) => {
                self.scalar_requested = self.scalar_requested.saturating_add(1);
            }
            Some(MatvecFallbackReason::TensorKindUnsupported) => {
                self.tensor_kind_unsupported = self.tensor_kind_unsupported.saturating_add(1);
            }
            Some(MatvecFallbackReason::FeatureDisabled) => {
                self.feature_disabled = self.feature_disabled.saturating_add(1);
            }
            Some(MatvecFallbackReason::TargetUnsupported) => {
                self.target_unsupported = self.target_unsupported.saturating_add(1);
            }
            None => {}
        }
    }
}

type Pq2RowKernel = fn(&[u8], &[Q8KBlock]) -> Result<f32>;
type Q8GroupRowKernel = fn(u32, &[u8], &[Q8Block]) -> Result<f32>;

impl Default for MatvecScratch {
    fn default() -> Self {
        Self {
            q8k: Vec::new(),
            q8: Vec::new(),
            activations: Vec::new(),
            row: Vec::new(),
            diagnostics: MatvecDiagnostics::new(MatvecBackend::Auto),
            statistics: MatvecStatistics::default(),
            #[cfg(test)]
            preparations: 0,
        }
    }
}

impl MatvecScratch {
    /// Prepares one exact activation vector for compatible sibling projections.
    ///
    /// Quantization or floating activation rounding occurs once. Compatibility
    /// requires equal input width and activation precision/quantizer; in
    /// particular, `Q8_K` and `Q8_0` preparations cannot be interchanged. The
    /// caller must establish identical transform basis, signs and permutation.
    /// Scratch retains reusable format buffers and one decoded row, without
    /// a decoded matrix or per-sibling activation copies.
    ///
    /// # Errors
    /// Rejects malformed tensor dimensions, unsupported kinds, mismatched or
    /// nonfinite input, activation precision overflow, quantization errors, and
    /// failed scratch allocation.
    pub fn prepare<'a>(
        &'a mut self,
        tensor: &Tensor,
        input: &'a [f32],
    ) -> Result<PreparedMatvec<'a>> {
        self.reset_diagnostics();
        self.statistics.preparations = self.statistics.preparations.saturating_add(1);
        let contract = match self.prepare_input(tensor, input) {
            Ok(contract) => contract,
            Err(error) => {
                self.statistics.preparation_failures =
                    self.statistics.preparation_failures.saturating_add(1);
                return Err(error);
            }
        };
        Ok(PreparedMatvec {
            input,
            contract,
            scratch: self,
        })
    }

    fn prepare_input(&mut self, tensor: &Tensor, input: &[f32]) -> Result<ActivationContract> {
        let (columns, _, _) = dimensions(tensor)?;
        if input.len() != columns || input.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "matvec requires a matching finite activation vector",
            ));
        }
        let contract = ActivationContract::for_tensor(tensor, columns);
        match contract {
            ActivationContract::Q8K => q8k_into(input, &mut self.q8k)?,
            ActivationContract::Q8 => q8_into(input, &mut self.q8)?,
            _ => collect_into(
                input.iter().map(|&value| {
                    finite(match contract {
                        ActivationContract::F16 => round_f16(value),
                        ActivationContract::BF16 => round_bf16(value),
                        _ => value,
                    })
                }),
                &mut self.activations,
            )?,
        }
        #[cfg(test)]
        {
            self.preparations += 1;
        }
        Ok(contract)
    }

    #[cfg(test)]
    pub(crate) const fn preparation_count(&self) -> usize {
        self.preparations
    }

    /// Creates reusable storage with an explicit kernel request.
    ///
    /// Unsupported requests fall back to scalar at each matrix operation. The
    /// default constructor requests automatic selection. Disabling default
    /// features omits NEON unless `experimental-neon` is explicitly enabled.
    #[must_use]
    pub fn with_backend(backend: MatvecBackend) -> Self {
        Self {
            diagnostics: MatvecDiagnostics::new(backend),
            ..Self::default()
        }
    }

    /// Configured dispatch policy, preserved across preparation and failures.
    ///
    /// Individual operations may select scalar fallback. Use
    /// [`Self::selected_backend`] for the last operation's actual selection.
    #[must_use]
    pub const fn requested_backend(&self) -> MatvecBackend {
        self.diagnostics.requested_backend
    }

    /// Returns the backend selected for the most recent matrix operation.
    ///
    /// Initially scalar. Selection resets for every preparation and matrix
    /// attempt, including failures. See [`Self::diagnostics`] to distinguish a
    /// dispatched operation from failure before dispatch or successful completion.
    #[must_use]
    pub const fn selected_backend(&self) -> MatvecBackend {
        self.diagnostics.selected_backend
    }

    /// Returns selection, fallback, build eligibility and completion information.
    ///
    /// This describes only the latest preparation or matrix attempt. A failed
    /// attempt replaces earlier success; it is not a whole-model backend report.
    #[must_use]
    pub const fn diagnostics(&self) -> MatvecDiagnostics {
        self.diagnostics
    }

    /// Aggregate preparation, dispatch and failure counts for this scratch owner.
    ///
    /// Prepared siblings each count as a matrix attempt while sharing one
    /// preparation. Preparation failures are counted separately. Later failures
    /// never erase earlier dispatches; all counters include provisional work.
    #[must_use]
    pub const fn statistics(&self) -> MatvecStatistics {
        self.statistics
    }

    const fn reset_diagnostics(&mut self) {
        self.diagnostics = MatvecDiagnostics::new(self.requested_backend());
    }

    const fn select_pq2(&mut self) -> Pq2RowKernel {
        self.diagnostics = self.diagnostics.select(true);
        #[cfg(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc",
            target_feature = "neon"
        ))]
        if matches!(self.selected_backend(), MatvecBackend::ExperimentalNeon) {
            return dot_pq2_neon;
        }
        dot_pq2
    }

    const fn select_q8_groups(&mut self) -> Q8GroupRowKernel {
        self.diagnostics = self.diagnostics.select(true);
        #[cfg(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc",
            target_feature = "neon"
        ))]
        if matches!(self.selected_backend(), MatvecBackend::ExperimentalNeon) {
            return dot_q8_groups_neon;
        }
        dot_q8_groups
    }
}

/// Multiplies packed rows by one activation vector in the stored tensor basis.
///
/// Floating tensors use their native activation precision. PQ2 uses `Q8_K`
/// for rows divisible by 256, otherwise the pinned `Q8_0` fallback. PTQ1 and
/// the supported legacy Q kinds use `Q8_0`. Activations are quantized once per
/// call and reused for every row. No matrix-sized decoded copy exists.
///
/// # Errors
/// Rejects malformed tensors, unsupported kinds, nonfinite values, invalid PTQ1
/// encodings, input dimension mismatches, and incomplete activation groups.
pub fn matvec(tensor: &Tensor, input: &[f32]) -> Result<Vec<f32>> {
    let mut output = Vec::new();
    matvec_into(tensor, input, &mut output, &mut MatvecScratch::default())?;
    Ok(output)
}

/// Multiplies packed rows into reusable output and activation storage.
///
/// Numerical operations follow [`matvec`]. Existing output elements are
/// discarded, and the output is empty on error. Output and scratch capacities
/// are retained for subsequent calls; any required growth is fallible.
///
/// # Errors
/// Rejects the same invalid tensors and activations as [`matvec`], and returns
/// an error if output or scratch storage cannot be allocated.
pub fn matvec_into(
    tensor: &Tensor,
    input: &[f32],
    output: &mut Vec<f32>,
    scratch: &mut MatvecScratch,
) -> Result<()> {
    output.clear();
    scratch.prepare(tensor, input)?.matvec_into(tensor, output)
}

impl PreparedMatvec<'_> {
    /// Checks whether a tensor uses this preparation's width and activation contract.
    ///
    /// This cannot establish the caller's transform basis, signs or permutation.
    ///
    /// # Errors
    /// Rejects unsupported tensor kinds and malformed dimensions or byte extents.
    pub fn is_compatible(&self, tensor: &Tensor) -> Result<bool> {
        let (columns, _, _) = dimensions(tensor)?;
        Ok(columns == self.input.len()
            && ActivationContract::for_tensor(tensor, columns) == self.contract)
    }

    /// Multiplies compatible packed rows using the already prepared activation.
    ///
    /// Arithmetic and backend dispatch match [`matvec_into`]. Existing output
    /// elements are discarded and output is empty on error. All growth is
    /// fallible; this operation does not transform or quantize activations again.
    ///
    /// # Errors
    /// Rejects incompatible activation contracts, malformed tensors, nonfinite
    /// results, invalid PTQ1 encodings and failed output or row allocation.
    pub fn matvec_into(&mut self, tensor: &Tensor, output: &mut Vec<f32>) -> Result<()> {
        self.scratch.reset_diagnostics();
        let result = fill_into(output, |output| self.multiply(tensor, output));
        self.scratch.diagnostics.succeeded = result.is_ok();
        self.scratch.statistics.record(self.scratch.diagnostics);
        result
    }

    fn multiply(&mut self, tensor: &Tensor, output: &mut Vec<f32>) -> Result<()> {
        let (columns, rows, row_bytes) = dimensions(tensor)?;
        if columns != self.input.len()
            || ActivationContract::for_tensor(tensor, columns) != self.contract
        {
            return Err(invalid("incompatible prepared matvec activation contract"));
        }
        let scratch = &mut *self.scratch;
        output.try_reserve_exact(rows)?;
        match tensor.kind {
            16..=23 | 29 | 34 | 35 | 39..=42 => {
                scratch.diagnostics = scratch.diagnostics.select(false);
                for bytes in tensor.data.chunks_exact(row_bytes) {
                    output.push(dot_additional(
                        tensor.kind,
                        bytes,
                        &scratch.q8,
                        &scratch.q8k,
                    )?);
                }
            }
            10..=14 => {
                scratch.diagnostics = scratch.diagnostics.select(false);
                for bytes in tensor.data.chunks_exact(row_bytes) {
                    output.push(kquants::dot(tensor.kind, bytes, &scratch.q8k)?);
                }
            }
            PQ2 if columns.is_multiple_of(256) => {
                let dot = scratch.select_pq2();
                for bytes in tensor.data.chunks_exact(row_bytes) {
                    output.push(dot(bytes, &scratch.q8k)?);
                }
            }
            PQ2 | PTQ1 => {
                let dot = scratch.select_q8_groups();
                for bytes in tensor.data.chunks_exact(row_bytes) {
                    output.push(dot(tensor.kind, bytes, &scratch.q8)?);
                }
            }
            Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 => {
                scratch.diagnostics = scratch.diagnostics.select(false);
                for bytes in tensor.data.chunks_exact(row_bytes) {
                    output.push(dot_legacy(tensor.kind, bytes, &scratch.q8)?);
                }
            }
            _ => {
                scratch.diagnostics = scratch.diagnostics.select(false);
                for (index, _) in tensor.data.chunks_exact(row_bytes).enumerate() {
                    row_into(tensor, index, &mut scratch.row)?;
                    output.push(finite(float_dot(&scratch.row, &scratch.activations))?);
                }
            }
        }
        Ok(())
    }
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "The converted binary16 sign and magnitude occupy at most sixteen bits."
)]
pub(crate) fn half_bits(value: f32) -> u16 {
    // Port of ggml_compute_fp32_to_fp16; the shift/add path rounds ties to
    // even via native f32 arithmetic and includes subnormals and signed zero.
    let mut base = (value.abs() * f32::from_bits(0x7780_0000)) * f32::from_bits(0x0880_0000);
    let bits = value.to_bits();
    let doubled = bits.wrapping_add(bits);
    let exponent_bias = (doubled & 0xff00_0000).max(0x7100_0000);
    base += f32::from_bits((exponent_bias >> 1) + 0x0780_0000);
    let rounded = base.to_bits();
    let magnitude = if doubled > 0xff00_0000 {
        0x7e00
    } else {
        ((rounded >> 13) & 0x7c00) + (rounded & 0x0fff)
    };
    ((bits >> 16) & 0x8000 | magnitude) as u16
}

/// Rounds to IEEE binary16, returning its exactly expanded binary32 value.
/// Signed zero, infinities, subnormals, and ties-to-even are preserved.
#[must_use]
pub fn round_f16(value: f32) -> f32 {
    half_to_float(half_bits(value))
}

const fn round_bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    if value.is_nan() {
        f32::from_bits((bits & 0xffff_0000) | 0x0040_0000)
    } else {
        f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
    }
}

/// Applies the `SiLU` activation to one value.
#[must_use]
pub fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

/// Normalizes a nonempty vector using the ggml scalar RMS accumulation order.
///
/// # Errors
/// Rejects unequal or empty vectors, nonfinite inputs, negative epsilon, and
/// a normalization whose scale is not finite and positive.
pub fn rms_norm(input: &[f32], weight: &[f32], epsilon: f32) -> Result<Vec<f32>> {
    let mut output = Vec::new();
    rms_norm_into(input, weight, epsilon, &mut output)?;
    Ok(output)
}

/// Normalizes a vector into reusable caller-owned storage.
///
/// Numerical operations follow [`rms_norm`]. Existing elements are discarded,
/// capacity is retained, and the output is empty on every error.
///
/// # Errors
/// Rejects the same invalid vectors and epsilon as [`rms_norm`], and returns
/// an error if the output cannot be allocated.
pub fn rms_norm_into(
    input: &[f32],
    weight: &[f32],
    epsilon: f32,
    output: &mut Vec<f32>,
) -> Result<()> {
    fill_into(output, |output| {
        rms_norm_impl(input, weight, epsilon, output)
    })
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "The scalar normalization contract rounds the binary64 mean to binary32."
)]
fn rms_norm_impl(input: &[f32], weight: &[f32], epsilon: f32, output: &mut Vec<f32>) -> Result<()> {
    if input.is_empty() || input.len() != weight.len() || !epsilon.is_finite() || epsilon < 0.0 {
        return Err(invalid(
            "RMS normalization requires equal nonempty vectors and nonnegative epsilon",
        ));
    }
    let count = u32::try_from(input.len())?;
    let mut sum = 0.0_f64;
    for (&value, &w) in input.iter().zip(weight) {
        finite(value)?;
        finite(w)?;
        sum += f64::from(value * value);
    }
    let mean = (sum / f64::from(count)) as f32;
    let scale = 1.0 / (mean + epsilon).sqrt();
    if !scale.is_finite() || scale <= 0.0 {
        return Err(invalid("RMS normalization produced an invalid scale"));
    }
    collect_into(
        input
            .iter()
            .zip(weight)
            .map(|(&value, &w)| finite(value * scale * w)),
        output,
    )
}

/// Applies a normalized Sylvester Hadamard transform to one contiguous block.
///
/// # Errors
/// Rejects zero or non-power-of-two block lengths and nonfinite input/output.
pub fn hadamard(values: &mut [f32]) -> Result<()> {
    if !values.len().is_power_of_two() || values.iter().any(|value| !value.is_finite()) {
        return Err(invalid("Hadamard requires a finite power-of-two block"));
    }
    let count = u32::try_from(values.len())?;
    let mut half = 1;
    while half < values.len() {
        for block in values.chunks_exact_mut(half * 2) {
            for index in 0..half {
                let left = block[index];
                let right = block[index + half];
                block[index] = left + right;
                block[index + half] = left - right;
            }
        }
        half *= 2;
    }
    let normalization = 1.0 / integer_float(i32::try_from(count)?).sqrt();
    for value in values {
        *value = finite(*value * normalization)?;
    }
    Ok(())
}

/// Replaces finite logits by their normalized, stable exponential weights.
///
/// Negative infinity is accepted as an attention mask. All-masked rows fail.
///
/// # Errors
/// Rejects empty vectors, NaNs, positive infinity, and all-masked rows.
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "The scalar softmax contract rounds the binary64 inverse sum to binary32."
)]
pub fn softmax(values: &mut [f32]) -> Result<()> {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !maximum.is_finite()
        || values
            .iter()
            .any(|value| value.is_nan() || *value == f32::INFINITY)
    {
        return Err(invalid(
            "softmax requires at least one finite logit and no NaN or positive infinity",
        ));
    }
    let mut total = 0.0_f64;
    for value in &mut *values {
        *value = (*value - maximum).exp();
        total += f64::from(*value);
    }
    let inverse_sum = (1.0 / total) as f32;
    for value in values {
        *value *= inverse_sum;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{MatvecScratch, Tensor, matvec_into, q8_into, q8k_into};

    #[test]
    fn prepared_siblings_quantize_once_and_retry_with_fresh_contents() -> crate::Result<()> {
        let mut block = [0x55; 34];
        block[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        block[2] = 0xaa;
        let tensor = Tensor {
            name: "prepared".to_owned(),
            dimensions: vec![256, 2],
            kind: super::PQ2,
            data: Arc::from(block.repeat(4)),
        };
        let mut scratch = MatvecScratch::default();
        let mut output = Vec::new();
        let mut input = [127.0; 256];
        {
            let mut prepared = scratch.prepare(&tensor, &input)?;
            for _ in 0..4 {
                prepared.matvec_into(&tensor, &mut output)?;
                assert_eq!(prepared.scratch.preparation_count(), 1);
            }
        }
        let expected = output.clone();
        input.fill(-127.0);
        scratch
            .prepare(&tensor, &input)?
            .matvec_into(&tensor, &mut output)?;
        assert_eq!(scratch.preparation_count(), 2);
        assert!(
            expected
                .iter()
                .zip(output)
                .all(|(a, b)| (-a).to_bits() == b.to_bits())
        );
        Ok(())
    }

    #[test]
    fn matvec_scratch_retains_quantized_and_floating_storage() -> crate::Result<()> {
        let mut scratch = MatvecScratch::default();
        q8k_into(&[127.0; 512], &mut scratch.q8k)?;
        let wide_pointer = scratch.q8k.as_ptr();
        q8k_into(&[0.0; 256], &mut scratch.q8k)?;
        assert_eq!(scratch.q8k.len(), 1);
        assert_eq!(scratch.q8k.as_ptr(), wide_pointer);
        assert_eq!(scratch.q8k[0].values, [0; 256]);
        q8_into(&[127.0; 128], &mut scratch.q8)?;
        let legacy_pointer = scratch.q8.as_ptr();
        q8_into(&[0.0; 32], &mut scratch.q8)?;
        assert_eq!(scratch.q8.len(), 1);
        assert_eq!(scratch.q8.as_ptr(), legacy_pointer);
        assert_eq!(scratch.q8[0].values, [0; 32]);
        let mut tensor = Tensor {
            name: "scratch".to_owned(),
            dimensions: vec![2, 2],
            kind: super::F32,
            data: Arc::from(
                [1.0_f32, 2.0, 3.0, 4.0]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>(),
            ),
        };
        let mut output = Vec::new();
        matvec_into(&tensor, &[1.0; 2], &mut output, &mut scratch)?;
        let activation_pointer = scratch.activations.as_ptr();
        let row_pointer = scratch.row.as_ptr();
        tensor.dimensions = vec![1];
        tensor.data = Arc::from(2.0_f32.to_le_bytes().as_slice());
        matvec_into(&tensor, &[3.0], &mut output, &mut scratch)?;
        assert_eq!(scratch.activations.len(), 1);
        assert_eq!(scratch.row.len(), 1);
        assert_eq!(scratch.activations.as_ptr(), activation_pointer);
        assert_eq!(scratch.row.as_ptr(), row_pointer);
        assert_eq!(output, [6.0]);
        Ok(())
    }
}
