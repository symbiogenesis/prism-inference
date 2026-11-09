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

pub use bonsai_ternary::Q8KBlock;
use bonsai_ternary::{decode_ptq_trits, half_to_float};

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
mod parallel_rows;
pub use parallel_rows::{MAX_ROW_WORKERS, RowParallelism};
#[cfg(any(
    test,
    all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    )
))]
mod packed_pq2;
mod projection;
pub use projection::Pq2ProjectionExecutor;
pub(crate) mod pinned_activations;
pub(crate) mod pinned_attention;
pub(crate) mod pinned_convolution;
pub(crate) mod pinned_gdn;
pub(crate) mod prefill;
mod signquants;
mod tquants;
pub use tquants::Tl1BlockLut;

#[cfg(test)]
mod pinned_float_tests;

#[cfg(test)]
mod pinned_q8_tests;

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
pub(crate) const PQ2: u32 = 142;
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
                // Every packed byte has native fractional-trit semantics.
                // Extents were checked above; decoding and discarding 128
                // trits cannot reject anything beyond the finite scale check.
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
#[expect(
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
    q8k_with_rounding::<false>(input, output)
}

fn pinned_q8k_integer(value: f32, inverse_scale: f32) -> Result<i8> {
    // The pinned binary inlines nearest_int(iscale * value) and fuses the
    // product with its rounding magic. Rounding the product first can change
    // a code at a half-integer boundary and then every projection using it.
    let rounded = finite(inverse_scale.mul_add(value, 12_582_912.0))?;
    let integer = i32::try_from(rounded.to_bits() & 0x007f_ffff)? - 0x0040_0000;
    Ok(i8::try_from(integer.min(127))?)
}

fn q8k_with_rounding<const PINNED: bool>(input: &[f32], output: &mut Vec<Q8KBlock>) -> Result<()> {
    if !input.len().is_multiple_of(256) {
        return Err(invalid(
            "Q8_K activations require a multiple of 256 columns",
        ));
    }
    collect_into(
        input.as_chunks::<256>().0.iter().map(|chunk| {
            #[cfg(all(
                feature = "experimental-neon",
                target_arch = "aarch64",
                target_os = "windows",
                target_env = "msvc",
                target_feature = "neon"
            ))]
            if let Some(result) = neon::q8k_block::<PINNED>(chunk) {
                return result;
            }
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
                *target = if PINNED {
                    pinned_q8k_integer(value, inverse_scale)?
                } else {
                    // Preserve the public scalar source contract, including
                    // binary32 product rounding before ties-to-even rounding.
                    quantized((value * inverse_scale).round_ties_even().min(127.0))?
                };
            }
            for (sum, values) in block.sums.iter_mut().zip(block.values.as_chunks::<16>().0) {
                *sum = values.iter().map(|&value| i16::from(value)).sum();
            }
            Ok(block)
        }),
        output,
    )
}

/// One already prepared 32-element activation block.
///
/// The scale retains its exact original preparation rounding. External projection
/// executors consume these codes directly rather than quantizing a second time.
pub struct Q8Block {
    /// Original finite activation scale, including its zero sign.
    pub scale: f32,
    /// Original scaled code sum; unused by PQ2 and PTQ1 integer projections.
    pub sum: f32,
    /// Original signed quantization codes in source lane order.
    pub values: [i8; 32],
}

fn q8_into(input: &[f32], output: &mut Vec<Q8Block>) -> Result<()> {
    q8_with_rounding::<false>(input, output)
}

fn q8_with_rounding<const PINNED: bool>(input: &[f32], output: &mut Vec<Q8Block>) -> Result<()> {
    q8_with_contract::<PINNED, false>(input, output)
}

fn q8_with_contract<const PINNED: bool, const NATIVE_SCALE: bool>(
    input: &[f32],
    output: &mut Vec<Q8Block>,
) -> Result<()> {
    if !input.len().is_multiple_of(32) {
        return Err(invalid("Q8_0 activations require a multiple of 32 columns"));
    }
    collect_into(
        input.as_chunks::<32>().0.iter().map(|chunk| {
            #[cfg(all(
                feature = "experimental-neon",
                target_arch = "aarch64",
                target_os = "windows",
                target_env = "msvc",
                target_feature = "neon"
            ))]
            if PINNED && let Some(result) = neon::q8_pinned_block::<NATIVE_SCALE>(chunk) {
                return result;
            }
            let mut maximum = 0.0_f32;
            for &value in chunk {
                maximum = maximum.max(finite(value)?.abs());
            }
            // The pinned ARM64 Q8_0 binary multiplies by this rounded
            // reciprocal for the stored scale, but divides 127 by the
            // unscaled maximum for integer preparation. Recovering the
            // inverse from d changes halfway cases by one quantization code.
            let d = if NATIVE_SCALE {
                maximum * f32::from_bits(0x3c01_0204)
            } else {
                maximum / 127.0
            };
            let inverse_scale = if d == 0.0 {
                0.0
            } else if NATIVE_SCALE {
                127.0 / maximum
            } else {
                1.0 / d
            };
            let mut block = Q8Block {
                scale: round_f16(d),
                sum: 0.0,
                values: [0; 32],
            };
            finite(block.scale)?;
            for (target, &value) in block.values.iter_mut().zip(chunk) {
                let product = value * inverse_scale;
                // The pinned ARM64 Q8_0/Q8_1 quantizers use vcvtnq_s32_f32,
                // which rounds the binary32 product to nearest, ties to even.
                // The public source quantizer keeps roundf's ties away from zero.
                *target = quantized(if PINNED {
                    product.round_ties_even()
                } else {
                    product.round()
                })?;
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
#[expect(
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

#[expect(
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
#[expect(
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

#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Native row variance divides by a float-converted bounded element count."
)]
const fn bounded_count_float(count: u32) -> f32 {
    count as f32
}

#[expect(
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

#[expect(
    clippy::suboptimal_flops,
    reason = "Preserve scalar reference multiply-then-add rounding."
)]
fn dot_pq2<const FUSED: bool>(bytes: &[u8], input: &[Q8KBlock]) -> Result<f32> {
    let mut output = 0.0;
    for (index, block) in bytes.as_chunks::<34>().0.iter().enumerate() {
        let activation = &input[index / 2];
        let start = (index % 2) * 128;
        let d = scale(block)? * activation.scale;
        let sum = dot_integer(&pq2_values(block), &activation.values[start..start + 128]);
        output = if FUSED {
            d.mul_add(sum, output)
        } else {
            output + d * sum
        };
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
#[expect(
    clippy::suboptimal_flops,
    reason = "Preserve scalar reference multiply-then-add rounding."
)]
fn dot_pq2_neon<const FUSED: bool>(bytes: &[u8], input: &[Q8KBlock]) -> Result<f32> {
    let mut output = 0.0;
    for (index, block) in bytes.as_chunks::<34>().0.iter().enumerate() {
        let activation = &input[index / 2];
        let start = (index % 2) * 128;
        let d = scale(block)? * activation.scale;
        let sum = integer_float(neon::dot(
            block[2..].try_into()?,
            activation.values[start..start + 128].try_into()?,
        ));
        output = if FUSED {
            d.mul_add(sum, output)
        } else {
            output + d * sum
        };
    }
    finite(output)
}

/// Computes a direct `PTQ1` dot product against `Q8_K` activation blocks.
///
/// Avoids a separate `Q8_0` activation quantization pass when 256-element
/// `Q8_K` blocks are already prepared.
///
/// # Errors
/// Rejects malformed or mismatched `PTQ1`/`Q8_K` extents, invalid ternary bytes,
/// or non-finite scales.
#[expect(
    clippy::suboptimal_flops,
    reason = "Preserve scalar reference multiply-then-add rounding."
)]
pub fn dot_ptq1_q8k(bytes: &[u8], input: &[Q8KBlock]) -> Result<f32> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(28) {
        return Err(invalid("PTQ1 packed dot requires complete 28-byte blocks"));
    }
    let ptq1_blocks = bytes.len() / 28;
    if ptq1_blocks.div_ceil(2) != input.len() {
        return Err(invalid("PTQ1 and Q8_K packed dot extents differ"));
    }
    let mut output = 0.0_f32;
    for (index, block) in bytes.as_chunks::<28>().0.iter().enumerate() {
        let activation = &input[index / 2];
        let start = (index % 2) * 128;
        let d = scale(&block[26..])? * finite(activation.scale)?;
        #[cfg(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc",
            target_feature = "neon"
        ))]
        let sum = {
            let act_half = &activation.values[start..start + 128];
            let groups = act_half.as_chunks::<32>().0;
            let group_sums =
                neon::dot_ptq1_groups(block, [&groups[0], &groups[1], &groups[2], &groups[3]])?;
            integer_float(group_sums.into_iter().sum())
        };
        #[cfg(not(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc",
            target_feature = "neon"
        )))]
        let sum = dot_integer(&ptq1_values(block)?, &activation.values[start..start + 128]);
        output += d * sum;
    }
    finite(output)
}

#[expect(
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

fn q8_native_lanes(weights: &[i16], activations: &[i8; 32]) -> [i32; 4] {
    // The pinned DOTPROD binary keeps four adjacent-product groups from each
    // 16-wide half, rather than reducing the complete Q8_0 group to one integer.
    std::array::from_fn(|lane| {
        [lane * 4, lane * 4 + 16]
            .into_iter()
            .flat_map(|start| start..start + 4)
            .map(|index| i32::from(weights[index]) * i32::from(activations[index]))
            .sum()
    })
}

fn q8_group_native_lanes<const NEON: bool>(
    block: &[u8],
    activations: &[Q8Block; 4],
) -> Result<[[i32; 4]; 4]> {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if NEON {
        let values = activations.each_ref().map(|activation| &activation.values);
        return neon::dot_ptq1_native_lanes(block.try_into()?, values);
    }
    let weights = ptq1_values(block)?;
    Ok(std::array::from_fn(|group| {
        q8_native_lanes(
            &weights[group * 32..group * 32 + 32],
            &activations[group].values,
        )
    }))
}

fn dot_q8_groups_pinned<const NEON: bool>(
    kind: u32,
    bytes: &[u8],
    input: &[Q8Block],
) -> Result<f32> {
    // Only PTQ1 has independent pinned-binary captures for this Q8_0 contract.
    // Preserve the existing PQ2 odd-128 fallback pending separate validation.
    if kind != PTQ1 {
        #[cfg(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc",
            target_feature = "neon"
        ))]
        if NEON {
            return dot_q8_groups_neon(kind, bytes, input);
        }
        return dot_q8_groups(kind, bytes, input);
    }
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if NEON {
        return neon::dot_ptq1_row_pinned(bytes, input);
    }
    let (_, block_bytes) = layout(kind)?;
    let mut lanes = [0.0_f32; 4];
    for (index, block) in bytes.chunks_exact(block_bytes).enumerate() {
        let activations = <&[Q8Block; 4]>::try_from(&input[index * 4..index * 4 + 4])?;
        let groups = q8_group_native_lanes::<NEON>(block, activations)?;
        let weight_scale = scale(&block[26..])?;
        for (integers, activation) in groups.into_iter().zip(activations) {
            let d = weight_scale * activation.scale;
            for (lane, integer) in lanes.iter_mut().zip(integers) {
                *lane = integer_float(integer).mul_add(d, *lane);
            }
        }
    }
    // ARM64 FADDP reduces adjacent lanes in pairs after the entire row.
    finite((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
#[expect(
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

#[expect(
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
        #[cfg(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc",
            target_feature = "neon"
        ))]
        let integer = if kind == Q4_0 {
            integer_float(neon::dot_q4_0_block(
                block[2..18].try_into()?,
                &activation.values,
            ))
        } else {
            integer_float(neon::dot_q8_0_block(
                block[2..34].try_into()?,
                &activation.values,
            ))
        };
        #[cfg(not(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc",
            target_feature = "neon"
        )))]
        let integer = {
            let values: [i16; 32] = std::array::from_fn(|index| {
                if kind == Q4_0 {
                    i16::from((block[2 + index % 16] >> (4 * (index / 16))) & 15) - 8
                } else {
                    i16::from(i8::from_ne_bytes([block[2 + index]]))
                }
            });
            dot_integer(&values, &activation.values)
        };
        output += (scale(block)? * activation.scale) * integer;
    }
    finite(output)
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "The scalar reference rounds the accumulated binary64 sum to binary32."
)]
pub(crate) fn float_dot(weights: &[f32], input: &[f32]) -> f32 {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(value) = neon::float_dot(weights, input) {
        return value;
    }
    // The non-SIMD ggml reference sums rounded f32 products in f64.
    weights
        .iter()
        .zip(input)
        .map(|(&a, &b)| f64::from(a * b))
        .sum::<f64>() as f32
}

pub(crate) fn float_dot_pair(state: &[f32], first: &[f32], second: &[f32]) -> (f32, f32) {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(pair) = neon::float_dot_pair(state, first, second) {
        return pair;
    }
    (float_dot(state, first), float_dot(state, second))
}

pub(crate) fn float_dot_pair_four_rows(
    states: [&[f32]; 4],
    first: &[f32],
    second: &[f32],
) -> [(f32, f32); 4] {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(pairs) = neon::float_dot_pair_four_rows(states, first, second) {
        return pairs;
    }
    [
        float_dot_pair(states[0], first, second),
        float_dot_pair(states[1], first, second),
        float_dot_pair(states[2], first, second),
        float_dot_pair(states[3], first, second),
    ]
}

pub(crate) fn float_dot_pair_eight_rows(
    states: [&[f32]; 8],
    first: &[f32],
    second: &[f32],
) -> [(f32, f32); 8] {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(pairs) = neon::float_dot_pair_eight_rows(states, first, second) {
        return pairs;
    }
    [
        float_dot_pair(states[0], first, second),
        float_dot_pair(states[1], first, second),
        float_dot_pair(states[2], first, second),
        float_dot_pair(states[3], first, second),
        float_dot_pair(states[4], first, second),
        float_dot_pair(states[5], first, second),
        float_dot_pair(states[6], first, second),
        float_dot_pair(states[7], first, second),
    ]
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Preserve binary32 products and binary64 accumulation when reading the binary16 cache."
)]
pub(crate) fn f16_dot(first: &[f32], second: &[u16]) -> f32 {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(value) = neon::f16_dot(first, second) {
        return value;
    }
    first
        .iter()
        .zip(second)
        .map(|(&first, &second)| f64::from(first * half_to_float(second)))
        .sum::<f64>() as f32
}

pub(crate) fn float_dot_four(queries: [&[f32]; 4], key: &[f32]) -> [f32; 4] {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(values) = neon::float_dot_four(queries, key) {
        return values;
    }
    queries.map(|query| float_dot(query, key))
}

pub(crate) fn f16_dot_four(queries: [&[f32]; 4], key: &[u16]) -> [f32; 4] {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(values) = neon::f16_dot_four(queries, key) {
        return values;
    }
    queries.map(|query| f16_dot(query, key))
}

pub(crate) fn f16_dot_two_token_four_queries(
    queries0: [&[f32]; 4],
    queries1: [&[f32]; 4],
    key: &[u16],
) -> ([f32; 4], [f32; 4]) {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(values) = neon::f16_dot_two_token_four_queries(queries0, queries1, key) {
        return values;
    }
    (f16_dot_four(queries0, key), f16_dot_four(queries1, key))
}

pub(crate) fn f16_dot_two_keys_four_queries(
    queries: [&[f32]; 4],
    keys: [&[u16]; 2],
) -> [[f32; 4]; 2] {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(values) = neon::f16_dot_two_keys_four_queries(queries, keys) {
        return values;
    }
    keys.map(|key| f16_dot_four(queries, key))
}

pub(crate) fn f16_dot_two_keys_two_token_four_queries(
    queries0: [&[f32]; 4],
    queries1: [&[f32]; 4],
    keys: [&[u16]; 2],
) -> (
    pinned_attention::TwoKeyFourQueryScores,
    pinned_attention::TwoKeyFourQueryScores,
) {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(values) = neon::f16_dot_two_keys_two_token_four_queries(queries0, queries1, keys) {
        return values;
    }
    let (k0_0, k0_1) = f16_dot_two_token_four_queries(queries0, queries1, keys[0]);
    let (k1_0, k1_1) = f16_dot_two_token_four_queries(queries0, queries1, keys[1]);
    ([k0_0, k1_0], [k0_1, k1_1])
}

pub(crate) fn f16_dot_four_keys_four_queries(
    queries: [&[f32]; 4],
    keys: [&[u16]; 4],
) -> [[f32; 4]; 4] {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(values) = neon::f16_dot_four_keys_four_queries(queries, keys) {
        return values;
    }
    keys.map(|key| f16_dot_four(queries, key))
}

pub(crate) fn f16_dot_four_keys_two_token_four_queries(
    queries0: [&[f32]; 4],
    queries1: [&[f32]; 4],
    keys: [&[u16]; 4],
) -> (
    pinned_attention::FourKeyFourQueryScores,
    pinned_attention::FourKeyFourQueryScores,
) {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(values) = neon::f16_dot_four_keys_two_token_four_queries(queries0, queries1, keys) {
        return values;
    }
    let (k0_0, k0_1) = f16_dot_two_token_four_queries(queries0, queries1, keys[0]);
    let (k1_0, k1_1) = f16_dot_two_token_four_queries(queries0, queries1, keys[1]);
    let (k2_0, k2_1) = f16_dot_two_token_four_queries(queries0, queries1, keys[2]);
    let (k3_0, k3_1) = f16_dot_two_token_four_queries(queries0, queries1, keys[3]);
    ([k0_0, k1_0, k2_0, k3_0], [k0_1, k1_1, k2_1, k3_1])
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Retain the scalar floating dot's rounded F32 products, binary64 sum and final F32 rounding."
)]
fn bf16_dot(bytes: &[u8], input: &[f32]) -> Result<f32> {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(result) = neon::bf16_dot(bytes, input) {
        return result;
    }
    // The caller checks the complete tensor geometry before borrowing this
    // row. Consume each BF16 scalar once instead of materializing a F32 row.
    // Result's sum delegates to the same f64 Sum used by float_dot, including
    // its reduction order and signed-zero identity. Every weight is still
    // checked, including weights multiplied by zero activations.
    let sum = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .zip(input)
        .map(|(bytes, &activation)| {
            finite(f32::from_bits(u32::from(u16::from_le_bytes(*bytes)) << 16))
                .map(|weight| f64::from(weight * activation))
        })
        .sum::<Result<f64>>()?;
    finite(sum as f32)
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
                dot_pq2::<false>(weights, &input)
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
/// Eligible PQ2 NEON dispatch additionally retains one input-width byte layout
/// and one f32 scale per 128 columns, shared by rows and prepared siblings.
pub struct MatvecScratch {
    q8k: Vec<Q8KBlock>,
    #[cfg(any(
        test,
        all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc",
            target_feature = "neon"
        )
    ))]
    pq2_consumption: Vec<packed_pq2::Activation>,
    q8: Vec<Q8Block>,
    activations: Vec<f32>,
    row: Vec<f32>,
    diagnostics: MatvecDiagnostics,
    statistics: MatvecStatistics,
    pinned_accumulation: bool,
    row_parallelism: RowParallelism,
    row_errors: Vec<Option<crate::Error>>,
    #[cfg(test)]
    preparations: usize,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum ActivationContract {
    Q8K,
    Q8,
    PinnedPtq1Q8,
    F32,
    F16,
    BF16,
}

impl ActivationContract {
    const fn for_tensor(tensor: &Tensor, columns: usize, pinned: bool) -> Self {
        match tensor.kind {
            PTQ1 if pinned => Self::PinnedPtq1Q8,
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
    quantized: bool,
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
    /// Packed matrix attempts using the explicit persistent row-worker pool.
    pub parallel_row_operations: u64,
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
            #[cfg(any(
                test,
                all(
                    feature = "experimental-neon",
                    target_arch = "aarch64",
                    target_os = "windows",
                    target_env = "msvc",
                    target_feature = "neon"
                )
            ))]
            pq2_consumption: Vec::new(),
            q8: Vec::new(),
            activations: Vec::new(),
            row: Vec::new(),
            diagnostics: MatvecDiagnostics::new(MatvecBackend::Auto),
            statistics: MatvecStatistics::default(),
            pinned_accumulation: false,
            row_parallelism: RowParallelism::default(),
            row_errors: Vec::new(),
            #[cfg(test)]
            preparations: 0,
        }
    }
}

impl MatvecScratch {
    /// Changes only row scheduling, preserving prepared arithmetic and storage.
    /// Existing callers remain serial until explicitly configured.
    pub fn set_row_parallelism(&mut self, parallelism: RowParallelism) {
        self.row_errors.clear();
        self.row_parallelism = parallelism;
    }

    /// Explicit persistent row-worker budget; small/floating operations stay serial.
    #[must_use]
    pub const fn row_workers(&self) -> usize {
        self.row_parallelism.workers()
    }

    /// Share the existing private pool with another scoped projection consumer.
    pub(crate) fn clone_row_parallelism(&self) -> RowParallelism {
        self.row_parallelism.clone()
    }

    pub(crate) const fn row_parallelism(&self) -> &RowParallelism {
        &self.row_parallelism
    }

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
        self.prepare_with_quantization(tensor, input, true)
    }

    /// Prepares one exact `F32` activation vector for an external `F32`
    /// projection executor without CPU `Q8_K` or `Q8_0` quantization.
    ///
    /// # Errors
    /// Rejects malformed tensor dimensions, unsupported kinds, or mismatched
    /// or nonfinite input.
    pub(crate) fn prepare_f32_executor<'a>(
        &'a mut self,
        tensor: &Tensor,
        input: &'a [f32],
    ) -> Result<PreparedMatvec<'a>> {
        self.prepare_with_quantization(tensor, input, false)
    }

    pub(crate) fn shares_contract(&self, first: &Tensor, second: &Tensor) -> Result<bool> {
        let (first_columns, _, _) = dimensions(first)?;
        let (second_columns, _, _) = dimensions(second)?;
        Ok(first_columns == second_columns
            && ActivationContract::for_tensor(first, first_columns, self.pinned_accumulation)
                == ActivationContract::for_tensor(second, second_columns, self.pinned_accumulation))
    }

    fn prepare_with_quantization<'a>(
        &'a mut self,
        tensor: &Tensor,
        input: &'a [f32],
        quantized: bool,
    ) -> Result<PreparedMatvec<'a>> {
        self.reset_diagnostics();
        self.statistics.preparations = self.statistics.preparations.saturating_add(1);
        let contract = match self.prepare_input_with_quantization(tensor, input, quantized) {
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
            quantized,
            scratch: self,
        })
    }

    fn prepare_input(&mut self, tensor: &Tensor, input: &[f32]) -> Result<ActivationContract> {
        self.prepare_input_with_quantization(tensor, input, true)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Keep activation contract dispatch and NEON rounding fallbacks together."
    )]
    fn prepare_input_with_quantization(
        &mut self,
        tensor: &Tensor,
        input: &[f32],
        quantized: bool,
    ) -> Result<ActivationContract> {
        // A new scoped preparation invalidates every rearranged activation,
        // including a preparation which fails before dispatch. Capacity alone
        // is retained; neither pointers nor prior input contents identify reuse.
        #[cfg(any(
            test,
            all(
                feature = "experimental-neon",
                target_arch = "aarch64",
                target_os = "windows",
                target_env = "msvc",
                target_feature = "neon"
            )
        ))]
        self.pq2_consumption.clear();
        let (columns, _, _) = dimensions(tensor)?;
        if input.len() != columns || !is_finite_slice(input) {
            return Err(invalid(
                "matvec requires a matching finite activation vector",
            ));
        }
        let contract = ActivationContract::for_tensor(tensor, columns, self.pinned_accumulation);
        if quantized {
            match contract {
                ActivationContract::PinnedPtq1Q8 => {
                    q8_with_contract::<true, true>(input, &mut self.q8)?;
                }
                ActivationContract::Q8K if self.pinned_accumulation => {
                    q8k_with_rounding::<true>(input, &mut self.q8k)?;
                }
                ActivationContract::Q8K => q8k_into(input, &mut self.q8k)?,
                ActivationContract::Q8 if self.pinned_accumulation => {
                    q8_with_rounding::<true>(input, &mut self.q8)?;
                }
                ActivationContract::Q8 => q8_into(input, &mut self.q8)?,
                ActivationContract::F32 => {
                    if self.activations.len() == input.len() {
                        self.activations.copy_from_slice(input);
                    } else {
                        fill_into(&mut self.activations, |activations| {
                            activations.try_reserve_exact(input.len())?;
                            activations.extend_from_slice(input);
                            Ok(())
                        })?;
                    }
                }
                ActivationContract::F16 => {
                    #[cfg(all(
                        feature = "experimental-neon",
                        target_arch = "aarch64",
                        target_os = "windows",
                        target_env = "msvc",
                        target_feature = "neon"
                    ))]
                    if let Some(result) = neon::round_f16_into(input, &mut self.activations) {
                        if result.is_err() {
                            self.activations.clear();
                        }
                        result?;
                    } else {
                        collect_into(
                            input.iter().map(|&value| finite(round_f16(value))),
                            &mut self.activations,
                        )?;
                    }
                    #[cfg(not(all(
                        feature = "experimental-neon",
                        target_arch = "aarch64",
                        target_os = "windows",
                        target_env = "msvc",
                        target_feature = "neon"
                    )))]
                    collect_into(
                        input.iter().map(|&value| finite(round_f16(value))),
                        &mut self.activations,
                    )?;
                }
                ActivationContract::BF16 => {
                    #[cfg(all(
                        feature = "experimental-neon",
                        target_arch = "aarch64",
                        target_os = "windows",
                        target_env = "msvc",
                        target_feature = "neon"
                    ))]
                    if let Some(result) = neon::round_bf16_into(input, &mut self.activations) {
                        if result.is_err() {
                            self.activations.clear();
                        }
                        result?;
                    } else {
                        collect_into(
                            input.iter().map(|&value| finite(round_bf16(value))),
                            &mut self.activations,
                        )?;
                    }
                    #[cfg(not(all(
                        feature = "experimental-neon",
                        target_arch = "aarch64",
                        target_os = "windows",
                        target_env = "msvc",
                        target_feature = "neon"
                    )))]
                    collect_into(
                        input.iter().map(|&value| finite(round_bf16(value))),
                        &mut self.activations,
                    )?;
                }
            }
        } else {
            self.q8k.clear();
            self.q8.clear();
            self.activations.clear();
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

    /// Selects the pinned Windows ARM64 binary's graph arithmetic contract.
    pub(crate) fn with_pinned_backend(backend: MatvecBackend) -> Self {
        Self {
            pinned_accumulation: true,
            ..Self::with_backend(backend)
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
            return if self.pinned_accumulation {
                dot_pq2_neon::<true>
            } else {
                dot_pq2_neon::<false>
            };
        }
        if self.pinned_accumulation {
            dot_pq2::<true>
        } else {
            dot_pq2::<false>
        }
    }

    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    fn fill_pq2_consumption(
        &mut self,
        data: &[u8],
        row_bytes: usize,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        // Share one byte-per-column consumption layout across all rows and
        // compatible sibling projections in this scope. Original Q8_K blocks
        // stay untouched for other consumers.
        if self.pq2_consumption.is_empty() {
            packed_pq2::prepare(&self.q8k, &mut self.pq2_consumption)?;
        }
        let pinned = self.pinned_accumulation;
        let activations = &self.pq2_consumption;
        self.row_parallelism.fill_four_rows(
            data,
            row_bytes,
            output,
            &mut self.row_errors,
            |rows| {
                if pinned {
                    packed_pq2::dot_eight_rows::<true>(rows, activations)
                } else {
                    packed_pq2::dot_eight_rows::<false>(rows, activations)
                }
            },
            |rows| {
                if pinned {
                    packed_pq2::dot_four_rows::<true>(rows, activations)
                } else {
                    packed_pq2::dot_four_rows::<false>(rows, activations)
                }
            },
            |bytes| {
                if pinned {
                    packed_pq2::dot_row::<true>(bytes, activations)
                } else {
                    packed_pq2::dot_row::<false>(bytes, activations)
                }
            },
        )
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
            return if self.pinned_accumulation {
                dot_q8_groups_pinned::<true>
            } else {
                dot_q8_groups_neon
            };
        }
        if self.pinned_accumulation {
            dot_q8_groups_pinned::<false>
        } else {
            dot_q8_groups
        }
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
    let result = scratch
        .prepare(tensor, input)
        .and_then(|mut prepared| prepared.matvec_into(tensor, output));
    if result.is_err() {
        output.clear();
    }
    result
}

pub(crate) fn lora_a_dot(
    a_values: &[f32],
    input: &[f32],
    output: &mut Vec<f32>,
    scratch: &MatvecScratch,
) -> Result<()> {
    fill_into(output, |output| {
        if a_values.is_empty() || input.len() != a_values.len() || !is_finite_slice(input) {
            return Err(invalid(
                "matvec requires a matching finite activation vector",
            ));
        }
        output.try_reserve_exact(1)?;
        let value = if scratch.pinned_accumulation && a_values.len().is_multiple_of(16) {
            pinned_gdn::dot(a_values, input)?
        } else {
            finite(float_dot(a_values, input))?
        };
        output.push(value);
        Ok(())
    })
}

pub(crate) fn lora_a_dot_pair(
    a_values: &[f32],
    input0: &[f32],
    input1: &[f32],
    scratch: &MatvecScratch,
) -> Option<(f32, f32)> {
    if a_values.is_empty()
        || input0.len() != a_values.len()
        || input1.len() != a_values.len()
        || !is_finite_slice(input0)
        || !is_finite_slice(input1)
    {
        return None;
    }
    if scratch.pinned_accumulation && a_values.len().is_multiple_of(16) {
        pinned_gdn::dot_pair(a_values, input0, input1).ok()
    } else {
        let (v0, v1) = float_dot_pair(a_values, input0, input1);
        if v0.is_finite() && v1.is_finite() {
            Some((v0, v1))
        } else {
            None
        }
    }
}

pub(crate) fn lora_a_dot_four(
    a_values: &[f32],
    inputs: [&[f32]; 4],
    scratch: &MatvecScratch,
) -> Option<[f32; 4]> {
    let width = a_values.len();
    if width == 0
        || inputs
            .iter()
            .any(|input| input.len() != width || !is_finite_slice(input))
    {
        return None;
    }
    if scratch.pinned_accumulation && width.is_multiple_of(16) {
        pinned_gdn::dot_four(a_values, inputs).ok()
    } else {
        let values = float_dot_four(inputs, a_values);
        if values.iter().all(|v| v.is_finite()) {
            Some(values)
        } else {
            None
        }
    }
}

pub(crate) fn lora_b_scale_and_finish_tokens<const TOKENS: usize>(
    b_values: &[f32],
    ranks: [f32; TOKENS],
    scale: f32,
    outputs: [&mut [f32]; TOKENS],
    correction: &mut Vec<f32>,
) -> bool {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    {
        neon::lora_b_scale_and_finish_tokens(b_values, ranks, scale, outputs, correction).is_some()
    }
    #[cfg(not(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    )))]
    {
        let _ = (b_values, ranks, scale, outputs, correction);
        false
    }
}

pub(crate) fn lora_b_scale_and_finish(
    b_values: &[f32],
    rank: &[f32],
    scale: f32,
    output: &mut [f32],
    correction: &mut Vec<f32>,
) -> Result<()> {
    let [rank_val] = *rank else {
        correction.clear();
        return Err(invalid(
            "matvec requires a matching finite activation vector",
        ));
    };
    if b_values.is_empty() || !rank_val.is_finite() {
        correction.clear();
        return Err(invalid(
            "matvec requires a matching finite activation vector",
        ));
    }
    if output.len() == b_values.len() {
        correction.try_reserve_exact(b_values.len().saturating_sub(correction.len()))?;
        #[cfg(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc",
            target_feature = "neon"
        ))]
        if let Some(result) =
            neon::lora_b_scale_and_finish(b_values, rank_val, scale, output, correction)
        {
            return result;
        }
    }
    lora_b_scale(b_values, rank, correction)?;
    lora_finish_in_place(scale, output, correction)
}

pub(crate) fn lora_b_scale(
    b_values: &[f32],
    rank: &[f32],
    correction: &mut Vec<f32>,
) -> Result<()> {
    fill_into(correction, |correction| {
        let [rank] = *rank else {
            return Err(invalid(
                "matvec requires a matching finite activation vector",
            ));
        };
        if b_values.is_empty() || !rank.is_finite() {
            return Err(invalid(
                "matvec requires a matching finite activation vector",
            ));
        }
        correction.try_reserve_exact(b_values.len())?;
        #[cfg(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc",
            target_feature = "neon"
        ))]
        if let Some(result) = neon::lora_b_scale(b_values, rank, correction) {
            return result;
        }
        for &weight in b_values {
            let weight = finite(weight)?;
            correction.push(finite(float_dot(&[weight], &[rank]))?);
        }
        Ok(())
    })
}

pub(crate) fn lora_finish_in_place(
    scale: f32,
    output: &mut [f32],
    correction: &mut [f32],
) -> Result<()> {
    if output.len() != correction.len() {
        return Err(invalid("rank-one correction extent mismatch"));
    }
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(result) = neon::lora_finish_in_place(scale, output, correction) {
        return result;
    }
    for (&base, corr) in output.iter().zip(correction.iter_mut()) {
        *corr *= scale;
        *corr += base;
        if !corr.is_finite() {
            return Err(invalid("nonfinite rank-one correction"));
        }
    }
    output.copy_from_slice(correction);
    Ok(())
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
            && ActivationContract::for_tensor(tensor, columns, self.scratch.pinned_accumulation)
                == self.contract)
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
        let result = self.multiply(tensor, output);
        if result.is_err() {
            output.clear();
        }
        self.scratch.diagnostics.succeeded = result.is_ok();
        self.scratch.statistics.record(self.scratch.diagnostics);
        result
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Keep checked format dispatch and its ordered arithmetic branches together."
    )]
    fn multiply(&mut self, tensor: &Tensor, output: &mut Vec<f32>) -> Result<()> {
        let (columns, rows, row_bytes) = dimensions(tensor)?;
        if !self.quantized
            || columns != self.input.len()
            || ActivationContract::for_tensor(tensor, columns, self.scratch.pinned_accumulation)
                != self.contract
        {
            return Err(invalid("incompatible prepared matvec activation contract"));
        }
        let scratch = &mut *self.scratch;
        match tensor.kind {
            F32 if columns == 1 => {
                scratch.diagnostics = scratch.diagnostics.select(false);
                output.clear();
                output.try_reserve_exact(rows)?;
                // Rank-one adapter B factors have one scalar per output row.
                // Geometry is already checked above: stream those scalars
                // instead of validating and preparing a row Vec for each one.
                // Keep the exact generic dot, including its binary64 reduction
                // and signed-zero behavior. The pinned FMA path requires 16
                // columns, so it never applies to these one-column matrices.
                for bytes in tensor.data.as_chunks::<4>().0 {
                    let weight = finite(f32::from_le_bytes(*bytes))?;
                    output.push(finite(float_dot(&[weight], &scratch.activations))?);
                }
            }
            BF16 => {
                scratch.diagnostics = scratch.diagnostics.select(false);
                // Swift's recurrent alpha/beta projections retain their
                // binary64 scalar contract even for pinned accumulation.
                #[cfg(all(
                    feature = "experimental-neon",
                    target_arch = "aarch64",
                    target_os = "windows",
                    target_env = "msvc",
                    target_feature = "neon"
                ))]
                {
                    let activations = &scratch.activations;
                    return scratch.row_parallelism.fill_four_rows(
                        &tensor.data,
                        row_bytes,
                        output,
                        &mut scratch.row_errors,
                        |rows| {
                            neon::bf16_dot_eight_rows(rows, activations)
                                .ok_or_else(|| invalid("missing NEON BF16 kernel"))?
                        },
                        |rows| {
                            neon::bf16_dot_four_rows(rows, activations)
                                .ok_or_else(|| invalid("missing NEON BF16 kernel"))?
                        },
                        |bytes| bf16_dot(bytes, activations),
                    );
                }
                #[cfg(not(all(
                    feature = "experimental-neon",
                    target_arch = "aarch64",
                    target_os = "windows",
                    target_env = "msvc",
                    target_feature = "neon"
                )))]
                {
                    let activations = &scratch.activations;
                    scratch.row_parallelism.fill(
                        &tensor.data,
                        row_bytes,
                        output,
                        &mut scratch.row_errors,
                        |bytes| bf16_dot(bytes, activations),
                    )?;
                }
            }
            16..=23 | 29 | 34 | 35 | 39..=42 => {
                scratch.diagnostics = scratch.diagnostics.select(false);
                let q8 = &scratch.q8;
                let q8k = &scratch.q8k;
                scratch.row_parallelism.fill(
                    &tensor.data,
                    row_bytes,
                    output,
                    &mut scratch.row_errors,
                    |bytes| dot_additional(tensor.kind, bytes, q8, q8k),
                )?;
            }
            10..=14 => {
                scratch.diagnostics = scratch.diagnostics.select(false);
                let q8k = &scratch.q8k;
                scratch.row_parallelism.fill(
                    &tensor.data,
                    row_bytes,
                    output,
                    &mut scratch.row_errors,
                    |bytes| kquants::dot(tensor.kind, bytes, q8k),
                )?;
            }
            PQ2 if columns.is_multiple_of(256) => {
                let dot = scratch.select_pq2();
                if scratch.row_parallelism.uses_pool(rows) {
                    scratch.statistics.parallel_row_operations =
                        scratch.statistics.parallel_row_operations.saturating_add(1);
                }
                #[cfg(all(
                    feature = "experimental-neon",
                    target_arch = "aarch64",
                    target_os = "windows",
                    target_env = "msvc",
                    target_feature = "neon"
                ))]
                if matches!(scratch.selected_backend(), MatvecBackend::ExperimentalNeon) {
                    return scratch.fill_pq2_consumption(&tensor.data, row_bytes, output);
                }
                scratch.row_parallelism.fill(
                    &tensor.data,
                    row_bytes,
                    output,
                    &mut scratch.row_errors,
                    |bytes| dot(bytes, &scratch.q8k),
                )?;
            }
            PQ2 | PTQ1 => {
                let dot = scratch.select_q8_groups();
                if scratch.row_parallelism.uses_pool(rows) {
                    scratch.statistics.parallel_row_operations =
                        scratch.statistics.parallel_row_operations.saturating_add(1);
                }
                #[cfg(all(
                    feature = "experimental-neon",
                    target_arch = "aarch64",
                    target_os = "windows",
                    target_env = "msvc",
                    target_feature = "neon"
                ))]
                if tensor.kind == PTQ1
                    && matches!(scratch.selected_backend(), MatvecBackend::ExperimentalNeon)
                {
                    let q8 = &scratch.q8;
                    if scratch.pinned_accumulation {
                        return scratch.row_parallelism.fill_four_rows(
                            &tensor.data,
                            row_bytes,
                            output,
                            &mut scratch.row_errors,
                            |rows| neon::dot_ptq1_eight_rows_pinned(rows, q8),
                            |rows| neon::dot_ptq1_four_rows_pinned(rows, q8),
                            |bytes| neon::dot_ptq1_row_pinned(bytes, q8),
                        );
                    }
                    return scratch.row_parallelism.fill_four_rows(
                        &tensor.data,
                        row_bytes,
                        output,
                        &mut scratch.row_errors,
                        |rows| neon::dot_ptq1_eight_rows_source(rows, q8),
                        |rows| neon::dot_ptq1_four_rows_source(rows, q8),
                        |bytes| neon::dot_ptq1_row_source(bytes, q8),
                    );
                }
                scratch.row_parallelism.fill(
                    &tensor.data,
                    row_bytes,
                    output,
                    &mut scratch.row_errors,
                    |bytes| dot(tensor.kind, bytes, &scratch.q8),
                )?;
            }
            Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 => {
                scratch.diagnostics = scratch.diagnostics.select(false);
                let q8 = &scratch.q8;
                #[cfg(all(
                    feature = "experimental-neon",
                    target_arch = "aarch64",
                    target_os = "windows",
                    target_env = "msvc",
                    target_feature = "neon"
                ))]
                if !matches!(scratch.requested_backend(), MatvecBackend::Scalar) {
                    if tensor.kind == Q8_0 {
                        return scratch.row_parallelism.fill_four_rows(
                            &tensor.data,
                            row_bytes,
                            output,
                            &mut scratch.row_errors,
                            |rows| neon::dot_q8_0_eight_rows(rows, q8),
                            |rows| neon::dot_q8_0_four_rows(rows, q8),
                            |bytes| dot_legacy(Q8_0, bytes, q8),
                        );
                    }
                    if tensor.kind == Q4_0 {
                        return scratch.row_parallelism.fill_four_rows(
                            &tensor.data,
                            row_bytes,
                            output,
                            &mut scratch.row_errors,
                            |rows| neon::dot_q4_0_eight_rows(rows, q8),
                            |rows| neon::dot_q4_0_four_rows(rows, q8),
                            |bytes| dot_legacy(Q4_0, bytes, q8),
                        );
                    }
                }
                scratch.row_parallelism.fill(
                    &tensor.data,
                    row_bytes,
                    output,
                    &mut scratch.row_errors,
                    |bytes| dot_legacy(tensor.kind, bytes, q8),
                )?;
            }
            _ => {
                scratch.diagnostics = scratch.diagnostics.select(false);
                output.clear();
                output.try_reserve_exact(rows)?;
                for (index, _) in tensor.data.chunks_exact(row_bytes).enumerate() {
                    row_into(tensor, index, &mut scratch.row)?;
                    // Native F32/F16 projections use four FMA registers on
                    // complete 16-element blocks. Restrict this qualified
                    // contract to aligned rows: compiled scalar tails have
                    // different rounding and retain their prior behavior.
                    // BF16's pinned ARM64 dot still uses the binary64 sum.
                    output.push(
                        if scratch.pinned_accumulation
                            && matches!(tensor.kind, F32 | F16)
                            && columns.is_multiple_of(16)
                        {
                            pinned_gdn::dot(&scratch.row, &scratch.activations)?
                        } else {
                            finite(float_dot(&scratch.row, &scratch.activations))?
                        },
                    );
                }
            }
        }
        Ok(())
    }
}

#[expect(
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

pub(crate) fn round_f16_slice(values: &mut [f32]) -> bool {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(all_finite) = neon::round_f16_slice(values) {
        return all_finite;
    }
    let mut all_finite = true;
    for value in values {
        let rounded = round_f16(*value);
        all_finite &= rounded.is_finite();
        *value = rounded;
    }
    all_finite
}

pub(crate) fn append_half_bits(values: &[f32], output: &mut Vec<u16>) {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if neon::append_half_bits(values, output) {
        return;
    }
    output.extend(values.iter().copied().map(half_bits));
}

/// Returns `true` when every binary32 element in `values` is finite.
#[must_use]
pub fn is_finite_slice(values: &[f32]) -> bool {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(finite) = neon::is_finite_slice(values) {
        return finite;
    }
    values.iter().all(|value| value.is_finite())
}

#[inline]
#[cfg_attr(
    not(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    )),
    allow(
        clippy::missing_const_for_fn,
        reason = "Scalar stub matches the runtime NEON signature."
    )
)]
pub(crate) fn block16_all_finite_le(values: &[f32; 16], threshold: f32) -> bool {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    {
        neon::block16_all_finite_le(values, threshold)
    }
    #[cfg(not(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    )))]
    {
        let _ = (values, threshold);
        false
    }
}

pub(crate) fn add_slice_in_place(target: &mut [f32], other: &[f32]) {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if neon::add_slice_in_place(target, other) {
        return;
    }
    for (target, &other) in target.iter_mut().zip(other) {
        *target += other;
    }
}

pub(crate) fn add_slice_in_place_all_finite(target: &mut [f32], other: &[f32]) -> bool {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(all_finite) = neon::add_slice_in_place_all_finite(target, other) {
        return all_finite;
    }
    let mut all_finite = true;
    for (target, &other) in target.iter_mut().zip(other) {
        let sum = *target + other;
        *target = sum;
        all_finite &= sum.is_finite();
    }
    if target.len() > other.len() {
        for &target in &target[other.len()..] {
            all_finite &= target.is_finite();
        }
    }
    all_finite
}

pub(crate) fn mul_slice_in_place(target: &mut [f32], other: &[f32]) {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if neon::mul_slice_in_place(target, other) {
        return;
    }
    for (target, &other) in target.iter_mut().zip(other) {
        *target *= other;
    }
}

pub(crate) fn mul_sigmoid_in_place(values: &mut [f32], gate: &[f32]) {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if neon::mul_sigmoid_in_place(values, gate) {
        return;
    }
    for (v, &g) in values.iter_mut().zip(gate) {
        *v *= 1.0 / (1.0 + (-g).exp());
    }
}

pub(crate) fn source_value_tile<'a>(
    past: impl Iterator<Item = &'a [u16]>,
    current: &[f32],
    offset: usize,
    probabilities: &[f32],
    output: &mut [f32],
) {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    {
        let _ = neon::source_value_tile(past, current, offset, probabilities, output);
    }
    #[cfg(not(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    )))]
    {
        let last = probabilities.len().saturating_sub(1);
        let zero = std::iter::empty::<f64>().sum::<f64>();
        let mut sums = [zero; pinned_attention::VALUE_TILE];
        for (past_row, &probability) in past.zip(&probabilities[..last]) {
            for (sum, &value) in sums
                .iter_mut()
                .zip(&past_row[offset..offset + output.len()])
            {
                *sum += f64::from(probability * half_to_float(value));
            }
        }
        let probability = probabilities[last];
        for ((out, sum), &value) in output.iter_mut().zip(sums).zip(current) {
            *out = (sum + f64::from(probability * value)) as f32;
        }
    }
}

pub(crate) fn source_value_tile_four<'a>(
    past: impl Iterator<Item = &'a [u16]>,
    current: &[f32],
    offset: usize,
    probabilities: [&[f32]; 4],
    outputs: [&mut [f32]; 4],
) {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    {
        let _ = neon::source_value_tile_four(past, current, offset, probabilities, outputs);
    }
    #[cfg(not(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    )))]
    {
        let last = probabilities[0].len().saturating_sub(1);
        let zero = std::iter::empty::<f64>().sum::<f64>();
        let out_len = outputs[0].len();
        let mut sums = [[zero; pinned_attention::VALUE_TILE_FOUR]; 4];
        for (pos, past_row) in past.enumerate() {
            let row = &past_row[offset..offset + out_len];
            for (channel, &value) in row.iter().enumerate() {
                let decoded = half_to_float(value);
                for h in 0..4 {
                    sums[h][channel] += f64::from(probabilities[h][pos] * decoded);
                }
            }
        }
        for (output, (prob_row, sum_row)) in outputs
            .into_iter()
            .zip(probabilities.into_iter().zip(&sums))
        {
            let probability = prob_row[last];
            for ((out, &sum), &value) in output.iter_mut().zip(&sum_row[..out_len]).zip(current) {
                *out = (sum + f64::from(probability * value)) as f32;
            }
        }
    }
}

pub(crate) fn source_value_tile_four_two_tokens<'a>(
    past: impl Iterator<Item = &'a [u16]>,
    currents: [&[f32]; 2],
    offset: usize,
    probabilities0: [&[f32]; 4],
    probabilities1: [&[f32]; 4],
    outputs0: [&mut [f32]; 4],
    outputs1: [&mut [f32]; 4],
) {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    {
        let _ = neon::source_value_tile_four_two_tokens(
            past,
            currents,
            offset,
            probabilities0,
            probabilities1,
            outputs0,
            outputs1,
        );
    }
    #[cfg(not(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    )))]
    {
        let last0 = probabilities0[0].len().saturating_sub(1);
        let last1 = probabilities1[0].len().saturating_sub(1);
        let zero = std::iter::empty::<f64>().sum::<f64>();
        let out_len = outputs0[0].len();
        let mut sums0 = [[zero; pinned_attention::VALUE_TILE_FOUR_TWO_TOKENS]; 4];
        let mut sums1 = [[zero; pinned_attention::VALUE_TILE_FOUR_TWO_TOKENS]; 4];
        for (pos, past_row) in past.enumerate() {
            let row = &past_row[offset..offset + out_len];
            if pos < last0 {
                for (channel, &value) in row.iter().enumerate() {
                    let decoded = half_to_float(value);
                    for h in 0..4 {
                        sums0[h][channel] += f64::from(probabilities0[h][pos] * decoded);
                        sums1[h][channel] += f64::from(probabilities1[h][pos] * decoded);
                    }
                }
            } else {
                for (channel, &value) in row.iter().enumerate() {
                    let decoded = half_to_float(value);
                    for h in 0..4 {
                        sums1[h][channel] += f64::from(probabilities1[h][pos] * decoded);
                    }
                }
            }
        }
        for (output, (prob_row, sum_row)) in outputs0
            .into_iter()
            .zip(probabilities0.into_iter().zip(&sums0))
        {
            let probability = prob_row[last0];
            for ((out, &sum), &value) in output.iter_mut().zip(&sum_row[..out_len]).zip(currents[0])
            {
                *out = (sum + f64::from(probability * value)) as f32;
            }
        }
        for (output, (prob_row, sum_row)) in outputs1
            .into_iter()
            .zip(probabilities1.into_iter().zip(&sums1))
        {
            let probability = prob_row[last1];
            for ((out, &sum), &value) in output.iter_mut().zip(&sum_row[..out_len]).zip(currents[1])
            {
                *out = (sum + f64::from(probability * value)) as f32;
            }
        }
    }
}

pub(crate) fn mul_slice_into(input: &[f32], signs: &[f32], output: &mut [f32]) {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if neon::mul_slice_into(input, signs, output) {
        return;
    }
    for ((out, &value), &sign) in output.iter_mut().zip(input).zip(signs) {
        *out = value * sign;
    }
}

pub(crate) fn scale_slice_into(input: &[f32], scale: f32, output: &mut [f32]) {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if neon::scale_slice_into(input, scale, output) {
        return;
    }
    for (out, &value) in output.iter_mut().zip(input) {
        *out = scale * value;
    }
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Preserve separate binary32 multiply and add roundings."
)]
pub(crate) fn add_scaled_slice_in_place(target: &mut [f32], input: &[f32], scale: f32) {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if neon::add_scaled_slice_in_place(target, input, scale) {
        return;
    }
    for (target, &value) in target.iter_mut().zip(input) {
        *target += scale * value;
    }
}

pub(crate) fn gdn_wy_update_four_rows(
    states: [&[f32]; 4],
    last_cumulative: f32,
    inputs: &[pinned_gdn::ChunkInput<'_>],
    scaled_deltas: &[[f32; 32]; 4],
    matrix_rows: [&mut [f32]; 4],
) {
    let [m0, m1, m2, m3] = matrix_rows;
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if neon::gdn_wy_update_four_rows(
        states,
        last_cumulative,
        inputs,
        scaled_deltas,
        [m0, m1, m2, m3],
    ) {
        return;
    }
    let rows = [m0, m1, m2, m3];
    for (r, matrix_row) in rows.into_iter().enumerate() {
        scale_slice_into(states[r], last_cumulative, matrix_row);
        for (prior, input) in inputs.iter().enumerate() {
            add_scaled_slice_in_place(matrix_row, input.key, scaled_deltas[r][prior]);
        }
    }
}

pub(crate) fn gdn_wy_update_eight_rows(
    states: [&[f32]; 8],
    last_cumulative: f32,
    inputs: &[pinned_gdn::ChunkInput<'_>],
    scaled_deltas: &[[f32; 32]; 8],
    matrix_rows: [&mut [f32]; 8],
) {
    let [m0, m1, m2, m3, m4, m5, m6, m7] = matrix_rows;
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if neon::gdn_wy_update_eight_rows(
        states,
        last_cumulative,
        inputs,
        scaled_deltas,
        [m0, m1, m2, m3, m4, m5, m6, m7],
    ) {
        return;
    }
    let rows = [m0, m1, m2, m3, m4, m5, m6, m7];
    for (r, matrix_row) in rows.into_iter().enumerate() {
        scale_slice_into(states[r], last_cumulative, matrix_row);
        for (prior, input) in inputs.iter().enumerate() {
            add_scaled_slice_in_place(matrix_row, input.key, scaled_deltas[r][prior]);
        }
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::as_conversions,
    reason = "The pinned scalar numerical contract explicitly rounds this value to binary32."
)]
pub(crate) fn l2_norm_in_place(input: &mut [f32], epsilon: f32) {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if neon::l2_norm_in_place(input, epsilon) {
        return;
    }
    let squares = input
        .iter()
        .map(|&value| f64::from(value * value))
        .sum::<f64>();
    let scale = 1.0 / (squares as f32).sqrt().max(epsilon);
    for value in input {
        *value *= scale;
    }
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
    let result = (|| {
        let scale = rms_norm_scale(input, weight, epsilon)?;
        #[cfg(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc",
            target_feature = "neon"
        ))]
        if let Some(result) = neon::rms_norm_scale_into::<false>(input, weight, scale, output) {
            return result;
        }
        collect_into(
            input
                .iter()
                .zip(weight)
                .map(|(&value, &w)| finite(value * scale * w)),
            output,
        )
    })();
    if result.is_err() {
        output.clear();
    }
    result
}

/// Applies the weighted RMS contract of the pinned Windows ARM64 Prism binary.
///
/// The compiled fused RMS/MUL loop rounds `input * weight` before multiplying
/// by the scale, although the upstream scalar source spells the other order.
/// Keep this separate from the public source-order scalar reference.
pub(crate) fn rms_norm_pinned_into(
    input: &[f32],
    weight: &[f32],
    epsilon: f32,
    output: &mut Vec<f32>,
) -> Result<()> {
    let result = (|| {
        let scale = rms_norm_scale(input, weight, epsilon)?;
        #[cfg(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc",
            target_feature = "neon"
        ))]
        if let Some(result) = neon::rms_norm_scale_into::<true>(input, weight, scale, output) {
            return result;
        }
        collect_into(
            input
                .iter()
                .zip(weight)
                .map(|(&value, &w)| finite((value * w) * scale)),
            output,
        )
    })();
    if result.is_err() {
        output.clear();
    }
    result
}

pub(crate) fn add_rms_norm_into(
    values: &mut [f32],
    delta: &[f32],
    weight: &[f32],
    epsilon: f32,
    output: &mut Vec<f32>,
) -> Result<()> {
    let result = (|| {
        let scale = add_rms_norm_scale(values, delta, weight, epsilon)?;
        #[cfg(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc",
            target_feature = "neon"
        ))]
        if let Some(result) = neon::rms_norm_scale_into::<false>(values, weight, scale, output) {
            return result;
        }
        collect_into(
            values
                .iter()
                .zip(weight)
                .map(|(&value, &w)| finite(value * scale * w)),
            output,
        )
    })();
    if result.is_err() {
        output.clear();
    }
    result
}

pub(crate) fn add_rms_norm_pinned_into(
    values: &mut [f32],
    delta: &[f32],
    weight: &[f32],
    epsilon: f32,
    output: &mut Vec<f32>,
) -> Result<()> {
    let result = (|| {
        let scale = add_rms_norm_scale(values, delta, weight, epsilon)?;
        #[cfg(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc",
            target_feature = "neon"
        ))]
        if let Some(result) = neon::rms_norm_scale_into::<true>(values, weight, scale, output) {
            return result;
        }
        collect_into(
            values
                .iter()
                .zip(weight)
                .map(|(&value, &w)| finite((value * w) * scale)),
            output,
        )
    })();
    if result.is_err() {
        output.clear();
    }
    result
}

#[cfg(test)]
pub(crate) fn rms_norm_in_place(values: &mut [f32], weight: &[f32], epsilon: f32) -> Result<()> {
    let scale = rms_norm_scale(values, weight, epsilon)?;
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(result) = neon::rms_norm_scale_in_place::<false>(values, weight, scale) {
        return result;
    }
    for (value, &w) in values.iter_mut().zip(weight) {
        *value = finite(*value * scale * w)?;
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn rms_norm_pinned_in_place(
    values: &mut [f32],
    weight: &[f32],
    epsilon: f32,
) -> Result<()> {
    let scale = rms_norm_scale(values, weight, epsilon)?;
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(result) = neon::rms_norm_scale_in_place::<true>(values, weight, scale) {
        return result;
    }
    for (value, &w) in values.iter_mut().zip(weight) {
        *value = finite((*value * w) * scale)?;
    }
    Ok(())
}

pub(crate) fn rms_norm_swiglu_in_place<const PINNED: bool>(
    values: &mut [f32],
    weight: &[f32],
    epsilon: f32,
    gate: &[f32],
) -> Result<()> {
    let scale = rms_norm_scale(values, weight, epsilon)?;
    if gate.len() != values.len() {
        return Err(invalid("SwiGLU requires equal gate and up vector lengths"));
    }
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(result) =
        neon::rms_norm_scale_swiglu_in_place::<PINNED>(values, weight, scale, gate)
    {
        return result;
    }
    if PINNED {
        for (value, &w) in values.iter_mut().zip(weight) {
            *value = finite((*value * w) * scale)?;
        }
        pinned_activations::swiglu_in_place(gate, values)
    } else {
        for (value, &w) in values.iter_mut().zip(weight) {
            *value = finite(*value * scale * w)?;
        }
        for (value, &g) in values.iter_mut().zip(gate) {
            *value *= silu(g);
        }
        Ok(())
    }
}

pub(crate) fn rms_norm_swiglu_into<const PINNED: bool>(
    input: &[f32],
    weight: &[f32],
    epsilon: f32,
    gate: &[f32],
    output: &mut [f32],
) -> Result<()> {
    let scale = rms_norm_scale(input, weight, epsilon)?;
    if gate.len() != input.len() || output.len() != input.len() {
        return Err(invalid("SwiGLU requires equal gate and up vector lengths"));
    }
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(result) =
        neon::rms_norm_scale_swiglu_into::<PINNED>(input, weight, scale, gate, output)
    {
        return result;
    }
    if PINNED {
        for ((out, &val), &w) in output.iter_mut().zip(input).zip(weight) {
            *out = finite((val * w) * scale)?;
        }
        pinned_activations::swiglu_in_place(gate, output)
    } else {
        for (((out, &val), &w), &g) in output.iter_mut().zip(input).zip(weight).zip(gate) {
            let norm = finite(val * scale * w)?;
            *out = norm * silu(g);
        }
        Ok(())
    }
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Keep explicit head/rotary geometry and separate multiply/add roundings for Source fallback."
)]
pub(crate) fn rms_norm_rope_round_f16_into<const PINNED: bool>(
    input: &[f32],
    weight: &[f32],
    epsilon: f32,
    head: usize,
    rotary: usize,
    frequencies: &[(f32, f32)],
    output: &mut [f32],
) -> Result<bool> {
    if output.len() != input.len() {
        return Err(invalid(
            "RMS normalization requires equal nonempty vectors and nonnegative epsilon",
        ));
    }
    let scale = rms_norm_scale(input, weight, epsilon)?;
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if head == 256
        && rotary == 64
        && let (Ok(in_arr), Ok(w_arr), Ok(freq_arr), Ok(out_arr)) = (
            <&[f32; 256]>::try_from(input),
            <&[f32; 256]>::try_from(weight),
            <&[(f32, f32); 32]>::try_from(frequencies),
            <&mut [f32; 256]>::try_from(&mut *output),
        )
    {
        if PINNED
            && freq_arr
                .iter()
                .any(|&(cosine, sine)| !cosine.is_finite() || !sine.is_finite())
        {
            return Err(invalid("invalid pinned Swift RoPE geometry or frequencies"));
        }
        if let Some(result) = neon::rms_norm_rope_round_f16_256_into::<PINNED>(
            in_arr, w_arr, scale, freq_arr, out_arr,
        ) {
            return result;
        }
    }
    for ((out, &val), &w) in output.iter_mut().zip(input).zip(weight) {
        *out = finite(if PINNED {
            (val * w) * scale
        } else {
            (val * scale) * w
        })?;
    }
    if PINNED {
        pinned_attention::rope_in_place(output, head, rotary, frequencies)?;
    } else {
        for head_slice in output.chunks_exact_mut(head) {
            for (index, &(cosine, sine)) in frequencies.iter().enumerate() {
                let first = head_slice[index];
                let second = head_slice[index + rotary / 2];
                head_slice[index] = first * cosine - second * sine;
                head_slice[index + rotary / 2] = first * sine + second * cosine;
            }
        }
        if !is_finite_slice(output) {
            return Err(invalid("nonfinite values in RoPE output"));
        }
    }
    Ok(round_f16_slice(output))
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Keep separate multiply/add roundings for Source fallback."
)]
pub(crate) fn rms_norm_rope_round_f16_in_place<const PINNED: bool>(
    values: &mut [f32],
    weight: &[f32],
    epsilon: f32,
    head: usize,
    rotary: usize,
    frequencies: &[(f32, f32)],
) -> Result<bool> {
    let scale = rms_norm_scale(values, weight, epsilon)?;
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if head == 256
        && rotary == 64
        && let (Ok(val_arr), Ok(w_arr), Ok(freq_arr)) = (
            <&mut [f32; 256]>::try_from(&mut *values),
            <&[f32; 256]>::try_from(weight),
            <&[(f32, f32); 32]>::try_from(frequencies),
        )
    {
        if PINNED
            && freq_arr
                .iter()
                .any(|&(cosine, sine)| !cosine.is_finite() || !sine.is_finite())
        {
            return Err(invalid("invalid pinned Swift RoPE geometry or frequencies"));
        }
        if let Some(result) =
            neon::rms_norm_rope_round_f16_256_in_place::<PINNED>(val_arr, w_arr, scale, freq_arr)
        {
            return result;
        }
    }
    for (val, &w) in values.iter_mut().zip(weight) {
        *val = finite(if PINNED {
            (*val * w) * scale
        } else {
            (*val * scale) * w
        })?;
    }
    if PINNED {
        pinned_attention::rope_in_place(values, head, rotary, frequencies)?;
    } else {
        for head_slice in values.chunks_exact_mut(head) {
            for (index, &(cosine, sine)) in frequencies.iter().enumerate() {
                let first = head_slice[index];
                let second = head_slice[index + rotary / 2];
                head_slice[index] = first * cosine - second * sine;
                head_slice[index + rotary / 2] = first * sine + second * cosine;
            }
        }
        if !is_finite_slice(values) {
            return Err(invalid("nonfinite values in RoPE output"));
        }
    }
    Ok(round_f16_slice(values))
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "The scalar normalization contract rounds the binary64 mean to binary32."
)]
fn rms_norm_scale(input: &[f32], weight: &[f32], epsilon: f32) -> Result<f32> {
    if input.is_empty() || input.len() != weight.len() || !epsilon.is_finite() || epsilon < 0.0 {
        return Err(invalid(
            "RMS normalization requires equal nonempty vectors and nonnegative epsilon",
        ));
    }
    let count = u32::try_from(input.len())?;
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    let sum = if let Some(sum) = neon::rms_norm_squares(input, weight) {
        sum?
    } else {
        let mut sum = 0.0_f64;
        for (&value, &w) in input.iter().zip(weight) {
            finite(value)?;
            finite(w)?;
            sum += f64::from(value * value);
        }
        sum
    };
    #[cfg(not(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    )))]
    let sum = {
        let mut sum = 0.0_f64;
        for (&value, &w) in input.iter().zip(weight) {
            finite(value)?;
            finite(w)?;
            sum += f64::from(value * value);
        }
        sum
    };
    let mean = (sum / f64::from(count)) as f32;
    let scale = 1.0 / (mean + epsilon).sqrt();
    if !scale.is_finite() || scale <= 0.0 {
        return Err(invalid("RMS normalization produced an invalid scale"));
    }
    Ok(scale)
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "The scalar normalization contract rounds the binary64 mean to binary32."
)]
fn add_rms_norm_scale(
    values: &mut [f32],
    delta: &[f32],
    weight: &[f32],
    epsilon: f32,
) -> Result<f32> {
    if values.is_empty()
        || values.len() != delta.len()
        || values.len() != weight.len()
        || !epsilon.is_finite()
        || epsilon < 0.0
    {
        add_slice_in_place(values, delta);
        return Err(invalid(
            "RMS normalization requires equal nonempty vectors and nonnegative epsilon",
        ));
    }
    let Ok(count) = u32::try_from(values.len()) else {
        add_slice_in_place(values, delta);
        return Err(invalid("RMS normalization vector length overflow"));
    };
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    let sum = if let Some(sum) = neon::add_rms_norm_squares(values, delta, weight) {
        sum?
    } else {
        let mut sum = 0.0_f64;
        let mut all_finite = true;
        for ((value, &d), &w) in values.iter_mut().zip(delta).zip(weight) {
            let v = *value + d;
            *value = v;
            all_finite &= v.is_finite() && w.is_finite();
            sum += f64::from(v * v);
        }
        if !all_finite {
            return Err(invalid("nonfinite tensor or activation value"));
        }
        sum
    };
    #[cfg(not(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    )))]
    let sum = {
        let mut sum = 0.0_f64;
        let mut all_finite = true;
        for ((value, &d), &w) in values.iter_mut().zip(delta).zip(weight) {
            let v = *value + d;
            *value = v;
            all_finite &= v.is_finite() && w.is_finite();
            sum += f64::from(v * v);
        }
        if !all_finite {
            return Err(invalid("nonfinite tensor or activation value"));
        }
        sum
    };
    let mean = (sum / f64::from(count)) as f32;
    let scale = 1.0 / (mean + epsilon).sqrt();
    if !scale.is_finite() || scale <= 0.0 {
        return Err(invalid("RMS normalization produced an invalid scale"));
    }
    Ok(scale)
}

/// Applies a normalized Sylvester Hadamard transform to one contiguous block.
///
/// # Errors
/// Rejects zero or non-power-of-two block lengths and nonfinite input/output.
pub fn hadamard(values: &mut [f32]) -> Result<()> {
    if !values.len().is_power_of_two() {
        return Err(invalid("Hadamard requires a finite power-of-two block"));
    }
    let count = u32::try_from(values.len())?;
    let normalization = 1.0 / integer_float(i32::try_from(count)?).sqrt();
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(result) = neon::hadamard(values, normalization) {
        return result;
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(invalid("Hadamard requires a finite power-of-two block"));
    }
    let mut half = 1;
    while half < values.len() {
        for block in values.chunks_exact_mut(half * 2) {
            let (left_half, right_half) = block.split_at_mut(half);
            for (left, right) in left_half.iter_mut().zip(right_half) {
                let l = *left;
                let r = *right;
                *left = l + r;
                *right = l - r;
            }
        }
        half *= 2;
    }
    for value in values {
        *value = finite(*value * normalization)?;
    }
    Ok(())
}

pub(crate) fn hadamard_into(
    input: &[f32],
    signs: Option<&[f32]>,
    output: &mut [f32],
) -> Result<()> {
    if !output.len().is_power_of_two() || input.len() != output.len() {
        return Err(invalid("Hadamard requires a finite power-of-two block"));
    }
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    {
        let count = u32::try_from(output.len())?;
        let normalization = 1.0 / integer_float(i32::try_from(count)?).sqrt();
        if let Some(result) = neon::hadamard_into(input, signs, output, normalization) {
            return result;
        }
    }
    if let Some(signs) = signs {
        mul_slice_into(input, signs, output);
    } else {
        output.copy_from_slice(input);
    }
    hadamard(output)
}

#[expect(
    clippy::too_many_arguments,
    reason = "Grouped-value Hadamard block fuses head permutation, optional sign multiply, and stage-64 butterfly."
)]
pub(crate) fn hadamard_grouped_strided_block_into(
    input: &[f32],
    head_stride: usize,
    signs: Option<&[f32]>,
    block_idx: usize,
    heads_per_block: usize,
    repetitions: usize,
    ssm_keys: usize,
    ssm_head: usize,
    output: &mut [f32],
) -> Result<()> {
    if !output.len().is_power_of_two()
        || ssm_head == 0
        || head_stride < ssm_head
        || output.len() != heads_per_block.saturating_mul(ssm_head)
    {
        return Err(invalid("Hadamard requires a finite power-of-two block"));
    }
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    {
        let count = u32::try_from(output.len())?;
        let normalization = 1.0 / integer_float(i32::try_from(count)?).sqrt();
        if let Some(result) = neon::hadamard_grouped_strided_block_into(
            input,
            head_stride,
            signs,
            block_idx,
            heads_per_block,
            repetitions,
            ssm_keys,
            ssm_head,
            output,
            normalization,
        ) {
            return result;
        }
    }
    for (h, out_head) in output.chunks_exact_mut(ssm_head).enumerate() {
        let target_head = block_idx * heads_per_block + h;
        let key = target_head / repetitions;
        let repeat = target_head % repetitions;
        let source = (repeat * ssm_keys + key) * head_stride;
        let target = target_head * ssm_head;
        let in_head = &input[source..source + ssm_head];
        if let Some(signs) = signs {
            mul_slice_into(in_head, &signs[target..target + ssm_head], out_head);
        } else {
            out_head.copy_from_slice(in_head);
        }
    }
    hadamard(output)
}

#[cfg(test)]
#[expect(
    clippy::too_many_arguments,
    reason = "Grouped-value Hadamard block fuses head permutation, optional sign multiply, and stage-64 butterfly."
)]
pub(crate) fn hadamard_grouped_block_into(
    input: &[f32],
    signs: Option<&[f32]>,
    block_idx: usize,
    heads_per_block: usize,
    repetitions: usize,
    ssm_keys: usize,
    ssm_head: usize,
    output: &mut [f32],
) -> Result<()> {
    hadamard_grouped_strided_block_into(
        input,
        ssm_head,
        signs,
        block_idx,
        heads_per_block,
        repetitions,
        ssm_keys,
        ssm_head,
        output,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "Block slice helper passes explicit RMSNorm scale, optional Hadamard signs and output slices."
)]
fn rms_norm_hadamard_blocks_into<const PINNED: bool>(
    base_block: usize,
    input: &[f32],
    weight: &[f32],
    scale: f32,
    signs: Option<&[f32]>,
    block: usize,
    norm_chunk: &mut [f32],
    rot_chunk: &mut [f32],
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    normalization: f32,
) -> Result<()> {
    for (local_idx, (norm_block, rot_block)) in norm_chunk
        .chunks_exact_mut(block)
        .zip(rot_chunk.chunks_exact_mut(block))
        .enumerate()
    {
        let offset = (base_block + local_idx) * block;
        let in_block = &input[offset..offset + block];
        let w_block = &weight[offset..offset + block];
        let sign_block = signs.map(|s| &s[offset..offset + block]);
        #[cfg(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc",
            target_feature = "neon"
        ))]
        if let Some(res) = neon::rms_norm_hadamard_block_into::<PINNED>(
            in_block,
            w_block,
            scale,
            sign_block,
            norm_block,
            rot_block,
            normalization,
        ) {
            res?;
            continue;
        }
        for ((n_dst, &val), &w) in norm_block.iter_mut().zip(in_block).zip(w_block) {
            *n_dst = finite(if PINNED {
                (val * w) * scale
            } else {
                (val * scale) * w
            })?;
        }
        hadamard_into(norm_block, sign_block, rot_block)?;
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "Fused RMSNorm and Hadamard rotation keeps explicit block geometry, parallelism and dual output buffers."
)]
fn rms_norm_scale_hadamard_into<const PINNED: bool>(
    input: &[f32],
    weight: &[f32],
    scale: f32,
    block: usize,
    signs: Option<&[f32]>,
    parallelism: &RowParallelism,
    norm_out: &mut Vec<f32>,
    rotated_out: &mut Vec<f32>,
) -> Result<()> {
    if norm_out.len() != input.len() {
        norm_out.clear();
        norm_out.try_reserve_exact(input.len())?;
        norm_out.resize(input.len(), 0.0);
    }
    if rotated_out.len() != input.len() {
        rotated_out.clear();
        rotated_out.try_reserve_exact(input.len())?;
        rotated_out.resize(input.len(), 0.0);
    }
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    let normalization = {
        let count = u32::try_from(block)?;
        1.0 / integer_float(i32::try_from(count)?).sqrt()
    };
    if parallelism.workers() > 1 && block >= 256 && input.len() >= block.saturating_mul(4) {
        return parallelism.update_paired_chunks(
            norm_out,
            block,
            rotated_out,
            block,
            |base_block, norm_chunk, rot_chunk| {
                rms_norm_hadamard_blocks_into::<PINNED>(
                    base_block,
                    input,
                    weight,
                    scale,
                    signs,
                    block,
                    norm_chunk,
                    rot_chunk,
                    #[cfg(all(
                        feature = "experimental-neon",
                        target_arch = "aarch64",
                        target_os = "windows",
                        target_env = "msvc",
                        target_feature = "neon"
                    ))]
                    normalization,
                )
            },
        );
    }
    rms_norm_hadamard_blocks_into::<PINNED>(
        0,
        input,
        weight,
        scale,
        signs,
        block,
        norm_out,
        rotated_out,
        #[cfg(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc",
            target_feature = "neon"
        ))]
        normalization,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "Fused RMSNorm and Hadamard rotation keeps explicit block geometry, parallelism and dual output buffers."
)]
pub(crate) fn rms_norm_hadamard_into<const PINNED: bool>(
    input: &[f32],
    weight: &[f32],
    epsilon: f32,
    block: usize,
    signs: Option<&[f32]>,
    parallelism: &RowParallelism,
    norm_out: &mut Vec<f32>,
    rotated_out: &mut Vec<f32>,
) -> Result<()> {
    let result = (|| {
        if !block.is_power_of_two()
            || !input.len().is_multiple_of(block)
            || signs.is_some_and(|s| s.len() < input.len())
        {
            return Err(invalid("Hadamard requires a finite power-of-two block"));
        }
        let scale = rms_norm_scale(input, weight, epsilon)?;
        rms_norm_scale_hadamard_into::<PINNED>(
            input,
            weight,
            scale,
            block,
            signs,
            parallelism,
            norm_out,
            rotated_out,
        )
    })();
    if result.is_err() {
        norm_out.clear();
        rotated_out.clear();
    }
    result
}

#[expect(
    clippy::too_many_arguments,
    reason = "Fused residual-add, RMSNorm and Hadamard rotation keeps explicit block geometry, parallelism and dual output buffers."
)]
pub(crate) fn add_rms_norm_hadamard_into<const PINNED: bool>(
    values: &mut [f32],
    delta: &[f32],
    weight: &[f32],
    epsilon: f32,
    block: usize,
    signs: Option<&[f32]>,
    parallelism: &RowParallelism,
    norm_out: &mut Vec<f32>,
    rotated_out: &mut Vec<f32>,
) -> Result<()> {
    let result = (|| {
        if !block.is_power_of_two()
            || !values.len().is_multiple_of(block)
            || signs.is_some_and(|s| s.len() < values.len())
        {
            add_slice_in_place(values, delta);
            return Err(invalid("Hadamard requires a finite power-of-two block"));
        }
        let scale = add_rms_norm_scale(values, delta, weight, epsilon)?;
        rms_norm_scale_hadamard_into::<PINNED>(
            values,
            weight,
            scale,
            block,
            signs,
            parallelism,
            norm_out,
            rotated_out,
        )
    })();
    if result.is_err() {
        norm_out.clear();
        rotated_out.clear();
    }
    result
}

pub(crate) fn swiglu_hadamard_block_read_into<const PINNED: bool>(
    gate: &[f32],
    up: &[f32],
    signs: Option<&[f32]>,
    output: &mut [f32],
) -> Result<()> {
    if !output.len().is_power_of_two()
        || gate.len() != output.len()
        || up.len() != output.len()
        || signs.is_some_and(|s| s.len() < output.len())
    {
        return Err(invalid("Hadamard requires a finite power-of-two block"));
    }
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if PINNED {
        let count = u32::try_from(output.len())?;
        let normalization = 1.0 / integer_float(i32::try_from(count)?).sqrt();
        if let Some(result) =
            neon::swiglu_hadamard_block_read_into(gate, up, signs, output, normalization)
        {
            return result;
        }
    }
    if PINNED {
        pinned_activations::swiglu_into(gate, up, output)?;
        if let Some(signs) = signs {
            mul_slice_in_place(output, &signs[..output.len()]);
        }
    } else if let Some(signs) = signs {
        for (((out, &u), &g), &sign) in output.iter_mut().zip(up).zip(gate).zip(signs) {
            *out = (u * silu(g)) * sign;
        }
    } else {
        for ((out, &u), &g) in output.iter_mut().zip(up).zip(gate) {
            *out = u * silu(g);
        }
    }
    hadamard(output)
}

/// Replaces finite logits by their normalized, stable exponential weights.
///
/// Negative infinity is accepted as an attention mask. All-masked rows fail.
///
/// # Errors
/// Rejects empty vectors, NaNs, positive infinity, and all-masked rows.
#[expect(
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

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "The scalar softmax contract rounds the binary64 inverse sum to binary32 before binary16 rounding."
)]
pub(crate) fn softmax_round_f16(values: &mut [f32]) -> Result<()> {
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
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if neon::softmax_scale_round_f16_in_place(values, inverse_sum) {
        return Ok(());
    }
    for value in values {
        *value = round_f16(*value * inverse_sum);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use sha2::{Digest, Sha256};

    use super::{
        MatvecScratch, Tensor, matvec_into, q8_into, q8k_into, rms_norm, rms_norm_pinned_into,
    };

    fn serialized_q8k(block: &super::Q8KBlock) -> Vec<u8> {
        block
            .scale
            .to_le_bytes()
            .into_iter()
            .chain(block.values.iter().flat_map(|value| value.to_le_bytes()))
            .chain(block.sums.iter().flat_map(|value| value.to_le_bytes()))
            .collect()
    }

    #[test]
    fn pinned_q8_preparation_matches_independent_native_rounding_capture() -> crate::Result<()> {
        let fixture = include_bytes!("../tests/fixtures/swift-q8-parity.bin");
        let provenance: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/swift-q8-parity.json"))?;
        let hash: String = Sha256::digest(fixture)
            .into_iter()
            .flat_map(|byte| [byte >> 4, byte & 15])
            .map(|nibble| char::from(b"0123456789abcdef"[usize::from(nibble)]))
            .collect();
        assert_eq!(Some(hash.as_str()), provenance["fixture_sha256"].as_str());
        assert_eq!(fixture.len(), 420);
        assert_eq!(
            provenance["runtime_commit"].as_str(),
            Some("73bfe6cc485554822f80c473a3c0c4f25e80f41b")
        );
        assert_eq!(
            provenance["ggml_cpu_sha256"].as_str(),
            Some("701b354a38b25e5c6dc57a1559f71cd26b2d0f8d69c1576a0e77a59a4273d2ab")
        );
        let cases = provenance["cases"]
            .as_array()
            .ok_or_else(|| super::invalid("native Q8 fixture cases missing"))?;
        assert_eq!(cases.len(), 6);
        for backend in [
            super::MatvecBackend::Scalar,
            super::MatvecBackend::Auto,
            super::MatvecBackend::ExperimentalNeon,
        ] {
            // PTQ1 uses Q8_0; Q4_1 uses Q8_1's additional scaled sum.
            for (kind, columns, bytes) in [(super::PTQ1, 128, 28), (super::Q4_1, 32, 20)] {
                let tensor = Tensor {
                    name: "q8-rounding".to_owned(),
                    dimensions: vec![columns, 1],
                    kind,
                    data: Arc::from(vec![0_u8; bytes]),
                };
                let mut pinned = MatvecScratch::with_pinned_backend(backend);
                let mut source = MatvecScratch::with_backend(backend);
                for (index, case) in cases.iter().enumerate() {
                    let words = case["input_bits"]
                        .as_array()
                        .ok_or_else(|| super::invalid("native Q8 fixture input missing"))?;
                    let mut input = Vec::new();
                    for word in words {
                        let word = word
                            .as_str()
                            .ok_or_else(|| super::invalid("native Q8 fixture word missing"))?;
                        input.push(f32::from_bits(u32::from_str_radix(word, 16)?));
                    }
                    assert_eq!(input.len(), 32);
                    input = input.repeat(columns / 32);
                    let prepared = pinned.prepare(&tensor, &input)?;
                    for block in &prepared.scratch.q8 {
                        let mut q8 = super::half_bits(block.scale).to_le_bytes().to_vec();
                        q8.extend(block.values.iter().flat_map(|value| value.to_le_bytes()));
                        assert_eq!(q8, fixture[index * 70..index * 70 + 34]);
                        let mut q8_1 = super::half_bits(block.scale).to_le_bytes().to_vec();
                        q8_1.extend(super::half_bits(block.sum).to_le_bytes());
                        q8_1.extend(block.values.iter().flat_map(|value| value.to_le_bytes()));
                        assert_eq!(q8_1, fixture[index * 70 + 34..(index + 1) * 70]);
                    }
                    if index == 1 {
                        // The public source API retains roundf's ties away
                        // from zero even when graph arithmetic follows ARM64.
                        let prepared = source.prepare(&tensor, &input)?;
                        assert_eq!(&prepared.scratch.q8[0].values[2..8], &[1, -1, 2, -2, 3, -3]);
                        let mut direct = Vec::new();
                        q8_into(&input, &mut direct)?;
                        assert_eq!(&direct[0].values[2..8], &[1, -1, 2, -2, 3, -3]);
                    }
                    input[17] = f32::NAN;
                    assert!(pinned.prepare(&tensor, &input).is_err());
                }
            }
        }
        Ok(())
    }

    #[test]
    fn pinned_q8k_preparation_matches_independent_native_half_boundary_capture() -> crate::Result<()>
    {
        let fixture = include_bytes!("../tests/fixtures/swift-q8k-parity.bin");
        let provenance: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/swift-q8k-parity.json"))?;
        let hash: String = Sha256::digest(fixture)
            .into_iter()
            .flat_map(|byte| [byte >> 4, byte & 15])
            .map(|nibble| char::from(b"0123456789abcdef"[usize::from(nibble)]))
            .collect();
        assert_eq!(Some(hash.as_str()), provenance["fixture_sha256"].as_str());
        assert_eq!(fixture.len(), 584);
        assert_eq!(
            provenance["runtime_commit"].as_str(),
            Some("73bfe6cc485554822f80c473a3c0c4f25e80f41b")
        );
        assert_eq!(
            provenance["ggml_cpu_sha256"].as_str(),
            Some("701b354a38b25e5c6dc57a1559f71cd26b2d0f8d69c1576a0e77a59a4273d2ab")
        );
        let tensor = Tensor {
            name: "q8k-boundary".to_owned(),
            dimensions: vec![256, 1],
            kind: super::PQ2,
            data: Arc::from([0_u8; 68]),
        };
        for backend in [
            super::MatvecBackend::Scalar,
            super::MatvecBackend::Auto,
            super::MatvecBackend::ExperimentalNeon,
        ] {
            let mut pinned = MatvecScratch::with_pinned_backend(backend);
            let mut source = MatvecScratch::with_backend(backend);
            for (case, (word, native_code, source_code)) in
                [(0xbe01_0f7e, 71, 72), (0x3e01_0f7e, -71, -72)]
                    .into_iter()
                    .enumerate()
            {
                let mut input = [0.0; 256];
                input[0] = f32::from_bits(0x3e65_3d88);
                input[1] = f32::from_bits(word);
                {
                    let prepared = pinned.prepare(&tensor, &input)?;
                    let block = &prepared.scratch.q8k[0];
                    assert_eq!(block.values[1], native_code);
                    assert_eq!(serialized_q8k(block), fixture[case * 292..(case + 1) * 292]);
                }
                {
                    let prepared = source.prepare(&tensor, &input)?;
                    assert_eq!(prepared.scratch.q8k[0].values[1], source_code);
                }
                let mut scalar_blocks = Vec::new();
                q8k_into(&input, &mut scalar_blocks)?;
                assert_eq!(scalar_blocks[0].values[1], source_code);
                // Failed preparation cannot expose a prepared projection or
                // prevent the next valid request from refreshing its storage.
                input[17] = f32::NAN;
                assert!(pinned.prepare(&tensor, &input).is_err());
            }
        }
        Ok(())
    }

    #[test]
    fn pinned_pq2_accumulation_matches_independent_native_binary_capture() -> crate::Result<()> {
        // Pinned 73bfe6cc Windows ARM64 ggml-cpu.dll, SHA256
        // 701b354a38b25e5c6dc57a1559f71cd26b2d0f8d69c1576a0e77a59a4273d2ab.
        // Invoke its type-142 vec_dot with these exact native packed buffers;
        // expected bits are the captured native result, not a Rust FMA oracle.
        let mut weights = [0xaa; 68];
        weights[..2].copy_from_slice(&0x3c01_u16.to_le_bytes());
        weights[34..36].copy_from_slice(&0x3787_u16.to_le_bytes());
        let mut values = [3; 256];
        values[128..].fill(-13);
        let mut sums = [48; 16];
        sums[8..].fill(-208);
        let activations = [super::Q8KBlock {
            scale: f32::from_bits(0x3da0_0001),
            values,
            sums,
        }];
        for backend in [
            super::MatvecBackend::Scalar,
            super::MatvecBackend::Auto,
            super::MatvecBackend::ExperimentalNeon,
        ] {
            let mut pinned = MatvecScratch::with_pinned_backend(backend);
            assert_eq!(
                pinned.select_pq2()(&weights, &activations)?.to_bits(),
                0xc1f9_0b01
            );
            // Public scalar/native-packed APIs retain the non-fused source order.
            let mut reference = MatvecScratch::with_backend(backend);
            assert_eq!(
                reference.select_pq2()(&weights, &activations)?.to_bits(),
                0xc1f9_0b02
            );
        }
        Ok(())
    }

    #[test]
    fn pinned_ptq1_accumulation_matches_independent_native_binary_capture() -> crate::Result<()> {
        let fixture = include_bytes!("../tests/fixtures/swift-ptq1-dot-parity.bin");
        let provenance: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/swift-ptq1-dot-parity.json"))?;
        let hash: String = Sha256::digest(fixture)
            .into_iter()
            .flat_map(|byte| [byte >> 4, byte & 15])
            .map(|nibble| char::from(b"0123456789abcdef"[usize::from(nibble)]))
            .collect();
        assert_eq!(Some(hash.as_str()), provenance["fixture_sha256"].as_str());
        assert_eq!(fixture.len(), 508);
        assert_eq!(
            provenance["runtime_commit"].as_str(),
            Some("73bfe6cc485554822f80c473a3c0c4f25e80f41b")
        );
        assert_eq!(
            provenance["ggml_cpu_sha256"].as_str(),
            Some("701b354a38b25e5c6dc57a1559f71cd26b2d0f8d69c1576a0e77a59a4273d2ab")
        );
        let mut remaining = fixture.as_slice();
        for case in 0..2 {
            let width = usize::try_from(u32::from_le_bytes(remaining[..4].try_into()?))?;
            let expected = u32::from_le_bytes(remaining[4..8].try_into()?);
            let weight_bytes = width / 128 * 28;
            let (weights, next) = remaining[8..].split_at(weight_bytes);
            let (activations, next) = next.split_at(width / 32 * 34);
            remaining = next;
            let input = activations
                .as_chunks::<34>()
                .0
                .iter()
                .map(|block| {
                    Ok(super::Q8Block {
                        scale: super::scale(block)?,
                        sum: 0.0,
                        values: std::array::from_fn(|index| i8::from_ne_bytes([block[index + 2]])),
                    })
                })
                .collect::<crate::Result<Vec<_>>>()?;
            for backend in [
                super::MatvecBackend::Scalar,
                super::MatvecBackend::Auto,
                super::MatvecBackend::ExperimentalNeon,
            ] {
                let mut pinned = MatvecScratch::with_pinned_backend(backend);
                assert_eq!(
                    pinned.select_q8_groups()(super::PTQ1, weights, &input)?.to_bits(),
                    expected
                );
                // Public source/reference APIs keep their unfused group reduction.
                let mut source = MatvecScratch::with_backend(backend);
                let source_bits =
                    source.select_q8_groups()(super::PTQ1, weights, &input)?.to_bits();
                assert_ne!(source_bits, expected);
                assert_eq!(
                    Some(u64::from(source_bits)),
                    provenance["fixtures"][case]["source_bits"].as_u64()
                );
            }
        }
        assert_eq!(remaining, b"");
        Ok(())
    }

    #[test]
    fn pinned_fused_rms_matches_independent_native_binary_capture() -> crate::Result<()> {
        // Captured from the pinned 73bfe6cc Windows ARM64 ggml RMS/MUL graph,
        // one thread, epsilon=1e-6; all installed DLL hashes were reverified.
        // ggml-cpu.dll SHA256:
        // 701b354a38b25e5c6dc57a1559f71cd26b2d0f8d69c1576a0e77a59a4273d2ab.
        // Inputs: Python random.Random(123), 16 binary32-rounded uniform values
        // in [-10, 10], followed by 16 binary32-rounded weights in [0.5, 1.5].
        // These expected bits are native outputs, not a Rust formula oracle.
        let input = [
            0xc10f_3e60,
            0xc104_19ab,
            0xbfed_760a,
            0xc0fb_1265,
            0x4100_6236,
            0xc113_ca74,
            0x3f39_5abd,
            0xc056_c975,
            0x40e1_55df,
            0xc0d9_d0ea,
            0xc050_5cd9,
            0xc054_bd99,
            0xc0a3_1870,
            0xc11f_7726,
            0xbfa3_224c,
            0xc103_f6b4,
        ]
        .map(f32::from_bits);
        let weight = [
            0x3f8c_7e24,
            0x3f11_e375,
            0x3f50_c1eb,
            0x3f72_c536,
            0x3fb3_e41a,
            0x3f17_bf6b,
            0x3f24_6596,
            0x3fa5_2113,
            0x3f05_6fb9,
            0x3fb4_a095,
            0x3f89_6570,
            0x3f43_ec20,
            0x3fab_3e44,
            0x3fa2_67de,
            0x3f57_ed23,
            0x3fa6_9771,
        ]
        .map(f32::from_bits);
        let native = [
            0xbfbf_e77d,
            0xbf37_c5b7,
            0xbe6c_5a32,
            0xbf91_4ecf,
            0x3fdc_3aa0,
            0xbf55_db96,
            0x3d91_490b,
            0xbf29_1af1,
            0x3f0f_5c43,
            0xbfbb_95d0,
            0xbf08_7efe,
            0xbec6_ba73,
            0xbf85_297f,
            0xbff6_f545,
            0xbe27_f2bd,
            0xbfd1_a270,
        ];
        let mut output = vec![99.0; 32];
        let pointer = output.as_ptr();
        rms_norm_pinned_into(&input, &weight, 1e-6, &mut output)?;
        assert_eq!(
            output
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            native
        );
        assert_eq!(output.as_ptr(), pointer);
        // The generic scalar API retains its documented source-order rounding.
        assert_eq!(rms_norm(&input, &weight, 1e-6)?[0].to_bits(), 0xbfbf_e77e);
        Ok(())
    }

    #[test]
    fn pinned_fused_rms_clears_nonfinite_intermediate_and_recovers_storage() -> crate::Result<()> {
        let mut output = vec![99.0; 32];
        let pointer = output.as_ptr();
        // Weight-first multiplication overflows even though the source-order
        // scalar result is finite. Reject this compiled-contract intermediate.
        assert!(rms_norm_pinned_into(&[2.0], &[f32::MAX], 0.0, &mut output).is_err());
        assert_eq!(output, [0.0; 0]);
        assert_eq!(output.as_ptr(), pointer);
        assert_eq!(
            rms_norm(&[2.0], &[f32::MAX], 0.0)?[0].to_bits(),
            f32::MAX.to_bits()
        );
        assert!(rms_norm_pinned_into(&[], &[], 0.0, &mut output).is_err());
        assert_eq!(output, [0.0; 0]);
        rms_norm_pinned_into(&[2.0], &[3.0], 0.0, &mut output)?;
        assert_eq!(output[0].to_bits(), 3.0_f32.to_bits());
        assert_eq!(output.as_ptr(), pointer);
        Ok(())
    }

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
        // The scalar-column projection does not touch decoded-row scratch.
        assert_eq!(scratch.row.len(), 2);
        assert_eq!(scratch.activations.as_ptr(), activation_pointer);
        assert_eq!(scratch.row.as_ptr(), row_pointer);
        assert_eq!(output, [6.0]);
        Ok(())
    }

    fn scalar_column_tensor(weights: &[f32]) -> Tensor {
        Tensor {
            name: "rank-one-columns".to_owned(),
            dimensions: vec![1, weights.len()],
            kind: super::F32,
            data: Arc::from(
                weights
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
        }
    }

    #[test]
    fn scalar_column_projection_matches_decoded_rows_without_row_storage() -> crate::Result<()> {
        let mut weights = vec![
            0.0,
            -0.0,
            f32::from_bits(1),
            -f32::from_bits(1),
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            1.0,
            -1.0,
            4097.0,
        ];
        // Exercise varying signs, exponents and mantissas without overflowing
        // for the finite inputs below. The expected route is the former row
        // decoder plus the same declared scalar accumulation contract.
        let mut bits = 0x1234_5678_u32;
        for _ in 0..1024 {
            bits = bits.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            weights.push(f32::from_bits(bits & 0xbf7f_ffff));
        }
        let tensor = scalar_column_tensor(&weights);
        let mut decoded = Vec::new();
        for pinned in [false, true] {
            let mut scratch = if pinned {
                MatvecScratch::with_pinned_backend(super::MatvecBackend::Scalar)
            } else {
                MatvecScratch::with_backend(super::MatvecBackend::Scalar)
            };
            let mut output = Vec::new();
            for input in [0.0, -0.0, f32::from_bits(1), -1.0, 0.5, 4097.0] {
                matvec_into(&tensor, &[input], &mut output, &mut scratch)?;
                for (index, actual) in output.iter().enumerate() {
                    super::row_into(&tensor, index, &mut decoded)?;
                    let expected = super::float_dot(&decoded, &[input]);
                    assert_eq!(actual.to_bits(), expected.to_bits());
                }
                assert_eq!(scratch.row.capacity(), 0);
                let diagnostics = scratch.diagnostics();
                assert!(diagnostics.dispatched);
                assert!(diagnostics.succeeded);
                assert_eq!(diagnostics.selected_backend, super::MatvecBackend::Scalar);
            }
        }
        Ok(())
    }

    #[test]
    fn scalar_column_projection_rejects_nonfinite_and_recovers_output() -> crate::Result<()> {
        let mut scratch = MatvecScratch::default();
        let mut output = vec![99.0; 3];
        let allocation = output.as_ptr();
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
            let tensor = scalar_column_tensor(&[1.0, invalid, 2.0]);
            assert!(matvec_into(&tensor, &[2.0], &mut output, &mut scratch).is_err());
            assert_eq!(output, Vec::<f32>::new());
            assert_eq!(output.as_ptr(), allocation);
            assert_eq!(scratch.row.capacity(), 0);
            assert!(scratch.diagnostics().dispatched);
            assert!(!scratch.diagnostics().succeeded);
        }
        let tensor = scalar_column_tensor(&[1.0, -0.0, f32::MAX]);
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matvec_into(&tensor, &[invalid], &mut output, &mut scratch).is_err());
            assert_eq!(output, Vec::<f32>::new());
            assert!(!scratch.diagnostics().dispatched);
        }
        matvec_into(&tensor, &[1.0], &mut output, &mut scratch)?;
        assert_eq!(output.len(), 3);
        assert_eq!(output[0].to_bits(), 1.0_f32.to_bits());
        assert_eq!(output[2].to_bits(), f32::MAX.to_bits());
        assert_eq!(output.as_ptr(), allocation);
        assert!(scratch.diagnostics().succeeded);
        Ok(())
    }

    fn bf16_tensor(columns: usize, rows: usize, weights: &[u16]) -> Tensor {
        Tensor {
            name: "streaming-bf16-projection".to_owned(),
            dimensions: vec![columns, rows],
            kind: super::BF16,
            data: Arc::from(
                weights
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
        }
    }

    #[test]
    fn bf16_projection_matches_decoded_rows_without_row_storage() -> crate::Result<()> {
        let codes = [
            0x0000, 0x8000, 0x0001, 0x8001, 0x0080, 0x8080, 0x3f80, 0xbf80, 0x4701, 0xc701,
        ];
        let mut decoded = Vec::new();
        for columns in [1, 3, 15, 16, 17, 32, 128, 5120] {
            let weights: Vec<_> = (0..columns * 7)
                .map(|index| codes[(index * 7 + index / columns) % codes.len()])
                .collect();
            let tensor = bf16_tensor(columns, 7, &weights);
            for pinned in [false, true] {
                for backend in [
                    super::MatvecBackend::Scalar,
                    super::MatvecBackend::Auto,
                    super::MatvecBackend::ExperimentalNeon,
                ] {
                    let mut scratch = if pinned {
                        MatvecScratch::with_pinned_backend(backend)
                    } else {
                        MatvecScratch::with_backend(backend)
                    };
                    let mut output = Vec::new();
                    for shift in 0..4 {
                        let input: Vec<_> = (0..columns)
                            .map(|index| {
                                [
                                    0.0,
                                    -0.0,
                                    f32::from_bits(1),
                                    -f32::MIN_POSITIVE,
                                    f32::from_bits(0x3f80_8001),
                                    -4097.0,
                                    0.5,
                                ][(index + shift) % 7]
                            })
                            .collect();
                        let prepared: Vec<_> =
                            input.iter().copied().map(super::round_bf16).collect();
                        matvec_into(&tensor, &input, &mut output, &mut scratch)?;
                        for (index, actual) in output.iter().enumerate() {
                            super::row_into(&tensor, index, &mut decoded)?;
                            let expected = super::float_dot(&decoded, &prepared);
                            assert_eq!(actual.to_bits(), expected.to_bits());
                        }
                        assert_eq!(scratch.row.capacity(), 0);
                        let diagnostics = scratch.diagnostics();
                        assert!(diagnostics.dispatched);
                        assert!(diagnostics.succeeded);
                        assert_eq!(diagnostics.selected_backend, super::MatvecBackend::Scalar);
                    }
                }
            }
        }
        let tensor = bf16_tensor(4, 1, &[0x7f7f, 0xff7f, 0x0001, 0x8000]);
        let input = [0.5, 0.5, f32::MIN_POSITIVE, -0.0];
        let mut scratch = MatvecScratch::default();
        let mut output = Vec::new();
        matvec_into(&tensor, &input, &mut output, &mut scratch)?;
        super::row_into(&tensor, 0, &mut decoded)?;
        assert_eq!(
            output[0].to_bits(),
            super::float_dot(&decoded, &input).to_bits()
        );
        assert_eq!(scratch.row.capacity(), 0);
        Ok(())
    }

    #[test]
    fn bf16_projection_rejects_bad_payload_input_and_geometry_then_recovers() -> crate::Result<()> {
        // Other floating formats may already own this storage. BF16 must not
        // mutate it, grow it or use stale decoded weights after a failed call.
        let mut scratch = MatvecScratch {
            row: vec![f32::NAN, -0.0, 17.0],
            ..MatvecScratch::default()
        };
        let row_pointer = scratch.row.as_ptr();
        let retained_bits: Vec<_> = scratch.row.iter().map(|value| value.to_bits()).collect();
        let mut output = vec![99.0; 3];
        let allocation = output.as_ptr();
        for invalid in [0x7fc1, 0x7f80, 0xff80, 0x7f7f] {
            let tensor = bf16_tensor(1, 3, &[0x3f80, invalid, 0x4000]);
            assert!(matvec_into(&tensor, &[2.0], &mut output, &mut scratch).is_err());
            assert_eq!(output, [0.0_f32; 0]);
            assert!(scratch.diagnostics().dispatched);
            assert!(!scratch.diagnostics().succeeded);
        }
        for invalid in [0x7fc1, 0x7f80, 0xff80] {
            let tensor = bf16_tensor(3, 1, &[0x3f80, 0x4000, invalid]);
            assert!(matvec_into(&tensor, &[1.0, 2.0, 0.0], &mut output, &mut scratch).is_err());
            assert_eq!(output, [0.0_f32; 0]);
        }
        let tensor = bf16_tensor(3, 1, &[0x3f80, 0x8000, 0x4000]);
        assert!(matvec_into(&tensor, &[1.0; 2], &mut output, &mut scratch).is_err());
        assert_eq!(output, [0.0_f32; 0]);
        assert!(!scratch.diagnostics().dispatched);
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
            assert!(matvec_into(&tensor, &[1.0, 0.0, invalid], &mut output, &mut scratch).is_err());
            assert_eq!(output, [0.0_f32; 0]);
            assert!(!scratch.diagnostics().dispatched);
        }
        let mut odd_payload = bf16_tensor(3, 1, &[0x3f80, 0x4000, 0x4040]);
        odd_payload.data = Arc::from([0_u8; 5]);
        for malformed in [
            bf16_tensor(3, 1, &[0x3f80, 0x4000]),
            bf16_tensor(2, 1, &[0x3f80, 0x4000, 0x4040]),
            bf16_tensor(0, 1, &[]),
            bf16_tensor(3, 0, &[]),
            odd_payload,
        ] {
            assert!(matvec_into(&malformed, &[1.0; 3], &mut output, &mut scratch).is_err());
            assert_eq!(output, [0.0_f32; 0]);
            assert!(!scratch.diagnostics().dispatched);
        }
        matvec_into(&tensor, &[1.0, -0.0, 2.0], &mut output, &mut scratch)?;
        assert_eq!(output, [5.0]);
        assert_eq!(output.as_ptr(), allocation);
        assert_eq!(scratch.row.as_ptr(), row_pointer);
        assert_eq!(
            scratch
                .row
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            retained_bits
        );
        assert!(scratch.diagnostics().succeeded);
        Ok(())
    }

    #[test]
    #[expect(
        clippy::suboptimal_flops,
        reason = "Test reference mirrors scalar non-fused multiply-then-add order."
    )]
    fn dot_ptq1_q8k_matches_dequantized_reference() -> crate::Result<()> {
        let mut weights = [0.0_f32; 256];
        for (i, w) in weights.iter_mut().enumerate() {
            *w = match i % 3 {
                0 => -0.5,
                1 => 0.0,
                _ => 0.5,
            };
        }
        let ptq1_bytes = super::quantize_row(super::PTQ1, &weights)?;
        let mut acts = [0.0_f32; 256];
        for (i, a) in acts.iter_mut().enumerate() {
            let signed = i32::try_from(i % 29).unwrap_or(0) - 14;
            *a = f32::from(i16::try_from(signed).unwrap_or(0)) * 0.125;
        }
        let mut q8k = Vec::new();
        q8k_into(&acts, &mut q8k)?;
        let actual = super::dot_ptq1_q8k(&ptq1_bytes, &q8k)?;
        let d0 = super::scale(&ptq1_bytes[26..28])? * q8k[0].scale;
        let d1 = super::scale(&ptq1_bytes[54..56])? * q8k[0].scale;
        let w0 = super::ptq1_values(&ptq1_bytes[..28])?;
        let w1 = super::ptq1_values(&ptq1_bytes[28..56])?;
        let s0 = super::dot_integer(&w0, &q8k[0].values[..128]);
        let s1 = super::dot_integer(&w1, &q8k[0].values[128..]);
        let expected = d0 * s0 + d1 * s1;
        assert!((actual - expected).abs() <= 1e-5);
        assert!(super::dot_ptq1_q8k(&ptq1_bytes[..27], &q8k).is_err());
        assert!(super::dot_ptq1_q8k(&ptq1_bytes, &[]).is_err());
        Ok(())
    }
}
