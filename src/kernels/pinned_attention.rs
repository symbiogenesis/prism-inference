//! Safe reproduction of the pinned Windows ARM64 nonflash attention arithmetic.
//!
//! Reference: Prism `73bfe6cc485554822f80c473a3c0c4f25e80f41b`,
//! `ggml/src/ggml-cpu/vec.cpp`, `simd-mappings.h` and `src/llama-kv-cache-73.cpp`.
//! Upstream MIT terms are retained in `PRISM-LICENSE.txt`.

#![allow(
    clippy::redundant_pub_crate,
    reason = "Keep exact-model attention helpers crate scoped."
)]

use crate::{Result, invalid};

const CACHE_PADDING: usize = 256;

/// Reduces decoded F16 operands with the pinned four-register F32 FMA tree.
///
/// Masked cache lanes are zero-filled through the native 256-token padding.
/// The caller supplies already-rounded F16 operands, including probabilities.
///
/// # Errors
/// Rejects empty, unequal or oversized rows and nonfinite results.
pub(crate) fn dot(mut values: impl Iterator<Item = f32>, weights: &[f32]) -> Result<f32> {
    if weights.is_empty() || weights.len() > crate::model::swift::CONTEXT_TOKENS {
        return Err(invalid("invalid pinned attention dot width"));
    }
    let physical = weights.len().div_ceil(CACHE_PADDING) * CACHE_PADDING;
    let mut sums = [[0.0_f32; 4]; 4];
    for index in 0..physical {
        let (value, weight) = if index < weights.len() {
            (
                values
                    .next()
                    .ok_or_else(|| invalid("pinned attention dot row is short"))?,
                weights[index],
            )
        } else {
            (0.0, 0.0)
        };
        let sum = &mut sums[index % 16 / 4][index % 4];
        *sum = value.mul_add(weight, *sum);
    }
    if values.next().is_some() {
        return Err(invalid("pinned attention dot row is long"));
    }
    let mut lanes = [0.0; 4];
    for (index, lane) in lanes.iter_mut().enumerate() {
        *lane = (sums[0][index] + sums[2][index]) + (sums[1][index] + sums[3][index]);
    }
    super::finite((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
}

/// Normalizes logical scores as the pinned physically padded masked row.
///
/// Native groups use four vector exponentials and a binary32 pairwise sum;
/// their totals and reciprocal are evaluated in binary64 before final scaling.
/// Masked padding contributes exact zeros without allocating padded storage.
///
/// # Errors
/// Rejects empty, oversized or nonfinite input before mutation.
#[allow(
    clippy::cast_possible_truncation,
    clippy::as_conversions,
    reason = "Match the native binary64 reciprocal rounded once to binary32."
)]
pub(crate) fn softmax(scores: &mut [f32]) -> Result<()> {
    if scores.is_empty()
        || scores.len() > crate::model::swift::CONTEXT_TOKENS
        || scores.iter().any(|score| !score.is_finite())
    {
        return Err(invalid("invalid pinned attention softmax row"));
    }
    let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let physical = scores.len().div_ceil(CACHE_PADDING) * CACHE_PADDING;
    let mut total = 0.0_f64;
    for offset in (0..physical).step_by(4) {
        let mut group = [0.0; 4];
        for (lane, output) in group.iter_mut().enumerate() {
            if let Some(score) = scores.get_mut(offset + lane) {
                *score = super::pinned_activations::vector_exp(*score - maximum);
                *output = *score;
            }
        }
        total += f64::from((group[0] + group[1]) + (group[2] + group[3]));
    }
    let inverse = (1.0_f64 / total) as f32;
    for score in scores {
        *score *= inverse;
    }
    Ok(())
}

/// Retains the pinned out-of-place operator oracle for focused comparisons.
///
/// # Errors
/// Rejects geometry or frequency drift before mutation and nonfinite results.
#[cfg(test)]
pub(crate) fn rope(
    input: &mut [f32],
    head: usize,
    rotary: usize,
    frequencies: &[(f32, f32)],
) -> Result<()> {
    rope_with_aliasing(input, head, rotary, frequencies, false)
}

/// Rotates Swift heads with the compiled native allocator's in-place ordering.
///
/// Native buffer reuse changes the first FMA's operand order. The second
/// component retains the same ordering as the out-of-place control.
///
/// # Errors
/// Rejects geometry or frequency drift before mutation and nonfinite results.
pub(crate) fn rope_in_place(
    input: &mut [f32],
    head: usize,
    rotary: usize,
    frequencies: &[(f32, f32)],
) -> Result<()> {
    rope_with_aliasing(input, head, rotary, frequencies, true)
}

fn rope_with_aliasing(
    input: &mut [f32],
    head: usize,
    rotary: usize,
    frequencies: &[(f32, f32)],
    in_place: bool,
) -> Result<()> {
    if head != 256
        || rotary != 64
        || input.is_empty()
        || !input.len().is_multiple_of(head)
        || frequencies.len() != rotary / 2
        || frequencies
            .iter()
            .any(|&(cosine, sine)| !cosine.is_finite() || !sine.is_finite())
    {
        return Err(invalid("invalid pinned Swift RoPE geometry or frequencies"));
    }
    for head in input.chunks_exact_mut(head) {
        for (index, &(cosine, sine)) in frequencies.iter().enumerate() {
            let first = head[index];
            let second = head[index + rotary / 2];
            head[index] = if in_place {
                first.mul_add(cosine, -(second * sine))
            } else {
                (-second).mul_add(sine, first * cosine)
            };
            head[index + rotary / 2] = second.mul_add(cosine, first * sine);
        }
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(invalid("nonfinite pinned Swift RoPE output"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use bonsai_ternary::half_to_float;
    use serde::Deserialize;

    use super::{dot, softmax};

    #[derive(Deserialize)]
    struct Capture {
        runtime_commit: String,
        ggml_cpu_sha256: String,
        cases: Vec<Case>,
        weighted_value: WeightedValue,
    }

    #[derive(Deserialize)]
    struct Case {
        name: String,
        first_bits: Vec<u32>,
        second_bits: Vec<u32>,
        native_bits: u32,
    }

    #[derive(Deserialize)]
    struct WeightedValue {
        #[serde(rename = "value_half_bits")]
        values: Vec<u16>,
        #[serde(rename = "probability_half_bits")]
        probabilities: Vec<u16>,
        #[serde(rename = "native_bits")]
        expected: u32,
    }

    #[derive(Deserialize)]
    struct SoftmaxCapture {
        runtime_commit: String,
        ggml_cpu_sha256: String,
        cases: Vec<SoftmaxCase>,
    }

    #[derive(Deserialize)]
    struct SoftmaxCase {
        valid_keys: usize,
        input_bits: Vec<u32>,
        native_bits: Vec<u32>,
    }

    #[test]
    fn pinned_attention_dot_matches_native_f16_rows_and_padded_cancellation() -> crate::Result<()> {
        let capture: Capture = serde_json::from_str(include_str!(
            "../../tests/fixtures/swift-attention-dot.json"
        ))?;
        assert_eq!(
            capture.runtime_commit,
            "73bfe6cc485554822f80c473a3c0c4f25e80f41b"
        );
        assert_eq!(
            capture.ggml_cpu_sha256,
            "701b354a38b25e5c6dc57a1559f71cd26b2d0f8d69c1576a0e77a59a4273d2ab"
        );
        for case in capture.cases {
            let first = case.first_bits.into_iter().map(f32::from_bits);
            let second = case
                .second_bits
                .into_iter()
                .map(f32::from_bits)
                .collect::<Vec<_>>();
            assert_eq!(
                dot(first, &second)?.to_bits(),
                case.native_bits,
                "{}",
                case.name
            );
        }
        let value = capture.weighted_value;
        let probabilities = value
            .probabilities
            .into_iter()
            .map(half_to_float)
            .collect::<Vec<_>>();
        assert_eq!(
            dot(value.values.into_iter().map(half_to_float), &probabilities)?.to_bits(),
            value.expected
        );
        assert!(dot([1.0].into_iter(), &[]).is_err());
        assert!(dot([].into_iter(), &[1.0]).is_err());
        assert!(dot([1.0, 2.0].into_iter(), &[1.0]).is_err());
        Ok(())
    }

    #[test]
    fn pinned_attention_softmax_matches_native_masked_physical_rows() -> crate::Result<()> {
        let capture: SoftmaxCapture = serde_json::from_str(include_str!(
            "../../tests/fixtures/swift-attention-softmax.json"
        ))?;
        assert_eq!(
            capture.runtime_commit,
            "73bfe6cc485554822f80c473a3c0c4f25e80f41b"
        );
        assert_eq!(
            capture.ggml_cpu_sha256,
            "701b354a38b25e5c6dc57a1559f71cd26b2d0f8d69c1576a0e77a59a4273d2ab"
        );
        for case in capture.cases {
            let mut scores = case
                .input_bits
                .into_iter()
                .map(f32::from_bits)
                .collect::<Vec<_>>();
            assert_eq!(scores.len(), case.valid_keys);
            softmax(&mut scores)?;
            assert_eq!(
                scores.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                case.native_bits,
                "{} valid keys",
                case.valid_keys
            );
        }
        assert!(softmax(&mut []).is_err());
        let mut invalid = [1.0, f32::INFINITY];
        assert!(softmax(&mut invalid).is_err());
        assert_eq!(
            invalid.map(f32::to_bits),
            [1.0_f32.to_bits(), f32::INFINITY.to_bits()]
        );
        Ok(())
    }
}
