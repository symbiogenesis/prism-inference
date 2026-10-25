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

#[cfg(test)]
mod tests {
    use bonsai_ternary::half_to_float;
    use serde::Deserialize;

    use super::dot;

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
        value_half_bits: Vec<u16>,
        probability_half_bits: Vec<u16>,
        native_bits: u32,
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
            .probability_half_bits
            .into_iter()
            .map(half_to_float)
            .collect::<Vec<_>>();
        assert_eq!(
            dot(
                value.value_half_bits.into_iter().map(half_to_float),
                &probabilities
            )?
            .to_bits(),
            value.native_bits
        );
        assert!(dot([1.0].into_iter(), &[]).is_err());
        assert!(dot([].into_iter(), &[1.0]).is_err());
        assert!(dot([1.0, 2.0].into_iter(), &[1.0]).is_err());
        Ok(())
    }
}
