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
/// A contiguous cache-line-sized segment of binary16 value channels.
pub(crate) const VALUE_TILE: usize = 32;

#[cfg(test)]
#[path = "pinned_attention_padding_tests.rs"]
mod padding_tests;

#[cfg(test)]
#[path = "pinned_attention_tile_tests.rs"]
mod tile_tests;

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
#[path = "pinned_attention_neon.rs"]
mod neon;

const fn padded_dot_steps(logical: usize) -> usize {
    let padding = logical.div_ceil(CACHE_PADDING) * CACHE_PADDING - logical;
    // A zero FMA can turn -0 into +0. Each accumulator lane therefore keeps
    // its first masked FMA, but repeating it cannot change a finite result.
    // Nonfinite accumulators still fail the final finite-result check.
    logical + if padding < 16 { padding } else { 16 }
}

/// Reduces adjacent value channels without rereading the paged cache per channel.
///
/// Each channel retains [`dot`]'s exact FMA lanes, zero padding and reduction
/// tree. The fixed stack accumulator occupies 2 KiB; no cache is expanded.
///
/// # Errors
/// Rejects invalid tile/row geometry, unequal token counts and nonfinite results.
pub(crate) fn value_tile<'a>(
    past: impl Iterator<Item = &'a [u16]>,
    current: &[f32],
    weights: &[f32],
    offset: usize,
    output: &mut [f32],
) -> Result<()> {
    if output.is_empty()
        || output.len() > VALUE_TILE
        || current.len() != output.len()
        || weights.is_empty()
        || weights.len() > crate::model::swift::CONTEXT_TOKENS
    {
        return Err(invalid("invalid pinned attention value tile"));
    }
    let end = offset
        .checked_add(output.len())
        .ok_or_else(|| invalid("pinned attention value offset overflow"))?;

    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if output.len().is_multiple_of(4)
        && let Some(neon) = fearless_simd::Level::baseline().as_neon()
    {
        return neon::value_tile(neon, past, current, weights, &(offset..end), output);
    }

    value_tile_scalar(past, current, weights, &(offset..end), output)
}

fn value_tile_scalar<'a>(
    mut past: impl Iterator<Item = &'a [u16]>,
    current: &[f32],
    weights: &[f32],
    range: &std::ops::Range<usize>,
    output: &mut [f32],
) -> Result<()> {
    let mut sums = [[[0.0_f32; 4]; 4]; VALUE_TILE];
    let last = weights.len() - 1;
    for (index, &weight) in weights[..last].iter().enumerate() {
        let values = past
            .next()
            .and_then(|row| row.get(range.start..range.end))
            .ok_or_else(|| invalid("pinned attention value row is short"))?;
        for (sum, &value) in sums.iter_mut().zip(values) {
            let lane = &mut sum[index % 16 / 4][index % 4];
            *lane = bonsai_ternary::half_to_float(value).mul_add(weight, *lane);
        }
    }
    if past.next().is_some() {
        return Err(invalid("pinned attention value row is long"));
    }
    for (sum, &value) in sums.iter_mut().zip(current) {
        let lane = &mut sum[last % 16 / 4][last % 4];
        *lane = value.mul_add(weights[last], *lane);
    }
    // Retain the first masked FMA for each lane, including signed-zero effects.
    for index in weights.len()..padded_dot_steps(weights.len()) {
        for sum in &mut sums[..output.len()] {
            let lane = &mut sum[index % 16 / 4][index % 4];
            *lane = 0.0_f32.mul_add(0.0, *lane);
        }
    }
    for (out, sum) in output.iter_mut().zip(sums) {
        let mut lanes = [0.0; 4];
        for (index, lane) in lanes.iter_mut().enumerate() {
            *lane = (sum[0][index] + sum[2][index]) + (sum[1][index] + sum[3][index]);
        }
        *out = super::finite((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))?;
    }
    Ok(())
}

/// Reduces adjacent value channels across four query heads sharing the same KV head.
///
/// Each cached value row is read and converted from FP16 once per four query heads,
/// while preserving each head's independent 16-lane temporal FMA accumulator and
/// reduction tree.
///
/// # Errors
/// Rejects invalid tile/row geometry, unequal token counts and nonfinite results.
pub(crate) fn value_tile_four<'a>(
    past: impl Iterator<Item = &'a [u16]>,
    current: &[f32],
    weights: [&[f32]; 4],
    offset: usize,
    outputs: [&mut [f32]; 4],
) -> Result<()> {
    let out_len = outputs[0].len();
    let weight_len = weights[0].len();
    if out_len == 0
        || out_len > VALUE_TILE
        || current.len() != out_len
        || weight_len == 0
        || weight_len > crate::model::swift::CONTEXT_TOKENS
        || outputs.iter().any(|out| out.len() != out_len)
        || weights.iter().any(|w| w.len() != weight_len)
    {
        return Err(invalid("invalid pinned attention value tile"));
    }
    let end = offset
        .checked_add(out_len)
        .ok_or_else(|| invalid("pinned attention value offset overflow"))?;

    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if out_len.is_multiple_of(4)
        && let Some(neon) = fearless_simd::Level::baseline().as_neon()
    {
        return neon::value_tile_four(neon, past, current, weights, &(offset..end), outputs);
    }

    value_tile_four_scalar(past, current, weights, &(offset..end), outputs)
}

fn value_tile_four_scalar<'a>(
    mut past: impl Iterator<Item = &'a [u16]>,
    current: &[f32],
    weights: [&[f32]; 4],
    range: &std::ops::Range<usize>,
    outputs: [&mut [f32]; 4],
) -> Result<()> {
    let out_len = outputs[0].len();
    let weight_len = weights[0].len();
    let mut sums = [[[[0.0_f32; 4]; 4]; VALUE_TILE]; 4];
    let last = weight_len - 1;
    for index in 0..last {
        let values = past
            .next()
            .and_then(|row| row.get(range.start..range.end))
            .ok_or_else(|| invalid("pinned attention value row is short"))?;
        let r = index % 16 / 4;
        let c = index % 4;
        for (channel, &value) in values.iter().enumerate() {
            let decoded = bonsai_ternary::half_to_float(value);
            for h in 0..4 {
                let lane = &mut sums[h][channel][r][c];
                *lane = decoded.mul_add(weights[h][index], *lane);
            }
        }
    }
    if past.next().is_some() {
        return Err(invalid("pinned attention value row is long"));
    }
    let last_r = last % 16 / 4;
    let last_c = last % 4;
    for (channel, &value) in current.iter().enumerate() {
        for h in 0..4 {
            let lane = &mut sums[h][channel][last_r][last_c];
            *lane = value.mul_add(weights[h][last], *lane);
        }
    }
    for index in weight_len..padded_dot_steps(weight_len) {
        let r = index % 16 / 4;
        let c = index % 4;
        for h in 0..4 {
            for sum in &mut sums[h][..out_len] {
                let lane = &mut sum[r][c];
                *lane = 0.0_f32.mul_add(0.0, *lane);
            }
        }
    }
    let mut staged = [[0.0_f32; VALUE_TILE]; 4];
    for h in 0..4 {
        for (out, sum) in staged[h][..out_len].iter_mut().zip(&sums[h][..out_len]) {
            let mut lanes = [0.0; 4];
            for (index, lane) in lanes.iter_mut().enumerate() {
                *lane = (sum[0][index] + sum[2][index]) + (sum[1][index] + sum[3][index]);
            }
            *out = super::finite((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))?;
        }
    }
    for (output, stage) in outputs.into_iter().zip(&staged) {
        output.copy_from_slice(&stage[..out_len]);
    }
    Ok(())
}

/// Reduces a cached F16 key row and F32 query row with the pinned four-register FMA tree.
///
/// # Errors
/// Rejects empty, unequal or oversized rows and nonfinite results.
pub(crate) fn dot_f16(key: &[u16], query: &[f32]) -> Result<f32> {
    if query.is_empty() || query.len() > crate::model::swift::CONTEXT_TOKENS {
        return Err(invalid("invalid pinned attention dot width"));
    }
    if key.len() < query.len() {
        return Err(invalid("pinned attention dot row is short"));
    }
    if key.len() > query.len() {
        return Err(invalid("pinned attention dot row is long"));
    }
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(neon) = fearless_simd::Level::baseline().as_neon() {
        return neon::dot_f16(neon, key, query);
    }
    dot(
        key.iter().copied().map(bonsai_ternary::half_to_float),
        query,
    )
}

/// Reduces a current F32 key row and F32 query row with the pinned four-register FMA tree.
///
/// # Errors
/// Rejects empty, unequal or oversized rows and nonfinite results.
pub(crate) fn dot_f32(key: &[f32], query: &[f32]) -> Result<f32> {
    if query.is_empty() || query.len() > crate::model::swift::CONTEXT_TOKENS {
        return Err(invalid("invalid pinned attention dot width"));
    }
    if key.len() < query.len() {
        return Err(invalid("pinned attention dot row is short"));
    }
    if key.len() > query.len() {
        return Err(invalid("pinned attention dot row is long"));
    }
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(neon) = fearless_simd::Level::baseline().as_neon() {
        return neon::dot_f32(neon, key, query);
    }
    dot(key.iter().copied(), query)
}

/// Reduces a cached F16 key row against four F32 query rows sharing the same KV head.
///
/// # Errors
/// Rejects empty, unequal or oversized rows and nonfinite results.
pub(crate) fn dot_f16_four(key: &[u16], queries: [&[f32]; 4]) -> Result<[f32; 4]> {
    let len = queries[0].len();
    if len == 0
        || len > crate::model::swift::CONTEXT_TOKENS
        || queries.iter().any(|q| q.len() != len)
    {
        return Err(invalid("invalid pinned attention dot width"));
    }
    if key.len() < len {
        return Err(invalid("pinned attention dot row is short"));
    }
    if key.len() > len {
        return Err(invalid("pinned attention dot row is long"));
    }
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(neon) = fearless_simd::Level::baseline().as_neon() {
        return neon::dot_f16_four(neon, key, queries);
    }
    Ok([
        dot(
            key.iter().copied().map(bonsai_ternary::half_to_float),
            queries[0],
        )?,
        dot(
            key.iter().copied().map(bonsai_ternary::half_to_float),
            queries[1],
        )?,
        dot(
            key.iter().copied().map(bonsai_ternary::half_to_float),
            queries[2],
        )?,
        dot(
            key.iter().copied().map(bonsai_ternary::half_to_float),
            queries[3],
        )?,
    ])
}

/// Reduces a current F32 key row against four F32 query rows sharing the same KV head.
///
/// # Errors
/// Rejects empty, unequal or oversized rows and nonfinite results.
pub(crate) fn dot_f32_four(key: &[f32], queries: [&[f32]; 4]) -> Result<[f32; 4]> {
    let len = queries[0].len();
    if len == 0
        || len > crate::model::swift::CONTEXT_TOKENS
        || queries.iter().any(|q| q.len() != len)
    {
        return Err(invalid("invalid pinned attention dot width"));
    }
    if key.len() < len {
        return Err(invalid("pinned attention dot row is short"));
    }
    if key.len() > len {
        return Err(invalid("pinned attention dot row is long"));
    }
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(neon) = fearless_simd::Level::baseline().as_neon() {
        return neon::dot_f32_four(neon, key, queries);
    }
    Ok([
        dot(key.iter().copied(), queries[0])?,
        dot(key.iter().copied(), queries[1])?,
        dot(key.iter().copied(), queries[2])?,
        dot(key.iter().copied(), queries[3])?,
    ])
}

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
    let mut sums = [[0.0_f32; 4]; 4];
    for index in 0..padded_dot_steps(weights.len()) {
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
    let mut total = 0.0_f64;
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(neon) = fearless_simd::Level::baseline().as_neon() {
        use fearless_simd::{f32x4, prelude::*};
        let max_vec = f32x4::splat(neon, maximum);
        let (chunks, tail) = scores.as_chunks_mut::<4>();
        for chunk in chunks {
            let exp = super::pinned_activations::vector_exp_x4(
                neon,
                f32x4::from_slice(neon, chunk) - max_vec,
            );
            exp.store_slice(chunk);
            total += f64::from((chunk[0] + chunk[1]) + (chunk[2] + chunk[3]));
        }
        if !tail.is_empty() {
            let mut group = [0.0; 4];
            for (score, output) in tail.iter_mut().zip(&mut group) {
                *score = super::pinned_activations::vector_exp(*score - maximum);
                *output = *score;
            }
            total += f64::from((group[0] + group[1]) + (group[2] + group[3]));
        }
        let inverse = (1.0_f64 / total) as f32;
        let inv_vec = f32x4::splat(neon, inverse);
        let (chunks, tail) = scores.as_chunks_mut::<4>();
        for chunk in chunks {
            (f32x4::from_slice(neon, chunk) * inv_vec).store_slice(chunk);
        }
        for score in tail {
            *score *= inverse;
        }
        return Ok(());
    }
    // Keep the final partial group's exact pairwise tree. Later physical
    // groups add only +0 to the positive total and cannot change its bits.
    for offset in (0..scores.len()).step_by(4) {
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
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let (Some(neon), Ok(freq_array)) = (
        fearless_simd::Level::baseline().as_neon(),
        <&[(f32, f32); 32]>::try_from(frequencies),
    ) {
        neon::rope_with_aliasing(neon, input, freq_array, in_place);
        if !super::is_finite_slice(input) {
            return Err(invalid("nonfinite pinned Swift RoPE output"));
        }
        return Ok(());
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
    if !super::is_finite_slice(input) {
        return Err(invalid("nonfinite pinned Swift RoPE output"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use bonsai_ternary::half_to_float;
    use serde::Deserialize;

    use super::{
        VALUE_TILE, dot, dot_f16, dot_f16_four, dot_f32, dot_f32_four, softmax, value_tile,
        value_tile_four,
    };

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
            let first = case
                .first_bits
                .into_iter()
                .map(f32::from_bits)
                .collect::<Vec<_>>();
            let second = case
                .second_bits
                .into_iter()
                .map(f32::from_bits)
                .collect::<Vec<_>>();
            assert_eq!(
                dot(first.iter().copied(), &second)?.to_bits(),
                case.native_bits,
                "{}",
                case.name
            );
            assert_eq!(
                dot_f32(&first, &second)?.to_bits(),
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
            dot(
                value.values.iter().copied().map(half_to_float),
                &probabilities
            )?
            .to_bits(),
            value.expected
        );
        assert_eq!(
            dot_f16(&value.values, &probabilities)?.to_bits(),
            value.expected
        );
        assert!(dot([1.0].into_iter(), &[]).is_err());
        assert!(dot([].into_iter(), &[1.0]).is_err());
        assert!(dot([1.0, 2.0].into_iter(), &[1.0]).is_err());
        assert!(dot_f32(&[1.0], &[]).is_err());
        assert!(dot_f32(&[], &[1.0]).is_err());
        assert!(dot_f32(&[1.0, 2.0], &[1.0]).is_err());
        assert!(dot_f16(&[0x3c00], &[]).is_err());
        assert!(dot_f16(&[], &[1.0]).is_err());
        assert!(dot_f16(&[0x3c00, 0x4000], &[1.0]).is_err());
        Ok(())
    }

    #[test]
    fn pinned_value_tiles_match_native_capture_and_visit_each_cached_row_once() -> crate::Result<()>
    {
        let capture: Capture = serde_json::from_str(include_str!(
            "../../tests/fixtures/swift-attention-dot.json"
        ))?;
        let values = capture.weighted_value;
        let rows: Vec<[u16; VALUE_TILE]> = values.values[..values.values.len() - 1]
            .iter()
            .map(|&value| [value; VALUE_TILE])
            .collect();
        let current = [half_to_float(values.values[values.values.len() - 1]); VALUE_TILE];
        let probabilities: Vec<f32> = values
            .probabilities
            .into_iter()
            .map(half_to_float)
            .collect();
        let visits = std::cell::Cell::new(0);
        let past = rows.iter().map(<[u16; VALUE_TILE]>::as_slice).inspect(|_| {
            visits.set(visits.get() + 1);
        });
        let mut output = [f32::NAN; VALUE_TILE];
        value_tile(past, &current, &probabilities, 0, &mut output)?;
        assert!(
            output
                .iter()
                .all(|value| value.to_bits() == values.expected)
        );
        assert_eq!(visits.get(), rows.len());
        Ok(())
    }

    #[test]
    fn pinned_value_tiles_reject_malformed_extents_and_nonfinite_results() {
        let row = [0x3c00_u16, 0x4000];
        let mut output = [19.0];
        assert!(value_tile([row.as_slice()].into_iter(), &[1.0], &[1.0], 0, &mut output).is_err());
        assert!(value_tile([].into_iter(), &[1.0], &[1.0, 1.0], 0, &mut output).is_err());
        assert!(
            value_tile(
                [row.as_slice()].into_iter(),
                &[1.0],
                &[1.0, 1.0],
                2,
                &mut output
            )
            .is_err()
        );
        assert!(value_tile([].into_iter(), &[], &[1.0], 0, &mut output).is_err());
        assert!(value_tile([].into_iter(), &[1.0], &[], 0, &mut output).is_err());
        assert!(value_tile([].into_iter(), &[1.0], &[1.0], usize::MAX, &mut output).is_err());
        assert!(value_tile([].into_iter(), &[f32::MAX], &[2.0], 0, &mut output).is_err());
        assert!(value_tile([].into_iter(), &[f32::NAN], &[1.0], 0, &mut output).is_err());
        assert_eq!(output, [19.0]);
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

    #[test]
    fn grouped_four_head_dot_and_value_tile_match_single_head_bits_and_visit_rows_once()
    -> crate::Result<()> {
        let capture: Capture = serde_json::from_str(include_str!(
            "../../tests/fixtures/swift-attention-dot.json"
        ))?;
        let values = capture.weighted_value;
        let q0: Vec<f32> = values
            .probabilities
            .iter()
            .copied()
            .map(half_to_float)
            .collect();
        let q1: Vec<f32> = q0
            .iter()
            .enumerate()
            .map(|(i, &x)| x * (1.0 + (i % 5) as f32 * 0.125))
            .collect();
        let q2: Vec<f32> = q0
            .iter()
            .enumerate()
            .map(|(i, &x)| x * (0.75 - (i % 3) as f32 * 0.25))
            .collect();
        let q3: Vec<f32> = q0.iter().copied().rev().collect();
        let key_f32: Vec<f32> = values.values.iter().copied().map(half_to_float).collect();

        let expected_f16 = [
            dot_f16(&values.values, &q0)?.to_bits(),
            dot_f16(&values.values, &q1)?.to_bits(),
            dot_f16(&values.values, &q2)?.to_bits(),
            dot_f16(&values.values, &q3)?.to_bits(),
        ];
        let actual_f16 = dot_f16_four(&values.values, [&q0, &q1, &q2, &q3])?.map(f32::to_bits);
        assert_eq!(actual_f16, expected_f16);

        let expected_f32 = [
            dot_f32(&key_f32, &q0)?.to_bits(),
            dot_f32(&key_f32, &q1)?.to_bits(),
            dot_f32(&key_f32, &q2)?.to_bits(),
            dot_f32(&key_f32, &q3)?.to_bits(),
        ];
        let actual_f32 = dot_f32_four(&key_f32, [&q0, &q1, &q2, &q3])?.map(f32::to_bits);
        assert_eq!(actual_f32, expected_f32);

        let rows: Vec<[u16; VALUE_TILE]> = values.values[..values.values.len() - 1]
            .iter()
            .enumerate()
            .map(|(r, &value)| std::array::from_fn(|c| value.wrapping_add(((r + c) % 7) as u16)))
            .collect();
        let current: [f32; VALUE_TILE] = std::array::from_fn(|c| {
            half_to_float(values.values[values.values.len() - 1]) + (c as f32) * 0.03125
        });

        for tile_width in [VALUE_TILE, 12, 7] {
            let mut single = [[0.0_f32; VALUE_TILE]; 4];
            for (h, weights) in [&q0, &q1, &q2, &q3].into_iter().enumerate() {
                value_tile(
                    rows.iter().map(<[u16; VALUE_TILE]>::as_slice),
                    &current[..tile_width],
                    weights,
                    0,
                    &mut single[h][..tile_width],
                )?;
            }

            let visits = std::cell::Cell::new(0);
            let past = rows.iter().map(<[u16; VALUE_TILE]>::as_slice).inspect(|_| {
                visits.set(visits.get() + 1);
            });
            let mut out0 = vec![f32::NAN; tile_width];
            let mut out1 = vec![f32::NAN; tile_width];
            let mut out2 = vec![f32::NAN; tile_width];
            let mut out3 = vec![f32::NAN; tile_width];
            value_tile_four(
                past,
                &current[..tile_width],
                [&q0, &q1, &q2, &q3],
                0,
                [&mut out0, &mut out1, &mut out2, &mut out3],
            )?;
            assert_eq!(visits.get(), rows.len());
            for (actual, expected) in [&out0, &out1, &out2, &out3].into_iter().zip(&single) {
                let actual_bits: Vec<u32> = actual.iter().copied().map(f32::to_bits).collect();
                let expected_bits: Vec<u32> = expected[..tile_width]
                    .iter()
                    .copied()
                    .map(f32::to_bits)
                    .collect();
                assert_eq!(actual_bits, expected_bits, "tile_width={tile_width}");
            }
        }
        Ok(())
    }
}
