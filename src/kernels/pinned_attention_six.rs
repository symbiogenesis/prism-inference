//! Exact six-query-head sharing for Swift's 24-query/4-KV-head decode geometry.

use crate::{Result, invalid};

/// Six heads by 64 channels by sixteen temporal FMA lanes occupy 24 KiB.
pub(crate) const VALUE_TILE: usize = 64;

// Twice the query rows at half the channel width retain the 24 KiB bound.
#[path = "pinned_attention_pair_values.rs"]
mod pair_values;
pub(crate) use pair_values::{PAIR_VALUE_TILE, value_pair_tile_gated};

/// Reads and converts each cached F16 key element once across six query heads.
///
/// # Errors
/// Rejects empty, mismatched or oversized rows and nonfinite reductions.
pub(crate) fn dot(key: &[u16], queries: [&[f32]; 6]) -> Result<[f32; 6]> {
    dot_many(key, queries)
}

/// Shares one cached key conversion across both tokens of a six-head pair.
///
/// Twelve native-lane accumulators occupy 768 bytes, below the existing value
/// stage's 24 KiB high water. Register pressure and total benefit need separate
/// ARM64 code-generation and whole-request qualification.
///
/// # Errors
/// Rejects malformed rows and nonfinite reductions with the original lane tree.
pub(crate) fn dot_pair(key: &[u16], queries: [[&[f32]; 6]; 2]) -> Result<[[f32; 6]; 2]> {
    let [first, second] = queries;
    let values = dot_many::<12>(
        key,
        std::array::from_fn(|head| {
            if head < 6 {
                first[head]
            } else {
                second[head - 6]
            }
        }),
    )?;
    Ok(std::array::from_fn(|token| {
        std::array::from_fn(|head| values[token * 6 + head])
    }))
}

fn dot_many<const HEADS: usize>(key: &[u16], queries: [&[f32]; HEADS]) -> Result<[f32; HEADS]> {
    let len = key.len();
    if len == 0
        || len > crate::model::swift::CONTEXT_TOKENS
        || queries.iter().any(|query| query.len() != len)
    {
        return Err(invalid("invalid pinned six-head attention dot width"));
    }
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(neon) = fearless_simd::Level::baseline().as_neon() {
        return dot_neon(neon, key, queries);
    }
    let mut sums = [[0.0_f32; 16]; HEADS];
    for (index, &value) in key.iter().enumerate() {
        let value = bonsai_ternary::half_to_float(value);
        for (head, sum) in sums.iter_mut().enumerate() {
            sum[index % 16] = value.mul_add(queries[head][index], sum[index % 16]);
        }
    }
    for index in len..super::padded_dot_steps(len) {
        for sum in &mut sums {
            sum[index % 16] = 0.0_f32.mul_add(0.0, sum[index % 16]);
        }
    }
    let mut output = [0.0; HEADS];
    for (output, sum) in output.iter_mut().zip(sums) {
        *output = reduce(&sum)?;
    }
    Ok(output)
}

fn reduce(sum: &[f32; 16]) -> Result<f32> {
    let lanes: [f32; 4] =
        std::array::from_fn(|lane| (sum[lane] + sum[8 + lane]) + (sum[4 + lane] + sum[12 + lane]));
    crate::kernels::finite((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
}

/// Reduces one KV head for all six query heads, then applies original gates.
///
/// F16 probabilities and each channel's native sixteen-lane FMA tree remain
/// unchanged. All reductions succeed before any gated output is published.
///
/// # Errors
/// Rejects invalid extents, cache-row counts and nonfinite reductions before
/// publishing output. Gates retain the existing separate sigmoid arithmetic.
pub(crate) fn value_tile_gated<'a>(
    past: impl Iterator<Item = &'a [u16]>,
    current: &[f32],
    weights: [&[f32]; 6],
    offset: usize,
    outputs: [&mut [f32]; 6],
    gates: [&[f32]; 6],
) -> Result<()> {
    let width = current.len();
    let count = weights[0].len();
    if width == 0
        || width > VALUE_TILE
        || count == 0
        || count > crate::model::swift::CONTEXT_TOKENS
        || outputs.iter().any(|output| output.len() != width)
        || weights.iter().any(|weight| weight.len() != count)
        || gates.iter().any(|gate| gate.len() != width)
    {
        return Err(invalid("invalid pinned six-head attention value tile"));
    }
    let end = offset
        .checked_add(width)
        .ok_or_else(|| invalid("pinned attention value offset overflow"))?;
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if width.is_multiple_of(4)
        && let Some(neon) = fearless_simd::Level::baseline().as_neon()
    {
        return value_tile_neon(neon, past, current, weights, &(offset..end), outputs, gates);
    }
    value_tile_scalar(past, current, weights, &(offset..end), outputs, gates)
}

#[expect(
    clippy::large_stack_arrays,
    reason = "The fixed six-head temporal accumulator is bounded to 24 KiB."
)]
fn value_tile_scalar<'a>(
    mut past: impl Iterator<Item = &'a [u16]>,
    current: &[f32],
    weights: [&[f32]; 6],
    range: &std::ops::Range<usize>,
    outputs: [&mut [f32]; 6],
    gates: [&[f32]; 6],
) -> Result<()> {
    let mut sums = [[[0.0_f32; 16]; VALUE_TILE]; 6];
    let last = weights[0].len() - 1;
    for token in 0..last {
        let values = past
            .next()
            .and_then(|row| row.get(range.clone()))
            .ok_or_else(|| invalid("pinned attention value row is short"))?;
        for (channel, &value) in values.iter().enumerate() {
            let value = bonsai_ternary::half_to_float(value);
            for (head, sums) in sums.iter_mut().enumerate() {
                let lane = &mut sums[channel][token % 16];
                *lane = value.mul_add(weights[head][token], *lane);
            }
        }
    }
    if past.next().is_some() {
        return Err(invalid("pinned attention value row is long"));
    }
    for (head, sums) in sums.iter_mut().enumerate() {
        for (channel, &value) in current.iter().enumerate() {
            let lane = &mut sums[channel][last % 16];
            *lane = value.mul_add(weights[head][last], *lane);
        }
        for token in last + 1..super::padded_dot_steps(last + 1) {
            for channel in &mut sums[..current.len()] {
                let lane = &mut channel[token % 16];
                *lane = 0.0_f32.mul_add(0.0, *lane);
            }
        }
    }
    // Reuse dead first-lane accumulator storage for atomic publication.
    for sums in &mut sums {
        for channel in &mut sums[..current.len()] {
            channel[0] = reduce(channel)?;
        }
    }
    for (head, output) in outputs.into_iter().enumerate() {
        for (channel, output) in output.iter_mut().enumerate() {
            *output = sums[head][channel][0] * (1.0 / (1.0 + (-gates[head][channel]).exp()));
        }
    }
    Ok(())
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
#[path = "pinned_attention_six_neon.rs"]
mod neon;

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
use neon::{dot_neon, value_tile_neon};

#[cfg(test)]
#[path = "pinned_attention_six_tests.rs"]
mod tests;
