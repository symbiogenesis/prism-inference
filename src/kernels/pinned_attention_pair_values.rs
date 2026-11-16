//! Share cached value conversion across two causal six-head prompt tokens.

use crate::{Result, invalid};

/// Twelve heads by 32 channels by sixteen temporal FMA lanes occupy 24 KiB.
pub(crate) const PAIR_VALUE_TILE: usize = 32;

/// Reduce both prompt rows with one conversion of their common cached prefix.
///
/// Each row retains its own globally normalized F16 probabilities and native
/// temporal lanes. The first token's current value is F32 for that token and
/// cached F16 for the second. Every reduction succeeds before tile publication.
///
/// # Errors
/// Rejects malformed extents, short/long cache iterators and nonfinite reductions.
#[allow(
    dead_code,
    reason = "Retain unpacked F32 paired value tile baseline for direct parity tests."
)]
pub(crate) fn value_pair_tile_gated<'a>(
    past: impl Iterator<Item = &'a [u16]>,
    current: [&[f32]; 2],
    weights: [[&[f32]; 6]; 2],
    offset: usize,
    outputs: [[&mut [f32]; 6]; 2],
    gates: [[&[f32]; 6]; 2],
) -> Result<()> {
    let width = current[0].len();
    let first_count = weights[0][0].len();
    if width == 0
        || width > PAIR_VALUE_TILE
        || current[1].len() != width
        || first_count == 0
        || first_count >= crate::model::swift::CONTEXT_TOKENS
        || outputs.iter().flatten().any(|row| row.len() != width)
        || gates.iter().flatten().any(|row| row.len() != width)
        || weights[0].iter().any(|row| row.len() != first_count)
        || weights[1].iter().any(|row| row.len() != first_count + 1)
    {
        return Err(invalid("invalid pinned paired attention value tile"));
    }
    let end = offset
        .checked_add(width)
        .ok_or_else(|| invalid("pinned paired attention value offset overflow"))?;
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
        return neon::values(neon, past, current, weights, &(offset..end), outputs, gates);
    }
    scalar(past, current, weights, &(offset..end), outputs, gates)
}

/// Reduces both prompt rows with one conversion of their common cached prefix
/// using `[token][head]` packed binary16 probabilities.
///
/// # Errors
/// Rejects malformed extents, nonfinite inputs, short/long cache iterators and
/// nonfinite reductions before publishing any output.
pub(crate) fn value_pair_tile_packed_gated<'a>(
    past: impl Iterator<Item = &'a [u16]>,
    current: [&[f32]; 2],
    weights: [&[[u16; 6]]; 2],
    offset: usize,
    outputs: [[&mut [f32]; 6]; 2],
    gates: [[&[f32]; 6]; 2],
) -> Result<()> {
    let width = current[0].len();
    let first_count = weights[0].len();
    if width == 0
        || width > PAIR_VALUE_TILE
        || current[1].len() != width
        || first_count == 0
        || first_count >= crate::model::swift::CONTEXT_TOKENS
        || weights[1].len() != first_count + 1
        || outputs.iter().flatten().any(|row| row.len() != width)
        || gates.iter().flatten().any(|row| row.len() != width)
        || weights
            .iter()
            .flat_map(|rows| rows.iter().flatten())
            .any(|&bits| (bits & 0x7c00) == 0x7c00)
        || current
            .iter()
            .any(|row| row.iter().any(|value| !value.is_finite()))
    {
        return Err(invalid("invalid pinned paired packed attention value tile"));
    }
    let end = offset
        .checked_add(width)
        .ok_or_else(|| invalid("pinned paired attention value offset overflow"))?;
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
        return neon::values_packed(neon, past, current, weights, &(offset..end), outputs, gates);
    }
    scalar_packed(past, current, weights, &(offset..end), outputs, gates)
}

/// Reduces both prompt rows with one conversion of their common cached prefix
/// using chunked stack-packed `[token][head]` binary16 probabilities from
/// normalized `f32` score slices.
///
/// # Errors
/// Rejects malformed extents, nonfinite inputs, short/long cache iterators and
/// nonfinite reductions before publishing any output.
pub(crate) fn value_pair_tile_from_scores_gated<'a>(
    past: impl Iterator<Item = &'a [u16]>,
    current: [&[f32]; 2],
    scores: [[&[f32]; 6]; 2],
    offset: usize,
    outputs: [[&mut [f32]; 6]; 2],
    gates: [[&[f32]; 6]; 2],
) -> Result<()> {
    let width = current[0].len();
    let first_count = scores[0][0].len();
    if width == 0
        || width > PAIR_VALUE_TILE
        || current[1].len() != width
        || first_count == 0
        || first_count >= crate::model::swift::CONTEXT_TOKENS
        || outputs.iter().flatten().any(|row| row.len() != width)
        || gates.iter().flatten().any(|row| row.len() != width)
        || scores[0]
            .iter()
            .any(|row| row.len() != first_count || row.iter().any(|value| !value.is_finite()))
        || scores[1]
            .iter()
            .any(|row| row.len() != first_count + 1 || row.iter().any(|value| !value.is_finite()))
        || current
            .iter()
            .any(|row| row.iter().any(|value| !value.is_finite()))
    {
        return Err(invalid("invalid pinned paired score attention value tile"));
    }
    let end = offset
        .checked_add(width)
        .ok_or_else(|| invalid("pinned paired attention value offset overflow"))?;
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
        return neon::values_from_scores(
            neon,
            past,
            current,
            scores,
            &(offset..end),
            outputs,
            gates,
        );
    }
    scalar_from_scores(past, current, scores, &(offset..end), outputs, gates)
}

pub(super) struct PairTailWeights {
    pub(super) first_last: usize,
    pub(super) last0: [u16; 6],
    pub(super) prev1: [u16; 6],
    pub(super) last1: [u16; 6],
}

fn accumulate_pair_packed_past_scalar<'a>(
    sums: &mut [[[[f32; 16]; PAIR_VALUE_TILE]; 6]; 2],
    past: &mut impl Iterator<Item = &'a [u16]>,
    w0: &[[u16; 6]],
    w1: &[[u16; 6]],
    base_position: usize,
    range: &std::ops::Range<usize>,
) -> Result<()> {
    for (local, (w0_bits, w1_bits)) in w0.iter().zip(w1).enumerate() {
        let values = next_row(past, range)?;
        let lane = (base_position + local) % 16;
        let w0_f32 = w0_bits.map(bonsai_ternary::half_to_float);
        let w1_f32 = w1_bits.map(bonsai_ternary::half_to_float);
        for (channel, &bits) in values.iter().enumerate() {
            let value = bonsai_ternary::half_to_float(bits);
            for (head_sums, &weight) in sums[0].iter_mut().zip(&w0_f32) {
                head_sums[channel][lane] = value.mul_add(weight, head_sums[channel][lane]);
            }
            for (head_sums, &weight) in sums[1].iter_mut().zip(&w1_f32) {
                head_sums[channel][lane] = value.mul_add(weight, head_sums[channel][lane]);
            }
        }
    }
    Ok(())
}

fn finish_pair_packed_scalar<'a>(
    sums: &mut [[[[f32; 16]; PAIR_VALUE_TILE]; 6]; 2],
    mut past: impl Iterator<Item = &'a [u16]>,
    current: [&[f32]; 2],
    tail: &PairTailWeights,
    range: &std::ops::Range<usize>,
    outputs: [[&mut [f32]; 6]; 2],
    gates: [[&[f32]; 6]; 2],
) -> Result<()> {
    let previous = next_row(&mut past, range)?;
    if past.next().is_some() {
        return Err(invalid("pinned paired attention value row is long"));
    }
    let prev1_f32 = tail.prev1.map(bonsai_ternary::half_to_float);
    let prev_lane = tail.first_last % 16;
    for (channel, &bits) in previous.iter().enumerate() {
        let value = bonsai_ternary::half_to_float(bits);
        for (head_sums, &weight) in sums[1].iter_mut().zip(&prev1_f32) {
            head_sums[channel][prev_lane] = value.mul_add(weight, head_sums[channel][prev_lane]);
        }
    }
    let last_weights = [
        tail.last0.map(bonsai_ternary::half_to_float),
        tail.last1.map(bonsai_ternary::half_to_float),
    ];
    for (token, token_sums) in sums.iter_mut().enumerate() {
        let last = tail.first_last + token;
        let lane = last % 16;
        for (head_sums, &weight) in token_sums.iter_mut().zip(&last_weights[token]) {
            for (channel, &value) in current[token].iter().enumerate() {
                head_sums[channel][lane] = value.mul_add(weight, head_sums[channel][lane]);
            }
            for position in last + 1..super::super::padded_dot_steps(last + 1) {
                for channel in &mut head_sums[..current[token].len()] {
                    let pad = &mut channel[position % 16];
                    *pad = 0.0_f32.mul_add(0.0, *pad);
                }
            }
        }
    }
    for token_sums in &mut *sums {
        for head_sums in token_sums {
            for channel in &mut head_sums[..current[0].len()] {
                channel[0] = super::reduce(channel)?;
            }
        }
    }
    for (token, token_outputs) in outputs.into_iter().enumerate() {
        for (head, output) in token_outputs.into_iter().enumerate() {
            for (channel, out) in output.iter_mut().enumerate() {
                *out = sums[token][head][channel][0]
                    * (1.0 / (1.0 + (-gates[token][head][channel]).exp()));
            }
        }
    }
    Ok(())
}

#[expect(
    clippy::large_stack_arrays,
    reason = "Twelve query rows use half-width channel tiles to retain the existing 24 KiB bound."
)]
fn scalar_packed<'a>(
    mut past: impl Iterator<Item = &'a [u16]>,
    current: [&[f32]; 2],
    weights: [&[[u16; 6]]; 2],
    range: &std::ops::Range<usize>,
    outputs: [[&mut [f32]; 6]; 2],
    gates: [[&[f32]; 6]; 2],
) -> Result<()> {
    let mut sums = [[[[0.0_f32; 16]; PAIR_VALUE_TILE]; 6]; 2];
    let first_last = weights[0].len() - 1;
    accumulate_pair_packed_past_scalar(
        &mut sums,
        &mut past,
        &weights[0][..first_last],
        &weights[1][..first_last],
        0,
        range,
    )?;
    let tail = PairTailWeights {
        first_last,
        last0: weights[0][first_last],
        prev1: weights[1][first_last],
        last1: weights[1][first_last + 1],
    };
    finish_pair_packed_scalar(&mut sums, past, current, &tail, range, outputs, gates)
}

#[expect(
    clippy::large_stack_arrays,
    reason = "Twelve query rows use half-width channel tiles to retain the existing 24 KiB bound."
)]
fn scalar_from_scores<'a>(
    mut past: impl Iterator<Item = &'a [u16]>,
    current: [&[f32]; 2],
    scores: [[&[f32]; 6]; 2],
    range: &std::ops::Range<usize>,
    outputs: [[&mut [f32]; 6]; 2],
    gates: [[&[f32]; 6]; 2],
) -> Result<()> {
    const PACK_CHUNK: usize = 256;
    let mut sums = [[[[0.0_f32; 16]; PAIR_VALUE_TILE]; 6]; 2];
    let first_last = scores[0][0].len() - 1;
    let mut w0_chunk = [[0_u16; 6]; PACK_CHUNK];
    let mut w1_chunk = [[0_u16; 6]; PACK_CHUNK];
    for chunk_start in (0..first_last).step_by(PACK_CHUNK) {
        let chunk_end = (chunk_start + PACK_CHUNK).min(first_last);
        let chunk_len = chunk_end - chunk_start;
        super::super::pack_f16_rows(
            scores[0].map(|row| &row[chunk_start..chunk_end]),
            &mut w0_chunk[..chunk_len],
        )?;
        super::super::pack_f16_rows(
            scores[1].map(|row| &row[chunk_start..chunk_end]),
            &mut w1_chunk[..chunk_len],
        )?;
        if w0_chunk[..chunk_len]
            .iter()
            .chain(&w1_chunk[..chunk_len])
            .flatten()
            .any(|&bits| (bits & 0x7c00) == 0x7c00)
        {
            return Err(invalid("nonfinite packed binary16 attention weight"));
        }
        accumulate_pair_packed_past_scalar(
            &mut sums,
            &mut past,
            &w0_chunk[..chunk_len],
            &w1_chunk[..chunk_len],
            chunk_start,
            range,
        )?;
    }
    let tail = PairTailWeights {
        first_last,
        last0: scores[0].map(|row| crate::kernels::half_bits(row[first_last])),
        prev1: scores[1].map(|row| crate::kernels::half_bits(row[first_last])),
        last1: scores[1].map(|row| crate::kernels::half_bits(row[first_last + 1])),
    };
    if tail
        .last0
        .iter()
        .chain(&tail.prev1)
        .chain(&tail.last1)
        .any(|&bits| (bits & 0x7c00) == 0x7c00)
    {
        return Err(invalid("nonfinite packed binary16 attention weight"));
    }
    finish_pair_packed_scalar(&mut sums, past, current, &tail, range, outputs, gates)
}

#[expect(
    clippy::large_stack_arrays,
    reason = "Twelve query rows use half-width channel tiles to retain the existing 24 KiB bound."
)]
fn scalar<'a>(
    mut past: impl Iterator<Item = &'a [u16]>,
    current: [&[f32]; 2],
    weights: [[&[f32]; 6]; 2],
    range: &std::ops::Range<usize>,
    outputs: [[&mut [f32]; 6]; 2],
    gates: [[&[f32]; 6]; 2],
) -> Result<()> {
    let mut sums = [[[[0.0_f32; 16]; PAIR_VALUE_TILE]; 6]; 2];
    let first_last = weights[0][0].len() - 1;
    for position in 0..first_last {
        let values = next_row(&mut past, range)?;
        for (channel, &value) in values.iter().enumerate() {
            let value = bonsai_ternary::half_to_float(value);
            for (token, sums) in sums.iter_mut().enumerate() {
                for (head, sums) in sums.iter_mut().enumerate() {
                    let lane = &mut sums[channel][position % 16];
                    *lane = value.mul_add(weights[token][head][position], *lane);
                }
            }
        }
    }
    // The iterator includes exactly the first token's cached row, which is
    // visible only to the second token. Never substitute its rounded value
    // for the first token's original current F32 value.
    let previous = next_row(&mut past, range)?;
    if past.next().is_some() {
        return Err(invalid("pinned paired attention value row is long"));
    }
    for (channel, &value) in previous.iter().enumerate() {
        let value = bonsai_ternary::half_to_float(value);
        for (head, sums) in sums[1].iter_mut().enumerate() {
            let lane = &mut sums[channel][first_last % 16];
            *lane = value.mul_add(weights[1][head][first_last], *lane);
        }
    }
    for (token, sums) in sums.iter_mut().enumerate() {
        let last = first_last + token;
        for (head, sums) in sums.iter_mut().enumerate() {
            for (channel, &value) in current[token].iter().enumerate() {
                let lane = &mut sums[channel][last % 16];
                *lane = value.mul_add(weights[token][head][last], *lane);
            }
            for position in last + 1..super::super::padded_dot_steps(last + 1) {
                for channel in &mut sums[..current[token].len()] {
                    let lane = &mut channel[position % 16];
                    *lane = 0.0_f32.mul_add(0.0, *lane);
                }
            }
        }
    }
    // Stage both tokens in dead first-lane storage before publishing either.
    for sums in &mut sums {
        for sums in sums {
            for channel in &mut sums[..current[0].len()] {
                channel[0] = super::reduce(channel)?;
            }
        }
    }
    for (token, outputs) in outputs.into_iter().enumerate() {
        for (head, output) in outputs.into_iter().enumerate() {
            for (channel, output) in output.iter_mut().enumerate() {
                *output = sums[token][head][channel][0]
                    * (1.0 / (1.0 + (-gates[token][head][channel]).exp()));
            }
        }
    }
    Ok(())
}

fn next_row<'a>(
    past: &mut impl Iterator<Item = &'a [u16]>,
    range: &std::ops::Range<usize>,
) -> Result<&'a [u16]> {
    past.next()
        .and_then(|row| row.get(range.clone()))
        .ok_or_else(|| invalid("pinned paired attention value row is short"))
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
#[path = "pinned_attention_pair_values_neon.rs"]
mod neon;

#[cfg(test)]
#[path = "pinned_attention_pair_values_tests.rs"]
mod tests;
