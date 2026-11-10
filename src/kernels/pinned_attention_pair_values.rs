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
