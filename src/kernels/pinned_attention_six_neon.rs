//! Baseline NEON six-head sharing with unchanged native FMA and reduction order.

use fearless_simd::{Neon, f32x4, prelude::*};

use crate::{Result, invalid};

use super::super::padded_dot_steps;

pub(super) fn dot_neon<const HEADS: usize>(
    neon: Neon,
    key: &[u16],
    queries: [&[f32]; HEADS],
) -> Result<[f32; HEADS]> {
    let zero = f32x4::splat(neon, 0.0);
    let mut sums = [[zero; 4]; HEADS];
    // A complete native sixteen-lane step addresses the four accumulators at
    // fixed indices. The former eight-column step selected two accumulator
    // registers with `index * 2 % 4` on every iteration. Keep each lane's exact
    // FMA sequence while halving full-width loop trips and query-row slicing.
    let (chunks, remainder) = key.as_chunks::<16>();
    for (index, chunk) in chunks.iter().enumerate() {
        let (halves, _) = chunk.as_chunks::<8>();
        let (first, second) = crate::kernels::neon::f16_bits_to_f32_x8(neon, &halves[0]);
        let (third, fourth) = crate::kernels::neon::f16_bits_to_f32_x8(neon, &halves[1]);
        for (head, sum) in sums.iter_mut().enumerate() {
            let query = &queries[head][index * 16..index * 16 + 16];
            sum[0] = first.mul_add_precise(f32x4::from_slice(neon, &query[..4]), sum[0]);
            sum[1] = second.mul_add_precise(f32x4::from_slice(neon, &query[4..8]), sum[1]);
            sum[2] = third.mul_add_precise(f32x4::from_slice(neon, &query[8..12]), sum[2]);
            sum[3] = fourth.mul_add_precise(f32x4::from_slice(neon, &query[12..]), sum[3]);
        }
    }
    let vector_steps = chunks.len() * 16;
    let (tail_chunks, _) = remainder.as_chunks::<8>();
    if let Some(chunk) = tail_chunks.first() {
        let (first, second) = crate::kernels::neon::f16_bits_to_f32_x8(neon, chunk);
        for (head, sum) in sums.iter_mut().enumerate() {
            let query = &queries[head][vector_steps..vector_steps + 8];
            sum[0] = first.mul_add_precise(f32x4::from_slice(neon, &query[..4]), sum[0]);
            sum[1] = second.mul_add_precise(f32x4::from_slice(neon, &query[4..]), sum[1]);
        }
    }
    let vector_steps = vector_steps + tail_chunks.len() * 8;
    let mut output = [0.0; HEADS];
    // Swift's complete 16-element key blocks need no scalar tail. Keep the
    // four native registers through padding and the original reduction tree,
    // rather than extracting sixteen lanes per head before adding them.
    if key.len().is_multiple_of(16) {
        let padded = padded_dot_steps(key.len()) != key.len();
        for (mut sum, output) in sums.into_iter().zip(&mut output) {
            if padded {
                // An aligned row's retained masked cycle visits every lane
                // once. Preserve the +0 FMA that normalizes a possible -0.
                for register in &mut sum {
                    *register = zero.mul_add_precise(zero, *register);
                }
            }
            let lanes = ((sum[0] + sum[2]) + (sum[1] + sum[3])).to_array();
            *output = crate::kernels::finite((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))?;
        }
        return Ok(output);
    }
    for (head, (sum, output)) in sums.into_iter().zip(&mut output).enumerate() {
        let mut scalar = sum.map(f32x4::to_array);
        for index in vector_steps..padded_dot_steps(key.len()) {
            let (value, weight) = if index < key.len() {
                (
                    bonsai_ternary::half_to_float(key[index]),
                    queries[head][index],
                )
            } else {
                (0.0, 0.0)
            };
            let lane = &mut scalar[index % 16 / 4][index % 4];
            *lane = value.mul_add(weight, *lane);
        }
        let lanes: [f32; 4] = std::array::from_fn(|lane| {
            (scalar[0][lane] + scalar[2][lane]) + (scalar[1][lane] + scalar[3][lane])
        });
        *output = crate::kernels::finite((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))?;
    }
    Ok(output)
}

#[expect(
    clippy::large_stack_arrays,
    reason = "Validated six-head inputs share one bounded 24 KiB temporal accumulator."
)]
pub(super) fn value_tile_neon<'a>(
    neon: Neon,
    mut past: impl Iterator<Item = &'a [u16]>,
    current: &[f32],
    weights: [&[f32]; 6],
    range: &std::ops::Range<usize>,
    outputs: [&mut [f32]; 6],
    gates: [&[f32]; 6],
) -> Result<()> {
    let zero = f32x4::splat(neon, 0.0);
    let mut sums = [[[zero; super::VALUE_TILE / 4]; 16]; 6];
    let groups = current.len() / 4;
    let last = weights[0].len() - 1;
    for token in 0..last {
        let values = past
            .next()
            .and_then(|row| row.get(range.clone()))
            .ok_or_else(|| invalid("pinned attention value row is short"))?;
        let weights: [f32x4<Neon>; 6] =
            std::array::from_fn(|head| f32x4::splat(neon, weights[head][token]));
        let (pairs, remainder) = values.as_chunks::<8>();
        for (pair, values) in pairs.iter().enumerate() {
            let (first, second) = crate::kernels::neon::f16_bits_to_f32_x8(neon, values);
            for (head, sums) in sums.iter_mut().enumerate() {
                let sums = &mut sums[token % 16];
                sums[pair * 2] = first.mul_add_precise(weights[head], sums[pair * 2]);
                sums[pair * 2 + 1] = second.mul_add_precise(weights[head], sums[pair * 2 + 1]);
            }
        }
        if let Some(values) = remainder.as_chunks::<4>().0.first() {
            let value = crate::kernels::neon::f16_bits_to_f32_x4(neon, *values);
            for (head, sums) in sums.iter_mut().enumerate() {
                let sum = &mut sums[token % 16][pairs.len() * 2];
                *sum = value.mul_add_precise(weights[head], *sum);
            }
        }
    }
    if past.next().is_some() {
        return Err(invalid("pinned attention value row is long"));
    }
    for (head, sums) in sums.iter_mut().enumerate() {
        let weight = f32x4::splat(neon, weights[head][last]);
        for (group, value) in current.as_chunks::<4>().0.iter().enumerate() {
            let sum = &mut sums[last % 16][group];
            *sum = f32x4::load_array(neon, *value).mul_add_precise(weight, *sum);
        }
        for token in last + 1..padded_dot_steps(last + 1) {
            for sum in &mut sums[token % 16][..groups] {
                *sum = zero.mul_add_precise(zero, *sum);
            }
        }
    }
    publish(neon, &mut sums, outputs, gates, groups)
}

fn publish(
    neon: Neon,
    sums: &mut [[[f32x4<Neon>; super::VALUE_TILE / 4]; 16]; 6],
    outputs: [&mut [f32]; 6],
    gates: [&[f32]; 6],
    groups: usize,
) -> Result<()> {
    publish_group(neon, sums, outputs, gates, groups)
}

#[inline]
fn decode_single_group_weights<const HEADS: usize>(neon: Neon, w: [u16; HEADS]) -> [f32; HEADS] {
    if HEADS <= 4 {
        let mut bits = [0_u16; 4];
        bits[..HEADS].copy_from_slice(&w);
        let decoded = crate::kernels::neon::f16_bits_to_f32_x4(neon, bits).to_array();
        return std::array::from_fn(|head| decoded[head]);
    }
    let mut bits = [0_u16; 8];
    bits[..HEADS].copy_from_slice(&w);
    let (low, high) = crate::kernels::neon::f16_bits_to_f32_x8(neon, &bits);
    let low_arr = low.to_array();
    let high_arr = high.to_array();
    std::array::from_fn(|head| {
        if head < 4 {
            low_arr[head]
        } else {
            high_arr[head - 4]
        }
    })
}

#[inline]
fn decode_pair_group_weights<const HEADS: usize>(
    neon: Neon,
    w0: [u16; HEADS],
    w1: [u16; HEADS],
) -> ([f32; HEADS], [f32; HEADS]) {
    if HEADS <= 4 {
        let mut bits = [0_u16; 8];
        bits[..HEADS].copy_from_slice(&w0);
        bits[4..4 + HEADS].copy_from_slice(&w1);
        let (first, second) = crate::kernels::neon::f16_bits_to_f32_x8(neon, &bits);
        let first_arr = first.to_array();
        let second_arr = second.to_array();
        return (
            std::array::from_fn(|head| first_arr[head]),
            std::array::from_fn(|head| second_arr[head]),
        );
    }
    (
        decode_single_group_weights(neon, w0),
        decode_single_group_weights(neon, w1),
    )
}

pub(super) fn value_tile_group_packed_neon<'a, const HEADS: usize>(
    neon: Neon,
    mut past: impl Iterator<Item = &'a [u16]>,
    current: &[f32],
    weights: &[[u16; HEADS]],
    range: &std::ops::Range<usize>,
    outputs: [&mut [f32]; HEADS],
    gates: [&[f32]; HEADS],
) -> Result<()> {
    const GROUPS: usize = super::VALUE_TILE / 4;
    let groups = outputs[0].len() / 4;
    let past_len = weights.len() - 1;
    let mut sums = [[[f32x4::splat(neon, 0.0); GROUPS]; 16]; HEADS];
    let mut token = 0;
    let (weight_pairs, weight_tail) = weights[..past_len].as_chunks::<2>();
    for [w0_bits, w1_bits] in weight_pairs {
        let Some(row0) = past.next().and_then(|row| row.get(range.start..range.end)) else {
            return Err(invalid("pinned attention value row is short"));
        };
        let Some(row1) = past.next().and_then(|row| row.get(range.start..range.end)) else {
            return Err(invalid("pinned attention value row is short"));
        };
        let (w0, w1) = decode_pair_group_weights(neon, *w0_bits, *w1_bits);
        let w0_splat = w0.map(|weight| f32x4::splat(neon, weight));
        let w1_splat = w1.map(|weight| f32x4::splat(neon, weight));
        let lane0 = token % 16;
        let lane1 = (token + 1) % 16;
        token += 2;
        let (chunks8_0, rem0) = row0.as_chunks::<8>();
        let (chunks8_1, rem1) = row1.as_chunks::<8>();
        for (pair_idx, (c0, c1)) in chunks8_0.iter().zip(chunks8_1).enumerate() {
            let (v0_0, v0_1) = crate::kernels::neon::f16_bits_to_f32_x8(neon, c0);
            let (v1_0, v1_1) = crate::kernels::neon::f16_bits_to_f32_x8(neon, c1);
            let g0 = pair_idx * 2;
            let g1 = g0 + 1;
            for h in 0..HEADS {
                sums[h][lane0][g0] = v0_0.mul_add(w0_splat[h], sums[h][lane0][g0]);
                sums[h][lane0][g1] = v0_1.mul_add(w0_splat[h], sums[h][lane0][g1]);
                sums[h][lane1][g0] = v1_0.mul_add(w1_splat[h], sums[h][lane1][g0]);
                sums[h][lane1][g1] = v1_1.mul_add(w1_splat[h], sums[h][lane1][g1]);
            }
        }
        if let ([vals0], [vals1]) = (rem0.as_chunks::<4>().0, rem1.as_chunks::<4>().0) {
            let v0 = crate::kernels::neon::f16_bits_to_f32_x4(neon, *vals0);
            let v1 = crate::kernels::neon::f16_bits_to_f32_x4(neon, *vals1);
            let g = groups - 1;
            for h in 0..HEADS {
                sums[h][lane0][g] = v0.mul_add(w0_splat[h], sums[h][lane0][g]);
                sums[h][lane1][g] = v1.mul_add(w1_splat[h], sums[h][lane1][g]);
            }
        }
    }
    if let Some(&tail_bits) = weight_tail.first() {
        let Some(row) = past.next().and_then(|row| row.get(range.start..range.end)) else {
            return Err(invalid("pinned attention value row is short"));
        };
        let w_tail = decode_single_group_weights(neon, tail_bits).map(|w| f32x4::splat(neon, w));
        let lane = token % 16;
        let (chunks8, rem) = row.as_chunks::<8>();
        for (pair_idx, chunk8) in chunks8.iter().enumerate() {
            let (v0, v1) = crate::kernels::neon::f16_bits_to_f32_x8(neon, chunk8);
            let g0 = pair_idx * 2;
            let g1 = g0 + 1;
            for h in 0..HEADS {
                sums[h][lane][g0] = v0.mul_add(w_tail[h], sums[h][lane][g0]);
                sums[h][lane][g1] = v1.mul_add(w_tail[h], sums[h][lane][g1]);
            }
        }
        if let [vals4] = rem.as_chunks::<4>().0 {
            let v = crate::kernels::neon::f16_bits_to_f32_x4(neon, *vals4);
            let g = groups - 1;
            for h in 0..HEADS {
                sums[h][lane][g] = v.mul_add(w_tail[h], sums[h][lane][g]);
            }
        }
    }
    if past.next().is_some() {
        return Err(invalid("pinned attention value row is long"));
    }
    let last_lane = past_len % 16;
    let w_last =
        decode_single_group_weights(neon, weights[past_len]).map(|w| f32x4::splat(neon, w));
    for (g, current_chunk) in current.as_chunks::<4>().0.iter().enumerate() {
        let v = f32x4::load_array(neon, *current_chunk);
        for h in 0..HEADS {
            sums[h][last_lane][g] = v.mul_add(w_last[h], sums[h][last_lane][g]);
        }
    }
    publish_group(neon, &mut sums, outputs, gates, groups)
}

fn publish_group<const HEADS: usize>(
    neon: Neon,
    sums: &mut [[[f32x4<Neon>; super::VALUE_TILE / 4]; 16]; HEADS],
    outputs: [&mut [f32]; HEADS],
    gates: [&[f32]; HEADS],
    groups: usize,
) -> Result<()> {
    // Stage in dead first-lane storage; reject the complete reduction before
    // gating or publishing any output, including failures in the last head.
    let mut finite = f32x4::splat(neon, 0.0).is_finite();
    for sums in &mut *sums {
        for group in 0..groups {
            let lanes: [f32x4<Neon>; 4] = std::array::from_fn(|lane| {
                (sums[lane][group] + sums[8 + lane][group])
                    + (sums[4 + lane][group] + sums[12 + lane][group])
            });
            let result = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
            finite &= result.is_finite();
            sums[0][group] = result;
        }
    }
    if !finite.all_true() {
        return Err(invalid("nonfinite pinned attention value reduction"));
    }
    let one = f32x4::splat(neon, 1.0);
    for (head, output) in outputs.into_iter().enumerate() {
        for (group, output) in output.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let exponentials = f32x4::load_array(
                neon,
                std::array::from_fn(|lane| (-gates[head][group * 4 + lane]).exp()),
            );
            (sums[head][0][group] * (one / (one + exponentials))).store_slice(output);
        }
    }
    Ok(())
}
