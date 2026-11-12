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
    // Stage in dead first-lane storage; reject the complete reduction before
    // gating or publishing any output, including failures in the sixth head.
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
