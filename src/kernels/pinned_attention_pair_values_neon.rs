//! Baseline NEON paired values with unchanged per-token temporal FMA lanes.

use fearless_simd::{Neon, f32x4, prelude::*};

use crate::{Result, invalid};

use super::{PAIR_VALUE_TILE, next_row};

#[expect(
    clippy::large_stack_arrays,
    reason = "Twelve query rows use 32 channels to retain the existing 24 KiB temporal accumulator."
)]
pub(super) fn values<'a>(
    neon: Neon,
    mut past: impl Iterator<Item = &'a [u16]>,
    current: [&[f32]; 2],
    weights: [[&[f32]; 6]; 2],
    range: &std::ops::Range<usize>,
    outputs: [[&mut [f32]; 6]; 2],
    gates: [[&[f32]; 6]; 2],
) -> Result<()> {
    let zero = f32x4::splat(neon, 0.0);
    let mut sums = [[[[zero; PAIR_VALUE_TILE / 4]; 16]; 6]; 2];
    let groups = current[0].len() / 4;
    let first_last = weights[0][0].len() - 1;
    for position in 0..first_last {
        let values = next_row(&mut past, range)?;
        let weights: [[f32x4<Neon>; 6]; 2] = std::array::from_fn(|token| {
            std::array::from_fn(|head| f32x4::splat(neon, weights[token][head][position]))
        });
        let (pairs, remainder) = values.as_chunks::<8>();
        for (pair, values) in pairs.iter().enumerate() {
            let (first, second) = crate::kernels::neon::f16_bits_to_f32_x8(neon, values);
            for (token, sums) in sums.iter_mut().enumerate() {
                for (head, sums) in sums.iter_mut().enumerate() {
                    let sums = &mut sums[position % 16];
                    sums[pair * 2] = first.mul_add_precise(weights[token][head], sums[pair * 2]);
                    sums[pair * 2 + 1] =
                        second.mul_add_precise(weights[token][head], sums[pair * 2 + 1]);
                }
            }
        }
        if let Some(values) = remainder.as_chunks::<4>().0.first() {
            let value = crate::kernels::neon::f16_bits_to_f32_x4(neon, *values);
            for (token, sums) in sums.iter_mut().enumerate() {
                for (head, sums) in sums.iter_mut().enumerate() {
                    let sum = &mut sums[position % 16][pairs.len() * 2];
                    *sum = value.mul_add_precise(weights[token][head], *sum);
                }
            }
        }
    }
    let previous = next_row(&mut past, range)?;
    if past.next().is_some() {
        return Err(invalid("pinned paired attention value row is long"));
    }
    for (group, value) in previous.as_chunks::<4>().0.iter().enumerate() {
        let value = crate::kernels::neon::f16_bits_to_f32_x4(neon, *value);
        for (head, sums) in sums[1].iter_mut().enumerate() {
            let weight = f32x4::splat(neon, weights[1][head][first_last]);
            let sum = &mut sums[first_last % 16][group];
            *sum = value.mul_add_precise(weight, *sum);
        }
    }
    for (token, sums) in sums.iter_mut().enumerate() {
        let last = first_last + token;
        for (head, sums) in sums.iter_mut().enumerate() {
            let weight = f32x4::splat(neon, weights[token][head][last]);
            for (group, value) in current[token].as_chunks::<4>().0.iter().enumerate() {
                let sum = &mut sums[last % 16][group];
                *sum = f32x4::load_array(neon, *value).mul_add_precise(weight, *sum);
            }
            for position in last + 1..super::super::super::padded_dot_steps(last + 1) {
                for sum in &mut sums[position % 16][..groups] {
                    *sum = zero.mul_add_precise(zero, *sum);
                }
            }
        }
    }
    publish(neon, &mut sums, outputs, gates, groups)
}

fn publish(
    neon: Neon,
    sums: &mut [[[[f32x4<Neon>; PAIR_VALUE_TILE / 4]; 16]; 6]; 2],
    outputs: [[&mut [f32]; 6]; 2],
    gates: [[&[f32]; 6]; 2],
    groups: usize,
) -> Result<()> {
    let mut finite = f32x4::splat(neon, 0.0).is_finite();
    for token in &mut *sums {
        for sums in token {
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
    }
    if !finite.all_true() {
        return Err(invalid("nonfinite pinned paired attention value reduction"));
    }
    let one = f32x4::splat(neon, 1.0);
    for (token, outputs) in outputs.into_iter().enumerate() {
        for (head, output) in outputs.into_iter().enumerate() {
            for (group, output) in output.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                let exponentials = f32x4::load_array(
                    neon,
                    std::array::from_fn(|lane| (-gates[token][head][group * 4 + lane]).exp()),
                );
                (sums[token][head][0][group] * (one / (one + exponentials))).store_slice(output);
            }
        }
    }
    Ok(())
}
