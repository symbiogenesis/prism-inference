//! Baseline NEON paired values with unchanged per-token temporal FMA lanes.

use fearless_simd::{Neon, f32x4, prelude::*};

use crate::{Result, invalid};

use super::{PAIR_VALUE_TILE, PairTailWeights, next_row};

#[inline]
fn decode_six_neon(neon: Neon, weights: [u16; 6]) -> [f32; 6] {
    let mut bits = [0_u16; 8];
    bits[..6].copy_from_slice(&weights);
    let (low, high) = crate::kernels::neon::f16_bits_to_f32_x8(neon, &bits);
    let low_arr = low.to_array();
    let high_arr = high.to_array();
    [
        low_arr[0],
        low_arr[1],
        low_arr[2],
        low_arr[3],
        high_arr[0],
        high_arr[1],
    ]
}

type DecodedSixPair = ([f32; 6], [f32; 6]);

#[inline]
fn decode_pair_six_neon(neon: Neon, first: [u16; 6], second: [u16; 6]) -> DecodedSixPair {
    let combined = [
        first[0], first[1], first[2], first[3], first[4], first[5], second[0], second[1],
    ];
    let (lo, hi) = crate::kernels::neon::f16_bits_to_f32_x8(neon, &combined);
    let lo_arr = lo.to_array();
    let hi_arr = hi.to_array();
    let tail = crate::kernels::neon::f16_bits_to_f32_x4(
        neon,
        [second[2], second[3], second[4], second[5]],
    )
    .to_array();
    (
        [
            lo_arr[0], lo_arr[1], lo_arr[2], lo_arr[3], hi_arr[0], hi_arr[1],
        ],
        [hi_arr[2], hi_arr[3], tail[0], tail[1], tail[2], tail[3]],
    )
}

#[inline]
fn decode_two_pairs_six_neon(
    neon: Neon,
    w0_a: [u16; 6],
    w1_a: [u16; 6],
    w0_b: [u16; 6],
    w1_b: [u16; 6],
) -> (DecodedSixPair, DecodedSixPair) {
    let block0 = [
        w0_a[0], w0_a[1], w0_a[2], w0_a[3], w0_a[4], w0_a[5], w1_a[0], w1_a[1],
    ];
    let block1 = [
        w1_a[2], w1_a[3], w1_a[4], w1_a[5], w0_b[0], w0_b[1], w0_b[2], w0_b[3],
    ];
    let block2 = [
        w0_b[4], w0_b[5], w1_b[0], w1_b[1], w1_b[2], w1_b[3], w1_b[4], w1_b[5],
    ];
    let (b0_lo, b0_hi) = crate::kernels::neon::f16_bits_to_f32_x8(neon, &block0);
    let (b1_lo, b1_hi) = crate::kernels::neon::f16_bits_to_f32_x8(neon, &block1);
    let (b2_lo, b2_hi) = crate::kernels::neon::f16_bits_to_f32_x8(neon, &block2);
    let a0 = b0_lo.to_array();
    let a1 = b0_hi.to_array();
    let a2 = b1_lo.to_array();
    let a3 = b1_hi.to_array();
    let a4 = b2_lo.to_array();
    let a5 = b2_hi.to_array();
    (
        (
            [a0[0], a0[1], a0[2], a0[3], a1[0], a1[1]],
            [a1[2], a1[3], a2[0], a2[1], a2[2], a2[3]],
        ),
        (
            [a3[0], a3[1], a3[2], a3[3], a4[0], a4[1]],
            [a4[2], a4[3], a5[0], a5[1], a5[2], a5[3]],
        ),
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "Accumulate packed paired weights in-place into the fixed 24 KiB stack array."
)]
fn accumulate_pair_packed_past_neon<'a>(
    neon: Neon,
    sums: &mut [[[[f32x4<Neon>; PAIR_VALUE_TILE / 4]; 16]; 6]; 2],
    past: &mut impl Iterator<Item = &'a [u16]>,
    w0: &[[u16; 6]],
    w1: &[[u16; 6]],
    base_position: usize,
    range: &std::ops::Range<usize>,
    groups: usize,
) -> Result<()> {
    let (w0_pairs, w0_tail) = w0.as_chunks::<2>();
    let (w1_pairs, w1_tail) = w1.as_chunks::<2>();
    for (pair_idx, ([w0_a, w0_b], [w1_a, w1_b])) in w0_pairs.iter().zip(w1_pairs).enumerate() {
        let row0 = next_row(past, range)?;
        let row1 = next_row(past, range)?;
        let ((f0_a, f1_a), (f0_b, f1_b)) =
            decode_two_pairs_six_neon(neon, *w0_a, *w1_a, *w0_b, *w1_b);
        let first_weights_a = f0_a.map(|w| f32x4::splat(neon, w));
        let second_weights_a = f1_a.map(|w| f32x4::splat(neon, w));
        let first_weights_b = f0_b.map(|w| f32x4::splat(neon, w));
        let second_weights_b = f1_b.map(|w| f32x4::splat(neon, w));
        let lane0 = (base_position + pair_idx * 2) % 16;
        let lane1 = (base_position + pair_idx * 2 + 1) % 16;
        let (chunks8_0, rem0) = row0.as_chunks::<8>();
        let (chunks8_1, rem1) = row1.as_chunks::<8>();
        for (pair_ch, (c0, c1)) in chunks8_0.iter().zip(chunks8_1).enumerate() {
            let (v0_0, v0_1) = crate::kernels::neon::f16_bits_to_f32_x8(neon, c0);
            let (v1_0, v1_1) = crate::kernels::neon::f16_bits_to_f32_x8(neon, c1);
            let g0 = pair_ch * 2;
            let g1 = g0 + 1;
            for h in 0..6 {
                sums[0][h][lane0][g0] =
                    v0_0.mul_add_precise(first_weights_a[h], sums[0][h][lane0][g0]);
                sums[0][h][lane0][g1] =
                    v0_1.mul_add_precise(first_weights_a[h], sums[0][h][lane0][g1]);
                sums[1][h][lane0][g0] =
                    v0_0.mul_add_precise(second_weights_a[h], sums[1][h][lane0][g0]);
                sums[1][h][lane0][g1] =
                    v0_1.mul_add_precise(second_weights_a[h], sums[1][h][lane0][g1]);
                sums[0][h][lane1][g0] =
                    v1_0.mul_add_precise(first_weights_b[h], sums[0][h][lane1][g0]);
                sums[0][h][lane1][g1] =
                    v1_1.mul_add_precise(first_weights_b[h], sums[0][h][lane1][g1]);
                sums[1][h][lane1][g0] =
                    v1_0.mul_add_precise(second_weights_b[h], sums[1][h][lane1][g0]);
                sums[1][h][lane1][g1] =
                    v1_1.mul_add_precise(second_weights_b[h], sums[1][h][lane1][g1]);
            }
        }
        if let ([vals0], [vals1]) = (rem0.as_chunks::<4>().0, rem1.as_chunks::<4>().0) {
            let v0 = crate::kernels::neon::f16_bits_to_f32_x4(neon, *vals0);
            let v1 = crate::kernels::neon::f16_bits_to_f32_x4(neon, *vals1);
            let g = groups - 1;
            for h in 0..6 {
                sums[0][h][lane0][g] = v0.mul_add_precise(first_weights_a[h], sums[0][h][lane0][g]);
                sums[1][h][lane0][g] =
                    v0.mul_add_precise(second_weights_a[h], sums[1][h][lane0][g]);
                sums[0][h][lane1][g] = v1.mul_add_precise(first_weights_b[h], sums[0][h][lane1][g]);
                sums[1][h][lane1][g] =
                    v1.mul_add_precise(second_weights_b[h], sums[1][h][lane1][g]);
            }
        }
    }
    if let (Some(&w0_last), Some(&w1_last)) = (w0_tail.first(), w1_tail.first()) {
        let row = next_row(past, range)?;
        let (f0, f1) = decode_pair_six_neon(neon, w0_last, w1_last);
        let w0_splat = f0.map(|w| f32x4::splat(neon, w));
        let w1_splat = f1.map(|w| f32x4::splat(neon, w));
        let lane = (base_position + w0_pairs.len() * 2) % 16;
        let (chunks8, rem) = row.as_chunks::<8>();
        for (pair_ch, chunk8) in chunks8.iter().enumerate() {
            let (v0, v1) = crate::kernels::neon::f16_bits_to_f32_x8(neon, chunk8);
            let g0 = pair_ch * 2;
            let g1 = g0 + 1;
            for h in 0..6 {
                sums[0][h][lane][g0] = v0.mul_add_precise(w0_splat[h], sums[0][h][lane][g0]);
                sums[0][h][lane][g1] = v1.mul_add_precise(w0_splat[h], sums[0][h][lane][g1]);
                sums[1][h][lane][g0] = v0.mul_add_precise(w1_splat[h], sums[1][h][lane][g0]);
                sums[1][h][lane][g1] = v1.mul_add_precise(w1_splat[h], sums[1][h][lane][g1]);
            }
        }
        if let [vals4] = rem.as_chunks::<4>().0 {
            let v = crate::kernels::neon::f16_bits_to_f32_x4(neon, *vals4);
            let g = groups - 1;
            for h in 0..6 {
                sums[0][h][lane][g] = v.mul_add_precise(w0_splat[h], sums[0][h][lane][g]);
                sums[1][h][lane][g] = v.mul_add_precise(w1_splat[h], sums[1][h][lane][g]);
            }
        }
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "Finalize paired packed accumulation without copying the 24 KiB stack array."
)]
fn finish_pair_packed_neon<'a>(
    neon: Neon,
    sums: &mut [[[[f32x4<Neon>; PAIR_VALUE_TILE / 4]; 16]; 6]; 2],
    mut past: impl Iterator<Item = &'a [u16]>,
    current: [&[f32]; 2],
    tail: &PairTailWeights,
    range: &std::ops::Range<usize>,
    outputs: [[&mut [f32]; 6]; 2],
    gates: [[&[f32]; 6]; 2],
    groups: usize,
) -> Result<()> {
    let previous = next_row(&mut past, range)?;
    if past.next().is_some() {
        return Err(invalid("pinned paired attention value row is long"));
    }
    let zero = f32x4::splat(neon, 0.0);
    let (last0_f32, prev1_f32) = decode_pair_six_neon(neon, tail.last0, tail.prev1);
    let last1_f32 = decode_six_neon(neon, tail.last1);
    let prev1_splat = prev1_f32.map(|w| f32x4::splat(neon, w));
    let prev_lane = tail.first_last % 16;
    for (group, value) in previous.as_chunks::<4>().0.iter().enumerate() {
        let value = crate::kernels::neon::f16_bits_to_f32_x4(neon, *value);
        for (head, head_sums) in sums[1].iter_mut().enumerate() {
            let sum = &mut head_sums[prev_lane][group];
            *sum = value.mul_add_precise(prev1_splat[head], *sum);
        }
    }
    let last_weights = [last0_f32, last1_f32];
    for (token, token_sums) in sums.iter_mut().enumerate() {
        let last = tail.first_last + token;
        let last_lane = last % 16;
        for (head, head_sums) in token_sums.iter_mut().enumerate() {
            let weight = f32x4::splat(neon, last_weights[token][head]);
            for (group, value) in current[token].as_chunks::<4>().0.iter().enumerate() {
                let sum = &mut head_sums[last_lane][group];
                *sum = f32x4::load_array(neon, *value).mul_add_precise(weight, *sum);
            }
            for position in last + 1..super::super::super::padded_dot_steps(last + 1) {
                for sum in &mut head_sums[position % 16][..groups] {
                    *sum = zero.mul_add_precise(zero, *sum);
                }
            }
        }
    }
    publish(neon, sums, outputs, gates, groups)
}

#[expect(
    clippy::large_stack_arrays,
    reason = "Twelve query rows use 32 channels to retain the existing 24 KiB temporal accumulator."
)]
pub(super) fn values_packed<'a>(
    neon: Neon,
    mut past: impl Iterator<Item = &'a [u16]>,
    current: [&[f32]; 2],
    weights: [&[[u16; 6]]; 2],
    range: &std::ops::Range<usize>,
    outputs: [[&mut [f32]; 6]; 2],
    gates: [[&[f32]; 6]; 2],
) -> Result<()> {
    let zero = f32x4::splat(neon, 0.0);
    let mut sums = [[[[zero; PAIR_VALUE_TILE / 4]; 16]; 6]; 2];
    let groups = current[0].len() / 4;
    let first_last = weights[0].len() - 1;
    accumulate_pair_packed_past_neon(
        neon,
        &mut sums,
        &mut past,
        &weights[0][..first_last],
        &weights[1][..first_last],
        0,
        range,
        groups,
    )?;
    let tail = PairTailWeights {
        first_last,
        last0: weights[0][first_last],
        prev1: weights[1][first_last],
        last1: weights[1][first_last + 1],
    };
    finish_pair_packed_neon(
        neon, &mut sums, past, current, &tail, range, outputs, gates, groups,
    )
}

#[expect(
    clippy::large_stack_arrays,
    reason = "Twelve query rows use 32 channels to retain the existing 24 KiB temporal accumulator."
)]
pub(super) fn values_from_scores<'a>(
    neon: Neon,
    mut past: impl Iterator<Item = &'a [u16]>,
    current: [&[f32]; 2],
    scores: [[&[f32]; 6]; 2],
    range: &std::ops::Range<usize>,
    outputs: [[&mut [f32]; 6]; 2],
    gates: [[&[f32]; 6]; 2],
) -> Result<()> {
    const PACK_CHUNK: usize = 256;
    let zero = f32x4::splat(neon, 0.0);
    let mut sums = [[[[zero; PAIR_VALUE_TILE / 4]; 16]; 6]; 2];
    let groups = current[0].len() / 4;
    let first_last = scores[0][0].len() - 1;
    let mut w0_chunk = [[0_u16; 6]; PACK_CHUNK];
    let mut w1_chunk = [[0_u16; 6]; PACK_CHUNK];
    for chunk_start in (0..first_last).step_by(PACK_CHUNK) {
        let chunk_end = (chunk_start + PACK_CHUNK).min(first_last);
        let chunk_len = chunk_end - chunk_start;
        super::super::super::pack_f16_rows(
            scores[0].map(|row| &row[chunk_start..chunk_end]),
            &mut w0_chunk[..chunk_len],
        )?;
        super::super::super::pack_f16_rows(
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
        accumulate_pair_packed_past_neon(
            neon,
            &mut sums,
            &mut past,
            &w0_chunk[..chunk_len],
            &w1_chunk[..chunk_len],
            chunk_start,
            range,
            groups,
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
    finish_pair_packed_neon(
        neon, &mut sums, past, current, &tail, range, outputs, gates, groups,
    )
}

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
