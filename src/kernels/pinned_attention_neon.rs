//! Exact baseline NEON value tiles with adjacent channels in vector lanes.

use fearless_simd::{Neon, f32x4, prelude::*};

use crate::{Result, invalid};

use super::{VALUE_TILE, VALUE_TILE_FOUR, VALUE_TILE_FOUR_TWO_TOKENS, padded_dot_steps};

/// Consumes a validated tile whose width is a positive multiple of four.
///
/// The sixteen native temporal FMA lanes stay independent. Each vector groups
/// four adjacent value channels, making an update contiguous without changing
/// any channel's token order. The accumulator remains exactly 2 KiB; decoded
/// FP16 values exist only for one four-channel load. No FP16 arithmetic or
/// optional instruction set is required.
pub(super) fn value_tile<'a>(
    neon: Neon,
    mut past: impl Iterator<Item = &'a [u16]>,
    current: &[f32],
    weights: &[f32],
    range: &std::ops::Range<usize>,
    output: &mut [f32],
) -> Result<()> {
    let zero = f32x4::splat(neon, 0.0);
    let mut sums = [[zero; VALUE_TILE / 4]; 16];
    let groups = output.len() / 4;
    let last = weights.len() - 1;
    for (index, &weight) in weights[..last].iter().enumerate() {
        let values = past
            .next()
            .and_then(|row| row.get(range.start..range.end))
            .ok_or_else(|| invalid("pinned attention value row is short"))?;
        let weight = f32x4::splat(neon, weight);
        let (sum_pairs, sum_tail) = sums[index % 16][..groups].as_chunks_mut::<2>();
        let (val_chunks8, val_rem) = values.as_chunks::<8>();
        for (pair, chunk8) in sum_pairs.iter_mut().zip(val_chunks8) {
            let (v0, v1) = super::super::neon::f16_bits_to_f32_x8(neon, chunk8);
            pair[0] = v0.mul_add_precise(weight, pair[0]);
            pair[1] = v1.mul_add_precise(weight, pair[1]);
        }
        for (sum, values) in sum_tail.iter_mut().zip(val_rem.as_chunks::<4>().0) {
            let values = super::super::neon::f16_bits_to_f32_x4(neon, *values);
            *sum = values.mul_add_precise(weight, *sum);
        }
    }
    if past.next().is_some() {
        return Err(invalid("pinned attention value row is long"));
    }
    let weight = f32x4::splat(neon, weights[last]);
    for (sum, values) in sums[last % 16][..groups]
        .iter_mut()
        .zip(current.as_chunks::<4>().0)
    {
        *sum = f32x4::load_array(neon, *values).mul_add_precise(weight, *sum);
    }
    // Preserve the first masked FMA in every temporal lane, including -0.
    for index in weights.len()..padded_dot_steps(weights.len()) {
        for sum in &mut sums[index % 16][..groups] {
            *sum = zero.mul_add_precise(zero, *sum);
        }
    }
    for (group, out) in output.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        let lanes: [f32x4<Neon>; 4] = std::array::from_fn(|lane| {
            (sums[lane][group] + sums[8 + lane][group])
                + (sums[4 + lane][group] + sums[12 + lane][group])
        });
        let result = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])).to_array();
        // Publication and finite rejection retain increasing channel order.
        for (out, value) in out.iter_mut().zip(result) {
            *out = super::super::finite(value)?;
        }
    }
    Ok(())
}

/// Consumes a validated tile across four query heads sharing the same KV head.
///
/// Each cached binary16 value row is read and converted to binary32 once for all
/// four query heads, while retaining each head's independent sixteen-lane FMA
/// accumulator and reduction tree.
#[allow(
    clippy::large_stack_arrays,
    clippy::needless_range_loop,
    reason = "Keep explicit 4-head and token indices aligned across the 32 KiB stack accumulator."
)]
pub(super) fn value_tile_four<'a>(
    neon: Neon,
    mut past: impl Iterator<Item = &'a [u16]>,
    current: &[f32],
    weights: [&[f32]; 4],
    range: &std::ops::Range<usize>,
    outputs: [&mut [f32]; 4],
) -> Result<()> {
    let zero = f32x4::splat(neon, 0.0);
    let mut sums = [[[zero; VALUE_TILE_FOUR / 4]; 16]; 4];
    let groups = outputs[0].len() / 4;
    let token_count = weights[0].len();
    let last = token_count - 1;
    for index in 0..last {
        let values = past
            .next()
            .and_then(|row| row.get(range.start..range.end))
            .ok_or_else(|| invalid("pinned attention value row is short"))?;
        let lane = index % 16;
        let ws: [f32x4<Neon>; 4] = std::array::from_fn(|h| f32x4::splat(neon, weights[h][index]));
        let (val_chunks8, val_rem) = values.as_chunks::<8>();
        for (pair_idx, chunk8) in val_chunks8.iter().enumerate() {
            let (v0, v1) = super::super::neon::f16_bits_to_f32_x8(neon, chunk8);
            let g0 = pair_idx * 2;
            let g1 = g0 + 1;
            for h in 0..4 {
                sums[h][lane][g0] = v0.mul_add_precise(ws[h], sums[h][lane][g0]);
                sums[h][lane][g1] = v1.mul_add_precise(ws[h], sums[h][lane][g1]);
            }
        }
        let rem_base = val_chunks8.len() * 2;
        for (rem_idx, vals4) in val_rem.as_chunks::<4>().0.iter().enumerate() {
            let v = super::super::neon::f16_bits_to_f32_x4(neon, *vals4);
            let g = rem_base + rem_idx;
            for h in 0..4 {
                sums[h][lane][g] = v.mul_add_precise(ws[h], sums[h][lane][g]);
            }
        }
    }
    if past.next().is_some() {
        return Err(invalid("pinned attention value row is long"));
    }
    let last_lane = last % 16;
    let ws: [f32x4<Neon>; 4] = std::array::from_fn(|h| f32x4::splat(neon, weights[h][last]));
    for (g, values) in current.as_chunks::<4>().0.iter().enumerate() {
        let v = f32x4::load_array(neon, *values);
        for h in 0..4 {
            sums[h][last_lane][g] = v.mul_add_precise(ws[h], sums[h][last_lane][g]);
        }
    }
    for index in token_count..padded_dot_steps(token_count) {
        let lane = index % 16;
        for head_sums in &mut sums {
            for sum in &mut head_sums[lane][..groups] {
                *sum = zero.mul_add_precise(zero, *sum);
            }
        }
    }
    for (h, output) in outputs.into_iter().enumerate() {
        for (group, out) in output.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let lanes: [f32x4<Neon>; 4] = std::array::from_fn(|lane| {
                (sums[h][lane][group] + sums[h][8 + lane][group])
                    + (sums[h][4 + lane][group] + sums[h][12 + lane][group])
            });
            let result = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])).to_array();
            for (out, value) in out.iter_mut().zip(result) {
                *out = super::super::finite(value)?;
            }
        }
    }
    Ok(())
}

/// Consumes a validated tile across two consecutive tokens and four query heads sharing a KV head.
///
/// Both tokens share the first `weights0[0].len() - 1` cached binary16 value rows,
/// which are read and converted to binary32 once for all eight `(token, head)`
/// accumulators while preserving each head's independent sixteen-lane FMA tree.
#[allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "Keep explicit 2-token x 4-head indices aligned across fixed arrays."
)]
pub(super) fn value_tile_four_two_tokens<'a>(
    neon: Neon,
    mut past: impl Iterator<Item = &'a [u16]>,
    currents: [&[f32]; 2],
    weights0: [&[f32]; 4],
    weights1: [&[f32]; 4],
    range: &std::ops::Range<usize>,
    outputs0: [&mut [f32]; 4],
    outputs1: [&mut [f32]; 4],
) -> Result<()> {
    let zero = f32x4::splat(neon, 0.0);
    let mut sums0 = [[[zero; VALUE_TILE_FOUR_TWO_TOKENS / 4]; 16]; 4];
    let mut sums1 = [[[zero; VALUE_TILE_FOUR_TWO_TOKENS / 4]; 16]; 4];
    let groups = outputs0[0].len() / 4;
    let token_count0 = weights0[0].len();
    let token_count1 = weights1[0].len();
    let last0 = token_count0 - 1;
    let last1 = token_count1 - 1;
    for index in 0..last0 {
        let values = past
            .next()
            .and_then(|row| row.get(range.start..range.end))
            .ok_or_else(|| invalid("pinned attention value row is short"))?;
        let lane = index % 16;
        let ws0: [f32x4<Neon>; 4] = std::array::from_fn(|h| f32x4::splat(neon, weights0[h][index]));
        let ws1: [f32x4<Neon>; 4] = std::array::from_fn(|h| f32x4::splat(neon, weights1[h][index]));
        let (val_chunks8, val_rem) = values.as_chunks::<8>();
        for (pair_idx, chunk8) in val_chunks8.iter().enumerate() {
            let (v0, v1) = super::super::neon::f16_bits_to_f32_x8(neon, chunk8);
            let g0 = pair_idx * 2;
            let g1 = g0 + 1;
            for h in 0..4 {
                sums0[h][lane][g0] = v0.mul_add_precise(ws0[h], sums0[h][lane][g0]);
                sums0[h][lane][g1] = v1.mul_add_precise(ws0[h], sums0[h][lane][g1]);
                sums1[h][lane][g0] = v0.mul_add_precise(ws1[h], sums1[h][lane][g0]);
                sums1[h][lane][g1] = v1.mul_add_precise(ws1[h], sums1[h][lane][g1]);
            }
        }
        let rem_base = val_chunks8.len() * 2;
        for (rem_idx, vals4) in val_rem.as_chunks::<4>().0.iter().enumerate() {
            let v = super::super::neon::f16_bits_to_f32_x4(neon, *vals4);
            let g = rem_base + rem_idx;
            for h in 0..4 {
                sums0[h][lane][g] = v.mul_add_precise(ws0[h], sums0[h][lane][g]);
                sums1[h][lane][g] = v.mul_add_precise(ws1[h], sums1[h][lane][g]);
            }
        }
    }
    let values_last0 = past
        .next()
        .and_then(|row| row.get(range.start..range.end))
        .ok_or_else(|| invalid("pinned attention value row is short"))?;
    if past.next().is_some() {
        return Err(invalid("pinned attention value row is long"));
    }
    let last0_lane = last0 % 16;
    let ws0_last: [f32x4<Neon>; 4] =
        std::array::from_fn(|h| f32x4::splat(neon, weights0[h][last0]));
    for (g, values) in currents[0].as_chunks::<4>().0.iter().enumerate() {
        let v = f32x4::load_array(neon, *values);
        for h in 0..4 {
            sums0[h][last0_lane][g] = v.mul_add_precise(ws0_last[h], sums0[h][last0_lane][g]);
        }
    }
    let ws1_last0: [f32x4<Neon>; 4] =
        std::array::from_fn(|h| f32x4::splat(neon, weights1[h][last0]));
    let (val_chunks8, val_rem) = values_last0.as_chunks::<8>();
    for (pair_idx, chunk8) in val_chunks8.iter().enumerate() {
        let (v0, v1) = super::super::neon::f16_bits_to_f32_x8(neon, chunk8);
        let g0 = pair_idx * 2;
        let g1 = g0 + 1;
        for h in 0..4 {
            sums1[h][last0_lane][g0] = v0.mul_add_precise(ws1_last0[h], sums1[h][last0_lane][g0]);
            sums1[h][last0_lane][g1] = v1.mul_add_precise(ws1_last0[h], sums1[h][last0_lane][g1]);
        }
    }
    let rem_base = val_chunks8.len() * 2;
    for (rem_idx, vals4) in val_rem.as_chunks::<4>().0.iter().enumerate() {
        let v = super::super::neon::f16_bits_to_f32_x4(neon, *vals4);
        let g = rem_base + rem_idx;
        for h in 0..4 {
            sums1[h][last0_lane][g] = v.mul_add_precise(ws1_last0[h], sums1[h][last0_lane][g]);
        }
    }
    let last1_lane = last1 % 16;
    let ws1_last1: [f32x4<Neon>; 4] =
        std::array::from_fn(|h| f32x4::splat(neon, weights1[h][last1]));
    for (g, values) in currents[1].as_chunks::<4>().0.iter().enumerate() {
        let v = f32x4::load_array(neon, *values);
        for h in 0..4 {
            sums1[h][last1_lane][g] = v.mul_add_precise(ws1_last1[h], sums1[h][last1_lane][g]);
        }
    }
    for index in token_count0..padded_dot_steps(token_count0) {
        let lane = index % 16;
        for head_sums in &mut sums0 {
            for sum in &mut head_sums[lane][..groups] {
                *sum = zero.mul_add_precise(zero, *sum);
            }
        }
    }
    for index in token_count1..padded_dot_steps(token_count1) {
        let lane = index % 16;
        for head_sums in &mut sums1 {
            for sum in &mut head_sums[lane][..groups] {
                *sum = zero.mul_add_precise(zero, *sum);
            }
        }
    }
    for (outputs, sums) in [(outputs0, &sums0), (outputs1, &sums1)] {
        for (h, output) in outputs.into_iter().enumerate() {
            for (group, out) in output.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                let lanes: [f32x4<Neon>; 4] = std::array::from_fn(|lane| {
                    (sums[h][lane][group] + sums[h][8 + lane][group])
                        + (sums[h][4 + lane][group] + sums[h][12 + lane][group])
                });
                let result = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])).to_array();
                for (out, value) in out.iter_mut().zip(result) {
                    *out = super::super::finite(value)?;
                }
            }
        }
    }
    Ok(())
}

/// Reduces a validated F16 key slice and F32 query slice with four NEON FMA registers.
pub(super) fn dot_f16(neon: Neon, key: &[u16], query: &[f32]) -> Result<f32> {
    let mut sums = [f32x4::splat(neon, 0.0); 4];
    let (key_chunks8, key_rem) = key.as_chunks::<8>();
    let (query_chunks8, query_rem) = query.as_chunks::<8>();
    let (key_pairs, key_odd) = key_chunks8.as_chunks::<2>();
    let (query_pairs, query_odd) = query_chunks8.as_chunks::<2>();
    for ([k8_0, k8_1], [q8_0, q8_1]) in key_pairs.iter().zip(query_pairs) {
        let (kv0, kv1) = super::super::neon::f16_bits_to_f32_x8(neon, k8_0);
        let qv0 = f32x4::from_slice(neon, &q8_0[..4]);
        let qv1 = f32x4::from_slice(neon, &q8_0[4..]);
        sums[0] = kv0.mul_add_precise(qv0, sums[0]);
        sums[1] = kv1.mul_add_precise(qv1, sums[1]);

        let (kv2, kv3) = super::super::neon::f16_bits_to_f32_x8(neon, k8_1);
        let qv2 = f32x4::from_slice(neon, &q8_1[..4]);
        let qv3 = f32x4::from_slice(neon, &q8_1[4..]);
        sums[2] = kv2.mul_add_precise(qv2, sums[2]);
        sums[3] = kv3.mul_add_precise(qv3, sums[3]);
    }
    for (k8, q8) in key_odd.iter().zip(query_odd) {
        let (kv0, kv1) = super::super::neon::f16_bits_to_f32_x8(neon, k8);
        let qv0 = f32x4::from_slice(neon, &q8[..4]);
        let qv1 = f32x4::from_slice(neon, &q8[4..]);
        sums[0] = kv0.mul_add_precise(qv0, sums[0]);
        sums[1] = kv1.mul_add_precise(qv1, sums[1]);
    }
    let rem_base = key_chunks8.len() * 2;
    let key_chunks4 = key_rem.as_chunks::<4>().0;
    let query_chunks4 = query_rem.as_chunks::<4>().0;
    for (offset, (k, q)) in key_chunks4.iter().zip(query_chunks4).enumerate() {
        let acc = &mut sums[(rem_base + offset) % 4];
        let kv = super::super::neon::f16_bits_to_f32_x4(neon, *k);
        let qv = f32x4::load_array(neon, *q);
        *acc = kv.mul_add_precise(qv, *acc);
    }
    let vector_steps = key_chunks8.len() * 8 + key_chunks4.len() * 4;
    let total_steps = padded_dot_steps(query.len());
    if vector_steps == total_steps {
        let lanes = ((sums[0] + sums[2]) + (sums[1] + sums[3])).to_array();
        return super::super::finite((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]));
    }
    let mut scalar_sums = sums.map(f32x4::to_array);
    for index in vector_steps..total_steps {
        let (value, weight) = if index < query.len() {
            (bonsai_ternary::half_to_float(key[index]), query[index])
        } else {
            (0.0, 0.0)
        };
        let sum = &mut scalar_sums[index % 16 / 4][index % 4];
        *sum = value.mul_add(weight, *sum);
    }
    let mut lanes = [0.0; 4];
    for (index, lane) in lanes.iter_mut().enumerate() {
        *lane = (scalar_sums[0][index] + scalar_sums[2][index])
            + (scalar_sums[1][index] + scalar_sums[3][index]);
    }
    super::super::finite((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
}

/// Reduces a validated F32 key slice and F32 query slice with four NEON FMA registers.
pub(super) fn dot_f32(neon: Neon, key: &[f32], query: &[f32]) -> Result<f32> {
    let mut sums = [f32x4::splat(neon, 0.0); 4];
    let key_chunks = key.as_chunks::<4>().0;
    let query_chunks = query.as_chunks::<4>().0;
    let (key_quads, key_rem4) = key_chunks.as_chunks::<4>();
    let (query_quads, query_rem4) = query_chunks.as_chunks::<4>();
    for (k_quad, q_quad) in key_quads.iter().zip(query_quads) {
        sums[0] = f32x4::load_array(neon, k_quad[0])
            .mul_add_precise(f32x4::load_array(neon, q_quad[0]), sums[0]);
        sums[1] = f32x4::load_array(neon, k_quad[1])
            .mul_add_precise(f32x4::load_array(neon, q_quad[1]), sums[1]);
        sums[2] = f32x4::load_array(neon, k_quad[2])
            .mul_add_precise(f32x4::load_array(neon, q_quad[2]), sums[2]);
        sums[3] = f32x4::load_array(neon, k_quad[3])
            .mul_add_precise(f32x4::load_array(neon, q_quad[3]), sums[3]);
    }
    for (lane, (k, q)) in key_rem4.iter().zip(query_rem4).enumerate() {
        let kv = f32x4::load_array(neon, *k);
        let qv = f32x4::load_array(neon, *q);
        sums[lane] = kv.mul_add_precise(qv, sums[lane]);
    }
    let vector_steps = key_chunks.len() * 4;
    let total_steps = padded_dot_steps(query.len());
    if vector_steps == total_steps {
        let lanes = ((sums[0] + sums[2]) + (sums[1] + sums[3])).to_array();
        return super::super::finite((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]));
    }
    let mut scalar_sums = sums.map(f32x4::to_array);
    for index in vector_steps..total_steps {
        let (value, weight) = if index < query.len() {
            (key[index], query[index])
        } else {
            (0.0, 0.0)
        };
        let sum = &mut scalar_sums[index % 16 / 4][index % 4];
        *sum = value.mul_add(weight, *sum);
    }
    let mut lanes = [0.0; 4];
    for (index, lane) in lanes.iter_mut().enumerate() {
        *lane = (scalar_sums[0][index] + scalar_sums[2][index])
            + (scalar_sums[1][index] + scalar_sums[3][index]);
    }
    super::super::finite((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
}

/// Reduces one validated F16 key slice against four F32 query slices sharing the same KV head.
pub(super) fn dot_f16_four(neon: Neon, key: &[u16], queries: [&[f32]; 4]) -> Result<[f32; 4]> {
    let mut sums = [[f32x4::splat(neon, 0.0); 4]; 4];
    let (key_chunks8, key_rem) = key.as_chunks::<8>();
    let q_chunks8 = queries.map(|q| q.as_chunks::<8>().0);
    let q_rems = queries.map(|q| q.as_chunks::<8>().1);
    let (key_pairs, key_odd) = key_chunks8.as_chunks::<2>();
    for (pair2_idx, [k8_0, k8_1]) in key_pairs.iter().enumerate() {
        let idx0 = pair2_idx * 2;
        let idx1 = idx0 + 1;
        let (kv0, kv1) = super::super::neon::f16_bits_to_f32_x8(neon, k8_0);
        let (kv2, kv3) = super::super::neon::f16_bits_to_f32_x8(neon, k8_1);
        for h in 0..4 {
            let q8_0 = &q_chunks8[h][idx0];
            let qv0 = f32x4::from_slice(neon, &q8_0[..4]);
            let qv1 = f32x4::from_slice(neon, &q8_0[4..]);
            sums[h][0] = kv0.mul_add_precise(qv0, sums[h][0]);
            sums[h][1] = kv1.mul_add_precise(qv1, sums[h][1]);

            let q8_1 = &q_chunks8[h][idx1];
            let qv2 = f32x4::from_slice(neon, &q8_1[..4]);
            let qv3 = f32x4::from_slice(neon, &q8_1[4..]);
            sums[h][2] = kv2.mul_add_precise(qv2, sums[h][2]);
            sums[h][3] = kv3.mul_add_precise(qv3, sums[h][3]);
        }
    }
    if let Some(k8) = key_odd.first() {
        let idx = key_pairs.len() * 2;
        let (kv0, kv1) = super::super::neon::f16_bits_to_f32_x8(neon, k8);
        for h in 0..4 {
            let q8 = &q_chunks8[h][idx];
            let qv0 = f32x4::from_slice(neon, &q8[..4]);
            let qv1 = f32x4::from_slice(neon, &q8[4..]);
            sums[h][0] = kv0.mul_add_precise(qv0, sums[h][0]);
            sums[h][1] = kv1.mul_add_precise(qv1, sums[h][1]);
        }
    }
    let rem_base = key_chunks8.len() * 2;
    let key_chunks4 = key_rem.as_chunks::<4>().0;
    let q_chunks4 = q_rems.map(|r| r.as_chunks::<4>().0);
    for (offset, k) in key_chunks4.iter().enumerate() {
        let lane = (rem_base + offset) % 4;
        let kv = super::super::neon::f16_bits_to_f32_x4(neon, *k);
        for h in 0..4 {
            let qv = f32x4::load_array(neon, q_chunks4[h][offset]);
            sums[h][lane] = kv.mul_add_precise(qv, sums[h][lane]);
        }
    }
    let len = queries[0].len();
    let vector_steps = key_chunks8.len() * 8 + key_chunks4.len() * 4;
    let total_steps = padded_dot_steps(len);
    let mut out = [0.0_f32; 4];
    if vector_steps == total_steps {
        for h in 0..4 {
            let lanes = ((sums[h][0] + sums[h][2]) + (sums[h][1] + sums[h][3])).to_array();
            out[h] = super::super::finite((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))?;
        }
        return Ok(out);
    }
    for h in 0..4 {
        let mut scalar_sums = sums[h].map(f32x4::to_array);
        for index in vector_steps..total_steps {
            let (value, weight) = if index < len {
                (bonsai_ternary::half_to_float(key[index]), queries[h][index])
            } else {
                (0.0, 0.0)
            };
            let sum = &mut scalar_sums[index % 16 / 4][index % 4];
            *sum = value.mul_add(weight, *sum);
        }
        let mut lanes = [0.0; 4];
        for (index, lane) in lanes.iter_mut().enumerate() {
            *lane = (scalar_sums[0][index] + scalar_sums[2][index])
                + (scalar_sums[1][index] + scalar_sums[3][index]);
        }
        out[h] = super::super::finite((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))?;
    }
    Ok(out)
}

#[inline]
fn dot_f16_two_keys_four_queries(
    neon: Neon,
    keys: [&[u16]; 2],
    queries: [&[f32]; 4],
) -> Result<[[f32; 4]; 2]> {
    let len = queries[0].len();
    let total_steps = padded_dot_steps(len);
    let (k0_chunks8, k0_rem) = keys[0].as_chunks::<8>();
    let (k1_chunks8, _) = keys[1].as_chunks::<8>();
    let vector_steps = k0_chunks8.len() * 8;
    if vector_steps != total_steps || !k0_rem.is_empty() {
        return Ok([
            dot_f16_four(neon, keys[0], queries)?,
            dot_f16_four(neon, keys[1], queries)?,
        ]);
    }
    let q_chunks8 = queries.map(|q| q.as_chunks::<8>().0);
    let mut s0 = [[f32x4::splat(neon, 0.0); 4]; 2];
    let mut s1 = [[f32x4::splat(neon, 0.0); 4]; 2];
    for pair_idx in (0..k0_chunks8.len()).step_by(2) {
        let (k0_v0, k0_v1) = super::super::neon::f16_bits_to_f32_x8(neon, &k0_chunks8[pair_idx]);
        let (k1_v0, k1_v1) = super::super::neon::f16_bits_to_f32_x8(neon, &k1_chunks8[pair_idx]);
        for h in 0..4 {
            let q8 = &q_chunks8[h][pair_idx];
            let qv0 = f32x4::from_slice(neon, &q8[..4]);
            let qv1 = f32x4::from_slice(neon, &q8[4..]);
            s0[0][h] = k0_v0.mul_add_precise(qv0, s0[0][h]);
            s1[0][h] = k0_v1.mul_add_precise(qv1, s1[0][h]);
            s0[1][h] = k1_v0.mul_add_precise(qv0, s0[1][h]);
            s1[1][h] = k1_v1.mul_add_precise(qv1, s1[1][h]);
        }
    }
    let mut s2 = [[f32x4::splat(neon, 0.0); 4]; 2];
    let mut s3 = [[f32x4::splat(neon, 0.0); 4]; 2];
    for pair_idx in (1..k0_chunks8.len()).step_by(2) {
        let (k0_v0, k0_v1) = super::super::neon::f16_bits_to_f32_x8(neon, &k0_chunks8[pair_idx]);
        let (k1_v0, k1_v1) = super::super::neon::f16_bits_to_f32_x8(neon, &k1_chunks8[pair_idx]);
        for h in 0..4 {
            let q8 = &q_chunks8[h][pair_idx];
            let qv0 = f32x4::from_slice(neon, &q8[..4]);
            let qv1 = f32x4::from_slice(neon, &q8[4..]);
            s2[0][h] = k0_v0.mul_add_precise(qv0, s2[0][h]);
            s3[0][h] = k0_v1.mul_add_precise(qv1, s3[0][h]);
            s2[1][h] = k1_v0.mul_add_precise(qv0, s2[1][h]);
            s3[1][h] = k1_v1.mul_add_precise(qv1, s3[1][h]);
        }
    }
    let mut out = [[0.0_f32; 4]; 2];
    for k in 0..2 {
        for h in 0..4 {
            let lanes = ((s0[k][h] + s2[k][h]) + (s1[k][h] + s3[k][h])).to_array();
            out[k][h] = super::super::finite((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))?;
        }
    }
    Ok(out)
}

#[inline]
fn dot_f16_four_keys_four_queries_stream(
    neon: Neon,
    k_chunks8: [&[[u16; 8]]; 4],
    q_chunks8: [&[[f32; 8]]; 4],
    start_pair: usize,
    half_offset: usize,
) -> [[f32x4<Neon>; 4]; 4] {
    let mut acc = [[f32x4::splat(neon, 0.0); 4]; 4];
    for pair_idx in (start_pair..k_chunks8[0].len()).step_by(2) {
        let (k0, k1) = super::super::neon::f16_pair_x4_to_f32_x4(
            neon,
            &k_chunks8[0][pair_idx][half_offset..half_offset + 4],
            &k_chunks8[1][pair_idx][half_offset..half_offset + 4],
        );
        let (k2, k3) = super::super::neon::f16_pair_x4_to_f32_x4(
            neon,
            &k_chunks8[2][pair_idx][half_offset..half_offset + 4],
            &k_chunks8[3][pair_idx][half_offset..half_offset + 4],
        );
        for h in 0..4 {
            let qv = f32x4::from_slice(neon, &q_chunks8[h][pair_idx][half_offset..half_offset + 4]);
            acc[0][h] = k0.mul_add_precise(qv, acc[0][h]);
            acc[1][h] = k1.mul_add_precise(qv, acc[1][h]);
            acc[2][h] = k2.mul_add_precise(qv, acc[2][h]);
            acc[3][h] = k3.mul_add_precise(qv, acc[3][h]);
        }
    }
    acc
}

/// Reduces four validated F16 key slices against four F32 query slices sharing the same KV head,
/// sharing each query vector load across all four keys without exceeding 16 accumulator registers per pass.
pub(super) fn dot_f16_four_keys_four_queries(
    neon: Neon,
    keys: [&[u16]; 4],
    queries: [&[f32]; 4],
) -> Result<[[f32; 4]; 4]> {
    let len = queries[0].len();
    let total_steps = padded_dot_steps(len);
    let (k0_chunks8, k0_rem) = keys[0].as_chunks::<8>();
    let (k1_chunks8, k1_rem) = keys[1].as_chunks::<8>();
    let (k2_chunks8, k2_rem) = keys[2].as_chunks::<8>();
    let (k3_chunks8, k3_rem) = keys[3].as_chunks::<8>();
    let vector_steps = k0_chunks8.len() * 8;
    if vector_steps != total_steps
        || !k0_rem.is_empty()
        || !k1_rem.is_empty()
        || !k2_rem.is_empty()
        || !k3_rem.is_empty()
    {
        let [r0, r1] = dot_f16_two_keys_four_queries(neon, [keys[0], keys[1]], queries)?;
        let [r2, r3] = dot_f16_two_keys_four_queries(neon, [keys[2], keys[3]], queries)?;
        return Ok([r0, r1, r2, r3]);
    }
    let key_blocks = [k0_chunks8, k1_chunks8, k2_chunks8, k3_chunks8];
    let q_chunks8 = queries.map(|q| q.as_chunks::<8>().0);
    let s0 = dot_f16_four_keys_four_queries_stream(neon, key_blocks, q_chunks8, 0, 0);
    let s1 = dot_f16_four_keys_four_queries_stream(neon, key_blocks, q_chunks8, 0, 4);
    let s2 = dot_f16_four_keys_four_queries_stream(neon, key_blocks, q_chunks8, 1, 0);
    let s3 = dot_f16_four_keys_four_queries_stream(neon, key_blocks, q_chunks8, 1, 4);
    let mut out = [[0.0_f32; 4]; 4];
    for k in 0..4 {
        for h in 0..4 {
            let lanes = ((s0[k][h] + s2[k][h]) + (s1[k][h] + s3[k][h])).to_array();
            out[k][h] = super::super::finite((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))?;
        }
    }
    Ok(out)
}

/// Reduces one validated F16 key slice against two tokens x four F32 query slices sharing a KV head,
/// converting each 8-element F16 key block once across all eight queries.
pub(super) fn dot_f16_two_token_four_queries(
    neon: Neon,
    key: &[u16],
    queries0: [&[f32]; 4],
    queries1: [&[f32]; 4],
) -> Result<([f32; 4], [f32; 4])> {
    let len = queries0[0].len();
    let total_steps = padded_dot_steps(len);
    let (k_chunks8, k_rem) = key.as_chunks::<8>();
    let vector_steps = k_chunks8.len() * 8;
    if vector_steps != total_steps || !k_rem.is_empty() {
        return Ok((
            dot_f16_four(neon, key, queries0)?,
            dot_f16_four(neon, key, queries1)?,
        ));
    }
    let q0_chunks8 = queries0.map(|q| q.as_chunks::<8>().0);
    let q1_chunks8 = queries1.map(|q| q.as_chunks::<8>().0);
    let mut s0_0 = [f32x4::splat(neon, 0.0); 4];
    let mut s1_0 = [f32x4::splat(neon, 0.0); 4];
    let mut s0_1 = [f32x4::splat(neon, 0.0); 4];
    let mut s1_1 = [f32x4::splat(neon, 0.0); 4];
    for pair_idx in (0..k_chunks8.len()).step_by(2) {
        let (kv0, kv1) = super::super::neon::f16_bits_to_f32_x8(neon, &k_chunks8[pair_idx]);
        for h in 0..4 {
            let q8_0 = &q0_chunks8[h][pair_idx];
            s0_0[h] = kv0.mul_add_precise(f32x4::from_slice(neon, &q8_0[..4]), s0_0[h]);
            s1_0[h] = kv1.mul_add_precise(f32x4::from_slice(neon, &q8_0[4..]), s1_0[h]);
            let q8_1 = &q1_chunks8[h][pair_idx];
            s0_1[h] = kv0.mul_add_precise(f32x4::from_slice(neon, &q8_1[..4]), s0_1[h]);
            s1_1[h] = kv1.mul_add_precise(f32x4::from_slice(neon, &q8_1[4..]), s1_1[h]);
        }
    }
    let mut s2_0 = [f32x4::splat(neon, 0.0); 4];
    let mut s3_0 = [f32x4::splat(neon, 0.0); 4];
    let mut s2_1 = [f32x4::splat(neon, 0.0); 4];
    let mut s3_1 = [f32x4::splat(neon, 0.0); 4];
    for pair_idx in (1..k_chunks8.len()).step_by(2) {
        let (kv0, kv1) = super::super::neon::f16_bits_to_f32_x8(neon, &k_chunks8[pair_idx]);
        for h in 0..4 {
            let q8_0 = &q0_chunks8[h][pair_idx];
            s2_0[h] = kv0.mul_add_precise(f32x4::from_slice(neon, &q8_0[..4]), s2_0[h]);
            s3_0[h] = kv1.mul_add_precise(f32x4::from_slice(neon, &q8_0[4..]), s3_0[h]);
            let q8_1 = &q1_chunks8[h][pair_idx];
            s2_1[h] = kv0.mul_add_precise(f32x4::from_slice(neon, &q8_1[..4]), s2_1[h]);
            s3_1[h] = kv1.mul_add_precise(f32x4::from_slice(neon, &q8_1[4..]), s3_1[h]);
        }
    }
    let mut out0 = [0.0_f32; 4];
    let mut out1 = [0.0_f32; 4];
    for h in 0..4 {
        let lanes0 = ((s0_0[h] + s2_0[h]) + (s1_0[h] + s3_0[h])).to_array();
        out0[h] = super::super::finite((lanes0[0] + lanes0[1]) + (lanes0[2] + lanes0[3]))?;
    }
    for h in 0..4 {
        let lanes1 = ((s0_1[h] + s2_1[h]) + (s1_1[h] + s3_1[h])).to_array();
        out1[h] = super::super::finite((lanes1[0] + lanes1[1]) + (lanes1[2] + lanes1[3]))?;
    }
    Ok((out0, out1))
}

type TwoKeyStreamPair = ([[f32x4<Neon>; 4]; 2], [[f32x4<Neon>; 4]; 2]);
type TwoKeyScorePair = ([[f32; 4]; 2], [[f32; 4]; 2]);

#[inline]
fn dot_f16_two_keys_two_token_four_queries_stream(
    neon: Neon,
    k_chunks8: [&[[u16; 8]]; 2],
    q0_chunks8: [&[[f32; 8]]; 4],
    q1_chunks8: [&[[f32; 8]]; 4],
    start_pair: usize,
    half_offset: usize,
) -> TwoKeyStreamPair {
    let mut acc0 = [[f32x4::splat(neon, 0.0); 4]; 2];
    let mut acc1 = [[f32x4::splat(neon, 0.0); 4]; 2];
    for pair_idx in (start_pair..k_chunks8[0].len()).step_by(2) {
        let (k0, k1) = super::super::neon::f16_pair_x4_to_f32_x4(
            neon,
            &k_chunks8[0][pair_idx][half_offset..half_offset + 4],
            &k_chunks8[1][pair_idx][half_offset..half_offset + 4],
        );
        for h in 0..4 {
            let qv0 =
                f32x4::from_slice(neon, &q0_chunks8[h][pair_idx][half_offset..half_offset + 4]);
            acc0[0][h] = k0.mul_add_precise(qv0, acc0[0][h]);
            acc0[1][h] = k1.mul_add_precise(qv0, acc0[1][h]);
            let qv1 =
                f32x4::from_slice(neon, &q1_chunks8[h][pair_idx][half_offset..half_offset + 4]);
            acc1[0][h] = k0.mul_add_precise(qv1, acc1[0][h]);
            acc1[1][h] = k1.mul_add_precise(qv1, acc1[1][h]);
        }
    }
    (acc0, acc1)
}

#[inline]
fn dot_f16_two_keys_two_token_four_queries(
    neon: Neon,
    keys: [&[u16]; 2],
    queries0: [&[f32]; 4],
    queries1: [&[f32]; 4],
) -> Result<TwoKeyScorePair> {
    let len = queries0[0].len();
    let total_steps = padded_dot_steps(len);
    let (k0_chunks8, k0_rem) = keys[0].as_chunks::<8>();
    let (k1_chunks8, k1_rem) = keys[1].as_chunks::<8>();
    let vector_steps = k0_chunks8.len() * 8;
    if vector_steps != total_steps || !k0_rem.is_empty() || !k1_rem.is_empty() {
        let (r0_0, r0_1) = dot_f16_two_token_four_queries(neon, keys[0], queries0, queries1)?;
        let (r1_0, r1_1) = dot_f16_two_token_four_queries(neon, keys[1], queries0, queries1)?;
        return Ok(([r0_0, r1_0], [r0_1, r1_1]));
    }
    let key_blocks = [k0_chunks8, k1_chunks8];
    let q0_chunks8 = queries0.map(|q| q.as_chunks::<8>().0);
    let q1_chunks8 = queries1.map(|q| q.as_chunks::<8>().0);
    let (s0_0, s0_1) = dot_f16_two_keys_two_token_four_queries_stream(
        neon, key_blocks, q0_chunks8, q1_chunks8, 0, 0,
    );
    let (s1_0, s1_1) = dot_f16_two_keys_two_token_four_queries_stream(
        neon, key_blocks, q0_chunks8, q1_chunks8, 0, 4,
    );
    let (s2_0, s2_1) = dot_f16_two_keys_two_token_four_queries_stream(
        neon, key_blocks, q0_chunks8, q1_chunks8, 1, 0,
    );
    let (s3_0, s3_1) = dot_f16_two_keys_two_token_four_queries_stream(
        neon, key_blocks, q0_chunks8, q1_chunks8, 1, 4,
    );
    let mut out0 = [[0.0_f32; 4]; 2];
    let mut out1 = [[0.0_f32; 4]; 2];
    for k in 0..2 {
        for h in 0..4 {
            let lanes0 = ((s0_0[k][h] + s2_0[k][h]) + (s1_0[k][h] + s3_0[k][h])).to_array();
            out0[k][h] = super::super::finite((lanes0[0] + lanes0[1]) + (lanes0[2] + lanes0[3]))?;
        }
        for h in 0..4 {
            let lanes1 = ((s0_1[k][h] + s2_1[k][h]) + (s1_1[k][h] + s3_1[k][h])).to_array();
            out1[k][h] = super::super::finite((lanes1[0] + lanes1[1]) + (lanes1[2] + lanes1[3]))?;
        }
    }
    Ok((out0, out1))
}

/// Reduces four validated F16 key slices against two tokens x four F32 query slices sharing a KV head.
pub(super) fn dot_f16_four_keys_two_token_four_queries(
    neon: Neon,
    keys: [&[u16]; 4],
    queries0: [&[f32]; 4],
    queries1: [&[f32]; 4],
) -> Result<(super::FourKeyFourQueryScores, super::FourKeyFourQueryScores)> {
    let ([r0_0, r1_0], [r0_1, r1_1]) =
        dot_f16_two_keys_two_token_four_queries(neon, [keys[0], keys[1]], queries0, queries1)?;
    let ([r2_0, r3_0], [r2_1, r3_1]) =
        dot_f16_two_keys_two_token_four_queries(neon, [keys[2], keys[3]], queries0, queries1)?;
    Ok(([r0_0, r1_0, r2_0, r3_0], [r0_1, r1_1, r2_1, r3_1]))
}

/// Reduces one validated F32 key slice against four F32 query slices sharing the same KV head.
pub(super) fn dot_f32_four(neon: Neon, key: &[f32], queries: [&[f32]; 4]) -> Result<[f32; 4]> {
    let mut sums = [[f32x4::splat(neon, 0.0); 4]; 4];
    let key_chunks = key.as_chunks::<4>().0;
    let q_chunks = queries.map(|q| q.as_chunks::<4>().0);
    let (key_quads, key_rem4) = key_chunks.as_chunks::<4>();
    for (quad_idx, k_quad) in key_quads.iter().enumerate() {
        let base = quad_idx * 4;
        let kv0 = f32x4::load_array(neon, k_quad[0]);
        let kv1 = f32x4::load_array(neon, k_quad[1]);
        let kv2 = f32x4::load_array(neon, k_quad[2]);
        let kv3 = f32x4::load_array(neon, k_quad[3]);
        for h in 0..4 {
            let qh = &q_chunks[h][base..base + 4];
            sums[h][0] = kv0.mul_add_precise(f32x4::load_array(neon, qh[0]), sums[h][0]);
            sums[h][1] = kv1.mul_add_precise(f32x4::load_array(neon, qh[1]), sums[h][1]);
            sums[h][2] = kv2.mul_add_precise(f32x4::load_array(neon, qh[2]), sums[h][2]);
            sums[h][3] = kv3.mul_add_precise(f32x4::load_array(neon, qh[3]), sums[h][3]);
        }
    }
    let rem_base = key_quads.len() * 4;
    for (lane, k) in key_rem4.iter().enumerate() {
        let chunk = rem_base + lane;
        let kv = f32x4::load_array(neon, *k);
        for h in 0..4 {
            let qv = f32x4::load_array(neon, q_chunks[h][chunk]);
            sums[h][lane] = kv.mul_add_precise(qv, sums[h][lane]);
        }
    }
    let len = queries[0].len();
    let vector_steps = key_chunks.len() * 4;
    let total_steps = padded_dot_steps(len);
    let mut out = [0.0_f32; 4];
    if vector_steps == total_steps {
        for h in 0..4 {
            let lanes = ((sums[h][0] + sums[h][2]) + (sums[h][1] + sums[h][3])).to_array();
            out[h] = super::super::finite((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))?;
        }
        return Ok(out);
    }
    for h in 0..4 {
        let mut scalar_sums = sums[h].map(f32x4::to_array);
        for index in vector_steps..total_steps {
            let (value, weight) = if index < len {
                (key[index], queries[h][index])
            } else {
                (0.0, 0.0)
            };
            let sum = &mut scalar_sums[index % 16 / 4][index % 4];
            *sum = value.mul_add(weight, *sum);
        }
        let mut lanes = [0.0; 4];
        for (index, lane) in lanes.iter_mut().enumerate() {
            *lane = (scalar_sums[0][index] + scalar_sums[2][index])
                + (scalar_sums[1][index] + scalar_sums[3][index]);
        }
        out[h] = super::super::finite((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))?;
    }
    Ok(out)
}

/// Rotates validated 256-wide Swift heads (with 64-element rotary prefix) in NEON `f32x4`.
pub(super) fn rope_with_aliasing(
    neon: Neon,
    input: &mut [f32],
    frequencies: &[(f32, f32); 32],
    in_place: bool,
) {
    let cos_vecs: [f32x4<Neon>; 8] = std::array::from_fn(|chunk| {
        let base = chunk * 4;
        f32x4::load_array(
            neon,
            [
                frequencies[base].0,
                frequencies[base + 1].0,
                frequencies[base + 2].0,
                frequencies[base + 3].0,
            ],
        )
    });
    let sin_vecs: [f32x4<Neon>; 8] = std::array::from_fn(|chunk| {
        let base = chunk * 4;
        f32x4::load_array(
            neon,
            [
                frequencies[base].1,
                frequencies[base + 1].1,
                frequencies[base + 2].1,
                frequencies[base + 3].1,
            ],
        )
    });
    for head in input.as_chunks_mut::<256>().0 {
        let (first_half, second_half) = head[..64].split_at_mut(32);
        for (((first_chunk, second_chunk), &cos), &sin) in first_half
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(second_half.as_chunks_mut::<4>().0)
            .zip(&cos_vecs)
            .zip(&sin_vecs)
        {
            let first = f32x4::from_slice(neon, first_chunk);
            let second = f32x4::from_slice(neon, second_chunk);
            let rotated_first = if in_place {
                first.mul_add_precise(cos, -(second * sin))
            } else {
                (-second).mul_add_precise(sin, first * cos)
            };
            let rotated_second = second.mul_add_precise(cos, first * sin);
            rotated_first.store_slice(first_chunk);
            rotated_second.store_slice(second_chunk);
        }
    }
}

#[cfg(test)]
mod tests {
    use bonsai_ternary::half_to_float;
    use fearless_simd::{Neon, f32x4, prelude::*};

    use super::{
        VALUE_TILE, VALUE_TILE_FOUR, VALUE_TILE_FOUR_TWO_TOKENS, value_tile, value_tile_four,
        value_tile_four_two_tokens,
    };

    #[derive(serde::Deserialize)]
    struct Capture {
        weighted_value: WeightedValue,
    }

    #[derive(serde::Deserialize)]
    #[allow(
        clippy::struct_field_names,
        reason = "Field names match the immutable JSON fixture schema."
    )]
    struct WeightedValue {
        value_half_bits: Vec<u16>,
        probability_half_bits: Vec<u16>,
        native_bits: u32,
    }

    #[test]
    fn neon_value_tile_baseline_is_available_and_accumulator_remains_two_kib() -> crate::Result<()>
    {
        let neon = fearless_simd::Level::baseline()
            .as_neon()
            .ok_or_else(|| crate::invalid("required attention baseline NEON token is absent"))?;
        assert_eq!(size_of::<[[f32x4<Neon>; VALUE_TILE / 4]; 16]>(), 2048);
        let capture: Capture = serde_json::from_str(include_str!(
            "../../tests/fixtures/swift-attention-dot.json"
        ))?;
        let captured = capture.weighted_value;
        assert_ne!(captured.value_half_bits, [0_u16; 0]);
        let last = captured.value_half_bits.len() - 1;
        let rows: Vec<_> = captured.value_half_bits[..last]
            .iter()
            .map(|&value| [value; VALUE_TILE])
            .collect();
        let current = [half_to_float(captured.value_half_bits[last]); VALUE_TILE];
        let weights: Vec<_> = captured
            .probability_half_bits
            .into_iter()
            .map(half_to_float)
            .collect();
        let mut output = [f32::NAN; VALUE_TILE];
        value_tile(
            neon,
            rows.iter().map(<[u16; VALUE_TILE]>::as_slice),
            &current,
            &weights,
            &(0..VALUE_TILE),
            &mut output,
        )?;
        assert!(
            output
                .iter()
                .all(|value| value.to_bits() == captured.native_bits)
        );
        Ok(())
    }

    #[test]
    fn neon_f16_conversions_preserve_all_binary16_bit_patterns() -> crate::Result<()> {
        let neon = fearless_simd::Level::baseline()
            .as_neon()
            .ok_or_else(|| crate::invalid("required attention baseline NEON token is absent"))?;
        let all_bits: Vec<u16> = (0..=u16::MAX).collect();
        for chunk8 in all_bits.as_chunks::<8>().0 {
            let (v0, v1) = super::super::super::neon::f16_bits_to_f32_x8(neon, chunk8);
            let a0 = v0.to_array();
            let a1 = v1.to_array();
            let c0 = super::super::super::neon::f16_bits_to_f32_x4(
                neon,
                [chunk8[0], chunk8[1], chunk8[2], chunk8[3]],
            )
            .to_array();
            let c1 = super::super::super::neon::f16_bits_to_f32_x4(
                neon,
                [chunk8[4], chunk8[5], chunk8[6], chunk8[7]],
            )
            .to_array();
            for lane in 0..4 {
                let expected0 = half_to_float(chunk8[lane]).to_bits();
                let expected1 = half_to_float(chunk8[lane + 4]).to_bits();
                assert_eq!(a0[lane].to_bits(), expected0);
                assert_eq!(c0[lane].to_bits(), expected0);
                assert_eq!(a1[lane].to_bits(), expected1);
                assert_eq!(c1[lane].to_bits(), expected1);
            }
        }
        Ok(())
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "Verify 1-token and 2-token 4-head value tiles across history lengths and tile widths."
    )]
    fn neon_value_tile_four_two_tokens_matches_single_token_four_head_tiles() -> crate::Result<()> {
        let neon = fearless_simd::Level::baseline()
            .as_neon()
            .ok_or_else(|| crate::invalid("required attention baseline NEON token is absent"))?;
        let cases = [0_u16, 0x8000, 1, 0x0400, 0x3c00, 0xbc00, 0x3800, 0xb800];
        for history in [0, 1, 7, 15, 16, 17, 31] {
            let rows: Vec<[u16; VALUE_TILE_FOUR]> = (0..=history)
                .map(|pos| std::array::from_fn(|ch| cases[(pos * 3 + ch) % cases.len()]))
                .collect();
            let current0: [f32; VALUE_TILE_FOUR] =
                std::array::from_fn(|ch| half_to_float(rows[history][ch]));
            let current1: [f32; VALUE_TILE_FOUR] =
                std::array::from_fn(|ch| half_to_float(cases[(history + ch + 5) % cases.len()]));
            let weights0: [[f32; 33]; 4] = std::array::from_fn(|h| {
                std::array::from_fn(|pos| [0.5_f32, -0.25, 0.125, -0.0, 0.0][(pos + h) % 5])
            });
            let weights1: [[f32; 33]; 4] = std::array::from_fn(|h| {
                std::array::from_fn(|pos| [0.25_f32, 0.125, -0.5, 0.0, -0.0][(pos + h + 1) % 5])
            });
            let w0_refs: [&[f32]; 4] = std::array::from_fn(|h| &weights0[h][..=history]);
            let w1_refs: [&[f32]; 4] = std::array::from_fn(|h| &weights1[h][..history + 2]);
            for tile_len in [12, 24, VALUE_TILE, VALUE_TILE_FOUR_TWO_TOKENS] {
                let mut exp0 = [[0.0_f32; VALUE_TILE_FOUR_TWO_TOKENS]; 4];
                let mut exp1 = [[0.0_f32; VALUE_TILE_FOUR_TWO_TOKENS]; 4];
                let (e0_0, e0_rest) = exp0.split_at_mut(1);
                let (e0_1, e0_rest) = e0_rest.split_at_mut(1);
                let (e0_2, e0_3) = e0_rest.split_at_mut(1);
                value_tile_four(
                    neon,
                    rows[..history]
                        .iter()
                        .map(<[u16; VALUE_TILE_FOUR]>::as_slice),
                    &current0[..tile_len],
                    w0_refs,
                    &(0..tile_len),
                    [
                        &mut e0_0[0][..tile_len],
                        &mut e0_1[0][..tile_len],
                        &mut e0_2[0][..tile_len],
                        &mut e0_3[0][..tile_len],
                    ],
                )?;
                let (e1_0, e1_rest) = exp1.split_at_mut(1);
                let (e1_1, e1_rest) = e1_rest.split_at_mut(1);
                let (e1_2, e1_3) = e1_rest.split_at_mut(1);
                value_tile_four(
                    neon,
                    rows[..=history]
                        .iter()
                        .map(<[u16; VALUE_TILE_FOUR]>::as_slice),
                    &current1[..tile_len],
                    w1_refs,
                    &(0..tile_len),
                    [
                        &mut e1_0[0][..tile_len],
                        &mut e1_1[0][..tile_len],
                        &mut e1_2[0][..tile_len],
                        &mut e1_3[0][..tile_len],
                    ],
                )?;

                let mut act0 = [[f32::NAN; VALUE_TILE_FOUR_TWO_TOKENS]; 4];
                let mut act1 = [[f32::NAN; VALUE_TILE_FOUR_TWO_TOKENS]; 4];
                let (a0_0, a0_rest) = act0.split_at_mut(1);
                let (a0_1, a0_rest) = a0_rest.split_at_mut(1);
                let (a0_2, a0_3) = a0_rest.split_at_mut(1);
                let (a1_0, a1_rest) = act1.split_at_mut(1);
                let (a1_1, a1_rest) = a1_rest.split_at_mut(1);
                let (a1_2, a1_3) = a1_rest.split_at_mut(1);
                value_tile_four_two_tokens(
                    neon,
                    rows[..=history]
                        .iter()
                        .map(<[u16; VALUE_TILE_FOUR]>::as_slice),
                    [&current0[..tile_len], &current1[..tile_len]],
                    w0_refs,
                    w1_refs,
                    &(0..tile_len),
                    [
                        &mut a0_0[0][..tile_len],
                        &mut a0_1[0][..tile_len],
                        &mut a0_2[0][..tile_len],
                        &mut a0_3[0][..tile_len],
                    ],
                    [
                        &mut a1_0[0][..tile_len],
                        &mut a1_1[0][..tile_len],
                        &mut a1_2[0][..tile_len],
                        &mut a1_3[0][..tile_len],
                    ],
                )?;
                for h in 0..4 {
                    for ch in 0..tile_len {
                        assert_eq!(act0[h][ch].to_bits(), exp0[h][ch].to_bits());
                        assert_eq!(act1[h][ch].to_bits(), exp1[h][ch].to_bits());
                    }
                }
            }
            let mut wide = [[0.0_f32; VALUE_TILE_FOUR]; 4];
            let (w0, w_rest) = wide.split_at_mut(1);
            let (w1, w_rest) = w_rest.split_at_mut(1);
            let (w2, w3) = w_rest.split_at_mut(1);
            value_tile_four(
                neon,
                rows[..history]
                    .iter()
                    .map(<[u16; VALUE_TILE_FOUR]>::as_slice),
                &current0,
                w0_refs,
                &(0..VALUE_TILE_FOUR),
                [&mut w0[0], &mut w1[0], &mut w2[0], &mut w3[0]],
            )?;
            for (h, head_out) in wide.iter().enumerate() {
                for tile_idx in 0..(VALUE_TILE_FOUR / VALUE_TILE) {
                    let start = tile_idx * VALUE_TILE;
                    let mut single = [0.0_f32; VALUE_TILE];
                    value_tile(
                        neon,
                        rows[..history]
                            .iter()
                            .map(<[u16; VALUE_TILE_FOUR]>::as_slice),
                        &current0[start..start + VALUE_TILE],
                        w0_refs[h],
                        &(start..start + VALUE_TILE),
                        &mut single,
                    )?;
                    for ch in 0..VALUE_TILE {
                        assert_eq!(head_out[start + ch].to_bits(), single[ch].to_bits());
                    }
                }
            }
        }
        Ok(())
    }
}
