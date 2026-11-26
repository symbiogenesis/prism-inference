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
    let mut s0 = [zero; HEADS];
    let mut s1 = [zero; HEADS];
    let mut s2 = [zero; HEADS];
    let mut s3 = [zero; HEADS];
    // Update the low and high eight-element halves sequentially inside each
    // 16-element stride so six heads keep peak live registers <= 27 with zero spills.
    let (chunks, remainder) = key.as_chunks::<16>();
    for (index, chunk) in chunks.iter().enumerate() {
        let base = index * 16;
        let (halves, _) = chunk.as_chunks::<8>();
        let (first, second) = crate::kernels::neon::f16_bits_to_f32_x8(neon, &halves[0]);
        for head in 0..HEADS {
            let query = &queries[head][base..base + 8];
            s0[head] = first.mul_add_precise(f32x4::from_slice(neon, &query[..4]), s0[head]);
            s1[head] = second.mul_add_precise(f32x4::from_slice(neon, &query[4..]), s1[head]);
        }
        let (third, fourth) = crate::kernels::neon::f16_bits_to_f32_x8(neon, &halves[1]);
        for head in 0..HEADS {
            let query = &queries[head][base + 8..base + 16];
            s2[head] = third.mul_add_precise(f32x4::from_slice(neon, &query[..4]), s2[head]);
            s3[head] = fourth.mul_add_precise(f32x4::from_slice(neon, &query[4..]), s3[head]);
        }
    }
    let vector_steps = chunks.len() * 16;
    let (tail_chunks, _) = remainder.as_chunks::<8>();
    if let Some(chunk) = tail_chunks.first() {
        let (first, second) = crate::kernels::neon::f16_bits_to_f32_x8(neon, chunk);
        for head in 0..HEADS {
            let query = &queries[head][vector_steps..vector_steps + 8];
            s0[head] = first.mul_add_precise(f32x4::from_slice(neon, &query[..4]), s0[head]);
            s1[head] = second.mul_add_precise(f32x4::from_slice(neon, &query[4..]), s1[head]);
        }
    }
    let vector_steps = vector_steps + tail_chunks.len() * 8;
    finish_dot_registers(neon, key, queries, s0, s1, s2, s3, vector_steps)
}

/// Computes two-token × six-head attention scores across one cached F16 key in two
/// half-stride passes (`[0..8]` then `[8..16]`) to hold <= 28 live NEON registers
/// without spills while preserving the 4-accumulator reduction tree bit-for-bit.
pub(super) fn dot_pair_six_neon(
    neon: Neon,
    key: &[u16],
    queries: [[&[f32]; 6]; 2],
) -> Result<[[f32; 6]; 2]> {
    let [first, second] = queries;
    let flat_queries: [&[f32]; 12] = std::array::from_fn(|head| {
        if head < 6 {
            first[head]
        } else {
            second[head - 6]
        }
    });
    let (chunks8, _) = key.as_chunks::<8>();
    let (s0, s1) = dot_single_key_pass(neon, chunks8, 0, flat_queries);
    let (s2, s3) = dot_single_key_pass(neon, chunks8, 1, flat_queries);
    let values = finish_dot_registers(neon, key, flat_queries, s0, s1, s2, s3, chunks8.len() * 8)?;
    Ok(std::array::from_fn(|token| {
        std::array::from_fn(|head| values[token * 6 + head])
    }))
}

#[inline]
fn dot_single_key_pass<const HEADS: usize>(
    neon: Neon,
    chunks8: &[[u16; 8]],
    start_chunk: usize,
    queries: [&[f32]; HEADS],
) -> ([f32x4<Neon>; HEADS], [f32x4<Neon>; HEADS]) {
    let zero = f32x4::splat(neon, 0.0);
    let mut s_even = [zero; HEADS];
    let mut s_odd = [zero; HEADS];
    for pair_idx in (start_chunk..chunks8.len()).step_by(2) {
        let base = pair_idx * 8;
        let (v0, v1) = crate::kernels::neon::f16_bits_to_f32_x8(neon, &chunks8[pair_idx]);
        for head in 0..HEADS {
            let query = &queries[head][base..base + 8];
            s_even[head] = v0.mul_add_precise(f32x4::from_slice(neon, &query[..4]), s_even[head]);
            s_odd[head] = v1.mul_add_precise(f32x4::from_slice(neon, &query[4..]), s_odd[head]);
        }
    }
    (s_even, s_odd)
}

/// Computes two cached F16 keys × `HEADS` query-head dot products in two half-stride
/// passes (`[0..8]` then `[8..16]`), loading each query vector once across both keys
/// while holding <= 29 NEON registers and preserving 0-ULP parity.
pub(super) fn dot_two_keys_six_neon<const HEADS: usize>(
    neon: Neon,
    keys: [&[u16]; 2],
    queries: [&[f32]; HEADS],
) -> Result<[[f32; HEADS]; 2]> {
    let (k0_chunks8, _) = keys[0].as_chunks::<8>();
    let (k1_chunks8, _) = keys[1].as_chunks::<8>();
    let (s0, s1) = dot_two_keys_pass(neon, k0_chunks8, k1_chunks8, 0, queries);
    let (s2, s3) = dot_two_keys_pass(neon, k0_chunks8, k1_chunks8, 1, queries);
    let [s0_0, s0_1] = s0;
    let [s1_0, s1_1] = s1;
    let [s2_0, s2_1] = s2;
    let [s3_0, s3_1] = s3;
    let vector_steps = k0_chunks8.len() * 8;
    Ok([
        finish_dot_registers(neon, keys[0], queries, s0_0, s1_0, s2_0, s3_0, vector_steps)?,
        finish_dot_registers(neon, keys[1], queries, s0_1, s1_1, s2_1, s3_1, vector_steps)?,
    ])
}

type TwoKeysPassAccumulators<const HEADS: usize> =
    ([[f32x4<Neon>; HEADS]; 2], [[f32x4<Neon>; HEADS]; 2]);

#[inline]
fn dot_two_keys_pass<const HEADS: usize>(
    neon: Neon,
    k0_chunks8: &[[u16; 8]],
    k1_chunks8: &[[u16; 8]],
    start_chunk: usize,
    queries: [&[f32]; HEADS],
) -> TwoKeysPassAccumulators<HEADS> {
    let zero = f32x4::splat(neon, 0.0);
    let mut s_even = [[zero; HEADS]; 2];
    let mut s_odd = [[zero; HEADS]; 2];
    for pair_idx in (start_chunk..k0_chunks8.len()).step_by(2) {
        let base = pair_idx * 8;
        let (k0_v0, k0_v1) = crate::kernels::neon::f16_bits_to_f32_x8(neon, &k0_chunks8[pair_idx]);
        let (k1_v0, k1_v1) = crate::kernels::neon::f16_bits_to_f32_x8(neon, &k1_chunks8[pair_idx]);
        for head in 0..HEADS {
            let query = &queries[head][base..base + 8];
            let qv0 = f32x4::from_slice(neon, &query[..4]);
            s_even[0][head] = k0_v0.mul_add_precise(qv0, s_even[0][head]);
            s_even[1][head] = k1_v0.mul_add_precise(qv0, s_even[1][head]);
            let qv1 = f32x4::from_slice(neon, &query[4..]);
            s_odd[0][head] = k0_v1.mul_add_precise(qv1, s_odd[0][head]);
            s_odd[1][head] = k1_v1.mul_add_precise(qv1, s_odd[1][head]);
        }
    }
    (s_even, s_odd)
}

#[expect(
    clippy::too_many_arguments,
    reason = "Combines the 4 half-stride accumulator arrays and exact tail/padding rules without duplication."
)]
#[inline]
fn finish_dot_registers<const HEADS: usize>(
    neon: Neon,
    key: &[u16],
    queries: [&[f32]; HEADS],
    mut s0: [f32x4<Neon>; HEADS],
    mut s1: [f32x4<Neon>; HEADS],
    mut s2: [f32x4<Neon>; HEADS],
    mut s3: [f32x4<Neon>; HEADS],
    vector_steps: usize,
) -> Result<[f32; HEADS]> {
    let zero = f32x4::splat(neon, 0.0);
    let mut output = [0.0; HEADS];
    // Swift's complete 16-element key blocks need no scalar tail. Keep the
    // four native registers through padding and the original reduction tree,
    // rather than extracting sixteen lanes per head before adding them.
    if key.len().is_multiple_of(16) {
        let padded = padded_dot_steps(key.len()) != key.len();
        for head in 0..HEADS {
            if padded {
                // An aligned row's retained masked cycle visits every lane
                // once. Preserve the +0 FMA that normalizes a possible -0.
                s0[head] = zero.mul_add_precise(zero, s0[head]);
                s1[head] = zero.mul_add_precise(zero, s1[head]);
                s2[head] = zero.mul_add_precise(zero, s2[head]);
                s3[head] = zero.mul_add_precise(zero, s3[head]);
            }
            let lanes = ((s0[head] + s2[head]) + (s1[head] + s3[head])).to_array();
            output[head] = crate::kernels::finite((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))?;
        }
        return Ok(output);
    }
    for head in 0..HEADS {
        let mut scalar = [
            s0[head].to_array(),
            s1[head].to_array(),
            s2[head].to_array(),
            s3[head].to_array(),
        ];
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
        output[head] = crate::kernels::finite((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))?;
    }
    Ok(output)
}

const STACK_PAST_ROWS: usize = 256;

fn collect_past_rows<'a, 'b>(
    mut past: impl Iterator<Item = &'a [u16]>,
    past_len: usize,
    range: &std::ops::Range<usize>,
    stack: &'b mut [&'a [u16]; STACK_PAST_ROWS],
    heap: &'b mut Vec<&'a [u16]>,
) -> Result<&'b [&'a [u16]]> {
    let rows: &'b [&'a [u16]] = if past_len <= STACK_PAST_ROWS {
        for slot in &mut stack[..past_len] {
            *slot = past
                .next()
                .and_then(|row| row.get(range.start..range.end))
                .ok_or_else(|| invalid("pinned attention value row is short"))?;
        }
        &stack[..past_len]
    } else {
        heap.try_reserve_exact(past_len)?;
        for _ in 0..past_len {
            let row = past
                .next()
                .and_then(|row| row.get(range.start..range.end))
                .ok_or_else(|| invalid("pinned attention value row is short"))?;
            heap.push(row);
        }
        heap.as_slice()
    };
    if past.next().is_some() {
        return Err(invalid("pinned attention value row is long"));
    }
    Ok(rows)
}

#[inline]
fn pad_pair_sums<const HEADS: usize>(
    neon: Neon,
    sums: &mut [[[f32x4<Neon>; HEADS]; 2]; 16],
    valid_tokens: usize,
    padded_end: usize,
) {
    let zero = f32x4::splat(neon, 0.0);
    for token in valid_tokens..padded_end {
        for slot in &mut sums[token % 16] {
            for sum in slot {
                *sum = zero.mul_add_precise(zero, *sum);
            }
        }
    }
}

#[inline]
fn pad_tail_sums<const HEADS: usize>(
    neon: Neon,
    sums: &mut [[f32x4<Neon>; HEADS]; 16],
    valid_tokens: usize,
    padded_end: usize,
) {
    let zero = f32x4::splat(neon, 0.0);
    for token in valid_tokens..padded_end {
        for sum in &mut sums[token % 16] {
            *sum = zero.mul_add_precise(zero, *sum);
        }
    }
}

#[inline]
fn reduce_pair_sums<const HEADS: usize>(
    sums: &[[[f32x4<Neon>; HEADS]; 2]; 16],
    pair_idx: usize,
    staged: &mut [[f32x4<Neon>; super::VALUE_TILE / 4]; HEADS],
    finite: &mut fearless_simd::mask32x4<Neon>,
) {
    for slot in 0..2 {
        let group = pair_idx * 2 + slot;
        for head in 0..HEADS {
            let lanes: [f32x4<Neon>; 4] = std::array::from_fn(|lane| {
                (sums[lane][slot][head] + sums[8 + lane][slot][head])
                    + (sums[4 + lane][slot][head] + sums[12 + lane][slot][head])
            });
            let result = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
            *finite &= result.is_finite();
            staged[head][group] = result;
        }
    }
}

#[inline]
fn reduce_tail_sums<const HEADS: usize>(
    sums: &[[f32x4<Neon>; HEADS]; 16],
    group: usize,
    staged: &mut [[f32x4<Neon>; super::VALUE_TILE / 4]; HEADS],
    finite: &mut fearless_simd::mask32x4<Neon>,
) {
    for head in 0..HEADS {
        let lanes: [f32x4<Neon>; 4] = std::array::from_fn(|lane| {
            (sums[lane][head] + sums[8 + lane][head])
                + (sums[4 + lane][head] + sums[12 + lane][head])
        });
        let result = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
        *finite &= result.is_finite();
        staged[head][group] = result;
    }
}

fn gate_and_store<const HEADS: usize>(
    neon: Neon,
    staged: &[[f32x4<Neon>; super::VALUE_TILE / 4]; HEADS],
    finite: fearless_simd::mask32x4<Neon>,
    outputs: [&mut [f32]; HEADS],
    gates: [&[f32]; HEADS],
) -> Result<()> {
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
            (staged[head][group] * (one / (one + exponentials))).store_slice(output);
        }
    }
    Ok(())
}

pub(super) fn value_tile_neon<'a>(
    neon: Neon,
    past: impl Iterator<Item = &'a [u16]>,
    current: &[f32],
    weights: [&[f32]; 6],
    range: &std::ops::Range<usize>,
    outputs: [&mut [f32]; 6],
    gates: [&[f32]; 6],
) -> Result<()> {
    let zero = f32x4::splat(neon, 0.0);
    let groups = current.len() / 4;
    let num_pairs = groups / 2;
    let last = weights[0].len() - 1;
    let mut stack_rows = [&[][..]; STACK_PAST_ROWS];
    let mut heap_rows = Vec::new();
    let past_rows = collect_past_rows(past, last, range, &mut stack_rows, &mut heap_rows)?;
    let padded_end = padded_dot_steps(last + 1);
    let mut staged = [[zero; super::VALUE_TILE / 4]; 6];
    let mut finite = zero.is_finite();
    let last_lane = last % 16;
    let w_last: [f32x4<Neon>; 6] =
        std::array::from_fn(|head| f32x4::splat(neon, weights[head][last]));

    for pair_idx in 0..num_pairs {
        let mut sums = [[[zero; 6]; 2]; 16];
        let ch_base = pair_idx * 8;
        for (token, &row) in past_rows.iter().enumerate() {
            let lane = token % 16;
            let (first, second) =
                crate::kernels::neon::f16_bits_to_f32_x8(neon, &row.as_chunks::<8>().0[pair_idx]);
            for head in 0..6 {
                let w = f32x4::splat(neon, weights[head][token]);
                sums[lane][0][head] = first.mul_add_precise(w, sums[lane][0][head]);
                sums[lane][1][head] = second.mul_add_precise(w, sums[lane][1][head]);
            }
        }
        let c0 = f32x4::from_slice(neon, &current[ch_base..ch_base + 4]);
        let c1 = f32x4::from_slice(neon, &current[ch_base + 4..ch_base + 8]);
        for head in 0..6 {
            sums[last_lane][0][head] = c0.mul_add_precise(w_last[head], sums[last_lane][0][head]);
            sums[last_lane][1][head] = c1.mul_add_precise(w_last[head], sums[last_lane][1][head]);
        }
        pad_pair_sums(neon, &mut sums, last + 1, padded_end);
        reduce_pair_sums(&sums, pair_idx, &mut staged, &mut finite);
    }

    if !groups.is_multiple_of(2) {
        let g = groups - 1;
        let ch_base = g * 4;
        let mut sums = [[zero; 6]; 16];
        for (token, &row) in past_rows.iter().enumerate() {
            let lane = token % 16;
            let value = crate::kernels::neon::f16_bits_to_f32_x4(
                neon,
                [
                    row[ch_base],
                    row[ch_base + 1],
                    row[ch_base + 2],
                    row[ch_base + 3],
                ],
            );
            for head in 0..6 {
                let w = f32x4::splat(neon, weights[head][token]);
                sums[lane][head] = value.mul_add_precise(w, sums[lane][head]);
            }
        }
        let c = f32x4::from_slice(neon, &current[ch_base..ch_base + 4]);
        for head in 0..6 {
            sums[last_lane][head] = c.mul_add_precise(w_last[head], sums[last_lane][head]);
        }
        pad_tail_sums(neon, &mut sums, last + 1, padded_end);
        reduce_tail_sums(&sums, g, &mut staged, &mut finite);
    }

    gate_and_store(neon, &staged, finite, outputs, gates)
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

#[expect(
    clippy::too_many_arguments,
    reason = "Keep channel-pair accumulation separated from tail handling cleanly."
)]
fn accumulate_packed_pair<const HEADS: usize>(
    neon: Neon,
    pair_idx: usize,
    weight_pairs: &[[[u16; HEADS]; 2]],
    weight_tail: &[[u16; HEADS]],
    row_pairs: &[[&[u16]; 2]],
    row_tail: &[&[u16]],
    current: &[f32],
    w_last: [f32x4<Neon>; HEADS],
    past_len: usize,
    padded_end: usize,
    staged: &mut [[f32x4<Neon>; super::VALUE_TILE / 4]; HEADS],
    finite: &mut fearless_simd::mask32x4<Neon>,
) {
    let zero = f32x4::splat(neon, 0.0);
    let mut sums = [[[zero; HEADS]; 2]; 16];
    let ch_base = pair_idx * 8;
    let mut token = 0;
    for (&[w0_bits, w1_bits], &[row0, row1]) in weight_pairs.iter().zip(row_pairs) {
        let (w0, w1) = decode_pair_group_weights(neon, w0_bits, w1_bits);
        let w0_splat = w0.map(|weight| f32x4::splat(neon, weight));
        let w1_splat = w1.map(|weight| f32x4::splat(neon, weight));
        let lane0 = token % 16;
        let lane1 = (token + 1) % 16;
        token += 2;
        let (v0_0, v0_1) =
            crate::kernels::neon::f16_bits_to_f32_x8(neon, &row0.as_chunks::<8>().0[pair_idx]);
        let (v1_0, v1_1) =
            crate::kernels::neon::f16_bits_to_f32_x8(neon, &row1.as_chunks::<8>().0[pair_idx]);
        for h in 0..HEADS {
            sums[lane0][0][h] = v0_0.mul_add_precise(w0_splat[h], sums[lane0][0][h]);
            sums[lane0][1][h] = v0_1.mul_add_precise(w0_splat[h], sums[lane0][1][h]);
            sums[lane1][0][h] = v1_0.mul_add_precise(w1_splat[h], sums[lane1][0][h]);
            sums[lane1][1][h] = v1_1.mul_add_precise(w1_splat[h], sums[lane1][1][h]);
        }
    }
    if let (Some(&tail_bits), Some(&row)) = (weight_tail.first(), row_tail.first()) {
        let w_tail = decode_single_group_weights(neon, tail_bits).map(|w| f32x4::splat(neon, w));
        let lane = token % 16;
        let (v0, v1) =
            crate::kernels::neon::f16_bits_to_f32_x8(neon, &row.as_chunks::<8>().0[pair_idx]);
        for h in 0..HEADS {
            sums[lane][0][h] = v0.mul_add_precise(w_tail[h], sums[lane][0][h]);
            sums[lane][1][h] = v1.mul_add_precise(w_tail[h], sums[lane][1][h]);
        }
    }
    let last_lane = past_len % 16;
    let c0 = f32x4::from_slice(neon, &current[ch_base..ch_base + 4]);
    let c1 = f32x4::from_slice(neon, &current[ch_base + 4..ch_base + 8]);
    for h in 0..HEADS {
        sums[last_lane][0][h] = c0.mul_add_precise(w_last[h], sums[last_lane][0][h]);
        sums[last_lane][1][h] = c1.mul_add_precise(w_last[h], sums[last_lane][1][h]);
    }
    pad_pair_sums(neon, &mut sums, past_len + 1, padded_end);
    reduce_pair_sums(&sums, pair_idx, staged, finite);
}

#[expect(
    clippy::too_many_arguments,
    reason = "Keep odd 4-channel tail accumulation separated cleanly."
)]
fn accumulate_packed_tail<const HEADS: usize>(
    neon: Neon,
    group: usize,
    weight_pairs: &[[[u16; HEADS]; 2]],
    weight_tail: &[[u16; HEADS]],
    row_pairs: &[[&[u16]; 2]],
    row_tail: &[&[u16]],
    current: &[f32],
    w_last: [f32x4<Neon>; HEADS],
    past_len: usize,
    padded_end: usize,
    staged: &mut [[f32x4<Neon>; super::VALUE_TILE / 4]; HEADS],
    finite: &mut fearless_simd::mask32x4<Neon>,
) {
    let zero = f32x4::splat(neon, 0.0);
    let ch_base = group * 4;
    let mut sums = [[zero; HEADS]; 16];
    let mut token = 0;
    for (&[w0_bits, w1_bits], &[row0, row1]) in weight_pairs.iter().zip(row_pairs) {
        let (w0, w1) = decode_pair_group_weights(neon, w0_bits, w1_bits);
        let w0_splat = w0.map(|weight| f32x4::splat(neon, weight));
        let w1_splat = w1.map(|weight| f32x4::splat(neon, weight));
        let lane0 = token % 16;
        let lane1 = (token + 1) % 16;
        token += 2;
        let v0 = crate::kernels::neon::f16_bits_to_f32_x4(neon, row0.as_chunks::<4>().0[group]);
        let v1 = crate::kernels::neon::f16_bits_to_f32_x4(neon, row1.as_chunks::<4>().0[group]);
        for h in 0..HEADS {
            sums[lane0][h] = v0.mul_add_precise(w0_splat[h], sums[lane0][h]);
            sums[lane1][h] = v1.mul_add_precise(w1_splat[h], sums[lane1][h]);
        }
    }
    if let (Some(&tail_bits), Some(&row)) = (weight_tail.first(), row_tail.first()) {
        let w_tail = decode_single_group_weights(neon, tail_bits).map(|w| f32x4::splat(neon, w));
        let lane = token % 16;
        let v = crate::kernels::neon::f16_bits_to_f32_x4(neon, row.as_chunks::<4>().0[group]);
        for h in 0..HEADS {
            sums[lane][h] = v.mul_add_precise(w_tail[h], sums[lane][h]);
        }
    }
    let last_lane = past_len % 16;
    let c = f32x4::from_slice(neon, &current[ch_base..ch_base + 4]);
    for h in 0..HEADS {
        sums[last_lane][h] = c.mul_add_precise(w_last[h], sums[last_lane][h]);
    }
    pad_tail_sums(neon, &mut sums, past_len + 1, padded_end);
    reduce_tail_sums(&sums, group, staged, finite);
}

pub(super) fn value_tile_group_packed_neon<'a, const HEADS: usize>(
    neon: Neon,
    past: impl Iterator<Item = &'a [u16]>,
    current: &[f32],
    weights: &[[u16; HEADS]],
    range: &std::ops::Range<usize>,
    outputs: [&mut [f32]; HEADS],
    gates: [&[f32]; HEADS],
) -> Result<()> {
    let zero = f32x4::splat(neon, 0.0);
    let groups = outputs[0].len() / 4;
    let num_pairs = groups / 2;
    let past_len = weights.len() - 1;
    let mut stack_rows = [&[][..]; STACK_PAST_ROWS];
    let mut heap_rows = Vec::new();
    let past_rows = collect_past_rows(past, past_len, range, &mut stack_rows, &mut heap_rows)?;
    let padded_end = padded_dot_steps(past_len + 1);
    let mut staged = [[zero; super::VALUE_TILE / 4]; HEADS];
    let mut finite = zero.is_finite();
    let (weight_pairs, weight_tail) = weights[..past_len].as_chunks::<2>();
    let (row_pairs, row_tail) = past_rows.as_chunks::<2>();
    let w_last =
        decode_single_group_weights(neon, weights[past_len]).map(|w| f32x4::splat(neon, w));
    for pair_idx in 0..num_pairs {
        accumulate_packed_pair(
            neon,
            pair_idx,
            weight_pairs,
            weight_tail,
            row_pairs,
            row_tail,
            current,
            w_last,
            past_len,
            padded_end,
            &mut staged,
            &mut finite,
        );
    }
    if !groups.is_multiple_of(2) {
        accumulate_packed_tail(
            neon,
            groups - 1,
            weight_pairs,
            weight_tail,
            row_pairs,
            row_tail,
            current,
            w_last,
            past_len,
            padded_end,
            &mut staged,
            &mut finite,
        );
    }
    gate_and_store(neon, &staged, finite, outputs, gates)
}
