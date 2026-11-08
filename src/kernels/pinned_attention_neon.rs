//! Exact baseline NEON value tiles with adjacent channels in vector lanes.

use fearless_simd::{Neon, f32x4, prelude::*};

use crate::{Result, invalid};

use super::{VALUE_TILE, padded_dot_steps};

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
    clippy::needless_range_loop,
    reason = "Keep explicit 4-head and token indices aligned across fixed arrays."
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
    let mut sums = [[[zero; VALUE_TILE / 4]; 16]; 4];
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

/// Reduces a validated F16 key slice and F32 query slice with four NEON FMA registers.
pub(super) fn dot_f16(neon: Neon, key: &[u16], query: &[f32]) -> Result<f32> {
    let mut sums = [f32x4::splat(neon, 0.0); 4];
    let (key_chunks8, key_rem) = key.as_chunks::<8>();
    let (query_chunks8, query_rem) = query.as_chunks::<8>();
    for (pair_idx, (k8, q8)) in key_chunks8.iter().zip(query_chunks8).enumerate() {
        let base = (pair_idx * 2) % 4;
        let (kv0, kv1) = super::super::neon::f16_bits_to_f32_x8(neon, k8);
        let qv0 = f32x4::from_slice(neon, &q8[..4]);
        let qv1 = f32x4::from_slice(neon, &q8[4..]);
        sums[base] = kv0.mul_add_precise(qv0, sums[base]);
        sums[base + 1] = kv1.mul_add_precise(qv1, sums[base + 1]);
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
    for (chunk, (k, q)) in key_chunks.iter().zip(query_chunks).enumerate() {
        let acc = &mut sums[chunk % 4];
        let kv = f32x4::load_array(neon, *k);
        let qv = f32x4::load_array(neon, *q);
        *acc = kv.mul_add_precise(qv, *acc);
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
    for (pair_idx, k8) in key_chunks8.iter().enumerate() {
        let base = (pair_idx * 2) % 4;
        let (kv0, kv1) = super::super::neon::f16_bits_to_f32_x8(neon, k8);
        for h in 0..4 {
            let q8 = &q_chunks8[h][pair_idx];
            let qv0 = f32x4::from_slice(neon, &q8[..4]);
            let qv1 = f32x4::from_slice(neon, &q8[4..]);
            sums[h][base] = kv0.mul_add_precise(qv0, sums[h][base]);
            sums[h][base + 1] = kv1.mul_add_precise(qv1, sums[h][base + 1]);
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

/// Reduces four validated F16 key slices against four F32 query slices sharing the same KV head.
pub(super) fn dot_f16_four_keys_four_queries(
    neon: Neon,
    keys: [&[u16]; 4],
    queries: [&[f32]; 4],
) -> Result<[[f32; 4]; 4]> {
    let [r0, r1] = dot_f16_two_keys_four_queries(neon, [keys[0], keys[1]], queries)?;
    let [r2, r3] = dot_f16_two_keys_four_queries(neon, [keys[2], keys[3]], queries)?;
    Ok([r0, r1, r2, r3])
}

/// Reduces one validated F32 key slice against four F32 query slices sharing the same KV head.
pub(super) fn dot_f32_four(neon: Neon, key: &[f32], queries: [&[f32]; 4]) -> Result<[f32; 4]> {
    let mut sums = [[f32x4::splat(neon, 0.0); 4]; 4];
    let key_chunks = key.as_chunks::<4>().0;
    let q_chunks = queries.map(|q| q.as_chunks::<4>().0);
    for (chunk, k) in key_chunks.iter().enumerate() {
        let lane = chunk % 4;
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

    use super::{VALUE_TILE, value_tile};

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
}
