//! Safe baseline NEON for the exact pinned GDN tree and checked row update.
//!
//! Multi-token rows use one private staging array for both decayed and updated
//! state. Decay consumes the old block before replacing it; the second pass
//! consumes that decayed block before replacing it with its checked update.
//! Caller-owned rows remain untouched until the helper's existing publication
//! point, so grouped fallback and single-row partial-error semantics are retained.
//! The graph-only provisional decode helper instead uses unpublished delta rows
//! directly; it never receives live state and rejects the entire graph delta on
//! failure. Its defensive replay always reads the immutable committed source.
//! Grouped rows combine every updated-lane finite predicate in a vector mask;
//! its only horizontal reduction occurs before publication. Dot-result checks
//! retain their original order. Single-row publication remains unchanged.

use fearless_simd::{Neon, f32x4, prelude::*};

use crate::Result;

#[cfg(test)]
#[path = "pinned_gdn_neon_tests.rs"]
mod tests;

/// Geometry is checked by the parent before any load. The fixed four-register
/// tree is the scalar reference's tree, including adjacent horizontal pairs.
#[inline]
#[expect(
    clippy::suboptimal_flops,
    reason = "Pinned scalar remainder retains separate multiply/add rounding."
)]
pub(super) fn dot(first: &[f32], second: &[f32]) -> Option<f32> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let vectorized = first.len() / 16 * 16;
    let mut s0 = f32x4::splat(neon, 0.0);
    let mut s1 = f32x4::splat(neon, 0.0);
    let mut s2 = f32x4::splat(neon, 0.0);
    let mut s3 = f32x4::splat(neon, 0.0);
    for (first, second) in first[..vectorized]
        .as_chunks::<16>()
        .0
        .iter()
        .zip(second[..vectorized].as_chunks::<16>().0)
    {
        s0 = f32x4::from_slice(neon, &first[0..4])
            .mul_add_precise(f32x4::from_slice(neon, &second[0..4]), s0);
        s1 = f32x4::from_slice(neon, &first[4..8])
            .mul_add_precise(f32x4::from_slice(neon, &second[4..8]), s1);
        s2 = f32x4::from_slice(neon, &first[8..12])
            .mul_add_precise(f32x4::from_slice(neon, &second[8..12]), s2);
        s3 = f32x4::from_slice(neon, &first[12..16])
            .mul_add_precise(f32x4::from_slice(neon, &second[12..16]), s3);
    }
    let mut result = ((s0 + s2) + (s1 + s3)).reduce_sum();
    for (&first, &second) in first[vectorized..].iter().zip(&second[vectorized..]) {
        result += first * second;
    }
    Some(result)
}

/// Computes `(dot(state, first), dot(state, second))` in a single pass over `state`,
/// reusing each `state` vector load across both four-register FMA reduction trees.
#[inline]
#[expect(
    clippy::suboptimal_flops,
    reason = "Pinned scalar remainder retains separate multiply/add rounding."
)]
pub(super) fn dot_pair(state: &[f32], first: &[f32], second: &[f32]) -> Option<(f32, f32)> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let vectorized = state.len() / 16 * 16;
    let mut f0 = f32x4::splat(neon, 0.0);
    let mut f1 = f32x4::splat(neon, 0.0);
    let mut f2 = f32x4::splat(neon, 0.0);
    let mut f3 = f32x4::splat(neon, 0.0);
    let mut s0 = f32x4::splat(neon, 0.0);
    let mut s1 = f32x4::splat(neon, 0.0);
    let mut s2 = f32x4::splat(neon, 0.0);
    let mut s3 = f32x4::splat(neon, 0.0);
    for ((state, first), second) in state[..vectorized]
        .as_chunks::<16>()
        .0
        .iter()
        .zip(first[..vectorized].as_chunks::<16>().0)
        .zip(second[..vectorized].as_chunks::<16>().0)
    {
        let st0 = f32x4::from_slice(neon, &state[0..4]);
        let st1 = f32x4::from_slice(neon, &state[4..8]);
        let st2 = f32x4::from_slice(neon, &state[8..12]);
        let st3 = f32x4::from_slice(neon, &state[12..16]);
        f0 = st0.mul_add_precise(f32x4::from_slice(neon, &first[0..4]), f0);
        f1 = st1.mul_add_precise(f32x4::from_slice(neon, &first[4..8]), f1);
        f2 = st2.mul_add_precise(f32x4::from_slice(neon, &first[8..12]), f2);
        f3 = st3.mul_add_precise(f32x4::from_slice(neon, &first[12..16]), f3);
        s0 = st0.mul_add_precise(f32x4::from_slice(neon, &second[0..4]), s0);
        s1 = st1.mul_add_precise(f32x4::from_slice(neon, &second[4..8]), s1);
        s2 = st2.mul_add_precise(f32x4::from_slice(neon, &second[8..12]), s2);
        s3 = st3.mul_add_precise(f32x4::from_slice(neon, &second[12..16]), s3);
    }
    let mut first_result = ((f0 + f2) + (f1 + f3)).reduce_sum();
    let mut second_result = ((s0 + s2) + (s1 + s3)).reduce_sum();
    for ((&state, &first), &second) in state[vectorized..]
        .iter()
        .zip(&first[vectorized..])
        .zip(&second[vectorized..])
    {
        first_result += state * first;
        second_result += state * second;
    }
    Some((first_result, second_result))
}

/// Computes `[dot(state, inputs[0]), dot(state, inputs[1]), dot(state, inputs[2]), dot(state, inputs[3])]`
/// in a single pass over `state`, reusing each `state` vector load across all four four-register FMA trees.
#[cfg(test)]
#[inline]
#[expect(
    clippy::suboptimal_flops,
    reason = "Pinned scalar remainder retains separate multiply/add rounding."
)]
pub(super) fn dot_four(state: &[f32], inputs: [&[f32]; 4]) -> Option<[f32; 4]> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let vectorized = state.len() / 16 * 16;
    let mut acc = [[f32x4::splat(neon, 0.0); 4]; 4];
    let in_chunks = inputs.map(|row| row[..vectorized].as_chunks::<16>().0);
    for (b, st_block) in state[..vectorized].as_chunks::<16>().0.iter().enumerate() {
        let st0 = f32x4::from_slice(neon, &st_block[0..4]);
        let st1 = f32x4::from_slice(neon, &st_block[4..8]);
        let st2 = f32x4::from_slice(neon, &st_block[8..12]);
        let st3 = f32x4::from_slice(neon, &st_block[12..16]);
        for t in 0..4 {
            let in_block = &in_chunks[t][b];
            acc[t][0] = st0.mul_add_precise(f32x4::from_slice(neon, &in_block[0..4]), acc[t][0]);
            acc[t][1] = st1.mul_add_precise(f32x4::from_slice(neon, &in_block[4..8]), acc[t][1]);
            acc[t][2] = st2.mul_add_precise(f32x4::from_slice(neon, &in_block[8..12]), acc[t][2]);
            acc[t][3] = st3.mul_add_precise(f32x4::from_slice(neon, &in_block[12..16]), acc[t][3]);
        }
    }
    let mut results: [f32; 4] =
        std::array::from_fn(|t| ((acc[t][0] + acc[t][2]) + (acc[t][1] + acc[t][3])).reduce_sum());
    for (idx, &st) in state[vectorized..].iter().enumerate() {
        let col = vectorized + idx;
        for (t, res) in results.iter_mut().enumerate() {
            *res += st * inputs[t][col];
        }
    }
    Some(results)
}

/// Computes `(dot(states[r], first), dot(states[r], second))` across two state rows in one pass,
/// sharing each `first` and `second` vector load across both rows' four-register FMA trees.
#[inline]
#[expect(
    clippy::suboptimal_flops,
    reason = "Pinned scalar remainder retains separate multiply/add rounding."
)]
fn dot_pair_two_rows(
    neon: Neon,
    states: [&[f32]; 2],
    first: &[f32],
    second: &[f32],
) -> [(f32, f32); 2] {
    let len = first.len().min(second.len());
    let vectorized = len / 16 * 16;
    let mut f = [[f32x4::splat(neon, 0.0); 4]; 2];
    let mut s = [[f32x4::splat(neon, 0.0); 4]; 2];
    let st0 = &states[0][..len];
    let st1 = &states[1][..len];
    for (((first_block, second_block), st0_block), st1_block) in first[..vectorized]
        .as_chunks::<16>()
        .0
        .iter()
        .zip(second[..vectorized].as_chunks::<16>().0)
        .zip(st0[..vectorized].as_chunks::<16>().0)
        .zip(st1[..vectorized].as_chunks::<16>().0)
    {
        let k0 = f32x4::from_slice(neon, &first_block[0..4]);
        let k1 = f32x4::from_slice(neon, &first_block[4..8]);
        let k2 = f32x4::from_slice(neon, &first_block[8..12]);
        let k3 = f32x4::from_slice(neon, &first_block[12..16]);
        let q0 = f32x4::from_slice(neon, &second_block[0..4]);
        let q1 = f32x4::from_slice(neon, &second_block[4..8]);
        let q2 = f32x4::from_slice(neon, &second_block[8..12]);
        let q3 = f32x4::from_slice(neon, &second_block[12..16]);
        for (r, st_block) in [st0_block, st1_block].into_iter().enumerate() {
            let v0 = f32x4::from_slice(neon, &st_block[0..4]);
            let v1 = f32x4::from_slice(neon, &st_block[4..8]);
            let v2 = f32x4::from_slice(neon, &st_block[8..12]);
            let v3 = f32x4::from_slice(neon, &st_block[12..16]);
            f[r][0] = v0.mul_add_precise(k0, f[r][0]);
            f[r][1] = v1.mul_add_precise(k1, f[r][1]);
            f[r][2] = v2.mul_add_precise(k2, f[r][2]);
            f[r][3] = v3.mul_add_precise(k3, f[r][3]);
            s[r][0] = v0.mul_add_precise(q0, s[r][0]);
            s[r][1] = v1.mul_add_precise(q1, s[r][1]);
            s[r][2] = v2.mul_add_precise(q2, s[r][2]);
            s[r][3] = v3.mul_add_precise(q3, s[r][3]);
        }
    }
    let mut results: [(f32, f32); 2] = std::array::from_fn(|r| {
        (
            ((f[r][0] + f[r][2]) + (f[r][1] + f[r][3])).reduce_sum(),
            ((s[r][0] + s[r][2]) + (s[r][1] + s[r][3])).reduce_sum(),
        )
    });
    for (((&k, &q), &a), &b) in first[vectorized..len]
        .iter()
        .zip(&second[vectorized..len])
        .zip(&st0[vectorized..])
        .zip(&st1[vectorized..])
    {
        results[0].0 += a * k;
        results[0].1 += a * q;
        results[1].0 += b * k;
        results[1].1 += b * q;
    }
    results
}

#[inline]
#[expect(
    clippy::suboptimal_flops,
    reason = "Pinned scalar remainder retains separate multiply/add rounding."
)]
fn dot_pair_two_rows_four_tokens(
    neon: Neon,
    st0: &[f32],
    st1: &[f32],
    keys: [&[f32]; 4],
    queries: [&[f32]; 4],
) -> ([[f32; 4]; 2], [[f32; 4]; 2]) {
    let len = keys[0].len();
    let vectorized = len / 16 * 16;
    let zero = f32x4::splat(neon, 0.0);
    let mut fk = [[[zero; 4]; 4]; 2];
    let mut sq = [[[zero; 4]; 4]; 2];
    let st0_chunks = st0[..vectorized].as_chunks::<16>().0;
    let st1_chunks = st1[..vectorized].as_chunks::<16>().0;
    let k_chunks = keys.map(|row| row[..vectorized].as_chunks::<16>().0);
    let q_chunks = queries.map(|row| row[..vectorized].as_chunks::<16>().0);
    for (b, (st0_block, st1_block)) in st0_chunks.iter().zip(st1_chunks).enumerate() {
        for half in 0..2 {
            let off0 = half * 8;
            let off1 = off0 + 4;
            let l0 = half * 2;
            let l1 = l0 + 1;
            let v0_0 = f32x4::from_slice(neon, &st0_block[off0..off1]);
            let v0_1 = f32x4::from_slice(neon, &st0_block[off1..off1 + 4]);
            let v1_0 = f32x4::from_slice(neon, &st1_block[off0..off1]);
            let v1_1 = f32x4::from_slice(neon, &st1_block[off1..off1 + 4]);
            for t in 0..4 {
                let kb = &k_chunks[t][b];
                let k0 = f32x4::from_slice(neon, &kb[off0..off1]);
                let k1 = f32x4::from_slice(neon, &kb[off1..off1 + 4]);
                fk[0][t][l0] = v0_0.mul_add_precise(k0, fk[0][t][l0]);
                fk[0][t][l1] = v0_1.mul_add_precise(k1, fk[0][t][l1]);
                fk[1][t][l0] = v1_0.mul_add_precise(k0, fk[1][t][l0]);
                fk[1][t][l1] = v1_1.mul_add_precise(k1, fk[1][t][l1]);
            }
            for t in 0..4 {
                let qb = &q_chunks[t][b];
                let q0 = f32x4::from_slice(neon, &qb[off0..off1]);
                let q1 = f32x4::from_slice(neon, &qb[off1..off1 + 4]);
                sq[0][t][l0] = v0_0.mul_add_precise(q0, sq[0][t][l0]);
                sq[0][t][l1] = v0_1.mul_add_precise(q1, sq[0][t][l1]);
                sq[1][t][l0] = v1_0.mul_add_precise(q0, sq[1][t][l0]);
                sq[1][t][l1] = v1_1.mul_add_precise(q1, sq[1][t][l1]);
            }
        }
    }
    let mut out_k: [[f32; 4]; 2] = std::array::from_fn(|r| {
        std::array::from_fn(|t| {
            ((fk[r][t][0] + fk[r][t][2]) + (fk[r][t][1] + fk[r][t][3])).reduce_sum()
        })
    });
    let mut out_q: [[f32; 4]; 2] = std::array::from_fn(|r| {
        std::array::from_fn(|t| {
            ((sq[r][t][0] + sq[r][t][2]) + (sq[r][t][1] + sq[r][t][3])).reduce_sum()
        })
    });
    for idx in vectorized..len {
        let a = st0[idx];
        let b = st1[idx];
        for t in 0..4 {
            let k = keys[t][idx];
            let q = queries[t][idx];
            out_k[0][t] += a * k;
            out_q[0][t] += a * q;
            out_k[1][t] += b * k;
            out_q[1][t] += b * q;
        }
    }
    (out_k, out_q)
}

/// Computes `(dot(states[r], key_t), dot(states[r], query_t))` across eight state rows for up
/// to 32 tokens, batching two state rows by four tokens in a single contiguous vector pass
/// (0-ULP bit-exact with `dot_pair_eight_rows`).
#[inline]
pub(super) fn dot_pair_eight_rows_tokens(
    states: [&[f32]; 8],
    tokens: &[(&[f32], &[f32])],
    key_dots: &mut [[f32; 32]; 8],
    query_dots: &mut [[f32; 32]; 8],
) -> bool {
    dot_pair_eight_rows_tokens_with(states, tokens, |token, row, k_val, q_val| {
        key_dots[row][token] = k_val;
        query_dots[row][token] = q_val;
    })
}

/// Token-major variant of `dot_pair_eight_rows_tokens` writing `[token][row]` slices directly.
#[inline]
pub(super) fn dot_pair_eight_rows_tokens_major(
    states: [&[f32]; 8],
    tokens: &[(&[f32], &[f32])],
    key_dots: &mut [[f32; 8]; 32],
    query_dots: &mut [[f32; 8]; 32],
) -> bool {
    dot_pair_eight_rows_tokens_with(states, tokens, |token, row, k_val, q_val| {
        key_dots[token][row] = k_val;
        query_dots[token][row] = q_val;
    })
}

#[inline]
fn dot_pair_eight_rows_tokens_with(
    states: [&[f32]; 8],
    tokens: &[(&[f32], &[f32])],
    mut store: impl FnMut(usize, usize, f32, f32),
) -> bool {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return false;
    };
    let (chunks4, remainder) = tokens.as_chunks::<4>();
    for (chunk_idx, chunk) in chunks4.iter().enumerate() {
        let base_t = chunk_idx * 4;
        let keys = [chunk[0].0, chunk[1].0, chunk[2].0, chunk[3].0];
        let queries = [chunk[0].1, chunk[1].1, chunk[2].1, chunk[3].1];
        for pair in 0..4 {
            let r0 = pair * 2;
            let (out_k, out_q) =
                dot_pair_two_rows_four_tokens(neon, states[r0], states[r0 + 1], keys, queries);
            for t in 0..4 {
                store(base_t + t, r0, out_k[0][t], out_q[0][t]);
                store(base_t + t, r0 + 1, out_k[1][t], out_q[1][t]);
            }
        }
    }
    let rem_base = chunks4.len() * 4;
    for (offset, &(first, second)) in remainder.iter().enumerate() {
        let token = rem_base + offset;
        let Some(pairs) = dot_pair_eight_rows(states, first, second) else {
            return false;
        };
        for (row, &(k_val, q_val)) in pairs.iter().enumerate() {
            store(token, row, k_val, q_val);
        }
    }
    true
}

/// Computes `(dot(states[r], first), dot(states[r], second))` across four state rows in two
/// spill-free two-row passes, sharing `first` and `second` vector loads per pair.
#[inline]
pub(super) fn dot_pair_four_rows(
    states: [&[f32]; 4],
    first: &[f32],
    second: &[f32],
) -> Option<[(f32, f32); 4]> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let [p0, p1] = dot_pair_two_rows(neon, [states[0], states[1]], first, second);
    let [p2, p3] = dot_pair_two_rows(neon, [states[2], states[3]], first, second);
    Some([p0, p1, p2, p3])
}

/// Computes `(dot(states[r], first), dot(states[r], second))` across eight state rows in four
/// spill-free two-row passes, sharing `first` and `second` vector loads per pair.
#[inline]
pub(super) fn dot_pair_eight_rows(
    states: [&[f32]; 8],
    first: &[f32],
    second: &[f32],
) -> Option<[(f32, f32); 8]> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let [p0, p1] = dot_pair_two_rows(neon, [states[0], states[1]], first, second);
    let [p2, p3] = dot_pair_two_rows(neon, [states[2], states[3]], first, second);
    let [p4, p5] = dot_pair_two_rows(neon, [states[4], states[5]], first, second);
    let [p6, p7] = dot_pair_two_rows(neon, [states[6], states[7]], first, second);
    Some([p0, p1, p2, p3, p4, p5, p6, p7])
}

/// Scales `state` by `decay` and accumulates its four-register FMA dot product with `key` in one pass.
#[inline]
#[expect(
    clippy::suboptimal_flops,
    reason = "Pinned scalar remainder retains separate multiply/add rounding."
)]
pub(super) fn decay_and_dot(state: &mut [f32], key: &[f32], decay: f32) -> Option<f32> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let vectorized = state.len() / 16 * 16;
    let decays = f32x4::splat(neon, decay);
    let mut s0 = f32x4::splat(neon, 0.0);
    let mut s1 = f32x4::splat(neon, 0.0);
    let mut s2 = f32x4::splat(neon, 0.0);
    let mut s3 = f32x4::splat(neon, 0.0);
    for (state, key) in state[..vectorized]
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(key[..vectorized].as_chunks::<16>().0)
    {
        let d0 = f32x4::from_slice(neon, &state[0..4]) * decays;
        let d1 = f32x4::from_slice(neon, &state[4..8]) * decays;
        let d2 = f32x4::from_slice(neon, &state[8..12]) * decays;
        let d3 = f32x4::from_slice(neon, &state[12..16]) * decays;
        d0.store_slice(&mut state[0..4]);
        d1.store_slice(&mut state[4..8]);
        d2.store_slice(&mut state[8..12]);
        d3.store_slice(&mut state[12..16]);
        s0 = d0.mul_add_precise(f32x4::from_slice(neon, &key[0..4]), s0);
        s1 = d1.mul_add_precise(f32x4::from_slice(neon, &key[4..8]), s1);
        s2 = d2.mul_add_precise(f32x4::from_slice(neon, &key[8..12]), s2);
        s3 = d3.mul_add_precise(f32x4::from_slice(neon, &key[12..16]), s3);
    }
    let mut result = ((s0 + s2) + (s1 + s3)).reduce_sum();
    for (state, &key) in state[vectorized..].iter_mut().zip(&key[vectorized..]) {
        *state *= decay;
        result += *state * key;
    }
    Some(result)
}

/// Scales `previous` by `decay` into `state` and accumulates its four-register FMA dot product with `key` in one pass.
#[inline]
#[expect(
    clippy::suboptimal_flops,
    reason = "Pinned scalar remainder retains separate multiply/add rounding."
)]
pub(super) fn decay_and_dot_from(
    previous: &[f32],
    state: &mut [f32],
    key: &[f32],
    decay: f32,
) -> Option<f32> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let vectorized = state.len() / 16 * 16;
    let decays = f32x4::splat(neon, decay);
    let mut s0 = f32x4::splat(neon, 0.0);
    let mut s1 = f32x4::splat(neon, 0.0);
    let mut s2 = f32x4::splat(neon, 0.0);
    let mut s3 = f32x4::splat(neon, 0.0);
    for ((prev, state), key) in previous[..vectorized]
        .as_chunks::<16>()
        .0
        .iter()
        .zip(state[..vectorized].as_chunks_mut::<16>().0)
        .zip(key[..vectorized].as_chunks::<16>().0)
    {
        let d0 = f32x4::from_slice(neon, &prev[0..4]) * decays;
        let d1 = f32x4::from_slice(neon, &prev[4..8]) * decays;
        let d2 = f32x4::from_slice(neon, &prev[8..12]) * decays;
        let d3 = f32x4::from_slice(neon, &prev[12..16]) * decays;
        d0.store_slice(&mut state[0..4]);
        d1.store_slice(&mut state[4..8]);
        d2.store_slice(&mut state[8..12]);
        d3.store_slice(&mut state[12..16]);
        s0 = d0.mul_add_precise(f32x4::from_slice(neon, &key[0..4]), s0);
        s1 = d1.mul_add_precise(f32x4::from_slice(neon, &key[4..8]), s1);
        s2 = d2.mul_add_precise(f32x4::from_slice(neon, &key[8..12]), s2);
        s3 = d3.mul_add_precise(f32x4::from_slice(neon, &key[12..16]), s3);
    }
    let mut result = ((s0 + s2) + (s1 + s3)).reduce_sum();
    for ((&prev, state), &key) in previous[vectorized..]
        .iter()
        .zip(&mut state[vectorized..])
        .zip(&key[vectorized..])
    {
        let decayed = prev * decay;
        *state = decayed;
        result += decayed * key;
    }
    Some(result)
}

/// The parent checks lengths; every safe load/store receives exactly four lanes.
/// A nonfinite vector is replayed in scalar lane order before any vector store,
/// preserving the original finite-prefix mutation and untouched failing lane.
#[inline]
pub(super) fn add_scaled(target: &mut [f32], input: &[f32], scale: f32) -> Option<Result<()>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    Some(add_scaled_with_neon(neon, target, input, scale))
}

/// Updates `target` with `scale * key` and accumulates its dot product with `query` in one pass.
#[inline]
pub(super) fn update_and_dot(
    target: &mut [f32],
    key: &[f32],
    scale: f32,
    query: &[f32],
) -> Option<Result<f32>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    Some(update_and_dot_with_neon(neon, target, key, scale, query))
}

#[inline]
#[expect(
    clippy::suboptimal_flops,
    reason = "Pinned scalar remainder retains separate multiply/add rounding."
)]
fn add_scaled_with_neon(neon: Neon, target: &mut [f32], input: &[f32], scale: f32) -> Result<()> {
    let vectorized = target.len() / 16 * 16;
    let scales = f32x4::splat(neon, scale);
    for (target_block, input_block) in target[..vectorized]
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(input[..vectorized].as_chunks::<16>().0)
    {
        let u0 = scales.mul_add_precise(
            f32x4::from_slice(neon, &input_block[0..4]),
            f32x4::from_slice(neon, &target_block[0..4]),
        );
        let u1 = scales.mul_add_precise(
            f32x4::from_slice(neon, &input_block[4..8]),
            f32x4::from_slice(neon, &target_block[4..8]),
        );
        let u2 = scales.mul_add_precise(
            f32x4::from_slice(neon, &input_block[8..12]),
            f32x4::from_slice(neon, &target_block[8..12]),
        );
        let u3 = scales.mul_add_precise(
            f32x4::from_slice(neon, &input_block[12..16]),
            f32x4::from_slice(neon, &target_block[12..16]),
        );
        if (u0.is_finite() & u1.is_finite() & u2.is_finite() & u3.is_finite()).all_true() {
            u0.store_slice(&mut target_block[0..4]);
            u1.store_slice(&mut target_block[4..8]);
            u2.store_slice(&mut target_block[8..12]);
            u3.store_slice(&mut target_block[12..16]);
        } else {
            for (target, &input) in target_block.iter_mut().zip(input_block) {
                *target = super::super::finite(scale.mul_add(input, *target))?;
            }
        }
    }
    for (target, &input) in target[vectorized..].iter_mut().zip(&input[vectorized..]) {
        *target = super::super::finite(*target + scale * input)?;
    }
    Ok(())
}

#[inline]
#[expect(
    clippy::suboptimal_flops,
    reason = "Pinned scalar remainder retains separate multiply/add rounding."
)]
fn update_and_dot_with_neon(
    neon: Neon,
    target: &mut [f32],
    key: &[f32],
    scale: f32,
    query: &[f32],
) -> Result<f32> {
    let vectorized = target.len() / 16 * 16;
    let scales = f32x4::splat(neon, scale);
    let mut s0 = f32x4::splat(neon, 0.0);
    let mut s1 = f32x4::splat(neon, 0.0);
    let mut s2 = f32x4::splat(neon, 0.0);
    let mut s3 = f32x4::splat(neon, 0.0);
    for ((target_block, key_block), query_block) in target[..vectorized]
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(key[..vectorized].as_chunks::<16>().0)
        .zip(query[..vectorized].as_chunks::<16>().0)
    {
        let u0 = scales.mul_add_precise(
            f32x4::from_slice(neon, &key_block[0..4]),
            f32x4::from_slice(neon, &target_block[0..4]),
        );
        let u1 = scales.mul_add_precise(
            f32x4::from_slice(neon, &key_block[4..8]),
            f32x4::from_slice(neon, &target_block[4..8]),
        );
        let u2 = scales.mul_add_precise(
            f32x4::from_slice(neon, &key_block[8..12]),
            f32x4::from_slice(neon, &target_block[8..12]),
        );
        let u3 = scales.mul_add_precise(
            f32x4::from_slice(neon, &key_block[12..16]),
            f32x4::from_slice(neon, &target_block[12..16]),
        );
        if (u0.is_finite() & u1.is_finite() & u2.is_finite() & u3.is_finite()).all_true() {
            u0.store_slice(&mut target_block[0..4]);
            u1.store_slice(&mut target_block[4..8]);
            u2.store_slice(&mut target_block[8..12]);
            u3.store_slice(&mut target_block[12..16]);
            s0 = u0.mul_add_precise(f32x4::from_slice(neon, &query_block[0..4]), s0);
            s1 = u1.mul_add_precise(f32x4::from_slice(neon, &query_block[4..8]), s1);
            s2 = u2.mul_add_precise(f32x4::from_slice(neon, &query_block[8..12]), s2);
            s3 = u3.mul_add_precise(f32x4::from_slice(neon, &query_block[12..16]), s3);
        } else {
            for (target_elem, &key_elem) in target_block.iter_mut().zip(key_block) {
                *target_elem = super::super::finite(scale.mul_add(key_elem, *target_elem))?;
            }
        }
    }
    let mut result = ((s0 + s2) + (s1 + s3)).reduce_sum();
    for ((target_elem, &key_elem), &query_elem) in target[vectorized..]
        .iter_mut()
        .zip(&key[vectorized..])
        .zip(&query[vectorized..])
    {
        let updated = super::super::finite(*target_elem + scale * key_elem)?;
        *target_elem = updated;
        result += updated * query_elem;
    }
    super::super::finite(result)
}

/// Fuses `decay_and_dot_from` and `update_and_dot` into a single write to `state`
/// when `state.len() <= 256` and `state.len().is_multiple_of(16)`, keeping the
/// decayed row in a register/L1 array across both dot-product passes.
#[inline]
pub(super) fn step_row_from(
    previous: &[f32],
    state: &mut [f32],
    key: &[f32],
    query: &[f32],
    decay: f32,
    value: f32,
    beta: f32,
) -> Option<Result<f32>> {
    if state.len() <= 128 {
        step_row_from_blocks::<8>(previous, state, key, query, decay, value, beta)
    } else {
        step_row_from_blocks::<16>(previous, state, key, query, decay, value, beta)
    }
}

// Swift uses 128-element rows. Its private staging need not initialize the
// unused upper half of the generic 256-element row on every decode step.
// Retain one arithmetic implementation for both bounded capacities.
#[inline]
fn step_row_from_blocks<const BLOCKS: usize>(
    previous: &[f32],
    state: &mut [f32],
    key: &[f32],
    query: &[f32],
    decay: f32,
    value: f32,
    beta: f32,
) -> Option<Result<f32>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let width = state.len();
    if width == 0 || width > BLOCKS * 16 || !width.is_multiple_of(16) {
        return None;
    }
    let blocks = width / 16;
    let decays = f32x4::splat(neon, decay);
    let mut decayed = [[f32x4::splat(neon, 0.0); 4]; BLOCKS];
    let mut s0 = f32x4::splat(neon, 0.0);
    let mut s1 = f32x4::splat(neon, 0.0);
    let mut s2 = f32x4::splat(neon, 0.0);
    let mut s3 = f32x4::splat(neon, 0.0);
    for (b, (prev_block, key_block)) in previous
        .as_chunks::<16>()
        .0
        .iter()
        .zip(key.as_chunks::<16>().0)
        .enumerate()
    {
        let d0 = f32x4::from_slice(neon, &prev_block[0..4]) * decays;
        let d1 = f32x4::from_slice(neon, &prev_block[4..8]) * decays;
        let d2 = f32x4::from_slice(neon, &prev_block[8..12]) * decays;
        let d3 = f32x4::from_slice(neon, &prev_block[12..16]) * decays;
        decayed[b] = [d0, d1, d2, d3];
        s0 = d0.mul_add_precise(f32x4::from_slice(neon, &key_block[0..4]), s0);
        s1 = d1.mul_add_precise(f32x4::from_slice(neon, &key_block[4..8]), s1);
        s2 = d2.mul_add_precise(f32x4::from_slice(neon, &key_block[8..12]), s2);
        s3 = d3.mul_add_precise(f32x4::from_slice(neon, &key_block[12..16]), s3);
    }
    let dot_k = match super::super::finite(((s0 + s2) + (s1 + s3)).reduce_sum()) {
        Ok(val) => val,
        Err(err) => {
            for (b, state_block) in state.as_chunks_mut::<16>().0.iter_mut().enumerate() {
                let [d0, d1, d2, d3] = decayed[b];
                d0.store_slice(&mut state_block[0..4]);
                d1.store_slice(&mut state_block[4..8]);
                d2.store_slice(&mut state_block[8..12]);
                d3.store_slice(&mut state_block[12..16]);
            }
            return Some(Err(err));
        }
    };
    let delta = (value - dot_k) * beta;
    let scales = f32x4::splat(neon, delta);
    let mut q0 = f32x4::splat(neon, 0.0);
    let mut q1 = f32x4::splat(neon, 0.0);
    let mut q2 = f32x4::splat(neon, 0.0);
    let mut q3 = f32x4::splat(neon, 0.0);
    for (b, ((state_block, key_block), query_block)) in state
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(key.as_chunks::<16>().0)
        .zip(query.as_chunks::<16>().0)
        .enumerate()
    {
        let [d0, d1, d2, d3] = decayed[b];
        let u0 = scales.mul_add_precise(f32x4::from_slice(neon, &key_block[0..4]), d0);
        let u1 = scales.mul_add_precise(f32x4::from_slice(neon, &key_block[4..8]), d1);
        let u2 = scales.mul_add_precise(f32x4::from_slice(neon, &key_block[8..12]), d2);
        let u3 = scales.mul_add_precise(f32x4::from_slice(neon, &key_block[12..16]), d3);
        if (u0.is_finite() & u1.is_finite() & u2.is_finite() & u3.is_finite()).all_true() {
            u0.store_slice(&mut state_block[0..4]);
            u1.store_slice(&mut state_block[4..8]);
            u2.store_slice(&mut state_block[8..12]);
            u3.store_slice(&mut state_block[12..16]);
            q0 = u0.mul_add_precise(f32x4::from_slice(neon, &query_block[0..4]), q0);
            q1 = u1.mul_add_precise(f32x4::from_slice(neon, &query_block[4..8]), q1);
            q2 = u2.mul_add_precise(f32x4::from_slice(neon, &query_block[8..12]), q2);
            q3 = u3.mul_add_precise(f32x4::from_slice(neon, &query_block[12..16]), q3);
        } else {
            d0.store_slice(&mut state_block[0..4]);
            d1.store_slice(&mut state_block[4..8]);
            d2.store_slice(&mut state_block[8..12]);
            d3.store_slice(&mut state_block[12..16]);
            for (target_elem, &key_elem) in state_block.iter_mut().zip(key_block) {
                match super::super::finite(delta.mul_add(key_elem, *target_elem)) {
                    Ok(updated) => *target_elem = updated,
                    Err(err) => {
                        for (rem_b, rem_state) in state
                            .as_chunks_mut::<16>()
                            .0
                            .iter_mut()
                            .enumerate()
                            .take(blocks)
                            .skip(b + 1)
                        {
                            let [rd0, rd1, rd2, rd3] = decayed[rem_b];
                            rd0.store_slice(&mut rem_state[0..4]);
                            rd1.store_slice(&mut rem_state[4..8]);
                            rd2.store_slice(&mut rem_state[8..12]);
                            rd3.store_slice(&mut rem_state[12..16]);
                        }
                        return Some(Err(err));
                    }
                }
            }
        }
    }
    Some(super::super::finite(((q0 + q2) + (q1 + q3)).reduce_sum()))
}

/// Fuses in-place `decay_and_dot` and `update_and_dot` into a single read and single write of `state`
/// when `state.len() <= 256` and `state.len().is_multiple_of(16)`.
#[inline]
pub(super) fn step_row_in_place(
    state: &mut [f32],
    key: &[f32],
    query: &[f32],
    decay: f32,
    value: f32,
    beta: f32,
) -> Option<Result<f32>> {
    if state.len() <= 128 {
        step_row_in_place_blocks::<8>(state, key, query, decay, value, beta)
    } else {
        step_row_in_place_blocks::<16>(state, key, query, decay, value, beta)
    }
}

#[inline]
fn step_row_in_place_blocks<const BLOCKS: usize>(
    state: &mut [f32],
    key: &[f32],
    query: &[f32],
    decay: f32,
    value: f32,
    beta: f32,
) -> Option<Result<f32>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let width = state.len();
    if width == 0 || width > BLOCKS * 16 || !width.is_multiple_of(16) {
        return None;
    }
    let blocks = width / 16;
    let decays = f32x4::splat(neon, decay);
    let mut decayed = [[f32x4::splat(neon, 0.0); 4]; BLOCKS];
    let mut s0 = f32x4::splat(neon, 0.0);
    let mut s1 = f32x4::splat(neon, 0.0);
    let mut s2 = f32x4::splat(neon, 0.0);
    let mut s3 = f32x4::splat(neon, 0.0);
    for (b, (state_block, key_block)) in state
        .as_chunks::<16>()
        .0
        .iter()
        .zip(key.as_chunks::<16>().0)
        .enumerate()
    {
        let d0 = f32x4::from_slice(neon, &state_block[0..4]) * decays;
        let d1 = f32x4::from_slice(neon, &state_block[4..8]) * decays;
        let d2 = f32x4::from_slice(neon, &state_block[8..12]) * decays;
        let d3 = f32x4::from_slice(neon, &state_block[12..16]) * decays;
        decayed[b] = [d0, d1, d2, d3];
        s0 = d0.mul_add_precise(f32x4::from_slice(neon, &key_block[0..4]), s0);
        s1 = d1.mul_add_precise(f32x4::from_slice(neon, &key_block[4..8]), s1);
        s2 = d2.mul_add_precise(f32x4::from_slice(neon, &key_block[8..12]), s2);
        s3 = d3.mul_add_precise(f32x4::from_slice(neon, &key_block[12..16]), s3);
    }
    let dot_k = match super::super::finite(((s0 + s2) + (s1 + s3)).reduce_sum()) {
        Ok(val) => val,
        Err(err) => {
            for (b, state_block) in state.as_chunks_mut::<16>().0.iter_mut().enumerate() {
                let [d0, d1, d2, d3] = decayed[b];
                d0.store_slice(&mut state_block[0..4]);
                d1.store_slice(&mut state_block[4..8]);
                d2.store_slice(&mut state_block[8..12]);
                d3.store_slice(&mut state_block[12..16]);
            }
            return Some(Err(err));
        }
    };
    let delta = (value - dot_k) * beta;
    let scales = f32x4::splat(neon, delta);
    let mut q0 = f32x4::splat(neon, 0.0);
    let mut q1 = f32x4::splat(neon, 0.0);
    let mut q2 = f32x4::splat(neon, 0.0);
    let mut q3 = f32x4::splat(neon, 0.0);
    for (b, ((state_block, key_block), query_block)) in state
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(key.as_chunks::<16>().0)
        .zip(query.as_chunks::<16>().0)
        .enumerate()
    {
        let [d0, d1, d2, d3] = decayed[b];
        let u0 = scales.mul_add_precise(f32x4::from_slice(neon, &key_block[0..4]), d0);
        let u1 = scales.mul_add_precise(f32x4::from_slice(neon, &key_block[4..8]), d1);
        let u2 = scales.mul_add_precise(f32x4::from_slice(neon, &key_block[8..12]), d2);
        let u3 = scales.mul_add_precise(f32x4::from_slice(neon, &key_block[12..16]), d3);
        if (u0.is_finite() & u1.is_finite() & u2.is_finite() & u3.is_finite()).all_true() {
            u0.store_slice(&mut state_block[0..4]);
            u1.store_slice(&mut state_block[4..8]);
            u2.store_slice(&mut state_block[8..12]);
            u3.store_slice(&mut state_block[12..16]);
            q0 = u0.mul_add_precise(f32x4::from_slice(neon, &query_block[0..4]), q0);
            q1 = u1.mul_add_precise(f32x4::from_slice(neon, &query_block[4..8]), q1);
            q2 = u2.mul_add_precise(f32x4::from_slice(neon, &query_block[8..12]), q2);
            q3 = u3.mul_add_precise(f32x4::from_slice(neon, &query_block[12..16]), q3);
        } else {
            d0.store_slice(&mut state_block[0..4]);
            d1.store_slice(&mut state_block[4..8]);
            d2.store_slice(&mut state_block[8..12]);
            d3.store_slice(&mut state_block[12..16]);
            for (target_elem, &key_elem) in state_block.iter_mut().zip(key_block) {
                match super::super::finite(delta.mul_add(key_elem, *target_elem)) {
                    Ok(updated) => *target_elem = updated,
                    Err(err) => {
                        for (rem_b, rem_state) in state
                            .as_chunks_mut::<16>()
                            .0
                            .iter_mut()
                            .enumerate()
                            .take(blocks)
                            .skip(b + 1)
                        {
                            let [rd0, rd1, rd2, rd3] = decayed[rem_b];
                            rd0.store_slice(&mut rem_state[0..4]);
                            rd1.store_slice(&mut rem_state[4..8]);
                            rd2.store_slice(&mut rem_state[8..12]);
                            rd3.store_slice(&mut rem_state[12..16]);
                        }
                        return Some(Err(err));
                    }
                }
            }
        }
    }
    Some(super::super::finite(((q0 + q2) + (q1 + q3)).reduce_sum()))
}

#[inline]
fn flush_blocks(state: &mut [f32], blocks: &[[f32x4<Neon>; 4]], count: usize) {
    for (state_block, &[v0, v1, v2, v3]) in state
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(blocks.iter())
        .take(count)
    {
        v0.store_slice(&mut state_block[0..4]);
        v1.store_slice(&mut state_block[4..8]);
        v2.store_slice(&mut state_block[8..12]);
        v3.store_slice(&mut state_block[12..16]);
    }
}

/// Steps one GDN state row across `inputs.len()` tokens while keeping the state row
/// in private width-selected blocks across tokens, writing `state` to memory only once
/// after all tokens complete (or flushing exact partial state on a non-finite error).
#[inline]
pub(super) fn step_row_tokens_from(
    previous: &[f32],
    state: &mut [f32],
    inputs: &[super::ChunkInput<'_>],
    row: usize,
    stride: usize,
    scale: f32,
    output: &mut [f32],
) -> Option<Result<()>> {
    if state.len() <= 128 {
        step_row_tokens_from_blocks::<8>(previous, state, inputs, row, stride, scale, output)
    } else {
        step_row_tokens_from_blocks::<16>(previous, state, inputs, row, stride, scale, output)
    }
}

#[inline]
fn step_row_tokens_from_blocks<const BLOCKS: usize>(
    previous: &[f32],
    state: &mut [f32],
    inputs: &[super::ChunkInput<'_>],
    row: usize,
    stride: usize,
    scale: f32,
    output: &mut [f32],
) -> Option<Result<()>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let width = state.len();
    if width == 0 || width > BLOCKS * 16 || !width.is_multiple_of(16) {
        return None;
    }
    let blocks = width / 16;
    let mut current = [[f32x4::splat(neon, 0.0); 4]; BLOCKS];
    for (b, prev_block) in previous.as_chunks::<16>().0.iter().enumerate() {
        current[b] = [
            f32x4::from_slice(neon, &prev_block[0..4]),
            f32x4::from_slice(neon, &prev_block[4..8]),
            f32x4::from_slice(neon, &prev_block[8..12]),
            f32x4::from_slice(neon, &prev_block[12..16]),
        ];
    }
    for (token, input) in inputs.iter().enumerate() {
        let decays = f32x4::splat(neon, input.decay);
        let mut s0 = f32x4::splat(neon, 0.0);
        let mut s1 = f32x4::splat(neon, 0.0);
        let mut s2 = f32x4::splat(neon, 0.0);
        let mut s3 = f32x4::splat(neon, 0.0);
        for (b, key_block) in input.key.as_chunks::<16>().0.iter().enumerate() {
            let [c0, c1, c2, c3] = current[b];
            let d0 = c0 * decays;
            let d1 = c1 * decays;
            let d2 = c2 * decays;
            let d3 = c3 * decays;
            current[b] = [d0, d1, d2, d3];
            s0 = d0.mul_add_precise(f32x4::from_slice(neon, &key_block[0..4]), s0);
            s1 = d1.mul_add_precise(f32x4::from_slice(neon, &key_block[4..8]), s1);
            s2 = d2.mul_add_precise(f32x4::from_slice(neon, &key_block[8..12]), s2);
            s3 = d3.mul_add_precise(f32x4::from_slice(neon, &key_block[12..16]), s3);
        }
        let dot_k = match super::super::finite(((s0 + s2) + (s1 + s3)).reduce_sum()) {
            Ok(val) => val,
            Err(err) => {
                flush_blocks(state, &current, blocks);
                return Some(Err(err));
            }
        };
        let delta = (input.value[row] - dot_k) * input.beta;
        let scales = f32x4::splat(neon, delta);
        let mut q0 = f32x4::splat(neon, 0.0);
        let mut q1 = f32x4::splat(neon, 0.0);
        let mut q2 = f32x4::splat(neon, 0.0);
        let mut q3 = f32x4::splat(neon, 0.0);
        for (b, (key_block, query_block)) in input
            .key
            .as_chunks::<16>()
            .0
            .iter()
            .zip(input.query.as_chunks::<16>().0)
            .enumerate()
        {
            let [d0, d1, d2, d3] = current[b];
            let u0 = scales.mul_add_precise(f32x4::from_slice(neon, &key_block[0..4]), d0);
            let u1 = scales.mul_add_precise(f32x4::from_slice(neon, &key_block[4..8]), d1);
            let u2 = scales.mul_add_precise(f32x4::from_slice(neon, &key_block[8..12]), d2);
            let u3 = scales.mul_add_precise(f32x4::from_slice(neon, &key_block[12..16]), d3);
            if (u0.is_finite() & u1.is_finite() & u2.is_finite() & u3.is_finite()).all_true() {
                current[b] = [u0, u1, u2, u3];
                q0 = u0.mul_add_precise(f32x4::from_slice(neon, &query_block[0..4]), q0);
                q1 = u1.mul_add_precise(f32x4::from_slice(neon, &query_block[4..8]), q1);
                q2 = u2.mul_add_precise(f32x4::from_slice(neon, &query_block[8..12]), q2);
                q3 = u3.mul_add_precise(f32x4::from_slice(neon, &query_block[12..16]), q3);
            } else {
                // Earlier blocks already contain checked updates; this and
                // later blocks still contain their exact decayed values.
                flush_blocks(state, &current, blocks);
                let state_block = &mut state[b * 16..(b + 1) * 16];
                for (target_elem, &key_elem) in state_block.iter_mut().zip(key_block) {
                    match super::super::finite(delta.mul_add(key_elem, *target_elem)) {
                        Ok(updated) => *target_elem = updated,
                        Err(err) => return Some(Err(err)),
                    }
                }
            }
        }
        match super::super::finite(((q0 + q2) + (q1 + q3)).reduce_sum()) {
            Ok(out) => output[token * stride + row] = out * scale,
            Err(err) => {
                flush_blocks(state, &current, blocks);
                return Some(Err(err));
            }
        }
    }
    flush_blocks(state, &current, blocks);
    Some(Ok(()))
}

/// Steps 4 GDN state rows simultaneously from `previous` into `states`, sharing
/// `key` and `query` vector loads across all 4 rows. Returns `None` if dimensions
/// are unsupported or if any non-finite intermediate occurs before `states` is mutated,
/// allowing the caller to fall back to the sequential per-row path for exact error state.
#[inline]
pub(super) fn step_four_rows_from(
    previous: [&[f32]; 4],
    states: [&mut [f32]; 4],
    key: &[f32],
    query: &[f32],
    decay: f32,
    values: [f32; 4],
    beta: f32,
) -> Option<[f32; 4]> {
    if states[0].len() <= 128 {
        step_four_rows_from_blocks::<8>(previous, states, key, query, decay, values, beta)
    } else {
        step_four_rows_from_blocks::<16>(previous, states, key, query, decay, values, beta)
    }
}

#[inline]
fn step_four_rows_from_blocks<const BLOCKS: usize>(
    previous: [&[f32]; 4],
    states: [&mut [f32]; 4],
    key: &[f32],
    query: &[f32],
    decay: f32,
    values: [f32; 4],
    beta: f32,
) -> Option<[f32; 4]> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let width = states[0].len();
    if width == 0 || width > BLOCKS * 16 || !width.is_multiple_of(16) {
        return None;
    }
    let mut work = [[[f32x4::splat(neon, 0.0); 4]; BLOCKS]; 4];
    let scales = decay_four_rows_from_blocks(neon, previous, &mut work, key, decay, values, beta)?;
    let outputs = update_four_rows_blocks(neon, &mut work, key, query, scales)?;
    for (state, row_blocks) in states.into_iter().zip(work.iter()) {
        flush_blocks(state, row_blocks, width / 16);
    }
    Some(outputs)
}

// Four rows retain sixteen FMA accumulators. An eight-row phase would need all
// thirty-two AArch64 vector registers before its input, decay or mask vectors.
// Keep the complete native four-register tree for each row, but bound its live
// arithmetic phase independently of the eight-row transactional staging owner.
#[inline]
fn decay_four_rows_from_blocks<const BLOCKS: usize>(
    neon: Neon,
    previous: [&[f32]; 4],
    work: &mut [[[f32x4<Neon>; 4]; BLOCKS]; 4],
    key: &[f32],
    decay: f32,
    values: [f32; 4],
    beta: f32,
) -> Option<[f32; 4]> {
    let decays = f32x4::splat(neon, decay);
    let mut s0 = [f32x4::splat(neon, 0.0); 4];
    let mut s1 = [f32x4::splat(neon, 0.0); 4];
    let mut s2 = [f32x4::splat(neon, 0.0); 4];
    let mut s3 = [f32x4::splat(neon, 0.0); 4];

    for (b, key_block) in key.as_chunks::<16>().0.iter().enumerate() {
        let k0 = f32x4::from_slice(neon, &key_block[0..4]);
        let k1 = f32x4::from_slice(neon, &key_block[4..8]);
        let k2 = f32x4::from_slice(neon, &key_block[8..12]);
        let k3 = f32x4::from_slice(neon, &key_block[12..16]);
        let offset = b * 16;
        for r in 0..4 {
            let prev_block = &previous[r][offset..offset + 16];
            let d0 = f32x4::from_slice(neon, &prev_block[0..4]) * decays;
            let d1 = f32x4::from_slice(neon, &prev_block[4..8]) * decays;
            let d2 = f32x4::from_slice(neon, &prev_block[8..12]) * decays;
            let d3 = f32x4::from_slice(neon, &prev_block[12..16]) * decays;
            work[r][b] = [d0, d1, d2, d3];
            s0[r] = d0.mul_add_precise(k0, s0[r]);
            s1[r] = d1.mul_add_precise(k1, s1[r]);
            s2[r] = d2.mul_add_precise(k2, s2[r]);
            s3[r] = d3.mul_add_precise(k3, s3[r]);
        }
    }

    let mut scales = [0.0; 4];
    for r in 0..4 {
        let dot_k = ((s0[r] + s2[r]) + (s1[r] + s3[r])).reduce_sum();
        if !dot_k.is_finite() {
            return None;
        }
        scales[r] = (values[r] - dot_k) * beta;
    }

    Some(scales)
}

#[inline]
fn decay_four_rows_blocks<const BLOCKS: usize>(
    neon: Neon,
    work: &mut [[[f32x4<Neon>; 4]; BLOCKS]; 4],
    key: &[f32],
    decay: f32,
    values: [f32; 4],
    beta: f32,
) -> Option<[f32; 4]> {
    let decays = f32x4::splat(neon, decay);
    let mut s0 = [f32x4::splat(neon, 0.0); 4];
    let mut s1 = [f32x4::splat(neon, 0.0); 4];
    let mut s2 = [f32x4::splat(neon, 0.0); 4];
    let mut s3 = [f32x4::splat(neon, 0.0); 4];

    for (b, key_block) in key.as_chunks::<16>().0.iter().enumerate() {
        let k0 = f32x4::from_slice(neon, &key_block[0..4]);
        let k1 = f32x4::from_slice(neon, &key_block[4..8]);
        let k2 = f32x4::from_slice(neon, &key_block[8..12]);
        let k3 = f32x4::from_slice(neon, &key_block[12..16]);
        for r in 0..4 {
            let [c0, c1, c2, c3] = work[r][b];
            let d0 = c0 * decays;
            let d1 = c1 * decays;
            let d2 = c2 * decays;
            let d3 = c3 * decays;
            work[r][b] = [d0, d1, d2, d3];
            s0[r] = d0.mul_add_precise(k0, s0[r]);
            s1[r] = d1.mul_add_precise(k1, s1[r]);
            s2[r] = d2.mul_add_precise(k2, s2[r]);
            s3[r] = d3.mul_add_precise(k3, s3[r]);
        }
    }

    let mut scales = [0.0; 4];
    for r in 0..4 {
        let dot_k = ((s0[r] + s2[r]) + (s1[r] + s3[r])).reduce_sum();
        if !dot_k.is_finite() {
            return None;
        }
        scales[r] = (values[r] - dot_k) * beta;
    }

    Some(scales)
}

#[inline]
fn update_four_rows_blocks<const BLOCKS: usize>(
    neon: Neon,
    work: &mut [[[f32x4<Neon>; 4]; BLOCKS]; 4],
    key: &[f32],
    query: &[f32],
    scales: [f32; 4],
) -> Option<[f32; 4]> {
    let mut finite_mask = f32x4::splat(neon, 0.0).is_finite();
    let output = update_four_rows_blocks_masked(neon, work, key, query, scales, &mut finite_mask)?;
    finite_mask.all_true().then_some(output)
}

#[inline]
fn update_four_rows_blocks_masked<const BLOCKS: usize>(
    neon: Neon,
    work: &mut [[[f32x4<Neon>; 4]; BLOCKS]; 4],
    key: &[f32],
    query: &[f32],
    scales: [f32; 4],
    finite_mask: &mut fearless_simd::mask32x4<Neon>,
) -> Option<[f32; 4]> {
    let scales = scales.map(|scale| f32x4::splat(neon, scale));
    let mut q0 = [f32x4::splat(neon, 0.0); 4];
    let mut q1 = [f32x4::splat(neon, 0.0); 4];
    let mut q2 = [f32x4::splat(neon, 0.0); 4];
    let mut q3 = [f32x4::splat(neon, 0.0); 4];

    for (b, (key_block, query_block)) in key
        .as_chunks::<16>()
        .0
        .iter()
        .zip(query.as_chunks::<16>().0)
        .enumerate()
    {
        let k0 = f32x4::from_slice(neon, &key_block[0..4]);
        let k1 = f32x4::from_slice(neon, &key_block[4..8]);
        let k2 = f32x4::from_slice(neon, &key_block[8..12]);
        let k3 = f32x4::from_slice(neon, &key_block[12..16]);
        let qv0 = f32x4::from_slice(neon, &query_block[0..4]);
        let qv1 = f32x4::from_slice(neon, &query_block[4..8]);
        let qv2 = f32x4::from_slice(neon, &query_block[8..12]);
        let qv3 = f32x4::from_slice(neon, &query_block[12..16]);
        for r in 0..4 {
            let [d0, d1, d2, d3] = work[r][b];
            let scale = scales[r];
            let u0 = scale.mul_add_precise(k0, d0);
            let u1 = scale.mul_add_precise(k1, d1);
            let u2 = scale.mul_add_precise(k2, d2);
            let u3 = scale.mul_add_precise(k3, d3);
            *finite_mask &= u0.is_finite() & u1.is_finite() & u2.is_finite() & u3.is_finite();
            work[r][b] = [u0, u1, u2, u3];
            q0[r] = u0.mul_add_precise(qv0, q0[r]);
            q1[r] = u1.mul_add_precise(qv1, q1[r]);
            q2[r] = u2.mul_add_precise(qv2, q2[r]);
            q3[r] = u3.mul_add_precise(qv3, q3[r]);
        }
    }

    let mut outputs = [0.0_f32; 4];
    for r in 0..4 {
        let out = ((q0[r] + q2[r]) + (q1[r] + q3[r])).reduce_sum();
        if !out.is_finite() {
            return None;
        }
        outputs[r] = out;
    }

    Some(outputs)
}

/// Steps 4 GDN state rows across `inputs.len()` tokens simultaneously, keeping all 4
/// state rows in registers/stack blocks across tokens and sharing `key` and `query`
/// vector loads across the 4 rows at every token. Returns `false` without mutating
/// `states` or `output` if any non-finite value occurs, allowing the caller to fall
/// back to the sequential per-row path for exact partial-mutation semantics.
#[inline]
pub(super) fn step_four_rows_tokens_from(
    previous: [&[f32]; 4],
    states: [&mut [f32]; 4],
    inputs: &[super::ChunkInput<'_>],
    row: usize,
    stride: usize,
    scale: f32,
    output: &mut [f32],
) -> bool {
    if states[0].len() <= 128 {
        step_four_rows_tokens_from_blocks::<8>(previous, states, inputs, row, stride, scale, output)
    } else {
        step_four_rows_tokens_from_blocks::<16>(
            previous, states, inputs, row, stride, scale, output,
        )
    }
}

#[inline]
#[expect(
    clippy::too_many_lines,
    reason = "Keep 4-row multi-token register-resident NEON GDN recurrence together."
)]
fn step_four_rows_tokens_from_blocks<const BLOCKS: usize>(
    previous: [&[f32]; 4],
    states: [&mut [f32]; 4],
    inputs: &[super::ChunkInput<'_>],
    row: usize,
    stride: usize,
    scale: f32,
    output: &mut [f32],
) -> bool {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return false;
    };
    let width = states[0].len();
    if width == 0 || width > BLOCKS * 16 || !width.is_multiple_of(16) || inputs.len() > 64 {
        return false;
    }
    let blocks = width / 16;
    // Every update remains private until the group commits. Preserve all lane
    // predicates while reducing their combined mask once before publication.
    let mut finite_mask = f32x4::splat(neon, 0.0).is_finite();
    let mut current = [[[f32x4::splat(neon, 0.0); 4]; BLOCKS]; 4];
    for (r, prev) in previous.iter().enumerate() {
        for (b, prev_block) in prev.as_chunks::<16>().0.iter().enumerate() {
            current[r][b] = [
                f32x4::from_slice(neon, &prev_block[0..4]),
                f32x4::from_slice(neon, &prev_block[4..8]),
                f32x4::from_slice(neon, &prev_block[8..12]),
                f32x4::from_slice(neon, &prev_block[12..16]),
            ];
        }
    }
    let mut staged_out = [[0.0_f32; 64]; 4];

    for (token, input) in inputs.iter().enumerate() {
        let decays = f32x4::splat(neon, input.decay);
        let mut s0 = [f32x4::splat(neon, 0.0); 4];
        let mut s1 = [f32x4::splat(neon, 0.0); 4];
        let mut s2 = [f32x4::splat(neon, 0.0); 4];
        let mut s3 = [f32x4::splat(neon, 0.0); 4];
        for (b, key_block) in input.key.as_chunks::<16>().0.iter().enumerate() {
            let k0 = f32x4::from_slice(neon, &key_block[0..4]);
            let k1 = f32x4::from_slice(neon, &key_block[4..8]);
            let k2 = f32x4::from_slice(neon, &key_block[8..12]);
            let k3 = f32x4::from_slice(neon, &key_block[12..16]);
            for r in 0..4 {
                let [c0, c1, c2, c3] = current[r][b];
                let d0 = c0 * decays;
                let d1 = c1 * decays;
                let d2 = c2 * decays;
                let d3 = c3 * decays;
                current[r][b] = [d0, d1, d2, d3];
                s0[r] = d0.mul_add_precise(k0, s0[r]);
                s1[r] = d1.mul_add_precise(k1, s1[r]);
                s2[r] = d2.mul_add_precise(k2, s2[r]);
                s3[r] = d3.mul_add_precise(k3, s3[r]);
            }
        }
        let mut scales = [f32x4::splat(neon, 0.0); 4];
        for r in 0..4 {
            let dot_k = ((s0[r] + s2[r]) + (s1[r] + s3[r])).reduce_sum();
            if !dot_k.is_finite() {
                return false;
            }
            scales[r] = f32x4::splat(neon, (input.value[row + r] - dot_k) * input.beta);
        }
        let mut q0 = [f32x4::splat(neon, 0.0); 4];
        let mut q1 = [f32x4::splat(neon, 0.0); 4];
        let mut q2 = [f32x4::splat(neon, 0.0); 4];
        let mut q3 = [f32x4::splat(neon, 0.0); 4];
        for (b, (key_block, query_block)) in input
            .key
            .as_chunks::<16>()
            .0
            .iter()
            .zip(input.query.as_chunks::<16>().0)
            .enumerate()
        {
            let k0 = f32x4::from_slice(neon, &key_block[0..4]);
            let k1 = f32x4::from_slice(neon, &key_block[4..8]);
            let k2 = f32x4::from_slice(neon, &key_block[8..12]);
            let k3 = f32x4::from_slice(neon, &key_block[12..16]);
            let qv0 = f32x4::from_slice(neon, &query_block[0..4]);
            let qv1 = f32x4::from_slice(neon, &query_block[4..8]);
            let qv2 = f32x4::from_slice(neon, &query_block[8..12]);
            let qv3 = f32x4::from_slice(neon, &query_block[12..16]);
            for r in 0..4 {
                let [d0, d1, d2, d3] = current[r][b];
                let delta = scales[r];
                let u0 = delta.mul_add_precise(k0, d0);
                let u1 = delta.mul_add_precise(k1, d1);
                let u2 = delta.mul_add_precise(k2, d2);
                let u3 = delta.mul_add_precise(k3, d3);
                finite_mask &= u0.is_finite() & u1.is_finite() & u2.is_finite() & u3.is_finite();
                current[r][b] = [u0, u1, u2, u3];
                q0[r] = u0.mul_add_precise(qv0, q0[r]);
                q1[r] = u1.mul_add_precise(qv1, q1[r]);
                q2[r] = u2.mul_add_precise(qv2, q2[r]);
                q3[r] = u3.mul_add_precise(qv3, q3[r]);
            }
        }
        for r in 0..4 {
            let out = ((q0[r] + q2[r]) + (q1[r] + q3[r])).reduce_sum();
            if !out.is_finite() {
                return false;
            }
            staged_out[r][token] = out * scale;
        }
    }

    if !finite_mask.all_true() {
        return false;
    }

    for (state, row_blocks) in states.into_iter().zip(current.iter()) {
        flush_blocks(state, row_blocks, blocks);
    }
    for (r, row_staged) in staged_out.iter().enumerate() {
        for (token, &val) in row_staged.iter().enumerate().take(inputs.len()) {
            output[token * stride + row + r] = val;
        }
    }
    true
}

/// Steps 8 GDN state rows transactionally from `previous` into `states`, sharing
/// `key` and `query` vector loads within bounded four-row arithmetic phases.
/// Returns `None` if dimensions are unsupported or if any non-finite intermediate
/// occurs before `states` is mutated,
/// allowing the caller to fall back to the sequential per-row path for exact error state.
#[inline]
pub(super) fn step_eight_rows_from(
    previous: [&[f32]; 8],
    states: [&mut [f32]; 8],
    key: &[f32],
    query: &[f32],
    decay: f32,
    values: [f32; 8],
    beta: f32,
) -> Option<[f32; 8]> {
    if states[0].len() <= 128 {
        step_eight_rows_from_blocks::<8>(previous, states, key, query, decay, values, beta)
    } else {
        step_eight_rows_from_blocks::<16>(previous, states, key, query, decay, values, beta)
    }
}

#[inline]
fn step_eight_rows_from_blocks<const BLOCKS: usize>(
    previous: [&[f32]; 8],
    states: [&mut [f32]; 8],
    key: &[f32],
    query: &[f32],
    decay: f32,
    values: [f32; 8],
    beta: f32,
) -> Option<[f32; 8]> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let width = states[0].len();
    if width == 0 || width > BLOCKS * 16 || !width.is_multiple_of(16) {
        return None;
    }
    let mut work = [[[f32x4::splat(neon, 0.0); 4]; BLOCKS]; 8];
    let mut scales = [0.0; 8];
    // Validate every key dot before beginning updates, retaining the original
    // phase order. Four-row phases reread the small key/query rows while
    // halving the maximum number of live dot accumulators from 32 to 16.
    for (group, ((previous, work), values)) in previous
        .as_chunks::<4>()
        .0
        .iter()
        .zip(work.as_chunks_mut::<4>().0)
        .zip(values.as_chunks::<4>().0)
        .enumerate()
    {
        let deltas = decay_four_rows_from_blocks(neon, *previous, work, key, decay, *values, beta)?;
        scales[group * 4..group * 4 + 4].copy_from_slice(&deltas);
    }
    let mut outputs = [0.0; 8];
    for (group, (work, scales)) in work
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(scales.as_chunks::<4>().0)
        .enumerate()
    {
        let values = update_four_rows_blocks(neon, work, key, query, *scales)?;
        outputs[group * 4..group * 4 + 4].copy_from_slice(&values);
    }
    // A failed second phase still leaves all eight destinations untouched.
    // The caller can replay rows in order to recover exact failure mutation.
    for (state, row_blocks) in states.into_iter().zip(work.iter()) {
        flush_blocks(state, row_blocks, width / 16);
    }
    Some(outputs)
}

/// Selects baseline NEON once for an admitted complete head, retaining the
/// existing eight-row arithmetic, group output boundary and defensive replay.
pub(super) fn try_step_head_provisional(
    previous: &[f32],
    state: &mut [f32],
    input: super::ChunkInput<'_>,
    scale: f32,
    output: &mut [f32],
) -> Option<Result<()>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let width = input.key.len();
    for (((previous, state), values), output) in previous
        .chunks_exact(width * 8)
        .zip(state.chunks_exact_mut(width * 8))
        .zip(input.value.as_chunks::<8>().0)
        .zip(output.as_chunks_mut::<8>().0)
    {
        let previous = std::array::from_fn(|row| &previous[row * width..(row + 1) * width]);
        let (h0, h1) = state.split_at_mut(4 * width);
        let (s01, s23) = h0.split_at_mut(2 * width);
        let (s45, s67) = h1.split_at_mut(2 * width);
        let (s0, s1) = s01.split_at_mut(width);
        let (s2, s3) = s23.split_at_mut(width);
        let (s4, s5) = s45.split_at_mut(width);
        let (s6, s7) = s67.split_at_mut(width);
        let mut states = [s0, s1, s2, s3, s4, s5, s6, s7];
        let values = match step_eight_rows_into(
            neon,
            previous,
            &mut states,
            input.key,
            input.query,
            input.decay,
            *values,
            input.beta,
        ) {
            Some(values) => values,
            None => match super::step_eight_rows_from(
                previous,
                states,
                input.key,
                input.query,
                input.decay,
                *values,
                input.beta,
            ) {
                Ok(values) => values,
                Err(error) => return Some(Err(error)),
            },
        };
        for (output, value) in output.iter_mut().zip(values) {
            *output = value * scale;
        }
    }
    Some(Ok(()))
}

/// Uses disposable graph delta rows for decayed and updated state. The source
/// remains immutable, allowing the ordinary helper to recover original errors
/// after any native failure without reading a partly updated destination.
pub(super) fn step_eight_rows_provisional(
    previous: [&[f32]; 8],
    mut states: [&mut [f32]; 8],
    key: &[f32],
    query: &[f32],
    decay: f32,
    values: [f32; 8],
    beta: f32,
) -> Result<[f32; 8]> {
    let width = key.len();
    if width > 0
        && width <= 256
        && width.is_multiple_of(16)
        && let Some(outputs) = fearless_simd::Level::baseline().as_neon().and_then(|neon| {
            step_eight_rows_into(neon, previous, &mut states, key, query, decay, values, beta)
        })
    {
        return Ok(outputs);
    }
    // The native attempt may have dirtied disposable rows. Defensive replay
    // reads only `previous`, preserving the original row/error precedence.
    super::step_eight_rows_from(previous, states, key, query, decay, values, beta)
}

#[expect(
    clippy::too_many_arguments,
    reason = "Keep immutable source and disposable eight-row work ownership explicit."
)]
fn step_eight_rows_into(
    neon: Neon,
    previous: [&[f32]; 8],
    states: &mut [&mut [f32]; 8],
    key: &[f32],
    query: &[f32],
    decay: f32,
    values: [f32; 8],
    beta: f32,
) -> Option<[f32; 8]> {
    let mut scales = [0.0; 8];
    for (group, (previous, values)) in previous
        .as_chunks::<4>()
        .0
        .iter()
        .zip(values.as_chunks::<4>().0)
        .enumerate()
    {
        let deltas = decay_four_rows_scales(neon, *previous, key, decay, *values, beta)?;
        scales[group * 4..group * 4 + 4].copy_from_slice(&deltas);
    }
    // Every key dot precedes updates, as in the existing eight-row helper.
    // Re-read L1-resident `previous` in pass 2 so `states` is written only once.
    let mut outputs = [0.0; 8];
    for (group, ((previous, states), scales)) in previous
        .as_chunks::<4>()
        .0
        .iter()
        .zip(states.as_chunks_mut::<4>().0)
        .zip(scales.as_chunks::<4>().0)
        .enumerate()
    {
        let values = update_four_rows_from(neon, *previous, states, key, query, decay, *scales)?;
        outputs[group * 4..group * 4 + 4].copy_from_slice(&values);
    }
    Some(outputs)
}

fn decay_four_rows_scales(
    neon: Neon,
    previous: [&[f32]; 4],
    key: &[f32],
    decay: f32,
    values: [f32; 4],
    beta: f32,
) -> Option<[f32; 4]> {
    let decays = f32x4::splat(neon, decay);
    let mut s0 = [f32x4::splat(neon, 0.0); 4];
    let mut s1 = [f32x4::splat(neon, 0.0); 4];
    let mut s2 = [f32x4::splat(neon, 0.0); 4];
    let mut s3 = [f32x4::splat(neon, 0.0); 4];
    for (b, key_block) in key.as_chunks::<16>().0.iter().enumerate() {
        let k0 = f32x4::from_slice(neon, &key_block[0..4]);
        let k1 = f32x4::from_slice(neon, &key_block[4..8]);
        let k2 = f32x4::from_slice(neon, &key_block[8..12]);
        let k3 = f32x4::from_slice(neon, &key_block[12..16]);
        let offset = b * 16;
        for r in 0..4 {
            let prev_block = &previous[r][offset..offset + 16];
            let d0 = f32x4::from_slice(neon, &prev_block[0..4]) * decays;
            let d1 = f32x4::from_slice(neon, &prev_block[4..8]) * decays;
            let d2 = f32x4::from_slice(neon, &prev_block[8..12]) * decays;
            let d3 = f32x4::from_slice(neon, &prev_block[12..16]) * decays;
            s0[r] = d0.mul_add_precise(k0, s0[r]);
            s1[r] = d1.mul_add_precise(k1, s1[r]);
            s2[r] = d2.mul_add_precise(k2, s2[r]);
            s3[r] = d3.mul_add_precise(k3, s3[r]);
        }
    }
    let mut scales = [0.0; 4];
    for r in 0..4 {
        let dot_k = ((s0[r] + s2[r]) + (s1[r] + s3[r])).reduce_sum();
        if !dot_k.is_finite() {
            return None;
        }
        scales[r] = (values[r] - dot_k) * beta;
    }
    Some(scales)
}

fn update_four_rows_from(
    neon: Neon,
    previous: [&[f32]; 4],
    states: &mut [&mut [f32]; 4],
    key: &[f32],
    query: &[f32],
    decay: f32,
    scales: [f32; 4],
) -> Option<[f32; 4]> {
    let decays = f32x4::splat(neon, decay);
    let scales = scales.map(|scale| f32x4::splat(neon, scale));
    let mut q0 = [f32x4::splat(neon, 0.0); 4];
    let mut q1 = [f32x4::splat(neon, 0.0); 4];
    let mut q2 = [f32x4::splat(neon, 0.0); 4];
    let mut q3 = [f32x4::splat(neon, 0.0); 4];
    for (b, (key_block, query_block)) in key
        .as_chunks::<16>()
        .0
        .iter()
        .zip(query.as_chunks::<16>().0)
        .enumerate()
    {
        let k0 = f32x4::from_slice(neon, &key_block[0..4]);
        let k1 = f32x4::from_slice(neon, &key_block[4..8]);
        let k2 = f32x4::from_slice(neon, &key_block[8..12]);
        let k3 = f32x4::from_slice(neon, &key_block[12..16]);
        let qv0 = f32x4::from_slice(neon, &query_block[0..4]);
        let qv1 = f32x4::from_slice(neon, &query_block[4..8]);
        let qv2 = f32x4::from_slice(neon, &query_block[8..12]);
        let qv3 = f32x4::from_slice(neon, &query_block[12..16]);
        let offset = b * 16;
        for r in 0..4 {
            let prev_block = &previous[r][offset..offset + 16];
            let state = &mut states[r][offset..offset + 16];
            let scale = scales[r];
            let d0 = f32x4::from_slice(neon, &prev_block[0..4]) * decays;
            let d1 = f32x4::from_slice(neon, &prev_block[4..8]) * decays;
            let d2 = f32x4::from_slice(neon, &prev_block[8..12]) * decays;
            let d3 = f32x4::from_slice(neon, &prev_block[12..16]) * decays;
            let u0 = scale.mul_add_precise(k0, d0);
            let u1 = scale.mul_add_precise(k1, d1);
            let u2 = scale.mul_add_precise(k2, d2);
            let u3 = scale.mul_add_precise(k3, d3);
            u0.store_slice(&mut state[0..4]);
            u1.store_slice(&mut state[4..8]);
            u2.store_slice(&mut state[8..12]);
            u3.store_slice(&mut state[12..16]);
            q0[r] = u0.mul_add_precise(qv0, q0[r]);
            q1[r] = u1.mul_add_precise(qv1, q1[r]);
            q2[r] = u2.mul_add_precise(qv2, q2[r]);
            q3[r] = u3.mul_add_precise(qv3, q3[r]);
        }
    }
    let mut outputs = [0.0; 4];
    for r in 0..4 {
        // Every updated element contributes to this complete pinned dot. An
        // infinity or NaN remains nonfinite through its query FMA and the
        // reduction, including a signed-zero query. A finite output therefore
        // also proves finite state without a second set of element predicates.
        // This is private disposable state; rejection replays immutable source.
        let out = ((q0[r] + q2[r]) + (q1[r] + q3[r])).reduce_sum();
        if !out.is_finite() {
            return None;
        }
        outputs[r] = out;
    }
    Some(outputs)
}

/// Steps 8 GDN state rows across `inputs.len()` tokens simultaneously, keeping all 8
/// state rows in registers/stack blocks across tokens and sharing `key` and `query`
/// vector loads across the 8 rows at every token. Returns `false` without mutating
/// `states` or `output` if any non-finite value occurs, allowing the caller to fall
/// back to the sequential per-row path for exact partial-mutation semantics.
#[inline]
pub(super) fn step_eight_rows_tokens_from(
    previous: [&[f32]; 8],
    states: [&mut [f32]; 8],
    inputs: &[super::ChunkInput<'_>],
    row: usize,
    stride: usize,
    scale: f32,
    output: &mut [f32],
) -> bool {
    if states[0].len() <= 128 {
        step_eight_rows_tokens_from_blocks::<8>(
            previous, states, inputs, row, stride, scale, output,
        )
    } else {
        step_eight_rows_tokens_from_blocks::<16>(
            previous, states, inputs, row, stride, scale, output,
        )
    }
}

/// Consumes the parent's complete head proof with one baseline selection.
pub(super) fn try_step_head_tokens_from(
    previous: &[f32],
    state: &mut [f32],
    inputs: &[super::ChunkInput<'_>],
    scale: f32,
    output: &mut [f32],
) -> Option<Result<()>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let width = inputs[0].key.len();
    for (group, (previous, state)) in previous
        .chunks_exact(width * 8)
        .zip(state.chunks_exact_mut(width * 8))
        .enumerate()
    {
        let row = group * 8;
        let previous = std::array::from_fn(|r| &previous[r * width..(r + 1) * width]);
        let (h0, h1) = state.split_at_mut(4 * width);
        let (s01, s23) = h0.split_at_mut(2 * width);
        let (s45, s67) = h1.split_at_mut(2 * width);
        let (s0, s1) = s01.split_at_mut(width);
        let (s2, s3) = s23.split_at_mut(width);
        let (s4, s5) = s45.split_at_mut(width);
        let (s6, s7) = s67.split_at_mut(width);
        let mut states = [s0, s1, s2, s3, s4, s5, s6, s7];
        let stepped = if width == 128 {
            step_eight_rows_tokens_from_selected::<8>(
                neon,
                previous,
                states.each_mut().map(|state| &mut **state),
                inputs,
                row,
                width,
                scale,
                output,
            )
        } else {
            step_eight_rows_tokens_from_selected::<16>(
                neon,
                previous,
                states.each_mut().map(|state| &mut **state),
                inputs,
                row,
                width,
                scale,
                output,
            )
        };
        if !stepped {
            // The failed group published nothing. Retain the defensive
            // row-major replay, including each row's native/scalar selection.
            for (offset, (previous, state)) in previous.into_iter().zip(states).enumerate() {
                if let Err(error) = super::step_row_tokens_from(
                    previous,
                    state,
                    inputs,
                    row + offset,
                    width,
                    scale,
                    output,
                ) {
                    return Some(Err(error));
                }
            }
        }
    }
    Some(Ok(()))
}

#[inline]
fn step_eight_rows_tokens_from_blocks<const BLOCKS: usize>(
    previous: [&[f32]; 8],
    states: [&mut [f32]; 8],
    inputs: &[super::ChunkInput<'_>],
    row: usize,
    stride: usize,
    scale: f32,
    output: &mut [f32],
) -> bool {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return false;
    };
    let width = states[0].len();
    if width == 0 || width > BLOCKS * 16 || !width.is_multiple_of(16) || inputs.len() > 64 {
        return false;
    }
    step_eight_rows_tokens_from_selected::<BLOCKS>(
        neon, previous, states, inputs, row, stride, scale, output,
    )
}

/// Geometry, tile bound and baseline selection are established by either
/// the defensive block entry point or the borrowed complete-head entry point.
#[inline]
#[expect(
    clippy::too_many_arguments,
    reason = "Keep admitted baseline, immutable source and temporal output placement explicit."
)]
fn step_eight_rows_tokens_from_selected<const BLOCKS: usize>(
    neon: Neon,
    previous: [&[f32]; 8],
    states: [&mut [f32]; 8],
    inputs: &[super::ChunkInput<'_>],
    row: usize,
    stride: usize,
    scale: f32,
    output: &mut [f32],
) -> bool {
    let width = states[0].len();
    let blocks = width / 16;
    // Every update remains private until the group commits. Preserve all lane
    // predicates while reducing their combined mask once before publication.
    let mut finite_mask = f32x4::splat(neon, 0.0).is_finite();
    let mut current = [[[f32x4::splat(neon, 0.0); 4]; BLOCKS]; 8];
    for (r, prev) in previous.iter().enumerate() {
        for (b, prev_block) in prev.as_chunks::<16>().0.iter().enumerate() {
            current[r][b] = [
                f32x4::from_slice(neon, &prev_block[0..4]),
                f32x4::from_slice(neon, &prev_block[4..8]),
                f32x4::from_slice(neon, &prev_block[8..12]),
                f32x4::from_slice(neon, &prev_block[12..16]),
            ];
        }
    }
    let mut staged_out = [[0.0_f32; 64]; 8];

    for (token, input) in inputs.iter().enumerate() {
        // Keep all eight key dots ahead of every update. Each arithmetic
        // phase owns only sixteen FMA accumulators, leaving registers for the
        // shared inputs while the eight-row transaction remains private.
        let mut scales = [[0.0; 4]; 2];
        for (group, (work, scales)) in current
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(&mut scales)
            .enumerate()
        {
            let values = std::array::from_fn(|r| input.value[row + group * 4 + r]);
            let Some(values) =
                decay_four_rows_blocks(neon, work, input.key, input.decay, values, input.beta)
            else {
                return false;
            };
            *scales = values;
        }
        for (group, (work, scales)) in current
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(scales)
            .enumerate()
        {
            let Some(values) = update_four_rows_blocks_masked(
                neon,
                work,
                input.key,
                input.query,
                scales,
                &mut finite_mask,
            ) else {
                return false;
            };
            for (r, value) in values.into_iter().enumerate() {
                staged_out[group * 4 + r][token] = value * scale;
            }
        }
    }

    if !finite_mask.all_true() {
        return false;
    }

    for (state, row_blocks) in states.into_iter().zip(current.iter()) {
        flush_blocks(state, row_blocks, blocks);
    }
    for (r, row_staged) in staged_out.iter().enumerate() {
        for (token, &val) in row_staged.iter().enumerate().take(inputs.len()) {
            output[token * stride + row + r] = val;
        }
    }
    true
}

/// Steps 4 GDN state rows in-place across `inputs.len()` tokens simultaneously,
/// keeping all 4 state rows in registers/stack blocks across tokens and sharing
/// `key` and `query` vector loads across the 4 rows at every token. Returns `false`
/// without mutating `states` if any non-finite value occurs.
#[inline]
pub(super) fn step_four_rows_tokens_in_place(
    states: [&mut [f32]; 4],
    inputs: &[super::ChunkInput<'_>],
    row: usize,
) -> bool {
    if states[0].len() <= 128 {
        step_four_rows_tokens_in_place_blocks::<8>(states, inputs, row)
    } else {
        step_four_rows_tokens_in_place_blocks::<16>(states, inputs, row)
    }
}

#[inline]
fn step_four_rows_tokens_in_place_blocks<const BLOCKS: usize>(
    states: [&mut [f32]; 4],
    inputs: &[super::ChunkInput<'_>],
    row: usize,
) -> bool {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return false;
    };
    let width = states[0].len();
    if width == 0 || width > BLOCKS * 16 || !width.is_multiple_of(16) || inputs.len() > 64 {
        return false;
    }
    let blocks = width / 16;
    // Every update remains private until the group commits. Preserve all lane
    // predicates while reducing their combined mask once before publication.
    let mut finite_mask = f32x4::splat(neon, 0.0).is_finite();
    let mut current = [[[f32x4::splat(neon, 0.0); 4]; BLOCKS]; 4];
    for (r, state) in states.iter().enumerate() {
        for (b, state_block) in state.as_chunks::<16>().0.iter().enumerate() {
            current[r][b] = [
                f32x4::from_slice(neon, &state_block[0..4]),
                f32x4::from_slice(neon, &state_block[4..8]),
                f32x4::from_slice(neon, &state_block[8..12]),
                f32x4::from_slice(neon, &state_block[12..16]),
            ];
        }
    }

    for input in inputs {
        let decays = f32x4::splat(neon, input.decay);
        let mut s0 = [f32x4::splat(neon, 0.0); 4];
        let mut s1 = [f32x4::splat(neon, 0.0); 4];
        let mut s2 = [f32x4::splat(neon, 0.0); 4];
        let mut s3 = [f32x4::splat(neon, 0.0); 4];
        for (b, key_block) in input.key.as_chunks::<16>().0.iter().enumerate() {
            let k0 = f32x4::from_slice(neon, &key_block[0..4]);
            let k1 = f32x4::from_slice(neon, &key_block[4..8]);
            let k2 = f32x4::from_slice(neon, &key_block[8..12]);
            let k3 = f32x4::from_slice(neon, &key_block[12..16]);
            for r in 0..4 {
                let [c0, c1, c2, c3] = current[r][b];
                let d0 = c0 * decays;
                let d1 = c1 * decays;
                let d2 = c2 * decays;
                let d3 = c3 * decays;
                current[r][b] = [d0, d1, d2, d3];
                s0[r] = d0.mul_add_precise(k0, s0[r]);
                s1[r] = d1.mul_add_precise(k1, s1[r]);
                s2[r] = d2.mul_add_precise(k2, s2[r]);
                s3[r] = d3.mul_add_precise(k3, s3[r]);
            }
        }
        let mut scales = [f32x4::splat(neon, 0.0); 4];
        for r in 0..4 {
            let dot_k = ((s0[r] + s2[r]) + (s1[r] + s3[r])).reduce_sum();
            if !dot_k.is_finite() {
                return false;
            }
            scales[r] = f32x4::splat(neon, (input.value[row + r] - dot_k) * input.beta);
        }
        let mut q0 = [f32x4::splat(neon, 0.0); 4];
        let mut q1 = [f32x4::splat(neon, 0.0); 4];
        let mut q2 = [f32x4::splat(neon, 0.0); 4];
        let mut q3 = [f32x4::splat(neon, 0.0); 4];
        for (b, (key_block, query_block)) in input
            .key
            .as_chunks::<16>()
            .0
            .iter()
            .zip(input.query.as_chunks::<16>().0)
            .enumerate()
        {
            let k0 = f32x4::from_slice(neon, &key_block[0..4]);
            let k1 = f32x4::from_slice(neon, &key_block[4..8]);
            let k2 = f32x4::from_slice(neon, &key_block[8..12]);
            let k3 = f32x4::from_slice(neon, &key_block[12..16]);
            let qv0 = f32x4::from_slice(neon, &query_block[0..4]);
            let qv1 = f32x4::from_slice(neon, &query_block[4..8]);
            let qv2 = f32x4::from_slice(neon, &query_block[8..12]);
            let qv3 = f32x4::from_slice(neon, &query_block[12..16]);
            for r in 0..4 {
                let [d0, d1, d2, d3] = current[r][b];
                let delta = scales[r];
                let u0 = delta.mul_add_precise(k0, d0);
                let u1 = delta.mul_add_precise(k1, d1);
                let u2 = delta.mul_add_precise(k2, d2);
                let u3 = delta.mul_add_precise(k3, d3);
                finite_mask &= u0.is_finite() & u1.is_finite() & u2.is_finite() & u3.is_finite();
                current[r][b] = [u0, u1, u2, u3];
                q0[r] = u0.mul_add_precise(qv0, q0[r]);
                q1[r] = u1.mul_add_precise(qv1, q1[r]);
                q2[r] = u2.mul_add_precise(qv2, q2[r]);
                q3[r] = u3.mul_add_precise(qv3, q3[r]);
            }
        }
        for r in 0..4 {
            if !((q0[r] + q2[r]) + (q1[r] + q3[r])).reduce_sum().is_finite() {
                return false;
            }
        }
    }

    if !finite_mask.all_true() {
        return false;
    }

    for (state, row_blocks) in states.into_iter().zip(current.iter()) {
        flush_blocks(state, row_blocks, blocks);
    }
    true
}

/// Steps 8 GDN state rows in-place across `inputs.len()` tokens simultaneously,
/// keeping all 8 state rows in registers/stack blocks across tokens and sharing
/// `key` and `query` vector loads across the 8 rows at every token. Returns `false`
/// without mutating `states` if any non-finite value occurs.
#[inline]
pub(super) fn step_eight_rows_tokens_in_place(
    states: [&mut [f32]; 8],
    inputs: &[super::ChunkInput<'_>],
    row: usize,
) -> bool {
    if states[0].len() <= 128 {
        step_eight_rows_tokens_in_place_blocks::<8>(states, inputs, row)
    } else {
        step_eight_rows_tokens_in_place_blocks::<16>(states, inputs, row)
    }
}

#[inline]
fn step_eight_rows_tokens_in_place_blocks<const BLOCKS: usize>(
    states: [&mut [f32]; 8],
    inputs: &[super::ChunkInput<'_>],
    row: usize,
) -> bool {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return false;
    };
    let width = states[0].len();
    if width == 0 || width > BLOCKS * 16 || !width.is_multiple_of(16) || inputs.len() > 64 {
        return false;
    }
    let blocks = width / 16;
    // Every update remains private until the group commits. Preserve all lane
    // predicates while reducing their combined mask once before publication.
    let mut finite_mask = f32x4::splat(neon, 0.0).is_finite();
    let mut current = [[[f32x4::splat(neon, 0.0); 4]; BLOCKS]; 8];
    for (r, state) in states.iter().enumerate() {
        for (b, state_block) in state.as_chunks::<16>().0.iter().enumerate() {
            current[r][b] = [
                f32x4::from_slice(neon, &state_block[0..4]),
                f32x4::from_slice(neon, &state_block[4..8]),
                f32x4::from_slice(neon, &state_block[8..12]),
                f32x4::from_slice(neon, &state_block[12..16]),
            ];
        }
    }

    for input in inputs {
        // Keep all eight key dots ahead of every update. Each arithmetic
        // phase owns only sixteen FMA accumulators, leaving registers for the
        // shared inputs while the eight-row transaction remains private.
        let mut scales = [[0.0; 4]; 2];
        for (group, (work, scales)) in current
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(&mut scales)
            .enumerate()
        {
            let values = std::array::from_fn(|r| input.value[row + group * 4 + r]);
            let Some(values) =
                decay_four_rows_blocks(neon, work, input.key, input.decay, values, input.beta)
            else {
                return false;
            };
            *scales = values;
        }
        for (work, scales) in current.as_chunks_mut::<4>().0.iter_mut().zip(scales) {
            if update_four_rows_blocks_masked(
                neon,
                work,
                input.key,
                input.query,
                scales,
                &mut finite_mask,
            )
            .is_none()
            {
                return false;
            }
        }
    }

    if !finite_mask.all_true() {
        return false;
    }

    for (state, row_blocks) in states.into_iter().zip(current.iter()) {
        flush_blocks(state, row_blocks, blocks);
    }
    true
}

#[inline]
#[expect(
    clippy::suboptimal_flops,
    reason = "Pinned scalar remainder retains separate multiply/add rounding."
)]
fn decay_four_rows_state_only_from(
    neon: Neon,
    previous: [&[f32]; 4],
    states: &mut [&mut [f32]; 4],
    key: &[f32],
    decay: f32,
) -> [f32; 4] {
    let width = key.len();
    let vectorized = width / 16 * 16;
    let decays = f32x4::splat(neon, decay);
    let mut s0 = [f32x4::splat(neon, 0.0); 4];
    let mut s1 = [f32x4::splat(neon, 0.0); 4];
    let mut s2 = [f32x4::splat(neon, 0.0); 4];
    let mut s3 = [f32x4::splat(neon, 0.0); 4];
    for (b, key_block) in key[..vectorized].as_chunks::<16>().0.iter().enumerate() {
        let offset = b * 16;
        let k0 = f32x4::from_slice(neon, &key_block[0..4]);
        let k1 = f32x4::from_slice(neon, &key_block[4..8]);
        let k2 = f32x4::from_slice(neon, &key_block[8..12]);
        let k3 = f32x4::from_slice(neon, &key_block[12..16]);
        for r in 0..4 {
            let prev_block = &previous[r][offset..offset + 16];
            let st_block = &mut states[r][offset..offset + 16];
            let d0 = f32x4::from_slice(neon, &prev_block[0..4]) * decays;
            let d1 = f32x4::from_slice(neon, &prev_block[4..8]) * decays;
            let d2 = f32x4::from_slice(neon, &prev_block[8..12]) * decays;
            let d3 = f32x4::from_slice(neon, &prev_block[12..16]) * decays;
            d0.store_slice(&mut st_block[0..4]);
            d1.store_slice(&mut st_block[4..8]);
            d2.store_slice(&mut st_block[8..12]);
            d3.store_slice(&mut st_block[12..16]);
            s0[r] = d0.mul_add_precise(k0, s0[r]);
            s1[r] = d1.mul_add_precise(k1, s1[r]);
            s2[r] = d2.mul_add_precise(k2, s2[r]);
            s3[r] = d3.mul_add_precise(k3, s3[r]);
        }
    }
    let mut dots: [f32; 4] =
        std::array::from_fn(|r| ((s0[r] + s2[r]) + (s1[r] + s3[r])).reduce_sum());
    for (idx, &k) in key[vectorized..].iter().enumerate() {
        let col = vectorized + idx;
        for r in 0..4 {
            let d = previous[r][col] * decay;
            states[r][col] = d;
            dots[r] += d * k;
        }
    }
    dots
}

#[inline]
#[expect(
    clippy::suboptimal_flops,
    reason = "Pinned scalar remainder retains separate multiply/add rounding."
)]
fn fused_step_four_rows_state_only_next(
    neon: Neon,
    states: &mut [&mut [f32]; 4],
    key_cur: &[f32],
    key_next: &[f32],
    deltas: [f32; 4],
    decay_next: f32,
) -> Option<[f32; 4]> {
    let width = key_cur.len();
    let vectorized = width / 16 * 16;
    let decay_next_vec = f32x4::splat(neon, decay_next);
    let mut finite_mask = f32x4::splat(neon, 0.0).is_finite();
    let mut n0 = [f32x4::splat(neon, 0.0); 4];
    let mut n1 = [f32x4::splat(neon, 0.0); 4];
    let mut n2 = [f32x4::splat(neon, 0.0); 4];
    let mut n3 = [f32x4::splat(neon, 0.0); 4];
    for (b, (key_cur_block, key_next_block)) in key_cur[..vectorized]
        .as_chunks::<16>()
        .0
        .iter()
        .zip(key_next[..vectorized].as_chunks::<16>().0)
        .enumerate()
    {
        let offset = b * 16;
        // Process two 8-float halves per 16-float block to keep live NEON registers <= 26 (zero spills).
        let kc0 = f32x4::from_slice(neon, &key_cur_block[0..4]);
        let kc1 = f32x4::from_slice(neon, &key_cur_block[4..8]);
        let kn0 = f32x4::from_slice(neon, &key_next_block[0..4]);
        let kn1 = f32x4::from_slice(neon, &key_next_block[4..8]);
        for r in 0..4 {
            let st_half = &mut states[r][offset..offset + 8];
            let delta_vec = f32x4::splat(neon, deltas[r]);
            let u0 = delta_vec.mul_add_precise(kc0, f32x4::from_slice(neon, &st_half[0..4]));
            let u1 = delta_vec.mul_add_precise(kc1, f32x4::from_slice(neon, &st_half[4..8]));
            finite_mask &= u0.is_finite() & u1.is_finite();
            let d0 = u0 * decay_next_vec;
            let d1 = u1 * decay_next_vec;
            d0.store_slice(&mut st_half[0..4]);
            d1.store_slice(&mut st_half[4..8]);
            n0[r] = d0.mul_add_precise(kn0, n0[r]);
            n1[r] = d1.mul_add_precise(kn1, n1[r]);
        }
        let kc2 = f32x4::from_slice(neon, &key_cur_block[8..12]);
        let kc3 = f32x4::from_slice(neon, &key_cur_block[12..16]);
        let kn2 = f32x4::from_slice(neon, &key_next_block[8..12]);
        let kn3 = f32x4::from_slice(neon, &key_next_block[12..16]);
        for r in 0..4 {
            let st_half = &mut states[r][offset + 8..offset + 16];
            let delta_vec = f32x4::splat(neon, deltas[r]);
            let u2 = delta_vec.mul_add_precise(kc2, f32x4::from_slice(neon, &st_half[0..4]));
            let u3 = delta_vec.mul_add_precise(kc3, f32x4::from_slice(neon, &st_half[4..8]));
            finite_mask &= u2.is_finite() & u3.is_finite();
            let d2 = u2 * decay_next_vec;
            let d3 = u3 * decay_next_vec;
            d2.store_slice(&mut st_half[0..4]);
            d3.store_slice(&mut st_half[4..8]);
            n2[r] = d2.mul_add_precise(kn2, n2[r]);
            n3[r] = d3.mul_add_precise(kn3, n3[r]);
        }
    }
    if !finite_mask.all_true() {
        return None;
    }
    let mut dots: [f32; 4] =
        std::array::from_fn(|r| ((n0[r] + n2[r]) + (n1[r] + n3[r])).reduce_sum());
    for (idx, (&kc, &kn)) in key_cur[vectorized..]
        .iter()
        .zip(&key_next[vectorized..])
        .enumerate()
    {
        let col = vectorized + idx;
        for r in 0..4 {
            let u = states[r][col] + deltas[r] * kc;
            if !u.is_finite() {
                return None;
            }
            let d = u * decay_next;
            states[r][col] = d;
            dots[r] += d * kn;
        }
    }
    Some(dots)
}

#[inline]
#[expect(
    clippy::suboptimal_flops,
    reason = "Pinned scalar remainder retains separate multiply/add rounding."
)]
fn finish_four_rows_state_only(
    neon: Neon,
    states: &mut [&mut [f32]; 4],
    key_cur: &[f32],
    deltas: [f32; 4],
) -> bool {
    let width = key_cur.len();
    let vectorized = width / 16 * 16;
    let mut finite_mask = f32x4::splat(neon, 0.0).is_finite();
    for (b, key_cur_block) in key_cur[..vectorized].as_chunks::<16>().0.iter().enumerate() {
        let offset = b * 16;
        let k0 = f32x4::from_slice(neon, &key_cur_block[0..4]);
        let k1 = f32x4::from_slice(neon, &key_cur_block[4..8]);
        let k2 = f32x4::from_slice(neon, &key_cur_block[8..12]);
        let k3 = f32x4::from_slice(neon, &key_cur_block[12..16]);
        for r in 0..4 {
            let st_block = &mut states[r][offset..offset + 16];
            let delta_vec = f32x4::splat(neon, deltas[r]);
            let u0 = delta_vec.mul_add_precise(k0, f32x4::from_slice(neon, &st_block[0..4]));
            let u1 = delta_vec.mul_add_precise(k1, f32x4::from_slice(neon, &st_block[4..8]));
            let u2 = delta_vec.mul_add_precise(k2, f32x4::from_slice(neon, &st_block[8..12]));
            let u3 = delta_vec.mul_add_precise(k3, f32x4::from_slice(neon, &st_block[12..16]));
            finite_mask &= u0.is_finite() & u1.is_finite() & u2.is_finite() & u3.is_finite();
            u0.store_slice(&mut st_block[0..4]);
            u1.store_slice(&mut st_block[4..8]);
            u2.store_slice(&mut st_block[8..12]);
            u3.store_slice(&mut st_block[12..16]);
        }
    }
    if !finite_mask.all_true() {
        return false;
    }
    for (idx, &kc) in key_cur[vectorized..].iter().enumerate() {
        let col = vectorized + idx;
        for r in 0..4 {
            let u = states[r][col] + deltas[r] * kc;
            if !u.is_finite() {
                return false;
            }
            states[r][col] = u;
        }
    }
    true
}

/// Steps 4 GDN state rows simultaneously from `previous` into `states` across `inputs.len()`
/// tokens without computing query outputs, fusing token `t`'s update with token `t + 1`'s
/// decay-and-key-dot into one read-modify-write pass over `states`. Returns `false` if NEON
/// is unavailable or any non-finite value occurs, allowing exact replay from `previous`.
#[inline]
pub(super) fn step_four_rows_state_only_tokens_from(
    previous: [&[f32]; 4],
    mut states: [&mut [f32]; 4],
    inputs: &[super::StateStepInput<'_>],
    row: usize,
) -> bool {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return false;
    };
    let Some(first) = inputs.first() else {
        return false;
    };
    let width = first.key.len();
    if width == 0
        || previous.iter().any(|prev| prev.len() != width)
        || states.iter().any(|state| state.len() != width)
        || inputs
            .iter()
            .any(|input| input.key.len() != width || input.value.len() < row.saturating_add(4))
    {
        return false;
    }
    let mut dot_k =
        decay_four_rows_state_only_from(neon, previous, &mut states, first.key, first.decay);
    for (token, input) in inputs.iter().enumerate() {
        let mut deltas = [0.0_f32; 4];
        for (r, slot) in deltas.iter_mut().enumerate() {
            if !dot_k[r].is_finite() {
                return false;
            }
            let delta = (input.value[row + r] - dot_k[r]) * input.beta;
            if !delta.is_finite() {
                return false;
            }
            *slot = delta;
        }
        if let Some(next_input) = inputs.get(token + 1) {
            let Some(next_dots) = fused_step_four_rows_state_only_next(
                neon,
                &mut states,
                input.key,
                next_input.key,
                deltas,
                next_input.decay,
            ) else {
                return false;
            };
            dot_k = next_dots;
        } else {
            return finish_four_rows_state_only(neon, &mut states, input.key, deltas);
        }
    }
    true
}

/// Token-major 16-column WY state update across eight rows, preserving the exact
/// separate binary32 multiply and add rounding order (`0 ULP` with scalar
/// `scale_slice_into` + `add_scaled_slice_in_place`).
#[inline]
#[expect(
    clippy::suboptimal_flops,
    reason = "Preserve separate binary32 multiply and add roundings matching scale_slice_into and add_scaled_slice_in_place."
)]
pub(super) fn gdn_wy_update_eight_rows_token_major(
    states: [&[f32]; 8],
    last_cumulative: f32,
    inputs: &[super::ChunkInput<'_>],
    scaled_deltas: &[[f32; 8]; 32],
    matrix_rows: [&mut [f32]; 8],
) -> bool {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return false;
    };
    let width = matrix_rows[0].len();
    let count = inputs.len();
    let scale_vec = f32x4::splat(neon, last_cumulative);
    let vec16_blocks = width / 16;
    let [m0, m1, m2, m3, m4, m5, m6, m7] = matrix_rows;
    let mut rows = [m0, m1, m2, m3, m4, m5, m6, m7];
    for b in 0..vec16_blocks {
        let offset = b * 16;
        for half in 0..2 {
            let base_r = half * 4;
            let mut acc = [[f32x4::splat(neon, 0.0); 4]; 4];
            for r in 0..4 {
                let s = &states[base_r + r][offset..offset + 16];
                acc[r] = [
                    scale_vec * f32x4::from_slice(neon, &s[0..4]),
                    scale_vec * f32x4::from_slice(neon, &s[4..8]),
                    scale_vec * f32x4::from_slice(neon, &s[8..12]),
                    scale_vec * f32x4::from_slice(neon, &s[12..16]),
                ];
            }
            for (prior, input) in inputs[..count].iter().enumerate() {
                let kb = &input.key[offset..offset + 16];
                let k0 = f32x4::from_slice(neon, &kb[0..4]);
                let k1 = f32x4::from_slice(neon, &kb[4..8]);
                let k2 = f32x4::from_slice(neon, &kb[8..12]);
                let k3 = f32x4::from_slice(neon, &kb[12..16]);
                let sd_row = &scaled_deltas[prior];
                for r in 0..4 {
                    let sd = f32x4::splat(neon, sd_row[base_r + r]);
                    acc[r][0] += sd * k0;
                    acc[r][1] += sd * k1;
                    acc[r][2] += sd * k2;
                    acc[r][3] += sd * k3;
                }
            }
            for (out_row, row_acc) in rows[base_r..base_r + 4].iter_mut().zip(acc.iter()) {
                let out_block = &mut out_row[offset..offset + 16];
                row_acc[0].store_slice(&mut out_block[0..4]);
                row_acc[1].store_slice(&mut out_block[4..8]);
                row_acc[2].store_slice(&mut out_block[8..12]);
                row_acc[3].store_slice(&mut out_block[12..16]);
            }
        }
    }
    let vec4_blocks = width / 4;
    for b in (vec16_blocks * 4)..vec4_blocks {
        let offset = b * 4;
        let mut a: [f32x4<Neon>; 8] = std::array::from_fn(|r| {
            scale_vec * f32x4::from_slice(neon, &states[r][offset..offset + 4])
        });
        for (prior, input) in inputs[..count].iter().enumerate() {
            let k = f32x4::from_slice(neon, &input.key[offset..offset + 4]);
            let sd_row = &scaled_deltas[prior];
            for r in 0..8 {
                a[r] += f32x4::splat(neon, sd_row[r]) * k;
            }
        }
        for (r, out_row) in rows.iter_mut().enumerate() {
            a[r].store_slice(&mut out_row[offset..offset + 4]);
        }
    }
    let rem_start = vec4_blocks * 4;
    if rem_start < width {
        for (r, out_row) in rows.iter_mut().enumerate() {
            for (idx, out_elem) in out_row.iter_mut().enumerate().take(width).skip(rem_start) {
                let mut acc = last_cumulative * states[r][idx];
                for (prior, input) in inputs[..count].iter().enumerate() {
                    acc += scaled_deltas[prior][r] * input.key[idx];
                }
                *out_elem = acc;
            }
        }
    }
    true
}
