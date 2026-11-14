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

/// Computes `(dot(states[r], first), dot(states[r], second))` across four state rows in one pass,
/// sharing each `first` and `second` vector load across all four rows' four-register FMA trees.
#[inline]
#[expect(
    clippy::suboptimal_flops,
    reason = "Pinned scalar remainder retains separate multiply/add rounding."
)]
pub(super) fn dot_pair_four_rows(
    states: [&[f32]; 4],
    first: &[f32],
    second: &[f32],
) -> Option<[(f32, f32); 4]> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let vectorized = first.len() / 16 * 16;
    let mut f = [[f32x4::splat(neon, 0.0); 4]; 4];
    let mut s = [[f32x4::splat(neon, 0.0); 4]; 4];
    let st_chunks = states.map(|row| row[..vectorized].as_chunks::<16>().0);
    for (b, (first_block, second_block)) in first[..vectorized]
        .as_chunks::<16>()
        .0
        .iter()
        .zip(second[..vectorized].as_chunks::<16>().0)
        .enumerate()
    {
        let k0 = f32x4::from_slice(neon, &first_block[0..4]);
        let k1 = f32x4::from_slice(neon, &first_block[4..8]);
        let k2 = f32x4::from_slice(neon, &first_block[8..12]);
        let k3 = f32x4::from_slice(neon, &first_block[12..16]);
        let q0 = f32x4::from_slice(neon, &second_block[0..4]);
        let q1 = f32x4::from_slice(neon, &second_block[4..8]);
        let q2 = f32x4::from_slice(neon, &second_block[8..12]);
        let q3 = f32x4::from_slice(neon, &second_block[12..16]);
        for r in 0..4 {
            let st_block = &st_chunks[r][b];
            let st0 = f32x4::from_slice(neon, &st_block[0..4]);
            let st1 = f32x4::from_slice(neon, &st_block[4..8]);
            let st2 = f32x4::from_slice(neon, &st_block[8..12]);
            let st3 = f32x4::from_slice(neon, &st_block[12..16]);
            f[r][0] = st0.mul_add_precise(k0, f[r][0]);
            f[r][1] = st1.mul_add_precise(k1, f[r][1]);
            f[r][2] = st2.mul_add_precise(k2, f[r][2]);
            f[r][3] = st3.mul_add_precise(k3, f[r][3]);
            s[r][0] = st0.mul_add_precise(q0, s[r][0]);
            s[r][1] = st1.mul_add_precise(q1, s[r][1]);
            s[r][2] = st2.mul_add_precise(q2, s[r][2]);
            s[r][3] = st3.mul_add_precise(q3, s[r][3]);
        }
    }
    let mut results: [(f32, f32); 4] = std::array::from_fn(|r| {
        (
            ((f[r][0] + f[r][2]) + (f[r][1] + f[r][3])).reduce_sum(),
            ((s[r][0] + s[r][2]) + (s[r][1] + s[r][3])).reduce_sum(),
        )
    });
    let rem_base = first[vectorized..].iter().zip(&second[vectorized..]);
    for (idx, (&k, &q)) in rem_base.enumerate() {
        let col = vectorized + idx;
        for (r, res) in results.iter_mut().enumerate() {
            let st = states[r][col];
            res.0 += st * k;
            res.1 += st * q;
        }
    }
    Some(results)
}

/// Computes `(dot(states[r], first), dot(states[r], second))` across eight state rows in one pass,
/// sharing each `first` and `second` vector load across all eight rows' four-register FMA trees.
#[inline]
#[expect(
    clippy::suboptimal_flops,
    reason = "Pinned scalar remainder retains separate multiply/add rounding."
)]
pub(super) fn dot_pair_eight_rows(
    states: [&[f32]; 8],
    first: &[f32],
    second: &[f32],
) -> Option<[(f32, f32); 8]> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let vectorized = first.len() / 16 * 16;
    let mut f = [[f32x4::splat(neon, 0.0); 4]; 8];
    let mut s = [[f32x4::splat(neon, 0.0); 4]; 8];
    let st_chunks = states.map(|row| row[..vectorized].as_chunks::<16>().0);
    for (b, (first_block, second_block)) in first[..vectorized]
        .as_chunks::<16>()
        .0
        .iter()
        .zip(second[..vectorized].as_chunks::<16>().0)
        .enumerate()
    {
        let k0 = f32x4::from_slice(neon, &first_block[0..4]);
        let k1 = f32x4::from_slice(neon, &first_block[4..8]);
        let k2 = f32x4::from_slice(neon, &first_block[8..12]);
        let k3 = f32x4::from_slice(neon, &first_block[12..16]);
        let q0 = f32x4::from_slice(neon, &second_block[0..4]);
        let q1 = f32x4::from_slice(neon, &second_block[4..8]);
        let q2 = f32x4::from_slice(neon, &second_block[8..12]);
        let q3 = f32x4::from_slice(neon, &second_block[12..16]);
        for r in 0..8 {
            let st_block = &st_chunks[r][b];
            let st0 = f32x4::from_slice(neon, &st_block[0..4]);
            let st1 = f32x4::from_slice(neon, &st_block[4..8]);
            let st2 = f32x4::from_slice(neon, &st_block[8..12]);
            let st3 = f32x4::from_slice(neon, &st_block[12..16]);
            f[r][0] = st0.mul_add_precise(k0, f[r][0]);
            f[r][1] = st1.mul_add_precise(k1, f[r][1]);
            f[r][2] = st2.mul_add_precise(k2, f[r][2]);
            f[r][3] = st3.mul_add_precise(k3, f[r][3]);
            s[r][0] = st0.mul_add_precise(q0, s[r][0]);
            s[r][1] = st1.mul_add_precise(q1, s[r][1]);
            s[r][2] = st2.mul_add_precise(q2, s[r][2]);
            s[r][3] = st3.mul_add_precise(q3, s[r][3]);
        }
    }
    let mut results: [(f32, f32); 8] = std::array::from_fn(|r| {
        (
            ((f[r][0] + f[r][2]) + (f[r][1] + f[r][3])).reduce_sum(),
            ((s[r][0] + s[r][2]) + (s[r][1] + s[r][3])).reduce_sum(),
        )
    });
    for (idx, (&k, &q)) in first[vectorized..]
        .iter()
        .zip(&second[vectorized..])
        .enumerate()
    {
        let col = vectorized + idx;
        for (r, res) in results.iter_mut().enumerate() {
            let st = states[r][col];
            res.0 += st * k;
            res.1 += st * q;
        }
    }
    Some(results)
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
#[cfg(test)]
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

#[cfg(test)]
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
    for (group, ((previous, states), values)) in previous
        .as_chunks::<4>()
        .0
        .iter()
        .zip(states.as_chunks_mut::<4>().0)
        .zip(values.as_chunks::<4>().0)
        .enumerate()
    {
        let deltas = decay_four_rows_into(neon, *previous, states, key, decay, *values, beta)?;
        scales[group * 4..group * 4 + 4].copy_from_slice(&deltas);
    }
    // Every key dot precedes updates, as in the existing eight-row helper.
    let mut outputs = [0.0; 8];
    for (group, (states, scales)) in states
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(scales.as_chunks::<4>().0)
        .enumerate()
    {
        let values = update_four_rows_in_place(neon, states, key, query, *scales)?;
        outputs[group * 4..group * 4 + 4].copy_from_slice(&values);
    }
    Some(outputs)
}

fn decay_four_rows_into(
    neon: Neon,
    previous: [&[f32]; 4],
    states: &mut [&mut [f32]; 4],
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
            let state = &mut states[r][offset..offset + 16];
            d0.store_slice(&mut state[0..4]);
            d1.store_slice(&mut state[4..8]);
            d2.store_slice(&mut state[8..12]);
            d3.store_slice(&mut state[12..16]);
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

fn update_four_rows_in_place(
    neon: Neon,
    states: &mut [&mut [f32]; 4],
    key: &[f32],
    query: &[f32],
    scales: [f32; 4],
) -> Option<[f32; 4]> {
    let scales = scales.map(|scale| f32x4::splat(neon, scale));
    let mut finite_mask = f32x4::splat(neon, 0.0).is_finite();
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
            let state = &mut states[r][offset..offset + 16];
            let scale = scales[r];
            let u0 = scale.mul_add_precise(k0, f32x4::from_slice(neon, &state[0..4]));
            let u1 = scale.mul_add_precise(k1, f32x4::from_slice(neon, &state[4..8]));
            let u2 = scale.mul_add_precise(k2, f32x4::from_slice(neon, &state[8..12]));
            let u3 = scale.mul_add_precise(k3, f32x4::from_slice(neon, &state[12..16]));
            finite_mask &= u0.is_finite() & u1.is_finite() & u2.is_finite() & u3.is_finite();
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
        let out = ((q0[r] + q2[r]) + (q1[r] + q3[r])).reduce_sum();
        if !out.is_finite() {
            return None;
        }
        outputs[r] = out;
    }
    finite_mask.all_true().then_some(outputs)
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
