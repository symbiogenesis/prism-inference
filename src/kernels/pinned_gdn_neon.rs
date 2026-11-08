//! Safe baseline NEON for the exact pinned GDN tree and checked row update.

use fearless_simd::{Neon, f32x4, prelude::*};

use crate::Result;

/// Geometry is checked by the parent before any load. The fixed four-register
/// tree is the scalar reference's tree, including adjacent horizontal pairs.
#[inline]
#[allow(
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
#[allow(
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

/// Scales `state` by `decay` and accumulates its four-register FMA dot product with `key` in one pass.
#[inline]
#[allow(
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
#[allow(
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
#[allow(
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
#[allow(
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
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::suboptimal_flops,
    reason = "Keep two-pass fused NEON GDN row step and exact partial-mutation fallback together."
)]
pub(super) fn step_row_from(
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
    if width == 0 || width > 256 || !width.is_multiple_of(16) {
        return None;
    }
    let blocks = width / 16;
    let decays = f32x4::splat(neon, decay);
    let mut decayed = [[f32x4::splat(neon, 0.0); 4]; 16];
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
#[allow(
    clippy::too_many_lines,
    clippy::suboptimal_flops,
    reason = "Keep two-pass fused in-place NEON GDN row step and exact partial-mutation fallback together."
)]
pub(super) fn step_row_in_place(
    state: &mut [f32],
    key: &[f32],
    query: &[f32],
    decay: f32,
    value: f32,
    beta: f32,
) -> Option<Result<f32>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let width = state.len();
    if width == 0 || width > 256 || !width.is_multiple_of(16) {
        return None;
    }
    let blocks = width / 16;
    let decays = f32x4::splat(neon, decay);
    let mut decayed = [[f32x4::splat(neon, 0.0); 4]; 16];
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
