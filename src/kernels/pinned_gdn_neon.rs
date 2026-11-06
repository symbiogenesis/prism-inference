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
    let mut sums = [f32x4::splat(neon, 0.0); 4];
    for (first, second) in first[..vectorized]
        .as_chunks::<16>()
        .0
        .iter()
        .zip(second[..vectorized].as_chunks::<16>().0)
    {
        for ((sum, first), second) in sums
            .iter_mut()
            .zip(first.as_chunks::<4>().0)
            .zip(second.as_chunks::<4>().0)
        {
            *sum = f32x4::from_slice(neon, first)
                .mul_add_precise(f32x4::from_slice(neon, second), *sum);
        }
    }
    let mut result = ((sums[0] + sums[2]) + (sums[1] + sums[3])).reduce_sum();
    for (&first, &second) in first[vectorized..].iter().zip(&second[vectorized..]) {
        result += first * second;
    }
    Some(result)
}

/// Scales `state` by `decay` and accumulates its four-register FMA dot product with `key` in one pass.
#[cfg(test)]
#[inline]
#[allow(
    clippy::suboptimal_flops,
    reason = "Pinned scalar remainder retains separate multiply/add rounding."
)]
pub(super) fn decay_and_dot(state: &mut [f32], key: &[f32], decay: f32) -> Option<f32> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let vectorized = state.len() / 16 * 16;
    let decays = f32x4::splat(neon, decay);
    let mut sums = [f32x4::splat(neon, 0.0); 4];
    for (state, key) in state[..vectorized]
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(key[..vectorized].as_chunks::<16>().0)
    {
        for ((sum, state), key) in sums
            .iter_mut()
            .zip(state.as_chunks_mut::<4>().0)
            .zip(key.as_chunks::<4>().0)
        {
            let decayed = f32x4::from_slice(neon, state) * decays;
            decayed.store_slice(state);
            *sum = decayed.mul_add_precise(f32x4::from_slice(neon, key), *sum);
        }
    }
    let mut result = ((sums[0] + sums[2]) + (sums[1] + sums[3])).reduce_sum();
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
    let mut sums = [f32x4::splat(neon, 0.0); 4];
    for ((prev, state), key) in previous[..vectorized]
        .as_chunks::<16>()
        .0
        .iter()
        .zip(state[..vectorized].as_chunks_mut::<16>().0)
        .zip(key[..vectorized].as_chunks::<16>().0)
    {
        for (((sum, prev), state), key) in sums
            .iter_mut()
            .zip(prev.as_chunks::<4>().0)
            .zip(state.as_chunks_mut::<4>().0)
            .zip(key.as_chunks::<4>().0)
        {
            let decayed = f32x4::from_slice(neon, prev) * decays;
            decayed.store_slice(state);
            *sum = decayed.mul_add_precise(f32x4::from_slice(neon, key), *sum);
        }
    }
    let mut result = ((sums[0] + sums[2]) + (sums[1] + sums[3])).reduce_sum();
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
    for (target, input) in target[..vectorized]
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(input[..vectorized].as_chunks::<4>().0)
    {
        let updated = scales.mul_add_precise(
            f32x4::from_slice(neon, input),
            f32x4::from_slice(neon, target),
        );
        if updated.is_finite().all_true() {
            updated.store_slice(target);
        } else {
            for (target, &input) in target.iter_mut().zip(input) {
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
    let mut sums = [f32x4::splat(neon, 0.0); 4];
    for ((target_block, key_block), query_block) in target[..vectorized]
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(key[..vectorized].as_chunks::<16>().0)
        .zip(query[..vectorized].as_chunks::<16>().0)
    {
        for (((sum, target_chunk), key_chunk), query_chunk) in sums
            .iter_mut()
            .zip(target_block.as_chunks_mut::<4>().0)
            .zip(key_block.as_chunks::<4>().0)
            .zip(query_block.as_chunks::<4>().0)
        {
            let updated = scales.mul_add_precise(
                f32x4::from_slice(neon, key_chunk),
                f32x4::from_slice(neon, target_chunk),
            );
            if updated.is_finite().all_true() {
                updated.store_slice(target_chunk);
                *sum = updated.mul_add_precise(f32x4::from_slice(neon, query_chunk), *sum);
            } else {
                for (target_elem, &key_elem) in target_chunk.iter_mut().zip(key_chunk) {
                    *target_elem = super::super::finite(scale.mul_add(key_elem, *target_elem))?;
                }
            }
        }
    }
    let mut result = ((sums[0] + sums[2]) + (sums[1] + sums[3])).reduce_sum();
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
