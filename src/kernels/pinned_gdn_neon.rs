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

/// The parent checks lengths; every safe load/store receives exactly four lanes.
/// A nonfinite vector is replayed in scalar lane order before any vector store,
/// preserving the original finite-prefix mutation and untouched failing lane.
#[inline]
pub(super) fn add_scaled(target: &mut [f32], input: &[f32], scale: f32) -> Option<Result<()>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    Some(add_scaled_with_neon(neon, target, input, scale))
}

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
