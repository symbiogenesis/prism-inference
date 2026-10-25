//! Safe scalar reproduction of pinned Windows ARM64 GDN vector arithmetic.
//!
//! Reference: Prism `73bfe6cc485554822f80c473a3c0c4f25e80f41b`,
//! `ggml/src/ggml-cpu/vec.cpp`, `vec.h` and `simd-mappings.h`.
//! Upstream MIT terms are retained in `PRISM-LICENSE.txt`.

#![allow(
    clippy::redundant_pub_crate,
    reason = "Keep graph-only GDN helpers crate scoped while satisfying unreachable_pub."
)]

use crate::{Result, invalid};

/// Matches the pinned NEON F32 dot, including its scalar remainder.
///
/// # Errors
/// Rejects unequal vector lengths or a nonfinite result.
#[allow(
    clippy::suboptimal_flops,
    reason = "The pinned reduction and scalar remainder use separate multiply/add rounding."
)]
pub(crate) fn dot(first: &[f32], second: &[f32]) -> Result<f32> {
    if first.len() != second.len() {
        return Err(invalid("pinned GDN dot requires matching vectors"));
    }
    let vectorized = first.len() / 16 * 16;
    let mut sums = [[0.0_f32; 4]; 4];
    for offset in (0..vectorized).step_by(16) {
        for (register, lanes) in sums.iter_mut().enumerate() {
            for (lane, sum) in lanes.iter_mut().enumerate() {
                let index = offset + register * 4 + lane;
                *sum = first[index].mul_add(second[index], *sum);
            }
        }
    }
    // GGML_F32x4_REDUCE first combines registers 0+2 and 1+3. The
    // horizontal vaddvq_f32 then adds adjacent pairs of the resulting lanes.
    let mut lanes = [0.0; 4];
    for (index, lane) in lanes.iter_mut().enumerate() {
        *lane = (sums[0][index] + sums[2][index]) + (sums[1][index] + sums[3][index]);
    }
    let mut result = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    for (&first, &second) in first[vectorized..].iter().zip(&second[vectorized..]) {
        result += first * second;
    }
    super::finite(result)
}

/// Matches the pinned NEON scaled update, including its scalar remainder.
///
/// # Errors
/// Rejects unequal lengths before mutation, or a nonfinite updated element.
#[allow(
    clippy::suboptimal_flops,
    reason = "The pinned scalar remainder retains separate multiply/add rounding."
)]
pub(crate) fn add_scaled(target: &mut [f32], input: &[f32], scale: f32) -> Result<()> {
    if target.len() != input.len() {
        return Err(invalid("pinned GDN update requires matching vectors"));
    }
    let vectorized = target.len() / 16 * 16;
    for (index, (target, &input)) in target.iter_mut().zip(input).enumerate() {
        *target = super::finite(if index < vectorized {
            scale.mul_add(input, *target)
        } else {
            *target + scale * input
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{add_scaled, dot};

    #[test]
    fn pinned_gdn_rejects_mismatched_rows_and_nonfinite_results() {
        let mut target = [1.0, 2.0];
        assert!(add_scaled(&mut target, &[3.0], 4.0).is_err());
        assert_eq!(
            target.map(f32::to_bits),
            [1.0_f32.to_bits(), 2.0_f32.to_bits()]
        );
        assert!(dot(&[1.0], &[]).is_err());
        assert!(dot(&[f32::MAX], &[2.0]).is_err());
        assert!(add_scaled(&mut target, &[f32::MAX, 1.0], 2.0).is_err());
    }
}
