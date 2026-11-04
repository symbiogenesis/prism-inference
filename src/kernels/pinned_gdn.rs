//! Exact pinned Windows ARM64 GDN arithmetic with a safe baseline NEON path.
//!
//! Reference: Prism `73bfe6cc485554822f80c473a3c0c4f25e80f41b`,
//! `ggml/src/ggml-cpu/vec.cpp`, `vec.h` and `simd-mappings.h`.
//! Upstream MIT terms are retained in `PRISM-LICENSE.txt`.

#![allow(
    clippy::redundant_pub_crate,
    reason = "Keep graph-only GDN helpers crate scoped while satisfying unreachable_pub."
)]

use crate::{Result, invalid};

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
#[path = "pinned_gdn_neon.rs"]
mod neon;

/// Matches the pinned NEON F32 dot, including its scalar remainder.
///
/// # Errors
/// Rejects unequal vector lengths or a nonfinite result.
pub(crate) fn dot(first: &[f32], second: &[f32]) -> Result<f32> {
    if first.len() != second.len() {
        return Err(invalid("pinned GDN dot requires matching vectors"));
    }
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(result) = neon::dot(first, second) {
        return super::finite(result);
    }
    dot_scalar(first, second)
}

// Floating matrix projections deliberately retain scalar diagnostics and their
// explicit scalar policy. Only the graph's pinned GDN helper selects NEON.
#[allow(
    clippy::suboptimal_flops,
    reason = "Keep the scalar oracle's exact multiply/add remainder rounding."
)]
pub(super) fn dot_scalar(first: &[f32], second: &[f32]) -> Result<f32> {
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
pub(crate) fn add_scaled(target: &mut [f32], input: &[f32], scale: f32) -> Result<()> {
    if target.len() != input.len() {
        return Err(invalid("pinned GDN update requires matching vectors"));
    }
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(result) = neon::add_scaled(target, input, scale) {
        return result;
    }
    scalar_add_scaled(target, input, scale)
}

#[allow(
    clippy::suboptimal_flops,
    reason = "Keep the scalar oracle's exact multiply/add remainder rounding."
)]
fn scalar_add_scaled(target: &mut [f32], input: &[f32], scale: f32) -> Result<()> {
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
    use super::{add_scaled, dot, dot_scalar, scalar_add_scaled};
    use crate::Result;

    fn vectors(width: usize) -> (Vec<f32>, Vec<f32>) {
        let mut seed = 0x2468_ace1_u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            // Mixed signs and mantissas with a fixed finite exponent. These
            // distinguish the four-register tree from a scalar running sum.
            f32::from_bits((seed & 0x807f_ffff) | 0x3d80_0000)
        };
        (
            (0..width).map(|_| next()).collect(),
            (0..width).map(|_| next()).collect(),
        )
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    #[test]
    fn pinned_gdn_dispatch_matches_scalar_bits_and_tail() -> Result<()> {
        for width in (0..=33).chain([63, 64, 65, 127, 128, 129, 255, 256, 257, 5120]) {
            let (first, second) = vectors(width);
            assert_eq!(dot(&first, &second)?.to_bits(), dot_scalar(&first, &second)?.to_bits());
            for scale in [0.0, -0.0, 0.375, -1.125, f32::from_bits(1)] {
                let mut expected = first.clone();
                let mut actual = first.clone();
                scalar_add_scaled(&mut expected, &second, scale)?;
                add_scaled(&mut actual, &second, scale)?;
                assert_eq!(bits(&actual), bits(&expected), "width {width}, scale {scale}");
            }
        }
        Ok(())
    }

    #[test]
    #[allow(
        clippy::suboptimal_flops,
        reason = "The regression must retain the unfused scalar remainder oracle."
    )]
    fn pinned_gdn_retains_fused_vector_and_nonfused_tail_rounding() -> Result<()> {
        let scale = f32::from_bits(0x3f80_0001);
        let input = f32::from_bits(0x3f7f_fffe);
        let fused = scale.mul_add(input, -1.0);
        let nonfused = scale * input - 1.0;
        assert_ne!(fused.to_bits(), nonfused.to_bits());
        for width in [15, 16, 17, 31, 32, 33, 128, 129] {
            let mut actual = vec![-1.0; width];
            add_scaled(&mut actual, &vec![input; width], scale)?;
            let vectorized = width / 16 * 16;
            assert!(actual[..vectorized].iter().all(|value| value.to_bits() == fused.to_bits()));
            assert!(actual[vectorized..].iter().all(|value| value.to_bits() == nonfused.to_bits()));
        }
        Ok(())
    }

    #[test]
    fn pinned_gdn_preserves_exceptional_bits_and_first_failure_mutation() -> Result<()> {
        let values = [
            0.0, -0.0, f32::from_bits(1), -f32::from_bits(1),
            f32::MIN_POSITIVE, -f32::MIN_POSITIVE, 1.0, -1.0,
        ];
        for width in [0, 1, 15, 16, 17, 31, 32, 33, 128, 129] {
            let first = (0..width).map(|index| values[index % values.len()]).collect::<Vec<_>>();
            let second = (0..width).map(|index| values[(index + 3) % values.len()]).collect::<Vec<_>>();
            assert_eq!(dot(&first, &second)?.to_bits(), dot_scalar(&first, &second)?.to_bits());
            for scale in [0.0, -0.0, 0.5, -1.0] {
                let mut actual = first.clone();
                let mut expected = first.clone();
                add_scaled(&mut actual, &second, scale)?;
                scalar_add_scaled(&mut expected, &second, scale)?;
                assert_eq!(bits(&actual), bits(&expected));
            }
        }
        for width in [17, 33] {
            for failing in 0..width {
                for invalid in [f32::MAX, f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
                    let mut input = vec![0.25; width];
                    input[failing] = invalid;
                    let mut actual = vec![1.0; width];
                    let mut expected = actual.clone();
                    assert!(scalar_add_scaled(&mut expected, &input, 2.0).is_err());
                    assert!(add_scaled(&mut actual, &input, 2.0).is_err());
                    assert_eq!(bits(&actual), bits(&expected), "width {width}, failing lane {failing}");
                    assert_eq!(actual[failing].to_bits(), 1.0_f32.to_bits());
                    assert!(actual[failing..].iter().all(|value| value.to_bits() == 1.0_f32.to_bits()));
                }
            }
        }
        for nonfinite in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            let mut actual = [1.0; 17];
            assert!(add_scaled(&mut actual, &[0.0; 17], nonfinite).is_err());
            assert_eq!(actual.map(f32::to_bits), [1.0_f32.to_bits(); 17]);
            let mut input = [1.0; 17];
            input[3] = nonfinite;
            assert!(dot(&input, &[1.0; 17]).is_err());
        }
        Ok(())
    }

    #[test]
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    fn pinned_gdn_neon_executes_exact_vector_and_tail_contract() -> Result<()> {
        let (first, second) = vectors(129);
        let actual = super::neon::dot(&first, &second);
        assert!(actual.is_some(), "qualified baseline must provide a NEON token");
        assert_eq!(actual.map(f32::to_bits), Some(dot_scalar(&first, &second)?.to_bits()));
        let mut expected = first.clone();
        let mut actual = first;
        scalar_add_scaled(&mut expected, &second, -0.375)?;
        let result = super::neon::add_scaled(&mut actual, &second, -0.375);
        assert!(result.is_some_and(|result| result.is_ok()));
        assert_eq!(bits(&actual), bits(&expected));
        Ok(())
    }

    #[test]
    fn pinned_gdn_rejects_mismatched_rows_and_nonfinite_results() {
        let mut target = [1.0, 2.0];
        assert!(add_scaled(&mut target, &[3.0], 4.0).is_err());
        assert_eq!(
            target.map(f32::to_bits),
            [1.0_f32.to_bits(), 2.0_f32.to_bits()]
        );
        assert!(dot(&[1.0], &[]).is_err());
        assert!(dot_scalar(&[1.0], &[]).is_err());
        assert!(dot(&[f32::MAX], &[2.0]).is_err());
        assert!(dot(&[f32::MAX; 16], &[2.0; 16]).is_err());
        assert!(add_scaled(&mut target, &[f32::MAX, 1.0], 2.0).is_err());
    }
}
