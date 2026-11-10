//! Exact candidate-lane pruning shared by bounded heaps and greedy selection.

// Scalar fallback visits every lane once, as before. Eligible NEON paths retain
// the comparison mask instead of rescanning all sixteen lanes after one hit.
// A heap root / greedy maximum never decreases as tokens are visited, so a lane
// excluded at block entry cannot become eligible later in the same block.
const ALL_LANES: u64 = (1 << 16) - 1;

#[inline]
#[cfg_attr(
    not(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    )),
    allow(
        clippy::missing_const_for_fn,
        reason = "Portable signature matches the dynamically selected NEON implementation."
    )
)]
pub(crate) fn candidate_lanes<const VALIDATE: bool>(values: &[f32; 16], threshold: f32) -> u64 {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(lanes) = neon_candidate_lanes::<VALIDATE>(values, threshold) {
        return lanes;
    }
    let _ = (values, threshold);
    ALL_LANES
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
#[inline]
fn neon_candidate_lanes<const VALIDATE: bool>(values: &[f32; 16], threshold: f32) -> Option<u64> {
    use fearless_simd::{f32x4, prelude::*};

    let neon = fearless_simd::Level::baseline().as_neon()?;
    let v0 = f32x4::from_slice(neon, &values[0..4]);
    let v1 = f32x4::from_slice(neon, &values[4..8]);
    let v2 = f32x4::from_slice(neon, &values[8..12]);
    let v3 = f32x4::from_slice(neon, &values[12..16]);
    if VALIDATE && !(v0.is_finite() & v1.is_finite() & v2.is_finite() & v3.is_finite()).all_true() {
        // Visit in original token order so the caller retains its original
        // first-error, scratch-clear and RNG semantics, including discarded NaNs.
        return Some(ALL_LANES);
    }
    let threshold = f32x4::splat(neon, threshold);
    let m0 = v0.simd_gt(threshold);
    let m1 = v1.simd_gt(threshold);
    let m2 = v2.simd_gt(threshold);
    let m3 = v3.simd_gt(threshold);
    if !(m0 | m1 | m2 | m3).any_true() {
        return Some(0);
    }
    Some(
        m0.to_bitmask() | (m1.to_bitmask() << 4) | (m2.to_bitmask() << 8) | (m3.to_bitmask() << 12),
    )
}

#[cfg(test)]
mod tests {
    use super::{ALL_LANES, candidate_lanes};
    use crate::{Result, runtime, sampling::FiniteLogits};

    #[cfg_attr(
        not(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc",
            target_feature = "neon"
        )),
        allow(
            clippy::missing_const_for_fn,
            reason = "Portable control matches the dynamic NEON availability check."
        )
    )]
    fn accelerated() -> bool {
        #[cfg(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc",
            target_feature = "neon"
        ))]
        {
            fearless_simd::Level::baseline().as_neon().is_some()
        }
        #[cfg(not(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc",
            target_feature = "neon"
        )))]
        {
            false
        }
    }

    #[test]
    fn lane_masks_preserve_signed_zero_winners_and_invalid_visitation() {
        for threshold in [f32::MIN, -1.0, -0.0, 0.0, 1.0, f32::MAX] {
            for lane in 0..16 {
                for value in [f32::MIN, -1.0, -0.0, 0.0, f32::from_bits(1), 1.0, f32::MAX] {
                    let mut block = [threshold; 16];
                    block[lane] = value;
                    let expected = if value > threshold { 1_u64 << lane } else { 0 };
                    let actual = candidate_lanes::<true>(&block, threshold);
                    assert_eq!(actual & expected, expected);
                    assert_eq!(actual, candidate_lanes::<false>(&block, threshold));
                    if accelerated() {
                        assert_eq!(actual, expected);
                    } else {
                        assert_eq!(actual, ALL_LANES);
                    }
                }
                for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                    let mut block = [threshold; 16];
                    block[lane] = invalid;
                    assert_eq!(candidate_lanes::<true>(&block, threshold), ALL_LANES);
                }
            }
        }
    }

    #[test]
    fn lane_masks_preserve_dense_and_sparse_visits_as_threshold_advances() -> Result<()> {
        let mut ascending = [0.0_f32; 16];
        for (index, value) in ascending.iter_mut().enumerate() {
            *value = f32::from(u8::try_from(index + 1)?);
        }
        for threshold in 0_u8..=16 {
            let expected = (ALL_LANES << threshold) & ALL_LANES;
            let actual = candidate_lanes::<false>(&ascending, f32::from(threshold));
            assert_eq!(actual & expected, expected);
            if accelerated() {
                assert_eq!(actual, expected);
                assert_eq!(actual.count_ones(), u32::from(16 - threshold));
            }
        }
        for lane in 0..16 {
            let mut sparse = [-1.0_f32; 16];
            sparse[lane] = 100.0;
            let actual = candidate_lanes::<true>(&sparse, 0.0);
            if accelerated() {
                assert_eq!(actual.count_ones(), 1);
                assert_eq!(actual, 1_u64 << lane);
            }
        }
        Ok(())
    }

    #[test]
    fn greedy_lane_pruning_preserves_scalar_ties_tails_and_rejections() -> Result<()> {
        for tail in 0..16 {
            for pattern in 0..6 {
                let mut logits = vec![-0.0_f32; 48 + tail];
                for (index, logit) in logits.iter_mut().enumerate() {
                    let value = f32::from(u16::try_from(index)?);
                    *logit = match pattern {
                        0 => {
                            if index.is_multiple_of(2) {
                                0.0
                            } else {
                                -0.0
                            }
                        }
                        1 => value,
                        2 => -value,
                        3 => {
                            if index.is_multiple_of(16) {
                                f32::MAX
                            } else {
                                f32::MIN
                            }
                        }
                        4 => f32::from(u8::try_from(index % 7)?),
                        _ => f32::from_bits(u32::try_from(index % 5)?),
                    };
                }
                let mut expected = 0;
                for (index, &value) in logits.iter().enumerate().skip(1) {
                    if value > logits[expected] {
                        expected = index;
                    }
                }
                assert_eq!(runtime::greedy(&logits)?, u32::try_from(expected)?);
                assert_eq!(
                    runtime::greedy_finite(FiniteLogits::new(&logits)?)?,
                    u32::try_from(expected)?
                );
                for index in 0..logits.len() {
                    let original = logits[index];
                    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                        logits[index] = invalid;
                        assert!(runtime::greedy(&logits).is_err());
                    }
                    logits[index] = original;
                }
            }
        }
        Ok(())
    }
}
