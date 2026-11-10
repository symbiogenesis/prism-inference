//! Fixed Swift sampler bridge to the production Prism candidate/RNG contract.

use super::{
    FiniteLogits, Sampler,
    upstream::{Candidate, Candidates, Config, State, top_k_finite_logits, top_k_logits},
};
use crate::{Result, invalid};

/// Seeded sampler for the pinned 248,320-token Swift PQ2 model.
///
/// The fixed chain is top-k 20, top-p 0.95 (minimum one), min-p zero,
/// temperature one and the production Windows MSVC MT19937 distribution.
/// Repetition is one and presence/frequency penalties are zero; these identity
/// stages retain no history and consume no draws. There is no reasoning cutoff.
/// Each call admits exactly the complete Swift vocabulary before filtering and
/// retains only the native-ordered top 20 candidates in reusable scratch.
/// The fixed top-p configuration is validated once at construction. Native
/// min-p zero and temperature one retain their exact identity semantics without
/// constructing temporary sampler states for each generated token.
/// Top-p's unnormalized binary32 exponentials are retained in 20 stack values
/// for the following distribution, preserving its independent binary64 sum.
/// Cloning independently copies the RNG without copying candidate scratch, so
/// [`crate::runtime::Runtime`] can roll it back on any request error.
pub struct SwiftSampler {
    top_p: State,
    distribution: State,
    backing: Vec<Candidate>,
}

impl Clone for SwiftSampler {
    fn clone(&self) -> Self {
        Self {
            top_p: self.top_p.clone(),
            distribution: self.distribution.clone(),
            backing: Vec::new(),
        }
    }
}

impl SwiftSampler {
    const TOP_K: usize = 20;

    /// Vocabulary extent declared by the pinned Swift model.
    pub const VOCABULARY: usize = 248_320;

    /// Creates the fixed chain with a reproducible explicit native seed.
    ///
    /// # Errors
    /// Rejects `u32::MAX`, the upstream entropy sentinel, so reset and replay
    /// retain the exact seed instead of silently selecting fresh entropy.
    pub fn new(seed: u32) -> Result<Self> {
        if seed == u32::MAX {
            return Err(invalid(
                "Swift sampling requires an explicit non-sentinel seed",
            ));
        }
        Ok(Self {
            top_p: State::new(Config::TopP {
                probability: 0.95,
                min_keep: 1,
            })?,
            distribution: State::new(Config::Distribution(seed))?,
            backing: Vec::new(),
        })
    }

    /// Restores the original seed, including the production draw ordering.
    pub fn reset(&mut self) {
        self.distribution.reset();
    }
}

impl Sampler for SwiftSampler {
    fn sample(&mut self, logits: &[f32]) -> Result<u32> {
        if logits.len() != Self::VOCABULARY {
            return Err(invalid(
                "Swift sampling requires the complete finite pinned vocabulary",
            ));
        }
        self.sample_selected(|backing| top_k_logits(logits, Self::TOP_K, backing))
    }

    fn sample_finite(&mut self, logits: FiniteLogits<'_>) -> Result<u32> {
        if logits.as_slice().len() != Self::VOCABULARY {
            return Err(invalid(
                "Swift sampling requires the complete finite pinned vocabulary",
            ));
        }
        self.sample_selected(|backing| top_k_finite_logits(logits, Self::TOP_K, backing))
    }
}

impl SwiftSampler {
    fn sample_selected(
        &mut self,
        select: impl FnOnce(&mut Vec<Candidate>) -> Result<()>,
    ) -> Result<u32> {
        let mut backing = std::mem::take(&mut self.backing);
        // Ordinary selection validates every logit, including discarded candidates, in
        // the same traversal. Preserve reusable storage if that validation fails.
        if let Err(error) = select(&mut backing) {
            self.backing = backing;
            return Err(error);
        }
        let mut candidates = Candidates::new(backing, -1, true)?;
        let selected = self.sample_candidates(&mut candidates);
        self.backing = candidates.into_backing();
        selected
    }
}

impl SwiftSampler {
    fn sample_candidates(&mut self, candidates: &mut Candidates) -> Result<u32> {
        if candidates.sorted() && candidates.data().len() <= Self::TOP_K {
            let mut weights = [0.0_f32; Self::TOP_K];
            self.distribution.apply_sorted_top_p_distribution(
                &self.top_p,
                candidates,
                &mut weights,
            )?;
        } else {
            // Keep the original independent stages for private unsorted inputs.
            // Ordinary Swift selection always supplies the bounded sorted heap.
            self.top_p.apply(candidates)?;
            self.distribution.apply(candidates)?;
        }
        // Native MinP(0) returns without mutation, and division by temperature
        // one preserves every finite logit bit (including signed zero). They
        // consume no draws and change no candidate metadata or probability.
        let selected = usize::try_from(candidates.selected())?;
        let token = candidates
            .data()
            .get(selected)
            .ok_or_else(|| invalid("Swift sampler selected an invalid candidate"))?;
        Ok(u32::try_from(token.id)?)
    }
}

#[cfg(test)]
mod tests {
    use super::{Sampler, SwiftSampler};
    use crate::sampling::upstream::{Candidate, Candidates, Config, State, top_k_logits};

    fn original_fixed_chain(
        distribution: &mut State,
        candidates: &mut Candidates,
    ) -> crate::Result<u32> {
        for config in [
            Config::TopP {
                probability: 0.95,
                min_keep: 1,
            },
            Config::MinP {
                probability: 0.0,
                min_keep: 1,
            },
            Config::Temperature(1.0),
        ] {
            State::new(config)?.apply(candidates)?;
        }
        distribution.apply(candidates)?;
        let selected = usize::try_from(candidates.selected())?;
        Ok(u32::try_from(candidates.data()[selected].id)?)
    }

    fn assert_candidate_bits(actual: &Candidates, expected: &Candidates) {
        assert_eq!(actual.selected(), expected.selected());
        assert_eq!(actual.sorted(), expected.sorted());
        assert_eq!(actual.offset(), expected.offset());
        assert_eq!(actual.data().len(), expected.data().len());
        let actual = actual.clone().into_backing();
        let expected = expected.clone().into_backing();
        assert_eq!(actual.len(), expected.len());
        for (observed, original) in actual.iter().zip(expected) {
            assert_eq!(observed.id, original.id);
            assert_eq!(observed.logit.to_bits(), original.logit.to_bits());
            assert_eq!(observed.probability.to_bits(), original.probability.to_bits());
        }
    }

    #[test]
    fn cached_top_p_exponents_preserve_original_backing_bits_draws_and_reset() -> crate::Result<()> {
        for width in [1_usize, 2, 19, 20] {
            let mut sampler = SwiftSampler::new(4242)?;
            let mut reference = State::new(Config::Distribution(4242))?;
            for round in 0_usize..64 {
                if round == 32 {
                    sampler.reset();
                    reference.reset();
                }
                let input = (0..width)
                    .map(|index| {
                        let value = f32::from(u16::try_from(index)?);
                        Ok(Candidate {
                            id: i32::try_from(index)?,
                            logit: match round % 10 {
                                0 => 0.0,
                                1 => if index.is_multiple_of(2) { -0.0 } else { 0.0 },
                                2 => -value,
                                3 => -value * 5.0,
                                4 => f32::MIN,
                                5 => f32::MAX,
                                6 => if index == 0 { f32::MAX } else { f32::MIN },
                                7 => -value * 1000.0,
                                8 => -value * f32::EPSILON,
                                _ => -value * f32::from_bits(1),
                            },
                            probability: -0.0,
                        })
                    })
                    .collect::<crate::Result<Vec<_>>>()?;
                let mut original = Candidates::new(input, -1, true)?;
                let mut actual = original.clone();
                let mut independent = sampler.clone();
                let mut cloned_candidates = original.clone();
                let token = original_fixed_chain(&mut reference, &mut original)?;
                assert_eq!(sampler.sample_candidates(&mut actual)?, token);
                assert_eq!(independent.sample_candidates(&mut cloned_candidates)?, token);
                assert_candidate_bits(&actual, &original);
                assert_candidate_bits(&cloned_candidates, &original);
            }
        }
        Ok(())
    }

    fn assert_cached_rejected(
        distribution: &mut State,
        filter: &State,
        original: &Candidates,
        weights: &mut [f32],
    ) {
        let mut actual = original.clone();
        assert!(distribution.apply_sorted_top_p_distribution(filter, &mut actual, weights).is_err());
        assert_candidate_bits(&actual, original);
    }

    #[test]
    fn cached_top_p_guards_preserve_candidates_and_rng_with_unsorted_fallback() -> crate::Result<()> {
        let filter = State::new(Config::TopP {
            probability: 0.95,
            min_keep: 1,
        })?;
        let mut sampler = SwiftSampler::new(4242)?;
        let mut reference = State::new(Config::Distribution(4242))?;
        let backing = (0..20)
            .map(|id| Candidate { id, logit: 0.0, probability: -0.0 })
            .collect::<Vec<_>>();
        let original = Candidates::new(backing, -1, true)?;
        for capacity in [0, 19] {
            let mut weights = vec![0.0_f32; capacity];
            assert_cached_rejected(&mut sampler.distribution, &filter, &original, &mut weights);
        }
        for invalid in [Config::Temperature(1.0), Config::TopP { probability: 1.0, min_keep: 1 }] {
            assert_cached_rejected(&mut sampler.distribution, &State::new(invalid)?, &original, &mut [0.0_f32; 20]);
        }
        let mut wrong_distribution = State::new(Config::TopK(20))?;
        assert_cached_rejected(&mut wrong_distribution, &filter, &original, &mut [0.0_f32; 20]);
        let mut candidates = original.clone();
        let mut expected = original.clone();
        assert_eq!(sampler.sample_candidates(&mut candidates)?, original_fixed_chain(&mut reference, &mut expected)?);
        assert_candidate_bits(&candidates, &expected);
        let mut unsorted = Candidates::new(original.into_backing(), -1, false)?;
        let before = unsorted.clone();
        assert_cached_rejected(&mut sampler.distribution, &filter, &unsorted, &mut [0.0_f32; 20]);
        let mut expected = before;
        assert_eq!(sampler.sample_candidates(&mut unsorted)?, original_fixed_chain(&mut reference, &mut expected)?);
        assert_candidate_bits(&unsorted, &expected);
        Ok(())
    }

    #[test]
    fn finite_selection_matches_checked_heap_bits_rng_reset_and_rejection() -> crate::Result<()> {
        let mut checked = SwiftSampler::new(4242)?;
        let mut finite = SwiftSampler::new(4242)?;
        let mut logits = vec![0.0; SwiftSampler::VOCABULARY];
        for round in 0_usize..32 {
            if round == 16 {
                checked.reset();
                finite.reset();
            }
            for (index, logit) in logits.iter_mut().enumerate() {
                *logit = match round % 8 {
                    0 => -0.0,
                    1 => {
                        if index.is_multiple_of(2) {
                            -0.0
                        } else {
                            0.0
                        }
                    }
                    2 => f32::MIN,
                    3 => f32::MAX,
                    4 => f32::from(u8::try_from(index % 7)?) - 3.0,
                    5 => f32::from(u16::try_from(index / 4)?),
                    6 => -f32::from(u16::try_from(index / 4)?),
                    _ => f32::from_bits(u32::try_from(index % 1024)?),
                };
            }
            let expected = checked.sample(&logits)?;
            let view = super::FiniteLogits::new(&logits)?;
            let mut independent = finite.clone();
            assert_eq!(finite.sample_finite(view)?, expected);
            assert_eq!(independent.sample_finite(view)?, expected);
            assert_eq!(checked.backing.len(), finite.backing.len());
            for (original, actual) in checked.backing.iter().zip(&finite.backing) {
                assert_eq!(actual.id, original.id);
                assert_eq!(actual.logit.to_bits(), original.logit.to_bits());
                assert_eq!(actual.probability.to_bits(), original.probability.to_bits());
            }
            // Rejected public and typed extents must not consume a draw or
            // replace the reusable allocation of either path.
            let allocation = finite.backing.as_ptr();
            assert!(
                finite
                    .sample_finite(super::FiniteLogits::new(&[0.0])?)
                    .is_err()
            );
            assert!(finite.sample(&[f32::NAN]).is_err());
            assert_eq!(finite.backing.as_ptr(), allocation);
        }
        assert_eq!(checked.sample(&logits)?, finite.sample(&logits)?);
        Ok(())
    }

    #[test]
    fn fixed_chain_reuses_filter_and_preserves_original_stage_bits_and_rng() -> crate::Result<()> {
        let mut actual = SwiftSampler::new(4242)?;
        let mut reference_distribution = State::new(Config::Distribution(4242))?;
        let mut invalid_logits = vec![0.0; SwiftSampler::VOCABULARY];
        invalid_logits[SwiftSampler::VOCABULARY - 1] = f32::NAN;
        for round in 0_usize..64 {
            if round == 32 {
                actual.reset();
                reference_distribution.reset();
            }
            // Rejected full-vocabulary input must not advance the retained
            // filter or distribution. The reference deliberately does no work.
            if round.is_multiple_of(16) {
                assert!(actual.sample(&invalid_logits).is_err());
            }
            let input = (0..SwiftSampler::TOP_K)
                .map(|index| {
                    let value = f32::from(u16::try_from(index)?);
                    Ok(Candidate {
                        id: i32::try_from(index)?,
                        logit: match round % 8 {
                            0 => 0.0,
                            1 => {
                                if index.is_multiple_of(2) {
                                    -0.0
                                } else {
                                    0.0
                                }
                            }
                            2 => value - 10.0,
                            3 => 10.0 - value,
                            4 => f32::MAX,
                            5 => f32::MIN,
                            6 => value * f32::from_bits(1),
                            _ => (value - 10.0) * 1000.0,
                        },
                        probability: -0.0,
                    })
                })
                .collect::<crate::Result<Vec<_>>>()?;
            let mut expected = Candidates::new(input, -1, false)?;
            let mut optimized = expected.clone();
            let mut independent_clone = actual.clone();
            let mut clone_candidates = expected.clone();
            let expected_token = original_fixed_chain(&mut reference_distribution, &mut expected)?;
            assert_eq!(actual.sample_candidates(&mut optimized)?, expected_token);
            assert_eq!(
                independent_clone.sample_candidates(&mut clone_candidates)?,
                expected_token
            );
            assert_eq!(optimized.selected(), expected.selected());
            assert_eq!(optimized.sorted(), expected.sorted());
            assert_eq!(optimized.offset(), expected.offset());
            assert_eq!(optimized.data().len(), expected.data().len());
            for (observed, original) in optimized.into_backing().iter().zip(expected.into_backing())
            {
                assert_eq!(observed.id, original.id);
                assert_eq!(observed.logit.to_bits(), original.logit.to_bits());
                assert_eq!(
                    observed.probability.to_bits(),
                    original.probability.to_bits()
                );
            }
        }
        Ok(())
    }

    #[test]
    fn bounded_top_k_matches_full_candidate_filter_bit_for_bit() -> crate::Result<()> {
        let mut bounded = Vec::new();
        for kind in 0..8 {
            let logits = (0..SwiftSampler::VOCABULARY)
                .map(|index| {
                    Ok(match kind {
                        0 => 0.0,
                        1 => {
                            if index.is_multiple_of(2) {
                                0.0
                            } else {
                                -0.0
                            }
                        }
                        2 => f32::from(u16::try_from(index * 17 % 7)?) - 3.0,
                        3 => f32::from(u16::try_from(index / 4)?),
                        4 => -f32::from(u16::try_from(index / 4)?),
                        5 => {
                            let bits = u32::try_from(index)?
                                .wrapping_mul(1_664_525)
                                .wrapping_add(1_013_904_223);
                            // Clear one exponent bit to retain finite values,
                            // including subnormals, signs and wide magnitudes.
                            f32::from_bits(bits & 0xff7f_ffff)
                        }
                        6 => {
                            if index == SwiftSampler::VOCABULARY - 1 {
                                f32::MAX
                            } else {
                                f32::MIN
                            }
                        }
                        _ => {
                            if index == SwiftSampler::VOCABULARY - 1 {
                                f32::MIN
                            } else {
                                f32::MAX
                            }
                        }
                    })
                })
                .collect::<crate::Result<Vec<_>>>()?;
            let full = logits
                .iter()
                .enumerate()
                .map(|(index, &logit)| {
                    Ok(Candidate {
                        id: i32::try_from(index)?,
                        logit,
                        probability: 0.0,
                    })
                })
                .collect::<crate::Result<Vec<_>>>()?;
            let mut reference = Candidates::new(full, -1, false)?;
            State::new(Config::TopK(20))?.apply(&mut reference)?;
            top_k_logits(&logits, SwiftSampler::TOP_K, &mut bounded)?;
            assert_eq!(bounded.len(), SwiftSampler::TOP_K);
            assert_eq!(bounded.capacity(), SwiftSampler::TOP_K);
            for (actual, expected) in bounded.iter().zip(reference.data()) {
                assert_eq!(actual.id, expected.id, "input kind {kind}");
                assert_eq!(actual.logit.to_bits(), expected.logit.to_bits());
                assert_eq!(actual.probability.to_bits(), expected.probability.to_bits());
            }

            // Compare the unchanged probability and RNG stages against the full
            // candidate route after exercising every replacement/tie pattern.
            let mut sampler = SwiftSampler::new(4242)?;
            let mut control = SwiftSampler::new(4242)?;
            assert_eq!(
                sampler.sample(&logits)?,
                original_fixed_chain(&mut control.distribution, &mut reference)?,
                "input kind {kind}"
            );
            let uniform = vec![0.0; SwiftSampler::VOCABULARY];
            assert_eq!(sampler.sample(&uniform)?, control.sample(&uniform)?);
        }
        Ok(())
    }

    #[test]
    fn candidate_storage_reuses_top_k_backing_across_filters_rejection_and_reset()
    -> crate::Result<()> {
        let mut sampler = SwiftSampler::new(0)?;
        let mut logits = vec![0.0; SwiftSampler::VOCABULARY];
        assert_eq!(sampler.sample(&logits)?, 13);
        let allocation = sampler.backing.as_ptr();
        let capacity = sampler.backing.capacity();
        assert_eq!(sampler.backing.len(), SwiftSampler::TOP_K);
        assert_eq!(capacity, SwiftSampler::TOP_K);

        // Transaction checkpoints copy RNG state but never the top-k
        // scratch allocation. Subsequent independent draws still match.
        let mut control = sampler.clone();
        assert_eq!(control.backing.capacity(), 0);
        for peak in [248_319, 21, 0] {
            logits.fill(0.0);
            logits[peak] = 1000.0;
            let expected = u32::try_from(peak)?;
            assert_eq!(sampler.sample(&logits)?, expected);
            assert_eq!(control.sample(&logits)?, expected);
            assert_eq!(sampler.backing.as_ptr(), allocation);
            assert_eq!(sampler.backing.capacity(), capacity);
        }
        assert_ne!(control.backing.as_ptr(), allocation);

        logits.fill(0.0);
        logits[248_319] = f32::NAN;
        assert!(sampler.sample(&logits).is_err());
        assert!(sampler.sample(&[]).is_err());
        assert_eq!(sampler.backing.as_ptr(), allocation);
        logits[248_319] = 0.0;
        assert_eq!(sampler.sample(&logits)?, control.sample(&logits)?);

        sampler.reset();
        for expected in [13, 0, 0, 0, 13, 1, 8, 10, 8, 5, 9, 5] {
            assert_eq!(sampler.sample(&logits)?, expected);
            assert_eq!(sampler.backing.as_ptr(), allocation);
            assert_eq!(sampler.backing.capacity(), capacity);
        }
        Ok(())
    }

    #[test]
    fn selection_rejects_nonfinite_discarded_logits_without_losing_rng_or_storage()
    -> crate::Result<()> {
        let mut actual = SwiftSampler::new(42)?;
        let mut expected = SwiftSampler::new(42)?;
        let mut logits = vec![0.0; SwiftSampler::VOCABULARY];
        assert_eq!(actual.sample(&logits)?, expected.sample(&logits)?);
        let allocation = actual.backing.as_ptr();
        let capacity = actual.backing.capacity();
        for index in [0, 19, 20, SwiftSampler::VOCABULARY - 1] {
            for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                logits[index] = value;
                assert!(actual.sample(&logits).is_err());
                assert_eq!(actual.backing.len(), 0);
                assert_eq!(actual.backing.as_ptr(), allocation);
                assert_eq!(actual.backing.capacity(), capacity);
                logits[index] = 0.0;
                assert_eq!(actual.sample(&logits)?, expected.sample(&logits)?);
            }
        }
        Ok(())
    }
}
