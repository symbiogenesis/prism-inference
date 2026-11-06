//! Fixed Swift sampler bridge to the production Prism candidate/RNG contract.

use super::{
    Sampler,
    upstream::{Candidate, Candidates, Config, State, top_k_logits},
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
/// Cloning independently copies the RNG without copying candidate scratch, so
/// [`crate::runtime::Runtime`] can roll it back on any request error.
pub struct SwiftSampler {
    distribution: State,
    backing: Vec<Candidate>,
}

impl Clone for SwiftSampler {
    fn clone(&self) -> Self {
        Self {
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
        let mut backing = std::mem::take(&mut self.backing);
        // Selection validates every logit, including discarded candidates, in
        // the same traversal. Preserve reusable storage if that validation fails.
        if let Err(error) = top_k_logits(logits, Self::TOP_K, &mut backing) {
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
        self.distribution.apply(candidates)?;
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
                control.sample_candidates(&mut reference)?,
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
