//! Fixed Swift sampler bridge to the production Prism candidate/RNG contract.

use super::{
    Sampler,
    upstream::{Candidate, Candidates, Config, State},
};
use crate::{Result, invalid};

/// Seeded sampler for the pinned 248,320-token Swift PQ2 model.
///
/// The fixed chain is top-k 20, top-p 0.95 (minimum one), min-p zero,
/// temperature one and the production Windows MSVC MT19937 distribution.
/// Repetition is one and presence/frequency penalties are zero; these identity
/// stages retain no history and consume no draws. There is no reasoning cutoff.
/// Each call admits exactly the complete Swift vocabulary before filtering and
/// reuses one bounded candidate allocation after its first successful reserve.
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
        if logits.len() != Self::VOCABULARY || logits.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "Swift sampling requires the complete finite pinned vocabulary",
            ));
        }
        let mut backing = std::mem::take(&mut self.backing);
        backing.clear();
        backing.try_reserve_exact(logits.len())?;
        for (index, &logit) in logits.iter().enumerate() {
            backing.push(Candidate {
                id: i32::try_from(index)?,
                logit,
                probability: 0.0,
            });
        }
        let mut candidates = Candidates::new(backing, -1, false)?;
        let selected = self.sample_candidates(&mut candidates);
        self.backing = candidates.into_backing();
        selected
    }
}

impl SwiftSampler {
    fn sample_candidates(&mut self, candidates: &mut Candidates) -> Result<u32> {
        for config in [
            Config::TopK(20),
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

    #[test]
    fn candidate_storage_reuses_full_backing_across_filters_rejection_and_reset()
    -> crate::Result<()> {
        let mut sampler = SwiftSampler::new(0)?;
        let mut logits = vec![0.0; SwiftSampler::VOCABULARY];
        assert_eq!(sampler.sample(&logits)?, 13);
        let allocation = sampler.backing.as_ptr();
        let capacity = sampler.backing.capacity();
        assert_eq!(sampler.backing.len(), SwiftSampler::VOCABULARY);

        // Transaction checkpoints copy RNG state but never the vocabulary-sized
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
}
