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
/// allocates at most one bounded candidate vector. Cloning independently copies
/// the RNG so [`crate::runtime::Runtime`] can roll it back on any request error.
#[derive(Clone)]
pub struct SwiftSampler {
    distribution: State,
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
        let mut backing = Vec::new();
        backing.try_reserve_exact(logits.len())?;
        for (index, &logit) in logits.iter().enumerate() {
            backing.push(Candidate {
                id: i32::try_from(index)?,
                logit,
                probability: 0.0,
            });
        }
        let mut candidates = Candidates::new(backing, -1, false)?;
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
            State::new(config)?.apply(&mut candidates)?;
        }
        self.distribution.apply(&mut candidates)?;
        let selected = usize::try_from(candidates.selected())?;
        let token = candidates
            .data()
            .get(selected)
            .ok_or_else(|| invalid("Swift sampler selected an invalid candidate"))?;
        Ok(u32::try_from(token.id)?)
    }
}
