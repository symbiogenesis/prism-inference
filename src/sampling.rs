//! Explicit, bounded token selection for the experimental reference runtime.
//!
//! [`ReferenceSampler`] defines its own reproducible seed and rank semantics. It
//! does not reproduce the Prism/C++ random generator or establish production
//! sampler qualification. Existing greedy generation uses [`Greedy`].

use crate::{Result, invalid, runtime::greedy};

pub mod reasoning_budget;
mod swift;
pub mod upstream;

pub use swift::SwiftSampler;

/// Select a token from the complete next-token logits.
///
/// A sampler may retain mutable state. Runtime requests that accept a sampler
/// clone it to restore its state if the request fails.
pub trait Sampler {
    /// Return an admitted token ID from a nonempty finite logits distribution.
    ///
    /// # Errors
    /// Rejects invalid logits, sampler options, or resource exhaustion.
    fn sample(&mut self, logits: &[f32]) -> Result<u32>;
}

/// Deterministic maximum-logit selection, with the lowest token ID winning ties.
#[derive(Clone, Copy, Debug, Default)]
pub struct Greedy;

impl Sampler for Greedy {
    fn sample(&mut self, logits: &[f32]) -> Result<u32> {
        greedy(logits)
    }
}

/// Explicit temperature, top-k and nucleus settings for [`ReferenceSampler`].
#[derive(Clone, Copy, Debug)]
pub struct Sampling {
    /// Finite, strictly positive temperature applied before stable softmax.
    pub temperature: f32,
    /// Maximum ranked candidates retained, at most 1,048,576; zero admits the
    /// full bounded vocabulary.
    pub top_k: usize,
    /// Finite nucleus probability in `(0, 1]`, retaining the crossing token.
    pub top_p: f32,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 20,
            top_p: 0.95,
        }
    }
}

/// Seeded experimental reference sampling with a bounded candidate allocation.
///
/// Each call first validates all 1 through 1,048,576 logits, ranks them by
/// descending value and ascending token ID, applies temperature and top-k,
/// computes stable softmax weights in `f64`, then retains the smallest ranked
/// top-p prefix including the threshold-crossing token and at least one token.
/// The retained weights are renormalized for one categorical draw.
///
/// The seed is the initial `SplitMix64` state. Every successful call adds
/// `0x9e3779b97f4a7c15` with wrapping arithmetic, then mixes with xor shifts
/// 30, 27 and 31 and wrapping multipliers `0xbf58476d1ce4e5b9` and
/// `0x94d049bb133111eb`. The upper 32 output bits divided by `2^32` form a
/// uniform draw in `[0, 1)`. Invalid inputs and allocation errors do not consume
/// a draw; even a single-candidate successful sample consumes exactly one.
/// Clones retain the exact options and generator state.
///
/// This contract is independent of Prism/C++ RNG parity. Floating-point
/// exponentiation may differ between platforms; cross-platform token identity
/// near probability boundaries is not qualified.
#[derive(Clone, Debug)]
pub struct ReferenceSampler {
    options: Sampling,
    state: u64,
}

impl ReferenceSampler {
    /// Validate explicit reference settings and initialize a reproducible seed.
    ///
    /// # Errors
    /// Rejects nonfinite or nonpositive temperature, top-k above 1,048,576 and
    /// top-p outside `(0, 1]`.
    pub fn new(options: Sampling, seed: u64) -> Result<Self> {
        if !options.temperature.is_finite()
            || options.temperature <= 0.0
            || options.top_k > 1_048_576
            || !options.top_p.is_finite()
            || options.top_p <= 0.0
            || options.top_p > 1.0
        {
            return Err(invalid("invalid experimental reference sampling options"));
        }
        Ok(Self {
            options,
            state: seed,
        })
    }

    fn next_draw(&self) -> (u64, f64) {
        let next_state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut mixed = next_state;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        mixed ^= mixed >> 31;
        let [first, second, third, fourth, ..] = mixed.to_be_bytes();
        let upper = u32::from_be_bytes([first, second, third, fourth]);
        (next_state, f64::from(upper) / 4_294_967_296.0)
    }
}

struct Candidate {
    token: u32,
    logit: f32,
    weight: f64,
}

impl Sampler for ReferenceSampler {
    fn sample(&mut self, logits: &[f32]) -> Result<u32> {
        if logits.is_empty()
            || logits.len() > 1_048_576
            || logits.iter().any(|logit| !logit.is_finite())
        {
            return Err(invalid(
                "reference sampling requires bounded nonempty finite logits",
            ));
        }
        let mut candidates = Vec::new();
        candidates.try_reserve_exact(logits.len())?;
        for (index, &logit) in logits.iter().enumerate() {
            candidates.push(Candidate {
                token: u32::try_from(index)?,
                logit,
                weight: 0.0,
            });
        }
        let rank = |left: &Candidate, right: &Candidate| {
            // Validation excludes incomparable values; partial_cmp treats both
            // signs of zero as equal, so token ID also breaks that exact tie.
            right
                .logit
                .partial_cmp(&left.logit)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.token.cmp(&right.token))
        };
        if self.options.top_k != 0 && self.options.top_k < candidates.len() {
            // Partition in linear time, then rank only the admitted prefix.
            // The token tie-breaker makes its membership identical to a full
            // sort, including at an equal-logit top-k boundary.
            candidates.select_nth_unstable_by(self.options.top_k, rank);
            candidates.truncate(self.options.top_k);
        }
        candidates.sort_unstable_by(rank);
        let maximum = f64::from(candidates[0].logit);
        let mut total = 0.0;
        for candidate in &mut candidates {
            // Subtract in f64 before dividing so every finite f32 temperature
            // and logit stays representable, even at their extreme values.
            candidate.weight = ((f64::from(candidate.logit) - maximum)
                / f64::from(self.options.temperature))
            .exp();
            total += candidate.weight;
        }
        let threshold = total * f64::from(self.options.top_p);
        let mut retained = 0;
        let mut retained_total = 0.0;
        for candidate in &candidates {
            retained += 1;
            retained_total += candidate.weight;
            if retained_total >= threshold {
                break;
            }
        }
        let (next_state, draw) = self.next_draw();
        let target = draw * retained_total;
        let mut cumulative = 0.0;
        let mut selected = candidates[0].token;
        for candidate in &candidates[..retained] {
            cumulative += candidate.weight;
            if candidate.weight > 0.0 {
                selected = candidate.token;
            }
            if target < cumulative {
                break;
            }
        }
        self.state = next_state;
        Ok(selected)
    }
}
