//! Explicit, bounded token selection for the experimental reference runtime.
//!
//! [`ReferenceSampler`] defines its own reproducible seed and rank semantics. It
//! does not reproduce the Prism/C++ random generator or establish production
//! sampler qualification. Existing greedy generation uses [`Greedy`].

use crate::{Result, invalid, runtime::greedy};

pub mod reasoning_budget;
pub(crate) mod selection;
mod swift;
pub mod upstream;

pub use swift::SwiftSampler;

/// Immutable, bounded, nonempty logits whose values have already been checked finite.
///
/// The borrow prevents mutation for the lifetime of the view. Runtime-owned
/// views originate only from checked decoder output or its independent snapshot.
#[derive(Clone, Copy)]
pub struct FiniteLogits<'a>(&'a [f32]);

impl<'a> FiniteLogits<'a> {
    /// Validate a borrowed vocabulary without changing its floating-point bits.
    ///
    /// # Errors
    /// Rejects empty, oversized (more than 1,048,576), or nonfinite logits.
    pub fn new(logits: &'a [f32]) -> Result<Self> {
        let view = Self::from_validated(logits)?;
        if !crate::kernels::is_finite_slice(logits) {
            return Err(invalid(
                "finite logits require a bounded nonempty finite vocabulary",
            ));
        }
        Ok(view)
    }

    /// Borrow the original finite values in vocabulary order.
    #[must_use]
    pub const fn as_slice(self) -> &'a [f32] {
        self.0
    }

    // Decoder::evaluate_finite_into checks every decoder value, or returns the
    // native graph's existing complete-output proof, before Runtime assigns a
    // token to this prefix. Reuse and rollback restore only independent copies
    // of that validated output. Keep the O(1) bounds check here: a custom
    // Decoder may change its reported vocabulary size after Runtime::new.
    pub(crate) fn from_validated(logits: &'a [f32]) -> Result<Self> {
        if logits.is_empty() || logits.len() > 1_048_576 {
            return Err(invalid(
                "finite logits require a bounded nonempty finite vocabulary",
            ));
        }
        Ok(Self(logits))
    }
}

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

    /// Select from logits already validated by the caller or runtime.
    ///
    /// The default preserves existing custom sampler behavior by delegating to
    /// [`Self::sample`]. Implementations may omit repeated finite checks while
    /// retaining their token, probability, tie, RNG and failure semantics.
    ///
    /// # Errors
    /// Returns the same sampler-specific errors as [`Self::sample`].
    fn sample_finite(&mut self, logits: FiniteLogits<'_>) -> Result<u32> {
        self.sample(logits.as_slice())
    }
}

/// Deterministic maximum-logit selection, with the lowest token ID winning ties.
#[derive(Clone, Copy, Debug, Default)]
pub struct Greedy;

impl Sampler for Greedy {
    fn sample(&mut self, logits: &[f32]) -> Result<u32> {
        greedy(logits)
    }

    fn sample_finite(&mut self, logits: FiniteLogits<'_>) -> Result<u32> {
        crate::runtime::greedy_finite(logits)
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

/// Seeded experimental reference sampling with a reusable bounded candidate allocation.
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
/// Clones retain the exact options and generator state without copying scratch.
///
/// This contract is independent of Prism/C++ RNG parity. Floating-point
/// exponentiation may differ between platforms; cross-platform token identity
/// near probability boundaries is not qualified.
pub struct ReferenceSampler {
    options: Sampling,
    state: u64,
    candidates: Vec<Candidate>,
}

impl Clone for ReferenceSampler {
    fn clone(&self) -> Self {
        Self {
            options: self.options,
            state: self.state,
            candidates: Vec::new(),
        }
    }
}

impl std::fmt::Debug for ReferenceSampler {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReferenceSampler")
            .field("options", &self.options)
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
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
            candidates: Vec::new(),
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

fn worse_candidate(left: &Candidate, right: &Candidate) -> bool {
    right
        .logit
        .partial_cmp(&left.logit)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| left.token.cmp(&right.token))
        .is_gt()
}

fn sift_down_worst(heap: &mut [Candidate], mut index: usize) {
    let len = heap.len();
    while let Some(left) = index.checked_mul(2).and_then(|v| v.checked_add(1)) {
        if left >= len {
            break;
        }
        let right = left + 1;
        let worst_child = if right < len && worse_candidate(&heap[right], &heap[left]) {
            right
        } else {
            left
        };
        if !worse_candidate(&heap[worst_child], &heap[index]) {
            break;
        }
        heap.swap(index, worst_child);
        index = worst_child;
    }
}

/// Computes a conservative lower-bound logit cutoff for large-`k` top-`k` pre-filtering
/// using a 16-bucket histogram over finite logits.
///
/// Clamps negative bucket indices to `0..=15` and ignores non-finite values at any
/// position (porting the `ik_llama.cpp` PR `#2857` bucketed top-k fixes).
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "Bucket index is explicitly clamped to 0..=15 before usize/f64 conversion."
)]
pub(crate) fn bucketed_top_k_cutoff(logits: &[f32], keep: usize) -> Option<f32> {
    if keep == 0 || keep >= logits.len() {
        return None;
    }
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;
    for &logit in logits {
        if logit.is_finite() {
            if logit < min_val {
                min_val = logit;
            }
            if logit > max_val {
                max_val = logit;
            }
        }
    }
    if !min_val.is_finite() || !max_val.is_finite() || max_val <= min_val {
        return None;
    }
    let range = f64::from(max_val) - f64::from(min_val);
    if !range.is_finite() || range <= 0.0 {
        return None;
    }
    let scale = 16.0 / range;
    let min_f64 = f64::from(min_val);
    let mut counts = [0_usize; 16];
    for &logit in logits {
        if logit.is_finite() {
            let raw = ((f64::from(logit) - min_f64) * scale).floor() as isize;
            let idx = raw.clamp(0, 15) as usize;
            counts[idx] += 1;
        }
    }
    let mut accumulated = 0_usize;
    for bucket in (0_usize..16).rev() {
        accumulated += counts[bucket];
        if accumulated >= keep {
            if bucket <= 1 {
                return None;
            }
            let cutoff = min_f64 + ((bucket - 1) as f64) / scale;
            return Some(cutoff as f32);
        }
    }
    None
}

fn populate_streaming_top_k(
    logits: &[f32],
    keep: usize,
    candidates: &mut Vec<Candidate>,
) -> Result<()> {
    candidates.clear();
    if candidates.capacity() < keep {
        candidates.try_reserve_exact(keep - candidates.len())?;
    }
    let initial = keep.min(logits.len());
    for (index, &logit) in logits[..initial].iter().enumerate() {
        if !logit.is_finite() {
            candidates.clear();
            return Err(invalid(
                "reference sampling requires bounded nonempty finite logits",
            ));
        }
        candidates.push(Candidate {
            token: u32::try_from(index)?,
            logit,
            weight: 0.0,
        });
    }
    if logits.len() <= keep {
        return Ok(());
    }
    for parent in (0..keep / 2).rev() {
        sift_down_worst(candidates, parent);
    }
    let tail = &logits[keep..];
    let (blocks, remainder) = tail.as_chunks::<16>();
    for (block_index, block) in blocks.iter().enumerate() {
        let mut lanes = selection::candidate_lanes::<true>(block, candidates[0].logit);
        let base = keep + block_index * 16;
        while lanes != 0 {
            let offset = usize::try_from(lanes.trailing_zeros())?;
            lanes &= lanes - 1;
            let logit = block[offset];
            if !logit.is_finite() {
                candidates.clear();
                return Err(invalid(
                    "reference sampling requires bounded nonempty finite logits",
                ));
            }
            if logit > candidates[0].logit {
                candidates[0] = Candidate {
                    token: u32::try_from(base + offset)?,
                    logit,
                    weight: 0.0,
                };
                sift_down_worst(candidates, 0);
            }
        }
    }
    let base = keep + blocks.len() * 16;
    for (offset, &logit) in remainder.iter().enumerate() {
        if !logit.is_finite() {
            candidates.clear();
            return Err(invalid(
                "reference sampling requires bounded nonempty finite logits",
            ));
        }
        if logit > candidates[0].logit {
            candidates[0] = Candidate {
                token: u32::try_from(base + offset)?,
                logit,
                weight: 0.0,
            };
            sift_down_worst(candidates, 0);
        }
    }
    Ok(())
}

impl Sampler for ReferenceSampler {
    fn sample(&mut self, logits: &[f32]) -> Result<u32> {
        if logits.is_empty() || logits.len() > 1_048_576 {
            return Err(invalid(
                "reference sampling requires bounded nonempty finite logits",
            ));
        }
        // Computing a prospective draw does not advance state on any failure.
        let (next_state, draw) = self.next_draw();
        let candidates = &mut self.candidates;
        let rank = |left: &Candidate, right: &Candidate| {
            // Validation excludes incomparable values; partial_cmp treats both
            // signs of zero as equal, so token ID also breaks that exact tie.
            right
                .logit
                .partial_cmp(&left.logit)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.token.cmp(&right.token))
        };
        if self.options.top_k != 0 && self.options.top_k <= 128 {
            populate_streaming_top_k(logits, self.options.top_k, candidates)?;
        } else {
            if !crate::kernels::is_finite_slice(logits) {
                return Err(invalid(
                    "reference sampling requires bounded nonempty finite logits",
                ));
            }
            candidates.clear();
            candidates.try_reserve_exact(logits.len())?;
            let cutoff = if self.options.top_k != 0 && self.options.top_k * 4 < logits.len() {
                bucketed_top_k_cutoff(logits, self.options.top_k)
            } else {
                None
            };
            if let Some(min_logit) = cutoff {
                for (index, &logit) in logits.iter().enumerate() {
                    if logit >= min_logit {
                        candidates.push(Candidate {
                            token: u32::try_from(index)?,
                            logit,
                            weight: 0.0,
                        });
                    }
                }
                if candidates.len() < self.options.top_k {
                    candidates.clear();
                    for (index, &logit) in logits.iter().enumerate() {
                        candidates.push(Candidate {
                            token: u32::try_from(index)?,
                            logit,
                            weight: 0.0,
                        });
                    }
                }
            } else {
                for (index, &logit) in logits.iter().enumerate() {
                    candidates.push(Candidate {
                        token: u32::try_from(index)?,
                        logit,
                        weight: 0.0,
                    });
                }
            }
            if self.options.top_k != 0 && self.options.top_k < candidates.len() {
                // Partition in linear time, then rank only the admitted prefix.
                // The token tie-breaker makes its membership identical to a full
                // sort, including at an equal-logit top-k boundary.
                candidates.select_nth_unstable_by(self.options.top_k, rank);
                candidates.truncate(self.options.top_k);
            }
        }
        candidates.sort_unstable_by(rank);
        let maximum = f64::from(candidates[0].logit);
        let mut total = 0.0;
        for candidate in &mut *candidates {
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
        for candidate in &*candidates {
            retained += 1;
            retained_total += candidate.weight;
            if retained_total >= threshold {
                break;
            }
        }
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

#[cfg(test)]
mod tests {
    use super::{ReferenceSampler, Sampler, Sampling};

    #[test]
    fn finite_logits_preserves_bits_bounds_and_custom_sampler_dispatch() -> crate::Result<()> {
        struct CustomSampler(usize);
        impl Sampler for CustomSampler {
            fn sample(&mut self, logits: &[f32]) -> crate::Result<u32> {
                self.0 += 1;
                crate::runtime::greedy(logits)
            }
        }
        let logits = [0.0, -0.0, f32::MIN, f32::MAX, f32::from_bits(1)];
        let view = super::FiniteLogits::new(&logits)?;
        assert_eq!(view.as_slice().as_ptr(), logits.as_ptr());
        assert_eq!(
            view.as_slice()
                .iter()
                .copied()
                .map(f32::to_bits)
                .collect::<Vec<_>>(),
            logits.map(f32::to_bits)
        );
        assert!(super::FiniteLogits::new(&[]).is_err());
        assert!(super::FiniteLogits::new(&vec![0.0; 1_048_577]).is_err());
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for index in 0..logits.len() {
                let mut rejected = logits;
                rejected[index] = invalid;
                assert!(super::FiniteLogits::new(&rejected).is_err());
            }
        }
        let mut sampler = CustomSampler(0);
        assert_eq!(sampler.sample_finite(view)?, 3);
        assert_eq!(sampler.0, 1);
        Ok(())
    }

    #[test]
    fn candidate_storage_reuses_capacity_and_clones_only_semantic_state() -> crate::Result<()> {
        let mut sampler = ReferenceSampler::new(Sampling::default(), 0)?;
        sampler.sample(&[0.0; 257])?;
        let allocation = sampler.candidates.as_ptr();
        let capacity = sampler.candidates.capacity();
        let mut control = sampler.clone();
        assert_eq!(control.candidates.capacity(), 0);

        for width in [257, 31, 1, 256] {
            let mut logits = vec![-1000.0; width];
            logits[width - 1] = 1000.0;
            let expected = u32::try_from(width - 1)?;
            assert_eq!(sampler.sample(&logits)?, expected);
            assert_eq!(control.sample(&logits)?, expected);
            assert_eq!(sampler.candidates.as_ptr(), allocation);
            assert_eq!(sampler.candidates.capacity(), capacity);
        }
        assert_ne!(control.candidates.as_ptr(), allocation);
        assert!(sampler.sample(&[f32::NAN; 257]).is_err());
        assert!(sampler.sample(&[]).is_err());
        assert_eq!(sampler.candidates.as_ptr(), allocation);
        assert_eq!(sampler.sample(&[0.0; 257])?, control.sample(&[0.0; 257])?);
        Ok(())
    }

    #[test]
    fn streaming_and_bucketed_top_k_match_exhaustive_rank_across_ties_and_extremes()
    -> crate::Result<()> {
        let logits: Vec<f32> = (0..4096_u32)
            .map(|i| {
                if i % 17 == 0 {
                    -0.0
                } else if i % 13 == 0 {
                    0.0
                } else {
                    let bits = i.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    f32::from_bits(bits & 0xbf7f_ffff)
                }
            })
            .collect();
        for top_k in [1_usize, 7, 20, 128, 256] {
            let mut sampler = ReferenceSampler::new(
                Sampling {
                    temperature: 0.8,
                    top_k,
                    top_p: 0.95,
                },
                12345,
            )?;
            let _ = sampler.sample(&logits)?;
            let mut full: Vec<super::Candidate> = logits
                .iter()
                .enumerate()
                .map(|(idx, &logit)| super::Candidate {
                    token: u32::try_from(idx).unwrap_or(0),
                    logit,
                    weight: 0.0,
                })
                .collect();
            full.sort_unstable_by(|left, right| {
                right
                    .logit
                    .partial_cmp(&left.logit)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| left.token.cmp(&right.token))
            });
            full.truncate(top_k);
            assert_eq!(sampler.candidates.len(), top_k);
            for (actual, expected) in sampler.candidates.iter().zip(&full) {
                assert_eq!(actual.token, expected.token);
                assert_eq!(actual.logit.to_bits(), expected.logit.to_bits());
            }
        }
        Ok(())
    }

    #[test]
    fn bucketed_top_k_cutoff_clamps_negative_buckets_and_handles_non_tail_infinities() {
        let mut logits = vec![-10.0_f32; 1024];
        logits[0] = f32::NEG_INFINITY;
        logits[511] = f32::NEG_INFINITY;
        for (idx, dst) in logits[100..132].iter_mut().enumerate() {
            *dst = f32::from(u8::try_from(idx).unwrap_or(0));
        }
        let cutoff = super::bucketed_top_k_cutoff(&logits, 16);
        assert!(cutoff.is_some());
        let threshold = cutoff.unwrap_or(f32::INFINITY);
        let surviving = logits
            .iter()
            .filter(|&&v| v.is_finite() && v >= threshold)
            .count();
        assert!(surviving >= 16);
    }
}
