//! Owned stateful Prism CPU samplers, ported from pinned `llama-sampler.cpp`.
//!
//! Candidate filters preserve backing storage and view offsets. Seeded draws
//! reproduce the Windows MSVC MT19937 canonical float/double distributions.
//! The experimental [`super::ReferenceSampler`] retains its independent contract.
#![expect(
    clippy::suboptimal_flops,
    reason = "Keep the pinned native sampler's separate floating-point operation roundings."
)]

use std::cmp::Ordering;
use std::collections::VecDeque;
use std::hash::{BuildHasher, Hasher};

use crate::{Result, invalid};

// Swift's pinned Qwen35 vocabulary contains 248,320 entries. This bound applies
// before filtering, so top-k must not hide an otherwise oversized vocabulary.
const MAX_CANDIDATES: usize = 1_048_576;

/// One token's mutable logit and probability.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    /// Vocabulary token identifier.
    pub id: i32,
    /// Raw IEEE binary32 logit, including native infinity masks.
    pub logit: f32,
    /// Current probability, recomputed by probability-based samplers.
    pub probability: f32,
}

/// Independently owned candidate storage and its current active view.
#[derive(Clone, Debug)]
pub struct Candidates {
    backing: Vec<Candidate>,
    offset: usize,
    size: usize,
    selected: i64,
    sorted: bool,
}

impl Candidates {
    /// Creates a bounded nonempty view without changing supplied float bits.
    ///
    /// # Errors
    /// Rejects empty/oversized backing storage or an out-of-range selected index.
    pub fn new(backing: Vec<Candidate>, selected: i64, sorted: bool) -> Result<Self> {
        if backing.is_empty()
            || backing.len() > MAX_CANDIDATES
            || selected < -1
            || (selected >= 0 && usize::try_from(selected)? >= backing.len())
        {
            return Err(invalid(
                "sampler candidate extent or selected index is invalid",
            ));
        }
        Ok(Self {
            size: backing.len(),
            backing,
            offset: 0,
            selected,
            sorted,
        })
    }

    /// Active candidate view after all in-place filtering and offset advances.
    #[must_use]
    pub fn data(&self) -> &[Candidate] {
        &self.backing[self.offset..self.offset + self.size]
    }

    fn data_mut(&mut self) -> &mut [Candidate] {
        &mut self.backing[self.offset..self.offset + self.size]
    }

    /// Selected position relative to the active view; -1 means unselected.
    #[must_use]
    pub const fn selected(&self) -> i64 {
        self.selected
    }

    /// Whether the active view is marked as descending by logit.
    #[must_use]
    pub const fn sorted(&self) -> bool {
        self.sorted
    }

    /// Current view's checked index into independently retained backing storage.
    #[must_use]
    pub const fn offset(&self) -> usize {
        self.offset
    }

    /// Retained backing extent, including candidates outside the active view.
    #[must_use]
    pub const fn backing_len(&self) -> usize {
        self.backing.len()
    }

    // Recover the complete allocation, including entries outside a filtered
    // view. A subsequent sample must rebuild every entry and its view metadata.
    pub(super) fn into_backing(self) -> Vec<Candidate> {
        self.backing
    }

    fn sort(&mut self) {
        self.data_mut().sort_unstable_by(|left, right| {
            descending(left.logit, right.logit).then_with(|| left.id.cmp(&right.id))
        });
        self.sorted = true;
    }

    fn softmax(&mut self, sort: bool) {
        if sort && !self.sorted {
            self.sort();
        }
        let mut maximum = self.data()[0].logit;
        if !self.sorted {
            for entry in &self.data()[1..] {
                if maximum < entry.logit {
                    maximum = entry.logit;
                }
            }
        }
        if !maximum.is_finite() {
            let mut sum = 0.0_f32;
            for entry in self.data_mut() {
                entry.probability = if entry.logit.to_bits() == maximum.to_bits() {
                    1.0
                } else {
                    0.0
                };
                sum += entry.probability;
            }
            for entry in self.data_mut() {
                entry.probability /= sum;
            }
            return;
        }
        let mut sum = 0.0_f32;
        for entry in self.data_mut() {
            entry.probability = (entry.logit - maximum).exp();
            sum += entry.probability;
        }
        for entry in self.data_mut() {
            entry.probability /= sum;
        }
    }
}

fn descending(left: f32, right: f32) -> Ordering {
    right.partial_cmp(&left).unwrap_or(Ordering::Equal)
}

/// Production sampler configuration; probability and temperature fields are finite.
#[derive(Clone, Debug)]
pub enum Config {
    /// Static temperature; zero or negative masks all but the first maximum.
    Temperature(f32),
    /// Entropy-controlled temperature.
    ExtendedTemperature {
        /// Center temperature.
        temperature: f32,
        /// Dynamic range half-width, disabled when nonpositive.
        delta: f32,
        /// Exponent applied to normalized entropy.
        exponent: f32,
    },
    /// Stable-softmax categorical sampling with an MSVC-compatible seed.
    Distribution(u32),
    /// Largest-logit prefix; nonpositive values disable the filter.
    TopK(i32),
    /// Cumulative probability filter, retaining the crossing token.
    TopP {
        /// Probability threshold.
        probability: f32,
        /// Minimum retained count.
        min_keep: usize,
    },
    /// Probability relative to the maximum-probability token.
    MinP {
        /// Relative probability threshold.
        probability: f32,
        /// Minimum retained count.
        min_keep: usize,
    },
    /// Locally typical sampling by deviation from distribution entropy.
    Typical {
        /// Cumulative probability threshold.
        probability: f32,
        /// Minimum retained count.
        min_keep: usize,
    },
    /// Excludes all but the least likely token above a probability threshold.
    Xtc {
        /// Probability of applying the exclusion.
        probability: f32,
        /// Token probability threshold.
        threshold: f32,
        /// Minimum remaining count.
        min_keep: usize,
        /// MT19937 seed, with `UINT32_MAX` requesting fresh entropy.
        seed: u32,
    },
    /// History-based repetition, frequency and presence penalties.
    Penalties {
        /// Vocabulary capacity.
        vocabulary_size: usize,
        /// Retained token history; nonpositive values disable penalties.
        last_n: i32,
        /// Positive repetition penalty.
        repeat: f32,
        /// Count-dependent penalty.
        frequency: f32,
        /// Presence-dependent penalty.
        presence: f32,
    },
    /// DRY sequence repetition penalties with already tokenized breakers.
    Dry {
        /// Penalty multiplier.
        multiplier: f32,
        /// Exponential penalty base.
        base: f32,
        /// Number of repeated tokens permitted before penalizing continuation.
        allowed_length: usize,
        /// Retained token history; nonpositive values disable penalties.
        last_n: i32,
        /// Token sequences that reset repetition matching.
        sequence_breakers: Vec<Vec<i32>>,
    },
    /// Masks logits more than N standard deviations below the maximum.
    TopNSigma(f32),
}

impl Config {
    fn validate(&self) -> Result<()> {
        let (floats, keep) = match self {
            Self::Temperature(value) | Self::TopNSigma(value) => (vec![*value], 0),
            Self::ExtendedTemperature {
                temperature,
                delta,
                exponent,
            } => (vec![*temperature, *delta, *exponent], 0),
            Self::TopP {
                probability,
                min_keep,
            }
            | Self::MinP {
                probability,
                min_keep,
            }
            | Self::Typical {
                probability,
                min_keep,
            } => (vec![*probability], *min_keep),
            Self::Xtc {
                probability,
                threshold,
                min_keep,
                ..
            } => (vec![*probability, *threshold], *min_keep),
            Self::Penalties {
                vocabulary_size,
                last_n,
                repeat,
                frequency,
                presence,
            } => {
                if *vocabulary_size == 0
                    || *vocabulary_size > MAX_CANDIDATES
                    || *last_n < -1
                    || *last_n > 131_072
                    || *repeat <= 0.0
                {
                    return Err(invalid("invalid sampler penalty configuration"));
                }
                (vec![*repeat, *frequency, *presence], 0)
            }
            Self::Dry {
                multiplier,
                base,
                allowed_length,
                last_n,
                sequence_breakers,
            } => {
                if *allowed_length > MAX_CANDIDATES
                    || *last_n < -1
                    || *last_n > 131_072
                    || sequence_breakers.len() > 128
                    || sequence_breakers.iter().any(|breaker| breaker.len() > 1024)
                    || sequence_breakers.iter().map(Vec::len).sum::<usize>() > 8192
                {
                    return Err(invalid("invalid sampler DRY configuration"));
                }
                (vec![*multiplier, *base], 0)
            }
            Self::Distribution(_) | Self::TopK(_) => (vec![], 0),
        };
        if keep > MAX_CANDIDATES || floats.iter().any(|value| !value.is_finite()) {
            return Err(invalid("sampler parameters must be finite and bounded"));
        }
        Ok(())
    }
    const fn seed(&self) -> Option<u32> {
        match self {
            Self::Distribution(seed) | Self::Xtc { seed, .. } => Some(*seed),
            _ => None,
        }
    }
}

/// Stateful production sampler, with independent clone, reset and accept state.
#[derive(Clone)]
pub struct State {
    config: Config,
    random: Option<Box<Random>>,
    history: VecDeque<i32>,
}

impl State {
    // The fixed Swift chain supplies an already sorted, finite top-20 view.
    // TopP and Distribution use the same maximum and binary32 exponential;
    // retain those unnormalized weights before TopP divides by its f32 sum.
    // Distribution must still sum the retained values in f64, draw once even
    // for a singleton, and normalize in its original order. Generic apply and
    // unsorted candidates keep the independent, uncached implementation below.
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "Preserve the native distribution's f64 division and final f32 rounding."
    )]
    pub(super) fn apply_sorted_top_p_distribution(
        &mut self,
        filter: &Self,
        candidates: &mut Candidates,
        weights: &mut [f32],
    ) -> Result<()> {
        let Config::TopP {
            probability,
            min_keep,
        } = filter.config
        else {
            return Err(invalid("cached distribution requires a top-p filter"));
        };
        if !matches!(self.config, Config::Distribution(_))
            || !candidates.sorted
            || candidates.size == 0
            || candidates.size > weights.len()
            || probability >= 1.0
        {
            return Err(invalid(
                "cached distribution requires bounded sorted top-p candidates",
            ));
        }
        let random = self
            .random
            .as_mut()
            .ok_or_else(|| invalid("missing sampler random state"))?;
        let maximum = candidates.data()[0].logit;
        let mut sum = 0.0_f32;
        if maximum.is_finite() {
            for (entry, weight) in candidates.data_mut().iter_mut().zip(weights.iter_mut()) {
                *weight = (entry.logit - maximum).exp();
                entry.probability = *weight;
                sum += *weight;
            }
        } else {
            for (entry, weight) in candidates.data_mut().iter_mut().zip(weights.iter_mut()) {
                *weight = if entry.logit.to_bits() == maximum.to_bits() {
                    1.0
                } else {
                    0.0
                };
                entry.probability = *weight;
                sum += *weight;
            }
            if sum <= 0.0 || !sum.is_finite() {
                sum = 0.0;
                for (entry, weight) in candidates.data_mut().iter_mut().zip(weights.iter_mut()) {
                    *weight = 1.0;
                    entry.probability = 1.0;
                    sum += 1.0;
                }
            }
        }
        for entry in candidates.data_mut() {
            entry.probability /= sum;
        }
        let mut cumulative = 0.0_f32;
        let mut size = candidates.size;
        for (index, entry) in candidates.data().iter().enumerate() {
            cumulative += entry.probability;
            if cumulative >= probability && index + 1 >= min_keep {
                size = index + 1;
                break;
            }
        }
        candidates.size = size;
        candidates.selected = 0;
        let draw = random.double();
        if candidates.size == 1 {
            candidates.data_mut()[0].probability = 1.0;
            return Ok(());
        }
        let weights = &weights[..candidates.size];
        let mut sum = 0.0_f64;
        for &weight in weights {
            sum += f64::from(weight);
        }
        let target = sum * draw;
        let mut cumulative = 0.0_f64;
        let mut selected = None;
        for (index, (entry, &weight)) in candidates.data_mut().iter_mut().zip(weights).enumerate() {
            if selected.is_none() {
                cumulative += f64::from(weight);
                if cumulative >= target {
                    selected = Some(index);
                }
            }
            entry.probability = (f64::from(weight) / sum) as f32;
        }
        candidates.selected = i64::try_from(selected.unwrap_or(candidates.size - 1))?;
        Ok(())
    }

    /// Collects sorted `(token_id, dry_penalty)` pairs for touched history tokens in `O(|history|)`.
    pub(super) fn collect_dry_penalties(&self, out: &mut Vec<(i32, f32)>) {
        out.clear();
        if let Config::Dry {
            multiplier,
            base,
            allowed_length,
            last_n,
            sequence_breakers,
        } = &self.config
            && *last_n > 0
            && *multiplier != 0.0
            && *base >= 1.0
        {
            collect_dry_penalties_from_history(
                &self.history,
                *multiplier,
                *base,
                *allowed_length,
                sequence_breakers,
                out,
            );
        }
    }

    /// Collects sorted `(token_id, count)` pairs and penalty parameters in `O(|history|)`.
    #[expect(
        clippy::float_cmp,
        reason = "Native penalty identity values disable exactly, without an epsilon domain change."
    )]
    pub(super) fn collect_repetition_penalties(
        &self,
        out: &mut Vec<(i32, usize)>,
    ) -> Option<(f32, f32, f32)> {
        out.clear();
        if let Config::Penalties {
            last_n,
            repeat,
            frequency,
            presence,
            ..
        } = self.config
            && last_n > 0
            && (repeat != 1.0 || frequency != 0.0 || presence != 0.0)
            && !self.history.is_empty()
        {
            collect_repetition_counts(&self.history, out);
            return Some((repeat, frequency, presence));
        }
        None
    }

    /// Validates configuration and initializes independent random/history state.
    ///
    /// # Errors
    /// Rejects nonfinite or out-of-bound sampler parameters.
    pub fn new(config: Config) -> Result<Self> {
        config.validate()?;
        let random = config.seed().map(|seed| Box::new(Random::new(seed)));
        Ok(Self {
            config,
            random,
            history: VecDeque::new(),
        })
    }

    /// Clears accepted history and reseeds random generators from their initial seed.
    pub fn reset(&mut self) {
        self.history.clear();
        self.random = self.config.seed().map(|seed| Box::new(Random::new(seed)));
    }

    /// Accepts one token into the bounded repetition history.
    pub fn accept(&mut self, token: i32) {
        let last = match self.config {
            Config::Penalties { last_n, .. } | Config::Dry { last_n, .. } => {
                usize::try_from(last_n).unwrap_or(0)
            }
            _ => 0,
        };
        if last > 0 {
            if self.history.len() == last {
                self.history.pop_front();
            }
            self.history.push_back(token);
        }
    }

    /// Applies the configured sampler to the owned active candidate view.
    ///
    /// # Errors
    /// Rejects an empty active candidate view or internal invalid random state.
    #[expect(
        clippy::float_cmp,
        reason = "Native penalty identity values disable exactly, without an epsilon domain change."
    )]
    pub fn apply(&mut self, candidates: &mut Candidates) -> Result<()> {
        if candidates.data().is_empty() {
            return Err(invalid("cannot apply a sampler to an empty view"));
        }
        match &self.config {
            Config::Temperature(temperature) => temperature_filter(candidates, *temperature),
            Config::ExtendedTemperature {
                temperature,
                delta,
                exponent,
            } => extended_temperature(candidates, *temperature, *delta, *exponent),
            Config::Distribution(_) => distribution(
                candidates,
                self.random
                    .as_mut()
                    .ok_or_else(|| invalid("missing sampler random state"))?,
            ),
            Config::TopK(k) => {
                if *k > 0 {
                    if !candidates.sorted {
                        let keep = candidates.size.min(usize::try_from(*k)?);
                        if keep <= 128 {
                            partial_top_k(candidates.data_mut(), keep);
                            candidates.sorted = true;
                        } else {
                            candidates.sort();
                        }
                    }
                    candidates.size = candidates.size.min(usize::try_from(*k)?);
                }
            }
            Config::TopP {
                probability,
                min_keep,
            } => top_p(candidates, *probability, *min_keep),
            Config::MinP {
                probability,
                min_keep,
            } => min_p(candidates, *probability, *min_keep),
            Config::Typical {
                probability,
                min_keep,
            } => typical(candidates, *probability, *min_keep),
            Config::Xtc {
                probability,
                threshold,
                min_keep,
                ..
            } => xtc(
                candidates,
                *probability,
                *threshold,
                *min_keep,
                self.random
                    .as_mut()
                    .ok_or_else(|| invalid("missing sampler random state"))?,
            ),
            Config::Penalties {
                last_n,
                repeat,
                frequency,
                presence,
                ..
            } => {
                if *last_n > 0 && (*repeat != 1.0 || *frequency != 0.0 || *presence != 0.0) {
                    penalties(candidates, &self.history, *repeat, *frequency, *presence);
                }
            }
            Config::Dry {
                multiplier,
                base,
                allowed_length,
                last_n,
                sequence_breakers,
            } => {
                if *last_n > 0 && *multiplier != 0.0 && *base >= 1.0 {
                    dry(
                        candidates,
                        &self.history,
                        *multiplier,
                        *base,
                        *allowed_length,
                        sequence_breakers,
                    );
                }
            }
            Config::TopNSigma(n) => top_n_sigma(candidates, *n),
        }
        Ok(())
    }
}

// The pinned small-k CPU path uses libstdc++ partial_sort's heap schedule.
// Equal logits compare neither way: child choice, replacement in original token
// order and sort_heap therefore all matter to seeded token identity. A Rust
// unstable full sort has different ties even for a completely uniform input.
fn partial_top_k(data: &mut [Candidate], keep: usize) {
    make_top_k_heap(&mut data[..keep]);
    for index in keep..data.len() {
        if data[index].logit > data[0].logit {
            let value = data[index];
            data[index] = data[0];
            adjust_heap(data, 0, keep, value);
        }
    }
    sort_top_k_heap(&mut data[..keep]);
}

// Validate the complete finite vocabulary during this bounded selection.
// Preserve partial_top_k's initial heap, strict comparison, token traversal and
// final heap sort, including equal logits and signed zero. Discarded tail entries
// are never read again by partial_top_k, so they need no Candidate storage here.
pub(super) fn top_k_logits(
    logits: &[f32],
    keep: usize,
    backing: &mut Vec<Candidate>,
) -> Result<()> {
    top_k_logits_inner::<true>(logits, keep, &[], backing)
}

pub(super) fn top_k_finite_logits(
    logits: super::FiniteLogits<'_>,
    keep: usize,
    backing: &mut Vec<Candidate>,
) -> Result<()> {
    top_k_logits_inner::<false>(logits.as_slice(), keep, &[], backing)
}

pub(super) fn top_k_adjusted_logits<const VALIDATE: bool>(
    logits: &[f32],
    keep: usize,
    adjusted: &[(usize, f32)],
    backing: &mut Vec<Candidate>,
) -> Result<()> {
    top_k_logits_inner::<VALIDATE>(logits, keep, adjusted, backing)
}

fn init_top_k_heap<const VALIDATE: bool>(
    logits: &[f32],
    keep: usize,
    adjusted: &[(usize, f32)],
    backing: &mut Vec<Candidate>,
) -> Result<usize> {
    backing.clear();
    backing.try_reserve_exact(keep)?;
    let mut t_cursor = 0_usize;
    for (index, &raw_logit) in logits[..keep].iter().enumerate() {
        if VALIDATE && !raw_logit.is_finite() {
            backing.clear();
            return Err(invalid(
                "Swift sampling requires the complete finite pinned vocabulary",
            ));
        }
        while t_cursor < adjusted.len() && adjusted[t_cursor].0 < index {
            t_cursor += 1;
        }
        let logit = if t_cursor < adjusted.len() && adjusted[t_cursor].0 == index {
            let adj = adjusted[t_cursor].1;
            t_cursor += 1;
            adj
        } else {
            raw_logit
        };
        backing.push(Candidate {
            id: i32::try_from(index)?,
            logit,
            probability: 0.0,
        });
    }
    make_top_k_heap(backing);
    Ok(t_cursor)
}

fn top_k_logits_inner<const VALIDATE: bool>(
    logits: &[f32],
    keep: usize,
    adjusted: &[(usize, f32)],
    backing: &mut Vec<Candidate>,
) -> Result<()> {
    if keep == 0 || keep > 128 || keep > logits.len() || logits.len() > MAX_CANDIDATES {
        return Err(invalid("bounded logit top-k extent is invalid"));
    }
    let mut t_cursor = init_top_k_heap::<VALIDATE>(logits, keep, adjusted, backing)?;
    let tail = &logits[keep..];
    let (blocks, remainder) = tail.as_chunks::<16>();
    for (block_index, block) in blocks.iter().enumerate() {
        let base = keep + block_index * 16;
        let mut lanes = super::selection::candidate_lanes::<VALIDATE>(block, backing[0].logit);
        let t_block_start = t_cursor;
        while t_cursor < adjusted.len() && adjusted[t_cursor].0 < base + 16 {
            lanes |= 1_u64 << (adjusted[t_cursor].0 - base);
            t_cursor += 1;
        }
        let touched_in_block = &adjusted[t_block_start..t_cursor];
        while lanes != 0 {
            let offset = usize::try_from(lanes.trailing_zeros())?;
            lanes &= lanes - 1;
            let raw_logit = block[offset];
            if VALIDATE && !raw_logit.is_finite() {
                backing.clear();
                return Err(invalid(
                    "Swift sampling requires the complete finite pinned vocabulary",
                ));
            }
            let token_idx = base + offset;
            let logit = if touched_in_block.is_empty() {
                raw_logit
            } else {
                touched_in_block
                    .iter()
                    .find(|&&(id, _)| id == token_idx)
                    .map_or(raw_logit, |&(_, adj)| adj)
            };
            if logit > backing[0].logit {
                let value = Candidate {
                    id: i32::try_from(token_idx)?,
                    logit,
                    probability: 0.0,
                };
                adjust_heap(backing, 0, keep, value);
            }
        }
    }
    let base = keep + blocks.len() * 16;
    for (offset, &raw_logit) in remainder.iter().enumerate() {
        if VALIDATE && !raw_logit.is_finite() {
            backing.clear();
            return Err(invalid(
                "Swift sampling requires the complete finite pinned vocabulary",
            ));
        }
        let token_idx = base + offset;
        while t_cursor < adjusted.len() && adjusted[t_cursor].0 < token_idx {
            t_cursor += 1;
        }
        let logit = if t_cursor < adjusted.len() && adjusted[t_cursor].0 == token_idx {
            let adj = adjusted[t_cursor].1;
            t_cursor += 1;
            adj
        } else {
            raw_logit
        };
        if logit > backing[0].logit {
            let value = Candidate {
                id: i32::try_from(token_idx)?,
                logit,
                probability: 0.0,
            };
            adjust_heap(backing, 0, keep, value);
        }
    }
    sort_top_k_heap(backing);
    Ok(())
}

fn make_top_k_heap(data: &mut [Candidate]) {
    let keep = data.len();
    if keep > 1 {
        for parent in (0..=(keep - 2) / 2).rev() {
            let value = data[parent];
            adjust_heap(data, parent, keep, value);
        }
    }
}

fn sort_top_k_heap(data: &mut [Candidate]) {
    for end in (1..data.len()).rev() {
        let value = data[end];
        data[end] = data[0];
        adjust_heap(data, 0, end, value);
    }
}

fn adjust_heap(data: &mut [Candidate], mut hole: usize, length: usize, value: Candidate) {
    let top = hole;
    let mut child = hole;
    while child < (length - 1) / 2 {
        child = 2 * (child + 1);
        if data[child].logit > data[child - 1].logit {
            child -= 1;
        }
        data[hole] = data[child];
        hole = child;
    }
    if length.is_multiple_of(2) && child == (length - 2) / 2 {
        child = 2 * (child + 1);
        data[hole] = data[child - 1];
        hole = child - 1;
    }
    while hole > top {
        let parent = (hole - 1) / 2;
        if data[parent].logit.partial_cmp(&value.logit) != Some(Ordering::Greater) {
            break;
        }
        data[hole] = data[parent];
        hole = parent;
    }
    data[hole] = value;
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Overflow-guarded extreme-logit temperature division narrows finite double differences into binary32."
)]
fn temperature_filter(candidates: &mut Candidates, temperature: f32) {
    if temperature <= 0.0 {
        let mut maximum = 0;
        for index in 1..candidates.size {
            if candidates.data()[index].logit > candidates.data()[maximum].logit {
                candidates.data_mut()[maximum].logit = f32::NEG_INFINITY;
                maximum = index;
            } else {
                candidates.data_mut()[index].logit = f32::NEG_INFINITY;
            }
        }
        return;
    }
    if temperature.to_bits() == 1.0_f32.to_bits() {
        return;
    }
    let mut maximum = f32::NEG_INFINITY;
    let mut overflows_f32 = false;
    let max_f32 = f64::from(f32::MAX);
    let temp_f64 = f64::from(temperature);
    for entry in candidates.data() {
        if entry.logit > maximum {
            maximum = entry.logit;
        }
        if entry.logit.is_finite() {
            let scaled = f64::from(entry.logit) / temp_f64;
            if !scaled.is_finite() || scaled.abs() > max_f32 {
                overflows_f32 = true;
            }
        }
    }
    if overflows_f32 && maximum.is_finite() {
        let max_f64 = f64::from(maximum);
        for entry in candidates.data_mut() {
            if entry.logit.is_finite() {
                entry.logit = ((f64::from(entry.logit) - max_f64) / temp_f64) as f32;
            }
        }
    } else {
        for entry in candidates.data_mut() {
            entry.logit /= temperature;
        }
    }
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "Preserve native binary32 size conversions and double-to-float probability stores."
)]
fn extended_temperature(candidates: &mut Candidates, temperature: f32, delta: f32, exponent: f32) {
    if delta <= 0.0 {
        temperature_filter(candidates, temperature);
        return;
    }
    if candidates.size <= 1 {
        return;
    }
    let minimum = 0.0_f32.max(temperature - delta);
    let maximum = temperature + delta;
    let maximum_entropy = -(1.0_f32 / candidates.size as f32).ln();
    candidates.softmax(true);
    let mut entropy = 0.0;
    for entry in candidates.data() {
        if entry.probability > 0.0 {
            entropy -= entry.probability * entry.probability.ln();
        }
    }
    let dynamic = minimum + (maximum - minimum) * (entropy / maximum_entropy).powf(exponent);
    temperature_filter(candidates, dynamic);
    let max_logit = candidates.data()[0].logit;
    if !max_logit.is_finite() {
        let mut sum = 0.0_f64;
        for entry in candidates.data_mut() {
            let weight = if entry.logit.to_bits() == max_logit.to_bits() {
                1.0
            } else {
                0.0
            };
            entry.probability = weight as f32;
            sum += weight;
        }
        for entry in candidates.data_mut() {
            entry.probability = (f64::from(entry.probability) / sum) as f32;
        }
        return;
    }
    let max_logit_f64 = f64::from(max_logit);
    let mut sum = 0.0_f64;
    for entry in candidates.data_mut() {
        let probability = (f64::from(entry.logit) - max_logit_f64).exp();
        entry.probability = probability as f32;
        sum += probability;
    }
    for entry in candidates.data_mut() {
        entry.probability = (f64::from(entry.probability) / sum) as f32;
    }
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Native distribution stores double-normalized probabilities into binary32."
)]
fn distribution(candidates: &mut Candidates, random: &mut Random) {
    candidates.selected = 0;
    let draw = random.double();
    if candidates.size == 1 {
        candidates.data_mut()[0].probability = 1.0;
        return;
    }
    let mut maximum = candidates.data()[0].logit;
    if !candidates.sorted {
        for entry in &candidates.data()[1..] {
            if maximum < entry.logit {
                maximum = entry.logit;
            }
        }
    }
    let mut sum = 0.0_f64;
    if maximum.is_finite() {
        for entry in candidates.data_mut() {
            entry.probability = (entry.logit - maximum).exp();
            sum += f64::from(entry.probability);
        }
    } else {
        for entry in candidates.data_mut() {
            let weight = if entry.logit.to_bits() == maximum.to_bits() {
                1.0
            } else {
                0.0
            };
            entry.probability = weight as f32;
            sum += weight;
        }
    }
    let target = sum * draw;
    let mut cumulative = 0.0_f64;
    let mut selected = None;
    for (index, entry) in candidates.data_mut().iter_mut().enumerate() {
        if selected.is_none() {
            cumulative += f64::from(entry.probability);
            if cumulative >= target {
                selected = Some(index);
            }
        }
        entry.probability = (f64::from(entry.probability) / sum) as f32;
    }
    candidates.selected = i64::try_from(selected.unwrap_or(candidates.size - 1)).unwrap_or(-1);
}

fn top_p(candidates: &mut Candidates, probability: f32, min_keep: usize) {
    if probability >= 1.0 {
        return;
    }
    candidates.softmax(false);
    if !candidates.sorted {
        candidates.sort();
    }
    let mut cumulative = 0.0_f32;
    let mut size = candidates.size;
    for (index, entry) in candidates.data().iter().enumerate() {
        cumulative += entry.probability;
        if cumulative >= probability && index + 1 >= min_keep {
            size = index + 1;
            break;
        }
    }
    candidates.size = size;
}

fn min_p(candidates: &mut Candidates, probability: f32, min_keep: usize) {
    if probability <= 0.0 {
        return;
    }
    let threshold = f64::from(probability).ln();
    if !candidates.sorted {
        let mut maximum = -f32::MAX;
        for entry in candidates.data() {
            if maximum < entry.logit {
                maximum = entry.logit;
            }
        }
        let max_f64 = f64::from(maximum);
        let filtered = candidates
            .data()
            .iter()
            .filter(|entry| {
                entry.logit.to_bits() == maximum.to_bits()
                    || f64::from(entry.logit) - max_f64 >= threshold
            })
            .copied()
            .collect::<Vec<_>>();
        if !filtered.is_empty() && filtered.len() >= min_keep {
            candidates.data_mut()[..filtered.len()].copy_from_slice(&filtered);
            candidates.size = filtered.len();
            return;
        }
        candidates.sort();
    }
    let maximum = candidates.data()[0].logit;
    let max_f64 = f64::from(maximum);
    let mut size = 1;
    while size < candidates.size {
        let logit = candidates.data()[size].logit;
        if logit.to_bits() != maximum.to_bits()
            && f64::from(logit) - max_f64 < threshold
            && size >= min_keep
        {
            break;
        }
        size += 1;
    }
    candidates.size = size;
}

fn typical(candidates: &mut Candidates, probability: f32, min_keep: usize) {
    if probability >= 1.0 {
        return;
    }
    candidates.softmax(true);
    let mut entropy = 0.0_f32;
    for entry in candidates.data() {
        if entry.probability > 0.0 {
            entropy += -entry.probability * entry.probability.ln();
        }
    }
    let scores = candidates
        .data()
        .iter()
        .map(|entry| {
            if entry.probability > 0.0 {
                (-entry.probability.ln() - entropy).abs()
            } else {
                f32::INFINITY
            }
        })
        .collect::<Vec<_>>();
    let mut indices = (0..candidates.size).collect::<Vec<_>>();
    indices.sort_unstable_by(|&left, &right| {
        scores[left]
            .partial_cmp(&scores[right])
            .unwrap_or(Ordering::Equal)
    });
    let mut cumulative = 0.0_f32;
    let mut size = indices.len();
    for (index, &position) in indices.iter().enumerate() {
        cumulative += candidates.data()[position].probability;
        if cumulative > probability && index + 1 >= min_keep {
            size = index + 1;
            break;
        }
    }
    let filtered = indices[..size]
        .iter()
        .map(|&index| candidates.data()[index])
        .collect::<Vec<_>>();
    candidates.data_mut()[..size].copy_from_slice(&filtered);
    candidates.size = size;
    candidates.sorted = false;
}

fn xtc(
    candidates: &mut Candidates,
    probability: f32,
    threshold: f32,
    min_keep: usize,
    random: &mut Random,
) {
    if probability <= 0.0 || threshold > 0.5 || candidates.size < 2 || random.float() > probability
    {
        return;
    }
    candidates.softmax(true);
    let mut last = 0;
    for (index, entry) in candidates.data().iter().enumerate() {
        if entry.probability >= threshold {
            last = index;
        } else {
            break;
        }
    }
    if last > 0 && candidates.size - last >= min_keep {
        candidates.offset += last;
        candidates.size -= last;
    }
}

fn build_token_bloom<T>(entries: &[(i32, T)]) -> [u64; 32] {
    let mut bloom = [0_u64; 32];
    for &(id, _) in entries {
        let bits = u32::from_ne_bytes(id.to_ne_bytes());
        let word = usize::from(u8::try_from((bits >> 6) & 31).unwrap_or(0));
        bloom[word] |= 1_u64 << (bits & 63);
    }
    bloom
}

#[inline]
fn bloom_contains(bloom: &[u64; 32], id: i32) -> bool {
    let bits = u32::from_ne_bytes(id.to_ne_bytes());
    let word = usize::from(u8::try_from((bits >> 6) & 31).unwrap_or(0));
    (bloom[word] & (1_u64 << (bits & 63))) != 0
}

fn collect_repetition_counts(history: &VecDeque<i32>, out: &mut Vec<(i32, usize)>) {
    out.clear();
    if history.is_empty() {
        return;
    }
    let mut tokens: Vec<i32> = history.iter().copied().collect();
    tokens.sort_unstable();
    let mut prev = tokens[0];
    let mut count = 1_usize;
    for &tok in &tokens[1..] {
        if tok == prev {
            count += 1;
        } else {
            out.push((prev, count));
            prev = tok;
            count = 1;
        }
    }
    out.push((prev, count));
}

#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Bounded native integer token counts convert to binary32 before penalties."
)]
fn penalties(
    candidates: &mut Candidates,
    history: &VecDeque<i32>,
    repeat: f32,
    frequency: f32,
    presence: f32,
) {
    let mut counts = Vec::new();
    collect_repetition_counts(history, &mut counts);
    if counts.is_empty() {
        candidates.sorted = false;
        return;
    }
    let bloom = build_token_bloom(&counts);
    for entry in candidates.data_mut() {
        if bloom_contains(&bloom, entry.id)
            && let Ok(pos) = counts.binary_search_by_key(&entry.id, |&(id, _)| id)
        {
            let count = counts[pos].1;
            if entry.logit <= 0.0 {
                entry.logit *= repeat;
            } else {
                entry.logit /= repeat;
            }
            entry.logit -= count as f32 * frequency + presence;
        }
    }
    candidates.sorted = false;
}

fn top_n_sigma(candidates: &mut Candidates, n: f32) {
    if n <= 0.0 || candidates.size <= 1 {
        return;
    }
    let mut maximum = candidates.data()[0].logit;
    let mut sum = 0.0_f64;
    let mut count = 0_u32;
    for entry in candidates.data() {
        if entry.logit != f32::NEG_INFINITY {
            if maximum < entry.logit {
                maximum = entry.logit;
            }
            sum += f64::from(entry.logit);
            count += 1;
        }
    }
    let count_f64 = f64::from(count);
    let mean = if count > 0 { sum / count_f64 } else { 0.0 };
    let mut accumulator = 0.0_f64;
    for entry in candidates.data() {
        if entry.logit != f32::NEG_INFINITY {
            let delta = f64::from(entry.logit) - mean;
            accumulator = delta.mul_add(delta, accumulator);
        }
    }
    let deviation = if count > 0 {
        (accumulator / count_f64).sqrt()
    } else {
        0.0
    };
    let threshold = (-f64::from(n)).mul_add(deviation, f64::from(maximum));
    for entry in candidates.data_mut() {
        if f64::from(entry.logit) < threshold {
            entry.logit = f32::NEG_INFINITY;
        }
    }
}

fn repetition_limit(reversed: &[i32], breakers: &[Vec<i32>]) -> usize {
    for (index, token) in reversed.iter().enumerate() {
        let mut longest = None;
        for breaker in breakers {
            if breaker.first() == Some(token) {
                let tail = &breaker[1..];
                if tail.len() <= index
                    && tail
                        .iter()
                        .enumerate()
                        .all(|(offset, token)| *token == reversed[index - offset - 1])
                {
                    longest = Some(longest.unwrap_or(0).max(tail.len()));
                }
            }
        }
        if let Some(longest) = longest {
            return index - longest;
        }
    }
    reversed.len()
}

fn repeat_lengths(reversed: &[i32], limit: usize) -> Vec<usize> {
    let mut lengths = vec![0; reversed.len()];
    let last = reversed.len() - 1;
    let mut right = 0;
    let mut left = 0;
    for offset in 1..reversed.len() {
        if offset > right {
            let mut length = 0;
            while length + offset < reversed.len() && reversed[length] == reversed[length + offset]
            {
                length += 1;
            }
            lengths[last - offset] = length.min(limit);
            if length > 0 {
                left = offset;
                right = offset + length - 1;
            }
        } else {
            let pair = offset - left;
            let remaining = right - offset + 1;
            if lengths[last - pair] < remaining {
                lengths[last - offset] = lengths[last - pair].min(limit);
            } else {
                let mut index = right + 1;
                while index < reversed.len() && reversed[index] == reversed[index - offset] {
                    index += 1;
                }
                lengths[last - offset] = (index - offset).min(limit);
                left = offset;
                right = index - 1;
            }
        }
    }
    lengths
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "Native DRY bounds its finite logarithmic exponent before int truncation and float pow."
)]
fn collect_dry_penalties_from_history(
    history: &VecDeque<i32>,
    multiplier: f32,
    base: f32,
    allowed: usize,
    breakers: &[Vec<i32>],
    out: &mut Vec<(i32, f32)>,
) {
    out.clear();
    if history.len() <= allowed {
        return;
    }
    let reversed = history.iter().rev().copied().collect::<Vec<_>>();
    let limit = repetition_limit(&reversed, breakers);
    if limit < allowed {
        return;
    }
    let lengths = repeat_lengths(&reversed, limit);
    let mut raw_hits: Vec<(i32, usize)> = Vec::new();
    for (index, &length) in lengths[..lengths.len() - 1].iter().enumerate() {
        if length >= allowed {
            let token = reversed[lengths.len() - 2 - index];
            raw_hits.push((token, length));
        }
    }
    if raw_hits.is_empty() {
        return;
    }
    raw_hits.sort_unstable_by_key(|&(id, _)| id);

    let mut single_breakers: Vec<i32> = breakers
        .iter()
        .filter(|b| b.len() == 1)
        .map(|b| b[0])
        .collect();
    single_breakers.sort_unstable();
    single_breakers.dedup();

    let max_exponent = if base > 1.000_001 {
        (88.722_84_f32 / base.ln()) as usize
    } else {
        0
    };
    let mut idx = 0_usize;
    while idx < raw_hits.len() {
        let token = raw_hits[idx].0;
        let mut max_len = raw_hits[idx].1;
        idx += 1;
        while idx < raw_hits.len() && raw_hits[idx].0 == token {
            max_len = max_len.max(raw_hits[idx].1);
            idx += 1;
        }
        if single_breakers.binary_search(&token).is_ok() {
            continue;
        }
        let mut exponent = max_len - allowed;
        if max_exponent > 0 {
            exponent = exponent.min(max_exponent);
        }
        let penalty = (f64::from(multiplier)
            * f64::from(base).powi(i32::try_from(exponent).unwrap_or(i32::MAX)))
        .min(f64::from(f32::MAX)) as f32;
        out.push((token, penalty));
    }
}

fn dry(
    candidates: &mut Candidates,
    history: &VecDeque<i32>,
    multiplier: f32,
    base: f32,
    allowed: usize,
    breakers: &[Vec<i32>],
) {
    let mut penalties = Vec::new();
    collect_dry_penalties_from_history(
        history,
        multiplier,
        base,
        allowed,
        breakers,
        &mut penalties,
    );
    if history.len() <= allowed {
        return;
    }
    if penalties.is_empty() {
        let reversed = history.iter().rev().copied().collect::<Vec<_>>();
        if repetition_limit(&reversed, breakers) >= allowed {
            candidates.sorted = false;
        }
        return;
    }
    let bloom = build_token_bloom(&penalties);
    for entry in candidates.data_mut() {
        if bloom_contains(&bloom, entry.id)
            && let Ok(pos) = penalties.binary_search_by_key(&entry.id, |&(id, _)| id)
        {
            entry.logit -= penalties[pos].1;
        }
    }
    candidates.sorted = false;
}

#[derive(Clone)]
pub(crate) struct Random {
    state: [u32; 624],
    index: usize,
}
impl Random {
    pub(crate) fn new(seed: u32) -> Self {
        let seed = if seed == u32::MAX {
            let mut hash = std::collections::hash_map::RandomState::new().build_hasher();
            hash.write_u32(std::process::id());
            let bytes = hash.finish().to_le_bytes();
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
        } else {
            seed
        };
        let mut state = [0; 624];
        state[0] = seed;
        for (index, increment) in (1..624).zip(1_u32..) {
            state[index] = 1_812_433_253_u32
                .wrapping_mul(state[index - 1] ^ (state[index - 1] >> 30))
                .wrapping_add(increment);
        }
        Self { state, index: 624 }
    }
    fn next(&mut self) -> u32 {
        if self.index == 624 {
            for index in 0..624 {
                let value = (self.state[index] & 0x8000_0000)
                    | (self.state[(index + 1) % 624] & 0x7fff_ffff);
                self.state[index] = self.state[(index + 397) % 624]
                    ^ (value >> 1)
                    ^ if value & 1 == 0 { 0 } else { 0x9908_b0df };
            }
            self.index = 0;
        }
        let mut value = self.state[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^ (value >> 18)
    }
    pub(crate) fn bounded(&mut self, bound: usize) -> Result<usize> {
        let bound = u32::try_from(bound)?;
        if bound == 0 {
            return Err(invalid("random integer bound must be positive"));
        }
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let product = u64::from(self.next()) * u64::from(bound);
            let low = u32::from_le_bytes(product.to_le_bytes()[..4].try_into()?);
            if low >= threshold {
                return Ok(usize::try_from(product >> 32)?);
            }
        }
    }
    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "24-bit and 53-bit canonical integers are exactly representable in their destination floats."
    )]
    fn float(&mut self) -> f32 {
        (self.next() >> 8) as f32 * (1.0 / 16_777_216.0)
    }
    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "MSVC canonical double uses low21+high32 bits, yielding an exact 53-bit integer."
    )]
    fn double(&mut self) -> f64 {
        let bits = u64::from(self.next() >> 11) + (u64::from(self.next()) << 21);
        bits as f64 * (1.0 / 9_007_199_254_740_992.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_rejects_out_of_range_probabilities() {
        for invalid_prob in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                State::new(Config::TopP {
                    probability: invalid_prob,
                    min_keep: 1,
                })
                .is_err()
            );
            assert!(
                State::new(Config::MinP {
                    probability: invalid_prob,
                    min_keep: 1,
                })
                .is_err()
            );
            assert!(
                State::new(Config::Typical {
                    probability: invalid_prob,
                    min_keep: 1,
                })
                .is_err()
            );
            assert!(
                State::new(Config::Xtc {
                    probability: invalid_prob,
                    threshold: 0.1,
                    min_keep: 1,
                    seed: 0,
                })
                .is_err()
            );
        }
        for valid_prob in [-0.1_f32, 0.0, 1.0, 1.05] {
            assert!(
                State::new(Config::TopP {
                    probability: valid_prob,
                    min_keep: 1,
                })
                .is_ok()
            );
            assert!(
                State::new(Config::MinP {
                    probability: valid_prob,
                    min_keep: 1,
                })
                .is_ok()
            );
            assert!(
                State::new(Config::Typical {
                    probability: valid_prob,
                    min_keep: 1,
                })
                .is_ok()
            );
            assert!(
                State::new(Config::Xtc {
                    probability: valid_prob,
                    threshold: 0.1,
                    min_keep: 1,
                    seed: 0,
                })
                .is_ok()
            );
        }
    }

    #[test]
    fn candidates_sort_breaks_equal_logit_ties_by_ascending_id() -> Result<()> {
        let mut candidates = Candidates::new(
            vec![
                Candidate {
                    id: 9,
                    logit: 2.0,
                    probability: 0.0,
                },
                Candidate {
                    id: 2,
                    logit: 2.0,
                    probability: 0.0,
                },
                Candidate {
                    id: 5,
                    logit: 3.0,
                    probability: 0.0,
                },
                Candidate {
                    id: 4,
                    logit: 2.0,
                    probability: 0.0,
                },
            ],
            -1,
            false,
        )?;
        candidates.sort();
        let ids: Vec<i32> = candidates.data().iter().map(|entry| entry.id).collect();
        assert_eq!(ids, [5, 2, 4, 9]);
        Ok(())
    }

    #[test]
    fn extreme_logits_temperature_top_n_sigma_and_min_p_avoid_f32_overflow() -> Result<()> {
        let mut extreme = Candidates::new(
            vec![
                Candidate {
                    id: 0,
                    logit: 3.0e38,
                    probability: 0.0,
                },
                Candidate {
                    id: 1,
                    logit: 2.0e38,
                    probability: 0.0,
                },
            ],
            -1,
            false,
        )?;
        State::new(Config::Temperature(0.1))?.apply(&mut extreme)?;
        State::new(Config::Distribution(7))?.apply(&mut extreme)?;
        assert!(
            extreme
                .data()
                .iter()
                .all(|entry| entry.probability.is_finite())
        );
        assert!((extreme.data()[0].probability - 1.0).abs() <= 1e-6);

        let mut sigma = Candidates::new(
            vec![
                Candidate {
                    id: 0,
                    logit: -2.0e20,
                    probability: 0.0,
                },
                Candidate {
                    id: 1,
                    logit: 2.0e20,
                    probability: 0.0,
                },
                Candidate {
                    id: 2,
                    logit: 2.0e20,
                    probability: 0.0,
                },
            ],
            -1,
            false,
        )?;
        State::new(Config::TopNSigma(1.0))?.apply(&mut sigma)?;
        assert_eq!(sigma.data()[0].logit, f32::NEG_INFINITY);
        assert_eq!(sigma.data()[1].logit, 2.0e20);
        assert_eq!(sigma.data()[2].logit, 2.0e20);

        let mut min_p = Candidates::new(
            vec![
                Candidate {
                    id: 0,
                    logit: -3.4e38,
                    probability: 0.0,
                },
                Candidate {
                    id: 1,
                    logit: -3.4e38,
                    probability: 0.0,
                },
            ],
            -1,
            false,
        )?;
        State::new(Config::MinP {
            probability: 0.5,
            min_keep: 1,
        })?
        .apply(&mut min_p)?;
        assert_eq!(min_p.data().len(), 2);
        Ok(())
    }

    #[test]
    fn typical_dry_and_top_p_distribution_guard_nonfinite_and_zero_prob() -> Result<()> {
        let mut typical_candidates = Candidates::new(
            vec![
                Candidate {
                    id: 1,
                    logit: 2.0,
                    probability: 0.0,
                },
                Candidate {
                    id: 2,
                    logit: 1.0,
                    probability: 0.0,
                },
                Candidate {
                    id: 3,
                    logit: f32::NEG_INFINITY,
                    probability: 0.0,
                },
            ],
            -1,
            false,
        )?;
        State::new(Config::Typical {
            probability: 0.8,
            min_keep: 1,
        })?
        .apply(&mut typical_candidates)?;
        assert!(
            typical_candidates
                .data()
                .iter()
                .all(|entry| entry.probability.is_finite() && entry.id != 3)
        );

        let mut history = VecDeque::new();
        for tok in [10_i32, 11, 12, 10, 11] {
            history.push_back(tok);
        }
        let mut penalties = Vec::new();
        collect_dry_penalties_from_history(&history, 1.0e38, 10.0, 1, &[], &mut penalties);
        assert_eq!(penalties.len(), 1);
        assert!(penalties[0].1.is_finite());
        assert_eq!(penalties[0].1, f32::MAX);

        let top_p = State::new(Config::TopP {
            probability: 0.9,
            min_keep: 1,
        })?;
        let mut dist = State::new(Config::Distribution(7))?;
        let mut inf_candidates = Candidates::new(
            vec![
                Candidate {
                    id: 0,
                    logit: f32::NEG_INFINITY,
                    probability: 0.0,
                },
                Candidate {
                    id: 1,
                    logit: f32::NEG_INFINITY,
                    probability: 0.0,
                },
            ],
            -1,
            false,
        )?;
        inf_candidates.sort();
        let mut weights = [0.0_f32; 2];
        dist.apply_sorted_top_p_distribution(&top_p, &mut inf_candidates, &mut weights)?;
        assert!(
            inf_candidates.data().iter().all(
                |entry| entry.probability.is_finite() && (entry.probability - 0.5).abs() < 1e-6
            )
        );
        Ok(())
    }

    #[test]
    fn upstream_sampling_guards_nan_typical_passthrough_and_dry_breaker() -> Result<()> {
        let top_p = State::new(Config::TopP {
            probability: 0.9,
            min_keep: 1,
        })?;
        let mut dist = State::new(Config::Distribution(11))?;
        let mut inf_pos = Candidates::new(
            vec![
                Candidate {
                    id: 0,
                    logit: f32::INFINITY,
                    probability: 0.0,
                },
                Candidate {
                    id: 1,
                    logit: f32::INFINITY,
                    probability: 0.0,
                },
            ],
            -1,
            false,
        )?;
        inf_pos.sort();
        let mut weights = [0.0_f32; 2];
        dist.apply_sorted_top_p_distribution(&top_p, &mut inf_pos, &mut weights)?;
        assert!(
            inf_pos
                .data()
                .iter()
                .all(|e| e.probability.is_finite() && (e.probability - 0.5).abs() < 1e-6)
        );

        let mut single = Candidates::new(
            vec![Candidate {
                id: 7,
                logit: 1.5,
                probability: 0.25,
            }],
            -1,
            false,
        )?;
        typical(&mut single, 1.0, 1);
        assert_eq!(single.size, 1);
        assert_eq!(single.data()[0].id, 7);

        let mut history = VecDeque::new();
        history.push_back(10_i32);
        let mut penalties = Vec::new();
        // history.len() == allowed (1): collect produces no penalties; with breaker matches skipped.
        collect_dry_penalties_from_history(&history, 2.0, 1.5, 1, &[], &mut penalties);
        assert_eq!(penalties, []);

        for tok in [11_i32, 10, 11] {
            history.push_back(tok);
        }
        collect_dry_penalties_from_history(&history, 2.0, 1.5, 1, &[vec![10]], &mut penalties);
        assert_eq!(penalties, []);
        Ok(())
    }
}
