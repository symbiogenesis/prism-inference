//! Seeded Swift sampler bridge to the production Prism candidate/RNG contract.

use super::{
    FiniteLogits, Sampler,
    reasoning_budget::{ReasoningBudget, State as BudgetState},
    upstream::{Candidate, Candidates, Config, State, top_k_finite_logits, top_k_logits},
};
use crate::{Result, invalid};
use std::collections::BTreeSet;

/// Delimiter sequences resolved from the verified Swift vocabulary for [`ReasoningBudget`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReasoningBudgetSequences {
    /// Opening delimiter token sequences (for example `<think>`).
    pub start: Vec<Vec<i32>>,
    /// Natural closing delimiter token sequences (for example `</think>`).
    pub end: Vec<Vec<i32>>,
    /// Forced suffix sequence emitted when budget exhausts (for example `\n</think>\n\n`).
    pub forced: Vec<i32>,
    /// Whether the prepared prompt already opens an active reasoning block.
    pub initial_thinking: bool,
}

/// Per-token model log-probability confidence captured from decoder logits before
/// sampling penalties or truncation.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenConfidence {
    /// Chosen vocabulary token identifier.
    pub token: u32,
    /// Exact `log_softmax` probability of [`Self::token`] from raw decoder logits (`<= 0.0`).
    pub logprob: f32,
    /// Top candidate `(token_id, logprob)` entries sorted descending by logprob.
    pub top_logprobs: Vec<(u32, f32)>,
}

/// Per-request sampling, anti-repetition, reasoning-budget, and logprob configuration
/// for [`SwiftSampler`].
#[derive(Clone, Debug, PartialEq)]
pub struct SwiftSamplerConfig {
    /// Sampling temperature in `(0.0, 2.0]` (default `1.0`).
    pub temperature: f32,
    /// Nucleus sampling cutoff in `(0.0, 1.0]` (default `0.95`).
    pub top_p: f32,
    /// Top-k cutoff in `0..=248_320`, where `0` admits the full vocabulary (default `20`).
    pub top_k: usize,
    /// Min-p threshold relative to the highest probability token in `[0.0, 1.0]` (default `0.0`).
    pub min_p: f32,
    /// Presence penalty in `[-2.0, 2.0]` applied to previously seen tokens (default `0.0`).
    pub presence_penalty: f32,
    /// Frequency penalty in `[-2.0, 2.0]` scaled by token count (default `0.0`).
    pub frequency_penalty: f32,
    /// Multiplicative repetition penalty in `(0.0, 2.0]` (default `1.0`).
    pub repeat_penalty: f32,
    /// Window length for repetition/frequency/presence penalties (default `64`).
    pub repeat_last_n: usize,
    /// DRY exponential repetition penalty multiplier in `[0.0, 4.0]` (default `0.0`).
    pub dry_multiplier: f32,
    /// DRY exponential base in `[1.0, 4.0]` (default `1.75`).
    pub dry_base: f32,
    /// DRY matched-prefix allowance before penalizing continuation in `1..=64` (default `2`).
    pub dry_allowed_length: usize,
    /// DRY sliding history window in `0..=8192` (default `1024`).
    pub dry_penalty_last_n: usize,
    /// Optional soft thinking-token budget; exhaustion forces `\n</think>\n\n` (default `None`).
    pub reasoning_budget: Option<u32>,
    /// Optional top-k raw log-probability count in `1..=20` streamed for `DeepConf` (default `None`).
    pub top_logprobs: Option<u8>,
}

impl Default for SwiftSamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 0.95,
            top_k: 20,
            min_p: 0.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_penalty_last_n: 1024,
            reasoning_budget: None,
            top_logprobs: None,
        }
    }
}

impl SwiftSamplerConfig {
    /// Validate all parameter bounds and reject nonfinite values.
    ///
    /// # Errors
    /// Rejects nonfinite or out-of-range sampling, penalty, DRY, budget, or logprob values.
    pub fn validate(&self) -> Result<()> {
        let finite = self.temperature.is_finite()
            && self.top_p.is_finite()
            && self.min_p.is_finite()
            && self.presence_penalty.is_finite()
            && self.frequency_penalty.is_finite()
            && self.repeat_penalty.is_finite()
            && self.dry_multiplier.is_finite()
            && self.dry_base.is_finite();
        if !(finite
            && 0.0 < self.temperature
            && self.temperature <= 2.0
            && 0.0 < self.top_p
            && self.top_p <= 1.0
            && self.top_k <= SwiftSampler::VOCABULARY
            && (0.0..=1.0).contains(&self.min_p)
            && (-2.0..=2.0).contains(&self.presence_penalty)
            && (-2.0..=2.0).contains(&self.frequency_penalty)
            && 0.0 < self.repeat_penalty
            && self.repeat_penalty <= 2.0
            && self.repeat_last_n <= 4096
            && (0.0..=4.0).contains(&self.dry_multiplier)
            && (1.0..=4.0).contains(&self.dry_base)
            && (self.dry_multiplier <= 0.0 || (1..=64).contains(&self.dry_allowed_length))
            && self.dry_allowed_length <= 64
            && self.dry_penalty_last_n <= 8192
            && self
                .reasoning_budget
                .is_none_or(|budget| (1..=262_144).contains(&budget))
            && self
                .top_logprobs
                .is_none_or(|count| (1..=20).contains(&count)))
        {
            return Err(invalid("Swift sampling configuration is out of bounds"));
        }
        Ok(())
    }

    /// True when the token-filtering/distribution stages match the canonical pinned default
    /// bitwise (`temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, no penalties, no DRY`).
    #[must_use]
    pub const fn is_canonical_default_sampling(&self) -> bool {
        self.temperature.to_bits() == 1.0_f32.to_bits()
            && self.top_p.to_bits() == 0.95_f32.to_bits()
            && self.top_k == SwiftSampler::TOP_K
            && self.min_p.to_bits() == 0.0_f32.to_bits()
            && self.presence_penalty.to_bits() == 0.0_f32.to_bits()
            && self.frequency_penalty.to_bits() == 0.0_f32.to_bits()
            && self.repeat_penalty.to_bits() == 1.0_f32.to_bits()
            && self.dry_multiplier.to_bits() == 0.0_f32.to_bits()
    }
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Raw log_softmax value is finite and bounded in f64 before narrowing to binary32."
)]
const fn f64_to_f32(value: f64) -> f32 {
    value as f32
}

fn compute_token_confidence(logits: &[f32], chosen: u32, top_k: usize) -> Result<TokenConfidence> {
    let mut max_logit = f32::NEG_INFINITY;
    for &logit in logits {
        if !logit.is_finite() {
            return Err(invalid(
                "Swift sampling requires the complete finite pinned vocabulary",
            ));
        }
        if logit > max_logit {
            max_logit = logit;
        }
    }
    let max_f64 = f64::from(max_logit);
    let mut sum_exp = 0.0_f64;
    for &logit in logits {
        sum_exp += (f64::from(logit) - max_f64).exp();
    }
    let log_z = max_f64 + sum_exp.ln();
    let chosen_idx = usize::try_from(chosen)?;
    let chosen_logit = *logits
        .get(chosen_idx)
        .ok_or_else(|| invalid("Swift sampler selected out-of-range token"))?;
    let logprob = f64_to_f32(f64::from(chosen_logit) - log_z);

    let mut top_buf = Vec::new();
    if top_k > 0 {
        top_k_logits(logits, top_k, &mut top_buf)?;
    }
    let top_logprobs = top_buf
        .into_iter()
        .filter_map(|candidate| {
            u32::try_from(candidate.id)
                .ok()
                .map(|id| (id, f64_to_f32(f64::from(candidate.logit) - log_z)))
        })
        .collect();
    Ok(TokenConfidence {
        token: chosen,
        logprob,
        top_logprobs,
    })
}

/// Seeded sampler for the pinned 248,320-token Swift model.
///
/// The canonical default configuration retains the fixed top-k 20, top-p 0.95
/// (`min_keep = 1`), min-p 0, temperature 1, zero-history, and Windows MSVC
/// MT19937 distribution fast path bit-for-bit. Explicit calibrated sampling,
/// penalties, DRY anti-repetition, soft reasoning budgets, and opt-in per-token
/// log-probability confidence are also supported while preserving independent
/// rollback cloning and reset semantics.
pub struct SwiftSampler {
    config: SwiftSamplerConfig,
    top_p: State,
    distribution: State,
    penalties: Option<State>,
    dry: Option<State>,
    budget: Option<ReasoningBudget>,
    top_k_stage: Option<State>,
    min_p_stage: Option<State>,
    temperature_stage: Option<State>,
    backing: Vec<Candidate>,
    all_backing: Vec<Candidate>,
    last_confidence: Option<TokenConfidence>,
}

impl Clone for SwiftSampler {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            top_p: self.top_p.clone(),
            distribution: self.distribution.clone(),
            penalties: self.penalties.clone(),
            dry: self.dry.clone(),
            budget: self.budget.clone(),
            top_k_stage: self.top_k_stage.clone(),
            min_p_stage: self.min_p_stage.clone(),
            temperature_stage: self.temperature_stage.clone(),
            backing: Vec::new(),
            all_backing: Vec::new(),
            last_confidence: None,
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
            config: SwiftSamplerConfig::default(),
            top_p: State::new(Config::TopP {
                probability: 0.95,
                min_keep: 1,
            })?,
            distribution: State::new(Config::Distribution(seed))?,
            penalties: None,
            dry: None,
            budget: None,
            top_k_stage: None,
            min_p_stage: None,
            temperature_stage: None,
            backing: Vec::new(),
            all_backing: Vec::new(),
            last_confidence: None,
        })
    }

    /// Creates a configured sampler with an explicit reproducible native seed.
    ///
    /// # Errors
    /// Rejects sentinel `u32::MAX` seeds or invalid sampling/budget options.
    pub fn with_config(
        seed: u32,
        config: SwiftSamplerConfig,
        budget_sequences: Option<ReasoningBudgetSequences>,
        dry_breakers: &[Vec<i32>],
    ) -> Result<Self> {
        let mut sampler = Self::new(seed)?;
        sampler.apply_request_config(config, budget_sequences, dry_breakers)?;
        Ok(sampler)
    }

    /// Reconfigures per-request sampling, penalties, DRY, and soft reasoning budget
    /// while preserving the active MT19937 RNG progression across requests.
    ///
    /// # Errors
    /// Validates all parameters atomically before replacing active stages.
    pub fn apply_request_config(
        &mut self,
        config: SwiftSamplerConfig,
        budget_sequences: Option<ReasoningBudgetSequences>,
        dry_breakers: &[Vec<i32>],
    ) -> Result<()> {
        config.validate()?;
        let top_p = State::new(Config::TopP {
            probability: config.top_p,
            min_keep: 1,
        })?;
        let top_k_stage = if config.top_k > 0 {
            Some(State::new(Config::TopK(i32::try_from(config.top_k)?))?)
        } else {
            None
        };
        let min_p_stage = if config.min_p > 0.0 {
            Some(State::new(Config::MinP {
                probability: config.min_p,
                min_keep: 1,
            })?)
        } else {
            None
        };
        let temperature_stage = if config.temperature.to_bits() == 1.0_f32.to_bits() {
            None
        } else {
            Some(State::new(Config::Temperature(config.temperature))?)
        };
        let has_penalties = config.repeat_last_n > 0
            && (config.repeat_penalty.to_bits() != 1.0_f32.to_bits()
                || config.frequency_penalty.to_bits() != 0.0_f32.to_bits()
                || config.presence_penalty.to_bits() != 0.0_f32.to_bits());
        let penalties = if has_penalties {
            Some(State::new(Config::Penalties {
                vocabulary_size: Self::VOCABULARY,
                last_n: i32::try_from(config.repeat_last_n)?,
                repeat: config.repeat_penalty,
                frequency: config.frequency_penalty,
                presence: config.presence_penalty,
            })?)
        } else {
            None
        };
        let has_dry = config.dry_multiplier > 0.0 && config.dry_penalty_last_n > 0;
        let dry = if has_dry {
            Some(State::new(Config::Dry {
                multiplier: config.dry_multiplier,
                base: config.dry_base,
                allowed_length: config.dry_allowed_length,
                last_n: i32::try_from(config.dry_penalty_last_n)?,
                sequence_breakers: unique_sorted_dry_breakers(dry_breakers.to_vec()),
            })?)
        } else {
            None
        };
        let budget = match (config.reasoning_budget, budget_sequences) {
            (Some(limit), Some(sequences)) => Some(ReasoningBudget::new(
                sequences.start,
                sequences.end,
                sequences.forced,
                i32::try_from(limit)?,
                if sequences.initial_thinking {
                    BudgetState::Counting
                } else {
                    BudgetState::Done
                },
            )?),
            (Some(_), None) => {
                return Err(invalid(
                    "Swift reasoning budget requires tokenizer delimiter sequences",
                ));
            }
            (None, _) => None,
        };
        self.config = config;
        self.top_p = top_p;
        self.top_k_stage = top_k_stage;
        self.min_p_stage = min_p_stage;
        self.temperature_stage = temperature_stage;
        self.penalties = penalties;
        self.dry = dry;
        self.budget = budget;
        self.last_confidence = None;
        Ok(())
    }

    /// Active sampler configuration.
    #[must_use]
    pub const fn config(&self) -> &SwiftSamplerConfig {
        &self.config
    }

    /// Record prior prompt history tokens into active repetition/DRY states without drawing RNG.
    pub fn seed_history(&mut self, tokens: &[u32]) {
        if self.penalties.is_none() && self.dry.is_none() {
            return;
        }
        for &tok in tokens {
            if let Ok(token) = i32::try_from(tok) {
                if let Some(penalties) = &mut self.penalties {
                    penalties.accept(token);
                }
                if let Some(dry) = &mut self.dry {
                    dry.accept(token);
                }
            }
        }
    }

    /// Restores the original seed, including the production draw ordering.
    pub fn reset(&mut self) {
        self.distribution.reset();
        if let Some(penalties) = &mut self.penalties {
            penalties.reset();
        }
        if let Some(dry) = &mut self.dry {
            dry.reset();
        }
        if let Some(budget) = &mut self.budget {
            budget.reset();
        }
        self.last_confidence = None;
    }

    fn accept_sampled_token(&mut self, token: u32) -> Result<()> {
        let token_i32 = i32::try_from(token)?;
        if let Some(penalties) = &mut self.penalties {
            penalties.accept(token_i32);
        }
        if let Some(dry) = &mut self.dry {
            dry.accept(token_i32);
        }
        if let Some(budget) = &mut self.budget {
            budget.accept(token_i32);
        }
        Ok(())
    }
}

impl Sampler for SwiftSampler {
    fn sample(&mut self, logits: &[f32]) -> Result<u32> {
        self.last_confidence = None;
        if logits.len() != Self::VOCABULARY {
            return Err(invalid(
                "Swift sampling requires the complete finite pinned vocabulary",
            ));
        }
        self.sample_with_logits::<true>(logits)
    }

    fn sample_finite(&mut self, logits: FiniteLogits<'_>) -> Result<u32> {
        self.last_confidence = None;
        let slice = logits.as_slice();
        if slice.len() != Self::VOCABULARY {
            return Err(invalid(
                "Swift sampling requires the complete finite pinned vocabulary",
            ));
        }
        self.sample_with_logits::<false>(slice)
    }

    fn take_last_confidence(&mut self) -> Option<TokenConfidence> {
        self.last_confidence.take()
    }
}

impl SwiftSampler {
    fn sample_with_logits<const VALIDATE: bool>(&mut self, logits: &[f32]) -> Result<u32> {
        if VALIDATE
            && self
                .budget
                .as_ref()
                .and_then(ReasoningBudget::forced_token)
                .is_some()
        {
            for &value in logits {
                if !value.is_finite() {
                    return Err(invalid(
                        "Swift sampling requires the complete finite pinned vocabulary",
                    ));
                }
            }
        }
        let token = if let Some(forced) =
            self.budget.as_ref().and_then(ReasoningBudget::forced_token)
        {
            let forced_idx = usize::try_from(forced)?;
            let forced_logit = *logits
                .get(forced_idx)
                .ok_or_else(|| invalid("Swift forced reasoning token out of bounds"))?;
            let mut backing = std::mem::take(&mut self.backing);
            backing.clear();
            backing.push(Candidate {
                id: forced,
                logit: forced_logit,
                probability: 0.0,
            });
            let mut candidates = Candidates::new(backing, -1, true)?;
            let selected = (|| -> Result<u32> {
                self.distribution.apply(&mut candidates)?;
                let idx = usize::try_from(candidates.selected())?;
                let token = candidates
                    .data()
                    .get(idx)
                    .ok_or_else(|| invalid("Swift sampler selected an invalid candidate"))?;
                Ok(u32::try_from(token.id)?)
            })();
            self.backing = candidates.into_backing();
            selected?
        } else if self.config.is_canonical_default_sampling() {
            self.sample_selected(|backing| {
                if VALIDATE {
                    top_k_logits(logits, Self::TOP_K, backing)
                } else {
                    top_k_finite_logits(FiniteLogits::from_validated(logits)?, Self::TOP_K, backing)
                }
            })?
        } else {
            self.sample_custom::<VALIDATE>(logits)?
        };
        if let Some(top_k_logprobs) = self.config.top_logprobs {
            self.last_confidence = Some(compute_token_confidence(
                logits,
                token,
                usize::from(top_k_logprobs),
            )?);
        }
        self.accept_sampled_token(token)?;
        Ok(token)
    }

    fn sample_custom<const VALIDATE: bool>(&mut self, logits: &[f32]) -> Result<u32> {
        // Fast bounded path when no full-vocabulary history stage (penalties/DRY) is active
        // and top_k <= 128. Preserves SIMD selection speed for typical calibrated presets.
        if self.penalties.is_none()
            && self.dry.is_none()
            && self.config.top_k > 0
            && self.config.top_k <= 128
        {
            let keep = self.config.top_k;
            let mut backing = std::mem::take(&mut self.backing);
            let res = if VALIDATE {
                top_k_logits(logits, keep, &mut backing)
            } else {
                top_k_finite_logits(FiniteLogits::from_validated(logits)?, keep, &mut backing)
            };
            if let Err(error) = res {
                self.backing = backing;
                return Err(error);
            }
            let mut candidates = Candidates::new(backing, -1, true)?;
            let result = self.apply_custom_filter_and_draw(&mut candidates);
            self.backing = candidates.into_backing();
            return result;
        }

        // Full-vocabulary path: validate logits if requested, populate reusable buffer,
        // apply active DRY and repeat/presence/frequency penalties, then filter & draw.
        let mut backing = std::mem::take(&mut self.all_backing);
        backing.clear();
        backing.try_reserve(logits.len())?;
        for (index, &logit) in logits.iter().enumerate() {
            if VALIDATE && !logit.is_finite() {
                backing.clear();
                self.all_backing = backing;
                return Err(invalid(
                    "Swift sampling requires the complete finite pinned vocabulary",
                ));
            }
            backing.push(Candidate {
                id: i32::try_from(index)?,
                logit,
                probability: 0.0,
            });
        }
        let mut candidates = Candidates::new(backing, -1, false)?;
        let result = (|| {
            if let Some(dry) = &mut self.dry {
                dry.apply(&mut candidates)?;
            }
            if let Some(penalties) = &mut self.penalties {
                penalties.apply(&mut candidates)?;
            }
            if let Some(top_k) = &mut self.top_k_stage {
                top_k.apply(&mut candidates)?;
            }
            self.apply_custom_filter_and_draw(&mut candidates)
        })();
        self.all_backing = candidates.into_backing();
        result
    }

    fn apply_custom_filter_and_draw(&mut self, candidates: &mut Candidates) -> Result<u32> {
        if let Some(min_p) = &mut self.min_p_stage {
            min_p.apply(candidates)?;
        }
        self.top_p.apply(candidates)?;
        if let Some(temperature) = &mut self.temperature_stage {
            temperature.apply(candidates)?;
        }
        self.distribution.apply(candidates)?;
        let selected = usize::try_from(candidates.selected())?;
        let token = candidates
            .data()
            .get(selected)
            .ok_or_else(|| invalid("Swift sampler selected an invalid candidate"))?;
        Ok(u32::try_from(token.id)?)
    }

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

/// Check DRY-breaker deduplication cleanly when callers resolve custom breaker slices.
fn unique_sorted_dry_breakers(breakers: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let seen: BTreeSet<Vec<i32>> = breakers.into_iter().filter(|seq| !seq.is_empty()).collect();
    seen.into_iter().collect()
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
            assert_eq!(
                observed.probability.to_bits(),
                original.probability.to_bits()
            );
        }
    }

    #[test]
    fn cached_top_p_exponents_preserve_original_backing_bits_draws_and_reset() -> crate::Result<()>
    {
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
                                1 => {
                                    if index.is_multiple_of(2) {
                                        -0.0
                                    } else {
                                        0.0
                                    }
                                }
                                2 => -value,
                                3 => -value * 5.0,
                                4 => f32::MIN,
                                5 => f32::MAX,
                                6 => {
                                    if index == 0 {
                                        f32::MAX
                                    } else {
                                        f32::MIN
                                    }
                                }
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
                assert_eq!(
                    independent.sample_candidates(&mut cloned_candidates)?,
                    token
                );
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
        assert!(
            distribution
                .apply_sorted_top_p_distribution(filter, &mut actual, weights)
                .is_err()
        );
        assert_candidate_bits(&actual, original);
    }

    #[test]
    fn cached_top_p_guards_preserve_candidates_and_rng_with_unsorted_fallback() -> crate::Result<()>
    {
        let filter = State::new(Config::TopP {
            probability: 0.95,
            min_keep: 1,
        })?;
        let mut sampler = SwiftSampler::new(4242)?;
        let mut reference = State::new(Config::Distribution(4242))?;
        let backing = (0..20)
            .map(|id| Candidate {
                id,
                logit: 0.0,
                probability: -0.0,
            })
            .collect::<Vec<_>>();
        let original = Candidates::new(backing, -1, true)?;
        for capacity in [0, 19] {
            let mut weights = vec![0.0_f32; capacity];
            assert_cached_rejected(&mut sampler.distribution, &filter, &original, &mut weights);
        }
        for invalid in [
            Config::Temperature(1.0),
            Config::TopP {
                probability: 1.0,
                min_keep: 1,
            },
        ] {
            assert_cached_rejected(
                &mut sampler.distribution,
                &State::new(invalid)?,
                &original,
                &mut [0.0_f32; 20],
            );
        }
        let mut wrong_distribution = State::new(Config::TopK(20))?;
        assert_cached_rejected(
            &mut wrong_distribution,
            &filter,
            &original,
            &mut [0.0_f32; 20],
        );
        let mut candidates = original.clone();
        let mut expected = original.clone();
        assert_eq!(
            sampler.sample_candidates(&mut candidates)?,
            original_fixed_chain(&mut reference, &mut expected)?
        );
        assert_candidate_bits(&candidates, &expected);
        let mut unsorted = Candidates::new(original.into_backing(), -1, false)?;
        let before = unsorted.clone();
        assert_cached_rejected(
            &mut sampler.distribution,
            &filter,
            &unsorted,
            &mut [0.0_f32; 20],
        );
        let mut expected = before;
        assert_eq!(
            sampler.sample_candidates(&mut unsorted)?,
            original_fixed_chain(&mut reference, &mut expected)?
        );
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
    fn lane_pruned_heap_preserves_native_ties_threshold_advances_and_tails() -> crate::Result<()> {
        let mut bounded = Vec::new();
        for keep in [1_usize, 2, 20, 128] {
            for tail in 0..16 {
                for pattern in 0..7 {
                    let logits = (0..keep + 32 + tail)
                        .map(|index| {
                            let value = f32::from(u16::try_from(index)?);
                            Ok(match pattern {
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
                                5 => f32::from_bits(u32::try_from(index % 5)?),
                                _ => {
                                    if index % 16 == 15 {
                                        value
                                    } else {
                                        -value
                                    }
                                }
                            })
                        })
                        .collect::<crate::Result<Vec<_>>>()?;
                    let backing = logits
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
                    let mut reference = Candidates::new(backing, -1, false)?;
                    State::new(Config::TopK(i32::try_from(keep)?))?.apply(&mut reference)?;
                    top_k_logits(&logits, keep, &mut bounded)?;
                    assert_eq!(bounded.len(), keep);
                    for (actual, expected) in bounded.iter().zip(reference.data()) {
                        assert_eq!(actual.id, expected.id);
                        assert_eq!(actual.logit.to_bits(), expected.logit.to_bits());
                        assert_eq!(actual.probability.to_bits(), expected.probability.to_bits());
                    }
                    let mut finite = Vec::new();
                    super::top_k_finite_logits(
                        super::FiniteLogits::new(&logits)?,
                        keep,
                        &mut finite,
                    )?;
                    for (actual, expected) in finite.iter().zip(&bounded) {
                        assert_eq!(actual.id, expected.id);
                        assert_eq!(actual.logit.to_bits(), expected.logit.to_bits());
                        assert_eq!(actual.probability.to_bits(), expected.probability.to_bits());
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn lane_pruned_heap_rejects_every_nonfinite_position_without_consuming_rng() -> crate::Result<()>
    {
        let mut logits = vec![0.0_f32; SwiftSampler::VOCABULARY];
        let mut sampler = SwiftSampler::new(4242)?;
        let mut control = sampler.clone();
        assert_eq!(sampler.sample(&logits)?, control.sample(&logits)?);
        let allocation = sampler.backing.as_ptr();
        // Initial heap, every SIMD lane after the heap, and every tail position
        // of the private selector all retain complete-vocabulary rejection.
        let mut small = vec![0.0_f32; SwiftSampler::TOP_K + 32 + 15];
        for index in 0..small.len() {
            for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                small[index] = invalid;
                assert!(top_k_logits(&small, SwiftSampler::TOP_K, &mut sampler.backing).is_err());
                assert_eq!(sampler.backing.len(), 0);
                assert_eq!(sampler.backing.as_ptr(), allocation);
            }
            small[index] = 0.0;
        }
        for lane in 0..16 {
            for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let index = SwiftSampler::TOP_K + lane;
                logits[index] = invalid;
                assert!(sampler.sample(&logits).is_err());
                assert_eq!(sampler.backing.as_ptr(), allocation);
                logits[index] = 0.0;
            }
        }
        assert_eq!(sampler.sample(&logits)?, control.sample(&logits)?);
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

    #[test]
    fn swift_sampler_config_preserves_canonical_default_and_supports_calibrated_controls()
    -> crate::Result<()> {
        let mut baseline = SwiftSampler::new(777)?;
        let mut configured_default =
            SwiftSampler::with_config(777, super::SwiftSamplerConfig::default(), None, &[])?;
        let mut logits = vec![-2.0_f32; SwiftSampler::VOCABULARY];
        logits[10] = 4.5;
        logits[11] = 4.25;
        logits[12] = 4.0;
        logits[13] = 3.75;
        for _ in 0..8 {
            assert_eq!(
                baseline.sample(&logits)?,
                configured_default.sample(&logits)?
            );
            assert!(configured_default.take_last_confidence().is_none());
        }

        // Reject out-of-range parameters atomically.
        for bad in [
            super::SwiftSamplerConfig {
                temperature: 0.0,
                ..super::SwiftSamplerConfig::default()
            },
            super::SwiftSamplerConfig {
                min_p: 1.5,
                ..super::SwiftSamplerConfig::default()
            },
            super::SwiftSamplerConfig {
                dry_multiplier: f32::NAN,
                ..super::SwiftSamplerConfig::default()
            },
            super::SwiftSamplerConfig {
                top_logprobs: Some(21),
                ..super::SwiftSamplerConfig::default()
            },
            super::SwiftSamplerConfig {
                reasoning_budget: Some(0),
                ..super::SwiftSamplerConfig::default()
            },
        ] {
            assert!(bad.validate().is_err());
        }

        // Verify calibrated sampling + penalties + DRY + top_logprobs confidence extraction.
        let mut calibrated = SwiftSampler::with_config(
            99,
            super::SwiftSamplerConfig {
                temperature: 0.7,
                top_p: 0.9,
                top_k: 40,
                min_p: 0.05,
                presence_penalty: 0.4,
                frequency_penalty: 0.0,
                repeat_penalty: 1.15,
                repeat_last_n: 64,
                dry_multiplier: 0.8,
                dry_base: 1.75,
                dry_allowed_length: 2,
                dry_penalty_last_n: 128,
                reasoning_budget: None,
                top_logprobs: Some(3),
            },
            None,
            &[vec![100]],
        )?;
        let token = calibrated.sample(&logits)?;
        let conf = calibrated
            .take_last_confidence()
            .ok_or_else(|| crate::invalid("missing token confidence"))?;
        assert_eq!(conf.token, token);
        assert!(conf.logprob <= 0.0);
        assert_eq!(conf.top_logprobs.len(), 3);
        assert_eq!(conf.top_logprobs[0].0, 10);
        assert!(conf.top_logprobs.windows(2).all(|w| w[0].1 >= w[1].1));
        assert!(calibrated.take_last_confidence().is_none());

        // Verify ReasoningBudget forces </think> sequence cleanly even when logits disfavor it.
        let sequences = super::ReasoningBudgetSequences {
            start: vec![vec![50]],
            end: vec![vec![51]],
            forced: vec![52, 51, 52],
            initial_thinking: true,
        };
        let mut budgeted = SwiftSampler::with_config(
            17,
            super::SwiftSamplerConfig {
                reasoning_budget: Some(2),
                top_logprobs: Some(2),
                ..super::SwiftSamplerConfig::default()
            },
            Some(sequences),
            &[],
        )?;
        logits.fill(-50.0);
        logits[10] = 10.0;
        assert_eq!(budgeted.sample(&logits)?, 10); // remaining: 2 -> 1
        assert_eq!(budgeted.sample(&logits)?, 10); // remaining: 1 -> 0 -> Forcing
        let saved = budgeted.clone();
        assert_eq!(budgeted.sample(&logits)?, 52); // forced[0] even though logit[52] == -50.0
        assert_eq!(budgeted.sample(&logits)?, 51); // forced[1]
        assert_eq!(budgeted.sample(&logits)?, 52); // forced[2] -> Done
        assert_eq!(budgeted.sample(&logits)?, 10); // normal sampling resumes
        // Clone rollback restores exact forcing cursor.
        let mut restored = saved;
        assert_eq!(restored.sample(&logits)?, 52);
        Ok(())
    }
}
