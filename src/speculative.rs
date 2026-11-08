//! Checked scheduling bounds, prompt/history suffix drafting (`SuffixDecoding`),
//! and exact greedy / stochastic Leviathan speculative verification.

use crate::{Result, invalid};

/// Maximum number of draft continuation tokens proposed in a single short-chain round.
///
/// Bounding draft chains to at most eight tokens keeps hybrid Gated `DeltaNet`
/// recurrent state and KV-cache verification overhead strictly bounded.
pub const MAX_SPECULATIVE_DRAFT_TOKENS: usize = 8;

/// Maximum suffix match window searched across prompt and committed token history.
pub const MAX_SUFFIX_MATCH_TOKENS: usize = 64;

/// Whether catch-up rows and sequence anchors fit the first decoder batch.
///
/// The addition uses a widened integer so valid `i32` inputs never overflow.
/// A nonpositive batch or a negative row/anchor count is rejected.
#[must_use]
pub fn mtp_first_decode_fits(batch: i32, catchup_rows: i32, anchors: i32) -> bool {
    batch > 0
        && catchup_rows >= 0
        && anchors >= 0
        && i64::from(catchup_rows) + i64::from(anchors) <= i64::from(batch)
}

/// Bounded configuration for training-free prompt and history suffix drafting (`SuffixDecoding`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SuffixDraftConfig {
    /// Shortest suffix match length eligible to propose continuation tokens (`1..=MAX_SUFFIX_MATCH_TOKENS`).
    pub min_match_tokens: usize,
    /// Longest suffix match length searched in committed history (`min_match_tokens..=MAX_SUFFIX_MATCH_TOKENS`).
    pub max_match_tokens: usize,
    /// Maximum continuation tokens proposed per round (`1..=MAX_SPECULATIVE_DRAFT_TOKENS`).
    pub max_draft_tokens: usize,
}

impl Default for SuffixDraftConfig {
    fn default() -> Self {
        Self {
            min_match_tokens: 2,
            max_match_tokens: 16,
            max_draft_tokens: 4,
        }
    }
}

impl SuffixDraftConfig {
    /// Construct and validate a bounded suffix-drafting configuration.
    ///
    /// # Errors
    /// Rejects zero bounds, inverted match ranges, or draft lengths exceeding
    /// [`MAX_SPECULATIVE_DRAFT_TOKENS`].
    pub fn new(
        min_match_tokens: usize,
        max_match_tokens: usize,
        max_draft_tokens: usize,
    ) -> Result<Self> {
        let config = Self {
            min_match_tokens,
            max_match_tokens,
            max_draft_tokens,
        };
        config.validate()?;
        Ok(config)
    }

    /// Validate that all suffix-matching and draft-chain bounds are within contract limits.
    ///
    /// # Errors
    /// Rejects zero bounds, `min_match_tokens > max_match_tokens`,
    /// `max_match_tokens > MAX_SUFFIX_MATCH_TOKENS`, or
    /// `max_draft_tokens > MAX_SPECULATIVE_DRAFT_TOKENS`.
    pub fn validate(&self) -> Result<()> {
        if self.min_match_tokens == 0
            || self.min_match_tokens > self.max_match_tokens
            || self.max_match_tokens > MAX_SUFFIX_MATCH_TOKENS
            || self.max_draft_tokens == 0
            || self.max_draft_tokens > MAX_SPECULATIVE_DRAFT_TOKENS
        {
            return Err(invalid("invalid suffix speculative draft configuration"));
        }
        Ok(())
    }
}

/// Accounting for speculative drafting and target verification across one request.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SpeculativeStats {
    /// Number of speculative draft-and-verify rounds entered with a non-empty proposal.
    pub draft_rounds: usize,
    /// Total continuation tokens proposed by the drafter across all rounds.
    pub draft_tokens_proposed: usize,
    /// Total proposed continuation tokens accepted by the target verifier.
    pub draft_tokens_accepted: usize,
    /// Total target decoder forward evaluations executed during generation.
    pub target_evaluations: usize,
    /// Number of transactional decoder checkpoint restores triggered by rejected block steps.
    pub rollback_restores: usize,
}

impl SpeculativeStats {
    /// Fraction of proposed draft tokens accepted by the target verifier in `[0.0, 1.0]`.
    #[must_use]
    #[allow(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "Diagnostic acceptance ratio converts bounded token counts to f64."
    )]
    pub const fn acceptance_rate(&self) -> f64 {
        if self.draft_tokens_proposed == 0 {
            0.0
        } else {
            (self.draft_tokens_accepted as f64) / (self.draft_tokens_proposed as f64)
        }
    }
}

/// Zero-weight prompt and token-history suffix drafter (`SuffixDecoding`).
///
/// Searches committed prompt and generated token history for the longest suffix match
/// of the current context and proposes up to `max_draft_tokens` continuation tokens.
/// Ties in match length prefer the most recent historical match.
#[derive(Clone, Copy, Debug, Default)]
pub struct SuffixDrafter;

impl SuffixDrafter {
    /// Propose up to `config.max_draft_tokens.min(remaining_tokens)` continuation tokens
    /// by matching the suffix of `history` against earlier spans in `history`.
    ///
    /// # Errors
    /// Rejects an invalid [`SuffixDraftConfig`] or allocation failure.
    pub fn propose(
        history: &[u32],
        config: SuffixDraftConfig,
        remaining_tokens: usize,
    ) -> Result<Vec<u32>> {
        config.validate()?;
        let limit = config.max_draft_tokens.min(remaining_tokens);
        if limit == 0 || history.len() <= config.min_match_tokens {
            return Ok(Vec::new());
        }
        let max_len = config.max_match_tokens.min(history.len() - 1);
        for match_len in (config.min_match_tokens..=max_len).rev() {
            let suffix_start = history.len() - match_len;
            let suffix = &history[suffix_start..];
            // Search backwards from suffix_start - 1 so the most recent prior match wins,
            // and require at least one continuation token after the matched span.
            for candidate_start in (0..suffix_start).rev() {
                if history[candidate_start..candidate_start + match_len] == *suffix {
                    let cont_start = candidate_start + match_len;
                    let available = (history.len() - cont_start).min(limit);
                    if available > 0 {
                        let mut draft = Vec::new();
                        draft.try_reserve_exact(available)?;
                        draft.extend_from_slice(&history[cont_start..cont_start + available]);
                        return Ok(draft);
                    }
                }
            }
        }
        Ok(Vec::new())
    }
}

/// Verify one speculative draft token under the exact stochastic acceptance-rejection
/// rule of Leviathan et al. (2023), preserving the target distribution $p(x)$ without
/// temperature distortion.
///
/// - `target_probs`: unnormalized or normalized non-negative target weights $p(x)$.
/// - `draft_probs`: unnormalized or normalized non-negative draft weights $q(x)$.
/// - `draft_token`: token ID proposed by the draft distribution.
/// - `accept_draw`: uniform random draw in `[0.0, 1.0)` for acceptance check $u < \min(1, p(x)/q(x))$.
/// - `residual_draw`: uniform random draw in `[0.0, 1.0)` for sampling from the residual
///   distribution $\max(0, p(x) - q(x))$ when `draft_token` is rejected.
///
/// Returns `Ok((accepted, final_token))` where `accepted` is `true` if `draft_token` was
/// kept, or `false` with `final_token` sampled from the normalized residual distribution.
///
/// # Errors
/// Rejects empty or oversized distributions (`> 1_048_576`), mismatched lengths,
/// negative/non-finite weights, zero-sum distributions, out-of-range `draft_token`,
/// or uniform draws outside `[0.0, 1.0)`.
pub fn verify_stochastic_step(
    target_probs: &[f64],
    draft_probs: &[f64],
    draft_token: u32,
    accept_draw: f64,
    residual_draw: f64,
) -> Result<(bool, u32)> {
    if target_probs.is_empty()
        || target_probs.len() > 1_048_576
        || target_probs.len() != draft_probs.len()
        || !accept_draw.is_finite()
        || !(0.0..1.0).contains(&accept_draw)
        || !residual_draw.is_finite()
        || !(0.0..1.0).contains(&residual_draw)
    {
        return Err(invalid(
            "stochastic speculative verification requires matching bounded distributions and draws in [0, 1)",
        ));
    }
    let draft_index = usize::try_from(draft_token)?;
    if draft_index >= target_probs.len() {
        return Err(invalid(
            "draft token is outside the verification vocabulary",
        ));
    }
    let mut target_sum = 0.0_f64;
    let mut draft_sum = 0.0_f64;
    for (&p, &q) in target_probs.iter().zip(draft_probs) {
        if !p.is_finite() || p < 0.0 || !q.is_finite() || q < 0.0 {
            return Err(invalid(
                "speculative verification probabilities must be finite and nonnegative",
            ));
        }
        target_sum += p;
        draft_sum += q;
    }
    if !target_sum.is_finite() || target_sum <= 0.0 || !draft_sum.is_finite() || draft_sum <= 0.0 {
        return Err(invalid(
            "speculative verification distributions must have positive total mass",
        ));
    }

    let p_draft = target_probs[draft_index] / target_sum;
    let q_draft = draft_probs[draft_index] / draft_sum;
    let acceptance_ratio = if q_draft <= 0.0 {
        if p_draft > 0.0 { 1.0 } else { 0.0 }
    } else {
        (p_draft / q_draft).min(1.0)
    };

    if accept_draw < acceptance_ratio {
        return Ok((true, draft_token));
    }

    // Compute total residual mass sum_x max(0, p_norm(x) - q_norm(x)).
    let mut residual_sum = 0.0_f64;
    for (&p, &q) in target_probs.iter().zip(draft_probs) {
        let p_norm = p / target_sum;
        let q_norm = q / draft_sum;
        let diff = (p_norm - q_norm).max(0.0);
        residual_sum += diff;
    }

    if residual_sum <= 0.0 {
        // Identical distributions; sample directly from target_probs.
        let target_threshold = residual_draw * target_sum;
        let mut cumulative = 0.0_f64;
        let mut selected = 0_u32;
        for (index, &p) in target_probs.iter().enumerate() {
            cumulative += p;
            if p > 0.0 {
                selected = u32::try_from(index)?;
            }
            if target_threshold < cumulative {
                break;
            }
        }
        return Ok((false, selected));
    }

    let threshold = residual_draw * residual_sum;
    let mut cumulative = 0.0_f64;
    let mut selected = 0_u32;
    for (index, (&p, &q)) in target_probs.iter().zip(draft_probs).enumerate() {
        let p_norm = p / target_sum;
        let q_norm = q / draft_sum;
        let diff = (p_norm - q_norm).max(0.0);
        cumulative += diff;
        if diff > 0.0 {
            selected = u32::try_from(index)?;
        }
        if threshold < cumulative {
            break;
        }
    }
    Ok((false, selected))
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_SPECULATIVE_DRAFT_TOKENS, SpeculativeStats, SuffixDraftConfig, SuffixDrafter,
        mtp_first_decode_fits, verify_stochastic_step,
    };
    use crate::Result;

    #[test]
    fn mtp_bounds_check_handles_extremes_and_negatives() {
        assert!(mtp_first_decode_fits(8, 4, 4));
        assert!(!mtp_first_decode_fits(8, 5, 4));
        assert!(!mtp_first_decode_fits(0, 0, 0));
        assert!(!mtp_first_decode_fits(8, -1, 2));
        assert!(!mtp_first_decode_fits(8, 2, -1));
        assert!(!mtp_first_decode_fits(i32::MAX, i32::MAX, 1));
    }

    #[test]
    fn suffix_drafter_prefers_longest_then_most_recent_match() -> Result<()> {
        let config = SuffixDraftConfig::new(2, 8, 4)?;
        // History has two matches for [10, 20]:
        // - earlier at index 0 followed by [30, 40, 50]
        // - later at index 5 followed by [31, 41, 10, 20]
        // And a longer match for [9, 10, 20] at index 11? Let's test both!
        let history = vec![10, 20, 30, 40, 50, 10, 20, 31, 41, 10, 20];
        let draft = SuffixDrafter::propose(&history, config, 4)?;
        // Most recent match of [10, 20] is at index 5, followed by [31, 41, 10, 20].
        assert_eq!(draft, vec![31, 41, 10, 20]);

        // Now test that a 3-token match beats a more recent 2-token match:
        let history_longer = vec![7, 10, 20, 99, 98, 8, 10, 20, 31, 41, 7, 10, 20];
        let draft_longer = SuffixDrafter::propose(&history_longer, config, 3)?;
        // Suffix [7, 10, 20] (length 3) matches at index 0, followed by [99, 98, 8].
        assert_eq!(draft_longer, vec![99, 98, 8]);

        // Respect remaining_tokens cap:
        let capped = SuffixDrafter::propose(&history_longer, config, 1)?;
        assert_eq!(capped, vec![99]);
        assert_eq!(
            SuffixDrafter::propose(&history_longer, config, 0)?,
            Vec::<u32>::new()
        );

        // Reject invalid configs:
        assert!(SuffixDraftConfig::new(0, 4, 2).is_err());
        assert!(SuffixDraftConfig::new(5, 4, 2).is_err());
        assert!(SuffixDraftConfig::new(2, 4, MAX_SPECULATIVE_DRAFT_TOKENS + 1).is_err());
        Ok(())
    }

    #[test]
    fn leviathan_stochastic_verification_preserves_exact_residual_distribution() -> Result<()> {
        let target = [0.2_f64, 0.5, 0.3];
        let draft = [0.5_f64, 0.2, 0.3];

        // Token 0 has p = 0.2, q = 0.5 -> acceptance ratio = 0.4.
        // Draw 0.39 < 0.4 -> accepted!
        assert_eq!(
            verify_stochastic_step(&target, &draft, 0, 0.39, 0.0)?,
            (true, 0)
        );
        // Draw 0.41 >= 0.4 -> rejected! Residual mass is only on token 1 (max(0, 0.5 - 0.2) = 0.3).
        assert_eq!(
            verify_stochastic_step(&target, &draft, 0, 0.41, 0.75)?,
            (false, 1)
        );

        // Token 1 has p = 0.5, q = 0.2 -> acceptance ratio = min(1, 2.5) = 1.0 -> always accepted!
        assert_eq!(
            verify_stochastic_step(&target, &draft, 1, 0.999, 0.0)?,
            (true, 1)
        );

        let stats = SpeculativeStats {
            draft_rounds: 2,
            draft_tokens_proposed: 4,
            draft_tokens_accepted: 3,
            target_evaluations: 5,
            rollback_restores: 1,
        };
        assert!((stats.acceptance_rate() - 0.75).abs() <= 1e-12);
        assert_eq!(SpeculativeStats::default().acceptance_rate(), 0.0);
        Ok(())
    }
}
