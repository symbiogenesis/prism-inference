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

/// Default 5-layer target hidden-state tap indices for `DFlash2` ternary block-diffusion
/// drafting on 64-layer `Qwen3.5-27B` (`Cyronius/Qwen3.8-27B-DFlash2-Bonsai2-ternary`).
pub const DFLASH2_DEFAULT_TAP_LAYERS: [usize; 5] = [2, 17, 32, 47, 63];

/// Validated configuration for `DFlash2` ternary block-diffusion speculative drafting.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DFlash2DraftConfig {
    /// Strictly ascending target decoder layer indices tapped to condition the drafter.
    pub tap_layers: [usize; 5],
    /// Number of draft tokens proposed in parallel per block-diffusion step (`1..=MAX_SPECULATIVE_DRAFT_TOKENS`).
    pub block_tokens: usize,
    /// Number of lightweight ternary cross-attention drafter layers (`1..=16`).
    pub drafter_layers: usize,
}

impl Default for DFlash2DraftConfig {
    fn default() -> Self {
        Self {
            tap_layers: DFLASH2_DEFAULT_TAP_LAYERS,
            block_tokens: 4,
            drafter_layers: 5,
        }
    }
}

impl DFlash2DraftConfig {
    /// Construct and validate a `DFlash2` block-diffusion drafter configuration against `target_layers`.
    ///
    /// # Errors
    /// Rejects non-ascending or out-of-range tap layer indices, zero or oversized `block_tokens`,
    /// or `drafter_layers` outside `1..=16`.
    pub fn new(
        tap_layers: [usize; 5],
        block_tokens: usize,
        drafter_layers: usize,
        target_layers: usize,
    ) -> Result<Self> {
        let config = Self {
            tap_layers,
            block_tokens,
            drafter_layers,
        };
        config.validate(target_layers)?;
        Ok(config)
    }

    /// Validate that all tap indices lie in `0..target_layers` in strictly ascending order
    /// and that `block_tokens` and `drafter_layers` are within contract limits.
    ///
    /// # Errors
    /// Rejects invalid tap ordering, out-of-bounds layer indices, or invalid block/layer counts.
    pub fn validate(&self, target_layers: usize) -> Result<()> {
        if self.block_tokens == 0
            || self.block_tokens > MAX_SPECULATIVE_DRAFT_TOKENS
            || self.drafter_layers == 0
            || self.drafter_layers > 16
            || self.tap_layers.windows(2).any(|pair| pair[0] >= pair[1])
            || self.tap_layers[4] >= target_layers
        {
            return Err(invalid(
                "invalid DFlash2 block-diffusion draft configuration",
            ));
        }
        Ok(())
    }
}

/// Projected per-token Gated `DeltaNet` (`GDN`) vectors captured during batched target
/// verification for `Rollback-by-Replay` state commitment.
#[derive(Clone, Debug, PartialEq)]
pub struct GdnVerifyStepTape {
    /// Post-conv, L2-normalized keys (`key_heads * key_dim`).
    pub keys: Vec<f32>,
    /// Post-conv, L2-normalized queries (`key_heads * key_dim`).
    pub queries: Vec<f32>,
    /// Post-conv values (`value_heads * value_dim`).
    pub values: Vec<f32>,
    /// Per-value-head exponential decay factors (`value_heads`).
    pub decays: Vec<f32>,
    /// Per-value-head sigmoid update gates (`value_heads`).
    pub betas: Vec<f32>,
}

/// Lightweight per-layer verification tape (`~164 KiB` for 8 tokens vs `3.117 MiB` full GDN state)
/// enabling `Rollback-by-Replay` when a speculative draft block is partially accepted.
///
/// Instead of cloning and restoring `149.625 MiB` of 48-layer GDN state across draft branches
/// or running a second full forward pass, the accepted prefix `0..accepted_steps` is replayed
/// directly into the base GDN state in-place with `0 ULP` bit-for-bit parity.
#[derive(Clone, Debug)]
pub struct GdnVerifyTape {
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    steps: Vec<GdnVerifyStepTape>,
}

impl GdnVerifyTape {
    /// Create a bounded GDN verification tape for a recurrent layer's head and channel geometry.
    ///
    /// # Errors
    /// Rejects zero dimensions, `value_heads % key_heads != 0`, or `max_steps > MAX_SPECULATIVE_DRAFT_TOKENS`.
    pub fn new(
        key_heads: usize,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        max_steps: usize,
    ) -> Result<Self> {
        if key_heads == 0
            || value_heads == 0
            || key_dim == 0
            || value_dim == 0
            || !value_heads.is_multiple_of(key_heads)
            || max_steps == 0
            || max_steps > MAX_SPECULATIVE_DRAFT_TOKENS
        {
            return Err(invalid("invalid GDN verification tape geometry"));
        }
        let mut steps = Vec::new();
        steps.try_reserve_exact(max_steps)?;
        Ok(Self {
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            steps,
        })
    }

    /// Number of recorded draft verification steps currently stored in the tape.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether no draft verification steps are currently recorded.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Clear all recorded steps while retaining allocated step capacity.
    pub fn clear(&mut self) {
        self.steps.clear();
    }

    /// Record one token's projected post-conv/post-L2-norm GDN inputs into the verification tape.
    ///
    /// # Errors
    /// Rejects capacity overflow (`>= MAX_SPECULATIVE_DRAFT_TOKENS`), mismatched slice lengths,
    /// or non-finite values.
    pub fn record_step(
        &mut self,
        keys: &[f32],
        queries: &[f32],
        values: &[f32],
        decays: &[f32],
        betas: &[f32],
    ) -> Result<()> {
        if self.steps.len() >= MAX_SPECULATIVE_DRAFT_TOKENS {
            return Err(invalid("GDN verification tape exceeded max draft steps"));
        }
        let expected_kq = self
            .key_heads
            .checked_mul(self.key_dim)
            .ok_or_else(|| invalid("GDN key dimension overflow"))?;
        let expected_v = self
            .value_heads
            .checked_mul(self.value_dim)
            .ok_or_else(|| invalid("GDN value dimension overflow"))?;
        if keys.len() != expected_kq
            || queries.len() != expected_kq
            || values.len() != expected_v
            || decays.len() != self.value_heads
            || betas.len() != self.value_heads
            || !crate::kernels::is_finite_slice(keys)
            || !crate::kernels::is_finite_slice(queries)
            || !crate::kernels::is_finite_slice(values)
            || !crate::kernels::is_finite_slice(decays)
            || !crate::kernels::is_finite_slice(betas)
        {
            return Err(invalid("invalid or nonfinite GDN verification step slice"));
        }
        self.steps.push(GdnVerifyStepTape {
            keys: keys.to_vec(),
            queries: queries.to_vec(),
            values: values.to_vec(),
            decays: decays.to_vec(),
            betas: betas.to_vec(),
        });
        Ok(())
    }

    /// Replay the first `accepted_steps` recorded tokens into `base_state` in-place
    /// using the pinned NEON/scalar GDN recurrent rule (`0 ULP` parity with sequential steps).
    ///
    /// # Errors
    /// Rejects `accepted_steps > self.steps.len()`, mismatched `base_state` length,
    /// or non-finite recurrent state updates.
    pub fn replay_accepted_prefix(
        &self,
        base_state: &mut [f32],
        accepted_steps: usize,
    ) -> Result<()> {
        let expected_state = self
            .value_heads
            .checked_mul(self.value_dim)
            .and_then(|rows| rows.checked_mul(self.key_dim))
            .ok_or_else(|| invalid("GDN state dimension overflow"))?;
        if accepted_steps > self.steps.len() || base_state.len() != expected_state {
            return Err(invalid("invalid GDN verification tape replay bounds"));
        }
        let group = self.value_heads / self.key_heads;
        let head_stride = self.value_dim * self.key_dim;
        for step in &self.steps[..accepted_steps] {
            for v_head in 0..self.value_heads {
                let k_head = v_head / group;
                let key = &step.keys[k_head * self.key_dim..(k_head + 1) * self.key_dim];
                let query = &step.queries[k_head * self.key_dim..(k_head + 1) * self.key_dim];
                let v_slice = &step.values[v_head * self.value_dim..(v_head + 1) * self.value_dim];
                let decay = step.decays[v_head];
                let beta = step.betas[v_head];
                let state_head = &mut base_state[v_head * head_stride..(v_head + 1) * head_stride];
                for (row_idx, &v_elem) in v_slice.iter().enumerate() {
                    let row_state =
                        &mut state_head[row_idx * self.key_dim..(row_idx + 1) * self.key_dim];
                    crate::kernels::pinned_gdn::step_row_in_place(
                        row_state, key, query, decay, v_elem, beta,
                    )?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DFLASH2_DEFAULT_TAP_LAYERS, DFlash2DraftConfig, GdnVerifyTape,
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

    #[test]
    fn dflash2_draft_config_validates_tap_layers_and_block_bounds() -> Result<()> {
        let cfg = DFlash2DraftConfig::default();
        cfg.validate(64)?;
        assert_eq!(cfg.tap_layers, DFLASH2_DEFAULT_TAP_LAYERS);
        assert!(DFlash2DraftConfig::new([2, 17, 17, 47, 63], 4, 5, 64).is_err());
        assert!(DFlash2DraftConfig::new([2, 17, 32, 47, 64], 4, 5, 64).is_err());
        assert!(DFlash2DraftConfig::new(DFLASH2_DEFAULT_TAP_LAYERS, 0, 5, 64).is_err());
        assert!(DFlash2DraftConfig::new(DFLASH2_DEFAULT_TAP_LAYERS, 9, 5, 64).is_err());
        Ok(())
    }

    #[test]
    fn gdn_verify_tape_replay_matches_sequential_step_zero_ulp() -> Result<()> {
        let key_heads = 2;
        let value_heads = 4;
        let key_dim = 16;
        let value_dim = 8;
        let state_len = value_heads * value_dim * key_dim;
        let mut sequential_state: Vec<f32> = (0..state_len)
            .map(|i| (f32::from(u16::try_from(i).unwrap_or(0)) - 128.0) * 0.005)
            .collect();
        let mut replay_state = sequential_state.clone();
        let mut tape = GdnVerifyTape::new(key_heads, value_heads, key_dim, value_dim, 4)?;
        assert!(tape.is_empty());

        for step_idx in 0..4_u8 {
            let s = f32::from(step_idx) + 1.0;
            let keys: Vec<f32> = (0..key_heads * key_dim)
                .map(|i| f32::from(u8::try_from(i).unwrap_or(0)).mul_add(0.03, s * 0.1))
                .collect();
            let queries: Vec<f32> = (0..key_heads * key_dim)
                .map(|i| f32::from(u8::try_from(i).unwrap_or(0)).mul_add(-0.02, s * 0.05))
                .collect();
            let values: Vec<f32> = (0..value_heads * value_dim)
                .map(|i| f32::from(u8::try_from(i).unwrap_or(0)).mul_add(0.04, -(s * 0.07)))
                .collect();
            let decays = vec![0.92_f32, 0.88, 0.95, 0.90];
            let betas = vec![0.25_f32, 0.30, 0.20, 0.35];
            tape.record_step(&keys, &queries, &values, &decays, &betas)?;

            // Advance sequential_state for only the first 2 accepted steps.
            if step_idx < 2 {
                let group = value_heads / key_heads;
                let head_stride = value_dim * key_dim;
                for v_head in 0..value_heads {
                    let k_head = v_head / group;
                    let key = &keys[k_head * key_dim..(k_head + 1) * key_dim];
                    let query = &queries[k_head * key_dim..(k_head + 1) * key_dim];
                    let v_slice = &values[v_head * value_dim..(v_head + 1) * value_dim];
                    let state_head =
                        &mut sequential_state[v_head * head_stride..(v_head + 1) * head_stride];
                    for (row_idx, &v_elem) in v_slice.iter().enumerate() {
                        let row_state = &mut state_head[row_idx * key_dim..(row_idx + 1) * key_dim];
                        crate::kernels::pinned_gdn::step_row_in_place(
                            row_state,
                            key,
                            query,
                            decays[v_head],
                            v_elem,
                            betas[v_head],
                        )?;
                    }
                }
            }
        }
        assert_eq!(tape.len(), 4);
        tape.replay_accepted_prefix(&mut replay_state, 2)?;
        for (lhs, rhs) in replay_state.iter().zip(&sequential_state) {
            assert_eq!(lhs.to_bits(), rhs.to_bits());
        }
        Ok(())
    }
}
