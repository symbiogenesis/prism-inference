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
    /// Total target-token forward evaluations, including batched proposals and
    /// accepted-prefix replay; excludes prompt preparation.
    pub target_evaluations: usize,
    /// Number of transactional decoder checkpoint restores triggered by rejected block steps.
    pub rollback_restores: usize,
    /// Total candidate draft positions verified across speculative rounds.
    pub draft_tokens_verified: u64,
    /// Count of rounds proposing a draft token at each block offset `0..MAX_SPECULATIVE_DRAFT_TOKENS`.
    pub position_proposed: [u64; MAX_SPECULATIVE_DRAFT_TOKENS],
    /// Count of rounds accepting the draft token at each block offset `0..MAX_SPECULATIVE_DRAFT_TOKENS`.
    pub position_accepted: [u64; MAX_SPECULATIVE_DRAFT_TOKENS],
    /// Histogram of accepted draft prefix lengths (`0..=MAX_SPECULATIVE_DRAFT_TOKENS`).
    pub accepted_len_histogram: [u64; MAX_SPECULATIVE_DRAFT_TOKENS + 1],
    /// Bonus tokens emitted after full-block draft acceptance.
    pub bonus_tokens: u64,
    /// Correction tokens emitted upon draft divergence.
    pub correction_tokens: u64,
    /// Single-token target evaluations emitted directly when no draft was proposed.
    pub fallback_tokens: u64,
    /// Number of accepted-prefix replay passes executed after batched rejection.
    pub replay_steps: u64,
    /// Total tokens replayed sequentially (`accepted_prefix + correction`) after batched rejection.
    pub replay_tokens: u64,
    /// Number of partial-rejection replay spans scheduled for folding into the next verification block.
    pub folded_replay_steps: u64,
    /// Total tokens (`accepted_prefix + correction`) carried in folded replay tapes across verification rounds.
    pub folded_replay_tokens: u64,
    /// Number of explicit deferred-replay flushes executed outside verification-block folding.
    pub folded_flushes: u64,
    /// Draft tokens proposed by [`MovingSourceCursor`] beyond exact suffix matches.
    pub cursor_draft_tokens: u64,
    /// Speculative rounds skipped because [`AdaptiveSpeculationGate`] selected an exact serial burst.
    pub adaptive_gate_skips: u64,
    /// Exact serial decode tokens generated inside adaptive break-even cooldown bursts.
    pub serial_burst_tokens: u64,
}

impl SpeculativeStats {
    /// Record an accepted-prefix replay span scheduled to fold into the following verification block.
    pub fn record_folded_carry(&mut self, carried_tokens: usize) {
        self.folded_replay_steps = self.folded_replay_steps.saturating_add(1);
        self.folded_replay_tokens = self
            .folded_replay_tokens
            .saturating_add(u64::try_from(carried_tokens).unwrap_or(u64::MAX));
    }

    /// Record an explicit flush of deferred folded replay state.
    pub const fn record_folded_flush(&mut self) {
        self.folded_flushes = self.folded_flushes.saturating_add(1);
    }

    /// Record draft tokens supplied by [`MovingSourceCursor`].
    pub fn record_cursor_draft(&mut self, tokens: usize) {
        self.cursor_draft_tokens = self
            .cursor_draft_tokens
            .saturating_add(u64::try_from(tokens).unwrap_or(u64::MAX));
    }

    /// Record an exact serial decode token executed under [`AdaptiveSpeculationGate`].
    pub const fn record_adaptive_serial_burst(&mut self, increment_skip: bool) {
        if increment_skip {
            self.adaptive_gate_skips = self.adaptive_gate_skips.saturating_add(1);
        }
        self.serial_burst_tokens = self.serial_burst_tokens.saturating_add(1);
    }
}

/// Result of proposing speculative draft tokens from [`SuffixTree`] plus [`MovingSourceCursor`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CursorDraft {
    /// Total draft tokens written into the caller buffer (`<= MAX_SPECULATIVE_DRAFT_TOKENS`).
    pub len: usize,
    /// Leading tokens supplied by an exact [`SuffixTree`] match.
    pub suffix_tokens: usize,
    /// Tokens supplied or bridged by [`MovingSourceCursor`].
    pub cursor_tokens: usize,
}

/// Bounded moving source cursor for bridging minor local edits during copying and code generation.
///
/// Tracks the most recent matched continuation position inside [`SuffixTree::history`] and searches
/// only a small bounded window around that cursor when exact suffix matching misses after a minor
/// local substitution, insertion, or deletion. Deactivates automatically after consecutive misses.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MovingSourceCursor {
    anchor: Option<usize>,
    active_draft_start: Option<usize>,
    stall_misses: usize,
    resync_window: usize,
    max_stall_misses: usize,
}

impl Default for MovingSourceCursor {
    fn default() -> Self {
        Self::new()
    }
}

impl MovingSourceCursor {
    /// Default bounded forward/local window used to re-align the cursor after minor edits.
    pub const DEFAULT_RESYNC_WINDOW: usize = 16;
    /// Default consecutive miss budget before deactivating an anchored cursor.
    pub const DEFAULT_MAX_STALL_MISSES: usize = 2;

    /// Create an inactive cursor with default bounded resync parameters.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            anchor: None,
            active_draft_start: None,
            stall_misses: 0,
            resync_window: Self::DEFAULT_RESYNC_WINDOW,
            max_stall_misses: Self::DEFAULT_MAX_STALL_MISSES,
        }
    }

    /// Clear any active source cursor state.
    pub const fn reset(&mut self) {
        self.anchor = None;
        self.active_draft_start = None;
        self.stall_misses = 0;
    }

    /// Current cursor position inside the suffix history, when active.
    #[must_use]
    pub const fn anchor(&self) -> Option<usize> {
        self.anchor
    }

    /// Propose up to `remaining_tokens` draft tokens from exact suffix match or active cursor.
    ///
    /// # Errors
    /// Propagates invalid [`SuffixDraftConfig`] or suffix tree index overflow errors.
    pub fn propose_with_tree(
        &mut self,
        tree: &SuffixTree,
        config: SuffixDraftConfig,
        remaining_tokens: usize,
        storage: &mut [u32; MAX_SPECULATIVE_DRAFT_TOKENS],
    ) -> Result<CursorDraft> {
        config.validate()?;
        let limit = config.max_draft_tokens.min(remaining_tokens);
        if limit == 0 {
            self.active_draft_start = None;
            return Ok(CursorDraft::default());
        }
        let (suffix_len, suffix_start) = tree.propose_into_with_start(config, limit, storage)?;
        if suffix_len > 0
            && let Some(start) = suffix_start
        {
            self.anchor = Some(start);
            self.active_draft_start = Some(start);
            self.stall_misses = 0;
            return Ok(CursorDraft {
                len: suffix_len,
                suffix_tokens: suffix_len,
                cursor_tokens: 0,
            });
        }

        let Some(pos) = self.anchor else {
            self.active_draft_start = None;
            return Ok(CursorDraft::default());
        };
        let history = tree.history();
        let available = history.len().saturating_sub(pos).min(limit);
        if available == 0 {
            self.reset();
            return Ok(CursorDraft::default());
        }
        storage[..available].copy_from_slice(&history[pos..pos + available]);
        self.active_draft_start = Some(pos);
        Ok(CursorDraft {
            len: available,
            suffix_tokens: 0,
            cursor_tokens: available,
        })
    }

    /// Update the bounded source cursor after committing one verification or serial step.
    ///
    /// `history` must already contain all newly committed tokens (including bonus or correction).
    pub fn observe_round(
        &mut self,
        history: &[u32],
        proposed: usize,
        accepted: usize,
        emitted_bonus: bool,
    ) {
        let Some(last) = history.last().copied() else {
            self.reset();
            return;
        };
        let Some(start) = self.active_draft_start.take().or(self.anchor) else {
            return;
        };
        if proposed == 0 {
            self.resync_around(history, start, last, true);
            return;
        }
        let base = start.saturating_add(accepted);
        if accepted >= proposed {
            if emitted_bonus && history.get(base).copied() == Some(last) {
                self.anchor = Some(base.saturating_add(1));
                self.stall_misses = 0;
            } else if emitted_bonus {
                self.resync_around(history, base, last, true);
            } else {
                self.anchor = Some(base);
                self.stall_misses = 0;
            }
            return;
        }

        // Partial or zero acceptance ending with one newly committed correction token `last`.
        if accepted > 0 {
            self.stall_misses = 0;
            if let Some(aligned) = self.find_local_anchor(history, base, last, true) {
                self.anchor = Some(aligned);
            } else {
                self.anchor = Some(base.saturating_add(1));
            }
            return;
        }

        self.stall_misses = self.stall_misses.saturating_add(1);
        if self.stall_misses >= self.max_stall_misses {
            self.reset();
            return;
        }
        if let Some(aligned) = self.find_local_anchor(history, base, last, false) {
            self.anchor = Some(aligned);
        } else {
            self.anchor = Some(base.saturating_add(1));
        }
    }

    fn resync_around(&mut self, history: &[u32], base: usize, last: u32, allow_single_jump: bool) {
        if history.get(base).copied() == Some(last) {
            self.anchor = Some(base.saturating_add(1));
            self.stall_misses = 0;
            return;
        }
        if let Some(aligned) = self.find_local_anchor(history, base, last, allow_single_jump) {
            self.anchor = Some(aligned);
            self.stall_misses = 0;
            return;
        }
        self.stall_misses = self.stall_misses.saturating_add(1);
        if self.stall_misses >= self.max_stall_misses {
            self.reset();
        } else {
            self.anchor = Some(base.saturating_add(1));
        }
    }

    fn find_local_anchor(
        &self,
        history: &[u32],
        base: usize,
        last: u32,
        allow_single_jump: bool,
    ) -> Option<usize> {
        // Keep the search strictly prior to the newly committed tail token(s).
        let tail_guard = history.len().saturating_sub(2);
        let lo = base.saturating_sub(2).max(1);
        let hi = base.saturating_add(self.resync_window).min(tail_guard);
        if history.len() >= 2 && lo < hi {
            let prev = history[history.len() - 2];
            for idx in lo..hi {
                if history[idx - 1] == prev && history[idx] == last {
                    return Some(idx + 1);
                }
            }
        }
        if allow_single_jump {
            let jump_lo = base.saturating_add(1);
            let jump_hi = base
                .saturating_add(MAX_SPECULATIVE_DRAFT_TOKENS)
                .min(history.len().saturating_sub(1));
            for (idx, &token) in history.iter().enumerate().take(jump_hi).skip(jump_lo) {
                if token == last {
                    return Some(idx + 1);
                }
            }
        }
        None
    }
}

/// Decision returned by [`AdaptiveSpeculationGate::decide`] before a decode step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptiveGateDecision {
    /// Attempt suffix/cursor drafting and batched verification.
    Speculate,
    /// Skip speculation for this step and emit an exact serial decode token.
    SerialBurst,
}

/// Configuration for the self-calibrating [`AdaptiveSpeculationGate`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaptiveGateConfig {
    /// Enable adaptive break-even throttling (`false` always allows speculation).
    pub enabled: bool,
    /// Minimum verified speculative rounds before throttling may engage.
    pub warmup_rounds: usize,
    /// EWMA smoothing factor in `(0.0, 1.0]` for acceptance and step timing estimates.
    pub smoothing: f64,
    /// Prior `verify_block / serial_step` cost ratio used before empirical serial/verify samples arrive.
    pub initial_verify_to_serial_ratio: f64,
    /// Multiplier applied to the empirical break-even emission rate (`1.0` = exact break-even).
    pub break_even_margin: f64,
    /// Initial exact serial burst length (in tokens) when rolling acceptance drops below break-even.
    pub min_burst_tokens: usize,
    /// Maximum exact serial burst length (in tokens) after repeated low-acceptance probes.
    pub max_burst_tokens: usize,
}

impl Default for AdaptiveGateConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            warmup_rounds: 3,
            smoothing: 0.25,
            initial_verify_to_serial_ratio: 2.0,
            break_even_margin: 1.0,
            min_burst_tokens: 4,
            max_burst_tokens: 32,
        }
    }
}

impl AdaptiveGateConfig {
    /// Validate gate configuration bounds.
    ///
    /// # Errors
    /// Rejects non-finite parameters, invalid smoothing or margin ranges, or inverted burst limits.
    pub fn validate(&self) -> Result<()> {
        if !self.smoothing.is_finite() || self.smoothing <= 0.0 || self.smoothing > 1.0 {
            return Err(invalid("adaptive gate smoothing factor must lie in (0, 1]"));
        }
        if !self.initial_verify_to_serial_ratio.is_finite()
            || self.initial_verify_to_serial_ratio < 1.0
        {
            return Err(invalid(
                "adaptive gate initial verify/serial ratio must be finite and >= 1.0",
            ));
        }
        if !self.break_even_margin.is_finite()
            || self.break_even_margin <= 0.0
            || self.break_even_margin > 4.0
        {
            return Err(invalid(
                "adaptive gate break-even margin must lie in (0, 4]",
            ));
        }
        if self.min_burst_tokens == 0 || self.min_burst_tokens > self.max_burst_tokens {
            return Err(invalid(
                "adaptive gate burst token bounds must satisfy 1 <= min <= max",
            ));
        }
        Ok(())
    }
}

/// Self-calibrating adaptive break-even speculation gate.
///
/// Tracks rolling `verify / serial` latency (including folded flush overhead) and rolling
/// emitted tokens per verified round. When empirical rolling acceptance falls below the measured
/// break-even rate, the gate pauses speculation for a bounded exact serial burst before probing
/// again so creative Thinking-OFF prose stays near native serial throughput while code/structured
/// regions recover immediately.
#[derive(Clone, Debug)]
pub struct AdaptiveSpeculationGate {
    config: AdaptiveGateConfig,
    verified_rounds: usize,
    serial_steps: usize,
    ewma_emitted_per_verify: f64,
    ewma_verify_ns: Option<f64>,
    ewma_serial_ns: Option<f64>,
    burst_remaining: usize,
    next_burst_tokens: usize,
    awaiting_probe: bool,
}

impl AdaptiveSpeculationGate {
    /// Create a new adaptive gate from a validated [`AdaptiveGateConfig`].
    ///
    /// # Errors
    /// Propagates [`AdaptiveGateConfig::validate`] failures.
    pub fn new(config: AdaptiveGateConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            verified_rounds: 0,
            serial_steps: 0,
            ewma_emitted_per_verify: config.initial_verify_to_serial_ratio,
            ewma_verify_ns: None,
            ewma_serial_ns: None,
            burst_remaining: 0,
            next_burst_tokens: config.min_burst_tokens,
            awaiting_probe: false,
        })
    }

    /// Gate configuration.
    #[must_use]
    pub const fn config(&self) -> AdaptiveGateConfig {
        self.config
    }

    /// Current rolling `verify_block / serial_step` cost ratio (`>= 1.0`).
    #[must_use]
    pub fn verify_to_serial_ratio(&self) -> f64 {
        if let (Some(verify), Some(serial)) = (self.ewma_verify_ns, self.ewma_serial_ns)
            && serial > 0.0
        {
            (verify / serial).clamp(1.0, 16.0)
        } else {
            self.config.initial_verify_to_serial_ratio
        }
    }

    /// Minimum rolling tokens emitted per verification block needed to break even with serial decode.
    #[must_use]
    pub fn break_even_emitted_tokens(&self) -> f64 {
        (self.verify_to_serial_ratio() * self.config.break_even_margin).max(1.0)
    }

    /// Rolling EWMA of emitted tokens per verified round.
    #[must_use]
    pub const fn rolling_emitted_per_verify(&self) -> f64 {
        self.ewma_emitted_per_verify
    }

    /// Decide whether the next step should attempt speculative drafting or execute an exact serial burst.
    pub const fn decide(&mut self) -> AdaptiveGateDecision {
        if !self.config.enabled {
            return AdaptiveGateDecision::Speculate;
        }
        if self.burst_remaining > 0 {
            self.burst_remaining -= 1;
            if self.burst_remaining == 0 {
                self.awaiting_probe = true;
            }
            return AdaptiveGateDecision::SerialBurst;
        }
        AdaptiveGateDecision::Speculate
    }

    /// Record an observed exact serial step duration (`fallback`, serial burst, or single-token decode).
    pub fn record_serial_step(&mut self, duration: std::time::Duration) {
        self.serial_steps = self.serial_steps.saturating_add(1);
        let ns = duration_to_ns_f64(duration);
        if ns > 0.0 {
            let alpha = self.config.smoothing;
            self.ewma_serial_ns = Some(
                self.ewma_serial_ns
                    .map_or(ns, |prev| alpha.mul_add(ns, (1.0 - alpha) * prev)),
            );
        }
    }

    /// Charge explicit folded-replay flush overhead against the speculation verify cost estimate.
    pub fn charge_flush_overhead(&mut self, flush_duration: std::time::Duration) {
        let flush_ns = duration_to_ns_f64(flush_duration);
        if flush_ns > 0.0 {
            let alpha = self.config.smoothing;
            self.ewma_verify_ns = Some(
                self.ewma_verify_ns
                    .map_or(flush_ns, |prev| alpha.mul_add(flush_ns, prev)),
            );
        }
    }

    /// Record one verified speculative round and update rolling cost and break-even state.
    pub fn record_verify_round(
        &mut self,
        proposed: usize,
        _accepted: usize,
        emitted: usize,
        verify_duration: std::time::Duration,
        flush_duration: std::time::Duration,
    ) {
        if proposed == 0 {
            self.awaiting_probe = false;
            return;
        }
        self.verified_rounds = self.verified_rounds.saturating_add(1);
        let alpha = self.config.smoothing;
        let total_ns = duration_to_ns_f64(verify_duration.saturating_add(flush_duration));
        if total_ns > 0.0 {
            self.ewma_verify_ns = Some(self.ewma_verify_ns.map_or(total_ns, |prev| {
                alpha.mul_add(total_ns, (1.0 - alpha) * prev)
            }));
        }

        let emitted_f64 = u64_to_f64(u64::try_from(emitted).unwrap_or(u64::MAX));
        let threshold = self.break_even_emitted_tokens();
        if self.verified_rounds == 1 {
            self.ewma_emitted_per_verify = emitted_f64;
        } else if self.awaiting_probe {
            self.awaiting_probe = false;
            if emitted_f64 >= threshold {
                self.ewma_emitted_per_verify = self.ewma_emitted_per_verify.max(emitted_f64);
                self.next_burst_tokens = self.config.min_burst_tokens;
                self.burst_remaining = 0;
                return;
            }
            self.ewma_emitted_per_verify =
                alpha.mul_add(emitted_f64, (1.0 - alpha) * self.ewma_emitted_per_verify);
            if self.config.enabled {
                self.burst_remaining = self.next_burst_tokens;
                self.next_burst_tokens = self
                    .next_burst_tokens
                    .saturating_mul(2)
                    .min(self.config.max_burst_tokens);
            }
            return;
        } else {
            self.ewma_emitted_per_verify =
                alpha.mul_add(emitted_f64, (1.0 - alpha) * self.ewma_emitted_per_verify);
        }

        if self.config.enabled && self.verified_rounds >= self.config.warmup_rounds {
            if self.ewma_emitted_per_verify < threshold {
                self.burst_remaining = self.next_burst_tokens;
                self.next_burst_tokens = self
                    .next_burst_tokens
                    .saturating_mul(2)
                    .min(self.config.max_burst_tokens);
            } else {
                self.next_burst_tokens = self.config.min_burst_tokens;
            }
        }
    }
}

fn duration_to_ns_f64(duration: std::time::Duration) -> f64 {
    let nanos = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
    u64_to_f64(nanos)
}

fn u64_to_f64(value: u64) -> f64 {
    let hi = u32::try_from(value >> 32).unwrap_or(u32::MAX);
    let lo = u32::try_from(value & u64::from(u32::MAX)).unwrap_or(u32::MAX);
    f64::from(hi).mul_add(4_294_967_296.0, f64::from(lo))
}

/// Outcome recorded for one verified speculative round or unspeculated fallback step.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VerifiedRound {
    /// Number of proposed draft tokens in this round (`0` for unspeculated fallback).
    pub proposed: usize,
    /// Number of proposed draft tokens accepted from index `0`.
    pub accepted: usize,
    /// Number of proposed draft positions inspected or batched-verified.
    pub verified: usize,
    /// Target forward evaluations executed during this round.
    pub target_evaluations: usize,
    /// Whether this round triggered rollback / divergence recovery.
    pub rolled_back: bool,
    /// Bonus tokens emitted after full acceptance (`0` or `1`).
    pub bonus_tokens: u64,
    /// Correction tokens emitted on divergence (`0` or `1`).
    pub correction_tokens: u64,
    /// Direct fallback tokens emitted when no draft proposal existed (`0` or `1`).
    pub fallback_tokens: u64,
    /// Total tokens replayed sequentially after batched rejection (`accepted + correction`).
    pub replay_tokens: usize,
}

impl SpeculativeStats {
    /// Fraction of proposed draft tokens accepted by the target verifier in `[0.0, 1.0]`.
    #[must_use]
    #[expect(
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

    /// Mean accepted draft tokens per non-empty speculative round (`0.0` when none entered).
    #[must_use]
    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "Diagnostic mean accepted length converts bounded token counts to f64."
    )]
    pub const fn mean_accepted_length(&self) -> f64 {
        if self.draft_rounds == 0 {
            0.0
        } else {
            (self.draft_tokens_accepted as f64) / (self.draft_rounds as f64)
        }
    }

    /// Fraction of proposed tokens accepted at zero-based draft offset `position`,
    /// or `None` when out of bounds or never proposed.
    #[must_use]
    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "Diagnostic per-position acceptance ratio converts bounded u64 counts to f64."
    )]
    pub const fn position_acceptance_rate(&self, position: usize) -> Option<f64> {
        if position >= MAX_SPECULATIVE_DRAFT_TOKENS {
            return None;
        }
        let proposed = self.position_proposed[position];
        if proposed == 0 {
            None
        } else {
            Some((self.position_accepted[position] as f64) / (proposed as f64))
        }
    }

    /// Record one speculative verification round using core round counters.
    pub fn record_round(
        &mut self,
        proposed: usize,
        accepted: usize,
        target_evaluations: usize,
        rolled_back: bool,
    ) {
        self.record_verified_round(VerifiedRound {
            proposed,
            accepted,
            verified: proposed,
            target_evaluations,
            rolled_back,
            bonus_tokens: 0,
            correction_tokens: 0,
            fallback_tokens: 0,
            replay_tokens: 0,
        });
    }

    /// Record full per-position, bonus/correction/fallback, and replay telemetry for one round.
    pub fn record_verified_round(&mut self, round: VerifiedRound) {
        if round.proposed > 0 {
            self.draft_rounds = self.draft_rounds.saturating_add(1);
            self.draft_tokens_proposed = self.draft_tokens_proposed.saturating_add(round.proposed);
            self.draft_tokens_accepted = self.draft_tokens_accepted.saturating_add(round.accepted);
            let proposed_positions = round.proposed.min(MAX_SPECULATIVE_DRAFT_TOKENS);
            let accepted_positions = round
                .accepted
                .min(proposed_positions)
                .min(MAX_SPECULATIVE_DRAFT_TOKENS);
            for slot in self.position_proposed.iter_mut().take(proposed_positions) {
                *slot = slot.saturating_add(1);
            }
            for slot in self.position_accepted.iter_mut().take(accepted_positions) {
                *slot = slot.saturating_add(1);
            }
            self.accepted_len_histogram[accepted_positions] =
                self.accepted_len_histogram[accepted_positions].saturating_add(1);
        }
        self.target_evaluations = self
            .target_evaluations
            .saturating_add(round.target_evaluations);
        if round.rolled_back {
            self.rollback_restores = self.rollback_restores.saturating_add(1);
        }
        self.draft_tokens_verified = self
            .draft_tokens_verified
            .saturating_add(u64::try_from(round.verified).unwrap_or(u64::MAX));
        self.bonus_tokens = self.bonus_tokens.saturating_add(round.bonus_tokens);
        self.correction_tokens = self
            .correction_tokens
            .saturating_add(round.correction_tokens);
        self.fallback_tokens = self.fallback_tokens.saturating_add(round.fallback_tokens);
        if round.replay_tokens > 0 {
            self.replay_steps = self.replay_steps.saturating_add(1);
            self.replay_tokens = self
                .replay_tokens
                .saturating_add(u64::try_from(round.replay_tokens).unwrap_or(u64::MAX));
        }
    }
}

/// Compute cumulative rejection complement `1.0 - prod_i (1.0 - r_i)` stably
/// via the exact survival-complement recurrence `c <- c + r_i * (1.0 - c)`.
///
/// # Errors
/// Rejects any non-finite rejection probability or value outside `[0.0, 1.0]`.
pub fn cumulative_rejection_complement(rejection_probs: &[f64]) -> Result<f64> {
    let mut complement = 0.0_f64;
    for &prob in rejection_probs {
        if !prob.is_finite() || !(0.0..=1.0).contains(&prob) {
            return Err(invalid(
                "cumulative rejection probabilities must be finite and within [0.0, 1.0]",
            ));
        }
        complement = prob.mul_add(1.0 - complement, complement);
    }
    Ok(complement.clamp(0.0, 1.0))
}

/// Propose continuation tokens from prompt/history suffix lookup first,
/// invoking the fallback drafter only when lookup finds no continuation.
///
/// # Errors
/// Rejects invalid [`SuffixDraftConfig`] or errors from lookup or fallback drafting.
pub fn propose_lookup_then_fallback<F>(
    history: &[u32],
    lookup_config: SuffixDraftConfig,
    remaining_tokens: usize,
    fallback: F,
) -> Result<Vec<u32>>
where
    F: FnOnce(&[u32], usize) -> Result<Vec<u32>>,
{
    lookup_config.validate()?;
    let limit = lookup_config.max_draft_tokens.min(remaining_tokens);
    if limit == 0 {
        return Ok(Vec::new());
    }
    let lookup = SuffixDrafter::propose(history, lookup_config, limit)?;
    if !lookup.is_empty() {
        return Ok(lookup);
    }
    let mut secondary = fallback(history, limit)?;
    if secondary.len() > limit {
        secondary.truncate(limit);
    }
    Ok(secondary)
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
        let continuation = Self::continuation(history, config, remaining_tokens)?;
        let mut draft = Vec::new();
        draft.try_reserve_exact(continuation.len())?;
        draft.extend_from_slice(continuation);
        Ok(draft)
    }

    #[cfg(test)]
    pub(crate) fn propose_into(
        history: &[u32],
        config: SuffixDraftConfig,
        remaining_tokens: usize,
        storage: &mut [u32; MAX_SPECULATIVE_DRAFT_TOKENS],
    ) -> Result<usize> {
        let continuation = Self::continuation(history, config, remaining_tokens)?;
        storage[..continuation.len()].copy_from_slice(continuation);
        Ok(continuation.len())
    }

    fn continuation(
        history: &[u32],
        config: SuffixDraftConfig,
        remaining_tokens: usize,
    ) -> Result<&[u32]> {
        config.validate()?;
        let limit = config.max_draft_tokens.min(remaining_tokens);
        if limit == 0 || history.len() <= config.min_match_tokens {
            return Ok(&[]);
        }
        let max_len = config.max_match_tokens.min(history.len() - 1);
        let Some(continuation_start) =
            suffix_match_start(history, config.min_match_tokens, max_len, |left, right| {
                left == right
            })?
        else {
            return Ok(&[]);
        };
        let available = (history.len() - continuation_start).min(limit);
        Ok(&history[continuation_start..continuation_start + available])
    }
}

// Search the reversed history against its bounded prefix (the current suffix
// in forward order). A Z box reuses matches already proved by earlier starts;
// only extensions beyond its right edge require new successful comparisons.
// Each scan has at most one failed equality per start, hence fewer than
// 2 * (history.len() + maximum) token comparisons including the prefix table.
// The equality parameter lets controls count the actual production algorithm.
fn suffix_match_start(
    history: &[u32],
    minimum: usize,
    maximum: usize,
    mut equal: impl FnMut(u32, u32) -> bool,
) -> Result<Option<usize>> {
    // Only validated private callers enter: 1 <= minimum <= maximum <= 64,
    // and maximum < history.len(). Every table entry therefore fits in u8.
    let length = history.len();
    let mut prefix = [0_u8; MAX_SUFFIX_MATCH_TOKENS];
    let mut box_start = 0;
    let mut box_end = 0;
    for offset in 1..maximum {
        let mut matched = if offset < box_end {
            usize::from(prefix[offset - box_start]).min(box_end - offset)
        } else {
            0
        };
        while matched < maximum - offset
            && equal(
                history[length - matched - 1],
                history[length - offset - matched - 1],
            )
        {
            matched += 1;
        }
        prefix[offset] = u8::try_from(matched)?;
        if offset + matched > box_end {
            box_start = offset;
            box_end = offset + matched;
        }
    }
    box_start = 0;
    box_end = 0;
    let mut best_len = minimum - 1;
    let mut continuation_start = None;
    // Reversed offsets grow from one, excluding the current suffix itself.
    // Their forward match ends decrease, retaining the most recent equal-length
    // match while allowing the same overlapping spans as the original search.
    for offset in 1..=length - minimum {
        let end = length - offset;
        if end <= best_len {
            break;
        }
        let mut matched = if offset < box_end {
            usize::from(prefix[offset - box_start]).min(box_end - offset)
        } else {
            0
        };
        while matched < maximum.min(end)
            && equal(history[length - matched - 1], history[end - matched - 1])
        {
            matched += 1;
        }
        if offset + matched > box_end {
            box_start = offset;
            box_end = offset + matched;
        }
        if matched > best_len {
            best_len = matched;
            continuation_start = Some(end);
            if matched == maximum {
                break;
            }
        }
    }
    Ok(continuation_start)
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
            || self.tap_layers.array_windows().any(|[a, b]| a >= b)
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
        if accepted_steps == 0 {
            return Ok(());
        }
        let group = self.value_heads / self.key_heads;
        let head_stride = self.value_dim * self.key_dim;
        let mut staged = Vec::new();
        staged.try_reserve_exact(base_state.len())?;
        staged.extend_from_slice(base_state);
        for v_head in 0..self.value_heads {
            let k_head = v_head / group;
            let mut inputs = [crate::kernels::pinned_gdn::ChunkInput {
                query: &[],
                key: &[],
                value: &[],
                decay: 0.0,
                beta: 0.0,
            }; MAX_SPECULATIVE_DRAFT_TOKENS];
            for (token, step) in self.steps[..accepted_steps].iter().enumerate() {
                inputs[token] = crate::kernels::pinned_gdn::ChunkInput {
                    query: &step.queries[k_head * self.key_dim..(k_head + 1) * self.key_dim],
                    key: &step.keys[k_head * self.key_dim..(k_head + 1) * self.key_dim],
                    value: &step.values[v_head * self.value_dim..(v_head + 1) * self.value_dim],
                    decay: step.decays[v_head],
                    beta: step.betas[v_head],
                };
            }
            let state_head = &mut staged[v_head * head_stride..(v_head + 1) * head_stride];
            crate::kernels::pinned_gdn::step_head_tokens_in_place(
                state_head,
                self.value_dim,
                self.key_dim,
                &inputs[..accepted_steps],
            )?;
            if !crate::kernels::is_finite_slice(state_head) {
                return Err(invalid("nonfinite GDN verification tape replay state"));
            }
        }
        base_state.copy_from_slice(&staged);
        Ok(())
    }
}

/// Maximum number of history tokens indexed by one incremental [`SuffixTree`].
pub const MAX_SUFFIX_TREE_TOKENS: usize = 262_144;

/// Maximum nodes supported in one [`DraftTreeTopology`] causal verification mask.
pub const MAX_DRAFT_TREE_NODES: usize = 16;

const NO_INDEX: u32 = u32::MAX;
const MAX_TRACKED_CONT_PER_NODE: usize = 8;

#[derive(Clone, Copy, Debug)]
struct ContEntry {
    token: u32,
    count: u32,
    next: u32,
}

#[derive(Clone, Copy, Debug)]
struct SuffixEdge {
    first_token: u32,
    child: u32,
    next: u32,
}

#[derive(Clone, Copy, Debug)]
struct SuffixNode {
    edge_end: u32,
    parent_depth: u8,
    depth: u8,
    last_end: u32,
    total_count: u32,
    cont_len: u8,
    first_cont: u32,
    first_edge: u32,
}

impl SuffixNode {
    const fn empty() -> Self {
        Self {
            edge_end: 0,
            parent_depth: 0,
            depth: 0,
            last_end: 0,
            total_count: 0,
            cont_len: 0,
            first_cont: NO_INDEX,
            first_edge: NO_INDEX,
        }
    }
}

/// Summary returned by [`SuffixTree::propose_adaptive`] alongside the drafted tokens.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AdaptiveSuffixDraft {
    /// Number of proposed draft continuation tokens written into the caller buffer.
    pub len: usize,
    /// Suffix match depth observed when proposing the first draft token.
    pub matched_tokens: usize,
    /// Cumulative empirical continuation confidence in `[0.0, 1.0]` across the accepted chain.
    pub confidence: f64,
}

/// Incremental compressed reverse-suffix radix trie over committed prompt and decode tokens.
///
/// Supports $O(\text{max\_match\_tokens})$ speculative continuation queries independent of
/// committed context length up to [`MAX_SUFFIX_TREE_TOKENS`], while tracking per-node
/// recency and bounded Misra-Gries empirical continuation counts for adaptive gating.
#[derive(Clone, Debug, Default)]
pub struct SuffixTree {
    history: Vec<u32>,
    nodes: Vec<SuffixNode>,
    edges: Vec<SuffixEdge>,
    cont_entries: Vec<ContEntry>,
}

impl SuffixTree {
    /// Create an empty incremental suffix radix tree with no upfront heap allocation.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            history: Vec::new(),
            nodes: Vec::new(),
            edges: Vec::new(),
            cont_entries: Vec::new(),
        }
    }

    /// Create an empty incremental suffix radix tree with preallocated capacity for `max_tokens`.
    ///
    /// # Errors
    /// Rejects `max_tokens > MAX_SUFFIX_TREE_TOKENS` or fallible allocation exhaustion.
    pub fn with_capacity(max_tokens: usize) -> Result<Self> {
        if max_tokens > MAX_SUFFIX_TREE_TOKENS {
            return Err(invalid("suffix tree capacity exceeds maximum token bound"));
        }
        let mut tree = Self::new();
        if max_tokens > 0 {
            tree.history.try_reserve_exact(max_tokens)?;
            tree.nodes.try_reserve(max_tokens.saturating_mul(2))?;
            tree.edges.try_reserve(max_tokens)?;
            tree.cont_entries
                .try_reserve(max_tokens.saturating_mul(2))?;
        }
        Ok(tree)
    }

    /// Number of history tokens currently indexed by this tree.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.history.len()
    }

    /// Whether no history tokens are indexed in this tree.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Remove all indexed tokens while preserving allocated slab capacities.
    pub fn clear(&mut self) {
        self.history.clear();
        self.nodes.clear();
        self.edges.clear();
        self.cont_entries.clear();
    }

    /// Synchronize the suffix tree incrementally against a target committed token history.
    ///
    /// When `tokens` extends the previously indexed sequence, only newly committed suffix
    /// tokens are inserted; rollbacks or branch switches rebuild cleanly within capacity.
    ///
    /// # Errors
    /// Rejects `tokens.len() > MAX_SUFFIX_TREE_TOKENS` or fallible allocation failure.
    pub fn sync(&mut self, tokens: &[u32]) -> Result<()> {
        if tokens.len() > MAX_SUFFIX_TREE_TOKENS {
            return Err(invalid("suffix tree history exceeds maximum token bound"));
        }
        if tokens.len() >= self.history.len() && tokens.starts_with(&self.history) {
            for &token in &tokens[self.history.len()..] {
                self.push(token)?;
            }
            return Ok(());
        }
        self.clear();
        self.history.try_reserve(tokens.len())?;
        self.nodes.try_reserve(tokens.len().saturating_mul(2))?;
        self.edges.try_reserve(tokens.len())?;
        for &token in tokens {
            self.push(token)?;
        }
        Ok(())
    }

    /// Append one committed token to the history and index the completed preceding suffix.
    ///
    /// # Errors
    /// Rejects growth past [`MAX_SUFFIX_TREE_TOKENS`] or fallible allocation failure.
    pub fn push(&mut self, token: u32) -> Result<()> {
        if self.history.len() >= MAX_SUFFIX_TREE_TOKENS {
            return Err(invalid("suffix tree history exceeds maximum token bound"));
        }
        self.history.try_reserve(1)?;
        self.nodes.try_reserve(2)?;
        self.edges.try_reserve(2)?;
        self.cont_entries
            .try_reserve(MAX_SUFFIX_MATCH_TOKENS + 1 + MAX_TRACKED_CONT_PER_NODE)?;

        if self.nodes.is_empty() {
            self.nodes.push(SuffixNode::empty());
        }
        let end = self.history.len();
        self.history.push(token);
        if end == 0 {
            return Ok(());
        }

        let max_depth = end.min(MAX_SUFFIX_MATCH_TOKENS);
        let end_u32 = u32::try_from(end)?;
        let max_depth_u8 = u8::try_from(max_depth)?;
        self.nodes[0].last_end = end_u32;
        self.nodes[0].total_count = self.nodes[0].total_count.saturating_add(1);

        let mut node_idx = 0_usize;
        let mut depth = 0_usize;
        while depth < max_depth {
            let next_rev_token = self.history[end - 1 - depth];
            let Some((edge_idx, child_u32)) = self.find_edge(node_idx, next_rev_token) else {
                self.attach_leaf(node_idx, end_u32, depth, max_depth_u8, token)?;
                break;
            };
            let child_idx = usize::try_from(child_u32)?;
            let child = self.nodes[child_idx];
            let child_end = usize::try_from(child.edge_end)?;
            let edge_len = usize::from(child.depth - child.parent_depth);
            let compare_len = edge_len.min(max_depth - depth);
            let mut matched = 1_usize;
            while matched < compare_len {
                let hist_tok = self.history[child_end - 1 - (depth + matched)];
                let new_tok = self.history[end - 1 - (depth + matched)];
                if hist_tok != new_tok {
                    break;
                }
                matched += 1;
            }
            if matched < edge_len {
                let split_depth = depth + matched;
                let internal_u32 =
                    self.split_child(edge_idx, child_idx, split_depth, end_u32, token)?;
                if split_depth < max_depth {
                    let internal_idx = usize::try_from(internal_u32)?;
                    self.attach_leaf(internal_idx, end_u32, split_depth, max_depth_u8, token)?;
                }
                break;
            }
            self.nodes[child_idx].last_end = end_u32;
            self.nodes[child_idx].total_count = self.nodes[child_idx].total_count.saturating_add(1);
            self.record_node_continuation(child_idx, token)?;
            node_idx = child_idx;
            depth = usize::from(child.depth);
        }
        Ok(())
    }

    fn find_edge(&self, node_idx: usize, first_token: u32) -> Option<(usize, u32)> {
        let mut cur = self.nodes[node_idx].first_edge;
        while cur != NO_INDEX {
            let Ok(edge_idx) = usize::try_from(cur) else {
                return None;
            };
            let edge = self.edges[edge_idx];
            if edge.first_token == first_token {
                return Some((edge_idx, edge.child));
            }
            cur = edge.next;
        }
        None
    }

    fn attach_leaf(
        &mut self,
        parent_idx: usize,
        end_u32: u32,
        parent_depth: usize,
        depth: u8,
        cont_token: u32,
    ) -> Result<()> {
        let leaf_u32 = u32::try_from(self.nodes.len())?;
        let cont_idx = u32::try_from(self.cont_entries.len())?;
        self.cont_entries.push(ContEntry {
            token: cont_token,
            count: 1,
            next: NO_INDEX,
        });
        let parent_depth_u8 = u8::try_from(parent_depth)?;
        self.nodes.push(SuffixNode {
            edge_end: end_u32,
            parent_depth: parent_depth_u8,
            depth,
            last_end: end_u32,
            total_count: 1,
            cont_len: 1,
            first_cont: cont_idx,
            first_edge: NO_INDEX,
        });
        let first_token = self.history[usize::try_from(end_u32)? - 1 - parent_depth];
        let edge_u32 = u32::try_from(self.edges.len())?;
        self.edges.push(SuffixEdge {
            first_token,
            child: leaf_u32,
            next: self.nodes[parent_idx].first_edge,
        });
        self.nodes[parent_idx].first_edge = edge_u32;
        Ok(())
    }

    fn split_child(
        &mut self,
        incoming_edge_idx: usize,
        child_idx: usize,
        split_depth: usize,
        new_end_u32: u32,
        cont_token: u32,
    ) -> Result<u32> {
        let old_child = self.nodes[child_idx];
        let cloned_cont = self.clone_cont_list(old_child.first_cont)?;
        let split_depth_u8 = u8::try_from(split_depth)?;
        let child_split_tok = self.history[usize::try_from(old_child.edge_end)? - 1 - split_depth];
        let edge_to_old_u32 = u32::try_from(self.edges.len())?;
        self.edges.push(SuffixEdge {
            first_token: child_split_tok,
            child: u32::try_from(child_idx)?,
            next: NO_INDEX,
        });
        self.nodes[child_idx].parent_depth = split_depth_u8;

        let internal_u32 = u32::try_from(self.nodes.len())?;
        let internal_idx = self.nodes.len();
        self.nodes.push(SuffixNode {
            edge_end: old_child.edge_end,
            parent_depth: old_child.parent_depth,
            depth: split_depth_u8,
            last_end: new_end_u32,
            total_count: old_child.total_count.saturating_add(1),
            cont_len: old_child.cont_len,
            first_cont: cloned_cont,
            first_edge: edge_to_old_u32,
        });
        self.edges[incoming_edge_idx].child = internal_u32;
        self.record_node_continuation(internal_idx, cont_token)?;
        Ok(internal_u32)
    }

    fn clone_cont_list(&mut self, mut cur: u32) -> Result<u32> {
        let mut head = NO_INDEX;
        while cur != NO_INDEX {
            let entry = self.cont_entries[usize::try_from(cur)?];
            let next_cur = entry.next;
            let new_idx = u32::try_from(self.cont_entries.len())?;
            self.cont_entries.push(ContEntry {
                token: entry.token,
                count: entry.count,
                next: head,
            });
            head = new_idx;
            cur = next_cur;
        }
        Ok(head)
    }

    fn record_node_continuation(&mut self, node_idx: usize, token: u32) -> Result<()> {
        let mut cur = self.nodes[node_idx].first_cont;
        let mut empty_slot = NO_INDEX;
        while cur != NO_INDEX {
            let idx = usize::try_from(cur)?;
            if self.cont_entries[idx].token == token {
                self.cont_entries[idx].count = self.cont_entries[idx].count.saturating_add(1);
                return Ok(());
            }
            if self.cont_entries[idx].count == 0 && empty_slot == NO_INDEX {
                empty_slot = cur;
            }
            cur = self.cont_entries[idx].next;
        }
        if usize::from(self.nodes[node_idx].cont_len) < MAX_TRACKED_CONT_PER_NODE {
            let new_idx = u32::try_from(self.cont_entries.len())?;
            self.cont_entries.push(ContEntry {
                token,
                count: 1,
                next: self.nodes[node_idx].first_cont,
            });
            self.nodes[node_idx].first_cont = new_idx;
            self.nodes[node_idx].cont_len = self.nodes[node_idx].cont_len.saturating_add(1);
            return Ok(());
        }
        if empty_slot != NO_INDEX {
            let idx = usize::try_from(empty_slot)?;
            self.cont_entries[idx].token = token;
            self.cont_entries[idx].count = 1;
            return Ok(());
        }
        let mut dec = self.nodes[node_idx].first_cont;
        while dec != NO_INDEX {
            let idx = usize::try_from(dec)?;
            self.cont_entries[idx].count = self.cont_entries[idx].count.saturating_sub(1);
            dec = self.cont_entries[idx].next;
        }
        Ok(())
    }

    fn best_suffix_match(&self, max_match: usize, suffix_ext: &[u32]) -> Result<(usize, usize)> {
        let hist_len = self.history.len();
        if hist_len <= 1 || self.nodes.is_empty() || max_match == 0 {
            return Ok((0, 0));
        }
        let query_at = |rev_idx: usize| -> Option<u32> {
            if rev_idx < suffix_ext.len() {
                suffix_ext.get(suffix_ext.len() - 1 - rev_idx).copied()
            } else {
                let hist_back = rev_idx - suffix_ext.len();
                hist_len
                    .checked_sub(1 + hist_back)
                    .and_then(|pos| self.history.get(pos).copied())
            }
        };

        let mut node_idx = 0_usize;
        let mut depth = 0_usize;
        let mut best_len = 0_usize;
        let mut best_node = 0_usize;

        while depth < max_match {
            let Some(next_q) = query_at(depth) else {
                break;
            };
            let Some((_, child_u32)) = self.find_edge(node_idx, next_q) else {
                break;
            };
            let child_idx = usize::try_from(child_u32)?;
            let child = self.nodes[child_idx];
            let child_end = usize::try_from(child.edge_end)?;
            let edge_len = usize::from(child.depth - child.parent_depth);
            let compare_len = edge_len.min(max_match - depth);

            best_len = depth + 1;
            best_node = child_idx;

            let mut matched = 1_usize;
            while matched < compare_len {
                let Some(q_tok) = query_at(depth + matched) else {
                    break;
                };
                let edge_tok = self.history[child_end - 1 - (depth + matched)];
                if edge_tok != q_tok {
                    break;
                }
                matched += 1;
                best_len = depth + matched;
            }
            if matched < edge_len {
                break;
            }
            node_idx = child_idx;
            depth = usize::from(child.depth);
        }
        Ok((best_len, best_node))
    }

    fn best_empirical_continuation(&self, node_idx: usize) -> Result<(u32, u32, u32)> {
        let node = self.nodes[node_idx];
        let recent_tok = self.history[usize::try_from(node.last_end)?];
        let mut best_tok = recent_tok;
        let mut best_count = 0_u32;
        let mut recent_count = 0_u32;

        let mut cur = node.first_cont;
        while cur != NO_INDEX {
            let entry = self.cont_entries[usize::try_from(cur)?];
            if entry.token == recent_tok {
                recent_count = entry.count;
            }
            if entry.count > best_count {
                best_count = entry.count;
                best_tok = entry.token;
            }
            cur = entry.next;
        }
        if recent_count == best_count {
            best_tok = recent_tok;
        }
        Ok((best_tok, best_count.max(1), node.total_count.max(1)))
    }

    /// Propose up to `config.max_draft_tokens.min(remaining_tokens)` continuation tokens
    /// matching [`SuffixDrafter::propose`] (longest match, breaking ties by most recent span).
    ///
    /// # Errors
    /// Rejects an invalid [`SuffixDraftConfig`] or fallible allocation failure.
    pub fn propose(&self, config: SuffixDraftConfig, remaining_tokens: usize) -> Result<Vec<u32>> {
        let mut storage = [0_u32; MAX_SPECULATIVE_DRAFT_TOKENS];
        let count = self.propose_into(config, remaining_tokens, &mut storage)?;
        let mut draft = Vec::new();
        draft.try_reserve_exact(count)?;
        draft.extend_from_slice(&storage[..count]);
        Ok(draft)
    }

    /// Propose continuation tokens directly into a fixed stack buffer without heap allocation.
    ///
    /// Leaves `storage` untouched if `config` is invalid.
    ///
    /// # Errors
    /// Rejects an invalid [`SuffixDraftConfig`] or internal index overflow.
    /// Propose up to `remaining_tokens` draft continuation tokens from `self.history`.
    ///
    /// # Errors
    /// Rejects invalid suffix configuration or internal index overflow.
    pub fn propose_into(
        &self,
        config: SuffixDraftConfig,
        remaining_tokens: usize,
        storage: &mut [u32; MAX_SPECULATIVE_DRAFT_TOKENS],
    ) -> Result<usize> {
        self.propose_into_with_start(config, remaining_tokens, storage)
            .map(|(count, _)| count)
    }

    /// Complete token history currently indexed by this suffix tree.
    #[must_use]
    pub fn history(&self) -> &[u32] {
        &self.history
    }

    /// Propose draft continuation tokens and return `(count, source_start)` inside [`Self::history`].
    ///
    /// # Errors
    /// Rejects invalid suffix configuration or internal index overflow.
    pub fn propose_into_with_start(
        &self,
        config: SuffixDraftConfig,
        remaining_tokens: usize,
        storage: &mut [u32; MAX_SPECULATIVE_DRAFT_TOKENS],
    ) -> Result<(usize, Option<usize>)> {
        config.validate()?;
        let limit = config.max_draft_tokens.min(remaining_tokens);
        if limit == 0 || self.history.len() <= config.min_match_tokens {
            return Ok((0, None));
        }
        let max_len = config.max_match_tokens.min(self.history.len() - 1);
        let (matched_len, best_node) = self.best_suffix_match(max_len, &[])?;
        if matched_len < config.min_match_tokens || best_node == 0 {
            return Ok((0, None));
        }
        let start = usize::try_from(self.nodes[best_node].last_end)?;
        let available = self.history.len().saturating_sub(start).min(limit);
        if available == 0 {
            return Ok((0, None));
        }
        storage[..available].copy_from_slice(&self.history[start..start + available]);
        Ok((available, Some(start)))
    }

    /// Propose a confidence-gated continuation chain using per-node empirical continuation counts.
    ///
    /// Extends step-by-step from the most frequent continuation (tie-breaking by recency) while
    /// both the suffix match meets `config.min_match_tokens` and cumulative empirical confidence
    /// remains `>= min_confidence`.
    ///
    /// # Errors
    /// Rejects an invalid [`SuffixDraftConfig`], non-finite `min_confidence` outside `[0.0, 1.0]`,
    /// or internal index overflow.
    pub fn propose_adaptive(
        &self,
        config: SuffixDraftConfig,
        min_confidence: f64,
        remaining_tokens: usize,
        storage: &mut [u32; MAX_SPECULATIVE_DRAFT_TOKENS],
    ) -> Result<AdaptiveSuffixDraft> {
        config.validate()?;
        if !min_confidence.is_finite() || !(0.0..=1.0).contains(&min_confidence) {
            return Err(invalid(
                "adaptive suffix confidence threshold must lie in [0.0, 1.0]",
            ));
        }
        let limit = config.max_draft_tokens.min(remaining_tokens);
        if limit == 0 || self.history.len() <= config.min_match_tokens {
            return Ok(AdaptiveSuffixDraft::default());
        }
        let max_len = config.max_match_tokens.min(self.history.len() - 1);
        let mut draft_len = 0_usize;
        let mut first_matched = 0_usize;
        let mut cumulative_confidence = 1.0_f64;

        for step in 0..limit {
            let (matched_len, best_node) =
                self.best_suffix_match(max_len, &storage[..draft_len])?;
            if matched_len < config.min_match_tokens || best_node == 0 {
                break;
            }
            let (cand_token, count, total) = self.best_empirical_continuation(best_node)?;
            let step_conf = (f64::from(count) / f64::from(total)).clamp(0.0, 1.0);
            let next_conf = cumulative_confidence * step_conf;
            if next_conf < min_confidence {
                break;
            }
            if step == 0 {
                first_matched = matched_len;
            }
            storage[step] = cand_token;
            draft_len += 1;
            cumulative_confidence = next_conf;
        }
        Ok(AdaptiveSuffixDraft {
            len: draft_len,
            matched_tokens: first_matched,
            confidence: if draft_len == 0 {
                0.0
            } else {
                cumulative_confidence
            },
        })
    }
}

/// Bounded packed causal ancestor topology for up to [`MAX_DRAFT_TREE_NODES`] tree-draft tokens.
///
/// Stores topological parent links, precomputed `u16` ancestor attention masks, and deterministic
/// greedy tree-branch verification without heap-allocated DAG matrices.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DraftTreeTopology {
    len: usize,
    parents: [Option<usize>; MAX_DRAFT_TREE_NODES],
    ancestor_masks: [u16; MAX_DRAFT_TREE_NODES],
    depths: [u8; MAX_DRAFT_TREE_NODES],
}

impl DraftTreeTopology {
    /// Validate and construct a causal draft tree from `parents` in topological order.
    ///
    /// `None` denotes a direct child of the committed prefix; `Some(parent)` must satisfy
    /// `parent < node_index`.
    ///
    /// # Errors
    /// Rejects empty trees, trees exceeding [`MAX_DRAFT_TREE_NODES`], or non-topological parents.
    pub fn from_parents(parents: &[Option<usize>]) -> Result<Self> {
        if parents.is_empty() || parents.len() > MAX_DRAFT_TREE_NODES {
            return Err(invalid("draft tree node count outside contract bounds"));
        }
        let mut stored_parents = [None; MAX_DRAFT_TREE_NODES];
        let mut ancestor_masks = [0_u16; MAX_DRAFT_TREE_NODES];
        let mut depths = [0_u8; MAX_DRAFT_TREE_NODES];
        for (idx, &parent) in parents.iter().enumerate() {
            let bit = 1_u16 << u32::try_from(idx)?;
            let (mask, depth) = match parent {
                None => (bit, 1_u8),
                Some(parent_idx) => {
                    if parent_idx >= idx {
                        return Err(invalid(
                            "draft tree parent index must precede child in topological order",
                        ));
                    }
                    (
                        ancestor_masks[parent_idx] | bit,
                        depths[parent_idx].saturating_add(1),
                    )
                }
            };
            stored_parents[idx] = parent;
            ancestor_masks[idx] = mask;
            depths[idx] = depth;
        }
        Ok(Self {
            len: parents.len(),
            parents: stored_parents,
            ancestor_masks,
            depths,
        })
    }

    /// Construct a linear chain topology of `len` nodes (`1..=MAX_DRAFT_TREE_NODES`).
    ///
    /// # Errors
    /// Rejects `len == 0` or `len > MAX_DRAFT_TREE_NODES`.
    pub fn chain(len: usize) -> Result<Self> {
        if len == 0 || len > MAX_DRAFT_TREE_NODES {
            return Err(invalid("draft chain node count outside contract bounds"));
        }
        let mut parents = [None; MAX_DRAFT_TREE_NODES];
        for (idx, slot) in parents[..len].iter_mut().enumerate().skip(1) {
            *slot = Some(idx - 1);
        }
        Self::from_parents(&parents[..len])
    }

    /// Number of draft nodes in this tree (`1..=MAX_DRAFT_TREE_NODES`).
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Always `false` since valid draft trees contain at least one node.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return the packed `u16` ancestor bitmask for `node` (inclusive of `node`), or `None`.
    #[must_use]
    pub const fn ancestor_mask(&self, node: usize) -> Option<u16> {
        if node < self.len {
            Some(self.ancestor_masks[node])
        } else {
            None
        }
    }

    /// Whether `query_node` causally attends to `key_node` in this draft tree.
    #[must_use]
    pub fn attends(&self, query_node: usize, key_node: usize) -> bool {
        if query_node >= self.len || key_node >= self.len {
            return false;
        }
        let bit = 1_u16.checked_shl(u32::try_from(key_node).unwrap_or(u32::MAX));
        bit.is_some_and(|mask_bit| (self.ancestor_masks[query_node] & mask_bit) != 0)
    }

    /// Return the 1-based depth from the committed prefix to `node`, or `None` if out of bounds.
    #[must_use]
    pub fn depth(&self, node: usize) -> Option<usize> {
        if node < self.len {
            Some(usize::from(self.depths[node]))
        } else {
            None
        }
    }

    /// Verify candidate tree nodes against greedy target predictions (`root_greedy` at the
    /// committed prefix and `node_greedy_next[i]` emitted after evaluating node `i`), returning
    /// the longest accepted draft token sequence together with its trailing bonus or correction token.
    ///
    /// Equal-depth accepted paths tie-break deterministically toward the lowest terminal node index.
    ///
    /// # Errors
    /// Rejects slice lengths that do not equal [`Self::len`].
    pub fn verify_greedy_tree(
        &self,
        draft_tokens: &[u32],
        root_greedy: u32,
        node_greedy_next: &[u32],
    ) -> Result<(Vec<u32>, u32)> {
        if draft_tokens.len() != self.len || node_greedy_next.len() != self.len {
            return Err(invalid(
                "draft tree verification slice length mismatch with topology",
            ));
        }
        let mut accepted = [false; MAX_DRAFT_TREE_NODES];
        let mut best_terminal: Option<usize> = None;
        let mut best_depth = 0_u8;
        for idx in 0..self.len {
            let is_accepted = self.parents[idx].map_or_else(
                || draft_tokens[idx] == root_greedy,
                |parent| accepted[parent] && draft_tokens[idx] == node_greedy_next[parent],
            );
            accepted[idx] = is_accepted;
            if is_accepted && self.depths[idx] > best_depth {
                best_depth = self.depths[idx];
                best_terminal = Some(idx);
            }
        }
        let Some(mut cur) = best_terminal else {
            return Ok((Vec::new(), root_greedy));
        };
        let bonus = node_greedy_next[cur];
        let mut path = Vec::new();
        path.try_reserve_exact(usize::from(best_depth))?;
        loop {
            path.push(draft_tokens[cur]);
            match self.parents[cur] {
                Some(parent) => cur = parent,
                None => break,
            }
        }
        path.reverse();
        Ok((path, bonus))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DFLASH2_DEFAULT_TAP_LAYERS, DFlash2DraftConfig, DraftTreeTopology, GdnVerifyTape,
        MAX_DRAFT_TREE_NODES, MAX_SPECULATIVE_DRAFT_TOKENS, MAX_SUFFIX_MATCH_TOKENS,
        SpeculativeStats, SuffixDraftConfig, SuffixDrafter, SuffixTree, VerifiedRound,
        cumulative_rejection_complement, mtp_first_decode_fits, propose_lookup_then_fallback,
        suffix_match_start, verify_stochastic_step,
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

    // Retain the original independent length-first search as the behavioral
    // oracle. It compares forward spans and ranks starts, whereas production
    // compares backwards by match end.
    fn reference_suffix_continuation(
        history: &[u32],
        config: SuffixDraftConfig,
        remaining: usize,
    ) -> &[u32] {
        let limit = config.max_draft_tokens.min(remaining);
        if limit == 0 || history.len() <= config.min_match_tokens {
            return &[];
        }
        for length in
            (config.min_match_tokens..=config.max_match_tokens.min(history.len() - 1)).rev()
        {
            let suffix = history.len() - length;
            for start in (0..suffix).rev() {
                if history[start..start + length] == history[suffix..] {
                    let end = start + length;
                    return &history[end..end + limit.min(history.len() - end)];
                }
            }
        }
        &[]
    }

    fn check_suffix_apis(
        history: &[u32],
        config: SuffixDraftConfig,
        remaining: usize,
    ) -> Result<()> {
        let expected = reference_suffix_continuation(history, config, remaining);
        assert_eq!(
            SuffixDrafter::propose(history, config, remaining)?,
            expected
        );
        let mut storage = [u32::MAX; MAX_SPECULATIVE_DRAFT_TOKENS];
        let length = SuffixDrafter::propose_into(history, config, remaining, &mut storage)?;
        assert_eq!(&storage[..length], expected);
        assert!(storage[length..].iter().all(|&token| token == u32::MAX));
        Ok(())
    }

    #[test]
    fn suffix_drafter_exhaustively_matches_length_first_overlap_and_tie_order() -> Result<()> {
        // Every binary history through ten tokens includes overlapping runs,
        // competing match lengths, recent ties and histories shorter than a window.
        for length in 0..=10 {
            for bits in 0_u32..1 << length {
                let history = (0..length)
                    .map(|index| (bits >> index) & 1)
                    .collect::<Vec<_>>();
                for min_match in 1..=5 {
                    for max_match in min_match..=5 {
                        for draft in [1, 4, MAX_SPECULATIVE_DRAFT_TOKENS] {
                            let config = SuffixDraftConfig::new(min_match, max_match, draft)?;
                            for remaining in [0, 1, 3, MAX_SPECULATIVE_DRAFT_TOKENS] {
                                check_suffix_apis(&history, config, remaining)?;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn suffix_drafter_maximum_windows_match_reference_and_reject_before_writing() -> Result<()> {
        let distinct = (0_u32..8192).collect::<Vec<_>>();
        let periodic = distinct.iter().map(|token| token % 97).collect::<Vec<_>>();
        let mut near_match = periodic.clone();
        near_match[8191 - MAX_SUFFIX_MATCH_TOKENS / 2] = u32::MAX;
        let mut constant = distinct.clone();
        constant.fill(7);
        for history in [&distinct, &periodic, &near_match, &constant] {
            for min_match in [1, MAX_SUFFIX_MATCH_TOKENS] {
                let config = SuffixDraftConfig::new(
                    min_match,
                    MAX_SUFFIX_MATCH_TOKENS,
                    MAX_SPECULATIVE_DRAFT_TOKENS,
                )?;
                for remaining in [0, 1, MAX_SPECULATIVE_DRAFT_TOKENS, usize::MAX] {
                    check_suffix_apis(history, config, remaining)?;
                }
            }
        }
        let valid = SuffixDraftConfig::default();
        for config in [
            SuffixDraftConfig {
                min_match_tokens: 0,
                ..valid
            },
            SuffixDraftConfig {
                min_match_tokens: 17,
                ..valid
            },
            SuffixDraftConfig {
                max_match_tokens: MAX_SUFFIX_MATCH_TOKENS + 1,
                ..valid
            },
            SuffixDraftConfig {
                max_draft_tokens: 0,
                ..valid
            },
            SuffixDraftConfig {
                max_draft_tokens: MAX_SPECULATIVE_DRAFT_TOKENS + 1,
                ..valid
            },
        ] {
            let mut storage = [42; MAX_SPECULATIVE_DRAFT_TOKENS];
            assert!(SuffixDrafter::propose_into(&periodic, config, 0, &mut storage).is_err());
            assert_eq!(storage, [42; MAX_SPECULATIVE_DRAFT_TOKENS]);
            assert!(SuffixDrafter::propose(&periodic, config, 0).is_err());
        }
        Ok(())
    }

    #[test]
    fn suffix_z_matcher_bounds_work_on_adversarial_near_matches() -> Result<()> {
        // A single differing token near the tail prevents a full-window match.
        // Most historical ends nevertheless share 63 trailing zeros, forcing
        // the earlier independent backwards matcher to repeat those comparisons.
        let mut history = vec![0; 8192];
        let length = history.len();
        history[length - MAX_SUFFIX_MATCH_TOKENS] = u32::MAX;
        let config = SuffixDraftConfig::new(2, MAX_SUFFIX_MATCH_TOKENS, 8)?;
        let expected = reference_suffix_continuation(&history, config, 8);
        let mut comparisons = 0;
        let start = suffix_match_start(
            &history,
            config.min_match_tokens,
            config.max_match_tokens,
            |left, right| {
                comparisons += 1;
                left == right
            },
        )?
        .ok_or_else(|| crate::invalid("adversarial history must contain a suffix match"))?;
        assert_eq!(&history[start..start + 8.min(length - start)], expected);
        assert!(comparisons < 2 * (length + MAX_SUFFIX_MATCH_TOKENS));

        // Count the previous one-pass search independently. Its token predicate
        // is evaluated once for every comparison, including failed matches.
        let mut old_comparisons = 0;
        let mut best = config.min_match_tokens - 1;
        let mut old_start = None;
        for end in (config.min_match_tokens..length).rev() {
            if end <= best {
                break;
            }
            let mut matched = 0;
            while matched < MAX_SUFFIX_MATCH_TOKENS.min(end) {
                old_comparisons += 1;
                if history[end - matched - 1] != history[length - matched - 1] {
                    break;
                }
                matched += 1;
            }
            if matched > best {
                best = matched;
                old_start = Some(end);
                if matched == MAX_SUFFIX_MATCH_TOKENS {
                    break;
                }
            }
        }
        assert_eq!(Some(start), old_start);
        assert!(old_comparisons > comparisons * 20);
        check_suffix_apis(&history, config, usize::MAX)
    }

    #[test]
    fn suffix_z_matcher_preserves_periodic_ties_overlap_and_linear_bound() -> Result<()> {
        for length in [2, 15, 63, 64, 65, 129, 8192] {
            for period in [1, 2, 3, 7, 31, 63, 64, 65, 97] {
                let mut history = (0..length)
                    .map(|index| u32::try_from(index % period).map_err(Into::into))
                    .collect::<Result<Vec<_>>>()?;
                // Exercise both exact periodic repeats and broken runs. The
                // asymmetric marker can change the winning match and its tie.
                for broken in [false, true] {
                    if broken {
                        history[length / 2] = u32::MAX;
                    }
                    for minimum in [1, 2, 16, 64] {
                        for maximum in [minimum, MAX_SUFFIX_MATCH_TOKENS] {
                            let config = SuffixDraftConfig::new(minimum, maximum, 8)?;
                            check_suffix_apis(&history, config, 8)?;
                            if length <= minimum {
                                continue;
                            }
                            let maximum = maximum.min(length - 1);
                            let mut comparisons = 0;
                            let start =
                                suffix_match_start(&history, minimum, maximum, |left, right| {
                                    comparisons += 1;
                                    left == right
                                })?;
                            let actual = start.map_or(&[][..], |start| {
                                &history[start..start + 8.min(length - start)]
                            });
                            assert_eq!(actual, reference_suffix_continuation(&history, config, 8));
                            assert!(comparisons < 2 * (length + maximum));
                        }
                    }
                }
            }
        }
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
            ..SpeculativeStats::default()
        };
        assert!((stats.acceptance_rate() - 0.75).abs() <= 1e-12);
        assert!((stats.mean_accepted_length() - 1.5).abs() <= 1e-12);
        assert_eq!(SpeculativeStats::default().acceptance_rate(), 0.0);
        Ok(())
    }

    #[test]
    fn speculative_stats_tracks_positions_histogram_and_outcomes() {
        let mut stats = SpeculativeStats::default();
        stats.record_round(3, 2, 4, true);
        stats.record_verified_round(VerifiedRound {
            proposed: 4,
            accepted: 4,
            verified: 4,
            target_evaluations: 5,
            rolled_back: false,
            bonus_tokens: 1,
            correction_tokens: 0,
            fallback_tokens: 0,
            replay_tokens: 0,
        });
        stats.record_verified_round(VerifiedRound {
            proposed: 0,
            accepted: 0,
            verified: 0,
            target_evaluations: 1,
            rolled_back: false,
            bonus_tokens: 0,
            correction_tokens: 0,
            fallback_tokens: 1,
            replay_tokens: 0,
        });
        stats.record_verified_round(VerifiedRound {
            proposed: 3,
            accepted: 1,
            verified: 3,
            target_evaluations: 5,
            rolled_back: true,
            bonus_tokens: 0,
            correction_tokens: 1,
            fallback_tokens: 0,
            replay_tokens: 2,
        });
        assert_eq!(stats.draft_rounds, 3);
        assert_eq!(stats.draft_tokens_proposed, 10);
        assert_eq!(stats.draft_tokens_accepted, 7);
        assert_eq!(stats.draft_tokens_verified, 10);
        assert_eq!(stats.target_evaluations, 15);
        assert_eq!(stats.rollback_restores, 2);
        assert_eq!(stats.bonus_tokens, 1);
        assert_eq!(stats.correction_tokens, 1);
        assert_eq!(stats.fallback_tokens, 1);
        assert_eq!(stats.replay_steps, 1);
        assert_eq!(stats.replay_tokens, 2);
        assert_eq!(stats.position_proposed, [3, 3, 3, 1, 0, 0, 0, 0]);
        assert_eq!(stats.position_accepted, [3, 2, 1, 1, 0, 0, 0, 0]);
        assert_eq!(stats.accepted_len_histogram, [0, 1, 1, 0, 1, 0, 0, 0, 0]);
        assert!((stats.mean_accepted_length() - (7.0 / 3.0)).abs() <= 1e-12);
        assert_eq!(
            stats
                .position_acceptance_rate(0)
                .map(|rate| (rate - 1.0).abs() <= 1e-12),
            Some(true)
        );
        assert_eq!(
            stats
                .position_acceptance_rate(1)
                .map(|rate| (rate - (2.0 / 3.0)).abs() <= 1e-12),
            Some(true)
        );
        assert_eq!(stats.position_acceptance_rate(4), None);
        assert_eq!(stats.position_acceptance_rate(8), None);
    }

    #[test]
    fn cumulative_rejection_complement_preserves_small_tail_precision() -> Result<()> {
        assert_eq!(
            cumulative_rejection_complement(&[])?.to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(
            cumulative_rejection_complement(&[1.0])?.to_bits(),
            1.0_f64.to_bits()
        );
        let two = cumulative_rejection_complement(&[0.5, 0.5])?;
        assert!((two - 0.75).abs() <= 1e-12);

        let tiny = 1e-12_f64;
        let tiny_res = cumulative_rejection_complement(&[tiny, tiny])?;
        assert!(2.0_f64.mul_add(-tiny, tiny_res).abs() <= 1e-22);

        for invalid_prob in [-0.01, 1.01, f64::NAN, f64::INFINITY] {
            assert!(cumulative_rejection_complement(&[invalid_prob]).is_err());
        }
        Ok(())
    }

    #[test]
    fn propose_lookup_then_fallback_runs_lookup_before_secondary_drafter() -> Result<()> {
        let config = SuffixDraftConfig::new(2, 8, 3)?;
        let history_with_match = vec![10, 20, 30, 40, 10, 20];
        let mut fallback_called = false;
        let lookup = propose_lookup_then_fallback(&history_with_match, config, 3, |_, _| {
            fallback_called = true;
            Ok(vec![99, 98, 97])
        })?;
        assert!(!fallback_called);
        assert_eq!(lookup, vec![30, 40, 10]);

        let history_without_match = vec![1, 2, 3, 4];
        let secondary =
            propose_lookup_then_fallback(&history_without_match, config, 2, |_, limit| {
                fallback_called = true;
                assert_eq!(limit, 2);
                Ok(vec![77, 88, 99])
            })?;
        assert!(fallback_called);
        assert_eq!(secondary, vec![77, 88]);
        Ok(())
    }

    #[test]
    fn gdn_verify_tape_atomic_rollback_on_late_head_failure() -> Result<()> {
        let key_heads = 1;
        let value_heads = 2;
        let key_dim = 4;
        let value_dim = 4;
        let head_stride = value_dim * key_dim;
        let mut tape = GdnVerifyTape::new(key_heads, value_heads, key_dim, value_dim, 2)?;
        tape.record_step(&[0.1; 4], &[0.2; 4], &[0.3; 8], &[0.9, 0.9], &[0.5, 0.5])?;
        let mut state = vec![0.25_f32; value_heads * head_stride];
        // Corrupt head 1 so head 0 computes finitely while head 1 fails finiteness.
        state[head_stride] = f32::NAN;
        let original = state.clone();
        assert!(tape.replay_accepted_prefix(&mut state, 1).is_err());
        for (actual, expected) in state.iter().zip(&original) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
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
        let value_dim = 14;
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

    #[test]
    fn suffix_tree_exhaustively_matches_reference_and_incremental_sync() -> Result<()> {
        let mut tree = SuffixTree::new();
        for length in 0..=10 {
            for bits in 0_u32..1 << length {
                let history = (0..length)
                    .map(|index| (bits >> index) & 1)
                    .collect::<Vec<_>>();
                tree.sync(&history)?;
                assert_eq!(tree.len(), history.len());
                for min_match in 1..=5 {
                    for max_match in min_match..=5 {
                        for draft in [1, 4, MAX_SPECULATIVE_DRAFT_TOKENS] {
                            let config = SuffixDraftConfig::new(min_match, max_match, draft)?;
                            for remaining in [0, 1, 3, MAX_SPECULATIVE_DRAFT_TOKENS] {
                                let expected =
                                    reference_suffix_continuation(&history, config, remaining);
                                assert_eq!(tree.propose(config, remaining)?, expected);
                                let mut buf = [u32::MAX; MAX_SPECULATIVE_DRAFT_TOKENS];
                                let count = tree.propose_into(config, remaining, &mut buf)?;
                                assert_eq!(&buf[..count], expected);
                                assert!(buf[count..].iter().all(|&tok| tok == u32::MAX));
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn suffix_tree_periodic_near_match_and_incremental_extension_match_reference() -> Result<()> {
        let distinct = (0_u32..2048).collect::<Vec<_>>();
        let periodic = distinct.iter().map(|token| token % 97).collect::<Vec<_>>();
        let mut near_match = periodic.clone();
        near_match[2047 - MAX_SUFFIX_MATCH_TOKENS / 2] = u32::MAX;
        let mut constant = distinct.clone();
        constant.fill(7);

        let mut tree = SuffixTree::with_capacity(2048)?;
        for history in [&distinct, &periodic, &near_match, &constant] {
            // Feed in chunks to exercise incremental sync extension and rebuilds.
            tree.clear();
            for chunk in history.chunks(137) {
                let target_len = tree.len() + chunk.len();
                tree.sync(&history[..target_len])?;
            }
            for min_match in [1, 2, MAX_SUFFIX_MATCH_TOKENS] {
                let config = SuffixDraftConfig::new(
                    min_match,
                    MAX_SUFFIX_MATCH_TOKENS,
                    MAX_SPECULATIVE_DRAFT_TOKENS,
                )?;
                let expected =
                    reference_suffix_continuation(history, config, MAX_SPECULATIVE_DRAFT_TOKENS);
                assert_eq!(
                    tree.propose(config, MAX_SPECULATIVE_DRAFT_TOKENS)?,
                    expected
                );
            }
        }
        Ok(())
    }

    #[test]
    fn suffix_tree_adaptive_draft_gates_low_confidence_ambiguity() -> Result<()> {
        // After [10, 20], 3 occurrences continue with [30, 40] and 1 most-recent occurrence
        // continues with [99, 88]. Most-recent tie favors [99, 88], whereas adaptive empirical
        // confidence selects the dominant [30] with confidence 0.75 at step 1, and gates below
        // 0.80 threshold.
        let mut tree = SuffixTree::new();
        let history = [
            10, 20, 30, 40, // opp 1
            10, 20, 30, 40, // opp 2
            10, 20, 30, 40, // opp 3
            10, 20, 99, 88, // opp 4 (most recent)
            10, 20,
        ];
        tree.sync(&history)?;
        let config = SuffixDraftConfig::new(2, 8, 4)?;
        let mut out = [0_u32; MAX_SPECULATIVE_DRAFT_TOKENS];

        let high = tree.propose_adaptive(config, 0.80, 4, &mut out)?;
        assert_eq!(high.len, 0);

        let medium = tree.propose_adaptive(config, 0.70, 4, &mut out)?;
        assert!(medium.len >= 2);
        assert_eq!(&out[..2], &[30, 40]);
        assert!(medium.confidence >= 0.70 && medium.confidence <= 0.75 + 1e-12);
        assert_eq!(medium.matched_tokens, 2);

        assert!(tree.propose_adaptive(config, -0.1, 4, &mut out).is_err());
        assert!(
            tree.propose_adaptive(config, f64::NAN, 4, &mut out)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn draft_tree_topology_computes_ancestor_masks_and_verifies_deepest_path() -> Result<()> {
        // Tree:
        // root -> 0 (tok 10, next=20)
        //   0 -> 1 (tok 20, next=30)
        //     1 -> 3 (tok 30, next=40)
        //   0 -> 2 (tok 21, next=99)
        // root -> 4 (tok 11, next=50)
        let tree = DraftTreeTopology::from_parents(&[None, Some(0), Some(0), Some(1), None])?;
        assert_eq!(tree.len(), 5);
        assert!(!tree.is_empty());
        assert_eq!(tree.ancestor_mask(0), Some(0b0_0001));
        assert_eq!(tree.ancestor_mask(1), Some(0b0_0011));
        assert_eq!(tree.ancestor_mask(2), Some(0b0_0101));
        assert_eq!(tree.ancestor_mask(3), Some(0b0_1011));
        assert_eq!(tree.ancestor_mask(4), Some(0b1_0000));
        assert!(tree.attends(3, 0));
        assert!(tree.attends(3, 1));
        assert!(!tree.attends(3, 2));
        assert!(!tree.attends(3, 4));
        assert_eq!(tree.depth(3), Some(3));

        let (accepted, bonus) =
            tree.verify_greedy_tree(&[10, 20, 21, 30, 11], 10, &[20, 30, 99, 40, 50])?;
        assert_eq!(accepted, [10, 20, 30]);
        assert_eq!(bonus, 40);

        let (none_accepted, correction) =
            tree.verify_greedy_tree(&[10, 20, 21, 30, 11], 77, &[20, 30, 99, 40, 50])?;
        assert_eq!(none_accepted, [0_u32; 0]);
        assert_eq!(correction, 77);

        let chain = DraftTreeTopology::chain(MAX_DRAFT_TREE_NODES)?;
        assert_eq!(
            chain.depth(MAX_DRAFT_TREE_NODES - 1),
            Some(MAX_DRAFT_TREE_NODES)
        );
        assert!(DraftTreeTopology::from_parents(&[Some(0)]).is_err());
        assert!(DraftTreeTopology::from_parents(&[]).is_err());
        Ok(())
    }

    #[test]
    fn moving_source_cursor_bridges_local_edit_and_deactivates_on_divergence() -> Result<()> {
        use super::{CursorDraft, MovingSourceCursor};
        let mut history = vec![
            100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 100, 101, 102,
        ];
        let mut tree = SuffixTree::new();
        tree.sync(&history)?;
        let config = SuffixDraftConfig::new(2, 8, 4)?;
        let mut cursor = MovingSourceCursor::new();
        let mut buf = [0_u32; MAX_SPECULATIVE_DRAFT_TOKENS];

        let round1 = cursor.propose_with_tree(&tree, config, 16, &mut buf)?;
        assert_eq!(
            round1,
            CursorDraft {
                len: 4,
                suffix_tokens: 4,
                cursor_tokens: 0,
            }
        );
        assert_eq!(&buf[..4], &[103, 104, 105, 106]);

        // Accept [103, 104] and substitute 999 for 105 (breaking exact suffix matching).
        history.extend_from_slice(&[103, 104, 999]);
        tree.sync(&history)?;
        cursor.observe_round(tree.history(), round1.len, 2, false);

        let round2 = cursor.propose_with_tree(&tree, config, 16, &mut buf)?;
        assert_eq!(round2.suffix_tokens, 0);
        assert_eq!(round2.cursor_tokens, 4);
        assert_eq!(&buf[..4], &[106, 107, 108, 109]);

        // Sustained divergence deactivates the cursor cleanly within max_stall_misses.
        history.push(888);
        tree.sync(&history)?;
        cursor.observe_round(tree.history(), round2.len, 0, false);
        let round3 = cursor.propose_with_tree(&tree, config, 16, &mut buf)?;
        assert!(round3.cursor_tokens > 0);

        history.push(889);
        tree.sync(&history)?;
        cursor.observe_round(tree.history(), round3.len, 0, false);
        assert_eq!(cursor.anchor(), None);

        let round4 = cursor.propose_with_tree(&tree, config, 16, &mut buf)?;
        assert_eq!(round4, CursorDraft::default());
        Ok(())
    }

    #[test]
    fn adaptive_speculation_gate_throttles_prose_and_recovers_on_code() -> Result<()> {
        use super::{AdaptiveGateConfig, AdaptiveGateDecision, AdaptiveSpeculationGate};
        let mut gate = AdaptiveSpeculationGate::new(AdaptiveGateConfig {
            enabled: true,
            warmup_rounds: 3,
            smoothing: 0.5,
            initial_verify_to_serial_ratio: 2.5,
            break_even_margin: 1.0,
            min_burst_tokens: 4,
            max_burst_tokens: 16,
        })?;
        // Calibrate serial ~1 ms, verify ~3 ms -> break-even = 3.0 emitted tokens.
        for _ in 0..3 {
            gate.record_serial_step(std::time::Duration::from_millis(1));
        }
        assert_eq!(gate.decide(), AdaptiveGateDecision::Speculate);
        for _ in 0..3 {
            gate.record_verify_round(
                4,
                0,
                1,
                std::time::Duration::from_millis(3),
                std::time::Duration::ZERO,
            );
        }
        assert!((gate.verify_to_serial_ratio() - 3.0).abs() <= 1e-9);
        for _ in 0..4 {
            assert_eq!(gate.decide(), AdaptiveGateDecision::SerialBurst);
        }
        // Probe after burst clears immediately when a high-acceptance block resumes.
        assert_eq!(gate.decide(), AdaptiveGateDecision::Speculate);
        gate.record_verify_round(
            4,
            4,
            5,
            std::time::Duration::from_millis(3),
            std::time::Duration::ZERO,
        );
        assert_eq!(gate.decide(), AdaptiveGateDecision::Speculate);
        Ok(())
    }

    #[test]
    fn speculative_stats_tracks_folded_cursor_and_gate_counters() {
        let mut stats = SpeculativeStats::default();
        stats.record_folded_carry(3);
        stats.record_folded_flush();
        stats.record_cursor_draft(4);
        stats.record_adaptive_serial_burst(true);
        stats.record_adaptive_serial_burst(true);
        assert_eq!(stats.folded_replay_steps, 1);
        assert_eq!(stats.folded_replay_tokens, 3);
        assert_eq!(stats.folded_flushes, 1);
        assert_eq!(stats.cursor_draft_tokens, 4);
        assert_eq!(stats.adaptive_gate_skips, 2);
        assert_eq!(stats.serial_burst_tokens, 2);
    }
}
