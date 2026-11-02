//! Optional fixed-size request attribution, without token IDs or output text.

use std::time::{Duration, Instant};

use crate::Result;

/// One nonoverlapping operation bracket inside synchronous generation.
///
/// These are runtime operation timings, including checks and allocation in the
/// bracket, rather than isolated kernel timings. Callback time includes the
/// supplied consumer and any synchronous backpressure it imposes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GenerationPhase {
    /// Independently clone the caller's sampler before the transaction.
    SamplerSnapshot,
    /// Capture complete state/tokens/logits for whole-request rollback.
    RollbackSnapshot,
    /// Reuse, restore or reset the committed prefix before new model work.
    PrefixPreparation,
    /// State-only prompt blocks, excluding the final prompt evaluation.
    PrefillIngestion,
    /// Final prompt-token evaluation producing the first sampling logits.
    PrefillEvaluation,
    /// Evaluate a sampled non-stop token and validate its next-token logits.
    DecodeEvaluation,
    /// Select a token, including possible stop tokens and rejected selections.
    Sampling,
    /// Invoke the token consumer; successful calls remain provisional.
    Callback,
    /// Optional retained-prefix capture after successful generation.
    CompletionSnapshot,
    /// Restore model state/tokens/logits following a failed transaction.
    RollbackRestore,
    /// Restore the caller's sampler after any returned generation error.
    SamplerRestore,
}

impl GenerationPhase {
    /// Complete fixed phase inventory in stable diagnostic order.
    pub const ALL: [Self; 11] = [
        Self::SamplerSnapshot,
        Self::RollbackSnapshot,
        Self::PrefixPreparation,
        Self::PrefillIngestion,
        Self::PrefillEvaluation,
        Self::DecodeEvaluation,
        Self::Sampling,
        Self::Callback,
        Self::CompletionSnapshot,
        Self::RollbackRestore,
        Self::SamplerRestore,
    ];

    /// Fixed diagnostic label containing no caller-provided values.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::SamplerSnapshot => "sampler-snapshot",
            Self::RollbackSnapshot => "rollback-snapshot",
            Self::PrefixPreparation => "prefix-preparation",
            Self::PrefillIngestion => "prefill-ingestion",
            Self::PrefillEvaluation => "prefill-evaluation",
            Self::DecodeEvaluation => "decode-evaluation",
            Self::Sampling => "sampling",
            Self::Callback => "callback",
            Self::CompletionSnapshot => "completion-snapshot",
            Self::RollbackRestore => "rollback-restore",
            Self::SamplerRestore => "sampler-restore",
        }
    }

    const fn index(self) -> usize {
        match self {
            Self::SamplerSnapshot => 0,
            Self::RollbackSnapshot => 1,
            Self::PrefixPreparation => 2,
            Self::PrefillIngestion => 3,
            Self::PrefillEvaluation => 4,
            Self::DecodeEvaluation => 5,
            Self::Sampling => 6,
            Self::Callback => 7,
            Self::CompletionSnapshot => 8,
            Self::RollbackRestore => 9,
            Self::SamplerRestore => 10,
        }
    }
}

/// Aggregate observations for one fixed operation kind.
#[derive(Clone, Copy, Debug, Default)]
pub struct PhaseTiming {
    /// Total wall time inside returned operation calls, including failed calls.
    pub elapsed: Duration,
    /// Calls entered, including calls rejected by checks inside their bracket.
    pub attempted_calls: usize,
    /// Calls that returned success before any subsequent request check.
    pub completed_calls: usize,
    /// Token units submitted to the bracket; zero for non-token operations.
    pub attempted_tokens: usize,
    /// Token units in successful calls, a lower bound when another call failed.
    /// Sampling units are selections and callback units are provisional calls;
    /// neither means graph evaluation or committed output.
    pub completed_tokens_lower_bound: usize,
    /// A failed operation may have performed partial work before returning.
    /// In particular, a failed prefill block does not establish zero tokens.
    pub partial_progress_unknown: bool,
}

/// Optional fixed-size attribution for one generation attempt.
///
/// A timed entry point replaces this record before each attempt. The total
/// includes sampler/model transaction work, validation and instrumentation;
/// phase sums exclude other runtime overhead. Loading, request rendering and
/// outer service delivery are excluded. Timings survive returned errors, but
/// cannot attest to work interrupted by process termination or a panic.
#[derive(Clone, Debug, Default)]
pub struct GenerationTimings {
    /// Whether a timed runtime entry point was actually entered.
    pub entered: bool,
    /// Whole timed runtime call, including rollback on returned errors.
    pub total: Duration,
    /// Whether generation returned a completed transaction.
    pub succeeded: bool,
    /// Supplied prompt length, without token identities.
    pub prompt_tokens: usize,
    /// Successfully prepared reused prefix length, absent if preparation failed.
    pub reused_tokens: Option<usize>,
    /// Committed generated count only on success; absent on every error.
    pub committed_generated_tokens: Option<usize>,
    phases: [PhaseTiming; 11],
}

impl GenerationTimings {
    pub(super) fn started(prompt_tokens: usize) -> Self {
        Self {
            entered: true,
            prompt_tokens,
            ..Self::default()
        }
    }

    /// Observations for a fixed phase; missing work has zero attempted calls.
    #[must_use]
    pub const fn phase(&self, phase: GenerationPhase) -> &PhaseTiming {
        &self.phases[phase.index()]
    }
}

pub(super) fn measure<T>(
    timings: &mut Option<&mut GenerationTimings>,
    phase: GenerationPhase,
    tokens: usize,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let started = timings.as_ref().map(|_| Instant::now());
    let result = operation();
    if let (Some(timings), Some(started)) = (timings.as_deref_mut(), started) {
        let timing = &mut timings.phases[phase.index()];
        timing.elapsed += started.elapsed();
        timing.attempted_calls += 1;
        timing.attempted_tokens += tokens;
        if result.is_ok() {
            timing.completed_calls += 1;
            timing.completed_tokens_lower_bound += tokens;
        } else {
            timing.partial_progress_unknown = true;
        }
    }
    result
}
