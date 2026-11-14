//! Opt-in, value-free single-token layer attribution.

use std::time::{Duration, Instant};

use super::Qwen35;
use crate::{Result, invalid};

/// Maximum layer count accepted by the fixed-storage timing entry point.
pub const MAX_TIMED_LAYERS: usize = 64;

/// One synchronous operation bracket in a provisional token evaluation.
#[derive(Clone, Copy, Debug, Default)]
pub struct OperationTiming {
    /// Wall time including validation, allocation and synchronous executor work.
    pub elapsed: Duration,
    /// Whether this operation was entered, including operations returning errors.
    pub attempted: bool,
    /// Whether the operation returned success; this does not mean token commit.
    pub completed: bool,
}

/// Disjoint attention and FFN brackets at one decoder layer.
#[derive(Clone, Copy, Debug, Default)]
pub struct LayerTimings {
    /// Input normalization and full or recurrent attention, excluding residual add.
    pub attention: OperationTiming,
    /// Attention residual add, post-attention normalization, FFN and FFN residual add.
    /// Includes surviving standard adapters and the actual selected fusion path.
    pub ffn: OperationTiming,
}

/// Fixed-storage attribution replaced on each explicit timed evaluation attempt.
///
/// No token IDs, activations, logits or output text are retained. Layer brackets
/// exclude embedding, output projection and state commit; their sum need not
/// equal total time. Successful layer work stays provisional if a later layer,
/// output projection or state append fails. Untimed entry points read no clocks.
#[derive(Clone, Debug)]
pub struct TokenTimings {
    /// Whether the bounded graph's ordinary token evaluator was entered.
    pub entered: bool,
    /// Ordinary evaluator wall time including its checks and layer brackets.
    /// Excludes record reset, layer-count admission and failed-output clearing.
    pub elapsed: Duration,
    /// Whether this token completed and committed its graph state.
    pub succeeded: bool,
    layer_count: usize,
    layers: [LayerTimings; MAX_TIMED_LAYERS],
}

impl Default for TokenTimings {
    fn default() -> Self {
        Self {
            entered: false,
            elapsed: Duration::ZERO,
            succeeded: false,
            layer_count: 0,
            layers: [LayerTimings::default(); MAX_TIMED_LAYERS],
        }
    }
}

impl TokenTimings {
    /// Per-layer observations in source execution order, including unvisited layers.
    #[must_use]
    pub fn layers(&self) -> &[LayerTimings] {
        &self.layers[..self.layer_count]
    }

    pub(super) fn attention(&mut self, layer: usize) -> Span<'_> {
        Span::new(&mut self.layers[layer].attention)
    }

    pub(super) fn ffn(&mut self, layer: usize) -> Span<'_> {
        Span::new(&mut self.layers[layer].ffn)
    }
}

pub(super) struct Span<'a> {
    timing: &'a mut OperationTiming,
    started: Instant,
}

impl<'a> Span<'a> {
    fn new(timing: &'a mut OperationTiming) -> Self {
        timing.attempted = true;
        Self {
            timing,
            started: Instant::now(),
        }
    }

    pub(super) fn complete(self) {
        self.timing.completed = true;
    }
}

impl Drop for Span<'_> {
    fn drop(&mut self) {
        self.timing.elapsed = self.started.elapsed();
    }
}

pub(super) fn complete(span: Option<Span<'_>>) {
    if let Some(span) = span {
        span.complete();
    }
}

impl Qwen35 {
    /// Evaluates one token with optional-research layer timing on the ordinary path.
    ///
    /// Preserves the configured arithmetic, row workers, standard adapters and
    /// synchronous executor. This is a full token evaluation, not tiled prefill,
    /// speculative verification, sampled generation or an isolated-kernel timer.
    /// Timing changes neither graph/checkpoint identity nor numerical ordering.
    /// Use an external original deadline and process supervisor for model trials;
    /// this synchronous method cannot preempt a projection or enforce a deadline.
    ///
    /// # Errors
    /// Rejects graphs above 64 layers before evaluation, and returns the ordinary
    /// graph errors. Every attempt replaces the record, including rejected ones.
    /// Errors clear output logits; completed brackets do not imply token commit.
    pub fn evaluate_with_timings_into(
        &mut self,
        token: u32,
        logits: &mut Vec<f32>,
        timings: &mut TokenTimings,
    ) -> Result<()> {
        *timings = TokenTimings::default();
        if self.layers.len() > MAX_TIMED_LAYERS {
            logits.clear();
            return Err(invalid("layer attribution supports at most 64 layers"));
        }
        timings.layer_count = self.layers.len();
        timings.entered = true;
        let started = Instant::now();
        let result = self.evaluate_token_recorded_into(token, Some(logits), Some(timings));
        timings.elapsed = started.elapsed();
        timings.succeeded = result.is_ok();
        if result.is_err() {
            logits.clear();
        }
        result
    }
}
