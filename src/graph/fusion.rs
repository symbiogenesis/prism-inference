//! Execution witnesses for the existing decode preparation paths.

/// Lifetime observations for one eligible fusion path.
///
/// Counts saturate and retain no inputs. Failed attempts can be followed by
/// separate successful operations; they do not count as completed fusions.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FusionCount {
    /// Eligible invocations entering the fused implementation.
    pub attempts: u64,
    /// Invocations returning complete successful output.
    pub completed: u64,
    /// Unsuccessful invocations explicitly continuing on the separate path.
    pub fallbacks: u64,
}

impl FusionCount {
    pub(super) const fn attempt(&mut self) {
        self.attempts = self.attempts.saturating_add(1);
    }

    pub(super) const fn complete(&mut self) {
        self.completed = self.completed.saturating_add(1);
    }

    pub(super) const fn fallback(&mut self) {
        self.fallbacks = self.fallbacks.saturating_add(1);
    }
}

/// Decode fusion observations, independent of committed model state.
///
/// A completed count witnesses this graph branch, not hardware execution or
/// equivalence to an unfused numerical oracle. Tiled prefill has separate paths.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FusionStatistics {
    /// RMS normalization and Hadamard preparation.
    pub norm_hadamard: FusionCount,
    /// Residual addition, RMS normalization and Hadamard preparation.
    pub add_norm_hadamard: FusionCount,
    /// Read-only SwiGLU and Hadamard preparation without an adapter.
    pub swiglu_hadamard: FusionCount,
    /// SwiGLU and Hadamard preparation preserving original-basis adapter input.
    pub adapter_swiglu_hadamard: FusionCount,
    /// Scoped external gate/up/SwiGLU submission.
    pub external_ffn_swiglu: FusionCount,
    /// Scoped external complete FFN-chain submission.
    pub external_ffn_chain: FusionCount,
}
