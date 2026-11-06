//! Explicit synchronous extension for the existing prepared PQ2 activation path.

use bonsai_ternary::Q8KBlock;

use super::{PQ2, PTQ1, PreparedMatvec, Q8Block, dimensions, finite};
use crate::{Result, invalid, model::Tensor};

/// Scoped executor for original PQ2 rows and already prepared `Q8_K` activations.
///
/// The graph retains transforms, activation quantization, adapter application and
/// all mutable model state. An executor must preserve original per-128-block
/// scale/reduction order, including `fused` multiply-add selection. It must not
/// retain borrowed slices after return; uncertain native consumers require owned
/// staging retained through verified teardown by an external process supervisor.
/// No implementation or successful invocation establishes device qualification.
pub trait Pq2ProjectionExecutor {
    /// Stable arithmetic/placement identity, excluding unverified device claims.
    fn identity(&self) -> &str;

    /// Remaining submissions admitted by this caller-bound executor session.
    ///
    /// `None` denotes an opaque capacity. A bounded executor must report counts
    /// for every selected projection, allowing complete-work admission before
    /// the first submission. The caller must bind the native session's actual
    /// remaining capacity, including work performed before borrowing it.
    fn operation_capacity(&self) -> Option<u64> {
        None
    }

    /// Stable submission count for one selected projection, or unknown.
    ///
    /// When capacity is bounded, an unknown count must reject admission. Counts
    /// include every row tile, original block and arithmetic lane composition.
    fn operation_count(&self, _tensor: &Tensor) -> Option<u64> {
        None
    }

    /// Decide eligibility from validated geometry before any device submission.
    ///
    /// `false` keeps this whole projection on its configured CPU backend. A later
    /// execution error is propagated without CPU replay or partial continuation.
    /// Eligibility and [`Self::operation_count`] must remain stable for the
    /// complete borrowed graph scope, whose admission may cache these decisions.
    fn supports(&self, tensor: &Tensor) -> bool;

    /// Replace the complete projection using the existing exact quantized codes.
    ///
    /// # Errors
    /// Rejects unavailable resources, cancellation, native failures and malformed
    /// output. The caller clears output and closes its scoped decoder on failure.
    fn project(
        &mut self,
        tensor: &Tensor,
        activation: &[Q8KBlock],
        fused: bool,
        output: &mut Vec<f32>,
    ) -> Result<()>;

    /// Opt in to PTQ1 or odd-block PQ2 with original prepared `Q8_0` codes.
    ///
    /// Existing executors leave these paths on CPU. Eligibility must precede
    /// submission and must not depend on a failed device attempt.
    /// Eligibility and operation counts must remain stable for the complete
    /// borrowed graph scope.
    fn supports_q8(&self, _tensor: &Tensor) -> bool {
        false
    }

    /// Execute a complete `Q8_0` projection without changing its float order.
    ///
    /// When `pinned` and PTQ1, retain four independent DOTPROD lanes across the
    /// complete row: each per-32 group's lane sums positions `4*i..4*i+4` and
    /// `4*i+16..4*i+20`, FMA into that lane, then reduce adjacent pairs once.
    /// Otherwise retain per-32 scale multiplication, per-128 group summation and
    /// outer weight-scale multiplication in their original nonfused order.
    ///
    /// # Errors
    /// Rejects unsupported arithmetic, submission failures and invalid output.
    fn project_q8(
        &mut self,
        _tensor: &Tensor,
        _activation: &[Q8Block],
        _pinned: bool,
        _output: &mut Vec<f32>,
    ) -> Result<()> {
        Err(invalid("projection executor does not support Q8_0"))
    }

    /// Opt in to direct full-`K` `F32` activation submission for `PQ2_0` tensors.
    ///
    /// Existing bitplane and integer-guarded executors leave this `false`.
    fn supports_f32(&self, _tensor: &Tensor) -> bool {
        false
    }

    /// Execute a complete `PQ2_0` projection directly from the `F32` activation vector.
    ///
    /// # Errors
    /// Rejects unsupported arithmetic, submission failures and invalid output.
    fn project_f32(
        &mut self,
        _tensor: &Tensor,
        _activation: &[f32],
        _pinned: bool,
        _output: &mut Vec<f32>,
    ) -> Result<()> {
        Err(invalid("projection executor does not support direct F32"))
    }
}

impl PreparedMatvec<'_> {
    /// Execute an eligible original-PQ2 projection through a scoped executor.
    ///
    /// PTQ1 and odd-block PQ2 use `Q8_0` only with an explicit executor opt-in.
    /// Other types retain configured CPU execution.
    /// Eligibility is decided before submission. Executor errors never trigger
    /// replay. Complete finite output is required before publication; CPU backend
    /// statistics do not count work performed by an external executor.
    ///
    /// # Errors
    /// Rejects incompatible preparation, executor failures or incomplete output.
    pub fn matvec_with_executor(
        &mut self,
        tensor: &Tensor,
        output: &mut Vec<f32>,
        executor: &mut dyn Pq2ProjectionExecutor,
    ) -> Result<()> {
        output.clear();
        let result = (|| {
            let (columns, rows, _) = dimensions(tensor)?;
            if !self.is_compatible(tensor)? {
                return Err(invalid(
                    "incompatible prepared projection executor activation",
                ));
            }
            if tensor.kind == PQ2 && columns.is_multiple_of(128) && executor.supports_f32(tensor) {
                executor.project_f32(
                    tensor,
                    self.input,
                    self.scratch.pinned_accumulation,
                    output,
                )?;
            } else if tensor.kind == PQ2 && columns.is_multiple_of(256) && executor.supports(tensor)
            {
                if !self.quantized {
                    return Err(invalid(
                        "prepared projection executor activation requires quantization",
                    ));
                }
                executor.project(
                    tensor,
                    &self.scratch.q8k,
                    self.scratch.pinned_accumulation,
                    output,
                )?;
            } else if (tensor.kind == PTQ1 || tensor.kind == PQ2 && !columns.is_multiple_of(256))
                && executor.supports_q8(tensor)
            {
                if !self.quantized {
                    return Err(invalid(
                        "prepared projection executor activation requires quantization",
                    ));
                }
                executor.project_q8(
                    tensor,
                    &self.scratch.q8,
                    self.scratch.pinned_accumulation,
                    output,
                )?;
            } else {
                return self.matvec_into(tensor, output);
            }
            if output.len() != rows {
                return Err(invalid("projection executor returned incomplete output"));
            }
            for &value in output.iter() {
                finite(value)?;
            }
            Ok(())
        })();
        if result.is_err() {
            output.clear();
        }
        result
    }
}
