//! Explicit synchronous extension for the existing prepared PQ2 activation path.

use bonsai_ternary::Q8KBlock;

use super::{PQ2, PreparedMatvec, dimensions, finite};
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

    /// Decide eligibility from validated geometry before any device submission.
    ///
    /// `false` keeps this whole projection on its configured CPU backend. A later
    /// execution error is propagated without CPU replay or partial continuation.
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
}

impl PreparedMatvec<'_> {
    /// Execute an eligible original-PQ2 projection through a scoped executor.
    ///
    /// Other types and the odd-block `Q8_0` path retain configured CPU execution.
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
            if tensor.kind != PQ2 || !columns.is_multiple_of(256) || !executor.supports(tensor) {
                return self.matvec_into(tensor, output);
            }
            executor.project(
                tensor,
                &self.scratch.q8k,
                self.scratch.pinned_accumulation,
                output,
            )?;
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
