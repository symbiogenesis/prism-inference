//! Explicit synchronous extension for the existing prepared PQ2 activation path.

use bonsai_ternary::Q8KBlock;

use super::{PQ2, PTQ1, PreparedMatvec, Q8Block, dimensions, is_finite_slice};
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

    /// Opt in to batched direct `F32` activation submission for a group of
    /// compatible sibling `PQ2_0` projections sharing one input vector.
    fn supports_f32_group(&self, _tensors: &[&Tensor]) -> bool {
        false
    }

    /// Execute a group of compatible `PQ2_0` projections sharing one `F32`
    /// activation vector in a single batched executor submission.
    ///
    /// # Errors
    /// Rejects unsupported groups, submission failures and invalid output.
    fn project_f32_group(
        &mut self,
        _tensors: &[&Tensor],
        _activation: &[f32],
        _pinned: bool,
        _outputs: &mut [&mut Vec<f32>],
    ) -> Result<()> {
        Err(invalid(
            "projection executor does not support batched F32 groups",
        ))
    }

    /// Execute a single `PQ2_0` projection across a tile of `F32` token activations.
    ///
    /// # Errors
    /// Rejects mismatched slice lengths, unsupported arithmetic or submission failures.
    fn project_f32_tokens(
        &mut self,
        tensor: &Tensor,
        activations: &[Vec<f32>],
        pinned: bool,
        outputs: &mut [Vec<f32>],
    ) -> Result<()> {
        if activations.len() != outputs.len() {
            return Err(invalid("mismatched F32 token projection outputs"));
        }
        for (activation, output) in activations.iter().zip(outputs.iter_mut()) {
            self.project_f32(tensor, activation, pinned, output)?;
        }
        Ok(())
    }

    /// Execute a pair of compatible `PQ2_0` sibling projections across a tile of
    /// shared `F32` token activations.
    ///
    /// # Errors
    /// Rejects mismatched slice lengths, unsupported groups or submission failures.
    fn project_f32_pair_tokens(
        &mut self,
        tensors: [&Tensor; 2],
        activations: &[Vec<f32>],
        pinned: bool,
        outputs_a: &mut [Vec<f32>],
        outputs_b: &mut [Vec<f32>],
    ) -> Result<()> {
        if activations.len() != outputs_a.len() || activations.len() != outputs_b.len() {
            return Err(invalid("mismatched F32 pair token projection outputs"));
        }
        for ((activation, out_a), out_b) in activations
            .iter()
            .zip(outputs_a.iter_mut())
            .zip(outputs_b.iter_mut())
        {
            self.project_f32_group(&tensors, activation, pinned, &mut [out_a, out_b])?;
        }
        Ok(())
    }

    /// Execute a triple of compatible `PQ2_0` sibling projections across a tile of
    /// shared `F32` token activations.
    ///
    /// # Errors
    /// Rejects mismatched slice lengths, unsupported groups or submission failures.
    fn project_f32_triple_tokens(
        &mut self,
        tensors: [&Tensor; 3],
        activations: &[Vec<f32>],
        pinned: bool,
        outputs_a: &mut [Vec<f32>],
        outputs_b: &mut [Vec<f32>],
        outputs_c: &mut [Vec<f32>],
    ) -> Result<()> {
        if activations.len() != outputs_a.len()
            || activations.len() != outputs_b.len()
            || activations.len() != outputs_c.len()
        {
            return Err(invalid("mismatched F32 triple token projection outputs"));
        }
        for (((activation, out_a), out_b), out_c) in activations
            .iter()
            .zip(outputs_a.iter_mut())
            .zip(outputs_b.iter_mut())
            .zip(outputs_c.iter_mut())
        {
            self.project_f32_group(&tensors, activation, pinned, &mut [out_a, out_b, out_c])?;
        }
        Ok(())
    }

    /// Opt in to a fused `ffn_gate` + `ffn_up` + `SwiGLU` submission for a pair
    /// of compatible `PQ2_0` FFN projections sharing one `F32` activation vector.
    fn supports_f32_ffn_swiglu(&self, _gate: &Tensor, _up: &Tensor) -> bool {
        false
    }

    /// Execute a fused `ffn_gate` + `ffn_up` + `SwiGLU(gate(x), up(x))` chain
    /// in a single executor submission.
    ///
    /// # Errors
    /// Rejects unsupported fusion, submission failures and invalid output.
    fn project_f32_ffn_swiglu(
        &mut self,
        _gate: &Tensor,
        _up: &Tensor,
        _activation: &[f32],
        _pinned: bool,
        _output: &mut Vec<f32>,
    ) -> Result<()> {
        Err(invalid(
            "projection executor does not support fused F32 FFN SwiGLU",
        ))
    }

    /// Execute a fused `ffn_gate` + `ffn_up` + `SwiGLU(gate(x), up(x))` chain
    /// across a tile of `F32` token activations.
    ///
    /// # Errors
    /// Rejects mismatched slice lengths, unsupported fusion or submission failures.
    fn project_f32_ffn_swiglu_tokens(
        &mut self,
        gate: &Tensor,
        up: &Tensor,
        activations: &[Vec<f32>],
        pinned: bool,
        outputs: &mut [Vec<f32>],
    ) -> Result<()> {
        if activations.len() != outputs.len() {
            return Err(invalid(
                "mismatched F32 FFN SwiGLU token projection outputs",
            ));
        }
        for (activation, output) in activations.iter().zip(outputs.iter_mut()) {
            self.project_f32_ffn_swiglu(gate, up, activation, pinned, output)?;
        }
        Ok(())
    }

    /// Opt in to a full intra-layer FFN chain (`[ffn_gate, ffn_up, ffn_down]` +
    /// `SwiGLU` + optional `Hadamard` + optional residual `ADD`) in a single
    /// executor submission with intermediate residency chaining.
    fn supports_f32_ffn_chain(&self, _tensors: [&Tensor; 3]) -> bool {
        false
    }

    /// Execute a full intra-layer FFN chain (`[ffn_gate, ffn_up, ffn_down]` +
    /// `SwiGLU(gate(x), up(x))` + optional `Hadamard` + optional `residual` add)
    /// in a single executor submission.
    ///
    /// # Errors
    /// Rejects unsupported chains, submission failures and invalid output.
    fn project_f32_ffn_chain(
        &mut self,
        _tensors: [&Tensor; 3],
        _activation: &[f32],
        _residual: Option<&[f32]>,
        _hadamard_block_size: Option<u32>,
        _pinned: bool,
        _output: &mut Vec<f32>,
    ) -> Result<()> {
        Err(invalid(
            "projection executor does not support fused F32 FFN chain",
        ))
    }

    /// Execute a full intra-layer FFN chain across a tile of `F32` token activations.
    ///
    /// # Errors
    /// Rejects mismatched slice lengths, unsupported chains or submission failures.
    fn project_f32_ffn_chain_tokens(
        &mut self,
        tensors: [&Tensor; 3],
        activations: &[Vec<f32>],
        residuals: Option<&[Vec<f32>]>,
        hadamard_block_size: Option<u32>,
        pinned: bool,
        outputs: &mut [Vec<f32>],
    ) -> Result<()> {
        if activations.len() != outputs.len()
            || residuals.is_some_and(|res| res.len() != activations.len())
        {
            return Err(invalid("mismatched F32 FFN chain token projection outputs"));
        }
        for (tok, (activation, output)) in activations.iter().zip(outputs.iter_mut()).enumerate() {
            let residual = residuals.map(|res| res[tok].as_slice());
            self.project_f32_ffn_chain(
                tensors,
                activation,
                residual,
                hadamard_block_size,
                pinned,
                output,
            )?;
        }
        Ok(())
    }
}

impl PreparedMatvec<'_> {
    /// Execute a full intra-layer FFN chain (`[gate, up, down]` + `SwiGLU` +
    /// optional `Hadamard` + optional `residual` add) in a single executor call
    /// when supported by `executor`.
    ///
    /// # Errors
    /// Rejects incompatible preparation, executor failures or incomplete output.
    pub fn matvec_ffn_chain_with_executor(
        &mut self,
        tensors: [&Tensor; 3],
        residual: Option<&[f32]>,
        hadamard_block_size: Option<u32>,
        output: &mut Vec<f32>,
        executor: &mut dyn Pq2ProjectionExecutor,
    ) -> Result<()> {
        let result = (|| {
            let [gate, up, down] = tensors;
            let (gate_cols, gate_rows, _) = dimensions(gate)?;
            let (up_cols, up_rows, _) = dimensions(up)?;
            let (down_cols, down_rows, _) = dimensions(down)?;
            if !self.is_compatible(gate)?
                || !self.is_compatible(up)?
                || gate.kind != PQ2
                || up.kind != PQ2
                || down.kind != PQ2
                || gate_cols != up_cols
                || gate_rows != up_rows
                || down_cols != gate_rows
                || down_rows != gate_cols
                || !gate_cols.is_multiple_of(128)
                || !down_cols.is_multiple_of(128)
                || residual.is_some_and(|res| res.len() != down_rows || !is_finite_slice(res))
                || !executor.supports_f32_ffn_chain(tensors)
            {
                return Err(invalid(
                    "incompatible prepared projection executor activation",
                ));
            }
            executor.project_f32_ffn_chain(
                tensors,
                self.input,
                residual,
                hadamard_block_size,
                self.scratch.pinned_accumulation,
                output,
            )?;
            if output.len() != down_rows {
                return Err(invalid("projection executor returned incomplete output"));
            }
            if !is_finite_slice(output) {
                return Err(invalid("non-finite value in runtime path"));
            }
            Ok(())
        })();
        if result.is_err() {
            output.clear();
        }
        result
    }
    /// Execute a fused `ffn_gate` + `ffn_up` + `SwiGLU` chain in a single
    /// executor call when supported by `executor`.
    ///
    /// # Errors
    /// Rejects incompatible preparation, executor failures or incomplete output.
    pub fn matvec_ffn_swiglu_with_executor(
        &mut self,
        gate: &Tensor,
        up: &Tensor,
        output: &mut Vec<f32>,
        executor: &mut dyn Pq2ProjectionExecutor,
    ) -> Result<()> {
        let result = (|| {
            let (gate_cols, gate_rows, _) = dimensions(gate)?;
            let (up_cols, up_rows, _) = dimensions(up)?;
            if !self.is_compatible(gate)?
                || !self.is_compatible(up)?
                || gate.kind != PQ2
                || up.kind != PQ2
                || gate_cols != up_cols
                || gate_rows != up_rows
                || !gate_cols.is_multiple_of(128)
                || !executor.supports_f32_ffn_swiglu(gate, up)
            {
                return Err(invalid(
                    "incompatible prepared projection executor activation",
                ));
            }
            executor.project_f32_ffn_swiglu(
                gate,
                up,
                self.input,
                self.scratch.pinned_accumulation,
                output,
            )?;
            if output.len() != gate_rows {
                return Err(invalid("projection executor returned incomplete output"));
            }
            if !is_finite_slice(output) {
                return Err(invalid("non-finite value in runtime path"));
            }
            Ok(())
        })();
        if result.is_err() {
            output.clear();
        }
        result
    }
    /// Execute a group of compatible `PQ2_0` sibling projections in a single
    /// batched executor call when supported by `executor`.
    ///
    /// # Errors
    /// Rejects incompatible preparation, executor failures or incomplete output.
    pub fn matvec_group_with_executor(
        &mut self,
        tensors: &[&Tensor],
        outputs: &mut [&mut Vec<f32>],
        executor: &mut dyn Pq2ProjectionExecutor,
    ) -> Result<()> {
        if tensors.is_empty() || tensors.len() != outputs.len() {
            return Err(invalid("mismatched projection group outputs"));
        }
        let result = (|| {
            if !executor.supports_f32_group(tensors) {
                return Err(invalid(
                    "projection executor does not support this F32 group",
                ));
            }
            for &tensor in tensors {
                let (columns, _, _) = dimensions(tensor)?;
                if !self.is_compatible(tensor)?
                    || tensor.kind != PQ2
                    || !columns.is_multiple_of(128)
                {
                    return Err(invalid(
                        "incompatible prepared projection executor activation",
                    ));
                }
            }
            executor.project_f32_group(
                tensors,
                self.input,
                self.scratch.pinned_accumulation,
                outputs,
            )?;
            for (output, &tensor) in outputs.iter().zip(tensors) {
                let (_, rows, _) = dimensions(tensor)?;
                if output.len() != rows {
                    return Err(invalid("projection executor returned incomplete output"));
                }
                if !is_finite_slice(output) {
                    return Err(invalid("non-finite value in runtime path"));
                }
            }
            Ok(())
        })();
        if result.is_err() {
            for output in outputs.iter_mut() {
                output.clear();
            }
        }
        result
    }

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
                output.clear();
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
                output.clear();
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
            if !is_finite_slice(output) {
                return Err(invalid("non-finite value in runtime path"));
            }
            Ok(())
        })();
        if result.is_err() {
            output.clear();
        }
        result
    }
}

#[cfg(test)]
mod tests;
