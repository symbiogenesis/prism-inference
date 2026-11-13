//! Joined BF16 sibling rows retaining the original rounded activation and dots.

use super::super::{ActivationContract, BF16, PreparedMatvec, bf16_dot, dimensions};
use crate::{Result, invalid, model::Tensor};

impl PreparedMatvec<'_> {
    /// One atomic BF16 output group with unchanged row partitions and arithmetic.
    pub(crate) fn matvec_bf16_siblings_into(
        &mut self,
        tensors: &[&Tensor],
        outputs: &mut [&mut Vec<f32>],
    ) -> Result<()> {
        self.scratch.reset_diagnostics();
        let result = self.multiply_bf16_siblings(tensors, outputs);
        if result.is_err() {
            for output in outputs {
                output.clear();
            }
        }
        self.scratch.diagnostics.succeeded = result.is_ok();
        for _ in tensors {
            self.scratch.statistics.record(self.scratch.diagnostics);
        }
        result
    }

    fn multiply_bf16_siblings(
        &mut self,
        tensors: &[&Tensor],
        outputs: &mut [&mut Vec<f32>],
    ) -> Result<()> {
        if !(2..=4).contains(&tensors.len())
            || outputs.len() != tensors.len()
            || !self.quantized
            || self.contract != ActivationContract::BF16
        {
            return Err(invalid("incompatible prepared BF16 sibling group"));
        }
        let mut row_bytes = 0;
        // Complete geometry and fallible output growth precede any worker work.
        // The wrapper clears all siblings on every failure, including allocation.
        for (tensor, output) in tensors.iter().zip(outputs.iter_mut()) {
            let (columns, rows, bytes) = dimensions(tensor)?;
            if tensor.kind != BF16 || columns != self.input.len() {
                return Err(invalid("incompatible prepared BF16 sibling tensor"));
            }
            row_bytes = bytes;
            if output.len() != rows {
                output.clear();
                output.try_reserve_exact(rows)?;
                output.resize(rows, 0.0);
            }
        }
        let scratch = &mut *self.scratch;
        // BF16 retains its existing scalar diagnostic classification even when
        // the native row implementation uses NEON loads and binary64 sums.
        scratch.diagnostics = scratch.diagnostics.select(false);
        if scratch.row_parallelism.workers() > 1 {
            scratch.statistics.grouped_row_dispatches =
                scratch.statistics.grouped_row_dispatches.saturating_add(1);
            for _ in tensors {
                scratch.statistics.parallel_row_operations =
                    scratch.statistics.parallel_row_operations.saturating_add(1);
            }
        }
        let mut data = [&[][..]; 4];
        for (data, tensor) in data.iter_mut().zip(tensors) {
            *data = &tensor.data;
        }
        let activations = &scratch.activations;
        scratch.row_parallelism.fill_siblings(
            &data[..tensors.len()],
            row_bytes,
            outputs,
            |bytes, output| {
                #[cfg(all(
                    feature = "experimental-neon",
                    target_arch = "aarch64",
                    target_os = "windows",
                    target_env = "msvc",
                    target_feature = "neon"
                ))]
                {
                    super::super::parallel_rows::fill_batched_chunk(
                        bytes,
                        row_bytes,
                        output,
                        &|rows| {
                            super::super::neon::bf16_dot_eight_rows(rows, activations)
                                .ok_or_else(|| invalid("missing NEON BF16 kernel"))?
                        },
                        &|rows| {
                            super::super::neon::bf16_dot_four_rows(rows, activations)
                                .ok_or_else(|| invalid("missing NEON BF16 kernel"))?
                        },
                        &|bytes| bf16_dot(bytes, activations),
                    )
                }
                #[cfg(not(all(
                    feature = "experimental-neon",
                    target_arch = "aarch64",
                    target_os = "windows",
                    target_env = "msvc",
                    target_feature = "neon"
                )))]
                {
                    for (row, value) in bytes.chunks_exact(row_bytes).zip(output) {
                        *value = bf16_dot(row, activations)?;
                    }
                    Ok(())
                }
            },
        )
    }
}
