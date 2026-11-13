//! CPU sibling execution sharing one prepared activation and pool join.

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
use super::MatvecBackend;
use super::{ActivationContract, PQ2, PreparedMatvec, dimensions};
use crate::{Result, invalid, model::Tensor};

#[cfg(test)]
#[path = "siblings_tests.rs"]
mod tests;

#[path = "bf16_siblings.rs"]
mod bf16;

impl PreparedMatvec<'_> {
    /// One atomic output group; original row arithmetic and partitioning remain.
    pub(crate) fn matvec_pq2_siblings_into(
        &mut self,
        tensors: &[&Tensor],
        outputs: &mut [&mut Vec<f32>],
    ) -> Result<()> {
        self.scratch.reset_diagnostics();
        let result = self.multiply_pq2_siblings(tensors, outputs);
        if result.is_err() {
            for output in outputs {
                output.clear();
            }
        }
        self.scratch.diagnostics.succeeded = result.is_ok();
        // Every matrix participates in this atomic attempt. Provisional rows
        // from a failed group are never reported as successful matrix output.
        for _ in tensors {
            self.scratch.statistics.record(self.scratch.diagnostics);
        }
        result
    }

    fn multiply_pq2_siblings(
        &mut self,
        tensors: &[&Tensor],
        outputs: &mut [&mut Vec<f32>],
    ) -> Result<()> {
        if !(2..=4).contains(&tensors.len())
            || outputs.len() != tensors.len()
            || !self.quantized
            || self.contract != ActivationContract::Q8K
        {
            return Err(invalid("incompatible prepared PQ2 sibling group"));
        }
        let mut row_bytes = 0;
        // Check every immutable extent before dispatching any sibling. Growth
        // is fallible, and any later failure clears all outputs in the wrapper.
        for (tensor, output) in tensors.iter().zip(outputs.iter_mut()) {
            let (columns, rows, bytes) = dimensions(tensor)?;
            if tensor.kind != PQ2 || columns != self.input.len() || !columns.is_multiple_of(256) {
                return Err(invalid("incompatible prepared PQ2 sibling tensor"));
            }
            row_bytes = bytes;
            if output.len() != rows {
                output.clear();
                output.try_reserve_exact(rows)?;
                output.resize(rows, 0.0);
            }
        }
        let scratch = &mut *self.scratch;
        let dot = scratch.select_pq2();
        #[cfg(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc",
            target_feature = "neon"
        ))]
        if matches!(scratch.selected_backend(), MatvecBackend::ExperimentalNeon)
            && scratch.pq2_consumption.is_empty()
        {
            scratch.prepare_pq2_consumption()?;
        }
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
        let data = &data[..tensors.len()];
        #[cfg(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc",
            target_feature = "neon"
        ))]
        if matches!(scratch.selected_backend(), MatvecBackend::ExperimentalNeon) {
            let activations = super::packed_pq2::Consumer::new(&scratch.pq2_consumption)?;
            let pinned = scratch.pinned_accumulation;
            return scratch.row_parallelism.fill_siblings(
                data,
                row_bytes,
                outputs,
                |bytes, output| {
                    super::parallel_rows::fill_batched_chunk(
                        bytes,
                        row_bytes,
                        output,
                        &|rows| {
                            if pinned {
                                activations.dot_eight_rows::<true>(rows)
                            } else {
                                activations.dot_eight_rows::<false>(rows)
                            }
                        },
                        &|rows| {
                            if pinned {
                                activations.dot_four_rows::<true>(rows)
                            } else {
                                activations.dot_four_rows::<false>(rows)
                            }
                        },
                        &|bytes| {
                            if pinned {
                                activations.dot_row::<true>(bytes)
                            } else {
                                activations.dot_row::<false>(bytes)
                            }
                        },
                    )
                },
            );
        }
        scratch
            .row_parallelism
            .fill_siblings(data, row_bytes, outputs, |bytes, output| {
                for (row, value) in bytes.chunks_exact(row_bytes).zip(output) {
                    *value = dot(row, &scratch.q8k)?;
                }
                Ok(())
            })
    }
}
