//! Joined Q8_0-backed siblings using the existing exact packed row kernels.

use super::super::{ActivationContract, PQ2, PTQ1, PreparedMatvec, dimensions};
use crate::{Result, invalid, model::Tensor};

impl PreparedMatvec<'_> {
    /// Join same-format PTQ1 or odd-group PQ2 rows without changing arithmetic.
    pub(crate) fn matvec_q8_siblings_into(
        &mut self,
        tensors: &[&Tensor],
        outputs: &mut [&mut Vec<f32>],
    ) -> Result<()> {
        self.scratch.reset_diagnostics();
        let result = self.multiply_q8_siblings(tensors, outputs);
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

    #[allow(
        clippy::too_many_lines,
        reason = "Keep complete group admission and exact native/portable row dispatch together."
    )]
    fn multiply_q8_siblings(
        &mut self,
        tensors: &[&Tensor],
        outputs: &mut [&mut Vec<f32>],
    ) -> Result<()> {
        if !(2..=4).contains(&tensors.len())
            || outputs.len() != tensors.len()
            || !self.quantized
            || !matches!(
                self.contract,
                ActivationContract::Q8 | ActivationContract::PinnedPtq1Q8
            )
        {
            return Err(invalid("incompatible prepared Q8 sibling group"));
        }
        let kind = tensors[0].kind;
        let mut row_bytes = 0;
        // Admit every extent and allocation before dispatch. Require one exact
        // format, including the pinned PTQ1-specific activation quantizer.
        for (tensor, output) in tensors.iter().zip(outputs.iter_mut()) {
            let (columns, rows, bytes) = dimensions(tensor)?;
            if tensor.kind != kind
                || !matches!(kind, PQ2 | PTQ1)
                || columns != self.input.len()
                || kind == PQ2 && columns.is_multiple_of(256)
                || ActivationContract::for_tensor(tensor, columns, self.scratch.pinned_accumulation)
                    != self.contract
            {
                return Err(invalid("incompatible prepared Q8 sibling tensor"));
            }
            row_bytes = bytes;
            if output.len() != rows {
                output.clear();
                output.try_reserve_exact(rows)?;
                output.resize(rows, 0.0);
            }
        }
        let scratch = &mut *self.scratch;
        let dot = scratch.select_q8_groups();
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
        #[cfg(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc",
            target_feature = "neon"
        ))]
        if kind == PTQ1
            && matches!(
                scratch.selected_backend(),
                super::super::MatvecBackend::ExperimentalNeon
            )
        {
            let q8 = &scratch.q8;
            let pinned = scratch.pinned_accumulation;
            return scratch.row_parallelism.fill_siblings(
                &data[..tensors.len()],
                row_bytes,
                outputs,
                |bytes, output| {
                    super::super::parallel_rows::fill_batched_chunk(
                        bytes,
                        row_bytes,
                        output,
                        &|rows| {
                            if pinned {
                                super::super::neon::dot_ptq1_eight_rows_pinned(rows, q8)
                            } else {
                                super::super::neon::dot_ptq1_eight_rows_source(rows, q8)
                            }
                        },
                        &|rows| {
                            if pinned {
                                super::super::neon::dot_ptq1_four_rows_pinned(rows, q8)
                            } else {
                                super::super::neon::dot_ptq1_four_rows_source(rows, q8)
                            }
                        },
                        &|bytes| {
                            if pinned {
                                super::super::neon::dot_ptq1_row_pinned(bytes, q8)
                            } else {
                                super::super::neon::dot_ptq1_row_source(bytes, q8)
                            }
                        },
                    )
                },
            );
        }
        scratch.row_parallelism.fill_siblings(
            &data[..tensors.len()],
            row_bytes,
            outputs,
            |bytes, output| {
                for (row, value) in bytes.chunks_exact(row_bytes).zip(output) {
                    *value = dot(kind, row, &scratch.q8)?;
                }
                Ok(())
            },
        )
    }
}
