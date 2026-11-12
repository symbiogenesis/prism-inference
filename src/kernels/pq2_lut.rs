//! Explicit exact activation-table projection experiment, independent of ISA selection.

use super::{
    PQ2, Pq2ProjectionExecutor, Q8Block, Q8KBlock, RowParallelism, Tl1BlockLut, dimensions, finite,
    integer_float, scale,
};
use crate::{Result, invalid, model::Tensor};

/// Opt-in PQ2 projection using one activation-pair table per 128 input columns.
///
/// Use through [`super::PreparedMatvec::matvec_with_executor`] or
/// [`crate::graph::Qwen35::with_projection_executor`]. Each projection prepares
/// its tables once and shares them across every output row and configured row
/// worker. Tables are rebuilt from the current borrowed quantized codes on every
/// invocation; pointer identity never permits stale activation reuse.
///
/// Tables retain 2,048 bytes per 128 columns (80 KiB at width 5,120), bounded by
/// the caller's explicit maximum width. Original packed weights, quantizers and
/// accumulation order remain unchanged, including PQ2's fourth (+2) symbol and
/// odd-group `Q8_0` scales. Unsupported tensors retain the graph's CPU path.
///
/// This experiment is never selected automatically. Compare complete table
/// preparation, row dispatch and dot cost against baseline NEON and eligible
/// SDOT before selecting it for a device; no speedup is established here.
pub struct Pq2LutExecutor {
    max_columns: usize,
    tables: Vec<Tl1BlockLut>,
    parallelism: RowParallelism,
    row_errors: Vec<Option<crate::Error>>,
    #[cfg(test)]
    preparations: usize,
}

impl Pq2LutExecutor {
    /// Creates a bounded table experiment sharing an existing row-worker policy.
    ///
    /// Allocation remains fallible and occurs only when preparing a projection.
    /// No thread pool or weight copy is created by this constructor.
    ///
    /// # Errors
    /// Rejects zero/incomplete maximum widths or overflowing table extents.
    pub fn new(max_columns: usize, parallelism: RowParallelism) -> Result<Self> {
        if max_columns == 0
            || !max_columns.is_multiple_of(128)
            || (max_columns / 128)
                .checked_mul(size_of::<Tl1BlockLut>())
                .is_none()
        {
            return Err(invalid("PQ2 LUT requires a bounded complete input width"));
        }
        Ok(Self {
            max_columns,
            tables: Vec::new(),
            parallelism,
            row_errors: Vec::new(),
            #[cfg(test)]
            preparations: 0,
        })
    }

    fn columns(&self, tensor: &Tensor) -> Option<usize> {
        tensor.dimensions.first().copied().filter(|&columns| {
            tensor.kind == PQ2
                && columns != 0
                && columns <= self.max_columns
                && columns.is_multiple_of(128)
        })
    }

    fn prepare(&mut self, count: usize) -> Result<()> {
        self.tables.clear();
        self.tables.try_reserve_exact(count)?;
        Ok(())
    }

    fn push(&mut self, values: &[i8; 128]) {
        self.tables
            .push(Tl1BlockLut::from_contiguous_activations(values));
        #[cfg(test)]
        {
            self.preparations += 1;
        }
    }

    fn finish(&mut self, result: Result<()>, output: &mut Vec<f32>) -> Result<()> {
        // No prepared activation survives a call, even when execution fails.
        self.tables.clear();
        self.row_errors.clear();
        if result.is_err() {
            output.clear();
        }
        result
    }
}

impl Pq2ProjectionExecutor for Pq2LutExecutor {
    fn identity(&self) -> &'static str {
        "cpu-pq2-tl1-exact-experimental"
    }

    fn supports(&self, tensor: &Tensor) -> bool {
        self.columns(tensor)
            .is_some_and(|columns| columns.is_multiple_of(256))
    }

    #[expect(
        clippy::suboptimal_flops,
        reason = "Retain the caller's exact separate multiply/add or fused block accumulation."
    )]
    fn project(
        &mut self,
        tensor: &Tensor,
        activation: &[Q8KBlock],
        fused: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        let result = (|| {
            let (columns, _, row_bytes) = dimensions(tensor)?;
            if !self.supports(tensor) || activation.len() != columns / 256 {
                return Err(invalid("incompatible PQ2 LUT Q8_K activation"));
            }
            self.prepare(columns / 128)?;
            for block in activation {
                finite(block.scale)?;
                for values in block.values.as_chunks::<128>().0 {
                    self.push(values);
                }
            }
            let tables = &self.tables;
            self.parallelism.fill(
                &tensor.data,
                row_bytes,
                output,
                &mut self.row_errors,
                |row| {
                    let mut sum = 0.0;
                    for (index, block) in row.as_chunks::<34>().0.iter().enumerate() {
                        let d = scale(block)? * activation[index / 2].scale;
                        let dot = integer_float(tables[index].dot(block[2..].try_into()?));
                        sum = if fused {
                            d.mul_add(dot, sum)
                        } else {
                            sum + d * dot
                        };
                    }
                    finite(sum)
                },
            )
        })();
        self.finish(result, output)
    }

    fn supports_q8(&self, tensor: &Tensor) -> bool {
        self.columns(tensor)
            .is_some_and(|columns| !columns.is_multiple_of(256))
    }

    #[expect(
        clippy::suboptimal_flops,
        reason = "Odd-group PQ2 keeps four ordered Q8_0 scale products before each weight scale."
    )]
    fn project_q8(
        &mut self,
        tensor: &Tensor,
        activation: &[Q8Block],
        _pinned: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        let result = (|| {
            let (columns, _, row_bytes) = dimensions(tensor)?;
            if !self.supports_q8(tensor) || activation.len() != columns / 32 {
                return Err(invalid("incompatible PQ2 LUT Q8_0 activation"));
            }
            self.prepare(columns / 128)?;
            for blocks in activation.as_chunks::<4>().0 {
                let mut values = [0_i8; 128];
                for (destination, block) in values.as_chunks_mut::<32>().0.iter_mut().zip(blocks) {
                    finite(block.scale)?;
                    destination.copy_from_slice(&block.values);
                }
                self.push(&values);
            }
            let tables = &self.tables;
            self.parallelism.fill(
                &tensor.data,
                row_bytes,
                output,
                &mut self.row_errors,
                |row| {
                    let mut sum = 0.0;
                    for (index, block) in row.as_chunks::<34>().0.iter().enumerate() {
                        let d = scale(block)?;
                        let dots = tables[index].dot_groups(block[2..].try_into()?);
                        let mut group = 0.0;
                        for (&dot, activation) in dots.iter().zip(&activation[index * 4..][..4]) {
                            group += activation.scale * integer_float(dot);
                        }
                        sum += d * group;
                    }
                    finite(sum)
                },
            )
        })();
        self.finish(result, output)
    }
}

#[cfg(test)]
#[path = "pq2_lut_tests.rs"]
mod tests;
