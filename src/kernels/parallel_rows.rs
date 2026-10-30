//! Explicit, bounded row scheduling over borrowed immutable weight/activation slices.
//!
//! Independently implemented after reviewing `OxiBonsai`'s borrowed-slice row
//! partitioning at 1d10c48e14754ff6999158f8fa13c7c92b40b0dc,
//! crates/oxibonsai-kernels/src/parallel.rs (Apache-2.0, COOLJAPAN OU).
//! The original Prism row kernels and their arithmetic remain unchanged.

#[cfg(feature = "parallel-rows")]
use std::sync::Arc;

use crate::{Result, invalid};

/// Largest explicit number of persistent projection workers.
pub const MAX_ROW_WORKERS: usize = 16;

/// Shared private row-worker pool, independent of Rayon global configuration.
///
/// One worker retains serial execution and creates no threads. Larger budgets
/// require `parallel-rows`; clones share the same persistent pool. Each thread
/// has a requested 512 KiB stack, additionally charged by the process owner.
/// This does not select an ISA or establish target-device benefit.
#[derive(Clone)]
pub struct RowParallelism {
    workers: usize,
    #[cfg(feature = "parallel-rows")]
    pool: Option<Arc<rayon::ThreadPool>>,
}

#[cfg(all(test, feature = "parallel-rows"))]
mod tests {
    use std::sync::Arc;

    use bonsai_ternary::pack_ptq;

    use super::super::{MatvecBackend, MatvecScratch, matvec_into};
    use super::RowParallelism;
    use crate::{Result, invalid, model::Tensor};

    #[test]
    fn pinned_parallel_rows_preserve_both_activation_contracts() -> Result<()> {
        let policy = RowParallelism::new(2)?;
        for (kind, columns) in [(142, 5120), (142, 384), (143, 6144)] {
            let mut block = [0x66_u8; 34];
            block[..2].copy_from_slice(&0x3555_u16.to_le_bytes());
            let encoded = if kind == 143 {
                pack_ptq(&block)?.to_vec()
            } else {
                block.to_vec()
            };
            let tensor = Tensor {
                name: "pinned-row-control".into(),
                dimensions: vec![columns, 7],
                kind,
                data: Arc::from(encoded.repeat(columns / 128 * 7)),
            };
            let input: Vec<_> = (0..columns)
                .map(|index| [1.5, -127.0, 0.0, -0.0, 63.5][index % 5])
                .collect();
            for backend in [MatvecBackend::Scalar, MatvecBackend::Auto] {
                let mut serial = MatvecScratch::with_pinned_backend(backend);
                let mut parallel = MatvecScratch::with_pinned_backend(backend);
                parallel.set_row_parallelism(policy.clone());
                let (mut expected, mut actual) = (Vec::new(), Vec::new());
                matvec_into(&tensor, &input, &mut expected, &mut serial)?;
                matvec_into(&tensor, &input, &mut actual, &mut parallel)?;
                assert_eq!(
                    actual
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    expected
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                );
                assert_eq!(parallel.statistics().parallel_row_operations, 1);
            }
        }
        Ok(())
    }

    #[test]
    fn row_tasks_borrow_original_storage_and_return_first_row_failure() -> Result<()> {
        let policy = RowParallelism::new(2)?;
        let data = [0_u8, 1, 2, 3, 4, 5, 6, 7];
        let mut output = Vec::with_capacity(data.len());
        let mut errors = Vec::new();
        let error = policy.fill(&data, 1, &mut output, &mut errors, |row| {
            assert_eq!(
                row.as_ptr(),
                data.as_ptr().wrapping_add(usize::from(row[0]))
            );
            assert!(
                std::thread::current()
                    .name()
                    .is_some_and(|name| name.starts_with("prism-row-"))
            );
            if row[0] == 3 || row[0] == 7 {
                Err(invalid(format!("rejected row {}", row[0])))
            } else {
                Ok(f32::from(row[0]))
            }
        });
        assert_eq!(
            error.map_err(|error| error.to_string()),
            Err("rejected row 3".to_owned())
        );
        assert!(errors.is_empty());
        assert!(errors.capacity() <= 2);
        Ok(())
    }
}

impl Default for RowParallelism {
    fn default() -> Self {
        Self {
            workers: 1,
            #[cfg(feature = "parallel-rows")]
            pool: None,
        }
    }
}

impl RowParallelism {
    /// Creates an explicit worker policy before model loading or graph work.
    ///
    /// # Errors
    /// Rejects zero, counts above [`MAX_ROW_WORKERS`], unavailable feature
    /// support, or failure to create every requested worker. No global pool is
    /// initialized and no larger or silently reduced budget is substituted.
    pub fn new(workers: usize) -> Result<Self> {
        if !(1..=MAX_ROW_WORKERS).contains(&workers) {
            return Err(invalid("row worker count must be from 1 through 16"));
        }
        if workers == 1 {
            return Ok(Self::default());
        }
        #[cfg(feature = "parallel-rows")]
        {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .stack_size(512 * 1024)
                .thread_name(|index| format!("prism-row-{index}"))
                .build()?;
            Ok(Self {
                workers,
                pool: Some(Arc::new(pool)),
            })
        }
        #[cfg(not(feature = "parallel-rows"))]
        Err(invalid("multiple row workers require parallel-rows"))
    }

    /// Configured worker budget, including the serial value one.
    #[must_use]
    pub const fn workers(&self) -> usize {
        self.workers
    }

    pub(super) const fn uses_pool(&self, rows: usize) -> bool {
        self.workers > 1 && rows > 1
    }

    pub(super) fn fill(
        &self,
        data: &[u8],
        row_bytes: usize,
        output: &mut Vec<f32>,
        errors: &mut Vec<Option<crate::Error>>,
        dot: impl Fn(&[u8]) -> Result<f32> + Sync,
    ) -> Result<()> {
        errors.clear();
        #[cfg(not(feature = "parallel-rows"))]
        if self.uses_pool(data.len() / row_bytes) {
            return Err(invalid("multiple row workers require parallel-rows"));
        }
        #[cfg(feature = "parallel-rows")]
        if let Some(pool) = &self.pool
            && data.len() / row_bytes > 1
        {
            use rayon::prelude::*;

            let rows = data.len() / row_bytes;
            let task_rows = rows.div_ceil(self.workers);
            let tasks = rows.div_ceil(task_rows);
            errors.try_reserve_exact(tasks)?;
            errors.resize_with(tasks, || None);
            // The caller reserves the exact row count fallibly before entry.
            // At most workers disjoint output chunks borrow original weights.
            output.resize(rows, 0.0);
            pool.install(|| {
                output
                    .par_chunks_mut(task_rows)
                    .zip(errors.par_iter_mut())
                    .enumerate()
                    .for_each(|(task, (chunk, error))| {
                        let start = task * task_rows * row_bytes;
                        let bytes = &data[start..start + chunk.len() * row_bytes];
                        for (value, row) in chunk.iter_mut().zip(bytes.chunks_exact(row_bytes)) {
                            match dot(row) {
                                Ok(result) => *value = result,
                                Err(failure) => {
                                    *error = Some(failure);
                                    break;
                                }
                            }
                        }
                    });
            });
            // Join all tasks, then return the earliest failing row chunk. The
            // outer matvec transaction clears all provisional output on error.
            let failure = errors.iter_mut().find_map(Option::take);
            errors.clear();
            return failure.map_or(Ok(()), Err);
        }
        for bytes in data.chunks_exact(row_bytes) {
            output.push(dot(bytes)?);
        }
        Ok(())
    }
}
