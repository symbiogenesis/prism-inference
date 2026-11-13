//! One joined pool submission over independent prepared sibling matrices.

#[cfg(feature = "parallel-rows")]
use super::MAX_ROW_WORKERS;
use super::RowParallelism;
use crate::{Result, invalid};

#[cfg(feature = "parallel-rows")]
struct Task<'a> {
    bytes: &'a [u8],
    output: &'a mut [f32],
    error: Option<crate::Error>,
}

impl RowParallelism {
    /// Retain each matrix's original contiguous worker partitions and row order.
    /// Task slots are sibling-major, so completion order cannot select an error.
    #[cfg_attr(
        not(feature = "parallel-rows"),
        expect(
            clippy::unused_self,
            reason = "The shared receiver selects the worker pool only when parallel rows are enabled."
        )
    )]
    pub(in crate::kernels) fn fill_siblings(
        &self,
        data: &[&[u8]],
        row_bytes: usize,
        outputs: &mut [&mut Vec<f32>],
        fill: impl Fn(&[u8], &mut [f32]) -> Result<()> + Sync,
    ) -> Result<()> {
        if !(2..=4).contains(&data.len())
            || outputs.len() != data.len()
            || row_bytes == 0
            || data.iter().zip(outputs.iter()).any(|(bytes, output)| {
                bytes.is_empty()
                    || !bytes.len().is_multiple_of(row_bytes)
                    || bytes.len() / row_bytes != output.len()
            })
        {
            return Err(invalid("invalid prepared sibling row geometry"));
        }
        #[cfg(feature = "parallel-rows")]
        if let Some(pool) = &self.pool {
            use rayon::prelude::*;

            // Four matrices times at most eighteen partitions. No payload,
            // expanded weights, error-vector allocation or nested pool exists.
            let mut tasks: [Task<'_>; 4 * MAX_ROW_WORKERS] = std::array::from_fn(|_| Task {
                bytes: &[],
                output: &mut [],
                error: None,
            });
            let mut count = 0;
            for (bytes, output) in data.iter().zip(outputs.iter_mut()) {
                let task_rows = output.len().div_ceil(self.workers);
                for (task, (row_bytes, values)) in tasks[count..].iter_mut().zip(
                    bytes
                        .chunks(task_rows * row_bytes)
                        .zip(output.chunks_mut(task_rows)),
                ) {
                    task.bytes = row_bytes;
                    task.output = values;
                    count += 1;
                }
            }
            pool.install(|| {
                tasks[..count].par_iter_mut().for_each(|task| {
                    if let Err(error) = fill(task.bytes, task.output) {
                        task.error = Some(error);
                    }
                });
            });
            return tasks
                .iter_mut()
                .find_map(|task| task.error.take())
                .map_or(Ok(()), Err);
        }
        for (bytes, output) in data.iter().zip(outputs) {
            fill(bytes, output)?;
        }
        Ok(())
    }
}
