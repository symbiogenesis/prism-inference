//! One joined pool submission over independent prepared sibling matrices.

use super::RowParallelism;
#[cfg(feature = "parallel-rows")]
use super::{CHUNK_WORK_FACTOR, DECODE_ROW_ALIGN, MAX_ROW_WORKERS};
use crate::{Result, invalid};

#[cfg(feature = "parallel-rows")]
struct Task<'a> {
    bytes: &'a [u8],
    output: &'a mut [f32],
    error: Option<crate::Error>,
}

#[cfg(feature = "parallel-rows")]
const fn sibling_chunk_rows(rows: usize, workers: usize) -> usize {
    if workers <= 1 || rows <= DECODE_ROW_ALIGN {
        return if rows == 0 { 1 } else { rows };
    }
    let target_chunks = workers * CHUNK_WORK_FACTOR;
    rows.div_ceil(target_chunks).div_ceil(DECODE_ROW_ALIGN) * DECODE_ROW_ALIGN
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
            // Up to four sibling matrices times at most CHUNK_WORK_FACTOR * MAX_ROW_WORKERS
            // 16-row-aligned chunks. No payload, expanded weights, error-vector allocation
            // or nested pool exists.
            let mut tasks: [Task<'_>; 4 * CHUNK_WORK_FACTOR * MAX_ROW_WORKERS] =
                std::array::from_fn(|_| Task {
                    bytes: &[],
                    output: &mut [],
                    error: None,
                });
            let mut count = 0;
            for (bytes, output) in data.iter().zip(outputs.iter_mut()) {
                let task_rows = sibling_chunk_rows(output.len(), self.workers);
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
            pool.dispatch_mut(&mut tasks[..count], |_, task| {
                if let Err(error) = fill(task.bytes, task.output) {
                    task.error = Some(error);
                }
            });
            return tasks[..count]
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

#[cfg(all(test, feature = "parallel-rows"))]
mod tests {
    use super::{CHUNK_WORK_FACTOR, DECODE_ROW_ALIGN, MAX_ROW_WORKERS, sibling_chunk_rows};

    #[test]
    fn sibling_chunk_rows_enforces_sixteen_row_floor_and_bounded_chunks() {
        assert_eq!(sibling_chunk_rows(16, 18), 16);
        assert_eq!(sibling_chunk_rows(32, 18), 16);
        assert_eq!(sibling_chunk_rows(48, 18), 16);
        assert_eq!(sibling_chunk_rows(48, 18), 16);
        assert_eq!(48_usize.div_ceil(sibling_chunk_rows(48, 18)), 3);

        for workers in [1_usize, 2, 4, 12, MAX_ROW_WORKERS] {
            for rows in [1_usize, 5, 16, 17, 32, 48, 65, 512, 2048, 12_288] {
                let chunk_rows = sibling_chunk_rows(rows, workers);
                if workers > 1 && rows > DECODE_ROW_ALIGN {
                    assert!(chunk_rows >= DECODE_ROW_ALIGN);
                    assert_eq!(chunk_rows % DECODE_ROW_ALIGN, 0);
                    assert!(rows.div_ceil(chunk_rows) <= workers * CHUNK_WORK_FACTOR);
                } else {
                    assert_eq!(chunk_rows, rows);
                }
            }
        }
    }
}
