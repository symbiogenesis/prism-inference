//! One joined bounded submission for compatible CPU prompt projections.

use super::RowParallelism;
use crate::{Result, invalid};

#[cfg(feature = "parallel-rows")]
use super::{Completion, MAX_ROW_WORKERS, Task, fill_tile_task, wait_for_tile_tasks};

#[cfg(feature = "parallel-rows")]
struct SiblingTask<'a, const TOKENS: usize> {
    bytes: &'a [u8],
    rows: Task<'a, TOKENS>,
}

impl RowParallelism {
    #[expect(
        clippy::too_many_arguments,
        reason = "Keep bounded sibling borrows, joined cancellation and ordered failures together."
    )]
    pub(in crate::kernels) fn fill_tile_siblings_four_rows<const TOKENS: usize>(
        &self,
        data: &[&[u8]],
        row_bytes: usize,
        outputs: &mut [&mut [Vec<f32>]],
        decoded_blocks: &mut u64,
        check: &dyn Fn() -> Result<()>,
        dot_eight: impl Fn([&[u8]; 8], &mut u64) -> Result<[[f32; TOKENS]; 8]> + Sync,
        dot_four: impl Fn([&[u8]; 4], &mut u64) -> Result<[[f32; TOKENS]; 4]> + Sync,
        dot: impl Fn(&[u8], &mut u64) -> Result<[f32; TOKENS]> + Sync,
    ) -> Result<()> {
        let tokens = outputs.first().map_or(0, |group| group.len());
        if !(2..=3).contains(&data.len())
            || data.len() != outputs.len()
            || tokens == 0
            || tokens > TOKENS
            || row_bytes == 0
            || data.iter().zip(outputs.iter()).any(|(bytes, group)| {
                bytes.is_empty() || !bytes.len().is_multiple_of(row_bytes) || group.len() != tokens
            })
        {
            return Err(invalid("invalid sibling tile row geometry"));
        }
        #[cfg(feature = "parallel-rows")]
        if let Some(pool) = &self.pool {
            use std::sync::atomic::{AtomicBool, AtomicUsize};

            let total_rows = data.iter().try_fold(0_usize, |sum, bytes| {
                sum.checked_add(bytes.len() / row_bytes)
                    .ok_or_else(|| invalid("sibling tile row count overflow"))
            })?;
            let task_rows = total_rows.div_ceil(self.workers());
            // ceil(a/t) + ceil(b/t) + ceil(c/t) <= workers + 2.
            // Twenty token-slice task owners remain within the existing 16 KiB
            // caller/task reserve; there is no heap task vector or weight copy.
            let mut tasks: [SiblingTask<'_, TOKENS>; MAX_ROW_WORKERS + 2] =
                std::array::from_fn(|_| SiblingTask {
                    bytes: &[],
                    rows: Task {
                        outputs: std::array::from_fn(|_| -> &mut [f32] { &mut [] }),
                        error: None,
                        decoded_blocks: 0,
                    },
                });
            let mut task_count = 0;
            for (bytes, group) in data.iter().zip(outputs.iter_mut()) {
                let rows = bytes.len() / row_bytes;
                let count = rows.div_ceil(task_rows);
                for (token, output) in group.iter_mut().enumerate() {
                    output.resize(rows, 0.0);
                    for (index, (task, chunk)) in tasks[task_count..task_count + count]
                        .iter_mut()
                        .zip(output.chunks_mut(task_rows))
                        .enumerate()
                    {
                        let start = index * task_rows * row_bytes;
                        task.bytes = &bytes[start..start + chunk.len() * row_bytes];
                        task.rows.outputs[token] = chunk;
                    }
                }
                task_count += count;
            }
            let caller = std::thread::current();
            let remaining = AtomicUsize::new(task_count);
            let cancelled = AtomicBool::new(false);
            let mut cancellation = None;
            pool.in_place_scope(|scope| {
                for task in &mut tasks[..task_count] {
                    let dot_eight = &dot_eight;
                    let dot_four = &dot_four;
                    let dot = &dot;
                    let remaining = &remaining;
                    let cancelled = &cancelled;
                    let caller = &caller;
                    scope.spawn(move |_| {
                        let _completion = Completion { remaining, caller };
                        fill_tile_task(
                            &mut task.rows,
                            task.bytes,
                            row_bytes,
                            cancelled,
                            dot_eight,
                            dot_four,
                            dot,
                        );
                    });
                }
                wait_for_tile_tasks(&remaining, &cancelled, &mut cancellation, check);
            });
            for task in &tasks[..task_count] {
                *decoded_blocks = decoded_blocks.saturating_add(task.rows.decoded_blocks);
            }
            if let Some(error) = cancellation {
                return Err(error);
            }
            // All tasks joined. Sibling-major slots retain the first failing
            // sibling and then its earliest row chunk, regardless of completion.
            if let Some(error) = tasks.iter_mut().find_map(|task| task.rows.error.take()) {
                return Err(error);
            }
            return check();
        }
        for (bytes, group) in data.iter().zip(outputs.iter_mut()) {
            self.fill_tile_four_rows(
                bytes,
                row_bytes,
                group,
                decoded_blocks,
                check,
                &dot_eight,
                &dot_four,
                &dot,
            )?;
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "parallel-rows"))]
#[path = "parallel_tile_siblings_tests.rs"]
mod tests;
