//! Joined tile-row scheduling with cancellation retained on the calling thread.

use super::RowParallelism;
use crate::{Result, invalid};

#[cfg(feature = "parallel-rows")]
use super::MAX_ROW_WORKERS;

#[cfg(feature = "parallel-rows")]
struct Task<'a, const TOKENS: usize> {
    outputs: [&'a mut [f32]; TOKENS],
    error: Option<crate::Error>,
    decoded_blocks: u64,
}

#[cfg(feature = "parallel-rows")]
struct Completion<'a> {
    remaining: &'a std::sync::atomic::AtomicUsize,
    caller: &'a std::thread::Thread,
}

#[cfg(feature = "parallel-rows")]
impl Drop for Completion<'_> {
    fn drop(&mut self) {
        self.remaining
            .fetch_sub(1, std::sync::atomic::Ordering::Release);
        self.caller.unpark();
    }
}

impl RowParallelism {
    /// Fill disjoint token-major row slices without duplicating weight decoding.
    /// The callback stays on its calling thread and needs neither `Send` nor `Sync`.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "Keep scoped borrows, joined cancellation and bounded stack task storage together."
    )]
    pub(in crate::kernels) fn fill_tile<const TOKENS: usize>(
        &self,
        data: &[u8],
        row_bytes: usize,
        outputs: &mut [Vec<f32>],
        decoded_blocks: &mut u64,
        check: &dyn Fn() -> Result<()>,
        dot: impl Fn(&[u8], &mut u64) -> Result<[f32; TOKENS]> + Sync,
    ) -> Result<()> {
        if outputs.is_empty()
            || outputs.len() > TOKENS
            || row_bytes == 0
            || !data.len().is_multiple_of(row_bytes)
        {
            return Err(invalid("invalid tile row geometry"));
        }
        let rows = data.len() / row_bytes;
        #[cfg(not(feature = "parallel-rows"))]
        if self.uses_pool(rows) {
            return Err(invalid("multiple row workers require parallel-rows"));
        }
        #[cfg(feature = "parallel-rows")]
        if let Some(pool) = &self.pool
            && self.uses_pool(rows)
        {
            use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
            use std::time::Duration;

            let task_rows = rows.div_ceil(self.workers());
            let task_count = rows.div_ceil(task_rows);
            // At most 16 * 32 mutable slice descriptors and one error/count per
            // worker, all stack-scoped. Existing output capacities were reserved
            // fallibly by the caller; no second matrix or result transpose exists.
            let mut tasks: [Task<'_, TOKENS>; MAX_ROW_WORKERS] = std::array::from_fn(|_| Task {
                outputs: std::array::from_fn(|_| -> &mut [f32] { &mut [] }),
                error: None,
                decoded_blocks: 0,
            });
            for (token, output) in outputs.iter_mut().enumerate() {
                output.resize(rows, 0.0);
                for (task, chunk) in tasks.iter_mut().zip(output.chunks_mut(task_rows)) {
                    task.outputs[token] = chunk;
                }
            }
            let caller = std::thread::current();
            let remaining = AtomicUsize::new(task_count);
            let cancelled = AtomicBool::new(false);
            let mut cancellation = None;
            // Unlike install/scope, in_place_scope keeps this closure on the
            // caller. Only row kernels and disjoint output slices cross threads.
            pool.in_place_scope(|scope| {
                for (index, task) in tasks[..task_count].iter_mut().enumerate() {
                    let start = index * task_rows * row_bytes;
                    let bytes = &data[start..start + task.outputs[0].len() * row_bytes];
                    let dot = &dot;
                    let remaining = &remaining;
                    let cancelled = &cancelled;
                    let caller = &caller;
                    scope.spawn(move |_| {
                        // Always notify on unwind too; the scope propagates a
                        // worker panic after joining instead of stranding a wait.
                        let _completion = Completion { remaining, caller };
                        for (row, bytes) in bytes.chunks_exact(row_bytes).enumerate() {
                            if cancelled.load(Ordering::Acquire) {
                                break;
                            }
                            match dot(bytes, &mut task.decoded_blocks) {
                                Ok(values) => {
                                    for (output, value) in task.outputs.iter_mut().zip(values) {
                                        if !output.is_empty() {
                                            output[row] = value;
                                        }
                                    }
                                }
                                Err(error) => {
                                    task.error = Some(error);
                                    break;
                                }
                            }
                        }
                    });
                }
                while remaining.load(Ordering::Acquire) != 0 {
                    if cancellation.is_none()
                        && let Err(error) = check()
                    {
                        cancellation = Some(error);
                        cancelled.store(true, Ordering::Release);
                    }
                    std::thread::park_timeout(Duration::from_millis(1));
                }
            });
            for task in &tasks[..task_count] {
                *decoded_blocks = decoded_blocks.saturating_add(task.decoded_blocks);
            }
            // Every row task has joined. Ordered task slots retain the earliest
            // failing row chunk independently of completion order.
            if let Some(error) = cancellation {
                return Err(error);
            }
            if let Some(error) = tasks.iter_mut().find_map(|task| task.error.take()) {
                return Err(error);
            }
            return check();
        }
        for row in data.chunks_exact(row_bytes) {
            check()?;
            let values = dot(row, decoded_blocks)?;
            for (output, value) in outputs.iter_mut().zip(values) {
                output.push(value);
            }
        }
        check()
    }

    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "Keep scoped borrows, eight/four-row batching, joined cancellation and bounded stack task storage together."
    )]
    pub(in crate::kernels) fn fill_tile_four_rows<const TOKENS: usize>(
        &self,
        data: &[u8],
        row_bytes: usize,
        outputs: &mut [Vec<f32>],
        decoded_blocks: &mut u64,
        check: &dyn Fn() -> Result<()>,
        dot_eight: impl Fn([&[u8]; 8], &mut u64) -> Result<[[f32; TOKENS]; 8]> + Sync,
        dot_four: impl Fn([&[u8]; 4], &mut u64) -> Result<[[f32; TOKENS]; 4]> + Sync,
        dot: impl Fn(&[u8], &mut u64) -> Result<[f32; TOKENS]> + Sync,
    ) -> Result<()> {
        if outputs.is_empty()
            || outputs.len() > TOKENS
            || row_bytes == 0
            || !data.len().is_multiple_of(row_bytes)
        {
            return Err(invalid("invalid tile row geometry"));
        }
        let rows = data.len() / row_bytes;
        #[cfg(not(feature = "parallel-rows"))]
        if self.uses_pool(rows) {
            return Err(invalid("multiple row workers require parallel-rows"));
        }
        #[cfg(feature = "parallel-rows")]
        if let Some(pool) = &self.pool
            && self.uses_pool(rows)
        {
            use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
            use std::time::Duration;

            let task_rows = rows.div_ceil(self.workers());
            let task_count = rows.div_ceil(task_rows);
            let mut tasks: [Task<'_, TOKENS>; MAX_ROW_WORKERS] = std::array::from_fn(|_| Task {
                outputs: std::array::from_fn(|_| -> &mut [f32] { &mut [] }),
                error: None,
                decoded_blocks: 0,
            });
            for (token, output) in outputs.iter_mut().enumerate() {
                output.resize(rows, 0.0);
                for (task, chunk) in tasks.iter_mut().zip(output.chunks_mut(task_rows)) {
                    task.outputs[token] = chunk;
                }
            }
            let caller = std::thread::current();
            let remaining = AtomicUsize::new(task_count);
            let cancelled = AtomicBool::new(false);
            let mut cancellation = None;
            pool.in_place_scope(|scope| {
                for (index, task) in tasks[..task_count].iter_mut().enumerate() {
                    let start = index * task_rows * row_bytes;
                    let bytes = &data[start..start + task.outputs[0].len() * row_bytes];
                    let dot_eight = &dot_eight;
                    let dot_four = &dot_four;
                    let dot = &dot;
                    let remaining = &remaining;
                    let cancelled = &cancelled;
                    let caller = &caller;
                    scope.spawn(move |_| {
                        let _completion = Completion { remaining, caller };
                        let mut row = 0;
                        let mut chunks8 = bytes.chunks_exact(row_bytes * 8);
                        for chunk in chunks8.by_ref() {
                            if cancelled.load(Ordering::Acquire) {
                                break;
                            }
                            let rows8: [&[u8]; 8] = std::array::from_fn(|idx| {
                                &chunk[idx * row_bytes..(idx + 1) * row_bytes]
                            });
                            let before_blocks = task.decoded_blocks;
                            if let Ok(out8) = dot_eight(rows8, &mut task.decoded_blocks) {
                                for (tok, output) in task.outputs.iter_mut().enumerate() {
                                    if !output.is_empty() {
                                        for r in 0..8 {
                                            output[row + r] = out8[r][tok];
                                        }
                                    }
                                }
                                row += 8;
                            } else {
                                task.decoded_blocks = before_blocks;
                                for single in rows8 {
                                    if cancelled.load(Ordering::Acquire) {
                                        break;
                                    }
                                    match dot(single, &mut task.decoded_blocks) {
                                        Ok(values) => {
                                            for (output, value) in
                                                task.outputs.iter_mut().zip(values)
                                            {
                                                if !output.is_empty() {
                                                    output[row] = value;
                                                }
                                            }
                                            row += 1;
                                        }
                                        Err(error) => {
                                            task.error = Some(error);
                                            break;
                                        }
                                    }
                                }
                                if task.error.is_some() {
                                    break;
                                }
                            }
                        }
                        let mut chunks4 = chunks8.remainder().chunks_exact(row_bytes * 4);
                        if task.error.is_none() && !cancelled.load(Ordering::Acquire) {
                            for chunk in chunks4.by_ref() {
                                if cancelled.load(Ordering::Acquire) {
                                    break;
                                }
                                let r0 = &chunk[..row_bytes];
                                let r1 = &chunk[row_bytes..row_bytes * 2];
                                let r2 = &chunk[row_bytes * 2..row_bytes * 3];
                                let r3 = &chunk[row_bytes * 3..];
                                let before_blocks = task.decoded_blocks;
                                if let Ok(rows4) =
                                    dot_four([r0, r1, r2, r3], &mut task.decoded_blocks)
                                {
                                    for (tok, output) in task.outputs.iter_mut().enumerate() {
                                        if !output.is_empty() {
                                            output[row] = rows4[0][tok];
                                            output[row + 1] = rows4[1][tok];
                                            output[row + 2] = rows4[2][tok];
                                            output[row + 3] = rows4[3][tok];
                                        }
                                    }
                                    row += 4;
                                } else {
                                    task.decoded_blocks = before_blocks;
                                    for single in [r0, r1, r2, r3] {
                                        if cancelled.load(Ordering::Acquire) {
                                            break;
                                        }
                                        match dot(single, &mut task.decoded_blocks) {
                                            Ok(values) => {
                                                for (output, value) in
                                                    task.outputs.iter_mut().zip(values)
                                                {
                                                    if !output.is_empty() {
                                                        output[row] = value;
                                                    }
                                                }
                                                row += 1;
                                            }
                                            Err(error) => {
                                                task.error = Some(error);
                                                break;
                                            }
                                        }
                                    }
                                    if task.error.is_some() {
                                        break;
                                    }
                                }
                            }
                        }
                        if task.error.is_none() && !cancelled.load(Ordering::Acquire) {
                            for single in chunks4.remainder().chunks_exact(row_bytes) {
                                if cancelled.load(Ordering::Acquire) {
                                    break;
                                }
                                match dot(single, &mut task.decoded_blocks) {
                                    Ok(values) => {
                                        for (output, value) in task.outputs.iter_mut().zip(values) {
                                            if !output.is_empty() {
                                                output[row] = value;
                                            }
                                        }
                                        row += 1;
                                    }
                                    Err(error) => {
                                        task.error = Some(error);
                                        break;
                                    }
                                }
                            }
                        }
                    });
                }
                while remaining.load(Ordering::Acquire) != 0 {
                    if cancellation.is_none()
                        && let Err(error) = check()
                    {
                        cancellation = Some(error);
                        cancelled.store(true, Ordering::Release);
                    }
                    std::thread::park_timeout(Duration::from_millis(1));
                }
            });
            for task in &tasks[..task_count] {
                *decoded_blocks = decoded_blocks.saturating_add(task.decoded_blocks);
            }
            if let Some(error) = cancellation {
                return Err(error);
            }
            if let Some(error) = tasks.iter_mut().find_map(|task| task.error.take()) {
                return Err(error);
            }
            return check();
        }
        let mut chunks8 = data.chunks_exact(row_bytes * 8);
        for chunk in chunks8.by_ref() {
            let rows8: [&[u8]; 8] =
                std::array::from_fn(|idx| &chunk[idx * row_bytes..(idx + 1) * row_bytes]);
            let before_blocks = *decoded_blocks;
            for completed in 0..8 {
                if let Err(error) = check() {
                    for single in &rows8[..completed] {
                        let values = dot(single, decoded_blocks)?;
                        for (output, value) in outputs.iter_mut().zip(values) {
                            output.push(value);
                        }
                    }
                    return Err(error);
                }
            }
            if let Ok(out8) = dot_eight(rows8, decoded_blocks) {
                for (tok, output) in outputs.iter_mut().enumerate() {
                    output.extend_from_slice(&[
                        out8[0][tok],
                        out8[1][tok],
                        out8[2][tok],
                        out8[3][tok],
                        out8[4][tok],
                        out8[5][tok],
                        out8[6][tok],
                        out8[7][tok],
                    ]);
                }
            } else {
                *decoded_blocks = before_blocks;
                for single in rows8 {
                    let values = dot(single, decoded_blocks)?;
                    for (output, value) in outputs.iter_mut().zip(values) {
                        output.push(value);
                    }
                }
            }
        }
        let mut chunks4 = chunks8.remainder().chunks_exact(row_bytes * 4);
        for chunk in chunks4.by_ref() {
            let r0 = &chunk[..row_bytes];
            let r1 = &chunk[row_bytes..row_bytes * 2];
            let r2 = &chunk[row_bytes * 2..row_bytes * 3];
            let r3 = &chunk[row_bytes * 3..];
            let before_blocks = *decoded_blocks;
            for completed in 0..4 {
                if let Err(error) = check() {
                    for single in &[r0, r1, r2, r3][..completed] {
                        let values = dot(single, decoded_blocks)?;
                        for (output, value) in outputs.iter_mut().zip(values) {
                            output.push(value);
                        }
                    }
                    return Err(error);
                }
            }
            if let Ok(rows4) = dot_four([r0, r1, r2, r3], decoded_blocks) {
                for (tok, output) in outputs.iter_mut().enumerate() {
                    output.extend_from_slice(&[
                        rows4[0][tok],
                        rows4[1][tok],
                        rows4[2][tok],
                        rows4[3][tok],
                    ]);
                }
            } else {
                *decoded_blocks = before_blocks;
                for single in [r0, r1, r2, r3] {
                    let values = dot(single, decoded_blocks)?;
                    for (output, value) in outputs.iter_mut().zip(values) {
                        output.push(value);
                    }
                }
            }
        }
        for row in chunks4.remainder().chunks_exact(row_bytes) {
            check()?;
            let values = dot(row, decoded_blocks)?;
            for (output, value) in outputs.iter_mut().zip(values) {
                output.push(value);
            }
        }
        check()
    }
}

#[cfg(all(test, feature = "parallel-rows"))]
mod tests {
    use std::cell::Cell;
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::{MAX_ROW_WORKERS, RowParallelism, Task};
    use crate::{Result, invalid};

    #[test]
    fn tiled_parallel_rows_borrow_outputs_and_keep_callback_on_caller() -> Result<()> {
        assert!(
            size_of::<[Task<'_, 32>; MAX_ROW_WORKERS]>() + 1024
                <= crate::kernels::prefill::ROW_TASK_SCRATCH_BYTES
        );
        let policy = RowParallelism::new(4)?;
        let data = [0_u8, 1, 2, 3, 4, 5, 6];
        let mut outputs: Vec<_> = (0..3).map(|_| Vec::with_capacity(data.len())).collect();
        let caller = std::thread::current().id();
        let checked = AtomicBool::new(false);
        let checks = Cell::new(0);
        let mut blocks = 0;
        policy.fill_tile(
            &data,
            1,
            &mut outputs,
            &mut blocks,
            &|| {
                assert_eq!(std::thread::current().id(), caller);
                checks.set(checks.get() + 1);
                checked.store(true, Ordering::Release);
                Ok(())
            },
            |row, blocks| {
                assert_eq!(
                    row.as_ptr(),
                    data.as_ptr().wrapping_add(usize::from(row[0]))
                );
                assert!(
                    std::thread::current()
                        .name()
                        .is_some_and(|name| name.starts_with("prism-row-"))
                );
                let wait = std::time::Instant::now();
                while !checked.load(Ordering::Acquire) {
                    assert!(wait.elapsed() < std::time::Duration::from_secs(2));
                    std::thread::yield_now();
                }
                *blocks += 1;
                let value = f32::from(row[0]);
                Ok([value, -value, value * 0.5])
            },
        )?;
        assert_eq!(blocks, 7);
        assert!(checks.get() >= 2);
        assert_eq!(outputs[0], data.map(f32::from));
        assert_eq!(outputs[1], data.map(|value| -f32::from(value)));
        assert_eq!(outputs[2], data.map(|value| f32::from(value) * 0.5));
        Ok(())
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "Keep ordered numerical failure, joined callback cancellation and replay in one control."
    )]
    fn tiled_parallel_rows_join_cancellation_and_select_first_row_failure() -> Result<()> {
        let policy = RowParallelism::new(2)?;
        let data = [0_u8, 1, 2, 3, 4, 5, 6, 7];
        let mut outputs = vec![Vec::with_capacity(data.len()); 1];
        let mut blocks = 0;
        let failure = policy.fill_tile(
            &data,
            1,
            &mut outputs,
            &mut blocks,
            &|| Ok(()),
            |row, blocks| {
                *blocks += 1;
                if row[0] == 3 || row[0] == 7 {
                    Err(invalid(format!("failed tile row {}", row[0])))
                } else {
                    Ok([f32::from(row[0])])
                }
            },
        );
        assert_eq!(
            failure.map_err(|error| error.to_string()),
            Err("failed tile row 3".to_owned())
        );
        assert_eq!(blocks, 8);

        let started = AtomicBool::new(false);
        let cancelled = AtomicBool::new(false);
        let checks = Cell::new(0);
        let failure = policy.fill_tile::<1>(
            &data,
            1,
            &mut outputs,
            &mut blocks,
            &|| {
                checks.set(checks.get() + 1);
                if started.load(Ordering::Acquire) {
                    cancelled.store(true, Ordering::Release);
                    Err(invalid("caller cancelled tile"))
                } else {
                    Ok(())
                }
            },
            |_, blocks| {
                started.store(true, Ordering::Release);
                let wait = std::time::Instant::now();
                while !cancelled.load(Ordering::Acquire) {
                    assert!(wait.elapsed() < std::time::Duration::from_secs(2));
                    std::thread::yield_now();
                }
                *blocks += 1;
                Err(invalid("provisional worker failure"))
            },
        );
        assert_eq!(
            failure.map_err(|error| error.to_string()),
            Err("caller cancelled tile".to_owned())
        );
        assert!(checks.get() >= 1);
        // After returning, every task has joined and subsequent use is safe.
        outputs[0].clear();
        policy.fill_tile(&data, 1, &mut outputs, &mut blocks, &|| Ok(()), |row, _| {
            Ok([f32::from(row[0])])
        })?;
        assert_eq!(outputs[0], data.map(f32::from));
        Ok(())
    }

    #[test]
    #[allow(
        clippy::panic,
        reason = "Inject a worker unwind to verify its completion guard and joined scope."
    )]
    fn tiled_parallel_rows_worker_panic_cannot_strand_caller() -> Result<()> {
        let policy = RowParallelism::new(2)?;
        let mut outputs = vec![Vec::with_capacity(2); 1];
        let mut blocks = 0;
        let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            policy.fill_tile::<1>(&[0, 1], 1, &mut outputs, &mut blocks, &|| Ok(()), |_, _| {
                panic!("injected tile worker unwind");
            })
        }));
        assert!(failure.is_err());
        Ok(())
    }
}
