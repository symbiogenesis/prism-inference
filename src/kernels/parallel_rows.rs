//! Explicit, bounded row scheduling over borrowed immutable weight/activation slices.
//!
//! Independently implemented after reviewing `OxiBonsai`'s borrowed-slice row
//! partitioning at 1d10c48e14754ff6999158f8fa13c7c92b40b0dc,
//! crates/oxibonsai-kernels/src/parallel.rs (Apache-2.0, COOLJAPAN OU).
//! The original Prism row kernels and their arithmetic remain unchanged.

#[cfg(feature = "parallel-rows")]
use std::sync::Arc;

#[cfg(feature = "parallel-rows")]
use bonsai_native_hash::row_pool::NativeRowPool;

use crate::{Result, invalid};

#[path = "parallel_tile.rs"]
mod tile;

#[path = "parallel_siblings.rs"]
mod siblings;

/// Largest explicit number of persistent projection workers.
pub const MAX_ROW_WORKERS: usize = 18;

/// Shared private row-worker pool, independent of global thread-pool state.
///
/// One worker retains serial execution and creates no threads. Larger budgets
/// require `parallel-rows`; clones share the same persistent pool. Each thread
/// has a requested 512 KiB stack, additionally charged by the process owner.
/// This does not select an ISA or establish target-device benefit.
/// In-place updates keep at most [`MAX_ROW_WORKERS`] optional boxed failures
/// inline on the calling thread's stack; successful dispatch needs no heap
/// allocation for those failure slots.
#[derive(Clone)]
pub struct RowParallelism {
    workers: usize,
    #[cfg(feature = "parallel-rows")]
    pool: Option<Arc<NativeRowPool>>,
}

#[cfg(all(test, feature = "parallel-rows"))]
#[path = "parallel_rows_scratch_tests.rs"]
mod scratch_tests;

#[cfg(all(test, feature = "parallel-rows"))]
mod tests {
    use std::sync::Arc;

    use bonsai_ternary::pack_ptq;

    use super::super::{MatvecBackend, MatvecScratch, matvec_into};
    use super::RowParallelism;
    use crate::{Result, invalid, model::Tensor};

    #[test]
    fn pinned_parallel_rows_preserve_both_activation_contracts() -> Result<()> {
        for workers in [2, 4, 18] {
            let policy = RowParallelism::new(workers)?;
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
                    data: Arc::<[u8]>::from(encoded.repeat(columns / 128 * 7)).into(),
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
                    assert_eq!(parallel.row_workers(), workers);
                }
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
        let caller = std::thread::current().id();
        let error = policy.fill(&data, 1, &mut output, &mut errors, |row| {
            assert_eq!(
                row.as_ptr(),
                data.as_ptr().wrapping_add(usize::from(row[0]))
            );
            let current = std::thread::current();
            assert!(
                current.id() == caller
                    || current
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

    #[test]
    fn update_rows_partitions_in_place_and_reports_earliest_chunk_failure() -> Result<()> {
        let policy = RowParallelism::new(4)?;
        let mut values = vec![1.0_f32; 10];
        policy.update_rows(&mut values, |base, chunk| {
            for (offset, target) in chunk.iter_mut().enumerate() {
                *target += f32::from(u8::try_from(base + offset)?);
            }
            Ok(())
        })?;
        let expected: Vec<f32> = (0_u8..10).map(|idx| 1.0 + f32::from(idx)).collect();
        assert_eq!(values, expected);
        let err = policy.update_rows(&mut values, |base, _chunk| {
            if base >= 3 {
                Err(invalid(format!("chunk-at-{base}")))
            } else {
                Ok(())
            }
        });
        assert_eq!(error_text(err), Err("chunk-at-3".to_owned()));
        let mut bytes = vec![0_u8; 24];
        policy.update_chunks(&mut bytes, 4, |base_unit, chunk| {
            for (unit_offset, unit) in chunk.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                let value = u8::try_from(base_unit + unit_offset)?;
                unit.fill(value);
            }
            Ok(())
        })?;
        for (unit, chunk) in bytes.as_chunks::<4>().0.iter().enumerate() {
            assert_eq!(chunk, &[u8::try_from(unit)?; 4]);
        }
        assert!(policy.update_chunks(&mut bytes, 0, |_, _| Ok(())).is_err());
        assert!(policy.update_chunks(&mut bytes, 5, |_, _| Ok(())).is_err());
        Ok(())
    }

    fn error_text(result: Result<()>) -> std::result::Result<(), String> {
        result.map_err(|error| error.to_string())
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
            return Err(invalid("row worker count must be from 1 through 18"));
        }
        if workers == 1 {
            return Ok(Self::default());
        }
        #[cfg(feature = "parallel-rows")]
        {
            let pool = NativeRowPool::new(workers)?;
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

    /// Maximum worker capacity of the backing private pool, or `1` when serial.
    #[cfg(feature = "parallel-rows")]
    #[must_use]
    pub fn pool_workers(&self) -> usize {
        if let Some(pool) = &self.pool {
            return pool.worker_count();
        }
        self.workers
    }

    /// Maximum worker capacity of the backing private pool, or `1` when serial.
    #[cfg(not(feature = "parallel-rows"))]
    #[must_use]
    pub const fn pool_workers(&self) -> usize {
        self.workers
    }

    /// Returns true when `self` and `other` share the same underlying worker pool instance.
    #[cfg(test)]
    #[cfg_attr(
        not(feature = "parallel-rows"),
        allow(
            clippy::missing_const_for_fn,
            reason = "Parallel builds compare Arc pointers at runtime."
        )
    )]
    pub(crate) fn shares_pool_with(&self, other: &Self) -> bool {
        #[cfg(feature = "parallel-rows")]
        match (&self.pool, &other.pool) {
            (Some(lhs), Some(rhs)) => Arc::ptr_eq(lhs, rhs),
            (None, None) => true,
            _ => false,
        }
        #[cfg(not(feature = "parallel-rows"))]
        {
            self.workers == 1 && other.workers == 1
        }
    }

    /// Derives a worker policy sharing this instance's underlying `NativeRowPool`
    /// with active concurrency capped to `1..=self.pool_workers()`.
    ///
    /// Selecting `1` retains serial execution without dispatching background threads.
    ///
    /// # Errors
    /// Rejects zero workers, counts exceeding the pool's spawned thread budget, or
    /// multi-worker counts when `parallel-rows` is disabled.
    pub fn with_active_workers(&self, workers: usize) -> Result<Self> {
        let cap = self.pool_workers();
        if workers == 0 || workers > cap {
            return Err(invalid(
                "active row worker count must be from 1 through pool capacity",
            ));
        }
        if workers == 1 {
            return Ok(Self::default());
        }
        #[cfg(feature = "parallel-rows")]
        if let Some(pool) = &self.pool {
            return Ok(Self {
                workers,
                pool: Some(Arc::clone(pool)),
            });
        }
        Err(invalid("multiple row workers require parallel-rows"))
    }

    /// Creates a `(decode, prefill)` pair of policies sharing one `NativeRowPool`
    /// sized to `max(decode_workers, prefill_workers)` (`<= MAX_ROW_WORKERS`).
    ///
    /// # Errors
    /// Rejects zero workers, counts above [`MAX_ROW_WORKERS`], unavailable
    /// `parallel-rows` support when either budget exceeds `1`, or thread creation failure.
    pub fn new_pair(decode_workers: usize, prefill_workers: usize) -> Result<(Self, Self)> {
        if !(1..=MAX_ROW_WORKERS).contains(&decode_workers)
            || !(1..=MAX_ROW_WORKERS).contains(&prefill_workers)
        {
            return Err(invalid("row worker count must be from 1 through 18"));
        }
        let pool_policy = Self::new(decode_workers.max(prefill_workers))?;
        let decode = pool_policy.with_active_workers(decode_workers)?;
        let prefill = pool_policy.with_active_workers(prefill_workers)?;
        Ok((decode, prefill))
    }

    /// Updates contiguous row chunks of `target` in place across the private pool.
    ///
    /// Serial execution (`workers == 1` or `target.len() <= 1`) runs directly on
    /// the calling thread. Parallel execution partitions `target` into at most
    /// `workers` contiguous chunks and returns the earliest chunk failure after
    /// all workers join.
    ///
    /// # Errors
    /// Rejects unavailable `parallel-rows` support or the first chunk failure
    /// returned by `update`.
    pub fn update_rows(
        &self,
        target: &mut [f32],
        update: impl Fn(usize, &mut [f32]) -> Result<()> + Sync,
    ) -> Result<()> {
        self.update_chunks(target, 1, update)
    }

    /// Updates contiguous fixed-unit chunks of `target` in place across the private pool.
    ///
    /// Each unit has `unit_size` elements, and `update(base_unit, chunk)` receives
    /// the starting unit index and a mutable slice whose length is a multiple of
    /// `unit_size`.
    ///
    /// # Errors
    /// Rejects a zero or non-dividing `unit_size`, unavailable `parallel-rows`
    /// support, or the first chunk failure returned by `update`.
    pub fn update_chunks<T: Send>(
        &self,
        target: &mut [T],
        unit_size: usize,
        update: impl Fn(usize, &mut [T]) -> Result<()> + Sync,
    ) -> Result<()> {
        if unit_size == 0 || !target.len().is_multiple_of(unit_size) {
            return Err(invalid(
                "target length must be a multiple of a positive unit size",
            ));
        }
        let units = target.len() / unit_size;
        #[cfg(not(feature = "parallel-rows"))]
        if self.uses_pool(units) {
            return Err(invalid("multiple row workers require parallel-rows"));
        }
        #[cfg(feature = "parallel-rows")]
        if let Some(pool) = &self.pool
            && units > 1
        {
            let task_units = units.div_ceil(self.workers);
            let task_elems = task_units * unit_size;
            // There is at most one task per configured worker. Keep chunk borrows
            // and cold failure slots on the calling thread's stack instead of
            // allocating a transient vector for every normalization/transform dispatch.
            let mut slots: [(&mut [T], Option<crate::Error>); MAX_ROW_WORKERS] =
                std::array::from_fn(|_| (&mut [][..], None));
            let mut count = 0;
            for (slot, chunk) in slots.iter_mut().zip(target.chunks_mut(task_elems)) {
                slot.0 = chunk;
                count += 1;
            }
            pool.dispatch_mut(&mut slots[..count], |task, (chunk, error)| {
                if let Err(failure) = update(task * task_units, chunk) {
                    *error = Some(failure);
                }
            });
            if let Some(failure) = slots[..count]
                .iter_mut()
                .find_map(|(_, error)| error.take())
            {
                return Err(failure);
            }
            return Ok(());
        }
        update(0, target)
    }

    /// Updates two aligned contiguous fixed-unit slices in lockstep across the private pool.
    ///
    /// # Errors
    /// Rejects zero or non-dividing unit sizes, unequal unit counts, unavailable
    /// `parallel-rows` support, or the first chunk failure returned by `update`.
    pub fn update_paired_chunks<A: Send, B: Send>(
        &self,
        first: &mut [A],
        first_unit_size: usize,
        second: &mut [B],
        second_unit_size: usize,
        update: impl Fn(usize, &mut [A], &mut [B]) -> Result<()> + Sync,
    ) -> Result<()> {
        if first_unit_size == 0
            || second_unit_size == 0
            || !first.len().is_multiple_of(first_unit_size)
            || !second.len().is_multiple_of(second_unit_size)
        {
            return Err(invalid(
                "paired target lengths must be multiples of positive unit sizes",
            ));
        }
        let units = first.len() / first_unit_size;
        if second.len() / second_unit_size != units {
            return Err(invalid("paired target unit counts must match"));
        }
        #[cfg(not(feature = "parallel-rows"))]
        if self.uses_pool(units) {
            return Err(invalid("multiple row workers require parallel-rows"));
        }
        #[cfg(feature = "parallel-rows")]
        if let Some(pool) = &self.pool
            && units > 1
        {
            let task_units = units.div_ceil(self.workers);
            let first_task_elems = task_units * first_unit_size;
            let second_task_elems = task_units * second_unit_size;
            let mut slots = std::array::from_fn::<_, MAX_ROW_WORKERS, _>(|_| {
                ((&mut [][..], &mut [][..]), None::<crate::Error>)
            });
            let mut count = 0;
            for (slot, (first_chunk, second_chunk)) in slots.iter_mut().zip(
                first
                    .chunks_mut(first_task_elems)
                    .zip(second.chunks_mut(second_task_elems)),
            ) {
                slot.0 = (first_chunk, second_chunk);
                count += 1;
            }
            pool.dispatch_mut(
                &mut slots[..count],
                |task, ((first_chunk, second_chunk), error)| {
                    if let Err(failure) = update(task * task_units, first_chunk, second_chunk) {
                        *error = Some(failure);
                    }
                },
            );
            if let Some(failure) = slots[..count]
                .iter_mut()
                .find_map(|(_, error)| error.take())
            {
                return Err(failure);
            }
            return Ok(());
        }
        update(0, first, second)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Triple-slice partition passes three slices, their unit sizes, and the task callback."
    )]
    pub(crate) fn update_triple_chunks<A: Send, B: Send, C: Send>(
        &self,
        first: &mut [A],
        first_unit_size: usize,
        second: &mut [B],
        second_unit_size: usize,
        third: &mut [C],
        third_unit_size: usize,
        update: impl Fn(usize, &mut [A], &mut [B], &mut [C]) -> Result<()> + Sync,
    ) -> Result<()> {
        if first_unit_size == 0
            || second_unit_size == 0
            || third_unit_size == 0
            || !first.len().is_multiple_of(first_unit_size)
            || !second.len().is_multiple_of(second_unit_size)
            || !third.len().is_multiple_of(third_unit_size)
        {
            return Err(invalid(
                "triple target lengths must be multiples of positive unit sizes",
            ));
        }
        let units = first.len() / first_unit_size;
        if second.len() / second_unit_size != units || third.len() / third_unit_size != units {
            return Err(invalid("triple target unit counts must match"));
        }
        #[cfg(not(feature = "parallel-rows"))]
        if self.uses_pool(units) {
            return Err(invalid("multiple row workers require parallel-rows"));
        }
        #[cfg(feature = "parallel-rows")]
        if let Some(pool) = &self.pool
            && units > 1
        {
            let task_units = units.div_ceil(self.workers);
            let first_task_elems = task_units * first_unit_size;
            let second_task_elems = task_units * second_unit_size;
            let third_task_elems = task_units * third_unit_size;
            let mut slots = std::array::from_fn::<_, MAX_ROW_WORKERS, _>(|_| {
                (
                    (&mut [][..], &mut [][..], &mut [][..]),
                    None::<crate::Error>,
                )
            });
            let mut count = 0;
            for (slot, ((first_chunk, second_chunk), third_chunk)) in slots.iter_mut().zip(
                first
                    .chunks_mut(first_task_elems)
                    .zip(second.chunks_mut(second_task_elems))
                    .zip(third.chunks_mut(third_task_elems)),
            ) {
                slot.0 = (first_chunk, second_chunk, third_chunk);
                count += 1;
            }
            pool.dispatch_mut(
                &mut slots[..count],
                |task, ((first_chunk, second_chunk, third_chunk), error)| {
                    if let Err(failure) =
                        update(task * task_units, first_chunk, second_chunk, third_chunk)
                    {
                        *error = Some(failure);
                    }
                },
            );
            if let Some(failure) = slots[..count]
                .iter_mut()
                .find_map(|(_, error)| error.take())
            {
                return Err(failure);
            }
            return Ok(());
        }
        update(0, first, second, third)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Dispatches five aligned per-head buffers under one pool barrier."
    )]
    pub(crate) fn update_quintuple_chunks<A: Send, B: Send, C: Send, D: Send, E: Send>(
        &self,
        first: &mut [A],
        first_unit_size: usize,
        second: &mut [B],
        second_unit_size: usize,
        third: &mut [C],
        third_unit_size: usize,
        fourth: &mut [D],
        fourth_unit_size: usize,
        fifth: &mut [E],
        fifth_unit_size: usize,
        update: impl Fn(usize, &mut [A], &mut [B], &mut [C], &mut [D], &mut [E]) -> Result<()> + Sync,
    ) -> Result<()> {
        if first_unit_size == 0
            || second_unit_size == 0
            || third_unit_size == 0
            || fourth_unit_size == 0
            || fifth_unit_size == 0
            || !first.len().is_multiple_of(first_unit_size)
            || !second.len().is_multiple_of(second_unit_size)
            || !third.len().is_multiple_of(third_unit_size)
            || !fourth.len().is_multiple_of(fourth_unit_size)
            || !fifth.len().is_multiple_of(fifth_unit_size)
        {
            return Err(invalid(
                "quintuple target lengths must be multiples of positive unit sizes",
            ));
        }
        let units = first.len() / first_unit_size;
        if second.len() / second_unit_size != units
            || third.len() / third_unit_size != units
            || fourth.len() / fourth_unit_size != units
            || fifth.len() / fifth_unit_size != units
        {
            return Err(invalid("quintuple target unit counts must match"));
        }
        #[cfg(not(feature = "parallel-rows"))]
        if self.uses_pool(units) {
            return Err(invalid("multiple row workers require parallel-rows"));
        }
        #[cfg(feature = "parallel-rows")]
        if let Some(pool) = &self.pool
            && units > 1
        {
            let task_units = units.div_ceil(self.workers);
            let first_task_elems = task_units * first_unit_size;
            let second_task_elems = task_units * second_unit_size;
            let third_task_elems = task_units * third_unit_size;
            let fourth_task_elems = task_units * fourth_unit_size;
            let fifth_task_elems = task_units * fifth_unit_size;
            let mut slots = std::array::from_fn::<_, MAX_ROW_WORKERS, _>(|_| {
                (
                    (
                        &mut [][..],
                        &mut [][..],
                        &mut [][..],
                        &mut [][..],
                        &mut [][..],
                    ),
                    None::<crate::Error>,
                )
            });
            let mut count = 0;
            for (slot, ((((first_chunk, second_chunk), third_chunk), fourth_chunk), fifth_chunk)) in
                slots.iter_mut().zip(
                    first
                        .chunks_mut(first_task_elems)
                        .zip(second.chunks_mut(second_task_elems))
                        .zip(third.chunks_mut(third_task_elems))
                        .zip(fourth.chunks_mut(fourth_task_elems))
                        .zip(fifth.chunks_mut(fifth_task_elems)),
                )
            {
                slot.0 = (
                    first_chunk,
                    second_chunk,
                    third_chunk,
                    fourth_chunk,
                    fifth_chunk,
                );
                count += 1;
            }
            pool.dispatch_mut(
                &mut slots[..count],
                |task,
                 ((first_chunk, second_chunk, third_chunk, fourth_chunk, fifth_chunk), error)| {
                    if let Err(failure) = update(
                        task * task_units,
                        first_chunk,
                        second_chunk,
                        third_chunk,
                        fourth_chunk,
                        fifth_chunk,
                    ) {
                        *error = Some(failure);
                    }
                },
            );
            if let Some(failure) = slots[..count]
                .iter_mut()
                .find_map(|(_, error)| error.take())
            {
                return Err(failure);
            }
            return Ok(());
        }
        update(0, first, second, third, fourth, fifth)
    }

    pub(super) const fn uses_pool(&self, rows: usize) -> bool {
        self.workers > 1 && rows > 1
    }

    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    #[expect(
        clippy::too_many_arguments,
        reason = "Preserves distinct typed 16/8/4/1 row closures without dynamic dispatch."
    )]
    pub(super) fn fill_sixteen_rows(
        &self,
        data: &[u8],
        row_bytes: usize,
        output: &mut Vec<f32>,
        errors: &mut Vec<Option<crate::Error>>,
        dot_sixteen: impl Fn([&[u8]; 16]) -> Result<[f32; 16]> + Sync,
        dot_eight: impl Fn([&[u8]; 8]) -> Result<[f32; 8]> + Sync,
        dot_four: impl Fn([&[u8]; 4]) -> Result<[f32; 4]> + Sync,
        dot: impl Fn(&[u8]) -> Result<f32> + Sync,
    ) -> Result<()> {
        errors.clear();
        let rows = data.len() / row_bytes;
        #[cfg(not(feature = "parallel-rows"))]
        if self.uses_pool(rows) {
            return Err(invalid("multiple row workers require parallel-rows"));
        }
        if output.len() != rows {
            output.clear();
            output.try_reserve_exact(rows)?;
            output.resize(rows, 0.0);
        }
        #[cfg(feature = "parallel-rows")]
        if let Some(pool) = &self.pool
            && rows > 1
        {
            return dispatch_output_chunks(
                pool,
                self.workers,
                data,
                row_bytes,
                output,
                |bytes, chunk| {
                    fill_sixteen_batched_chunk(
                        chunk,
                        bytes,
                        row_bytes,
                        &dot_sixteen,
                        &dot_eight,
                        &dot_four,
                        &dot,
                    )
                },
            );
        }
        fill_sixteen_batched_chunk(
            output,
            data,
            row_bytes,
            &dot_sixteen,
            &dot_eight,
            &dot_four,
            &dot,
        )
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
        let rows = data.len() / row_bytes;
        #[cfg(not(feature = "parallel-rows"))]
        if self.uses_pool(rows) {
            return Err(invalid("multiple row workers require parallel-rows"));
        }
        if output.len() != rows {
            output.clear();
            output.try_reserve_exact(rows)?;
            output.resize(rows, 0.0);
        }
        #[cfg(feature = "parallel-rows")]
        if let Some(pool) = &self.pool
            && rows > 1
        {
            return dispatch_output_chunks(
                pool,
                self.workers,
                data,
                row_bytes,
                output,
                |bytes, chunk| {
                    for (value, row) in chunk.iter_mut().zip(bytes.chunks_exact(row_bytes)) {
                        *value = dot(row)?;
                    }
                    Ok(())
                },
            );
        }
        for (value, bytes) in output.iter_mut().zip(data.chunks_exact(row_bytes)) {
            *value = dot(bytes)?;
        }
        Ok(())
    }

    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    #[expect(
        clippy::too_many_arguments,
        reason = "Batch eight-row, four-row, and single-row kernels without allocating a wrapper."
    )]
    pub(super) fn fill_four_rows(
        &self,
        data: &[u8],
        row_bytes: usize,
        output: &mut Vec<f32>,
        errors: &mut Vec<Option<crate::Error>>,
        dot_eight: impl Fn([&[u8]; 8]) -> Result<[f32; 8]> + Sync,
        dot_four: impl Fn([&[u8]; 4]) -> Result<[f32; 4]> + Sync,
        dot: impl Fn(&[u8]) -> Result<f32> + Sync,
    ) -> Result<()> {
        errors.clear();
        let rows = data.len() / row_bytes;
        #[cfg(not(feature = "parallel-rows"))]
        if self.uses_pool(rows) {
            return Err(invalid("multiple row workers require parallel-rows"));
        }
        if output.len() != rows {
            output.clear();
            output.try_reserve_exact(rows)?;
            output.resize(rows, 0.0);
        }
        #[cfg(feature = "parallel-rows")]
        if let Some(pool) = &self.pool
            && rows > 1
        {
            return dispatch_output_chunks(
                pool,
                self.workers,
                data,
                row_bytes,
                output,
                |bytes, chunk| {
                    fill_batched_chunk(bytes, row_bytes, chunk, &dot_eight, &dot_four, &dot)
                },
            );
        }
        fill_batched_chunk(data, row_bytes, output, &dot_eight, &dot_four, &dot)
    }
}

#[cfg(feature = "parallel-rows")]
const CHUNK_WORK_FACTOR: usize = 4;

#[cfg(feature = "parallel-rows")]
const DECODE_ROW_ALIGN: usize = 16;

#[cfg(feature = "parallel-rows")]
pub(super) const fn aligned_chunk_rows(rows: usize, workers: usize, align: usize) -> usize {
    if workers <= 1 || rows <= 1 {
        return if rows == 0 { 1 } else { rows };
    }
    if align > 1 && rows >= workers * align {
        let target_chunks = workers * CHUNK_WORK_FACTOR;
        rows.div_ceil(target_chunks).div_ceil(align) * align
    } else {
        rows.div_ceil(workers)
    }
}

#[cfg(feature = "parallel-rows")]
fn dispatch_output_chunks(
    pool: &NativeRowPool,
    workers: usize,
    data: &[u8],
    row_bytes: usize,
    output: &mut [f32],
    run_chunk: impl Fn(&[u8], &mut [f32]) -> Result<()> + Sync,
) -> Result<()> {
    let rows = output.len();
    let task_rows = aligned_chunk_rows(rows, workers, DECODE_ROW_ALIGN);
    let mut slots: [(&mut [f32], Option<crate::Error>); CHUNK_WORK_FACTOR * MAX_ROW_WORKERS] =
        std::array::from_fn(|_| (&mut [][..], None));
    let mut count = 0;
    for (slot, chunk) in slots.iter_mut().zip(output.chunks_mut(task_rows)) {
        slot.0 = chunk;
        count += 1;
    }
    pool.dispatch_mut(&mut slots[..count], |task, (chunk, error)| {
        let start = task * task_rows * row_bytes;
        let bytes = &data[start..start + chunk.len() * row_bytes];
        if let Err(failure) = run_chunk(bytes, chunk) {
            *error = Some(failure);
        }
    });
    slots[..count]
        .iter_mut()
        .find_map(|(_, error)| error.take())
        .map_or(Ok(()), Err)
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
pub(super) fn fill_batched_chunk(
    bytes: &[u8],
    row_bytes: usize,
    chunk: &mut [f32],
    dot_eight: &impl Fn([&[u8]; 8]) -> Result<[f32; 8]>,
    dot_four: &impl Fn([&[u8]; 4]) -> Result<[f32; 4]>,
    dot: &impl Fn(&[u8]) -> Result<f32>,
) -> Result<()> {
    let (octets, after_octets) = chunk.as_chunks_mut::<8>();
    let (octet_bytes, after_octet_bytes) = bytes.split_at(octets.len() * 8 * row_bytes);
    for (out_octet, group) in octets
        .iter_mut()
        .zip(octet_bytes.chunks_exact(8 * row_bytes))
    {
        let rows: [&[u8]; 8] =
            std::array::from_fn(|idx| &group[idx * row_bytes..(idx + 1) * row_bytes]);
        if let Ok(values) = dot_eight(rows) {
            *out_octet = values;
        } else {
            for (target, row) in out_octet.iter_mut().zip(rows) {
                *target = dot(row)?;
            }
        }
    }
    let (quads, tail_out) = after_octets.as_chunks_mut::<4>();
    let (quad_bytes, tail_bytes) = after_octet_bytes.split_at(quads.len() * 4 * row_bytes);
    for (out_quad, group) in quads.iter_mut().zip(quad_bytes.chunks_exact(4 * row_bytes)) {
        let rows: [&[u8]; 4] =
            std::array::from_fn(|idx| &group[idx * row_bytes..(idx + 1) * row_bytes]);
        if let Ok(values) = dot_four(rows) {
            *out_quad = values;
        } else {
            for (target, row) in out_quad.iter_mut().zip(rows) {
                *target = dot(row)?;
            }
        }
    }
    for (value, row) in tail_out.iter_mut().zip(tail_bytes.chunks_exact(row_bytes)) {
        *value = dot(row)?;
    }
    Ok(())
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
pub(super) fn fill_sixteen_batched_chunk(
    chunk: &mut [f32],
    bytes: &[u8],
    row_bytes: usize,
    dot_sixteen: &(impl Fn([&[u8]; 16]) -> Result<[f32; 16]> + Sync),
    dot_eight: &(impl Fn([&[u8]; 8]) -> Result<[f32; 8]> + Sync),
    dot_four: &(impl Fn([&[u8]; 4]) -> Result<[f32; 4]> + Sync),
    dot: &(impl Fn(&[u8]) -> Result<f32> + Sync),
) -> Result<()> {
    let (sixteens, after_sixteens) = chunk.as_chunks_mut::<16>();
    let sixteens_len = sixteens.len() * 16 * row_bytes;
    let (sixteen_bytes, rest_bytes) = bytes.split_at(sixteens_len);
    for (out_sixteen, group) in sixteens
        .iter_mut()
        .zip(sixteen_bytes.chunks_exact(16 * row_bytes))
    {
        let rows: [&[u8]; 16] =
            std::array::from_fn(|row| &group[row * row_bytes..(row + 1) * row_bytes]);
        *out_sixteen = dot_sixteen(rows)?;
    }
    fill_batched_chunk(
        rest_bytes,
        row_bytes,
        after_sixteens,
        dot_eight,
        dot_four,
        dot,
    )
}

#[cfg(test)]
mod pair_tests {
    use super::super::MatvecScratch;
    use super::{MAX_ROW_WORKERS, RowParallelism};
    use crate::Result;

    #[test]
    fn row_parallelism_pair_and_matvec_scratch_prefill_defaults() -> Result<()> {
        let (serial_decode, serial_prefill) = RowParallelism::new_pair(1, 1)?;
        assert_eq!(serial_decode.workers(), 1);
        assert_eq!(serial_prefill.workers(), 1);
        assert!(serial_decode.shares_pool_with(&serial_prefill));
        assert!(RowParallelism::new_pair(0, 1).is_err());
        assert!(RowParallelism::new_pair(1, MAX_ROW_WORKERS + 1).is_err());
        assert!(serial_decode.with_active_workers(0).is_err());
        assert!(serial_decode.with_active_workers(2).is_err());

        #[cfg_attr(
            not(feature = "parallel-rows"),
            allow(
                unused_mut,
                reason = "Mutated in the parallel-rows feature branch below."
            )
        )]
        let mut scratch = MatvecScratch::default();
        assert_eq!(scratch.row_workers(), 1);
        assert_eq!(scratch.prefill_row_workers(), 1);
        assert_eq!(scratch.prefill_row_parallelism().workers(), 1);

        #[cfg(feature = "parallel-rows")]
        {
            let (decode, prefill) = RowParallelism::new_pair(8, 16)?;
            assert_eq!(decode.workers(), 8);
            assert_eq!(decode.pool_workers(), 16);
            assert_eq!(prefill.workers(), 16);
            assert_eq!(prefill.pool_workers(), 16);
            assert!(decode.shares_pool_with(&prefill));
            assert!(decode.with_active_workers(17).is_err());

            let sub = decode.with_active_workers(12)?;
            assert_eq!(sub.workers(), 12);
            assert!(sub.shares_pool_with(&prefill));
            assert_eq!(decode.with_active_workers(1)?.workers(), 1);

            let (serial_dec, shared_pref) = RowParallelism::new_pair(1, 4)?;
            assert_eq!(serial_dec.workers(), 1);
            assert_eq!(shared_pref.workers(), 4);

            scratch.set_row_parallelism(decode);
            assert_eq!(scratch.row_workers(), 8);
            assert_eq!(scratch.prefill_row_workers(), 8);
            assert_eq!(scratch.prefill_row_parallelism().workers(), 8);

            scratch.set_prefill_row_parallelism(prefill.clone());
            assert_eq!(scratch.row_workers(), 8);
            assert_eq!(scratch.prefill_row_workers(), 16);
            assert!(
                scratch
                    .clone_prefill_row_parallelism()
                    .shares_pool_with(&prefill)
            );
        }
        Ok(())
    }
}
