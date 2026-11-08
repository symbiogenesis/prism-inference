//! Explicit, bounded row scheduling over borrowed immutable weight/activation slices.
//!
//! Independently implemented after reviewing `OxiBonsai`'s borrowed-slice row
//! partitioning at 1d10c48e14754ff6999158f8fa13c7c92b40b0dc,
//! crates/oxibonsai-kernels/src/parallel.rs (Apache-2.0, COOLJAPAN OU).
//! The original Prism row kernels and their arithmetic remain unchanged.

#[cfg(feature = "parallel-rows")]
use std::sync::Arc;

use crate::{Result, invalid};

#[path = "parallel_tile.rs"]
mod tile;

/// Largest explicit number of persistent projection workers.
pub const MAX_ROW_WORKERS: usize = 18;

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

    /// Updates contiguous row chunks of `target` in place across the private pool.
    ///
    /// Serial execution (`workers == 1` or `target.len() <= 1`) runs directly on
    /// the calling thread. Parallel execution partitions `target` into at most
    /// `workers` contiguous chunks and returns the earliest chunk failure after
    /// all workers join.
    ///
    /// # Errors
    /// Rejects unavailable `parallel-rows` support, error-slot allocation
    /// failure, or the first chunk failure returned by `update`.
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
    /// support, error-slot allocation failure, or the first chunk failure
    /// returned by `update`.
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
            use rayon::prelude::*;

            let task_units = units.div_ceil(self.workers);
            let task_elems = task_units * unit_size;
            let tasks = units.div_ceil(task_units);
            let mut errors = Vec::new();
            errors.try_reserve_exact(tasks)?;
            errors.resize_with(tasks, || None);
            pool.install(|| {
                target
                    .par_chunks_mut(task_elems)
                    .zip(errors.par_iter_mut())
                    .enumerate()
                    .for_each(|(task, (chunk, error))| {
                        if let Err(failure) = update(task * task_units, chunk) {
                            *error = Some(failure);
                        }
                    });
            });
            if let Some(failure) = errors.into_iter().flatten().next() {
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
    /// `parallel-rows` support, error-slot allocation failure, or the first chunk
    /// failure returned by `update`.
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
            use rayon::prelude::*;

            let task_units = units.div_ceil(self.workers);
            let first_task_elems = task_units * first_unit_size;
            let second_task_elems = task_units * second_unit_size;
            let tasks = units.div_ceil(task_units);
            let mut errors = Vec::new();
            errors.try_reserve_exact(tasks)?;
            errors.resize_with(tasks, || None);
            pool.install(|| {
                first
                    .par_chunks_mut(first_task_elems)
                    .zip(second.par_chunks_mut(second_task_elems))
                    .zip(errors.par_iter_mut())
                    .enumerate()
                    .for_each(|(task, ((first_chunk, second_chunk), error))| {
                        if let Err(failure) = update(task * task_units, first_chunk, second_chunk) {
                            *error = Some(failure);
                        }
                    });
            });
            if let Some(failure) = errors.into_iter().flatten().next() {
                return Err(failure);
            }
            return Ok(());
        }
        update(0, first, second)
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
            use rayon::prelude::*;

            let task_rows = rows.div_ceil(self.workers);
            let tasks = rows.div_ceil(task_rows);
            errors.try_reserve_exact(tasks)?;
            errors.resize_with(tasks, || None);
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
    #[allow(
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
            use rayon::prelude::*;

            let task_rows = rows.div_ceil(self.workers);
            let tasks = rows.div_ceil(task_rows);
            errors.try_reserve_exact(tasks)?;
            errors.resize_with(tasks, || None);
            pool.install(|| {
                output
                    .par_chunks_mut(task_rows)
                    .zip(errors.par_iter_mut())
                    .enumerate()
                    .for_each(|(task, (chunk, error))| {
                        let start = task * task_rows * row_bytes;
                        let bytes = &data[start..start + chunk.len() * row_bytes];
                        if let Err(failure) =
                            fill_batched_chunk(bytes, row_bytes, chunk, &dot_eight, &dot_four, &dot)
                        {
                            *error = Some(failure);
                        }
                    });
            });
            let failure = errors.iter_mut().find_map(Option::take);
            errors.clear();
            return failure.map_or(Ok(()), Err);
        }
        fill_batched_chunk(data, row_bytes, output, &dot_eight, &dot_four, &dot)
    }
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
fn fill_batched_chunk(
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
