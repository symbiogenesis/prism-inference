//! Full-`K` 1-op native `HTP_TYPE_Q2_0` projection executor over `libggml-htp-v81.so`.
//!
//! Losslessly transcodes Swift's `PQ2_0` (`34` bytes per 128 weights) into
//! `llama.cpp`'s native tiled `HTP_TYPE_Q2_0` (`576` bytes per 32x64 tile,
//! repeating each 128-weight `FP16` scale across both 64-weight sub-tiles) and
//! executes the full `K <= 5120`, `M <= 17408` projection in a single DSP
//! operation (`HTP_MM_KERNEL_HVX_QUANT_ROW_N1`).

use std::sync::Arc;

use bonsai_prism_runtime::{
    kernels::{Pq2ProjectionExecutor, RowParallelism, is_finite_slice},
    model::Tensor,
};
use bonsai_ternary::Q8KBlock;
use hexagon_native_rpc::{NativeQ2ProjectionLimits, NativeQ2Submitter, NativeQ2WeightHandle};

use crate::guarded::{finite, scale, zeroed};
use crate::{HmxResidencyStatistics, ProjectionStatistics, Result, invalid};

#[inline]
fn finite_slice(values: &[f32]) -> Result<()> {
    if is_finite_slice(values) {
        Ok(())
    } else {
        Err(invalid("nonfinite Hexagon projection value"))
    }
}

/// Residency statistics for [`NativeQ2Executor`].
pub type NativeQ2ResidencyStatistics = HmxResidencyStatistics;

/// Default 28-projection (`blk.0`..=`blk.2` recurrent, `blk.3` full-attention,
/// and `blk.4` MLP) Hexagon NPU `native-q2` resident placement filling 99.9%
/// of the 480 MiB resident weight arena.
pub const DEFAULT_NATIVE_Q2_PROJECTIONS: [&str; 28] = [
    "blk.0.ffn_up.weight",
    "blk.0.ffn_gate.weight",
    "blk.0.ffn_down.weight",
    "blk.0.attn_qkv.weight",
    "blk.0.attn_gate.weight",
    "blk.0.ssm_out.weight",
    "blk.1.ffn_up.weight",
    "blk.1.ffn_gate.weight",
    "blk.1.ffn_down.weight",
    "blk.1.attn_qkv.weight",
    "blk.1.attn_gate.weight",
    "blk.1.ssm_out.weight",
    "blk.2.ffn_up.weight",
    "blk.2.ffn_gate.weight",
    "blk.2.ffn_down.weight",
    "blk.2.attn_qkv.weight",
    "blk.2.attn_gate.weight",
    "blk.2.ssm_out.weight",
    "blk.3.ffn_up.weight",
    "blk.3.ffn_gate.weight",
    "blk.3.ffn_down.weight",
    "blk.3.attn_q.weight",
    "blk.3.attn_k.weight",
    "blk.3.attn_v.weight",
    "blk.3.attn_output.weight",
    "blk.4.ffn_up.weight",
    "blk.4.ffn_gate.weight",
    "blk.4.ffn_down.weight",
];

/// Exact packed `HTP_TYPE_Q2_0` resident weight payload bytes for [`DEFAULT_NATIVE_Q2_PROJECTIONS`].
pub const DEFAULT_NATIVE_Q2_RESIDENT_WEIGHT_BYTES: usize = 502_824_960;

/// Host metadata reservation for resident `native-q2` projection handles.
pub const DEFAULT_NATIVE_Q2_RESIDENT_METADATA_BYTES: usize = 256 * 1024;

struct ResidentProjection {
    source: Arc<[u8]>,
    columns: usize,
    rows: usize,
    first_handle: usize,
    evaluated: bool,
}

struct ResidentCache {
    projections: Vec<ResidentProjection>,
    handles: Vec<NativeQ2WeightHandle>,
    reservation: usize,
    statistics: NativeQ2ResidencyStatistics,
}

impl ResidentCache {
    fn new(reservation: usize) -> Result<Self> {
        let slots =
            reservation / (size_of::<ResidentProjection>() + size_of::<NativeQ2WeightHandle>());
        if slots == 0 {
            return Err(invalid(
                "native Q2 resident metadata reservation is too small",
            ));
        }
        let mut projections = Vec::new();
        let mut handles = Vec::new();
        projections
            .try_reserve_exact(slots)
            .map_err(|_| invalid("native Q2 resident projection metadata allocation failed"))?;
        handles
            .try_reserve_exact(slots)
            .map_err(|_| invalid("native Q2 resident handle metadata allocation failed"))?;
        let metadata_payload_bytes = projections
            .capacity()
            .checked_mul(size_of::<ResidentProjection>())
            .and_then(|bytes| {
                handles
                    .capacity()
                    .checked_mul(size_of::<NativeQ2WeightHandle>())
                    .and_then(|handle_bytes| bytes.checked_add(handle_bytes))
            })
            .filter(|&bytes| bytes <= reservation)
            .ok_or_else(|| invalid("native Q2 resident metadata allocation exceeds reservation"))?;
        Ok(Self {
            projections,
            handles,
            reservation,
            statistics: NativeQ2ResidencyStatistics {
                metadata_payload_bytes,
                ..NativeQ2ResidencyStatistics::default()
            },
        })
    }

    fn find(&self, tensor: &Tensor) -> Option<usize> {
        self.projections.iter().position(|projection| {
            Arc::ptr_eq(&projection.source, &tensor.data)
                && tensor.kind == 142
                && tensor.dimensions.as_slice() == [projection.columns, projection.rows]
        })
    }

    const fn has_metadata_capacity(&self, operations: usize) -> bool {
        self.projections.len() < self.projections.capacity()
            && operations <= self.handles.capacity() - self.handles.len()
    }

    fn commit(&mut self, tensor: &Tensor, first_handle: usize, evaluated: bool) -> usize {
        let index = self.projections.len();
        self.projections.push(ResidentProjection {
            source: Arc::clone(&tensor.data),
            columns: tensor.dimensions[0],
            rows: tensor.dimensions[1],
            first_handle,
            evaluated,
        });
        self.statistics.tensors_prepared = self.statistics.tensors_prepared.saturating_add(1);
        index
    }
}

/// Full-`K` 1-op native `HTP_TYPE_Q2_0` executor using `libggml-htp-v81.so`'s
/// signed 8-thread HVX `q2_0 x q8_1` (`HTP_MM_KERNEL_HVX_QUANT_ROW_N1`) kernel.
pub struct NativeQ2Executor<S = &'static mut dyn NativeQ2Submitter> {
    submitter: S,
    maximum_columns: usize,
    maximum_rows: usize,
    maximum_operations: u32,
    maximum_owned_payload_bytes: usize,
    failed: bool,
    statistics: ProjectionStatistics,
    queue_writes_attempted: u64,
    queue_writes_completed: u64,
    residency: Option<ResidentCache>,
    row_parallelism: Option<RowParallelism>,
    allowed_tensors: Option<Vec<Tensor>>,
    scratch_weights: Vec<u8>,
}

impl<S: NativeQ2Submitter> NativeQ2Executor<S> {
    /// Bind one native `Q2_0` session and an admitted host-payload cap.
    #[must_use]
    pub const fn new(
        submitter: S,
        limits: NativeQ2ProjectionLimits,
        maximum_owned_payload_bytes: usize,
    ) -> Self {
        Self {
            submitter,
            maximum_columns: limits.maximum_columns(),
            maximum_rows: limits.maximum_rows(),
            maximum_operations: limits.maximum_operations(),
            maximum_owned_payload_bytes,
            failed: false,
            statistics: ProjectionStatistics {
                projections_completed: 0,
                operations_attempted: 0,
                operations_completed: 0,
            },
            queue_writes_attempted: 0,
            queue_writes_completed: 0,
            residency: None,
            row_parallelism: None,
            allowed_tensors: None,
            scratch_weights: Vec::new(),
        }
    }

    /// Configure row-parallel tile packing across an explicit private pool.
    ///
    /// # Errors
    /// Rejects repeated or late configuration after any attempted operation.
    pub fn with_row_parallelism(mut self, row_parallelism: RowParallelism) -> Result<Self> {
        if self.row_parallelism.is_some()
            || self.statistics.operations_attempted != 0
            || self.failed
        {
            return Err(invalid("invalid native Q2 row-parallelism configuration"));
        }
        self.row_parallelism = Some(row_parallelism);
        Ok(self)
    }

    /// Enable immutable tiled `Q2_0` weight caching in the session's resident arena.
    ///
    /// # Errors
    /// Rejects repeated/late configuration, absent native capacity or invalid bounds.
    pub fn with_resident_weights(mut self, maximum_cache_metadata_bytes: usize) -> Result<Self> {
        if self.residency.is_some()
            || self.statistics.operations_attempted != 0
            || self.failed
            || self.submitter.residency_capacity_bytes() == 0
            || maximum_cache_metadata_bytes > self.maximum_owned_payload_bytes
        {
            return Err(invalid("invalid native Q2 resident-weight configuration"));
        }
        self.residency = Some(ResidentCache::new(maximum_cache_metadata_bytes)?);
        Ok(self)
    }

    /// Native-arena preparation and host metadata observations.
    #[must_use]
    pub fn residency_statistics(&self) -> NativeQ2ResidencyStatistics {
        self.residency
            .as_ref()
            .map_or_else(NativeQ2ResidencyStatistics::default, |cache| {
                cache.statistics
            })
    }

    /// Actual synchronous projection/submission counts.
    #[must_use]
    pub const fn statistics(&self) -> ProjectionStatistics {
        self.statistics
    }

    /// Actual synchronous queue writes attempted.
    #[must_use]
    pub const fn queue_writes_attempted(&self) -> u64 {
        self.queue_writes_attempted
    }

    /// Actual synchronous queue writes completed.
    #[must_use]
    pub const fn queue_writes_completed(&self) -> u64 {
        self.queue_writes_completed
    }

    /// Any submitted failure permanently closes this executor.
    #[must_use]
    pub const fn is_faulted(&self) -> bool {
        self.failed
    }

    fn payload_bytes(&self, columns: usize, rows: usize) -> Option<usize> {
        let active_cols = columns.min(self.maximum_columns);
        let padded_rows = rows.min(self.maximum_rows).next_multiple_of(32);
        let weight_bytes = padded_rows
            .checked_mul(36)?
            .checked_mul(active_cols / 128)?;
        let chunk_out_bytes = padded_rows.checked_mul(4)?;
        rows.checked_mul(4)?
            .checked_add(weight_bytes)?
            .checked_add(chunk_out_bytes)
            .and_then(|bytes| {
                bytes.checked_add(self.residency.as_ref().map_or(0, |cache| cache.reservation))
            })
    }

    fn prepare_resident(
        &mut self,
        tensor: &Tensor,
        weights: &mut [u8],
        evaluated: bool,
    ) -> Result<Option<usize>> {
        let Some(cache) = &mut self.residency else {
            return Ok(None);
        };
        let (columns, rows) = native_dimensions(tensor)?;
        let col_tiles = columns.div_ceil(self.maximum_columns);
        let row_tiles = rows.div_ceil(self.maximum_rows);
        let operations = col_tiles
            .checked_mul(row_tiles)
            .ok_or_else(|| invalid("native Q2 resident operation count overflow"))?;
        let total_blocks = columns / 128;
        let packed_bytes = rows
            .next_multiple_of(32)
            .checked_mul(36)
            .and_then(|bytes| bytes.checked_mul(total_blocks))
            .ok_or_else(|| invalid("native Q2 resident packed extent overflow"))?;
        if !cache.has_metadata_capacity(operations)
            || packed_bytes > self.submitter.residency_capacity_bytes()
        {
            return Ok(None);
        }
        let first_handle = cache.handles.len();
        for row_start in (0..rows).step_by(self.maximum_rows) {
            let active_rows = (rows - row_start).min(self.maximum_rows);
            let mapped_rows = active_rows.next_multiple_of(32);
            for col_start in (0..columns).step_by(self.maximum_columns) {
                let active_cols = (columns - col_start).min(self.maximum_columns);
                let chunk_blocks = active_cols / 128;
                let block_start = col_start / 128;
                let chunk_weights = &mut weights[..mapped_rows * 36 * chunk_blocks];
                pack_native_chunk(
                    &tensor.data,
                    ChunkGeometry {
                        total_blocks,
                        row_start,
                        active_rows,
                        block_start,
                        chunk_blocks,
                    },
                    chunk_weights,
                    self.row_parallelism.as_ref(),
                )?;
                let handle = self
                    .submitter
                    .prepare_weights(active_cols, mapped_rows, chunk_weights)?
                    .ok_or_else(|| {
                        invalid("native Q2 resident preparation contradicted capacity")
                    })?;
                if handle.columns() != active_cols
                    || handle.rows() != mapped_rows
                    || handle.packed_bytes() != chunk_weights.len()
                {
                    return Err(invalid(
                        "native Q2 resident preparation returned invalid geometry",
                    ));
                }
                cache.handles.push(handle);
                cache.statistics.weight_payload_bytes_prepared = cache
                    .statistics
                    .weight_payload_bytes_prepared
                    .saturating_add(u64::try_from(chunk_weights.len())?);
            }
        }
        Ok(Some(cache.commit(tensor, first_handle, evaluated)))
    }

    /// Pre-pack and stage a `PQ2_0` projection tensor into the session's
    /// resident arena before token evaluation so Token 0 flushes the resident
    /// weight arena only once on its first submission.
    ///
    /// # Errors
    /// Rejects faulted executors, unsupported tensor geometry, or invalid scale headers.
    pub fn prewarm_tensor(&mut self, tensor: &Tensor) -> Result<bool> {
        let mut weights = std::mem::take(&mut self.scratch_weights);
        let result = self.prewarm_tensor_with_scratch(tensor, &mut weights);
        if result.is_ok() {
            self.scratch_weights = weights;
        }
        result
    }

    fn prewarm_tensor_with_scratch(
        &mut self,
        tensor: &Tensor,
        weights: &mut Vec<u8>,
    ) -> Result<bool> {
        if self.failed {
            return Err(invalid(
                "native Q2 Hexagon executor requires terminal teardown",
            ));
        }
        let (columns, rows) = native_dimensions(tensor)?;
        if !self.supports_f32(tensor) {
            return Err(invalid("unsupported native Q2 Hexagon projection contract"));
        }
        let Some(cache) = &self.residency else {
            return Ok(false);
        };
        if cache.find(tensor).is_some() {
            return Ok(true);
        }
        let single_op =
            columns <= self.maximum_columns && rows <= self.maximum_rows && rows.is_multiple_of(32);
        if !single_op {
            for block in tensor.data.as_chunks::<34>().0 {
                scale(block)?;
            }
        }
        let max_chunk_cols = columns.min(self.maximum_columns);
        let max_mapped_rows = rows.min(self.maximum_rows).next_multiple_of(32);
        let required_bytes = max_mapped_rows * 36 * (max_chunk_cols / 128);
        if weights.len() < required_bytes {
            weights
                .try_reserve(required_bytes - weights.len())
                .map_err(|_| invalid("native Q2 scratch allocation failed"))?;
            weights.resize(required_bytes, 0_u8);
        }
        let result = self.prepare_resident(tensor, &mut weights[..required_bytes], false);
        if result.is_err() {
            self.failed = true;
        }
        Ok(result?.is_some())
    }

    /// Restrict NPU execution to the supplied live model tensors and pre-warm
    /// them into the resident weight arena when residency is enabled.
    ///
    /// # Errors
    /// Rejects empty selections, unsupported tensor contracts, or failed pre-warming.
    pub fn select_and_prewarm_tensors(&mut self, tensors: &[Tensor]) -> Result<()> {
        if tensors.is_empty() || self.allowed_tensors.is_some() || self.failed {
            return Err(invalid("invalid native Q2 selected tensor configuration"));
        }
        let mut selected = Vec::new();
        selected
            .try_reserve_exact(tensors.len())
            .map_err(|_| invalid("native Q2 selected tensor metadata allocation failed"))?;
        for tensor in tensors {
            if !self.supports_f32(tensor) {
                return Err(invalid("unsupported native Q2 selected projection tensor"));
            }
            selected.push(tensor.clone());
        }
        self.allowed_tensors = Some(selected);
        if self.residency.is_some() {
            let mut weights = std::mem::take(&mut self.scratch_weights);
            for tensor in tensors {
                if !self.prewarm_tensor_with_scratch(tensor, &mut weights)? {
                    self.failed = true;
                    return Err(invalid(
                        "native Q2 resident arena exhausted during pre-warming",
                    ));
                }
            }
        }
        Ok(())
    }

    fn project_single_op(
        &mut self,
        tensor: &Tensor,
        (columns, rows): (usize, usize),
        resident: Option<usize>,
        weights: &mut [u8],
        activation: &[f32],
        output: &mut Vec<f32>,
    ) -> Result<()> {
        output.clear();
        output
            .try_reserve(rows)
            .map_err(|_| invalid("native Q2 output allocation failed"))?;
        output.resize(rows, f32::NAN);
        let total_blocks = columns / 128;
        if let Some(index) = resident {
            self.statistics.operations_attempted =
                self.statistics.operations_attempted.saturating_add(1);
            self.queue_writes_attempted = self.queue_writes_attempted.saturating_add(1);
            let cache = self
                .residency
                .as_ref()
                .ok_or_else(|| invalid("native Q2 resident cache is unavailable"))?;
            let handle = &cache.handles[cache.projections[index].first_handle];
            self.submitter
                .submit_resident(handle, activation, output.as_mut_slice())?;
        } else {
            pack_native_chunk(
                &tensor.data,
                ChunkGeometry {
                    total_blocks,
                    row_start: 0,
                    active_rows: rows,
                    block_start: 0,
                    chunk_blocks: total_blocks,
                },
                weights,
                self.row_parallelism.as_ref(),
            )?;
            self.statistics.operations_attempted =
                self.statistics.operations_attempted.saturating_add(1);
            self.queue_writes_attempted = self.queue_writes_attempted.saturating_add(1);
            self.submitter
                .submit(columns, rows, weights, activation, output.as_mut_slice())?;
        }
        finite_slice(output)?;
        self.statistics.operations_completed =
            self.statistics.operations_completed.saturating_add(1);
        self.queue_writes_completed = self.queue_writes_completed.saturating_add(1);
        self.statistics.projections_completed =
            self.statistics.projections_completed.saturating_add(1);
        Ok(())
    }

    #[allow(
        clippy::too_many_lines,
        reason = "Keep resident cache lookup, pre-warmed first-hit evaluation and fallback tiling auditable together."
    )]
    fn project_f32_inner(
        &mut self,
        tensor: &Tensor,
        activation: &[f32],
        output: &mut Vec<f32>,
    ) -> Result<()> {
        if self.failed {
            return Err(invalid(
                "native Q2 Hexagon executor requires terminal teardown",
            ));
        }
        let (columns, rows) = native_dimensions(tensor)?;
        if !self.supports_f32(tensor) || activation.len() != columns {
            return Err(invalid("unsupported native Q2 Hexagon projection contract"));
        }
        let single_op =
            columns <= self.maximum_columns && rows <= self.maximum_rows && rows.is_multiple_of(32);
        let cached = self.residency.as_ref().and_then(|cache| cache.find(tensor));
        if cached.is_none() && !single_op {
            for block in tensor.data.as_chunks::<34>().0 {
                scale(block)?;
            }
        }
        finite_slice(activation)?;
        let max_chunk_cols = columns.min(self.maximum_columns);
        let max_mapped_rows = rows.min(self.maximum_rows).next_multiple_of(32);
        let mut weights = if cached.is_none() {
            zeroed(max_mapped_rows * 36 * (max_chunk_cols / 128), 0_u8)?
        } else {
            self.scratch_weights = Vec::new();
            Vec::new()
        };
        let resident = if let Some(index) = cached {
            if let Some(cache) = &mut self.residency {
                if cache.projections[index].evaluated {
                    cache.statistics.cache_hits = cache.statistics.cache_hits.saturating_add(1);
                } else {
                    cache.projections[index].evaluated = true;
                }
            }
            Some(index)
        } else {
            self.prepare_resident(tensor, &mut weights, true)?
        };
        if single_op {
            return self.project_single_op(
                tensor,
                (columns, rows),
                resident,
                &mut weights,
                activation,
                output,
            );
        }
        let mut pending = zeroed(rows, 0.0_f32)?;
        let mut chunk_output = zeroed(max_mapped_rows, 0.0_f32)?;
        let mut operation = 0;
        for row_start in (0..rows).step_by(self.maximum_rows) {
            let active_rows = (rows - row_start).min(self.maximum_rows);
            let mapped_rows = active_rows.next_multiple_of(32);
            let out_slice = &mut chunk_output[..mapped_rows];
            for col_start in (0..columns).step_by(self.maximum_columns) {
                let active_cols = (columns - col_start).min(self.maximum_columns);
                let act_slice = &activation[col_start..col_start + active_cols];
                out_slice.fill(f32::NAN);
                self.statistics.operations_attempted =
                    self.statistics.operations_attempted.saturating_add(1);
                self.queue_writes_attempted = self.queue_writes_attempted.saturating_add(1);
                if let Some(index) = resident {
                    let cache = self
                        .residency
                        .as_ref()
                        .ok_or_else(|| invalid("native Q2 resident cache is unavailable"))?;
                    let handle = &cache.handles[cache.projections[index].first_handle + operation];
                    self.submitter
                        .submit_resident(handle, act_slice, out_slice)?;
                } else {
                    let chunk_blocks = active_cols / 128;
                    let geom = ChunkGeometry {
                        total_blocks: columns / 128,
                        row_start,
                        active_rows,
                        block_start: col_start / 128,
                        chunk_blocks,
                    };
                    let chunk_weights = &mut weights[..mapped_rows * 36 * chunk_blocks];
                    pack_native_chunk(
                        &tensor.data,
                        geom,
                        chunk_weights,
                        self.row_parallelism.as_ref(),
                    )?;
                    self.submitter.submit(
                        active_cols,
                        mapped_rows,
                        chunk_weights,
                        act_slice,
                        out_slice,
                    )?;
                }
                operation += 1;
                accumulate_tile_output(
                    &mut pending[row_start..row_start + active_rows],
                    out_slice,
                )?;
                self.statistics.operations_completed =
                    self.statistics.operations_completed.saturating_add(1);
                self.queue_writes_completed = self.queue_writes_completed.saturating_add(1);
            }
        }
        self.statistics.projections_completed =
            self.statistics.projections_completed.saturating_add(1);
        *output = pending;
        Ok(())
    }

    fn supports_f32_group_inner(&self, tensors: &[&Tensor]) -> bool {
        if self.failed || !matches!(tensors.len(), 2 | 3) {
            return false;
        }
        let Some(cache) = &self.residency else {
            return false;
        };
        let Ok((columns, _)) = native_dimensions(tensors[0]) else {
            return false;
        };
        if columns > self.maximum_columns {
            return false;
        }
        let mut total_output_bytes = 0_usize;
        let mut uncached_packed_bytes = 0_usize;
        let mut uncached_ops = 0_usize;
        for &tensor in tensors {
            if !self.supports_f32(tensor) {
                return false;
            }
            let Ok((cols, rows)) = native_dimensions(tensor) else {
                return false;
            };
            if cols != columns || rows > self.maximum_rows || !rows.is_multiple_of(32) {
                return false;
            }
            let Some(out_bytes) = rows.checked_mul(4) else {
                return false;
            };
            let Some(next_total_out) = total_output_bytes.checked_add(out_bytes) else {
                return false;
            };
            total_output_bytes = next_total_out;
            if cache.find(tensor).is_none() {
                let Some(packed) = rows
                    .checked_mul(36)
                    .and_then(|b| b.checked_mul(columns / 128))
                else {
                    return false;
                };
                let Some(next_uncached) = uncached_packed_bytes.checked_add(packed) else {
                    return false;
                };
                uncached_packed_bytes = next_uncached;
                uncached_ops += 1;
            }
        }
        if uncached_ops > 0
            && (!cache.has_metadata_capacity(uncached_ops)
                || uncached_packed_bytes > self.submitter.residency_capacity_bytes())
        {
            return false;
        }
        total_output_bytes
            .checked_add(cache.reservation)
            .is_some_and(|bytes| bytes <= self.maximum_owned_payload_bytes)
    }

    #[allow(
        clippy::too_many_lines,
        reason = "Keep resident group preparation, 2/3-op dispatch and output validation together."
    )]
    fn project_f32_group_inner(
        &mut self,
        tensors: &[&Tensor],
        activation: &[f32],
        outputs: &mut [&mut Vec<f32>],
    ) -> Result<()> {
        if self.failed {
            return Err(invalid(
                "native Q2 Hexagon executor requires terminal teardown",
            ));
        }
        if tensors.len() != outputs.len() || !self.supports_f32_group_inner(tensors) {
            return Err(invalid(
                "unsupported native Q2 Hexagon batched projection group",
            ));
        }
        let (columns, _) = native_dimensions(tensors[0])?;
        if activation.len() != columns {
            return Err(invalid("unsupported native Q2 Hexagon projection contract"));
        }
        finite_slice(activation)?;
        let mut weights = Vec::new();
        let mut indices = [0_usize; 3];
        let mut rows_per_member = [0_usize; 3];
        for (member, &tensor) in tensors.iter().enumerate() {
            let (_, rows) = native_dimensions(tensor)?;
            rows_per_member[member] = rows;
            let cached = self.residency.as_ref().and_then(|cache| cache.find(tensor));
            let index = if let Some(idx) = cached {
                if let Some(cache) = &mut self.residency {
                    if cache.projections[idx].evaluated {
                        cache.statistics.cache_hits = cache.statistics.cache_hits.saturating_add(1);
                    } else {
                        cache.projections[idx].evaluated = true;
                    }
                }
                idx
            } else {
                let max_mapped_rows = rows.next_multiple_of(32);
                let required_bytes = max_mapped_rows * 36 * (columns / 128);
                if weights.len() < required_bytes {
                    weights
                        .try_reserve(required_bytes - weights.len())
                        .map_err(|_| invalid("native Q2 scratch allocation failed"))?;
                    weights.resize(required_bytes, 0_u8);
                }
                self.prepare_resident(tensor, &mut weights[..required_bytes], true)?
                    .ok_or_else(|| invalid("native Q2 resident preparation failed"))?
            };
            indices[member] = index;
        }
        for (member, output) in outputs.iter_mut().enumerate() {
            let rows = rows_per_member[member];
            output.clear();
            output
                .try_reserve(rows)
                .map_err(|_| invalid("native Q2 output allocation failed"))?;
            output.resize(rows, f32::NAN);
        }
        let count_u64 = u64::try_from(tensors.len())?;
        self.statistics.operations_attempted = self
            .statistics
            .operations_attempted
            .saturating_add(count_u64);
        self.queue_writes_attempted = self.queue_writes_attempted.saturating_add(1);
        let cache = self
            .residency
            .as_ref()
            .ok_or_else(|| invalid("native Q2 resident cache is unavailable"))?;
        match outputs {
            [out0, out1] => {
                let h0 = &cache.handles[cache.projections[indices[0]].first_handle];
                let h1 = &cache.handles[cache.projections[indices[1]].first_handle];
                self.submitter.submit_resident_batch2(
                    [h0, h1],
                    activation,
                    [out0.as_mut_slice(), out1.as_mut_slice()],
                )?;
            }
            [out0, out1, out2] => {
                let h0 = &cache.handles[cache.projections[indices[0]].first_handle];
                let h1 = &cache.handles[cache.projections[indices[1]].first_handle];
                let h2 = &cache.handles[cache.projections[indices[2]].first_handle];
                self.submitter.submit_resident_batch3(
                    [h0, h1, h2],
                    activation,
                    [
                        out0.as_mut_slice(),
                        out1.as_mut_slice(),
                        out2.as_mut_slice(),
                    ],
                )?;
            }
            _ => return Err(invalid("unsupported native Q2 batch size")),
        }
        for output in outputs.iter() {
            finite_slice(output)?;
        }
        self.statistics.operations_completed = self
            .statistics
            .operations_completed
            .saturating_add(count_u64);
        self.queue_writes_completed = self.queue_writes_completed.saturating_add(1);
        self.statistics.projections_completed = self
            .statistics
            .projections_completed
            .saturating_add(count_u64);
        Ok(())
    }
}

fn accumulate_tile_output(dst_rows: &mut [f32], out_slice: &[f32]) -> Result<()> {
    let active_rows = dst_rows.len();
    for &pad_value in &out_slice[active_rows..] {
        if finite(pad_value)? != 0.0 {
            return Err(invalid("Hexagon native Q2 padding output changed"));
        }
    }
    for (dst, &val) in dst_rows.iter_mut().zip(&out_slice[..active_rows]) {
        *dst = finite(*dst + finite(val)?)?;
    }
    Ok(())
}

impl<S: NativeQ2Submitter> Pq2ProjectionExecutor for NativeQ2Executor<S> {
    fn identity(&self) -> &'static str {
        if self.residency.is_some() {
            "hexagon-hvx-native-q2-resident-v1-experimental"
        } else {
            "hexagon-hvx-native-q2-v1-experimental"
        }
    }

    fn operation_capacity(&self) -> Option<u64> {
        if self.maximum_operations == u32::MAX && self.submitter.remaining_operations().is_none() {
            return None;
        }
        let local =
            u64::from(self.maximum_operations).saturating_sub(self.statistics.operations_attempted);
        Some(local.min(self.submitter.remaining_operations().unwrap_or(local)))
    }

    fn operation_count(&self, tensor: &Tensor) -> Option<u64> {
        let (columns, rows) = native_dimensions(tensor).ok()?;
        let col_tiles = columns.div_ceil(self.maximum_columns);
        let row_tiles = rows.div_ceil(self.maximum_rows);
        u64::try_from(col_tiles.checked_mul(row_tiles)?).ok()
    }

    fn supports(&self, _tensor: &Tensor) -> bool {
        false
    }

    fn project(
        &mut self,
        _tensor: &Tensor,
        _activation: &[Q8KBlock],
        _fused: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        output.clear();
        self.failed = true;
        Err(invalid("NativeQ2Executor uses direct F32 activations"))
    }

    fn supports_f32(&self, tensor: &Tensor) -> bool {
        if let Some(allowed) = &self.allowed_tensors
            && !allowed.iter().any(|candidate| {
                Arc::ptr_eq(&candidate.data, &tensor.data)
                    && candidate.kind == tensor.kind
                    && candidate.dimensions == tensor.dimensions
                    && candidate.name == tensor.name
            })
        {
            return false;
        }
        native_dimensions(tensor).is_ok_and(|(columns, rows)| {
            self.payload_bytes(columns, rows)
                .is_some_and(|bytes| bytes <= self.maximum_owned_payload_bytes)
        })
    }

    fn project_f32(
        &mut self,
        tensor: &Tensor,
        activation: &[f32],
        _pinned: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        output.clear();
        let result = self.project_f32_inner(tensor, activation, output);
        if result.is_err() {
            self.failed = true;
            output.clear();
        }
        result
    }

    fn supports_f32_group(&self, tensors: &[&Tensor]) -> bool {
        self.supports_f32_group_inner(tensors)
    }

    fn project_f32_group(
        &mut self,
        tensors: &[&Tensor],
        activation: &[f32],
        _pinned: bool,
        outputs: &mut [&mut Vec<f32>],
    ) -> Result<()> {
        for output in outputs.iter_mut() {
            output.clear();
        }
        let result = self.project_f32_group_inner(tensors, activation, outputs);
        if result.is_err() {
            self.failed = true;
            for output in outputs.iter_mut() {
                output.clear();
            }
        }
        result
    }
}

fn native_dimensions(tensor: &Tensor) -> Result<(usize, usize)> {
    if tensor.kind != 142 || tensor.dimensions.len() != 2 {
        return Err(invalid("unsupported Hexagon native Q2 tensor type or rank"));
    }
    let (columns, rows) = (tensor.dimensions[0], tensor.dimensions[1]);
    if columns == 0
        || rows == 0
        || !columns.is_multiple_of(128)
        || (columns / 128)
            .checked_mul(34)
            .and_then(|bytes| bytes.checked_mul(rows))
            != Some(tensor.data.len())
    {
        return Err(invalid(
            "invalid Hexagon native Q2 tensor geometry or extent",
        ));
    }
    Ok((columns, rows))
}

#[derive(Clone, Copy)]
struct ChunkGeometry {
    total_blocks: usize,
    row_start: usize,
    active_rows: usize,
    block_start: usize,
    chunk_blocks: usize,
}

fn initialize_native_weights(weights: &mut [u8]) {
    for tile in weights.as_chunks_mut::<576>().0 {
        tile[..512].fill(0x55);
        tile[512..576].fill(0);
    }
}

fn pack_row_group(
    tensor_data: &[u8],
    geometry: ChunkGeometry,
    row_group: usize,
    group_weights: &mut [u8],
) -> Result<()> {
    let group_row_start = row_group * 32;
    let group_rows = geometry.active_rows.saturating_sub(group_row_start).min(32);
    if group_rows < 32 {
        initialize_native_weights(group_weights);
    }
    for row_in_tile in 0..group_rows {
        let local_row = group_row_start + row_in_tile;
        let row_block_base =
            (geometry.row_start + local_row) * geometry.total_blocks + geometry.block_start;
        for local_block in 0..geometry.chunk_blocks {
            let offset = (row_block_base + local_block) * 34;
            let original = &tensor_data[offset..offset + 34];
            scale(original)?;
            let tile0 = (local_block * 2) * 576;
            let tile1 = tile0 + 576;
            let scale_offset0 = tile0 + 512 + row_in_tile * 2;
            let scale_offset1 = tile1 + 512 + row_in_tile * 2;
            group_weights[scale_offset0..scale_offset0 + 2].copy_from_slice(&original[..2]);
            group_weights[scale_offset1..scale_offset1 + 2].copy_from_slice(&original[..2]);
            for (pair, dword_bytes) in original[2..34].as_chunks::<8>().0.iter().enumerate() {
                let mut packed = u64::from_le_bytes(*dword_bytes);
                let swap = (packed ^ (packed >> 6)) & 0x00cc_00cc_00cc_00cc;
                packed ^= swap ^ (swap << 6);
                let swap = (packed ^ (packed >> 12)) & 0x0000_f0f0_0000_f0f0;
                packed ^= swap ^ (swap << 12);
                let bytes = packed.to_le_bytes();
                let tile = if pair < 2 { tile0 } else { tile1 };
                let base = tile + (pair % 2) * 256 + row_in_tile * 4;
                group_weights[base..base + 4].copy_from_slice(&bytes[..4]);
                group_weights[base + 128..base + 132].copy_from_slice(&bytes[4..8]);
            }
        }
    }
    Ok(())
}

fn pack_native_chunk(
    tensor_data: &[u8],
    geometry: ChunkGeometry,
    weights: &mut [u8],
    row_parallelism: Option<&RowParallelism>,
) -> Result<()> {
    let row_group_bytes = geometry.chunk_blocks * 2 * 576;
    if let Some(parallelism) = row_parallelism {
        return parallelism.update_chunks(weights, row_group_bytes, |base_group, chunk| {
            for (group_offset, group_weights) in chunk.chunks_exact_mut(row_group_bytes).enumerate()
            {
                pack_row_group(
                    tensor_data,
                    geometry,
                    base_group + group_offset,
                    group_weights,
                )?;
            }
            Ok(())
        });
    }
    for (row_group, group_weights) in weights.chunks_exact_mut(row_group_bytes).enumerate() {
        pack_row_group(tensor_data, geometry, row_group, group_weights)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    struct MockNativeSubmitter {
        calls: u64,
        resident_calls: u64,
        remaining: u64,
        residency_bytes: usize,
    }

    impl NativeQ2Submitter for MockNativeSubmitter {
        fn remaining_operations(&self) -> Option<u64> {
            Some(self.remaining.saturating_sub(self.calls))
        }

        fn residency_capacity_bytes(&self) -> usize {
            self.residency_bytes
        }

        fn prepare_weights(
            &mut self,
            columns: usize,
            rows: usize,
            weights: &[u8],
        ) -> std::result::Result<Option<NativeQ2WeightHandle>, hexagon_native_rpc::Error> {
            let mut registry =
                hexagon_native_rpc::NativeQ2WeightRegistry::new(self.residency_bytes)?;
            let handle = registry.prepare(columns, rows, weights)?;
            if handle.is_some() {
                self.residency_bytes = self.residency_bytes.saturating_sub(weights.len());
            }
            Ok(handle)
        }

        fn submit_resident(
            &mut self,
            handle: &NativeQ2WeightHandle,
            activation: &[f32],
            output: &mut [f32],
        ) -> std::result::Result<(), hexagon_native_rpc::Error> {
            self.calls += 1;
            self.resident_calls += 1;
            if activation.len() != handle.columns() || output.len() != handle.rows() {
                return Err(hexagon_native_rpc::Error::Contract);
            }
            output.fill(0.0);
            output[0] = activation.iter().sum();
            Ok(())
        }

        fn submit_resident_batch2(
            &mut self,
            handles: [&NativeQ2WeightHandle; 2],
            activation: &[f32],
            outputs: [&mut [f32]; 2],
        ) -> std::result::Result<(), hexagon_native_rpc::Error> {
            self.calls += 2;
            self.resident_calls += 2;
            for (member, handle) in handles.into_iter().enumerate() {
                if activation.len() != handle.columns() || outputs[member].len() != handle.rows() {
                    return Err(hexagon_native_rpc::Error::Contract);
                }
                outputs[member].fill(0.0);
                outputs[member][0] =
                    activation.iter().sum::<f32>() + f32::from(u8::try_from(member).unwrap_or(0));
            }
            Ok(())
        }

        fn submit_resident_batch3(
            &mut self,
            handles: [&NativeQ2WeightHandle; 3],
            activation: &[f32],
            outputs: [&mut [f32]; 3],
        ) -> std::result::Result<(), hexagon_native_rpc::Error> {
            self.calls += 3;
            self.resident_calls += 3;
            for (member, handle) in handles.into_iter().enumerate() {
                if activation.len() != handle.columns() || outputs[member].len() != handle.rows() {
                    return Err(hexagon_native_rpc::Error::Contract);
                }
                outputs[member].fill(0.0);
                outputs[member][0] =
                    activation.iter().sum::<f32>() + f32::from(u8::try_from(member).unwrap_or(0));
            }
            Ok(())
        }

        fn submit(
            &mut self,
            columns: usize,
            rows: usize,
            weights: &[u8],
            activation: &[f32],
            output: &mut [f32],
        ) -> std::result::Result<(), hexagon_native_rpc::Error> {
            self.calls += 1;
            if activation.len() != columns
                || output.len() != rows
                || weights.len() != rows * 36 * (columns / 128)
            {
                return Err(hexagon_native_rpc::Error::Contract);
            }
            output.fill(0.0);
            output[0] = activation.iter().sum();
            Ok(())
        }
    }

    #[test]
    fn native_q2_transcodes_scales_and_executes_full_k_in_one_operation() -> Result<()> {
        let columns = 256;
        let rows = 32;
        let blocks = columns / 128;
        let mut data = vec![0x55_u8; rows * blocks * 34];
        for block in data.as_chunks_mut::<34>().0 {
            block[0] = 0x00;
            block[1] = 0x38; // 0.5 in FP16
        }
        let tensor = Tensor {
            name: "blk.0.ffn_up.weight".to_owned(),
            dimensions: vec![columns, rows],
            kind: 142,
            data: Arc::from(data),
        };
        let limits = NativeQ2ProjectionLimits::new(5120, 17_408, 100, Duration::from_secs(10))
            .map_err(|_| invalid("limits"))?
            .with_resident_weights(2304)
            .map_err(|_| invalid("residency"))?;
        let mut submitter = MockNativeSubmitter {
            calls: 0,
            resident_calls: 0,
            remaining: 100,
            residency_bytes: 2304,
        };
        let mut executor = NativeQ2Executor::new(&mut submitter, limits, 64 * 1024 * 1024)
            .with_resident_weights(4096)?;
        assert_eq!(executor.operation_count(&tensor), Some(1));
        assert!(executor.supports_f32(&tensor));

        let activation = vec![1.0_f32; columns];
        let mut output = Vec::new();
        executor.project_f32(&tensor, &activation, true, &mut output)?;
        assert_eq!(output.len(), rows);
        assert!((output[0] - 256.0).abs() < 1e-5);
        assert_eq!(executor.statistics().operations_completed, 1);
        assert_eq!(executor.residency_statistics().tensors_prepared, 1);

        executor.project_f32(&tensor, &activation, true, &mut output)?;
        assert_eq!(executor.residency_statistics().cache_hits, 1);
        assert_eq!(executor.statistics().operations_completed, 2);
        Ok(())
    }

    #[test]
    fn pack_row_group_matches_reference_word_interleave() -> Result<()> {
        let total_blocks = 2;
        for active_rows in [17, 32] {
            let mut tensor_data = vec![0_u8; 32 * total_blocks * 34];
            for (idx, b) in tensor_data.iter_mut().enumerate() {
                *b = u8::try_from((idx * 173 + 29) & 0xff).unwrap_or(0);
            }
            for block in tensor_data.as_chunks_mut::<34>().0 {
                block[0] = 0x55;
                block[1] = 0x35;
            }
            let geometry = ChunkGeometry {
                total_blocks,
                row_start: 0,
                active_rows,
                block_start: 0,
                chunk_blocks: total_blocks,
            };
            let mut actual = vec![0xaa_u8; total_blocks * 2 * 576];
            pack_row_group(&tensor_data, geometry, 0, &mut actual)?;

            let mut expected = vec![0_u8; total_blocks * 2 * 576];
            initialize_native_weights(&mut expected);
            for row_in_tile in 0..active_rows {
                let row_block_base = row_in_tile * total_blocks;
                for local_block in 0..total_blocks {
                    let offset = (row_block_base + local_block) * 34;
                    let original = &tensor_data[offset..offset + 34];
                    let tile0 = (local_block * 2) * 576;
                    let tile1 = tile0 + 576;
                    expected[tile0 + 512 + row_in_tile * 2..tile0 + 514 + row_in_tile * 2]
                        .copy_from_slice(&original[..2]);
                    expected[tile1 + 512 + row_in_tile * 2..tile1 + 514 + row_in_tile * 2]
                        .copy_from_slice(&original[..2]);
                    for (chunk, word_bytes) in original[2..34].as_chunks::<4>().0.iter().enumerate()
                    {
                        let mut packed = u32::from_le_bytes(*word_bytes);
                        let swap = (packed ^ (packed >> 6)) & 0x00cc_00cc;
                        packed ^= swap ^ (swap << 6);
                        let swap = (packed ^ (packed >> 12)) & 0x0000_f0f0;
                        packed ^= swap ^ (swap << 12);
                        let tile = if chunk < 4 { tile0 } else { tile1 };
                        let byte = tile + (chunk % 4) * 128 + row_in_tile * 4;
                        expected[byte..byte + 4].copy_from_slice(&packed.to_le_bytes());
                    }
                }
            }
            assert_eq!(actual, expected);
        }
        Ok(())
    }

    #[test]
    fn default_four_layer_projection_set_matches_exact_resident_arena_budget() -> Result<()> {
        assert_eq!(DEFAULT_NATIVE_Q2_PROJECTIONS.len(), 28);
        let mut total_bytes = 0_usize;
        for name in DEFAULT_NATIVE_Q2_PROJECTIONS {
            let (columns, rows): (usize, usize) = if name.ends_with(".ffn_up.weight")
                || name.ends_with(".ffn_gate.weight")
            {
                (5120, 17_408)
            } else if name.ends_with(".ffn_down.weight") {
                (17_408, 5120)
            } else if name.ends_with(".attn_qkv.weight") {
                (5120, 10_240)
            } else if name.ends_with(".attn_gate.weight") {
                (5120, 6144)
            } else if name.ends_with(".ssm_out.weight") || name.ends_with(".attn_output.weight") {
                (6144, 5120)
            } else if name.ends_with(".attn_q.weight") {
                (5120, 12_288)
            } else if name.ends_with(".attn_k.weight") || name.ends_with(".attn_v.weight") {
                (5120, 1024)
            } else {
                return Err(invalid("unexpected projection in default resident set"));
            };
            let packed_bytes = rows * 36 * (columns / 128);
            assert_eq!(packed_bytes % 128, 0);
            total_bytes += packed_bytes;
        }
        assert_eq!(total_bytes, DEFAULT_NATIVE_Q2_RESIDENT_WEIGHT_BYTES);
        const {
            assert!(
                DEFAULT_NATIVE_Q2_RESIDENT_WEIGHT_BYTES
                    <= hexagon_native_rpc::NATIVE_Q2_MAX_RESIDENT_WEIGHT_BYTES
            );
        };
        Ok(())
    }

    #[test]
    fn native_q2_batches_two_and_three_resident_sibling_projections_in_single_queue_writes()
    -> Result<()> {
        let columns = 256;
        let blocks = columns / 128;
        let make_tensor = |name: &str, rows: usize| {
            let mut data = vec![0x55_u8; rows * blocks * 34];
            for block in data.as_chunks_mut::<34>().0 {
                block[0] = 0x00;
                block[1] = 0x38;
            }
            Tensor {
                name: name.to_owned(),
                dimensions: vec![columns, rows],
                kind: 142,
                data: Arc::from(data),
            }
        };
        let qkv = make_tensor("blk.0.attn_qkv.weight", 64);
        let gate = make_tensor("blk.0.attn_gate.weight", 32);
        let q = make_tensor("blk.3.attn_q.weight", 64);
        let k = make_tensor("blk.3.attn_k.weight", 32);
        let v = make_tensor("blk.3.attn_v.weight", 32);

        let limits = NativeQ2ProjectionLimits::new(5120, 17_408, 100, Duration::from_secs(10))
            .map_err(|_| invalid("limits"))?
            .with_resident_weights(65_536)
            .map_err(|_| invalid("residency"))?;
        let mut submitter = MockNativeSubmitter {
            calls: 0,
            resident_calls: 0,
            remaining: 100,
            residency_bytes: 65_536,
        };
        let mut executor = NativeQ2Executor::new(&mut submitter, limits, 64 * 1024 * 1024)
            .with_resident_weights(16_384)?;
        executor.select_and_prewarm_tensors(&[
            qkv.clone(),
            gate.clone(),
            q.clone(),
            k.clone(),
            v.clone(),
        ])?;
        assert!(executor.supports_f32_group(&[&qkv, &gate]));
        assert!(executor.supports_f32_group(&[&q, &k, &v]));

        let activation = vec![1.0_f32; columns];
        let mut out0 = Vec::new();
        let mut out1 = Vec::new();
        let mut out2 = Vec::new();
        executor.project_f32_group(
            &[&qkv, &gate],
            &activation,
            true,
            &mut [&mut out0, &mut out1],
        )?;
        assert_eq!(out0.len(), 64);
        assert_eq!(out1.len(), 32);
        assert_eq!(out0[0], 256.0);
        assert_eq!(out1[0], 257.0);
        assert_eq!(executor.queue_writes_completed(), 1);
        assert_eq!(executor.statistics().operations_completed, 2);
        assert_eq!(executor.statistics().projections_completed, 2);

        executor.project_f32_group(
            &[&q, &k, &v],
            &activation,
            true,
            &mut [&mut out0, &mut out1, &mut out2],
        )?;
        assert_eq!(out0.len(), 64);
        assert_eq!(out1.len(), 32);
        assert_eq!(out2.len(), 32);
        assert_eq!(out0[0], 256.0);
        assert_eq!(out1[0], 257.0);
        assert_eq!(out2[0], 258.0);
        assert_eq!(executor.queue_writes_completed(), 2);
        assert_eq!(executor.statistics().operations_completed, 5);
        assert_eq!(executor.statistics().projections_completed, 5);
        Ok(())
    }
}
