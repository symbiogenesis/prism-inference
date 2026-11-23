//! Full-`K` 1-op native `HTP_TYPE_Q2_0` projection executor over `libggml-htp-v81.so`.
//!
//! Losslessly transcodes Swift's `PQ2_0` (`34` bytes per 128 weights) and
//! `PTQ1_0` (`28` bytes per 128 weights) into `llama.cpp`'s native tiled
//! `HTP_TYPE_Q2_0` (`576` bytes per 32x64 tile, repeating each 128-weight
//! `FP16` scale across both 64-weight sub-tiles) and executes the full
//! `K <= 5120`, `M <= 17408` projection in a single DSP operation
//! (`HTP_MM_KERNEL_HVX_QUANT_ROW_N1`).

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bonsai_prism_runtime::{
    kernels::{Pq2ProjectionExecutor, RowParallelism, is_finite_slice},
    model::{Model, Tensor},
};
use bonsai_ternary::Q8KBlock;
use hexagon_native_rpc::{
    HMX_Q2_VTCM_REQUEST_BYTES, HtpArchProfile, NATIVE_Q2_MAX_ARENA_WEIGHT_BYTES,
    NATIVE_Q2_MAX_COLUMNS, NATIVE_Q2_MAX_ROWS, NATIVE_Q2_MAX_WEIGHT_ARENAS,
    NATIVE_Q2_VTCM_REQUEST_BYTES, NativeQ2ProjectionLimits, NativeQ2Submitter,
    NativeQ2WeightHandle, is_supported_htp_architecture,
};

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

/// Default 168-projection Hexagon NPU `native-q2` multi-arena resident placement (`blk.0`..=`blk.26`).
///
/// Covers 27 layers (21 recurrent GDN layers and 6 full-attention layers,
/// `2,888,663,040` bytes across six `<= 480` MiB `FastRPC` weight arenas).
pub const DEFAULT_NATIVE_Q2_PROJECTIONS: [&str; 168] = [
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
    "blk.4.attn_qkv.weight",
    "blk.4.attn_gate.weight",
    "blk.4.ssm_out.weight",
    "blk.5.ffn_up.weight",
    "blk.5.ffn_gate.weight",
    "blk.5.ffn_down.weight",
    "blk.5.attn_qkv.weight",
    "blk.5.attn_gate.weight",
    "blk.5.ssm_out.weight",
    "blk.6.ffn_up.weight",
    "blk.6.ffn_gate.weight",
    "blk.6.ffn_down.weight",
    "blk.6.attn_qkv.weight",
    "blk.6.attn_gate.weight",
    "blk.6.ssm_out.weight",
    "blk.7.ffn_up.weight",
    "blk.7.ffn_gate.weight",
    "blk.7.ffn_down.weight",
    "blk.7.attn_q.weight",
    "blk.7.attn_k.weight",
    "blk.7.attn_v.weight",
    "blk.7.attn_output.weight",
    "blk.8.ffn_up.weight",
    "blk.8.ffn_gate.weight",
    "blk.8.ffn_down.weight",
    "blk.8.attn_qkv.weight",
    "blk.8.attn_gate.weight",
    "blk.8.ssm_out.weight",
    "blk.9.ffn_up.weight",
    "blk.9.ffn_gate.weight",
    "blk.9.ffn_down.weight",
    "blk.9.attn_qkv.weight",
    "blk.9.attn_gate.weight",
    "blk.9.ssm_out.weight",
    "blk.10.ffn_up.weight",
    "blk.10.ffn_gate.weight",
    "blk.10.ffn_down.weight",
    "blk.10.attn_qkv.weight",
    "blk.10.attn_gate.weight",
    "blk.10.ssm_out.weight",
    "blk.11.ffn_up.weight",
    "blk.11.ffn_gate.weight",
    "blk.11.ffn_down.weight",
    "blk.11.attn_q.weight",
    "blk.11.attn_k.weight",
    "blk.11.attn_v.weight",
    "blk.11.attn_output.weight",
    "blk.12.ffn_up.weight",
    "blk.12.ffn_gate.weight",
    "blk.12.ffn_down.weight",
    "blk.12.attn_qkv.weight",
    "blk.12.attn_gate.weight",
    "blk.12.ssm_out.weight",
    "blk.13.ffn_up.weight",
    "blk.13.ffn_gate.weight",
    "blk.13.ffn_down.weight",
    "blk.13.attn_qkv.weight",
    "blk.13.attn_gate.weight",
    "blk.13.ssm_out.weight",
    "blk.14.ffn_up.weight",
    "blk.14.ffn_gate.weight",
    "blk.14.ffn_down.weight",
    "blk.14.attn_qkv.weight",
    "blk.14.attn_gate.weight",
    "blk.14.ssm_out.weight",
    "blk.15.ffn_up.weight",
    "blk.15.ffn_gate.weight",
    "blk.15.ffn_down.weight",
    "blk.15.attn_q.weight",
    "blk.15.attn_k.weight",
    "blk.15.attn_v.weight",
    "blk.15.attn_output.weight",
    "blk.16.ffn_up.weight",
    "blk.16.ffn_gate.weight",
    "blk.16.ffn_down.weight",
    "blk.16.attn_qkv.weight",
    "blk.16.attn_gate.weight",
    "blk.16.ssm_out.weight",
    "blk.17.ffn_up.weight",
    "blk.17.ffn_gate.weight",
    "blk.17.ffn_down.weight",
    "blk.17.attn_qkv.weight",
    "blk.17.attn_gate.weight",
    "blk.17.ssm_out.weight",
    "blk.18.ffn_up.weight",
    "blk.18.ffn_gate.weight",
    "blk.18.ffn_down.weight",
    "blk.18.attn_qkv.weight",
    "blk.18.attn_gate.weight",
    "blk.18.ssm_out.weight",
    "blk.19.ffn_up.weight",
    "blk.19.ffn_gate.weight",
    "blk.19.ffn_down.weight",
    "blk.19.attn_q.weight",
    "blk.19.attn_k.weight",
    "blk.19.attn_v.weight",
    "blk.19.attn_output.weight",
    "blk.20.ffn_up.weight",
    "blk.20.ffn_gate.weight",
    "blk.20.ffn_down.weight",
    "blk.20.attn_qkv.weight",
    "blk.20.attn_gate.weight",
    "blk.20.ssm_out.weight",
    "blk.21.ffn_up.weight",
    "blk.21.ffn_gate.weight",
    "blk.21.ffn_down.weight",
    "blk.21.attn_qkv.weight",
    "blk.21.attn_gate.weight",
    "blk.21.ssm_out.weight",
    "blk.22.ffn_up.weight",
    "blk.22.ffn_gate.weight",
    "blk.22.ffn_down.weight",
    "blk.22.attn_qkv.weight",
    "blk.22.attn_gate.weight",
    "blk.22.ssm_out.weight",
    "blk.23.ffn_up.weight",
    "blk.23.ffn_gate.weight",
    "blk.23.ffn_down.weight",
    "blk.23.attn_q.weight",
    "blk.23.attn_k.weight",
    "blk.23.attn_v.weight",
    "blk.23.attn_output.weight",
    "blk.24.ffn_up.weight",
    "blk.24.ffn_gate.weight",
    "blk.24.ffn_down.weight",
    "blk.24.attn_qkv.weight",
    "blk.24.attn_gate.weight",
    "blk.24.ssm_out.weight",
    "blk.25.ffn_up.weight",
    "blk.25.ffn_gate.weight",
    "blk.25.ffn_down.weight",
    "blk.25.attn_qkv.weight",
    "blk.25.attn_gate.weight",
    "blk.25.ssm_out.weight",
    "blk.26.ffn_up.weight",
    "blk.26.ffn_gate.weight",
    "blk.26.ffn_down.weight",
    "blk.26.attn_qkv.weight",
    "blk.26.attn_gate.weight",
    "blk.26.ssm_out.weight",
];

/// Exact packed `HTP_TYPE_Q2_0` resident weight payload bytes for [`DEFAULT_NATIVE_Q2_PROJECTIONS`].
pub const DEFAULT_NATIVE_Q2_RESIDENT_WEIGHT_BYTES: usize = 2_888_663_040;

/// Host metadata reservation for resident `native-q2` projection handles.
pub const DEFAULT_NATIVE_Q2_RESIDENT_METADATA_BYTES: usize = 512 * 1024;

struct ResidentProjection {
    source: bonsai_prism_runtime::model::TensorBytes,
    kind: u32,
    columns: usize,
    rows: usize,
    first_handle: usize,
    evaluated: bool,
    host_released: bool,
}

struct ResidentCache {
    projections: Vec<ResidentProjection>,
    // Sorted indirection keeps projection indices stable while new sources are
    // prepared. Encoding kind and geometry are part of the key: one immutable
    // byte allocation may legitimately be viewed as differently shaped matrices
    // or distinct packed representations without colliding.
    by_storage: Vec<usize>,
    handles: Vec<NativeQ2WeightHandle>,
    reservation: usize,
    statistics: NativeQ2ResidencyStatistics,
}

impl ResidentCache {
    fn new(reservation: usize) -> Result<Self> {
        let slots = reservation
            / (size_of::<ResidentProjection>()
                + size_of::<NativeQ2WeightHandle>()
                + size_of::<usize>());
        if slots == 0 {
            return Err(invalid(
                "native Q2 resident metadata reservation is too small",
            ));
        }
        let mut projections = Vec::new();
        let mut by_storage = Vec::new();
        let mut handles = Vec::new();
        projections
            .try_reserve_exact(slots)
            .map_err(|_| invalid("native Q2 resident projection metadata allocation failed"))?;
        handles
            .try_reserve_exact(slots)
            .map_err(|_| invalid("native Q2 resident handle metadata allocation failed"))?;
        by_storage
            .try_reserve_exact(slots)
            .map_err(|_| invalid("native Q2 resident index metadata allocation failed"))?;
        let metadata_payload_bytes = projections
            .capacity()
            .checked_mul(size_of::<ResidentProjection>())
            .and_then(|bytes| {
                handles
                    .capacity()
                    .checked_mul(size_of::<NativeQ2WeightHandle>())
                    .and_then(|handle_bytes| bytes.checked_add(handle_bytes))
            })
            .and_then(|bytes| {
                by_storage
                    .capacity()
                    .checked_mul(size_of::<usize>())
                    .and_then(|index_bytes| bytes.checked_add(index_bytes))
            })
            .filter(|&bytes| bytes <= reservation)
            .ok_or_else(|| invalid("native Q2 resident metadata allocation exceeds reservation"))?;
        Ok(Self {
            projections,
            by_storage,
            handles,
            reservation,
            statistics: NativeQ2ResidencyStatistics {
                metadata_payload_bytes,
                ..NativeQ2ResidencyStatistics::default()
            },
        })
    }

    fn find(&self, tensor: &Tensor) -> Option<usize> {
        if !matches!(tensor.kind, PQ2_KIND | PTQ1_KIND) {
            return None;
        }
        let &[columns, rows] = tensor.dimensions.as_slice() else {
            return None;
        };
        let key = (tensor.data.as_ptr().addr(), tensor.kind, columns, rows);
        let position = self
            .by_storage
            .binary_search_by_key(&key, |&index| {
                let projection = &self.projections[index];
                (
                    projection.source.as_ptr().addr(),
                    projection.kind,
                    projection.columns,
                    projection.rows,
                )
            })
            .ok()?;
        let index = self.by_storage[position];
        let projection = &self.projections[index];
        (bonsai_prism_runtime::model::TensorBytes::ptr_eq(&projection.source, &tensor.data)
            && tensor.kind == projection.kind
            && tensor.dimensions.as_slice() == [projection.columns, projection.rows])
        .then_some(index)
    }

    const fn has_metadata_capacity(&self, operations: usize) -> bool {
        self.projections.len() < self.projections.capacity()
            && self.by_storage.len() < self.by_storage.capacity()
            && operations <= self.handles.capacity() - self.handles.len()
    }

    fn commit(&mut self, tensor: &Tensor, first_handle: usize, evaluated: bool) -> usize {
        let key = (
            tensor.data.as_ptr().addr(),
            tensor.kind,
            tensor.dimensions[0],
            tensor.dimensions[1],
        );
        let position = self
            .by_storage
            .binary_search_by_key(&key, |&index| {
                let projection = &self.projections[index];
                (
                    projection.source.as_ptr().addr(),
                    projection.kind,
                    projection.columns,
                    projection.rows,
                )
            })
            .unwrap_or_else(|pos| pos);
        let index = self.projections.len();
        self.projections.push(ResidentProjection {
            source: tensor.data.clone(),
            kind: tensor.kind,
            columns: tensor.dimensions[0],
            rows: tensor.dimensions[1],
            first_handle,
            evaluated,
            host_released: false,
        });
        self.by_storage.insert(position, index);
        self.statistics.tensors_prepared = self.statistics.tensors_prepared.saturating_add(1);
        index
    }

    fn rebind_released(
        &mut self,
        index: usize,
        released: bonsai_prism_runtime::model::TensorBytes,
    ) -> bool {
        let Some(projection) = self.projections.get(index) else {
            return false;
        };
        if projection.host_released || !released.is_empty() || released.is_mapped() {
            return false;
        }
        let old_key = (
            projection.source.as_ptr().addr(),
            projection.kind,
            projection.columns,
            projection.rows,
        );
        let new_key = (
            released.as_ptr().addr(),
            projection.kind,
            projection.columns,
            projection.rows,
        );
        if self
            .by_storage
            .binary_search_by_key(&new_key, |&existing| {
                let entry = &self.projections[existing];
                (
                    entry.source.as_ptr().addr(),
                    entry.kind,
                    entry.columns,
                    entry.rows,
                )
            })
            .is_ok()
        {
            return false;
        }
        let Ok(old_pos) = self.by_storage.binary_search_by_key(&old_key, |&existing| {
            let entry = &self.projections[existing];
            (
                entry.source.as_ptr().addr(),
                entry.kind,
                entry.columns,
                entry.rows,
            )
        }) else {
            return false;
        };
        if self.by_storage[old_pos] != index {
            return false;
        }
        self.by_storage.remove(old_pos);
        let projection = &mut self.projections[index];
        projection.source = released;
        projection.host_released = true;
        let Err(new_pos) = self.by_storage.binary_search_by_key(&new_key, |&existing| {
            let entry = &self.projections[existing];
            (
                entry.source.as_ptr().addr(),
                entry.kind,
                entry.columns,
                entry.rows,
            )
        }) else {
            return false;
        };
        self.by_storage.insert(new_pos, index);
        true
    }
}

// Split one bounded owned scratch allocation into disjoint native destinations.
fn token_outputs<const B: usize>(scratch: &mut [f32], rows: usize) -> [&mut [f32]; B] {
    let mut tail = scratch;
    std::array::from_fn(|_| {
        let (output, rest) = std::mem::take(&mut tail).split_at_mut(rows);
        tail = rest;
        output
    })
}

fn submit_resident_token_tile<S: NativeQ2Submitter>(
    submitter: &mut S,
    handle: &NativeQ2WeightHandle,
    activations: &[Vec<f32>],
    columns: std::ops::Range<usize>,
    scratch: &mut [f32],
    rows: usize,
) -> Result<()> {
    match activations.len() {
        8 => submitter.submit_resident_multi8(
            [handle; 8],
            std::array::from_fn(|token| &activations[token][columns.clone()]),
            token_outputs::<8>(scratch, rows),
        )?,
        4 => submitter.submit_resident_multi4(
            [handle; 4],
            std::array::from_fn(|token| &activations[token][columns.clone()]),
            token_outputs::<4>(scratch, rows),
        )?,
        3 => submitter.submit_resident_multi3(
            [handle; 3],
            std::array::from_fn(|token| &activations[token][columns.clone()]),
            token_outputs::<3>(scratch, rows),
        )?,
        2 => submitter.submit_resident_multi2(
            [handle; 2],
            std::array::from_fn(|token| &activations[token][columns.clone()]),
            token_outputs::<2>(scratch, rows),
        )?,
        1 => submitter.submit_resident(handle, &activations[0][columns], scratch)?,
        _ => return Err(invalid("invalid native Q2 resident token batch")),
    }
    Ok(())
}

/// Full-`K` 1-op native `HTP_TYPE_Q2_0` executor using `libggml-htp-v81.so`'s
/// signed 8-thread HVX `q2_0 x q8_1` (`HTP_MM_KERNEL_HVX_QUANT_ROW_N1`) kernel.
///
/// Resident cache misses pack directly into supporting submitters' admitted arenas;
/// copied preparation and nonresident execution allocate bounded packing scratch
/// only after selecting those paths. Conservative payload admission is unchanged.
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
    allowed_addr_range: (usize, usize),
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
            allowed_addr_range: (0, 0),
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

    /// Active HTP DSP architecture version reported by the underlying submitter (`81` or `73`).
    #[must_use]
    pub fn htp_architecture(&self) -> u32 {
        self.submitter.htp_architecture()
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
        weights: &mut Vec<u8>,
        evaluated: bool,
    ) -> Result<Option<usize>> {
        let Some(cache) = &mut self.residency else {
            return Ok(None);
        };
        if let Some(index) = cache.find(tensor) {
            return Ok(Some(index));
        }
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
                let packed_bytes = mapped_rows * 36 * chunk_blocks;
                let geometry = ChunkGeometry {
                    kind: tensor.kind,
                    total_blocks,
                    row_start,
                    active_rows,
                    block_start,
                    chunk_blocks,
                };
                let prepared = if self.submitter.supports_in_place_weight_preparation() {
                    let row_parallelism = self.row_parallelism.as_ref();
                    self.submitter.prepare_weights_in_place(
                        active_cols,
                        mapped_rows,
                        &mut |destination| {
                            if destination.len() != packed_bytes {
                                return Err(hexagon_native_rpc::Error::Contract);
                            }
                            pack_native_chunk(&tensor.data, geometry, destination, row_parallelism)
                                .map_err(|_| hexagon_native_rpc::Error::Contract)
                        },
                    )?
                } else {
                    resize_packed_scratch(weights, packed_bytes)?;
                    let chunk_weights = weights
                        .get_mut(..packed_bytes)
                        .ok_or_else(|| invalid("native Q2 preparation scratch extent mismatch"))?;
                    pack_native_chunk(
                        &tensor.data,
                        geometry,
                        chunk_weights,
                        self.row_parallelism.as_ref(),
                    )?;
                    self.submitter
                        .prepare_weights(active_cols, mapped_rows, chunk_weights)?
                };
                let handle = prepared.ok_or_else(|| {
                    invalid("native Q2 resident preparation contradicted capacity")
                })?;
                if handle.columns() != active_cols
                    || handle.rows() != mapped_rows
                    || handle.packed_bytes() != packed_bytes
                {
                    return Err(invalid(
                        "native Q2 resident preparation returned invalid geometry",
                    ));
                }
                cache.handles.push(handle);
                cache.statistics.weight_payload_bytes_prepared = cache
                    .statistics
                    .weight_payload_bytes_prepared
                    .saturating_add(u64::try_from(packed_bytes)?);
            }
        }
        Ok(Some(cache.commit(tensor, first_handle, evaluated)))
    }

    /// Pre-pack and stage a `PQ2_0` or `PTQ1_0` projection tensor into the
    /// session's resident arena before token evaluation so Token 0 flushes the
    /// resident weight arena only once on its first submission.
    /// Native sessions pack directly into fresh admitted resident extents, without
    /// a temporary packed-weight buffer or a subsequent host copy. Other submitters
    /// retain the bounded scratch-and-copy preparation contract.
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

    fn tensor_dimensions(&self, tensor: &Tensor) -> Result<(usize, usize)> {
        if let Some(cache) = &self.residency
            && let Some(index) = cache.find(tensor)
            && cache.projections[index].host_released
        {
            return native_dimensions_released(tensor);
        }
        native_dimensions(tensor)
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
        let (columns, rows) = self.tensor_dimensions(tensor)?;
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
            validate_source_scales(tensor)?;
        }
        let result = self.prepare_resident(tensor, weights, false);
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
        self.bind_allowed_tensors(selected);
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
            self.scratch_weights = weights;
        }
        Ok(())
    }

    /// Pre-warm as many leading `tensors` as fit in the opened resident weight
    /// arena(s), binding the executor's allowed tensor set strictly to that
    /// admitted prefix.
    ///
    /// Intended for `Hexagon v73` (`X1`) devices whose DSP/FastRPC secondary
    /// arena ceiling may admit a layer prefix rather than all 168 projections;
    /// unselected tail tensors return `false` from
    /// [`Pq2ProjectionExecutor::supports_f32`] and fall back cleanly to CPU or
    /// GPU execution.
    ///
    /// # Errors
    /// Rejects empty selections, already-bound executors, unsupported tensors
    /// before reaching capacity, or an arena too small to hold even one tensor.
    pub fn select_prefix_and_prewarm_tensors(&mut self, tensors: &[Tensor]) -> Result<usize> {
        if tensors.is_empty() || self.allowed_tensors.is_some() || self.failed {
            return Err(invalid("invalid native Q2 selected tensor configuration"));
        }
        let mut selected = Vec::new();
        if self.residency.is_some() {
            let mut weights = std::mem::take(&mut self.scratch_weights);
            for tensor in tensors {
                if !self.supports_f32(tensor) {
                    self.scratch_weights = weights;
                    return Err(invalid("unsupported native Q2 selected projection tensor"));
                }
                if !self.prewarm_tensor_with_scratch(tensor, &mut weights)? {
                    break;
                }
                selected
                    .try_reserve(1)
                    .map_err(|_| invalid("native Q2 selected tensor metadata allocation failed"))?;
                selected.push(tensor.clone());
            }
            self.scratch_weights = weights;
        } else {
            selected
                .try_reserve_exact(tensors.len())
                .map_err(|_| invalid("native Q2 selected tensor metadata allocation failed"))?;
            for tensor in tensors {
                if !self.supports_f32(tensor) {
                    return Err(invalid("unsupported native Q2 selected projection tensor"));
                }
                selected.push(tensor.clone());
            }
        }
        let prepared = selected.len();
        if prepared == 0 {
            self.failed = true;
            return Err(invalid(
                "native Q2 resident arena exhausted before preparing any tensor",
            ));
        }
        self.bind_allowed_tensors(selected);
        Ok(prepared)
    }

    fn bind_allowed_tensors(&mut self, mut selected: Vec<Tensor>) {
        selected.sort_by(|left, right| selected_tensor_key(left).cmp(&selected_tensor_key(right)));
        let min_addr = selected
            .first()
            .map_or(0, |tensor| tensor.data.as_ptr().addr());
        let max_addr = selected
            .last()
            .map_or(0, |tensor| tensor.data.as_ptr().addr());
        self.allowed_addr_range = (min_addr, max_addr);
        self.allowed_tensors = Some(selected);
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
        if output.len() != rows {
            output.clear();
            output
                .try_reserve(rows)
                .map_err(|_| invalid("native Q2 output allocation failed"))?;
            output.resize(rows, f32::NAN);
        }
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
            let destination = (output.as_ptr(), output.len());
            self.submitter
                .submit_resident_validated(handle, activation, output.as_mut_slice())?
                .validate_destinations([destination])?;
        } else {
            pack_native_chunk(
                &tensor.data,
                ChunkGeometry {
                    kind: tensor.kind,
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
            finite_slice(output)?;
        }
        self.statistics.operations_completed =
            self.statistics.operations_completed.saturating_add(1);
        self.queue_writes_completed = self.queue_writes_completed.saturating_add(1);
        self.statistics.projections_completed =
            self.statistics.projections_completed.saturating_add(1);
        Ok(())
    }

    #[expect(
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
        let (columns, rows) = self.tensor_dimensions(tensor)?;
        if !self.supports_f32(tensor) || activation.len() != columns {
            return Err(invalid("unsupported native Q2 Hexagon projection contract"));
        }
        let single_op =
            columns <= self.maximum_columns && rows <= self.maximum_rows && rows.is_multiple_of(32);
        let cached = self.residency.as_ref().and_then(|cache| cache.find(tensor));
        if cached.is_none() && !single_op {
            validate_source_scales(tensor)?;
        }
        finite_slice(activation)?;
        let max_chunk_cols = columns.min(self.maximum_columns);
        let max_mapped_rows = rows.min(self.maximum_rows).next_multiple_of(32);
        if cached.is_some() {
            self.scratch_weights = Vec::new();
        }
        let mut weights = Vec::new();
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
        if resident.is_none() {
            resize_packed_scratch(&mut weights, max_mapped_rows * 36 * (max_chunk_cols / 128))?;
        }
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
                        kind: tensor.kind,
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

    fn validate_f32_group(
        &self,
        tensors: &[&Tensor],
    ) -> Option<(usize, [usize; 3], [Option<usize>; 3])> {
        if self.failed || !matches!(tensors.len(), 2 | 3) {
            return None;
        }
        let cache = self.residency.as_ref()?;
        let (columns, _) = self.tensor_dimensions(tensors[0]).ok()?;
        if columns > self.maximum_columns {
            return None;
        }
        let mut total_output_bytes = 0_usize;
        let mut uncached_packed_bytes = 0_usize;
        let mut uncached_ops = 0_usize;
        let mut rows_per_member = [0_usize; 3];
        let mut cached_indices = [None; 3];
        for (member, &tensor) in tensors.iter().enumerate() {
            if !self.supports_f32(tensor) {
                return None;
            }
            let (cols, rows) = self.tensor_dimensions(tensor).ok()?;
            if cols != columns || rows > self.maximum_rows || !rows.is_multiple_of(32) {
                return None;
            }
            rows_per_member[member] = rows;
            let out_bytes = rows.checked_mul(4)?;
            total_output_bytes = total_output_bytes.checked_add(out_bytes)?;
            let cached = cache.find(tensor);
            cached_indices[member] = cached;
            if cached.is_none() {
                let packed = rows
                    .checked_mul(36)
                    .and_then(|b| b.checked_mul(columns / 128))?;
                uncached_packed_bytes = uncached_packed_bytes.checked_add(packed)?;
                uncached_ops += 1;
            }
        }
        if uncached_ops > 0
            && (!cache.has_metadata_capacity(uncached_ops)
                || uncached_packed_bytes > self.submitter.residency_capacity_bytes())
        {
            return None;
        }
        total_output_bytes
            .checked_add(cache.reservation)
            .filter(|&bytes| bytes <= self.maximum_owned_payload_bytes)
            .map(|_| (columns, rows_per_member, cached_indices))
    }

    fn supports_f32_group_inner(&self, tensors: &[&Tensor]) -> bool {
        self.validate_f32_group(tensors).is_some()
    }

    #[expect(
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
        if tensors.len() != outputs.len() {
            return Err(invalid(
                "unsupported native Q2 Hexagon batched projection group",
            ));
        }
        let Some((columns, rows_per_member, cached_indices)) = self.validate_f32_group(tensors)
        else {
            return Err(invalid(
                "unsupported native Q2 Hexagon batched projection group",
            ));
        };
        if activation.len() != columns {
            return Err(invalid("unsupported native Q2 Hexagon projection contract"));
        }
        finite_slice(activation)?;
        let mut weights = Vec::new();
        let mut indices = [0_usize; 3];
        for (member, &tensor) in tensors.iter().enumerate() {
            let index = if let Some(idx) = cached_indices[member] {
                if let Some(cache) = &mut self.residency {
                    if cache.projections[idx].evaluated {
                        cache.statistics.cache_hits = cache.statistics.cache_hits.saturating_add(1);
                    } else {
                        cache.projections[idx].evaluated = true;
                    }
                }
                idx
            } else {
                self.prepare_resident(tensor, &mut weights, true)?
                    .ok_or_else(|| invalid("native Q2 resident preparation failed"))?
            };
            indices[member] = index;
        }
        for (member, output) in outputs.iter_mut().enumerate() {
            let rows = rows_per_member[member];
            if output.len() != rows {
                output.clear();
                output
                    .try_reserve(rows)
                    .map_err(|_| invalid("native Q2 output allocation failed"))?;
                output.resize(rows, f32::NAN);
            }
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
                let destinations = [(out0.as_ptr(), out0.len()), (out1.as_ptr(), out1.len())];
                self.submitter
                    .submit_resident_batch2_validated(
                        [h0, h1],
                        activation,
                        [out0.as_mut_slice(), out1.as_mut_slice()],
                    )?
                    .validate_destinations(destinations)?;
            }
            [out0, out1, out2] => {
                let h0 = &cache.handles[cache.projections[indices[0]].first_handle];
                let h1 = &cache.handles[cache.projections[indices[1]].first_handle];
                let h2 = &cache.handles[cache.projections[indices[2]].first_handle];
                let destinations = [
                    (out0.as_ptr(), out0.len()),
                    (out1.as_ptr(), out1.len()),
                    (out2.as_ptr(), out2.len()),
                ];
                self.submitter
                    .submit_resident_batch3_validated(
                        [h0, h1, h2],
                        activation,
                        [
                            out0.as_mut_slice(),
                            out1.as_mut_slice(),
                            out2.as_mut_slice(),
                        ],
                    )?
                    .validate_destinations(destinations)?;
            }
            _ => return Err(invalid("unsupported native Q2 batch size")),
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

    #[expect(
        clippy::too_many_lines,
        reason = "Keep multi-token 4/3/2/1-op resident dispatch and fallback together."
    )]
    fn project_f32_tokens_inner(
        &mut self,
        tensor: &Tensor,
        activations: &[Vec<f32>],
        outputs: &mut [Vec<f32>],
    ) -> Result<()> {
        if self.failed {
            return Err(invalid(
                "native Q2 Hexagon executor requires terminal teardown",
            ));
        }
        if activations.len() != outputs.len() {
            return Err(invalid("mismatched F32 token projection outputs"));
        }
        if activations.is_empty() {
            return Ok(());
        }
        let (columns, rows) = self.tensor_dimensions(tensor)?;
        if !self.supports_f32(tensor) {
            return Err(invalid("unsupported native Q2 Hexagon projection contract"));
        }
        let single_op =
            columns <= self.maximum_columns && rows <= self.maximum_rows && rows.is_multiple_of(32);
        let cached = self.residency.as_ref().and_then(|cache| cache.find(tensor));
        if !single_op {
            return self.project_multitile_tokens(
                tensor,
                (columns, rows),
                cached,
                activations,
                outputs,
            );
        }
        if cached.is_none() && self.residency.is_none() {
            return self.project_transient_token_tile(
                tensor,
                (columns, rows),
                activations,
                outputs,
            );
        }
        for activation in activations {
            if activation.len() != columns {
                return Err(invalid("unsupported native Q2 Hexagon projection contract"));
            }
            finite_slice(activation)?;
        }
        let resident_index = if let Some(idx) = cached {
            idx
        } else {
            let mut weights = Vec::new();
            let Some(idx) = self.prepare_resident(tensor, &mut weights, false)? else {
                return self.project_transient_token_tile(
                    tensor,
                    (columns, rows),
                    activations,
                    outputs,
                );
            };
            idx
        };
        if let Some(cache) = &mut self.residency {
            let hits = if cache.projections[resident_index].evaluated {
                u64::try_from(activations.len())?
            } else {
                cache.projections[resident_index].evaluated = true;
                u64::try_from(activations.len().saturating_sub(1))?
            };
            cache.statistics.cache_hits = cache.statistics.cache_hits.saturating_add(hits);
        }
        for output in outputs.iter_mut() {
            if output.len() != rows {
                output.clear();
                output
                    .try_reserve(rows)
                    .map_err(|_| invalid("native Q2 output allocation failed"))?;
                output.resize(rows, f32::NAN);
            }
        }
        let cache = self
            .residency
            .as_ref()
            .ok_or_else(|| invalid("native Q2 resident cache is unavailable"))?;
        let handle = &cache.handles[cache.projections[resident_index].first_handle];
        let mut rem_act = activations;
        let mut rem_out = outputs;
        while !rem_act.is_empty() {
            let chunk_len = if rem_act.len() >= 8 {
                8
            } else {
                rem_act.len().min(4)
            };
            let (act_chunk, next_act) = rem_act.split_at(chunk_len);
            let (out_chunk, next_out) = rem_out.split_at_mut(chunk_len);
            rem_act = next_act;
            rem_out = next_out;
            let chunk_u64 = u64::try_from(chunk_len)?;
            self.statistics.operations_attempted = self
                .statistics
                .operations_attempted
                .saturating_add(chunk_u64);
            self.queue_writes_attempted = self.queue_writes_attempted.saturating_add(1);
            match out_chunk {
                [o0, o1, o2, o3, o4, o5, o6, o7] => {
                    self.submitter.submit_resident_multi8(
                        [
                            handle, handle, handle, handle, handle, handle, handle, handle,
                        ],
                        [
                            act_chunk[0].as_slice(),
                            act_chunk[1].as_slice(),
                            act_chunk[2].as_slice(),
                            act_chunk[3].as_slice(),
                            act_chunk[4].as_slice(),
                            act_chunk[5].as_slice(),
                            act_chunk[6].as_slice(),
                            act_chunk[7].as_slice(),
                        ],
                        [
                            o0.as_mut_slice(),
                            o1.as_mut_slice(),
                            o2.as_mut_slice(),
                            o3.as_mut_slice(),
                            o4.as_mut_slice(),
                            o5.as_mut_slice(),
                            o6.as_mut_slice(),
                            o7.as_mut_slice(),
                        ],
                    )?;
                }
                [o0, o1, o2, o3] => {
                    self.submitter.submit_resident_multi4(
                        [handle, handle, handle, handle],
                        [
                            act_chunk[0].as_slice(),
                            act_chunk[1].as_slice(),
                            act_chunk[2].as_slice(),
                            act_chunk[3].as_slice(),
                        ],
                        [
                            o0.as_mut_slice(),
                            o1.as_mut_slice(),
                            o2.as_mut_slice(),
                            o3.as_mut_slice(),
                        ],
                    )?;
                }
                [o0, o1, o2] => {
                    self.submitter.submit_resident_multi3(
                        [handle, handle, handle],
                        [
                            act_chunk[0].as_slice(),
                            act_chunk[1].as_slice(),
                            act_chunk[2].as_slice(),
                        ],
                        [o0.as_mut_slice(), o1.as_mut_slice(), o2.as_mut_slice()],
                    )?;
                }
                [o0, o1] => {
                    self.submitter.submit_resident_multi2(
                        [handle, handle],
                        [act_chunk[0].as_slice(), act_chunk[1].as_slice()],
                        [o0.as_mut_slice(), o1.as_mut_slice()],
                    )?;
                }
                [o0] => {
                    self.submitter.submit_resident(
                        handle,
                        act_chunk[0].as_slice(),
                        o0.as_mut_slice(),
                    )?;
                }
                _ => return Err(invalid("unexpected token batch chunk size")),
            }
            for output in out_chunk.iter() {
                finite_slice(output)?;
            }
            self.statistics.operations_completed = self
                .statistics
                .operations_completed
                .saturating_add(chunk_u64);
            self.queue_writes_completed = self.queue_writes_completed.saturating_add(1);
            self.statistics.projections_completed = self
                .statistics
                .projections_completed
                .saturating_add(chunk_u64);
        }
        Ok(())
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Keep bounded tile scratch, immutable weight reuse and transactional token accumulation together"
    )]
    fn project_multitile_tokens(
        &mut self,
        tensor: &Tensor,
        (columns, rows): (usize, usize),
        cached: Option<usize>,
        activations: &[Vec<f32>],
        outputs: &mut [Vec<f32>],
    ) -> Result<()> {
        if activations[0].len() != columns {
            return Err(invalid("unsupported native Q2 Hexagon projection contract"));
        }
        // Retain the scalar tiled path's first-shape/scale/finite precedence.
        // No token is submitted before the complete tile's inputs are valid.
        if cached.is_none() {
            validate_source_scales(tensor)?;
        }
        for activation in activations {
            if activation.len() != columns {
                return Err(invalid("unsupported native Q2 Hexagon projection contract"));
            }
            finite_slice(activation)?;
        }
        let mut weights = Vec::new();
        let resident = if let Some(index) = cached {
            self.scratch_weights = Vec::new();
            Some(index)
        } else {
            self.prepare_resident(tensor, &mut weights, false)?
        };
        if let Some(index) = resident
            && let Some(cache) = &mut self.residency
        {
            let hits = if cache.projections[index].evaluated {
                activations.len()
            } else {
                cache.projections[index].evaluated = true;
                activations.len() - 1
            };
            cache.statistics.cache_hits = cache
                .statistics
                .cache_hits
                .saturating_add(u64::try_from(hits)?);
        }
        let max_mapped_rows = rows.min(self.maximum_rows).next_multiple_of(32);
        if resident.is_none() {
            resize_packed_scratch(
                &mut weights,
                max_mapped_rows * 36 * (columns.min(self.maximum_columns) / 128),
            )?;
        }
        // Resident tiles need no packing buffer after preparation. Eight
        // padded outputs cost 32 bytes/row, within the existing reservation
        // of at least 36 packed-weight bytes plus four output bytes per row.
        // Keep that full conservative reservation; do not lower admission.
        let batch_limit = if resident.is_some() {
            self.scratch_weights = Vec::new();
            // Release packing storage before charging the batched output buffer.
            drop(std::mem::take(&mut weights));
            let limit = self.submitter.maximum_resident_token_batch();
            if matches!(limit, 2..=4 | 8) { limit } else { 1 }
        } else {
            1
        };
        let scratch_tokens = batch_limit.min(activations.len());
        let mut chunk_output = zeroed(max_mapped_rows * scratch_tokens, f32::NAN)?;
        for output in outputs.iter_mut() {
            if output.len() == rows {
                output.fill(0.0);
            } else {
                output.clear();
                output
                    .try_reserve(rows)
                    .map_err(|_| invalid("native Q2 output allocation failed"))?;
                output.resize(rows, 0.0);
            }
        }
        let mut operation = 0;
        for row_start in (0..rows).step_by(self.maximum_rows) {
            let active_rows = (rows - row_start).min(self.maximum_rows);
            let mapped_rows = active_rows.next_multiple_of(32);
            for col_start in (0..columns).step_by(self.maximum_columns) {
                let active_cols = (columns - col_start).min(self.maximum_columns);
                let chunk_blocks = active_cols / 128;
                let packed_bytes = mapped_rows * 36 * chunk_blocks;
                if resident.is_none() {
                    pack_native_chunk(
                        &tensor.data,
                        ChunkGeometry {
                            kind: tensor.kind,
                            total_blocks: columns / 128,
                            row_start,
                            active_rows,
                            block_start: col_start / 128,
                            chunk_blocks,
                        },
                        &mut weights[..packed_bytes],
                        self.row_parallelism.as_ref(),
                    )?;
                }
                // Each token retains its original tile and addition order;
                // a resident packet shares one response across bounded tokens.
                let mut token = 0;
                while token < activations.len() {
                    let remaining = activations.len() - token;
                    let batch = if remaining >= 8 && batch_limit == 8 {
                        8
                    } else {
                        remaining.min(batch_limit.min(4))
                    };
                    let out_slice = &mut chunk_output[..batch * mapped_rows];
                    out_slice.fill(f32::NAN);
                    let batch_u64 = u64::try_from(batch)?;
                    self.statistics.operations_attempted = self
                        .statistics
                        .operations_attempted
                        .saturating_add(batch_u64);
                    self.queue_writes_attempted = self.queue_writes_attempted.saturating_add(1);
                    if let Some(index) = resident {
                        let cache = self
                            .residency
                            .as_ref()
                            .ok_or_else(|| invalid("native Q2 resident cache is unavailable"))?;
                        let handle =
                            &cache.handles[cache.projections[index].first_handle + operation];
                        submit_resident_token_tile(
                            &mut self.submitter,
                            handle,
                            &activations[token..token + batch],
                            col_start..col_start + active_cols,
                            out_slice,
                            mapped_rows,
                        )?;
                    } else {
                        self.submitter.submit(
                            active_cols,
                            mapped_rows,
                            &weights[..packed_bytes],
                            &activations[token][col_start..col_start + active_cols],
                            out_slice,
                        )?;
                    }
                    // Validate and accumulate every member, including padding,
                    // before successful packet counters. The complete call's
                    // outputs remain provisional until the outer publication.
                    for (output, partial) in outputs[token..token + batch]
                        .iter_mut()
                        .zip(out_slice.chunks_exact(mapped_rows))
                    {
                        accumulate_tile_output(
                            &mut output[row_start..row_start + active_rows],
                            partial,
                        )?;
                    }
                    self.statistics.operations_completed = self
                        .statistics
                        .operations_completed
                        .saturating_add(batch_u64);
                    self.queue_writes_completed = self.queue_writes_completed.saturating_add(1);
                    if row_start + active_rows == rows && col_start + active_cols == columns {
                        self.statistics.projections_completed = self
                            .statistics
                            .projections_completed
                            .saturating_add(batch_u64);
                    }
                    token += batch;
                }
                operation += 1;
            }
        }
        Ok(())
    }

    fn project_transient_token_tile(
        &mut self,
        tensor: &Tensor,
        (columns, rows): (usize, usize),
        activations: &[Vec<f32>],
        outputs: &mut [Vec<f32>],
    ) -> Result<()> {
        // Keep the existing per-token input/allocation/submit ordering. Only
        // immutable packing is shared, and its buffer lives for this call.
        let mut weights = Vec::new();
        for (token, (activation, output)) in activations.iter().zip(outputs).enumerate() {
            if activation.len() != columns {
                return Err(invalid("unsupported native Q2 Hexagon projection contract"));
            }
            finite_slice(activation)?;
            if token == 0 {
                resize_packed_scratch(&mut weights, rows * 36 * (columns / 128))?;
            }
            if output.len() != rows {
                output.clear();
                output
                    .try_reserve(rows)
                    .map_err(|_| invalid("native Q2 output allocation failed"))?;
                output.resize(rows, f32::NAN);
            }
            if token == 0 {
                pack_native_chunk(
                    &tensor.data,
                    ChunkGeometry {
                        kind: tensor.kind,
                        total_blocks: columns / 128,
                        row_start: 0,
                        active_rows: rows,
                        block_start: 0,
                        chunk_blocks: columns / 128,
                    },
                    &mut weights,
                    self.row_parallelism.as_ref(),
                )?;
            }
            self.statistics.operations_attempted =
                self.statistics.operations_attempted.saturating_add(1);
            self.queue_writes_attempted = self.queue_writes_attempted.saturating_add(1);
            self.submitter
                .submit(columns, rows, &weights, activation, output.as_mut_slice())?;
            finite_slice(output)?;
            self.statistics.operations_completed =
                self.statistics.operations_completed.saturating_add(1);
            self.queue_writes_completed = self.queue_writes_completed.saturating_add(1);
            self.statistics.projections_completed =
                self.statistics.projections_completed.saturating_add(1);
        }
        Ok(())
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Keep 4-token sibling pair 8-op resident dispatch and partial-tail handling together."
    )]
    fn project_f32_pair_tokens_inner(
        &mut self,
        tensors: [&Tensor; 2],
        activations: &[Vec<f32>],
        outputs_a: &mut [Vec<f32>],
        outputs_b: &mut [Vec<f32>],
    ) -> Result<()> {
        if self.failed {
            return Err(invalid(
                "native Q2 Hexagon executor requires terminal teardown",
            ));
        }
        if activations.len() != outputs_a.len() || activations.len() != outputs_b.len() {
            return Err(invalid("mismatched F32 pair token projection outputs"));
        }
        if activations.is_empty() {
            return Ok(());
        }
        let Some((columns, rows_per_member, cached_indices)) = self.validate_f32_group(&tensors)
        else {
            return Err(invalid(
                "unsupported native Q2 Hexagon batched projection group",
            ));
        };
        for activation in activations {
            if activation.len() != columns {
                return Err(invalid("unsupported native Q2 Hexagon projection contract"));
            }
            finite_slice(activation)?;
        }
        let mut weights = Vec::new();
        for (member, &tensor) in tensors.iter().enumerate() {
            if cached_indices[member].is_none() {
                self.prepare_resident(tensor, &mut weights, false)?
                    .ok_or_else(|| invalid("native Q2 resident preparation failed"))?;
            }
        }
        let mut indices = [0_usize; 2];
        if let Some(cache) = &mut self.residency {
            for (member, &tensor) in tensors.iter().enumerate() {
                let idx = cache
                    .find(tensor)
                    .ok_or_else(|| invalid("native Q2 resident preparation failed"))?;
                let hits = if cache.projections[idx].evaluated {
                    u64::try_from(activations.len())?
                } else {
                    cache.projections[idx].evaluated = true;
                    u64::try_from(activations.len().saturating_sub(1))?
                };
                cache.statistics.cache_hits = cache.statistics.cache_hits.saturating_add(hits);
                indices[member] = idx;
            }
        }
        for output in outputs_a.iter_mut() {
            let rows = rows_per_member[0];
            if output.len() != rows {
                output.clear();
                output
                    .try_reserve(rows)
                    .map_err(|_| invalid("native Q2 output allocation failed"))?;
                output.resize(rows, f32::NAN);
            }
        }
        for output in outputs_b.iter_mut() {
            let rows = rows_per_member[1];
            if output.len() != rows {
                output.clear();
                output
                    .try_reserve(rows)
                    .map_err(|_| invalid("native Q2 output allocation failed"))?;
                output.resize(rows, f32::NAN);
            }
        }
        let cache = self
            .residency
            .as_ref()
            .ok_or_else(|| invalid("native Q2 resident cache is unavailable"))?;
        let h0 = &cache.handles[cache.projections[indices[0]].first_handle];
        let h1 = &cache.handles[cache.projections[indices[1]].first_handle];
        let mut rem_act = activations;
        let mut rem_a = outputs_a;
        let mut rem_b = outputs_b;
        while !rem_act.is_empty() {
            if rem_act.len() >= 4 {
                let (act_chunk, next_act) = rem_act.split_at(4);
                let (a_chunk, next_a) = rem_a.split_at_mut(4);
                let (b_chunk, next_b) = rem_b.split_at_mut(4);
                rem_act = next_act;
                rem_a = next_a;
                rem_b = next_b;
                self.statistics.operations_attempted =
                    self.statistics.operations_attempted.saturating_add(8);
                self.queue_writes_attempted = self.queue_writes_attempted.saturating_add(1);
                let [oa0, oa1, oa2, oa3] = a_chunk else {
                    return Err(invalid("unexpected pair token slice"));
                };
                let [ob0, ob1, ob2, ob3] = b_chunk else {
                    return Err(invalid("unexpected pair token slice"));
                };
                let a0 = act_chunk[0].as_slice();
                let a1 = act_chunk[1].as_slice();
                let a2 = act_chunk[2].as_slice();
                let a3 = act_chunk[3].as_slice();
                self.submitter.submit_resident_multi8(
                    [h0, h1, h0, h1, h0, h1, h0, h1],
                    [a0, a0, a1, a1, a2, a2, a3, a3],
                    [
                        oa0.as_mut_slice(),
                        ob0.as_mut_slice(),
                        oa1.as_mut_slice(),
                        ob1.as_mut_slice(),
                        oa2.as_mut_slice(),
                        ob2.as_mut_slice(),
                        oa3.as_mut_slice(),
                        ob3.as_mut_slice(),
                    ],
                )?;
                for out in [oa0, ob0, oa1, ob1, oa2, ob2, oa3, ob3] {
                    finite_slice(out)?;
                }
                self.statistics.operations_completed =
                    self.statistics.operations_completed.saturating_add(8);
                self.queue_writes_completed = self.queue_writes_completed.saturating_add(1);
                self.statistics.projections_completed =
                    self.statistics.projections_completed.saturating_add(8);
            } else if rem_act.len() >= 2 {
                let (act_chunk, next_act) = rem_act.split_at(2);
                let (a_chunk, next_a) = rem_a.split_at_mut(2);
                let (b_chunk, next_b) = rem_b.split_at_mut(2);
                rem_act = next_act;
                rem_a = next_a;
                rem_b = next_b;
                self.statistics.operations_attempted =
                    self.statistics.operations_attempted.saturating_add(4);
                self.queue_writes_attempted = self.queue_writes_attempted.saturating_add(1);
                let [oa0, oa1] = a_chunk else {
                    return Err(invalid("unexpected pair token slice"));
                };
                let [ob0, ob1] = b_chunk else {
                    return Err(invalid("unexpected pair token slice"));
                };
                let a0 = act_chunk[0].as_slice();
                let a1 = act_chunk[1].as_slice();
                self.submitter.submit_resident_multi4(
                    [h0, h1, h0, h1],
                    [a0, a0, a1, a1],
                    [
                        oa0.as_mut_slice(),
                        ob0.as_mut_slice(),
                        oa1.as_mut_slice(),
                        ob1.as_mut_slice(),
                    ],
                )?;
                finite_slice(oa0)?;
                finite_slice(ob0)?;
                finite_slice(oa1)?;
                finite_slice(ob1)?;
                self.statistics.operations_completed =
                    self.statistics.operations_completed.saturating_add(4);
                self.queue_writes_completed = self.queue_writes_completed.saturating_add(1);
                self.statistics.projections_completed =
                    self.statistics.projections_completed.saturating_add(4);
            } else {
                let a0 = rem_act[0].as_slice();
                let oa0 = &mut rem_a[0];
                let ob0 = &mut rem_b[0];
                rem_act = &[];
                rem_a = &mut [];
                rem_b = &mut [];
                self.statistics.operations_attempted =
                    self.statistics.operations_attempted.saturating_add(2);
                self.queue_writes_attempted = self.queue_writes_attempted.saturating_add(1);
                self.submitter.submit_resident_batch2(
                    [h0, h1],
                    a0,
                    [oa0.as_mut_slice(), ob0.as_mut_slice()],
                )?;
                finite_slice(oa0)?;
                finite_slice(ob0)?;
                self.statistics.operations_completed =
                    self.statistics.operations_completed.saturating_add(2);
                self.queue_writes_completed = self.queue_writes_completed.saturating_add(1);
                self.statistics.projections_completed =
                    self.statistics.projections_completed.saturating_add(2);
            }
        }
        Ok(())
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Keep 8/4/2-token sibling triple resident dispatch and partial-tail handling together."
    )]
    fn project_f32_triple_tokens_inner(
        &mut self,
        tensors: [&Tensor; 3],
        activations: &[Vec<f32>],
        outputs_a: &mut [Vec<f32>],
        outputs_b: &mut [Vec<f32>],
        outputs_c: &mut [Vec<f32>],
    ) -> Result<()> {
        if self.failed {
            return Err(invalid(
                "native Q2 Hexagon executor requires terminal teardown",
            ));
        }
        if activations.len() != outputs_a.len()
            || activations.len() != outputs_b.len()
            || activations.len() != outputs_c.len()
        {
            return Err(invalid("mismatched F32 triple token projection outputs"));
        }
        if activations.is_empty() {
            return Ok(());
        }
        let Some((columns, rows_per_member, cached_indices)) = self.validate_f32_group(&tensors)
        else {
            return Err(invalid(
                "unsupported native Q2 Hexagon batched projection group",
            ));
        };
        for activation in activations {
            if activation.len() != columns {
                return Err(invalid("unsupported native Q2 Hexagon projection contract"));
            }
            finite_slice(activation)?;
        }
        let mut weights = Vec::new();
        for (member, &tensor) in tensors.iter().enumerate() {
            if cached_indices[member].is_none() {
                self.prepare_resident(tensor, &mut weights, false)?
                    .ok_or_else(|| invalid("native Q2 resident preparation failed"))?;
            }
        }
        let mut indices = [0_usize; 3];
        if let Some(cache) = &mut self.residency {
            for (member, &tensor) in tensors.iter().enumerate() {
                let idx = cache
                    .find(tensor)
                    .ok_or_else(|| invalid("native Q2 resident preparation failed"))?;
                let hits = if cache.projections[idx].evaluated {
                    u64::try_from(activations.len())?
                } else {
                    cache.projections[idx].evaluated = true;
                    u64::try_from(activations.len().saturating_sub(1))?
                };
                cache.statistics.cache_hits = cache.statistics.cache_hits.saturating_add(hits);
                indices[member] = idx;
            }
        }
        for (member, outputs) in [&mut *outputs_a, &mut *outputs_b, &mut *outputs_c]
            .into_iter()
            .enumerate()
        {
            let rows = rows_per_member[member];
            for output in outputs.iter_mut() {
                if output.len() != rows {
                    output.clear();
                    output
                        .try_reserve(rows)
                        .map_err(|_| invalid("native Q2 output allocation failed"))?;
                    output.resize(rows, f32::NAN);
                }
            }
        }
        let cache = self
            .residency
            .as_ref()
            .ok_or_else(|| invalid("native Q2 resident cache is unavailable"))?;
        let h0 = &cache.handles[cache.projections[indices[0]].first_handle];
        let h1 = &cache.handles[cache.projections[indices[1]].first_handle];
        let h2 = &cache.handles[cache.projections[indices[2]].first_handle];
        let mut rem_act = activations;
        let mut rem_a = outputs_a;
        let mut rem_b = outputs_b;
        let mut rem_c = outputs_c;
        while !rem_act.is_empty() {
            if rem_act.len() >= 8 {
                let (act_chunk, next_act) = rem_act.split_at(8);
                let (a_chunk, next_a) = rem_a.split_at_mut(8);
                let (b_chunk, next_b) = rem_b.split_at_mut(8);
                let (c_chunk, next_c) = rem_c.split_at_mut(8);
                rem_act = next_act;
                rem_a = next_a;
                rem_b = next_b;
                rem_c = next_c;
                let [oa0, oa1, oa2, oa3, oa4, oa5, oa6, oa7] = a_chunk else {
                    return Err(invalid("unexpected triple token slice"));
                };
                let [ob0, ob1, ob2, ob3, ob4, ob5, ob6, ob7] = b_chunk else {
                    return Err(invalid("unexpected triple token slice"));
                };
                let [oc0, oc1, oc2, oc3, oc4, oc5, oc6, oc7] = c_chunk else {
                    return Err(invalid("unexpected triple token slice"));
                };
                let x = [
                    act_chunk[0].as_slice(),
                    act_chunk[1].as_slice(),
                    act_chunk[2].as_slice(),
                    act_chunk[3].as_slice(),
                    act_chunk[4].as_slice(),
                    act_chunk[5].as_slice(),
                    act_chunk[6].as_slice(),
                    act_chunk[7].as_slice(),
                ];
                for (handles, acts, outs) in [
                    (
                        [h0, h1, h2, h0, h1, h2, h0, h1],
                        [x[0], x[0], x[0], x[1], x[1], x[1], x[2], x[2]],
                        [
                            oa0.as_mut_slice(),
                            ob0.as_mut_slice(),
                            oc0.as_mut_slice(),
                            oa1.as_mut_slice(),
                            ob1.as_mut_slice(),
                            oc1.as_mut_slice(),
                            oa2.as_mut_slice(),
                            ob2.as_mut_slice(),
                        ],
                    ),
                    (
                        [h2, h0, h1, h2, h0, h1, h2, h0],
                        [x[2], x[3], x[3], x[3], x[4], x[4], x[4], x[5]],
                        [
                            oc2.as_mut_slice(),
                            oa3.as_mut_slice(),
                            ob3.as_mut_slice(),
                            oc3.as_mut_slice(),
                            oa4.as_mut_slice(),
                            ob4.as_mut_slice(),
                            oc4.as_mut_slice(),
                            oa5.as_mut_slice(),
                        ],
                    ),
                    (
                        [h1, h2, h0, h1, h2, h0, h1, h2],
                        [x[5], x[5], x[6], x[6], x[6], x[7], x[7], x[7]],
                        [
                            ob5.as_mut_slice(),
                            oc5.as_mut_slice(),
                            oa6.as_mut_slice(),
                            ob6.as_mut_slice(),
                            oc6.as_mut_slice(),
                            oa7.as_mut_slice(),
                            ob7.as_mut_slice(),
                            oc7.as_mut_slice(),
                        ],
                    ),
                ] {
                    self.statistics.operations_attempted =
                        self.statistics.operations_attempted.saturating_add(8);
                    self.queue_writes_attempted = self.queue_writes_attempted.saturating_add(1);
                    let [o0, o1, o2, o3, o4, o5, o6, o7] = outs;
                    self.submitter.submit_resident_multi8(
                        handles,
                        acts,
                        [o0, o1, o2, o3, o4, o5, o6, o7],
                    )?;
                    for out in [&*o0, &*o1, &*o2, &*o3, &*o4, &*o5, &*o6, &*o7] {
                        finite_slice(out)?;
                    }
                    self.statistics.operations_completed =
                        self.statistics.operations_completed.saturating_add(8);
                    self.queue_writes_completed = self.queue_writes_completed.saturating_add(1);
                    self.statistics.projections_completed =
                        self.statistics.projections_completed.saturating_add(8);
                }
            } else if rem_act.len() >= 4 {
                let (act_chunk, next_act) = rem_act.split_at(4);
                let (a_chunk, next_a) = rem_a.split_at_mut(4);
                let (b_chunk, next_b) = rem_b.split_at_mut(4);
                let (c_chunk, next_c) = rem_c.split_at_mut(4);
                rem_act = next_act;
                rem_a = next_a;
                rem_b = next_b;
                rem_c = next_c;
                let [oa0, oa1, oa2, oa3] = a_chunk else {
                    return Err(invalid("unexpected triple token slice"));
                };
                let [ob0, ob1, ob2, ob3] = b_chunk else {
                    return Err(invalid("unexpected triple token slice"));
                };
                let [oc0, oc1, oc2, oc3] = c_chunk else {
                    return Err(invalid("unexpected triple token slice"));
                };
                let x0 = act_chunk[0].as_slice();
                let x1 = act_chunk[1].as_slice();
                let x2 = act_chunk[2].as_slice();
                let x3 = act_chunk[3].as_slice();
                self.statistics.operations_attempted =
                    self.statistics.operations_attempted.saturating_add(8);
                self.queue_writes_attempted = self.queue_writes_attempted.saturating_add(1);
                self.submitter.submit_resident_multi8(
                    [h0, h1, h2, h0, h1, h2, h0, h1],
                    [x0, x0, x0, x1, x1, x1, x2, x2],
                    [
                        oa0.as_mut_slice(),
                        ob0.as_mut_slice(),
                        oc0.as_mut_slice(),
                        oa1.as_mut_slice(),
                        ob1.as_mut_slice(),
                        oc1.as_mut_slice(),
                        oa2.as_mut_slice(),
                        ob2.as_mut_slice(),
                    ],
                )?;
                for out in [oa0, ob0, oc0, oa1, ob1, oc1, oa2, ob2] {
                    finite_slice(out)?;
                }
                self.statistics.operations_completed =
                    self.statistics.operations_completed.saturating_add(8);
                self.queue_writes_completed = self.queue_writes_completed.saturating_add(1);
                self.statistics.projections_completed =
                    self.statistics.projections_completed.saturating_add(8);

                self.statistics.operations_attempted =
                    self.statistics.operations_attempted.saturating_add(4);
                self.queue_writes_attempted = self.queue_writes_attempted.saturating_add(1);
                self.submitter.submit_resident_multi4(
                    [h2, h0, h1, h2],
                    [x2, x3, x3, x3],
                    [
                        oc2.as_mut_slice(),
                        oa3.as_mut_slice(),
                        ob3.as_mut_slice(),
                        oc3.as_mut_slice(),
                    ],
                )?;
                for out in [oc2, oa3, ob3, oc3] {
                    finite_slice(out)?;
                }
                self.statistics.operations_completed =
                    self.statistics.operations_completed.saturating_add(4);
                self.queue_writes_completed = self.queue_writes_completed.saturating_add(1);
                self.statistics.projections_completed =
                    self.statistics.projections_completed.saturating_add(4);
            } else if rem_act.len() >= 2 {
                let (act_chunk, next_act) = rem_act.split_at(2);
                let (a_chunk, next_a) = rem_a.split_at_mut(2);
                let (b_chunk, next_b) = rem_b.split_at_mut(2);
                let (c_chunk, next_c) = rem_c.split_at_mut(2);
                rem_act = next_act;
                rem_a = next_a;
                rem_b = next_b;
                rem_c = next_c;
                let [oa0, oa1] = a_chunk else {
                    return Err(invalid("unexpected triple token slice"));
                };
                let [ob0, ob1] = b_chunk else {
                    return Err(invalid("unexpected triple token slice"));
                };
                let [oc0, oc1] = c_chunk else {
                    return Err(invalid("unexpected triple token slice"));
                };
                let x0 = act_chunk[0].as_slice();
                let x1 = act_chunk[1].as_slice();
                self.statistics.operations_attempted =
                    self.statistics.operations_attempted.saturating_add(4);
                self.queue_writes_attempted = self.queue_writes_attempted.saturating_add(1);
                self.submitter.submit_resident_multi4(
                    [h0, h1, h2, h0],
                    [x0, x0, x0, x1],
                    [
                        oa0.as_mut_slice(),
                        ob0.as_mut_slice(),
                        oc0.as_mut_slice(),
                        oa1.as_mut_slice(),
                    ],
                )?;
                finite_slice(oa0)?;
                finite_slice(ob0)?;
                finite_slice(oc0)?;
                finite_slice(oa1)?;
                self.statistics.operations_completed =
                    self.statistics.operations_completed.saturating_add(4);
                self.queue_writes_completed = self.queue_writes_completed.saturating_add(1);
                self.statistics.projections_completed =
                    self.statistics.projections_completed.saturating_add(4);

                self.statistics.operations_attempted =
                    self.statistics.operations_attempted.saturating_add(2);
                self.queue_writes_attempted = self.queue_writes_attempted.saturating_add(1);
                self.submitter.submit_resident_batch2(
                    [h1, h2],
                    x1,
                    [ob1.as_mut_slice(), oc1.as_mut_slice()],
                )?;
                finite_slice(ob1)?;
                finite_slice(oc1)?;
                self.statistics.operations_completed =
                    self.statistics.operations_completed.saturating_add(2);
                self.queue_writes_completed = self.queue_writes_completed.saturating_add(1);
                self.statistics.projections_completed =
                    self.statistics.projections_completed.saturating_add(2);
            } else {
                let x0 = rem_act[0].as_slice();
                let oa0 = &mut rem_a[0];
                let ob0 = &mut rem_b[0];
                let oc0 = &mut rem_c[0];
                rem_act = &[];
                rem_a = &mut [];
                rem_b = &mut [];
                rem_c = &mut [];
                self.statistics.operations_attempted =
                    self.statistics.operations_attempted.saturating_add(3);
                self.queue_writes_attempted = self.queue_writes_attempted.saturating_add(1);
                self.submitter.submit_resident_batch3(
                    [h0, h1, h2],
                    x0,
                    [oa0.as_mut_slice(), ob0.as_mut_slice(), oc0.as_mut_slice()],
                )?;
                finite_slice(oa0)?;
                finite_slice(ob0)?;
                finite_slice(oc0)?;
                self.statistics.operations_completed =
                    self.statistics.operations_completed.saturating_add(3);
                self.queue_writes_completed = self.queue_writes_completed.saturating_add(1);
                self.statistics.projections_completed =
                    self.statistics.projections_completed.saturating_add(3);
            }
        }
        Ok(())
    }

    /// Execute a fused `ffn_gate` + `ffn_up` + `HTP_OP_GLU_SWIGLU` chain in a
    /// single 3-operation DSP queue submission with VTCM intermediate residency.
    ///
    /// # Errors
    /// Rejects non-resident or mismatched tensors, non-finite inputs/outputs or
    /// failed submissions, closing the executor on any failure.
    pub fn project_f32_ffn_swiglu(
        &mut self,
        gate: &Tensor,
        up: &Tensor,
        activation: &[f32],
        output: &mut Vec<f32>,
    ) -> Result<()> {
        let result = self.project_f32_ffn_swiglu_inner(gate, up, activation, output);
        if result.is_err() {
            self.failed = true;
            output.clear();
        }
        result
    }

    fn project_f32_ffn_swiglu_inner(
        &mut self,
        gate: &Tensor,
        up: &Tensor,
        activation: &[f32],
        output: &mut Vec<f32>,
    ) -> Result<()> {
        if self.failed {
            return Err(invalid(
                "native Q2 Hexagon executor requires terminal teardown",
            ));
        }
        let tensors = [gate, up];
        let Some((columns, rows_per_member, cached_indices)) = self.validate_f32_group(&tensors)
        else {
            return Err(invalid(
                "unsupported native Q2 Hexagon fused FFN SwiGLU projection group",
            ));
        };
        let rows = rows_per_member[0];
        if rows_per_member[1] != rows || activation.len() != columns {
            return Err(invalid("unsupported native Q2 Hexagon projection contract"));
        }
        finite_slice(activation)?;
        let mut weights = Vec::new();
        for (member, &tensor) in tensors.iter().enumerate() {
            if cached_indices[member].is_none() {
                self.prepare_resident(tensor, &mut weights, false)?
                    .ok_or_else(|| invalid("native Q2 resident preparation failed"))?;
            }
        }
        let mut indices = [0_usize; 2];
        if let Some(cache) = &mut self.residency {
            for (member, &tensor) in tensors.iter().enumerate() {
                let idx = cache
                    .find(tensor)
                    .ok_or_else(|| invalid("native Q2 resident preparation failed"))?;
                if cache.projections[idx].evaluated {
                    cache.statistics.cache_hits = cache.statistics.cache_hits.saturating_add(1);
                } else {
                    cache.projections[idx].evaluated = true;
                }
                indices[member] = idx;
            }
        }
        if output.len() != rows {
            output.clear();
            output
                .try_reserve(rows)
                .map_err(|_| invalid("native Q2 output allocation failed"))?;
            output.resize(rows, f32::NAN);
        }
        self.statistics.operations_attempted =
            self.statistics.operations_attempted.saturating_add(3);
        self.queue_writes_attempted = self.queue_writes_attempted.saturating_add(1);
        let cache = self
            .residency
            .as_ref()
            .ok_or_else(|| invalid("native Q2 resident cache is unavailable"))?;
        let gate_handle = &cache.handles[cache.projections[indices[0]].first_handle];
        let up_handle = &cache.handles[cache.projections[indices[1]].first_handle];
        self.submitter.submit_resident_ffn_swiglu(
            gate_handle,
            up_handle,
            activation,
            output.as_mut_slice(),
        )?;
        finite_slice(output)?;
        self.statistics.operations_completed =
            self.statistics.operations_completed.saturating_add(3);
        self.queue_writes_completed = self.queue_writes_completed.saturating_add(1);
        self.statistics.projections_completed =
            self.statistics.projections_completed.saturating_add(2);
        Ok(())
    }

    fn supports_f32_ffn_swiglu_inner(&self, gate: &Tensor, up: &Tensor) -> bool {
        self.validate_f32_group(&[gate, up])
            .is_some_and(|(_, rows, _)| rows[0] == rows[1])
    }

    fn supports_f32_ffn_chain_inner(&self, tensors: [&Tensor; 3]) -> bool {
        let [gate, up, down] = tensors;
        let Some((hidden, gate_up_rows, _)) = self.validate_f32_group(&[gate, up]) else {
            return false;
        };
        let ffn_dim = gate_up_rows[0];
        if gate_up_rows[1] != ffn_dim || !self.supports_f32(down) {
            return false;
        }
        let Ok((down_cols, down_rows)) = self.tensor_dimensions(down) else {
            return false;
        };
        down_cols == ffn_dim
            && down_rows == hidden
            && down_cols <= self.maximum_columns
            && down_rows <= self.maximum_rows
            && down_rows.is_multiple_of(32)
    }

    fn supports_f32_ffn_chain_lora_inner(&self, tensors: [&Tensor; 3]) -> bool {
        self.submitter.supports_ffn_chain_lora() && self.supports_f32_ffn_chain_inner(tensors)
    }

    /// Execute a fused `ffn_gate` + `ffn_up` + `HTP_OP_GLU_SWIGLU` chain across
    /// a tile of `F32` token activations, batching pairs of tokens into 6-op
    /// DSP queue submissions with per-token VTCM intermediate residency.
    ///
    /// # Errors
    /// Rejects non-resident or mismatched tensors, non-finite inputs/outputs or
    /// failed submissions, closing the executor on any failure.
    pub fn project_f32_ffn_swiglu_tokens(
        &mut self,
        gate: &Tensor,
        up: &Tensor,
        activations: &[Vec<f32>],
        outputs: &mut [Vec<f32>],
    ) -> Result<()> {
        let result = self.project_f32_ffn_swiglu_tokens_inner(gate, up, activations, outputs);
        if result.is_err() {
            self.failed = true;
            for output in outputs.iter_mut() {
                output.clear();
            }
        }
        result
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Keep 2-token 6-op fused FFN SwiGLU resident dispatch and odd-tail handling together."
    )]
    fn project_f32_ffn_swiglu_tokens_inner(
        &mut self,
        gate: &Tensor,
        up: &Tensor,
        activations: &[Vec<f32>],
        outputs: &mut [Vec<f32>],
    ) -> Result<()> {
        if self.failed {
            return Err(invalid(
                "native Q2 Hexagon executor requires terminal teardown",
            ));
        }
        if activations.len() != outputs.len() {
            return Err(invalid(
                "mismatched F32 FFN SwiGLU token projection outputs",
            ));
        }
        if activations.is_empty() {
            return Ok(());
        }
        let tensors = [gate, up];
        let Some((columns, rows_per_member, cached_indices)) = self.validate_f32_group(&tensors)
        else {
            return Err(invalid(
                "unsupported native Q2 Hexagon fused FFN SwiGLU projection group",
            ));
        };
        let rows = rows_per_member[0];
        if rows_per_member[1] != rows {
            return Err(invalid("unsupported native Q2 Hexagon projection contract"));
        }
        for activation in activations {
            if activation.len() != columns {
                return Err(invalid("unsupported native Q2 Hexagon projection contract"));
            }
            finite_slice(activation)?;
        }
        let mut weights = Vec::new();
        for (member, &tensor) in tensors.iter().enumerate() {
            if cached_indices[member].is_none() {
                self.prepare_resident(tensor, &mut weights, false)?
                    .ok_or_else(|| invalid("native Q2 resident preparation failed"))?;
            }
        }
        let mut indices = [0_usize; 2];
        if let Some(cache) = &mut self.residency {
            for (member, &tensor) in tensors.iter().enumerate() {
                let idx = cache
                    .find(tensor)
                    .ok_or_else(|| invalid("native Q2 resident preparation failed"))?;
                let hits = if cache.projections[idx].evaluated {
                    u64::try_from(activations.len())?
                } else {
                    cache.projections[idx].evaluated = true;
                    u64::try_from(activations.len().saturating_sub(1))?
                };
                cache.statistics.cache_hits = cache.statistics.cache_hits.saturating_add(hits);
                indices[member] = idx;
            }
        }
        for output in outputs.iter_mut() {
            if output.len() != rows {
                output.clear();
                output
                    .try_reserve(rows)
                    .map_err(|_| invalid("native Q2 output allocation failed"))?;
                output.resize(rows, f32::NAN);
            }
        }
        let cache = self
            .residency
            .as_ref()
            .ok_or_else(|| invalid("native Q2 resident cache is unavailable"))?;
        let gate_handle = &cache.handles[cache.projections[indices[0]].first_handle];
        let up_handle = &cache.handles[cache.projections[indices[1]].first_handle];
        let mut rem_act = activations;
        let mut rem_out = outputs;
        while !rem_act.is_empty() {
            if rem_act.len() >= 2 {
                let (act_chunk, next_act) = rem_act.split_at(2);
                let (out_chunk, next_out) = rem_out.split_at_mut(2);
                rem_act = next_act;
                rem_out = next_out;
                let [o0, o1] = out_chunk else {
                    return Err(invalid("unexpected FFN SwiGLU token slice"));
                };
                self.statistics.operations_attempted =
                    self.statistics.operations_attempted.saturating_add(6);
                self.queue_writes_attempted = self.queue_writes_attempted.saturating_add(1);
                self.submitter.submit_resident_ffn_swiglu_multi2(
                    gate_handle,
                    up_handle,
                    [act_chunk[0].as_slice(), act_chunk[1].as_slice()],
                    [o0.as_mut_slice(), o1.as_mut_slice()],
                )?;
                finite_slice(o0)?;
                finite_slice(o1)?;
                self.statistics.operations_completed =
                    self.statistics.operations_completed.saturating_add(6);
                self.queue_writes_completed = self.queue_writes_completed.saturating_add(1);
                self.statistics.projections_completed =
                    self.statistics.projections_completed.saturating_add(4);
            } else {
                let a0 = rem_act[0].as_slice();
                let o0 = &mut rem_out[0];
                rem_act = &[];
                rem_out = &mut [];
                self.statistics.operations_attempted =
                    self.statistics.operations_attempted.saturating_add(3);
                self.queue_writes_attempted = self.queue_writes_attempted.saturating_add(1);
                self.submitter.submit_resident_ffn_swiglu(
                    gate_handle,
                    up_handle,
                    a0,
                    o0.as_mut_slice(),
                )?;
                finite_slice(o0)?;
                self.statistics.operations_completed =
                    self.statistics.operations_completed.saturating_add(3);
                self.queue_writes_completed = self.queue_writes_completed.saturating_add(1);
                self.statistics.projections_completed =
                    self.statistics.projections_completed.saturating_add(2);
            }
        }
        Ok(())
    }

    /// Execute a single `F32` projection while overlapping/fusing the host CPU
    /// rank-1 adapter update (`output += (activation · adapter_a) * adapter_b`).
    ///
    /// # Errors
    /// Rejects non-finite activations/adapters, mismatched adapter dimensions or
    /// failed submissions, closing the executor on any failure.
    pub fn project_f32_with_rank1_adapter(
        &mut self,
        tensor: &Tensor,
        activation: &[f32],
        adapter_a: &[f32],
        adapter_b: &[f32],
        output: &mut Vec<f32>,
    ) -> Result<()> {
        let result = self
            .project_f32_with_rank1_adapter_inner(tensor, activation, adapter_a, adapter_b, output);
        if result.is_err() {
            self.failed = true;
            output.clear();
        }
        result
    }

    fn project_f32_with_rank1_adapter_inner(
        &mut self,
        tensor: &Tensor,
        activation: &[f32],
        adapter_a: &[f32],
        adapter_b: &[f32],
        output: &mut Vec<f32>,
    ) -> Result<()> {
        let (columns, rows) = self.tensor_dimensions(tensor)?;
        if activation.len() != columns || adapter_a.len() != columns || adapter_b.len() != rows {
            return Err(invalid("mismatched rank-1 adapter dimensions"));
        }
        finite_slice(adapter_a)?;
        finite_slice(adapter_b)?;
        let mut dot_accum = 0.0_f32;
        for (&x, &a) in activation.iter().zip(adapter_a) {
            dot_accum = finite(x.mul_add(a, dot_accum))?;
        }
        self.project_f32_inner(tensor, activation, output)?;
        if dot_accum != 0.0 {
            for (dst, &b) in output.iter_mut().zip(adapter_b) {
                *dst = finite(dot_accum.mul_add(b, *dst))?;
            }
        }
        Ok(())
    }

    /// Execute a full intra-layer FFN chain (`[gate, up, down]` + `HTP_OP_GLU_SWIGLU` +
    /// optional `HTP_OP_HADAMARD` + optional `HTP_OP_ADD` residual) in a single DSP
    /// queue submission with VTCM intermediate residency chaining.
    ///
    /// # Errors
    /// Rejects non-resident or mismatched tensors, non-finite inputs/outputs or
    /// failed submissions, closing the executor on any failure.
    pub fn project_f32_ffn_chain(
        &mut self,
        tensors: [&Tensor; 3],
        activation: &[f32],
        residual: Option<&[f32]>,
        hadamard_block_size: Option<u32>,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        let result = self.project_f32_ffn_chain_common(
            tensors,
            activation,
            residual,
            hadamard_block_size,
            None,
            output,
        );
        if result.is_err() {
            self.failed = true;
            output.clear();
        }
        result.map(|_| ())
    }

    /// Execute a full intra-layer FFN chain with an inline `F32` `ffn_down`
    /// rank-one `LoRA` A dot side-branch (`lora_a · SwiGLU(gate(x), up(x))`) in
    /// one DSP queue submission, returning the finite scalar dot product.
    ///
    /// # Errors
    /// Rejects unsupported sessions, non-resident or mismatched tensors,
    /// mismatched/nonfinite `lora_a`, nonfinite inputs/outputs or failed
    /// submissions, closing the executor on any failure.
    pub fn project_f32_ffn_chain_lora(
        &mut self,
        tensors: [&Tensor; 3],
        activation: &[f32],
        residual: Option<&[f32]>,
        hadamard_block_size: Option<u32>,
        lora_a: &[f32],
        output: &mut Vec<f32>,
    ) -> Result<f32> {
        let result = self.project_f32_ffn_chain_common(
            tensors,
            activation,
            residual,
            hadamard_block_size,
            Some(lora_a),
            output,
        );
        match result {
            Ok(Some(dot)) => Ok(dot),
            Ok(None) => {
                self.failed = true;
                output.clear();
                Err(invalid("missing native Q2 FFN chain LoRA dot output"))
            }
            Err(err) => {
                self.failed = true;
                output.clear();
                Err(err)
            }
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Keep full intra-layer FFN VTCM chain validation, optional LoRA side-branch, resident preparation and statistics together."
    )]
    fn project_f32_ffn_chain_common(
        &mut self,
        tensors: [&Tensor; 3],
        activation: &[f32],
        residual: Option<&[f32]>,
        hadamard_block_size: Option<u32>,
        lora_a: Option<&[f32]>,
        output: &mut Vec<f32>,
    ) -> Result<Option<f32>> {
        if self.failed {
            return Err(invalid(
                "native Q2 Hexagon executor requires terminal teardown",
            ));
        }
        if lora_a.is_some() && !self.submitter.supports_ffn_chain_lora() {
            return Err(invalid(
                "unsupported native Q2 Hexagon fused FFN chain LoRA submission",
            ));
        }
        let [gate, up, down] = tensors;
        let Some((hidden, gate_up_rows, cached_gate_up)) = self.validate_f32_group(&[gate, up])
        else {
            return Err(invalid(
                "unsupported native Q2 Hexagon fused FFN chain projection group",
            ));
        };
        let ffn_dim = gate_up_rows[0];
        if gate_up_rows[1] != ffn_dim || activation.len() != hidden || !self.supports_f32(down) {
            return Err(invalid("unsupported native Q2 Hexagon projection contract"));
        }
        let (down_cols, down_rows) = self.tensor_dimensions(down)?;
        if down_cols != ffn_dim
            || down_rows != hidden
            || down_cols > self.maximum_columns
            || down_rows > self.maximum_rows
            || !down_rows.is_multiple_of(32)
        {
            return Err(invalid("unsupported native Q2 Hexagon projection contract"));
        }
        if let Some(res) = residual {
            if res.len() != hidden {
                return Err(invalid("mismatched FFN chain residual length"));
            }
            finite_slice(res)?;
        }
        if let Some(lora_a_vals) = lora_a {
            if lora_a_vals.len() != ffn_dim {
                return Err(invalid("mismatched FFN chain LoRA A length"));
            }
            finite_slice(lora_a_vals)?;
        }
        if let Some(block_size) = hadamard_block_size {
            let block_usize = usize::try_from(block_size)?;
            if block_size < 32
                || !block_size.is_power_of_two()
                || !ffn_dim.is_multiple_of(block_usize)
            {
                return Err(invalid("invalid FFN chain Hadamard block size"));
            }
        }
        finite_slice(activation)?;
        let cached_down = self.residency.as_ref().and_then(|cache| cache.find(down));
        let cached_all = [cached_gate_up[0], cached_gate_up[1], cached_down];
        let mut weights = Vec::new();
        for (member, &tensor) in tensors.iter().enumerate() {
            if cached_all[member].is_none() {
                self.prepare_resident(tensor, &mut weights, false)?
                    .ok_or_else(|| invalid("native Q2 resident preparation failed"))?;
            }
        }
        let mut indices = [0_usize; 3];
        if let Some(cache) = &mut self.residency {
            for (member, &tensor) in tensors.iter().enumerate() {
                let idx = cache
                    .find(tensor)
                    .ok_or_else(|| invalid("native Q2 resident preparation failed"))?;
                if cache.projections[idx].evaluated {
                    cache.statistics.cache_hits = cache.statistics.cache_hits.saturating_add(1);
                } else {
                    cache.projections[idx].evaluated = true;
                }
                indices[member] = idx;
            }
        }
        if output.len() != hidden {
            output.clear();
            output
                .try_reserve(hidden)
                .map_err(|_| invalid("native Q2 output allocation failed"))?;
            output.resize(hidden, f32::NAN);
        }
        let op_count = 4_u64
            + u64::from(lora_a.is_some())
            + u64::from(hadamard_block_size.is_some())
            + u64::from(residual.is_some());
        self.statistics.operations_attempted = self
            .statistics
            .operations_attempted
            .saturating_add(op_count);
        self.queue_writes_attempted = self.queue_writes_attempted.saturating_add(1);
        let cache = self
            .residency
            .as_ref()
            .ok_or_else(|| invalid("native Q2 resident cache is unavailable"))?;
        let gate_handle = &cache.handles[cache.projections[indices[0]].first_handle];
        let up_handle = &cache.handles[cache.projections[indices[1]].first_handle];
        let down_handle = &cache.handles[cache.projections[indices[2]].first_handle];
        let lora_dot = if let Some(lora_a_vals) = lora_a {
            let dot = self.submitter.submit_resident_ffn_chain_lora(
                [gate_handle, up_handle, down_handle],
                activation,
                residual,
                hadamard_block_size,
                lora_a_vals,
                output.as_mut_slice(),
            )?;
            if !dot.is_finite() {
                return Err(invalid("nonfinite Hexagon projection value"));
            }
            Some(dot)
        } else {
            self.submitter.submit_resident_ffn_chain(
                [gate_handle, up_handle, down_handle],
                activation,
                residual,
                hadamard_block_size,
                output.as_mut_slice(),
            )?;
            None
        };
        finite_slice(output)?;
        self.statistics.operations_completed = self
            .statistics
            .operations_completed
            .saturating_add(op_count);
        self.queue_writes_completed = self.queue_writes_completed.saturating_add(1);
        self.statistics.projections_completed =
            self.statistics.projections_completed.saturating_add(3);
        Ok(lora_dot)
    }
    #[expect(
        clippy::too_many_lines,
        reason = "Keep tile-wide validation, residency, bounded paired submissions and counters together"
    )]
    fn project_f32_ffn_chain_tokens_inner(
        &mut self,
        tensors: [&Tensor; 3],
        activations: &[Vec<f32>],
        residuals: Option<&[Vec<f32>]>,
        hadamard_block_size: Option<u32>,
        outputs: &mut [Vec<f32>],
    ) -> Result<()> {
        if self.failed {
            return Err(invalid(
                "native Q2 Hexagon executor requires terminal teardown",
            ));
        }
        if activations.len() != outputs.len()
            || residuals.is_some_and(|res| res.len() != activations.len())
        {
            return Err(invalid("mismatched F32 FFN chain token projection outputs"));
        }
        if activations.is_empty() {
            return Ok(());
        }
        let [gate, up, down] = tensors;
        let Some((hidden, gate_up_rows, cached_gate_up)) = self.validate_f32_group(&[gate, up])
        else {
            return Err(invalid(
                "unsupported native Q2 Hexagon fused FFN chain projection group",
            ));
        };
        let ffn_dim = gate_up_rows[0];
        if gate_up_rows[1] != ffn_dim
            || activations.iter().any(|act| act.len() != hidden)
            || !self.supports_f32(down)
        {
            return Err(invalid("unsupported native Q2 Hexagon projection contract"));
        }
        let (down_cols, down_rows) = self.tensor_dimensions(down)?;
        if down_cols != ffn_dim
            || down_rows != hidden
            || down_cols > self.maximum_columns
            || down_rows > self.maximum_rows
            || !down_rows.is_multiple_of(32)
        {
            return Err(invalid("unsupported native Q2 Hexagon projection contract"));
        }
        if let Some(residuals) = residuals {
            for res in residuals {
                if res.len() != hidden {
                    return Err(invalid("mismatched FFN chain residual length"));
                }
                finite_slice(res)?;
            }
        }
        if let Some(block_size) = hadamard_block_size {
            let block_usize = usize::try_from(block_size)?;
            if block_size < 32
                || !block_size.is_power_of_two()
                || !ffn_dim.is_multiple_of(block_usize)
            {
                return Err(invalid("invalid FFN chain Hadamard block size"));
            }
        }
        for activation in activations {
            finite_slice(activation)?;
        }
        let cached_down = self.residency.as_ref().and_then(|cache| cache.find(down));
        let cached_all = [cached_gate_up[0], cached_gate_up[1], cached_down];
        let mut weights = Vec::new();
        for (member, &tensor) in tensors.iter().enumerate() {
            if cached_all[member].is_none() {
                self.prepare_resident(tensor, &mut weights, false)?
                    .ok_or_else(|| invalid("native Q2 resident preparation failed"))?;
            }
        }
        let mut indices = [0_usize; 3];
        if let Some(cache) = &mut self.residency {
            for (member, &tensor) in tensors.iter().enumerate() {
                let idx = cache
                    .find(tensor)
                    .ok_or_else(|| invalid("native Q2 resident preparation failed"))?;
                if cache.projections[idx].evaluated {
                    cache.statistics.cache_hits = cache
                        .statistics
                        .cache_hits
                        .saturating_add(u64::try_from(activations.len())?);
                } else {
                    cache.projections[idx].evaluated = true;
                    cache.statistics.cache_hits = cache
                        .statistics
                        .cache_hits
                        .saturating_add(u64::try_from(activations.len() - 1)?);
                }
                indices[member] = idx;
            }
        }
        for output in outputs.iter_mut() {
            if output.len() != hidden {
                output.clear();
                output
                    .try_reserve(hidden)
                    .map_err(|_| invalid("native Q2 output allocation failed"))?;
                output.resize(hidden, f32::NAN);
            }
        }
        let op_count =
            4_u64 + u64::from(hadamard_block_size.is_some()) + u64::from(residuals.is_some());
        let cache = self
            .residency
            .as_ref()
            .ok_or_else(|| invalid("native Q2 resident cache is unavailable"))?;
        let handles = indices.map(|index| &cache.handles[cache.projections[index].first_handle]);
        let batch_tokens = if self.submitter.supports_ffn_chain_multi2() {
            2
        } else {
            1
        };
        for (batch, out_chunk) in outputs.chunks_mut(batch_tokens).enumerate() {
            let token = batch * batch_tokens;
            let count = u64::try_from(out_chunk.len())?;
            self.statistics.operations_attempted = self
                .statistics
                .operations_attempted
                .saturating_add(op_count * count);
            self.queue_writes_attempted = self.queue_writes_attempted.saturating_add(1);
            match out_chunk {
                [out0, out1] => self.submitter.submit_resident_ffn_chain_multi2(
                    handles,
                    [
                        activations[token].as_slice(),
                        activations[token + 1].as_slice(),
                    ],
                    [
                        residuals.map(|res| res[token].as_slice()),
                        residuals.map(|res| res[token + 1].as_slice()),
                    ],
                    hadamard_block_size,
                    [out0.as_mut_slice(), out1.as_mut_slice()],
                )?,
                [out] => self.submitter.submit_resident_ffn_chain(
                    handles,
                    &activations[token],
                    residuals.map(|res| res[token].as_slice()),
                    hadamard_block_size,
                    out,
                )?,
                _ => return Err(invalid("unexpected FFN chain token batch")),
            }
            for output in out_chunk.iter() {
                finite_slice(output)?;
            }
            self.statistics.operations_completed = self
                .statistics
                .operations_completed
                .saturating_add(op_count * count);
            self.queue_writes_completed = self.queue_writes_completed.saturating_add(1);
            self.statistics.projections_completed = self
                .statistics
                .projections_completed
                .saturating_add(3 * count);
        }
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
        let (columns, rows) = self.tensor_dimensions(tensor).ok()?;
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
        if let Some(allowed) = &self.allowed_tensors {
            let addr = tensor.data.as_ptr().addr();
            let (min_addr, max_addr) = self.allowed_addr_range;
            if addr < min_addr || addr > max_addr {
                return false;
            }
            let Ok(idx) = allowed.binary_search_by(|candidate| {
                selected_tensor_key(candidate).cmp(&selected_tensor_key(tensor))
            }) else {
                return false;
            };
            let candidate = &allowed[idx];
            if !bonsai_prism_runtime::model::TensorBytes::ptr_eq(&candidate.data, &tensor.data)
                || candidate.kind != tensor.kind
                || candidate.dimensions != tensor.dimensions
                || candidate.name != tensor.name
            {
                return false;
            }
        }
        self.tensor_dimensions(tensor).is_ok_and(|(columns, rows)| {
            self.payload_bytes(columns, rows)
                .is_some_and(|bytes| bytes <= self.maximum_owned_payload_bytes)
        })
    }

    fn release_host_weight_binding(
        &mut self,
        previous_tensor: &Tensor,
        released_data: &bonsai_prism_runtime::model::TensorBytes,
    ) -> bool {
        if self.failed || !released_data.is_empty() || released_data.is_mapped() {
            return false;
        }
        let Some(resident_idx) = self
            .residency
            .as_ref()
            .and_then(|cache| cache.find(previous_tensor))
        else {
            return false;
        };
        let allowed_idx = if let Some(allowed) = &self.allowed_tensors {
            let addr = previous_tensor.data.as_ptr().addr();
            let (min_addr, max_addr) = self.allowed_addr_range;
            if addr < min_addr || addr > max_addr {
                return false;
            }
            let Ok(idx) = allowed.binary_search_by(|candidate| {
                selected_tensor_key(candidate).cmp(&selected_tensor_key(previous_tensor))
            }) else {
                return false;
            };
            let candidate = &allowed[idx];
            if !bonsai_prism_runtime::model::TensorBytes::ptr_eq(
                &candidate.data,
                &previous_tensor.data,
            ) || candidate.kind != previous_tensor.kind
                || candidate.dimensions != previous_tensor.dimensions
                || candidate.name != previous_tensor.name
            {
                return false;
            }
            Some(idx)
        } else {
            None
        };
        let Some(cache) = &mut self.residency else {
            return false;
        };
        if !cache.rebind_released(resident_idx, released_data.clone()) {
            return false;
        }
        if let (Some(allowed), Some(idx)) = (&mut self.allowed_tensors, allowed_idx) {
            allowed[idx].data = released_data.clone();
            allowed.sort_by(|left, right| selected_tensor_key(left).cmp(&selected_tensor_key(right)));
            let min_addr = allowed
                .first()
                .map_or(0, |tensor| tensor.data.as_ptr().addr());
            let max_addr = allowed
                .last()
                .map_or(0, |tensor| tensor.data.as_ptr().addr());
            self.allowed_addr_range = (min_addr, max_addr);
        }
        true
    }

    fn has_released_host_weight(&self, tensor: &Tensor) -> bool {
        tensor.data.is_empty()
            && self.residency.as_ref().is_some_and(|cache| {
                cache
                    .find(tensor)
                    .is_some_and(|index| cache.projections[index].host_released)
            })
    }

    fn project_f32(
        &mut self,
        tensor: &Tensor,
        activation: &[f32],
        _pinned: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
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
        let result = self.project_f32_group_inner(tensors, activation, outputs);
        if result.is_err() {
            self.failed = true;
            for output in outputs.iter_mut() {
                output.clear();
            }
        }
        result
    }

    fn project_f32_tokens(
        &mut self,
        tensor: &Tensor,
        activations: &[Vec<f32>],
        _pinned: bool,
        outputs: &mut [Vec<f32>],
    ) -> Result<()> {
        let result = self.project_f32_tokens_inner(tensor, activations, outputs);
        if result.is_err() {
            self.failed = true;
            for output in outputs.iter_mut() {
                output.clear();
            }
        }
        result
    }

    fn project_f32_pair_tokens(
        &mut self,
        tensors: [&Tensor; 2],
        activations: &[Vec<f32>],
        _pinned: bool,
        outputs_a: &mut [Vec<f32>],
        outputs_b: &mut [Vec<f32>],
    ) -> Result<()> {
        let result = self.project_f32_pair_tokens_inner(tensors, activations, outputs_a, outputs_b);
        if result.is_err() {
            self.failed = true;
            for output in outputs_a.iter_mut() {
                output.clear();
            }
            for output in outputs_b.iter_mut() {
                output.clear();
            }
        }
        result
    }

    fn project_f32_triple_tokens(
        &mut self,
        tensors: [&Tensor; 3],
        activations: &[Vec<f32>],
        _pinned: bool,
        outputs_a: &mut [Vec<f32>],
        outputs_b: &mut [Vec<f32>],
        outputs_c: &mut [Vec<f32>],
    ) -> Result<()> {
        let result = self.project_f32_triple_tokens_inner(
            tensors,
            activations,
            outputs_a,
            outputs_b,
            outputs_c,
        );
        if result.is_err() {
            self.failed = true;
            for output in outputs_a.iter_mut() {
                output.clear();
            }
            for output in outputs_b.iter_mut() {
                output.clear();
            }
            for output in outputs_c.iter_mut() {
                output.clear();
            }
        }
        result
    }

    fn supports_f32_ffn_swiglu(&self, gate: &Tensor, up: &Tensor) -> bool {
        self.supports_f32_ffn_swiglu_inner(gate, up)
    }

    fn project_f32_ffn_swiglu(
        &mut self,
        gate: &Tensor,
        up: &Tensor,
        activation: &[f32],
        _pinned: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        Self::project_f32_ffn_swiglu(self, gate, up, activation, output)
    }

    fn project_f32_ffn_swiglu_tokens(
        &mut self,
        gate: &Tensor,
        up: &Tensor,
        activations: &[Vec<f32>],
        _pinned: bool,
        outputs: &mut [Vec<f32>],
    ) -> Result<()> {
        Self::project_f32_ffn_swiglu_tokens(self, gate, up, activations, outputs)
    }

    fn supports_f32_ffn_chain(&self, tensors: [&Tensor; 3]) -> bool {
        self.supports_f32_ffn_chain_inner(tensors)
    }

    fn project_f32_ffn_chain(
        &mut self,
        tensors: [&Tensor; 3],
        activation: &[f32],
        residual: Option<&[f32]>,
        hadamard_block_size: Option<u32>,
        _pinned: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        Self::project_f32_ffn_chain(
            self,
            tensors,
            activation,
            residual,
            hadamard_block_size,
            output,
        )
    }

    fn supports_f32_ffn_chain_lora(&self, tensors: [&Tensor; 3]) -> bool {
        self.supports_f32_ffn_chain_lora_inner(tensors)
    }

    fn project_f32_ffn_chain_lora(
        &mut self,
        tensors: [&Tensor; 3],
        activation: &[f32],
        residual: Option<&[f32]>,
        hadamard_block_size: Option<u32>,
        lora_a: &[f32],
        _pinned: bool,
        output: &mut Vec<f32>,
    ) -> Result<f32> {
        Self::project_f32_ffn_chain_lora(
            self,
            tensors,
            activation,
            residual,
            hadamard_block_size,
            lora_a,
            output,
        )
    }
    fn project_f32_ffn_chain_tokens(
        &mut self,
        tensors: [&Tensor; 3],
        activations: &[Vec<f32>],
        residuals: Option<&[Vec<f32>]>,
        hadamard_block_size: Option<u32>,
        _pinned: bool,
        outputs: &mut [Vec<f32>],
    ) -> Result<()> {
        let result = self.project_f32_ffn_chain_tokens_inner(
            tensors,
            activations,
            residuals,
            hadamard_block_size,
            outputs,
        );
        if result.is_err() {
            self.failed = true;
            for output in outputs {
                output.clear();
            }
        }
        result
    }
}

fn resize_packed_scratch(weights: &mut Vec<u8>, required_bytes: usize) -> Result<()> {
    if weights.len() < required_bytes {
        weights
            .try_reserve_exact(required_bytes - weights.len())
            .map_err(|_| invalid("native Q2 scratch allocation failed"))?;
        weights.resize(required_bytes, 0_u8);
    }
    Ok(())
}

fn selected_tensor_key(tensor: &Tensor) -> (usize, u32, &[usize], &str) {
    (
        tensor.data.as_ptr().addr(),
        tensor.kind,
        &tensor.dimensions,
        &tensor.name,
    )
}

const PQ2_KIND: u32 = 142;
const PTQ1_KIND: u32 = 143;

const fn source_block_bytes(kind: u32) -> Option<usize> {
    match kind {
        PQ2_KIND => Some(34),
        PTQ1_KIND => Some(28),
        _ => None,
    }
}

const fn ptq1_symbol_table() -> [[u8; 5]; 256] {
    let mut table = [[0_u8; 5]; 256];
    let mut index = 0_usize;
    let mut byte = 0_u16;
    while byte < 256 {
        let mut fraction = byte;
        let mut digit = 0_usize;
        while digit < 5 {
            fraction *= 3;
            let trit = fraction >> 8;
            table[index][digit] = trit.to_le_bytes()[0];
            fraction &= 255;
            digit += 1;
        }
        byte += 1;
        index += 1;
    }
    table
}

const PTQ1_SYMBOLS: [[u8; 5]; 256] = ptq1_symbol_table();

fn validate_source_scales(tensor: &Tensor) -> Result<()> {
    match tensor.kind {
        PQ2_KIND => {
            for block in tensor.data.as_chunks::<34>().0 {
                scale(block)?;
            }
        }
        PTQ1_KIND => {
            for block in tensor.data.as_chunks::<28>().0 {
                scale(&block[26..])?;
            }
        }
        _ => return Err(invalid("unsupported Hexagon native Q2 tensor type or rank")),
    }
    Ok(())
}

fn native_dimensions_geometry(tensor: &Tensor) -> Result<(usize, usize, usize)> {
    let Some(block_bytes) = source_block_bytes(tensor.kind) else {
        return Err(invalid("unsupported Hexagon native Q2 tensor type or rank"));
    };
    if tensor.dimensions.len() != 2 {
        return Err(invalid("unsupported Hexagon native Q2 tensor type or rank"));
    }
    let (columns, rows) = (tensor.dimensions[0], tensor.dimensions[1]);
    let Some(expected_bytes) = (columns / 128)
        .checked_mul(block_bytes)
        .and_then(|bytes| bytes.checked_mul(rows))
    else {
        return Err(invalid(
            "invalid Hexagon native Q2 tensor geometry or extent",
        ));
    };
    if columns == 0 || rows == 0 || !columns.is_multiple_of(128) {
        return Err(invalid(
            "invalid Hexagon native Q2 tensor geometry or extent",
        ));
    }
    Ok((columns, rows, expected_bytes))
}

fn native_dimensions(tensor: &Tensor) -> Result<(usize, usize)> {
    let (columns, rows, expected_bytes) = native_dimensions_geometry(tensor)?;
    if tensor.data.len() != expected_bytes {
        return Err(invalid(
            "invalid Hexagon native Q2 tensor geometry or extent",
        ));
    }
    Ok((columns, rows))
}

fn native_dimensions_released(tensor: &Tensor) -> Result<(usize, usize)> {
    let (columns, rows, _) = native_dimensions_geometry(tensor)?;
    if !tensor.data.is_empty() {
        return Err(invalid(
            "invalid Hexagon native Q2 tensor geometry or extent",
        ));
    }
    Ok((columns, rows))
}

#[derive(Clone, Copy)]
struct ChunkGeometry {
    kind: u32,
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

fn pack_ptq1_tile_into(original: &[u8], row_in_tile: usize, pair_buf: &mut [u8]) -> Result<()> {
    let original: &[u8; 28] = original
        .try_into()
        .map_err(|_| invalid("invalid PTQ1 block length"))?;
    if row_in_tile >= 32 {
        return Err(invalid("invalid PTQ1 tile row index"));
    }
    let pair_buf: &mut [u8; 1152] = pair_buf
        .try_into()
        .map_err(|_| invalid("invalid HTP Q2 tile pair buffer length"))?;

    let scale_bytes = [original[26], original[27]];
    if (u16::from_le_bytes(scale_bytes) & 0x7c00) == 0x7c00 {
        return Err(invalid("nonfinite Hexagon Q2 scale"));
    }
    let scale_pos0 = 512 + row_in_tile * 2;
    let scale_pos1 = 576 + 512 + row_in_tile * 2;
    pair_buf[scale_pos0..scale_pos0 + 2].copy_from_slice(&scale_bytes);
    pair_buf[scale_pos1..scale_pos1 + 2].copy_from_slice(&scale_bytes);

    let wide: [[u8; 5]; 16] = std::array::from_fn(|m| PTQ1_SYMBOLS[usize::from(original[m])]);
    let q_pos = [
        row_in_tile * 4,
        128 + row_in_tile * 4,
        256 + row_in_tile * 4,
        384 + row_in_tile * 4,
        576 + row_in_tile * 4,
    ];
    for n in 0..5 {
        let base = q_pos[n];
        for digit in 0..4 {
            pair_buf[base + digit] = wide[digit][n]
                | (wide[digit + 4][n] << 2)
                | (wide[digit + 8][n] << 4)
                | (wide[digit + 12][n] << 6);
        }
    }

    let mid: [[u8; 5]; 8] = std::array::from_fn(|m| PTQ1_SYMBOLS[usize::from(original[16 + m])]);
    let q_pos5 = 576 + 128 + row_in_tile * 4;
    let q_pos6 = 576 + 256 + row_in_tile * 4;
    for digit in 0..4 {
        pair_buf[q_pos5 + digit] = mid[digit][0]
            | (mid[digit + 4][0] << 2)
            | (mid[digit][1] << 4)
            | (mid[digit + 4][1] << 6);
        pair_buf[q_pos6 + digit] = mid[digit][2]
            | (mid[digit + 4][2] << 2)
            | (mid[digit][3] << 4)
            | (mid[digit + 4][3] << 6);
    }

    let tail0 = PTQ1_SYMBOLS[usize::from(original[24])];
    let tail1 = PTQ1_SYMBOLS[usize::from(original[25])];
    let q_pos7 = 576 + 384 + row_in_tile * 4;
    pair_buf[q_pos7] = mid[0][4] | (mid[4][4] << 2) | (tail0[0] << 4) | (tail0[2] << 6);
    pair_buf[q_pos7 + 1] = mid[1][4] | (mid[5][4] << 2) | (tail1[0] << 4) | (tail1[2] << 6);
    pair_buf[q_pos7 + 2] = mid[2][4] | (mid[6][4] << 2) | (tail0[1] << 4) | (tail0[3] << 6);
    pair_buf[q_pos7 + 3] = mid[3][4] | (mid[7][4] << 2) | (tail1[1] << 4) | (tail1[3] << 6);
    Ok(())
}

fn pack_ptq1_row_group(
    tensor_data: &[u8],
    geometry: ChunkGeometry,
    group_row_start: usize,
    group_rows: usize,
    tile_pairs: &mut [[u8; 1152]],
) -> Result<()> {
    for row_in_tile in 0..group_rows {
        let local_row = group_row_start + row_in_tile;
        let row_block_base =
            (geometry.row_start + local_row) * geometry.total_blocks + geometry.block_start;
        let row_blocks = tensor_data
            [row_block_base * 28..(row_block_base + geometry.chunk_blocks) * 28]
            .as_chunks::<28>()
            .0;
        for (original, pair_buf) in row_blocks.iter().zip(tile_pairs.iter_mut()) {
            pack_ptq1_tile_into(original, row_in_tile, pair_buf)?;
        }
    }
    Ok(())
}

fn pack_pq2_row_group(
    tensor_data: &[u8],
    geometry: ChunkGeometry,
    group_row_start: usize,
    group_rows: usize,
    tile_pairs: &mut [[u8; 1152]],
) -> Result<()> {
    #[inline]
    const fn permute_q2_dword(bytes: [u8; 8]) -> [u8; 8] {
        let mut packed = u64::from_le_bytes(bytes);
        let swap = (packed ^ (packed >> 6)) & 0x00cc_00cc_00cc_00cc;
        packed ^= swap ^ (swap << 6);
        let swap = (packed ^ (packed >> 12)) & 0x0000_f0f0_0000_f0f0;
        packed ^= swap ^ (swap << 12);
        packed.to_le_bytes()
    }

    for row_in_tile in 0..group_rows {
        let local_row = group_row_start + row_in_tile;
        let row_block_base =
            (geometry.row_start + local_row) * geometry.total_blocks + geometry.block_start;
        let row_blocks = tensor_data
            [row_block_base * 34..(row_block_base + geometry.chunk_blocks) * 34]
            .as_chunks::<34>()
            .0;
        let scale_pos0 = 512 + row_in_tile * 2;
        let scale_pos1 = 576 + 512 + row_in_tile * 2;
        let q_pos0 = row_in_tile * 4;
        let q_pos1 = 256 + row_in_tile * 4;
        let q_pos2 = 576 + row_in_tile * 4;
        let q_pos3 = 576 + 256 + row_in_tile * 4;
        for (original, pair_buf) in row_blocks.iter().zip(tile_pairs.iter_mut()) {
            if (u16::from_le_bytes([original[0], original[1]]) & 0x7c00) == 0x7c00 {
                return Err(invalid("nonfinite Hexagon Q2 scale"));
            }
            pair_buf[scale_pos0..scale_pos0 + 2].copy_from_slice(&original[..2]);
            pair_buf[scale_pos1..scale_pos1 + 2].copy_from_slice(&original[..2]);
            let b0 = permute_q2_dword([
                original[2],
                original[3],
                original[4],
                original[5],
                original[6],
                original[7],
                original[8],
                original[9],
            ]);
            let b1 = permute_q2_dword([
                original[10],
                original[11],
                original[12],
                original[13],
                original[14],
                original[15],
                original[16],
                original[17],
            ]);
            let b2 = permute_q2_dword([
                original[18],
                original[19],
                original[20],
                original[21],
                original[22],
                original[23],
                original[24],
                original[25],
            ]);
            let b3 = permute_q2_dword([
                original[26],
                original[27],
                original[28],
                original[29],
                original[30],
                original[31],
                original[32],
                original[33],
            ]);
            pair_buf[q_pos0..q_pos0 + 4].copy_from_slice(&b0[..4]);
            pair_buf[q_pos0 + 128..q_pos0 + 132].copy_from_slice(&b0[4..8]);
            pair_buf[q_pos1..q_pos1 + 4].copy_from_slice(&b1[..4]);
            pair_buf[q_pos1 + 128..q_pos1 + 132].copy_from_slice(&b1[4..8]);
            pair_buf[q_pos2..q_pos2 + 4].copy_from_slice(&b2[..4]);
            pair_buf[q_pos2 + 128..q_pos2 + 132].copy_from_slice(&b2[4..8]);
            pair_buf[q_pos3..q_pos3 + 4].copy_from_slice(&b3[..4]);
            pair_buf[q_pos3 + 128..q_pos3 + 132].copy_from_slice(&b3[4..8]);
        }
    }
    Ok(())
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
    let tile_pairs = group_weights.as_chunks_mut::<1152>().0;
    match geometry.kind {
        PTQ1_KIND => pack_ptq1_row_group(
            tensor_data,
            geometry,
            group_row_start,
            group_rows,
            tile_pairs,
        ),
        PQ2_KIND => pack_pq2_row_group(
            tensor_data,
            geometry,
            group_row_start,
            group_rows,
            tile_pairs,
        ),
        _ => Err(invalid("unsupported Hexagon native Q2 tensor type or rank")),
    }
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

const NATIVE_Q2_ARENA_CACHE_MAGIC: &[u8; 8] = b"SB2HQ2C1";
const NATIVE_Q2_ARENA_CACHE_HEADER_BYTES: usize = 8 + 4 + 32 + 4 + 32;
const NATIVE_Q2_ARENA_CACHE_PAYLOAD_SHA_OFFSET: u64 = 48;
const NATIVE_Q2_ARENA_CACHE_MAX_TENSORS: usize = 1024;
const NATIVE_Q2_ARENA_CACHE_MAX_NAME_BYTES: usize = 512;
const NATIVE_Q2_ARENA_CACHE_DIGEST_CHUNK_BYTES: usize = 65_536;
static NEXT_CACHE_TEMP_ID: AtomicU64 = AtomicU64::new(0);

/// Canonical index entry inside a [`NativeQ2ArenaCache`] binary file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeQ2ArenaCacheEntry {
    /// Original model tensor name, matching `Tensor::name`.
    pub name: String,
    /// Source tensor encoding (`142` for `PQ2_0`, `143` for `PTQ1_0`).
    pub kind: u32,
    /// Original active input-column extent (`K`), always a positive multiple of `128`.
    pub columns: usize,
    /// Original active output-row extent (`M`).
    pub rows: usize,
    /// Byte offset from the start of the contiguous payload region.
    pub offset: u64,
    /// Exact packed `HTP_TYPE_Q2_0` payload byte length (`rows.next_multiple_of(32) * 36 * (columns / 128)`).
    pub byte_len: usize,
}

/// Digest-verified persistent binary cache of pre-transcoded native `HTP_TYPE_Q2_0` weights.
///
/// Format (`"SB2HQ2C1"`):
/// - `0..8`: magic `"SB2HQ2C1"`
/// - `8..12`: HTP architecture (`u32` LE: `73` or `81`)
/// - `12..44`: bound model + tensor-kind/geometry key SHA-256 (`[u8; 32]`)
/// - `44..48`: tensor entry count (`u32` LE)
/// - `48..80`: full SHA-256 digest over the contiguous payload region
/// - `80..`: `tensor_count` entries of `(name_len u32 LE, name utf8, kind u32 LE, columns u64 LE, rows u64 LE, offset u64 LE, byte_len u64 LE)`
/// - contiguous packed `HTP_TYPE_Q2_0` tensor payloads
pub struct NativeQ2ArenaCache;

fn cache_binding_sha256(model_sha256: [u8; 32], tensors: &[Tensor]) -> Result<[u8; 32]> {
    let mut hasher = bonsai_native_hash::Hasher::new()?;
    hasher.update(b"SB2HQ2C1-binding-v2\0")?;
    hasher.update(&model_sha256)?;
    hasher.update(&u64::try_from(tensors.len())?.to_le_bytes())?;
    for tensor in tensors {
        hasher.update(&u64::try_from(tensor.name.len())?.to_le_bytes())?;
        hasher.update(tensor.name.as_bytes())?;
        hasher.update(&tensor.kind.to_le_bytes())?;
        hasher.update(&u64::try_from(tensor.dimensions.len())?.to_le_bytes())?;
        for &dimension in &tensor.dimensions {
            hasher.update(&u64::try_from(dimension)?.to_le_bytes())?;
        }
    }
    Ok(hasher.finish()?)
}

struct TempCacheFileGuard {
    path: PathBuf,
    armed: bool,
}

impl TempCacheFileGuard {
    const fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    const fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for TempCacheFileGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

fn cache_temp_path(dest: &Path) -> PathBuf {
    let seq = NEXT_CACHE_TEMP_ID.fetch_add(1, Ordering::Relaxed);
    let mut os = dest.as_os_str().to_os_string();
    os.push(format!(".tmp.{}.{seq}", std::process::id()));
    PathBuf::from(os)
}

fn cache_tensor_geometry(
    tensor: &Tensor,
    max_columns: usize,
    max_rows: usize,
) -> Result<(usize, usize, usize, usize)> {
    let (columns, rows) = native_dimensions(tensor)?;
    let mapped_rows = rows.next_multiple_of(32);
    if columns > max_columns || mapped_rows > max_rows {
        return Err(invalid(
            "native Q2 arena cache tensor exceeds single-op native geometry",
        ));
    }
    let blocks = columns / 128;
    let packed_bytes = mapped_rows
        .checked_mul(36)
        .and_then(|bytes| bytes.checked_mul(blocks))
        .ok_or_else(|| invalid("native Q2 arena cache packed size overflow"))?;
    Ok((columns, rows, mapped_rows, packed_bytes))
}

fn read_exact_bytes<const N: usize>(reader: &mut impl Read) -> Result<[u8; N]> {
    let mut buf = [0_u8; N];
    reader.read_exact(&mut buf)?;
    Ok(buf)
}

fn read_u32_le(reader: &mut impl Read) -> Result<u32> {
    Ok(u32::from_le_bytes(read_exact_bytes::<4>(reader)?))
}

fn read_u64_le(reader: &mut impl Read) -> Result<u64> {
    Ok(u64::from_le_bytes(read_exact_bytes::<8>(reader)?))
}

struct CacheHeader {
    architecture: u32,
    model_sha256: [u8; 32],
    tensor_count: usize,
    payload_sha256: [u8; 32],
}

fn read_cache_header(reader: &mut impl Read) -> Result<CacheHeader> {
    let magic = read_exact_bytes::<8>(reader)?;
    if &magic != NATIVE_Q2_ARENA_CACHE_MAGIC {
        return Err(invalid("native Q2 arena cache magic mismatch"));
    }
    let architecture = read_u32_le(reader)?;
    if !matches!(architecture, 73 | 81) {
        return Err(invalid("unsupported native Q2 arena cache architecture"));
    }
    let model_sha256 = read_exact_bytes::<32>(reader)?;
    let count_u32 = read_u32_le(reader)?;
    let tensor_count = usize::try_from(count_u32)
        .map_err(|_| invalid("native Q2 arena cache tensor count overflow"))?;
    if tensor_count == 0 || tensor_count > NATIVE_Q2_ARENA_CACHE_MAX_TENSORS {
        return Err(invalid("invalid native Q2 arena cache tensor count"));
    }
    let payload_sha256 = read_exact_bytes::<32>(reader)?;
    Ok(CacheHeader {
        architecture,
        model_sha256,
        tensor_count,
        payload_sha256,
    })
}

fn read_and_validate_cache_entries<S: NativeQ2Submitter>(
    reader: &mut impl Read,
    executor: &NativeQ2Executor<S>,
    tensors: &[Tensor],
) -> Result<(Vec<NativeQ2ArenaCacheEntry>, Vec<usize>, usize)> {
    let mut entries = Vec::with_capacity(tensors.len());
    let mut mapped_rows_list = Vec::with_capacity(tensors.len());
    let mut seen_names = BTreeSet::new();
    let mut running_offset = 0_u64;
    let mut total_payload_bytes = 0_usize;
    for tensor in tensors {
        if !executor.supports_f32(tensor) {
            return Err(invalid("unsupported native Q2 tensor in arena cache load"));
        }
        let (expected_cols, expected_rows, mapped_rows, expected_len) =
            cache_tensor_geometry(tensor, executor.maximum_columns, executor.maximum_rows)?;
        let name_len_u32 = read_u32_le(reader)?;
        let name_len = usize::try_from(name_len_u32)
            .map_err(|_| invalid("native Q2 arena cache tensor name length overflow"))?;
        if name_len == 0 || name_len > NATIVE_Q2_ARENA_CACHE_MAX_NAME_BYTES {
            return Err(invalid("invalid native Q2 arena cache tensor name length"));
        }
        let mut name_bytes = vec![0_u8; name_len];
        reader.read_exact(&mut name_bytes)?;
        let name = String::from_utf8(name_bytes)
            .map_err(|_| invalid("non-UTF8 tensor name in native Q2 arena cache"))?;
        let kind = read_u32_le(reader)?;
        let cols_u64 = read_u64_le(reader)?;
        let rows_u64 = read_u64_le(reader)?;
        let offset = read_u64_le(reader)?;
        let byte_len_u64 = read_u64_le(reader)?;
        let columns = usize::try_from(cols_u64)
            .map_err(|_| invalid("native Q2 arena cache columns overflow"))?;
        let rows = usize::try_from(rows_u64)
            .map_err(|_| invalid("native Q2 arena cache rows overflow"))?;
        let byte_len = usize::try_from(byte_len_u64)
            .map_err(|_| invalid("native Q2 arena cache byte length overflow"))?;
        if name != tensor.name
            || !seen_names.insert(name.clone())
            || kind != tensor.kind
            || columns != expected_cols
            || rows != expected_rows
            || offset != running_offset
            || byte_len != expected_len
        {
            return Err(invalid("native Q2 arena cache entry mismatch"));
        }
        running_offset = running_offset
            .checked_add(byte_len_u64)
            .ok_or_else(|| invalid("native Q2 arena cache offset overflow"))?;
        total_payload_bytes = total_payload_bytes
            .checked_add(byte_len)
            .ok_or_else(|| invalid("native Q2 arena cache total payload overflow"))?;
        entries.push(NativeQ2ArenaCacheEntry {
            name,
            kind,
            columns,
            rows,
            offset,
            byte_len,
        });
        mapped_rows_list.push(mapped_rows);
    }
    Ok((entries, mapped_rows_list, total_payload_bytes))
}

impl NativeQ2ArenaCache {
    /// Transcode `tensors` into native `HTP_TYPE_Q2_0` tile order and atomically
    /// write a digest-verified binary cache at `path`.
    ///
    /// # Errors
    /// Rejects invalid architectures (`73 | 81`), empty/duplicate/over-limit
    /// tensor lists, unsupported tensor shapes, invalid PQ2 scales, or I/O and
    /// hashing failures, cleaning up any temporary file on failure.
    pub fn save_verified_cache(
        path: &Path,
        architecture: u32,
        model_sha256: [u8; 32],
        tensors: &[Tensor],
        row_parallelism: Option<&RowParallelism>,
    ) -> Result<Vec<NativeQ2ArenaCacheEntry>> {
        if !matches!(architecture, 73 | 81)
            || tensors.is_empty()
            || tensors.len() > NATIVE_Q2_ARENA_CACHE_MAX_TENSORS
        {
            return Err(invalid("invalid native Q2 arena cache save configuration"));
        }
        let mut seen_names = BTreeSet::new();
        let mut entries = Vec::with_capacity(tensors.len());
        let mut running_offset = 0_u64;
        for tensor in tensors {
            if tensor.name.is_empty()
                || tensor.name.len() > NATIVE_Q2_ARENA_CACHE_MAX_NAME_BYTES
                || !seen_names.insert(tensor.name.as_str())
            {
                return Err(invalid("invalid or duplicate tensor name for arena cache"));
            }
            let (columns, rows, _mapped_rows, packed_bytes) =
                cache_tensor_geometry(tensor, NATIVE_Q2_MAX_COLUMNS, NATIVE_Q2_MAX_ROWS)?;
            let byte_len_u64 = u64::try_from(packed_bytes)?;
            entries.push(NativeQ2ArenaCacheEntry {
                name: tensor.name.clone(),
                kind: tensor.kind,
                columns,
                rows,
                offset: running_offset,
                byte_len: packed_bytes,
            });
            running_offset = running_offset
                .checked_add(byte_len_u64)
                .ok_or_else(|| invalid("native Q2 arena cache total offset overflow"))?;
        }

        let binding_sha256 = cache_binding_sha256(model_sha256, tensors)?;
        let temp_path = cache_temp_path(path);
        let mut guard = TempCacheFileGuard::new(temp_path.clone());
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(NATIVE_Q2_ARENA_CACHE_MAGIC)?;
        writer.write_all(&architecture.to_le_bytes())?;
        writer.write_all(&binding_sha256)?;
        writer.write_all(&u32::try_from(tensors.len())?.to_le_bytes())?;
        writer.write_all(&[0_u8; 32])?;
        for entry in &entries {
            writer.write_all(&u32::try_from(entry.name.len())?.to_le_bytes())?;
            writer.write_all(entry.name.as_bytes())?;
            writer.write_all(&entry.kind.to_le_bytes())?;
            writer.write_all(&u64::try_from(entry.columns)?.to_le_bytes())?;
            writer.write_all(&u64::try_from(entry.rows)?.to_le_bytes())?;
            writer.write_all(&entry.offset.to_le_bytes())?;
            writer.write_all(&u64::try_from(entry.byte_len)?.to_le_bytes())?;
        }

        let mut hasher = bonsai_native_hash::Hasher::new()?;
        let mut scratch = Vec::new();
        for (tensor, entry) in tensors.iter().zip(&entries) {
            let geometry = ChunkGeometry {
                kind: tensor.kind,
                total_blocks: entry.columns / 128,
                row_start: 0,
                active_rows: entry.rows,
                block_start: 0,
                chunk_blocks: entry.columns / 128,
            };
            resize_packed_scratch(&mut scratch, entry.byte_len)?;
            let chunk = &mut scratch[..entry.byte_len];
            pack_native_chunk(&tensor.data, geometry, chunk, row_parallelism)?;
            hasher.update(chunk)?;
            writer.write_all(chunk)?;
        }
        let payload_sha256 = hasher.finish()?;
        let mut file = writer
            .into_inner()
            .map_err(|err| invalid(&format!("native Q2 arena cache flush failed: {err}")))?;
        file.seek(SeekFrom::Start(NATIVE_Q2_ARENA_CACHE_PAYLOAD_SHA_OFFSET))?;
        file.write_all(&payload_sha256)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&temp_path, path)?;
        guard.disarm();
        Ok(entries)
    }

    /// Verify every header field and the full SHA-256 payload digest of `path`
    /// before preparing any weights, then load all cached `HTP_TYPE_Q2_0`
    /// tensors directly into `executor`'s resident arena and bind `allowed_tensors`.
    ///
    /// # Errors
    /// Rejects faulted/used/already-bound executors, missing residency support,
    /// magic/architecture/model-SHA256/entry mismatches, truncated or oversized
    /// files, digest corruption (before calling `prepare_weights`), insufficient
    /// arena capacity, or native preparation failures.
    #[expect(
        clippy::too_many_lines,
        reason = "Keep strict pre-prepare SHA-256 verification and transactional resident arena loading in one cohesive entrypoint"
    )]
    pub fn load_and_prewarm_from_verified_cache<S: NativeQ2Submitter>(
        executor: &mut NativeQ2Executor<S>,
        path: &Path,
        expected_model_sha256: [u8; 32],
        tensors: &[Tensor],
    ) -> Result<Vec<NativeQ2ArenaCacheEntry>> {
        if executor.failed
            || executor.allowed_tensors.is_some()
            || executor.statistics.operations_attempted != 0
            || tensors.is_empty()
            || tensors.len() > NATIVE_Q2_ARENA_CACHE_MAX_TENSORS
        {
            return Err(invalid("invalid executor state for native Q2 arena cache"));
        }
        let (has_metadata_capacity, remaining_arena_bytes) = {
            let Some(cache) = &executor.residency else {
                return Err(invalid(
                    "native Q2 arena cache requires an executor with residency enabled",
                ));
            };
            if !cache.projections.is_empty() || !cache.handles.is_empty() {
                return Err(invalid(
                    "native Q2 arena cache requires an empty resident cache",
                ));
            }
            let fits_meta = cache.projections.capacity() - cache.projections.len() >= tensors.len()
                && cache.by_storage.capacity() - cache.by_storage.len() >= tensors.len()
                && cache.handles.capacity() - cache.handles.len() >= tensors.len();
            (fits_meta, executor.submitter.residency_capacity_bytes())
        };
        if !has_metadata_capacity {
            return Err(invalid(
                "native Q2 resident metadata capacity insufficient for arena cache",
            ));
        }

        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        if file_len < u64::try_from(NATIVE_Q2_ARENA_CACHE_HEADER_BYTES)? {
            return Err(invalid("native Q2 arena cache file is truncated"));
        }
        let mut reader = BufReader::new(file);
        let header = read_cache_header(&mut reader)?;
        let expected_binding_sha256 = cache_binding_sha256(expected_model_sha256, tensors)?;
        if header.architecture != executor.htp_architecture()
            || header.model_sha256 != expected_binding_sha256
            || header.tensor_count != tensors.len()
        {
            return Err(invalid("native Q2 arena cache identity mismatch"));
        }
        let (entries, mapped_rows_list, total_payload_bytes) =
            read_and_validate_cache_entries(&mut reader, executor, tensors)?;
        let payload_start = reader.stream_position()?;
        let total_payload_u64 = u64::try_from(total_payload_bytes)?;
        let expected_end = payload_start
            .checked_add(total_payload_u64)
            .ok_or_else(|| invalid("native Q2 arena cache file length overflow"))?;
        if expected_end != file_len {
            return Err(invalid("native Q2 arena cache file length mismatch"));
        }
        if total_payload_bytes > remaining_arena_bytes {
            return Err(invalid(
                "native Q2 resident arena capacity insufficient for arena cache",
            ));
        }

        // Verify the complete payload digest before invoking any submitter weight preparation.
        let mut digest_buf = vec![0_u8; NATIVE_Q2_ARENA_CACHE_DIGEST_CHUNK_BYTES];
        let actual_payload_sha256 = bonsai_native_hash::hash_reader(
            &mut reader,
            total_payload_u64,
            &mut digest_buf,
            &mut || Ok(()),
        )?;
        if actual_payload_sha256 != header.payload_sha256 {
            return Err(invalid("native Q2 arena cache payload SHA-256 mismatch"));
        }

        reader.seek(SeekFrom::Start(payload_start))?;
        let mut scratch = std::mem::take(&mut executor.scratch_weights);
        let load_result = (|| -> Result<Vec<Tensor>> {
            let mut selected = Vec::with_capacity(tensors.len());
            for ((tensor, entry), &mapped_rows) in
                tensors.iter().zip(&entries).zip(&mapped_rows_list)
            {
                let already_cached = executor
                    .residency
                    .as_ref()
                    .and_then(|cache| cache.find(tensor))
                    .is_some();
                if already_cached {
                    let skip_i64 = i64::try_from(entry.byte_len)?;
                    reader.seek_relative(skip_i64)?;
                    selected.push(tensor.clone());
                    continue;
                }
                let prepared = if executor.submitter.supports_in_place_weight_preparation() {
                    executor.submitter.prepare_weights_in_place(
                        entry.columns,
                        mapped_rows,
                        &mut |destination| {
                            if destination.len() != entry.byte_len {
                                return Err(hexagon_native_rpc::Error::Contract);
                            }
                            reader
                                .read_exact(destination)
                                .map_err(hexagon_native_rpc::Error::File)
                        },
                    )?
                } else {
                    resize_packed_scratch(&mut scratch, entry.byte_len)?;
                    let chunk = scratch
                        .get_mut(..entry.byte_len)
                        .ok_or_else(|| invalid("native Q2 cache scratch extent mismatch"))?;
                    reader.read_exact(chunk)?;
                    executor
                        .submitter
                        .prepare_weights(entry.columns, mapped_rows, chunk)?
                };
                let handle = prepared.ok_or_else(|| {
                    invalid("native Q2 resident preparation contradicted arena cache capacity")
                })?;
                if handle.columns() != entry.columns
                    || handle.rows() != mapped_rows
                    || handle.packed_bytes() != entry.byte_len
                {
                    return Err(invalid(
                        "native Q2 arena cache preparation returned mismatched handle geometry",
                    ));
                }
                let Some(cache) = &mut executor.residency else {
                    return Err(invalid("native Q2 resident cache missing during load"));
                };
                let first_handle = cache.handles.len();
                cache.handles.push(handle);
                cache.statistics.weight_payload_bytes_prepared = cache
                    .statistics
                    .weight_payload_bytes_prepared
                    .saturating_add(u64::try_from(entry.byte_len)?);
                let _ = cache.commit(tensor, first_handle, false);
                selected.push(tensor.clone());
            }
            Ok(selected)
        })();
        match load_result {
            Ok(selected) => {
                executor.scratch_weights = scratch;
                executor.bind_allowed_tensors(selected);
                Ok(entries)
            }
            Err(err) => {
                executor.failed = true;
                Err(err)
            }
        }
    }
}

/// Single resident `FastRPC` weight arena assignment produced by
/// [`plan_native_q2_layers_for_budget`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArenaAssignment {
    /// Zero-based resident arena index (`0..max_arenas`).
    pub arena_index: usize,
    /// Aligned resident `HTP_TYPE_Q2_0` payload bytes placed in this arena.
    pub used_bytes: usize,
    /// Unused bytes remaining under `max_arena_bytes` in this arena.
    pub remaining_bytes: usize,
    /// Ordered list of projection tensor names placed in this arena.
    pub projections: Vec<String>,
}

/// Dynamic `#41` VTCM-aware multi-arena Hexagon `native-q2` placement plan.
///
/// Inspects real model tensor dimensions and quantization kinds (`PQ2_0`
/// `kind == 142` and `PTQ1_0` `kind == 143`), validates per-projection VTCM
/// working-set fit against queried `vtcm_bytes`, and packs the maximal
/// contiguous decoder-layer prefix `blk.0..admitted_layers` across
/// `1..=max_arenas` arenas bounded by `max_arena_bytes`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VtcmPlacementPlan {
    /// Validated Hexagon HTP DSP architecture (`73` for X1 `v73`, `81` for X2 `v81`).
    pub architecture: u32,
    /// Total contiguous decoder layers found in the model (`blk.0..total_layers`).
    pub total_layers: usize,
    /// Number of contiguous prefix layers (`blk.0..admitted_layers`) admitted on Hexagon NPU.
    pub admitted_layers: usize,
    /// Total resident `HTP_TYPE_Q2_0` payload bytes across all admitted arenas.
    pub resident_bytes: usize,
    /// Usable HTP VTCM budget in bytes derived from queried `vtcm_bytes`.
    pub usable_vtcm_bytes: u32,
    /// Peak per-projection VTCM working-set footprint in bytes across NPU layers.
    pub peak_vtcm_working_bytes: u32,
    /// Ordered NPU-admitted projection tensor names (`blk.0..admitted_layers`).
    pub npu_projections: Vec<String>,
    /// Ordered overflow projection tensor names (`blk.admitted_layers..total_layers`).
    pub overflow_projections: Vec<String>,
    /// Non-empty per-arena breakdown covering every admitted NPU projection.
    pub arena_assignments: Vec<ArenaAssignment>,
}

const GDN_LAYER_SUFFIXES: [&str; 6] = [
    "ffn_up.weight",
    "ffn_gate.weight",
    "ffn_down.weight",
    "attn_qkv.weight",
    "attn_gate.weight",
    "ssm_out.weight",
];

const FULL_ATTN_LAYER_SUFFIXES: [&str; 7] = [
    "ffn_up.weight",
    "ffn_gate.weight",
    "ffn_down.weight",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
];

struct PlannedProjection {
    name: String,
    columns: usize,
    padded_rows: usize,
    packed_bytes: usize,
}

fn parse_blk_projection_name(name: &str) -> Option<(usize, &str)> {
    let rest = name.strip_prefix("blk.")?;
    let (digits, suffix) = rest.split_once('.')?;
    if digits.is_empty()
        || suffix.is_empty()
        || !digits.bytes().all(|byte| byte.is_ascii_digit())
        || (digits.len() > 1 && digits.starts_with('0'))
    {
        return None;
    }
    let layer = digits.parse::<usize>().ok()?;
    Some((layer, suffix))
}

fn is_canonical_layer_projection_suffix(suffix: &str) -> bool {
    GDN_LAYER_SUFFIXES.contains(&suffix) || FULL_ATTN_LAYER_SUFFIXES.contains(&suffix)
}

fn ordered_layer_projections(
    mut layer_map: BTreeMap<&str, PlannedProjection>,
) -> Result<Vec<PlannedProjection>> {
    let ordered_suffixes: &[&str] = if layer_map.len() == GDN_LAYER_SUFFIXES.len()
        && GDN_LAYER_SUFFIXES
            .iter()
            .all(|suffix| layer_map.contains_key(suffix))
    {
        &GDN_LAYER_SUFFIXES
    } else if layer_map.len() == FULL_ATTN_LAYER_SUFFIXES.len()
        && FULL_ATTN_LAYER_SUFFIXES
            .iter()
            .all(|suffix| layer_map.contains_key(suffix))
    {
        &FULL_ATTN_LAYER_SUFFIXES
    } else {
        return Err(invalid(
            "decoder layer does not match canonical Swift 6- or 7-projection structure",
        ));
    };

    let mut ordered = Vec::with_capacity(ordered_suffixes.len());
    for &suffix in ordered_suffixes {
        let projection = layer_map
            .remove(suffix)
            .ok_or_else(|| invalid("missing canonical decoder layer projection"))?;
        ordered.push(projection);
    }
    Ok(ordered)
}

fn collect_planned_layers(model: &Model) -> Result<Vec<Vec<PlannedProjection>>> {
    let mut layers_map: BTreeMap<usize, BTreeMap<&str, PlannedProjection>> = BTreeMap::new();
    for (key, tensor) in &model.tensors {
        if key != &tensor.name {
            return Err(invalid("mismatched model tensor map key and tensor name"));
        }
        if !key.starts_with("blk.") {
            continue;
        }
        let parsed = parse_blk_projection_name(key);
        let is_quantized = tensor.kind == PQ2_KIND || tensor.kind == PTQ1_KIND;
        if !is_quantized {
            if parsed.is_some_and(|(_, suffix)| is_canonical_layer_projection_suffix(suffix)) {
                return Err(invalid(
                    "canonical decoder projection must use PQ2_0 (142) or PTQ1_0 (143)",
                ));
            }
            continue;
        }
        let Some((layer_idx, suffix)) = parsed else {
            return Err(invalid("invalid quantized decoder tensor name format"));
        };
        if !is_canonical_layer_projection_suffix(suffix) {
            return Err(invalid("unrecognized quantized decoder projection suffix"));
        }
        let (columns, rows) = native_dimensions(tensor)?;
        let padded_rows = rows.next_multiple_of(32);
        let blocks = columns / 128;
        let packed_bytes = padded_rows
            .checked_mul(36)
            .and_then(|bytes| bytes.checked_mul(blocks))
            .ok_or_else(|| invalid("packed HTP_TYPE_Q2_0 projection size overflow"))?;
        let layer_entry = layers_map.entry(layer_idx).or_default();
        if layer_entry
            .insert(
                suffix,
                PlannedProjection {
                    name: tensor.name.clone(),
                    columns,
                    padded_rows,
                    packed_bytes,
                },
            )
            .is_some()
        {
            return Err(invalid("duplicate quantized decoder projection in layer"));
        }
    }

    if layers_map.is_empty() {
        return Err(invalid(
            "model contains no PQ2_0 or PTQ1_0 decoder layer projections",
        ));
    }

    let mut layers = Vec::with_capacity(layers_map.len());
    for (expected_idx, (actual_idx, layer_map)) in layers_map.into_iter().enumerate() {
        if actual_idx != expected_idx {
            return Err(invalid(
                "non-contiguous decoder layer indices in Hexagon placement model",
            ));
        }
        layers.push(ordered_layer_projections(layer_map)?);
    }
    Ok(layers)
}

fn usable_vtcm_budget(vtcm_bytes: u32) -> Result<u32> {
    if vtcm_bytes == 0 {
        return Err(invalid("Hexagon VTCM capacity must be non-zero"));
    }
    let usable = if vtcm_bytes >= 8 * 1024 * 1024 {
        NATIVE_Q2_VTCM_REQUEST_BYTES
    } else {
        let headroom = vtcm_bytes / 8;
        vtcm_bytes
            .saturating_sub(headroom)
            .min(NATIVE_Q2_VTCM_REQUEST_BYTES)
    };
    if usable < 65_536 {
        return Err(invalid("Hexagon VTCM capacity below 64 KiB floor"));
    }
    Ok(usable)
}

fn projection_vtcm_working_bytes(
    profile: HtpArchProfile,
    usable_vtcm_bytes: u32,
    columns: usize,
    padded_rows: usize,
) -> Option<u32> {
    if columns == 0
        || columns > NATIVE_Q2_MAX_COLUMNS
        || !columns.is_multiple_of(128)
        || padded_rows == 0
        || padded_rows > NATIVE_Q2_MAX_ROWS
        || !padded_rows.is_multiple_of(32)
    {
        return None;
    }
    let desired_vtcm = if columns > 6144 {
        NATIVE_Q2_VTCM_REQUEST_BYTES
    } else {
        HMX_Q2_VTCM_REQUEST_BYTES
    };
    let vtcm_budget = usize::try_from(desired_vtcm.min(usable_vtcm_bytes)).ok()?;
    let n_threads = usize::try_from(profile.hvx_workers.max(1)).ok()?;
    let k_groups = ((columns + 127) >> 5) & !3;
    let src1_row_size = k_groups.checked_mul(1280)?;
    let k_tiles = (columns + 63) >> 6;
    let dst_row_bytes = padded_rows.checked_mul(4)?;
    let src1_vtcm = src1_row_size.next_multiple_of(256);
    let dst_vtcm = dst_row_bytes
        .div_ceil(n_threads)
        .next_multiple_of(128)
        .checked_mul(n_threads)?;
    let act_vtcm = columns.checked_mul(4)?.next_multiple_of(512);
    let mut n_prefetch = 16_usize;
    while n_prefetch > 0 {
        let src0_vtcm = n_prefetch
            .checked_mul(k_tiles)?
            .checked_mul(640)?
            .next_multiple_of(256)
            .checked_mul(n_threads)?;
        let total_vtcm = src1_vtcm.checked_add(dst_vtcm.checked_add(src0_vtcm)?.max(act_vtcm))?;
        if total_vtcm <= vtcm_budget {
            return u32::try_from(total_vtcm).ok();
        }
        n_prefetch >>= 1;
    }
    None
}

fn try_place_layer_in_arenas(
    layer: &[PlannedProjection],
    profile: HtpArchProfile,
    usable_vtcm_bytes: u32,
    max_arena_bytes: usize,
    arena_used: &mut [usize],
    active_arena: &mut usize,
) -> Option<(Vec<(usize, String)>, u32)> {
    let mut trial_used = arena_used.to_vec();
    let mut trial_arena = *active_arena;
    let mut placements = Vec::with_capacity(layer.len());
    let mut layer_peak_vtcm = 0_u32;

    for proj in layer {
        if proj.packed_bytes > max_arena_bytes {
            return None;
        }
        let vtcm_work = projection_vtcm_working_bytes(
            profile,
            usable_vtcm_bytes,
            proj.columns,
            proj.padded_rows,
        )?;
        layer_peak_vtcm = layer_peak_vtcm.max(vtcm_work);

        let mut placed = false;
        for (idx, used) in trial_used.iter_mut().enumerate().skip(trial_arena) {
            let start = used.next_multiple_of(128);
            if let Some(end) = start.checked_add(proj.packed_bytes)
                && end <= max_arena_bytes
            {
                *used = end;
                trial_arena = idx;
                placements.push((idx, proj.name.clone()));
                placed = true;
                break;
            }
        }
        if !placed {
            return None;
        }
    }

    arena_used.copy_from_slice(&trial_used);
    *active_arena = trial_arena;
    Some((placements, layer_peak_vtcm))
}

/// Plan dynamic VTCM-aware multi-arena Hexagon `native-q2` layer placement from
/// actual `model` tensor geometry and queried hardware limits (`#41`).
///
/// Inspects every canonical `blk.<L>.<proj>` tensor (`PQ2_0` `kind == 142` or
/// `PTQ1_0` `kind == 143`), computes its exact transcoded `HTP_TYPE_Q2_0` tile
/// payload size, validates single-op geometry and per-tile VTCM working-set fit
/// against `vtcm_bytes`, and greedily packs the maximal contiguous layer prefix
/// `blk.0..admitted_layers` across up to `max_arenas` arenas bounded by
/// `max_arena_bytes`. Any remaining layers become [`VtcmPlacementPlan::overflow_projections`].
///
/// [`DEFAULT_NATIVE_Q2_PROJECTIONS`] remains untouched as the verified 27-layer
/// baseline.
///
/// # Errors
/// Rejects unsupported HTP architectures (`arch` outside `{73, 81}`), zero or
/// out-of-range `max_arena_bytes` / `max_arenas`, zero or sub-64 KiB
/// `vtcm_bytes`, malformed/non-contiguous `model` layers, or budgets that admit
/// zero complete layers.
pub fn plan_native_q2_layers_for_budget(
    model: &Model,
    arch: u32,
    max_arena_bytes: usize,
    max_arenas: usize,
    vtcm_bytes: u32,
) -> Result<VtcmPlacementPlan> {
    if !is_supported_htp_architecture(arch) {
        return Err(invalid(
            "unsupported Hexagon HTP architecture for native Q2 planning",
        ));
    }
    if max_arena_bytes == 0 || max_arena_bytes > NATIVE_Q2_MAX_ARENA_WEIGHT_BYTES {
        return Err(invalid("invalid Hexagon max_arena_bytes budget"));
    }
    if max_arenas == 0 || max_arenas > NATIVE_Q2_MAX_WEIGHT_ARENAS {
        return Err(invalid("invalid Hexagon max_arenas count"));
    }
    let usable_vtcm_bytes = usable_vtcm_budget(vtcm_bytes)?;
    let profile = HtpArchProfile::for_architecture(arch, 0, vtcm_bytes)
        .ok_or_else(|| invalid("unsupported Hexagon HTP architecture profile"))?;
    let layers = collect_planned_layers(model)?;
    let total_layers = layers.len();

    let mut arena_used = vec![0_usize; max_arenas];
    let mut arena_projections = vec![Vec::<String>::new(); max_arenas];
    let mut active_arena = 0_usize;
    let mut admitted_layers = 0_usize;
    let mut npu_projections = Vec::new();
    let mut overflow_projections = Vec::new();
    let mut peak_vtcm_working_bytes = 0_u32;
    let mut stopped = false;

    for (layer_idx, layer) in layers.into_iter().enumerate() {
        if !stopped
            && let Some((placements, layer_vtcm)) = try_place_layer_in_arenas(
                &layer,
                profile,
                usable_vtcm_bytes,
                max_arena_bytes,
                &mut arena_used,
                &mut active_arena,
            )
        {
            admitted_layers = layer_idx + 1;
            peak_vtcm_working_bytes = peak_vtcm_working_bytes.max(layer_vtcm);
            for (arena_idx, proj_name) in placements {
                arena_projections[arena_idx].push(proj_name.clone());
                npu_projections.push(proj_name);
            }
        } else {
            stopped = true;
            for proj in layer {
                overflow_projections.push(proj.name);
            }
        }
    }

    if admitted_layers == 0 {
        return Err(invalid(
            "Hexagon VTCM and arena budgets admit zero native Q2 layers",
        ));
    }

    let mut arena_assignments = Vec::new();
    let mut resident_bytes = 0_usize;
    for (arena_index, (used_bytes, projections)) in
        arena_used.into_iter().zip(arena_projections).enumerate()
    {
        if projections.is_empty() {
            continue;
        }
        resident_bytes = resident_bytes
            .checked_add(used_bytes)
            .ok_or_else(|| invalid("Hexagon resident arena bytes overflow"))?;
        arena_assignments.push(ArenaAssignment {
            arena_index,
            used_bytes,
            remaining_bytes: max_arena_bytes - used_bytes,
            projections,
        });
    }

    Ok(VtcmPlacementPlan {
        architecture: arch,
        total_layers,
        admitted_layers,
        resident_bytes,
        usable_vtcm_bytes,
        peak_vtcm_working_bytes,
        npu_projections,
        overflow_projections,
        arena_assignments,
    })
}

#[cfg(test)]
#[path = "native_q2/multitile_tests.rs"]
mod multitile_tests;

#[cfg(test)]
#[path = "native_q2/residency_tests.rs"]
mod residency_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    struct ArenaSubmitter {
        registry: hexagon_native_rpc::NativeQ2WeightRegistry,
        memory: Vec<u8>,
        direct: bool,
        copies: usize,
        initializations: usize,
    }

    impl NativeQ2Submitter for ArenaSubmitter {
        fn residency_capacity_bytes(&self) -> usize {
            self.registry.remaining_bytes()
        }

        fn supports_in_place_weight_preparation(&self) -> bool {
            self.direct
        }

        fn prepare_weights(
            &mut self,
            columns: usize,
            rows: usize,
            weights: &[u8],
        ) -> std::result::Result<Option<NativeQ2WeightHandle>, hexagon_native_rpc::Error> {
            let handle = self.registry.prepare(columns, rows, weights)?;
            if let Some(handle) = &handle {
                let range = self.registry.range(handle)?;
                self.memory[range].copy_from_slice(weights);
                self.copies += 1;
            }
            Ok(handle)
        }

        fn prepare_weights_in_place(
            &mut self,
            columns: usize,
            rows: usize,
            initialize: &mut dyn FnMut(
                &mut [u8],
            )
                -> std::result::Result<(), hexagon_native_rpc::Error>,
        ) -> std::result::Result<Option<NativeQ2WeightHandle>, hexagon_native_rpc::Error> {
            let start = self.registry.used_bytes().next_multiple_of(128);
            let end = start + rows * 36 * (columns / 128);
            let destination = self
                .memory
                .get_mut(start..end)
                .ok_or(hexagon_native_rpc::Error::Contract)?;
            self.initializations += 1;
            initialize(destination)?;
            self.registry.prepare(columns, rows, destination)
        }

        fn submit(
            &mut self,
            _columns: usize,
            _rows: usize,
            _weights: &[u8],
            _activation: &[f32],
            _output: &mut [f32],
        ) -> std::result::Result<(), hexagon_native_rpc::Error> {
            Err(hexagon_native_rpc::Error::Contract)
        }
    }

    fn arena_submitter(direct: bool) -> Result<ArenaSubmitter> {
        arena_submitter_at_capacity(direct, 8192)
    }

    fn arena_submitter_at_capacity(direct: bool, capacity: usize) -> Result<ArenaSubmitter> {
        Ok(ArenaSubmitter {
            registry: hexagon_native_rpc::NativeQ2WeightRegistry::new(capacity)
                .map_err(|_| invalid("registry"))?,
            memory: vec![0xa5; capacity],
            direct,
            copies: 0,
            initializations: 0,
        })
    }

    fn lazy_tensor(name: &str, columns: usize, rows: usize) -> Tensor {
        Tensor {
            name: name.to_owned(),
            dimensions: vec![columns, rows],
            kind: 142,
            data: Arc::<[u8]>::from(vec![0x55; columns / 128 * rows * 34]).into(),
        }
    }

    #[test]
    fn native_q2_lazy_preparation_allocates_only_copied_scratch_after_admission() -> Result<()> {
        let tensor = lazy_tensor("blk.0.ffn_up.weight", 256, 49);
        let limits = NativeQ2ProjectionLimits::new(128, 32, 100, Duration::from_secs(10))
            .map_err(|_| invalid("limits"))?;
        let mut copied = NativeQ2Executor::new(arena_submitter(false)?, limits, 1024 * 1024)
            .with_resident_weights(4096)?;
        let mut direct = NativeQ2Executor::new(arena_submitter(true)?, limits, 1024 * 1024)
            .with_resident_weights(4096)?;
        let mut copied_weights = Vec::new();
        let mut direct_weights = Vec::new();
        assert!(
            copied
                .prepare_resident(&tensor, &mut copied_weights, false)?
                .is_some()
        );
        assert!(
            direct
                .prepare_resident(&tensor, &mut direct_weights, false)?
                .is_some()
        );
        assert_eq!(direct.submitter.memory, copied.submitter.memory);
        assert_eq!(direct_weights.capacity(), 0);
        assert_eq!(copied_weights.len(), 1152);
        assert_eq!(direct.submitter.initializations, 4);
        assert_eq!(copied.submitter.copies, 4);

        let oversized = lazy_tensor("blk.0.ffn_gate.weight", 256, 81);
        for executor in [&mut copied, &mut direct] {
            let mut refused_weights = Vec::new();
            let bytes_before = executor.submitter.registry.used_bytes();
            assert!(
                executor
                    .prepare_resident(&oversized, &mut refused_weights, false)?
                    .is_none()
            );
            assert_eq!(refused_weights.capacity(), 0);
            assert_eq!(executor.submitter.registry.used_bytes(), bytes_before);
            assert_eq!(executor.residency_statistics().tensors_prepared, 1);
        }
        Ok(())
    }

    fn lazy_submit_failure(
        executor: &mut NativeQ2Executor<ArenaSubmitter>,
        tensors: [&Tensor; 3],
        mode: u8,
    ) -> Result<()> {
        let activation = vec![1.0; 128];
        let activations = vec![activation.clone(); 2];
        let mut single = vec![9.0];
        let mut second = vec![9.0];
        let mut third = vec![9.0];
        let mut first_tokens = vec![vec![9.0]; 2];
        let mut second_tokens = first_tokens.clone();
        let mut third_tokens = first_tokens.clone();
        let result = match mode {
            0 => executor.project_f32(tensors[0], &activation, true, &mut single),
            1 => executor.project_f32_group(
                &tensors,
                &activation,
                true,
                &mut [&mut single, &mut second, &mut third],
            ),
            2 => executor.project_f32_tokens(tensors[0], &activations, true, &mut first_tokens),
            3 => executor.project_f32_pair_tokens(
                [tensors[0], tensors[1]],
                &activations,
                true,
                &mut first_tokens,
                &mut second_tokens,
            ),
            4 => executor.project_f32_triple_tokens(
                tensors,
                &activations,
                true,
                &mut first_tokens,
                &mut second_tokens,
                &mut third_tokens,
            ),
            5 => executor.project_f32_ffn_swiglu(tensors[0], tensors[1], &activation, &mut single),
            6 => executor.project_f32_ffn_swiglu_tokens(
                tensors[0],
                tensors[1],
                &activations,
                &mut first_tokens,
            ),
            7 => executor.project_f32_ffn_chain(tensors, &activation, None, None, &mut single),
            _ => return Err(invalid("invalid lazy projection test mode")),
        };
        assert!(result.is_err());
        assert!(executor.is_faulted());
        assert_eq!(executor.statistics().operations_completed, 0);
        assert_eq!(executor.queue_writes_completed(), 0);
        assert_lazy_failure_outputs(
            mode,
            [&single, &second, &third],
            [&first_tokens, &second_tokens, &third_tokens],
        );
        Ok(())
    }

    fn assert_lazy_failure_outputs(mode: u8, singles: [&[f32]; 3], tokens: [&[Vec<f32>]; 3]) {
        for output in singles
            .into_iter()
            .chain(tokens.into_iter().flatten().map(Vec::as_slice))
        {
            assert!(
                output.is_empty()
                    || (output.len() == 1 && output[0].to_bits() == 9.0_f32.to_bits())
            );
        }
        // Every output selected for this entrypoint must be cleared on failure.
        if matches!(mode, 0 | 1 | 5 | 7) {
            let empty: &[f32] = &[];
            assert_eq!(singles[0], empty);
        }
        if mode == 1 {
            assert!(singles[1].is_empty() && singles[2].is_empty());
        }
        if matches!(mode, 2 | 3 | 4 | 6) {
            assert!(tokens[0].iter().all(Vec::is_empty));
        }
        if matches!(mode, 3 | 4) {
            assert!(tokens[1].iter().all(Vec::is_empty));
        }
        if mode == 4 {
            assert!(tokens[2].iter().all(Vec::is_empty));
        }
    }

    #[test]
    fn native_q2_lazy_entrypoints_match_copied_bytes_and_clear_failed_publication() -> Result<()> {
        let gate = lazy_tensor("blk.0.ffn_gate.weight", 128, 128);
        let up = lazy_tensor("blk.0.ffn_up.weight", 128, 128);
        let down = lazy_tensor("blk.0.ffn_down.weight", 128, 128);
        let limits = NativeQ2ProjectionLimits::new(128, 128, 100, Duration::from_secs(10))
            .map_err(|_| invalid("limits"))?;
        for mode in 0..8 {
            let mut copied = NativeQ2Executor::new(
                arena_submitter_at_capacity(false, 32_768)?,
                limits,
                1024 * 1024,
            )
            .with_resident_weights(4096)?;
            let mut direct = NativeQ2Executor::new(
                arena_submitter_at_capacity(true, 32_768)?,
                limits,
                1024 * 1024,
            )
            .with_resident_weights(4096)?;
            lazy_submit_failure(&mut copied, [&gate, &up, &down], mode)?;
            lazy_submit_failure(&mut direct, [&gate, &up, &down], mode)?;
            assert_eq!(direct.submitter.memory, copied.submitter.memory);
            assert_eq!(direct.submitter.copies, 0);
            assert_eq!(direct.submitter.initializations, copied.submitter.copies);
            assert_eq!(
                direct.submitter.registry.used_bytes(),
                copied.submitter.registry.used_bytes()
            );
            let prepared = match mode {
                0 | 2 => 1,
                3 | 5 | 6 => 2,
                _ => 3,
            };
            assert_eq!(direct.residency_statistics().tensors_prepared, prepared);
            assert_eq!(
                direct.statistics().operations_attempted,
                copied.statistics().operations_attempted
            );
        }
        Ok(())
    }

    #[test]
    fn native_q2_lazy_direct_scale_failure_commits_no_handle_or_output() -> Result<()> {
        let mut tensor = lazy_tensor("blk.0.ffn_up.weight", 128, 32);
        let mut data = tensor.data.to_vec();
        data[34..36].copy_from_slice(&0x7e00_u16.to_le_bytes());
        tensor.data = Arc::<[u8]>::from(data).into();
        let limits = NativeQ2ProjectionLimits::new(128, 32, 100, Duration::from_secs(10))
            .map_err(|_| invalid("limits"))?;
        let mut executor = NativeQ2Executor::new(arena_submitter(true)?, limits, 1024 * 1024)
            .with_resident_weights(4096)?;
        let mut output = vec![9.0];
        assert!(
            executor
                .project_f32(&tensor, &[1.0; 128], true, &mut output)
                .is_err()
        );
        assert_eq!(output, Vec::<f32>::new());
        assert!(executor.is_faulted());
        assert_eq!(executor.submitter.registry.used_bytes(), 0);
        assert_eq!(executor.submitter.initializations, 1);
        assert_eq!(executor.statistics().operations_attempted, 0);
        assert_eq!(executor.residency_statistics().tensors_prepared, 0);
        Ok(())
    }

    #[test]
    fn native_q2_direct_prewarm_matches_copied_tiles_without_scratch() -> Result<()> {
        let mut data: Vec<u8> = (0..256 * 49 * 34 / 128)
            .map(|index| u8::try_from((index * 173 + 29) & 255).unwrap_or(0))
            .collect();
        for block in data.as_chunks_mut::<34>().0 {
            block[..2].copy_from_slice(&0x3555_u16.to_le_bytes());
        }
        let tensor = Tensor {
            name: "blk.0.ffn_up.weight".to_owned(),
            dimensions: vec![256, 49],
            kind: 142,
            data: Arc::<[u8]>::from(data).into(),
        };
        let limits = NativeQ2ProjectionLimits::new(128, 32, 100, Duration::from_secs(10))
            .map_err(|_| invalid("limits"))?
            .with_resident_weights(8192)
            .map_err(|_| invalid("residency"))?;
        let mut copied = NativeQ2Executor::new(arena_submitter(false)?, limits, 1024 * 1024)
            .with_resident_weights(4096)?;
        let mut direct = NativeQ2Executor::new(arena_submitter(true)?, limits, 1024 * 1024)
            .with_resident_weights(4096)?;
        assert!(copied.prewarm_tensor(&tensor)?);
        assert!(direct.prewarm_tensor(&tensor)?);
        assert_eq!(direct.submitter.memory, copied.submitter.memory);
        assert_eq!(copied.submitter.copies, 4);
        assert_eq!(direct.submitter.copies, 0);
        assert_eq!(direct.submitter.initializations, 4);
        assert_eq!(direct.scratch_weights.capacity(), 0);
        assert_ne!(copied.scratch_weights, Vec::<u8>::new());
        assert_eq!(
            direct.residency_statistics().weight_payload_bytes_prepared,
            4608
        );
        assert_eq!(direct.statistics().operations_attempted, 0);
        assert!(direct.prewarm_tensor(&tensor)?);
        assert_eq!(direct.submitter.initializations, 4);
        Ok(())
    }

    #[test]
    fn native_q2_direct_prewarm_rejects_partially_packed_invalid_scale() -> Result<()> {
        let mut data = vec![0x55_u8; 32 * 34];
        for block in data.as_chunks_mut::<34>().0 {
            block[..2].copy_from_slice(&0x3800_u16.to_le_bytes());
        }
        data[34..36].copy_from_slice(&0x7e00_u16.to_le_bytes());
        let tensor = Tensor {
            name: "blk.0.ffn_up.weight".to_owned(),
            dimensions: vec![128, 32],
            kind: 142,
            data: Arc::<[u8]>::from(data).into(),
        };
        let limits = NativeQ2ProjectionLimits::new(128, 32, 100, Duration::from_secs(10))
            .map_err(|_| invalid("limits"))?
            .with_resident_weights(8192)
            .map_err(|_| invalid("residency"))?;
        let mut executor = NativeQ2Executor::new(arena_submitter(true)?, limits, 1024 * 1024)
            .with_resident_weights(4096)?;
        assert!(executor.prewarm_tensor(&tensor).is_err());
        assert!(executor.is_faulted());
        assert_eq!(executor.submitter.registry.used_bytes(), 0);
        assert_eq!(executor.residency_statistics().tensors_prepared, 0);
        assert_eq!(executor.submitter.initializations, 1);
        assert_eq!(executor.scratch_weights.capacity(), 0);
        assert!(executor.prewarm_tensor(&tensor).is_err());
        assert_eq!(executor.submitter.initializations, 1);
        Ok(())
    }

    struct MockNativeSubmitter {
        batch_ffn: bool,
        ffn_pair_calls: u64,
        calls: u64,
        resident_calls: u64,
        remaining: u64,
        residency_bytes: usize,
    }

    #[test]
    fn resident_cache_index_duplicate_siblings_prepare_one_extent() -> Result<()> {
        let tensor = lazy_tensor("duplicate-sibling", 128, 32);
        let limits = NativeQ2ProjectionLimits::new(128, 32, 10, Duration::from_secs(10))
            .map_err(|_| invalid("limits"))?
            .with_resident_weights(8192)
            .map_err(|_| invalid("residency"))?;
        let submitter = MockNativeSubmitter {
            batch_ffn: false,
            ffn_pair_calls: 0,
            calls: 0,
            resident_calls: 0,
            remaining: 10,
            residency_bytes: 8192,
        };
        let mut executor =
            NativeQ2Executor::new(submitter, limits, 1024 * 1024).with_resident_weights(4096)?;
        let mut first = Vec::new();
        let mut second = Vec::new();
        for _ in 0..2 {
            executor.project_f32_group(
                &[&tensor, &tensor],
                &[1.0; 128],
                true,
                &mut [&mut first, &mut second],
            )?;
            assert_eq!(first.len(), 32);
            assert_eq!(second.len(), 32);
            assert_eq!(first[0], 128.0);
            assert_eq!(second[0], 129.0);
            assert!(first[1..].iter().all(|value| value.to_bits() == 0));
            assert!(second[1..].iter().all(|value| value.to_bits() == 0));
            assert_eq!(executor.residency_statistics().tensors_prepared, 1);
            assert_eq!(
                executor
                    .residency_statistics()
                    .weight_payload_bytes_prepared,
                1152
            );
        }
        assert_eq!(executor.statistics().operations_attempted, 4);
        assert_eq!(executor.statistics().operations_completed, 4);
        assert_eq!(executor.operation_capacity(), Some(6));
        Ok(())
    }

    #[test]
    fn resident_cache_index_selected_aliases_admit_each_exact_name_and_geometry() -> Result<()> {
        let source = lazy_tensor("wide", 512, 32);
        let mut tensors = vec![source.clone()];
        for (name, dimensions) in [
            ("middle", [256, 64]),
            ("narrow", [128, 128]),
            ("alias", [512, 32]),
        ] {
            let mut tensor = source.clone();
            tensor.name = name.into();
            tensor.dimensions = dimensions.into();
            tensors.push(tensor);
        }
        let limits = NativeQ2ProjectionLimits::new(512, 128, 10, Duration::from_secs(10))
            .map_err(|_| invalid("limits"))?
            .with_resident_weights(32_768)
            .map_err(|_| invalid("residency"))?;
        let mut executor = NativeQ2Executor::new(
            arena_submitter_at_capacity(true, 32_768)?,
            limits,
            1024 * 1024,
        )
        .with_resident_weights(4096)?;
        executor.select_and_prewarm_tensors(&tensors)?;
        for tensor in &tensors {
            assert!(executor.supports_f32(tensor));
            assert!(executor.prewarm_tensor(tensor)?);
            let mut unselected = tensor.clone();
            unselected.name.push_str("-unselected");
            assert!(!executor.supports_f32(&unselected));
            unselected = tensor.clone();
            unselected.data = Arc::<[u8]>::from(tensor.data.to_vec()).into();
            assert!(!executor.supports_f32(&unselected));
        }
        assert_eq!(executor.residency_statistics().tensors_prepared, 3);
        assert_eq!(executor.submitter.registry.used_bytes(), 3 * 4608);
        assert_eq!(executor.statistics().operations_attempted, 0);
        Ok(())
    }

    #[expect(
        clippy::suspicious_operation_groupings,
        clippy::suboptimal_flops,
        reason = "Test mock compares handle geometry against activation/output slice lengths."
    )]
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

        fn submit_resident_multi2(
            &mut self,
            handles: [&NativeQ2WeightHandle; 2],
            activations: [&[f32]; 2],
            outputs: [&mut [f32]; 2],
        ) -> std::result::Result<(), hexagon_native_rpc::Error> {
            self.calls += 2;
            self.resident_calls += 2;
            for (member, handle) in handles.into_iter().enumerate() {
                let activation = activations[member];
                if activation.len() != handle.columns() || outputs[member].len() != handle.rows() {
                    return Err(hexagon_native_rpc::Error::Contract);
                }
                outputs[member].fill(0.0);
                outputs[member][0] =
                    activation.iter().sum::<f32>() + f32::from(u8::try_from(member).unwrap_or(0));
            }
            Ok(())
        }

        fn submit_resident_multi3(
            &mut self,
            handles: [&NativeQ2WeightHandle; 3],
            activations: [&[f32]; 3],
            outputs: [&mut [f32]; 3],
        ) -> std::result::Result<(), hexagon_native_rpc::Error> {
            self.calls += 3;
            self.resident_calls += 3;
            for (member, handle) in handles.into_iter().enumerate() {
                let activation = activations[member];
                if activation.len() != handle.columns() || outputs[member].len() != handle.rows() {
                    return Err(hexagon_native_rpc::Error::Contract);
                }
                outputs[member].fill(0.0);
                outputs[member][0] =
                    activation.iter().sum::<f32>() + f32::from(u8::try_from(member).unwrap_or(0));
            }
            Ok(())
        }

        fn submit_resident_multi4(
            &mut self,
            handles: [&NativeQ2WeightHandle; 4],
            activations: [&[f32]; 4],
            outputs: [&mut [f32]; 4],
        ) -> std::result::Result<(), hexagon_native_rpc::Error> {
            self.calls += 4;
            self.resident_calls += 4;
            for (member, handle) in handles.into_iter().enumerate() {
                let activation = activations[member];
                if activation.len() != handle.columns() || outputs[member].len() != handle.rows() {
                    return Err(hexagon_native_rpc::Error::Contract);
                }
                outputs[member].fill(0.0);
                outputs[member][0] =
                    activation.iter().sum::<f32>() + f32::from(u8::try_from(member).unwrap_or(0));
            }
            Ok(())
        }

        fn submit_resident_multi8(
            &mut self,
            handles: [&NativeQ2WeightHandle; 8],
            activations: [&[f32]; 8],
            outputs: [&mut [f32]; 8],
        ) -> std::result::Result<(), hexagon_native_rpc::Error> {
            self.calls += 8;
            self.resident_calls += 8;
            for (member, handle) in handles.into_iter().enumerate() {
                let activation = activations[member];
                if activation.len() != handle.columns() || outputs[member].len() != handle.rows() {
                    return Err(hexagon_native_rpc::Error::Contract);
                }
                outputs[member].fill(0.0);
                outputs[member][0] =
                    activation.iter().sum::<f32>() + f32::from(u8::try_from(member).unwrap_or(0));
            }
            Ok(())
        }

        fn submit_resident_ffn_swiglu(
            &mut self,
            gate_handle: &NativeQ2WeightHandle,
            up_handle: &NativeQ2WeightHandle,
            activation: &[f32],
            output: &mut [f32],
        ) -> std::result::Result<(), hexagon_native_rpc::Error> {
            self.calls += 3;
            self.resident_calls += 2;
            if gate_handle.columns() != up_handle.columns()
                || gate_handle.rows() != up_handle.rows()
                || activation.len() != gate_handle.columns()
                || output.len() != gate_handle.rows()
            {
                return Err(hexagon_native_rpc::Error::Contract);
            }
            let sum = activation.iter().sum::<f32>();
            output.fill(0.0);
            output[0] = sum * 2.0;
            Ok(())
        }

        fn submit_resident_ffn_chain(
            &mut self,
            handles: [&NativeQ2WeightHandle; 3],
            activation: &[f32],
            residual: Option<&[f32]>,
            hadamard_block_size: Option<u32>,
            output: &mut [f32],
        ) -> std::result::Result<(), hexagon_native_rpc::Error> {
            let ops =
                4_u64 + u64::from(hadamard_block_size.is_some()) + u64::from(residual.is_some());
            self.calls += ops;
            self.resident_calls += 3;
            let [gate, up, down] = handles;
            if gate.columns() != up.columns()
                || gate.rows() != up.rows()
                || down.columns() != gate.rows()
                || down.rows() != gate.columns()
                || activation.len() != gate.columns()
                || output.len() != down.rows()
            {
                return Err(hexagon_native_rpc::Error::Contract);
            }
            let sum = activation.iter().sum::<f32>();
            let res0 = residual.and_then(|r| r.first().copied()).unwrap_or(0.0);
            output.fill(0.0);
            output[0] = sum.mul_add(3.0, res0);
            Ok(())
        }

        fn supports_ffn_chain_multi2(&self) -> bool {
            self.batch_ffn
        }

        fn supports_ffn_chain_lora(&self) -> bool {
            true
        }

        fn submit_resident_ffn_chain_lora(
            &mut self,
            handles: [&NativeQ2WeightHandle; 3],
            activation: &[f32],
            residual: Option<&[f32]>,
            hadamard_block_size: Option<u32>,
            lora_a: &[f32],
            output: &mut [f32],
        ) -> std::result::Result<f32, hexagon_native_rpc::Error> {
            self.submit_resident_ffn_chain(
                handles,
                activation,
                residual,
                hadamard_block_size,
                output,
            )?;
            self.calls += 1;
            let ffn_dim = handles[2].columns();
            if lora_a.len() != ffn_dim {
                return Err(hexagon_native_rpc::Error::Contract);
            }
            Ok(lora_a.iter().sum::<f32>())
        }

        fn submit_resident_ffn_chain_multi2(
            &mut self,
            handles: [&NativeQ2WeightHandle; 3],
            activations: [&[f32]; 2],
            residuals: [Option<&[f32]>; 2],
            hadamard_block_size: Option<u32>,
            outputs: [&mut [f32]; 2],
        ) -> std::result::Result<(), hexagon_native_rpc::Error> {
            self.ffn_pair_calls += 1;
            for ((activation, residual), output) in
                activations.into_iter().zip(residuals).zip(outputs)
            {
                self.submit_resident_ffn_chain(
                    handles,
                    activation,
                    residual,
                    hadamard_block_size,
                    output,
                )?;
            }
            Ok(())
        }

        fn submit_resident_ffn_swiglu_multi2(
            &mut self,
            gate_handle: &NativeQ2WeightHandle,
            up_handle: &NativeQ2WeightHandle,
            activations: [&[f32]; 2],
            outputs: [&mut [f32]; 2],
        ) -> std::result::Result<(), hexagon_native_rpc::Error> {
            self.calls += 6;
            self.resident_calls += 4;
            if gate_handle.columns() != up_handle.columns()
                || gate_handle.rows() != up_handle.rows()
            {
                return Err(hexagon_native_rpc::Error::Contract);
            }
            for token in 0..2 {
                if activations[token].len() != gate_handle.columns()
                    || outputs[token].len() != gate_handle.rows()
                {
                    return Err(hexagon_native_rpc::Error::Contract);
                }
                let sum = activations[token].iter().sum::<f32>();
                outputs[token].fill(0.0);
                outputs[token][0] = sum * 2.0 + f32::from(u8::try_from(token).unwrap_or(0));
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

    struct ResidentTokenCapture {
        registry: hexagon_native_rpc::NativeQ2WeightRegistry,
        batches: Vec<Vec<(usize, u32)>>,
        fail_at: Option<usize>,
        nonfinite_at: Option<usize>,
    }

    impl ResidentTokenCapture {
        fn capture<const B: usize>(
            &mut self,
            handles: [&NativeQ2WeightHandle; B],
            activations: [&[f32]; B],
            outputs: [&mut [f32]; B],
        ) -> std::result::Result<(), hexagon_native_rpc::Error> {
            let mut batch = Vec::new();
            for ((handle, activation), output) in handles.into_iter().zip(activations).zip(outputs)
            {
                self.registry.range(handle)?;
                if activation.len() != handle.columns() || output.len() != handle.rows() {
                    return Err(hexagon_native_rpc::Error::Contract);
                }
                batch.push((handle.rows(), activation[0].to_bits()));
                let rows = u16::try_from(handle.rows())
                    .map_err(|_| hexagon_native_rpc::Error::Contract)?;
                output.fill(activation[0] + f32::from(rows));
                if self.nonfinite_at == Some(self.batches.len() + 1) {
                    output[0] = f32::NAN;
                }
            }
            self.batches.push(batch);
            if self.fail_at == Some(self.batches.len()) {
                return Err(hexagon_native_rpc::Error::Contract);
            }
            Ok(())
        }
    }

    impl NativeQ2Submitter for ResidentTokenCapture {
        fn residency_capacity_bytes(&self) -> usize {
            self.registry.remaining_bytes()
        }

        fn prepare_weights(
            &mut self,
            columns: usize,
            rows: usize,
            weights: &[u8],
        ) -> std::result::Result<Option<NativeQ2WeightHandle>, hexagon_native_rpc::Error> {
            self.registry.prepare(columns, rows, weights)
        }

        fn submit(
            &mut self,
            _columns: usize,
            _rows: usize,
            _weights: &[u8],
            _activation: &[f32],
            _output: &mut [f32],
        ) -> std::result::Result<(), hexagon_native_rpc::Error> {
            Err(hexagon_native_rpc::Error::Contract)
        }

        fn submit_resident(
            &mut self,
            handle: &NativeQ2WeightHandle,
            activation: &[f32],
            output: &mut [f32],
        ) -> std::result::Result<(), hexagon_native_rpc::Error> {
            self.capture([handle], [activation], [output])
        }

        fn submit_resident_batch2(
            &mut self,
            handles: [&NativeQ2WeightHandle; 2],
            activation: &[f32],
            outputs: [&mut [f32]; 2],
        ) -> std::result::Result<(), hexagon_native_rpc::Error> {
            self.capture(handles, [activation; 2], outputs)
        }

        fn submit_resident_batch3(
            &mut self,
            handles: [&NativeQ2WeightHandle; 3],
            activation: &[f32],
            outputs: [&mut [f32]; 3],
        ) -> std::result::Result<(), hexagon_native_rpc::Error> {
            self.capture(handles, [activation; 3], outputs)
        }

        fn submit_resident_multi4(
            &mut self,
            handles: [&NativeQ2WeightHandle; 4],
            activations: [&[f32]; 4],
            outputs: [&mut [f32]; 4],
        ) -> std::result::Result<(), hexagon_native_rpc::Error> {
            self.capture(handles, activations, outputs)
        }

        fn submit_resident_multi8(
            &mut self,
            handles: [&NativeQ2WeightHandle; 8],
            activations: [&[f32]; 8],
            outputs: [&mut [f32]; 8],
        ) -> std::result::Result<(), hexagon_native_rpc::Error> {
            self.capture(handles, activations, outputs)
        }
    }

    fn resident_token_executor() -> Result<NativeQ2Executor<ResidentTokenCapture>> {
        let limits = NativeQ2ProjectionLimits::new(128, 128, 1000, Duration::from_secs(10))
            .map_err(|_| invalid("limits"))?
            .with_resident_weights(65_536)
            .map_err(|_| invalid("residency"))?;
        let submitter = ResidentTokenCapture {
            registry: hexagon_native_rpc::NativeQ2WeightRegistry::new(65_536)
                .map_err(|_| invalid("registry"))?,
            batches: Vec::new(),
            fail_at: None,
            nonfinite_at: None,
        };
        NativeQ2Executor::new(submitter, limits, 1024 * 1024).with_resident_weights(4096)
    }

    fn project_output_evidence_family<S: NativeQ2Submitter>(
        executor: &mut NativeQ2Executor<S>,
        tensors: &[Tensor; 3],
        outputs: &mut [Vec<f32>; 3],
        members: usize,
    ) -> Result<()> {
        let activation = [2.0_f32; 128];
        let [out0, out1, out2] = outputs;
        match members {
            1 => executor.project_f32(&tensors[0], &activation, false, out0),
            2 => executor.project_f32_group(
                &[&tensors[0], &tensors[1]],
                &activation,
                false,
                &mut [out0, out1],
            ),
            3 => executor.project_f32_group(
                &[&tensors[0], &tensors[1], &tensors[2]],
                &activation,
                false,
                &mut [out0, out1, out2],
            ),
            _ => Err(invalid("invalid output evidence test family")),
        }
    }

    #[test]
    fn native_q2_default_output_evidence_preserves_success_and_terminal_failures() -> Result<()> {
        let tensors = [
            lazy_tensor("first", 128, 32),
            lazy_tensor("second", 128, 64),
            lazy_tensor("third", 128, 96),
        ];
        for members in 1..=3 {
            for case in 0..3 {
                let mut executor = resident_token_executor()?;
                executor.submitter.nonfinite_at = (case == 1).then_some(1);
                executor.submitter.fail_at = (case == 2).then_some(1);
                let mut outputs = [Vec::new(), Vec::new(), Vec::new()];
                let result =
                    project_output_evidence_family(&mut executor, &tensors, &mut outputs, members);
                assert_eq!(result.is_ok(), case == 0);
                assert_eq!(
                    executor.statistics.operations_attempted,
                    u64::try_from(members)?
                );
                assert_eq!(executor.queue_writes_attempted, 1);
                if case == 0 {
                    for (tensor, output) in tensors.iter().zip(&outputs).take(members) {
                        let rows = u16::try_from(tensor.dimensions[1])?;
                        let expected = (2.0_f32 + f32::from(rows)).to_bits();
                        assert_eq!(output.len(), usize::from(rows));
                        assert!(output.iter().all(|value| value.to_bits() == expected));
                    }
                    assert_eq!(
                        executor.statistics.operations_completed,
                        u64::try_from(members)?
                    );
                    assert_eq!(executor.queue_writes_completed, 1);
                } else {
                    assert!(executor.is_faulted());
                    assert!(outputs.iter().all(Vec::is_empty));
                    assert_eq!(executor.statistics.operations_completed, 0);
                    assert_eq!(executor.queue_writes_completed, 0);
                    assert!(
                        project_output_evidence_family(
                            &mut executor,
                            &tensors,
                            &mut outputs,
                            members
                        )
                        .is_err()
                    );
                    assert_eq!(executor.submitter.batches.len(), 1);
                }
            }
        }
        Ok(())
    }

    struct UnrelatedOutputEvidence {
        inner: ResidentTokenCapture,
    }

    impl NativeQ2Submitter for UnrelatedOutputEvidence {
        fn residency_capacity_bytes(&self) -> usize {
            self.inner.residency_capacity_bytes()
        }

        fn prepare_weights(
            &mut self,
            columns: usize,
            rows: usize,
            weights: &[u8],
        ) -> std::result::Result<Option<NativeQ2WeightHandle>, hexagon_native_rpc::Error> {
            self.inner.prepare_weights(columns, rows, weights)
        }

        fn submit_resident_validated<'a>(
            &mut self,
            handle: &NativeQ2WeightHandle,
            activation: &[f32],
            output: &'a mut [f32],
        ) -> std::result::Result<
            hexagon_native_rpc::NativeQ2ValidatedOutputs<'a, 1>,
            hexagon_native_rpc::Error,
        > {
            static OTHER: [f32; 32] = [0.0; 32];
            self.inner.submit_resident(handle, activation, output)?;
            hexagon_native_rpc::NativeQ2ValidatedOutputs::validate([&OTHER])
        }

        fn submit_resident_batch2_validated<'a>(
            &mut self,
            handles: [&NativeQ2WeightHandle; 2],
            activation: &[f32],
            outputs: [&'a mut [f32]; 2],
        ) -> std::result::Result<
            hexagon_native_rpc::NativeQ2ValidatedOutputs<'a, 2>,
            hexagon_native_rpc::Error,
        > {
            let [out0, out1] = outputs;
            self.inner
                .submit_resident_batch2(handles, activation, [&mut *out0, &mut *out1])?;
            hexagon_native_rpc::NativeQ2ValidatedOutputs::validate([out1, out0])
        }

        fn submit_resident_batch3_validated<'a>(
            &mut self,
            handles: [&NativeQ2WeightHandle; 3],
            activation: &[f32],
            outputs: [&'a mut [f32]; 3],
        ) -> std::result::Result<
            hexagon_native_rpc::NativeQ2ValidatedOutputs<'a, 3>,
            hexagon_native_rpc::Error,
        > {
            let [out0, out1, out2] = outputs;
            self.inner.submit_resident_batch3(
                handles,
                activation,
                [&mut *out0, &mut *out1, &mut *out2],
            )?;
            hexagon_native_rpc::NativeQ2ValidatedOutputs::validate([
                out0,
                out1,
                &out2[..out2.len() - 1],
            ])
        }

        fn submit(
            &mut self,
            columns: usize,
            rows: usize,
            weights: &[u8],
            activation: &[f32],
            output: &mut [f32],
        ) -> std::result::Result<(), hexagon_native_rpc::Error> {
            self.inner
                .submit(columns, rows, weights, activation, output)
        }
    }

    #[test]
    fn native_q2_unrelated_output_evidence_cannot_publish() -> Result<()> {
        let tensors = [
            lazy_tensor("first", 128, 32),
            lazy_tensor("second", 128, 64),
            lazy_tensor("third", 128, 96),
        ];
        for members in 1..=3 {
            let ordinary = resident_token_executor()?;
            let limits = NativeQ2ProjectionLimits::new(128, 128, 1000, Duration::from_secs(10))
                .map_err(|_| invalid("limits"))?
                .with_resident_weights(65_536)
                .map_err(|_| invalid("residency"))?;
            let submitter = UnrelatedOutputEvidence {
                inner: ordinary.submitter,
            };
            let mut executor = NativeQ2Executor::new(submitter, limits, 1024 * 1024)
                .with_resident_weights(4096)?;
            let mut outputs = [Vec::new(), Vec::new(), Vec::new()];
            assert!(
                project_output_evidence_family(&mut executor, &tensors, &mut outputs, members)
                    .is_err()
            );
            assert!(executor.is_faulted());
            assert!(outputs.iter().all(Vec::is_empty));
            assert_eq!(
                executor.statistics.operations_attempted,
                u64::try_from(members)?
            );
            assert_eq!(executor.statistics.operations_completed, 0);
            assert_eq!(executor.queue_writes_completed, 0);
            assert!(
                project_output_evidence_family(&mut executor, &tensors, &mut outputs, members)
                    .is_err()
            );
            assert_eq!(executor.submitter.inner.batches.len(), 1);
        }
        Ok(())
    }

    fn expected_sibling_batches(members: usize, tokens: usize) -> Vec<usize> {
        let mut batches = Vec::new();
        let mut remaining = tokens;
        if members == 3 {
            while remaining >= 8 {
                batches.extend_from_slice(&[8, 8, 8]);
                remaining -= 8;
            }
        }
        while remaining >= 4 {
            if members == 2 {
                batches.push(8);
            } else {
                batches.extend_from_slice(&[8, 4]);
            }
            remaining -= 4;
        }
        if remaining >= 2 {
            batches.push(4);
            if members == 3 {
                batches.push(2);
            }
            remaining -= 2;
        }
        if remaining == 1 {
            batches.push(members);
        }
        batches
    }

    #[test]
    fn native_q2_resident_sibling_four_token_batches_preserve_order_and_tails() -> Result<()> {
        let tensors = [
            lazy_tensor("first", 128, 32),
            lazy_tensor("second", 128, 64),
            lazy_tensor("third", 128, 96),
        ];
        for members in [2, 3] {
            for tokens in [0_u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 32] {
                let mut executor = resident_token_executor()?;
                let activations: Vec<_> = (0..tokens)
                    .map(|token| vec![f32::from(token); 128])
                    .collect();
                let mut first = vec![Vec::new(); usize::from(tokens)];
                let mut second = first.clone();
                let mut third = first.clone();
                if members == 2 {
                    executor.project_f32_pair_tokens(
                        [&tensors[0], &tensors[1]],
                        &activations,
                        true,
                        &mut first,
                        &mut second,
                    )?;
                } else {
                    executor.project_f32_triple_tokens(
                        [&tensors[0], &tensors[1], &tensors[2]],
                        &activations,
                        true,
                        &mut first,
                        &mut second,
                        &mut third,
                    )?;
                }
                let expected_batches = expected_sibling_batches(members, usize::from(tokens));
                let batches = &executor.submitter.batches;
                assert_eq!(
                    batches.iter().map(Vec::len).collect::<Vec<_>>(),
                    expected_batches
                );
                let expected: Vec<_> = (0..tokens)
                    .flat_map(|token| {
                        [32, 64, 96]
                            .into_iter()
                            .take(members)
                            .map(move |rows| (rows, f32::from(token).to_bits()))
                    })
                    .collect();
                assert_eq!(
                    batches.iter().flatten().copied().collect::<Vec<_>>(),
                    expected
                );
                assert_eq!(
                    executor.queue_writes_completed(),
                    u64::try_from(batches.len())?
                );
                assert_eq!(
                    executor.statistics().operations_completed,
                    u64::from(tokens) * u64::try_from(members)?
                );
                for (member, outputs) in [&first, &second, &third]
                    .into_iter()
                    .take(members)
                    .enumerate()
                {
                    let rows = (member + 1) * 32;
                    for (token, output) in (0..tokens).zip(outputs) {
                        assert_eq!(
                            output,
                            &vec![f32::from(token) + f32::from(u16::try_from(rows)?); rows]
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn native_q2_resident_sibling_eight_op_failure_clears_entire_tile() -> Result<()> {
        let tensors = [
            lazy_tensor("first", 128, 32),
            lazy_tensor("second", 128, 64),
            lazy_tensor("third", 128, 96),
        ];
        for members in [2, 3] {
            for tokens in [4_u8, 5, 8] {
                let expected_batches = expected_sibling_batches(members, usize::from(tokens));
                for fail_at in 1..=expected_batches.len() {
                    for nonfinite in [false, true] {
                        let mut executor = resident_token_executor()?;
                        if nonfinite {
                            executor.submitter.nonfinite_at = Some(fail_at);
                        } else {
                            executor.submitter.fail_at = Some(fail_at);
                        }
                        let activations: Vec<_> = (0..tokens)
                            .map(|token| vec![f32::from(token); 128])
                            .collect();
                        let mut first = vec![vec![123.0]; usize::from(tokens)];
                        let mut second = first.clone();
                        let mut third = first.clone();
                        let result = if members == 2 {
                            executor.project_f32_pair_tokens(
                                [&tensors[0], &tensors[1]],
                                &activations,
                                true,
                                &mut first,
                                &mut second,
                            )
                        } else {
                            executor.project_f32_triple_tokens(
                                [&tensors[0], &tensors[1], &tensors[2]],
                                &activations,
                                true,
                                &mut first,
                                &mut second,
                                &mut third,
                            )
                        };
                        assert!(result.is_err());
                        assert!(executor.is_faulted());
                        for outputs in [&first, &second, &third].into_iter().take(members) {
                            assert!(outputs.iter().all(Vec::is_empty));
                        }
                        assert_eq!(executor.submitter.batches.len(), fail_at);
                        assert_eq!(executor.queue_writes_attempted(), u64::try_from(fail_at)?);
                        assert_eq!(
                            executor.queue_writes_completed(),
                            u64::try_from(fail_at - 1)?
                        );
                        let completed: usize = expected_batches[..fail_at - 1].iter().sum();
                        assert_eq!(
                            executor.statistics().operations_completed,
                            u64::try_from(completed)?
                        );
                        assert_eq!(
                            executor.statistics().operations_attempted,
                            u64::try_from(completed + expected_batches[fail_at - 1])?
                        );
                        assert!(
                            executor
                                .project_f32_pair_tokens(
                                    [&tensors[0], &tensors[1]],
                                    &activations,
                                    true,
                                    &mut first,
                                    &mut second,
                                )
                                .is_err()
                        );
                        assert_eq!(executor.submitter.batches.len(), fail_at);
                    }
                }
            }
        }
        Ok(())
    }

    struct TransientCapture {
        mock: MockNativeSubmitter,
        scale_headers: usize,
        maximum_packed_bytes: usize,
    }

    struct TransientTokenCapture {
        columns: usize,
        rows: usize,
        expected_weights: Vec<u8>,
        activations: Vec<Vec<f32>>,
        fail_call: Option<usize>,
        nonfinite_call: Option<usize>,
    }

    #[expect(
        clippy::suboptimal_flops,
        reason = "The independent scalar test decoder retains explicit multiply then add order."
    )]
    impl NativeQ2Submitter for TransientTokenCapture {
        fn residency_capacity_bytes(&self) -> usize {
            // Admit cache metadata, but refuse even the smallest test tensor.
            1
        }

        fn submit(
            &mut self,
            columns: usize,
            rows: usize,
            weights: &[u8],
            activation: &[f32],
            output: &mut [f32],
        ) -> std::result::Result<(), hexagon_native_rpc::Error> {
            assert_eq!((columns, rows), (self.columns, self.rows));
            assert_eq!(weights, self.expected_weights);
            self.activations.push(activation.to_vec());
            if self.fail_call == Some(self.activations.len()) {
                output.fill(123.0);
                return Err(hexagon_native_rpc::Error::Contract);
            }
            // Decode the native digit lanes directly. This does not call the
            // production permutation or any CPU/DSP projection implementation.
            for (row, destination) in output.iter_mut().enumerate() {
                let mut sum = 0.0_f32;
                for block in 0..columns / 128 {
                    let pair = (row / 32 * (columns / 128) + block) * 1152;
                    let scale_at = pair + 512 + (row % 32) * 2;
                    let bits = u16::from_le_bytes([weights[scale_at], weights[scale_at + 1]]);
                    let factor = if bits == 0x3800 { 0.5 } else { 1.0 };
                    let mut dot = 0.0_f32;
                    for index in 0..128 {
                        let group = (index % 32) / 4;
                        let lane = index % 4;
                        let at = pair
                            + (group / 4) * 576
                            + (group % 4) * 128
                            + (row % 32) * 4
                            + index / 32;
                        let code = (weights[at] >> (lane * 2)) & 3;
                        dot += f32::from(i16::from(code) - 1) * activation[block * 128 + index];
                    }
                    sum += factor * dot;
                }
                *destination = sum;
            }
            if self.nonfinite_call == Some(self.activations.len()) {
                output[rows - 1] = f32::INFINITY;
            }
            Ok(())
        }
    }

    fn transient_token_tensor(rows: usize) -> Tensor {
        let mut tensor = lazy_tensor("blk.0.ffn_up.weight", 256, rows);
        let mut data = tensor.data.to_vec();
        for (index, block) in data.as_chunks_mut::<34>().0.iter_mut().enumerate() {
            let header = if index.is_multiple_of(3) {
                0x3800_u16
            } else {
                0x3c00_u16
            };
            block[..2].copy_from_slice(&header.to_le_bytes());
            for (lane, byte) in block[2..].iter_mut().enumerate() {
                *byte = u8::try_from((index * 13 + lane * 7) % 256).unwrap_or(0);
            }
        }
        tensor.data = Arc::<[u8]>::from(data).into();
        tensor
    }

    fn transient_token_executor(
        tensor: &Tensor,
        refused_residency: bool,
    ) -> Result<NativeQ2Executor<TransientTokenCapture>> {
        let (columns, rows) = native_dimensions(tensor)?;
        let mut expected_weights = vec![0_u8; rows * 36 * (columns / 128)];
        // Build the native layout one two-bit digit at a time, independently
        // of the production word-swap permutation and row-group packer.
        for row in 0..rows {
            for block in 0..columns / 128 {
                let original = &tensor.data[(row * (columns / 128) + block) * 34..][..34];
                let pair = (row / 32 * (columns / 128) + block) * 1152;
                for half in 0..2 {
                    let at = pair + half * 576 + 512 + (row % 32) * 2;
                    expected_weights[at..at + 2].copy_from_slice(&original[..2]);
                }
                for group in 0..8 {
                    for digit in 0..4 {
                        let at =
                            pair + (group / 4) * 576 + (group % 4) * 128 + (row % 32) * 4 + digit;
                        for lane in 0..4 {
                            let code = (original[2 + group * 4 + lane] >> (digit * 2)) & 3;
                            expected_weights[at] |= code << (lane * 2);
                        }
                    }
                }
            }
        }
        let limits = NativeQ2ProjectionLimits::new(columns, rows, 100, Duration::from_secs(10))
            .map_err(|_| invalid("limits"))?;
        let metadata = if refused_residency { 4096 } else { 0 };
        let payload = rows * 4 + expected_weights.len() + rows * 4 + metadata;
        let submitter = TransientTokenCapture {
            columns,
            rows,
            expected_weights,
            activations: Vec::new(),
            fail_call: None,
            nonfinite_call: None,
        };
        let executor = NativeQ2Executor::new(submitter, limits, payload);
        if refused_residency {
            executor.with_resident_weights(metadata)
        } else {
            Ok(executor)
        }
    }

    #[test]
    #[expect(
        clippy::suboptimal_flops,
        reason = "The independent original-PQ2 test oracle retains explicit multiply then add order."
    )]
    fn native_q2_transient_token_tile_matches_independent_layout_and_scalar_outputs() -> Result<()>
    {
        for rows in [32, 64] {
            let tensor = transient_token_tensor(rows);
            for refused_residency in [false, true] {
                for tokens in [0, 1, 8, 16, 32] {
                    let mut executor = transient_token_executor(&tensor, refused_residency)?;
                    let activations: Vec<Vec<f32>> = (0..tokens)
                        .map(|token| {
                            (0..256)
                                .map(|index| {
                                    f32::from(
                                        i16::try_from((token * 3 + index) % 17).unwrap_or(0) - 8,
                                    )
                                })
                                .collect()
                        })
                        .collect();
                    let mut outputs = vec![vec![91.0]; tokens];
                    executor.project_f32_tokens(&tensor, &activations, true, &mut outputs)?;
                    for (activation, output) in activations.iter().zip(&outputs) {
                        for (row, value) in output.iter().enumerate() {
                            let mut expected = 0.0_f32;
                            for block in 0..2 {
                                let original = &tensor.data[(row * 2 + block) * 34..][..34];
                                let mut dot = 0.0_f32;
                                for index in 0..128 {
                                    let code = (original[2 + index % 32] >> (2 * (index / 32))) & 3;
                                    dot += f32::from(i16::from(code) - 1)
                                        * activation[block * 128 + index];
                                }
                                expected += scale(original)? * dot;
                            }
                            assert_eq!(value.to_bits(), expected.to_bits());
                        }
                    }
                    assert_eq!(executor.submitter.activations, activations);
                    let count = u64::try_from(tokens)?;
                    assert_eq!(executor.statistics().operations_attempted, count);
                    assert_eq!(executor.statistics().operations_completed, count);
                    assert_eq!(executor.statistics().projections_completed, count);
                    assert_eq!(executor.queue_writes_attempted(), count);
                    assert_eq!(executor.queue_writes_completed(), count);
                    assert_eq!(executor.residency_statistics().tensors_prepared, 0);
                    assert_eq!(executor.scratch_weights.capacity(), 0);
                    assert!(!executor.is_faulted());
                    executor.maximum_owned_payload_bytes -= 1;
                    assert!(!executor.supports_f32(&tensor));
                }
            }
        }
        Ok(())
    }

    #[test]
    fn native_q2_transient_token_tile_preserves_failure_order_and_terminal_publication()
    -> Result<()> {
        for refused_residency in [false, true] {
            for failure in 0..7 {
                let mut tensor = transient_token_tensor(32);
                let mut executor = transient_token_executor(&tensor, refused_residency)?;
                let mut activations = vec![vec![1.0; 256]; 3];
                let mut outputs = vec![vec![91.0]; 3];
                match failure {
                    0 => activations[0][0] = f32::NAN,
                    1 => activations[2][0] = f32::INFINITY,
                    2 => activations[2].truncate(255),
                    3 => executor.submitter.fail_call = Some(2),
                    4 => executor.submitter.nonfinite_call = Some(2),
                    5 | 6 => {
                        let mut data = tensor.data.to_vec();
                        data[34..36].copy_from_slice(&0x7e00_u16.to_le_bytes());
                        tensor.data = Arc::<[u8]>::from(data).into();
                        if failure == 6 {
                            activations[0][0] = f32::NAN;
                        }
                    }
                    _ => return Err(invalid("unsupported test failure selection")),
                }
                let result = executor.project_f32_tokens(&tensor, &activations, true, &mut outputs);
                assert!(result.is_err());
                assert!(outputs.iter().all(Vec::is_empty));
                assert!(executor.is_faulted());
                let expected_calls = match failure {
                    1 | 2 if !refused_residency => 2,
                    3 | 4 => 2,
                    _ => 0,
                };
                assert_eq!(executor.submitter.activations.len(), expected_calls);
                if failure == 6 {
                    let error = result
                        .err()
                        .ok_or_else(|| invalid("missing expected error"))?;
                    assert!(
                        error
                            .to_string()
                            .contains("nonfinite Hexagon projection value")
                    );
                }
                let attempted = u64::try_from(expected_calls)?;
                let completed = if matches!(failure, 3 | 4) {
                    1
                } else {
                    attempted
                };
                assert_eq!(executor.statistics().operations_attempted, attempted);
                assert_eq!(executor.statistics().operations_completed, completed);
                assert_eq!(executor.queue_writes_attempted(), attempted);
                assert_eq!(executor.queue_writes_completed(), completed);
                assert!(
                    executor
                        .project_f32_tokens(&tensor, &activations, true, &mut outputs)
                        .is_err()
                );
                assert_eq!(executor.submitter.activations.len(), expected_calls);
                assert_eq!(executor.residency_statistics().tensors_prepared, 0);
            }
        }
        Ok(())
    }
    impl NativeQ2Submitter for TransientCapture {
        fn submit(
            &mut self,
            columns: usize,
            rows: usize,
            weights: &[u8],
            activation: &[f32],
            output: &mut [f32],
        ) -> std::result::Result<(), hexagon_native_rpc::Error> {
            for tile in weights.as_chunks::<576>().0 {
                assert!(tile[..512].iter().all(|&code| code == 0x55));
                for scale in tile[512..].as_chunks::<2>().0 {
                    let bits = u16::from_le_bytes(*scale);
                    assert!(matches!(bits, 0 | 0x3800));
                    self.scale_headers += usize::from(bits == 0x3800);
                }
            }
            self.maximum_packed_bytes = self.maximum_packed_bytes.max(weights.len());
            self.mock.submit(columns, rows, weights, activation, output)
        }
    }
    #[test]
    fn native_q2_nonresident_single_and_tiled_fallback_preserve_packing_and_bounds() -> Result<()> {
        for (rows, maximum_columns, expected_calls) in [(32, 256, 1), (49, 128, 4)] {
            let columns = 256;
            let mut tensor = lazy_tensor("blk.0.ffn_up.weight", columns, rows);
            let mut data = tensor.data.to_vec();
            for block in data.as_chunks_mut::<34>().0 {
                block[..2].copy_from_slice(&0x3800_u16.to_le_bytes());
            }
            tensor.data = Arc::<[u8]>::from(data).into();
            let limits =
                NativeQ2ProjectionLimits::new(maximum_columns, 32, 100, Duration::from_secs(10))
                    .map_err(|_| invalid("limits"))?;
            let packed_bytes = 32 * 36 * (maximum_columns / 128);
            let payload_bytes = rows * 4 + packed_bytes + 32 * 4;
            let submitter = TransientCapture {
                mock: MockNativeSubmitter {
                    batch_ffn: true,
                    ffn_pair_calls: 0,
                    calls: 0,
                    resident_calls: 0,
                    remaining: 100,
                    residency_bytes: 0,
                },
                scale_headers: 0,
                maximum_packed_bytes: 0,
            };
            let mut executor = NativeQ2Executor::new(submitter, limits, payload_bytes);
            assert!(executor.supports_f32(&tensor));
            assert!(executor.residency.is_none());
            let mut output = vec![9.0];
            executor.project_f32(&tensor, &[1.0; 256], true, &mut output)?;
            assert_eq!(output.len(), rows);
            for (index, value) in output.iter().enumerate() {
                let expected = if index.is_multiple_of(32) {
                    256.0_f32
                } else {
                    0.0_f32
                };
                assert_eq!(value.to_bits(), expected.to_bits());
            }
            assert_eq!(executor.submitter.mock.calls, expected_calls);
            assert_eq!(executor.submitter.mock.resident_calls, 0);
            assert_eq!(executor.statistics().operations_completed, expected_calls);
            assert_eq!(executor.queue_writes_completed(), expected_calls);
            assert_eq!(executor.submitter.scale_headers, rows * (columns / 128) * 2);
            assert_eq!(executor.submitter.maximum_packed_bytes, packed_bytes);
            assert!(!executor.is_faulted());
            executor.maximum_owned_payload_bytes = payload_bytes - 1;
            assert!(!executor.supports_f32(&tensor));
            assert!(
                executor
                    .project_f32(&tensor, &[1.0; 256], true, &mut output)
                    .is_err()
            );
            assert_eq!(output, Vec::<f32>::new());
            assert_eq!(executor.submitter.mock.calls, expected_calls);
        }
        Ok(())
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
            data: Arc::<[u8]>::from(data).into(),
        };
        let limits = NativeQ2ProjectionLimits::new(5120, 17_408, 100, Duration::from_secs(10))
            .map_err(|_| invalid("limits"))?
            .with_resident_weights(2304)
            .map_err(|_| invalid("residency"))?;
        let mut submitter = MockNativeSubmitter {
            batch_ffn: true,
            ffn_pair_calls: 0,
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
                kind: PQ2_KIND,
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
        assert_eq!(DEFAULT_NATIVE_Q2_PROJECTIONS.len(), 168);
        let mut total_bytes = 0_usize;
        let mut arena_bytes = [0_usize; 6];
        let mut registry = hexagon_native_rpc::NativeQ2WeightRegistry::new(
            DEFAULT_NATIVE_Q2_RESIDENT_WEIGHT_BYTES,
        )
        .map_err(|_| invalid("registry"))?;
        let scratch = vec![0_u8; 17_408 * 36 * (5120 / 128)];
        for (idx, name) in DEFAULT_NATIVE_Q2_PROJECTIONS.into_iter().enumerate() {
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
            let handle = registry
                .prepare(columns, rows, &scratch[..packed_bytes])
                .map_err(|_| invalid("prepare"))?
                .ok_or_else(|| invalid("arena exhausted before 168 projections"))?;
            let expected_arena = u16::try_from(idx / 28).unwrap_or(0);
            assert_eq!(handle.arena_index(), expected_arena);
            arena_bytes[idx / 28] += packed_bytes;
            total_bytes += packed_bytes;
        }
        assert_eq!(registry.remaining_bytes(), 0);
        assert_eq!(
            arena_bytes,
            [
                hexagon_native_rpc::NATIVE_Q2_ARENA_WEIGHT_STEP_BYTES,
                hexagon_native_rpc::NATIVE_Q2_ARENA_ODD_WEIGHT_STEP_BYTES,
                hexagon_native_rpc::NATIVE_Q2_ARENA_WEIGHT_STEP_BYTES,
                hexagon_native_rpc::NATIVE_Q2_ARENA_ODD_WEIGHT_STEP_BYTES,
                hexagon_native_rpc::NATIVE_Q2_ARENA_WEIGHT_STEP_BYTES,
                hexagon_native_rpc::NATIVE_Q2_ARENA_ODD_WEIGHT_STEP_BYTES,
            ]
        );
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
    #[expect(
        clippy::too_many_lines,
        clippy::similar_names,
        reason = "Keep single-token and multi-token 2/3-sibling batching assertions together."
    )]
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
                data: Arc::<[u8]>::from(data).into(),
            }
        };
        let qkv = make_tensor("blk.0.attn_qkv.weight", 64);
        let gate = make_tensor("blk.0.attn_gate.weight", 32);
        let q = make_tensor("blk.3.attn_q.weight", 64);
        let k = make_tensor("blk.3.attn_k.weight", 32);
        let v = make_tensor("blk.3.attn_v.weight", 32);

        let limits = NativeQ2ProjectionLimits::new(5120, 17_408, 100, Duration::from_secs(10))
            .map_err(|_| invalid("limits"))?
            .with_resident_weights(131_072)
            .map_err(|_| invalid("residency"))?;
        let mut submitter = MockNativeSubmitter {
            batch_ffn: true,
            ffn_pair_calls: 0,
            calls: 0,
            resident_calls: 0,
            remaining: 100,
            residency_bytes: 131_072,
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

        // Multi-token pair batching uses 8-op submissions for every four tokens.
        let acts = vec![
            vec![1.0_f32; columns],
            vec![2.0_f32; columns],
            vec![3.0_f32; columns],
            vec![4.0_f32; columns],
            vec![5.0_f32; columns],
        ];
        let mut outs_a = vec![Vec::new(); 5];
        let mut outs_b = vec![Vec::new(); 5];
        executor.project_f32_pair_tokens([&qkv, &gate], &acts, true, &mut outs_a, &mut outs_b)?;
        // 5 tokens * 2 projections = 10 ops across 2 queue writes (8 + 2).
        assert_eq!(executor.queue_writes_completed(), 4);
        assert_eq!(executor.statistics().operations_completed, 15);
        assert_eq!(outs_a[0][0], 256.0);
        assert_eq!(outs_b[0][0], 257.0);
        assert_eq!(outs_a[1][0], 514.0);
        assert_eq!(outs_b[1][0], 515.0);
        assert_eq!(outs_a[4][0], 1280.0);
        assert_eq!(outs_b[4][0], 1281.0);

        executor.project_f32_tokens(&q, &acts, true, &mut outs_a)?;
        // 5 tokens * 1 projection = 5 ops across 2 queue writes (4 + 1).
        assert_eq!(executor.queue_writes_completed(), 6);
        assert_eq!(executor.statistics().operations_completed, 20);
        assert_eq!(outs_a[0][0], 256.0);
        assert_eq!(outs_a[3][0], 1027.0);
        assert_eq!(outs_a[4][0], 1280.0);

        let mut outs_c = vec![Vec::new(); 5];
        executor.project_f32_triple_tokens(
            [&q, &k, &v],
            &acts,
            true,
            &mut outs_a,
            &mut outs_b,
            &mut outs_c,
        )?;
        // 5 tokens * 3 projections = 15 ops across 3 queue writes (8 + 4 + 3).
        assert_eq!(executor.queue_writes_completed(), 9);
        assert_eq!(executor.statistics().operations_completed, 35);
        assert_eq!(outs_a[0][0], 256.0);
        assert_eq!(outs_b[0][0], 257.0);
        assert_eq!(outs_c[0][0], 258.0);
        assert_eq!(outs_a[1][0], 515.0);
        assert_eq!(outs_b[1][0], 516.0);
        assert_eq!(outs_c[1][0], 517.0);
        assert_eq!(outs_a[2][0], 774.0);
        assert_eq!(outs_b[2][0], 775.0);
        assert_eq!(outs_c[2][0], 768.0);
        assert_eq!(outs_a[3][0], 1025.0);
        assert_eq!(outs_b[3][0], 1026.0);
        assert_eq!(outs_c[3][0], 1027.0);
        assert_eq!(outs_a[4][0], 1280.0);
        assert_eq!(outs_b[4][0], 1281.0);
        assert_eq!(outs_c[4][0], 1282.0);

        // Verify 8-token batching via submit_resident_multi8 (1 queue write for 8 single-proj tokens,
        // 2 queue writes for 8 pair-proj tokens, 3 queue writes for 8 triple-proj tokens) and fused FFN SwiGLU.
        let acts8: Vec<Vec<f32>> = (1..=8_u8).map(|i| vec![f32::from(i); columns]).collect();
        let mut outs8_a = vec![Vec::new(); 8];
        let mut outs8_b = vec![Vec::new(); 8];
        let mut outs8_c = vec![Vec::new(); 8];
        executor.project_f32_tokens(&q, &acts8, true, &mut outs8_a)?;
        assert_eq!(executor.queue_writes_completed(), 10);
        assert_eq!(executor.statistics().operations_completed, 43);
        assert_eq!(outs8_a[0][0], 256.0);
        assert_eq!(outs8_a[7][0], 2048.0 + 7.0);

        executor.project_f32_pair_tokens(
            [&qkv, &gate],
            &acts8,
            true,
            &mut outs8_a,
            &mut outs8_b,
        )?;
        // 8 tokens * 2 projections = 16 ops in 2 queue writes (8 + 8).
        assert_eq!(executor.queue_writes_completed(), 12);
        assert_eq!(executor.statistics().operations_completed, 59);

        executor.project_f32_triple_tokens(
            [&q, &k, &v],
            &acts8,
            true,
            &mut outs8_a,
            &mut outs8_b,
            &mut outs8_c,
        )?;
        // 8 tokens * 3 projections = 24 ops in 3 queue writes (8 + 8 + 8).
        assert_eq!(executor.queue_writes_completed(), 15);
        assert_eq!(executor.statistics().operations_completed, 83);

        let ffn_gate = make_tensor("blk.0.ffn_gate.weight", 64);
        let ffn_up = make_tensor("blk.0.ffn_up.weight", 64);
        let mut ffn_executor = NativeQ2Executor::new(&mut submitter, limits, 64 * 1024 * 1024)
            .with_resident_weights(16_384)?;
        let mut swiglu_out = Vec::new();
        ffn_executor.project_f32_ffn_swiglu(&ffn_gate, &ffn_up, &activation, &mut swiglu_out)?;
        assert_eq!(swiglu_out.len(), 64);
        assert_eq!(swiglu_out[0], 512.0);
        assert_eq!(ffn_executor.queue_writes_completed(), 1);
        assert_eq!(ffn_executor.statistics().operations_completed, 3);
        assert_eq!(ffn_executor.statistics().projections_completed, 2);

        let mut swiglu_outs = vec![Vec::new(); 5];
        assert!(Pq2ProjectionExecutor::supports_f32_ffn_swiglu(
            &ffn_executor,
            &ffn_gate,
            &ffn_up
        ));
        Pq2ProjectionExecutor::project_f32_ffn_swiglu_tokens(
            &mut ffn_executor,
            &ffn_gate,
            &ffn_up,
            &acts,
            true,
            &mut swiglu_outs,
        )?;
        // 5 tokens = two 2-token 6-op submissions + one 1-token 3-op submission = 15 ops across 3 queue writes.
        assert_eq!(ffn_executor.queue_writes_completed(), 4);
        assert_eq!(ffn_executor.statistics().operations_completed, 18);
        assert_eq!(ffn_executor.statistics().projections_completed, 12);
        assert_eq!(swiglu_outs[0][0], 512.0);
        assert_eq!(swiglu_outs[1][0], 1025.0);
        assert_eq!(swiglu_outs[4][0], 2560.0);

        let ffn_gate_128 = make_tensor("blk.0.ffn_gate_128.weight", columns);
        let ffn_up_128 = make_tensor("blk.0.ffn_up_128.weight", columns);
        let ffn_down_128 = make_tensor("blk.0.ffn_down_128.weight", columns);
        let residual = vec![10.0_f32; columns];
        let mut chain_out = Vec::new();
        assert!(Pq2ProjectionExecutor::supports_f32_ffn_chain(
            &ffn_executor,
            [&ffn_gate_128, &ffn_up_128, &ffn_down_128]
        ));
        Pq2ProjectionExecutor::project_f32_ffn_chain(
            &mut ffn_executor,
            [&ffn_gate_128, &ffn_up_128, &ffn_down_128],
            &activation,
            Some(&residual),
            Some(64),
            true,
            &mut chain_out,
        )?;
        assert_eq!(chain_out.len(), columns);
        assert_eq!(chain_out[0], 778.0);
        assert_eq!(ffn_executor.queue_writes_completed(), 5);
        assert_eq!(ffn_executor.statistics().operations_completed, 24);
        assert_eq!(ffn_executor.statistics().projections_completed, 15);

        let lora_a = vec![0.25_f32; columns];
        assert!(Pq2ProjectionExecutor::supports_f32_ffn_chain_lora(
            &ffn_executor,
            [&ffn_gate_128, &ffn_up_128, &ffn_down_128]
        ));
        let mut chain_lora_out = Vec::new();
        let lora_dot = Pq2ProjectionExecutor::project_f32_ffn_chain_lora(
            &mut ffn_executor,
            [&ffn_gate_128, &ffn_up_128, &ffn_down_128],
            &activation,
            Some(&residual),
            Some(64),
            &lora_a,
            true,
            &mut chain_lora_out,
        )?;
        assert_eq!(lora_dot, 64.0);
        assert_eq!(chain_lora_out.len(), columns);
        assert_eq!(chain_lora_out[0], 778.0);
        assert_eq!(ffn_executor.queue_writes_completed(), 6);
        assert_eq!(ffn_executor.statistics().operations_completed, 31);
        assert_eq!(ffn_executor.statistics().projections_completed, 18);

        let adapter_a = vec![0.5_f32; columns];
        let adapter_b = vec![2.0_f32; 64];
        let mut adapted_out = Vec::new();
        ffn_executor.project_f32_with_rank1_adapter(
            &ffn_gate,
            &activation,
            &adapter_a,
            &adapter_b,
            &mut adapted_out,
        )?;
        assert_eq!(adapted_out.len(), 64);
        // base = 256.0, dot = 128.0, adapter_b[0] = 2.0 -> 256.0 + 256.0 = 512.0.
        assert_eq!(adapted_out[0], 512.0);
        assert_eq!(adapted_out[1], 256.0);

        let schedule = crate::plan_hybrid_prefill_pipeline(64, 27, 20, 8)?;
        assert_eq!(schedule.npu_layer_range, 0..27);
        assert_eq!(schedule.cpu_layer_range, 27..64);
        assert_eq!(schedule.microbatch_count, 3);
        assert_eq!(schedule.steps.len(), 4);
        assert_eq!(schedule.steps[0].npu_microbatch, Some(0));
        assert_eq!(schedule.steps[0].cpu_microbatch, None);
        assert_eq!(schedule.steps[1].npu_microbatch, Some(1));
        assert_eq!(schedule.steps[1].cpu_microbatch, Some(0));
        assert_eq!(schedule.steps[3].npu_microbatch, None);
        assert_eq!(schedule.steps[3].cpu_microbatch, Some(2));
        assert_eq!(schedule.steps[3].cpu_token_range, 16..20);

        let het = crate::plan_three_processor_placement(
            64,
            27,
            37,
            4,
            crate::AcceleratorTarget::AdrenoGpuOpenCl,
        )?;
        assert_eq!(het.npu_layers, 27);
        assert_eq!(het.gpu_layers, 37);
        assert_eq!(het.cpu_layers, 0);
        assert_eq!(het.npu_q2_projections, 168);
        assert_eq!(het.gpu_q2_projections, 232);
        assert_eq!(
            het.layers[0].projection_target,
            crate::AcceleratorTarget::HexagonNpu
        );
        assert_eq!(
            het.layers[27].projection_target,
            crate::AcceleratorTarget::AdrenoGpuOpenCl
        );
        assert_eq!(
            het.layers[27].adreno_fwht_kernel,
            Some(crate::AdrenoQ2KernelKind::FwhtLocalMem1024)
        );
        Ok(())
    }
    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "Cover capability-disabled and paired tile tails with every optional chain shape and terminal input refusal"
    )]
    fn native_q2_full_ffn_tiles_batch_pairs_and_preserve_tail_accounting() -> Result<()> {
        let tensor = |name: &str| Tensor {
            name: name.to_owned(),
            dimensions: vec![128, 128],
            kind: 142,
            data: Arc::<[u8]>::from(vec![0_u8; 128 * 34]).into(),
        };
        let gate = tensor("blk.0.ffn_gate.weight");
        let up = tensor("blk.0.ffn_up.weight");
        let down = tensor("blk.0.ffn_down.weight");
        for batching in [false, true] {
            for tokens in [0_u8, 1, 2, 3, 8, 16, 32] {
                for (hadamard, add) in [
                    (None, false),
                    (Some(64), false),
                    (None, true),
                    (Some(64), true),
                ] {
                    let limits =
                        NativeQ2ProjectionLimits::new(128, 128, 256, Duration::from_secs(4))?
                            .with_resident_weights(65_536)?;
                    let mut submitter = MockNativeSubmitter {
                        batch_ffn: batching,
                        ffn_pair_calls: 0,
                        calls: 0,
                        resident_calls: 0,
                        remaining: 256,
                        residency_bytes: 65_536,
                    };
                    let mut executor = NativeQ2Executor::new(&mut submitter, limits, 1_048_576)
                        .with_resident_weights(16_384)?;
                    let activations: Vec<_> = (0..tokens)
                        .map(|token| vec![f32::from(token); 128])
                        .collect();
                    let residuals = vec![vec![2.0_f32; 128]; usize::from(tokens)];
                    let mut outputs = vec![vec![99.0_f32]; usize::from(tokens)];
                    Pq2ProjectionExecutor::project_f32_ffn_chain_tokens(
                        &mut executor,
                        [&gate, &up, &down],
                        &activations,
                        add.then_some(residuals.as_slice()),
                        hadamard,
                        true,
                        &mut outputs,
                    )?;
                    assert_eq!(
                        executor.queue_writes_completed(),
                        u64::from(if batching { tokens.div_ceil(2) } else { tokens })
                    );
                    assert_eq!(
                        executor.statistics().operations_completed,
                        u64::from(tokens) * (4 + u64::from(hadamard.is_some()) + u64::from(add))
                    );
                    assert_eq!(
                        executor.statistics().projections_completed,
                        u64::from(tokens) * 3
                    );
                    for (token, output) in (0..tokens).zip(&outputs) {
                        assert_eq!(output.len(), 128);
                        assert_eq!(
                            output[0],
                            f32::from(token).mul_add(384.0, if add { 2.0 } else { 0.0 })
                        );
                    }
                    assert_eq!(
                        executor.submitter.ffn_pair_calls,
                        u64::from(if batching { tokens / 2 } else { 0 })
                    );
                    if tokens > 0 {
                        let mut invalid_activations = activations;
                        invalid_activations[usize::from(tokens) - 1][127] = f32::NAN;
                        let writes = executor.queue_writes_attempted();
                        assert!(
                            Pq2ProjectionExecutor::project_f32_ffn_chain_tokens(
                                &mut executor,
                                [&gate, &up, &down],
                                &invalid_activations,
                                None,
                                hadamard,
                                true,
                                &mut outputs
                            )
                            .is_err()
                        );
                        assert!(outputs.iter().all(Vec::is_empty));
                        assert_eq!(executor.queue_writes_attempted(), writes);
                        assert!(
                            Pq2ProjectionExecutor::project_f32_ffn_chain_tokens(
                                &mut executor,
                                [&gate, &up, &down],
                                &[],
                                None,
                                None,
                                true,
                                &mut []
                            )
                            .is_err()
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn select_prefix_and_prewarm_tensors_admits_fitting_prefix_and_falls_back_on_tail() -> Result<()>
    {
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
                data: Arc::<[u8]>::from(data).into(),
            }
        };
        let first = make_tensor("blk.0.ffn_gate.weight", 32);
        let second = make_tensor("blk.0.ffn_up.weight", 32);
        let third = make_tensor("blk.0.ffn_down.weight", 32);
        // Native packed [256, 32] Q2_0 extent is 2304 bytes; 5000 bytes admits 2 tensors and skips the 3rd.
        let limits = NativeQ2ProjectionLimits::new(5120, 17_408, 64, Duration::from_secs(4))
            .map_err(|_| invalid("limits"))?
            .with_resident_weights(5000)
            .map_err(|_| invalid("residency"))?;
        let mut submitter = MockNativeSubmitter {
            batch_ffn: false,
            ffn_pair_calls: 0,
            calls: 0,
            resident_calls: 0,
            remaining: 64,
            residency_bytes: 5000,
        };
        let mut executor = NativeQ2Executor::new(&mut submitter, limits, 64 * 1024 * 1024)
            .with_resident_weights(16_384)?;
        let prepared =
            executor.select_prefix_and_prewarm_tensors(&[first.clone(), second, third.clone()])?;
        assert_eq!(prepared, 2);
        assert_eq!(executor.residency_statistics().tensors_prepared, 2);

        assert!(Pq2ProjectionExecutor::supports_f32(&executor, &first));
        assert!(!Pq2ProjectionExecutor::supports_f32(&executor, &third));
        let activation = vec![1.0_f32; columns];
        let mut output = Vec::new();
        Pq2ProjectionExecutor::project_f32(&mut executor, &first, &activation, true, &mut output)?;
        assert_eq!(output.len(), 32);
        assert!(
            Pq2ProjectionExecutor::project_f32(
                &mut executor,
                &third,
                &activation,
                true,
                &mut output
            )
            .is_err()
        );
        Ok(())
    }

    struct TempArenaCacheGuard {
        path: PathBuf,
    }

    impl Drop for TempArenaCacheGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    fn temp_arena_cache_path(tag: &str) -> TempArenaCacheGuard {
        static NEXT_TEST_CACHE: AtomicU64 = AtomicU64::new(0);
        let id = NEXT_TEST_CACHE.fetch_add(1, Ordering::Relaxed);
        TempArenaCacheGuard {
            path: std::env::temp_dir().join(format!(
                "SwiftBonsai-{tag}-{}-{id}.hq2c",
                std::process::id()
            )),
        }
    }

    struct ArchArenaSubmitter {
        inner: ArenaSubmitter,
        arch: u32,
    }

    impl NativeQ2Submitter for ArchArenaSubmitter {
        fn htp_architecture(&self) -> u32 {
            self.arch
        }

        fn residency_capacity_bytes(&self) -> usize {
            self.inner.residency_capacity_bytes()
        }

        fn supports_in_place_weight_preparation(&self) -> bool {
            self.inner.supports_in_place_weight_preparation()
        }

        fn prepare_weights(
            &mut self,
            columns: usize,
            rows: usize,
            weights: &[u8],
        ) -> std::result::Result<Option<NativeQ2WeightHandle>, hexagon_native_rpc::Error> {
            self.inner.prepare_weights(columns, rows, weights)
        }

        fn prepare_weights_in_place(
            &mut self,
            columns: usize,
            rows: usize,
            initialize: &mut dyn FnMut(
                &mut [u8],
            )
                -> std::result::Result<(), hexagon_native_rpc::Error>,
        ) -> std::result::Result<Option<NativeQ2WeightHandle>, hexagon_native_rpc::Error> {
            self.inner
                .prepare_weights_in_place(columns, rows, initialize)
        }

        fn submit(
            &mut self,
            columns: usize,
            rows: usize,
            weights: &[u8],
            activation: &[f32],
            output: &mut [f32],
        ) -> std::result::Result<(), hexagon_native_rpc::Error> {
            self.inner
                .submit(columns, rows, weights, activation, output)
        }
    }

    fn cache_fixture_tensors() -> Vec<Tensor> {
        let mut t0 = lazy_tensor("blk.0.ffn_gate.weight", 256, 32);
        let mut t1 = lazy_tensor("blk.0.ffn_up.weight", 256, 49);
        let mut t2 = lazy_tensor("blk.0.ffn_down.weight", 256, 32);
        let stamp = |tensor: &mut Tensor, seed: u8| {
            let mut buf = tensor.data.to_vec();
            for (idx, block) in buf.as_chunks_mut::<34>().0.iter_mut().enumerate() {
                block[..2].copy_from_slice(&0x3800_u16.to_le_bytes());
                block[2] = u8::try_from((idx + usize::from(seed)) % 256).unwrap_or(0);
            }
            tensor.data = Arc::<[u8]>::from(buf).into();
        };
        stamp(&mut t0, 1);
        stamp(&mut t1, 7);
        stamp(&mut t2, 19);
        vec![t0, t1, t2]
    }

    #[test]
    fn native_q2_arena_cache_round_trip_prewarms_copied_and_in_place_and_projects() -> Result<()> {
        let tensors = cache_fixture_tensors();
        let guard = temp_arena_cache_path("cache-roundtrip");
        let model_sha = [0x42_u8; 32];
        let entries =
            NativeQ2ArenaCache::save_verified_cache(&guard.path, 81, model_sha, &tensors, None)?;
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].offset, 0);
        assert_eq!(entries[0].byte_len, 2304);
        assert_eq!(entries[1].rows, 49);
        assert_eq!(entries[1].offset, 2304);
        assert_eq!(entries[1].byte_len, 4608);
        assert_eq!(entries[2].offset, 6912);
        assert_eq!(entries[2].byte_len, 2304);
        let saved_bytes = std::fs::read(&guard.path)?;
        assert_eq!(&saved_bytes[..8], b"SB2HQ2C1");

        let limits = NativeQ2ProjectionLimits::new(256, 64, 16, Duration::from_secs(4))
            .map_err(|_| invalid("limits"))?
            .with_resident_weights(16_384)
            .map_err(|_| invalid("residency"))?;

        let mut reference = NativeQ2Executor::new(
            arena_submitter_at_capacity(true, 16_384)?,
            limits,
            1024 * 1024,
        )
        .with_resident_weights(4096)?;
        reference.select_and_prewarm_tensors(&tensors)?;

        // 1. In-place load matches reference arena bytes exactly without temporary scratch copies.
        let mut direct_cached = NativeQ2Executor::new(
            arena_submitter_at_capacity(true, 16_384)?,
            limits,
            1024 * 1024,
        )
        .with_resident_weights(4096)?;
        let loaded_direct = NativeQ2ArenaCache::load_and_prewarm_from_verified_cache(
            &mut direct_cached,
            &guard.path,
            model_sha,
            &tensors,
        )?;
        assert_eq!(loaded_direct, entries);
        assert_eq!(direct_cached.submitter.initializations, 3);
        assert_eq!(direct_cached.submitter.copies, 0);
        assert_eq!(direct_cached.residency_statistics().tensors_prepared, 3);
        assert_eq!(
            direct_cached
                .residency_statistics()
                .weight_payload_bytes_prepared,
            9216
        );
        assert_eq!(direct_cached.statistics().operations_attempted, 0);
        assert_eq!(direct_cached.submitter.memory, reference.submitter.memory);

        // 2. Copied load prepares via prepare_weights and matches arena bytes too.
        let mut copied_cached = NativeQ2Executor::new(
            arena_submitter_at_capacity(false, 16_384)?,
            limits,
            1024 * 1024,
        )
        .with_resident_weights(4096)?;
        NativeQ2ArenaCache::load_and_prewarm_from_verified_cache(
            &mut copied_cached,
            &guard.path,
            model_sha,
            &tensors,
        )?;
        assert_eq!(copied_cached.submitter.copies, 3);
        assert_eq!(copied_cached.submitter.initializations, 0);
        assert_eq!(copied_cached.submitter.memory, reference.submitter.memory);

        // 3. Cached resident handles execute live resident projections cleanly.
        let mut mock_exec = NativeQ2Executor::new(
            MockNativeSubmitter {
                batch_ffn: false,
                ffn_pair_calls: 0,
                calls: 0,
                resident_calls: 0,
                remaining: 16,
                residency_bytes: 16_384,
            },
            limits,
            1024 * 1024,
        )
        .with_resident_weights(4096)?;
        NativeQ2ArenaCache::load_and_prewarm_from_verified_cache(
            &mut mock_exec,
            &guard.path,
            model_sha,
            &tensors,
        )?;
        let activation = vec![1.0_f32; 256];
        let mut output = Vec::new();
        Pq2ProjectionExecutor::project_f32(
            &mut mock_exec,
            &tensors[0],
            &activation,
            true,
            &mut output,
        )?;
        assert_eq!(output.len(), 32);
        assert_eq!(mock_exec.submitter.resident_calls, 1);
        Ok(())
    }

    #[test]
    fn native_q2_arena_cache_rejects_corrupted_payload_before_prepare_weights() -> Result<()> {
        let tensors = cache_fixture_tensors();
        let guard = temp_arena_cache_path("cache-corrupt");
        let model_sha = [0x99_u8; 32];
        NativeQ2ArenaCache::save_verified_cache(&guard.path, 81, model_sha, &tensors, None)?;

        let mut corrupted = std::fs::read(&guard.path)?;
        let last_idx = corrupted.len() - 1;
        corrupted[last_idx] ^= 0x01;
        std::fs::write(&guard.path, &corrupted)?;

        let limits = NativeQ2ProjectionLimits::new(256, 64, 16, Duration::from_secs(4))
            .map_err(|_| invalid("limits"))?
            .with_resident_weights(16_384)
            .map_err(|_| invalid("residency"))?;
        for direct in [false, true] {
            let mut exec = NativeQ2Executor::new(
                arena_submitter_at_capacity(direct, 16_384)?,
                limits,
                1024 * 1024,
            )
            .with_resident_weights(4096)?;
            assert!(
                NativeQ2ArenaCache::load_and_prewarm_from_verified_cache(
                    &mut exec,
                    &guard.path,
                    model_sha,
                    &tensors,
                )
                .is_err()
            );
            assert_eq!(exec.submitter.copies, 0);
            assert_eq!(exec.submitter.initializations, 0);
            assert_eq!(exec.submitter.registry.used_bytes(), 0);
            assert_eq!(exec.residency_statistics().tensors_prepared, 0);
            assert!(exec.allowed_tensors.is_none());
        }

        // Truncated or oversized payload lengths also reject before touching submitter.
        corrupted.pop();
        std::fs::write(&guard.path, &corrupted)?;
        let mut trunc_exec = NativeQ2Executor::new(
            arena_submitter_at_capacity(false, 16_384)?,
            limits,
            1024 * 1024,
        )
        .with_resident_weights(4096)?;
        assert!(
            NativeQ2ArenaCache::load_and_prewarm_from_verified_cache(
                &mut trunc_exec,
                &guard.path,
                model_sha,
                &tensors,
            )
            .is_err()
        );
        assert_eq!(trunc_exec.submitter.copies, 0);
        Ok(())
    }

    #[test]
    fn native_q2_arena_cache_rejects_architecture_model_hash_and_magic_mismatches() -> Result<()> {
        let tensors = cache_fixture_tensors();
        let guard = temp_arena_cache_path("cache-mismatch");
        let model_sha = [0x11_u8; 32];
        NativeQ2ArenaCache::save_verified_cache(&guard.path, 73, model_sha, &tensors, None)?;

        let limits = NativeQ2ProjectionLimits::new(256, 64, 16, Duration::from_secs(4))
            .map_err(|_| invalid("limits"))?
            .with_resident_weights(16_384)
            .map_err(|_| invalid("residency"))?;

        // 1. Architecture 73 cache rejects 81 executor before prepare_weights.
        let mut exec_v81 = NativeQ2Executor::new(
            ArchArenaSubmitter {
                inner: arena_submitter_at_capacity(false, 16_384)?,
                arch: 81,
            },
            limits,
            1024 * 1024,
        )
        .with_resident_weights(4096)?;
        assert!(
            NativeQ2ArenaCache::load_and_prewarm_from_verified_cache(
                &mut exec_v81,
                &guard.path,
                model_sha,
                &tensors,
            )
            .is_err()
        );
        assert_eq!(exec_v81.submitter.inner.copies, 0);

        // 2. Wrong model SHA-256 rejects on matching arch 73 executor.
        let mut exec_v73 = NativeQ2Executor::new(
            ArchArenaSubmitter {
                inner: arena_submitter_at_capacity(false, 16_384)?,
                arch: 73,
            },
            limits,
            1024 * 1024,
        )
        .with_resident_weights(4096)?;
        assert!(
            NativeQ2ArenaCache::load_and_prewarm_from_verified_cache(
                &mut exec_v73,
                &guard.path,
                [0x22_u8; 32],
                &tensors,
            )
            .is_err()
        );
        assert_eq!(exec_v73.submitter.inner.copies, 0);

        // 3. Reordered / wrong tensor list rejects before prepare_weights.
        assert!(
            NativeQ2ArenaCache::load_and_prewarm_from_verified_cache(
                &mut exec_v73,
                &guard.path,
                model_sha,
                &[tensors[1].clone(), tensors[0].clone(), tensors[2].clone()],
            )
            .is_err()
        );
        assert_eq!(exec_v73.submitter.inner.copies, 0);

        // 4. Matching arch 73 + model SHA succeeds.
        NativeQ2ArenaCache::load_and_prewarm_from_verified_cache(
            &mut exec_v73,
            &guard.path,
            model_sha,
            &tensors,
        )?;
        assert_eq!(exec_v73.submitter.inner.copies, 3);

        // 5. Wrong magic rejects cleanly.
        let mut bad_magic = std::fs::read(&guard.path)?;
        bad_magic[7] = b'0';
        std::fs::write(&guard.path, &bad_magic)?;
        let mut exec_v73_b = NativeQ2Executor::new(
            ArchArenaSubmitter {
                inner: arena_submitter_at_capacity(false, 16_384)?,
                arch: 73,
            },
            limits,
            1024 * 1024,
        )
        .with_resident_weights(4096)?;
        assert!(
            NativeQ2ArenaCache::load_and_prewarm_from_verified_cache(
                &mut exec_v73_b,
                &guard.path,
                model_sha,
                &tensors,
            )
            .is_err()
        );
        assert_eq!(exec_v73_b.submitter.inner.copies, 0);
        Ok(())
    }

    fn repack_pq2_to_ptq1_tensor(pq2: &Tensor) -> Result<Tensor> {
        let mut ptq1_data = Vec::with_capacity((pq2.data.len() / 34) * 28);
        for block in pq2.data.as_chunks::<34>().0 {
            let mut unit: [u8; 34] = *block;
            let orig_scale = [unit[0], unit[1]];
            let is_zero = matches!(u16::from_le_bytes(orig_scale), 0 | 0x8000);
            if is_zero {
                unit[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
            }
            let mut packed = bonsai_ternary::pack_ptq(&unit)
                .map_err(|_| invalid("test PQ2 block contained non-ternary symbol"))?;
            if is_zero {
                packed[26..28].copy_from_slice(&orig_scale);
            }
            ptq1_data.extend_from_slice(&packed);
        }
        Ok(Tensor {
            name: pq2.name.clone(),
            dimensions: pq2.dimensions.clone(),
            kind: PTQ1_KIND,
            data: Arc::<[u8]>::from(ptq1_data).into(),
        })
    }

    fn make_ternary_pq2_tensor(name: &str, columns: usize, rows: usize, scales: &[u16]) -> Tensor {
        let blocks = columns / 128;
        let mut data = vec![0_u8; rows * blocks * 34];
        for (block_idx, block) in data.as_chunks_mut::<34>().0.iter_mut().enumerate() {
            let scale_bits = scales[block_idx % scales.len()];
            block[..2].copy_from_slice(&scale_bits.to_le_bytes());
            for (byte_idx, byte) in block[2..].iter_mut().enumerate() {
                let mut packed = 0_u8;
                for lane in 0..4 {
                    let trit = ((block_idx * 7 + byte_idx * 5 + lane * 3) % 3).to_le_bytes()[0];
                    packed |= trit << (lane * 2);
                }
                *byte = packed;
            }
        }
        Tensor {
            name: name.to_owned(),
            dimensions: vec![columns, rows],
            kind: PQ2_KIND,
            data: Arc::<[u8]>::from(data).into(),
        }
    }

    #[test]
    fn pack_ptq1_tile_into_matches_pq2_pack_row_group_byte_for_byte() -> Result<()> {
        let total_blocks = 3;
        let scales = [
            0x3c00_u16, // +1.0
            0xbc00,     // -1.0
            0x0001,     // +min subnormal
            0x8001,     // -min subnormal
            0x03ff,     // +max subnormal
            0x0000,     // +0.0
            0x8000,     // -0.0
            0xb555,     // negative normal
        ];
        for active_rows in [1, 17, 32] {
            let pq2 =
                make_ternary_pq2_tensor("tile-parity", total_blocks * 128, active_rows, &scales);
            let ptq1 = repack_pq2_to_ptq1_tensor(&pq2)?;

            let mut pq2_tiles = vec![0xaa_u8; total_blocks * 2 * 576];
            pack_row_group(
                &pq2.data,
                ChunkGeometry {
                    kind: PQ2_KIND,
                    total_blocks,
                    row_start: 0,
                    active_rows,
                    block_start: 0,
                    chunk_blocks: total_blocks,
                },
                0,
                &mut pq2_tiles,
            )?;

            let mut ptq1_tiles = vec![0xaa_u8; total_blocks * 2 * 576];
            pack_row_group(
                &ptq1.data,
                ChunkGeometry {
                    kind: PTQ1_KIND,
                    total_blocks,
                    row_start: 0,
                    active_rows,
                    block_start: 0,
                    chunk_blocks: total_blocks,
                },
                0,
                &mut ptq1_tiles,
            )?;
            assert_eq!(ptq1_tiles, pq2_tiles, "active_rows={active_rows}");
        }

        // Verify all 256 PTQ byte encodings match bonsai_ternary::decode_ptq_trits
        // when transcoded by pack_ptq1_tile_into.
        let mut pair_buf = vec![0x55_u8; 1152];
        for seed in 0_u16..256 {
            let mut block = [0_u8; 28];
            for (slot, byte) in block[..26].iter_mut().enumerate() {
                *byte = u8::try_from((usize::from(seed) + slot * 11) & 0xff)?;
            }
            block[26..28].copy_from_slice(&0xb800_u16.to_le_bytes()); // -0.5
            initialize_native_weights(&mut pair_buf);
            pack_ptq1_tile_into(&block, 19, &mut pair_buf)?;

            let expected_trits = bonsai_ternary::decode_ptq_trits(&block);
            assert_eq!(&pair_buf[512 + 19 * 2..512 + 19 * 2 + 2], &[0x00, 0xb8]);
            assert_eq!(
                &pair_buf[576 + 512 + 19 * 2..576 + 512 + 19 * 2 + 2],
                &[0x00, 0xb8]
            );
            for (index, &trit) in expected_trits.iter().enumerate() {
                let group = index / 16;
                let lane = (index % 16) / 4;
                let digit = index % 4;
                let at = (group / 4) * 576 + (group % 4) * 128 + 19 * 4 + digit;
                let actual_sym = (pair_buf[at] >> (lane * 2)) & 3;
                assert_eq!(i16::from(actual_sym) - 1, trit, "seed={seed} index={index}");
            }
        }

        // Reject non-finite FP16 scales (+Inf, -Inf, qNaN, sNaN) and bad slice bounds.
        for bad_scale in [0x7c00_u16, 0xfc00, 0x7e00, 0x7c01] {
            let mut bad_block = [0_u8; 28];
            bad_block[26..28].copy_from_slice(&bad_scale.to_le_bytes());
            assert!(pack_ptq1_tile_into(&bad_block, 0, &mut pair_buf).is_err());
        }
        assert!(pack_ptq1_tile_into(&[0_u8; 27], 0, &mut pair_buf).is_err());
        assert!(pack_ptq1_tile_into(&[0_u8; 28], 32, &mut pair_buf).is_err());
        assert!(pack_ptq1_tile_into(&[0_u8; 28], 0, &mut [0_u8; 1151]).is_err());
        Ok(())
    }

    #[test]
    fn native_q2_ptq1_resident_tiles_and_execution_match_pq2() -> Result<()> {
        let scales = [0x3800_u16, 0xbc00, 0x0001, 0x3c00];
        let pq2_gate = make_ternary_pq2_tensor("blk.0.ffn_gate.weight", 256, 49, &scales);
        let pq2_up = make_ternary_pq2_tensor("blk.0.ffn_up.weight", 256, 49, &scales);
        let pq2_down = make_ternary_pq2_tensor("blk.0.ffn_down.weight", 256, 49, &scales);
        let ptq1_gate = repack_pq2_to_ptq1_tensor(&pq2_gate)?;
        let ptq1_up = repack_pq2_to_ptq1_tensor(&pq2_up)?;
        let ptq1_down = repack_pq2_to_ptq1_tensor(&pq2_down)?;

        let limits = NativeQ2ProjectionLimits::new(128, 32, 256, Duration::from_secs(10))
            .map_err(|_| invalid("limits"))?
            .with_resident_weights(65_536)
            .map_err(|_| invalid("residency"))?;

        for direct in [false, true] {
            let mut pq2_exec = NativeQ2Executor::new(
                arena_submitter_at_capacity(direct, 32_768)?,
                limits,
                4 * 1024 * 1024,
            )
            .with_resident_weights(8192)?;
            let mut ptq1_exec = NativeQ2Executor::new(
                arena_submitter_at_capacity(direct, 32_768)?,
                limits,
                4 * 1024 * 1024,
            )
            .with_resident_weights(8192)?;

            assert!(!Pq2ProjectionExecutor::supports(&ptq1_exec, &ptq1_gate));
            assert!(!Pq2ProjectionExecutor::supports_q8(&ptq1_exec, &ptq1_gate));
            assert!(Pq2ProjectionExecutor::supports_f32(&ptq1_exec, &ptq1_gate));

            pq2_exec.select_and_prewarm_tensors(&[
                pq2_gate.clone(),
                pq2_up.clone(),
                pq2_down.clone(),
            ])?;
            ptq1_exec.select_and_prewarm_tensors(&[
                ptq1_gate.clone(),
                ptq1_up.clone(),
                ptq1_down.clone(),
            ])?;
            assert_eq!(ptq1_exec.submitter.memory, pq2_exec.submitter.memory);
            assert_eq!(
                ptq1_exec
                    .residency_statistics()
                    .weight_payload_bytes_prepared,
                pq2_exec
                    .residency_statistics()
                    .weight_payload_bytes_prepared
            );
            assert!(!Pq2ProjectionExecutor::supports_f32(&ptq1_exec, &pq2_gate));
        }

        // Single-op resident + group + FFN chain execution on PTQ1_0.
        let ptq1_sq = repack_pq2_to_ptq1_tensor(&make_ternary_pq2_tensor(
            "blk.0.ffn_gate.weight",
            256,
            256,
            &scales,
        ))?;
        let ptq1_sq_up = repack_pq2_to_ptq1_tensor(&make_ternary_pq2_tensor(
            "blk.0.ffn_up.weight",
            256,
            256,
            &scales,
        ))?;
        let ptq1_sq_down = repack_pq2_to_ptq1_tensor(&make_ternary_pq2_tensor(
            "blk.0.ffn_down.weight",
            256,
            256,
            &scales,
        ))?;

        let single_limits =
            NativeQ2ProjectionLimits::new(5120, 17_408, 128, Duration::from_secs(10))
                .map_err(|_| invalid("limits"))?
                .with_resident_weights(131_072)
                .map_err(|_| invalid("residency"))?;
        let mut mock_sub = MockNativeSubmitter {
            batch_ffn: true,
            ffn_pair_calls: 0,
            calls: 0,
            resident_calls: 0,
            remaining: 128,
            residency_bytes: 131_072,
        };
        let mut exec = NativeQ2Executor::new(&mut mock_sub, single_limits, 16 * 1024 * 1024)
            .with_resident_weights(16_384)?;
        let act = vec![1.25_f32; 256];
        let mut out = Vec::new();
        Pq2ProjectionExecutor::project_f32(&mut exec, &ptq1_sq, &act, true, &mut out)?;
        assert_eq!(out.len(), 256);
        assert!(Pq2ProjectionExecutor::supports_f32_ffn_chain(
            &exec,
            [&ptq1_sq, &ptq1_sq_up, &ptq1_sq_down]
        ));
        Pq2ProjectionExecutor::project_f32_ffn_chain(
            &mut exec,
            [&ptq1_sq, &ptq1_sq_up, &ptq1_sq_down],
            &act,
            None,
            Some(64),
            true,
            &mut out,
        )?;
        assert_eq!(out.len(), 256);
        Ok(())
    }

    #[test]
    fn native_q2_ptq1_streaming_and_scale_rejection() -> Result<()> {
        let scales = [0x3800_u16, 0xbc00, 0x0001, 0x3c00];
        let pq2_gate = make_ternary_pq2_tensor("blk.0.ffn_gate.weight", 256, 49, &scales);
        let ptq1_gate = repack_pq2_to_ptq1_tensor(&pq2_gate)?;

        // Streaming (nonresident) single-op + multi-tile fallback parity with PQ2_0.
        let stream_limits = NativeQ2ProjectionLimits::new(128, 32, 64, Duration::from_secs(10))
            .map_err(|_| invalid("limits"))?;
        let mut pq2_stream = NativeQ2Executor::new(
            MockNativeSubmitter {
                batch_ffn: false,
                ffn_pair_calls: 0,
                calls: 0,
                resident_calls: 0,
                remaining: 64,
                residency_bytes: 0,
            },
            stream_limits,
            1024 * 1024,
        );
        let mut ptq1_stream = NativeQ2Executor::new(
            MockNativeSubmitter {
                batch_ffn: false,
                ffn_pair_calls: 0,
                calls: 0,
                resident_calls: 0,
                remaining: 64,
                residency_bytes: 0,
            },
            stream_limits,
            1024 * 1024,
        );
        let act = vec![1.25_f32; 256];
        let mut pq2_out = Vec::new();
        let mut ptq1_out = Vec::new();
        pq2_stream.project_f32(&pq2_gate, &act, true, &mut pq2_out)?;
        ptq1_stream.project_f32(&ptq1_gate, &act, true, &mut ptq1_out)?;
        assert_eq!(ptq1_out, pq2_out);

        let limits = NativeQ2ProjectionLimits::new(128, 32, 256, Duration::from_secs(10))
            .map_err(|_| invalid("limits"))?
            .with_resident_weights(65_536)
            .map_err(|_| invalid("residency"))?;

        // Non-finite PTQ1 scale rejects without committing handles (both prewarm & project_f32).
        let mut bad_ptq1 = ptq1_gate;
        let mut bad_bytes = bad_ptq1.data.to_vec();
        bad_bytes[26..28].copy_from_slice(&0x7e00_u16.to_le_bytes());
        bad_ptq1.data = Arc::<[u8]>::from(bad_bytes).into();
        let mut err_exec = NativeQ2Executor::new(arena_submitter(true)?, limits, 1024 * 1024)
            .with_resident_weights(4096)?;
        assert!(err_exec.prewarm_tensor(&bad_ptq1).is_err());
        assert_eq!(err_exec.residency_statistics().tensors_prepared, 0);
        let mut out = Vec::new();
        assert!(
            err_exec
                .project_f32(&bad_ptq1, &act, true, &mut out)
                .is_err()
        );
        assert!(err_exec.is_faulted());

        // Single-op prewarm on non-finite PTQ1 scale faults immediately without committing handles.
        let bad_single_ptq1 = {
            let mut t = repack_pq2_to_ptq1_tensor(&make_ternary_pq2_tensor(
                "bad-single",
                128,
                32,
                &scales,
            ))?;
            let mut bytes = t.data.to_vec();
            bytes[26..28].copy_from_slice(&0x7c00_u16.to_le_bytes());
            t.data = Arc::<[u8]>::from(bytes).into();
            t
        };
        let mut single_err_exec =
            NativeQ2Executor::new(arena_submitter(true)?, limits, 1024 * 1024)
                .with_resident_weights(4096)?;
        assert!(single_err_exec.prewarm_tensor(&bad_single_ptq1).is_err());
        assert!(single_err_exec.is_faulted());
        assert_eq!(single_err_exec.residency_statistics().tensors_prepared, 0);
        assert_eq!(single_err_exec.submitter.registry.used_bytes(), 0);
        Ok(())
    }

    #[test]
    fn native_q2_prepare_resident_parallel_parity_pq2_and_ptq1() -> Result<()> {
        let scales = [0x3800_u16, 0xbc00, 0x0001, 0x3c00, 0xb555];
        let pq2 = make_ternary_pq2_tensor("blk.0.ffn_up.weight", 256, 81, &scales);
        let ptq1 = repack_pq2_to_ptq1_tensor(&pq2)?;
        let limits = NativeQ2ProjectionLimits::new(256, 128, 32, Duration::from_secs(10))
            .map_err(|_| invalid("limits"))?
            .with_resident_weights(32_768)
            .map_err(|_| invalid("residency"))?;

        let workers_to_test: &[usize] = if cfg!(feature = "parallel-rows") {
            &[1, 4]
        } else {
            &[1]
        };

        for tensor in [&pq2, &ptq1] {
            let mut serial = NativeQ2Executor::new(
                arena_submitter_at_capacity(true, 32_768)?,
                limits,
                4 * 1024 * 1024,
            )
            .with_resident_weights(4096)?;
            assert!(serial.prewarm_tensor(tensor)?);

            for &workers in workers_to_test {
                let mut par = NativeQ2Executor::new(
                    arena_submitter_at_capacity(true, 32_768)?,
                    limits,
                    4 * 1024 * 1024,
                )
                .with_resident_weights(4096)?
                .with_row_parallelism(RowParallelism::new(workers)?)?;
                assert!(par.prewarm_tensor(tensor)?);
                assert_eq!(par.submitter.memory, serial.submitter.memory);
            }
        }
        Ok(())
    }

    #[test]
    fn native_q2_arena_cache_ptq1_round_trip_and_kind_isolation() -> Result<()> {
        let scales = [0x3800_u16, 0xbc00, 0x3c00];
        let pq2_tensors = vec![
            make_ternary_pq2_tensor("blk.0.ffn_gate.weight", 256, 32, &scales),
            make_ternary_pq2_tensor("blk.0.ffn_up.weight", 256, 49, &scales),
            make_ternary_pq2_tensor("blk.0.ffn_down.weight", 256, 32, &scales),
        ];
        let ptq1_tensors: Vec<Tensor> = pq2_tensors
            .iter()
            .map(repack_pq2_to_ptq1_tensor)
            .collect::<Result<_>>()?;

        let pq2_guard = temp_arena_cache_path("cache-pq2-kind");
        let ptq1_guard = temp_arena_cache_path("cache-ptq1-kind");
        let shared_model_sha = [0x5a_u8; 32];

        let pq2_entries = NativeQ2ArenaCache::save_verified_cache(
            &pq2_guard.path,
            81,
            shared_model_sha,
            &pq2_tensors,
            None,
        )?;
        let ptq1_entries = NativeQ2ArenaCache::save_verified_cache(
            &ptq1_guard.path,
            81,
            shared_model_sha,
            &ptq1_tensors,
            None,
        )?;
        assert_ne!(pq2_entries, ptq1_entries);
        assert!(ptq1_entries.iter().all(|entry| entry.kind == PTQ1_KIND));

        let limits = NativeQ2ProjectionLimits::new(256, 64, 16, Duration::from_secs(4))
            .map_err(|_| invalid("limits"))?
            .with_resident_weights(16_384)
            .map_err(|_| invalid("residency"))?;

        let mut pq2_exec = NativeQ2Executor::new(
            arena_submitter_at_capacity(true, 16_384)?,
            limits,
            1024 * 1024,
        )
        .with_resident_weights(4096)?;
        NativeQ2ArenaCache::load_and_prewarm_from_verified_cache(
            &mut pq2_exec,
            &pq2_guard.path,
            shared_model_sha,
            &pq2_tensors,
        )?;

        let mut ptq1_exec = NativeQ2Executor::new(
            arena_submitter_at_capacity(true, 16_384)?,
            limits,
            1024 * 1024,
        )
        .with_resident_weights(4096)?;
        NativeQ2ArenaCache::load_and_prewarm_from_verified_cache(
            &mut ptq1_exec,
            &ptq1_guard.path,
            shared_model_sha,
            &ptq1_tensors,
        )?;
        assert_eq!(ptq1_exec.submitter.memory, pq2_exec.submitter.memory);

        // Cross-kind loads with identical model_sha, names and shapes must reject before prepare_weights.
        let mut wrong_kind_a = NativeQ2Executor::new(
            arena_submitter_at_capacity(false, 16_384)?,
            limits,
            1024 * 1024,
        )
        .with_resident_weights(4096)?;
        assert!(
            NativeQ2ArenaCache::load_and_prewarm_from_verified_cache(
                &mut wrong_kind_a,
                &pq2_guard.path,
                shared_model_sha,
                &ptq1_tensors,
            )
            .is_err()
        );
        assert_eq!(wrong_kind_a.submitter.copies, 0);

        let mut wrong_kind_b = NativeQ2Executor::new(
            arena_submitter_at_capacity(false, 16_384)?,
            limits,
            1024 * 1024,
        )
        .with_resident_weights(4096)?;
        assert!(
            NativeQ2ArenaCache::load_and_prewarm_from_verified_cache(
                &mut wrong_kind_b,
                &ptq1_guard.path,
                shared_model_sha,
                &pq2_tensors,
            )
            .is_err()
        );
        assert_eq!(wrong_kind_b.submitter.copies, 0);
        Ok(())
    }

    fn swift_layer_specs(layer_idx: usize) -> &'static [(&'static str, usize, usize)] {
        if (layer_idx + 1).is_multiple_of(4) {
            &[
                ("ffn_up.weight", 5120, 17_408),
                ("ffn_gate.weight", 5120, 17_408),
                ("ffn_down.weight", 17_408, 5120),
                ("attn_q.weight", 5120, 12_288),
                ("attn_k.weight", 5120, 1024),
                ("attn_v.weight", 5120, 1024),
                ("attn_output.weight", 6144, 5120),
            ]
        } else {
            &[
                ("ffn_up.weight", 5120, 17_408),
                ("ffn_gate.weight", 5120, 17_408),
                ("ffn_down.weight", 17_408, 5120),
                ("attn_qkv.weight", 5120, 10_240),
                ("attn_gate.weight", 5120, 6144),
                ("ssm_out.weight", 6144, 5120),
            ]
        }
    }

    fn synthetic_swift_model(layer_count: usize, kind_for_layer: impl Fn(usize) -> u32) -> Model {
        let mut shape_cache: BTreeMap<(u32, usize, usize), Arc<[u8]>> = BTreeMap::new();
        let mut tensors = BTreeMap::new();
        for layer_idx in 0..layer_count {
            let kind = kind_for_layer(layer_idx);
            let block_bytes = source_block_bytes(kind).unwrap_or(34);
            for &(suffix, columns, rows) in swift_layer_specs(layer_idx) {
                let bytes = shape_cache
                    .entry((kind, columns, rows))
                    .or_insert_with(|| {
                        Arc::<[u8]>::from(vec![0_u8; rows * (columns / 128) * block_bytes])
                    })
                    .clone();
                let name = format!("blk.{layer_idx}.{suffix}");
                tensors.insert(
                    name.clone(),
                    Tensor {
                        name,
                        kind,
                        dimensions: vec![columns, rows],
                        data: bytes.into(),
                    },
                );
            }
            if layer_idx == 0 {
                tensors.insert(
                    "blk.0.attn_norm.weight".to_owned(),
                    Tensor {
                        name: "blk.0.attn_norm.weight".to_owned(),
                        kind: 0,
                        dimensions: vec![5120],
                        data: Arc::<[u8]>::from(vec![0_u8; 5120 * 4]).into(),
                    },
                );
            }
        }
        tensors.insert(
            "output.weight".to_owned(),
            Tensor {
                name: "output.weight".to_owned(),
                kind: PTQ1_KIND,
                dimensions: vec![5120, 1024],
                data: Arc::<[u8]>::from(vec![0_u8; 1024 * (5120 / 128) * 28]).into(),
            },
        );
        Model {
            identity: "synthetic-swift-vtcm-plan".into(),
            metadata: BTreeMap::new(),
            tensors,
        }
    }

    #[test]
    fn vtcm_planner_v81_64_layers_matches_verified_27_layer_baseline_for_pq2_and_ptq1() -> Result<()>
    {
        let pq2_model = synthetic_swift_model(64, |_| PQ2_KIND);
        let ptq1_model = synthetic_swift_model(64, |_| PTQ1_KIND);
        let pq2_plan = plan_native_q2_layers_for_budget(
            &pq2_model,
            81,
            NATIVE_Q2_MAX_ARENA_WEIGHT_BYTES,
            NATIVE_Q2_MAX_WEIGHT_ARENAS,
            8 * 1024 * 1024,
        )?;
        let ptq1_plan = plan_native_q2_layers_for_budget(
            &ptq1_model,
            81,
            NATIVE_Q2_MAX_ARENA_WEIGHT_BYTES,
            NATIVE_Q2_MAX_WEIGHT_ARENAS,
            8 * 1024 * 1024,
        )?;
        assert_eq!(pq2_plan, ptq1_plan);
        assert_eq!(pq2_plan.architecture, 81);
        assert_eq!(pq2_plan.total_layers, 64);
        assert_eq!(pq2_plan.admitted_layers, 27);
        assert_eq!(
            pq2_plan.resident_bytes,
            DEFAULT_NATIVE_Q2_RESIDENT_WEIGHT_BYTES
        );
        assert_eq!(pq2_plan.usable_vtcm_bytes, NATIVE_Q2_VTCM_REQUEST_BYTES);
        assert!(pq2_plan.peak_vtcm_working_bytes > 0);
        assert!(pq2_plan.peak_vtcm_working_bytes <= pq2_plan.usable_vtcm_bytes);
        assert_eq!(
            pq2_plan.npu_projections,
            DEFAULT_NATIVE_Q2_PROJECTIONS
                .iter()
                .map(|&name| name.to_owned())
                .collect::<Vec<_>>()
        );
        assert_eq!(pq2_plan.overflow_projections.len(), 232);
        assert_eq!(pq2_plan.overflow_projections[0], "blk.27.ffn_up.weight");
        assert_eq!(
            pq2_plan.overflow_projections.last().map(String::as_str),
            Some("blk.63.attn_output.weight")
        );
        assert_eq!(pq2_plan.arena_assignments.len(), 6);
        let arena_sum: usize = pq2_plan
            .arena_assignments
            .iter()
            .map(|a| a.used_bytes)
            .sum();
        assert_eq!(arena_sum, DEFAULT_NATIVE_Q2_RESIDENT_WEIGHT_BYTES);
        for arena in &pq2_plan.arena_assignments {
            assert_eq!(
                arena.used_bytes + arena.remaining_bytes,
                NATIVE_Q2_MAX_ARENA_WEIGHT_BYTES
            );
            assert_ne!(arena.projections.len(), 0);
        }
        Ok(())
    }

    #[test]
    fn vtcm_planner_v73_smaller_arenas_and_overflow_boundary_layers() -> Result<()> {
        let mixed_model = synthetic_swift_model(64, |idx| {
            if idx.is_multiple_of(2) {
                PTQ1_KIND
            } else {
                PQ2_KIND
            }
        });
        let small_arena_bytes = 256 * 1024 * 1024;
        let v73_small = plan_native_q2_layers_for_budget(
            &mixed_model,
            73,
            small_arena_bytes,
            4,
            4 * 1024 * 1024,
        )?;
        assert_eq!(v73_small.architecture, 73);
        assert_eq!(v73_small.total_layers, 64);
        assert_eq!(v73_small.usable_vtcm_bytes, 3_670_016);
        assert!(v73_small.peak_vtcm_working_bytes <= v73_small.usable_vtcm_bytes);
        assert!(v73_small.admitted_layers > 0 && v73_small.admitted_layers < 27);
        assert_eq!(v73_small.arena_assignments.len(), 4);
        assert_eq!(
            v73_small.overflow_projections[0],
            format!("blk.{}.ffn_up.weight", v73_small.admitted_layers)
        );
        let expected_npu_count: usize = (0..v73_small.admitted_layers)
            .map(|idx| swift_layer_specs(idx).len())
            .sum();
        assert_eq!(v73_small.npu_projections.len(), expected_npu_count);
        assert_eq!(
            v73_small.npu_projections.len() + v73_small.overflow_projections.len(),
            400
        );
        let flattened_arena_projections: Vec<String> = v73_small
            .arena_assignments
            .iter()
            .flat_map(|arena| arena.projections.iter().cloned())
            .collect();
        assert_eq!(flattened_arena_projections, v73_small.npu_projections);
        let v73_small_arena_sum: usize = v73_small
            .arena_assignments
            .iter()
            .map(|arena| arena.used_bytes)
            .sum();
        assert_eq!(v73_small_arena_sum, v73_small.resident_bytes);

        let v73_full = plan_native_q2_layers_for_budget(
            &mixed_model,
            73,
            NATIVE_Q2_MAX_ARENA_WEIGHT_BYTES,
            6,
            4 * 1024 * 1024,
        )?;
        assert_eq!(v73_full.admitted_layers, 27);
        assert_eq!(
            v73_full.resident_bytes,
            DEFAULT_NATIVE_Q2_RESIDENT_WEIGHT_BYTES
        );

        let small_model = synthetic_swift_model(8, |_| PTQ1_KIND);
        let fit_all = plan_native_q2_layers_for_budget(
            &small_model,
            73,
            NATIVE_Q2_MAX_ARENA_WEIGHT_BYTES,
            6,
            4 * 1024 * 1024,
        )?;
        assert_eq!(fit_all.admitted_layers, 8);
        assert_eq!(fit_all.overflow_projections.len(), 0);
        assert_eq!(fit_all.arena_assignments.len(), 2);
        Ok(())
    }

    #[test]
    fn vtcm_planner_rejects_invalid_budgets_vtcm_and_malformed_models() {
        let model = synthetic_swift_model(4, |_| PTQ1_KIND);
        let max_arena = NATIVE_Q2_MAX_ARENA_WEIGHT_BYTES;

        // Zero, sub-floor, and insufficient VTCM budgets.
        assert!(plan_native_q2_layers_for_budget(&model, 81, max_arena, 6, 0).is_err());
        assert!(plan_native_q2_layers_for_budget(&model, 73, max_arena, 6, 0).is_err());
        assert!(plan_native_q2_layers_for_budget(&model, 73, max_arena, 6, 65_536).is_err());
        assert!(
            plan_native_q2_layers_for_budget(&model, 81, max_arena, 6, 2 * 1024 * 1024).is_err()
        );
        assert!(plan_native_q2_layers_for_budget(&model, 73, max_arena, 6, 1024 * 1024).is_err());

        // Invalid architectures and arena budgets.
        assert!(
            plan_native_q2_layers_for_budget(&model, 68, max_arena, 6, 8 * 1024 * 1024).is_err()
        );
        assert!(plan_native_q2_layers_for_budget(&model, 81, 0, 6, 8 * 1024 * 1024).is_err());
        assert!(
            plan_native_q2_layers_for_budget(&model, 81, max_arena + 1, 6, 8 * 1024 * 1024)
                .is_err()
        );
        assert!(
            plan_native_q2_layers_for_budget(&model, 81, max_arena, 0, 8 * 1024 * 1024).is_err()
        );
        assert!(
            plan_native_q2_layers_for_budget(&model, 81, max_arena, 7, 8 * 1024 * 1024).is_err()
        );
        // Cannot fit even layer 0 in 64 MiB single arena.
        assert!(
            plan_native_q2_layers_for_budget(&model, 81, 64 * 1024 * 1024, 1, 8 * 1024 * 1024)
                .is_err()
        );

        // Malformed models: empty, non-contiguous gap, incomplete layer, wrong canonical kind.
        let empty = synthetic_swift_model(0, |_| PTQ1_KIND);
        assert!(
            plan_native_q2_layers_for_budget(&empty, 81, max_arena, 6, 8 * 1024 * 1024).is_err()
        );

        let mut gapped = synthetic_swift_model(3, |_| PTQ1_KIND);
        let blk1_keys: Vec<String> = gapped
            .tensors
            .keys()
            .filter(|name| name.starts_with("blk.1."))
            .cloned()
            .collect();
        for key in blk1_keys {
            gapped.tensors.remove(&key);
        }
        assert!(
            plan_native_q2_layers_for_budget(&gapped, 81, max_arena, 6, 8 * 1024 * 1024).is_err()
        );

        let mut incomplete = synthetic_swift_model(2, |_| PTQ1_KIND);
        incomplete.tensors.remove("blk.0.ssm_out.weight");
        assert!(
            plan_native_q2_layers_for_budget(&incomplete, 81, max_arena, 6, 8 * 1024 * 1024)
                .is_err()
        );

        let mut bad_kind = synthetic_swift_model(1, |_| PTQ1_KIND);
        if let Some(tensor) = bad_kind.tensors.get_mut("blk.0.ffn_up.weight") {
            tensor.kind = 0;
        }
        assert!(
            plan_native_q2_layers_for_budget(&bad_kind, 81, max_arena, 6, 8 * 1024 * 1024).is_err()
        );
    }

    struct HostReleaseTestSubmitter {
        registry: hexagon_native_rpc::NativeQ2WeightRegistry,
    }

    impl HostReleaseTestSubmitter {
        fn new(capacity: usize) -> Result<Self> {
            Ok(Self {
                registry: hexagon_native_rpc::NativeQ2WeightRegistry::new(capacity)
                    .map_err(|_| invalid("registry"))?,
            })
        }

        fn check_handles(&self, handles: &[&NativeQ2WeightHandle]) -> std::result::Result<(), hexagon_native_rpc::Error> {
            for &handle in handles {
                let _ = self.registry.range(handle)?;
            }
            Ok(())
        }
    }

    #[expect(
        clippy::suspicious_operation_groupings,
        reason = "Test mock compares handle geometry against activation/output slice lengths."
    )]
    impl NativeQ2Submitter for HostReleaseTestSubmitter {
        fn residency_capacity_bytes(&self) -> usize {
            self.registry.remaining_bytes()
        }

        fn prepare_weights(
            &mut self,
            columns: usize,
            rows: usize,
            weights: &[u8],
        ) -> std::result::Result<Option<NativeQ2WeightHandle>, hexagon_native_rpc::Error> {
            self.registry.prepare(columns, rows, weights)
        }

        fn submit(
            &mut self,
            _columns: usize,
            _rows: usize,
            _weights: &[u8],
            _activation: &[f32],
            _output: &mut [f32],
        ) -> std::result::Result<(), hexagon_native_rpc::Error> {
            Err(hexagon_native_rpc::Error::Contract)
        }

        fn submit_resident(
            &mut self,
            handle: &NativeQ2WeightHandle,
            activation: &[f32],
            output: &mut [f32],
        ) -> std::result::Result<(), hexagon_native_rpc::Error> {
            self.check_handles(&[handle])?;
            if activation.len() != handle.columns() || output.len() != handle.rows() {
                return Err(hexagon_native_rpc::Error::Contract);
            }
            let sum: f32 = activation.iter().sum();
            output.fill(sum);
            Ok(())
        }

        fn submit_resident_batch2(
            &mut self,
            handles: [&NativeQ2WeightHandle; 2],
            activation: &[f32],
            outputs: [&mut [f32]; 2],
        ) -> std::result::Result<(), hexagon_native_rpc::Error> {
            self.check_handles(&handles)?;
            for (member, handle) in handles.into_iter().enumerate() {
                if activation.len() != handle.columns() || outputs[member].len() != handle.rows() {
                    return Err(hexagon_native_rpc::Error::Contract);
                }
                let sum: f32 = activation.iter().sum();
                outputs[member].fill(sum + f32::from(u8::try_from(member).unwrap_or(0)));
            }
            Ok(())
        }

        fn submit_resident_ffn_swiglu(
            &mut self,
            gate_handle: &NativeQ2WeightHandle,
            up_handle: &NativeQ2WeightHandle,
            activation: &[f32],
            output: &mut [f32],
        ) -> std::result::Result<(), hexagon_native_rpc::Error> {
            self.check_handles(&[gate_handle, up_handle])?;
            if activation.len() != gate_handle.columns()
                || gate_handle.columns() != up_handle.columns()
                || gate_handle.rows() != up_handle.rows()
                || output.len() != gate_handle.rows()
            {
                return Err(hexagon_native_rpc::Error::Contract);
            }
            let sum: f32 = activation.iter().sum();
            output.fill(sum * 2.0);
            Ok(())
        }

        fn submit_resident_ffn_chain(
            &mut self,
            handles: [&NativeQ2WeightHandle; 3],
            activation: &[f32],
            residual: Option<&[f32]>,
            _hadamard_block_size: Option<u32>,
            output: &mut [f32],
        ) -> std::result::Result<(), hexagon_native_rpc::Error> {
            self.check_handles(&handles)?;
            let [gate, up, down] = handles;
            if gate.columns() != up.columns()
                || gate.rows() != up.rows()
                || down.columns() != gate.rows()
                || down.rows() != gate.columns()
                || activation.len() != gate.columns()
                || output.len() != down.rows()
            {
                return Err(hexagon_native_rpc::Error::Contract);
            }
            let sum: f32 = activation.iter().sum();
            let res0 = residual.and_then(|r| r.first().copied()).unwrap_or(0.0);
            output.fill(sum.mul_add(3.0, res0));
            Ok(())
        }

        fn supports_ffn_chain_lora(&self) -> bool {
            true
        }

        fn submit_resident_ffn_chain_lora(
            &mut self,
            handles: [&NativeQ2WeightHandle; 3],
            activation: &[f32],
            residual: Option<&[f32]>,
            hadamard_block_size: Option<u32>,
            lora_a: &[f32],
            output: &mut [f32],
        ) -> std::result::Result<f32, hexagon_native_rpc::Error> {
            self.submit_resident_ffn_chain(
                handles,
                activation,
                residual,
                hadamard_block_size,
                output,
            )?;
            if lora_a.len() != handles[2].columns() {
                return Err(hexagon_native_rpc::Error::Contract);
            }
            Ok(lora_a.iter().sum())
        }
    }

    fn host_release_test_tensor(name: &str, kind: u32) -> Result<Tensor> {
        let mut block = [0x55_u8; 34];
        block[..2].copy_from_slice(&0x3800_u16.to_le_bytes());
        let data = if kind == PTQ1_KIND {
            bonsai_ternary::pack_ptq(&block)?.repeat(128)
        } else {
            block.repeat(128)
        };
        Ok(Tensor {
            name: name.to_owned(),
            dimensions: vec![128, 128],
            kind,
            data: Arc::<[u8]>::from(data).into(),
        })
    }

    #[test]
    fn resident_host_weight_release_preserves_f32_and_ffn_chain_execution_and_rejects_forged()
    -> Result<()> {
        use bonsai_prism_runtime::model::TensorBytes;

        let tok_a = TensorBytes::default();
        let tok_b = TensorBytes::default();
        assert!(!TensorBytes::ptr_eq(&tok_a, &tok_b));
        assert_ne!(tok_a.as_ptr().addr(), tok_b.as_ptr().addr());

        let limits = NativeQ2ProjectionLimits::new(128, 128, 256, Duration::from_secs(10))
            .map_err(|_| invalid("limits"))?
            .with_resident_weights(65_536)
            .map_err(|_| invalid("residency"))?;
        let mut executor =
            NativeQ2Executor::new(HostReleaseTestSubmitter::new(65_536)?, limits, 1024 * 1024)
                .with_resident_weights(4096)?;

        let mut gate = host_release_test_tensor("blk.0.ffn_gate.weight", PQ2_KIND)?;
        let mut up = host_release_test_tensor("blk.0.ffn_up.weight", PTQ1_KIND)?;
        let mut down = host_release_test_tensor("blk.0.ffn_down.weight", PTQ1_KIND)?;
        executor.select_and_prewarm_tensors(&[gate.clone(), up.clone(), down.clone()])?;

        let act = vec![0.5_f32; 128];
        let res = vec![1.25_f32; 128];
        let lora_a = vec![0.25_f32; 128];
        let (mut b_gate, mut b_up, mut b_grp0, mut b_grp1, mut b_swi, mut b_chn, mut b_lora) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        executor.project_f32(&gate, &act, true, &mut b_gate)?;
        executor.project_f32(&up, &act, true, &mut b_up)?;
        executor.project_f32_group(&[&gate, &up], &act, true, &mut [&mut b_grp0, &mut b_grp1])?;
        executor.project_f32_ffn_swiglu(&gate, &up, &act, &mut b_swi)?;
        executor.project_f32_ffn_chain([&gate, &up, &down], &act, Some(&res), Some(64), &mut b_chn)?;
        let b_dot = executor.project_f32_ffn_chain_lora(
            [&gate, &up, &down],
            &act,
            Some(&res),
            Some(64),
            &lora_a,
            &mut b_lora,
        )?;

        for tensor in [&mut gate, &mut up, &mut down] {
            let orig = tensor.data.clone();
            assert_eq!(TensorBytes::strong_count(&orig), 4);
            let token = TensorBytes::default();
            assert!(executor.release_host_weight_binding(tensor, &token));
            tensor.data = token;
            assert_eq!(TensorBytes::strong_count(&orig), 1);
            assert!(!executor.release_host_weight_binding(tensor, &TensorBytes::default()));
            assert!(executor.has_released_host_weight(tensor));
            assert!(executor.supports_f32(tensor));
        }

        let (mut r_gate, mut r_up, mut r_grp0, mut r_grp1, mut r_swi, mut r_chn, mut r_lora) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        executor.project_f32(&gate, &act, true, &mut r_gate)?;
        executor.project_f32(&up, &act, true, &mut r_up)?;
        executor.project_f32_group(&[&gate, &up], &act, true, &mut [&mut r_grp0, &mut r_grp1])?;
        executor.project_f32_ffn_swiglu(&gate, &up, &act, &mut r_swi)?;
        executor.project_f32_ffn_chain([&gate, &up, &down], &act, Some(&res), Some(64), &mut r_chn)?;
        let r_dot = executor.project_f32_ffn_chain_lora(
            [&gate, &up, &down],
            &act,
            Some(&res),
            Some(64),
            &lora_a,
            &mut r_lora,
        )?;
        assert_eq!(r_gate.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), b_gate.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        assert_eq!(r_up.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), b_up.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        assert_eq!(r_grp0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), b_grp0.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        assert_eq!(r_grp1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), b_grp1.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        assert_eq!(r_swi.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), b_swi.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        assert_eq!(r_chn.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), b_chn.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        assert_eq!(r_lora.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), b_lora.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        assert_eq!(r_dot.to_bits(), b_dot.to_bits());

        let mut forged = gate.clone();
        forged.data = TensorBytes::default();
        assert!(!executor.has_released_host_weight(&forged));
        assert!(!executor.supports_f32(&forged));
        assert!(executor.project_f32(&forged, &act, true, &mut r_gate).is_err());
        Ok(())
    }

    #[test]
    fn resident_host_weight_release_interoperates_with_prepared_matvec() -> Result<()> {
        use bonsai_prism_runtime::{kernels::MatvecScratch, model::TensorBytes};

        let limits = NativeQ2ProjectionLimits::new(128, 128, 256, Duration::from_secs(10))
            .map_err(|_| invalid("limits"))?
            .with_resident_weights(65_536)
            .map_err(|_| invalid("residency"))?;
        let mut executor =
            NativeQ2Executor::new(HostReleaseTestSubmitter::new(65_536)?, limits, 1024 * 1024)
                .with_resident_weights(4096)?;

        let mut gate = host_release_test_tensor("blk.0.ffn_gate.weight", PQ2_KIND)?;
        let mut up = host_release_test_tensor("blk.0.ffn_up.weight", PTQ1_KIND)?;
        let mut down = host_release_test_tensor("blk.0.ffn_down.weight", PTQ1_KIND)?;
        let compat_unreleased = gate.clone();
        executor.select_and_prewarm_tensors(&[gate.clone(), up.clone(), down.clone()])?;

        let act = vec![0.5_f32; 128];
        let res = vec![1.25_f32; 128];
        let mut base_gate = Vec::new();
        let (mut base_g0, mut base_g1, mut base_chain) = (Vec::new(), Vec::new(), Vec::new());
        executor.project_f32(&gate, &act, false, &mut base_gate)?;
        executor.project_f32_group(&[&gate, &up], &act, false, &mut [&mut base_g0, &mut base_g1])?;
        executor.project_f32_ffn_chain([&gate, &up, &down], &act, Some(&res), Some(64), &mut base_chain)?;

        for tensor in [&mut gate, &mut up, &mut down] {
            let token = TensorBytes::default();
            assert!(executor.release_host_weight_binding(tensor, &token));
            tensor.data = token;
        }

        let mut scratch = MatvecScratch::default();
        assert!(scratch.prepare(&gate, &act).is_err());
        let mut prepared = scratch.prepare(&compat_unreleased, &act)?;

        let (mut out_single, mut out_g0, mut out_g1, mut out_chain) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        prepared.matvec_with_executor(&gate, &mut out_single, &mut executor)?;
        prepared.matvec_group_with_executor(
            &[&gate, &up],
            &mut [&mut out_g0, &mut out_g1],
            &mut executor,
        )?;
        prepared.matvec_ffn_chain_with_executor(
            [&gate, &up, &down],
            Some(&res),
            Some(64),
            &mut out_chain,
            &mut executor,
        )?;
        assert_eq!(out_single, base_gate);
        assert_eq!(out_g0, base_g0);
        assert_eq!(out_g1, base_g1);
        assert_eq!(out_chain, base_chain);

        let mut forged = gate.clone();
        forged.data = TensorBytes::default();
        assert!(prepared.matvec_with_executor(&forged, &mut out_single, &mut executor).is_err());
        Ok(())
    }
}
