//! Full-`K` 1-op native `HTP_TYPE_Q2_0` projection executor over `libggml-htp-v81.so`.
//!
//! Losslessly transcodes Swift's `PQ2_0` (`34` bytes per 128 weights) into
//! `llama.cpp`'s native tiled `HTP_TYPE_Q2_0` (`576` bytes per 32x64 tile,
//! repeating each 128-weight `FP16` scale across both 64-weight sub-tiles) and
//! executes the full `K <= 5120`, `M <= 17408` projection in a single DSP
//! operation (`HTP_MM_KERNEL_HVX_QUANT_ROW_N1`).

use std::sync::Arc;

use bonsai_prism_runtime::{kernels::Pq2ProjectionExecutor, model::Tensor};
use bonsai_ternary::Q8KBlock;
use hexagon_native_rpc::{NativeQ2ProjectionLimits, NativeQ2Submitter, NativeQ2WeightHandle};

use crate::guarded::{finite, scale, zeroed};
use crate::{HmxResidencyStatistics, ProjectionStatistics, Result, invalid};

/// Residency statistics for [`NativeQ2Executor`].
pub type NativeQ2ResidencyStatistics = HmxResidencyStatistics;

struct ResidentProjection {
    source: Arc<[u8]>,
    columns: usize,
    rows: usize,
    first_handle: usize,
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
            tensor.kind == 142
                && tensor.dimensions.as_slice() == [projection.columns, projection.rows]
                && Arc::ptr_eq(&projection.source, &tensor.data)
        })
    }

    const fn has_metadata_capacity(&self, operations: usize) -> bool {
        self.projections.len() < self.projections.capacity()
            && operations <= self.handles.capacity() - self.handles.len()
    }

    fn commit(&mut self, tensor: &Tensor, first_handle: usize) -> usize {
        let index = self.projections.len();
        self.projections.push(ResidentProjection {
            source: Arc::clone(&tensor.data),
            columns: tensor.dimensions[0],
            rows: tensor.dimensions[1],
            first_handle,
        });
        self.statistics.tensors_prepared = self.statistics.tensors_prepared.saturating_add(1);
        index
    }
}

/// Full-`K` 1-op native `HTP_TYPE_Q2_0` executor using `libggml-htp-v81.so`'s
/// signed 8-thread HVX `q2_0 x q8_1` (`HTP_MM_KERNEL_HVX_QUANT_ROW_N1`) kernel.
pub struct NativeQ2Executor<'a> {
    submitter: &'a mut dyn NativeQ2Submitter,
    maximum_columns: usize,
    maximum_rows: usize,
    maximum_operations: u32,
    maximum_owned_payload_bytes: usize,
    failed: bool,
    statistics: ProjectionStatistics,
    queue_writes_attempted: u64,
    queue_writes_completed: u64,
    residency: Option<ResidentCache>,
}

impl<'a> NativeQ2Executor<'a> {
    /// Bind one native `Q2_0` session and an admitted host-payload cap.
    #[must_use]
    pub const fn new(
        submitter: &'a mut dyn NativeQ2Submitter,
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
        }
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

    fn prepare_resident(&mut self, tensor: &Tensor, weights: &mut [u8]) -> Result<Option<usize>> {
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
                    total_blocks,
                    row_start,
                    active_rows,
                    block_start,
                    chunk_blocks,
                    chunk_weights,
                );
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
        Ok(Some(cache.commit(tensor, first_handle)))
    }

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
        let cached = self.residency.as_ref().and_then(|cache| cache.find(tensor));
        if cached.is_none() {
            for block in tensor.data.as_chunks::<34>().0 {
                scale(block)?;
            }
        }
        for &value in activation {
            finite(value)?;
        }
        let mut pending = zeroed(rows, 0.0_f32)?;
        let max_chunk_cols = columns.min(self.maximum_columns);
        let max_mapped_rows = rows.min(self.maximum_rows).next_multiple_of(32);
        let mut weights = if cached.is_none() {
            zeroed(max_mapped_rows * 36 * (max_chunk_cols / 128), 0_u8)?
        } else {
            Vec::new()
        };
        let mut chunk_output = zeroed(max_mapped_rows, 0.0_f32)?;
        let resident = if let Some(index) = cached {
            if let Some(cache) = &mut self.residency {
                cache.statistics.cache_hits = cache.statistics.cache_hits.saturating_add(1);
            }
            Some(index)
        } else {
            self.prepare_resident(tensor, &mut weights)?
        };
        let total_blocks = columns / 128;
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
                    let block_start = col_start / 128;
                    let chunk_weights = &mut weights[..mapped_rows * 36 * chunk_blocks];
                    pack_native_chunk(
                        &tensor.data,
                        total_blocks,
                        row_start,
                        active_rows,
                        block_start,
                        chunk_blocks,
                        chunk_weights,
                    );
                    self.submitter.submit(
                        active_cols,
                        mapped_rows,
                        chunk_weights,
                        act_slice,
                        out_slice,
                    )?;
                }
                operation += 1;
                for &pad_value in &out_slice[active_rows..mapped_rows] {
                    if finite(pad_value)? != 0.0 {
                        return Err(invalid("Hexagon native Q2 padding output changed"));
                    }
                }
                for (dst, &val) in pending[row_start..row_start + active_rows]
                    .iter_mut()
                    .zip(&out_slice[..active_rows])
                {
                    *dst = finite(*dst + finite(val)?)?;
                }
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
}

impl Pq2ProjectionExecutor for NativeQ2Executor<'_> {
    fn identity(&self) -> &'static str {
        if self.residency.is_some() {
            "hexagon-hvx-native-q2-resident-v1-experimental"
        } else {
            "hexagon-hvx-native-q2-v1-experimental"
        }
    }

    fn operation_capacity(&self) -> Option<u64> {
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

fn initialize_native_weights(weights: &mut [u8]) {
    for tile in weights.as_chunks_mut::<576>().0 {
        tile[..512].fill(0x55);
        tile[512..576].fill(0);
    }
}

fn pack_native_chunk(
    tensor_data: &[u8],
    total_blocks: usize,
    row_start: usize,
    active_rows: usize,
    block_start: usize,
    chunk_blocks: usize,
    weights: &mut [u8],
) {
    initialize_native_weights(weights);
    let tiles_per_row_group = chunk_blocks * 2;
    for local_row in 0..active_rows {
        let row_group = local_row / 32;
        let row_in_tile = local_row % 32;
        let row_block_base = (row_start + local_row) * total_blocks + block_start;
        for local_block in 0..chunk_blocks {
            let offset = (row_block_base + local_block) * 34;
            let original = &tensor_data[offset..offset + 34];
            let tile0 = (row_group * tiles_per_row_group + local_block * 2) * 576;
            let tile1 = tile0 + 576;
            let scale_offset0 = tile0 + 512 + row_in_tile * 2;
            let scale_offset1 = tile1 + 512 + row_in_tile * 2;
            weights[scale_offset0..scale_offset0 + 2].copy_from_slice(&original[..2]);
            weights[scale_offset1..scale_offset1 + 2].copy_from_slice(&original[..2]);
            for (chunk, word_bytes) in original[2..34].as_chunks::<4>().0.iter().enumerate() {
                let mut packed = u32::from_le_bytes(*word_bytes);
                let swap = (packed ^ (packed >> 6)) & 0x00cc_00cc;
                packed ^= swap ^ (swap << 6);
                let swap = (packed ^ (packed >> 12)) & 0x0000_f0f0;
                packed ^= swap ^ (swap << 12);
                let tile = if chunk < 4 { tile0 } else { tile1 };
                let byte = tile + (chunk % 4) * 128 + row_in_tile * 4;
                weights[byte..byte + 4].copy_from_slice(&packed.to_le_bytes());
            }
        }
    }
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
}
