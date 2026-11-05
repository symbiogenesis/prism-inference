//! Original integer dots represented as eight exact binary HMX rows.

use bonsai_prism_runtime::{kernels::Pq2ProjectionExecutor, model::Tensor};
use bonsai_ternary::Q8KBlock;
use hexagon_native_rpc::{HmxQ2ProjectionLimits, HmxQ2Submitter};

use crate::guarded::{dimensions, finite, scale, zeroed};
use crate::{ProjectionStatistics, Result, invalid};

mod residency;
pub use residency::HmxResidencyStatistics;

/// Experimental original-PQ2 executor using the signed Q2 HMX operation.
///
/// K=128/N=8 preserves one original block and represents its original `Q8_K` codes
/// as eight binary FP16-exact activation planes. The signed high plane has
/// coefficient -128. Original scale bits and CPU reduction order stay unchanged.
/// Eight planes are one token's integer decomposition, not eight prefill tokens.
/// The caller owns signed-device qualification and the terminal session bounds.
pub struct HmxQ2Executor<'a> {
    submitter: &'a mut dyn HmxQ2Submitter,
    maximum_rows: usize,
    maximum_operations: u32,
    maximum_owned_payload_bytes: usize,
    batch_four: bool,
    failed: bool,
    statistics: ProjectionStatistics,
    queue_writes_attempted: u64,
    queue_writes_completed: u64,
    residency: Option<residency::Cache>,
}

impl<'a> HmxQ2Executor<'a> {
    /// Bind one existing HMX session and a separately admitted host-payload cap.
    ///
    /// The cap charges provisional output, padded weights, plane results and
    /// activation-plane payload. Caller input/output, native/DSP memory and
    /// allocator/stack overhead require separate accounting. No allocation or
    /// device operation occurs until an admitted projection is executed.
    #[must_use]
    pub const fn new(
        submitter: &'a mut dyn HmxQ2Submitter,
        limits: HmxQ2ProjectionLimits,
        maximum_owned_payload_bytes: usize,
    ) -> Self {
        Self {
            submitter,
            maximum_rows: limits.maximum_rows(),
            maximum_operations: limits.maximum_operations(),
            maximum_owned_payload_bytes,
            batch_four: limits.batch_operations() == 4,
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

    /// Enable immutable packed-weight reuse in the caller's admitted native arena.
    ///
    /// The native session must separately reserve its arena through
    /// `HmxQ2ProjectionLimits::with_resident_weights`. This host metadata
    /// reservation is included in `maximum_owned_payload_bytes` for every
    /// projection, keeping eligibility stable as the cache fills. Equal bounded
    /// numbers of projection records and handles are reserved up front; the
    /// caller must leave sufficient room for both. Source bytes are retained by
    /// a strong immutable `Arc`, without copying the original model.
    ///
    /// Entire tensors that do not fit either remaining reservation use the
    /// ordinary streaming HMX path, decided before preparation or submission.
    /// There is no eviction and no CPU replay. Preparation after successful
    /// capacity admission must complete in full or fault the executor.
    ///
    /// # Errors
    /// Rejects repeated/late configuration, absent native capacity, invalid host
    /// bounds or metadata allocation failure before any native operation.
    pub fn with_resident_weights(mut self, maximum_cache_metadata_bytes: usize) -> Result<Self> {
        if self.residency.is_some()
            || self.statistics.operations_attempted != 0
            || self.failed
            || self.submitter.residency_capacity_bytes() == 0
            || maximum_cache_metadata_bytes > self.maximum_owned_payload_bytes
        {
            return Err(invalid("invalid HMX resident-weight configuration"));
        }
        self.residency = Some(residency::Cache::new(maximum_cache_metadata_bytes)?);
        Ok(self)
    }

    /// Native-arena preparation and host metadata observations; no speed claim.
    #[must_use]
    pub fn residency_statistics(&self) -> HmxResidencyStatistics {
        self.residency
            .as_ref()
            .map_or_else(HmxResidencyStatistics::default, |cache| cache.statistics)
    }

    /// Actual synchronous projection/submission counts; no performance claim.
    #[must_use]
    pub const fn statistics(&self) -> ProjectionStatistics {
        self.statistics
    }

    /// Actual synchronous queue writes attempted; no performance claim.
    #[must_use]
    pub const fn queue_writes_attempted(&self) -> u64 {
        self.queue_writes_attempted
    }

    /// Actual synchronous queue writes completed; no performance claim.
    #[must_use]
    pub const fn queue_writes_completed(&self) -> u64 {
        self.queue_writes_completed
    }

    /// Any submitted failure permanently closes this executor.
    #[must_use]
    pub const fn is_faulted(&self) -> bool {
        self.failed
    }

    fn payload_bytes(&self, rows: usize) -> Option<usize> {
        let padded_rows = rows.min(self.maximum_rows).next_multiple_of(32);
        let batch = if self.batch_four { 4 } else { 1 };
        rows.checked_mul(4)?
            .checked_add(padded_rows.checked_mul(68 * batch)?)?
            .checked_add(4096 * batch)
            .and_then(|bytes| {
                bytes.checked_add(self.residency.as_ref().map_or(0, |cache| cache.reservation))
            })
    }

    fn prepare_resident(&mut self, tensor: &Tensor, weights: &mut [u8]) -> Result<Option<usize>> {
        let Some(cache) = &mut self.residency else {
            return Ok(None);
        };
        let (columns, rows) = dimensions(tensor)?;
        let blocks = columns / 128;
        let tiles = rows.div_ceil(self.maximum_rows);
        let operations = blocks
            .checked_mul(tiles)
            .ok_or_else(|| invalid("HMX resident operation count overflow"))?;
        let packed_bytes = rows
            .next_multiple_of(32)
            .checked_mul(36)
            .and_then(|bytes| bytes.checked_mul(blocks))
            .ok_or_else(|| invalid("HMX resident packed extent overflow"))?;
        if !cache.has_metadata_capacity(operations)
            || packed_bytes > self.submitter.residency_capacity_bytes()
        {
            return Ok(None);
        }
        let first_handle = cache.handles.len();
        for row_start in (0..rows).step_by(self.maximum_rows) {
            let active = (rows - row_start).min(self.maximum_rows);
            let mapped = active.next_multiple_of(32);
            let weights = &mut weights[..mapped * 36];
            for block in 0..blocks {
                initialize_weights(weights);
                for row in 0..active {
                    let offset = ((row_start + row) * blocks + block) * 34;
                    pack_weights(&tensor.data[offset..offset + 34], row, weights);
                }
                // A refusal after complete-capacity admission is terminal. Any
                // prepared prefix remains owned by the same native session.
                let handle = self
                    .submitter
                    .prepare_weights(weights)?
                    .ok_or_else(|| invalid("HMX resident preparation contradicted capacity"))?;
                if handle.rows() != mapped || handle.packed_bytes() != weights.len() {
                    return Err(invalid(
                        "HMX resident preparation returned invalid geometry",
                    ));
                }
                cache.handles.push(handle);
                cache.statistics.weight_payload_bytes_prepared = cache
                    .statistics
                    .weight_payload_bytes_prepared
                    .saturating_add(u64::try_from(weights.len())?);
            }
        }
        Ok(Some(cache.commit(tensor, first_handle)))
    }

    #[allow(
        clippy::too_many_lines,
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::suboptimal_flops,
        reason = "Checked integral plane dots reconstruct an exact bounded i32 before declared fused/nonfused F32 reduction."
    )]
    fn project_inner(
        &mut self,
        tensor: &Tensor,
        activation: &[Q8KBlock],
        fused: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        if self.failed {
            return Err(invalid("HMX Hexagon executor requires terminal teardown"));
        }
        let (columns, rows) = dimensions(tensor)?;
        if !self.supports(tensor) || activation.len() != columns / 256 {
            return Err(invalid("unsupported HMX Hexagon projection contract"));
        }
        let cached = self.residency.as_ref().and_then(|cache| cache.find(tensor));
        if cached.is_none() {
            for block in tensor.data.as_chunks::<34>().0 {
                scale(block)?;
            }
        }
        for block in activation {
            finite(block.scale)?;
        }
        let mut pending = zeroed(rows, 0.0_f32)?;
        let padded_rows = rows.min(self.maximum_rows).next_multiple_of(32);
        let batch = if self.batch_four { 4 } else { 1 };
        let mut weights = if cached.is_none() {
            zeroed(padded_rows * 36 * batch, 0_u8)?
        } else {
            Vec::new()
        };
        let mut dots = zeroed(padded_rows * 8 * batch, 0.0_f32)?;
        let resident = if let Some(index) = cached {
            if let Some(cache) = &mut self.residency {
                cache.statistics.cache_hits = cache.statistics.cache_hits.saturating_add(1);
            }
            Some(index)
        } else {
            self.prepare_resident(tensor, &mut weights)?
        };
        let blocks = columns / 128;
        let mut operation = 0;
        for row_start in (0..rows).step_by(self.maximum_rows) {
            let active_rows = (rows - row_start).min(self.maximum_rows);
            let mapped_rows = active_rows.next_multiple_of(32);
            if self.batch_four {
                let batch_dots = &mut dots[..4 * mapped_rows * 8];
                for block_base in (0..blocks).step_by(4) {
                    let planes: [[[f32; 128]; 8]; 4] = std::array::from_fn(|member| {
                        let block_index = block_base + member;
                        let quantized = &activation[block_index / 2];
                        let start = block_index % 2 * 128;
                        activation_planes(&quantized.values[start..start + 128])
                    });
                    batch_dots.fill(f32::NAN);
                    self.statistics.operations_attempted =
                        self.statistics.operations_attempted.saturating_add(4);
                    self.queue_writes_attempted = self.queue_writes_attempted.saturating_add(1);
                    {
                        let (out0, rest) = batch_dots.split_at_mut(mapped_rows * 8);
                        let (out1, rest) = rest.split_at_mut(mapped_rows * 8);
                        let (out2, out3) = rest.split_at_mut(mapped_rows * 8);
                        if let Some(index) = resident {
                            let cache = self
                                .residency
                                .as_ref()
                                .ok_or_else(|| invalid("HMX resident cache is unavailable"))?;
                            let base_handle = cache.projections[index].first_handle + operation;
                            let handles = [
                                &cache.handles[base_handle],
                                &cache.handles[base_handle + 1],
                                &cache.handles[base_handle + 2],
                                &cache.handles[base_handle + 3],
                            ];
                            self.submitter.submit_resident_batch4(
                                handles,
                                [&planes[0], &planes[1], &planes[2], &planes[3]],
                                [out0, out1, out2, out3],
                            )?;
                        } else {
                            let weight_stride = mapped_rows * 36;
                            let batch_weights = &mut weights[..4 * weight_stride];
                            for member in 0..4 {
                                let block_index = block_base + member;
                                let member_weights = &mut batch_weights
                                    [member * weight_stride..(member + 1) * weight_stride];
                                initialize_weights(member_weights);
                                for row in 0..active_rows {
                                    let offset = ((row_start + row) * blocks + block_index) * 34;
                                    pack_weights(
                                        &tensor.data[offset..offset + 34],
                                        row,
                                        member_weights,
                                    );
                                }
                            }
                            let (w0, rest) = batch_weights.split_at(weight_stride);
                            let (w1, rest) = rest.split_at(weight_stride);
                            let (w2, w3) = rest.split_at(weight_stride);
                            self.submitter.submit_batch4(
                                [w0, w1, w2, w3],
                                [&planes[0], &planes[1], &planes[2], &planes[3]],
                                [out0, out1, out2, out3],
                            )?;
                        }
                    }
                    operation += 4;
                    for &value in batch_dots.iter() {
                        if !value.is_finite()
                            || value.fract() != 0.0
                            || !(-128.0..=256.0).contains(&value)
                        {
                            return Err(invalid("invalid Hexagon HMX plane result"));
                        }
                    }
                    for member in 0..4 {
                        let member_dots =
                            &batch_dots[member * mapped_rows * 8..(member + 1) * mapped_rows * 8];
                        for row in active_rows..mapped_rows {
                            if (0..8).any(|plane| member_dots[plane * mapped_rows + row] != 0.0) {
                                return Err(invalid("Hexagon HMX padding output changed"));
                            }
                        }
                    }
                    for member in 0..4 {
                        let block_index = block_base + member;
                        let quantized = &activation[block_index / 2];
                        let member_dots =
                            &batch_dots[member * mapped_rows * 8..(member + 1) * mapped_rows * 8];
                        for row in 0..active_rows {
                            let mut integer = 0_i32;
                            for plane in 0..8 {
                                let coefficient = if plane == 7 { -128 } else { 1 << plane };
                                integer +=
                                    coefficient * member_dots[plane * mapped_rows + row] as i32;
                            }
                            if !(-32768..=32768).contains(&integer) {
                                return Err(invalid(
                                    "Hexagon HMX reconstructed dot is out of range",
                                ));
                            }
                            let index = ((row_start + row) * blocks + block_index) * 34;
                            let factor = scale(&tensor.data[index..index + 34])? * quantized.scale;
                            let prior = pending[row_start + row];
                            pending[row_start + row] = finite(if fused {
                                factor.mul_add(integer as f32, prior)
                            } else {
                                prior + factor * integer as f32
                            })?;
                        }
                    }
                    self.statistics.operations_completed =
                        self.statistics.operations_completed.saturating_add(4);
                    self.queue_writes_completed = self.queue_writes_completed.saturating_add(1);
                }
            } else {
                let dots = &mut dots[..mapped_rows * 8];
                for block_index in 0..blocks {
                    let quantized = &activation[block_index / 2];
                    let start = block_index % 2 * 128;
                    let planes = activation_planes(&quantized.values[start..start + 128]);
                    dots.fill(f32::NAN);
                    self.statistics.operations_attempted =
                        self.statistics.operations_attempted.saturating_add(1);
                    self.queue_writes_attempted = self.queue_writes_attempted.saturating_add(1);
                    if let Some(index) = resident {
                        let cache = self
                            .residency
                            .as_ref()
                            .ok_or_else(|| invalid("HMX resident cache is unavailable"))?;
                        let handle =
                            &cache.handles[cache.projections[index].first_handle + operation];
                        self.submitter.submit_resident(handle, &planes, dots)?;
                    } else {
                        let weights = &mut weights[..mapped_rows * 36];
                        initialize_weights(weights);
                        for row in 0..active_rows {
                            let index = ((row_start + row) * blocks + block_index) * 34;
                            pack_weights(&tensor.data[index..index + 34], row, weights);
                        }
                        self.submitter.submit(weights, &planes, dots)?;
                    }
                    operation += 1;
                    for &value in dots.iter() {
                        if !value.is_finite()
                            || value.fract() != 0.0
                            || !(-128.0..=256.0).contains(&value)
                        {
                            return Err(invalid("invalid Hexagon HMX plane result"));
                        }
                    }
                    for row in active_rows..mapped_rows {
                        if (0..8).any(|plane| dots[plane * mapped_rows + row] != 0.0) {
                            return Err(invalid("Hexagon HMX padding output changed"));
                        }
                    }
                    self.statistics.operations_completed =
                        self.statistics.operations_completed.saturating_add(1);
                    self.queue_writes_completed = self.queue_writes_completed.saturating_add(1);
                    for row in 0..active_rows {
                        let mut integer = 0_i32;
                        for plane in 0..8 {
                            let coefficient = if plane == 7 { -128 } else { 1 << plane };
                            integer += coefficient * dots[plane * mapped_rows + row] as i32;
                        }
                        if !(-32768..=32768).contains(&integer) {
                            return Err(invalid("Hexagon HMX reconstructed dot is out of range"));
                        }
                        let index = ((row_start + row) * blocks + block_index) * 34;
                        let factor = scale(&tensor.data[index..index + 34])? * quantized.scale;
                        let prior = pending[row_start + row];
                        pending[row_start + row] = finite(if fused {
                            factor.mul_add(integer as f32, prior)
                        } else {
                            prior + factor * integer as f32
                        })?;
                    }
                }
            }
        }
        self.statistics.projections_completed =
            self.statistics.projections_completed.saturating_add(1);
        *output = pending;
        Ok(())
    }
}

impl Pq2ProjectionExecutor for HmxQ2Executor<'_> {
    fn identity(&self) -> &'static str {
        match (self.residency.is_some(), self.batch_four) {
            (false, false) => "hexagon-hmx-binary-pq2-q8k-v1-experimental",
            (true, false) => "hexagon-hmx-binary-pq2-q8k-resident-v1-experimental",
            (false, true) => "hexagon-hmx-binary-pq2-q8k-batch4-v1-experimental",
            (true, true) => "hexagon-hmx-binary-pq2-q8k-resident-batch4-v1-experimental",
        }
    }
    fn operation_capacity(&self) -> Option<u64> {
        let local =
            u64::from(self.maximum_operations).saturating_sub(self.statistics.operations_attempted);
        Some(local.min(self.submitter.remaining_operations().unwrap_or(local)))
    }
    fn operation_count(&self, tensor: &Tensor) -> Option<u64> {
        let (columns, rows) = dimensions(tensor).ok()?;
        u64::try_from((columns / 128).checked_mul(rows.div_ceil(self.maximum_rows))?).ok()
    }
    fn supports(&self, tensor: &Tensor) -> bool {
        dimensions(tensor).is_ok_and(|(columns, rows)| {
            (!self.batch_four || (columns / 128).is_multiple_of(4))
                && self
                    .payload_bytes(rows)
                    .is_some_and(|bytes| bytes <= self.maximum_owned_payload_bytes)
        })
    }
    fn project(
        &mut self,
        tensor: &Tensor,
        activation: &[Q8KBlock],
        fused: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        output.clear();
        let result = self.project_inner(tensor, activation, fused, output);
        if result.is_err() {
            self.failed = true;
            output.clear();
        }
        result
    }
}

fn activation_planes(codes: &[i8]) -> [[f32; 128]; 8] {
    std::array::from_fn(|bit| {
        std::array::from_fn(|lane| f32::from((codes[lane].cast_unsigned() >> bit) & 1))
    })
}

fn initialize_weights(weights: &mut [u8]) {
    for tile in weights.as_chunks_mut::<576>().0 {
        tile[..512].fill(0x55);
        for scale in tile[512..].as_chunks_mut::<2>().0 {
            *scale = [0, 0x3c];
        }
    }
}

fn pack_weights(original: &[u8], row: usize, weights: &mut [u8]) {
    for (chunk, original) in original[2..].as_chunks::<4>().0.iter().enumerate() {
        // Each original byte holds four consecutive two-bit symbols. The signed
        // tile wants the four equally positioned symbols in one byte: transpose
        // the 4x4 two-bit matrix with two nonoverlapping bit-pair swaps. Writing
        // complete bytes avoids 128 individual read/modify/write lane updates.
        let mut packed = u32::from_le_bytes(*original);
        let swap = (packed ^ (packed >> 6)) & 0x00cc_00cc;
        packed ^= swap ^ (swap << 6);
        let swap = (packed ^ (packed >> 12)) & 0x0000_f0f0;
        packed ^= swap ^ (swap << 12);
        let tile = (row / 32 * 2 + chunk / 4) * 576;
        let byte = tile + chunk % 4 * 128 + row % 32 * 4;
        weights[byte..byte + 4].copy_from_slice(&packed.to_le_bytes());
    }
}

#[cfg(test)]
mod tests;
