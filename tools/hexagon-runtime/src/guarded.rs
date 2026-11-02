//! Experimental exact-PQ2 bridge to one caller-owned signed X2 queue session.
//!
//! This implements the guarded HVX compatibility path, not HMX prefill or a
//! qualified production backend. Native session limits and outer supervision
//! remain mandatory. PTQ1 and PQ2's odd-block `Q8_0` path remain CPU-owned.
#![allow(
    clippy::redundant_pub_crate,
    reason = "Share checked arithmetic helpers with the HMX sibling without exporting a new public API."
)]

use bonsai_ternary::Q8KBlock;
use hexagon_native_rpc::{GuardedQ2Submitter, Q2ProjectionLimits};

use crate::{Result, invalid};
use bonsai_prism_runtime::{kernels::Pq2ProjectionExecutor, model::Tensor};
use bonsai_ternary::half_to_float;

const PQ2: u32 = 142;

pub(crate) fn finite(value: f32) -> Result<f32> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid("nonfinite Hexagon projection value"))
    }
}

pub(crate) fn scale(bytes: &[u8]) -> Result<f32> {
    finite(half_to_float(u16::from_le_bytes([bytes[0], bytes[1]])))
}

pub(crate) fn dimensions(tensor: &Tensor) -> Result<(usize, usize)> {
    if tensor.kind != PQ2 || tensor.dimensions.len() != 2 {
        return Err(invalid("unsupported Hexagon tensor type or rank"));
    }
    let (columns, rows) = (tensor.dimensions[0], tensor.dimensions[1]);
    if columns == 0
        || rows == 0
        || !columns.is_multiple_of(256)
        || (columns / 128)
            .checked_mul(34)
            .and_then(|bytes| bytes.checked_mul(rows))
            != Some(tensor.data.len())
    {
        return Err(invalid("invalid Hexagon tensor geometry or extent"));
    }
    Ok((columns, rows))
}

const TILE_ROWS: usize = 32;
const TILE_BYTES: usize = 576;
const K_TILES: usize = 4;

/// Aggregate observations of an executor's actual synchronous submissions.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ProjectionStatistics {
    /// Complete projections accepted by this executor.
    pub projections_completed: u64,
    /// Native submissions attempted, including a failing last operation.
    pub operations_attempted: u64,
    /// Returned operations passing complete output-shape and integer checks.
    pub operations_completed: u64,
}

/// Original PQ2/`Q8_K` projections over a borrowed terminal native session.
///
/// One guarded K=256/N=2 operation preserves each original 128-weight block;
/// row tiling bounds staging rather than repacking a complete model. Original
/// `Q8_K` codes/scales and fused/nonfused reduction ordering stay CPU-owned. The
/// signed quantizer and all used shapes still need independent device parity.
/// F32 integer checks detect corruption, not plausible but incorrect results.
pub struct GuardedQ2Executor<'a> {
    submitter: &'a mut dyn GuardedQ2Submitter,
    maximum_rows: usize,
    maximum_operations: u32,
    maximum_owned_payload_bytes: usize,
    failed: bool,
    statistics: ProjectionStatistics,
}

impl<'a> GuardedQ2Executor<'a> {
    /// Bind the same checked row bound used to create the native session.
    ///
    /// `maximum_owned_payload_bytes` bounds provisional output, packed tile and
    /// returned limbs for one projection, plus guarded activation payload. It
    /// excludes caller model/prepared activation/output storage, allocator/stack
    /// overhead and native shared/DSP memory, which require separate admission.
    /// No device is opened and no allocation occurs here.
    #[must_use]
    pub const fn new(
        submitter: &'a mut dyn GuardedQ2Submitter,
        limits: Q2ProjectionLimits,
        maximum_owned_payload_bytes: usize,
    ) -> Self {
        Self {
            submitter,
            maximum_rows: limits.maximum_rows(),
            maximum_operations: limits.maximum_operations(),
            maximum_owned_payload_bytes,
            failed: false,
            statistics: ProjectionStatistics {
                projections_completed: 0,
                operations_attempted: 0,
                operations_completed: 0,
            },
        }
    }

    /// Separate accelerator counts; these establish no timing or model coverage.
    #[must_use]
    pub const fn statistics(&self) -> ProjectionStatistics {
        self.statistics
    }

    /// Native submission/arithmetic failure permanently closes this executor.
    #[must_use]
    pub const fn is_faulted(&self) -> bool {
        self.failed
    }

    fn payload_bytes(&self, rows: usize) -> Option<usize> {
        let tile_rows = rows.min(self.maximum_rows).next_multiple_of(TILE_ROWS);
        rows.checked_mul(4)?
            .checked_add(tile_rows.checked_mul(80)?)?
            .checked_add(2048)
    }

    #[allow(
        clippy::too_many_lines,
        clippy::suboptimal_flops,
        reason = "Keep exact block ordering, bounded packing and nonfused source rounding together."
    )]
    fn project_inner(
        &mut self,
        tensor: &Tensor,
        activation: &[Q8KBlock],
        fused: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        if self.failed {
            return Err(invalid(
                "guarded Hexagon executor requires terminal teardown",
            ));
        }
        let (columns, rows) = dimensions(tensor)?;
        if !self.supports(tensor) || activation.len() != columns / 256 {
            return Err(invalid("unsupported guarded Hexagon projection contract"));
        }
        // Validate every original scale and activation before submitting any work.
        for block in tensor.data.as_chunks::<34>().0 {
            scale(block)?;
        }
        for block in activation {
            finite(block.scale)?;
        }
        let mut pending = zeroed(rows, 0.0_f32)?;
        let padded_rows = rows.min(self.maximum_rows).next_multiple_of(TILE_ROWS);
        let mut weights = zeroed(padded_rows * 72, 0_u8)?;
        let mut dots = zeroed(padded_rows * 2, 0.0_f32)?;
        let blocks = columns / 128;
        for row_start in (0..rows).step_by(self.maximum_rows) {
            let active_rows = (rows - row_start).min(self.maximum_rows);
            let mapped_rows = active_rows.next_multiple_of(TILE_ROWS);
            let weights = &mut weights[..mapped_rows * 72];
            let dots = &mut dots[..mapped_rows * 2];
            for block_index in 0..blocks {
                let quantized = &activation[block_index / 2];
                let start = block_index % 2 * 128;
                let limbs = guarded_activation(&quantized.values[start..start + 128]);
                initialize_weights(weights);
                for row in 0..active_rows {
                    let index = ((row_start + row) * blocks + block_index) * 34;
                    pack_weights(&tensor.data[index..index + 34], row, weights);
                }
                dots.fill(f32::NAN);
                self.statistics.operations_attempted =
                    self.statistics.operations_attempted.saturating_add(1);
                self.submitter.submit(weights, &limbs, dots)?;
                for &value in dots.iter() {
                    if !value.is_finite() || value.fract() != 0.0 || value.abs() > 16_384.0 {
                        return Err(invalid("invalid guarded Hexagon integer result"));
                    }
                }
                for row in active_rows..mapped_rows {
                    if dots[row] != 0.0 || dots[mapped_rows + row] != 0.0 {
                        return Err(invalid("guarded Hexagon padding output changed"));
                    }
                }
                self.statistics.operations_completed =
                    self.statistics.operations_completed.saturating_add(1);
                for row in 0..active_rows {
                    let index = ((row_start + row) * blocks + block_index) * 34;
                    let integer = dots[row] + dots[mapped_rows + row];
                    // Original integer-to-F32 conversion always produces +0.
                    let integer = if integer == 0.0 { 0.0 } else { integer };
                    let factor = scale(&tensor.data[index..index + 34])? * quantized.scale;
                    let prior = pending[row_start + row];
                    pending[row_start + row] = finite(if fused {
                        factor.mul_add(integer, prior)
                    } else {
                        prior + factor * integer
                    })?;
                }
            }
        }
        self.statistics.projections_completed =
            self.statistics.projections_completed.saturating_add(1);
        *output = pending;
        Ok(())
    }
}

impl Pq2ProjectionExecutor for GuardedQ2Executor<'_> {
    fn identity(&self) -> &'static str {
        "hexagon-guarded-hvx-pq2-q8k-v1-experimental"
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
        tensor.kind == PQ2
            && tensor.dimensions.len() == 2
            && tensor.dimensions[0].is_multiple_of(256)
            && tensor.dimensions[0] > 0
            && tensor.dimensions[1] > 0
            && self
                .payload_bytes(tensor.dimensions[1])
                .is_some_and(|bytes| bytes <= self.maximum_owned_payload_bytes)
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

pub(crate) fn zeroed<T: Clone>(count: usize, value: T) -> Result<Vec<T>> {
    let mut result = Vec::new();
    result.try_reserve_exact(count)?;
    result.resize(count, value);
    Ok(result)
}

fn guarded_activation(codes: &[i8]) -> [[f32; 256]; 2] {
    let mut result = [[0.0; 256]; 2];
    for (lane, &code) in codes.iter().enumerate() {
        let destination = lane / 31 * 32 + lane % 31;
        let first = i16::from(code) / 2;
        result[0][destination] = f32::from(first);
        result[1][destination] = f32::from(i16::from(code) - first);
    }
    for limb in &mut result {
        for group in limb.as_chunks_mut::<32>().0 {
            group[31] = if group[..31].iter().sum::<f32>() >= 0.0 {
                -127.0
            } else {
                127.0
            };
        }
    }
    result
}

fn initialize_weights(weights: &mut [u8]) {
    for tile in weights.as_chunks_mut::<TILE_BYTES>().0 {
        tile[..512].fill(0x55); // Symbol one decodes to zero, including all guards/padded rows.
        for scale in tile[512..].as_chunks_mut::<2>().0 {
            *scale = [0, 0x3c];
        }
    }
}

fn pack_weights(original: &[u8], row: usize, weights: &mut [u8]) {
    for lane in 0..128 {
        let destination = lane / 31 * 32 + lane % 31;
        let tile = (row / TILE_ROWS * K_TILES + destination / 64) * TILE_BYTES;
        let local = destination % 64;
        let byte = tile + local / 16 * 128 + row % TILE_ROWS * 4 + local % 4;
        let shift = local / 4 % 4 * 2;
        let symbol = (original[2 + lane / 4] >> (lane % 4 * 2)) & 3;
        weights[byte] = (weights[byte] & !(3 << shift)) | symbol << shift;
    }
}

#[cfg(test)]
mod tests;
