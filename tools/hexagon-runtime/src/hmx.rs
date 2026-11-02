//! Original integer dots represented as eight exact binary HMX rows.

use bonsai_prism_runtime::{kernels::Pq2ProjectionExecutor, model::Tensor};
use bonsai_ternary::Q8KBlock;
use hexagon_native_rpc::{HmxQ2ProjectionLimits, HmxQ2Submitter};

use crate::guarded::{dimensions, finite, scale, zeroed};
use crate::{ProjectionStatistics, Result, invalid};

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
    failed: bool,
    statistics: ProjectionStatistics,
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
            failed: false,
            statistics: ProjectionStatistics {
                projections_completed: 0,
                operations_attempted: 0,
                operations_completed: 0,
            },
        }
    }

    /// Actual synchronous projection/submission counts; no performance claim.
    #[must_use]
    pub const fn statistics(&self) -> ProjectionStatistics {
        self.statistics
    }

    /// Any submitted failure permanently closes this executor.
    #[must_use]
    pub const fn is_faulted(&self) -> bool {
        self.failed
    }

    fn payload_bytes(&self, rows: usize) -> Option<usize> {
        let padded_rows = rows.min(self.maximum_rows).next_multiple_of(32);
        rows.checked_mul(4)?
            .checked_add(padded_rows.checked_mul(68)?)?
            .checked_add(4096)
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
        for block in tensor.data.as_chunks::<34>().0 {
            scale(block)?;
        }
        for block in activation {
            finite(block.scale)?;
        }
        let mut pending = zeroed(rows, 0.0_f32)?;
        let padded_rows = rows.min(self.maximum_rows).next_multiple_of(32);
        let mut weights = zeroed(padded_rows * 36, 0_u8)?;
        let mut dots = zeroed(padded_rows * 8, 0.0_f32)?;
        let blocks = columns / 128;
        for row_start in (0..rows).step_by(self.maximum_rows) {
            let active_rows = (rows - row_start).min(self.maximum_rows);
            let mapped_rows = active_rows.next_multiple_of(32);
            let weights = &mut weights[..mapped_rows * 36];
            let dots = &mut dots[..mapped_rows * 8];
            for block_index in 0..blocks {
                let quantized = &activation[block_index / 2];
                let start = block_index % 2 * 128;
                let planes = activation_planes(&quantized.values[start..start + 128]);
                initialize_weights(weights);
                for row in 0..active_rows {
                    let index = ((row_start + row) * blocks + block_index) * 34;
                    pack_weights(&tensor.data[index..index + 34], row, weights);
                }
                dots.fill(f32::NAN);
                self.statistics.operations_attempted =
                    self.statistics.operations_attempted.saturating_add(1);
                self.submitter.submit(weights, &planes, dots)?;
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
        self.statistics.projections_completed =
            self.statistics.projections_completed.saturating_add(1);
        *output = pending;
        Ok(())
    }
}

impl Pq2ProjectionExecutor for HmxQ2Executor<'_> {
    fn identity(&self) -> &'static str {
        "hexagon-hmx-binary-pq2-q8k-v1-experimental"
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
        dimensions(tensor).is_ok_and(|(_, rows)| {
            self.payload_bytes(rows)
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
    for lane in 0..128 {
        let tile = (row / 32 * 2 + lane / 64) * 576;
        let local = lane % 64;
        let byte = tile + local / 16 * 128 + row % 32 * 4 + local % 4;
        let shift = local / 4 % 4 * 2;
        let symbol = (original[2 + lane / 4] >> (lane % 4 * 2)) & 3;
        weights[byte] = (weights[byte] & !(3 << shift)) | symbol << shift;
    }
}

#[cfg(test)]
mod tests;
