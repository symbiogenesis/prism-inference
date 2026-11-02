//! Exact per-32 scales and DOTPROD lanes over the bounded signed HMX primitive.

use bonsai_prism_runtime::{
    kernels::{Pq2ProjectionExecutor, Q8Block},
    model::Tensor,
};
use bonsai_ternary::{Q8KBlock, decode_ptq_trits};
use hexagon_native_rpc::{HmxQ2ProjectionLimits, HmxQ2Submitter};

use crate::guarded::{finite, scale, zeroed};
use crate::{ProjectionStatistics, Result, invalid};

const PQ2: u32 = 142;
const PTQ1: u32 = 143;

fn geometry(tensor: &Tensor) -> Option<(usize, usize, usize)> {
    if tensor.dimensions.len() != 2 {
        return None;
    }
    let (columns, rows) = (tensor.dimensions[0], tensor.dimensions[1]);
    let bytes = match tensor.kind {
        PQ2 if !columns.is_multiple_of(256) => 34,
        PTQ1 => 28,
        _ => return None,
    };
    (columns > 0
        && rows > 0
        && columns.is_multiple_of(128)
        && (columns / 128).checked_mul(rows)?.checked_mul(bytes)? == tensor.data.len())
    .then_some((columns, rows, bytes))
}

fn weight_scale(block: &[u8], kind: u32) -> Result<f32> {
    scale(if kind == PTQ1 { &block[26..] } else { block })
}

/// Experimental PTQ1 and odd-block PQ2 executor retaining original `Q8_0`.
///
/// Four sparse output rows represent the four native integer DOTPROD lanes of
/// one original row. Each original 128-weight block requires four K128/N8 HMX
/// submissions, one per original 32-code activation scale. Eight binary planes
/// preserve every signed code, while original float reductions remain on CPU.
/// This compatibility composition establishes neither efficient placement nor
/// signed-device arithmetic, actual-model parity or a daily service contract.
pub struct HmxQ8Executor<'a> {
    submitter: &'a mut dyn HmxQ2Submitter,
    maximum_rows: usize,
    maximum_operations: u32,
    maximum_owned_payload_bytes: usize,
    failed: bool,
    statistics: ProjectionStatistics,
}

impl<'a> HmxQ8Executor<'a> {
    /// Bind a fresh or independently capacity-adjusted terminal HMX session.
    ///
    /// The payload cap includes provisional lane/output storage, packed tiles,
    /// returned planes, block sums and activation planes. Native/DSP storage,
    /// caller/model input and allocator overhead require separate admission.
    #[must_use]
    pub const fn new(
        submitter: &'a mut dyn HmxQ2Submitter,
        limits: HmxQ2ProjectionLimits,
        maximum_owned_payload_bytes: usize,
    ) -> Self {
        Self {
            submitter,
            maximum_rows: limits.maximum_rows() / 4,
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

    /// Actual synchronous work counts; these are not performance measurements.
    #[must_use]
    pub const fn statistics(&self) -> ProjectionStatistics {
        self.statistics
    }

    /// A failure permanently closes this executor until terminal teardown.
    #[must_use]
    pub const fn is_faulted(&self) -> bool {
        self.failed
    }

    fn payload_bytes(&self, rows: usize) -> Option<usize> {
        let active = rows.min(self.maximum_rows);
        let mapped = active.checked_mul(4)?.next_multiple_of(32);
        rows.checked_mul(20)?
            .checked_add(mapped.checked_mul(68)?)?
            .checked_add(active.checked_mul(4)?)?
            .checked_add(4096)
    }

    #[allow(
        clippy::too_many_lines,
        clippy::suboptimal_flops,
        reason = "Keep prevalidation, complete tile verification and original float reduction order together."
    )]
    fn project_inner(
        &mut self,
        tensor: &Tensor,
        activation: &[Q8Block],
        pinned: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        if self.failed {
            return Err(invalid("Q8_0 HMX executor requires terminal teardown"));
        }
        let (columns, rows, block_bytes) =
            geometry(tensor).ok_or_else(|| invalid("unsupported Q8_0 HMX tensor"))?;
        if !self.supports_q8(tensor) || activation.len() != columns / 32 {
            return Err(invalid("unsupported Q8_0 HMX projection contract"));
        }
        for block in tensor.data.chunks_exact(block_bytes) {
            weight_scale(block, tensor.kind)?;
        }
        for block in activation {
            finite(block.scale)?;
        }
        let mut pending = zeroed(rows, [0.0_f32; 4])?;
        let tile_rows = rows.min(self.maximum_rows);
        let mapped = (tile_rows * 4).next_multiple_of(32);
        let mut weights = zeroed(mapped * 36, 0_u8)?;
        let mut dots = zeroed(mapped * 8, 0.0_f32)?;
        let mut block_sums = zeroed(tile_rows, 0.0_f32)?;
        let blocks = columns / 128;
        let native_lanes = pinned && tensor.kind == PTQ1;
        for row_start in (0..rows).step_by(self.maximum_rows) {
            let active = (rows - row_start).min(self.maximum_rows);
            let mapped = (active * 4).next_multiple_of(32);
            let weights = &mut weights[..mapped * 36];
            let dots = &mut dots[..mapped * 8];
            for block_index in 0..blocks {
                block_sums.fill(0.0);
                let codes = std::array::from_fn(|index| {
                    activation[block_index * 4 + index / 32].values[index % 32]
                });
                let planes = planes(&codes);
                for group in 0..4 {
                    initialize_weights(weights);
                    for row in 0..active {
                        let offset = ((row_start + row) * blocks + block_index) * block_bytes;
                        pack_lanes(
                            &tensor.data[offset..offset + block_bytes],
                            tensor.kind,
                            group,
                            row,
                            weights,
                        )?;
                    }
                    dots.fill(f32::NAN);
                    self.statistics.operations_attempted =
                        self.statistics.operations_attempted.saturating_add(1);
                    self.submitter.submit(weights, &planes, dots)?;
                    for &value in dots.iter() {
                        if !value.is_finite()
                            || value.fract() != 0.0
                            || !(-8.0..=16.0).contains(&value)
                        {
                            return Err(invalid("invalid Q8_0 HMX lane plane"));
                        }
                    }
                    for row in active * 4..mapped {
                        if (0..8).any(|plane| dots[plane * mapped + row] != 0.0) {
                            return Err(invalid("Q8_0 HMX padding output changed"));
                        }
                    }
                    for row in 0..active {
                        let mut integers = [0.0_f32; 4];
                        for (lane, value) in integers.iter_mut().enumerate() {
                            *value = integer(dots, mapped, row * 4 + lane)?;
                        }
                        let offset = ((row_start + row) * blocks + block_index) * block_bytes;
                        let d =
                            weight_scale(&tensor.data[offset..offset + block_bytes], tensor.kind)?;
                        let activation_scale = activation[block_index * 4 + group].scale;
                        if native_lanes {
                            let factor = d * activation_scale;
                            for (prior, value) in pending[row_start + row].iter_mut().zip(integers)
                            {
                                *prior = finite(value.mul_add(factor, *prior))?;
                            }
                        } else {
                            let sum = integers.into_iter().sum::<f32>();
                            block_sums[row] += activation_scale * sum;
                            if group == 3 {
                                pending[row_start + row][0] =
                                    finite(pending[row_start + row][0] + d * block_sums[row])?;
                            }
                        }
                    }
                    self.statistics.operations_completed =
                        self.statistics.operations_completed.saturating_add(1);
                }
            }
        }
        output.try_reserve_exact(rows)?;
        for lanes in pending {
            output.push(finite(if native_lanes {
                (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
            } else {
                lanes[0]
            })?);
        }
        self.statistics.projections_completed =
            self.statistics.projections_completed.saturating_add(1);
        Ok(())
    }
}

impl Pq2ProjectionExecutor for HmxQ8Executor<'_> {
    fn identity(&self) -> &'static str {
        "hexagon-hmx-q8-zero-native-lanes-v1-experimental"
    }

    fn operation_capacity(&self) -> Option<u64> {
        let local =
            u64::from(self.maximum_operations).saturating_sub(self.statistics.operations_attempted);
        Some(local.min(self.submitter.remaining_operations().unwrap_or(local)))
    }

    fn operation_count(&self, tensor: &Tensor) -> Option<u64> {
        let (columns, rows, _) = geometry(tensor)?;
        u64::try_from((columns / 32).checked_mul(rows.div_ceil(self.maximum_rows))?).ok()
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
        Err(invalid("Q8_0 HMX executor does not accept Q8_K"))
    }

    fn supports_q8(&self, tensor: &Tensor) -> bool {
        geometry(tensor).is_some_and(|(_, rows, _)| {
            self.payload_bytes(rows)
                .is_some_and(|bytes| bytes <= self.maximum_owned_payload_bytes)
        })
    }

    fn project_q8(
        &mut self,
        tensor: &Tensor,
        activation: &[Q8Block],
        pinned: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        output.clear();
        let result = self.project_inner(tensor, activation, pinned, output);
        if result.is_err() {
            self.failed = true;
            output.clear();
        }
        result
    }
}

fn planes(codes: &[i8; 128]) -> [[f32; 128]; 8] {
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

fn pack_lanes(block: &[u8], kind: u32, group: usize, row: usize, output: &mut [u8]) -> Result<()> {
    let trits = if kind == PTQ1 {
        decode_ptq_trits(block.try_into()?).map(|trit| (trit + 1).to_le_bytes()[0])
    } else {
        std::array::from_fn(|lane| (block[2 + lane / 4] >> (lane % 4 * 2)) & 3)
    };
    for lane in 0..4 {
        for source in [group * 32 + lane * 4, group * 32 + lane * 4 + 16] {
            for index in source..source + 4 {
                let destination_row = row * 4 + lane;
                let tile = (destination_row / 32 * 2 + index / 64) * 576;
                let local = index % 64;
                let byte = tile + local / 16 * 128 + destination_row % 32 * 4 + local % 4;
                let shift = local / 4 % 4 * 2;
                output[byte] = (output[byte] & !(3 << shift)) | trits[index] << shift;
            }
        }
    }
    Ok(())
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "Validated integral planes reconstruct a bounded exact i32 lane dot."
)]
fn integer(dots: &[f32], rows: usize, row: usize) -> Result<f32> {
    let value = (0..8)
        .map(|plane| {
            let coefficient = if plane == 7 { -128 } else { 1 << plane };
            coefficient * dots[plane * rows + row] as i32
        })
        .sum::<i32>();
    if !(-2048..=2048).contains(&value) {
        return Err(invalid("Q8_0 HMX lane integer out of range"));
    }
    Ok(value as f32)
}

#[cfg(test)]
mod tests;
