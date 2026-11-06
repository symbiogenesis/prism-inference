//! Bounded multi-token projection with a single decoded packed weight block.
//!
//! The block/row loop surrounds the token loop. Unlike repeated matvec calls,
//! every PQ2/PTQ1 block is decoded once for the entire tile. Each token retains
//! its original activation quantizer, ordered block accumulation and pinned
//! PTQ1 four-lane reduction. No decoded matrix survives this call.

#![allow(
    clippy::redundant_pub_crate,
    reason = "Keep graph-consumed tile types crate scoped inside the private kernel module."
)]

use super::{
    ActivationContract, MatvecBackend, MatvecScratch, PQ2, PTQ1, RowParallelism, dimensions,
    finite, integer_float, matvec_into, scale,
};
use crate::{Result, invalid, model::Tensor};

#[path = "prefill_dot.rs"]
mod dot;

pub(crate) const MAX_TILE_TOKENS: usize = 32;
pub(crate) const ROW_TASK_SCRATCH_BYTES: usize = 16 * 1024;

#[derive(Default)]
pub(crate) struct ProjectionTile {
    backend: MatvecBackend,
    preparations: Vec<MatvecScratch>,
    row_parallelism: RowParallelism,
    pub(crate) packed_projections: u64,
    pub(crate) decoded_blocks: u64,
}

impl ProjectionTile {
    pub(crate) fn with_backend(backend: MatvecBackend) -> Self {
        Self {
            backend,
            ..Self::default()
        }
    }

    #[cfg_attr(
        not(feature = "parallel-rows"),
        allow(
            clippy::missing_const_for_fn,
            reason = "The parallel-rows build drops the replaced pool owner; retain one non-const signature across feature sets."
        )
    )]
    pub(crate) fn set_row_parallelism(&mut self, parallelism: RowParallelism) {
        self.row_parallelism = parallelism;
    }

    #[cfg(test)]
    pub(crate) fn project(
        &mut self,
        tensor: &Tensor,
        inputs: &[Vec<f32>],
        outputs: &mut [Vec<f32>],
        pinned: bool,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        self.project_with_previous(None, tensor, inputs, outputs, pinned, check)
    }

    #[allow(
        clippy::suboptimal_flops,
        clippy::too_many_lines,
        reason = "Retain the separately rounded source arithmetic and auditable packed block order."
    )]
    pub(crate) fn project_with_previous(
        &mut self,
        previous: Option<&Tensor>,
        tensor: &Tensor,
        inputs: &[Vec<f32>],
        outputs: &mut [Vec<f32>],
        pinned: bool,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        for output in outputs.iter_mut() {
            output.clear();
        }
        let result = (|| {
            let (columns, rows, block_bytes) = dimensions(tensor)?;
            if inputs.is_empty() || inputs.len() > MAX_TILE_TOKENS || outputs.len() != inputs.len()
            {
                return Err(invalid("invalid bounded projection tile"));
            }
            let reuse_prepared = matches!(tensor.kind, PQ2 | PTQ1)
                && self.preparations.len() >= inputs.len()
                && previous.is_some_and(|prev| {
                    matches!(prev.kind, PQ2 | PTQ1)
                        && dimensions(prev).is_ok_and(|(prev_cols, _, _)| {
                            prev_cols == columns
                                && ActivationContract::for_tensor(prev, prev_cols, pinned)
                                    == ActivationContract::for_tensor(tensor, columns, pinned)
                        })
                });
            while self.preparations.len() < inputs.len() {
                self.preparations.try_reserve_exact(1)?;
                self.preparations.push(if pinned {
                    MatvecScratch::with_pinned_backend(self.backend)
                } else {
                    MatvecScratch::with_backend(self.backend)
                });
            }
            for ((scratch, input), output) in self
                .preparations
                .iter_mut()
                .zip(inputs)
                .zip(outputs.iter_mut())
            {
                // Preparation also validates every input before publishing any
                // tile output, including unsupported-format fallback inputs.
                check()?;
                scratch.pinned_accumulation = pinned;
                if !reuse_prepared {
                    scratch.prepare_input(tensor, input)?;
                }
                output.try_reserve_exact(rows)?;
            }
            if !matches!(tensor.kind, PQ2 | PTQ1) {
                for ((input, output), scratch) in inputs
                    .iter()
                    .zip(outputs.iter_mut())
                    .zip(self.preparations.iter_mut())
                {
                    check()?;
                    matvec_into(tensor, input, output, scratch)?;
                }
                return Ok(());
            }
            self.packed_projections = self.packed_projections.saturating_add(1);
            let backend = self.backend;
            let preparations = &self.preparations[..inputs.len()];
            self.row_parallelism.fill_tile(
                &tensor.data,
                block_bytes,
                outputs,
                &mut self.decoded_blocks,
                check,
                |row, decoded_blocks| {
                    let mut sums = [0.0_f32; MAX_TILE_TOKENS];
                    let mut lanes = [[0.0_f32; 4]; MAX_TILE_TOKENS];
                    let bytes = if tensor.kind == PQ2 { 34 } else { 28 };
                    for (index, block) in row.chunks_exact(bytes).enumerate() {
                        let weight_scale = if tensor.kind == PQ2 {
                            scale(block)?
                        } else {
                            scale(&block[26..])?
                        };
                        // Prepare one SIMD/scalar block for all tokens. This stays
                        // stack-scoped and never expands or retains a matrix.
                        let weights = if tensor.kind == PQ2 {
                            dot::Block::from_pq2(block[2..34].try_into()?, backend)
                        } else {
                            dot::Block::from_ptq1(block.try_into()?, backend)?
                        };
                        *decoded_blocks = decoded_blocks.saturating_add(1);
                        if tensor.kind == PQ2 && columns.is_multiple_of(256) {
                            let start = (index % 2) * 128;
                            let (chunks8, rem8) = preparations.as_chunks::<8>();
                            for (chunk_idx, chunk) in chunks8.iter().enumerate() {
                                let base = chunk_idx * 8;
                                let a0 = &chunk[0].q8k[index / 2];
                                let a1 = &chunk[1].q8k[index / 2];
                                let a2 = &chunk[2].q8k[index / 2];
                                let a3 = &chunk[3].q8k[index / 2];
                                let a4 = &chunk[4].q8k[index / 2];
                                let a5 = &chunk[5].q8k[index / 2];
                                let a6 = &chunk[6].q8k[index / 2];
                                let a7 = &chunk[7].q8k[index / 2];
                                let dots = weights.dot128_eight([
                                    a0.values[start..start + 128].try_into()?,
                                    a1.values[start..start + 128].try_into()?,
                                    a2.values[start..start + 128].try_into()?,
                                    a3.values[start..start + 128].try_into()?,
                                    a4.values[start..start + 128].try_into()?,
                                    a5.values[start..start + 128].try_into()?,
                                    a6.values[start..start + 128].try_into()?,
                                    a7.values[start..start + 128].try_into()?,
                                ]);
                                for (offset, (act, dot_int)) in [a0, a1, a2, a3, a4, a5, a6, a7]
                                    .into_iter()
                                    .zip(dots)
                                    .enumerate()
                                {
                                    let d = weight_scale * act.scale;
                                    let dot = integer_float(dot_int);
                                    let token = base + offset;
                                    sums[token] = if pinned {
                                        d.mul_add(dot, sums[token])
                                    } else {
                                        sums[token] + d * dot
                                    };
                                }
                            }
                            let base4 = chunks8.len() * 8;
                            let (chunks4, rem) = rem8.as_chunks::<4>();
                            for (chunk_idx, chunk) in chunks4.iter().enumerate() {
                                let base = base4 + chunk_idx * 4;
                                let a0 = &chunk[0].q8k[index / 2];
                                let a1 = &chunk[1].q8k[index / 2];
                                let a2 = &chunk[2].q8k[index / 2];
                                let a3 = &chunk[3].q8k[index / 2];
                                let dots = weights.dot128_four([
                                    a0.values[start..start + 128].try_into()?,
                                    a1.values[start..start + 128].try_into()?,
                                    a2.values[start..start + 128].try_into()?,
                                    a3.values[start..start + 128].try_into()?,
                                ]);
                                for (offset, (act, dot_int)) in
                                    [a0, a1, a2, a3].into_iter().zip(dots).enumerate()
                                {
                                    let d = weight_scale * act.scale;
                                    let dot = integer_float(dot_int);
                                    let token = base + offset;
                                    sums[token] = if pinned {
                                        d.mul_add(dot, sums[token])
                                    } else {
                                        sums[token] + d * dot
                                    };
                                }
                            }
                            let rem_base = base4 + chunks4.len() * 4;
                            for (offset, scratch) in rem.iter().enumerate() {
                                let activation = &scratch.q8k[index / 2];
                                let d = weight_scale * activation.scale;
                                let dot = integer_float(
                                    weights
                                        .dot128(activation.values[start..start + 128].try_into()?),
                                );
                                let token = rem_base + offset;
                                sums[token] = if pinned {
                                    d.mul_add(dot, sums[token])
                                } else {
                                    sums[token] + d * dot
                                };
                            }
                        } else if tensor.kind == PTQ1 && pinned {
                            let (chunks4, rem) = preparations.as_chunks::<4>();
                            for (chunk_idx, chunk) in chunks4.iter().enumerate() {
                                let base = chunk_idx * 4;
                                for group in 0..4 {
                                    let a0 = &chunk[0].q8[index * 4 + group];
                                    let a1 = &chunk[1].q8[index * 4 + group];
                                    let a2 = &chunk[2].q8[index * 4 + group];
                                    let a3 = &chunk[3].q8[index * 4 + group];
                                    let group_lanes = weights.lanes32_four(
                                        group,
                                        [&a0.values, &a1.values, &a2.values, &a3.values],
                                    );
                                    for (offset, (act, integers)) in
                                        [a0, a1, a2, a3].into_iter().zip(group_lanes).enumerate()
                                    {
                                        let d = weight_scale * act.scale;
                                        for (lane, integer) in
                                            lanes[base + offset].iter_mut().zip(integers)
                                        {
                                            *lane = integer_float(integer).mul_add(d, *lane);
                                        }
                                    }
                                }
                            }
                            let rem_base = chunks4.len() * 4;
                            for (offset, scratch) in rem.iter().enumerate() {
                                let token = rem_base + offset;
                                let activations = &scratch.q8[index * 4..index * 4 + 4];
                                for (group, activation) in activations.iter().enumerate() {
                                    let d = weight_scale * activation.scale;
                                    for (lane, integer) in lanes[token]
                                        .iter_mut()
                                        .zip(weights.lanes32(group, &activation.values))
                                    {
                                        *lane = integer_float(integer).mul_add(d, *lane);
                                    }
                                }
                            }
                        } else {
                            let (chunks4, rem) = preparations.as_chunks::<4>();
                            for (chunk_idx, chunk) in chunks4.iter().enumerate() {
                                let base = chunk_idx * 4;
                                let mut chunk_sums = [0.0_f32; 4];
                                for group in 0..4 {
                                    let a0 = &chunk[0].q8[index * 4 + group];
                                    let a1 = &chunk[1].q8[index * 4 + group];
                                    let a2 = &chunk[2].q8[index * 4 + group];
                                    let a3 = &chunk[3].q8[index * 4 + group];
                                    let dots = weights.dot32_four(
                                        group,
                                        [&a0.values, &a1.values, &a2.values, &a3.values],
                                    );
                                    for (offset, (act, dot_int)) in
                                        [a0, a1, a2, a3].into_iter().zip(dots).enumerate()
                                    {
                                        chunk_sums[offset] += act.scale * integer_float(dot_int);
                                    }
                                }
                                for (offset, sum) in chunk_sums.into_iter().enumerate() {
                                    sums[base + offset] += weight_scale * sum;
                                }
                            }
                            let rem_base = chunks4.len() * 4;
                            for (offset, scratch) in rem.iter().enumerate() {
                                let token = rem_base + offset;
                                let activations = &scratch.q8[index * 4..index * 4 + 4];
                                let mut sum = 0.0;
                                for (group, activation) in activations.iter().enumerate() {
                                    sum += activation.scale
                                        * integer_float(weights.dot32(group, &activation.values));
                                }
                                sums[token] += weight_scale * sum;
                            }
                        }
                    }
                    for (sum, lanes) in sums.iter_mut().zip(lanes).take(inputs.len()) {
                        *sum = finite(if tensor.kind == PTQ1 && pinned {
                            (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
                        } else {
                            *sum
                        })?;
                    }
                    Ok(sums)
                },
            )
        })();
        if result.is_err() {
            for output in outputs {
                output.clear();
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::sync::Arc;

    use super::*;

    #[cfg(feature = "parallel-rows")]
    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "Keep exact format/backend/tile parity and malformed-row recovery in one control."
    )]
    fn tiled_parallel_projection_matches_serial_and_recovers() -> Result<()> {
        for workers in [2, 4] {
            let parallelism = RowParallelism::new(workers)?;
            for (kind, columns) in [(PQ2, 256), (PQ2, 384), (PTQ1, 256)] {
                let mut packed = [if kind == PTQ1 { 0x66 } else { 0xe4 }; 34];
                packed[..2].copy_from_slice(&0x3555_u16.to_le_bytes());
                let block = if kind == PTQ1 {
                    bonsai_ternary::pack_ptq(&packed)?.to_vec()
                } else {
                    packed.to_vec()
                };
                let mut tensor = Tensor {
                    name: "parallel-packed-tile".into(),
                    dimensions: vec![columns, 7],
                    kind,
                    data: Arc::from(block.repeat(columns / 128 * 7)),
                };
                for count in [8, 16, 32] {
                    let inputs: Vec<Vec<f32>> = (0..count)
                        .map(|token| {
                            (0..columns)
                                .map(|index| {
                                    [1.5, -128.0, 0.0, -0.0, 63.5, -3.25][(index + token) % 6]
                                })
                                .collect()
                        })
                        .collect();
                    for backend in [
                        MatvecBackend::Scalar,
                        MatvecBackend::Auto,
                        MatvecBackend::ExperimentalNeon,
                    ] {
                        let mut tile = ProjectionTile::with_backend(backend);
                        tile.set_row_parallelism(parallelism.clone());
                        for pinned in [false, true] {
                            let mut actual = vec![vec![f32::NAN]; count];
                            let previous_blocks = tile.decoded_blocks;
                            tile.project(&tensor, &inputs, &mut actual, pinned, &|| Ok(()))?;
                            assert_eq!(
                                tile.decoded_blocks - previous_blocks,
                                u64::try_from(columns / 128 * 7)?
                            );
                            for (input, output) in inputs.iter().zip(&actual) {
                                let mut scratch = if pinned {
                                    MatvecScratch::with_pinned_backend(MatvecBackend::Scalar)
                                } else {
                                    MatvecScratch::with_backend(MatvecBackend::Scalar)
                                };
                                let mut expected = Vec::new();
                                matvec_into(&tensor, input, &mut expected, &mut scratch)?;
                                assert_eq!(
                                    output
                                        .iter()
                                        .map(|value| value.to_bits())
                                        .collect::<Vec<_>>(),
                                    expected
                                        .iter()
                                        .map(|value| value.to_bits())
                                        .collect::<Vec<_>>()
                                );
                            }
                            let valid = tensor.data.clone();
                            let mut bad = valid.to_vec();
                            let offset =
                                bad.len() - block.len() + if kind == PTQ1 { 26 } else { 0 };
                            bad[offset..offset + 2].copy_from_slice(&0x7c00_u16.to_le_bytes());
                            tensor.data = Arc::from(bad);
                            assert!(
                                tile.project(&tensor, &inputs, &mut actual, pinned, &|| Ok(()))
                                    .is_err()
                            );
                            assert!(actual.iter().all(Vec::is_empty));
                            tensor.data = valid;
                            tile.project(&tensor, &inputs, &mut actual, pinned, &|| Ok(()))?;
                            assert!(actual.iter().all(|output| output.len() == 7));
                        }
                        assert_eq!(tile.row_parallelism.workers(), workers);
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn tiled_packed_projection_reuses_blocks_and_preserves_independent_rows() -> Result<()> {
        for (kind, columns) in [(142, 256), (142, 384), (143, 256)] {
            let mut block = [if kind == 143 { 0x66 } else { 0xe4 }; 34];
            block[..2].copy_from_slice(&0x3555_u16.to_le_bytes());
            let encoded = if kind == 143 {
                bonsai_ternary::pack_ptq(&block)?.to_vec()
            } else {
                block.to_vec()
            };
            let tensor = Tensor {
                name: "tile-packed-reference".into(),
                dimensions: vec![columns, 3],
                kind,
                data: Arc::from(encoded.repeat(columns / 128 * 3)),
            };
            for count in [1, 8, 16, 32] {
                let inputs: Vec<Vec<f32>> = (0..count)
                    .map(|token| {
                        (0..columns)
                            .map(|index| [1.5, -128.0, 0.0, -0.0, 63.5, -3.25][(index + token) % 6])
                            .collect()
                    })
                    .collect();
                for backend in [
                    MatvecBackend::Scalar,
                    MatvecBackend::Auto,
                    MatvecBackend::ExperimentalNeon,
                ] {
                    let mut tiled = ProjectionTile::with_backend(backend);
                    for (pass, pinned) in [false, true, false].into_iter().enumerate() {
                        let mut actual = vec![vec![f32::NAN]; count];
                        tiled.project(&tensor, &inputs, &mut actual, pinned, &|| Ok(()))?;
                        assert_eq!(
                            tiled.decoded_blocks,
                            u64::try_from(columns / 128 * 3 * (pass + 1))?
                        );
                        assert!(tiled.preparations.iter().all(|scratch| {
                            scratch.requested_backend() == backend
                                && scratch.pinned_accumulation == pinned
                                && scratch.pq2_consumption.capacity() == 0
                        }));
                        for (input, actual) in inputs.iter().zip(actual) {
                            let mut scratch = if pinned {
                                MatvecScratch::with_pinned_backend(MatvecBackend::Scalar)
                            } else {
                                MatvecScratch::with_backend(MatvecBackend::Scalar)
                            };
                            let mut expected = Vec::new();
                            matvec_into(&tensor, input, &mut expected, &mut scratch)?;
                            assert_eq!(
                                actual.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                                expected.into_iter().map(f32::to_bits).collect::<Vec<_>>()
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn tiled_projection_clears_all_outputs_on_late_rejection() {
        let tensor = Tensor {
            name: "reject".into(),
            dimensions: vec![128, 2],
            kind: 142,
            data: Arc::from([0_u8; 68]),
        };
        let mut scratch = ProjectionTile::default();
        for count in [0, 8, 33] {
            let mut inputs = vec![vec![0.0; 128]; count];
            if count == 8 {
                inputs[7][127] = f32::NAN;
            }
            let mut outputs = vec![vec![1.0]; count];
            assert!(
                scratch
                    .project(&tensor, &inputs, &mut outputs, true, &|| Ok(()))
                    .is_err()
            );
            assert!(outputs.iter().all(Vec::is_empty));
        }
    }

    #[test]
    fn tiled_projection_clears_completed_rows_on_cancel_and_bad_scale() -> Result<()> {
        for backend in [
            MatvecBackend::Scalar,
            MatvecBackend::Auto,
            MatvecBackend::ExperimentalNeon,
        ] {
            for kind in [PQ2, PTQ1] {
                let mut packed = [0x66; 34];
                packed[..2].copy_from_slice(&0x3555_u16.to_le_bytes());
                let block = if kind == PTQ1 {
                    bonsai_ternary::pack_ptq(&packed)?.to_vec()
                } else {
                    packed.to_vec()
                };
                let mut tensor = Tensor {
                    name: "late-tile-error".into(),
                    dimensions: vec![256, 2],
                    kind,
                    data: Arc::from(block.repeat(4)),
                };
                let inputs = vec![vec![1.5; 256]; 8];
                let mut output = vec![vec![1.0]; 8];
                let mut tile = ProjectionTile::with_backend(backend);
                let calls = Cell::new(0);
                assert!(
                    tile.project(&tensor, &inputs, &mut output, true, &|| {
                        calls.set(calls.get() + 1);
                        // Eight preparations, first row, then cancel at row two.
                        if calls.get() == 10 {
                            Err(invalid("cancel after completed tile row"))
                        } else {
                            Ok(())
                        }
                    })
                    .is_err()
                );
                assert_eq!(tile.decoded_blocks, 2);
                assert!(output.iter().all(Vec::is_empty));
                tile.project(&tensor, &inputs, &mut output, true, &|| Ok(()))?;
                assert!(output.iter().all(|row| row.len() == 2));
                let mut bad = tensor.data.to_vec();
                let offset = block.len() * 3 + if kind == PTQ1 { 26 } else { 0 };
                bad[offset..offset + 2].copy_from_slice(&0x7c00_u16.to_le_bytes());
                tensor.data = Arc::from(bad);
                assert!(
                    tile.project(&tensor, &inputs, &mut output, false, &|| Ok(()))
                        .is_err()
                );
                assert!(output.iter().all(Vec::is_empty));
            }
        }
        Ok(())
    }
}
