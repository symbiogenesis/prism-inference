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
    ActivationContract, BF16, MatvecBackend, MatvecScratch, PQ2, PTQ1, RowParallelism, bf16_dot,
    dimensions, finite, integer_float, matvec_into, scale,
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

    pub(crate) const fn row_parallelism(&self) -> &RowParallelism {
        &self.row_parallelism
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
            let reuse_prepared = matches!(tensor.kind, PQ2 | PTQ1 | BF16)
                && self.preparations.len() >= inputs.len()
                && previous.is_some_and(|prev| {
                    matches!(prev.kind, PQ2 | PTQ1 | BF16)
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
            let parallel_prepare = !reuse_prepared && self.row_parallelism.uses_pool(inputs.len());
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
                if !reuse_prepared && !parallel_prepare {
                    scratch.prepare_input(tensor, input)?;
                }
                output.try_reserve_exact(rows)?;
            }
            if parallel_prepare {
                self.row_parallelism.update_chunks(
                    &mut self.preparations[..inputs.len()],
                    1,
                    |base, chunk| {
                        for (offset, scratch) in chunk.iter_mut().enumerate() {
                            scratch.prepare_input(tensor, &inputs[base + offset])?;
                        }
                        Ok(())
                    },
                )?;
            }
            if tensor.kind == BF16 {
                let backend = self.backend;
                let preparations = &self.preparations[..inputs.len()];
                let mut unused_blocks = 0_u64;
                let dot_bf16_one = |row: &[u8], _: &mut u64| -> Result<[f32; MAX_TILE_TOKENS]> {
                    let mut sums = [0.0_f32; MAX_TILE_TOKENS];
                    for (sum, scratch) in sums.iter_mut().zip(preparations) {
                        *sum = bf16_dot(row, &scratch.activations)?;
                    }
                    Ok(sums)
                };
                return self.row_parallelism.fill_tile_four_rows(
                    &tensor.data,
                    block_bytes,
                    outputs,
                    &mut unused_blocks,
                    check,
                    |rows8, decoded| {
                        if let Some(res) = dot::dot_bf16_eight_rows::<MAX_TILE_TOKENS>(
                            rows8,
                            preparations,
                            backend,
                        ) {
                            return res;
                        }
                        Ok([
                            dot_bf16_one(rows8[0], decoded)?,
                            dot_bf16_one(rows8[1], decoded)?,
                            dot_bf16_one(rows8[2], decoded)?,
                            dot_bf16_one(rows8[3], decoded)?,
                            dot_bf16_one(rows8[4], decoded)?,
                            dot_bf16_one(rows8[5], decoded)?,
                            dot_bf16_one(rows8[6], decoded)?,
                            dot_bf16_one(rows8[7], decoded)?,
                        ])
                    },
                    |rows4, decoded| {
                        if let Some(res) =
                            dot::dot_bf16_four_rows::<MAX_TILE_TOKENS>(rows4, preparations, backend)
                        {
                            return res;
                        }
                        Ok([
                            dot_bf16_one(rows4[0], decoded)?,
                            dot_bf16_one(rows4[1], decoded)?,
                            dot_bf16_one(rows4[2], decoded)?,
                            dot_bf16_one(rows4[3], decoded)?,
                        ])
                    },
                    dot_bf16_one,
                );
            }
            if !matches!(tensor.kind, PQ2 | PTQ1) {
                for ((input, output), scratch) in inputs
                    .iter()
                    .zip(outputs.iter_mut())
                    .zip(self.preparations.iter_mut())
                {
                    check()?;
                    scratch.set_row_parallelism(self.row_parallelism.clone());
                    matvec_into(tensor, input, output, scratch)?;
                }
                return Ok(());
            }
            self.packed_projections = self.packed_projections.saturating_add(1);
            let backend = self.backend;
            let preparations = &self.preparations[..inputs.len()];
            let dot_one = |row: &[u8],
                           decoded_blocks: &mut u64|
             -> Result<[f32; MAX_TILE_TOKENS]> {
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
                                weights.dot128(activation.values[start..start + 128].try_into()?),
                            );
                            let token = rem_base + offset;
                            sums[token] = if pinned {
                                d.mul_add(dot, sums[token])
                            } else {
                                sums[token] + d * dot
                            };
                        }
                    } else if tensor.kind == PTQ1 && pinned {
                        for (token, scratch) in preparations.iter().enumerate() {
                            let activations = <&[super::Q8Block; 4]>::try_from(
                                &scratch.q8[index * 4..index * 4 + 4],
                            )?;
                            weights.accumulate_lanes128(
                                weight_scale,
                                activations,
                                &mut lanes[token],
                            );
                        }
                    } else {
                        for (token, scratch) in preparations.iter().enumerate() {
                            let activations = <&[super::Q8Block; 4]>::try_from(
                                &scratch.q8[index * 4..index * 4 + 4],
                            )?;
                            sums[token] += weight_scale * weights.dot128_q8(activations);
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
            };
            self.row_parallelism.fill_tile_four_rows(
                &tensor.data,
                block_bytes,
                outputs,
                &mut self.decoded_blocks,
                check,
                |rows8, decoded_blocks| {
                    if let Some(res) = dot::dot_eight_rows::<MAX_TILE_TOKENS>(
                        rows8,
                        preparations,
                        tensor.kind,
                        columns,
                        pinned,
                        backend,
                        decoded_blocks,
                    ) {
                        return res;
                    }
                    Ok([
                        dot_one(rows8[0], decoded_blocks)?,
                        dot_one(rows8[1], decoded_blocks)?,
                        dot_one(rows8[2], decoded_blocks)?,
                        dot_one(rows8[3], decoded_blocks)?,
                        dot_one(rows8[4], decoded_blocks)?,
                        dot_one(rows8[5], decoded_blocks)?,
                        dot_one(rows8[6], decoded_blocks)?,
                        dot_one(rows8[7], decoded_blocks)?,
                    ])
                },
                |rows4, decoded_blocks| {
                    if let Some(res) = dot::dot_four_rows::<MAX_TILE_TOKENS>(
                        rows4,
                        preparations,
                        tensor.kind,
                        columns,
                        pinned,
                        backend,
                        decoded_blocks,
                    ) {
                        return res;
                    }
                    Ok([
                        dot_one(rows4[0], decoded_blocks)?,
                        dot_one(rows4[1], decoded_blocks)?,
                        dot_one(rows4[2], decoded_blocks)?,
                        dot_one(rows4[3], decoded_blocks)?,
                    ])
                },
                dot_one,
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

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "Cover 8-row and 4-row PQ2/PTQ1 GEMM groups, remainder rows and mid-group error/cancel recovery together."
    )]
    fn tiled_four_row_gemm_matches_scalar_and_recovers_mid_group_failure() -> Result<()> {
        let rows = 13_usize;
        for (kind, columns) in [(PQ2, 256_usize), (PQ2, 384), (PTQ1, 256)] {
            let blocks_per_row = columns / 128;
            let mut data = Vec::new();
            for row in 0..rows {
                for blk in 0..blocks_per_row {
                    let mut raw = [0_u8; 34];
                    let scale = u16::try_from(0x3400_usize + ((row * 3 + blk) % 16) * 0x10)?;
                    raw[..2].copy_from_slice(&scale.to_le_bytes());
                    for (idx, slot) in raw[2..].iter_mut().enumerate() {
                        *slot = if kind == PTQ1 {
                            [0x00, 0x55, 0x66, 0x99, 0xaa][(row + blk + idx) % 5]
                        } else {
                            u8::try_from((row * 29 + blk * 13 + idx * 7) & 0xff)?
                        };
                    }
                    if kind == PTQ1 {
                        data.extend_from_slice(&bonsai_ternary::pack_ptq(&raw)?);
                    } else {
                        data.extend_from_slice(&raw);
                    }
                }
            }
            let tensor = Tensor {
                name: "tile-four-row-gemm".into(),
                dimensions: vec![columns, rows],
                kind,
                data: Arc::from(data),
            };
            for count in [1, 4, 8, 16, 32] {
                let inputs: Vec<Vec<f32>> = (0..count)
                    .map(|token| {
                        (0..columns)
                            .map(|idx| {
                                [1.5, -64.0, 0.25, -0.0, 31.75, -3.25, 7.125][(idx + token * 3) % 7]
                            })
                            .collect()
                    })
                    .collect();
                for pinned in [false, true] {
                    let expected: Vec<Vec<u32>> = inputs
                        .iter()
                        .map(|input| {
                            let mut scratch = if pinned {
                                MatvecScratch::with_pinned_backend(MatvecBackend::Scalar)
                            } else {
                                MatvecScratch::with_backend(MatvecBackend::Scalar)
                            };
                            let mut out = Vec::new();
                            matvec_into(&tensor, input, &mut out, &mut scratch)?;
                            Ok(out.into_iter().map(f32::to_bits).collect())
                        })
                        .collect::<Result<_>>()?;
                    for backend in [
                        MatvecBackend::Scalar,
                        MatvecBackend::Auto,
                        MatvecBackend::ExperimentalNeon,
                    ] {
                        let mut tile = ProjectionTile::with_backend(backend);
                        let mut actual = vec![Vec::new(); count];
                        tile.project(&tensor, &inputs, &mut actual, pinned, &|| Ok(()))?;
                        assert_eq!(tile.decoded_blocks, u64::try_from(rows * blocks_per_row)?);
                        let actual_bits: Vec<Vec<u32>> = actual
                            .into_iter()
                            .map(|row| row.into_iter().map(f32::to_bits).collect())
                            .collect();
                        assert_eq!(actual_bits, expected);
                    }
                }
            }
            // Verify mid-8-row-group cancellation and mid-group bad scale rewind.
            let inputs = vec![vec![1.25; columns]; 8];
            let mut output = vec![vec![1.0]; 8];
            let mut tile = ProjectionTile::with_backend(MatvecBackend::ExperimentalNeon);
            let calls = Cell::new(0);
            assert!(
                tile.project(&tensor, &inputs, &mut output, true, &|| {
                    calls.set(calls.get() + 1);
                    // Eight preparations + row 0 ok + cancel at row 1 inside first 8-row group.
                    if calls.get() == 10 {
                        Err(invalid("cancel inside 8-row group"))
                    } else {
                        Ok(())
                    }
                })
                .is_err()
            );
            assert_eq!(tile.decoded_blocks, u64::try_from(blocks_per_row)?);
            assert!(output.iter().all(Vec::is_empty));

            let block_bytes = if kind == PTQ1 { 28 } else { 34 };
            let mut bad = tensor.data.to_vec();
            // Corrupt row 2 (third row inside the first 8-row group).
            let offset = (2 * blocks_per_row) * block_bytes + if kind == PTQ1 { 26 } else { 0 };
            bad[offset..offset + 2].copy_from_slice(&0x7c00_u16.to_le_bytes());
            let bad_tensor = Tensor {
                data: Arc::from(bad),
                ..tensor
            };
            let mut bad_tile = ProjectionTile::with_backend(MatvecBackend::ExperimentalNeon);
            assert!(
                bad_tile
                    .project(&bad_tensor, &inputs, &mut output, true, &|| Ok(()))
                    .is_err()
            );
            assert_eq!(bad_tile.decoded_blocks, u64::try_from(2 * blocks_per_row)?);
            assert!(output.iter().all(Vec::is_empty));
        }
        Ok(())
    }

    #[test]
    fn tiled_bf16_gemm_matches_scalar_and_reuses_prepared_activations() -> Result<()> {
        let columns = 20_usize;
        let rows = 13_usize;
        let mut data = Vec::with_capacity(columns * rows * 2);
        for row in 0..rows {
            for col in 0..columns {
                let bits = u16::try_from(0x3c00_usize + ((row * 5 + col * 3) % 64) * 4)?;
                data.extend_from_slice(&bits.to_le_bytes());
            }
        }
        let tensor = Tensor {
            name: "tile-bf16-gemm".into(),
            dimensions: vec![columns, rows],
            kind: BF16,
            data: Arc::from(data),
        };
        let inputs: Vec<Vec<f32>> = (0..8)
            .map(|token| {
                (0..columns)
                    .map(|idx| [1.25, -3.5, 0.0, -0.0, 7.125, -0.375][(idx + token) % 6])
                    .collect()
            })
            .collect();
        let expected: Vec<Vec<u32>> = inputs
            .iter()
            .map(|input| {
                let mut scratch = MatvecScratch::with_backend(MatvecBackend::Scalar);
                let mut out = Vec::new();
                matvec_into(&tensor, input, &mut out, &mut scratch)?;
                Ok(out.into_iter().map(f32::to_bits).collect())
            })
            .collect::<Result<_>>()?;
        for backend in [
            MatvecBackend::Scalar,
            MatvecBackend::Auto,
            MatvecBackend::ExperimentalNeon,
        ] {
            let mut tile = ProjectionTile::with_backend(backend);
            #[cfg(feature = "parallel-rows")]
            tile.set_row_parallelism(RowParallelism::new(2)?);
            let mut actual = vec![Vec::new(); inputs.len()];
            tile.project(&tensor, &inputs, &mut actual, true, &|| Ok(()))?;
            assert_eq!(tile.packed_projections, 0);
            assert_eq!(tile.decoded_blocks, 0);
            let actual_bits: Vec<Vec<u32>> = actual
                .iter()
                .map(|row| row.iter().copied().map(f32::to_bits).collect())
                .collect();
            assert_eq!(actual_bits, expected);
            let preps_before: Vec<usize> = tile
                .preparations
                .iter()
                .map(MatvecScratch::preparation_count)
                .collect();
            tile.project_with_previous(
                Some(&tensor),
                &tensor,
                &inputs,
                &mut actual,
                true,
                &|| Ok(()),
            )?;
            let preps_after: Vec<usize> = tile
                .preparations
                .iter()
                .map(MatvecScratch::preparation_count)
                .collect();
            assert_eq!(preps_after, preps_before);
        }
        Ok(())
    }
}
