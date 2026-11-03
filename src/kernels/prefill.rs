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
    MatvecBackend, MatvecScratch, PQ2, PTQ1, dimensions, dot_integer, finite, integer_float,
    matvec_into, pq2_values, ptq1_values, q8_native_lanes, scale,
};
use crate::{Result, invalid, model::Tensor};

pub(crate) const MAX_TILE_TOKENS: usize = 32;

#[derive(Default)]
pub(crate) struct ProjectionTile {
    preparations: Vec<MatvecScratch>,
    pub(crate) packed_projections: u64,
    pub(crate) decoded_blocks: u64,
}

impl ProjectionTile {
    #[allow(
        clippy::suboptimal_flops,
        clippy::too_many_lines,
        reason = "Retain the separately rounded source arithmetic and auditable packed block order."
    )]
    pub(crate) fn project(
        &mut self,
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
            while self.preparations.len() < inputs.len() {
                self.preparations.try_reserve_exact(1)?;
                self.preparations.push(if pinned {
                    MatvecScratch::with_pinned_backend(MatvecBackend::Scalar)
                } else {
                    MatvecScratch::with_backend(MatvecBackend::Scalar)
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
                scratch.prepare_input(tensor, input)?;
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
            for row in tensor.data.chunks_exact(block_bytes) {
                check()?;
                let mut sums = [0.0_f32; MAX_TILE_TOKENS];
                let mut lanes = [[0.0_f32; 4]; MAX_TILE_TOKENS];
                let bytes = if tensor.kind == PQ2 { 34 } else { 28 };
                for (index, block) in row.chunks_exact(bytes).enumerate() {
                    let weights = if tensor.kind == PQ2 {
                        pq2_values(block)
                    } else {
                        ptq1_values(block)?
                    };
                    let weight_scale = if tensor.kind == PQ2 {
                        scale(block)?
                    } else {
                        scale(&block[26..])?
                    };
                    self.decoded_blocks = self.decoded_blocks.saturating_add(1);
                    for (token, scratch) in self.preparations[..inputs.len()].iter().enumerate() {
                        if tensor.kind == PQ2 && columns.is_multiple_of(256) {
                            let activation = &scratch.q8k[index / 2];
                            let start = (index % 2) * 128;
                            let d = weight_scale * activation.scale;
                            let dot = dot_integer(&weights, &activation.values[start..start + 128]);
                            sums[token] = if pinned {
                                d.mul_add(dot, sums[token])
                            } else {
                                sums[token] + d * dot
                            };
                        } else {
                            let activations = &scratch.q8[index * 4..index * 4 + 4];
                            let mut sum = 0.0;
                            for (group, activation) in
                                weights.as_chunks::<32>().0.iter().zip(activations)
                            {
                                if tensor.kind == PTQ1 && pinned {
                                    let d = weight_scale * activation.scale;
                                    for (lane, integer) in lanes[token]
                                        .iter_mut()
                                        .zip(q8_native_lanes(group, &activation.values))
                                    {
                                        *lane = integer_float(integer).mul_add(d, *lane);
                                    }
                                } else {
                                    sum +=
                                        activation.scale * dot_integer(group, &activation.values);
                                }
                            }
                            if tensor.kind != PTQ1 || !pinned {
                                sums[token] += weight_scale * sum;
                            }
                        }
                    }
                }
                for (token, output) in outputs.iter_mut().enumerate() {
                    output.push(finite(if tensor.kind == PTQ1 && pinned {
                        (lanes[token][0] + lanes[token][1]) + (lanes[token][2] + lanes[token][3])
                    } else {
                        sums[token]
                    })?);
                }
            }
            check()?;
            Ok(())
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
    use std::sync::Arc;

    use super::*;

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
                for pinned in [false, true] {
                    let mut actual = vec![vec![f32::NAN]; count];
                    let mut tiled = ProjectionTile::default();
                    tiled.project(&tensor, &inputs, &mut actual, pinned, &|| Ok(()))?;
                    assert_eq!(tiled.decoded_blocks, u64::try_from(columns / 128 * 3)?);
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
}
