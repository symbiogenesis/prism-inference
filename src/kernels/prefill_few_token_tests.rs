use std::sync::Arc;

#[cfg(feature = "parallel-rows")]
use super::RowParallelism;
use super::{
    MAX_TILE_TOKENS, MatvecBackend, MatvecScratch, PQ2, PTQ1, ProjectionTile, Q4_0, Q8_0, dot,
    matvec_into,
};
use crate::kernels::{Q8Block, integer_float};
use crate::{Result, model::Tensor};

#[test]
#[expect(
    clippy::suboptimal_flops,
    reason = "The independent reference retains separately rounded per-block scale, product and sum."
)]
fn few_token_legacy_tiles_decode_once_and_match_independent_integer_equations() -> Result<()> {
    for kind in [Q4_0, Q8_0] {
        let weights: [[i16; 32]; 3] = std::array::from_fn(|block| {
            std::array::from_fn(|column| {
                if kind == Q4_0 {
                    i16::try_from((column * 7 + block * 3) % 16).unwrap_or(0) - 8
                } else {
                    i16::from([-128_i8, 127, -1, 0, 63, -64][(column + block) % 6])
                }
            })
        });
        let mut row = Vec::new();
        for (values, bits) in weights.iter().zip([0x3c00_u16, 0xb800, 0x8000]) {
            row.extend_from_slice(&bits.to_le_bytes());
            if kind == Q4_0 {
                for column in 0..16 {
                    row.push(
                        u8::try_from(values[column] + 8)?
                            | (u8::try_from(values[column + 16] + 8)? << 4),
                    );
                }
            } else {
                for &value in values {
                    row.extend_from_slice(&i8::try_from(value)?.to_ne_bytes());
                }
            }
        }
        for count in [2, 3, 7, 9, 17, 24, MAX_TILE_TOKENS] {
            let preparations: Vec<MatvecScratch> = (0..count)
                .map(|token| MatvecScratch {
                    q8: (0..3)
                        .map(|block| Q8Block {
                            scale: [0.25, -0.125, -0.0][block],
                            sum: 0.0,
                            values: std::array::from_fn(|column| {
                                [-128_i8, -1, 0, 1, 127, 63, -64][(column + token + block) % 7]
                            }),
                        })
                        .collect(),
                    ..MatvecScratch::default()
                })
                .collect();
            let mut decoded = 0;
            let actual = dot::dot_legacy_tile_row::<MAX_TILE_TOKENS>(
                kind,
                &row,
                &preparations,
                &mut decoded,
            )?;
            assert_eq!(
                decoded, 3,
                "weight decoding must not scale with {count} tokens"
            );
            for (value, scratch) in actual.iter().zip(&preparations) {
                let mut expected = 0.0_f32;
                for ((weights, activation), scale) in
                    weights.iter().zip(&scratch.q8).zip([1.0, -0.5, -0.0])
                {
                    let integer: i32 = weights
                        .iter()
                        .zip(activation.values)
                        .map(|(&w, a)| i32::from(w) * i32::from(a))
                        .sum();
                    expected += (scale * activation.scale) * integer_float(integer);
                }
                assert_eq!(value.to_bits(), expected.to_bits());
            }
            assert!(actual[count..].iter().all(|value| value.to_bits() == 0));
            let mut malformed = row.clone();
            let last = malformed.len() - if kind == Q4_0 { 18 } else { 34 };
            malformed[last..last + 2].copy_from_slice(&0x7c00_u16.to_le_bytes());
            decoded = 0;
            assert!(
                dot::dot_legacy_tile_row::<MAX_TILE_TOKENS>(
                    kind,
                    &malformed,
                    &preparations,
                    &mut decoded,
                )
                .is_err()
            );
            assert_eq!(decoded, 2);
        }
    }
    Ok(())
}

#[test]
fn few_token_projection_tails_preserve_scalar_bits_and_failure_recovery() -> Result<()> {
    let rows = 13;
    for (kind, columns) in [(PQ2, 256), (PQ2, 384), (PTQ1, 256), (Q4_0, 96), (Q8_0, 96)] {
        let mut data = Vec::new();
        for row in 0..rows {
            let values: Vec<f32> = (0..columns)
                .map(|column| [1.0, -2.0, 0.0, -0.0, 0.5, -0.25, 3.0][(column + row) % 7])
                .collect();
            data.extend_from_slice(&super::super::quantize_row(kind, &values)?);
        }
        let mut tensor = Tensor {
            name: "few-token-tails".into(),
            dimensions: vec![columns, rows],
            kind,
            data: Arc::from(data),
        };
        for count in [2, 3, 7, 9, 17, 24] {
            let inputs: Vec<Vec<f32>> = (0..count)
                .map(|token| {
                    (0..columns)
                        .map(|column| [1.5, -128.0, 0.0, -0.0, 63.5, -3.25][(column + token) % 6])
                        .collect()
                })
                .collect();
            let expected: Vec<Vec<u32>> = inputs
                .iter()
                .map(|input| {
                    let mut scratch = MatvecScratch::with_pinned_backend(MatvecBackend::Scalar);
                    let mut output = Vec::new();
                    matvec_into(&tensor, input, &mut output, &mut scratch)?;
                    Ok(output.into_iter().map(f32::to_bits).collect())
                })
                .collect::<Result<_>>()?;
            for backend in [MatvecBackend::Scalar, MatvecBackend::Auto] {
                let mut tile = ProjectionTile::with_backend(backend);
                #[cfg(feature = "parallel-rows")]
                tile.set_row_parallelism(RowParallelism::new(4)?);
                let mut actual = vec![vec![f32::NAN]; count];
                tile.project(&tensor, &inputs, &mut actual, true, &|| Ok(()))?;
                if matches!(kind, PQ2 | PTQ1) {
                    assert_eq!(tile.decoded_blocks, u64::try_from(rows * columns / 128)?);
                }
                let bits: Vec<Vec<u32>> = actual
                    .iter()
                    .map(|row| row.iter().copied().map(f32::to_bits).collect())
                    .collect();
                assert_eq!(bits, expected);
                // A late malformed scale must clear all token rows, including
                // completed work, before a clean retry through the same owner.
                let valid = tensor.data.clone();
                let bytes = match kind {
                    PTQ1 => 28,
                    Q4_0 => 18,
                    _ => 34,
                };
                let mut bad = valid.to_vec();
                let scale = bad.len() - bytes + if kind == PTQ1 { 26 } else { 0 };
                bad[scale..scale + 2].copy_from_slice(&0x7c00_u16.to_le_bytes());
                tensor.data = Arc::from(bad);
                assert!(
                    tile.project(&tensor, &inputs, &mut actual, true, &|| Ok(()))
                        .is_err()
                );
                assert!(actual.iter().all(Vec::is_empty));
                tensor.data = valid;
                tile.project(&tensor, &inputs, &mut actual, true, &|| Ok(()))?;
                let bits: Vec<Vec<u32>> = actual
                    .iter()
                    .map(|row| row.iter().copied().map(f32::to_bits).collect())
                    .collect();
                assert_eq!(bits, expected);
            }
        }
    }
    Ok(())
}
