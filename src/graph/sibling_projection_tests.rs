//! Real grouped-projection consumer controls without model or device timing.

use std::sync::Arc;

use super::{Arithmetic, Geometry, ProjectionScratch, Transforms};
use crate::{
    Result,
    kernels::{MatvecBackend, RowParallelism},
    model::Tensor,
};

fn weight(name: &str, rows: usize) -> Tensor {
    let mut block = [0; 34];
    block[..2].copy_from_slice(&0x3555_u16.to_le_bytes());
    for (value, index) in block[2..].iter_mut().zip(0_u8..) {
        *value = index.wrapping_mul(37);
    }
    Tensor {
        name: name.to_owned(),
        dimensions: vec![512, rows],
        kind: 142,
        data: Arc::from(block.repeat(4 * rows)),
    }
}

#[test]
fn cpu_sibling_projection_matches_independent_rows_and_retains_preparation() -> Result<()> {
    let geometry = Geometry {
        embedding: 512,
        feed_forward: 512,
        heads: 1,
        kv_heads: 1,
        head: 512,
        rope: 512,
        rope_sections: [256, 0, 0, 0],
        rope_base: 10_000.0,
        epsilon: 0.000_001,
        ssm_head: 128,
        ssm_keys: 2,
        ssm_values: 4,
        convolution: 1,
    };
    let weights = [weight("q", 17), weight("k", 5), weight("v", 9)];
    let input: Vec<_> = (0..512)
        .map(|index| [127.0, -127.0, 0.5, -0.0, 63.5, 1.5, -0.5][index % 7])
        .collect();
    let transforms = Transforms {
        block: 256,
        weights: weights.iter().map(|weight| weight.name.clone()).collect(),
        signs: std::collections::BTreeMap::from([(
            512,
            (0..512)
                .map(|index| if index % 3 == 0 { -1.0 } else { 1.0 })
                .collect(),
        )]),
        ..Transforms::default()
    };
    for arithmetic in [Arithmetic::Source, Arithmetic::PinnedSwift] {
        for backend in [
            MatvecBackend::Scalar,
            MatvecBackend::Auto,
            MatvecBackend::ExperimentalNeon,
        ] {
            let mut reference = ProjectionScratch {
                matvec: arithmetic.scratch(backend),
                ..ProjectionScratch::default()
            };
            let mut expected = std::array::from_fn::<_, 3, _>(|_| Vec::new());
            for (weight, output) in weights.iter().zip(&mut expected) {
                transforms.project_into(weight, &input, &geometry, output, &mut reference)?;
            }
            for workers in [2, 4, 18] {
                let mut scratch = ProjectionScratch {
                    matvec: arithmetic.scratch(backend),
                    ..ProjectionScratch::default()
                };
                scratch
                    .matvec
                    .set_row_parallelism(RowParallelism::new(workers)?);
                let mut actual = std::array::from_fn::<_, 3, _>(|_| vec![f32::NAN]);
                transforms.project_group_into(
                    weights.each_ref(),
                    &input,
                    &geometry,
                    actual.each_mut(),
                    &mut scratch,
                )?;
                for (actual, expected) in actual.iter().zip(&expected) {
                    assert!(
                        actual
                            .iter()
                            .map(|value| value.to_bits())
                            .eq(expected.iter().map(|value| value.to_bits()))
                    );
                }
                assert_eq!(scratch.transforms, 1);
                assert_eq!(scratch.matvec.preparation_count(), 1);
                assert_eq!(scratch.matvec.statistics().grouped_row_dispatches, 1);
                assert_eq!(scratch.matvec.statistics().matrix_successes, 3);
            }
        }
    }
    Ok(())
}

#[test]
fn recurrent_bf16_siblings_keep_exact_dots_and_share_one_worker_submission() -> Result<()> {
    let geometry = Geometry {
        embedding: 512,
        feed_forward: 512,
        heads: 1,
        kv_heads: 1,
        head: 512,
        rope: 512,
        rope_sections: [256, 0, 0, 0],
        rope_base: 10_000.0,
        epsilon: 0.000_001,
        ssm_head: 128,
        ssm_keys: 2,
        ssm_values: 4,
        convolution: 1,
    };
    let mut weights = [
        weight("qkv", 17),
        weight("gate", 9),
        weight("beta", 5),
        weight("alpha", 33),
    ];
    for weight in &mut weights[2..] {
        weight.kind = 30;
        weight.data = Arc::from(
            [0x3f80_u16, 0xbfaa, 0x8000, 0x0080, 0x0001, 0x3e55, 0xc000]
                .into_iter()
                .cycle()
                .take(weight.dimensions[0] * weight.dimensions[1])
                .flat_map(u16::to_le_bytes)
                .collect::<Vec<_>>(),
        );
    }
    let input: Vec<_> = (0..512)
        .map(|index| [127.0, -127.0, 0.5001, -0.0, 63.5001, 1.5, -0.5001][index % 7])
        .collect();
    let transforms = Transforms {
        block: 256,
        weights: weights[..2]
            .iter()
            .map(|weight| weight.name.clone())
            .collect(),
        ..Transforms::default()
    };
    for arithmetic in [Arithmetic::Source, Arithmetic::PinnedSwift] {
        for backend in [MatvecBackend::Scalar, MatvecBackend::Auto] {
            let mut reference = ProjectionScratch {
                matvec: arithmetic.scratch(backend),
                ..ProjectionScratch::default()
            };
            let mut expected = std::array::from_fn::<_, 4, _>(|_| Vec::new());
            for (weight, output) in weights.iter().zip(&mut expected) {
                transforms.project_into(weight, &input, &geometry, output, &mut reference)?;
            }
            for workers in [2, 4, 18] {
                let mut scratch = ProjectionScratch {
                    matvec: arithmetic.scratch(backend),
                    ..ProjectionScratch::default()
                };
                scratch
                    .matvec
                    .set_row_parallelism(RowParallelism::new(workers)?);
                let mut actual = std::array::from_fn::<_, 4, _>(|_| vec![f32::NAN]);
                transforms.project_group_into(
                    weights.each_ref(),
                    &input,
                    &geometry,
                    actual.each_mut(),
                    &mut scratch,
                )?;
                for (actual, expected) in actual.iter().zip(&expected) {
                    assert!(
                        actual
                            .iter()
                            .map(|value| value.to_bits())
                            .eq(expected.iter().map(|value| value.to_bits()))
                    );
                }
                assert_eq!(scratch.transforms, 1);
                assert_eq!(scratch.matvec.preparation_count(), 2);
                assert_eq!(scratch.matvec.statistics().grouped_row_dispatches, 2);
                assert_eq!(scratch.matvec.statistics().matrix_successes, 4);
            }
        }
    }
    Ok(())
}
