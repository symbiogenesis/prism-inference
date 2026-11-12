//! Adapter FFN consumer controls against the original separate operations.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use bonsai_gguf::MetadataValue;

use super::super::ProjectionScratch;
use super::{
    AdapterScratch, Arithmetic, Evaluation, Geometry, MatvecBackend, ProjectionTile,
    RankOneAdapter, RowParallelism, Tensor, TileReuse, Transforms, project_tile,
    project_tile_ffn_swiglu,
};
use crate::{Result, model::Model};

fn tensor(name: &str, dimensions: &[usize], values: &[f32]) -> Tensor {
    Tensor {
        name: name.into(),
        dimensions: dimensions.to_vec(),
        kind: 0,
        data: Arc::from(
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        ),
    }
}

struct Fixture {
    geometry: Geometry,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
    transforms: Transforms,
    adapter: RankOneAdapter,
}

impl Fixture {
    fn new(width: usize, block: usize) -> Result<Self> {
        let pattern = [-0.75, -0.0, 0.25, 0.5, -0.125, 1.0, -0.5];
        let values = |length, shift| {
            (0..length)
                .map(|index| pattern[(index + shift) % pattern.len()])
                .collect::<Vec<_>>()
        };
        let gate = tensor("blk.0.ffn_gate.weight", &[2, width], &values(2 * width, 0));
        let up = tensor("blk.0.ffn_up.weight", &[2, width], &values(2 * width, 2));
        let down = tensor("blk.0.ffn_down.weight", &[width, 2], &values(2 * width, 4));
        let base = Model {
            metadata: BTreeMap::from([(
                "general.architecture".into(),
                MetadataValue::from("qwen35"),
            )]),
            tensors: BTreeMap::from([(down.name.clone(), down.clone())]),
            identity: "adapter-ffn-base-control".into(),
        };
        let a = tensor(
            "blk.0.ffn_down.weight.lora_a",
            &[width, 1],
            &values(width, 1),
        );
        let b = tensor("blk.0.ffn_down.weight.lora_b", &[1, 2], &[0.25, -0.5]);
        let factors = Model {
            metadata: BTreeMap::from([
                ("general.architecture".into(), MetadataValue::from("qwen35")),
                ("general.type".into(), MetadataValue::from("adapter")),
                ("adapter.type".into(), MetadataValue::from("lora")),
                ("adapter.lora_alpha".into(), MetadataValue::from(1.0_f32)),
            ]),
            tensors: BTreeMap::from([(a.name.clone(), a), (b.name.clone(), b)]),
            identity: "adapter-ffn-factor-control".into(),
        };
        Ok(Self {
            geometry: Geometry {
                embedding: 2,
                feed_forward: width,
                heads: 1,
                kv_heads: 1,
                head: 2,
                rope: 2,
                rope_sections: [1, 0, 0, 0],
                rope_base: 10_000.0,
                epsilon: 0.000_001,
                ssm_head: 2,
                ssm_keys: 1,
                ssm_values: 1,
                convolution: 2,
            },
            transforms: Transforms {
                block,
                weights: BTreeSet::from([down.name.clone()]),
                signs: BTreeMap::from([(
                    width,
                    (0..width)
                        .map(|index| if index % 3 == 0 { -1.0 } else { 1.0 })
                        .collect(),
                )]),
                ..Transforms::default()
            },
            adapter: RankOneAdapter::from_model(factors, &base)?,
            gate,
            up,
            down,
        })
    }

    fn decode(
        &self,
        arithmetic: Arithmetic,
        workers: usize,
        scale: f32,
        input: &[f32],
        fused: bool,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let mut projection = ProjectionScratch::default();
        projection
            .matvec
            .set_row_parallelism(RowParallelism::new(workers)?);
        let mut adapter_scratch = AdapterScratch::default();
        let mut evaluation = Evaluation {
            geometry: &self.geometry,
            arithmetic,
            transforms: &self.transforms,
            position: 0,
            frequencies_ready: false,
            projection: &mut projection,
            adapter: Some(&self.adapter),
            adapter_scale: scale,
            adapter_scratch: &mut adapter_scratch,
            executor: None,
        };
        let mut gate = Vec::new();
        let mut up = Vec::new();
        if fused {
            evaluation.project_ffn_swiglu_into(
                &self.gate, &self.up, &self.down, input, &mut gate, &mut up,
            )?;
            assert_eq!(
                evaluation.projection.pre_transformed_len,
                self.transforms
                    .can_block_hadamard(up.len())
                    .then_some(up.len())
            );
        } else {
            evaluation.project_group_into([&self.up, &self.gate], input, [&mut up, &mut gate])?;
            arithmetic.swiglu_in_place_with_parallelism(
                &gate,
                &mut up,
                evaluation.projection.matvec.row_parallelism(),
            )?;
        }
        let mut output = Vec::new();
        evaluation.project_into(&self.down, &up, &mut output)?;
        Ok((up, output))
    }
}

fn same_bits(actual: &[f32], expected: &[f32]) {
    assert!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .eq(expected.iter().map(|value| value.to_bits()))
    );
}

#[test]
fn adapter_ffn_consumers_retain_original_basis_and_exact_down_correction() -> Result<()> {
    let workers = if cfg!(feature = "parallel-rows") {
        4
    } else {
        1
    };
    for arithmetic in [Arithmetic::Source, Arithmetic::PinnedSwift] {
        for (width, block) in [(32, 16), (4096, 256), (32, 8)] {
            let fixture = Fixture::new(width, block)?;
            for scale in [0.0, 1.3] {
                let inputs = vec![vec![0.75, -0.5], vec![-0.25, 1.0], vec![0.0, -0.0]];
                let expected = inputs
                    .iter()
                    .map(|input| fixture.decode(arithmetic, workers, scale, input, false))
                    .collect::<Result<Vec<_>>>()?;
                for (input, (expected_up, expected_output)) in inputs.iter().zip(&expected) {
                    let (up, output) = fixture.decode(arithmetic, workers, scale, input, true)?;
                    same_bits(&up, expected_up);
                    same_bits(&output, expected_output);
                }

                let mut projector = ProjectionTile::with_backend(MatvecBackend::Scalar);
                projector.set_row_parallelism(RowParallelism::new(workers)?);
                let mut projection_scratch = arithmetic.scratch(MatvecBackend::Scalar);
                let mut adapter_scratch = AdapterScratch::default();
                let mut gate = vec![Vec::new(); inputs.len()];
                let mut up = vec![Vec::new(); inputs.len()];
                let mut transformed = vec![Vec::new(); inputs.len()];
                let reuse = project_tile_ffn_swiglu(
                    &fixture.transforms,
                    &fixture.geometry,
                    arithmetic,
                    Some(&fixture.adapter),
                    scale,
                    &mut adapter_scratch,
                    &mut projection_scratch,
                    TileReuse::default(),
                    &fixture.gate,
                    &fixture.up,
                    &fixture.down,
                    &inputs,
                    &mut gate,
                    &mut up,
                    &mut transformed,
                    &mut projector,
                    None,
                    &|| Ok(()),
                )?;
                assert_eq!(reuse.basis.is_some(), block >= 16);
                let mut outputs = vec![Vec::new(); inputs.len()];
                let _ = project_tile(
                    &fixture.transforms,
                    &fixture.geometry,
                    arithmetic,
                    Some(&fixture.adapter),
                    scale,
                    &mut adapter_scratch,
                    &mut projection_scratch,
                    reuse,
                    &fixture.down,
                    &up,
                    &mut outputs,
                    &mut transformed,
                    &mut projector,
                    None,
                    &|| Ok(()),
                )?;
                for ((up, output), (expected_up, expected_output)) in
                    up.iter().zip(&outputs).zip(&expected)
                {
                    same_bits(up, expected_up);
                    same_bits(output, expected_output);
                }
            }
        }
    }
    Ok(())
}

#[test]
fn adapter_ffn_preparation_rejects_invalid_geometry_and_never_replays_failed_swiglu() -> Result<()>
{
    for arithmetic in [Arithmetic::Source, Arithmetic::PinnedSwift] {
        let gate = vec![0.5; 32];
        let original_up = vec![0.25; 32];
        for (gate, block, signs) in [
            (&gate[..31], 16, None),
            (&gate[..], 0, None),
            (&gate[..], 32, Some(&gate[..31])),
        ] {
            let mut up = original_up.clone();
            let mut transformed = vec![42.0];
            assert!(
                arithmetic
                    .swiglu_hadamard_preserving_up_into(
                        gate,
                        &mut up,
                        block,
                        signs,
                        &RowParallelism::default(),
                        &mut transformed,
                    )
                    .is_err()
            );
            same_bits(&up, &original_up);
            assert!(transformed.is_empty());
        }
        let mut up = original_up.clone();
        let mut signs = vec![1.0; 32];
        signs[0] = f32::INFINITY;
        let mut expected = original_up[..16].to_vec();
        arithmetic.swiglu_in_place(&gate[..16], &mut expected)?;
        let mut transformed = Vec::new();
        assert!(
            arithmetic
                .swiglu_hadamard_preserving_up_into(
                    &gate,
                    &mut up,
                    16,
                    Some(&signs),
                    &RowParallelism::default(),
                    &mut transformed,
                )
                .is_err_and(
                    |error| error.to_string() == "Hadamard requires a finite power-of-two block"
                )
        );
        same_bits(&up[..16], &expected);
        same_bits(&up[16..], &original_up[16..]);
        assert!(transformed.is_empty());
    }
    Ok(())
}
