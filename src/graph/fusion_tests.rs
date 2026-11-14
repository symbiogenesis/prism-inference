//! Real decode preparation paths checked against separate operations.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use super::{Arithmetic, Evaluation, FusionCount, Geometry, ProjectionScratch, Transforms};
use crate::{
    Result,
    kernels::{MatvecBackend, RowParallelism},
    model::{Tensor, lora::Scratch as AdapterScratch},
};

fn geometry(width: usize) -> Geometry {
    Geometry {
        embedding: width,
        feed_forward: width,
        heads: 1,
        kv_heads: 1,
        head: width,
        rope: width,
        rope_sections: [width / 2, 0, 0, 0],
        rope_base: 10_000.0,
        epsilon: 0.000_001,
        ssm_head: width,
        ssm_keys: 1,
        ssm_values: 1,
        convolution: 1,
    }
}

fn tensor(name: &str, width: usize, shift: usize) -> Tensor {
    let pattern = [0.125_f32, -0.25, 0.5, -0.0, 0.0625];
    Tensor {
        name: name.into(),
        dimensions: vec![width, width],
        kind: 0,
        data: Arc::from(
            (0..width * width)
                .flat_map(|index| pattern[(index + shift) % pattern.len()].to_le_bytes())
                .collect::<Vec<_>>(),
        ),
    }
}

fn compare(actual: &[f32], expected: &[f32], count: FusionCount) {
    assert_eq!(actual.len(), expected.len());
    let max_error = actual.iter().zip(expected).fold(0.0_f64, |max, (&a, &b)| {
        max.max((f64::from(a) - f64::from(b)).abs())
    });
    assert!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .eq(expected.iter().map(|value| value.to_bits())),
        "fusion={count:?}; max absolute error={max_error}"
    );
    assert_eq!(
        count,
        FusionCount {
            attempts: 1,
            completed: 1,
            fallbacks: 0
        }
    );
}

#[test]
fn fusion_verification_counts_real_norm_paths_and_separate_arithmetic() -> Result<()> {
    for width in [16, 32, 1024] {
        let geometry = geometry(width);
        let target = tensor("target", width, 0);
        let input: Vec<_> = (0..width)
            .map(|index| [0.25, -0.5, -0.0, f32::from_bits(1), 1.0][index % 5])
            .collect();
        let weights = vec![0.75; width];
        let delta = vec![0.125; width];
        for arithmetic in [Arithmetic::Source, Arithmetic::PinnedSwift] {
            for unrelated in [false, true] {
                let mut transforms = Transforms {
                    block: if width >= 1024 { 256 } else { 16 },
                    weights: BTreeSet::from([target.name.clone()]),
                    signs: BTreeMap::from([(
                        width,
                        (0..width)
                            .map(|index| if index % 3 == 0 { -1.0 } else { 1.0 })
                            .collect(),
                    )]),
                    ..Transforms::default()
                };
                if unrelated {
                    transforms.weights.insert("unrelated.graph.node".into());
                }
                let worker_counts: &[usize] = if cfg!(feature = "parallel-rows") {
                    &[1, 4]
                } else {
                    &[1]
                };
                for &workers in worker_counts {
                    let mut projection = ProjectionScratch {
                        matvec: arithmetic.scratch(MatvecBackend::Scalar),
                        ..ProjectionScratch::default()
                    };
                    projection
                        .matvec
                        .set_row_parallelism(RowParallelism::new(workers)?);
                    let mut adapter_scratch = AdapterScratch::new(
                        MatvecBackend::Scalar,
                        matches!(arithmetic, Arithmetic::PinnedSwift),
                    );
                    let mut evaluation = Evaluation {
                        geometry: &geometry,
                        arithmetic,
                        transforms: &transforms,
                        position: 0,
                        frequencies_ready: false,
                        projection: &mut projection,
                        adapter: None,
                        adapter_scale: 0.0,
                        adapter_scratch: &mut adapter_scratch,
                        executor: None,
                    };
                    let mut expected = Vec::new();
                    let mut expected_rotated = Vec::new();
                    arithmetic.normalize(&input, &weights, geometry.epsilon, &mut expected)?;
                    transforms.transform_into_with_parallelism(
                        &target,
                        &expected,
                        &geometry,
                        &RowParallelism::default(),
                        &mut expected_rotated,
                    )?;
                    let mut actual = Vec::new();
                    evaluation.normalize_for_into(&input, &weights, &mut actual, &target)?;
                    let count = evaluation.projection.fusions.norm_hadamard;
                    compare(&actual, &expected, count);
                    compare(&evaluation.projection.transformed, &expected_rotated, count);

                    let mut expected_values = input.clone();
                    arithmetic.add_normalize(
                        &mut expected_values,
                        &delta,
                        &weights,
                        geometry.epsilon,
                        &mut expected,
                    )?;
                    transforms.transform_into_with_parallelism(
                        &target,
                        &expected,
                        &geometry,
                        &RowParallelism::default(),
                        &mut expected_rotated,
                    )?;
                    let mut values = input.clone();
                    evaluation.add_normalize_for_into(
                        &mut values,
                        &delta,
                        &weights,
                        &mut actual,
                        &target,
                    )?;
                    let count = evaluation.projection.fusions.add_norm_hadamard;
                    compare(&values, &expected_values, count);
                    compare(&actual, &expected, count);
                    compare(&evaluation.projection.transformed, &expected_rotated, count);
                }
            }
        }
    }
    Ok(())
}

#[test]
fn fusion_verification_swiglu_counts_execution_against_separate_operations() -> Result<()> {
    let width = 32;
    let geometry = geometry(width);
    let gate_weight = tensor("gate", width, 1);
    let up_weight = tensor("up", width, 2);
    let down_weight = tensor("down", width, 3);
    let input = vec![0.25; width];
    for arithmetic in [Arithmetic::Source, Arithmetic::PinnedSwift] {
        let transforms = Transforms {
            block: 16,
            weights: BTreeSet::from([down_weight.name.clone()]),
            ..Transforms::default()
        };
        let mut projection = ProjectionScratch {
            matvec: arithmetic.scratch(MatvecBackend::Scalar),
            ..ProjectionScratch::default()
        };
        let mut reference = ProjectionScratch {
            matvec: arithmetic.scratch(MatvecBackend::Scalar),
            ..ProjectionScratch::default()
        };
        let mut expected_gate = Vec::new();
        let mut expected_up = Vec::new();
        transforms.project_into(
            &gate_weight,
            &input,
            &geometry,
            &mut expected_gate,
            &mut reference,
        )?;
        transforms.project_into(
            &up_weight,
            &input,
            &geometry,
            &mut expected_up,
            &mut reference,
        )?;
        arithmetic.swiglu_in_place(&expected_gate, &mut expected_up)?;
        let mut expected = Vec::new();
        transforms.project_into(
            &down_weight,
            &expected_up,
            &geometry,
            &mut expected,
            &mut reference,
        )?;
        let mut adapter_scratch = AdapterScratch::new(
            MatvecBackend::Scalar,
            matches!(arithmetic, Arithmetic::PinnedSwift),
        );
        let mut evaluation = Evaluation {
            geometry: &geometry,
            arithmetic,
            transforms: &transforms,
            position: 0,
            frequencies_ready: false,
            projection: &mut projection,
            adapter: None,
            adapter_scale: 0.0,
            adapter_scratch: &mut adapter_scratch,
            executor: None,
        };
        let mut gate = Vec::new();
        let mut up = Vec::new();
        let mut actual = Vec::new();
        evaluation.project_ffn_swiglu_into(
            &gate_weight,
            &up_weight,
            &down_weight,
            &input,
            &mut gate,
            &mut up,
        )?;
        evaluation.project_into(&down_weight, &up, &mut actual)?;
        compare(
            &actual,
            &expected,
            evaluation.projection.fusions.swiglu_hadamard,
        );
    }
    Ok(())
}

#[test]
fn fusion_verification_rejects_successful_fallback_as_completed_fusion() -> Result<()> {
    let width = 32;
    let geometry = geometry(width);
    let target = tensor("target", width, 0);
    let input = vec![0.25; width];
    // Invalid signs make the fused preparation fail while ordinary RMSNorm
    // remains valid. Successful fallback must never count as a fusion.
    let transforms = Transforms {
        block: 16,
        weights: BTreeSet::from([target.name.clone()]),
        signs: BTreeMap::from([(width, vec![f32::NAN; width])]),
        ..Transforms::default()
    };
    for arithmetic in [Arithmetic::Source, Arithmetic::PinnedSwift] {
        let mut projection = ProjectionScratch {
            matvec: arithmetic.scratch(MatvecBackend::Scalar),
            ..ProjectionScratch::default()
        };
        let mut adapter_scratch = AdapterScratch::new(
            MatvecBackend::Scalar,
            matches!(arithmetic, Arithmetic::PinnedSwift),
        );
        let mut evaluation = Evaluation {
            geometry: &geometry,
            arithmetic,
            transforms: &transforms,
            position: 0,
            frequencies_ready: false,
            projection: &mut projection,
            adapter: None,
            adapter_scale: 0.0,
            adapter_scratch: &mut adapter_scratch,
            executor: None,
        };
        let mut actual = Vec::new();
        evaluation.normalize_for_into(&input, &vec![1.0; width], &mut actual, &target)?;
        assert_eq!(
            evaluation.projection.fusions.norm_hadamard,
            FusionCount {
                attempts: 1,
                completed: 0,
                fallbacks: 1
            }
        );
        assert_eq!(evaluation.projection.pre_transformed_len, None);
    }
    Ok(())
}
