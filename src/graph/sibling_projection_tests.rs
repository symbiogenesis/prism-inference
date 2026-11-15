//! Real grouped-projection consumer controls without model or device timing.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use super::{Arithmetic, Geometry, ProjectionScratch, Transforms};
use crate::{
    Result,
    kernels::{MatvecBackend, Pq2ProjectionExecutor, Q8KBlock, RowParallelism},
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
        signs: BTreeMap::from([(
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

#[test]
fn q8_sibling_graph_consumer_joins_both_formats_with_exact_preparation() -> Result<()> {
    for (kind, columns) in [(142, 384), (143, 256)] {
        let geometry = Geometry {
            embedding: columns,
            feed_forward: columns,
            heads: 1,
            kv_heads: 1,
            head: columns,
            rope: columns,
            rope_sections: [columns / 2, 0, 0, 0],
            rope_base: 10_000.0,
            epsilon: 0.000_001,
            ssm_head: 128,
            ssm_keys: 2,
            ssm_values: 4,
            convolution: 1,
        };
        let mut weights = [weight("q", 33), weight("k", 9), weight("v", 5)];
        for weight in &mut weights {
            let mut block = [0x66; 34];
            block[..2].copy_from_slice(&0x3555_u16.to_le_bytes());
            let encoded = if kind == 143 {
                bonsai_ternary::pack_ptq(&block)?.to_vec()
            } else {
                block[2] = 0xe4;
                block.to_vec()
            };
            weight.kind = kind;
            weight.dimensions[0] = columns;
            weight.data = Arc::from(encoded.repeat(columns / 128 * weight.dimensions[1]));
        }
        let input: Vec<_> = (0..columns)
            .map(|index| [127.0, -128.0, 0.5001, -0.0, 63.5001, 1.5][index % 6])
            .collect();
        let transforms = Transforms::default();
        for arithmetic in [Arithmetic::Source, Arithmetic::PinnedSwift] {
            let mut reference = ProjectionScratch {
                matvec: arithmetic.scratch(MatvecBackend::Scalar),
                ..ProjectionScratch::default()
            };
            let mut expected: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::new());
            for (weight, output) in weights.iter().zip(&mut expected) {
                transforms.project_into(weight, &input, &geometry, output, &mut reference)?;
            }
            for backend in [MatvecBackend::Scalar, MatvecBackend::Auto] {
                let mut scratch = ProjectionScratch {
                    matvec: arithmetic.scratch(backend),
                    ..ProjectionScratch::default()
                };
                scratch.matvec.set_row_parallelism(RowParallelism::new(4)?);
                let mut actual: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::new());
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
                assert_eq!(scratch.matvec.preparation_count(), 1);
                assert_eq!(scratch.matvec.statistics().grouped_row_dispatches, 1);
            }
        }
    }
    Ok(())
}

struct PartialExecutorFixture {
    f32_claimed: BTreeSet<String>,
    f32_group_claimed: bool,
    q8k_claimed: BTreeSet<String>,
    q8k_expected: BTreeMap<String, Vec<f32>>,
    backend: MatvecBackend,
    calls: Vec<String>,
}

impl Pq2ProjectionExecutor for PartialExecutorFixture {
    fn identity(&self) -> &'static str {
        "test-partial-executor"
    }

    fn supports(&self, tensor: &Tensor) -> bool {
        self.q8k_claimed.contains(&tensor.name)
    }

    fn project(
        &mut self,
        tensor: &Tensor,
        activation: &[Q8KBlock],
        _fused: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        let cols = tensor.dimensions.first().copied().unwrap_or(0);
        assert_eq!(activation.len(), cols / 256);
        let expected = self
            .q8k_expected
            .get(&tensor.name)
            .ok_or_else(|| crate::invalid("missing q8k expected fixture output"))?;
        output.clear();
        output.extend_from_slice(expected);
        self.calls.push(format!("q8k:{}", tensor.name));
        Ok(())
    }

    fn supports_f32(&self, tensor: &Tensor) -> bool {
        self.f32_claimed.contains(&tensor.name)
    }

    fn project_f32(
        &mut self,
        tensor: &Tensor,
        activation: &[f32],
        pinned: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        let mut scratch = if pinned {
            crate::kernels::MatvecScratch::with_pinned_backend(self.backend)
        } else {
            crate::kernels::MatvecScratch::with_backend(self.backend)
        };
        scratch
            .prepare(tensor, activation)?
            .matvec_into(tensor, output)?;
        self.calls.push(format!("f32:{}", tensor.name));
        Ok(())
    }

    fn supports_f32_group(&self, tensors: &[&Tensor]) -> bool {
        self.f32_group_claimed
            && tensors
                .iter()
                .all(|tensor| self.f32_claimed.contains(&tensor.name))
    }

    fn project_f32_group(
        &mut self,
        tensors: &[&Tensor],
        activation: &[f32],
        pinned: bool,
        outputs: &mut [&mut Vec<f32>],
    ) -> Result<()> {
        for (tensor, output) in tensors.iter().zip(outputs.iter_mut()) {
            let mut scratch = if pinned {
                crate::kernels::MatvecScratch::with_pinned_backend(self.backend)
            } else {
                crate::kernels::MatvecScratch::with_backend(self.backend)
            };
            scratch
                .prepare(tensor, activation)?
                .matvec_into(tensor, output)?;
            self.calls.push(format!("f32_group:{}", tensor.name));
        }
        Ok(())
    }
}

fn bf16_weight(name: &str, cols: usize, rows: usize) -> Tensor {
    Tensor {
        name: name.to_owned(),
        dimensions: vec![cols, rows],
        kind: 30,
        data: Arc::from(
            [0x3f80_u16, 0xbfaa, 0x8000, 0x0080, 0x0001, 0x3e55, 0xc000]
                .into_iter()
                .cycle()
                .take(cols * rows)
                .flat_map(u16::to_le_bytes)
                .collect::<Vec<_>>(),
        ),
    }
}

fn partial_executor_test_context() -> (Geometry, Vec<f32>) {
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
    let input: Vec<_> = (0..512)
        .map(|index| [127.0, -127.0, 0.5001, -0.0, 63.5001, 1.5, -0.5001][index % 7])
        .collect();
    (geometry, input)
}

fn run_partial_executor_case<const N: usize>(
    transforms: &Transforms,
    weights: [&Tensor; N],
    (geometry, input): (&Geometry, &[f32]),
    arithmetic: Arithmetic,
    executor: &mut PartialExecutorFixture,
    expected: &[Vec<f32>; N],
    (expected_calls, expected_dispatches, expected_successes): (&[&str], u64, u64),
) -> Result<()> {
    let mut scratch = ProjectionScratch {
        matvec: arithmetic.scratch(executor.backend),
        ..ProjectionScratch::default()
    };
    scratch.matvec.set_row_parallelism(RowParallelism::new(4)?);
    let mut actual = std::array::from_fn::<_, N, _>(|_| Vec::new());
    let mut exec_opt: Option<&mut dyn Pq2ProjectionExecutor> = Some(executor);
    transforms.project_group_into_with_executor(
        weights,
        input,
        geometry,
        actual.each_mut(),
        &mut scratch,
        &mut exec_opt,
    )?;
    for (act, exp) in actual.iter().zip(expected) {
        assert!(
            act.iter()
                .map(|v| v.to_bits())
                .eq(exp.iter().map(|v| v.to_bits()))
        );
    }
    assert_eq!(executor.calls, expected_calls);
    assert_eq!(
        scratch.matvec.statistics().grouped_row_dispatches,
        expected_dispatches
    );
    assert_eq!(
        scratch.matvec.statistics().matrix_successes,
        expected_successes
    );
    Ok(())
}

#[test]
fn partial_executor_joins_unclaimed_siblings_and_splits_mixed() -> Result<()> {
    let (geometry, input) = partial_executor_test_context();
    let qkv_weights = [weight("q", 17), weight("k", 5), weight("v", 9)];
    let transforms_qkv = Transforms::default();
    for arithmetic in [Arithmetic::Source, Arithmetic::PinnedSwift] {
        for backend in [MatvecBackend::Scalar, MatvecBackend::Auto] {
            let mut ref_scratch = ProjectionScratch {
                matvec: arithmetic.scratch(backend),
                ..ProjectionScratch::default()
            };
            let mut expected = std::array::from_fn::<_, 3, _>(|_| Vec::new());
            for (w, out) in qkv_weights.iter().zip(&mut expected) {
                transforms_qkv.project_into(w, &input, &geometry, out, &mut ref_scratch)?;
            }

            // 1. All-CPU fallback group [q, k, v] while a partial executor is active must still join.
            let mut executor = PartialExecutorFixture {
                f32_claimed: BTreeSet::new(),
                f32_group_claimed: false,
                q8k_claimed: BTreeSet::new(),
                q8k_expected: BTreeMap::new(),
                backend,
                calls: Vec::new(),
            };
            run_partial_executor_case(
                &transforms_qkv,
                qkv_weights.each_ref(),
                (&geometry, &input),
                arithmetic,
                &mut executor,
                &expected,
                (&[], 1, 3),
            )?;

            // 2. Mixed ownership within same-kind group [q (claimed), k, v]: safe fallback.
            let mut mixed_exec = PartialExecutorFixture {
                f32_claimed: BTreeSet::from(["q".to_owned()]),
                f32_group_claimed: false,
                q8k_claimed: BTreeSet::new(),
                q8k_expected: BTreeMap::new(),
                backend,
                calls: Vec::new(),
            };
            run_partial_executor_case(
                &transforms_qkv,
                qkv_weights.each_ref(),
                (&geometry, &input),
                arithmetic,
                &mut mixed_exec,
                &expected,
                (&["f32:q"], 0, 2),
            )?;
        }
    }
    Ok(())
}

#[test]
fn partial_executor_joins_recurrent_bf16_siblings() -> Result<()> {
    let (geometry, input) = partial_executor_test_context();
    // 3. Recurrent [qkv, gate (PQ2 claimed), beta, alpha (BF16 unclaimed)] under both F32-group and Q8_K.
    let rec_weights = [
        weight("qkv", 17),
        weight("gate", 9),
        bf16_weight("beta", 512, 5),
        bf16_weight("alpha", 512, 33),
    ];
    let rec_transforms = Transforms {
        block: 256,
        weights: rec_weights[..2].iter().map(|w| w.name.clone()).collect(),
        ..Transforms::default()
    };
    for arithmetic in [Arithmetic::Source, Arithmetic::PinnedSwift] {
        for backend in [MatvecBackend::Scalar, MatvecBackend::Auto] {
            let mut ref_scratch = ProjectionScratch {
                matvec: arithmetic.scratch(backend),
                ..ProjectionScratch::default()
            };
            let mut expected = std::array::from_fn::<_, 4, _>(|_| Vec::new());
            for (w, out) in rec_weights.iter().zip(&mut expected) {
                rec_transforms.project_into(w, &input, &geometry, out, &mut ref_scratch)?;
            }

            // 3a: Partial executor claims [qkv, gate] via F32 group.
            let mut f32_exec = PartialExecutorFixture {
                f32_claimed: BTreeSet::from(["qkv".to_owned(), "gate".to_owned()]),
                f32_group_claimed: true,
                q8k_claimed: BTreeSet::new(),
                q8k_expected: BTreeMap::new(),
                backend,
                calls: Vec::new(),
            };
            run_partial_executor_case(
                &rec_transforms,
                rec_weights.each_ref(),
                (&geometry, &input),
                arithmetic,
                &mut f32_exec,
                &expected,
                (&["f32_group:qkv", "f32_group:gate"], 1, 2),
            )?;

            // 3b: Partial executor claims [qkv, gate] via Q8_K.
            let mut q8k_exec = PartialExecutorFixture {
                f32_claimed: BTreeSet::new(),
                f32_group_claimed: false,
                q8k_claimed: BTreeSet::from(["qkv".to_owned(), "gate".to_owned()]),
                q8k_expected: BTreeMap::from([
                    ("qkv".to_owned(), expected[0].clone()),
                    ("gate".to_owned(), expected[1].clone()),
                ]),
                backend,
                calls: Vec::new(),
            };
            run_partial_executor_case(
                &rec_transforms,
                rec_weights.each_ref(),
                (&geometry, &input),
                arithmetic,
                &mut q8k_exec,
                &expected,
                (&["q8k:qkv", "q8k:gate"], 1, 2),
            )?;
        }
    }
    Ok(())
}
