//! Adapter FFN consumer controls against the original separate operations.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use bonsai_gguf::MetadataValue;

use super::super::ProjectionScratch;
use super::{
    AdapterScratch, Arithmetic, Evaluation, Geometry, MatvecBackend, ProjectionTile,
    RankOneAdapter, RowParallelism, Tensor, TileReuse, Transforms, project_tile,
    project_tile_ffn_swiglu, try_project_tile_ffn_chain,
};
use crate::{Result, model::Model};

fn tensor(name: &str, dimensions: &[usize], values: &[f32]) -> Tensor {
    Tensor {
        name: name.into(),
        dimensions: dimensions.to_vec(),
        kind: 0,
        data: Arc::<[u8]>::from(
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .into(),
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

struct TileResult {
    up: Vec<Vec<f32>>,
    outputs: Vec<Vec<f32>>,
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
        // Match Qwen35 graph construction: arithmetic controls both the
        // projection reduction contract and the adapter's original-basis dot.
        // A default scratch would silently compare source-order F32 decode
        // against pinned FMA tiled projection for aligned down weights.
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
            geometry: &self.geometry,
            arithmetic,
            transforms: &self.transforms,
            position: 0,
            mrope_positions: [0; 4],
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
                    .then_some(up.len()),
                "decode reuse: width={}, block={}, workers={workers}, scale={scale}",
                self.geometry.feed_forward,
                self.transforms.block,
            );
            assert_eq!(
                evaluation
                    .projection
                    .fusions
                    .adapter_swiglu_hadamard
                    .completed,
                u64::from(self.transforms.can_block_hadamard(up.len())),
                "adapter fusion must identify the branch actually completed"
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

    fn tile(
        &self,
        arithmetic: Arithmetic,
        workers: usize,
        scale: f32,
        inputs: &[Vec<f32>],
        scenario: &str,
    ) -> Result<TileResult> {
        let mut projector = ProjectionTile::with_backend(MatvecBackend::Scalar);
        projector.set_row_parallelism(RowParallelism::new(workers)?);
        let mut projection_scratch = arithmetic.scratch(MatvecBackend::Scalar);
        let mut adapter_scratch = AdapterScratch::new(
            MatvecBackend::Scalar,
            matches!(arithmetic, Arithmetic::PinnedSwift),
        );
        let mut gate = vec![Vec::new(); inputs.len()];
        let mut up = vec![Vec::new(); inputs.len()];
        let mut transformed = vec![Vec::new(); inputs.len()];
        let reuse = project_tile_ffn_swiglu(
            &self.transforms,
            &self.geometry,
            arithmetic,
            Some(&self.adapter),
            scale,
            &mut adapter_scratch,
            &mut projection_scratch,
            TileReuse::default(),
            &self.gate,
            &self.up,
            &self.down,
            inputs,
            &mut gate,
            &mut up,
            &mut transformed,
            &mut projector,
            None,
            &|| Ok(()),
        )?;
        assert_eq!(
            reuse.basis.is_some(),
            self.transforms.block >= 16,
            "{scenario}: tile reuse"
        );
        let mut outputs = vec![Vec::new(); inputs.len()];
        let _ = project_tile(
            &self.transforms,
            &self.geometry,
            arithmetic,
            Some(&self.adapter),
            scale,
            &mut adapter_scratch,
            &mut projection_scratch,
            reuse,
            &self.down,
            &up,
            &mut outputs,
            &mut transformed,
            &mut projector,
            None,
            &|| Ok(()),
        )?;
        Ok(TileResult { up, outputs })
    }
}

fn same_bits(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: result lengths");
    let mismatch =
        actual
            .iter()
            .zip(expected)
            .enumerate()
            .find_map(|(index, (actual, expected))| {
                (actual.to_bits() != expected.to_bits()).then_some((
                    index,
                    actual.to_bits(),
                    expected.to_bits(),
                ))
            });
    assert!(
        mismatch.is_none(),
        "{context}: first differing (index, actual bits, expected bits) = {mismatch:?}"
    );
}

#[test]
fn adapter_ffn_consumers_retain_original_basis_and_exact_down_correction() -> Result<()> {
    let workers = if cfg!(feature = "parallel-rows") {
        4
    } else {
        1
    };
    for (arithmetic, name) in [
        (Arithmetic::Source, "source"),
        (Arithmetic::PinnedSwift, "pinned"),
    ] {
        for (width, block) in [(32, 16), (4096, 256), (32, 8)] {
            let fixture = Fixture::new(width, block)?;
            for scale in [0.0, 1.3] {
                let scenario = format!(
                    "{name}, width={width}, block={block}, workers={workers}, scale={scale}"
                );
                let inputs = vec![vec![0.75, -0.5], vec![-0.25, 1.0], vec![0.0, -0.0]];
                let expected = inputs
                    .iter()
                    .map(|input| fixture.decode(arithmetic, workers, scale, input, false))
                    .collect::<Result<Vec<_>>>()?;
                for (token, (input, (expected_up, expected_output))) in
                    inputs.iter().zip(&expected).enumerate()
                {
                    let (up, output) = fixture.decode(arithmetic, workers, scale, input, true)?;
                    same_bits(
                        &up,
                        expected_up,
                        &format!("{scenario}: decode token {token} up"),
                    );
                    same_bits(
                        &output,
                        expected_output,
                        &format!("{scenario}: decode token {token} down output"),
                    );
                }

                let TileResult { up, outputs } =
                    fixture.tile(arithmetic, workers, scale, &inputs, &scenario)?;
                for (token, ((up, output), (expected_up, expected_output))) in
                    up.iter().zip(&outputs).zip(&expected).enumerate()
                {
                    same_bits(
                        up,
                        expected_up,
                        &format!("{scenario}: tile token {token} up"),
                    );
                    same_bits(
                        output,
                        expected_output,
                        &format!("{scenario}: tile token {token} down output"),
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn adapter_ffn_preparation_rejects_invalid_geometry_and_never_replays_failed_swiglu() -> Result<()>
{
    for (arithmetic, name) in [
        (Arithmetic::Source, "source"),
        (Arithmetic::PinnedSwift, "pinned"),
    ] {
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
            same_bits(
                &up,
                &original_up,
                &format!("{name}: invalid block {block} up"),
            );
            assert_eq!(transformed.len(), 0);
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
        same_bits(
            &up[..16],
            &expected,
            &format!("{name}: failed first block up"),
        );
        same_bits(
            &up[16..],
            &original_up[16..],
            &format!("{name}: untouched later block up"),
        );
        assert_eq!(transformed.len(), 0);
    }
    Ok(())
}

fn pq2_tensor(name: &str, seed: u8) -> Tensor {
    let mut block = [0_u8; 34];
    block[..2].copy_from_slice(&0x3800_u16.to_le_bytes());
    for (index, byte) in block[2..].iter_mut().enumerate() {
        *byte = match (usize::from(seed) + index) % 4 {
            0 => 0xaa,
            1 => 0x00,
            2 => 0x56,
            _ => 0x91,
        };
    }
    Tensor {
        name: name.into(),
        dimensions: vec![128, 128],
        kind: 142,
        data: Arc::<[u8]>::from(block.repeat(128)).into(),
    }
}

fn ffn_transforms(down_name: &str) -> Transforms {
    Transforms {
        block: 32,
        weights: BTreeSet::from([down_name.to_owned()]),
        signs: BTreeMap::from([(128, vec![1.0; 128])]),
        ..Transforms::default()
    }
}

const fn as_executor(
    exec: &mut RecordingFfnExecutor,
) -> &mut dyn crate::kernels::Pq2ProjectionExecutor {
    exec
}

struct RecordingFfnExecutor {
    arithmetic: Arithmetic,
    transforms: Transforms,
    geometry: Geometry,
    chain_calls: usize,
    swiglu_calls: usize,
    project_calls: usize,
}

impl RecordingFfnExecutor {
    fn cpu_matvec(&self, tensor: &Tensor, activation: &[f32], output: &mut Vec<f32>) -> Result<()> {
        let mut scratch = self.arithmetic.scratch(MatvecBackend::Scalar);
        scratch
            .prepare(tensor, activation)?
            .matvec_into(tensor, output)
    }
}

impl crate::kernels::Pq2ProjectionExecutor for RecordingFfnExecutor {
    fn identity(&self) -> &'static str {
        "recording-ffn-test-executor"
    }

    fn supports(&self, _tensor: &Tensor) -> bool {
        false
    }

    fn project(
        &mut self,
        _tensor: &Tensor,
        _activation: &[crate::kernels::Q8KBlock],
        _fused: bool,
        _output: &mut Vec<f32>,
    ) -> Result<()> {
        Err(crate::invalid("unexpected quantized submission"))
    }

    fn supports_f32(&self, tensor: &Tensor) -> bool {
        tensor.kind == 142
    }

    fn project_f32(
        &mut self,
        tensor: &Tensor,
        activation: &[f32],
        _pinned: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        self.project_calls += 1;
        self.cpu_matvec(tensor, activation, output)
    }

    fn supports_f32_ffn_swiglu(&self, gate: &Tensor, up: &Tensor) -> bool {
        gate.kind == 142 && up.kind == 142
    }

    fn project_f32_ffn_swiglu(
        &mut self,
        gate: &Tensor,
        up: &Tensor,
        activation: &[f32],
        _pinned: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        self.swiglu_calls += 1;
        let mut gate_out = Vec::new();
        self.cpu_matvec(gate, activation, &mut gate_out)?;
        self.cpu_matvec(up, activation, output)?;
        self.arithmetic.swiglu_in_place(&gate_out, output)
    }

    fn supports_f32_ffn_chain(&self, tensors: [&Tensor; 3]) -> bool {
        tensors.iter().all(|tensor| tensor.kind == 142)
    }

    fn project_f32_ffn_chain(
        &mut self,
        [gate, up, down]: [&Tensor; 3],
        activation: &[f32],
        residual: Option<&[f32]>,
        hadamard_block_size: Option<u32>,
        pinned: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        self.chain_calls += 1;
        let mut mid = Vec::new();
        let mut gate_out = Vec::new();
        self.cpu_matvec(gate, activation, &mut gate_out)?;
        self.cpu_matvec(up, activation, &mut mid)?;
        self.arithmetic.swiglu_in_place(&gate_out, &mut mid)?;
        let _ = (hadamard_block_size, pinned);
        let mut rotated = Vec::new();
        let down_input = if self.transforms.basis(down).0 {
            self.transforms
                .transform_into(down, &mid, &self.geometry, &mut rotated)?;
            rotated.as_slice()
        } else {
            mid.as_slice()
        };
        self.cpu_matvec(down, down_input, output)?;
        if let Some(residual) = residual {
            super::super::add_finite(output, residual, "test ffn chain")?;
        }
        Ok(())
    }
}

struct Pq2FfnFixture {
    geometry: Geometry,
    transforms: Transforms,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
    down_adapter: RankOneAdapter,
    attn_adapter: RankOneAdapter,
}

impl Pq2FfnFixture {
    fn new() -> Result<Self> {
        let gate = pq2_tensor("blk.0.ffn_gate.weight", 0);
        let up = pq2_tensor("blk.0.ffn_up.weight", 1);
        let down = pq2_tensor("blk.0.ffn_down.weight", 2);
        let attn_out = pq2_tensor("blk.0.attn_output.weight", 3);
        let adapter_for = |target: &Tensor| -> Result<RankOneAdapter> {
            let base = Model {
                metadata: BTreeMap::from([(
                    "general.architecture".into(),
                    MetadataValue::from("qwen35"),
                )]),
                tensors: BTreeMap::from([(target.name.clone(), target.clone())]),
                identity: "pq2-adapter-base".into(),
            };
            let a_vals: Vec<f32> = (0..128)
                .map(|i| if i % 3 == 0 { 0.125 } else { -0.0625 })
                .collect();
            let b_vals: Vec<f32> = (0..128)
                .map(|i| if i % 2 == 0 { 0.25 } else { -0.125 })
                .collect();
            let a = tensor(&format!("{}.lora_a", target.name), &[128, 1], &a_vals);
            let b = tensor(&format!("{}.lora_b", target.name), &[1, 128], &b_vals);
            let factors = Model {
                metadata: BTreeMap::from([
                    ("general.architecture".into(), MetadataValue::from("qwen35")),
                    ("general.type".into(), MetadataValue::from("adapter")),
                    ("adapter.type".into(), MetadataValue::from("lora")),
                    ("adapter.lora_alpha".into(), MetadataValue::from(1.0_f32)),
                ]),
                tensors: BTreeMap::from([(a.name.clone(), a), (b.name.clone(), b)]),
                identity: "pq2-adapter-factors".into(),
            };
            RankOneAdapter::from_model(factors, &base)
        };
        Ok(Self {
            geometry: Geometry {
                embedding: 128,
                feed_forward: 128,
                heads: 1,
                kv_heads: 1,
                head: 128,
                rope: 64,
                rope_sections: [32, 0, 0, 0],
                rope_base: 10_000.0,
                epsilon: 0.000_001,
                ssm_head: 128,
                ssm_keys: 1,
                ssm_values: 1,
                convolution: 4,
            },
            transforms: ffn_transforms(&down.name),
            down_adapter: adapter_for(&down)?,
            attn_adapter: adapter_for(&attn_out)?,
            gate,
            up,
            down,
        })
    }

    fn executor(&self, arithmetic: Arithmetic) -> RecordingFfnExecutor {
        RecordingFfnExecutor {
            arithmetic,
            transforms: ffn_transforms(&self.down.name),
            geometry: self.geometry.clone(),
            chain_calls: 0,
            swiglu_calls: 0,
            project_calls: 0,
        }
    }

    fn run_decode(
        &self,
        arithmetic: Arithmetic,
        adapter: Option<&RankOneAdapter>,
        scale: f32,
        input: &[f32],
        residual: &[f32],
        executor: Option<&mut RecordingFfnExecutor>,
    ) -> Result<(Vec<f32>, crate::graph::fusion::FusionStatistics)> {
        let mut projection = ProjectionScratch {
            matvec: arithmetic.scratch(MatvecBackend::Scalar),
            ..ProjectionScratch::default()
        };
        let mut adapter_scratch = AdapterScratch::new(
            MatvecBackend::Scalar,
            matches!(arithmetic, Arithmetic::PinnedSwift),
        );
        let mut evaluation = Evaluation {
            geometry: &self.geometry,
            arithmetic,
            transforms: &self.transforms,
            position: 0,
            mrope_positions: [0; 4],
            frequencies_ready: false,
            projection: &mut projection,
            adapter,
            adapter_scale: scale,
            adapter_scratch: &mut adapter_scratch,
            executor: executor.map(as_executor),
        };
        let mut output = Vec::new();
        if !evaluation.try_project_ffn_chain_into(
            &self.gate,
            &self.up,
            &self.down,
            input,
            Some(residual),
            &mut output,
        )? {
            let mut gate_out = Vec::new();
            let mut up_out = Vec::new();
            evaluation.project_ffn_swiglu_into(
                &self.gate,
                &self.up,
                &self.down,
                input,
                &mut gate_out,
                &mut up_out,
            )?;
            evaluation.project_into(&self.down, &up_out, &mut output)?;
            output
                .iter_mut()
                .zip(residual)
                .for_each(|(out, res)| *out += *res);
        }
        let stats = evaluation.projection.fusions;
        Ok((output, stats))
    }

    fn run_tile(
        &self,
        arithmetic: Arithmetic,
        adapter: Option<&RankOneAdapter>,
        scale: f32,
        inputs: &[Vec<f32>],
        residuals: &[Vec<f32>],
        mut executor: Option<&mut RecordingFfnExecutor>,
    ) -> Result<Vec<Vec<f32>>> {
        let mut projector = ProjectionTile::with_backend(MatvecBackend::Scalar);
        let mut projection_scratch = arithmetic.scratch(MatvecBackend::Scalar);
        let mut adapter_scratch = AdapterScratch::new(
            MatvecBackend::Scalar,
            matches!(arithmetic, Arithmetic::PinnedSwift),
        );
        let mut outputs = vec![Vec::new(); inputs.len()];
        let mut transformed = vec![Vec::new(); inputs.len()];
        let mut gate_out = vec![Vec::new(); inputs.len()];
        let mut up_out = vec![Vec::new(); inputs.len()];
        if !try_project_tile_ffn_chain(
            &self.transforms,
            &self.geometry,
            arithmetic,
            adapter,
            scale,
            TileReuse::default(),
            [&self.gate, &self.up, &self.down],
            inputs,
            Some(residuals),
            &mut outputs,
            &mut transformed,
            &projector,
            executor.as_deref_mut().map(as_executor),
            &|| Ok(()),
        )? {
            let reuse = project_tile_ffn_swiglu(
                &self.transforms,
                &self.geometry,
                arithmetic,
                adapter,
                scale,
                &mut adapter_scratch,
                &mut projection_scratch,
                TileReuse::default(),
                &self.gate,
                &self.up,
                &self.down,
                inputs,
                &mut gate_out,
                &mut up_out,
                &mut transformed,
                &mut projector,
                executor.as_deref_mut().map(as_executor),
                &|| Ok(()),
            )?;
            let _ = project_tile(
                &self.transforms,
                &self.geometry,
                arithmetic,
                adapter,
                scale,
                &mut adapter_scratch,
                &mut projection_scratch,
                reuse,
                &self.down,
                &up_out,
                &mut outputs,
                &mut transformed,
                &mut projector,
                executor.map(as_executor),
                &|| Ok(()),
            )?;
            for (out_row, res_row) in outputs.iter_mut().zip(residuals) {
                out_row
                    .iter_mut()
                    .zip(res_row)
                    .for_each(|(out, res)| *out += *res);
            }
        }
        Ok(outputs)
    }
}

fn verify_arithmetic_external_ffn_lora(
    fixture: &Pq2FfnFixture,
    arithmetic: Arithmetic,
    inputs: &[Vec<f32>],
    residuals: &[Vec<f32>],
) -> Result<()> {
    let (expected_zero, _) = fixture.run_decode(
        arithmetic,
        Some(&fixture.down_adapter),
        0.0,
        &inputs[0],
        &residuals[0],
        None,
    )?;
    let mut exec = fixture.executor(arithmetic);
    let (actual_zero, summary_zero) = fixture.run_decode(
        arithmetic,
        Some(&fixture.down_adapter),
        0.0,
        &inputs[0],
        &residuals[0],
        Some(&mut exec),
    )?;
    same_bits(&actual_zero, &expected_zero, "decode scale=0 chain");
    assert_eq!(summary_zero.external_ffn_chain.completed, 1);
    assert_eq!(exec.chain_calls, 1);

    let mut tile_exec = fixture.executor(arithmetic);
    let tile_zero = fixture.run_tile(
        arithmetic,
        Some(&fixture.down_adapter),
        0.0,
        inputs,
        residuals,
        Some(&mut tile_exec),
    )?;
    assert_eq!(tile_exec.chain_calls, 2);
    same_bits(&tile_zero[0], &expected_zero, "tile scale=0 chain");

    let (expected_down, _) = fixture.run_decode(
        arithmetic,
        Some(&fixture.down_adapter),
        1.3,
        &inputs[0],
        &residuals[0],
        None,
    )?;
    let mut exec = fixture.executor(arithmetic);
    let (actual_down, summary_down) = fixture.run_decode(
        arithmetic,
        Some(&fixture.down_adapter),
        1.3,
        &inputs[0],
        &residuals[0],
        Some(&mut exec),
    )?;
    same_bits(&actual_down, &expected_down, "decode down-only swiglu");
    assert_eq!(summary_down.external_ffn_chain.completed, 0);
    assert_eq!(summary_down.external_ffn_swiglu.completed, 1);
    assert_eq!(exec.chain_calls, 0);
    assert_eq!(exec.swiglu_calls, 1);
    assert_eq!(exec.project_calls, 1);

    let mut tile_exec = fixture.executor(arithmetic);
    let tile_down = fixture.run_tile(
        arithmetic,
        Some(&fixture.down_adapter),
        1.3,
        inputs,
        residuals,
        Some(&mut tile_exec),
    )?;
    assert_eq!(tile_exec.chain_calls, 0);
    assert_eq!(tile_exec.swiglu_calls, 2);
    assert_eq!(tile_exec.project_calls, 2);
    same_bits(&tile_down[0], &expected_down, "tile down-only swiglu");

    // Non-FFN adapter (targeting attn_output only) unlocks full FFN chain even at nonzero scale.
    let (expected_attn, _) = fixture.run_decode(
        arithmetic,
        Some(&fixture.attn_adapter),
        1.3,
        &inputs[0],
        &residuals[0],
        None,
    )?;
    let mut exec = fixture.executor(arithmetic);
    let (actual_attn, summary_attn) = fixture.run_decode(
        arithmetic,
        Some(&fixture.attn_adapter),
        1.3,
        &inputs[0],
        &residuals[0],
        Some(&mut exec),
    )?;
    same_bits(&actual_attn, &expected_attn, "decode attn-only lora chain");
    assert_eq!(summary_attn.external_ffn_chain.completed, 1);
    assert_eq!(exec.chain_calls, 1);
    Ok(())
}

#[test]
fn external_ffn_chain_and_swiglu_unlock_for_inactive_and_down_only_lora() -> Result<()> {
    let fixture = Pq2FfnFixture::new()?;
    let input_a: Vec<f32> = (0..128)
        .map(|i| if i % 2 == 0 { 0.375 } else { -0.25 })
        .collect();
    let input_b: Vec<f32> = (0..128)
        .map(|i| if i % 5 == 0 { -0.5 } else { 0.125 })
        .collect();
    let inputs = vec![input_a, input_b];
    let residuals = vec![vec![0.125; 128], vec![-0.25; 128]];
    let untargeted_gate = pq2_tensor("blk.1.ffn_gate.weight", 0);
    let untargeted_up = pq2_tensor("blk.1.ffn_up.weight", 1);
    let untargeted_down = pq2_tensor("blk.1.ffn_down.weight", 2);

    for arithmetic in [Arithmetic::Source, Arithmetic::PinnedSwift] {
        verify_arithmetic_external_ffn_lora(&fixture, arithmetic, &inputs, &residuals)?;

        let mut proj = ProjectionScratch {
            matvec: arithmetic.scratch(MatvecBackend::Scalar),
            ..ProjectionScratch::default()
        };
        let mut adapter_scratch = AdapterScratch::new(
            MatvecBackend::Scalar,
            matches!(arithmetic, Arithmetic::PinnedSwift),
        );
        let mut untargeted_exec = fixture.executor(arithmetic);
        let mut eval = Evaluation {
            geometry: &fixture.geometry,
            arithmetic,
            transforms: &fixture.transforms,
            position: 0,
            mrope_positions: [0; 4],
            frequencies_ready: false,
            projection: &mut proj,
            adapter: Some(&fixture.down_adapter),
            adapter_scale: 1.3,
            adapter_scratch: &mut adapter_scratch,
            executor: Some(&mut untargeted_exec),
        };
        let mut out = Vec::new();
        assert!(eval.try_project_ffn_chain_into(
            &untargeted_gate,
            &untargeted_up,
            &untargeted_down,
            &inputs[0],
            Some(&residuals[0]),
            &mut out,
        )?);
        assert_eq!(untargeted_exec.chain_calls, 1);
    }
    Ok(())
}
