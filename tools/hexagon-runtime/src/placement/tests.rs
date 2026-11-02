//! Model-free controls for immutable placement and actual prepared CPU routing.

use std::{cell::Cell, collections::BTreeMap, rc::Rc};

use bonsai_prism_runtime::kernels::{MatvecBackend, MatvecScratch};

use super::*;

struct Executor {
    calls: Rc<Cell<usize>>,
    remaining: u64,
    fail: bool,
    count: Option<u64>,
    value: f32,
}

impl Pq2ProjectionExecutor for Executor {
    fn identity(&self) -> &'static str {
        "placement-control"
    }

    fn supports(&self, _: &Tensor) -> bool {
        true
    }

    fn supports_q8(&self, _: &Tensor) -> bool {
        true
    }

    fn operation_capacity(&self) -> Option<u64> {
        Some(self.remaining)
    }

    fn operation_count(&self, _: &Tensor) -> Option<u64> {
        self.count
    }

    fn project(
        &mut self,
        tensor: &Tensor,
        _: &[Q8KBlock],
        _: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        self.calls.set(self.calls.get() + 1);
        output.resize(tensor.dimensions[1], self.value);
        if self.fail {
            return Err(invalid("submitted placement control failure"));
        }
        self.remaining -= 2;
        Ok(())
    }

    fn project_q8(
        &mut self,
        tensor: &Tensor,
        _: &[Q8Block],
        pinned: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        self.project(tensor, &[], pinned, output)
    }
}

fn executor() -> Executor {
    Executor {
        calls: Rc::new(Cell::new(0)),
        remaining: 8,
        fail: false,
        count: Some(2),
        value: 42.0,
    }
}

fn tensor(name: &str, kind: u32, columns: usize) -> Tensor {
    let bytes = if kind == 143 { 28 } else { 34 };
    let mut data = vec![0x55; columns / 128 * bytes];
    for block in data.chunks_exact_mut(bytes) {
        let offset = if kind == 143 { 26 } else { 0 };
        block[offset..offset + 2].copy_from_slice(&0x3c00_u16.to_le_bytes());
    }
    Tensor {
        name: name.into(),
        kind,
        dimensions: vec![columns, 1],
        data: Arc::from(data),
    }
}

fn model() -> Model {
    let tensors = [
        tensor("blk.0.ffn_up.weight", 142, 256),
        tensor("blk.0.ffn_gate.weight", 142, 256),
        tensor("blk.0.ssm_out.weight", 142, 128),
        tensor("output.weight", 143, 256),
    ];
    Model {
        identity: "verified-placement-model".into(),
        metadata: BTreeMap::new(),
        tensors: tensors
            .into_iter()
            .map(|tensor| (tensor.name.clone(), tensor))
            .collect(),
    }
}

#[test]
fn placement_rejects_unknown_duplicate_empty_and_missing_bounded_counts() {
    let model = model();
    for names in [
        Vec::new(),
        vec!["missing"],
        vec!["blk.0.ffn_up.weight", "blk.0.ffn_up.weight"],
    ] {
        let mut inner = executor();
        assert!(PlacedProjectionExecutor::new(&mut inner, &model, &names).is_err());
        assert_eq!(inner.calls.get(), 0);
    }
    for count in [None, Some(0)] {
        let mut inner = executor();
        inner.count = count;
        assert!(
            PlacedProjectionExecutor::new(&mut inner, &model, &["blk.0.ffn_up.weight"]).is_err()
        );
        assert_eq!(inner.calls.get(), 0);
    }
}

#[test]
fn placement_identity_is_order_independent_and_binds_model_and_selection() -> Result<()> {
    let mut model = model();
    let identity = |model: &Model, names: &[&str]| -> Result<String> {
        let mut inner = executor();
        Ok(PlacedProjectionExecutor::new(&mut inner, model, names)?
            .identity()
            .into())
    };
    let both = ["blk.0.ffn_up.weight", "output.weight"];
    let expected = identity(&model, &both)?;
    assert_eq!(expected, identity(&model, &[both[1], both[0]])?);
    assert_ne!(expected, identity(&model, &[both[0]])?);
    model.identity.push('2');
    assert_ne!(expected, identity(&model, &both)?);
    Ok(())
}

#[test]
fn placement_matches_immutable_storage_geometry_and_original_activation_route() -> Result<()> {
    let model = model();
    let mut inner = executor();
    let names = [
        "blk.0.ffn_up.weight",
        "blk.0.ssm_out.weight",
        "output.weight",
    ];
    let placed = PlacedProjectionExecutor::new(&mut inner, &model, &names)?;
    for &name in &names {
        let tensor = &model.tensors[name];
        assert_eq!(placed.operation_count(tensor), Some(2));
        let q8k = tensor.kind == 142 && tensor.dimensions[0].is_multiple_of(256);
        assert_eq!(placed.supports(tensor), q8k);
        assert_eq!(placed.supports_q8(tensor), !q8k);
        assert_eq!(placed.supports(&tensor.clone()), q8k);
        let mut changed = tensor.clone();
        changed.data = Arc::from(tensor.data.to_vec());
        assert!(!placed.supports(&changed));
        assert!(!placed.supports_q8(&changed));
        changed = tensor.clone();
        changed.dimensions[1] += 1;
        assert_eq!(placed.operation_count(&changed), None);
    }
    let unselected = &model.tensors["blk.0.ffn_gate.weight"];
    assert!(!placed.supports(unselected));
    assert_eq!(placed.operation_count(unselected), None);
    assert_eq!(placed.operation_capacity(), Some(8));
    Ok(())
}

#[test]
fn prepared_projection_routes_unselected_work_to_cpu_before_any_submission() -> Result<()> {
    let model = model();
    let mut inner = executor();
    let calls = inner.calls.clone();
    let mut placed = PlacedProjectionExecutor::new(&mut inner, &model, &["blk.0.ffn_up.weight"])?;
    let selected = &model.tensors["blk.0.ffn_up.weight"];
    let unselected = &model.tensors["blk.0.ffn_gate.weight"];
    let input = vec![1.0; 256];
    let mut scratch = MatvecScratch::with_backend(MatvecBackend::Scalar);
    let mut prepared = scratch.prepare(selected, &input)?;
    let mut output = vec![99.0];
    prepared.matvec_with_executor(unselected, &mut output, &mut placed)?;
    assert_eq!(output, [0.0]);
    assert_eq!(calls.get(), 0);
    prepared.matvec_with_executor(selected, &mut output, &mut placed)?;
    assert_eq!(output, [42.0]);
    assert_eq!(calls.get(), 1);
    assert_eq!(placed.operation_capacity(), Some(6));
    Ok(())
}

#[test]
fn submitted_failure_clears_output_and_permanently_closes_placement() -> Result<()> {
    let model = model();
    for name in ["blk.0.ffn_up.weight", "output.weight"] {
        let mut inner = executor();
        inner.fail = true;
        let calls = inner.calls.clone();
        let mut placed = PlacedProjectionExecutor::new(&mut inner, &model, &[name])?;
        let tensor = &model.tensors[name];
        let mut output = vec![99.0];
        for _ in 0..2 {
            let result = if tensor.kind == 143 {
                placed.project_q8(tensor, &[], true, &mut output)
            } else {
                placed.project(tensor, &[], true, &mut output)
            };
            assert!(result.is_err());
            assert_eq!(output, Vec::<f32>::new());
            assert!(placed.is_faulted());
            assert_eq!(placed.operation_capacity(), Some(0));
        }
        assert_eq!(calls.get(), 1);
    }
    Ok(())
}

fn graph_model() -> Model {
    use bonsai_gguf::MetadataValue;

    let mut model = Model {
        identity: "partial-graph-placement-control".into(),
        metadata: BTreeMap::new(),
        tensors: BTreeMap::new(),
    };
    model
        .metadata
        .insert("general.architecture".into(), "qwen35".into());
    for (key, value) in [
        ("block_count", 1_u64),
        ("context_length", 16),
        ("embedding_length", 256),
        ("feed_forward_length", 2),
        ("attention.head_count", 1),
        ("attention.head_count_kv", 1),
        ("attention.key_length", 2),
        ("attention.value_length", 2),
        ("rope.dimension_count", 2),
        ("ssm.state_size", 2),
        ("ssm.group_count", 1),
        ("ssm.time_step_rank", 1),
        ("ssm.inner_size", 2),
        ("ssm.conv_kernel", 2),
        ("full_attention_interval", 1),
    ] {
        model.metadata.insert(format!("qwen35.{key}"), value.into());
    }
    model.metadata.insert(
        "qwen35.attention.layer_norm_rms_epsilon".into(),
        MetadataValue::from(0.000_001_f32),
    );
    model.metadata.insert(
        "qwen35.rope.freq_base".into(),
        MetadataValue::from(10000.0_f32),
    );
    model.metadata.insert(
        "qwen35.rope.dimension_sections".into(),
        vec![1_u64, 0, 0, 0].into(),
    );
    model.metadata.insert(
        "qwen35.attention.recurrent_layers".into(),
        vec![false].into(),
    );
    for (name, dimensions, value) in [
        ("token_embd.weight", vec![256, 2], 1.0_f32),
        ("output.weight", vec![256, 2], 0.01),
        ("output_norm.weight", vec![256], 1.0),
        ("blk.0.attn_norm.weight", vec![256], 1.0),
        ("blk.0.post_attention_norm.weight", vec![256], 1.0),
        ("blk.0.attn_q.weight", vec![256, 4], 0.0),
        ("blk.0.attn_k.weight", vec![256, 2], 0.0),
        ("blk.0.attn_v.weight", vec![256, 2], 0.01),
        ("blk.0.attn_output.weight", vec![2, 256], 0.01),
        ("blk.0.attn_q_norm.weight", vec![2], 1.0),
        ("blk.0.attn_k_norm.weight", vec![2], 1.0),
        ("blk.0.ffn_down.weight", vec![2, 256], 0.0),
    ] {
        let tensor = Tensor {
            name: name.into(),
            kind: 0,
            data: Arc::from(value.to_le_bytes().repeat(dimensions.iter().product())),
            dimensions,
        };
        model.tensors.insert(name.into(), tensor);
    }
    for name in ["blk.0.ffn_up.weight", "blk.0.ffn_gate.weight"] {
        let mut tensor = tensor(name, 142, 256);
        tensor.dimensions[1] = 2;
        tensor.data = Arc::from(tensor.data.repeat(2));
        model.tensors.insert(name.into(), tensor);
    }
    model
}

#[test]
fn actual_graph_admits_only_selected_work_and_preserves_logits_and_state() -> Result<()> {
    use bonsai_prism_runtime::{Decoder, graph::Qwen35};

    let bits = |graph: &Qwen35| -> Result<Vec<u32>> {
        let mut bits = Vec::new();
        graph.visit_state(&mut |_, _, values| {
            bits.extend(values.iter().map(|value| value.to_bits()));
            Ok(())
        })?;
        Ok(bits)
    };
    let model = graph_model();
    let mut inner = executor();
    inner.remaining = 2;
    inner.value = 0.0;
    let calls = inner.calls.clone();
    let mut placed = PlacedProjectionExecutor::new(&mut inner, &model, &["blk.0.ffn_up.weight"])?;
    let mut graph = Qwen35::new(model, 16)?.with_projection_executor(&mut placed)?;
    let mut control = Qwen35::new(graph_model(), 16)?;
    let before = bits(graph.decoder())?;
    assert!(graph.admit_work(0, 2).is_err());
    assert_eq!(calls.get(), 0);
    assert_eq!(bits(graph.decoder())?, before);
    assert!(!graph.is_faulted());
    assert!(graph.admit_work(0, 1).is_ok());
    assert_eq!(
        graph
            .evaluate(0)?
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        control
            .evaluate(0)?
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(calls.get(), 1);
    assert_eq!(bits(graph.decoder())?, bits(&control)?);
    let before = bits(graph.decoder())?;
    assert!(graph.evaluate(1).is_err());
    assert_eq!(calls.get(), 1);
    assert_eq!(bits(graph.decoder())?, before);
    assert!(!graph.is_faulted());
    graph.reset()?;
    assert!(graph.admit_work(0, 1).is_err());
    Ok(())
}
