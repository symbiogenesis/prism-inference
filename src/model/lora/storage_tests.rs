//! Ownership controls for exact, shared rank-one factor storage.

use std::{collections::BTreeMap, sync::Arc};

use bonsai_gguf::MetadataValue;

use super::{Model, RankOneAdapter, Result, Scratch, Tensor};
use crate::kernels::MatvecBackend;

const LINEAR: [&str; 2] = ["blk.0.ffn_down.weight", "blk.0.ssm_out.weight"];
const EMBEDDING: &str = "token_embd.weight";

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

fn fixture() -> (Model, Model) {
    let mut base = Model {
        identity: "base".into(),
        metadata: BTreeMap::from([("general.architecture".into(), MetadataValue::from("qwen35"))]),
        tensors: BTreeMap::new(),
    };
    let mut adapter = Model {
        identity: "adapter".into(),
        metadata: BTreeMap::from([
            ("general.architecture".into(), MetadataValue::from("qwen35")),
            ("general.type".into(), MetadataValue::from("adapter")),
            ("adapter.type".into(), MetadataValue::from("lora")),
            ("adapter.lora_alpha".into(), MetadataValue::from(1.0_f32)),
        ]),
        tensors: BTreeMap::new(),
    };
    let shared_b = tensor("unused", &[1, 2], &[1.0, -0.0]).data;
    for target in LINEAR.into_iter().chain([EMBEDDING]) {
        let (dimensions, a_dimensions, a_values): (&[usize], &[usize], &[f32]) =
            if target == EMBEDDING {
                (&[2, 3], &[1, 3], &[4097.0, -0.0, 1.0])
            } else {
                (&[2, 2], &[2, 1], &[4097.0, 0.0])
            };
        base.tensors.insert(
            target.into(),
            tensor(target, dimensions, &vec![0.0; dimensions.iter().product()]),
        );
        let a_name = format!("{target}.lora_a");
        adapter
            .tensors
            .insert(a_name.clone(), tensor(&a_name, a_dimensions, a_values));
        let b_name = format!("{target}.lora_b");
        adapter.tensors.insert(
            b_name.clone(),
            Tensor {
                name: b_name,
                dimensions: vec![1, 2],
                kind: 0,
                data: Arc::clone(&shared_b),
            },
        );
    }
    (base, adapter)
}

#[test]
fn rank_one_storage_releases_linear_raw_a_and_preserves_decoded_b_sharing() -> Result<()> {
    let (base, source) = fixture();
    let linear_a = LINEAR.map(|target| {
        source
            .tensors
            .get(&format!("{target}.lora_a"))
            .map(|tensor| Arc::downgrade(&tensor.data))
    });
    let embedding_a = source
        .tensors
        .get("token_embd.weight.lora_a")
        .map(|tensor| Arc::downgrade(&tensor.data))
        .ok_or("missing embedding fixture")?;
    let adapter = RankOneAdapter::from_model(source, &base)?;
    for weak in linear_a {
        assert!(weak.ok_or("missing linear fixture")?.upgrade().is_none());
    }
    assert!(embedding_a.upgrade().is_some());
    let factors = LINEAR
        .into_iter()
        .chain([EMBEDDING])
        .map(|target| adapter.factors.get(target).ok_or("missing factor"))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    assert_eq!(adapter.target_count(), 3);
    assert!(Arc::ptr_eq(&factors[0].b_values, &factors[1].b_values));
    assert!(Arc::ptr_eq(&factors[0].b_values, &factors[2].b_values));
    assert_eq!(factors[0].b_values[1].to_bits(), (-0.0_f32).to_bits());
    assert_eq!(factors[2].a_values, Vec::<f32>::new());

    // Arithmetic still rounds A's product before B scaling and the base add.
    let mut scratch = Scratch::new(MatvecBackend::Scalar, true);
    let mut linear = [-16_785_408.0, -0.0];
    adapter.linear(LINEAR[0], &[4097.0, 1.0], 1.0, &mut linear, &mut scratch)?;
    // The NEON correction adds +0 before scaling; the portable one-term f64
    // sum retains -0. Storage sharing must preserve each selected path's bits.
    let corrected_zero = if cfg!(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    )) {
        0.0_f32.to_bits()
    } else {
        (-0.0_f32).to_bits()
    };
    assert_eq!(
        linear.map(f32::to_bits),
        [0.0_f32.to_bits(), corrected_zero]
    );
    let mut embedding = [0.0, -0.0];
    adapter.embedding(0, 1.0, &mut embedding, &mut scratch)?;
    assert_eq!(
        embedding.map(f32::to_bits),
        [4097.0_f32.to_bits(), corrected_zero]
    );
    for scale in [0.0, -0.0] {
        let mut output = [-0.0, -0.0];
        adapter.embedding(usize::MAX, scale, &mut output, &mut scratch)?;
        assert_eq!(output.map(f32::to_bits), [(-0.0_f32).to_bits(); 2]);
    }
    let mut output = [3.0, 4.0];
    assert!(
        adapter
            .linear(EMBEDDING, &[1.0], 1.0, &mut output, &mut scratch)
            .is_err()
    );
    assert_eq!(output, [3.0, 4.0]);
    drop(adapter);
    assert!(embedding_a.upgrade().is_none());
    Ok(())
}

#[test]
fn rank_one_storage_keeps_distinct_b_bits_and_validates_shared_factors() -> Result<()> {
    let (base, mut source) = fixture();
    let name = format!("{}.lora_b", LINEAR[1]);
    source
        .tensors
        .insert(name.clone(), tensor(&name, &[1, 2], &[1.0, 0.0]));
    let adapter = RankOneAdapter::from_model(source, &base)?;
    let first = adapter
        .factors
        .get(LINEAR[0])
        .ok_or("missing first factor")?;
    let second = adapter
        .factors
        .get(LINEAR[1])
        .ok_or("missing second factor")?;
    assert!(!Arc::ptr_eq(&first.b_values, &second.b_values));
    assert_eq!(first.b_values[1].to_bits(), (-0.0_f32).to_bits());
    assert_eq!(second.b_values[1].to_bits(), 0.0_f32.to_bits());

    let (_, mut source) = fixture();
    source
        .tensors
        .get_mut(&name)
        .ok_or("missing B fixture")?
        .dimensions = vec![2, 1];
    assert!(RankOneAdapter::from_model(source, &base).is_err());
    Ok(())
}

#[test]
fn rank_one_edited_sparse_storage_releases_dense_a_and_binds_arithmetic() -> Result<()> {
    let (base, source) = fixture();
    let mut adapter = RankOneAdapter::from_model(source, &base)?;
    let dense = adapter.storage();
    assert_eq!(adapter.identity(), adapter.execution_identity());
    adapter.enable_sparse_a()?;
    assert_eq!(adapter.identity(), "adapter");
    assert_eq!(
        adapter.execution_identity(),
        "adapter:edited-sparse-a-u16-f64-v1"
    );
    let sparse = adapter.storage();
    assert_eq!(dense.linear_a_coefficients, 4);
    assert_eq!(sparse.linear_a_coefficients, 4);
    assert_eq!(sparse.stored_linear_a_coefficients, 2);
    assert_eq!(sparse.sparse_sites, 2);
    assert_eq!(sparse.linear_a_bytes, 8);
    assert_eq!(sparse.index_bytes, 4);
    assert_eq!(sparse.unique_b_bytes, 8);
    assert_eq!(sparse.embedding_a_bytes, 12);
    assert_eq!(sparse.total_payload_bytes, 32);
    assert_eq!(dense.total_payload_bytes - sparse.total_payload_bytes, 4);
    for target in LINEAR {
        let factors = adapter.factors.get(target).ok_or("missing sparse factor")?;
        assert_eq!(factors.a_values, [] as [f32; 0]);
        assert_eq!(factors.a_values.capacity(), 0);
        let compact = factors.sparse_a.as_ref().ok_or("missing compact A")?;
        assert_eq!(compact.indices, [0]);
        assert_eq!(compact.values, [4097.0]);
    }
    Ok(())
}

#[test]
fn rank_one_edited_sparse_dots_round_and_reject_nonfinite_omitted_inputs() -> Result<()> {
    let (base, source) = fixture();
    let mut adapter = RankOneAdapter::from_model(source, &base)?;
    adapter.enable_sparse_a()?;
    let mut scratch = Scratch::new(MatvecBackend::ExperimentalNeon, true);
    let mut output = [-16_785_408.0, 0.0];
    adapter.linear(LINEAR[0], &[4097.0, 1.0], 1.0, &mut output, &mut scratch)?;
    assert_eq!(output[0], 0.0);
    assert_eq!(scratch.rank, [16_785_408.0]);
    for invalid in [vec![1.0], vec![1.0, f32::NAN], vec![1.0, f32::INFINITY]] {
        let mut output = [3.0, 4.0];
        assert!(
            adapter
                .linear(LINEAR[0], &invalid, 1.0, &mut output, &mut scratch)
                .is_err()
        );
        assert_eq!(output, [3.0, 4.0]);
        assert_eq!(scratch.rank, [] as [f32; 0]);
    }
    let mut output = [-0.0, -0.0];
    adapter.linear(LINEAR[0], &[f32::NAN], 0.0, &mut output, &mut scratch)?;
    assert_eq!(output.map(f32::to_bits), [(-0.0_f32).to_bits(); 2]);
    Ok(())
}

#[test]
fn rank_one_edited_sparse_batch_matches_single_and_keeps_every_site() -> Result<()> {
    let (base, source) = fixture();
    let mut adapter = RankOneAdapter::from_model(source, &base)?;
    adapter.enable_sparse_a()?;
    let inputs = vec![
        vec![1.0, 1.0],
        vec![2.0, 0.0],
        vec![-2.0, 3.0],
        vec![0.0, 4.0],
        vec![3.0, 5.0],
    ];
    let mut single = vec![vec![0.0, 0.0]; inputs.len()];
    let mut batched = single.clone();
    let mut scratch = Scratch::default();
    for (input, output) in inputs.iter().zip(&mut single) {
        adapter.linear(LINEAR[0], input, 1.3, output, &mut scratch)?;
    }
    adapter.linear_tokens(LINEAR[0], &inputs, 1.3, &mut batched, &mut scratch)?;
    assert_eq!(single, batched);
    assert_eq!(adapter.target_count(), 3);

    let (_, mut source) = fixture();
    let name = format!("{}.lora_a", LINEAR[0]);
    source
        .tensors
        .insert(name.clone(), tensor(&name, &[2, 1], &[0.0, -0.0]));
    let mut empty = RankOneAdapter::from_model(source, &base)?;
    assert!(empty.enable_sparse_a().is_err());
    Ok(())
}

#[test]
fn rank_one_edited_sparse_dot_uses_nonzero_input_indices_in_order() -> Result<()> {
    let (mut base, mut source) = fixture();
    for target in LINEAR {
        base.tensors
            .insert(target.into(), tensor(target, &[6, 2], &[0.0; 12]));
        let name = format!("{target}.lora_a");
        source.tensors.insert(
            name.clone(),
            tensor(&name, &[6, 1], &[0.0, 2.0, -0.0, 0.0, -3.0, 0.0]),
        );
    }
    let mut adapter = RankOneAdapter::from_model(source, &base)?;
    adapter.enable_sparse_a()?;
    let compact = adapter
        .factors
        .get(LINEAR[0])
        .and_then(|factors| factors.sparse_a.as_ref())
        .ok_or("missing indexed sparse A")?;
    assert_eq!(compact.indices, [1, 4]);
    let mut output = [0.0, 0.0];
    let mut scratch = Scratch::default();
    adapter.linear(
        LINEAR[0],
        &[100.0, 4.0, 30.0, 0.0, 5.0, 0.0],
        1.0,
        &mut output,
        &mut scratch,
    )?;
    assert_eq!(scratch.rank, [-7.0]);
    assert_eq!(output[0], -7.0);
    Ok(())
}
