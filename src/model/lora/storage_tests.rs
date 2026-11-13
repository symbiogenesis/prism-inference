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
    assert!(factors[2].a_values.is_empty());

    // Arithmetic still rounds A's product before B scaling and the base add.
    let mut scratch = Scratch::new(MatvecBackend::Scalar, true);
    let mut linear = [-16_785_408.0, -0.0];
    adapter.linear(LINEAR[0], &[4097.0, 1.0], 1.0, &mut linear, &mut scratch)?;
    assert_eq!(
        linear.map(f32::to_bits),
        [0.0_f32.to_bits(), (-0.0_f32).to_bits()]
    );
    let mut embedding = [0.0, -0.0];
    adapter.embedding(0, 1.0, &mut embedding, &mut scratch)?;
    assert_eq!(
        embedding.map(f32::to_bits),
        [4097.0_f32.to_bits(), (-0.0_f32).to_bits()]
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
