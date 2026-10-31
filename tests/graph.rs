#![forbid(unsafe_code)]
//! Focused scalar graph, state ownership, and model contract regressions.

use std::collections::BTreeMap;
use std::sync::Arc;

#[cfg(feature = "cli")]
use bonsai_gguf::MetadataValueRef;
use bonsai_gguf::{MetadataArray, MetadataValue};
use bonsai_prism_runtime::graph::{Qwen35, StateComponent};
use bonsai_prism_runtime::kernels::{MatvecBackend, MatvecFallbackReason, round_f16};
use bonsai_prism_runtime::model::lora::RankOneAdapter;
use bonsai_prism_runtime::model::{Model, Tensor};
use bonsai_prism_runtime::runtime::{
    Cancellation, Generation, Limits, Prefill, PromptReuse, Runtime,
};
use bonsai_prism_runtime::{Decoder, MAX_PREFILL_BLOCK_TOKENS, Result};
#[cfg(feature = "cli")]
use serde_json::Value;

macro_rules! metadata {
    ({ $($key:literal : $value:tt),* $(,)? }) => {
        BTreeMap::from([$(($key.to_owned(), metadata!($value))),*])
    };
    ([$($value:expr),* $(,)?]) => { MetadataValue::from(vec![$($value),*]) };
    ($value:expr) => { MetadataValue::from($value) };
}

fn weight(name: &str, dimensions: &[usize], values: &[f32]) -> Tensor {
    Tensor {
        name: name.to_owned(),
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

fn fixture(recurrent: bool) -> Model {
    let metadata = metadata!({
        "general.architecture": "qwen35",
        "qwen35.block_count": 1,
        "qwen35.context_length": 16,
        "qwen35.embedding_length": 2,
        "qwen35.feed_forward_length": 2,
        "qwen35.attention.head_count": 1,
        "qwen35.attention.head_count_kv": 1,
        "qwen35.attention.key_length": 2,
        "qwen35.attention.value_length": 2,
        "qwen35.attention.layer_norm_rms_epsilon": 0.000_001,
        "qwen35.rope.dimension_count": 2,
        "qwen35.rope.dimension_sections": [1,0,0,0],
        "qwen35.rope.freq_base": 10000.0,
        "qwen35.ssm.state_size": 2,
        "qwen35.ssm.group_count": 1,
        "qwen35.ssm.time_step_rank": 1,
        "qwen35.ssm.inner_size": 2,
        "qwen35.ssm.conv_kernel": 2,
        "qwen35.full_attention_interval": 1,
        "qwen35.attention.recurrent_layers": [recurrent]
    });
    let mut tensors = BTreeMap::new();
    let mut insert = |name: &str, dimensions: &[usize], values: &[f32]| {
        tensors.insert(name.to_owned(), weight(name, dimensions, values));
    };
    insert("token_embd.weight", &[2, 2], &[1.0, 0.0, 0.0, 1.0]);
    insert("output.weight", &[2, 2], &[1.0, 0.0, 0.0, 1.0]);
    insert("output_norm.weight", &[2], &[1.0, 1.0]);
    insert("blk.0.attn_norm.weight", &[2], &[1.0, 1.0]);
    insert("blk.0.post_attention_norm.weight", &[2], &[1.0, 1.0]);
    insert("blk.0.ffn_up.weight", &[2, 2], &[0.0; 4]);
    insert("blk.0.ffn_gate.weight", &[2, 2], &[0.0; 4]);
    insert("blk.0.ffn_down.weight", &[2, 2], &[0.0; 4]);
    if recurrent {
        insert(
            "blk.0.attn_qkv.weight",
            &[2, 6],
            &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        );
        insert("blk.0.attn_gate.weight", &[2, 2], &[1.0, 0.0, 0.0, 1.0]);
        insert(
            "blk.0.ssm_conv1d.weight",
            &[2, 6],
            &[
                0.25, 1.0, 0.25, 1.0, 0.25, 1.0, 0.25, 1.0, 0.25, 1.0, 0.25, 1.0,
            ],
        );
        insert("blk.0.ssm_dt.bias", &[1], &[0.0]);
        insert("blk.0.ssm_a", &[1], &[-1.0]);
        insert("blk.0.ssm_beta.weight", &[2, 1], &[0.0, 0.0]);
        insert("blk.0.ssm_alpha.weight", &[2, 1], &[0.0, 0.0]);
        insert("blk.0.ssm_norm.weight", &[2], &[1.0, 1.0]);
        insert("blk.0.ssm_out.weight", &[2, 2], &[1.0, 0.0, 0.0, 1.0]);
    } else {
        insert("blk.0.attn_q.weight", &[2, 4], &[0.0; 8]);
        insert("blk.0.attn_k.weight", &[2, 2], &[0.0; 4]);
        insert("blk.0.attn_v.weight", &[2, 2], &[1.0, 0.0, 0.0, 1.0]);
        insert("blk.0.attn_output.weight", &[2, 2], &[1.0, 0.0, 0.0, 1.0]);
        insert("blk.0.attn_q_norm.weight", &[2], &[1.0, 1.0]);
        insert("blk.0.attn_k_norm.weight", &[2], &[1.0, 1.0]);
    }
    Model {
        metadata,
        tensors,
        identity: "graph-fixture".to_owned(),
    }
}

fn rank_one_factors(base: &Model, targets: &[&str], value: f32) -> Model {
    let mut tensors = BTreeMap::new();
    for &target in targets {
        let dimensions = &base.tensors[target].dimensions;
        let input = dimensions[0];
        let output = dimensions[1];
        let a_shape = if target == "token_embd.weight" {
            [1, output]
        } else {
            [input, 1]
        };
        let b_shape = [
            1,
            if target == "token_embd.weight" {
                input
            } else {
                output
            },
        ];
        for (suffix, shape) in [("lora_a", a_shape), ("lora_b", b_shape)] {
            let name = format!("{target}.{suffix}");
            tensors.insert(
                name.clone(),
                weight(&name, &shape, &vec![value; shape.iter().product()]),
            );
        }
    }
    Model {
        metadata: metadata!({"general.architecture":"qwen35", "general.type":"adapter", "adapter.type":"lora", "adapter.lora_alpha":1.0_f32}),
        tensors,
        identity: "rank-one-fixture".to_owned(),
    }
}

fn rank_one_graph(base: Model, factors: Model, scale: f32) -> Result<Qwen35> {
    let adapter = RankOneAdapter::from_model(factors, &base)?;
    Qwen35::with_rank_one_adapter(base, 16, 1024 * 1024, MatvecBackend::Scalar, adapter, scale)
}

fn graph_bits(graph: &Qwen35) -> Result<Vec<u32>> {
    let mut bits = Vec::new();
    graph.visit_state(&mut |_, _, values| {
        bits.extend(values.iter().map(|value| value.to_bits()));
        Ok(())
    })?;
    Ok(bits)
}

#[test]
fn rank_one_zero_scale_preserves_full_and_recurrent_logits_and_state_bits() -> Result<()> {
    for recurrent in [false, true] {
        for scale in [0.0, -0.0] {
            let target = if recurrent {
                "blk.0.ssm_out.weight"
            } else {
                "blk.0.attn_output.weight"
            };
            let base = fixture(recurrent);
            // These finite factors overflow if either correction product runs.
            let factors = rank_one_factors(
                &base,
                &["token_embd.weight", "blk.0.ffn_down.weight", target],
                f32::MAX,
            );
            let mut actual = rank_one_graph(base, factors, scale)?;
            let mut expected = Qwen35::new(fixture(recurrent), 16)?;
            for token in [0, 1, 1, 0] {
                assert_eq!(
                    actual
                        .evaluate(token)?
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    expected
                        .evaluate(token)?
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                );
                assert_eq!(graph_bits(&actual)?, graph_bits(&expected)?);
            }
        }
    }
    Ok(())
}

fn writer_transform(base: &mut Model, target: &str, inverse_embedding: bool) {
    for (key, value) in [
        ("prism.hadamard.version", metadata!(1)),
        ("prism.hadamard.block_size", metadata!(2)),
        (
            "prism.hadamard.transform",
            metadata!("normalized-sylvester-walsh-hadamard"),
        ),
        ("prism.hadamard.axis", metadata!("input-last-dimension")),
        ("prism.hadamard.sign_mode", metadata!("explicit")),
        ("prism.hadamard.sign_widths", metadata!([2])),
        ("prism.hadamard.sign_values", metadata!([-1, 1])),
        ("prism.hadamard.weight_names", metadata!([target])),
    ] {
        base.metadata.insert(key.to_owned(), value);
    }
    if inverse_embedding {
        base.metadata.insert(
            "prism.hadamard.inverse_weight_names".into(),
            metadata!(["token_embd.weight"]),
        );
    }
}

#[test]
fn rank_one_residual_writer_uses_unrotated_activations() -> Result<()> {
    for recurrent in [false, true] {
        let target = if recurrent {
            "blk.0.ssm_out.weight"
        } else {
            "blk.0.attn_output.weight"
        };
        let mut base = fixture(recurrent);
        base.tensors
            .insert(target.to_owned(), weight(target, &[2, 2], &[0.0; 4]));
        writer_transform(&mut base, target, false);
        let mut factors = rank_one_factors(&base, &[target], 0.0);
        let a_name = format!("{target}.lora_a");
        let b_name = format!("{target}.lora_b");
        factors
            .tensors
            .insert(a_name.clone(), weight(&a_name, &[2, 1], &[1.0, 0.0]));
        factors
            .tensors
            .insert(b_name.clone(), weight(&b_name, &[1, 2], &[0.0, 1.0]));
        let mut actual = rank_one_graph(base, factors, 1.0)?;
        let mut control = fixture(recurrent);
        control.tensors.insert(
            target.to_owned(),
            weight(target, &[2, 2], &[0.0, 0.0, 1.0, 0.0]),
        );
        let mut expected = Qwen35::new(control, 16)?;
        for token in [0, 1, 0] {
            assert_eq!(actual.evaluate(token)?, expected.evaluate(token)?);
            assert_eq!(graph_bits(&actual)?, graph_bits(&expected)?);
        }
    }
    Ok(())
}

#[test]
fn rank_one_embedding_correction_follows_inverse_transform() -> Result<()> {
    let mut base = fixture(false);
    writer_transform(&mut base, "output.weight", true);
    let mut factors = rank_one_factors(&base, &["token_embd.weight"], 0.0);
    factors.tensors.insert(
        "token_embd.weight.lora_a".into(),
        weight("token_embd.weight.lora_a", &[1, 2], &[0.5, -0.25]),
    );
    factors.tensors.insert(
        "token_embd.weight.lora_b".into(),
        weight("token_embd.weight.lora_b", &[1, 2], &[0.25, 1.0]),
    );
    // Put the independently corrected primal rows into a plain embedding;
    // output.weight's transform remains identical in both graphs.
    let mut control = fixture(false);
    writer_transform(&mut control, "output.weight", false);
    let half = 1.0_f32 / 2.0_f32.sqrt();
    control.tensors.insert(
        "token_embd.weight".into(),
        weight(
            "token_embd.weight",
            &[2, 2],
            &[-half + 0.125, half + 0.5, -half - 0.0625, -half - 0.25],
        ),
    );
    let mut actual = rank_one_graph(base, factors, 1.0)?;
    let mut expected = Qwen35::new(control, 16)?;
    for token in [0, 1, 1, 0] {
        assert_eq!(actual.evaluate(token)?, expected.evaluate(token)?);
    }
    Ok(())
}

#[test]
fn rank_one_embedding_admission_rejects_tied_output_head_at_every_scale() -> Result<()> {
    for scale in [0.0, 1.3] {
        let mut tied = fixture(false);
        tied.tensors.remove("output.weight");
        let factors = rank_one_factors(&tied, &["token_embd.weight"], 0.125);
        let Err(error) = rank_one_graph(tied, factors, scale) else {
            return Err("tied-output embedding adapter was admitted".into());
        };
        assert_eq!(
            error.to_string(),
            "rank-one embedding adapter requires a separate output head"
        );

        let separate = fixture(false);
        let factors = rank_one_factors(&separate, &["token_embd.weight"], 0.125);
        rank_one_graph(separate, factors, scale)?.evaluate(0)?;

        let mut tied = fixture(false);
        tied.tensors.remove("output.weight");
        let factors = rank_one_factors(&tied, &["blk.0.ffn_down.weight"], 0.125);
        rank_one_graph(tied, factors, scale)?.evaluate(0)?;
    }
    Ok(())
}

#[test]
fn rank_one_admission_rejects_factor_schema_and_base_drift() -> Result<()> {
    for invalid in 0..9 {
        let base = fixture(false);
        let mut factors = rank_one_factors(&base, &["blk.0.ffn_down.weight"], 0.25);
        let name = "blk.0.ffn_down.weight.lora_a";
        match invalid {
            0 => {
                factors
                    .metadata
                    .insert("adapter.type".into(), metadata!("other"));
            }
            1 => {
                factors
                    .metadata
                    .insert("adapter.lora_alpha".into(), metadata!(f32::NAN));
            }
            2 => {
                factors
                    .metadata
                    .insert("adapter.lora_alpha".into(), metadata!(0.0_f32));
            }
            3 => {
                factors.tensors.remove("blk.0.ffn_down.weight.lora_b");
            }
            4 => {
                factors.tensors.get_mut(name).ok_or("missing factor")?.kind = 1;
            }
            5 => {
                factors
                    .tensors
                    .get_mut(name)
                    .ok_or("missing factor")?
                    .dimensions = vec![2, 2];
            }
            6 => {
                factors.tensors.get_mut(name).ok_or("missing factor")?.data =
                    Arc::from(f32::INFINITY.to_le_bytes());
            }
            7 => {
                factors.tensors.get_mut(name).ok_or("missing factor")?.name = "alias".into();
            }
            _ => {
                factors.identity.clear();
            }
        }
        assert!(RankOneAdapter::from_model(factors, &base).is_err());
    }
    for scale in [f32::NAN, f32::INFINITY, -0.1, 2.1] {
        let base = fixture(false);
        let factors = rank_one_factors(&base, &["blk.0.ffn_down.weight"], 0.25);
        assert!(rank_one_graph(base, factors, scale).is_err());
    }
    let base = fixture(false);
    let adapter = RankOneAdapter::from_model(
        rank_one_factors(&base, &["blk.0.ffn_down.weight"], 0.25),
        &base,
    )?;
    let mut other = fixture(false);
    other.identity = "other".into();
    assert!(
        Qwen35::with_rank_one_adapter(other, 16, 1024 * 1024, MatvecBackend::Scalar, adapter, 1.0)
            .is_err()
    );
    Ok(())
}

#[test]
fn rank_one_checkpoint_and_failed_generation_preserve_complete_state() -> Result<()> {
    let base = fixture(true);
    let factors = rank_one_factors(
        &base,
        &[
            "token_embd.weight",
            "blk.0.ssm_out.weight",
            "blk.0.ffn_down.weight",
        ],
        0.125,
    );
    let mut graph = rank_one_graph(base, factors, 1.3)?;
    assert!(
        graph
            .identity()
            .contains(":rank-one-lora-v1:rank-one-fixture:3fa66666")
    );
    graph.evaluate(0)?;
    let checkpoint = graph.checkpoint()?;
    let expected = graph.evaluate(1)?;
    let state = graph_bits(&graph)?;
    graph.restore(&checkpoint)?;
    assert_eq!(graph.evaluate(1)?, expected);
    assert_eq!(graph_bits(&graph)?, state);
    graph.reset()?;
    let mut runtime = Runtime::new(
        graph,
        Limits {
            context_tokens: 16,
            checkpoints: 0,
            checkpoint_bytes: 1024 * 1024,
        },
    )?;
    let before = graph_bits(runtime.decoder())?;
    let result = runtime.generate_until(
        &[0],
        &Generation {
            max_tokens: 1,
            stop_tokens: vec![],
        },
        &Cancellation::default(),
        std::time::Instant::now() + std::time::Duration::from_secs(10),
        |_| Err("delivery failed".into()),
    );
    assert!(result.is_err());
    assert_eq!(graph_bits(runtime.decoder())?, before);
    Ok(())
}

#[test]
fn rank_one_writer_overflow_restores_previous_committed_request() -> Result<()> {
    let base = fixture(false);
    let target = "blk.0.attn_output.weight";
    let mut factors = rank_one_factors(&base, &[target], f32::MAX);
    let a_name = format!("{target}.lora_a");
    factors
        .tensors
        .insert(a_name.clone(), weight(&a_name, &[2, 1], &[0.0, f32::MAX]));
    let graph = rank_one_graph(base, factors, 1.3)?;
    let mut runtime = Runtime::new(
        graph,
        Limits {
            context_tokens: 16,
            checkpoints: 0,
            checkpoint_bytes: 1024 * 1024,
        },
    )?;
    let options = Generation {
        max_tokens: 1,
        stop_tokens: vec![],
    };
    let cancellation = Cancellation::default();
    runtime.generate(&[0], &options, &cancellation, |_| Ok(()))?;
    let state = graph_bits(runtime.decoder())?;
    let tokens = runtime.tokens().to_vec();
    assert!(
        runtime
            .generate(&[0, 1], &options, &cancellation, |_| Ok(()))
            .is_err()
    );
    assert_eq!(runtime.tokens(), tokens);
    assert_eq!(graph_bits(runtime.decoder())?, state);
    assert!(!runtime.is_faulted());
    runtime.generate(&[0], &options, &cancellation, |_| Ok(()))?;
    assert_eq!(graph_bits(runtime.decoder())?, state);
    Ok(())
}

fn hybrid_fixture() -> Model {
    let mut model = fixture(false);
    model
        .metadata
        .insert("qwen35.block_count".to_owned(), metadata!(2));
    model.metadata.insert(
        "qwen35.attention.recurrent_layers".to_owned(),
        metadata!([false, true]),
    );
    for (name, mut tensor) in fixture(true).tensors {
        if let Some(suffix) = name.strip_prefix("blk.0.") {
            let name = format!("blk.1.{suffix}");
            tensor.name.clone_from(&name);
            model.tensors.insert(name, tensor);
        }
    }
    model
}

fn scratch_lifetime_fixture(feed_forward: u8) -> Model {
    let mut model = preparation_fixture(false, false);
    model
        .metadata
        .insert("qwen35.block_count".to_owned(), metadata!(2));
    model.metadata.insert(
        "qwen35.attention.recurrent_layers".to_owned(),
        metadata!([false, true]),
    );
    model.metadata.insert(
        "qwen35.feed_forward_length".to_owned(),
        MetadataValue::from(u64::from(feed_forward)),
    );
    for (name, mut tensor) in preparation_fixture(true, false).tensors {
        if let Some(suffix) = name.strip_prefix("blk.0.") {
            let name = format!("blk.1.{suffix}");
            tensor.name.clone_from(&name);
            model.tensors.insert(name, tensor);
        }
    }
    let feed_forward = usize::from(feed_forward);
    for block in 0..2 {
        for (suffix, values) in [
            (
                "ffn_up.weight",
                [1.5, -0.25, -0.5, 0.75].repeat(feed_forward / 2),
            ),
            (
                "ffn_gate.weight",
                [0.5, 0.25, -0.125, 1.25].repeat(feed_forward / 2),
            ),
            (
                "ffn_down.weight",
                [
                    [0.25, -0.5].repeat(feed_forward / 2),
                    [0.75, 0.125].repeat(feed_forward / 2),
                ]
                .concat(),
            ),
        ] {
            let name = format!("blk.{block}.{suffix}");
            let dimensions = if suffix == "ffn_down.weight" {
                vec![feed_forward, 2]
            } else {
                vec![2, feed_forward]
            };
            model
                .tensors
                .insert(name.clone(), weight(&name, &dimensions, &values));
        }
    }
    model
}

fn scratch_state_bits(decoder: &Qwen35) -> Result<Vec<u32>> {
    let mut state = Vec::new();
    decoder.visit_state(&mut |_, _, values| {
        state.extend(values.iter().map(|value| value.to_bits()));
        Ok(())
    })?;
    Ok(state)
}

fn scratch_step_bits(decoder: &mut Qwen35, token: u32) -> Result<Vec<u32>> {
    let mut bits = decoder
        .evaluate(token)?
        .into_iter()
        .map(f32::to_bits)
        .collect::<Vec<_>>();
    bits.extend(scratch_state_bits(decoder)?);
    Ok(bits)
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "Retain the complete independent pre-sharing logits and state bit fixtures with their recovery assertions."
)]
fn phase_shared_scratch_preserves_frozen_hybrid_state_and_recovery() -> Result<()> {
    // Captured before lifetime-based scratch sharing at
    // 1f22c4dd2236513a239eea4d172864a8b7332419, native Windows ARM64.
    // Each row contains both logits, then every committed KV, convolution and
    // GDN element in visit_state order. Both blocks have nonzero FFN gates.
    let expected: [[&[u32]; 3]; 2] = [
        [
            &[
                0x3fa1_1d29,
                0x3f25_0c96,
                0x3f48_e000,
                0xbf96_a000,
                0x3fb5_0000,
                0x0000_0000,
                0x3fa5_10d0,
                0x3f14_9c66,
                0x3fa5_10d0,
                0x3f14_9c66,
                0x3fa5_10d0,
                0x3f14_9c66,
                0x3ef4_dd0c,
                0x3e34_4743,
                0x3e34_4743,
                0x3d84_ba6c,
            ],
            &[
                0xbef1_12f4,
                0x3faa_b11c,
                0x3f48_e000,
                0xbf96_a000,
                0x3e1a_e000,
                0x3fb4_0000,
                0x3fb5_0000,
                0x0000_0000,
                0x0000_0000,
                0x3fb5_0000,
                0xbe7b_119b,
                0x3fb2_472a,
                0xbe7b_119b,
                0x3fb2_472a,
                0xbe7b_119b,
                0x3fb2_472a,
                0x3e5d_2120,
                0x3d79_405f,
                0x3dc3_c520,
                0x3f07_d81c,
            ],
            &[
                0x3f70_ba49,
                0x3f87_349e,
                0x3f48_e000,
                0xbf96_a000,
                0x3e1a_e000,
                0x3fb4_0000,
                0x3f3e_6000,
                0x3f9a_0000,
                0x3fb5_0000,
                0x0000_0000,
                0x0000_0000,
                0x3fb5_0000,
                0x3fb5_0000,
                0x0000_0000,
                0x3f8b_63da,
                0x3f66_fbe6,
                0x3f8b_63da,
                0x3f66_fbe6,
                0x3f8b_63da,
                0x3f66_fbe6,
                0x3e9a_7cfd,
                0x3e90_3c88,
                0x3e86_e938,
                0x3f09_adbe,
            ],
        ],
        [
            &[
                0x3f0c_aaf9,
                0x3fa6_cbf4,
                0x3f48_e000,
                0xbf96_a000,
                0x3fb5_0000,
                0x0000_0000,
                0x3f5d_c2c2,
                0x3f8f_15f4,
                0x3f5d_c2c2,
                0x3f8f_15f4,
                0x3f5d_c2c2,
                0x3f8f_15f4,
                0x3e27_7313,
                0x3e67_517c,
                0x3e67_517d,
                0x3e9f_c625,
            ],
            &[
                0xbf8d_e975,
                0x3f60_c1f4,
                0x3f48_e000,
                0xbf96_a000,
                0x3e1a_e000,
                0x3fb4_0000,
                0x3fb5_0000,
                0x0000_0000,
                0x0000_0000,
                0x3fb5_0000,
                0xbf7c_9aaa,
                0x3f81_acf9,
                0xbf7c_9aaa,
                0x3f81_acf9,
                0xbf7c_9aaa,
                0x3f81_acf9,
                0x3dd7_334a,
                0xbcda_0164,
                0x3c88_e484,
                0x3efe_d447,
            ],
            &[
                0x3ea8_521b,
                0x3fb0_0f74,
                0x3f48_e000,
                0xbf96_a000,
                0x3e1a_e000,
                0x3fb4_0000,
                0x3f3e_6000,
                0x3f9a_0000,
                0x3fb5_0000,
                0x0000_0000,
                0x0000_0000,
                0x3fb5_0000,
                0x3fb5_0000,
                0x0000_0000,
                0x3f4c_560d,
                0x3f95_6e63,
                0x3f4c_560d,
                0x3f95_6e63,
                0x3f4c_560d,
                0x3f95_6e63,
                0x3dc4_818c,
                0x3e0c_d6f6,
                0x3e04_14b7,
                0x3f21_f92e,
            ],
        ],
    ];
    // Attention QKV is larger than the narrow FFN; the wide FFN is larger than
    // QKV. Exercise both high-water transitions on every hybrid token step.
    for (width, expected) in [2, 8].into_iter().zip(expected) {
        let mut decoder =
            Qwen35::with_backend(scratch_lifetime_fixture(width), 16, MatvecBackend::Scalar)?;
        assert_eq!(scratch_step_bits(&mut decoder, 0)?, expected[0]);
        let checkpoint = decoder.checkpoint()?;
        assert_eq!(scratch_step_bits(&mut decoder, 1)?, expected[1]);
        assert_eq!(scratch_step_bits(&mut decoder, 0)?, expected[2]);
        assert!(decoder.evaluate(2).is_err());
        assert_eq!(scratch_state_bits(&decoder)?, expected[2][2..]);
        decoder.restore(&checkpoint)?;
        assert_eq!(scratch_step_bits(&mut decoder, 1)?, expected[1]);
        decoder.reset()?;
        assert_eq!(scratch_step_bits(&mut decoder, 0)?, expected[0]);
        decoder.restore(&checkpoint)?;
        assert_eq!(scratch_step_bits(&mut decoder, 1)?, expected[1]);
    }

    let mut fault = scratch_lifetime_fixture(2);
    fault.tensors.insert(
        "output.weight".to_owned(),
        weight("output.weight", &[2, 2], &[0.0, f32::MAX, 0.0, 0.0]),
    );
    let mut decoder = Qwen35::with_backend(fault, 16, MatvecBackend::Scalar)?;
    decoder.evaluate(0)?;
    let before_failure = scratch_state_bits(&decoder)?;
    let mut logits = vec![f32::NAN; 2];
    // Token 1 overflows only the final output projection, after both layer
    // deltas and every shared attention/FFN temporary have been populated.
    assert!(decoder.evaluate_into(1, &mut logits).is_err());
    assert_eq!(logits, Vec::<f32>::new());
    assert_eq!(scratch_state_bits(&decoder)?, before_failure);
    // Retrying the same token without output must overwrite all stale scratch
    // and produce exactly the ordinary trajectory's committed state.
    decoder.ingest(1)?;
    assert_eq!(scratch_state_bits(&decoder)?, expected[0][1][2..]);
    Ok(())
}

fn paged_hybrid_fixture() -> Model {
    let mut model = hybrid_fixture();
    model.metadata.insert(
        "qwen35.context_length".to_owned(),
        MetadataValue::from(192_u64),
    );
    model
}

fn backend_fixture(recurrent: bool, kind: u32, width: usize) -> Result<Model> {
    let mut model = fixture(recurrent);
    model.metadata.insert(
        "qwen35.feed_forward_length".to_owned(),
        MetadataValue::from(u64::try_from(width)?),
    );
    for name in ["blk.0.ffn_up.weight", "blk.0.ffn_gate.weight"] {
        let values = [0.25_f32, -0.125, -0.5, 0.75, 0.125, 0.25, -0.25, -0.25];
        model.tensors.insert(
            name.to_owned(),
            weight(name, &[2, width], &values.repeat(width / 4)),
        );
    }
    let mut data = Vec::new();
    for symbols in [0x66, 0x99] {
        let mut block = [symbols; 34];
        block[..2].copy_from_slice(&0x2800_u16.to_le_bytes());
        let bytes = if kind == 143 {
            bonsai_ternary::pack_ptq(&block)?.to_vec()
        } else {
            block.to_vec()
        };
        data.extend(bytes.repeat(width / 128));
    }
    let name = "blk.0.ffn_down.weight";
    model.tensors.insert(
        name.to_owned(),
        Tensor {
            name: name.to_owned(),
            dimensions: vec![width, 2],
            kind,
            data: Arc::from(data),
        },
    );
    Ok(model)
}

fn assert_backend_state_parity(
    mut scalar: Qwen35,
    mut selected: Qwen35,
    request: MatvecBackend,
) -> Result<()> {
    let bits = |values: Vec<f32>| values.into_iter().map(f32::to_bits).collect::<Vec<_>>();
    let row_workers = selected.row_workers();
    assert_eq!(scalar.requested_backend(), MatvecBackend::Scalar);
    assert_eq!(selected.requested_backend(), request);
    assert_eq!(selected.identity(), scalar.identity());
    for token in [0, 1, 0] {
        assert_eq!(
            bits(selected.evaluate(token)?),
            bits(scalar.evaluate(token)?)
        );
        assert_eq!(graph_bits(&selected)?, graph_bits(&scalar)?);
    }
    let scalar_checkpoint = scalar.checkpoint()?;
    let selected_checkpoint = selected.checkpoint()?;
    assert_eq!(selected.checkpoint_bytes(), scalar.checkpoint_bytes());
    let expected = bits(scalar.evaluate(1)?);
    let expected_state = graph_bits(&scalar)?;
    assert_eq!(bits(selected.evaluate(1)?), expected);
    assert_eq!(graph_bits(&selected)?, expected_state);
    selected.ingest(0)?;
    scalar.ingest(0)?;
    assert_eq!(graph_bits(&selected)?, graph_bits(&scalar)?);
    selected.restore(&selected_checkpoint)?;
    scalar.restore(&scalar_checkpoint)?;
    assert_eq!(graph_bits(&selected)?, graph_bits(&scalar)?);
    assert_eq!(bits(selected.evaluate(1)?), expected);
    assert_eq!(bits(scalar.evaluate(1)?), expected);
    assert_eq!(graph_bits(&selected)?, expected_state);
    assert_eq!(graph_bits(&scalar)?, expected_state);
    let mut rejected = vec![f32::NAN];
    assert!(selected.evaluate_into(2, &mut rejected).is_err());
    assert_eq!(rejected, Vec::<f32>::new());
    assert_eq!(graph_bits(&selected)?, expected_state);
    assert_eq!(bits(selected.evaluate(0)?), bits(scalar.evaluate(0)?));
    assert_eq!(graph_bits(&selected)?, graph_bits(&scalar)?);
    selected.reset()?;
    scalar.reset()?;
    assert_eq!(graph_bits(&selected)?, graph_bits(&scalar)?);
    assert_eq!(selected.requested_backend(), request);
    assert_eq!(bits(selected.evaluate(1)?), bits(scalar.evaluate(1)?));
    assert_eq!(graph_bits(&selected)?, graph_bits(&scalar)?);
    // Reset and scratch reuse must not mutate either decoder's owned prefix.
    selected.restore(&selected_checkpoint)?;
    assert_eq!(bits(selected.evaluate(1)?), expected);
    assert_eq!(graph_bits(&selected)?, expected_state);
    assert_eq!(selected.requested_backend(), request);
    assert_eq!(selected.row_workers(), row_workers);
    Ok(())
}

#[test]
fn graph_backend_selection_matches_scalar_logits_checkpoints_and_recovery() -> Result<()> {
    // F32 siblings feed a packed down projection, followed by an F32 output
    // head. Exercise both PQ2 activation contracts and canonical PTQ1 blocks;
    // the final scalar projection must not overwrite the configured request.
    for recurrent in [false, true] {
        for (kind, width) in [(142, 128), (142, 256), (143, 128)] {
            for request in [MatvecBackend::Auto, MatvecBackend::ExperimentalNeon] {
                assert_backend_state_parity(
                    Qwen35::with_backend(
                        backend_fixture(recurrent, kind, width)?,
                        16,
                        MatvecBackend::Scalar,
                    )?,
                    Qwen35::with_backend(backend_fixture(recurrent, kind, width)?, 16, request)?,
                    request,
                )?;
            }
            assert_backend_state_parity(
                Qwen35::with_backend(
                    backend_fixture(recurrent, kind, width)?,
                    16,
                    MatvecBackend::Scalar,
                )?,
                Qwen35::new(backend_fixture(recurrent, kind, width)?, 16)?,
                MatvecBackend::Auto,
            )?;
        }
    }
    Ok(())
}

#[cfg(feature = "parallel-rows")]
#[test]
fn graph_parallel_rows_preserve_logits_checkpoints_and_recovery() -> Result<()> {
    use bonsai_prism_runtime::kernels::RowParallelism;

    for workers in [2, 4] {
        let policy = RowParallelism::new(workers)?;
        for recurrent in [false, true] {
            for (kind, width) in [(142, 128), (142, 256), (143, 128)] {
                let serial = Qwen35::with_backend(
                    backend_fixture(recurrent, kind, width)?,
                    16,
                    MatvecBackend::Scalar,
                )?;
                let mut parallel = Qwen35::with_backend(
                    backend_fixture(recurrent, kind, width)?,
                    16,
                    MatvecBackend::Auto,
                )?;
                parallel.set_row_parallelism(policy.clone());
                assert_eq!(parallel.row_workers(), workers);
                parallel.ingest(0)?;
                assert!(parallel.backend_statistics().parallel_row_operations > 0);
                parallel.reset()?;
                assert_eq!(parallel.row_workers(), workers);
                assert_backend_state_parity(serial, parallel, MatvecBackend::Auto)?;
            }
        }
    }
    Ok(())
}

#[test]
fn graph_backend_diagnostics_distinguish_latest_projection_from_complete_graph() -> Result<()> {
    let mut decoder = Qwen35::new(backend_fixture(false, 142, 128)?, 16)?;
    assert!(!decoder.backend_diagnostics().dispatched);
    decoder.ingest(0)?;
    let packed = decoder.backend_diagnostics();
    assert!(packed.dispatched && packed.succeeded);
    assert_eq!(packed.requested_backend, MatvecBackend::Auto);
    if packed.neon_build_eligible {
        assert_eq!(packed.selected_backend, MatvecBackend::ExperimentalNeon);
        assert_eq!(packed.fallback_reason, None);
    } else {
        assert_eq!(packed.selected_backend, MatvecBackend::Scalar);
        assert_eq!(
            packed.fallback_reason,
            Some(if cfg!(feature = "experimental-neon") {
                MatvecFallbackReason::TargetUnsupported
            } else {
                MatvecFallbackReason::FeatureDisabled
            })
        );
    }
    decoder.evaluate(1)?;
    let floating = decoder.backend_diagnostics();
    let statistics = decoder.backend_statistics();
    let checkpoint = decoder.checkpoint()?;
    assert!(statistics.matrix_attempts > 1);
    assert!(statistics.scalar_operations > 0);
    if packed.selected_backend == MatvecBackend::ExperimentalNeon {
        assert!(statistics.neon_operations > 0);
    }
    assert!(floating.dispatched && floating.succeeded);
    assert_eq!(floating.selected_backend, MatvecBackend::Scalar);
    assert_eq!(
        floating.fallback_reason,
        Some(MatvecFallbackReason::TensorKindUnsupported)
    );
    assert!(decoder.evaluate(2).is_err());
    assert_eq!(decoder.backend_diagnostics(), floating);
    assert_eq!(decoder.backend_statistics(), statistics);
    decoder.reset()?;
    assert_eq!(decoder.backend_diagnostics(), floating);
    assert_eq!(decoder.backend_statistics(), statistics);
    decoder.restore(&checkpoint)?;
    assert_eq!(decoder.backend_statistics(), statistics);
    Ok(())
}

#[cfg(not(feature = "experimental-neon"))]
#[test]
fn graph_backend_neon_request_falls_back_without_feature() -> Result<()> {
    for (kind, width) in [(142, 128), (142, 256), (143, 128)] {
        assert_backend_state_parity(
            Qwen35::with_backend(
                backend_fixture(true, kind, width)?,
                16,
                MatvecBackend::Scalar,
            )?,
            Qwen35::with_backend(
                backend_fixture(true, kind, width)?,
                16,
                MatvecBackend::ExperimentalNeon,
            )?,
            MatvecBackend::ExperimentalNeon,
        )?;
    }
    Ok(())
}

fn preparation_fixture(recurrent: bool, transformed: bool) -> Model {
    let mut model = fixture(recurrent);
    for (name, dimensions, values) in [
        (
            "blk.0.ffn_up.weight",
            vec![2, 2],
            vec![1.5, -0.25, -0.5, 0.75],
        ),
        (
            "blk.0.ffn_gate.weight",
            vec![2, 2],
            vec![0.5, 0.25, -0.125, 1.25],
        ),
        (
            "blk.0.ffn_down.weight",
            vec![2, 2],
            vec![0.25, -0.5, 0.75, 0.125],
        ),
    ] {
        model
            .tensors
            .insert(name.to_owned(), weight(name, &dimensions, &values));
    }
    let siblings = if recurrent {
        for (name, values) in [
            ("blk.0.ssm_beta.weight", [0.125, -0.25]),
            ("blk.0.ssm_alpha.weight", [-0.0625, 0.125]),
        ] {
            model
                .tensors
                .insert(name.to_owned(), weight(name, &[2, 1], &values));
        }
        ["blk.0.attn_qkv.weight", "blk.0.attn_gate.weight"]
    } else {
        model.tensors.insert(
            "blk.0.attn_q.weight".to_owned(),
            weight(
                "blk.0.attn_q.weight",
                &[2, 4],
                &[0.75, -0.25, 0.125, 0.5, -0.5, 1.0, 0.25, -0.125],
            ),
        );
        model.tensors.insert(
            "blk.0.attn_k.weight".to_owned(),
            weight("blk.0.attn_k.weight", &[2, 2], &[0.5, 0.25, -0.75, 0.125]),
        );
        ["blk.0.attn_q.weight", "blk.0.attn_k.weight"]
    };
    if transformed {
        for (key, value) in [
            ("prism.hadamard.version", metadata!(1)),
            ("prism.hadamard.block_size", metadata!(2)),
            (
                "prism.hadamard.transform",
                metadata!("normalized-sylvester-walsh-hadamard"),
            ),
            ("prism.hadamard.axis", metadata!("input-last-dimension")),
            ("prism.hadamard.sign_mode", metadata!("explicit")),
            ("prism.hadamard.sign_widths", metadata!([2])),
            ("prism.hadamard.sign_values", metadata!([-1, 1])),
            (
                "prism.hadamard.weight_names",
                metadata!([
                    siblings[0],
                    siblings[1],
                    "blk.0.ffn_up.weight",
                    "blk.0.ffn_gate.weight"
                ]),
            ),
        ] {
            model.metadata.insert(key.to_owned(), value);
        }
    }
    model
}

#[test]
fn shared_preparation_matches_independent_baseline_logits_and_recovery() -> Result<()> {
    // Captured with the independent, ungrouped projection implementation at
    // 18e6d6b5cadb985d25a3af9460c620b7514facaa, native Windows ARM64.
    // Both attention routes have nonzero FFN siblings; transformed lanes mix
    // signed Hadamard and plain siblings within the same attention group.
    let expected = [
        [
            0x3fa5_10d0,
            0x3f14_9c66,
            0xbe7b_119b,
            0x3fb2_472a,
            0x3e6e_52e4,
            0x3fb2_8d17,
            0x3f89_bb5f,
            0x3f6a_edcd,
        ],
        [
            0x3fb3_0cb9,
            0x3e55_0b5c,
            0xbd3e_3ea0,
            0x3fb4_ebef,
            0xbc0c_b00d,
            0x3fb5_0414,
            0x3f9c_d9c5,
            0x3f34_badd,
        ],
        [
            0x3fae_d262,
            0x3ebb_d7f0,
            0xbed1_b0de,
            0x3fad_432b,
            0xbece_7676,
            0x3fad_8128,
            0x3fae_a37a,
            0x3ebe_8d18,
        ],
        [
            0x3faa_f9ff,
            0x3eed_d29b,
            0x3f4a_7b2d,
            0x3f96_0fa0,
            0x3f5a_ff0f,
            0x3f90_256b,
            0x3fac_395c,
            0x3ede_f4bd,
        ],
    ];
    for (lane, (recurrent, transformed)) in
        [(false, false), (false, true), (true, false), (true, true)]
            .into_iter()
            .enumerate()
    {
        let mut decoder = Qwen35::new(preparation_fixture(recurrent, transformed), 16)?;
        let first = decoder.evaluate(0)?;
        assert_eq!(
            first
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected[lane][..2]
        );
        let checkpoint = decoder.checkpoint()?;
        for (index, token) in [1, 1, 0].into_iter().enumerate() {
            let mut output = vec![f32::NAN];
            assert!(decoder.evaluate_into(2, &mut output).is_err());
            assert_eq!(output, Vec::<f32>::new());
            decoder.evaluate_into(token, &mut output)?;
            assert_eq!(
                output
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected[lane][2 + index * 2..4 + index * 2]
            );
        }
        decoder.restore(&checkpoint)?;
        assert_eq!(
            decoder
                .evaluate(1)?
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected[lane][2..4]
        );
        decoder.reset()?;
        assert_eq!(decoder.evaluate(0)?, first);
    }
    Ok(())
}

#[test]
fn pinned_arithmetic_requires_exact_swift_identity_and_survives_recovery() -> Result<()> {
    // Synthetic labels exercise selection and recovery, not actual-model
    // qualification. The separate verified-loader tests bind production
    // identities to complete file hashes, and native captures own arithmetic.
    for identity in [
        bonsai_prism_runtime::model::swift::MODEL_SHA256,
        bonsai_prism_runtime::model::swift::REPACKED_PTQ1_SHA256,
    ] {
        let mut model = preparation_fixture(true, true);
        model.identity = identity.to_owned();
        let mut pinned = Qwen35::new(model, 16)?;
        let mut near = preparation_fixture(true, true);
        near.identity = format!("{identity}0");
        let mut near = Qwen35::new(near, 16)?;
        let mut source = Qwen35::new(preparation_fixture(true, true), 16)?;
        assert!(pinned.identity().ends_with(":prism-arm64-fused-v1"));
        assert!(!near.identity().ends_with(":prism-arm64-fused-v1"));
        let mut pinned_bits = Vec::new();
        let mut source_bits = Vec::new();
        for token in [0, 1, 1, 0] {
            let checkpoint = pinned.checkpoint()?;
            let expected = pinned.evaluate(token)?;
            pinned.restore(&checkpoint)?;
            assert!(pinned.evaluate(2).is_err());
            assert_eq!(pinned.evaluate(token)?, expected);
            pinned_bits.push(expected.into_iter().map(f32::to_bits).collect::<Vec<_>>());
            let expected = source.evaluate(token)?;
            assert_eq!(near.evaluate(token)?, expected);
            source_bits.push(expected.into_iter().map(f32::to_bits).collect::<Vec<_>>());
        }
        assert_ne!(pinned_bits, source_bits);
        pinned.reset()?;
        assert_eq!(
            pinned
                .evaluate(0)?
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>(),
            pinned_bits[0]
        );
    }
    Ok(())
}

fn two_full_layer_fixture() -> Model {
    let mut model = fixture(false);
    model
        .metadata
        .insert("qwen35.block_count".to_owned(), metadata!(2));
    model.metadata.insert(
        "qwen35.attention.recurrent_layers".to_owned(),
        metadata!([false, false]),
    );
    for (suffix, dimensions, values) in [
        (
            "attn_q.weight",
            &[2, 4][..],
            &[1.0, 0.25, -0.5, 0.75, 0.2, -0.1, 0.4, 0.3][..],
        ),
        ("attn_k.weight", &[2, 2][..], &[0.5, -0.25, 0.75, 1.0][..]),
    ] {
        let name = format!("blk.0.{suffix}");
        model
            .tensors
            .insert(name.clone(), weight(&name, dimensions, values));
    }
    for (_, mut tensor) in model.tensors.clone() {
        if let Some(suffix) = tensor.name.strip_prefix("blk.0.") {
            let name = format!("blk.1.{suffix}");
            tensor.name.clone_from(&name);
            model.tensors.insert(name, tensor);
        }
    }
    model
}

#[test]
fn shared_rope_frequencies_preserve_multi_layer_reference_bits_and_restoration() -> Result<()> {
    // Captured from the scalar runtime at 3e403cb, before frequency sharing
    // and geometric KV growth, using this nonzero Q/K two-layer fixture.
    let expected = [
        [1_068_827_889, 0],
        [1_051_613_334, 1_068_478_857],
        [1_032_297_567, 1_068_814_879],
        [1_065_414_184, 1_065_230_382],
        [1_049_817_547, 1_068_581_028],
        [1_067_000_921, 1_061_226_664],
        [1_067_657_464, 1_058_852_234],
        [1_054_981_332, 1_068_236_638],
    ];
    let tokens = [0, 1, 1, 0, 1, 0, 0, 1];
    let mut decoder = Qwen35::new(two_full_layer_fixture(), 16)?;
    let mut logits = Vec::new();
    for (&token, expected) in tokens[..3].iter().zip(&expected[..3]) {
        decoder.evaluate_into(token, &mut logits)?;
        assert_eq!(
            logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
        );
    }
    let checkpoint = decoder.checkpoint()?;
    for (&token, expected) in tokens[3..].iter().zip(&expected[3..]) {
        decoder.evaluate_into(token, &mut logits)?;
        assert_eq!(
            logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
        );
    }
    decoder.restore(&checkpoint)?;
    assert!(decoder.evaluate_into(2, &mut logits).is_err());
    assert_eq!(logits, Vec::<f32>::new());
    for (&token, expected) in tokens[3..].iter().zip(&expected[3..]) {
        decoder.evaluate_into(token, &mut logits)?;
        assert_eq!(
            logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
        );
    }
    decoder.reset()?;
    for (&token, expected) in tokens.iter().zip(&expected) {
        decoder.evaluate_into(token, &mut logits)?;
        assert_eq!(
            logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
        );
    }
    Ok(())
}

#[test]
fn staged_multi_layer_rope_failure_does_not_advance_state() -> Result<()> {
    let fault_fixture = || {
        let mut model = two_full_layer_fixture();
        model.tensors.insert(
            "output.weight".to_owned(),
            weight("output.weight", &[2, 2], &[0.0, f32::MAX, 0.0, 0.0]),
        );
        model
    };
    let mut decoder = Qwen35::new(fault_fixture(), 16)?;
    let mut fresh = Qwen35::new(fault_fixture(), 16)?;
    let mut logits = Vec::new();
    decoder.evaluate_into(0, &mut logits)?;
    assert_eq!(logits, fresh.evaluate(0)?);
    // Both full layers and their RoPE tables have run before output overflows.
    assert!(decoder.evaluate_into(1, &mut logits).is_err());
    assert_eq!(logits, Vec::<f32>::new());
    decoder.evaluate_into(0, &mut logits)?;
    assert_eq!(logits, fresh.evaluate(0)?);
    decoder.reset()?;
    fresh.reset()?;
    decoder.evaluate_into(0, &mut logits)?;
    assert_eq!(logits, fresh.evaluate(0)?);
    Ok(())
}

#[test]
fn state_only_ingestion_matches_hybrid_outputs_checkpoints_and_reset() -> Result<()> {
    let bits = |values: Vec<f32>| values.into_iter().map(f32::to_bits).collect::<Vec<_>>();
    let mut reference = Qwen35::new(hybrid_fixture(), 16)?;
    let mut ingested = Qwen35::new(hybrid_fixture(), 16)?;
    for token in [0, 1, 1] {
        reference.evaluate(token)?;
        ingested.ingest(token)?;
    }
    assert_eq!(ingested.checkpoint_bytes(), reference.checkpoint_bytes());
    let reference_checkpoint = reference.checkpoint()?;
    let ingested_checkpoint = ingested.checkpoint()?;
    for token in [0, 1, 0] {
        assert_eq!(
            bits(ingested.evaluate(token)?),
            bits(reference.evaluate(token)?)
        );
    }
    ingested.reset()?;
    reference.reset()?;
    assert_eq!(bits(ingested.evaluate(1)?), bits(reference.evaluate(1)?));
    ingested.restore(&ingested_checkpoint)?;
    reference.restore(&reference_checkpoint)?;
    for token in [1, 0, 0] {
        assert_eq!(
            bits(ingested.evaluate(token)?),
            bits(reference.evaluate(token)?)
        );
    }
    Ok(())
}

#[test]
fn state_only_ingestion_skips_only_output_validation() -> Result<()> {
    let mut output_fault = hybrid_fixture();
    output_fault.tensors.insert(
        "output.weight".to_owned(),
        weight("output.weight", &[2, 2], &[f32::MAX; 4]),
    );
    let mut decoder = Qwen35::new(output_fault, 2)?;
    assert!(decoder.evaluate(0).is_err());
    // Unrequested output norm/projection is deliberately not computed or
    // validated. All hybrid state still commits exactly one token.
    decoder.ingest(0)?;
    decoder.ingest(1)?;
    assert!(decoder.ingest(0).is_err());
    decoder.reset()?;
    decoder.ingest(0)?;

    let mut state_fault = fixture(false);
    state_fault.tensors.insert(
        "blk.0.attn_q.weight".to_owned(),
        weight("blk.0.attn_q.weight", &[2, 4], &[f32::MAX; 8]),
    );
    let mut decoder = Qwen35::new(state_fault, 1)?;
    assert!(decoder.ingest(0).is_err());
    assert!(decoder.ingest(0).is_err());
    Ok(())
}

#[test]
fn state_only_ingestion_rejects_invalid_tokens_without_advancing() -> Result<()> {
    let mut decoder = Qwen35::new(hybrid_fixture(), 2)?;
    let mut reference = Qwen35::new(hybrid_fixture(), 2)?;
    assert!(decoder.ingest(2).is_err());
    decoder.ingest(0)?;
    reference.evaluate(0)?;
    assert!(decoder.ingest(u32::MAX).is_err());
    assert_eq!(decoder.evaluate(1)?, reference.evaluate(1)?);
    assert!(decoder.ingest(0).is_err());
    Ok(())
}

#[test]
fn reusable_logits_match_owned_evaluation_across_hybrid_restore_and_reset() -> Result<()> {
    let mut owned = Qwen35::new(hybrid_fixture(), 16)?;
    let mut reused = Qwen35::new(hybrid_fixture(), 16)?;
    let mut logits = Vec::with_capacity(8);
    let allocation = logits.as_ptr();
    for token in [0, 1, 1] {
        reused.evaluate_into(token, &mut logits)?;
        assert_eq!(logits, owned.evaluate(token)?);
        assert_eq!(logits.as_ptr(), allocation);
    }
    let checkpoint = reused.checkpoint()?;
    let owned_checkpoint = owned.checkpoint()?;
    let expected = owned.evaluate(0)?;
    assert_eq!(reused.evaluate(0)?, expected);
    reused.evaluate_into(1, &mut logits)?;
    logits.fill(f32::NAN);
    reused.restore(&checkpoint)?;
    owned.restore(&owned_checkpoint)?;
    reused.evaluate_into(0, &mut logits)?;
    assert_eq!(logits, owned.evaluate(0)?);
    assert_eq!(logits, expected);
    assert_eq!(logits.as_ptr(), allocation);
    reused.reset()?;
    let mut fresh = Qwen35::new(hybrid_fixture(), 16)?;
    reused.evaluate_into(1, &mut logits)?;
    assert_eq!(logits, fresh.evaluate(1)?);
    // The checkpoint owns its state even after scratch reuse and a full reset.
    reused.restore(&checkpoint)?;
    reused.evaluate_into(0, &mut logits)?;
    assert_eq!(logits, expected);
    Ok(())
}

#[test]
fn reusable_logits_clear_on_rejection_and_preserve_hybrid_state() -> Result<()> {
    let mut model = hybrid_fixture();
    model.tensors.insert(
        "output.weight".to_owned(),
        weight("output.weight", &[2, 2], &[0.0, f32::MAX, 0.0, 0.0]),
    );
    let mut decoder = Qwen35::new(model, 3)?;
    let mut logits = vec![f32::NAN; 8];
    let allocation = logits.as_ptr();
    decoder.evaluate_into(0, &mut logits)?;
    let checkpoint = decoder.checkpoint()?;
    // Token 1 overflows output projection after both layer deltas are staged.
    assert!(decoder.evaluate_into(1, &mut logits).is_err());
    assert_eq!(logits, Vec::<f32>::new());
    assert_eq!(logits.as_ptr(), allocation);
    decoder.evaluate_into(0, &mut logits)?;
    let expected = logits.clone();
    decoder.restore(&checkpoint)?;
    decoder.evaluate_into(0, &mut logits)?;
    assert_eq!(logits, expected);
    assert!(decoder.evaluate_into(2, &mut logits).is_err());
    assert_eq!(logits, Vec::<f32>::new());
    decoder.evaluate_into(0, &mut logits)?;
    assert!(decoder.evaluate_into(0, &mut logits).is_err());
    assert_eq!(logits, Vec::<f32>::new());
    assert_eq!(logits.as_ptr(), allocation);
    Ok(())
}

#[test]
#[allow(
    clippy::suboptimal_flops,
    clippy::manual_midpoint,
    reason = "The hand calculation preserves scalar reference operation ordering on bounded fixture values."
)]
fn full_attention_uses_causal_binary16_values_and_gate() -> Result<()> {
    let mut decoder = Qwen35::new(fixture(false), 16)?;
    decoder.evaluate(0)?;
    let logits = decoder.evaluate(1)?;
    // Zero Q/K gives uniform causal attention. Two orthogonal normalized
    // embeddings, cached in f16, contribute half their average through the gate.
    let cached = round_f16(1.0_f32 / (0.5_f32 + 0.000_001).sqrt());
    let hidden = [cached * 0.25, 1.0 + cached * 0.25];
    let scale = 1.0 / ((hidden[0] * hidden[0] + hidden[1] * hidden[1]) / 2.0 + 0.000_001).sqrt();
    assert!((logits[0] - hidden[0] * scale).abs() < 0.000_001);
    assert!((logits[1] - hidden[1] * scale).abs() < 0.000_001);
    Ok(())
}

#[test]
fn hybrid_state_checkpoint_restores_exact_recurrent_prefix() -> Result<()> {
    let mut decoder = Qwen35::new(fixture(true), 16)?;
    decoder.evaluate(0)?;
    let checkpoint = decoder.checkpoint()?;
    let checkpoint_bytes = decoder.checkpoint_bytes();
    let expected = decoder.evaluate(1)?;
    decoder.evaluate(0)?;
    decoder.restore(&checkpoint)?;
    assert_eq!(decoder.checkpoint_bytes(), checkpoint_bytes);
    assert_eq!(decoder.evaluate(1)?, expected);
    decoder.reset()?;
    let mut fresh = Qwen35::new(fixture(true), 16)?;
    assert_eq!(decoder.evaluate(1)?, fresh.evaluate(1)?);
    Ok(())
}

#[test]
fn qualification_observer_reports_committed_state_without_mutation() -> Result<()> {
    for recurrent in [false, true] {
        let mut graph = Qwen35::new(fixture(recurrent), 16)?;
        graph.evaluate(0)?;
        graph.evaluate(1)?;
        let mut observations = Vec::new();
        graph.visit_state(&mut |block, component, values| {
            assert_eq!(block, 0);
            observations.push((component, values.to_vec()));
            Ok(())
        })?;
        if recurrent {
            assert_eq!(observations[0].0, StateComponent::Convolution);
            assert_eq!(observations[0].1.len(), 6);
            assert_eq!(observations[1].0, StateComponent::Recurrent);
            assert_eq!(observations[1].1.len(), 4);
        } else {
            assert_eq!(observations.len(), 4);
            assert_eq!(observations[0].0, StateComponent::Key(0));
            assert_eq!(observations[1].0, StateComponent::Key(1));
            assert_eq!(observations[2].0, StateComponent::Value(0));
            assert_eq!(observations[3].0, StateComponent::Value(1));
            assert!(observations.iter().all(|(_, values)| values.len() == 2));
            assert!(
                observations
                    .iter()
                    .flat_map(|(_, values)| values)
                    .all(|&value| value.to_bits() == round_f16(value).to_bits())
            );
        }
        let checkpoint = graph.checkpoint()?;
        let error = graph
            .visit_state(&mut |_, _, _| Err(std::io::Error::other("observer rejected").into()));
        assert!(error.is_err());
        let after_observation = graph.evaluate(0)?;
        graph.restore(&checkpoint)?;
        assert_eq!(graph.evaluate(0)?, after_observation);
    }
    Ok(())
}

#[test]
fn full_attention_checkpoint_bytes_charge_binary16_capacity_and_restore_prefix() -> Result<()> {
    let mut decoder = Qwen35::new(fixture(false), 16)?;
    let empty_bytes = decoder.checkpoint_bytes();
    decoder.evaluate(0)?;
    let checkpoint = decoder.checkpoint()?;
    // One key and one value, each containing two physical binary16 elements.
    assert_eq!(
        decoder.checkpoint_bytes() - empty_bytes,
        4 * size_of::<u16>()
    );
    let expected = decoder.evaluate(1)?;
    decoder.evaluate(0)?;
    // Geometric growth retains room for four tokens after the third append.
    assert_eq!(
        decoder.checkpoint_bytes() - empty_bytes,
        16 * size_of::<u16>()
    );
    decoder.restore(&checkpoint)?;
    assert_eq!(
        decoder.checkpoint_bytes() - empty_bytes,
        4 * size_of::<u16>()
    );
    assert_eq!(decoder.evaluate(1)?, expected);
    decoder.reset()?;
    decoder.restore(&checkpoint)?;
    assert_eq!(decoder.evaluate(1)?, expected);
    Ok(())
}

#[test]
fn paged_hybrid_checkpoints_restore_exact_logits_on_both_sides_of_seals() -> Result<()> {
    let bits = |logits: Vec<f32>| logits.into_iter().map(f32::to_bits).collect::<Vec<_>>();
    let mut decoder = Qwen35::new(paged_hybrid_fixture(), 192)?;
    let mut checkpoints = Vec::new();
    for position in 0..130 {
        if [63, 64, 65, 127, 128, 129].contains(&position) {
            checkpoints.push((position, decoder.checkpoint()?));
        }
        decoder.ingest(u32::try_from(position % 2)?)?;
    }
    // Later appends, a rejected token and reset cannot mutate saved pages,
    // copied tails, convolution histories or recurrent matrices.
    assert!(decoder.evaluate(2).is_err());
    decoder.reset()?;
    for (position, checkpoint) in checkpoints {
        let mut fresh = Qwen35::new(paged_hybrid_fixture(), 192)?;
        for index in 0..position {
            fresh.ingest(u32::try_from(index % 2)?)?;
        }
        decoder.restore(&checkpoint)?;
        for token in [1, 1, 0, 1] {
            assert_eq!(bits(decoder.evaluate(token)?), bits(fresh.evaluate(token)?));
        }
        // A second restoration still yields the exact original prefix after
        // the first continuation crossed a different seal boundary.
        decoder.restore(&checkpoint)?;
        fresh.reset()?;
        for index in 0..position {
            fresh.ingest(u32::try_from(index % 2)?)?;
        }
        assert_eq!(bits(decoder.evaluate(0)?), bits(fresh.evaluate(0)?));
    }
    Ok(())
}

#[test]
fn bounded_prefill_matches_exact_hybrid_logits_across_page_boundaries() -> Result<()> {
    for length in [1, 63, 64, 65, 127, 128, 129] {
        let prompt: Vec<u32> = [0, 1].into_iter().cycle().take(length).collect();
        let mut reference = Qwen35::new(paged_hybrid_fixture(), 192)?;
        for &token in &prompt[..length - 1] {
            reference.ingest(token)?;
        }
        let expected: Vec<u32> = reference
            .evaluate(prompt[length - 1])?
            .into_iter()
            .map(f32::to_bits)
            .collect();
        for block_tokens in [1, 2, 4, MAX_PREFILL_BLOCK_TOKENS] {
            let mut decoder = Qwen35::new(paged_hybrid_fixture(), 192)?;
            for block in prompt[..length - 1].chunks(block_tokens) {
                decoder.ingest_block(block, &|| Ok(()))?;
            }
            let actual: Vec<u32> = decoder
                .evaluate(prompt[length - 1])?
                .into_iter()
                .map(f32::to_bits)
                .collect();
            assert_eq!(actual, expected, "prompt {length}, block {block_tokens}");
            assert_eq!(decoder.evaluate(0)?, reference.evaluate(0)?);
            // Restore the reference so each block size compares the same continuation.
            reference.reset()?;
            for &token in &prompt {
                reference.ingest(token)?;
            }
        }
    }
    Ok(())
}

#[test]
fn paged_hybrid_runtime_reuses_edited_prefix_and_rolls_back_across_seals() -> Result<()> {
    let limits = Limits {
        context_tokens: 192,
        checkpoints: 4,
        checkpoint_bytes: 1_048_576,
    };
    let mut runtime = Runtime::new(Qwen35::new(paged_hybrid_fixture(), 192)?, limits)?;
    let cancellation = Cancellation::default();
    let prefill = Prefill { block_tokens: 4 };
    let stop = Generation {
        max_tokens: 1,
        stop_tokens: vec![0, 1],
    };
    let prefix: Vec<u32> = [0, 1].into_iter().cycle().take(65).collect();
    runtime.generate_with_prefill(&prefix, &stop, prefill, &cancellation, |_| Ok(()))?;
    let mut longer = prefix.clone();
    longer.extend([1, 0].into_iter().cycle().take(65));
    runtime.generate_with_prefill(&longer, &stop, prefill, &cancellation, |_| Ok(()))?;
    assert_eq!(runtime.tokens(), longer);
    let mut edited = prefix.clone();
    edited.extend([0, 0, 1, 1].into_iter().cycle().take(66));
    // The failure occurs after restoring the saved 65-token prefix and
    // creating another sealed page. Its rollback retains the 130-token state.
    let generate = Generation {
        max_tokens: 1,
        stop_tokens: Vec::new(),
    };
    assert!(
        runtime
            .generate_with_prefill(&edited, &generate, prefill, &cancellation, |_| {
                Err(std::io::Error::other("synthetic consumer failure").into())
            })
            .is_err()
    );
    assert_eq!(runtime.tokens(), longer);
    assert!(!runtime.is_faulted());
    let mut emitted = Vec::new();
    let completion =
        runtime.generate_with_prefill(&edited, &generate, prefill, &cancellation, |token| {
            emitted.push(token);
            Ok(())
        })?;
    assert_eq!(completion.reused_tokens, prefix.len());
    let mut fresh = Runtime::new(Qwen35::new(paged_hybrid_fixture(), 192)?, limits)?;
    let mut expected = Vec::new();
    fresh.generate(&edited, &generate, &cancellation, |token| {
        expected.push(token);
        Ok(())
    })?;
    assert_eq!(emitted, expected);
    assert_eq!(runtime.tokens(), fresh.tokens());
    Ok(())
}

#[test]
fn paged_hybrid_checkpoint_budget_rejection_preserves_state_and_reset_recovers() -> Result<()> {
    let decoder = Qwen35::new(paged_hybrid_fixture(), 192)?;
    let limits = Limits {
        context_tokens: 192,
        checkpoints: 4,
        checkpoint_bytes: decoder.checkpoint_bytes() + 512,
    };
    let mut runtime = Runtime::new(decoder, limits)?;
    let cancellation = Cancellation::default();
    let stop = Generation {
        max_tokens: 1,
        stop_tokens: vec![0, 1],
    };
    let prefix: Vec<u32> = [0, 1].into_iter().cycle().take(65).collect();
    runtime.generate(&prefix, &stop, &cancellation, |_| Ok(()))?;
    // The optional save cannot fit. The next request cannot admit even its
    // mandatory rollback, so it must leave the committed prefix untouched.
    assert_eq!(runtime.checkpoint_count(), 0);
    assert!(
        runtime
            .generate(&[0], &stop, &cancellation, |_| Ok(()))
            .is_err()
    );
    assert_eq!(runtime.tokens(), prefix);
    assert!(!runtime.is_faulted());
    runtime.reset()?;
    runtime.generate(&[1], &stop, &cancellation, |_| Ok(()))?;
    assert_eq!(runtime.tokens(), [1]);
    Ok(())
}

#[test]
fn full_attention_binary16_overflow_does_not_commit_cache() -> Result<()> {
    let model = || {
        let mut model = fixture(false);
        model.tensors.insert(
            "blk.0.attn_v.weight".to_owned(),
            weight("blk.0.attn_v.weight", &[2, 2], &[0.0, 65_504.0, 0.0, 0.0]),
        );
        model
    };
    let mut decoder = Qwen35::new(model(), 2)?;
    let mut fresh = Qwen35::new(model(), 2)?;
    let mut logits = Vec::new();
    decoder.evaluate_into(0, &mut logits)?;
    assert_eq!(logits, fresh.evaluate(0)?);
    let checkpoint = decoder.checkpoint()?;
    let prefix_bytes = decoder.checkpoint_bytes();
    // The finite projected value exceeds binary16's finite range after RMS
    // normalization. Rejection must leave both the prefix and position intact.
    assert!(decoder.evaluate_into(1, &mut logits).is_err());
    assert_eq!(logits, Vec::<f32>::new());
    assert_eq!(decoder.checkpoint_bytes(), prefix_bytes);
    let expected = fresh.evaluate(0)?;
    assert_eq!(decoder.evaluate(0)?, expected);
    assert!(decoder.evaluate(0).is_err());
    decoder.restore(&checkpoint)?;
    assert_eq!(decoder.evaluate(0)?, expected);
    Ok(())
}

#[test]
fn state_admission_charges_binary16_cache_and_binary32_recurrent_bytes() -> Result<()> {
    let model = |head: usize, recurrent: bool| -> Result<Model> {
        let mut model = if recurrent {
            hybrid_fixture()
        } else {
            fixture(false)
        };
        model
            .metadata
            .insert("qwen35.context_length".to_owned(), metadata!(1_048_576));
        for key in [
            "qwen35.attention.key_length",
            "qwen35.attention.value_length",
        ] {
            model
                .metadata
                .insert(key.to_owned(), MetadataValue::from(u64::try_from(head)?));
        }
        for (suffix, dimensions) in [
            ("attn_q.weight", vec![2, 2 * head]),
            ("attn_k.weight", vec![2, head]),
            ("attn_v.weight", vec![2, head]),
            ("attn_output.weight", vec![head, 2]),
            ("attn_q_norm.weight", vec![head]),
            ("attn_k_norm.weight", vec![head]),
        ] {
            let name = format!("blk.0.{suffix}");
            let values = vec![1.0; dimensions.iter().product()];
            model
                .tensors
                .insert(name.clone(), weight(&name, &dimensions, &values));
        }
        Ok(model)
    };
    // At 1M tokens, 1024 key/value elements exactly fill the 4 GiB state
    // budget in binary16. Admission remains lazy and allocates no KV prefix.
    let decoder = Qwen35::new(model(1024, false)?, 1_048_576)?;
    assert!(decoder.checkpoint_bytes() < 1_048_576);
    assert!(Qwen35::new(model(1025, false)?, 1_048_576).is_err());
    // Adding any recurrent state exceeds that exact bound; a smaller cache
    // leaves room for its convolution and matrix, which remain binary32.
    assert!(Qwen35::new(model(1024, true)?, 1_048_576).is_err());
    assert!(Qwen35::new(model(1023, true)?, 1_048_576).is_ok());
    Ok(())
}

#[test]
fn explicit_state_budget_admits_exact_hybrid_payload_and_preserves_execution() -> Result<()> {
    let cases: [(fn() -> Model, usize); 3] = [
        (|| fixture(false), 128),
        (|| fixture(true), 40),
        (hybrid_fixture, 168),
    ];
    for (model, payload_bytes) in cases {
        assert!(
            Qwen35::with_state_budget(model(), 16, payload_bytes - 1, MatvecBackend::Scalar)
                .is_err()
        );
        let mut reference = Qwen35::new(model(), 16)?;
        let mut bounded =
            Qwen35::with_state_budget(model(), 16, payload_bytes, MatvecBackend::Scalar)?;
        let checkpoint = bounded.checkpoint()?;
        for token in [0, 1].into_iter().cycle().take(16) {
            assert_eq!(bounded.evaluate(token)?, reference.evaluate(token)?);
        }
        assert!(bounded.evaluate(0).is_err());
        bounded.restore(&checkpoint)?;
        reference.reset()?;
        assert_eq!(bounded.evaluate(1)?, reference.evaluate(1)?);
    }
    for budget in [0, bonsai_prism_runtime::graph::MAX_STATE_BYTES + 1] {
        assert!(
            Qwen35::with_state_budget(fixture(false), 16, budget, MatvecBackend::Auto).is_err()
        );
    }
    Ok(())
}

#[test]
fn checkpoint_rejects_foreign_cache_contract_and_model() -> Result<()> {
    let mut first = Qwen35::new(fixture(false), 16)?;
    first.evaluate(0)?;
    let checkpoint = first.checkpoint()?;
    let mut other_context = Qwen35::new(fixture(false), 8)?;
    assert!(other_context.restore(&checkpoint).is_err());
    let mut other_instance = Qwen35::new(fixture(false), 16)?;
    assert!(other_instance.restore(&checkpoint).is_err());
    let mut other_model = fixture(false);
    other_model.identity = "different-model".to_owned();
    assert!(Qwen35::new(other_model, 16)?.restore(&checkpoint).is_err());
    Ok(())
}

#[test]
fn invalid_tokens_context_and_weights_fail_without_advancing() -> Result<()> {
    let mut decoder = Qwen35::new(fixture(false), 1)?;
    assert!(decoder.evaluate(2).is_err());
    let expected = Qwen35::new(fixture(false), 1)?.evaluate(0)?;
    assert_eq!(decoder.evaluate(0)?, expected);
    assert!(decoder.evaluate(0).is_err());
    let mut bad_shape = fixture(false);
    bad_shape.tensors.insert(
        "blk.0.attn_q.weight".to_owned(),
        weight("blk.0.attn_q.weight", &[2, 2], &[0.0; 4]),
    );
    assert!(Qwen35::new(bad_shape, 1).is_err());
    let mut bad_value = fixture(true);
    bad_value.tensors.insert(
        "blk.0.ssm_a".to_owned(),
        weight("blk.0.ssm_a", &[1], &[f32::NAN]),
    );
    assert!(Qwen35::new(bad_value, 1).is_err());
    Ok(())
}

#[test]
fn malformed_and_unsupported_metadata_is_rejected() {
    for (key, value) in [
        ("qwen35.full_attention_interval", metadata!(0)),
        ("qwen35.ssm.group_count", metadata!(0)),
        ("qwen35.rope.dimension_sections", metadata!([1, 1, 0, 0])),
        (
            "qwen35.rope.dimension_sections",
            metadata!([u64::MAX, 1, 0, 0]),
        ),
        ("qwen35.rope.scaling.type", metadata!("yarn")),
        ("qwen35.attention.causal", metadata!(false)),
        ("prism.hadamard.block_size", metadata!(2)),
    ] {
        let mut model = fixture(false);
        model.metadata.insert(key.to_owned(), value);
        assert!(Qwen35::new(model, 1).is_err(), "accepted {key}");
    }
}

#[test]
fn unsupported_rope_overrides_and_unconsumed_tensors_fail_closed() {
    for (key, value) in [
        ("qwen35.rope.scale_linear", metadata!(2.0)),
        ("qwen35.rope.scaling.alpha", metadata!(2.0)),
        ("prism.hadamard.unknown_transform", metadata!(true)),
    ] {
        let mut model = fixture(false);
        model.metadata.insert(key.to_owned(), value);
        assert!(Qwen35::new(model, 1).is_err(), "accepted {key}");
    }
    let mut model = fixture(false);
    model.tensors.insert(
        "output.bias".to_owned(),
        weight("output.bias", &[2], &[1.0, 2.0]),
    );
    assert!(Qwen35::new(model, 1).is_err());
}

#[test]
fn hadamard_embedding_inverse_and_output_rotation_preserve_basis() -> Result<()> {
    let mut plain = Qwen35::new(fixture(false), 16)?;
    let mut latent = fixture(false);
    let root_half = 1.0_f32 / 2.0_f32.sqrt();
    latent.tensors.insert(
        "token_embd.weight".to_owned(),
        weight(
            "token_embd.weight",
            &[2, 2],
            &[root_half, root_half, root_half, -root_half],
        ),
    );
    latent.tensors.insert(
        "output.weight".to_owned(),
        weight(
            "output.weight",
            &[2, 2],
            &[root_half, root_half, root_half, -root_half],
        ),
    );
    for (key, value) in [
        ("prism.hadamard.version", metadata!(1)),
        ("prism.hadamard.block_size", metadata!(2)),
        (
            "prism.hadamard.transform",
            metadata!("normalized-sylvester-walsh-hadamard"),
        ),
        ("prism.hadamard.axis", metadata!("input-last-dimension")),
        ("prism.hadamard.sign_mode", metadata!("identity")),
        ("prism.hadamard.weight_names", metadata!(["output.weight"])),
        (
            "prism.hadamard.inverse_weight_names",
            metadata!(["token_embd.weight"]),
        ),
    ] {
        latent.metadata.insert(key.to_owned(), value);
    }
    let mut transformed = Qwen35::new(latent, 16)?;
    for token in [0, 1, 1, 0] {
        let expected = plain.evaluate(token)?;
        let actual = transformed.evaluate(token)?;
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 0.000_001);
        }
    }
    Ok(())
}

#[test]
fn explicit_signs_require_complete_valid_widths() {
    let mut model = fixture(false);
    for (key, value) in [
        ("prism.hadamard.version", metadata!(1)),
        ("prism.hadamard.block_size", metadata!(2)),
        (
            "prism.hadamard.transform",
            metadata!("normalized-sylvester-walsh-hadamard"),
        ),
        ("prism.hadamard.axis", metadata!("input-last-dimension")),
        ("prism.hadamard.sign_mode", metadata!("explicit")),
        ("prism.hadamard.weight_names", metadata!(["output.weight"])),
        (
            "prism.hadamard.sign_widths",
            MetadataValue::Array(MetadataArray::U64(Vec::new())),
        ),
        (
            "prism.hadamard.sign_values",
            MetadataValue::Array(MetadataArray::U64(Vec::new())),
        ),
    ] {
        model.metadata.insert(key.to_owned(), value);
    }
    assert!(Qwen35::new(model, 1).is_err());
}

#[test]
fn tied_latent_embedding_with_explicit_signs_restores_primal_basis() -> Result<()> {
    let mut plain = Qwen35::new(fixture(false), 16)?;
    let mut latent = fixture(false);
    let half = 1.0_f32 / 2.0_f32.sqrt();
    latent.tensors.remove("output.weight");
    // H * diag([-1,+1]) applied to the two primal embedding rows.
    latent.tensors.insert(
        "token_embd.weight".to_owned(),
        weight("token_embd.weight", &[2, 2], &[-half, -half, half, -half]),
    );
    for (key, value) in [
        ("prism.hadamard.version", metadata!(2)),
        ("prism.hadamard.tied_output", metadata!(true)),
        ("prism.hadamard.block_size", metadata!(2)),
        (
            "prism.hadamard.transform",
            metadata!("normalized-sylvester-walsh-hadamard"),
        ),
        ("prism.hadamard.axis", metadata!("input-last-dimension")),
        ("prism.hadamard.sign_mode", metadata!("explicit")),
        ("prism.hadamard.sign_widths", metadata!([2])),
        ("prism.hadamard.sign_values", metadata!([-1, 1])),
        (
            "prism.hadamard.weight_names",
            metadata!(["blk.0.ffn_up.weight"]),
        ),
        (
            "prism.hadamard.inverse_weight_names",
            metadata!(["token_embd.weight"]),
        ),
    ] {
        latent.metadata.insert(key.to_owned(), value);
    }
    let mut transformed = Qwen35::new(latent, 16)?;
    for token in [1, 0, 1, 1] {
        let expected = plain.evaluate(token)?;
        let actual = transformed.evaluate(token)?;
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 0.000_001);
        }
    }
    Ok(())
}

fn grouped_gdn_models() -> (Model, Model) {
    let mut model = fixture(true);
    model
        .metadata
        .insert("qwen35.ssm.group_count".to_owned(), metadata!(2));
    model
        .metadata
        .insert("qwen35.ssm.time_step_rank".to_owned(), metadata!(4));
    model
        .metadata
        .insert("qwen35.ssm.inner_size".to_owned(), metadata!(8));
    let qkv: Vec<f32> = (0..16)
        .flat_map(|index| {
            if index % 2 == 0 {
                [1.0, 0.25]
            } else {
                [0.25, 1.0]
            }
        })
        .collect();
    model.tensors.insert(
        "blk.0.attn_qkv.weight".to_owned(),
        weight("blk.0.attn_qkv.weight", &[2, 16], &qkv),
    );
    let gate = [
        1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5, 0.25, 0.0, 0.0, 0.25, 0.75, 0.0, 0.0, 0.75,
    ];
    model.tensors.insert(
        "blk.0.attn_gate.weight".to_owned(),
        weight("blk.0.attn_gate.weight", &[2, 8], &gate),
    );
    model.tensors.insert(
        "blk.0.ssm_conv1d.weight".to_owned(),
        weight("blk.0.ssm_conv1d.weight", &[2, 16], &[0.25, 1.0].repeat(16)),
    );
    for (name, dims, values) in [
        ("blk.0.ssm_dt.bias", vec![4], vec![0.0; 4]),
        ("blk.0.ssm_a", vec![4], vec![-1.0; 4]),
        ("blk.0.ssm_beta.weight", vec![2, 4], vec![0.0; 8]),
        ("blk.0.ssm_alpha.weight", vec![2, 4], vec![0.0; 8]),
    ] {
        model
            .tensors
            .insert(name.to_owned(), weight(name, &dims, &values));
    }
    let first_row = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0];
    let second_row = [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125];
    let output = [first_row, second_row].concat();
    model.tensors.insert(
        "blk.0.ssm_out.weight".to_owned(),
        weight("blk.0.ssm_out.weight", &[8, 2], &output),
    );
    let mut latent = Model {
        metadata: model.metadata.clone(),
        tensors: model.tensors.clone(),
        identity: model.identity.clone(),
    };
    // P maps [key0/repeat0,key1/repeat0,key0/repeat1,key1/repeat1]
    // to [key0/repeat0,key0/repeat1,key1/repeat0,key1/repeat1].
    let half = 1.0_f32 / 2.0_f32.sqrt();
    let mut folded = Vec::new();
    for original in [first_row, second_row] {
        for head in [0, 2, 1, 3] {
            let first = original[2 * head];
            let second = original[2 * head + 1];
            folded.extend([(first + second) * half, (first - second) * half]);
        }
    }
    latent.tensors.insert(
        "blk.0.ssm_out.weight".to_owned(),
        weight("blk.0.ssm_out.weight", &[8, 2], &folded),
    );
    for (key, value) in [
        ("prism.hadamard.version", metadata!(1)),
        ("prism.hadamard.block_size", metadata!(2)),
        (
            "prism.hadamard.transform",
            metadata!("normalized-sylvester-walsh-hadamard"),
        ),
        ("prism.hadamard.axis", metadata!("input-last-dimension")),
        ("prism.hadamard.sign_mode", metadata!("identity")),
        (
            "prism.hadamard.weight_names",
            metadata!(["blk.0.ssm_out.weight"]),
        ),
        ("prism.hadamard.gdn_v_grouped", metadata!(true)),
    ] {
        latent.metadata.insert(key.to_owned(), value);
    }
    (model, latent)
}

#[test]
fn grouped_gdn_value_rotation_matches_unfolded_output_weights() -> Result<()> {
    let (model, latent) = grouped_gdn_models();
    let mut plain = Qwen35::new(model, 16)?;
    let mut transformed = Qwen35::new(latent, 16)?;
    for token in [0, 1, 0, 1] {
        let expected = plain.evaluate(token)?;
        let actual = transformed.evaluate(token)?;
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 0.000_002);
        }
    }
    Ok(())
}

#[test]
fn rank_one_grouped_ssm_branch_retains_original_untiled_input_order() -> Result<()> {
    let (mut control, mut base) = grouped_gdn_models();
    let target = "blk.0.ssm_out.weight";
    base.tensors
        .insert(target.into(), weight(target, &[8, 2], &[0.0; 16]));
    let a = [1.0, 0.0, 0.0, -0.25, 0.0, 0.5, 0.0, 0.0];
    let b = [0.5, -0.25];
    let mut factors = rank_one_factors(&base, &[target], 0.0);
    let a_name = format!("{target}.lora_a");
    let b_name = format!("{target}.lora_b");
    factors
        .tensors
        .insert(a_name.clone(), weight(&a_name, &[8, 1], &a));
    factors
        .tensors
        .insert(b_name.clone(), weight(&b_name, &[1, 2], &b));
    let matrix: Vec<_> = b
        .iter()
        .flat_map(|b| a.iter().map(move |a| a * b))
        .collect();
    control
        .tensors
        .insert(target.into(), weight(target, &[8, 2], &matrix));
    let mut actual = rank_one_graph(base, factors, 1.0)?;
    let mut expected = Qwen35::new(control, 16)?;
    for token in [0, 1, 0, 1] {
        assert_eq!(actual.evaluate(token)?, expected.evaluate(token)?);
        assert_eq!(graph_bits(&actual)?, graph_bits(&expected)?);
    }
    Ok(())
}

#[cfg(feature = "cli")]
fn wire_text(output: &mut Vec<u8>, text: &str) -> Result<()> {
    output.extend(u64::try_from(text.len())?.to_le_bytes());
    output.extend(text.as_bytes());
    Ok(())
}

#[cfg(feature = "cli")]
fn wire_value(value: MetadataValueRef<'_>) -> Result<(u32, Vec<u8>)> {
    let mut output = Vec::new();
    let kind = match value {
        MetadataValueRef::Bool(value) => {
            output.push(u8::from(value));
            7
        }
        MetadataValueRef::String(value) => {
            wire_text(&mut output, value)?;
            8
        }
        MetadataValueRef::Unsigned(value) => {
            output.extend(value.to_le_bytes());
            10
        }
        MetadataValueRef::Signed(value) => {
            output.extend(value.to_le_bytes());
            11
        }
        MetadataValueRef::Float(value) => {
            output.extend(value.to_le_bytes());
            12
        }
        MetadataValueRef::Array(values) => {
            let first = values.get(0).ok_or("fixture array must be nonempty")?;
            let (kind, _) = wire_value(first)?;
            output.extend(kind.to_le_bytes());
            output.extend(u64::try_from(values.len())?.to_le_bytes());
            for value in values.iter() {
                let (item_kind, bytes) = wire_value(value)?;
                if item_kind != kind {
                    return Err("fixture array types differ".into());
                }
                output.extend(bytes);
            }
            9
        }
    };
    Ok((kind, output))
}

#[cfg(feature = "cli")]
fn wire_model(model: &Model) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    output.extend(0x4655_4747_u32.to_le_bytes());
    output.extend(3_u32.to_le_bytes());
    output.extend(u64::try_from(model.tensors.len())?.to_le_bytes());
    output.extend(u64::try_from(model.metadata.len())?.to_le_bytes());
    for (key, value) in &model.metadata {
        wire_text(&mut output, key)?;
        // Tokenizer contracts require these exact GGUF integer widths.
        let (kind, bytes) = match value {
            MetadataValue::U32(value) => (4, value.to_le_bytes().to_vec()),
            MetadataValue::Array(MetadataArray::I32(values)) => {
                let mut bytes = Vec::new();
                bytes.extend(5_u32.to_le_bytes());
                bytes.extend(u64::try_from(values.len())?.to_le_bytes());
                bytes.extend(values.iter().flat_map(|value| value.to_le_bytes()));
                (9, bytes)
            }
            MetadataValue::Array(MetadataArray::String(values)) => {
                let mut bytes = Vec::new();
                bytes.extend(8_u32.to_le_bytes());
                bytes.extend(u64::try_from(values.len())?.to_le_bytes());
                for value in values {
                    wire_text(&mut bytes, value)?;
                }
                (9, bytes)
            }
            _ => wire_value(value.as_ref())?,
        };
        output.extend(kind.to_le_bytes());
        output.extend(bytes);
    }
    let mut payload = Vec::new();
    for tensor in model.tensors.values() {
        payload.resize(payload.len().div_ceil(32) * 32, 0);
        wire_text(&mut output, &tensor.name)?;
        output.extend(u32::try_from(tensor.dimensions.len())?.to_le_bytes());
        for &dimension in &tensor.dimensions {
            output.extend(u64::try_from(dimension)?.to_le_bytes());
        }
        output.extend(tensor.kind.to_le_bytes());
        output.extend(u64::try_from(payload.len())?.to_le_bytes());
        payload.extend_from_slice(&tensor.data);
    }
    output.resize(output.len().div_ceil(32) * 32, 0);
    output.extend(payload);
    Ok(output)
}

#[cfg(feature = "cli")]
static NEXT_DRIVER_ARTIFACT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(feature = "cli")]
struct DriverArtifact {
    directory: std::path::PathBuf,
    path: std::path::PathBuf,
    bytes: u64,
    digest: String,
}

#[cfg(feature = "cli")]
impl DriverArtifact {
    fn new(model: &Model) -> Result<Self> {
        use sha2::{Digest, Sha256};
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let sequence = NEXT_DRIVER_ARTIFACT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "Bonsai-NativeDriver-{}-{nonce}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir(&directory)?;
        let bytes = wire_model(model)?;
        let artifact = Self {
            path: directory.join("fixture.gguf"),
            directory,
            bytes: u64::try_from(bytes.len())?,
            digest: Sha256::digest(&bytes)
                .into_iter()
                .flat_map(|byte| [byte >> 4, byte & 15])
                .map(|nibble| char::from(b"0123456789abcdef"[usize::from(nibble)]))
                .collect(),
        };
        std::fs::write(&artifact.path, bytes)?;
        Ok(artifact)
    }

    fn command(&self, max_tokens: usize, valid_digest: bool) -> std::process::Command {
        self.command_input(max_tokens, valid_digest, "--tokens", "0,1")
    }

    fn command_input(
        &self,
        max_tokens: usize,
        valid_digest: bool,
        input_option: &str,
        input: &str,
    ) -> std::process::Command {
        let digest = if valid_digest {
            self.digest.clone()
        } else {
            "0".repeat(64)
        };
        let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_prism-inference"));
        command
            .current_dir(&self.directory)
            .arg("--model")
            .arg(&self.path)
            .args([
                "--bytes",
                &self.bytes.to_string(),
                "--sha256",
                &digest,
                "--memory-bytes",
                "8388608",
                "--context",
                "16",
                "--checkpoint-bytes",
                "1048576",
                "--max-tokens",
                &max_tokens.to_string(),
                input_option,
                input,
            ]);
        command
    }
}

#[cfg(feature = "cli")]
impl Drop for DriverArtifact {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
        let _ = std::fs::remove_dir(&self.directory);
    }
}

#[cfg(feature = "cli")]
#[test]
#[allow(
    clippy::too_many_lines,
    reason = "Keep the complete driver invocation and JSON output contract together."
)]
fn native_driver_executes_verified_synthetic_gguf_and_enforces_state_budget() -> Result<()> {
    let mut model = fixture(false);
    model.metadata.insert(
        "tokenizer.ggml.eos_token_id".to_owned(),
        MetadataValue::U32(0),
    );
    let artifact = DriverArtifact::new(&model)?;
    for (backend, state_bytes, valid_digest, success) in [
        (None, None, true, true),
        (Some("auto"), None, true, true),
        (Some("scalar"), None, true, true),
        (Some("neon"), None, true, true),
        (None, Some(128_usize), true, true),
        (None, Some(127), true, false),
        (None, None, false, false),
    ] {
        let mut command = artifact.command(2, valid_digest);
        if let Some(backend) = backend {
            command.args(["--backend", backend, "--timeout-seconds", "60"]);
        }
        if backend == Some("neon") {
            command.args(["--prefill-block-tokens", "4"]);
        }
        if let Some(bytes) = state_bytes {
            command.args(["--state-bytes", &bytes.to_string()]);
        }
        let output = command.output()?;
        assert_eq!(
            output.status.success(),
            success,
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        if success {
            let records = String::from_utf8(output.stdout)?
                .lines()
                .map(serde_json::from_str::<Value>)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            assert_eq!(
                records,
                [
                    serde_json::json!({"token": 1}),
                    serde_json::json!({"token": 1}),
                    serde_json::json!({
                        "finish_reason": "length",
                        "prompt_tokens": 2,
                        "generated_tokens": 2,
                        "reused_tokens": 0,
                        "stop_tokens": [0],
                        "sampling": { "mode": "greedy" },
                        "backend_requested": backend.unwrap_or("auto"),
                        "state_budget_bytes": state_bytes.unwrap_or(bonsai_prism_runtime::graph::MAX_STATE_BYTES),
                        "prefill_block_tokens": if backend == Some("neon") { 4 } else { 1 },
                        "backend_statistics": {
                            "scope": "decoder_lifetime_including_provisional_work",
                            "preparations": 19,
                            "preparation_failures": 0,
                            "matrix_attempts": 31,
                            "matrix_successes": 31,
                            "rejected_before_dispatch": 0,
                            "scalar_operations": 31,
                            "neon_operations": 0,
                            "fallbacks": {
                                "scalar_requested": if backend == Some("scalar") { 31 } else { 0 },
                                "tensor_kind_unsupported": if backend == Some("scalar") { 0 } else { 31 },
                                "feature_disabled": 0,
                                "target_unsupported": 0,
                            },
                        },
                        "backend_last_operation": {
                            "selected": "scalar",
                            "neon_build_eligible": cfg!(all(
                                feature = "experimental-neon",
                                target_arch = "aarch64",
                                target_os = "windows",
                                target_env = "msvc",
                                target_feature = "neon"
                            )),
                            "fallback_reason": if backend == Some("scalar") {
                                "scalar_requested"
                            } else {
                                "tensor_kind_unsupported"
                            },
                            "dispatched": true,
                            "succeeded": true,
                        },
                    })
                ]
            );
        } else {
            assert_eq!(output.stdout, Vec::<u8>::new());
            if state_bytes.is_some() {
                assert!(String::from_utf8_lossy(&output.stderr).contains("state budget"));
            }
        }
    }
    Ok(())
}

#[cfg(feature = "cli")]
fn text_driver_fixture(eos: u32) -> Model {
    let mut model = fixture(false);
    for (key, value) in [
        ("tokenizer.ggml.model", metadata!("gpt2")),
        ("tokenizer.ggml.pre", metadata!("qwen35")),
        ("tokenizer.ggml.tokens", metadata!(["a", "Ġb"])),
        (
            "tokenizer.ggml.token_type",
            MetadataValue::Array(MetadataArray::I32(vec![1, 1])),
        ),
        ("tokenizer.ggml.eos_token_id", MetadataValue::U32(eos)),
    ] {
        model.metadata.insert(key.to_owned(), value);
    }
    model
}

#[cfg(feature = "cli")]
fn driver_records(output: std::process::Output) -> Result<Vec<Value>> {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(String::from_utf8(output.stdout)?
        .lines()
        .map(serde_json::from_str::<Value>)
        .collect::<std::result::Result<Vec<_>, _>>()?)
}

#[cfg(feature = "cli")]
#[test]
fn native_driver_text_prompt_matches_exact_ids_and_reserves_complete_output() -> Result<()> {
    for (tokens, kinds, prompt, special_policy, expected_ids) in [
        (["a", "b"], [1_i32, 1], "aa", None, "0,0"),
        (["a", "<stop>"], [1, 3], "a<stop>", Some("parse"), "0,1"),
    ] {
        let mut model = text_driver_fixture(1);
        model
            .metadata
            .insert("tokenizer.ggml.tokens".to_owned(), tokens.to_vec().into());
        model.metadata.insert(
            "tokenizer.ggml.token_type".to_owned(),
            kinds.to_vec().into(),
        );
        model.metadata.insert(
            "tokenizer.ggml.merges".to_owned(),
            Vec::<String>::new().into(),
        );
        let artifact = DriverArtifact::new(&model)?;
        let expected = driver_records(
            artifact
                .command_input(2, true, "--tokens", expected_ids)
                .output()?,
        )?;
        let mut command = artifact.command_input(2, true, "--prompt", prompt);
        if let Some(policy) = special_policy {
            command.args(["--special-input", policy]);
        }
        let mut actual = driver_records(command.output()?)?;
        let completion = actual.last_mut().ok_or("missing completion")?;
        assert_eq!(completion["input_mode"], "text");
        assert_eq!(
            completion["special_input"],
            special_policy.unwrap_or("literal")
        );
        let object = completion
            .as_object_mut()
            .ok_or("completion is not an object")?;
        object.remove("input_mode");
        object.remove("special_input");
        object.remove("insert_special");
        assert_eq!(actual, expected);
        if special_policy.is_some() {
            let rejected = artifact
                .command_input(2, true, "--prompt", prompt)
                .output()?;
            assert!(!rejected.status.success());
            assert_eq!(rejected.stdout, Vec::<u8>::new());
        }
        for (text, succeeds) in [("a".repeat(14), true), ("a".repeat(15), false)] {
            let result = artifact
                .command_input(2, true, "--prompt", &text)
                .output()?;
            assert_eq!(
                result.status.success(),
                succeeds,
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
            if !succeeds {
                assert_eq!(result.stdout, Vec::<u8>::new());
            }
        }
    }
    Ok(())
}

#[cfg(feature = "cli")]
#[test]
fn native_driver_declared_insertion_matches_exact_ids_and_tokenization_count() -> Result<()> {
    let mut model = text_driver_fixture(1);
    model.metadata.insert(
        "tokenizer.ggml.bos_token_id".to_owned(),
        MetadataValue::U32(1),
    );
    for key in [
        "tokenizer.ggml.add_bos_token",
        "tokenizer.ggml.add_eos_token",
    ] {
        model.metadata.insert(key.to_owned(), true.into());
    }
    model.metadata.insert(
        "tokenizer.ggml.merges".to_owned(),
        Vec::<String>::new().into(),
    );
    let artifact = DriverArtifact::new(&model)?;
    let expected = driver_records(
        artifact
            .command_input(2, true, "--tokens", "1,0,0,1")
            .output()?,
    )?;
    let mut actual = driver_records(
        artifact
            .command_input(2, true, "--prompt", "aa")
            .args(["--insert-special", "model"])
            .output()?,
    )?;
    let completion = actual.last_mut().ok_or("missing insertion completion")?;
    assert_eq!(completion["prompt_tokens"], 4);
    assert_eq!(completion["insert_special"], "model");
    let object = completion.as_object_mut().ok_or("invalid completion")?;
    for key in ["input_mode", "special_input", "insert_special"] {
        object.remove(key);
    }
    assert_eq!(actual, expected);

    let counted = std::process::Command::new(env!("CARGO_BIN_EXE_prism-inference"))
        .current_dir(&artifact.directory)
        .arg("--model")
        .arg(&artifact.path)
        .args([
            "--mode",
            "tokenize",
            "--bytes",
            &artifact.bytes.to_string(),
            "--sha256",
            &artifact.digest,
            "--memory-bytes",
            "8388608",
            "--context",
            "16",
            "--max-tokens",
            "2",
            "--prompt",
            "aa",
            "--insert-special",
            "model",
        ])
        .output()?;
    let records = driver_records(counted)?;
    assert_eq!(records.len(), 1);
    assert_eq!(records[0]["token_ids"], serde_json::json!([1, 0, 0, 1]));
    assert_eq!(records[0]["token_count"], 4);
    assert_eq!(records[0]["reserved_output_tokens"], 2);
    assert_eq!(records[0]["insert_special"], "model");

    for (length, succeeds) in [(12, true), (13, false)] {
        let output = artifact
            .command_input(2, true, "--prompt", &"a".repeat(length))
            .args(["--insert-special", "model"])
            .output()?;
        assert_eq!(output.status.success(), succeeds);
        if !succeeds {
            assert_eq!(output.stdout, Vec::<u8>::new());
        }
    }
    Ok(())
}

#[cfg(feature = "cli")]
#[test]
fn native_driver_streams_verified_vocabulary_bytes_and_text() -> Result<()> {
    let artifact = DriverArtifact::new(&text_driver_fixture(0))?;
    let records = driver_records(
        artifact
            .command(1, true)
            .args(["--output", "text"])
            .output()?,
    )?;
    assert_eq!(records.len(), 2);
    assert_eq!(
        records[0],
        serde_json::json!({"token": 1, "bytes": [32, 98], "text": " b", "special": false})
    );
    let completion = &records[1];
    assert_eq!(completion["finish_reason"], "length");
    assert_eq!(completion["prompt_tokens"], 2);
    assert_eq!(completion["generated_tokens"], 1);
    assert_eq!(completion["stop_tokens"], serde_json::json!([0]));
    assert_eq!(completion["text"], "");
    Ok(())
}

#[cfg(feature = "cli")]
#[test]
fn native_driver_reference_sampling_matches_seeded_runtime_and_reports_configuration() -> Result<()>
{
    use bonsai_prism_runtime::sampling::{ReferenceSampler, Sampling};

    let artifact = DriverArtifact::new(&text_driver_fixture(0))?;
    for seed in [0, 42] {
        let options = Sampling {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
        };
        let mut sampler = ReferenceSampler::new(options, seed)?;
        let mut runtime = Runtime::new(
            Qwen35::new(text_driver_fixture(0), 16)?,
            Limits {
                context_tokens: 16,
                checkpoints: 0,
                checkpoint_bytes: 1_048_576,
            },
        )?;
        let mut expected_tokens = Vec::new();
        let completion = runtime.generate_sampled(
            &[0, 1],
            &Generation {
                max_tokens: 6,
                stop_tokens: vec![0],
            },
            Prefill { block_tokens: 4 },
            &Cancellation::default(),
            &mut sampler,
            |token| {
                expected_tokens.push(token);
                Ok(())
            },
        )?;
        let run = || {
            artifact
                .command(6, true)
                .args([
                    "--sampling",
                    "reference",
                    "--seed",
                    &seed.to_string(),
                    "--temperature",
                    "1",
                    "--top-k",
                    "0",
                    "--top-p",
                    "1",
                    "--prefill-block-tokens",
                    "4",
                    "--timeout-seconds",
                    "60",
                ])
                .output()
        };
        let records = driver_records(run()?)?;
        assert_eq!(records, driver_records(run()?)?);
        assert_eq!(records.len(), expected_tokens.len() + 1);
        for (record, token) in records.iter().zip(&expected_tokens) {
            assert_eq!(record["token"], *token);
        }
        let final_record = records.last().ok_or("missing driver completion")?;
        assert_eq!(
            final_record["generated_tokens"],
            completion.generated_tokens
        );
        assert_eq!(
            final_record["sampling"],
            serde_json::json!({
                "mode": "reference", "rng": "splitmix64", "seed": seed,
                "temperature": 1.0, "top_k": 0, "top_p": 1.0,
            })
        );
    }
    Ok(())
}

#[cfg(feature = "cli")]
#[test]
fn native_driver_stops_on_model_ids_and_additive_deduplicated_user_ids() -> Result<()> {
    for (eos, other_stop, user_stops) in [
        (1, None, None),
        (0, Some("tokenizer.ggml.eot_token_id"), None),
        (0, Some("tokenizer.ggml.eom_token_id"), None),
        (0, None, Some("1,0,1")),
    ] {
        let mut model = text_driver_fixture(eos);
        if let Some(key) = other_stop {
            model.metadata.insert(key.to_owned(), MetadataValue::U32(1));
        }
        let artifact = DriverArtifact::new(&model)?;
        let mut command = artifact.command(2, true);
        if let Some(ids) = user_stops {
            command.args(["--stop-tokens", ids]);
        }
        let records = driver_records(command.output()?)?;
        assert_eq!(records.len(), 1);
        let completion = &records[0];
        assert_eq!(completion["finish_reason"], "stop");
        assert_eq!(completion["prompt_tokens"], 2);
        assert_eq!(completion["generated_tokens"], 0);
        assert_eq!(
            completion["stop_tokens"],
            if eos == 1 {
                serde_json::json!([1])
            } else {
                serde_json::json!([0, 1])
            }
        );
        assert!(completion.get("text").is_none());
        assert!(completion.get("token").is_none());
    }
    Ok(())
}

#[cfg(feature = "cli")]
#[test]
fn native_driver_rejects_missing_mistyped_and_out_of_vocabulary_model_eos() -> Result<()> {
    for eos in [
        None,
        Some(MetadataValue::U64(0)),
        Some(metadata!("0")),
        Some(MetadataValue::U32(2)),
    ] {
        let mut model = fixture(false);
        if let Some(eos) = eos {
            model
                .metadata
                .insert("tokenizer.ggml.eos_token_id".to_owned(), eos);
        }
        let artifact = DriverArtifact::new(&model)?;
        let output = artifact.command(2, true).output()?;
        assert!(!output.status.success());
        assert_eq!(output.stdout, Vec::<u8>::new());
        assert_ne!(output.stderr, Vec::<u8>::new());
    }
    Ok(())
}

#[cfg(feature = "cli")]
#[test]
fn native_driver_rejects_text_vocabulary_mismatch_before_emitting() -> Result<()> {
    for (key, value) in [
        ("tokenizer.ggml.tokens", metadata!(["a"])),
        (
            "tokenizer.ggml.token_type",
            MetadataValue::Array(MetadataArray::I32(vec![1])),
        ),
    ] {
        let mut model = text_driver_fixture(0);
        model.metadata.insert(key.to_owned(), value);
        let artifact = DriverArtifact::new(&model)?;
        let output = artifact
            .command(2, true)
            .args(["--output", "text"])
            .output()?;
        assert!(!output.status.success());
        assert_eq!(output.stdout, Vec::<u8>::new());
        assert_ne!(output.stderr, Vec::<u8>::new());
    }
    Ok(())
}

#[test]
fn rank_one_scale_reset_clears_live_state_and_invalidates_same_scale_and_change_back_checkpoints()
-> Result<()> {
    for recurrent in [false, true] {
        let target = if recurrent {
            "blk.0.ssm_out.weight"
        } else {
            "blk.0.attn_output.weight"
        };
        let base = fixture(recurrent);
        let factors = rank_one_factors(&base, &[target, "blk.0.ffn_down.weight"], 0.125);
        let mut graph = rank_one_graph(base, factors, 1.3)?;
        graph.evaluate(0)?;
        let first = graph.checkpoint()?;
        for scale in [1.3, 0.0, -0.0, 2.0, 1.3] {
            let previous = graph.checkpoint()?;
            graph.reset_with_adapter_scale(scale)?;
            assert_eq!(
                graph.adapter_scale().map(f32::to_bits),
                Some(scale.to_bits())
            );
            assert!(graph.restore(&previous).is_err());
            assert!(graph.restore(&first).is_err());
            assert!(graph_bits(&graph)?.iter().all(|bits| *bits == 0));
            let base = fixture(recurrent);
            let factors = rank_one_factors(&base, &[target, "blk.0.ffn_down.weight"], 0.125);
            let mut fresh = rank_one_graph(base, factors, scale)?;
            assert_eq!(graph.identity(), fresh.identity());
            assert_eq!(graph.evaluate(1)?, fresh.evaluate(1)?);
            assert_eq!(graph_bits(&graph)?, graph_bits(&fresh)?);
        }
    }
    Ok(())
}

#[test]
fn rank_one_runtime_scale_reset_drops_prefixes_and_retains_cancel_and_failure_rollback()
-> Result<()> {
    let base = fixture(true);
    let factors = rank_one_factors(&base, &["blk.0.ssm_out.weight"], 0.125);
    let mut runtime = Runtime::new(
        rank_one_graph(base, factors, 1.3)?,
        Limits {
            context_tokens: 16,
            checkpoints: 2,
            checkpoint_bytes: 1024 * 1024,
        },
    )?;
    let options = Generation {
        max_tokens: 1,
        stop_tokens: vec![],
    };
    runtime.generate(&[0], &options, &Cancellation::default(), |_| Ok(()))?;
    assert_ne!(runtime.tokens(), &[0_u32; 0]);
    assert_eq!(runtime.checkpoint_count(), 1);
    runtime.reset_with_adapter_scale(0.0)?;
    assert_eq!(runtime.tokens(), &[0_u32; 0]);
    assert_eq!(runtime.checkpoint_count(), 0);
    let empty = graph_bits(runtime.decoder())?;
    let cancelled = Cancellation::default();
    cancelled.cancel();
    assert!(
        runtime
            .generate(&[1], &options, &cancelled, |_| Err("must not emit".into()))
            .is_err()
    );
    assert_eq!(graph_bits(runtime.decoder())?, empty);
    assert!(
        runtime
            .generate(&[1], &options, &Cancellation::default(), |_| Err(
                "failed delivery".into()
            ))
            .is_err()
    );
    assert_eq!(graph_bits(runtime.decoder())?, empty);
    assert_eq!(runtime.tokens(), &[0_u32; 0]);
    assert!(!runtime.is_faulted());
    let completion = runtime.generate(&[1], &options, &Cancellation::default(), |_| Ok(()))?;
    assert_eq!(completion.reused_tokens, 0);
    assert_eq!(runtime.decoder().adapter_scale(), Some(0.0));
    Ok(())
}

#[test]
fn rank_one_invalid_scale_or_absent_adapter_rejects_before_mutation() -> Result<()> {
    let base = fixture(true);
    let factors = rank_one_factors(&base, &["blk.0.ssm_out.weight"], 0.125);
    let mut graph = rank_one_graph(base, factors, 1.3)?;
    graph.evaluate(0)?;
    let before = graph_bits(&graph)?;
    let identity = graph.identity().to_owned();
    let checkpoint = graph.checkpoint()?;
    for scale in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.01, 2.01] {
        assert!(graph.reset_with_adapter_scale(scale).is_err());
        assert_eq!(graph.identity(), identity);
        assert_eq!(graph_bits(&graph)?, before);
        graph.restore(&checkpoint)?;
    }
    for adapter in [false, true] {
        let base = fixture(true);
        let graph = if adapter {
            let factors = rank_one_factors(&base, &["blk.0.ssm_out.weight"], 0.125);
            rank_one_graph(base, factors, 1.3)?
        } else {
            Qwen35::new(base, 16)?
        };
        let mut runtime = Runtime::new(
            graph,
            Limits {
                context_tokens: 16,
                checkpoints: 1,
                checkpoint_bytes: 1024 * 1024,
            },
        )?;
        runtime.generate(
            &[0],
            &Generation {
                max_tokens: 1,
                stop_tokens: vec![],
            },
            &Cancellation::default(),
            |_| Ok(()),
        )?;
        let tokens = runtime.tokens().to_vec();
        let bits = graph_bits(runtime.decoder())?;
        assert!(
            runtime
                .reset_with_adapter_scale(if adapter { f32::NAN } else { 0.0 })
                .is_err()
        );
        assert_eq!(runtime.tokens(), tokens);
        assert_eq!(graph_bits(runtime.decoder())?, bits);
        assert_eq!(runtime.checkpoint_count(), 1);
        assert!(!runtime.is_faulted());
    }
    Ok(())
}

fn profile_runtime(recurrent: bool, scale: Option<f32>) -> Result<Runtime<Qwen35>> {
    let base = fixture(recurrent);
    let decoder = if let Some(scale) = scale {
        let target = if recurrent {
            "blk.0.ssm_out.weight"
        } else {
            "blk.0.attn_output.weight"
        };
        let factors = rank_one_factors(&base, &[target, "blk.0.ffn_down.weight"], 0.125);
        rank_one_graph(base, factors, scale)?
    } else {
        Qwen35::new(base, 16)?
    };
    Runtime::new(
        decoder,
        Limits {
            context_tokens: 16,
            checkpoints: 0,
            checkpoint_bytes: 1024 * 1024,
        },
    )
}

#[test]
fn profile_reuse_matches_fresh_state_and_sampled_failure_replay() -> Result<()> {
    use bonsai_prism_runtime::sampling::{ReferenceSampler, Sampling};

    for recurrent in [false, true] {
        for scale in [None, Some(1.3)] {
            let mut runtime = profile_runtime(recurrent, scale)?;
            let mut sampler = ReferenceSampler::new(Sampling::default(), 42)?;
            let options = Generation {
                max_tokens: 2,
                stop_tokens: vec![],
            };
            let prefill = Prefill { block_tokens: 4 };
            for case in 0..5 {
                let previous_tokens = runtime.tokens().to_vec();
                let prompt = match case {
                    0 => vec![0],
                    1 => [previous_tokens.clone(), vec![1]].concat(),
                    2 => vec![1], // Edited history must recompute, never truncate recurrence.
                    3 => vec![0, 1], // Switched history has no retained snapshot.
                    _ => previous_tokens.clone(),
                };
                runtime.prepare_profile_request(scale, PromptReuse::Latest)?;
                let mut reference_sampler = sampler.clone();
                profile_request_failures(&mut runtime, &prompt, &options, &mut sampler)?;
                let mut reference = profile_runtime(recurrent, scale)?;
                let mut expected = Vec::new();
                reference.generate_sampled(
                    &prompt,
                    &options,
                    prefill,
                    &Cancellation::default(),
                    &mut reference_sampler,
                    |token| {
                        expected.push(token);
                        Ok(())
                    },
                )?;
                let mut actual = Vec::new();
                let completion = runtime.generate_sampled(
                    &prompt,
                    &options,
                    prefill,
                    &Cancellation::default(),
                    &mut sampler,
                    |token| {
                        actual.push(token);
                        Ok(())
                    },
                )?;
                assert_eq!(actual, expected);
                assert_eq!(runtime.tokens(), reference.tokens());
                assert_eq!(
                    graph_bits(runtime.decoder())?,
                    graph_bits(reference.decoder())?
                );
                assert_eq!(
                    completion.reused_tokens,
                    if prompt.starts_with(&previous_tokens) {
                        previous_tokens.len()
                    } else {
                        0
                    }
                );
                assert_eq!(runtime.checkpoint_count(), 0);
            }
            if scale.is_none() {
                let tokens = runtime.tokens().to_vec();
                let bits = graph_bits(runtime.decoder())?;
                assert!(
                    runtime
                        .prepare_profile_request(Some(0.0), PromptReuse::Latest)
                        .is_err()
                );
                assert_eq!(runtime.tokens(), tokens);
                assert_eq!(graph_bits(runtime.decoder())?, bits);
            }
            runtime.prepare_profile_request(scale, PromptReuse::Disabled)?;
            assert!(runtime.tokens().is_empty());
        }
    }
    Ok(())
}

fn profile_request_failures(
    runtime: &mut Runtime<Qwen35>,
    prompt: &[u32],
    options: &Generation,
    sampler: &mut bonsai_prism_runtime::sampling::ReferenceSampler,
) -> Result<()> {
    let tokens = runtime.tokens().to_vec();
    let bits = graph_bits(runtime.decoder())?;
    for cancel in [false, true] {
        let cancellation = Cancellation::default();
        assert!(
            runtime
                .generate_sampled(
                    prompt,
                    options,
                    Prefill { block_tokens: 4 },
                    &cancellation,
                    sampler,
                    |_| {
                        if cancel {
                            cancellation.cancel();
                            Ok(())
                        } else {
                            Err("abandoned provisional output".into())
                        }
                    },
                )
                .is_err()
        );
        assert_eq!(runtime.tokens(), tokens);
        assert_eq!(graph_bits(runtime.decoder())?, bits);
        assert!(!runtime.is_faulted());
    }
    Ok(())
}

#[test]
fn profile_reuse_preserves_only_exact_scale_bits_and_explicit_reset_boundary() -> Result<()> {
    let options = Generation {
        max_tokens: 1,
        stop_tokens: vec![],
    };
    for recurrent in [false, true] {
        let mut runtime = profile_runtime(recurrent, Some(1.3))?;
        for scale in [1.3_f32, 1.3, 0.0, -0.0, 2.0, 1.3] {
            let old_scale = runtime.decoder().adapter_scale();
            let tokens = runtime.tokens().to_vec();
            let bits = graph_bits(runtime.decoder())?;
            runtime.prepare_profile_request(Some(scale), PromptReuse::Latest)?;
            if old_scale.map(f32::to_bits) == Some(scale.to_bits()) {
                assert_eq!(runtime.tokens(), tokens);
                assert_eq!(graph_bits(runtime.decoder())?, bits);
            } else {
                assert!(runtime.tokens().is_empty());
                assert!(graph_bits(runtime.decoder())?.iter().all(|&bits| bits == 0));
            }
            assert_eq!(
                runtime.decoder().adapter_scale().map(f32::to_bits),
                Some(scale.to_bits())
            );
            runtime.generate(&[0], &options, &Cancellation::default(), |_| Ok(()))?;
            let mut reference = profile_runtime(recurrent, Some(scale))?;
            reference.generate(&[0], &options, &Cancellation::default(), |_| Ok(()))?;
            assert_eq!(runtime.tokens(), reference.tokens());
            assert_eq!(
                graph_bits(runtime.decoder())?,
                graph_bits(reference.decoder())?
            );
            let tokens = runtime.tokens().to_vec();
            let bits = graph_bits(runtime.decoder())?;
            for invalid in [None, Some(f32::NAN), Some(-1.0)] {
                assert!(
                    runtime
                        .prepare_profile_request(invalid, PromptReuse::Latest)
                        .is_err()
                );
                assert_eq!(runtime.tokens(), tokens);
                assert_eq!(graph_bits(runtime.decoder())?, bits);
                assert!(!runtime.is_faulted());
            }
        }
        runtime.prepare_profile_request(Some(1.3), PromptReuse::Disabled)?;
        assert!(runtime.tokens().is_empty());
        runtime.generate(&[0], &options, &Cancellation::default(), |_| Ok(()))?;
        runtime.reset()?;
        runtime.prepare_profile_request(Some(1.3), PromptReuse::Latest)?;
        assert_eq!(
            runtime
                .generate(&[0], &options, &Cancellation::default(), |_| Ok(()))?
                .reused_tokens,
            0
        );
    }
    Ok(())
}
