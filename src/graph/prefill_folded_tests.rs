//! Bit-exact folded speculative verification parity, adapters, carry flush, and cancellation tests.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::Arc;

use bonsai_gguf::MetadataValue;

use super::*;
use crate::Decoder;
use crate::graph::KvCachePolicy;
use crate::model::Model;
use crate::model::lora::{MultiRankAdapter, MultiRankPolicy};

fn f32_tensor(name: &str, dimensions: &[usize], values: &[f32]) -> Tensor {
    Tensor {
        name: name.to_owned(),
        dimensions: dimensions.to_vec(),
        kind: 0,
        data: Arc::<[u8]>::from(
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .into(),
    }
}

fn pattern(len: usize, seed: usize, scale: f32) -> Vec<f32> {
    const TABLE: [f32; 8] = [0.25, -0.375, 0.5, -0.125, 0.3125, -0.25, 0.1875, -0.5];
    (0..len)
        .map(|i| TABLE[(i.wrapping_mul(5).wrapping_add(seed)) % TABLE.len()] * scale)
        .collect()
}

struct HybridConfig {
    vocab: usize,
    embedding: usize,
    ffn: usize,
    heads: usize,
    kv_heads: usize,
    head: usize,
    rope: usize,
    rope_sections: [u32; 4],
    ssm_head: usize,
    ssm_keys: usize,
    ssm_values: usize,
    conv: usize,
    context: usize,
}

impl HybridConfig {
    const fn small_source() -> Self {
        Self {
            vocab: 6,
            embedding: 4,
            ffn: 4,
            heads: 1,
            kv_heads: 1,
            head: 6,
            rope: 6,
            rope_sections: [1, 1, 1, 0],
            ssm_head: 2,
            ssm_keys: 1,
            ssm_values: 2,
            conv: 2,
            context: 32,
        }
    }

    const fn pinned_swift_aligned() -> Self {
        Self {
            vocab: 6,
            embedding: 256,
            ffn: 256,
            heads: 6,
            kv_heads: 1,
            head: 256,
            rope: 64,
            rope_sections: [11, 11, 10, 0],
            ssm_head: 128,
            ssm_keys: 1,
            ssm_values: 2,
            conv: 4,
            context: 32,
        }
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "Build explicit hybrid recurrent + full-attention Qwen35 test models with exact tensor layouts."
)]
fn hybrid_model(cfg: &HybridConfig) -> Model {
    let key_w = cfg.ssm_keys * cfg.ssm_head;
    let val_w = cfg.ssm_values * cfg.ssm_head;
    let conv_w = 2 * key_w + val_w;
    let q_gate_w = cfg.heads * cfg.head * 2;
    let kv_w = cfg.kv_heads * cfg.head;
    let attn_out_in = cfg.heads * cfg.head;

    let metadata = BTreeMap::from([
        (
            "general.architecture".to_owned(),
            MetadataValue::from("qwen35"),
        ),
        ("qwen35.block_count".to_owned(), MetadataValue::from(2_u32)),
        (
            "qwen35.context_length".to_owned(),
            MetadataValue::from(u32::try_from(cfg.context).unwrap_or(32)),
        ),
        (
            "qwen35.embedding_length".to_owned(),
            MetadataValue::from(u32::try_from(cfg.embedding).unwrap_or(4)),
        ),
        (
            "qwen35.feed_forward_length".to_owned(),
            MetadataValue::from(u32::try_from(cfg.ffn).unwrap_or(4)),
        ),
        (
            "qwen35.attention.head_count".to_owned(),
            MetadataValue::from(u32::try_from(cfg.heads).unwrap_or(1)),
        ),
        (
            "qwen35.attention.head_count_kv".to_owned(),
            MetadataValue::from(u32::try_from(cfg.kv_heads).unwrap_or(1)),
        ),
        (
            "qwen35.attention.key_length".to_owned(),
            MetadataValue::from(u32::try_from(cfg.head).unwrap_or(6)),
        ),
        (
            "qwen35.attention.value_length".to_owned(),
            MetadataValue::from(u32::try_from(cfg.head).unwrap_or(6)),
        ),
        (
            "qwen35.attention.layer_norm_rms_epsilon".to_owned(),
            MetadataValue::from(1e-6_f32),
        ),
        (
            "qwen35.rope.dimension_count".to_owned(),
            MetadataValue::from(u32::try_from(cfg.rope).unwrap_or(6)),
        ),
        (
            "qwen35.rope.dimension_sections".to_owned(),
            MetadataValue::from(cfg.rope_sections.to_vec()),
        ),
        (
            "qwen35.rope.freq_base".to_owned(),
            MetadataValue::from(10_000.0_f32),
        ),
        (
            "qwen35.ssm.state_size".to_owned(),
            MetadataValue::from(u32::try_from(cfg.ssm_head).unwrap_or(2)),
        ),
        (
            "qwen35.ssm.group_count".to_owned(),
            MetadataValue::from(u32::try_from(cfg.ssm_keys).unwrap_or(1)),
        ),
        (
            "qwen35.ssm.time_step_rank".to_owned(),
            MetadataValue::from(u32::try_from(cfg.ssm_values).unwrap_or(2)),
        ),
        (
            "qwen35.ssm.inner_size".to_owned(),
            MetadataValue::from(u32::try_from(val_w).unwrap_or(4)),
        ),
        (
            "qwen35.ssm.conv_kernel".to_owned(),
            MetadataValue::from(u32::try_from(cfg.conv).unwrap_or(2)),
        ),
        (
            "qwen35.full_attention_interval".to_owned(),
            MetadataValue::from(2_u32),
        ),
        (
            "qwen35.attention.recurrent_layers".to_owned(),
            MetadataValue::from(vec![true, false]),
        ),
    ]);

    let mut tensors = BTreeMap::new();
    let mut insert = |name: &str, dims: &[usize], vals: Vec<f32>| {
        tensors.insert(name.to_owned(), f32_tensor(name, dims, &vals));
    };

    insert(
        "token_embd.weight",
        &[cfg.embedding, cfg.vocab],
        pattern(cfg.embedding * cfg.vocab, 1, 0.5),
    );
    insert(
        "output.weight",
        &[cfg.embedding, cfg.vocab],
        pattern(cfg.embedding * cfg.vocab, 2, 0.5),
    );
    insert(
        "output_norm.weight",
        &[cfg.embedding],
        vec![1.0; cfg.embedding],
    );

    // Layer 0: Recurrent (GDN + conv)
    insert(
        "blk.0.attn_norm.weight",
        &[cfg.embedding],
        vec![1.0; cfg.embedding],
    );
    insert(
        "blk.0.post_attention_norm.weight",
        &[cfg.embedding],
        vec![1.0; cfg.embedding],
    );
    insert(
        "blk.0.attn_qkv.weight",
        &[cfg.embedding, conv_w],
        pattern(cfg.embedding * conv_w, 3, 0.25),
    );
    insert(
        "blk.0.attn_gate.weight",
        &[cfg.embedding, val_w],
        pattern(cfg.embedding * val_w, 4, 0.25),
    );
    insert(
        "blk.0.ssm_conv1d.weight",
        &[cfg.conv, conv_w],
        pattern(cfg.conv * conv_w, 5, 0.25),
    );
    insert(
        "blk.0.ssm_dt.bias",
        &[cfg.ssm_values],
        vec![0.125; cfg.ssm_values],
    );
    insert(
        "blk.0.ssm_a",
        &[cfg.ssm_values],
        vec![-0.25; cfg.ssm_values],
    );
    insert(
        "blk.0.ssm_beta.weight",
        &[cfg.embedding, cfg.ssm_values],
        pattern(cfg.embedding * cfg.ssm_values, 6, 0.25),
    );
    insert(
        "blk.0.ssm_alpha.weight",
        &[cfg.embedding, cfg.ssm_values],
        pattern(cfg.embedding * cfg.ssm_values, 7, 0.25),
    );
    insert(
        "blk.0.ssm_norm.weight",
        &[cfg.ssm_head],
        vec![1.0; cfg.ssm_head],
    );
    insert(
        "blk.0.ssm_out.weight",
        &[val_w, cfg.embedding],
        pattern(val_w * cfg.embedding, 8, 0.25),
    );
    insert(
        "blk.0.ffn_up.weight",
        &[cfg.embedding, cfg.ffn],
        pattern(cfg.embedding * cfg.ffn, 9, 0.25),
    );
    insert(
        "blk.0.ffn_gate.weight",
        &[cfg.embedding, cfg.ffn],
        pattern(cfg.embedding * cfg.ffn, 10, 0.25),
    );
    insert(
        "blk.0.ffn_down.weight",
        &[cfg.ffn, cfg.embedding],
        pattern(cfg.ffn * cfg.embedding, 11, 0.25),
    );

    // Layer 1: Full attention
    insert(
        "blk.1.attn_norm.weight",
        &[cfg.embedding],
        vec![1.0; cfg.embedding],
    );
    insert(
        "blk.1.post_attention_norm.weight",
        &[cfg.embedding],
        vec![1.0; cfg.embedding],
    );
    insert(
        "blk.1.attn_q.weight",
        &[cfg.embedding, q_gate_w],
        pattern(cfg.embedding * q_gate_w, 12, 0.25),
    );
    insert(
        "blk.1.attn_k.weight",
        &[cfg.embedding, kv_w],
        pattern(cfg.embedding * kv_w, 13, 0.25),
    );
    insert(
        "blk.1.attn_v.weight",
        &[cfg.embedding, kv_w],
        pattern(cfg.embedding * kv_w, 14, 0.25),
    );
    insert(
        "blk.1.attn_output.weight",
        &[attn_out_in, cfg.embedding],
        pattern(attn_out_in * cfg.embedding, 15, 0.25),
    );
    insert("blk.1.attn_q_norm.weight", &[cfg.head], vec![1.0; cfg.head]);
    insert("blk.1.attn_k_norm.weight", &[cfg.head], vec![1.0; cfg.head]);
    insert(
        "blk.1.ffn_up.weight",
        &[cfg.embedding, cfg.ffn],
        pattern(cfg.embedding * cfg.ffn, 16, 0.25),
    );
    insert(
        "blk.1.ffn_gate.weight",
        &[cfg.embedding, cfg.ffn],
        pattern(cfg.embedding * cfg.ffn, 17, 0.25),
    );
    insert(
        "blk.1.ffn_down.weight",
        &[cfg.ffn, cfg.embedding],
        pattern(cfg.ffn * cfg.embedding, 18, 0.25),
    );

    Model {
        metadata,
        tensors,
        identity: "prefill-folded-hybrid-fixture".to_owned(),
    }
}

fn hybrid_model_with_identity(cfg: &HybridConfig, arithmetic: Arithmetic) -> Model {
    let mut model = hybrid_model(cfg);
    if matches!(arithmetic, Arithmetic::PinnedSwift) {
        model.identity = crate::model::swift::BONSAI_2_5_PQ2_SHA256.to_owned();
    }
    model
}

fn build_folded_test_adapters(model: &Model, cfg: &HybridConfig) -> Result<MultiRankAdapter> {
    let mut mr_tensors = BTreeMap::new();
    let val_w = cfg.ssm_values * cfg.ssm_head;
    let conv_w = 2 * cfg.ssm_keys * cfg.ssm_head + val_w;
    let q_gate_w = cfg.heads * cfg.head * 2;
    for (target, in_f, out_f, rank, seed) in [
        (
            "blk.0.ffn_down.weight",
            cfg.ffn,
            cfg.embedding,
            1_usize,
            21_usize,
        ),
        ("blk.0.attn_qkv.weight", cfg.embedding, conv_w, 2, 31),
        ("blk.1.attn_q.weight", cfg.embedding, q_gate_w, 2, 32),
    ] {
        mr_tensors.insert(
            format!("{target}.lora_a"),
            f32_tensor(
                &format!("{target}.lora_a"),
                &[in_f, rank],
                &pattern(in_f * rank, seed, 0.125),
            ),
        );
        mr_tensors.insert(
            format!("{target}.lora_b"),
            f32_tensor(
                &format!("{target}.lora_b"),
                &[rank, out_f],
                &pattern(rank * out_f, seed + 5, 0.125),
            ),
        );
    }
    let mr_model = Model {
        metadata: BTreeMap::from([
            (
                "general.architecture".to_owned(),
                MetadataValue::from("qwen35"),
            ),
            ("general.type".to_owned(), MetadataValue::from("adapter")),
            ("adapter.type".to_owned(), MetadataValue::from("lora")),
            (
                "adapter.lora_alpha".to_owned(),
                MetadataValue::from(2.0_f32),
            ),
        ]),
        tensors: mr_tensors,
        identity: "mr-folded-test".to_owned(),
    };
    MultiRankAdapter::from_model(model, &mr_model, MultiRankPolicy::AnyLinear)
}

fn configure_folded(
    model: Model,
    cfg: &HybridConfig,
    _arithmetic: Arithmetic,
    workers: usize,
    kv_policy: KvCachePolicy,
    with_stacked_adapters: bool,
) -> Result<Qwen35> {
    let mut decoder = if with_stacked_adapters {
        let mr = build_folded_test_adapters(&model, cfg)?;
        Qwen35::with_adapters_and_kv(
            model,
            cfg.context,
            64 * 1024 * 1024,
            MatvecBackend::Scalar,
            Some((mr, 0.5)),
            kv_policy,
        )?
    } else {
        Qwen35::with_state_budget_and_kv(
            model,
            cfg.context,
            64 * 1024 * 1024,
            MatvecBackend::Scalar,
            kv_policy,
        )?
    };
    let prefill_scratch = decoder.tiled_prefill_scratch_bytes(8)?;
    let parallelism = RowParallelism::new(workers)?;
    decoder = decoder.with_tiled_prefill(8, prefill_scratch)?;
    if let Some(tile) = decoder.tiled_prefill.as_mut() {
        tile.set_row_parallelism(parallelism);
    }
    let verify_scratch = decoder.tiled_verification_scratch_bytes()?;
    decoder = decoder.with_tiled_verification(verify_scratch)?;
    Ok(decoder)
}

fn assert_full_state_bit_eq(left: &Qwen35, right: &Qwen35) -> Result<()> {
    assert_eq!(left.position, right.position);
    assert_eq!(left.mrope_position, right.mrope_position);
    assert_eq!(left.state.len(), right.state.len());
    for (idx, (l, r)) in left.state.iter().zip(&right.state).enumerate() {
        match (l, r) {
            (
                LayerState::Recurrent {
                    convolution: lc,
                    matrix: lm,
                },
                LayerState::Recurrent {
                    convolution: rc,
                    matrix: rm,
                },
            ) => {
                assert_eq!(lc.len(), rc.len(), "layer {idx} conv len");
                assert!(
                    lc.iter().zip(rc).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "layer {idx} conv mismatch"
                );
                assert_eq!(lm.len(), rm.len(), "layer {idx} matrix len");
                assert!(
                    lm.iter().zip(rm).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "layer {idx} matrix mismatch"
                );
            }
            (
                LayerState::Full {
                    keys: lk,
                    values: lv,
                },
                LayerState::Full {
                    keys: rk,
                    values: rv,
                },
            ) => {
                assert_eq!(lk.len, rk.len, "layer {idx} key len");
                assert_eq!(lv.len, rv.len, "layer {idx} val len");
                let mut l_rows = Vec::new();
                let mut r_rows = Vec::new();
                left.visit_state(&mut |l_idx, comp, vals| {
                    if l_idx == idx {
                        l_rows.push((format!("{comp:?}"), vals.to_vec()));
                    }
                    Ok(())
                })?;
                right.visit_state(&mut |r_idx, comp, vals| {
                    if r_idx == idx {
                        r_rows.push((format!("{comp:?}"), vals.to_vec()));
                    }
                    Ok(())
                })?;
                assert_eq!(l_rows.len(), r_rows.len(), "layer {idx} kv row count");
                for ((lc, lv_row), (rc, rv_row)) in l_rows.iter().zip(&r_rows) {
                    assert_eq!(lc, rc);
                    assert!(
                        lv_row
                            .iter()
                            .zip(rv_row)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "layer {idx} {lc} kv mismatch"
                    );
                }
            }
            _ => return Err(invalid("layer state variant mismatch")),
        }
    }
    Ok(())
}

fn assert_logits_bit_eq(left: &[f32], right: &[f32], ctx: &str) {
    assert_eq!(left.len(), right.len(), "{ctx} logit len");
    for (idx, (a, b)) in left.iter().zip(right).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{ctx} logit[{idx}] bit mismatch: {a} != {b}"
        );
    }
}

#[test]
fn folded_verification_matches_sequential_zero_ulp_across_arithmetics_workers_and_kv() -> Result<()>
{
    #[allow(
        unused_mut,
        reason = "Parallel-rows worker cases are appended only when the parallel-rows feature is enabled."
    )]
    let mut cases = vec![
        (
            HybridConfig::small_source(),
            Arithmetic::Source,
            1,
            KvCachePolicy::F16,
            false,
        ),
        (
            HybridConfig::small_source(),
            Arithmetic::Source,
            1,
            KvCachePolicy::F16,
            true,
        ),
        (
            HybridConfig::pinned_swift_aligned(),
            Arithmetic::PinnedSwift,
            1,
            KvCachePolicy::F16,
            true,
        ),
        (
            HybridConfig::pinned_swift_aligned(),
            Arithmetic::PinnedSwift,
            1,
            KvCachePolicy::Rotated4,
            true,
        ),
    ];
    #[cfg(feature = "parallel-rows")]
    {
        cases.push((
            HybridConfig::small_source(),
            Arithmetic::Source,
            2,
            KvCachePolicy::F16,
            true,
        ));
        cases.push((
            HybridConfig::pinned_swift_aligned(),
            Arithmetic::PinnedSwift,
            2,
            KvCachePolicy::Rotated4,
            true,
        ));
    }

    for (cfg, arithmetic, workers, kv_policy, stacked) in cases {
        let m1 = hybrid_model_with_identity(&cfg, arithmetic);
        let m2 = hybrid_model_with_identity(&cfg, arithmetic);
        let mut seq = configure_folded(m1, &cfg, arithmetic, workers, kv_policy, stacked)?;
        let mut folded = configure_folded(m2, &cfg, arithmetic, workers, kv_policy, stacked)?;

        assert!(folded.supports_folded_verification());
        assert!(folded.identity().ends_with(":cpu-tiled-verify-v1"));
        assert!(folded.folded_verification_tape_bytes()? > 0);

        // Seed initial token in both decoders.
        let l_init_seq = seq.evaluate(0)?;
        let mut l_init_folded = Vec::new();
        folded.evaluate_into(0, &mut l_init_folded)?;
        assert_logits_bit_eq(&l_init_seq, &l_init_folded, "init token");

        // Round 1: block [1, 2, 3, 4], keep 2 ([1, 2]), reject 3.
        let mut rows = Vec::new();
        folded.evaluate_verification_block_folded(&[], &[1, 2, 3, 4], &mut rows, &|| Ok(()))?;
        let s1 = seq.evaluate(1)?;
        let s2 = seq.evaluate(2)?;
        assert_logits_bit_eq(&s1, &rows[0], "round1 tok 1");
        assert_logits_bit_eq(&s2, &rows[1], "round1 tok 2");
        folded.commit_folded_verification(2)?;
        assert!(folded.has_pending_folded_verification());

        // Round 2 folds carried [1, 2] into next block [5, 0, 1] and keeps full 3.
        folded.evaluate_verification_block_folded(&[1, 2], &[5, 0, 1], &mut rows, &|| Ok(()))?;
        let s5 = seq.evaluate(5)?;
        let s0 = seq.evaluate(0)?;
        let s1b = seq.evaluate(1)?;
        assert_logits_bit_eq(&s5, &rows[0], "round2 tok 5");
        assert_logits_bit_eq(&s0, &rows[1], "round2 tok 0");
        assert_logits_bit_eq(&s1b, &rows[2], "round2 tok 1");
        folded.commit_folded_verification(3)?;

        // Round 3 folds carried [5, 0, 1] into [2, 3] and keeps 0 (full discard!).
        folded.evaluate_verification_block_folded(&[5, 0, 1], &[2, 3], &mut rows, &|| Ok(()))?;
        folded.commit_folded_verification(0)?;
        assert!(!folded.has_pending_folded_verification());
        assert_full_state_bit_eq(&seq, &folded)?;

        // Round 4 keeps 1 ([4]), then explicitly flushes before next sequential evaluate().
        folded.evaluate_verification_block_folded(&[], &[4, 5], &mut rows, &|| Ok(()))?;
        let s4 = seq.evaluate(4)?;
        assert_logits_bit_eq(&s4, &rows[0], "round4 tok 4");
        folded.commit_folded_verification(1)?;
        assert!(folded.has_pending_folded_verification());
        folded.flush_folded_verification(&|| Ok(()))?;
        assert!(!folded.has_pending_folded_verification());
        assert_full_state_bit_eq(&seq, &folded)?;

        let s_tail_seq = seq.evaluate(2)?;
        let mut s_tail_folded = Vec::new();
        folded.evaluate_into(2, &mut s_tail_folded)?;
        assert_logits_bit_eq(&s_tail_seq, &s_tail_folded, "post-flush tail");
        assert_full_state_bit_eq(&seq, &folded)?;
    }
    Ok(())
}

#[test]
fn folded_verification_guards_observers_and_rolls_back_mid_block_cancellation() -> Result<()> {
    let cfg = HybridConfig::small_source();
    let mut seq = configure_folded(
        hybrid_model(&cfg),
        &cfg,
        Arithmetic::Source,
        1,
        KvCachePolicy::F16,
        false,
    )?;
    let mut folded = configure_folded(
        hybrid_model(&cfg),
        &cfg,
        Arithmetic::Source,
        1,
        KvCachePolicy::F16,
        false,
    )?;

    let mut rows = Vec::new();
    folded.evaluate_verification_block_folded(&[], &[1, 2, 3], &mut rows, &|| Ok(()))?;
    // Uncommitted pending block rejects observers, checkpointing, and serial eval.
    assert!(folded.checkpoint().is_err());
    let mut ckpt = seq.checkpoint()?;
    assert!(folded.checkpoint_into(&mut ckpt).is_err());
    assert!(folded.visit_state(&mut |_, _, _| Ok(())).is_err());
    assert!(folded.evaluate(0).is_err());
    assert!(folded.flush_folded_verification(&|| Ok(())).is_err());

    // Commit kept=2 -> deferred carry rejects observers/serial until flush.
    folded.commit_folded_verification(2)?;
    assert!(folded.has_pending_folded_verification());
    assert!(folded.checkpoint().is_err());
    assert!(folded.visit_state(&mut |_, _, _| Ok(())).is_err());
    assert!(folded.evaluate(0).is_err());

    // Advance reference seq to exact position 2 ([1, 2]).
    let _ = seq.evaluate(1)?;
    let _ = seq.evaluate(2)?;

    // Inject cancellation mid-tile inside Round 2 while carried [1, 2] is active.
    let checks = Cell::new(0_usize);
    let cancel_mid_tile = || {
        let n = checks.get() + 1;
        checks.set(n);
        if n >= 6 {
            return Err(invalid("injected mid-block cancel"));
        }
        Ok(())
    };
    let err =
        folded.evaluate_verification_block_folded(&[1, 2], &[3, 4], &mut rows, &cancel_mid_tile);
    assert!(err.is_err());
    assert_eq!(rows, Vec::<Vec<f32>>::new());
    assert!(!folded.has_pending_folded_verification());
    assert!(folded.state_consistent);
    assert_full_state_bit_eq(&seq, &folded)?;

    // Post-cancellation serial decode and checkpointing remain bit-exact and consistent.
    let c_seq = seq.checkpoint()?;
    let c_folded = folded.checkpoint()?;
    let _ = c_seq;
    let _ = c_folded;
    let next_seq = seq.evaluate(5)?;
    let mut next_folded = Vec::new();
    folded.evaluate_into(5, &mut next_folded)?;
    assert_logits_bit_eq(&next_seq, &next_folded, "post-cancel eval");
    assert_full_state_bit_eq(&seq, &folded)?;
    Ok(())
}

#[expect(
    clippy::too_many_lines,
    reason = "Build explicit 4-layer hybrid Qwen35 model with 256-channel full-attention heads for rsq4kvf tests."
)]
fn four_layer_hybrid_model(cfg: &HybridConfig, arithmetic: Arithmetic) -> Model {
    let key_w = cfg.ssm_keys * cfg.ssm_head;
    let val_w = cfg.ssm_values * cfg.ssm_head;
    let conv_w = 2 * key_w + val_w;
    let q_gate_w = cfg.heads * cfg.head * 2;
    let kv_w = cfg.kv_heads * cfg.head;
    let attn_out_in = cfg.heads * cfg.head;

    let mut model = hybrid_model_with_identity(cfg, arithmetic);
    model
        .metadata
        .insert("qwen35.block_count".to_owned(), MetadataValue::from(4_u32));
    model.metadata.insert(
        "qwen35.full_attention_interval".to_owned(),
        MetadataValue::from(4_u32),
    );
    model.metadata.insert(
        "qwen35.attention.recurrent_layers".to_owned(),
        MetadataValue::from(vec![true, true, true, false]),
    );

    let mut tensors = BTreeMap::new();
    for common in ["token_embd.weight", "output.weight", "output_norm.weight"] {
        if let Some(t) = model.tensors.get(common) {
            tensors.insert(common.to_owned(), t.clone());
        }
    }
    let mut insert = |name: &str, dims: &[usize], vals: Vec<f32>| {
        tensors.insert(name.to_owned(), f32_tensor(name, dims, &vals));
    };

    for layer in 0..4_usize {
        let p = format!("blk.{layer}");
        let seed_off = layer * 19;
        insert(
            &format!("{p}.attn_norm.weight"),
            &[cfg.embedding],
            vec![1.0; cfg.embedding],
        );
        insert(
            &format!("{p}.post_attention_norm.weight"),
            &[cfg.embedding],
            vec![1.0; cfg.embedding],
        );
        insert(
            &format!("{p}.ffn_up.weight"),
            &[cfg.embedding, cfg.ffn],
            pattern(cfg.embedding * cfg.ffn, 9 + seed_off, 0.25),
        );
        insert(
            &format!("{p}.ffn_gate.weight"),
            &[cfg.embedding, cfg.ffn],
            pattern(cfg.embedding * cfg.ffn, 10 + seed_off, 0.25),
        );
        insert(
            &format!("{p}.ffn_down.weight"),
            &[cfg.ffn, cfg.embedding],
            pattern(cfg.ffn * cfg.embedding, 11 + seed_off, 0.25),
        );
        if layer < 3 {
            insert(
                &format!("{p}.attn_qkv.weight"),
                &[cfg.embedding, conv_w],
                pattern(cfg.embedding * conv_w, 3 + seed_off, 0.25),
            );
            insert(
                &format!("{p}.attn_gate.weight"),
                &[cfg.embedding, val_w],
                pattern(cfg.embedding * val_w, 4 + seed_off, 0.25),
            );
            insert(
                &format!("{p}.ssm_conv1d.weight"),
                &[cfg.conv, conv_w],
                pattern(cfg.conv * conv_w, 5 + seed_off, 0.25),
            );
            insert(
                &format!("{p}.ssm_dt.bias"),
                &[cfg.ssm_values],
                vec![0.125; cfg.ssm_values],
            );
            insert(
                &format!("{p}.ssm_a"),
                &[cfg.ssm_values],
                vec![-0.25; cfg.ssm_values],
            );
            insert(
                &format!("{p}.ssm_beta.weight"),
                &[cfg.embedding, cfg.ssm_values],
                pattern(cfg.embedding * cfg.ssm_values, 6 + seed_off, 0.25),
            );
            insert(
                &format!("{p}.ssm_alpha.weight"),
                &[cfg.embedding, cfg.ssm_values],
                pattern(cfg.embedding * cfg.ssm_values, 7 + seed_off, 0.25),
            );
            insert(
                &format!("{p}.ssm_norm.weight"),
                &[cfg.ssm_head],
                vec![1.0; cfg.ssm_head],
            );
            insert(
                &format!("{p}.ssm_out.weight"),
                &[val_w, cfg.embedding],
                pattern(val_w * cfg.embedding, 8 + seed_off, 0.25),
            );
        } else {
            insert(
                &format!("{p}.attn_q.weight"),
                &[cfg.embedding, q_gate_w],
                pattern(cfg.embedding * q_gate_w, 12 + seed_off, 0.25),
            );
            insert(
                &format!("{p}.attn_k.weight"),
                &[cfg.embedding, kv_w],
                pattern(cfg.embedding * kv_w, 13 + seed_off, 0.25),
            );
            insert(
                &format!("{p}.attn_v.weight"),
                &[cfg.embedding, kv_w],
                pattern(cfg.embedding * kv_w, 14 + seed_off, 0.25),
            );
            insert(
                &format!("{p}.attn_output.weight"),
                &[attn_out_in, cfg.embedding],
                pattern(attn_out_in * cfg.embedding, 15 + seed_off, 0.25),
            );
            insert(
                &format!("{p}.attn_q_norm.weight"),
                &[cfg.head],
                vec![1.0; cfg.head],
            );
            insert(
                &format!("{p}.attn_k_norm.weight"),
                &[cfg.head],
                vec![1.0; cfg.head],
            );
        }
    }
    model.tensors = tensors;
    model
}

fn build_sparse_heterogeneous_folded_adapters(
    model: &Model,
    cfg: &HybridConfig,
) -> Result<MultiRankAdapter> {
    let val_w = cfg.ssm_values * cfg.ssm_head;
    let conv_w = 2 * cfg.ssm_keys * cfg.ssm_head + val_w;
    let q_gate_w = cfg.heads * cfg.head * 2;
    let kv_w = cfg.kv_heads * cfg.head;
    let attn_out_in = cfg.heads * cfg.head;

    let mut mr_tensors = BTreeMap::new();
    // blk.1.ffn_down.weight omitted intentionally; ranks vary across sites (2, 3, 4).
    for (idx, (target, in_f, out_f, rank)) in [
        ("blk.0.ffn_down.weight", cfg.ffn, cfg.embedding, 2_usize),
        ("blk.2.attn_qkv.weight", cfg.embedding, conv_w, 3),
        ("blk.3.ffn_down.weight", cfg.ffn, cfg.embedding, 4),
        ("blk.3.attn_q.weight", cfg.embedding, q_gate_w, 4),
        ("blk.3.attn_k.weight", cfg.embedding, kv_w, 4),
        ("blk.3.attn_v.weight", cfg.embedding, kv_w, 4),
        ("blk.3.attn_output.weight", attn_out_in, cfg.embedding, 4),
    ]
    .into_iter()
    .enumerate()
    {
        mr_tensors.insert(
            format!("{target}.lora_a"),
            f32_tensor(
                &format!("{target}.lora_a"),
                &[in_f, rank],
                &pattern(in_f * rank, 41 + idx, 0.125),
            ),
        );
        mr_tensors.insert(
            format!("{target}.lora_b"),
            f32_tensor(
                &format!("{target}.lora_b"),
                &[rank, out_f],
                &pattern(rank * out_f, 53 + idx, 0.125),
            ),
        );
    }
    let mr = MultiRankAdapter::from_model(
        model,
        &Model {
            metadata: BTreeMap::from([
                (
                    "general.architecture".to_owned(),
                    MetadataValue::from("qwen35"),
                ),
                ("general.type".to_owned(), MetadataValue::from("adapter")),
                ("adapter.type".to_owned(), MetadataValue::from("lora")),
                (
                    "adapter.lora.alpha".to_owned(),
                    MetadataValue::from(4.0_f32),
                ),
                (
                    "adapter.lora_alpha".to_owned(),
                    MetadataValue::from(4.0_f32),
                ),
            ]),
            tensors: mr_tensors,
            identity: "mr-folded-heterogeneous".to_owned(),
        },
        MultiRankPolicy::NpuFriendly,
    )?;
    assert_eq!(mr.max_rank(), 4);
    assert!(!mr.has_target("blk.1.ffn_down.weight"));
    Ok(mr)
}

fn configure_four_layer_folded(
    cfg: &HybridConfig,
    arithmetic: Arithmetic,
    workers: usize,
    mr_scale: Option<f32>,
) -> Result<Qwen35> {
    let model = four_layer_hybrid_model(cfg, arithmetic);
    let mr = build_sparse_heterogeneous_folded_adapters(&model, cfg)?;
    let mut decoder = Qwen35::with_adapters_and_kv(
        model,
        cfg.context,
        64 * 1024 * 1024,
        MatvecBackend::Scalar,
        mr_scale.map(|scale| (mr, scale)),
        KvCachePolicy::Rotated4FittedKv,
    )?;
    let prefill_scratch = decoder.tiled_prefill_scratch_bytes(8)?;
    let parallelism = RowParallelism::new(workers)?;
    decoder = decoder.with_tiled_prefill(8, prefill_scratch)?;
    if let Some(tile) = decoder.tiled_prefill.as_mut() {
        tile.set_row_parallelism(parallelism);
    }
    let verify_scratch = decoder.tiled_verification_scratch_bytes()?;
    decoder.with_tiled_verification(verify_scratch)
}

fn verify_folded_heterogeneous_zero_scale(
    cfg: &HybridConfig,
    arithmetic: Arithmetic,
    workers: usize,
) -> Result<Vec<f32>> {
    let mut orca_seq = configure_four_layer_folded(cfg, arithmetic, workers, None)?;
    let mut zero_seq = configure_four_layer_folded(cfg, arithmetic, workers, Some(0.0))?;
    let mut zero_folded = configure_four_layer_folded(cfg, arithmetic, workers, Some(0.0))?;

    let orca_l0 = orca_seq.evaluate(0)?;
    let zero_l0 = zero_seq.evaluate(0)?;
    assert_logits_bit_eq(&zero_l0, &orca_l0, "rsq4kvf scale=0.0 init vs Orca-only");
    let mut zf_l0 = Vec::new();
    zero_folded.evaluate_into(0, &mut zf_l0)?;
    assert_logits_bit_eq(
        &zf_l0,
        &orca_l0,
        "rsq4kvf scale=0.0 folded init vs Orca-only",
    );

    let mut rows = Vec::new();
    zero_folded.evaluate_verification_block_folded(&[], &[1, 2, 3, 4], &mut rows, &|| Ok(()))?;
    let orca_l1 = orca_seq.evaluate(1)?;
    let orca_l2 = orca_seq.evaluate(2)?;
    assert_logits_bit_eq(&rows[0], &orca_l1, "rsq4kvf scale=0.0 round1 tok 1");
    assert_logits_bit_eq(&rows[1], &orca_l2, "rsq4kvf scale=0.0 round1 tok 2");
    zero_folded.commit_folded_verification(2)?;
    zero_folded.evaluate_verification_block_folded(&[1, 2], &[5, 0, 1], &mut rows, &|| Ok(()))?;
    let orca_l5 = orca_seq.evaluate(5)?;
    assert_logits_bit_eq(&rows[0], &orca_l5, "rsq4kvf scale=0.0 round2 tok 5");
    zero_folded.commit_folded_verification(1)?;
    zero_folded.flush_folded_verification(&|| Ok(()))?;
    assert_full_state_bit_eq(&orca_seq, &zero_folded)?;
    Ok(orca_l0)
}

#[test]
fn folded_speculative_verification_with_stacked_orca_sparse_heterogeneous_multi_rank_and_rsq4kvf()
-> Result<()> {
    let cfg = HybridConfig::pinned_swift_aligned();
    #[allow(
        unused_mut,
        reason = "Parallel-rows worker cases are appended only when the parallel-rows feature is enabled."
    )]
    let mut workers_list = vec![1_usize];
    #[cfg(feature = "parallel-rows")]
    workers_list.push(2);

    for arithmetic in [Arithmetic::Source, Arithmetic::PinnedSwift] {
        for &workers in &workers_list {
            let orca_l0 = verify_folded_heterogeneous_zero_scale(&cfg, arithmetic, workers)?;

            // Scale 1.0: bit-exact parity between folded speculative verification and eager single-token decode.
            let mut rows = Vec::new();
            let mut seq = configure_four_layer_folded(&cfg, arithmetic, workers, Some(1.0))?;
            let mut folded = configure_four_layer_folded(&cfg, arithmetic, workers, Some(1.0))?;
            let seq_l0 = seq.evaluate(0)?;
            assert_ne!(
                seq_l0.iter().copied().map(f32::to_bits).collect::<Vec<_>>(),
                orca_l0
                    .iter()
                    .copied()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>()
            );
            let mut fold_l0 = Vec::new();
            folded.evaluate_into(0, &mut fold_l0)?;
            assert_logits_bit_eq(&fold_l0, &seq_l0, "rsq4kvf scale=1.0 init");

            folded.evaluate_verification_block_folded(&[], &[1, 2, 3, 4], &mut rows, &|| Ok(()))?;
            let s1 = seq.evaluate(1)?;
            let s2 = seq.evaluate(2)?;
            assert_logits_bit_eq(&rows[0], &s1, "rsq4kvf scale=1.0 round1 tok 1");
            assert_logits_bit_eq(&rows[1], &s2, "rsq4kvf scale=1.0 round1 tok 2");
            folded.commit_folded_verification(2)?;

            folded.evaluate_verification_block_folded(
                &[1, 2],
                &[5, 0, 1],
                &mut rows,
                &|| Ok(()),
            )?;
            let s5 = seq.evaluate(5)?;
            let s0 = seq.evaluate(0)?;
            let s1b = seq.evaluate(1)?;
            assert_logits_bit_eq(&rows[0], &s5, "rsq4kvf scale=1.0 round2 tok 5");
            assert_logits_bit_eq(&rows[1], &s0, "rsq4kvf scale=1.0 round2 tok 0");
            assert_logits_bit_eq(&rows[2], &s1b, "rsq4kvf scale=1.0 round2 tok 1");
            folded.commit_folded_verification(3)?;

            folded.evaluate_verification_block_folded(
                &[5, 0, 1],
                &[2, 3],
                &mut rows,
                &|| Ok(()),
            )?;
            folded.commit_folded_verification(0)?;
            assert!(!folded.has_pending_folded_verification());
            assert_full_state_bit_eq(&seq, &folded)?;

            folded.evaluate_verification_block_folded(&[], &[4, 5], &mut rows, &|| Ok(()))?;
            let s4 = seq.evaluate(4)?;
            assert_logits_bit_eq(&rows[0], &s4, "rsq4kvf scale=1.0 round4 tok 4");
            folded.commit_folded_verification(1)?;
            folded.flush_folded_verification(&|| Ok(()))?;
            assert_full_state_bit_eq(&seq, &folded)?;

            let tail_seq = seq.evaluate(2)?;
            let mut tail_folded = Vec::new();
            folded.evaluate_into(2, &mut tail_folded)?;
            assert_logits_bit_eq(&tail_folded, &tail_seq, "rsq4kvf scale=1.0 tail");
            assert_full_state_bit_eq(&seq, &folded)?;
        }
    }
    Ok(())
}

fn pq2_mixed_hybrid_model(cfg: &HybridConfig) -> Model {
    let mut model = hybrid_model(cfg);
    for (idx, (name, tensor)) in model.tensors.iter_mut().enumerate() {
        if let &[cols, rows] = tensor.dimensions.as_slice()
            && name.starts_with("blk.")
            && cols >= 128
            && cols.is_multiple_of(128)
        {
            let mut block = [0x55_u8; 34];
            let scale = 0x3400_u16.wrapping_add(u16::try_from((idx % 7) * 0x0100).unwrap_or(0));
            block[..2].copy_from_slice(&scale.to_le_bytes());
            for (j, byte) in block[2..].iter_mut().enumerate() {
                *byte = u8::try_from((idx * 17 + j * 31) & 0xff).unwrap_or(0x55);
            }
            tensor.kind = crate::kernels::PQ2;
            tensor.data = Arc::<[u8]>::from(block.repeat((cols / 128) * rows)).into();
        }
    }
    model
}

struct MockDecodeOnlyExecutor {
    decode_calls: Arc<std::sync::atomic::AtomicUsize>,
    released: Vec<crate::model::TensorBytes>,
}

impl MockDecodeOnlyExecutor {
    fn cpu_f32_matvec(
        tensor: &Tensor,
        activation: &[f32],
        pinned: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        let mut scratch = if pinned {
            crate::kernels::MatvecScratch::with_pinned_backend(MatvecBackend::Scalar)
        } else {
            crate::kernels::MatvecScratch::with_backend(MatvecBackend::Scalar)
        };
        crate::kernels::matvec_into(tensor, activation, output, &mut scratch)
    }
}

impl crate::kernels::Pq2ProjectionExecutor for MockDecodeOnlyExecutor {
    fn identity(&self) -> &'static str {
        "mock-decode-only-hexagon-executor"
    }

    fn supports(&self, _tensor: &Tensor) -> bool {
        false
    }

    fn supports_tile_tokens(&self) -> bool {
        false
    }

    fn project(
        &mut self,
        _tensor: &Tensor,
        _activation: &[crate::kernels::Q8KBlock],
        _fused: bool,
        _output: &mut Vec<f32>,
    ) -> Result<()> {
        Err(invalid("unexpected quantized mock call"))
    }

    fn supports_f32(&self, tensor: &Tensor) -> bool {
        matches!(tensor.kind, crate::kernels::PQ2 | crate::kernels::PTQ1)
            && tensor.dimensions.len() == 2
            && tensor.name.starts_with("blk.")
    }

    fn project_f32(
        &mut self,
        tensor: &Tensor,
        activation: &[f32],
        pinned: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        self.decode_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self::cpu_f32_matvec(tensor, activation, pinned, output)
    }

    fn project_f32_tokens(
        &mut self,
        _tensor: &Tensor,
        _activations: &[Vec<f32>],
        _pinned: bool,
        _outputs: &mut [Vec<f32>],
    ) -> Result<()> {
        Err(invalid(
            "decode-only executor must not receive tile token projections",
        ))
    }

    fn release_host_weight_binding(
        &mut self,
        previous_tensor: &Tensor,
        released_data: &crate::model::TensorBytes,
    ) -> bool {
        if self.supports_f32(previous_tensor)
            && released_data.is_empty()
            && !released_data.is_mapped()
        {
            self.released.push(released_data.clone());
            true
        } else {
            false
        }
    }

    fn has_released_host_weight(&self, tensor: &Tensor) -> bool {
        tensor.data.is_empty()
            && self.supports_f32(tensor)
            && self
                .released
                .iter()
                .any(|token| crate::model::TensorBytes::ptr_eq(token, &tensor.data))
    }
}

fn build_decode_only_mixed_session(
    model: Model,
    context: usize,
    decode_calls: Arc<std::sync::atomic::AtomicUsize>,
) -> Result<Qwen35> {
    let mut mixed = Qwen35::with_state_budget_and_kv(
        model,
        context,
        64 * 1024 * 1024,
        MatvecBackend::Scalar,
        KvCachePolicy::F16,
    )?;
    mixed.set_projection_executor(Some(Box::new(MockDecodeOnlyExecutor {
        decode_calls,
        released: Vec::new(),
    })));
    let prefill_scratch = mixed.tiled_prefill_scratch_bytes(8)?;
    mixed = mixed.with_tiled_prefill(8, prefill_scratch)?;
    let verify_scratch = mixed.tiled_verification_scratch_bytes()?;
    mixed.with_tiled_verification(verify_scratch)
}

fn assert_missing_host_weight_rejects_tiled_verification(
    cfg: &HybridConfig,
    decode_calls: Arc<std::sync::atomic::AtomicUsize>,
) {
    let mut released_model = pq2_mixed_hybrid_model(cfg);
    if let Some(tensor) = released_model.tensors.get_mut("blk.0.ffn_up.weight") {
        tensor.data = crate::model::TensorBytes::default();
    }
    assert!(build_decode_only_mixed_session(released_model, cfg.context, decode_calls).is_err());
}

#[test]
fn folded_verification_with_decode_only_executor_matches_cpu_and_preserves_rollback() -> Result<()>
{
    let cfg = HybridConfig::pinned_swift_aligned();
    let mut seq = configure_folded(
        pq2_mixed_hybrid_model(&cfg),
        &cfg,
        Arithmetic::Source,
        1,
        KvCachePolicy::F16,
        false,
    )?;

    let decode_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut mixed = build_decode_only_mixed_session(
        pq2_mixed_hybrid_model(&cfg),
        cfg.context,
        Arc::clone(&decode_calls),
    )?;

    // Host-weight release must stay fail-closed whenever tiled verification is enabled.
    assert!(mixed.release_executor_resident_host_weights().is_err());

    // Single-token decode routes through the decode-only executor.
    let s0 = seq.evaluate(0)?;
    let mut m0 = Vec::new();
    mixed.evaluate_into(0, &mut m0)?;
    assert_logits_bit_eq(&s0, &m0, "mixed init decode");
    let after_init_calls = decode_calls.load(std::sync::atomic::Ordering::Relaxed);
    assert!(after_init_calls > 0);

    // Folded verification rounds and carry-flush bypass the non-tile executor and match CPU bit-for-bit.
    let mut rows = Vec::new();
    mixed.evaluate_verification_block_folded(&[], &[1, 2, 3], &mut rows, &|| Ok(()))?;
    assert_eq!(
        decode_calls.load(std::sync::atomic::Ordering::Relaxed),
        after_init_calls
    );
    let s1 = seq.evaluate(1)?;
    let s2 = seq.evaluate(2)?;
    assert_logits_bit_eq(&s1, &rows[0], "mixed verify tok 1");
    assert_logits_bit_eq(&s2, &rows[1], "mixed verify tok 2");
    mixed.commit_folded_verification(2)?;

    // Mid-block cancellation must roll back cleanly while keeping decode executor intact.
    let checks = Cell::new(0_usize);
    let cancel = || {
        let next = checks.get() + 1;
        checks.set(next);
        if next >= 5 {
            return Err(invalid("mixed mid-block cancel"));
        }
        Ok(())
    };
    assert!(
        mixed
            .evaluate_verification_block_folded(&[1, 2], &[3, 4], &mut rows, &cancel)
            .is_err()
    );
    assert_eq!(rows, Vec::<Vec<f32>>::new());
    assert_full_state_bit_eq(&seq, &mixed)?;

    // Standard verification also falls back to CPU tiles cleanly when executor lacks tile support.
    let std_s3 = seq.evaluate(3)?;
    mixed.evaluate_verification_block(&[3], &mut rows, &|| Ok(()))?;
    assert_eq!(
        decode_calls.load(std::sync::atomic::Ordering::Relaxed),
        after_init_calls
    );
    assert_logits_bit_eq(&std_s3, &rows[0], "mixed std verify tok 3");
    assert_full_state_bit_eq(&seq, &mixed)?;

    // Subsequent single-token decode resumes using the decode-only executor and matches checkpoints.
    let s4 = seq.evaluate(4)?;
    let mut m4 = Vec::new();
    mixed.evaluate_into(4, &mut m4)?;
    assert!(decode_calls.load(std::sync::atomic::Ordering::Relaxed) > after_init_calls);
    assert_logits_bit_eq(&s4, &m4, "mixed resumed decode");
    assert_full_state_bit_eq(&seq, &mixed)?;
    assert_eq!(
        seq.checkpoint()?.layers.len(),
        mixed.checkpoint()?.layers.len()
    );

    assert_missing_host_weight_rejects_tiled_verification(&cfg, decode_calls);
    Ok(())
}

#[test]
fn bonsai25_f32_ssm_gates_and_stacked_adapters_preserve_prefill_decode_and_folded_parity()
-> Result<()> {
    let cfg = HybridConfig::pinned_swift_aligned();
    let f32_gates = hybrid_model(&cfg);
    for identity in [
        crate::model::swift::BONSAI_2_5_PQ2_SHA256,
        crate::model::swift::BONSAI_2_5_PTQ1_SHA256,
    ] {
        let mut seq_model = pq2_mixed_hybrid_model(&cfg);
        let mut tiled_model = pq2_mixed_hybrid_model(&cfg);
        let mut wy_model = pq2_mixed_hybrid_model(&cfg);
        for model in [&mut seq_model, &mut tiled_model, &mut wy_model] {
            model.identity = identity.to_owned();
            for (name, tensor) in &f32_gates.tensors {
                if name.ends_with(".ssm_alpha.weight") || name.ends_with(".ssm_beta.weight") {
                    model.tensors.insert(name.clone(), tensor.clone());
                }
            }
        }
        // Confirm F32 recurrent gate weights (`ssm_alpha.weight`, `ssm_beta.weight`) are accepted.
        assert_eq!(seq_model.tensors["blk.0.ssm_alpha.weight"].kind, 0);
        assert_eq!(seq_model.tensors["blk.0.ssm_beta.weight"].kind, 0);

        let mr_seq = build_folded_test_adapters(&seq_model, &cfg)?;
        let mr_tile = build_folded_test_adapters(&tiled_model, &cfg)?;
        let mr_wy = build_folded_test_adapters(&wy_model, &cfg)?;

        let mut seq = Qwen35::with_state_budget_and_adapters_scope(
            seq_model,
            cfg.context,
            64 * 1024 * 1024,
            MatvecBackend::Scalar,
            KvCachePolicy::F16,
            Some((mr_seq, 0.75)),
        )?;

        let mut tiled = Qwen35::with_state_budget_and_adapters_scope(
            tiled_model,
            cfg.context,
            64 * 1024 * 1024,
            MatvecBackend::Scalar,
            KvCachePolicy::F16,
            Some((mr_tile, 0.75)),
        )?;
        let prefill_scratch = tiled.tiled_prefill_scratch_bytes(8)?;
        tiled = tiled.with_tiled_prefill(8, prefill_scratch)?;
        let verify_scratch = tiled.tiled_verification_scratch_bytes()?;
        tiled = tiled.with_tiled_verification(verify_scratch)?;

        let mut wy = Qwen35::with_state_budget_and_adapters_scope(
            wy_model,
            cfg.context,
            64 * 1024 * 1024,
            MatvecBackend::Scalar,
            KvCachePolicy::F16,
            Some((mr_wy, 0.75)),
        )?;
        let wy_prefill = wy.tiled_prefill_scratch_bytes(8)?;
        wy = wy
            .with_tiled_prefill(8, wy_prefill)?
            .with_chunkwise_gdn_prefill()?;

        // Prompt prefill across multiple recurrent/full tokens must match sequential decode.
        for token in [0, 1, 2] {
            let _ = seq.evaluate(token)?;
        }
        let s_prompt = seq.evaluate(3)?;
        tiled.ingest_block(&[0, 1, 2], &|| Ok(()))?;
        let t_prompt = tiled.evaluate(3)?;
        wy.ingest_block(&[0, 1, 2], &|| Ok(()))?;
        let w_prompt = wy.evaluate(3)?;
        assert_logits_bit_eq(&s_prompt, &t_prompt, "bonsai25 tiled prefill");
        assert_eq!(s_prompt.len(), w_prompt.len());
        for (idx, (a, b)) in s_prompt.iter().zip(&w_prompt).enumerate() {
            assert!(
                (a - b).abs() <= 1e-4,
                "bonsai25 wy-f32 prefill logit[{idx}] mismatch: {a} != {b}"
            );
        }

        // Folded verification with stacked adapters matches sequential decode bit-for-bit.
        let mut rows = Vec::new();
        tiled.evaluate_verification_block_folded(&[], &[4, 5], &mut rows, &|| Ok(()))?;
        let s4 = seq.evaluate(4)?;
        let s5 = seq.evaluate(5)?;
        assert_logits_bit_eq(&s4, &rows[0], "bonsai25 folded verify tok 4");
        assert_logits_bit_eq(&s5, &rows[1], "bonsai25 folded verify tok 5");
        tiled.commit_folded_verification(2)?;
        tiled.flush_folded_verification(&|| Ok(()))?;
        assert_full_state_bit_eq(&seq, &tiled)?;
    }
    Ok(())
}

#[test]
fn commit_folded_verification_prevalidation_and_mutation_atomicity() -> Result<()> {
    let cfg = HybridConfig::small_source();
    let m1 = hybrid_model_with_identity(&cfg, Arithmetic::Source);
    let mut folded = configure_folded(m1, &cfg, Arithmetic::Source, 1, KvCachePolicy::F16, false)?;

    folded.evaluate(0)?;
    let mut rows = Vec::new();
    folded.evaluate_verification_block_folded(&[], &[1, 2], &mut rows, &|| Ok(()))?;
    assert_eq!(folded.position, 3);

    // Pre-validation failure (missing recurrent tape) must not mutate decoder state or position.
    let saved_tape = folded
        .tiled_prefill
        .as_mut()
        .and_then(|tile| tile.folded.as_mut())
        .and_then(|state| state.layers.pop())
        .ok_or_else(|| invalid("missing folded layer tape"))?;
    let Err(pre_err) = folded.commit_folded_verification(1) else {
        return Err(invalid("expected recurrent layer count mismatch"));
    };
    assert!(pre_err.to_string().contains("layer count mismatch"));
    assert!(folded.state_consistent);
    assert_eq!(folded.position, 3);

    // Restore tape; pre-validation failure on capacity must also leave state consistent.
    if let Some(state) = folded
        .tiled_prefill
        .as_mut()
        .and_then(|tile| tile.folded.as_mut())
    {
        state.layers.push(saved_tape);
        state.carried_tokens = Vec::new();
    }
    let Err(cap_err) = folded.commit_folded_verification(1) else {
        return Err(invalid("expected carried token capacity error"));
    };
    assert!(cap_err.to_string().contains("capacity"));
    assert!(folded.state_consistent);
    assert_eq!(folded.position, 3);

    // Restore capacity, then corrupt Full KV cache to trigger failure inside the mutation loop.
    if let Some(state) = folded
        .tiled_prefill
        .as_mut()
        .and_then(|tile| tile.folded.as_mut())
    {
        state.carried_tokens.try_reserve_exact(cfg.context)?;
    }
    for layer_state in &mut folded.state {
        if let LayerState::Full { keys, .. } = layer_state {
            keys.width = 0;
            break;
        }
    }
    let Err(mut_err) = folded.commit_folded_verification(1) else {
        return Err(invalid("expected KV truncation failure during commit"));
    };
    assert!(mut_err.to_string().contains("geometry"));
    assert!(!folded.state_consistent);
    assert!(!folded.has_pending_folded_verification());
    let pending_empty = folded
        .tiled_prefill
        .as_ref()
        .and_then(|tile| tile.folded.as_ref())
        .is_some_and(|state| state.pending_tokens.is_empty() && state.carried_tokens.is_empty());
    assert!(pending_empty);

    let Err(inconsistent_err) = folded.commit_folded_verification(0) else {
        return Err(invalid("expected inconsistent state rejection"));
    };
    assert!(inconsistent_err.to_string().contains("inconsistent"));
    Ok(())
}
