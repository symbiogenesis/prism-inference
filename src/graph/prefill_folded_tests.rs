//! Bit-exact folded speculative verification parity, adapters, carry flush, and cancellation tests.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::Arc;

use bonsai_gguf::MetadataValue;

use super::*;
use crate::Decoder;
use crate::graph::KvCachePolicy;
use crate::model::Model;
use crate::model::lora::{MultiRankAdapter, MultiRankPolicy, RankOneAdapter};

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
        model.identity = crate::model::swift::MODEL_SHA256.to_owned();
    }
    model
}

fn build_folded_test_adapters(
    model: &Model,
    cfg: &HybridConfig,
) -> Result<(RankOneAdapter, MultiRankAdapter)> {
    let mut rank_one_tensors = BTreeMap::new();
    let target_r1 = "blk.0.ffn_down.weight";
    rank_one_tensors.insert(
        format!("{target_r1}.lora_a"),
        f32_tensor(
            &format!("{target_r1}.lora_a"),
            &[cfg.ffn, 1],
            &pattern(cfg.ffn, 21, 0.125),
        ),
    );
    rank_one_tensors.insert(
        format!("{target_r1}.lora_b"),
        f32_tensor(
            &format!("{target_r1}.lora_b"),
            &[1, cfg.embedding],
            &pattern(cfg.embedding, 22, 0.125),
        ),
    );
    let r1_model = Model {
        metadata: BTreeMap::from([
            (
                "general.architecture".to_owned(),
                MetadataValue::from("qwen35"),
            ),
            ("general.type".to_owned(), MetadataValue::from("adapter")),
            ("adapter.type".to_owned(), MetadataValue::from("lora")),
            (
                "adapter.lora_alpha".to_owned(),
                MetadataValue::from(1.0_f32),
            ),
        ]),
        tensors: rank_one_tensors,
        identity: "r1-folded-test".to_owned(),
    };
    let r1 = RankOneAdapter::from_model(r1_model, model)?;

    let mut mr_tensors = BTreeMap::new();
    let val_w = cfg.ssm_values * cfg.ssm_head;
    let conv_w = 2 * cfg.ssm_keys * cfg.ssm_head + val_w;
    let q_gate_w = cfg.heads * cfg.head * 2;
    for (target, in_f, out_f, seed) in [
        ("blk.0.attn_qkv.weight", cfg.embedding, conv_w, 31),
        ("blk.1.attn_q.weight", cfg.embedding, q_gate_w, 32),
    ] {
        let rank = 2;
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
    let mr = MultiRankAdapter::from_model(model, &mr_model, MultiRankPolicy::AnyLinear)?;
    Ok((r1, mr))
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
        let (r1, mr) = build_folded_test_adapters(&model, cfg)?;
        Qwen35::with_adapters_and_kv(
            model,
            cfg.context,
            64 * 1024 * 1024,
            MatvecBackend::Scalar,
            Some((r1, 0.75)),
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
