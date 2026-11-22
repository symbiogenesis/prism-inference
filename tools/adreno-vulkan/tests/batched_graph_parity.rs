//! Parity, TDR chunk-policy, SPIR-V validation, and state-checkpoint tests for
//! [`VulkanBatchedGraphExecutor`].
#![forbid(unsafe_code)]

use std::{collections::BTreeMap, sync::Arc};

use bonsai_adreno_vulkan::{
    CommandBatchRecorder, TdrChunkPolicy, VulkanBatchedConfig, VulkanBatchedGraphExecutor,
    VulkanKvPrecision, spirv,
};
use bonsai_gguf::MetadataValue;
use bonsai_prism_runtime::{
    Decoder as _,
    graph::Qwen35,
    model::{Model, Tensor},
};
use bonsai_ternary::{float_to_half, pack_ptq};

type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;
type ModelResult = Result<Model, Box<dyn std::error::Error + Send + Sync>>;

fn f16_scale(value: f32) -> [u8; 2] {
    float_to_half(value).to_le_bytes()
}

fn f32_tensor(name: &str, dims: &[usize], values: &[f32]) -> Tensor {
    let mut data = Vec::with_capacity(values.len() * 4);
    for &v in values {
        data.extend_from_slice(&v.to_le_bytes());
    }
    Tensor {
        name: name.to_owned(),
        dimensions: dims.to_vec(),
        kind: 0,
        data: Arc::<[u8]>::from(data).into(),
    }
}

fn fill_pattern(len: usize, freq: f32, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let s = f32::from(u8::try_from((i + seed * 7) % 29).unwrap_or(0)) - 14.0;
            s * freq
        })
        .collect()
}

fn build_pq2_tensor(name: &str, cols: usize, rows: usize, seed: usize) -> Tensor {
    let blocks_per_row = cols / 128;
    let mut data = Vec::with_capacity(rows * blocks_per_row * 34);
    for r in 0..rows {
        for b in 0..blocks_per_row {
            let scale_val =
                0.015_625_f32 * f32::from(u8::try_from((r + b + seed) % 5 + 1).unwrap_or(1));
            data.extend_from_slice(&f16_scale(scale_val));
            for j in 0..32 {
                let word = match (r + b + j + seed) % 4 {
                    0 => 0x55_u8,
                    1 => 0x9E_u8,
                    2 => 0xE1_u8,
                    _ => 0x1B_u8,
                };
                data.push(word);
            }
        }
    }
    Tensor {
        name: name.to_owned(),
        dimensions: vec![cols, rows],
        kind: 142,
        data: Arc::<[u8]>::from(data).into(),
    }
}

fn build_ptq1_tensor(
    name: &str,
    cols: usize,
    rows: usize,
    seed: usize,
) -> Result<Tensor, Box<dyn std::error::Error + Send + Sync>> {
    let blocks_per_row = cols / 128;
    let mut data = Vec::with_capacity(rows * blocks_per_row * 28);
    for r in 0..rows {
        for b in 0..blocks_per_row {
            let mut pq2_block = [0_u8; 34];
            let scale_val =
                0.03125_f32 * f32::from(u8::try_from((r + b + seed) % 4 + 1).unwrap_or(1));
            pq2_block[..2].copy_from_slice(&f16_scale(scale_val));
            for i in 0..128 {
                let code: u8 = match (r + b * 3 + i + seed) % 5 {
                    0 => 2, // +1
                    1 => 0, // -1
                    _ => 1, //  0
                };
                let byte_idx = 2 + i / 4;
                let shift = (i % 4) * 2;
                pq2_block[byte_idx] |= code << shift;
            }
            let packed = pack_ptq(&pq2_block).map_err(|err| format!("pack_ptq failed: {err}"))?;
            data.extend_from_slice(&packed);
        }
    }
    Ok(Tensor {
        name: name.to_owned(),
        dimensions: vec![cols, rows],
        kind: 143,
        data: Arc::<[u8]>::from(data).into(),
    })
}

#[expect(
    clippy::too_many_lines,
    reason = "Constructs an explicit synthetic hybrid recurrent-GDN + full-attention Qwen35 test model."
)]
fn synthetic_hybrid_qwen35_model(identity: &str) -> ModelResult {
    let embd = 256_usize;
    let ffn = 256_usize;
    let vocab = 16_usize;
    let heads = 4_usize;
    let kv_heads = 2_usize;
    let head = 64_usize;
    let rope = 64_usize;
    let ssm_head = 64_usize;
    let ssm_keys = 2_usize;
    let ssm_values = 4_usize;
    let ssm_inner = ssm_head * ssm_values;
    let conv_k = 4_usize;
    let conv_ch = 2 * ssm_head * ssm_keys + ssm_inner; // 512

    let mut metadata = BTreeMap::new();
    metadata.insert(
        "general.architecture".to_owned(),
        MetadataValue::from("qwen35"),
    );
    metadata.insert(
        "qwen35.context_length".to_owned(),
        MetadataValue::from(32_u32),
    );
    metadata.insert("qwen35.block_count".to_owned(), MetadataValue::from(3_u32));
    metadata.insert(
        "qwen35.embedding_length".to_owned(),
        MetadataValue::from(u32::try_from(embd)?),
    );
    metadata.insert(
        "qwen35.feed_forward_length".to_owned(),
        MetadataValue::from(u32::try_from(ffn)?),
    );
    metadata.insert(
        "qwen35.attention.head_count".to_owned(),
        MetadataValue::from(u32::try_from(heads)?),
    );
    metadata.insert(
        "qwen35.attention.head_count_kv".to_owned(),
        MetadataValue::from(u32::try_from(kv_heads)?),
    );
    metadata.insert(
        "qwen35.attention.key_length".to_owned(),
        MetadataValue::from(u32::try_from(head)?),
    );
    metadata.insert(
        "qwen35.attention.value_length".to_owned(),
        MetadataValue::from(u32::try_from(head)?),
    );
    metadata.insert(
        "qwen35.attention.layer_norm_rms_epsilon".to_owned(),
        MetadataValue::from(1e-6_f32),
    );
    metadata.insert(
        "qwen35.rope.dimension_count".to_owned(),
        MetadataValue::from(u32::try_from(rope)?),
    );
    metadata.insert(
        "qwen35.rope.dimension_sections".to_owned(),
        MetadataValue::from(vec![16_u32, 8, 8, 0]),
    );
    metadata.insert(
        "qwen35.rope.freq_base".to_owned(),
        MetadataValue::from(10_000.0_f32),
    );
    metadata.insert(
        "qwen35.ssm.state_size".to_owned(),
        MetadataValue::from(u32::try_from(ssm_head)?),
    );
    metadata.insert(
        "qwen35.ssm.group_count".to_owned(),
        MetadataValue::from(u32::try_from(ssm_keys)?),
    );
    metadata.insert(
        "qwen35.ssm.time_step_rank".to_owned(),
        MetadataValue::from(u32::try_from(ssm_values)?),
    );
    metadata.insert(
        "qwen35.ssm.inner_size".to_owned(),
        MetadataValue::from(u32::try_from(ssm_inner)?),
    );
    metadata.insert(
        "qwen35.ssm.conv_kernel".to_owned(),
        MetadataValue::from(u32::try_from(conv_k)?),
    );
    metadata.insert(
        "qwen35.full_attention_interval".to_owned(),
        MetadataValue::from(3_u32),
    );

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "token_embd.weight".to_owned(),
        f32_tensor(
            "token_embd.weight",
            &[embd, vocab],
            &fill_pattern(embd * vocab, 0.03, 1),
        ),
    );
    tensors.insert(
        "output.weight".to_owned(),
        f32_tensor(
            "output.weight",
            &[embd, vocab],
            &fill_pattern(embd * vocab, 0.02, 2),
        ),
    );
    tensors.insert(
        "output_norm.weight".to_owned(),
        f32_tensor("output_norm.weight", &[embd], &vec![1.0_f32; embd]),
    );

    // Layers 0 and 1: Recurrent GDN.
    for layer_idx in [0_usize, 1] {
        let prefix = format!("blk.{layer_idx}.");
        tensors.insert(
            format!("{prefix}attn_norm.weight"),
            f32_tensor(
                &format!("{prefix}attn_norm.weight"),
                &[embd],
                &vec![1.0; embd],
            ),
        );
        tensors.insert(
            format!("{prefix}post_attention_norm.weight"),
            f32_tensor(
                &format!("{prefix}post_attention_norm.weight"),
                &[embd],
                &vec![1.0; embd],
            ),
        );
        tensors.insert(
            format!("{prefix}attn_qkv.weight"),
            build_pq2_tensor(
                &format!("{prefix}attn_qkv.weight"),
                embd,
                conv_ch,
                layer_idx + 1,
            ),
        );
        tensors.insert(
            format!("{prefix}attn_gate.weight"),
            build_pq2_tensor(
                &format!("{prefix}attn_gate.weight"),
                embd,
                ssm_inner,
                layer_idx + 2,
            ),
        );
        tensors.insert(
            format!("{prefix}ssm_beta.weight"),
            f32_tensor(
                &format!("{prefix}ssm_beta.weight"),
                &[embd, ssm_values],
                &fill_pattern(embd * ssm_values, 0.02, layer_idx + 3),
            ),
        );
        tensors.insert(
            format!("{prefix}ssm_alpha.weight"),
            f32_tensor(
                &format!("{prefix}ssm_alpha.weight"),
                &[embd, ssm_values],
                &fill_pattern(embd * ssm_values, 0.02, layer_idx + 4),
            ),
        );
        tensors.insert(
            format!("{prefix}ssm_conv1d.weight"),
            f32_tensor(
                &format!("{prefix}ssm_conv1d.weight"),
                &[conv_k, conv_ch],
                &fill_pattern(conv_k * conv_ch, 0.05, layer_idx + 5),
            ),
        );
        tensors.insert(
            format!("{prefix}ssm_dt.bias"),
            f32_tensor(
                &format!("{prefix}ssm_dt.bias"),
                &[ssm_values],
                &vec![0.1_f32; ssm_values],
            ),
        );
        tensors.insert(
            format!("{prefix}ssm_a"),
            f32_tensor(
                &format!("{prefix}ssm_a"),
                &[ssm_values],
                &vec![-0.25_f32; ssm_values],
            ),
        );
        tensors.insert(
            format!("{prefix}ssm_norm.weight"),
            f32_tensor(
                &format!("{prefix}ssm_norm.weight"),
                &[ssm_head],
                &vec![1.0_f32; ssm_head],
            ),
        );
        tensors.insert(
            format!("{prefix}ssm_out.weight"),
            build_pq2_tensor(
                &format!("{prefix}ssm_out.weight"),
                ssm_inner,
                embd,
                layer_idx + 6,
            ),
        );
        tensors.insert(
            format!("{prefix}ffn_gate.weight"),
            build_pq2_tensor(
                &format!("{prefix}ffn_gate.weight"),
                embd,
                ffn,
                layer_idx + 7,
            ),
        );
        tensors.insert(
            format!("{prefix}ffn_up.weight"),
            build_pq2_tensor(&format!("{prefix}ffn_up.weight"), embd, ffn, layer_idx + 8),
        );
        tensors.insert(
            format!("{prefix}ffn_down.weight"),
            build_pq2_tensor(
                &format!("{prefix}ffn_down.weight"),
                ffn,
                embd,
                layer_idx + 9,
            ),
        );
    }

    // Layer 2: Full Attention with interleaved Q/Gate, GQA, RoPE, PTQ1 QKV, and PQ2 FFN.
    tensors.insert(
        "blk.2.attn_norm.weight".to_owned(),
        f32_tensor("blk.2.attn_norm.weight", &[embd], &vec![1.0; embd]),
    );
    tensors.insert(
        "blk.2.post_attention_norm.weight".to_owned(),
        f32_tensor(
            "blk.2.post_attention_norm.weight",
            &[embd],
            &vec![1.0; embd],
        ),
    );
    tensors.insert(
        "blk.2.attn_q_norm.weight".to_owned(),
        f32_tensor("blk.2.attn_q_norm.weight", &[head], &vec![1.0; head]),
    );
    tensors.insert(
        "blk.2.attn_k_norm.weight".to_owned(),
        f32_tensor("blk.2.attn_k_norm.weight", &[head], &vec![1.0; head]),
    );
    tensors.insert(
        "blk.2.attn_q.weight".to_owned(),
        build_ptq1_tensor("blk.2.attn_q.weight", embd, heads * head * 2, 11)?,
    );
    tensors.insert(
        "blk.2.attn_k.weight".to_owned(),
        build_ptq1_tensor("blk.2.attn_k.weight", embd, kv_heads * head, 12)?,
    );
    tensors.insert(
        "blk.2.attn_v.weight".to_owned(),
        build_ptq1_tensor("blk.2.attn_v.weight", embd, kv_heads * head, 13)?,
    );
    tensors.insert(
        "blk.2.attn_output.weight".to_owned(),
        build_pq2_tensor("blk.2.attn_output.weight", heads * head, embd, 14),
    );
    tensors.insert(
        "blk.2.ffn_gate.weight".to_owned(),
        build_pq2_tensor("blk.2.ffn_gate.weight", embd, ffn, 15),
    );
    tensors.insert(
        "blk.2.ffn_up.weight".to_owned(),
        build_pq2_tensor("blk.2.ffn_up.weight", embd, ffn, 16),
    );
    tensors.insert(
        "blk.2.ffn_down.weight".to_owned(),
        build_pq2_tensor("blk.2.ffn_down.weight", ffn, embd, 17),
    );

    Ok(Model {
        metadata,
        tensors,
        identity: identity.to_owned(),
    })
}

fn assert_close(actual: &[f32], expected: &[f32], max_tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    let mut max_diff = 0.0_f32;
    for (&a, &e) in actual.iter().zip(expected) {
        let diff = (a - e).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    assert!(
        max_diff <= max_tol,
        "{label}: max abs diff {max_diff} exceeds tolerance {max_tol}"
    );
}

#[test]
fn spirv_layer_and_helper_modules_pass_structural_validation() {
    let portable_modules = [
        ("rms_norm_f32", spirv::rms_norm_f32()),
        ("add_rms_norm_f32", spirv::add_rms_norm_f32()),
        ("residual_add_f32", spirv::residual_add_f32()),
        ("rope_f32", spirv::rope_f32()),
        ("mul_sigmoid_f32", spirv::mul_sigmoid_f32()),
        ("conv1d_silu_f32", spirv::conv1d_silu_f32()),
        ("l2_norm_f32", spirv::l2_norm_f32()),
        ("gdn_step_f32", spirv::gdn_step_f32()),
        ("gdn_gates_f32", spirv::gdn_gates_f32()),
        ("deinterleave_q_gate_f32", spirv::deinterleave_q_gate_f32()),
        ("round_f16_f32", spirv::round_f16_f32()),
        ("kv_append_f32", spirv::kv_append_f32()),
        ("kv_append_f16", spirv::kv_append_f16()),
        ("kv_unpack_f16_to_f32", spirv::kv_unpack_f16_to_f32()),
        ("quant_q8_0_src", spirv::quantize_f32_to_q8_0(false, false)),
        ("quant_q8_0_pinned", spirv::quantize_f32_to_q8_0(true, true)),
        ("f32_matvec", spirv::f32_matvec()),
        ("mul_f32_elemwise", spirv::mul_f32_elemwise()),
        ("permute_ssm_heads_f32", spirv::permute_ssm_heads_f32()),
        ("copy_slice_f32", spirv::copy_slice_f32()),
        ("attn_fused_f16prob", spirv::attn_fused_f16prob()),
        ("attn_qk_scores", spirv::attn_qk_scores()),
        ("attn_softmax_f16", spirv::attn_softmax_f16()),
        ("attn_pv_accum", spirv::attn_pv_accum()),
    ];
    for (name, words) in portable_modules {
        assert_eq!(
            spirv::validate_spirv_module(&words, Some(false)),
            Ok(()),
            "portable SPIR-V validation failed for {name}"
        );
    }
    assert_eq!(
        spirv::validate_spirv_module(
            &spirv::pq2_q8k_gemv_coalesced(true, true, false),
            Some(true)
        ),
        Ok(())
    );
}

#[test]
fn command_batch_recorder_starts_empty_and_rejects_zero_chunk_limit() -> TestResult {
    let mut rec = CommandBatchRecorder::new();
    assert!(rec.is_empty());
    assert_eq!(rec.len(), 0);
    assert_eq!(rec.total_descriptors(), 0);
    assert_eq!(rec.intra_batch_barrier_count(), 0);
    assert!(rec.split_chunks(0).is_err());
    assert_eq!(rec.split_chunks(8)?.len(), 0);
    assert_eq!(rec.identity_sha256()?, rec.identity_sha256()?);
    rec.clear();
    assert!(rec.is_empty());
    Ok(())
}

#[test]
fn tdr_chunk_policy_enforces_bounds_and_cuts_64_layer_submits() -> TestResult {
    let default = TdrChunkPolicy::default();
    assert_eq!(default.layers_per_submit, 8);
    assert!(default.expected_submits_for_layers(64)? <= 8);
    assert_eq!(
        default.layer_chunk_ranges(64)?,
        vec![0..8, 8..16, 16..24, 24..32, 32..40, 40..48, 48..56, 56..64]
    );
    assert_eq!(
        TdrChunkPolicy::new(1, 512)?.expected_submits_for_layers(64)?,
        64
    );

    assert!(TdrChunkPolicy::new(0, 2048).is_err());
    assert!(TdrChunkPolicy::new(65, 2048).is_err());
    assert!(TdrChunkPolicy::new(8, 0).is_err());
    assert!(TdrChunkPolicy::new(8, 4097).is_err());
    Ok(())
}

#[test]
fn synthetic_hybrid_model_builds_and_config_defaults_are_well_formed() -> TestResult {
    let model = synthetic_hybrid_qwen35_model("offline-hybrid-parity-check")?;
    let mut cpu = Qwen35::new(model, 16)?;
    let l0 = cpu.evaluate(0)?;
    assert_eq!(l0.len(), 16);
    assert!(l0.iter().all(|v| v.is_finite()));

    let cfg = VulkanBatchedConfig::default();
    assert_eq!(cfg.kv_precision, VulkanKvPrecision::Fp16);
    assert_eq!(cfg.tdr_policy.layers_per_submit, 8);
    assert_eq!(cfg.kv_precision.name(), "fp16");
    assert_eq!(VulkanKvPrecision::Fp32.name(), "fp32");
    Ok(())
}

#[test]
#[ignore = "Requires physical Qualcomm Adreno Vulkan Windows ARM64 hardware"]
fn live_batched_hybrid_graph_matches_cpu_qwen35_and_controls_submit_count() -> TestResult {
    let cpu_model = synthetic_hybrid_qwen35_model("live-batched-hybrid-parity")?;
    let gpu_model_batched = synthetic_hybrid_qwen35_model("live-batched-hybrid-parity")?;
    let gpu_model_single = synthetic_hybrid_qwen35_model("live-batched-hybrid-parity")?;
    let gpu_model_fp32kv = synthetic_hybrid_qwen35_model("live-batched-hybrid-parity")?;

    let mut cpu = Qwen35::new(cpu_model, 16)?;
    let mut batched =
        VulkanBatchedGraphExecutor::open(&gpu_model_batched, VulkanBatchedConfig::default())?;
    let id1 = batched.identity_sha256().to_owned();

    // Token 0: evaluate across all 3 layers (2 recurrent GDN + 1 full attention) in 1 GPU submit.
    let cpu_t0 = cpu.evaluate(0)?;
    let gpu_t0 = batched.evaluate_token(0)?;
    assert_close(&gpu_t0, &cpu_t0, 1e-4, "batched hybrid token 0 (fp16 kv)");
    assert_eq!(batched.submits_completed(), 1);
    assert_eq!(batched.position(), 1);

    // Checkpoint after token 0, advance through ingest + token 2, verify CPU parity, then restore.
    let cp0 = batched.checkpoint()?;
    assert_eq!(cp0.position(), 1);
    assert_eq!(cp0.layer_count(), 3);

    cpu.ingest(1)?;
    batched.ingest_token(1)?;
    assert_eq!(batched.submits_completed(), 2);

    let cpu_t2 = cpu.evaluate(2)?;
    let gpu_t2 = batched.evaluate_token(2)?;
    assert_close(
        &gpu_t2,
        &cpu_t2,
        1e-4,
        "batched hybrid token 2 after ingest",
    );
    assert_eq!(batched.submits_completed(), 3);

    // Restore back to checkpoint after token 0 and verify deterministic replay at position 1.
    batched.restore(&cp0)?;
    assert_eq!(batched.position(), 1);
    let gpu_t1 = batched.evaluate_token(1)?;
    let mut cpu_replay = Qwen35::new(
        synthetic_hybrid_qwen35_model("live-batched-hybrid-parity")?,
        16,
    )?;
    let _ = cpu_replay.evaluate(0)?;
    let cpu_t1 = cpu_replay.evaluate(1)?;
    assert_close(&gpu_t1, &cpu_t1, 1e-4, "batched hybrid restore token 1");

    // Reset back to position 0 and confirm identical logits to initial token 0.
    batched.reset()?;
    assert_eq!(batched.position(), 0);
    let gpu_t0_reset = batched.evaluate_token(0)?;
    assert_close(&gpu_t0_reset, &gpu_t0, 1e-5, "batched hybrid reset replay");
    assert!(!batched.is_faulted());

    // Verify `layers_per_submit = 1` splits the 3-layer token into 3 submits while producing identical logits.
    let cfg_1layer = VulkanBatchedConfig {
        tdr_policy: TdrChunkPolicy::new(1, 1024)?,
        ..VulkanBatchedConfig::default()
    };
    let mut chunked_1 = VulkanBatchedGraphExecutor::open(&gpu_model_single, cfg_1layer)?;
    let chunked_t0 = chunked_1.evaluate_token(0)?;
    assert_close(&chunked_t0, &gpu_t0, 1e-5, "1-layer chunking vs 8-layer");
    assert_eq!(chunked_1.submits_completed(), 3);
    assert_ne!(chunked_1.identity_sha256(), id1);

    // Verify Fp32 KV storage precision path also matches CPU.
    let cfg_fp32 = VulkanBatchedConfig {
        kv_precision: VulkanKvPrecision::Fp32,
        ..VulkanBatchedConfig::default()
    };
    let mut fp32_kv = VulkanBatchedGraphExecutor::open(&gpu_model_fp32kv, cfg_fp32)?;
    let fp32_t0 = fp32_kv.evaluate_token(0)?;
    assert_close(&fp32_t0, &cpu_t0, 1e-4, "fp32 kv token 0");
    assert_eq!(fp32_kv.submits_completed(), 1);

    Ok(())
}
