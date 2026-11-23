//! Parity, TDR chunk-policy, SPIR-V validation, and state-checkpoint tests for
//! [`VulkanBatchedGraphExecutor`].
#![forbid(unsafe_code)]

use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use bonsai_adreno_vulkan::{
    AttentionStrategy, CommandBatchRecorder, TdrChunkPolicy, VulkanBatchedAdapters,
    VulkanBatchedConfig, VulkanBatchedGraphExecutor, VulkanKvPrecision, VulkanMultiRankAdapterSpec,
    VulkanRankOneAdapterSpec, spirv,
};
use bonsai_gguf::MetadataValue;
use bonsai_prism_runtime::{
    Decoder,
    graph::{KvCachePolicy, Qwen35},
    kernels::MatvecBackend,
    model::{
        Model, Tensor,
        lora::{MultiRankAdapter, MultiRankPolicy, RankOneAdapter},
    },
    runtime::{Limits, Runtime},
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

fn f16_tensor(name: &str, dims: &[usize], values: &[f32]) -> Tensor {
    let mut data = Vec::with_capacity(values.len() * 2);
    for &v in values {
        data.extend_from_slice(&float_to_half(v).to_le_bytes());
    }
    Tensor {
        name: name.to_owned(),
        dimensions: dims.to_vec(),
        kind: 1,
        data: Arc::<[u8]>::from(data).into(),
    }
}

fn adapter_tensor(name: &str, dims: &[usize], kind: u32, values: &[f32]) -> Result<Tensor, String> {
    match kind {
        0 => Ok(f32_tensor(name, dims, values)),
        1 => Ok(f16_tensor(name, dims, values)),
        other => Err(format!("unsupported adapter test tensor kind {other}")),
    }
}

fn base_adapter_metadata(alpha: f32) -> BTreeMap<String, MetadataValue> {
    let mut metadata = BTreeMap::new();
    metadata.insert(
        "general.architecture".to_owned(),
        MetadataValue::from("qwen35"),
    );
    metadata.insert("general.type".to_owned(), MetadataValue::from("adapter"));
    metadata.insert("adapter.type".to_owned(), MetadataValue::from("lora"));
    metadata.insert("adapter.lora_alpha".to_owned(), MetadataValue::from(alpha));
    metadata
}

fn synthetic_rank_one_adapter(
    base: &Model,
    identity: &str,
    alpha: f32,
    include_embed: bool,
    kind: u32,
) -> ModelResult {
    let mut tensors = BTreeMap::new();
    let targets = [
        "token_embd.weight",
        "blk.0.ssm_out.weight",
        "blk.2.attn_output.weight",
        "blk.2.ffn_down.weight",
    ];
    for (idx, target) in targets.into_iter().enumerate() {
        if target == "token_embd.weight" && !include_embed {
            continue;
        }
        let base_tensor = base
            .tensors
            .get(target)
            .ok_or_else(|| format!("missing base tensor {target}"))?;
        let [input_dim, output_dim] = base_tensor.dimensions.as_slice() else {
            return Err(format!("base tensor {target} is not 2D").into());
        };
        let (a_dims, b_dims) = if target == "token_embd.weight" {
            ([1_usize, *output_dim], [1_usize, *input_dim])
        } else {
            ([*input_dim, 1_usize], [1_usize, *output_dim])
        };
        let a_len = a_dims[0] * a_dims[1];
        let b_len = b_dims[0] * b_dims[1];
        let a_vals = fill_pattern(a_len, 0.013, 100 + idx * 2);
        let b_vals = fill_pattern(b_len, 0.011, 101 + idx * 2);
        let a_name = format!("{target}.lora_a");
        let b_name = format!("{target}.lora_b");
        tensors.insert(
            a_name.clone(),
            adapter_tensor(&a_name, &a_dims, kind, &a_vals)?,
        );
        tensors.insert(
            b_name.clone(),
            adapter_tensor(&b_name, &b_dims, kind, &b_vals)?,
        );
    }
    Ok(Model {
        metadata: base_adapter_metadata(alpha),
        tensors,
        identity: identity.to_owned(),
    })
}

fn synthetic_multi_rank_adapter(
    base: &Model,
    identity: &str,
    alpha: f32,
    rank: usize,
    kind: u32,
) -> ModelResult {
    let mut metadata = base_adapter_metadata(alpha);
    metadata.insert("adapter.lora.alpha".to_owned(), MetadataValue::from(alpha));
    let mut tensors = BTreeMap::new();
    let targets = [
        "blk.2.attn_q.weight",
        "blk.2.ffn_gate.weight",
        "blk.2.ffn_down.weight",
    ];
    for (idx, target) in targets.into_iter().enumerate() {
        let base_tensor = base
            .tensors
            .get(target)
            .ok_or_else(|| format!("missing base tensor {target}"))?;
        let [input_dim, output_dim] = base_tensor.dimensions.as_slice() else {
            return Err(format!("base tensor {target} is not 2D").into());
        };
        let a_dims = [*input_dim, rank];
        let b_dims = [rank, *output_dim];
        let a_vals = fill_pattern(input_dim * rank, 0.009, 200 + idx * 2);
        let b_vals = fill_pattern(rank * output_dim, 0.008, 201 + idx * 2);
        let a_name = format!("{target}.lora_a");
        let b_name = format!("{target}.lora_b");
        tensors.insert(
            a_name.clone(),
            adapter_tensor(&a_name, &a_dims, kind, &a_vals)?,
        );
        tensors.insert(
            b_name.clone(),
            adapter_tensor(&b_name, &b_dims, kind, &b_vals)?,
        );
    }
    Ok(Model {
        metadata,
        tensors,
        identity: identity.to_owned(),
    })
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

fn synthetic_hybrid_qwen35_model(identity: &str) -> ModelResult {
    synthetic_hybrid_qwen35_model_with_layout(identity, 256, 256, 32)
}

#[expect(
    clippy::too_many_lines,
    reason = "Constructs an explicit synthetic hybrid recurrent-GDN + full-attention Qwen35 test model with configurable dims and context."
)]
fn synthetic_hybrid_qwen35_model_with_layout(
    identity: &str,
    embd: usize,
    ffn: usize,
    context_length: u32,
) -> ModelResult {
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
        MetadataValue::from(context_length),
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

fn assert_close_rel(actual: &[f32], expected: &[f32], abs_tol: f32, rel_tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (idx, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let diff = (a - e).abs();
        let allowed = rel_tol.mul_add(a.abs().max(e.abs()), abs_tol);
        assert!(
            diff <= allowed,
            "{label}: idx {idx} abs diff {diff} exceeds allowed {allowed} (actual={a}, expected={e})"
        );
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "Validates all single-token, batched, 4-token GEMM, and attention SPIR-V modules."
)]
#[test]
fn spirv_layer_and_helper_modules_pass_structural_validation() {
    let portable_modules = [
        ("rms_norm_f32", spirv::rms_norm_f32()),
        ("rms_norm_batched_f32", spirv::rms_norm_batched_f32()),
        ("add_rms_norm_f32", spirv::add_rms_norm_f32()),
        (
            "add_rms_norm_batched_f32",
            spirv::add_rms_norm_batched_f32(),
        ),
        ("residual_add_f32", spirv::residual_add_f32()),
        (
            "residual_add_batched_f32",
            spirv::residual_add_batched_f32(),
        ),
        ("rope_f32", spirv::rope_f32()),
        ("rope_batched_f32", spirv::rope_batched_f32()),
        ("mul_sigmoid_f32", spirv::mul_sigmoid_f32()),
        ("conv1d_silu_f32", spirv::conv1d_silu_f32()),
        ("conv1d_silu_offset_f32", spirv::conv1d_silu_offset_f32()),
        ("l2_norm_f32", spirv::l2_norm_f32()),
        ("gdn_step_f32", spirv::gdn_step_f32()),
        (
            "gdn_step_batched_token_f32",
            spirv::gdn_step_batched_token_f32(),
        ),
        ("gdn_gates_f32", spirv::gdn_gates_f32()),
        ("gdn_gates_batched_f32", spirv::gdn_gates_batched_f32()),
        ("deinterleave_q_gate_f32", spirv::deinterleave_q_gate_f32()),
        (
            "deinterleave_q_gate_batched_f32",
            spirv::deinterleave_q_gate_batched_f32(),
        ),
        ("round_f16_f32", spirv::round_f16_f32()),
        ("kv_append_f32", spirv::kv_append_f32()),
        ("kv_append_batched_f32", spirv::kv_append_batched_f32()),
        ("kv_append_f16", spirv::kv_append_f16()),
        ("kv_append_batched_f16", spirv::kv_append_batched_f16()),
        ("kv_unpack_f16_to_f32", spirv::kv_unpack_f16_to_f32()),
        (
            "quant_q8k_4plane_src",
            spirv::quantize_f32_to_q8k_4plane(false),
        ),
        (
            "quant_q8k_4plane_batched_pinned",
            spirv::quantize_f32_to_q8k_4plane_batched(true),
        ),
        ("quant_q8_0_src", spirv::quantize_f32_to_q8_0(false, false)),
        ("quant_q8_0_pinned", spirv::quantize_f32_to_q8_0(true, true)),
        (
            "quant_q8_0_batched_src",
            spirv::quantize_f32_to_q8_0_batched(false, false),
        ),
        (
            "quant_q8_0_batched_pinned",
            spirv::quantize_f32_to_q8_0_batched(true, false),
        ),
        (
            "quant_q8_0_batched_native",
            spirv::quantize_f32_to_q8_0_batched(true, true),
        ),
        (
            "ffn_hadamard_batched_f32",
            spirv::ffn_hadamard_batched_f32(),
        ),
        ("f32_matvec", spirv::f32_matvec()),
        ("f32_gemm_batched", spirv::f32_gemm_batched()),
        ("mul_f32_elemwise", spirv::mul_f32_elemwise()),
        (
            "mul_f32_elemwise_batched",
            spirv::mul_f32_elemwise_batched(),
        ),
        ("permute_ssm_heads_f32", spirv::permute_ssm_heads_f32()),
        (
            "permute_ssm_heads_batched_f32",
            spirv::permute_ssm_heads_batched_f32(),
        ),
        ("copy_slice_f32", spirv::copy_slice_f32()),
        ("copy_slice_offset_f32", spirv::copy_slice_offset_f32()),
        ("attn_fused_f16prob", spirv::attn_fused_f16prob()),
        (
            "attn_fused_f16prob_f16kv",
            spirv::attn_fused_f16prob_f16kv(),
        ),
        ("attn_qk_scores", spirv::attn_qk_scores()),
        ("attn_qk_scores_f16kv", spirv::attn_qk_scores_f16kv()),
        ("attn_softmax_f16", spirv::attn_softmax_f16()),
        ("attn_pv_accum", spirv::attn_pv_accum()),
        ("attn_pv_accum_f16kv", spirv::attn_pv_accum_f16kv()),
        ("lora_b_accum_f32", spirv::lora_b_accum_f32()),
        (
            "lora_b_accum_batched_f32",
            spirv::lora_b_accum_batched_f32(),
        ),
        (
            "pq2_q8k_4tok_false",
            spirv::pq2_q8k_gemm_4tok_coalesced(false, false, false),
        ),
        (
            "pq2_q8_0_4tok_false",
            spirv::pq2_q8_0_gemm_4tok_coalesced(false),
        ),
        (
            "ptq1_q8_0_4tok_false_src",
            spirv::ptq1_q8_0_gemm_4tok_coalesced(false, false),
        ),
        (
            "ptq1_q8_0_4tok_false_pin",
            spirv::ptq1_q8_0_gemm_4tok_coalesced(false, true),
        ),
    ];
    for (name, words) in portable_modules {
        assert_eq!(
            spirv::validate_spirv_module(&words, Some(false)),
            Ok(()),
            "portable SPIR-V validation failed for {name}"
        );
    }
    for (name, words) in [
        (
            "pq2_q8k_gemv_sdot",
            spirv::pq2_q8k_gemv_coalesced(true, true, false),
        ),
        (
            "pq2_q8k_4tok_sdot",
            spirv::pq2_q8k_gemm_4tok_coalesced(true, false, false),
        ),
        (
            "pq2_q8_0_4tok_sdot",
            spirv::pq2_q8_0_gemm_4tok_coalesced(true),
        ),
        (
            "ptq1_q8_0_4tok_sdot_src",
            spirv::ptq1_q8_0_gemm_4tok_coalesced(true, false),
        ),
        (
            "ptq1_q8_0_4tok_sdot_pin",
            spirv::ptq1_q8_0_gemm_4tok_coalesced(true, true),
        ),
    ] {
        assert_eq!(
            spirv::validate_spirv_module(&words, Some(true)),
            Ok(()),
            "OpSDot SPIR-V validation failed for {name}"
        );
    }
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
    assert_eq!(cfg.prefill_tile_tokens, 16);
    assert_eq!(cfg.kv_precision.name(), "fp16");
    assert_eq!(VulkanKvPrecision::Fp32.name(), "fp32");

    // Odd-128 width synthetic hybrid model builds cleanly for PQ2_0 x Q8_0 / PTQ1_0 x Q8_0 parity.
    let odd_model =
        synthetic_hybrid_qwen35_model_with_layout("offline-odd128-hybrid", 384, 384, 32)?;
    let mut cpu_odd = Qwen35::new(odd_model, 16)?;
    let odd_l0 = cpu_odd.evaluate(0)?;
    assert_eq!(odd_l0.len(), 16);
    assert!(odd_l0.iter().all(|v| v.is_finite()));
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

#[test]
#[ignore = "Requires physical Qualcomm Adreno Vulkan Windows ARM64 hardware"]
fn live_direct_f16kv_attention_strategies_match_cpu_qwen35() -> TestResult {
    for (label, strategy) in [
        ("direct-f16kv-fused", AttentionStrategy::ForceFused),
        (
            "direct-f16kv-decomposed",
            AttentionStrategy::ForceDecomposed,
        ),
    ] {
        let cpu_model = synthetic_hybrid_qwen35_model(label)?;
        let gpu_model = synthetic_hybrid_qwen35_model(label)?;
        let mut cpu = Qwen35::with_adapters_and_kv(
            cpu_model,
            16,
            64 * 1024 * 1024,
            MatvecBackend::Auto,
            None,
            None,
            KvCachePolicy::F16,
        )?;
        let mut gpu = VulkanBatchedGraphExecutor::open(
            &gpu_model,
            VulkanBatchedConfig {
                kv_precision: VulkanKvPrecision::Fp16,
                attention_strategy: strategy,
                ..VulkanBatchedConfig::default()
            },
        )?;
        let cpu_t0 = cpu.evaluate(0)?;
        let gpu_t0 = gpu.evaluate_token(0)?;
        assert_close(&gpu_t0, &cpu_t0, 1e-4, &format!("{label} token 0"));

        cpu.ingest(1)?;
        gpu.ingest_token(1)?;
        cpu.ingest(2)?;
        gpu.ingest_token(2)?;

        let cpu_t3 = cpu.evaluate(3)?;
        let gpu_t3 = gpu.evaluate_token(3)?;
        assert_close(&gpu_t3, &cpu_t3, 1e-4, &format!("{label} token 3"));
    }
    Ok(())
}

#[expect(
    clippy::too_many_lines,
    reason = "Tests stacked RankOne + MultiRank GPU LoRA parity, 0.0 bypass, scale invalidation, and F32 multi-rank."
)]
#[test]
#[ignore = "Requires physical Qualcomm Adreno Vulkan Windows ARM64 hardware"]
fn live_lora_rank_one_multi_rank_and_dynamic_scale_match_cpu_qwen35() -> TestResult {
    let base_id = "live-lora-parity";
    let cpu_model_adapted = synthetic_hybrid_qwen35_model(base_id)?;
    let gpu_model_adapted = synthetic_hybrid_qwen35_model(base_id)?;
    let r1_model = synthetic_rank_one_adapter(&cpu_model_adapted, "r1-lora-id", 1.5, true, 0)?;
    let r1_gpu_model = synthetic_rank_one_adapter(&gpu_model_adapted, "r1-lora-id", 1.5, true, 0)?;
    let mr_model = synthetic_multi_rank_adapter(&cpu_model_adapted, "mr-f16-lora-id", 2.0, 4, 1)?;
    let mr_gpu_model =
        synthetic_multi_rank_adapter(&gpu_model_adapted, "mr-f16-lora-id", 2.0, 4, 1)?;

    let cpu_r1 = RankOneAdapter::from_model(r1_model, &cpu_model_adapted)?;
    let cpu_mr =
        MultiRankAdapter::from_model(&cpu_model_adapted, &mr_model, MultiRankPolicy::AnyLinear)?;
    let mut cpu_adapted = Qwen35::with_adapters_and_kv(
        cpu_model_adapted,
        16,
        64 * 1024 * 1024,
        MatvecBackend::Auto,
        Some((cpu_r1, 0.75)),
        Some((cpu_mr, 1.25)),
        KvCachePolicy::F16,
    )?;

    let mut gpu = VulkanBatchedGraphExecutor::open_with_adapters(
        &gpu_model_adapted,
        VulkanBatchedConfig::default(),
        VulkanBatchedAdapters {
            rank_one: Some(VulkanRankOneAdapterSpec {
                adapter: &r1_gpu_model,
                scale: 0.75,
            }),
            multi_rank: Some(VulkanMultiRankAdapterSpec {
                adapter: &mr_gpu_model,
                policy: MultiRankPolicy::AnyLinear,
                scale: 1.25,
            }),
        },
    )?;
    assert_eq!(gpu.rank_one_scale(), Some(0.75));
    assert_eq!(gpu.multi_rank_scale(), Some(1.25));
    let id_adapted = gpu.identity_sha256().to_owned();

    let ref_t0 = cpu_adapted.evaluate(0)?;
    let gpu_t0 = gpu.evaluate_token(0)?;
    assert_close(&gpu_t0, &ref_t0, 2e-4, "stacked lora token 0");
    let saved_snap = gpu.checkpoint()?;

    cpu_adapted.ingest(1)?;
    gpu.ingest_token(1)?;
    let ref_t2 = cpu_adapted.evaluate(2)?;
    let gpu_t2 = gpu.evaluate_token(2)?;
    assert_close(&gpu_t2, &ref_t2, 2e-4, "stacked lora token 2");

    // Fast-path `scale == 0.0` bypass must change identity, invalidate stale checkpoints,
    // and match unadapted base CPU logits cleanly.
    gpu.reset_with_adapter_scales(Some(0.0), Some(0.0))?;
    assert_eq!(gpu.rank_one_scale(), Some(0.0));
    assert_eq!(gpu.multi_rank_scale(), Some(0.0));
    let id_zero = gpu.identity_sha256().to_owned();
    assert_ne!(id_zero, id_adapted);
    assert!(gpu.restore(&saved_snap).is_err());

    let mut cpu_base = Qwen35::with_adapters_and_kv(
        synthetic_hybrid_qwen35_model(base_id)?,
        16,
        64 * 1024 * 1024,
        MatvecBackend::Auto,
        None,
        None,
        KvCachePolicy::F16,
    )?;
    let cpu_base_t0 = cpu_base.evaluate(0)?;
    let gpu_zero_t0 = gpu.evaluate_token(0)?;
    assert_close(&gpu_zero_t0, &cpu_base_t0, 1e-4, "scale 0.0 bypass token 0");
    let cpu_base_t1 = cpu_base.evaluate(1)?;
    let gpu_zero_t1 = gpu.evaluate_token(1)?;
    assert_close(&gpu_zero_t1, &cpu_base_t1, 1e-4, "scale 0.0 bypass token 1");

    // Dynamic scale updates invalidate identity and reproduce updated CPU reference after reset.
    gpu.set_rank_one_scale(0.5)?;
    let id_half_zero = gpu.identity_sha256().to_owned();
    assert_ne!(id_half_zero, id_zero);
    assert_ne!(id_half_zero, id_adapted);
    gpu.set_multi_rank_scale(1.0)?;
    let id_half_one = gpu.identity_sha256().to_owned();
    assert_ne!(id_half_one, id_half_zero);
    gpu.reset()?;
    assert_eq!(gpu.identity_sha256(), id_half_one);

    // Reject invalid scales and verify error does not mutate configured scales.
    assert!(gpu.set_rank_one_scale(-0.1).is_err());
    assert!(gpu.set_rank_one_scale(2.5).is_err());
    assert!(gpu.set_multi_rank_scale(f32::NAN).is_err());
    assert_eq!(gpu.rank_one_scale(), Some(0.5));
    assert_eq!(gpu.multi_rank_scale(), Some(1.0));

    let base_cpu2 = synthetic_hybrid_qwen35_model(base_id)?;
    let r1_2 = synthetic_rank_one_adapter(&base_cpu2, "r1-lora-id", 1.5, true, 0)?;
    let mr_2 = synthetic_multi_rank_adapter(&base_cpu2, "mr-f16-lora-id", 2.0, 4, 1)?;
    let mut cpu_updated = Qwen35::with_adapters_and_kv(
        base_cpu2,
        16,
        64 * 1024 * 1024,
        MatvecBackend::Auto,
        Some((RankOneAdapter::from_model(r1_2, &gpu_model_adapted)?, 0.5)),
        Some((
            MultiRankAdapter::from_model(&gpu_model_adapted, &mr_2, MultiRankPolicy::AnyLinear)?,
            1.0,
        )),
        KvCachePolicy::F16,
    )?;
    let cpu_upd_t0 = cpu_updated.evaluate(0)?;
    let gpu_upd_t0 = gpu.evaluate_token(0)?;
    assert_close(&gpu_upd_t0, &cpu_upd_t0, 2e-4, "rescaled lora token 0");

    // Verify F32 multi-rank adapter path and error when setting rank-one on multi-rank-only executor.
    let f32_host_model = synthetic_hybrid_qwen35_model("mr-f32-base")?;
    let f32_device_model = synthetic_hybrid_qwen35_model("mr-f32-base")?;
    let mr_f32 = synthetic_multi_rank_adapter(&f32_host_model, "mr-f32-lora-id", 3.0, 4, 0)?;
    let mut cpu_mr_f32 = Qwen35::with_adapters_and_kv(
        f32_host_model,
        16,
        64 * 1024 * 1024,
        MatvecBackend::Auto,
        None,
        Some((
            MultiRankAdapter::from_model(&f32_device_model, &mr_f32, MultiRankPolicy::AnyLinear)?,
            1.25,
        )),
        KvCachePolicy::F16,
    )?;
    let mut gpu_mr_f32 = VulkanBatchedGraphExecutor::open_with_adapters(
        &f32_device_model,
        VulkanBatchedConfig::default(),
        VulkanBatchedAdapters {
            rank_one: None,
            multi_rank: Some(VulkanMultiRankAdapterSpec {
                adapter: &mr_f32,
                policy: MultiRankPolicy::AnyLinear,
                scale: 1.25,
            }),
        },
    )?;
    assert!(gpu_mr_f32.set_rank_one_scale(1.0).is_err());
    let cpu_f32_t0 = cpu_mr_f32.evaluate(0)?;
    let gpu_f32_t0 = gpu_mr_f32.evaluate_token(0)?;
    assert_close(&gpu_f32_t0, &cpu_f32_t0, 2e-4, "f32 multi-rank token 0");
    Ok(())
}

#[test]
#[ignore = "Requires physical Qualcomm Adreno Vulkan Windows ARM64 hardware"]
fn live_decoder_trait_and_runtime_parity() -> TestResult {
    let cpu_model = synthetic_hybrid_qwen35_model("decoder-trait-parity")?;
    let gpu_model = synthetic_hybrid_qwen35_model("decoder-trait-parity")?;
    let limited_model = synthetic_hybrid_qwen35_model("decoder-trait-limit")?;

    let mut cpu = Qwen35::new(cpu_model, 16)?;
    let mut gpu = VulkanBatchedGraphExecutor::open(&gpu_model, VulkanBatchedConfig::default())?;

    assert_eq!(Decoder::vocab_size(&gpu), 16);
    assert_ne!(Decoder::identity(&gpu), "");
    Decoder::admit_work(&gpu, 2, 1)?;

    let mut gpu_logits = Vec::new();
    Decoder::evaluate_into(&mut gpu, 0, &mut gpu_logits)?;
    let cpu_t0 = cpu.evaluate(0)?;
    assert_close(&gpu_logits, &cpu_t0, 1e-4, "Decoder::evaluate_into token 0");

    let mut cp = Decoder::checkpoint(&gpu)?;
    Decoder::ingest_block(&mut gpu, &[1, 2], &|| Ok(()))?;
    cpu.ingest(1)?;
    cpu.ingest(2)?;

    let mut finite_logits = Vec::new();
    Decoder::evaluate_finite_into(&mut gpu, 3, &mut finite_logits)?;
    let cpu_t3 = cpu.evaluate(3)?;
    assert_close(
        &finite_logits,
        &cpu_t3,
        1e-4,
        "Decoder::ingest_block + evaluate_finite_into",
    );

    Decoder::checkpoint_into(&gpu, &mut cp)?;
    assert!(Decoder::checkpoint_bytes(&gpu) > 0);
    Decoder::restore(&mut gpu, &cp)?;
    Decoder::reset(&mut gpu)?;
    let reset_logits = Decoder::evaluate(&mut gpu, 0)?;
    assert_close(&reset_logits, &cpu_t0, 1e-4, "Decoder::reset + evaluate");

    // Verify Runtime::new accepts VulkanBatchedGraphExecutor cleanly.
    Decoder::reset(&mut gpu)?;
    let cp_bytes = Decoder::checkpoint_bytes(&gpu).max(4096);
    let _runtime = Runtime::new(
        gpu,
        Limits {
            context_tokens: 16,
            checkpoints: 2,
            checkpoint_bytes: cp_bytes.saturating_mul(8),
        },
    )?;

    // Verify `evaluate_into` clears stale output buffer when exceeding `context_limit`.
    let mut limited = VulkanBatchedGraphExecutor::open(
        &limited_model,
        VulkanBatchedConfig {
            context_limit: Some(1),
            ..VulkanBatchedConfig::default()
        },
    )?;
    let mut err_logits = vec![42.0_f32; 4];
    Decoder::evaluate_into(&mut limited, 0, &mut err_logits)?;
    assert!(Decoder::evaluate_into(&mut limited, 1, &mut err_logits).is_err());
    assert_eq!(err_logits, Vec::<f32>::new());
    Ok(())
}

#[expect(
    clippy::too_many_lines,
    reason = "Tests batched ingest parity across Fp16/Fp32 KV caches, tile sizes 1..=19, submit amortization, and checkpoint identity."
)]
#[test]
#[ignore = "Requires physical Qualcomm Adreno Vulkan Windows ARM64 hardware"]
fn live_batched_prefill_matches_sequential_ingest_and_checkpoint() -> TestResult {
    let invalid_zero = synthetic_hybrid_qwen35_model("invalid-tile-0")?;
    assert!(
        VulkanBatchedGraphExecutor::open(
            &invalid_zero,
            VulkanBatchedConfig {
                prefill_tile_tokens: 0,
                ..VulkanBatchedConfig::default()
            },
        )
        .is_err()
    );
    let invalid_over = synthetic_hybrid_qwen35_model("invalid-tile-257")?;
    assert!(
        VulkanBatchedGraphExecutor::open(
            &invalid_over,
            VulkanBatchedConfig {
                prefill_tile_tokens: bonsai_prism_runtime::MAX_PREFILL_BLOCK_TOKENS + 1,
                ..VulkanBatchedConfig::default()
            },
        )
        .is_err()
    );

    let full_seq: [u32; 19] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 3, 7, 11,
    ];
    for kv_precision in [VulkanKvPrecision::Fp16, VulkanKvPrecision::Fp32] {
        let mut seq_1tok_exec = VulkanBatchedGraphExecutor::open(
            &synthetic_hybrid_qwen35_model("batched-vs-seq-base")?,
            VulkanBatchedConfig {
                kv_precision,
                prefill_tile_tokens: 1,
                ..VulkanBatchedConfig::default()
            },
        )?;
        let mut cpu = Qwen35::with_adapters_and_kv(
            synthetic_hybrid_qwen35_model("batched-vs-seq-base")?,
            32,
            64 * 1024 * 1024,
            MatvecBackend::Auto,
            None,
            None,
            KvCachePolicy::F16,
        )?;
        let mut seq_refs = Vec::new();
        for &len in &[1_usize, 3, 4, 7, 8, 16, 19] {
            seq_1tok_exec.reset()?;
            cpu.reset()?;
            let tokens = &full_seq[..len];
            let seq_submits_before = seq_1tok_exec.submits_completed();
            for &tok in tokens {
                seq_1tok_exec.ingest_token(tok)?;
                cpu.ingest(tok)?;
            }
            let seq_submits = seq_1tok_exec.submits_completed() - seq_submits_before;
            let seq_logits = seq_1tok_exec.evaluate_token(2)?;
            let cpu_logits = cpu.evaluate(2)?;
            assert_close_rel(&seq_logits, &cpu_logits, 5e-3, 5e-3, "seq vs cpu reference");
            seq_refs.push((len, seq_submits, seq_logits, cpu_logits));
        }

        for &tile_cfg in &[8_usize, 16] {
            let tile_config = VulkanBatchedConfig {
                kv_precision,
                prefill_tile_tokens: tile_cfg,
                ..VulkanBatchedConfig::default()
            };
            let mut same_cfg_seq = VulkanBatchedGraphExecutor::open(
                &synthetic_hybrid_qwen35_model("batched-vs-seq-base")?,
                tile_config.clone(),
            )?;
            let mut bat_exec = VulkanBatchedGraphExecutor::open(
                &synthetic_hybrid_qwen35_model("batched-vs-seq-base")?,
                tile_config,
            )?;

            for (len, seq_submits, seq_logits, cpu_logits) in &seq_refs {
                let len = *len;
                let tokens = &full_seq[..len];
                same_cfg_seq.reset()?;
                bat_exec.reset()?;

                for &tok in tokens {
                    same_cfg_seq.ingest_token(tok)?;
                }
                let seq_cp = same_cfg_seq.checkpoint()?;

                let submits_before = bat_exec.submits_completed();
                Decoder::ingest_block(&mut bat_exec, tokens, &|| Ok(()))?;
                let bat_submits = bat_exec.submits_completed() - submits_before;
                assert_eq!(bat_exec.position(), len);
                if len >= 4 {
                    assert!(
                        bat_submits < *seq_submits,
                        "expected submit amortization for len={len}, tile={tile_cfg}: bat={bat_submits} seq={seq_submits}"
                    );
                }

                let bat_cp = bat_exec.checkpoint()?;
                assert_eq!(bat_cp.position(), seq_cp.position());
                assert_eq!(bat_cp.layer_count(), seq_cp.layer_count());
                assert_eq!(
                    bat_cp.identity_sha256()?,
                    seq_cp.identity_sha256()?,
                    "checkpoint SHA-256 mismatch for len={len} tile={tile_cfg} {kv_precision:?}"
                );
                let bat_logits = bat_exec.evaluate_token(2)?;
                assert_close(
                    &bat_logits,
                    seq_logits,
                    1e-5,
                    &format!("batched vs sequential len={len} tile={tile_cfg} {kv_precision:?}"),
                );
                assert_close_rel(
                    &bat_logits,
                    cpu_logits,
                    5e-3,
                    5e-3,
                    &format!("batched vs cpu len={len} tile={tile_cfg} {kv_precision:?}"),
                );
            }
        }
    }
    Ok(())
}

#[expect(
    clippy::too_many_lines,
    reason = "Tests odd-128 Q8_0 4-token GEMM, stacked RankOne+MultiRank LoRA, zero-scale bypass, sub-tiled attention fallback, and cancellation rollback."
)]
#[test]
#[ignore = "Requires physical Qualcomm Adreno Vulkan Windows ARM64 hardware"]
fn live_batched_prefill_odd128_q8_0_lora_subtile_and_cancellation() -> TestResult {
    // 1. Odd-128 column hybrid model + stacked RankOne + MultiRank LoRA across T=1, 3, 7, 16.
    let base_seq_model = synthetic_hybrid_qwen35_model_with_layout("odd128-lora", 384, 384, 32)?;
    let base_bat_model = synthetic_hybrid_qwen35_model_with_layout("odd128-lora", 384, 384, 32)?;
    let r1_model = synthetic_rank_one_adapter(&base_seq_model, "r1-odd128", 1.5, true, 0)?;
    let mr_model = synthetic_multi_rank_adapter(&base_seq_model, "mr-odd128", 2.0, 4, 1)?;
    let make_adapters = || VulkanBatchedAdapters {
        rank_one: Some(VulkanRankOneAdapterSpec {
            adapter: &r1_model,
            scale: 0.75,
        }),
        multi_rank: Some(VulkanMultiRankAdapterSpec {
            adapter: &mr_model,
            policy: MultiRankPolicy::AnyLinear,
            scale: 1.25,
        }),
    };

    let mut seq_exec = VulkanBatchedGraphExecutor::open_with_adapters(
        &base_seq_model,
        VulkanBatchedConfig {
            prefill_tile_tokens: 1,
            ..VulkanBatchedConfig::default()
        },
        make_adapters(),
    )?;
    let mut bat_exec = VulkanBatchedGraphExecutor::open_with_adapters(
        &base_bat_model,
        VulkanBatchedConfig {
            prefill_tile_tokens: 16,
            ..VulkanBatchedConfig::default()
        },
        make_adapters(),
    )?;

    let tokens: [u32; 7] = [0, 2, 4, 6, 8, 10, 12];
    for &t in &tokens {
        seq_exec.ingest_token(t)?;
    }
    bat_exec.ingest_tokens_batched(&tokens)?;
    assert_eq!(bat_exec.position(), seq_exec.position());
    let seq_logits = seq_exec.evaluate_token(5)?;
    let bat_logits = bat_exec.evaluate_token(5)?;
    assert_close(&bat_logits, &seq_logits, 2e-4, "odd128 lora batched vs seq");

    // Zero-scale LoRA bypass in batched mode must match unadapted sequential base.
    seq_exec.reset_with_adapter_scales(Some(0.0), Some(0.0))?;
    bat_exec.reset_with_adapter_scales(Some(0.0), Some(0.0))?;
    for &t in &tokens[..4] {
        seq_exec.ingest_token(t)?;
    }
    bat_exec.ingest_tokens_batched(&tokens[..4])?;
    let seq_zero = seq_exec.evaluate_token(1)?;
    let bat_zero = bat_exec.evaluate_token(1)?;
    assert_close(&bat_zero, &seq_zero, 1e-4, "odd128 zero-scale batched");

    // 2. Tight scratch budget forcing attention sub-tiling (score_tokens < tile_tokens)
    //    with Decomposed3Stage attention on context_length=1024 (1-tok score = 16 KiB).
    let sub_model = synthetic_hybrid_qwen35_model_with_layout("subtile-attn", 384, 384, 1024)?;
    let mut subtile_exec = VulkanBatchedGraphExecutor::open(
        &sub_model,
        VulkanBatchedConfig {
            prefill_tile_tokens: 8,
            attention_strategy: AttentionStrategy::ForceDecomposed,
            scratch_budget_bytes: 450 * 1024,
            ..VulkanBatchedConfig::default()
        },
    )?;
    let mut sub_seq = VulkanBatchedGraphExecutor::open(
        &synthetic_hybrid_qwen35_model_with_layout("subtile-attn", 384, 384, 1024)?,
        VulkanBatchedConfig {
            prefill_tile_tokens: 1,
            attention_strategy: AttentionStrategy::ForceDecomposed,
            ..VulkanBatchedConfig::default()
        },
    )?;
    let sub_tokens: [u32; 5] = [1, 3, 5, 7, 9];
    for &t in &sub_tokens {
        sub_seq.ingest_token(t)?;
    }
    subtile_exec.ingest_tokens_batched(&sub_tokens)?;
    assert_close(
        &subtile_exec.evaluate_token(2)?,
        &sub_seq.evaluate_token(2)?,
        1e-4,
        "sub-tiled decomposed attention parity",
    );

    // 3. Cancellation & fault rollback: pre-tile cancellation leaves state intact; mid-tile
    //    cancellation with `layers_per_submit = 1` latches inconsistent state until `restore`.
    let mut cancel_exec = VulkanBatchedGraphExecutor::open(
        &synthetic_hybrid_qwen35_model("cancel-test")?,
        VulkanBatchedConfig {
            tdr_policy: TdrChunkPolicy::new(1, 2048)?,
            prefill_tile_tokens: 4,
            ..VulkanBatchedConfig::default()
        },
    )?;
    cancel_exec.ingest_tokens_batched(&[0, 1])?;
    let good_cp = cancel_exec.checkpoint()?;

    let err_before = Decoder::ingest_block(&mut cancel_exec, &[2, 3, 4], &|| {
        Err("pre-tile cancel".into())
    });
    assert!(err_before.is_err());
    assert_eq!(cancel_exec.position(), 2);
    let _ = cancel_exec.evaluate_token(2)?;

    cancel_exec.restore(&good_cp)?;
    let checks = AtomicUsize::new(0);
    let err_mid = Decoder::ingest_block(&mut cancel_exec, &[2, 3, 4], &|| {
        let count = checks.fetch_add(1, Ordering::Relaxed);
        if count >= 1 {
            return Err("mid-tile cancel".into());
        }
        Ok(())
    });
    assert!(err_mid.is_err());
    assert!(cancel_exec.evaluate_token(2).is_err());
    cancel_exec.restore(&good_cp)?;
    assert_eq!(cancel_exec.position(), 2);
    assert_eq!(cancel_exec.evaluate_token(2)?.len(), 16);
    Ok(())
}
