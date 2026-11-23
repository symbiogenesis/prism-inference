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
    AttentionStrategy, CommandBatchRecorder, ProjectionKernelPath, TdrChunkPolicy,
    VulkanBatchedAdapters, VulkanBatchedConfig, VulkanBatchedGraphExecutor, VulkanKvPrecision,
    VulkanMultiRankAdapterSpec, VulkanPq2Executor, VulkanRankOneAdapterSpec,
    device::{AdrenoVulkanDevice, BufferId, BufferSpan, ComputeStage},
    spirv,
};
use bonsai_gguf::MetadataValue;
use bonsai_prism_runtime::{
    Decoder,
    graph::{KvCachePolicy, Qwen35},
    kernels::{MatvecBackend, Pq2ProjectionExecutor, Q8Block, Q8KBlock},
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

fn build_q8_0_tensor(name: &str, cols: usize, rows: usize, seed: usize) -> Tensor {
    let blocks_per_row = cols / 32;
    let mut data = Vec::with_capacity(rows * blocks_per_row * 34);
    for r in 0..rows {
        for b in 0..blocks_per_row {
            let sign = if (r + b + seed).is_multiple_of(3) {
                -1.0_f32
            } else {
                1.0_f32
            };
            let scale_val = sign
                * 0.007_812_5_f32
                * f32::from(u8::try_from((r + b + seed) % 7 + 1).unwrap_or(1));
            data.extend_from_slice(&f16_scale(scale_val));
            for i in 0..32 {
                let raw = i32::try_from((r * 13 + b * 7 + i * 5 + seed) % 241).unwrap_or(0) - 120;
                data.push(i8::try_from(raw).unwrap_or(0).cast_unsigned());
            }
        }
    }
    Tensor {
        name: name.to_owned(),
        dimensions: vec![cols, rows],
        kind: 8,
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
        (
            "conv1d_silu_batched_inplace_f32",
            spirv::conv1d_silu_batched_inplace_f32(),
        ),
        ("l2_norm_f32", spirv::l2_norm_f32()),
        (
            "l2_norm_batched_strided_f32",
            spirv::l2_norm_batched_strided_f32(),
        ),
        ("gdn_step_f32", spirv::gdn_step_f32()),
        (
            "gdn_step_batched_token_f32",
            spirv::gdn_step_batched_token_f32(),
        ),
        ("gdn_scan_batched_f32", spirv::gdn_scan_batched_f32()),
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
            "fused_basis_hadamard1024_q8k_unpinned",
            spirv::fused_basis_hadamard1024_q8k_4plane(false),
        ),
        (
            "fused_basis_hadamard1024_q8k_pinned",
            spirv::fused_basis_hadamard1024_q8k_4plane(true),
        ),
        (
            "fused_basis_hadamard1024_q8_0_src",
            spirv::fused_basis_hadamard1024_q8_0(false, false),
        ),
        (
            "fused_basis_hadamard1024_q8_0_pinned",
            spirv::fused_basis_hadamard1024_q8_0(true, false),
        ),
        (
            "fused_basis_hadamard1024_q8_0_native",
            spirv::fused_basis_hadamard1024_q8_0(true, true),
        ),
        (
            "pq2_q8k_4tok_false",
            spirv::pq2_q8k_gemm_4tok_coalesced(false, false, false),
        ),
        (
            "pq2_q8k_16tok_false",
            spirv::pq2_q8k_gemm_16tok_coalesced(false, false, false),
        ),
        (
            "pq2_q8k_16tok_false_fused_res",
            spirv::pq2_q8k_gemm_16tok_coalesced(false, true, true),
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
        (
            "ptq1_q8_0_16tok_false_src",
            spirv::ptq1_q8_0_gemm_16tok_coalesced(false, false),
        ),
        (
            "ptq1_q8_0_16tok_false_pin",
            spirv::ptq1_q8_0_gemm_16tok_coalesced(false, true),
        ),
        ("q8_0_q8_0_false", spirv::q8_0_q8_0_gemv_coalesced(false)),
        (
            "q8_0_q8_0_4tok_false",
            spirv::q8_0_q8_0_gemm_4tok_coalesced(false),
        ),
        (
            "q8_0_q8_0_16tok_false",
            spirv::q8_0_q8_0_gemm_16tok_coalesced(false),
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
            "pq2_q8k_16tok_sdot",
            spirv::pq2_q8k_gemm_16tok_coalesced(true, false, false),
        ),
        (
            "pq2_q8k_16tok_sdot_fused_res",
            spirv::pq2_q8k_gemm_16tok_coalesced(true, true, true),
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
        (
            "ptq1_q8_0_16tok_sdot_src",
            spirv::ptq1_q8_0_gemm_16tok_coalesced(true, false),
        ),
        (
            "ptq1_q8_0_16tok_sdot_pin",
            spirv::ptq1_q8_0_gemm_16tok_coalesced(true, true),
        ),
        ("q8_0_q8_0_sdot", spirv::q8_0_q8_0_gemv_coalesced(true)),
        (
            "q8_0_q8_0_4tok_sdot",
            spirv::q8_0_q8_0_gemm_4tok_coalesced(true),
        ),
        (
            "q8_0_q8_0_16tok_sdot",
            spirv::q8_0_q8_0_gemm_16tok_coalesced(true),
        ),
    ] {
        assert_eq!(
            spirv::validate_spirv_module(&words, Some(true)),
            Ok(()),
            "OpSDot SPIR-V validation failed for {name}"
        );
    }
    assert_eq!(
        spirv::extract_local_size(&spirv::fused_basis_hadamard1024_q8k_4plane(true)),
        Ok([256, 1, 1])
    );
    assert_eq!(
        spirv::extract_local_size(&spirv::pq2_q8k_gemm_16tok_coalesced(true, false, false)),
        Ok([64, 8, 1])
    );
    assert_eq!(
        spirv::extract_local_size(&spirv::q8_0_q8_0_gemm_16tok_coalesced(false)),
        Ok([64, 8, 1])
    );
    assert_eq!(
        spirv::extract_local_size(&spirv::ptq1_q8_0_gemm_16tok_coalesced(true, true)),
        Ok([64, 8, 1])
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

#[expect(
    clippy::suboptimal_flops,
    reason = "Scalar reference preserves exact left-to-right (w_scale * a_scale) * isum rounding."
)]
fn cpu_q8_0_q8_0_reference(tensor: &Tensor, activation: &[Q8Block]) -> Vec<f32> {
    let cols = tensor.dimensions[0];
    let rows = tensor.dimensions[1];
    let blocks_per_row = cols / 32;
    let row_bytes = blocks_per_row * 34;
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut acc = 0.0_f32;
        for (b, act) in activation.iter().enumerate() {
            let off = r * row_bytes + b * 34;
            let scale_bits = u16::from_le_bytes([tensor.data[off], tensor.data[off + 1]]);
            let w_scale = bonsai_ternary::half_to_float(scale_bits);
            let mut isum = 0_i32;
            for (i, &av) in act.values.iter().enumerate() {
                let wv = i8::from_le_bytes([tensor.data[off + 2 + i]]);
                isum += i32::from(wv) * i32::from(av);
            }
            // All Q8_0 32-elem integer dot sums satisfy |isum| <= 32*128*127 = 520_192, which fits in i16*32768 + i16 exact f32.
            let hi = isum >> 15;
            let lo = isum - (hi << 15);
            let exact_f32 = f32::from(i16::try_from(hi).unwrap_or(0)) * 32768.0_f32
                + f32::from(i16::try_from(lo).unwrap_or(0));
            acc += (w_scale * act.scale) * exact_f32;
        }
        out.push(acc);
    }
    out
}

fn clone_with_q8_0_weights(base: &Model, seed: usize) -> Model {
    let mut tensors = base.tensors.clone();
    let embd = tensors["output.weight"].dimensions[0];
    let vocab = tensors["output.weight"].dimensions[1];
    tensors.insert(
        "output.weight".to_owned(),
        build_q8_0_tensor("output.weight", embd, vocab, seed + 1),
    );
    // Mix Q8_0 with PQ2 (`ffn_up`) and PTQ1 (`attn_q`, `attn_v`) inside sibling projection groups.
    let ffn = tensors["blk.0.ffn_gate.weight"].dimensions[1];
    tensors.insert(
        "blk.0.ffn_gate.weight".to_owned(),
        build_q8_0_tensor("blk.0.ffn_gate.weight", embd, ffn, seed + 2),
    );
    let ssm_values = tensors["blk.1.ssm_alpha.weight"].dimensions[1];
    tensors.insert(
        "blk.1.ssm_alpha.weight".to_owned(),
        build_q8_0_tensor("blk.1.ssm_alpha.weight", embd, ssm_values, seed + 3),
    );
    let kv_width = tensors["blk.2.attn_k.weight"].dimensions[1];
    tensors.insert(
        "blk.2.attn_k.weight".to_owned(),
        build_q8_0_tensor("blk.2.attn_k.weight", embd, kv_width, seed + 4),
    );
    Model {
        metadata: base.metadata.clone(),
        tensors,
        identity: format!("{}-q8_0", base.identity),
    }
}

#[test]
#[ignore = "Requires physical Qualcomm Adreno Vulkan Windows ARM64 hardware"]
fn live_q8_0_direct_projection_matches_cpu() -> TestResult {
    // Direct Q8_0 x Q8_0 projection parity on both OpSDot and Portable fallback with non-64 rows
    // and non-128-multiple width (`cols = 160`, `rows = 71`).
    let tensor = build_q8_0_tensor("direct.q8_0", 160, 71, 9);
    let act: Vec<Q8Block> = (0..5)
        .map(|b| {
            let values = std::array::from_fn(|i| {
                let raw = i32::try_from((b * 17 + i * 9) % 251).unwrap_or(0) - 125;
                i8::try_from(raw).unwrap_or(0)
            });
            Q8Block {
                scale: 0.015_625_f32 * f32::from(u8::try_from(b + 1).unwrap_or(1)),
                sum: 0.0,
                values,
            }
        })
        .collect();
    let expected = cpu_q8_0_q8_0_reference(&tensor, &act);

    for path in [
        ProjectionKernelPath::IntegerDotPacked4x8,
        ProjectionKernelPath::PortableIntegerFallback,
    ] {
        let mut exec = match VulkanPq2Executor::open_with_path(path, 64 * 1024 * 1024) {
            Ok(e) => e,
            Err(bonsai_adreno_vulkan::Error::Unsupported)
                if path == ProjectionKernelPath::IntegerDotPacked4x8 =>
            {
                continue;
            }
            Err(e) => return Err(e.into()),
        };
        assert!(exec.supports_q8(&tensor));
        assert_eq!(exec.prewarm_tensors(&[&tensor])?, 1);
        let mut out = Vec::new();
        exec.project_q8(&tensor, &act, false, &mut out)?;
        assert_eq!(out.len(), expected.len());
        for (idx, (&a, &e)) in out.iter().zip(&expected).enumerate() {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "{path:?} direct Q8_0 row {idx} bit mismatch: {a} != {e}"
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "Requires physical Qualcomm Adreno Vulkan Windows ARM64 hardware"]
fn live_q8_0_coalesced_projection_and_lm_head_matches_cpu() -> TestResult {
    // End-to-end graph parity with Q8_0 output.weight (lm_head) + mixed sibling groups,
    // both single-token and batched multi-token (T = 5), pinned and unpinned, sdot and portable.
    let base = synthetic_hybrid_qwen35_model_with_layout("q8-0-hybrid", 256, 256, 32)?;
    let q8_model = clone_with_q8_0_weights(&base, 17);

    for pinned in [false, true] {
        for path in [
            ProjectionKernelPath::IntegerDotPacked4x8,
            ProjectionKernelPath::PortableIntegerFallback,
        ] {
            let mut cpu = (!pinned)
                .then(|| Qwen35::new(clone_with_q8_0_weights(&base, 17), 16))
                .transpose()?;
            let mut seq_exec = match VulkanBatchedGraphExecutor::open(
                &q8_model,
                VulkanBatchedConfig {
                    kernel_path: Some(path),
                    pinned_arithmetic: Some(pinned),
                    prefill_tile_tokens: 1,
                    ..VulkanBatchedConfig::default()
                },
            ) {
                Ok(e) => e,
                Err(bonsai_adreno_vulkan::Error::Unsupported)
                    if path == ProjectionKernelPath::IntegerDotPacked4x8 =>
                {
                    continue;
                }
                Err(e) => return Err(e.into()),
            };
            let mut bat_exec = VulkanBatchedGraphExecutor::open(
                &q8_model,
                VulkanBatchedConfig {
                    kernel_path: Some(path),
                    pinned_arithmetic: Some(pinned),
                    prefill_tile_tokens: 16,
                    ..VulkanBatchedConfig::default()
                },
            )?;

            let prompt = [0_u32, 3, 5, 7, 9];
            for &tok in &prompt {
                if let Some(cpu_ref) = cpu.as_mut() {
                    cpu_ref.ingest(tok)?;
                }
                seq_exec.ingest_token(tok)?;
            }
            bat_exec.ingest_tokens_batched(&prompt)?;

            let seq_logits = seq_exec.evaluate_token(4)?;
            let bat_logits = bat_exec.evaluate_token(4)?;

            if let Some(cpu_ref) = cpu.as_mut() {
                let cpu_logits = cpu_ref.evaluate(4)?;
                assert_close_rel(
                    &seq_logits,
                    &cpu_logits,
                    5e-3,
                    5e-3,
                    &format!("Q8_0 seq vs cpu pinned={pinned} {path:?}"),
                );
            }
            assert_close(
                &bat_logits,
                &seq_logits,
                1e-5,
                &format!("Q8_0 bat vs seq pinned={pinned} {path:?}"),
            );
        }
    }
    Ok(())
}

fn synthetic_gdn_edge_model(
    identity: &str,
    ssm_head: usize,
    ssm_keys: usize,
    ssm_values: usize,
    conv_k: usize,
) -> ModelResult {
    let mut model = synthetic_hybrid_qwen35_model_with_layout(identity, 256, 256, 32)?;
    let embd = 256_usize;
    let ssm_inner = ssm_head * ssm_values;
    let conv_ch = 2 * ssm_head * ssm_keys + ssm_inner;
    model
        .metadata
        .insert("qwen35.block_count".to_owned(), MetadataValue::from(1_u32));
    model.metadata.insert(
        "qwen35.attention.recurrent_layers".to_owned(),
        MetadataValue::from(vec![true]),
    );
    model.metadata.insert(
        "qwen35.ssm.state_size".to_owned(),
        MetadataValue::from(u32::try_from(ssm_head)?),
    );
    model.metadata.insert(
        "qwen35.ssm.group_count".to_owned(),
        MetadataValue::from(u32::try_from(ssm_keys)?),
    );
    model.metadata.insert(
        "qwen35.ssm.time_step_rank".to_owned(),
        MetadataValue::from(u32::try_from(ssm_values)?),
    );
    model.metadata.insert(
        "qwen35.ssm.inner_size".to_owned(),
        MetadataValue::from(u32::try_from(ssm_inner)?),
    );
    model.metadata.insert(
        "qwen35.ssm.conv_kernel".to_owned(),
        MetadataValue::from(u32::try_from(conv_k)?),
    );
    model
        .tensors
        .retain(|k, _| !k.starts_with("blk.1.") && !k.starts_with("blk.2."));
    model.tensors.insert(
        "blk.0.attn_qkv.weight".to_owned(),
        build_pq2_tensor("blk.0.attn_qkv.weight", embd, conv_ch, 21),
    );
    model.tensors.insert(
        "blk.0.attn_gate.weight".to_owned(),
        build_pq2_tensor("blk.0.attn_gate.weight", embd, ssm_inner, 22),
    );
    model.tensors.insert(
        "blk.0.ssm_beta.weight".to_owned(),
        f32_tensor(
            "blk.0.ssm_beta.weight",
            &[embd, ssm_values],
            &fill_pattern(embd * ssm_values, 0.02, 23),
        ),
    );
    model.tensors.insert(
        "blk.0.ssm_alpha.weight".to_owned(),
        f32_tensor(
            "blk.0.ssm_alpha.weight",
            &[embd, ssm_values],
            &fill_pattern(embd * ssm_values, 0.02, 24),
        ),
    );
    model.tensors.insert(
        "blk.0.ssm_conv1d.weight".to_owned(),
        f32_tensor(
            "blk.0.ssm_conv1d.weight",
            &[conv_k, conv_ch],
            &fill_pattern(conv_k * conv_ch, 0.05, 25),
        ),
    );
    model.tensors.insert(
        "blk.0.ssm_dt.bias".to_owned(),
        f32_tensor(
            "blk.0.ssm_dt.bias",
            &[ssm_values],
            &vec![0.1_f32; ssm_values],
        ),
    );
    model.tensors.insert(
        "blk.0.ssm_a".to_owned(),
        f32_tensor("blk.0.ssm_a", &[ssm_values], &vec![-0.25_f32; ssm_values]),
    );
    model.tensors.insert(
        "blk.0.ssm_norm.weight".to_owned(),
        f32_tensor(
            "blk.0.ssm_norm.weight",
            &[ssm_head],
            &vec![1.0_f32; ssm_head],
        ),
    );
    model.tensors.insert(
        "blk.0.ssm_out.weight".to_owned(),
        build_pq2_tensor("blk.0.ssm_out.weight", ssm_inner, embd, 26),
    );
    Ok(model)
}

#[test]
#[ignore = "Requires physical Qualcomm Adreno Vulkan Windows ARM64 hardware"]
fn live_batched_gdn_scan_and_conv1d_edge_geometries_match_sequential() -> TestResult {
    let prompt = [0_u32, 3, 6, 9, 12, 2];
    for (ssm_head, ssm_keys, ssm_values, conv_k) in [
        (64_usize, 2_usize, 4_usize, 1_usize), // conv_k = 1 (0 history taps)
        (128, 1, 2, 8),                        // conv_k = 8 (7 history taps), 2 floats/lane
        (256, 1, 2, 4),                        // ssm_head = 256 (4 floats/lane)
        (64, 2, 4, 9),                         // conv_k = 9 (>8 serial fallback)
    ] {
        let id = format!("gdn-edge-h{ssm_head}-k{conv_k}");
        let cpu_model = synthetic_gdn_edge_model(&id, ssm_head, ssm_keys, ssm_values, conv_k)?;
        let gpu_seq_model = synthetic_gdn_edge_model(&id, ssm_head, ssm_keys, ssm_values, conv_k)?;
        let gpu_bat_model = synthetic_gdn_edge_model(&id, ssm_head, ssm_keys, ssm_values, conv_k)?;

        let mut cpu = Qwen35::new(cpu_model, 16)?;
        let cfg = VulkanBatchedConfig {
            prefill_tile_tokens: 16,
            ..VulkanBatchedConfig::default()
        };
        let mut seq_exec = VulkanBatchedGraphExecutor::open(&gpu_seq_model, cfg.clone())?;
        let mut bat_exec = VulkanBatchedGraphExecutor::open(&gpu_bat_model, cfg)?;

        for &tok in &prompt {
            cpu.ingest(tok)?;
            seq_exec.ingest_token(tok)?;
        }
        bat_exec.ingest_tokens_batched(&prompt)?;

        let seq_cp = seq_exec.checkpoint()?;
        let bat_cp = bat_exec.checkpoint()?;
        assert_eq!(
            bat_cp.identity_sha256()?,
            seq_cp.identity_sha256()?,
            "GDN checkpoint SHA-256 mismatch for ssm_head={ssm_head}, conv_k={conv_k}"
        );

        let cpu_logits = cpu.evaluate(5)?;
        let seq_logits = seq_exec.evaluate_token(5)?;
        let bat_logits = bat_exec.evaluate_token(5)?;
        assert_close(
            &bat_logits,
            &seq_logits,
            1e-5,
            &format!("bat vs seq ssm_head={ssm_head} conv_k={conv_k}"),
        );
        assert_close_rel(
            &bat_logits,
            &cpu_logits,
            5e-3,
            5e-3,
            &format!("bat vs cpu ssm_head={ssm_head} conv_k={conv_k}"),
        );
    }
    Ok(())
}

fn assert_bits_eq(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (idx, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "{label}: idx {idx} bit mismatch ({a} vs {e})"
        );
    }
}

fn repack_test_32b_coalesced(raw: &[u8], cols: usize, rows: usize, block_cols: usize) -> Vec<u32> {
    let blocks_per_row = cols / block_cols;
    let row_bytes = blocks_per_row * 34;
    let row_groups = rows.div_ceil(64);
    let mut dst = vec![0_u32; row_groups * blocks_per_row * 544];
    for rg in 0..row_groups {
        let r_base = rg * 64;
        for blk in 0..blocks_per_row {
            let tile_off = (rg * blocks_per_row + blk) * 544;
            let blk_off = blk * 34;
            for pair in 0..32 {
                let r_ev = r_base + pair * 2;
                let r_od = r_ev + 1;
                let s_ev = if r_ev < rows {
                    let b = &raw[r_ev * row_bytes + blk_off..];
                    u16::from_le_bytes([b[0], b[1]])
                } else {
                    0
                };
                let s_od = if r_od < rows {
                    let b = &raw[r_od * row_bytes + blk_off..];
                    u16::from_le_bytes([b[0], b[1]])
                } else {
                    0
                };
                dst[tile_off + pair] = u32::from(s_ev) | (u32::from(s_od) << 16);
            }
            for lane in 0..64 {
                let r = r_base + lane;
                if r >= rows {
                    continue;
                }
                let block = &raw[r * row_bytes + blk_off..][..34];
                for word_j in 0..8 {
                    let c = 2 + word_j * 4;
                    dst[tile_off + 32 + word_j * 64 + lane] =
                        u32::from_le_bytes([block[c], block[c + 1], block[c + 2], block[c + 3]]);
                }
            }
        }
    }
    dst
}

fn repack_test_ptq1_coalesced(raw: &[u8], cols: usize, rows: usize) -> Vec<u32> {
    let blocks_per_row = cols / 128;
    let row_bytes = blocks_per_row * 28;
    let row_groups = rows.div_ceil(64);
    let mut dst = vec![0_u32; row_groups * blocks_per_row * 448];
    for rg in 0..row_groups {
        let r_base = rg * 64;
        for blk in 0..blocks_per_row {
            let tile_off = (rg * blocks_per_row + blk) * 448;
            let blk_off = blk * 28;
            for lane in 0..64 {
                let r = r_base + lane;
                if r >= rows {
                    continue;
                }
                let block = &raw[r * row_bytes + blk_off..][..28];
                for (w_idx, chunk) in block.as_chunks::<4>().0.iter().enumerate() {
                    dst[tile_off + w_idx * 64 + lane] = u32::from_le_bytes(*chunk);
                }
            }
        }
    }
    dst
}

fn build_q8k_4plane_words(cols: usize, tokens: usize) -> (Vec<Q8KBlock>, Vec<u32>) {
    let q8k_per_tok = cols / 256;
    let mut blocks = Vec::with_capacity(tokens * q8k_per_tok);
    let mut words = Vec::with_capacity(tokens * q8k_per_tok * 68);
    for t in 0..tokens {
        for b in 0..q8k_per_tok {
            let mut values = [0_i8; 256];
            let mut sums = [0_i16; 16];
            for (i, slot) in values.iter_mut().enumerate() {
                let v = if i == 0 && (t + b).is_multiple_of(2) {
                    i8::MIN
                } else if i == 255 {
                    i8::MAX
                } else {
                    let raw = i32::try_from((i * 11 + t * 17 + b * 5) % 197).unwrap_or(0) - 98;
                    i8::try_from(raw).unwrap_or(0)
                };
                *slot = v;
                sums[i / 16] += i16::from(v);
            }
            let sign = if (t + b).is_multiple_of(2) { 1.0 } else { -1.0 };
            let scale =
                sign * (0.015_625_f32 * f32::from(u8::try_from((t + b) % 7 + 1).unwrap_or(1)));
            let blk = Q8KBlock {
                scale,
                values,
                sums,
            };
            for half in 0..2 {
                let hvals = &blk.values[half * 128..(half + 1) * 128];
                let hsum: i32 = blk.sums[half * 8..(half + 1) * 8]
                    .iter()
                    .map(|&s| i32::from(s))
                    .sum();
                words.push(blk.scale.to_bits());
                words.push((64 * hsum).cast_unsigned());
                for chunk4 in hvals.as_chunks::<4>().0 {
                    words.push(u32::from_le_bytes([
                        chunk4[0].cast_unsigned(),
                        chunk4[1].cast_unsigned(),
                        chunk4[2].cast_unsigned(),
                        chunk4[3].cast_unsigned(),
                    ]));
                }
            }
            blocks.push(blk);
        }
    }
    (blocks, words)
}

fn build_q8_0_words(cols: usize, tokens: usize) -> Vec<u32> {
    let blocks_per_tok = cols / 32;
    let mut words = Vec::with_capacity(tokens * blocks_per_tok * 9);
    for t in 0..tokens {
        for b in 0..blocks_per_tok {
            let sign = if (t + b).is_multiple_of(2) { 1.0 } else { -1.0 };
            let scale =
                sign * (0.031_25_f32 * f32::from(u8::try_from((t + b) % 5 + 1).unwrap_or(1)));
            words.push(scale.to_bits());
            for j in 0..8 {
                let b0 =
                    i8::try_from(i32::try_from((t * 7 + b * 3 + j * 4) % 181).unwrap_or(0) - 90)
                        .unwrap_or(0);
                let b1 =
                    i8::try_from(i32::try_from((t * 5 + b * 9 + j * 3) % 181).unwrap_or(0) - 90)
                        .unwrap_or(0);
                let b2 =
                    i8::try_from(i32::try_from((t * 3 + b * 7 + j * 5) % 181).unwrap_or(0) - 90)
                        .unwrap_or(0);
                let b3 =
                    i8::try_from(i32::try_from((t * 11 + b * 2 + j * 6) % 181).unwrap_or(0) - 90)
                        .unwrap_or(0);
                words.push(u32::from_le_bytes([
                    b0.cast_unsigned(),
                    b1.cast_unsigned(),
                    b2.cast_unsigned(),
                    b3.cast_unsigned(),
                ]));
            }
        }
    }
    words
}

fn clone_model(model: &Model) -> Model {
    Model {
        metadata: model.metadata.clone(),
        tensors: model.tensors.clone(),
        identity: model.identity.clone(),
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Encodes and executes both unfused and fused Hadamard1024+quant paths in a single batch."
)]
fn verify_fused_hadamard_case(
    dev: &mut AdrenoVulkanDevice,
    base_rec: &CommandBatchRecorder,
    fused_pipe: bonsai_adreno_vulkan::device::PipelineId,
    unfused_pipe: bonsai_adreno_vulkan::device::PipelineId,
    src_buf: BufferId,
    sign_buf: BufferId,
    rot_tmp: BufferId,
    fused_out: BufferId,
    unfused_out: BufferId,
    unfused_push: [u32; 2],
    unfused_grid: [u32; 3],
    fused_push: &[u32; 7],
    tokens: u32,
    byte_len: usize,
    label: &str,
) -> TestResult {
    let mut rec = base_rec.clone();
    rec.push_barrier(
        unfused_pipe,
        &[BufferSpan::whole(rot_tmp), BufferSpan::whole(unfused_out)],
        &unfused_push,
        unfused_grid,
    );
    rec.push_barrier(
        fused_pipe,
        &[
            BufferSpan::whole(src_buf),
            BufferSpan::whole(sign_buf),
            BufferSpan::whole(fused_out),
        ],
        fused_push,
        [2, tokens, 1],
    );
    dev.submit_compute_batch(&rec)?;

    let mut f_bytes = vec![0_u8; byte_len];
    let mut u_bytes = vec![0_u8; byte_len];
    dev.read_buffer_bytes(fused_out, &mut f_bytes)?;
    dev.read_buffer_bytes(unfused_out, &mut u_bytes)?;
    assert_eq!(f_bytes, u_bytes, "{label}");
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "Encodes unfused PermuteSSM + MulSign + Hadamard1024 chain for token batch parity."
)]
fn build_basis_unfused_batch(
    p_perm: bonsai_adreno_vulkan::device::PipelineId,
    p_mul: bonsai_adreno_vulkan::device::PipelineId,
    p_had: bonsai_adreno_vulkan::device::PipelineId,
    src_buf: BufferId,
    sign_buf: BufferId,
    perm_tmp: BufferId,
    sign_tmp: BufferId,
    rot_tmp: BufferId,
    cols_u32: u32,
    norm_bits: u32,
    use_sign: u32,
    use_perm: u32,
    kh: u32,
    rep: u32,
    hd: u32,
    tokens: u32,
) -> CommandBatchRecorder {
    let mut base_rec = CommandBatchRecorder::new();
    let mut cur = src_buf;
    if use_perm == 1 {
        base_rec.push_barrier(
            p_perm,
            &[BufferSpan::whole(cur), BufferSpan::whole(perm_tmp)],
            &[kh, rep, hd, tokens],
            [cols_u32.div_ceil(64), tokens, 1],
        );
        cur = perm_tmp;
    }
    if use_sign == 1 {
        base_rec.push_barrier(
            p_mul,
            &[
                BufferSpan::whole(cur),
                BufferSpan::whole(sign_buf),
                BufferSpan::whole(sign_tmp),
            ],
            &[cols_u32, tokens],
            [cols_u32.div_ceil(64), tokens, 1],
        );
        cur = sign_tmp;
    }
    base_rec.push_barrier(
        p_had,
        &[BufferSpan::whole(cur), BufferSpan::whole(rot_tmp)],
        &[1024, norm_bits, 2, tokens],
        [2, tokens, 1],
    );
    base_rec
}

#[test]
#[ignore = "Requires physical Qualcomm Adreno Vulkan Windows ARM64 hardware"]
#[expect(
    clippy::too_many_lines,
    reason = "Verifies Q8K and Q8_0 fused Hadamard1024+quant across signs, permutation, and token counts."
)]
fn live_fused_hadamard1024_quant_matches_unfused_zero_ulp() -> TestResult {
    let mut dev = AdrenoVulkanDevice::open()?;
    let cols = 2048_usize;
    let cols_u32 = 2048_u32;
    let max_tokens = 17_usize;
    let (kh, rep, hd) = (2_u32, 2_u32, 512_u32);
    let norm_bits = (1.0_f32 / 32.0_f32).to_bits();

    let src_vals: Vec<f32> = (0..max_tokens * cols)
        .map(|i| {
            let phase = i32::try_from(i % 257).unwrap_or(0) - 128;
            f32::from(i16::try_from(phase).unwrap_or(0)) * 0.013_75_f32
        })
        .collect();
    let sign_vals: Vec<f32> = (0..cols)
        .map(|i| {
            if (i * 7 + 3) % 5 < 2 {
                -1.0_f32
            } else {
                1.0_f32
            }
        })
        .collect();

    let f32_bytes = u64::try_from(max_tokens * cols * 4)?;
    let src_buf = dev.create_storage_buffer(f32_bytes)?;
    let sign_buf = dev.create_storage_buffer(u64::try_from(cols * 4)?)?;
    let perm_tmp = dev.create_storage_buffer(f32_bytes)?;
    let sign_tmp = dev.create_storage_buffer(f32_bytes)?;
    let rot_tmp = dev.create_storage_buffer(f32_bytes)?;
    let fused_q8k =
        dev.create_storage_buffer(u64::try_from(max_tokens * (cols / 128) * 34 * 4)?)?;
    let unfused_q8k =
        dev.create_storage_buffer(u64::try_from(max_tokens * (cols / 128) * 34 * 4)?)?;
    let fused_q8_0 = dev.create_storage_buffer(u64::try_from(max_tokens * (cols / 32) * 9 * 4)?)?;
    let unfused_q8_0 =
        dev.create_storage_buffer(u64::try_from(max_tokens * (cols / 32) * 9 * 4)?)?;

    dev.write_buffer_f32(src_buf, &src_vals)?;
    dev.write_buffer_f32(sign_buf, &sign_vals)?;
    let mut initial_src_bytes = vec![0_u8; src_vals.len() * 4];
    dev.read_buffer_bytes(src_buf, &mut initial_src_bytes)?;

    let p_perm = dev.create_compute_pipeline(&spirv::permute_ssm_heads_batched_f32(), 2, 4)?;
    let p_mul = dev.create_compute_pipeline(&spirv::mul_f32_elemwise_batched(), 3, 2)?;
    let p_had = dev.create_compute_pipeline(&spirv::ffn_hadamard_batched_f32(), 2, 4)?;
    let q8k_variant_pipes = [false, true].map(|pinned| {
        (
            pinned,
            dev.create_compute_pipeline(&spirv::quantize_f32_to_q8k_4plane(pinned), 2, 2),
            dev.create_compute_pipeline(&spirv::fused_basis_hadamard1024_q8k_4plane(pinned), 3, 7),
        )
    });
    let q8_0_variant_pipes = [(false, false), (true, false), (true, true)].map(|(pin, nat)| {
        (
            pin,
            nat,
            dev.create_compute_pipeline(&spirv::quantize_f32_to_q8_0_batched(pin, nat), 2, 2),
            dev.create_compute_pipeline(&spirv::fused_basis_hadamard1024_q8_0(pin, nat), 3, 7),
        )
    });

    for (tokens, use_sign, use_perm) in [(1_u32, 0_u32, 0_u32), (8, 1, 0), (17, 1, 1)] {
        let base_rec = build_basis_unfused_batch(
            p_perm, p_mul, p_had, src_buf, sign_buf, perm_tmp, sign_tmp, rot_tmp, cols_u32,
            norm_bits, use_sign, use_perm, kh, rep, hd, tokens,
        );
        let fused_push = [cols_u32, norm_bits, use_sign, use_perm, kh, rep, hd];

        for (pinned, pipe_uq8k, pipe_fq8k) in &q8k_variant_pipes {
            let uq8k = (*pipe_uq8k).clone()?;
            let fq8k = (*pipe_fq8k).clone()?;
            let byte_len = usize::try_from(tokens)? * (cols / 128) * 34 * 4;
            verify_fused_hadamard_case(
                &mut dev,
                &base_rec,
                fq8k,
                uq8k,
                src_buf,
                sign_buf,
                rot_tmp,
                fused_q8k,
                unfused_q8k,
                [cols_u32, tokens],
                [cols_u32 / 256, tokens, 1],
                &fused_push,
                tokens,
                byte_len,
                &format!(
                    "fused Hadamard1024 Q8K T={tokens} s={use_sign} p={use_perm} pin={pinned}"
                ),
            )?;
        }

        for (pinned, native, pipe_uq8, pipe_fq8) in &q8_0_variant_pipes {
            let uq8 = (*pipe_uq8).clone()?;
            let fq8 = (*pipe_fq8).clone()?;
            let q8_blocks = cols_u32 / 32;
            let byte_len = usize::try_from(tokens)? * (cols / 32) * 9 * 4;
            verify_fused_hadamard_case(
                &mut dev,
                &base_rec,
                fq8,
                uq8,
                src_buf,
                sign_buf,
                rot_tmp,
                fused_q8_0,
                unfused_q8_0,
                [q8_blocks, tokens],
                [q8_blocks.div_ceil(64), tokens, 1],
                &fused_push,
                tokens,
                byte_len,
                &format!(
                    "fused Hadamard1024 Q8_0 T={tokens} s={use_sign} p={use_perm} pin={pinned} nat={native}"
                ),
            )?;
        }
    }

    let mut after_src_bytes = vec![0_u8; src_vals.len() * 4];
    dev.read_buffer_bytes(src_buf, &mut after_src_bytes)?;
    assert_eq!(
        after_src_bytes, initial_src_bytes,
        "fused Hadamard1024 shader must not mutate src_buf (protects LoRA adapters)"
    );
    Ok(())
}

fn verify_pq2_executor_tokens_zero_ulp(
    pq2_tensor: &Tensor,
    q8k_blocks: &[Q8KBlock],
    cols: usize,
    rows: usize,
    max_t: usize,
) -> TestResult {
    for path in [
        ProjectionKernelPath::IntegerDotPacked4x8,
        ProjectionKernelPath::PortableIntegerFallback,
    ] {
        let mut pq2_exec = match VulkanPq2Executor::open_with_path(path, 64 * 1024 * 1024) {
            Ok(e) => e,
            Err(bonsai_adreno_vulkan::Error::Unsupported)
                if path == ProjectionKernelPath::IntegerDotPacked4x8 =>
            {
                continue;
            }
            Err(e) => return Err(e.into()),
        };
        assert_eq!(pq2_exec.prewarm_tensors(&[pq2_tensor])?, 1);
        for fused in [false, true] {
            let mut single_ref = Vec::with_capacity(max_t * rows);
            let q8k_per_tok = cols / 256;
            for t in 0..max_t {
                let mut single = Vec::new();
                pq2_exec.project(
                    pq2_tensor,
                    &q8k_blocks[t * q8k_per_tok..(t + 1) * q8k_per_tok],
                    fused,
                    &mut single,
                )?;
                single_ref.extend_from_slice(&single);
            }
            for &t in &[1_usize, 3, 8, 15, 16, 17] {
                let mut out = Vec::new();
                pq2_exec.project_pq2_q8k_tokens(
                    pq2_tensor,
                    &q8k_blocks[..t * q8k_per_tok],
                    t,
                    fused,
                    &mut out,
                )?;
                assert_bits_eq(
                    &out,
                    &single_ref[..t * rows],
                    &format!("{path:?} pq2_q8k_tokens T={t} fused={fused}"),
                );
            }
        }
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "Tests coalesced 16-tok shared-memory PQ2 and Q8_0 variants against 1-tok GEMV at 0 ULP."
)]
fn verify_coalesced_16tok_and_ptq1_for_mode(
    dev: &mut AdrenoVulkanDevice,
    use_sdot: bool,
    cols: usize,
    rows: usize,
    rows_u32: u32,
    b_pq2: BufferId,
    b_q8_0: BufferId,
    b_ptq1: BufferId,
    b_act_k: BufferId,
    b_act_8: BufferId,
    b_res: BufferId,
    b_out_ref: BufferId,
    b_out_4: BufferId,
    b_out_16: BufferId,
) -> TestResult {
    let row_tiles = rows_u32.div_ceil(64);
    let blk_128 = u32::try_from(cols / 128)?;
    let blk_32 = u32::try_from(cols / 32)?;
    let act_w_k = blk_128 * 34;
    let act_w_8 = blk_32 * 9;

    for fused in [false, true] {
        let p_ref = dev.create_compute_pipeline(
            &spirv::pq2_q8k_gemv_coalesced(use_sdot, fused, true),
            4,
            4,
        )?;
        let p_16 = dev.create_compute_pipeline(
            &spirv::pq2_q8k_gemm_16tok_coalesced(use_sdot, fused, true),
            4,
            4,
        )?;
        for &t in &[1_u32, 3, 8, 15, 16, 17] {
            let count = usize::try_from(t)? * rows;
            let mut ref_out = vec![0.0_f32; count];
            let mut g16_out = vec![0.0_f32; count];
            dev.submit_compute_stages(&[
                ComputeStage {
                    pipeline: p_ref,
                    buffers: &[b_pq2, b_act_k, b_out_ref, b_res],
                    push_u32s: &[rows_u32, blk_128, t, act_w_k],
                    workgroups: [row_tiles, t, 1],
                },
                ComputeStage {
                    pipeline: p_16,
                    buffers: &[b_pq2, b_act_k, b_out_16, b_res],
                    push_u32s: &[rows_u32, blk_128, t, act_w_k],
                    workgroups: [row_tiles, t.div_ceil(16), 1],
                },
            ])?;
            dev.read_buffer_f32(b_out_ref, &mut ref_out)?;
            dev.read_buffer_f32(b_out_16, &mut g16_out)?;
            assert_bits_eq(
                &g16_out,
                &ref_out,
                &format!("pq2_q8k 16tok res sdot={use_sdot} fused={fused} T={t}"),
            );
        }
    }

    let p_q8_ref = dev.create_compute_pipeline(&spirv::q8_0_q8_0_gemv_coalesced(use_sdot), 3, 4)?;
    let p_q8_16 =
        dev.create_compute_pipeline(&spirv::q8_0_q8_0_gemm_16tok_coalesced(use_sdot), 3, 4)?;
    for &t in &[1_u32, 3, 8, 15, 16, 17] {
        let count = usize::try_from(t)? * rows;
        let mut ref_out = vec![0.0_f32; count];
        let mut g16_out = vec![0.0_f32; count];
        dev.submit_compute_stages(&[
            ComputeStage {
                pipeline: p_q8_ref,
                buffers: &[b_q8_0, b_act_8, b_out_ref],
                push_u32s: &[rows_u32, blk_32, t, act_w_8],
                workgroups: [row_tiles, t, 1],
            },
            ComputeStage {
                pipeline: p_q8_16,
                buffers: &[b_q8_0, b_act_8, b_out_16],
                push_u32s: &[rows_u32, blk_32, t, act_w_8],
                workgroups: [row_tiles, t.div_ceil(16), 1],
            },
        ])?;
        dev.read_buffer_f32(b_out_ref, &mut ref_out)?;
        dev.read_buffer_f32(b_out_16, &mut g16_out)?;
        assert_bits_eq(
            &g16_out,
            &ref_out,
            &format!("q8_0_q8_0 16tok sdot={use_sdot} T={t}"),
        );
    }

    verify_ptq1_streaming_4_and_16tok(
        dev, use_sdot, rows, rows_u32, blk_128, act_w_8, row_tiles, b_ptq1, b_act_8, b_out_ref,
        b_out_4, b_out_16,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "Tests streaming 4-tok and 16-tok PTQ1 x Q8_0 kernels against 1-tok GEMV reference at 0 ULP."
)]
fn verify_ptq1_streaming_4_and_16tok(
    dev: &mut AdrenoVulkanDevice,
    use_sdot: bool,
    rows: usize,
    rows_u32: u32,
    blk_128: u32,
    act_w_8: u32,
    row_tiles: u32,
    b_ptq1: BufferId,
    b_act_8: BufferId,
    b_out_ref: BufferId,
    b_out_4: BufferId,
    b_out_16: BufferId,
) -> TestResult {
    for pinned in [false, true] {
        let p_ptq_ref =
            dev.create_compute_pipeline(&spirv::ptq1_q8_0_gemv_coalesced(use_sdot, pinned), 3, 4)?;
        let p_ptq_4 = dev.create_compute_pipeline(
            &spirv::ptq1_q8_0_gemm_4tok_coalesced(use_sdot, pinned),
            3,
            4,
        )?;
        let p_ptq_16 = dev.create_compute_pipeline(
            &spirv::ptq1_q8_0_gemm_16tok_coalesced(use_sdot, pinned),
            3,
            4,
        )?;
        for &t in &[1_u32, 3, 8, 15, 16, 17] {
            let count = usize::try_from(t)? * rows;
            let mut ref_out = vec![0.0_f32; count];
            let mut g4_out = vec![0.0_f32; count];
            let mut g16_out = vec![0.0_f32; count];
            dev.submit_compute_stages(&[
                ComputeStage {
                    pipeline: p_ptq_ref,
                    buffers: &[b_ptq1, b_act_8, b_out_ref],
                    push_u32s: &[rows_u32, blk_128, t, act_w_8],
                    workgroups: [row_tiles, t, 1],
                },
                ComputeStage {
                    pipeline: p_ptq_4,
                    buffers: &[b_ptq1, b_act_8, b_out_4],
                    push_u32s: &[rows_u32, blk_128, t, act_w_8],
                    workgroups: [row_tiles, t.div_ceil(4), 1],
                },
                ComputeStage {
                    pipeline: p_ptq_16,
                    buffers: &[b_ptq1, b_act_8, b_out_16],
                    push_u32s: &[rows_u32, blk_128, t, act_w_8],
                    workgroups: [row_tiles, t.div_ceil(16), 1],
                },
            ])?;
            dev.read_buffer_f32(b_out_ref, &mut ref_out)?;
            dev.read_buffer_f32(b_out_4, &mut g4_out)?;
            dev.read_buffer_f32(b_out_16, &mut g16_out)?;
            assert_bits_eq(
                &g4_out,
                &ref_out,
                &format!("ptq1 4tok streaming sdot={use_sdot} pinned={pinned} T={t}"),
            );
            assert_bits_eq(
                &g16_out,
                &ref_out,
                &format!("ptq1 16tok streaming sdot={use_sdot} pinned={pinned} T={t}"),
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "Requires physical Qualcomm Adreno Vulkan Windows ARM64 hardware"]
fn live_16tok_shared_gemm_and_streaming_ptq1_zero_ulp_parity() -> TestResult {
    let cols = 512_usize;
    let rows = 131_usize;
    let rows_u32 = 131_u32;
    let max_t = 17_usize;
    let pq2_tensor = build_pq2_tensor("gemm16.pq2", cols, rows, 3);
    let q8_0_tensor = build_q8_0_tensor("gemm16.q8_0", cols, rows, 5);
    let ptq1_tensor = build_ptq1_tensor("gemm16.ptq1", cols, rows, 7)?;
    let (q8k_blocks, q8k_words) = build_q8k_4plane_words(cols, max_t);
    let q8_0_words = build_q8_0_words(cols, max_t);
    let res_vals = fill_pattern(max_t * rows, 0.05, 19);

    verify_pq2_executor_tokens_zero_ulp(&pq2_tensor, &q8k_blocks, cols, rows, max_t)?;

    let mut dev = AdrenoVulkanDevice::open()?;
    let sdot_supported = dev.capabilities().supports_integer_dot_4x8();
    let pq2_coal = repack_test_32b_coalesced(&pq2_tensor.data, cols, rows, 128);
    let q8_0_coal = repack_test_32b_coalesced(&q8_0_tensor.data, cols, rows, 32);
    let ptq1_coal = repack_test_ptq1_coalesced(&ptq1_tensor.data, cols, rows);

    let b_pq2 = dev.create_storage_buffer(u64::try_from(pq2_coal.len() * 4)?)?;
    let b_q8_0 = dev.create_storage_buffer(u64::try_from(q8_0_coal.len() * 4)?)?;
    let b_ptq1 = dev.create_storage_buffer(u64::try_from(ptq1_coal.len() * 4)?)?;
    let b_act_k = dev.create_storage_buffer(u64::try_from(q8k_words.len() * 4)?)?;
    let b_act_8 = dev.create_storage_buffer(u64::try_from(q8_0_words.len() * 4)?)?;
    let b_res = dev.create_storage_buffer(u64::try_from(res_vals.len() * 4)?)?;
    let b_out_ref = dev.create_storage_buffer(u64::try_from(max_t * rows * 4)?)?;
    let b_out_4 = dev.create_storage_buffer(u64::try_from(max_t * rows * 4)?)?;
    let b_out_16 = dev.create_storage_buffer(u64::try_from(max_t * rows * 4)?)?;

    dev.write_buffer_u32(b_pq2, &pq2_coal)?;
    dev.write_buffer_u32(b_q8_0, &q8_0_coal)?;
    dev.write_buffer_u32(b_ptq1, &ptq1_coal)?;
    dev.write_buffer_u32(b_act_k, &q8k_words)?;
    dev.write_buffer_u32(b_act_8, &q8_0_words)?;
    dev.write_buffer_f32(b_res, &res_vals)?;

    let sdot_modes = if sdot_supported {
        vec![false, true]
    } else {
        vec![false]
    };
    for use_sdot in sdot_modes {
        verify_coalesced_16tok_and_ptq1_for_mode(
            &mut dev, use_sdot, cols, rows, rows_u32, b_pq2, b_q8_0, b_ptq1, b_act_k, b_act_8,
            b_res, b_out_ref, b_out_4, b_out_16,
        )?;
    }
    Ok(())
}

fn synthetic_hadamard1024_hybrid_model(identity: &str) -> ModelResult {
    let embd = 1024_usize;
    let ffn = 1024_usize;
    let mut model = synthetic_hybrid_qwen35_model_with_layout(identity, embd, ffn, 64)?;
    let signs: Vec<i32> = (0..embd)
        .map(|i| if (i * 5 + 1) % 3 == 0 { -1 } else { 1 })
        .collect();
    let mut rotated_weights: Vec<String> = Vec::new();
    for layer in 0..3 {
        let p = format!("blk.{layer}.");
        for suffix in ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"] {
            rotated_weights.push(format!("{p}{suffix}"));
        }
        if layer < 2 {
            rotated_weights.push(format!("{p}attn_qkv.weight"));
            rotated_weights.push(format!("{p}attn_gate.weight"));
        } else {
            for suffix in ["attn_q.weight", "attn_k.weight", "attn_v.weight"] {
                rotated_weights.push(format!("{p}{suffix}"));
            }
        }
    }
    model.metadata.insert(
        "prism.hadamard.version".to_owned(),
        MetadataValue::from(1_u32),
    );
    model.metadata.insert(
        "prism.hadamard.block_size".to_owned(),
        MetadataValue::from(1024_u32),
    );
    model.metadata.insert(
        "prism.hadamard.transform".to_owned(),
        MetadataValue::from("normalized-sylvester-walsh-hadamard"),
    );
    model.metadata.insert(
        "prism.hadamard.axis".to_owned(),
        MetadataValue::from("input-last-dimension"),
    );
    model.metadata.insert(
        "prism.hadamard.weight_names".to_owned(),
        MetadataValue::from(rotated_weights),
    );
    model.metadata.insert(
        "prism.hadamard.sign_mode".to_owned(),
        MetadataValue::from("explicit"),
    );
    model.metadata.insert(
        "prism.hadamard.sign_widths".to_owned(),
        MetadataValue::from(vec![1024_u32]),
    );
    model.metadata.insert(
        "prism.hadamard.sign_values".to_owned(),
        MetadataValue::from(signs),
    );
    Ok(model)
}

#[test]
#[ignore = "Requires physical Qualcomm Adreno Vulkan Windows ARM64 hardware"]
fn live_prefill_16tok_fused_h1024_and_decode_gdn_hybrid_parity() -> TestResult {
    let prompt_all: [u32; 17] = [0, 3, 6, 9, 12, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 15, 2];
    let base_model = synthetic_hadamard1024_hybrid_model("h1024-hybrid-16tok")?;
    let q8_model = clone_with_q8_0_weights(&base_model, 29);

    for model in [&base_model, &q8_model] {
        let mut cpu = Qwen35::with_adapters_and_kv(
            clone_model(model),
            64,
            64 * 1024 * 1024,
            MatvecBackend::Auto,
            None,
            None,
            KvCachePolicy::F16,
        )?;
        let mut seq_exec = VulkanBatchedGraphExecutor::open(
            model,
            VulkanBatchedConfig {
                prefill_tile_tokens: 16,
                ..VulkanBatchedConfig::default()
            },
        )?;
        let mut bat_exec = VulkanBatchedGraphExecutor::open(
            model,
            VulkanBatchedConfig {
                prefill_tile_tokens: 16,
                ..VulkanBatchedConfig::default()
            },
        )?;

        for &len in &[8_usize, 15, 16, 17] {
            let prompt = &prompt_all[..len];
            cpu.reset()?;
            seq_exec.reset()?;
            bat_exec.reset()?;

            for &tok in prompt {
                cpu.ingest(tok)?;
                seq_exec.ingest_token(tok)?;
            }
            bat_exec.ingest_tokens_batched(prompt)?;
            assert_eq!(
                bat_exec.checkpoint()?.identity_sha256()?,
                seq_exec.checkpoint()?.identity_sha256()?,
                "prefill T={len} checkpoint mismatch"
            );

            // Exercise register-resident single-token GDN decode + evaluate_finite_into after prefill.
            for &dec_tok in &[4_u32, 9] {
                cpu.ingest(dec_tok)?;
                seq_exec.ingest_token(dec_tok)?;
                bat_exec.ingest_token(dec_tok)?;
            }
            assert_eq!(
                bat_exec.checkpoint()?.identity_sha256()?,
                seq_exec.checkpoint()?.identity_sha256()?,
                "post-decode T={len} checkpoint mismatch"
            );

            let cpu_logits = cpu.evaluate(11)?;
            let seq_logits = seq_exec.evaluate_token(11)?;
            let mut bat_out = Vec::new();
            let finite = Decoder::evaluate_finite_into(&mut bat_exec, 11, &mut bat_out)?;
            assert_close(
                finite.as_slice(),
                &seq_logits,
                1e-4,
                &format!("bat vs seq h1024 hybrid T={len}"),
            );
            assert_close_rel(
                finite.as_slice(),
                &cpu_logits,
                5e-3,
                5e-3,
                &format!("bat vs cpu h1024 hybrid T={len}"),
            );
        }
    }
    Ok(())
}
