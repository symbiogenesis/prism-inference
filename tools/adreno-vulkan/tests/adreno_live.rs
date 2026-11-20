//! Live Qualcomm Adreno Vulkan compute parity, Qwen3.5 integration, Composite routing, and
//! bounded timing tests for `#39` (`PQ2_0`/`PTQ1_0`) and `#40` (fused vs 3-stage decomposed attention).
#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::io::Write as _;
use std::sync::Arc;
use std::time::Instant;

use bonsai_adreno_vulkan::{
    AdrenoAttentionPath, AttentionDescriptor, AttentionStrategy, Error, ProjectionKernelPath,
    VulkanAttentionExecutor, VulkanPq2Executor, attention_cache_key, cpu_attention_reference,
};
use bonsai_gguf::{MetadataArray, MetadataValue};
use bonsai_hexagon_runtime::{
    AcceleratorTarget, CompositeProjectionExecutor, plan_x1_v73_three_processor_placement,
};
use bonsai_prism_runtime::{
    Decoder as _,
    graph::Qwen35,
    kernels::{
        MatvecBackend, MatvecScratch, Pq2ProjectionExecutor, Q8Block, dot_ptq1_q8k, hadamard, silu,
    },
    model::{Model, Tensor},
};
use bonsai_ternary::{Q8KBlock, decode_ptq_trits, float_to_half, half_to_float, pack_ptq};

type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;
type TensorResult = Result<Tensor, Box<dyn std::error::Error + Send + Sync>>;

fn f16_scale(value: f32) -> [u8; 2] {
    float_to_half(value).to_le_bytes()
}

fn build_pq2_tensor(name: &str, cols: usize, rows: usize, include_plus_two: bool) -> Tensor {
    let blocks_per_row = cols / 128;
    let mut data = Vec::with_capacity(rows * blocks_per_row * 34);
    for r in 0..rows {
        for b in 0..blocks_per_row {
            let scale_val = 0.03125_f32 * f32::from(u8::try_from((r + b) % 7 + 1).unwrap_or(1));
            data.extend_from_slice(&f16_scale(scale_val));
            for j in 0..32 {
                let code_word = if include_plus_two && ((r + b + j) % 5 == 0) {
                    0xFF_u8 // four +2 symbols (0b11 = 3)
                } else {
                    let s0 = u8::try_from((r + j) % 3).unwrap_or(0);
                    let s1 = u8::try_from((b + j + 1) % 3).unwrap_or(0);
                    let s2 = u8::try_from((r + b + j) % 3).unwrap_or(0);
                    let s3 = u8::try_from((j + 2) % 3).unwrap_or(0);
                    s0 | (s1 << 2) | (s2 << 4) | (s3 << 6)
                };
                data.push(code_word);
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

fn build_ptq1_tensor(name: &str, cols: usize, rows: usize) -> TensorResult {
    let blocks_per_row = cols / 128;
    let mut data = Vec::with_capacity(rows * blocks_per_row * 28);
    for r in 0..rows {
        for b in 0..blocks_per_row {
            let mut src = [0_u8; 34];
            let scale_val = 0.0625_f32 * f32::from(u8::try_from((r + b) % 5 + 1).unwrap_or(1));
            src[..2].copy_from_slice(&f16_scale(scale_val));
            for (j, byte) in src[2..].iter_mut().enumerate() {
                let s0 = u8::try_from((r + j) % 3).unwrap_or(0);
                let s1 = u8::try_from((b + j + 1) % 3).unwrap_or(0);
                let s2 = u8::try_from((r + j + 2) % 3).unwrap_or(0);
                let s3 = u8::try_from((b + j) % 3).unwrap_or(0);
                *byte = s0 | (s1 << 2) | (s2 << 4) | (s3 << 6);
            }
            let ptq = pack_ptq(&src)?;
            data.extend_from_slice(&ptq);
        }
    }
    Ok(Tensor {
        name: name.to_owned(),
        dimensions: vec![cols, rows],
        kind: 143,
        data: Arc::<[u8]>::from(data).into(),
    })
}

fn build_q8k_activations(cols: usize, tokens: usize, seed_offset: usize) -> Vec<Q8KBlock> {
    let blocks_per_token = cols.div_ceil(256);
    let mut result = Vec::with_capacity(tokens * blocks_per_token);
    for t in 0..tokens {
        for b in 0..blocks_per_token {
            let mut values = [0_i8; 256];
            let mut sums = [0_i16; 16];
            for (idx, slot) in values.iter_mut().enumerate() {
                let v = if idx == 0 && (t + b) % 2 == 0 {
                    i8::MIN
                } else if idx == 255 {
                    i8::MAX
                } else {
                    let raw = i32::try_from((idx * 7 + t * 13 + b * 3 + seed_offset) % 199)
                        .unwrap_or(0)
                        - 99;
                    i8::try_from(raw).unwrap_or(0)
                };
                *slot = v;
                sums[idx / 16] += i16::from(v);
            }
            let sign = if (t + b).is_multiple_of(2) {
                1.0_f32
            } else {
                -1.0_f32
            };
            let scale = sign * (0.0125_f32 * f32::from(u8::try_from((t + b) % 9 + 1).unwrap_or(1)));
            result.push(Q8KBlock {
                scale,
                values,
                sums,
            });
        }
    }
    result
}

fn build_q8_activations(cols: usize, seed_offset: usize) -> Vec<Q8Block> {
    let blocks = cols / 32;
    let mut result = Vec::with_capacity(blocks);
    for b in 0..blocks {
        let mut values = [0_i8; 32];
        let mut int_sum = 0_i32;
        for (idx, slot) in values.iter_mut().enumerate() {
            let v = if idx == 0 && b % 2 == 0 {
                i8::MIN
            } else {
                let raw = i32::try_from((idx * 11 + b * 5 + seed_offset) % 181).unwrap_or(0) - 90;
                i8::try_from(raw).unwrap_or(0)
            };
            *slot = v;
            int_sum += i32::from(v);
        }
        let scale = half_to_float(float_to_half(
            0.015_f32 * f32::from(u8::try_from((b % 6) + 1).unwrap_or(1)),
        ));
        #[expect(
            clippy::as_conversions,
            clippy::cast_precision_loss,
            reason = "Bounded 32-element i8 sums are exact in binary32."
        )]
        let sum = (int_sum as f32) * scale;
        result.push(Q8Block { scale, sum, values });
    }
    result
}

#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Bounded 128-element integer dot sums are exact in binary32."
)]
const fn i32_to_f32_exact(val: i32) -> f32 {
    val as f32
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Matches exact Prism unfused vs fused reference accumulation."
)]
fn cpu_pq2_q8k_reference(tensor: &Tensor, activation: &[Q8KBlock], fused: bool) -> Vec<f32> {
    let cols = tensor.dimensions[0];
    let rows = tensor.dimensions[1];
    let blocks_per_row = cols / 128;
    let row_bytes = blocks_per_row * 34;
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let row_slice = &tensor.data[r * row_bytes..(r + 1) * row_bytes];
        let mut acc = 0.0_f32;
        for (blk_idx, block) in row_slice.as_chunks::<34>().0.iter().enumerate() {
            let w_scale = half_to_float(u16::from_le_bytes([block[0], block[1]]));
            let q8k = &activation[blk_idx / 2];
            let start = (blk_idx % 2) * 128;
            let mut sum = 0_i32;
            for idx in 0..128 {
                let sym = i32::from((block[2 + idx / 4] >> (2 * (idx % 4))) & 3) - 1;
                sum += sym * i32::from(q8k.values[start + idx]);
            }
            let d = w_scale * q8k.scale;
            let sum_f32 = i32_to_f32_exact(sum);
            acc = if fused {
                d.mul_add(sum_f32, acc)
            } else {
                acc + d * sum_f32
            };
        }
        out.push(acc);
    }
    out
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Matches exact Prism dot_q8_groups scalar reference."
)]
fn cpu_pq2_q8_0_reference(tensor: &Tensor, activation: &[Q8Block]) -> Vec<f32> {
    let cols = tensor.dimensions[0];
    let rows = tensor.dimensions[1];
    let blocks_per_row = cols / 128;
    let row_bytes = blocks_per_row * 34;
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let row_slice = &tensor.data[r * row_bytes..(r + 1) * row_bytes];
        let mut acc = 0.0_f32;
        for (blk_idx, block) in row_slice.as_chunks::<34>().0.iter().enumerate() {
            let w_scale = half_to_float(u16::from_le_bytes([block[0], block[1]]));
            let mut block_sum = 0.0_f32;
            for grp in 0..4 {
                let act = &activation[blk_idx * 4 + grp];
                let mut sum = 0_i32;
                for (local, &val) in act.values.iter().enumerate() {
                    let idx = grp * 32 + local;
                    let sym = i32::from((block[2 + idx / 4] >> (2 * (idx % 4))) & 3) - 1;
                    sum += sym * i32::from(val);
                }
                block_sum += act.scale * i32_to_f32_exact(sum);
            }
            acc += w_scale * block_sum;
        }
        out.push(acc);
    }
    out
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Matches exact Prism dot_q8_groups vs dot_q8_groups_pinned scalar reference."
)]
fn cpu_ptq1_q8_0_reference(tensor: &Tensor, activation: &[Q8Block], pinned: bool) -> Vec<f32> {
    let cols = tensor.dimensions[0];
    let rows = tensor.dimensions[1];
    let blocks_per_row = cols / 128;
    let row_bytes = blocks_per_row * 28;
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let row_slice = &tensor.data[r * row_bytes..(r + 1) * row_bytes];
        let mut acc = 0.0_f32;
        let mut lanes = [0.0_f32; 4];
        for (blk_idx, block) in row_slice.as_chunks::<28>().0.iter().enumerate() {
            let w_scale = half_to_float(u16::from_le_bytes([block[26], block[27]]));
            let trits = decode_ptq_trits(block);
            let mut block_sum = 0.0_f32;
            for grp in 0..4 {
                let act = &activation[blk_idx * 4 + grp];
                if pinned {
                    let d = w_scale * act.scale;
                    for (l, lane_slot) in lanes.iter_mut().enumerate() {
                        let base0 = grp * 32 + l * 4;
                        let base1 = base0 + 16;
                        let mut sum_l = 0_i32;
                        for off in 0..4 {
                            sum_l +=
                                i32::from(trits[base0 + off]) * i32::from(act.values[l * 4 + off]);
                            sum_l += i32::from(trits[base1 + off])
                                * i32::from(act.values[16 + l * 4 + off]);
                        }
                        *lane_slot = i32_to_f32_exact(sum_l).mul_add(d, *lane_slot);
                    }
                } else {
                    let mut sum = 0_i32;
                    for local in 0..32 {
                        sum += i32::from(trits[grp * 32 + local]) * i32::from(act.values[local]);
                    }
                    block_sum += act.scale * i32_to_f32_exact(sum);
                }
            }
            if !pinned {
                acc += w_scale * block_sum;
            }
        }
        if pinned {
            acc = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
        }
        out.push(acc);
    }
    out
}

fn assert_zero_ulp(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (idx, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "{label}: element {idx} diverged: gpu={a} (0x{:08x}) cpu={e} (0x{:08x})",
            a.to_bits(),
            e.to_bits()
        );
    }
}

#[test]
fn pq2_q8k_exact_zero_ulp_parity_sdot_and_portable() -> TestResult {
    let tensor_small = build_pq2_tensor("test.pq2.512x137", 512, 137, true);
    let act_1 = build_q8k_activations(512, 1, 1);
    let act_4 = build_q8k_activations(512, 4, 7);

    let tensor_swift = build_pq2_tensor("test.pq2.5120x17408", 5120, 17408, true);
    let act_swift = build_q8k_activations(5120, 1, 19);
    let expected_swift_unfused = cpu_pq2_q8k_reference(&tensor_swift, &act_swift, false);
    let expected_swift_fused = cpu_pq2_q8k_reference(&tensor_swift, &act_swift, true);
    let expected_small_unfused = cpu_pq2_q8k_reference(&tensor_small, &act_1, false);
    let expected_small_fused = cpu_pq2_q8k_reference(&tensor_small, &act_1, true);

    let mut fused_small_paths = Vec::with_capacity(2);
    let mut fused_swift_paths = Vec::with_capacity(2);

    for path in [
        ProjectionKernelPath::IntegerDotPacked4x8,
        ProjectionKernelPath::PortableIntegerFallback,
    ] {
        let mut exec = VulkanPq2Executor::open_with_path(path, 64 * 1024 * 1024)?;
        assert!(exec.identity().bytes().all(|b| b.is_ascii_graphic()));
        assert!(exec.identity().len() <= 256);

        let mut out = Vec::new();
        exec.project(&tensor_small, &act_1, false, &mut out)?;
        assert_zero_ulp(
            &out,
            &expected_small_unfused,
            &format!("{path:?} fused=false"),
        );

        // Vulkan GLSL.std.450 Fma inherits `a * b + c` precision and lowers to `mad.f32` on
        // Adreno X2-90 (observed 1 ULP vs CPU single-rounded `f32::mul_add`). Verify exact
        // bit-identity across OpSDot vs Portable paths plus bounded closeness vs CPU `mul_add`.
        exec.project(&tensor_small, &act_1, true, &mut out)?;
        assert_close(
            &out,
            &expected_small_fused,
            1e-4,
            &format!("{path:?} fused=true"),
        );
        fused_small_paths.push(out.clone());

        exec.project_pq2_q8k_tokens(&tensor_small, &act_4, 4, false, &mut out)?;
        let mut expected_multi = Vec::new();
        for tok in 0..4 {
            expected_multi.extend(cpu_pq2_q8k_reference(
                &tensor_small,
                &act_4[tok * 2..(tok + 1) * 2],
                false,
            ));
        }
        assert_zero_ulp(&out, &expected_multi, &format!("{path:?} tokens=4"));

        exec.project(&tensor_swift, &act_swift, false, &mut out)?;
        assert_zero_ulp(
            &out,
            &expected_swift_unfused,
            &format!("{path:?} 5120x17408 fused=false"),
        );
        assert!(exec.cached_weight_bytes() >= tensor_swift.data.len());
        let cached_before = exec.cached_weight_bytes();
        exec.project(&tensor_swift, &act_swift, true, &mut out)?;
        assert_eq!(exec.cached_weight_bytes(), cached_before);
        assert_close(
            &out,
            &expected_swift_fused,
            1e-4,
            &format!("{path:?} 5120x17408 fused=true"),
        );
        fused_swift_paths.push(out.clone());
        let released = exec.release_resident_weights();
        assert_eq!(released, cached_before);
        assert_eq!(exec.cached_weight_bytes(), 0);

        let bad_act = vec![
            Q8KBlock {
                scale: f32::NAN,
                values: [0; 256],
                sums: [0; 16],
            };
            2
        ];
        assert!(
            exec.project(&tensor_small, &bad_act, false, &mut out)
                .is_err()
        );
        assert_eq!(out.len(), 0);
    }
    assert_zero_ulp(
        &fused_small_paths[0],
        &fused_small_paths[1],
        "sdot vs portable fused=true small",
    );
    assert_zero_ulp(
        &fused_swift_paths[0],
        &fused_swift_paths[1],
        "sdot vs portable fused=true 5120x17408",
    );
    Ok(())
}

#[test]
fn pq2_q8_0_and_ptq1_exact_zero_ulp_parity_sdot_and_portable() -> TestResult {
    let pq2_odd = build_pq2_tensor("test.pq2.384x73", 384, 73, true);
    let q8_384 = build_q8_activations(384, 3);
    let ptq_odd = build_ptq1_tensor("test.ptq1.384x67", 384, 67)?;
    let kgroup_acts_384 = build_q8k_activations(384, 1, 11);
    let ptq_even = build_ptq1_tensor("test.ptq1.512x64", 512, 64)?;
    let q8_512 = build_q8_activations(512, 5);

    for path in [
        ProjectionKernelPath::IntegerDotPacked4x8,
        ProjectionKernelPath::PortableIntegerFallback,
    ] {
        let mut exec = VulkanPq2Executor::open_with_path(path, 16 * 1024 * 1024)?;
        let mut out = Vec::new();

        exec.project_q8(&pq2_odd, &q8_384, false, &mut out)?;
        assert_zero_ulp(
            &out,
            &cpu_pq2_q8_0_reference(&pq2_odd, &q8_384),
            &format!("{path:?} pq2_q8_0_odd"),
        );

        for pinned in [false, true] {
            exec.project_q8(&ptq_odd, &q8_384, pinned, &mut out)?;
            assert_zero_ulp(
                &out,
                &cpu_ptq1_q8_0_reference(&ptq_odd, &q8_384, pinned),
                &format!("{path:?} ptq1_384 pinned={pinned}"),
            );

            exec.project_q8(&ptq_even, &q8_512, pinned, &mut out)?;
            assert_zero_ulp(
                &out,
                &cpu_ptq1_q8_0_reference(&ptq_even, &q8_512, pinned),
                &format!("{path:?} ptq1_512 pinned={pinned}"),
            );
        }

        exec.project_ptq1_q8k(&ptq_odd, &kgroup_acts_384, &mut out)?;
        let row_bytes = 3 * 28;
        let expected_q8k: Vec<f32> = (0..67)
            .map(|r| {
                dot_ptq1_q8k(
                    &ptq_odd.data[r * row_bytes..(r + 1) * row_bytes],
                    &kgroup_acts_384,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        assert_zero_ulp(&out, &expected_q8k, &format!("{path:?} ptq1_q8k_odd"));
    }
    Ok(())
}

#[expect(
    clippy::too_many_lines,
    reason = "Constructs complete multi-tensor synthetic Qwen35 fixture cleanly in one place."
)]
fn synthetic_qwen35_model() -> Result<Model, Box<dyn std::error::Error + Send + Sync>> {
    let mut metadata = BTreeMap::new();
    let str_val = |s: &str| MetadataValue::String(s.to_owned());
    metadata.insert("general.architecture".to_owned(), str_val("qwen35"));
    metadata.insert("qwen35.block_count".to_owned(), MetadataValue::U32(1));
    metadata.insert("qwen35.context_length".to_owned(), MetadataValue::U32(16));
    metadata.insert(
        "qwen35.embedding_length".to_owned(),
        MetadataValue::U32(256),
    );
    metadata.insert(
        "qwen35.feed_forward_length".to_owned(),
        MetadataValue::U32(256),
    );
    metadata.insert(
        "qwen35.attention.head_count".to_owned(),
        MetadataValue::U32(2),
    );
    metadata.insert(
        "qwen35.attention.head_count_kv".to_owned(),
        MetadataValue::U32(1),
    );
    metadata.insert(
        "qwen35.attention.key_length".to_owned(),
        MetadataValue::U32(128),
    );
    metadata.insert(
        "qwen35.attention.value_length".to_owned(),
        MetadataValue::U32(128),
    );
    metadata.insert(
        "qwen35.attention.layer_norm_rms_epsilon".to_owned(),
        MetadataValue::F32(1e-6),
    );
    metadata.insert(
        "qwen35.rope.dimension_count".to_owned(),
        MetadataValue::U32(128),
    );
    metadata.insert(
        "qwen35.rope.dimension_sections".to_owned(),
        MetadataValue::Array(MetadataArray::I32(vec![64, 0, 0, 0])),
    );
    metadata.insert(
        "qwen35.rope.freq_base".to_owned(),
        MetadataValue::F32(10_000.0),
    );
    metadata.insert("qwen35.ssm.state_size".to_owned(), MetadataValue::U32(128));
    metadata.insert("qwen35.ssm.group_count".to_owned(), MetadataValue::U32(1));
    metadata.insert(
        "qwen35.ssm.time_step_rank".to_owned(),
        MetadataValue::U32(1),
    );
    metadata.insert("qwen35.ssm.inner_size".to_owned(), MetadataValue::U32(128));
    metadata.insert("qwen35.ssm.conv_kernel".to_owned(), MetadataValue::U32(2));
    metadata.insert(
        "qwen35.full_attention_interval".to_owned(),
        MetadataValue::U32(1),
    );
    metadata.insert(
        "qwen35.attention.recurrent_layers".to_owned(),
        MetadataValue::Array(MetadataArray::Bool(vec![false])),
    );

    let f32_tensor = |name: &str, dims: &[usize], vals: &[f32]| -> Tensor {
        Tensor {
            name: name.to_owned(),
            dimensions: dims.to_vec(),
            kind: 0,
            data: Arc::<[u8]>::from(
                vals.iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<u8>>(),
            )
            .into(),
        }
    };

    let mut tensors = BTreeMap::new();
    let mut embd = vec![0.0_f32; 2 * 256];
    for (i, slot) in embd.iter_mut().enumerate() {
        *slot = if i % 2 == 0 { 0.0625 } else { -0.03125 };
    }
    tensors.insert(
        "token_embd.weight".to_owned(),
        f32_tensor("token_embd.weight", &[256, 2], &embd),
    );
    tensors.insert(
        "output.weight".to_owned(),
        f32_tensor("output.weight", &[256, 2], &embd),
    );
    tensors.insert(
        "output_norm.weight".to_owned(),
        f32_tensor("output_norm.weight", &[256], &[1.0; 256]),
    );
    tensors.insert(
        "blk.0.attn_norm.weight".to_owned(),
        f32_tensor("blk.0.attn_norm.weight", &[256], &[1.0; 256]),
    );
    tensors.insert(
        "blk.0.post_attention_norm.weight".to_owned(),
        f32_tensor("blk.0.post_attention_norm.weight", &[256], &[1.0; 256]),
    );
    tensors.insert(
        "blk.0.attn_q_norm.weight".to_owned(),
        f32_tensor("blk.0.attn_q_norm.weight", &[128], &[1.0; 128]),
    );
    tensors.insert(
        "blk.0.attn_k_norm.weight".to_owned(),
        f32_tensor("blk.0.attn_k_norm.weight", &[128], &[1.0; 128]),
    );

    tensors.insert(
        "blk.0.attn_q.weight".to_owned(),
        build_ptq1_tensor("blk.0.attn_q.weight", 256, 512)?,
    );
    tensors.insert(
        "blk.0.attn_k.weight".to_owned(),
        build_ptq1_tensor("blk.0.attn_k.weight", 256, 128)?,
    );
    tensors.insert(
        "blk.0.attn_v.weight".to_owned(),
        build_ptq1_tensor("blk.0.attn_v.weight", 256, 128)?,
    );
    tensors.insert(
        "blk.0.attn_output.weight".to_owned(),
        build_pq2_tensor("blk.0.attn_output.weight", 256, 256, false),
    );
    tensors.insert(
        "blk.0.ffn_up.weight".to_owned(),
        build_pq2_tensor("blk.0.ffn_up.weight", 256, 256, true),
    );
    tensors.insert(
        "blk.0.ffn_gate.weight".to_owned(),
        build_pq2_tensor("blk.0.ffn_gate.weight", 256, 256, true),
    );
    tensors.insert(
        "blk.0.ffn_down.weight".to_owned(),
        build_pq2_tensor("blk.0.ffn_down.weight", 256, 256, false),
    );

    Ok(Model {
        metadata,
        tensors,
        identity: "adreno-vulkan-qwen35-fixture".to_owned(),
    })
}

#[test]
fn qwen35_with_vulkan_executor_matches_cpu_logits_checkpoint_and_rollback() -> TestResult {
    let cpu_model = synthetic_qwen35_model()?;
    let gpu_model = synthetic_qwen35_model()?;
    let tiled_model = synthetic_qwen35_model()?;

    let mut cpu = Qwen35::new(cpu_model, 8)?;
    let mut vulkan_exec = VulkanPq2Executor::open_default(32 * 1024 * 1024)?;
    let gpu_base = Qwen35::new(gpu_model, 8)?;
    let mut projected = gpu_base.with_projection_executor(&mut vulkan_exec)?;

    let cpu_l0 = cpu.evaluate(0)?;
    let gpu_l0 = projected.evaluate(0)?;
    assert_close(&gpu_l0, &cpu_l0, 1e-5, "qwen35 token 0");

    let cp = projected.checkpoint()?;
    let cpu_l1 = cpu.evaluate(1)?;
    let gpu_l1 = projected.evaluate(1)?;
    assert_close(&gpu_l1, &cpu_l1, 1e-5, "qwen35 token 1");

    projected.restore(&cp)?;
    let gpu_l1_replay = projected.evaluate(1)?;
    assert_zero_ulp(&gpu_l1_replay, &gpu_l1, "qwen35 rollback token 1");
    assert!(!projected.is_faulted());
    drop(projected);

    let tiled_base = Qwen35::new(tiled_model, 16)?;
    let tiled_scratch = tiled_base.tiled_prefill_scratch_bytes(8)?;
    let tiled_base = tiled_base.with_tiled_prefill(8, tiled_scratch)?;
    let mut tiled_proj = tiled_base.with_projection_executor(&mut vulkan_exec)?;
    let prompt = [0_u32, 1, 0, 1, 0, 1, 0, 1];
    tiled_proj.ingest_block(&prompt, &|| Ok(()))?;
    let mut cpu2 = Qwen35::new(synthetic_qwen35_model()?, 16)?;
    for &tok in &prompt {
        let _ = cpu2.evaluate(tok)?;
    }
    let cpu_l_next = cpu2.evaluate(0)?;
    let tiled_l_next = tiled_proj.evaluate(0)?;
    assert_close(
        &tiled_l_next,
        &cpu_l_next,
        1e-4,
        "qwen35 tiled prefill token 8",
    );
    Ok(())
}

#[test]
fn gpu_quantize_q8k_4plane_matches_cpu_bit_for_bit() -> TestResult {
    use bonsai_adreno_vulkan::{AdrenoVulkanDevice, spirv};
    let mut device = AdrenoVulkanDevice::open()?;
    let pipe = device.create_compute_pipeline(&spirv::quantize_f32_to_q8k_4plane(false), 2, 2)?;
    let cols = 512_usize;
    let mut act = fill_f32_pattern(cols, 0.125);
    act[0] = 3.5;
    act[4] = -3.5;
    act[32] = -3.5;

    let in_buf = device.create_storage_buffer(u64::try_from(cols * 4)?)?;
    device.write_buffer_f32(in_buf, &act)?;
    let words_out = (cols / 128) * 34;
    let out_buf = device.create_storage_buffer(u64::try_from(words_out * 4)?)?;
    device.submit_compute(
        pipe,
        &[in_buf, out_buf],
        &[u32::try_from(cols)?, 1],
        [u32::try_from(cols / 256)?, 1, 1],
    )?;

    let mut raw_f32 = vec![0.0_f32; words_out];
    device.read_buffer_f32(out_buf, &mut raw_f32)?;
    let gpu_u32: Vec<u32> = raw_f32.iter().map(|v| v.to_bits()).collect();

    #[expect(
        clippy::suboptimal_flops,
        reason = "Unpinned Q8K contract rounds product before adding magic."
    )]
    let cpu_q8k = |chunk: &[f32]| -> (u32, i32, Vec<u32>) {
        let mut maximum = 0.0_f32;
        for &v in chunk {
            if v.abs() > maximum.abs() {
                maximum = v;
            }
        }
        let iscale = -127.0_f32 / maximum;
        let scale = 1.0_f32 / iscale;
        let qs: Vec<i8> = chunk
            .iter()
            .map(|&v| {
                let r = (v * iscale) + 12_582_912.0_f32;
                let mant = i32::try_from(r.to_bits() & 0x007F_FFFF).unwrap_or(0) - 0x0040_0000;
                i8::try_from(mant.clamp(-127, 127)).unwrap_or(0)
            })
            .collect();
        let h0: i32 = qs[..128].iter().map(|&v| i32::from(v)).sum();
        let h1: i32 = qs[128..].iter().map(|&v| i32::from(v)).sum();
        let mut expected = Vec::new();
        for (h_idx, sub) in qs.as_chunks::<128>().0.iter().enumerate() {
            expected.push(scale.to_bits());
            expected.push((if h_idx == 0 { h0 } else { h1 }).cast_unsigned());
            for c16 in sub.as_chunks::<16>().0 {
                for d in 0..4 {
                    expected.push(u32::from_le_bytes([
                        c16[d].cast_unsigned(),
                        c16[d + 4].cast_unsigned(),
                        c16[d + 8].cast_unsigned(),
                        c16[d + 12].cast_unsigned(),
                    ]));
                }
            }
        }
        (scale.to_bits(), h0, expected)
    };
    let mut expected = Vec::new();
    for chunk in act.as_chunks::<256>().0 {
        expected.extend(cpu_q8k(chunk).2);
    }
    device.free_buffer(out_buf);
    device.free_buffer(in_buf);
    for (idx, (&g, &e)) in gpu_u32.iter().zip(&expected).enumerate() {
        assert_eq!(
            g,
            e,
            "q8k_4plane word {idx} (blk={}, w={}): gpu=0x{g:08x} cpu=0x{e:08x}",
            idx / 34,
            idx % 34
        );
    }
    Ok(())
}

#[test]
fn pq2_zero_f16_scale_block_accepted_and_matches_cpu() -> TestResult {
    let cols = 256_usize;
    let rows = 67_usize;
    let blocks_per_row = cols / 128;
    let mut data = Vec::with_capacity(rows * blocks_per_row * 34);
    for r in 0..rows {
        for b in 0..blocks_per_row {
            let scale_bytes = if (r + b).is_multiple_of(3) {
                [0x00, 0x00]
            } else if (r + b) % 5 == 0 {
                [0x00, 0x80]
            } else {
                f16_scale(0.0625)
            };
            data.extend_from_slice(&scale_bytes);
            data.extend_from_slice(&[0xAA_u8; 32]);
        }
    }
    let tensor = Tensor {
        name: "test.pq2.zero_scale".to_owned(),
        dimensions: vec![cols, rows],
        kind: 142,
        data: Arc::<[u8]>::from(data).into(),
    };
    let act = build_q8k_activations(cols, 1, 9);
    let expected = cpu_pq2_q8k_reference(&tensor, &act, false);
    let mut exec = VulkanPq2Executor::open_default(16 * 1024 * 1024)?;
    let mut out = Vec::new();
    exec.project(&tensor, &act, false, &mut out)?;
    assert_zero_ulp(&out, &expected, "zero f16 scale block");
    Ok(())
}

#[test]
fn f32_direct_and_group_tokens_match_cpu() -> TestResult {
    let mut exec = VulkanPq2Executor::open_default(64 * 1024 * 1024)?;
    let cols = 512_usize;
    let gate = build_pq2_tensor("blk.0.ffn_gate.weight", cols, 512, true);
    let up = build_pq2_tensor("blk.0.ffn_up.weight", cols, 512, true);
    let extra = build_pq2_tensor("blk.0.attn_k.weight", cols, 256, true);
    assert_eq!(exec.prewarm_tensors(&[&gate, &up, &extra])?, 3);

    let mut scratch = MatvecScratch::with_backend(MatvecBackend::Auto);
    let mut act_tie = fill_f32_pattern(cols, 0.125);
    act_tie[0] = 3.5;
    act_tie[4] = -3.5;
    act_tie[32] = -3.5;

    let mut cpu_gate = Vec::new();
    scratch
        .prepare(&gate, &act_tie)?
        .matvec_into(&gate, &mut cpu_gate)?;
    let mut gpu_gate = Vec::new();
    exec.project_f32(&gate, &act_tie, false, &mut gpu_gate)?;
    assert_zero_ulp(&gpu_gate, &cpu_gate, "project_f32 with +/- tie");

    let mut g_a = Vec::new();
    let mut g_b = Vec::new();
    let mut g_c = Vec::new();
    exec.project_f32_group(
        &[&gate, &up, &extra],
        &act_tie,
        false,
        &mut [&mut g_a, &mut g_b, &mut g_c],
    )?;
    assert_zero_ulp(&g_a, &cpu_gate, "group[0]");

    let tok_acts: Vec<Vec<f32>> = (0..5_usize)
        .map(|t| {
            fill_f32_pattern(
                cols,
                0.01_f32.mul_add(f32::from(u8::try_from(t).unwrap_or(0)), 0.05),
            )
        })
        .collect();
    let mut gpu_toks = vec![Vec::new(); 5];
    exec.project_f32_tokens(&gate, &tok_acts, false, &mut gpu_toks)?;
    let mut pair_a = vec![Vec::new(); 5];
    let mut pair_b = vec![Vec::new(); 5];
    exec.project_f32_pair_tokens([&gate, &up], &tok_acts, false, &mut pair_a, &mut pair_b)?;
    let mut tri_a = vec![Vec::new(); 5];
    let mut tri_b = vec![Vec::new(); 5];
    let mut tri_c = vec![Vec::new(); 5];
    exec.project_f32_triple_tokens(
        [&gate, &up, &extra],
        &tok_acts,
        false,
        &mut tri_a,
        &mut tri_b,
        &mut tri_c,
    )?;
    for (t, act) in tok_acts.iter().enumerate() {
        let mut ref_g = Vec::new();
        let mut ref_u = Vec::new();
        scratch
            .prepare(&gate, act)?
            .matvec_into(&gate, &mut ref_g)?;
        scratch.prepare(&up, act)?.matvec_into(&up, &mut ref_u)?;
        assert_zero_ulp(&gpu_toks[t], &ref_g, &format!("project_f32_tokens[{t}]"));
        assert_zero_ulp(&pair_a[t], &ref_g, &format!("pair_a[{t}]"));
        assert_zero_ulp(&pair_b[t], &ref_u, &format!("pair_b[{t}]"));
        assert_zero_ulp(&tri_a[t], &ref_g, &format!("tri_a[{t}]"));
        assert_zero_ulp(&tri_b[t], &ref_u, &format!("tri_b[{t}]"));
    }
    Ok(())
}

#[test]
fn f32_ffn_chain_hadamard_and_lora_match_cpu() -> TestResult {
    let mut exec = VulkanPq2Executor::open_default(64 * 1024 * 1024)?;
    let cols = 512_usize;
    let inter = 512_usize;
    let gate = build_pq2_tensor("blk.0.ffn_gate.weight", cols, inter, true);
    let up = build_pq2_tensor("blk.0.ffn_up.weight", cols, inter, true);
    let down = build_pq2_tensor("blk.0.ffn_down.weight", inter, cols, false);

    let mut scratch = MatvecScratch::with_backend(MatvecBackend::Auto);
    let act = fill_f32_pattern(cols, 0.125);
    let mut cpu_gate = Vec::new();
    let mut cpu_up = Vec::new();
    scratch
        .prepare(&gate, &act)?
        .matvec_into(&gate, &mut cpu_gate)?;
    scratch.prepare(&up, &act)?.matvec_into(&up, &mut cpu_up)?;

    for pinned in [false, true] {
        let mut sw_out = Vec::new();
        exec.project_f32_ffn_swiglu(&gate, &up, &act, pinned, &mut sw_out)?;
        let cpu_mid: Vec<f32> = cpu_gate
            .iter()
            .zip(&cpu_up)
            .map(|(&g, &u)| u * silu(g))
            .collect();
        assert_close(
            &sw_out,
            &cpu_mid,
            1e-4,
            &format!("ffn_swiglu pinned={pinned}"),
        );
    }

    let mut lora_a = vec![0.0_f32; inter];
    for (i, slot) in lora_a.iter_mut().enumerate() {
        *slot = if i % 2 == 0 { 0.0625 } else { -0.03125 };
    }
    let residual = fill_f32_pattern(cols, 0.02);
    let mut sw_out = Vec::new();
    exec.project_f32_ffn_swiglu(&gate, &up, &act, false, &mut sw_out)?;
    let mut chain_out = Vec::new();
    let gpu_lora_dot = exec.project_f32_ffn_chain_lora(
        [&gate, &up, &down],
        &act,
        Some(&residual),
        Some(256),
        &lora_a,
        false,
        &mut chain_out,
    )?;
    let expected_dot: f32 = sw_out.iter().zip(&lora_a).map(|(&m, &a)| m * a).sum();
    assert_close(
        &[gpu_lora_dot],
        &[expected_dot],
        1e-4,
        "lora_dot pre-hadamard",
    );

    let mut rotated = sw_out;
    for chunk in rotated.as_chunks_mut::<256>().0 {
        hadamard(chunk)?;
    }
    let mut cpu_down = Vec::new();
    scratch
        .prepare(&down, &rotated)?
        .matvec_into(&down, &mut cpu_down)?;
    for (slot, &r) in cpu_down.iter_mut().zip(&residual) {
        *slot += r;
    }
    assert_close(&chain_out, &cpu_down, 1e-3, "ffn_chain_lora output");

    let tok_acts: Vec<Vec<f32>> = (0..5_usize)
        .map(|t| {
            fill_f32_pattern(
                cols,
                0.01_f32.mul_add(f32::from(u8::try_from(t).unwrap_or(0)), 0.05),
            )
        })
        .collect();
    let res_vecs: Vec<Vec<f32>> = (0..5).map(|_| residual.clone()).collect();
    let mut chain_toks = vec![Vec::new(); 5];
    exec.project_f32_ffn_chain_tokens(
        [&gate, &up, &down],
        &tok_acts,
        Some(&res_vecs),
        Some(256),
        false,
        &mut chain_toks,
    )?;
    for (t, got) in chain_toks.iter().enumerate() {
        let mut one = Vec::new();
        exec.project_f32_ffn_chain(
            [&gate, &up, &down],
            &tok_acts[t],
            Some(&res_vecs[t]),
            Some(256),
            false,
            &mut one,
        )?;
        assert_close(got, &one, 1e-5, &format!("ffn_chain_tokens[{t}]"));
    }
    Ok(())
}

#[test]
fn composite_executor_routes_overflow_layers_to_vulkan_executor() -> TestResult {
    struct MockNpuExecutor;

    impl Pq2ProjectionExecutor for MockNpuExecutor {
        #[expect(
            clippy::unnecessary_literal_bound,
            reason = "Matches Pq2ProjectionExecutor trait signature returning &str."
        )]
        fn identity(&self) -> &str {
            "mock-npu-v73"
        }
        fn operation_capacity(&self) -> Option<u64> {
            Some(1024 * 1024)
        }
        fn operation_count(&self, tensor: &Tensor) -> Option<u64> {
            u64::try_from(tensor.dimensions.first().copied()? * tensor.dimensions.get(1).copied()?)
                .ok()
        }
        fn supports(&self, _tensor: &Tensor) -> bool {
            true
        }
        fn project(
            &mut self,
            tensor: &Tensor,
            _activation: &[Q8KBlock],
            _fused: bool,
            output: &mut Vec<f32>,
        ) -> bonsai_prism_runtime::Result<()> {
            output.resize(tensor.dimensions[1], 1.0);
            Ok(())
        }
    }

    let plan = plan_x1_v73_three_processor_placement(64, 4, AcceleratorTarget::AdrenoGpuOpenCl)?;
    let mut npu = MockNpuExecutor;
    let mut gpu = VulkanPq2Executor::open_default(16 * 1024 * 1024)?;
    let mut composite = CompositeProjectionExecutor::new(&plan, &mut npu, &mut gpu)?;

    let t27 = build_pq2_tensor("blk.27.ffn_up.weight", 256, 64, true);
    let act = build_q8k_activations(256, 1, 2);
    let mut out = Vec::new();
    composite.project(&t27, &act, false, &mut out)?;
    assert_zero_ulp(
        &out,
        &cpu_pq2_q8k_reference(&t27, &act, false),
        "composite overflow gpu route",
    );
    assert_eq!(composite.telemetry().gpu_projections, 1);
    assert_eq!(composite.telemetry().npu_projections, 0);
    assert!(!composite.telemetry().faulted);
    Ok(())
}

fn fill_f32_pattern(len: usize, freq: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let s = f32::from(u8::try_from(i % 29).unwrap_or(0)) - 14.0;
            s * freq
        })
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], max_tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: len mismatch");
    let mut worst = 0.0_f32;
    for (&a, &e) in actual.iter().zip(expected) {
        let diff = (a - e).abs();
        if diff > worst {
            worst = diff;
        }
    }
    assert!(
        worst <= max_tol,
        "{label}: max abs diff {worst} exceeds tolerance {max_tol}"
    );
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "Covers short fused, decomposed, causal, long-seq pipelined, and resource-limit parity."
)]
fn attention_fused_decomposed_and_pipelined_match_cpu_f16_prob_oracle() -> TestResult {
    let mut exec = VulkanAttentionExecutor::open(64 * 1024 * 1024)?;

    let short_desc = AttentionDescriptor {
        query_len: 1,
        total_seq: 64,
        past_len: 63,
        num_heads: 24,
        num_kv_heads: 4,
        head_dim: 256,
        scale: 1.0 / 16.0,
        causal: true,
    };
    let (q_len, kv_len, _) = short_desc.element_counts()?;
    let q_short = fill_f32_pattern(q_len, 0.05);
    let k_short = fill_f32_pattern(kv_len, 0.04);
    let v_short = fill_f32_pattern(kv_len, 0.03);
    let (cpu_short, _) = cpu_attention_reference(&short_desc, &q_short, &k_short, &v_short)?;

    let fused_out = exec.execute(
        &short_desc,
        &q_short,
        &k_short,
        &v_short,
        AttentionStrategy::Auto,
        false,
        || false,
    )?;
    assert_eq!(fused_out.witness.plan.path, AdrenoAttentionPath::Fused);
    assert_eq!(fused_out.witness.stage_count, 1);
    assert_close(&fused_out.output, &cpu_short, 1e-4, "fused vs cpu seq=64");

    let decomp_out = exec.execute(
        &short_desc,
        &q_short,
        &k_short,
        &v_short,
        AttentionStrategy::ForceDecomposed,
        true,
        || false,
    )?;
    assert_eq!(
        decomp_out.witness.plan.path,
        AdrenoAttentionPath::Decomposed3Stage
    );
    assert_eq!(decomp_out.witness.stage_count, 3);
    assert_close(
        &decomp_out.output,
        &cpu_short,
        1e-4,
        "decomposed vs cpu seq=64",
    );
    assert_close(
        &fused_out.output,
        &decomp_out.output,
        1e-5,
        "fused vs decomposed seq=64",
    );
    let stages = decomp_out.stages.ok_or("expected stage capture")?;
    for &p in &stages.rounded_probs {
        assert_eq!(half_to_float(float_to_half(p)).to_bits(), p.to_bits());
    }

    let causal_desc = AttentionDescriptor {
        query_len: 4,
        total_seq: 24,
        past_len: 13,
        num_heads: 24,
        num_kv_heads: 4,
        head_dim: 256,
        scale: 1.0 / 16.0,
        causal: true,
    };
    let (qc, kvc, _) = causal_desc.element_counts()?;
    let q_c = fill_f32_pattern(qc, 0.04);
    let k_c = fill_f32_pattern(kvc, 0.03);
    let v_c = fill_f32_pattern(kvc, 0.02);
    let (cpu_c, _) = cpu_attention_reference(&causal_desc, &q_c, &k_c, &v_c)?;
    let causal_decomp = exec.execute(
        &causal_desc,
        &q_c,
        &k_c,
        &v_c,
        AttentionStrategy::ForceDecomposed,
        true,
        || false,
    )?;
    assert_close(&causal_decomp.output, &cpu_c, 1e-4, "causal q=4 decomposed");
    let c_stages = causal_decomp.stages.ok_or("expected causal stages")?;
    for &prob in c_stages.rounded_probs.iter().take(24).skip(14) {
        assert_eq!(prob, 0.0);
    }

    let long_desc = AttentionDescriptor {
        query_len: 1,
        total_seq: 600,
        past_len: 599,
        num_heads: 24,
        num_kv_heads: 4,
        head_dim: 256,
        scale: 1.0 / 16.0,
        causal: true,
    };
    let (ql, kvl, _) = long_desc.element_counts()?;
    let q_l = fill_f32_pattern(ql, 0.04);
    let k_l = fill_f32_pattern(kvl, 0.03);
    let v_l = fill_f32_pattern(kvl, 0.02);
    let (cpu_l, _) = cpu_attention_reference(&long_desc, &q_l, &k_l, &v_l)?;
    assert!(matches!(
        exec.execute(
            &long_desc,
            &q_l,
            &k_l,
            &v_l,
            AttentionStrategy::ForceFused,
            false,
            || false
        ),
        Err(Error::ResourceLimit(_))
    ));
    let long_auto = exec.execute(
        &long_desc,
        &q_l,
        &k_l,
        &v_l,
        AttentionStrategy::Auto,
        false,
        || false,
    )?;
    assert_eq!(
        long_auto.witness.plan.path,
        AdrenoAttentionPath::Decomposed3Stage
    );
    assert_close(
        &long_auto.output,
        &cpu_l,
        1e-4,
        "auto long decomposed seq=600",
    );

    let pipelined = exec.execute_pipelined_decomposed(&long_desc, &q_l, &k_l, &v_l)?;
    assert_close(
        &pipelined.output,
        &cpu_l,
        1e-4,
        "pipelined decomposed seq=600",
    );

    let mut stage_checks = 0_u32;
    let cancel_err = exec.execute(
        &short_desc,
        &q_short,
        &k_short,
        &v_short,
        AttentionStrategy::ForceDecomposed,
        false,
        || {
            stage_checks += 1;
            stage_checks >= 2
        },
    );
    assert_eq!(cancel_err, Err(Error::Cancelled));

    let k1 = attention_cache_key(
        &short_desc,
        &fused_out.witness.plan,
        exec.capabilities(),
        &fused_out.witness.qk_shader_digest,
        &fused_out.witness.softmax_shader_digest,
        &fused_out.witness.pv_shader_digest,
        &fused_out.witness.fused_shader_digest,
    )?;
    assert_eq!(k1, fused_out.witness.cache_key);

    let mut tight_exec = VulkanAttentionExecutor::open(1024)?;
    assert!(matches!(
        tight_exec.execute(
            &long_desc,
            &q_l,
            &k_l,
            &v_l,
            AttentionStrategy::Auto,
            false,
            || false
        ),
        Err(Error::ResourceLimit(_))
    ));
    Ok(())
}

fn time_ms_stats<F: FnMut() -> TestResult>(
    mut f: F,
    iters: usize,
) -> Result<(f64, f64, f64), Box<dyn std::error::Error + Send + Sync>> {
    f()?;
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        f()?;
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(f64::total_cmp);
    Ok((samples[0], samples[iters / 2], samples[iters - 1]))
}

#[test]
#[ignore = "Predeclared bounded AC-logged owner-device timing comparison."]
fn bench_arm1_pq2_q8k() -> TestResult {
    let tensor = build_pq2_tensor("bench.pq2.5120x17408", 5120, 17408, true);
    let act = build_q8k_activations(5120, 1, 1);

    let mut sdot_exec = VulkanPq2Executor::open_with_path(
        ProjectionKernelPath::IntegerDotPacked4x8,
        64 * 1024 * 1024,
    )?;
    let mut port_exec = VulkanPq2Executor::open_with_path(
        ProjectionKernelPath::PortableIntegerFallback,
        64 * 1024 * 1024,
    )?;
    let mut out = Vec::new();

    let (s_min, s_med, s_max) =
        time_ms_stats(|| sdot_exec.project(&tensor, &act, false, &mut out), 5)?;
    let (p_min, p_med, p_max) =
        time_ms_stats(|| port_exec.project(&tensor, &act, false, &mut out), 5)?;
    let (c_min, c_med, c_max) = time_ms_stats(
        || {
            std::hint::black_box(cpu_pq2_q8k_reference(&tensor, &act, false));
            Ok(())
        },
        5,
    )?;

    let mut stdout = std::io::stdout().lock();
    writeln!(
        stdout,
        "ARM1_PQ2_Q8K_5120x17408 sdot_ms={s_med:.3} (min={s_min:.3}, max={s_max:.3}) portable_ms={p_med:.3} (min={p_min:.3}, max={p_max:.3}) cpu_scalar_ms={c_med:.3} (min={c_min:.3}, max={c_max:.3})"
    )?;
    Ok(())
}

#[test]
#[ignore = "Predeclared bounded AC-logged owner-device timing comparison."]
fn bench_arm2_ptq1_q8_0() -> TestResult {
    let tensor = build_ptq1_tensor("bench.ptq1.5120x17408", 5120, 17408)?;
    let act = build_q8_activations(5120, 2);

    let mut sdot_exec = VulkanPq2Executor::open_with_path(
        ProjectionKernelPath::IntegerDotPacked4x8,
        64 * 1024 * 1024,
    )?;
    let mut port_exec = VulkanPq2Executor::open_with_path(
        ProjectionKernelPath::PortableIntegerFallback,
        64 * 1024 * 1024,
    )?;
    let mut out = Vec::new();

    let (s_min, s_med, s_max) =
        time_ms_stats(|| sdot_exec.project_q8(&tensor, &act, true, &mut out), 5)?;
    let (p_min, p_med, p_max) =
        time_ms_stats(|| port_exec.project_q8(&tensor, &act, true, &mut out), 5)?;
    let (c_min, c_med, c_max) = time_ms_stats(
        || {
            std::hint::black_box(cpu_ptq1_q8_0_reference(&tensor, &act, true));
            Ok(())
        },
        5,
    )?;

    let mut stdout = std::io::stdout().lock();
    writeln!(
        stdout,
        "ARM2_PTQ1_Q8_0_PINNED_5120x17408 sdot_ms={s_med:.3} (min={s_min:.3}, max={s_max:.3}) portable_ms={p_med:.3} (min={p_min:.3}, max={p_max:.3}) cpu_scalar_ms={c_med:.3} (min={c_min:.3}, max={c_max:.3})"
    )?;
    Ok(())
}

#[test]
#[ignore = "Predeclared bounded AC-logged owner-device timing comparison."]
fn bench_arm3_attention() -> TestResult {
    let mut exec = VulkanAttentionExecutor::open(128 * 1024 * 1024)?;
    let mut stdout = std::io::stdout().lock();

    for seq in [64_u32, 512, 2048] {
        let desc = AttentionDescriptor {
            query_len: 1,
            total_seq: seq,
            past_len: seq - 1,
            num_heads: 24,
            num_kv_heads: 4,
            head_dim: 256,
            scale: 1.0 / 16.0,
            causal: true,
        };
        let (ql, kvl, _) = desc.element_counts()?;
        let q = fill_f32_pattern(ql, 0.04);
        let k = fill_f32_pattern(kvl, 0.03);
        let v = fill_f32_pattern(kvl, 0.02);

        if seq <= 512 {
            let (f_min, f_med, f_max) = time_ms_stats(
                || {
                    let _ = exec.execute(
                        &desc,
                        &q,
                        &k,
                        &v,
                        AttentionStrategy::ForceFused,
                        false,
                        || false,
                    )?;
                    Ok(())
                },
                5,
            )?;
            let (d_min, d_med, d_max) = time_ms_stats(
                || {
                    let _ = exec.execute_pipelined_decomposed(&desc, &q, &k, &v)?;
                    Ok(())
                },
                5,
            )?;
            writeln!(
                stdout,
                "ARM3_ATTN seq={seq} fused_ms={f_med:.3} (min={f_min:.3}, max={f_max:.3}) decomposed_ms={d_med:.3} (min={d_min:.3}, max={d_max:.3})"
            )?;
        } else {
            let (d_min, d_med, d_max) = time_ms_stats(
                || {
                    let _ = exec.execute_pipelined_decomposed(&desc, &q, &k, &v)?;
                    Ok(())
                },
                5,
            )?;
            writeln!(
                stdout,
                "ARM3_ATTN seq={seq} decomposed_ms={d_med:.3} (min={d_min:.3}, max={d_max:.3})"
            )?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "Predeclared bounded AC-logged owner-device timing comparison."]
fn bench_arm4_coalesced_and_ffn_chain() -> TestResult {
    let gate = build_pq2_tensor("bench.ffn_gate.5120x17408", 5120, 17408, true);
    let up = build_pq2_tensor("bench.ffn_up.5120x17408", 5120, 17408, true);
    let down = build_pq2_tensor("bench.ffn_down.17408x5120", 17408, 5120, true);
    let mut exec = VulkanPq2Executor::open_default(512 * 1024 * 1024)?;
    let prewarm_start = Instant::now();
    let warmed = exec.prewarm_tensors(&[&gate, &up, &down])?;
    let prewarm_ms = prewarm_start.elapsed().as_secs_f64() * 1000.0;

    let act1 = fill_f32_pattern(5120, 0.03125);
    let res1 = fill_f32_pattern(5120, 0.01);
    let acts4 = vec![act1.clone(); 4];
    let ress4 = vec![res1.clone(); 4];
    let mut out1 = Vec::new();
    let mut outs4 = vec![Vec::new(); 4];

    let (g1_min, g1_med, g1_max) =
        time_ms_stats(|| exec.project_f32(&gate, &act1, true, &mut out1), 5)?;
    let (g4_min, g4_med, g4_max) = time_ms_stats(
        || exec.project_f32_tokens(&gate, &acts4, true, &mut outs4),
        5,
    )?;
    let (c1_min, c1_med, c1_max) = time_ms_stats(
        || {
            exec.project_f32_ffn_chain(
                [&gate, &up, &down],
                &act1,
                Some(&res1),
                Some(128),
                true,
                &mut out1,
            )
        },
        5,
    )?;
    let (c4_min, c4_med, c4_max) = time_ms_stats(
        || {
            exec.project_f32_ffn_chain_tokens(
                [&gate, &up, &down],
                &acts4,
                Some(&ress4),
                Some(128),
                true,
                &mut outs4,
            )
        },
        5,
    )?;

    let mut stdout = std::io::stdout().lock();
    writeln!(
        stdout,
        "ARM4_COALESCED_FFN warmed={warmed} cached_bytes={} prewarm_ms={prewarm_ms:.3} gemv1_5120x17408_ms={g1_med:.3} (min={g1_min:.3}, max={g1_max:.3}) gemm4_5120x17408_ms={g4_med:.3} (min={g4_min:.3}, max={g4_max:.3}) ffn_chain1_ms={c1_med:.3} (min={c1_min:.3}, max={c1_max:.3}) ffn_chain4_ms={c4_med:.3} (min={c4_min:.3}, max={c4_max:.3})",
        exec.cached_weight_bytes()
    )?;
    Ok(())
}
