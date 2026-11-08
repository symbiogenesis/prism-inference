//! End-to-end exact tiles, separately qualified GDN chunks and request recovery.

use std::cell::Cell;

use super::*;

fn tiled(model: Model, tokens: usize, chunkwise: bool) -> Result<Qwen35> {
    let graph = Qwen35::new(model, 192)?;
    let bytes = graph.tiled_prefill_scratch_bytes(tokens)?;
    let graph = graph.with_tiled_prefill(tokens, bytes)?;
    if chunkwise {
        graph.with_chunkwise_gdn_prefill()
    } else {
        Ok(graph)
    }
}

fn state_values(graph: &Qwen35) -> Result<Vec<f32>> {
    let mut result = Vec::new();
    graph.visit_state(&mut |_, _, values| {
        result.extend_from_slice(values);
        Ok(())
    })?;
    Ok(result)
}

#[allow(
    clippy::suboptimal_flops,
    reason = "Retain the declared separately rounded absolute-plus-relative tolerance."
)]
fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 0.001 + 0.001 * expected.abs(),
            "chunk value {actual} differs from {expected}"
        );
    }
}

#[test]
fn tiled_prefill_exact_hybrid_state_and_logits_cross_pages() -> Result<()> {
    for tokens in [8, 16, 32] {
        for length in [1, 7, 8, 9, 31, 32, 33, 63, 64, 65] {
            let prompt: Vec<_> = [0, 1].into_iter().cycle().take(length).collect();
            let mut expected = Qwen35::new(paged_hybrid_fixture(), 192)?;
            let mut actual = tiled(paged_hybrid_fixture(), tokens, false)?;
            let checkpoint = actual.checkpoint()?;
            for &token in &prompt {
                expected.ingest(token)?;
            }
            actual.ingest_block(&prompt, &|| Ok(()))?;
            assert_eq!(graph_bits(&actual)?, graph_bits(&expected)?);
            assert_eq!(actual.evaluate(1)?, expected.evaluate(1)?);
            actual.restore(&checkpoint)?;
            actual.ingest_block(&prompt, &|| Ok(()))?;
            actual.evaluate(1)?;
            assert_eq!(graph_bits(&actual)?, graph_bits(&expected)?);
        }
    }
    Ok(())
}

#[test]
fn tiled_prefill_packed_ffn_and_adapter_preserve_original_basis_bits() -> Result<()> {
    for recurrent in [false, true] {
        for (kind, width) in [(142, 128), (142, 256), (143, 128)] {
            for scale in [0.0, 1.3] {
                let build = || -> Result<Qwen35> {
                    let model = backend_fixture(recurrent, kind, width)?;
                    let factors = rank_one_factors(&model, &["blk.0.ffn_down.weight"], 0.125);
                    rank_one_graph(model, factors, scale)
                };
                let mut expected = build()?;
                let graph = build()?;
                let reserve = graph.tiled_prefill_scratch_bytes(8)?;
                let mut actual = graph.with_tiled_prefill(8, reserve)?;
                let prompt = [0, 1, 1, 0, 1, 0, 0, 1, 0];
                for token in prompt {
                    expected.ingest(token)?;
                }
                actual.ingest_block(&prompt, &|| Ok(()))?;
                assert_eq!(graph_bits(&actual)?, graph_bits(&expected)?);
                assert_eq!(actual.evaluate(1)?, expected.evaluate(1)?);
                let stats = actual.tiled_prefill_statistics();
                assert_eq!(stats.completed_tiles, 2);
                assert_eq!(stats.completed_tokens, 9);
                assert_eq!(stats.packed_projections, 2);
                assert_eq!(
                    stats.decoded_weight_blocks,
                    u64::try_from(width / 128 * 2 * 2)?
                );
            }
        }
    }
    Ok(())
}

#[cfg(feature = "parallel-rows")]
#[test]
fn tiled_prefill_parallel_workers_preserve_state_restore_and_policy_changes() -> Result<()> {
    use bonsai_prism_runtime::kernels::RowParallelism;

    let four = RowParallelism::new(4)?;
    let two = RowParallelism::new(2)?;
    for recurrent in [false, true] {
        for (kind, width) in [(142, 128), (142, 256), (143, 128)] {
            for scale in [0.0, 1.3] {
                for select_before_tile in [false, true] {
                    let build = || -> Result<Qwen35> {
                        let model = backend_fixture(recurrent, kind, width)?;
                        let factors = rank_one_factors(&model, &["blk.0.ffn_down.weight"], 0.125);
                        rank_one_graph(model, factors, scale)
                    };
                    let mut expected = build()?;
                    let mut graph = build()?;
                    if select_before_tile {
                        graph.set_row_parallelism(four.clone());
                    }
                    let required = graph.tiled_prefill_scratch_bytes(8)?;
                    let mut actual = graph.with_tiled_prefill(8, required)?;
                    if !select_before_tile {
                        actual.set_row_parallelism(four.clone());
                    }
                    assert_eq!(actual.row_workers(), 4);
                    let checkpoint = actual.checkpoint()?;
                    let prompt = [0, 1, 1, 0, 1, 0, 0, 1, 0];
                    for token in prompt {
                        expected.ingest(token)?;
                    }
                    let caller = std::thread::current().id();
                    let calls = Cell::new(0);
                    actual.ingest_block(&prompt, &|| {
                        assert_eq!(std::thread::current().id(), caller);
                        calls.set(calls.get() + 1);
                        Ok(())
                    })?;
                    assert_eq!(graph_bits(&actual)?, graph_bits(&expected)?);
                    assert_eq!(actual.evaluate(1)?, expected.evaluate(1)?);
                    assert!(calls.get() > 0);
                    actual.set_row_parallelism(two.clone());
                    actual.restore(&checkpoint)?;
                    assert_eq!(actual.row_workers(), 2);
                    actual.ingest_block(&prompt, &|| Ok(()))?;
                    actual.evaluate(1)?;
                    assert_eq!(graph_bits(&actual)?, graph_bits(&expected)?);
                    actual.reset()?;
                    assert_eq!(actual.row_workers(), 2);
                }
            }
        }
    }
    Ok(())
}

#[test]
fn tiled_prefill_failure_restores_request_and_explicit_reset_clears_first_tile() -> Result<()> {
    for chunkwise in [false, true] {
        let mut graph = tiled(paged_hybrid_fixture(), 8, chunkwise)?;
        let original = graph.checkpoint()?;
        let checks = Cell::new(0);
        let result = graph.ingest_block(&[0, 1, 0, 1, 0, 1, 0, 1], &|| {
            checks.set(checks.get() + 1);
            if checks.get() == 14 {
                Err("synthetic tile cancellation".into())
            } else {
                Ok(())
            }
        });
        assert!(result.is_err());
        graph.reset()?;
        assert!(state_values(&graph)?.iter().all(|&value| value == 0.0));
        graph.restore(&original)?;
        graph.ingest_block(&[0, 1, 0], &|| Ok(()))?;
        let mut reference = Qwen35::new(paged_hybrid_fixture(), 192)?;
        for token in [0, 1, 0] {
            reference.ingest(token)?;
        }
        if chunkwise {
            close(&state_values(&graph)?, &state_values(&reference)?);
        } else {
            assert_eq!(graph_bits(&graph)?, graph_bits(&reference)?);
        }
    }
    Ok(())
}

#[test]
fn chunkwise_gdn_prefill_continuation_restore_and_identity_are_separate() -> Result<()> {
    for tokens in [8, 16, 32] {
        let mut actual = tiled(paged_hybrid_fixture(), tokens, true)?;
        let mut expected = Qwen35::new(paged_hybrid_fixture(), 192)?;
        assert_ne!(actual.identity(), expected.identity());
        let prompt: Vec<_> = [0, 1].into_iter().cycle().take(65).collect();
        actual.ingest_block(&prompt, &|| Ok(()))?;
        for &token in &prompt {
            expected.ingest(token)?;
        }
        close(&state_values(&actual)?, &state_values(&expected)?);
        let saved = actual.checkpoint()?;
        for token in [0, 1, 1, 0] {
            close(&actual.evaluate(token)?, &expected.evaluate(token)?);
        }
        actual.restore(&saved)?;
        let mut fresh = Qwen35::new(paged_hybrid_fixture(), 192)?;
        for token in prompt {
            fresh.ingest(token)?;
        }
        close(&actual.evaluate(0)?, &fresh.evaluate(0)?);
        assert!(actual.tiled_prefill_statistics().chunkwise_gdn_heads > 0);
    }
    Ok(())
}

#[test]
fn tiled_prefill_rejects_unadmitted_geometry_and_preserves_state() -> Result<()> {
    for tokens in [0, 1, 4, 33] {
        assert!(
            Qwen35::new(hybrid_fixture(), 16)?
                .with_tiled_prefill(tokens, usize::MAX)
                .is_err()
        );
    }
    let graph = Qwen35::new(hybrid_fixture(), 16)?;
    let required = graph.tiled_prefill_scratch_bytes(8)?;
    assert!(graph.with_tiled_prefill(8, required - 1).is_err());
    let mut graph = tiled(paged_hybrid_fixture(), 8, false)?;
    let before = graph_bits(&graph)?;
    for prompt in [vec![], vec![2], vec![0; 193]] {
        assert!(graph.ingest_block(&prompt, &|| Ok(())).is_err());
        assert_eq!(before, graph_bits(&graph)?);
    }
    Ok(())
}

#[test]
fn tiled_prefill_runtime_failure_restores_all_layer_state_before_replay() -> Result<()> {
    for chunkwise in [false, true] {
        let limits = Limits {
            context_tokens: 192,
            checkpoints: 0,
            checkpoint_bytes: 1_048_576,
        };
        let mut runtime = Runtime::new(tiled(paged_hybrid_fixture(), 8, chunkwise)?, limits)?;
        let cancellation = Cancellation::default();
        let generation = Generation {
            max_tokens: 1,
            stop_tokens: Vec::new(),
        };
        let prefill = Prefill { block_tokens: 32 };
        let prompt: Vec<_> = [0, 1].into_iter().cycle().take(65).collect();
        assert!(
            runtime
                .generate_with_prefill(&prompt, &generation, prefill, &cancellation, |_| Err(
                    "synthetic emitted-token failure".into()
                ))
                .is_err()
        );
        assert_eq!(runtime.tokens(), [0_u32; 0]);
        assert!(
            state_values(runtime.decoder())?
                .iter()
                .all(|&value| value == 0.0)
        );
        assert!(!runtime.is_faulted());
        let mut actual = Vec::new();
        runtime.generate_with_prefill(&prompt, &generation, prefill, &cancellation, |token| {
            actual.push(token);
            Ok(())
        })?;
        let mut reference = Runtime::new(Qwen35::new(paged_hybrid_fixture(), 192)?, limits)?;
        let mut expected = Vec::new();
        reference.generate(&prompt, &generation, &cancellation, |token| {
            expected.push(token);
            Ok(())
        })?;
        assert_eq!(actual, expected);
        if chunkwise {
            close(
                &state_values(runtime.decoder())?,
                &state_values(reference.decoder())?,
            );
        } else {
            assert_eq!(
                graph_bits(runtime.decoder())?,
                graph_bits(reference.decoder())?
            );
        }
    }
    Ok(())
}

#[test]
fn tiled_prefill_adapter_scale_reset_preserves_execution_identity_and_provenance() -> Result<()> {
    for chunkwise in [false, true] {
        let base = fixture(true);
        let factors = rank_one_factors(&base, &["blk.0.ssm_out.weight"], 0.125);
        let graph = rank_one_graph(base, factors, 1.3)?;
        let required = graph.tiled_prefill_scratch_bytes(8)?;
        let graph = graph.with_tiled_prefill(8, required)?;
        let mut graph = if chunkwise {
            graph.with_chunkwise_gdn_prefill()?
        } else {
            graph
        };
        let original_identity = graph.identity().to_owned();
        graph.ingest_block(&[0, 1, 0], &|| Ok(()))?;
        let original_checkpoint = graph.checkpoint()?;
        for scale in [1.3, 0.5, 1.3] {
            let old = graph.checkpoint()?;
            graph.reset_with_adapter_scale(scale)?;
            assert!(graph.identity().contains(":cpu-tiled-ffn-v1:8"));
            assert_eq!(graph.identity().contains(":gdn-wy-f32-v1"), chunkwise);
            assert!(graph.restore(&old).is_err());
            graph.ingest_block(&[0, 1, 0], &|| Ok(()))?;
        }
        assert_eq!(graph.identity(), original_identity);
        assert!(graph.restore(&original_checkpoint).is_err());
    }
    Ok(())
}

#[test]
fn tiled_prefill_batches_packed_attention_and_recurrent_projections() -> Result<()> {
    let packed_tensor = |name: &str, cols: usize, rows: usize, kind: u32| -> Result<Tensor> {
        let mut block = [0x66_u8; 34];
        block[..2].copy_from_slice(&0x2800_u16.to_le_bytes());
        let bytes = if kind == 143 {
            bonsai_ternary::pack_ptq(&block)?.to_vec()
        } else {
            block.to_vec()
        };
        Ok(Tensor {
            name: name.to_owned(),
            dimensions: vec![cols, rows],
            kind,
            data: Arc::from(bytes.repeat((cols / 128) * rows)),
        })
    };
    for recurrent in [false, true] {
        for kind in [142_u32, 143] {
            let build = || -> Result<Qwen35> {
                let mut model = fixture(recurrent);
                for (key, val) in [
                    ("qwen35.embedding_length", 128_u64),
                    ("qwen35.feed_forward_length", 128),
                    ("qwen35.attention.key_length", 128),
                    ("qwen35.attention.value_length", 128),
                    ("qwen35.ssm.state_size", 128),
                    ("qwen35.ssm.inner_size", 128),
                ] {
                    model
                        .metadata
                        .insert(key.to_owned(), MetadataValue::from(val));
                }
                let f32_insert = |model: &mut Model, name: &str, dims: &[usize], fill: f32| {
                    let count = dims.iter().product();
                    model
                        .tensors
                        .insert(name.to_owned(), weight(name, dims, &vec![fill; count]));
                };
                f32_insert(&mut model, "token_embd.weight", &[128, 2], 0.125);
                f32_insert(&mut model, "output.weight", &[128, 2], 0.125);
                f32_insert(&mut model, "output_norm.weight", &[128], 1.0);
                f32_insert(&mut model, "blk.0.attn_norm.weight", &[128], 1.0);
                f32_insert(&mut model, "blk.0.post_attention_norm.weight", &[128], 1.0);
                for name in [
                    "blk.0.ffn_up.weight",
                    "blk.0.ffn_gate.weight",
                    "blk.0.ffn_down.weight",
                ] {
                    model
                        .tensors
                        .insert(name.to_owned(), packed_tensor(name, 128, 128, kind)?);
                }
                if recurrent {
                    for (name, rows) in [
                        ("blk.0.attn_qkv.weight", 384),
                        ("blk.0.attn_gate.weight", 128),
                        ("blk.0.ssm_out.weight", 128),
                    ] {
                        model
                            .tensors
                            .insert(name.to_owned(), packed_tensor(name, 128, rows, kind)?);
                    }
                    f32_insert(&mut model, "blk.0.ssm_conv1d.weight", &[2, 384], 0.25);
                    f32_insert(&mut model, "blk.0.ssm_beta.weight", &[128, 1], 0.0625);
                    f32_insert(&mut model, "blk.0.ssm_alpha.weight", &[128, 1], -0.0625);
                    f32_insert(&mut model, "blk.0.ssm_norm.weight", &[128], 1.0);
                } else {
                    for (name, rows) in [
                        ("blk.0.attn_q.weight", 256),
                        ("blk.0.attn_k.weight", 128),
                        ("blk.0.attn_v.weight", 128),
                        ("blk.0.attn_output.weight", 128),
                    ] {
                        model
                            .tensors
                            .insert(name.to_owned(), packed_tensor(name, 128, rows, kind)?);
                    }
                    f32_insert(&mut model, "blk.0.attn_q_norm.weight", &[128], 1.0);
                    f32_insert(&mut model, "blk.0.attn_k_norm.weight", &[128], 1.0);
                }
                Qwen35::new(model, 16)
            };
            let mut expected = build()?;
            let base = build()?;
            let reserve = base.tiled_prefill_scratch_bytes(8)?;
            let mut actual = base.with_tiled_prefill(8, reserve)?;
            let prompt = [0, 1, 1, 0, 1, 0, 0, 1, 0];
            for token in prompt {
                expected.ingest(token)?;
            }
            actual.ingest_block(&prompt, &|| Ok(()))?;
            assert_eq!(graph_bits(&actual)?, graph_bits(&expected)?);
            assert_eq!(actual.evaluate(1)?, expected.evaluate(1)?);
            let stats = actual.tiled_prefill_statistics();
            assert_eq!(stats.completed_tiles, 2);
            assert_eq!(stats.completed_tokens, 9);
            assert_eq!(stats.packed_projections, if recurrent { 12 } else { 14 });
        }
    }
    Ok(())
}

#[cfg(feature = "parallel-rows")]
fn parallel_grouped_hybrid_fixture(pinned: bool) -> Model {
    let mut model = paged_hybrid_fixture();
    for (key, val) in [
        ("qwen35.embedding_length", 16_u64),
        ("qwen35.feed_forward_length", 16),
        ("qwen35.attention.head_count", 8),
        ("qwen35.attention.head_count_kv", 2),
        ("qwen35.attention.key_length", 256),
        ("qwen35.attention.value_length", 256),
        ("qwen35.rope.dimension_count", 64),
        ("qwen35.ssm.state_size", 16),
        ("qwen35.ssm.group_count", 2),
        ("qwen35.ssm.time_step_rank", 4),
        ("qwen35.ssm.inner_size", 64),
        ("qwen35.ssm.conv_kernel", 4),
        ("qwen35.full_attention_interval", 2),
    ] {
        model
            .metadata
            .insert(key.to_owned(), MetadataValue::from(val));
    }
    model.metadata.insert(
        "qwen35.rope.dimension_sections".to_owned(),
        MetadataValue::from(vec![16_i64, 8, 8, 0]),
    );
    let mut tensors = BTreeMap::new();
    let p0: &[f32] = &[0.125, -0.25, 0.0625, 0.25, -0.125];
    let p1: &[f32] = &[0.25, 0.125, -0.125, 0.0625, -0.25];
    let p2: &[f32] = &[-0.125, 0.25, 0.125, -0.0625, 0.25];
    let p3: &[f32] = &[0.0625, -0.125, 0.125, 0.0625, -0.0625];
    let specs: &[(&str, &[usize], &[f32])] = &[
        ("token_embd.weight", &[16, 2], p0),
        ("output.weight", &[16, 2], p1),
        ("output_norm.weight", &[16], &[1.0]),
        ("blk.0.attn_norm.weight", &[16], &[1.0]),
        ("blk.0.post_attention_norm.weight", &[16], &[1.0]),
        ("blk.0.ffn_up.weight", &[16, 16], p0),
        ("blk.0.ffn_gate.weight", &[16, 16], p1),
        ("blk.0.ffn_down.weight", &[16, 16], p2),
        ("blk.0.attn_q.weight", &[16, 4096], p0),
        ("blk.0.attn_k.weight", &[16, 512], p1),
        ("blk.0.attn_v.weight", &[16, 512], p2),
        ("blk.0.attn_output.weight", &[2048, 16], p3),
        ("blk.0.attn_q_norm.weight", &[256], &[1.0]),
        ("blk.0.attn_k_norm.weight", &[256], &[1.0]),
        ("blk.1.attn_norm.weight", &[16], &[1.0]),
        ("blk.1.post_attention_norm.weight", &[16], &[1.0]),
        ("blk.1.ffn_up.weight", &[16, 16], p0),
        ("blk.1.ffn_gate.weight", &[16, 16], p1),
        ("blk.1.ffn_down.weight", &[16, 16], p2),
        ("blk.1.attn_qkv.weight", &[16, 128], p0),
        ("blk.1.attn_gate.weight", &[16, 64], p1),
        ("blk.1.ssm_conv1d.weight", &[4, 128], p2),
        ("blk.1.ssm_dt.bias", &[4], &[0.1, -0.1, 0.2, -0.2]),
        ("blk.1.ssm_a", &[4], &[-0.5, -0.25, -0.75, -0.125]),
        ("blk.1.ssm_beta.weight", &[16, 4], p0),
        ("blk.1.ssm_alpha.weight", &[16, 4], p1),
        ("blk.1.ssm_norm.weight", &[16], &[1.0]),
        ("blk.1.ssm_out.weight", &[64, 16], p3),
    ];
    for &(name, dimensions, pattern) in specs {
        let len: usize = dimensions.iter().product();
        let values: Vec<f32> = pattern.iter().copied().cycle().take(len).collect();
        tensors.insert(name.to_owned(), weight(name, dimensions, &values));
    }
    model.tensors = tensors;
    model.identity = if pinned {
        bonsai_prism_runtime::model::swift::MODEL_SHA256.to_owned()
    } else {
        "parallel-grouped-hybrid-fixture".to_owned()
    };
    model
}

#[cfg(feature = "parallel-rows")]
#[test]
fn tiled_prefill_parallel_grouped_kv_and_head_major_recurrent_match_single_token() -> Result<()> {
    use bonsai_prism_runtime::kernels::RowParallelism;

    let workers = RowParallelism::new(4)?;
    let prompt: Vec<u32> = [0, 1, 1, 0, 1, 0, 0].into_iter().cycle().take(65).collect();
    for pinned in [false, true] {
        let mut expected = Qwen35::new(parallel_grouped_hybrid_fixture(pinned), 192)?;
        expected.set_row_parallelism(workers.clone());
        for &token in &prompt {
            expected.ingest(token)?;
        }
        let expected_state = state_values(&expected)?;
        let expected_bits = graph_bits(&expected)?;
        let expected_logits = expected.evaluate(1)?;

        for tokens in [8, 16, 32] {
            for chunkwise in [false, true] {
                let mut actual = tiled(parallel_grouped_hybrid_fixture(pinned), tokens, chunkwise)?;
                actual.set_row_parallelism(workers.clone());
                actual.ingest_block(&prompt, &|| Ok(()))?;
                if chunkwise {
                    close(&state_values(&actual)?, &expected_state);
                    close(&actual.evaluate(1)?, &expected_logits);
                } else {
                    assert_eq!(graph_bits(&actual)?, expected_bits);
                    assert_eq!(
                        actual
                            .evaluate(1)?
                            .into_iter()
                            .map(f32::to_bits)
                            .collect::<Vec<_>>(),
                        expected_logits
                            .iter()
                            .copied()
                            .map(f32::to_bits)
                            .collect::<Vec<_>>()
                    );
                }
            }
        }
    }
    Ok(())
}
