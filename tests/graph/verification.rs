//! Per-position target logits and independent complete-state replay controls.

use std::cell::Cell;

use super::*;

struct RefusingExecutor;

impl bonsai_prism_runtime::kernels::Pq2ProjectionExecutor for RefusingExecutor {
    fn identity(&self) -> &'static str {
        "verification-refused-executor"
    }
    fn supports(&self, _: &Tensor) -> bool {
        false
    }
    fn project(
        &mut self,
        _: &Tensor,
        _: &[bonsai_ternary::Q8KBlock],
        _: bool,
        _: &mut Vec<f32>,
    ) -> Result<()> {
        Err("verification must not invoke an external executor".into())
    }
}

fn verified(model: Model) -> Result<Qwen35> {
    let graph = Qwen35::new(model, 192)?;
    let tile_bytes = graph.tiled_prefill_scratch_bytes(8)?;
    let graph = graph.with_tiled_prefill(8, tile_bytes)?;
    let logits_bytes = graph.tiled_verification_scratch_bytes()?;
    let graph = graph.with_tiled_verification(logits_bytes)?;
    assert_eq!(
        graph.tiled_prefill_reserved_bytes(),
        tile_bytes + logits_bytes
    );
    assert!(graph.identity().contains(":cpu-tiled-verify-v1"));
    Ok(graph)
}

fn bits(rows: &[Vec<f32>]) -> Vec<Vec<u32>> {
    rows.iter()
        .map(|row| row.iter().map(|value| value.to_bits()).collect())
        .collect()
}

#[test]
fn tiled_verification_matches_each_position_and_complete_hybrid_state() -> Result<()> {
    for base in [0, 63, 64] {
        for count in [1, 2, 4, 8] {
            let mut expected = Qwen35::new(paged_hybrid_fixture(), 192)?;
            let mut actual = verified(paged_hybrid_fixture())?;
            // Force the state-only optimization on even this tiny fixture;
            // verification must still execute the complete final layer.
            actual.set_last_layer_prefill_narrowing(true);
            for token in [0, 1].into_iter().cycle().take(base) {
                assert_eq!(actual.evaluate(token)?, expected.evaluate(token)?);
            }
            let checkpoint = actual.checkpoint()?;
            let draft: Vec<_> = [1, 0, 0, 1].into_iter().cycle().take(count).collect();
            let mut reference_rows = Vec::new();
            for &token in &draft {
                reference_rows.push(expected.evaluate(token)?);
            }
            let mut rows = vec![vec![f32::NAN]];
            actual.evaluate_batch_into(&draft, &mut rows, &|| Ok(()))?;
            assert_eq!(bits(&rows), bits(&reference_rows));
            assert_eq!(graph_bits(&actual)?, graph_bits(&expected)?);
            let continuation = expected.evaluate(1)?;
            assert_eq!(actual.evaluate(1)?, continuation);
            actual.restore(&checkpoint)?;
            actual.evaluate_batch_into(&draft, &mut rows, &|| Ok(()))?;
            assert_eq!(bits(&rows), bits(&reference_rows));
            assert_eq!(actual.evaluate(1)?, continuation);
            assert_eq!(graph_bits(&actual)?, graph_bits(&expected)?);
        }
    }
    Ok(())
}

#[test]
fn tiled_verification_shares_packed_output_head_and_adapter() -> Result<()> {
    for recurrent in [false, true] {
        for scale in [0.0, 1.3] {
            let build = || -> Result<Qwen35> {
                let model = packed_output_fixture(recurrent);
                let factors = rank_one_factors(&model, &["blk.0.ffn_down.weight"], 0.125);
                rank_one_graph(model, factors, scale)
            };
            let mut expected = build()?;
            let graph = build()?;
            let tile_bytes = graph.tiled_prefill_scratch_bytes(8)?;
            let graph = graph.with_tiled_prefill(8, tile_bytes)?;
            let logits_bytes = graph.tiled_verification_scratch_bytes()?;
            let mut actual = graph.with_tiled_verification(logits_bytes)?;
            let mut rows = Vec::new();
            let draft = [0, 1, 1, 0];
            let reference: Vec<_> = draft
                .into_iter()
                .map(|token| expected.evaluate(token))
                .collect::<Result<_>>()?;
            actual.evaluate_batch_into(&draft, &mut rows, &|| Ok(()))?;
            assert_eq!(bits(&rows), bits(&reference));
            assert_eq!(graph_bits(&actual)?, graph_bits(&expected)?);
            let packed = packed_output_fixture(recurrent);
            let projections: Vec<_> = packed
                .tensors
                .values()
                .filter(|tensor| tensor.kind == 142 && tensor.name != "token_embd.weight")
                .collect();
            let blocks = projections
                .iter()
                .map(|tensor| tensor.dimensions[0] / 128 * tensor.dimensions[1])
                .sum::<usize>();
            // This includes output.weight. A sequential vocabulary projection
            // would omit its projection and blocks from these exact counts.
            let stats = actual.tiled_prefill_statistics();
            assert_eq!(stats.packed_projections, u64::try_from(projections.len())?);
            assert_eq!(stats.decoded_weight_blocks, u64::try_from(blocks)?);
            let old = actual.checkpoint()?;
            actual.reset_with_adapter_scale(scale)?;
            assert!(actual.identity().contains(":cpu-tiled-verify-v1"));
            assert_eq!(actual.batched_evaluation_capacity(), 8);
            assert!(actual.restore(&old).is_err());
        }
    }
    Ok(())
}

fn packed_output_fixture(recurrent: bool) -> Model {
    let mut model = fixture(recurrent);
    model.metadata.insert(
        "qwen35.embedding_length".into(),
        MetadataValue::from(256_u64),
    );
    for tensor in model.tensors.values_mut() {
        let name = tensor.name.as_str();
        if matches!(
            name,
            "token_embd.weight"
                | "output.weight"
                | "blk.0.ffn_up.weight"
                | "blk.0.ffn_gate.weight"
                | "blk.0.attn_q.weight"
                | "blk.0.attn_k.weight"
                | "blk.0.attn_v.weight"
                | "blk.0.attn_qkv.weight"
                | "blk.0.attn_gate.weight"
                | "blk.0.ssm_beta.weight"
                | "blk.0.ssm_alpha.weight"
        ) {
            tensor.dimensions[0] = 256;
            tensor.kind = 142;
            let mut block = [0x66; 34];
            block[..2].copy_from_slice(&0x2800_u16.to_le_bytes());
            tensor.data = Arc::from(block.repeat(2 * tensor.dimensions[1]));
        } else if matches!(
            name,
            "blk.0.attn_output.weight" | "blk.0.ssm_out.weight" | "blk.0.ffn_down.weight"
        ) {
            tensor.dimensions[1] = 256;
            tensor.data = Arc::from(vec![0; tensor.dimensions.iter().product::<usize>() * 4]);
        } else if matches!(
            name,
            "output_norm.weight" | "blk.0.attn_norm.weight" | "blk.0.post_attention_norm.weight"
        ) {
            tensor.dimensions[0] = 256;
            tensor.data = Arc::from(1.0_f32.to_le_bytes().repeat(256));
        }
    }
    model
}

#[test]
fn tiled_verification_rejects_unadmitted_modes_and_clears_failed_output() -> Result<()> {
    let mut plain = Qwen35::new(paged_hybrid_fixture(), 192)?;
    assert_eq!(plain.batched_evaluation_capacity(), 0);
    let mut rows = vec![vec![1.0]];
    assert!(
        plain
            .evaluate_batch_into(&[0, 1], &mut rows, &|| Ok(()))
            .is_err()
    );
    assert!(rows.is_empty());
    assert!(plain.with_tiled_verification(usize::MAX).is_err());
    let graph = Qwen35::new(paged_hybrid_fixture(), 192)?;
    let tile_bytes = graph.tiled_prefill_scratch_bytes(8)?;
    let graph = graph.with_tiled_prefill(8, tile_bytes)?;
    let logits_bytes = graph.tiled_verification_scratch_bytes()?;
    assert!(graph.with_tiled_verification(logits_bytes - 1).is_err());
    let graph = verified(paged_hybrid_fixture())?;
    assert!(graph.with_chunkwise_gdn_prefill().is_err());
    let mut graph = Qwen35::new(paged_hybrid_fixture(), 192)?;
    let tile_bytes = graph.tiled_prefill_scratch_bytes(8)?;
    graph = graph.with_tiled_prefill(8, tile_bytes)?;
    graph.set_projection_executor(Some(Box::new(RefusingExecutor)));
    assert!(graph.with_tiled_verification(usize::MAX).is_err());
    let mut actual = verified(paged_hybrid_fixture())?;
    let initial = graph_bits(&actual)?;
    actual.set_projection_executor(Some(Box::new(RefusingExecutor)));
    assert!(
        actual
            .evaluate_batch_into(&[0, 1], &mut rows, &|| Ok(()))
            .is_err()
    );
    assert!(rows.is_empty());
    assert_eq!(graph_bits(&actual)?, initial);
    actual.set_projection_executor(None);
    for draft in [&[][..], &[0; 9][..], &[2][..]] {
        rows.push(vec![f32::NAN]);
        assert!(
            actual
                .evaluate_batch_into(draft, &mut rows, &|| Ok(()))
                .is_err()
        );
        assert!(rows.is_empty());
        assert_eq!(graph_bits(&actual)?, initial);
    }
    let checkpoint = actual.checkpoint()?;
    let checks = Cell::new(0);
    assert!(
        actual
            .evaluate_batch_into(&[0, 1, 0, 1], &mut rows, &|| {
                checks.set(checks.get() + 1);
                if checks.get() >= 10 {
                    Err("verification cancellation".into())
                } else {
                    Ok(())
                }
            })
            .is_err()
    );
    assert!(rows.is_empty());
    actual.restore(&checkpoint)?;
    assert_eq!(graph_bits(&actual)?, initial);
    let mut fresh = Qwen35::new(paged_hybrid_fixture(), 192)?;
    assert_eq!(actual.evaluate(1)?, fresh.evaluate(1)?);
    Ok(())
}

#[test]
fn tiled_verification_runtime_accepts_bonus_and_replays_every_rejected_prefix() -> Result<()> {
    let limits = Limits {
        context_tokens: 192,
        checkpoints: 0,
        checkpoint_bytes: 1_048_576,
    };
    for rejected in 0..=4 {
        let mut actual = Runtime::new(verified(paged_hybrid_fixture())?, limits)?;
        let mut expected = Runtime::new(Qwen35::new(paged_hybrid_fixture(), 192)?, limits)?;
        let generation = Generation {
            max_tokens: 1,
            stop_tokens: Vec::new(),
        };
        let cancellation = Cancellation::default();
        actual.generate(&[0, 1], &generation, &cancellation, |_| Ok(()))?;
        expected.generate(&[0, 1], &generation, &cancellation, |_| Ok(()))?;
        let mut probe = Qwen35::new(paged_hybrid_fixture(), 192)?;
        let mut logits = Vec::new();
        for &token in expected.tokens() {
            logits = probe.evaluate(token)?;
        }
        let mut draft = Vec::new();
        for _ in 0..4 {
            let token = bonsai_prism_runtime::runtime::greedy(&logits)?;
            draft.push(token);
            logits = probe.evaluate(token)?;
        }
        if rejected < draft.len() {
            draft[rejected] ^= 1;
        }
        let (actual_tokens, actual_stats) = actual.verify_draft_block(&draft, &cancellation)?;
        let (expected_tokens, expected_stats) =
            expected.verify_draft_block(&draft, &cancellation)?;
        assert_eq!(actual_tokens, expected_tokens);
        assert_eq!(
            actual_stats.draft_tokens_accepted,
            expected_stats.draft_tokens_accepted
        );
        assert_eq!(
            actual_stats.target_evaluations,
            4 + if rejected == 4 { 1 } else { rejected + 1 }
        );
        assert_eq!(actual.tokens(), expected.tokens());
        assert_eq!(
            graph_bits(actual.decoder())?,
            graph_bits(expected.decoder())?
        );
        let prompt = actual.tokens().to_vec();
        let mut a = Vec::new();
        let mut e = Vec::new();
        actual.generate(&prompt, &generation, &cancellation, |token| {
            a.push(token);
            Ok(())
        })?;
        expected.generate(&prompt, &generation, &cancellation, |token| {
            e.push(token);
            Ok(())
        })?;
        assert_eq!(a, e);
        assert_eq!(
            graph_bits(actual.decoder())?,
            graph_bits(expected.decoder())?
        );
    }
    Ok(())
}
