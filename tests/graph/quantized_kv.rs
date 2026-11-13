//! Synthetic graph/state controls for separately selected lossy KV profiles.
//!
//! Two hidden coordinates project to one 256-coordinate attention head, then a
//! second block retains the original tiny FP32 GDN geometry. The shared `RoPE`
//! geometry supports both source and pinned arithmetic. These fixtures do not
//! qualify actual Swift quality, performance or target-device memory use.

use super::*;
use bonsai_prism_runtime::graph::KvCachePolicy;
use std::cell::Cell;

const CONTEXT: usize = 70;
const GDN_BYTES: usize = 40;
const BUDGET: usize = 1_048_576;

#[expect(
    clippy::redundant_pub_crate,
    reason = "The private fixture is shared with sibling graph-test modules."
)]
pub(super) fn profile_fixture() -> Model {
    let mut model = hybrid_fixture();
    for (key, value) in [
        ("qwen35.context_length", 70_u64),
        ("qwen35.attention.key_length", 256_u64),
        ("qwen35.attention.value_length", 256_u64),
        ("qwen35.rope.dimension_count", 64_u64),
    ] {
        model
            .metadata
            .insert(key.to_owned(), MetadataValue::from(value));
    }
    model.metadata.insert(
        "qwen35.rope.dimension_sections".to_owned(),
        MetadataValue::from(vec![16_i64, 8, 8, 0]),
    );
    model.metadata.insert(
        "qwen35.rope.freq_base".to_owned(),
        MetadataValue::from(10_000_000.0_f32),
    );
    for (name, rows, salt) in [
        ("blk.0.attn_q.weight", 512, 3_u8),
        ("blk.0.attn_k.weight", 256, 5_u8),
        ("blk.0.attn_v.weight", 256, 11_u8),
    ] {
        let values: Vec<f32> = (0..rows * 2)
            .map(|index: usize| {
                let code = index.to_le_bytes()[0].wrapping_mul(salt) % 17;
                (f32::from(code) - 8.0) * 0.03125
            })
            .collect();
        model
            .tensors
            .insert(name.to_owned(), weight(name, &[2, rows], &values));
    }
    let name = "blk.0.attn_output.weight";
    let values: Vec<f32> = (0..512_usize)
        .map(|index| (f32::from(index.to_le_bytes()[0] % 7) - 2.0) * 0.003_906_25)
        .collect();
    model
        .tensors
        .insert(name.to_owned(), weight(name, &[256, 2], &values));
    for name in ["blk.0.attn_q_norm.weight", "blk.0.attn_k_norm.weight"] {
        model
            .tensors
            .insert(name.to_owned(), weight(name, &[256], &[1.0; 256]));
    }
    model
}

fn graph(policy: KvCachePolicy, backend: MatvecBackend) -> Result<Qwen35> {
    Qwen35::with_state_budget_and_kv(profile_fixture(), CONTEXT, BUDGET, backend, policy)
}

#[cfg(feature = "parallel-rows")]
fn parallel_profile_graph(policy: KvCachePolicy, workers: usize) -> Result<Qwen35> {
    const HEADS: usize = 12;
    const KV_HEADS: usize = 2;
    const HEAD: usize = 256;
    let mut model = profile_fixture();
    model.metadata.insert(
        "qwen35.attention.head_count".to_owned(),
        MetadataValue::from(12_u64),
    );
    model.metadata.insert(
        "qwen35.attention.head_count_kv".to_owned(),
        MetadataValue::from(2_u64),
    );
    for (name, rows, salt) in [
        ("blk.0.attn_q.weight", HEADS * HEAD * 2, 3_u8),
        ("blk.0.attn_k.weight", KV_HEADS * HEAD, 5_u8),
        ("blk.0.attn_v.weight", KV_HEADS * HEAD, 11_u8),
    ] {
        let values: Vec<f32> = (0..rows * 2)
            .map(|index| {
                let code = index.to_le_bytes()[0].wrapping_mul(salt) % 17;
                (f32::from(code) - 8.0) * 0.03125
            })
            .collect();
        model
            .tensors
            .insert(name.to_owned(), weight(name, &[2, rows], &values));
    }
    let name = "blk.0.attn_output.weight";
    let values: Vec<f32> = (0..HEADS * HEAD * 2)
        .map(|index| (f32::from(index.to_le_bytes()[0] % 7) - 2.0) * 0.003_906_25)
        .collect();
    model
        .tensors
        .insert(name.to_owned(), weight(name, &[HEADS * HEAD, 2], &values));
    let mut graph =
        Qwen35::with_state_budget_and_kv(model, CONTEXT, BUDGET, MatvecBackend::Auto, policy)?;
    graph.set_row_parallelism(bonsai_prism_runtime::kernels::RowParallelism::new(workers)?);
    Ok(graph)
}

fn tokens(count: usize) -> Vec<u32> {
    [0, 1, 1, 0, 1].into_iter().cycle().take(count).collect()
}

fn bits(values: Vec<f32>) -> Vec<u32> {
    values.into_iter().map(f32::to_bits).collect()
}

fn tile_profile(graph: Qwen35, verification: bool) -> Result<Qwen35> {
    let reserve = graph.tiled_prefill_scratch_bytes(8)?;
    let graph = graph.with_tiled_prefill(8, reserve)?;
    assert_eq!(graph.tiled_prefill_reserved_bytes(), reserve);
    if verification {
        let logits = graph.tiled_verification_scratch_bytes()?;
        graph.with_tiled_verification(logits)
    } else {
        Ok(graph)
    }
}

fn ordered_profile_fixture(pinned: bool, full_last: bool) -> Model {
    let mut model = profile_fixture();
    if pinned {
        model.identity = bonsai_prism_runtime::model::swift::MODEL_SHA256.to_owned();
    }
    if full_last {
        model.metadata.insert(
            "qwen35.attention.recurrent_layers".to_owned(),
            MetadataValue::from(vec![true, false]),
        );
        model.tensors = model
            .tensors
            .into_iter()
            .map(|(name, mut tensor)| {
                let name = match (name.strip_prefix("blk.0."), name.strip_prefix("blk.1.")) {
                    (Some(suffix), _) => format!("blk.1.{suffix}"),
                    (None, Some(suffix)) => format!("blk.0.{suffix}"),
                    (None, None) => name,
                };
                tensor.name.clone_from(&name);
                (name, tensor)
            })
            .collect();
    }
    model
}

#[test]
fn rotated_kv_tiled_prefill_retains_policy_bits_and_packed_page_storage() -> Result<()> {
    for policy in [KvCachePolicy::Rotated4, KvCachePolicy::Rotated4Keys3Values] {
        let empty = tile_profile(graph(policy, MatvecBackend::Scalar)?, false)?;
        assert!(empty.with_chunkwise_gdn_prefill().is_err());
        for pinned in [false, true] {
            for backend in [MatvecBackend::Scalar, MatvecBackend::Auto] {
                for full_last in [false, true] {
                    let build = || {
                        Qwen35::with_state_budget_and_kv(
                            ordered_profile_fixture(pinned, full_last),
                            CONTEXT,
                            BUDGET,
                            backend,
                            policy,
                        )
                    };
                    let mut expected = build()?;
                    let mut actual = tile_profile(build()?, false)?;
                    actual.set_last_layer_prefill_narrowing(true);
                    for token in tokens(63) {
                        expected.ingest(token)?;
                    }
                    actual.ingest_block(&tokens(63), &|| Ok(()))?;
                    assert_eq!(graph_bits(&actual)?, graph_bits(&expected)?);
                    assert_eq!(actual.kv_cache_policy(), policy);
                    assert_eq!(
                        actual.state_memory()?.kv_payload_bytes,
                        policy.layer_payload_bytes(256, 1, 63)?,
                    );
                    let checkpoint = actual.checkpoint()?;
                    for token in [1, 0] {
                        assert_eq!(
                            bits(actual.evaluate(token)?),
                            bits(expected.evaluate(token)?)
                        );
                    }
                    let sealed = graph_bits(&expected)?;
                    actual.restore(&checkpoint)?;
                    actual.ingest_block(&[1, 0], &|| Ok(()))?;
                    assert_eq!(graph_bits(&actual)?, sealed);
                    assert_eq!(
                        actual.state_memory()?.kv_payload_bytes,
                        policy.layer_payload_bytes(256, 1, 65)?,
                    );
                    assert!(actual.with_chunkwise_gdn_prefill().is_err());
                }
            }
        }
    }
    Ok(())
}

#[test]
fn rotated_kv_tiled_verification_matches_every_row_and_restored_causal_state() -> Result<()> {
    for policy in [KvCachePolicy::Rotated4, KvCachePolicy::Rotated4Keys3Values] {
        for pinned in [false, true] {
            for backend in [MatvecBackend::Scalar, MatvecBackend::Auto] {
                for base in [0, 63] {
                    let build = || {
                        Qwen35::with_state_budget_and_kv(
                            ordered_profile_fixture(pinned, true),
                            CONTEXT,
                            BUDGET,
                            backend,
                            policy,
                        )
                    };
                    let mut expected = build()?;
                    let mut actual = tile_profile(build()?, true)?;
                    actual.set_last_layer_prefill_narrowing(true);
                    if base > 0 {
                        for token in tokens(base) {
                            expected.ingest(token)?;
                        }
                        actual.ingest_block(&tokens(base), &|| Ok(()))?;
                    }
                    let checkpoint = actual.checkpoint()?;
                    let draft = [1, 0, 1, 1];
                    let reference = draft
                        .into_iter()
                        .map(|token| expected.evaluate(token).map(bits))
                        .collect::<Result<Vec<_>>>()?;
                    let mut rows = Vec::new();
                    actual.evaluate_batch_into(&draft, &mut rows, &|| Ok(()))?;
                    assert_eq!(rows.into_iter().map(bits).collect::<Vec<_>>(), reference);
                    assert_eq!(graph_bits(&actual)?, graph_bits(&expected)?);
                    actual.restore(&checkpoint)?;
                    let mut rows = Vec::new();
                    actual.evaluate_batch_into(&draft, &mut rows, &|| Ok(()))?;
                    assert_eq!(rows.into_iter().map(bits).collect::<Vec<_>>(), reference);
                    assert_eq!(bits(actual.evaluate(0)?), bits(expected.evaluate(0)?));
                    assert_eq!(graph_bits(&actual)?, graph_bits(&expected)?);
                }
            }
        }
    }
    Ok(())
}

#[test]
fn rotated_kv_tiled_cancellation_restores_pages_and_adapter_provenance() -> Result<()> {
    for policy in [KvCachePolicy::Rotated4, KvCachePolicy::Rotated4Keys3Values] {
        let build = |scale| -> Result<Qwen35> {
            let model = profile_fixture();
            let factors = rank_one_factors(
                &model,
                &["blk.0.attn_output.weight", "blk.1.ssm_out.weight"],
                0.125,
            );
            let adapter = RankOneAdapter::from_model(factors, &model)?;
            Qwen35::with_rank_one_adapter_and_kv(
                model,
                CONTEXT,
                BUDGET,
                MatvecBackend::Auto,
                adapter,
                scale,
                policy,
            )
        };
        let mut expected = build(1.3)?;
        let mut actual = tile_profile(build(1.3)?, true)?;
        for token in tokens(63) {
            expected.ingest(token)?;
        }
        actual.ingest_block(&tokens(63), &|| Ok(()))?;
        let before = graph_bits(&actual)?;
        assert_eq!(before, graph_bits(&expected)?);
        let checkpoint = actual.checkpoint()?;
        let checks = Cell::new(0);
        let mut rows = vec![vec![f32::NAN]];
        assert!(
            actual
                .evaluate_batch_into(&[1, 0, 1, 0], &mut rows, &|| {
                    checks.set(checks.get() + 1);
                    if checks.get() >= 10 {
                        Err(std::io::Error::other("cancel rotated tile").into())
                    } else {
                        Ok(())
                    }
                })
                .is_err()
        );
        assert_eq!(rows, Vec::<Vec<f32>>::new());
        actual.restore(&checkpoint)?;
        assert_eq!(graph_bits(&actual)?, before);
        assert_eq!(bits(actual.evaluate(1)?), bits(expected.evaluate(1)?));
        assert_eq!(graph_bits(&actual)?, graph_bits(&expected)?);
        actual.reset_with_adapter_scale(0.0)?;
        assert!(actual.restore(&checkpoint).is_err());
        assert_eq!(actual.batched_evaluation_capacity(), 8);
        assert!(actual.identity().contains(":cpu-tiled-verify-v1"));
        assert!(
            actual
                .identity()
                .contains(&format!("kv-{}-d256-rht3-v1", policy.name()))
        );
        let mut reference = build(0.0)?;
        let draft = [0, 1, 0, 1];
        let expected_rows = draft
            .into_iter()
            .map(|token| reference.evaluate(token).map(bits))
            .collect::<Result<Vec<_>>>()?;
        actual.evaluate_batch_into(&draft, &mut rows, &|| Ok(()))?;
        assert_eq!(
            rows.into_iter().map(bits).collect::<Vec<_>>(),
            expected_rows
        );
        assert_eq!(graph_bits(&actual)?, graph_bits(&reference)?);
    }
    Ok(())
}

fn adapted_runtime_graph(policy: KvCachePolicy, tiled: bool) -> Result<Qwen35> {
    let model = profile_fixture();
    let factors = rank_one_factors(
        &model,
        &["blk.0.attn_output.weight", "blk.1.ssm_out.weight"],
        0.125,
    );
    let adapter = RankOneAdapter::from_model(factors, &model)?;
    let graph = Qwen35::with_rank_one_adapter_and_kv(
        model,
        CONTEXT,
        BUDGET,
        MatvecBackend::Auto,
        adapter,
        1.3,
        policy,
    )?;
    if tiled {
        tile_profile(graph, false)
    } else {
        Ok(graph)
    }
}

#[test]
fn rotated_kv_tiled_latest_prefix_restores_committed_history_on_cancel_and_edit() -> Result<()> {
    let limits = Limits {
        context_tokens: CONTEXT,
        checkpoints: 0,
        checkpoint_bytes: BUDGET,
    };
    let generation = Generation {
        max_tokens: 1,
        stop_tokens: Vec::new(),
    };
    let prefill = Prefill { block_tokens: 8 };
    for policy in [KvCachePolicy::Rotated4, KvCachePolicy::Rotated4Keys3Values] {
        for cancel in [false, true] {
            let mut actual = Runtime::new(adapted_runtime_graph(policy, true)?, limits)?;
            actual.generate_with_prefill(
                &tokens(63),
                &generation,
                prefill,
                &Cancellation::default(),
                |_| Ok(()),
            )?;
            // The first response seals page 64. Its complete hybrid state must
            // survive a different conversation's failed tiled computation.
            let committed_tokens = actual.tokens().to_vec();
            let committed_state = graph_bits(actual.decoder())?;
            assert_eq!(committed_tokens.len(), 64);
            let mut edited = tokens(67);
            edited[31] ^= 1;
            actual.prepare_profile_request(Some(1.3), PromptReuse::Latest)?;
            let cancellation = Cancellation::default();
            let mut delivered = 0;
            let result = actual.generate_with_prefill(
                &edited,
                &Generation {
                    max_tokens: 2,
                    stop_tokens: Vec::new(),
                },
                prefill,
                &cancellation,
                |_| {
                    delivered += 1;
                    if cancel {
                        cancellation.cancel();
                        Ok(())
                    } else {
                        Err("injected edited-history stream failure".into())
                    }
                },
            );
            assert!(result.is_err());
            assert_eq!(delivered, 1);
            assert_eq!(actual.tokens(), committed_tokens);
            assert_eq!(graph_bits(actual.decoder())?, committed_state);
            assert!(!actual.is_faulted());
            let mut continued = committed_tokens;
            continued.extend_from_slice(&[0, 1, 0]);
            // First reuse the restored page, then replace the edited history.
            // A fresh untiled graph independently recomputes each full prompt.
            for (prompt, reused) in [(&continued, 64), (&edited, 0)] {
                actual.prepare_profile_request(Some(1.3), PromptReuse::Latest)?;
                let mut expected = Runtime::new(adapted_runtime_graph(policy, false)?, limits)?;
                let mut actual_output = Vec::new();
                let completion = actual.generate_with_prefill(
                    prompt,
                    &generation,
                    prefill,
                    &Cancellation::default(),
                    |token| {
                        actual_output.push(token);
                        Ok(())
                    },
                )?;
                let mut expected_output = Vec::new();
                expected.generate(prompt, &generation, &Cancellation::default(), |token| {
                    expected_output.push(token);
                    Ok(())
                })?;
                assert_eq!(completion.reused_tokens, reused);
                assert_eq!(actual_output, expected_output);
                assert_eq!(actual.tokens(), expected.tokens());
                assert_eq!(
                    graph_bits(actual.decoder())?,
                    graph_bits(expected.decoder())?
                );
                let state = actual.decoder().state_memory()?;
                assert_eq!(
                    state.kv_payload_bytes,
                    policy.layer_payload_bytes(256, 1, state.position)?
                );
            }
            assert!(actual.decoder().tiled_prefill_statistics().completed_tiles > 0);
            assert_eq!(actual.decoder().kv_cache_policy(), policy);
        }
    }
    Ok(())
}

#[cfg(feature = "parallel-rows")]
#[test]
fn parallel_rotated_kv_tiled_verification_retains_shared_head_workers() -> Result<()> {
    for policy in [KvCachePolicy::Rotated4, KvCachePolicy::Rotated4Keys3Values] {
        for workers in [1, 4, 16] {
            let mut expected = parallel_profile_graph(policy, 1)?;
            let mut actual = tile_profile(parallel_profile_graph(policy, workers)?, true)?;
            for token in tokens(63) {
                expected.ingest(token)?;
            }
            actual.ingest_block(&tokens(63), &|| Ok(()))?;
            assert_eq!(graph_bits(&actual)?, graph_bits(&expected)?);
            let checkpoint = actual.checkpoint()?;
            let draft = [1, 0, 1, 0];
            let expected_rows = draft
                .into_iter()
                .map(|token| expected.evaluate(token).map(bits))
                .collect::<Result<Vec<_>>>()?;
            let mut rows = Vec::new();
            actual.evaluate_batch_into(&draft, &mut rows, &|| Ok(()))?;
            assert_eq!(
                rows.into_iter().map(bits).collect::<Vec<_>>(),
                expected_rows
            );
            assert_eq!(graph_bits(&actual)?, graph_bits(&expected)?);
            actual.restore(&checkpoint)?;
            actual.ingest_block(&draft, &|| Ok(()))?;
            assert_eq!(graph_bits(&actual)?, graph_bits(&expected)?);
            assert_eq!(actual.row_workers(), workers);
        }
    }
    Ok(())
}

#[test]
fn rotated_kv_admission_charges_record_headers_and_unchanged_gdn_payload() -> Result<()> {
    for (policy, record_bytes, name) in [
        (KvCachePolicy::F16, 1024, "f16"),
        (KvCachePolicy::Rotated4, 264, "rsq4"),
        (KvCachePolicy::Rotated4Keys3Values, 232, "rsq4k3v"),
    ] {
        assert_eq!(policy.name(), name);
        let kv_bytes = policy.layer_payload_bytes(256, 1, CONTEXT)?;
        assert_eq!(kv_bytes, record_bytes * CONTEXT);
        let required = kv_bytes + GDN_BYTES;
        let accepted = Qwen35::with_state_budget_and_kv(
            profile_fixture(),
            CONTEXT,
            required,
            MatvecBackend::Scalar,
            policy,
        )?;
        assert_eq!(accepted.kv_cache_policy(), policy);
        let mut recurrent_elements = 0;
        accepted.visit_state(&mut |_, component, values| {
            if matches!(
                component,
                StateComponent::Convolution | StateComponent::Recurrent
            ) {
                recurrent_elements += values.len();
            }
            Ok(())
        })?;
        assert_eq!(recurrent_elements * size_of::<f32>(), GDN_BYTES);
        assert!(
            Qwen35::with_state_budget_and_kv(
                profile_fixture(),
                CONTEXT,
                required - 1,
                MatvecBackend::Scalar,
                policy,
            )
            .is_err()
        );
        for (head, heads, context) in [(0, 1, 1), (256, 0, 1), (256, 1, 0), (256, 1, usize::MAX)] {
            assert!(policy.layer_payload_bytes(head, heads, context).is_err());
        }
    }
    assert!(
        KvCachePolicy::Rotated4
            .layer_payload_bytes(2, 1, CONTEXT)
            .is_err()
    );
    assert!(
        Qwen35::with_state_budget_and_kv(
            fixture(false),
            16,
            BUDGET,
            MatvecBackend::Scalar,
            KvCachePolicy::Rotated4,
        )
        .is_err()
    );
    let default =
        Qwen35::with_state_budget(profile_fixture(), CONTEXT, BUDGET, MatvecBackend::Scalar)?;
    assert_eq!(default.kv_cache_policy(), KvCachePolicy::F16);
    assert_eq!(
        default.identity(),
        graph(KvCachePolicy::F16, MatvecBackend::Scalar)?.identity()
    );
    Ok(())
}

#[test]
fn kv_memory_observations_separate_payload_capacity_and_context_bound() -> Result<()> {
    for (policy, record_bytes) in [
        (KvCachePolicy::F16, 1024),
        (KvCachePolicy::Rotated4, 264),
        (KvCachePolicy::Rotated4Keys3Values, 232),
    ] {
        let mut decoder = graph(policy, MatvecBackend::Scalar)?;
        let empty = decoder.state_memory()?;
        assert_eq!(empty.position, 0);
        assert_eq!(empty.context_tokens, CONTEXT);
        assert_eq!(empty.kv_payload_bytes, 0);
        assert_eq!(empty.full_context_kv_payload_bytes, record_bytes * CONTEXT);
        assert_eq!(empty.recurrent_payload_bytes, GDN_BYTES);
        for token in tokens(65) {
            decoder.ingest(token)?;
        }
        let populated = decoder.state_memory()?;
        assert_eq!(populated.position, 65);
        assert_eq!(populated.kv_payload_bytes, record_bytes * 65);
        assert_eq!(populated.recurrent_payload_bytes, GDN_BYTES);
        assert!(populated.live_state_storage_bytes >= populated.kv_payload_bytes + GDN_BYTES);
        let snapshot = decoder.checkpoint()?;
        decoder.ingest(1)?;
        decoder.restore(&snapshot)?;
        assert_eq!(
            decoder.state_memory()?.kv_payload_bytes,
            populated.kv_payload_bytes
        );
        decoder.reset()?;
        let reset = decoder.state_memory()?;
        assert_eq!(reset.position, 0);
        assert_eq!(reset.kv_payload_bytes, 0);
        assert_eq!(reset.recurrent_payload_bytes, GDN_BYTES);
        assert_eq!(
            reset.full_context_kv_payload_bytes,
            empty.full_context_kv_payload_bytes
        );
        // Cleared tails and page tables retain capacity even though no token
        // payload remains. A snapshot elsewhere also retains its own pages.
        assert!(reset.live_state_storage_bytes > empty.live_state_storage_bytes);
    }
    Ok(())
}

#[test]
fn rotated_kv_graph_and_adapter_identities_reject_foreign_checkpoints() -> Result<()> {
    let mut exact = graph(KvCachePolicy::F16, MatvecBackend::Scalar)?;
    exact.evaluate(0)?;
    let exact_checkpoint = exact.checkpoint()?;
    for policy in [
        KvCachePolicy::Rotated4,
        KvCachePolicy::Rotated4FittedValues,
        KvCachePolicy::Rotated4Keys3Values,
    ] {
        let mut actual = graph(policy, MatvecBackend::Scalar)?;
        assert_ne!(actual.identity(), exact.identity());
        assert!(
            actual
                .identity()
                .contains(&format!("kv-{}-d256-rht3-v1", policy.name()))
        );
        let foreign = graph(policy, MatvecBackend::Scalar)?.checkpoint()?;
        let before = graph_bits(&actual)?;
        assert!(actual.restore(&foreign).is_err());
        assert!(actual.restore(&exact_checkpoint).is_err());
        assert_eq!(graph_bits(&actual)?, before);

        let base = profile_fixture();
        let factors = rank_one_factors(&base, &["blk.0.attn_output.weight"], 0.125);
        let correction = RankOneAdapter::from_model(factors, &base)?;
        let mut adapted = Qwen35::with_rank_one_adapter_and_kv(
            base,
            CONTEXT,
            BUDGET,
            MatvecBackend::Scalar,
            correction,
            1.3,
            policy,
        )?;
        assert_eq!(
            adapted.adapter_scale().map(f32::to_bits),
            Some(1.3_f32.to_bits())
        );
        assert_eq!(adapted.kv_cache_policy(), policy);
        assert!(
            adapted
                .identity()
                .contains(&format!("kv-{}-d256-rht3-v1", policy.name()))
        );
        assert!(
            adapted
                .identity()
                .contains(":rank-one-lora-v1:rank-one-fixture:3fa66666")
        );
        adapted.evaluate(0)?;
        let previous = adapted.checkpoint()?;
        adapted.reset_with_adapter_scale(0.0)?;
        assert_eq!(adapted.kv_cache_policy(), policy);
        assert!(
            adapted
                .identity()
                .contains(&format!("kv-{}-d256-rht3-v1", policy.name()))
        );
        assert!(adapted.restore(&previous).is_err());
        assert_eq!(
            graph_bits(&adapted)?,
            graph_bits(&graph(policy, MatvecBackend::Scalar)?)?
        );
    }
    Ok(())
}

#[test]
fn fitted_scale_graph_replays_page_boundary_and_rejects_norm_profile() -> Result<()> {
    let mut actual = graph(KvCachePolicy::Rotated4FittedValues, MatvecBackend::Scalar)?;
    let mut automatic = graph(KvCachePolicy::Rotated4FittedValues, MatvecBackend::Auto)?;
    let mut norm = graph(KvCachePolicy::Rotated4, MatvecBackend::Scalar)?;
    assert_ne!(actual.identity(), norm.identity());
    assert!(actual.identity().contains("kv-rsq4vf-d256-rht3-v1"));
    for token in tokens(63) {
        assert_eq!(
            bits(actual.evaluate(token)?),
            bits(automatic.evaluate(token)?)
        );
    }
    let checkpoint = actual.checkpoint()?;
    let frozen = graph_bits(&actual)?;
    let expected = [1, 0, 1]
        .into_iter()
        .map(|token| actual.evaluate(token).map(bits))
        .collect::<Result<Vec<_>>>()?;
    let completed = graph_bits(&actual)?;
    actual.restore(&checkpoint)?;
    assert_eq!(graph_bits(&actual)?, frozen);
    for (token, logits) in [1, 0, 1].into_iter().zip(expected) {
        assert_eq!(bits(actual.evaluate(token)?), logits);
    }
    assert_eq!(graph_bits(&actual)?, completed);
    let foreign = norm.checkpoint()?;
    assert!(actual.restore(&foreign).is_err());
    assert_eq!(graph_bits(&actual)?, completed);
    let norm_before = graph_bits(&norm)?;
    assert!(norm.restore(&checkpoint).is_err());
    assert_eq!(graph_bits(&norm)?, norm_before);
    Ok(())
}

#[test]
fn rotated_kv_pages_restore_all_state_at_63_64_65_and_final_partial_page() -> Result<()> {
    for policy in [
        KvCachePolicy::Rotated4,
        KvCachePolicy::Rotated4FittedValues,
        KvCachePolicy::Rotated4Keys3Values,
    ] {
        let mut actual = graph(policy, MatvecBackend::Scalar)?;
        let empty = actual.checkpoint()?;
        let empty_bits = graph_bits(&actual)?;
        let mut checkpoints = Vec::new();
        for (index, token) in tokens(CONTEXT).into_iter().enumerate() {
            actual.ingest(token)?;
            if matches!(index + 1, 63 | 64 | 65 | CONTEXT) {
                checkpoints.push((index + 1, actual.checkpoint()?, graph_bits(&actual)?));
            }
        }
        let final_state = graph_bits(&actual)?;
        assert!(actual.evaluate(0).is_err());
        assert_eq!(graph_bits(&actual)?, final_state);
        for (position, checkpoint, expected) in &checkpoints {
            actual.restore(checkpoint)?;
            assert_eq!(graph_bits(&actual)?, *expected);
            assert!(actual.evaluate(2).is_err());
            assert_eq!(graph_bits(&actual)?, *expected);
            if *position < CONTEXT {
                let logits = bits(actual.evaluate(1)?);
                let continued = graph_bits(&actual)?;
                actual.restore(checkpoint)?;
                assert_eq!(bits(actual.evaluate(1)?), logits);
                assert_eq!(graph_bits(&actual)?, continued);
            }
        }
        // Earlier snapshots remain unchanged after independent appends and
        // restoring the final short page. Empty restore releases logical state.
        for (_, checkpoint, expected) in &checkpoints {
            actual.restore(checkpoint)?;
            assert_eq!(graph_bits(&actual)?, *expected);
        }
        actual.restore(&empty)?;
        assert_eq!(graph_bits(&actual)?, empty_bits);
        let fresh_first = bits(graph(policy, MatvecBackend::Scalar)?.evaluate(0)?);
        assert_eq!(bits(actual.evaluate(0)?), fresh_first);
        actual.reset()?;
        assert_eq!(graph_bits(&actual)?, empty_bits);
    }
    Ok(())
}

#[test]
fn rotated_kv_scalar_auto_and_sequential_block_ingestion_preserve_profile_bits() -> Result<()> {
    for policy in [KvCachePolicy::Rotated4, KvCachePolicy::Rotated4Keys3Values] {
        let mut scalar = graph(policy, MatvecBackend::Scalar)?;
        let mut automatic = graph(policy, MatvecBackend::Auto)?;
        let mut block = graph(policy, MatvecBackend::Auto)?;
        assert_eq!(scalar.identity(), automatic.identity());
        let prompt = tokens(5);
        for token in &prompt {
            assert_eq!(
                bits(automatic.evaluate(*token)?),
                bits(scalar.evaluate(*token)?)
            );
            assert_eq!(graph_bits(&automatic)?, graph_bits(&scalar)?);
        }
        block.ingest_block(&prompt, &|| Ok(()))?;
        assert_eq!(graph_bits(&block)?, graph_bits(&scalar)?);
        let continuation = bits(scalar.evaluate(1)?);
        assert_eq!(bits(block.evaluate(1)?), continuation);
        assert_eq!(graph_bits(&block)?, graph_bits(&scalar)?);
        let selected = graph(policy, MatvecBackend::Auto)?;
        let reserve = selected.tiled_prefill_scratch_bytes(8)?;
        let mut tiled = selected.with_tiled_prefill(8, reserve)?;
        tiled.ingest_block(&prompt, &|| Ok(()))?;
        assert_eq!(bits(tiled.evaluate(1)?), continuation);
        assert_eq!(graph_bits(&tiled)?, graph_bits(&scalar)?);
    }
    Ok(())
}

#[test]
fn rotated_kv_late_logit_failure_preserves_committed_full_and_recurrent_state() -> Result<()> {
    for policy in [KvCachePolicy::Rotated4, KvCachePolicy::Rotated4Keys3Values] {
        let mut model = profile_fixture();
        for (name, dimensions, values) in [
            ("token_embd.weight", &[2, 2][..], &[0.0, 0.0, 1.0, 0.0][..]),
            ("output_norm.weight", &[2][..], &[f32::MAX, f32::MAX][..]),
            (
                "output.weight",
                &[2, 2][..],
                &[f32::MAX, 0.0, 0.0, f32::MAX][..],
            ),
        ] {
            model
                .tensors
                .insert(name.to_owned(), weight(name, dimensions, values));
        }
        let mut actual = Qwen35::with_state_budget_and_kv(
            model,
            CONTEXT,
            BUDGET,
            MatvecBackend::Scalar,
            policy,
        )?;
        actual.evaluate(0)?;
        let before = graph_bits(&actual)?;
        let checkpoint = actual.checkpoint()?;
        let mut logits = vec![1.0];
        assert!(actual.evaluate_into(1, &mut logits).is_err());
        assert_eq!(logits, Vec::<f32>::new());
        assert_eq!(graph_bits(&actual)?, before);
        actual.restore(&checkpoint)?;
        actual.evaluate(0)?;
        assert_eq!(graph_bits(&actual)?.len(), before.len() + 512);
    }
    Ok(())
}

#[test]
fn rotated_kv_runtime_delivery_and_cancellation_roll_back_across_page_seal() -> Result<()> {
    for policy in [
        KvCachePolicy::Rotated4,
        KvCachePolicy::Rotated4FittedValues,
        KvCachePolicy::Rotated4Keys3Values,
    ] {
        let mut runtime = Runtime::new(
            graph(policy, MatvecBackend::Scalar)?,
            Limits {
                context_tokens: CONTEXT,
                checkpoints: 2,
                checkpoint_bytes: BUDGET,
            },
        )?;
        let prefix = tokens(63);
        let stop = Generation {
            max_tokens: 1,
            stop_tokens: vec![0, 1],
        };
        runtime.generate(&prefix, &stop, &Cancellation::default(), |_| Ok(()))?;
        let before = graph_bits(runtime.decoder())?;
        let committed = runtime.tokens().to_vec();
        let changed = tokens(65);
        let options = Generation {
            max_tokens: 2,
            stop_tokens: vec![],
        };
        assert!(
            runtime
                .generate_with_prefill(
                    &changed,
                    &options,
                    Prefill { block_tokens: 4 },
                    &Cancellation::default(),
                    |_| Err(std::io::Error::other("synthetic consumer failure").into()),
                )
                .is_err()
        );
        assert_eq!(runtime.tokens(), committed);
        assert_eq!(graph_bits(runtime.decoder())?, before);
        assert!(!runtime.is_faulted());

        let cancellation = Cancellation::default();
        let emitted = Cell::new(0);
        assert!(
            runtime
                .generate(&changed, &options, &cancellation, |_| {
                    emitted.set(emitted.get() + 1);
                    cancellation.cancel();
                    Ok(())
                })
                .is_err()
        );
        assert_eq!(emitted.get(), 1);
        assert_eq!(runtime.tokens(), committed);
        assert_eq!(graph_bits(runtime.decoder())?, before);
        assert!(!runtime.is_faulted());
        runtime.generate(&changed, &stop, &Cancellation::default(), |_| Ok(()))?;
        assert_eq!(runtime.tokens(), changed);
        runtime.reset()?;
        assert_eq!(runtime.tokens(), [0_u32; 0]);
        assert_eq!(
            graph_bits(runtime.decoder())?,
            graph_bits(&graph(policy, MatvecBackend::Scalar)?)?
        );
    }
    Ok(())
}

#[cfg(feature = "parallel-rows")]
#[test]
fn parallel_rotated_kv_preserves_complete_state_logits_and_page_restore() -> Result<()> {
    for policy in [KvCachePolicy::Rotated4, KvCachePolicy::Rotated4Keys3Values] {
        let mut serial = parallel_profile_graph(policy, 1)?;
        let mut parallel = parallel_profile_graph(policy, 4)?;
        assert_eq!(serial.identity(), parallel.identity());
        let mut checkpoints = None;
        for (index, token) in tokens(65).into_iter().enumerate() {
            assert_eq!(
                bits(parallel.evaluate(token)?),
                bits(serial.evaluate(token)?)
            );
            assert_eq!(graph_bits(&parallel)?, graph_bits(&serial)?);
            if index == 62 {
                checkpoints = Some((serial.checkpoint()?, parallel.checkpoint()?));
            }
        }
        let (serial_checkpoint, parallel_checkpoint) =
            checkpoints.ok_or_else(|| std::io::Error::other("missing partial-page checkpoint"))?;
        serial.restore(&serial_checkpoint)?;
        parallel.restore(&parallel_checkpoint)?;
        assert_eq!(graph_bits(&parallel)?, graph_bits(&serial)?);
        for token in [1, 0, 1] {
            assert_eq!(
                bits(parallel.evaluate(token)?),
                bits(serial.evaluate(token)?)
            );
            assert_eq!(graph_bits(&parallel)?, graph_bits(&serial)?);
        }
        assert_eq!(parallel.row_workers(), 4);
    }
    Ok(())
}

#[cfg(feature = "parallel-rows")]
#[test]
fn parallel_rotated_kv_delivery_and_cancellation_restore_complete_page_boundary() -> Result<()> {
    for policy in [KvCachePolicy::Rotated4, KvCachePolicy::Rotated4Keys3Values] {
        let limits = Limits {
            context_tokens: CONTEXT,
            checkpoints: 0,
            checkpoint_bytes: BUDGET,
        };
        let mut runtime = Runtime::new(parallel_profile_graph(policy, 4)?, limits)?;
        let mut serial = Runtime::new(parallel_profile_graph(policy, 1)?, limits)?;
        let prefix = tokens(63);
        let stop = Generation {
            max_tokens: 1,
            stop_tokens: vec![0, 1],
        };
        runtime.generate(&prefix, &stop, &Cancellation::default(), |_| Ok(()))?;
        serial.generate(&prefix, &stop, &Cancellation::default(), |_| Ok(()))?;
        let before = graph_bits(runtime.decoder())?;
        let committed = runtime.tokens().to_vec();
        assert_eq!(before, graph_bits(serial.decoder())?);
        let changed = tokens(65);
        let options = Generation {
            max_tokens: 2,
            stop_tokens: vec![],
        };
        for cancel in [false, true] {
            let cancellation = Cancellation::default();
            let emitted = Cell::new(0);
            assert!(
                runtime
                    .generate(&changed, &options, &cancellation, |_| {
                        emitted.set(emitted.get() + 1);
                        if cancel {
                            cancellation.cancel();
                            Ok(())
                        } else {
                            Err(std::io::Error::other("synthetic parallel consumer failure").into())
                        }
                    })
                    .is_err()
            );
            assert_eq!(emitted.get(), 1);
            assert_eq!(runtime.tokens(), committed);
            assert_eq!(graph_bits(runtime.decoder())?, before);
            assert!(!runtime.is_faulted());
        }
        runtime.generate(&changed, &stop, &Cancellation::default(), |_| Ok(()))?;
        serial.generate(&changed, &stop, &Cancellation::default(), |_| Ok(()))?;
        assert_eq!(runtime.tokens(), serial.tokens());
        assert_eq!(
            graph_bits(runtime.decoder())?,
            graph_bits(serial.decoder())?
        );
        assert_eq!(runtime.decoder().row_workers(), 4);
    }
    Ok(())
}

fn profile_runtime(policy: KvCachePolicy, scale: f32) -> Result<Runtime<Qwen35>> {
    let base = profile_fixture();
    let factors = rank_one_factors(
        &base,
        &[
            "blk.0.attn_output.weight",
            "blk.1.ssm_out.weight",
            "blk.1.ffn_down.weight",
        ],
        0.125,
    );
    let adapter = RankOneAdapter::from_model(factors, &base)?;
    Runtime::new(
        Qwen35::with_rank_one_adapter_and_kv(
            base,
            CONTEXT,
            BUDGET,
            MatvecBackend::Auto,
            adapter,
            scale,
            policy,
        )?,
        Limits {
            context_tokens: CONTEXT,
            checkpoints: 0,
            checkpoint_bytes: BUDGET,
        },
    )
}

fn compare_profile_request(
    runtime: &mut Runtime<Qwen35>,
    sampler: &mut ProfileSampler,
    prompt: &[u32],
    scale: f32,
) -> Result<()> {
    let options = Generation {
        max_tokens: 2,
        stop_tokens: vec![],
    };
    let policy = runtime.decoder().kv_cache_policy();
    let previous = runtime.tokens().to_vec();
    let previous_scale = runtime.decoder().adapter_scale().map(f32::to_bits);
    let before = graph_bits(runtime.decoder())?;
    runtime.prepare_profile_request(Some(scale), PromptReuse::Latest)?;
    let unchanged = previous_scale == Some(scale.to_bits());
    if unchanged {
        assert_eq!(runtime.tokens(), previous);
        assert_eq!(graph_bits(runtime.decoder())?, before);
    } else {
        assert_eq!(runtime.tokens().len(), 0);
        assert!(graph_bits(runtime.decoder())?.iter().all(|&bits| bits == 0));
    }
    assert_eq!(runtime.decoder().kv_cache_policy(), policy);
    assert_eq!(
        runtime.decoder().adapter_scale().map(f32::to_bits),
        Some(scale.to_bits())
    );
    let mut reference_sampler = sampler.clone();
    profile_request_failures(runtime, prompt, &options, sampler)?;
    let mut reference = profile_runtime(policy, scale)?;
    let mut expected = Vec::new();
    reference.generate_sampled(
        prompt,
        &options,
        Prefill { block_tokens: 4 },
        &Cancellation::default(),
        &mut reference_sampler,
        |token| {
            expected.push(token);
            Ok(())
        },
    )?;
    let mut actual = Vec::new();
    let completion = runtime.generate_sampled(
        prompt,
        &options,
        Prefill { block_tokens: 4 },
        &Cancellation::default(),
        sampler,
        |token| {
            actual.push(token);
            Ok(())
        },
    )?;
    assert_eq!(actual, expected);
    assert_eq!(sampler.logits, reference_sampler.logits);
    assert_eq!(runtime.tokens(), reference.tokens());
    assert_eq!(
        graph_bits(runtime.decoder())?,
        graph_bits(reference.decoder())?
    );
    assert_eq!(
        completion.reused_tokens,
        if unchanged && prompt.starts_with(&previous) {
            previous.len()
        } else {
            0
        }
    );
    assert_eq!(runtime.checkpoint_count(), 0);
    Ok(())
}

#[test]
fn rotated_kv_latest_prefix_and_adapter_changes_preserve_selected_profile() -> Result<()> {
    use bonsai_prism_runtime::sampling::{ReferenceSampler, Sampling};

    for policy in [KvCachePolicy::Rotated4, KvCachePolicy::Rotated4Keys3Values] {
        let mut runtime = profile_runtime(policy, 1.3)?;
        let mut sampler = ProfileSampler {
            inner: ReferenceSampler::new(Sampling::default(), 42)?,
            logits: Vec::new(),
        };
        // Include a page-sealing extension, an edited history, a shortened
        // history and signed-zero/change-back adapter transitions. Every
        // reference uses the same selected lossy profile, never FP16 parity.
        for (length, scale, edit) in [
            (61, 1.3_f32, false),
            (64, 1.3, false),
            (64, 1.3, true),
            (3, 1.3, false),
            (3, 0.0, false),
            (3, -0.0, false),
            (3, 1.3, false),
        ] {
            let mut prompt = if length == 64 && !edit {
                let mut extended = runtime.tokens().to_vec();
                extended.resize(length, 1);
                extended
            } else {
                tokens(length)
            };
            if edit {
                prompt[0] = 1 - prompt[0];
            }
            compare_profile_request(&mut runtime, &mut sampler, &prompt, scale)?;
        }
        let committed = runtime.tokens().to_vec();
        let before = graph_bits(runtime.decoder())?;
        for invalid in [None, Some(f32::NAN), Some(-1.0)] {
            assert!(
                runtime
                    .prepare_profile_request(invalid, PromptReuse::Latest)
                    .is_err()
            );
            assert_eq!(runtime.tokens(), committed);
            assert_eq!(graph_bits(runtime.decoder())?, before);
            assert_eq!(runtime.decoder().kv_cache_policy(), policy);
            assert!(!runtime.is_faulted());
        }
        runtime.prepare_profile_request(Some(1.3), PromptReuse::Disabled)?;
        assert_eq!(runtime.tokens().len(), 0);
        assert_eq!(runtime.decoder().kv_cache_policy(), policy);
        runtime.reset()?;
        assert_eq!(runtime.tokens().len(), 0);
        assert_eq!(runtime.decoder().kv_cache_policy(), policy);
    }
    Ok(())
}
