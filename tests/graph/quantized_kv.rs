//! Synthetic graph/state controls for separately selected lossy KV profiles.
//!
//! Two hidden coordinates project to one 256-coordinate attention head, then a
//! second block retains the original tiny FP32 GDN geometry. These fixtures do
//! not qualify actual Swift quality, performance or target-device memory use.

use super::*;
use bonsai_prism_runtime::graph::KvCachePolicy;
use std::cell::Cell;

const CONTEXT: usize = 70;
const GDN_BYTES: usize = 40;
const BUDGET: usize = 1_048_576;

fn profile_fixture() -> Model {
    let mut model = hybrid_fixture();
    for (key, value) in [
        ("qwen35.context_length", 70_u64),
        ("qwen35.attention.key_length", 256_u64),
        ("qwen35.attention.value_length", 256_u64),
    ] {
        model
            .metadata
            .insert(key.to_owned(), MetadataValue::from(value));
    }
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

fn tokens(count: usize) -> Vec<u32> {
    [0, 1, 1, 0, 1].into_iter().cycle().take(count).collect()
}

fn bits(values: Vec<f32>) -> Vec<u32> {
    values.into_iter().map(f32::to_bits).collect()
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
fn rotated_kv_graph_and_adapter_identities_reject_foreign_checkpoints() -> Result<()> {
    let mut exact = graph(KvCachePolicy::F16, MatvecBackend::Scalar)?;
    exact.evaluate(0)?;
    let exact_checkpoint = exact.checkpoint()?;
    for policy in [KvCachePolicy::Rotated4, KvCachePolicy::Rotated4Keys3Values] {
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
        let adapter = RankOneAdapter::from_model(factors, &base)?;
        let mut adapted = Qwen35::with_rank_one_adapter_and_kv(
            base,
            CONTEXT,
            BUDGET,
            MatvecBackend::Scalar,
            adapter,
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
fn rotated_kv_pages_restore_all_state_at_63_64_65_and_final_partial_page() -> Result<()> {
    for policy in [KvCachePolicy::Rotated4, KvCachePolicy::Rotated4Keys3Values] {
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
        assert_eq!(bits(block.evaluate(1)?), bits(scalar.evaluate(1)?));
        assert_eq!(graph_bits(&block)?, graph_bits(&scalar)?);
        let declined = graph(policy, MatvecBackend::Auto)?;
        let reserve = declined.tiled_prefill_scratch_bytes(8)?;
        assert!(declined.with_tiled_prefill(8, reserve).is_err());
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
        assert!(logits.is_empty());
        assert_eq!(graph_bits(&actual)?, before);
        actual.restore(&checkpoint)?;
        actual.evaluate(0)?;
        assert_eq!(graph_bits(&actual)?.len(), before.len() + 512);
    }
    Ok(())
}

#[test]
fn rotated_kv_runtime_delivery_and_cancellation_roll_back_across_page_seal() -> Result<()> {
    for policy in [KvCachePolicy::Rotated4, KvCachePolicy::Rotated4Keys3Values] {
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
        assert!(runtime.tokens().is_empty());
        assert_eq!(
            graph_bits(runtime.decoder())?,
            graph_bits(&graph(policy, MatvecBackend::Scalar)?)?
        );
    }
    Ok(())
}
