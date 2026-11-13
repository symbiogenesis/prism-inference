//! A larger aggregate state scope only for the exact Swift K4/V4 geometry.
use super::{KvCachePolicy, Metadata, integer, optional_integer};
use crate::{Result, invalid, model::swift};

const CONTEXT_TOKENS: usize = 262_144;
const KV_TOKEN_BYTES: usize = 16 * 2 * 4 * (256 / 2 + 4);
const RECURRENT_BYTES: usize = 48 * (128 * 128 * 48 + 3 * (2 * 128 * 16 + 128 * 48)) * 4;
const STATE_TOKEN_BYTES: usize = KV_TOKEN_BYTES + 24 * 4 + 128;

// Complete selected live/checkpoint allowance, including sequential scores and
// bookkeeping, rather than a new cap on any individual allocation or scratch.
pub(crate) const SWIFT_RSQ4_FULL_STATE_BYTES: usize =
    CONTEXT_TOKENS * STATE_TOKEN_BYTES + RECURRENT_BYTES;

pub(super) fn ceiling(
    identity: &str,
    metadata: &Metadata,
    context_tokens: usize,
    cache: KvCachePolicy,
) -> Result<usize> {
    if !matches!(identity, swift::MODEL_SHA256 | swift::REPACKED_PTQ1_SHA256)
        || cache != KvCachePolicy::Rotated4
        || !(1..=CONTEXT_TOKENS).contains(&context_tokens)
        || optional_integer(metadata, "qwen35.nextn_predict_layers", 0)? != 0
    {
        return Err(invalid("extended Swift state scope rejected"));
    }
    for (key, expected) in [
        ("qwen35.block_count", 64),
        ("qwen35.context_length", CONTEXT_TOKENS),
        ("qwen35.embedding_length", 5_120),
        ("qwen35.feed_forward_length", 17_408),
        ("qwen35.attention.head_count", 24),
        ("qwen35.attention.head_count_kv", 4),
        ("qwen35.attention.key_length", 256),
        ("qwen35.ssm.conv_kernel", 4),
        ("qwen35.ssm.state_size", 128),
        ("qwen35.ssm.group_count", 16),
        ("qwen35.ssm.time_step_rank", 48),
        ("qwen35.full_attention_interval", 4),
    ] {
        if integer(metadata, key)? != expected {
            return Err(invalid("extended Swift state geometry rejected"));
        }
    }
    if let Some(flags) = metadata.get("qwen35.attention.recurrent_layers") {
        let flags = flags
            .as_array()
            .ok_or_else(|| invalid("extended Swift recurrence map rejected"))?;
        if flags.len() != 64
            || flags
                .iter()
                .enumerate()
                .any(|(index, value)| value.as_bool() != Some(!(index + 1).is_multiple_of(4)))
        {
            return Err(invalid("extended Swift recurrence map rejected"));
        }
    }
    // All multiplicands are bounded above by the exact compiled geometry.
    Ok(context_tokens * STATE_TOKEN_BYTES + RECURRENT_BYTES)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bonsai_gguf::{MetadataArray, MetadataValue};

    fn metadata() -> Metadata {
        [
            ("qwen35.block_count", 64),
            ("qwen35.context_length", 262_144_u64),
            ("qwen35.embedding_length", 5_120),
            ("qwen35.feed_forward_length", 17_408),
            ("qwen35.attention.head_count", 24),
            ("qwen35.attention.head_count_kv", 4),
            ("qwen35.attention.key_length", 256),
            ("qwen35.ssm.conv_kernel", 4),
            ("qwen35.ssm.state_size", 128),
            ("qwen35.ssm.group_count", 16),
            ("qwen35.ssm.time_step_rank", 48),
            ("qwen35.full_attention_interval", 4),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_owned(), MetadataValue::U64(value)))
        .collect()
    }

    #[test]
    fn full_context_state_scope_is_geometry_derived_and_preserves_generic_cap() -> Result<()> {
        let metadata = metadata();
        assert_eq!(KV_TOKEN_BYTES, 16_896);
        assert_eq!(RECURRENT_BYTES, 156_893_184);
        assert_eq!(SWIFT_RSQ4_FULL_STATE_BYTES, 4_644_798_464);
        assert_eq!(super::super::MAX_STATE_BYTES, 4 * 1024 * 1024 * 1024);
        for identity in [swift::MODEL_SHA256, swift::REPACKED_PTQ1_SHA256] {
            assert_eq!(
                ceiling(identity, &metadata, CONTEXT_TOKENS, KvCachePolicy::Rotated4)?,
                SWIFT_RSQ4_FULL_STATE_BYTES
            );
            for tokens in [0, CONTEXT_TOKENS + 1, CONTEXT_TOKENS + 256, usize::MAX] {
                assert!(ceiling(identity, &metadata, tokens, KvCachePolicy::Rotated4).is_err());
            }
        }
        assert!(
            ceiling(
                "other model",
                &metadata,
                CONTEXT_TOKENS,
                KvCachePolicy::Rotated4
            )
            .is_err()
        );
        for cache in [KvCachePolicy::F16, KvCachePolicy::Rotated4Keys3Values] {
            assert!(ceiling(swift::MODEL_SHA256, &metadata, CONTEXT_TOKENS, cache).is_err());
        }
        Ok(())
    }

    #[test]
    fn state_scope_rejects_changed_counts_recurrence_and_mtp() {
        let original = metadata();
        for key in original.keys() {
            let mut changed = original.clone();
            changed.insert(key.clone(), MetadataValue::U64(1));
            assert!(
                ceiling(
                    swift::MODEL_SHA256,
                    &changed,
                    CONTEXT_TOKENS,
                    KvCachePolicy::Rotated4
                )
                .is_err()
            );
        }
        let mut changed = original.clone();
        changed.insert("qwen35.nextn_predict_layers".into(), MetadataValue::U64(1));
        assert!(
            ceiling(
                swift::MODEL_SHA256,
                &changed,
                CONTEXT_TOKENS,
                KvCachePolicy::Rotated4
            )
            .is_err()
        );
        changed = original;
        changed.insert(
            "qwen35.attention.recurrent_layers".into(),
            MetadataValue::Array(MetadataArray::Bool(vec![true; 64])),
        );
        assert!(
            ceiling(
                swift::MODEL_SHA256,
                &changed,
                CONTEXT_TOKENS,
                KvCachePolicy::Rotated4
            )
            .is_err()
        );
    }

    #[test]
    fn generic_constructors_and_excess_selected_budgets_reject_before_tensor_loading() -> Result<()>
    {
        use super::super::Qwen35;
        use crate::{kernels::MatvecBackend, model::Model};
        let model = || Model {
            identity: swift::MODEL_SHA256.to_owned(),
            metadata: metadata(),
            tensors: std::collections::BTreeMap::new(),
        };
        for cache in [
            KvCachePolicy::F16,
            KvCachePolicy::Rotated4,
            KvCachePolicy::Rotated4Keys3Values,
        ] {
            let error = Qwen35::with_state_budget_and_kv(
                model(),
                CONTEXT_TOKENS,
                super::super::MAX_STATE_BYTES + 1,
                MatvecBackend::Scalar,
                cache,
            )
            .err()
            .ok_or_else(|| invalid("generic graph unexpectedly accepted extended state"))?;
            assert_eq!(
                error.to_string(),
                "invalid native live-state payload budget"
            );
        }
        let error = Qwen35::with_swift_state_budget_and_kv(
            model(),
            CONTEXT_TOKENS,
            SWIFT_RSQ4_FULL_STATE_BYTES + 1,
            MatvecBackend::Scalar,
            KvCachePolicy::Rotated4,
            None,
        )
        .err()
        .ok_or_else(|| invalid("Swift graph unexpectedly accepted excess state"))?;
        assert_eq!(
            error.to_string(),
            "invalid native live-state payload budget"
        );
        Ok(())
    }
}
