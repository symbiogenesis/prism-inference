use super::*;

fn input(width: usize, salt: u16) -> Result<Vec<f32>> {
    (0..width)
        .map(|index| {
            let sample = (u16::try_from(index)? % 31 + salt) % 47;
            Ok((f32::from(sample) - 23.0) / 19.0)
        })
        .collect()
}

fn defensive_value_result(
    values: &KvCache,
    current_value: &[f32],
    projected: &[f32],
    head: usize,
    kv_head: usize,
    probabilities: &[f32],
) -> Result<[f32; HEAD_DIM]> {
    let cache = values
        .packed
        .as_ref()
        .ok_or_else(|| invalid("missing values"))?;
    let mut output = [0.0; HEAD_DIM];
    for (&weight, row) in probabilities.iter().zip(cache.rows()) {
        cache
            .quantizer
            .accumulate(cache.record(row, kv_head), weight, &mut output)?;
    }
    let mut current = vec![0; cache.quantizer.encoded_len()];
    cache.quantizer.encode(
        &current_value[kv_head * HEAD_DIM..(kv_head + 1) * HEAD_DIM],
        &mut current,
    )?;
    cache.quantizer.accumulate(
        &current,
        *probabilities
            .last()
            .ok_or_else(|| invalid("missing current weight"))?,
        &mut output,
    )?;
    cache.quantizer.inverse(&mut output)?;
    super::super::mul_sigmoid_in_place(
        &mut output,
        super::super::attention_gate(projected, head, HEAD_DIM),
    );
    Ok(output)
}

fn defensive_probabilities(
    keys: &PackedCache,
    current: &[u8],
    queries: &[f32],
    head: usize,
    heads_per_kv: usize,
    arithmetic: super::super::Arithmetic,
) -> Result<Vec<f32>> {
    let mut query = [0.0; HEAD_DIM];
    keys.quantizer
        .rotate(&queries[head * HEAD_DIM..(head + 1) * HEAD_DIM], &mut query)?;
    let mut expected = Vec::new();
    for row in keys.rows().chain(std::iter::once(current)) {
        expected.push(
            keys.quantizer
                .score(&query, keys.record(row, head / heads_per_kv))?
                * (1.0 / 16.0),
        );
    }
    match arithmetic {
        super::super::Arithmetic::Source => crate::kernels::softmax_round_f16(&mut expected)?,
        super::super::Arithmetic::PinnedSwift => {
            crate::kernels::pinned_attention::softmax_rotated_kv(&mut expected)?;
        }
    }
    Ok(expected)
}

#[test]
fn fitted_scale_values_preserve_key_bytes_and_attention_probabilities() -> Result<()> {
    let (baseline_keys, baseline_values) = populated_caches(KvCachePolicy::Rotated4, 2, 65, true)?;
    let (keys, values) = populated_caches(KvCachePolicy::Rotated4FittedValues, 2, 65, true)?;
    assert_eq!(packed_bytes(&keys), packed_bytes(&baseline_keys));
    assert_ne!(packed_bytes(&values), packed_bytes(&baseline_values));
    let packed_keys = keys.packed.as_ref().ok_or_else(|| invalid("keys"))?;
    let packed_values = values.packed.as_ref().ok_or_else(|| invalid("values"))?;
    let old_keys = baseline_keys
        .packed
        .as_ref()
        .ok_or_else(|| invalid("keys"))?;
    assert_eq!(packed_keys.quantizer.identity().version, 1);
    assert_eq!(packed_values.quantizer.identity().version, 2);
    let current = input(2 * HEAD_DIM, 77)?;
    let queries = input(4 * HEAD_DIM, 19)?;
    let mut encoded = vec![0; packed_keys.row_bytes];
    for (row, output) in current
        .as_chunks::<HEAD_DIM>()
        .0
        .iter()
        .zip(encoded.as_chunks_mut::<132>().0)
    {
        packed_keys.quantizer.encode(row, output)?;
    }
    for head in 0..4 {
        let expected = defensive_probabilities(
            old_keys,
            &encoded,
            &queries,
            head,
            2,
            super::super::Arithmetic::PinnedSwift,
        )?;
        let actual = defensive_probabilities(
            packed_keys,
            &encoded,
            &queries,
            head,
            2,
            super::super::Arithmetic::PinnedSwift,
        )?;
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }
    assert_eq!(
        KvCachePolicy::Rotated4FittedValues.layer_payload_bytes(256, 4, 8192)? * 16,
        132 * 1024 * 1024
    );
    Ok(())
}

#[test]
fn x1_prepared_kv_attention_matches_defensive_scores_across_pages() -> Result<()> {
    let heads = 12;
    for policy in [KvCachePolicy::Rotated4, KvCachePolicy::Rotated4Keys3Values] {
        for scalar_only in [false, true] {
            let (mut keys, mut values) = populated_caches(policy, 2, 65, scalar_only)?;
            let before_keys = packed_bytes(&keys);
            let before_values = packed_bytes(&values);
            let queries = input(heads * HEAD_DIM, 9)?;
            let projected = input(2 * heads * HEAD_DIM, 11)?;
            let current_key = input(keys.width, 31)?;
            let current_value = input(values.width, 41)?;
            let key_cache = keys
                .packed
                .as_ref()
                .ok_or_else(|| invalid("missing keys"))?;
            let mut current = vec![0; key_cache.row_bytes];
            for (key, output) in current_key
                .as_chunks::<HEAD_DIM>()
                .0
                .iter()
                .zip(current.chunks_exact_mut(key_cache.quantizer.encoded_len()))
            {
                key_cache.quantizer.encode(key, output)?;
            }
            for arithmetic in [
                super::super::Arithmetic::Source,
                super::super::Arithmetic::PinnedSwift,
            ] {
                let mut attended = vec![0.0; queries.len()];
                let mut scores = vec![0.0; 66];
                attend(
                    &mut keys,
                    &mut values,
                    &queries,
                    &projected,
                    &current_key,
                    &current_value,
                    &mut attended,
                    &mut scores,
                    heads,
                    1.0 / 16.0,
                    arithmetic,
                    &RowParallelism::default(),
                )?;
                let key_cache = keys
                    .packed
                    .as_ref()
                    .ok_or_else(|| invalid("missing keys"))?;
                for head in 0..heads {
                    let expected = defensive_probabilities(
                        key_cache, &current, &queries, head, 6, arithmetic,
                    )?;
                    // A serial task overwrites its one row after each complete
                    // head. Its final probabilities still expose exact bits.
                    if head + 1 == heads {
                        assert_eq!(
                            scores
                                .iter()
                                .map(|value| value.to_bits())
                                .collect::<Vec<_>>(),
                            expected
                                .iter()
                                .map(|value| value.to_bits())
                                .collect::<Vec<_>>()
                        );
                    }
                    let output = defensive_value_result(
                        &values,
                        &current_value,
                        &projected,
                        head,
                        head / 6,
                        &expected,
                    )?;
                    assert_eq!(
                        attended[head * HEAD_DIM..(head + 1) * HEAD_DIM]
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>(),
                        output.map(f32::to_bits),
                    );
                }
                assert_eq!(packed_bytes(&keys), before_keys);
                assert_eq!(packed_bytes(&values), before_values);
            }
        }
    }
    Ok(())
}

fn packed_bytes(cache: &KvCache) -> Vec<u8> {
    cache
        .packed
        .as_ref()
        .into_iter()
        .flat_map(|cache| cache.rows())
        .flatten()
        .copied()
        .collect()
}

#[test]
fn packed_attention_reuses_current_encoding_for_commit_without_extra_records() -> Result<()> {
    for policy in [
        KvCachePolicy::Rotated4,
        KvCachePolicy::Rotated4FittedValues,
        KvCachePolicy::Rotated4Keys3Values,
    ] {
        for scalar in [true, false] {
            let (mut keys, mut values) = populated_caches(policy, 2, 63, scalar)?;
            let current_key = input(keys.width, 31)?;
            let current_value = input(values.width, 41)?;
            let mut expected_keys = keys.copy()?;
            let mut expected_values = values.copy()?;
            expected_keys.prepare_append(&current_key)?;
            expected_values.prepare_append(&current_value)?;
            expected_keys.append(&current_key);
            expected_values.append(&current_value);
            let queries = input(4 * HEAD_DIM, 9)?;
            let projected = input(8 * HEAD_DIM, 11)?;
            let mut output = vec![0.0; queries.len()];
            attend(
                &mut keys,
                &mut values,
                &queries,
                &projected,
                &current_key,
                &current_value,
                &mut output,
                &mut [0.0; 64],
                4,
                1.0 / 16.0,
                super::super::Arithmetic::PinnedSwift,
                &RowParallelism::default(),
            )?;
            for cache in [&keys, &values] {
                let packed = cache.packed.as_ref().ok_or_else(|| invalid("cache"))?;
                assert!(packed.prepared_ready);
                let copied = cache.copy()?;
                let snapshot = copied.packed.as_ref().ok_or_else(|| invalid("snapshot"))?;
                assert!(snapshot.prepared.is_empty());
                assert!(!snapshot.prepared_ready);
            }
            let key_pointer = keys
                .packed
                .as_ref()
                .ok_or_else(|| invalid("keys"))?
                .prepared
                .as_ptr();
            let value_pointer = values
                .packed
                .as_ref()
                .ok_or_else(|| invalid("values"))?
                .prepared
                .as_ptr();
            keys.prepare_staged_append(&current_key)?;
            values.prepare_staged_append(&current_value)?;
            keys.append(&current_key);
            values.append(&current_value);
            assert_eq!(packed_bytes(&keys), packed_bytes(&expected_keys));
            assert_eq!(packed_bytes(&values), packed_bytes(&expected_values));
            assert_eq!(keys.len, expected_keys.len);
            assert_eq!(values.len, expected_values.len);
            for (cache, pointer) in [(&keys, key_pointer), (&values, value_pointer)] {
                let packed = cache.packed.as_ref().ok_or_else(|| invalid("cache"))?;
                assert_eq!(packed.prepared.as_ptr(), pointer);
                assert!(!packed.prepared_ready);
                assert_eq!(packed.pages.len(), 1);
            }
        }
    }
    Ok(())
}

#[test]
fn packed_attention_failed_attempt_and_narrowing_cannot_publish_stale_encoding() -> Result<()> {
    for policy in [
        KvCachePolicy::Rotated4,
        KvCachePolicy::Rotated4FittedValues,
        KvCachePolicy::Rotated4Keys3Values,
    ] {
        let (mut keys, mut values) = populated_caches(policy, 2, 2, false)?;
        let old_key = input(keys.width, 31)?;
        let old_value = input(values.width, 41)?;
        let next_key = input(keys.width, 33)?;
        let next_value = input(values.width, 43)?;
        let queries = input(4 * HEAD_DIM, 9)?;
        let mut projected = input(8 * HEAD_DIM, 11)?;
        let mut output = vec![0.0; queries.len()];
        for fail in [false, true] {
            let frozen_keys = packed_bytes(&keys);
            let frozen_values = packed_bytes(&values);
            projected[HEAD_DIM] = if fail { f32::NAN } else { 0.0 };
            let mut scores = vec![0.0; keys.len / keys.width + 1];
            let result = attend(
                &mut keys,
                &mut values,
                &queries,
                &projected,
                &old_key,
                &old_value,
                &mut output,
                &mut scores,
                4,
                1.0 / 16.0,
                super::super::Arithmetic::PinnedSwift,
                &RowParallelism::default(),
            );
            assert_eq!(result.is_err(), fail);
            assert_eq!(packed_bytes(&keys), frozen_keys);
            assert_eq!(packed_bytes(&values), frozen_values);
            if fail {
                assert!(
                    !keys
                        .packed
                        .as_ref()
                        .ok_or_else(|| invalid("keys"))?
                        .prepared_ready
                );
                assert!(
                    !values
                        .packed
                        .as_ref()
                        .ok_or_else(|| invalid("values"))?
                        .prepared_ready
                );
            }
            let mut expected_keys = keys.copy()?;
            let mut expected_values = values.copy()?;
            expected_keys.prepare_append(&next_key)?;
            expected_values.prepare_append(&next_value)?;
            expected_keys.append(&next_key);
            expected_values.append(&next_value);
            // A new attempt may narrow its final layer to KV-only work. Its
            // invalidation must force current projections to encode afresh.
            keys.invalidate_prepared();
            values.invalidate_prepared();
            keys.prepare_staged_append(&next_key)?;
            values.prepare_staged_append(&next_value)?;
            keys.append(&next_key);
            values.append(&next_value);
            assert_eq!(packed_bytes(&keys), packed_bytes(&expected_keys));
            assert_eq!(packed_bytes(&values), packed_bytes(&expected_values));
        }
    }
    Ok(())
}

#[test]
fn x1_value_scan_late_record_failure_never_publishes_attended_head() -> Result<()> {
    for policy in [KvCachePolicy::Rotated4, KvCachePolicy::Rotated4Keys3Values] {
        for scalar_only in [false, true] {
            let (mut keys, mut values) = populated_caches(policy, 1, 65, scalar_only)?;
            let mut correct_values = values.copy()?;
            let frozen_keys = packed_bytes(&keys);
            let cache = values
                .packed
                .as_mut()
                .ok_or_else(|| invalid("missing values"))?;
            let scale: [u8; 4] = cache.tail[..4].try_into()?;
            // The tail is the 65th cached row, after a complete sealed page.
            // Reject its scale only after 64 successful value updates.
            cache.tail[..4].copy_from_slice(&(-0.0_f32).to_le_bytes());
            let malformed_values = packed_bytes(&values);
            let queries = input(HEAD_DIM, 9)?;
            let projected = input(2 * HEAD_DIM, 11)?;
            let current_key = input(keys.width, 31)?;
            let current_value = input(values.width, 41)?;
            let run = |keys: &mut KvCache, values: &mut KvCache, output: &mut [f32]| {
                attend(
                    keys,
                    values,
                    &queries,
                    &projected,
                    &current_key,
                    &current_value,
                    output,
                    &mut [0.0; 66],
                    1,
                    1.0 / 16.0,
                    super::super::Arithmetic::PinnedSwift,
                    &RowParallelism::default(),
                )
            };
            let mut output = [13.0; HEAD_DIM];
            let error = run(&mut keys, &mut values, &mut output)
                .err()
                .ok_or_else(|| invalid("late invalid value record must fail"))?;
            assert!(
                error
                    .to_string()
                    .contains("scale must be finite and nonnegative")
            );
            assert_eq!(output.map(f32::to_bits), [13.0_f32.to_bits(); HEAD_DIM]);
            assert_eq!(packed_bytes(&keys), frozen_keys);
            assert_eq!(packed_bytes(&values), malformed_values);
            values
                .packed
                .as_mut()
                .ok_or_else(|| invalid("missing values"))?
                .tail[..4]
                .copy_from_slice(&scale);
            let mut expected = [0.0; HEAD_DIM];
            run(&mut keys, &mut correct_values, &mut expected)?;
            run(&mut keys, &mut values, &mut output)?;
            assert_eq!(output.map(f32::to_bits), expected.map(f32::to_bits));
        }
    }
    Ok(())
}

#[test]
fn packed_pages_and_partial_tail_snapshots_are_independent() -> Result<()> {
    for policy in [
        KvCachePolicy::Rotated4,
        KvCachePolicy::Rotated4FittedValues,
        KvCachePolicy::Rotated4Keys3Values,
    ] {
        let mut cache = KvCache::with_policy(2 * HEAD_DIM, 66, policy, false, true)?;
        for token in 0..63 {
            let row = input(2 * HEAD_DIM, token)?;
            cache.prepare_append(&row)?;
            cache.append(&row);
        }
        let before_seal = cache.copy()?;
        let frozen_before = packed_bytes(&before_seal);
        let row = input(2 * HEAD_DIM, 64)?;
        cache.prepare_append(&row)?;
        cache.append(&row);
        let sealed = cache.copy()?;
        let frozen_sealed = packed_bytes(&sealed);
        assert_eq!(
            cache.packed.as_ref().map(|value| value.pages.len()),
            Some(1)
        );
        assert!(Arc::ptr_eq(
            &cache.packed.as_ref().ok_or_else(|| invalid("cache"))?.pages[0],
            &sealed
                .packed
                .as_ref()
                .ok_or_else(|| invalid("cache"))?
                .pages[0]
        ));
        for token in [65, 66] {
            let row = input(2 * HEAD_DIM, token)?;
            cache.prepare_append(&row)?;
            cache.append(&row);
        }
        assert_eq!(packed_bytes(&before_seal), frozen_before);
        assert_eq!(packed_bytes(&sealed), frozen_sealed);
        assert!(cache.prepare_append(&row).is_err());
        assert!(cache.pages.is_empty() && cache.tail.is_empty());
        assert_eq!(cache.len, 66 * 2 * HEAD_DIM);
        cache.clear();
        assert_eq!(cache.len, 0);
        assert_eq!(packed_bytes(&cache), Vec::<u8>::new());
        let mut restored = sealed.copy()?;
        restored.prepare_append(&row)?;
        restored.append(&row);
        assert_eq!(
            &packed_bytes(&restored)[..frozen_sealed.len()],
            frozen_sealed
        );
    }
    Ok(())
}

#[test]
fn failed_second_head_preparation_preserves_committed_packed_bytes() -> Result<()> {
    let mut cache = KvCache::with_policy(2 * HEAD_DIM, 8, KvCachePolicy::Rotated4, true, true)?;
    let row = input(2 * HEAD_DIM, 1)?;
    cache.prepare_append(&row)?;
    cache.append(&row);
    let frozen = packed_bytes(&cache);
    let mut invalid_row = row.clone();
    invalid_row[HEAD_DIM + 7] = f32::NAN;
    assert!(cache.prepare_append(&invalid_row).is_err());
    assert_eq!(cache.len, row.len());
    assert_eq!(packed_bytes(&cache), frozen);
    cache.prepare_append(&row)?;
    cache.append(&row);
    assert_eq!(packed_bytes(&cache), frozen.repeat(2));
    Ok(())
}

#[test]
fn packed_gqa_attention_matches_independently_decoded_dense_reference() -> Result<()> {
    for policy in [
        KvCachePolicy::Rotated4,
        KvCachePolicy::Rotated4FittedValues,
        KvCachePolicy::Rotated4Keys3Values,
    ] {
        let mut keys = KvCache::with_policy(2 * HEAD_DIM, 8, policy, true, true)?;
        let mut values = KvCache::with_policy(2 * HEAD_DIM, 8, policy, false, true)?;
        for position in 0..3 {
            let key = input(2 * HEAD_DIM, position)?;
            let value = input(2 * HEAD_DIM, position + 17)?;
            keys.prepare_append(&key)?;
            values.prepare_append(&value)?;
            keys.append(&key);
            values.append(&value);
        }
        let current_key = input(2 * HEAD_DIM, 31)?;
        let current_value = input(2 * HEAD_DIM, 41)?;
        let queries = input(4 * HEAD_DIM, 9)?;
        let projected = input(8 * HEAD_DIM, 11)?;
        let mut actual = vec![0.0; 4 * HEAD_DIM];
        let mut scores = vec![0.0; 4];
        attend(
            &mut keys,
            &mut values,
            &queries,
            &projected,
            &current_key,
            &current_value,
            &mut actual,
            &mut scores,
            4,
            1.0 / 16.0,
            super::super::Arithmetic::Source,
            &RowParallelism::default(),
        )?;
        let mut all_keys = Vec::new();
        let mut all_values = Vec::new();
        keys.observe_packed(|_, row| {
            all_keys.push(row.to_vec());
            Ok(())
        })?;
        values.observe_packed(|_, row| {
            all_values.push(row.to_vec());
            Ok(())
        })?;
        let decode_current =
            |cache: &KvCache, row: &[f32], output: &mut Vec<Vec<f32>>| -> Result<()> {
                let packed = cache.packed.as_ref().ok_or_else(|| invalid("cache"))?;
                let mut decoded_row = vec![0.0; row.len()];
                for (head, chunk) in row.as_chunks::<HEAD_DIM>().0.iter().enumerate() {
                    let mut codes = vec![0; packed.quantizer.encoded_len()];
                    let mut decoded = [0.0; HEAD_DIM];
                    packed.quantizer.encode(chunk, &mut codes)?;
                    packed.quantizer.decode(&codes, &mut decoded)?;
                    decoded_row[head * HEAD_DIM..(head + 1) * HEAD_DIM].copy_from_slice(&decoded);
                }
                output.push(decoded_row);
                Ok(())
            };
        decode_current(&keys, &current_key, &mut all_keys)?;
        decode_current(&values, &current_value, &mut all_values)?;
        for head in 0..4 {
            let offset = (head / 2) * HEAD_DIM;
            let query = &queries[head * HEAD_DIM..(head + 1) * HEAD_DIM];
            let mut probabilities = all_keys
                .iter()
                .map(|row| {
                    query
                        .iter()
                        .zip(&row[offset..offset + HEAD_DIM])
                        .map(|(&query, &key)| query * key)
                        .sum::<f32>()
                        / 16.0
                })
                .collect::<Vec<_>>();
            crate::kernels::softmax_round_f16(&mut probabilities)?;
            for dimension in 0..HEAD_DIM {
                let value = all_values
                    .iter()
                    .zip(&probabilities)
                    .map(|(row, &weight)| row[offset + dimension] * weight)
                    .sum::<f32>();
                let gate = projected[(2 * head + 1) * HEAD_DIM + dimension];
                let expected = value / (1.0 + (-gate).exp());
                let observed = actual[head * HEAD_DIM + dimension];
                assert!(
                    (observed - expected).abs() <= 0.000_2,
                    "head {head}, channel {dimension}: {observed} vs {expected}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn payload_accounts_for_scales_and_extended_softmax_preserves_old_boundary() -> Result<()> {
    assert_eq!(
        KvCachePolicy::Rotated4.layer_payload_bytes(256, 4, 8192)? * 16,
        132 * 1024 * 1024
    );
    assert_eq!(
        KvCachePolicy::Rotated4Keys3Values.layer_payload_bytes(256, 4, 8192)? * 16,
        116 * 1024 * 1024
    );
    assert_eq!(
        KvCachePolicy::F16.layer_payload_bytes(256, 4, 8192)? * 16,
        512 * 1024 * 1024
    );
    assert!(
        KvCachePolicy::Rotated4
            .layer_payload_bytes(128, 4, 8192)
            .is_err()
    );
    assert!(
        KvCachePolicy::Rotated4
            .layer_payload_bytes(256, usize::MAX, 8192)
            .is_err()
    );
    let mut original = vec![0.0; 8193];
    assert!(crate::kernels::pinned_attention::softmax_round_f16(&mut original).is_err());
    crate::kernels::pinned_attention::softmax_rotated_kv(&mut original)?;
    assert!(
        original
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
    );
    Ok(())
}

fn populated_caches(
    policy: KvCachePolicy,
    kv_heads: usize,
    history: usize,
    scalar: bool,
) -> Result<(KvCache, KvCache)> {
    let mut keys = KvCache::with_policy(kv_heads * HEAD_DIM, 67, policy, true, scalar)?;
    let mut values = KvCache::with_policy(kv_heads * HEAD_DIM, 67, policy, false, scalar)?;
    for position in 0..history {
        let salt = u16::try_from(position)?;
        let key = input(kv_heads * HEAD_DIM, salt)?;
        let value = input(kv_heads * HEAD_DIM, salt + 17)?;
        keys.prepare_append(&key)?;
        values.prepare_append(&value)?;
        keys.append(&key);
        values.append(&value);
    }
    Ok((keys, values))
}

#[cfg(feature = "parallel-rows")]
fn attention_bits(
    keys: &mut KvCache,
    values: &mut KvCache,
    heads: usize,
    arithmetic: super::super::Arithmetic,
    parallelism: &RowParallelism,
) -> Result<Vec<u32>> {
    let queries = input(heads * HEAD_DIM, 9)?;
    let projected = input(2 * heads * HEAD_DIM, 11)?;
    let current_key = input(keys.width, 31)?;
    let current_value = input(values.width, 41)?;
    let mut attended = vec![0.0; queries.len()];
    let mut scores = vec![0.0; score_rows(heads, parallelism) * (keys.len / keys.width + 1)];
    attend(
        keys,
        values,
        &queries,
        &projected,
        &current_key,
        &current_value,
        &mut attended,
        &mut scores,
        heads,
        1.0 / 16.0,
        arithmetic,
        parallelism,
    )?;
    let key_cache = keys
        .packed
        .as_ref()
        .ok_or_else(|| invalid("missing keys"))?;
    let mut current = vec![0; key_cache.row_bytes];
    for (key, output) in current_key
        .as_chunks::<HEAD_DIM>()
        .0
        .iter()
        .zip(current.chunks_exact_mut(key_cache.quantizer.encoded_len()))
    {
        key_cache.quantizer.encode(key, output)?;
    }
    let chunk_heads = heads.div_ceil(parallelism.workers());
    for (chunk, actual) in scores.chunks_exact(keys.len / keys.width + 1).enumerate() {
        let last_head = ((chunk + 1) * chunk_heads).min(heads) - 1;
        let expected = defensive_probabilities(
            key_cache,
            &current,
            &queries,
            last_head,
            heads / key_cache.heads,
            arithmetic,
        )?;
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }
    Ok(attended.iter().map(|value| value.to_bits()).collect())
}

#[cfg(feature = "parallel-rows")]
#[test]
fn parallel_packed_attention_preserves_serial_bits_across_pages_and_head_tails() -> Result<()> {
    for policy in [KvCachePolicy::Rotated4, KvCachePolicy::Rotated4Keys3Values] {
        for scalar in [true, false] {
            for (kv_heads, heads, history) in [(1, 1, 2), (2, 4, 0), (2, 12, 65), (3, 9, 2)] {
                let (mut keys, mut values) = populated_caches(policy, kv_heads, history, scalar)?;
                for arithmetic in [
                    super::super::Arithmetic::Source,
                    super::super::Arithmetic::PinnedSwift,
                ] {
                    let expected = attention_bits(
                        &mut keys,
                        &mut values,
                        heads,
                        arithmetic,
                        &RowParallelism::default(),
                    )?;
                    for workers in [2, 4, 18] {
                        assert_eq!(
                            attention_bits(
                                &mut keys,
                                &mut values,
                                heads,
                                arithmetic,
                                &RowParallelism::new(workers)?,
                            )?,
                            expected,
                            "policy {policy:?}, scalar {scalar}, heads {heads}, history {history}, workers {workers}"
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(feature = "parallel-rows")]
#[test]
fn parallel_packed_attention_joins_failures_and_preserves_committed_cache() -> Result<()> {
    let heads = 12;
    for policy in [KvCachePolicy::Rotated4, KvCachePolicy::Rotated4Keys3Values] {
        let (mut keys, mut values) = populated_caches(policy, 2, 65, false)?;
        let frozen_keys = packed_bytes(&keys);
        let frozen_values = packed_bytes(&values);
        let current_key = input(keys.width, 31)?;
        let current_value = input(values.width, 41)?;
        let mut queries = input(heads * HEAD_DIM, 9)?;
        let mut projected = input(2 * heads * HEAD_DIM, 11)?;
        projected[(2 * (heads - 1) + 1) * HEAD_DIM + 7] = f32::NAN;
        for invalid_query in [false, true] {
            if invalid_query {
                // This earlier failure wins even when the final head's late
                // output failure is returned by a different worker first.
                queries[2 * HEAD_DIM + 3] = f32::NAN;
            }
            let mut expected_error = None;
            for workers in [1, 4, 18] {
                let mut attended = vec![13.0; queries.len()];
                let parallelism = RowParallelism::new(workers)?;
                let mut scores = vec![17.0; score_rows(heads, &parallelism) * 66];
                let error = attend(
                    &mut keys,
                    &mut values,
                    &queries,
                    &projected,
                    &current_key,
                    &current_value,
                    &mut attended,
                    &mut scores,
                    heads,
                    1.0 / 16.0,
                    super::super::Arithmetic::PinnedSwift,
                    &parallelism,
                )
                .err()
                .ok_or_else(|| invalid("the invalid query or final gate must fail"))?
                .to_string();
                if let Some(expected) = &expected_error {
                    assert_eq!(&error, expected);
                } else {
                    expected_error = Some(error);
                }
                assert_eq!(packed_bytes(&keys), frozen_keys);
                assert_eq!(packed_bytes(&values), frozen_values);
                // Reusing the same committed caches after failure cannot race
                // an unjoined worker or inherit provisional attention output.
                assert_eq!(
                    attention_bits(
                        &mut keys,
                        &mut values,
                        heads,
                        super::super::Arithmetic::PinnedSwift,
                        &RowParallelism::new(workers)?,
                    )?,
                    attention_bits(
                        &mut keys,
                        &mut values,
                        heads,
                        super::super::Arithmetic::PinnedSwift,
                        &RowParallelism::default(),
                    )?
                );
            }
        }
    }
    Ok(())
}
