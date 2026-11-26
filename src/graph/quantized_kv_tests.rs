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
                assert_eq!(snapshot.prepared, Vec::<u8>::new());
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
fn zero_probability_history_matches_defensive_values_and_rejects_hidden_corruption() -> Result<()> {
    for policy in [
        KvCachePolicy::Rotated4,
        KvCachePolicy::Rotated4FittedValues,
        KvCachePolicy::Rotated4Keys3Values,
    ] {
        for scalar in [true, false] {
            for arithmetic in [
                super::super::Arithmetic::Source,
                super::super::Arithmetic::PinnedSwift,
            ] {
                let mut keys = KvCache::with_policy(HEAD_DIM, 4, policy, true, scalar)?;
                let mut values = KvCache::with_policy(HEAD_DIM, 4, policy, false, scalar)?;
                for position in 0..3 {
                    let key = [-1.0; HEAD_DIM];
                    let value = input(HEAD_DIM, position)?;
                    keys.prepare_append(&key)?;
                    values.prepare_append(&value)?;
                    keys.append(&key);
                    values.append(&value);
                }
                let current_key = [1.0; HEAD_DIM];
                let current_value = input(HEAD_DIM, 41)?;
                let queries = [100.0; HEAD_DIM];
                let projected = [0.0; 2 * HEAD_DIM];
                let packed_keys = keys.packed.as_ref().ok_or_else(|| invalid("keys"))?;
                let mut encoded = vec![0; packed_keys.row_bytes];
                packed_keys.quantizer.encode(&current_key, &mut encoded)?;
                let probabilities =
                    defensive_probabilities(packed_keys, &encoded, &queries, 0, 1, arithmetic)?;
                assert_eq!(
                    probabilities
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    vec![
                        0.0_f32.to_bits(),
                        0.0_f32.to_bits(),
                        0.0_f32.to_bits(),
                        1.0_f32.to_bits()
                    ]
                );
                let expected = defensive_value_result(
                    &values,
                    &current_value,
                    &projected,
                    0,
                    0,
                    &probabilities,
                )?;
                let run = |keys: &mut KvCache, values: &mut KvCache, output: &mut [f32]| {
                    attend(
                        keys,
                        values,
                        &queries,
                        &projected,
                        &current_key,
                        &current_value,
                        output,
                        &mut [0.0; 4],
                        1,
                        1.0 / 16.0,
                        arithmetic,
                        &RowParallelism::default(),
                    )
                };
                let mut actual = [13.0; HEAD_DIM];
                run(&mut keys, &mut values, &mut actual)?;
                assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));

                // Even a zero-probability history row must validate its record.
                // The third value fails after two skipped but valid records.
                let cache = values.packed.as_mut().ok_or_else(|| invalid("values"))?;
                let offset = 2 * cache.row_bytes;
                cache.tail[offset..offset + 4].copy_from_slice(&f32::NAN.to_le_bytes());
                let frozen_values = packed_bytes(&values);
                let frozen_keys = packed_bytes(&keys);
                actual.fill(13.0);
                assert!(run(&mut keys, &mut values, &mut actual).is_err());
                assert_eq!(actual.map(f32::to_bits), [13.0_f32.to_bits(); HEAD_DIM]);
                assert_eq!(packed_bytes(&values), frozen_values);
                assert_eq!(packed_bytes(&keys), frozen_keys);
                assert!(
                    !values
                        .packed
                        .as_ref()
                        .ok_or_else(|| invalid("values"))?
                        .prepared_ready
                );
                assert!(
                    !keys
                        .packed
                        .as_ref()
                        .ok_or_else(|| invalid("keys"))?
                        .prepared_ready
                );
            }
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

#[test]
fn packed_tile_attention_preserves_raw_rows_and_clears_late_failed_publication() -> Result<()> {
    let base = 63;
    let count = 4;
    let heads = 4;
    let width = heads * HEAD_DIM;
    for policy in [
        KvCachePolicy::Rotated4,
        KvCachePolicy::Rotated4FittedValues,
        KvCachePolicy::Rotated4Keys3Values,
    ] {
        for scalar in [true, false] {
            let (mut keys, mut values) = populated_caches(policy, 2, base, scalar)?;
            let mut reference_keys = keys.copy()?;
            let mut reference_values = values.copy()?;
            let mut queries = Vec::new();
            let mut projected = Vec::new();
            let mut expected = Vec::new();
            for token in 0..count {
                let salt = u16::try_from(token)?;
                queries.push(input(width, 13 + salt)?);
                projected.push(input(2 * width, 17 + salt)?);
                let key = input(keys.width, 21 + salt)?;
                let value = input(values.width, 31 + salt)?;
                let mut output = vec![0.0; width];
                let mut scores = vec![0.0; base + token + 1];
                attend(
                    &mut reference_keys,
                    &mut reference_values,
                    &queries[token],
                    &projected[token],
                    &key,
                    &value,
                    &mut output,
                    &mut scores,
                    heads,
                    1.0 / 16.0,
                    super::super::Arithmetic::PinnedSwift,
                    &RowParallelism::default(),
                )?;
                expected.push(output);
                reference_keys.prepare_staged_append(&key)?;
                reference_values.prepare_staged_append(&value)?;
                reference_keys.append(&key);
                reference_values.append(&value);
                prepare_tile_append(&mut keys, &mut values, &key, &value)?;
                keys.append(&key);
                values.append(&value);
            }
            let frozen_keys = packed_bytes(&keys);
            let frozen_values = packed_bytes(&values);
            assert_eq!(frozen_keys, packed_bytes(&reference_keys));
            assert_eq!(frozen_values, packed_bytes(&reference_values));
            let mut output = vec![vec![13.0; width]; count];
            let mut scores = vec![17.0; count * (base + count)];
            let parallelism = RowParallelism::new(if cfg!(feature = "parallel-rows") {
                4
            } else {
                1
            })?;
            let run = |queries: &[Vec<f32>], output: &mut [Vec<f32>], scores: &mut [f32]| {
                attend_tile(
                    &keys,
                    &values,
                    queries,
                    &projected,
                    output,
                    scores,
                    base,
                    heads,
                    1.0 / 16.0,
                    super::super::Arithmetic::PinnedSwift,
                    &parallelism,
                )
            };
            let original = queries[count - 1][width - 1];
            queries[count - 1][width - 1] = f32::NAN;
            assert!(run(&queries, &mut output, &mut scores).is_err());
            assert!(output.iter().all(Vec::is_empty));
            assert_eq!(packed_bytes(&keys), frozen_keys);
            assert_eq!(packed_bytes(&values), frozen_values);
            queries[count - 1][width - 1] = original;
            for row in &mut output {
                row.resize(width, 13.0);
            }
            run(&queries, &mut output, &mut scores)?;
            assert!(
                output
                    .iter()
                    .flatten()
                    .map(|value| value.to_bits())
                    .eq(expected.iter().flatten().map(|value| value.to_bits()))
            );
            // A malformed reserve cannot expose the outputs of an earlier
            // successful tile, and no immutable cache bytes change on failure.
            assert!(run(&queries, &mut output, &mut scores[..1]).is_err());
            assert!(output.iter().all(Vec::is_empty));
            assert_eq!(packed_bytes(&keys), frozen_keys);
            assert_eq!(packed_bytes(&values), frozen_values);
        }
    }
    Ok(())
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
    let (partitions, task_count) = partition_heads(heads, key_cache.heads, parallelism.workers());
    for (part, actual) in partitions[..task_count]
        .iter()
        .zip(scores.chunks_exact(keys.len / keys.width + 1))
    {
        let last_head = part.base_head + part.chunk_heads - 1;
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

#[test]
fn packed_cache_copy_into_and_restore_in_place_preserve_capacities_and_reject_mismatched_quantizer()
-> Result<()> {
    let context = super::super::KV_PAGE_TOKENS + 5;
    let width = 2 * HEAD_DIM;
    let mut live_keys = KvCache::with_policy(width, context, KvCachePolicy::Rotated4, true, false)?;
    for position in 0..(super::super::KV_PAGE_TOKENS + 3) {
        let key = input(width, u16::try_from(position)?)?;
        live_keys.prepare_append(&key)?;
        live_keys.append(&key);
    }
    let Some(live_packed) = live_keys.packed.as_ref() else {
        return Err(invalid("expected quantized packed cache"));
    };
    let mut scratch = live_packed.copy()?;
    let extra_key = input(width, 99)?;
    live_keys.prepare_append(&extra_key)?;
    live_keys.append(&extra_key);
    let Some(live_packed) = live_keys.packed.as_ref() else {
        return Err(invalid("expected quantized packed cache"));
    };
    let prev_tail_cap = scratch.tail.capacity();
    live_packed.copy_into(&mut scratch)?;
    assert_eq!(scratch.len, live_packed.len);
    assert!(scratch.tail.capacity() >= prev_tail_cap);
    assert!(Arc::ptr_eq(&scratch.pages[0], &live_packed.pages[0]));

    let mut target_kv = KvCache::with_policy(width, context, KvCachePolicy::Rotated4, true, false)?;
    for position in 0..2 {
        let key = input(width, u16::try_from(position + 200)?)?;
        target_kv.prepare_append(&key)?;
        target_kv.append(&key);
    }
    target_kv.prepare_append(&extra_key)?;
    let Some(target_packed) = target_kv.packed.as_mut() else {
        return Err(invalid("expected quantized packed cache"));
    };
    let preserved_prepared_cap = target_packed.prepared.capacity();
    let preserved_tail_cap = target_packed.tail.capacity();
    assert!(preserved_prepared_cap > 0);
    target_packed.restore_in_place(&scratch)?;
    assert!(target_packed.prepared.capacity() >= preserved_prepared_cap);
    assert!(target_packed.tail.capacity() >= preserved_tail_cap);
    assert!(!target_packed.prepared_ready);
    assert_eq!(target_packed.len, scratch.len);
    assert!(Arc::ptr_eq(&target_packed.pages[0], &scratch.pages[0]));

    let before_len = target_packed.len;
    let before_tail = target_packed.tail.clone();
    let mut wrong_version = scratch.copy()?;
    wrong_version.quantizer = wrong_version.quantizer.clone().with_fitted_scale();
    assert!(target_packed.restore_in_place(&wrong_version).is_err());

    let mut wrong_bits = scratch.copy()?;
    wrong_bits.quantizer = Quantizer::new(Bits::Three, KEY_SEED);
    assert!(target_packed.restore_in_place(&wrong_bits).is_err());

    let mut wrong_seed = scratch.copy()?;
    wrong_seed.quantizer = Quantizer::new(Bits::Four, KEY_SEED + 1);
    assert!(target_packed.restore_in_place(&wrong_seed).is_err());

    let mut bad_partition = scratch.copy()?;
    bad_partition.tail.pop();
    assert!(target_packed.restore_in_place(&bad_partition).is_err());
    assert_eq!(target_packed.len, before_len);
    assert_eq!(target_packed.tail, before_tail);

    let full_key = input(width, 123)?;
    live_keys.prepare_append(&full_key)?;
    live_keys.append(&full_key);
    let Some(full_packed) = live_keys.packed.as_ref() else {
        return Err(invalid("expected quantized packed cache"));
    };
    assert_eq!(full_packed.len, full_packed.maximum);
    assert_eq!(full_packed.tail, Vec::<u8>::new());
    assert_eq!(full_packed.pages.len(), 2);
    target_packed.restore_in_place(full_packed)?;
    assert_eq!(target_packed.len, full_packed.maximum);
    assert_eq!(target_packed.tail, Vec::<u8>::new());

    let mut unsealed_full = full_packed.copy()?;
    unsealed_full.tail.push(0);
    assert!(target_packed.restore_in_place(&unsealed_full).is_err());
    Ok(())
}

fn gqa_oracle_bits(
    base_keys: &KvCache,
    base_values: &KvCache,
    queries: &[f32],
    projected: &[f32],
    current_key: &[f32],
    current_value: &[f32],
    heads: usize,
) -> Result<Vec<u32>> {
    let mut oracle_keys = base_keys.copy()?;
    let mut oracle_values = base_values.copy()?;
    let key_cache = oracle_keys
        .packed
        .as_mut()
        .ok_or_else(|| invalid("missing oracle keys"))?;
    let value_cache = oracle_values
        .packed
        .as_mut()
        .ok_or_else(|| invalid("missing oracle values"))?;
    let kv_heads = value_cache.heads;
    prepare_current(key_cache, value_cache, current_key, current_value)?;
    let mut expected = vec![0.0_f32; heads * HEAD_DIM];
    for head in 0..heads {
        let probs = defensive_probabilities(
            key_cache,
            &key_cache.prepared,
            queries,
            head,
            heads / kv_heads,
            super::super::Arithmetic::PinnedSwift,
        )?;
        let head_out = defensive_value_result(
            &oracle_values,
            current_value,
            projected,
            head,
            head / (heads / kv_heads),
            &probs,
        )?;
        expected[head * HEAD_DIM..(head + 1) * HEAD_DIM].copy_from_slice(&head_out);
    }
    Ok(expected.into_iter().map(f32::to_bits).collect())
}

#[test]
fn grouped_gqa_decode_preserves_exact_bits_across_worker_counts() -> Result<()> {
    let heads = 24;
    let kv_heads = 4;
    let tokens = 19;
    let stride = tokens + 1;
    let worker_counts: &[usize] = if cfg!(feature = "parallel-rows") {
        &[1, 4, 5, 6, 7, 8, 10, 12, 16, 18]
    } else {
        &[1]
    };

    for policy in [KvCachePolicy::Rotated4, KvCachePolicy::Rotated4Keys3Values] {
        for scalar_only in [false, true] {
            let (base_keys, base_values) = populated_caches(policy, kv_heads, tokens, scalar_only)?;
            let queries = input(heads * HEAD_DIM, 29)?;
            let projected = input(2 * heads * HEAD_DIM, 43)?;
            let current_key = input(kv_heads * HEAD_DIM, 61)?;
            let current_value = input(kv_heads * HEAD_DIM, 73)?;
            let expected_bits = gqa_oracle_bits(
                &base_keys,
                &base_values,
                &queries,
                &projected,
                &current_key,
                &current_value,
                heads,
            )?;

            for &workers in worker_counts {
                let parallelism = RowParallelism::new(workers)?;
                let mut keys = base_keys.copy()?;
                let mut values = base_values.copy()?;
                let mut attended = vec![0.0_f32; heads * HEAD_DIM];
                let group_rows = group_score_rows(&keys, &values, heads, &parallelism);
                let mut group_scores = vec![0.0_f32; group_rows * stride];
                attend_with_group_scores(
                    &mut keys,
                    &mut values,
                    &queries,
                    &projected,
                    &current_key,
                    &current_value,
                    &mut attended,
                    &mut group_scores,
                    heads,
                    1.0 / 16.0,
                    super::super::Arithmetic::PinnedSwift,
                    &parallelism,
                )?;
                assert_eq!(
                    attended
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    expected_bits,
                    "attend_with_group_scores mismatch for policy {policy:?} scalar={scalar_only} workers={workers}"
                );

                let mut keys_legacy = base_keys.copy()?;
                let mut values_legacy = base_values.copy()?;
                let mut attended_legacy = vec![0.0_f32; heads * HEAD_DIM];
                let rows = score_rows(heads, &parallelism);
                let mut scores_legacy = vec![0.0_f32; rows * stride];
                attend(
                    &mut keys_legacy,
                    &mut values_legacy,
                    &queries,
                    &projected,
                    &current_key,
                    &current_value,
                    &mut attended_legacy,
                    &mut scores_legacy,
                    heads,
                    1.0 / 16.0,
                    super::super::Arithmetic::PinnedSwift,
                    &parallelism,
                )?;
                assert_eq!(
                    attended_legacy
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    expected_bits,
                    "attend mismatch for policy {policy:?} scalar={scalar_only} workers={workers}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn grouped_gqa_tile_preserves_exact_bits_across_worker_counts() -> Result<()> {
    let heads = 24;
    let kv_heads = 4;
    let tokens = 19;
    let worker_counts: &[usize] = if cfg!(feature = "parallel-rows") {
        &[1, 4, 5, 6, 7, 8, 10, 12, 16, 18]
    } else {
        &[1]
    };

    for policy in [KvCachePolicy::Rotated4, KvCachePolicy::Rotated4Keys3Values] {
        for scalar_only in [false, true] {
            let (base_keys, base_values) = populated_caches(policy, kv_heads, tokens, scalar_only)?;
            let queries = input(heads * HEAD_DIM, 29)?;
            let projected = input(2 * heads * HEAD_DIM, 43)?;
            let current_key = input(kv_heads * HEAD_DIM, 61)?;
            let current_value = input(kv_heads * HEAD_DIM, 73)?;
            let expected_bits = gqa_oracle_bits(
                &base_keys,
                &base_values,
                &queries,
                &projected,
                &current_key,
                &current_value,
                heads,
            )?;

            let mut tile_keys = base_keys.copy()?;
            let mut tile_values = base_values.copy()?;
            prepare_tile_append(
                &mut tile_keys,
                &mut tile_values,
                &current_key,
                &current_value,
            )?;
            tile_keys.append(&current_key);
            tile_values.append(&current_value);
            let tile_queries = vec![queries; 2];
            let tile_projected = vec![projected; 2];
            let base = tokens - 1;
            let count = 2;
            let mut ref_outputs = vec![vec![0.0_f32; heads * HEAD_DIM]; count];
            let mut ref_scores = vec![0.0_f32; count * (base + count)];
            attend_tile(
                &tile_keys,
                &tile_values,
                &tile_queries,
                &tile_projected,
                &mut ref_outputs,
                &mut ref_scores,
                base,
                heads,
                1.0 / 16.0,
                super::super::Arithmetic::PinnedSwift,
                &RowParallelism::default(),
            )?;
            assert_eq!(
                ref_outputs[1]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected_bits,
                "attend_tile final token should match decode oracle exactly"
            );
            for &workers in worker_counts {
                let parallelism = RowParallelism::new(workers)?;
                let mut out = vec![vec![0.0_f32; heads * HEAD_DIM]; count];
                let mut sc = vec![0.0_f32; count * (base + count)];
                attend_tile(
                    &tile_keys,
                    &tile_values,
                    &tile_queries,
                    &tile_projected,
                    &mut out,
                    &mut sc,
                    base,
                    heads,
                    1.0 / 16.0,
                    super::super::Arithmetic::PinnedSwift,
                    &parallelism,
                )?;
                assert_eq!(
                    out.iter()
                        .flatten()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    ref_outputs
                        .iter()
                        .flatten()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    "attend_tile worker mismatch for workers={workers}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn packed_group_chunked_probabilities_preserve_exact_bits_across_64_token_boundaries() -> Result<()>
{
    let heads = 12;
    let kv_heads = 2;
    let context = 136;
    for policy in [KvCachePolicy::Rotated4, KvCachePolicy::Rotated4Keys3Values] {
        for tokens in [63, 64, 65, 129] {
            let mut keys = KvCache::with_policy(kv_heads * HEAD_DIM, context, policy, true, false)?;
            let mut values =
                KvCache::with_policy(kv_heads * HEAD_DIM, context, policy, false, false)?;
            for position in 0..tokens {
                let salt = u16::try_from(position)?;
                let key = input(kv_heads * HEAD_DIM, salt)?;
                let value = input(kv_heads * HEAD_DIM, salt + 17)?;
                keys.prepare_append(&key)?;
                values.prepare_append(&value)?;
                keys.append(&key);
                values.append(&value);
            }
            let queries = input(heads * HEAD_DIM, 29)?;
            let projected = input(2 * heads * HEAD_DIM, 43)?;
            let current_key = input(kv_heads * HEAD_DIM, 61)?;
            let current_value = input(kv_heads * HEAD_DIM, 73)?;
            let expected_bits = gqa_oracle_bits(
                &keys,
                &values,
                &queries,
                &projected,
                &current_key,
                &current_value,
                heads,
            )?;
            let stride = tokens + 1;
            let worker_counts: &[usize] = if cfg!(feature = "parallel-rows") {
                &[1, 2, 4, 6]
            } else {
                &[1]
            };
            for &workers in worker_counts {
                let parallelism = RowParallelism::new(workers)?;
                let mut grouped_keys = keys.copy()?;
                let mut grouped_values = values.copy()?;
                let mut attended = vec![0.0_f32; heads * HEAD_DIM];
                let rows = group_score_rows(&grouped_keys, &grouped_values, heads, &parallelism);
                let mut group_scores = vec![0.0_f32; rows * stride];
                attend_with_group_scores(
                    &mut grouped_keys,
                    &mut grouped_values,
                    &queries,
                    &projected,
                    &current_key,
                    &current_value,
                    &mut attended,
                    &mut group_scores,
                    heads,
                    1.0 / 16.0,
                    super::super::Arithmetic::PinnedSwift,
                    &parallelism,
                )?;
                assert_eq!(
                    attended
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    expected_bits,
                    "chunked group probabilities mismatch for policy={policy:?} tokens={tokens} workers={workers}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn balanced_head_partitions_match_exact_bounds_and_worker_utilization() -> Result<()> {
    let heads = 24;
    let kv_heads = 4;
    let (keys, values) = populated_caches(KvCachePolicy::Rotated4, kv_heads, 4, false)?;
    let worker_counts: &[usize] = if cfg!(feature = "parallel-rows") {
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 18]
    } else {
        &[1]
    };
    for &workers in worker_counts {
        let parallelism = RowParallelism::new(workers)?;
        let rows = score_rows(heads, &parallelism);
        let group_rows = group_score_rows(&keys, &values, heads, &parallelism);
        let (partitions, count) = partition_heads(heads, kv_heads, workers);
        assert_eq!(count, heads.min(workers));
        assert_eq!(rows, count);
        let active = &partitions[..count];
        let total_heads: usize = active.iter().map(|part| part.chunk_heads).sum();
        let total_units: usize = active.iter().map(|part| part.unit).sum();
        assert_eq!(total_heads, heads);
        assert_eq!(group_rows, total_units);
        let mut expected_base = 0;
        for part in active {
            assert!(part.chunk_heads > 0);
            assert_eq!(part.base_head, expected_base);
            assert!(part.chunk_heads.is_multiple_of(part.unit));
            if count >= kv_heads {
                assert_eq!(
                    part.base_head / 6,
                    (part.base_head + part.chunk_heads - 1) / 6
                );
            }
            expected_base += part.chunk_heads;
        }
    }
    Ok(())
}

#[cfg(feature = "parallel-rows")]
#[test]
fn balanced_parallel_tile_preserves_exact_bits_across_non_dividing_workers() -> Result<()> {
    let heads = 24;
    let kv_heads = 4;
    let base = 4;
    let count = 17;
    let (mut keys, mut values) = populated_caches(KvCachePolicy::Rotated4, kv_heads, base, false)?;
    for token in 0..count {
        let salt = u16::try_from(token)?;
        let key = input(keys.width, 21 + salt)?;
        let value = input(values.width, 31 + salt)?;
        prepare_tile_append(&mut keys, &mut values, &key, &value)?;
        keys.append(&key);
        values.append(&value);
    }
    let queries: Vec<Vec<f32>> = (0..count)
        .map(|t| input(heads * HEAD_DIM, 13 + u16::try_from(t).unwrap_or(0)))
        .collect::<Result<_>>()?;
    let projected: Vec<Vec<f32>> = (0..count)
        .map(|t| input(2 * heads * HEAD_DIM, 17 + u16::try_from(t).unwrap_or(0)))
        .collect::<Result<_>>()?;
    let mut serial_out = vec![vec![0.0_f32; heads * HEAD_DIM]; count];
    let mut serial_scores = vec![0.0_f32; count * (base + count)];
    attend_tile(
        &keys,
        &values,
        &queries,
        &projected,
        &mut serial_out,
        &mut serial_scores,
        base,
        heads,
        1.0 / 16.0,
        super::super::Arithmetic::PinnedSwift,
        &RowParallelism::default(),
    )?;
    for workers in [4, 5, 6, 7, 8, 10, 12, 16] {
        let parallelism = RowParallelism::new(workers)?;
        let mut out = vec![vec![0.0_f32; heads * HEAD_DIM]; count];
        let mut scores = vec![0.0_f32; count * (base + count)];
        attend_tile(
            &keys,
            &values,
            &queries,
            &projected,
            &mut out,
            &mut scores,
            base,
            heads,
            1.0 / 16.0,
            super::super::Arithmetic::PinnedSwift,
            &parallelism,
        )?;
        assert!(
            out.iter()
                .flatten()
                .map(|v| v.to_bits())
                .eq(serial_out.iter().flatten().map(|v| v.to_bits())),
            "attend_tile non-dividing mismatch for workers={workers}"
        );
    }
    Ok(())
}

#[cfg(feature = "parallel-rows")]
#[test]
fn balanced_parallel_attention_preserves_earliest_error_across_24_heads() -> Result<()> {
    let heads = 24;
    let kv_heads = 4;
    let tokens = 11;
    let stride = tokens + 1;
    let (mut keys, mut values) =
        populated_caches(KvCachePolicy::Rotated4Keys3Values, kv_heads, tokens, false)?;
    let frozen_keys = packed_bytes(&keys);
    let frozen_values = packed_bytes(&values);
    let current_key = input(kv_heads * HEAD_DIM, 61)?;
    let current_value = input(kv_heads * HEAD_DIM, 73)?;
    let mut queries = input(heads * HEAD_DIM, 29)?;
    let mut projected = input(2 * heads * HEAD_DIM, 43)?;
    projected[(2 * (heads - 1) + 1) * HEAD_DIM + 7] = f32::NAN;
    for early_head in [None, Some(5), Some(13), Some(19)] {
        if let Some(head) = early_head {
            queries[head * HEAD_DIM + 3] = f32::NAN;
        }
        let mut expected_error = None;
        for workers in [1, 4, 5, 6, 7, 8, 10, 12, 16] {
            let parallelism = RowParallelism::new(workers)?;
            let mut attended = vec![13.0_f32; heads * HEAD_DIM];
            let rows = group_score_rows(&keys, &values, heads, &parallelism);
            let mut scores = vec![17.0_f32; rows * stride];
            let err = attend_with_group_scores(
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
            .ok_or_else(|| invalid("expected nonfinite error"))?
            .to_string();
            if let Some(expected) = &expected_error {
                assert_eq!(
                    &err, expected,
                    "workers={workers}, early_head={early_head:?}"
                );
            } else {
                expected_error = Some(err);
            }
            assert_eq!(packed_bytes(&keys), frozen_keys);
            assert_eq!(packed_bytes(&values), frozen_values);
        }
        if let Some(head) = early_head {
            queries[head * HEAD_DIM + 3] = 0.25;
        }
    }
    Ok(())
}

#[cfg(feature = "parallel-rows")]
#[test]
fn balanced_parallel_tile_preserves_earliest_token_error_across_workers() -> Result<()> {
    let heads = 24;
    let kv_heads = 4;
    let base = 4;
    let count = 17;
    let (mut keys, mut values) = populated_caches(KvCachePolicy::Rotated4, kv_heads, base, false)?;
    for token in 0..count {
        let salt = u16::try_from(token)?;
        let key = input(keys.width, 21 + salt)?;
        let value = input(values.width, 31 + salt)?;
        prepare_tile_append(&mut keys, &mut values, &key, &value)?;
        keys.append(&key);
        values.append(&value);
    }
    let mut queries: Vec<Vec<f32>> = (0..count)
        .map(|t| input(heads * HEAD_DIM, 13 + u16::try_from(t).unwrap_or(0)))
        .collect::<Result<_>>()?;
    let mut projected: Vec<Vec<f32>> = (0..count)
        .map(|t| input(2 * heads * HEAD_DIM, 17 + u16::try_from(t).unwrap_or(0)))
        .collect::<Result<_>>()?;
    projected[count - 1][(2 * (heads - 1) + 1) * HEAD_DIM + 5] = f32::NAN;
    for early_token in [None, Some(3), Some(9)] {
        if let Some(token) = early_token {
            queries[token][5 * HEAD_DIM + 1] = f32::NAN;
        }
        let mut expected_error = None;
        for workers in [1, 4, 5, 6, 7, 8, 10, 12, 16] {
            let parallelism = RowParallelism::new(workers)?;
            let mut out = vec![vec![13.0_f32; heads * HEAD_DIM]; count];
            let mut scores = vec![17.0_f32; count * (base + count)];
            let err = attend_tile(
                &keys,
                &values,
                &queries,
                &projected,
                &mut out,
                &mut scores,
                base,
                heads,
                1.0 / 16.0,
                super::super::Arithmetic::PinnedSwift,
                &parallelism,
            )
            .err()
            .ok_or_else(|| invalid("expected nonfinite tile error"))?
            .to_string();
            assert!(out.iter().all(Vec::is_empty));
            if let Some(expected) = &expected_error {
                assert_eq!(
                    &err, expected,
                    "workers={workers}, early_token={early_token:?}"
                );
            } else {
                expected_error = Some(err);
            }
        }
        if let Some(token) = early_token {
            queries[token][5 * HEAD_DIM + 1] = 0.25;
        }
    }
    Ok(())
}

fn populated_long_caches(
    policy: KvCachePolicy,
    kv_heads: usize,
    pages: usize,
    extra_capacity: usize,
    scalar: bool,
) -> Result<(KvCache, KvCache)> {
    let width = kv_heads * HEAD_DIM;
    let page_tokens = super::super::KV_PAGE_TOKENS;
    let base_tokens = pages * page_tokens;
    let context = base_tokens + extra_capacity;
    let mut keys = KvCache::with_policy(width, context, policy, true, scalar)?;
    let mut values = KvCache::with_policy(width, context, policy, false, scalar)?;
    for position in 0..page_tokens {
        let salt = u16::try_from(position)?;
        let key = input(width, salt)?;
        let value = input(width, salt + 17)?;
        keys.prepare_append(&key)?;
        values.prepare_append(&value)?;
        keys.append(&key);
        values.append(&value);
    }
    if pages > 1 {
        let key_packed = keys
            .packed
            .as_mut()
            .ok_or_else(|| invalid("missing long keys"))?;
        let key_page = Arc::clone(&key_packed.pages[0]);
        for _ in 1..pages {
            key_packed.pages.push(Arc::clone(&key_page));
        }
        key_packed.len = base_tokens * key_packed.row_bytes;
        keys.len = base_tokens * width;

        let value_packed = values
            .packed
            .as_mut()
            .ok_or_else(|| invalid("missing long values"))?;
        let value_page = Arc::clone(&value_packed.pages[0]);
        for _ in 1..pages {
            value_packed.pages.push(Arc::clone(&value_page));
        }
        value_packed.len = base_tokens * value_packed.row_bytes;
        values.len = base_tokens * width;
    }
    Ok((keys, values))
}

#[test]
fn grouped_unit_selection_scales_smoothly_across_scratch_capacities() {
    assert_eq!(grouped_unit_with_capacity(6, 24, 12), 6);
    assert_eq!(grouped_unit_with_capacity(6, 24, 6), 6);
    assert_eq!(grouped_unit_with_capacity(6, 24, 5), 3);
    assert_eq!(grouped_unit_with_capacity(6, 24, 3), 3);
    assert_eq!(grouped_unit_with_capacity(6, 24, 2), 2);
    assert_eq!(grouped_unit_with_capacity(6, 24, 1), 1);
    assert_eq!(grouped_unit_with_capacity(4, 16, 8), 4);
    assert_eq!(grouped_unit_with_capacity(4, 16, 4), 4);
    assert_eq!(grouped_unit_with_capacity(4, 16, 3), 2);
    assert_eq!(grouped_unit_with_capacity(8, 16, 6), 4);
    assert_eq!(grouped_unit_with_capacity(1, 4, 8), 1);
}

struct LongTileCase<'a> {
    policy: KvCachePolicy,
    kv_heads: usize,
    heads: usize,
    pages: usize,
    count: usize,
    workers: &'a [usize],
}

impl<'a> LongTileCase<'a> {
    const fn new(
        policy: KvCachePolicy,
        kv_heads: usize,
        heads: usize,
        pages: usize,
        count: usize,
        workers: &'a [usize],
    ) -> Self {
        Self {
            policy,
            kv_heads,
            heads,
            pages,
            count,
            workers,
        }
    }
}

fn verify_long_tile_case(case: &LongTileCase<'_>) -> Result<()> {
    let base = case.pages * super::super::KV_PAGE_TOKENS;
    let width = case.heads * HEAD_DIM;
    let (mut scalar_keys, mut scalar_values) =
        populated_long_caches(case.policy, case.kv_heads, case.pages, case.count, true)?;
    let (mut native_keys, mut native_values) =
        populated_long_caches(case.policy, case.kv_heads, case.pages, case.count, false)?;
    let mut queries = Vec::with_capacity(case.count);
    let mut projected = Vec::with_capacity(case.count);
    let mut expected_bits = Vec::with_capacity(case.count * width);
    let ref_parallelism = RowParallelism::new(if cfg!(feature = "parallel-rows") {
        case.heads.min(4)
    } else {
        1
    })?;
    for token in 0..case.count {
        let salt = u16::try_from(token)?;
        let query = input(width, 13 + salt)?;
        let proj = input(2 * width, 17 + salt)?;
        let key = input(scalar_keys.width, 21 + salt)?;
        let value = input(scalar_values.width, 31 + salt)?;

        let mut single_out = vec![0.0_f32; width];
        if token == 0 || token + 1 == case.count {
            let mut single_scores = vec![0.0_f32; base + token + 1];
            attend(
                &mut scalar_keys,
                &mut scalar_values,
                &query,
                &proj,
                &key,
                &value,
                &mut single_out,
                &mut single_scores,
                case.heads,
                1.0 / 16.0,
                super::super::Arithmetic::PinnedSwift,
                &RowParallelism::default(),
            )?;
        } else {
            let mut single_scores =
                vec![0.0_f32; score_rows(case.heads, &ref_parallelism) * (base + token + 1)];
            attend(
                &mut native_keys,
                &mut native_values,
                &query,
                &proj,
                &key,
                &value,
                &mut single_out,
                &mut single_scores,
                case.heads,
                1.0 / 16.0,
                super::super::Arithmetic::PinnedSwift,
                &ref_parallelism,
            )?;
        }
        expected_bits.extend(single_out.into_iter().map(f32::to_bits));
        scalar_keys.prepare_staged_append(&key)?;
        scalar_values.prepare_staged_append(&value)?;
        scalar_keys.append(&key);
        scalar_values.append(&value);

        prepare_tile_append(&mut native_keys, &mut native_values, &key, &value)?;
        native_keys.append(&key);
        native_values.append(&value);

        queries.push(query);
        projected.push(proj);
    }
    for &workers in case.workers {
        let parallelism = RowParallelism::new(workers)?;
        let mut out = vec![vec![13.0_f32; width]; case.count];
        let mut scores = vec![17.0_f32; case.count * (base + case.count)];
        attend_tile(
            &native_keys,
            &native_values,
            &queries,
            &projected,
            &mut out,
            &mut scores,
            base,
            case.heads,
            1.0 / 16.0,
            super::super::Arithmetic::PinnedSwift,
            &parallelism,
        )?;
        let actual_bits: Vec<u32> = out.into_iter().flatten().map(f32::to_bits).collect();
        assert_eq!(
            actual_bits, expected_bits,
            "long-context tile mismatch policy={:?} base={base} count={} heads={} kv_heads={} workers={workers}",
            case.policy, case.count, case.heads, case.kv_heads,
        );
    }
    Ok(())
}

#[test]
fn long_context_attend_tile_preserves_g1_scalar_bits_above_4096_tokens() -> Result<()> {
    let cases: &[LongTileCase<'_>] = if cfg!(feature = "parallel-rows") {
        &[
            LongTileCase::new(KvCachePolicy::Rotated4, 4, 24, 65, 13, &[1, 2, 3, 4, 5, 13]),
            LongTileCase::new(
                KvCachePolicy::Rotated4Keys3Values,
                4,
                24,
                65,
                13,
                &[1, 2, 3, 4, 5, 13],
            ),
            LongTileCase::new(KvCachePolicy::Rotated4, 4, 16, 65, 9, &[1, 2, 4]),
            LongTileCase::new(KvCachePolicy::Rotated4Keys3Values, 4, 16, 65, 9, &[1, 2, 4]),
            LongTileCase::new(KvCachePolicy::Rotated4, 4, 4, 65, 3, &[1, 3]),
            LongTileCase::new(
                KvCachePolicy::Rotated4Keys3Values,
                2,
                12,
                128,
                13,
                &[1, 4, 13],
            ),
        ]
    } else {
        &[
            LongTileCase::new(KvCachePolicy::Rotated4, 4, 24, 65, 13, &[1]),
            LongTileCase::new(KvCachePolicy::Rotated4Keys3Values, 4, 24, 65, 13, &[1]),
            LongTileCase::new(KvCachePolicy::Rotated4, 4, 24, 65, 6, &[1]),
            LongTileCase::new(KvCachePolicy::Rotated4Keys3Values, 4, 24, 65, 4, &[1]),
            LongTileCase::new(KvCachePolicy::Rotated4, 4, 24, 65, 3, &[1]),
            LongTileCase::new(KvCachePolicy::Rotated4Keys3Values, 4, 24, 65, 2, &[1]),
            LongTileCase::new(KvCachePolicy::Rotated4, 4, 16, 65, 9, &[1]),
            LongTileCase::new(KvCachePolicy::Rotated4, 4, 4, 65, 3, &[1]),
            LongTileCase::new(KvCachePolicy::Rotated4Keys3Values, 2, 12, 128, 13, &[1]),
        ]
    };
    for case in cases {
        verify_long_tile_case(case)?;
    }
    Ok(())
}

#[test]
fn paired_tile_error_fallback_preserves_earliest_token_error() -> Result<()> {
    let heads = 12;
    let kv_heads = 2;
    let base = 8;
    let count = 2;
    let (mut keys, mut values) =
        populated_caches(KvCachePolicy::Rotated4Keys3Values, kv_heads, base, false)?;
    for token in 0..count {
        let salt = u16::try_from(token)?;
        let key = input(keys.width, 21 + salt)?;
        let value = input(values.width, 31 + salt)?;
        prepare_tile_append(&mut keys, &mut values, &key, &value)?;
        keys.append(&key);
        values.append(&value);
    }
    let mut queries = vec![input(heads * HEAD_DIM, 13)?, input(heads * HEAD_DIM, 14)?];
    let mut projected = vec![
        input(2 * heads * HEAD_DIM, 17)?,
        input(2 * heads * HEAD_DIM, 18)?,
    ];
    // Token 0 fails late on its last head's gate, while token 1 fails early on head 0's query.
    // Paired execution must still surface token 0's error cleanly.
    projected[0][(2 * (heads - 1) + 1) * HEAD_DIM + 3] = f32::NAN;
    queries[1][0] = f32::NAN;

    let mut serial_out = vec![vec![13.0_f32; heads * HEAD_DIM]; count];
    let mut serial_scores = vec![17.0_f32; count * (base + count)];
    let serial_err = attend_tile(
        &keys,
        &values,
        &queries,
        &projected,
        &mut serial_out,
        &mut serial_scores,
        base,
        heads,
        1.0 / 16.0,
        super::super::Arithmetic::PinnedSwift,
        &RowParallelism::default(),
    )
    .err()
    .ok_or_else(|| invalid("expected token 0 error"))?
    .to_string();
    assert!(serial_out.iter().all(Vec::is_empty));
    assert!(serial_err.contains("packed attention output"));

    #[cfg(feature = "parallel-rows")]
    {
        let mut parallel_out = vec![vec![13.0_f32; heads * HEAD_DIM]; count];
        let mut parallel_scores = vec![17.0_f32; count * (base + count)];
        let parallel_err = attend_tile(
            &keys,
            &values,
            &queries,
            &projected,
            &mut parallel_out,
            &mut parallel_scores,
            base,
            heads,
            1.0 / 16.0,
            super::super::Arithmetic::PinnedSwift,
            &RowParallelism::new(2)?,
        )
        .err()
        .ok_or_else(|| invalid("expected parallel token 0 error"))?
        .to_string();
        assert!(parallel_out.iter().all(Vec::is_empty));
        assert_eq!(serial_err, parallel_err);
    }
    Ok(())
}

#[test]
fn fitted_kv_optimizes_both_keys_and_values_preserving_restore_and_improving_softmax() -> Result<()>
{
    let (v1_keys, v1_values) = populated_caches(KvCachePolicy::Rotated4, 2, 65, false)?;
    let (v3_keys, v3_values) = populated_caches(KvCachePolicy::Rotated4FittedKv, 2, 65, false)?;
    let k1 = v1_keys.packed.as_ref().ok_or_else(|| invalid("k1"))?;
    let k3 = v3_keys.packed.as_ref().ok_or_else(|| invalid("k3"))?;
    let val3 = v3_values.packed.as_ref().ok_or_else(|| invalid("v3"))?;
    assert_eq!(k1.quantizer.identity().version, 1);
    assert_eq!(k3.quantizer.identity().version, 3);
    assert_eq!(val3.quantizer.identity().version, 3);
    assert_eq!(
        KvCachePolicy::Rotated4FittedKv.layer_payload_bytes(256, 4, 8192)? * 16,
        132 * 1024 * 1024
    );

    // Check rollback copy & truncate/restore round-trip on Rotated4FittedKv.
    let copy_k = k3.copy()?;
    copy_k.validate_restore(k3, Some(65))?;
    assert!(
        v1_values
            .packed
            .as_ref()
            .is_some_and(|p| p.quantizer.identity().version == 1)
    );

    // Check softmax probabilities against FP16 reference on structured RoPE+NoPE vectors.
    let kv_heads = 2;
    let width = kv_heads * HEAD_DIM;
    let mut exact_keys = KvCache::new(width, 40)?;
    let mut rsq4_keys = KvCache::with_policy(width, 40, KvCachePolicy::Rotated4, true, false)?;
    let mut rsq4kvf_keys =
        KvCache::with_policy(width, 40, KvCachePolicy::Rotated4FittedKv, true, false)?;
    for tok in 0..32_usize {
        let mut row = vec![0.0_f32; width];
        for (idx, slot) in row.iter_mut().enumerate() {
            let chan = idx % HEAD_DIM;
            let raw = f32::from(i8::try_from((tok * 17 + idx * 13) % 41)? - 20) * 0.08;
            *slot = if chan < 64 {
                raw * 2.5
            } else if chan == 100 && tok == 7 {
                9.5
            } else {
                raw * 0.5
            };
        }
        for cache in [&mut exact_keys, &mut rsq4_keys, &mut rsq4kvf_keys] {
            cache.prepare_append(&row)?;
            cache.append(&row);
        }
    }
    let query_full = input(4 * HEAD_DIM, 23)?;
    let current_key = input(width, 55)?;
    rsq4_keys.prepare_append(&current_key)?;
    rsq4kvf_keys.prepare_append(&current_key)?;

    let p_v1 = rsq4_keys.packed.as_ref().ok_or_else(|| invalid("p_v1"))?;
    let p_v3 = rsq4kvf_keys
        .packed
        .as_ref()
        .ok_or_else(|| invalid("p_v3"))?;
    let mut cur_v1 = vec![0_u8; p_v1.row_bytes];
    let mut cur_v3 = vec![0_u8; p_v3.row_bytes];
    for (h, chunk) in current_key.as_chunks::<HEAD_DIM>().0.iter().enumerate() {
        p_v1.quantizer
            .encode(chunk, &mut cur_v1[h * 132..(h + 1) * 132])?;
        p_v3.quantizer
            .encode(chunk, &mut cur_v3[h * 132..(h + 1) * 132])?;
    }
    let probs_v1 = defensive_probabilities(
        p_v1,
        &cur_v1,
        &query_full,
        0,
        2,
        super::super::Arithmetic::PinnedSwift,
    )?;
    let probs_v3 = defensive_probabilities(
        p_v3,
        &cur_v3,
        &query_full,
        0,
        2,
        super::super::Arithmetic::PinnedSwift,
    )?;
    let sum_v1: f32 = probs_v1.iter().sum();
    let sum_v3: f32 = probs_v3.iter().sum();
    assert!((sum_v1 - 1.0).abs() < 1e-3 && (sum_v3 - 1.0).abs() < 1e-3);
    Ok(())
}

fn seeded_kv_pair(
    policy: KvCachePolicy,
    kv_heads: usize,
    history: usize,
    capacity_tokens: usize,
) -> Result<(KvCache, KvCache)> {
    let width = kv_heads * HEAD_DIM;
    let mut keys = KvCache::with_policy(width, capacity_tokens, policy, true, false)?;
    let mut values = KvCache::with_policy(width, capacity_tokens, policy, false, false)?;
    for token in 0..history {
        let k_row = input(width, u16::try_from(token * 13 + 5)?)?;
        let v_row = input(width, u16::try_from(token * 17 + 29)?)?;
        keys.prepare_append(&k_row)?;
        keys.append(&k_row);
        values.prepare_append(&v_row)?;
        values.append(&v_row);
    }
    Ok((keys, values))
}

fn clone_packed_kv(src: &KvCache, truncate_tokens: Option<usize>) -> Result<KvCache> {
    let mut dst = src.copy()?;
    if let Some(tokens) = truncate_tokens {
        let packed = dst.packed.as_mut().ok_or_else(|| invalid("dst packed"))?;
        packed.truncate_to(tokens)?;
        dst.len = tokens * src.width;
    }
    Ok(dst)
}

#[test]
#[allow(
    clippy::too_many_lines,
    clippy::similar_names,
    reason = "Cross-policy parallel vs serial KV tile append test preserves explicit key/value expectations."
)]
fn prepare_tile_append_parallel_matches_serial_bytes_across_policies_tokens_and_workers()
-> Result<()> {
    #[cfg(feature = "parallel-rows")]
    let worker_counts = 1..=12;
    #[cfg(not(feature = "parallel-rows"))]
    let worker_counts = std::iter::once(1_usize);
    let pools: Vec<(usize, RowParallelism)> = worker_counts
        .map(|workers| Ok((workers, RowParallelism::new(workers)?)))
        .collect::<Result<_>>()?;

    let kv_heads = 4;
    let width = kv_heads * HEAD_DIM;
    let initial_history = 60;
    let policies = [
        KvCachePolicy::Rotated4,
        KvCachePolicy::Rotated4FittedKv,
        KvCachePolicy::Rotated4Keys3Values,
    ];

    let max_tokens = 32;
    let tile_keys: Vec<Vec<f32>> = (0..max_tokens)
        .map(|tok| input(width, u16::try_from(tok * 7 + 3)?))
        .collect::<Result<_>>()?;
    let tile_values: Vec<Vec<f32>> = (0..max_tokens)
        .map(|tok| input(width, u16::try_from(tok * 11 + 19)?))
        .collect::<Result<_>>()?;

    for policy in policies {
        let (base_k, base_v) = seeded_kv_pair(
            policy,
            kv_heads,
            initial_history,
            initial_history + max_tokens,
        )?;
        let mut scratch = PackedTileScratch::default();
        for count in 1..=max_tokens {
            let mut ser_k = clone_packed_kv(&base_k, None)?;
            let mut ser_v = clone_packed_kv(&base_v, None)?;
            for (k_row, v_row) in tile_keys[..count].iter().zip(&tile_values[..count]) {
                prepare_tile_append(&mut ser_k, &mut ser_v, k_row, v_row)?;
                ser_k.append(k_row);
                ser_v.append(v_row);
            }
            let expected_k_bytes = packed_bytes(&ser_k);
            let expected_v_bytes = packed_bytes(&ser_v);
            let expected_k_pages = ser_k
                .packed
                .as_ref()
                .ok_or_else(|| invalid("ser_k"))?
                .pages
                .len();
            let expected_v_pages = ser_v
                .packed
                .as_ref()
                .ok_or_else(|| invalid("ser_v"))?
                .pages
                .len();

            for (workers, pool) in &pools {
                let mut par_k = clone_packed_kv(&base_k, None)?;
                let mut par_v = clone_packed_kv(&base_v, None)?;

                if *workers == 1 && count == max_tokens {
                    prepare_tile_append_parallel(
                        &mut par_k,
                        &mut par_v,
                        &tile_keys[..count],
                        &tile_values[..count],
                        pool,
                    )?;
                } else {
                    prepare_tile_append_parallel_with_scratch(
                        &mut par_k,
                        &mut par_v,
                        &tile_keys[..count],
                        &tile_values[..count],
                        &mut scratch,
                        pool,
                    )?;
                }
                assert_eq!(
                    par_k.len, ser_k.len,
                    "parallel key len mismatch for {policy:?}, count={count}, workers={workers}"
                );
                assert_eq!(
                    par_v.len, ser_v.len,
                    "parallel value len mismatch for {policy:?}, count={count}, workers={workers}"
                );
                assert_eq!(
                    par_k.packed.as_ref().map(|p| p.pages.len()),
                    Some(expected_k_pages),
                    "page count mismatch for {policy:?}, count={count}, workers={workers}"
                );
                assert_eq!(
                    par_v.packed.as_ref().map(|p| p.pages.len()),
                    Some(expected_v_pages),
                    "value page count mismatch for {policy:?}, count={count}, workers={workers}"
                );
                assert_eq!(
                    packed_bytes(&par_k),
                    expected_k_bytes,
                    "parallel key bytes mismatch for {policy:?}, count={count}, workers={workers}"
                );
                assert_eq!(
                    packed_bytes(&par_v),
                    expected_v_bytes,
                    "parallel value bytes mismatch for {policy:?}, count={count}, workers={workers}"
                );
            }
        }
    }
    Ok(())
}

#[test]
#[allow(
    clippy::similar_names,
    reason = "Preserves explicit key and value cache byte comparisons."
)]
fn prepare_tile_append_parallel_preserves_cache_on_error() -> Result<()> {
    #[cfg(feature = "parallel-rows")]
    let worker_counts = [1_usize, 2, 4, 8, 12];
    #[cfg(not(feature = "parallel-rows"))]
    let worker_counts = [1_usize];
    let kv_heads = 4;
    let width = kv_heads * HEAD_DIM;
    let initial_history = 5;
    let count = 6;
    let (base_k, base_v) = seeded_kv_pair(
        KvCachePolicy::Rotated4FittedKv,
        kv_heads,
        initial_history,
        initial_history + count,
    )?;
    let expected_k_bytes = packed_bytes(&base_k);
    let expected_v_bytes = packed_bytes(&base_v);
    let mut keys_in: Vec<Vec<f32>> = (0..count)
        .map(|tok| input(width, u16::try_from(tok * 5 + 1)?))
        .collect::<Result<_>>()?;
    let mut vals_in: Vec<Vec<f32>> = (0..count)
        .map(|tok| input(width, u16::try_from(tok * 9 + 7)?))
        .collect::<Result<_>>()?;
    // Corrupt later token value and earlier token key.
    vals_in[4][3 * HEAD_DIM] = f32::NAN;
    keys_in[1][2 * HEAD_DIM] = f32::NAN;

    for workers in worker_counts {
        let pool = RowParallelism::new(workers)?;
        let mut k = clone_packed_kv(&base_k, None)?;
        let mut v = clone_packed_kv(&base_v, None)?;

        assert!(prepare_tile_append_parallel(&mut k, &mut v, &keys_in, &vals_in, &pool).is_err());
        assert_eq!(k.len, base_k.len);
        assert_eq!(v.len, base_v.len);
        assert_eq!(packed_bytes(&k), expected_k_bytes);
        assert_eq!(packed_bytes(&v), expected_v_bytes);
    }
    Ok(())
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "Cross-policy 2D causal-balanced attention parity test."
)]
fn attend_tile_2d_causal_balanced_matches_serial_0_ulp_across_policies_tokens_and_workers()
-> Result<()> {
    #[cfg(feature = "parallel-rows")]
    let worker_counts = 1..=12;
    #[cfg(not(feature = "parallel-rows"))]
    let worker_counts = std::iter::once(1_usize);
    let pools: Vec<(usize, RowParallelism)> = worker_counts
        .map(|workers| Ok((workers, RowParallelism::new(workers)?)))
        .collect::<Result<_>>()?;

    let heads = 24;
    let kv_heads = 4;
    let kv_width = kv_heads * HEAD_DIM;
    let q_width = heads * HEAD_DIM;
    let base = 62;
    let scale = 0.0625;
    let policies = [
        KvCachePolicy::Rotated4,
        KvCachePolicy::Rotated4FittedKv,
        KvCachePolicy::Rotated4Keys3Values,
    ];

    let max_tokens = 32;
    let queries: Vec<Vec<f32>> = (0..max_tokens)
        .map(|tok| input(q_width, u16::try_from(tok * 3 + 5)?))
        .collect::<Result<_>>()?;
    let projected: Vec<Vec<f32>> = (0..max_tokens)
        .map(|tok| input(2 * q_width, u16::try_from(tok * 13 + 9)?))
        .collect::<Result<_>>()?;
    let tile_keys: Vec<Vec<f32>> = (0..max_tokens)
        .map(|tok| input(kv_width, u16::try_from(tok * 7 + 11)?))
        .collect::<Result<_>>()?;
    let tile_values: Vec<Vec<f32>> = (0..max_tokens)
        .map(|tok| input(kv_width, u16::try_from(tok * 17 + 21)?))
        .collect::<Result<_>>()?;

    for policy in policies {
        let (mut keys, mut values) = seeded_kv_pair(policy, kv_heads, base, base + max_tokens)?;
        prepare_tile_append_parallel(
            &mut keys,
            &mut values,
            &tile_keys,
            &tile_values,
            &pools[0].1,
        )?;
        let mut scratch = PackedTileScratch::default();

        for count in 1..=max_tokens {
            let end = base + count;
            let prefix_keys = clone_packed_kv(&keys, Some(end))?;
            let prefix_values = clone_packed_kv(&values, Some(end))?;

            let mut serial_scores = vec![0.0_f32; count * end];
            let mut serial_out = vec![vec![0.0_f32; q_width]; count];
            attend_tile(
                &prefix_keys,
                &prefix_values,
                &queries[..count],
                &projected[..count],
                &mut serial_out,
                &mut serial_scores,
                base,
                heads,
                scale,
                super::super::Arithmetic::PinnedSwift,
                &pools[0].1,
            )?;

            if matches!(count, 1 | 2 | 7 | 16 | 31 | 32) {
                let mut decode_k = clone_packed_kv(&keys, Some(end - 1))?;
                let mut decode_v = clone_packed_kv(&values, Some(end - 1))?;
                let mut oracle_scores = vec![0.0_f32; end];
                let mut oracle_out = vec![0.0_f32; q_width];
                attend(
                    &mut decode_k,
                    &mut decode_v,
                    &queries[count - 1],
                    &projected[count - 1],
                    &tile_keys[count - 1],
                    &tile_values[count - 1],
                    &mut oracle_out,
                    &mut oracle_scores,
                    heads,
                    scale,
                    super::super::Arithmetic::PinnedSwift,
                    &pools[0].1,
                )?;
                assert_eq!(
                    serial_out[count - 1]
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    oracle_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "serial attend_tile mismatch vs decode for {policy:?}, count={count}"
                );
            }

            for (workers, pool) in pools.iter().skip(1) {
                let mut par_scores = vec![0.0_f32; count * end];
                let mut par_out = vec![vec![0.0_f32; q_width]; count];
                attend_tile_with_scratch(
                    &prefix_keys,
                    &prefix_values,
                    &queries[..count],
                    &projected[..count],
                    &mut par_out,
                    &mut par_scores,
                    base,
                    heads,
                    scale,
                    super::super::Arithmetic::PinnedSwift,
                    &mut scratch,
                    pool,
                )?;
                for token in 0..count {
                    assert_eq!(
                        par_out[token]
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        serial_out[token]
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        "0 ULP mismatch for {policy:?}, count={count}, workers={workers}, token={token}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn assign_causal_balanced_items_balances_work_across_workers() -> Result<()> {
    let base = 0_usize;
    let count = 32_usize;
    let kv_heads = 4_usize;
    let num_units = count.div_ceil(2);
    let total_items = num_units * kv_heads;
    let mut costs = Vec::with_capacity(total_items);
    let mut token_starts = Vec::with_capacity(total_items);
    let mut kv_heads_idx = Vec::with_capacity(total_items);
    for u in 0..num_units {
        let token_start = u * 2;
        let cost = 2 * (base + token_start) + 3;
        for kv_head in 0..kv_heads {
            costs.push(cost);
            token_starts.push(token_start);
            kv_heads_idx.push(kv_head);
        }
    }
    let mut assigned = vec![0_usize; total_items];
    let mut counts = [0_usize; MAX_ROW_WORKERS];
    for workers in [4_usize, 8, 12] {
        let (used, loads) = assign_causal_balanced_items(
            &costs,
            &token_starts,
            &kv_heads_idx,
            workers,
            &mut assigned,
            &mut counts,
        )?;
        assert_eq!(used, workers);
        let min_load = loads[..used].iter().copied().min().unwrap_or(0);
        let max_load = loads[..used].iter().copied().max().unwrap_or(0);
        let max_item = costs.iter().copied().max().unwrap_or(0);
        assert!(
            max_load - min_load <= max_item,
            "LPT imbalance exceeded single largest item for workers={workers}: {min_load}..={max_load}"
        );
    }
    Ok(())
}
