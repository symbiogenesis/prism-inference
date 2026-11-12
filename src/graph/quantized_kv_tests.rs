use super::*;

fn input(width: usize, salt: u16) -> Result<Vec<f32>> {
    (0..width)
        .map(|index| {
            let sample = (u16::try_from(index)? % 31 + salt) % 47;
            Ok((f32::from(sample) - 23.0) / 19.0)
        })
        .collect()
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
fn packed_pages_and_partial_tail_snapshots_are_independent() -> Result<()> {
    for policy in [KvCachePolicy::Rotated4, KvCachePolicy::Rotated4Keys3Values] {
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
    for policy in [KvCachePolicy::Rotated4, KvCachePolicy::Rotated4Keys3Values] {
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
        let mut scores = vec![0.0; 4 * 4];
        attend(
            &keys,
            &values,
            &queries,
            &projected,
            &current_key,
            &current_value,
            &mut actual,
            &mut scores,
            4,
            1.0 / 16.0,
            super::super::Arithmetic::Source,
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
