//! Independent causal-prefix and private-worker controls for six-head prefill.

use bonsai_ternary::half_to_float;

use super::super::{Geometry, KvCache};
use super::{
    attend_group, attend_prefix, attend_tile_chunk, attend_tile_kv_group, pinned_attention,
};
use crate::kernels::RowParallelism;

struct Fixture {
    geometry: Geometry,
    queries: Vec<Vec<f32>>,
    projected: Vec<Vec<f32>>,
    keys: KvCache,
    values: KvCache,
    current_keys: Vec<Vec<f32>>,
    current_values: Vec<Vec<f32>>,
}

fn sample(index: usize) -> f32 {
    [0.0, -0.0, 0.33337, -0.251, 0.501, -1.001, 2.001][index % 7]
}

fn fixture(base: usize, count: usize) -> crate::Result<Fixture> {
    fixture_with_geometry(12, 2, 128, base, count)
}

fn fixture_with_geometry(
    heads: usize,
    kv_heads: usize,
    context: usize,
    base: usize,
    count: usize,
) -> crate::Result<Fixture> {
    // Include a partial channel tile and independent KV groups. The
    // normalization/rotary stage is upstream of the selected attention helper.
    let width = 68;
    let geometry = Geometry {
        embedding: 4,
        feed_forward: 4,
        heads,
        kv_heads,
        head: width,
        rope: 4,
        rope_sections: [1, 1, 0, 0],
        rope_base: 10_000.0,
        epsilon: 1e-6,
        ssm_head: 4,
        ssm_keys: 1,
        ssm_values: 1,
        convolution: 4,
    };
    let mut keys = KvCache::new(geometry.kv_heads * width, context)?;
    let mut values = KvCache::new(geometry.kv_heads * width, context)?;
    let mut current_keys = Vec::new();
    let mut current_values = Vec::new();
    for position in 0..base + count {
        let key: Vec<f32> = (0..geometry.kv_heads * width)
            .map(|index| sample(index + position))
            .collect();
        let value: Vec<f32> = (0..geometry.kv_heads * width)
            .map(|index| sample(index + 3 * position + 1))
            .collect();
        keys.reserve_append()?;
        values.reserve_append()?;
        keys.append(&key);
        values.append(&value);
        if position >= base {
            current_keys.push(key);
            current_values.push(value);
        }
    }
    let queries = (0..count)
        .map(|token| {
            (0..geometry.heads * width)
                .map(|index| sample(index + 2 * token + 2))
                .collect()
        })
        .collect();
    let projected = (0..count)
        .map(|token| {
            (0..2 * geometry.heads * width)
                .map(|index| sample(index + 3 * token + 3))
                .collect()
        })
        .collect();
    Ok(Fixture {
        geometry,
        queries,
        projected,
        keys,
        values,
        current_keys,
        current_values,
    })
}

fn reference(fixture: &Fixture, position: usize, token: usize) -> crate::Result<Vec<f32>> {
    let width = fixture.geometry.head;
    let mut output = vec![0.0; fixture.geometry.heads * width];
    for (head, output) in output.chunks_exact_mut(width).enumerate() {
        let offset = head / 6 * width;
        let query = &fixture.queries[token][head * width..(head + 1) * width];
        let mut probabilities = Vec::new();
        for key in fixture.keys.tokens().take(position) {
            probabilities.push(
                pinned_attention::dot(
                    key[offset..offset + width]
                        .iter()
                        .copied()
                        .map(half_to_float),
                    query,
                )? * 0.125,
            );
        }
        probabilities.push(
            pinned_attention::dot(
                fixture.current_keys[token][offset..offset + width]
                    .iter()
                    .copied(),
                query,
            )? * 0.125,
        );
        pinned_attention::softmax_round_f16(&mut probabilities)?;
        for (channel, output) in output.iter_mut().enumerate() {
            let value = pinned_attention::dot(
                fixture
                    .values
                    .tokens()
                    .take(position)
                    .map(|row| half_to_float(row[offset + channel]))
                    .chain(std::iter::once(
                        fixture.current_values[token][offset + channel],
                    )),
                &probabilities,
            )?;
            let gate = fixture.projected[token][(2 * head + 1) * width + channel];
            *output = value * (1.0 / (1.0 + (-gate).exp()));
        }
    }
    Ok(output)
}

#[test]
fn six_head_prefill_tiles_match_causal_oracle_across_pages_and_workers() -> crate::Result<()> {
    #[cfg(feature = "parallel-rows")]
    let worker_counts = [1, 2, 3, 4, 7, 18];
    #[cfg(not(feature = "parallel-rows"))]
    let worker_counts = [1];
    for count in [1, 2, 3, 8, 16, 32] {
        for base in [0, 63] {
            let fixture = fixture(base, count)?;
            let key_bits: Vec<u16> = fixture.keys.tokens().flatten().copied().collect();
            let value_bits: Vec<u16> = fixture.values.tokens().flatten().copied().collect();
            let expected: Vec<_> = (0..count)
                .map(|token| reference(&fixture, base + token, token))
                .collect::<crate::Result<_>>()?;
            let score_stride = 6 * (base + count);
            for workers in worker_counts {
                let parallelism = RowParallelism::new(workers)?;
                let mut actual =
                    vec![vec![19.0; fixture.geometry.heads * fixture.geometry.head]; count];
                let mut scores = vec![19.0; count * score_stride];
                parallelism.update_paired_chunks(
                    &mut actual,
                    1,
                    &mut scores,
                    score_stride,
                    |first, outputs, scores| {
                        let end = first + outputs.len();
                        attend_tile_chunk(
                            &fixture.geometry,
                            base + first,
                            &fixture.queries[first..end],
                            &fixture.projected[first..end],
                            &fixture.keys,
                            &fixture.values,
                            &fixture.current_keys[first..end],
                            &fixture.current_values[first..end],
                            outputs,
                            scores,
                            score_stride,
                            0.125,
                        )
                    },
                )?;
                assert!(
                    actual
                        .iter()
                        .flatten()
                        .map(|value| value.to_bits())
                        .eq(expected.iter().flatten().map(|value| value.to_bits()))
                );
            }
            assert!(fixture.keys.tokens().flatten().copied().eq(key_bits));
            assert!(fixture.values.tokens().flatten().copied().eq(value_bits));
        }
    }
    Ok(())
}

fn prefix(
    fixture: &Fixture,
    position: usize,
    output: &mut [f32],
    scores: &mut [f32],
) -> crate::Result<()> {
    attend_prefix(
        &fixture.geometry,
        0,
        &fixture.queries[0],
        &fixture.projected[0],
        &fixture.keys,
        &fixture.values,
        position,
        &fixture.current_keys[0],
        &fixture.current_values[0],
        output,
        scores,
        0.125,
    )
}

#[test]
fn six_head_prefill_excludes_future_nan_and_rejects_bad_prefix_before_output() -> crate::Result<()>
{
    let mut fixture = fixture(0, 1)?;
    let width = fixture.geometry.head;
    let poisoned = vec![f32::NAN; fixture.geometry.kv_heads * width];
    fixture.keys.reserve_append()?;
    fixture.values.reserve_append()?;
    fixture.keys.append(&poisoned);
    fixture.values.append(&poisoned);
    let key_bits: Vec<u16> = fixture.keys.tokens().flatten().copied().collect();
    let value_bits: Vec<u16> = fixture.values.tokens().flatten().copied().collect();
    let expected = reference(&fixture, 0, 0)?;
    let mut actual = vec![19.0; 6 * width];
    let mut scores = vec![19.0; 6];
    prefix(&fixture, 0, &mut actual, &mut scores)?;
    assert!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .eq(expected[..6 * width].iter().map(|value| value.to_bits()))
    );
    for (position, score_len) in [(3, 24), (0, 5)] {
        actual.fill(19.0);
        scores.resize(score_len, 19.0);
        assert!(prefix(&fixture, position, &mut actual, &mut scores).is_err());
        assert!(
            actual
                .iter()
                .all(|value| value.to_bits() == 19.0_f32.to_bits())
        );
    }
    // The sixth head fails during score evaluation before any value output.
    scores.resize(6, 19.0);
    fixture.queries[0][5 * width] = f32::NAN;
    assert!(prefix(&fixture, 0, &mut actual, &mut scores).is_err());
    assert!(
        actual
            .iter()
            .all(|value| value.to_bits() == 19.0_f32.to_bits())
    );
    fixture.queries[0][5 * width] = 0.0;
    fixture.current_values[0][0] = f32::NAN;
    assert!(prefix(&fixture, 0, &mut actual, &mut scores).is_err());
    assert!(
        actual
            .iter()
            .all(|value| value.to_bits() == 19.0_f32.to_bits())
    );
    assert!(fixture.keys.tokens().flatten().copied().eq(key_bits));
    assert!(fixture.values.tokens().flatten().copied().eq(value_bits));
    Ok(())
}

#[test]
fn six_head_paired_prefill_excludes_future_rows_and_rejects_malformed_input() -> crate::Result<()> {
    let mut fixture = fixture(63, 3)?;
    let poisoned = vec![f32::NAN; fixture.geometry.kv_heads * fixture.geometry.head];
    fixture.keys.reserve_append()?;
    fixture.values.reserve_append()?;
    fixture.keys.append(&poisoned);
    fixture.values.append(&poisoned);
    let score_stride = 6 * 66;
    let mut actual = vec![vec![19.0; fixture.geometry.heads * fixture.geometry.head]; 3];
    let mut scores = vec![19.0; 3 * score_stride];
    let run = |fixture: &Fixture, actual: &mut [Vec<f32>], scores: &mut [f32]| {
        attend_tile_chunk(
            &fixture.geometry,
            63,
            &fixture.queries,
            &fixture.projected,
            &fixture.keys,
            &fixture.values,
            &fixture.current_keys,
            &fixture.current_values,
            actual,
            scores,
            score_stride,
            0.125,
        )
    };
    run(&fixture, &mut actual, &mut scores)?;
    for (token, actual) in actual.iter().enumerate() {
        let expected = reference(&fixture, 63 + token, token)?;
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
        );
    }
    for row in &mut actual {
        row.fill(19.0);
    }
    let retained_scores = scores.clone();
    let original_query = fixture.queries[1].clone();
    assert!(fixture.queries[1].pop().is_some());
    assert!(run(&fixture, &mut actual, &mut scores).is_err());
    assert_eq!(scores, retained_scores);
    fixture.queries[1] = original_query;
    fixture.queries[1][5 * fixture.geometry.head] = f32::NAN;
    assert!(run(&fixture, &mut actual, &mut scores).is_err());
    assert!(
        actual
            .iter()
            .flatten()
            .all(|value| value.to_bits() == 19.0_f32.to_bits())
    );
    Ok(())
}

#[test]
fn six_head_prefill_long_prefix_and_paired_chunks_match_causal_oracle() -> crate::Result<()> {
    for (base, count) in [(254, 1), (255, 2), (256, 3), (257, 5)] {
        let mut fixture = fixture_with_geometry(12, 2, 320, base, count)?;
        let width = fixture.geometry.head;
        let expected: Vec<_> = (0..count)
            .map(|token| reference(&fixture, base + token, token))
            .collect::<crate::Result<_>>()?;

        // Check `attend_prefix` directly across the 256-token stack chunk boundary.
        let mut prefix_out = vec![19.0_f32; 6 * width];
        let mut prefix_scores = vec![19.0_f32; 6 * (base + 1)];
        prefix(&fixture, base, &mut prefix_out, &mut prefix_scores)?;
        assert_eq!(
            prefix_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected[0][..6 * width]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            "attend_prefix mismatch at base={base}"
        );

        // Check `attend_tile_chunk` (both paired and odd trailing prefix) across 256-token chunks.
        let score_stride = 6 * (base + count);
        let mut actual = vec![vec![19.0; fixture.geometry.heads * width]; count];
        let mut scores = vec![19.0; count * score_stride];
        attend_tile_chunk(
            &fixture.geometry,
            base,
            &fixture.queries,
            &fixture.projected,
            &fixture.keys,
            &fixture.values,
            &fixture.current_keys,
            &fixture.current_values,
            &mut actual,
            &mut scores,
            score_stride,
            0.125,
        )?;
        assert_eq!(
            actual
                .iter()
                .flatten()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .flatten()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            "attend_tile_chunk mismatch at base={base}, count={count}"
        );

        // A late-tile NaN in current_values must fail atomically before writing tile 0.
        if base == 256 && count == 3 {
            prefix_out.fill(19.0);
            prefix_scores.fill(19.0);
            fixture.current_values[0][65] = f32::NAN;
            assert!(prefix(&fixture, base, &mut prefix_out, &mut prefix_scores).is_err());
            assert!(
                prefix_out
                    .iter()
                    .all(|value| value.to_bits() == 19.0_f32.to_bits())
            );
        }
    }
    Ok(())
}

#[test]
fn six_head_prefill_24_head_gqa_matches_oracle_across_non_dividing_workers() -> crate::Result<()> {
    #[cfg(feature = "parallel-rows")]
    let worker_counts = [1, 4, 5, 6, 7, 8, 10, 12, 16];
    #[cfg(not(feature = "parallel-rows"))]
    let worker_counts = [1];
    let base = 17;
    for count in [5, 7, 10, 16, 24] {
        let fixture = fixture_with_geometry(24, 4, 128, base, count)?;
        let expected: Vec<_> = (0..count)
            .map(|token| reference(&fixture, base + token, token))
            .collect::<crate::Result<_>>()?;
        let score_stride = 6 * (base + count);
        for workers in worker_counts {
            let parallelism = RowParallelism::new(workers)?;
            let mut actual =
                vec![vec![19.0; fixture.geometry.heads * fixture.geometry.head]; count];
            let mut scores = vec![19.0; count * score_stride];
            parallelism.update_paired_chunks(
                &mut actual,
                1,
                &mut scores,
                score_stride,
                |first, outputs, scores| {
                    let end = first + outputs.len();
                    attend_tile_chunk(
                        &fixture.geometry,
                        base + first,
                        &fixture.queries[first..end],
                        &fixture.projected[first..end],
                        &fixture.keys,
                        &fixture.values,
                        &fixture.current_keys[first..end],
                        &fixture.current_values[first..end],
                        outputs,
                        scores,
                        score_stride,
                        0.125,
                    )
                },
            )?;
            assert!(
                actual
                    .iter()
                    .flatten()
                    .map(|value| value.to_bits())
                    .eq(expected.iter().flatten().map(|value| value.to_bits())),
                "24-head prefill mismatch for count={count}, workers={workers}"
            );
        }
    }
    Ok(())
}

#[cfg(feature = "parallel-rows")]
#[test]
fn six_head_prefill_24_head_gqa_preserves_earliest_error_across_workers() -> crate::Result<()> {
    let base = 17;
    let count = 17;
    let mut fixture = fixture_with_geometry(24, 4, 128, base, count)?;
    let width = fixture.geometry.head;
    fixture.queries[count - 1][23 * width] = f32::NAN;
    let score_stride = 6 * (base + count);
    for early_token in [None, Some(3), Some(11)] {
        if let Some(token) = early_token {
            fixture.queries[token][6 * width] = f32::NAN;
        }
        let mut expected_error = None;
        for workers in [1, 4, 5, 6, 7, 8, 10, 12, 16] {
            let parallelism = RowParallelism::new(workers)?;
            let mut actual = vec![vec![19.0; fixture.geometry.heads * width]; count];
            let mut scores = vec![19.0; count * score_stride];
            let err = parallelism
                .update_paired_chunks(
                    &mut actual,
                    1,
                    &mut scores,
                    score_stride,
                    |first, outputs, scores| {
                        let end = first + outputs.len();
                        attend_tile_chunk(
                            &fixture.geometry,
                            base + first,
                            &fixture.queries[first..end],
                            &fixture.projected[first..end],
                            &fixture.keys,
                            &fixture.values,
                            &fixture.current_keys[first..end],
                            &fixture.current_values[first..end],
                            outputs,
                            scores,
                            score_stride,
                            0.125,
                        )
                    },
                )
                .err()
                .ok_or_else(|| crate::invalid("expected nonfinite error"))?
                .to_string();
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
            fixture.queries[token][6 * width] = 0.25;
        }
    }
    Ok(())
}

fn check_attend_group_two_keys_lengths<const HEADS: usize>() -> crate::Result<()> {
    for past_tokens in [1, 15, 16, 255, 256, 257, 513, 1025] {
        let fixture = fixture_with_geometry(12, 2, past_tokens + 4, past_tokens - 1, 1)?;
        let width = fixture.geometry.head;
        let head = 6;
        let offset = head / 6 * width;
        let mut actual = vec![19.0_f32; HEADS * width];
        let mut scores = vec![19.0_f32; HEADS * (past_tokens + 1)];
        let mut packed_probs = vec![[0_u16; HEADS]; past_tokens + 1];
        attend_group::<HEADS>(
            &fixture.geometry,
            head,
            &fixture.queries[0],
            &fixture.projected[0],
            &fixture.keys,
            &fixture.values,
            &fixture.current_keys[0],
            &fixture.current_values[0],
            &mut actual,
            &mut scores,
            &mut packed_probs,
            0.125,
        )?;
        for local_head in 0..HEADS {
            let h = head + local_head;
            let query = &fixture.queries[0][h * width..(h + 1) * width];
            let mut probs = Vec::with_capacity(past_tokens + 1);
            for key in fixture.keys.tokens().take(past_tokens) {
                probs.push(
                    pinned_attention::dot(
                        key[offset..offset + width]
                            .iter()
                            .copied()
                            .map(half_to_float),
                        query,
                    )? * 0.125,
                );
            }
            probs.push(
                pinned_attention::dot(
                    fixture.current_keys[0][offset..offset + width]
                        .iter()
                        .copied(),
                    query,
                )? * 0.125,
            );
            pinned_attention::softmax_round_f16(&mut probs)?;
            for channel in 0..width {
                let reduction = pinned_attention::dot(
                    fixture
                        .values
                        .tokens()
                        .take(past_tokens)
                        .map(|row| half_to_float(row[offset + channel]))
                        .chain(std::iter::once(fixture.current_values[0][offset + channel])),
                    &probs,
                )?;
                let gate = fixture.projected[0][(h * 2 + 1) * width + channel];
                let expected = reduction * (1.0 / (1.0 + (-gate).exp()));
                assert_eq!(
                    actual[local_head * width + channel].to_bits(),
                    expected.to_bits(),
                    "attend_group<{HEADS}> mismatch at past_tokens={past_tokens}, local_head={local_head}, channel={channel}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn six_head_decode_attend_group_two_keys_matches_oracle_across_lengths() -> crate::Result<()> {
    check_attend_group_two_keys_lengths::<6>()?;
    check_attend_group_two_keys_lengths::<3>()?;
    check_attend_group_two_keys_lengths::<2>()?;
    Ok(())
}

#[test]
fn six_head_prefill_single_pack_and_tile_chunk_match_oracle_across_lengths() -> crate::Result<()> {
    for (base, count) in [
        (0, 1),
        (14, 1),
        (15, 2),
        (254, 2),
        (255, 2),
        (256, 3),
        (512, 2),
        (1024, 2),
    ] {
        let fixture = fixture_with_geometry(12, 2, base + count + 4, base, count)?;
        let width = fixture.geometry.head;
        let expected: Vec<_> = (0..count)
            .map(|token| reference(&fixture, base + token, token))
            .collect::<crate::Result<_>>()?;

        let mut prefix_out = vec![19.0_f32; 6 * width];
        let mut prefix_scores = vec![19.0_f32; 6 * (base + 1)];
        prefix(&fixture, base, &mut prefix_out, &mut prefix_scores)?;
        assert_eq!(
            prefix_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected[0][..6 * width]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            "attend_prefix mismatch at base={base}"
        );

        let score_stride = 6 * (base + count);
        let mut actual = vec![vec![19.0_f32; fixture.geometry.heads * width]; count];
        let mut scores = vec![19.0_f32; count * score_stride];
        attend_tile_chunk(
            &fixture.geometry,
            base,
            &fixture.queries,
            &fixture.projected,
            &fixture.keys,
            &fixture.values,
            &fixture.current_keys,
            &fixture.current_values,
            &mut actual,
            &mut scores,
            score_stride,
            0.125,
        )?;
        assert_eq!(
            actual
                .iter()
                .flatten()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .flatten()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            "attend_tile_chunk mismatch at base={base}, count={count}"
        );

        for kv_head in 0..fixture.geometry.kv_heads {
            let head_start = kv_head * 6;
            let mut kv_out = vec![19.0_f32; count * 6 * width];
            let mut kv_scores = vec![19.0_f32; count * score_stride];
            attend_tile_kv_group(
                &fixture.geometry,
                head_start,
                base,
                &fixture.queries,
                &fixture.projected,
                &fixture.keys,
                &fixture.values,
                &fixture.current_keys,
                &fixture.current_values,
                &mut kv_out,
                &mut kv_scores,
                score_stride,
                0.125,
            )?;
            for (tok, expected_row) in expected.iter().enumerate() {
                let got = &kv_out[tok * 6 * width..(tok + 1) * 6 * width];
                let want = &expected_row[head_start * width..(head_start + 6) * width];
                assert_eq!(
                    got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "attend_tile_kv_group mismatch at base={base}, count={count}, kv_head={kv_head}, tok={tok}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn six_head_prefill_24_head_gqa_2d_counts_match_oracle_across_workers() -> crate::Result<()> {
    #[cfg(feature = "parallel-rows")]
    let worker_counts = [1, 2, 4, 12, 18];
    #[cfg(not(feature = "parallel-rows"))]
    let worker_counts = [1];
    let base = 255;
    for count in [1, 2, 3, 4, 5, 8] {
        let fixture = fixture_with_geometry(24, 4, base + count + 4, base, count)?;
        let width = fixture.geometry.head;
        let expected: Vec<_> = (0..count)
            .map(|token| reference(&fixture, base + token, token))
            .collect::<crate::Result<_>>()?;
        let score_stride = 6 * (base + count);
        for workers in worker_counts {
            let parallelism = RowParallelism::new(workers)?;
            let kv_heads = fixture.geometry.kv_heads;
            let n_chunks = if count < workers && count > 1 {
                count
            } else {
                1
            };
            let max_chunk_tokens = count.div_ceil(n_chunks);
            let kv_out_len = 6 * width;
            let out_unit = max_chunk_tokens * kv_out_len;
            let score_unit = max_chunk_tokens * score_stride;
            let total_tasks = n_chunks * kv_heads;
            let mut flat_out = vec![19.0_f32; total_tasks * out_unit];
            let mut scores = vec![19.0_f32; total_tasks * score_unit];
            parallelism.update_paired_chunks(
                &mut flat_out,
                out_unit,
                &mut scores,
                score_unit,
                |base_task, out_chunk, scores_chunk| {
                    for (local, (task_out, task_scores)) in out_chunk
                        .chunks_exact_mut(out_unit)
                        .zip(scores_chunk.chunks_exact_mut(score_unit))
                        .enumerate()
                    {
                        let task_idx = base_task + local;
                        let chunk_idx = task_idx / kv_heads;
                        let kv_head = task_idx % kv_heads;
                        let q_len = count / n_chunks;
                        let rem = count % n_chunks;
                        let tok_start = chunk_idx * q_len + chunk_idx.min(rem);
                        let chunk_tokens = q_len + usize::from(chunk_idx < rem);
                        let tok_end = tok_start + chunk_tokens;
                        attend_tile_kv_group(
                            &fixture.geometry,
                            kv_head * 6,
                            base + tok_start,
                            &fixture.queries[tok_start..tok_end],
                            &fixture.projected[tok_start..tok_end],
                            &fixture.keys,
                            &fixture.values,
                            &fixture.current_keys[tok_start..tok_end],
                            &fixture.current_values[tok_start..tok_end],
                            &mut task_out[..chunk_tokens * kv_out_len],
                            &mut task_scores[..chunk_tokens * score_stride],
                            score_stride,
                            0.125,
                        )?;
                    }
                    Ok(())
                },
            )?;
            for chunk_idx in 0..n_chunks {
                let q_len = count / n_chunks;
                let rem = count % n_chunks;
                let tok_start = chunk_idx * q_len + chunk_idx.min(rem);
                let chunk_tokens = q_len + usize::from(chunk_idx < rem);
                for kv_head in 0..kv_heads {
                    let task_idx = chunk_idx * kv_heads + kv_head;
                    for local_tok in 0..chunk_tokens {
                        let got = &flat_out[task_idx * out_unit + local_tok * kv_out_len
                            ..task_idx * out_unit + (local_tok + 1) * kv_out_len];
                        let want = &expected[tok_start + local_tok]
                            [kv_head * kv_out_len..(kv_head + 1) * kv_out_len];
                        assert_eq!(
                            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            "2D prefill mismatch count={count}, workers={workers}, tok={}, kv_head={kv_head}",
                            tok_start + local_tok
                        );
                    }
                }
            }
        }
    }
    Ok(())
}
