//! Independent causal-prefix and private-worker controls for six-head prefill.

use bonsai_ternary::half_to_float;

use super::super::{Geometry, KvCache};
use super::{attend_prefix, attend_tile_chunk, pinned_attention};
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
    // Include a partial channel tile and two independent KV groups. The
    // normalization/rotary stage is upstream of the selected attention helper.
    let width = 68;
    let geometry = Geometry {
        embedding: 4,
        feed_forward: 4,
        heads: 12,
        kv_heads: 2,
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
    let mut keys = KvCache::new(geometry.kv_heads * width, 128)?;
    let mut values = KvCache::new(geometry.kv_heads * width, 128)?;
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
        assert!(actual.iter().all(|&value| value == 19.0));
    }
    // The sixth head fails during score evaluation before any value output.
    scores.resize(6, 19.0);
    fixture.queries[0][5 * width] = f32::NAN;
    assert!(prefix(&fixture, 0, &mut actual, &mut scores).is_err());
    assert!(actual.iter().all(|&value| value == 19.0));
    fixture.queries[0][5 * width] = 0.0;
    fixture.current_values[0][0] = f32::NAN;
    assert!(prefix(&fixture, 0, &mut actual, &mut scores).is_err());
    assert!(actual.iter().all(|&value| value == 19.0));
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
    assert!(actual.iter().flatten().all(|&value| value == 19.0));
    Ok(())
}
