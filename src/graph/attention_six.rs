//! Complete six-head decode and causal-prefill groups without tail cache rereads.

use super::{Geometry, KvCache, attention_gate};
use crate::kernels::pinned_attention::{self, six};
use crate::{Result, invalid};

fn rows<T>(values: &mut [T], width: usize) -> Result<[&mut [T]; 6]> {
    if width == 0 || values.len() != 6 * width {
        return Err(invalid("invalid six-head attention scratch geometry"));
    }
    let mut rows = values.chunks_exact_mut(width);
    let (Some(first), Some(second), Some(third), Some(fourth), Some(fifth), Some(sixth)) = (
        rows.next(),
        rows.next(),
        rows.next(),
        rows.next(),
        rows.next(),
        rows.next(),
    ) else {
        return Err(invalid("incomplete six-head attention scratch rows"));
    };
    Ok([first, second, third, fourth, fifth, sixth])
}

#[expect(
    clippy::too_many_arguments,
    reason = "Borrow existing transactional graph projections, caches and disjoint worker scratch directly."
)]
pub(super) fn attend(
    geometry: &Geometry,
    head: usize,
    queries: &[f32],
    projected: &[f32],
    keys: &KvCache,
    values: &KvCache,
    current_key: &[f32],
    current_value: &[f32],
    output: &mut [f32],
    scores: &mut [f32],
    scale: f32,
) -> Result<()> {
    attend_prefix(
        geometry,
        head,
        queries,
        projected,
        keys,
        values,
        keys.len / keys.width,
        current_key,
        current_value,
        output,
        scores,
        scale,
    )
}

// Tiled prefill has appended the whole tile before attention. Restrict both
// caches to the causal prefix; the current row retains its original F32 dot.
#[expect(
    clippy::too_many_arguments,
    reason = "Explicit causal extent keeps already-appended future KV rows unreachable."
)]
fn attend_prefix(
    geometry: &Geometry,
    head: usize,
    queries: &[f32],
    projected: &[f32],
    keys: &KvCache,
    values: &KvCache,
    past_tokens: usize,
    current_key: &[f32],
    current_value: &[f32],
    output: &mut [f32],
    scores: &mut [f32],
    scale: f32,
) -> Result<()> {
    let width = geometry.head;
    if width == 0
        || !head.is_multiple_of(6)
        || head + 6 > geometry.heads
        || geometry.kv_heads == 0
        || geometry.heads / geometry.kv_heads != 6
        || keys.width != geometry.kv_heads * width
        || values.width != keys.width
        || past_tokens > keys.len / keys.width
        || past_tokens > values.len / values.width
        || queries.len() != geometry.heads * width
        || projected.len() != 2 * geometry.heads * width
        || current_key.len() != keys.width
        || current_value.len() != values.width
    {
        return Err(invalid("invalid six-head attention causal geometry"));
    }
    let offset = head / 6 * width;
    let query_rows: [&[f32]; 6] =
        std::array::from_fn(|index| &queries[(head + index) * width..(head + index + 1) * width]);
    let score_len = past_tokens + 1;
    let mut scores = rows(scores, score_len)?;
    for (position, key) in keys.tokens().take(past_tokens).enumerate() {
        let dots = six::dot(&key[offset..offset + width], query_rows)?;
        for (score, dot) in scores.iter_mut().zip(dots) {
            score[position] = dot * scale;
        }
    }
    for (score, query) in scores.iter_mut().zip(query_rows) {
        score[score_len - 1] =
            pinned_attention::dot_f32(&current_key[offset..offset + width], query)? * scale;
        pinned_attention::softmax_round_f16(score)?;
    }
    let scores = scores.each_ref().map(|score| &score[..]);
    let gates: [&[f32]; 6] =
        std::array::from_fn(|index| attention_gate(projected, head + index, width));
    let [o0, o1, o2, o3, o4, o5] = rows(output, width)?;
    for (tile, (((((o0, o1), o2), o3), o4), o5)) in o0
        .chunks_mut(six::VALUE_TILE)
        .zip(o1.chunks_mut(six::VALUE_TILE))
        .zip(o2.chunks_mut(six::VALUE_TILE))
        .zip(o3.chunks_mut(six::VALUE_TILE))
        .zip(o4.chunks_mut(six::VALUE_TILE))
        .zip(o5.chunks_mut(six::VALUE_TILE))
        .enumerate()
    {
        let start = tile * six::VALUE_TILE;
        let end = start + o0.len();
        six::value_tile_gated(
            values.tokens().take(past_tokens),
            &current_value[offset + start..offset + end],
            scores,
            offset + start,
            [o0, o1, o2, o3, o4, o5],
            gates.map(|gate| &gate[start..end]),
        )?;
    }
    Ok(())
}

// Each caller worker owns complete token rows and its six-head score storage.
// Keep the cache owner immutable and reuse one six-head group at a time rather
// than falling back to two independent tail heads in Swift's 6:1 GQA geometry.
#[expect(
    clippy::too_many_arguments,
    reason = "Borrow the existing per-token tile rows and caller-owned score storage without staging copies."
)]
pub(super) fn attend_tile_chunk(
    geometry: &Geometry,
    first_position: usize,
    queries: &[Vec<f32>],
    projected: &[Vec<f32>],
    keys: &KvCache,
    values: &KvCache,
    current_keys: &[Vec<f32>],
    current_values: &[Vec<f32>],
    outputs: &mut [Vec<f32>],
    scores: &mut [f32],
    score_stride: usize,
    scale: f32,
) -> Result<()> {
    let count = outputs.len();
    if count == 0
        || count > 32
        || geometry.kv_heads == 0
        || !geometry.heads.is_multiple_of(6)
        || geometry.heads / geometry.kv_heads != 6
        || queries.len() != count
        || projected.len() != count
        || current_keys.len() != count
        || current_values.len() != count
        || scores.len() != count * score_stride
        || score_stride < 6 * (first_position + count)
        || outputs
            .iter()
            .any(|output| output.len() != geometry.heads * geometry.head)
    {
        return Err(invalid("invalid six-head attention tile geometry"));
    }
    for (token, output) in outputs.iter_mut().enumerate() {
        let position = first_position + token;
        let token_scores = &mut scores[token * score_stride..(token + 1) * score_stride];
        for head in (0..geometry.heads).step_by(6) {
            attend_prefix(
                geometry,
                head,
                &queries[token],
                &projected[token],
                keys,
                values,
                position,
                &current_keys[token],
                &current_values[token],
                &mut output[head * geometry.head..(head + 6) * geometry.head],
                &mut token_scores[..6 * (position + 1)],
                scale,
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "attention_six_prefill_tests.rs"]
mod prefill_tests;

#[cfg(test)]
mod tests {
    use bonsai_ternary::half_to_float;

    use super::super::{Geometry, KvCache};
    use super::{attend, pinned_attention};

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "Compare complete six-head output against independent head/channel recomputation across page boundaries."
    )]
    fn six_head_decode_matches_single_head_recompute_across_pages() -> crate::Result<()> {
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
        let pattern = [0.0_f32, -0.0, 0.125, -0.25, 0.5, -1.0, 2.0];
        let queries: Vec<f32> = (0..geometry.heads * width)
            .map(|index| pattern[(index + 3) % 7])
            .collect();
        let projected: Vec<f32> = (0..geometry.heads * width * 2)
            .map(|index| pattern[(index + 5) % 7])
            .collect();
        let current_key: Vec<f32> = (0..geometry.kv_heads * width)
            .map(|index| pattern[(index + 1) % 7])
            .collect();
        let current_value: Vec<f32> = (0..geometry.kv_heads * width)
            .map(|index| pattern[(index + 2) % 7])
            .collect();
        for cached in [0, 1, 63, 64, 65] {
            let mut keys = KvCache::new(geometry.kv_heads * width, 128)?;
            let mut values = KvCache::new(geometry.kv_heads * width, 128)?;
            for token in 0..cached {
                let key: Vec<f32> = (0..geometry.kv_heads * width)
                    .map(|index| pattern[(index + token) % 7])
                    .collect();
                let value: Vec<f32> = (0..geometry.kv_heads * width)
                    .map(|index| pattern[(index + token + 4) % 7])
                    .collect();
                keys.reserve_append()?;
                values.reserve_append()?;
                keys.append(&key);
                values.append(&value);
            }
            let key_bits: Vec<u16> = keys.tokens().flatten().copied().collect();
            let value_bits: Vec<u16> = values.tokens().flatten().copied().collect();
            for head in [0, 6] {
                let mut actual = vec![19.0; 6 * width];
                let mut scores = vec![19.0; 6 * (cached + 1)];
                attend(
                    &geometry,
                    head,
                    &queries,
                    &projected,
                    &keys,
                    &values,
                    &current_key,
                    &current_value,
                    &mut actual,
                    &mut scores,
                    0.125,
                )?;
                let offset = head / 6 * width;
                for local_head in 0..6 {
                    let h = head + local_head;
                    let query = &queries[h * width..(h + 1) * width];
                    let mut probs = Vec::new();
                    for key in keys.tokens() {
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
                            current_key[offset..offset + width].iter().copied(),
                            query,
                        )? * 0.125,
                    );
                    pinned_attention::softmax_round_f16(&mut probs)?;
                    for channel in 0..width {
                        let reduction = pinned_attention::dot(
                            values
                                .tokens()
                                .map(|row| half_to_float(row[offset + channel]))
                                .chain(std::iter::once(current_value[offset + channel])),
                            &probs,
                        )?;
                        let gate = projected[(h * 2 + 1) * width + channel];
                        let expected = reduction * (1.0 / (1.0 + (-gate).exp()));
                        assert_eq!(
                            actual[local_head * width + channel].to_bits(),
                            expected.to_bits()
                        );
                    }
                }
            }
            assert!(keys.tokens().flatten().copied().eq(key_bits));
            assert!(values.tokens().flatten().copied().eq(value_bits));
        }
        Ok(())
    }
}
