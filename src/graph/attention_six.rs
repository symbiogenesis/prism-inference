//! Complete six-head decode and causal-prefill groups without tail cache rereads.

use super::{Geometry, KvCache, attention_gate, checked_product, checked_sum};
use crate::kernels::pinned_attention::{self, six};
use crate::kernels::{MAX_ROW_WORKERS, RowParallelism};
use crate::{Error, Result, invalid};

fn rows<T>(values: &mut [T], width: usize) -> Result<[&mut [T]; 6]> {
    rows_group::<T, 6>(values, width)
}

fn rows_group<T, const HEADS: usize>(values: &mut [T], width: usize) -> Result<[&mut [T]; HEADS]> {
    if HEADS == 0 || width == 0 || values.len() != HEADS * width {
        return Err(invalid("invalid group attention scratch geometry"));
    }
    let mut rows = values.chunks_exact_mut(width);
    Ok(std::array::from_fn(|_| rows.next().unwrap_or_default()))
}

#[cfg(test)]
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

fn resize_packed_rows(buf: &mut Vec<[u16; 6]>, len: usize) -> Result<&mut [[u16; 6]]> {
    if len == 0 || len > crate::model::swift::CONTEXT_TOKENS {
        return Err(invalid("invalid pinned packed probability length"));
    }
    if len > buf.len() {
        let additional = len - buf.len();
        buf.try_reserve_exact(additional)?;
        buf.resize(len, [0_u16; 6]);
    }
    Ok(&mut buf[..len])
}

fn score_past_prefix<const HEADS: usize>(
    keys: &KvCache,
    past_tokens: usize,
    offset: usize,
    width: usize,
    query_rows: [&[f32]; HEADS],
    scores: &mut [&mut [f32]; HEADS],
    scale: f32,
) -> Result<()> {
    let mut key_iter = keys.tokens().take(past_tokens).enumerate();
    while let Some((pos0, key0)) = key_iter.next() {
        if let Some((pos1, key1)) = key_iter.next() {
            let dots = six::dot_two_keys(
                [&key0[offset..offset + width], &key1[offset..offset + width]],
                query_rows,
            )?;
            for (score, (dot0, dot1)) in scores.iter_mut().zip(dots[0].into_iter().zip(dots[1])) {
                score[pos0] = dot0 * scale;
                score[pos1] = dot1 * scale;
            }
        } else {
            let dots = six::dot_group(&key0[offset..offset + width], query_rows)?;
            for (score, dot) in scores.iter_mut().zip(dots) {
                score[pos0] = dot * scale;
            }
        }
    }
    Ok(())
}

// Tiled prefill has appended the whole tile before attention. Restrict both
// caches to the causal prefix; the current row retains its original F32 dot.
#[expect(
    clippy::too_many_arguments,
    reason = "Explicit causal extent keeps already-appended future KV rows unreachable."
)]
fn attend_prefix_with_scratch(
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
    packed_scratch: &mut Vec<[u16; 6]>,
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
    score_past_prefix(
        keys,
        past_tokens,
        offset,
        width,
        query_rows,
        &mut scores,
        scale,
    )?;
    for (score, query) in scores.iter_mut().zip(query_rows) {
        score[score_len - 1] =
            pinned_attention::dot_f32(&current_key[offset..offset + width], query)? * scale;
        pinned_attention::softmax(score)?;
    }
    apply_values(
        geometry,
        head,
        projected,
        values,
        current_value,
        output,
        scores.each_ref().map(|score| &score[..]),
        past_tokens,
        packed_scratch,
    )
}

#[cfg(test)]
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
    let mut packed_scratch = Vec::new();
    attend_prefix_with_scratch(
        geometry,
        head,
        queries,
        projected,
        keys,
        values,
        past_tokens,
        current_key,
        current_value,
        output,
        scores,
        scale,
        &mut packed_scratch,
    )
}

const CAUSAL_PACK_CHUNK: usize = 256;

#[expect(
    clippy::too_many_arguments,
    reason = "Borrow causal score rows, original cache/projection, and reusable packed scratch."
)]
fn apply_values(
    geometry: &Geometry,
    head: usize,
    projected: &[f32],
    values: &KvCache,
    current_value: &[f32],
    output: &mut [f32],
    scores: [&[f32]; 6],
    past_tokens: usize,
    packed_scratch: &mut Vec<[u16; 6]>,
) -> Result<()> {
    let width = geometry.head;
    let offset = head / 6 * width;
    if current_value[offset..offset + width]
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(invalid("nonfinite six-head attention current value"));
    }
    let gates: [&[f32]; 6] =
        std::array::from_fn(|index| attention_gate(projected, head + index, width));
    let mut out_rows = rows(output, width)?.map(|row| row.chunks_mut(six::VALUE_TILE));
    let score_len = past_tokens + 1;
    let mut run_tiles = |weights: &[[u16; 6]]| -> Result<()> {
        for start in (0..width).step_by(six::VALUE_TILE) {
            let end = (start + six::VALUE_TILE).min(width);
            let [o0, o1, o2, o3, o4, o5] = out_rows.each_mut().map(|row| {
                row.next()
                    .ok_or_else(|| invalid("short six-head value output tile"))
            });
            let tile_outputs = [o0?, o1?, o2?, o3?, o4?, o5?];
            let tile_gates = gates.map(|gate| &gate[start..end]);
            six::value_tile_group_packed_gated::<6>(
                values.tokens().take(past_tokens),
                &current_value[offset + start..offset + end],
                weights,
                offset + start,
                tile_outputs,
                tile_gates,
            )?;
        }
        Ok(())
    };
    if score_len <= CAUSAL_PACK_CHUNK {
        let mut packed = [[0_u16; 6]; CAUSAL_PACK_CHUNK];
        pinned_attention::pack_f16_rows(scores, &mut packed[..score_len])?;
        return run_tiles(&packed[..score_len]);
    }
    let packed = resize_packed_rows(packed_scratch, score_len)?;
    pinned_attention::pack_f16_rows(scores, packed)?;
    run_tiles(packed)
}

/// Evaluates `HEADS` (`HEADS` in `{2, 3, 6}`) contiguous query heads sharing a single
/// six-head KV group during decode using consumer-ordered `[token][head]` packed
/// binary16 attention probabilities.
///
/// # Errors
/// Rejects unsupported group sizes, misaligned heads, cross-KV spans or nonfinite reductions.
#[expect(
    clippy::too_many_arguments,
    reason = "Borrow existing transactional graph projections, caches and disjoint worker scratch directly."
)]
pub(super) fn attend_group<const HEADS: usize>(
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
    packed_probs: &mut [[u16; HEADS]],
    scale: f32,
) -> Result<()> {
    let width = geometry.head;
    if !matches!(HEADS, 2 | 3 | 6)
        || width == 0
        || !head.is_multiple_of(HEADS)
        || head + HEADS > geometry.heads
        || geometry.kv_heads == 0
        || geometry.heads / geometry.kv_heads != 6
        || head / 6 != (head + HEADS - 1) / 6
        || keys.width != geometry.kv_heads * width
        || values.width != keys.width
        || queries.len() != geometry.heads * width
        || projected.len() != 2 * geometry.heads * width
        || current_key.len() != keys.width
        || current_value.len() != values.width
    {
        return Err(invalid("invalid group attention decode geometry"));
    }
    let past_tokens = keys.len / keys.width;
    if past_tokens > values.len / values.width {
        return Err(invalid("invalid group attention cache length"));
    }
    let offset = head / 6 * width;
    let query_rows: [&[f32]; HEADS] =
        std::array::from_fn(|index| &queries[(head + index) * width..(head + index + 1) * width]);
    let score_len = past_tokens + 1;
    let mut scores = rows_group::<f32, HEADS>(scores, score_len)?;
    if packed_probs.len() != score_len {
        return Err(invalid("invalid group attention packed probability length"));
    }
    score_past_prefix(
        keys,
        past_tokens,
        offset,
        width,
        query_rows,
        &mut scores,
        scale,
    )?;
    for (score, query) in scores.iter_mut().zip(query_rows) {
        score[score_len - 1] =
            pinned_attention::dot_f32(&current_key[offset..offset + width], query)? * scale;
    }
    let score_refs = scores.each_mut().map(|row| &mut **row);
    pinned_attention::softmax_pack_f16(score_refs, packed_probs)?;
    apply_values_group(
        geometry,
        head,
        projected,
        values,
        current_value,
        output,
        packed_probs,
        past_tokens,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "Borrow packed probability slice and the original cache/projection without extra scratch."
)]
fn apply_values_group<const HEADS: usize>(
    geometry: &Geometry,
    head: usize,
    projected: &[f32],
    values: &KvCache,
    current_value: &[f32],
    output: &mut [f32],
    packed_probs: &[[u16; HEADS]],
    past_tokens: usize,
) -> Result<()> {
    let width = geometry.head;
    let offset = head / 6 * width;
    let gates: [&[f32]; HEADS] =
        std::array::from_fn(|index| attention_gate(projected, head + index, width));
    let mut out_rows = rows_group::<f32, HEADS>(output, width)?;
    for start in (0..width).step_by(six::VALUE_TILE) {
        let end = (start + six::VALUE_TILE).min(width);
        six::value_tile_group_packed_gated(
            values.tokens().take(past_tokens),
            &current_value[offset + start..offset + end],
            packed_probs,
            offset + start,
            out_rows.each_mut().map(|row| &mut row[start..end]),
            gates.map(|gate| &gate[start..end]),
        )?;
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "Pair existing causal score, cache, projection, output borrows, and reusable packed scratch."
)]
fn apply_values_pair(
    geometry: &Geometry,
    head: usize,
    projected: [&[f32]; 2],
    values: &KvCache,
    current: [&[f32]; 2],
    outputs: [&mut [f32]; 2],
    scores: [[&[f32]; 6]; 2],
    past_tokens: usize,
    packed_scratch: &mut [Vec<[u16; 6]>; 2],
) -> Result<()> {
    let width = geometry.head;
    let offset = head / 6 * width;
    if current.iter().any(|row| {
        row[offset..offset + width]
            .iter()
            .any(|value| !value.is_finite())
    }) {
        return Err(invalid("nonfinite six-head paired attention current value"));
    }
    let gates =
        projected.map(|row| std::array::from_fn(|index| attention_gate(row, head + index, width)));
    let [first, second] = outputs;
    let mut first_rows = rows(first, width)?.map(|row| row.chunks_mut(six::PAIR_VALUE_TILE));
    let mut second_rows = rows(second, width)?.map(|row| row.chunks_mut(six::PAIR_VALUE_TILE));
    let first_len = past_tokens + 1;
    let second_len = past_tokens + 2;
    let mut run_tiles = |weights: [&[[u16; 6]]; 2]| -> Result<()> {
        for start in (0..width).step_by(six::PAIR_VALUE_TILE) {
            let end = (start + six::PAIR_VALUE_TILE).min(width);
            let [f0, f1, f2, f3, f4, f5] = first_rows.each_mut().map(|row| {
                row.next()
                    .ok_or_else(|| invalid("short paired value output tile"))
            });
            let [s0, s1, s2, s3, s4, s5] = second_rows.each_mut().map(|row| {
                row.next()
                    .ok_or_else(|| invalid("short paired value output tile"))
            });
            let tile_outputs = [
                [f0?, f1?, f2?, f3?, f4?, f5?],
                [s0?, s1?, s2?, s3?, s4?, s5?],
            ];
            let tile_current = current.map(|row| &row[offset + start..offset + end]);
            let tile_gates = gates.map(|row| row.map(|gate| &gate[start..end]));
            six::value_pair_tile_packed_gated(
                values.tokens().take(past_tokens + 1),
                tile_current,
                weights,
                offset + start,
                tile_outputs,
                tile_gates,
            )?;
        }
        Ok(())
    };
    if second_len <= CAUSAL_PACK_CHUNK {
        let mut packed0 = [[0_u16; 6]; CAUSAL_PACK_CHUNK];
        let mut packed1 = [[0_u16; 6]; CAUSAL_PACK_CHUNK];
        pinned_attention::pack_f16_rows(scores[0], &mut packed0[..first_len])?;
        pinned_attention::pack_f16_rows(scores[1], &mut packed1[..second_len])?;
        return run_tiles([&packed0[..first_len], &packed1[..second_len]]);
    }
    let [buf0, buf1] = packed_scratch;
    let packed0 = resize_packed_rows(buf0, first_len)?;
    pinned_attention::pack_f16_rows(scores[0], packed0)?;
    let packed1 = resize_packed_rows(buf1, second_len)?;
    pinned_attention::pack_f16_rows(scores[1], packed1)?;
    run_tiles([&*packed0, &*packed1])
}

#[expect(
    clippy::too_many_arguments,
    reason = "Paired causal six-head group scoring and publication borrow cache and caller score storage."
)]
fn attend_group_pair(
    geometry: &Geometry,
    head: usize,
    position: usize,
    queries: [&[f32]; 2],
    projected: [&[f32]; 2],
    keys: &KvCache,
    values: &KvCache,
    current_keys: [&[f32]; 2],
    current_values: [&[f32]; 2],
    outputs: [&mut [f32]; 2],
    first_scores: &mut [f32],
    second_scores: &mut [f32],
    scale: f32,
    packed_scratch: &mut [Vec<[u16; 6]>; 2],
) -> Result<()> {
    let width = geometry.head;
    let offset = head / 6 * width;
    let query_rows: [[&[f32]; 6]; 2] = std::array::from_fn(|token| {
        std::array::from_fn(|index| {
            &queries[token][(head + index) * width..(head + index + 1) * width]
        })
    });
    let mut score_rows = [
        rows(&mut first_scores[..6 * (position + 1)], position + 1)?,
        rows(&mut second_scores[..6 * (position + 2)], position + 2)?,
    ];
    for (cached, key) in keys.tokens().take(position + 1).enumerate() {
        if cached == position {
            // Only the second token sees the first token's F16 key;
            // each token's own key retains the original F32 path.
            let dots = six::dot(&key[offset..offset + width], query_rows[1])?;
            for (score, dot) in score_rows[1].iter_mut().zip(dots) {
                score[cached] = dot * scale;
            }
        } else {
            let dots = six::dot_pair(&key[offset..offset + width], query_rows)?;
            for (scores, dots) in score_rows.iter_mut().zip(dots) {
                for (score, dot) in scores.iter_mut().zip(dots) {
                    score[cached] = dot * scale;
                }
            }
        }
    }
    for (token, scores) in score_rows.iter_mut().enumerate() {
        for (score, query) in scores.iter_mut().zip(query_rows[token]) {
            score[position + token] =
                pinned_attention::dot_f32(&current_keys[token][offset..offset + width], query)?
                    * scale;
            pinned_attention::softmax(score)?;
        }
    }
    let [first_rows, second_rows] = score_rows;
    apply_values_pair(
        geometry,
        head,
        projected,
        values,
        current_values,
        outputs,
        [
            first_rows.each_ref().map(|score| &score[..]),
            second_rows.each_ref().map(|score| &score[..]),
        ],
        position,
        packed_scratch,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "Validate full six-head prefill tile invariants before chunk or KV-group dispatch."
)]
fn validate_tile_inputs(
    geometry: &Geometry,
    first_position: usize,
    queries: &[Vec<f32>],
    projected: &[Vec<f32>],
    keys: &KvCache,
    values: &KvCache,
    current_keys: &[Vec<f32>],
    current_values: &[Vec<f32>],
    count: usize,
    scores_len: usize,
    score_stride: usize,
) -> Result<()> {
    let score_extent = first_position
        .checked_add(count)
        .and_then(|positions| positions.checked_mul(6));
    if count == 0
        || count > 32
        || geometry.kv_heads == 0
        || !geometry.heads.is_multiple_of(6)
        || geometry.heads / geometry.kv_heads != 6
        || queries.len() != count
        || projected.len() != count
        || current_keys.len() != count
        || current_values.len() != count
        || count.checked_mul(score_stride) != Some(scores_len)
        || score_extent.is_none_or(|extent| score_stride < extent)
    {
        return Err(invalid("invalid six-head attention tile geometry"));
    }
    let width = geometry.head;
    let last_position = first_position
        .checked_add(count - 1)
        .ok_or_else(|| invalid("six-head attention position overflow"))?;
    if width == 0
        || keys.width != geometry.kv_heads * width
        || values.width != keys.width
        || last_position > keys.len / keys.width
        || last_position > values.len / values.width
        || queries
            .iter()
            .any(|row| row.len() != geometry.heads * width)
        || projected
            .iter()
            .any(|row| row.len() != 2 * geometry.heads * width)
        || current_keys.iter().any(|row| row.len() != keys.width)
        || current_values.iter().any(|row| row.len() != values.width)
    {
        return Err(invalid("invalid six-head attention paired causal geometry"));
    }
    Ok(())
}

// Each caller worker owns complete token rows and its six-head score storage.
// Keep the cache owner immutable and reuse one six-head group at a time rather
// than falling back to two independent tail heads in Swift's 6:1 GQA geometry.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "Retained reference chunk kernel tested directly alongside balanced dispatch."
    )
)]
#[expect(
    clippy::too_many_arguments,
    reason = "Paired causal scoring and publication retain borrowed token rows, exact extents and caller-owned score storage."
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
    let width = geometry.head;
    if outputs
        .iter()
        .any(|output| output.len() != geometry.heads * width)
    {
        return Err(invalid("invalid six-head attention tile geometry"));
    }
    validate_tile_inputs(
        geometry,
        first_position,
        queries,
        projected,
        keys,
        values,
        current_keys,
        current_values,
        count,
        scores.len(),
        score_stride,
    )?;
    let mut packed_scratch = [Vec::new(), Vec::new()];
    for (pair_index, pair_output) in outputs.chunks_mut(2).enumerate() {
        let first = pair_index * 2;
        let position = first_position + first;
        let pair_scores =
            &mut scores[first * score_stride..(first + pair_output.len()) * score_stride];
        let [first_output, second_output] = pair_output else {
            let output = &mut pair_output[0];
            for head in (0..geometry.heads).step_by(6) {
                attend_prefix_with_scratch(
                    geometry,
                    head,
                    &queries[first],
                    &projected[first],
                    keys,
                    values,
                    position,
                    &current_keys[first],
                    &current_values[first],
                    &mut output[head * width..(head + 6) * width],
                    &mut pair_scores[..6 * (position + 1)],
                    scale,
                    &mut packed_scratch[0],
                )?;
            }
            continue;
        };
        let (first_scores, second_scores) = pair_scores.split_at_mut(score_stride);
        for head in (0..geometry.heads).step_by(6) {
            attend_group_pair(
                geometry,
                head,
                position,
                [&queries[first], &queries[first + 1]],
                [&projected[first], &projected[first + 1]],
                keys,
                values,
                [&current_keys[first], &current_keys[first + 1]],
                [&current_values[first], &current_values[first + 1]],
                [
                    &mut first_output[head * width..(head + 6) * width],
                    &mut second_output[head * width..(head + 6) * width],
                ],
                first_scores,
                second_scores,
                scale,
                &mut packed_scratch,
            )?;
        }
    }
    Ok(())
}

/// Evaluates one six-head KV group (`head..head + 6`) across a contiguous causal
/// token chunk into a packed `[chunk_tokens][6][width]` output slice.
///
/// # Errors
/// Rejects malformed geometry, out-of-range group heads, oversized chunks, and
/// nonfinite causal reductions before returning.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "Retained reference KV-group kernel tested directly alongside balanced dispatch."
    )
)]
#[expect(
    clippy::too_many_arguments,
    reason = "Disjoint 2D (token_chunk, kv_head) prefill tasks borrow immutable prompt slices and write disjoint output/score slices."
)]
pub(super) fn attend_tile_kv_group(
    geometry: &Geometry,
    head: usize,
    first_position: usize,
    queries: &[Vec<f32>],
    projected: &[Vec<f32>],
    keys: &KvCache,
    values: &KvCache,
    current_keys: &[Vec<f32>],
    current_values: &[Vec<f32>],
    outputs: &mut [f32],
    scores: &mut [f32],
    score_stride: usize,
    scale: f32,
) -> Result<()> {
    let count = queries.len();
    let width = geometry.head;
    if !head.is_multiple_of(6)
        || head + 6 > geometry.heads
        || outputs.len() != count.saturating_mul(6 * width)
    {
        return Err(invalid("invalid six-head attention tile geometry"));
    }
    validate_tile_inputs(
        geometry,
        first_position,
        queries,
        projected,
        keys,
        values,
        current_keys,
        current_values,
        count,
        scores.len(),
        score_stride,
    )?;
    let group_stride = 6 * width;
    let mut packed_scratch = [Vec::new(), Vec::new()];
    for (pair_index, pair_output) in outputs.chunks_mut(2 * group_stride).enumerate() {
        let first = pair_index * 2;
        let pair_tokens = pair_output.len() / group_stride;
        let position = first_position + first;
        let pair_scores = &mut scores[first * score_stride..(first + pair_tokens) * score_stride];
        if pair_tokens == 1 {
            attend_prefix_with_scratch(
                geometry,
                head,
                &queries[first],
                &projected[first],
                keys,
                values,
                position,
                &current_keys[first],
                &current_values[first],
                pair_output,
                &mut pair_scores[..6 * (position + 1)],
                scale,
                &mut packed_scratch[0],
            )?;
            continue;
        }
        let outputs_pair = pair_output.split_at_mut(group_stride).into();
        let (first_scores, second_scores) = pair_scores.split_at_mut(score_stride);
        attend_group_pair(
            geometry,
            head,
            position,
            [&queries[first], &queries[first + 1]],
            [&projected[first], &projected[first + 1]],
            keys,
            values,
            [&current_keys[first], &current_keys[first + 1]],
            [&current_values[first], &current_values[first + 1]],
            outputs_pair,
            first_scores,
            second_scores,
            scale,
            &mut packed_scratch,
        )?;
    }
    Ok(())
}

/// Reusable scratch storage for causal-cost-balanced 2D `(token_pair, kv_head)`
/// six-head FP16 prefill tile attention.
#[derive(Default)]
pub(super) struct SixHeadTileScratch {
    kv_major_outputs: Vec<f32>,
    worker_scores: Vec<f32>,
    packed_scratch: Vec<[Vec<[u16; 6]>; 2]>,
}

struct BalancedTileContext<'a> {
    geometry: &'a Geometry,
    first_position: usize,
    queries: &'a [Vec<f32>],
    projected: &'a [Vec<f32>],
    keys: &'a KvCache,
    values: &'a KvCache,
    current_keys: &'a [Vec<f32>],
    current_values: &'a [Vec<f32>],
    scale: f32,
}

impl BalancedTileContext<'_> {
    fn run_single(
        &self,
        token: usize,
        kv_head: usize,
        output: &mut [f32],
        scores: &mut [f32],
        packed_scratch: &mut Vec<[u16; 6]>,
    ) -> std::result::Result<(), (usize, usize, Error)> {
        let head = kv_head * 6;
        let position = self.first_position + token;
        let needed_scores = 6 * (position + 1);
        attend_prefix_with_scratch(
            self.geometry,
            head,
            &self.queries[token],
            &self.projected[token],
            self.keys,
            self.values,
            position,
            &self.current_keys[token],
            &self.current_values[token],
            output,
            &mut scores[..needed_scores],
            self.scale,
            packed_scratch,
        )
        .map_err(|err| (token, kv_head, err))
    }

    fn run_pair(
        &self,
        token0: usize,
        kv_head: usize,
        output: &mut [f32],
        scores: &mut [f32],
        score_stride: usize,
        packed_scratch: &mut [Vec<[u16; 6]>; 2],
    ) -> std::result::Result<(), (usize, usize, Error)> {
        let token1 = token0 + 1;
        let head = kv_head * 6;
        let position = self.first_position + token0;
        let group_stride = 6 * self.geometry.head;
        let (out0, out1) = output.split_at_mut(group_stride);
        let (first_scores, second_scores) = scores.split_at_mut(score_stride);
        if let Err(pair_err) = attend_group_pair(
            self.geometry,
            head,
            position,
            [&self.queries[token0], &self.queries[token1]],
            [&self.projected[token0], &self.projected[token1]],
            self.keys,
            self.values,
            [&self.current_keys[token0], &self.current_keys[token1]],
            [&self.current_values[token0], &self.current_values[token1]],
            [out0, out1],
            first_scores,
            second_scores,
            self.scale,
            packed_scratch,
        ) {
            // Preserve token-0 error precedence even when a shared pair fails.
            self.run_single(token0, kv_head, out0, first_scores, &mut packed_scratch[0])?;
            self.run_single(token1, kv_head, out1, second_scores, &mut packed_scratch[1])?;
            return Err((token0, kv_head, pair_err));
        }
        Ok(())
    }

    fn run_item(
        &self,
        item: &mut BalancedWorkItem<'_>,
        scores: &mut [f32],
        score_stride: usize,
        packed_scratch: &mut [Vec<[u16; 6]>; 2],
    ) -> std::result::Result<(), (usize, usize, Error)> {
        if item.token_len == 2 {
            self.run_pair(
                item.token_start,
                item.kv_head,
                item.output,
                scores,
                score_stride,
                packed_scratch,
            )
        } else {
            self.run_single(
                item.token_start,
                item.kv_head,
                item.output,
                scores,
                &mut packed_scratch[0],
            )
        }
    }
}

struct BalancedWorkItem<'a> {
    token_start: usize,
    token_len: usize,
    kv_head: usize,
    output: &'a mut [f32],
}

struct BalancedWorkerTask<'a> {
    items: &'a mut [Option<BalancedWorkItem<'a>>],
    scores: &'a mut [f32],
    packed: &'a mut [Vec<[u16; 6]>; 2],
    error: Option<(usize, usize, Error)>,
}

#[expect(
    clippy::too_many_arguments,
    reason = "Validate full six-head prefill tile invariants before causal-balanced dispatch."
)]
fn validate_balanced_tile_inputs(
    geometry: &Geometry,
    first_position: usize,
    queries: &[Vec<f32>],
    projected: &[Vec<f32>],
    keys: &KvCache,
    values: &KvCache,
    current_keys: &[Vec<f32>],
    current_values: &[Vec<f32>],
    attended: &[Vec<f32>],
) -> Result<()> {
    let count = queries.len();
    let last_position = count
        .checked_sub(1)
        .and_then(|last| first_position.checked_add(last))
        .ok_or_else(|| invalid("invalid six-head attention tile geometry"))?;
    let width = geometry.head;
    if count == 0
        || count > crate::MAX_PREFILL_BLOCK_TOKENS
        || width == 0
        || geometry.kv_heads == 0
        || !geometry.heads.is_multiple_of(6)
        || geometry.heads / geometry.kv_heads != 6
        || projected.len() != count
        || current_keys.len() != count
        || current_values.len() != count
        || attended.len() != count
        || keys.width != checked_product(&[geometry.kv_heads, width])?
        || values.width != keys.width
        || last_position > keys.len / keys.width
        || last_position > values.len / values.width
        || queries
            .iter()
            .any(|row| row.len() != geometry.heads * width)
        || projected
            .iter()
            .any(|row| row.len() != 2 * geometry.heads * width)
        || attended
            .iter()
            .any(|row| row.len() != geometry.heads * width)
        || current_keys.iter().any(|row| row.len() != keys.width)
        || current_values.iter().any(|row| row.len() != values.width)
    {
        return Err(invalid("invalid six-head attention balanced tile geometry"));
    }
    Ok(())
}

fn assign_causal_balanced_items(
    costs: &[usize],
    token_starts: &[usize],
    kv_heads_idx: &[usize],
    workers: usize,
    assigned: &mut [usize],
    counts: &mut [usize; MAX_ROW_WORKERS],
) -> Result<(usize, [usize; MAX_ROW_WORKERS])> {
    let active_workers = costs.len().min(workers.max(1)).min(MAX_ROW_WORKERS);
    counts.fill(0);
    let mut loads = [0_usize; MAX_ROW_WORKERS];
    if active_workers == 0 {
        return Ok((0, loads));
    }
    let mut order = Vec::new();
    order.try_reserve_exact(costs.len())?;
    order.extend(0..costs.len());
    order.sort_by(|&lhs, &rhs| {
        costs[rhs]
            .cmp(&costs[lhs])
            .then_with(|| token_starts[lhs].cmp(&token_starts[rhs]))
            .then_with(|| kv_heads_idx[lhs].cmp(&kv_heads_idx[rhs]))
    });
    for idx in order {
        let mut best_worker = 0_usize;
        let mut best_load = loads[0];
        for (worker, &load) in loads[..active_workers].iter().enumerate().skip(1) {
            if load < best_load {
                best_load = load;
                best_worker = worker;
            }
        }
        loads[best_worker] = loads[best_worker].saturating_add(costs[idx]);
        counts[best_worker] += 1;
        assigned[idx] = best_worker;
    }
    Ok((active_workers, loads))
}

fn resize_balanced_f32s(count: usize, output: &mut Vec<f32>) -> Result<()> {
    if count > output.len() {
        output.try_reserve_exact(count - output.len())?;
    }
    output.resize(count, 0.0);
    Ok(())
}

/// Evaluates all six-head KV groups across a prefill tile using causal-cost-balanced
/// 2D `(token_pair, kv_head)` scheduling with 0 ULP parity against serial decoding.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "Wired into tiled FP16 prefill during integration and tested across worker counts."
    )
)]
#[expect(
    clippy::too_many_arguments,
    reason = "Matches caller-owned projections, KV history caches, outputs, and row worker pool."
)]
pub(super) fn attend_tile_balanced(
    geometry: &Geometry,
    first_position: usize,
    queries: &[Vec<f32>],
    projected: &[Vec<f32>],
    keys: &KvCache,
    values: &KvCache,
    current_keys: &[Vec<f32>],
    current_values: &[Vec<f32>],
    attended: &mut [Vec<f32>],
    scale: f32,
    parallelism: &RowParallelism,
) -> Result<()> {
    let mut scratch = SixHeadTileScratch::default();
    attend_tile_balanced_with_scratch(
        geometry,
        first_position,
        queries,
        projected,
        keys,
        values,
        current_keys,
        current_values,
        attended,
        &mut [],
        scale,
        &mut scratch,
        parallelism,
    )
}

/// Evaluates all six-head KV groups across a prefill tile with reusable scratch storage
/// and causal-cost-balanced 2D `(token_pair, kv_head)` worker scheduling.
#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "Coordinates validation, 2D LPT scheduling, parallel execution, and scatter."
)]
pub(super) fn attend_tile_balanced_with_scratch(
    geometry: &Geometry,
    first_position: usize,
    queries: &[Vec<f32>],
    projected: &[Vec<f32>],
    keys: &KvCache,
    values: &KvCache,
    current_keys: &[Vec<f32>],
    current_values: &[Vec<f32>],
    attended: &mut [Vec<f32>],
    caller_scores: &mut [f32],
    scale: f32,
    tile_scratch: &mut SixHeadTileScratch,
    parallelism: &RowParallelism,
) -> Result<()> {
    let result = (|| {
        validate_balanced_tile_inputs(
            geometry,
            first_position,
            queries,
            projected,
            keys,
            values,
            current_keys,
            current_values,
            attended,
        )?;
        let count = queries.len();
        let end = checked_sum(first_position, count)?;
        let width = geometry.head;
        let kv_heads = geometry.kv_heads;
        let group_stride = checked_product(&[6, width])?;
        let total_out_elems = checked_product(&[count, kv_heads, group_stride])?;
        resize_balanced_f32s(total_out_elems, &mut tile_scratch.kv_major_outputs)?;

        let num_token_units = count.div_ceil(2);
        let total_items = checked_product(&[num_token_units, kv_heads])?;
        let active_workers = total_items
            .min(parallelism.workers().max(1))
            .min(MAX_ROW_WORKERS);
        let score_stride = checked_product(&[6, end])?;
        let worker_score_len = if count >= 2 {
            checked_product(&[2, score_stride])?
        } else {
            score_stride
        };
        let caller_worker_slots = caller_scores.len() / worker_score_len;
        let reused_active_workers = caller_worker_slots.min(active_workers);
        let scratch_active_workers = active_workers - reused_active_workers;
        resize_balanced_f32s(
            checked_product(&[scratch_active_workers, worker_score_len])?,
            &mut tile_scratch.worker_scores,
        )?;
        if tile_scratch.packed_scratch.len() < active_workers {
            tile_scratch
                .packed_scratch
                .try_reserve_exact(active_workers - tile_scratch.packed_scratch.len())?;
            tile_scratch
                .packed_scratch
                .resize_with(active_workers, || [Vec::new(), Vec::new()]);
        }

        let ctx = BalancedTileContext {
            geometry,
            first_position,
            queries,
            projected,
            keys,
            values,
            current_keys,
            current_values,
            scale,
        };

        if active_workers == 1 {
            let mut rem_out = &mut tile_scratch.kv_major_outputs[..total_out_elems];
            let worker_scores = if reused_active_workers == 1 {
                &mut caller_scores[..worker_score_len]
            } else {
                &mut tile_scratch.worker_scores[..worker_score_len]
            };
            let packed = &mut tile_scratch.packed_scratch[0];
            let mut earliest: Option<(usize, usize, Error)> = None;
            for u in 0..num_token_units {
                let token_start = u * 2;
                let token_len = (count - token_start).min(2);
                let chunk_len = token_len * group_stride;
                for kv_head in 0..kv_heads {
                    let (out_slice, tail) = std::mem::take(&mut rem_out).split_at_mut(chunk_len);
                    rem_out = tail;
                    let mut item = BalancedWorkItem {
                        token_start,
                        token_len,
                        kv_head,
                        output: out_slice,
                    };
                    if let Err(err) = ctx.run_item(&mut item, worker_scores, score_stride, packed) {
                        let key = (err.0, err.1);
                        if earliest
                            .as_ref()
                            .is_none_or(|(tok, kh, _)| key < (*tok, *kh))
                        {
                            earliest = Some(err);
                        }
                    }
                }
            }
            if let Some((_, _, err)) = earliest {
                return Err(err);
            }
        } else {
            let mut items = Vec::new();
            items.try_reserve_exact(total_items)?;
            let mut costs = Vec::new();
            costs.try_reserve_exact(total_items)?;
            let mut token_starts = Vec::new();
            token_starts.try_reserve_exact(total_items)?;
            let mut kv_heads_idx = Vec::new();
            kv_heads_idx.try_reserve_exact(total_items)?;

            let mut rem_out = &mut tile_scratch.kv_major_outputs[..total_out_elems];
            for u in 0..num_token_units {
                let token_start = u * 2;
                let token_len = (count - token_start).min(2);
                let causal_cost = if token_len == 2 {
                    2_usize
                        .saturating_mul(first_position + token_start)
                        .saturating_add(3)
                } else {
                    (first_position + token_start).saturating_add(1)
                };
                let chunk_len = token_len * group_stride;
                for kv_head in 0..kv_heads {
                    let (out_slice, tail) = std::mem::take(&mut rem_out).split_at_mut(chunk_len);
                    rem_out = tail;
                    costs.push(causal_cost);
                    token_starts.push(token_start);
                    kv_heads_idx.push(kv_head);
                    items.push(BalancedWorkItem {
                        token_start,
                        token_len,
                        kv_head,
                        output: out_slice,
                    });
                }
            }
            let mut assigned = Vec::new();
            assigned.try_reserve_exact(total_items)?;
            assigned.resize(total_items, 0);
            let mut counts = [0_usize; MAX_ROW_WORKERS];
            let (used_workers, _) = assign_causal_balanced_items(
                &costs,
                &token_starts,
                &kv_heads_idx,
                active_workers,
                &mut assigned,
                &mut counts,
            )?;

            let mut cursor = [0_usize; MAX_ROW_WORKERS];
            let mut running = 0_usize;
            for (c_off, &cnt) in cursor[..used_workers]
                .iter_mut()
                .zip(&counts[..used_workers])
            {
                *c_off = running;
                running += cnt;
            }
            let mut ordered: Vec<Option<BalancedWorkItem<'_>>> = Vec::new();
            ordered.try_reserve_exact(total_items)?;
            ordered.resize_with(total_items, || None);
            for (item, worker) in items.into_iter().zip(assigned) {
                let slot = cursor[worker];
                cursor[worker] += 1;
                ordered[slot] = Some(item);
            }

            let mut tasks: [Option<BalancedWorkerTask<'_>>; MAX_ROW_WORKERS] =
                std::array::from_fn(|_| None);
            let mut rem_ordered = ordered.as_mut_slice();
            let used_reused_workers = caller_worker_slots.min(used_workers);
            let used_scratch_workers = used_workers - used_reused_workers;
            let mut rem_caller_scores =
                &mut caller_scores[..used_reused_workers * worker_score_len];
            let mut rem_scores =
                &mut tile_scratch.worker_scores[..used_scratch_workers * worker_score_len];
            let mut rem_packed = &mut tile_scratch.packed_scratch[..used_workers];
            for (worker, task) in tasks[..used_workers].iter_mut().enumerate() {
                let (item_chunk, item_tail) =
                    std::mem::take(&mut rem_ordered).split_at_mut(counts[worker]);
                rem_ordered = item_tail;
                let score_chunk = if worker < used_reused_workers {
                    let (chunk, score_tail) =
                        std::mem::take(&mut rem_caller_scores).split_at_mut(worker_score_len);
                    rem_caller_scores = score_tail;
                    chunk
                } else {
                    let (chunk, score_tail) =
                        std::mem::take(&mut rem_scores).split_at_mut(worker_score_len);
                    rem_scores = score_tail;
                    chunk
                };
                let (packed_pair, packed_tail) = std::mem::take(&mut rem_packed).split_at_mut(1);
                rem_packed = packed_tail;
                *task = Some(BalancedWorkerTask {
                    items: item_chunk,
                    scores: score_chunk,
                    packed: &mut packed_pair[0],
                    error: None,
                });
            }

            parallelism.update_chunks(&mut tasks[..used_workers], 1, |_, worker_slice| {
                for task in worker_slice.iter_mut().flatten() {
                    for item in task.items.iter_mut().flatten() {
                        if let Err(err) = ctx.run_item(item, task.scores, score_stride, task.packed)
                        {
                            let key = (err.0, err.1);
                            if task
                                .error
                                .as_ref()
                                .is_none_or(|(tok, kh, _)| key < (*tok, *kh))
                            {
                                task.error = Some(err);
                            }
                        }
                    }
                }
                Ok(())
            })?;

            let earliest = tasks[..used_workers]
                .iter_mut()
                .filter_map(|task| task.as_mut().and_then(|t| t.error.take()))
                .min_by_key(|(tok, kh, _)| (*tok, *kh));
            if let Some((_, _, err)) = earliest {
                return Err(err);
            }
        }

        let mut rem_out = &tile_scratch.kv_major_outputs[..total_out_elems];
        for u in 0..num_token_units {
            let token_start = u * 2;
            let token_len = (count - token_start).min(2);
            let chunk_len = token_len * group_stride;
            for kv_head in 0..kv_heads {
                let (kv_block, tail) = rem_out.split_at(chunk_len);
                rem_out = tail;
                let dst_offset = kv_head * group_stride;
                for local_tok in 0..token_len {
                    let src = &kv_block[local_tok * group_stride..(local_tok + 1) * group_stride];
                    attended[token_start + local_tok][dst_offset..dst_offset + group_stride]
                        .copy_from_slice(src);
                }
            }
        }
        Ok(())
    })();
    if result.is_err() {
        for row in attended {
            row.fill(0.0);
        }
    }
    result
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
