//! Exact six-query-head sharing for Swift's 24-query/4-KV-head decode geometry.

use crate::{Result, invalid};

/// Six heads by 64 channels by sixteen temporal FMA lanes occupy 24 KiB.
pub(crate) const VALUE_TILE: usize = 64;

// Twice the query rows at half the channel width retain the 24 KiB bound.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "Retain score-chunked fallback for direct paired value kernel parity tests."
    )
)]
#[path = "pinned_attention_pair_values.rs"]
mod pair_values;
pub(crate) use pair_values::{PAIR_VALUE_TILE, value_pair_tile_packed_gated};

/// Reads and converts each cached F16 key element once across six query heads.
///
/// # Errors
/// Rejects empty, mismatched or oversized rows and nonfinite reductions.
pub(crate) fn dot(key: &[u16], queries: [&[f32]; 6]) -> Result<[f32; 6]> {
    dot_many(key, queries)
}

/// Reads and converts each cached F16 key element once across `HEADS` query heads
/// (`HEADS` in `{2, 3, 6}`) sharing a six-head KV group.
///
/// # Errors
/// Rejects unsupported group sizes, empty, mismatched or oversized rows and
/// nonfinite reductions.
pub(crate) fn dot_group<const HEADS: usize>(
    key: &[u16],
    queries: [&[f32]; HEADS],
) -> Result<[f32; HEADS]> {
    if !matches!(HEADS, 2 | 3 | 6) {
        return Err(invalid("invalid six-group query head count"));
    }
    dot_many(key, queries)
}

/// Shares one cached key conversion across both tokens of a six-head pair.
///
/// Uses a zero-spill two-pass half-stride NEON reduction (`dot_pair_six_neon`) on
/// ARM64 Windows MSVC NEON to stay within 28 vector registers with 0-ULP parity.
///
/// # Errors
/// Rejects malformed rows and nonfinite reductions with the original lane tree.
pub(crate) fn dot_pair(key: &[u16], queries: [[&[f32]; 6]; 2]) -> Result<[[f32; 6]; 2]> {
    let len = key.len();
    if len == 0
        || len > super::MAX_ATTENTION_HEAD_DIM
        || queries.iter().flatten().any(|query| query.len() != len)
    {
        return Err(invalid(
            "invalid pinned six-head paired attention dot width",
        ));
    }
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(neon) = fearless_simd::Level::baseline().as_neon() {
        return dot_pair_six_neon(neon, key, queries);
    }
    let [first, second] = queries;
    let values = dot_many::<12>(
        key,
        std::array::from_fn(|head| {
            if head < 6 {
                first[head]
            } else {
                second[head - 6]
            }
        }),
    )?;
    Ok(std::array::from_fn(|token| {
        std::array::from_fn(|head| values[token * 6 + head])
    }))
}

/// Evaluates two cached F16 keys against `HEADS` (`HEADS` in `{2, 3, 6}`) query heads
/// sharing a six-head KV group, halving query loads while preserving the exact lane tree.
///
/// # Errors
/// Rejects unsupported group sizes, empty, mismatched or oversized rows and
/// nonfinite reductions.
pub(crate) fn dot_two_keys<const HEADS: usize>(
    keys: [&[u16]; 2],
    queries: [&[f32]; HEADS],
) -> Result<[[f32; HEADS]; 2]> {
    let len = keys[0].len();
    if !matches!(HEADS, 2 | 3 | 6)
        || len == 0
        || len > super::MAX_ATTENTION_HEAD_DIM
        || keys[1].len() != len
        || queries.iter().any(|query| query.len() != len)
    {
        return Err(invalid("invalid pinned two-key group attention dot width"));
    }
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(neon) = fearless_simd::Level::baseline().as_neon() {
        return dot_two_keys_six_neon(neon, keys, queries);
    }
    Ok([dot_many(keys[0], queries)?, dot_many(keys[1], queries)?])
}

fn dot_many<const HEADS: usize>(key: &[u16], queries: [&[f32]; HEADS]) -> Result<[f32; HEADS]> {
    let len = key.len();
    if len == 0
        || len > super::MAX_ATTENTION_HEAD_DIM
        || queries.iter().any(|query| query.len() != len)
    {
        return Err(invalid("invalid pinned six-head attention dot width"));
    }
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(neon) = fearless_simd::Level::baseline().as_neon() {
        return dot_neon(neon, key, queries);
    }
    let mut sums = [[0.0_f32; 16]; HEADS];
    for (index, &value) in key.iter().enumerate() {
        let value = bonsai_ternary::half_to_float(value);
        for (head, sum) in sums.iter_mut().enumerate() {
            sum[index % 16] = value.mul_add(queries[head][index], sum[index % 16]);
        }
    }
    for index in len..super::padded_dot_steps(len) {
        for sum in &mut sums {
            sum[index % 16] = 0.0_f32.mul_add(0.0, sum[index % 16]);
        }
    }
    let mut output = [0.0; HEADS];
    for (output, sum) in output.iter_mut().zip(sums) {
        *output = reduce(&sum)?;
    }
    Ok(output)
}

fn reduce(sum: &[f32; 16]) -> Result<f32> {
    let lanes: [f32; 4] =
        std::array::from_fn(|lane| (sum[lane] + sum[8 + lane]) + (sum[4 + lane] + sum[12 + lane]));
    crate::kernels::finite((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
}

/// Reduces one KV head for all six query heads, then applies original gates.
///
/// F16 probabilities and each channel's native sixteen-lane FMA tree remain
/// unchanged. All reductions succeed before any gated output is published.
///
/// # Errors
/// Rejects invalid extents, cache-row counts and nonfinite reductions before
/// publishing output. Gates retain the existing separate sigmoid arithmetic.
#[allow(
    dead_code,
    reason = "Retain unpacked F32 six-head value tile baseline for direct parity tests."
)]
pub(crate) fn value_tile_gated<'a>(
    past: impl Iterator<Item = &'a [u16]>,
    current: &[f32],
    weights: [&[f32]; 6],
    offset: usize,
    outputs: [&mut [f32]; 6],
    gates: [&[f32]; 6],
) -> Result<()> {
    let width = current.len();
    let count = weights[0].len();
    if width == 0
        || width > VALUE_TILE
        || count == 0
        || count > crate::model::swift::CONTEXT_TOKENS
        || outputs.iter().any(|output| output.len() != width)
        || weights.iter().any(|weight| weight.len() != count)
        || gates.iter().any(|gate| gate.len() != width)
    {
        return Err(invalid("invalid pinned six-head attention value tile"));
    }
    let end = offset
        .checked_add(width)
        .ok_or_else(|| invalid("pinned attention value offset overflow"))?;
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if width.is_multiple_of(4)
        && let Some(neon) = fearless_simd::Level::baseline().as_neon()
    {
        return value_tile_neon(neon, past, current, weights, &(offset..end), outputs, gates);
    }
    value_tile_scalar(past, current, weights, &(offset..end), outputs, gates)
}

/// Reduces one KV head for `HEADS` query heads (`HEADS` in `{2, 3, 6}`) using
/// `[token][head]` packed binary16 weights, then applies original gates.
///
/// # Errors
/// Rejects invalid group sizes, extents, cache-row counts, nonfinite inputs
/// and nonfinite reductions before publishing any output.
pub(crate) fn value_tile_group_packed_gated<'a, const HEADS: usize>(
    past: impl Iterator<Item = &'a [u16]>,
    current: &[f32],
    weights: &[[u16; HEADS]],
    offset: usize,
    mut outputs: [&mut [f32]; HEADS],
    gates: [&[f32]; HEADS],
) -> Result<()> {
    let width = current.len();
    let count = weights.len();
    if !matches!(HEADS, 2 | 3 | 6)
        || width == 0
        || width > VALUE_TILE
        || count == 0
        || count > crate::model::swift::CONTEXT_TOKENS
        || outputs.iter().any(|output| output.len() != width)
        || gates.iter().any(|gate| gate.len() != width)
        || weights
            .iter()
            .flatten()
            .any(|&bits| (bits & 0x7c00) == 0x7c00)
        || current.iter().any(|value| !value.is_finite())
    {
        return Err(invalid("invalid pinned group attention value tile"));
    }
    let end = offset
        .checked_add(width)
        .ok_or_else(|| invalid("pinned attention value offset overflow"))?;
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if width.is_multiple_of(4)
        && let Some(neon) = fearless_simd::Level::baseline().as_neon()
    {
        return neon::value_tile_group_packed_neon(
            neon,
            past,
            current,
            weights,
            &(offset..end),
            outputs,
            gates,
        );
    }
    value_tile_group_packed_scalar(past, current, weights, &(offset..end), &mut outputs, gates)
}

fn accumulate_group_packed_chunk_scalar<'a, const HEADS: usize>(
    past: &mut impl Iterator<Item = &'a [u16]>,
    weights: &[[u16; HEADS]],
    range: &std::ops::Range<usize>,
    sums: &mut [[[f32; 16]; VALUE_TILE]; HEADS],
) -> Result<()> {
    for (token, weight_bits) in weights.iter().enumerate() {
        let values = past
            .next()
            .and_then(|row| row.get(range.start..range.end))
            .ok_or_else(|| invalid("pinned attention value row is short"))?;
        let lane = token % 16;
        let decoded_weights = weight_bits.map(bonsai_ternary::half_to_float);
        for (channel, &bits) in values.iter().enumerate() {
            let value = bonsai_ternary::half_to_float(bits);
            for (head_sums, &weight) in sums.iter_mut().zip(&decoded_weights) {
                head_sums[channel][lane] = value.mul_add(weight, head_sums[channel][lane]);
            }
        }
    }
    Ok(())
}

fn finish_group_packed_scalar<'a, const HEADS: usize>(
    mut past: impl Iterator<Item = &'a [u16]>,
    current: &[f32],
    last_bits: [u16; HEADS],
    past_len: usize,
    sums: &mut [[[f32; 16]; VALUE_TILE]; HEADS],
    outputs: &mut [&mut [f32]; HEADS],
    gates: [&[f32]; HEADS],
) -> Result<()> {
    if past.next().is_some() {
        return Err(invalid("pinned attention value row is long"));
    }
    let width = current.len();
    let last_lane = past_len % 16;
    let last_weights = last_bits.map(bonsai_ternary::half_to_float);
    for (head_sums, &weight) in sums.iter_mut().zip(&last_weights) {
        for (channel, &value) in current.iter().enumerate() {
            head_sums[channel][last_lane] = value.mul_add(weight, head_sums[channel][last_lane]);
        }
        for token in past_len + 1..super::padded_dot_steps(past_len + 1) {
            for channel in &mut head_sums[..width] {
                channel[token % 16] = 0.0_f32.mul_add(0.0, channel[token % 16]);
            }
        }
    }
    let mut reduced = [[0.0_f32; VALUE_TILE]; HEADS];
    for (head_reduced, head_sums) in reduced.iter_mut().zip(sums.iter()) {
        for (dst, channel_sums) in head_reduced[..width].iter_mut().zip(&head_sums[..width]) {
            *dst = reduce(channel_sums)?;
        }
    }
    for ((output, head_reduced), gate) in outputs.iter_mut().zip(&reduced).zip(gates) {
        output.copy_from_slice(&head_reduced[..width]);
        crate::kernels::mul_sigmoid_in_place(output, gate);
    }
    Ok(())
}

fn value_tile_group_packed_scalar<'a, const HEADS: usize>(
    mut past: impl Iterator<Item = &'a [u16]>,
    current: &[f32],
    weights: &[[u16; HEADS]],
    range: &std::ops::Range<usize>,
    outputs: &mut [&mut [f32]; HEADS],
    gates: [&[f32]; HEADS],
) -> Result<()> {
    let past_len = weights.len() - 1;
    let mut sums = [[[0.0_f32; 16]; VALUE_TILE]; HEADS];
    accumulate_group_packed_chunk_scalar(&mut past, &weights[..past_len], range, &mut sums)?;
    finish_group_packed_scalar(
        past,
        current,
        weights[past_len],
        past_len,
        &mut sums,
        outputs,
        gates,
    )
}

/// Reduces one KV head for six query heads using chunked stack-packed
/// `[token][head]` binary16 probabilities from normalized `f32` score slices,
/// then applies original gates.
///
/// # Errors
/// Rejects invalid extents, cache-row counts, nonfinite inputs and nonfinite
/// reductions before publishing any output.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "Retain chunked-score kernel for direct kernel parity tests."
    )
)]
pub(crate) fn value_tile_six_from_scores_gated<'a>(
    past: impl Iterator<Item = &'a [u16]>,
    current: &[f32],
    scores: [&[f32]; 6],
    offset: usize,
    mut outputs: [&mut [f32]; 6],
    gates: [&[f32]; 6],
) -> Result<()> {
    let width = current.len();
    let count = scores[0].len();
    if width == 0
        || width > VALUE_TILE
        || count == 0
        || count > crate::model::swift::CONTEXT_TOKENS
        || outputs.iter().any(|output| output.len() != width)
        || scores
            .iter()
            .any(|row| row.len() != count || row.iter().any(|value| !value.is_finite()))
        || gates.iter().any(|gate| gate.len() != width)
        || current.iter().any(|value| !value.is_finite())
    {
        return Err(invalid("invalid pinned six-head attention value tile"));
    }
    let end = offset
        .checked_add(width)
        .ok_or_else(|| invalid("pinned attention value offset overflow"))?;
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if width.is_multiple_of(4)
        && let Some(neon) = fearless_simd::Level::baseline().as_neon()
    {
        return value_tile_six_from_scores_neon(
            neon,
            past,
            current,
            scores,
            &(offset..end),
            outputs,
            gates,
        );
    }
    value_tile_six_from_scores_scalar(past, current, scores, &(offset..end), &mut outputs, gates)
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "Retain chunked-score scalar fallback for direct parity tests."
    )
)]
#[expect(
    clippy::large_stack_arrays,
    reason = "The fixed six-head temporal accumulator is bounded to 24 KiB."
)]
fn value_tile_six_from_scores_scalar<'a>(
    mut past: impl Iterator<Item = &'a [u16]>,
    current: &[f32],
    scores: [&[f32]; 6],
    range: &std::ops::Range<usize>,
    outputs: &mut [&mut [f32]; 6],
    gates: [&[f32]; 6],
) -> Result<()> {
    const PACK_CHUNK: usize = 256;
    let past_len = scores[0].len() - 1;
    let mut sums = [[[0.0_f32; 16]; VALUE_TILE]; 6];
    let mut packed_chunk = [[0_u16; 6]; PACK_CHUNK];
    for chunk_start in (0..past_len).step_by(PACK_CHUNK) {
        let chunk_end = (chunk_start + PACK_CHUNK).min(past_len);
        let chunk_len = chunk_end - chunk_start;
        super::pack_f16_rows(
            scores.map(|row| &row[chunk_start..chunk_end]),
            &mut packed_chunk[..chunk_len],
        )?;
        if packed_chunk[..chunk_len]
            .iter()
            .flatten()
            .any(|&bits| (bits & 0x7c00) == 0x7c00)
        {
            return Err(invalid("nonfinite packed binary16 attention weight"));
        }
        accumulate_group_packed_chunk_scalar(
            &mut past,
            &packed_chunk[..chunk_len],
            range,
            &mut sums,
        )?;
    }
    let last_bits = scores.map(|row| crate::kernels::half_bits(row[past_len]));
    if last_bits.iter().any(|&bits| (bits & 0x7c00) == 0x7c00) {
        return Err(invalid("nonfinite packed binary16 attention weight"));
    }
    finish_group_packed_scalar(
        past, current, last_bits, past_len, &mut sums, outputs, gates,
    )
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "Retain chunked-score NEON decode helper for direct parity tests."
    )
)]
#[inline]
fn decode_six_single_neon(neon: fearless_simd::Neon, weights: [u16; 6]) -> [f32; 6] {
    use fearless_simd::SimdBase as _;
    let mut bits = [0_u16; 8];
    bits[..6].copy_from_slice(&weights);
    let (low, high) = crate::kernels::neon::f16_bits_to_f32_x8(neon, &bits);
    let low_arr = low.to_array();
    let high_arr = high.to_array();
    [
        low_arr[0],
        low_arr[1],
        low_arr[2],
        low_arr[3],
        high_arr[0],
        high_arr[1],
    ]
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "Retain chunked-score NEON pair decode helper for direct parity tests."
    )
)]
#[inline]
fn decode_six_pair_neon(
    neon: fearless_simd::Neon,
    first: [u16; 6],
    second: [u16; 6],
) -> ([f32; 6], [f32; 6]) {
    use fearless_simd::SimdBase as _;
    let combined = [
        first[0], first[1], first[2], first[3], first[4], first[5], second[0], second[1],
    ];
    let (lo, hi) = crate::kernels::neon::f16_bits_to_f32_x8(neon, &combined);
    let lo_arr = lo.to_array();
    let hi_arr = hi.to_array();
    let tail = crate::kernels::neon::f16_bits_to_f32_x4(
        neon,
        [second[2], second[3], second[4], second[5]],
    )
    .to_array();
    (
        [
            lo_arr[0], lo_arr[1], lo_arr[2], lo_arr[3], hi_arr[0], hi_arr[1],
        ],
        [hi_arr[2], hi_arr[3], tail[0], tail[1], tail[2], tail[3]],
    )
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "Retain chunked-score NEON fallback for direct parity tests."
    )
)]
#[expect(
    clippy::large_stack_arrays,
    clippy::too_many_lines,
    reason = "The fixed six-head temporal accumulator is bounded to 24 KiB."
)]
fn value_tile_six_from_scores_neon<'a>(
    neon: fearless_simd::Neon,
    mut past: impl Iterator<Item = &'a [u16]>,
    current: &[f32],
    scores: [&[f32]; 6],
    range: &std::ops::Range<usize>,
    outputs: [&mut [f32]; 6],
    gates: [&[f32]; 6],
) -> Result<()> {
    use fearless_simd::{f32x4, prelude::*};

    const PACK_CHUNK: usize = 256;
    const GROUPS: usize = VALUE_TILE / 4;
    let zero = f32x4::splat(neon, 0.0);
    let mut sums = [[[zero; GROUPS]; 16]; 6];
    let groups = outputs[0].len() / 4;
    let past_len = scores[0].len() - 1;
    let mut packed_chunk = [[0_u16; 6]; PACK_CHUNK];

    for chunk_start in (0..past_len).step_by(PACK_CHUNK) {
        let chunk_end = (chunk_start + PACK_CHUNK).min(past_len);
        let chunk_len = chunk_end - chunk_start;
        super::pack_f16_rows(
            scores.map(|row| &row[chunk_start..chunk_end]),
            &mut packed_chunk[..chunk_len],
        )?;
        if packed_chunk[..chunk_len]
            .iter()
            .flatten()
            .any(|&bits| (bits & 0x7c00) == 0x7c00)
        {
            return Err(invalid("nonfinite packed binary16 attention weight"));
        }
        let (weight_pairs, weight_tail) = packed_chunk[..chunk_len].as_chunks::<2>();
        for (pair_idx, &[w0_bits, w1_bits]) in weight_pairs.iter().enumerate() {
            let row0 = past
                .next()
                .and_then(|row| row.get(range.start..range.end))
                .ok_or_else(|| invalid("pinned attention value row is short"))?;
            let row1 = past
                .next()
                .and_then(|row| row.get(range.start..range.end))
                .ok_or_else(|| invalid("pinned attention value row is short"))?;
            let (w0, w1) = decode_six_pair_neon(neon, w0_bits, w1_bits);
            let w0_splat = w0.map(|weight| f32x4::splat(neon, weight));
            let w1_splat = w1.map(|weight| f32x4::splat(neon, weight));
            let token = pair_idx * 2;
            let lane0 = token % 16;
            let lane1 = (token + 1) % 16;
            let (chunks8_0, rem0) = row0.as_chunks::<8>();
            let (chunks8_1, rem1) = row1.as_chunks::<8>();
            for (pair_ch, (c0, c1)) in chunks8_0.iter().zip(chunks8_1).enumerate() {
                let (v0_0, v0_1) = crate::kernels::neon::f16_bits_to_f32_x8(neon, c0);
                let (v1_0, v1_1) = crate::kernels::neon::f16_bits_to_f32_x8(neon, c1);
                let g0 = pair_ch * 2;
                let g1 = g0 + 1;
                for h in 0..6 {
                    sums[h][lane0][g0] = v0_0.mul_add_precise(w0_splat[h], sums[h][lane0][g0]);
                    sums[h][lane0][g1] = v0_1.mul_add_precise(w0_splat[h], sums[h][lane0][g1]);
                    sums[h][lane1][g0] = v1_0.mul_add_precise(w1_splat[h], sums[h][lane1][g0]);
                    sums[h][lane1][g1] = v1_1.mul_add_precise(w1_splat[h], sums[h][lane1][g1]);
                }
            }
            if let ([vals0], [vals1]) = (rem0.as_chunks::<4>().0, rem1.as_chunks::<4>().0) {
                let v0 = crate::kernels::neon::f16_bits_to_f32_x4(neon, *vals0);
                let v1 = crate::kernels::neon::f16_bits_to_f32_x4(neon, *vals1);
                let g = groups - 1;
                for h in 0..6 {
                    sums[h][lane0][g] = v0.mul_add_precise(w0_splat[h], sums[h][lane0][g]);
                    sums[h][lane1][g] = v1.mul_add_precise(w1_splat[h], sums[h][lane1][g]);
                }
            }
        }
        if let Some(&tail_bits) = weight_tail.first() {
            let row = past
                .next()
                .and_then(|row| row.get(range.start..range.end))
                .ok_or_else(|| invalid("pinned attention value row is short"))?;
            let w_tail = decode_six_single_neon(neon, tail_bits).map(|w| f32x4::splat(neon, w));
            let lane = (weight_pairs.len() * 2) % 16;
            let (chunks8, rem) = row.as_chunks::<8>();
            for (pair_ch, chunk8) in chunks8.iter().enumerate() {
                let (v0, v1) = crate::kernels::neon::f16_bits_to_f32_x8(neon, chunk8);
                let g0 = pair_ch * 2;
                let g1 = g0 + 1;
                for h in 0..6 {
                    sums[h][lane][g0] = v0.mul_add_precise(w_tail[h], sums[h][lane][g0]);
                    sums[h][lane][g1] = v1.mul_add_precise(w_tail[h], sums[h][lane][g1]);
                }
            }
            if let [vals4] = rem.as_chunks::<4>().0 {
                let v = crate::kernels::neon::f16_bits_to_f32_x4(neon, *vals4);
                let g = groups - 1;
                for h in 0..6 {
                    sums[h][lane][g] = v.mul_add_precise(w_tail[h], sums[h][lane][g]);
                }
            }
        }
    }
    if past.next().is_some() {
        return Err(invalid("pinned attention value row is long"));
    }
    let last_lane = past_len % 16;
    let last_bits = scores.map(|row| crate::kernels::half_bits(row[past_len]));
    if last_bits.iter().any(|&bits| (bits & 0x7c00) == 0x7c00) {
        return Err(invalid("nonfinite packed binary16 attention weight"));
    }
    let w_last = decode_six_single_neon(neon, last_bits).map(|w| f32x4::splat(neon, w));
    for (g, current_chunk) in current.as_chunks::<4>().0.iter().enumerate() {
        let v = f32x4::load_array(neon, *current_chunk);
        for h in 0..6 {
            sums[h][last_lane][g] = v.mul_add_precise(w_last[h], sums[h][last_lane][g]);
        }
    }
    for token in past_len + 1..super::padded_dot_steps(past_len + 1) {
        for head_sums in &mut sums {
            for sum in &mut head_sums[token % 16][..groups] {
                *sum = zero.mul_add_precise(zero, *sum);
            }
        }
    }
    let mut finite = zero.is_finite();
    for head_sums in &mut sums {
        for group in 0..groups {
            let lanes: [f32x4<fearless_simd::Neon>; 4] = std::array::from_fn(|lane| {
                (head_sums[lane][group] + head_sums[8 + lane][group])
                    + (head_sums[4 + lane][group] + head_sums[12 + lane][group])
            });
            let result = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
            finite &= result.is_finite();
            head_sums[0][group] = result;
        }
    }
    if !finite.all_true() {
        return Err(invalid("nonfinite pinned attention value reduction"));
    }
    let one = f32x4::splat(neon, 1.0);
    for (head, output) in outputs.into_iter().enumerate() {
        for (group, out) in output.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let exponentials = f32x4::load_array(
                neon,
                std::array::from_fn(|lane| (-gates[head][group * 4 + lane]).exp()),
            );
            (sums[head][0][group] * (one / (one + exponentials))).store_slice(out);
        }
    }
    Ok(())
}

#[expect(
    clippy::large_stack_arrays,
    reason = "The fixed six-head temporal accumulator is bounded to 24 KiB."
)]
fn value_tile_scalar<'a>(
    mut past: impl Iterator<Item = &'a [u16]>,
    current: &[f32],
    weights: [&[f32]; 6],
    range: &std::ops::Range<usize>,
    outputs: [&mut [f32]; 6],
    gates: [&[f32]; 6],
) -> Result<()> {
    let mut sums = [[[0.0_f32; 16]; VALUE_TILE]; 6];
    let last = weights[0].len() - 1;
    for token in 0..last {
        let values = past
            .next()
            .and_then(|row| row.get(range.clone()))
            .ok_or_else(|| invalid("pinned attention value row is short"))?;
        for (channel, &value) in values.iter().enumerate() {
            let value = bonsai_ternary::half_to_float(value);
            for (head, sums) in sums.iter_mut().enumerate() {
                let lane = &mut sums[channel][token % 16];
                *lane = value.mul_add(weights[head][token], *lane);
            }
        }
    }
    if past.next().is_some() {
        return Err(invalid("pinned attention value row is long"));
    }
    for (head, sums) in sums.iter_mut().enumerate() {
        for (channel, &value) in current.iter().enumerate() {
            let lane = &mut sums[channel][last % 16];
            *lane = value.mul_add(weights[head][last], *lane);
        }
        for token in last + 1..super::padded_dot_steps(last + 1) {
            for channel in &mut sums[..current.len()] {
                let lane = &mut channel[token % 16];
                *lane = 0.0_f32.mul_add(0.0, *lane);
            }
        }
    }
    // Reuse dead first-lane accumulator storage for atomic publication.
    for sums in &mut sums {
        for channel in &mut sums[..current.len()] {
            channel[0] = reduce(channel)?;
        }
    }
    for (head, output) in outputs.into_iter().enumerate() {
        for (channel, output) in output.iter_mut().enumerate() {
            *output = sums[head][channel][0] * (1.0 / (1.0 + (-gates[head][channel]).exp()));
        }
    }
    Ok(())
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
#[path = "pinned_attention_six_neon.rs"]
mod neon;

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
use neon::{dot_neon, dot_pair_six_neon, dot_two_keys_six_neon, value_tile_neon};

#[cfg(test)]
#[path = "pinned_attention_six_tests.rs"]
mod tests;
