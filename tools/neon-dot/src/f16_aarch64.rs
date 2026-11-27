//! Ordered binary64 and pinned four-FMA binary16 row and tile reductions on baseline NEON.

#![expect(
    clippy::redundant_pub_crate,
    reason = "Explicit parent-only visibility keeps baseline-NEON F16 intrinsics inside the audited module boundary."
)]

use core::arch::aarch64::{
    float32x4_t, float64x2_t, uint16x8_t, vaddq_f32, vaddq_f64, vaddvq_f32, vandq_u16,
    vcvt_f32_f16, vcvt_f64_f32, vcvt_high_f32_f16, vcvt_high_f64_f32, vdupq_n_f32, vdupq_n_f64,
    vdupq_n_u16, vfmaq_f32, vget_low_f16, vget_low_f32, vld1q_f32, vld1q_u8, vmaxq_u16, vmaxvq_u16,
    vmulq_f32, vreinterpretq_f16_u16, vreinterpretq_u16_u8, vst1q_f32, vst1q_f64, vzip1q_f32,
    vzip2q_f32,
};

#[inline]
#[target_feature(enable = "neon")]
fn load_f32x4(slice: &[f32]) -> float32x4_t {
    debug_assert!(slice.len() >= 4);
    // SAFETY: caller's chunked slice geometry guarantees at least four readable `f32` elements;
    // baseline NEON allows unaligned loads.
    unsafe { vld1q_f32(slice.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_u16x8_bytes(bytes: &[u8; 16]) -> uint16x8_t {
    // SAFETY: fixed 16-byte reference covers the unaligned NEON load.
    let raw = unsafe { vld1q_u8(bytes.as_ptr()) };
    vreinterpretq_u16_u8(raw)
}

#[inline]
#[target_feature(enable = "neon")]
fn widen_f16_x8(
    raw: uint16x8_t,
    magnitude_mask: uint16x8_t,
    weight_magnitude: &mut uint16x8_t,
) -> (float32x4_t, float32x4_t) {
    *weight_magnitude = vmaxq_u16(*weight_magnitude, vandq_u16(raw, magnitude_mask));
    let half = vreinterpretq_f16_u16(raw);
    (vcvt_f32_f16(vget_low_f16(half)), vcvt_high_f32_f16(half))
}

#[inline]
#[target_feature(enable = "neon")]
fn add_columns(mut sum: float64x2_t, first: float32x4_t, second: float32x4_t) -> float64x2_t {
    let low = vzip1q_f32(first, second);
    let high = vzip2q_f32(first, second);
    // Each vector lane retains its independent left-to-right rounded product
    // converted to binary64 and added without reassociation across columns.
    sum = vaddq_f64(sum, vcvt_f64_f32(vget_low_f32(low)));
    sum = vaddq_f64(sum, vcvt_high_f64_f32(low));
    sum = vaddq_f64(sum, vcvt_f64_f32(vget_low_f32(high)));
    vaddq_f64(sum, vcvt_high_f64_f32(high))
}

#[inline]
#[target_feature(enable = "neon")]
fn add_f32x4_to_f64_ordered(mut sum: f64, prod: float32x4_t) -> f64 {
    let mut lanes = [0.0_f32; 4];
    // SAFETY: `lanes` is a unique initialized four-element array.
    unsafe { vst1q_f32(lanes.as_mut_ptr(), prod) };
    sum += f64::from(lanes[0]);
    sum += f64::from(lanes[1]);
    sum += f64::from(lanes[2]);
    sum + f64::from(lanes[3])
}

#[inline]
fn decode_f16_tail(rem: &[u8]) -> Option<([f32; 7], usize)> {
    let mut weights = [0.0_f32; 7];
    let chunks = rem.as_chunks::<2>().0;
    if chunks.len() > weights.len() {
        return None;
    }
    for (dst, bytes) in weights.iter_mut().zip(chunks) {
        let weight = bonsai_ternary::half_to_float(u16::from_le_bytes(*bytes));
        if !weight.is_finite() {
            return None;
        }
        *dst = weight;
    }
    Some((weights, chunks.len()))
}

#[inline]
#[target_feature(enable = "neon")]
fn reduce_pinned_lanes(s: [float32x4_t; 4]) -> Option<f32> {
    let val = vaddvq_f32(vaddq_f32(vaddq_f32(s[0], s[2]), vaddq_f32(s[1], s[3])));
    val.is_finite().then_some(val)
}

#[inline]
pub(super) fn dot_row(row: &[u8], input: &[f32]) -> Option<f32> {
    // SAFETY: the safe parent checks nonempty matching extents; baseline NEON is always present.
    unsafe { dot_row_neon(row, input) }
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Retain float_dot's left-to-right binary64 sum of binary32 products and final binary32 rounding."
)]
#[target_feature(enable = "neon")]
fn dot_row_neon(row: &[u8], input: &[f32]) -> Option<f32> {
    let mut sum = -0.0_f64;
    let mut weight_magnitude = vdupq_n_u16(0);
    let magnitude_mask = vdupq_n_u16(0x7fff);
    let (w_chunks, w_rem) = row.as_chunks::<16>();
    let (in_chunks, in_rem) = input.as_chunks::<8>();
    for (w_bytes, act) in w_chunks.iter().zip(in_chunks) {
        let raw = load_u16x8_bytes(w_bytes);
        let (w0, w1) = widen_f16_x8(raw, magnitude_mask, &mut weight_magnitude);
        let p0 = vmulq_f32(w0, load_f32x4(&act[..4]));
        let p1 = vmulq_f32(w1, load_f32x4(&act[4..]));
        sum = add_f32x4_to_f64_ordered(sum, p0);
        sum = add_f32x4_to_f64_ordered(sum, p1);
    }
    if vmaxvq_u16(weight_magnitude) >= 0x7c00 {
        return None;
    }
    let (tail_w, tail_len) = decode_f16_tail(w_rem)?;
    for (&w, &act) in tail_w[..tail_len].iter().zip(in_rem) {
        sum += f64::from(w * act);
    }
    let value = sum as f32;
    value.is_finite().then_some(value)
}

#[inline]
pub(super) fn dot_rows<const ROWS: usize>(
    rows: [&[u8]; ROWS],
    input: &[f32],
) -> Option<[f32; ROWS]> {
    // SAFETY: the safe parent checks four/eight nonempty rows and exact byte extents.
    unsafe { dot_rows_neon::<ROWS>(rows, input) }
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Each ordered binary64 sum rounds once to the binary32 result."
)]
#[target_feature(enable = "neon")]
fn dot_rows_neon<const ROWS: usize>(rows: [&[u8]; ROWS], input: &[f32]) -> Option<[f32; ROWS]> {
    let mut sums = [vdupq_n_f64(-0.0); 4];
    let mut weight_magnitude = vdupq_n_u16(0);
    let magnitude_mask = vdupq_n_u16(0x7fff);
    let (chunks, tail) = input.as_chunks::<8>();
    let r_chunks = rows.map(|row| row.as_chunks::<16>().0);
    let r_rems = rows.map(|row| row.as_chunks::<16>().1);
    for (column, activation) in chunks.iter().enumerate() {
        let first = load_f32x4(&activation[..4]);
        let second = load_f32x4(&activation[4..]);
        for (pair, sum) in sums[..ROWS / 2].iter_mut().enumerate() {
            let (we0, we1) = widen_f16_x8(
                load_u16x8_bytes(&r_chunks[pair * 2][column]),
                magnitude_mask,
                &mut weight_magnitude,
            );
            let (wo0, wo1) = widen_f16_x8(
                load_u16x8_bytes(&r_chunks[pair * 2 + 1][column]),
                magnitude_mask,
                &mut weight_magnitude,
            );
            *sum = add_columns(*sum, vmulq_f32(we0, first), vmulq_f32(wo0, first));
            *sum = add_columns(*sum, vmulq_f32(we1, second), vmulq_f32(wo1, second));
        }
    }
    if vmaxvq_u16(weight_magnitude) >= 0x7c00 {
        return None;
    }
    let mut output = [0.0_f32; ROWS];
    for (pair, sum) in sums[..ROWS / 2].iter().enumerate() {
        let mut values = [0.0_f64; 2];
        // SAFETY: `values` is a unique initialized two-element array.
        unsafe { vst1q_f64(values.as_mut_ptr(), *sum) };
        for (lane, value) in values.iter_mut().enumerate() {
            let (tail_w, tail_len) = decode_f16_tail(r_rems[pair * 2 + lane])?;
            for (&weight, &activation) in tail_w[..tail_len].iter().zip(tail) {
                *value += f64::from(weight * activation);
            }
            let val = *value as f32;
            if !val.is_finite() {
                return None;
            }
            output[pair * 2 + lane] = val;
        }
    }
    Some(output)
}

#[inline]
pub(super) fn dot_row_pinned(row: &[u8], input: &[f32]) -> Option<f32> {
    // SAFETY: the safe parent checks nonempty 16-aligned matching extents.
    unsafe { dot_row_pinned_neon(row, input) }
}

#[target_feature(enable = "neon")]
fn dot_row_pinned_neon(row: &[u8], input: &[f32]) -> Option<f32> {
    let mut s = [vdupq_n_f32(0.0); 4];
    let mut weight_magnitude = vdupq_n_u16(0);
    let magnitude_mask = vdupq_n_u16(0x7fff);
    for (w_blk, act) in row
        .as_chunks::<32>()
        .0
        .iter()
        .zip(input.as_chunks::<16>().0)
    {
        let w_halves = w_blk.as_chunks::<16>().0;
        let (w0, w1) = widen_f16_x8(
            load_u16x8_bytes(&w_halves[0]),
            magnitude_mask,
            &mut weight_magnitude,
        );
        let (w2, w3) = widen_f16_x8(
            load_u16x8_bytes(&w_halves[1]),
            magnitude_mask,
            &mut weight_magnitude,
        );
        s[0] = vfmaq_f32(s[0], w0, load_f32x4(&act[0..4]));
        s[1] = vfmaq_f32(s[1], w1, load_f32x4(&act[4..8]));
        s[2] = vfmaq_f32(s[2], w2, load_f32x4(&act[8..12]));
        s[3] = vfmaq_f32(s[3], w3, load_f32x4(&act[12..16]));
    }
    if vmaxvq_u16(weight_magnitude) >= 0x7c00 {
        return None;
    }
    reduce_pinned_lanes(s)
}

#[inline]
pub(super) fn dot_rows_pinned<const ROWS: usize>(
    rows: [&[u8]; ROWS],
    input: &[f32],
) -> Option<[f32; ROWS]> {
    // SAFETY: the safe parent checks four/eight rows and 16-aligned matching extents.
    unsafe { dot_rows_pinned_neon::<ROWS>(rows, input) }
}

#[target_feature(enable = "neon")]
fn dot_rows_pinned_neon<const ROWS: usize>(
    rows: [&[u8]; ROWS],
    input: &[f32],
) -> Option<[f32; ROWS]> {
    let mut acc = [[vdupq_n_f32(0.0); 4]; ROWS];
    let mut weight_magnitude = vdupq_n_u16(0);
    let magnitude_mask = vdupq_n_u16(0x7fff);
    let r_chunks = rows.map(|row| row.as_chunks::<32>().0);
    for (idx, act) in input.as_chunks::<16>().0.iter().enumerate() {
        let a0 = load_f32x4(&act[0..4]);
        let a1 = load_f32x4(&act[4..8]);
        let a2 = load_f32x4(&act[8..12]);
        let a3 = load_f32x4(&act[12..16]);
        for (r_chunk, r_acc) in r_chunks.iter().zip(acc.iter_mut()) {
            let w_halves = r_chunk[idx].as_chunks::<16>().0;
            let (w0, w1) = widen_f16_x8(
                load_u16x8_bytes(&w_halves[0]),
                magnitude_mask,
                &mut weight_magnitude,
            );
            let (w2, w3) = widen_f16_x8(
                load_u16x8_bytes(&w_halves[1]),
                magnitude_mask,
                &mut weight_magnitude,
            );
            r_acc[0] = vfmaq_f32(r_acc[0], w0, a0);
            r_acc[1] = vfmaq_f32(r_acc[1], w1, a1);
            r_acc[2] = vfmaq_f32(r_acc[2], w2, a2);
            r_acc[3] = vfmaq_f32(r_acc[3], w3, a3);
        }
    }
    if vmaxvq_u16(weight_magnitude) >= 0x7c00 {
        return None;
    }
    let mut out = [0.0_f32; ROWS];
    for (dst, s) in out.iter_mut().zip(acc) {
        *dst = reduce_pinned_lanes(s)?;
    }
    Some(out)
}

#[inline]
pub(super) fn dot_tile_row<const TOKENS: usize>(
    row: &[u8],
    inputs: &[&[f32]],
    out: &mut [f32; TOKENS],
) -> Option<()> {
    // SAFETY: the safe parent validated token bounds and matching row/input extents.
    unsafe { dot_tile_row_neon::<TOKENS>(row, inputs, out) }
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Retain float_dot's left-to-right binary64 sum of binary32 products and final binary32 rounding."
)]
#[target_feature(enable = "neon")]
fn dot_tile_row_neon<const TOKENS: usize>(
    row: &[u8],
    inputs: &[&[f32]],
    out: &mut [f32; TOKENS],
) -> Option<()> {
    const K_TILE: usize = 16;
    let (w_chunks, w_rem) = row.as_chunks::<16>();
    let chunk_count = w_chunks.len();
    let mut weight_magnitude = vdupq_n_u16(0);
    let magnitude_mask = vdupq_n_u16(0x7fff);
    let (tok_pairs, tok_rem) = inputs.as_chunks::<2>();
    let mut pair_sums = [vdupq_n_f64(-0.0); TOKENS];
    let mut odd_sum = -0.0_f64;

    for k_base in (0..chunk_count).step_by(K_TILE) {
        let tile_len = (chunk_count - k_base).min(K_TILE);
        let mut w_tile = [[vdupq_n_f32(0.0); 2]; K_TILE];
        for (k_off, w_slot) in w_tile.iter_mut().take(tile_len).enumerate() {
            *w_slot = widen_f16_x8(
                load_u16x8_bytes(&w_chunks[k_base + k_off]),
                magnitude_mask,
                &mut weight_magnitude,
            )
            .into();
        }
        for (pair_idx, [in0, in1]) in tok_pairs.iter().enumerate() {
            let mut sum = pair_sums[pair_idx];
            for (k_off, &[w0, w1]) in w_tile.iter().take(tile_len).enumerate() {
                let col = (k_base + k_off) * 8;
                let a0_lo = load_f32x4(&in0[col..col + 4]);
                let a1_lo = load_f32x4(&in1[col..col + 4]);
                sum = add_columns(sum, vmulq_f32(w0, a0_lo), vmulq_f32(w0, a1_lo));
                let a0_hi = load_f32x4(&in0[col + 4..col + 8]);
                let a1_hi = load_f32x4(&in1[col + 4..col + 8]);
                sum = add_columns(sum, vmulq_f32(w1, a0_hi), vmulq_f32(w1, a1_hi));
            }
            pair_sums[pair_idx] = sum;
        }
        if let [odd_in] = tok_rem {
            for (k_off, &[w0, w1]) in w_tile.iter().take(tile_len).enumerate() {
                let col = (k_base + k_off) * 8;
                let p0 = vmulq_f32(w0, load_f32x4(&odd_in[col..col + 4]));
                let p1 = vmulq_f32(w1, load_f32x4(&odd_in[col + 4..col + 8]));
                odd_sum = add_f32x4_to_f64_ordered(odd_sum, p0);
                odd_sum = add_f32x4_to_f64_ordered(odd_sum, p1);
            }
        }
    }
    if vmaxvq_u16(weight_magnitude) >= 0x7c00 {
        return None;
    }

    let (tail_w, tail_len) = decode_f16_tail(w_rem)?;
    let rem_base = chunk_count * 8;
    let mut tmp = [0.0_f32; TOKENS];
    for (pair_idx, [in0, in1]) in tok_pairs.iter().enumerate() {
        let mut vals = [0.0_f64; 2];
        // SAFETY: `vals` is a unique initialized two-element array.
        unsafe { vst1q_f64(vals.as_mut_ptr(), pair_sums[pair_idx]) };
        for (idx, &w) in tail_w[..tail_len].iter().enumerate() {
            let col = rem_base + idx;
            vals[0] += f64::from(w * in0[col]);
            vals[1] += f64::from(w * in1[col]);
        }
        let v0 = vals[0] as f32;
        let v1 = vals[1] as f32;
        if !v0.is_finite() || !v1.is_finite() {
            return None;
        }
        tmp[pair_idx * 2] = v0;
        tmp[pair_idx * 2 + 1] = v1;
    }
    if let [odd_in] = tok_rem {
        for (idx, &w) in tail_w[..tail_len].iter().enumerate() {
            odd_sum += f64::from(w * odd_in[rem_base + idx]);
        }
        let v_odd = odd_sum as f32;
        if !v_odd.is_finite() {
            return None;
        }
        tmp[tok_pairs.len() * 2] = v_odd;
    }
    *out = tmp;
    Some(())
}

#[inline]
pub(super) fn dot_tile_rows<const PAIRS: usize, const ROWS: usize, const TOKENS: usize>(
    rows: [&[u8]; ROWS],
    inputs: &[&[f32]],
    out: &mut [[f32; TOKENS]; ROWS],
) -> Option<()> {
    // SAFETY: the safe parent validated row/token counts and matching extents.
    unsafe { dot_tile_rows_neon::<PAIRS, ROWS, TOKENS>(rows, inputs, out) }
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    reason = "Widen F16 row pairs once per K-tile across up to 64 tokens while retaining exact binary64 column order."
)]
#[target_feature(enable = "neon")]
fn dot_tile_rows_neon<const PAIRS: usize, const ROWS: usize, const TOKENS: usize>(
    rows: [&[u8]; ROWS],
    inputs: &[&[f32]],
    out: &mut [[f32; TOKENS]; ROWS],
) -> Option<()> {
    const K_TILE: usize = 16;
    let r_chunks = rows.map(|row| row.as_chunks::<16>().0);
    let r_rems = rows.map(|row| row.as_chunks::<16>().1);
    let chunk_count = r_chunks[0].len();
    let mut weight_magnitude = vdupq_n_u16(0);
    let magnitude_mask = vdupq_n_u16(0x7fff);
    let mut sums = [[vdupq_n_f64(-0.0); TOKENS]; PAIRS];
    let (tok_quads, tok_rem) = inputs.as_chunks::<4>();

    for k_base in (0..chunk_count).step_by(K_TILE) {
        let tile_len = (chunk_count - k_base).min(K_TILE);
        for (p, pair_sums) in sums.iter_mut().enumerate() {
            let r_even = &r_chunks[p * 2];
            let r_odd = &r_chunks[p * 2 + 1];
            let mut even_tile = [[vdupq_n_f32(0.0); 2]; K_TILE];
            let mut odd_tile = [[vdupq_n_f32(0.0); 2]; K_TILE];
            for k_off in 0..tile_len {
                let idx = k_base + k_off;
                even_tile[k_off] = widen_f16_x8(
                    load_u16x8_bytes(&r_even[idx]),
                    magnitude_mask,
                    &mut weight_magnitude,
                )
                .into();
                odd_tile[k_off] = widen_f16_x8(
                    load_u16x8_bytes(&r_odd[idx]),
                    magnitude_mask,
                    &mut weight_magnitude,
                )
                .into();
            }

            for (quad_idx, quad) in tok_quads.iter().enumerate() {
                let base_t = quad_idx * 4;
                let mut acc = [
                    pair_sums[base_t],
                    pair_sums[base_t + 1],
                    pair_sums[base_t + 2],
                    pair_sums[base_t + 3],
                ];
                for k_off in 0..tile_len {
                    let col = (k_base + k_off) * 8;
                    let [we0, we1] = even_tile[k_off];
                    let [wo0, wo1] = odd_tile[k_off];
                    for (t, acc_t) in acc.iter_mut().enumerate() {
                        let a_lo = load_f32x4(&quad[t][col..col + 4]);
                        let a_hi = load_f32x4(&quad[t][col + 4..col + 8]);
                        let s = add_columns(*acc_t, vmulq_f32(we0, a_lo), vmulq_f32(wo0, a_lo));
                        *acc_t = add_columns(s, vmulq_f32(we1, a_hi), vmulq_f32(wo1, a_hi));
                    }
                }
                pair_sums[base_t..base_t + 4].copy_from_slice(&acc);
            }

            if !tok_rem.is_empty() {
                let rem_base = tok_quads.len() * 4;
                let mut acc = [vdupq_n_f64(-0.0); 3];
                acc[..tok_rem.len()]
                    .copy_from_slice(&pair_sums[rem_base..rem_base + tok_rem.len()]);
                for k_off in 0..tile_len {
                    let col = (k_base + k_off) * 8;
                    let [we0, we1] = even_tile[k_off];
                    let [wo0, wo1] = odd_tile[k_off];
                    for (t_off, input) in tok_rem.iter().enumerate() {
                        let a_lo = load_f32x4(&input[col..col + 4]);
                        let a_hi = load_f32x4(&input[col + 4..col + 8]);
                        let s = add_columns(acc[t_off], vmulq_f32(we0, a_lo), vmulq_f32(wo0, a_lo));
                        acc[t_off] = add_columns(s, vmulq_f32(we1, a_hi), vmulq_f32(wo1, a_hi));
                    }
                }
                pair_sums[rem_base..rem_base + tok_rem.len()]
                    .copy_from_slice(&acc[..tok_rem.len()]);
            }
        }
    }
    if vmaxvq_u16(weight_magnitude) >= 0x7c00 {
        return None;
    }

    let rem_base = chunk_count * 8;
    let mut tmp = [[0.0_f32; TOKENS]; ROWS];
    for (p, pair_sums) in sums.iter().enumerate() {
        let r0 = p * 2;
        let r1 = r0 + 1;
        let (tail_w0, tail_len) = decode_f16_tail(r_rems[r0])?;
        let (tail_w1, tail_len1) = decode_f16_tail(r_rems[r1])?;
        if tail_len != tail_len1 {
            return None;
        }
        for (tok, (&input, &pair_sum)) in inputs.iter().zip(pair_sums).enumerate() {
            let mut vals = [0.0_f64; 2];
            // SAFETY: `vals` is a unique initialized two-element array.
            unsafe { vst1q_f64(vals.as_mut_ptr(), pair_sum) };
            for (idx, (&w0, &w1)) in tail_w0[..tail_len]
                .iter()
                .zip(&tail_w1[..tail_len])
                .enumerate()
            {
                let act = input[rem_base + idx];
                vals[0] += f64::from(w0 * act);
                vals[1] += f64::from(w1 * act);
            }
            let v0 = vals[0] as f32;
            let v1 = vals[1] as f32;
            if !v0.is_finite() || !v1.is_finite() {
                return None;
            }
            tmp[r0][tok] = v0;
            tmp[r1][tok] = v1;
        }
    }
    *out = tmp;
    Some(())
}

#[inline]
pub(super) fn dot_tile_row_pinned<const TOKENS: usize>(
    row: &[u8],
    inputs: &[&[f32]],
    out: &mut [f32; TOKENS],
) -> Option<()> {
    // SAFETY: the safe parent validated token bounds and 16-aligned matching extents.
    unsafe { dot_tile_row_pinned_neon::<TOKENS>(row, inputs, out) }
}

#[target_feature(enable = "neon")]
fn dot_tile_row_pinned_neon<const TOKENS: usize>(
    row: &[u8],
    inputs: &[&[f32]],
    out: &mut [f32; TOKENS],
) -> Option<()> {
    const K_TILE: usize = 16;
    let chunks16 = row.as_chunks::<32>().0;
    let block_count = chunks16.len();
    let mut weight_magnitude = vdupq_n_u16(0);
    let magnitude_mask = vdupq_n_u16(0x7fff);
    let mut lanes = [[vdupq_n_f32(0.0); 4]; TOKENS];
    let (tok_quads, tok_rem) = inputs.as_chunks::<4>();

    for k_base in (0..block_count).step_by(K_TILE) {
        let tile_len = (block_count - k_base).min(K_TILE);
        let mut w_tile = [[vdupq_n_f32(0.0); 4]; K_TILE];
        for (k_off, w_slot) in w_tile.iter_mut().take(tile_len).enumerate() {
            let halves = chunks16[k_base + k_off].as_chunks::<16>().0;
            let (w0, w1) = widen_f16_x8(
                load_u16x8_bytes(&halves[0]),
                magnitude_mask,
                &mut weight_magnitude,
            );
            let (w2, w3) = widen_f16_x8(
                load_u16x8_bytes(&halves[1]),
                magnitude_mask,
                &mut weight_magnitude,
            );
            *w_slot = [w0, w1, w2, w3];
        }
        for (quad_idx, quad) in tok_quads.iter().enumerate() {
            let base_t = quad_idx * 4;
            let mut acc = [
                lanes[base_t],
                lanes[base_t + 1],
                lanes[base_t + 2],
                lanes[base_t + 3],
            ];
            for (k_off, &[w0, w1, w2, w3]) in w_tile.iter().take(tile_len).enumerate() {
                let col = (k_base + k_off) * 16;
                for (input, acc_tok) in quad.iter().zip(acc.iter_mut()) {
                    let act = &input[col..col + 16];
                    acc_tok[0] = vfmaq_f32(acc_tok[0], w0, load_f32x4(&act[0..4]));
                    acc_tok[1] = vfmaq_f32(acc_tok[1], w1, load_f32x4(&act[4..8]));
                    acc_tok[2] = vfmaq_f32(acc_tok[2], w2, load_f32x4(&act[8..12]));
                    acc_tok[3] = vfmaq_f32(acc_tok[3], w3, load_f32x4(&act[12..16]));
                }
            }
            lanes[base_t..base_t + 4].copy_from_slice(&acc);
        }
        if !tok_rem.is_empty() {
            let rem_base = tok_quads.len() * 4;
            for (t_off, input) in tok_rem.iter().enumerate() {
                let mut acc = lanes[rem_base + t_off];
                for (k_off, &[w0, w1, w2, w3]) in w_tile.iter().take(tile_len).enumerate() {
                    let col = (k_base + k_off) * 16;
                    let act = &input[col..col + 16];
                    acc[0] = vfmaq_f32(acc[0], w0, load_f32x4(&act[0..4]));
                    acc[1] = vfmaq_f32(acc[1], w1, load_f32x4(&act[4..8]));
                    acc[2] = vfmaq_f32(acc[2], w2, load_f32x4(&act[8..12]));
                    acc[3] = vfmaq_f32(acc[3], w3, load_f32x4(&act[12..16]));
                }
                lanes[rem_base + t_off] = acc;
            }
        }
    }
    if vmaxvq_u16(weight_magnitude) >= 0x7c00 {
        return None;
    }

    let mut tmp = [0.0_f32; TOKENS];
    for (dst, &s) in tmp.iter_mut().zip(lanes.iter()).take(inputs.len()) {
        *dst = reduce_pinned_lanes(s)?;
    }
    *out = tmp;
    Some(())
}

#[inline]
pub(super) fn dot_tile_rows_pinned<const PAIRS: usize, const ROWS: usize, const TOKENS: usize>(
    rows: [&[u8]; ROWS],
    inputs: &[&[f32]],
    out: &mut [[f32; TOKENS]; ROWS],
) -> Option<()> {
    // SAFETY: the safe parent validated row/token counts and 16-aligned matching extents.
    unsafe { dot_tile_rows_pinned_neon::<PAIRS, ROWS, TOKENS>(rows, inputs, out) }
}

#[expect(
    clippy::too_many_lines,
    reason = "Keep 2x2 F16 pinned FMA register-blocked tile and weight widening hoisted across tokens."
)]
#[target_feature(enable = "neon")]
fn dot_tile_rows_pinned_neon<const PAIRS: usize, const ROWS: usize, const TOKENS: usize>(
    rows: [&[u8]; ROWS],
    inputs: &[&[f32]],
    out: &mut [[f32; TOKENS]; ROWS],
) -> Option<()> {
    const K_TILE: usize = 8;
    let r_chunks = rows.map(|row| row.as_chunks::<32>().0);
    let block_count = r_chunks[0].len();
    let mut weight_magnitude = vdupq_n_u16(0);
    let magnitude_mask = vdupq_n_u16(0x7fff);
    let mut lanes = [[[vdupq_n_f32(0.0); 4]; TOKENS]; ROWS];
    let (tok_pairs, tok_rem) = inputs.as_chunks::<2>();

    for k_base in (0..block_count).step_by(K_TILE) {
        let tile_len = (block_count - k_base).min(K_TILE);
        for row_pair in 0..PAIRS {
            let r0 = row_pair * 2;
            let r1 = r0 + 1;
            let mut w0_tile = [[vdupq_n_f32(0.0); 4]; K_TILE];
            let mut w1_tile = [[vdupq_n_f32(0.0); 4]; K_TILE];
            for k_off in 0..tile_len {
                let idx = k_base + k_off;
                let h0 = r_chunks[r0][idx].as_chunks::<16>().0;
                let (w00, w01) = widen_f16_x8(
                    load_u16x8_bytes(&h0[0]),
                    magnitude_mask,
                    &mut weight_magnitude,
                );
                let (w02, w03) = widen_f16_x8(
                    load_u16x8_bytes(&h0[1]),
                    magnitude_mask,
                    &mut weight_magnitude,
                );
                w0_tile[k_off] = [w00, w01, w02, w03];

                let h1 = r_chunks[r1][idx].as_chunks::<16>().0;
                let (w10, w11) = widen_f16_x8(
                    load_u16x8_bytes(&h1[0]),
                    magnitude_mask,
                    &mut weight_magnitude,
                );
                let (w12, w13) = widen_f16_x8(
                    load_u16x8_bytes(&h1[1]),
                    magnitude_mask,
                    &mut weight_magnitude,
                );
                w1_tile[k_off] = [w10, w11, w12, w13];
            }

            for (pair_idx, [in0, in1]) in tok_pairs.iter().enumerate() {
                let t0 = pair_idx * 2;
                let t1 = t0 + 1;
                let mut acc00 = lanes[r0][t0];
                let mut acc01 = lanes[r0][t1];
                let mut acc10 = lanes[r1][t0];
                let mut acc11 = lanes[r1][t1];
                for k_off in 0..tile_len {
                    let col = (k_base + k_off) * 16;
                    let act0 = &in0[col..col + 16];
                    let act1 = &in1[col..col + 16];
                    let a00 = load_f32x4(&act0[0..4]);
                    let a01 = load_f32x4(&act0[4..8]);
                    let a02 = load_f32x4(&act0[8..12]);
                    let a03 = load_f32x4(&act0[12..16]);
                    let a10 = load_f32x4(&act1[0..4]);
                    let a11 = load_f32x4(&act1[4..8]);
                    let a12 = load_f32x4(&act1[8..12]);
                    let a13 = load_f32x4(&act1[12..16]);

                    let [w00, w01, w02, w03] = w0_tile[k_off];
                    acc00[0] = vfmaq_f32(acc00[0], w00, a00);
                    acc01[0] = vfmaq_f32(acc01[0], w00, a10);
                    acc00[1] = vfmaq_f32(acc00[1], w01, a01);
                    acc01[1] = vfmaq_f32(acc01[1], w01, a11);
                    acc00[2] = vfmaq_f32(acc00[2], w02, a02);
                    acc01[2] = vfmaq_f32(acc01[2], w02, a12);
                    acc00[3] = vfmaq_f32(acc00[3], w03, a03);
                    acc01[3] = vfmaq_f32(acc01[3], w03, a13);

                    let [w10, w11, w12, w13] = w1_tile[k_off];
                    acc10[0] = vfmaq_f32(acc10[0], w10, a00);
                    acc11[0] = vfmaq_f32(acc11[0], w10, a10);
                    acc10[1] = vfmaq_f32(acc10[1], w11, a01);
                    acc11[1] = vfmaq_f32(acc11[1], w11, a11);
                    acc10[2] = vfmaq_f32(acc10[2], w12, a02);
                    acc11[2] = vfmaq_f32(acc11[2], w12, a12);
                    acc10[3] = vfmaq_f32(acc10[3], w13, a03);
                    acc11[3] = vfmaq_f32(acc11[3], w13, a13);
                }
                lanes[r0][t0] = acc00;
                lanes[r0][t1] = acc01;
                lanes[r1][t0] = acc10;
                lanes[r1][t1] = acc11;
            }
            if let [odd_in] = tok_rem {
                let t_last = tok_pairs.len() * 2;
                let mut acc0 = lanes[r0][t_last];
                let mut acc1 = lanes[r1][t_last];
                for k_off in 0..tile_len {
                    let col = (k_base + k_off) * 16;
                    let act = &odd_in[col..col + 16];
                    let a0 = load_f32x4(&act[0..4]);
                    let a1 = load_f32x4(&act[4..8]);
                    let a2 = load_f32x4(&act[8..12]);
                    let a3 = load_f32x4(&act[12..16]);

                    let [w00, w01, w02, w03] = w0_tile[k_off];
                    acc0[0] = vfmaq_f32(acc0[0], w00, a0);
                    acc0[1] = vfmaq_f32(acc0[1], w01, a1);
                    acc0[2] = vfmaq_f32(acc0[2], w02, a2);
                    acc0[3] = vfmaq_f32(acc0[3], w03, a3);

                    let [w10, w11, w12, w13] = w1_tile[k_off];
                    acc1[0] = vfmaq_f32(acc1[0], w10, a0);
                    acc1[1] = vfmaq_f32(acc1[1], w11, a1);
                    acc1[2] = vfmaq_f32(acc1[2], w12, a2);
                    acc1[3] = vfmaq_f32(acc1[3], w13, a3);
                }
                lanes[r0][t_last] = acc0;
                lanes[r1][t_last] = acc1;
            }
        }
    }
    if vmaxvq_u16(weight_magnitude) >= 0x7c00 {
        return None;
    }

    let mut tmp = [[0.0_f32; TOKENS]; ROWS];
    for (out_row, row_lanes) in tmp.iter_mut().zip(lanes) {
        for (dst, &s) in out_row.iter_mut().zip(row_lanes.iter()).take(inputs.len()) {
            *dst = reduce_pinned_lanes(s)?;
        }
    }
    *out = tmp;
    Some(())
}
