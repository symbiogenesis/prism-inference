//! Quantized row dot products, 8-row repacking, and activation quantizers.

#[cfg(target_arch = "aarch64")]
use super::aarch64;

/// Number of float activations per `Q8_0` block.
pub const Q8_0_BLOCK_ELEMENTS: usize = 32;
/// Encoded byte length of one GGML `Q8_0` weight block (`f16` scale + 32 `i8` quants).
pub const Q8_0_BLOCK_BYTES: usize = 34;
/// Number of elements per K-quant superblock (`Q4_K`, `Q6_K`, `Q8_K`).
pub const QK_K: usize = 256;
/// Encoded byte length of one GGML `Q4_K` weight superblock.
pub const Q4_K_BLOCK_BYTES: usize = 144;
/// Encoded byte length of one GGML `Q6_K` weight superblock.
pub const Q6_K_BLOCK_BYTES: usize = 210;
/// Encoded byte length of one 8-row repacked `Q4_K` strip superblock (`1184` bytes).
pub const Q4_K_REPACK8_BLOCK_BYTES: usize = 1184;

/// Cached ARM64 optional SIMD capabilities governing `SDOT` and `I8MM` dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arm64QuantCaps {
    /// Whether ARMv8.2-A `dotprod` (`SDOT`) is available.
    pub dotprod: bool,
    /// Whether ARMv8.6-A `i8mm` (`SMMLA`) is available.
    pub i8mm: bool,
}

impl Arm64QuantCaps {
    /// Detects CPU support for `dotprod` and `i8mm` once per process and caches the result.
    #[must_use]
    #[cfg_attr(
        not(target_arch = "aarch64"),
        expect(
            clippy::missing_const_for_fn,
            reason = "Runtime ARM64 feature detection is non-const on aarch64 targets."
        )
    )]
    pub fn detect() -> Self {
        Self {
            dotprod: crate::has_dotprod(),
            i8mm: crate::has_i8mm(),
        }
    }

    /// Returns a capability set with optional `dotprod` and `i8mm` disabled for fallback controls.
    #[must_use]
    pub const fn portable() -> Self {
        Self {
            dotprod: false,
            i8mm: false,
        }
    }
}

impl Default for Arm64QuantCaps {
    fn default() -> Self {
        Self::detect()
    }
}

/// On-the-fly symmetric `Q8_0` quantized activation block (`32` values).
#[allow(
    non_camel_case_types,
    reason = "Match GGML Q8_0 activation block naming."
)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Q8_0ActBlock {
    /// Block scale factor (`max(|x|) / 127`).
    pub d: f32,
    /// Quantized signed activation lanes `[-127, 127]`.
    pub qs: [i8; 32],
}

/// On-the-fly symmetric `Q8_K` quantized activation superblock (`256` values + `16` sub-sums).
#[allow(
    non_camel_case_types,
    reason = "Match GGML Q8_K activation block naming."
)]
#[derive(Clone, Copy, Debug)]
pub struct Q8_KActBlock {
    /// Superblock scale factor (`max(|x|) / 127`).
    pub d: f32,
    /// Quantized signed activation lanes `[-127, 127]`.
    pub qs: [i8; 256],
    /// Exact integer sums for each 16-element sub-group `qs[g*16..(g+1)*16]`.
    pub bsums: [i16; 16],
}

impl Default for Q8_KActBlock {
    fn default() -> Self {
        Self {
            d: 0.0,
            qs: [0; 256],
            bsums: [0; 16],
        }
    }
}

#[inline]
#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Quantization scale multiplication converts bounded i32 dot sums back to f32."
)]
pub(super) const fn i32_to_f32(value: i32) -> f32 {
    value as f32
}

#[inline]
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Superblock f64 scale product narrows once per block to f32."
)]
pub(super) const fn f64_to_f32(value: f64) -> f32 {
    value as f32
}

#[inline]
pub(super) fn f16_bits_to_f32(bytes: [u8; 2]) -> f32 {
    super::half::scalar(u16::from_le_bytes(bytes))
}

/// Unpacks `(scale, min)` 6-bit integers for sub-block `group` (`0..8`) from a `Q4_K` scale table.
#[inline]
#[must_use]
pub const fn unpack_q4_k_scale_min(group: usize, scales: &[u8; 12]) -> (u8, u8) {
    if group < 4 {
        (scales[group] & 63, scales[group + 4] & 63)
    } else {
        (
            (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4),
            (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4),
        )
    }
}

#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
#[inline]
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Scalar fallback clamps scaled float input to [-127, 127] before rounding to i8."
)]
const fn quantize_i8_scalar(value: f32) -> i8 {
    value.round_ties_even().clamp(-127.0, 127.0) as i8
}

/// Quantizes `input` (`out.len() * 32` floats) into symmetric `Q8_0ActBlock`s once per matmul.
pub fn quantize_row_q8_0(input: &[f32], out: &mut [Q8_0ActBlock]) {
    if input.len() != out.len().saturating_mul(Q8_0_BLOCK_ELEMENTS) {
        return;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        aarch64::quantize_row_q8_0(input, out);
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        for (chunk, block) in input.as_chunks::<32>().0.iter().zip(out.iter_mut()) {
            let amax = chunk
                .iter()
                .copied()
                .fold(0.0_f32, |acc, val| acc.max(val.abs()));
            if amax == 0.0 || !amax.is_finite() {
                block.d = 0.0;
                block.qs = [0; 32];
                continue;
            }
            let d = amax / 127.0_f32;
            let inv_d = 127.0_f32 / amax;
            for (src, dst) in chunk.iter().zip(block.qs.iter_mut()) {
                *dst = quantize_i8_scalar(src * inv_d);
            }
            block.d = d;
        }
    }
}

/// Quantizes `input` (`out.len() * 256` floats) into `Q8_KActBlock`s with exact 16-lane `bsums`.
pub fn quantize_row_q8_k(input: &[f32], out: &mut [Q8_KActBlock]) {
    if input.len() != out.len().saturating_mul(QK_K) {
        return;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        aarch64::quantize_row_q8_k(input, out);
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        for (chunk, block) in input.as_chunks::<256>().0.iter().zip(out.iter_mut()) {
            let amax = chunk
                .iter()
                .copied()
                .fold(0.0_f32, |acc, val| acc.max(val.abs()));
            if amax == 0.0 || !amax.is_finite() {
                block.d = 0.0;
                block.qs = [0; 256];
                block.bsums = [0; 16];
                continue;
            }
            let d = amax / 127.0_f32;
            let inv_d = 127.0_f32 / amax;
            for ((src16, dst16), bsum) in chunk
                .as_chunks::<16>()
                .0
                .iter()
                .zip(block.qs.as_chunks_mut::<16>().0.iter_mut())
                .zip(block.bsums.iter_mut())
            {
                let mut sum = 0_i16;
                for (&x_val, q_slot) in src16.iter().zip(dst16.iter_mut()) {
                    let quantized = quantize_i8_scalar(x_val * inv_d);
                    *q_slot = quantized;
                    sum += i16::from(quantized);
                }
                *bsum = sum;
            }
            block.d = d;
        }
    }
}

/// Computes exact 8-lane `(index % 8)` integer products and minimum sum for one `Q4_K` superblock.
#[must_use]
pub fn dot_kquant_8lanes_q4_k(
    block: &[u8; Q4_K_BLOCK_BYTES],
    values: &[i8; QK_K],
    sums: &[i16; 16],
) -> ([i32; 8], i32) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        aarch64::dot_kquant_8lanes_q4_k(block, values, sums)
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        portable_dot_kquant_8lanes_q4_k(block, values, sums)
    }
}

/// Computes exact 8-lane `(index % 8)` integer products for one `Q6_K` superblock.
#[must_use]
pub fn dot_kquant_8lanes_q6_k(block: &[u8; Q6_K_BLOCK_BYTES], values: &[i8; QK_K]) -> [i32; 8] {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        aarch64::dot_kquant_8lanes_q6_k(block, values)
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        portable_dot_kquant_8lanes_q6_k(block, values)
    }
}

#[cfg(any(test, not(all(target_arch = "aarch64", target_feature = "neon"))))]
fn portable_dot_kquant_8lanes_q4_k(
    block: &[u8; Q4_K_BLOCK_BYTES],
    values: &[i8; QK_K],
    sums: &[i16; 16],
) -> ([i32; 8], i32) {
    let mut integers = [0_i32; 8];
    let mut min_sum = 0_i32;
    let Ok(scales) = <&[u8; 12]>::try_from(&block[4..16]) else {
        return (integers, min_sum);
    };
    for group in 0..8 {
        let (scale_val, min_val) = unpack_q4_k_scale_min(group, scales);
        let pair_sum = i32::from(sums[2 * group]) + i32::from(sums[2 * group + 1]);
        min_sum += i32::from(min_val) * pair_sum;
        let qs_base = 16 + (group / 2) * 32;
        let a_base = group * 32;
        let sc = i32::from(scale_val);
        for elem in 0..32 {
            let byte = block[qs_base + elem];
            let q_val = if group.is_multiple_of(2) {
                byte & 0x0f
            } else {
                byte >> 4
            };
            integers[elem % 8] += sc * i32::from(q_val) * i32::from(values[a_base + elem]);
        }
    }
    (integers, min_sum)
}

#[cfg(any(test, not(all(target_arch = "aarch64", target_feature = "neon"))))]
fn portable_dot_kquant_8lanes_q6_k(
    block: &[u8; Q6_K_BLOCK_BYTES],
    values: &[i8; QK_K],
) -> [i32; 8] {
    let mut integers = [0_i32; 8];
    for half_idx in 0..2 {
        for quarter in 0..4 {
            let scale0 = i32::from(block[192 + half_idx * 8 + quarter * 2].cast_signed());
            let scale1 = i32::from(block[192 + half_idx * 8 + quarter * 2 + 1].cast_signed());
            let low_off = half_idx * 64 + (quarter % 2) * 32;
            let high_off = 128 + half_idx * 32;
            let a_off = half_idx * 128 + quarter * 32;
            for elem in 0..16 {
                let ql = block[low_off + elem];
                let low = if quarter < 2 { ql & 0x0f } else { ql >> 4 };
                let high = (block[high_off + elem] >> (quarter * 2)) & 0x03;
                let weight = i32::from(low | (high << 4)) - 32;
                integers[elem % 8] += scale0 * weight * i32::from(values[a_off + elem]);
            }
            for elem in 0..16 {
                let ql = block[low_off + 16 + elem];
                let low = if quarter < 2 { ql & 0x0f } else { ql >> 4 };
                let high = (block[high_off + 16 + elem] >> (quarter * 2)) & 0x03;
                let weight = i32::from(low | (high << 4)) - 32;
                integers[elem % 8] += scale1 * weight * i32::from(values[a_off + 16 + elem]);
            }
        }
    }
    integers
}

fn portable_dot_q8_0_row(w_row: &[u8], act: &[Q8_0ActBlock]) -> f32 {
    let mut sum = 0.0_f32;
    for (w_block, a_block) in w_row.as_chunks::<34>().0.iter().zip(act.iter()) {
        let w_scale = f16_bits_to_f32([w_block[0], w_block[1]]);
        let mut int_sum = 0_i32;
        for (&w_q, &a_q) in w_block[2..].iter().zip(a_block.qs.iter()) {
            int_sum += i32::from(w_q.cast_signed()) * i32::from(a_q);
        }
        #[expect(
            clippy::suboptimal_flops,
            reason = "Matches non-FMA Q8_0 accumulation semantics."
        )]
        {
            sum += w_scale * (a_block.d * i32_to_f32(int_sum));
        }
    }
    sum
}

/// Computes `dot(w_row, act)` for one packed GGML `Q8_0` weight row against `Q8_0ActBlock`s.
#[must_use]
pub fn dot_q8_0_row(caps: Arm64QuantCaps, w_row: &[u8], act: &[Q8_0ActBlock]) -> f32 {
    if w_row.len() != act.len().saturating_mul(Q8_0_BLOCK_BYTES) {
        return 0.0;
    }
    #[cfg(target_arch = "aarch64")]
    if caps.dotprod
        && let Some(sum) = aarch64::quant_dot_q8_0_row(w_row, act)
    {
        return sum;
    }
    #[cfg(not(target_arch = "aarch64"))]
    let _ = caps;
    portable_dot_q8_0_row(w_row, act)
}

/// Computes four `Q8_0` row dot products while loading activations once per block.
#[must_use]
pub fn dot_q8_0_4rows(caps: Arm64QuantCaps, rows: [&[u8]; 4], act: &[Q8_0ActBlock]) -> [f32; 4] {
    let expected = act.len().saturating_mul(Q8_0_BLOCK_BYTES);
    if rows.iter().any(|row| row.len() != expected) {
        return [0.0; 4];
    }
    #[cfg(target_arch = "aarch64")]
    if caps.dotprod
        && let Some(sums) = aarch64::quant_dot_q8_0_4rows(rows, act)
    {
        return sums;
    }
    #[cfg(not(target_arch = "aarch64"))]
    let _ = caps;
    rows.map(|row| portable_dot_q8_0_row(row, act))
}

/// Computes a 4-row by 2-token `Q8_0` tile using `I8MM` (`SMMLA`) or `SDOT` fallback.
#[must_use]
pub fn gemm_q8_0_4x2(
    caps: Arm64QuantCaps,
    rows: [&[u8]; 4],
    act0: &[Q8_0ActBlock],
    act1: &[Q8_0ActBlock],
) -> [[f32; 2]; 4] {
    let expected = act0.len().saturating_mul(Q8_0_BLOCK_BYTES);
    if act0.len() != act1.len() || rows.iter().any(|row| row.len() != expected) {
        return [[0.0; 2]; 4];
    }
    #[cfg(target_arch = "aarch64")]
    if caps.i8mm
        && let Some(tile) = aarch64::quant_gemm_q8_0_4x2(rows, act0, act1)
    {
        return tile;
    }
    let col0 = dot_q8_0_4rows(caps, rows, act0);
    let col1 = dot_q8_0_4rows(caps, rows, act1);
    std::array::from_fn(|row_idx| [col0[row_idx], col1[row_idx]])
}

fn portable_dot_q4_k_row(w_row: &[u8], act: &[Q8_KActBlock]) -> f32 {
    let mut acc = 0.0_f64;
    for (block, a_block) in w_row.as_chunks::<144>().0.iter().zip(act.iter()) {
        let [d_w, dmin_w, _, _] = crate::widen_f16_four(&[
            u16::from_le_bytes([block[0], block[1]]),
            u16::from_le_bytes([block[2], block[3]]),
            0,
            0,
        ]);
        let d_w = f64::from(d_w);
        let dmin_w = f64::from(dmin_w);
        let d_act = f64::from(a_block.d);
        let Ok(scales) = <&[u8; 12]>::try_from(&block[4..16]) else {
            continue;
        };
        let mut sumi = 0_i32;
        let mut mint = 0_i32;
        for group in 0..8 {
            let (scale_val, min_val) = unpack_q4_k_scale_min(group, scales);
            let pair_sum =
                i32::from(a_block.bsums[2 * group]) + i32::from(a_block.bsums[2 * group + 1]);
            mint += i32::from(min_val) * pair_sum;
            let qs_base = 16 + (group / 2) * 32;
            let a_base = group * 32;
            let mut g_dot = 0_i32;
            for elem in 0..32 {
                let byte = block[qs_base + elem];
                let q_val = if group.is_multiple_of(2) {
                    byte & 0x0f
                } else {
                    byte >> 4
                };
                g_dot += i32::from(q_val) * i32::from(a_block.qs[a_base + elem]);
            }
            sumi += g_dot * i32::from(scale_val);
        }
        #[expect(
            clippy::suboptimal_flops,
            reason = "Matches non-FMA Q4_K block accumulation order."
        )]
        {
            acc += (d_w * d_act) * f64::from(sumi) - (dmin_w * d_act) * f64::from(mint);
        }
    }
    f64_to_f32(acc)
}

/// Computes `dot(w_row, act)` for one packed GGML `Q4_K` weight row against `Q8_KActBlock`s.
#[must_use]
pub fn dot_q4_k_row(caps: Arm64QuantCaps, w_row: &[u8], act: &[Q8_KActBlock]) -> f32 {
    if w_row.len() != act.len().saturating_mul(Q4_K_BLOCK_BYTES) {
        return 0.0;
    }
    #[cfg(target_arch = "aarch64")]
    if caps.dotprod
        && let Some(sum) = aarch64::quant_dot_q4_k_row(w_row, act)
    {
        return sum;
    }
    #[cfg(not(target_arch = "aarch64"))]
    let _ = caps;
    portable_dot_q4_k_row(w_row, act)
}

/// Computes four `Q4_K` row dot products while sharing activations per superblock.
#[must_use]
pub fn dot_q4_k_4rows(caps: Arm64QuantCaps, rows: [&[u8]; 4], act: &[Q8_KActBlock]) -> [f32; 4] {
    let expected = act.len().saturating_mul(Q4_K_BLOCK_BYTES);
    if rows.iter().any(|row| row.len() != expected) {
        return [0.0; 4];
    }
    #[cfg(target_arch = "aarch64")]
    if caps.dotprod
        && let Some(sums) = aarch64::quant_dot_q4_k_4rows(rows, act)
    {
        return sums;
    }
    #[cfg(not(target_arch = "aarch64"))]
    let _ = caps;
    rows.map(|row| portable_dot_q4_k_row(row, act))
}

/// Computes a 4-row by 2-token `Q4_K` tile using `I8MM` (`SMMLA`) or `SDOT` fallback.
#[must_use]
pub fn gemm_q4_k_4x2(
    caps: Arm64QuantCaps,
    rows: [&[u8]; 4],
    act0: &[Q8_KActBlock],
    act1: &[Q8_KActBlock],
) -> [[f32; 2]; 4] {
    let expected = act0.len().saturating_mul(Q4_K_BLOCK_BYTES);
    if act0.len() != act1.len() || rows.iter().any(|row| row.len() != expected) {
        return [[0.0; 2]; 4];
    }
    #[cfg(target_arch = "aarch64")]
    if caps.i8mm
        && let Some(tile) = aarch64::quant_gemm_q4_k_4x2(rows, act0, act1)
    {
        return tile;
    }
    let col0 = dot_q4_k_4rows(caps, rows, act0);
    let col1 = dot_q4_k_4rows(caps, rows, act1);
    std::array::from_fn(|row_idx| [col0[row_idx], col1[row_idx]])
}

/// Repacks row-major GGML `Q4_K` matrix bytes into 8-row sequential strips (`1184` B per block).
///
/// Returns `None` if dimensions are not divisible by `(256, 8)` or any block scale is nonfinite.
#[must_use]
pub fn repack_q4_k_8rows(src: &[u8], in_dim: usize, out_dim: usize) -> Option<Vec<u8>> {
    if in_dim == 0 || out_dim == 0 || !in_dim.is_multiple_of(QK_K) || !out_dim.is_multiple_of(8) {
        return None;
    }
    let blocks_per_row = in_dim / QK_K;
    let src_row_bytes = blocks_per_row.checked_mul(Q4_K_BLOCK_BYTES)?;
    if src.len() != out_dim.checked_mul(src_row_bytes)? {
        return None;
    }
    let strips = out_dim / 8;
    let out_bytes = strips
        .checked_mul(blocks_per_row)?
        .checked_mul(Q4_K_REPACK8_BLOCK_BYTES)?;
    let mut dst = Vec::new();
    dst.try_reserve_exact(out_bytes).ok()?;
    dst.resize(out_bytes, 0);

    for strip in 0..strips {
        let r_base = strip * 8;
        for blk in 0..blocks_per_row {
            let out_off = (strip * blocks_per_row + blk) * Q4_K_REPACK8_BLOCK_BYTES;
            let out_blk = &mut dst[out_off..out_off + Q4_K_REPACK8_BLOCK_BYTES];
            for row_offset in 0..8 {
                let src_off = (r_base + row_offset) * src_row_bytes + blk * Q4_K_BLOCK_BYTES;
                let sb = &src[src_off..src_off + Q4_K_BLOCK_BYTES];
                let [row_scale, row_min_scale, _, _] = crate::widen_f16_four(&[
                    u16::from_le_bytes([sb[0], sb[1]]),
                    u16::from_le_bytes([sb[2], sb[3]]),
                    0,
                    0,
                ]);
                if !row_scale.is_finite() || !row_min_scale.is_finite() {
                    return None;
                }
                let scales = <&[u8; 12]>::try_from(&sb[4..16]).ok()?;
                out_blk[row_offset * 2..row_offset * 2 + 2].copy_from_slice(&sb[0..2]);
                out_blk[16 + row_offset * 2..16 + row_offset * 2 + 2].copy_from_slice(&sb[2..4]);
                for group in 0..8 {
                    let (scale_val, min_val) = unpack_q4_k_scale_min(group, scales);
                    out_blk[32 + group * 8 + row_offset] = scale_val;
                    out_blk[96 + group * 8 + row_offset] = min_val;
                }
                for chunk in 0..4 {
                    let s_off = 16 + chunk * 32;
                    let d_off = 160 + chunk * 256 + row_offset * 32;
                    out_blk[d_off..d_off + 32].copy_from_slice(&sb[s_off..s_off + 32]);
                }
            }
        }
    }
    Some(dst)
}

fn portable_dot_q4_k_repacked8(strip: &[u8], act: &[Q8_KActBlock]) -> [f32; 8] {
    let mut acc = [0.0_f64; 8];
    for (block, a_block) in strip
        .as_chunks::<Q4_K_REPACK8_BLOCK_BYTES>()
        .0
        .iter()
        .zip(act.iter())
    {
        let d_act = f64::from(a_block.d);
        let scale_f16: [u16; 8] =
            std::array::from_fn(|row| u16::from_le_bytes([block[row * 2], block[row * 2 + 1]]));
        let min_f16: [u16; 8] = std::array::from_fn(|row| {
            u16::from_le_bytes([block[16 + row * 2], block[16 + row * 2 + 1]])
        });
        let scale_widened = crate::widen_f16_eight(&scale_f16);
        let min_widened = crate::widen_f16_eight(&min_f16);
        let mut mint = [0_i32; 8];
        let mut sumi = [0_i32; 8];
        for group in 0..8 {
            let pair_sum =
                i32::from(a_block.bsums[2 * group]) + i32::from(a_block.bsums[2 * group + 1]);
            for (row_idx, mint_slot) in mint.iter_mut().enumerate() {
                *mint_slot += i32::from(block[96 + group * 8 + row_idx]) * pair_sum;
            }
        }
        for chunk in 0..4 {
            let qs_base = 160 + chunk * 256;
            let a_lo = &a_block.qs[(chunk * 2) * 32..(chunk * 2 + 1) * 32];
            let a_hi = &a_block.qs[(chunk * 2 + 1) * 32..(chunk * 2 + 2) * 32];
            for (row_idx, sum_slot) in sumi.iter_mut().enumerate() {
                let row_qs = &block[qs_base + row_idx * 32..qs_base + (row_idx + 1) * 32];
                let mut dot_lo = 0_i32;
                let mut dot_hi = 0_i32;
                for elem in 0..32 {
                    dot_lo += i32::from(row_qs[elem] & 0x0f) * i32::from(a_lo[elem]);
                    dot_hi += i32::from(row_qs[elem] >> 4) * i32::from(a_hi[elem]);
                }
                *sum_slot += dot_lo * i32::from(block[32 + (chunk * 2) * 8 + row_idx])
                    + dot_hi * i32::from(block[32 + (chunk * 2 + 1) * 8 + row_idx]);
            }
        }
        for row_idx in 0..8 {
            let dw = f64::from(scale_widened[row_idx / 4][row_idx % 4]);
            let dm = f64::from(min_widened[row_idx / 4][row_idx % 4]);
            #[expect(
                clippy::suboptimal_flops,
                reason = "Matches non-FMA repacked Q4_K block accumulation order."
            )]
            {
                acc[row_idx] += (dw * d_act) * f64::from(sumi[row_idx])
                    - (dm * d_act) * f64::from(mint[row_idx]);
            }
        }
    }
    acc.map(f64_to_f32)
}

/// Computes an 8-row repacked `Q4_K` strip dot product.
#[must_use]
pub fn dot_q4_k_repacked8(caps: Arm64QuantCaps, strip: &[u8], act: &[Q8_KActBlock]) -> [f32; 8] {
    if strip.len() != act.len().saturating_mul(Q4_K_REPACK8_BLOCK_BYTES) {
        return [0.0; 8];
    }
    #[cfg(target_arch = "aarch64")]
    if caps.dotprod
        && let Some(sums) = aarch64::quant_dot_q4_k_repacked8(strip, act)
    {
        return sums;
    }
    #[cfg(not(target_arch = "aarch64"))]
    let _ = caps;
    portable_dot_q4_k_repacked8(strip, act)
}

/// Computes an 8-row by 2-token repacked `Q4_K` strip GEMM tile.
#[must_use]
pub fn gemm_q4_k_repacked8_8x2(
    caps: Arm64QuantCaps,
    strip: &[u8],
    act0: &[Q8_KActBlock],
    act1: &[Q8_KActBlock],
) -> [[f32; 2]; 8] {
    if act0.len() != act1.len()
        || strip.len() != act0.len().saturating_mul(Q4_K_REPACK8_BLOCK_BYTES)
    {
        return [[0.0; 2]; 8];
    }
    #[cfg(target_arch = "aarch64")]
    if caps.i8mm
        && let Some(tile) = aarch64::quant_gemm_q4_k_repacked8_8x2(strip, act0, act1)
    {
        return tile;
    }
    let col0 = dot_q4_k_repacked8(caps, strip, act0);
    let col1 = dot_q4_k_repacked8(caps, strip, act1);
    std::array::from_fn(|row_idx| [col0[row_idx], col1[row_idx]])
}

fn portable_dot_q6_k_row(w_row: &[u8], act: &[Q8_KActBlock]) -> f32 {
    let mut acc = 0.0_f64;
    for (block, a_block) in w_row.as_chunks::<210>().0.iter().zip(act.iter()) {
        let d_w = f64::from(f16_bits_to_f32([block[208], block[209]]));
        let d_act = f64::from(a_block.d);
        let mut sumi = 0_i32;
        for half_idx in 0..2 {
            for quarter in 0..4 {
                let s0 = i32::from(block[192 + half_idx * 8 + quarter * 2].cast_signed());
                let s1 = i32::from(block[192 + half_idx * 8 + quarter * 2 + 1].cast_signed());
                let mut d0 = 0_i32;
                let mut d1 = 0_i32;
                let low_off = half_idx * 64 + (quarter % 2) * 32;
                let high_off = 128 + half_idx * 32;
                let a_off = half_idx * 128 + quarter * 32;
                for elem in 0..16 {
                    let ql = block[low_off + elem];
                    let low = if quarter < 2 { ql & 0x0f } else { ql >> 4 };
                    let high = (block[high_off + elem] >> (quarter * 2)) & 0x03;
                    let q_val = i32::from(low | (high << 4)) - 32;
                    d0 += q_val * i32::from(a_block.qs[a_off + elem]);
                }
                for elem in 0..16 {
                    let ql = block[low_off + 16 + elem];
                    let low = if quarter < 2 { ql & 0x0f } else { ql >> 4 };
                    let high = (block[high_off + 16 + elem] >> (quarter * 2)) & 0x03;
                    let q_val = i32::from(low | (high << 4)) - 32;
                    d1 += q_val * i32::from(a_block.qs[a_off + 16 + elem]);
                }
                sumi += d0 * s0 + d1 * s1;
            }
        }
        #[expect(
            clippy::suboptimal_flops,
            reason = "Matches non-FMA Q6_K block accumulation order."
        )]
        {
            acc += (d_w * d_act) * f64::from(sumi);
        }
    }
    f64_to_f32(acc)
}

/// Computes `dot(w_row, act)` for one packed GGML `Q6_K` weight row against `Q8_KActBlock`s.
#[must_use]
pub fn dot_q6_k_row(caps: Arm64QuantCaps, w_row: &[u8], act: &[Q8_KActBlock]) -> f32 {
    if w_row.len() != act.len().saturating_mul(Q6_K_BLOCK_BYTES) {
        return 0.0;
    }
    #[cfg(target_arch = "aarch64")]
    if caps.dotprod
        && let Some(sum) = aarch64::quant_dot_q6_k_row(w_row, act)
    {
        return sum;
    }
    #[cfg(not(target_arch = "aarch64"))]
    let _ = caps;
    portable_dot_q6_k_row(w_row, act)
}

/// Computes four `Q6_K` row dot products while sharing activations per superblock.
#[must_use]
pub fn dot_q6_k_4rows(caps: Arm64QuantCaps, rows: [&[u8]; 4], act: &[Q8_KActBlock]) -> [f32; 4] {
    let expected = act.len().saturating_mul(Q6_K_BLOCK_BYTES);
    if rows.iter().any(|row| row.len() != expected) {
        return [0.0; 4];
    }
    #[cfg(target_arch = "aarch64")]
    if caps.dotprod
        && let Some(sums) = aarch64::quant_dot_q6_k_4rows(rows, act)
    {
        return sums;
    }
    #[cfg(not(target_arch = "aarch64"))]
    let _ = caps;
    rows.map(|row| portable_dot_q6_k_row(row, act))
}

/// Computes a 4-row by 2-token `Q6_K` tile using `I8MM` (`SMMLA`) or `SDOT` fallback.
#[must_use]
pub fn gemm_q6_k_4x2(
    caps: Arm64QuantCaps,
    rows: [&[u8]; 4],
    act0: &[Q8_KActBlock],
    act1: &[Q8_KActBlock],
) -> [[f32; 2]; 4] {
    let expected = act0.len().saturating_mul(Q6_K_BLOCK_BYTES);
    if act0.len() != act1.len() || rows.iter().any(|row| row.len() != expected) {
        return [[0.0; 2]; 4];
    }
    #[cfg(target_arch = "aarch64")]
    if caps.i8mm
        && let Some(tile) = aarch64::quant_gemm_q6_k_4x2(rows, act0, act1)
    {
        return tile;
    }
    let col0 = dot_q6_k_4rows(caps, rows, act0);
    let col1 = dot_q6_k_4rows(caps, rows, act1);
    std::array::from_fn(|row_idx| [col0[row_idx], col1[row_idx]])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::narrow_f32_four;

    fn make_q8_0_row(blocks: usize, seed: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(blocks * Q8_0_BLOCK_BYTES);
        for blk in 0..blocks {
            let scale_f = 0.015_f32 * f32::from(u8::try_from((seed + blk) % 13 + 1).unwrap_or(1));
            let bits = narrow_f32_four([scale_f, 0.0, 0.0, 0.0])[0];
            bytes.extend_from_slice(&bits.to_le_bytes());
            for elem in 0..32 {
                let q = i8::try_from(
                    isize::try_from((seed * 17 + blk * 7 + elem * 5) % 199).unwrap_or(0) - 99,
                )
                .unwrap_or(0);
                bytes.push(q.cast_unsigned());
            }
        }
        bytes
    }

    fn make_q4_k_row(blocks: usize, seed: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(blocks * Q4_K_BLOCK_BYTES);
        for blk in 0..blocks {
            let d = 0.012_f32 * f32::from(u8::try_from((seed + blk) % 9 + 1).unwrap_or(1));
            let dmin = 0.004_f32 * f32::from(u8::try_from((seed + blk * 3) % 7 + 1).unwrap_or(1));
            let narrowed = narrow_f32_four([d, dmin, 0.0, 0.0]);
            bytes.extend_from_slice(&narrowed[0].to_le_bytes());
            bytes.extend_from_slice(&narrowed[1].to_le_bytes());
            for idx in 0..12 {
                bytes.push(u8::try_from((seed * 13 + blk * 29 + idx * 41) % 251).unwrap_or(0));
            }
            for idx in 0..128 {
                bytes.push(u8::try_from((seed * 31 + blk * 19 + idx * 37) % 256).unwrap_or(0));
            }
        }
        bytes
    }

    fn make_q6_k_row(blocks: usize, seed: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(blocks * Q6_K_BLOCK_BYTES);
        for blk in 0..blocks {
            for idx in 0..128 {
                bytes.push(u8::try_from((seed * 23 + blk * 11 + idx * 43) % 256).unwrap_or(0));
            }
            for idx in 0..64 {
                bytes.push(u8::try_from((seed * 7 + blk * 17 + idx * 59) % 256).unwrap_or(0));
            }
            for idx in 0..16 {
                let s = i8::try_from(
                    isize::try_from((seed * 19 + blk * 13 + idx * 29) % 121).unwrap_or(0) - 60,
                )
                .unwrap_or(0);
                bytes.push(s.cast_unsigned());
            }
            let d = 0.008_f32 * f32::from(u8::try_from((seed + blk * 5) % 11 + 1).unwrap_or(1));
            let bits = narrow_f32_four([d, 0.0, 0.0, 0.0])[0];
            bytes.extend_from_slice(&bits.to_le_bytes());
        }
        bytes
    }

    fn sample_q8_k_inputs(in_dim: usize) -> (Vec<f32>, Vec<Q8_KActBlock>, Vec<Q8_KActBlock>) {
        let x0: Vec<f32> = (0..in_dim)
            .map(|idx| f32::from(i16::try_from((idx * 11) % 61).unwrap_or(0) - 30) * 0.05)
            .collect();
        let x1: Vec<f32> = (0..in_dim)
            .map(|idx| f32::from(i16::try_from((idx * 7 + 19) % 53).unwrap_or(0) - 26) * 0.06)
            .collect();
        let mut act0 = vec![Q8_KActBlock::default(); in_dim / 256];
        let mut act1 = vec![Q8_KActBlock::default(); in_dim / 256];
        quantize_row_q8_k(&x0, &mut act0);
        quantize_row_q8_k(&x1, &mut act1);
        (x0, act0, act1)
    }

    #[test]
    fn q8_0_and_q4_k_kernels_match_portable_reference() {
        let in_dim = 512_usize;
        let (x0, act0, act1) = sample_q8_k_inputs(in_dim);
        let mut act_q8_0 = vec![Q8_0ActBlock::default(); in_dim / 32];
        quantize_row_q8_0(&x0, &mut act_q8_0);
        let caps = Arm64QuantCaps::detect();
        let port = Arm64QuantCaps::portable();

        let q8_rows: Vec<Vec<u8>> = (0..4)
            .map(|row| make_q8_0_row(in_dim / 32, row + 1))
            .collect();
        let q8_refs = [
            &q8_rows[0][..],
            &q8_rows[1][..],
            &q8_rows[2][..],
            &q8_rows[3][..],
        ];
        assert_eq!(
            dot_q8_0_row(caps, q8_refs[0], &act_q8_0).to_bits(),
            dot_q8_0_row(port, q8_refs[0], &act_q8_0).to_bits()
        );
        assert_eq!(
            dot_q8_0_4rows(caps, q8_refs, &act_q8_0).map(f32::to_bits),
            dot_q8_0_4rows(port, q8_refs, &act_q8_0).map(f32::to_bits)
        );
        let g8_nat = gemm_q8_0_4x2(caps, q8_refs, &act_q8_0, &act_q8_0);
        let g8_prt = gemm_q8_0_4x2(port, q8_refs, &act_q8_0, &act_q8_0);
        assert_eq!(
            g8_nat.map(|pair| pair.map(f32::to_bits)),
            g8_prt.map(|pair| pair.map(f32::to_bits))
        );

        let q4_rows: Vec<Vec<u8>> = (0..8)
            .map(|row| make_q4_k_row(in_dim / 256, row + 1))
            .collect();
        let q4_refs = [
            &q4_rows[0][..],
            &q4_rows[1][..],
            &q4_rows[2][..],
            &q4_rows[3][..],
        ];
        assert_eq!(
            dot_q4_k_row(caps, q4_refs[0], &act0).to_bits(),
            dot_q4_k_row(port, q4_refs[0], &act0).to_bits()
        );
        assert_eq!(
            dot_q4_k_4rows(caps, q4_refs, &act0).map(f32::to_bits),
            dot_q4_k_4rows(port, q4_refs, &act0).map(f32::to_bits)
        );
        assert_eq!(
            gemm_q4_k_4x2(caps, q4_refs, &act0, &act1).map(|pair| pair.map(f32::to_bits)),
            gemm_q4_k_4x2(port, q4_refs, &act0, &act1).map(|pair| pair.map(f32::to_bits))
        );

        let q4_cat: Vec<u8> = q4_rows.iter().flat_map(|row| row.iter().copied()).collect();
        let repacked = repack_q4_k_8rows(&q4_cat, in_dim, 8);
        assert!(repacked.is_some());
        if let Some(strip) = repacked {
            assert_eq!(
                dot_q4_k_repacked8(caps, &strip, &act0).map(f32::to_bits),
                dot_q4_k_repacked8(port, &strip, &act0).map(f32::to_bits)
            );
            assert_eq!(
                gemm_q4_k_repacked8_8x2(caps, &strip, &act0, &act1)
                    .map(|pair| pair.map(f32::to_bits)),
                gemm_q4_k_repacked8_8x2(port, &strip, &act0, &act1)
                    .map(|pair| pair.map(f32::to_bits))
            );
        }
    }

    #[test]
    fn q6_k_and_8lane_kquant_dots_match_portable_reference() {
        let in_dim = 512_usize;
        let (_, act0, act1) = sample_q8_k_inputs(in_dim);
        let caps = Arm64QuantCaps::detect();
        let port = Arm64QuantCaps::portable();

        let q4_row = make_q4_k_row(in_dim / 256, 1);
        let block4 = <&[u8; Q4_K_BLOCK_BYTES]>::try_from(&q4_row[..Q4_K_BLOCK_BYTES]);
        assert!(block4.is_ok());
        if let Ok(blk4) = block4 {
            assert_eq!(
                dot_kquant_8lanes_q4_k(blk4, &act0[0].qs, &act0[0].bsums),
                portable_dot_kquant_8lanes_q4_k(blk4, &act0[0].qs, &act0[0].bsums)
            );
        }

        let q6_rows: Vec<Vec<u8>> = (0..4)
            .map(|row| make_q6_k_row(in_dim / 256, row + 1))
            .collect();
        let q6_refs = [
            &q6_rows[0][..],
            &q6_rows[1][..],
            &q6_rows[2][..],
            &q6_rows[3][..],
        ];
        assert_eq!(
            dot_q6_k_row(caps, q6_refs[0], &act0).to_bits(),
            dot_q6_k_row(port, q6_refs[0], &act0).to_bits()
        );
        assert_eq!(
            dot_q6_k_4rows(caps, q6_refs, &act0).map(f32::to_bits),
            dot_q6_k_4rows(port, q6_refs, &act0).map(f32::to_bits)
        );
        assert_eq!(
            gemm_q6_k_4x2(caps, q6_refs, &act0, &act1).map(|pair| pair.map(f32::to_bits)),
            gemm_q6_k_4x2(port, q6_refs, &act0, &act1).map(|pair| pair.map(f32::to_bits))
        );

        let block6 = <&[u8; Q6_K_BLOCK_BYTES]>::try_from(&q6_rows[0][..Q6_K_BLOCK_BYTES]);
        assert!(block6.is_ok());
        if let Ok(blk6) = block6 {
            assert_eq!(
                dot_kquant_8lanes_q6_k(blk6, &act0[0].qs),
                portable_dot_kquant_8lanes_q6_k(blk6, &act0[0].qs)
            );
        }
    }
}
