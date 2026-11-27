//! Pure-Rust `NariQ` trellis linear and row-lookup kernels with parallel tile scheduling.

#![forbid(unsafe_code)]
#![expect(
    clippy::redundant_pub_crate,
    reason = "Keep graph-consumed trellis types crate scoped under -D unreachable-pub."
)]

use crate::model::{Tensor, TensorBytes};
use crate::{Result, invalid};
use bonsai_gguf::{MetadataArray, MetadataValue};

use super::parallel_rows::RowParallelism;
use super::prefill::MAX_TILE_TOKENS;
use super::{F16, F32, I32, MatvecBackend, scale, trellis_bits};

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
use fearless_simd::{Neon, f32x4, prelude::*, u32x4};

/// Maximum number of fused output slices supported by a `NariQ` trellis descriptor.
pub(crate) const MAX_PARTS: usize = 8;

const TILE_DIM: usize = 16;
const LCG_MULTIPLIER: u32 = 0x83dc_d12d;

const fn make_chi_table(q: usize) -> [i8; 67] {
    let mut table = [-1_i8; 67];
    table[0] = 0;
    let mut i = 1_usize;
    while i < q {
        table[(i * i) % q] = 1;
        i += 1;
    }
    table
}

const fn make_paley<const M: usize>() -> [[i8; M]; M] {
    let q = M - 1;
    let chi = make_chi_table(q);
    let mut h = [[0_i8; M]; M];
    let mut row = 0_usize;
    while row < M {
        let mut col = 0_usize;
        while col < M {
            let s = if row == 0 {
                if col > 0 { 1_i8 } else { 0_i8 }
            } else if col == 0 {
                -1_i8
            } else {
                let diff = (col + q - row) % q;
                chi[diff]
            };
            h[row][col] = s + if row == col { 1_i8 } else { 0_i8 };
            col += 1;
        }
        row += 1;
    }
    h
}

const H_12: [[i8; 12]; 12] = make_paley::<12>();
const H_20: [[i8; 20]; 20] = make_paley::<20>();
const H_68: [[i8; 68]; 68] = make_paley::<68>();

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
const fn make_paley_masks<const M: usize>(mat: &[[i8; M]; M], transpose: bool) -> [[u32; M]; M] {
    let mut out = [[0_u32; M]; M];
    let mut a = 0_usize;
    while a < M {
        let mut b = 0_usize;
        while b < M {
            let sign = if transpose { mat[b][a] } else { mat[a][b] };
            out[a][b] = if sign > 0 { 0_u32 } else { 0x8000_0000_u32 };
            b += 1;
        }
        a += 1;
    }
    out
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
static H_12_MASKS: [[u32; 12]; 12] = make_paley_masks::<12>(&H_12, false);
#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
static H_12_T_MASKS: [[u32; 12]; 12] = make_paley_masks::<12>(&H_12, true);
#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
static H_20_MASKS: [[u32; 20]; 20] = make_paley_masks::<20>(&H_20, false);
#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
static H_20_T_MASKS: [[u32; 20]; 20] = make_paley_masks::<20>(&H_20, true);
#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
static H_68_MASKS: [[u32; 68]; 68] = make_paley_masks::<68>(&H_68, false);
#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
static H_68_T_MASKS: [[u32; 68]; 68] = make_paley_masks::<68>(&H_68, true);

#[inline]
#[expect(
    clippy::cast_possible_truncation,
    clippy::as_conversions,
    reason = "Matches NariQ libtrellis / ggml-trellis 1/sqrt(n) f64->f32 rounding."
)]
const fn narrow_f64(value: f64) -> f32 {
    value as f32
}

fn inv_sqrt_f32(n: usize) -> Result<f32> {
    if n == 0 {
        return Err(invalid("trellis transform length must be positive"));
    }
    let nf = f64::from(u32::try_from(n).map_err(|_| invalid("trellis transform length overflow"))?);
    Ok(narrow_f64(nf.powf(-0.5)))
}

const fn hadfull_m(len: usize) -> usize {
    const FACTORS: [usize; 4] = [1, 12, 20, 68];
    let mut idx = 0_usize;
    while idx < FACTORS.len() {
        let m = FACTORS[idx];
        if len >= m && len.is_multiple_of(m) && (len / m).is_power_of_two() {
            return m;
        }
        idx += 1;
    }
    0
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Parts {
    start: [usize; MAX_PARTS],
    shard: [usize; MAX_PARTS],
    cum: [usize; MAX_PARTS + 1],
    count: usize,
}

impl Parts {
    fn new(total: usize, parts: Option<&[usize]>, shards: usize) -> Result<Self> {
        if shards < 1 {
            return Err(invalid("trellis shard count must be positive"));
        }
        let mut out = Self {
            start: [0; MAX_PARTS],
            shard: [0; MAX_PARTS],
            cum: [0; MAX_PARTS + 1],
            count: 0,
        };
        if let Some(parts) = parts {
            if parts.is_empty() || parts.len() > MAX_PARTS {
                return Err(invalid("trellis fused part count out of bounds"));
            }
            out.count = parts.len();
            let mut offset = 0_usize;
            let mut cum = 0_usize;
            for (idx, &size) in parts.iter().enumerate() {
                if size == 0 || !size.is_multiple_of(shards) {
                    return Err(invalid(
                        "trellis fused part size is not divisible by shards",
                    ));
                }
                let shard_len = size / shards;
                out.start[idx] = offset;
                out.shard[idx] = shard_len;
                out.cum[idx] = cum;
                offset = offset
                    .checked_add(size)
                    .ok_or_else(|| invalid("trellis fused part sum overflow"))?;
                cum = cum
                    .checked_add(shard_len)
                    .ok_or_else(|| invalid("trellis fused shard sum overflow"))?;
            }
            out.cum[out.count] = cum;
            if offset != total {
                return Err(invalid("trellis fused parts do not sum to matrix extent"));
            }
        } else {
            if !total.is_multiple_of(shards) {
                return Err(invalid("trellis matrix extent is not divisible by shards"));
            }
            out.count = 1;
            out.start[0] = 0;
            out.shard[0] = total / shards;
            out.cum[0] = 0;
            out.cum[1] = total / shards;
        }
        Ok(out)
    }

    const fn stored(&self, shard_idx: usize, local_idx: usize) -> usize {
        let mut part = 0_usize;
        while part + 1 < self.count && local_idx >= self.cum[part + 1] {
            part += 1;
        }
        self.start[part] + shard_idx * self.shard[part] + (local_idx - self.cum[part])
    }
}

fn fwht_scaled_scalar(values: &mut [f32], block: usize, scale_factor: f32) {
    let mut half = 1_usize;
    while half < block {
        let step = half * 2;
        for chunk in values.chunks_exact_mut(step) {
            let (left, right) = chunk.split_at_mut(half);
            for (a_slot, b_slot) in left.iter_mut().zip(right.iter_mut()) {
                let a = *a_slot;
                let b = *b_slot;
                *a_slot = a + b;
                *b_slot = a - b;
            }
        }
        half = step;
    }
    for slot in values.iter_mut() {
        *slot *= scale_factor;
    }
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
fn fwht_scaled_neon(neon: Neon, values: &mut [f32], block: usize, scale_factor: f32) {
    if !block.is_power_of_two() || !values.len().is_multiple_of(block) {
        fwht_scaled_scalar(values, block, scale_factor);
        return;
    }
    let scale_vec = f32x4::splat(neon, scale_factor);
    if block == 1 {
        let (chunks, tail) = values.as_chunks_mut::<4>();
        for c4 in chunks {
            let scaled = f32x4::from_slice(neon, c4) * scale_vec;
            scaled.store_slice(c4);
        }
        for slot in tail {
            *slot *= scale_factor;
        }
        return;
    }
    if block == 2 {
        let (chunks, tail) = values.as_chunks_mut::<4>();
        for c4 in chunks {
            let v = f32x4::from_slice(neon, c4);
            let even = v.unzip_low(v);
            let odd = v.unzip_high(v);
            let sums = even + odd;
            let diffs = even - odd;
            let out = sums.zip_low(diffs) * scale_vec;
            out.store_slice(c4);
        }
        fwht_scaled_scalar(tail, 2, scale_factor);
        return;
    }
    if block == 4 {
        for c4 in values.as_chunks_mut::<4>().0 {
            let vec4 = f32x4::from_slice(neon, c4);
            let even = vec4.unzip_low(vec4);
            let odd = vec4.unzip_high(vec4);
            let sums = even + odd;
            let diffs = even - odd;
            let lhs = sums.unzip_low(diffs);
            let rhs = sums.unzip_high(diffs);
            let plus = lhs + rhs;
            let minus = lhs - rhs;
            let out = plus.unzip_low(minus) * scale_vec;
            out.store_slice(c4);
        }
        return;
    }

    // `block >= 8` and `values.len()` is a multiple of `block`, so 8-lane chunks cover all elements.
    for c8 in values.as_chunks_mut::<8>().0 {
        let v0 = f32x4::from_slice(neon, &c8[0..4]);
        let v1 = f32x4::from_slice(neon, &c8[4..8]);
        let even = v0.unzip_low(v1);
        let odd = v0.unzip_high(v1);
        let sums = even + odd;
        let diffs = even - odd;
        let lhs = sums.unzip_low(diffs);
        let rhs = sums.unzip_high(diffs);
        let plus = lhs + rhs;
        let minus = lhs - rhs;
        let lo = plus.unzip_low(minus);
        let hi = plus.unzip_high(minus);
        let mut z0 = lo + hi;
        let mut z1 = lo - hi;
        if block == 8 {
            z0 *= scale_vec;
            z1 *= scale_vec;
        }
        z0.store_slice(&mut c8[0..4]);
        z1.store_slice(&mut c8[4..8]);
    }

    let mut half = 8_usize;
    while half < block {
        let step = half * 2;
        let final_stage = step == block;
        for chunk in values.chunks_exact_mut(step) {
            let (left, right) = chunk.split_at_mut(half);
            for (l4, r4) in left
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(right.as_chunks_mut::<4>().0.iter_mut())
            {
                let a = f32x4::from_slice(neon, l4);
                let b = f32x4::from_slice(neon, r4);
                let mut sum = a + b;
                let mut diff = a - b;
                if final_stage {
                    sum *= scale_vec;
                    diff *= scale_vec;
                }
                sum.store_slice(l4);
                diff.store_slice(r4);
            }
        }
        half = step;
    }
}

fn fwht_scaled(values: &mut [f32], block: usize, scale_factor: f32) {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(neon) = fearless_simd::Level::baseline().as_neon() {
        fwht_scaled_neon(neon, values, block, scale_factor);
        return;
    }
    fwht_scaled_scalar(values, block, scale_factor);
}

/// Applies the normalized blockwise Sylvester Hadamard transform matching `ggml_trellis_rht`.
///
/// # Errors
/// Returns an error if `block` is zero, not a power of two, or does not divide `values.len()`.
pub(crate) fn rht(values: &mut [f32], block: usize) -> Result<()> {
    if !block.is_power_of_two() || !values.len().is_multiple_of(block) {
        return Err(invalid(
            "trellis rht requires power-of-two block dividing vector length",
        ));
    }
    let factor = inv_sqrt_f32(block)?;
    for chunk in values.chunks_exact_mut(block) {
        fwht_scaled(chunk, block, factor);
    }
    Ok(())
}

fn paley_row(m: usize, row: usize) -> Result<&'static [i8]> {
    match m {
        12 => Ok(&H_12[row]),
        20 => Ok(&H_20[row]),
        68 => Ok(&H_68[row]),
        _ => Err(invalid("unsupported Paley matrix dimension")),
    }
}

fn paley_elem(m: usize, row: usize, col: usize) -> Result<i8> {
    match m {
        12 => Ok(H_12[row][col]),
        20 => Ok(H_20[row][col]),
        68 => Ok(H_68[row][col]),
        _ => Err(invalid("unsupported Paley matrix dimension")),
    }
}

fn paley_cross_scalar(
    input: &[f32],
    output: &mut [f32],
    m: usize,
    run: usize,
    transpose: bool,
    factor: f32,
) -> Result<()> {
    let needed = m
        .checked_mul(run)
        .ok_or_else(|| invalid("trellis paley_cross length overflow"))?;
    if input.len() < needed || output.len() < needed {
        return Err(invalid("trellis paley_cross buffer length mismatch"));
    }
    for a in 0..m {
        let out_row = &mut output[a * run..(a + 1) * run];
        out_row.fill(0.0);
        if transpose {
            for b in 0..m {
                let sign = paley_elem(m, b, a)?;
                let x_row = &input[b * run..(b + 1) * run];
                if sign > 0 {
                    for (dst, &src) in out_row.iter_mut().zip(x_row) {
                        *dst += src;
                    }
                } else {
                    for (dst, &src) in out_row.iter_mut().zip(x_row) {
                        *dst -= src;
                    }
                }
            }
        } else {
            let row_signs = paley_row(m, a)?;
            for (b, &sign) in row_signs.iter().enumerate() {
                let x_row = &input[b * run..(b + 1) * run];
                if sign > 0 {
                    for (dst, &src) in out_row.iter_mut().zip(x_row) {
                        *dst += src;
                    }
                } else {
                    for (dst, &src) in out_row.iter_mut().zip(x_row) {
                        *dst -= src;
                    }
                }
            }
        }
        for dst in out_row.iter_mut() {
            *dst *= factor;
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
#[inline]
fn xor_sign_f32x4(neon: Neon, x: u32x4<Neon>, sign_mask: u32) -> f32x4<Neon> {
    (x ^ u32x4::splat(neon, sign_mask)).bitcast::<f32x4<Neon>>()
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
#[expect(
    clippy::too_many_arguments,
    reason = "Passes four-row sign masks, column range, and SIMD token without bundling."
)]
fn paley_4rows_8lanes<const M: usize>(
    neon: Neon,
    input: &[f32],
    output: &mut [f32],
    run: usize,
    a_base: usize,
    col: usize,
    rows: [&[u32; M]; 4],
    factor_vec: f32x4<Neon>,
) {
    let [r0, r1, r2, r3] = rows;
    let mut acc0_lo = f32x4::splat(neon, 0.0);
    let mut acc0_hi = f32x4::splat(neon, 0.0);
    let mut acc1_lo = f32x4::splat(neon, 0.0);
    let mut acc1_hi = f32x4::splat(neon, 0.0);
    let mut acc2_lo = f32x4::splat(neon, 0.0);
    let mut acc2_hi = f32x4::splat(neon, 0.0);
    let mut acc3_lo = f32x4::splat(neon, 0.0);
    let mut acc3_hi = f32x4::splat(neon, 0.0);
    for b in 0..M {
        let x_slice = &input[b * run + col..b * run + col + 8];
        let x_lo = f32x4::from_slice(neon, &x_slice[0..4]).bitcast::<u32x4<Neon>>();
        let x_hi = f32x4::from_slice(neon, &x_slice[4..8]).bitcast::<u32x4<Neon>>();
        acc0_lo += xor_sign_f32x4(neon, x_lo, r0[b]);
        acc0_hi += xor_sign_f32x4(neon, x_hi, r0[b]);
        acc1_lo += xor_sign_f32x4(neon, x_lo, r1[b]);
        acc1_hi += xor_sign_f32x4(neon, x_hi, r1[b]);
        acc2_lo += xor_sign_f32x4(neon, x_lo, r2[b]);
        acc2_hi += xor_sign_f32x4(neon, x_hi, r2[b]);
        acc3_lo += xor_sign_f32x4(neon, x_lo, r3[b]);
        acc3_hi += xor_sign_f32x4(neon, x_hi, r3[b]);
    }
    let out_accs = [
        (0, acc0_lo, acc0_hi),
        (1, acc1_lo, acc1_hi),
        (2, acc2_lo, acc2_hi),
        (3, acc3_lo, acc3_hi),
    ];
    for (dr, lo, hi) in out_accs {
        let row_off = (a_base + dr) * run + col;
        (lo * factor_vec).store_slice(&mut output[row_off..row_off + 4]);
        (hi * factor_vec).store_slice(&mut output[row_off + 4..row_off + 8]);
    }
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
fn paley_cross_neon_m<const M: usize>(
    neon: Neon,
    input: &[f32],
    output: &mut [f32],
    run: usize,
    masks: &[[u32; M]; M],
    factor: f32,
) -> Result<()> {
    let needed = M
        .checked_mul(run)
        .ok_or_else(|| invalid("trellis paley_cross length overflow"))?;
    if input.len() < needed || output.len() < needed {
        return Err(invalid("trellis paley_cross buffer length mismatch"));
    }
    if run == 0 {
        return Ok(());
    }
    let factor_vec = f32x4::splat(neon, factor);
    if run == 1 {
        for a_base in (0..M).step_by(4) {
            let [r0, r1, r2, r3] = [
                &masks[a_base],
                &masks[a_base + 1],
                &masks[a_base + 2],
                &masks[a_base + 3],
            ];
            let mut acc = f32x4::splat(neon, 0.0);
            for (b, &x_val) in input[..M].iter().enumerate() {
                let xb = f32x4::splat(neon, x_val).bitcast::<u32x4<Neon>>();
                let sign_mask = u32x4::from_slice(neon, &[r0[b], r1[b], r2[b], r3[b]]);
                acc += (xb ^ sign_mask).bitcast::<f32x4<Neon>>();
            }
            (acc * factor_vec).store_slice(&mut output[a_base..a_base + 4]);
        }
        return Ok(());
    }

    for a_base in (0..M).step_by(4) {
        let rows = [
            &masks[a_base],
            &masks[a_base + 1],
            &masks[a_base + 2],
            &masks[a_base + 3],
        ];
        let [r0, r1, r2, r3] = rows;
        let mut col = 0_usize;
        while col + 8 <= run {
            paley_4rows_8lanes(neon, input, output, run, a_base, col, rows, factor_vec);
            col += 8;
        }
        if col + 4 <= run {
            let mut acc0 = f32x4::splat(neon, 0.0);
            let mut acc1 = f32x4::splat(neon, 0.0);
            let mut acc2 = f32x4::splat(neon, 0.0);
            let mut acc3 = f32x4::splat(neon, 0.0);
            for b in 0..M {
                let x = f32x4::from_slice(neon, &input[b * run + col..b * run + col + 4])
                    .bitcast::<u32x4<Neon>>();
                acc0 += xor_sign_f32x4(neon, x, r0[b]);
                acc1 += xor_sign_f32x4(neon, x, r1[b]);
                acc2 += xor_sign_f32x4(neon, x, r2[b]);
                acc3 += xor_sign_f32x4(neon, x, r3[b]);
            }
            for (dr, acc) in [(0, acc0), (1, acc1), (2, acc2), (3, acc3)] {
                let row_off = (a_base + dr) * run + col;
                (acc * factor_vec).store_slice(&mut output[row_off..row_off + 4]);
            }
            col += 4;
        }
        for tail_col in col..run {
            let mut acc0 = 0.0_f32;
            let mut acc1 = 0.0_f32;
            let mut acc2 = 0.0_f32;
            let mut acc3 = 0.0_f32;
            for b in 0..M {
                let x_bits = input[b * run + tail_col].to_bits();
                acc0 += f32::from_bits(x_bits ^ r0[b]);
                acc1 += f32::from_bits(x_bits ^ r1[b]);
                acc2 += f32::from_bits(x_bits ^ r2[b]);
                acc3 += f32::from_bits(x_bits ^ r3[b]);
            }
            output[a_base * run + tail_col] = acc0 * factor;
            output[(a_base + 1) * run + tail_col] = acc1 * factor;
            output[(a_base + 2) * run + tail_col] = acc2 * factor;
            output[(a_base + 3) * run + tail_col] = acc3 * factor;
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
fn paley_cross_neon(
    neon: Neon,
    input: &[f32],
    output: &mut [f32],
    m: usize,
    run: usize,
    transpose: bool,
    factor: f32,
) -> Result<()> {
    match (m, transpose) {
        (12, false) => paley_cross_neon_m(neon, input, output, run, &H_12_MASKS, factor),
        (12, true) => paley_cross_neon_m(neon, input, output, run, &H_12_T_MASKS, factor),
        (20, false) => paley_cross_neon_m(neon, input, output, run, &H_20_MASKS, factor),
        (20, true) => paley_cross_neon_m(neon, input, output, run, &H_20_T_MASKS, factor),
        (68, false) => paley_cross_neon_m(neon, input, output, run, &H_68_MASKS, factor),
        (68, true) => paley_cross_neon_m(neon, input, output, run, &H_68_T_MASKS, factor),
        _ => Err(invalid("unsupported Paley matrix dimension")),
    }
}

fn paley_cross(
    input: &[f32],
    output: &mut [f32],
    m: usize,
    run: usize,
    transpose: bool,
    factor: f32,
) -> Result<()> {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(neon) = fearless_simd::Level::baseline().as_neon() {
        return paley_cross_neon(neon, input, output, m, run, transpose, factor);
    }
    paley_cross_scalar(input, output, m, run, transpose, factor)
}

/// Applies the whole-slice `NariQ` Hadamard transform (`m * 2^j`, `m in {1, 12, 20, 68}`) on shard `shard_idx`.
///
/// # Errors
/// Returns an error if slice parts, shard bounds, or temporary workspace length are invalid.
#[expect(
    clippy::too_many_arguments,
    reason = "Matches ggml_trellis_hadfull_slice parameters directly without hidden allocations."
)]
pub(crate) fn hadfull_slice(
    values: &mut [f32],
    parts: Option<&[usize]>,
    shards: usize,
    shard_idx: usize,
    inverse: bool,
    pre: Option<&[f32]>,
    post: Option<&[f32]>,
    tmp: &mut [f32],
) -> Result<()> {
    let n = values.len();
    let parsed_parts = Parts::new(n, parts, shards)?;
    if shard_idx >= shards {
        return Err(invalid("trellis shard index out of bounds"));
    }
    let len = n / shards;
    let m = hadfull_m(len);
    if m == 0 {
        return Err(invalid("trellis whole-slice length is not of form m * 2^j"));
    }
    let needed = len
        .checked_mul(2)
        .ok_or_else(|| invalid("trellis scratch length overflow"))?;
    if tmp.len() < needed {
        return Err(invalid("trellis hadfull temporary buffer is undersized"));
    }
    let run = len / m;
    let c_run = inv_sqrt_f32(run)?;
    let c_m = inv_sqrt_f32(m)?;
    let (a_buf, b_rest) = tmp.split_at_mut(len);
    let b_buf = &mut b_rest[..len];

    for (l, dst) in a_buf.iter_mut().enumerate() {
        let idx = parsed_parts.stored(shard_idx, l);
        *dst = pre.map_or(values[idx], |scale_pre| values[idx] * scale_pre[idx]);
    }

    let result_slice: &[f32] = if m == 1 {
        fwht_scaled(a_buf, len, c_run);
        a_buf
    } else if !inverse {
        for chunk in a_buf.chunks_exact_mut(run) {
            fwht_scaled(chunk, run, c_run);
        }
        paley_cross(a_buf, b_buf, m, run, false, c_m)?;
        b_buf
    } else {
        paley_cross(a_buf, b_buf, m, run, true, c_m)?;
        for chunk in b_buf.chunks_exact_mut(run) {
            fwht_scaled(chunk, run, c_run);
        }
        b_buf
    };

    for (l, &src) in result_slice.iter().enumerate() {
        let idx = parsed_parts.stored(shard_idx, l);
        values[idx] = post.map_or(src, |scale_post| src * scale_post[idx]);
    }
    Ok(())
}

/// Transform specification for one side of a `NariQ` coded matrix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SideTransform {
    /// Identity transform (`kind = 0`).
    Identity,
    /// Blockwise normalized Sylvester Hadamard (`kind = 1`).
    BlockRht {
        /// Power-of-two transform block length (`>= 2`).
        block: usize,
    },
    /// Sharded whole-slice Hadamard with optional Paley factor (`kind = 2`).
    WholeSliceHadfull {
        /// Number of equal-length shards along the dimension.
        shards: usize,
        /// Optional fused slice partition sizes (`parts` on output side).
        parts: Option<Vec<usize>>,
    },
}

/// Parsed and validated `trellis.desc.<stem>` metadata for one coded tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TrellisDesc {
    pub(crate) rows: SideTransform,
    pub(crate) cols: SideTransform,
}

impl TrellisDesc {
    /// Parses a `trellis.desc.*` `I32` metadata array and validates its transform constraints.
    ///
    /// # Errors
    /// Rejects non-`I32` arrays, invalid lengths, unsupported transform kinds, bad divisors,
    /// non-matching part sums, or shapes not divisible by `16`.
    pub(crate) fn from_metadata(k: usize, n: usize, value: &MetadataValue) -> Result<Self> {
        let raw = match value {
            MetadataValue::Array(MetadataArray::I32(values)) => values.as_slice(),
            _ => return Err(invalid("trellis.desc metadata must be an I32 array")),
        };
        Self::from_i32_slice(k, n, raw)
    }

    pub(crate) fn from_i32_slice(k: usize, n: usize, raw: &[i32]) -> Result<Self> {
        if !k.is_multiple_of(TILE_DIM) || !n.is_multiple_of(TILE_DIM) || k == 0 || n == 0 {
            return Err(invalid(
                "trellis matrix dimensions must be positive multiples of 16",
            ));
        }
        if raw.len() < 7 {
            return Err(invalid("trellis.desc requires at least 7 entries"));
        }
        let nparts = usize::try_from(raw[6]).map_err(|_| invalid("negative trellis part count"))?;
        if nparts > MAX_PARTS || raw.len() != 7 + nparts {
            return Err(invalid("trellis.desc length or part count invalid"));
        }
        let mut parts_vec = Vec::with_capacity(nparts);
        for &entry in &raw[7..] {
            parts_vec
                .push(usize::try_from(entry).map_err(|_| invalid("negative trellis part size"))?);
        }
        let rows_parts = (!parts_vec.is_empty()).then_some(parts_vec);
        let rows = parse_side(n, raw[0], raw[1], raw[2], rows_parts)?;
        let cols = parse_side(k, raw[3], raw[4], raw[5], None)?;
        Ok(Self { rows, cols })
    }
}

fn parse_side(
    extent: usize,
    kind: i32,
    block_raw: i32,
    shards_raw: i32,
    parts: Option<Vec<usize>>,
) -> Result<SideTransform> {
    match kind {
        0 => Ok(SideTransform::Identity),
        1 => {
            let block = usize::try_from(block_raw)
                .map_err(|_| invalid("trellis RHT block must be nonnegative"))?;
            if block < 2 || !block.is_power_of_two() || !extent.is_multiple_of(block) {
                return Err(invalid("invalid trellis RHT block size"));
            }
            Ok(SideTransform::BlockRht { block })
        }
        2 => {
            let shards = usize::try_from(shards_raw)
                .map_err(|_| invalid("trellis shard count must be positive"))?;
            let _ = Parts::new(extent, parts.as_deref(), shards)?;
            let slice_len = extent / shards;
            if hadfull_m(slice_len) == 0 {
                return Err(invalid("invalid trellis whole-slice Hadamard length"));
            }
            Ok(SideTransform::WholeSliceHadfull { shards, parts })
        }
        _ => Err(invalid("unsupported trellis transform kind")),
    }
}

fn read_f16_vector(tensor: &Tensor, expected_len: usize, label: &str) -> Result<Box<[f32]>> {
    if tensor.kind != F16 || tensor.dimensions != [expected_len] {
        return Err(invalid(format!(
            "{label} must be 1D F16 of length {expected_len}"
        )));
    }
    let bytes: &[u8] = &tensor.data;
    if bytes.len() != expected_len * 2 {
        return Err(invalid(format!("{label} byte length mismatch")));
    }
    let mut out = Vec::with_capacity(expected_len);
    for pair in bytes.as_chunks::<2>().0 {
        let value = scale(pair)?;
        out.push(value);
    }
    Ok(out.into_boxed_slice())
}

fn read_f32_vector(tensor: &Tensor, expected_len: usize, label: &str) -> Result<Box<[f32]>> {
    if tensor.kind != F32 || tensor.dimensions != [expected_len] {
        return Err(invalid(format!(
            "{label} must be 1D F32 of length {expected_len}"
        )));
    }
    let bytes: &[u8] = &tensor.data;
    if bytes.len() != expected_len * 4 {
        return Err(invalid(format!("{label} byte length mismatch")));
    }
    let mut out = Vec::with_capacity(expected_len);
    for quad in bytes.as_chunks::<4>().0 {
        let value = f32::from_le_bytes(*quad);
        if !value.is_finite() {
            return Err(invalid(format!("nonfinite float in {label}")));
        }
        out.push(value);
    }
    Ok(out.into_boxed_slice())
}

/// Validates and decodes an `I32` permutation tensor of length `expected_len`.
///
/// Enforces an exact bijection on `0..expected_len` so downstream gathers remain strictly bounded.
///
/// # Errors
/// Rejects wrong type/dimensions, negative indices, out-of-range values, and duplicate entries.
pub(crate) fn validate_permutation_i32(tensor: &Tensor, expected_len: usize) -> Result<Box<[u32]>> {
    if tensor.kind != I32 || tensor.dimensions != [expected_len] {
        return Err(invalid(format!(
            "{} must be 1D I32 of length {expected_len}",
            tensor.name
        )));
    }
    let bytes: &[u8] = &tensor.data;
    if bytes.len() != expected_len * 4 {
        return Err(invalid(format!("{} byte length mismatch", tensor.name)));
    }
    let mut seen = vec![false; expected_len];
    let mut indices = Vec::with_capacity(expected_len);
    for quad in bytes.as_chunks::<4>().0 {
        let raw = i32::from_le_bytes(*quad);
        let idx = usize::try_from(raw)
            .map_err(|_| invalid(format!("negative gather index in {}", tensor.name)))?;
        let slot = seen
            .get_mut(idx)
            .ok_or_else(|| invalid(format!("out-of-range gather index in {}", tensor.name)))?;
        if *slot {
            return Err(invalid(format!(
                "duplicate gather index in {}",
                tensor.name
            )));
        }
        *slot = true;
        indices.push(u32::try_from(idx)?);
    }
    Ok(indices.into_boxed_slice())
}

/// Verified coded `NariQ` linear or embedding weight plus side scales, permutations, and transforms.
#[derive(Clone, Debug)]
pub(crate) struct TrellisWeight {
    pub(crate) name: String,
    pub(crate) bits: usize,
    pub(crate) k: usize,
    pub(crate) n: usize,
    pub(crate) data: TensorBytes,
    pub(crate) s16: Box<[f32]>,
    pub(crate) pre: Box<[f32]>,
    pub(crate) post: Box<[f32]>,
    pub(crate) pin: Option<Box<[u32]>>,
    pub(crate) pout: Option<Box<[u32]>>,
    pub(crate) desc: TrellisDesc,
}

impl TrellisWeight {
    /// Constructs a validated [`TrellisWeight`] from loaded tensors and metadata descriptor.
    ///
    /// # Errors
    /// Rejects non-trellis tensor types, non-2D or unaligned dimensions, mismatched tile byte counts,
    /// nonfinite scales, non-permutation gathers, or invalid descriptors.
    pub(crate) fn from_tensors(
        weight: &Tensor,
        s16: &Tensor,
        pre: &Tensor,
        post: &Tensor,
        pin: Option<&Tensor>,
        out_perm: Option<&Tensor>,
        desc: TrellisDesc,
    ) -> Result<Self> {
        let bits = trellis_bits(weight.kind)
            .ok_or_else(|| invalid(format!("{} is not a TRL1..TRL6 tensor", weight.name)))?;
        if weight.dimensions.len() != 2 {
            return Err(invalid(format!(
                "{} must be a 2D coded tensor",
                weight.name
            )));
        }
        let k = weight.dimensions[0];
        let n = weight.dimensions[1];
        if !k.is_multiple_of(TILE_DIM) || !n.is_multiple_of(TILE_DIM) || k == 0 || n == 0 {
            return Err(invalid(format!(
                "{} dimensions [{k}, {n}] must be positive multiples of 16",
                weight.name
            )));
        }
        let tile_cols = k / TILE_DIM;
        let tile_rows = n / TILE_DIM;
        let expected_bytes = tile_cols
            .checked_mul(tile_rows)
            .and_then(|tiles| tiles.checked_mul(32 * bits))
            .ok_or_else(|| invalid("trellis payload size overflow"))?;
        if weight.data.len() != expected_bytes {
            return Err(invalid(format!(
                "{} payload length {} does not match expected {expected_bytes}",
                weight.name,
                weight.data.len()
            )));
        }
        let s16_vec = read_f16_vector(s16, tile_cols, &s16.name)?;
        let pre_vec = read_f32_vector(pre, k, &pre.name)?;
        let post_vec = read_f32_vector(post, n, &post.name)?;
        let pin_vec = pin
            .map(|tensor| validate_permutation_i32(tensor, k))
            .transpose()?;
        let out_perm_vec = out_perm
            .map(|tensor| validate_permutation_i32(tensor, n))
            .transpose()?;
        Ok(Self {
            name: weight.name.clone(),
            bits,
            k,
            n,
            data: weight.data.clone(),
            s16: s16_vec,
            pre: pre_vec,
            post: post_vec,
            pin: pin_vec,
            pout: out_perm_vec,
            desc,
        })
    }

    pub(crate) const fn strip_bytes(&self) -> usize {
        (self.k / TILE_DIM) * 32 * self.bits
    }
}

/// Reusable scratch buffers for `NariQ` trellis matvec and multi-token tiled execution.
#[derive(Clone, Debug, Default)]
pub(crate) struct TrellisScratch {
    pub(crate) xr: Vec<f32>,
    pub(crate) yb: Vec<f32>,
    pub(crate) tmp: Vec<f32>,
    pub(crate) band_out: Vec<f32>,
    pub(crate) head_concat: Vec<f32>,
    pub(crate) multi_xr: Vec<Vec<f32>>,
    pub(crate) multi_yb_flat: Vec<f32>,
    pub(crate) multi_yb_token: Vec<Vec<f32>>,
    pub(crate) head_concat_tokens: Vec<Vec<f32>>,
}

fn ensure_len(buf: &mut Vec<f32>, len: usize) {
    if buf.len() == len {
        buf.fill(0.0);
    } else {
        buf.resize(len, 0.0);
    }
}

fn decode_tile_single_value(tile: &[u8], bits: usize, row: usize, col: usize) -> f32 {
    let r_hi = row / 8;
    let r_lo = row % 8;
    let c_hi = col / 8;
    let c_lo = col % 8;
    let p = 8 * (4 * r_lo + (c_lo / 2)) + 4 * r_hi + 2 * c_hi + (col & 1);
    let total_bits = 256 * bits;
    let n_words = 8 * bits;
    let start = ((p + 1) * bits + total_bits - 16) % total_bits;
    let w = start >> 5;
    let o = start & 31;
    let w_next = (w + 1) % n_words;
    let read_word = |idx: usize| -> u32 {
        let off = idx * 4;
        tile.get(off..off + 4)
            .and_then(|slice| <[u8; 4]>::try_from(slice).ok())
            .map_or(0, u32::from_le_bytes)
    };
    let pair = (u64::from(read_word(w)) << 32) | u64::from(read_word(w_next));
    let state = u32::try_from((pair >> (48 - o)) & 0xffff).unwrap_or(0);
    let x = state.wrapping_mul(LCG_MULTIPLIER);
    let sum = ((x >> 2) & 63) + ((x >> 10) & 63) + ((x >> 18) & 63) + ((x >> 26) & 63);
    f32::from(i16::try_from(sum).map_or(0, |v| v - 126))
}

fn scalar_tile_row_matvec(
    row_tiles: &[u8],
    bits: usize,
    x_scaled: &[f32],
    out: &mut [f32; TILE_DIM],
) -> Result<()> {
    let tile_bytes = 32 * bits;
    if !x_scaled.len().is_multiple_of(TILE_DIM) {
        return Err(invalid("trellis input vector length must divide 16"));
    }
    let tile_cols = x_scaled.len() / TILE_DIM;
    if row_tiles.len() != tile_cols * tile_bytes {
        return Err(invalid("trellis strip byte length mismatch"));
    }
    let mut acc = [0.0_f32; TILE_DIM];
    for i in 0..tile_cols {
        let tile = &row_tiles[i * tile_bytes..(i + 1) * tile_bytes];
        let x_tile = &x_scaled[i * TILE_DIM..(i + 1) * TILE_DIM];
        for (c, &xc) in x_tile.iter().enumerate() {
            for (r, slot) in acc.iter_mut().enumerate() {
                let wf = decode_tile_single_value(tile, bits, r, c);
                *slot = wf.mul_add(xc, *slot);
            }
        }
    }
    *out = acc;
    Ok(())
}

fn dispatch_strip_matvec(
    row_tiles: &[u8],
    bits: usize,
    x_scaled: &[f32],
    out: &mut [f32; TILE_DIM],
    backend: MatvecBackend,
) -> Result<()> {
    #[cfg(all(
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        feature = "experimental-neon",
        target_feature = "neon"
    ))]
    if backend != MatvecBackend::Scalar {
        return bonsai_neon_dot::trellis::tile_row_matvec(row_tiles, bits, x_scaled, out)
            .map_err(invalid);
    }
    let _ = backend;
    scalar_tile_row_matvec(row_tiles, bits, x_scaled, out)
}

fn dispatch_strip_matmul_tokens(
    row_tiles: &[u8],
    bits: usize,
    x_tokens: &[&[f32]],
    out_tokens: &mut [&mut [f32; TILE_DIM]],
    backend: MatvecBackend,
) -> Result<()> {
    #[cfg(all(
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        feature = "experimental-neon",
        target_feature = "neon"
    ))]
    if backend != MatvecBackend::Scalar {
        return bonsai_neon_dot::trellis::tile_row_matmul_tokens(
            row_tiles, bits, x_tokens, out_tokens,
        )
        .map_err(invalid);
    }
    let _ = backend;
    for (&x_scaled, out) in x_tokens.iter().zip(out_tokens.iter_mut()) {
        scalar_tile_row_matvec(row_tiles, bits, x_scaled, out)?;
    }
    Ok(())
}

fn apply_side_transform(
    values: &mut [f32],
    side: &SideTransform,
    inverse: bool,
    pre: Option<&[f32]>,
    post: Option<&[f32]>,
    tmp: &mut [f32],
) -> Result<()> {
    if let Some(pre) = pre {
        for (dst, &scale_pre) in values.iter_mut().zip(pre) {
            *dst *= scale_pre;
        }
    }
    match side {
        SideTransform::Identity => {
            if let Some(post) = post {
                for (dst, &scale_post) in values.iter_mut().zip(post) {
                    *dst *= scale_post;
                }
            }
        }
        SideTransform::BlockRht { block } => {
            rht(values, *block)?;
            if let Some(post) = post {
                for (dst, &scale_post) in values.iter_mut().zip(post) {
                    *dst *= scale_post;
                }
            }
        }
        SideTransform::WholeSliceHadfull { shards, parts } => {
            for shard_idx in 0..*shards {
                hadfull_slice(
                    values,
                    parts.as_deref(),
                    *shards,
                    shard_idx,
                    inverse,
                    None,
                    post,
                    tmp,
                )?;
            }
        }
    }
    Ok(())
}

fn transform_input_into(
    weight: &TrellisWeight,
    input: &[f32],
    xr: &mut [f32],
    tmp: &mut [f32],
) -> Result<()> {
    if input.len() != weight.k {
        return Err(invalid(format!(
            "trellis {} expected {} input values, got {}",
            weight.name,
            weight.k,
            input.len()
        )));
    }
    if let Some(pin) = &weight.pin {
        for ((dst, &src_idx), &pre_scale) in xr.iter_mut().zip(pin.iter()).zip(weight.pre.iter()) {
            *dst = input[usize::try_from(src_idx)?] * pre_scale;
        }
    } else {
        for ((dst, &src), &pre_scale) in xr.iter_mut().zip(input).zip(weight.pre.iter()) {
            *dst = src * pre_scale;
        }
    }
    apply_side_transform(xr, &weight.desc.cols, false, None, None, tmp)?;
    for (chunk, &tile_scale) in xr
        .as_chunks_mut::<TILE_DIM>()
        .0
        .iter_mut()
        .zip(weight.s16.iter())
    {
        for dst in chunk {
            *dst *= tile_scale;
        }
    }
    Ok(())
}

fn check_all_finite(values: &[f32], label: &str) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(invalid(format!("nonfinite value in {label}")));
    }
    Ok(())
}

/// Decodes one row (`row < weight.n`) of an un-row-rotated coded embedding tensor (`rows_kind == 0`).
///
/// # Errors
/// Rejects tensors with non-identity row transforms, out-of-range rows, mismatched workspace, or nonfinite outputs.
pub(crate) fn trellis_get_row(
    weight: &TrellisWeight,
    row: usize,
    output: &mut Vec<f32>,
    tmp: &mut Vec<f32>,
) -> Result<()> {
    if weight.desc.rows != SideTransform::Identity {
        return Err(invalid(format!(
            "trellis row lookup requires identity row transform on {}",
            weight.name
        )));
    }
    if row >= weight.n {
        return Err(invalid(format!(
            "trellis row index {row} exceeds height {}",
            weight.n
        )));
    }
    let src_row = match &weight.pout {
        Some(pout) => usize::try_from(pout[row])?,
        None => row,
    };
    let strip_idx = src_row / TILE_DIM;
    let in_tile_row = src_row % TILE_DIM;
    let tile_bytes = 32 * weight.bits;
    let strip_bytes = weight.strip_bytes();
    let strip = &weight.data[strip_idx * strip_bytes..(strip_idx + 1) * strip_bytes];

    let needed_tmp = weight.k * 2 + weight.k;
    if tmp.len() < needed_tmp {
        tmp.resize(needed_tmp, 0.0);
    }
    let (row_buf, scratch_tmp) = tmp.split_at_mut(weight.k);
    for (tile_col_idx, (chunk, &s16_scale)) in row_buf
        .as_chunks_mut::<TILE_DIM>()
        .0
        .iter_mut()
        .zip(weight.s16.iter())
        .enumerate()
    {
        let tile = &strip[tile_col_idx * tile_bytes..(tile_col_idx + 1) * tile_bytes];
        for (col, slot) in chunk.iter_mut().enumerate() {
            *slot = decode_tile_single_value(tile, weight.bits, in_tile_row, col) * s16_scale;
        }
    }

    apply_side_transform(row_buf, &weight.desc.cols, true, None, None, scratch_tmp)?;
    let post_scale = weight.post[src_row];
    for (slot, &pre_scale) in row_buf.iter_mut().zip(weight.pre.iter()) {
        *slot *= pre_scale * post_scale;
    }

    ensure_len(output, weight.k);
    if let Some(pin) = &weight.pin {
        for (&val, &dst_idx) in row_buf.iter().zip(pin.iter()) {
            output[usize::try_from(dst_idx)?] = val;
        }
    } else {
        output.copy_from_slice(row_buf);
    }
    check_all_finite(output, &weight.name)
}

/// Evaluates `output = W * input` for one coded trellis matrix using preallocated `scratch`.
///
/// # Errors
/// Rejects dimension mismatches, nonfinite inputs/outputs, and invalid descriptor transforms.
pub(crate) fn trellis_matvec_into(
    weight: &TrellisWeight,
    input: &[f32],
    output: &mut Vec<f32>,
    backend: MatvecBackend,
    parallelism: &RowParallelism,
    scratch: &mut TrellisScratch,
) -> Result<()> {
    check_all_finite(input, "trellis activation input")?;
    ensure_len(&mut scratch.xr, weight.k);
    ensure_len(&mut scratch.yb, weight.n);
    let tmp_len = (weight.k.max(weight.n)) * 2;
    if scratch.tmp.len() < tmp_len {
        scratch.tmp.resize(tmp_len, 0.0);
    }

    transform_input_into(weight, input, &mut scratch.xr, &mut scratch.tmp)?;

    let strip_bytes = weight.strip_bytes();
    let data: &[u8] = &weight.data;
    if parallelism.workers() > 1 && weight.n >= TILE_DIM * 2 {
        parallelism.update_chunks(&mut scratch.yb, TILE_DIM, |base_strip, chunk| {
            for (local_idx, out_strip) in chunk.as_chunks_mut::<TILE_DIM>().0.iter_mut().enumerate()
            {
                let strip_idx = base_strip + local_idx;
                let row_tiles = &data[strip_idx * strip_bytes..(strip_idx + 1) * strip_bytes];
                let mut local = [0.0_f32; TILE_DIM];
                dispatch_strip_matvec(row_tiles, weight.bits, &scratch.xr, &mut local, backend)?;
                out_strip.copy_from_slice(&local);
            }
            Ok(())
        })?;
    } else {
        for (strip_idx, out_strip) in scratch
            .yb
            .as_chunks_mut::<TILE_DIM>()
            .0
            .iter_mut()
            .enumerate()
        {
            let row_tiles = &data[strip_idx * strip_bytes..(strip_idx + 1) * strip_bytes];
            let mut local = [0.0_f32; TILE_DIM];
            dispatch_strip_matvec(row_tiles, weight.bits, &scratch.xr, &mut local, backend)?;
            out_strip.copy_from_slice(&local);
        }
    }

    apply_side_transform(
        &mut scratch.yb,
        &weight.desc.rows,
        true,
        None,
        Some(&weight.post),
        &mut scratch.tmp,
    )?;

    ensure_len(output, weight.n);
    if let Some(pout) = &weight.pout {
        for (dst, &src_idx) in output.iter_mut().zip(pout.iter()) {
            *dst = scratch.yb[usize::try_from(src_idx)?];
        }
    } else {
        output.copy_from_slice(&scratch.yb);
    }
    check_all_finite(output, &weight.name)
}

fn finish_trellis_token_output(
    weight: &TrellisWeight,
    yb_flat: &[f32],
    strip_stride: usize,
    token_idx: usize,
    yb: &mut Vec<f32>,
    output: &mut Vec<f32>,
    tmp: &mut [f32],
) -> Result<()> {
    if yb.len() != weight.n {
        yb.resize(weight.n, 0.0);
    }
    let strip_count = weight.n / TILE_DIM;
    let token_col_offset = token_idx * TILE_DIM;
    for strip_idx in 0..strip_count {
        let src_off = strip_idx * strip_stride + token_col_offset;
        let dst_off = strip_idx * TILE_DIM;
        yb[dst_off..dst_off + TILE_DIM].copy_from_slice(&yb_flat[src_off..src_off + TILE_DIM]);
    }
    apply_side_transform(yb, &weight.desc.rows, true, None, Some(&weight.post), tmp)?;
    ensure_len(output, weight.n);
    if let Some(pout) = &weight.pout {
        for (dst, &src_idx) in output.iter_mut().zip(pout.iter()) {
            *dst = yb[usize::try_from(src_idx)?];
        }
    } else {
        output.copy_from_slice(yb);
    }
    check_all_finite(output, &weight.name)
}

/// Evaluates `outputs[t] = W * inputs[t]` across a batch of tokens, sharing decoded weight tiles.
///
/// # Errors
/// Rejects token count mismatches, mismatched activation lengths, nonfinite values, and invalid transforms.
#[expect(
    clippy::too_many_lines,
    reason = "Keeps pre-transform, parallel strip dispatch, scatter, and post-transform in one scratch-borrowing pass."
)]
pub(crate) fn trellis_matmul_tokens_into(
    weight: &TrellisWeight,
    inputs: &[Vec<f32>],
    outputs: &mut [Vec<f32>],
    backend: MatvecBackend,
    parallelism: &RowParallelism,
    scratch: &mut TrellisScratch,
) -> Result<()> {
    if inputs.len() != outputs.len() {
        return Err(invalid(
            "trellis token batch input and output counts differ",
        ));
    }
    let token_count = inputs.len();
    if token_count == 0 {
        return Ok(());
    }
    if token_count == 1 {
        return trellis_matvec_into(
            weight,
            &inputs[0],
            &mut outputs[0],
            backend,
            parallelism,
            scratch,
        );
    }

    let per_token_tmp_len = (weight.k.max(weight.n)) * 2;
    let parallel_tokens = token_count >= 2 && parallelism.workers() > 1;
    let needed_tmp = if parallel_tokens {
        token_count
            .checked_mul(per_token_tmp_len)
            .ok_or_else(|| invalid("trellis scratch length overflow"))?
    } else {
        per_token_tmp_len
    };
    if scratch.tmp.len() < needed_tmp {
        scratch.tmp.resize(needed_tmp, 0.0);
    }
    if scratch.multi_xr.len() < token_count {
        scratch.multi_xr.resize_with(token_count, Vec::new);
    }
    if scratch.multi_yb_token.len() < token_count {
        scratch.multi_yb_token.resize_with(token_count, Vec::new);
    }

    if parallel_tokens {
        parallelism.update_paired_chunks(
            &mut scratch.multi_xr[..token_count],
            1,
            &mut scratch.tmp[..needed_tmp],
            per_token_tmp_len,
            |base_token, xr_chunk, tmp_chunk| {
                for (local_idx, (xr, tmp)) in xr_chunk
                    .iter_mut()
                    .zip(tmp_chunk.chunks_exact_mut(per_token_tmp_len))
                    .enumerate()
                {
                    let input = &inputs[base_token + local_idx];
                    check_all_finite(input, "trellis batch activation input")?;
                    if xr.len() != weight.k {
                        xr.resize(weight.k, 0.0);
                    }
                    transform_input_into(weight, input, xr, tmp)?;
                }
                Ok(())
            },
        )?;
    } else {
        for (token_idx, input) in inputs.iter().enumerate() {
            check_all_finite(input, "trellis batch activation input")?;
            if scratch.multi_xr[token_idx].len() != weight.k {
                scratch.multi_xr[token_idx].resize(weight.k, 0.0);
            }
            transform_input_into(
                weight,
                input,
                &mut scratch.multi_xr[token_idx],
                &mut scratch.tmp[..per_token_tmp_len],
            )?;
        }
    }

    let strip_count = weight.n / TILE_DIM;
    let strip_stride = TILE_DIM * token_count;
    ensure_len(&mut scratch.multi_yb_flat, strip_count * strip_stride);

    let mut x_stack: [&[f32]; MAX_TILE_TOKENS] = [&[]; MAX_TILE_TOKENS];
    let mut x_heap = Vec::new();
    let x_refs: &[&[f32]] = if token_count <= MAX_TILE_TOKENS {
        for (dst, src) in x_stack[..token_count]
            .iter_mut()
            .zip(scratch.multi_xr[..token_count].iter())
        {
            *dst = src.as_slice();
        }
        &x_stack[..token_count]
    } else {
        x_heap.extend(scratch.multi_xr[..token_count].iter().map(Vec::as_slice));
        &x_heap
    };

    let strip_bytes = weight.strip_bytes();
    let data: &[u8] = &weight.data;

    let compute_strip = |strip_idx: usize,
                         flat_chunk: &mut [f32],
                         out_refs: &mut [&mut [f32; TILE_DIM]]|
     -> Result<()> {
        let row_tiles = &data[strip_idx * strip_bytes..(strip_idx + 1) * strip_bytes];
        for slot in out_refs.iter_mut() {
            **slot = [0.0_f32; TILE_DIM];
        }
        dispatch_strip_matmul_tokens(row_tiles, weight.bits, x_refs, out_refs, backend)?;
        for (dst, src) in flat_chunk
            .as_chunks_mut::<TILE_DIM>()
            .0
            .iter_mut()
            .zip(out_refs.iter())
        {
            dst.copy_from_slice(*src);
        }
        Ok(())
    };

    let run_strips = |base_strip: usize, chunk: &mut [f32]| -> Result<()> {
        if token_count <= MAX_TILE_TOKENS {
            let mut local_outs = [[0.0_f32; TILE_DIM]; MAX_TILE_TOKENS];
            let mut out_refs = local_outs.each_mut();
            let out_slice = &mut out_refs[..token_count];
            for (local_idx, strip_chunk) in chunk.chunks_exact_mut(strip_stride).enumerate() {
                compute_strip(base_strip + local_idx, strip_chunk, out_slice)?;
            }
        } else {
            let mut local_outs = vec![[0.0_f32; TILE_DIM]; token_count];
            let mut out_refs: Vec<&mut [f32; TILE_DIM]> = local_outs.iter_mut().collect();
            for (local_idx, strip_chunk) in chunk.chunks_exact_mut(strip_stride).enumerate() {
                compute_strip(base_strip + local_idx, strip_chunk, &mut out_refs)?;
            }
        }
        Ok(())
    };

    if parallelism.workers() > 1 && strip_count >= 2 {
        parallelism.update_chunks(&mut scratch.multi_yb_flat, strip_stride, run_strips)?;
    } else {
        run_strips(0, &mut scratch.multi_yb_flat)?;
    }

    let yb_flat: &[f32] = &scratch.multi_yb_flat;
    if parallel_tokens {
        parallelism.update_triple_chunks(
            &mut scratch.multi_yb_token[..token_count],
            1,
            outputs,
            1,
            &mut scratch.tmp[..needed_tmp],
            per_token_tmp_len,
            |base_token, yb_chunk, out_chunk, tmp_chunk| {
                for (local_idx, ((yb, output), tmp)) in yb_chunk
                    .iter_mut()
                    .zip(out_chunk.iter_mut())
                    .zip(tmp_chunk.chunks_exact_mut(per_token_tmp_len))
                    .enumerate()
                {
                    finish_trellis_token_output(
                        weight,
                        yb_flat,
                        strip_stride,
                        base_token + local_idx,
                        yb,
                        output,
                        tmp,
                    )?;
                }
                Ok(())
            },
        )?;
    } else {
        for (token_idx, output) in outputs.iter_mut().enumerate() {
            finish_trellis_token_output(
                weight,
                yb_flat,
                strip_stride,
                token_idx,
                &mut scratch.multi_yb_token[token_idx],
                output,
                &mut scratch.tmp[..per_token_tmp_len],
            )?;
        }
    }

    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::Arc;

    use crate::Result;
    use crate::model::{Tensor, TensorBytes};

    use super::{
        MAX_TILE_TOKENS, MatvecBackend, RowParallelism, SideTransform, TILE_DIM, TrellisDesc,
        TrellisScratch, TrellisWeight, apply_side_transform, decode_tile_single_value, fwht_scaled,
        fwht_scaled_scalar, hadfull_slice, inv_sqrt_f32, paley_cross, paley_cross_scalar, rht,
        trellis_get_row, trellis_matmul_tokens_into, trellis_matvec_into,
    };

    fn make_prng_bytes(len: usize, seed: u64) -> Vec<u8> {
        let mut s = seed | 1;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            out.push(u8::try_from((s >> 24) & 0xff).unwrap_or(0));
        }
        out
    }

    fn make_f32_tensor(name: &str, values: &[f32]) -> Tensor {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for &v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Tensor {
            name: name.to_owned(),
            dimensions: vec![values.len()],
            kind: super::F32,
            data: TensorBytes::Shared(Arc::from(bytes)),
        }
    }

    fn make_f16_ones_tensor(name: &str, len: usize) -> Tensor {
        let mut bytes = Vec::with_capacity(len * 2);
        for i in 0..len {
            let bits: u16 = match i % 3 {
                0 => 0x3c00, // 1.0
                1 => 0x3800, // 0.5
                _ => 0xbc00, // -1.0
            };
            bytes.extend_from_slice(&bits.to_le_bytes());
        }
        Tensor {
            name: name.to_owned(),
            dimensions: vec![len],
            kind: super::F16,
            data: TensorBytes::Shared(Arc::from(bytes)),
        }
    }

    fn make_i32_tensor(name: &str, values: &[i32]) -> Tensor {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for &v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Tensor {
            name: name.to_owned(),
            dimensions: vec![values.len()],
            kind: super::I32,
            data: TensorBytes::Shared(Arc::from(bytes)),
        }
    }

    /// Independently builds the full `N × K` float weight matrix matching `ggml_trellis_dequantize`.
    pub(crate) fn dequantize_reference(weight: &TrellisWeight) -> Result<Vec<f32>> {
        let k = weight.k;
        let n = weight.n;
        let mut out = vec![0.0_f32; n * k];
        let tile_cols = k / TILE_DIM;
        let tile_rows = n / TILE_DIM;
        let tile_bytes = 32 * weight.bits;
        let data: &[u8] = &weight.data;
        for j in 0..tile_rows {
            for i in 0..tile_cols {
                let tile =
                    &data[(j * tile_cols + i) * tile_bytes..(j * tile_cols + i + 1) * tile_bytes];
                let s16 = weight.s16[i];
                for r in 0..TILE_DIM {
                    for c in 0..TILE_DIM {
                        out[(j * TILE_DIM + r) * k + i * TILE_DIM + c] =
                            decode_tile_single_value(tile, weight.bits, r, c) * s16;
                    }
                }
            }
        }
        let mut tmp = vec![0.0_f32; k.max(n) * 2];
        let mut row_tmp = vec![0.0_f32; k];
        for row_idx in 0..n {
            let w_row = &mut out[row_idx * k..(row_idx + 1) * k];
            apply_side_transform(w_row, &weight.desc.cols, true, None, None, &mut tmp)?;
            for (slot, &pre_v) in w_row.iter_mut().zip(weight.pre.iter()) {
                *slot *= pre_v;
            }
            if let Some(pin) = &weight.pin {
                row_tmp.fill(0.0);
                for (j, &dst) in pin.iter().enumerate() {
                    row_tmp[usize::try_from(dst)?] += w_row[j];
                }
                w_row.copy_from_slice(&row_tmp);
            }
        }

        let mut col = vec![0.0_f32; n];
        for c_idx in 0..k {
            for row_idx in 0..n {
                col[row_idx] = out[row_idx * k + c_idx];
            }
            apply_side_transform(
                &mut col,
                &weight.desc.rows,
                true,
                None,
                Some(&weight.post),
                &mut tmp,
            )?;
            for row_idx in 0..n {
                out[row_idx * k + c_idx] = col[row_idx];
            }
        }

        if let Some(pout) = &weight.pout {
            let mut permuted = vec![0.0_f32; n * k];
            for (dst_row, &src_row) in pout.iter().enumerate() {
                let src = usize::try_from(src_row)?;
                permuted[dst_row * k..(dst_row + 1) * k]
                    .copy_from_slice(&out[src * k..(src + 1) * k]);
            }
            out = permuted;
        }
        Ok(out)
    }

    fn build_synthetic_weight(
        bits: usize,
        k: usize,
        n: usize,
        desc_raw: &[i32],
        with_perms: bool,
        seed: u64,
    ) -> Result<TrellisWeight> {
        let type_id = u32::try_from(43 + bits)?;
        let byte_len = (k / 16) * (n / 16) * 32 * bits;
        let w_tensor = Tensor {
            name: format!("test_trl{bits}.weight"),
            dimensions: vec![k, n],
            kind: type_id,
            data: TensorBytes::Shared(Arc::from(make_prng_bytes(byte_len, seed))),
        };
        let s16_tensor = make_f16_ones_tensor("test.trellis_s16", k / 16);
        let pre_vals: Vec<f32> = (0..k)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.25 })
            .collect();
        let post_vals: Vec<f32> = (0..n)
            .map(|i| if i % 3 == 0 { 0.75 } else { -0.5 })
            .collect();
        let pre_tensor = make_f32_tensor("test.trellis_pre", &pre_vals);
        let post_tensor = make_f32_tensor("test.trellis_post", &post_vals);

        let pin_vals: Vec<i32> = (0..k)
            .map(|i| i32::try_from((i * 5 + 3) % k).unwrap_or(0))
            .collect();
        let perm_out_vals: Vec<i32> = (0..n)
            .map(|i| i32::try_from((n - 1) - i).unwrap_or(0))
            .collect();
        // Ensure `pin` is a valid permutation only when gcd(5, k) == 1 (k powers of 2 / multiples not div by 5)
        let pin_perm: Vec<i32> = if k.is_multiple_of(5) {
            (0..k)
                .rev()
                .map(|i| i32::try_from(i).unwrap_or(0))
                .collect()
        } else {
            pin_vals
        };
        let pin_tensor = make_i32_tensor("test.trellis_pin", &pin_perm);
        let perm_out_tensor = make_i32_tensor("test.trellis_pout", &perm_out_vals);

        let desc = TrellisDesc::from_i32_slice(k, n, desc_raw)?;
        TrellisWeight::from_tensors(
            &w_tensor,
            &s16_tensor,
            &pre_tensor,
            &post_tensor,
            with_perms.then_some(&pin_tensor),
            with_perms.then_some(&perm_out_tensor),
            desc,
        )
    }

    #[test]
    fn trellis_matvec_tokens_and_get_row_match_dequantize_reference() -> Result<()> {
        // Test TRL1..=TRL6 across identity, block RHT, Paley hadfull (m=1, 12, 20, 68), fused parts, and shards.
        let cases: &[(usize, usize, usize, &[i32], bool)] = &[
            (1, 32, 32, &[0, 0, 1, 0, 0, 1, 0], false),
            (2, 64, 48, &[1, 16, 1, 1, 32, 1, 0], true),
            (3, 64, 64, &[2, 1, 1, 2, 1, 1, 2, 16, 48], true),
            (4, 12 * 16, 20 * 16, &[2, 1, 2, 2, 1, 2, 2, 160, 160], false),
            (5, 68 * 16, 48, &[2, 1, 1, 2, 1, 1, 0], true),
            (6, 32, 12 * 16, &[2, 1, 1, 1, 16, 1, 0], false),
        ];
        let parallelisms = std::iter::once(Ok(RowParallelism::default()))
            .chain(cfg!(feature = "parallel-rows").then(|| RowParallelism::new(4)))
            .collect::<Result<Vec<_>>>()?;
        for (case_idx, &(bits, k, n, desc_raw, perms)) in cases.iter().enumerate() {
            let weight = build_synthetic_weight(
                bits,
                k,
                n,
                desc_raw,
                perms,
                0x1234_5678_u64 + u64::try_from(case_idx)?,
            )?;
            let w_dense = dequantize_reference(&weight)?;
            let inputs: Vec<Vec<f32>> = (0..5)
                .map(|t| {
                    (0..k)
                        .map(|j| {
                            f32::from(i16::try_from((j * 11 + t * 7) % 19).unwrap_or(0) - 9) * 0.25
                        })
                        .collect()
                })
                .collect();

            let mut expected_outputs = Vec::with_capacity(inputs.len());
            for input in &inputs {
                let mut expected = vec![0.0_f32; n];
                for r in 0..n {
                    let mut sum = 0.0_f32;
                    for c in 0..k {
                        sum = w_dense[r * k + c].mul_add(input[c], sum);
                    }
                    expected[r] = sum;
                }
                expected_outputs.push(expected);
            }

            let mut scratch = TrellisScratch::default();
            for backend in [MatvecBackend::Scalar, MatvecBackend::Auto] {
                for par in &parallelisms {
                    let mut actual = Vec::new();
                    trellis_matvec_into(
                        &weight,
                        &inputs[0],
                        &mut actual,
                        backend,
                        par,
                        &mut scratch,
                    )?;
                    for (r, (&a, &e)) in actual.iter().zip(&expected_outputs[0]).enumerate() {
                        assert!(
                            (a - e).abs() <= 2e-2,
                            "case={case_idx} bits={bits} row={r}: actual={a} vs expected={e}"
                        );
                    }

                    let mut batch_outs = vec![Vec::new(); inputs.len()];
                    trellis_matmul_tokens_into(
                        &weight,
                        &inputs,
                        &mut batch_outs,
                        backend,
                        par,
                        &mut scratch,
                    )?;
                    for (t, (actual_t, exp_t)) in
                        batch_outs.iter().zip(&expected_outputs).enumerate()
                    {
                        for (r, (&a, &e)) in actual_t.iter().zip(exp_t).enumerate() {
                            assert!(
                                (a - e).abs() <= 2e-2,
                                "batch case={case_idx} token={t} row={r}: actual={a} vs expected={e}"
                            );
                        }
                    }
                }
            }

            if weight.desc.rows == SideTransform::Identity {
                let mut row_out = Vec::new();
                let mut row_tmp = Vec::new();
                for probe_row in [0_usize, 15, 16, n - 1] {
                    trellis_get_row(&weight, probe_row, &mut row_out, &mut row_tmp)?;
                    for (c, (&a, &e)) in row_out
                        .iter()
                        .zip(&w_dense[probe_row * k..(probe_row + 1) * k])
                        .enumerate()
                    {
                        assert!(
                            (a - e).abs() <= 1e-4,
                            "get_row case={case_idx} row={probe_row} col={c}: actual={a} vs expected={e}"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn trellis_constructors_and_helpers_reject_malformed_permutations_descriptors_and_scales()
    -> Result<()> {
        assert!(TrellisDesc::from_i32_slice(16, 16, &[0, 0, 1, 0, 0, 1]).is_err());
        assert!(TrellisDesc::from_i32_slice(16, 16, &[9, 0, 1, 0, 0, 1, 0]).is_err());
        assert!(TrellisDesc::from_i32_slice(16, 32, &[2, 1, 1, 0, 0, 1, 2, 16, 32]).is_err());
        assert!(TrellisDesc::from_i32_slice(16, 16, &[1, 3, 1, 0, 0, 1, 0]).is_err());
        assert!(rht(&mut [1.0, 2.0, 3.0], 2).is_err());

        let base = build_synthetic_weight(2, 16, 16, &[0, 0, 1, 0, 0, 1, 0], false, 7)?;
        let dup_pin = make_i32_tensor("bad.trellis_pin", &[0_i32; 16]);
        let neg_pin = make_i32_tensor(
            "bad.trellis_pin",
            &[-1_i32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        );
        let oob_pout = make_i32_tensor(
            "bad.trellis_pout",
            &[0_i32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16],
        );
        let s16_tensor = make_f16_ones_tensor("t.trellis_s16", 1);
        let pre_tensor = make_f32_tensor("t.trellis_pre", &[1.0; 16]);
        let post_tensor = make_f32_tensor("t.trellis_post", &[1.0; 16]);
        let nan_pre = make_f32_tensor("t.trellis_pre", &{
            let mut v = vec![1.0_f32; 16];
            v[3] = f32::NAN;
            v
        });
        let w_tensor = Tensor {
            name: "t.weight".to_owned(),
            dimensions: vec![16, 16],
            kind: 45,
            data: base.data.clone(),
        };

        assert!(
            TrellisWeight::from_tensors(
                &w_tensor,
                &s16_tensor,
                &pre_tensor,
                &post_tensor,
                Some(&dup_pin),
                None,
                base.desc.clone(),
            )
            .is_err()
        );
        assert!(
            TrellisWeight::from_tensors(
                &w_tensor,
                &s16_tensor,
                &pre_tensor,
                &post_tensor,
                Some(&neg_pin),
                None,
                base.desc.clone(),
            )
            .is_err()
        );
        assert!(
            TrellisWeight::from_tensors(
                &w_tensor,
                &s16_tensor,
                &pre_tensor,
                &post_tensor,
                None,
                Some(&oob_pout),
                base.desc.clone(),
            )
            .is_err()
        );
        assert!(
            TrellisWeight::from_tensors(
                &w_tensor,
                &s16_tensor,
                &nan_pre,
                &post_tensor,
                None,
                None,
                base.desc.clone(),
            )
            .is_err()
        );

        let rht_row_weight = build_synthetic_weight(2, 16, 16, &[1, 16, 1, 0, 0, 1, 0], false, 9)?;
        let mut row_out = Vec::new();
        let mut row_tmp = Vec::new();
        assert!(trellis_get_row(&rht_row_weight, 0, &mut row_out, &mut row_tmp).is_err());

        let mut bad_input = vec![0.0_f32; 16];
        bad_input[4] = f32::INFINITY;
        let mut out = Vec::new();
        let mut scratch = TrellisScratch::default();
        assert!(
            trellis_matvec_into(
                &base,
                &bad_input,
                &mut out,
                MatvecBackend::Auto,
                &RowParallelism::default(),
                &mut scratch,
            )
            .is_err()
        );

        Ok(())
    }

    fn make_pattern_vec(len: usize, seed: usize) -> Vec<f32> {
        (0..len)
            .map(|idx| match (idx + seed) % 7 {
                0 => -0.0_f32,
                1 => 0.0_f32,
                _ => {
                    let raw = i16::try_from((idx * 17 + seed * 13) % 41).unwrap_or(0) - 20;
                    f32::from(raw) * 0.0625
                }
            })
            .collect()
    }

    #[test]
    fn fwht_scaled_and_paley_cross_neon_match_scalar_zero_ulp() -> Result<()> {
        for (idx, block) in [1_usize, 2, 4, 8, 16, 32, 64, 128, 256]
            .into_iter()
            .enumerate()
        {
            for repeat in [1_usize, 2, 3] {
                let len = block * repeat;
                let factor = inv_sqrt_f32(block)?;
                let mut scalar_buf = make_pattern_vec(len, idx + repeat);
                let mut neon_buf = scalar_buf.clone();
                fwht_scaled_scalar(&mut scalar_buf, block, factor);
                fwht_scaled(&mut neon_buf, block, factor);
                assert_eq!(
                    neon_buf
                        .iter()
                        .copied()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>(),
                    scalar_buf
                        .iter()
                        .copied()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>(),
                    "fwht_scaled 0-ULP mismatch for block={block}, repeat={repeat}"
                );
            }
        }

        for m in [12_usize, 20, 68] {
            let factor = inv_sqrt_f32(m)?;
            for run in [1_usize, 2, 3, 4, 5, 7, 8, 11, 12, 16, 20, 32] {
                for transpose in [false, true] {
                    let input = make_pattern_vec(m * run, m + run + usize::from(transpose));
                    let mut scalar_out = vec![0.0_f32; m * run];
                    let mut neon_out = vec![0.0_f32; m * run];
                    paley_cross_scalar(&input, &mut scalar_out, m, run, transpose, factor)?;
                    paley_cross(&input, &mut neon_out, m, run, transpose, factor)?;
                    assert_eq!(
                        neon_out
                            .iter()
                            .copied()
                            .map(f32::to_bits)
                            .collect::<Vec<_>>(),
                        scalar_out
                            .iter()
                            .copied()
                            .map(f32::to_bits)
                            .collect::<Vec<_>>(),
                        "paley_cross 0-ULP mismatch for m={m}, run={run}, transpose={transpose}"
                    );
                }
            }
        }

        // Cover full hadfull_slice across m in {1, 12, 20, 68} including small power-of-two runs.
        for &(m, run) in &[
            (1_usize, 16_usize),
            (1, 64),
            (12, 1),
            (12, 2),
            (12, 4),
            (12, 16),
            (20, 1),
            (20, 2),
            (20, 8),
            (68, 1),
            (68, 4),
            (68, 16),
        ] {
            let len = m * run;
            let pre = make_pattern_vec(len, m + 3);
            let post = make_pattern_vec(len, run + 5);
            let mut tmp = vec![0.0_f32; len * 2];
            for inverse in [false, true] {
                let mut values = make_pattern_vec(len, m * 3 + run);
                let orig = values.clone();
                hadfull_slice(
                    &mut values,
                    None,
                    1,
                    0,
                    inverse,
                    Some(&pre),
                    Some(&post),
                    &mut tmp,
                )?;
                assert_ne!(values, orig);
                assert!(values.iter().all(|v| v.is_finite()));
            }
        }

        Ok(())
    }

    #[test]
    fn trellis_matmul_tokens_parallel_matches_serial() -> Result<()> {
        let cases: &[(usize, usize, usize, &[i32], bool)] = &[
            (2, 64, 48, &[1, 16, 1, 1, 32, 1, 0], true),
            (3, 12 * 16, 20 * 16, &[2, 1, 2, 2, 1, 2, 2, 160, 160], false),
            (4, 68 * 16, 64, &[2, 1, 1, 2, 1, 1, 0], true),
        ];
        let serial = RowParallelism::default();
        let parallel = if cfg!(feature = "parallel-rows") {
            Some(RowParallelism::new(4)?)
        } else {
            None
        };

        for (case_idx, &(bits, k, n, desc_raw, perms)) in cases.iter().enumerate() {
            let weight = build_synthetic_weight(
                bits,
                k,
                n,
                desc_raw,
                perms,
                0xdead_beef_u64 + u64::try_from(case_idx)?,
            )?;
            // Test small batch, exact MAX_TILE_TOKENS boundary, and > MAX_TILE_TOKENS fallback.
            for token_count in [2_usize, 5, MAX_TILE_TOKENS, MAX_TILE_TOKENS + 3] {
                let inputs: Vec<Vec<f32>> = (0..token_count)
                    .map(|t| make_pattern_vec(k, case_idx * 100 + t))
                    .collect();

                for backend in [MatvecBackend::Scalar, MatvecBackend::Auto] {
                    let mut serial_scratch = TrellisScratch::default();
                    let mut serial_outs = vec![Vec::new(); token_count];
                    trellis_matmul_tokens_into(
                        &weight,
                        &inputs,
                        &mut serial_outs,
                        backend,
                        &serial,
                        &mut serial_scratch,
                    )?;

                    if let Some(par) = &parallel {
                        let mut par_scratch = TrellisScratch::default();
                        let mut par_outs = vec![Vec::new(); token_count];
                        trellis_matmul_tokens_into(
                            &weight,
                            &inputs,
                            &mut par_outs,
                            backend,
                            par,
                            &mut par_scratch,
                        )?;
                        for (t, (p, s)) in par_outs.iter().zip(&serial_outs).enumerate() {
                            assert_eq!(
                                p.iter().copied().map(f32::to_bits).collect::<Vec<_>>(),
                                s.iter().copied().map(f32::to_bits).collect::<Vec<_>>(),
                                "parallel mismatch case={case_idx} tokens={token_count} token={t}"
                            );
                        }
                    }
                }
            }
        }

        Ok(())
    }

    #[test]
    fn trellis_matmul_tokens_parallel_preserves_error_semantics() -> Result<()> {
        if !cfg!(feature = "parallel-rows") {
            return Ok(());
        }
        let serial = RowParallelism::default();
        let par = RowParallelism::new(4)?;
        let k = 64_usize;
        let n = 48_usize;
        let weight = build_synthetic_weight(2, k, n, &[1, 16, 1, 1, 32, 1, 0], true, 0xfeed_face)?;

        let mut bad_inputs: Vec<Vec<f32>> = (0..6).map(|t| make_pattern_vec(k, t + 11)).collect();
        bad_inputs[4][k / 2] = f32::NAN;
        let mut serial_scratch = TrellisScratch::default();
        let mut par_scratch = TrellisScratch::default();
        let mut serial_outs = vec![Vec::new(); bad_inputs.len()];
        let mut par_outs = vec![Vec::new(); bad_inputs.len()];
        let serial_err = trellis_matmul_tokens_into(
            &weight,
            &bad_inputs,
            &mut serial_outs,
            MatvecBackend::Auto,
            &serial,
            &mut serial_scratch,
        );
        let par_err = trellis_matmul_tokens_into(
            &weight,
            &bad_inputs,
            &mut par_outs,
            MatvecBackend::Auto,
            &par,
            &mut par_scratch,
        );
        assert_eq!(
            par_err.as_ref().map_err(ToString::to_string),
            serial_err.as_ref().map_err(ToString::to_string),
        );
        assert!(par_err.is_err());

        let mut overflow_inputs: Vec<Vec<f32>> =
            (0..6).map(|t| make_pattern_vec(k, t + 19)).collect();
        overflow_inputs[3].fill(f32::MAX);
        let serial_ov = trellis_matmul_tokens_into(
            &weight,
            &overflow_inputs,
            &mut serial_outs,
            MatvecBackend::Auto,
            &serial,
            &mut serial_scratch,
        );
        let par_ov = trellis_matmul_tokens_into(
            &weight,
            &overflow_inputs,
            &mut par_outs,
            MatvecBackend::Auto,
            &par,
            &mut par_scratch,
        );
        assert_eq!(
            par_ov.as_ref().map_err(ToString::to_string),
            serial_ov.as_ref().map_err(ToString::to_string),
        );
        assert!(par_ov.is_err());

        Ok(())
    }
}
