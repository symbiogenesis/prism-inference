//! Experimental baseline NEON integer arithmetic for native PQ2/PTQ1 blocks.

use fearless_simd::{Neon, f32x4, i8x16, i16x8, i32x4, prelude::*, u8x16, u16x8, u32x4, u64x2};

use crate::Result;

#[inline]
fn decode(packed: u8x16<Neon>) -> [i8x16<Neon>; 4] {
    let a = packed & 3;
    let b = (packed >> 2) & 3;
    let c = (packed >> 4) & 3;
    let d = packed >> 6;
    // Interleave the four digits of each byte into consecutive activation
    // order. Pairing bytes as u16 lanes keeps each adjacent pair together.
    let ab_low: u16x8<Neon> = a.zip_low(b).bitcast();
    let ab_high: u16x8<Neon> = a.zip_high(b).bitcast();
    let cd_low: u16x8<Neon> = c.zip_low(d).bitcast();
    let cd_high: u16x8<Neon> = c.zip_high(d).bitcast();
    [
        ab_low.zip_low(cd_low).bitcast::<i8x16<Neon>>() - 1,
        ab_low.zip_high(cd_low).bitcast::<i8x16<Neon>>() - 1,
        ab_high.zip_low(cd_high).bitcast::<i8x16<Neon>>() - 1,
        ab_high.zip_high(cd_high).bitcast::<i8x16<Neon>>() - 1,
    ]
}

#[inline]
fn pq2_pair_sums(weights: i8x16<Neon>, activations: i8x16<Neon>) -> i16x8<Neon> {
    let (weight_low, weight_high) = weights.widen();
    let (activation_low, activation_high) = activations.widen();
    // PQ2 weights are -1..=2 and activations are -128..=127. Each signed
    // product is -256..=254, so each pair is -512..=508 in i16. Only the
    // PQ2 block dots below use this helper; native PTQ1 lane sums stay separate.
    weight_low * activation_low + weight_high * activation_high
}

#[inline]
fn pq2_reduce(pairs: i16x8<Neon>) -> i32 {
    // A 128-weight block contributes eight pairs to each lane: -4096..=4064.
    // Widen before adding low/high lanes or horizontally reducing. The full
    // i32 dot is -32768..=32512; no narrow horizontal sum is performed.
    let (low, high) = pairs.widen();
    (low + high).reduce_sum()
}

#[inline]
fn scalar(packed: &[u8; 32], activations: &[i8; 128]) -> i32 {
    activations
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let weight = i32::from((packed[index / 4] >> (2 * (index % 4))) & 3) - 1;
            weight * i32::from(*value)
        })
        .sum()
}

/// Computes the exact integer dot of 128 consecutive native PQ2 symbols.
///
/// Codes 0..=3 decode to -1..=2, including the native fourth symbol. Arrays
/// establish the complete extents; safe slice loads require no alignment.
/// The absolute sum is at most 128 * 2 * 128 = 32,768, so all i32 partial
/// sums and the horizontal reduction fit, including activation -128.
#[inline]
pub(super) fn dot(packed: &[u8; 32], activations: &[i8; 128]) -> i32 {
    // This statically selected baseline token performs no runtime detection.
    // The Option branch disappears on the qualified baseline target; retain
    // defined scalar behavior if the dependency cannot provide the token.
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return scalar(packed, activations);
    };
    let mut even = i16x8::splat(neon, 0);
    let mut odd = i16x8::splat(neon, 0);
    for (bytes, values) in packed
        .as_chunks::<16>()
        .0
        .iter()
        .zip(activations.as_chunks::<64>().0)
    {
        let [a, b, c, d] = decode(u8x16::from_slice(neon, bytes));
        let planes = values.as_chunks::<16>().0;
        even += pq2_pair_sums(a, i8x16::from_slice(neon, &planes[0]));
        odd += pq2_pair_sums(b, i8x16::from_slice(neon, &planes[1]));
        even += pq2_pair_sums(c, i8x16::from_slice(neon, &planes[2]));
        odd += pq2_pair_sums(d, i8x16::from_slice(neon, &planes[3]));
    }
    pq2_reduce(even + odd)
}

/// Computes the exact 256-element integer dot product of a 64-byte `TQ2_0`
/// payload against a 256-element `Q8_K` activation slice using 128-bit NEON
/// plane extraction.
#[inline]
pub(super) fn dot_tq2_block(packed: &[u8; 64], activations: &[i8; 256]) -> i32 {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return (0..256)
            .map(|index| {
                let weight = i32::from(
                    (packed[index / 128 * 32 + index % 32] >> (2 * (index % 128 / 32))) & 3,
                ) - 1;
                weight * i32::from(activations[index])
            })
            .sum();
    };
    let mut total = 0_i32;
    for (half_bytes, half_acts) in packed
        .as_chunks::<32>()
        .0
        .iter()
        .zip(activations.as_chunks::<128>().0)
    {
        let mut acc0 = i16x8::splat(neon, 0);
        let mut acc1 = i16x8::splat(neon, 0);
        let planes = half_acts.as_chunks::<32>().0;
        for chunk in 0..2 {
            let offset = chunk * 16;
            let w = u8x16::from_slice(neon, &half_bytes[offset..offset + 16]);
            let p0 = (w & 3).bitcast::<i8x16<Neon>>() - 1;
            let p1 = ((w >> 2) & 3).bitcast::<i8x16<Neon>>() - 1;
            let p2 = ((w >> 4) & 3).bitcast::<i8x16<Neon>>() - 1;
            let p3 = (w >> 6).bitcast::<i8x16<Neon>>() - 1;
            let a0 = i8x16::from_slice(neon, &planes[0][offset..offset + 16]);
            let a1 = i8x16::from_slice(neon, &planes[1][offset..offset + 16]);
            let a2 = i8x16::from_slice(neon, &planes[2][offset..offset + 16]);
            let a3 = i8x16::from_slice(neon, &planes[3][offset..offset + 16]);
            acc0 += pq2_pair_sums(p0, a0) + pq2_pair_sums(p2, a2);
            acc1 += pq2_pair_sums(p1, a1) + pq2_pair_sums(p3, a3);
        }
        total += pq2_reduce(acc0 + acc1);
    }
    total
}

#[inline]
fn dot_q8_0_block_preloaded(
    neon: Neon,
    weights: &[u8; 32],
    a0_l: i16x8<Neon>,
    a0_h: i16x8<Neon>,
    a1_l: i16x8<Neon>,
    a1_h: i16x8<Neon>,
) -> i32 {
    let (w0_l, w0_h) = u8x16::from_slice(neon, &weights[..16])
        .bitcast::<i8x16<Neon>>()
        .widen();
    let (w1_l, w1_h) = u8x16::from_slice(neon, &weights[16..])
        .bitcast::<i8x16<Neon>>()
        .widen();
    let (p00_l, p00_h) = (w0_l * a0_l).widen();
    let (p01_l, p01_h) = (w0_h * a0_h).widen();
    let (p10_l, p10_h) = (w1_l * a1_l).widen();
    let (p11_l, p11_h) = (w1_h * a1_h).widen();
    ((p00_l + p00_h) + (p01_l + p01_h) + (p10_l + p10_h) + (p11_l + p11_h)).reduce_sum()
}

/// Computes the exact 32-element integer dot product of a `Q8_0` weight block
/// against a 32-element `Q8_0` activation slice.
#[inline]
pub(super) fn dot_q8_0_block(weights: &[u8; 32], activations: &[i8; 32]) -> i32 {
    if bonsai_neon_dot::has_dotprod() {
        return bonsai_neon_dot::dot_q8_0_block(weights, activations);
    }
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return weights
            .iter()
            .zip(activations)
            .map(|(&w, &a)| i32::from(i8::from_ne_bytes([w])) * i32::from(a))
            .sum();
    };
    let (a0_l, a0_h) = i8x16::from_slice(neon, &activations[..16]).widen();
    let (a1_l, a1_h) = i8x16::from_slice(neon, &activations[16..]).widen();
    dot_q8_0_block_preloaded(neon, weights, a0_l, a0_h, a1_l, a1_h)
}

#[inline]
fn dot_q4_0_block_preloaded(
    neon: Neon,
    packed: &[u8; 16],
    a0_l: i16x8<Neon>,
    a0_h: i16x8<Neon>,
    a1_l: i16x8<Neon>,
    a1_h: i16x8<Neon>,
) -> i32 {
    let raw = u8x16::from_slice(neon, packed);
    let (w0_l, w0_h) = ((raw & 15).bitcast::<i8x16<Neon>>() - 8).widen();
    let (w1_l, w1_h) = ((raw >> 4).bitcast::<i8x16<Neon>>() - 8).widen();
    let pairs = (w0_l * a0_l + w0_h * a0_h) + (w1_l * a1_l + w1_h * a1_h);
    pq2_reduce(pairs)
}

/// Computes the exact 32-element integer dot product of a 16-byte `Q4_0` nibble
/// payload against a 32-element `Q8_0` activation slice.
#[inline]
pub(super) fn dot_q4_0_block(packed: &[u8; 16], activations: &[i8; 32]) -> i32 {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return (0..32)
            .map(|index| {
                let w = i32::from((packed[index % 16] >> (4 * (index / 16))) & 15) - 8;
                w * i32::from(activations[index])
            })
            .sum();
    };
    let (a0_l, a0_h) = i8x16::from_slice(neon, &activations[..16]).widen();
    let (a1_l, a1_h) = i8x16::from_slice(neon, &activations[16..]).widen();
    dot_q4_0_block_preloaded(neon, packed, a0_l, a0_h, a1_l, a1_h)
}

/// Computes the exact 32-element integer dot product of a 16-byte `Q4_1` nibble
/// payload against a 32-element `Q8_1` activation slice.
#[inline]
pub(super) fn dot_q4_1_block(packed: &[u8; 16], activations: &[i8; 32]) -> i32 {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return (0..32)
            .map(|index| {
                let w = i32::from((packed[index % 16] >> (4 * (index / 16))) & 15);
                w * i32::from(activations[index])
            })
            .sum();
    };
    let raw = u8x16::from_slice(neon, packed);
    let (w0_l, w0_h) = (raw & 15).bitcast::<i8x16<Neon>>().widen();
    let (w1_l, w1_h) = (raw >> 4).bitcast::<i8x16<Neon>>().widen();
    let (a0_l, a0_h) = i8x16::from_slice(neon, &activations[..16]).widen();
    let (a1_l, a1_h) = i8x16::from_slice(neon, &activations[16..]).widen();
    let pairs = (w0_l * a0_l + w0_h * a0_h) + (w1_l * a1_l + w1_h * a1_h);
    pq2_reduce(pairs)
}

/// Computes the exact 32-element integer dot product of a decoded 5-bit `i16`
/// weight block against a 32-element `Q8` activation slice.
#[inline]
pub(super) fn dot_i16_block(weights: &[i16; 32], activations: &[i8; 32]) -> i32 {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return weights
            .iter()
            .zip(activations)
            .map(|(&w, &a)| i32::from(w) * i32::from(a))
            .sum();
    };
    let (a0_l, a0_h) = i8x16::from_slice(neon, &activations[..16]).widen();
    let (a1_l, a1_h) = i8x16::from_slice(neon, &activations[16..]).widen();
    let (p0_l, p0_h) = (i16x8::from_slice(neon, &weights[..8]) * a0_l
        + i16x8::from_slice(neon, &weights[8..16]) * a0_h)
        .widen();
    let (p1_l, p1_h) = (i16x8::from_slice(neon, &weights[16..24]) * a1_l
        + i16x8::from_slice(neon, &weights[24..]) * a1_h)
        .widen();
    ((p0_l + p0_h) + (p1_l + p1_h)).reduce_sum()
}

#[inline]
#[allow(
    clippy::too_many_arguments,
    reason = "Passes the eight pre-widened activation planes directly in NEON registers."
)]
fn pq2_half_unsigned_sum(
    w: u8x16<Neon>,
    p0_low: i16x8<Neon>,
    p0_high: i16x8<Neon>,
    p1_low: i16x8<Neon>,
    p1_high: i16x8<Neon>,
    p2_low: i16x8<Neon>,
    p2_high: i16x8<Neon>,
    p3_low: i16x8<Neon>,
    p3_high: i16x8<Neon>,
) -> i16x8<Neon> {
    // Widen the sixteen packed bytes once to u16x8 (0..=255), then extract all
    // four unsigned two-bit planes in 16-bit registers. Across both halves
    // (16 products per lane), each i16 lane is bounded by -6144..=6096.
    let (u_low, u_high) = w.widen();
    let w_low = u_low.bitcast::<i16x8<Neon>>();
    let w_high = u_high.bitcast::<i16x8<Neon>>();
    let s0 = (w_low & 3) * p0_low + (w_high & 3) * p0_high;
    let s1 = ((w_low >> 2) & 3) * p1_low + ((w_high >> 2) & 3) * p1_high;
    let s2 = ((w_low >> 4) & 3) * p2_low + ((w_high >> 4) & 3) * p2_high;
    let s3 = (w_low >> 6) * p3_low + (w_high >> 6) * p3_high;
    (s0 + s1) + (s2 + s3)
}

#[inline]
fn reduce4_i32x4(
    v0: i32x4<Neon>,
    v1: i32x4<Neon>,
    v2: i32x4<Neon>,
    v3: i32x4<Neon>,
) -> i32x4<Neon> {
    let p01 = v0.unzip_low(v1) + v0.unzip_high(v1);
    let p23 = v2.unzip_low(v3) + v2.unzip_high(v3);
    p01.unzip_low(p23) + p01.unzip_high(p23)
}

#[inline]
fn reduce4_i16x8_to_i32x4(
    s0: i16x8<Neon>,
    s1: i16x8<Neon>,
    s2: i16x8<Neon>,
    s3: i16x8<Neon>,
) -> i32x4<Neon> {
    let (l0, h0) = s0.widen();
    let (l1, h1) = s1.widen();
    let (l2, h2) = s2.widen();
    let (l3, h3) = s3.widen();
    reduce4_i32x4(l0 + h0, l1 + h1, l2 + h2, l3 + h3)
}

#[inline]
pub(super) fn prepare_pq2_activations(source: &[i8; 128]) -> ([i8; 128], [i16; 128], i32) {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        let values = std::array::from_fn(|index| {
            source[(index / 64) * 64 + (index % 16) * 4 + (index / 16) % 4]
        });
        let sum = source.iter().map(|&v| i32::from(v)).sum();
        return (values, values.map(i16::from), sum);
    };
    let use_dotprod = bonsai_neon_dot::has_dotprod();
    let mut output = [0_i8; 128];
    let mut widened = [0_i16; 128];
    let mut sum_acc = i16x8::splat(neon, 0);
    for ((src_half, dst_half), wide_half) in source
        .as_chunks::<64>()
        .0
        .iter()
        .zip(output.as_chunks_mut::<64>().0)
        .zip(widened.as_chunks_mut::<64>().0)
    {
        let chunks = src_half.as_chunks::<16>().0;
        let v0 = i8x16::from_slice(neon, &chunks[0]);
        let v1 = i8x16::from_slice(neon, &chunks[1]);
        let v2 = i8x16::from_slice(neon, &chunks[2]);
        let v3 = i8x16::from_slice(neon, &chunks[3]);
        let e01 = v0.unzip_low(v1);
        let o01 = v0.unzip_high(v1);
        let e23 = v2.unzip_low(v3);
        let o23 = v2.unzip_high(v3);
        let p0 = e01.unzip_low(e23);
        let p1 = o01.unzip_low(o23);
        let p2 = e01.unzip_high(e23);
        let p3 = o01.unzip_high(o23);
        let dst = dst_half.as_chunks_mut::<16>().0;
        p0.store_slice(&mut dst[0]);
        p1.store_slice(&mut dst[1]);
        p2.store_slice(&mut dst[2]);
        p3.store_slice(&mut dst[3]);
        let wide = wide_half.as_chunks_mut::<8>().0;
        for (plane, pair) in [p0, p1, p2, p3]
            .into_iter()
            .zip(wide.as_chunks_mut::<2>().0)
        {
            let (low, high) = plane.widen();
            sum_acc += low + high;
            if !use_dotprod {
                low.store_slice(&mut pair[0]);
                high.store_slice(&mut pair[1]);
            }
        }
    }
    (output, widened, pq2_reduce(sum_acc))
}

#[inline]
pub(super) fn f16_u32x4_to_f32_x4(neon: Neon, raw: u32x4<Neon>) -> f32x4<Neon> {
    let sign = (raw & 0x8000) << 16;
    let exp = (raw >> 10) & 31;
    let mant = raw & 1023;
    let normal_bits = sign | ((exp + 112) << 23) | (mant << 13);
    let inf_nan_bits = sign | (u32x4::splat(neon, 255) << 23) | (mant << 13);
    let subnormal_mag =
        mant.to_float::<f32x4<Neon>>() * f32x4::splat(neon, f32::from_bits(0x3380_0000));
    let subnormal_bits = sign | subnormal_mag.bitcast::<u32x4<Neon>>();
    let is_zero_exp = exp.simd_eq(u32x4::splat(neon, 0));
    let is_max_exp = exp.simd_eq(u32x4::splat(neon, 31));
    let selected = is_zero_exp.select(subnormal_bits, is_max_exp.select(inf_nan_bits, normal_bits));
    selected.bitcast::<f32x4<Neon>>()
}

#[inline]
pub(super) fn f16_bits_to_f32_x4(neon: Neon, bits: [u16; 4]) -> f32x4<Neon> {
    f16_u32x4_to_f32_x4(neon, u32x4::from_slice(neon, &bits.map(u32::from)))
}

#[inline]
pub(super) fn f16_bits_to_f32_x8(neon: Neon, bits: &[u16; 8]) -> (f32x4<Neon>, f32x4<Neon>) {
    let raw = u16x8::from_slice(neon, bits);
    let zero = u16x8::splat(neon, 0);
    let low = raw.zip_low(zero).bitcast::<u32x4<Neon>>();
    let high = raw.zip_high(zero).bitcast::<u32x4<Neon>>();
    (
        f16_u32x4_to_f32_x4(neon, low),
        f16_u32x4_to_f32_x4(neon, high),
    )
}

#[inline]
pub(super) fn f16_pair_x4_to_f32_x4(
    neon: Neon,
    first: &[u16],
    second: &[u16],
) -> (f32x4<Neon>, f32x4<Neon>) {
    let packed = [
        first[0], first[1], first[2], first[3], second[0], second[1], second[2], second[3],
    ];
    f16_bits_to_f32_x8(neon, &packed)
}

#[inline]
#[allow(
    clippy::suboptimal_flops,
    reason = "Preserve ggml_compute_fp32_to_fp16 separate multiply/add rounding."
)]
fn half_bits_u32x4(neon: Neon, v: f32x4<Neon>) -> u32x4<Neon> {
    let mut base = (v.abs() * f32x4::splat(neon, f32::from_bits(0x7780_0000)))
        * f32x4::splat(neon, f32::from_bits(0x0880_0000));
    let bits = v.bitcast::<u32x4<Neon>>();
    let doubled = bits + bits;
    let exponent_bias = (doubled & 0xff00_0000).max(u32x4::splat(neon, 0x7100_0000));
    base += ((exponent_bias >> 1) + u32x4::splat(neon, 0x0780_0000)).bitcast::<f32x4<Neon>>();
    let rounded = base.bitcast::<u32x4<Neon>>();
    let normal_mag = ((rounded >> 13) & 0x7c00) + (rounded & 0x0fff);
    let magnitude = doubled
        .simd_gt(u32x4::splat(neon, 0xff00_0000))
        .select(u32x4::splat(neon, 0x7e00), normal_mag);
    (((bits >> 16) & 0x8000) | magnitude) & 0xffff
}

#[inline]
pub(super) fn round_f16_x4(neon: Neon, v: f32x4<Neon>) -> f32x4<Neon> {
    f16_u32x4_to_f32_x4(neon, half_bits_u32x4(neon, v))
}

#[inline]
const fn scale_u16(block: &[u8; 34]) -> u16 {
    u16::from_le_bytes([block[0], block[1]])
}

#[inline]
pub(super) fn dot_consumption_order_prewidened(
    packed: &[u8; 32],
    values: &[i8; 128],
    widened: &[i16; 128],
    activation_sum: i32,
) -> i32 {
    if bonsai_neon_dot::has_dotprod() {
        return bonsai_neon_dot::dot_pq2_consumption_block(packed, values, activation_sum);
    }
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return packed
            .as_chunks::<16>()
            .0
            .iter()
            .zip(widened.as_chunks::<64>().0)
            .map(|(bytes, values)| {
                bytes
                    .iter()
                    .enumerate()
                    .map(|(index, byte)| {
                        (0..4)
                            .map(|digit| {
                                (i32::from((byte >> (digit * 2)) & 3) - 1)
                                    * i32::from(values[digit * 16 + index])
                            })
                            .sum::<i32>()
                    })
                    .sum::<i32>()
            })
            .sum();
    };
    let mut acc = i16x8::splat(neon, 0);
    for (bytes, values) in packed
        .as_chunks::<16>()
        .0
        .iter()
        .zip(widened.as_chunks::<64>().0)
    {
        let w = u8x16::from_slice(neon, bytes);
        let planes = values.as_chunks::<8>().0;
        acc += pq2_half_unsigned_sum(
            w,
            i16x8::from_slice(neon, &planes[0]),
            i16x8::from_slice(neon, &planes[1]),
            i16x8::from_slice(neon, &planes[2]),
            i16x8::from_slice(neon, &planes[3]),
            i16x8::from_slice(neon, &planes[4]),
            i16x8::from_slice(neon, &planes[5]),
            i16x8::from_slice(neon, &planes[6]),
            i16x8::from_slice(neon, &planes[7]),
        );
    }
    pq2_reduce(acc) - activation_sum
}

#[inline]
pub(super) fn dot_four_rows_consumption<const FUSED: bool>(
    rows: [&[u8]; 4],
    input: &[super::packed_pq2::Activation],
) -> Option<Result<[f32; 4]>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let use_dotprod = bonsai_neon_dot::has_dotprod();
    let c0 = rows[0].as_chunks::<34>().0;
    let c1 = rows[1].as_chunks::<34>().0;
    let c2 = rows[2].as_chunks::<34>().0;
    let c3 = rows[3].as_chunks::<34>().0;
    let mut outputs = f32x4::splat(neon, 0.0);
    let mut scales_finite = outputs.is_finite();
    for ((((b0, b1), b2), b3), activation) in c0.iter().zip(c1).zip(c2).zip(c3).zip(input) {
        let row_scales = f16_bits_to_f32_x4(
            neon,
            [scale_u16(b0), scale_u16(b1), scale_u16(b2), scale_u16(b3)],
        );
        scales_finite &= row_scales.is_finite();
        let ds = row_scales * f32x4::splat(neon, activation.scale);
        let sums = if use_dotprod {
            let [_, _, ref p0 @ ..] = *b0;
            let [_, _, ref p1 @ ..] = *b1;
            let [_, _, ref p2 @ ..] = *b2;
            let [_, _, ref p3 @ ..] = *b3;
            i32x4::from_slice(
                neon,
                &bonsai_neon_dot::dot_pq2_four_consumption_precomputed(
                    [p0, p1, p2, p3],
                    &activation.values,
                    activation.sum,
                ),
            )
            .to_float::<f32x4<Neon>>()
        } else {
            let blocks = [b0, b1, b2, b3];
            let mut acc = [i16x8::splat(neon, 0); 4];
            for (half, values) in activation.widened.as_chunks::<64>().0.iter().enumerate() {
                let planes = values.as_chunks::<8>().0;
                let p0_low = i16x8::from_slice(neon, &planes[0]);
                let p0_high = i16x8::from_slice(neon, &planes[1]);
                let p1_low = i16x8::from_slice(neon, &planes[2]);
                let p1_high = i16x8::from_slice(neon, &planes[3]);
                let p2_low = i16x8::from_slice(neon, &planes[4]);
                let p2_high = i16x8::from_slice(neon, &planes[5]);
                let p3_low = i16x8::from_slice(neon, &planes[6]);
                let p3_high = i16x8::from_slice(neon, &planes[7]);
                let offset = 2 + half * 16;
                for row in 0..4 {
                    let w = u8x16::from_slice(neon, &blocks[row][offset..offset + 16]);
                    acc[row] += pq2_half_unsigned_sum(
                        w, p0_low, p0_high, p1_low, p1_high, p2_low, p2_high, p3_low, p3_high,
                    );
                }
            }
            (reduce4_i16x8_to_i32x4(acc[0], acc[1], acc[2], acc[3])
                - i32x4::splat(neon, activation.sum))
            .to_float::<f32x4<Neon>>()
        };
        outputs = if FUSED {
            ds.mul_add(sums, outputs)
        } else {
            outputs + ds * sums
        };
    }
    if !(scales_finite & outputs.is_finite()).all_true() {
        return Some(Err(crate::invalid("nonfinite tensor or activation value")));
    }
    let mut result = [0.0_f32; 4];
    outputs.store_slice(&mut result);
    Some(Ok(result))
}

#[inline]
pub(super) fn dot_eight_rows_consumption<const FUSED: bool>(
    rows: [&[u8]; 8],
    input: &[super::packed_pq2::Activation],
) -> Option<Result<[f32; 8]>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let use_dotprod = bonsai_neon_dot::has_dotprod();
    let chunks: [&[[u8; 34]]; 8] = rows.map(|row| row.as_chunks::<34>().0);
    let mut out_lo = f32x4::splat(neon, 0.0);
    let mut out_hi = f32x4::splat(neon, 0.0);
    let mut scales_finite = out_lo.is_finite();
    for (block_idx, activation) in input.iter().enumerate() {
        let blocks: [&[u8; 34]; 8] = std::array::from_fn(|row| &chunks[row][block_idx]);
        let (scales_lo, scales_hi) = f16_bits_to_f32_x8(
            neon,
            &[
                scale_u16(blocks[0]),
                scale_u16(blocks[1]),
                scale_u16(blocks[2]),
                scale_u16(blocks[3]),
                scale_u16(blocks[4]),
                scale_u16(blocks[5]),
                scale_u16(blocks[6]),
                scale_u16(blocks[7]),
            ],
        );
        scales_finite = scales_finite & scales_lo.is_finite() & scales_hi.is_finite();
        let act_scale = f32x4::splat(neon, activation.scale);
        let ds_lo = scales_lo * act_scale;
        let ds_hi = scales_hi * act_scale;
        let (sums_lo, sums_hi) = if use_dotprod {
            let [_, _, ref p0 @ ..] = *blocks[0];
            let [_, _, ref p1 @ ..] = *blocks[1];
            let [_, _, ref p2 @ ..] = *blocks[2];
            let [_, _, ref p3 @ ..] = *blocks[3];
            let [_, _, ref p4 @ ..] = *blocks[4];
            let [_, _, ref p5 @ ..] = *blocks[5];
            let [_, _, ref p6 @ ..] = *blocks[6];
            let [_, _, ref p7 @ ..] = *blocks[7];
            let dots = bonsai_neon_dot::dot_pq2_eight_consumption_precomputed(
                [p0, p1, p2, p3, p4, p5, p6, p7],
                &activation.values,
                activation.sum,
            );
            (
                i32x4::from_slice(neon, &dots[..4]).to_float::<f32x4<Neon>>(),
                i32x4::from_slice(neon, &dots[4..]).to_float::<f32x4<Neon>>(),
            )
        } else {
            let mut acc = [i16x8::splat(neon, 0); 8];
            for (half, values) in activation.widened.as_chunks::<64>().0.iter().enumerate() {
                let planes = values.as_chunks::<8>().0;
                let p0_low = i16x8::from_slice(neon, &planes[0]);
                let p0_high = i16x8::from_slice(neon, &planes[1]);
                let p1_low = i16x8::from_slice(neon, &planes[2]);
                let p1_high = i16x8::from_slice(neon, &planes[3]);
                let p2_low = i16x8::from_slice(neon, &planes[4]);
                let p2_high = i16x8::from_slice(neon, &planes[5]);
                let p3_low = i16x8::from_slice(neon, &planes[6]);
                let p3_high = i16x8::from_slice(neon, &planes[7]);
                let offset = 2 + half * 16;
                for row in 0..8 {
                    let w = u8x16::from_slice(neon, &blocks[row][offset..offset + 16]);
                    acc[row] += pq2_half_unsigned_sum(
                        w, p0_low, p0_high, p1_low, p1_high, p2_low, p2_high, p3_low, p3_high,
                    );
                }
            }
            let act_sum = i32x4::splat(neon, activation.sum);
            (
                (reduce4_i16x8_to_i32x4(acc[0], acc[1], acc[2], acc[3]) - act_sum)
                    .to_float::<f32x4<Neon>>(),
                (reduce4_i16x8_to_i32x4(acc[4], acc[5], acc[6], acc[7]) - act_sum)
                    .to_float::<f32x4<Neon>>(),
            )
        };
        if FUSED {
            out_lo = ds_lo.mul_add(sums_lo, out_lo);
            out_hi = ds_hi.mul_add(sums_hi, out_hi);
        } else {
            out_lo += ds_lo * sums_lo;
            out_hi += ds_hi * sums_hi;
        }
    }
    if !(scales_finite & out_lo.is_finite() & out_hi.is_finite()).all_true() {
        return Some(Err(crate::invalid("nonfinite tensor or activation value")));
    }
    let mut result = [0.0_f32; 8];
    out_lo.store_slice(&mut result[..4]);
    out_hi.store_slice(&mut result[4..]);
    Some(Ok(result))
}

#[inline]
fn max_abs_with_sign_256(neon: Neon, chunk: &[f32; 256]) -> Result<(f32, f32)> {
    let mut max_abs_vec = f32x4::splat(neon, 0.0);
    let mut finite_mask = max_abs_vec.is_finite();
    for group in chunk.as_chunks::<16>().0 {
        let v0 = f32x4::from_slice(neon, &group[0..4]);
        let v1 = f32x4::from_slice(neon, &group[4..8]);
        let v2 = f32x4::from_slice(neon, &group[8..12]);
        let v3 = f32x4::from_slice(neon, &group[12..16]);
        finite_mask &= (v0.is_finite() & v1.is_finite()) & (v2.is_finite() & v3.is_finite());
        max_abs_vec = max_abs_vec
            .max(v0.abs())
            .max(v1.abs())
            .max(v2.abs())
            .max(v3.abs());
    }
    if !finite_mask.all_true() {
        return Err(crate::invalid("nonfinite tensor or activation value"));
    }
    let mut max_lanes = [0.0_f32; 4];
    max_abs_vec.store_slice(&mut max_lanes);
    let max_abs = max_lanes[0]
        .max(max_lanes[1])
        .max(max_lanes[2])
        .max(max_lanes[3]);
    if max_abs == 0.0 {
        return Ok((0.0, 0.0));
    }
    let mut maximum = 0.0_f32;
    let target_abs = f32x4::splat(neon, max_abs);
    for sub in chunk.as_chunks::<4>().0 {
        if f32x4::from_slice(neon, sub)
            .abs()
            .simd_eq(target_abs)
            .any_true()
        {
            for &value in sub {
                if value.abs().to_bits() == max_abs.to_bits() {
                    maximum = value;
                    break;
                }
            }
            break;
        }
    }
    Ok((max_abs, maximum))
}

#[inline]
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Each narrowed Q8_K lane and 16-element i8 sum is bounded within i16."
)]
pub(super) fn q8k_block<const PINNED: bool>(
    chunk: &[f32; 256],
) -> Option<Result<bonsai_ternary::Q8KBlock>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let (max_abs, maximum) = match max_abs_with_sign_256(neon, chunk) {
        Ok(pair) => pair,
        Err(err) => return Some(Err(err)),
    };
    let mut block = bonsai_ternary::Q8KBlock {
        scale: 0.0,
        values: [0; 256],
        sums: [0; 16],
    };
    if max_abs == 0.0 {
        return Some(Ok(block));
    }
    let inverse_scale = -127.0 / maximum;
    if !inverse_scale.is_finite() {
        return Some(Err(crate::invalid("Q8_K activation scale overflow")));
    }
    block.scale = 1.0 / inverse_scale;
    let iscale = f32x4::splat(neon, inverse_scale);
    let magic = f32x4::splat(neon, 12_582_912.0);
    let mask_mant = u32x4::splat(neon, 0x007f_ffff);
    let bias = i32x4::splat(neon, 0x0040_0000);
    let max_code = i32x4::splat(neon, 127);
    let min_code = i32x4::splat(neon, if PINNED { -128 } else { -127 });
    let mask16 = u32x4::splat(neon, 0xffff);
    let mask8 = u16x8::splat(neon, 0xff);
    let mut valid_mask = iscale.is_finite();
    for ((group, dst_values), dst_sum) in chunk
        .as_chunks::<16>()
        .0
        .iter()
        .zip(block.values.as_chunks_mut::<16>().0)
        .zip(&mut block.sums)
    {
        let subs = group.as_chunks::<4>().0;
        let v0 = f32x4::from_slice(neon, &subs[0]);
        let v1 = f32x4::from_slice(neon, &subs[1]);
        let v2 = f32x4::from_slice(neon, &subs[2]);
        let v3 = f32x4::from_slice(neon, &subs[3]);
        let (r0, r1, r2, r3) = if PINNED {
            (
                iscale.mul_add(v0, magic),
                iscale.mul_add(v1, magic),
                iscale.mul_add(v2, magic),
                iscale.mul_add(v3, magic),
            )
        } else {
            (
                (v0 * iscale) + magic,
                (v1 * iscale) + magic,
                (v2 * iscale) + magic,
                (v3 * iscale) + magic,
            )
        };
        let i0 = ((r0.bitcast::<u32x4<Neon>>() & mask_mant).bitcast::<i32x4<Neon>>() - bias)
            .min(max_code);
        let i1 = ((r1.bitcast::<u32x4<Neon>>() & mask_mant).bitcast::<i32x4<Neon>>() - bias)
            .min(max_code);
        let i2 = ((r2.bitcast::<u32x4<Neon>>() & mask_mant).bitcast::<i32x4<Neon>>() - bias)
            .min(max_code);
        let i3 = ((r3.bitcast::<u32x4<Neon>>() & mask_mant).bitcast::<i32x4<Neon>>() - bias)
            .min(max_code);
        valid_mask &= (r0.is_finite() & r1.is_finite())
            & (r2.is_finite() & r3.is_finite())
            & (i0.simd_ge(min_code) & i1.simd_ge(min_code))
            & (i2.simd_ge(min_code) & i3.simd_ge(min_code));
        let low16 = (i0.bitcast::<u32x4<Neon>>() & mask16)
            .narrow(i1.bitcast::<u32x4<Neon>>() & mask16)
            .bitcast::<i16x8<Neon>>();
        let high16 = (i2.bitcast::<u32x4<Neon>>() & mask16)
            .narrow(i3.bitcast::<u32x4<Neon>>() & mask16)
            .bitcast::<i16x8<Neon>>();
        let bytes = (low16.bitcast::<u16x8<Neon>>() & mask8)
            .narrow(high16.bitcast::<u16x8<Neon>>() & mask8)
            .bitcast::<i8x16<Neon>>();
        bytes.store_slice(dst_values);
        let (sum_lo, sum_hi) = (low16 + high16).widen();
        *dst_sum = (sum_lo + sum_hi).reduce_sum() as i16;
    }
    if !valid_mask.all_true() {
        return None;
    }
    Some(Ok(block))
}

#[inline]
pub(super) fn q8_pinned_block<const NATIVE_SCALE: bool>(
    chunk: &[f32; 32],
) -> Option<Result<super::Q8Block>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let mut max_abs_vec = f32x4::splat(neon, 0.0);
    let mut finite_mask = max_abs_vec.is_finite();
    for group in chunk.as_chunks::<16>().0 {
        let v0 = f32x4::from_slice(neon, &group[0..4]);
        let v1 = f32x4::from_slice(neon, &group[4..8]);
        let v2 = f32x4::from_slice(neon, &group[8..12]);
        let v3 = f32x4::from_slice(neon, &group[12..16]);
        finite_mask &= (v0.is_finite() & v1.is_finite()) & (v2.is_finite() & v3.is_finite());
        max_abs_vec = max_abs_vec
            .max(v0.abs())
            .max(v1.abs())
            .max(v2.abs())
            .max(v3.abs());
    }
    if !finite_mask.all_true() {
        return Some(Err(crate::invalid("nonfinite tensor or activation value")));
    }
    let max_lanes = max_abs_vec.to_array();
    let maximum = max_lanes[0]
        .max(max_lanes[1])
        .max(max_lanes[2])
        .max(max_lanes[3]);
    let d = if NATIVE_SCALE {
        maximum * f32::from_bits(0x3c01_0204)
    } else {
        maximum / 127.0
    };
    let inverse_scale = if d == 0.0 {
        0.0
    } else if NATIVE_SCALE {
        127.0 / maximum
    } else {
        1.0 / d
    };
    let scale = super::round_f16(d);
    if !scale.is_finite() {
        return Some(Err(crate::invalid("nonfinite tensor or activation value")));
    }
    let mut block = super::Q8Block {
        scale,
        sum: 0.0,
        values: [0; 32],
    };
    if inverse_scale == 0.0 {
        block.sum = super::round_f16(0.0 * d);
        return Some(Ok(block));
    }
    let iscale = f32x4::splat(neon, inverse_scale);
    let magic = f32x4::splat(neon, 12_582_912.0);
    let mask_mant = u32x4::splat(neon, 0x007f_ffff);
    let bias = i32x4::splat(neon, 0x0040_0000);
    let max_code = i32x4::splat(neon, 127);
    let min_code = i32x4::splat(neon, -127);
    let mask16 = u32x4::splat(neon, 0xffff);
    let mask8 = u16x8::splat(neon, 0xff);
    let mut total_sum = i32x4::splat(neon, 0);
    let mut valid_mask = iscale.is_finite();
    for (group, dst_values) in chunk
        .as_chunks::<16>()
        .0
        .iter()
        .zip(block.values.as_chunks_mut::<16>().0)
    {
        let subs = group.as_chunks::<4>().0;
        let r0 = (f32x4::from_slice(neon, &subs[0]) * iscale) + magic;
        let r1 = (f32x4::from_slice(neon, &subs[1]) * iscale) + magic;
        let r2 = (f32x4::from_slice(neon, &subs[2]) * iscale) + magic;
        let r3 = (f32x4::from_slice(neon, &subs[3]) * iscale) + magic;
        let i0 = (r0.bitcast::<u32x4<Neon>>() & mask_mant).bitcast::<i32x4<Neon>>() - bias;
        let i1 = (r1.bitcast::<u32x4<Neon>>() & mask_mant).bitcast::<i32x4<Neon>>() - bias;
        let i2 = (r2.bitcast::<u32x4<Neon>>() & mask_mant).bitcast::<i32x4<Neon>>() - bias;
        let i3 = (r3.bitcast::<u32x4<Neon>>() & mask_mant).bitcast::<i32x4<Neon>>() - bias;
        valid_mask &= (r0.is_finite() & r1.is_finite())
            & (r2.is_finite() & r3.is_finite())
            & (i0.simd_ge(min_code) & i0.simd_le(max_code))
            & (i1.simd_ge(min_code) & i1.simd_le(max_code))
            & (i2.simd_ge(min_code) & i2.simd_le(max_code))
            & (i3.simd_ge(min_code) & i3.simd_le(max_code));
        total_sum += i0 + i1 + i2 + i3;
        let low16 =
            (i0.bitcast::<u32x4<Neon>>() & mask16).narrow(i1.bitcast::<u32x4<Neon>>() & mask16);
        let high16 =
            (i2.bitcast::<u32x4<Neon>>() & mask16).narrow(i3.bitcast::<u32x4<Neon>>() & mask16);
        let bytes = (low16 & mask8)
            .narrow(high16 & mask8)
            .bitcast::<i8x16<Neon>>();
        bytes.store_slice(dst_values);
    }
    if !valid_mask.all_true() {
        return None;
    }
    block.sum = super::round_f16(super::integer_float(total_sum.reduce_sum()) * d);
    Some(Ok(block))
}

#[inline]
#[allow(
    clippy::suboptimal_flops,
    reason = "Preserve separate binary32 product rounding and signed-zero normalization."
)]
pub(super) fn lora_b_scale(
    b_values: &[f32],
    rank: f32,
    correction: &mut Vec<f32>,
) -> Option<Result<()>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    correction.resize(b_values.len(), 0.0);
    let rank_vec = f32x4::splat(neon, rank);
    let zero_vec = f32x4::splat(neon, 0.0);
    let (b_chunks, b_tail) = b_values.as_chunks::<4>();
    let (c_chunks, c_tail) = correction.as_chunks_mut::<4>();
    for (b_chunk, c_chunk) in b_chunks.iter().zip(c_chunks) {
        let b = f32x4::from_slice(neon, b_chunk);
        // Adding +0.0 normalizes -0.0 to +0.0 identically to f64::sum::<f64>().
        let prod = (b * rank_vec) + zero_vec;
        if !(b.is_finite() & prod.is_finite()).all_true() {
            return Some(Err(crate::invalid("nonfinite tensor or activation value")));
        }
        prod.store_slice(c_chunk);
    }
    for (&b, c) in b_tail.iter().zip(c_tail) {
        if !b.is_finite() {
            return Some(Err(crate::invalid("nonfinite tensor or activation value")));
        }
        let prod = (b * rank) + 0.0;
        if !prod.is_finite() {
            return Some(Err(crate::invalid("nonfinite tensor or activation value")));
        }
        *c = prod;
    }
    Some(Ok(()))
}

#[inline]
#[allow(
    clippy::suboptimal_flops,
    reason = "Preserve separate binary32 scale and base addition roundings."
)]
pub(super) fn lora_finish_in_place(
    scale: f32,
    output: &mut [f32],
    correction: &mut [f32],
) -> Option<Result<()>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let scale_vec = f32x4::splat(neon, scale);
    let (out_chunks, out_tail) = output.as_chunks::<4>();
    let (corr_chunks, corr_tail) = correction.as_chunks_mut::<4>();
    for (base_chunk, corr_chunk) in out_chunks.iter().zip(corr_chunks) {
        let base = f32x4::from_slice(neon, base_chunk);
        let corr = (f32x4::from_slice(neon, corr_chunk) * scale_vec) + base;
        if !corr.is_finite().all_true() {
            return Some(Err(crate::invalid("nonfinite rank-one correction")));
        }
        corr.store_slice(corr_chunk);
    }
    for (&base, corr) in out_tail.iter().zip(corr_tail) {
        *corr *= scale;
        *corr += base;
        if !corr.is_finite() {
            return Some(Err(crate::invalid("nonfinite rank-one correction")));
        }
    }
    output.copy_from_slice(correction);
    Some(Ok(()))
}

#[inline]
#[allow(
    clippy::suboptimal_flops,
    reason = "Preserve separate binary32 product rounding, signed-zero normalization, scale and base addition."
)]
pub(super) fn lora_b_scale_and_finish(
    b_values: &[f32],
    rank: f32,
    scale: f32,
    output: &mut [f32],
    correction: &mut Vec<f32>,
) -> Option<Result<()>> {
    if output.len() != b_values.len() {
        return None;
    }
    let neon = fearless_simd::Level::baseline().as_neon()?;
    correction.resize(b_values.len(), 0.0);
    let rank_vec = f32x4::splat(neon, rank);
    let zero_vec = f32x4::splat(neon, 0.0);
    let scale_vec = f32x4::splat(neon, scale);
    let mut prod_finite = zero_vec.is_finite();
    let mut corr_finite = zero_vec.is_finite();
    let (b_chunks, b_tail) = b_values.as_chunks::<4>();
    let (out_chunks, out_tail) = output.as_chunks::<4>();
    let (c_chunks, c_tail) = correction.as_chunks_mut::<4>();
    for ((b_chunk, base_chunk), c_chunk) in b_chunks.iter().zip(out_chunks).zip(c_chunks) {
        let b = f32x4::from_slice(neon, b_chunk);
        let prod = (b * rank_vec) + zero_vec;
        prod_finite &= b.is_finite() & prod.is_finite();
        let base = f32x4::from_slice(neon, base_chunk);
        let corr = (prod * scale_vec) + base;
        corr_finite &= corr.is_finite();
        corr.store_slice(c_chunk);
    }
    if !prod_finite.all_true() {
        correction.clear();
        return Some(Err(crate::invalid("nonfinite tensor or activation value")));
    }
    for ((&b, &base), c) in b_tail.iter().zip(out_tail).zip(c_tail) {
        if !b.is_finite() {
            correction.clear();
            return Some(Err(crate::invalid("nonfinite tensor or activation value")));
        }
        let prod = (b * rank) + 0.0;
        if !prod.is_finite() {
            correction.clear();
            return Some(Err(crate::invalid("nonfinite tensor or activation value")));
        }
        let corr = (prod * scale) + base;
        if !corr.is_finite() {
            return Some(Err(crate::invalid("nonfinite rank-one correction")));
        }
        *c = corr;
    }
    if !corr_finite.all_true() {
        return Some(Err(crate::invalid("nonfinite rank-one correction")));
    }
    output.copy_from_slice(correction);
    Some(Ok(()))
}

#[inline]
pub(super) fn round_f16_slice(values: &mut [f32]) -> Option<bool> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let mut finite_mask = f32x4::splat(neon, 0.0).is_finite();
    let (chunks, tail) = values.as_chunks_mut::<4>();
    for chunk in chunks {
        let rounded = round_f16_x4(neon, f32x4::from_slice(neon, chunk));
        finite_mask &= rounded.is_finite();
        rounded.store_slice(chunk);
    }
    let mut all_finite = finite_mask.all_true();
    for value in tail {
        let rounded = super::round_f16(*value);
        all_finite &= rounded.is_finite();
        *value = rounded;
    }
    Some(all_finite)
}

#[inline]
pub(super) fn softmax_scale_round_f16_in_place(values: &mut [f32], inverse_sum: f32) -> bool {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return false;
    };
    let inv_vec = f32x4::splat(neon, inverse_sum);
    let (chunks, tail) = values.as_chunks_mut::<4>();
    for chunk in chunks {
        round_f16_x4(neon, f32x4::from_slice(neon, chunk) * inv_vec).store_slice(chunk);
    }
    for value in tail {
        *value = super::round_f16(*value * inverse_sum);
    }
    true
}

#[inline]
pub(super) fn round_f16_into(input: &[f32], output: &mut Vec<f32>) -> Option<Result<()>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    if output.len() != input.len() {
        output.clear();
        if let Err(err) = output.try_reserve_exact(input.len()) {
            return Some(Err(err.into()));
        }
        output.resize(input.len(), 0.0);
    }
    let mut finite_mask = f32x4::splat(neon, 0.0).is_finite();
    let (in_chunks, in_tail) = input.as_chunks::<4>();
    let (out_chunks, out_tail) = output.as_chunks_mut::<4>();
    for (in_chunk, out_chunk) in in_chunks.iter().zip(out_chunks) {
        let rounded = round_f16_x4(neon, f32x4::from_slice(neon, in_chunk));
        finite_mask &= rounded.is_finite();
        rounded.store_slice(out_chunk);
    }
    if !finite_mask.all_true() {
        return Some(Err(crate::invalid("nonfinite tensor or activation value")));
    }
    for (&value, out) in in_tail.iter().zip(out_tail) {
        let rounded = super::round_f16(value);
        if !rounded.is_finite() {
            return Some(Err(crate::invalid("nonfinite tensor or activation value")));
        }
        *out = rounded;
    }
    Some(Ok(()))
}

#[inline]
pub(super) fn round_bf16_into(input: &[f32], output: &mut Vec<f32>) -> Option<Result<()>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    if output.len() != input.len() {
        output.clear();
        if let Err(err) = output.try_reserve_exact(input.len()) {
            return Some(Err(err.into()));
        }
        output.resize(input.len(), 0.0);
    }
    let bias_base = u32x4::splat(neon, 0x7fff);
    let one = u32x4::splat(neon, 1);
    let mask = u32x4::splat(neon, 0xffff_0000);
    let mut finite_mask = f32x4::splat(neon, 0.0).is_finite();
    let (in_chunks, in_tail) = input.as_chunks::<4>();
    let (out_chunks, out_tail) = output.as_chunks_mut::<4>();
    for (in_chunk, out_chunk) in in_chunks.iter().zip(out_chunks) {
        let bits = f32x4::from_slice(neon, in_chunk).bitcast::<u32x4<Neon>>();
        let rounded = ((bits + (bias_base + ((bits >> 16) & one))) & mask).bitcast::<f32x4<Neon>>();
        finite_mask &= rounded.is_finite();
        rounded.store_slice(out_chunk);
    }
    if !finite_mask.all_true() {
        return Some(Err(crate::invalid("nonfinite tensor or activation value")));
    }
    for (&value, out) in in_tail.iter().zip(out_tail) {
        let rounded = super::round_bf16(value);
        if !rounded.is_finite() {
            return Some(Err(crate::invalid("nonfinite tensor or activation value")));
        }
        *out = rounded;
    }
    Some(Ok(()))
}

#[inline]
pub(super) fn append_half_bits(values: &[f32], output: &mut Vec<u16>) -> bool {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return false;
    };
    let (chunks, tail) = values.as_chunks::<8>();
    let mut converted = [0_u16; 8];
    for chunk in chunks {
        let low = half_bits_u32x4(neon, f32x4::from_slice(neon, &chunk[..4]));
        let high = half_bits_u32x4(neon, f32x4::from_slice(neon, &chunk[4..]));
        low.narrow(high).store_slice(&mut converted);
        output.extend_from_slice(&converted);
    }
    output.extend(tail.iter().copied().map(super::half_bits));
    true
}

#[inline]
pub(super) fn is_finite_slice(values: &[f32]) -> Option<bool> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let (groups, remainder) = values.as_chunks::<16>();
    for group in groups {
        let v0 = f32x4::from_slice(neon, &group[0..4]);
        let v1 = f32x4::from_slice(neon, &group[4..8]);
        let v2 = f32x4::from_slice(neon, &group[8..12]);
        let v3 = f32x4::from_slice(neon, &group[12..16]);
        if !(v0.is_finite() & v1.is_finite() & v2.is_finite() & v3.is_finite()).all_true() {
            return Some(false);
        }
    }
    let (chunks, tail) = remainder.as_chunks::<4>();
    for chunk in chunks {
        if !f32x4::from_slice(neon, chunk).is_finite().all_true() {
            return Some(false);
        }
    }
    for &value in tail {
        if !value.is_finite() {
            return Some(false);
        }
    }
    Some(true)
}

#[inline]
pub(super) fn block16_all_finite_le(values: &[f32; 16], threshold: f32) -> bool {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return false;
    };
    let v0 = f32x4::from_slice(neon, &values[0..4]);
    let v1 = f32x4::from_slice(neon, &values[4..8]);
    let v2 = f32x4::from_slice(neon, &values[8..12]);
    let v3 = f32x4::from_slice(neon, &values[12..16]);
    let t = f32x4::splat(neon, threshold);
    (v0.is_finite() & v1.is_finite() & v2.is_finite() & v3.is_finite()).all_true()
        && !(v0.simd_gt(t) | v1.simd_gt(t) | v2.simd_gt(t) | v3.simd_gt(t)).any_true()
}

#[inline]
pub(super) fn add_slice_in_place(target: &mut [f32], other: &[f32]) -> bool {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return false;
    };
    let (t_chunks, t_tail) = target.as_chunks_mut::<4>();
    let (o_chunks, o_tail) = other.as_chunks::<4>();
    for (t_chunk, o_chunk) in t_chunks.iter_mut().zip(o_chunks) {
        (f32x4::from_slice(neon, t_chunk) + f32x4::from_slice(neon, o_chunk)).store_slice(t_chunk);
    }
    for (t, &o) in t_tail.iter_mut().zip(o_tail) {
        *t += o;
    }
    true
}

#[inline]
pub(super) fn add_slice_in_place_all_finite(target: &mut [f32], other: &[f32]) -> Option<bool> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let mut finite_mask = f32x4::splat(neon, 0.0).is_finite();
    let (t_chunks, t_tail) = target.as_chunks_mut::<4>();
    let (o_chunks, o_tail) = other.as_chunks::<4>();
    for (t_chunk, o_chunk) in t_chunks.iter_mut().zip(o_chunks) {
        let sum = f32x4::from_slice(neon, t_chunk) + f32x4::from_slice(neon, o_chunk);
        finite_mask &= sum.is_finite();
        sum.store_slice(t_chunk);
    }
    let mut tail_finite = true;
    for (t, &o) in t_tail.iter_mut().zip(o_tail) {
        let sum = *t + o;
        *t = sum;
        tail_finite &= sum.is_finite();
    }
    if target.len() > other.len() {
        for &t in &target[other.len()..] {
            tail_finite &= t.is_finite();
        }
    }
    Some(finite_mask.all_true() && tail_finite)
}

#[inline]
pub(super) fn mul_slice_in_place(target: &mut [f32], other: &[f32]) -> bool {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return false;
    };
    let (t_chunks, t_tail) = target.as_chunks_mut::<4>();
    let (o_chunks, o_tail) = other.as_chunks::<4>();
    for (t_chunk, o_chunk) in t_chunks.iter_mut().zip(o_chunks) {
        (f32x4::from_slice(neon, t_chunk) * f32x4::from_slice(neon, o_chunk)).store_slice(t_chunk);
    }
    for (t, &o) in t_tail.iter_mut().zip(o_tail) {
        *t *= o;
    }
    true
}

#[inline]
pub(super) fn mul_sigmoid_in_place(values: &mut [f32], gate: &[f32]) -> bool {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return false;
    };
    let count = values.len().min(gate.len());
    let one = f32x4::splat(neon, 1.0);
    let (v_chunks, v_tail) = values[..count].as_chunks_mut::<4>();
    let (g_chunks, g_tail) = gate[..count].as_chunks::<4>();
    for (v_chunk, g_chunk) in v_chunks.iter_mut().zip(g_chunks) {
        let v = f32x4::from_slice(neon, v_chunk);
        let exp_neg = f32x4::load_array(
            neon,
            [
                (-g_chunk[0]).exp(),
                (-g_chunk[1]).exp(),
                (-g_chunk[2]).exp(),
                (-g_chunk[3]).exp(),
            ],
        );
        (v * (one / (one + exp_neg))).store_slice(v_chunk);
    }
    for (v, &g) in v_tail.iter_mut().zip(g_tail) {
        *v *= 1.0 / (1.0 + (-g).exp());
    }
    true
}

#[inline]
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Preserve each channel's binary64 sum and final binary32 rounding."
)]
pub(super) fn source_value_tile<'a>(
    past: impl Iterator<Item = &'a [u16]>,
    current: &[f32],
    offset: usize,
    probabilities: &[f32],
    output: &mut [f32],
) -> bool {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return false;
    };
    let out_len = output.len();
    let last = probabilities.len().saturating_sub(1);
    let mut sums = [-0.0_f64; super::pinned_attention::VALUE_TILE];
    for (past_row, &prob) in past.zip(&probabilities[..last]) {
        let row = &past_row[offset..offset + out_len];
        let prob_vec = f32x4::splat(neon, prob);
        let (chunks8, rem) = row.as_chunks::<8>();
        for (chunk_idx, chunk8) in chunks8.iter().enumerate() {
            let (v0, v1) = f16_bits_to_f32_x8(neon, chunk8);
            let p0 = (prob_vec * v0).to_array();
            let p1 = (prob_vec * v1).to_array();
            let base = chunk_idx * 8;
            let s = &mut sums[base..base + 8];
            s[0] += f64::from(p0[0]);
            s[1] += f64::from(p0[1]);
            s[2] += f64::from(p0[2]);
            s[3] += f64::from(p0[3]);
            s[4] += f64::from(p1[0]);
            s[5] += f64::from(p1[1]);
            s[6] += f64::from(p1[2]);
            s[7] += f64::from(p1[3]);
        }
        let rem_base = chunks8.len() * 8;
        for (i, &val) in rem.iter().enumerate() {
            sums[rem_base + i] += f64::from(prob * bonsai_ternary::half_to_float(val));
        }
    }
    let prob = probabilities[last];
    for ((out, &sum), &val) in output.iter_mut().zip(&sums[..out_len]).zip(current) {
        *out = (sum + f64::from(prob * val)) as f32;
    }
    true
}

#[inline]
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    reason = "Preserve each channel's binary64 sum and final binary32 rounding across 4 heads."
)]
pub(super) fn source_value_tile_four<'a>(
    past: impl Iterator<Item = &'a [u16]>,
    current: &[f32],
    offset: usize,
    probabilities: [&[f32]; 4],
    outputs: [&mut [f32]; 4],
) -> bool {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return false;
    };
    let out_len = outputs[0].len();
    let last = probabilities[0].len().saturating_sub(1);
    let mut sums = [[-0.0_f64; super::pinned_attention::VALUE_TILE_FOUR]; 4];
    for (pos, past_row) in past.enumerate() {
        let row = &past_row[offset..offset + out_len];
        let probs: [f32x4<Neon>; 4] =
            std::array::from_fn(|h| f32x4::splat(neon, probabilities[h][pos]));
        let (chunks8, rem) = row.as_chunks::<8>();
        for (chunk_idx, chunk8) in chunks8.iter().enumerate() {
            let (v0, v1) = f16_bits_to_f32_x8(neon, chunk8);
            let base = chunk_idx * 8;
            for h in 0..4 {
                let p0 = (probs[h] * v0).to_array();
                let p1 = (probs[h] * v1).to_array();
                let s = &mut sums[h][base..base + 8];
                s[0] += f64::from(p0[0]);
                s[1] += f64::from(p0[1]);
                s[2] += f64::from(p0[2]);
                s[3] += f64::from(p0[3]);
                s[4] += f64::from(p1[0]);
                s[5] += f64::from(p1[1]);
                s[6] += f64::from(p1[2]);
                s[7] += f64::from(p1[3]);
            }
        }
        let rem_base = chunks8.len() * 8;
        for (i, &val) in rem.iter().enumerate() {
            let decoded = bonsai_ternary::half_to_float(val);
            for h in 0..4 {
                sums[h][rem_base + i] += f64::from(probabilities[h][pos] * decoded);
            }
        }
    }
    for (output, (prob_row, sum_row)) in outputs
        .into_iter()
        .zip(probabilities.into_iter().zip(&sums))
    {
        let probability = prob_row[last];
        for ((out, &sum), &value) in output.iter_mut().zip(&sum_row[..out_len]).zip(current) {
            *out = (sum + f64::from(probability * value)) as f32;
        }
    }
    true
}

#[inline]
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "Preserve each channel's binary64 sum and final binary32 rounding across two tokens and four heads."
)]
pub(super) fn source_value_tile_four_two_tokens<'a>(
    past: impl Iterator<Item = &'a [u16]>,
    currents: [&[f32]; 2],
    offset: usize,
    probabilities0: [&[f32]; 4],
    probabilities1: [&[f32]; 4],
    outputs0: [&mut [f32]; 4],
    outputs1: [&mut [f32]; 4],
) -> bool {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return false;
    };
    let out_len = outputs0[0].len();
    let last0 = probabilities0[0].len().saturating_sub(1);
    let last1 = probabilities1[0].len().saturating_sub(1);
    let mut sums0 = [[-0.0_f64; super::pinned_attention::VALUE_TILE_FOUR_TWO_TOKENS]; 4];
    let mut sums1 = [[-0.0_f64; super::pinned_attention::VALUE_TILE_FOUR_TWO_TOKENS]; 4];
    for (pos, past_row) in past.enumerate() {
        let row = &past_row[offset..offset + out_len];
        let probs1: [f32x4<Neon>; 4] =
            std::array::from_fn(|h| f32x4::splat(neon, probabilities1[h][pos]));
        let (chunks8, rem) = row.as_chunks::<8>();
        if pos < last0 {
            let probs0: [f32x4<Neon>; 4] =
                std::array::from_fn(|h| f32x4::splat(neon, probabilities0[h][pos]));
            for (chunk_idx, chunk8) in chunks8.iter().enumerate() {
                let (v0, v1) = f16_bits_to_f32_x8(neon, chunk8);
                let base = chunk_idx * 8;
                for h in 0..4 {
                    let p0_0 = (probs0[h] * v0).to_array();
                    let p0_1 = (probs0[h] * v1).to_array();
                    let s0 = &mut sums0[h][base..base + 8];
                    s0[0] += f64::from(p0_0[0]);
                    s0[1] += f64::from(p0_0[1]);
                    s0[2] += f64::from(p0_0[2]);
                    s0[3] += f64::from(p0_0[3]);
                    s0[4] += f64::from(p0_1[0]);
                    s0[5] += f64::from(p0_1[1]);
                    s0[6] += f64::from(p0_1[2]);
                    s0[7] += f64::from(p0_1[3]);
                    let p1_0 = (probs1[h] * v0).to_array();
                    let p1_1 = (probs1[h] * v1).to_array();
                    let s1 = &mut sums1[h][base..base + 8];
                    s1[0] += f64::from(p1_0[0]);
                    s1[1] += f64::from(p1_0[1]);
                    s1[2] += f64::from(p1_0[2]);
                    s1[3] += f64::from(p1_0[3]);
                    s1[4] += f64::from(p1_1[0]);
                    s1[5] += f64::from(p1_1[1]);
                    s1[6] += f64::from(p1_1[2]);
                    s1[7] += f64::from(p1_1[3]);
                }
            }
            let rem_base = chunks8.len() * 8;
            for (i, &val) in rem.iter().enumerate() {
                let decoded = bonsai_ternary::half_to_float(val);
                for h in 0..4 {
                    sums0[h][rem_base + i] += f64::from(probabilities0[h][pos] * decoded);
                    sums1[h][rem_base + i] += f64::from(probabilities1[h][pos] * decoded);
                }
            }
        } else {
            for (chunk_idx, chunk8) in chunks8.iter().enumerate() {
                let (v0, v1) = f16_bits_to_f32_x8(neon, chunk8);
                let base = chunk_idx * 8;
                for h in 0..4 {
                    let p1_0 = (probs1[h] * v0).to_array();
                    let p1_1 = (probs1[h] * v1).to_array();
                    let s1 = &mut sums1[h][base..base + 8];
                    s1[0] += f64::from(p1_0[0]);
                    s1[1] += f64::from(p1_0[1]);
                    s1[2] += f64::from(p1_0[2]);
                    s1[3] += f64::from(p1_0[3]);
                    s1[4] += f64::from(p1_1[0]);
                    s1[5] += f64::from(p1_1[1]);
                    s1[6] += f64::from(p1_1[2]);
                    s1[7] += f64::from(p1_1[3]);
                }
            }
            let rem_base = chunks8.len() * 8;
            for (i, &val) in rem.iter().enumerate() {
                let decoded = bonsai_ternary::half_to_float(val);
                for h in 0..4 {
                    sums1[h][rem_base + i] += f64::from(probabilities1[h][pos] * decoded);
                }
            }
        }
    }
    for (output, (prob_row, sum_row)) in outputs0
        .into_iter()
        .zip(probabilities0.into_iter().zip(&sums0))
    {
        let probability = prob_row[last0];
        for ((out, &sum), &value) in output.iter_mut().zip(&sum_row[..out_len]).zip(currents[0]) {
            *out = (sum + f64::from(probability * value)) as f32;
        }
    }
    for (output, (prob_row, sum_row)) in outputs1
        .into_iter()
        .zip(probabilities1.into_iter().zip(&sums1))
    {
        let probability = prob_row[last1];
        for ((out, &sum), &value) in output.iter_mut().zip(&sum_row[..out_len]).zip(currents[1]) {
            *out = (sum + f64::from(probability * value)) as f32;
        }
    }
    true
}

#[inline]
pub(super) fn mul_slice_into(input: &[f32], signs: &[f32], output: &mut [f32]) -> bool {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return false;
    };
    let (in_chunks, in_tail) = input.as_chunks::<4>();
    let (s_chunks, s_tail) = signs.as_chunks::<4>();
    let (out_chunks, out_tail) = output.as_chunks_mut::<4>();
    for ((in_chunk, s_chunk), out_chunk) in in_chunks.iter().zip(s_chunks).zip(out_chunks) {
        (f32x4::from_slice(neon, in_chunk) * f32x4::from_slice(neon, s_chunk))
            .store_slice(out_chunk);
    }
    for ((&v, &s), out) in in_tail.iter().zip(s_tail).zip(out_tail) {
        *out = v * s;
    }
    true
}

#[inline]
pub(super) fn scale_slice_into(input: &[f32], scale: f32, output: &mut [f32]) -> bool {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return false;
    };
    let count = input.len().min(output.len());
    let scale_vec = f32x4::splat(neon, scale);
    let (in_chunks, in_tail) = input[..count].as_chunks::<4>();
    let (out_chunks, out_tail) = output[..count].as_chunks_mut::<4>();
    for (in_chunk, out_chunk) in in_chunks.iter().zip(out_chunks) {
        (scale_vec * f32x4::from_slice(neon, in_chunk)).store_slice(out_chunk);
    }
    for (&v, out) in in_tail.iter().zip(out_tail) {
        *out = scale * v;
    }
    true
}

#[inline]
#[allow(
    clippy::suboptimal_flops,
    reason = "Preserve separate binary32 multiply and add roundings."
)]
pub(super) fn add_scaled_slice_in_place(target: &mut [f32], input: &[f32], scale: f32) -> bool {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return false;
    };
    let count = target.len().min(input.len());
    let scale_vec = f32x4::splat(neon, scale);
    let (t_chunks, t_tail) = target[..count].as_chunks_mut::<4>();
    let (in_chunks, in_tail) = input[..count].as_chunks::<4>();
    for (t_chunk, in_chunk) in t_chunks.iter_mut().zip(in_chunks) {
        let t = f32x4::from_slice(neon, t_chunk);
        let prod = scale_vec * f32x4::from_slice(neon, in_chunk);
        (t + prod).store_slice(t_chunk);
    }
    for (t, &v) in t_tail.iter_mut().zip(in_tail) {
        *t += scale * v;
    }
    true
}

#[inline]
#[allow(
    clippy::suboptimal_flops,
    reason = "Preserve separate binary32 multiply and add roundings matching scale_slice_into and add_scaled_slice_in_place."
)]
pub(super) fn gdn_wy_update_four_rows(
    states: [&[f32]; 4],
    last_cumulative: f32,
    inputs: &[super::pinned_gdn::ChunkInput<'_>],
    scaled_deltas: &[[f32; 32]; 4],
    matrix_rows: [&mut [f32]; 4],
) -> bool {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return false;
    };
    let width = matrix_rows[0].len();
    let count = inputs.len();
    let scale_vec = f32x4::splat(neon, last_cumulative);
    let vec16_blocks = width / 16;
    let [m0, m1, m2, m3] = matrix_rows;
    let mut rows = [m0, m1, m2, m3];
    for b in 0..vec16_blocks {
        let offset = b * 16;
        let mut acc = [[f32x4::splat(neon, 0.0); 4]; 4];
        for r in 0..4 {
            let s = &states[r][offset..offset + 16];
            acc[r] = [
                scale_vec * f32x4::from_slice(neon, &s[0..4]),
                scale_vec * f32x4::from_slice(neon, &s[4..8]),
                scale_vec * f32x4::from_slice(neon, &s[8..12]),
                scale_vec * f32x4::from_slice(neon, &s[12..16]),
            ];
        }
        for (prior, input) in inputs[..count].iter().enumerate() {
            let kb = &input.key[offset..offset + 16];
            let k0 = f32x4::from_slice(neon, &kb[0..4]);
            let k1 = f32x4::from_slice(neon, &kb[4..8]);
            let k2 = f32x4::from_slice(neon, &kb[8..12]);
            let k3 = f32x4::from_slice(neon, &kb[12..16]);
            for r in 0..4 {
                let sd = f32x4::splat(neon, scaled_deltas[r][prior]);
                acc[r][0] += sd * k0;
                acc[r][1] += sd * k1;
                acc[r][2] += sd * k2;
                acc[r][3] += sd * k3;
            }
        }
        for (out_row, row_acc) in rows.iter_mut().zip(acc.iter()) {
            let out_block = &mut out_row[offset..offset + 16];
            row_acc[0].store_slice(&mut out_block[0..4]);
            row_acc[1].store_slice(&mut out_block[4..8]);
            row_acc[2].store_slice(&mut out_block[8..12]);
            row_acc[3].store_slice(&mut out_block[12..16]);
        }
    }
    let vec4_blocks = width / 4;
    for b in (vec16_blocks * 4)..vec4_blocks {
        let offset = b * 4;
        let mut a0 = scale_vec * f32x4::from_slice(neon, &states[0][offset..offset + 4]);
        let mut a1 = scale_vec * f32x4::from_slice(neon, &states[1][offset..offset + 4]);
        let mut a2 = scale_vec * f32x4::from_slice(neon, &states[2][offset..offset + 4]);
        let mut a3 = scale_vec * f32x4::from_slice(neon, &states[3][offset..offset + 4]);
        for (prior, input) in inputs[..count].iter().enumerate() {
            let k = f32x4::from_slice(neon, &input.key[offset..offset + 4]);
            a0 += f32x4::splat(neon, scaled_deltas[0][prior]) * k;
            a1 += f32x4::splat(neon, scaled_deltas[1][prior]) * k;
            a2 += f32x4::splat(neon, scaled_deltas[2][prior]) * k;
            a3 += f32x4::splat(neon, scaled_deltas[3][prior]) * k;
        }
        a0.store_slice(&mut rows[0][offset..offset + 4]);
        a1.store_slice(&mut rows[1][offset..offset + 4]);
        a2.store_slice(&mut rows[2][offset..offset + 4]);
        a3.store_slice(&mut rows[3][offset..offset + 4]);
    }
    let rem_start = vec4_blocks * 4;
    if rem_start < width {
        for (r, out_row) in rows.iter_mut().enumerate() {
            for (idx, out_elem) in out_row.iter_mut().enumerate().take(width).skip(rem_start) {
                let mut acc = last_cumulative * states[r][idx];
                for (prior, input) in inputs[..count].iter().enumerate() {
                    acc += scaled_deltas[r][prior] * input.key[idx];
                }
                *out_elem = acc;
            }
        }
    }
    true
}

#[inline]
#[allow(
    clippy::suboptimal_flops,
    reason = "Preserve separate binary32 multiply and add roundings matching scale_slice_into and add_scaled_slice_in_place."
)]
pub(super) fn gdn_wy_update_eight_rows(
    states: [&[f32]; 8],
    last_cumulative: f32,
    inputs: &[super::pinned_gdn::ChunkInput<'_>],
    scaled_deltas: &[[f32; 32]; 8],
    matrix_rows: [&mut [f32]; 8],
) -> bool {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return false;
    };
    let width = matrix_rows[0].len();
    let count = inputs.len();
    let scale_vec = f32x4::splat(neon, last_cumulative);
    let vec16_blocks = width / 16;
    let [m0, m1, m2, m3, m4, m5, m6, m7] = matrix_rows;
    let mut rows = [m0, m1, m2, m3, m4, m5, m6, m7];
    for b in 0..vec16_blocks {
        let offset = b * 16;
        let mut acc = [[f32x4::splat(neon, 0.0); 4]; 8];
        for r in 0..8 {
            let s = &states[r][offset..offset + 16];
            acc[r] = [
                scale_vec * f32x4::from_slice(neon, &s[0..4]),
                scale_vec * f32x4::from_slice(neon, &s[4..8]),
                scale_vec * f32x4::from_slice(neon, &s[8..12]),
                scale_vec * f32x4::from_slice(neon, &s[12..16]),
            ];
        }
        for (prior, input) in inputs[..count].iter().enumerate() {
            let kb = &input.key[offset..offset + 16];
            let k0 = f32x4::from_slice(neon, &kb[0..4]);
            let k1 = f32x4::from_slice(neon, &kb[4..8]);
            let k2 = f32x4::from_slice(neon, &kb[8..12]);
            let k3 = f32x4::from_slice(neon, &kb[12..16]);
            for r in 0..8 {
                let sd = f32x4::splat(neon, scaled_deltas[r][prior]);
                acc[r][0] += sd * k0;
                acc[r][1] += sd * k1;
                acc[r][2] += sd * k2;
                acc[r][3] += sd * k3;
            }
        }
        for (out_row, row_acc) in rows.iter_mut().zip(acc.iter()) {
            let out_block = &mut out_row[offset..offset + 16];
            row_acc[0].store_slice(&mut out_block[0..4]);
            row_acc[1].store_slice(&mut out_block[4..8]);
            row_acc[2].store_slice(&mut out_block[8..12]);
            row_acc[3].store_slice(&mut out_block[12..16]);
        }
    }
    let vec4_blocks = width / 4;
    for b in (vec16_blocks * 4)..vec4_blocks {
        let offset = b * 4;
        let mut a: [f32x4<Neon>; 8] = std::array::from_fn(|r| {
            scale_vec * f32x4::from_slice(neon, &states[r][offset..offset + 4])
        });
        for (prior, input) in inputs[..count].iter().enumerate() {
            let k = f32x4::from_slice(neon, &input.key[offset..offset + 4]);
            for r in 0..8 {
                a[r] += f32x4::splat(neon, scaled_deltas[r][prior]) * k;
            }
        }
        for (r, out_row) in rows.iter_mut().enumerate() {
            a[r].store_slice(&mut out_row[offset..offset + 4]);
        }
    }
    let rem_start = vec4_blocks * 4;
    if rem_start < width {
        for (r, out_row) in rows.iter_mut().enumerate() {
            for (idx, out_elem) in out_row.iter_mut().enumerate().take(width).skip(rem_start) {
                let mut acc = last_cumulative * states[r][idx];
                for (prior, input) in inputs[..count].iter().enumerate() {
                    acc += scaled_deltas[r][prior] * input.key[idx];
                }
                *out_elem = acc;
            }
        }
    }
    true
}

#[inline]
#[allow(
    clippy::cast_possible_truncation,
    clippy::as_conversions,
    reason = "The pinned scalar numerical contract explicitly rounds this value to binary32."
)]
pub(super) fn l2_norm_in_place(input: &mut [f32], epsilon: f32) -> bool {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return false;
    };
    let mut squares = 0.0_f64;
    let (chunks, tail) = input.as_chunks::<4>();
    for chunk in chunks {
        let v = f32x4::from_slice(neon, chunk);
        let sq_lanes = (v * v).to_array();
        squares += f64::from(sq_lanes[0]);
        squares += f64::from(sq_lanes[1]);
        squares += f64::from(sq_lanes[2]);
        squares += f64::from(sq_lanes[3]);
    }
    for &value in tail {
        squares += f64::from(value * value);
    }
    let scale = 1.0 / (squares as f32).sqrt().max(epsilon);
    let scale_vec = f32x4::splat(neon, scale);
    let (chunks, tail) = input.as_chunks_mut::<4>();
    for chunk in chunks {
        (f32x4::from_slice(neon, chunk) * scale_vec).store_slice(chunk);
    }
    for value in tail {
        *value *= scale;
    }
    true
}

#[inline]
pub(super) fn rms_norm_squares(input: &[f32], weight: &[f32]) -> Option<Result<f64>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let mut sum = 0.0_f64;
    let mut finite_mask = f32x4::splat(neon, 0.0).is_finite();
    let (in_chunks, in_tail) = input.as_chunks::<4>();
    let (w_chunks, w_tail) = weight.as_chunks::<4>();
    for (in_chunk, w_chunk) in in_chunks.iter().zip(w_chunks) {
        let v = f32x4::from_slice(neon, in_chunk);
        let w = f32x4::from_slice(neon, w_chunk);
        finite_mask = finite_mask & v.is_finite() & w.is_finite();
        let sq_lanes = (v * v).to_array();
        sum += f64::from(sq_lanes[0]);
        sum += f64::from(sq_lanes[1]);
        sum += f64::from(sq_lanes[2]);
        sum += f64::from(sq_lanes[3]);
    }
    if !finite_mask.all_true() {
        return Some(Err(crate::invalid("nonfinite tensor or activation value")));
    }
    for (&value, &w) in in_tail.iter().zip(w_tail) {
        if !value.is_finite() || !w.is_finite() {
            return Some(Err(crate::invalid("nonfinite tensor or activation value")));
        }
        sum += f64::from(value * value);
    }
    Some(Ok(sum))
}

#[inline]
pub(super) fn add_rms_norm_squares(
    values: &mut [f32],
    delta: &[f32],
    weight: &[f32],
) -> Option<Result<f64>> {
    if values.len() != delta.len() || values.len() != weight.len() {
        return None;
    }
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let mut sum = 0.0_f64;
    let mut finite_mask = f32x4::splat(neon, 0.0).is_finite();
    let (val_chunks, val_tail) = values.as_chunks_mut::<4>();
    let (d_chunks, d_tail) = delta.as_chunks::<4>();
    let (w_chunks, w_tail) = weight.as_chunks::<4>();
    for ((val_chunk, d_chunk), w_chunk) in val_chunks.iter_mut().zip(d_chunks).zip(w_chunks) {
        let v = f32x4::from_slice(neon, val_chunk) + f32x4::from_slice(neon, d_chunk);
        v.store_slice(val_chunk);
        let w = f32x4::from_slice(neon, w_chunk);
        finite_mask = finite_mask & v.is_finite() & w.is_finite();
        let sq_lanes = (v * v).to_array();
        sum += f64::from(sq_lanes[0]);
        sum += f64::from(sq_lanes[1]);
        sum += f64::from(sq_lanes[2]);
        sum += f64::from(sq_lanes[3]);
    }
    let mut tail_finite = true;
    for ((val, &d), &w) in val_tail.iter_mut().zip(d_tail).zip(w_tail) {
        let v = *val + d;
        *val = v;
        tail_finite &= v.is_finite() && w.is_finite();
        sum += f64::from(v * v);
    }
    if !finite_mask.all_true() || !tail_finite {
        return Some(Err(crate::invalid("nonfinite tensor or activation value")));
    }
    Some(Ok(sum))
}

#[inline]
#[allow(
    clippy::suboptimal_flops,
    reason = "Preserve separate binary32 product roundings for both RMS contracts."
)]
pub(super) fn rms_norm_scale_into<const PINNED: bool>(
    input: &[f32],
    weight: &[f32],
    scale: f32,
    output: &mut Vec<f32>,
) -> Option<Result<()>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    if output.len() != input.len() {
        output.clear();
        if let Err(err) = output.try_reserve_exact(input.len()) {
            return Some(Err(err.into()));
        }
        output.resize(input.len(), 0.0);
    }
    let scale_vec = f32x4::splat(neon, scale);
    let mut finite_mask = f32x4::splat(neon, 0.0).is_finite();
    let (in_chunks, in_tail) = input.as_chunks::<4>();
    let (w_chunks, w_tail) = weight.as_chunks::<4>();
    let (out_chunks, out_tail) = output.as_chunks_mut::<4>();
    for ((in_chunk, w_chunk), out_chunk) in in_chunks.iter().zip(w_chunks).zip(out_chunks) {
        let v = f32x4::from_slice(neon, in_chunk);
        let w = f32x4::from_slice(neon, w_chunk);
        let out = if PINNED {
            (v * w) * scale_vec
        } else {
            (v * scale_vec) * w
        };
        finite_mask &= out.is_finite();
        out.store_slice(out_chunk);
    }
    if !finite_mask.all_true() {
        return Some(Err(crate::invalid("nonfinite tensor or activation value")));
    }
    for ((&value, &w), out) in in_tail.iter().zip(w_tail).zip(out_tail) {
        let scaled = if PINNED {
            (value * w) * scale
        } else {
            (value * scale) * w
        };
        if !scaled.is_finite() {
            return Some(Err(crate::invalid("nonfinite tensor or activation value")));
        }
        *out = scaled;
    }
    Some(Ok(()))
}

#[cfg(test)]
#[inline]
#[allow(
    clippy::suboptimal_flops,
    reason = "Preserve separate binary32 product roundings for both RMS contracts."
)]
pub(super) fn rms_norm_scale_in_place<const PINNED: bool>(
    values: &mut [f32],
    weight: &[f32],
    scale: f32,
) -> Option<Result<()>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let scale_vec = f32x4::splat(neon, scale);
    let mut finite_mask = f32x4::splat(neon, 0.0).is_finite();
    let (val_chunks, val_tail) = values.as_chunks_mut::<4>();
    let (w_chunks, w_tail) = weight.as_chunks::<4>();
    for (val_chunk, w_chunk) in val_chunks.iter_mut().zip(w_chunks) {
        let v = f32x4::from_slice(neon, val_chunk);
        let w = f32x4::from_slice(neon, w_chunk);
        let out = if PINNED {
            (v * w) * scale_vec
        } else {
            (v * scale_vec) * w
        };
        finite_mask &= out.is_finite();
        out.store_slice(val_chunk);
    }
    if !finite_mask.all_true() {
        return Some(Err(crate::invalid("nonfinite tensor or activation value")));
    }
    for (value, &w) in val_tail.iter_mut().zip(w_tail) {
        let scaled = if PINNED {
            (*value * w) * scale
        } else {
            (*value * scale) * w
        };
        if !scaled.is_finite() {
            return Some(Err(crate::invalid("nonfinite tensor or activation value")));
        }
        *value = scaled;
    }
    Some(Ok(()))
}

#[inline]
#[allow(
    clippy::suboptimal_flops,
    reason = "Preserve separate binary32 product roundings for pinned RMSNorm followed by pinned SwiGLU."
)]
pub(super) fn rms_norm_scale_pinned_swiglu_in_place(
    values: &mut [f32],
    weight: &[f32],
    scale: f32,
    gate: &[f32],
) -> Option<Result<()>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let scale_vec = f32x4::splat(neon, scale);
    let zero = f32x4::splat(neon, 0.0);
    let one = f32x4::splat(neon, 1.0);
    let mut finite_mask = zero.is_finite();
    let (val_chunks, val_tail) = values.as_chunks_mut::<4>();
    let (w_chunks, w_tail) = weight.as_chunks::<4>();
    let (g_chunks, g_tail) = gate.as_chunks::<4>();
    for ((val_chunk, w_chunk), g_chunk) in val_chunks.iter_mut().zip(w_chunks).zip(g_chunks) {
        let v = f32x4::from_slice(neon, val_chunk);
        let w = f32x4::from_slice(neon, w_chunk);
        let g = f32x4::from_slice(neon, g_chunk);
        let norm = (v * w) * scale_vec;
        finite_mask &= norm.is_finite();
        let out = (g * norm) / (one + super::pinned_activations::vector_exp_x4(neon, zero - g));
        out.store_slice(val_chunk);
    }
    if !finite_mask.all_true() {
        return Some(Err(crate::invalid("nonfinite tensor or activation value")));
    }
    for ((value, &w), &g) in val_tail.iter_mut().zip(w_tail).zip(g_tail) {
        let norm = (*value * w) * scale;
        if !norm.is_finite() {
            return Some(Err(crate::invalid("nonfinite tensor or activation value")));
        }
        *value = (g * norm) / (1.0 + (-g).exp());
    }
    Some(Ok(()))
}

#[inline]
#[allow(
    clippy::suboptimal_flops,
    reason = "Preserve separate binary32 product roundings for pinned RMSNorm followed by pinned SwiGLU."
)]
pub(super) fn rms_norm_scale_pinned_swiglu_into(
    input: &[f32],
    weight: &[f32],
    scale: f32,
    gate: &[f32],
    output: &mut [f32],
) -> Option<Result<()>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let scale_vec = f32x4::splat(neon, scale);
    let zero = f32x4::splat(neon, 0.0);
    let one = f32x4::splat(neon, 1.0);
    let mut finite_mask = zero.is_finite();
    let (in_chunks, in_tail) = input.as_chunks::<4>();
    let (w_chunks, w_tail) = weight.as_chunks::<4>();
    let (g_chunks, g_tail) = gate.as_chunks::<4>();
    let (out_chunks, out_tail) = output.as_chunks_mut::<4>();
    for (((in_chunk, w_chunk), g_chunk), out_chunk) in
        in_chunks.iter().zip(w_chunks).zip(g_chunks).zip(out_chunks)
    {
        let v = f32x4::from_slice(neon, in_chunk);
        let w = f32x4::from_slice(neon, w_chunk);
        let g = f32x4::from_slice(neon, g_chunk);
        let norm = (v * w) * scale_vec;
        finite_mask &= norm.is_finite();
        let out = (g * norm) / (one + super::pinned_activations::vector_exp_x4(neon, zero - g));
        out.store_slice(out_chunk);
    }
    if !finite_mask.all_true() {
        return Some(Err(crate::invalid("nonfinite tensor or activation value")));
    }
    for (((&value, &w), &g), out) in in_tail.iter().zip(w_tail).zip(g_tail).zip(out_tail) {
        let norm = (value * w) * scale;
        if !norm.is_finite() {
            return Some(Err(crate::invalid("nonfinite tensor or activation value")));
        }
        *out = (g * norm) / (1.0 + (-g).exp());
    }
    Some(Ok(()))
}

#[inline]
#[allow(
    clippy::suboptimal_flops,
    reason = "Preserve separate binary32 product roundings for RMSNorm and Source RoPE while fusing PinnedSwift FMA RoPE and binary16 rounding."
)]
pub(super) fn rms_norm_rope_round_f16_256_into<const PINNED: bool>(
    input: &[f32; 256],
    weight: &[f32; 256],
    scale: f32,
    frequencies: &[(f32, f32); 32],
    output: &mut [f32; 256],
) -> Option<Result<bool>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let scale_vec = f32x4::splat(neon, scale);
    let zero_mask = f32x4::splat(neon, 0.0).is_finite();
    let mut norm_finite = zero_mask;
    let mut rope_finite = zero_mask;
    let mut f16_finite = zero_mask;
    for chunk in 0..8 {
        let base = chunk * 4;
        let x0 = f32x4::from_slice(neon, &input[base..base + 4]);
        let w0 = f32x4::from_slice(neon, &weight[base..base + 4]);
        let x1 = f32x4::from_slice(neon, &input[32 + base..36 + base]);
        let w1 = f32x4::from_slice(neon, &weight[32 + base..36 + base]);
        let (first, second) = if PINNED {
            ((x0 * w0) * scale_vec, (x1 * w1) * scale_vec)
        } else {
            ((x0 * scale_vec) * w0, (x1 * scale_vec) * w1)
        };
        norm_finite = norm_finite & first.is_finite() & second.is_finite();
        let cos = f32x4::load_array(
            neon,
            [
                frequencies[base].0,
                frequencies[base + 1].0,
                frequencies[base + 2].0,
                frequencies[base + 3].0,
            ],
        );
        let sin = f32x4::load_array(
            neon,
            [
                frequencies[base].1,
                frequencies[base + 1].1,
                frequencies[base + 2].1,
                frequencies[base + 3].1,
            ],
        );
        let (rot0, rot1) = if PINNED {
            (
                first.mul_add_precise(cos, -(second * sin)),
                second.mul_add_precise(cos, first * sin),
            )
        } else {
            (
                (first * cos) - (second * sin),
                (first * sin) + (second * cos),
            )
        };
        rope_finite = rope_finite & rot0.is_finite() & rot1.is_finite();
        let r0 = round_f16_x4(neon, rot0);
        let r1 = round_f16_x4(neon, rot1);
        f16_finite = f16_finite & r0.is_finite() & r1.is_finite();
        r0.store_slice(&mut output[base..base + 4]);
        r1.store_slice(&mut output[32 + base..36 + base]);
    }
    for ((in_chunk, w_chunk), out_chunk) in input[64..]
        .as_chunks::<4>()
        .0
        .iter()
        .zip(weight[64..].as_chunks::<4>().0)
        .zip(output[64..].as_chunks_mut::<4>().0)
    {
        let v = f32x4::from_slice(neon, in_chunk);
        let w = f32x4::from_slice(neon, w_chunk);
        let norm = if PINNED {
            (v * w) * scale_vec
        } else {
            (v * scale_vec) * w
        };
        norm_finite &= norm.is_finite();
        rope_finite &= norm.is_finite();
        let rounded = round_f16_x4(neon, norm);
        f16_finite &= rounded.is_finite();
        rounded.store_slice(out_chunk);
    }
    if !norm_finite.all_true() {
        return Some(Err(crate::invalid("nonfinite tensor or activation value")));
    }
    if !rope_finite.all_true() {
        return Some(Err(crate::invalid(if PINNED {
            "nonfinite pinned Swift RoPE output"
        } else {
            "nonfinite values in RoPE output"
        })));
    }
    Some(Ok(f16_finite.all_true()))
}

#[inline]
#[allow(
    clippy::suboptimal_flops,
    reason = "Preserve separate binary32 product roundings for RMSNorm and Source RoPE while fusing PinnedSwift FMA RoPE and binary16 rounding."
)]
pub(super) fn rms_norm_rope_round_f16_256_in_place<const PINNED: bool>(
    values: &mut [f32; 256],
    weight: &[f32; 256],
    scale: f32,
    frequencies: &[(f32, f32); 32],
) -> Option<Result<bool>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let scale_vec = f32x4::splat(neon, scale);
    let zero_mask = f32x4::splat(neon, 0.0).is_finite();
    let mut norm_finite = zero_mask;
    let mut rope_finite = zero_mask;
    let mut f16_finite = zero_mask;
    let (rotary, suffix) = values.split_at_mut(64);
    let (first_half, second_half) = rotary.split_at_mut(32);
    for (chunk, (first_chunk, second_chunk)) in first_half
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(second_half.as_chunks_mut::<4>().0)
        .enumerate()
    {
        let base = chunk * 4;
        let x0 = f32x4::from_slice(neon, first_chunk);
        let w0 = f32x4::from_slice(neon, &weight[base..base + 4]);
        let x1 = f32x4::from_slice(neon, second_chunk);
        let w1 = f32x4::from_slice(neon, &weight[32 + base..36 + base]);
        let (first, second) = if PINNED {
            ((x0 * w0) * scale_vec, (x1 * w1) * scale_vec)
        } else {
            ((x0 * scale_vec) * w0, (x1 * scale_vec) * w1)
        };
        norm_finite = norm_finite & first.is_finite() & second.is_finite();
        let cos = f32x4::load_array(
            neon,
            [
                frequencies[base].0,
                frequencies[base + 1].0,
                frequencies[base + 2].0,
                frequencies[base + 3].0,
            ],
        );
        let sin = f32x4::load_array(
            neon,
            [
                frequencies[base].1,
                frequencies[base + 1].1,
                frequencies[base + 2].1,
                frequencies[base + 3].1,
            ],
        );
        let (rot0, rot1) = if PINNED {
            (
                first.mul_add_precise(cos, -(second * sin)),
                second.mul_add_precise(cos, first * sin),
            )
        } else {
            (
                (first * cos) - (second * sin),
                (first * sin) + (second * cos),
            )
        };
        rope_finite = rope_finite & rot0.is_finite() & rot1.is_finite();
        let r0 = round_f16_x4(neon, rot0);
        let r1 = round_f16_x4(neon, rot1);
        f16_finite = f16_finite & r0.is_finite() & r1.is_finite();
        r0.store_slice(first_chunk);
        r1.store_slice(second_chunk);
    }
    for (val_chunk, w_chunk) in suffix
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(weight[64..].as_chunks::<4>().0)
    {
        let v = f32x4::from_slice(neon, val_chunk);
        let w = f32x4::from_slice(neon, w_chunk);
        let norm = if PINNED {
            (v * w) * scale_vec
        } else {
            (v * scale_vec) * w
        };
        norm_finite &= norm.is_finite();
        rope_finite &= norm.is_finite();
        let rounded = round_f16_x4(neon, norm);
        f16_finite &= rounded.is_finite();
        rounded.store_slice(val_chunk);
    }
    if !norm_finite.all_true() {
        return Some(Err(crate::invalid("nonfinite tensor or activation value")));
    }
    if !rope_finite.all_true() {
        return Some(Err(crate::invalid(if PINNED {
            "nonfinite pinned Swift RoPE output"
        } else {
            "nonfinite values in RoPE output"
        })));
    }
    Some(Ok(f16_finite.all_true()))
}

#[inline]
#[allow(
    clippy::too_many_arguments,
    reason = "Passes the eight pre-widened activation planes directly in NEON registers."
)]
fn pq2_half_signed_sum(
    w: u8x16<Neon>,
    p0_low: i16x8<Neon>,
    p0_high: i16x8<Neon>,
    p1_low: i16x8<Neon>,
    p1_high: i16x8<Neon>,
    p2_low: i16x8<Neon>,
    p2_high: i16x8<Neon>,
    p3_low: i16x8<Neon>,
    p3_high: i16x8<Neon>,
) -> i16x8<Neon> {
    let (u_low, u_high) = w.widen();
    let w_low = u_low.bitcast::<i16x8<Neon>>();
    let w_high = u_high.bitcast::<i16x8<Neon>>();
    let s0 = ((w_low & 3) - 1) * p0_low + ((w_high & 3) - 1) * p0_high;
    let s1 = (((w_low >> 2) & 3) - 1) * p1_low + (((w_high >> 2) & 3) - 1) * p1_high;
    let s2 = (((w_low >> 4) & 3) - 1) * p2_low + (((w_high >> 4) & 3) - 1) * p2_high;
    let s3 = ((w_low >> 6) - 1) * p3_low + ((w_high >> 6) - 1) * p3_high;
    (s0 + s1) + (s2 + s3)
}

/// Consumes scoped digit-plane activations without unpack interleaves per row.
///
/// Each sixteen packed bytes describe sixty-four consecutive weights. Their
/// low/high two-bit digits pair directly with activation planes prepared once
/// in `packed_pq2`. Two independent i16 accumulators retain all four symbols,
/// including +2 and activation -128. Each accumulated lane is -2048..=2032;
/// combining them gives -4096..=4064 before widening for the i32 reduction.
/// No floating arithmetic or optional ISA is used here.
#[inline]
pub(super) fn dot_consumption_order(packed: &[u8; 32], activations: &[i8; 128]) -> i32 {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return packed
            .as_chunks::<16>()
            .0
            .iter()
            .zip(activations.as_chunks::<64>().0)
            .map(|(bytes, values)| {
                bytes
                    .iter()
                    .enumerate()
                    .map(|(index, byte)| {
                        (0..4)
                            .map(|digit| {
                                (i32::from((byte >> (digit * 2)) & 3) - 1)
                                    * i32::from(values[digit * 16 + index])
                            })
                            .sum::<i32>()
                    })
                    .sum::<i32>()
            })
            .sum();
    };
    let mut acc = i16x8::splat(neon, 0);
    for (bytes, values) in packed
        .as_chunks::<16>()
        .0
        .iter()
        .zip(activations.as_chunks::<64>().0)
    {
        let w = u8x16::from_slice(neon, bytes);
        let planes = values.as_chunks::<16>().0;
        let (p0_low, p0_high) = i8x16::from_slice(neon, &planes[0]).widen();
        let (p1_low, p1_high) = i8x16::from_slice(neon, &planes[1]).widen();
        let (p2_low, p2_high) = i8x16::from_slice(neon, &planes[2]).widen();
        let (p3_low, p3_high) = i8x16::from_slice(neon, &planes[3]).widen();
        acc += pq2_half_signed_sum(
            w, p0_low, p0_high, p1_low, p1_high, p2_low, p2_high, p3_low, p3_high,
        );
    }
    pq2_reduce(acc)
}

/// Evaluates four 128-weight `PQ2` blocks against shared pre-widened activation planes.
#[inline]
pub(super) fn dot_consumption_order_four(
    packed: [&[u8; 32]; 4],
    activations: &[i8; 128],
) -> [i32; 4] {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return packed.map(|row| dot_consumption_order(row, activations));
    };
    let mut acc = [i16x8::splat(neon, 0); 4];
    for (half, values) in activations.as_chunks::<64>().0.iter().enumerate() {
        let planes = values.as_chunks::<16>().0;
        let (p0_low, p0_high) = i8x16::from_slice(neon, &planes[0]).widen();
        let (p1_low, p1_high) = i8x16::from_slice(neon, &planes[1]).widen();
        let (p2_low, p2_high) = i8x16::from_slice(neon, &planes[2]).widen();
        let (p3_low, p3_high) = i8x16::from_slice(neon, &planes[3]).widen();
        for row in 0..4 {
            let bytes = &packed[row].as_chunks::<16>().0[half];
            let w = u8x16::from_slice(neon, bytes);
            acc[row] += pq2_half_signed_sum(
                w, p0_low, p0_high, p1_low, p1_high, p2_low, p2_high, p3_low, p3_high,
            );
        }
    }
    reduce4_i16x8_to_i32x4(acc[0], acc[1], acc[2], acc[3]).to_array()
}

#[inline]
fn dot32(neon: Neon, low: i8x16<Neon>, high: i8x16<Neon>, values: &[i8; 32]) -> i32 {
    let (w0_l, w0_h) = low.widen();
    let (w1_l, w1_h) = high.widen();
    let (v0_l, v0_h) = i8x16::from_slice(neon, &values[..16]).widen();
    let (v1_l, v1_h) = i8x16::from_slice(neon, &values[16..]).widen();
    let sum = (w0_l * v0_l + w0_h * v0_h) + (w1_l * v1_l + w1_h * v1_h);
    let (sum_l, sum_h) = sum.widen();
    (sum_l + sum_h).reduce_sum()
}

/// Retains four independent `Q8_0` integer sums, each bounded by 8,192.
#[inline]
pub(super) fn dot_groups(packed: &[u8; 32], activations: [&[i8; 32]; 4]) -> [i32; 4] {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return std::array::from_fn(|group| {
            activations[group]
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    let byte = packed[group * 8 + index / 4];
                    let weight = i32::from((byte >> (2 * (index % 4))) & 3) - 1;
                    weight * i32::from(*value)
                })
                .sum()
        });
    };
    let mut sums = [0; 4];
    for (half, bytes) in packed.as_chunks::<16>().0.iter().enumerate() {
        let [a, b, c, d] = decode(u8x16::from_slice(neon, bytes));
        sums[half * 2] = dot32(neon, a, b, activations[half * 2]);
        sums[half * 2 + 1] = dot32(neon, c, d, activations[half * 2 + 1]);
    }
    sums
}

#[inline]
const fn ptq1_scale_u16(block: &[u8; 28]) -> u16 {
    u16::from_le_bytes([block[26], block[27]])
}

#[inline]
#[allow(
    clippy::too_many_arguments,
    reason = "Passes the four 16-bit weight halves and four 16-bit activation halves in registers."
)]
fn dot32_w16(
    w0_l: i16x8<Neon>,
    w0_h: i16x8<Neon>,
    w1_l: i16x8<Neon>,
    w1_h: i16x8<Neon>,
    v0_l: i16x8<Neon>,
    v0_h: i16x8<Neon>,
    v1_l: i16x8<Neon>,
    v1_h: i16x8<Neon>,
) -> i32 {
    let sum = (w0_l * v0_l + w0_h * v0_h) + (w1_l * v1_l + w1_h * v1_h);
    let (sum_l, sum_h) = sum.widen();
    (sum_l + sum_h).reduce_sum()
}

#[inline]
#[allow(
    clippy::too_many_arguments,
    reason = "Passes the four 16-bit weight halves and four 16-bit activation halves in registers."
)]
fn dot32_native_lanes_w16_i32x4(
    w0_l: i16x8<Neon>,
    w0_h: i16x8<Neon>,
    w1_l: i16x8<Neon>,
    w1_h: i16x8<Neon>,
    v0_l: i16x8<Neon>,
    v0_h: i16x8<Neon>,
    v1_l: i16x8<Neon>,
    v1_h: i16x8<Neon>,
) -> i32x4<Neon> {
    let (a, b) = (w0_l * v0_l + w1_l * v1_l).widen();
    let (c, d) = (w0_h * v0_h + w1_h * v1_h).widen();
    reduce4_i32x4(a, b, c, d)
}

#[inline]
fn extract_trits_u8x16(v: u8x16<Neon>) -> i8x16<Neon> {
    // Overflow-free u8 halving average: (v & (v >> 1)) + ((v ^ (v >> 1)) >> 1) == (3 * v) >> 2.
    // Shifting right by 6 yields (3 * v) >> 8 in 0..=2 for all v in 0..=255 without widening.
    let b = v >> 1;
    let avg = (v & b) + ((v ^ b) >> 1);
    (avg >> 6).bitcast::<i8x16<Neon>>() - 1
}

#[inline]
pub(in crate::kernels) fn decode_ptq1_i8(neon: Neon, packed: &[u8; 28]) -> [i8x16<Neon>; 8] {
    let qs = u8x16::from_slice(neon, &packed[..16]);
    let t0 = extract_trits_u8x16(qs);
    let t1 = extract_trits_u8x16(qs * u8x16::splat(neon, 3));
    let t2 = extract_trits_u8x16(qs * u8x16::splat(neon, 9));
    let t3 = extract_trits_u8x16(qs * u8x16::splat(neon, 27));
    let t4 = extract_trits_u8x16(qs * u8x16::splat(neon, 81));

    let qm_u64 = u64::from_le_bytes([
        packed[16], packed[17], packed[18], packed[19], packed[20], packed[21], packed[22],
        packed[23],
    ]);
    let qm_dup = u64x2::splat(neon, qm_u64).bitcast::<u8x16<Neon>>();
    let m5 = u8x16::from_slice(neon, &[1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3]);
    let m6 = u8x16::from_slice(
        neon,
        &[9, 9, 9, 9, 9, 9, 9, 9, 27, 27, 27, 27, 27, 27, 27, 27],
    );
    let t5 = extract_trits_u8x16(qm_dup * m5);
    let t6 = extract_trits_u8x16(qm_dup * m6);

    let qh_pair = u16::from_le_bytes([packed[24], packed[25]]);
    let tail_u64 = u64::from(qh_pair) * 0x0001_0001_0001_0001_u64;
    let qm_qh = u64x2::from_slice(neon, &[qm_u64, tail_u64]).bitcast::<u8x16<Neon>>();
    let m7 = u8x16::from_slice(
        neon,
        &[81, 81, 81, 81, 81, 81, 81, 81, 1, 1, 3, 3, 9, 9, 27, 27],
    );
    let t7 = extract_trits_u8x16(qm_qh * m7);

    [t0, t1, t2, t3, t4, t5, t6, t7]
}

#[inline]
pub(in crate::kernels) fn decode_ptq1_i16(neon: Neon, packed: &[u8; 28]) -> [i16x8<Neon>; 16] {
    let [t0, t1, t2, t3, t4, t5, t6, t7] = decode_ptq1_i8(neon, packed);
    let (w0, w1) = t0.widen();
    let (w2, w3) = t1.widen();
    let (w4, w5) = t2.widen();
    let (w6, w7) = t3.widen();
    let (w8, w9) = t4.widen();
    let (w10, w11) = t5.widen();
    let (w12, w13) = t6.widen();
    let (w14, w15) = t7.widen();
    [
        w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15,
    ]
}

#[inline]
#[allow(
    clippy::too_many_arguments,
    reason = "Streams the four 32-element PTQ1 groups against pre-widened Q8_0 activations without stack spills."
)]
fn accumulate_ptq1_block_pinned(
    neon: Neon,
    packed: &[u8; 28],
    v00_l: i16x8<Neon>,
    v00_h: i16x8<Neon>,
    v01_l: i16x8<Neon>,
    v01_h: i16x8<Neon>,
    v10_l: i16x8<Neon>,
    v10_h: i16x8<Neon>,
    v11_l: i16x8<Neon>,
    v11_h: i16x8<Neon>,
    v20_l: i16x8<Neon>,
    v20_h: i16x8<Neon>,
    v21_l: i16x8<Neon>,
    v21_h: i16x8<Neon>,
    v30_l: i16x8<Neon>,
    v30_h: i16x8<Neon>,
    v31_l: i16x8<Neon>,
    v31_h: i16x8<Neon>,
    row_scale: f32,
    a0: f32,
    a1: f32,
    a2: f32,
    a3: f32,
    mut lane: f32x4<Neon>,
) -> f32x4<Neon> {
    let [
        w0,
        w1,
        w2,
        w3,
        w4,
        w5,
        w6,
        w7,
        w8,
        w9,
        w10,
        w11,
        w12,
        w13,
        w14,
        w15,
    ] = decode_ptq1_i16(neon, packed);
    let f0 = dot32_native_lanes_w16_i32x4(w0, w1, w2, w3, v00_l, v00_h, v01_l, v01_h)
        .to_float::<f32x4<Neon>>();
    lane = f0.mul_add_precise(f32x4::splat(neon, row_scale * a0), lane);
    let f1 = dot32_native_lanes_w16_i32x4(w4, w5, w6, w7, v10_l, v10_h, v11_l, v11_h)
        .to_float::<f32x4<Neon>>();
    lane = f1.mul_add_precise(f32x4::splat(neon, row_scale * a1), lane);
    let f2 = dot32_native_lanes_w16_i32x4(w8, w9, w10, w11, v20_l, v20_h, v21_l, v21_h)
        .to_float::<f32x4<Neon>>();
    lane = f2.mul_add_precise(f32x4::splat(neon, row_scale * a2), lane);
    let f3 = dot32_native_lanes_w16_i32x4(w12, w13, w14, w15, v30_l, v30_h, v31_l, v31_h)
        .to_float::<f32x4<Neon>>();
    f3.mul_add_precise(f32x4::splat(neon, row_scale * a3), lane)
}

#[inline]
#[allow(
    clippy::too_many_arguments,
    reason = "Streams the four 32-element PTQ1 groups via ARMv8.2-A SDOT while preserving exact lane FMA order."
)]
fn accumulate_ptq1_block_pinned_dotprod(
    neon: Neon,
    packed: &[u8; 28],
    act_values: [&[i8; 32]; 4],
    row_scale: f32,
    a0: f32,
    a1: f32,
    a2: f32,
    a3: f32,
    mut lane: f32x4<Neon>,
) -> f32x4<Neon> {
    let group_lanes = bonsai_neon_dot::dot_ptq1_block_lanes(packed, act_values);
    let f0 = i32x4::from_slice(neon, &group_lanes[0]).to_float::<f32x4<Neon>>();
    lane = f0.mul_add_precise(f32x4::splat(neon, row_scale * a0), lane);
    let f1 = i32x4::from_slice(neon, &group_lanes[1]).to_float::<f32x4<Neon>>();
    lane = f1.mul_add_precise(f32x4::splat(neon, row_scale * a1), lane);
    let f2 = i32x4::from_slice(neon, &group_lanes[2]).to_float::<f32x4<Neon>>();
    lane = f2.mul_add_precise(f32x4::splat(neon, row_scale * a2), lane);
    let f3 = i32x4::from_slice(neon, &group_lanes[3]).to_float::<f32x4<Neon>>();
    f3.mul_add_precise(f32x4::splat(neon, row_scale * a3), lane)
}

/// Keeps each native PTQ1 DOTPROD lane separate until floating accumulation.
#[inline]
pub(super) fn dot_ptq1_native_lanes(
    packed: &[u8; 28],
    activations: [&[i8; 32]; 4],
) -> Result<[[i32; 4]; 4]> {
    if bonsai_neon_dot::has_dotprod() {
        return Ok(bonsai_neon_dot::dot_ptq1_block_lanes(packed, activations));
    }
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        let weights = super::ptq1_values(packed)?;
        return Ok(std::array::from_fn(|group| {
            super::q8_native_lanes(&weights[group * 32..group * 32 + 32], activations[group])
        }));
    };
    let decoded = decode_ptq1_i16(neon, packed);
    Ok(std::array::from_fn(|group| {
        let values = activations[group];
        let (v0_l, v0_h) = i8x16::from_slice(neon, &values[..16]).widen();
        let (v1_l, v1_h) = i8x16::from_slice(neon, &values[16..]).widen();
        dot32_native_lanes_w16_i32x4(
            decoded[group * 4],
            decoded[group * 4 + 1],
            decoded[group * 4 + 2],
            decoded[group * 4 + 3],
            v0_l,
            v0_h,
            v1_l,
            v1_h,
        )
        .to_array()
    }))
}

/// Decodes all native PTQ1 bytes directly, without a temporary PQ2 block.
///
/// Bytes 0..16 contain five stride-16 digits, 16..24 five stride-8 digits,
/// and 24..26 four stride-2 digits. Every load stays within the 28-byte block.
/// Four independent sums preserve the `Q8_0` scale boundaries. Ternary products
/// fit i16 and each 32-element sum has absolute magnitude at most 4,096.
#[inline]
pub(super) fn dot_ptq1_groups(packed: &[u8; 28], activations: [&[i8; 32]; 4]) -> Result<[i32; 4]> {
    if bonsai_neon_dot::has_dotprod() {
        return Ok(bonsai_neon_dot::dot_ptq1_block_groups(packed, activations));
    }
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        let weights = super::ptq1_values(packed)?;
        return Ok(std::array::from_fn(|group| {
            weights[group * 32..group * 32 + 32]
                .iter()
                .zip(activations[group])
                .map(|(&weight, &value)| i32::from(weight) * i32::from(value))
                .sum()
        }));
    };
    let decoded = decode_ptq1_i16(neon, packed);
    Ok(std::array::from_fn(|group| {
        let values = activations[group];
        let (v0_l, v0_h) = i8x16::from_slice(neon, &values[..16]).widen();
        let (v1_l, v1_h) = i8x16::from_slice(neon, &values[16..]).widen();
        dot32_w16(
            decoded[group * 4],
            decoded[group * 4 + 1],
            decoded[group * 4 + 2],
            decoded[group * 4 + 3],
            v0_l,
            v0_h,
            v1_l,
            v1_h,
        )
    }))
}

#[inline]
pub(super) fn dot_ptq1_row_pinned(bytes: &[u8], input: &[super::Q8Block]) -> Result<f32> {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return super::dot_q8_groups_pinned::<false>(super::PTQ1, bytes, input);
    };
    let use_dotprod = bonsai_neon_dot::has_dotprod();
    let mut lanes = f32x4::splat(neon, 0.0);
    for (index, block) in bytes.as_chunks::<28>().0.iter().enumerate() {
        let activations = <&[super::Q8Block; 4]>::try_from(&input[index * 4..index * 4 + 4])?;
        let weight_scale = super::scale(&block[26..])?;
        if use_dotprod {
            lanes = accumulate_ptq1_block_pinned_dotprod(
                neon,
                block,
                [
                    &activations[0].values,
                    &activations[1].values,
                    &activations[2].values,
                    &activations[3].values,
                ],
                weight_scale,
                activations[0].scale,
                activations[1].scale,
                activations[2].scale,
                activations[3].scale,
                lanes,
            );
        } else {
            let (v00_l, v00_h) = i8x16::from_slice(neon, &activations[0].values[..16]).widen();
            let (v01_l, v01_h) = i8x16::from_slice(neon, &activations[0].values[16..]).widen();
            let (v10_l, v10_h) = i8x16::from_slice(neon, &activations[1].values[..16]).widen();
            let (v11_l, v11_h) = i8x16::from_slice(neon, &activations[1].values[16..]).widen();
            let (v20_l, v20_h) = i8x16::from_slice(neon, &activations[2].values[..16]).widen();
            let (v21_l, v21_h) = i8x16::from_slice(neon, &activations[2].values[16..]).widen();
            let (v30_l, v30_h) = i8x16::from_slice(neon, &activations[3].values[..16]).widen();
            let (v31_l, v31_h) = i8x16::from_slice(neon, &activations[3].values[16..]).widen();
            lanes = accumulate_ptq1_block_pinned(
                neon,
                block,
                v00_l,
                v00_h,
                v01_l,
                v01_h,
                v10_l,
                v10_h,
                v11_l,
                v11_h,
                v20_l,
                v20_h,
                v21_l,
                v21_h,
                v30_l,
                v30_h,
                v31_l,
                v31_h,
                weight_scale,
                activations[0].scale,
                activations[1].scale,
                activations[2].scale,
                activations[3].scale,
                lanes,
            );
        }
    }
    let [l0, l1, l2, l3] = lanes.to_array();
    super::finite((l0 + l1) + (l2 + l3))
}

#[inline]
#[allow(
    clippy::too_many_lines,
    reason = "Dispatches between ARMv8.2-A SDOT and pre-widened baseline NEON across four rows."
)]
pub(super) fn dot_ptq1_four_rows_pinned(
    rows: [&[u8]; 4],
    input: &[super::Q8Block],
) -> Result<[f32; 4]> {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return Ok([
            super::dot_q8_groups_pinned::<false>(super::PTQ1, rows[0], input)?,
            super::dot_q8_groups_pinned::<false>(super::PTQ1, rows[1], input)?,
            super::dot_q8_groups_pinned::<false>(super::PTQ1, rows[2], input)?,
            super::dot_q8_groups_pinned::<false>(super::PTQ1, rows[3], input)?,
        ]);
    };
    let use_dotprod = bonsai_neon_dot::has_dotprod();
    let mut lanes = [f32x4::splat(neon, 0.0); 4];
    let mut scales_finite = lanes[0].is_finite();
    let r0 = rows[0].as_chunks::<28>().0;
    let r1 = rows[1].as_chunks::<28>().0;
    let r2 = rows[2].as_chunks::<28>().0;
    let r3 = rows[3].as_chunks::<28>().0;
    for (index, (((b0, b1), b2), b3)) in r0.iter().zip(r1).zip(r2).zip(r3).enumerate() {
        let activations = <&[super::Q8Block; 4]>::try_from(&input[index * 4..index * 4 + 4])?;
        let scales_vec = f16_bits_to_f32_x4(
            neon,
            [
                ptq1_scale_u16(b0),
                ptq1_scale_u16(b1),
                ptq1_scale_u16(b2),
                ptq1_scale_u16(b3),
            ],
        );
        scales_finite &= scales_vec.is_finite();
        let row_scales = scales_vec.to_array();
        let a0 = activations[0].scale;
        let a1 = activations[1].scale;
        let a2 = activations[2].scale;
        let a3 = activations[3].scale;
        let blocks = [b0, b1, b2, b3];
        if use_dotprod {
            let act_values = [
                &activations[0].values,
                &activations[1].values,
                &activations[2].values,
                &activations[3].values,
            ];
            let all_lanes = bonsai_neon_dot::dot_ptq1_four_rows_lanes(blocks, act_values);
            for (r, row_lanes) in all_lanes.iter().enumerate() {
                let rs = row_scales[r];
                let mut lane = lanes[r];
                let f0 = i32x4::from_slice(neon, &row_lanes[0]).to_float::<f32x4<Neon>>();
                lane = f0.mul_add_precise(f32x4::splat(neon, rs * a0), lane);
                let f1 = i32x4::from_slice(neon, &row_lanes[1]).to_float::<f32x4<Neon>>();
                lane = f1.mul_add_precise(f32x4::splat(neon, rs * a1), lane);
                let f2 = i32x4::from_slice(neon, &row_lanes[2]).to_float::<f32x4<Neon>>();
                lane = f2.mul_add_precise(f32x4::splat(neon, rs * a2), lane);
                let f3 = i32x4::from_slice(neon, &row_lanes[3]).to_float::<f32x4<Neon>>();
                lanes[r] = f3.mul_add_precise(f32x4::splat(neon, rs * a3), lane);
            }
        } else {
            let (v00_l, v00_h) = i8x16::from_slice(neon, &activations[0].values[..16]).widen();
            let (v01_l, v01_h) = i8x16::from_slice(neon, &activations[0].values[16..]).widen();
            let (v10_l, v10_h) = i8x16::from_slice(neon, &activations[1].values[..16]).widen();
            let (v11_l, v11_h) = i8x16::from_slice(neon, &activations[1].values[16..]).widen();
            let (v20_l, v20_h) = i8x16::from_slice(neon, &activations[2].values[..16]).widen();
            let (v21_l, v21_h) = i8x16::from_slice(neon, &activations[2].values[16..]).widen();
            let (v30_l, v30_h) = i8x16::from_slice(neon, &activations[3].values[..16]).widen();
            let (v31_l, v31_h) = i8x16::from_slice(neon, &activations[3].values[16..]).widen();
            for r in 0..4 {
                lanes[r] = accumulate_ptq1_block_pinned(
                    neon,
                    blocks[r],
                    v00_l,
                    v00_h,
                    v01_l,
                    v01_h,
                    v10_l,
                    v10_h,
                    v11_l,
                    v11_h,
                    v20_l,
                    v20_h,
                    v21_l,
                    v21_h,
                    v30_l,
                    v30_h,
                    v31_l,
                    v31_h,
                    row_scales[r],
                    a0,
                    a1,
                    a2,
                    a3,
                    lanes[r],
                );
            }
        }
    }
    if !scales_finite.all_true() {
        return Err(crate::invalid("nonfinite tensor or activation value"));
    }
    let mut out = [0.0_f32; 4];
    for r in 0..4 {
        let [l0, l1, l2, l3] = lanes[r].to_array();
        out[r] = super::finite((l0 + l1) + (l2 + l3))?;
    }
    Ok(out)
}

#[inline]
#[allow(
    clippy::too_many_lines,
    reason = "Dispatches between ARMv8.2-A SDOT and pre-widened baseline NEON across eight rows."
)]
pub(super) fn dot_ptq1_eight_rows_pinned(
    rows: [&[u8]; 8],
    input: &[super::Q8Block],
) -> Result<[f32; 8]> {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        let first = dot_ptq1_four_rows_pinned([rows[0], rows[1], rows[2], rows[3]], input)?;
        let second = dot_ptq1_four_rows_pinned([rows[4], rows[5], rows[6], rows[7]], input)?;
        return Ok([
            first[0], first[1], first[2], first[3], second[0], second[1], second[2], second[3],
        ]);
    };
    let use_dotprod = bonsai_neon_dot::has_dotprod();
    let mut lanes = [f32x4::splat(neon, 0.0); 8];
    let mut scales_finite = lanes[0].is_finite();
    let chunks = rows.map(|row| row.as_chunks::<28>().0);
    for index in 0..chunks[0].len() {
        let activations = <&[super::Q8Block; 4]>::try_from(&input[index * 4..index * 4 + 4])?;
        let (scales_lo, scales_hi) = f16_bits_to_f32_x8(
            neon,
            &[
                ptq1_scale_u16(&chunks[0][index]),
                ptq1_scale_u16(&chunks[1][index]),
                ptq1_scale_u16(&chunks[2][index]),
                ptq1_scale_u16(&chunks[3][index]),
                ptq1_scale_u16(&chunks[4][index]),
                ptq1_scale_u16(&chunks[5][index]),
                ptq1_scale_u16(&chunks[6][index]),
                ptq1_scale_u16(&chunks[7][index]),
            ],
        );
        scales_finite = scales_finite & scales_lo.is_finite() & scales_hi.is_finite();
        let scales_lo_arr = scales_lo.to_array();
        let scales_hi_arr = scales_hi.to_array();
        let row_scales = [
            scales_lo_arr[0],
            scales_lo_arr[1],
            scales_lo_arr[2],
            scales_lo_arr[3],
            scales_hi_arr[0],
            scales_hi_arr[1],
            scales_hi_arr[2],
            scales_hi_arr[3],
        ];
        let a0 = activations[0].scale;
        let a1 = activations[1].scale;
        let a2 = activations[2].scale;
        let a3 = activations[3].scale;
        if use_dotprod {
            let act_values = [
                &activations[0].values,
                &activations[1].values,
                &activations[2].values,
                &activations[3].values,
            ];
            let blocks = [
                &chunks[0][index],
                &chunks[1][index],
                &chunks[2][index],
                &chunks[3][index],
                &chunks[4][index],
                &chunks[5][index],
                &chunks[6][index],
                &chunks[7][index],
            ];
            let all_lanes = bonsai_neon_dot::dot_ptq1_eight_rows_lanes(blocks, act_values);
            for (r, row_lanes) in all_lanes.iter().enumerate() {
                let rs = row_scales[r];
                let mut lane = lanes[r];
                let f0 = i32x4::from_slice(neon, &row_lanes[0]).to_float::<f32x4<Neon>>();
                lane = f0.mul_add_precise(f32x4::splat(neon, rs * a0), lane);
                let f1 = i32x4::from_slice(neon, &row_lanes[1]).to_float::<f32x4<Neon>>();
                lane = f1.mul_add_precise(f32x4::splat(neon, rs * a1), lane);
                let f2 = i32x4::from_slice(neon, &row_lanes[2]).to_float::<f32x4<Neon>>();
                lane = f2.mul_add_precise(f32x4::splat(neon, rs * a2), lane);
                let f3 = i32x4::from_slice(neon, &row_lanes[3]).to_float::<f32x4<Neon>>();
                lanes[r] = f3.mul_add_precise(f32x4::splat(neon, rs * a3), lane);
            }
        } else {
            let (v00_l, v00_h) = i8x16::from_slice(neon, &activations[0].values[..16]).widen();
            let (v01_l, v01_h) = i8x16::from_slice(neon, &activations[0].values[16..]).widen();
            let (v10_l, v10_h) = i8x16::from_slice(neon, &activations[1].values[..16]).widen();
            let (v11_l, v11_h) = i8x16::from_slice(neon, &activations[1].values[16..]).widen();
            let (v20_l, v20_h) = i8x16::from_slice(neon, &activations[2].values[..16]).widen();
            let (v21_l, v21_h) = i8x16::from_slice(neon, &activations[2].values[16..]).widen();
            let (v30_l, v30_h) = i8x16::from_slice(neon, &activations[3].values[..16]).widen();
            let (v31_l, v31_h) = i8x16::from_slice(neon, &activations[3].values[16..]).widen();
            for r in 0..8 {
                lanes[r] = accumulate_ptq1_block_pinned(
                    neon,
                    &chunks[r][index],
                    v00_l,
                    v00_h,
                    v01_l,
                    v01_h,
                    v10_l,
                    v10_h,
                    v11_l,
                    v11_h,
                    v20_l,
                    v20_h,
                    v21_l,
                    v21_h,
                    v30_l,
                    v30_h,
                    v31_l,
                    v31_h,
                    row_scales[r],
                    a0,
                    a1,
                    a2,
                    a3,
                    lanes[r],
                );
            }
        }
    }
    if !scales_finite.all_true() {
        return Err(crate::invalid("nonfinite tensor or activation value"));
    }
    let mut out = [0.0_f32; 8];
    for r in 0..8 {
        let [l0, l1, l2, l3] = lanes[r].to_array();
        out[r] = super::finite((l0 + l1) + (l2 + l3))?;
    }
    Ok(out)
}

#[inline]
#[allow(
    clippy::too_many_arguments,
    clippy::suboptimal_flops,
    reason = "Streams the four 32-element PTQ1 groups while preserving scalar reference multiply-then-add rounding."
)]
fn accumulate_ptq1_block_source(
    neon: Neon,
    packed: &[u8; 28],
    v00_l: i16x8<Neon>,
    v00_h: i16x8<Neon>,
    v01_l: i16x8<Neon>,
    v01_h: i16x8<Neon>,
    v10_l: i16x8<Neon>,
    v10_h: i16x8<Neon>,
    v11_l: i16x8<Neon>,
    v11_h: i16x8<Neon>,
    v20_l: i16x8<Neon>,
    v20_h: i16x8<Neon>,
    v21_l: i16x8<Neon>,
    v21_h: i16x8<Neon>,
    v30_l: i16x8<Neon>,
    v30_h: i16x8<Neon>,
    v31_l: i16x8<Neon>,
    v31_h: i16x8<Neon>,
    row_scale: f32,
    act_scales: f32x4<Neon>,
    output: f32,
) -> f32 {
    let [
        w0,
        w1,
        w2,
        w3,
        w4,
        w5,
        w6,
        w7,
        w8,
        w9,
        w10,
        w11,
        w12,
        w13,
        w14,
        w15,
    ] = decode_ptq1_i16(neon, packed);
    let (g0_l, g0_h) = ((w0 * v00_l + w1 * v00_h) + (w2 * v01_l + w3 * v01_h)).widen();
    let s0 = g0_l + g0_h;
    let (g1_l, g1_h) = ((w4 * v10_l + w5 * v10_h) + (w6 * v11_l + w7 * v11_h)).widen();
    let s1 = g1_l + g1_h;
    let (g2_l, g2_h) = ((w8 * v20_l + w9 * v20_h) + (w10 * v21_l + w11 * v21_h)).widen();
    let s2 = g2_l + g2_h;
    let (g3_l, g3_h) = ((w12 * v30_l + w13 * v30_h) + (w14 * v31_l + w15 * v31_h)).widen();
    let s3 = g3_l + g3_h;
    let prod = (reduce4_i32x4(s0, s1, s2, s3).to_float::<f32x4<Neon>>() * act_scales).to_array();
    let sum = (((0.0 + prod[0]) + prod[1]) + prod[2]) + prod[3];
    output + row_scale * sum
}

#[inline]
#[allow(
    clippy::suboptimal_flops,
    reason = "Streams the four 32-element PTQ1 groups via ARMv8.2-A SDOT while preserving scalar reference multiply-then-add rounding."
)]
fn accumulate_ptq1_block_source_dotprod(
    neon: Neon,
    packed: &[u8; 28],
    act_values: [&[i8; 32]; 4],
    row_scale: f32,
    act_scales: f32x4<Neon>,
    output: f32,
) -> f32 {
    let groups = bonsai_neon_dot::dot_ptq1_block_groups(packed, act_values);
    let prod = (i32x4::from_slice(neon, &groups).to_float::<f32x4<Neon>>() * act_scales).to_array();
    let sum = (((0.0 + prod[0]) + prod[1]) + prod[2]) + prod[3];
    output + row_scale * sum
}

#[inline]
pub(super) fn dot_ptq1_row_source(bytes: &[u8], input: &[super::Q8Block]) -> Result<f32> {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return super::dot_q8_groups(super::PTQ1, bytes, input);
    };
    let use_dotprod = bonsai_neon_dot::has_dotprod();
    let mut output = 0.0_f32;
    for (index, block) in bytes.as_chunks::<28>().0.iter().enumerate() {
        let activations = <&[super::Q8Block; 4]>::try_from(&input[index * 4..index * 4 + 4])?;
        let weight_scale = super::scale(&block[26..])?;
        let act_scales = f32x4::from_slice(
            neon,
            &[
                activations[0].scale,
                activations[1].scale,
                activations[2].scale,
                activations[3].scale,
            ],
        );
        if use_dotprod {
            output = accumulate_ptq1_block_source_dotprod(
                neon,
                block,
                [
                    &activations[0].values,
                    &activations[1].values,
                    &activations[2].values,
                    &activations[3].values,
                ],
                weight_scale,
                act_scales,
                output,
            );
        } else {
            let (v00_l, v00_h) = i8x16::from_slice(neon, &activations[0].values[..16]).widen();
            let (v01_l, v01_h) = i8x16::from_slice(neon, &activations[0].values[16..]).widen();
            let (v10_l, v10_h) = i8x16::from_slice(neon, &activations[1].values[..16]).widen();
            let (v11_l, v11_h) = i8x16::from_slice(neon, &activations[1].values[16..]).widen();
            let (v20_l, v20_h) = i8x16::from_slice(neon, &activations[2].values[..16]).widen();
            let (v21_l, v21_h) = i8x16::from_slice(neon, &activations[2].values[16..]).widen();
            let (v30_l, v30_h) = i8x16::from_slice(neon, &activations[3].values[..16]).widen();
            let (v31_l, v31_h) = i8x16::from_slice(neon, &activations[3].values[16..]).widen();
            output = accumulate_ptq1_block_source(
                neon,
                block,
                v00_l,
                v00_h,
                v01_l,
                v01_h,
                v10_l,
                v10_h,
                v11_l,
                v11_h,
                v20_l,
                v20_h,
                v21_l,
                v21_h,
                v30_l,
                v30_h,
                v31_l,
                v31_h,
                weight_scale,
                act_scales,
                output,
            );
        }
    }
    super::finite(output)
}

#[inline]
#[allow(
    clippy::suboptimal_flops,
    reason = "Preserve scalar reference multiply-then-add block accumulation order."
)]
pub(super) fn dot_ptq1_four_rows_source(
    rows: [&[u8]; 4],
    input: &[super::Q8Block],
) -> Result<[f32; 4]> {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return Ok([
            super::dot_q8_groups(super::PTQ1, rows[0], input)?,
            super::dot_q8_groups(super::PTQ1, rows[1], input)?,
            super::dot_q8_groups(super::PTQ1, rows[2], input)?,
            super::dot_q8_groups(super::PTQ1, rows[3], input)?,
        ]);
    };
    let use_dotprod = bonsai_neon_dot::has_dotprod();
    let mut outputs = [0.0_f32; 4];
    let mut scales_finite = f32x4::splat(neon, 0.0).is_finite();
    let r0 = rows[0].as_chunks::<28>().0;
    let r1 = rows[1].as_chunks::<28>().0;
    let r2 = rows[2].as_chunks::<28>().0;
    let r3 = rows[3].as_chunks::<28>().0;
    for (index, (((b0, b1), b2), b3)) in r0.iter().zip(r1).zip(r2).zip(r3).enumerate() {
        let activations = <&[super::Q8Block; 4]>::try_from(&input[index * 4..index * 4 + 4])?;
        let scales_vec = f16_bits_to_f32_x4(
            neon,
            [
                ptq1_scale_u16(b0),
                ptq1_scale_u16(b1),
                ptq1_scale_u16(b2),
                ptq1_scale_u16(b3),
            ],
        );
        scales_finite &= scales_vec.is_finite();
        let row_scales = scales_vec.to_array();
        let act_scales = f32x4::from_slice(
            neon,
            &[
                activations[0].scale,
                activations[1].scale,
                activations[2].scale,
                activations[3].scale,
            ],
        );
        let blocks = [b0, b1, b2, b3];
        if use_dotprod {
            let act_values = [
                &activations[0].values,
                &activations[1].values,
                &activations[2].values,
                &activations[3].values,
            ];
            let all_groups = bonsai_neon_dot::dot_ptq1_four_rows_groups(blocks, act_values);
            for (r, groups) in all_groups.iter().enumerate() {
                let prod = (i32x4::from_slice(neon, groups).to_float::<f32x4<Neon>>() * act_scales)
                    .to_array();
                let sum = (((0.0 + prod[0]) + prod[1]) + prod[2]) + prod[3];
                outputs[r] += row_scales[r] * sum;
            }
        } else {
            let (v00_l, v00_h) = i8x16::from_slice(neon, &activations[0].values[..16]).widen();
            let (v01_l, v01_h) = i8x16::from_slice(neon, &activations[0].values[16..]).widen();
            let (v10_l, v10_h) = i8x16::from_slice(neon, &activations[1].values[..16]).widen();
            let (v11_l, v11_h) = i8x16::from_slice(neon, &activations[1].values[16..]).widen();
            let (v20_l, v20_h) = i8x16::from_slice(neon, &activations[2].values[..16]).widen();
            let (v21_l, v21_h) = i8x16::from_slice(neon, &activations[2].values[16..]).widen();
            let (v30_l, v30_h) = i8x16::from_slice(neon, &activations[3].values[..16]).widen();
            let (v31_l, v31_h) = i8x16::from_slice(neon, &activations[3].values[16..]).widen();
            for r in 0..4 {
                outputs[r] = accumulate_ptq1_block_source(
                    neon,
                    blocks[r],
                    v00_l,
                    v00_h,
                    v01_l,
                    v01_h,
                    v10_l,
                    v10_h,
                    v11_l,
                    v11_h,
                    v20_l,
                    v20_h,
                    v21_l,
                    v21_h,
                    v30_l,
                    v30_h,
                    v31_l,
                    v31_h,
                    row_scales[r],
                    act_scales,
                    outputs[r],
                );
            }
        }
    }
    if !scales_finite.all_true() {
        return Err(crate::invalid("nonfinite tensor or activation value"));
    }
    for out in &mut outputs {
        *out = super::finite(*out)?;
    }
    Ok(outputs)
}

#[inline]
#[allow(
    clippy::suboptimal_flops,
    clippy::too_many_lines,
    reason = "Dispatches between ARMv8.2-A SDOT and pre-widened baseline NEON across eight rows."
)]
pub(super) fn dot_ptq1_eight_rows_source(
    rows: [&[u8]; 8],
    input: &[super::Q8Block],
) -> Result<[f32; 8]> {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        let first = dot_ptq1_four_rows_source([rows[0], rows[1], rows[2], rows[3]], input)?;
        let second = dot_ptq1_four_rows_source([rows[4], rows[5], rows[6], rows[7]], input)?;
        return Ok([
            first[0], first[1], first[2], first[3], second[0], second[1], second[2], second[3],
        ]);
    };
    let use_dotprod = bonsai_neon_dot::has_dotprod();
    let mut outputs = [0.0_f32; 8];
    let mut scales_finite = f32x4::splat(neon, 0.0).is_finite();
    let chunks = rows.map(|row| row.as_chunks::<28>().0);
    for index in 0..chunks[0].len() {
        let activations = <&[super::Q8Block; 4]>::try_from(&input[index * 4..index * 4 + 4])?;
        let (scales_lo, scales_hi) = f16_bits_to_f32_x8(
            neon,
            &[
                ptq1_scale_u16(&chunks[0][index]),
                ptq1_scale_u16(&chunks[1][index]),
                ptq1_scale_u16(&chunks[2][index]),
                ptq1_scale_u16(&chunks[3][index]),
                ptq1_scale_u16(&chunks[4][index]),
                ptq1_scale_u16(&chunks[5][index]),
                ptq1_scale_u16(&chunks[6][index]),
                ptq1_scale_u16(&chunks[7][index]),
            ],
        );
        scales_finite = scales_finite & scales_lo.is_finite() & scales_hi.is_finite();
        let scales_lo_arr = scales_lo.to_array();
        let scales_hi_arr = scales_hi.to_array();
        let row_scales = [
            scales_lo_arr[0],
            scales_lo_arr[1],
            scales_lo_arr[2],
            scales_lo_arr[3],
            scales_hi_arr[0],
            scales_hi_arr[1],
            scales_hi_arr[2],
            scales_hi_arr[3],
        ];
        let act_scales = f32x4::from_slice(
            neon,
            &[
                activations[0].scale,
                activations[1].scale,
                activations[2].scale,
                activations[3].scale,
            ],
        );
        if use_dotprod {
            let act_values = [
                &activations[0].values,
                &activations[1].values,
                &activations[2].values,
                &activations[3].values,
            ];
            let blocks = [
                &chunks[0][index],
                &chunks[1][index],
                &chunks[2][index],
                &chunks[3][index],
                &chunks[4][index],
                &chunks[5][index],
                &chunks[6][index],
                &chunks[7][index],
            ];
            let all_groups = bonsai_neon_dot::dot_ptq1_eight_rows_groups(blocks, act_values);
            for (r, groups) in all_groups.iter().enumerate() {
                let prod = (i32x4::from_slice(neon, groups).to_float::<f32x4<Neon>>() * act_scales)
                    .to_array();
                let sum = (((0.0 + prod[0]) + prod[1]) + prod[2]) + prod[3];
                outputs[r] += row_scales[r] * sum;
            }
        } else {
            let (v00_l, v00_h) = i8x16::from_slice(neon, &activations[0].values[..16]).widen();
            let (v01_l, v01_h) = i8x16::from_slice(neon, &activations[0].values[16..]).widen();
            let (v10_l, v10_h) = i8x16::from_slice(neon, &activations[1].values[..16]).widen();
            let (v11_l, v11_h) = i8x16::from_slice(neon, &activations[1].values[16..]).widen();
            let (v20_l, v20_h) = i8x16::from_slice(neon, &activations[2].values[..16]).widen();
            let (v21_l, v21_h) = i8x16::from_slice(neon, &activations[2].values[16..]).widen();
            let (v30_l, v30_h) = i8x16::from_slice(neon, &activations[3].values[..16]).widen();
            let (v31_l, v31_h) = i8x16::from_slice(neon, &activations[3].values[16..]).widen();
            for r in 0..8 {
                outputs[r] = accumulate_ptq1_block_source(
                    neon,
                    &chunks[r][index],
                    v00_l,
                    v00_h,
                    v01_l,
                    v01_h,
                    v10_l,
                    v10_h,
                    v11_l,
                    v11_h,
                    v20_l,
                    v20_h,
                    v21_l,
                    v21_h,
                    v30_l,
                    v30_h,
                    v31_l,
                    v31_h,
                    row_scales[r],
                    act_scales,
                    outputs[r],
                );
            }
        }
    }
    if !scales_finite.all_true() {
        return Err(crate::invalid("nonfinite tensor or activation value"));
    }
    for out in &mut outputs {
        *out = super::finite(*out)?;
    }
    Ok(outputs)
}

#[inline]
#[allow(
    clippy::suboptimal_flops,
    reason = "Preserve scalar reference multiply-then-add rounding."
)]
pub(super) fn dot_q8_0_four_rows(rows: [&[u8]; 4], input: &[super::Q8Block]) -> Result<[f32; 4]> {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return Ok([
            super::dot_legacy(super::Q8_0, rows[0], input)?,
            super::dot_legacy(super::Q8_0, rows[1], input)?,
            super::dot_legacy(super::Q8_0, rows[2], input)?,
            super::dot_legacy(super::Q8_0, rows[3], input)?,
        ]);
    };
    let use_dotprod = bonsai_neon_dot::has_dotprod();
    let mut outputs = [0.0_f32; 4];
    let mut scales_finite = f32x4::splat(neon, 0.0).is_finite();
    let chunks = rows.map(|row| row.as_chunks::<34>().0);
    for (index, activation) in input.iter().enumerate() {
        let b0 = &chunks[0][index];
        let b1 = &chunks[1][index];
        let b2 = &chunks[2][index];
        let b3 = &chunks[3][index];
        let scales_vec = f16_bits_to_f32_x4(
            neon,
            [
                u16::from_le_bytes([b0[0], b0[1]]),
                u16::from_le_bytes([b1[0], b1[1]]),
                u16::from_le_bytes([b2[0], b2[1]]),
                u16::from_le_bytes([b3[0], b3[1]]),
            ],
        );
        scales_finite &= scales_vec.is_finite();
        let row_scales = scales_vec.to_array();
        let blocks = [b0, b1, b2, b3];
        if use_dotprod {
            for r in 0..4 {
                let [_, _, ref weights @ ..] = *blocks[r];
                let sum = super::integer_float(bonsai_neon_dot::dot_q8_0_block(
                    weights,
                    &activation.values,
                ));
                outputs[r] += (row_scales[r] * activation.scale) * sum;
            }
        } else {
            let (a0_l, a0_h) = i8x16::from_slice(neon, &activation.values[..16]).widen();
            let (a1_l, a1_h) = i8x16::from_slice(neon, &activation.values[16..]).widen();
            for r in 0..4 {
                let [_, _, ref weights @ ..] = *blocks[r];
                let sum = super::integer_float(dot_q8_0_block_preloaded(
                    neon, weights, a0_l, a0_h, a1_l, a1_h,
                ));
                outputs[r] += (row_scales[r] * activation.scale) * sum;
            }
        }
    }
    if !scales_finite.all_true() {
        return Err(crate::invalid("nonfinite tensor or activation value"));
    }
    for out in &mut outputs {
        *out = super::finite(*out)?;
    }
    Ok(outputs)
}

#[inline]
#[allow(
    clippy::suboptimal_flops,
    reason = "Preserve scalar reference multiply-then-add rounding."
)]
pub(super) fn dot_q8_0_eight_rows(rows: [&[u8]; 8], input: &[super::Q8Block]) -> Result<[f32; 8]> {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        let first = dot_q8_0_four_rows([rows[0], rows[1], rows[2], rows[3]], input)?;
        let second = dot_q8_0_four_rows([rows[4], rows[5], rows[6], rows[7]], input)?;
        return Ok([
            first[0], first[1], first[2], first[3], second[0], second[1], second[2], second[3],
        ]);
    };
    let use_dotprod = bonsai_neon_dot::has_dotprod();
    let mut outputs = [0.0_f32; 8];
    let mut scales_finite = f32x4::splat(neon, 0.0).is_finite();
    let chunks = rows.map(|row| row.as_chunks::<34>().0);
    for (index, activation) in input.iter().enumerate() {
        let (scales_lo, scales_hi) = f16_bits_to_f32_x8(
            neon,
            &[
                u16::from_le_bytes([chunks[0][index][0], chunks[0][index][1]]),
                u16::from_le_bytes([chunks[1][index][0], chunks[1][index][1]]),
                u16::from_le_bytes([chunks[2][index][0], chunks[2][index][1]]),
                u16::from_le_bytes([chunks[3][index][0], chunks[3][index][1]]),
                u16::from_le_bytes([chunks[4][index][0], chunks[4][index][1]]),
                u16::from_le_bytes([chunks[5][index][0], chunks[5][index][1]]),
                u16::from_le_bytes([chunks[6][index][0], chunks[6][index][1]]),
                u16::from_le_bytes([chunks[7][index][0], chunks[7][index][1]]),
            ],
        );
        scales_finite = scales_finite & scales_lo.is_finite() & scales_hi.is_finite();
        let scales_lo_arr = scales_lo.to_array();
        let scales_hi_arr = scales_hi.to_array();
        let row_scales = [
            scales_lo_arr[0],
            scales_lo_arr[1],
            scales_lo_arr[2],
            scales_lo_arr[3],
            scales_hi_arr[0],
            scales_hi_arr[1],
            scales_hi_arr[2],
            scales_hi_arr[3],
        ];
        if use_dotprod {
            for r in 0..8 {
                let [_, _, ref weights @ ..] = chunks[r][index];
                let sum = super::integer_float(bonsai_neon_dot::dot_q8_0_block(
                    weights,
                    &activation.values,
                ));
                outputs[r] += (row_scales[r] * activation.scale) * sum;
            }
        } else {
            let (a0_l, a0_h) = i8x16::from_slice(neon, &activation.values[..16]).widen();
            let (a1_l, a1_h) = i8x16::from_slice(neon, &activation.values[16..]).widen();
            for r in 0..8 {
                let [_, _, ref weights @ ..] = chunks[r][index];
                let sum = super::integer_float(dot_q8_0_block_preloaded(
                    neon, weights, a0_l, a0_h, a1_l, a1_h,
                ));
                outputs[r] += (row_scales[r] * activation.scale) * sum;
            }
        }
    }
    if !scales_finite.all_true() {
        return Err(crate::invalid("nonfinite tensor or activation value"));
    }
    for out in &mut outputs {
        *out = super::finite(*out)?;
    }
    Ok(outputs)
}

#[inline]
#[allow(
    clippy::suboptimal_flops,
    reason = "Preserve scalar reference multiply-then-add rounding."
)]
pub(super) fn dot_q4_0_four_rows(rows: [&[u8]; 4], input: &[super::Q8Block]) -> Result<[f32; 4]> {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return Ok([
            super::dot_legacy(super::Q4_0, rows[0], input)?,
            super::dot_legacy(super::Q4_0, rows[1], input)?,
            super::dot_legacy(super::Q4_0, rows[2], input)?,
            super::dot_legacy(super::Q4_0, rows[3], input)?,
        ]);
    };
    let mut outputs = [0.0_f32; 4];
    let mut scales_finite = f32x4::splat(neon, 0.0).is_finite();
    let chunks = rows.map(|row| row.as_chunks::<18>().0);
    for (index, activation) in input.iter().enumerate() {
        let b0 = &chunks[0][index];
        let b1 = &chunks[1][index];
        let b2 = &chunks[2][index];
        let b3 = &chunks[3][index];
        let scales_vec = f16_bits_to_f32_x4(
            neon,
            [
                u16::from_le_bytes([b0[0], b0[1]]),
                u16::from_le_bytes([b1[0], b1[1]]),
                u16::from_le_bytes([b2[0], b2[1]]),
                u16::from_le_bytes([b3[0], b3[1]]),
            ],
        );
        scales_finite &= scales_vec.is_finite();
        let row_scales = scales_vec.to_array();
        let (a0_l, a0_h) = i8x16::from_slice(neon, &activation.values[..16]).widen();
        let (a1_l, a1_h) = i8x16::from_slice(neon, &activation.values[16..]).widen();
        let blocks = [b0, b1, b2, b3];
        for r in 0..4 {
            let packed: &[u8; 16] = blocks[r][2..18].try_into()?;
            let sum = super::integer_float(dot_q4_0_block_preloaded(
                neon, packed, a0_l, a0_h, a1_l, a1_h,
            ));
            outputs[r] += (row_scales[r] * activation.scale) * sum;
        }
    }
    if !scales_finite.all_true() {
        return Err(crate::invalid("nonfinite tensor or activation value"));
    }
    for out in &mut outputs {
        *out = super::finite(*out)?;
    }
    Ok(outputs)
}

#[inline]
#[allow(
    clippy::suboptimal_flops,
    reason = "Preserve scalar reference multiply-then-add rounding."
)]
pub(super) fn dot_q4_0_eight_rows(rows: [&[u8]; 8], input: &[super::Q8Block]) -> Result<[f32; 8]> {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        let first = dot_q4_0_four_rows([rows[0], rows[1], rows[2], rows[3]], input)?;
        let second = dot_q4_0_four_rows([rows[4], rows[5], rows[6], rows[7]], input)?;
        return Ok([
            first[0], first[1], first[2], first[3], second[0], second[1], second[2], second[3],
        ]);
    };
    let mut outputs = [0.0_f32; 8];
    let mut scales_finite = f32x4::splat(neon, 0.0).is_finite();
    let chunks = rows.map(|row| row.as_chunks::<18>().0);
    for (index, activation) in input.iter().enumerate() {
        let (scales_lo, scales_hi) = f16_bits_to_f32_x8(
            neon,
            &[
                u16::from_le_bytes([chunks[0][index][0], chunks[0][index][1]]),
                u16::from_le_bytes([chunks[1][index][0], chunks[1][index][1]]),
                u16::from_le_bytes([chunks[2][index][0], chunks[2][index][1]]),
                u16::from_le_bytes([chunks[3][index][0], chunks[3][index][1]]),
                u16::from_le_bytes([chunks[4][index][0], chunks[4][index][1]]),
                u16::from_le_bytes([chunks[5][index][0], chunks[5][index][1]]),
                u16::from_le_bytes([chunks[6][index][0], chunks[6][index][1]]),
                u16::from_le_bytes([chunks[7][index][0], chunks[7][index][1]]),
            ],
        );
        scales_finite = scales_finite & scales_lo.is_finite() & scales_hi.is_finite();
        let scales_lo_arr = scales_lo.to_array();
        let scales_hi_arr = scales_hi.to_array();
        let row_scales = [
            scales_lo_arr[0],
            scales_lo_arr[1],
            scales_lo_arr[2],
            scales_lo_arr[3],
            scales_hi_arr[0],
            scales_hi_arr[1],
            scales_hi_arr[2],
            scales_hi_arr[3],
        ];
        let (a0_l, a0_h) = i8x16::from_slice(neon, &activation.values[..16]).widen();
        let (a1_l, a1_h) = i8x16::from_slice(neon, &activation.values[16..]).widen();
        for r in 0..8 {
            let packed: &[u8; 16] = chunks[r][index][2..18].try_into()?;
            let sum = super::integer_float(dot_q4_0_block_preloaded(
                neon, packed, a0_l, a0_h, a1_l, a1_h,
            ));
            outputs[r] += (row_scales[r] * activation.scale) * sum;
        }
    }
    if !scales_finite.all_true() {
        return Err(crate::invalid("nonfinite tensor or activation value"));
    }
    for out in &mut outputs {
        *out = super::finite(*out)?;
    }
    Ok(outputs)
}

#[inline]
#[allow(
    clippy::suboptimal_flops,
    reason = "Preserve scalar reference multiply-then-add rounding."
)]
pub(super) fn dot_q8_0_tile_row<const TOKENS: usize>(
    row: &[u8],
    preparations: &[super::MatvecScratch],
) -> Option<Result<[f32; TOKENS]>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let use_i8mm = bonsai_neon_dot::has_i8mm();
    let use_dotprod = bonsai_neon_dot::has_dotprod();
    let mut sums = [0.0_f32; TOKENS];
    for (index, block) in row.as_chunks::<34>().0.iter().enumerate() {
        let weight_scale = match super::scale(block) {
            Ok(s) => s,
            Err(err) => return Some(Err(err)),
        };
        let [_, _, ref weights @ ..] = *block;
        if use_i8mm && preparations.len() >= 2 {
            let token_count = preparations.len().min(TOKENS);
            let mut t = 0;
            while t + 2 <= token_count {
                let a0 = &preparations[t].q8[index];
                let a1 = &preparations[t + 1].q8[index];
                let tile =
                    bonsai_neon_dot::smmla_q8_0_2x2(weights, weights, &a0.values, &a1.values);
                sums[t] += (weight_scale * a0.scale) * super::integer_float(tile[0][0]);
                sums[t + 1] += (weight_scale * a1.scale) * super::integer_float(tile[0][1]);
                t += 2;
            }
            if t < token_count {
                let activation = &preparations[t].q8[index];
                let int_dot = bonsai_neon_dot::dot_q8_0_block(weights, &activation.values);
                sums[t] += (weight_scale * activation.scale) * super::integer_float(int_dot);
            }
        } else if use_dotprod {
            for (sum, scratch) in sums.iter_mut().zip(preparations) {
                let activation = &scratch.q8[index];
                let int_dot = bonsai_neon_dot::dot_q8_0_block(weights, &activation.values);
                *sum += (weight_scale * activation.scale) * super::integer_float(int_dot);
            }
        } else {
            let (w0_l, w0_h) = u8x16::from_slice(neon, &block[2..18])
                .bitcast::<i8x16<Neon>>()
                .widen();
            let (w1_l, w1_h) = u8x16::from_slice(neon, &block[18..34])
                .bitcast::<i8x16<Neon>>()
                .widen();
            for (sum, scratch) in sums.iter_mut().zip(preparations) {
                let activation = &scratch.q8[index];
                let (a0_l, a0_h) = i8x16::from_slice(neon, &activation.values[..16]).widen();
                let (a1_l, a1_h) = i8x16::from_slice(neon, &activation.values[16..]).widen();
                let (p00_l, p00_h) = (w0_l * a0_l).widen();
                let (p01_l, p01_h) = (w0_h * a0_h).widen();
                let (p10_l, p10_h) = (w1_l * a1_l).widen();
                let (p11_l, p11_h) = (w1_h * a1_h).widen();
                let int_dot =
                    ((p00_l + p00_h) + (p01_l + p01_h) + (p10_l + p10_h) + (p11_l + p11_h))
                        .reduce_sum();
                *sum += (weight_scale * activation.scale) * super::integer_float(int_dot);
            }
        }
    }
    for sum in sums.iter_mut().take(preparations.len()) {
        match super::finite(*sum) {
            Ok(v) => *sum = v,
            Err(err) => return Some(Err(err)),
        }
    }
    Some(Ok(sums))
}

#[inline]
#[allow(
    clippy::suboptimal_flops,
    reason = "Preserve scalar reference multiply-then-add rounding."
)]
pub(super) fn dot_q4_0_tile_row<const TOKENS: usize>(
    row: &[u8],
    preparations: &[super::MatvecScratch],
) -> Option<Result<[f32; TOKENS]>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let mut sums = [0.0_f32; TOKENS];
    for (index, block) in row.as_chunks::<18>().0.iter().enumerate() {
        let weight_scale = match super::scale(block) {
            Ok(s) => s,
            Err(err) => return Some(Err(err)),
        };
        let raw = u8x16::from_slice(neon, &block[2..18]);
        let (w0_l, w0_h) = ((raw & 15).bitcast::<i8x16<Neon>>() - 8).widen();
        let (w1_l, w1_h) = ((raw >> 4).bitcast::<i8x16<Neon>>() - 8).widen();
        for (sum, scratch) in sums.iter_mut().zip(preparations) {
            let activation = &scratch.q8[index];
            let (a0_l, a0_h) = i8x16::from_slice(neon, &activation.values[..16]).widen();
            let (a1_l, a1_h) = i8x16::from_slice(neon, &activation.values[16..]).widen();
            let pairs = (w0_l * a0_l + w0_h * a0_h) + (w1_l * a1_l + w1_h * a1_h);
            *sum += (weight_scale * activation.scale) * super::integer_float(pq2_reduce(pairs));
        }
    }
    for sum in sums.iter_mut().take(preparations.len()) {
        match super::finite(*sum) {
            Ok(v) => *sum = v,
            Err(err) => return Some(Err(err)),
        }
    }
    Some(Ok(sums))
}

#[inline]
#[allow(
    clippy::suboptimal_flops,
    clippy::too_many_lines,
    reason = "Preserve scalar reference multiply-then-add rounding across 4-row Q8_0 I8MM/SDOT tiles."
)]
pub(super) fn dot_q8_0_four_rows_tile<const TOKENS: usize>(
    rows: [&[u8]; 4],
    preparations: &[super::MatvecScratch],
) -> Option<Result<[[f32; TOKENS]; 4]>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let use_i8mm = bonsai_neon_dot::has_i8mm();
    let use_dotprod = bonsai_neon_dot::has_dotprod();
    let mut sums = [[0.0_f32; TOKENS]; 4];
    let chunks = rows.map(|row| row.as_chunks::<34>().0);
    let block_count = chunks[0].len();
    let token_count = preparations.len().min(TOKENS);
    for index in 0..block_count {
        let b0 = &chunks[0][index];
        let b1 = &chunks[1][index];
        let b2 = &chunks[2][index];
        let b3 = &chunks[3][index];
        let scales_vec = f16_bits_to_f32_x4(
            neon,
            [
                u16::from_le_bytes([b0[0], b0[1]]),
                u16::from_le_bytes([b1[0], b1[1]]),
                u16::from_le_bytes([b2[0], b2[1]]),
                u16::from_le_bytes([b3[0], b3[1]]),
            ],
        );
        if !scales_vec.is_finite().all_true() {
            return Some(Err(crate::invalid("nonfinite tensor or activation value")));
        }
        let row_scales = scales_vec.to_array();
        let [_, _, ref w0 @ ..] = *b0;
        let [_, _, ref w1 @ ..] = *b1;
        let [_, _, ref w2 @ ..] = *b2;
        let [_, _, ref w3 @ ..] = *b3;
        let weights = [w0, w1, w2, w3];
        if use_i8mm && token_count >= 2 {
            let mut t = 0;
            while t + 2 <= token_count {
                let a0 = &preparations[t].q8[index];
                let a1 = &preparations[t + 1].q8[index];
                let tile01 = bonsai_neon_dot::smmla_q8_0_2x2(w0, w1, &a0.values, &a1.values);
                let tile23 = bonsai_neon_dot::smmla_q8_0_2x2(w2, w3, &a0.values, &a1.values);
                for (pair, tile) in [(0, tile01), (2, tile23)] {
                    sums[pair][t] +=
                        (row_scales[pair] * a0.scale) * super::integer_float(tile[0][0]);
                    sums[pair][t + 1] +=
                        (row_scales[pair] * a1.scale) * super::integer_float(tile[0][1]);
                    sums[pair + 1][t] +=
                        (row_scales[pair + 1] * a0.scale) * super::integer_float(tile[1][0]);
                    sums[pair + 1][t + 1] +=
                        (row_scales[pair + 1] * a1.scale) * super::integer_float(tile[1][1]);
                }
                t += 2;
            }
            if t < token_count {
                let activation = &preparations[t].q8[index];
                for r in 0..4 {
                    let int_dot = bonsai_neon_dot::dot_q8_0_block(weights[r], &activation.values);
                    sums[r][t] +=
                        (row_scales[r] * activation.scale) * super::integer_float(int_dot);
                }
            }
        } else if use_dotprod {
            for (t, scratch) in preparations.iter().take(token_count).enumerate() {
                let activation = &scratch.q8[index];
                for r in 0..4 {
                    let int_dot = bonsai_neon_dot::dot_q8_0_block(weights[r], &activation.values);
                    sums[r][t] +=
                        (row_scales[r] * activation.scale) * super::integer_float(int_dot);
                }
            }
        } else {
            let widened_w: [_; 4] = std::array::from_fn(|r| {
                let (w0_l, w0_h) = u8x16::from_slice(neon, &weights[r][..16])
                    .bitcast::<i8x16<Neon>>()
                    .widen();
                let (w1_l, w1_h) = u8x16::from_slice(neon, &weights[r][16..])
                    .bitcast::<i8x16<Neon>>()
                    .widen();
                (w0_l, w0_h, w1_l, w1_h)
            });
            for (t, scratch) in preparations.iter().take(token_count).enumerate() {
                let activation = &scratch.q8[index];
                let (a0_l, a0_h) = i8x16::from_slice(neon, &activation.values[..16]).widen();
                let (a1_l, a1_h) = i8x16::from_slice(neon, &activation.values[16..]).widen();
                for r in 0..4 {
                    let (w0_l, w0_h, w1_l, w1_h) = widened_w[r];
                    let (p00_l, p00_h) = (w0_l * a0_l).widen();
                    let (p01_l, p01_h) = (w0_h * a0_h).widen();
                    let (p10_l, p10_h) = (w1_l * a1_l).widen();
                    let (p11_l, p11_h) = (w1_h * a1_h).widen();
                    let int_dot =
                        ((p00_l + p00_h) + (p01_l + p01_h) + (p10_l + p10_h) + (p11_l + p11_h))
                            .reduce_sum();
                    sums[r][t] +=
                        (row_scales[r] * activation.scale) * super::integer_float(int_dot);
                }
            }
        }
    }
    for row_sums in &mut sums {
        for sum in row_sums.iter_mut().take(token_count) {
            match super::finite(*sum) {
                Ok(v) => *sum = v,
                Err(err) => return Some(Err(err)),
            }
        }
    }
    Some(Ok(sums))
}

#[inline]
#[allow(
    clippy::suboptimal_flops,
    clippy::too_many_lines,
    reason = "Preserve scalar reference multiply-then-add rounding across 8-row Q8_0 I8MM/SDOT tiles."
)]
pub(super) fn dot_q8_0_eight_rows_tile<const TOKENS: usize>(
    rows: [&[u8]; 8],
    preparations: &[super::MatvecScratch],
) -> Option<Result<[[f32; TOKENS]; 8]>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let use_i8mm = bonsai_neon_dot::has_i8mm();
    let use_dotprod = bonsai_neon_dot::has_dotprod();
    let mut sums = [[0.0_f32; TOKENS]; 8];
    let chunks = rows.map(|row| row.as_chunks::<34>().0);
    let block_count = chunks[0].len();
    let token_count = preparations.len().min(TOKENS);
    for index in 0..block_count {
        let (scales_lo, scales_hi) = f16_bits_to_f32_x8(
            neon,
            &[
                u16::from_le_bytes([chunks[0][index][0], chunks[0][index][1]]),
                u16::from_le_bytes([chunks[1][index][0], chunks[1][index][1]]),
                u16::from_le_bytes([chunks[2][index][0], chunks[2][index][1]]),
                u16::from_le_bytes([chunks[3][index][0], chunks[3][index][1]]),
                u16::from_le_bytes([chunks[4][index][0], chunks[4][index][1]]),
                u16::from_le_bytes([chunks[5][index][0], chunks[5][index][1]]),
                u16::from_le_bytes([chunks[6][index][0], chunks[6][index][1]]),
                u16::from_le_bytes([chunks[7][index][0], chunks[7][index][1]]),
            ],
        );
        if !(scales_lo.is_finite() & scales_hi.is_finite()).all_true() {
            return Some(Err(crate::invalid("nonfinite tensor or activation value")));
        }
        let scales_lo_arr = scales_lo.to_array();
        let scales_hi_arr = scales_hi.to_array();
        let row_scales = [
            scales_lo_arr[0],
            scales_lo_arr[1],
            scales_lo_arr[2],
            scales_lo_arr[3],
            scales_hi_arr[0],
            scales_hi_arr[1],
            scales_hi_arr[2],
            scales_hi_arr[3],
        ];
        let weights: [&[u8; 32]; 8] = std::array::from_fn(|r| {
            let [_, _, ref w @ ..] = chunks[r][index];
            w
        });
        if use_i8mm && token_count >= 2 {
            let mut t = 0;
            while t + 2 <= token_count {
                let a0 = &preparations[t].q8[index];
                let a1 = &preparations[t + 1].q8[index];
                for pair in [0, 2, 4, 6] {
                    let tile = bonsai_neon_dot::smmla_q8_0_2x2(
                        weights[pair],
                        weights[pair + 1],
                        &a0.values,
                        &a1.values,
                    );
                    sums[pair][t] +=
                        (row_scales[pair] * a0.scale) * super::integer_float(tile[0][0]);
                    sums[pair][t + 1] +=
                        (row_scales[pair] * a1.scale) * super::integer_float(tile[0][1]);
                    sums[pair + 1][t] +=
                        (row_scales[pair + 1] * a0.scale) * super::integer_float(tile[1][0]);
                    sums[pair + 1][t + 1] +=
                        (row_scales[pair + 1] * a1.scale) * super::integer_float(tile[1][1]);
                }
                t += 2;
            }
            if t < token_count {
                let activation = &preparations[t].q8[index];
                for r in 0..8 {
                    let int_dot = bonsai_neon_dot::dot_q8_0_block(weights[r], &activation.values);
                    sums[r][t] +=
                        (row_scales[r] * activation.scale) * super::integer_float(int_dot);
                }
            }
        } else if use_dotprod {
            for (t, scratch) in preparations.iter().take(token_count).enumerate() {
                let activation = &scratch.q8[index];
                for r in 0..8 {
                    let int_dot = bonsai_neon_dot::dot_q8_0_block(weights[r], &activation.values);
                    sums[r][t] +=
                        (row_scales[r] * activation.scale) * super::integer_float(int_dot);
                }
            }
        } else {
            let widened_w: [_; 8] = std::array::from_fn(|r| {
                let (w0_l, w0_h) = u8x16::from_slice(neon, &weights[r][..16])
                    .bitcast::<i8x16<Neon>>()
                    .widen();
                let (w1_l, w1_h) = u8x16::from_slice(neon, &weights[r][16..])
                    .bitcast::<i8x16<Neon>>()
                    .widen();
                (w0_l, w0_h, w1_l, w1_h)
            });
            for (t, scratch) in preparations.iter().take(token_count).enumerate() {
                let activation = &scratch.q8[index];
                let (a0_l, a0_h) = i8x16::from_slice(neon, &activation.values[..16]).widen();
                let (a1_l, a1_h) = i8x16::from_slice(neon, &activation.values[16..]).widen();
                for r in 0..8 {
                    let (w0_l, w0_h, w1_l, w1_h) = widened_w[r];
                    let (p00_l, p00_h) = (w0_l * a0_l).widen();
                    let (p01_l, p01_h) = (w0_h * a0_h).widen();
                    let (p10_l, p10_h) = (w1_l * a1_l).widen();
                    let (p11_l, p11_h) = (w1_h * a1_h).widen();
                    let int_dot =
                        ((p00_l + p00_h) + (p01_l + p01_h) + (p10_l + p10_h) + (p11_l + p11_h))
                            .reduce_sum();
                    sums[r][t] +=
                        (row_scales[r] * activation.scale) * super::integer_float(int_dot);
                }
            }
        }
    }
    for row_sums in &mut sums {
        for sum in row_sums.iter_mut().take(token_count) {
            match super::finite(*sum) {
                Ok(v) => *sum = v,
                Err(err) => return Some(Err(err)),
            }
        }
    }
    Some(Ok(sums))
}

#[inline]
#[allow(
    clippy::suboptimal_flops,
    reason = "Preserve scalar reference multiply-then-add rounding across 4-row Q4_0 tiles."
)]
pub(super) fn dot_q4_0_four_rows_tile<const TOKENS: usize>(
    rows: [&[u8]; 4],
    preparations: &[super::MatvecScratch],
) -> Option<Result<[[f32; TOKENS]; 4]>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let mut sums = [[0.0_f32; TOKENS]; 4];
    let chunks = rows.map(|row| row.as_chunks::<18>().0);
    let block_count = chunks[0].len();
    let token_count = preparations.len().min(TOKENS);
    for index in 0..block_count {
        let b0 = &chunks[0][index];
        let b1 = &chunks[1][index];
        let b2 = &chunks[2][index];
        let b3 = &chunks[3][index];
        let scales_vec = f16_bits_to_f32_x4(
            neon,
            [
                u16::from_le_bytes([b0[0], b0[1]]),
                u16::from_le_bytes([b1[0], b1[1]]),
                u16::from_le_bytes([b2[0], b2[1]]),
                u16::from_le_bytes([b3[0], b3[1]]),
            ],
        );
        if !scales_vec.is_finite().all_true() {
            return Some(Err(crate::invalid("nonfinite tensor or activation value")));
        }
        let row_scales = scales_vec.to_array();
        let blocks = [b0, b1, b2, b3];
        let widened_w: [_; 4] = std::array::from_fn(|r| {
            let raw = u8x16::from_slice(neon, &blocks[r][2..18]);
            let (w0_l, w0_h) = ((raw & 15).bitcast::<i8x16<Neon>>() - 8).widen();
            let (w1_l, w1_h) = ((raw >> 4).bitcast::<i8x16<Neon>>() - 8).widen();
            (w0_l, w0_h, w1_l, w1_h)
        });
        for (t, scratch) in preparations.iter().take(token_count).enumerate() {
            let activation = &scratch.q8[index];
            let (a0_l, a0_h) = i8x16::from_slice(neon, &activation.values[..16]).widen();
            let (a1_l, a1_h) = i8x16::from_slice(neon, &activation.values[16..]).widen();
            for r in 0..4 {
                let (w0_l, w0_h, w1_l, w1_h) = widened_w[r];
                let pairs = (w0_l * a0_l + w0_h * a0_h) + (w1_l * a1_l + w1_h * a1_h);
                sums[r][t] +=
                    (row_scales[r] * activation.scale) * super::integer_float(pq2_reduce(pairs));
            }
        }
    }
    for row_sums in &mut sums {
        for sum in row_sums.iter_mut().take(token_count) {
            match super::finite(*sum) {
                Ok(v) => *sum = v,
                Err(err) => return Some(Err(err)),
            }
        }
    }
    Some(Ok(sums))
}

#[inline]
#[allow(
    clippy::suboptimal_flops,
    reason = "Preserve scalar reference multiply-then-add rounding across 8-row Q4_0 tiles."
)]
pub(super) fn dot_q4_0_eight_rows_tile<const TOKENS: usize>(
    rows: [&[u8]; 8],
    preparations: &[super::MatvecScratch],
) -> Option<Result<[[f32; TOKENS]; 8]>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let mut sums = [[0.0_f32; TOKENS]; 8];
    let chunks = rows.map(|row| row.as_chunks::<18>().0);
    let block_count = chunks[0].len();
    let token_count = preparations.len().min(TOKENS);
    for index in 0..block_count {
        let (scales_lo, scales_hi) = f16_bits_to_f32_x8(
            neon,
            &[
                u16::from_le_bytes([chunks[0][index][0], chunks[0][index][1]]),
                u16::from_le_bytes([chunks[1][index][0], chunks[1][index][1]]),
                u16::from_le_bytes([chunks[2][index][0], chunks[2][index][1]]),
                u16::from_le_bytes([chunks[3][index][0], chunks[3][index][1]]),
                u16::from_le_bytes([chunks[4][index][0], chunks[4][index][1]]),
                u16::from_le_bytes([chunks[5][index][0], chunks[5][index][1]]),
                u16::from_le_bytes([chunks[6][index][0], chunks[6][index][1]]),
                u16::from_le_bytes([chunks[7][index][0], chunks[7][index][1]]),
            ],
        );
        if !(scales_lo.is_finite() & scales_hi.is_finite()).all_true() {
            return Some(Err(crate::invalid("nonfinite tensor or activation value")));
        }
        let scales_lo_arr = scales_lo.to_array();
        let scales_hi_arr = scales_hi.to_array();
        let row_scales = [
            scales_lo_arr[0],
            scales_lo_arr[1],
            scales_lo_arr[2],
            scales_lo_arr[3],
            scales_hi_arr[0],
            scales_hi_arr[1],
            scales_hi_arr[2],
            scales_hi_arr[3],
        ];
        let widened_w: [_; 8] = std::array::from_fn(|r| {
            let raw = u8x16::from_slice(neon, &chunks[r][index][2..18]);
            let (w0_l, w0_h) = ((raw & 15).bitcast::<i8x16<Neon>>() - 8).widen();
            let (w1_l, w1_h) = ((raw >> 4).bitcast::<i8x16<Neon>>() - 8).widen();
            (w0_l, w0_h, w1_l, w1_h)
        });
        for (t, scratch) in preparations.iter().take(token_count).enumerate() {
            let activation = &scratch.q8[index];
            let (a0_l, a0_h) = i8x16::from_slice(neon, &activation.values[..16]).widen();
            let (a1_l, a1_h) = i8x16::from_slice(neon, &activation.values[16..]).widen();
            for r in 0..8 {
                let (w0_l, w0_h, w1_l, w1_h) = widened_w[r];
                let pairs = (w0_l * a0_l + w0_h * a0_h) + (w1_l * a1_l + w1_h * a1_h);
                sums[r][t] +=
                    (row_scales[r] * activation.scale) * super::integer_float(pq2_reduce(pairs));
            }
        }
    }
    for row_sums in &mut sums {
        for sum in row_sums.iter_mut().take(token_count) {
            match super::finite(*sum) {
                Ok(v) => *sum = v,
                Err(err) => return Some(Err(err)),
            }
        }
    }
    Some(Ok(sums))
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Retain the scalar floating dot's rounded F32 products, binary64 sum and final F32 rounding."
)]
#[inline]
pub(super) fn float_dot(weights: &[f32], input: &[f32]) -> Option<f32> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let count = weights.len().min(input.len());
    let (w_chunks, w_rem) = weights[..count].as_chunks::<4>();
    let (in_chunks, in_rem) = input[..count].as_chunks::<4>();
    let mut sum = -0.0_f64;
    for (w_chunk, in_chunk) in w_chunks.iter().zip(in_chunks) {
        let prod =
            (f32x4::from_slice(neon, w_chunk) * f32x4::from_slice(neon, in_chunk)).to_array();
        sum += f64::from(prod[0]);
        sum += f64::from(prod[1]);
        sum += f64::from(prod[2]);
        sum += f64::from(prod[3]);
    }
    for (&w, &act) in w_rem.iter().zip(in_rem) {
        sum += f64::from(w * act);
    }
    Some(sum as f32)
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Retain the scalar floating dot's rounded F32 products, binary64 sum and final F32 rounding."
)]
#[inline]
pub(super) fn float_dot_pair(state: &[f32], first: &[f32], second: &[f32]) -> Option<(f32, f32)> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let count = state.len().min(first.len()).min(second.len());
    let (s_chunks, s_rem) = state[..count].as_chunks::<4>();
    let (f_chunks, f_rem) = first[..count].as_chunks::<4>();
    let (sec_chunks, sec_rem) = second[..count].as_chunks::<4>();
    let mut sum0 = -0.0_f64;
    let mut sum1 = -0.0_f64;
    for ((s_chunk, f_chunk), sec_chunk) in s_chunks.iter().zip(f_chunks).zip(sec_chunks) {
        let s_vec = f32x4::from_slice(neon, s_chunk);
        let prod0 = (s_vec * f32x4::from_slice(neon, f_chunk)).to_array();
        let prod1 = (s_vec * f32x4::from_slice(neon, sec_chunk)).to_array();
        sum0 += f64::from(prod0[0]);
        sum0 += f64::from(prod0[1]);
        sum0 += f64::from(prod0[2]);
        sum0 += f64::from(prod0[3]);
        sum1 += f64::from(prod1[0]);
        sum1 += f64::from(prod1[1]);
        sum1 += f64::from(prod1[2]);
        sum1 += f64::from(prod1[3]);
    }
    for ((&s, &f), &sec) in s_rem.iter().zip(f_rem).zip(sec_rem) {
        sum0 += f64::from(s * f);
        sum1 += f64::from(s * sec);
    }
    Some((sum0 as f32, sum1 as f32))
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Retain the scalar floating dot's rounded F32 products, binary64 sum and final F32 rounding."
)]
#[inline]
pub(super) fn float_dot_pair_four_rows(
    states: [&[f32]; 4],
    first: &[f32],
    second: &[f32],
) -> Option<[(f32, f32); 4]> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let count = first
        .len()
        .min(second.len())
        .min(states[0].len())
        .min(states[1].len())
        .min(states[2].len())
        .min(states[3].len());
    let (f_chunks, f_rem) = first[..count].as_chunks::<4>();
    let (sec_chunks, sec_rem) = second[..count].as_chunks::<4>();
    let mut sum0 = [-0.0_f64; 4];
    let mut sum1 = [-0.0_f64; 4];
    for (idx, (f_chunk, sec_chunk)) in f_chunks.iter().zip(sec_chunks).enumerate() {
        let f_vec = f32x4::from_slice(neon, f_chunk);
        let sec_vec = f32x4::from_slice(neon, sec_chunk);
        let offset = idx * 4;
        for r in 0..4 {
            let s_vec = f32x4::from_slice(neon, &states[r][offset..offset + 4]);
            let prod0 = (s_vec * f_vec).to_array();
            let prod1 = (s_vec * sec_vec).to_array();
            sum0[r] += f64::from(prod0[0]);
            sum0[r] += f64::from(prod0[1]);
            sum0[r] += f64::from(prod0[2]);
            sum0[r] += f64::from(prod0[3]);
            sum1[r] += f64::from(prod1[0]);
            sum1[r] += f64::from(prod1[1]);
            sum1[r] += f64::from(prod1[2]);
            sum1[r] += f64::from(prod1[3]);
        }
    }
    let rem_base = f_chunks.len() * 4;
    for (i, (&f, &sec)) in f_rem.iter().zip(sec_rem).enumerate() {
        let idx = rem_base + i;
        for r in 0..4 {
            let s = states[r][idx];
            sum0[r] += f64::from(s * f);
            sum1[r] += f64::from(s * sec);
        }
    }
    Some([
        (sum0[0] as f32, sum1[0] as f32),
        (sum0[1] as f32, sum1[1] as f32),
        (sum0[2] as f32, sum1[2] as f32),
        (sum0[3] as f32, sum1[3] as f32),
    ])
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Retain the scalar floating dot's rounded F32 products, binary64 sum and final F32 rounding."
)]
#[inline]
pub(super) fn float_dot_pair_eight_rows(
    states: [&[f32]; 8],
    first: &[f32],
    second: &[f32],
) -> Option<[(f32, f32); 8]> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let count = states
        .iter()
        .fold(first.len().min(second.len()), |acc, s| acc.min(s.len()));
    let (f_chunks, f_rem) = first[..count].as_chunks::<4>();
    let (sec_chunks, sec_rem) = second[..count].as_chunks::<4>();
    let mut sum0 = [-0.0_f64; 8];
    let mut sum1 = [-0.0_f64; 8];
    for (idx, (f_chunk, sec_chunk)) in f_chunks.iter().zip(sec_chunks).enumerate() {
        let f_vec = f32x4::from_slice(neon, f_chunk);
        let sec_vec = f32x4::from_slice(neon, sec_chunk);
        let offset = idx * 4;
        for r in 0..8 {
            let s_vec = f32x4::from_slice(neon, &states[r][offset..offset + 4]);
            let prod0 = (s_vec * f_vec).to_array();
            let prod1 = (s_vec * sec_vec).to_array();
            sum0[r] += f64::from(prod0[0]);
            sum0[r] += f64::from(prod0[1]);
            sum0[r] += f64::from(prod0[2]);
            sum0[r] += f64::from(prod0[3]);
            sum1[r] += f64::from(prod1[0]);
            sum1[r] += f64::from(prod1[1]);
            sum1[r] += f64::from(prod1[2]);
            sum1[r] += f64::from(prod1[3]);
        }
    }
    let rem_base = f_chunks.len() * 4;
    for (i, (&f, &sec)) in f_rem.iter().zip(sec_rem).enumerate() {
        let idx = rem_base + i;
        for r in 0..8 {
            let s = states[r][idx];
            sum0[r] += f64::from(s * f);
            sum1[r] += f64::from(s * sec);
        }
    }
    Some(std::array::from_fn(|r| (sum0[r] as f32, sum1[r] as f32)))
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Retain the scalar floating dot's rounded F32 products, binary64 sum and final F32 rounding."
)]
#[inline]
pub(super) fn f16_dot(first: &[f32], second: &[u16]) -> Option<f32> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let count = first.len().min(second.len());
    let (f_chunks, f_rem) = first[..count].as_chunks::<8>();
    let (s_chunks, s_rem) = second[..count].as_chunks::<8>();
    let mut sum = -0.0_f64;
    for (f_chunk, s_chunk) in f_chunks.iter().zip(s_chunks) {
        let (s0, s1) = f16_bits_to_f32_x8(neon, s_chunk);
        let prod0 = (f32x4::from_slice(neon, &f_chunk[..4]) * s0).to_array();
        let prod1 = (f32x4::from_slice(neon, &f_chunk[4..]) * s1).to_array();
        sum += f64::from(prod0[0]);
        sum += f64::from(prod0[1]);
        sum += f64::from(prod0[2]);
        sum += f64::from(prod0[3]);
        sum += f64::from(prod1[0]);
        sum += f64::from(prod1[1]);
        sum += f64::from(prod1[2]);
        sum += f64::from(prod1[3]);
    }
    for (&f, &s) in f_rem.iter().zip(s_rem) {
        sum += f64::from(f * super::half_to_float(s));
    }
    Some(sum as f32)
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Retain the scalar floating dot's rounded F32 products, binary64 sum and final F32 rounding."
)]
#[inline]
pub(super) fn float_dot_four(queries: [&[f32]; 4], key: &[f32]) -> Option<[f32; 4]> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let count = key
        .len()
        .min(queries[0].len())
        .min(queries[1].len())
        .min(queries[2].len())
        .min(queries[3].len());
    let (k_chunks, k_rem) = key[..count].as_chunks::<4>();
    let mut sums = [-0.0_f64; 4];
    for (idx, k_chunk) in k_chunks.iter().enumerate() {
        let k_vec = f32x4::from_slice(neon, k_chunk);
        let offset = idx * 4;
        for h in 0..4 {
            let prod =
                (f32x4::from_slice(neon, &queries[h][offset..offset + 4]) * k_vec).to_array();
            sums[h] += f64::from(prod[0]);
            sums[h] += f64::from(prod[1]);
            sums[h] += f64::from(prod[2]);
            sums[h] += f64::from(prod[3]);
        }
    }
    let rem_base = k_chunks.len() * 4;
    for (i, &k) in k_rem.iter().enumerate() {
        let idx = rem_base + i;
        for h in 0..4 {
            sums[h] += f64::from(queries[h][idx] * k);
        }
    }
    Some([
        sums[0] as f32,
        sums[1] as f32,
        sums[2] as f32,
        sums[3] as f32,
    ])
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Retain the scalar floating dot's rounded F32 products, binary64 sum and final F32 rounding."
)]
#[inline]
pub(super) fn f16_dot_four(queries: [&[f32]; 4], key: &[u16]) -> Option<[f32; 4]> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let count = key
        .len()
        .min(queries[0].len())
        .min(queries[1].len())
        .min(queries[2].len())
        .min(queries[3].len());
    let (s_chunks, s_rem) = key[..count].as_chunks::<8>();
    let mut sums = [-0.0_f64; 4];
    for (idx, s_chunk) in s_chunks.iter().enumerate() {
        let (s0, s1) = f16_bits_to_f32_x8(neon, s_chunk);
        let offset = idx * 8;
        for h in 0..4 {
            let q = &queries[h][offset..offset + 8];
            let prod0 = (f32x4::from_slice(neon, &q[..4]) * s0).to_array();
            let prod1 = (f32x4::from_slice(neon, &q[4..]) * s1).to_array();
            sums[h] += f64::from(prod0[0]);
            sums[h] += f64::from(prod0[1]);
            sums[h] += f64::from(prod0[2]);
            sums[h] += f64::from(prod0[3]);
            sums[h] += f64::from(prod1[0]);
            sums[h] += f64::from(prod1[1]);
            sums[h] += f64::from(prod1[2]);
            sums[h] += f64::from(prod1[3]);
        }
    }
    let rem_base = s_chunks.len() * 8;
    for (i, &s) in s_rem.iter().enumerate() {
        let decoded = super::half_to_float(s);
        let idx = rem_base + i;
        for h in 0..4 {
            sums[h] += f64::from(queries[h][idx] * decoded);
        }
    }
    Some([
        sums[0] as f32,
        sums[1] as f32,
        sums[2] as f32,
        sums[3] as f32,
    ])
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Retain the scalar floating dot's rounded F32 products, binary64 sum and final F32 rounding."
)]
#[inline]
pub(super) fn f16_dot_two_token_four_queries(
    queries0: [&[f32]; 4],
    queries1: [&[f32]; 4],
    key: &[u16],
) -> Option<([f32; 4], [f32; 4])> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let count = key
        .len()
        .min(queries0[0].len())
        .min(queries0[1].len())
        .min(queries0[2].len())
        .min(queries0[3].len())
        .min(queries1[0].len())
        .min(queries1[1].len())
        .min(queries1[2].len())
        .min(queries1[3].len());
    let (s_chunks, s_rem) = key[..count].as_chunks::<8>();
    let mut sums0 = [-0.0_f64; 4];
    let mut sums1 = [-0.0_f64; 4];
    for (idx, s_chunk) in s_chunks.iter().enumerate() {
        let (s0, s1) = f16_bits_to_f32_x8(neon, s_chunk);
        let offset = idx * 8;
        for h in 0..4 {
            let q0 = &queries0[h][offset..offset + 8];
            let p0_0 = (f32x4::from_slice(neon, &q0[..4]) * s0).to_array();
            let p0_1 = (f32x4::from_slice(neon, &q0[4..]) * s1).to_array();
            sums0[h] += f64::from(p0_0[0]);
            sums0[h] += f64::from(p0_0[1]);
            sums0[h] += f64::from(p0_0[2]);
            sums0[h] += f64::from(p0_0[3]);
            sums0[h] += f64::from(p0_1[0]);
            sums0[h] += f64::from(p0_1[1]);
            sums0[h] += f64::from(p0_1[2]);
            sums0[h] += f64::from(p0_1[3]);

            let q1 = &queries1[h][offset..offset + 8];
            let p1_0 = (f32x4::from_slice(neon, &q1[..4]) * s0).to_array();
            let p1_1 = (f32x4::from_slice(neon, &q1[4..]) * s1).to_array();
            sums1[h] += f64::from(p1_0[0]);
            sums1[h] += f64::from(p1_0[1]);
            sums1[h] += f64::from(p1_0[2]);
            sums1[h] += f64::from(p1_0[3]);
            sums1[h] += f64::from(p1_1[0]);
            sums1[h] += f64::from(p1_1[1]);
            sums1[h] += f64::from(p1_1[2]);
            sums1[h] += f64::from(p1_1[3]);
        }
    }
    let rem_base = s_chunks.len() * 8;
    for (i, &s) in s_rem.iter().enumerate() {
        let decoded = super::half_to_float(s);
        let idx = rem_base + i;
        for h in 0..4 {
            sums0[h] += f64::from(queries0[h][idx] * decoded);
            sums1[h] += f64::from(queries1[h][idx] * decoded);
        }
    }
    Some((
        [
            sums0[0] as f32,
            sums0[1] as f32,
            sums0[2] as f32,
            sums0[3] as f32,
        ],
        [
            sums1[0] as f32,
            sums1[1] as f32,
            sums1[2] as f32,
            sums1[3] as f32,
        ],
    ))
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Retain the scalar floating dot's rounded F32 products, binary64 sum and final F32 rounding."
)]
#[inline]
pub(super) fn f16_dot_two_keys_four_queries(
    queries: [&[f32]; 4],
    keys: [&[u16]; 2],
) -> Option<[[f32; 4]; 2]> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let count = keys[0].len();
    if keys[1].len() != count || queries.iter().any(|q| q.len() != count) {
        return None;
    }
    let k_chunks = keys.map(|k| k.as_chunks::<8>().0);
    let k_rems = keys.map(|k| k.as_chunks::<8>().1);
    let q_chunks = queries.map(|q| q.as_chunks::<8>().0);
    let mut sums = [[-0.0_f64; 4]; 2];
    for idx in 0..k_chunks[0].len() {
        let (k0_0, k0_1) = f16_bits_to_f32_x8(neon, &k_chunks[0][idx]);
        let (k1_0, k1_1) = f16_bits_to_f32_x8(neon, &k_chunks[1][idx]);
        let k_low = [k0_0, k1_0];
        let k_high = [k0_1, k1_1];
        for h in 0..4 {
            let q8 = &q_chunks[h][idx];
            let q0 = f32x4::from_slice(neon, &q8[..4]);
            let q1 = f32x4::from_slice(neon, &q8[4..]);
            for k in 0..2 {
                let prod0 = (q0 * k_low[k]).to_array();
                let prod1 = (q1 * k_high[k]).to_array();
                sums[k][h] += f64::from(prod0[0]);
                sums[k][h] += f64::from(prod0[1]);
                sums[k][h] += f64::from(prod0[2]);
                sums[k][h] += f64::from(prod0[3]);
                sums[k][h] += f64::from(prod1[0]);
                sums[k][h] += f64::from(prod1[1]);
                sums[k][h] += f64::from(prod1[2]);
                sums[k][h] += f64::from(prod1[3]);
            }
        }
    }
    let rem_base = k_chunks[0].len() * 8;
    for i in 0..k_rems[0].len() {
        let idx = rem_base + i;
        let decoded = [
            super::half_to_float(k_rems[0][i]),
            super::half_to_float(k_rems[1][i]),
        ];
        for h in 0..4 {
            let q = queries[h][idx];
            for k in 0..2 {
                sums[k][h] += f64::from(q * decoded[k]);
            }
        }
    }
    Some(sums.map(|row| row.map(|v| v as f32)))
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Retain the scalar floating dot's rounded F32 products, binary64 sum and final F32 rounding."
)]
#[inline]
pub(super) fn f16_dot_two_keys_two_token_four_queries(
    queries0: [&[f32]; 4],
    queries1: [&[f32]; 4],
    keys: [&[u16]; 2],
) -> Option<(
    super::pinned_attention::TwoKeyFourQueryScores,
    super::pinned_attention::TwoKeyFourQueryScores,
)> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let count = keys[0].len();
    if keys[1].len() != count
        || queries0.iter().any(|q| q.len() != count)
        || queries1.iter().any(|q| q.len() != count)
    {
        return None;
    }
    let k_chunks = keys.map(|k| k.as_chunks::<8>().0);
    let k_rems = keys.map(|k| k.as_chunks::<8>().1);
    let q0_chunks = queries0.map(|q| q.as_chunks::<8>().0);
    let q1_chunks = queries1.map(|q| q.as_chunks::<8>().0);
    let mut sums0 = [[-0.0_f64; 4]; 2];
    let mut sums1 = [[-0.0_f64; 4]; 2];
    for idx in 0..k_chunks[0].len() {
        let (k0_0, k0_1) = f16_bits_to_f32_x8(neon, &k_chunks[0][idx]);
        let (k1_0, k1_1) = f16_bits_to_f32_x8(neon, &k_chunks[1][idx]);
        let k_low = [k0_0, k1_0];
        let k_high = [k0_1, k1_1];
        for h in 0..4 {
            let q8_0 = &q0_chunks[h][idx];
            let q0_low = f32x4::from_slice(neon, &q8_0[..4]);
            let q0_high = f32x4::from_slice(neon, &q8_0[4..]);
            let q8_1 = &q1_chunks[h][idx];
            let q1_low = f32x4::from_slice(neon, &q8_1[..4]);
            let q1_high = f32x4::from_slice(neon, &q8_1[4..]);
            for k in 0..2 {
                let p0_0 = (q0_low * k_low[k]).to_array();
                let p0_1 = (q0_high * k_high[k]).to_array();
                sums0[k][h] += f64::from(p0_0[0]);
                sums0[k][h] += f64::from(p0_0[1]);
                sums0[k][h] += f64::from(p0_0[2]);
                sums0[k][h] += f64::from(p0_0[3]);
                sums0[k][h] += f64::from(p0_1[0]);
                sums0[k][h] += f64::from(p0_1[1]);
                sums0[k][h] += f64::from(p0_1[2]);
                sums0[k][h] += f64::from(p0_1[3]);

                let p1_0 = (q1_low * k_low[k]).to_array();
                let p1_1 = (q1_high * k_high[k]).to_array();
                sums1[k][h] += f64::from(p1_0[0]);
                sums1[k][h] += f64::from(p1_0[1]);
                sums1[k][h] += f64::from(p1_0[2]);
                sums1[k][h] += f64::from(p1_0[3]);
                sums1[k][h] += f64::from(p1_1[0]);
                sums1[k][h] += f64::from(p1_1[1]);
                sums1[k][h] += f64::from(p1_1[2]);
                sums1[k][h] += f64::from(p1_1[3]);
            }
        }
    }
    let rem_base = k_chunks[0].len() * 8;
    for i in 0..k_rems[0].len() {
        let idx = rem_base + i;
        let decoded = [
            super::half_to_float(k_rems[0][i]),
            super::half_to_float(k_rems[1][i]),
        ];
        for h in 0..4 {
            let q0 = queries0[h][idx];
            let q1 = queries1[h][idx];
            for k in 0..2 {
                sums0[k][h] += f64::from(q0 * decoded[k]);
                sums1[k][h] += f64::from(q1 * decoded[k]);
            }
        }
    }
    Some((
        sums0.map(|row| row.map(|v| v as f32)),
        sums1.map(|row| row.map(|v| v as f32)),
    ))
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Retain the scalar floating dot's rounded F32 products, binary64 sum and final F32 rounding."
)]
#[inline]
pub(super) fn f16_dot_four_keys_four_queries(
    queries: [&[f32]; 4],
    keys: [&[u16]; 4],
) -> Option<[[f32; 4]; 4]> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let count = keys[0].len();
    if keys.iter().any(|k| k.len() != count) || queries.iter().any(|q| q.len() != count) {
        return None;
    }
    let k_chunks = keys.map(|k| k.as_chunks::<8>().0);
    let k_rems = keys.map(|k| k.as_chunks::<8>().1);
    let q_chunks = queries.map(|q| q.as_chunks::<8>().0);
    let mut sums = [[-0.0_f64; 4]; 4];
    for idx in 0..k_chunks[0].len() {
        let (k0_0, k0_1) = f16_bits_to_f32_x8(neon, &k_chunks[0][idx]);
        let (k1_0, k1_1) = f16_bits_to_f32_x8(neon, &k_chunks[1][idx]);
        let (k2_0, k2_1) = f16_bits_to_f32_x8(neon, &k_chunks[2][idx]);
        let (k3_0, k3_1) = f16_bits_to_f32_x8(neon, &k_chunks[3][idx]);
        let k_low = [k0_0, k1_0, k2_0, k3_0];
        let k_high = [k0_1, k1_1, k2_1, k3_1];
        for h in 0..4 {
            let q8 = &q_chunks[h][idx];
            let q0 = f32x4::from_slice(neon, &q8[..4]);
            let q1 = f32x4::from_slice(neon, &q8[4..]);
            for k in 0..4 {
                let prod0 = (q0 * k_low[k]).to_array();
                let prod1 = (q1 * k_high[k]).to_array();
                sums[k][h] += f64::from(prod0[0]);
                sums[k][h] += f64::from(prod0[1]);
                sums[k][h] += f64::from(prod0[2]);
                sums[k][h] += f64::from(prod0[3]);
                sums[k][h] += f64::from(prod1[0]);
                sums[k][h] += f64::from(prod1[1]);
                sums[k][h] += f64::from(prod1[2]);
                sums[k][h] += f64::from(prod1[3]);
            }
        }
    }
    let rem_base = k_chunks[0].len() * 8;
    for i in 0..k_rems[0].len() {
        let idx = rem_base + i;
        let decoded = [
            super::half_to_float(k_rems[0][i]),
            super::half_to_float(k_rems[1][i]),
            super::half_to_float(k_rems[2][i]),
            super::half_to_float(k_rems[3][i]),
        ];
        for h in 0..4 {
            let q = queries[h][idx];
            for k in 0..4 {
                sums[k][h] += f64::from(q * decoded[k]);
            }
        }
    }
    Some(sums.map(|row| row.map(|v| v as f32)))
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Retain the scalar floating dot's rounded F32 products, binary64 sum and final F32 rounding."
)]
#[inline]
pub(super) fn f16_dot_four_keys_two_token_four_queries(
    queries0: [&[f32]; 4],
    queries1: [&[f32]; 4],
    keys: [&[u16]; 4],
) -> Option<(
    super::pinned_attention::FourKeyFourQueryScores,
    super::pinned_attention::FourKeyFourQueryScores,
)> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let count = keys[0].len();
    if keys.iter().any(|k| k.len() != count)
        || queries0.iter().any(|q| q.len() != count)
        || queries1.iter().any(|q| q.len() != count)
    {
        return None;
    }
    let k_chunks = keys.map(|k| k.as_chunks::<8>().0);
    let k_rems = keys.map(|k| k.as_chunks::<8>().1);
    let q0_chunks = queries0.map(|q| q.as_chunks::<8>().0);
    let q1_chunks = queries1.map(|q| q.as_chunks::<8>().0);
    let mut sums0 = [[-0.0_f64; 4]; 4];
    let mut sums1 = [[-0.0_f64; 4]; 4];
    for idx in 0..k_chunks[0].len() {
        let (k0_0, k0_1) = f16_bits_to_f32_x8(neon, &k_chunks[0][idx]);
        let (k1_0, k1_1) = f16_bits_to_f32_x8(neon, &k_chunks[1][idx]);
        let (k2_0, k2_1) = f16_bits_to_f32_x8(neon, &k_chunks[2][idx]);
        let (k3_0, k3_1) = f16_bits_to_f32_x8(neon, &k_chunks[3][idx]);
        let k_low = [k0_0, k1_0, k2_0, k3_0];
        let k_high = [k0_1, k1_1, k2_1, k3_1];
        for h in 0..4 {
            let q8_0 = &q0_chunks[h][idx];
            let q0_low = f32x4::from_slice(neon, &q8_0[..4]);
            let q0_high = f32x4::from_slice(neon, &q8_0[4..]);
            let q8_1 = &q1_chunks[h][idx];
            let q1_low = f32x4::from_slice(neon, &q8_1[..4]);
            let q1_high = f32x4::from_slice(neon, &q8_1[4..]);
            for k in 0..4 {
                let p0_0 = (q0_low * k_low[k]).to_array();
                let p0_1 = (q0_high * k_high[k]).to_array();
                sums0[k][h] += f64::from(p0_0[0]);
                sums0[k][h] += f64::from(p0_0[1]);
                sums0[k][h] += f64::from(p0_0[2]);
                sums0[k][h] += f64::from(p0_0[3]);
                sums0[k][h] += f64::from(p0_1[0]);
                sums0[k][h] += f64::from(p0_1[1]);
                sums0[k][h] += f64::from(p0_1[2]);
                sums0[k][h] += f64::from(p0_1[3]);

                let p1_0 = (q1_low * k_low[k]).to_array();
                let p1_1 = (q1_high * k_high[k]).to_array();
                sums1[k][h] += f64::from(p1_0[0]);
                sums1[k][h] += f64::from(p1_0[1]);
                sums1[k][h] += f64::from(p1_0[2]);
                sums1[k][h] += f64::from(p1_0[3]);
                sums1[k][h] += f64::from(p1_1[0]);
                sums1[k][h] += f64::from(p1_1[1]);
                sums1[k][h] += f64::from(p1_1[2]);
                sums1[k][h] += f64::from(p1_1[3]);
            }
        }
    }
    let rem_base = k_chunks[0].len() * 8;
    for i in 0..k_rems[0].len() {
        let idx = rem_base + i;
        let decoded = [
            super::half_to_float(k_rems[0][i]),
            super::half_to_float(k_rems[1][i]),
            super::half_to_float(k_rems[2][i]),
            super::half_to_float(k_rems[3][i]),
        ];
        for h in 0..4 {
            let q0 = queries0[h][idx];
            let q1 = queries1[h][idx];
            for k in 0..4 {
                sums0[k][h] += f64::from(q0 * decoded[k]);
                sums1[k][h] += f64::from(q1 * decoded[k]);
            }
        }
    }
    Some((
        sums0.map(|row| row.map(|v| v as f32)),
        sums1.map(|row| row.map(|v| v as f32)),
    ))
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Retain the scalar floating dot's rounded F32 products, binary64 sum and final F32 rounding."
)]
#[inline]
pub(super) fn bf16_dot(bytes: &[u8], input: &[f32]) -> Option<Result<f32>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let zero = u16x8::splat(neon, 0);
    let mut finite_mask = f32x4::splat(neon, 0.0).is_finite();
    let (b_chunks, b_rem) = bytes.as_chunks::<16>();
    let (in_chunks, in_rem) = input.as_chunks::<8>();
    let mut sum = -0.0_f64;
    for (b_chunk, in_chunk) in b_chunks.iter().zip(in_chunks) {
        let raw = u8x16::from_slice(neon, b_chunk).bitcast::<u16x8<Neon>>();
        let w0 = zero.zip_low(raw).bitcast::<f32x4<Neon>>();
        let w1 = zero.zip_high(raw).bitcast::<f32x4<Neon>>();
        finite_mask = finite_mask & w0.is_finite() & w1.is_finite();
        let prod0 = (w0 * f32x4::from_slice(neon, &in_chunk[..4])).to_array();
        let prod1 = (w1 * f32x4::from_slice(neon, &in_chunk[4..])).to_array();
        sum += f64::from(prod0[0]);
        sum += f64::from(prod0[1]);
        sum += f64::from(prod0[2]);
        sum += f64::from(prod0[3]);
        sum += f64::from(prod1[0]);
        sum += f64::from(prod1[1]);
        sum += f64::from(prod1[2]);
        sum += f64::from(prod1[3]);
    }
    if !finite_mask.all_true() {
        return Some(Err(crate::invalid("nonfinite tensor or activation value")));
    }
    for (b, &act) in b_rem.as_chunks::<2>().0.iter().zip(in_rem) {
        match super::finite(f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16)) {
            Ok(w) => sum += f64::from(w * act),
            Err(err) => return Some(Err(err)),
        }
    }
    Some(super::finite(sum as f32))
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Retain the scalar floating dot's rounded F32 products, binary64 sum and final F32 rounding."
)]
#[inline]
fn bf16_dot_rows<const ROWS: usize>(
    rows: [&[u8]; ROWS],
    input: &[f32],
) -> Option<Result<[f32; ROWS]>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let zero = u16x8::splat(neon, 0);
    let mut finite_mask = f32x4::splat(neon, 0.0).is_finite();
    let b_chunks = rows.map(|row| row.as_chunks::<16>().0);
    let b_rems = rows.map(|row| row.as_chunks::<16>().1);
    let (in_chunks, in_rem) = input.as_chunks::<8>();
    let mut sums = [-0.0_f64; ROWS];
    for (idx, in_chunk) in in_chunks.iter().enumerate() {
        let a0 = f32x4::from_slice(neon, &in_chunk[..4]);
        let a1 = f32x4::from_slice(neon, &in_chunk[4..]);
        for r in 0..ROWS {
            let raw = u8x16::from_slice(neon, &b_chunks[r][idx]).bitcast::<u16x8<Neon>>();
            let w0 = zero.zip_low(raw).bitcast::<f32x4<Neon>>();
            let w1 = zero.zip_high(raw).bitcast::<f32x4<Neon>>();
            finite_mask = finite_mask & w0.is_finite() & w1.is_finite();
            let prod0 = (w0 * a0).to_array();
            let prod1 = (w1 * a1).to_array();
            sums[r] += f64::from(prod0[0]);
            sums[r] += f64::from(prod0[1]);
            sums[r] += f64::from(prod0[2]);
            sums[r] += f64::from(prod0[3]);
            sums[r] += f64::from(prod1[0]);
            sums[r] += f64::from(prod1[1]);
            sums[r] += f64::from(prod1[2]);
            sums[r] += f64::from(prod1[3]);
        }
    }
    if !finite_mask.all_true() {
        return Some(Err(crate::invalid("nonfinite tensor or activation value")));
    }
    for r in 0..ROWS {
        for (b, &act) in b_rems[r].as_chunks::<2>().0.iter().zip(in_rem) {
            match super::finite(f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16)) {
                Ok(w) => sums[r] += f64::from(w * act),
                Err(err) => return Some(Err(err)),
            }
        }
    }
    let mut out = [0.0_f32; ROWS];
    for r in 0..ROWS {
        match super::finite(sums[r] as f32) {
            Ok(val) => out[r] = val,
            Err(err) => return Some(Err(err)),
        }
    }
    Some(Ok(out))
}

#[inline]
pub(super) fn bf16_dot_four_rows(rows: [&[u8]; 4], input: &[f32]) -> Option<Result<[f32; 4]>> {
    bf16_dot_rows::<4>(rows, input)
}

#[inline]
pub(super) fn bf16_dot_eight_rows(rows: [&[u8]; 8], input: &[f32]) -> Option<Result<[f32; 8]>> {
    bf16_dot_rows::<8>(rows, input)
}

#[inline]
fn transpose_4x4(
    r0: f32x4<Neon>,
    r1: f32x4<Neon>,
    r2: f32x4<Neon>,
    r3: f32x4<Neon>,
) -> [f32x4<Neon>; 4] {
    let t0 = r0.zip_low(r2);
    let t1 = r0.zip_high(r2);
    let t2 = r1.zip_low(r3);
    let t3 = r1.zip_high(r3);
    [
        t0.zip_low(t2),
        t0.zip_high(t2),
        t1.zip_low(t3),
        t1.zip_high(t3),
    ]
}

#[inline]
fn hadamard_16_registers(
    neon: Neon,
    block: &[f32],
    signs: Option<&[f32]>,
) -> ([f32x4<Neon>; 4], fearless_simd::mask32x4<Neon>) {
    let (r0, r1, r2, r3) = signs.map_or_else(
        || {
            (
                f32x4::from_slice(neon, &block[0..4]),
                f32x4::from_slice(neon, &block[4..8]),
                f32x4::from_slice(neon, &block[8..12]),
                f32x4::from_slice(neon, &block[12..16]),
            )
        },
        |s| {
            (
                f32x4::from_slice(neon, &block[0..4]) * f32x4::from_slice(neon, &s[0..4]),
                f32x4::from_slice(neon, &block[4..8]) * f32x4::from_slice(neon, &s[4..8]),
                f32x4::from_slice(neon, &block[8..12]) * f32x4::from_slice(neon, &s[8..12]),
                f32x4::from_slice(neon, &block[12..16]) * f32x4::from_slice(neon, &s[12..16]),
            )
        },
    );
    let finite = r0.is_finite() & r1.is_finite() & r2.is_finite() & r3.is_finite();
    let [a0, a1, a2, a3] = transpose_4x4(r0, r1, r2, r3);
    let (b0, b1, b2, b3) = (a0 + a1, a0 - a1, a2 + a3, a2 - a3);
    let [v0, v1, v2, v3] = transpose_4x4(b0 + b2, b1 + b3, b0 - b2, b1 - b3);
    let (c0, c1, c2, c3) = (v0 + v1, v0 - v1, v2 + v3, v2 - v3);
    ([c0 + c2, c1 + c3, c0 - c2, c1 - c3], finite)
}

#[inline]
fn hadamard_radix4_stage(neon: Neon, values: &mut [f32], half: usize) {
    for block in values.chunks_exact_mut(half * 4) {
        let (first_half, second_half) = block.split_at_mut(half * 2);
        let (q0, q1) = first_half.split_at_mut(half);
        let (q2, q3) = second_half.split_at_mut(half);
        for (((c0_slice, c1_slice), c2_slice), c3_slice) in q0
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(q1.as_chunks_mut::<4>().0)
            .zip(q2.as_chunks_mut::<4>().0)
            .zip(q3.as_chunks_mut::<4>().0)
        {
            let v0 = f32x4::from_slice(neon, c0_slice);
            let v1 = f32x4::from_slice(neon, c1_slice);
            let v2 = f32x4::from_slice(neon, c2_slice);
            let v3 = f32x4::from_slice(neon, c3_slice);
            let b0 = v0 + v1;
            let b1 = v0 - v1;
            let b2 = v2 + v3;
            let b3 = v2 - v3;
            (b0 + b2).store_slice(c0_slice);
            (b1 + b3).store_slice(c1_slice);
            (b0 - b2).store_slice(c2_slice);
            (b1 - b3).store_slice(c3_slice);
        }
    }
}

#[inline]
fn hadamard_finish(
    neon: Neon,
    values: &mut [f32],
    mut half: usize,
    norm: f32x4<Neon>,
    in_finite: fearless_simd::mask32x4<Neon>,
    mut finite_mask: fearless_simd::mask32x4<Neon>,
) -> Result<()> {
    if !in_finite.all_true() {
        return Err(crate::invalid(
            "Hadamard requires a finite power-of-two block",
        ));
    }
    if half == values.len() && values.len() >= 16 {
        if !finite_mask.all_true() {
            return Err(crate::invalid("nonfinite kernel output"));
        }
        return Ok(());
    }
    while half * 4 < values.len() {
        hadamard_radix4_stage(neon, values, half);
        half *= 4;
    }
    if half * 4 == values.len() {
        let (first_half, second_half) = values.split_at_mut(half * 2);
        let (q0, q1) = first_half.split_at_mut(half);
        let (q2, q3) = second_half.split_at_mut(half);
        for (((c0_slice, c1_slice), c2_slice), c3_slice) in q0
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(q1.as_chunks_mut::<4>().0)
            .zip(q2.as_chunks_mut::<4>().0)
            .zip(q3.as_chunks_mut::<4>().0)
        {
            let v0 = f32x4::from_slice(neon, c0_slice);
            let v1 = f32x4::from_slice(neon, c1_slice);
            let v2 = f32x4::from_slice(neon, c2_slice);
            let v3 = f32x4::from_slice(neon, c3_slice);
            let b0 = v0 + v1;
            let b1 = v0 - v1;
            let b2 = v2 + v3;
            let b3 = v2 - v3;
            let out0 = (b0 + b2) * norm;
            let out1 = (b1 + b3) * norm;
            let out2 = (b0 - b2) * norm;
            let out3 = (b1 - b3) * norm;
            finite_mask &=
                out0.is_finite() & out1.is_finite() & out2.is_finite() & out3.is_finite();
            out0.store_slice(c0_slice);
            out1.store_slice(c1_slice);
            out2.store_slice(c2_slice);
            out3.store_slice(c3_slice);
        }
    } else if half < values.len() {
        let (left_half, right_half) = values.split_at_mut(half);
        for (left, right) in left_half
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(right_half.as_chunks_mut::<4>().0)
        {
            let l = f32x4::from_slice(neon, left);
            let r = f32x4::from_slice(neon, right);
            let out_l = (l + r) * norm;
            let out_r = (l - r) * norm;
            finite_mask = finite_mask & out_l.is_finite() & out_r.is_finite();
            out_l.store_slice(left);
            out_r.store_slice(right);
        }
    } else {
        for chunk in values.as_chunks_mut::<4>().0 {
            let v = f32x4::from_slice(neon, chunk) * norm;
            finite_mask &= v.is_finite();
            v.store_slice(chunk);
        }
    }
    if !finite_mask.all_true() {
        return Err(crate::invalid("nonfinite kernel output"));
    }
    Ok(())
}

/// Evaluates the normalized Sylvester Hadamard butterfly stages in NEON `f32x4`.
///
/// Requires a power-of-two slice of at least four elements. Stages `half = 1..=32`
/// are fused in registers per 64-element block when `values.len() >= 64` (or
/// `half = 1..=8` per 16-element block when `values.len() >= 16`), subsequent
/// stages run in radix-4 pairs (`half` and `half * 2`), and the final radix-4 or
/// radix-2 butterfly stage is fused with normalization while preserving exact
/// IEEE 754 addition and subtraction order.
#[inline]
pub(super) fn hadamard(values: &mut [f32], normalization: f32) -> Option<Result<()>> {
    if values.len() < 4 {
        return None;
    }
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let norm = f32x4::splat(neon, normalization);
    let mut in_finite = f32x4::splat(neon, 0.0).is_finite();
    let mut finite_mask = in_finite;
    let half = if values.len() >= 64 {
        let fuse_norm = values.len() == 64;
        for block in values.as_chunks_mut::<64>().0 {
            let (q0, f0) = hadamard_16_registers(neon, &block[0..16], None);
            let (q1, f1) = hadamard_16_registers(neon, &block[16..32], None);
            let (q2, f2) = hadamard_16_registers(neon, &block[32..48], None);
            let (q3, f3) = hadamard_16_registers(neon, &block[48..64], None);
            in_finite &= f0 & f1 & f2 & f3;
            for i in 0..4 {
                let b0 = q0[i] + q1[i];
                let b1 = q0[i] - q1[i];
                let b2 = q2[i] + q3[i];
                let b3 = q2[i] - q3[i];
                let (mut out0, mut out1, mut out2, mut out3) = (b0 + b2, b1 + b3, b0 - b2, b1 - b3);
                if fuse_norm {
                    out0 *= norm;
                    out1 *= norm;
                    out2 *= norm;
                    out3 *= norm;
                    finite_mask &=
                        out0.is_finite() & out1.is_finite() & out2.is_finite() & out3.is_finite();
                }
                let base = i * 4;
                out0.store_slice(&mut block[base..base + 4]);
                out1.store_slice(&mut block[16 + base..20 + base]);
                out2.store_slice(&mut block[32 + base..36 + base]);
                out3.store_slice(&mut block[48 + base..52 + base]);
            }
        }
        64
    } else if values.len() >= 16 {
        let fuse_norm = values.len() == 16;
        for block in values.as_chunks_mut::<16>().0 {
            let ([mut d0, mut d1, mut d2, mut d3], f0) = hadamard_16_registers(neon, block, None);
            in_finite &= f0;
            if fuse_norm {
                d0 *= norm;
                d1 *= norm;
                d2 *= norm;
                d3 *= norm;
                finite_mask &= d0.is_finite() & d1.is_finite() & d2.is_finite() & d3.is_finite();
            }
            d0.store_slice(&mut block[0..4]);
            d1.store_slice(&mut block[4..8]);
            d2.store_slice(&mut block[8..12]);
            d3.store_slice(&mut block[12..16]);
        }
        16
    } else {
        for chunk in values.as_chunks_mut::<4>().0 {
            in_finite &= f32x4::from_slice(neon, chunk).is_finite();
            let [a0, a1, a2, a3] = *chunk;
            let (b0, b1, b2, b3) = (a0 + a1, a0 - a1, a2 + a3, a2 - a3);
            *chunk = [b0 + b2, b1 + b3, b0 - b2, b1 - b3];
        }
        4
    };
    Some(hadamard_finish(
        neon,
        values,
        half,
        norm,
        in_finite,
        finite_mask,
    ))
}

/// Fuses sign multiplication (or input load) with the normalized Sylvester Hadamard butterfly stages.
#[inline]
#[allow(
    clippy::too_many_lines,
    reason = "Keep register-fused 16/64-element sign-multiplication and Hadamard stages together."
)]
pub(super) fn hadamard_into(
    input: &[f32],
    signs: Option<&[f32]>,
    values: &mut [f32],
    normalization: f32,
) -> Option<Result<()>> {
    if values.len() < 4
        || input.len() != values.len()
        || signs.is_some_and(|s| s.len() < values.len())
    {
        return None;
    }
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let norm = f32x4::splat(neon, normalization);
    let mut in_finite = f32x4::splat(neon, 0.0).is_finite();
    let mut finite_mask = in_finite;
    let half = if values.len() >= 64 {
        let fuse_norm = values.len() == 64;
        for (block_idx, (in_block, out_block)) in input
            .as_chunks::<64>()
            .0
            .iter()
            .zip(values.as_chunks_mut::<64>().0)
            .enumerate()
        {
            let s_block = signs.map(|s| &s[block_idx * 64..(block_idx + 1) * 64]);
            let (q0, f0) =
                hadamard_16_registers(neon, &in_block[0..16], s_block.map(|s| &s[0..16]));
            let (q1, f1) =
                hadamard_16_registers(neon, &in_block[16..32], s_block.map(|s| &s[16..32]));
            let (q2, f2) =
                hadamard_16_registers(neon, &in_block[32..48], s_block.map(|s| &s[32..48]));
            let (q3, f3) =
                hadamard_16_registers(neon, &in_block[48..64], s_block.map(|s| &s[48..64]));
            in_finite &= f0 & f1 & f2 & f3;
            for i in 0..4 {
                let b0 = q0[i] + q1[i];
                let b1 = q0[i] - q1[i];
                let b2 = q2[i] + q3[i];
                let b3 = q2[i] - q3[i];
                let (mut out0, mut out1, mut out2, mut out3) = (b0 + b2, b1 + b3, b0 - b2, b1 - b3);
                if fuse_norm {
                    out0 *= norm;
                    out1 *= norm;
                    out2 *= norm;
                    out3 *= norm;
                    finite_mask &=
                        out0.is_finite() & out1.is_finite() & out2.is_finite() & out3.is_finite();
                }
                let base = i * 4;
                out0.store_slice(&mut out_block[base..base + 4]);
                out1.store_slice(&mut out_block[16 + base..20 + base]);
                out2.store_slice(&mut out_block[32 + base..36 + base]);
                out3.store_slice(&mut out_block[48 + base..52 + base]);
            }
        }
        64
    } else if values.len() >= 16 {
        let fuse_norm = values.len() == 16;
        for (block_idx, (in_block, out_block)) in input
            .as_chunks::<16>()
            .0
            .iter()
            .zip(values.as_chunks_mut::<16>().0)
            .enumerate()
        {
            let s_block = signs.map(|s| &s[block_idx * 16..(block_idx + 1) * 16]);
            let ([mut d0, mut d1, mut d2, mut d3], f0) =
                hadamard_16_registers(neon, in_block, s_block);
            in_finite &= f0;
            if fuse_norm {
                d0 *= norm;
                d1 *= norm;
                d2 *= norm;
                d3 *= norm;
                finite_mask &= d0.is_finite() & d1.is_finite() & d2.is_finite() & d3.is_finite();
            }
            d0.store_slice(&mut out_block[0..4]);
            d1.store_slice(&mut out_block[4..8]);
            d2.store_slice(&mut out_block[8..12]);
            d3.store_slice(&mut out_block[12..16]);
        }
        16
    } else {
        for (chunk_idx, (in_chunk, out_chunk)) in input
            .as_chunks::<4>()
            .0
            .iter()
            .zip(values.as_chunks_mut::<4>().0)
            .enumerate()
        {
            let v = signs.map_or_else(
                || f32x4::from_slice(neon, in_chunk),
                |s| {
                    f32x4::from_slice(neon, in_chunk)
                        * f32x4::from_slice(neon, &s[chunk_idx * 4..(chunk_idx + 1) * 4])
                },
            );
            in_finite &= v.is_finite();
            let [a0, a1, a2, a3] = v.to_array();
            let (b0, b1, b2, b3) = (a0 + a1, a0 - a1, a2 + a3, a2 - a3);
            *out_chunk = [b0 + b2, b1 + b3, b0 - b2, b1 - b3];
        }
        4
    };
    Some(hadamard_finish(
        neon,
        values,
        half,
        norm,
        in_finite,
        finite_mask,
    ))
}

#[inline]
#[allow(
    clippy::too_many_arguments,
    reason = "Grouped-value Hadamard block fuses head permutation, optional sign multiply, and stage-64 butterfly."
)]
pub(super) fn hadamard_grouped_block_into(
    input: &[f32],
    signs: Option<&[f32]>,
    block_idx: usize,
    heads_per_block: usize,
    repetitions: usize,
    ssm_keys: usize,
    ssm_head: usize,
    values: &mut [f32],
    normalization: f32,
) -> Option<Result<()>> {
    if values.len() < 64
        || ssm_head < 64
        || !ssm_head.is_multiple_of(64)
        || values.len() != heads_per_block.saturating_mul(ssm_head)
    {
        return None;
    }
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let norm = f32x4::splat(neon, normalization);
    let mut in_finite = f32x4::splat(neon, 0.0).is_finite();
    let mut finite_mask = in_finite;
    let fuse_norm = values.len() == 64;
    for (h, out_head) in values.chunks_exact_mut(ssm_head).enumerate() {
        let target_head = block_idx * heads_per_block + h;
        let key = target_head / repetitions;
        let repeat = target_head % repetitions;
        let source = (repeat * ssm_keys + key) * ssm_head;
        let target = target_head * ssm_head;
        let in_head = &input[source..source + ssm_head];
        let s_head = signs.map(|s| &s[target..target + ssm_head]);
        for (sub_idx, (in_block, out_block)) in in_head
            .as_chunks::<64>()
            .0
            .iter()
            .zip(out_head.as_chunks_mut::<64>().0)
            .enumerate()
        {
            let s_block = s_head.map(|s| &s[sub_idx * 64..(sub_idx + 1) * 64]);
            let (q0, f0) =
                hadamard_16_registers(neon, &in_block[0..16], s_block.map(|s| &s[0..16]));
            let (q1, f1) =
                hadamard_16_registers(neon, &in_block[16..32], s_block.map(|s| &s[16..32]));
            let (q2, f2) =
                hadamard_16_registers(neon, &in_block[32..48], s_block.map(|s| &s[32..48]));
            let (q3, f3) =
                hadamard_16_registers(neon, &in_block[48..64], s_block.map(|s| &s[48..64]));
            in_finite &= f0 & f1 & f2 & f3;
            for i in 0..4 {
                let b0 = q0[i] + q1[i];
                let b1 = q0[i] - q1[i];
                let b2 = q2[i] + q3[i];
                let b3 = q2[i] - q3[i];
                let (mut out0, mut out1, mut out2, mut out3) = (b0 + b2, b1 + b3, b0 - b2, b1 - b3);
                if fuse_norm {
                    out0 *= norm;
                    out1 *= norm;
                    out2 *= norm;
                    out3 *= norm;
                    finite_mask &=
                        out0.is_finite() & out1.is_finite() & out2.is_finite() & out3.is_finite();
                }
                let base = i * 4;
                out0.store_slice(&mut out_block[base..base + 4]);
                out1.store_slice(&mut out_block[16 + base..20 + base]);
                out2.store_slice(&mut out_block[32 + base..36 + base]);
                out3.store_slice(&mut out_block[48 + base..52 + base]);
            }
        }
    }
    Some(hadamard_finish(
        neon,
        values,
        64,
        norm,
        in_finite,
        finite_mask,
    ))
}

#[inline]
#[allow(
    clippy::suboptimal_flops,
    reason = "Preserve separate binary32 product roundings for both RMS contracts."
)]
fn hadamard_16_registers_rms_norm<const PINNED: bool>(
    neon: Neon,
    in_block: &[f32],
    w_block: &[f32],
    scale_vec: f32x4<Neon>,
    signs: Option<&[f32]>,
    norm_out: &mut [f32],
) -> (
    [f32x4<Neon>; 4],
    fearless_simd::mask32x4<Neon>,
    fearless_simd::mask32x4<Neon>,
) {
    let x0 = f32x4::from_slice(neon, &in_block[0..4]);
    let x1 = f32x4::from_slice(neon, &in_block[4..8]);
    let x2 = f32x4::from_slice(neon, &in_block[8..12]);
    let x3 = f32x4::from_slice(neon, &in_block[12..16]);
    let w0 = f32x4::from_slice(neon, &w_block[0..4]);
    let w1 = f32x4::from_slice(neon, &w_block[4..8]);
    let w2 = f32x4::from_slice(neon, &w_block[8..12]);
    let w3 = f32x4::from_slice(neon, &w_block[12..16]);
    let (n0, n1, n2, n3) = if PINNED {
        (
            (x0 * w0) * scale_vec,
            (x1 * w1) * scale_vec,
            (x2 * w2) * scale_vec,
            (x3 * w3) * scale_vec,
        )
    } else {
        (
            (x0 * scale_vec) * w0,
            (x1 * scale_vec) * w1,
            (x2 * scale_vec) * w2,
            (x3 * scale_vec) * w3,
        )
    };
    n0.store_slice(&mut norm_out[0..4]);
    n1.store_slice(&mut norm_out[4..8]);
    n2.store_slice(&mut norm_out[8..12]);
    n3.store_slice(&mut norm_out[12..16]);
    let norm_finite = n0.is_finite() & n1.is_finite() & n2.is_finite() & n3.is_finite();
    let (r0, r1, r2, r3) = signs.map_or((n0, n1, n2, n3), |s| {
        (
            n0 * f32x4::from_slice(neon, &s[0..4]),
            n1 * f32x4::from_slice(neon, &s[4..8]),
            n2 * f32x4::from_slice(neon, &s[8..12]),
            n3 * f32x4::from_slice(neon, &s[12..16]),
        )
    });
    let had_finite = r0.is_finite() & r1.is_finite() & r2.is_finite() & r3.is_finite();
    let [a0, a1, a2, a3] = transpose_4x4(r0, r1, r2, r3);
    let (b0, b1, b2, b3) = (a0 + a1, a0 - a1, a2 + a3, a2 - a3);
    let [v0, v1, v2, v3] = transpose_4x4(b0 + b2, b1 + b3, b0 - b2, b1 - b3);
    let (c0, c1, c2, c3) = (v0 + v1, v0 - v1, v2 + v3, v2 - v3);
    (
        [c0 + c2, c1 + c3, c0 - c2, c1 - c3],
        norm_finite,
        had_finite,
    )
}

/// Fuses weighted `RMSNorm` scaling, optional sign multiplication, and normalized
/// Sylvester Hadamard butterfly stages for one power-of-two block (`>= 16` elements).
#[inline]
#[allow(
    clippy::too_many_lines,
    reason = "Keep register-fused 16/64-element RMSNorm, sign multiplication, and Hadamard stages together."
)]
pub(super) fn rms_norm_hadamard_block_into<const PINNED: bool>(
    input: &[f32],
    weight: &[f32],
    scale: f32,
    signs: Option<&[f32]>,
    norm_out: &mut [f32],
    values: &mut [f32],
    normalization: f32,
) -> Option<Result<()>> {
    if values.len() < 16
        || input.len() != values.len()
        || weight.len() != values.len()
        || norm_out.len() != values.len()
        || signs.is_some_and(|s| s.len() < values.len())
    {
        return None;
    }
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let scale_vec = f32x4::splat(neon, scale);
    let norm = f32x4::splat(neon, normalization);
    let mut norm_finite = f32x4::splat(neon, 0.0).is_finite();
    let mut in_finite = norm_finite;
    let mut finite_mask = norm_finite;
    let half = if values.len() >= 64 {
        let fuse_norm = values.len() == 64;
        for (block_idx, (((in_block, w_block), n_block), out_block)) in input
            .as_chunks::<64>()
            .0
            .iter()
            .zip(weight.as_chunks::<64>().0)
            .zip(norm_out.as_chunks_mut::<64>().0)
            .zip(values.as_chunks_mut::<64>().0)
            .enumerate()
        {
            let s_block = signs.map(|s| &s[block_idx * 64..(block_idx + 1) * 64]);
            let (q0, nf0, f0) = hadamard_16_registers_rms_norm::<PINNED>(
                neon,
                &in_block[0..16],
                &w_block[0..16],
                scale_vec,
                s_block.map(|s| &s[0..16]),
                &mut n_block[0..16],
            );
            let (q1, nf1, f1) = hadamard_16_registers_rms_norm::<PINNED>(
                neon,
                &in_block[16..32],
                &w_block[16..32],
                scale_vec,
                s_block.map(|s| &s[16..32]),
                &mut n_block[16..32],
            );
            let (q2, nf2, f2) = hadamard_16_registers_rms_norm::<PINNED>(
                neon,
                &in_block[32..48],
                &w_block[32..48],
                scale_vec,
                s_block.map(|s| &s[32..48]),
                &mut n_block[32..48],
            );
            let (q3, nf3, f3) = hadamard_16_registers_rms_norm::<PINNED>(
                neon,
                &in_block[48..64],
                &w_block[48..64],
                scale_vec,
                s_block.map(|s| &s[48..64]),
                &mut n_block[48..64],
            );
            norm_finite &= nf0 & nf1 & nf2 & nf3;
            in_finite &= f0 & f1 & f2 & f3;
            for i in 0..4 {
                let b0 = q0[i] + q1[i];
                let b1 = q0[i] - q1[i];
                let b2 = q2[i] + q3[i];
                let b3 = q2[i] - q3[i];
                let (mut out0, mut out1, mut out2, mut out3) = (b0 + b2, b1 + b3, b0 - b2, b1 - b3);
                if fuse_norm {
                    out0 *= norm;
                    out1 *= norm;
                    out2 *= norm;
                    out3 *= norm;
                    finite_mask &=
                        out0.is_finite() & out1.is_finite() & out2.is_finite() & out3.is_finite();
                }
                let base = i * 4;
                out0.store_slice(&mut out_block[base..base + 4]);
                out1.store_slice(&mut out_block[16 + base..20 + base]);
                out2.store_slice(&mut out_block[32 + base..36 + base]);
                out3.store_slice(&mut out_block[48 + base..52 + base]);
            }
        }
        64
    } else {
        let fuse_norm = values.len() == 16;
        for (block_idx, (((in_block, w_block), n_block), out_block)) in input
            .as_chunks::<16>()
            .0
            .iter()
            .zip(weight.as_chunks::<16>().0)
            .zip(norm_out.as_chunks_mut::<16>().0)
            .zip(values.as_chunks_mut::<16>().0)
            .enumerate()
        {
            let s_block = signs.map(|s| &s[block_idx * 16..(block_idx + 1) * 16]);
            let ([mut d0, mut d1, mut d2, mut d3], nf0, f0) =
                hadamard_16_registers_rms_norm::<PINNED>(
                    neon, in_block, w_block, scale_vec, s_block, n_block,
                );
            norm_finite &= nf0;
            in_finite &= f0;
            if fuse_norm {
                d0 *= norm;
                d1 *= norm;
                d2 *= norm;
                d3 *= norm;
                finite_mask &= d0.is_finite() & d1.is_finite() & d2.is_finite() & d3.is_finite();
            }
            d0.store_slice(&mut out_block[0..4]);
            d1.store_slice(&mut out_block[4..8]);
            d2.store_slice(&mut out_block[8..12]);
            d3.store_slice(&mut out_block[12..16]);
        }
        16
    };
    if !norm_finite.all_true() {
        return Some(Err(crate::invalid("nonfinite tensor or activation value")));
    }
    Some(hadamard_finish(
        neon,
        values,
        half,
        norm,
        in_finite,
        finite_mask,
    ))
}

#[inline]
fn hadamard_16_registers_swiglu_read(
    neon: Neon,
    gate_block: &[f32],
    up_block: &[f32],
    signs: Option<&[f32]>,
) -> ([f32x4<Neon>; 4], fearless_simd::mask32x4<Neon>) {
    let zero = f32x4::splat(neon, 0.0);
    let one = f32x4::splat(neon, 1.0);
    let g0 = f32x4::from_slice(neon, &gate_block[0..4]);
    let g1 = f32x4::from_slice(neon, &gate_block[4..8]);
    let g2 = f32x4::from_slice(neon, &gate_block[8..12]);
    let g3 = f32x4::from_slice(neon, &gate_block[12..16]);
    let u0 = f32x4::from_slice(neon, &up_block[0..4]);
    let u1 = f32x4::from_slice(neon, &up_block[4..8]);
    let u2 = f32x4::from_slice(neon, &up_block[8..12]);
    let u3 = f32x4::from_slice(neon, &up_block[12..16]);
    let s0 = (g0 * u0) / (one + super::pinned_activations::vector_exp_x4(neon, zero - g0));
    let s1 = (g1 * u1) / (one + super::pinned_activations::vector_exp_x4(neon, zero - g1));
    let s2 = (g2 * u2) / (one + super::pinned_activations::vector_exp_x4(neon, zero - g2));
    let s3 = (g3 * u3) / (one + super::pinned_activations::vector_exp_x4(neon, zero - g3));
    let (r0, r1, r2, r3) = signs.map_or((s0, s1, s2, s3), |s| {
        (
            s0 * f32x4::from_slice(neon, &s[0..4]),
            s1 * f32x4::from_slice(neon, &s[4..8]),
            s2 * f32x4::from_slice(neon, &s[8..12]),
            s3 * f32x4::from_slice(neon, &s[12..16]),
        )
    });
    let finite = r0.is_finite() & r1.is_finite() & r2.is_finite() & r3.is_finite();
    let [a0, a1, a2, a3] = transpose_4x4(r0, r1, r2, r3);
    let (b0, b1, b2, b3) = (a0 + a1, a0 - a1, a2 + a3, a2 - a3);
    let [v0, v1, v2, v3] = transpose_4x4(b0 + b2, b1 + b3, b0 - b2, b1 - b3);
    let (c0, c1, c2, c3) = (v0 + v1, v0 - v1, v2 + v3, v2 - v3);
    ([c0 + c2, c1 + c3, c0 - c2, c1 - c3], finite)
}

/// Fuses pinned `SwiGLU`, optional sign multiplication, and normalized Sylvester Hadamard
/// butterfly stages for one power-of-two block (`>= 16` elements) without mutating `up`.
#[inline]
#[allow(
    clippy::too_many_lines,
    reason = "Keep register-fused 16/64-element SwiGLU, sign multiplication, and Hadamard stages together."
)]
pub(super) fn swiglu_hadamard_block_read_into(
    gate: &[f32],
    up: &[f32],
    signs: Option<&[f32]>,
    values: &mut [f32],
    normalization: f32,
) -> Option<Result<()>> {
    if values.len() < 16
        || gate.len() != values.len()
        || up.len() != values.len()
        || signs.is_some_and(|s| s.len() < values.len())
    {
        return None;
    }
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let norm = f32x4::splat(neon, normalization);
    let mut in_finite = f32x4::splat(neon, 0.0).is_finite();
    let mut finite_mask = in_finite;
    let half = if values.len() >= 64 {
        let fuse_norm = values.len() == 64;
        for (block_idx, ((g_block, u_block), out_block)) in gate
            .as_chunks::<64>()
            .0
            .iter()
            .zip(up.as_chunks::<64>().0)
            .zip(values.as_chunks_mut::<64>().0)
            .enumerate()
        {
            let s_block = signs.map(|s| &s[block_idx * 64..(block_idx + 1) * 64]);
            let (q0, f0) = hadamard_16_registers_swiglu_read(
                neon,
                &g_block[0..16],
                &u_block[0..16],
                s_block.map(|s| &s[0..16]),
            );
            let (q1, f1) = hadamard_16_registers_swiglu_read(
                neon,
                &g_block[16..32],
                &u_block[16..32],
                s_block.map(|s| &s[16..32]),
            );
            let (q2, f2) = hadamard_16_registers_swiglu_read(
                neon,
                &g_block[32..48],
                &u_block[32..48],
                s_block.map(|s| &s[32..48]),
            );
            let (q3, f3) = hadamard_16_registers_swiglu_read(
                neon,
                &g_block[48..64],
                &u_block[48..64],
                s_block.map(|s| &s[48..64]),
            );
            in_finite &= f0 & f1 & f2 & f3;
            for i in 0..4 {
                let b0 = q0[i] + q1[i];
                let b1 = q0[i] - q1[i];
                let b2 = q2[i] + q3[i];
                let b3 = q2[i] - q3[i];
                let (mut out0, mut out1, mut out2, mut out3) = (b0 + b2, b1 + b3, b0 - b2, b1 - b3);
                if fuse_norm {
                    out0 *= norm;
                    out1 *= norm;
                    out2 *= norm;
                    out3 *= norm;
                    finite_mask &=
                        out0.is_finite() & out1.is_finite() & out2.is_finite() & out3.is_finite();
                }
                let base = i * 4;
                out0.store_slice(&mut out_block[base..base + 4]);
                out1.store_slice(&mut out_block[16 + base..20 + base]);
                out2.store_slice(&mut out_block[32 + base..36 + base]);
                out3.store_slice(&mut out_block[48 + base..52 + base]);
            }
        }
        64
    } else {
        let fuse_norm = values.len() == 16;
        for (block_idx, ((g_block, u_block), out_block)) in gate
            .as_chunks::<16>()
            .0
            .iter()
            .zip(up.as_chunks::<16>().0)
            .zip(values.as_chunks_mut::<16>().0)
            .enumerate()
        {
            let s_block = signs.map(|s| &s[block_idx * 16..(block_idx + 1) * 16]);
            let ([mut d0, mut d1, mut d2, mut d3], f0) =
                hadamard_16_registers_swiglu_read(neon, g_block, u_block, s_block);
            in_finite &= f0;
            if fuse_norm {
                d0 *= norm;
                d1 *= norm;
                d2 *= norm;
                d3 *= norm;
                finite_mask &= d0.is_finite() & d1.is_finite() & d2.is_finite() & d3.is_finite();
            }
            d0.store_slice(&mut out_block[0..4]);
            d1.store_slice(&mut out_block[4..8]);
            d2.store_slice(&mut out_block[8..12]);
            d3.store_slice(&mut out_block[12..16]);
        }
        16
    };
    Some(hadamard_finish(
        neon,
        values,
        half,
        norm,
        in_finite,
        finite_mask,
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        block16_all_finite_le, dot, dot_consumption_order, dot_groups, dot_i16_block,
        dot_ptq1_groups, dot_q4_0_block, dot_q4_0_eight_rows, dot_q4_0_four_rows, dot_q4_1_block,
        dot_q8_0_block, dot_q8_0_eight_rows, dot_q8_0_four_rows, hadamard, hadamard_into,
        round_bf16_into, round_f16_into, round_f16_slice,
    };

    type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

    fn reference(packed: &[u8; 32], activations: &[i8; 128]) -> i32 {
        let mut sum = 0;
        for (symbols, values) in packed.iter().zip(activations.as_chunks::<4>().0) {
            for (digit, value) in values.iter().enumerate() {
                let code = (symbols >> (digit * 2)) & 3;
                sum += (i32::from(code) - 1) * i32::from(*value);
            }
        }
        sum
    }

    fn consumption_order(activations: &[i8; 128]) -> [i8; 128] {
        let mut output = [0; 128];
        for (source, value) in activations.iter().enumerate() {
            let packed_byte = source / 4;
            let half = packed_byte / 16;
            let digit = source % 4;
            output[half * 64 + digit * 16 + packed_byte % 16] = *value;
        }
        output
    }

    #[test]
    fn simd_neon_packed_consumption_exhaustive_bytes_and_extreme_lanes() {
        let original = std::array::from_fn(|index: usize| {
            i8::from_ne_bytes([index.to_le_bytes()[0].wrapping_mul(197).wrapping_add(13)])
        });
        let prepared = consumption_order(&original);
        for position in 0..32 {
            for byte in 0..=u8::MAX {
                let mut packed = [0x55; 32];
                packed[position] = byte;
                assert_eq!(
                    dot_consumption_order(&packed, &prepared),
                    reference(&packed, &original)
                );
            }
        }
        for position in 0..128 {
            for code in 0..=3 {
                let mut packed = [0x55; 32];
                packed[position / 4] &= !(3 << (2 * (position % 4)));
                packed[position / 4] |= code << (2 * (position % 4));
                for value in [i8::MIN, -127, -1, 0, 1, i8::MAX] {
                    let mut original = [0; 128];
                    original[position] = value;
                    assert_eq!(
                        dot_consumption_order(&packed, &consumption_order(&original)),
                        (i32::from(code) - 1) * i32::from(value)
                    );
                }
            }
        }
        for value in [i8::MIN, i8::MAX] {
            let original = [value; 128];
            assert_eq!(
                dot_consumption_order(&[0xff; 32], &consumption_order(&original)),
                reference(&[0xff; 32], &original)
            );
        }
    }

    #[test]
    fn simd_neon_integer_dot_exhaustive_packed_bytes() {
        let mut activations = std::array::from_fn(|index: usize| {
            i8::from_ne_bytes([index.to_le_bytes()[0].wrapping_mul(197).wrapping_add(13)])
        });
        activations[..6].copy_from_slice(&[i8::MIN, -127, -1, 0, 1, i8::MAX]);
        for position in 0..32 {
            for byte in 0..=u8::MAX {
                let mut packed = [0x55; 32];
                packed[position] = byte;
                assert_eq!(dot(&packed, &activations), reference(&packed, &activations));
            }
        }
    }

    #[test]
    fn simd_neon_integer_dot_each_lane_and_extremes() {
        for position in 0..128 {
            for code in 0..=3 {
                let mut packed = [0x55; 32];
                let shift = 2 * (position % 4);
                packed[position / 4] &= !(3 << shift);
                packed[position / 4] |= code << shift;
                for value in [i8::MIN, -127, -1, 0, 1, i8::MAX] {
                    let mut activations = [0; 128];
                    activations[position] = value;
                    let expected = (i32::from(code) - 1) * i32::from(value);
                    assert_eq!(dot(&packed, &activations), expected);
                }
            }
        }
        for byte in 0..=u8::MAX {
            for value in [i8::MIN, -127, -1, 0, 1, i8::MAX] {
                let packed = [byte; 32];
                let activations = [value; 128];
                assert_eq!(dot(&packed, &activations), reference(&packed, &activations));
            }
        }
        assert_eq!(dot(&[0xff; 32], &[i8::MIN; 128]), -32_768);
        assert_eq!(dot(&[0xff; 32], &[i8::MAX; 128]), 32_512);
    }

    #[test]
    fn simd_neon_pq2_pair_accumulation_extremal_mixed_lanes() {
        // Exercise cancellation across widened lane halves, digit planes and
        // both packed loads, including the fourth symbol's maximum products.
        // The reference accumulates each original symbol independently in i32.
        for mask in 0..=u8::MAX {
            for byte in [0x00, 0xff, 0x1b, 0xe4] {
                let packed = [byte; 32];
                let activations = std::array::from_fn(|index| {
                    if mask & (1 << ((index % 8 + index / 16) % 8)) == 0 {
                        i8::MIN
                    } else {
                        i8::MAX
                    }
                });
                let expected = reference(&packed, &activations);
                assert_eq!(dot(&packed, &activations), expected);
                assert_eq!(
                    dot_consumption_order(&packed, &consumption_order(&activations)),
                    expected
                );
            }
        }
    }

    #[test]
    fn simd_neon_integer_dot_unaligned_random_blocks() -> TestResult {
        #[repr(align(16))]
        struct Aligned<T>(T);

        let mut packed_storage = Aligned([0; 48]);
        let mut activation_storage = Aligned([0; 144]);
        let mut state = 0x9e37_79b9_u32;
        for _ in 0..4096 {
            let mut next_byte = || {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state.to_le_bytes()[0]
            };
            for byte in &mut packed_storage.0[1..33] {
                *byte = next_byte();
            }
            for value in &mut activation_storage.0[3..131] {
                *value = i8::from_ne_bytes([next_byte()]);
            }
            let packed = <&[u8; 32]>::try_from(&packed_storage.0[1..33])?;
            let activations = <&[i8; 128]>::try_from(&activation_storage.0[3..131])?;
            assert_eq!(dot(packed, activations), reference(packed, activations));
            assert_eq!(
                dot_consumption_order(packed, &consumption_order(activations)),
                reference(packed, activations)
            );
            let groups = <&[[i8; 32]; 4]>::try_from(activations.as_chunks::<32>().0)?;
            let sums = dot_groups(packed, groups.each_ref());
            for (group, sum) in sums.into_iter().enumerate() {
                let mut isolated = [0; 128];
                isolated[group * 32..group * 32 + 32].copy_from_slice(&groups[group]);
                assert_eq!(sum, reference(packed, &isolated));
            }
        }
        Ok(())
    }

    fn ptq1_reference(packed: &[u8; 28], values: &[[i8; 32]; 4]) -> crate::Result<[i32; 4]> {
        let weights = super::super::ptq1_values(packed)?;
        Ok(std::array::from_fn(|group| {
            weights[group * 32..group * 32 + 32]
                .iter()
                .zip(values[group])
                .map(|(&weight, value)| i32::from(weight) * i32::from(value))
                .sum()
        }))
    }

    #[test]
    fn simd_neon_ptq1_exhaustive_native_bytes() -> TestResult {
        let values = std::array::from_fn(|group: usize| {
            std::array::from_fn(|lane: usize| {
                i8::from_ne_bytes([(group * 32 + lane).to_le_bytes()[0].wrapping_mul(197)])
            })
        });
        for position in 0..26 {
            for byte in 0..=u8::MAX {
                let mut packed = [0; 28];
                packed[26..].copy_from_slice(&0x3c00_u16.to_le_bytes());
                packed[position] = byte;
                let expected = ptq1_reference(&packed, &values);
                let actual = dot_ptq1_groups(&packed, values.each_ref());
                assert_eq!(actual.is_ok(), expected.is_ok());
                if let Ok(expected) = expected {
                    assert_eq!(actual?, expected);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn simd_neon_ptq1_each_lane_and_extremes() -> TestResult {
        for position in 0..128 {
            for code in 0..=2 {
                let mut pq2 = [0x55; 34];
                pq2[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
                let shift = 2 * (position % 4);
                pq2[2 + position / 4] &= !(3 << shift);
                pq2[2 + position / 4] |= code << shift;
                let packed = bonsai_ternary::pack_ptq(&pq2)?;
                for value in [i8::MIN, -127, -1, 0, 1, i8::MAX] {
                    let mut values = [[0; 32]; 4];
                    values[position / 32][position % 32] = value;
                    assert_eq!(
                        dot_ptq1_groups(&packed, values.each_ref())?,
                        ptq1_reference(&packed, &values)?
                    );
                }
            }
        }
        for code in [0x00, 0x55, 0xaa] {
            let mut pq2 = [code; 34];
            pq2[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
            let packed = bonsai_ternary::pack_ptq(&pq2)?;
            for value in [i8::MIN, -127, -1, 0, 1, i8::MAX] {
                let values = [[value; 32]; 4];
                assert_eq!(
                    dot_ptq1_groups(&packed, values.each_ref())?,
                    ptq1_reference(&packed, &values)?
                );
            }
        }
        Ok(())
    }

    #[test]
    fn simd_neon_ptq1_unaligned_random_blocks() -> TestResult {
        #[repr(align(16))]
        struct Aligned<T>(T);

        let mut storage = Aligned([0; 44]);
        let mut activation_storage = Aligned([0; 144]);
        let mut state = 0x85eb_ca6b_u32;
        for _ in 0..4096 {
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state.to_le_bytes()[0]
            };
            let mut pq2 = [0; 34];
            pq2[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
            for index in 0..128 {
                pq2[2 + index / 4] |= (next() % 3) << (2 * (index % 4));
            }
            storage.0[1..29].copy_from_slice(&bonsai_ternary::pack_ptq(&pq2)?);
            for value in &mut activation_storage.0[3..131] {
                *value = i8::from_ne_bytes([next()]);
            }
            let packed = <&[u8; 28]>::try_from(&storage.0[1..29])?;
            let groups =
                <&[[i8; 32]; 4]>::try_from(activation_storage.0[3..131].as_chunks::<32>().0)?;
            assert_eq!(
                dot_ptq1_groups(packed, groups.each_ref())?,
                ptq1_reference(packed, groups)?
            );
        }
        Ok(())
    }

    fn fixture_take<'a>(bytes: &mut &'a [u8], count: usize) -> crate::Result<&'a [u8]> {
        let (head, tail) = bytes
            .split_at_checked(count)
            .ok_or("truncated C++ codec fixture")?;
        *bytes = tail;
        Ok(head)
    }

    fn fixture_count(bytes: &mut &[u8]) -> crate::Result<usize> {
        Ok(usize::try_from(u32::from_le_bytes(
            fixture_take(bytes, 4)?.try_into()?,
        ))?)
    }

    #[test]
    fn simd_neon_ptq1_matches_immutable_cpp_packed_fixture() -> TestResult {
        // Captured native C++ codec bytes are independent of this Rust decoder.
        // The fixture contains no C++ matvec or activation-quantizer outputs.
        let mut bytes = include_bytes!("../../tests/fixtures/ptq-reference.bin").as_slice();
        assert_eq!(fixture_take(&mut bytes, 8)?, b"PTPAR001");
        let half_count = fixture_count(&mut bytes)?;
        assert_eq!(half_count, 26);
        fixture_take(&mut bytes, half_count * 6)?;
        let cases = fixture_count(&mut bytes)?;
        assert_eq!(cases, 161);
        let varied = std::array::from_fn(|group: usize| {
            std::array::from_fn(|index: usize| {
                i8::from_ne_bytes([(group * 32 + index).to_le_bytes()[0].wrapping_mul(197)])
            })
        });
        let mut row_count = 0;
        for _ in 0..cases {
            let rows = fixture_count(&mut bytes)?;
            let blocks = fixture_count(&mut bytes)?;
            assert!((1..=3).contains(&rows));
            assert!([2, 40, 48, 136].contains(&blocks));
            row_count += rows;
            let length = rows.checked_mul(blocks).ok_or("fixture length overflow")?;
            let pq2 = fixture_take(&mut bytes, length * 34)?.as_chunks::<34>().0;
            let ptq1 = fixture_take(&mut bytes, length * 28)?.as_chunks::<28>().0;
            for (pq2, ptq1) in pq2.iter().zip(ptq1) {
                for values in [varied, [[i8::MIN; 32]; 4], [[i8::MAX; 32]; 4], [[0; 32]; 4]] {
                    let expected: [i32; 4] = std::array::from_fn(|group| {
                        values[group]
                            .iter()
                            .enumerate()
                            .map(|(lane, value)| {
                                let index = group * 32 + lane;
                                let code = (pq2[2 + index / 4] >> (2 * (index % 4))) & 3;
                                (i32::from(code) - 1) * i32::from(*value)
                            })
                            .sum()
                    });
                    assert_eq!(
                        dot_groups(pq2[2..].try_into()?, values.each_ref()),
                        expected
                    );
                    assert_eq!(dot_ptq1_groups(ptq1, values.each_ref())?, expected);
                }
            }
        }
        assert_eq!(row_count, 169);
        assert_eq!(bytes.len(), 0);
        Ok(())
    }

    #[test]
    fn simd_neon_block16_all_finite_le_lanes_and_extremes() {
        assert!(block16_all_finite_le(&[0.0; 16], -0.0));
        assert!(block16_all_finite_le(&[-0.0; 16], 0.0));
        assert!(block16_all_finite_le(&[f32::MIN; 16], f32::MAX));
        assert!(block16_all_finite_le(&[f32::MAX; 16], f32::MAX));
        for lane in 0..16 {
            let mut block = [1.0_f32; 16];
            assert!(block16_all_finite_le(&block, 1.0));
            block[lane] = f32::from_bits(1.0_f32.to_bits() + 1);
            assert!(!block16_all_finite_le(&block, 1.0));
            for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                block[lane] = bad;
                assert!(!block16_all_finite_le(&block, f32::MAX));
            }
        }
    }

    #[test]
    fn simd_neon_hadamard_matches_scalar_across_power_of_two_blocks() -> TestResult {
        for len in [4_usize, 8, 16, 32, 64, 128, 256, 512, 1024, 2048] {
            let norm = 1.0 / super::super::integer_float(i32::try_from(len)?).sqrt();
            let mut actual: Vec<f32> = (0..len)
                .map(|idx| {
                    let s = i32::try_from((idx.wrapping_mul(73) ^ (idx >> 2)) % 251).unwrap_or(0);
                    super::super::integer_float(s - 125) * 0.03125
                })
                .collect();
            let mut expected = actual.clone();
            let mut half = 1;
            while half < len {
                for block in expected.chunks_exact_mut(half * 2) {
                    let (left_half, right_half) = block.split_at_mut(half);
                    for (left, right) in left_half.iter_mut().zip(right_half) {
                        let l = *left;
                        let r = *right;
                        *left = l + r;
                        *right = l - r;
                    }
                }
                half *= 2;
            }
            for value in &mut expected {
                *value *= norm;
            }
            let input_snapshot = actual.clone();
            hadamard(&mut actual, norm).ok_or("missing baseline NEON token")??;
            let actual_bits: Vec<u32> = actual.iter().copied().map(f32::to_bits).collect();
            let expected_bits: Vec<u32> = expected.iter().copied().map(f32::to_bits).collect();
            assert_eq!(actual_bits, expected_bits, "Hadamard mismatch at len={len}");

            let mut into_actual = vec![0.0_f32; len];
            hadamard_into(&input_snapshot, None, &mut into_actual, norm)
                .ok_or("missing baseline NEON token")??;
            let into_bits: Vec<u32> = into_actual.iter().copied().map(f32::to_bits).collect();
            assert_eq!(
                into_bits, expected_bits,
                "hadamard_into(None) mismatch at len={len}"
            );

            let signs: Vec<f32> = (0..len)
                .map(|idx| if idx % 3 == 0 { -1.0 } else { 1.0 })
                .collect();
            let mut signed_expected: Vec<f32> = input_snapshot
                .iter()
                .zip(&signs)
                .map(|(v, s)| *v * *s)
                .collect();
            hadamard(&mut signed_expected, norm).ok_or("missing baseline NEON token")??;
            let mut signed_into = vec![0.0_f32; len];
            hadamard_into(&input_snapshot, Some(&signs), &mut signed_into, norm)
                .ok_or("missing baseline NEON token")??;
            let signed_expected_bits: Vec<u32> =
                signed_expected.iter().copied().map(f32::to_bits).collect();
            let signed_into_bits: Vec<u32> =
                signed_into.iter().copied().map(f32::to_bits).collect();
            assert_eq!(
                signed_into_bits, signed_expected_bits,
                "hadamard_into(Some(signs)) mismatch at len={len}"
            );

            let mut nonfinite = vec![1.0_f32; len];
            nonfinite[len - 1] = f32::NAN;
            assert!(
                hadamard(&mut nonfinite, norm)
                    .ok_or("missing baseline NEON token")?
                    .is_err()
            );

            let mut overflow = vec![f32::MAX; len];
            assert!(
                hadamard(&mut overflow, norm)
                    .ok_or("missing baseline NEON token")?
                    .is_err()
            );
        }
        Ok(())
    }

    #[test]
    fn simd_neon_round_into_matches_in_place_and_scalar() -> TestResult {
        for len in [1_usize, 3, 4, 7, 16, 65, 256] {
            let input: Vec<f32> = (0..len)
                .map(|idx| {
                    let s = i32::try_from((idx.wrapping_mul(73) ^ (idx >> 2)) % 251).unwrap_or(0);
                    super::super::integer_float(s - 125) * 0.03125
                })
                .collect();
            let bf16_scalar: Vec<f32> = input
                .iter()
                .copied()
                .map(super::super::round_bf16)
                .collect();
            let mut bf16_into = vec![0.0_f32; len];
            round_bf16_into(&input, &mut bf16_into).ok_or("missing baseline NEON token")??;
            assert_eq!(
                bf16_into
                    .iter()
                    .copied()
                    .map(f32::to_bits)
                    .collect::<Vec<u32>>(),
                bf16_scalar
                    .iter()
                    .copied()
                    .map(f32::to_bits)
                    .collect::<Vec<u32>>(),
                "round_bf16_into mismatch at len={len}"
            );

            let mut f16_in_place = input.clone();
            assert_eq!(round_f16_slice(&mut f16_in_place), Some(true));
            let mut f16_into = vec![0.0_f32; len];
            round_f16_into(&input, &mut f16_into).ok_or("missing baseline NEON token")??;
            assert_eq!(
                f16_into
                    .iter()
                    .copied()
                    .map(f32::to_bits)
                    .collect::<Vec<u32>>(),
                f16_in_place
                    .iter()
                    .copied()
                    .map(f32::to_bits)
                    .collect::<Vec<u32>>(),
                "round_f16_into mismatch at len={len}"
            );
        }
        Ok(())
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "Cover Q8_0, Q4_0, Q4_1, and i16 block dots plus 4/8-row NEON batching together."
    )]
    fn simd_neon_q8_0_and_q4_0_block_and_multirow_parity() -> TestResult {
        for value in [i8::MIN, -127, -1, 0, 1, i8::MAX] {
            let activations = [value; 32];
            for byte in [0_u8, 1, 0x7f, 0x80, 0xfe, 0xff] {
                let w8 = [byte; 32];
                let expected_q8: i32 = w8
                    .iter()
                    .zip(activations)
                    .map(|(&w, a)| i32::from(i8::from_ne_bytes([w])) * i32::from(a))
                    .sum();
                assert_eq!(dot_q8_0_block(&w8, &activations), expected_q8);

                let w4 = [byte; 16];
                let expected_q4_0: i32 = (0..32)
                    .map(|idx| {
                        (i32::from((w4[idx % 16] >> (4 * (idx / 16))) & 15) - 8)
                            * i32::from(activations[idx])
                    })
                    .sum();
                assert_eq!(dot_q4_0_block(&w4, &activations), expected_q4_0);

                let expected_q4_1: i32 = (0..32)
                    .map(|idx| {
                        i32::from((w4[idx % 16] >> (4 * (idx / 16))) & 15)
                            * i32::from(activations[idx])
                    })
                    .sum();
                assert_eq!(dot_q4_1_block(&w4, &activations), expected_q4_1);
            }
            let w5: [i16; 32] =
                std::array::from_fn(|idx| i16::try_from((idx * 7) % 32).unwrap_or(0) - 16);
            let expected_i16: i32 = w5
                .iter()
                .zip(activations)
                .map(|(&w, a)| i32::from(w) * i32::from(a))
                .sum();
            assert_eq!(dot_i16_block(&w5, &activations), expected_i16);
        }

        let input_blocks = [
            super::super::Q8Block {
                scale: 0.25,
                sum: 0.0,
                values: std::array::from_fn(|idx| {
                    i8::from_ne_bytes([idx.to_le_bytes()[0].wrapping_mul(37).wrapping_add(5)])
                }),
            },
            super::super::Q8Block {
                scale: -0.125,
                sum: 0.0,
                values: std::array::from_fn(|idx| {
                    i8::from_ne_bytes([idx.to_le_bytes()[0].wrapping_mul(73).wrapping_add(19)])
                }),
            },
        ];

        let q8_rows: Vec<Vec<u8>> = (0_u8..8)
            .map(|r| {
                let mut row = Vec::new();
                for b in 0_u8..2 {
                    let scale = [0x3400_u16, 0xb400, 0x3800, 0x3c00][usize::from(r + b) % 4];
                    row.extend_from_slice(&scale.to_le_bytes());
                    for idx in 0_u8..32 {
                        row.push(
                            r.wrapping_mul(17)
                                .wrapping_add(b.wrapping_mul(31))
                                .wrapping_add(idx.wrapping_mul(9)),
                        );
                    }
                }
                row
            })
            .collect();
        let q8_refs: [&[u8]; 8] = std::array::from_fn(|r| q8_rows[r].as_slice());
        let q8_four = dot_q8_0_four_rows(
            [q8_refs[0], q8_refs[1], q8_refs[2], q8_refs[3]],
            &input_blocks,
        )?;
        let q8_eight = dot_q8_0_eight_rows(q8_refs, &input_blocks)?;
        for r in 0..8 {
            let expected = super::super::dot_legacy(super::super::Q8_0, q8_refs[r], &input_blocks)?;
            assert_eq!(q8_eight[r].to_bits(), expected.to_bits());
            if r < 4 {
                assert_eq!(q8_four[r].to_bits(), expected.to_bits());
            }
        }

        let q4_rows: Vec<Vec<u8>> = (0_u8..8)
            .map(|r| {
                let mut row = Vec::new();
                for b in 0_u8..2 {
                    let scale = [0x3400_u16, 0xb400, 0x3800, 0x3c00][usize::from(r + b) % 4];
                    row.extend_from_slice(&scale.to_le_bytes());
                    for idx in 0_u8..16 {
                        row.push(
                            r.wrapping_mul(29)
                                .wrapping_add(b.wrapping_mul(13))
                                .wrapping_add(idx.wrapping_mul(7)),
                        );
                    }
                }
                row
            })
            .collect();
        let q4_refs: [&[u8]; 8] = std::array::from_fn(|r| q4_rows[r].as_slice());
        let q4_four = dot_q4_0_four_rows(
            [q4_refs[0], q4_refs[1], q4_refs[2], q4_refs[3]],
            &input_blocks,
        )?;
        let q4_eight = dot_q4_0_eight_rows(q4_refs, &input_blocks)?;
        for r in 0..8 {
            let expected = super::super::dot_legacy(super::super::Q4_0, q4_refs[r], &input_blocks)?;
            assert_eq!(q4_eight[r].to_bits(), expected.to_bits());
            if r < 4 {
                assert_eq!(q4_four[r].to_bits(), expected.to_bits());
            }
        }
        Ok(())
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        clippy::as_conversions,
        clippy::cast_precision_loss,
        clippy::suboptimal_flops,
        reason = "Cover fused RMSNorm, SwiGLU, and grouped Hadamard block sizes in one test."
    )]
    fn fused_rms_norm_and_swiglu_hadamard_match_unfused_zero_ulp() -> TestResult {
        for block in [16_usize, 64, 256, 1024] {
            let input: Vec<f32> = (0..block * 2)
                .map(|i| {
                    let raw = u16::try_from((i * 197 + 13) % 1024).unwrap_or(0);
                    (f32::from(raw) - 512.0) * 0.015_625
                })
                .collect();
            let weight: Vec<f32> = (0..block * 2)
                .map(|i| {
                    let raw = u16::try_from((i * 73 + 29) % 512).unwrap_or(0);
                    f32::from(raw).mul_add(0.001_953_125, 0.5)
                })
                .collect();
            let signs: Vec<f32> = (0..block * 2)
                .map(|i| if (i * 31 + 7) % 2 == 0 { 1.0 } else { -1.0 })
                .collect();

            for sign_opt in [None, Some(signs.as_slice())] {
                let mut ref_norm = Vec::new();
                let mut ref_rot = vec![0.0_f32; block * 2];
                super::super::rms_norm_into(&input, &weight, 1e-6, &mut ref_norm)?;
                for (idx, out_b) in ref_rot.chunks_exact_mut(block).enumerate() {
                    let off = idx * block;
                    super::super::hadamard_into(
                        &ref_norm[off..off + block],
                        sign_opt.map(|s| &s[off..off + block]),
                        out_b,
                    )?;
                }
                let mut fused_norm = Vec::new();
                let mut fused_rot = Vec::new();
                super::super::rms_norm_hadamard_into::<false>(
                    &input,
                    &weight,
                    1e-6,
                    block,
                    sign_opt,
                    &super::super::RowParallelism::default(),
                    &mut fused_norm,
                    &mut fused_rot,
                )?;
                for (a, b) in fused_norm.iter().zip(&ref_norm) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
                for (a, b) in fused_rot.iter().zip(&ref_rot) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }

                super::super::rms_norm_pinned_into(&input, &weight, 1e-6, &mut ref_norm)?;
                for (idx, out_b) in ref_rot.chunks_exact_mut(block).enumerate() {
                    let off = idx * block;
                    super::super::hadamard_into(
                        &ref_norm[off..off + block],
                        sign_opt.map(|s| &s[off..off + block]),
                        out_b,
                    )?;
                }
                super::super::rms_norm_hadamard_into::<true>(
                    &input,
                    &weight,
                    1e-6,
                    block,
                    sign_opt,
                    &super::super::RowParallelism::default(),
                    &mut fused_norm,
                    &mut fused_rot,
                )?;
                for (a, b) in fused_norm.iter().zip(&ref_norm) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
                for (a, b) in fused_rot.iter().zip(&ref_rot) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }

                let delta: Vec<f32> = (0..block * 2)
                    .map(|i| {
                        let raw = u16::try_from((i * 109 + 41) % 512).unwrap_or(0);
                        (f32::from(raw) - 256.0) * 0.007_812_5
                    })
                    .collect();
                let mut ref_added = input.clone();
                assert!(super::super::add_slice_in_place_all_finite(
                    &mut ref_added,
                    &delta
                ));
                let mut unfused_added = input.clone();
                super::super::add_slice_in_place(&mut unfused_added, &delta);
                for (a, b) in ref_added.iter().zip(&unfused_added) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }

                super::super::rms_norm_into(&ref_added, &weight, 1e-6, &mut ref_norm)?;
                for (idx, out_b) in ref_rot.chunks_exact_mut(block).enumerate() {
                    let off = idx * block;
                    super::super::hadamard_into(
                        &ref_norm[off..off + block],
                        sign_opt.map(|s| &s[off..off + block]),
                        out_b,
                    )?;
                }
                let mut add_norm_vec = input.clone();
                super::super::add_rms_norm_into(
                    &mut add_norm_vec,
                    &delta,
                    &weight,
                    1e-6,
                    &mut fused_norm,
                )?;
                for (a, b) in add_norm_vec.iter().zip(&ref_added) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
                for (a, b) in fused_norm.iter().zip(&ref_norm) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
                let mut add_had_vec = input.clone();
                super::super::add_rms_norm_hadamard_into::<false>(
                    &mut add_had_vec,
                    &delta,
                    &weight,
                    1e-6,
                    block,
                    sign_opt,
                    &super::super::RowParallelism::default(),
                    &mut fused_norm,
                    &mut fused_rot,
                )?;
                for (a, b) in add_had_vec.iter().zip(&ref_added) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
                for (a, b) in fused_norm.iter().zip(&ref_norm) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
                for (a, b) in fused_rot.iter().zip(&ref_rot) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }

                super::super::rms_norm_pinned_into(&ref_added, &weight, 1e-6, &mut ref_norm)?;
                for (idx, out_b) in ref_rot.chunks_exact_mut(block).enumerate() {
                    let off = idx * block;
                    super::super::hadamard_into(
                        &ref_norm[off..off + block],
                        sign_opt.map(|s| &s[off..off + block]),
                        out_b,
                    )?;
                }
                let mut add_pinned_vec = input.clone();
                super::super::add_rms_norm_pinned_into(
                    &mut add_pinned_vec,
                    &delta,
                    &weight,
                    1e-6,
                    &mut fused_norm,
                )?;
                for (a, b) in add_pinned_vec.iter().zip(&ref_added) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
                for (a, b) in fused_norm.iter().zip(&ref_norm) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
                let mut add_pinned_had_vec = input.clone();
                super::super::add_rms_norm_hadamard_into::<true>(
                    &mut add_pinned_had_vec,
                    &delta,
                    &weight,
                    1e-6,
                    block,
                    sign_opt,
                    &super::super::RowParallelism::default(),
                    &mut fused_norm,
                    &mut fused_rot,
                )?;
                for (a, b) in add_pinned_had_vec.iter().zip(&ref_added) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
                for (a, b) in fused_norm.iter().zip(&ref_norm) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
                for (a, b) in fused_rot.iter().zip(&ref_rot) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }

                let mut up_pinned = weight[..block].to_vec();
                super::super::pinned_activations::swiglu_in_place(&input[..block], &mut up_pinned)?;
                let mut expected_swiglu_rot = vec![0.0_f32; block];
                super::super::hadamard_into(
                    &up_pinned,
                    sign_opt.map(|s| &s[..block]),
                    &mut expected_swiglu_rot,
                )?;
                let mut actual_swiglu_rot = vec![0.0_f32; block];
                super::super::swiglu_hadamard_block_read_into::<true>(
                    &input[..block],
                    &weight[..block],
                    sign_opt.map(|s| &s[..block]),
                    &mut actual_swiglu_rot,
                )?;
                for (a, b) in actual_swiglu_rot.iter().zip(&expected_swiglu_rot) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }

                for len in [block, block + 3] {
                    let gate_vec: Vec<f32> = (0..len)
                        .map(|i| ((i * 53 + 11) % 64) as f32 * 0.0625 - 2.0)
                        .collect();
                    let val_vec: Vec<f32> = (0..len)
                        .map(|i| ((i * 37 + 19) % 64) as f32 * 0.03125 - 0.5)
                        .collect();
                    let w_vec: Vec<f32> = (0..len)
                        .map(|i| ((i * 29 + 7) % 64) as f32 * 0.015_625 + 0.25)
                        .collect();
                    let mut exp_pinned = val_vec.clone();
                    super::super::rms_norm_pinned_in_place(&mut exp_pinned, &w_vec, 1e-6)?;
                    super::super::pinned_activations::swiglu_in_place(&gate_vec, &mut exp_pinned)?;
                    let mut act_pinned = val_vec.clone();
                    super::super::rms_norm_swiglu_in_place::<true>(
                        &mut act_pinned,
                        &w_vec,
                        1e-6,
                        &gate_vec,
                    )?;
                    let mut into_pinned = vec![0.0_f32; len];
                    super::super::rms_norm_swiglu_into::<true>(
                        &val_vec,
                        &w_vec,
                        1e-6,
                        &gate_vec,
                        &mut into_pinned,
                    )?;
                    for idx in 0..len {
                        assert_eq!(act_pinned[idx].to_bits(), exp_pinned[idx].to_bits());
                        assert_eq!(into_pinned[idx].to_bits(), exp_pinned[idx].to_bits());
                    }
                }
            }
        }
        let ssm_head = 128;
        let ssm_keys = 2;
        let repetitions = 3;
        let total = ssm_keys * repetitions * ssm_head;
        let block = 256;
        let heads_per_block = block / ssm_head;
        let input: Vec<f32> = (0..total)
            .map(|i| ((i * 43 + 9) % 128) as f32 * 0.015_625 - 1.0)
            .collect();
        let signs: Vec<f32> = (0..total)
            .map(|i| if (i * 17 + 3) % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        for sign_opt in [None, Some(signs.as_slice())] {
            for block_idx in 0..(total / block) {
                let mut exp_block = vec![0.0_f32; block];
                for h in 0..heads_per_block {
                    let target_head = block_idx * heads_per_block + h;
                    let key = target_head / repetitions;
                    let repeat = target_head % repetitions;
                    let source = (repeat * ssm_keys + key) * ssm_head;
                    let target = target_head * ssm_head;
                    let out_head = &mut exp_block[h * ssm_head..(h + 1) * ssm_head];
                    if let Some(s) = sign_opt {
                        super::super::mul_slice_into(
                            &input[source..source + ssm_head],
                            &s[target..target + ssm_head],
                            out_head,
                        );
                    } else {
                        out_head.copy_from_slice(&input[source..source + ssm_head]);
                    }
                }
                super::super::hadamard(&mut exp_block)?;
                let mut act_block = vec![0.0_f32; block];
                super::super::hadamard_grouped_block_into(
                    &input,
                    sign_opt,
                    block_idx,
                    heads_per_block,
                    repetitions,
                    ssm_keys,
                    ssm_head,
                    &mut act_block,
                )?;
                for (a, b) in act_block.iter().zip(&exp_block) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
            }
        }
        Ok(())
    }

    #[test]
    #[allow(
        clippy::cast_precision_loss,
        clippy::as_conversions,
        clippy::suboptimal_flops,
        reason = "Deterministic synthetic test values and Source RoPE reference."
    )]
    fn fused_rms_norm_rope_round_f16_matches_unfused_zero_ulp() -> TestResult {
        let head = 256;
        let rotary = 64;
        let input: Vec<f32> = (0..head)
            .map(|i| ((i * 59 + 13) % 97) as f32 * 0.03125 - 1.5)
            .collect();
        let weight: Vec<f32> = (0..head)
            .map(|i| ((i * 31 + 7) % 67) as f32 * 0.015_625 + 0.25)
            .collect();
        let frequencies: Vec<(f32, f32)> = (0..rotary / 2)
            .map(|i| {
                let angle = (i as f32) * 0.11;
                (angle.cos(), angle.sin())
            })
            .collect();
        let eps = 1e-6_f32;

        // PinnedSwift (PINNED = true)
        let mut exp_pinned = input.clone();
        super::super::rms_norm_pinned_in_place(&mut exp_pinned, &weight, eps)?;
        super::super::pinned_attention::rope_in_place(&mut exp_pinned, head, rotary, &frequencies)?;
        assert!(super::super::round_f16_slice(&mut exp_pinned));
        let mut act_pinned_in_place = input.clone();
        assert!(super::super::rms_norm_rope_round_f16_in_place::<true>(
            &mut act_pinned_in_place,
            &weight,
            eps,
            head,
            rotary,
            &frequencies,
        )?);
        let mut act_pinned_into = vec![0.0_f32; head];
        assert!(super::super::rms_norm_rope_round_f16_into::<true>(
            &input,
            &weight,
            eps,
            head,
            rotary,
            &frequencies,
            &mut act_pinned_into,
        )?);
        for i in 0..head {
            assert_eq!(act_pinned_in_place[i].to_bits(), exp_pinned[i].to_bits());
            assert_eq!(act_pinned_into[i].to_bits(), exp_pinned[i].to_bits());
        }

        // Source (PINNED = false)
        let mut exp_source = input.clone();
        super::super::rms_norm_in_place(&mut exp_source, &weight, eps)?;
        for (idx, &(c, s)) in frequencies.iter().enumerate() {
            let first = exp_source[idx];
            let second = exp_source[idx + rotary / 2];
            exp_source[idx] = first * c - second * s;
            exp_source[idx + rotary / 2] = first * s + second * c;
        }
        assert!(super::super::round_f16_slice(&mut exp_source));
        let mut act_source_in_place = input.clone();
        assert!(super::super::rms_norm_rope_round_f16_in_place::<false>(
            &mut act_source_in_place,
            &weight,
            eps,
            head,
            rotary,
            &frequencies,
        )?);
        let mut act_source_into = vec![0.0_f32; head];
        assert!(super::super::rms_norm_rope_round_f16_into::<false>(
            &input,
            &weight,
            eps,
            head,
            rotary,
            &frequencies,
            &mut act_source_into,
        )?);
        for i in 0..head {
            assert_eq!(act_source_in_place[i].to_bits(), exp_source[i].to_bits());
            assert_eq!(act_source_into[i].to_bits(), exp_source[i].to_bits());
        }
        Ok(())
    }
}
