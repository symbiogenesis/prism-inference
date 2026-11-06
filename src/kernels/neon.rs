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
fn dot16(weights: i8x16<Neon>, activations: i8x16<Neon>) -> i32x4<Neon> {
    let (weight_low, weight_high) = weights.widen();
    let (activation_low, activation_high) = activations.widen();
    // Signed widening preserves -128. Every i16 product lies in -256..=254;
    // widening each product before addition prevents narrow accumulation.
    let (low_low, low_high) = (weight_low * activation_low).widen();
    let (high_low, high_high) = (weight_high * activation_high).widen();
    low_low + low_high + high_low + high_high
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

#[inline]
fn pq2_pair_sums_prewidened(
    weights: i8x16<Neon>,
    activation_low: i16x8<Neon>,
    activation_high: i16x8<Neon>,
) -> i16x8<Neon> {
    let (weight_low, weight_high) = weights.widen();
    weight_low * activation_low + weight_high * activation_high
}

#[inline]
pub(super) fn prepare_pq2_activations(source: &[i8; 128]) -> ([i8; 128], [i16; 128]) {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        let values = std::array::from_fn(|index| {
            source[(index / 64) * 64 + (index % 16) * 4 + (index / 16) % 4]
        });
        return (values, values.map(i16::from));
    };
    let mut output = [0_i8; 128];
    let mut widened = [0_i16; 128];
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
            low.store_slice(&mut pair[0]);
            high.store_slice(&mut pair[1]);
        }
    }
    (output, widened)
}

#[inline]
fn f16_u32x4_to_f32_x4(neon: Neon, raw: u32x4<Neon>) -> f32x4<Neon> {
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
fn f16_bits_to_f32_x4(neon: Neon, bits: [u16; 4]) -> f32x4<Neon> {
    f16_u32x4_to_f32_x4(neon, u32x4::from_slice(neon, &bits.map(u32::from)))
}

#[inline]
#[allow(
    clippy::suboptimal_flops,
    reason = "Preserve ggml_compute_fp32_to_fp16 separate multiply/add rounding."
)]
fn round_f16_x4(neon: Neon, v: f32x4<Neon>) -> f32x4<Neon> {
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
    let half_raw = (((bits >> 16) & 0x8000) | magnitude) & 0xffff;
    f16_u32x4_to_f32_x4(neon, half_raw)
}

#[inline]
const fn scale_u16(block: &[u8; 34]) -> u16 {
    u16::from_le_bytes([block[0], block[1]])
}

#[inline]
pub(super) fn dot_consumption_order_prewidened(packed: &[u8; 32], widened: &[i16; 128]) -> i32 {
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
    let mut even = i16x8::splat(neon, 0);
    let mut odd = i16x8::splat(neon, 0);
    for (bytes, values) in packed
        .as_chunks::<16>()
        .0
        .iter()
        .zip(widened.as_chunks::<64>().0)
    {
        let w = u8x16::from_slice(neon, bytes);
        let planes = values.as_chunks::<8>().0;
        even += pq2_pair_sums_prewidened(
            (w & 3).bitcast::<i8x16<Neon>>() - 1,
            i16x8::from_slice(neon, &planes[0]),
            i16x8::from_slice(neon, &planes[1]),
        );
        odd += pq2_pair_sums_prewidened(
            ((w >> 2) & 3).bitcast::<i8x16<Neon>>() - 1,
            i16x8::from_slice(neon, &planes[2]),
            i16x8::from_slice(neon, &planes[3]),
        );
        even += pq2_pair_sums_prewidened(
            ((w >> 4) & 3).bitcast::<i8x16<Neon>>() - 1,
            i16x8::from_slice(neon, &planes[4]),
            i16x8::from_slice(neon, &planes[5]),
        );
        odd += pq2_pair_sums_prewidened(
            (w >> 6).bitcast::<i8x16<Neon>>() - 1,
            i16x8::from_slice(neon, &planes[6]),
            i16x8::from_slice(neon, &planes[7]),
        );
    }
    pq2_reduce(even + odd)
}

#[inline]
pub(super) fn dot_four_rows_consumption<const FUSED: bool>(
    rows: [&[u8]; 4],
    input: &[super::packed_pq2::Activation],
) -> Option<Result<[f32; 4]>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let c0 = rows[0].as_chunks::<34>().0;
    let c1 = rows[1].as_chunks::<34>().0;
    let c2 = rows[2].as_chunks::<34>().0;
    let c3 = rows[3].as_chunks::<34>().0;
    let mut outputs = f32x4::splat(neon, 0.0);
    for ((((b0, b1), b2), b3), activation) in c0.iter().zip(c1).zip(c2).zip(c3).zip(input) {
        let row_scales = f16_bits_to_f32_x4(
            neon,
            [scale_u16(b0), scale_u16(b1), scale_u16(b2), scale_u16(b3)],
        );
        if !row_scales.is_finite().all_true() {
            return Some(Err(crate::invalid("nonfinite tensor or activation value")));
        }
        let ds = row_scales * f32x4::splat(neon, activation.scale);
        let blocks = [b0, b1, b2, b3];
        let mut even = [i16x8::splat(neon, 0); 4];
        let mut odd = [i16x8::splat(neon, 0); 4];
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
                even[row] +=
                    pq2_pair_sums_prewidened((w & 3).bitcast::<i8x16<Neon>>() - 1, p0_low, p0_high);
                odd[row] += pq2_pair_sums_prewidened(
                    ((w >> 2) & 3).bitcast::<i8x16<Neon>>() - 1,
                    p1_low,
                    p1_high,
                );
                even[row] += pq2_pair_sums_prewidened(
                    ((w >> 4) & 3).bitcast::<i8x16<Neon>>() - 1,
                    p2_low,
                    p2_high,
                );
                odd[row] += pq2_pair_sums_prewidened(
                    (w >> 6).bitcast::<i8x16<Neon>>() - 1,
                    p3_low,
                    p3_high,
                );
            }
        }
        let sums = i32x4::from_slice(
            neon,
            &std::array::from_fn::<i32, 4, _>(|row| pq2_reduce(even[row] + odd[row])),
        )
        .to_float::<f32x4<Neon>>();
        outputs = if FUSED {
            ds.mul_add(sums, outputs)
        } else {
            outputs + ds * sums
        };
    }
    if !outputs.is_finite().all_true() {
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
    let chunks: [&[[u8; 34]]; 8] = rows.map(|row| row.as_chunks::<34>().0);
    let mut out_lo = f32x4::splat(neon, 0.0);
    let mut out_hi = f32x4::splat(neon, 0.0);
    for (block_idx, activation) in input.iter().enumerate() {
        let blocks: [&[u8; 34]; 8] = std::array::from_fn(|row| &chunks[row][block_idx]);
        let scales_lo = f16_bits_to_f32_x4(
            neon,
            [
                scale_u16(blocks[0]),
                scale_u16(blocks[1]),
                scale_u16(blocks[2]),
                scale_u16(blocks[3]),
            ],
        );
        let scales_hi = f16_bits_to_f32_x4(
            neon,
            [
                scale_u16(blocks[4]),
                scale_u16(blocks[5]),
                scale_u16(blocks[6]),
                scale_u16(blocks[7]),
            ],
        );
        if !(scales_lo.is_finite() & scales_hi.is_finite()).all_true() {
            return Some(Err(crate::invalid("nonfinite tensor or activation value")));
        }
        let act_scale = f32x4::splat(neon, activation.scale);
        let ds_lo = scales_lo * act_scale;
        let ds_hi = scales_hi * act_scale;
        let mut even = [i16x8::splat(neon, 0); 8];
        let mut odd = [i16x8::splat(neon, 0); 8];
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
                even[row] +=
                    pq2_pair_sums_prewidened((w & 3).bitcast::<i8x16<Neon>>() - 1, p0_low, p0_high);
                odd[row] += pq2_pair_sums_prewidened(
                    ((w >> 2) & 3).bitcast::<i8x16<Neon>>() - 1,
                    p1_low,
                    p1_high,
                );
                even[row] += pq2_pair_sums_prewidened(
                    ((w >> 4) & 3).bitcast::<i8x16<Neon>>() - 1,
                    p2_low,
                    p2_high,
                );
                odd[row] += pq2_pair_sums_prewidened(
                    (w >> 6).bitcast::<i8x16<Neon>>() - 1,
                    p3_low,
                    p3_high,
                );
            }
        }
        let sums_lo = i32x4::from_slice(
            neon,
            &std::array::from_fn::<i32, 4, _>(|row| pq2_reduce(even[row] + odd[row])),
        )
        .to_float::<f32x4<Neon>>();
        let sums_hi = i32x4::from_slice(
            neon,
            &std::array::from_fn::<i32, 4, _>(|row| pq2_reduce(even[row + 4] + odd[row + 4])),
        )
        .to_float::<f32x4<Neon>>();
        if FUSED {
            out_lo = ds_lo.mul_add(sums_lo, out_lo);
            out_hi = ds_hi.mul_add(sums_hi, out_hi);
        } else {
            out_lo += ds_lo * sums_lo;
            out_hi += ds_hi * sums_hi;
        }
    }
    if !(out_lo.is_finite() & out_hi.is_finite()).all_true() {
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
    for group in chunk.as_chunks::<16>().0 {
        let v0 = f32x4::from_slice(neon, &group[0..4]);
        let v1 = f32x4::from_slice(neon, &group[4..8]);
        let v2 = f32x4::from_slice(neon, &group[8..12]);
        let v3 = f32x4::from_slice(neon, &group[12..16]);
        if !(v0.is_finite() & v1.is_finite() & v2.is_finite() & v3.is_finite()).all_true() {
            return Err(crate::invalid("nonfinite tensor or activation value"));
        }
        max_abs_vec = max_abs_vec
            .max(v0.abs())
            .max(v1.abs())
            .max(v2.abs())
            .max(v3.abs());
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
        if !(r0.is_finite() & r1.is_finite() & r2.is_finite() & r3.is_finite()).all_true() {
            return None;
        }
        let i0 = ((r0.bitcast::<u32x4<Neon>>() & mask_mant).bitcast::<i32x4<Neon>>() - bias)
            .min(max_code);
        let i1 = ((r1.bitcast::<u32x4<Neon>>() & mask_mant).bitcast::<i32x4<Neon>>() - bias)
            .min(max_code);
        let i2 = ((r2.bitcast::<u32x4<Neon>>() & mask_mant).bitcast::<i32x4<Neon>>() - bias)
            .min(max_code);
        let i3 = ((r3.bitcast::<u32x4<Neon>>() & mask_mant).bitcast::<i32x4<Neon>>() - bias)
            .min(max_code);
        if !(i0.simd_ge(min_code)
            & i1.simd_ge(min_code)
            & i2.simd_ge(min_code)
            & i3.simd_ge(min_code))
        .all_true()
        {
            return None;
        }
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
    Some(Ok(block))
}

#[inline]
pub(super) fn q8_pinned_block<const NATIVE_SCALE: bool>(
    chunk: &[f32; 32],
) -> Option<Result<super::Q8Block>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let mut max_abs_vec = f32x4::splat(neon, 0.0);
    for group in chunk.as_chunks::<16>().0 {
        let v0 = f32x4::from_slice(neon, &group[0..4]);
        let v1 = f32x4::from_slice(neon, &group[4..8]);
        let v2 = f32x4::from_slice(neon, &group[8..12]);
        let v3 = f32x4::from_slice(neon, &group[12..16]);
        if !(v0.is_finite() & v1.is_finite() & v2.is_finite() & v3.is_finite()).all_true() {
            return Some(Err(crate::invalid("nonfinite tensor or activation value")));
        }
        max_abs_vec = max_abs_vec
            .max(v0.abs())
            .max(v1.abs())
            .max(v2.abs())
            .max(v3.abs());
    }
    let mut max_lanes = [0.0_f32; 4];
    max_abs_vec.store_slice(&mut max_lanes);
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
        if !(r0.is_finite() & r1.is_finite() & r2.is_finite() & r3.is_finite()).all_true() {
            return None;
        }
        let i0 = (r0.bitcast::<u32x4<Neon>>() & mask_mant).bitcast::<i32x4<Neon>>() - bias;
        let i1 = (r1.bitcast::<u32x4<Neon>>() & mask_mant).bitcast::<i32x4<Neon>>() - bias;
        let i2 = (r2.bitcast::<u32x4<Neon>>() & mask_mant).bitcast::<i32x4<Neon>>() - bias;
        let i3 = (r3.bitcast::<u32x4<Neon>>() & mask_mant).bitcast::<i32x4<Neon>>() - bias;
        if !(i0.simd_ge(min_code)
            & i0.simd_le(max_code)
            & i1.simd_ge(min_code)
            & i1.simd_le(max_code)
            & i2.simd_ge(min_code)
            & i2.simd_le(max_code)
            & i3.simd_ge(min_code)
            & i3.simd_le(max_code))
        .all_true()
        {
            return None;
        }
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
pub(super) fn round_f16_slice(values: &mut [f32]) -> bool {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return false;
    };
    let (chunks, tail) = values.as_chunks_mut::<4>();
    for chunk in chunks {
        round_f16_x4(neon, f32x4::from_slice(neon, chunk)).store_slice(chunk);
    }
    for value in tail {
        *value = super::round_f16(*value);
    }
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
    let mut sq_lanes = [0.0_f32; 4];
    for chunk in chunks {
        let v = f32x4::from_slice(neon, chunk);
        (v * v).store_slice(&mut sq_lanes);
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
    let (in_chunks, in_tail) = input.as_chunks::<4>();
    let (w_chunks, w_tail) = weight.as_chunks::<4>();
    let mut sq_lanes = [0.0_f32; 4];
    for (in_chunk, w_chunk) in in_chunks.iter().zip(w_chunks) {
        let v = f32x4::from_slice(neon, in_chunk);
        let w = f32x4::from_slice(neon, w_chunk);
        if !(v.is_finite() & w.is_finite()).all_true() {
            return Some(Err(crate::invalid("nonfinite tensor or activation value")));
        }
        (v * v).store_slice(&mut sq_lanes);
        sum += f64::from(sq_lanes[0]);
        sum += f64::from(sq_lanes[1]);
        sum += f64::from(sq_lanes[2]);
        sum += f64::from(sq_lanes[3]);
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
    if let Err(err) = output.try_reserve_exact(input.len()) {
        return Some(Err(err.into()));
    }
    output.resize(input.len(), 0.0);
    let scale_vec = f32x4::splat(neon, scale);
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
        if !out.is_finite().all_true() {
            return Some(Err(crate::invalid("nonfinite tensor or activation value")));
        }
        out.store_slice(out_chunk);
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
        if !out.is_finite().all_true() {
            return Some(Err(crate::invalid("nonfinite tensor or activation value")));
        }
        out.store_slice(val_chunk);
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
    let mut even = i16x8::splat(neon, 0);
    let mut odd = i16x8::splat(neon, 0);
    for (bytes, values) in packed
        .as_chunks::<16>()
        .0
        .iter()
        .zip(activations.as_chunks::<64>().0)
    {
        let packed = u8x16::from_slice(neon, bytes);
        let planes = values.as_chunks::<16>().0;
        even += pq2_pair_sums(
            (packed & 3).bitcast::<i8x16<Neon>>() - 1,
            i8x16::from_slice(neon, &planes[0]),
        );
        odd += pq2_pair_sums(
            ((packed >> 2) & 3).bitcast::<i8x16<Neon>>() - 1,
            i8x16::from_slice(neon, &planes[1]),
        );
        even += pq2_pair_sums(
            ((packed >> 4) & 3).bitcast::<i8x16<Neon>>() - 1,
            i8x16::from_slice(neon, &planes[2]),
        );
        odd += pq2_pair_sums(
            (packed >> 6).bitcast::<i8x16<Neon>>() - 1,
            i8x16::from_slice(neon, &planes[3]),
        );
    }
    pq2_reduce(even + odd)
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
    let mut even = [i16x8::splat(neon, 0); 4];
    let mut odd = [i16x8::splat(neon, 0); 4];
    for (half, values) in activations.as_chunks::<64>().0.iter().enumerate() {
        let planes = values.as_chunks::<16>().0;
        let (p0_low, p0_high) = i8x16::from_slice(neon, &planes[0]).widen();
        let (p1_low, p1_high) = i8x16::from_slice(neon, &planes[1]).widen();
        let (p2_low, p2_high) = i8x16::from_slice(neon, &planes[2]).widen();
        let (p3_low, p3_high) = i8x16::from_slice(neon, &planes[3]).widen();
        for row in 0..4 {
            let bytes = &packed[row].as_chunks::<16>().0[half];
            let w = u8x16::from_slice(neon, bytes);
            even[row] +=
                pq2_pair_sums_prewidened((w & 3).bitcast::<i8x16<Neon>>() - 1, p0_low, p0_high);
            odd[row] += pq2_pair_sums_prewidened(
                ((w >> 2) & 3).bitcast::<i8x16<Neon>>() - 1,
                p1_low,
                p1_high,
            );
            even[row] += pq2_pair_sums_prewidened(
                ((w >> 4) & 3).bitcast::<i8x16<Neon>>() - 1,
                p2_low,
                p2_high,
            );
            odd[row] +=
                pq2_pair_sums_prewidened((w >> 6).bitcast::<i8x16<Neon>>() - 1, p3_low, p3_high);
        }
    }
    std::array::from_fn(|row| pq2_reduce(even[row] + odd[row]))
}

#[inline]
fn dot32(neon: Neon, low: i8x16<Neon>, high: i8x16<Neon>, values: &[i8; 32]) -> i32 {
    (dot16(low, i8x16::from_slice(neon, &values[..16]))
        + dot16(high, i8x16::from_slice(neon, &values[16..])))
    .reduce_sum()
}

#[inline]
fn dot16_native_lanes(weights: i8x16<Neon>, values: i8x16<Neon>) -> [i32; 4] {
    let (weight_low, weight_high) = weights.widen();
    let (value_low, value_high) = values.widen();
    let (a, b) = (weight_low * value_low).widen();
    let (c, d) = (weight_high * value_high).widen();
    [
        a.reduce_sum(),
        b.reduce_sum(),
        c.reduce_sum(),
        d.reduce_sum(),
    ]
}

#[inline]
fn dot32_native_lanes(
    neon: Neon,
    low: i8x16<Neon>,
    high: i8x16<Neon>,
    values: &[i8; 32],
) -> [i32; 4] {
    let low = dot16_native_lanes(low, i8x16::from_slice(neon, &values[..16]));
    let high = dot16_native_lanes(high, i8x16::from_slice(neon, &values[16..]));
    std::array::from_fn(|lane| low[lane] + high[lane])
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
fn trits(packed: u8x16<Neon>) -> [i8x16<Neon>; 5] {
    let (mut low, mut high) = packed.widen();
    std::array::from_fn(|_| {
        // The native representation is a base-three fraction with denominator
        // 256. Each product is at most 765; no u16 operation can wrap.
        low *= 3;
        high *= 3;
        let digit = (low >> 8).narrow(high >> 8).bitcast::<i8x16<Neon>>() - 1;
        low &= 255;
        high &= 255;
        digit
    })
}

#[inline]
fn join_low8(low: i8x16<Neon>, high: i8x16<Neon>) -> i8x16<Neon> {
    low.bitcast::<u64x2<Neon>>()
        .zip_low(high.bitcast::<u64x2<Neon>>())
        .bitcast()
}

#[inline]
fn decode_ptq1(neon: Neon, packed: &[u8; 28]) -> [i8x16<Neon>; 8] {
    let first = trits(u8x16::from_slice(neon, &packed[..16]));
    // Overlap only bytes 10..26; no load reaches the scale or exceeds the block.
    let rest = u8x16::from_slice(neon, &packed[10..26]);
    let middle = trits(rest.swizzle_dyn([6, 7, 8, 9, 10, 11, 12, 13, 6, 7, 8, 9, 10, 11, 12, 13]));
    let last = trits(rest.swizzle_dyn([
        14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
    ]));
    let tail_low: u32x4<Neon> = last[0]
        .bitcast::<u16x8<Neon>>()
        .zip_low(last[1].bitcast::<u16x8<Neon>>())
        .bitcast();
    let tail_high: u32x4<Neon> = last[2]
        .bitcast::<u16x8<Neon>>()
        .zip_low(last[3].bitcast::<u16x8<Neon>>())
        .bitcast();
    let tail = tail_low.zip_low(tail_high).bitcast();
    [
        first[0],
        first[1],
        first[2],
        first[3],
        first[4],
        join_low8(middle[0], middle[1]),
        join_low8(middle[2], middle[3]),
        join_low8(middle[4], tail),
    ]
}

/// Keeps each native PTQ1 DOTPROD lane separate until floating accumulation.
#[inline]
pub(super) fn dot_ptq1_native_lanes(
    packed: &[u8; 28],
    activations: [&[i8; 32]; 4],
) -> Result<[[i32; 4]; 4]> {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        let weights = super::ptq1_values(packed)?;
        return Ok(std::array::from_fn(|group| {
            super::q8_native_lanes(&weights[group * 32..group * 32 + 32], activations[group])
        }));
    };
    let decoded = decode_ptq1(neon, packed);
    Ok(std::array::from_fn(|group| {
        dot32_native_lanes(
            neon,
            decoded[group * 2],
            decoded[group * 2 + 1],
            activations[group],
        )
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
    let decoded = decode_ptq1(neon, packed);
    Ok(std::array::from_fn(|group| {
        dot32(
            neon,
            decoded[group * 2],
            decoded[group * 2 + 1],
            activations[group],
        )
    }))
}

/// Evaluates the normalized Sylvester Hadamard butterfly stages in NEON `f32x4`.
///
/// Requires a power-of-two slice of at least four elements. Stages `half = 1`
/// and `half = 2` are fused per four-element vector, and all subsequent stages
/// operate on four-lane vectors while preserving exact IEEE 754 addition and
/// subtraction order.
#[inline]
pub(super) fn hadamard(values: &mut [f32], normalization: f32) -> Option<Result<()>> {
    if values.len() < 4 {
        return None;
    }
    let neon = fearless_simd::Level::baseline().as_neon()?;
    for chunk in values.as_chunks::<4>().0 {
        if !f32x4::from_slice(neon, chunk).is_finite().all_true() {
            return Some(Err(crate::invalid(
                "Hadamard requires a finite power-of-two block",
            )));
        }
    }
    for chunk in values.as_chunks_mut::<4>().0 {
        let [a0, a1, a2, a3] = *chunk;
        let b0 = a0 + a1;
        let b1 = a0 - a1;
        let b2 = a2 + a3;
        let b3 = a2 - a3;
        chunk[0] = b0 + b2;
        chunk[1] = b1 + b3;
        chunk[2] = b0 - b2;
        chunk[3] = b1 - b3;
    }
    let mut half = 4;
    while half < values.len() {
        for block in values.chunks_exact_mut(half * 2) {
            let (left_half, right_half) = block.split_at_mut(half);
            for (left, right) in left_half
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(right_half.as_chunks_mut::<4>().0)
            {
                let l = f32x4::from_slice(neon, left);
                let r = f32x4::from_slice(neon, right);
                (l + r).store_slice(left);
                (l - r).store_slice(right);
            }
        }
        half *= 2;
    }
    let norm = f32x4::splat(neon, normalization);
    for chunk in values.as_chunks_mut::<4>().0 {
        let v = f32x4::from_slice(neon, chunk) * norm;
        if !v.is_finite().all_true() {
            return Some(Err(crate::invalid("nonfinite kernel output")));
        }
        v.store_slice(chunk);
    }
    Some(Ok(()))
}

#[cfg(test)]
mod tests {
    use super::{dot, dot_consumption_order, dot_groups, dot_ptq1_groups};

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
}
