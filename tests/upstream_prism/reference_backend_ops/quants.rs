//! Clean-room GGML quantized block codecs and type introspection helpers.
//!
//! No `bonsai_prism_runtime`, `bonsai_gguf`, or native GGML libraries are linked or called.

use super::types::{
    TYPE_BF16, TYPE_F16, TYPE_F32, TYPE_F64, TYPE_I8, TYPE_I16, TYPE_I32, TYPE_I64, TYPE_IQ1_M,
    TYPE_IQ1_S, TYPE_IQ2_S, TYPE_IQ2_XS, TYPE_IQ2_XXS, TYPE_IQ4_NL, TYPE_IQ4_XS, TYPE_MXFP4,
    TYPE_Q1_0, TYPE_Q2_0, TYPE_Q2_K, TYPE_Q3_K, TYPE_Q4_0, TYPE_Q4_1, TYPE_Q4_K, TYPE_Q5_0,
    TYPE_Q5_1, TYPE_Q5_K, TYPE_Q6_K, TYPE_Q8_0, TYPE_Q8_1, TYPE_Q8_K, TYPE_TQ1_0, TYPE_TQ2_0,
    blck_size, f16_to_f32, f32_to_bf16, f32_to_f16, f32_to_i64_trunc, is_quantized,
    read_unquant_elem_f32, type_size,
};
use crate::{Result, invalid};

const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

const KVALUES_E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

pub(super) const fn type_info(kind: i32) -> (bool, bool, bool) {
    let quantized = is_quantized(kind);
    let requires_importance = matches!(
        kind,
        TYPE_IQ1_S | TYPE_IQ1_M | TYPE_IQ2_XXS | TYPE_IQ2_XS | TYPE_IQ2_S
    );
    let has_to_float = matches!(
        kind,
        TYPE_F32
            | TYPE_F16
            | TYPE_BF16
            | TYPE_F64
            | TYPE_I8
            | TYPE_I16
            | TYPE_I32
            | TYPE_I64
            | TYPE_Q4_0
            | TYPE_Q4_1
            | TYPE_Q5_0
            | TYPE_Q5_1
            | TYPE_Q8_0
            | TYPE_Q8_1
            | TYPE_Q2_K
            | TYPE_Q3_K
            | TYPE_Q4_K
            | TYPE_Q5_K
            | TYPE_Q6_K
            | TYPE_Q8_K
            | TYPE_IQ4_NL
            | TYPE_IQ4_XS
            | TYPE_TQ1_0
            | TYPE_TQ2_0
            | TYPE_MXFP4
            | TYPE_Q1_0
            | TYPE_Q2_0
    );
    (quantized, requires_importance, has_to_float)
}

fn read_f16_le(slice: &[u8], offset: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([slice[offset], slice[offset + 1]]))
}

fn read_u32_le(slice: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        slice[offset],
        slice[offset + 1],
        slice[offset + 2],
        slice[offset + 3],
    ])
}

fn read_i8(byte: u8) -> i8 {
    i8::from_le_bytes([byte])
}

fn e8m0_to_fp32_half(val: u8) -> f32 {
    if val < 2 {
        return f32::from_bits(0x0020_0000 << u32::from(val));
    }
    let exp_bits = u32::from(val - 1) << 23;
    f32::from_bits(exp_bits)
}

fn i8_clamp(val: f32) -> i8 {
    let clamped = f32_to_i64_trunc(val.round()).clamp(-128, 127);
    i8::try_from(clamped).unwrap_or(0)
}

fn u8_clamp(val: f32, hi: i64) -> u8 {
    let clamped = f32_to_i64_trunc(val.round()).clamp(0, hi);
    u8::try_from(clamped).unwrap_or(0)
}

fn get_scale_min_k4(idx: usize, q: &[u8]) -> (u8, u8) {
    if idx < 4 {
        (q[idx] & 63, q[idx + 4] & 63)
    } else {
        (
            (q[idx + 4] & 0x0f) | ((q[idx - 4] >> 6) << 4),
            (q[idx + 4] >> 4) | ((q[idx] >> 6) << 4),
        )
    }
}

fn dequant_q4_0(block: &[u8], out: &mut [f32]) {
    let d = read_f16_le(block, 0);
    for idx in 0..16 {
        let packed = block[2 + idx];
        let low = i8::try_from(i32::from(packed & 0x0f) - 8).unwrap_or(0);
        let high = i8::try_from(i32::from(packed >> 4) - 8).unwrap_or(0);
        out[idx] = f32::from(low) * d;
        out[idx + 16] = f32::from(high) * d;
    }
}

fn dequant_q4_1(block: &[u8], out: &mut [f32]) {
    let d = read_f16_le(block, 0);
    let m = read_f16_le(block, 2);
    for idx in 0..16 {
        let packed = block[4 + idx];
        out[idx] = f32::from(packed & 0x0f).mul_add(d, m);
        out[idx + 16] = f32::from(packed >> 4).mul_add(d, m);
    }
}

fn dequant_q5_0(block: &[u8], out: &mut [f32]) {
    let d = read_f16_le(block, 0);
    let qh = read_u32_le(block, 2);
    for idx in 0..16 {
        let packed = block[6 + idx];
        let xh_low = u8::try_from(((qh >> idx) << 4) & 0x10).unwrap_or(0);
        let xh_high = u8::try_from((qh >> (idx + 12)) & 0x10).unwrap_or(0);
        let q_low = i8::try_from(i32::from((packed & 0x0f) | xh_low) - 16).unwrap_or(0);
        let q_high = i8::try_from(i32::from((packed >> 4) | xh_high) - 16).unwrap_or(0);
        out[idx] = f32::from(q_low) * d;
        out[idx + 16] = f32::from(q_high) * d;
    }
}

fn dequant_q5_1(block: &[u8], out: &mut [f32]) {
    let d = read_f16_le(block, 0);
    let m = read_f16_le(block, 2);
    let qh = read_u32_le(block, 4);
    for idx in 0..16 {
        let packed = block[8 + idx];
        let xh_low = u8::try_from(((qh >> idx) << 4) & 0x10).unwrap_or(0);
        let xh_high = u8::try_from((qh >> (idx + 12)) & 0x10).unwrap_or(0);
        out[idx] = f32::from((packed & 0x0f) | xh_low).mul_add(d, m);
        out[idx + 16] = f32::from((packed >> 4) | xh_high).mul_add(d, m);
    }
}

fn dequant_q8(block: &[u8], qs_offset: usize, out: &mut [f32]) {
    let d = read_f16_le(block, 0);
    for (idx, dst) in out[..32].iter_mut().enumerate() {
        *dst = f32::from(read_i8(block[qs_offset + idx])) * d;
    }
}

fn dequant_iq4_nl(block: &[u8], out: &mut [f32]) {
    let d = read_f16_le(block, 0);
    for idx in 0..16 {
        let packed = block[2 + idx];
        out[idx] = d * f32::from(KVALUES_IQ4NL[usize::from(packed & 0x0f)]);
        out[idx + 16] = d * f32::from(KVALUES_IQ4NL[usize::from(packed >> 4)]);
    }
}

fn dequant_mxfp4(block: &[u8], out: &mut [f32]) {
    let d = e8m0_to_fp32_half(block[0]);
    for idx in 0..16 {
        let packed = block[1 + idx];
        out[idx] = KVALUES_E2M1[usize::from(packed & 0x0f)] * d;
        out[idx + 16] = KVALUES_E2M1[usize::from(packed >> 4)] * d;
    }
}

fn dequant_q4_k(block: &[u8], out: &mut [f32]) {
    let d = read_f16_le(block, 0);
    let dmin = read_f16_le(block, 2);
    let scales = &block[4..16];
    let qs = &block[16..144];
    for group in 0..4 {
        let (sc_low, min_low) = get_scale_min_k4(group * 2, scales);
        let (sc_high, min_high) = get_scale_min_k4(group * 2 + 1, scales);
        let d_low = d * f32::from(sc_low);
        let m_low = dmin * f32::from(min_low);
        let d_high = d * f32::from(sc_high);
        let m_high = dmin * f32::from(min_high);
        let q_base = group * 32;
        let out_base = group * 64;
        for idx in 0..32 {
            out[out_base + idx] = d_low.mul_add(f32::from(qs[q_base + idx] & 0x0f), -m_low);
            out[out_base + 32 + idx] = d_high.mul_add(f32::from(qs[q_base + idx] >> 4), -m_high);
        }
    }
}

fn dequant_q5_k(block: &[u8], out: &mut [f32]) {
    let d = read_f16_le(block, 0);
    let dmin = read_f16_le(block, 2);
    let scales = &block[4..16];
    let qh = &block[16..48];
    let qs = &block[48..176];
    for group in 0..4 {
        let (sc_low, min_low) = get_scale_min_k4(group * 2, scales);
        let (sc_high, min_high) = get_scale_min_k4(group * 2 + 1, scales);
        let d_low = d * f32::from(sc_low);
        let m_low = dmin * f32::from(min_low);
        let d_high = d * f32::from(sc_high);
        let m_high = dmin * f32::from(min_high);
        let mask_low = u8::try_from(1_u32 << (group * 2)).unwrap_or(0);
        let mask_high = u8::try_from(1_u32 << (group * 2 + 1)).unwrap_or(0);
        let q_base = group * 32;
        let out_base = group * 64;
        for idx in 0..32 {
            let hi_a = if qh[idx] & mask_low != 0 { 16 } else { 0 };
            let hi_b = if qh[idx] & mask_high != 0 { 16 } else { 0 };
            out[out_base + idx] =
                d_low.mul_add(f32::from((qs[q_base + idx] & 0x0f) + hi_a), -m_low);
            out[out_base + 32 + idx] =
                d_high.mul_add(f32::from((qs[q_base + idx] >> 4) + hi_b), -m_high);
        }
    }
}

fn dequant_q6_k(block: &[u8], out: &mut [f32]) {
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    let d = read_f16_le(block, 208);
    for half in 0..2 {
        let low_off = half * 64;
        let high_off = half * 32;
        let sc_off = half * 8;
        let out_off = half * 128;
        for l in 0..32 {
            let is = l / 16;
            let q1 = i8::try_from(
                i32::from((ql[low_off + l] & 0x0f) | ((qh[high_off + l] & 3) << 4)) - 32,
            )
            .unwrap_or(0);
            let q2 = i8::try_from(
                i32::from((ql[low_off + l + 32] & 0x0f) | (((qh[high_off + l] >> 2) & 3) << 4))
                    - 32,
            )
            .unwrap_or(0);
            let q3 = i8::try_from(
                i32::from((ql[low_off + l] >> 4) | (((qh[high_off + l] >> 4) & 3) << 4)) - 32,
            )
            .unwrap_or(0);
            let q4 = i8::try_from(
                i32::from((ql[low_off + l + 32] >> 4) | (((qh[high_off + l] >> 6) & 3) << 4)) - 32,
            )
            .unwrap_or(0);
            out[out_off + l] = d * f32::from(read_i8(scales[sc_off + is])) * f32::from(q1);
            out[out_off + l + 32] = d * f32::from(read_i8(scales[sc_off + is + 2])) * f32::from(q2);
            out[out_off + l + 64] = d * f32::from(read_i8(scales[sc_off + is + 4])) * f32::from(q3);
            out[out_off + l + 96] = d * f32::from(read_i8(scales[sc_off + is + 6])) * f32::from(q4);
        }
    }
}

fn dequant_q2_k(block: &[u8], out: &mut [f32]) {
    let scales = &block[0..16];
    let qs = &block[16..80];
    let d = read_f16_le(block, 80);
    let dmin = read_f16_le(block, 82);
    let mut is = 0_usize;
    for chunk in 0..2 {
        let q_chunk = &qs[chunk * 32..chunk * 32 + 32];
        let mut out_off = chunk * 128;
        let mut shift = 0_u8;
        for _ in 0..4 {
            let sc_lo = scales[is];
            is += 1;
            let dl = d * f32::from(sc_lo & 0x0f);
            let ml = dmin * f32::from(sc_lo >> 4);
            for l in 0..16 {
                out[out_off + l] = dl.mul_add(f32::from((q_chunk[l] >> shift) & 3), -ml);
            }
            let sc_hi = scales[is];
            is += 1;
            let dl2 = d * f32::from(sc_hi & 0x0f);
            let ml2 = dmin * f32::from(sc_hi >> 4);
            for l in 0..16 {
                out[out_off + 16 + l] =
                    dl2.mul_add(f32::from((q_chunk[l + 16] >> shift) & 3), -ml2);
            }
            out_off += 32;
            shift += 2;
        }
    }
}

fn dequant_q3_k(block: &[u8], out: &mut [f32]) {
    let hmask = &block[0..32];
    let qs = &block[32..96];
    let sc_bytes = &block[96..108];
    let d_all = read_f16_le(block, 108);
    let mut scales_i8 = [0_i8; 16];
    let w0 = read_u32_le(sc_bytes, 0);
    let w1 = read_u32_le(sc_bytes, 4);
    let w2 = read_u32_le(sc_bytes, 8);
    let aux = [
        (w0 & 0x0f0f_0f0f) | ((w2 & 0x0303_0303) << 4),
        (w1 & 0x0f0f_0f0f) | (((w2 >> 2) & 0x0303_0303) << 4),
        ((w0 >> 4) & 0x0f0f_0f0f) | (((w2 >> 4) & 0x0303_0303) << 4),
        ((w1 >> 4) & 0x0f0f_0f0f) | (((w2 >> 6) & 0x0303_0303) << 4),
    ];
    for (idx, word) in aux.into_iter().enumerate() {
        let bytes = word.to_le_bytes();
        for b in 0..4 {
            scales_i8[idx * 4 + b] = i8::try_from(i32::from(bytes[b]) - 32).unwrap_or(0);
        }
    }
    let mut is = 0_usize;
    let mut mask = 1_u8;
    for chunk in 0..2 {
        let q_chunk = &qs[chunk * 32..chunk * 32 + 32];
        let mut out_off = chunk * 128;
        let mut shift = 0_u8;
        for _ in 0..4 {
            let dl = d_all * f32::from(scales_i8[is]);
            is += 1;
            for l in 0..16 {
                let sub = if hmask[l] & mask != 0 { 0 } else { 4 };
                let q = i8::try_from(i32::from((q_chunk[l] >> shift) & 3) - sub).unwrap_or(0);
                out[out_off + l] = dl * f32::from(q);
            }
            let dl2 = d_all * f32::from(scales_i8[is]);
            is += 1;
            for l in 0..16 {
                let sub = if hmask[l + 16] & mask != 0 { 0 } else { 4 };
                let q = i8::try_from(i32::from((q_chunk[l + 16] >> shift) & 3) - sub).unwrap_or(0);
                out[out_off + 16 + l] = dl2 * f32::from(q);
            }
            out_off += 32;
            shift += 2;
            mask = mask.rotate_left(1);
        }
    }
}

fn dequant_q8_k(block: &[u8], out: &mut [f32]) {
    let d = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
    for (idx, dst) in out[..256].iter_mut().enumerate() {
        *dst = d * f32::from(read_i8(block[4 + idx]));
    }
}

fn dequant_tq2_0(block: &[u8], out: &mut [f32]) {
    let d = read_f16_le(block, 64);
    for half in 0..2 {
        let q = &block[half * 32..half * 32 + 32];
        let out_off = half * 128;
        for j in 0..4 {
            let shift = j * 2;
            for l in 0..32 {
                let trit = i8::try_from(i32::from((q[l] >> shift) & 3) - 1).unwrap_or(0);
                out[out_off + j * 32 + l] = f32::from(trit) * d;
            }
        }
    }
}

fn dequant_tq1_0(block: &[u8], out: &mut [f32]) {
    const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];
    let d = read_f16_le(block, 52);
    let qs = &block[0..48];
    let qh = &block[48..52];
    for (j, &pow) in POW3.iter().take(5).enumerate() {
        let p = u16::from(pow);
        for (l, dst) in out[j * 32..j * 32 + 32].iter_mut().enumerate() {
            let q = u16::from(qs[l]).wrapping_mul(p);
            let xi = i8::try_from(i32::from(u8::try_from(((q & 0xff) * 3) >> 8).unwrap_or(0)) - 1)
                .unwrap_or(0);
            *dst = f32::from(xi) * d;
        }
        for (l, dst) in out[160 + j * 16..160 + j * 16 + 16].iter_mut().enumerate() {
            let q = u16::from(qs[32 + l]).wrapping_mul(p);
            let xi = i8::try_from(i32::from(u8::try_from(((q & 0xff) * 3) >> 8).unwrap_or(0)) - 1)
                .unwrap_or(0);
            *dst = f32::from(xi) * d;
        }
    }
    for (j, &pow) in POW3.iter().take(4).enumerate() {
        let p = u16::from(pow);
        for (l, dst) in out[240 + j * 4..240 + j * 4 + 4].iter_mut().enumerate() {
            let q = u16::from(qh[l]).wrapping_mul(p);
            let xi = i8::try_from(i32::from(u8::try_from(((q & 0xff) * 3) >> 8).unwrap_or(0)) - 1)
                .unwrap_or(0);
            *dst = f32::from(xi) * d;
        }
    }
}

fn dequant_block(kind: i32, block: &[u8], out: &mut [f32]) -> Result<()> {
    match kind {
        TYPE_Q4_0 => dequant_q4_0(block, out),
        TYPE_Q4_1 => dequant_q4_1(block, out),
        TYPE_Q5_0 => dequant_q5_0(block, out),
        TYPE_Q5_1 => dequant_q5_1(block, out),
        TYPE_Q8_0 => dequant_q8(block, 2, out),
        TYPE_Q8_1 => dequant_q8(block, 4, out),
        TYPE_IQ4_NL => dequant_iq4_nl(block, out),
        TYPE_MXFP4 => dequant_mxfp4(block, out),
        TYPE_Q2_K => dequant_q2_k(block, out),
        TYPE_Q3_K => dequant_q3_k(block, out),
        TYPE_Q4_K => dequant_q4_k(block, out),
        TYPE_Q5_K => dequant_q5_k(block, out),
        TYPE_Q6_K => dequant_q6_k(block, out),
        TYPE_Q8_K => dequant_q8_k(block, out),
        TYPE_TQ1_0 => dequant_tq1_0(block, out),
        TYPE_TQ2_0 => dequant_tq2_0(block, out),
        _ => {
            return Err(invalid(format!(
                "unsupported quantized dequantize type {kind}"
            )));
        }
    }
    Ok(())
}

pub(super) fn dequantize_to_f32(kind: i32, raw: &[u8], count: usize) -> Result<Vec<f32>> {
    let block_elems = usize::try_from(blck_size(kind))?;
    let block_bytes = usize::try_from(type_size(kind))?;
    if block_elems == 0 || !count.is_multiple_of(block_elems) {
        return Err(invalid(
            "dequantize element count not aligned to block size",
        ));
    }
    let blocks = count / block_elems;
    if raw.len() != blocks * block_bytes {
        return Err(invalid("dequantize input byte length mismatch"));
    }
    let mut out = vec![0.0_f32; count];
    if !is_quantized(kind) {
        for (idx, dst) in out.iter_mut().enumerate() {
            let start = idx * block_bytes;
            *dst = read_unquant_elem_f32(kind, &raw[start..start + block_bytes])?;
        }
        return Ok(out);
    }
    for b in 0..blocks {
        let src = &raw[b * block_bytes..(b + 1) * block_bytes];
        let dst = &mut out[b * block_elems..(b + 1) * block_elems];
        dequant_block(kind, src, dst)?;
    }
    Ok(out)
}

fn inv_scale(d: f32) -> f32 {
    if d == 0.0 { 0.0 } else { 1.0 / d }
}

fn quant_q4_0(src: &[f32], out: &mut [u8]) {
    let mut amax = 0.0_f32;
    let mut max_signed = 0.0_f32;
    for &v in &src[..32] {
        if v.abs() > amax {
            amax = v.abs();
            max_signed = v;
        }
    }
    let d = max_signed / -8.0;
    let inv = inv_scale(d);
    out[0..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
    for idx in 0..16 {
        let low = u8_clamp(src[idx].mul_add(inv, 8.5), 15);
        let high = u8_clamp(src[idx + 16].mul_add(inv, 8.5), 15);
        out[2 + idx] = low | (high << 4);
    }
}

fn quant_q4_1(src: &[f32], out: &mut [u8]) {
    let mut min_v = src[0];
    let mut max_v = src[0];
    for &v in &src[..32] {
        min_v = min_v.min(v);
        max_v = max_v.max(v);
    }
    let d = (max_v - min_v) / 15.0;
    let inv = inv_scale(d);
    out[0..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
    out[2..4].copy_from_slice(&f32_to_f16(min_v).to_le_bytes());
    for idx in 0..16 {
        let low = u8_clamp((src[idx] - min_v).mul_add(inv, 0.5), 15);
        let high = u8_clamp((src[idx + 16] - min_v).mul_add(inv, 0.5), 15);
        out[4 + idx] = low | (high << 4);
    }
}

fn quant_q5_0(src: &[f32], out: &mut [u8]) {
    let mut amax = 0.0_f32;
    let mut max_signed = 0.0_f32;
    for &v in &src[..32] {
        if v.abs() > amax {
            amax = v.abs();
            max_signed = v;
        }
    }
    let d = max_signed / -16.0;
    let inv = inv_scale(d);
    out[0..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
    let mut qh = 0_u32;
    for idx in 0..16 {
        let low = u8_clamp(src[idx].mul_add(inv, 16.5), 31);
        let high = u8_clamp(src[idx + 16].mul_add(inv, 16.5), 31);
        if (low & 16) != 0 {
            qh |= 1_u32 << idx;
        }
        if (high & 16) != 0 {
            qh |= 1_u32 << (idx + 16);
        }
        out[6 + idx] = (low & 0x0f) | ((high & 0x0f) << 4);
    }
    out[2..6].copy_from_slice(&qh.to_le_bytes());
}

fn quant_q5_1(src: &[f32], out: &mut [u8]) {
    let mut min_v = src[0];
    let mut max_v = src[0];
    for &v in &src[..32] {
        min_v = min_v.min(v);
        max_v = max_v.max(v);
    }
    let d = (max_v - min_v) / 31.0;
    let inv = inv_scale(d);
    out[0..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
    out[2..4].copy_from_slice(&f32_to_f16(min_v).to_le_bytes());
    let mut qh = 0_u32;
    for idx in 0..16 {
        let low = u8_clamp((src[idx] - min_v).mul_add(inv, 0.5), 31);
        let high = u8_clamp((src[idx + 16] - min_v).mul_add(inv, 0.5), 31);
        if (low & 16) != 0 {
            qh |= 1_u32 << idx;
        }
        if (high & 16) != 0 {
            qh |= 1_u32 << (idx + 16);
        }
        out[8 + idx] = (low & 0x0f) | ((high & 0x0f) << 4);
    }
    out[4..8].copy_from_slice(&qh.to_le_bytes());
}

fn quant_q8(src: &[f32], out: &mut [u8], with_sum: bool) {
    let amax = src[..32]
        .iter()
        .copied()
        .fold(0.0_f32, |m, v| m.max(v.abs()));
    let d = amax / 127.0;
    let inv = inv_scale(d);
    out[0..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
    let qs_off = if with_sum { 4 } else { 2 };
    let mut sum_v = 0.0_f32;
    for (idx, &v) in src[..32].iter().enumerate() {
        let q = i8_clamp(v * inv);
        out[qs_off + idx] = q.to_le_bytes()[0];
        sum_v = f32::from(q).mul_add(d, sum_v);
    }
    if with_sum {
        out[2..4].copy_from_slice(&f32_to_f16(sum_v).to_le_bytes());
    }
}

fn quant_q8_k(src: &[f32], out: &mut [u8]) {
    let amax = src[..256]
        .iter()
        .copied()
        .fold(0.0_f32, |m, v| m.max(v.abs()));
    let d = amax / 127.0;
    let inv = inv_scale(d);
    out[0..4].copy_from_slice(&d.to_le_bytes());
    for g in 0..16 {
        let mut bsum = 0_i32;
        for l in 0..16 {
            let idx = g * 16 + l;
            let q = i8_clamp(src[idx] * inv);
            out[4 + idx] = q.to_le_bytes()[0];
            bsum += i32::from(q);
        }
        let s = i16::try_from(bsum).unwrap_or(0);
        out[260 + g * 2..262 + g * 2].copy_from_slice(&s.to_le_bytes());
    }
}

fn quant_iq4_nl(src: &[f32], out: &mut [u8]) {
    let amax = src[..32]
        .iter()
        .copied()
        .fold(0.0_f32, |m, v| m.max(v.abs()));
    let d = amax / 127.0;
    let inv = inv_scale(d);
    out[0..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
    let nearest = |val: f32| -> u8 {
        let mut best = 0_usize;
        let mut best_err = f32::INFINITY;
        for (k, &entry) in KVALUES_IQ4NL.iter().enumerate() {
            let err = (val - f32::from(entry)).abs();
            if err < best_err {
                best_err = err;
                best = k;
            }
        }
        u8::try_from(best).unwrap_or(0)
    };
    for idx in 0..16 {
        let lo = nearest(src[idx] * inv);
        let hi = nearest(src[idx + 16] * inv);
        out[2 + idx] = lo | (hi << 4);
    }
}

fn quant_q4_k(src: &[f32], out: &mut [u8]) {
    let mut min_v = 0.0_f32;
    let mut max_rng = 0.0_f32;
    for g in 0..8 {
        let sub = &src[g * 32..g * 32 + 32];
        let lo = sub.iter().copied().fold(f32::INFINITY, f32::min).min(0.0);
        let hi = sub.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        min_v = min_v.max(-lo);
        max_rng = max_rng.max(hi - lo);
    }
    let d = max_rng / (15.0 * 63.0);
    let dmin = min_v / 63.0;
    out[0..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
    out[2..4].copy_from_slice(&f32_to_f16(dmin).to_le_bytes());
    let mut scales_q = [0_u8; 8];
    let mut mins_q = [0_u8; 8];
    for g in 0..8 {
        let sub = &src[g * 32..g * 32 + 32];
        let lo = sub.iter().copied().fold(0.0_f32, f32::min);
        let hi = sub.iter().copied().fold(lo, f32::max);
        let m_q = if dmin > 0.0 {
            u8_clamp(-lo / dmin, 63)
        } else {
            0
        };
        let s_q = if d > 0.0 {
            u8_clamp(f32::from(m_q).mul_add(dmin, hi) / (15.0 * d), 63)
        } else {
            0
        };
        scales_q[g] = s_q;
        mins_q[g] = m_q;
    }
    for idx in 0..4 {
        out[4 + idx] = scales_q[idx] | ((scales_q[idx + 4] >> 4) << 6);
        out[8 + idx] = mins_q[idx] | ((mins_q[idx + 4] >> 4) << 6);
        out[12 + idx] = (scales_q[idx + 4] & 0x0f) | ((mins_q[idx + 4] & 0x0f) << 4);
    }
    for pair in 0..4 {
        let g0 = pair * 2;
        let g1 = g0 + 1;
        let d0 = d * f32::from(scales_q[g0]);
        let m0 = dmin * f32::from(mins_q[g0]);
        let d1 = d * f32::from(scales_q[g1]);
        let m1 = dmin * f32::from(mins_q[g1]);
        let inv0 = inv_scale(d0);
        let inv1 = inv_scale(d1);
        for idx in 0..32 {
            let q0 = u8_clamp((src[g0 * 32 + idx] + m0) * inv0, 15);
            let q1 = u8_clamp((src[g1 * 32 + idx] + m1) * inv1, 15);
            out[16 + pair * 32 + idx] = q0 | (q1 << 4);
        }
    }
}

fn quant_q6_k(src: &[f32], out: &mut [u8]) {
    let amax = src[..256]
        .iter()
        .copied()
        .fold(0.0_f32, |m, v| m.max(v.abs()));
    let d = amax / (31.0 * 127.0);
    out[208..210].copy_from_slice(&f32_to_f16(d).to_le_bytes());
    let mut qs = [0_u8; 256];
    for g in 0..16 {
        let sub = &src[g * 16..g * 16 + 16];
        let s_max = sub.iter().copied().fold(0.0_f32, |m, v| m.max(v.abs()));
        let sc = if d > 0.0 {
            i8_clamp(s_max / (31.0 * d))
        } else {
            0
        };
        out[192 + g] = sc.to_le_bytes()[0];
        let inv = inv_scale(d * f32::from(sc));
        for l in 0..16 {
            qs[g * 16 + l] = u8_clamp(sub[l].mul_add(inv, 32.0), 63);
        }
    }
    for half in 0..2 {
        let low_off = half * 64;
        let high_off = half * 32;
        let base_idx = half * 128;
        for l in 0..32 {
            let q1 = qs[base_idx + l];
            let q2 = qs[base_idx + l + 32];
            let q3 = qs[base_idx + l + 64];
            let q4 = qs[base_idx + l + 96];
            out[low_off + l] = (q1 & 0x0f) | ((q3 & 0x0f) << 4);
            out[low_off + l + 32] = (q2 & 0x0f) | ((q4 & 0x0f) << 4);
            out[128 + high_off + l] =
                (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6);
        }
    }
}

fn quant_block(kind: i32, src: &[f32], out: &mut [u8]) -> Result<()> {
    match kind {
        TYPE_Q4_0 => quant_q4_0(src, out),
        TYPE_Q4_1 => quant_q4_1(src, out),
        TYPE_Q5_0 => quant_q5_0(src, out),
        TYPE_Q5_1 => quant_q5_1(src, out),
        TYPE_Q8_0 => quant_q8(src, out, false),
        TYPE_Q8_1 => quant_q8(src, out, true),
        TYPE_Q8_K => quant_q8_k(src, out),
        TYPE_IQ4_NL => quant_iq4_nl(src, out),
        TYPE_Q4_K => quant_q4_k(src, out),
        TYPE_Q6_K => quant_q6_k(src, out),
        _ => {
            return Err(invalid(format!(
                "unsupported quantized quantize type {kind}"
            )));
        }
    }
    Ok(())
}

pub(super) fn quantize_from_f32(kind: i32, values: &[f32]) -> Result<Vec<u8>> {
    let block_elems = usize::try_from(blck_size(kind))?;
    let block_bytes = usize::try_from(type_size(kind))?;
    if block_elems == 0 || !values.len().is_multiple_of(block_elems) {
        return Err(invalid("quantize element count not aligned to block size"));
    }
    let blocks = values.len() / block_elems;
    let mut out = vec![0_u8; blocks * block_bytes];
    if !is_quantized(kind) {
        match kind {
            TYPE_F32 => {
                for (b, &v) in values.iter().enumerate() {
                    out[b * 4..b * 4 + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
            TYPE_F16 => {
                for (b, &v) in values.iter().enumerate() {
                    out[b * 2..b * 2 + 2].copy_from_slice(&f32_to_f16(v).to_le_bytes());
                }
            }
            TYPE_BF16 => {
                for (b, &v) in values.iter().enumerate() {
                    out[b * 2..b * 2 + 2].copy_from_slice(&f32_to_bf16(v).to_le_bytes());
                }
            }
            _ => {
                return Err(invalid(format!(
                    "quantize-file unsupported scalar type {kind}"
                )));
            }
        }
        return Ok(out);
    }
    for b in 0..blocks {
        let src = &values[b * block_elems..(b + 1) * block_elems];
        let dst = &mut out[b * block_bytes..(b + 1) * block_bytes];
        quant_block(kind, src, dst)?;
    }
    Ok(out)
}
