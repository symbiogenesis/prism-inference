//! Safe K-block codecs and native `Q8_K` integer dot products.
//!
//! Ported from the pinned ggml-quants.c and ggml-cpu/quants.c generic kernels.
//! Upstream MIT attribution: tools/prism/upstream/LICENSE. Packed matrices remain
//! packed; only one 256-element integer block is unpacked at a time.

#![expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::suboptimal_flops,
    reason = "Pinned quantizer bit fields and scalar floating association are explicit ABI operations."
)]

use super::{Q8KBlock, finite, scale};
use crate::{Result, invalid};
mod encode;

pub(super) fn quantize(kind: u32, input: &[f32]) -> Result<Vec<u8>> {
    encode::quantize(kind, input, None)
}

pub(super) fn quantize_weighted(kind: u32, input: &[f32], importance: &[f32]) -> Result<Vec<u8>> {
    encode::quantize(kind, input, Some(importance))
}

pub(super) const fn block_bytes(kind: u32) -> Option<usize> {
    match kind {
        10 => Some(84),
        11 => Some(110),
        12 => Some(144),
        13 => Some(176),
        14 => Some(210),
        _ => None,
    }
}

fn scale_min(group: usize, packed: &[u8]) -> (i16, i16) {
    let (scale_val, min_val) = bonsai_ternary::unpack_q4_k_scale_min(group, packed);
    (i16::from(scale_val), i16::from(min_val))
}

struct Block {
    values: [i16; 256],
    scales: [i16; 16],
    minima: [i16; 16],
    d: f32,
    minimum: f32,
    group: usize,
}

fn unpack(kind: u32, bytes: &[u8]) -> Result<Block> {
    if block_bytes(kind) != Some(bytes.len()) {
        return Err(invalid("K quant block extent differs"));
    }
    let mut block = Block {
        values: [0; 256],
        scales: [0; 16],
        minima: [0; 16],
        d: 0.0,
        minimum: 0.0,
        group: 16,
    };
    match kind {
        10 => {
            block.d = scale(&bytes[80..82])?;
            block.minimum = scale(&bytes[82..84])?;
            for group in 0..16 {
                block.scales[group] = i16::from(bytes[group] & 15);
                block.minima[group] = i16::from(bytes[group] >> 4);
            }
            for (index, value) in block.values.iter_mut().enumerate() {
                *value = i16::from(
                    (bytes[16 + index / 128 * 32 + index % 32] >> (2 * (index % 128 / 32))) & 3,
                );
            }
        }
        11 => {
            block.d = scale(&bytes[108..110])?;
            for group in 0..16 {
                block.scales[group] = i16::from(
                    ((bytes[96 + group % 8] >> (4 * (group / 8))) & 15)
                        | (((bytes[104 + group % 4] >> (2 * (group / 4))) & 3) << 4),
                ) - 32;
            }
            for (index, value) in block.values.iter_mut().enumerate() {
                *value = i16::from(
                    (bytes[32 + index / 128 * 32 + index % 32] >> (2 * (index % 128 / 32))) & 3,
                ) - if bytes[index % 32] & (1 << (index / 32)) == 0 {
                    4
                } else {
                    0
                };
            }
        }
        12 | 13 => {
            block.group = 32;
            block.d = scale(bytes)?;
            block.minimum = scale(&bytes[2..4])?;
            for group in 0..8 {
                (block.scales[group], block.minima[group]) = scale_min(group, &bytes[4..16]);
            }
            let offset = if kind == 12 { 16 } else { 48 };
            for (index, value) in block.values.iter_mut().enumerate() {
                *value = i16::from(
                    (bytes[offset + index / 64 * 32 + index % 32] >> (4 * (index % 64 / 32))) & 15,
                );
                if kind == 13 && bytes[16 + index % 32] & (1 << (index / 32)) != 0 {
                    *value += 16;
                }
            }
        }
        14 => {
            block.d = scale(&bytes[208..210])?;
            for group in 0..16 {
                block.scales[group] = i16::from(i8::from_ne_bytes([bytes[192 + group]]));
            }
            for (index, value) in block.values.iter_mut().enumerate() {
                let part = index % 128 / 32;
                let low =
                    (bytes[index / 128 * 64 + part % 2 * 32 + index % 32] >> (4 * (part / 2))) & 15;
                let high = (bytes[128 + index / 128 * 32 + index % 32] >> (2 * part)) & 3;
                *value = i16::from(low | (high << 4)) - 32;
            }
        }
        _ => return Err(invalid("unknown K quant kind")),
    }
    Ok(block)
}

pub(super) fn validate(kind: u32, bytes: &[u8]) -> Result<()> {
    unpack(kind, bytes)?;
    Ok(())
}

pub(super) fn decode(kind: u32, bytes: &[u8], output: &mut Vec<f32>) -> Result<()> {
    let size = block_bytes(kind).ok_or_else(|| invalid("unknown K quant kind"))?;
    if !bytes.len().is_multiple_of(size) {
        return Err(invalid("K quant row has partial block"));
    }
    output.try_reserve_exact(bytes.len() / size * 256)?;
    for bytes in bytes.chunks_exact(size) {
        let block = unpack(kind, bytes)?;
        for index in 0..256 {
            let group = index / block.group;
            output.push(finite(
                (block.d * f32::from(block.scales[group])) * f32::from(block.values[index])
                    - block.minimum * f32::from(block.minima[group]),
            )?);
        }
    }
    Ok(())
}

fn accumulate_lanes(integers: &mut [i32; 8], sc: i32, sub_lanes: [i32; 8]) {
    for (dst, src) in integers.iter_mut().zip(sub_lanes) {
        *dst += sc * src;
    }
}

fn dot_q2_k_block(
    block_bytes: &[u8],
    q8k: &Q8KBlock,
    activation_scale: f32,
    sum: &mut f32,
) -> Result<()> {
    let d = scale(&block_bytes[80..82])?;
    let minimum = scale(&block_bytes[82..84])?;
    let mut integer_sum = 0_i32;
    let mut minimum_sum = 0_i32;
    for half in 0..2 {
        let q_half = &block_bytes[16 + half * 32..48 + half * 32];
        for part in 0..4 {
            let shift = 2 * part;
            for sub in 0..2 {
                let group = half * 8 + part * 2 + sub;
                let sc = i32::from(block_bytes[group] & 15);
                let mn = i32::from(block_bytes[group] >> 4);
                minimum_sum += i32::from(q8k.sums[group]) * mn;
                let j_base = sub * 16;
                let a_base = group * 16;
                let mut sub_dot = 0_i32;
                for (&q_byte, &act) in q_half[j_base..j_base + 16]
                    .iter()
                    .zip(&q8k.values[a_base..a_base + 16])
                {
                    sub_dot += i32::from((q_byte >> shift) & 3) * i32::from(act);
                }
                integer_sum += sc * sub_dot;
            }
        }
    }
    *sum += (activation_scale * d) * integer_sum as f32
        - (activation_scale * minimum) * minimum_sum as f32;
    Ok(())
}

fn dot_q3_k_block(
    block_bytes: &[u8],
    q8k: &Q8KBlock,
    activation_scale: f32,
    lanes: &mut [f32; 8],
) -> Result<()> {
    let d = scale(&block_bytes[108..110])?;
    let hm = &block_bytes[..32];
    let mut integers = [0_i32; 8];
    for half in 0..2 {
        let q_half = &block_bytes[32 + half * 32..64 + half * 32];
        for part in 0..4 {
            let shift = 2 * part;
            let hm_mask = 1_u8 << (half * 4 + part);
            for sub in 0..2 {
                let group = half * 8 + part * 2 + sub;
                let sc = i32::from(
                    ((block_bytes[96 + group % 8] >> (4 * (group / 8))) & 15)
                        | (((block_bytes[104 + group % 4] >> (2 * (group / 4))) & 3) << 4),
                ) - 32;
                let j_base = sub * 16;
                let a_base = group * 16;
                let mut sub_lanes = [0_i32; 8];
                for half8 in 0..2 {
                    let off = half8 * 8;
                    for (lane, acc) in sub_lanes.iter_mut().enumerate() {
                        let j = j_base + off + lane;
                        let low2 = i32::from((q_half[j] >> shift) & 3);
                        let hsub = if hm[j] & hm_mask == 0 { 4 } else { 0 };
                        *acc += (low2 - hsub) * i32::from(q8k.values[a_base + off + lane]);
                    }
                }
                accumulate_lanes(&mut integers, sc, sub_lanes);
            }
        }
    }
    let d_act = d * activation_scale;
    for (lane, integer) in lanes.iter_mut().zip(integers) {
        *lane += d_act * integer as f32;
    }
    Ok(())
}

fn dot_q4_q5_k_block(
    kind: u32,
    block_bytes: &[u8],
    q8k: &Q8KBlock,
    activation_scale: f32,
    lanes: &mut [f32; 8],
    sum: &mut f32,
) -> Result<()> {
    let d = scale(block_bytes)?;
    let minimum = scale(&block_bytes[2..4])?;
    let scale_bytes = &block_bytes[4..16];
    let qs_offset = if kind == 12 { 16 } else { 48 };
    let high_bits = &block_bytes[16..48];
    let mut integers = [0_i32; 8];
    let mut minimum_sum = 0_i32;
    for pair in 0..4 {
        let q_slice = &block_bytes[qs_offset + pair * 32..qs_offset + (pair + 1) * 32];
        for sub in 0..2 {
            let group = pair * 2 + sub;
            let (sc16, mn16) = scale_min(group, scale_bytes);
            let sc = i32::from(sc16);
            let mn = i32::from(mn16);
            minimum_sum +=
                (i32::from(q8k.sums[group * 2]) + i32::from(q8k.sums[group * 2 + 1])) * mn;
            let shift = 4 * sub;
            let high_mask = 1_u8 << group;
            let a_base = group * 32;
            let mut sub_lanes = [0_i32; 8];
            for oct in 0..4 {
                let off = oct * 8;
                for (lane, acc) in sub_lanes.iter_mut().enumerate() {
                    let j = off + lane;
                    let mut q = i32::from((q_slice[j] >> shift) & 15);
                    if kind == 13 && high_bits[j] & high_mask != 0 {
                        q += 16;
                    }
                    *acc += q * i32::from(q8k.values[a_base + j]);
                }
            }
            accumulate_lanes(&mut integers, sc, sub_lanes);
        }
    }
    let d_act = d * activation_scale;
    for (lane, integer) in lanes.iter_mut().zip(integers) {
        *lane += d_act * integer as f32;
    }
    *sum -= (minimum * activation_scale) * minimum_sum as f32;
    Ok(())
}

#[cfg_attr(
    all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc"
    ),
    expect(
        dead_code,
        reason = "ARM64 NEON 8-lane Q6_K kernel handles kind 14 when experimental-neon is enabled."
    )
)]
fn dot_q6_k_block(
    block_bytes: &[u8],
    q8k: &Q8KBlock,
    activation_scale: f32,
    lanes: &mut [f32; 8],
) -> Result<()> {
    let d = scale(&block_bytes[208..210])?;
    let mut integers = [0_i32; 8];
    for half in 0..2 {
        let high_bytes = &block_bytes[128 + half * 32..160 + half * 32];
        for part in 0..4 {
            let low_bytes =
                &block_bytes[half * 64 + (part % 2) * 32..half * 64 + (part % 2) * 32 + 32];
            let shift_l = 4 * (part / 2);
            let shift_h = 2 * part;
            for sub in 0..2 {
                let group = half * 8 + part * 2 + sub;
                let sc = i32::from(i8::from_ne_bytes([block_bytes[192 + group]]));
                let j_base = sub * 16;
                let a_base = group * 16;
                let mut sub_lanes = [0_i32; 8];
                for half8 in 0..2 {
                    let off = half8 * 8;
                    for (lane, acc) in sub_lanes.iter_mut().enumerate() {
                        let j = j_base + off + lane;
                        let low = (low_bytes[j] >> shift_l) & 15;
                        let high = (high_bytes[j] >> shift_h) & 3;
                        let q = i32::from(low | (high << 4)) - 32;
                        *acc += q * i32::from(q8k.values[a_base + off + lane]);
                    }
                }
                accumulate_lanes(&mut integers, sc, sub_lanes);
            }
        }
    }
    let d_act = d * activation_scale;
    for (lane, integer) in lanes.iter_mut().zip(integers) {
        *lane += d_act * integer as f32;
    }
    Ok(())
}

pub(super) fn dot(kind: u32, weights: &[u8], input: &[Q8KBlock]) -> Result<f32> {
    let size = block_bytes(kind).ok_or_else(|| invalid("unknown K quant kind"))?;
    if input.len().checked_mul(size) != Some(weights.len()) {
        return Err(invalid("K quant dot block counts differ"));
    }
    let mut sum = 0.0_f32;
    let mut lanes = [0.0_f32; 8];
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc"
    ))]
    if kind == 12 {
        for (weights, input) in weights.as_chunks::<144>().0.iter().zip(input) {
            let block_d = scale(weights)?;
            let block_minimum = scale(&weights[2..4])?;
            let activation_scale = finite(input.scale)?;
            let (integers, minimum_sum) = bonsai_neon_dot::kquants::dot_kquant_8lanes_q4_k(
                weights,
                &input.values,
                &input.sums,
            );
            let d = block_d * activation_scale;
            for (lane, integer) in lanes.iter_mut().zip(integers) {
                *lane += d * integer as f32;
            }
            sum -= (block_minimum * activation_scale) * minimum_sum as f32;
        }
        for lane in lanes {
            sum += lane;
        }
        return finite(sum);
    }
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc"
    ))]
    if kind == 14 {
        for (weights, input) in weights.as_chunks::<210>().0.iter().zip(input) {
            let block_d = scale(&weights[208..210])?;
            let activation_scale = finite(input.scale)?;
            let integers = bonsai_neon_dot::kquants::dot_kquant_8lanes_q6_k(weights, &input.values);
            let d = block_d * activation_scale;
            for (lane, integer) in lanes.iter_mut().zip(integers) {
                *lane += d * integer as f32;
            }
        }
        for lane in lanes {
            sum += lane;
        }
        return finite(sum);
    }
    for (block_bytes, q8k) in weights.chunks_exact(size).zip(input) {
        let activation_scale = finite(q8k.scale)?;
        match kind {
            10 => dot_q2_k_block(block_bytes, q8k, activation_scale, &mut sum)?,
            11 => dot_q3_k_block(block_bytes, q8k, activation_scale, &mut lanes)?,
            12 | 13 => dot_q4_q5_k_block(
                kind,
                block_bytes,
                q8k,
                activation_scale,
                &mut lanes,
                &mut sum,
            )?,
            #[cfg(not(all(
                feature = "experimental-neon",
                target_arch = "aarch64",
                target_os = "windows",
                target_env = "msvc"
            )))]
            14 => dot_q6_k_block(block_bytes, q8k, activation_scale, &mut lanes)?,
            _ => return Err(invalid("unknown K quant kind")),
        }
    }
    if kind != 10 {
        for lane in lanes {
            sum += lane;
        }
    }
    finite(sum)
}

#[cfg(test)]
mod tests {
    use super::{Q8KBlock, block_bytes, dot, finite, unpack};
    use crate::Result;

    fn dot_via_unpack(kind: u32, weights: &[u8], input: &[Q8KBlock]) -> Result<f32> {
        let size = block_bytes(kind).unwrap_or(1);
        let mut sum = 0.0_f32;
        let mut lanes = [0.0_f32; 8];
        for (weights, input) in weights.chunks_exact(size).zip(input) {
            let block = unpack(kind, weights)?;
            let activation_scale = finite(input.scale)?;
            let mut integers = [0_i32; 8];
            let mut minimum_sum = 0_i32;
            for index in 0..256 {
                integers[index % 8] += i32::from(block.scales[index / block.group])
                    * i32::from(block.values[index])
                    * i32::from(input.values[index]);
            }
            for group in 0..16 {
                minimum_sum += i32::from(input.sums[group])
                    * i32::from(block.minima[group * 16 / block.group]);
            }
            if kind == 10 {
                let integer_sum: i32 = integers.into_iter().sum();
                sum += (activation_scale * block.d) * integer_sum as f32
                    - (activation_scale * block.minimum) * minimum_sum as f32;
            } else {
                let d = block.d * activation_scale;
                for (lane, integer) in lanes.iter_mut().zip(integers) {
                    *lane += d * integer as f32;
                }
                if matches!(kind, 12 | 13) {
                    sum -= (block.minimum * activation_scale) * minimum_sum as f32;
                }
            }
        }
        if kind != 10 {
            for lane in lanes {
                sum += lane;
            }
        }
        finite(sum)
    }

    #[test]
    fn direct_kquant_dot_matches_unpack_bits_across_all_kinds() -> Result<()> {
        let f16_d = 0x3555_u16.to_le_bytes();
        let f16_m = 0xb955_u16.to_le_bytes();
        for kind in 10..=14 {
            let size = block_bytes(kind).unwrap_or(0);
            let blocks = 3_usize;
            let mut weights = vec![0_u8; size * blocks];
            for (b, blk) in weights.chunks_exact_mut(size).enumerate() {
                for (i, slot) in blk.iter_mut().enumerate() {
                    *slot = u8::try_from((b * 73 + i * 37 + 19) % 256).unwrap_or(0);
                }
                match kind {
                    10 => {
                        blk[80..82].copy_from_slice(&f16_d);
                        blk[82..84].copy_from_slice(&f16_m);
                    }
                    11 => blk[108..110].copy_from_slice(&f16_d),
                    12 | 13 => {
                        blk[0..2].copy_from_slice(&f16_d);
                        blk[2..4].copy_from_slice(&f16_m);
                    }
                    14 => blk[208..210].copy_from_slice(&f16_d),
                    _ => {}
                }
            }
            let input: Vec<Q8KBlock> = (0..blocks)
                .map(|b| {
                    let mut values = [0_i8; 256];
                    let mut sums = [0_i16; 16];
                    for (i, slot) in values.iter_mut().enumerate() {
                        let v = i16::try_from((b * 41 + i * 17) % 251).unwrap_or(0) - 125;
                        *slot = i8::try_from(v).unwrap_or(0);
                        sums[i / 16] += v;
                    }
                    Q8KBlock {
                        scale: 0.03125 * (f32::from(u8::try_from(b + 1).unwrap_or(1))),
                        values,
                        sums,
                    }
                })
                .collect();

            let actual = dot(kind, &weights, &input)?;
            let expected = dot_via_unpack(kind, &weights, &input)?;
            assert_eq!(actual.to_bits(), expected.to_bits(), "kind={kind}");

            // Reject non-matching lengths and non-finite half/activation scales.
            assert!(dot(kind, &weights[..size - 1], &input[..1]).is_err());
            let mut bad_input = input;
            bad_input[0].scale = f32::NAN;
            assert!(dot(kind, &weights, &bad_input).is_err());
        }
        assert!(dot(99, &[], &[]).is_err());
        Ok(())
    }
}
