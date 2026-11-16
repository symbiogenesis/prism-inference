//! Packed 8-lane K-quant integer dot products with baseline-NEON fast paths.

#[cfg(target_arch = "aarch64")]
use super::aarch64;

/// Number of elements per K-quant superblock (`Q4_K`, `Q6_K`, `Q8_K`).
pub const QK_K: usize = 256;
/// Encoded byte length of one GGML `Q4_K` weight superblock.
pub const Q4_K_BLOCK_BYTES: usize = 144;
/// Encoded byte length of one GGML `Q6_K` weight superblock.
pub const Q6_K_BLOCK_BYTES: usize = 210;

#[inline]
pub(super) const fn unpack_q4_k_scale_min(group: usize, scales: &[u8; 12]) -> (u8, u8) {
    if group < 4 {
        (scales[group] & 63, scales[group + 4] & 63)
    } else {
        (
            (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4),
            (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4),
        )
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kquant_8lane_dots_match_portable_reference() {
        let mut values = [0_i8; QK_K];
        for (idx, slot) in values.iter_mut().enumerate() {
            *slot =
                i8::try_from(isize::try_from((idx * 11 + 7) % 255).unwrap_or(0) - 127).unwrap_or(0);
        }
        let sums: [i16; 16] = std::array::from_fn(|group| {
            values[group * 16..(group + 1) * 16]
                .iter()
                .map(|&val| i16::from(val))
                .sum()
        });

        let mut blk4 = [0_u8; Q4_K_BLOCK_BYTES];
        for (idx, byte) in blk4.iter_mut().enumerate() {
            *byte = u8::try_from((idx * 37 + 19) % 256).unwrap_or(0);
        }
        assert_eq!(
            dot_kquant_8lanes_q4_k(&blk4, &values, &sums),
            portable_dot_kquant_8lanes_q4_k(&blk4, &values, &sums)
        );

        let mut blk6 = [0_u8; Q6_K_BLOCK_BYTES];
        for (idx, byte) in blk6.iter_mut().enumerate() {
            *byte = u8::try_from((idx * 43 + 29) % 256).unwrap_or(0);
        }
        assert_eq!(
            dot_kquant_8lanes_q6_k(&blk6, &values),
            portable_dot_kquant_8lanes_q6_k(&blk6, &values)
        );
    }
}
