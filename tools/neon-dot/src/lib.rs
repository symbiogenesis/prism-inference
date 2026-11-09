//! Audited ARMv8.2-A `dotprod` (`vdotq_s32`) and `i8mm` (`vmmlaq_s32`) SIMD
//! dot-product and non-widening `PTQ1_0` base-3 unpacking primitives.
//!
//! All public entry points accept fixed-size array references and fall back
//! cleanly to exact portable integer arithmetic when `aarch64`, `dotprod`, or
//! `i8mm` is unavailable.

#![deny(unsafe_code)]

#[cfg(target_arch = "aarch64")]
#[allow(
    unsafe_code,
    reason = "Wraps pure value-based ARM64 NEON, dotprod, and i8mm intrinsics behind safe fixed-array boundaries."
)]
mod aarch64;

/// Returns whether the current CPU supports ARMv8.2-A `dotprod` (`SDOT`).
#[inline]
#[must_use]
pub fn has_dotprod() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        aarch64::has_dotprod()
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// Returns whether the current CPU supports ARMv8.2-A `i8mm` (`SMMLA`).
#[inline]
#[must_use]
pub fn has_i8mm() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        aarch64::has_i8mm()
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
const fn extract_trit_scalar(v: u8) -> i8 {
    if v < 86 {
        -1
    } else if v < 171 {
        0
    } else {
        1
    }
}

/// Decodes all 128 signed trits (`-1..=1`) of a 28-byte `PTQ1_0` block into
/// eight 16-element `i8` vectors without 16-bit widening or serial dependencies.
#[inline]
#[must_use]
pub fn decode_ptq1_trits_i8(packed: &[u8; 28]) -> [[i8; 16]; 8] {
    #[cfg(target_arch = "aarch64")]
    {
        aarch64::decode_ptq1_trits_i8(packed)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        const POW3: [u8; 5] = [1, 3, 9, 27, 81];
        const POW3_TAIL: [u8; 8] = [1, 1, 3, 3, 9, 9, 27, 27];
        let mut out = [[0_i8; 16]; 8];
        for (k, &mul) in POW3.iter().enumerate() {
            for m in 0..16 {
                out[k][m] = extract_trit_scalar(packed[m].wrapping_mul(mul));
            }
        }
        for m in 0..8 {
            out[5][m] = extract_trit_scalar(packed[16 + m]);
            out[5][8 + m] = extract_trit_scalar(packed[16 + m].wrapping_mul(3));
            out[6][m] = extract_trit_scalar(packed[16 + m].wrapping_mul(9));
            out[6][8 + m] = extract_trit_scalar(packed[16 + m].wrapping_mul(27));
            out[7][m] = extract_trit_scalar(packed[16 + m].wrapping_mul(81));
            let qh = packed[24 + (m & 1)];
            out[7][8 + m] = extract_trit_scalar(qh.wrapping_mul(POW3_TAIL[m]));
        }
        out
    }
}

/// Computes the four 8-element `DOTPROD` lane sums of two 16-element `i8`
/// weight halves against two 16-element `i8` activation halves:
/// `lane[l] = sum_{k=0..4} (w0[4*l+k]*a0[4*l+k] + w1[4*l+k]*a1[4*l+k])`.
#[inline]
#[must_use]
pub fn dot32_s8_lanes(w0: &[i8; 16], w1: &[i8; 16], a0: &[i8; 16], a1: &[i8; 16]) -> [i32; 4] {
    #[cfg(target_arch = "aarch64")]
    if let Some(lanes) = aarch64::dot32_s8_lanes(w0, w1, a0, a1) {
        return lanes;
    }
    std::array::from_fn(|lane| {
        let base = lane * 4;
        (0..4)
            .map(|k| {
                i32::from(w0[base + k]) * i32::from(a0[base + k])
                    + i32::from(w1[base + k]) * i32::from(a1[base + k])
            })
            .sum()
    })
}

/// Computes the 32-element signed integer dot product of two 16-element `i8`
/// weight halves against two 16-element `i8` activation halves.
#[inline]
#[must_use]
pub fn dot32_s8_sum(w0: &[i8; 16], w1: &[i8; 16], a0: &[i8; 16], a1: &[i8; 16]) -> i32 {
    #[cfg(target_arch = "aarch64")]
    if let Some(sum) = aarch64::dot32_s8_sum(w0, w1, a0, a1) {
        return sum;
    }
    let lanes = dot32_s8_lanes(w0, w1, a0, a1);
    lanes[0] + lanes[1] + lanes[2] + lanes[3]
}

/// Unpacks a 28-byte `PTQ1_0` block in 8-bit registers and computes the four
/// 4-lane `DOTPROD` sums against four 32-element `Q8_0` activation groups.
#[inline]
#[must_use]
pub fn dot_ptq1_block_lanes(packed: &[u8; 28], activations: [&[i8; 32]; 4]) -> [[i32; 4]; 4] {
    #[cfg(target_arch = "aarch64")]
    if let Some(lanes) = aarch64::dot_ptq1_block_lanes(packed, activations) {
        return lanes;
    }
    let trits = decode_ptq1_trits_i8(packed);
    std::array::from_fn(|group| {
        let halves = activations[group].as_chunks::<16>().0;
        dot32_s8_lanes(
            &trits[group * 2],
            &trits[group * 2 + 1],
            &halves[0],
            &halves[1],
        )
    })
}

/// Unpacks a 28-byte `PTQ1_0` block in 8-bit registers and computes the four
/// 32-element group sums against four 32-element `Q8_0`/`Q8_K` activation slices.
#[inline]
#[must_use]
pub fn dot_ptq1_block_groups(packed: &[u8; 28], activations: [&[i8; 32]; 4]) -> [i32; 4] {
    #[cfg(target_arch = "aarch64")]
    if let Some(groups) = aarch64::dot_ptq1_block_groups(packed, activations) {
        return groups;
    }
    let trits = decode_ptq1_trits_i8(packed);
    std::array::from_fn(|group| {
        let halves = activations[group].as_chunks::<16>().0;
        dot32_s8_sum(
            &trits[group * 2],
            &trits[group * 2 + 1],
            &halves[0],
            &halves[1],
        )
    })
}

/// Unpacks four 28-byte `PTQ1_0` blocks and computes their four 4-lane
/// `DOTPROD` sums against four shared 32-element `Q8_0` activation groups.
#[inline]
#[must_use]
pub fn dot_ptq1_four_rows_lanes(
    packed: [&[u8; 28]; 4],
    activations: [&[i8; 32]; 4],
) -> [[[i32; 4]; 4]; 4] {
    #[cfg(target_arch = "aarch64")]
    if let Some(lanes) = aarch64::dot_ptq1_four_rows_lanes(packed, activations) {
        return lanes;
    }
    packed.map(|row| dot_ptq1_block_lanes(row, activations))
}

/// Unpacks four 28-byte `PTQ1_0` blocks and computes their four 32-element
/// group sums against four shared 32-element `Q8_0` activation groups.
#[inline]
#[must_use]
pub fn dot_ptq1_four_rows_groups(
    packed: [&[u8; 28]; 4],
    activations: [&[i8; 32]; 4],
) -> [[i32; 4]; 4] {
    #[cfg(target_arch = "aarch64")]
    if let Some(groups) = aarch64::dot_ptq1_four_rows_groups(packed, activations) {
        return groups;
    }
    packed.map(|row| dot_ptq1_block_groups(row, activations))
}

/// Four-row by four-token by four-group by four-lane `DOTPROD` sums (`[row][token][group][lane]`).
pub type Ptq1FourRowFourTokenLanes = [[[[i32; 4]; 4]; 4]; 4];

/// Four-row by four-token by four-group 32-element integer dot sums (`[row][token][group]`).
pub type Ptq1FourRowFourTokenGroups = [[[i32; 4]; 4]; 4];

/// Computes the 4-row by 4-token by 4-group 4-lane `DOTPROD` sums from
/// pre-decoded `PTQ1_0` trit rows and four tokens of `Q8_0` activation groups.
#[inline]
#[must_use]
pub fn dot_ptq1_4x4_lanes(
    trits: &[[[i8; 16]; 8]; 4],
    activations: [[&[i8; 32]; 4]; 4],
) -> Ptq1FourRowFourTokenLanes {
    #[cfg(target_arch = "aarch64")]
    if let Some(lanes) = aarch64::dot_ptq1_4x4_lanes(trits, activations) {
        return lanes;
    }
    std::array::from_fn(|r| {
        std::array::from_fn(|t| {
            std::array::from_fn(|group| {
                let halves = activations[t][group].as_chunks::<16>().0;
                dot32_s8_lanes(
                    &trits[r][group * 2],
                    &trits[r][group * 2 + 1],
                    &halves[0],
                    &halves[1],
                )
            })
        })
    })
}

/// Computes the 4-row by 4-token by 4-group 32-element integer dot sums from
/// pre-decoded `PTQ1_0` trit rows and four tokens of `Q8_0` activation groups.
#[inline]
#[must_use]
pub fn dot_ptq1_4x4_groups(
    trits: &[[[i8; 16]; 8]; 4],
    activations: [[&[i8; 32]; 4]; 4],
) -> Ptq1FourRowFourTokenGroups {
    #[cfg(target_arch = "aarch64")]
    if let Some(groups) = aarch64::dot_ptq1_4x4_groups(trits, activations) {
        return groups;
    }
    std::array::from_fn(|r| {
        std::array::from_fn(|t| {
            std::array::from_fn(|group| {
                let halves = activations[t][group].as_chunks::<16>().0;
                dot32_s8_sum(
                    &trits[r][group * 2],
                    &trits[r][group * 2 + 1],
                    &halves[0],
                    &halves[1],
                )
            })
        })
    })
}

/// Computes the 128-element `PQ2_0` dot product against digit-plane-ordered
/// `i8` activations and their precomputed `activation_sum` using `SDOT`.
#[inline]
#[must_use]
pub fn dot_pq2_consumption_block(
    packed: &[u8; 32],
    planes: &[i8; 128],
    activation_sum: i32,
) -> i32 {
    #[cfg(target_arch = "aarch64")]
    if let Some(sum) = aarch64::dot_pq2_consumption_block(packed, planes, activation_sum) {
        return sum;
    }
    let unsigned_sum: i32 = packed
        .as_chunks::<16>()
        .0
        .iter()
        .zip(planes.as_chunks::<64>().0)
        .map(|(bytes, values)| {
            bytes
                .iter()
                .enumerate()
                .map(|(index, &byte)| {
                    (0..4)
                        .map(|digit| {
                            i32::from((byte >> (digit * 2)) & 3)
                                * i32::from(values[digit * 16 + index])
                        })
                        .sum::<i32>()
                })
                .sum::<i32>()
        })
        .sum();
    unsigned_sum - activation_sum
}

/// Computes four 128-element `PQ2_0` dot products against shared digit-plane
/// `i8` activations and their precomputed `activation_sum` using `SDOT`.
#[inline]
#[must_use]
pub fn dot_pq2_four_consumption_precomputed(
    packed: [&[u8; 32]; 4],
    planes: &[i8; 128],
    activation_sum: i32,
) -> [i32; 4] {
    #[cfg(target_arch = "aarch64")]
    if let Some(sums) =
        aarch64::dot_pq2_four_consumption_precomputed(packed, planes, activation_sum)
    {
        return sums;
    }
    packed.map(|row| dot_pq2_consumption_block(row, planes, activation_sum))
}

/// Computes eight 128-element `PQ2_0` dot products against shared digit-plane
/// `i8` activations and their precomputed `activation_sum` using `SDOT`.
#[inline]
#[must_use]
pub fn dot_pq2_eight_consumption_precomputed(
    packed: [&[u8; 32]; 8],
    planes: &[i8; 128],
    activation_sum: i32,
) -> [i32; 8] {
    #[cfg(target_arch = "aarch64")]
    if let Some(sums) =
        aarch64::dot_pq2_eight_consumption_precomputed(packed, planes, activation_sum)
    {
        return sums;
    }
    packed.map(|row| dot_pq2_consumption_block(row, planes, activation_sum))
}

/// Computes four 128-element `PQ2_0` dot products against shared digit-plane
/// `i8` activations using `SDOT`.
#[inline]
#[must_use]
pub fn dot_pq2_four_consumption(packed: [&[u8; 32]; 4], planes: &[i8; 128]) -> [i32; 4] {
    #[cfg(target_arch = "aarch64")]
    if let Some(sums) = aarch64::dot_pq2_four_consumption(packed, planes) {
        return sums;
    }
    let activation_sum: i32 = planes.iter().map(|&v| i32::from(v)).sum();
    packed.map(|row| dot_pq2_consumption_block(row, planes, activation_sum))
}

/// Computes the 32-element signed integer dot product of a `Q8_0` 32-byte
/// weight payload against a 32-element `Q8_0` activation slice using `SDOT`.
#[inline]
#[must_use]
pub fn dot_q8_0_block(weights: &[u8; 32], activations: &[i8; 32]) -> i32 {
    #[cfg(target_arch = "aarch64")]
    if let Some(sum) = aarch64::dot_q8_0_block(weights, activations) {
        return sum;
    }
    weights
        .iter()
        .zip(activations)
        .map(|(&w, &a)| i32::from(i8::from_ne_bytes([w])) * i32::from(a))
        .sum()
}

/// Computes a 2-row by 2-column 32-element `Q8_0` matrix-matrix dot tile using
/// ARMv8.2-A `I8MM` (`SMMLA`) when available, or `SDOT`/scalar fallback.
#[inline]
#[must_use]
pub fn smmla_q8_0_2x2(r0: &[u8; 32], r1: &[u8; 32], c0: &[i8; 32], c1: &[i8; 32]) -> [[i32; 2]; 2] {
    #[cfg(target_arch = "aarch64")]
    if let Some(tile) = aarch64::smmla_q8_0_2x2(r0, r1, c0, c1) {
        return tile;
    }
    [
        [dot_q8_0_block(r0, c0), dot_q8_0_block(r0, c1)],
        [dot_q8_0_block(r1, c0), dot_q8_0_block(r1, c1)],
    ]
}

/// Computes a 2-row by 2-column 32-element `i8` matrix-matrix dot tile using
/// ARMv8.2-A `I8MM` (`vmmlaq_s32`) when available, or `SDOT`/scalar fallback.
#[inline]
#[must_use]
pub fn smmla_dot32_2x2(
    r0: &[i8; 32],
    r1: &[i8; 32],
    c0: &[i8; 32],
    c1: &[i8; 32],
) -> [[i32; 2]; 2] {
    #[cfg(target_arch = "aarch64")]
    if let Some(tile) = aarch64::smmla_dot32_2x2(r0, r1, c0, c1) {
        return tile;
    }
    let r0_h = r0.as_chunks::<16>().0;
    let r1_h = r1.as_chunks::<16>().0;
    let c0_h = c0.as_chunks::<16>().0;
    let c1_h = c1.as_chunks::<16>().0;
    [
        [
            dot32_s8_sum(&r0_h[0], &r0_h[1], &c0_h[0], &c0_h[1]),
            dot32_s8_sum(&r0_h[0], &r0_h[1], &c1_h[0], &c1_h[1]),
        ],
        [
            dot32_s8_sum(&r1_h[0], &r1_h[1], &c0_h[0], &c0_h[1]),
            dot32_s8_sum(&r1_h[0], &r1_h[1], &c1_h[0], &c1_h[1]),
        ],
    ]
}

#[inline]
fn dot_pq2_sequential_scalar(
    packed: &[u8; 32],
    activation: &[i8; 128],
    activation_sum: i32,
) -> i32 {
    let unsigned_sum: i32 = packed
        .as_chunks::<16>()
        .0
        .iter()
        .zip(activation.as_chunks::<64>().0)
        .map(|(bytes, values)| {
            bytes
                .iter()
                .enumerate()
                .map(|(index, &byte)| {
                    (0..4)
                        .map(|digit| {
                            i32::from((byte >> (digit * 2)) & 3)
                                * i32::from(values[index * 4 + digit])
                        })
                        .sum::<i32>()
                })
                .sum::<i32>()
        })
        .sum();
    unsigned_sum - activation_sum
}

/// Computes four 128-element `PQ2_0` row dot products against one sequential
/// `Q8_K` 128-element activation slice and its precomputed `activation_sum`.
#[inline]
#[must_use]
pub fn dot_pq2_four_rows_one_token(
    packed: [&[u8; 32]; 4],
    activation: &[i8; 128],
    activation_sum: i32,
) -> [i32; 4] {
    #[cfg(target_arch = "aarch64")]
    if let Some(sums) = aarch64::dot_pq2_four_rows_one_token(packed, activation, activation_sum) {
        return sums;
    }
    packed.map(|row| dot_pq2_sequential_scalar(row, activation, activation_sum))
}

/// Computes a 4-row by 4-token 128-element `PQ2_0` dot tile against sequential
/// `Q8_K` 128-element activation slices and their precomputed `activation_sums`
/// using ARMv8.2-A `I8MM` (`SMMLA`) or `SDOT` when available.
#[inline]
#[must_use]
pub fn dot_pq2_four_rows_four_tokens(
    packed: [&[u8; 32]; 4],
    activations: [&[i8; 128]; 4],
    activation_sums: [i32; 4],
) -> [[i32; 4]; 4] {
    #[cfg(target_arch = "aarch64")]
    if let Some(tile) = aarch64::dot_pq2_four_rows_four_tokens(packed, activations, activation_sums)
    {
        return tile;
    }
    packed.map(|row| {
        std::array::from_fn(|tok| {
            dot_pq2_sequential_scalar(row, activations[tok], activation_sums[tok])
        })
    })
}

/// Computes a 4-row by 4-token 128-element `PQ2_0` dot tile against
/// pre-deinterleaved digit-plane `i8` activations and their precomputed
/// `activation_sums` using ARMv8.2-A `I8MM` (`SMMLA`) or `SDOT` when available.
#[inline]
#[must_use]
pub fn dot_pq2_four_rows_four_tokens_precomputed(
    packed: [&[u8; 32]; 4],
    planes: [&[i8; 128]; 4],
    activation_sums: [i32; 4],
) -> [[i32; 4]; 4] {
    #[cfg(target_arch = "aarch64")]
    if let Some(tile) =
        aarch64::dot_pq2_four_rows_four_tokens_precomputed(packed, planes, activation_sums)
    {
        return tile;
    }
    packed.map(|row| {
        std::array::from_fn(|tok| {
            dot_pq2_consumption_block(row, planes[tok], activation_sums[tok])
        })
    })
}

/// Computes an 8-row by 4-token 128-element `PQ2_0` dot tile against
/// pre-deinterleaved digit-plane `i8` activations and their precomputed
/// `activation_sums` using ARMv8.2-A `I8MM` (`SMMLA`) or `SDOT` when available.
#[inline]
#[must_use]
pub fn dot_pq2_eight_rows_four_tokens_precomputed(
    packed: [&[u8; 32]; 8],
    planes: [&[i8; 128]; 4],
    activation_sums: [i32; 4],
) -> [[i32; 4]; 8] {
    #[cfg(target_arch = "aarch64")]
    if let Some(tile) =
        aarch64::dot_pq2_eight_rows_four_tokens_precomputed(packed, planes, activation_sums)
    {
        return tile;
    }
    packed.map(|row| {
        std::array::from_fn(|tok| {
            dot_pq2_consumption_block(row, planes[tok], activation_sums[tok])
        })
    })
}

/// Computes an 8-row by 8-token 128-element `PQ2_0` dot tile against
/// pre-deinterleaved digit-plane `i8` activations and their precomputed
/// `activation_sums` using ARMv8.2-A `I8MM` (`SMMLA`) or `SDOT` when available.
#[inline]
#[must_use]
pub fn dot_pq2_eight_rows_eight_tokens_precomputed(
    packed: [&[u8; 32]; 8],
    planes: [&[i8; 128]; 8],
    activation_sums: [i32; 8],
) -> [[i32; 8]; 8] {
    #[cfg(target_arch = "aarch64")]
    if let Some(tile) =
        aarch64::dot_pq2_eight_rows_eight_tokens_precomputed(packed, planes, activation_sums)
    {
        return tile;
    }
    packed.map(|row| {
        std::array::from_fn(|tok| {
            dot_pq2_consumption_block(row, planes[tok], activation_sums[tok])
        })
    })
}

/// Computes an 8-row by 8-token 128-element `PQ2_0` dot tile against sequential
/// `Q8_K` 128-element activation slices and their precomputed `activation_sums`
/// using ARMv8.2-A `I8MM` (`SMMLA`) or `SDOT` when available.
#[inline]
#[must_use]
pub fn dot_pq2_eight_rows_eight_tokens(
    packed: [&[u8; 32]; 8],
    activations: [&[i8; 128]; 8],
    activation_sums: [i32; 8],
) -> [[i32; 8]; 8] {
    #[cfg(target_arch = "aarch64")]
    if let Some(tile) =
        aarch64::dot_pq2_eight_rows_eight_tokens(packed, activations, activation_sums)
    {
        return tile;
    }
    packed.map(|row| {
        std::array::from_fn(|tok| {
            dot_pq2_sequential_scalar(row, activations[tok], activation_sums[tok])
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_ptq1_trits(packed: &[u8; 28]) -> [i8; 128] {
        let mut digits = [[0_u8; 5]; 26];
        for (idx, &b) in packed[..26].iter().enumerate() {
            let mut frac = u16::from(b);
            for slot in &mut digits[idx] {
                frac *= 3;
                *slot = (frac >> 8).to_le_bytes()[0];
                frac &= 255;
            }
        }
        std::array::from_fn(|index| {
            let (lane, digit) = if index < 80 {
                (index % 16, index / 16)
            } else if index < 120 {
                (16 + (index - 80) % 8, (index - 80) / 8)
            } else {
                (24 + (index - 120) % 2, (index - 120) / 2)
            };
            i8::try_from(i16::from(digits[lane][digit]) - 1).unwrap_or(0)
        })
    }

    #[test]
    fn exhaustive_ptq1_u8_unpack_matches_reference_for_all_bytes() {
        for pos in 0..26 {
            for byte in 0..=u8::MAX {
                let mut packed = [0_u8; 28];
                packed[pos] = byte;
                let expected = reference_ptq1_trits(&packed);
                let decoded = decode_ptq1_trits_i8(&packed);
                for (group_idx, chunk) in decoded.iter().enumerate() {
                    assert_eq!(
                        chunk.as_slice(),
                        &expected[group_idx * 16..(group_idx + 1) * 16],
                        "mismatch at pos={pos}, byte={byte}, group={group_idx}"
                    );
                }
            }
        }
    }

    #[test]
    fn sdot_and_smmla_match_scalar_reference() {
        let r0: [i8; 32] = std::array::from_fn(|i| {
            i8::from_ne_bytes([i.to_le_bytes()[0].wrapping_mul(53).wrapping_add(11)])
        });
        let r1: [i8; 32] = std::array::from_fn(|i| {
            i8::from_ne_bytes([i.to_le_bytes()[0].wrapping_mul(97).wrapping_add(29)])
        });
        let c0: [i8; 32] = std::array::from_fn(|i| {
            i8::from_ne_bytes([i.to_le_bytes()[0].wrapping_mul(131).wrapping_add(7)])
        });
        let c1: [i8; 32] = std::array::from_fn(|i| {
            i8::from_ne_bytes([i.to_le_bytes()[0].wrapping_mul(193).wrapping_add(3)])
        });

        let tile = smmla_dot32_2x2(&r0, &r1, &c0, &c1);
        let ref_dot = |a: &[i8; 32], b: &[i8; 32]| -> i32 {
            a.iter()
                .zip(b)
                .map(|(&x, &y)| i32::from(x) * i32::from(y))
                .sum()
        };
        assert_eq!(tile[0][0], ref_dot(&r0, &c0));
        assert_eq!(tile[0][1], ref_dot(&r0, &c1));
        assert_eq!(tile[1][0], ref_dot(&r1, &c0));
        assert_eq!(tile[1][1], ref_dot(&r1, &c1));
    }

    #[test]
    fn pq2_multi_row_multi_token_tiles_match_scalar_reference() {
        let rows: [[u8; 32]; 8] = std::array::from_fn(|r| {
            std::array::from_fn(|i| {
                r.to_le_bytes()[0]
                    .wrapping_mul(37)
                    .wrapping_add(i.to_le_bytes()[0].wrapping_mul(73))
                    .wrapping_add(19)
            })
        });
        let acts: [[i8; 128]; 8] = std::array::from_fn(|t| {
            std::array::from_fn(|i| {
                i8::from_ne_bytes([t.to_le_bytes()[0]
                    .wrapping_mul(59)
                    .wrapping_add(i.to_le_bytes()[0].wrapping_mul(107))
                    .wrapping_add(5)])
            })
        });
        let sums: [i32; 8] = std::array::from_fn(|t| acts[t].iter().map(|&v| i32::from(v)).sum());
        let row_refs8: [&[u8; 32]; 8] = std::array::from_fn(|r| &rows[r]);
        let act_refs8: [&[i8; 128]; 8] = std::array::from_fn(|t| &acts[t]);
        let tile8 = dot_pq2_eight_rows_eight_tokens(row_refs8, act_refs8, sums);
        for r in 0..8 {
            for t in 0..8 {
                let expected = dot_pq2_sequential_scalar(&rows[r], &acts[t], sums[t]);
                assert_eq!(tile8[r][t], expected, "mismatch at 8x8 r={r}, t={t}");
            }
        }
        let row_refs4: [&[u8; 32]; 4] = [&rows[0], &rows[1], &rows[2], &rows[3]];
        let act_refs4: [&[i8; 128]; 4] = [&acts[0], &acts[1], &acts[2], &acts[3]];
        let sums4 = [sums[0], sums[1], sums[2], sums[3]];
        let tile4 = dot_pq2_four_rows_four_tokens(row_refs4, act_refs4, sums4);
        for r in 0..4 {
            assert_eq!(
                tile4[r],
                [tile8[r][0], tile8[r][1], tile8[r][2], tile8[r][3]]
            );
        }
        let col0 = dot_pq2_four_rows_one_token(row_refs4, &acts[0], sums[0]);
        for r in 0..4 {
            assert_eq!(col0[r], tile8[r][0]);
        }

        let planes: [[i8; 128]; 8] = std::array::from_fn(|t| {
            std::array::from_fn(|index| {
                acts[t][(index / 64) * 64 + (index % 16) * 4 + (index / 16) % 4]
            })
        });
        let plane_refs8: [&[i8; 128]; 8] = std::array::from_fn(|t| &planes[t]);
        let plane_refs4: [&[i8; 128]; 4] = [&planes[0], &planes[1], &planes[2], &planes[3]];
        assert_eq!(
            dot_pq2_eight_rows_eight_tokens_precomputed(row_refs8, plane_refs8, sums),
            tile8
        );
        assert_eq!(
            dot_pq2_four_rows_four_tokens_precomputed(row_refs4, plane_refs4, sums4),
            tile4
        );
        let tile8x4 = dot_pq2_eight_rows_four_tokens_precomputed(row_refs8, plane_refs4, sums4);
        for r in 0..8 {
            assert_eq!(
                tile8x4[r],
                [tile8[r][0], tile8[r][1], tile8[r][2], tile8[r][3]]
            );
        }
    }

    #[test]
    fn ptq1_4x4_lanes_and_groups_match_single_row_single_token_primitives() {
        let rows: [[u8; 28]; 4] = std::array::from_fn(|r| {
            std::array::from_fn(|i| {
                r.to_le_bytes()[0]
                    .wrapping_mul(41)
                    .wrapping_add(i.to_le_bytes()[0].wrapping_mul(89))
                    .wrapping_add(13)
            })
        });
        let trits = rows.map(|row| decode_ptq1_trits_i8(&row));
        let acts: [[[i8; 32]; 4]; 4] = std::array::from_fn(|t| {
            std::array::from_fn(|g| {
                std::array::from_fn(|i| {
                    i8::from_ne_bytes([t.to_le_bytes()[0]
                        .wrapping_mul(67)
                        .wrapping_add(g.to_le_bytes()[0].wrapping_mul(23))
                        .wrapping_add(i.to_le_bytes()[0].wrapping_mul(113))
                        .wrapping_add(7)])
                })
            })
        });
        let act_refs: [[&[i8; 32]; 4]; 4] =
            std::array::from_fn(|t| std::array::from_fn(|g| &acts[t][g]));
        let tile_lanes = dot_ptq1_4x4_lanes(&trits, act_refs);
        let tile_groups = dot_ptq1_4x4_groups(&trits, act_refs);
        for r in 0..4 {
            for t in 0..4 {
                let expected_lanes = dot_ptq1_block_lanes(&rows[r], act_refs[t]);
                let expected_groups = dot_ptq1_block_groups(&rows[r], act_refs[t]);
                assert_eq!(tile_lanes[r][t], expected_lanes);
                assert_eq!(tile_groups[r][t], expected_groups);
            }
        }
    }
}
