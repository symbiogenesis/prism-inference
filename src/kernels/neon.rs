//! Experimental baseline NEON integer arithmetic for native PQ2/PTQ1 blocks.

use fearless_simd::{Neon, i8x16, i32x4, prelude::*, u8x16, u16x8, u32x4, u64x2};

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
    let mut total = i32x4::splat(neon, 0);
    for (bytes, values) in packed
        .as_chunks::<16>()
        .0
        .iter()
        .zip(activations.as_chunks::<64>().0)
    {
        let decoded = decode(u8x16::from_slice(neon, bytes));
        for (weights, activations) in decoded.into_iter().zip(values.as_chunks::<16>().0) {
            total += dot16(weights, i8x16::from_slice(neon, activations));
        }
    }
    total.reduce_sum()
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

#[cfg(test)]
mod tests {
    use super::{dot, dot_groups, dot_ptq1_groups};

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
