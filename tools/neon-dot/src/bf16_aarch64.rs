//! Pair independent rows in binary64 lanes without changing column order.

use core::arch::aarch64::{
    float32x4_t, float64x2_t, uint16x8_t, vaddq_f64, vandq_u16, vcvt_f64_f32, vcvt_high_f64_f32,
    vdupq_n_f64, vdupq_n_u16, vget_low_f32, vld1q_f32, vld1q_u8, vmaxq_u16, vmaxvq_u16, vmulq_f32,
    vreinterpretq_f32_u16, vreinterpretq_u16_u8, vst1q_f64, vzip1q_f32, vzip1q_u16, vzip2q_f32,
    vzip2q_u16,
};

#[inline]
pub(super) fn dot_rows<const ROWS: usize>(
    rows: [&[u8]; ROWS],
    input: &[f32],
) -> Option<[f32; ROWS]> {
    // SAFETY: the safe parent checks four/eight nonempty rows and exact byte
    // extents. This module is compiled only for little-endian baseline NEON.
    unsafe { paired::<ROWS>(rows, input) }
}

#[inline]
#[target_feature(enable = "neon")]
fn add_columns(mut sum: float64x2_t, first: float32x4_t, second: float32x4_t) -> float64x2_t {
    let low = vzip1q_f32(first, second);
    let high = vzip2q_f32(first, second);
    // A vector lane belongs to one row for the complete reduction. Each
    // column keeps its separate rounded product, conversion and addition.
    sum = vaddq_f64(sum, vcvt_f64_f32(vget_low_f32(low)));
    sum = vaddq_f64(sum, vcvt_high_f64_f32(low));
    sum = vaddq_f64(sum, vcvt_f64_f32(vget_low_f32(high)));
    vaddq_f64(sum, vcvt_high_f64_f32(high))
}

#[inline]
#[target_feature(enable = "neon")]
fn products(raw: uint16x8_t, first: float32x4_t, second: float32x4_t) -> [float32x4_t; 2] {
    let zero = vdupq_n_u16(0);
    let low = vreinterpretq_f32_u16(vzip1q_u16(zero, raw));
    let high = vreinterpretq_f32_u16(vzip2q_u16(zero, raw));
    [vmulq_f32(low, first), vmulq_f32(high, second)]
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Each original ordered binary64 sum rounds once to the binary32 result."
)]
#[target_feature(enable = "neon")]
fn paired<const ROWS: usize>(rows: [&[u8]; ROWS], input: &[f32]) -> Option<[f32; ROWS]> {
    let mut sums = [vdupq_n_f64(-0.0); 4];
    let mut weight_magnitude = vdupq_n_u16(0);
    let magnitude_mask = vdupq_n_u16(0x7fff);
    let (chunks, tail) = input.as_chunks::<8>();
    for (column, activation) in chunks.iter().enumerate() {
        // SAFETY: each eight-element activation array contains both complete
        // four-lane unaligned loads; the enclosing target provides NEON.
        let first = unsafe { vld1q_f32(activation.as_ptr()) };
        // SAFETY: the high four lanes lie inside the same immutable array.
        let second = unsafe { vld1q_f32(activation[4..].as_ptr()) };
        for (pair, sum) in sums[..ROWS / 2].iter_mut().enumerate() {
            let offset = column * 16;
            let first_row = &rows[pair * 2][offset..offset + 16];
            let second_row = &rows[pair * 2 + 1][offset..offset + 16];
            // SAFETY: checked geometry and the eight-column loop supply both
            // sixteen-byte immutable ranges. NEON allows unaligned loads.
            let first_bytes = unsafe { vld1q_u8(first_row.as_ptr()) };
            // SAFETY: the second range has the same checked complete extent.
            let second_bytes = unsafe { vld1q_u8(second_row.as_ptr()) };
            let first_payload = vreinterpretq_u16_u8(first_bytes);
            let second_payload = vreinterpretq_u16_u8(second_bytes);
            weight_magnitude = vmaxq_u16(
                weight_magnitude,
                vmaxq_u16(
                    vandq_u16(first_payload, magnitude_mask),
                    vandq_u16(second_payload, magnitude_mask),
                ),
            );
            let a = products(first_payload, first, second);
            let b = products(second_payload, first, second);
            *sum = add_columns(*sum, a[0], b[0]);
            *sum = add_columns(*sum, a[1], b[1]);
        }
    }
    if vmaxvq_u16(weight_magnitude) >= 0x7f80 {
        return None;
    }
    let mut output = [0.0; ROWS];
    let offset = chunks.len() * 16;
    for (pair, sum) in sums[..ROWS / 2].iter().enumerate() {
        let mut values = [0.0; 2];
        // SAFETY: the destination is a unique initialized two-element array;
        // the vector has one complete ordered sum for each independent row.
        unsafe { vst1q_f64(values.as_mut_ptr(), *sum) };
        for (lane, value) in values.iter_mut().enumerate() {
            let row = rows[pair * 2 + lane];
            for (bytes, &activation) in row[offset..].as_chunks::<2>().0.iter().zip(tail) {
                let weight = f32::from_bits(u32::from(u16::from_le_bytes(*bytes)) << 16);
                if !weight.is_finite() {
                    return None;
                }
                *value += f64::from(weight * activation);
            }
            let value = *value as f32;
            if !value.is_finite() {
                return None;
            }
            output[pair * 2 + lane] = value;
        }
    }
    Some(output)
}
