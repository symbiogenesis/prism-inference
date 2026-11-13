//! Canonical scalar arithmetic and byte packing, independent of native helpers.

use super::{HEAD_DIM, finite_values};
use crate::{Result, invalid};

pub(super) const fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut word = *state;
    word = (word ^ (word >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    word = (word ^ (word >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    word ^ (word >> 31)
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "The wire format explicitly rounds finite binary64 arithmetic to binary32, rejecting overflow."
)]
pub(super) fn checked_f32(value: f64) -> Result<f32> {
    let rounded = value as f32;
    if !rounded.is_finite() {
        return Err(invalid("KV codec binary32 arithmetic overflow"));
    }
    Ok(rounded)
}

pub(super) fn hadamard(row: &mut [f32; HEAD_DIM]) -> Result<()> {
    let mut next = *row;
    let mut half = 1;
    while half < HEAD_DIM {
        for block in next.chunks_exact_mut(half * 2) {
            let (left, right) = block.split_at_mut(half);
            for (left, right) in left.iter_mut().zip(right) {
                let first = *left;
                let second = *right;
                *left = first + second;
                *right = first - second;
            }
        }
        half *= 2;
    }
    for value in &mut next {
        *value *= 1.0 / 16.0;
    }
    finite_values(&next)?;
    *row = next;
    Ok(())
}

pub(super) fn nearest(value: f32, centroids: &[f32; 16], levels: u8) -> u8 {
    let mut code = 0_u8;
    let mut distance = (value - centroids[0]).abs();
    for candidate in 1..levels {
        let candidate_distance = (value - centroids[usize::from(candidate)]).abs();
        if candidate_distance < distance {
            code = candidate;
            distance = candidate_distance;
        }
    }
    code
}

pub(super) fn write_code(bytes: &mut [u8], bits: u8, index: usize, code: u8) {
    let offset = index * usize::from(bits);
    let byte = offset / 8;
    let shift = offset % 8;
    bytes[byte] |= code << shift;
    if shift + usize::from(bits) > 8 {
        bytes[byte + 1] |= code >> (8 - shift);
    }
}

pub(super) fn read_code(bytes: &[u8], bits: u8, index: usize) -> usize {
    let offset = index * usize::from(bits);
    let byte = offset / 8;
    let shift = offset % 8;
    let mut code = bytes[byte] >> shift;
    if shift + usize::from(bits) > 8 {
        code |= bytes[byte + 1] << (8 - shift);
    }
    usize::from(code & ((1_u8 << bits) - 1))
}

#[expect(
    clippy::suboptimal_flops,
    reason = "The canonical scalar/native contract requires separate binary64 products and additions with a fixed four-lane reduction."
)]
pub(super) fn dot(
    codes: &[u8],
    bits: u8,
    centroids: &[f32; 16],
    query: &[f32; HEAD_DIM],
) -> Result<f32> {
    let mut lanes = [0.0_f64; 4];
    for (index, &query) in query.iter().enumerate() {
        let centroid = centroids[read_code(codes, bits, index)];
        lanes[index % 4] += f64::from(query) * f64::from(centroid);
    }
    checked_f32(((lanes[0] + lanes[1]) + lanes[2]) + lanes[3])
}

pub(super) fn accumulate(
    codes: &[u8],
    bits: u8,
    centroids: &[f32; 16],
    scale: f32,
    out: &mut [f32; HEAD_DIM],
) -> Result<()> {
    let mut next = *out;
    accumulate_in_place(codes, bits, centroids, scale, &mut next)?;
    *out = next;
    Ok(())
}

/// Only scan-owned provisional sums use the nontransactional inner update.
pub(super) fn accumulate_in_place(
    codes: &[u8],
    bits: u8,
    centroids: &[f32; 16],
    scale: f32,
    out: &mut [f32; HEAD_DIM],
) -> Result<()> {
    for (index, value) in out.iter_mut().enumerate() {
        let increment = scale * centroids[read_code(codes, bits, index)];
        *value += increment;
    }
    finite_values(out)
}
