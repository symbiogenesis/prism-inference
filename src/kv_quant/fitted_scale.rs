//! Encoding-only scale fitting; packed readers retain their existing arithmetic.

use super::{Bits, HEAD_DIM, scalar};

/// The target is the normalized, rotated binary32 row times the norm scale in
/// binary64. Reconstruction uses actual binary32 scale/codebook products. This
/// metric precedes the rounded inverse transform and does not certify its error.
pub(super) fn select(
    row: &[f32; HEAD_DIM],
    codes: &[u8],
    bits: Bits,
    centroids: &[f32; 16],
    norm_scale: f32,
) -> f32 {
    let mut cross = 0.0_f64;
    let mut energy = 0.0_f64;
    for (index, &value) in row.iter().enumerate() {
        let code = scalar::read_code(codes, bits.bit_width(), index);
        let centroid = f64::from(centroids[code]);
        cross = f64::from(value).mul_add(centroid, cross);
        energy = centroid.mul_add(centroid, energy);
    }
    let Ok(candidate) = scalar::checked_f32(f64::from(norm_scale) * (cross / energy)) else {
        return norm_scale;
    };
    if candidate <= 0.0 || candidate.to_bits() == norm_scale.to_bits() {
        return norm_scale;
    }
    let mut baseline_error = 0.0_f64;
    let mut candidate_error = 0.0_f64;
    for (index, &value) in row.iter().enumerate() {
        let code = scalar::read_code(codes, bits.bit_width(), index);
        let centroid = centroids[code];
        let reconstructed = candidate * centroid;
        if !reconstructed.is_finite() {
            return norm_scale;
        }
        let target = f64::from(value) * f64::from(norm_scale);
        let baseline_delta = target - f64::from(norm_scale * centroid);
        let candidate_delta = target - f64::from(reconstructed);
        baseline_error = baseline_delta.mul_add(baseline_delta, baseline_error);
        candidate_error = candidate_delta.mul_add(candidate_delta, candidate_error);
    }
    if candidate_error < baseline_error {
        candidate
    } else {
        norm_scale
    }
}
