//! Encoding-only scale fitting; packed readers retain their existing arithmetic.

use super::{Bits, HEAD_DIM, SCALE_BYTES, scalar};

const MAX_CODE_BYTES: usize = super::MAX_RECORD_BYTES - SCALE_BYTES;

fn reconstruction_error_f64(
    row: &[f32; HEAD_DIM],
    codes: &[u8],
    bits: Bits,
    centroids: &[f32; 16],
    norm_scale: f32,
    scale: f32,
) -> Option<f64> {
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    let mut error = 0.0_f64;
    for (index, &value) in row.iter().enumerate() {
        let code = scalar::read_code(codes, bits.bit_width(), index);
        let reconstructed = scale * centroids[code];
        if !reconstructed.is_finite() {
            return None;
        }
        let target = f64::from(value) * f64::from(norm_scale);
        let delta = target - f64::from(reconstructed);
        error = delta.mul_add(delta, error);
    }
    error.is_finite().then_some(error)
}

fn least_squares_scale(
    row: &[f32; HEAD_DIM],
    codes: &[u8],
    bits: Bits,
    centroids: &[f32; 16],
    norm_scale: f32,
) -> Option<f32> {
    let mut cross = 0.0_f64;
    let mut energy = 0.0_f64;
    for (index, &value) in row.iter().enumerate() {
        let code = scalar::read_code(codes, bits.bit_width(), index);
        let centroid = f64::from(centroids[code]);
        cross = f64::from(value).mul_add(centroid, cross);
        energy = centroid.mul_add(centroid, energy);
    }
    if !energy.is_finite() || energy <= 0.0 {
        return None;
    }
    let candidate = scalar::checked_f32(f64::from(norm_scale) * (cross / energy)).ok()?;
    (candidate.is_finite() && candidate > 0.0).then_some(candidate)
}

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
    let Some(candidate) = least_squares_scale(row, codes, bits, centroids, norm_scale) else {
        return norm_scale;
    };
    if candidate.to_bits() == norm_scale.to_bits() {
        return norm_scale;
    }
    let Some(baseline_error) =
        reconstruction_error_f64(row, codes, bits, centroids, norm_scale, norm_scale)
    else {
        return norm_scale;
    };
    let Some(candidate_error) =
        reconstruction_error_f64(row, codes, bits, centroids, norm_scale, candidate)
    else {
        return norm_scale;
    };
    if candidate_error < baseline_error {
        candidate
    } else {
        norm_scale
    }
}

/// Two-step alternating least-squares scale and Lloyd-Max code co-optimization
/// for rotated KV rows. Strictly monotonic in binary64 rotated squared error
/// against binary32 reconstruction products; falls back cleanly to single-step
/// scale fitting if arithmetic overflows or fails to improve.
pub(super) fn cooptimize(
    row: &[f32; HEAD_DIM],
    codes: &mut [u8],
    bits: Bits,
    centroids: &[f32; 16],
    norm_scale: f32,
    scalar_only: bool,
) -> f32 {
    let code_bytes = codes.len();
    let levels = bits.levels();
    let bit_width = bits.bit_width();

    let Some(baseline_error) =
        reconstruction_error_f64(row, codes, bits, centroids, norm_scale, norm_scale)
    else {
        return norm_scale;
    };
    let mut best_scale = select(row, codes, bits, centroids, norm_scale);
    let mut best_error = if best_scale.to_bits() == norm_scale.to_bits() {
        baseline_error
    } else {
        reconstruction_error_f64(row, codes, bits, centroids, norm_scale, best_scale)
            .unwrap_or(baseline_error)
    };
    let mut best_codes = [0_u8; MAX_CODE_BYTES];
    best_codes[..code_bytes].copy_from_slice(codes);

    let mut current_scale = best_scale;
    for _ in 0..2 {
        let mut candidate_codes = [0_u8; MAX_CODE_BYTES];
        let mut requant_ok = true;
        for (index, &value) in row.iter().enumerate() {
            let Ok(rescaled) = scalar::checked_f32(
                (f64::from(value) * f64::from(norm_scale)) / f64::from(current_scale),
            ) else {
                requant_ok = false;
                break;
            };
            let code = if scalar_only {
                scalar::nearest(rescaled, centroids, levels)
            } else {
                scalar::nearest_ordered(rescaled, centroids, levels)
            };
            if !(current_scale * centroids[usize::from(code)]).is_finite() {
                requant_ok = false;
                break;
            }
            scalar::write_code(&mut candidate_codes[..code_bytes], bit_width, index, code);
        }
        if !requant_ok {
            break;
        }
        if let Some(error) = reconstruction_error_f64(
            row,
            &candidate_codes[..code_bytes],
            bits,
            centroids,
            norm_scale,
            current_scale,
        ) && error < best_error
        {
            best_error = error;
            best_scale = current_scale;
            best_codes[..code_bytes].copy_from_slice(&candidate_codes[..code_bytes]);
        }
        if let Some(refit_scale) = least_squares_scale(
            row,
            &candidate_codes[..code_bytes],
            bits,
            centroids,
            norm_scale,
        ) && let Some(error) = reconstruction_error_f64(
            row,
            &candidate_codes[..code_bytes],
            bits,
            centroids,
            norm_scale,
            refit_scale,
        ) && error < best_error
        {
            best_error = error;
            best_scale = refit_scale;
            best_codes[..code_bytes].copy_from_slice(&candidate_codes[..code_bytes]);
        }
        if best_scale.to_bits() == current_scale.to_bits()
            && best_codes[..code_bytes] == candidate_codes[..code_bytes]
        {
            break;
        }
        current_scale = best_scale;
    }

    codes.copy_from_slice(&best_codes[..code_bytes]);
    best_scale
}
