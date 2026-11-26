//! Encoding-only scale fitting; packed readers retain their existing arithmetic.

use super::{Bits, HEAD_DIM, scalar};

fn unpack_codes(codes: &[u8], bit_width: u8) -> [u8; HEAD_DIM] {
    std::array::from_fn(|index| {
        u8::try_from(scalar::read_code(codes, bit_width, index)).unwrap_or(0)
    })
}

fn pack_codes(unpacked: &[u8; HEAD_DIM], bit_width: u8, codes: &mut [u8]) {
    codes.fill(0);
    for (index, &code) in unpacked.iter().enumerate() {
        scalar::write_code(codes, bit_width, index, code);
    }
}

fn precompute_targets(row: &[f32; HEAD_DIM], norm_scale: f32) -> [f64; HEAD_DIM] {
    let norm_scale_f64 = f64::from(norm_scale);
    std::array::from_fn(|index| f64::from(row[index]) * norm_scale_f64)
}

fn reconstruction_error_unpacked(
    targets: &[f64; HEAD_DIM],
    unpacked_codes: &[u8; HEAD_DIM],
    centroids: &[f32; 16],
    scale: f32,
) -> Option<f64> {
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    let mut error = 0.0_f64;
    for (&target, &code) in targets.iter().zip(unpacked_codes) {
        let reconstructed = scale * centroids[usize::from(code)];
        if !reconstructed.is_finite() {
            return None;
        }
        let delta = target - f64::from(reconstructed);
        error = delta.mul_add(delta, error);
    }
    error.is_finite().then_some(error)
}

fn least_squares_scale_unpacked(
    row: &[f32; HEAD_DIM],
    unpacked_codes: &[u8; HEAD_DIM],
    centroids: &[f32; 16],
    norm_scale: f32,
) -> Option<f32> {
    let mut cross = 0.0_f64;
    let mut energy = 0.0_f64;
    for (&value, &code) in row.iter().zip(unpacked_codes) {
        let centroid = f64::from(centroids[usize::from(code)]);
        cross = f64::from(value).mul_add(centroid, cross);
        energy = centroid.mul_add(centroid, energy);
    }
    if !energy.is_finite() || energy <= 0.0 {
        return None;
    }
    let candidate = scalar::checked_f32(f64::from(norm_scale) * (cross / energy)).ok()?;
    (candidate.is_finite() && candidate > 0.0).then_some(candidate)
}

fn select_from_baseline(
    row: &[f32; HEAD_DIM],
    targets: &[f64; HEAD_DIM],
    unpacked_codes: &[u8; HEAD_DIM],
    centroids: &[f32; 16],
    norm_scale: f32,
    baseline_error: f64,
) -> (f32, f64) {
    let Some(candidate) = least_squares_scale_unpacked(row, unpacked_codes, centroids, norm_scale)
    else {
        return (norm_scale, baseline_error);
    };
    if candidate.to_bits() == norm_scale.to_bits() {
        return (norm_scale, baseline_error);
    }
    let Some(candidate_error) =
        reconstruction_error_unpacked(targets, unpacked_codes, centroids, candidate)
    else {
        return (norm_scale, baseline_error);
    };
    if candidate_error < baseline_error {
        (candidate, candidate_error)
    } else {
        (norm_scale, baseline_error)
    }
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
    let unpacked_codes = unpack_codes(codes, bits.bit_width());
    let Some(candidate) = least_squares_scale_unpacked(row, &unpacked_codes, centroids, norm_scale)
    else {
        return norm_scale;
    };
    if candidate.to_bits() == norm_scale.to_bits() {
        return norm_scale;
    }
    let targets = precompute_targets(row, norm_scale);
    let Some(baseline_error) =
        reconstruction_error_unpacked(&targets, &unpacked_codes, centroids, norm_scale)
    else {
        return norm_scale;
    };
    let Some(candidate_error) =
        reconstruction_error_unpacked(&targets, &unpacked_codes, centroids, candidate)
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
    let levels = bits.levels();
    let bit_width = bits.bit_width();
    let initial_codes = unpack_codes(codes, bit_width);
    let targets = precompute_targets(row, norm_scale);

    let Some(baseline_error) =
        reconstruction_error_unpacked(&targets, &initial_codes, centroids, norm_scale)
    else {
        return norm_scale;
    };
    let (mut best_scale, mut best_error) = select_from_baseline(
        row,
        &targets,
        &initial_codes,
        centroids,
        norm_scale,
        baseline_error,
    );
    let mut best_codes = initial_codes;
    let mut codes_changed = false;

    let mut current_scale = best_scale;
    for _ in 0..2 {
        let mut candidate_codes = [0_u8; HEAD_DIM];
        let mut requant_ok = true;
        let current_scale_f64 = f64::from(current_scale);
        for (index, &target) in targets.iter().enumerate() {
            let Ok(rescaled) = scalar::checked_f32(target / current_scale_f64) else {
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
            candidate_codes[index] = code;
        }
        if !requant_ok {
            break;
        }
        if let Some(error) =
            reconstruction_error_unpacked(&targets, &candidate_codes, centroids, current_scale)
            && error < best_error
        {
            best_error = error;
            best_scale = current_scale;
            best_codes = candidate_codes;
            codes_changed = true;
        }
        if let Some(refit_scale) =
            least_squares_scale_unpacked(row, &candidate_codes, centroids, norm_scale)
            && let Some(error) =
                reconstruction_error_unpacked(&targets, &candidate_codes, centroids, refit_scale)
            && error < best_error
        {
            best_error = error;
            best_scale = refit_scale;
            best_codes = candidate_codes;
            codes_changed = true;
        }
        if best_scale.to_bits() == current_scale.to_bits() && best_codes == candidate_codes {
            break;
        }
        current_scale = best_scale;
    }

    if codes_changed {
        pack_codes(&best_codes, bit_width, codes);
    }
    best_scale
}
