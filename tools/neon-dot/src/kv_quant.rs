//! Bounded 256-channel rotated scalar KV consumption.
//!
//! Three-bit codes occupy 96 bytes and four-bit codes occupy 128 bytes. Codes
//! are contiguous, least-significant-bit first; record scales are supplied by
//! callers separately. Consumers decode only four coordinates at a time and
//! never construct an expanded cache. Both Snapdragon X1 and X2 use this same
//! runtime-detected 128-bit NEON path; no optional ISA or speed claim is assumed.

const DIMENSION: usize = 256;

/// Computes a dot against one code-only rotated KV row, excluding its scale.
///
/// Rejects unsupported widths, incorrect byte lengths, nonfinite centroids or
/// query coordinates, and a result outside finite binary32. Four binary64 lane
/// sums consume indices modulo four; the final reduction is left-associated.
/// Portable and NEON implementations use the same separate multiply/add order.
#[must_use]
pub fn dot(
    packed: &[u8],
    bits: u8,
    centroids: &[f32; 16],
    query: &[f32; DIMENSION],
) -> Option<f32> {
    if !valid(packed, bits, centroids) || !query.iter().all(|value| value.is_finite()) {
        return None;
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return finite_binary32(native::dot(packed, bits, centroids, query));
    }
    finite_binary32(portable_dot(packed, bits, centroids, query))
}

/// Adds a scaled code-only rotated value row into the supplied accumulator.
///
/// Every multiply and addition rounds separately to binary32. Invalid input or
/// nonfinite arithmetic returns false with the accumulator entirely unchanged.
/// This is a per-row operation; no complete cache is expanded or retained.
#[must_use]
pub fn accumulate(
    packed: &[u8],
    bits: u8,
    centroids: &[f32; 16],
    scale: f32,
    output: &mut [f32; DIMENSION],
) -> bool {
    if !valid(packed, bits, centroids)
        || !scale.is_finite()
        || !output.iter().all(|value| value.is_finite())
    {
        return false;
    }
    let mut staged = *output;
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        native::accumulate(packed, bits, centroids, scale, &mut staged);
    } else {
        portable_accumulate(packed, bits, centroids, scale, &mut staged);
    }
    #[cfg(not(target_arch = "aarch64"))]
    portable_accumulate(packed, bits, centroids, scale, &mut staged);
    if !staged.iter().all(|value| value.is_finite()) {
        return false;
    }
    *output = staged;
    true
}

/// Applies a normalized 256-coordinate Walsh-Hadamard transform in place.
///
/// Butterflies visit increasing half-widths from one through 128, followed by
/// exact power-of-two normalization by 1/16. Both backends preserve that order.
/// Nonfinite input or arithmetic returns false without changing any coordinate.
#[must_use]
pub fn hadamard(values: &mut [f32; DIMENSION]) -> bool {
    if !values.iter().all(|value| value.is_finite()) {
        return false;
    }
    let mut staged = *values;
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        native::hadamard(&mut staged);
    } else {
        portable_hadamard(&mut staged);
    }
    #[cfg(not(target_arch = "aarch64"))]
    portable_hadamard(&mut staged);
    if !staged.iter().all(|value| value.is_finite()) {
        return false;
    }
    *values = staged;
    true
}

fn valid(packed: &[u8], bits: u8, centroids: &[f32; 16]) -> bool {
    matches!((bits, packed.len()), (3, 96) | (4, 128))
        && centroids.iter().all(|value| value.is_finite())
}

fn coordinate(packed: &[u8], bits: u8, centroids: &[f32; 16], index: usize) -> f32 {
    let bit = index * usize::from(bits);
    let byte = bit / 8;
    let shift = bit % 8;
    let low = u16::from(packed[byte]);
    let high = if shift + usize::from(bits) > 8 {
        u16::from(packed[byte + 1]) << 8
    } else {
        0
    };
    let code = ((low | high) >> shift) & ((1_u16 << bits) - 1);
    centroids[usize::from(code)]
}

fn tile(packed: &[u8], bits: u8, centroids: &[f32; 16], index: usize) -> [f32; 4] {
    std::array::from_fn(|lane| coordinate(packed, bits, centroids, index + lane))
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::as_conversions,
    reason = "Binary64 accumulation is rounded once to binary32 and its finite range is checked immediately."
)]
fn finite_binary32(value: f64) -> Option<f32> {
    let value = value as f32;
    value.is_finite().then_some(value)
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Separate multiplication and addition preserve the deterministic scalar/NEON arithmetic contract."
)]
fn portable_dot(packed: &[u8], bits: u8, centroids: &[f32; 16], query: &[f32; DIMENSION]) -> f64 {
    let mut sums = [0.0_f64; 4];
    for (index, query) in query.as_chunks::<4>().0.iter().enumerate() {
        let values = tile(packed, bits, centroids, index * 4);
        for ((sum, value), query) in sums.iter_mut().zip(values).zip(query) {
            *sum += f64::from(value) * f64::from(*query);
        }
    }
    reduce(sums)
}

fn reduce(sums: [f64; 4]) -> f64 {
    ((sums[0] + sums[1]) + sums[2]) + sums[3]
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Value accumulation intentionally retains separate binary32 multiply/add rounding."
)]
fn portable_accumulate(
    packed: &[u8],
    bits: u8,
    centroids: &[f32; 16],
    scale: f32,
    output: &mut [f32; DIMENSION],
) {
    for (index, output) in output.iter_mut().enumerate() {
        *output += scale * coordinate(packed, bits, centroids, index);
    }
}

fn butterfly(values: &mut [f32; DIMENSION], half: usize) {
    for block in values.chunks_exact_mut(half * 2) {
        let (left, right) = block.split_at_mut(half);
        for (left, right) in left.iter_mut().zip(right) {
            let a = *left;
            let b = *right;
            *left = a + b;
            *right = a - b;
        }
    }
}

fn portable_hadamard(values: &mut [f32; DIMENSION]) {
    let mut half = 1;
    while half < DIMENSION {
        butterfly(values, half);
        half *= 2;
    }
    for value in values {
        *value *= 0.0625;
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(
    unsafe_code,
    reason = "Private baseline-NEON intrinsics use checked fixed-size tiles behind runtime detection and failure-atomic safe public APIs."
)]
mod native {
    use core::arch::aarch64::{
        vaddq_f32, vaddq_f64, vcvt_f64_f32, vcvt_high_f64_f32, vdupq_n_f32, vdupq_n_f64,
        vget_low_f32, vld1q_f32, vmulq_f32, vmulq_f64, vst1q_f32, vst1q_f64, vsubq_f32,
    };

    use super::DIMENSION;

    pub(super) fn dot(
        packed: &[u8],
        bits: u8,
        centroids: &[f32; 16],
        query: &[f32; DIMENSION],
    ) -> f64 {
        // SAFETY: the safe public caller proves baseline NEON at runtime and
        // validates exactly 256 codes and finite fixed-array inputs.
        unsafe { dot_neon(packed, bits, centroids, query) }
    }

    #[target_feature(enable = "neon")]
    fn dot_neon(packed: &[u8], bits: u8, centroids: &[f32; 16], query: &[f32; DIMENSION]) -> f64 {
        let mut low_sum = vdupq_n_f64(0.0);
        let mut high_sum = vdupq_n_f64(0.0);
        for (index, query) in query.as_chunks::<4>().0.iter().enumerate() {
            let values = super::tile(packed, bits, centroids, index * 4);
            // SAFETY: values contains four initialized coordinates and NEON
            // permits an unaligned load; the temporary outlives the load.
            let values = unsafe { vld1q_f32(values.as_ptr()) };
            // SAFETY: the chunk contains exactly four initialized query values.
            let query = unsafe { vld1q_f32(query.as_ptr()) };
            let low = vmulq_f64(
                vcvt_f64_f32(vget_low_f32(values)),
                vcvt_f64_f32(vget_low_f32(query)),
            );
            let high = vmulq_f64(vcvt_high_f64_f32(values), vcvt_high_f64_f32(query));
            low_sum = vaddq_f64(low_sum, low);
            high_sum = vaddq_f64(high_sum, high);
        }
        let mut low = [0.0; 2];
        let mut high = [0.0; 2];
        // SAFETY: low is a unique initialized two-element output array.
        unsafe { vst1q_f64(low.as_mut_ptr(), low_sum) };
        // SAFETY: high is a disjoint unique two-element output array.
        unsafe { vst1q_f64(high.as_mut_ptr(), high_sum) };
        super::reduce([low[0], low[1], high[0], high[1]])
    }

    pub(super) fn accumulate(
        packed: &[u8],
        bits: u8,
        centroids: &[f32; 16],
        scale: f32,
        output: &mut [f32; DIMENSION],
    ) {
        // SAFETY: caller detects baseline NEON, validates code length and
        // finite inputs, and supplies a unique 256-coordinate staging array.
        unsafe { accumulate_neon(packed, bits, centroids, scale, output) };
    }

    #[target_feature(enable = "neon")]
    fn accumulate_neon(
        packed: &[u8],
        bits: u8,
        centroids: &[f32; 16],
        scale: f32,
        output: &mut [f32; DIMENSION],
    ) {
        let scale = vdupq_n_f32(scale);
        for (index, output) in output.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let values = super::tile(packed, bits, centroids, index * 4);
            // SAFETY: values is a valid initialized four-element array.
            let values = unsafe { vld1q_f32(values.as_ptr()) };
            // SAFETY: output is a unique four-element chunk of the staging row.
            let previous = unsafe { vld1q_f32(output.as_ptr()) };
            let product = vmulq_f32(scale, values);
            let result = vaddq_f32(previous, product);
            // SAFETY: the same unique four-element chunk covers the store.
            unsafe { vst1q_f32(output.as_mut_ptr(), result) };
        }
    }

    pub(super) fn hadamard(values: &mut [f32; DIMENSION]) {
        // SAFETY: caller detects baseline NEON and supplies one unique fixed
        // 256-coordinate staging array, retained entirely within this call.
        unsafe { hadamard_neon(values) };
    }

    #[target_feature(enable = "neon")]
    fn hadamard_neon(values: &mut [f32; DIMENSION]) {
        super::butterfly(values, 1);
        super::butterfly(values, 2);
        let mut half = 4;
        while half < DIMENSION {
            for block in values.chunks_exact_mut(half * 2) {
                let (left, right) = block.split_at_mut(half);
                for (left, right) in left
                    .as_chunks_mut::<4>()
                    .0
                    .iter_mut()
                    .zip(right.as_chunks_mut::<4>().0)
                {
                    // SAFETY: both disjoint chunks contain four initialized
                    // values and remain valid until their corresponding stores.
                    let a = unsafe { vld1q_f32(left.as_ptr()) };
                    // SAFETY: right is the disjoint valid four-element chunk.
                    let b = unsafe { vld1q_f32(right.as_ptr()) };
                    let sum = vaddq_f32(a, b);
                    let difference = vsubq_f32(a, b);
                    // SAFETY: left is a unique valid four-element output chunk.
                    unsafe { vst1q_f32(left.as_mut_ptr(), sum) };
                    // SAFETY: right is disjoint and covers four output values.
                    unsafe { vst1q_f32(right.as_mut_ptr(), difference) };
                }
            }
            half *= 2;
        }
        let normalization = vdupq_n_f32(0.0625);
        for chunk in values.as_chunks_mut::<4>().0 {
            // SAFETY: the fixed chunk contains four initialized coordinates.
            let value = unsafe { vld1q_f32(chunk.as_ptr()) };
            let scaled = vmulq_f32(value, normalization);
            // SAFETY: chunk is unique and has room for all four result values.
            unsafe { vst1q_f32(chunk.as_mut_ptr(), scaled) };
        }
    }
}

#[cfg(test)]
#[expect(
    clippy::suboptimal_flops,
    reason = "Independent arithmetic oracles preserve the specified separate multiplication and addition rounding."
)]
mod tests {
    use super::{DIMENSION, accumulate, dot, hadamard};

    // Independent bit-at-a-time encoder: no production decode/packing helpers.
    fn packed(codes: &[u8; DIMENSION], bits: u8) -> Vec<u8> {
        let mut bytes = vec![0_u8; DIMENSION * usize::from(bits) / 8];
        for (index, code) in codes.iter().enumerate() {
            for bit in 0..bits {
                let offset = index * usize::from(bits) + usize::from(bit);
                bytes[offset / 8] |= ((code >> bit) & 1) << (offset % 8);
            }
        }
        bytes
    }

    fn codebook() -> [f32; 16] {
        std::array::from_fn(|index| f32::from(index.to_le_bytes()[0]) - 7.5)
    }

    fn codes(bits: u8, offset: u8) -> [u8; DIMENSION] {
        std::array::from_fn(|index: usize| {
            index.to_le_bytes()[0].wrapping_add(offset) & ((1 << bits) - 1)
        })
    }

    #[test]
    fn all_codes_cross_byte_boundaries_and_native_scalar_consumption_agree() {
        let centroids = codebook();
        let query = std::array::from_fn(|index: usize| {
            (f32::from(index.to_le_bytes()[0]) - 128.0) * 0.03125
        });
        for bits in [3, 4] {
            for offset in 0..(1 << bits) {
                let codes = codes(bits, offset);
                let bytes = packed(&codes, bits);
                let expected = codes.iter().zip(query).fold(0.0_f64, |sum, (code, q)| {
                    sum + f64::from(centroids[usize::from(*code)]) * f64::from(q)
                });
                assert_eq!(
                    dot(&bytes, bits, &centroids, &query).map(f32::to_bits),
                    super::finite_binary32(expected).map(f32::to_bits)
                );
                assert_eq!(
                    dot(&bytes, bits, &centroids, &query).map(f32::to_bits),
                    super::finite_binary32(super::portable_dot(&bytes, bits, &centroids, &query))
                        .map(f32::to_bits)
                );
                let initial = [0.25; DIMENSION];
                let mut output = initial;
                assert!(accumulate(&bytes, bits, &centroids, -0.125, &mut output));
                let expected: [f32; DIMENSION] = std::array::from_fn(|index| {
                    initial[index] - 0.125 * centroids[usize::from(codes[index])]
                });
                assert_eq!(output.map(f32::to_bits), expected.map(f32::to_bits));
                let mut scalar = initial;
                super::portable_accumulate(&bytes, bits, &centroids, -0.125, &mut scalar);
                assert_eq!(output.map(f32::to_bits), scalar.map(f32::to_bits));
            }
        }
    }

    #[test]
    fn repeated_zero_and_finite_extremes_preserve_failure_atomicity() {
        for bits in [3, 4] {
            let bytes = packed(&[0; DIMENSION], bits);
            assert_eq!(
                dot(&bytes, bits, &[0.0; 16], &[f32::MAX; DIMENSION]).map(f32::to_bits),
                Some(0.0_f32.to_bits())
            );
            let mut output = [1.0; DIMENSION];
            assert!(accumulate(&bytes, bits, &[1.0; 16], 0.25, &mut output));
            assert!(accumulate(&bytes, bits, &[1.0; 16], 0.25, &mut output));
            assert_eq!(output.map(f32::to_bits), [1.5_f32.to_bits(); DIMENSION]);
            let before = output.map(f32::to_bits);
            assert!(!accumulate(&bytes, bits, &[f32::MAX; 16], 2.0, &mut output));
            assert_eq!(output.map(f32::to_bits), before);
            assert_eq!(
                dot(&bytes, bits, &[f32::MAX; 16], &[f32::MAX; DIMENSION]),
                None
            );
        }
    }

    #[test]
    fn invalid_width_lengths_and_nonfinite_inputs_never_mutate() {
        let centroids = codebook();
        for (bits, length) in [
            (0, 0),
            (2, 64),
            (3, 95),
            (3, 97),
            (4, 127),
            (4, 129),
            (8, 256),
        ] {
            let bytes = vec![0; length];
            let mut output = [1.0; DIMENSION];
            assert!(dot(&bytes, bits, &centroids, &output).is_none());
            assert!(!accumulate(&bytes, bits, &centroids, 1.0, &mut output));
            assert_eq!(output.map(f32::to_bits), [1.0_f32.to_bits(); DIMENSION]);
        }
        let bytes = packed(&codes(3, 0), 3);
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut book = centroids;
            book[15] = invalid;
            let mut output = [1.0; DIMENSION];
            assert!(dot(&bytes, 3, &book, &output).is_none());
            assert!(!accumulate(&bytes, 3, &book, 1.0, &mut output));
            assert!(!accumulate(&bytes, 3, &centroids, invalid, &mut output));
            assert_eq!(output.map(f32::to_bits), [1.0_f32.to_bits(); DIMENSION]);
            output[255] = invalid;
            let before = output.map(f32::to_bits);
            assert!(dot(&bytes, 3, &centroids, &output).is_none());
            assert!(!accumulate(&bytes, 3, &centroids, 1.0, &mut output));
            assert!(!hadamard(&mut output));
            assert_eq!(output.map(f32::to_bits), before);
        }
    }

    #[test]
    fn hadamard_matches_independent_matrix_and_native_scalar_rounding() {
        let original =
            std::array::from_fn(|index: usize| (f32::from(index.to_le_bytes()[0]) - 127.0) * 0.125);
        let expected: [f32; DIMENSION] = std::array::from_fn(|row: usize| {
            original
                .iter()
                .enumerate()
                .fold(0.0, |sum, (column, value)| {
                    if (row & column).count_ones().is_multiple_of(2) {
                        sum + *value
                    } else {
                        sum - *value
                    }
                })
                * 0.0625
        });
        let mut transformed = original;
        assert!(hadamard(&mut transformed));
        assert_eq!(transformed.map(f32::to_bits), expected.map(f32::to_bits));
        let mut scalar = original;
        super::portable_hadamard(&mut scalar);
        assert_eq!(transformed.map(f32::to_bits), scalar.map(f32::to_bits));
        assert!(hadamard(&mut transformed));
        assert_eq!(transformed.map(f32::to_bits), original.map(f32::to_bits));
        let mut overflow = [f32::MAX; DIMENSION];
        assert!(!hadamard(&mut overflow));
        assert_eq!(overflow.map(f32::to_bits), [f32::MAX.to_bits(); DIMENSION]);
    }
}
