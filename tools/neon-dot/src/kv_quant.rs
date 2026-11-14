//! Bounded 256-channel rotated scalar KV consumption.
//!
//! Three-bit codes occupy 96 bytes and four-bit codes occupy 128 bytes. Codes
//! are contiguous, least-significant-bit first; record scales are supplied by
//! callers separately. Consumers decode only four coordinates at a time and
//! never construct an expanded cache. Both Snapdragon X1 and X2 use this same
//! runtime-detected 128-bit NEON path; no optional ISA or speed claim is assumed.

const DIMENSION: usize = 256;

/// An immutable finite query and codebook checked once for a packed-key scan.
///
/// The borrows prevent mutation of either input while the proof is in use.
/// Packed-row extents and finite dot results remain checked on every call.
pub struct PreparedQuery<'a> {
    bits: u8,
    centroids: &'a [f32; 16],
    query: &'a [f32; DIMENSION],
    #[cfg(target_arch = "aarch64")]
    neon: bool,
}

impl<'a> PreparedQuery<'a> {
    /// Checks a query and codebook without allocating or copying either row.
    ///
    /// Returns `None` for unsupported precision or nonfinite input. Optional
    /// native dispatch is selected once and retained only for this borrow.
    #[must_use]
    pub fn new(bits: u8, centroids: &'a [f32; 16], query: &'a [f32; DIMENSION]) -> Option<Self> {
        if !matches!(bits, 3 | 4)
            || !centroids.iter().all(|value| value.is_finite())
            || !query.iter().all(|value| value.is_finite())
        {
            return None;
        }
        Some(Self {
            bits,
            centroids,
            query,
            #[cfg(target_arch = "aarch64")]
            neon: std::arch::is_aarch64_feature_detected!("neon"),
        })
    }

    /// Computes one code-only dot with the ordinary four-lane arithmetic.
    ///
    /// Returns `None` for an incorrect code extent or nonfinite binary32 result.
    /// The current packed bytes are consumed on every call; pointer identity
    /// never permits stale row reuse.
    #[must_use]
    pub fn dot(&self, packed: &[u8]) -> Option<f32> {
        if packed.len() != DIMENSION * usize::from(self.bits) / 8 {
            return None;
        }
        #[cfg(target_arch = "aarch64")]
        if self.neon {
            return finite_binary32(native::dot(packed, self.bits, self.centroids, self.query));
        }
        finite_binary32(portable_dot(packed, self.bits, self.centroids, self.query))
    }
}

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
    PreparedQuery::new(bits, centroids, query)?.dot(packed)
}

/// An exclusively owned running value sum for one packed-attention scan.
///
/// The codebook and native dispatch are checked once. Each row retains its
/// extent, scale and finite-result checks, with separate binary32 multiply/add
/// rounding. A failed append permanently prevents publication; no partially
/// accumulated or nonfinite row can be retrieved through this safe API.
pub struct ValueAccumulator<'a> {
    bits: u8,
    centroids: &'a [f32; 16],
    output: [f32; DIMENSION],
    failed: bool,
    #[cfg(target_arch = "aarch64")]
    neon: bool,
}

impl<'a> ValueAccumulator<'a> {
    /// Starts a zero sum, rejecting unsupported precision or nonfinite centroids.
    #[must_use]
    pub fn new(bits: u8, centroids: &'a [f32; 16]) -> Option<Self> {
        if !matches!(bits, 3 | 4) || !centroids.iter().all(|value| value.is_finite()) {
            return None;
        }
        Some(Self {
            bits,
            centroids,
            output: [0.0; DIMENSION],
            failed: false,
            #[cfg(target_arch = "aarch64")]
            neon: std::arch::is_aarch64_feature_detected!("neon"),
        })
    }

    /// Adds one code-only row without copying or rescanning the previous sum.
    ///
    /// Returns false on invalid input, overflow or any earlier failed append.
    /// Failure makes [`Self::finish`] return `None`, including after later valid
    /// appends. The private sum cannot escape while it is being accumulated.
    #[must_use]
    pub fn accumulate(&mut self, packed: &[u8], scale: f32) -> bool {
        if self.failed
            || packed.len() != DIMENSION * usize::from(self.bits) / 8
            || !scale.is_finite()
        {
            self.failed = true;
            return false;
        }
        #[cfg(target_arch = "aarch64")]
        let finite = if self.neon {
            native::accumulate(packed, self.bits, self.centroids, scale, &mut self.output)
        } else {
            portable_accumulate(packed, self.bits, self.centroids, scale, &mut self.output)
        };
        #[cfg(not(target_arch = "aarch64"))]
        let finite =
            portable_accumulate(packed, self.bits, self.centroids, scale, &mut self.output);
        self.failed = !finite;
        !self.failed
    }

    /// Publishes the completed sum only if every append succeeded.
    #[must_use]
    pub const fn finish(self) -> Option<[f32; DIMENSION]> {
        if self.failed { None } else { Some(self.output) }
    }
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
    let finite = if std::arch::is_aarch64_feature_detected!("neon") {
        native::accumulate(packed, bits, centroids, scale, &mut staged)
    } else {
        portable_accumulate(packed, bits, centroids, scale, &mut staged)
    };
    #[cfg(not(target_arch = "aarch64"))]
    let finite = portable_accumulate(packed, bits, centroids, scale, &mut staged);
    if !finite {
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

/// Applies three signed, normalized Hadamard passes in one transaction.
///
/// Each boolean selects a negative diagonal sign before its forward pass.
/// Inverse execution reverses the passes and applies each sign after its
/// Hadamard. Sign-bit toggles preserve finite multiplication by `-1` exactly,
/// including signed zero and subnormals. Every pass retains increasing-width
/// butterflies, normalization by `1/16` and its finite-result boundary.
///
/// Native dispatch is selected once. The input is checked once and one staging
/// row serves all passes; there are no per-pass row copies. Nonfinite input or
/// arithmetic returns false without changing any caller-owned coordinate.
#[must_use]
pub fn rotate(
    values: &mut [f32; DIMENSION],
    negative_signs: &[[bool; DIMENSION]; 3],
    inverse: bool,
) -> bool {
    if !values.iter().all(|value| value.is_finite()) {
        return false;
    }
    let mut staged = *values;
    #[cfg(target_arch = "aarch64")]
    let neon = std::arch::is_aarch64_feature_detected!("neon");
    for pass in 0..negative_signs.len() {
        let signs = &negative_signs[if inverse {
            negative_signs.len() - 1 - pass
        } else {
            pass
        }];
        if !inverse {
            apply_signs(&mut staged, signs);
        }
        #[cfg(target_arch = "aarch64")]
        if neon {
            native::hadamard(&mut staged);
        } else {
            portable_hadamard(&mut staged);
        }
        #[cfg(not(target_arch = "aarch64"))]
        portable_hadamard(&mut staged);
        if !staged.iter().all(|value| value.is_finite()) {
            return false;
        }
        if inverse {
            apply_signs(&mut staged, signs);
        }
    }
    *values = staged;
    true
}

fn apply_signs(values: &mut [f32; DIMENSION], negative_signs: &[bool; DIMENSION]) {
    for (value, &negative) in values.iter_mut().zip(negative_signs) {
        *value = f32::from_bits(value.to_bits() ^ (u32::from(negative) << 31));
    }
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
) -> bool {
    let mut finite = true;
    for (index, output) in output.iter_mut().enumerate() {
        *output += scale * coordinate(packed, bits, centroids, index);
        finite &= output.is_finite();
    }
    finite
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
        float32x4_t, uint8x16x4_t, vaddq_f32, vaddq_f64, vandq_u32, vcltq_u32, vcvt_f64_f32,
        vcvt_high_f64_f32, vdupq_n_f32, vdupq_n_f64, vdupq_n_u32, vget_low_f32, vld1q_f32,
        vld1q_s32, vld1q_u8, vld1q_u8_x4, vminvq_u32, vmulq_f32, vmulq_f64, vmulq_u32, vorrq_u8,
        vqtbl4q_u8, vreinterpretq_f32_u8, vreinterpretq_u8_u32, vreinterpretq_u32_f32, vshlq_u32,
        vst1q_f32, vst1q_f64, vsubq_f32,
    };

    use super::DIMENSION;

    pub(super) fn dot(
        packed: &[u8],
        bits: u8,
        centroids: &[f32; 16],
        query: &[f32; DIMENSION],
    ) -> f64 {
        if bits == 4 {
            // SAFETY: the safe public caller proves baseline NEON at runtime and
            // validates exactly 256 four-bit codes and finite fixed-array inputs.
            unsafe { dot_neon::<4>(packed, centroids, query) }
        } else {
            // SAFETY: the same caller admits only three- or four-bit codes;
            // this branch has exactly 256 three-bit codes and detected NEON.
            unsafe { dot_neon::<3>(packed, centroids, query) }
        }
    }

    #[target_feature(enable = "neon")]
    fn dot_neon<const BITS: u8>(
        packed: &[u8],
        centroids: &[f32; 16],
        query: &[f32; DIMENSION],
    ) -> f64 {
        let mut low_sum = vdupq_n_f64(0.0);
        let mut high_sum = vdupq_n_f64(0.0);
        let table = codebook_bytes(centroids);
        for (index, query) in query.as_chunks::<4>().0.iter().enumerate() {
            let values = if BITS == 4 {
                four_bit_tile(packed, index, table)
            } else {
                let values = super::tile(packed, BITS, centroids, index * 4);
                // SAFETY: values contains four initialized coordinates and NEON
                // permits an unaligned load; the temporary outlives the load.
                unsafe { vld1q_f32(values.as_ptr()) }
            };
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
    ) -> bool {
        if bits == 4 {
            // SAFETY: caller detects NEON, validates 256 four-bit codes and
            // finite inputs, and supplies a unique 256-coordinate staging array.
            unsafe { accumulate_neon::<4>(packed, centroids, scale, output) }
        } else {
            // SAFETY: the same caller admits only three- or four-bit codes;
            // this branch has exactly 256 three-bit codes and detected NEON.
            unsafe { accumulate_neon::<3>(packed, centroids, scale, output) }
        }
    }

    #[target_feature(enable = "neon")]
    fn accumulate_neon<const BITS: u8>(
        packed: &[u8],
        centroids: &[f32; 16],
        scale: f32,
        output: &mut [f32; DIMENSION],
    ) -> bool {
        let scale = vdupq_n_f32(scale);
        let magnitude = vdupq_n_u32(0x7fff_ffff);
        let infinity = vdupq_n_u32(0x7f80_0000);
        let mut finite = vdupq_n_u32(u32::MAX);
        let table = codebook_bytes(centroids);
        for (index, output) in output.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let values = if BITS == 4 {
                four_bit_tile(packed, index, table)
            } else {
                let values = super::tile(packed, BITS, centroids, index * 4);
                // SAFETY: values is a valid initialized four-element array.
                unsafe { vld1q_f32(values.as_ptr()) }
            };
            // SAFETY: output is a unique four-element chunk of the staging row.
            let previous = unsafe { vld1q_f32(output.as_ptr()) };
            let product = vmulq_f32(scale, values);
            let result = vaddq_f32(previous, product);
            // Check each result while it is live, rather than reading the
            // complete output row again. Infinity and every NaN encoding have
            // magnitudes at least the binary32 infinity exponent pattern.
            let absolute = vandq_u32(vreinterpretq_u32_f32(result), magnitude);
            finite = vandq_u32(finite, vcltq_u32(absolute, infinity));
            // SAFETY: the same unique four-element chunk covers the store.
            unsafe { vst1q_f32(output.as_mut_ptr(), result) };
        }
        vminvq_u32(finite) != 0
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn codebook_bytes(centroids: &[f32; 16]) -> uint8x16x4_t {
        // SAFETY: sixteen initialized binary32 centroids occupy exactly the
        // sixty-four bytes required by the four unaligned vector loads.
        unsafe { vld1q_u8_x4(centroids.as_ptr().cast()) }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn four_bit_tile(packed: &[u8], index: usize, table: uint8x16x4_t) -> float32x4_t {
        let byte = index * 2;
        let pair = u16::from_le_bytes([packed[byte], packed[byte + 1]]);
        // SAFETY: this initialized fixed array supplies exactly four shifts.
        let shifts = unsafe { vld1q_s32([0, -4, -8, -12].as_ptr()) };
        let codes = vandq_u32(
            vshlq_u32(vdupq_n_u32(u32::from(pair)), shifts),
            vdupq_n_u32(15),
        );
        // Each nibble selects four consecutive bytes of its binary32 centroid.
        // Multiplication broadcasts code * 4 into every byte of its lane; those
        // bases are multiples of four, so OR appends the offsets 0, 1, 2, 3.
        let bases = vreinterpretq_u8_u32(vmulq_u32(codes, vdupq_n_u32(0x0404_0404)));
        // SAFETY: the initialized array supplies sixteen byte offsets. Loading
        // explicit bytes keeps the table order independent of integer endianness.
        let offsets =
            unsafe { vld1q_u8([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3].as_ptr()) };
        vreinterpretq_f32_u8(vqtbl4q_u8(table, vorrq_u8(bases, offsets)))
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

    #[cfg(test)]
    mod tests {
        #[test]
        fn four_bit_lookup_preserves_every_nibble_lane_and_centroid_bits() {
            if !std::arch::is_aarch64_feature_detected!("neon") {
                return;
            }
            let centroids = [
                -f32::MAX,
                -17.25,
                -1.0,
                -f32::MIN_POSITIVE,
                -f32::from_bits(1),
                -0.0,
                0.0,
                f32::from_bits(1),
                f32::MIN_POSITIVE,
                0.125,
                0.5,
                1.0,
                3.25,
                17.25,
                f32::MAX * 0.5,
                f32::MAX,
            ];
            // SAFETY: NEON was detected above and the array owns all 64 bytes.
            let table = unsafe { super::codebook_bytes(&centroids) };
            for offset in 0_u8..16 {
                let codes: [u8; super::DIMENSION] =
                    std::array::from_fn(|index| index.to_le_bytes()[0].wrapping_add(offset) & 15);
                // Independent bit-at-a-time packing exercises both nibble
                // positions and all four output lanes, including the last tile.
                let mut packed = [0_u8; super::DIMENSION / 2];
                for (index, code) in codes.iter().enumerate() {
                    for bit in 0..4 {
                        let position = index * 4 + bit;
                        packed[position / 8] |= ((code >> bit) & 1) << (position % 8);
                    }
                }
                for (index, codes) in codes.as_chunks::<4>().0.iter().enumerate() {
                    // SAFETY: NEON was detected, packed owns 128 initialized
                    // bytes, and the enumerated tile is within its 64 tiles.
                    let tile = unsafe { super::four_bit_tile(&packed, index, table) };
                    let mut decoded = [0.0; 4];
                    // SAFETY: decoded is a unique initialized four-value array.
                    unsafe { super::vst1q_f32(decoded.as_mut_ptr(), tile) };
                    assert_eq!(
                        decoded.map(f32::to_bits),
                        codes.map(|code| centroids[usize::from(code)].to_bits()),
                    );
                }
            }
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

    // Keep this oracle independent of the production butterfly and sign helpers.
    // Match binary32 stage boundaries rather than using a reassociated dense dot.
    fn reference_rotation(
        mut values: [f32; DIMENSION],
        negative_signs: &[[bool; DIMENSION]; 3],
        inverse: bool,
    ) -> Option<[f32; DIMENSION]> {
        if values.iter().any(|value| !value.is_finite()) {
            return None;
        }
        for pass in 0..3 {
            let signs = &negative_signs[if inverse { 2 - pass } else { pass }];
            if !inverse {
                for (value, &negative) in values.iter_mut().zip(signs) {
                    *value *= if negative { -1.0 } else { 1.0 };
                }
            }
            for stage in 0..8 {
                let width = 1 << stage;
                for base in (0..DIMENSION).step_by(2 * width) {
                    for lane in 0..width {
                        let left = values[base + lane];
                        let right = values[base + width + lane];
                        values[base + lane] = left + right;
                        values[base + width + lane] = left - right;
                    }
                }
            }
            for value in &mut values {
                *value *= 0.0625;
            }
            if values.iter().any(|value| !value.is_finite()) {
                return None;
            }
            if inverse {
                for (value, &negative) in values.iter_mut().zip(signs) {
                    *value *= if negative { -1.0 } else { 1.0 };
                }
            }
        }
        Some(values)
    }

    #[test]
    fn x1_rotation_preserves_ordered_passes_signed_zero_and_subnormals() {
        let signs = std::array::from_fn(|pass: usize| {
            std::array::from_fn(|index: usize| (index * (2 * pass + 1) + pass) % 11 < 5)
        });
        let mixed = std::array::from_fn(|index: usize| {
            let value = f32::from(index.to_le_bytes()[0]) - 127.5;
            value / 32.0
        });
        let subnormal = std::array::from_fn(|index: usize| {
            f32::from_bits(u32::from(index.to_le_bytes()[0]))
        });
        for input in [mixed, subnormal, [0.0; DIMENSION], [-0.0; DIMENSION]] {
            for inverse in [false, true] {
                let mut actual = input;
                assert!(super::rotate(&mut actual, &signs, inverse));
                assert_eq!(
                    Some(actual.map(f32::to_bits)),
                    reference_rotation(input, &signs, inverse)
                        .map(|values| values.map(f32::to_bits)),
                );
            }
        }
    }

    #[test]
    fn x1_rotation_rejects_late_pass_overflow_without_publication() {
        let signs = [[false; DIMENSION]; 3];
        let mut late_overflow = [0.0; DIMENSION];
        late_overflow[0] = f32::MAX * 0.5;
        // The impulse's first Hadamard is finite; its second pass over the
        // resulting constant row overflows before normalization.
        let mut first = late_overflow;
        assert!(hadamard(&mut first));
        assert!(!hadamard(&mut first));
        for input in [
            late_overflow,
            [f32::MAX; DIMENSION],
            [f32::NAN; DIMENSION],
            [f32::INFINITY; DIMENSION],
            [f32::NEG_INFINITY; DIMENSION],
        ] {
            for inverse in [false, true] {
                let mut actual = input;
                assert!(!super::rotate(&mut actual, &signs, inverse));
                assert_eq!(actual.map(f32::to_bits), input.map(f32::to_bits));
                assert!(reference_rotation(input, &signs, inverse).is_none());
            }
        }
        let mut recovery = [0.25; DIMENSION];
        assert!(super::rotate(&mut recovery, &signs, false));
    }

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
    fn x1_value_scan_matches_independent_rounding_and_poisoned_failure() -> Result<(), &'static str>
    {
        let centroids = codebook();
        for bits in [3, 4] {
            let mut sum = super::ValueAccumulator::new(bits, &centroids)
                .ok_or("finite accumulator must be accepted")?;
            let mut expected = [0.0_f32; DIMENSION];
            for (offset, scale) in [0.125, -0.25, -0.0, f32::from_bits(1), 0.375]
                .into_iter()
                .enumerate()
            {
                let codes = codes(bits, offset.to_le_bytes()[0]);
                assert!(sum.accumulate(&packed(&codes, bits), scale));
                for (value, code) in expected.iter_mut().zip(codes) {
                    *value += scale * centroids[usize::from(code)];
                }
            }
            assert_eq!(
                sum.finish().map(|values| values.map(f32::to_bits)),
                Some(expected.map(f32::to_bits))
            );
            let bytes = packed(&codes(bits, 0), bits);
            for invalid_scale in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut sum = super::ValueAccumulator::new(bits, &centroids)
                    .ok_or("finite accumulator must be accepted")?;
                assert!(sum.accumulate(&bytes, 0.25));
                assert!(!sum.accumulate(&bytes, invalid_scale));
                assert!(!sum.accumulate(&bytes, 0.25));
                assert!(sum.finish().is_none());
            }
            let mut sum = super::ValueAccumulator::new(bits, &centroids)
                .ok_or("finite accumulator must be accepted")?;
            assert!(sum.accumulate(&bytes, 0.25));
            assert!(!sum.accumulate(&bytes[..bytes.len() - 1], 0.25));
            assert!(sum.finish().is_none());
            let huge = [f32::MAX; 16];
            let mut sum = super::ValueAccumulator::new(bits, &huge)
                .ok_or("finite extreme codebook must be accepted")?;
            assert!(sum.accumulate(&bytes, 0.25));
            assert!(!sum.accumulate(&bytes, 2.0));
            assert!(!sum.accumulate(&bytes, 0.0));
            assert!(sum.finish().is_none());
            let mut last_code = [0; DIMENSION];
            last_code[DIMENSION - 1] = (1 << bits) - 1;
            let mut last_huge = [1.0; 16];
            last_huge[usize::from((1_u8 << bits) - 1)] = f32::MAX;
            let bytes = packed(&last_code, bits);
            let mut sum = super::ValueAccumulator::new(bits, &last_huge)
                .ok_or("finite late-lane codebook must be accepted")?;
            assert!(sum.accumulate(&bytes, 0.25));
            // All earlier lanes update finitely before the final addition overflows.
            assert!(!sum.accumulate(&bytes, 1.0));
            assert!(sum.finish().is_none());
        }
        assert!(super::ValueAccumulator::new(2, &centroids).is_none());
        let mut invalid = centroids;
        invalid[15] = f32::NAN;
        assert!(super::ValueAccumulator::new(4, &invalid).is_none());
        Ok(())
    }

    #[test]
    fn packed_value_finite_checks_cover_each_vector_lane_and_late_overflow()
    -> Result<(), &'static str> {
        for bits in [3, 4] {
            for lane in [0, 1, 2, 3, 124, 125, 126, 127, 252, 253, 254, 255] {
                let mut codes = [0; DIMENSION];
                codes[lane] = 1;
                let bytes = packed(&codes, bits);
                for extreme in [f32::MAX, -f32::MAX] {
                    let mut centroids = [1.0; 16];
                    centroids[1] = extreme;
                    for scale in [1.0, 2.0] {
                        let mut sum = super::ValueAccumulator::new(bits, &centroids)
                            .ok_or("finite codebook must be accepted")?;
                        assert!(sum.accumulate(&bytes, 0.25));
                        // Scale one overflows on addition; scale two overflows
                        // on multiplication. Both failures poison the scan.
                        assert!(!sum.accumulate(&bytes, scale));
                        assert!(!sum.accumulate(&bytes, 0.0));
                        assert!(sum.finish().is_none());

                        let mut output = [0.25; DIMENSION];
                        output[lane] = extreme * 0.25;
                        let before = output.map(f32::to_bits);
                        assert!(!accumulate(&bytes, bits, &centroids, scale, &mut output));
                        assert_eq!(output.map(f32::to_bits), before);
                        assert!(!super::portable_accumulate(
                            &bytes,
                            bits,
                            &centroids,
                            scale,
                            &mut output,
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn x1_prepared_kv_dot_matches_independent_lanes_and_checks_every_row()
    -> Result<(), &'static str> {
        let centroids = codebook();
        let query = std::array::from_fn(|index: usize| {
            let value = f32::from(index.to_le_bytes()[0]) - 127.5;
            value * 0.03125 + (value * 0.17).sin()
        });
        for bits in [3, 4] {
            let prepared = super::PreparedQuery::new(bits, &centroids, &query)
                .ok_or("finite prepared query must be accepted")?;
            // Reuse exactly the same packed allocation with different bytes.
            let mut bytes = packed(&codes(bits, 0), bits);
            for offset in 0..(1 << bits) {
                let codes = codes(bits, offset);
                bytes.copy_from_slice(&packed(&codes, bits));
                let mut lanes = [0.0_f64; 4];
                for (index, (&code, &query)) in codes.iter().zip(&query).enumerate() {
                    lanes[index % 4] += f64::from(centroids[usize::from(code)]) * f64::from(query);
                }
                let expected =
                    super::finite_binary32(((lanes[0] + lanes[1]) + lanes[2]) + lanes[3]);
                assert_eq!(
                    prepared.dot(&bytes).map(f32::to_bits),
                    expected.map(f32::to_bits)
                );
            }
            assert!(prepared.dot(&bytes[..bytes.len() - 1]).is_none());
            bytes.push(0);
            assert!(prepared.dot(&bytes).is_none());
        }
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut invalid_query = query;
            invalid_query[255] = invalid;
            assert!(super::PreparedQuery::new(4, &centroids, &invalid_query).is_none());
            let mut invalid_book = centroids;
            invalid_book[15] = invalid;
            assert!(super::PreparedQuery::new(4, &invalid_book, &query).is_none());
        }
        assert!(super::PreparedQuery::new(2, &centroids, &query).is_none());
        let huge = [f32::MAX; DIMENSION];
        let book = [f32::MAX; 16];
        assert!(
            super::PreparedQuery::new(4, &book, &huge)
                .and_then(|query| query.dot(&[0; 128]))
                .is_none()
        );
        Ok(())
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
                assert!(super::portable_accumulate(
                    &bytes,
                    bits,
                    &centroids,
                    -0.125,
                    &mut scalar,
                ));
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
