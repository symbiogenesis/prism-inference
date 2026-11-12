//! Experimental, clean-room rotated scalar quantization for 256-channel KV rows.
//!
//! This is an approximate codec, not Google's `TurboQuant` implementation or its
//! QJL estimator. Three seeded signed, normalized Walsh-Hadamard passes spread
//! a row before a symmetric Gaussian Lloyd-Max scalar codebook quantizes it.
//! Each record contains a positive binary32 scale followed by 256 tightly packed
//! codes. The scale is the original row's Euclidean norm divided by 16; no
//! reconstructed-norm correction is applied. Zero has one canonical encoding.
//!
//! [`Quantizer::score`] and [`Quantizer::accumulate`] consume packed codes directly
//! in the rotated basis. Only one 256-channel scratch row is needed; a complete
//! history never needs expanding. Record identity is supplied by the enclosing
//! cache owner and must match [`Quantizer::identity`]. The codec does not encode
//! that identity in every row or silently change the context limit.

use crate::{Result, invalid};

mod scalar;
mod tables;

#[cfg(test)]
mod tests;

/// Fixed key/value head width supported by this experimental profile.
pub const HEAD_DIM: usize = 256;
const SCALE_BYTES: usize = size_of::<f32>();
const MAX_RECORD_BYTES: usize = SCALE_BYTES + HEAD_DIM / 2;
const ROTATION_PASSES: usize = 3;
const FORMAT_VERSION: u32 = 1;

/// Packed scalar precision; both key and value precision are explicit policies.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Bits {
    /// Eight symmetric Gaussian reconstruction levels, 96 code bytes per row.
    Three,
    /// Sixteen symmetric Gaussian reconstruction levels, 128 code bytes per row.
    Four,
}

impl Bits {
    /// Number of bits used by each of the 256 scalar codes.
    #[must_use]
    pub const fn bit_width(self) -> u8 {
        match self {
            Self::Three => 3,
            Self::Four => 4,
        }
    }

    const fn levels(self) -> u8 {
        match self {
            Self::Three => 8,
            Self::Four => 16,
        }
    }
}

/// Immutable mathematical and packed-layout identity, independent of CPU ISA.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Identity {
    /// Version binds packing, codebooks, rotation order, scale and arithmetic.
    pub version: u32,
    /// Scalar code precision.
    pub bits: Bits,
    /// `SplitMix64` seed defining all three independent diagonal sign masks.
    pub seed: u64,
}

/// Fixed-size, allocation-free row codec with deterministic rotation identity.
#[derive(Clone)]
pub struct Quantizer {
    identity: Identity,
    signs: [[f32; HEAD_DIM]; ROTATION_PASSES],
    scalar_only: bool,
}

impl Quantizer {
    /// Construct a codec, using native helpers when compiled for Windows ARM64.
    #[must_use]
    pub fn new(bits: Bits, seed: u64) -> Self {
        let mut stream = seed;
        let mut signs = [[1.0; HEAD_DIM]; ROTATION_PASSES];
        for pass in &mut signs {
            for sign in pass {
                *sign = if scalar::splitmix64(&mut stream) & 1 == 0 {
                    1.0
                } else {
                    -1.0
                };
            }
        }
        Self {
            identity: Identity {
                version: FORMAT_VERSION,
                bits,
                seed,
            },
            signs,
            scalar_only: false,
        }
    }

    /// Construct the canonical scalar reference, even on a native ARM64 target.
    #[must_use]
    pub fn scalar(bits: Bits, seed: u64) -> Self {
        Self {
            scalar_only: true,
            ..Self::new(bits, seed)
        }
    }

    /// Mathematical and wire identity; scalar and native paths share it.
    #[must_use]
    pub const fn identity(&self) -> Identity {
        self.identity
    }

    /// Whether this instance selects the compiled Windows ARM64 native helpers.
    #[must_use]
    pub const fn uses_neon(&self) -> bool {
        cfg!(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc"
        )) && !self.scalar_only
    }

    /// Exact record length: 100 bytes at three bits, 132 bytes at four bits.
    #[must_use]
    pub const fn encoded_len(&self) -> usize {
        match self.identity.bits {
            Bits::Three => 100,
            Bits::Four => 132,
        }
    }

    /// Rotate one finite row, without quantization or normalization of its norm.
    ///
    /// # Errors
    /// Rejects non-256 lengths, nonfinite input and nonfinite arithmetic. `out`
    /// is unchanged on failure, including intermediate transform overflow.
    pub fn rotate(&self, input: &[f32], out: &mut [f32; HEAD_DIM]) -> Result<()> {
        let mut row = finite_row(input)?;
        self.forward(&mut row)?;
        *out = row;
        Ok(())
    }

    /// Restore the original basis of one finite rotated row in place.
    ///
    /// # Errors
    /// Rejects nonfinite input or arithmetic; the row is unchanged on failure.
    pub fn inverse(&self, inout: &mut [f32; HEAD_DIM]) -> Result<()> {
        let mut row = finite_row(inout)?;
        for signs in self.signs.iter().rev() {
            self.hadamard(&mut row)?;
            for (value, sign) in row.iter_mut().zip(signs) {
                *value *= sign;
            }
        }
        *inout = row;
        Ok(())
    }

    /// Encode a complete row into an exactly sized, caller-owned packed record.
    ///
    /// A finite all-zero row produces scale `+0` and all-zero codes. For nonzero
    /// rows the stored scale is the binary32-rounded Euclidean norm divided by
    /// 16. Normalization uses that same stored scale, including for subnormals.
    ///
    /// # Errors
    /// Rejects incorrect lengths, nonfinite input, a nonzero row whose scale
    /// underflows to zero, or nonfinite arithmetic. `out` is unchanged on error.
    pub fn encode(&self, input: &[f32], out: &mut [u8]) -> Result<()> {
        if out.len() != self.encoded_len() {
            return Err(invalid("KV codec record length mismatch"));
        }
        let mut row = finite_row(input)?;
        let squared_norm = row.iter().fold(0.0_f64, |sum, &value| {
            f64::from(value).mul_add(f64::from(value), sum)
        });
        let mut record = [0_u8; MAX_RECORD_BYTES];
        if squared_norm == 0.0 {
            out.copy_from_slice(&record[..self.encoded_len()]);
            return Ok(());
        }
        let scale = scalar::checked_f32(squared_norm.sqrt() / 16.0)?;
        if scale == 0.0 {
            return Err(invalid("nonzero KV row scale underflows"));
        }
        for value in &mut row {
            *value = scalar::checked_f32(f64::from(*value) / f64::from(scale))?;
        }
        self.forward(&mut row)?;
        record[..SCALE_BYTES].copy_from_slice(&scale.to_le_bytes());
        let codes = &mut record[SCALE_BYTES..self.encoded_len()];
        let centroids = self.centroids();
        for (index, &value) in row.iter().enumerate() {
            let code = scalar::nearest(value, centroids, self.identity.bits.levels());
            if !(scale * centroids[usize::from(code)]).is_finite() {
                return Err(invalid("KV codec reconstructed coordinate overflow"));
            }
            scalar::write_code(codes, self.identity.bits.bit_width(), index, code);
        }
        out.copy_from_slice(&record[..self.encoded_len()]);
        Ok(())
    }

    /// Dot a full-precision, already rotated query with one packed key row.
    ///
    /// Four binary64 lanes accumulate binary32 products, reduce in fixed order,
    /// round to binary32, then apply the record scale. Native helpers preserve
    /// that ordering; this is deliberately not a fused multiply-add contract.
    ///
    /// # Errors
    /// Rejects malformed records, nonfinite queries and nonfinite arithmetic.
    pub fn score(&self, rotated_query: &[f32; HEAD_DIM], packed: &[u8]) -> Result<f32> {
        finite_values(rotated_query)?;
        let (scale, codes) = self.record(packed)?;
        if scale == 0.0 {
            return Ok(0.0);
        }
        let unscaled = self.dot(codes, rotated_query)?;
        let score = unscaled * scale;
        if !score.is_finite() {
            return Err(invalid("KV codec score overflow"));
        }
        Ok(score)
    }

    /// Add one weighted packed value row into a rotated-space accumulator.
    ///
    /// Products and additions are separate binary32 operations. The caller
    /// applies [`Self::inverse`] once after accumulating all attention weights.
    ///
    /// # Errors
    /// Rejects malformed records, nonfinite weights/accumulators and nonfinite
    /// arithmetic. The complete accumulator is unchanged on failure.
    pub fn accumulate(&self, packed: &[u8], weight: f32, out: &mut [f32; HEAD_DIM]) -> Result<()> {
        finite_values(out)?;
        if !weight.is_finite() {
            return Err(invalid("KV codec weight must be finite"));
        }
        let (scale, codes) = self.record(packed)?;
        let weighted_scale = weight * scale;
        if !weighted_scale.is_finite() {
            return Err(invalid("KV codec weighted scale overflow"));
        }
        if scale == 0.0 {
            return Ok(());
        }
        self.add_codes(codes, weighted_scale, out)
    }

    /// Decode one row into the original basis, for diagnostics or explicit replay.
    ///
    /// # Errors
    /// Rejects malformed records and nonfinite reconstruction arithmetic. `out`
    /// is unchanged on failure; ordinary attention needs no full-history decode.
    pub fn decode(&self, packed: &[u8], out: &mut [f32; HEAD_DIM]) -> Result<()> {
        let mut row = [0.0; HEAD_DIM];
        self.accumulate(packed, 1.0, &mut row)?;
        self.inverse(&mut row)?;
        *out = row;
        Ok(())
    }

    const fn centroids(&self) -> &'static [f32; 16] {
        match self.identity.bits {
            Bits::Three => &tables::THREE,
            Bits::Four => &tables::FOUR,
        }
    }

    fn forward(&self, row: &mut [f32; HEAD_DIM]) -> Result<()> {
        for signs in &self.signs {
            for (value, sign) in row.iter_mut().zip(signs) {
                *value *= sign;
            }
            self.hadamard(row)?;
        }
        Ok(())
    }

    fn record<'a>(&self, packed: &'a [u8]) -> Result<(f32, &'a [u8])> {
        if packed.len() != self.encoded_len() {
            return Err(invalid("KV codec record length mismatch"));
        }
        let bytes: [u8; SCALE_BYTES] = packed[..SCALE_BYTES]
            .try_into()
            .map_err(|_| invalid("KV codec scale length mismatch"))?;
        let scale = f32::from_le_bytes(bytes);
        let codes = &packed[SCALE_BYTES..];
        if !scale.is_finite() || scale.is_sign_negative() {
            return Err(invalid("KV codec scale must be finite and nonnegative"));
        }
        if scale == 0.0 && codes.iter().any(|&code| code != 0) {
            return Err(invalid("KV codec zero row must be canonical"));
        }
        Ok((scale, codes))
    }

    #[cfg_attr(
        not(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc"
        )),
        expect(
            clippy::unused_self,
            reason = "The native target reads the instance backend policy; portable targets share this method with the scalar path."
        )
    )]
    fn hadamard(&self, row: &mut [f32; HEAD_DIM]) -> Result<()> {
        #[cfg(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc"
        ))]
        if !self.scalar_only {
            return if bonsai_neon_dot::kv_quant::hadamard(row) {
                Ok(())
            } else {
                Err(invalid("KV codec native transform rejected arithmetic"))
            };
        }
        scalar::hadamard(row)
    }

    fn dot(&self, codes: &[u8], query: &[f32; HEAD_DIM]) -> Result<f32> {
        #[cfg(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc"
        ))]
        if !self.scalar_only {
            return bonsai_neon_dot::kv_quant::dot(
                codes,
                self.identity.bits.bit_width(),
                self.centroids(),
                query,
            )
            .ok_or_else(|| invalid("KV codec native dot rejected arithmetic"));
        }
        scalar::dot(
            codes,
            self.identity.bits.bit_width(),
            self.centroids(),
            query,
        )
    }

    fn add_codes(&self, codes: &[u8], scale: f32, out: &mut [f32; HEAD_DIM]) -> Result<()> {
        #[cfg(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc"
        ))]
        if !self.scalar_only {
            return if bonsai_neon_dot::kv_quant::accumulate(
                codes,
                self.identity.bits.bit_width(),
                self.centroids(),
                scale,
                out,
            ) {
                Ok(())
            } else {
                Err(invalid("KV codec native accumulation rejected arithmetic"))
            };
        }
        scalar::accumulate(
            codes,
            self.identity.bits.bit_width(),
            self.centroids(),
            scale,
            out,
        )
    }
}

fn finite_values(row: &[f32]) -> Result<()> {
    if row.iter().any(|value| !value.is_finite()) {
        return Err(invalid("KV codec row must contain only finite values"));
    }
    Ok(())
}

fn finite_row(input: &[f32]) -> Result<[f32; HEAD_DIM]> {
    let row: [f32; HEAD_DIM] = input
        .try_into()
        .map_err(|_| invalid("KV codec requires exactly 256 channels"))?;
    finite_values(&row)?;
    Ok(row)
}
