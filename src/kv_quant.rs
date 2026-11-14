//! Experimental, clean-room rotated scalar quantization for 256-channel KV rows.
//!
//! This is an approximate codec, not Google's `TurboQuant` implementation or its
//! QJL estimator. Three seeded signed, normalized Walsh-Hadamard passes spread
//! a row before a symmetric Gaussian Lloyd-Max scalar codebook quantizes it.
//! Each record contains a positive binary32 scale followed by 256 tightly packed
//! codes. Version 1 uses the original row's Euclidean norm divided by 16.
//! Explicit version 2 fits that scale to the selected codes, with a rounded
//! rotated-reconstruction error guard. Zero has one canonical encoding.
//!
//! [`Quantizer::score`] and [`Quantizer::accumulate`] consume packed codes directly
//! in the rotated basis. Only one 256-channel scratch row is needed; a complete
//! history never needs expanding. Record identity is supplied by the enclosing
//! cache owner and must match [`Quantizer::identity`]. The codec does not encode
//! that identity in every row or silently change the context limit.

use crate::{Result, invalid};

mod fitted_scale;
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
const FITTED_SCALE_VERSION: u32 = 2;

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

/// A finite rotated query whose immutable borrow spans one attention scan.
pub(crate) struct PreparedQuery<'a> {
    quantizer: &'a Quantizer,
    query: &'a [f32; HEAD_DIM],
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc"
    ))]
    native: Option<bonsai_neon_dot::kv_quant::PreparedQuery<'a>>,
}

impl PreparedQuery<'_> {
    /// Retains per-record validation and result rounding without rescanning the query.
    pub(crate) fn score(&self, packed: &[u8]) -> Result<f32> {
        let (scale, codes) = self.quantizer.record(packed)?;
        if scale == 0.0 {
            return Ok(0.0);
        }
        #[cfg(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc"
        ))]
        let unscaled = if let Some(native) = &self.native {
            native
                .dot(codes)
                .ok_or_else(|| invalid("KV codec native dot rejected arithmetic"))?
        } else {
            self.scalar_dot(codes)?
        };
        #[cfg(not(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc"
        )))]
        let unscaled = self.scalar_dot(codes)?;
        let score = unscaled * scale;
        if !score.is_finite() {
            return Err(invalid("KV codec score overflow"));
        }
        Ok(score)
    }

    fn scalar_dot(&self, codes: &[u8]) -> Result<f32> {
        scalar::dot(
            codes,
            self.quantizer.identity.bits.bit_width(),
            self.quantizer.centroids(),
            self.query,
        )
    }
}

enum AccumulatorState {
    Scalar([f32; HEAD_DIM]),
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc"
    ))]
    Native(bonsai_neon_dot::kv_quant::ValueAccumulator<'static>),
}

/// Owns a provisional finite sum, so prior successful rows need no input rescan.
pub(crate) struct ValueAccumulator<'a> {
    quantizer: &'a Quantizer,
    state: AccumulatorState,
    failed: bool,
}

impl ValueAccumulator<'_> {
    /// Each row still checks its record, weight and resulting finite sum.
    pub(crate) fn accumulate(&mut self, packed: &[u8], weight: f32) -> Result<()> {
        if self.failed {
            return Err(invalid("KV codec accumulation already failed"));
        }
        let result = self.append(packed, weight);
        self.failed = result.is_err();
        result
    }

    fn append(&mut self, packed: &[u8], weight: f32) -> Result<()> {
        if !weight.is_finite() {
            return Err(invalid("KV codec weight must be finite"));
        }
        let (scale, codes) = self.quantizer.record(packed)?;
        let weighted_scale = weight * scale;
        if !weighted_scale.is_finite() {
            return Err(invalid("KV codec weighted scale overflow"));
        }
        // This private sum starts at +0 and uses separate binary32 products
        // and additions. Adding finite binary32 operands cannot underflow an
        // exact nonzero sum below the smallest subnormal; exact cancellation
        // produces +0. Thus no successful append can leave a -0 coordinate.
        // Adding either signed zero is consequently bit-preserving, including
        // when weight * scale underflows. Keep all record/weight/scale checks
        // above this shortcut; a zero probability cannot hide malformed rows.
        // Public accumulate accepts caller-owned -0 and retains its full path.
        if weighted_scale == 0.0 {
            return Ok(());
        }
        match &mut self.state {
            AccumulatorState::Scalar(output) => scalar::accumulate_in_place(
                codes,
                self.quantizer.identity.bits.bit_width(),
                self.quantizer.centroids(),
                weighted_scale,
                output,
            ),
            #[cfg(all(
                feature = "experimental-neon",
                target_arch = "aarch64",
                target_os = "windows",
                target_env = "msvc"
            ))]
            AccumulatorState::Native(output) => {
                if output.accumulate(codes, weighted_scale) {
                    Ok(())
                } else {
                    Err(invalid("KV codec native accumulation rejected arithmetic"))
                }
            }
        }
    }

    /// Failure is sticky; no intermediate or invalid sum can be published.
    pub(crate) fn finish(self) -> Result<[f32; HEAD_DIM]> {
        if self.failed {
            return Err(invalid("KV codec accumulation already failed"));
        }
        match self.state {
            AccumulatorState::Scalar(output) => Ok(output),
            #[cfg(all(
                feature = "experimental-neon",
                target_arch = "aarch64",
                target_os = "windows",
                target_env = "msvc"
            ))]
            AccumulatorState::Native(output) => output
                .finish()
                .ok_or_else(|| invalid("KV codec native accumulation rejected arithmetic")),
        }
    }
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

    /// Select version 2's guarded least-squares scale for fixed scalar codes.
    ///
    /// Packing, rotation and decoding are unchanged. Only encoding fits the
    /// scale; owners must bind this different numerical identity to their cache.
    /// The guard measures rotated reconstruction, not attention or answer quality.
    #[must_use]
    pub const fn with_fitted_scale(mut self) -> Self {
        self.identity.version = FITTED_SCALE_VERSION;
        self
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
    /// rows normalization uses the binary32-rounded Euclidean norm divided by
    /// 16, including for subnormals. Version 1 stores that norm scale unchanged.
    /// Version 2 keeps the same codes and fits the stored scale, accepting only
    /// a finite positive candidate with strictly lower rotated squared error
    /// after binary32 reconstruction products. Otherwise it keeps the norm scale.
    ///
    /// # Errors
    /// Rejects incorrect lengths, nonfinite input, a nonzero row whose scale
    /// underflows to zero, or nonfinite arithmetic. `out` is unchanged on error.
    pub fn encode(&self, input: &[f32], out: &mut [u8]) -> Result<()> {
        if out.len() != self.encoded_len() {
            return Err(invalid("KV codec record length mismatch"));
        }
        let mut row: [f32; HEAD_DIM] = input
            .try_into()
            .map_err(|_| invalid("KV codec requires exactly 256 channels"))?;
        // The norm already visits every coordinate before normalization or
        // publication. Validate there without a separate finite-input scan,
        // retaining the original ordered binary64 FMA accumulation.
        let squared_norm = row.iter().try_fold(0.0_f64, |sum, &value| {
            if !value.is_finite() {
                return Err(invalid("KV codec row must contain only finite values"));
            }
            Ok(f64::from(value).mul_add(f64::from(value), sum))
        })?;
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
        let codes = &mut record[SCALE_BYTES..self.encoded_len()];
        let centroids = self.centroids();
        // Preserve the explicit scalar codec's exhaustive nearest-code control;
        // ordinary encoding uses the same rounded-distance rule over the
        // immutable, strictly ordered active codebook.
        for (index, &value) in row.iter().enumerate() {
            let code = if self.scalar_only {
                scalar::nearest(value, centroids, self.identity.bits.levels())
            } else {
                scalar::nearest_ordered(value, centroids, self.identity.bits.levels())
            };
            if !(scale * centroids[usize::from(code)]).is_finite() {
                return Err(invalid("KV codec reconstructed coordinate overflow"));
            }
            scalar::write_code(codes, self.identity.bits.bit_width(), index, code);
        }
        let scale = if self.identity.version == FITTED_SCALE_VERSION {
            fitted_scale::select(&row, codes, self.identity.bits, centroids, scale)
        } else {
            scale
        };
        record[..SCALE_BYTES].copy_from_slice(&scale.to_le_bytes());
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
        self.prepare_query(rotated_query)?.score(packed)
    }

    /// Checks an immutable rotated query once before scanning packed history.
    pub(crate) fn prepare_query<'a>(
        &'a self,
        query: &'a [f32; HEAD_DIM],
    ) -> Result<PreparedQuery<'a>> {
        #[cfg(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc"
        ))]
        let native = if self.scalar_only {
            finite_values(query)?;
            None
        } else {
            Some(
                bonsai_neon_dot::kv_quant::PreparedQuery::new(
                    self.identity.bits.bit_width(),
                    self.centroids(),
                    query,
                )
                .ok_or_else(|| invalid("KV codec row must contain only finite values"))?,
            )
        };
        #[cfg(not(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc"
        )))]
        finite_values(query)?;
        Ok(PreparedQuery {
            quantizer: self,
            query,
            #[cfg(all(
                feature = "experimental-neon",
                target_arch = "aarch64",
                target_os = "windows",
                target_env = "msvc"
            ))]
            native,
        })
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

    /// Starts one exclusive packed-value scan without retaining a second row.
    #[cfg_attr(
        not(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc"
        )),
        expect(
            clippy::unnecessary_wraps,
            reason = "The native branch validates its helper codebook before starting the scan; portable construction is infallible."
        )
    )]
    #[cfg_attr(
        not(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc"
        )),
        allow(
            clippy::missing_const_for_fn,
            reason = "The shared native entry performs runtime ISA selection; portable construction keeps the same non-const API."
        )
    )]
    pub(crate) fn value_accumulator(&self) -> Result<ValueAccumulator<'_>> {
        #[cfg(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc"
        ))]
        if !self.scalar_only {
            let state = bonsai_neon_dot::kv_quant::ValueAccumulator::new(
                self.identity.bits.bit_width(),
                self.centroids(),
            )
            .ok_or_else(|| invalid("KV codec native accumulation rejected codebook"))?;
            return Ok(ValueAccumulator {
                quantizer: self,
                state: AccumulatorState::Native(state),
                failed: false,
            });
        }
        Ok(ValueAccumulator {
            quantizer: self,
            state: AccumulatorState::Scalar([0.0; HEAD_DIM]),
            failed: false,
        })
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
