//! Audited ARMv8.2-A `dotprod` (`vdotq_s32`) and `i8mm` (`vmmlaq_s32`) SIMD
//! dot-product, exact binary16 widening, ordered BF16 row reductions and
//! non-widening `PTQ1_0` unpacking.
//!
//! Public entry points use fixed-array or checked borrowed-slice boundaries
//! and retain exact portable arithmetic when their target features are absent.
//! Binary16 widening preserves every bit pattern through a baseline-NEON
//! finite fast path and exact exceptional decoding.

#![deny(unsafe_code)]

mod scaled;
pub use scaled::{
    Pq2Activation, Pq2I8mmPairBlock, Ptq1Activation, Ptq1PinnedI8mmPairBlock, Q8ActivationBlock,
    Q8ActivationView,
};

pub use bonsai_ternary::{
    checked_float_to_half, float_to_half, half_to_float, unpack_q4_k_scale_min,
};

mod half;
pub use half::{
    narrow_f32_eight, narrow_f32_four, narrow_f32_slice, narrow_f32_slice_checked, widen_f16_eight,
    widen_f16_four, widen_f16_slice,
};

/// K-quant activation quantizers, `SDOT`/`I8MM` row/strip/tile kernels, and 8-lane dots.
pub mod kquants;

mod bf16;
pub use bf16::dot_bf16_rows;

#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_endian = "little"
))]
#[allow(
    unsafe_code,
    reason = "Baseline-NEON F16 widening and ordered binary64/pinned FMA arithmetic stay behind checked slice boundaries."
)]
mod f16_aarch64;

/// Multiplies one `F16` row by one binary32 activation vector using ordered binary64 summation.
///
/// Matches left-to-right `float_dot` rounding (`-0.0_f64` seed, per-product `f32` rounding,
/// binary64 addition, single final `f32` cast). Rejects empty/mismatched extents, nonfinite
/// `F16` weights (even when multiplied by zero activations), or nonfinite results.
#[inline]
#[must_use]
pub fn dot_f16_row(row: &[u8], input: &[f32]) -> Option<f32> {
    if input.is_empty() || input.len().checked_mul(2) != Some(row.len()) {
        return None;
    }
    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    ))]
    {
        f16_aarch64::dot_row(row, input)
    }
    #[cfg(not(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    )))]
    {
        portable_f16_f64_row(row, input)
    }
}

/// Multiplies four or eight `F16` rows by one shared binary32 activation vector.
///
/// Preserves exact `0 ULP` left-to-right binary64 summation per row and rejects empty/mismatched
/// extents, unsupported row counts, nonfinite weights, or nonfinite results.
#[inline]
#[must_use]
pub fn dot_f16_rows<const ROWS: usize>(rows: [&[u8]; ROWS], input: &[f32]) -> Option<[f32; ROWS]> {
    if !matches!(ROWS, 4 | 8)
        || input.is_empty()
        || input
            .len()
            .checked_mul(2)
            .is_none_or(|bytes| rows.iter().any(|row| row.len() != bytes))
    {
        return None;
    }
    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    ))]
    {
        f16_aarch64::dot_rows(rows, input)
    }
    #[cfg(not(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    )))]
    {
        let mut out = [0.0_f32; ROWS];
        for (dst, row) in out.iter_mut().zip(rows) {
            *dst = portable_f16_f64_row(row, input)?;
        }
        Some(out)
    }
}

/// Multiplies one 16-aligned `F16` row by one binary32 activation vector using the
/// pinned four-register FMA reduction tree matching `pinned_gdn::dot`.
///
/// Rejects empty/unaligned/mismatched extents, nonfinite weights, or nonfinite results.
#[inline]
#[must_use]
pub fn dot_f16_row_pinned(row: &[u8], input: &[f32]) -> Option<f32> {
    if input.is_empty()
        || !input.len().is_multiple_of(16)
        || input.len().checked_mul(2) != Some(row.len())
    {
        return None;
    }
    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    ))]
    {
        f16_aarch64::dot_row_pinned(row, input)
    }
    #[cfg(not(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    )))]
    {
        portable_f16_pinned_row(row, input)
    }
}

/// Multiplies four or eight 16-aligned `F16` rows by one binary32 activation vector using
/// the pinned four-register FMA reduction tree matching `pinned_gdn::dot`.
#[inline]
#[must_use]
pub fn dot_f16_rows_pinned<const ROWS: usize>(
    rows: [&[u8]; ROWS],
    input: &[f32],
) -> Option<[f32; ROWS]> {
    if !matches!(ROWS, 4 | 8)
        || input.is_empty()
        || !input.len().is_multiple_of(16)
        || input
            .len()
            .checked_mul(2)
            .is_none_or(|bytes| rows.iter().any(|row| row.len() != bytes))
    {
        return None;
    }
    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    ))]
    {
        f16_aarch64::dot_rows_pinned(rows, input)
    }
    #[cfg(not(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    )))]
    {
        let mut out = [0.0_f32; ROWS];
        for (dst, row) in out.iter_mut().zip(rows) {
            *dst = portable_f16_pinned_row(row, input)?;
        }
        Some(out)
    }
}

fn valid_f16_tile_inputs<const ROWS: usize, const TOKENS: usize>(
    rows: [&[u8]; ROWS],
    inputs: &[&[f32]],
) -> bool {
    if inputs.is_empty() || inputs.len() > TOKENS {
        return false;
    }
    let columns = inputs[0].len();
    columns != 0
        && columns
            .checked_mul(2)
            .is_some_and(|bytes| rows.iter().all(|row| row.len() == bytes))
        && inputs.iter().all(|input| input.len() == columns)
}

/// Multiplies one `F16` row across `1..=TOKENS` activations using ordered binary64 summation,
/// reusing widened `F16` weights across tokens. Leaves `out` unchanged on failure.
#[inline]
#[must_use]
pub fn dot_f16_tile_row<const TOKENS: usize>(
    row: &[u8],
    inputs: &[&[f32]],
    out: &mut [f32; TOKENS],
) -> Option<()> {
    if !valid_f16_tile_inputs::<1, TOKENS>([row], inputs) {
        return None;
    }
    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    ))]
    {
        f16_aarch64::dot_tile_row(row, inputs, out)
    }
    #[cfg(not(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    )))]
    {
        let mut tmp = [0.0_f32; TOKENS];
        for (dst, input) in tmp.iter_mut().zip(inputs) {
            *dst = portable_f16_f64_row(row, input)?;
        }
        *out = tmp;
        Some(())
    }
}

/// Multiplies four or eight `F16` rows across `1..=TOKENS` activations using ordered binary64
/// summation, reusing widened `F16` weight pairs across tokens. Leaves `out` unchanged on failure.
#[inline]
#[must_use]
pub fn dot_f16_tile_rows<const ROWS: usize, const TOKENS: usize>(
    rows: [&[u8]; ROWS],
    inputs: &[&[f32]],
    out: &mut [[f32; TOKENS]; ROWS],
) -> Option<()> {
    if !matches!(ROWS, 4 | 8) || !valid_f16_tile_inputs::<ROWS, TOKENS>(rows, inputs) {
        return None;
    }
    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    ))]
    {
        match ROWS {
            4 => f16_aarch64::dot_tile_rows::<2, ROWS, TOKENS>(rows, inputs, out),
            8 => f16_aarch64::dot_tile_rows::<4, ROWS, TOKENS>(rows, inputs, out),
            _ => None,
        }
    }
    #[cfg(not(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    )))]
    {
        let mut tmp = [[0.0_f32; TOKENS]; ROWS];
        for (dst_row, row) in tmp.iter_mut().zip(rows) {
            for (dst, input) in dst_row.iter_mut().zip(inputs) {
                *dst = portable_f16_f64_row(row, input)?;
            }
        }
        *out = tmp;
        Some(())
    }
}

/// Multiplies one 16-aligned `F16` row across `1..=TOKENS` activations using the pinned four-FMA
/// reduction tree, reusing widened `F16` weights across tokens. Leaves `out` unchanged on failure.
#[inline]
#[must_use]
pub fn dot_f16_tile_row_pinned<const TOKENS: usize>(
    row: &[u8],
    inputs: &[&[f32]],
    out: &mut [f32; TOKENS],
) -> Option<()> {
    if !valid_f16_tile_inputs::<1, TOKENS>([row], inputs) || !inputs[0].len().is_multiple_of(16) {
        return None;
    }
    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    ))]
    {
        f16_aarch64::dot_tile_row_pinned(row, inputs, out)
    }
    #[cfg(not(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    )))]
    {
        let mut tmp = [0.0_f32; TOKENS];
        for (dst, input) in tmp.iter_mut().zip(inputs) {
            *dst = portable_f16_pinned_row(row, input)?;
        }
        *out = tmp;
        Some(())
    }
}

/// Multiplies four or eight 16-aligned `F16` rows across `1..=TOKENS` activations using the
/// pinned four-FMA reduction tree, reusing widened `F16` weights across tokens. Leaves `out`
/// unchanged on failure.
#[inline]
#[must_use]
pub fn dot_f16_tile_rows_pinned<const ROWS: usize, const TOKENS: usize>(
    rows: [&[u8]; ROWS],
    inputs: &[&[f32]],
    out: &mut [[f32; TOKENS]; ROWS],
) -> Option<()> {
    if !matches!(ROWS, 4 | 8)
        || !valid_f16_tile_inputs::<ROWS, TOKENS>(rows, inputs)
        || !inputs[0].len().is_multiple_of(16)
    {
        return None;
    }
    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    ))]
    {
        match ROWS {
            4 => f16_aarch64::dot_tile_rows_pinned::<2, ROWS, TOKENS>(rows, inputs, out),
            8 => f16_aarch64::dot_tile_rows_pinned::<4, ROWS, TOKENS>(rows, inputs, out),
            _ => None,
        }
    }
    #[cfg(not(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    )))]
    {
        let mut tmp = [[0.0_f32; TOKENS]; ROWS];
        for (dst_row, row) in tmp.iter_mut().zip(rows) {
            for (dst, input) in dst_row.iter_mut().zip(inputs) {
                *dst = portable_f16_pinned_row(row, input)?;
            }
        }
        *out = tmp;
        Some(())
    }
}

#[cfg(any(
    test,
    not(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    ))
))]
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Retain left-to-right ordered binary64 sum of rounded binary32 products."
)]
fn portable_f16_f64_row(row: &[u8], input: &[f32]) -> Option<f32> {
    let mut sum = -0.0_f64;
    for (bytes, &act) in row.as_chunks::<2>().0.iter().zip(input) {
        let w = half_to_float(u16::from_le_bytes(*bytes));
        if !w.is_finite() {
            return None;
        }
        sum += f64::from(w * act);
    }
    let val = sum as f32;
    val.is_finite().then_some(val)
}

#[cfg(any(
    test,
    not(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    ))
))]
fn portable_f16_pinned_row(row: &[u8], input: &[f32]) -> Option<f32> {
    let mut sums = [[0.0_f32; 4]; 4];
    for (w_blk, a_blk) in row
        .as_chunks::<32>()
        .0
        .iter()
        .zip(input.as_chunks::<16>().0)
    {
        for (reg, (w_quad, a_quad)) in w_blk
            .as_chunks::<8>()
            .0
            .iter()
            .zip(a_blk.as_chunks::<4>().0)
            .enumerate()
        {
            for (lane, (w_bytes, &act)) in w_quad.as_chunks::<2>().0.iter().zip(a_quad).enumerate()
            {
                let w = half_to_float(u16::from_le_bytes(*w_bytes));
                if !w.is_finite() {
                    return None;
                }
                sums[reg][lane] = w.mul_add(act, sums[reg][lane]);
            }
        }
    }
    let mut lanes = [0.0_f32; 4];
    for (lane, dst) in lanes.iter_mut().enumerate() {
        *dst = (sums[0][lane] + sums[2][lane]) + (sums[1][lane] + sums[3][lane]);
    }
    let val = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    val.is_finite().then_some(val)
}

/// Packed rotated scalar KV consumption with portable and baseline-NEON paths.
pub mod kv_quant;
pub mod trellis;

#[cfg(target_arch = "aarch64")]
#[allow(
    unsafe_code,
    reason = "Wraps pure value-based ARM64 NEON, dotprod, and i8mm intrinsics behind safe fixed-array boundaries."
)]
mod aarch64;

/// Returns whether the current CPU supports ARMv8.2-A `dotprod` (`SDOT`).
#[cfg(target_arch = "aarch64")]
#[inline]
#[must_use]
pub fn has_dotprod() -> bool {
    aarch64::has_dotprod()
}

/// Returns false because `dotprod` is unavailable on non-ARM64 targets.
#[cfg(not(target_arch = "aarch64"))]
#[inline]
#[must_use]
pub const fn has_dotprod() -> bool {
    false
}

/// Detected `SDOT` support and optional `I8MM` selection reusable across PQ2 blocks.
///
/// Construction checks CPU/OS eligibility once. Private fields prevent safe
/// callers from creating a capability without that check; copying it preserves
/// the same process-wide feature eligibility as the ordinary dispatch APIs.
#[derive(Clone, Copy, Debug)]
pub struct Pq2Dotprod {
    _detected: (),
    #[cfg(target_arch = "aarch64")]
    i8mm: bool,
}

/// A borrowed, finite-scale PQ2 activation scope for complete-row execution.
///
/// Construction checks every activation scale once. The immutable borrow
/// prevents changes or replacement while row workers share this scope; no
/// pointer identity, allocation or retained activation copy is involved.
#[derive(Clone, Copy)]
pub struct Pq2Prepared<'a> {
    detected: Pq2Dotprod,
    activation: &'a [Pq2Activation],
}

impl Pq2Prepared<'_> {
    /// Consumes one, four, eight or sixteen complete rows using the validated scope.
    ///
    /// Row extents, packed weight scales and final results remain checked on
    /// every call. Original integer correction and floating accumulation order
    /// are identical to [`Pq2Dotprod::dot_scaled_rows`].
    /// Returns `None` for unsupported geometry or nonfinite weights/results.
    #[inline]
    #[must_use]
    pub fn dot_scaled_rows<const ROWS: usize, const FUSED: bool>(
        self,
        rows: [&[u8]; ROWS],
    ) -> Option<[f32; ROWS]> {
        if !scaled::valid_rows(rows, self.activation.len()) {
            return None;
        }
        #[cfg(target_arch = "aarch64")]
        {
            aarch64::pq2_detected_scaled_rows::<ROWS, FUSED>(self.detected, rows, self.activation)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let _ = self.detected;
            scaled::portable_validated::<ROWS, FUSED>(rows, self.activation)
        }
    }
}

/// A borrowed, finite-scale PQ2 activation chunk scope for multi-token prefill tiles.
///
/// Construction validates every chunk extent and `scale.is_finite()` once per
/// `ProjectionTile` instead of re-scanning all activation blocks on every 8-row tile.
#[derive(Clone, Copy)]
pub struct Pq2PreparedChunks<'a> {
    detected: Pq2Dotprod,
    chunks: &'a [[&'a [Pq2Activation]; 8]],
    block_count: usize,
}

impl Pq2PreparedChunks<'_> {
    /// Returns the number of 128-column blocks in each prepared activation chunk.
    #[inline]
    #[must_use]
    pub const fn block_count(self) -> usize {
        self.block_count
    }

    /// Consumes 8 complete PQ2 rows across all `1..=8` prepared chunks without
    /// re-validating activation scales on each row tile.
    ///
    /// Returns `None` for mismatched row/output extents or nonfinite weights/outputs.
    #[inline]
    #[must_use]
    pub fn dot_scaled_8x8_chunks<const FUSED: bool>(
        self,
        rows: [&[u8]; 8],
        outputs: &mut [[[f32; 8]; 8]],
    ) -> Option<()> {
        if outputs.len() != self.chunks.len() || !scaled::valid_rows(rows, self.block_count) {
            return None;
        }
        #[cfg(target_arch = "aarch64")]
        {
            aarch64::pq2_detected_dot_scaled_8x8_chunks::<FUSED>(
                self.detected,
                rows,
                self.chunks,
                outputs,
            )
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let _ = self.detected;
            scaled::portable_scaled_8x8_chunks_up_to_8::<FUSED>(rows, self.chunks, outputs)
        }
    }

    /// Consumes 8 complete PQ2 rows across all `1..=8` prepared chunks using pre-packed
    /// `(2p, 2p+1)` `I8MM` token-pair panels when `I8MM` is active.
    ///
    /// Preserves exact `0 ULP` parity and falls back cleanly when `I8MM` is absent.
    /// Returns `None` for mismatched row/panel/output extents or nonfinite weights/outputs.
    #[inline]
    #[must_use]
    pub fn dot_scaled_8x8_chunks_with_panels<const FUSED: bool>(
        self,
        rows: [&[u8]; 8],
        panels: &[[&[Pq2I8mmPairBlock]; 4]],
        outputs: &mut [[[f32; 8]; 8]],
    ) -> Option<()> {
        if outputs.len() != self.chunks.len()
            || panels.len() != self.chunks.len()
            || !scaled::valid_rows(rows, self.block_count)
            || !panels
                .iter()
                .flatten()
                .all(|pair_blocks| pair_blocks.len() == self.block_count)
        {
            return None;
        }
        #[cfg(target_arch = "aarch64")]
        {
            aarch64::pq2_detected_dot_scaled_8x8_chunks_with_panels::<FUSED>(
                self.detected,
                rows,
                self.chunks,
                panels,
                outputs,
            )
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let _ = self.detected;
            scaled::portable_scaled_8x8_chunks_up_to_8::<FUSED>(rows, self.chunks, outputs)
        }
    }
}

/// A borrowed, finite-scale PQ2 token scope for `1..=8` remainder tokens.
///
/// Construction validates every token extent and `scale.is_finite()` once per
/// projection tile instead of every 8-row tile.
#[derive(Clone, Copy)]
pub struct Pq2PreparedTokens<'a> {
    detected: Pq2Dotprod,
    tokens: &'a [&'a [Pq2Activation]],
    block_count: usize,
}

impl Pq2PreparedTokens<'_> {
    /// Consumes 8 complete PQ2 rows across the prepared `1..=8` tokens.
    ///
    /// Columns `tokens.len()..8` in `out` are zeroed. Returns `None` for
    /// mismatched row extents or nonfinite weights/outputs.
    #[inline]
    #[must_use]
    pub fn dot_scaled_8x_tokens<const FUSED: bool>(
        self,
        rows: [&[u8]; 8],
        out: &mut [[f32; 8]; 8],
    ) -> Option<()> {
        if !scaled::valid_rows(rows, self.block_count) {
            return None;
        }
        #[cfg(target_arch = "aarch64")]
        {
            aarch64::pq2_detected_dot_scaled_8x_tokens::<FUSED>(
                self.detected,
                rows,
                self.tokens,
                out,
            )
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let _ = self.detected;
            scaled::portable_scaled_8x_tokens::<FUSED>(rows, self.tokens, out)
        }
    }
}

impl Pq2Dotprod {
    /// Returns whether runtime CPU/OS detection enabled `I8MM` (`SMMLA`) on this capability.
    #[inline]
    #[must_use]
    pub const fn has_i8mm(self) -> bool {
        #[cfg(target_arch = "aarch64")]
        {
            self.i8mm
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            false
        }
    }

    /// Pre-packs a pair of aligned PQ2 activation slices `(t0, t1)` into contiguous
    /// `SMMLA` panels (`Pq2I8mmPairBlock`), clearing and reusing `out`'s allocation.
    ///
    /// Returns `None` for empty/mismatched extents or nonfinite activation scales.
    #[inline]
    #[must_use]
    pub fn pack_i8mm_pair_blocks(
        self,
        t0: &[Pq2Activation],
        t1: &[Pq2Activation],
        out: &mut Vec<Pq2I8mmPairBlock>,
    ) -> Option<()> {
        out.clear();
        self.append_i8mm_pair_blocks(t0, t1, out)
    }

    /// Appends pre-packed `SMMLA` panels (`Pq2I8mmPairBlock`) for `(t0, t1)` into `out`
    /// without clearing prior panels.
    ///
    /// Returns `None` for empty/mismatched extents or nonfinite activation scales.
    #[inline]
    #[must_use]
    pub fn append_i8mm_pair_blocks(
        self,
        t0: &[Pq2Activation],
        t1: &[Pq2Activation],
        out: &mut Vec<Pq2I8mmPairBlock>,
    ) -> Option<()> {
        let _ = self;
        if !Self::valid_pair_block_inputs(t0, t1) {
            return None;
        }
        #[cfg(target_arch = "aarch64")]
        {
            aarch64::pq2_append_i8mm_pair_blocks(t0, t1, out);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            scaled::portable_append_i8mm_pair_blocks(t0, t1, out);
        }
        Some(())
    }

    /// Pre-packs a pair of aligned PQ2 activation slices `(t0, t1)` into a pre-sized
    /// destination slice `out`, allowing parallel packing across independent token pairs.
    ///
    /// Returns `None` for empty/mismatched extents or nonfinite activation scales.
    #[inline]
    #[must_use]
    pub fn pack_i8mm_pair_blocks_into(
        self,
        t0: &[Pq2Activation],
        t1: &[Pq2Activation],
        out: &mut [Pq2I8mmPairBlock],
    ) -> Option<()> {
        let _ = self;
        if out.len() != t0.len() || !Self::valid_pair_block_inputs(t0, t1) {
            return None;
        }
        #[cfg(target_arch = "aarch64")]
        {
            aarch64::pq2_pack_i8mm_pair_blocks_into(t0, t1, out);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            scaled::portable_pack_i8mm_pair_blocks_into(t0, t1, out);
        }
        Some(())
    }

    #[inline]
    fn valid_pair_block_inputs(t0: &[Pq2Activation], t1: &[Pq2Activation]) -> bool {
        !t0.is_empty()
            && t0.len() == t1.len()
            && t0.len().checked_mul(34).is_some()
            && t0.iter().chain(t1.iter()).all(|b| b.scale.is_finite())
    }

    /// Validates `1..=8` eight-token PQ2 activation chunks once across all row tiles.
    ///
    /// Returns `None` for empty/`> 8` chunk slices, mismatched lengths, overflowing
    /// row extents, or nonfinite activation scales.
    #[inline]
    #[must_use]
    pub fn prepare_scaled_chunks<'a>(
        self,
        chunks: &'a [[&'a [Pq2Activation]; 8]],
    ) -> Option<Pq2PreparedChunks<'a>> {
        let block_count = scaled::valid_pq2_chunks_only(chunks)?;
        Some(Pq2PreparedChunks {
            detected: self,
            chunks,
            block_count,
        })
    }

    /// Validates `1..=8` PQ2 activation token slices once across all row tiles.
    ///
    /// Returns `None` for empty/`> 8` token slices, mismatched lengths, overflowing
    /// row extents, or nonfinite activation scales.
    #[inline]
    #[must_use]
    pub fn prepare_scaled_tokens<'a>(
        self,
        tokens: &'a [&'a [Pq2Activation]],
    ) -> Option<Pq2PreparedTokens<'a>> {
        let block_count = scaled::valid_pq2_tokens_only(tokens)?;
        Some(Pq2PreparedTokens {
            detected: self,
            tokens,
            block_count,
        })
    }

    /// Validates current activation scales once for any number of row groups.
    ///
    /// The returned scope borrows the activation slice immutably. Replacing or
    /// clearing prepared activation storage requires dropping every scope first.
    /// Returns `None` for empty input, overflowing extents or nonfinite scales.
    #[inline]
    #[must_use]
    pub fn prepare_scaled(self, activation: &[Pq2Activation]) -> Option<Pq2Prepared<'_>> {
        if activation.is_empty()
            || activation.len().checked_mul(34).is_none()
            || !activation.iter().all(|block| block.scale.is_finite())
        {
            return None;
        }
        Some(Pq2Prepared {
            detected: self,
            activation,
        })
    }

    /// Consumes complete packed rows inside one detected `SDOT` invocation.
    ///
    /// Supports one, four, eight or sixteen rows of original 34-byte PQ2 blocks. Every
    /// block retains its original scale product and ordered fused or separate
    /// accumulation; no weights or activations are copied or retained.
    /// Returns `None` for unsupported geometry or a nonfinite scale/result.
    #[inline]
    #[must_use]
    pub fn dot_scaled_rows<const ROWS: usize, const FUSED: bool>(
        self,
        rows: [&[u8]; ROWS],
        activation: &[Pq2Activation],
    ) -> Option<[f32; ROWS]> {
        if !scaled::valid(rows, activation) {
            return None;
        }
        #[cfg(target_arch = "aarch64")]
        {
            aarch64::pq2_detected_scaled_rows::<ROWS, FUSED>(self, rows, activation)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            scaled::portable::<ROWS, FUSED>(rows, activation)
        }
    }

    /// Detects support without enabling an optional ISA for baseline callers.
    #[inline]
    #[must_use]
    pub fn detect() -> Option<Self> {
        let detected = has_dotprod();
        #[cfg(target_arch = "aarch64")]
        let i8mm = detected && has_i8mm();
        detected.then_some(Self {
            _detected: (),
            #[cfg(target_arch = "aarch64")]
            i8mm,
        })
    }

    /// Computes a 4-row by 2-token exact integer tile from pre-transposed planes.
    #[inline]
    #[must_use]
    pub fn dot_four_rows_two_tokens(
        self,
        packed: [&[u8; 32]; 4],
        planes: [&[i8; 128]; 2],
        activation_sums: [i32; 2],
    ) -> [[i32; 2]; 4] {
        #[cfg(target_arch = "aarch64")]
        {
            aarch64::pq2_detected_dot_four_rows_two_tokens(self, packed, planes, activation_sums)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            packed.map(|row| {
                [
                    self.dot(row, planes[0], activation_sums[0]),
                    self.dot(row, planes[1], activation_sums[1]),
                ]
            })
        }
    }

    /// Computes an 8-row by 2-token exact integer tile from pre-transposed planes.
    #[inline]
    #[must_use]
    pub fn dot_eight_rows_two_tokens(
        self,
        packed: [&[u8; 32]; 8],
        planes: [&[i8; 128]; 2],
        activation_sums: [i32; 2],
    ) -> [[i32; 2]; 8] {
        #[cfg(target_arch = "aarch64")]
        {
            aarch64::pq2_detected_dot_eight_rows_two_tokens(self, packed, planes, activation_sums)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            packed.map(|row| {
                [
                    self.dot(row, planes[0], activation_sums[0]),
                    self.dot(row, planes[1], activation_sums[1]),
                ]
            })
        }
    }

    /// Computes one exact unsigned-code dot minus the supplied activation sum.
    #[inline]
    #[must_use]
    pub fn dot(self, packed: &[u8; 32], planes: &[i8; 128], activation_sum: i32) -> i32 {
        #[cfg(target_arch = "aarch64")]
        {
            aarch64::pq2_detected_dot(self, packed, planes, activation_sum)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            dot_pq2_consumption_block(packed, planes, activation_sum)
        }
    }

    /// Computes four exact dots while loading their activation planes once.
    #[inline]
    #[must_use]
    pub fn dot_four(
        self,
        packed: [&[u8; 32]; 4],
        planes: &[i8; 128],
        activation_sum: i32,
    ) -> [i32; 4] {
        #[cfg(target_arch = "aarch64")]
        {
            aarch64::pq2_detected_dot_four(self, packed, planes, activation_sum)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            packed.map(|row| self.dot(row, planes, activation_sum))
        }
    }

    /// Computes eight exact dots while loading their activation planes once.
    #[inline]
    #[must_use]
    pub fn dot_eight(
        self,
        packed: [&[u8; 32]; 8],
        planes: &[i8; 128],
        activation_sum: i32,
    ) -> [i32; 8] {
        #[cfg(target_arch = "aarch64")]
        {
            aarch64::pq2_detected_dot_eight(self, packed, planes, activation_sum)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            packed.map(|row| self.dot(row, planes, activation_sum))
        }
    }

    /// Computes an exact 4-row by 4-token tile using detected `I8MM` or `SDOT`.
    ///
    /// Activations use the same pre-deinterleaved digit planes as [`Self::dot`].
    #[inline]
    #[must_use]
    pub fn dot_four_tokens(
        self,
        packed: [&[u8; 32]; 4],
        planes: [&[i8; 128]; 4],
        activation_sums: [i32; 4],
    ) -> [[i32; 4]; 4] {
        #[cfg(target_arch = "aarch64")]
        {
            aarch64::pq2_detected_dot_four_tokens(self, packed, planes, activation_sums)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            packed.map(|row| {
                std::array::from_fn(|token| self.dot(row, planes[token], activation_sums[token]))
            })
        }
    }

    /// Computes an exact 8-row by 4-token tile using detected `I8MM` or `SDOT`.
    ///
    /// Activations use the same pre-deinterleaved digit planes as [`Self::dot`].
    #[inline]
    #[must_use]
    pub fn dot_eight_rows_four_tokens(
        self,
        packed: [&[u8; 32]; 8],
        planes: [&[i8; 128]; 4],
        activation_sums: [i32; 4],
    ) -> [[i32; 4]; 8] {
        #[cfg(target_arch = "aarch64")]
        {
            aarch64::pq2_detected_dot_eight_rows_four_tokens(self, packed, planes, activation_sums)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            packed.map(|row| {
                std::array::from_fn(|token| self.dot(row, planes[token], activation_sums[token]))
            })
        }
    }

    /// Computes an exact 8-row by 8-token tile using detected `I8MM` or `SDOT`.
    ///
    /// Activations use the same pre-deinterleaved digit planes as [`Self::dot`].
    #[inline]
    #[must_use]
    pub fn dot_eight_tokens(
        self,
        packed: [&[u8; 32]; 8],
        planes: [&[i8; 128]; 8],
        activation_sums: [i32; 8],
    ) -> [[i32; 8]; 8] {
        #[cfg(target_arch = "aarch64")]
        {
            aarch64::pq2_detected_dot_eight_tokens(self, packed, planes, activation_sums)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            packed.map(|row| {
                std::array::from_fn(|token| self.dot(row, planes[token], activation_sums[token]))
            })
        }
    }

    /// Consumes complete PQ2 rows across `1..=4` eight-token chunks (`8..=32` tokens)
    /// Consumes complete PQ2 rows across `1..=4` aligned eight-token activation
    /// chunks (`8..=32` tokens) inside a single full-K invocation, preserving bit-exact `0 ULP` parity.
    ///
    /// Returns `None` for unsupported chunk/output geometry or nonfinite scales/outputs.
    #[inline]
    #[must_use]
    pub fn dot_scaled_8x8_chunks<const FUSED: bool>(
        self,
        rows: [&[u8]; 8],
        chunks: &[[&[Pq2Activation]; 8]],
        outputs: &mut [[[f32; 8]; 8]],
    ) -> Option<()> {
        if !scaled::valid_scaled_8x8_chunks(rows, chunks, outputs.len()) {
            return None;
        }
        #[cfg(target_arch = "aarch64")]
        {
            aarch64::pq2_detected_dot_scaled_8x8_chunks::<FUSED>(self, rows, chunks, outputs)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let _ = self;
            scaled::portable_scaled_8x8_chunks::<FUSED>(rows, chunks, outputs)
        }
    }

    /// Consumes complete PQ2 rows across `1..=8` aligned eight-token activation
    /// chunks (`8..=64` tokens) inside a single full-K invocation, preserving bit-exact `0 ULP` parity.
    ///
    /// Returns `None` for unsupported chunk/output geometry or nonfinite scales/outputs.
    #[inline]
    #[must_use]
    pub fn dot_scaled_8x8_chunks_up_to_8<const FUSED: bool>(
        self,
        rows: [&[u8]; 8],
        chunks: &[[&[Pq2Activation]; 8]],
        outputs: &mut [[[f32; 8]; 8]],
    ) -> Option<()> {
        let first_blocks = chunks.first().map(|chunk| chunk[0].len())?;
        if chunks.len() > 8
            || outputs.len() != chunks.len()
            || !chunks.chunks(4).zip(outputs.chunks(4)).all(|(c, o)| {
                c[0][0].len() == first_blocks && scaled::valid_scaled_8x8_chunks(rows, c, o.len())
            })
        {
            return None;
        }
        #[cfg(target_arch = "aarch64")]
        {
            aarch64::pq2_detected_dot_scaled_8x8_chunks::<FUSED>(self, rows, chunks, outputs)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let _ = self;
            for (c, o) in chunks.chunks(4).zip(outputs.chunks_mut(4)) {
                scaled::portable_scaled_8x8_chunks::<FUSED>(rows, c, o)?;
            }
            Some(())
        }
    }

    /// Consumes complete PQ2 rows across `1..=8` tokens inside a single full-K
    /// invocation, preserving bit-exact `0 ULP` parity.
    ///
    /// Columns `tokens.len()..8` in `out` are zeroed. Returns `None` for empty
    /// or `> 8` token slices, unsupported geometry, or nonfinite scales/outputs.
    #[inline]
    #[must_use]
    pub fn dot_scaled_8x_tokens<const FUSED: bool>(
        &self,
        rows: [&[u8]; 8],
        tokens: &[&[Pq2Activation]],
        out: &mut [[f32; 8]; 8],
    ) -> Option<()> {
        if !scaled::valid_scaled_8x_tokens(rows, tokens) {
            return None;
        }
        #[cfg(target_arch = "aarch64")]
        {
            aarch64::pq2_detected_dot_scaled_8x_tokens::<FUSED>(*self, rows, tokens, out)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let _ = self;
            scaled::portable_scaled_8x_tokens::<FUSED>(rows, tokens, out)
        }
    }
}

/// Detected `SDOT` support and optional `I8MM` selection reusable across `PTQ1_0` blocks.
///
/// Construction checks CPU/OS eligibility once. Private fields prevent safe
/// callers from creating a capability without that check.
#[derive(Clone, Copy, Debug)]
pub struct Ptq1Dotprod {
    _detected: (),
    #[cfg(target_arch = "aarch64")]
    i8mm: bool,
    #[cfg(target_arch = "aarch64")]
    sve: bool,
}

/// A borrowed, finite-scale `PTQ1_0` activation scope for complete-row execution.
///
/// Construction checks every `Q8_0` activation scale and alignment once. The immutable
/// borrow prevents changes or replacement while row workers share this scope.
pub struct Ptq1Prepared<'a, A = Q8ActivationBlock> {
    detected: Ptq1Dotprod,
    activation: &'a [A],
    block_count: usize,
}

impl<A> Clone for Ptq1Prepared<'_, A> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<A> Copy for Ptq1Prepared<'_, A> {}

impl<A: Q8ActivationView> Ptq1Prepared<'_, A> {
    /// Consumes `1`, `4`, `8`, or `16` complete 28-byte `PTQ1_0` rows using the validated scope.
    ///
    /// Row extents, packed weight scales and final results remain checked on every call.
    /// Preserves exact `0 ULP` parity for both `PINNED = true` (4-lane FMA reduction)
    /// and `PINNED = false` (`SOURCE` non-FMA group reduction).
    /// Returns `None` for unsupported geometry or nonfinite weights/results.
    #[inline]
    #[must_use]
    pub fn dot_scaled_rows<const ROWS: usize, const PINNED: bool>(
        self,
        rows: [&[u8]; ROWS],
    ) -> Option<[f32; ROWS]> {
        if !scaled::valid_ptq1_row_extents(rows, self.block_count) {
            return None;
        }
        #[cfg(target_arch = "aarch64")]
        {
            aarch64::ptq1_detected_dot_scaled_rows::<ROWS, PINNED, A>(
                self.detected,
                rows,
                self.activation,
            )
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let _ = self.detected;
            scaled::portable_ptq1_rows_validated::<ROWS, PINNED, A>(rows, self.activation)
        }
    }
}

/// A borrowed, finite-scale packed `PTQ1_0` activation scope for complete-row execution.
///
/// Construction checks every activation scale once. The immutable borrow
/// prevents changes or replacement while row workers share this scope.
#[derive(Clone, Copy)]
pub struct Ptq1PackedPrepared<'a> {
    detected: Ptq1Dotprod,
    activation: &'a [Ptq1Activation],
}

impl Ptq1PackedPrepared<'_> {
    /// Consumes `1`, `4`, `8`, or `16` complete 28-byte `PTQ1_0` rows using the packed scope.
    ///
    /// Row extents, packed weight scales and final results remain checked on every call.
    /// Preserves exact `0 ULP` parity for both `PINNED = true` (4-lane FMA reduction)
    /// and `PINNED = false` (`SOURCE` non-FMA group reduction).
    /// Returns `None` for unsupported geometry or nonfinite weights/results.
    #[inline]
    #[must_use]
    pub fn dot_scaled_rows<const ROWS: usize, const PINNED: bool>(
        self,
        rows: [&[u8]; ROWS],
    ) -> Option<[f32; ROWS]> {
        if !scaled::valid_ptq1_row_extents(rows, self.activation.len()) {
            return None;
        }
        #[cfg(target_arch = "aarch64")]
        {
            aarch64::ptq1_detected_dot_packed_rows::<ROWS, PINNED>(
                self.detected,
                rows,
                self.activation,
            )
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let _ = self.detected;
            scaled::portable_ptq1_packed_rows_validated::<ROWS, PINNED>(rows, self.activation)
        }
    }
}

impl Ptq1Dotprod {
    /// Detects `PTQ1_0` `SDOT` and optional `I8MM` capability once.
    #[inline]
    #[must_use]
    pub fn detect() -> Option<Self> {
        let detected = has_dotprod();
        #[cfg(target_arch = "aarch64")]
        let i8mm = detected && has_i8mm();
        #[cfg(target_arch = "aarch64")]
        let sve = detected && aarch64::has_sve();
        detected.then_some(Self {
            _detected: (),
            #[cfg(target_arch = "aarch64")]
            i8mm,
            #[cfg(target_arch = "aarch64")]
            sve,
        })
    }

    /// Returns whether runtime CPU/OS detection enabled `I8MM` (`SMMLA`) on this capability.
    #[inline]
    #[must_use]
    pub const fn has_i8mm(self) -> bool {
        #[cfg(target_arch = "aarch64")]
        {
            self.i8mm
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            false
        }
    }

    /// Returns whether runtime CPU/OS detection enabled `SVE` on this capability.
    #[inline]
    #[must_use]
    pub const fn has_sve(self) -> bool {
        #[cfg(target_arch = "aarch64")]
        {
            self.sve
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            false
        }
    }

    /// Appends pre-packed lane-deinterleaved `SMMLA` panels ([`Ptq1PinnedI8mmPairBlock`])
    /// for `(t0, t1)` into `out` without clearing prior panels.
    ///
    /// Returns `None` for empty, unaligned, mismatched, or overflowing extents, or nonfinite scales.
    #[inline]
    #[must_use]
    pub fn append_pinned_i8mm_pair_blocks<A: Q8ActivationView>(
        self,
        t0: &[A],
        t1: &[A],
        out: &mut Vec<Ptq1PinnedI8mmPairBlock>,
    ) -> Option<()> {
        let _ = self;
        if !Self::valid_pinned_pair_inputs(t0, t1) {
            return None;
        }
        #[cfg(target_arch = "aarch64")]
        {
            aarch64::ptq1_append_pinned_i8mm_pair_blocks(t0, t1, out);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            scaled::portable_append_ptq1_pinned_i8mm_pair_blocks(t0, t1, out);
        }
        Some(())
    }

    /// Pre-packs lane-deinterleaved `SMMLA` panels ([`Ptq1PinnedI8mmPairBlock`])
    /// for `(t0, t1)` into a pre-sized destination slice `out`, allowing parallel packing.
    ///
    /// Returns `None` for empty, unaligned, mismatched, or overflowing extents, or nonfinite scales.
    #[inline]
    #[must_use]
    pub fn pack_pinned_i8mm_pair_blocks_into<A: Q8ActivationView>(
        self,
        t0: &[A],
        t1: &[A],
        out: &mut [Ptq1PinnedI8mmPairBlock],
    ) -> Option<()> {
        let _ = self;
        if out.len() != t0.len() || !Self::valid_pinned_pair_inputs(t0, t1) {
            return None;
        }
        #[cfg(target_arch = "aarch64")]
        {
            aarch64::ptq1_pack_pinned_i8mm_pair_blocks_into(t0, t1, out);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            scaled::portable_pack_ptq1_pinned_i8mm_pair_blocks_into(t0, t1, out);
        }
        Some(())
    }

    #[inline]
    fn valid_pinned_pair_inputs<A: Q8ActivationView>(t0: &[A], t1: &[A]) -> bool {
        !t0.is_empty()
            && t0.len() == t1.len()
            && t0.len().is_multiple_of(4)
            && (t0.len() / 4).checked_mul(28).is_some()
            && t0.iter().chain(t1.iter()).all(|b| b.scale().is_finite())
    }

    /// Validates current `Q8_0` activation scales once for any number of `PTQ1_0` row groups.
    ///
    /// Returns `None` for empty/unaligned inputs, overflowing extents, or nonfinite scales.
    #[inline]
    #[must_use]
    pub fn prepare_scaled(
        self,
        activation: &[Q8ActivationBlock],
    ) -> Option<Ptq1Prepared<'_, Q8ActivationBlock>> {
        self.prepare_scaled_view(activation)
    }

    /// Validates packed [`Ptq1Activation`] scales once for any number of `PTQ1_0` row groups.
    ///
    /// Returns `None` for empty inputs, overflowing extents, or nonfinite scales.
    #[inline]
    #[must_use]
    pub fn prepare_packed(self, activation: &[Ptq1Activation]) -> Option<Ptq1PackedPrepared<'_>> {
        scaled::valid_ptq1_packed_activation(activation)?;
        Some(Ptq1PackedPrepared {
            detected: self,
            activation,
        })
    }

    /// Validates any [`Q8ActivationView`] slice once for `PTQ1_0` row groups.
    ///
    /// Returns `None` for empty/unaligned inputs, overflowing extents, or nonfinite scales.
    #[inline]
    #[must_use]
    pub fn prepare_scaled_view<A: Q8ActivationView>(
        self,
        activation: &[A],
    ) -> Option<Ptq1Prepared<'_, A>> {
        let block_count = scaled::valid_ptq1_activation(activation)?;
        Some(Ptq1Prepared {
            detected: self,
            activation,
            block_count,
        })
    }

    /// Consumes `1`, `4`, `8`, or `16` complete 28-byte `PTQ1_0` rows inside one
    /// full-K invocation using [`Q8ActivationBlock`] slices.
    ///
    /// Preserves exact `0 ULP` parity for both `PINNED = true` (4-lane FMA
    /// reduction) and `PINNED = false` (`SOURCE` non-FMA group reduction).
    /// Returns `None` for unsupported geometry or nonfinite scales/outputs.
    #[inline]
    #[must_use]
    pub fn dot_scaled_rows<const ROWS: usize, const PINNED: bool>(
        self,
        rows: [&[u8]; ROWS],
        activation: &[Q8ActivationBlock],
    ) -> Option<[f32; ROWS]> {
        self.dot_scaled_rows_view::<ROWS, PINNED, Q8ActivationBlock>(rows, activation)
    }

    /// Consumes `1`, `4`, `8`, or `16` complete 28-byte `PTQ1_0` rows inside one
    /// full-K invocation over any [`Q8ActivationView`] slice.
    ///
    /// Preserves exact `0 ULP` parity for both `PINNED = true` and `PINNED = false`.
    /// Returns `None` for unsupported geometry or nonfinite scales/outputs.
    #[inline]
    #[must_use]
    pub fn dot_scaled_rows_view<const ROWS: usize, const PINNED: bool, A: Q8ActivationView>(
        self,
        rows: [&[u8]; ROWS],
        activation: &[A],
    ) -> Option<[f32; ROWS]> {
        if !scaled::valid_ptq1_rows(rows, activation) {
            return None;
        }
        #[cfg(target_arch = "aarch64")]
        {
            aarch64::ptq1_detected_dot_scaled_rows::<ROWS, PINNED, A>(self, rows, activation)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let _ = self;
            scaled::portable_ptq1_rows_validated::<ROWS, PINNED, A>(rows, activation)
        }
    }

    /// Consumes `8` complete 28-byte `PTQ1_0` rows across `1..=8` tokens inside
    /// one full-K invocation using [`Q8ActivationBlock`] token slices.
    ///
    /// Columns `tokens.len()..8` in `out` are zeroed. Returns `None` for empty
    /// or `> 8` token slices, unsupported geometry, or nonfinite scales/outputs.
    #[inline]
    #[must_use]
    pub fn dot_scaled_8x_tokens<const PINNED: bool>(
        &self,
        rows: [&[u8]; 8],
        tokens: &[&[Q8ActivationBlock]],
        out: &mut [[f32; 8]; 8],
    ) -> Option<()> {
        self.dot_scaled_8x_tokens_view::<PINNED, Q8ActivationBlock>(rows, tokens, out)
    }

    /// Consumes `8` complete 28-byte `PTQ1_0` rows across `1..=8` tokens inside
    /// one full-K invocation over any [`Q8ActivationView`] token slices.
    ///
    /// Columns `tokens.len()..8` in `out` are zeroed. Returns `None` for empty
    /// or `> 8` token slices, unsupported geometry, or nonfinite scales/outputs.
    #[inline]
    #[must_use]
    pub fn dot_scaled_8x_tokens_view<const PINNED: bool, A: Q8ActivationView>(
        &self,
        rows: [&[u8]; 8],
        tokens: &[&[A]],
        out: &mut [[f32; 8]; 8],
    ) -> Option<()> {
        if !scaled::valid_ptq1_8x_tokens(rows, tokens) {
            return None;
        }
        #[cfg(target_arch = "aarch64")]
        {
            aarch64::ptq1_detected_dot_scaled_8x_tokens::<PINNED, A>(*self, rows, tokens, out)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let _ = self;
            scaled::portable_ptq1_8x_tokens_validated::<PINNED, A>(rows, tokens, out)
        }
    }

    /// Consumes `8` complete 28-byte `PTQ1_0` rows across `1..=4` eight-token
    /// chunks (`8..=32` tokens) inside a single full-K invocation over any
    /// [`Q8ActivationView`] slices, preserving exact `0 ULP` parity.
    ///
    /// Returns `None` for unsupported chunk/output geometry or nonfinite scales/outputs.
    #[inline]
    #[must_use]
    pub fn dot_scaled_8x8_chunks<const PINNED: bool, A: Q8ActivationView>(
        self,
        rows: [&[u8]; 8],
        chunks: &[[&[A]; 8]],
        outputs: &mut [[[f32; 8]; 8]],
    ) -> Option<()> {
        if !scaled::valid_ptq1_8x8_chunks(rows, chunks, outputs.len()) {
            return None;
        }
        #[cfg(target_arch = "aarch64")]
        {
            aarch64::ptq1_detected_dot_scaled_8x8_chunks::<PINNED, A>(self, rows, chunks, outputs)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let _ = self;
            scaled::portable_ptq1_8x8_chunks_validated::<PINNED, A>(rows, chunks, outputs)
        }
    }

    /// Consumes complete `PTQ1_0` rows across `1..=8` aligned eight-token
    /// chunks (`8..=64` tokens) inside a single full-K invocation over any
    /// [`Q8ActivationView`] slices, preserving exact `0 ULP` parity.
    ///
    /// Returns `None` for unsupported chunk/output geometry or nonfinite scales/outputs.
    #[inline]
    #[must_use]
    pub fn dot_scaled_8x8_chunks_up_to_8<const PINNED: bool, A: Q8ActivationView>(
        self,
        rows: [&[u8]; 8],
        chunks: &[[&[A]; 8]],
        outputs: &mut [[[f32; 8]; 8]],
    ) -> Option<()> {
        let first_blocks = chunks.first().map(|chunk| chunk[0].len())?;
        if chunks.len() > 8
            || outputs.len() != chunks.len()
            || !chunks.chunks(4).zip(outputs.chunks(4)).all(|(c, o)| {
                c[0][0].len() == first_blocks && scaled::valid_ptq1_8x8_chunks(rows, c, o.len())
            })
        {
            return None;
        }
        #[cfg(target_arch = "aarch64")]
        {
            aarch64::ptq1_detected_dot_scaled_8x8_chunks::<PINNED, A>(self, rows, chunks, outputs)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let _ = self;
            for (c, o) in chunks.chunks(4).zip(outputs.chunks_mut(4)) {
                scaled::portable_ptq1_8x8_chunks_validated::<PINNED, A>(rows, c, o)?;
            }
            Some(())
        }
    }

    /// Consumes `8` complete 28-byte `PTQ1_0` rows across `1..=8` aligned eight-token
    /// chunks (`8..=64` tokens) in `PinnedSwift` mode using pre-packed lane-deinterleaved
    /// `SMMLA` panels ([`Ptq1PinnedI8mmPairBlock`]) when `I8MM` is active.
    ///
    /// Preserves exact `0 ULP` parity and falls back cleanly when `I8MM` is absent.
    /// Returns `None` for unsupported geometry, panel mismatch, or nonfinite scales/outputs.
    #[inline]
    #[must_use]
    pub fn dot_scaled_8x8_chunks_with_pinned_panels<A: Q8ActivationView>(
        self,
        rows: [&[u8]; 8],
        chunks: &[[&[A]; 8]],
        panels: &[[&[Ptq1PinnedI8mmPairBlock]; 4]],
        outputs: &mut [[[f32; 8]; 8]],
    ) -> Option<()> {
        let q8_len = scaled::valid_ptq1_8x8_chunks_up_to_8(rows, chunks, outputs.len())?;
        if panels.len() != chunks.len()
            || !panels
                .iter()
                .flatten()
                .all(|pair_blocks| pair_blocks.len() == q8_len)
        {
            return None;
        }
        #[cfg(target_arch = "aarch64")]
        {
            if self.i8mm {
                aarch64::ptq1_detected_dot_scaled_8x8_chunks_with_pinned_panels(
                    self, rows, panels, outputs,
                )
            } else {
                aarch64::ptq1_detected_dot_scaled_8x8_chunks::<true, A>(self, rows, chunks, outputs)
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let _ = self;
            for (c, o) in chunks.chunks(4).zip(outputs.chunks_mut(4)) {
                scaled::portable_ptq1_8x8_chunks_validated::<true, A>(rows, c, o)?;
            }
            Some(())
        }
    }
}

/// Detected `Q8_0` SIMD capability (`dotprod` and/or `i8mm`) reusable across full-K calls.
///
/// Construction checks CPU/OS eligibility once. Private fields prevent safe
/// callers from forging a capability without that check.
#[derive(Clone, Copy, Debug)]
pub struct Q8Dotprod {
    _detected: (),
    #[cfg(target_arch = "aarch64")]
    dotprod: bool,
    #[cfg(target_arch = "aarch64")]
    i8mm: bool,
}

/// A borrowed, finite-scale `Q8_0` activation scope for complete-row execution.
///
/// Construction validates every `Q8_0` activation scale once per projection instead
/// of re-scanning activation scales on every `1`/`4`/`8`/`16`-row tile.
pub struct Q8Prepared<'a, A = Q8ActivationBlock> {
    detected: Q8Dotprod,
    activation: &'a [A],
}

impl<A> Clone for Q8Prepared<'_, A> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<A> Copy for Q8Prepared<'_, A> {}

impl<A: Q8ActivationView> Q8Prepared<'_, A> {
    /// Consumes `1`, `4`, `8`, or `16` complete 34-byte `Q8_0` rows using the
    /// validated activation scope.
    ///
    /// Preserves exact `0 ULP` parity with scalar/NEON left-to-right non-FMA
    /// `(row_scale * act_scale) * int_dot` accumulation. Returns `None` for
    /// unsupported geometry or nonfinite weights/outputs.
    #[inline]
    #[must_use]
    pub fn dot_scaled_rows<const ROWS: usize>(self, rows: [&[u8]; ROWS]) -> Option<[f32; ROWS]> {
        if !scaled::valid_q8_0_rows_count(rows, self.activation.len()) {
            return None;
        }
        #[cfg(target_arch = "aarch64")]
        {
            aarch64::q8_0_detected_dot_scaled_rows::<ROWS, A>(self.detected, rows, self.activation)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let _ = self.detected;
            scaled::portable_q8_0_rows_validated::<ROWS, A>(rows, self.activation)
        }
    }
}

impl Q8Dotprod {
    /// Validates current [`Q8ActivationBlock`] activation scales once per projection.
    ///
    /// Returns `None` for empty input, overflowing extents, or nonfinite scales.
    #[inline]
    #[must_use]
    pub fn prepare_scaled(
        self,
        activation: &[Q8ActivationBlock],
    ) -> Option<Q8Prepared<'_, Q8ActivationBlock>> {
        self.prepare_scaled_view(activation)
    }

    /// Validates any [`Q8ActivationView`] slice once per projection.
    ///
    /// Returns `None` for empty input, overflowing extents, or nonfinite scales.
    #[inline]
    #[must_use]
    pub fn prepare_scaled_view<A: Q8ActivationView>(
        self,
        activation: &[A],
    ) -> Option<Q8Prepared<'_, A>> {
        let _ = scaled::valid_q8_0_activation(activation)?;
        Some(Q8Prepared {
            detected: self,
            activation,
        })
    }
    /// Detects `Q8_0` `SDOT` and/or `I8MM` capability once.
    #[inline]
    #[must_use]
    pub fn detect() -> Option<Self> {
        let dotprod = has_dotprod();
        let i8mm = has_i8mm();
        (dotprod || i8mm).then_some(Self {
            _detected: (),
            #[cfg(target_arch = "aarch64")]
            dotprod,
            #[cfg(target_arch = "aarch64")]
            i8mm,
        })
    }

    /// Consumes `1`, `4`, `8`, or `16` complete 34-byte `Q8_0` weight rows inside one
    /// full-K invocation using [`Q8ActivationBlock`] slices.
    ///
    /// Preserves exact `0 ULP` parity with scalar/NEON left-to-right non-FMA
    /// `(row_scale * act_scale) * int_dot` accumulation. Returns `None` for
    /// unsupported geometry or nonfinite scales/outputs.
    #[inline]
    #[must_use]
    pub fn dot_scaled_rows<const ROWS: usize>(
        self,
        rows: [&[u8]; ROWS],
        activation: &[Q8ActivationBlock],
    ) -> Option<[f32; ROWS]> {
        self.dot_scaled_rows_view::<ROWS, Q8ActivationBlock>(rows, activation)
    }

    /// Consumes `1`, `4`, `8`, or `16` complete 34-byte `Q8_0` weight rows inside one
    /// full-K invocation over any [`Q8ActivationView`] slice.
    ///
    /// Preserves exact `0 ULP` parity with scalar/NEON left-to-right non-FMA
    /// `(row_scale * act_scale) * int_dot` accumulation. Returns `None` for
    /// unsupported geometry or nonfinite scales/outputs.
    #[inline]
    #[must_use]
    pub fn dot_scaled_rows_view<const ROWS: usize, A: Q8ActivationView>(
        self,
        rows: [&[u8]; ROWS],
        activation: &[A],
    ) -> Option<[f32; ROWS]> {
        if !scaled::valid_q8_0_rows(rows, activation) {
            return None;
        }
        #[cfg(target_arch = "aarch64")]
        {
            aarch64::q8_0_detected_dot_scaled_rows::<ROWS, A>(self, rows, activation)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let _ = self;
            scaled::portable_q8_0_rows_validated::<ROWS, A>(rows, activation)
        }
    }

    /// Consumes `8` complete 34-byte `Q8_0` weight rows across `1..=8` tokens inside
    /// one full-K invocation using [`Q8ActivationBlock`] slices.
    ///
    /// Preserves exact `0 ULP` parity with scalar/NEON left-to-right non-FMA
    /// `(row_scale * act_scale) * int_dot` accumulation. Columns `tokens.len()..8`
    /// in `out` are zeroed. Returns `None` for invalid geometry or nonfinite scales/outputs.
    #[inline]
    #[must_use]
    pub fn dot_scaled_8x_tokens(
        &self,
        rows: [&[u8]; 8],
        tokens: &[&[Q8ActivationBlock]],
        out: &mut [[f32; 8]; 8],
    ) -> Option<()> {
        self.dot_scaled_8x_tokens_view::<Q8ActivationBlock>(rows, tokens, out)
    }

    /// Consumes `8` complete 34-byte `Q8_0` weight rows across `1..=8` tokens inside
    /// one full-K invocation over any [`Q8ActivationView`] token slices.
    ///
    /// Preserves exact `0 ULP` parity with scalar/NEON left-to-right non-FMA
    /// `(row_scale * act_scale) * int_dot` accumulation. Columns `tokens.len()..8`
    /// in `out` are zeroed. Returns `None` for invalid geometry or nonfinite scales/outputs.
    #[inline]
    #[must_use]
    pub fn dot_scaled_8x_tokens_view<A: Q8ActivationView>(
        &self,
        rows: [&[u8]; 8],
        tokens: &[&[A]],
        out: &mut [[f32; 8]; 8],
    ) -> Option<()> {
        if !scaled::valid_q8_0_8x_tokens(rows, tokens) {
            return None;
        }
        #[cfg(target_arch = "aarch64")]
        {
            aarch64::q8_0_detected_dot_scaled_8x_tokens(*self, rows, tokens, out)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let _ = self;
            scaled::portable_q8_0_8x_tokens_validated(rows, tokens, out)
        }
    }

    /// Consumes `8` complete 34-byte `Q8_0` weight rows across `1..=4` eight-token
    /// chunks (`8..=32` tokens) inside a single full-K invocation over any
    /// [`Q8ActivationView`] slices, preserving exact `0 ULP` non-FMA parity.
    ///
    /// Returns `None` for unsupported chunk/output geometry or nonfinite scales/outputs.
    #[inline]
    #[must_use]
    pub fn dot_scaled_8x8_chunks<A: Q8ActivationView>(
        self,
        rows: [&[u8]; 8],
        chunks: &[[&[A]; 8]],
        outputs: &mut [[[f32; 8]; 8]],
    ) -> Option<()> {
        if !scaled::valid_q8_0_8x8_chunks(rows, chunks, outputs.len()) {
            return None;
        }
        #[cfg(target_arch = "aarch64")]
        {
            aarch64::q8_0_detected_dot_scaled_8x8_chunks::<A>(self, rows, chunks, outputs)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let _ = self;
            scaled::portable_q8_0_8x8_chunks_validated::<A>(rows, chunks, outputs)
        }
    }

    /// Consumes complete `Q8_0` rows across `1..=8` aligned eight-token
    /// chunks (`8..=64` tokens) inside a single full-K invocation over any
    /// [`Q8ActivationView`] slices, preserving exact `0 ULP` non-FMA parity.
    ///
    /// Returns `None` for unsupported chunk/output geometry or nonfinite scales/outputs.
    #[inline]
    #[must_use]
    pub fn dot_scaled_8x8_chunks_up_to_8<A: Q8ActivationView>(
        self,
        rows: [&[u8]; 8],
        chunks: &[[&[A]; 8]],
        outputs: &mut [[[f32; 8]; 8]],
    ) -> Option<()> {
        let first_blocks = chunks.first().map(|chunk| chunk[0].len())?;
        if chunks.len() > 8
            || outputs.len() != chunks.len()
            || !chunks.chunks(4).zip(outputs.chunks(4)).all(|(c, o)| {
                c[0][0].len() == first_blocks && scaled::valid_q8_0_8x8_chunks(rows, c, o.len())
            })
        {
            return None;
        }
        #[cfg(target_arch = "aarch64")]
        {
            aarch64::q8_0_detected_dot_scaled_8x8_chunks::<A>(self, rows, chunks, outputs)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let _ = self;
            for (c, o) in chunks.chunks(4).zip(outputs.chunks_mut(4)) {
                scaled::portable_q8_0_8x8_chunks_validated::<A>(rows, c, o)?;
            }
            Some(())
        }
    }
}

/// Returns whether the current CPU supports ARMv8.2-A `i8mm` (`SMMLA`).
#[cfg(target_arch = "aarch64")]
#[inline]
#[must_use]
pub fn has_i8mm() -> bool {
    aarch64::has_i8mm()
}

/// Returns false because `i8mm` is unavailable on non-ARM64 targets.
#[cfg(not(target_arch = "aarch64"))]
#[inline]
#[must_use]
pub const fn has_i8mm() -> bool {
    false
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
const fn extract_trit_scalar(v: u8) -> i8 {
    if v < 86 {
        -1
    } else if v < 171 {
        0
    } else {
        1
    }
}

/// Decodes all 128 signed trits (`-1..=1`) of a 28-byte `PTQ1_0` block into
/// eight 16-element `i8` vectors without 16-bit widening or serial dependencies.
#[inline]
#[must_use]
pub fn decode_ptq1_trits_i8(packed: &[u8; 28]) -> [[i8; 16]; 8] {
    #[cfg(target_arch = "aarch64")]
    {
        aarch64::decode_ptq1_trits_i8(packed)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        const POW3: [u8; 5] = [1, 3, 9, 27, 81];
        const POW3_TAIL: [u8; 8] = [1, 1, 3, 3, 9, 9, 27, 27];
        let mut out = [[0_i8; 16]; 8];
        for (k, &mul) in POW3.iter().enumerate() {
            for m in 0..16 {
                out[k][m] = extract_trit_scalar(packed[m].wrapping_mul(mul));
            }
        }
        for m in 0..8 {
            out[5][m] = extract_trit_scalar(packed[16 + m]);
            out[5][8 + m] = extract_trit_scalar(packed[16 + m].wrapping_mul(3));
            out[6][m] = extract_trit_scalar(packed[16 + m].wrapping_mul(9));
            out[6][8 + m] = extract_trit_scalar(packed[16 + m].wrapping_mul(27));
            out[7][m] = extract_trit_scalar(packed[16 + m].wrapping_mul(81));
            let qh = packed[24 + (m & 1)];
            out[7][8 + m] = extract_trit_scalar(qh.wrapping_mul(POW3_TAIL[m]));
        }
        out
    }
}

/// Computes the four 8-element `DOTPROD` lane sums of two 16-element `i8`
/// weight halves against two 16-element `i8` activation halves:
/// `lane[l] = sum_{k=0..4} (w0[4*l+k]*a0[4*l+k] + w1[4*l+k]*a1[4*l+k])`.
#[inline]
#[must_use]
pub fn dot32_s8_lanes(w0: &[i8; 16], w1: &[i8; 16], a0: &[i8; 16], a1: &[i8; 16]) -> [i32; 4] {
    #[cfg(target_arch = "aarch64")]
    if let Some(lanes) = aarch64::dot32_s8_lanes(w0, w1, a0, a1) {
        return lanes;
    }
    std::array::from_fn(|lane| {
        let base = lane * 4;
        (0..4)
            .map(|k| {
                i32::from(w0[base + k]) * i32::from(a0[base + k])
                    + i32::from(w1[base + k]) * i32::from(a1[base + k])
            })
            .sum()
    })
}

/// Computes the 32-element signed integer dot product of two 16-element `i8`
/// weight halves against two 16-element `i8` activation halves.
#[inline]
#[must_use]
pub fn dot32_s8_sum(w0: &[i8; 16], w1: &[i8; 16], a0: &[i8; 16], a1: &[i8; 16]) -> i32 {
    #[cfg(target_arch = "aarch64")]
    if let Some(sum) = aarch64::dot32_s8_sum(w0, w1, a0, a1) {
        return sum;
    }
    let lanes = dot32_s8_lanes(w0, w1, a0, a1);
    lanes[0] + lanes[1] + lanes[2] + lanes[3]
}

/// Unpacks a 28-byte `PTQ1_0` block in 8-bit registers and computes the four
/// 4-lane `DOTPROD` sums against four 32-element `Q8_0` activation groups.
#[inline]
#[must_use]
pub fn dot_ptq1_block_lanes(packed: &[u8; 28], activations: [&[i8; 32]; 4]) -> [[i32; 4]; 4] {
    #[cfg(target_arch = "aarch64")]
    if let Some(lanes) = aarch64::dot_ptq1_block_lanes(packed, activations) {
        return lanes;
    }
    let trits = decode_ptq1_trits_i8(packed);
    std::array::from_fn(|group| {
        let halves = activations[group].as_chunks::<16>().0;
        dot32_s8_lanes(
            &trits[group * 2],
            &trits[group * 2 + 1],
            &halves[0],
            &halves[1],
        )
    })
}

/// Unpacks a 28-byte `PTQ1_0` block in 8-bit registers and computes the four
/// 32-element group sums against four 32-element `Q8_0`/`Q8_K` activation slices.
#[inline]
#[must_use]
pub fn dot_ptq1_block_groups(packed: &[u8; 28], activations: [&[i8; 32]; 4]) -> [i32; 4] {
    #[cfg(target_arch = "aarch64")]
    if let Some(groups) = aarch64::dot_ptq1_block_groups(packed, activations) {
        return groups;
    }
    let trits = decode_ptq1_trits_i8(packed);
    std::array::from_fn(|group| {
        let halves = activations[group].as_chunks::<16>().0;
        dot32_s8_sum(
            &trits[group * 2],
            &trits[group * 2 + 1],
            &halves[0],
            &halves[1],
        )
    })
}

/// Unpacks four 28-byte `PTQ1_0` blocks and computes their four 4-lane
/// `DOTPROD` sums against four shared 32-element `Q8_0` activation groups.
#[inline]
#[must_use]
pub fn dot_ptq1_four_rows_lanes(
    packed: [&[u8; 28]; 4],
    activations: [&[i8; 32]; 4],
) -> [[[i32; 4]; 4]; 4] {
    #[cfg(target_arch = "aarch64")]
    if let Some(lanes) = aarch64::dot_ptq1_four_rows_lanes(packed, activations) {
        return lanes;
    }
    packed.map(|row| dot_ptq1_block_lanes(row, activations))
}

/// Unpacks four 28-byte `PTQ1_0` blocks and computes their four 32-element
/// group sums against four shared 32-element `Q8_0` activation groups.
#[inline]
#[must_use]
pub fn dot_ptq1_four_rows_groups(
    packed: [&[u8; 28]; 4],
    activations: [&[i8; 32]; 4],
) -> [[i32; 4]; 4] {
    #[cfg(target_arch = "aarch64")]
    if let Some(groups) = aarch64::dot_ptq1_four_rows_groups(packed, activations) {
        return groups;
    }
    packed.map(|row| dot_ptq1_block_groups(row, activations))
}

/// Unpacks eight 28-byte `PTQ1_0` blocks and computes their four 4-lane
/// `DOTPROD` sums against four shared 32-element `Q8_0` activation groups.
#[inline]
#[must_use]
pub fn dot_ptq1_eight_rows_lanes(
    packed: [&[u8; 28]; 8],
    activations: [&[i8; 32]; 4],
) -> [[[i32; 4]; 4]; 8] {
    #[cfg(target_arch = "aarch64")]
    if let Some(lanes) = aarch64::dot_ptq1_eight_rows_lanes(packed, activations) {
        return lanes;
    }
    packed.map(|row| dot_ptq1_block_lanes(row, activations))
}

/// Unpacks eight 28-byte `PTQ1_0` blocks and computes their four 32-element
/// group sums against four shared 32-element `Q8_0` activation groups.
#[inline]
#[must_use]
pub fn dot_ptq1_eight_rows_groups(
    packed: [&[u8; 28]; 8],
    activations: [&[i8; 32]; 4],
) -> [[i32; 4]; 8] {
    #[cfg(target_arch = "aarch64")]
    if let Some(groups) = aarch64::dot_ptq1_eight_rows_groups(packed, activations) {
        return groups;
    }
    packed.map(|row| dot_ptq1_block_groups(row, activations))
}

/// Four-row by four-token by four-group by four-lane `DOTPROD` sums (`[row][token][group][lane]`).
pub type Ptq1FourRowFourTokenLanes = [[[[i32; 4]; 4]; 4]; 4];

/// Four-row by four-token by four-group 32-element integer dot sums (`[row][token][group]`).
pub type Ptq1FourRowFourTokenGroups = [[[i32; 4]; 4]; 4];

/// Eight-row by four-token by four-group by four-lane `DOTPROD` sums (`[row][token][group][lane]`).
pub type Ptq1EightRowFourTokenLanes = [[[[i32; 4]; 4]; 4]; 8];

/// Eight-row by four-token by four-group 32-element integer dot sums (`[row][token][group]`).
pub type Ptq1EightRowFourTokenGroups = [[[i32; 4]; 4]; 8];

/// Eight-row by eight-token by four-group by four-lane `DOTPROD` sums (`[row][token][group][lane]`).
pub type Ptq1EightRowEightTokenLanes = [[[[i32; 4]; 4]; 8]; 8];

/// Eight-row by eight-token by four-group 32-element integer dot sums (`[row][token][group]`).
pub type Ptq1EightRowEightTokenGroups = [[[i32; 4]; 8]; 8];

/// Computes the 4-row by 1-token by 4-group 4-lane `DOTPROD` sums from
/// pre-decoded `PTQ1_0` trit rows and one token of `Q8_0` activation groups.
#[inline]
#[must_use]
pub fn dot_ptq1_4x1_lanes(
    trits: &[[[i8; 16]; 8]; 4],
    activations: [&[i8; 32]; 4],
) -> [[[i32; 4]; 4]; 4] {
    #[cfg(target_arch = "aarch64")]
    if let Some(lanes) = aarch64::dot_ptq1_4x1_lanes(trits, activations) {
        return lanes;
    }
    std::array::from_fn(|r| {
        std::array::from_fn(|group| {
            let halves = activations[group].as_chunks::<16>().0;
            dot32_s8_lanes(
                &trits[r][group * 2],
                &trits[r][group * 2 + 1],
                &halves[0],
                &halves[1],
            )
        })
    })
}

/// Computes the 4-row by 1-token by 4-group 32-element integer dot sums from
/// pre-decoded `PTQ1_0` trit rows and one token of `Q8_0` activation groups.
#[inline]
#[must_use]
pub fn dot_ptq1_4x1_groups(
    trits: &[[[i8; 16]; 8]; 4],
    activations: [&[i8; 32]; 4],
) -> [[i32; 4]; 4] {
    #[cfg(target_arch = "aarch64")]
    if let Some(groups) = aarch64::dot_ptq1_4x1_groups(trits, activations) {
        return groups;
    }
    std::array::from_fn(|r| {
        std::array::from_fn(|group| {
            let halves = activations[group].as_chunks::<16>().0;
            dot32_s8_sum(
                &trits[r][group * 2],
                &trits[r][group * 2 + 1],
                &halves[0],
                &halves[1],
            )
        })
    })
}

/// Computes the 8-row by 1-token by 4-group 4-lane `DOTPROD` sums from
/// pre-decoded `PTQ1_0` trit rows and one token of `Q8_0` activation groups.
#[inline]
#[must_use]
pub fn dot_ptq1_8x1_lanes(
    trits: &[[[i8; 16]; 8]; 8],
    activations: [&[i8; 32]; 4],
) -> [[[i32; 4]; 4]; 8] {
    #[cfg(target_arch = "aarch64")]
    if let Some(lanes) = aarch64::dot_ptq1_8x1_lanes(trits, activations) {
        return lanes;
    }
    std::array::from_fn(|r| {
        std::array::from_fn(|group| {
            let halves = activations[group].as_chunks::<16>().0;
            dot32_s8_lanes(
                &trits[r][group * 2],
                &trits[r][group * 2 + 1],
                &halves[0],
                &halves[1],
            )
        })
    })
}

/// Computes the 8-row by 1-token by 4-group 32-element integer dot sums from
/// pre-decoded `PTQ1_0` trit rows and one token of `Q8_0` activation groups.
#[inline]
#[must_use]
pub fn dot_ptq1_8x1_groups(
    trits: &[[[i8; 16]; 8]; 8],
    activations: [&[i8; 32]; 4],
) -> [[i32; 4]; 8] {
    #[cfg(target_arch = "aarch64")]
    if let Some(groups) = aarch64::dot_ptq1_8x1_groups(trits, activations) {
        return groups;
    }
    std::array::from_fn(|r| {
        std::array::from_fn(|group| {
            let halves = activations[group].as_chunks::<16>().0;
            dot32_s8_sum(
                &trits[r][group * 2],
                &trits[r][group * 2 + 1],
                &halves[0],
                &halves[1],
            )
        })
    })
}

/// Computes the 16-row by 4-group 4-lane `DOTPROD` sums for packed `PTQ1_0` blocks.
#[inline]
#[must_use]
pub fn dot_ptq1_sixteen_rows_lanes(
    packed: [&[u8; 28]; 16],
    activations: [&[i8; 32]; 4],
) -> [[[i32; 4]; 4]; 16] {
    #[cfg(target_arch = "aarch64")]
    if let Some(lanes) = aarch64::dot_ptq1_sixteen_rows_lanes(packed, activations) {
        return lanes;
    }
    packed.map(|block| dot_ptq1_block_lanes(block, activations))
}

/// Computes the 16-row by 4-group 32-element integer dot sums for packed `PTQ1_0` blocks.
#[inline]
#[must_use]
pub fn dot_ptq1_sixteen_rows_groups(
    packed: [&[u8; 28]; 16],
    activations: [&[i8; 32]; 4],
) -> [[i32; 4]; 16] {
    #[cfg(target_arch = "aarch64")]
    if let Some(groups) = aarch64::dot_ptq1_sixteen_rows_groups(packed, activations) {
        return groups;
    }
    packed.map(|block| dot_ptq1_block_groups(block, activations))
}

/// Computes a 4-row by 2-token exact unsigned-code dot tile for `PQ2` weights.
#[inline]
#[must_use]
pub fn dot_pq2_four_rows_two_tokens(
    packed: [&[u8; 32]; 4],
    acts: [&[i8; 128]; 2],
    activation_sums: [i32; 2],
) -> [[i32; 2]; 4] {
    #[cfg(target_arch = "aarch64")]
    if let Some(tile) = aarch64::dot_pq2_four_rows_two_tokens(packed, acts, activation_sums) {
        return tile;
    }
    packed.map(|row| {
        [
            dot_pq2_sequential_scalar(row, acts[0], activation_sums[0]),
            dot_pq2_sequential_scalar(row, acts[1], activation_sums[1]),
        ]
    })
}

/// Computes an 8-row by 2-token exact unsigned-code dot tile for `PQ2` weights.
#[inline]
#[must_use]
pub fn dot_pq2_eight_rows_two_tokens(
    packed: [&[u8; 32]; 8],
    acts: [&[i8; 128]; 2],
    activation_sums: [i32; 2],
) -> [[i32; 2]; 8] {
    #[cfg(target_arch = "aarch64")]
    if let Some(tile) = aarch64::dot_pq2_eight_rows_two_tokens(packed, acts, activation_sums) {
        return tile;
    }
    packed.map(|row| {
        [
            dot_pq2_sequential_scalar(row, acts[0], activation_sums[0]),
            dot_pq2_sequential_scalar(row, acts[1], activation_sums[1]),
        ]
    })
}

/// Computes the 8-row by 4-token by 4-group 4-lane `DOTPROD` sums from
/// pre-decoded `PTQ1_0` trit rows and four tokens of `Q8_0` activation groups.
#[inline]
#[must_use]
pub fn dot_ptq1_8x4_lanes(
    trits: &[[[i8; 16]; 8]; 8],
    activations: [[&[i8; 32]; 4]; 4],
) -> Ptq1EightRowFourTokenLanes {
    #[cfg(target_arch = "aarch64")]
    if let Some(lanes) = aarch64::dot_ptq1_8x4_lanes(trits, activations) {
        return lanes;
    }
    std::array::from_fn(|r| {
        std::array::from_fn(|t| {
            std::array::from_fn(|group| {
                let halves = activations[t][group].as_chunks::<16>().0;
                dot32_s8_lanes(
                    &trits[r][group * 2],
                    &trits[r][group * 2 + 1],
                    &halves[0],
                    &halves[1],
                )
            })
        })
    })
}

/// Computes the 8-row by 4-token by 4-group 32-element integer dot sums from
/// pre-decoded `PTQ1_0` trit rows and four tokens of `Q8_0` activation groups.
#[inline]
#[must_use]
pub fn dot_ptq1_8x4_groups(
    trits: &[[[i8; 16]; 8]; 8],
    activations: [[&[i8; 32]; 4]; 4],
) -> Ptq1EightRowFourTokenGroups {
    #[cfg(target_arch = "aarch64")]
    if let Some(groups) = aarch64::dot_ptq1_8x4_groups(trits, activations) {
        return groups;
    }
    std::array::from_fn(|r| {
        std::array::from_fn(|t| {
            std::array::from_fn(|group| {
                let halves = activations[t][group].as_chunks::<16>().0;
                dot32_s8_sum(
                    &trits[r][group * 2],
                    &trits[r][group * 2 + 1],
                    &halves[0],
                    &halves[1],
                )
            })
        })
    })
}

/// Computes the 8-row by 8-token by 4-group 4-lane `DOTPROD` sums from
/// pre-decoded `PTQ1_0` trit rows and eight tokens of `Q8_0` activation groups.
#[inline]
#[must_use]
pub fn dot_ptq1_8x8_lanes(
    trits: &[[[i8; 16]; 8]; 8],
    activations: [[&[i8; 32]; 4]; 8],
) -> Ptq1EightRowEightTokenLanes {
    #[cfg(target_arch = "aarch64")]
    if let Some(lanes) = aarch64::dot_ptq1_8x8_lanes(trits, activations) {
        return lanes;
    }
    std::array::from_fn(|r| {
        std::array::from_fn(|t| {
            std::array::from_fn(|group| {
                let halves = activations[t][group].as_chunks::<16>().0;
                dot32_s8_lanes(
                    &trits[r][group * 2],
                    &trits[r][group * 2 + 1],
                    &halves[0],
                    &halves[1],
                )
            })
        })
    })
}

/// Computes the 8-row by 8-token by 4-group 32-element integer dot sums from
/// pre-decoded `PTQ1_0` trit rows and eight tokens of `Q8_0` activation groups.
#[inline]
#[must_use]
pub fn dot_ptq1_8x8_groups(
    trits: &[[[i8; 16]; 8]; 8],
    activations: [[&[i8; 32]; 4]; 8],
) -> Ptq1EightRowEightTokenGroups {
    #[cfg(target_arch = "aarch64")]
    if let Some(groups) = aarch64::dot_ptq1_8x8_groups(trits, activations) {
        return groups;
    }
    std::array::from_fn(|r| {
        std::array::from_fn(|t| {
            std::array::from_fn(|group| {
                let halves = activations[t][group].as_chunks::<16>().0;
                dot32_s8_sum(
                    &trits[r][group * 2],
                    &trits[r][group * 2 + 1],
                    &halves[0],
                    &halves[1],
                )
            })
        })
    })
}

/// Computes the 4-row by 4-token by 4-group 4-lane `DOTPROD` sums from
/// pre-decoded `PTQ1_0` trit rows and four tokens of `Q8_0` activation groups.
#[inline]
#[must_use]
pub fn dot_ptq1_4x4_lanes(
    trits: &[[[i8; 16]; 8]; 4],
    activations: [[&[i8; 32]; 4]; 4],
) -> Ptq1FourRowFourTokenLanes {
    #[cfg(target_arch = "aarch64")]
    if let Some(lanes) = aarch64::dot_ptq1_4x4_lanes(trits, activations) {
        return lanes;
    }
    std::array::from_fn(|r| {
        std::array::from_fn(|t| {
            std::array::from_fn(|group| {
                let halves = activations[t][group].as_chunks::<16>().0;
                dot32_s8_lanes(
                    &trits[r][group * 2],
                    &trits[r][group * 2 + 1],
                    &halves[0],
                    &halves[1],
                )
            })
        })
    })
}

/// Computes the 4-row by 4-token by 4-group 32-element integer dot sums from
/// pre-decoded `PTQ1_0` trit rows and four tokens of `Q8_0` activation groups.
#[inline]
#[must_use]
pub fn dot_ptq1_4x4_groups(
    trits: &[[[i8; 16]; 8]; 4],
    activations: [[&[i8; 32]; 4]; 4],
) -> Ptq1FourRowFourTokenGroups {
    #[cfg(target_arch = "aarch64")]
    if let Some(groups) = aarch64::dot_ptq1_4x4_groups(trits, activations) {
        return groups;
    }
    std::array::from_fn(|r| {
        std::array::from_fn(|t| {
            std::array::from_fn(|group| {
                let halves = activations[t][group].as_chunks::<16>().0;
                dot32_s8_sum(
                    &trits[r][group * 2],
                    &trits[r][group * 2 + 1],
                    &halves[0],
                    &halves[1],
                )
            })
        })
    })
}

/// Computes the 128-element `PQ2_0` dot product against digit-plane-ordered
/// `i8` activations and their precomputed `activation_sum` using `SDOT`.
#[inline]
#[must_use]
pub fn dot_pq2_consumption_block(
    packed: &[u8; 32],
    planes: &[i8; 128],
    activation_sum: i32,
) -> i32 {
    #[cfg(target_arch = "aarch64")]
    if let Some(sum) = aarch64::dot_pq2_consumption_block(packed, planes, activation_sum) {
        return sum;
    }
    let unsigned_sum: i32 = packed
        .as_chunks::<16>()
        .0
        .iter()
        .zip(planes.as_chunks::<64>().0)
        .map(|(bytes, values)| {
            bytes
                .iter()
                .enumerate()
                .map(|(index, &byte)| {
                    (0..4)
                        .map(|digit| {
                            i32::from((byte >> (digit * 2)) & 3)
                                * i32::from(values[digit * 16 + index])
                        })
                        .sum::<i32>()
                })
                .sum::<i32>()
        })
        .sum();
    unsigned_sum - activation_sum
}

/// Computes four 128-element `PQ2_0` dot products against shared digit-plane
/// `i8` activations and their precomputed `activation_sum` using `SDOT`.
#[inline]
#[must_use]
pub fn dot_pq2_four_consumption_precomputed(
    packed: [&[u8; 32]; 4],
    planes: &[i8; 128],
    activation_sum: i32,
) -> [i32; 4] {
    #[cfg(target_arch = "aarch64")]
    if let Some(sums) =
        aarch64::dot_pq2_four_consumption_precomputed(packed, planes, activation_sum)
    {
        return sums;
    }
    packed.map(|row| dot_pq2_consumption_block(row, planes, activation_sum))
}

/// Computes eight 128-element `PQ2_0` dot products against shared digit-plane
/// `i8` activations and their precomputed `activation_sum` using `SDOT`.
#[inline]
#[must_use]
pub fn dot_pq2_eight_consumption_precomputed(
    packed: [&[u8; 32]; 8],
    planes: &[i8; 128],
    activation_sum: i32,
) -> [i32; 8] {
    #[cfg(target_arch = "aarch64")]
    if let Some(sums) =
        aarch64::dot_pq2_eight_consumption_precomputed(packed, planes, activation_sum)
    {
        return sums;
    }
    packed.map(|row| dot_pq2_consumption_block(row, planes, activation_sum))
}

/// Computes four 128-element `PQ2_0` dot products against shared digit-plane
/// `i8` activations using `SDOT`.
#[inline]
#[must_use]
pub fn dot_pq2_four_consumption(packed: [&[u8; 32]; 4], planes: &[i8; 128]) -> [i32; 4] {
    #[cfg(target_arch = "aarch64")]
    if let Some(sums) = aarch64::dot_pq2_four_consumption(packed, planes) {
        return sums;
    }
    let activation_sum: i32 = planes.iter().map(|&v| i32::from(v)).sum();
    packed.map(|row| dot_pq2_consumption_block(row, planes, activation_sum))
}

/// Computes the 32-element signed integer dot product of a `Q8_0` 32-byte
/// weight payload against a 32-element `Q8_0` activation slice using `SDOT`.
#[inline]
#[must_use]
pub fn dot_q8_0_block(weights: &[u8; 32], activations: &[i8; 32]) -> i32 {
    #[cfg(target_arch = "aarch64")]
    if let Some(sum) = aarch64::dot_q8_0_block(weights, activations) {
        return sum;
    }
    weights
        .iter()
        .zip(activations)
        .map(|(&w, &a)| i32::from(i8::from_ne_bytes([w])) * i32::from(a))
        .sum()
}

/// Computes a 2-row by 2-column 32-element `Q8_0` matrix-matrix dot tile using
/// ARMv8.2-A `I8MM` (`SMMLA`) when available, or `SDOT`/scalar fallback.
#[inline]
#[must_use]
pub fn smmla_q8_0_2x2(r0: &[u8; 32], r1: &[u8; 32], c0: &[i8; 32], c1: &[i8; 32]) -> [[i32; 2]; 2] {
    #[cfg(target_arch = "aarch64")]
    if let Some(tile) = aarch64::smmla_q8_0_2x2(r0, r1, c0, c1) {
        return tile;
    }
    [
        [dot_q8_0_block(r0, c0), dot_q8_0_block(r0, c1)],
        [dot_q8_0_block(r1, c0), dot_q8_0_block(r1, c1)],
    ]
}

/// Computes a 4-row by 2-column 32-element `Q8_0` matrix-matrix dot tile using
/// ARMv8.2-A `I8MM` (`SMMLA`) or `SDOT` when available, or scalar fallback.
#[inline]
#[must_use]
pub fn smmla_q8_0_4x2(rows: [&[u8; 32]; 4], cols: [&[i8; 32]; 2]) -> [[i32; 2]; 4] {
    #[cfg(target_arch = "aarch64")]
    if let Some(tile) = aarch64::smmla_q8_0_4x2(rows, cols) {
        return tile;
    }
    rows.map(|row| [dot_q8_0_block(row, cols[0]), dot_q8_0_block(row, cols[1])])
}

/// Computes a 4-row by 4-column 32-element `Q8_0` matrix-matrix dot tile using
/// ARMv8.2-A `I8MM` (`SMMLA`) or `SDOT` when available, or scalar fallback.
#[inline]
#[must_use]
pub fn smmla_q8_0_4x4(rows: [&[u8; 32]; 4], cols: [&[i8; 32]; 4]) -> [[i32; 4]; 4] {
    #[cfg(target_arch = "aarch64")]
    if let Some(tile) = aarch64::smmla_q8_0_4x4(rows, cols) {
        return tile;
    }
    rows.map(|row| cols.map(|col| dot_q8_0_block(row, col)))
}

/// Computes an 8-row by 2-column 32-element `Q8_0` matrix-matrix dot tile using
/// ARMv8.2-A `I8MM` (`SMMLA`) or `SDOT` when available, or scalar fallback.
#[inline]
#[must_use]
pub fn smmla_q8_0_8x2(rows: [&[u8; 32]; 8], cols: [&[i8; 32]; 2]) -> [[i32; 2]; 8] {
    #[cfg(target_arch = "aarch64")]
    if let Some(tile) = aarch64::smmla_q8_0_8x2(rows, cols) {
        return tile;
    }
    rows.map(|row| [dot_q8_0_block(row, cols[0]), dot_q8_0_block(row, cols[1])])
}

/// Computes an 8-row by 4-column 32-element `Q8_0` matrix-matrix dot tile using
/// ARMv8.2-A `I8MM` (`SMMLA`) or `SDOT` when available, or scalar fallback.
#[inline]
#[must_use]
pub fn smmla_q8_0_8x4(rows: [&[u8; 32]; 8], cols: [&[i8; 32]; 4]) -> [[i32; 4]; 8] {
    #[cfg(target_arch = "aarch64")]
    if let Some(tile) = aarch64::smmla_q8_0_8x4(rows, cols) {
        return tile;
    }
    rows.map(|row| cols.map(|col| dot_q8_0_block(row, col)))
}

/// Consumes `1`, `4`, `8`, or `16` complete 34-byte `Q8_0` weight rows against
/// [`Q8ActivationBlock`] slices inside one full-K invocation.
///
/// Uses `SDOT` on ARM64 or portable fallback otherwise, preserving exact
/// `0 ULP` non-FMA rounding. Returns `None` for unsupported geometry or
/// nonfinite scales/outputs.
#[inline]
#[must_use]
pub fn dot_q8_0_scaled_rows<const ROWS: usize>(
    rows: [&[u8]; ROWS],
    activation: &[Q8ActivationBlock],
) -> Option<[f32; ROWS]> {
    dot_q8_0_scaled_rows_view::<ROWS, Q8ActivationBlock>(rows, activation)
}

/// Consumes `1`, `4`, `8`, or `16` complete 34-byte `Q8_0` weight rows against
/// generic [`Q8ActivationView`] slices inside one full-K invocation.
///
/// Uses `SDOT` on ARM64 or portable fallback otherwise, preserving exact
/// `0 ULP` non-FMA rounding. Returns `None` for unsupported geometry or
/// nonfinite scales/outputs.
#[inline]
#[must_use]
pub fn dot_q8_0_scaled_rows_view<const ROWS: usize, A: Q8ActivationView>(
    rows: [&[u8]; ROWS],
    activation: &[A],
) -> Option<[f32; ROWS]> {
    if !scaled::valid_q8_0_rows(rows, activation) {
        return None;
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(detected) = Q8Dotprod::detect() {
        return aarch64::q8_0_detected_dot_scaled_rows::<ROWS, A>(detected, rows, activation);
    }
    scaled::portable_q8_0_rows_validated::<ROWS, A>(rows, activation)
}

/// Consumes `8` complete 34-byte `Q8_0` weight rows across `1..=8` [`Q8ActivationBlock`]
/// token slices inside one full-K invocation.
///
/// Uses `I8MM`/`SDOT` on ARM64 or portable fallback otherwise, preserving exact
/// `0 ULP` non-FMA rounding. Columns `tokens.len()..8` in `out` are zeroed.
/// Returns `None` for empty or `> 8` token slices, invalid geometry, or nonfinite scales/outputs.
#[inline]
#[must_use]
pub fn dot_q8_0_scaled_8x_tokens(
    rows: [&[u8]; 8],
    tokens: &[&[Q8ActivationBlock]],
    out: &mut [[f32; 8]; 8],
) -> Option<()> {
    dot_q8_0_scaled_8x_tokens_view::<Q8ActivationBlock>(rows, tokens, out)
}

/// Consumes `8` complete 34-byte `Q8_0` weight rows across `1..=8` generic
/// [`Q8ActivationView`] token slices inside one full-K invocation.
///
/// Uses `I8MM`/`SDOT` on ARM64 or portable fallback otherwise, preserving exact
/// `0 ULP` non-FMA rounding. Columns `tokens.len()..8` in `out` are zeroed.
/// Returns `None` for empty or `> 8` token slices, invalid geometry, or nonfinite scales/outputs.
#[inline]
#[must_use]
pub fn dot_q8_0_scaled_8x_tokens_view<A: Q8ActivationView>(
    rows: [&[u8]; 8],
    tokens: &[&[A]],
    out: &mut [[f32; 8]; 8],
) -> Option<()> {
    if !scaled::valid_q8_0_8x_tokens(rows, tokens) {
        return None;
    }
    #[cfg(target_arch = "aarch64")]
    {
        aarch64::q8_0_detected_dot_scaled_8x_tokens(Q8Dotprod::detect()?, rows, tokens, out)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scaled::portable_q8_0_8x_tokens_validated(rows, tokens, out)
    }
}

/// Computes a 2-row by 2-column 32-element `i8` matrix-matrix dot tile using
/// ARMv8.2-A `I8MM` (`vmmlaq_s32`) when available, or `SDOT`/scalar fallback.
#[inline]
#[must_use]
pub fn smmla_dot32_2x2(
    r0: &[i8; 32],
    r1: &[i8; 32],
    c0: &[i8; 32],
    c1: &[i8; 32],
) -> [[i32; 2]; 2] {
    #[cfg(target_arch = "aarch64")]
    if let Some(tile) = aarch64::smmla_dot32_2x2(r0, r1, c0, c1) {
        return tile;
    }
    let r0_h = r0.as_chunks::<16>().0;
    let r1_h = r1.as_chunks::<16>().0;
    let c0_h = c0.as_chunks::<16>().0;
    let c1_h = c1.as_chunks::<16>().0;
    [
        [
            dot32_s8_sum(&r0_h[0], &r0_h[1], &c0_h[0], &c0_h[1]),
            dot32_s8_sum(&r0_h[0], &r0_h[1], &c1_h[0], &c1_h[1]),
        ],
        [
            dot32_s8_sum(&r1_h[0], &r1_h[1], &c0_h[0], &c0_h[1]),
            dot32_s8_sum(&r1_h[0], &r1_h[1], &c1_h[0], &c1_h[1]),
        ],
    ]
}

#[inline]
fn dot_pq2_sequential_scalar(
    packed: &[u8; 32],
    activation: &[i8; 128],
    activation_sum: i32,
) -> i32 {
    let unsigned_sum: i32 = packed
        .as_chunks::<16>()
        .0
        .iter()
        .zip(activation.as_chunks::<64>().0)
        .map(|(bytes, values)| {
            bytes
                .iter()
                .enumerate()
                .map(|(index, &byte)| {
                    (0..4)
                        .map(|digit| {
                            i32::from((byte >> (digit * 2)) & 3)
                                * i32::from(values[index * 4 + digit])
                        })
                        .sum::<i32>()
                })
                .sum::<i32>()
        })
        .sum();
    unsigned_sum - activation_sum
}

/// Computes four 128-element `PQ2_0` row dot products against one sequential
/// `Q8_K` 128-element activation slice and its precomputed `activation_sum`.
#[inline]
#[must_use]
pub fn dot_pq2_four_rows_one_token(
    packed: [&[u8; 32]; 4],
    activation: &[i8; 128],
    activation_sum: i32,
) -> [i32; 4] {
    #[cfg(target_arch = "aarch64")]
    if let Some(sums) = aarch64::dot_pq2_four_rows_one_token(packed, activation, activation_sum) {
        return sums;
    }
    packed.map(|row| dot_pq2_sequential_scalar(row, activation, activation_sum))
}

/// Computes a 4-row by 4-token 128-element `PQ2_0` dot tile against sequential
/// `Q8_K` 128-element activation slices and their precomputed `activation_sums`
/// using ARMv8.2-A `I8MM` (`SMMLA`) or `SDOT` when available.
#[inline]
#[must_use]
pub fn dot_pq2_four_rows_four_tokens(
    packed: [&[u8; 32]; 4],
    activations: [&[i8; 128]; 4],
    activation_sums: [i32; 4],
) -> [[i32; 4]; 4] {
    #[cfg(target_arch = "aarch64")]
    if let Some(tile) = aarch64::dot_pq2_four_rows_four_tokens(packed, activations, activation_sums)
    {
        return tile;
    }
    packed.map(|row| {
        std::array::from_fn(|tok| {
            dot_pq2_sequential_scalar(row, activations[tok], activation_sums[tok])
        })
    })
}

/// Computes a 4-row by 4-token 128-element `PQ2_0` dot tile against
/// pre-deinterleaved digit-plane `i8` activations and their precomputed
/// `activation_sums` using ARMv8.2-A `I8MM` (`SMMLA`) or `SDOT` when available.
#[inline]
#[must_use]
pub fn dot_pq2_four_rows_four_tokens_precomputed(
    packed: [&[u8; 32]; 4],
    planes: [&[i8; 128]; 4],
    activation_sums: [i32; 4],
) -> [[i32; 4]; 4] {
    #[cfg(target_arch = "aarch64")]
    if let Some(tile) =
        aarch64::dot_pq2_four_rows_four_tokens_precomputed(packed, planes, activation_sums)
    {
        return tile;
    }
    packed.map(|row| {
        std::array::from_fn(|tok| dot_pq2_consumption_block(row, planes[tok], activation_sums[tok]))
    })
}

/// Computes an 8-row by 4-token 128-element `PQ2_0` dot tile against
/// pre-deinterleaved digit-plane `i8` activations and their precomputed
/// `activation_sums` using ARMv8.2-A `I8MM` (`SMMLA`) or `SDOT` when available.
#[inline]
#[must_use]
pub fn dot_pq2_eight_rows_four_tokens_precomputed(
    packed: [&[u8; 32]; 8],
    planes: [&[i8; 128]; 4],
    activation_sums: [i32; 4],
) -> [[i32; 4]; 8] {
    #[cfg(target_arch = "aarch64")]
    if let Some(tile) =
        aarch64::dot_pq2_eight_rows_four_tokens_precomputed(packed, planes, activation_sums)
    {
        return tile;
    }
    packed.map(|row| {
        std::array::from_fn(|tok| dot_pq2_consumption_block(row, planes[tok], activation_sums[tok]))
    })
}

/// Computes an 8-row by 8-token 128-element `PQ2_0` dot tile against
/// pre-deinterleaved digit-plane `i8` activations and their precomputed
/// `activation_sums` using ARMv8.2-A `I8MM` (`SMMLA`) or `SDOT` when available.
#[inline]
#[must_use]
pub fn dot_pq2_eight_rows_eight_tokens_precomputed(
    packed: [&[u8; 32]; 8],
    planes: [&[i8; 128]; 8],
    activation_sums: [i32; 8],
) -> [[i32; 8]; 8] {
    #[cfg(target_arch = "aarch64")]
    if let Some(tile) =
        aarch64::dot_pq2_eight_rows_eight_tokens_precomputed(packed, planes, activation_sums)
    {
        return tile;
    }
    packed.map(|row| {
        std::array::from_fn(|tok| dot_pq2_consumption_block(row, planes[tok], activation_sums[tok]))
    })
}

/// Computes an 8-row by 8-token 128-element `PQ2_0` dot tile against sequential
/// `Q8_K` 128-element activation slices and their precomputed `activation_sums`
/// using ARMv8.2-A `I8MM` (`SMMLA`) or `SDOT` when available.
#[inline]
#[must_use]
pub fn dot_pq2_eight_rows_eight_tokens(
    packed: [&[u8; 32]; 8],
    activations: [&[i8; 128]; 8],
    activation_sums: [i32; 8],
) -> [[i32; 8]; 8] {
    #[cfg(target_arch = "aarch64")]
    if let Some(tile) =
        aarch64::dot_pq2_eight_rows_eight_tokens(packed, activations, activation_sums)
    {
        return tile;
    }
    packed.map(|row| {
        std::array::from_fn(|tok| {
            dot_pq2_sequential_scalar(row, activations[tok], activation_sums[tok])
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn x1_pq2_detected_capability_preserves_four_symbols_and_extremes() {
        let detected = Pq2Dotprod::detect();
        assert_eq!(detected.is_some(), has_dotprod());
        let Some(detected) = detected else {
            return;
        };
        for phase in 0..=u8::MAX {
            let rows: [[u8; 32]; 8] = std::array::from_fn(|row| {
                std::array::from_fn(|index| {
                    phase
                        .wrapping_add(row.to_le_bytes()[0].wrapping_mul(37))
                        .wrapping_add(index.to_le_bytes()[0].wrapping_mul(73))
                })
            });
            for values in [
                [i8::MIN; 128],
                [i8::MAX; 128],
                std::array::from_fn(|index: usize| {
                    i8::from_ne_bytes([phase.wrapping_add(index.to_le_bytes()[0])])
                }),
            ] {
                let sum = values.iter().map(|&value| i32::from(value)).sum();
                let expected: [i32; 8] = rows.map(|row| {
                    row.iter()
                        .enumerate()
                        .map(|(index, byte)| {
                            (0..4)
                                .map(|digit| {
                                    let weight = i32::from((byte >> (digit * 2)) & 3) - 1;
                                    weight
                                        * i32::from(
                                            values[index / 16 * 64 + digit * 16 + index % 16],
                                        )
                                })
                                .sum::<i32>()
                        })
                        .sum()
                });
                let refs = std::array::from_fn(|row| &rows[row]);
                assert_eq!(detected.dot_eight(refs, &values, sum), expected);
                assert_eq!(
                    detected.dot_four([refs[0], refs[1], refs[2], refs[3]], &values, sum),
                    [expected[0], expected[1], expected[2], expected[3]]
                );
                for (row, expected) in refs.into_iter().zip(expected) {
                    assert_eq!(detected.dot(row, &values, sum), expected);
                }
            }
        }
    }

    fn reference_ptq1_trits(packed: &[u8; 28]) -> [i8; 128] {
        let mut digits = [[0_u8; 5]; 26];
        for (idx, &b) in packed[..26].iter().enumerate() {
            let mut frac = u16::from(b);
            for slot in &mut digits[idx] {
                frac *= 3;
                *slot = (frac >> 8).to_le_bytes()[0];
                frac &= 255;
            }
        }
        std::array::from_fn(|index| {
            let (lane, digit) = if index < 80 {
                (index % 16, index / 16)
            } else if index < 120 {
                (16 + (index - 80) % 8, (index - 80) / 8)
            } else {
                (24 + (index - 120) % 2, (index - 120) / 2)
            };
            i8::try_from(i16::from(digits[lane][digit]) - 1).unwrap_or(0)
        })
    }

    #[test]
    fn exhaustive_ptq1_u8_unpack_matches_reference_for_all_bytes() {
        for pos in 0..26 {
            for byte in 0..=u8::MAX {
                let mut packed = [0_u8; 28];
                packed[pos] = byte;
                let expected = reference_ptq1_trits(&packed);
                let decoded = decode_ptq1_trits_i8(&packed);
                for (group_idx, chunk) in decoded.iter().enumerate() {
                    assert_eq!(
                        chunk.as_slice(),
                        &expected[group_idx * 16..(group_idx + 1) * 16],
                        "mismatch at pos={pos}, byte={byte}, group={group_idx}"
                    );
                }
            }
        }
    }

    #[test]
    fn sdot_and_smmla_match_scalar_reference() {
        let r0: [i8; 32] = std::array::from_fn(|i| {
            i8::from_ne_bytes([i.to_le_bytes()[0].wrapping_mul(53).wrapping_add(11)])
        });
        let r1: [i8; 32] = std::array::from_fn(|i| {
            i8::from_ne_bytes([i.to_le_bytes()[0].wrapping_mul(97).wrapping_add(29)])
        });
        let c0: [i8; 32] = std::array::from_fn(|i| {
            i8::from_ne_bytes([i.to_le_bytes()[0].wrapping_mul(131).wrapping_add(7)])
        });
        let c1: [i8; 32] = std::array::from_fn(|i| {
            i8::from_ne_bytes([i.to_le_bytes()[0].wrapping_mul(193).wrapping_add(3)])
        });

        let tile = smmla_dot32_2x2(&r0, &r1, &c0, &c1);
        let ref_dot = |a: &[i8; 32], b: &[i8; 32]| -> i32 {
            a.iter()
                .zip(b)
                .map(|(&x, &y)| i32::from(x) * i32::from(y))
                .sum()
        };
        assert_eq!(tile[0][0], ref_dot(&r0, &c0));
        assert_eq!(tile[0][1], ref_dot(&r0, &c1));
        assert_eq!(tile[1][0], ref_dot(&r1, &c0));
        assert_eq!(tile[1][1], ref_dot(&r1, &c1));
    }

    #[test]
    fn pq2_multi_row_multi_token_tiles_match_scalar_reference() {
        let rows: [[u8; 32]; 8] = std::array::from_fn(|r| {
            std::array::from_fn(|i| {
                r.to_le_bytes()[0]
                    .wrapping_mul(37)
                    .wrapping_add(i.to_le_bytes()[0].wrapping_mul(73))
                    .wrapping_add(19)
            })
        });
        let acts: [[i8; 128]; 8] = std::array::from_fn(|t| {
            std::array::from_fn(|i| {
                i8::from_ne_bytes([t.to_le_bytes()[0]
                    .wrapping_mul(59)
                    .wrapping_add(i.to_le_bytes()[0].wrapping_mul(107))
                    .wrapping_add(5)])
            })
        });
        let sums: [i32; 8] = std::array::from_fn(|t| acts[t].iter().map(|&v| i32::from(v)).sum());
        let row_refs8: [&[u8; 32]; 8] = std::array::from_fn(|r| &rows[r]);
        let act_refs8: [&[i8; 128]; 8] = std::array::from_fn(|t| &acts[t]);
        let tile8 = dot_pq2_eight_rows_eight_tokens(row_refs8, act_refs8, sums);
        for r in 0..8 {
            for t in 0..8 {
                let expected = dot_pq2_sequential_scalar(&rows[r], &acts[t], sums[t]);
                assert_eq!(tile8[r][t], expected, "mismatch at 8x8 r={r}, t={t}");
            }
        }
        let row_refs4: [&[u8; 32]; 4] = [&rows[0], &rows[1], &rows[2], &rows[3]];
        let act_refs4: [&[i8; 128]; 4] = [&acts[0], &acts[1], &acts[2], &acts[3]];
        let sums4 = [sums[0], sums[1], sums[2], sums[3]];
        let tile4 = dot_pq2_four_rows_four_tokens(row_refs4, act_refs4, sums4);
        for r in 0..4 {
            assert_eq!(
                tile4[r],
                [tile8[r][0], tile8[r][1], tile8[r][2], tile8[r][3]]
            );
        }
        let col0 = dot_pq2_four_rows_one_token(row_refs4, &acts[0], sums[0]);
        for r in 0..4 {
            assert_eq!(col0[r], tile8[r][0]);
        }

        let planes: [[i8; 128]; 8] = std::array::from_fn(|t| {
            std::array::from_fn(|index| {
                acts[t][(index / 64) * 64 + (index % 16) * 4 + (index / 16) % 4]
            })
        });
        let plane_refs8: [&[i8; 128]; 8] = std::array::from_fn(|t| &planes[t]);
        let plane_refs4: [&[i8; 128]; 4] = [&planes[0], &planes[1], &planes[2], &planes[3]];
        assert_eq!(
            dot_pq2_eight_rows_eight_tokens_precomputed(row_refs8, plane_refs8, sums),
            tile8
        );
        assert_eq!(
            dot_pq2_four_rows_four_tokens_precomputed(row_refs4, plane_refs4, sums4),
            tile4
        );
        let tile8x4 = dot_pq2_eight_rows_four_tokens_precomputed(row_refs8, plane_refs4, sums4);
        for r in 0..8 {
            assert_eq!(
                tile8x4[r],
                [tile8[r][0], tile8[r][1], tile8[r][2], tile8[r][3]]
            );
        }

        let detected = Pq2Dotprod::detect();
        assert_eq!(detected.is_some(), has_dotprod());
        if let Some(detected) = detected {
            // Disabling I8MM on an eligible capability also exercises the SDOT
            // tile kernels on machines where ordinary dispatch chooses I8MM.
            #[cfg(target_arch = "aarch64")]
            let sdot_only = Pq2Dotprod {
                i8mm: false,
                ..detected
            };
            #[cfg(not(target_arch = "aarch64"))]
            let sdot_only = detected;
            for capability in [detected, sdot_only] {
                assert_eq!(
                    capability.dot_eight_tokens(row_refs8, plane_refs8, sums),
                    tile8
                );
                assert_eq!(
                    capability.dot_four_tokens(row_refs4, plane_refs4, sums4),
                    tile4
                );
                assert_eq!(
                    capability.dot_eight_rows_four_tokens(row_refs8, plane_refs4, sums4),
                    tile8x4
                );
            }
        }
    }

    #[test]
    fn pq2_two_token_and_ptq1_sixteen_row_kernels_match_reference() {
        let rows: [[u8; 32]; 8] = std::array::from_fn(|r| {
            std::array::from_fn(|i| {
                r.to_le_bytes()[0]
                    .wrapping_mul(31)
                    .wrapping_add(i.to_le_bytes()[0].wrapping_mul(73))
                    .wrapping_add(11)
            })
        });
        let acts: [[i8; 128]; 2] = std::array::from_fn(|t| {
            std::array::from_fn(|i| {
                i8::from_ne_bytes([t.to_le_bytes()[0]
                    .wrapping_mul(41)
                    .wrapping_add(i.to_le_bytes()[0].wrapping_mul(29))
                    .wrapping_sub(63)])
            })
        });
        let sums = [
            acts[0].iter().map(|&v| i32::from(v)).sum(),
            acts[1].iter().map(|&v| i32::from(v)).sum(),
        ];
        let row_refs8: [&[u8; 32]; 8] = std::array::from_fn(|r| &rows[r]);
        let row_refs4: [&[u8; 32]; 4] = [&rows[0], &rows[1], &rows[2], &rows[3]];
        let act_refs2: [&[i8; 128]; 2] = [&acts[0], &acts[1]];
        let expected8: [[i32; 2]; 8] = std::array::from_fn(|r| {
            [
                dot_pq2_sequential_scalar(&rows[r], &acts[0], sums[0]),
                dot_pq2_sequential_scalar(&rows[r], &acts[1], sums[1]),
            ]
        });
        assert_eq!(
            dot_pq2_eight_rows_two_tokens(row_refs8, act_refs2, sums),
            expected8
        );
        assert_eq!(
            dot_pq2_four_rows_two_tokens(row_refs4, act_refs2, sums),
            [expected8[0], expected8[1], expected8[2], expected8[3]]
        );

        let planes: [[i8; 128]; 2] = std::array::from_fn(|t| {
            std::array::from_fn(|index| {
                acts[t][(index / 64) * 64 + (index % 16) * 4 + (index / 16) % 4]
            })
        });
        let plane_refs2: [&[i8; 128]; 2] = [&planes[0], &planes[1]];
        if let Some(detected) = Pq2Dotprod::detect() {
            #[cfg(target_arch = "aarch64")]
            let sdot_only = Pq2Dotprod {
                i8mm: false,
                ..detected
            };
            #[cfg(not(target_arch = "aarch64"))]
            let sdot_only = detected;
            for capability in [detected, sdot_only] {
                assert_eq!(
                    capability.dot_eight_rows_two_tokens(row_refs8, plane_refs2, sums),
                    expected8
                );
                assert_eq!(
                    capability.dot_four_rows_two_tokens(row_refs4, plane_refs2, sums),
                    [expected8[0], expected8[1], expected8[2], expected8[3]]
                );
            }
        }

        let ptq_rows: [[u8; 28]; 16] = std::array::from_fn(|r| {
            std::array::from_fn(|i| {
                r.to_le_bytes()[0]
                    .wrapping_mul(37)
                    .wrapping_add(i.to_le_bytes()[0].wrapping_mul(83))
                    .wrapping_add(5)
            })
        });
        let ptq_acts: [[i8; 32]; 4] = std::array::from_fn(|g| {
            std::array::from_fn(|i| {
                i8::from_ne_bytes([g.to_le_bytes()[0]
                    .wrapping_mul(53)
                    .wrapping_add(i.to_le_bytes()[0].wrapping_mul(19))
                    .wrapping_sub(31)])
            })
        });
        let ptq_row_refs: [&[u8; 28]; 16] = std::array::from_fn(|r| &ptq_rows[r]);
        let ptq_act_refs: [&[i8; 32]; 4] = [&ptq_acts[0], &ptq_acts[1], &ptq_acts[2], &ptq_acts[3]];
        let actual_lanes = dot_ptq1_sixteen_rows_lanes(ptq_row_refs, ptq_act_refs);
        let actual_groups = dot_ptq1_sixteen_rows_groups(ptq_row_refs, ptq_act_refs);
        for r in 0..16 {
            assert_eq!(
                actual_lanes[r],
                dot_ptq1_block_lanes(&ptq_rows[r], ptq_act_refs)
            );
            assert_eq!(
                actual_groups[r],
                dot_ptq1_block_groups(&ptq_rows[r], ptq_act_refs)
            );
        }
    }

    #[test]
    #[expect(
        clippy::similar_names,
        reason = "Test compares 4x4, 8x4, and 8x8 lane and group variants side by side."
    )]
    fn ptq1_4x4_lanes_and_groups_match_single_row_single_token_primitives() {
        let rows: [[u8; 28]; 4] = std::array::from_fn(|r| {
            std::array::from_fn(|i| {
                r.to_le_bytes()[0]
                    .wrapping_mul(41)
                    .wrapping_add(i.to_le_bytes()[0].wrapping_mul(89))
                    .wrapping_add(13)
            })
        });
        let trits = rows.map(|row| decode_ptq1_trits_i8(&row));
        let acts: [[[i8; 32]; 4]; 4] = std::array::from_fn(|t| {
            std::array::from_fn(|g| {
                std::array::from_fn(|i| {
                    i8::from_ne_bytes([t.to_le_bytes()[0]
                        .wrapping_mul(67)
                        .wrapping_add(g.to_le_bytes()[0].wrapping_mul(23))
                        .wrapping_add(i.to_le_bytes()[0].wrapping_mul(113))
                        .wrapping_add(7)])
                })
            })
        });
        let act_refs: [[&[i8; 32]; 4]; 4] =
            std::array::from_fn(|t| std::array::from_fn(|g| &acts[t][g]));
        let tile_lanes = dot_ptq1_4x4_lanes(&trits, act_refs);
        let tile_groups = dot_ptq1_4x4_groups(&trits, act_refs);
        for r in 0..4 {
            for t in 0..4 {
                let expected_lanes = dot_ptq1_block_lanes(&rows[r], act_refs[t]);
                let expected_groups = dot_ptq1_block_groups(&rows[r], act_refs[t]);
                assert_eq!(tile_lanes[r][t], expected_lanes);
                assert_eq!(tile_groups[r][t], expected_groups);
            }
        }
        let rows8: [[u8; 28]; 8] = std::array::from_fn(|r| {
            std::array::from_fn(|i| {
                r.to_le_bytes()[0]
                    .wrapping_mul(41)
                    .wrapping_add(i.to_le_bytes()[0].wrapping_mul(89))
                    .wrapping_add(13)
            })
        });
        let row_refs8: [&[u8; 28]; 8] = std::array::from_fn(|r| &rows8[r]);
        let trits8 = rows8.map(|row| decode_ptq1_trits_i8(&row));
        let tile8_lanes = dot_ptq1_8x4_lanes(&trits8, act_refs);
        let tile8_groups = dot_ptq1_8x4_groups(&trits8, act_refs);
        let eight_lanes = dot_ptq1_eight_rows_lanes(row_refs8, act_refs[0]);
        let eight_groups = dot_ptq1_eight_rows_groups(row_refs8, act_refs[0]);
        let eight_1_lanes = dot_ptq1_8x1_lanes(&trits8, act_refs[0]);
        let eight_1_groups = dot_ptq1_8x1_groups(&trits8, act_refs[0]);
        let four_1_lanes = dot_ptq1_4x1_lanes(&trits, act_refs[0]);
        let four_1_groups = dot_ptq1_4x1_groups(&trits, act_refs[0]);
        for r in 0..8 {
            assert_eq!(eight_lanes[r], tile8_lanes[r][0]);
            assert_eq!(eight_groups[r], tile8_groups[r][0]);
            assert_eq!(eight_1_lanes[r], tile8_lanes[r][0]);
            assert_eq!(eight_1_groups[r], tile8_groups[r][0]);
            for t in 0..4 {
                assert_eq!(
                    tile8_lanes[r][t],
                    dot_ptq1_block_lanes(&rows8[r], act_refs[t])
                );
                assert_eq!(
                    tile8_groups[r][t],
                    dot_ptq1_block_groups(&rows8[r], act_refs[t])
                );
            }
        }
        for r in 0..4 {
            assert_eq!(four_1_lanes[r], tile_lanes[r][0]);
            assert_eq!(four_1_groups[r], tile_groups[r][0]);
        }

        let acts8: [[[i8; 32]; 4]; 8] = std::array::from_fn(|t| {
            std::array::from_fn(|g| {
                std::array::from_fn(|i| {
                    i8::from_ne_bytes([t.to_le_bytes()[0]
                        .wrapping_mul(67)
                        .wrapping_add(g.to_le_bytes()[0].wrapping_mul(23))
                        .wrapping_add(i.to_le_bytes()[0].wrapping_mul(113))
                        .wrapping_add(7)])
                })
            })
        });
        let act_refs8: [[&[i8; 32]; 4]; 8] =
            std::array::from_fn(|t| std::array::from_fn(|g| &acts8[t][g]));
        let tile8x8_lanes = dot_ptq1_8x8_lanes(&trits8, act_refs8);
        let tile8x8_groups = dot_ptq1_8x8_groups(&trits8, act_refs8);
        for r in 0..8 {
            for t in 0..8 {
                assert_eq!(
                    tile8x8_lanes[r][t],
                    dot_ptq1_block_lanes(&rows8[r], act_refs8[t])
                );
                assert_eq!(
                    tile8x8_groups[r][t],
                    dot_ptq1_block_groups(&rows8[r], act_refs8[t])
                );
            }
        }
    }

    #[test]
    fn q8_0_multi_row_multi_token_tiles_match_single_block_dot() {
        let rows: [[u8; 32]; 8] = std::array::from_fn(|r| {
            std::array::from_fn(|i| {
                r.to_le_bytes()[0]
                    .wrapping_mul(47)
                    .wrapping_add(i.to_le_bytes()[0].wrapping_mul(83))
                    .wrapping_add(29)
            })
        });
        let cols: [[i8; 32]; 4] = std::array::from_fn(|c| {
            std::array::from_fn(|i| {
                i8::from_ne_bytes([c.to_le_bytes()[0]
                    .wrapping_mul(61)
                    .wrapping_add(i.to_le_bytes()[0].wrapping_mul(101))
                    .wrapping_add(17)])
            })
        });
        let rows4: [&[u8; 32]; 4] = [&rows[0], &rows[1], &rows[2], &rows[3]];
        let rows8: [&[u8; 32]; 8] = std::array::from_fn(|r| &rows[r]);
        let cols2: [&[i8; 32]; 2] = [&cols[0], &cols[1]];
        let cols4: [&[i8; 32]; 4] = std::array::from_fn(|c| &cols[c]);

        let tile4x2 = smmla_q8_0_4x2(rows4, cols2);
        let tile4x4 = smmla_q8_0_4x4(rows4, cols4);
        let tile8x2 = smmla_q8_0_8x2(rows8, cols2);
        let tile8x4 = smmla_q8_0_8x4(rows8, cols4);
        for r in 0..8 {
            for c in 0..4 {
                let expected = dot_q8_0_block(&rows[r], &cols[c]);
                assert_eq!(tile8x4[r][c], expected);
                if c < 2 {
                    assert_eq!(tile8x2[r][c], expected);
                }
                if r < 4 {
                    assert_eq!(tile4x4[r][c], expected);
                    if c < 2 {
                        assert_eq!(tile4x2[r][c], expected);
                    }
                }
            }
        }
    }

    fn assert_64_token_chunk_bits(expected: &[[[f32; 8]; 8]], actual: &[[[f32; 8]; 8]]) {
        assert_eq!(expected.len(), actual.len());
        for (exp_chunk, act_chunk) in expected.iter().zip(actual.iter()) {
            for r in 0..8 {
                for t in 0..8 {
                    assert_eq!(act_chunk[r][t].to_bits(), exp_chunk[r][t].to_bits());
                }
            }
        }
    }

    #[test]
    fn pq2_64_token_chunks_match_single_chunk_bits() {
        const BLOCKS: usize = 3;
        let pq2_rows: [[u8; 34 * BLOCKS]; 8] = std::array::from_fn(|r| {
            let mut row = [0_u8; 34 * BLOCKS];
            for b in 0..BLOCKS {
                let off = b * 34;
                let idx_f = f32::from(u8::try_from(r + b + 1).unwrap_or(1));
                let scale = idx_f.mul_add(0.0125, 0.025);
                row[off..off + 2].copy_from_slice(&float_to_half(scale).to_le_bytes());
                for (j, byte) in row[off + 2..off + 34].iter_mut().enumerate() {
                    *byte = u8::try_from((r * 19 + b * 37 + j * 11 + 5) & 0xff).unwrap_or(0);
                }
            }
            row
        });
        let pq2_row_refs: [&[u8]; 8] = std::array::from_fn(|r| pq2_rows[r].as_slice());
        let pq2_acts: Vec<Vec<Pq2Activation>> = (0..72)
            .map(|t| {
                (0..BLOCKS)
                    .map(|b| {
                        let values: [i8; 128] = std::array::from_fn(|i| {
                            let v = i32::try_from((t * 17 + b * 29 + i * 7) % 19).unwrap_or(0) - 9;
                            i8::try_from(v).unwrap_or(0)
                        });
                        let sum = values.iter().map(|&v| i32::from(v)).sum();
                        let scale_idx = f32::from(u8::try_from((t + b) % 13 + 1).unwrap_or(1));
                        let scale = scale_idx.mul_add(0.004, 0.03);
                        Pq2Activation { scale, values, sum }
                    })
                    .collect()
            })
            .collect();
        let pq2_chunks_9: [[&[Pq2Activation]; 8]; 9] =
            std::array::from_fn(|c| std::array::from_fn(|t| pq2_acts[c * 8 + t].as_slice()));
        let Some(pq2) = Pq2Dotprod::detect() else {
            return;
        };
        let mut single = [[[0.0_f32; 8]; 8]; 8];
        for (c, slot) in single.iter_mut().enumerate() {
            assert!(
                pq2.dot_scaled_8x8_chunks::<true>(
                    pq2_row_refs,
                    &pq2_chunks_9[c..=c],
                    std::slice::from_mut(slot),
                )
                .is_some()
            );
        }
        for count in 1..=8 {
            let mut wide = [[[0.0_f32; 8]; 8]; 8];
            assert!(
                pq2.dot_scaled_8x8_chunks_up_to_8::<true>(
                    pq2_row_refs,
                    &pq2_chunks_9[..count],
                    &mut wide[..count],
                )
                .is_some()
            );
            assert_64_token_chunk_bits(&single[..count], &wide[..count]);
        }
        let mut too_many = [[[0.0_f32; 8]; 8]; 9];
        assert!(
            pq2.dot_scaled_8x8_chunks_up_to_8::<true>(pq2_row_refs, &pq2_chunks_9, &mut too_many)
                .is_none()
        );
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "Compares 64-token PTQ1 and Q8_0 chunks against single-chunk outputs across shapes."
    )]
    fn ptq1_and_q8_64_token_chunks_match_single_chunk_bits() {
        const BLOCKS: usize = 3;
        let ptq1_rows: [[u8; 28 * BLOCKS]; 8] = std::array::from_fn(|r| {
            let mut row = [0_u8; 28 * BLOCKS];
            for b in 0..BLOCKS {
                let off = b * 28;
                let idx_f = f32::from(u8::try_from(r + b + 1).unwrap_or(1));
                let scale = idx_f.mul_add(0.015, 0.04);
                for (j, byte) in row[off..off + 26].iter_mut().enumerate() {
                    *byte = u8::try_from((r * 23 + b * 41 + j * 13 + 9) & 0xff).unwrap_or(0);
                }
                row[off + 26..off + 28].copy_from_slice(&float_to_half(scale).to_le_bytes());
            }
            row
        });
        let q8_rows: [[u8; 34 * BLOCKS]; 8] = std::array::from_fn(|r| {
            let mut row = [0_u8; 34 * BLOCKS];
            for b in 0..BLOCKS {
                let off = b * 34;
                let idx_f = f32::from(u8::try_from(r + b + 2).unwrap_or(2));
                let scale = idx_f.mul_add(0.01, 0.02);
                row[off..off + 2].copy_from_slice(&float_to_half(scale).to_le_bytes());
                for (j, byte) in row[off + 2..off + 34].iter_mut().enumerate() {
                    *byte = u8::try_from((r * 17 + b * 31 + j * 7 + 3) & 0xff).unwrap_or(0);
                }
            }
            row
        });
        let ptq1_row_refs: [&[u8]; 8] = std::array::from_fn(|r| ptq1_rows[r].as_slice());
        let q8_row_refs: [&[u8]; 8] = std::array::from_fn(|r| q8_rows[r].as_slice());
        let q8_acts: Vec<Vec<Q8ActivationBlock>> = (0..72)
            .map(|t| {
                (0..BLOCKS * 4)
                    .map(|b| {
                        let scale_idx = f32::from(u8::try_from((t + b) % 11 + 1).unwrap_or(1));
                        Q8ActivationBlock {
                            scale: scale_idx.mul_add(0.005, 0.02),
                            values: std::array::from_fn(|i| {
                                let v =
                                    i32::try_from((t * 13 + b * 17 + i * 5) % 21).unwrap_or(0) - 10;
                                i8::try_from(v).unwrap_or(0)
                            }),
                        }
                    })
                    .collect()
            })
            .collect();
        let ptq1_chunks_9: [[&[Q8ActivationBlock]; 8]; 9] =
            std::array::from_fn(|c| std::array::from_fn(|t| q8_acts[c * 8 + t].as_slice()));
        if let Some(ptq1) = Ptq1Dotprod::detect() {
            for pinned in [false, true] {
                let mut single = [[[0.0_f32; 8]; 8]; 8];
                for (c, slot) in single.iter_mut().enumerate() {
                    let ok = if pinned {
                        ptq1.dot_scaled_8x8_chunks::<true, _>(
                            ptq1_row_refs,
                            &ptq1_chunks_9[c..=c],
                            std::slice::from_mut(slot),
                        )
                    } else {
                        ptq1.dot_scaled_8x8_chunks::<false, _>(
                            ptq1_row_refs,
                            &ptq1_chunks_9[c..=c],
                            std::slice::from_mut(slot),
                        )
                    };
                    assert!(ok.is_some());
                }
                for count in 5..=8 {
                    let mut wide = [[[0.0_f32; 8]; 8]; 8];
                    let ok = if pinned {
                        ptq1.dot_scaled_8x8_chunks_up_to_8::<true, _>(
                            ptq1_row_refs,
                            &ptq1_chunks_9[..count],
                            &mut wide[..count],
                        )
                    } else {
                        ptq1.dot_scaled_8x8_chunks_up_to_8::<false, _>(
                            ptq1_row_refs,
                            &ptq1_chunks_9[..count],
                            &mut wide[..count],
                        )
                    };
                    assert!(ok.is_some());
                    assert_64_token_chunk_bits(&single[..count], &wide[..count]);
                }
            }
            let mut too_many = [[[0.0_f32; 8]; 8]; 9];
            assert!(
                ptq1.dot_scaled_8x8_chunks_up_to_8::<true, _>(
                    ptq1_row_refs,
                    &ptq1_chunks_9,
                    &mut too_many
                )
                .is_none()
            );
        }
        let q8_chunks_9: [[&[Q8ActivationBlock]; 8]; 9] =
            std::array::from_fn(|c| std::array::from_fn(|t| &q8_acts[c * 8 + t][..BLOCKS]));
        if let Some(q8) = Q8Dotprod::detect() {
            let mut single = [[[0.0_f32; 8]; 8]; 8];
            for (c, slot) in single.iter_mut().enumerate() {
                assert!(
                    q8.dot_scaled_8x8_chunks(
                        q8_row_refs,
                        &q8_chunks_9[c..=c],
                        std::slice::from_mut(slot)
                    )
                    .is_some()
                );
            }
            for count in 5..=8 {
                let mut wide = [[[0.0_f32; 8]; 8]; 8];
                assert!(
                    q8.dot_scaled_8x8_chunks_up_to_8(
                        q8_row_refs,
                        &q8_chunks_9[..count],
                        &mut wide[..count]
                    )
                    .is_some()
                );
                assert_64_token_chunk_bits(&single[..count], &wide[..count]);
            }
            let mut too_many = [[[0.0_f32; 8]; 8]; 9];
            assert!(
                q8.dot_scaled_8x8_chunks_up_to_8(q8_row_refs, &q8_chunks_9, &mut too_many)
                    .is_none()
            );
        }
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "Verify single-row, multi-row, and 64-token tiled F16 kernels across unpinned, pinned, and error paths."
    )]
    fn f16_single_and_tiled_kernels_match_portable_and_reject_nonfinite() {
        let make_rows = |rows_n: usize, cols: usize| -> Vec<Vec<u8>> {
            let bits = [
                0x0000_u16, 0x8000, 0x0001, 0x8001, 0x3c00, 0xbc00, 0x3e00, 0x4200,
            ];
            (0..rows_n)
                .map(|r| {
                    (0..cols)
                        .flat_map(|c| bits[(r * 5 + c * 3) % bits.len()].to_le_bytes())
                        .collect()
                })
                .collect()
        };
        let make_inputs = |tokens_n: usize, cols: usize| -> Vec<Vec<f32>> {
            let vals = [0.0_f32, -0.0, 1.5, -2.25, 0.125, -0.375, 4.5, -0.0625];
            (0..tokens_n)
                .map(|t| (0..cols).map(|c| vals[(t * 7 + c) % vals.len()]).collect())
                .collect()
        };

        for cols in [1_usize, 7, 15, 16, 17, 32, 67, 144] {
            let rows = make_rows(8, cols);
            let inputs = make_inputs(64, cols);
            let row_refs8: [&[u8]; 8] = std::array::from_fn(|r| rows[r].as_slice());
            let row_refs4: [&[u8]; 4] = std::array::from_fn(|r| rows[r].as_slice());

            for tok_count in [1_usize, 3, 5, 16, 64] {
                let in_refs: Vec<&[f32]> = inputs[..tok_count].iter().map(Vec::as_slice).collect();
                let mut out1 = [f32::NAN; 64];
                let mut out4 = [[f32::NAN; 64]; 4];
                let mut out8 = [[f32::NAN; 64]; 8];
                assert!(dot_f16_tile_row(row_refs8[0], &in_refs, &mut out1).is_some());
                assert!(dot_f16_tile_rows(row_refs4, &in_refs, &mut out4).is_some());
                assert!(dot_f16_tile_rows(row_refs8, &in_refs, &mut out8).is_some());
                for (t, &inp) in in_refs.iter().enumerate() {
                    let s1 = dot_f16_row(row_refs8[0], inp).map(f32::to_bits);
                    let s4 = dot_f16_rows(row_refs4, inp).map(|v| v.map(f32::to_bits));
                    let s8 = dot_f16_rows(row_refs8, inp).map(|v| v.map(f32::to_bits));
                    let exp: [u32; 8] = std::array::from_fn(|r| {
                        portable_f16_f64_row(row_refs8[r], inp).map_or(u32::MAX, f32::to_bits)
                    });
                    assert_eq!(s1, Some(exp[0]));
                    assert_eq!(s4, Some([exp[0], exp[1], exp[2], exp[3]]));
                    assert_eq!(s8, Some(exp));
                    assert_eq!(out1[t].to_bits(), exp[0]);
                    for r in 0..4 {
                        assert_eq!(out4[r][t].to_bits(), exp[r]);
                    }
                    for r in 0..8 {
                        assert_eq!(out8[r][t].to_bits(), exp[r]);
                    }
                }

                if cols.is_multiple_of(16) {
                    let mut p_out1 = [f32::NAN; 64];
                    let mut p_out4 = [[f32::NAN; 64]; 4];
                    let mut p_out8 = [[f32::NAN; 64]; 8];
                    assert!(dot_f16_tile_row_pinned(row_refs8[0], &in_refs, &mut p_out1).is_some());
                    assert!(dot_f16_tile_rows_pinned(row_refs4, &in_refs, &mut p_out4).is_some());
                    assert!(dot_f16_tile_rows_pinned(row_refs8, &in_refs, &mut p_out8).is_some());
                    for (t, &inp) in in_refs.iter().enumerate() {
                        let s1 = dot_f16_row_pinned(row_refs8[0], inp).map(f32::to_bits);
                        let s4 = dot_f16_rows_pinned(row_refs4, inp).map(|v| v.map(f32::to_bits));
                        let s8 = dot_f16_rows_pinned(row_refs8, inp).map(|v| v.map(f32::to_bits));
                        let exp: [u32; 8] = std::array::from_fn(|r| {
                            portable_f16_pinned_row(row_refs8[r], inp)
                                .map_or(u32::MAX, f32::to_bits)
                        });
                        assert_eq!(s1, Some(exp[0]));
                        assert_eq!(s4, Some([exp[0], exp[1], exp[2], exp[3]]));
                        assert_eq!(s8, Some(exp));
                        assert_eq!(p_out1[t].to_bits(), exp[0]);
                        for r in 0..4 {
                            assert_eq!(p_out4[r][t].to_bits(), exp[r]);
                        }
                        for r in 0..8 {
                            assert_eq!(p_out8[r][t].to_bits(), exp[r]);
                        }
                    }
                } else {
                    assert!(dot_f16_row_pinned(row_refs8[0], in_refs[0]).is_none());
                    assert!(dot_f16_rows_pinned(row_refs4, in_refs[0]).is_none());
                }
            }
        }

        // Reject NaN/Inf weights even when all activations are zero, preserving sentinel `out`.
        for bad_f16 in [0x7c00_u16, 0x7e00, 0xfc00] {
            let mut bad_rows = make_rows(8, 32);
            bad_rows[7][62..64].copy_from_slice(&bad_f16.to_le_bytes());
            let bad_refs8: [&[u8]; 8] = std::array::from_fn(|r| bad_rows[r].as_slice());
            let zeros = [0.0_f32; 32];
            let zero_refs = [zeros.as_slice(); 4];
            let sentinel = [[42.5_f32; 4]; 8];
            let mut out_u = sentinel;
            let mut out_p = sentinel;
            assert!(dot_f16_row(bad_refs8[7], &zeros).is_none());
            assert!(dot_f16_row_pinned(bad_refs8[7], &zeros).is_none());
            assert!(dot_f16_rows(bad_refs8, &zeros).is_none());
            assert!(dot_f16_rows_pinned(bad_refs8, &zeros).is_none());
            assert!(dot_f16_tile_rows(bad_refs8, &zero_refs, &mut out_u).is_none());
            assert!(dot_f16_tile_rows_pinned(bad_refs8, &zero_refs, &mut out_p).is_none());
            assert_eq!(
                out_u.map(|r| r.map(f32::to_bits)),
                sentinel.map(|r| r.map(f32::to_bits))
            );
            assert_eq!(
                out_p.map(|r| r.map(f32::to_bits)),
                sentinel.map(|r| r.map(f32::to_bits))
            );
        }
    }
}
