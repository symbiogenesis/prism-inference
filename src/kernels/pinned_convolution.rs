//! Exact pinned Windows ARM64 SSM convolution accumulation.
//!
//! Reference: Prism `73bfe6cc485554822f80c473a3c0c4f25e80f41b`,
//! `ggml/src/ggml-cpu/ops.cpp`, compiled `ggml_ssm_conv` arithmetic.
//! Upstream MIT terms are retained in `PRISM-LICENSE.txt`.

#![expect(
    clippy::redundant_pub_crate,
    reason = "Keep the exact-model convolution helper crate scoped."
)]

use crate::{Result, invalid};

/// Immutable model coefficients, prepared once without retaining a second copy.
pub(crate) struct Coefficients {
    values: Vec<f32>,
    prepared: bool,
}

impl Coefficients {
    /// Transposes eligible pinned four-tap blocks in their existing allocation.
    /// Other numerical contracts and geometries retain channel-major rows.
    pub(crate) fn new(mut values: Vec<f32>, width: usize, pinned: bool) -> Self {
        let prepared = pinned && width == 4 && values.len().is_multiple_of(16);
        if prepared {
            for block in values.as_chunks_mut::<16>().0 {
                for row in 0..4 {
                    for column in row + 1..4 {
                        block.swap(row * 4 + column, column * 4 + row);
                    }
                }
            }
        }
        Self { values, prepared }
    }

    /// Borrows all coefficients with their exact prepared-layout identity.
    pub(crate) const fn view(&self) -> CoefficientSlice<'_> {
        CoefficientSlice {
            values: self.values.as_slice(),
            prepared: self.prepared,
        }
    }

    /// Borrows an aligned head partition without allocating or changing layout.
    ///
    /// # Panics
    /// Panics if a private graph partition is out of bounds or splits a prepared block.
    pub(crate) fn range(&self, range: std::ops::Range<usize>) -> CoefficientSlice<'_> {
        // Graph head partitions are multiples of four channels. A prepared
        // view must begin/end at a complete 4-channel, 4-tap block.
        assert!(!self.prepared || (range.start.is_multiple_of(16) && range.end.is_multiple_of(16)));
        CoefficientSlice {
            values: &self.values[range],
            prepared: self.prepared,
        }
    }
}

/// Borrowed exact coefficient representation; construction stays with its owner.
#[derive(Clone, Copy)]
pub(crate) struct CoefficientSlice<'a> {
    values: &'a [f32],
    prepared: bool,
}

impl<'a> CoefficientSlice<'a> {
    #[cfg(test)]
    const fn raw(values: &'a [f32]) -> Self {
        Self {
            values,
            prepared: false,
        }
    }

    /// Generic arithmetic only receives unprepared channel-major coefficients.
    ///
    /// # Panics
    /// Panics if a private graph caller requests source rows from a prepared view.
    pub(crate) const fn source(self) -> &'a [f32] {
        assert!(!self.prepared);
        self.values
    }
}

/// Accumulates oldest-first history and the current projection with native FMA.
///
/// # Errors
/// Rejects an empty or mismatched coefficient row, or a nonfinite result.
pub(crate) fn row(history: &[f32], projected: f32, coefficients: &[f32]) -> Result<f32> {
    let Some((&current, previous)) = coefficients.split_last() else {
        return Err(invalid("pinned convolution requires coefficients"));
    };
    if previous.len() != history.len() {
        return Err(invalid("pinned convolution history width mismatch"));
    }
    let mut sum = 0.0;
    for (&value, &coefficient) in history.iter().zip(previous) {
        sum = value.mul_add(coefficient, sum);
    }
    super::finite(projected.mul_add(current, sum))
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
#[inline]
fn transpose_4x4(
    r0: fearless_simd::f32x4<fearless_simd::Neon>,
    r1: fearless_simd::f32x4<fearless_simd::Neon>,
    r2: fearless_simd::f32x4<fearless_simd::Neon>,
    r3: fearless_simd::f32x4<fearless_simd::Neon>,
) -> [fearless_simd::f32x4<fearless_simd::Neon>; 4] {
    use fearless_simd::prelude::*;
    let t0 = r0.zip_low(r2);
    let t1 = r0.zip_high(r2);
    let t2 = r1.zip_low(r3);
    let t3 = r1.zip_high(r3);
    [
        t0.zip_low(t2),
        t0.zip_high(t2),
        t1.zip_low(t3),
        t1.zip_high(t3),
    ]
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
#[inline]
fn coefficient_taps<const PREPARED: bool>(
    neon: fearless_simd::Neon,
    block: &[f32],
) -> [fearless_simd::f32x4<fearless_simd::Neon>; 4] {
    use fearless_simd::{f32x4, prelude::*};
    let taps =
        std::array::from_fn(|index| f32x4::from_slice(neon, &block[index * 4..index * 4 + 4]));
    if PREPARED {
        taps
    } else {
        transpose_4x4(taps[0], taps[1], taps[2], taps[3])
    }
}

fn prepared_coefficient_row(coefficients: &[f32], channel: usize) -> [f32; 4] {
    let base = channel / 4 * 16;
    let lane = channel % 4;
    std::array::from_fn(|tap| coefficients[base + tap * 4 + lane])
}

/// Evaluates all channels of one causal convolution step and writes updated history.
///
/// # Errors
/// Rejects invalid geometry or nonfinite outputs.
#[cfg(test)]
pub(crate) fn step_into(
    history: &[f32],
    mixed: &mut [f32],
    coefficients: &[f32],
    width: usize,
    next_history: &mut [f32],
) -> Result<()> {
    step_into_inner::<false, false>(history, mixed, coefficients, width, next_history)
}

/// Evaluates all channels of one causal convolution step, applies pinned `SiLU`
/// in registers before storing `mixed`, and writes updated history.
///
/// # Errors
/// Rejects invalid geometry or nonfinite outputs.
#[cfg(test)]
pub(crate) fn step_silu_into(
    history: &[f32],
    mixed: &mut [f32],
    coefficients: &[f32],
    width: usize,
    next_history: &mut [f32],
) -> Result<()> {
    step_into_inner::<true, false>(history, mixed, coefficients, width, next_history)
}

/// Runs the exact one-token convolution using immutable prepared model weights.
///
/// # Errors
/// Rejects invalid geometry or nonfinite outputs.
pub(crate) fn step_silu_into_coefficients(
    history: &[f32],
    mixed: &mut [f32],
    coefficients: CoefficientSlice<'_>,
    width: usize,
    next_history: &mut [f32],
) -> Result<()> {
    if coefficients.prepared {
        step_into_inner::<true, true>(history, mixed, coefficients.values, width, next_history)
    } else {
        step_into_inner::<true, false>(history, mixed, coefficients.values, width, next_history)
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "Keep NEON four-channel convolution, optional SiLU fusion and scalar fallback together."
)]
fn step_into_inner<const SILU: bool, const PREPARED: bool>(
    history: &[f32],
    mixed: &mut [f32],
    coefficients: &[f32],
    width: usize,
    next_history: &mut [f32],
) -> Result<()> {
    if width == 0 {
        return Err(invalid("pinned convolution requires coefficients"));
    }
    let hist_width = width - 1;
    if coefficients.len() != mixed.len().saturating_mul(width)
        || history.len() != mixed.len().saturating_mul(hist_width)
        || next_history.len() != history.len()
        || (PREPARED && (width != 4 || !mixed.len().is_multiple_of(4)))
    {
        return Err(invalid("pinned convolution history width mismatch"));
    }
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if width == 4
        && let Some(neon) = fearless_simd::Level::baseline().as_neon()
    {
        use fearless_simd::{f32x4, prelude::*};
        let zero = f32x4::splat(neon, 0.0);
        let one = f32x4::splat(neon, 1.0);
        let mut finite_mask = zero.is_finite();
        let (m_pairs, m_rem8) = mixed.as_chunks_mut::<8>();
        let (h_pairs, h_rem24) = history.as_chunks::<24>();
        let (n_pairs, n_rem24) = next_history.as_chunks_mut::<24>();
        let (c_pairs, c_rem32) = coefficients.as_chunks::<32>();
        for (((m_pair, h_pair), n_pair), c_pair) in
            m_pairs.iter_mut().zip(h_pairs).zip(n_pairs).zip(c_pairs)
        {
            let [tap0_0, tap0_1, tap0_2, tap0_3] =
                coefficient_taps::<PREPARED>(neon, &c_pair[0..16]);
            let [tap1_0, tap1_1, tap1_2, tap1_3] =
                coefficient_taps::<PREPARED>(neon, &c_pair[16..32]);
            let h0_t0 = f32x4::load_array(neon, [h_pair[0], h_pair[3], h_pair[6], h_pair[9]]);
            let h0_t1 = f32x4::load_array(neon, [h_pair[1], h_pair[4], h_pair[7], h_pair[10]]);
            let h0_t2 = f32x4::load_array(neon, [h_pair[2], h_pair[5], h_pair[8], h_pair[11]]);
            let h1_t0 = f32x4::load_array(neon, [h_pair[12], h_pair[15], h_pair[18], h_pair[21]]);
            let h1_t1 = f32x4::load_array(neon, [h_pair[13], h_pair[16], h_pair[19], h_pair[22]]);
            let h1_t2 = f32x4::load_array(neon, [h_pair[14], h_pair[17], h_pair[20], h_pair[23]]);
            let p0 = f32x4::from_slice(neon, &m_pair[0..4]);
            let p1 = f32x4::from_slice(neon, &m_pair[4..8]);
            let mut sum0 = zero;
            let mut sum1 = zero;
            sum0 = h0_t0.mul_add_precise(tap0_0, sum0);
            sum1 = h1_t0.mul_add_precise(tap1_0, sum1);
            sum0 = h0_t1.mul_add_precise(tap0_1, sum0);
            sum1 = h1_t1.mul_add_precise(tap1_1, sum1);
            sum0 = h0_t2.mul_add_precise(tap0_2, sum0);
            sum1 = h1_t2.mul_add_precise(tap1_2, sum1);
            let out0 = p0.mul_add_precise(tap0_3, sum0);
            let out1 = p1.mul_add_precise(tap1_3, sum1);
            finite_mask = finite_mask & out0.is_finite() & out1.is_finite();
            let [p0_0, p0_1, p0_2, p0_3, p1_0, p1_1, p1_2, p1_3] = *m_pair;
            if SILU {
                let s0 = out0 / (one + super::pinned_activations::vector_exp_x4(neon, zero - out0));
                let s1 = out1 / (one + super::pinned_activations::vector_exp_x4(neon, zero - out1));
                s0.store_slice(&mut m_pair[0..4]);
                s1.store_slice(&mut m_pair[4..8]);
            } else {
                out0.store_slice(&mut m_pair[0..4]);
                out1.store_slice(&mut m_pair[4..8]);
            }
            *n_pair = [
                h_pair[1], h_pair[2], p0_0, h_pair[4], h_pair[5], p0_1, h_pair[7], h_pair[8], p0_2,
                h_pair[10], h_pair[11], p0_3, h_pair[13], h_pair[14], p1_0, h_pair[16], h_pair[17],
                p1_1, h_pair[19], h_pair[20], p1_2, h_pair[22], h_pair[23], p1_3,
            ];
        }
        let (m_chunks, m_rem) = m_rem8.as_chunks_mut::<4>();
        let (h_chunks, h_rem) = h_rem24.as_chunks::<12>();
        let (n_chunks, n_rem) = n_rem24.as_chunks_mut::<12>();
        let (c_chunks, c_rem) = c_rem32.as_chunks::<16>();
        for (((m_chunk, h_chunk), n_chunk), c_chunk) in m_chunks
            .iter_mut()
            .zip(h_chunks)
            .zip(n_chunks)
            .zip(c_chunks)
        {
            let [tap0, tap1, tap2, tap3] = coefficient_taps::<PREPARED>(neon, &c_chunk[0..16]);
            let h_t0 = f32x4::load_array(neon, [h_chunk[0], h_chunk[3], h_chunk[6], h_chunk[9]]);
            let h_t1 = f32x4::load_array(neon, [h_chunk[1], h_chunk[4], h_chunk[7], h_chunk[10]]);
            let h_t2 = f32x4::load_array(neon, [h_chunk[2], h_chunk[5], h_chunk[8], h_chunk[11]]);
            let p = f32x4::from_slice(neon, m_chunk);
            let mut sum = zero;
            sum = h_t0.mul_add_precise(tap0, sum);
            sum = h_t1.mul_add_precise(tap1, sum);
            sum = h_t2.mul_add_precise(tap2, sum);
            let out = p.mul_add_precise(tap3, sum);
            finite_mask &= out.is_finite();
            let [p0, p1, p2, p3] = *m_chunk;
            if SILU {
                (out / (one + super::pinned_activations::vector_exp_x4(neon, zero - out)))
                    .store_slice(m_chunk);
            } else {
                out.store_slice(m_chunk);
            }
            *n_chunk = [
                h_chunk[1],
                h_chunk[2],
                p0,
                h_chunk[4],
                h_chunk[5],
                p1,
                h_chunk[7],
                h_chunk[8],
                p2,
                h_chunk[10],
                h_chunk[11],
                p3,
            ];
        }
        if !finite_mask.all_true() {
            return Err(invalid("nonfinite kernel output"));
        }
        for (((mixed_value, h_slice), n_slice), c_slice) in m_rem
            .iter_mut()
            .zip(h_rem.as_chunks::<3>().0)
            .zip(n_rem.as_chunks_mut::<3>().0)
            .zip(c_rem.as_chunks::<4>().0)
        {
            let projected = *mixed_value;
            *mixed_value = row(h_slice, projected, c_slice)?;
            n_slice[0] = h_slice[1];
            n_slice[1] = h_slice[2];
            n_slice[2] = projected;
        }
        if SILU {
            super::pinned_activations::silu_in_place(m_rem);
        }
        return Ok(());
    }
    for (channel, mixed_value) in mixed.iter_mut().enumerate() {
        let projected = *mixed_value;
        let start = channel * hist_width;
        let prepared_row;
        let coeff = if PREPARED {
            prepared_row = prepared_coefficient_row(coefficients, channel);
            &prepared_row[..]
        } else {
            &coefficients[channel * width..(channel + 1) * width]
        };
        *mixed_value = row(&history[start..start + hist_width], projected, coeff)?;
        if hist_width == 3 {
            next_history[start] = history[start + 1];
            next_history[start + 1] = history[start + 2];
            next_history[start + 2] = projected;
        } else if hist_width > 0 {
            next_history[start..start + hist_width - 1]
                .copy_from_slice(&history[start + 1..start + hist_width]);
            next_history[start + hist_width - 1] = projected;
        }
    }
    if SILU {
        super::pinned_activations::silu_in_place(mixed);
    }
    Ok(())
}

/// Evaluates all channels of one causal convolution step and updates history in place.
///
/// # Errors
/// Rejects invalid geometry or nonfinite outputs.
#[cfg(test)]
pub(crate) fn step_in_place(
    history: &mut [f32],
    mixed: &mut [f32],
    coefficients: &[f32],
    width: usize,
) -> Result<()> {
    step_in_place_inner::<false, false>(history, mixed, coefficients, width)
}

/// Evaluates all channels of one causal convolution step, applies pinned `SiLU`
/// in registers before storing `mixed`, and updates history in place.
///
/// # Errors
/// Rejects invalid geometry or nonfinite outputs.
#[cfg(test)]
pub(crate) fn step_silu_in_place(
    history: &mut [f32],
    mixed: &mut [f32],
    coefficients: &[f32],
    width: usize,
) -> Result<()> {
    step_in_place_inner::<true, false>(history, mixed, coefficients, width)
}

/// Runs the exact in-place convolution using immutable prepared model weights.
///
/// # Errors
/// Rejects invalid geometry or nonfinite outputs.
pub(crate) fn step_silu_in_place_coefficients(
    history: &mut [f32],
    mixed: &mut [f32],
    coefficients: CoefficientSlice<'_>,
    width: usize,
) -> Result<()> {
    if coefficients.prepared {
        step_in_place_inner::<true, true>(history, mixed, coefficients.values, width)
    } else {
        step_in_place_inner::<true, false>(history, mixed, coefficients.values, width)
    }
}

/// Evaluates `tokens.len() / channels` contiguous causal convolution steps,
/// applying pinned `SiLU` in registers before storing each token slice and
/// sliding history in NEON registers across tokens before writing updated
/// history once per 4-channel group.
///
/// # Errors
/// Rejects invalid geometry or nonfinite outputs.
#[cfg(test)]
pub(crate) fn step_silu_tokens_in_place(
    history: &mut [f32],
    tokens: &mut [f32],
    channels: usize,
    coefficients: &[f32],
    width: usize,
) -> Result<()> {
    step_tokens_inner::<true, false, false>(
        history,
        tokens,
        channels,
        coefficients,
        width,
        ProjectionRows {
            rows: &[],
            offset: 0,
        },
    )
}

/// Convolves a channel range directly from immutable token projection rows.
///
/// The head-major prefill consumer writes transformed values into `output`
/// without first copying the raw projections there. Sliding history, FMA order
/// and pinned `SiLU` retain the in-place kernel's arithmetic. No input row is
/// changed, and the caller discards partial output and history on graph failure.
///
/// # Errors
/// Rejects invalid row ranges or geometry before mutation, or nonfinite outputs.
#[cfg(test)]
pub(crate) fn step_silu_token_rows_into(
    history: &mut [f32],
    rows: &[Vec<f32>],
    channel_range: &std::ops::Range<usize>,
    coefficients: &[f32],
    width: usize,
    output: &mut [f32],
) -> Result<()> {
    step_silu_token_rows_into_coefficients(
        history,
        rows,
        channel_range,
        CoefficientSlice::raw(coefficients),
        width,
        output,
    )
}

/// Convolves immutable projected rows with prepared or raw model coefficients.
///
/// # Errors
/// Rejects invalid geometry before mutation, or nonfinite outputs.
pub(crate) fn step_silu_token_rows_into_coefficients(
    history: &mut [f32],
    rows: &[Vec<f32>],
    channel_range: &std::ops::Range<usize>,
    coefficients: CoefficientSlice<'_>,
    width: usize,
    output: &mut [f32],
) -> Result<()> {
    let channels = channel_range
        .end
        .checked_sub(channel_range.start)
        .filter(|&channels| channels != 0)
        .ok_or_else(|| invalid("pinned convolution projection range mismatch"))?;
    if rows.iter().any(|row| row.len() < channel_range.end)
        || rows.len().checked_mul(channels) != Some(output.len())
    {
        return Err(invalid("pinned convolution projection row mismatch"));
    }
    let projections = ProjectionRows {
        rows,
        offset: channel_range.start,
    };
    if coefficients.prepared {
        step_tokens_inner::<true, true, true>(
            history,
            output,
            channels,
            coefficients.values,
            width,
            projections,
        )
    } else {
        step_tokens_inner::<true, true, false>(
            history,
            output,
            channels,
            coefficients.values,
            width,
            projections,
        )
    }
}

#[derive(Clone, Copy)]
struct ProjectionRows<'a> {
    rows: &'a [Vec<f32>],
    offset: usize,
}

impl<'a> ProjectionRows<'a> {
    fn chunk(self, token: usize, column: usize, width: usize) -> &'a [f32] {
        &self.rows[token][self.offset + column..self.offset + column + width]
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "Keep multi-token NEON register-sliding convolution, optional SiLU fusion and fallback together."
)]
fn step_tokens_inner<const SILU: bool, const DIRECT: bool, const PREPARED: bool>(
    history: &mut [f32],
    tokens: &mut [f32],
    channels: usize,
    coefficients: &[f32],
    width: usize,
    projections: ProjectionRows<'_>,
) -> Result<()> {
    if width == 0 {
        return Err(invalid("pinned convolution requires coefficients"));
    }
    let hist_width = width - 1;
    if channels == 0
        || !tokens.len().is_multiple_of(channels)
        || coefficients.len() != channels.saturating_mul(width)
        || history.len() != channels.saturating_mul(hist_width)
        || (PREPARED && (width != 4 || !channels.is_multiple_of(4)))
    {
        return Err(invalid("pinned convolution history width mismatch"));
    }
    let count = tokens.len() / channels;
    if count == 0 {
        return Ok(());
    }
    if count == 1 {
        if DIRECT {
            tokens.copy_from_slice(projections.chunk(0, 0, channels));
        }
        return step_in_place_inner::<SILU, PREPARED>(history, tokens, coefficients, width);
    }
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if width == 4
        && channels.is_multiple_of(4)
        && let Some(neon) = fearless_simd::Level::baseline().as_neon()
    {
        use fearless_simd::{f32x4, prelude::*};
        let zero = f32x4::splat(neon, 0.0);
        let one = f32x4::splat(neon, 1.0);
        let mut finite_mask = zero.is_finite();
        let (h_pairs, h_rem24) = history.as_chunks_mut::<24>();
        let (c_pairs, c_rem32) = coefficients.as_chunks::<32>();
        for (pair_idx, (h_pair, c_pair)) in h_pairs.iter_mut().zip(c_pairs).enumerate() {
            let [tap0_0, tap0_1, tap0_2, tap0_3] =
                coefficient_taps::<PREPARED>(neon, &c_pair[0..16]);
            let [tap1_0, tap1_1, tap1_2, tap1_3] =
                coefficient_taps::<PREPARED>(neon, &c_pair[16..32]);
            let mut h0_t0 = f32x4::load_array(neon, [h_pair[0], h_pair[3], h_pair[6], h_pair[9]]);
            let mut h0_t1 = f32x4::load_array(neon, [h_pair[1], h_pair[4], h_pair[7], h_pair[10]]);
            let mut h0_t2 = f32x4::load_array(neon, [h_pair[2], h_pair[5], h_pair[8], h_pair[11]]);
            let mut h1_t0 =
                f32x4::load_array(neon, [h_pair[12], h_pair[15], h_pair[18], h_pair[21]]);
            let mut h1_t1 =
                f32x4::load_array(neon, [h_pair[13], h_pair[16], h_pair[19], h_pair[22]]);
            let mut h1_t2 =
                f32x4::load_array(neon, [h_pair[14], h_pair[17], h_pair[20], h_pair[23]]);
            let col = pair_idx * 8;
            for token in 0..count {
                let offset = token * channels + col;
                let m_slice = &mut tokens[offset..offset + 8];
                let projected = if DIRECT {
                    projections.chunk(token, col, 8)
                } else {
                    &*m_slice
                };
                let p0 = f32x4::from_slice(neon, &projected[0..4]);
                let p1 = f32x4::from_slice(neon, &projected[4..8]);
                let mut sum0 = zero;
                let mut sum1 = zero;
                sum0 = h0_t0.mul_add_precise(tap0_0, sum0);
                sum1 = h1_t0.mul_add_precise(tap1_0, sum1);
                sum0 = h0_t1.mul_add_precise(tap0_1, sum0);
                sum1 = h1_t1.mul_add_precise(tap1_1, sum1);
                sum0 = h0_t2.mul_add_precise(tap0_2, sum0);
                sum1 = h1_t2.mul_add_precise(tap1_2, sum1);
                let out0 = p0.mul_add_precise(tap0_3, sum0);
                let out1 = p1.mul_add_precise(tap1_3, sum1);
                finite_mask = finite_mask & out0.is_finite() & out1.is_finite();
                if SILU {
                    let s0 =
                        out0 / (one + super::pinned_activations::vector_exp_x4(neon, zero - out0));
                    let s1 =
                        out1 / (one + super::pinned_activations::vector_exp_x4(neon, zero - out1));
                    s0.store_slice(&mut m_slice[0..4]);
                    s1.store_slice(&mut m_slice[4..8]);
                } else {
                    out0.store_slice(&mut m_slice[0..4]);
                    out1.store_slice(&mut m_slice[4..8]);
                }
                h0_t0 = h0_t1;
                h0_t1 = h0_t2;
                h0_t2 = p0;
                h1_t0 = h1_t1;
                h1_t1 = h1_t2;
                h1_t2 = p1;
            }
            let t0_0 = h0_t0.to_array();
            let t0_1 = h0_t1.to_array();
            let t0_2 = h0_t2.to_array();
            let t1_0 = h1_t0.to_array();
            let t1_1 = h1_t1.to_array();
            let t1_2 = h1_t2.to_array();
            *h_pair = [
                t0_0[0], t0_1[0], t0_2[0], t0_0[1], t0_1[1], t0_2[1], t0_0[2], t0_1[2], t0_2[2],
                t0_0[3], t0_1[3], t0_2[3], t1_0[0], t1_1[0], t1_2[0], t1_0[1], t1_1[1], t1_2[1],
                t1_0[2], t1_1[2], t1_2[2], t1_0[3], t1_1[3], t1_2[3],
            ];
        }
        let base_group = h_pairs.len() * 2;
        let (h_chunks, _) = h_rem24.as_chunks_mut::<12>();
        let (c_chunks, _) = c_rem32.as_chunks::<16>();
        for (group, (h_chunk, c_chunk)) in h_chunks.iter_mut().zip(c_chunks).enumerate() {
            let [tap0, tap1, tap2, tap3] = coefficient_taps::<PREPARED>(neon, &c_chunk[0..16]);
            let mut h_t0 =
                f32x4::load_array(neon, [h_chunk[0], h_chunk[3], h_chunk[6], h_chunk[9]]);
            let mut h_t1 =
                f32x4::load_array(neon, [h_chunk[1], h_chunk[4], h_chunk[7], h_chunk[10]]);
            let mut h_t2 =
                f32x4::load_array(neon, [h_chunk[2], h_chunk[5], h_chunk[8], h_chunk[11]]);
            let col = (base_group + group) * 4;
            for token in 0..count {
                let offset = token * channels + col;
                let m_chunk = &mut tokens[offset..offset + 4];
                let projected = if DIRECT {
                    projections.chunk(token, col, 4)
                } else {
                    &*m_chunk
                };
                let p = f32x4::from_slice(neon, projected);
                let mut sum = zero;
                sum = h_t0.mul_add_precise(tap0, sum);
                sum = h_t1.mul_add_precise(tap1, sum);
                sum = h_t2.mul_add_precise(tap2, sum);
                let out = p.mul_add_precise(tap3, sum);
                finite_mask &= out.is_finite();
                if SILU {
                    (out / (one + super::pinned_activations::vector_exp_x4(neon, zero - out)))
                        .store_slice(m_chunk);
                } else {
                    out.store_slice(m_chunk);
                }
                h_t0 = h_t1;
                h_t1 = h_t2;
                h_t2 = p;
            }
            let t0 = h_t0.to_array();
            let t1 = h_t1.to_array();
            let t2 = h_t2.to_array();
            *h_chunk = [
                t0[0], t1[0], t2[0], t0[1], t1[1], t2[1], t0[2], t1[2], t2[2], t0[3], t1[3], t2[3],
            ];
        }
        if !finite_mask.all_true() {
            return Err(invalid("nonfinite kernel output"));
        }
        return Ok(());
    }
    // Unsupported shapes and targets retain the original complete raw staging
    // before scalar execution, including unchanged later tokens on failure.
    if DIRECT {
        for (token, token_slice) in tokens.chunks_exact_mut(channels).enumerate() {
            token_slice.copy_from_slice(projections.chunk(token, 0, channels));
        }
    }
    for token_slice in tokens.chunks_exact_mut(channels) {
        step_in_place_inner::<SILU, PREPARED>(history, token_slice, coefficients, width)?;
    }
    Ok(())
}

#[expect(
    clippy::too_many_lines,
    reason = "Keep in-place NEON four-channel convolution, optional SiLU fusion and scalar fallback together."
)]
fn step_in_place_inner<const SILU: bool, const PREPARED: bool>(
    history: &mut [f32],
    mixed: &mut [f32],
    coefficients: &[f32],
    width: usize,
) -> Result<()> {
    if width == 0 {
        return Err(invalid("pinned convolution requires coefficients"));
    }
    let hist_width = width - 1;
    if coefficients.len() != mixed.len().saturating_mul(width)
        || history.len() != mixed.len().saturating_mul(hist_width)
        || (PREPARED && (width != 4 || !mixed.len().is_multiple_of(4)))
    {
        return Err(invalid("pinned convolution history width mismatch"));
    }
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if width == 4
        && let Some(neon) = fearless_simd::Level::baseline().as_neon()
    {
        use fearless_simd::{f32x4, prelude::*};
        let zero = f32x4::splat(neon, 0.0);
        let one = f32x4::splat(neon, 1.0);
        let mut finite_mask = zero.is_finite();
        let (m_pairs, m_rem8) = mixed.as_chunks_mut::<8>();
        let (h_pairs, h_rem24) = history.as_chunks_mut::<24>();
        let (c_pairs, c_rem32) = coefficients.as_chunks::<32>();
        for ((m_pair, h_pair), c_pair) in m_pairs.iter_mut().zip(h_pairs).zip(c_pairs) {
            let [tap0_0, tap0_1, tap0_2, tap0_3] =
                coefficient_taps::<PREPARED>(neon, &c_pair[0..16]);
            let [tap1_0, tap1_1, tap1_2, tap1_3] =
                coefficient_taps::<PREPARED>(neon, &c_pair[16..32]);
            let h0_t0 = f32x4::load_array(neon, [h_pair[0], h_pair[3], h_pair[6], h_pair[9]]);
            let h0_t1 = f32x4::load_array(neon, [h_pair[1], h_pair[4], h_pair[7], h_pair[10]]);
            let h0_t2 = f32x4::load_array(neon, [h_pair[2], h_pair[5], h_pair[8], h_pair[11]]);
            let h1_t0 = f32x4::load_array(neon, [h_pair[12], h_pair[15], h_pair[18], h_pair[21]]);
            let h1_t1 = f32x4::load_array(neon, [h_pair[13], h_pair[16], h_pair[19], h_pair[22]]);
            let h1_t2 = f32x4::load_array(neon, [h_pair[14], h_pair[17], h_pair[20], h_pair[23]]);
            let p0 = f32x4::from_slice(neon, &m_pair[0..4]);
            let p1 = f32x4::from_slice(neon, &m_pair[4..8]);
            let mut sum0 = zero;
            let mut sum1 = zero;
            sum0 = h0_t0.mul_add_precise(tap0_0, sum0);
            sum1 = h1_t0.mul_add_precise(tap1_0, sum1);
            sum0 = h0_t1.mul_add_precise(tap0_1, sum0);
            sum1 = h1_t1.mul_add_precise(tap1_1, sum1);
            sum0 = h0_t2.mul_add_precise(tap0_2, sum0);
            sum1 = h1_t2.mul_add_precise(tap1_2, sum1);
            let out0 = p0.mul_add_precise(tap0_3, sum0);
            let out1 = p1.mul_add_precise(tap1_3, sum1);
            finite_mask = finite_mask & out0.is_finite() & out1.is_finite();
            let [p0_0, p0_1, p0_2, p0_3, p1_0, p1_1, p1_2, p1_3] = *m_pair;
            if SILU {
                let s0 = out0 / (one + super::pinned_activations::vector_exp_x4(neon, zero - out0));
                let s1 = out1 / (one + super::pinned_activations::vector_exp_x4(neon, zero - out1));
                s0.store_slice(&mut m_pair[0..4]);
                s1.store_slice(&mut m_pair[4..8]);
            } else {
                out0.store_slice(&mut m_pair[0..4]);
                out1.store_slice(&mut m_pair[4..8]);
            }
            *h_pair = [
                h_pair[1], h_pair[2], p0_0, h_pair[4], h_pair[5], p0_1, h_pair[7], h_pair[8], p0_2,
                h_pair[10], h_pair[11], p0_3, h_pair[13], h_pair[14], p1_0, h_pair[16], h_pair[17],
                p1_1, h_pair[19], h_pair[20], p1_2, h_pair[22], h_pair[23], p1_3,
            ];
        }
        let (m_chunks, m_rem) = m_rem8.as_chunks_mut::<4>();
        let (h_chunks, h_rem) = h_rem24.as_chunks_mut::<12>();
        let (c_chunks, c_rem) = c_rem32.as_chunks::<16>();
        for ((m_chunk, h_chunk), c_chunk) in m_chunks.iter_mut().zip(h_chunks).zip(c_chunks) {
            let [tap0, tap1, tap2, tap3] = coefficient_taps::<PREPARED>(neon, &c_chunk[0..16]);
            let h_t0 = f32x4::load_array(neon, [h_chunk[0], h_chunk[3], h_chunk[6], h_chunk[9]]);
            let h_t1 = f32x4::load_array(neon, [h_chunk[1], h_chunk[4], h_chunk[7], h_chunk[10]]);
            let h_t2 = f32x4::load_array(neon, [h_chunk[2], h_chunk[5], h_chunk[8], h_chunk[11]]);
            let p = f32x4::from_slice(neon, m_chunk);
            let mut sum = zero;
            sum = h_t0.mul_add_precise(tap0, sum);
            sum = h_t1.mul_add_precise(tap1, sum);
            sum = h_t2.mul_add_precise(tap2, sum);
            let out = p.mul_add_precise(tap3, sum);
            finite_mask &= out.is_finite();
            let [p0, p1, p2, p3] = *m_chunk;
            if SILU {
                (out / (one + super::pinned_activations::vector_exp_x4(neon, zero - out)))
                    .store_slice(m_chunk);
            } else {
                out.store_slice(m_chunk);
            }
            *h_chunk = [
                h_chunk[1],
                h_chunk[2],
                p0,
                h_chunk[4],
                h_chunk[5],
                p1,
                h_chunk[7],
                h_chunk[8],
                p2,
                h_chunk[10],
                h_chunk[11],
                p3,
            ];
        }
        if !finite_mask.all_true() {
            return Err(invalid("nonfinite kernel output"));
        }
        for ((mixed_value, h_slice), c_slice) in m_rem
            .iter_mut()
            .zip(h_rem.as_chunks_mut::<3>().0)
            .zip(c_rem.as_chunks::<4>().0)
        {
            let projected = *mixed_value;
            *mixed_value = row(h_slice, projected, c_slice)?;
            h_slice[0] = h_slice[1];
            h_slice[1] = h_slice[2];
            h_slice[2] = projected;
        }
        if SILU {
            super::pinned_activations::silu_in_place(m_rem);
        }
        return Ok(());
    }
    for (channel, mixed_value) in mixed.iter_mut().enumerate() {
        let projected = *mixed_value;
        let start = channel * hist_width;
        let prepared_row;
        let coeff = if PREPARED {
            prepared_row = prepared_coefficient_row(coefficients, channel);
            &prepared_row[..]
        } else {
            &coefficients[channel * width..(channel + 1) * width]
        };
        *mixed_value = row(&history[start..start + hist_width], projected, coeff)?;
        if hist_width == 3 {
            history[start] = history[start + 1];
            history[start + 1] = history[start + 2];
            history[start + 2] = projected;
        } else if hist_width > 0 {
            history[start..start + hist_width].rotate_left(1);
            history[start + hist_width - 1] = projected;
        }
    }
    if SILU {
        super::pinned_activations::silu_in_place(mixed);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::io::{Cursor, Read};

    use serde::Deserialize;
    use sha2::{Digest, Sha256};

    use super::{
        row, step_in_place, step_into, step_silu_in_place, step_silu_into,
        step_silu_token_rows_into, step_silu_tokens_in_place,
    };
    use crate::Result;

    #[derive(Deserialize)]
    struct Capture {
        runtime_commit: String,
        runtime_dll_sha256: BTreeMap<String, String>,
        fixture_bytes: usize,
        fixture_sha256: String,
    }

    fn word(input: &mut Cursor<&[u8]>) -> Result<u32> {
        let mut bytes = [0; 4];
        input.read_exact(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn values(input: &mut Cursor<&[u8]>, count: usize) -> Result<Vec<f32>> {
        (0..count)
            .map(|_| word(input).map(f32::from_bits))
            .collect()
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().copied().map(f32::to_bits).collect()
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "Verify row, step_into, step_in_place, step_silu_into and step_silu_in_place together."
    )]
    fn pinned_convolution_matches_native_zero_partial_and_full_history() -> Result<()> {
        let bytes = include_bytes!("../../tests/fixtures/swift-convolution-native.bin");
        let capture: Capture = serde_json::from_str(include_str!(
            "../../tests/fixtures/swift-convolution-native.json"
        ))?;
        let hash: String = Sha256::digest(bytes)
            .into_iter()
            .flat_map(|byte| [byte >> 4, byte & 15])
            .map(|nibble| char::from(b"0123456789abcdef"[usize::from(nibble)]))
            .collect();
        assert_eq!(bytes.len(), capture.fixture_bytes);
        assert_eq!(hash, capture.fixture_sha256);
        assert_eq!(
            hash,
            "02adf53bc96410e4402e3f77a0e4201a0d3cbb4a4b7aa53784025cc65de2119d"
        );
        assert_eq!(
            capture.runtime_commit,
            "73bfe6cc485554822f80c473a3c0c4f25e80f41b"
        );
        assert_eq!(
            capture
                .runtime_dll_sha256
                .get("ggml-cpu.dll")
                .map(String::as_str),
            Some("701b354a38b25e5c6dc57a1559f71cd26b2d0f8d69c1576a0e77a59a4273d2ab")
        );
        let mut input = Cursor::new(bytes.as_slice());
        let mut magic = [0; 8];
        input.read_exact(&mut magic)?;
        assert_eq!(&magic, b"SWCONV01");
        let cases = word(&mut input)?;
        assert_eq!(cases, 15);
        for case in 0..cases {
            let width = usize::try_from(word(&mut input)?)?;
            let channels = usize::try_from(word(&mut input)?)?;
            let history = usize::try_from(word(&mut input)?)?;
            let cancellation = word(&mut input)?;
            assert!(matches!(width, 1 | 2 | 4) && matches!(channels, 64 | 65));
            assert!(history < width && cancellation <= 1);
            let signal = values(&mut input, width * channels)?;
            let coefficients = values(&mut input, width * channels)?;
            let native = values(&mut input, channels)?;
            let hist_width = width - 1;
            let mut flat_history = Vec::with_capacity(channels * hist_width);
            let mut mixed = Vec::with_capacity(channels);
            for (channel, ((signal, coeff), &expected)) in signal
                .chunks_exact(width)
                .zip(coefficients.chunks_exact(width))
                .zip(&native)
                .enumerate()
            {
                assert_eq!(
                    row(&signal[..hist_width], signal[hist_width], coeff)?.to_bits(),
                    expected.to_bits(),
                    "case {case}, channel {channel}"
                );
                flat_history.extend_from_slice(&signal[..hist_width]);
                mixed.push(signal[hist_width]);
            }
            let mut mixed_into = mixed.clone();
            let mut next_history = vec![0.0; flat_history.len()];
            step_into(
                &flat_history,
                &mut mixed_into,
                &coefficients,
                width,
                &mut next_history,
            )?;
            let mut mixed_silu_into = mixed.clone();
            let mut next_silu_history = vec![0.0; flat_history.len()];
            step_silu_into(
                &flat_history,
                &mut mixed_silu_into,
                &coefficients,
                width,
                &mut next_silu_history,
            )?;
            let mut mixed_silu_in_place = mixed.clone();
            let mut in_place_silu_history = flat_history.clone();
            step_silu_in_place(
                &mut in_place_silu_history,
                &mut mixed_silu_in_place,
                &coefficients,
                width,
            )?;
            let mut mixed_in_place = mixed;
            let mut in_place_history = flat_history;
            step_in_place(
                &mut in_place_history,
                &mut mixed_in_place,
                &coefficients,
                width,
            )?;
            let expected_bits = bits(&native);
            assert_eq!(bits(&mixed_into), expected_bits, "step_into case {case}");
            assert_eq!(
                bits(&mixed_in_place),
                expected_bits,
                "step_in_place case {case}"
            );
            assert_eq!(
                bits(&next_history),
                bits(&in_place_history),
                "history update case {case}"
            );
            let mut expected_silu = native;
            super::super::pinned_activations::silu_in_place(&mut expected_silu);
            let expected_silu_bits = bits(&expected_silu);
            assert_eq!(
                bits(&mixed_silu_into),
                expected_silu_bits,
                "step_silu_into case {case}"
            );
            assert_eq!(
                bits(&mixed_silu_in_place),
                expected_silu_bits,
                "step_silu_in_place case {case}"
            );
            assert_eq!(
                bits(&next_silu_history),
                bits(&next_history),
                "step_silu_into history case {case}"
            );
            assert_eq!(
                bits(&in_place_silu_history),
                bits(&in_place_history),
                "step_silu_in_place history case {case}"
            );
        }
        assert_eq!(input.position(), u64::try_from(bytes.len())?);
        Ok(())
    }

    #[test]
    fn pinned_convolution_rejects_invalid_rows_and_nonfinite_results() {
        assert!(row(&[], 1.0, &[]).is_err());
        assert!(row(&[1.0], 2.0, &[3.0]).is_err());
        assert!(row(&[], f32::MAX, &[2.0]).is_err());
        assert!(row(&[f32::NAN], 1.0, &[1.0, 1.0]).is_err());
    }

    #[test]
    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        clippy::suboptimal_flops,
        reason = "Deterministic synthetic test inputs."
    )]
    fn pinned_convolution_multi_token_matches_single_token_steps_bit_for_bit() -> Result<()> {
        for channels in [8, 128, 65] {
            for width in [4, 2] {
                let hist_width = width - 1;
                let history: Vec<f32> = (0..channels * hist_width)
                    .map(|idx| ((idx * 17 + 3) % 31) as f32 * 0.0625 - 0.5)
                    .collect();
                let coefficients: Vec<f32> = (0..channels * width)
                    .map(|idx| ((idx * 23 + 7) % 29) as f32 * 0.03125 - 0.375)
                    .collect();
                let count = 8;
                let tokens: Vec<f32> = (0..count * channels)
                    .map(|idx| ((idx * 41 + 11) % 37) as f32 * 0.0625 - 0.75)
                    .collect();

                let mut expected_history = history.clone();
                let mut expected_tokens = tokens.clone();
                for token_slice in expected_tokens.chunks_exact_mut(channels) {
                    step_silu_in_place(&mut expected_history, token_slice, &coefficients, width)?;
                }

                let mut actual_history = history;
                let mut actual_tokens = tokens;
                step_silu_tokens_in_place(
                    &mut actual_history,
                    &mut actual_tokens,
                    channels,
                    &coefficients,
                    width,
                )?;
                assert_eq!(bits(&actual_tokens), bits(&expected_tokens));
                assert_eq!(bits(&actual_history), bits(&expected_history));
            }
        }
        Ok(())
    }

    fn scalar_projected_rows(
        history: &mut [f32],
        rows: &[Vec<f32>],
        offset: usize,
        channels: usize,
        coefficients: &[f32],
        width: usize,
    ) -> Vec<f32> {
        let mut output = Vec::new();
        let history_width = width - 1;
        for row in rows {
            for channel in 0..channels {
                let projected = row[offset + channel];
                let history_base = channel * history_width;
                let coefficient_base = channel * width;
                let mut sum = 0.0_f32;
                for tap in 0..history_width {
                    sum = history[history_base + tap]
                        .mul_add(coefficients[coefficient_base + tap], sum);
                }
                sum = projected.mul_add(coefficients[coefficient_base + history_width], sum);
                let exponential = if channel < channels / 4 * 4 {
                    crate::kernels::pinned_activations::vector_exp(0.0 - sum)
                } else {
                    (-sum).exp()
                };
                output.push(sum / (1.0 + exponential));
                for tap in 1..history_width {
                    history[history_base + tap - 1] = history[history_base + tap];
                }
                if history_width != 0 {
                    history[history_base + history_width - 1] = projected;
                }
            }
        }
        output
    }

    #[test]
    fn pinned_convolution_projected_rows_match_independent_scalar_bits() -> Result<()> {
        let samples = [
            -0.0,
            0.0,
            f32::from_bits(1),
            -f32::from_bits(1),
            0.125,
            -0.375,
            100.0,
            -100.0,
            0.75,
        ];
        for channels in [1, 4, 8, 12, 65, 128] {
            for width in [1, 2, 4, 5] {
                for count in [0, 1, 2, 8, 16, 32] {
                    let history: Vec<_> = samples
                        .iter()
                        .copied()
                        .cycle()
                        .take(channels * (width - 1))
                        .collect();
                    let coefficients: Vec<_> = [0.125, -0.25, 0.5, 0.0, -0.0]
                        .into_iter()
                        .cycle()
                        .take(channels * width)
                        .collect();
                    let rows: Vec<Vec<_>> = (0..count)
                        .map(|token| {
                            let mut row = vec![f32::NAN; 3];
                            row.extend(samples.iter().copied().cycle().skip(token).take(channels));
                            row.extend([f32::INFINITY; 2]);
                            row
                        })
                        .collect();
                    let original_rows: Vec<_> = rows.iter().map(|row| bits(row)).collect();
                    let mut expected_history = history.clone();
                    let expected = scalar_projected_rows(
                        &mut expected_history,
                        &rows,
                        3,
                        channels,
                        &coefficients,
                        width,
                    );
                    let mut actual_history = history;
                    let mut actual = vec![f32::NAN; count * channels];
                    step_silu_token_rows_into(
                        &mut actual_history,
                        &rows,
                        &(3..3 + channels),
                        &coefficients,
                        width,
                        &mut actual,
                    )?;
                    assert_eq!(bits(&actual), bits(&expected), "{channels}/{width}/{count}");
                    assert_eq!(bits(&actual_history), bits(&expected_history));
                    assert_eq!(
                        rows.iter().map(|row| bits(row)).collect::<Vec<_>>(),
                        original_rows
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn x1_convolution_preparation_preserves_storage_and_fallback_layouts() {
        for width in [1, 2, 3, 4, 5] {
            for channels in [1, 3, 4, 8, 12, 128] {
                for pinned in [false, true] {
                    let values: Vec<_> = [0.0, -0.0, f32::from_bits(1), -0.375, 1.25]
                        .into_iter()
                        .cycle()
                        .take(channels * width)
                        .collect();
                    let original = bits(&values);
                    let address = values.as_ptr();
                    let capacity = values.capacity();
                    let coefficients = super::Coefficients::new(values, width, pinned);
                    assert_eq!(coefficients.values.as_ptr(), address);
                    assert_eq!(coefficients.values.capacity(), capacity);
                    assert_eq!(coefficients.values.len(), channels * width);
                    let prepared = pinned && width == 4 && channels.is_multiple_of(4);
                    assert_eq!(coefficients.prepared, prepared);
                    if prepared {
                        for channel in 0..channels {
                            let row =
                                super::prepared_coefficient_row(&coefficients.values, channel);
                            assert_eq!(bits(&row), original[channel * 4..channel * 4 + 4]);
                        }
                        for channel in (0..channels).step_by(4) {
                            let part = coefficients.range(channel * 4..channel * 4 + 16);
                            assert!(part.prepared);
                            for lane in 0..4 {
                                let row = super::prepared_coefficient_row(part.values, lane);
                                assert_eq!(
                                    bits(&row),
                                    original[(channel + lane) * 4..(channel + lane + 1) * 4]
                                );
                            }
                        }
                    } else {
                        assert_eq!(bits(coefficients.view().source()), original);
                    }
                }
            }
        }
    }

    #[test]
    fn x1_prepared_convolution_matches_independent_decode_and_prefill_bits() -> Result<()> {
        let samples = [0.0, -0.0, f32::from_bits(1), -0.375, 0.125, 0.75];
        for channels in [4, 8, 12, 128] {
            let coefficients: Vec<_> = [0.125, -0.25, 0.5, -0.0, 0.75]
                .into_iter()
                .cycle()
                .take(channels * 4)
                .collect();
            let prepared = super::Coefficients::new(coefficients.clone(), 4, true);
            for count in [1, 2, 7, 32] {
                let history: Vec<_> = samples.into_iter().cycle().take(channels * 3).collect();
                let rows: Vec<Vec<_>> = (0..count)
                    .map(|token| {
                        let mut row = vec![f32::NAN; 3];
                        row.extend(samples.into_iter().cycle().skip(token).take(channels));
                        row
                    })
                    .collect();
                let mut expected_history = history.clone();
                let expected = scalar_projected_rows(
                    &mut expected_history,
                    &rows,
                    3,
                    channels,
                    &coefficients,
                    4,
                );
                let mut actual_history = history.clone();
                let mut actual = vec![f32::NAN; channels * count];
                super::step_silu_token_rows_into_coefficients(
                    &mut actual_history,
                    &rows,
                    &(3..3 + channels),
                    prepared.view(),
                    4,
                    &mut actual,
                )?;
                assert_eq!(bits(&actual), bits(&expected));
                assert_eq!(bits(&actual_history), bits(&expected_history));
                let mut previous = history;
                for (token, row) in rows.iter().enumerate() {
                    let mut mixed = row[3..].to_vec();
                    let mut next = vec![f32::NAN; channels * 3];
                    super::step_silu_into_coefficients(
                        &previous,
                        &mut mixed,
                        prepared.view(),
                        4,
                        &mut next,
                    )?;
                    assert_eq!(
                        bits(&mixed),
                        bits(&expected[token * channels..(token + 1) * channels])
                    );
                    let mut in_place = row[3..].to_vec();
                    super::step_silu_in_place_coefficients(
                        &mut previous,
                        &mut in_place,
                        prepared.view(),
                        4,
                    )?;
                    assert_eq!(bits(&in_place), bits(&mixed));
                    assert_eq!(bits(&previous), bits(&next));
                }
                assert_eq!(bits(&previous), bits(&expected_history));
            }
        }
        Ok(())
    }

    #[test]
    fn x1_prepared_convolution_keeps_failure_and_geometry_publication() {
        for channels in [4, 8, 12] {
            for count in [1, 3] {
                for fault in 0..3 {
                    let mut history = vec![0.25; channels * 3];
                    let mut rows = vec![vec![0.5; channels + 2]; count];
                    let mut raw = vec![0.125; channels * 4];
                    match fault {
                        0 => rows[count - 1][channels] = f32::INFINITY,
                        1 => raw[channels * 4 - 2] = f32::NAN,
                        _ => history[channels * 3 - 1] = f32::INFINITY,
                    }
                    let prepared = super::Coefficients::new(raw.clone(), 4, true);
                    let mut expected_history = history.clone();
                    let mut actual = vec![f32::NAN; count * channels];
                    let mut expected = actual.clone();
                    let raw_result = step_silu_token_rows_into(
                        &mut expected_history,
                        &rows,
                        &(1..channels + 1),
                        &raw,
                        4,
                        &mut expected,
                    );
                    let prepared_result = super::step_silu_token_rows_into_coefficients(
                        &mut history,
                        &rows,
                        &(1..channels + 1),
                        prepared.view(),
                        4,
                        &mut actual,
                    );
                    assert!(raw_result.is_err());
                    assert!(prepared_result.is_err());
                    assert_eq!(bits(&actual), bits(&expected));
                    assert_eq!(bits(&history), bits(&expected_history));
                }
            }
        }
        let prepared = super::Coefficients::new(vec![0.125; 16], 4, true);
        for width in [0, 1, 3, 5] {
            let mut history = vec![0.25; 12];
            let mut output = vec![0.5; 4];
            let before_history = bits(&history);
            let before_output = bits(&output);
            assert!(
                super::step_silu_in_place_coefficients(
                    &mut history,
                    &mut output,
                    prepared.view(),
                    width
                )
                .is_err()
            );
            assert_eq!(bits(&history), before_history);
            assert_eq!(bits(&output), before_output);
        }
    }

    #[test]
    fn pinned_convolution_projected_rows_preserve_failure_publication() {
        for channels in [4, 8, 12, 65] {
            for width in [1, 2, 4, 5] {
                for count in [1, 3] {
                    for fault in 0..3 {
                        if fault == 2 && width == 1 {
                            continue;
                        }
                        let mut rows = vec![vec![0.5; channels + 2]; count];
                        let mut history = vec![0.25; channels * (width - 1)];
                        let mut coefficients = vec![0.125; channels * width];
                        match fault {
                            0 => rows[count - 1][channels] = f32::INFINITY,
                            1 => coefficients[channels * width - 1] = f32::MAX,
                            _ => history[channels * (width - 1) - 1] = f32::NAN,
                        }
                        if fault == 1 {
                            rows[count - 1][channels] = 2.0;
                        }
                        let original_rows: Vec<_> = rows.iter().map(|row| bits(row)).collect();
                        let mut expected_history = history.clone();
                        let mut expected: Vec<_> = rows
                            .iter()
                            .flat_map(|row| row[1..=channels].iter().copied())
                            .collect();
                        assert!(
                            step_silu_tokens_in_place(
                                &mut expected_history,
                                &mut expected,
                                channels,
                                &coefficients,
                                width,
                            )
                            .is_err()
                        );
                        let mut actual = vec![-999.0; count * channels];
                        assert!(
                            step_silu_token_rows_into(
                                &mut history,
                                &rows,
                                &(1..1 + channels),
                                &coefficients,
                                width,
                                &mut actual,
                            )
                            .is_err()
                        );
                        assert_eq!(
                            bits(&actual),
                            bits(&expected),
                            "{channels}/{width}/{count}/{fault}"
                        );
                        assert_eq!(bits(&history), bits(&expected_history));
                        assert_eq!(
                            rows.iter().map(|row| bits(row)).collect::<Vec<_>>(),
                            original_rows
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn pinned_convolution_projected_rows_reject_geometry_before_mutation() {
        let rows = vec![vec![0.5; 12]; 2];
        for (start, end, width, history_len, coefficients_len, output_len) in [
            (3, 2, 4, 24, 32, 16),
            (3, 3, 4, 24, 32, 16),
            (5, 13, 4, 24, 32, 16),
            (3, 11, 0, 24, 32, 16),
            (3, 11, 4, 23, 32, 16),
            (3, 11, 4, 24, 31, 16),
            (3, 11, 4, 24, 32, 15),
        ] {
            let mut history = vec![0.25; history_len];
            let coefficients = vec![0.125; coefficients_len];
            let mut output = vec![-999.0; output_len];
            assert!(
                step_silu_token_rows_into(
                    &mut history,
                    &rows,
                    &(start..end),
                    &coefficients,
                    width,
                    &mut output,
                )
                .is_err()
            );
            assert_eq!(history, vec![0.25; history_len]);
            assert_eq!(output, vec![-999.0; output_len]);
        }
        let mut short = rows;
        short[1].truncate(10);
        let mut history = [0.25; 24];
        let mut output = [-999.0; 16];
        assert!(step_silu_token_rows_into(
            &mut history, &short, &(3..11), &[0.125; 32], 4, &mut output,
        ).is_err());
        assert_eq!(history, [0.25; 24]);
        assert_eq!(output, [-999.0; 16]);
    }
}
