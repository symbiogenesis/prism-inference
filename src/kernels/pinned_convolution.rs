//! Exact pinned Windows ARM64 SSM convolution accumulation.
//!
//! Reference: Prism `73bfe6cc485554822f80c473a3c0c4f25e80f41b`,
//! `ggml/src/ggml-cpu/ops.cpp`, compiled `ggml_ssm_conv` arithmetic.
//! Upstream MIT terms are retained in `PRISM-LICENSE.txt`.

#![allow(
    clippy::redundant_pub_crate,
    reason = "Keep the exact-model convolution helper crate scoped."
)]

use crate::{Result, invalid};

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
    step_into_inner::<false>(history, mixed, coefficients, width, next_history)
}

/// Evaluates all channels of one causal convolution step, applies pinned `SiLU`
/// in registers before storing `mixed`, and writes updated history.
///
/// # Errors
/// Rejects invalid geometry or nonfinite outputs.
pub(crate) fn step_silu_into(
    history: &[f32],
    mixed: &mut [f32],
    coefficients: &[f32],
    width: usize,
    next_history: &mut [f32],
) -> Result<()> {
    step_into_inner::<true>(history, mixed, coefficients, width, next_history)
}

#[allow(
    clippy::too_many_lines,
    reason = "Keep NEON four-channel convolution, optional SiLU fusion and scalar fallback together."
)]
fn step_into_inner<const SILU: bool>(
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
        let (m_chunks, m_rem) = mixed.as_chunks_mut::<4>();
        let (h_chunks, h_rem) = history.as_chunks::<12>();
        let (n_chunks, n_rem) = next_history.as_chunks_mut::<12>();
        let (c_chunks, c_rem) = coefficients.as_chunks::<16>();
        for (((m_chunk, h_chunk), n_chunk), c_chunk) in m_chunks
            .iter_mut()
            .zip(h_chunks)
            .zip(n_chunks)
            .zip(c_chunks)
        {
            let [tap0, tap1, tap2, tap3] = transpose_4x4(
                f32x4::from_slice(neon, &c_chunk[0..4]),
                f32x4::from_slice(neon, &c_chunk[4..8]),
                f32x4::from_slice(neon, &c_chunk[8..12]),
                f32x4::from_slice(neon, &c_chunk[12..16]),
            );
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
        let coeff = &coefficients[channel * width..(channel + 1) * width];
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
    step_in_place_inner::<false>(history, mixed, coefficients, width)
}

/// Evaluates all channels of one causal convolution step, applies pinned `SiLU`
/// in registers before storing `mixed`, and updates history in place.
///
/// # Errors
/// Rejects invalid geometry or nonfinite outputs.
pub(crate) fn step_silu_in_place(
    history: &mut [f32],
    mixed: &mut [f32],
    coefficients: &[f32],
    width: usize,
) -> Result<()> {
    step_in_place_inner::<true>(history, mixed, coefficients, width)
}

/// Evaluates `tokens.len() / channels` contiguous causal convolution steps,
/// applying pinned `SiLU` in registers before storing each token slice and
/// sliding history in NEON registers across tokens before writing updated
/// history once per 4-channel group.
///
/// # Errors
/// Rejects invalid geometry or nonfinite outputs.
pub(crate) fn step_silu_tokens_in_place(
    history: &mut [f32],
    tokens: &mut [f32],
    channels: usize,
    coefficients: &[f32],
    width: usize,
) -> Result<()> {
    step_tokens_in_place_inner::<true>(history, tokens, channels, coefficients, width)
}

#[allow(
    clippy::too_many_lines,
    reason = "Keep multi-token NEON register-sliding convolution, optional SiLU fusion and fallback together."
)]
fn step_tokens_in_place_inner<const SILU: bool>(
    history: &mut [f32],
    tokens: &mut [f32],
    channels: usize,
    coefficients: &[f32],
    width: usize,
) -> Result<()> {
    if width == 0 {
        return Err(invalid("pinned convolution requires coefficients"));
    }
    let hist_width = width - 1;
    if channels == 0
        || !tokens.len().is_multiple_of(channels)
        || coefficients.len() != channels.saturating_mul(width)
        || history.len() != channels.saturating_mul(hist_width)
    {
        return Err(invalid("pinned convolution history width mismatch"));
    }
    let count = tokens.len() / channels;
    if count == 0 {
        return Ok(());
    }
    if count == 1 {
        return step_in_place_inner::<SILU>(history, tokens, coefficients, width);
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
        let (h_chunks, _) = history.as_chunks_mut::<12>();
        let (c_chunks, _) = coefficients.as_chunks::<16>();
        for (group, (h_chunk, c_chunk)) in h_chunks.iter_mut().zip(c_chunks).enumerate() {
            let [tap0, tap1, tap2, tap3] = transpose_4x4(
                f32x4::from_slice(neon, &c_chunk[0..4]),
                f32x4::from_slice(neon, &c_chunk[4..8]),
                f32x4::from_slice(neon, &c_chunk[8..12]),
                f32x4::from_slice(neon, &c_chunk[12..16]),
            );
            let mut h_t0 =
                f32x4::load_array(neon, [h_chunk[0], h_chunk[3], h_chunk[6], h_chunk[9]]);
            let mut h_t1 =
                f32x4::load_array(neon, [h_chunk[1], h_chunk[4], h_chunk[7], h_chunk[10]]);
            let mut h_t2 =
                f32x4::load_array(neon, [h_chunk[2], h_chunk[5], h_chunk[8], h_chunk[11]]);
            let col = group * 4;
            for token in 0..count {
                let offset = token * channels + col;
                let m_chunk = &mut tokens[offset..offset + 4];
                let p = f32x4::from_slice(neon, m_chunk);
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
            let mut t0 = [0.0; 4];
            let mut t1 = [0.0; 4];
            let mut t2 = [0.0; 4];
            h_t0.store_slice(&mut t0);
            h_t1.store_slice(&mut t1);
            h_t2.store_slice(&mut t2);
            *h_chunk = [
                t0[0], t1[0], t2[0], t0[1], t1[1], t2[1], t0[2], t1[2], t2[2], t0[3], t1[3], t2[3],
            ];
        }
        if !finite_mask.all_true() {
            return Err(invalid("nonfinite kernel output"));
        }
        return Ok(());
    }
    for token_slice in tokens.chunks_exact_mut(channels) {
        step_in_place_inner::<SILU>(history, token_slice, coefficients, width)?;
    }
    Ok(())
}

#[allow(
    clippy::too_many_lines,
    reason = "Keep in-place NEON four-channel convolution, optional SiLU fusion and scalar fallback together."
)]
fn step_in_place_inner<const SILU: bool>(
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
        let (m_chunks, m_rem) = mixed.as_chunks_mut::<4>();
        let (h_chunks, h_rem) = history.as_chunks_mut::<12>();
        let (c_chunks, c_rem) = coefficients.as_chunks::<16>();
        for ((m_chunk, h_chunk), c_chunk) in m_chunks.iter_mut().zip(h_chunks).zip(c_chunks) {
            let [tap0, tap1, tap2, tap3] = transpose_4x4(
                f32x4::from_slice(neon, &c_chunk[0..4]),
                f32x4::from_slice(neon, &c_chunk[4..8]),
                f32x4::from_slice(neon, &c_chunk[8..12]),
                f32x4::from_slice(neon, &c_chunk[12..16]),
            );
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
        let coeff = &coefficients[channel * width..(channel + 1) * width];
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
        step_silu_tokens_in_place,
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
    #[allow(
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
}
