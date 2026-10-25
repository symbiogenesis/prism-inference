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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::io::{Cursor, Read};

    use serde::Deserialize;
    use sha2::{Digest, Sha256};

    use super::row;
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

    #[test]
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
            assert!(matches!(width, 1 | 2 | 4));
            assert!(matches!(channels, 64 | 65));
            assert!(history < width && cancellation <= 1);
            let signal = values(&mut input, width * channels)?;
            let coefficients = values(&mut input, width * channels)?;
            let native = values(&mut input, channels)?;
            for (channel, ((signal, coefficients), native)) in signal
                .chunks_exact(width)
                .zip(coefficients.chunks_exact(width))
                .zip(native)
                .enumerate()
            {
                assert_eq!(
                    row(&signal[..width - 1], signal[width - 1], coefficients)?.to_bits(),
                    native.to_bits(),
                    "case {case}, channel {channel}"
                );
            }
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
}
