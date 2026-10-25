//! Focused production K-block decoder and packed-dot native observations.

use bonsai_prism_runtime::{Result, kernels, model::Tensor};
use serde_json::Value;
use std::sync::Arc;

fn bytes(value: &Value) -> Result<Vec<u8>> {
    let text = value
        .as_str()
        .ok_or_else(|| std::io::Error::other("fixture hex missing"))?;
    text.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| Ok(u8::from_str_radix(std::str::from_utf8(pair)?, 16)?))
        .collect()
}

#[test]
fn native_k_blocks_decode_and_dot_without_requantizing_activations() -> Result<()> {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/upstream-kquants.json"))?;
    let cases = fixture["Cases"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("fixture cases missing"))?;
    assert_eq!(cases.len(), 25);
    for case in cases {
        let kind = u32::try_from(
            case["TypeId"]
                .as_u64()
                .ok_or_else(|| std::io::Error::other("fixture kind missing"))?,
        )?;
        let count = usize::try_from(
            case["Count"]
                .as_u64()
                .ok_or_else(|| std::io::Error::other("fixture count missing"))?,
        )?;
        let weights = bytes(&case["OutputHex"])?;
        if case["InputHex"].is_string() {
            let input = bytes(&case["InputHex"])?
                .as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| f32::from_le_bytes(*bytes))
                .collect::<Vec<_>>();
            assert_eq!(
                kernels::quantize_row(kind, &input)?,
                weights,
                "quantize {kind}: {}",
                case["Name"]
            );
        }
        let tensor = Tensor {
            name: case["Name"].to_string(),
            kind,
            dimensions: vec![count],
            data: Arc::from(weights.clone()),
        };
        kernels::validate_tensor(&tensor)?;
        let decoded = kernels::row(&tensor, 0)?;
        let decoded_bytes = decoded
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(
            decoded_bytes,
            bytes(&case["DecodedHex"])?,
            "decode {kind}: {}",
            case["Name"]
        );
        let actual = kernels::packed_dot(kind, count, &weights, &bytes(&case["ActivationsHex"])?)?;
        assert_eq!(
            actual.to_le_bytes().as_slice(),
            bytes(&case["GenericDotHex"])?,
            "dot {kind}: {}",
            case["Name"]
        );
    }
    Ok(())
}

#[test]
fn k_quantizers_reject_partial_nonfinite_and_overflowing_blocks() {
    for kind in 10..=14 {
        assert!(kernels::quantize_row(kind, &[0.0; 255]).is_err());
        assert!(kernels::quantize_row(kind, &[f32::NAN; 256]).is_err());
        assert!(kernels::quantize_row(kind, &[f32::INFINITY; 256]).is_err());
        assert!(kernels::quantize_row(kind, &[f32::MAX; 256]).is_err());
    }
}

#[test]
fn native_importance_weighted_k_encoders_preserve_every_packed_byte() -> Result<()> {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/upstream-kquants-weighted.json"))?;
    let cases = fixture["Cases"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("weighted fixture cases missing"))?;
    assert_eq!(cases.len(), 40);
    for case in cases {
        let kind = u32::try_from(
            case["TypeId"]
                .as_u64()
                .ok_or_else(|| std::io::Error::other("weighted fixture type missing"))?,
        )?;
        let floats = |value: &Value| -> Result<Vec<f32>> {
            Ok(bytes(value)?
                .as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| f32::from_le_bytes(*bytes))
                .collect())
        };
        assert_eq!(
            kernels::quantize_row_weighted(
                kind,
                &floats(&case["InputHex"])?,
                &floats(&case["ImportanceHex"])?
            )?,
            bytes(&case["OutputHex"])?,
            "weighted {kind}:{}",
            case["Name"]
        );
    }
    Ok(())
}
