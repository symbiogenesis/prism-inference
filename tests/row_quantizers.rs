//! Immutable native-production bytes for focused row-quantizer verification.
#![forbid(unsafe_code)]

use bonsai_prism_runtime::kernels::{
    packed_dot, quantize_row, quantize_row_weighted, row, validate_tensor,
};
use bonsai_prism_runtime::model::Tensor;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::sync::Arc;

type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

fn unhex(value: &Value) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    let value = value.as_str().ok_or("hex fixture must be text")?;
    if !value.len().is_multiple_of(2) {
        return Err("hex fixture length must be even".into());
    }
    value
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| Ok(u8::from_str_radix(std::str::from_utf8(pair)?, 16)?))
        .collect()
}

#[test]
fn row_quantizers_match_native_production_bytes_for_zeros_ties_and_lcg_inputs() -> TestResult {
    let expected_hash = unhex(&Value::String(
        "7b520199c64e045fb2b153df38fe488f3043557c04248727766bd2a6b8944d54".to_owned(),
    ))?;
    assert_eq!(
        Sha256::digest(include_bytes!("fixtures/row-quantizers.json")).as_slice(),
        expected_hash
    );
    let fixture: Value = serde_json::from_str(include_str!("fixtures/row-quantizers.json"))?;
    assert_eq!(fixture["SchemaVersion"], 1);
    assert_eq!(
        fixture["UpstreamCommit"],
        "b87ca7225456b06b0975b8239c66b5aecf07e37c"
    );
    let cases = fixture["Cases"]
        .as_array()
        .ok_or("quantizer Cases must be an array")?;
    assert_eq!(cases.len(), 28);
    for case in cases {
        let kind = u32::try_from(
            case["TypeId"]
                .as_u64()
                .ok_or("fixture type must be unsigned")?,
        )?;
        let input = unhex(&case["InputHex"])?;
        let values = input
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect::<Vec<_>>();
        let output = quantize_row(kind, &values)?;
        assert_eq!(
            output,
            unhex(&case["OutputHex"])?,
            "type={kind}, case={}",
            case["Name"]
        );
        let tensor = Tensor {
            name: "native-quantizer-reference".to_owned(),
            kind,
            dimensions: vec![values.len()],
            data: Arc::from(output),
        };
        validate_tensor(&tensor)?;
        assert_eq!(row(&tensor, 0)?.len(), values.len());
    }
    Ok(())
}

#[test]
fn row_quantizers_reject_incomplete_nonfinite_and_unrepresentable_packed_scales() {
    for kind in [0, 1, 2, 8, 15, 30, 142, 143] {
        assert!(quantize_row(kind, &[]).is_err());
        assert!(quantize_row(kind, &[f32::NAN; 256]).is_err());
    }
    for kind in [2, 8, 15, 142, 143] {
        assert!(quantize_row(kind, &[1.0; 127]).is_err());
    }
    for kind in [1, 2, 8, 142, 143] {
        assert!(quantize_row(kind, &[f32::MAX; 256]).is_err());
    }
    assert!(packed_dot(15, 256, &[0; 292], &[0; 292]).is_err());
}

#[test]
fn weighted_q4_matches_original_native_scale_search_with_ones_and_varied_importance() -> TestResult
{
    let expected_hash = unhex(&Value::String(
        "78cecc196fc25aff174a03c69df01bd90d4ad5357f793c41b547a3abeb866994".to_owned(),
    ))?;
    assert_eq!(
        Sha256::digest(include_bytes!("fixtures/weighted-q4.json")).as_slice(),
        expected_hash
    );
    let fixture: Value = serde_json::from_str(include_str!("fixtures/weighted-q4.json"))?;
    let cases = fixture["Cases"]
        .as_array()
        .ok_or("weighted cases must be array")?;
    assert_eq!(cases.len(), 8);
    let mut differs = false;
    for case in cases {
        let floats = |field: &str| -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
            Ok(unhex(&case[field])?
                .as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| f32::from_le_bytes(*bytes))
                .collect())
        };
        let input = floats("InputHex")?;
        let importance = floats("ImportanceHex")?;
        let output = quantize_row_weighted(2, &input, &importance)?;
        assert_eq!(output, unhex(&case["OutputHex"])?, "case={}", case["Name"]);
        differs |= output != quantize_row(2, &input)?;
        for kind in [8, 142, 143] {
            assert_eq!(
                quantize_row_weighted(kind, &input, &importance)?,
                quantize_row(kind, &input)?
            );
        }
    }
    assert!(
        differs,
        "all-ones importance must still invoke native Q4_0 scale optimization"
    );
    assert!(quantize_row_weighted(2, &[1.0; 32], &[1.0; 31]).is_err());
    assert!(quantize_row_weighted(2, &[1.0; 32], &[-1.0; 32]).is_err());
    Ok(())
}

#[test]
fn legacy_affine_and_five_bit_rows_match_native_bytes_decoders_and_packed_dots() -> TestResult {
    for (bytes, hash, count) in [
        (
            include_bytes!("fixtures/legacy-row-quantizers.json").as_slice(),
            "085a94cd73be5fe410faa6149264a2f6e724bf7ff47412fec969ffaf2e8029dc",
            16,
        ),
        (
            include_bytes!("fixtures/weighted-legacy.json").as_slice(),
            "b9fd8de01c5b7705e373bae311b899bbacec397d2e32db3fcdd7bd52e20a7e52",
            24,
        ),
    ] {
        assert_eq!(
            Sha256::digest(bytes).as_slice(),
            unhex(&Value::String(hash.to_owned()))?
        );
        let fixture: Value = serde_json::from_slice(bytes)?;
        let cases = fixture["Cases"]
            .as_array()
            .ok_or("native cases must be an array")?;
        assert_eq!(cases.len(), count);
        for case in cases {
            let kind = u32::try_from(
                case["TypeId"]
                    .as_u64()
                    .ok_or("native type must be unsigned")?,
            )?;
            let values = unhex(&case["InputHex"])?
                .as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| f32::from_le_bytes(*bytes))
                .collect::<Vec<_>>();
            let output = if case["ImportanceHex"].is_string() {
                let importance = unhex(&case["ImportanceHex"])?
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|bytes| f32::from_le_bytes(*bytes))
                    .collect::<Vec<_>>();
                quantize_row_weighted(kind, &values, &importance)?
            } else {
                quantize_row(kind, &values)?
            };
            assert_eq!(
                output,
                unhex(&case["OutputHex"])?,
                "quantizer type={kind} name={}",
                case["Name"]
            );
            let tensor = Tensor {
                name: "native-legacy-reference".to_owned(),
                kind,
                dimensions: vec![256],
                data: Arc::from(output),
            };
            validate_tensor(&tensor)?;
            if kind == 9 {
                assert!(packed_dot(kind, 256, &tensor.data, &tensor.data).is_err());
                continue;
            }
            let decoded = row(&tensor, 0)?
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>();
            assert_eq!(
                decoded,
                unhex(&case["DecodedHex"])?,
                "decoder type={kind} name={}",
                case["Name"]
            );
            let actual = packed_dot(kind, 256, &tensor.data, &unhex(&case["ActivationsHex"])?)?;
            let expected = f32::from_le_bytes(
                unhex(&case["DotHex"])?
                    .try_into()
                    .map_err(|_| "native dot size differs")?,
            );
            assert!(
                (actual - expected).abs() <= 2e-6 * expected.abs().max(1.0),
                "dot type={kind} name={} expected={expected} actual={actual}",
                case["Name"]
            );
        }
    }
    Ok(())
}
