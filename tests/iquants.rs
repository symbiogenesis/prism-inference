//! Focused nonlinear IQ4 native encoder, decoder and generic-dot observations.

use bonsai_prism_runtime::{Result, kernels, model::Tensor};
use serde_json::Value;
use std::sync::Arc;

fn bytes(value: &Value) -> Result<Vec<u8>> {
    let text = value
        .as_str()
        .ok_or_else(|| std::io::Error::other("IQ4 fixture hex missing"))?;
    text.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| Ok(u8::from_str_radix(std::str::from_utf8(pair)?, 16)?))
        .collect()
}
fn floats(value: &Value) -> Result<Vec<f32>> {
    Ok(bytes(value)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect())
}
fn number(value: &Value) -> Result<usize> {
    Ok(usize::try_from(value.as_u64().ok_or_else(|| {
        std::io::Error::other("IQ4 fixture number missing")
    })?)?)
}

#[test]
fn native_nonlinear_blocks_preserve_reference_bytes_and_generic_dots() -> Result<()> {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/upstream-iquants.json"))?;
    let cases = fixture["Cases"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("IQ4 fixture cases missing"))?;
    assert_eq!(cases.len(), 10);
    for case in cases {
        let kind = u32::try_from(number(&case["TypeId"])?)?;
        let count = number(&case["Count"])?;
        let packed = bytes(&case["OutputHex"])?;
        let tensor = Tensor {
            name: case["Name"].to_string(),
            kind,
            dimensions: vec![count],
            data: Arc::from(packed.clone()),
        };
        kernels::validate_tensor(&tensor)?;
        let values = kernels::row(&tensor, 0)?;
        assert_eq!(
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
            bytes(&case["DecodedHex"])?,
            "decode {kind}:{}",
            case["Name"]
        );
        let dot = kernels::packed_dot(kind, count, &packed, &bytes(&case["ActivationsHex"])?)?;
        assert_eq!(
            dot.to_le_bytes().as_slice(),
            bytes(&case["GenericDotHex"])?,
            "generic dot {kind}:{}",
            case["Name"]
        );
        if case["InputHex"].is_string() {
            let actual = kernels::quantize_row(kind, &floats(&case["InputHex"])?)?;
            if case["UndefinedZeroScalePayload"] == true {
                for (actual, native) in actual
                    .as_chunks::<18>()
                    .0
                    .iter()
                    .zip(packed.as_chunks::<18>().0)
                {
                    assert_eq!(&actual[..2], &native[..2]);
                    assert!(actual[2..].iter().all(|byte| *byte == 0));
                }
                assert!(values.iter().all(|value| *value == 0.0));
                let canonical = Tensor {
                    data: Arc::from(actual),
                    ..tensor
                };
                assert!(
                    kernels::row(&canonical, 0)?
                        .iter()
                        .all(|value| *value == 0.0)
                );
            } else {
                assert_eq!(actual, packed, "encode {kind}:{}", case["Name"]);
            }
        }
    }
    Ok(())
}

#[test]
fn native_weighted_nonlinear_rows_match_every_defined_byte() -> Result<()> {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/upstream-iquants-weighted.json"))?;
    let cases = fixture["Cases"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("weighted IQ4 cases missing"))?;
    assert_eq!(cases.len(), 16);
    for case in cases {
        let kind = u32::try_from(number(&case["TypeId"])?)?;
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
