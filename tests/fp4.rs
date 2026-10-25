//! Focused native FP4 packed-layout and exact generic-dot observations.

use bonsai_prism_runtime::{Result, kernels, model::Tensor};
use serde_json::Value;
use std::sync::Arc;

fn bytes(value: &Value) -> Result<Vec<u8>> {
    let text = value
        .as_str()
        .ok_or_else(|| std::io::Error::other("FP4 fixture hex missing"))?;
    text.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| Ok(u8::from_str_radix(std::str::from_utf8(pair)?, 16)?))
        .collect()
}

#[test]
fn native_fp4_rows_preserve_bytes_decoder_and_generic_dot() -> Result<()> {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/upstream-fp4.json"))?;
    let cases = fixture["Cases"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("FP4 cases missing"))?;
    assert_eq!(cases.len(), 10);
    for case in cases {
        let kind = u32::try_from(
            case["TypeId"]
                .as_u64()
                .ok_or_else(|| std::io::Error::other("FP4 kind missing"))?,
        )?;
        let count = usize::try_from(
            case["Count"]
                .as_u64()
                .ok_or_else(|| std::io::Error::other("FP4 count missing"))?,
        )?;
        let packed = bytes(&case["OutputHex"])?;
        let tensor = Tensor {
            name: case["Name"].to_string(),
            kind,
            dimensions: vec![count],
            data: Arc::from(packed.clone()),
        };
        kernels::validate_tensor(&tensor)?;
        assert_eq!(
            kernels::row(&tensor, 0)?
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
            bytes(&case["DecodedHex"])?,
            "decode {kind}:{}",
            case["Name"]
        );
        assert_eq!(
            kernels::packed_dot(kind, count, &packed, &bytes(&case["ActivationsHex"])?)?
                .to_le_bytes()
                .as_slice(),
            bytes(&case["GenericDotHex"])?,
            "generic dot {kind}:{}",
            case["Name"]
        );
        if case["InputHex"].is_string() {
            let input = bytes(&case["InputHex"])?
                .as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| f32::from_le_bytes(*bytes))
                .collect::<Vec<_>>();
            assert_eq!(
                kernels::quantize_row(kind, &input)?,
                packed,
                "encode {kind}:{}",
                case["Name"]
            );
            assert_eq!(
                kernels::quantize_row_weighted(kind, &input, &vec![0.5; count])?,
                packed,
                "native FP4 importance is explicitly unused"
            );
        }
    }
    Ok(())
}
