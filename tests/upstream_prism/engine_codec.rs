//! Adapts supported production Rust codec operations without substitute kernels.

use bonsai_prism_runtime::codec::{self, BufferRegion, CodecBuffers, CodecFunction};
use serde_json::{Value, json};

use super::{Result, invalid};

fn bytes(value: &Value) -> Result<Vec<u8>> {
    let text = value
        .as_str()
        .ok_or_else(|| invalid("Rust codec bytes must be text"))?;
    if text.len() > 2 * 1024 * 1024 || !text.len().is_multiple_of(2) {
        return Err(invalid("Rust codec byte extent exceeds bound"));
    }
    text.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| {
            let nibble = |byte| match byte {
                b'0'..=b'9' => Ok(byte - b'0'),
                b'a'..=b'f' => Ok(byte - b'a' + 10),
                _ => Err(invalid("Rust codec requires lowercase hexadecimal")),
            };
            Ok(16 * nibble(pair[0])? + nibble(pair[1])?)
        })
        .collect()
}

fn byte_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8] = b"0123456789abcdef";
    bytes
        .iter()
        .flat_map(|byte| {
            [
                char::from(DIGITS[usize::from(byte >> 4)]),
                char::from(DIGITS[usize::from(byte & 15)]),
            ]
        })
        .collect()
}

fn header(operation: &Value) -> Result<(u32, usize)> {
    let kind = u32::try_from(
        operation["TypeId"]
            .as_u64()
            .ok_or_else(|| invalid("Rust codec TypeId must be unsigned"))?,
    )?;
    let count = usize::try_from(
        operation["Count"]
            .as_u64()
            .ok_or_else(|| invalid("Rust codec Count must be unsigned"))?,
    )?;
    if count == 0 || count > 8192 {
        return Err(invalid("Rust codec Count outside bound"));
    }
    Ok((kind, count))
}

fn codec_call(operation: &Value) -> Result<Vec<u8>> {
    let (kind, count) = header(operation)?;
    let traits = codec::traits(kind)?;
    let (function, input, rhs, output_bytes) = match operation["Kind"].as_str() {
        Some("codec-quantize") => {
            let function = match operation["Mode"].as_str() {
                Some("cpu") => CodecFunction::FromFloatCpu,
                Some("reference") => CodecFunction::FromFloatReference,
                _ => return Err(invalid("Rust codec quantization Mode outside bound")),
            };
            let input = bytes(&operation["DataHex"])?;
            if count.checked_mul(4) != Some(input.len()) {
                return Err(invalid("Rust codec float input extent differs"));
            }
            (function, input, None, traits.row_bytes(count)?)
        }
        Some("codec-dequantize") => (
            CodecFunction::ToFloat,
            bytes(&operation["DataHex"])?,
            None,
            count * 4,
        ),
        Some("codec-vec-dot") => (
            CodecFunction::VecDot,
            bytes(&operation["WeightsHex"])?,
            Some(bytes(&operation["ActivationsHex"])?),
            4,
        ),
        _ => return Err(invalid("unknown Rust codec operation")),
    };
    let expected_input = match function {
        CodecFunction::FromFloatCpu | CodecFunction::FromFloatReference => count * 4,
        CodecFunction::ToFloat | CodecFunction::VecDot => traits.row_bytes(count)?,
    };
    if input.len() != expected_input
        || rhs.as_ref().is_some_and(|rhs| {
            !codec::traits(traits.vec_dot_type)
                .and_then(|traits| traits.row_bytes(count))
                .is_ok_and(|size| rhs.len() == size)
        })
    {
        return Err(invalid("Rust codec input byte extent differs"));
    }
    let mut buffers = CodecBuffers::default();
    let input_region = BufferRegion {
        buffer: buffers.new_buffer(input.len(), 1, 0)?,
        offset: 0,
    };
    buffers.write(input_region, &input)?;
    let rhs_region = if let Some(rhs) = rhs {
        let region = BufferRegion {
            buffer: buffers.new_buffer(rhs.len(), 1, 0)?,
            offset: 0,
        };
        buffers.write(region, &rhs)?;
        Some(region)
    } else {
        None
    };
    let output_region = BufferRegion {
        buffer: buffers.new_buffer(output_bytes, 1, 0)?,
        offset: 0,
    };
    buffers.call(
        function,
        kind,
        count,
        input_region,
        rhs_region,
        output_region,
    )?;
    Ok(buffers.read(output_region, output_bytes)?.to_vec())
}

pub(super) fn execute(request: &Value) -> Result<Option<Value>> {
    if request["SchemaVersion"] != 1 {
        return Err(invalid("Rust codec request version mismatch"));
    }
    let operations = request["Operations"]
        .as_array()
        .ok_or_else(|| invalid("Rust codec Operations must be an array"))?;
    if operations.is_empty() || operations.len() > 128 {
        return Err(invalid("Rust codec operation count outside bound"));
    }
    let results = operations
        .iter()
        .map(|operation| match operation["Kind"].as_str() {
            Some("codec-inventory") => {
                let types: Vec<_> = codec::inventory()
                    .map(|traits| {
                        json!({
                            "TypeId": traits.type_id,
                            "Name": traits.name,
                            "BlockSize": traits.block_size,
                            "TypeSize": traits.type_size,
                            "HasFromFloatCpu": traits.native.from_float_cpu,
                            "HasFromFloatRef": traits.native.from_float_reference,
                            "HasToFloat": traits.native.to_float,
                            "HasVecDot": traits.native.vec_dot,
                            "VecDotType": traits.vec_dot_type,
                        })
                    })
                    .collect();
                Ok(json!({"Types": types}))
            }
            Some("codec-quantize" | "codec-vec-dot" | "codec-dequantize") => {
                Ok(match codec_call(operation) {
                    Ok(bytes) => json!({"DataHex":byte_hex(&bytes)}),
                    Err(error) => json!({"Error":error.to_string()}),
                })
            }
            _ => Err(invalid("unknown Rust codec operation")),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(json!({"SchemaVersion":1,"Results":results})))
}

#[cfg(test)]
mod focused_guards {
    use super::*;

    #[test]
    #[ignore = "requires a separately captured pinned C++ codec inventory"]
    fn complete_inventory_matches_pinned_cpp_observations() -> Result<()> {
        let path = std::env::var_os("BONSAI_CODEC_NATIVE_INVENTORY")
            .ok_or_else(|| invalid("BONSAI_CODEC_NATIVE_INVENTORY is required"))?;
        let native: Value = serde_json::from_slice(&std::fs::read(path)?)?;
        let rust = execute(&json!({
            "SchemaVersion": 1,
            "Operations": [{"Kind": "codec-inventory"}],
        }))?
        .ok_or_else(|| invalid("Rust codec inventory missing"))?;
        assert_eq!(rust, native);
        Ok(())
    }

    #[test]
    fn inventory_preserves_native_metadata_and_unavailable_cpu_calls_reject() -> Result<()> {
        let response = execute(&json!({
            "SchemaVersion": 1,
            "Operations": [{"Kind": "codec-inventory"}],
        }))?
        .ok_or_else(|| invalid("Rust codec inventory missing"))?;
        let types = response["Results"][0]["Types"]
            .as_array()
            .ok_or_else(|| invalid("Rust codec type inventory missing"))?;
        assert_eq!(types.len(), usize::try_from(codec::TYPE_COUNT)?);
        for (index, entry) in types.iter().enumerate() {
            assert_eq!(entry["TypeId"], index);
        }
        // Q4_0 has a native CPU encoder; the Rust reference encoder must not
        // silently stand in for the unported architecture-specific CPU call.
        assert_eq!(types[2]["HasFromFloatCpu"], true);
        let response = execute(&json!({
            "SchemaVersion": 1,
            "Operations": [{
                "Kind": "codec-quantize", "TypeId": 2, "Mode": "cpu",
                "Count": 32, "DataHex": byte_hex(&[0; 128]),
            }],
        }))?
        .ok_or_else(|| invalid("Rust codec operation missing"))?;
        assert!(response["Results"][0]["Error"].is_string());
        assert!(response["Results"][0].get("DataHex").is_none());
        Ok(())
    }

    #[test]
    fn reference_rows_use_owned_buffers_and_reject_trailing_input() -> Result<()> {
        let input: Vec<_> = [-1.0_f32, 0.0, 1.0, 1.0]
            .into_iter()
            .cycle()
            .take(128)
            .flat_map(f32::to_le_bytes)
            .collect();
        let encoded = codec_call(&json!({
            "Kind": "codec-quantize", "TypeId": 142, "Mode": "reference",
            "Count": 128, "DataHex": byte_hex(&input),
        }))?;
        assert_eq!(encoded, [&[0, 0x3c][..], &[0xa4; 32]].concat());
        let decoded = codec_call(&json!({
            "Kind": "codec-dequantize", "TypeId": 142,
            "Count": 128, "DataHex": byte_hex(&encoded),
        }))?;
        assert_eq!(decoded, input);
        assert!(
            codec_call(&json!({
                "Kind": "codec-dequantize", "TypeId": 142,
                "Count": 128, "DataHex": byte_hex(&[encoded.as_slice(), &[0]].concat()),
            }))
            .is_err()
        );
        Ok(())
    }
}
