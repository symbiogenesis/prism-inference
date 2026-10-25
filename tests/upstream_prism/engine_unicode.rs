//! Bounded serialization for the runtime's raw integer Unicode primitives.

use bonsai_prism_runtime::unicode;
use serde_json::{Value, json};

use super::generated_inputs::hex;
use super::llama_grammar::{array, exact_fields};
use super::{Result, invalid};

fn bytes(value: &Value) -> Result<Vec<u8>> {
    let text = value
        .as_str()
        .ok_or_else(|| invalid("Unicode byte input must be hexadecimal text"))?;
    if text.len() % 2 != 0
        || text.len() > 2 * 1024 * 1024
        || !text.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(invalid("invalid bounded Unicode byte input"));
    }
    text.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| Ok(u8::from_str_radix(std::str::from_utf8(pair)?, 16)?))
        .collect()
}

fn operation(operation: &Value) -> Result<Value> {
    let encode = operation["Kind"] == "unicode-encode";
    let key = if encode { "CodePoints" } else { "InputsHex" };
    exact_fields(operation, &["Kind", key])?;
    let inputs = array(operation, key)?;
    if inputs.len() > 512 {
        return Err(invalid("Unicode batch exceeds512"));
    }
    let mut values = Vec::new();
    let mut errors = Vec::new();
    let mut supplied = 0;
    let mut returned = 0;
    for input in inputs {
        let result = if encode {
            let point = u32::try_from(
                input
                    .as_u64()
                    .ok_or_else(|| invalid("Unicode point must be unsigned"))?,
            )?;
            unicode::codepoint_to_utf8(point).and_then(|bytes| {
                returned += bytes.len();
                Ok(json!(hex(&bytes)?))
            })
        } else {
            let input = bytes(input)?;
            supplied += input.len();
            if supplied > 1024 * 1024 {
                return Err(invalid("Unicode input exceeds1MiB"));
            }
            unicode::codepoints_from_utf8(&input).map(|points| {
                returned += points.len() * 4;
                json!(points)
            })
        };
        if returned > 4 * 1024 * 1024 {
            return Err(invalid("Unicode output exceeds4MiB"));
        }
        match result {
            Ok(value) => {
                values.push(value);
                errors.push(Value::Null);
            }
            Err(error) => {
                values.push(Value::Null);
                errors.push(json!({"Kind":"invalid-argument","Message":error.to_string()}));
            }
        }
    }
    Ok(json!({"Values":values,"Errors":errors}))
}

pub(super) fn execute(request: &Value) -> Result<Option<Value>> {
    let Some(operations) = request["Operations"].as_array() else {
        return Ok(None);
    };
    if !operations.iter().all(|operation| {
        matches!(
            operation["Kind"].as_str(),
            Some("unicode-encode" | "unicode-decode")
        )
    }) {
        return Ok(None);
    }
    exact_fields(request, &["SchemaVersion", "Operations"])?;
    if request["SchemaVersion"] != 1 || operations.is_empty() || operations.len() > 128 {
        return Err(invalid("invalid Unicode transport framing"));
    }
    Ok(Some(
        json!({"SchemaVersion":1,"Results":operations.iter().map(operation).collect::<Result<Vec<_>>>()?}),
    ))
}
