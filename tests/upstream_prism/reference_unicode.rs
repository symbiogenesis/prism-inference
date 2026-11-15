//! Independent UTF-8 codepoint, bit-pattern decode, and regex-split observations.
//!
//! This module uses only standard library primitives and `fancy_regex` without
//! invoking `bonsai_prism_runtime`, any native bridge, or expected-answer fixtures.
//! It claims no native execution or production tokenizer parity.

use fancy_regex::Regex;
use serde::Deserialize;
use serde_json::{Value, json};

use super::{Result, invalid};

pub(super) const ID: &str = "tests/test-unicode.cpp";

const MAX_OPERATIONS: usize = 128;
const MAX_TEXT_BYTES: usize = 65_536;
const MAX_SPLIT_PIECES: usize = 65_536;
const MAX_EXPRESSIONS: usize = 16;
const MAX_EXPRESSION_BYTES: usize = 8192;
const MAX_BATCH_ITEMS: usize = 512;
const MAX_INPUT_BYTES: usize = 1024 * 1024;
const MAX_INPUT_HEX_BYTES: usize = 2 * MAX_INPUT_BYTES;
const MAX_OUTPUT_BYTES: usize = 4 * 1024 * 1024;
const HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
struct Request {
    schema_version: u32,
    operations: Vec<Operation>,
}

#[derive(Deserialize)]
#[serde(tag = "Kind", rename_all_fields = "PascalCase", deny_unknown_fields)]
enum Operation {
    #[serde(rename = "unicode-regex-split")]
    RegexSplit {
        text: String,
        expressions: Vec<String>,
        byte_encoding: bool,
    },
    #[serde(rename = "unicode-encode")]
    Encode { code_points: Vec<u32> },
    #[serde(rename = "unicode-decode")]
    Decode { inputs_hex: Vec<String> },
}

impl Operation {
    fn validate(&self) -> Result<()> {
        match self {
            Self::RegexSplit {
                text, expressions, ..
            } => validate_regex_split(text, expressions),
            Self::Encode { code_points } => {
                if code_points.len() > MAX_BATCH_ITEMS {
                    return Err(invalid("independent Unicode encode batch exceeds 512"));
                }
                Ok(())
            }
            Self::Decode { inputs_hex } => validate_decode_inputs(inputs_hex),
        }
    }

    fn execute(&self) -> Result<Value> {
        match self {
            Self::RegexSplit {
                text,
                expressions,
                byte_encoding,
            } => execute_regex_split(text, expressions, *byte_encoding),
            Self::Encode { code_points } => execute_encode(code_points),
            Self::Decode { inputs_hex } => execute_decode(inputs_hex),
        }
    }
}

const fn decode_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

const fn is_continuation(byte: u8) -> bool {
    byte & 0xc0 == 0x80
}

const fn is_direct_gpt_byte(byte: u8) -> bool {
    matches!(byte, b'!'..=b'~' | 0xa1..=0xac | 0xae..=u8::MAX)
}

fn validate_regex_split(text: &str, expressions: &[String]) -> Result<()> {
    if text.len() > MAX_TEXT_BYTES
        || expressions.is_empty()
        || expressions.len() > MAX_EXPRESSIONS
        || expressions
            .iter()
            .any(|expression| expression.len() > MAX_EXPRESSION_BYTES)
    {
        return Err(invalid(
            "independent Unicode regex-split input or expression bound exceeded",
        ));
    }
    Ok(())
}

fn validate_decode_inputs(inputs_hex: &[String]) -> Result<()> {
    if inputs_hex.len() > MAX_BATCH_ITEMS {
        return Err(invalid("independent Unicode decode batch exceeds 512"));
    }
    let mut supplied = 0_usize;
    for hex in inputs_hex {
        if !hex.len().is_multiple_of(2)
            || hex.len() > MAX_INPUT_HEX_BYTES
            || !hex.bytes().all(|byte| decode_nibble(byte).is_some())
        {
            return Err(invalid(
                "independent Unicode hex input invalid or outside bound",
            ));
        }
        supplied += hex.len() / 2;
        if supplied > MAX_INPUT_BYTES {
            return Err(invalid("independent Unicode supplied input exceeds 1 MiB"));
        }
    }
    Ok(())
}

fn hex_decode(hex: &str) -> Result<Vec<u8>> {
    let (pairs, remainder) = hex.as_bytes().as_chunks::<2>();
    if !remainder.is_empty() {
        return Err(invalid("independent Unicode hex length must be even"));
    }
    let mut bytes = Vec::with_capacity(pairs.len());
    for &[hi_byte, lo_byte] in pairs {
        let hi = decode_nibble(hi_byte)
            .ok_or_else(|| invalid("independent Unicode hex must be lowercase"))?;
        let lo = decode_nibble(lo_byte)
            .ok_or_else(|| invalid("independent Unicode hex must be lowercase"))?;
        bytes.push((hi << 4) | lo);
    }
    Ok(bytes)
}

fn hex_encode(bytes: &[u8]) -> Result<String> {
    if bytes.len() > MAX_INPUT_BYTES {
        return Err(invalid("independent Unicode output bytes exceed 1 MiB"));
    }
    let mut text = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        text.push(char::from(HEX_DIGITS[usize::from(byte >> 4)]));
        text.push(char::from(HEX_DIGITS[usize::from(byte & 0x0f)]));
    }
    Ok(text)
}

fn encode_codepoint(point: u32) -> Result<Vec<u8>> {
    let raw = match point {
        0..=0x7f => vec![point],
        0x80..=0x7ff => vec![0xc0 | (point >> 6), 0x80 | (point & 0x3f)],
        0x800..=0xffff => vec![
            0xe0 | (point >> 12),
            0x80 | ((point >> 6) & 0x3f),
            0x80 | (point & 0x3f),
        ],
        0x1_0000..=0x0010_ffff => vec![
            0xf0 | (point >> 18),
            0x80 | ((point >> 12) & 0x3f),
            0x80 | ((point >> 6) & 0x3f),
            0x80 | (point & 0x3f),
        ],
        _ => return Err(invalid("invalid codepoint")),
    };
    raw.into_iter()
        .map(|byte| u8::try_from(byte).map_err(Into::into))
        .collect()
}

fn decode_utf8_bitpattern(bytes: &[u8]) -> Vec<u32> {
    let mut points = Vec::with_capacity(bytes.len());
    let mut offset = 0_usize;
    while offset < bytes.len() {
        let lead = bytes[offset];
        let (length, mask) = match lead {
            0x00..=0x7f => (1_usize, 0x7f_u8),
            0xc0..=0xdf => (2, 0x1f),
            0xe0..=0xef => (3, 0x0f),
            0xf0..=0xf7 => (4, 0x07),
            _ => (0, 0),
        };
        if length == 0
            || offset + length > bytes.len()
            || !bytes[offset + 1..offset + length]
                .iter()
                .copied()
                .all(is_continuation)
        {
            points.push(0xfffd);
            offset += 1;
            continue;
        }
        let mut point = u32::from(lead & mask);
        for &byte in &bytes[offset + 1..offset + length] {
            point = (point << 6) | u32::from(byte & 0x3f);
        }
        points.push(point);
        offset += length;
    }
    points
}

fn execute_encode(code_points: &[u32]) -> Result<Value> {
    let mut values = Vec::with_capacity(code_points.len());
    let mut errors = Vec::with_capacity(code_points.len());
    let mut returned = 0_usize;
    for &point in code_points {
        match encode_codepoint(point) {
            Ok(raw) => {
                returned += raw.len();
                if returned > MAX_OUTPUT_BYTES {
                    return Err(invalid("independent Unicode encode output exceeds 4 MiB"));
                }
                values.push(Value::String(hex_encode(&raw)?));
                errors.push(Value::Null);
            }
            Err(error) => {
                values.push(Value::Null);
                let mut message = error.to_string();
                message.truncate(4096);
                errors.push(json!({"Kind": "invalid-argument", "Message": message}));
            }
        }
    }
    Ok(json!({"Values": values, "Errors": errors}))
}

fn execute_decode(inputs_hex: &[String]) -> Result<Value> {
    let mut values = Vec::with_capacity(inputs_hex.len());
    let mut errors = Vec::with_capacity(inputs_hex.len());
    let mut returned = 0_usize;
    for hex in inputs_hex {
        let raw = hex_decode(hex)?;
        let points = decode_utf8_bitpattern(&raw);
        returned += points.len() * 4;
        if returned > MAX_OUTPUT_BYTES {
            return Err(invalid("independent Unicode decode output exceeds 4 MiB"));
        }
        values.push(json!(points));
        errors.push(Value::Null);
    }
    Ok(json!({"Values": values, "Errors": errors}))
}

fn split_by_regex<'a>(pieces: &[&'a str], regex: &Regex) -> Result<Vec<&'a str>> {
    let mut next = Vec::new();
    for &piece in pieces {
        let mut end = 0_usize;
        for found in regex.find_iter(piece) {
            let found =
                found.map_err(|error| invalid(format!("Unicode regex match failed: {error}")))?;
            if found.start() > end {
                next.push(&piece[end..found.start()]);
            }
            if found.end() > found.start() {
                next.push(found.as_str());
            }
            end = found.end();
        }
        if end < piece.len() {
            next.push(&piece[end..]);
        }
        if next.len() > MAX_SPLIT_PIECES {
            return Err(invalid("independent Unicode split piece bound exceeded"));
        }
    }
    Ok(next)
}

fn gpt_byte_alphabet() -> Result<[char; 256]> {
    let mut alphabet = ['\0'; 256];
    let mut next_extra = 256_u32;
    for byte in 0..=u8::MAX {
        let scalar = if is_direct_gpt_byte(byte) {
            u32::from(byte)
        } else {
            let allocated = next_extra;
            next_extra += 1;
            allocated
        };
        alphabet[usize::from(byte)] =
            char::from_u32(scalar).ok_or_else(|| invalid("invalid GPT byte encoding scalar"))?;
    }
    Ok(alphabet)
}

fn execute_regex_split(text: &str, expressions: &[String], byte_encoding: bool) -> Result<Value> {
    let compiled = expressions
        .iter()
        .map(|expression| {
            Regex::new(expression)
                .map_err(|error| invalid(format!("invalid Unicode split regex: {error}")))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut pieces = if text.is_empty() {
        Vec::new()
    } else {
        vec![text]
    };
    for regex in &compiled {
        pieces = split_by_regex(&pieces, regex)?;
    }
    let output = if byte_encoding {
        let alphabet = gpt_byte_alphabet()?;
        pieces
            .into_iter()
            .map(|piece| {
                piece
                    .bytes()
                    .map(|byte| alphabet[usize::from(byte)])
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
    } else {
        pieces.into_iter().map(str::to_owned).collect::<Vec<_>>()
    };
    Ok(json!({"Pieces": output}))
}

pub(super) fn handles(value: &Value) -> bool {
    value["Operations"].as_array().is_some_and(|operations| {
        !operations.is_empty()
            && operations.iter().all(|operation| {
                matches!(
                    operation["Kind"].as_str(),
                    Some("unicode-regex-split" | "unicode-encode" | "unicode-decode")
                )
            })
    })
}

pub(super) fn execute(value: &Value) -> Result<Value> {
    if !value["Operations"]
        .as_array()
        .is_some_and(|operations| (1..=MAX_OPERATIONS).contains(&operations.len()))
    {
        return Err(invalid(
            "independent Unicode requires 1 through 128 operations",
        ));
    }
    let request: Request = serde_json::from_value(value.clone())?;
    if request.schema_version != 1 {
        return Err(invalid("unsupported independent Unicode schema version"));
    }
    for operation in &request.operations {
        operation.validate()?;
    }
    let results = request
        .operations
        .iter()
        .map(Operation::execute)
        .collect::<Result<Vec<_>>>()?;
    Ok(json!({"SchemaVersion": 1, "Results": results}))
}

pub(super) fn identity() -> Value {
    json!({"Kind": "independent-rust-unicode", "SupportedSourceIds": [ID],
        "UpstreamCommit": super::UPSTREAM,
        "AssertionSource": {"Path": ID, "Sha256": "3f90e347bc639d914a360b197c75f9754d01b751a9a39a240d84b31f8303373a"},
        "Architecture": "utf8-bitpattern-and-regex-subdivision-v1",
        "NativeReferenceReplay": false, "ProductionRuntimeCalled": false,
        "CompleteSourceGate": false, "NativeRetirementQualified": false})
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wrap(operation: Value) -> Value {
        let mut object = serde_json::Map::new();
        object.insert("SchemaVersion".into(), json!(1));
        object.insert("Operations".into(), Value::Array(vec![operation]));
        Value::Object(object)
    }

    #[test]
    fn independent_unicode_preserves_upstream_regex_split_and_subdivision() -> Result<()> {
        let upstream = wrap(json!({
            "Kind": "unicode-regex-split",
            "Text": " ~foo",
            "Expressions": [r"[~][A-Za-z]+| ?[\p{S}]+|\s+"],
            "ByteEncoding": false,
        }));
        assert!(handles(&upstream));
        assert_eq!(
            execute(&upstream)?["Results"][0],
            json!({"Pieces": [" ~", "foo"]})
        );

        let subdivided = wrap(json!({
            "Kind": "unicode-regex-split",
            "Text": "ab12 cd",
            "Expressions": [r"\w+", r"[0-9]+"],
            "ByteEncoding": false,
        }));
        assert_eq!(
            execute(&subdivided)?["Results"][0],
            json!({"Pieces": ["ab", "12", " ", "cd"]})
        );

        let byte_encoded = wrap(json!({
            "Kind": "unicode-regex-split",
            "Text": " ",
            "Expressions": [r"\s+"],
            "ByteEncoding": true,
        }));
        assert_eq!(
            execute(&byte_encoded)?["Results"][0],
            json!({"Pieces": ["Ġ"]})
        );

        let lookahead = wrap(json!({
            "Kind": "unicode-regex-split",
            "Text": "a   b",
            "Expressions": [r"\s+(?!\S)|\s+", r"[a-z]+"],
            "ByteEncoding": false,
        }));
        assert_eq!(
            execute(&lookahead)?["Results"][0],
            json!({"Pieces": ["a", "  ", " ", "b"]})
        );

        let empty_text = wrap(json!({
            "Kind": "unicode-regex-split",
            "Text": "",
            "Expressions": [r"\s+"],
            "ByteEncoding": false,
        }));
        assert_eq!(execute(&empty_text)?["Results"][0], json!({"Pieces": []}));
        Ok(())
    }

    #[test]
    fn independent_unicode_encode_decode_preserves_surrogates_malformed_and_errors() -> Result<()> {
        let encoded = execute(&wrap(json!({
            "Kind": "unicode-encode",
            "CodePoints": [0, 0x7f, 0x80, 0x7ff, 0x800, 0xd800, 0xdc00, 0xdfff, 0xffff, 0x1_0000, 0x0010_ffff, 0x0011_0000],
        })))?;
        let values = encoded["Results"][0]["Values"]
            .as_array()
            .ok_or_else(|| invalid("missing Values"))?;
        let errors = encoded["Results"][0]["Errors"]
            .as_array()
            .ok_or_else(|| invalid("missing Errors"))?;
        assert_eq!(values.len(), 12);
        assert_eq!(values[0], "00");
        assert_eq!(values[5], "eda080");
        assert_eq!(values[6], "edb080");
        assert_eq!(values[10], "f48fbfbf");
        assert!(values[11].is_null());
        assert_eq!(errors[10], Value::Null);
        assert_eq!(errors[11]["Kind"], "invalid-argument");

        let decoded = execute(&wrap(json!({
            "Kind": "unicode-decode",
            "InputsHex": ["00", "eda080", "edb080", "c080", "ffe27880", "f48fbfbf", ""],
        })))?;
        assert_eq!(
            decoded["Results"][0],
            json!({
                "Values": [
                    [0],
                    [0xd800],
                    [0xdc00],
                    [0],
                    [0xfffd, 0xfffd, 0x78, 0xfffd],
                    [0x0010_ffff],
                    []
                ],
                "Errors": [null, null, null, null, null, null, null]
            })
        );

        let info = identity();
        assert_eq!(info["SupportedSourceIds"], json!([ID]));
        assert_eq!(info["NativeReferenceReplay"], false);
        assert_eq!(info["ProductionRuntimeCalled"], false);
        Ok(())
    }

    #[test]
    fn independent_unicode_rejects_session_ops_bad_hex_and_protocol_bounds() {
        assert!(!handles(
            &json!({"Op": "unicode-decode", "InputsHex": ["00"]})
        ));
        assert!(!handles(&json!({"SchemaVersion": 1, "Operations": []})));
        assert!(!handles(&wrap(json!({"Kind": "cpu-threadpool"}))));

        for invalid_request in [
            wrap(json!({
                "Kind": "unicode-regex-split",
                "Text": "text",
                "Expressions": ["["],
                "ByteEncoding": false,
            })),
            wrap(json!({
                "Kind": "unicode-regex-split",
                "Text": "",
                "Expressions": ["["],
                "ByteEncoding": false,
            })),
            wrap(json!({
                "Kind": "unicode-regex-split",
                "Text": "a".repeat(MAX_TEXT_BYTES + 1),
                "Expressions": ["a"],
                "ByteEncoding": false,
            })),
            wrap(json!({
                "Kind": "unicode-regex-split",
                "Text": "a",
                "Expressions": [],
                "ByteEncoding": false,
            })),
            wrap(json!({
                "Kind": "unicode-regex-split",
                "Text": "a",
                "Expressions": vec!["a"; MAX_EXPRESSIONS + 1],
                "ByteEncoding": false,
            })),
            wrap(json!({"Kind": "unicode-decode", "InputsHex": ["FF"]})),
            wrap(json!({"Kind": "unicode-decode", "InputsHex": ["abc"]})),
            wrap(json!({"Kind": "unicode-decode", "InputsHex": vec!["00"; MAX_BATCH_ITEMS + 1]})),
            wrap(json!({"Kind": "unicode-encode", "CodePoints": [-1]})),
            wrap(json!({"Kind": "unicode-encode", "CodePoints": [u64::from(u32::MAX) + 1]})),
            wrap(json!({"Kind": "unicode-encode", "CodePoints": vec![0; MAX_BATCH_ITEMS + 1]})),
            wrap(json!({"Kind": "unicode-decode", "InputsHex": [], "Extra": 1})),
            json!({"SchemaVersion": 2, "Operations": [{"Kind": "unicode-decode", "InputsHex": []}]}),
            json!({"SchemaVersion": 1, "Operations": []}),
            json!({"SchemaVersion": 1, "Operations": vec![json!({"Kind": "unicode-decode", "InputsHex": []}); MAX_OPERATIONS + 1]}),
        ] {
            assert!(execute(&invalid_request).is_err(), "{invalid_request}");
        }
    }
}
