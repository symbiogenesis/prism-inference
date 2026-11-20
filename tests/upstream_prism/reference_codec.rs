//! Clean-room safe Rust codec reference provider for upstream quantization and PTQ-CUDA conformance.
//!
//! Independently implements protocol validation, pinned 144-slot trait inventory,
//! IEEE-754 binary16/bfloat16 round-ties-to-even conversions, scalar/legacy/K-quant/
//! IQ/FP4/sign/ternary quantizers, `to_float` decoders, and `vec_dot` kernels.
//! Never imports or calls `bonsai_prism_runtime`, `bonsai_ternary`, `bonsai_gguf`,
//! any native bridge executable, or precomputed answer fixtures.

use std::collections::HashMap;

use serde_json::{Value, json};

use super::{Result, invalid};

/// Upstream conformance route identifiers backed by this codec reference provider.
pub(super) const IDS: [&str; 2] = [
    "tests/test-quantize-fns.cpp",
    "tests/test-ptq1_0-cuda-dot.cpp",
];

const QUANTIZE_SHA256: &str = "830e197b34efc57030cef3b83449264f8b2d4db76121796ae962602cdf675d6b";
const CUDA_SHA256: &str = "17d72e91495a7ffe4e726e90ebe99bf9d7b9acbe1d451a38c865352c3559092a";

pub(super) const TYPE_COUNT: usize = 144;
const MAX_OPS: usize = 128;
const MAX_COUNT: usize = 8192;
const MAX_BYTES: usize = 1024 * 1024;
const MAX_BLOCK_DIM: usize = 4096;

#[expect(
    clippy::struct_excessive_bools,
    reason = "Mirrors the four independent native codec capability flags."
)]
#[derive(Clone, Copy)]
pub(super) struct Traits {
    pub(super) type_id: usize,
    pub(super) name: Option<&'static str>,
    pub(super) block_size: usize,
    pub(super) type_size: usize,
    pub(super) from_float_cpu: bool,
    pub(super) from_float_ref: bool,
    pub(super) to_float: bool,
    pub(super) vec_dot: bool,
    pub(super) vec_dot_type: usize,
}

const fn type_name(id: usize) -> Option<&'static str> {
    Some(match id {
        0 => "f32",
        1 => "f16",
        2 => "q4_0",
        3 => "q4_1",
        4 | 5 => "DEPRECATED",
        6 => "q5_0",
        7 => "q5_1",
        8 => "q8_0",
        9 => "q8_1",
        10 => "q2_K",
        11 => "q3_K",
        12 => "q4_K",
        13 => "q5_K",
        14 => "q6_K",
        15 => "q8_K",
        16 => "iq2_xxs",
        17 => "iq2_xs",
        18 => "iq3_xxs",
        19 => "iq1_s",
        20 => "iq4_nl",
        21 => "iq3_s",
        22 => "iq2_s",
        23 => "iq4_xs",
        24 => "i8",
        25 => "i16",
        26 => "i32",
        27 => "i64",
        28 => "f64",
        29 => "iq1_m",
        30 => "bf16",
        31 => "TYPE_Q4_0_4_4 REMOVED, use Q4_0 with runtime repacking",
        32 => "TYPE_Q4_0_4_8 REMOVED, use Q4_0 with runtime repacking",
        33 => "TYPE_Q4_0_8_8 REMOVED, use Q4_0 with runtime repacking",
        34 => "tq1_0",
        35 => "tq2_0",
        36 => "TYPE_IQ4_NL_4_4 REMOVED, use IQ4_NL with runtime repacking",
        37 => "TYPE_IQ4_NL_4_8 REMOVED, use IQ4_NL with runtime repacking",
        38 => "TYPE_IQ4_NL_8_8 REMOVED, use IQ4_NL with runtime repacking",
        39 => "mxfp4",
        40 => "nvfp4",
        41 => "q1_0",
        42 => "q2_0",
        142 => "pq2_0",
        143 => "ptq1_0",
        _ => return None,
    })
}

const fn layout(id: usize) -> (usize, usize) {
    match id {
        0 | 26 => (1, 4),
        1 | 25 | 30 => (1, 2),
        2 | 20 => (32, 18),
        3 => (32, 20),
        6 => (32, 22),
        7 => (32, 24),
        8 => (32, 34),
        9 => (32, 36),
        10 => (256, 84),
        11 | 21 => (256, 110),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        15 => (256, 292),
        16 | 35 => (256, 66),
        17 => (256, 74),
        18 => (256, 98),
        19 => (256, 50),
        22 => (256, 82),
        23 => (256, 136),
        24 => (1, 1),
        27 | 28 => (1, 8),
        29 => (256, 56),
        34 => (256, 54),
        39 => (32, 17),
        40 => (64, 36),
        41 => (128, 18),
        42 => (64, 18),
        142 => (128, 34),
        143 => (128, 28),
        _ => (0, 0),
    }
}

pub(super) const fn traits_of(id: usize) -> Traits {
    let (block_size, type_size) = layout(id);
    let from_float_cpu = matches!(
        id,
        0..=3 | 6..=15 | 20 | 23 | 26 | 30 | 34 | 35 | 39..=42 | 142 | 143
    );
    let from_float_ref = matches!(
        id,
        1..=3 | 6..=14 | 18 | 20..=23 | 30 | 34 | 35 | 39..=42 | 142 | 143
    );
    let to_float = matches!(
        id,
        1..=3 | 6..=8 | 10..=14 | 16..=23 | 29 | 30 | 34 | 35 | 39..=42 | 142 | 143
    );
    let vec_dot = id == 0 || to_float;
    let vec_dot_type = match id {
        1 => 1,
        2 | 6 | 8 | 20 | 39..=42 | 143 => 8,
        3 | 7 | 9 => 9,
        10..=14 | 16..=19 | 21..=23 | 29 | 34 | 35 | 142 => 15,
        30 => 30,
        _ => 0,
    };
    Traits {
        type_id: id,
        name: type_name(id),
        block_size,
        type_size,
        from_float_cpu,
        from_float_ref,
        to_float,
        vec_dot,
        vec_dot_type,
    }
}

// ---------------------------------------------------------------------------
// Wire protocol validation and serialization
// ---------------------------------------------------------------------------

fn require(condition: bool, message: &'static str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(invalid(message))
    }
}

fn exact_fields<'a>(
    value: &'a Value,
    names: &[&str],
) -> Result<&'a serde_json::Map<String, Value>> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid("codec value must be an object"))?;
    require(
        object.len() == names.len() && names.iter().all(|name| object.contains_key(*name)),
        "codec object fields mismatch",
    )?;
    Ok(object)
}

fn uint_bound(value: &Value, maximum: usize) -> Result<usize> {
    let number = value
        .as_u64()
        .ok_or_else(|| invalid("codec integer must be nonnegative"))?;
    let converted = usize::try_from(number)?;
    require(converted <= maximum, "codec integer exceeds bound")?;
    Ok(converted)
}

fn unhex(value: &Value, expected: usize) -> Result<Vec<u8>> {
    let text = value
        .as_str()
        .ok_or_else(|| invalid("codec DataHex must be text"))?;
    require(expected <= MAX_BYTES, "codec byte extent exceeds bound")?;
    require(text.len() == expected * 2, "codec DataHex extent mismatch")?;
    require(
        text.bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "codec DataHex must use lowercase hexadecimal",
    )?;
    Ok(hex::decode(text)?)
}

fn decode_floats(value: &Value, count: usize) -> Result<Vec<f32>> {
    let bytes = unhex(value, count * 4)?;
    let mut result = Vec::with_capacity(count);
    for chunk in bytes.as_chunks::<4>().0 {
        let number = f32::from_le_bytes(*chunk);
        require(number.is_finite(), "codec float input must be finite")?;
        result.push(number);
    }
    Ok(result)
}

fn encode_floats(values: &[f32]) -> Result<String> {
    let byte_len = values
        .len()
        .checked_mul(4)
        .ok_or_else(|| invalid("float byte length overflow"))?;
    require(byte_len <= MAX_BYTES, "codec output exceeds byte bound")?;
    let mut raw = Vec::with_capacity(byte_len);
    for &value in values {
        require(value.is_finite(), "codec float output must be finite")?;
        raw.extend(value.to_le_bytes());
    }
    Ok(hex::encode(raw))
}

fn extent(id: usize, count: usize) -> Result<usize> {
    let info = traits_of(id);
    require(
        info.block_size > 0
            && info.block_size <= MAX_BLOCK_DIM
            && info.type_size > 0
            && info.type_size <= MAX_BLOCK_DIM,
        "codec type has no active block layout",
    )?;
    require(
        count > 0 && count <= MAX_COUNT && count.is_multiple_of(info.block_size),
        "codec count must contain whole blocks",
    )?;
    let size = (count / info.block_size)
        .checked_mul(info.type_size)
        .ok_or_else(|| invalid("codec packed byte extent overflow"))?;
    require(size <= MAX_BYTES, "codec packed byte extent exceeds bound")?;
    Ok(size)
}

#[derive(Clone, Copy)]
enum QuantMode {
    Cpu,
    Reference,
}

enum ValidatedOp {
    Inventory,
    Quantize {
        type_id: usize,
        _mode: QuantMode,
        input: Vec<f32>,
    },
    Dequantize {
        type_id: usize,
        count: usize,
        packed: Vec<u8>,
    },
    VecDot {
        type_id: usize,
        count: usize,
        weights: Vec<u8>,
        activations: Vec<u8>,
    },
}

fn validate_operation(operation: &Value) -> Result<ValidatedOp> {
    let object = operation
        .as_object()
        .ok_or_else(|| invalid("codec operation must be an object"))?;
    let kind = object
        .get("Kind")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("codec Kind must be text"))?;
    match kind {
        "codec-inventory" => {
            exact_fields(operation, &["Kind"])?;
            Ok(ValidatedOp::Inventory)
        }
        "codec-quantize" => {
            exact_fields(operation, &["Kind", "TypeId", "Mode", "Count", "DataHex"])?;
            let type_id = uint_bound(&operation["TypeId"], TYPE_COUNT - 1)?;
            let count = uint_bound(&operation["Count"], MAX_COUNT)?;
            let _ = extent(type_id, count)?;
            let info = traits_of(type_id);
            let mode_str = operation["Mode"]
                .as_str()
                .ok_or_else(|| invalid("codec quantizer Mode must be text"))?;
            let mode = match mode_str {
                "cpu" if info.from_float_cpu => QuantMode::Cpu,
                "reference" if info.from_float_ref => QuantMode::Reference,
                _ => return Err(invalid("requested production quantizer is unavailable")),
            };
            let input = decode_floats(&operation["DataHex"], count)?;
            Ok(ValidatedOp::Quantize {
                type_id,
                _mode: mode,
                input,
            })
        }
        "codec-dequantize" => {
            exact_fields(operation, &["Kind", "TypeId", "Count", "DataHex"])?;
            let type_id = uint_bound(&operation["TypeId"], TYPE_COUNT - 1)?;
            let count = uint_bound(&operation["Count"], MAX_COUNT)?;
            let size = extent(type_id, count)?;
            let info = traits_of(type_id);
            require(info.to_float, "production decoder is unavailable")?;
            let packed = unhex(&operation["DataHex"], size)?;
            Ok(ValidatedOp::Dequantize {
                type_id,
                count,
                packed,
            })
        }
        "codec-vec-dot" => {
            exact_fields(
                operation,
                &["Kind", "TypeId", "Count", "WeightsHex", "ActivationsHex"],
            )?;
            let type_id = uint_bound(&operation["TypeId"], TYPE_COUNT - 1)?;
            let count = uint_bound(&operation["Count"], MAX_COUNT)?;
            let weight_size = extent(type_id, count)?;
            let info = traits_of(type_id);
            require(
                info.vec_dot && info.vec_dot_type < TYPE_COUNT,
                "production CPU dot is unavailable",
            )?;
            let act_size = extent(info.vec_dot_type, count)?;
            let weights = unhex(&operation["WeightsHex"], weight_size)?;
            let activations = unhex(&operation["ActivationsHex"], act_size)?;
            Ok(ValidatedOp::VecDot {
                type_id,
                count,
                weights,
                activations,
            })
        }
        _ => Err(invalid("unknown codec Kind")),
    }
}

fn execute_operation(op: ValidatedOp) -> Result<Value> {
    match op {
        ValidatedOp::Inventory => {
            let types: Vec<Value> = (0..TYPE_COUNT)
                .map(|id| {
                    let info = traits_of(id);
                    json!({
                        "TypeId": info.type_id,
                        "Name": info.name,
                        "BlockSize": info.block_size,
                        "TypeSize": info.type_size,
                        "HasFromFloatCpu": info.from_float_cpu,
                        "HasFromFloatRef": info.from_float_ref,
                        "HasToFloat": info.to_float,
                        "HasVecDot": info.vec_dot,
                        "VecDotType": info.vec_dot_type,
                    })
                })
                .collect();
            Ok(json!({"Types": types}))
        }
        ValidatedOp::Quantize { type_id, input, .. } => {
            let packed = quantize_row(type_id, &input)?;
            require(packed.len() <= MAX_BYTES, "codec output exceeds byte bound")?;
            Ok(json!({"DataHex": hex::encode(packed)}))
        }
        ValidatedOp::Dequantize {
            type_id,
            count,
            packed,
        } => {
            let decoded = decode_row(type_id, count, &packed)?;
            Ok(json!({"DataHex": encode_floats(&decoded)?}))
        }
        ValidatedOp::VecDot {
            type_id,
            count,
            weights,
            activations,
        } => {
            let dot = vec_dot_row(type_id, count, &weights, &activations)?;
            require(dot.is_finite(), "codec dot output must be finite")?;
            Ok(json!({"DataHex": hex::encode(dot.to_le_bytes())}))
        }
    }
}

/// Reports whether every operation in `request` targets one of the codec wire operations.
pub(super) fn handles(request: &Value) -> bool {
    request["Operations"].as_array().is_some_and(|operations| {
        !operations.is_empty()
            && operations.iter().all(|operation| {
                operation["Kind"].as_str().is_some_and(|kind| {
                    matches!(
                        kind,
                        "codec-inventory" | "codec-quantize" | "codec-dequantize" | "codec-vec-dot"
                    )
                })
            })
    })
}

/// Validates and executes a complete `codec-*` batch request using clean-room safe Rust codecs.
pub(super) fn execute(request: &Value) -> Result<Value> {
    exact_fields(request, &["SchemaVersion", "Operations"])?;
    require(
        uint_bound(&request["SchemaVersion"], 1)? == 1,
        "codec request SchemaVersion must equal 1",
    )?;
    let operations = request["Operations"]
        .as_array()
        .ok_or_else(|| invalid("codec Operations must be an array"))?;
    require(
        !operations.is_empty() && operations.len() <= MAX_OPS,
        "codec Operations must contain 1 through 128 operations",
    )?;
    let validated = operations
        .iter()
        .map(validate_operation)
        .collect::<Result<Vec<_>>>()?;
    let results = validated
        .into_iter()
        .map(execute_operation)
        .collect::<Result<Vec<_>>>()?;
    Ok(json!({"SchemaVersion": 1, "Results": results}))
}

/// Provider identity metadata for clean-room codec conformance reporting.
pub(super) fn identity() -> Value {
    json!({
        "Kind": "independent-rust-codec",
        "SupportedSourceIds": IDS,
        "UpstreamCommit": super::UPSTREAM,
        "AssertionSources": [
            {"Path": IDS[0], "Sha256": QUANTIZE_SHA256},
            {"Path": IDS[1], "Sha256": CUDA_SHA256}
        ],
        "Architecture": "clean-room-scalar-quantize-decode-and-packed-dot-v1",
        "NativeReferenceReplay": false,
        "ProductionRuntimeCalled": false,
        "CompleteSourceGate": false,
        "NativeRetirementQualified": false
    })
}

// ---------------------------------------------------------------------------
// Scalar IEEE-754 conversions and numeric helpers
// ---------------------------------------------------------------------------

const fn widen_u16(value: u16) -> u32 {
    let [b0, b1] = value.to_le_bytes();
    u32::from_le_bytes([b0, b1, 0, 0])
}

const fn narrow_u32_to_u16(value: u32) -> u16 {
    let [b0, b1, _, _] = value.to_le_bytes();
    u16::from_le_bytes([b0, b1])
}

pub(super) const fn f16_to_f32(bits: u16) -> f32 {
    let raw = widen_u16(bits);
    let sign = (raw & 0x8000) << 16;
    let exp = (raw >> 10) & 0x1f;
    let mant = raw & 0x03ff;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let shift = mant.leading_zeros() - 21;
        let norm_mant = ((mant << shift) & 0x03ff) << 13;
        return f32::from_bits(sign | ((113 - shift) << 23) | norm_mant);
    }
    if exp == 31 {
        if mant == 0 {
            return f32::from_bits(sign | 0x7f80_0000);
        }
        return f32::from_bits(sign | 0x7fc0_0000 | (mant << 13));
    }
    f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
}

const fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = narrow_u32_to_u16((bits >> 16) & 0x8000);
    let abs = bits & 0x7fff_ffff;
    let exp = (bits >> 23) & 0xff;
    let mant = bits & 0x007f_ffff;
    if exp == 255 {
        if mant == 0 {
            return sign | 0x7c00;
        }
        return sign | 0x7e00 | narrow_u32_to_u16(mant >> 13);
    }
    if exp > 142 {
        return sign | 0x7c00;
    }
    if exp <= 112 {
        if abs <= 0x3300_0000 {
            return sign;
        }
        let shift = 113 - exp;
        let total = 13 + shift;
        let sig = 0x0080_0000 | mant;
        let sub = sig >> total;
        let half = 1_u32 << (total - 1);
        let rem = sig & ((1_u32 << total) - 1);
        let round_up = rem > half || (rem == half && (sub & 1) != 0);
        let rounded = if round_up { sub + 1 } else { sub };
        return sign | narrow_u32_to_u16(rounded);
    }
    let he = exp - 112;
    let hm = mant >> 13;
    let rem = mant & 0x1fff;
    let round_up = rem > 0x1000 || (rem == 0x1000 && (hm & 1) != 0);
    let rounded_hm = if round_up { hm + 1 } else { hm };
    let combined = (he << 10) + rounded_hm;
    sign | narrow_u32_to_u16(combined)
}

const fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(widen_u16(bits) << 16)
}

const fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        return narrow_u32_to_u16(bits >> 16) | 0x0040;
    }
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7fff + lsb);
    narrow_u32_to_u16(rounded >> 16)
}

fn finite(value: f32) -> Result<f32> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid("nonfinite codec scalar or scale"))
    }
}

fn scale_f16(bytes: &[u8]) -> Result<f32> {
    finite(f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])))
}

fn half_bytes(value: f32) -> Result<[u8; 2]> {
    finite(value)?;
    let bits = f32_to_f16(value);
    finite(f16_to_f32(bits))?;
    Ok(bits.to_le_bytes())
}

fn nearest_int(value: f32) -> Result<i32> {
    if !value.is_finite() || value.abs() > 4_194_303.0 {
        return Err(invalid("quantizer integer conversion outside range"));
    }
    Ok(((value + 12_582_912.0).to_bits() & 0x007f_ffff).cast_signed() - 0x0040_0000)
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Range-checked finite float narrowing to signed 32-bit integer."
)]
fn checked_trunc_i32(value: f32) -> Result<i32> {
    if !value.is_finite() || !(-2_147_483_648.0_f32..2_147_483_648.0_f32).contains(&value) {
        return Err(invalid("float value outside i32 range"));
    }
    Ok(value as i32)
}

fn checked_i8(value: f32) -> Result<i8> {
    let clamped = checked_trunc_i32(value)?;
    i8::try_from(clamped).map_err(|_| invalid("quantized value outside i8 range"))
}

#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Convert bounded integer accumulators to binary32."
)]
const fn i32_to_f32(value: i32) -> f32 {
    value as f32
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "IEEE-754 double-to-float cast matches scalar dot_product."
)]
fn float_dot(left: &[f32], right: &[f32]) -> Result<f32> {
    require(left.len() == right.len(), "float dot length mismatch")?;
    let sum: f64 = left
        .iter()
        .zip(right)
        .map(|(&a, &b)| f64::from(a * b))
        .sum();
    finite(sum as f32)
}

// ---------------------------------------------------------------------------
// Quantizer / Decoder / VecDot top-level dispatch
// ---------------------------------------------------------------------------

pub(super) fn quantize_row(type_id: usize, input: &[f32]) -> Result<Vec<u8>> {
    match type_id {
        0 => Ok(input.iter().flat_map(|v| v.to_le_bytes()).collect()),
        1 => quantize_f16(input),
        30 => quantize_bf16(input),
        26 => quantize_i32(input),
        2 => quantize_q4_0(input),
        3 | 6 | 7 | 9 => quantize_legacy(type_id, input),
        8 => quantize_q8_0(input),
        15 => quantize_q8_k(input),
        10..=14 => quantize_kquant(type_id, input),
        18 | 21 | 22 => quantize_iq_fit(type_id, input),
        20 | 23 => quantize_iq4(type_id, input),
        34 | 35 => quantize_tquant(type_id, input),
        39 | 40 => quantize_fp4(type_id, input),
        41 | 42 => quantize_sign(type_id, input),
        142 | 143 => quantize_packed_ternary(type_id, input),
        _ => Err(invalid("unsupported quantize type")),
    }
}

pub(super) fn decode_row(type_id: usize, count: usize, packed: &[u8]) -> Result<Vec<f32>> {
    let mut output = Vec::with_capacity(count);
    match type_id {
        1 => {
            for chunk in packed.as_chunks::<2>().0 {
                output.push(finite(f16_to_f32(u16::from_le_bytes(*chunk)))?);
            }
        }
        30 => {
            for chunk in packed.as_chunks::<2>().0 {
                output.push(finite(bf16_to_f32(u16::from_le_bytes(*chunk)))?);
            }
        }
        2 => decode_q4_0(packed, &mut output)?,
        3 | 6 | 7 => decode_legacy(type_id, packed, &mut output)?,
        8 => decode_q8_0(packed, &mut output)?,
        10..=14 => decode_kquant(type_id, packed, &mut output)?,
        16 | 17 | 22 => decode_iq2(type_id, packed, &mut output)?,
        18 | 21 => decode_iq3(type_id, packed, &mut output)?,
        19 | 29 => decode_iq1(type_id, packed, &mut output)?,
        20 | 23 => decode_iq4(type_id, packed, &mut output)?,
        34 | 35 => decode_tquant(type_id, packed, &mut output)?,
        39 | 40 => decode_fp4(type_id, packed, &mut output)?,
        41 | 42 => decode_sign(type_id, packed, &mut output)?,
        142 | 143 => decode_packed_ternary(type_id, packed, &mut output)?,
        _ => return Err(invalid("unsupported decoder type")),
    }
    Ok(output)
}

pub(super) fn vec_dot_row(
    type_id: usize,
    count: usize,
    weights: &[u8],
    activations: &[u8],
) -> Result<f32> {
    match type_id {
        0 => {
            let a = bytes_to_f32(weights)?;
            let b = bytes_to_f32(activations)?;
            float_dot(&a, &b)
        }
        1 | 30 => {
            let a = decode_row(type_id, count, weights)?;
            let b = decode_row(type_id, count, activations)?;
            float_dot(&a, &b)
        }
        2 | 8 => dot_q4_0_or_q8_0(type_id, weights, activations),
        3 | 6 | 7 => dot_legacy(type_id, weights, activations),
        10..=14 => {
            let q8k = parse_q8k_blocks(activations)?;
            dot_kquant(type_id, weights, &q8k)
        }
        16 | 17 | 22 => {
            let q8k = parse_q8k_blocks(activations)?;
            dot_iq2(type_id, weights, &q8k)
        }
        18 | 21 => {
            let q8k = parse_q8k_blocks(activations)?;
            dot_iq3(type_id, weights, &q8k)
        }
        19 | 29 => {
            let q8k = parse_q8k_blocks(activations)?;
            dot_iq1(type_id, weights, &q8k)
        }
        20 => dot_iq4_nl(weights, activations),
        23 => {
            let q8k = parse_q8k_blocks(activations)?;
            dot_iq4_xs(weights, &q8k)
        }
        34 | 35 => {
            let q8k = parse_q8k_blocks(activations)?;
            dot_tquant(type_id, weights, &q8k)
        }
        39 | 40 => dot_fp4(type_id, weights, activations),
        41 | 42 => dot_sign(type_id, weights, activations),
        142 => {
            let q8k = parse_q8k_blocks(activations)?;
            dot_pq2(weights, &q8k)
        }
        143 => dot_ptq1(weights, activations),
        _ => Err(invalid("unsupported vec_dot type")),
    }
}

fn bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| finite(f32::from_le_bytes(*chunk)))
        .collect()
}

// ---------------------------------------------------------------------------
// Scalar F16 / BF16 / I32
// ---------------------------------------------------------------------------

fn quantize_f16(input: &[f32]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(input.len() * 2);
    for &value in input {
        out.extend(half_bytes(value)?);
    }
    Ok(out)
}

fn quantize_bf16(input: &[f32]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(input.len() * 2);
    for &value in input {
        finite(value)?;
        let bits = f32_to_bf16(value);
        finite(bf16_to_f32(bits))?;
        out.extend(bits.to_le_bytes());
    }
    Ok(out)
}

fn quantize_i32(input: &[f32]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(input.len() * 4);
    for &value in input {
        out.extend(checked_trunc_i32(value)?.to_le_bytes());
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Legacy block formats: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q8_K
// ---------------------------------------------------------------------------

struct Q8Block {
    scale: f32,
    sum: f32,
    values: [i8; 32],
}

#[derive(Clone)]
struct Q8KBlock {
    scale: f32,
    values: [i8; 256],
    sums: [i16; 16],
}

fn parse_q8_0_blocks(bytes: &[u8]) -> Result<Vec<Q8Block>> {
    bytes
        .as_chunks::<34>()
        .0
        .iter()
        .map(|block| {
            Ok(Q8Block {
                scale: scale_f16(block)?,
                sum: 0.0,
                values: std::array::from_fn(|i| block[2 + i].cast_signed()),
            })
        })
        .collect()
}

fn parse_q8_1_blocks(bytes: &[u8]) -> Result<Vec<Q8Block>> {
    bytes
        .as_chunks::<36>()
        .0
        .iter()
        .map(|block| {
            Ok(Q8Block {
                scale: scale_f16(block)?,
                sum: scale_f16(&block[2..])?,
                values: std::array::from_fn(|i| block[4 + i].cast_signed()),
            })
        })
        .collect()
}

fn parse_q8k_blocks(bytes: &[u8]) -> Result<Vec<Q8KBlock>> {
    bytes
        .as_chunks::<292>()
        .0
        .iter()
        .map(|block| {
            Ok(Q8KBlock {
                scale: finite(f32::from_le_bytes(block[..4].try_into()?))?,
                values: std::array::from_fn(|i| block[4 + i].cast_signed()),
                sums: std::array::from_fn(|i| {
                    i16::from_le_bytes([block[260 + 2 * i], block[261 + 2 * i]])
                }),
            })
        })
        .collect()
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Preserve scalar multiply-then-add association."
)]
fn quantize_q4_0(input: &[f32]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity((input.len() / 32) * 18);
    for values in input.as_chunks::<32>().0 {
        let mut maximum = 0.0_f32;
        for &value in values {
            if value.abs() > maximum.abs() {
                maximum = value;
            }
        }
        let d = maximum / -8.0;
        let inverse = if d == 0.0 { 0.0 } else { finite(1.0 / d)? };
        out.extend(half_bytes(d)?);
        for index in 0..16 {
            let low = u8::try_from(checked_trunc_i32(
                (values[index] * inverse + 8.5).trunc().clamp(0.0, 15.0),
            )?)?;
            let high = u8::try_from(checked_trunc_i32(
                (values[index + 16] * inverse + 8.5)
                    .trunc()
                    .clamp(0.0, 15.0),
            )?)?;
            out.push(low | (high << 4));
        }
    }
    Ok(out)
}

fn decode_q4_0(packed: &[u8], output: &mut Vec<f32>) -> Result<()> {
    for block in packed.as_chunks::<18>().0 {
        let d = scale_f16(block)?;
        for shift in [0, 4] {
            for &byte in &block[2..] {
                output.push(f32::from(i16::from((byte >> shift) & 15) - 8) * d);
            }
        }
    }
    Ok(())
}

fn quantize_q8_0(input: &[f32]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity((input.len() / 32) * 34);
    for values in input.as_chunks::<32>().0 {
        let maximum = values
            .iter()
            .fold(0.0_f32, |acc, &value| acc.max(value.abs()));
        let d = maximum / 127.0;
        let inverse = if d == 0.0 { 0.0 } else { finite(1.0 / d)? };
        out.extend(half_bytes(d)?);
        for &value in values {
            let q = checked_i8((value * inverse).round().clamp(-127.0, 127.0))?;
            out.push(q.cast_unsigned());
        }
    }
    Ok(out)
}

fn decode_q8_0(packed: &[u8], output: &mut Vec<f32>) -> Result<()> {
    for block in packed.as_chunks::<34>().0 {
        let d = scale_f16(block)?;
        for &byte in &block[2..] {
            output.push(f32::from(byte.cast_signed()) * d);
        }
    }
    Ok(())
}

fn quantize_q8_k(input: &[f32]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity((input.len() / 256) * 292);
    for values in input.as_chunks::<256>().0 {
        let mut maximum = 0.0_f32;
        for &value in values {
            if value.abs() > maximum.abs() {
                maximum = value;
            }
        }
        if maximum == 0.0 {
            out.resize(out.len() + 292, 0);
            continue;
        }
        let inverse = -127.0 / maximum;
        require(inverse.is_finite(), "Q8_K scale overflow")?;
        let d = 1.0 / inverse;
        out.extend(d.to_le_bytes());
        let mut qs = [0_i8; 256];
        for (dst, &value) in qs.iter_mut().zip(values) {
            *dst = checked_i8((value * inverse).round_ties_even().clamp(-128.0, 127.0))?;
            out.push(dst.cast_unsigned());
        }
        for group in qs.as_chunks::<16>().0 {
            let sum: i16 = group.iter().map(|&q| i16::from(q)).sum();
            out.extend(sum.to_le_bytes());
        }
    }
    Ok(out)
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Preserve scalar multiply-then-add association."
)]
fn quantize_legacy(type_id: usize, input: &[f32]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for values in input.as_chunks::<32>().0 {
        if type_id == 9 {
            let maximum = values
                .iter()
                .fold(0.0_f32, |acc, &value| acc.max(value.abs()));
            let d = maximum / 127.0;
            let inverse = if d == 0.0 { 0.0 } else { finite(1.0 / d)? };
            let mut packed = [0_i8; 32];
            for (dst, &value) in packed.iter_mut().zip(values) {
                *dst = checked_i8((value * inverse).round().clamp(-127.0, 127.0))?;
            }
            let sum: i32 = packed.iter().map(|&v| i32::from(v)).sum();
            out.extend(half_bytes(d)?);
            out.extend(half_bytes(i32_to_f32(sum) * d)?);
            out.extend(packed.iter().map(|&v| v.cast_unsigned()));
            continue;
        }
        let affine = type_id != 6;
        let mut minimum = f32::MAX;
        let mut maximum = if affine { -f32::MAX } else { 0.0 };
        for &value in values {
            if value < minimum {
                minimum = value;
            }
            if (affine && value > maximum) || (!affine && value.abs() > maximum.abs()) {
                maximum = value;
            }
        }
        let d = if affine {
            (maximum - minimum) / if type_id == 3 { 15.0 } else { 31.0 }
        } else {
            maximum / -16.0
        };
        let inverse = if d == 0.0 { 0.0 } else { finite(1.0 / d)? };
        let limit = if type_id == 3 { 15.0 } else { 31.0 };
        let mut codes = [0_u8; 32];
        for (dst, &value) in codes.iter_mut().zip(values) {
            let shifted = if affine {
                (value - minimum) * inverse + 0.5
            } else {
                value * inverse + 16.5
            };
            *dst = u8::try_from(checked_trunc_i32(shifted.trunc().clamp(0.0, limit))?)?;
        }
        out.extend(half_bytes(d)?);
        if affine {
            out.extend(half_bytes(minimum)?);
        }
        if type_id != 3 {
            let mut high = [0_u8; 4];
            for (index, &code) in codes.iter().enumerate() {
                high[index / 8] |= ((code >> 4) & 1) << (index % 8);
            }
            out.extend(high);
        }
        out.extend((0..16).map(|index| (codes[index] & 15) | ((codes[index + 16] & 15) << 4)));
    }
    Ok(out)
}

fn legacy_codes(type_id: usize, block: &[u8]) -> [i16; 32] {
    let affine = matches!(type_id, 3 | 7);
    let five = type_id != 3;
    let offset = if affine { 4 } else { 2 };
    let high = if five {
        &block[offset..offset + 4]
    } else {
        &[]
    };
    let low = &block[offset + if five { 4 } else { 0 }..];
    std::array::from_fn(|index| {
        let mut value = i16::from((low[index % 16] >> (4 * (index / 16))) & 15);
        if five {
            value |= i16::from((high[index / 8] >> (index % 8)) & 1) << 4;
        }
        value - if affine { 0 } else { 16 }
    })
}

fn decode_legacy(type_id: usize, packed: &[u8], output: &mut Vec<f32>) -> Result<()> {
    let size = layout(type_id).1;
    for block in packed.chunks_exact(size) {
        let d = scale_f16(block)?;
        let minimum = if matches!(type_id, 3 | 7) {
            scale_f16(&block[2..])?
        } else {
            0.0
        };
        for code in legacy_codes(type_id, block) {
            let value = f32::from(code) * d;
            output.push(finite(if type_id == 6 { value } else { value + minimum })?);
        }
    }
    Ok(())
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Preserve scalar multiply-then-add association."
)]
fn dot_q4_0_or_q8_0(type_id: usize, weights: &[u8], activations: &[u8]) -> Result<f32> {
    let block_size = layout(type_id).1;
    let q8_blocks = parse_q8_0_blocks(activations)?;
    let mut sum = 0.0_f32;
    for (block, act) in weights.chunks_exact(block_size).zip(&q8_blocks) {
        let mut int_sum = 0_i32;
        for i in 0..32 {
            let w = if type_id == 2 {
                i32::from((block[2 + i % 16] >> (4 * (i / 16))) & 15) - 8
            } else {
                i32::from(block[2 + i].cast_signed())
            };
            int_sum += w * i32::from(act.values[i]);
        }
        sum += (scale_f16(block)? * act.scale) * i32_to_f32(int_sum);
    }
    finite(sum)
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Preserve scalar multiply-then-add association."
)]
fn dot_legacy(type_id: usize, weights: &[u8], activations: &[u8]) -> Result<f32> {
    let block_size = layout(type_id).1;
    let q8_blocks = if type_id == 6 {
        parse_q8_0_blocks(activations)?
    } else {
        parse_q8_1_blocks(activations)?
    };
    let mut sum = 0.0_f32;
    for (block, act) in weights.chunks_exact(block_size).zip(&q8_blocks) {
        let codes = legacy_codes(type_id, block);
        let int_sum: i32 = codes
            .iter()
            .zip(&act.values)
            .map(|(&w, &a)| i32::from(w) * i32::from(a))
            .sum();
        let mut val = (scale_f16(block)? * act.scale) * i32_to_f32(int_sum);
        if type_id != 6 {
            val += scale_f16(&block[2..])? * act.sum;
        }
        sum += val;
    }
    finite(sum)
}

// ---------------------------------------------------------------------------
// K-quant formats: Q2_K (10), Q3_K (11), Q4_K (12), Q5_K (13), Q6_K (14)
// ---------------------------------------------------------------------------

const fn unpack_q4_k_scale_min(group: usize, scales: &[u8]) -> (i16, i16) {
    if group < 4 {
        (
            i16::from_le_bytes([scales[group] & 63, 0]),
            i16::from_le_bytes([scales[group + 4] & 63, 0]),
        )
    } else {
        (
            i16::from_le_bytes([
                (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4),
                0,
            ]),
            i16::from_le_bytes([(scales[group + 4] >> 4) | ((scales[group] >> 6) << 4), 0]),
        )
    }
}

struct KBlock {
    values: [i16; 256],
    scales: [i16; 16],
    minima: [i16; 16],
    d: f32,
    minimum: f32,
    group_size: usize,
}

fn unpack_kquant(type_id: usize, bytes: &[u8]) -> Result<KBlock> {
    let mut block = KBlock {
        values: [0; 256],
        scales: [0; 16],
        minima: [0; 16],
        d: 0.0,
        minimum: 0.0,
        group_size: 16,
    };
    match type_id {
        10 => {
            block.d = scale_f16(&bytes[80..82])?;
            block.minimum = scale_f16(&bytes[82..84])?;
            for group in 0..16 {
                block.scales[group] = i16::from(bytes[group] & 15);
                block.minima[group] = i16::from(bytes[group] >> 4);
            }
            for (index, value) in block.values.iter_mut().enumerate() {
                *value = i16::from(
                    (bytes[16 + index / 128 * 32 + index % 32] >> (2 * (index % 128 / 32))) & 3,
                );
            }
        }
        11 => {
            block.d = scale_f16(&bytes[108..110])?;
            for group in 0..16 {
                block.scales[group] = i16::from(
                    ((bytes[96 + group % 8] >> (4 * (group / 8))) & 15)
                        | (((bytes[104 + group % 4] >> (2 * (group / 4))) & 3) << 4),
                ) - 32;
            }
            for (index, value) in block.values.iter_mut().enumerate() {
                *value = i16::from(
                    (bytes[32 + index / 128 * 32 + index % 32] >> (2 * (index % 128 / 32))) & 3,
                ) - if bytes[index % 32] & (1 << (index / 32)) == 0 {
                    4
                } else {
                    0
                };
            }
        }
        12 | 13 => {
            block.group_size = 32;
            block.d = scale_f16(bytes)?;
            block.minimum = scale_f16(&bytes[2..4])?;
            for group in 0..8 {
                (block.scales[group], block.minima[group]) =
                    unpack_q4_k_scale_min(group, &bytes[4..16]);
            }
            let offset = if type_id == 12 { 16 } else { 48 };
            for (index, value) in block.values.iter_mut().enumerate() {
                *value = i16::from(
                    (bytes[offset + index / 64 * 32 + index % 32] >> (4 * (index % 64 / 32))) & 15,
                );
                if type_id == 13 && bytes[16 + index % 32] & (1 << (index / 32)) != 0 {
                    *value += 16;
                }
            }
        }
        14 => {
            block.d = scale_f16(&bytes[208..210])?;
            for group in 0..16 {
                block.scales[group] = i16::from(bytes[192 + group].cast_signed());
            }
            for (index, value) in block.values.iter_mut().enumerate() {
                let part = index % 128 / 32;
                let low = (bytes[index / 128 * 64 + (part % 2) * 32 + index % 32]
                    >> (4 * (part / 2)))
                    & 15;
                let high = (bytes[128 + index / 128 * 32 + index % 32] >> (2 * part)) & 3;
                *value = i16::from(low | (high << 4)) - 32;
            }
        }
        _ => return Err(invalid("unknown K quant kind")),
    }
    Ok(block)
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Preserve scalar multiply-then-subtract association."
)]
fn decode_kquant(type_id: usize, packed: &[u8], output: &mut Vec<f32>) -> Result<()> {
    let size = layout(type_id).1;
    for bytes in packed.chunks_exact(size) {
        let block = unpack_kquant(type_id, bytes)?;
        for index in 0..256 {
            let group = index / block.group_size;
            output.push(finite(
                (block.d * f32::from(block.scales[group])) * f32::from(block.values[index])
                    - block.minimum * f32::from(block.minima[group]),
            )?);
        }
    }
    Ok(())
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Preserve scalar K-quant dot association."
)]
fn dot_kquant(type_id: usize, weights: &[u8], input: &[Q8KBlock]) -> Result<f32> {
    let size = layout(type_id).1;
    let mut sum = 0.0_f32;
    let mut lanes = [0.0_f32; 8];
    for (bytes, q8k) in weights.chunks_exact(size).zip(input) {
        let block = unpack_kquant(type_id, bytes)?;
        let act_scale = finite(q8k.scale)?;
        let mut integers = [0_i32; 8];
        let mut minimum_sum = 0_i32;
        for index in 0..256 {
            integers[index % 8] += i32::from(block.scales[index / block.group_size])
                * i32::from(block.values[index])
                * i32::from(q8k.values[index]);
        }
        for group in 0..16 {
            minimum_sum +=
                i32::from(q8k.sums[group]) * i32::from(block.minima[group * 16 / block.group_size]);
        }
        if type_id == 10 {
            let integer_sum: i32 = integers.into_iter().sum();
            sum += (act_scale * block.d) * i32_to_f32(integer_sum)
                - (act_scale * block.minimum) * i32_to_f32(minimum_sum);
        } else {
            let d = block.d * act_scale;
            for (lane, integer) in lanes.iter_mut().zip(integers) {
                *lane += d * i32_to_f32(integer);
            }
            if matches!(type_id, 12 | 13) {
                sum -= (block.minimum * act_scale) * i32_to_f32(minimum_sum);
            }
        }
    }
    if type_id != 10 {
        for lane in lanes {
            sum += lane;
        }
    }
    finite(sum)
}

fn signed_maximum(input: &[f32]) -> f32 {
    let mut maximum = 0.0_f32;
    let mut magnitude = 0.0_f32;
    for &value in input {
        if value.abs() > magnitude {
            maximum = value;
            magnitude = value.abs();
        }
    }
    maximum
}

#[expect(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::suspicious_operation_groupings,
    reason = "Preserve native affine K-quant normal-equation search and exact zero check."
)]
fn kquant_affine(
    input: &[f32],
    weights: &[f32],
    nmax: i32,
    rmin: f32,
    rdelta: f32,
    steps: usize,
    mad: bool,
) -> Result<(f32, f32, Vec<u8>)> {
    let mut minimum = input[0];
    let mut maximum = input[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * input[0];
    for (&value, &weight) in input[1..].iter().zip(&weights[1..]) {
        if value < minimum {
            minimum = value;
        }
        if value > maximum {
            maximum = value;
        }
        sum_w += weight;
        sum_x += weight * value;
    }
    if minimum > 0.0 {
        minimum = 0.0;
    }
    let mut codes = vec![0_u8; input.len()];
    if maximum == minimum {
        return Ok((0.0, -minimum, codes));
    }
    let nmax_f = i32_to_f32(nmax);
    let inverse = nmax_f / (maximum - minimum);
    let mut scale = 1.0 / inverse;
    let mut best_error = 0.0_f32;
    for ((&value, &weight), code) in input.iter().zip(weights).zip(&mut codes) {
        *code = u8::try_from(nearest_int(inverse * (value - minimum))?.clamp(0, nmax))?;
        let difference = scale * f32::from(*code) + minimum - value;
        best_error += weight
            * if mad {
                difference.abs()
            } else {
                difference * difference
            };
    }
    let mut candidate = vec![0_u8; input.len()];
    for step in 0..=steps {
        let inverse =
            (rmin + rdelta * f32::from(u16::try_from(step)?) + nmax_f) / (maximum - minimum);
        let mut sum_l = 0.0_f32;
        let mut sum_l2 = 0.0_f32;
        let mut sum_xl = 0.0_f32;
        for ((&value, &weight), code) in input.iter().zip(weights).zip(&mut candidate) {
            *code = u8::try_from(nearest_int(inverse * (value - minimum))?.clamp(0, nmax))?;
            let level = f32::from(*code);
            sum_l += weight * level;
            sum_l2 += weight * level * level;
            sum_xl += weight * level * value;
        }
        let determinant = sum_w * sum_l2 - sum_l * sum_l;
        if determinant <= 0.0 {
            continue;
        }
        let mut next_scale = (sum_w * sum_xl - sum_x * sum_l) / determinant;
        let mut next_minimum = (sum_l2 * sum_x - sum_l * sum_xl) / determinant;
        if next_minimum > 0.0 {
            next_minimum = 0.0;
            next_scale = sum_xl / sum_l2;
        }
        let mut error = 0.0_f32;
        for ((&value, &weight), &code) in input.iter().zip(weights).zip(&candidate) {
            let difference = next_scale * f32::from(code) + next_minimum - value;
            error += weight
                * if mad {
                    difference.abs()
                } else {
                    difference * difference
                };
        }
        if error < best_error {
            codes.copy_from_slice(&candidate);
            best_error = error;
            scale = next_scale;
            minimum = next_minimum;
        }
    }
    Ok((finite(scale)?, finite(-minimum)?, codes))
}

#[expect(
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::suspicious_operation_groupings,
    reason = "Preserve native symmetric K-quant iterative refinement."
)]
fn kquant_symmetric(input: &[f32], nmax: i32, three: bool) -> Result<(f32, Vec<u8>)> {
    let maximum = signed_maximum(input);
    let mut levels = vec![0_i32; input.len()];
    if maximum.abs() < 1.0e-15 {
        return Ok((0.0, vec![0; input.len()]));
    }
    let nmax_f = i32_to_f32(nmax);
    let inverse = -nmax_f / maximum;
    let mut sum_lx = 0.0_f32;
    let mut sum_l2 = 0.0_f32;
    for (&value, level) in input.iter().zip(&mut levels) {
        *level = nearest_int(inverse * value)?.clamp(-nmax, nmax - 1);
        let weight = value * value;
        let lf = i32_to_f32(*level);
        sum_lx += weight * value * lf;
        sum_l2 += weight * lf * lf;
    }
    if three {
        for _ in 0..5 {
            let mut changed = false;
            for (&value, level) in input.iter().zip(&mut levels) {
                let weight = value * value;
                let lf = i32_to_f32(*level);
                let mut next_lx = sum_lx - weight * value * lf;
                if next_lx <= 0.0 {
                    continue;
                }
                let mut next_l2 = sum_l2 - weight * lf * lf;
                let next_level = nearest_int(value * next_l2 / next_lx)?.clamp(-nmax, nmax - 1);
                if next_level == *level {
                    continue;
                }
                let nlf = i32_to_f32(next_level);
                next_lx += weight * value * nlf;
                next_l2 += weight * nlf * nlf;
                if next_l2 > 0.0 && next_lx * next_lx * sum_l2 > sum_lx * sum_lx * next_l2 {
                    *level = next_level;
                    sum_lx = next_lx;
                    sum_l2 = next_l2;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let scale = if sum_l2 > 0.0 { sum_lx / sum_l2 } else { 0.0 };
        let packed = levels
            .iter()
            .map(|&level| u8::try_from(level + nmax))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        return Ok((finite(scale)?, packed));
    }
    let mut scale = if sum_l2 == 0.0 { 0.0 } else { sum_lx / sum_l2 };
    let mut best = scale * sum_lx;
    for step in -9_i16..=9 {
        if step == 0 {
            continue;
        }
        let inverse = -(nmax_f + 0.1 * f32::from(step)) / maximum;
        let mut cand_lx = 0.0_f32;
        let mut cand_l2 = 0.0_f32;
        for &value in input {
            let level = i32_to_f32(nearest_int(inverse * value)?.clamp(-nmax, nmax - 1));
            let weight = value * value;
            cand_lx += weight * value * level;
            cand_l2 += weight * level * level;
        }
        if cand_l2 > 0.0 && cand_lx * cand_lx > best * cand_l2 {
            for (&value, level) in input.iter().zip(&mut levels) {
                *level = nearest_int(inverse * value)?.clamp(-nmax, nmax - 1);
            }
            scale = cand_lx / cand_l2;
            best = scale * cand_lx;
        }
    }
    let packed = levels
        .iter()
        .map(|&level| u8::try_from(level + nmax))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok((finite(scale)?, packed))
}

fn pack_kquant_affine(
    type_id: usize,
    input: &[f32],
    codes: &mut [u8; 256],
    scales: &[f32; 16],
    minima: &[f32; 16],
    output: &mut [u8],
) -> Result<()> {
    let groups = if type_id == 10 { 16 } else { 8 };
    let group_size = 256 / groups;
    let limit_i32 = if type_id == 10 { 15 } else { 63 };
    let limit = i32_to_f32(limit_i32);
    let max_scale = scales[..groups].iter().copied().fold(0.0_f32, f32::max);
    let max_minimum = minima[..groups].iter().copied().fold(0.0_f32, f32::max);
    let scale_offset = if type_id == 10 { 0 } else { 4 };
    let inv_scale = if max_scale > 0.0 {
        limit / max_scale
    } else {
        0.0
    };
    let inv_min = if max_minimum > 0.0 {
        limit / max_minimum
    } else {
        0.0
    };
    for group in 0..groups {
        let sc = u8::try_from(nearest_int(inv_scale * scales[group])?.min(limit_i32))?;
        let mn = u8::try_from(nearest_int(inv_min * minima[group])?.min(limit_i32))?;
        if type_id == 10 {
            output[group] = sc | (mn << 4);
        } else if group < 4 {
            output[4 + group] = sc;
            output[8 + group] = mn;
        } else {
            output[8 + group] = (sc & 15) | ((mn & 15) << 4);
            output[group] |= (sc >> 4) << 6;
            output[4 + group] |= (mn >> 4) << 6;
        }
    }
    let d_bytes = half_bytes(max_scale / limit)?;
    let m_bytes = half_bytes(max_minimum / limit)?;
    let d_offset = if type_id == 10 { 80 } else { 0 };
    output[d_offset..d_offset + 2].copy_from_slice(&d_bytes);
    output[d_offset + 2..d_offset + 4].copy_from_slice(&m_bytes);
    let d_val = f16_to_f32(u16::from_le_bytes(d_bytes));
    let m_val = f16_to_f32(u16::from_le_bytes(m_bytes));
    let max_code = match type_id {
        10 => 3,
        12 => 15,
        _ => 31,
    };
    for group in 0..groups {
        let (sc, mn) = if type_id == 10 {
            (i16::from(output[group] & 15), i16::from(output[group] >> 4))
        } else {
            unpack_q4_k_scale_min(group, &output[scale_offset..scale_offset + 12])
        };
        let scale = d_val * f32::from(sc);
        if scale == 0.0 {
            continue;
        }
        let minimum = m_val * f32::from(mn);
        for index in group * group_size..(group + 1) * group_size {
            codes[index] =
                u8::try_from(nearest_int((input[index] + minimum) / scale)?.clamp(0, max_code))?;
        }
    }
    for (index, &code) in codes.iter().enumerate() {
        if type_id == 10 {
            output[16 + index / 128 * 32 + index % 32] |= code << (2 * (index % 128 / 32));
        } else {
            let offset = if type_id == 12 { 16 } else { 48 };
            output[offset + index / 64 * 32 + index % 32] |= (code & 15) << (4 * (index % 64 / 32));
            if type_id == 13 && code > 15 {
                output[16 + index % 32] |= 1 << (index / 32);
            }
        }
    }
    Ok(())
}

fn quantize_kquant_symmetric_block(type_id: usize, input: &[f32], output: &mut [u8]) -> Result<()> {
    let mut codes = [0_u8; 256];
    let mut scales = [0.0_f32; 16];
    let mut maximum = 0.0_f32;
    for (group, values) in input.as_chunks::<16>().0.iter().enumerate() {
        let (scale, levels) =
            kquant_symmetric(values, if type_id == 11 { 4 } else { 32 }, type_id == 11)?;
        scales[group] = scale;
        if scale.abs() > maximum.abs() {
            maximum = scale;
        }
        codes[group * 16..(group + 1) * 16].copy_from_slice(&levels);
    }
    if type_id == 14 && maximum.abs() < 1.0e-15 {
        return Ok(());
    }
    let inverse = if maximum == 0.0 {
        0.0
    } else {
        (if type_id == 11 { -32.0 } else { -128.0 }) / maximum
    };
    let d_bytes = half_bytes(if maximum == 0.0 { 0.0 } else { 1.0 / inverse })?;
    let d_offset = if type_id == 11 { 108 } else { 208 };
    output[d_offset..d_offset + 2].copy_from_slice(&d_bytes);
    let d_val = f16_to_f32(u16::from_le_bytes(d_bytes));
    let mut subscales = [0_i32; 16];
    for group in 0..16 {
        let sc = if type_id == 11 {
            nearest_int(inverse * scales[group])?.clamp(-32, 31)
        } else {
            nearest_int(inverse * scales[group])?.clamp(-128, 127)
        };
        subscales[group] = sc;
        if type_id == 11 {
            let packed = u8::try_from(sc + 32)?;
            output[96 + group % 8] |= (packed & 15) << (4 * (group / 8));
            output[104 + group % 4] |= (packed >> 4) << (2 * (group / 4));
        } else {
            output[192 + group] = i8::try_from(sc)?.cast_unsigned();
        }
    }
    if type_id == 11 && maximum == 0.0 {
        output[96..108].fill(0);
        subscales.fill(-32);
    }
    for (group, &subscale) in subscales.iter().enumerate() {
        let scale = d_val * i32_to_f32(subscale);
        if scale == 0.0 {
            continue;
        }
        let limit = if type_id == 11 { 4 } else { 32 };
        for index in group * 16..(group + 1) * 16 {
            codes[index] =
                u8::try_from(nearest_int(input[index] / scale)?.clamp(-limit, limit - 1) + limit)?;
        }
    }
    for (index, &code) in codes.iter().enumerate() {
        if type_id == 11 {
            if code > 3 {
                output[index % 32] |= 1 << (index / 32);
            }
            output[32 + index / 128 * 32 + index % 32] |= (code & 3) << (2 * (index % 128 / 32));
        } else {
            let part = index % 128 / 32;
            output[index / 128 * 64 + (part % 2) * 32 + index % 32] |=
                (code & 15) << (4 * (part / 2));
            output[128 + index / 128 * 32 + index % 32] |= (code >> 4) << (2 * part);
        }
    }
    Ok(())
}

fn quantize_kquant(type_id: usize, input: &[f32]) -> Result<Vec<u8>> {
    let size = layout(type_id).1;
    let mut output = vec![0_u8; (input.len() / 256) * size];
    for (input_block, out_block) in input
        .as_chunks::<256>()
        .0
        .iter()
        .zip(output.chunks_exact_mut(size))
    {
        if matches!(type_id, 10 | 12 | 13) {
            let mut codes = [0_u8; 256];
            let mut scales = [0.0_f32; 16];
            let mut minima = [0.0_f32; 16];
            let group_size = if type_id == 10 { 16 } else { 32 };
            let nmax = match type_id {
                10 => 3,
                12 => 15,
                _ => 31,
            };
            for (group, values) in input_block.chunks_exact(group_size).enumerate() {
                let squares: f32 = values.iter().map(|&v| v * v).sum();
                let average = (squares / 32.0).sqrt();
                let weights: Vec<f32> = values
                    .iter()
                    .map(|&v| {
                        if type_id == 10 {
                            v.abs()
                        } else {
                            average + v.abs()
                        }
                    })
                    .collect();
                let (scale, minimum, levels) = kquant_affine(
                    values,
                    &weights,
                    nmax,
                    if type_id == 12 { -1.0 } else { -0.5 },
                    0.1,
                    if type_id == 12 { 20 } else { 15 },
                    type_id == 10,
                )?;
                scales[group] = scale;
                minima[group] = minimum;
                codes[group * group_size..(group + 1) * group_size].copy_from_slice(&levels);
            }
            pack_kquant_affine(
                type_id,
                input_block,
                &mut codes,
                &scales,
                &minima,
                out_block,
            )?;
        } else {
            quantize_kquant_symmetric_block(type_id, input_block, out_block)?;
        }
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// IQ4 formats: IQ4_NL (20) and IQ4_XS (23)
// ---------------------------------------------------------------------------

const IQ4_NL_VALUES: [i16; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

fn iq4_best_index(value: f32) -> u8 {
    if value <= f32::from(IQ4_NL_VALUES[0]) {
        return 0;
    }
    if value >= f32::from(IQ4_NL_VALUES[15]) {
        return 15;
    }
    let mut lower = 0_usize;
    let mut upper = 15_usize;
    while upper - lower > 1 {
        let middle = usize::midpoint(lower, upper);
        if value < f32::from(IQ4_NL_VALUES[middle]) {
            upper = middle;
        } else {
            lower = middle;
        }
    }
    if value - f32::from(IQ4_NL_VALUES[upper - 1]) < f32::from(IQ4_NL_VALUES[upper]) - value {
        u8::try_from(upper - 1).unwrap_or(0)
    } else {
        u8::try_from(upper).unwrap_or(15)
    }
}

const fn iq4_xs_subscale(bytes: &[u8], group: usize) -> i16 {
    let high = u16::from_le_bytes([bytes[2], bytes[3]]);
    let low = i16::from_le_bytes([(bytes[4 + group / 2] >> (4 * (group % 2))) & 15, 0]);
    let hi = ((high >> (2 * group)) & 3).cast_signed() * 16;
    low + hi - 32
}

#[expect(
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::suspicious_operation_groupings,
    reason = "Preserve native IQ4 scale search and exact zero checks."
)]
fn quantize_iq4(type_id: usize, input: &[f32]) -> Result<Vec<u8>> {
    let (count, size) = layout(type_id);
    let mut output = vec![0_u8; (input.len() / count) * size];
    let tries = if type_id == 20 { -1_i32 } else { 7_i32 };
    for (input_block, out_block) in input.chunks_exact(count).zip(output.chunks_exact_mut(size)) {
        let mut levels = vec![0_u8; count];
        let mut scales = [0.0_f32; 8];
        let mut max_scale = 0.0_f32;
        for (group, values) in input_block.as_chunks::<32>().0.iter().enumerate() {
            let weights: [f32; 32] = std::array::from_fn(|i| values[i] * values[i]);
            let maximum = signed_maximum(values);
            if maximum.abs() < 1.0e-15 {
                continue;
            }
            let initial = if tries > 0 {
                -maximum / f32::from(IQ4_NL_VALUES[0])
            } else {
                maximum / f32::from(IQ4_NL_VALUES[0])
            };
            let inverse = 1.0 / initial;
            let mut sum_qx = 0.0_f32;
            let mut sum_q2 = 0.0_f32;
            for i in 0..32 {
                let level = iq4_best_index(inverse * values[i]);
                levels[group * 32 + i] = level;
                let q = f32::from(IQ4_NL_VALUES[usize::from(level)]);
                sum_qx += weights[i] * q * values[i];
                sum_q2 += weights[i] * q * q;
            }
            let mut d = if sum_q2 > 0.0 { sum_qx / sum_q2 } else { 0.0 };
            let mut best = d * sum_qx;
            for trial in -tries..=tries {
                let inverse = (i32_to_f32(trial) + f32::from(IQ4_NL_VALUES[0])) / maximum;
                sum_qx = 0.0;
                sum_q2 = 0.0;
                for i in 0..32 {
                    let q =
                        f32::from(IQ4_NL_VALUES[usize::from(iq4_best_index(inverse * values[i]))]);
                    sum_qx += weights[i] * q * values[i];
                    sum_q2 += weights[i] * q * q;
                }
                if sum_q2 > 0.0 && sum_qx * sum_qx > best * sum_q2 {
                    d = sum_qx / sum_q2;
                    best = d * sum_qx;
                }
            }
            scales[group] = finite(d)?;
            if d.abs() > max_scale.abs() {
                max_scale = d;
            }
        }
        if type_id == 23 {
            let d = -max_scale / 32.0;
            out_block[..2].copy_from_slice(&half_bytes(d)?);
            let inverse = if d == 0.0 { 0.0 } else { 1.0 / d };
            let mut high = 0_u16;
            for group in 0..8 {
                let level = nearest_int(inverse * scales[group])?.clamp(-32, 31);
                let local = d * i32_to_f32(level);
                let local_inv = if local == 0.0 { 0.0 } else { 1.0 / local };
                for i in 0..32 {
                    levels[group * 32 + i] =
                        iq4_best_index(local_inv * input_block[group * 32 + i]);
                }
                let packed = u8::try_from(level + 32)?;
                out_block[4 + group / 2] |= (packed & 15) << (4 * (group % 2));
                high |= u16::from(packed >> 4) << (2 * group);
            }
            out_block[2..4].copy_from_slice(&high.to_le_bytes());
        } else {
            out_block[..2].copy_from_slice(&half_bytes(scales[0])?);
            if tries > 0 {
                let inverse = if scales[0] == 0.0 {
                    0.0
                } else {
                    1.0 / scales[0]
                };
                for (level, &value) in levels.iter_mut().zip(input_block) {
                    *level = iq4_best_index(inverse * value);
                }
            }
        }
        let offset = if type_id == 20 { 2 } else { 8 };
        for (i, &level) in levels.iter().enumerate() {
            out_block[offset + i / 32 * 16 + i % 16] |= level << (4 * (i % 32 / 16));
        }
    }
    Ok(output)
}

fn decode_iq4(type_id: usize, packed: &[u8], output: &mut Vec<f32>) -> Result<()> {
    let (count, size) = layout(type_id);
    for block in packed.chunks_exact(size) {
        let base_d = scale_f16(block)?;
        let offset = if type_id == 20 { 2 } else { 8 };
        for group in 0..count / 32 {
            let d = if type_id == 20 {
                base_d
            } else {
                base_d * f32::from(iq4_xs_subscale(block, group))
            };
            for i in 0..32 {
                let code = (block[offset + group * 16 + i % 16] >> (4 * (i / 16))) & 15;
                output.push(finite(d * f32::from(IQ4_NL_VALUES[usize::from(code)]))?);
            }
        }
    }
    Ok(())
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Preserve scalar multiply-then-add association."
)]
fn dot_iq4_nl(weights: &[u8], activations: &[u8]) -> Result<f32> {
    let q8_blocks = parse_q8_0_blocks(activations)?;
    let mut sum = 0.0_f32;
    for (block, act) in weights.as_chunks::<18>().0.iter().zip(&q8_blocks) {
        let mut int_sum = 0_i32;
        for i in 0..32 {
            let code = (block[2 + i % 16] >> (4 * (i / 16))) & 15;
            int_sum += i32::from(act.values[i]) * i32::from(IQ4_NL_VALUES[usize::from(code)]);
        }
        sum += (act.scale * scale_f16(block)?) * i32_to_f32(int_sum);
    }
    finite(sum)
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Preserve scalar multiply-then-add association."
)]
fn dot_iq4_xs(weights: &[u8], input: &[Q8KBlock]) -> Result<f32> {
    let mut sum = 0.0_f32;
    for (block, act) in weights.as_chunks::<136>().0.iter().zip(input) {
        let d = scale_f16(block)? * finite(act.scale)?;
        for group in 0..8 {
            let mut int_sum = 0_i32;
            for i in 0..32 {
                let code = (block[8 + group * 16 + i % 16] >> (4 * (i / 16))) & 15;
                int_sum += i32::from(act.values[group * 32 + i])
                    * i32::from(IQ4_NL_VALUES[usize::from(code)]);
            }
            sum += (d * f32::from(iq4_xs_subscale(block, group))) * i32_to_f32(int_sum);
        }
    }
    finite(sum)
}

// ---------------------------------------------------------------------------
// FP4 formats: MXFP4 (39) and NVFP4 (40)
// ---------------------------------------------------------------------------

const FP4_VALUES: [i16; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

fn mxfp4_scale(value: u8) -> f32 {
    f32::from_bits(if value < 2 {
        0x0020_0000 << value
    } else {
        (u32::from(value) - 1) << 23
    })
}

fn ue4m3_to_f32(value: u8) -> f32 {
    if matches!(value, 0 | 0x7f) {
        return 0.0;
    }
    let exponent = i32::from((value >> 3) & 15);
    let mantissa = f32::from(value & 7);
    if exponent == 0 {
        mantissa * 2.0_f32.powi(-9) * 0.5
    } else {
        (1.0 + mantissa / 8.0) * 2.0_f32.powi(exponent - 7) * 0.5
    }
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Preserve native UE4M3 rounding arithmetic."
)]
fn f32_to_ue4m3(mut value: f32) -> Result<u8> {
    if value <= 0.0 {
        return Ok(0);
    }
    if value > 448.0 {
        value = 448.0;
    }
    let bits = value.to_bits();
    let exponent = ((bits >> 23) & 255).cast_signed() - 127;
    let mantissa = (bits >> 20) & 7;
    let mut encoded_exp = exponent + 7;
    if encoded_exp <= 0 {
        let m = checked_trunc_i32(value * 512.0 + 0.5)?;
        return Ok(if m < 1 { 0 } else { u8::try_from(m.min(7))? });
    }
    if encoded_exp >= 15 {
        return Ok(0x7e);
    }
    let mut encoded_mant = mantissa + ((bits >> 19) & 1);
    if encoded_mant > 7 {
        encoded_mant = 0;
        encoded_exp += 1;
        if encoded_exp >= 15 {
            return Ok(0x7e);
        }
    }
    let encoded_exp_u32 = u32::try_from(encoded_exp)?;
    Ok(u8::try_from((encoded_exp_u32 << 3) | encoded_mant)?)
}

fn fp4_best_index(value: f32, scale: f32) -> u8 {
    let mut best = 0_u8;
    let mut best_err = f32::from(FP4_VALUES[0]).mul_add(scale, -value).abs();
    for (idx, &code) in FP4_VALUES.iter().enumerate().skip(1) {
        let candidate = f32::from(code).mul_add(scale, -value).abs();
        if candidate < best_err {
            best = u8::try_from(idx).unwrap_or(0);
            best_err = candidate;
        }
    }
    best
}

fn quantize_fp4(type_id: usize, input: &[f32]) -> Result<Vec<u8>> {
    let (count, size) = layout(type_id);
    let group_size = if type_id == 39 { 32 } else { 16 };
    let offset = if type_id == 39 { 1 } else { 4 };
    let mut output = vec![0_u8; (input.len() / count) * size];
    for (in_blk, out_blk) in input.chunks_exact(count).zip(output.chunks_exact_mut(size)) {
        for (group, values) in in_blk.chunks_exact(group_size).enumerate() {
            let maximum = values.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
            let d = if type_id == 39 {
                let exponent = if maximum > 0.0 {
                    let e = checked_trunc_i32(maximum.log2().floor())? + 125;
                    u8::try_from(e.clamp(0, 254))?
                } else {
                    0
                };
                out_blk[0] = exponent;
                mxfp4_scale(exponent)
            } else {
                let encoded = f32_to_ue4m3(maximum / 6.0)?;
                out_blk[group] = encoded;
                ue4m3_to_f32(encoded)
            };
            for (i, &value) in values.iter().enumerate() {
                out_blk[offset + group * group_size / 2 + i % (group_size / 2)] |=
                    fp4_best_index(value, d) << (4 * (i / (group_size / 2)));
            }
        }
    }
    Ok(output)
}

fn decode_fp4(type_id: usize, packed: &[u8], output: &mut Vec<f32>) -> Result<()> {
    let (count, size) = layout(type_id);
    let group_size = if type_id == 39 { 32 } else { 16 };
    let offset = if type_id == 39 { 1 } else { 4 };
    for block in packed.chunks_exact(size) {
        for group in 0..count / group_size {
            let d = if type_id == 39 {
                mxfp4_scale(block[0])
            } else {
                ue4m3_to_f32(block[group])
            };
            for i in 0..group_size {
                let code = (block[offset + group * group_size / 2 + i % (group_size / 2)]
                    >> (4 * (i / (group_size / 2))))
                    & 15;
                output.push(finite(f32::from(FP4_VALUES[usize::from(code)]) * d)?);
            }
        }
    }
    Ok(())
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Preserve scalar multiply-then-add association."
)]
fn dot_fp4(type_id: usize, weights: &[u8], activations: &[u8]) -> Result<f32> {
    let (count, size) = layout(type_id);
    let q8_blocks = parse_q8_0_blocks(activations)?;
    let group_size = if type_id == 39 { 32 } else { 16 };
    let offset = if type_id == 39 { 1 } else { 4 };
    let mut sum = 0.0_f32;
    for (block_idx, block) in weights.chunks_exact(size).enumerate() {
        for group in 0..count / group_size {
            let act = &q8_blocks[block_idx * (count / 32) + group * group_size / 32];
            let act_off = (group * group_size) % 32;
            let d = if type_id == 39 {
                mxfp4_scale(block[0])
            } else {
                ue4m3_to_f32(block[group])
            };
            let mut int_sum = 0_i32;
            for i in 0..group_size {
                let code = (block[offset + group * group_size / 2 + i % (group_size / 2)]
                    >> (4 * (i / (group_size / 2))))
                    & 15;
                int_sum +=
                    i32::from(act.values[act_off + i]) * i32::from(FP4_VALUES[usize::from(code)]);
            }
            sum += (act.scale * d) * i32_to_f32(int_sum);
        }
    }
    finite(sum)
}

// ---------------------------------------------------------------------------
// Sign-quantized formats: Q1_0 (41) and Q2_0 (42)
// ---------------------------------------------------------------------------

const fn sign_value(type_id: usize, block: &[u8], index: usize) -> i16 {
    if type_id == 41 {
        i16::from_le_bytes([(block[2 + index / 8] >> (index % 8)) & 1, 0]) * 2 - 1
    } else {
        i16::from_le_bytes([(block[2 + index / 4] >> (2 * (index % 4))) & 3, 0]) - 1
    }
}

fn quantize_sign(type_id: usize, input: &[f32]) -> Result<Vec<u8>> {
    let count = layout(type_id).0;
    let mut output = Vec::with_capacity((input.len() / count) * 18);
    for values in input.chunks_exact(count) {
        let d = if type_id == 41 {
            let sum: f32 = values.iter().map(|v| v.abs()).sum();
            finite(sum / 128.0)?
        } else {
            values.iter().fold(0.0_f32, |acc, &v| acc.max(v.abs()))
        };
        let mut block = [0_u8; 18];
        block[..2].copy_from_slice(&half_bytes(d)?);
        if type_id == 41 {
            for (i, &value) in values.iter().enumerate() {
                if value >= 0.0 {
                    block[2 + i / 8] |= 1 << (i % 8);
                }
            }
        } else {
            let inverse = if d > 0.0 { finite(1.0 / d)? } else { 0.0 };
            for (i, &value) in values.iter().enumerate() {
                let rounded = finite(value * inverse)?.round();
                let code: u8 = if rounded < -0.5 {
                    0
                } else if rounded < 0.5 {
                    1
                } else {
                    2
                };
                block[2 + i / 4] |= code << (2 * (i % 4));
            }
        }
        output.extend(block);
    }
    Ok(output)
}

fn decode_sign(type_id: usize, packed: &[u8], output: &mut Vec<f32>) -> Result<()> {
    let count = layout(type_id).0;
    for block in packed.as_chunks::<18>().0 {
        let d = scale_f16(block)?;
        for i in 0..count {
            let val = if type_id == 41 {
                if sign_value(type_id, block, i) > 0 {
                    d
                } else {
                    -d
                }
            } else {
                f32::from(sign_value(type_id, block, i)) * d
            };
            output.push(finite(val)?);
        }
    }
    Ok(())
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Preserve scalar multiply-then-add association."
)]
fn dot_sign(type_id: usize, weights: &[u8], activations: &[u8]) -> Result<f32> {
    let count = layout(type_id).0;
    let groups = count / 32;
    let q8_blocks = parse_q8_0_blocks(activations)?;
    let mut sum = 0.0_f32;
    for (block, acts) in weights
        .as_chunks::<18>()
        .0
        .iter()
        .zip(q8_blocks.chunks_exact(groups))
    {
        let mut local = 0.0_f32;
        for (g, act) in acts.iter().enumerate() {
            let mut int_sum = 0_i32;
            for (i, &a) in act.values.iter().enumerate() {
                int_sum += i32::from(sign_value(type_id, block, g * 32 + i)) * i32::from(a);
            }
            local += act.scale * i32_to_f32(int_sum);
        }
        sum += scale_f16(block)? * local;
    }
    finite(sum)
}

// ---------------------------------------------------------------------------
// Ternary formats: TQ1_0 (34), TQ2_0 (35), PQ2_0 (142), PTQ1_0 (143)
// ---------------------------------------------------------------------------

const fn build_base3_lut() -> [[i8; 5]; 256] {
    let mut lut = [[0_i8; 5]; 256];
    let mut idx = 0_usize;
    let mut byte = 0_u16;
    while byte < 256 {
        let mut fraction = byte;
        let mut digit = 0_usize;
        while digit < 5 {
            fraction *= 3;
            let trit = (fraction >> 8).to_le_bytes()[0].cast_signed() - 1;
            lut[idx][digit] = trit;
            fraction &= 255;
            digit += 1;
        }
        byte += 1;
        idx += 1;
    }
    lut
}

const fn build_2bit_lut() -> [[i8; 4]; 256] {
    let mut lut = [[0_i8; 4]; 256];
    let mut idx = 0_usize;
    let mut byte = 0_u16;
    while byte < 256 {
        let b = byte.to_le_bytes()[0];
        lut[idx][0] = (b & 3).cast_signed() - 1;
        lut[idx][1] = ((b >> 2) & 3).cast_signed() - 1;
        lut[idx][2] = ((b >> 4) & 3).cast_signed() - 1;
        lut[idx][3] = ((b >> 6) & 3).cast_signed() - 1;
        byte += 1;
        idx += 1;
    }
    lut
}

static BASE3_LUT: [[i8; 5]; 256] = build_base3_lut();
static TWOBIT_LUT: [[i8; 4]; 256] = build_2bit_lut();

fn quantize_tquant(type_id: usize, input: &[f32]) -> Result<Vec<u8>> {
    let size = layout(type_id).1;
    let mut output = vec![0_u8; (input.len() / 256) * size];
    for (in_blk, out_blk) in input
        .as_chunks::<256>()
        .0
        .iter()
        .zip(output.chunks_exact_mut(size))
    {
        let d = in_blk.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        out_blk[size - 2..].copy_from_slice(&half_bytes(d)?);
        let inverse = if d == 0.0 { 0.0 } else { finite(1.0 / d)? };
        let mut levels = [0_u8; 256];
        for (dst, &value) in levels.iter_mut().zip(in_blk) {
            let rounded = checked_trunc_i32((value * inverse).round())?.clamp(-1, 1);
            *dst = u8::try_from(rounded + 1)?;
        }
        if type_id == 35 {
            for (i, &level) in levels.iter().enumerate() {
                out_blk[i / 128 * 32 + i % 32] |= level << (2 * (i % 128 / 32));
            }
        } else {
            for (byte_idx, dst) in out_blk[..52].iter_mut().enumerate() {
                let (first, stride, count) = if byte_idx < 32 {
                    (byte_idx, 32, 5)
                } else if byte_idx < 48 {
                    (160 + byte_idx - 32, 16, 5)
                } else {
                    (240 + byte_idx - 48, 4, 4)
                };
                let mut q = 0_u16;
                for digit in 0..count {
                    q = q * 3 + u16::from(levels[first + digit * stride]);
                }
                if count == 4 {
                    q *= 3;
                }
                *dst = u8::try_from((q * 256).div_ceil(243))?;
            }
        }
    }
    Ok(output)
}

fn unpack_tquant_codes(type_id: usize, block: &[u8]) -> [i8; 256] {
    let mut out = [0_i8; 256];
    if type_id == 35 {
        for half in 0..2 {
            let byte_base = half * 32;
            let elem_base = half * 128;
            for col in 0..32 {
                let digits = TWOBIT_LUT[usize::from(block[byte_base + col])];
                out[elem_base + col] = digits[0];
                out[elem_base + col + 32] = digits[1];
                out[elem_base + col + 64] = digits[2];
                out[elem_base + col + 96] = digits[3];
            }
        }
    } else {
        for col in 0..32 {
            let digits = BASE3_LUT[usize::from(block[col])];
            out[col] = digits[0];
            out[col + 32] = digits[1];
            out[col + 64] = digits[2];
            out[col + 96] = digits[3];
            out[col + 128] = digits[4];
        }
        for col in 0..16 {
            let digits = BASE3_LUT[usize::from(block[32 + col])];
            out[160 + col] = digits[0];
            out[176 + col] = digits[1];
            out[192 + col] = digits[2];
            out[208 + col] = digits[3];
            out[224 + col] = digits[4];
        }
        for col in 0..4 {
            let digits = BASE3_LUT[usize::from(block[48 + col])];
            out[240 + col] = digits[0];
            out[244 + col] = digits[1];
            out[248 + col] = digits[2];
            out[252 + col] = digits[3];
        }
    }
    out
}

fn decode_tquant(type_id: usize, packed: &[u8], output: &mut Vec<f32>) -> Result<()> {
    let size = layout(type_id).1;
    for block in packed.chunks_exact(size) {
        let d = scale_f16(&block[size - 2..])?;
        for code in unpack_tquant_codes(type_id, block) {
            output.push(finite(f32::from(code) * d)?);
        }
    }
    Ok(())
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Preserve scalar multiply-then-add association."
)]
fn dot_tquant(type_id: usize, weights: &[u8], input: &[Q8KBlock]) -> Result<f32> {
    let size = layout(type_id).1;
    let mut sum = 0.0_f32;
    for (block, act) in weights.chunks_exact(size).zip(input) {
        let codes = unpack_tquant_codes(type_id, block);
        let int_sum: i32 = codes
            .iter()
            .zip(&act.values)
            .map(|(&w, &a)| i32::from(w) * i32::from(a))
            .sum();
        let d = scale_f16(&block[size - 2..])?;
        sum += i32_to_f32(int_sum) * (d * finite(act.scale)?);
    }
    finite(sum)
}

fn quantize_packed_ternary(type_id: usize, input: &[f32]) -> Result<Vec<u8>> {
    let size = layout(type_id).1;
    let mut output = Vec::with_capacity((input.len() / 128) * size);
    for values in input.as_chunks::<128>().0 {
        let maximum = values.iter().fold(0.0_f32, |acc, &v| acc.max(v.abs()));
        let inverse = if maximum > 0.0 {
            finite(1.0 / maximum)?
        } else {
            0.0
        };
        let scale = half_bytes(maximum)?;
        let mut codes = [0_u8; 128];
        for (dst, &value) in codes.iter_mut().zip(values) {
            let rounded = checked_trunc_i32((value * inverse).round())?.clamp(-1, 1);
            *dst = u8::try_from(rounded + 1)?;
        }
        if type_id == 142 {
            let mut block = [0_u8; 34];
            block[..2].copy_from_slice(&scale);
            for (i, &c) in codes.iter().enumerate() {
                block[2 + i / 4] |= c << (2 * (i % 4));
            }
            output.extend(block);
        } else {
            let mut block = [0_u8; 28];
            let mut pos = 0_usize;
            for width in [16_usize, 8] {
                let start = pos / 5;
                for m in 0..width {
                    let mut q = 0_u16;
                    for n in 0..5 {
                        q = q * 3 + u16::from(codes[pos + m + n * width]);
                    }
                    block[start + m] = u8::try_from((q * 256).div_ceil(243))?;
                }
                pos += width * 5;
            }
            for h in 0..2 {
                let mut q = 0_u16;
                for n in 0..4 {
                    q = q * 3 + u16::from(codes[120 + h + n * 2]);
                }
                block[24 + h] = u8::try_from((q * 3 * 256).div_ceil(243))?;
            }
            block[26..].copy_from_slice(&scale);
            output.extend(block);
        }
    }
    Ok(output)
}

fn pq2_trits(block: &[u8]) -> [i8; 128] {
    let mut out = [0_i8; 128];
    for (byte_idx, &byte) in block[2..34].iter().enumerate() {
        let digits = TWOBIT_LUT[usize::from(byte)];
        out[byte_idx * 4..byte_idx * 4 + 4].copy_from_slice(&digits);
    }
    out
}

fn ptq1_trits(block: &[u8]) -> [i8; 128] {
    let mut out = [0_i8; 128];
    for m in 0..16 {
        let digits = BASE3_LUT[usize::from(block[m])];
        for (d, &trit) in digits.iter().enumerate() {
            out[d * 16 + m] = trit;
        }
    }
    for m in 0..8 {
        let digits = BASE3_LUT[usize::from(block[16 + m])];
        for (d, &trit) in digits.iter().enumerate() {
            out[80 + d * 8 + m] = trit;
        }
    }
    for h in 0..2 {
        let digits = BASE3_LUT[usize::from(block[24 + h])];
        for (d, &trit) in digits[..4].iter().enumerate() {
            out[120 + d * 2 + h] = trit;
        }
    }
    out
}

fn decode_packed_ternary(type_id: usize, packed: &[u8], output: &mut Vec<f32>) -> Result<()> {
    let size = layout(type_id).1;
    for block in packed.chunks_exact(size) {
        let (trits, d) = if type_id == 142 {
            (pq2_trits(block), scale_f16(block)?)
        } else {
            (ptq1_trits(block), scale_f16(&block[26..])?)
        };
        for trit in trits {
            output.push(f32::from(trit) * d);
        }
    }
    Ok(())
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Preserve exact scalar multiply-then-add association for power-of-two scale parity."
)]
fn dot_pq2(weights: &[u8], q8k: &[Q8KBlock]) -> Result<f32> {
    let mut output = 0.0_f32;
    for (index, block) in weights.as_chunks::<34>().0.iter().enumerate() {
        let act = &q8k[index / 2];
        let start = (index % 2) * 128;
        let d = scale_f16(block)? * finite(act.scale)?;
        let trits = pq2_trits(block);
        let int_sum: i32 = trits
            .iter()
            .zip(&act.values[start..start + 128])
            .map(|(&w, &a)| i32::from(w) * i32::from(a))
            .sum();
        output += d * i32_to_f32(int_sum);
    }
    finite(output)
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Preserve exact scalar multiply-then-add association for power-of-two scale parity."
)]
fn dot_ptq1(weights: &[u8], activations: &[u8]) -> Result<f32> {
    let q8_blocks = parse_q8_0_blocks(activations)?;
    let mut output = 0.0_f32;
    for (index, block) in weights.as_chunks::<28>().0.iter().enumerate() {
        let d_w = scale_f16(&block[26..])?;
        let trits = ptq1_trits(block);
        let mut block_sum = 0.0_f32;
        for (values, act) in trits
            .as_chunks::<32>()
            .0
            .iter()
            .zip(&q8_blocks[index * 4..index * 4 + 4])
        {
            let int_sum: i32 = values
                .iter()
                .zip(&act.values)
                .map(|(&w, &a)| i32::from(w) * i32::from(a))
                .sum();
            block_sum += act.scale * i32_to_f32(int_sum);
        }
        output += d_w * block_sum;
    }
    finite(output)
}

// ---------------------------------------------------------------------------
// IQ1 / IQ2 / IQ3 decoders, dots, and reference quantizers
// ---------------------------------------------------------------------------

const fn read_u16_le(block: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([block[offset], block[offset + 1]])
}

fn iq1_base_scale(type_id: usize, block: &[u8]) -> Result<f32> {
    let bits = if type_id == 19 {
        read_u16_le(block, 0)
    } else {
        (read_u16_le(block, 48) >> 12)
            | ((read_u16_le(block, 50) >> 8) & 0x00f0)
            | ((read_u16_le(block, 52) >> 4) & 0x0f00)
            | (read_u16_le(block, 54) & 0xf000)
    };
    finite(f16_to_f32(bits))
}

const fn iq1_subscale(type_id: usize, block: &[u8], group: usize, half: usize) -> i32 {
    let bits = if type_id == 19 {
        (read_u16_le(block, 34 + 2 * group) >> 12) & 7
    } else {
        (read_u16_le(block, 48 + 2 * (group / 2)) >> (6 * (group % 2) + 3 * half)) & 7
    };
    let [lo, _] = (bits & 7).to_le_bytes();
    2 * i32::from_le_bytes([lo, 0, 0, 0]) + 1
}

fn iq1_grid(type_id: usize, block: &[u8], group: usize, part: usize) -> ([i8; 8], i32) {
    let (low, high, negative) = if type_id == 19 {
        let high = read_u16_le(block, 34 + 2 * group);
        (
            block[2 + group * 4 + part],
            usize::from((high >> (3 * part)) & 7),
            high & 0x8000 != 0,
        )
    } else {
        let high = block[32 + group * 2 + part / 2] >> (4 * (part % 2));
        (
            block[group * 4 + part],
            usize::from(high & 7),
            high & 8 != 0,
        )
    };
    let code = usize::from(low) | (high << 8);
    (
        IQ1_GRID[code].to_le_bytes().map(u8::cast_signed),
        if negative { -1 } else { 1 },
    )
}

fn decode_iq1(type_id: usize, packed: &[u8], output: &mut Vec<f32>) -> Result<()> {
    let size = layout(type_id).1;
    for block in packed.chunks_exact(size) {
        let base = iq1_base_scale(type_id, block)?;
        for group in 0..8 {
            for part in 0..4 {
                let scale = base * i32_to_f32(iq1_subscale(type_id, block, group, part / 2));
                let (grid, shift) = iq1_grid(type_id, block, group, part);
                let shift_f = 0.125 * i32_to_f32(shift);
                for value in grid {
                    output.push(finite(scale * (f32::from(value) + shift_f))?);
                }
            }
        }
    }
    Ok(())
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Preserve scalar IQ1 dot association."
)]
fn dot_iq1(type_id: usize, weights: &[u8], input: &[Q8KBlock]) -> Result<f32> {
    let size = layout(type_id).1;
    let mut result = 0.0_f32;
    for (block, act) in weights.chunks_exact(size).zip(input) {
        let mut integer = 0_i32;
        let mut shift_integer = 0_i32;
        for group in 0..8 {
            let mut halves = [0_i32; 2];
            let mut shift_halves = [0_i32; 2];
            for part in 0..4 {
                let (grid, shift) = iq1_grid(type_id, block, group, part);
                let mut local = 0_i32;
                let mut local_sum = 0_i32;
                for (index, value) in grid.into_iter().enumerate() {
                    let a = i32::from(act.values[group * 32 + part * 8 + index]);
                    local += a * i32::from(value);
                    local_sum += a;
                }
                halves[part / 2] += local;
                shift_halves[part / 2] += local_sum * shift;
            }
            if type_id == 19 {
                let scale = iq1_subscale(type_id, block, group, 0);
                let (_, shift) = iq1_grid(type_id, block, group, 0);
                integer += scale * (halves[0] + halves[1]);
                let supplied_sum =
                    i32::from(act.sums[group * 2]) + i32::from(act.sums[group * 2 + 1]);
                shift_integer += scale * shift * supplied_sum;
            } else {
                for half in 0..2 {
                    let scale = iq1_subscale(type_id, block, group, half);
                    integer += scale * halves[half];
                    shift_integer += scale * shift_halves[half];
                }
            }
        }
        result += (iq1_base_scale(type_id, block)? * finite(act.scale)?)
            * (i32_to_f32(integer) + 0.125 * i32_to_f32(shift_integer));
    }
    finite(result)
}

const fn iq2_sign_pattern(code: u8) -> u8 {
    code | if code.count_ones().is_multiple_of(2) {
        0
    } else {
        128
    }
}

fn iq2_group(type_id: usize, block: &[u8], index: usize) -> ([u8; 8], u8, u8) {
    let group32 = index / 4;
    let local = index % 4;
    match type_id {
        16 => {
            let offset = 2 + group32 * 8;
            let high = u32::from_le_bytes([
                block[offset + 4],
                block[offset + 5],
                block[offset + 6],
                block[offset + 7],
            ]);
            (
                IQ2XXS_GRID[usize::from(block[offset + local])].to_le_bytes(),
                iq2_sign_pattern((high >> (7 * local)).to_le_bytes()[0] & 127),
                block[offset + 7] >> 4,
            )
        }
        17 => {
            let offset = 2 + index * 2;
            let code = read_u16_le(block, offset);
            (
                IQ2XS_GRID[usize::from(code & 511)].to_le_bytes(),
                iq2_sign_pattern(block[offset + 1] >> 1),
                (block[66 + group32] >> (4 * (local / 2))) & 15,
            )
        }
        _ => {
            let code = usize::from(block[2 + index])
                | (usize::from((block[66 + group32] >> (2 * local)) & 3) << 8);
            (
                IQ2S_GRID[code].to_le_bytes(),
                block[34 + index],
                (block[74 + group32] >> (4 * (local / 2))) & 15,
            )
        }
    }
}

fn decode_iq2(type_id: usize, packed: &[u8], output: &mut Vec<f32>) -> Result<()> {
    let size = layout(type_id).1;
    for block in packed.chunks_exact(size) {
        let base_d = scale_f16(block)?;
        for index in 0..32 {
            let (grid, signs, local) = iq2_group(type_id, block, index);
            let d = base_d * (0.5 + f32::from(local)) * 0.25;
            for (lane, value) in grid.into_iter().enumerate() {
                let scaled = d * f32::from(value);
                output.push(finite(if signs & (1 << lane) == 0 {
                    scaled
                } else {
                    -scaled
                })?);
            }
        }
    }
    Ok(())
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Preserve scalar IQ2 dot association."
)]
fn dot_iq2(type_id: usize, weights: &[u8], input: &[Q8KBlock]) -> Result<f32> {
    let size = layout(type_id).1;
    let mut sum = 0.0_f32;
    for (block, act) in weights.chunks_exact(size).zip(input) {
        let d = scale_f16(block)? * finite(act.scale)?;
        let mut integer = 0_i32;
        for index in 0..32 {
            let (grid, signs, local) = iq2_group(type_id, block, index);
            let mut group_sum = 0_i32;
            for (lane, value) in grid.into_iter().enumerate() {
                let prod = i32::from(value) * i32::from(act.values[index * 8 + lane]);
                group_sum += if signs & (1 << lane) == 0 {
                    prod
                } else {
                    -prod
                };
            }
            integer += group_sum * (2 * i32::from(local) + 1);
        }
        sum += d * i32_to_f32(integer);
    }
    finite(0.125 * sum)
}

fn iq3_unpack(type_id: usize, bytes: &[u8]) -> ([i16; 256], [i16; 8]) {
    let mut values = [0_i16; 256];
    let mut scales = [0_i16; 8];
    for (group, subscale) in scales.iter_mut().enumerate() {
        let auxiliary = if type_id == 18 {
            u32::from_le_bytes([
                bytes[66 + group * 4],
                bytes[67 + group * 4],
                bytes[68 + group * 4],
                bytes[69 + group * 4],
            ])
        } else {
            0
        };
        *subscale = if type_id == 18 {
            2 * i16::from(auxiliary.to_le_bytes()[3] >> 4) + 1
        } else {
            2 * i16::from((bytes[106 + group / 2] >> (4 * (group % 2))) & 15) + 1
        };
        for octet in 0..4 {
            let signs = if type_id == 18 {
                let low = (auxiliary >> (7 * octet)).to_le_bytes()[0] & 127;
                low | (u8::from(!low.count_ones().is_multiple_of(2)) << 7)
            } else {
                bytes[74 + group * 4 + octet]
            };
            for half in 0..2 {
                let low = usize::from(bytes[2 + group * 8 + octet * 2 + half]);
                let grid = if type_id == 18 {
                    IQ3XXS_GRID[low]
                } else {
                    let high = usize::from((bytes[66 + group] >> (octet * 2 + half)) & 1);
                    IQ3S_GRID[low | (high << 8)]
                };
                for (index, magnitude) in grid.to_le_bytes().into_iter().enumerate() {
                    let sign = if signs & (1 << (half * 4 + index)) == 0 {
                        1
                    } else {
                        -1
                    };
                    values[group * 32 + octet * 8 + half * 4 + index] = i16::from(magnitude) * sign;
                }
            }
        }
    }
    (values, scales)
}

fn decode_iq3(type_id: usize, packed: &[u8], output: &mut Vec<f32>) -> Result<()> {
    let size = layout(type_id).1;
    for block in packed.chunks_exact(size) {
        let base_d = scale_f16(block)?;
        let (values, scales) = iq3_unpack(type_id, block);
        for (group, chunk) in values.as_chunks::<32>().0.iter().enumerate() {
            let d = if type_id == 18 {
                base_d * (0.5 + f32::from((scales[group] - 1) / 2)) * 0.5
            } else {
                base_d * f32::from(scales[group])
            };
            for &value in chunk {
                output.push(finite(d * f32::from(value))?);
            }
        }
    }
    Ok(())
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Preserve scalar IQ3 dot association."
)]
fn dot_iq3(type_id: usize, weights: &[u8], input: &[Q8KBlock]) -> Result<f32> {
    let size = layout(type_id).1;
    let mut sum = 0.0_f32;
    for (block, act) in weights.chunks_exact(size).zip(input) {
        let (values, scales) = iq3_unpack(type_id, block);
        let mut integer = 0_i32;
        for (group, (&subscale, chunk)) in scales.iter().zip(values.as_chunks::<32>().0).enumerate()
        {
            let mut local = 0_i32;
            for (index, &value) in chunk.iter().enumerate() {
                local += i32::from(value) * i32::from(act.values[group * 32 + index]);
            }
            integer += local * i32::from(subscale);
        }
        let d = scale_f16(block)? * finite(act.scale)?;
        sum += d * i32_to_f32(integer);
    }
    finite(if type_id == 18 { 0.25 * sum } else { sum })
}

struct IqGrid<'a, const N: usize> {
    points: &'a [[u8; N]],
    map: Vec<i16>,
    neighbours: HashMap<usize, Vec<usize>>,
    bits: usize,
    shells: usize,
}

impl<'a, const N: usize> IqGrid<'a, N> {
    fn new(points: &'a [[u8; N]], bits: usize, shells: usize) -> Result<Self> {
        let mut map = vec![-1_i16; 1 << (bits * N)];
        for (index, point) in points.iter().enumerate() {
            let key = Self::key(&point.map(|v| (v - 1) / 2), bits);
            map[key] = i16::try_from(index)?;
        }
        Ok(Self {
            points,
            map,
            neighbours: HashMap::new(),
            bits,
            shells,
        })
    }

    fn key(levels: &[u8], bits: usize) -> usize {
        levels
            .iter()
            .enumerate()
            .fold(0, |acc, (i, &lvl)| acc | (usize::from(lvl) << (bits * i)))
    }

    fn index(&self, levels: &[u8]) -> Result<usize> {
        usize::try_from(self.map[Self::key(levels, self.bits)])
            .map_err(|_| invalid("IQ fitted point absent from grid"))
    }

    fn candidates(&mut self, key: usize, levels: &[u8]) -> &[usize] {
        if !self.neighbours.contains_key(&key) {
            let mut distances: Vec<(i32, usize)> = self
                .points
                .iter()
                .enumerate()
                .map(|(index, point)| {
                    let d = point.iter().zip(levels).fold(0_i32, |sum, (&q, &l)| {
                        let diff = i32::from(q) - (2 * i32::from(l) + 1);
                        sum + diff * diff
                    });
                    (d, index)
                })
                .collect();
            distances.sort_unstable();
            let mut previous = distances[0].0;
            let mut shells = 1_usize;
            let mut neighbours = Vec::new();
            for (distance, index) in distances {
                if distance > previous {
                    if shells == self.shells {
                        break;
                    }
                    previous = distance;
                    shells += 1;
                }
                neighbours.push(index);
            }
            self.neighbours.insert(key, neighbours);
        }
        &self.neighbours[&key]
    }

    #[expect(
        clippy::suboptimal_flops,
        reason = "Preserve scalar neighbour distance association."
    )]
    fn fit(
        &mut self,
        levels: &mut [u8],
        values: &[f32],
        weights: &[f32],
        scale: f32,
    ) -> Result<bool> {
        let key = Self::key(levels, self.bits);
        if self.map[key] >= 0 {
            return Ok(true);
        }
        let points = self.points;
        let candidates = self.candidates(key, levels);
        let mut best_distance = f32::MAX;
        let mut best_index = None;
        for &candidate in candidates {
            let mut distance = 0.0_f32;
            for ((&q, &value), &weight) in points[candidate].iter().zip(values).zip(weights) {
                let diff = scale * f32::from(q) - value;
                distance += weight * diff * diff;
            }
            if distance < best_distance {
                best_distance = distance;
                best_index = Some(candidate);
            }
        }
        let idx = best_index.ok_or_else(|| invalid("IQ neighbour search empty"))?;
        for (level, &point) in levels.iter_mut().zip(&points[idx]) {
            *level = (point - 1) / 2;
        }
        Ok(false)
    }
}

fn iq_levels(values: &[f32], inverse: f32, maximum: u8, output: &mut [u8]) -> Result<()> {
    for (level, &value) in output.iter_mut().zip(values) {
        *level = u8::try_from(
            nearest_int(0.5 * inverse.mul_add(value, -1.0))?.clamp(0, i32::from(maximum)),
        )?;
    }
    Ok(())
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Preserve scalar IQ grid sum association."
)]
fn iq_sums(levels: &[u8], values: &[f32], weights: &[f32]) -> Result<(f32, f32)> {
    let mut dot_qx = 0.0_f32;
    let mut norm_q2 = 0.0_f32;
    for ((&level, &value), &weight) in levels.iter().zip(values).zip(weights) {
        let q = f32::from(2 * level + 1);
        dot_qx += weight * value * q;
        norm_q2 += weight * q * q;
    }
    Ok((finite(dot_qx)?, finite(norm_q2)?))
}

struct IqConfig {
    type_id: usize,
    bytes: usize,
    width: usize,
    maximum: u8,
    trials: i32,
    delta: f32,
    epsilon: f32,
    factor: f32,
}

impl IqConfig {
    fn new(type_id: usize) -> Result<Self> {
        match type_id {
            18 => Ok(Self {
                type_id,
                bytes: 98,
                width: 32,
                maximum: 7,
                trials: 15,
                delta: 0.2,
                epsilon: 1.0e-8,
                factor: 1.0125,
            }),
            21 => Ok(Self {
                type_id,
                bytes: 110,
                width: 32,
                maximum: 7,
                trials: 9,
                delta: 0.2,
                epsilon: 0.0,
                factor: 1.033,
            }),
            22 => Ok(Self {
                type_id,
                bytes: 82,
                width: 16,
                maximum: 2,
                trials: 9,
                delta: 0.1,
                epsilon: 1.0e-8,
                factor: 0.9875,
            }),
            _ => Err(invalid("unsupported IQ fitting type")),
        }
    }

    fn pack<const N: usize>(
        &self,
        grid: &IqGrid<'_, N>,
        group: usize,
        written_group: usize,
        levels: &[u8],
        signs: &[u8],
        output: &mut [u8],
    ) -> Result<()> {
        for (local, chunk) in levels.as_chunks::<N>().0.iter().enumerate() {
            let index = grid.index(chunk)?;
            let absolute = group * (self.width / N) + local;
            let written = written_group * (self.width / N) + local;
            output[2 + written] = index.to_le_bytes()[0];
            if self.type_id == 21 {
                output[66 + absolute / 8] |= u8::try_from(index >> 8)? << (absolute % 8);
            } else if self.type_id == 22 {
                output[66 + absolute / 4] |= u8::try_from(index >> 8)? << (2 * (absolute % 4));
            }
        }
        if self.type_id == 18 {
            let word = u32::from(signs[0])
                | (u32::from(signs[1]) << 7)
                | (u32::from(signs[2]) << 14)
                | (u32::from(signs[3]) << 21);
            output[66 + group * 4..70 + group * 4].copy_from_slice(&word.to_le_bytes());
        } else {
            let offset =
                if self.type_id == 21 { 74 } else { 34 } + written_group * (self.width / 8);
            output[offset..offset + signs.len()].copy_from_slice(signs);
        }
        Ok(())
    }
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Preserve native IQ grid fitting and exact zero scale branch."
)]
fn iq_fit_group<const N: usize>(
    cfg: &IqConfig,
    grid: &mut IqGrid<'_, N>,
    values: &[f32],
    weight: &[f32],
    aux_weight: &[f32],
    signs: &mut [u8],
    maximum: f32,
) -> Result<(f32, [u8; 32])> {
    let mut best = 0.0_f32;
    let mut scale = maximum / f32::from(2 * cfg.maximum + 1);
    let mut chosen = [0_u8; 32];
    let mut on_grid = [cfg.type_id != 21; 8];
    for trial in -cfg.trials..=cfg.trials {
        let inverse = (f32::from(2 * cfg.maximum + 1) + i32_to_f32(trial) * cfg.delta) / maximum;
        let cand_scale = finite(1.0 / inverse)?;
        let mut candidate = [0_u8; 32];
        let mut cand_on_grid = [true; 8];
        iq_levels(values, inverse, cfg.maximum, &mut candidate[..cfg.width])?;
        for (idx, chunk) in candidate[..cfg.width]
            .as_chunks_mut::<N>()
            .0
            .iter_mut()
            .enumerate()
        {
            cand_on_grid[idx] = grid.fit(
                chunk,
                &values[idx * N..(idx + 1) * N],
                &aux_weight[idx * N..(idx + 1) * N],
                cand_scale,
            )?;
        }
        let (dot_qx, norm_q2) = iq_sums(&candidate[..cfg.width], values, weight)?;
        if norm_q2 > 0.0 && finite(dot_qx * dot_qx)? > finite(best * norm_q2)? {
            scale = finite(dot_qx / norm_q2)?;
            best = finite(scale * dot_qx)?;
            chosen = candidate;
            on_grid = cand_on_grid;
        }
    }
    if on_grid[..cfg.width / N].iter().any(|&ok| !ok) && scale > 0.0 {
        let inverse = finite(1.0 / scale)?;
        for idx in 0..cfg.width / N {
            if cfg.type_id != 21 && on_grid[idx] {
                continue;
            }
            let selected = &mut chosen[idx * N..(idx + 1) * N];
            iq_levels(
                &values[idx * N..(idx + 1) * N],
                inverse,
                cfg.maximum,
                selected,
            )?;
            grid.fit(
                selected,
                &values[idx * N..(idx + 1) * N],
                &aux_weight[idx * N..(idx + 1) * N],
                scale,
            )?;
        }
        let (dot_qx, norm_q2) = iq_sums(&chosen[..cfg.width], values, weight)?;
        if norm_q2 > 0.0 {
            scale = finite(dot_qx / norm_q2)?;
        }
    }
    if scale < 0.0 {
        scale = -scale;
        for sign in signs {
            *sign = !*sign;
            if cfg.type_id == 18 {
                *sign &= 127;
            }
        }
    }
    Ok((scale, chosen))
}

struct PreparedIqGroup {
    values: [f32; 32],
    weight: [f32; 32],
    aux_weight: [f32; 32],
    signs: [u8; 4],
    maximum: f32,
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Preserve native IQ group preparation arithmetic."
)]
fn iq_prepare_group(
    type_id: usize,
    sigma2: f32,
    epsilon: f32,
    source: &[f32],
    width: usize,
) -> Result<Option<PreparedIqGroup>> {
    let mut values = [0.0_f32; 32];
    let mut weight = [0.0_f32; 32];
    let mut aux_weight = [0.0_f32; 32];
    let mut signs = [0_u8; 4];
    for i in 0..width {
        let sq = source[i] * source[i];
        weight[i] = finite(if type_id == 22 {
            0.25 * sigma2 + sq
        } else {
            sq
        })?;
        aux_weight[i] = weight[i].sqrt();
    }
    for (octet, oct_src) in source.as_chunks::<8>().0.iter().enumerate() {
        for (lane, &val) in oct_src.iter().enumerate() {
            if val >= 0.0 {
                values[octet * 8 + lane] = val;
            } else {
                values[octet * 8 + lane] = -val;
                signs[octet] |= 1 << lane;
            }
        }
        if type_id == 18 && !signs[octet].count_ones().is_multiple_of(2) {
            let mut min_idx = 0_usize;
            let off = octet * 8;
            let mut min_val = weight[off] * oct_src[0] * oct_src[0];
            for i in 1..8 {
                let cand = weight[off + i] * oct_src[i] * oct_src[i];
                if cand < min_val {
                    min_val = cand;
                    min_idx = i;
                }
            }
            values[off + min_idx] = -values[off + min_idx];
            signs[octet] ^= 1 << min_idx;
        }
        if type_id == 18 {
            signs[octet] &= 127;
        }
    }
    let maximum = values[..width].iter().copied().fold(values[0], f32::max);
    if maximum == 0.0 || maximum < epsilon {
        return Ok(None);
    }
    Ok(Some(PreparedIqGroup {
        values,
        weight,
        aux_weight,
        signs,
        maximum,
    }))
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Preserve native IQ scale encoding arithmetic."
)]
fn quantize_iq_grid<const N: usize>(
    type_id: usize,
    input: &[f32],
    points: &[[u8; N]],
) -> Result<Vec<u8>> {
    let cfg = IqConfig::new(type_id)?;
    let mut grid = IqGrid::new(
        points,
        if type_id == 22 { 2 } else { 3 },
        match type_id {
            18 => 2,
            21 => 3,
            _ => 1,
        },
    )?;
    let mut output = vec![0_u8; (input.len() / 256) * cfg.bytes];
    for (in_blk, out_blk) in input
        .as_chunks::<256>()
        .0
        .iter()
        .zip(output.chunks_exact_mut(cfg.bytes))
    {
        let sum_x2: f32 = in_blk.iter().map(|&v| v * v).sum();
        let sigma2 = finite(2.0 * sum_x2 / 256.0)?;
        let mut scales = [0.0_f32; 16];
        let mut max_scale = 0.0_f32;
        let mut written_groups = 0_usize;
        for (group, source) in in_blk.chunks_exact(cfg.width).enumerate() {
            let Some(mut prepared) =
                iq_prepare_group(type_id, sigma2, cfg.epsilon, source, cfg.width)?
            else {
                continue;
            };
            let (scale, chosen) = iq_fit_group(
                &cfg,
                &mut grid,
                &prepared.values[..cfg.width],
                &prepared.weight[..cfg.width],
                &prepared.aux_weight[..cfg.width],
                &mut prepared.signs[..cfg.width / 8],
                prepared.maximum,
            )?;
            cfg.pack(
                &grid,
                group,
                if type_id == 21 { written_groups } else { group },
                &chosen[..cfg.width],
                &prepared.signs[..cfg.width / 8],
                out_blk,
            )?;
            written_groups += 1;
            scales[group] = scale;
            max_scale = max_scale.max(scale);
        }
        if max_scale == 0.0 {
            continue;
        }
        let d = max_scale / 31.0;
        out_blk[..2].copy_from_slice(&half_bytes(finite(d * cfg.factor)?)?);
        let inverse = finite(1.0 / d)?;
        for (group, &scale) in scales[..256 / cfg.width].iter().enumerate() {
            let level = u8::try_from(nearest_int(0.5 * (inverse * scale - 1.0))?.clamp(0, 15))?;
            if type_id == 18 {
                out_blk[69 + group * 4] |= level << 4;
            } else {
                let offset = if type_id == 21 { 106 } else { 74 };
                out_blk[offset + group / 2] |= level << (4 * (group % 2));
            }
        }
    }
    Ok(output)
}

fn quantize_iq_fit(type_id: usize, input: &[f32]) -> Result<Vec<u8>> {
    match type_id {
        18 => {
            let points = IQ3XXS_GRID.map(|w| w.to_le_bytes().map(|v| 2 * (v / 8) + 1));
            quantize_iq_grid(18, input, &points)
        }
        21 => {
            let points = IQ3S_GRID.map(u32::to_le_bytes);
            quantize_iq_grid(21, input, &points)
        }
        22 => {
            let points = IQ2S_GRID.map(|w| w.to_le_bytes().map(|v| 2 * (v / 16) + 1));
            quantize_iq_grid(22, input, &points)
        }
        _ => Err(invalid("unsupported IQ fit type")),
    }
}

// ---------------------------------------------------------------------------
// Pinned IQ1 / IQ2 / IQ3 codebooks (MIT, ggml-common.h)
// ---------------------------------------------------------------------------

static IQ1_GRID: [u64; 2048] = [
    0xffff_ffff_ffff_ffff,
    0xffff_ffff_ffff_ff01,
    0xffff_ffff_ffff_0000,
    0xffff_ffff_ffff_01ff,
    0xffff_ffff_ffff_0101,
    0xffff_ffff_ff00_ff00,
    0xffff_ffff_ff00_0000,
    0xffff_ffff_ff01_ffff,
    0xffff_ffff_ff01_ff01,
    0xffff_ffff_ff01_01ff,
    0xffff_ffff_ff01_0101,
    0xffff_ffff_00ff_0000,
    0xffff_ffff_0000_ff00,
    0xffff_ffff_0000_00ff,
    0xffff_ffff_0000_0001,
    0xffff_ffff_0001_0000,
    0xffff_ffff_01ff_ffff,
    0xffff_ffff_01ff_ff01,
    0xffff_ffff_01ff_01ff,
    0xffff_ffff_01ff_0101,
    0xffff_ffff_0100_0000,
    0xffff_ffff_0101_ffff,
    0xffff_ffff_0101_ff01,
    0xffff_ffff_0101_01ff,
    0xffff_ffff_0101_0101,
    0xffff_ff00_ffff_00ff,
    0xffff_ff00_ffff_0000,
    0xffff_ff00_ff00_ff00,
    0xffff_ff00_ff00_00ff,
    0xffff_ff00_ff00_0001,
    0xffff_ff00_ff00_0100,
    0xffff_ff00_ff00_0101,
    0xffff_ff00_ff01_0000,
    0xffff_ff00_00ff_ff00,
    0xffff_ff00_00ff_0001,
    0xffff_ff00_00ff_0100,
    0xffff_ff00_0000_ff01,
    0xffff_ff00_0000_0000,
    0xffff_ff00_0000_0101,
    0xffff_ff00_0001_ff00,
    0xffff_ff00_0001_00ff,
    0xffff_ff00_0001_0001,
    0xffff_ff00_0001_01ff,
    0xffff_ff00_01ff_0000,
    0xffff_ff00_0100_ff00,
    0xffff_ff00_0100_00ff,
    0xffff_ff00_0100_0001,
    0xffff_ff00_0101_0000,
    0xffff_ff01_ffff_ffff,
    0xffff_ff01_ffff_ff01,
    0xffff_ff01_ffff_01ff,
    0xffff_ff01_ffff_0101,
    0xffff_ff01_ff00_0000,
    0xffff_ff01_ff01_ffff,
    0xffff_ff01_ff01_ff01,
    0xffff_ff01_ff01_01ff,
    0xffff_ff01_ff01_0101,
    0xffff_ff01_00ff_0000,
    0xffff_ff01_0000_ff00,
    0xffff_ff01_0000_0100,
    0xffff_ff01_0001_00ff,
    0xffff_ff01_0001_0100,
    0xffff_ff01_01ff_ffff,
    0xffff_ff01_01ff_ff01,
    0xffff_ff01_01ff_01ff,
    0xffff_ff01_01ff_0101,
    0xffff_ff01_0100_ff00,
    0xffff_ff01_0100_0000,
    0xffff_ff01_0100_0100,
    0xffff_ff01_0101_ffff,
    0xffff_ff01_0101_ff01,
    0xffff_ff01_0101_01ff,
    0xffff_ff01_0101_0101,
    0xffff_00ff_ff00_ff00,
    0xffff_00ff_ff00_00ff,
    0xffff_00ff_ff00_0001,
    0xffff_00ff_ff01_0000,
    0xffff_00ff_00ff_ff00,
    0xffff_00ff_00ff_0100,
    0xffff_00ff_0000_0000,
    0xffff_00ff_0000_0101,
    0xffff_00ff_0001_00ff,
    0xffff_00ff_0001_0000,
    0xffff_00ff_0100_ff00,
    0xffff_00ff_0100_0100,
    0xffff_00ff_0101_0000,
    0xffff_0000_ffff_ff00,
    0xffff_0000_ffff_00ff,
    0xffff_0000_ffff_0000,
    0xffff_0000_ffff_0001,
    0xffff_0000_ff00_0000,
    0xffff_0000_ff00_01ff,
    0xffff_0000_ff00_0101,
    0xffff_0000_ff01_0100,
    0xffff_0000_00ff_ffff,
    0xffff_0000_00ff_0000,
    0xffff_0000_00ff_0101,
    0xffff_0000_0000_ffff,
    0xffff_0000_0000_ff00,
    0xffff_0000_0000_00ff,
    0xffff_0000_0000_0000,
    0xffff_0000_0000_0001,
    0xffff_0000_0000_0100,
    0xffff_0000_0001_ffff,
    0xffff_0000_0001_ff01,
    0xffff_0000_0001_0000,
    0xffff_0000_0001_01ff,
    0xffff_0000_0001_0101,
    0xffff_0000_01ff_ff00,
    0xffff_0000_0100_ff00,
    0xffff_0000_0100_0000,
    0xffff_0000_0100_01ff,
    0xffff_0000_0100_0101,
    0xffff_0000_0101_ff00,
    0xffff_0000_0101_00ff,
    0xffff_0000_0101_0000,
    0xffff_0000_0101_0001,
    0xffff_0000_0101_0100,
    0xffff_0001_ff00_00ff,
    0xffff_0001_ff00_0100,
    0xffff_0001_00ff_ff00,
    0xffff_0001_00ff_00ff,
    0xffff_0001_0000_ffff,
    0xffff_0001_0000_ff01,
    0xffff_0001_0000_0000,
    0xffff_0001_0000_01ff,
    0xffff_0001_0001_ffff,
    0xffff_0001_0001_ff00,
    0xffff_0001_0001_0001,
    0xffff_0001_0001_0100,
    0xffff_0001_01ff_0000,
    0xffff_0001_0100_ff00,
    0xffff_0001_0100_00ff,
    0xffff_0001_0100_0100,
    0xffff_01ff_ffff_ffff,
    0xffff_01ff_ffff_ff01,
    0xffff_01ff_ffff_01ff,
    0xffff_01ff_ffff_0101,
    0xffff_01ff_ff00_0000,
    0xffff_01ff_ff01_ffff,
    0xffff_01ff_ff01_ff01,
    0xffff_01ff_ff01_01ff,
    0xffff_01ff_ff01_0101,
    0xffff_01ff_00ff_0000,
    0xffff_01ff_0000_ff00,
    0xffff_01ff_0000_0001,
    0xffff_01ff_0001_0000,
    0xffff_01ff_01ff_ffff,
    0xffff_01ff_01ff_ff01,
    0xffff_01ff_01ff_01ff,
    0xffff_01ff_01ff_0101,
    0xffff_01ff_0100_0000,
    0xffff_01ff_0101_ffff,
    0xffff_01ff_0101_ff01,
    0xffff_01ff_0101_01ff,
    0xffff_01ff_0101_0101,
    0xffff_0100_ffff_0000,
    0xffff_0100_ff00_ff00,
    0xffff_0100_ff00_00ff,
    0xffff_0100_ff00_0100,
    0xffff_0100_ff01_00ff,
    0xffff_0100_ff01_0000,
    0xffff_0100_00ff_ff00,
    0xffff_0100_0000_ffff,
    0xffff_0100_0000_ff00,
    0xffff_0100_0000_0000,
    0xffff_0100_0001_ff00,
    0xffff_0100_0001_00ff,
    0xffff_0100_0001_0100,
    0xffff_0100_0100_ff00,
    0xffff_0100_0100_00ff,
    0xffff_0100_0100_0001,
    0xffff_0100_0100_0100,
    0xffff_0100_0101_0000,
    0xffff_0101_ffff_ffff,
    0xffff_0101_ffff_ff01,
    0xffff_0101_ffff_01ff,
    0xffff_0101_ffff_0101,
    0xffff_0101_ff00_0000,
    0xffff_0101_ff01_ffff,
    0xffff_0101_ff01_ff01,
    0xffff_0101_ff01_01ff,
    0xffff_0101_ff01_0101,
    0xffff_0101_00ff_0000,
    0xffff_0101_0000_ff00,
    0xffff_0101_0000_0100,
    0xffff_0101_0001_ff00,
    0xffff_0101_0001_0000,
    0xffff_0101_01ff_ffff,
    0xffff_0101_01ff_ff01,
    0xffff_0101_01ff_0000,
    0xffff_0101_01ff_01ff,
    0xffff_0101_01ff_0101,
    0xffff_0101_0100_0000,
    0xffff_0101_0101_ffff,
    0xffff_0101_0101_ff01,
    0xffff_0101_0101_01ff,
    0xffff_0101_0101_0101,
    0xff00_ffff_ff00_ffff,
    0xff00_ffff_ff00_ff00,
    0xff00_ffff_ff00_00ff,
    0xff00_ffff_ff00_0100,
    0xff00_ffff_ff01_00ff,
    0xff00_ffff_ff01_0000,
    0xff00_ffff_00ff_ff00,
    0xff00_ffff_00ff_00ff,
    0xff00_ffff_0000_ffff,
    0xff00_ffff_0000_0000,
    0xff00_ffff_0000_01ff,
    0xff00_ffff_0001_ff00,
    0xff00_ffff_0001_00ff,
    0xff00_ffff_0001_0000,
    0xff00_ffff_0001_0100,
    0xff00_ffff_0100_ff00,
    0xff00_ffff_0100_00ff,
    0xff00_ffff_0100_0001,
    0xff00_ffff_0101_ff00,
    0xff00_ffff_0101_0000,
    0xff00_ff00_ffff_ff00,
    0xff00_ff00_ffff_00ff,
    0xff00_ff00_ffff_0001,
    0xff00_ff00_ffff_0100,
    0xff00_ff00_ff00_ffff,
    0xff00_ff00_ff00_ff01,
    0xff00_ff00_ff00_0000,
    0xff00_ff00_ff00_01ff,
    0xff00_ff00_ff01_ff00,
    0xff00_ff00_ff01_00ff,
    0xff00_ff00_ff01_0100,
    0xff00_ff00_00ff_0000,
    0xff00_ff00_00ff_0101,
    0xff00_ff00_0000_ffff,
    0xff00_ff00_0000_ff00,
    0xff00_ff00_0000_ff01,
    0xff00_ff00_0000_00ff,
    0xff00_ff00_0000_0000,
    0xff00_ff00_0000_0001,
    0xff00_ff00_0000_0100,
    0xff00_ff00_0001_ffff,
    0xff00_ff00_0001_0000,
    0xff00_ff00_01ff_00ff,
    0xff00_ff00_0100_ff01,
    0xff00_ff00_0100_0000,
    0xff00_ff00_0101_ff00,
    0xff00_ff00_0101_00ff,
    0xff00_ff01_ff00_ff00,
    0xff00_ff01_ff00_00ff,
    0xff00_ff01_ff00_0001,
    0xff00_ff01_ff01_0000,
    0xff00_ff01_00ff_ffff,
    0xff00_ff01_00ff_0001,
    0xff00_ff01_00ff_0100,
    0xff00_ff01_0000_ff01,
    0xff00_ff01_0000_0000,
    0xff00_ff01_0000_01ff,
    0xff00_ff01_0000_0101,
    0xff00_ff01_0001_00ff,
    0xff00_ff01_0001_0001,
    0xff00_ff01_01ff_0000,
    0xff00_ff01_0100_ff00,
    0xff00_ff01_0100_00ff,
    0xff00_ff01_0100_0001,
    0xff00_ff01_0101_0000,
    0xff00_00ff_ffff_ff00,
    0xff00_00ff_ffff_0001,
    0xff00_00ff_ffff_0100,
    0xff00_00ff_ff00_00ff,
    0xff00_00ff_ff00_0000,
    0xff00_00ff_ff00_01ff,
    0xff00_00ff_ff00_0100,
    0xff00_00ff_ff01_ff00,
    0xff00_00ff_ff01_0001,
    0xff00_00ff_00ff_ff00,
    0xff00_00ff_00ff_0000,
    0xff00_00ff_00ff_0001,
    0xff00_00ff_00ff_01ff,
    0xff00_00ff_00ff_0101,
    0xff00_00ff_0000_ff00,
    0xff00_00ff_0000_00ff,
    0xff00_00ff_0000_0000,
    0xff00_00ff_0000_0001,
    0xff00_00ff_0000_0100,
    0xff00_00ff_0001_ff01,
    0xff00_00ff_0001_0000,
    0xff00_00ff_0001_01ff,
    0xff00_00ff_01ff_00ff,
    0xff00_00ff_01ff_0100,
    0xff00_00ff_0100_ffff,
    0xff00_00ff_0100_00ff,
    0xff00_00ff_0100_0000,
    0xff00_00ff_0100_01ff,
    0xff00_00ff_0100_0100,
    0xff00_00ff_0100_0101,
    0xff00_00ff_0101_ff00,
    0xff00_00ff_0101_00ff,
    0xff00_00ff_0101_0000,
    0xff00_00ff_0101_0100,
    0xff00_0000_ffff_ff01,
    0xff00_0000_ffff_0000,
    0xff00_0000_ffff_0101,
    0xff00_0000_ff00_ff00,
    0xff00_0000_ff00_00ff,
    0xff00_0000_ff00_0000,
    0xff00_0000_ff00_0001,
    0xff00_0000_ff00_0100,
    0xff00_0000_ff01_ffff,
    0xff00_0000_ff01_ff01,
    0xff00_0000_ff01_0000,
    0xff00_0000_ff01_01ff,
    0xff00_0000_ff01_0101,
    0xff00_0000_00ff_ff00,
    0xff00_0000_00ff_00ff,
    0xff00_0000_00ff_0000,
    0xff00_0000_00ff_0001,
    0xff00_0000_0000_ff00,
    0xff00_0000_0000_ff01,
    0xff00_0000_0000_00ff,
    0xff00_0000_0000_0000,
    0xff00_0000_0000_0001,
    0xff00_0000_0000_0100,
    0xff00_0000_0000_0101,
    0xff00_0000_0001_ff00,
    0xff00_0000_0001_00ff,
    0xff00_0000_0001_0000,
    0xff00_0000_0001_0001,
    0xff00_0000_0001_0100,
    0xff00_0000_01ff_ffff,
    0xff00_0000_01ff_ff01,
    0xff00_0000_01ff_00ff,
    0xff00_0000_01ff_0000,
    0xff00_0000_01ff_01ff,
    0xff00_0000_01ff_0101,
    0xff00_0000_0100_ffff,
    0xff00_0000_0100_ff00,
    0xff00_0000_0100_00ff,
    0xff00_0000_0100_0000,
    0xff00_0000_0100_0001,
    0xff00_0000_0100_0100,
    0xff00_0000_0100_0101,
    0xff00_0000_0101_ffff,
    0xff00_0000_0101_ff01,
    0xff00_0000_0101_0000,
    0xff00_0001_ffff_ff00,
    0xff00_0001_ffff_00ff,
    0xff00_0001_ffff_0000,
    0xff00_0001_ffff_0001,
    0xff00_0001_ff00_0000,
    0xff00_0001_ff00_0001,
    0xff00_0001_ff00_01ff,
    0xff00_0001_ff00_0101,
    0xff00_0001_ff01_ff00,
    0xff00_0001_ff01_0001,
    0xff00_0001_00ff_ffff,
    0xff00_0001_00ff_ff01,
    0xff00_0001_00ff_00ff,
    0xff00_0001_00ff_0000,
    0xff00_0001_00ff_01ff,
    0xff00_0001_00ff_0101,
    0xff00_0001_0000_ff00,
    0xff00_0001_0000_0000,
    0xff00_0001_0000_0001,
    0xff00_0001_0000_01ff,
    0xff00_0001_0000_0100,
    0xff00_0001_0001_ff00,
    0xff00_0001_0001_00ff,
    0xff00_0001_0001_0000,
    0xff00_0001_0001_01ff,
    0xff00_0001_0001_0100,
    0xff00_0001_0001_0101,
    0xff00_0001_01ff_0001,
    0xff00_0001_01ff_0101,
    0xff00_0001_0100_ff01,
    0xff00_0001_0100_0000,
    0xff00_0001_0101_00ff,
    0xff00_0001_0101_0100,
    0xff00_01ff_ff00_ff00,
    0xff00_01ff_ff00_0001,
    0xff00_01ff_ff01_0000,
    0xff00_01ff_00ff_ff00,
    0xff00_01ff_00ff_00ff,
    0xff00_01ff_00ff_0001,
    0xff00_01ff_00ff_0100,
    0xff00_01ff_0000_ffff,
    0xff00_01ff_0000_0000,
    0xff00_01ff_0000_01ff,
    0xff00_01ff_0000_0101,
    0xff00_01ff_0001_ffff,
    0xff00_01ff_0001_ff00,
    0xff00_01ff_0001_00ff,
    0xff00_01ff_0001_0001,
    0xff00_01ff_0001_0100,
    0xff00_01ff_01ff_0000,
    0xff00_01ff_0100_ff00,
    0xff00_01ff_0100_00ff,
    0xff00_01ff_0101_0000,
    0xff00_0100_ff00_ffff,
    0xff00_0100_ff00_ff01,
    0xff00_0100_ff00_0000,
    0xff00_0100_ff00_0101,
    0xff00_0100_ff01_ff00,
    0xff00_0100_ff01_0000,
    0xff00_0100_00ff_ff01,
    0xff00_0100_00ff_00ff,
    0xff00_0100_00ff_0000,
    0xff00_0100_00ff_01ff,
    0xff00_0100_0000_ff00,
    0xff00_0100_0000_00ff,
    0xff00_0100_0000_0000,
    0xff00_0100_0000_0001,
    0xff00_0100_0000_0100,
    0xff00_0100_0000_0101,
    0xff00_0100_0001_ffff,
    0xff00_0100_0001_0000,
    0xff00_0100_0001_0101,
    0xff00_0100_01ff_0100,
    0xff00_0100_0100_ff00,
    0xff00_0100_0100_ff01,
    0xff00_0100_0100_0000,
    0xff00_0100_0100_01ff,
    0xff00_0100_0101_ff00,
    0xff00_0100_0101_0001,
    0xff00_0100_0101_0100,
    0xff00_0101_ffff_0100,
    0xff00_0101_ff00_0001,
    0xff00_0101_ff01_00ff,
    0xff00_0101_ff01_0001,
    0xff00_0101_00ff_00ff,
    0xff00_0101_00ff_0001,
    0xff00_0101_00ff_0100,
    0xff00_0101_0000_ffff,
    0xff00_0101_0000_ff01,
    0xff00_0101_0000_0000,
    0xff00_0101_0000_01ff,
    0xff00_0101_0001_ff00,
    0xff00_0101_0001_0001,
    0xff00_0101_0001_0100,
    0xff00_0101_01ff_0000,
    0xff00_0101_0100_ff00,
    0xff00_0101_0100_0001,
    0xff00_0101_0100_0101,
    0xff01_ffff_ffff_ffff,
    0xff01_ffff_ffff_ff01,
    0xff01_ffff_ffff_01ff,
    0xff01_ffff_ffff_0101,
    0xff01_ffff_ff00_0000,
    0xff01_ffff_ff01_ffff,
    0xff01_ffff_ff01_ff01,
    0xff01_ffff_ff01_0000,
    0xff01_ffff_ff01_01ff,
    0xff01_ffff_ff01_0101,
    0xff01_ffff_00ff_0000,
    0xff01_ffff_0000_ff00,
    0xff01_ffff_0000_0100,
    0xff01_ffff_0001_ff00,
    0xff01_ffff_0001_0000,
    0xff01_ffff_01ff_ffff,
    0xff01_ffff_01ff_ff01,
    0xff01_ffff_01ff_01ff,
    0xff01_ffff_01ff_0101,
    0xff01_ffff_0100_0000,
    0xff01_ffff_0101_ffff,
    0xff01_ffff_0101_ff01,
    0xff01_ffff_0101_0000,
    0xff01_ffff_0101_01ff,
    0xff01_ffff_0101_0101,
    0xff01_ff00_ffff_0000,
    0xff01_ff00_ff00_ff00,
    0xff01_ff00_ff00_00ff,
    0xff01_ff00_ff00_0100,
    0xff01_ff00_ff01_0000,
    0xff01_ff00_00ff_ff01,
    0xff01_ff00_00ff_00ff,
    0xff01_ff00_00ff_0100,
    0xff01_ff00_0000_0000,
    0xff01_ff00_0000_01ff,
    0xff01_ff00_0000_0101,
    0xff01_ff00_0001_ff00,
    0xff01_ff00_0001_00ff,
    0xff01_ff00_0001_0000,
    0xff01_ff00_0001_0001,
    0xff01_ff00_01ff_0000,
    0xff01_ff00_0100_ffff,
    0xff01_ff00_0100_0001,
    0xff01_ff00_0100_0100,
    0xff01_ff00_0101_0000,
    0xff01_ff01_ffff_ff00,
    0xff01_ff01_ffff_01ff,
    0xff01_ff01_ffff_0101,
    0xff01_ff01_ff00_ff00,
    0xff01_ff01_ff00_0000,
    0xff01_ff01_ff01_ffff,
    0xff01_ff01_ff01_ff01,
    0xff01_ff01_ff01_01ff,
    0xff01_ff01_ff01_0101,
    0xff01_ff01_00ff_0000,
    0xff01_ff01_0000_ff00,
    0xff01_ff01_0000_0001,
    0xff01_ff01_0000_0100,
    0xff01_ff01_0001_0000,
    0xff01_ff01_01ff_ff00,
    0xff01_ff01_01ff_01ff,
    0xff01_ff01_01ff_0101,
    0xff01_ff01_0100_ff00,
    0xff01_ff01_0100_0000,
    0xff01_ff01_0101_ffff,
    0xff01_ff01_0101_ff01,
    0xff01_ff01_0101_01ff,
    0xff01_ff01_0101_0101,
    0xff01_00ff_ffff_0000,
    0xff01_00ff_ff00_00ff,
    0xff01_00ff_ff00_0001,
    0xff01_00ff_ff00_0100,
    0xff01_00ff_ff01_0000,
    0xff01_00ff_00ff_00ff,
    0xff01_00ff_00ff_0000,
    0xff01_00ff_00ff_0001,
    0xff01_00ff_00ff_0100,
    0xff01_00ff_0000_ff01,
    0xff01_00ff_0000_0000,
    0xff01_00ff_0000_01ff,
    0xff01_00ff_0000_0101,
    0xff01_00ff_0001_0001,
    0xff01_00ff_01ff_0000,
    0xff01_00ff_0100_ff00,
    0xff01_00ff_0100_00ff,
    0xff01_00ff_0100_0100,
    0xff01_00ff_0101_ff00,
    0xff01_00ff_0101_0000,
    0xff01_0000_ffff_0100,
    0xff01_0000_ff00_0000,
    0xff01_0000_ff01_ff00,
    0xff01_0000_ff01_0100,
    0xff01_0000_00ff_ffff,
    0xff01_0000_00ff_0000,
    0xff01_0000_00ff_01ff,
    0xff01_0000_0000_ff00,
    0xff01_0000_0000_00ff,
    0xff01_0000_0000_0000,
    0xff01_0000_0000_0100,
    0xff01_0000_0001_ff01,
    0xff01_0000_0001_0000,
    0xff01_0000_0001_01ff,
    0xff01_0000_01ff_0100,
    0xff01_0000_0100_ffff,
    0xff01_0000_0100_00ff,
    0xff01_0000_0100_0000,
    0xff01_0000_0100_01ff,
    0xff01_0000_0100_0101,
    0xff01_0000_0101_ff00,
    0xff01_0000_0101_00ff,
    0xff01_0000_0101_0001,
    0xff01_0000_0101_0100,
    0xff01_0001_ffff_0000,
    0xff01_0001_ff00_ffff,
    0xff01_0001_ff00_ff01,
    0xff01_0001_ff00_0100,
    0xff01_0001_ff01_0000,
    0xff01_0001_00ff_ff00,
    0xff01_0001_00ff_0100,
    0xff01_0001_0000_0000,
    0xff01_0001_0001_ffff,
    0xff01_0001_0001_ff00,
    0xff01_0001_0001_0100,
    0xff01_0001_01ff_00ff,
    0xff01_0001_01ff_0001,
    0xff01_0001_0100_ffff,
    0xff01_0001_0100_0101,
    0xff01_01ff_ffff_ffff,
    0xff01_01ff_ffff_ff01,
    0xff01_01ff_ffff_01ff,
    0xff01_01ff_ffff_0101,
    0xff01_01ff_ff00_0000,
    0xff01_01ff_ff01_ffff,
    0xff01_01ff_ff01_ff01,
    0xff01_01ff_ff01_01ff,
    0xff01_01ff_ff01_0101,
    0xff01_01ff_00ff_0000,
    0xff01_01ff_0000_ff00,
    0xff01_01ff_0000_00ff,
    0xff01_01ff_0001_0000,
    0xff01_01ff_01ff_ffff,
    0xff01_01ff_01ff_ff01,
    0xff01_01ff_01ff_01ff,
    0xff01_01ff_01ff_0101,
    0xff01_01ff_0101_ffff,
    0xff01_01ff_0101_ff01,
    0xff01_01ff_0101_01ff,
    0xff01_01ff_0101_0101,
    0xff01_0100_ffff_0100,
    0xff01_0100_ff00_ff00,
    0xff01_0100_ff00_00ff,
    0xff01_0100_ff00_0100,
    0xff01_0100_ff01_0000,
    0xff01_0100_00ff_0001,
    0xff01_0100_00ff_0100,
    0xff01_0100_0000_ff01,
    0xff01_0100_0000_0000,
    0xff01_0100_0001_ff00,
    0xff01_0100_0001_00ff,
    0xff01_0100_0001_0001,
    0xff01_0100_0001_0100,
    0xff01_0100_01ff_0000,
    0xff01_0100_0100_ffff,
    0xff01_0100_0100_0001,
    0xff01_0100_0100_0100,
    0xff01_0100_0101_00ff,
    0xff01_0100_0101_0000,
    0xff01_0101_ffff_ffff,
    0xff01_0101_ffff_ff01,
    0xff01_0101_ffff_01ff,
    0xff01_0101_ffff_0101,
    0xff01_0101_ff01_ffff,
    0xff01_0101_ff01_ff01,
    0xff01_0101_ff01_01ff,
    0xff01_0101_ff01_0101,
    0xff01_0101_00ff_0000,
    0xff01_0101_0000_ff00,
    0xff01_0101_0000_0001,
    0xff01_0101_0000_0100,
    0xff01_0101_0001_0000,
    0xff01_0101_01ff_ffff,
    0xff01_0101_01ff_ff01,
    0xff01_0101_01ff_01ff,
    0xff01_0101_01ff_0101,
    0xff01_0101_0100_0000,
    0xff01_0101_0101_ffff,
    0xff01_0101_0101_ff01,
    0xff01_0101_0101_01ff,
    0xff01_0101_0101_0101,
    0x00ff_ffff_ffff_0000,
    0x00ff_ffff_ff00_ff00,
    0x00ff_ffff_ff00_0001,
    0x00ff_ffff_ff01_0000,
    0x00ff_ffff_00ff_0100,
    0x00ff_ffff_0000_ff01,
    0x00ff_ffff_0000_0000,
    0x00ff_ffff_0000_01ff,
    0x00ff_ffff_0000_0101,
    0x00ff_ffff_0001_ff00,
    0x00ff_ffff_0001_00ff,
    0x00ff_ffff_0001_0001,
    0x00ff_ffff_0100_00ff,
    0x00ff_ffff_0100_0100,
    0x00ff_ffff_0101_ff00,
    0x00ff_ffff_0101_0001,
    0x00ff_ff00_ffff_ffff,
    0x00ff_ff00_ffff_ff00,
    0x00ff_ff00_ffff_00ff,
    0x00ff_ff00_ffff_0001,
    0x00ff_ff00_ffff_0100,
    0x00ff_ff00_ff00_ff01,
    0x00ff_ff00_ff00_0000,
    0x00ff_ff00_ff00_0001,
    0x00ff_ff00_ff00_01ff,
    0x00ff_ff00_ff00_0101,
    0x00ff_ff00_ff01_ff00,
    0x00ff_ff00_ff01_0001,
    0x00ff_ff00_ff01_0100,
    0x00ff_ff00_00ff_0000,
    0x00ff_ff00_00ff_01ff,
    0x00ff_ff00_00ff_0101,
    0x00ff_ff00_0000_ff00,
    0x00ff_ff00_0000_00ff,
    0x00ff_ff00_0000_0000,
    0x00ff_ff00_0000_0001,
    0x00ff_ff00_0000_0100,
    0x00ff_ff00_0000_0101,
    0x00ff_ff00_0001_0000,
    0x00ff_ff00_0001_01ff,
    0x00ff_ff00_0001_0101,
    0x00ff_ff00_01ff_ff00,
    0x00ff_ff00_01ff_00ff,
    0x00ff_ff00_01ff_0001,
    0x00ff_ff00_0100_ffff,
    0x00ff_ff00_0100_ff01,
    0x00ff_ff00_0100_0000,
    0x00ff_ff00_0101_ffff,
    0x00ff_ff00_0101_ff00,
    0x00ff_ff00_0101_ff01,
    0x00ff_ff01_ffff_0000,
    0x00ff_ff01_ff00_ff00,
    0x00ff_ff01_ff00_00ff,
    0x00ff_ff01_ff00_0001,
    0x00ff_ff01_ff01_0000,
    0x00ff_ff01_00ff_ff00,
    0x00ff_ff01_0000_ff01,
    0x00ff_ff01_0000_0000,
    0x00ff_ff01_0000_0101,
    0x00ff_ff01_0001_00ff,
    0x00ff_ff01_0001_0100,
    0x00ff_ff01_01ff_0100,
    0x00ff_ff01_0100_00ff,
    0x00ff_ff01_0101_0000,
    0x00ff_00ff_ffff_ff00,
    0x00ff_00ff_ff00_0000,
    0x00ff_00ff_ff00_0100,
    0x00ff_00ff_ff01_0100,
    0x00ff_00ff_00ff_0000,
    0x00ff_00ff_00ff_01ff,
    0x00ff_00ff_00ff_0101,
    0x00ff_00ff_0000_ff00,
    0x00ff_00ff_0000_00ff,
    0x00ff_00ff_0000_0000,
    0x00ff_00ff_0000_0001,
    0x00ff_00ff_0001_ff00,
    0x00ff_00ff_0001_ff01,
    0x00ff_00ff_0001_0000,
    0x00ff_00ff_0001_01ff,
    0x00ff_00ff_0001_0101,
    0x00ff_00ff_01ff_ff00,
    0x00ff_00ff_01ff_0001,
    0x00ff_00ff_01ff_0100,
    0x00ff_00ff_0100_ffff,
    0x00ff_00ff_0100_ff01,
    0x00ff_00ff_0100_0000,
    0x00ff_00ff_0101_ffff,
    0x00ff_00ff_0101_ff00,
    0x00ff_00ff_0101_0100,
    0x00ff_0000_ffff_ff00,
    0x00ff_0000_ffff_ff01,
    0x00ff_0000_ffff_0000,
    0x00ff_0000_ffff_0101,
    0x00ff_0000_ff00_ff00,
    0x00ff_0000_ff00_00ff,
    0x00ff_0000_ff00_0000,
    0x00ff_0000_ff00_0001,
    0x00ff_0000_ff00_0100,
    0x00ff_0000_ff01_ffff,
    0x00ff_0000_ff01_0000,
    0x00ff_0000_ff01_0101,
    0x00ff_0000_00ff_ff00,
    0x00ff_0000_00ff_00ff,
    0x00ff_0000_00ff_0000,
    0x00ff_0000_00ff_0001,
    0x00ff_0000_00ff_0100,
    0x00ff_0000_0000_ffff,
    0x00ff_0000_0000_ff00,
    0x00ff_0000_0000_00ff,
    0x00ff_0000_0000_0000,
    0x00ff_0000_0000_0001,
    0x00ff_0000_0000_01ff,
    0x00ff_0000_0000_0100,
    0x00ff_0000_0001_ff00,
    0x00ff_0000_0001_00ff,
    0x00ff_0000_0001_0000,
    0x00ff_0000_0001_0001,
    0x00ff_0000_0001_0100,
    0x00ff_0000_01ff_ff01,
    0x00ff_0000_01ff_00ff,
    0x00ff_0000_01ff_0000,
    0x00ff_0000_01ff_01ff,
    0x00ff_0000_0100_ff00,
    0x00ff_0000_0100_00ff,
    0x00ff_0000_0100_0000,
    0x00ff_0000_0100_0001,
    0x00ff_0000_0100_0100,
    0x00ff_0000_0100_0101,
    0x00ff_0000_0101_0000,
    0x00ff_0000_0101_01ff,
    0x00ff_0000_0101_0101,
    0x00ff_0001_ffff_ff00,
    0x00ff_0001_ffff_0000,
    0x00ff_0001_ffff_0100,
    0x00ff_0001_ff00_00ff,
    0x00ff_0001_ff00_0000,
    0x00ff_0001_ff00_01ff,
    0x00ff_0001_ff00_0101,
    0x00ff_0001_ff01_ff00,
    0x00ff_0001_ff01_00ff,
    0x00ff_0001_ff01_0100,
    0x00ff_0001_00ff_ffff,
    0x00ff_0001_00ff_ff01,
    0x00ff_0001_00ff_0000,
    0x00ff_0001_00ff_01ff,
    0x00ff_0001_0000_ffff,
    0x00ff_0001_0000_ff00,
    0x00ff_0001_0000_ff01,
    0x00ff_0001_0000_0000,
    0x00ff_0001_0000_0001,
    0x00ff_0001_0000_0100,
    0x00ff_0001_0001_ff01,
    0x00ff_0001_0001_0000,
    0x00ff_0001_0001_01ff,
    0x00ff_0001_01ff_ff00,
    0x00ff_0001_01ff_0000,
    0x00ff_0001_01ff_0101,
    0x00ff_0001_0100_00ff,
    0x00ff_0001_0100_0000,
    0x00ff_0001_0101_ff00,
    0x00ff_0001_0101_00ff,
    0x00ff_0001_0101_0001,
    0x00ff_01ff_ffff_0000,
    0x00ff_01ff_ff00_ff00,
    0x00ff_01ff_ff00_0000,
    0x00ff_01ff_ff00_0101,
    0x00ff_01ff_ff01_0000,
    0x00ff_01ff_00ff_ff01,
    0x00ff_01ff_00ff_0100,
    0x00ff_01ff_0000_ffff,
    0x00ff_01ff_0000_0000,
    0x00ff_01ff_0000_01ff,
    0x00ff_01ff_0001_ff00,
    0x00ff_01ff_0001_00ff,
    0x00ff_01ff_0001_0001,
    0x00ff_01ff_0001_0100,
    0x00ff_01ff_01ff_0000,
    0x00ff_01ff_0100_ff00,
    0x00ff_01ff_0100_00ff,
    0x00ff_01ff_0100_0001,
    0x00ff_01ff_0100_0100,
    0x00ff_01ff_0101_0000,
    0x00ff_0100_ffff_ff00,
    0x00ff_0100_ffff_0000,
    0x00ff_0100_ffff_0001,
    0x00ff_0100_ffff_0101,
    0x00ff_0100_ff00_ffff,
    0x00ff_0100_ff00_00ff,
    0x00ff_0100_ff00_0000,
    0x00ff_0100_ff00_01ff,
    0x00ff_0100_ff01_ff00,
    0x00ff_0100_ff01_00ff,
    0x00ff_0100_ff01_0001,
    0x00ff_0100_00ff_ffff,
    0x00ff_0100_00ff_0000,
    0x00ff_0100_00ff_0101,
    0x00ff_0100_0000_ff00,
    0x00ff_0100_0000_ff01,
    0x00ff_0100_0000_00ff,
    0x00ff_0100_0000_0000,
    0x00ff_0100_0000_0001,
    0x00ff_0100_0000_0100,
    0x00ff_0100_0001_ffff,
    0x00ff_0100_0001_ff01,
    0x00ff_0100_0001_0000,
    0x00ff_0100_0001_0001,
    0x00ff_0100_0001_0101,
    0x00ff_0100_01ff_0001,
    0x00ff_0100_01ff_0100,
    0x00ff_0100_0100_ff01,
    0x00ff_0100_0100_0000,
    0x00ff_0100_0100_0001,
    0x00ff_0100_0100_01ff,
    0x00ff_0100_0101_ff00,
    0x00ff_0100_0101_00ff,
    0x00ff_0100_0101_0001,
    0x00ff_0100_0101_0100,
    0x00ff_0101_ff00_0001,
    0x00ff_0101_00ff_00ff,
    0x00ff_0101_00ff_0001,
    0x00ff_0101_00ff_0100,
    0x00ff_0101_0000_0000,
    0x00ff_0101_0000_01ff,
    0x00ff_0101_0000_0101,
    0x00ff_0101_0001_00ff,
    0x00ff_0101_0001_0100,
    0x00ff_0101_0100_00ff,
    0x00ff_0101_0101_0000,
    0x0000_ffff_ffff_ff00,
    0x0000_ffff_ffff_00ff,
    0x0000_ffff_ffff_0000,
    0x0000_ffff_ffff_0001,
    0x0000_ffff_ffff_0100,
    0x0000_ffff_ff00_ff01,
    0x0000_ffff_ff00_0000,
    0x0000_ffff_ff00_0101,
    0x0000_ffff_ff01_ff00,
    0x0000_ffff_ff01_00ff,
    0x0000_ffff_ff01_0100,
    0x0000_ffff_00ff_ffff,
    0x0000_ffff_00ff_0000,
    0x0000_ffff_00ff_01ff,
    0x0000_ffff_0000_ff00,
    0x0000_ffff_0000_00ff,
    0x0000_ffff_0000_0000,
    0x0000_ffff_0000_0001,
    0x0000_ffff_0000_0100,
    0x0000_ffff_0001_0000,
    0x0000_ffff_0001_01ff,
    0x0000_ffff_01ff_0001,
    0x0000_ffff_01ff_0100,
    0x0000_ffff_0100_0000,
    0x0000_ffff_0100_01ff,
    0x0000_ffff_0101_ffff,
    0x0000_ffff_0101_ff00,
    0x0000_ffff_0101_0001,
    0x0000_ffff_0101_0100,
    0x0000_ff00_ffff_0000,
    0x0000_ff00_ffff_01ff,
    0x0000_ff00_ffff_0100,
    0x0000_ff00_ffff_0101,
    0x0000_ff00_ff00_ff00,
    0x0000_ff00_ff00_00ff,
    0x0000_ff00_ff00_0000,
    0x0000_ff00_ff00_0001,
    0x0000_ff00_ff00_01ff,
    0x0000_ff00_ff00_0100,
    0x0000_ff00_ff01_ffff,
    0x0000_ff00_ff01_0000,
    0x0000_ff00_ff01_0001,
    0x0000_ff00_ff01_01ff,
    0x0000_ff00_ff01_0101,
    0x0000_ff00_00ff_ff00,
    0x0000_ff00_00ff_00ff,
    0x0000_ff00_00ff_0000,
    0x0000_ff00_00ff_0001,
    0x0000_ff00_00ff_0100,
    0x0000_ff00_0000_ffff,
    0x0000_ff00_0000_ff00,
    0x0000_ff00_0000_ff01,
    0x0000_ff00_0000_00ff,
    0x0000_ff00_0000_0000,
    0x0000_ff00_0000_0001,
    0x0000_ff00_0000_01ff,
    0x0000_ff00_0000_0100,
    0x0000_ff00_0000_0101,
    0x0000_ff00_0001_ff00,
    0x0000_ff00_0001_00ff,
    0x0000_ff00_0001_0000,
    0x0000_ff00_0001_0001,
    0x0000_ff00_0001_0100,
    0x0000_ff00_01ff_ff01,
    0x0000_ff00_01ff_0000,
    0x0000_ff00_0100_ff00,
    0x0000_ff00_0100_00ff,
    0x0000_ff00_0100_0000,
    0x0000_ff00_0100_0001,
    0x0000_ff00_0100_0100,
    0x0000_ff00_0101_ffff,
    0x0000_ff00_0101_0000,
    0x0000_ff00_0101_0101,
    0x0000_ff01_ffff_ff00,
    0x0000_ff01_ffff_0001,
    0x0000_ff01_ff00_ff01,
    0x0000_ff01_ff00_0000,
    0x0000_ff01_ff00_0101,
    0x0000_ff01_ff01_ff00,
    0x0000_ff01_ff01_00ff,
    0x0000_ff01_00ff_ff01,
    0x0000_ff01_00ff_0000,
    0x0000_ff01_00ff_0101,
    0x0000_ff01_0000_ff00,
    0x0000_ff01_0000_00ff,
    0x0000_ff01_0000_0000,
    0x0000_ff01_0000_0001,
    0x0000_ff01_0000_0100,
    0x0000_ff01_0001_ff01,
    0x0000_ff01_0001_0000,
    0x0000_ff01_01ff_0000,
    0x0000_ff01_0100_ffff,
    0x0000_ff01_0100_ff01,
    0x0000_ff01_0100_0000,
    0x0000_ff01_0100_0100,
    0x0000_ff01_0100_0101,
    0x0000_ff01_0101_00ff,
    0x0000_00ff_ffff_00ff,
    0x0000_00ff_ffff_0000,
    0x0000_00ff_ff00_ff00,
    0x0000_00ff_ff00_00ff,
    0x0000_00ff_ff00_0000,
    0x0000_00ff_ff00_0001,
    0x0000_00ff_ff00_01ff,
    0x0000_00ff_ff00_0100,
    0x0000_00ff_ff01_ff00,
    0x0000_00ff_ff01_0000,
    0x0000_00ff_ff01_01ff,
    0x0000_00ff_ff01_0101,
    0x0000_00ff_00ff_ff00,
    0x0000_00ff_00ff_00ff,
    0x0000_00ff_00ff_0000,
    0x0000_00ff_00ff_0001,
    0x0000_00ff_00ff_0100,
    0x0000_00ff_00ff_0101,
    0x0000_00ff_0000_ffff,
    0x0000_00ff_0000_ff00,
    0x0000_00ff_0000_00ff,
    0x0000_00ff_0000_0000,
    0x0000_00ff_0000_0001,
    0x0000_00ff_0000_01ff,
    0x0000_00ff_0000_0100,
    0x0000_00ff_0000_0101,
    0x0000_00ff_0001_ff00,
    0x0000_00ff_0001_ff01,
    0x0000_00ff_0001_00ff,
    0x0000_00ff_0001_0000,
    0x0000_00ff_0001_0001,
    0x0000_00ff_0001_0100,
    0x0000_00ff_01ff_ffff,
    0x0000_00ff_01ff_01ff,
    0x0000_00ff_01ff_0101,
    0x0000_00ff_0100_ff00,
    0x0000_00ff_0100_00ff,
    0x0000_00ff_0100_0000,
    0x0000_00ff_0100_0001,
    0x0000_00ff_0100_0100,
    0x0000_00ff_0101_ff00,
    0x0000_00ff_0101_00ff,
    0x0000_00ff_0101_0000,
    0x0000_00ff_0101_0101,
    0x0000_0000_ffff_ff00,
    0x0000_0000_ffff_ff01,
    0x0000_0000_ffff_00ff,
    0x0000_0000_ffff_0000,
    0x0000_0000_ffff_0001,
    0x0000_0000_ffff_0100,
    0x0000_0000_ff00_ffff,
    0x0000_0000_ff00_ff00,
    0x0000_0000_ff00_ff01,
    0x0000_0000_ff00_00ff,
    0x0000_0000_ff00_0000,
    0x0000_0000_ff00_0001,
    0x0000_0000_ff00_0100,
    0x0000_0000_ff00_0101,
    0x0000_0000_ff01_ff00,
    0x0000_0000_ff01_00ff,
    0x0000_0000_ff01_0000,
    0x0000_0000_ff01_0001,
    0x0000_0000_ff01_0100,
    0x0000_0000_00ff_ffff,
    0x0000_0000_00ff_ff00,
    0x0000_0000_00ff_ff01,
    0x0000_0000_00ff_00ff,
    0x0000_0000_00ff_0000,
    0x0000_0000_00ff_0001,
    0x0000_0000_00ff_01ff,
    0x0000_0000_00ff_0100,
    0x0000_0000_0000_ffff,
    0x0000_0000_0000_ff00,
    0x0000_0000_0000_ff01,
    0x0000_0000_0000_00ff,
    0x0000_0000_0000_0000,
    0x0000_0000_0000_0001,
    0x0000_0000_0000_01ff,
    0x0000_0000_0000_0100,
    0x0000_0000_0000_0101,
    0x0000_0000_0001_ffff,
    0x0000_0000_0001_ff00,
    0x0000_0000_0001_00ff,
    0x0000_0000_0001_0000,
    0x0000_0000_0001_0001,
    0x0000_0000_0001_01ff,
    0x0000_0000_0001_0100,
    0x0000_0000_0001_0101,
    0x0000_0000_01ff_ff00,
    0x0000_0000_01ff_00ff,
    0x0000_0000_01ff_0000,
    0x0000_0000_01ff_0100,
    0x0000_0000_01ff_0101,
    0x0000_0000_0100_ffff,
    0x0000_0000_0100_ff00,
    0x0000_0000_0100_00ff,
    0x0000_0000_0100_0000,
    0x0000_0000_0100_0001,
    0x0000_0000_0100_01ff,
    0x0000_0000_0100_0100,
    0x0000_0000_0101_ff00,
    0x0000_0000_0101_00ff,
    0x0000_0000_0101_0000,
    0x0000_0000_0101_0001,
    0x0000_0000_0101_0100,
    0x0000_0001_ffff_ffff,
    0x0000_0001_ffff_ff00,
    0x0000_0001_ffff_ff01,
    0x0000_0001_ffff_00ff,
    0x0000_0001_ffff_0001,
    0x0000_0001_ffff_01ff,
    0x0000_0001_ffff_0100,
    0x0000_0001_ff00_ff00,
    0x0000_0001_ff00_00ff,
    0x0000_0001_ff00_0000,
    0x0000_0001_ff00_01ff,
    0x0000_0001_ff00_0100,
    0x0000_0001_ff01_ffff,
    0x0000_0001_ff01_ff00,
    0x0000_0001_ff01_ff01,
    0x0000_0001_ff01_00ff,
    0x0000_0001_ff01_0000,
    0x0000_0001_ff01_0001,
    0x0000_0001_ff01_01ff,
    0x0000_0001_ff01_0100,
    0x0000_0001_00ff_ff00,
    0x0000_0001_00ff_0000,
    0x0000_0001_00ff_0001,
    0x0000_0001_00ff_01ff,
    0x0000_0001_00ff_0100,
    0x0000_0001_00ff_0101,
    0x0000_0001_0000_ffff,
    0x0000_0001_0000_ff00,
    0x0000_0001_0000_ff01,
    0x0000_0001_0000_00ff,
    0x0000_0001_0000_0000,
    0x0000_0001_0000_0001,
    0x0000_0001_0000_01ff,
    0x0000_0001_0000_0100,
    0x0000_0001_0000_0101,
    0x0000_0001_0001_ff00,
    0x0000_0001_0001_00ff,
    0x0000_0001_0001_0000,
    0x0000_0001_0001_0100,
    0x0000_0001_01ff_ff01,
    0x0000_0001_01ff_0000,
    0x0000_0001_01ff_0001,
    0x0000_0001_01ff_01ff,
    0x0000_0001_01ff_0100,
    0x0000_0001_01ff_0101,
    0x0000_0001_0100_ff00,
    0x0000_0001_0100_0000,
    0x0000_0001_0100_0101,
    0x0000_0001_0101_ff01,
    0x0000_0001_0101_0000,
    0x0000_0001_0101_0001,
    0x0000_0001_0101_01ff,
    0x0000_0001_0101_0100,
    0x0000_01ff_ffff_00ff,
    0x0000_01ff_ffff_0000,
    0x0000_01ff_ffff_0001,
    0x0000_01ff_ffff_0100,
    0x0000_01ff_ff00_ffff,
    0x0000_01ff_ff00_0000,
    0x0000_01ff_ff00_01ff,
    0x0000_01ff_ff01_ff00,
    0x0000_01ff_ff01_0101,
    0x0000_01ff_00ff_0000,
    0x0000_01ff_00ff_01ff,
    0x0000_01ff_00ff_0101,
    0x0000_01ff_0000_ff00,
    0x0000_01ff_0000_00ff,
    0x0000_01ff_0000_0000,
    0x0000_01ff_0000_0001,
    0x0000_01ff_0000_01ff,
    0x0000_01ff_0000_0100,
    0x0000_01ff_0001_ffff,
    0x0000_01ff_0001_ff01,
    0x0000_01ff_0001_00ff,
    0x0000_01ff_0001_0000,
    0x0000_01ff_01ff_ff01,
    0x0000_01ff_01ff_0100,
    0x0000_01ff_0100_ffff,
    0x0000_01ff_0100_ff01,
    0x0000_01ff_0100_0000,
    0x0000_01ff_0100_01ff,
    0x0000_01ff_0101_ff00,
    0x0000_01ff_0101_0100,
    0x0000_0100_ffff_ff00,
    0x0000_0100_ffff_ff01,
    0x0000_0100_ffff_0000,
    0x0000_0100_ffff_0101,
    0x0000_0100_ff00_ff00,
    0x0000_0100_ff00_00ff,
    0x0000_0100_ff00_0000,
    0x0000_0100_ff00_0001,
    0x0000_0100_ff00_0100,
    0x0000_0100_ff01_0000,
    0x0000_0100_00ff_ff00,
    0x0000_0100_00ff_00ff,
    0x0000_0100_00ff_0000,
    0x0000_0100_00ff_0001,
    0x0000_0100_00ff_0100,
    0x0000_0100_0000_ffff,
    0x0000_0100_0000_ff00,
    0x0000_0100_0000_ff01,
    0x0000_0100_0000_00ff,
    0x0000_0100_0000_0000,
    0x0000_0100_0000_0001,
    0x0000_0100_0000_01ff,
    0x0000_0100_0000_0100,
    0x0000_0100_0000_0101,
    0x0000_0100_0001_ff00,
    0x0000_0100_0001_00ff,
    0x0000_0100_0001_0000,
    0x0000_0100_0001_0001,
    0x0000_0100_0001_0100,
    0x0000_0100_01ff_ff00,
    0x0000_0100_01ff_0000,
    0x0000_0100_01ff_0100,
    0x0000_0100_0100_ff00,
    0x0000_0100_0100_00ff,
    0x0000_0100_0100_0000,
    0x0000_0100_0100_0001,
    0x0000_0100_0100_01ff,
    0x0000_0100_0100_0100,
    0x0000_0100_0101_0000,
    0x0000_0101_ffff_00ff,
    0x0000_0101_ffff_01ff,
    0x0000_0101_ff00_0000,
    0x0000_0101_ff00_0101,
    0x0000_0101_ff01_ffff,
    0x0000_0101_ff01_0000,
    0x0000_0101_ff01_0001,
    0x0000_0101_ff01_0100,
    0x0000_0101_00ff_0000,
    0x0000_0101_00ff_01ff,
    0x0000_0101_00ff_0100,
    0x0000_0101_0000_ff00,
    0x0000_0101_0000_0000,
    0x0000_0101_0000_0001,
    0x0000_0101_0000_01ff,
    0x0000_0101_0000_0100,
    0x0000_0101_0001_ff01,
    0x0000_0101_0001_0000,
    0x0000_0101_0001_01ff,
    0x0000_0101_0001_0101,
    0x0000_0101_01ff_ff00,
    0x0000_0101_01ff_0101,
    0x0000_0101_0100_ff01,
    0x0000_0101_0100_0000,
    0x0000_0101_0100_0001,
    0x0000_0101_0100_01ff,
    0x0000_0101_0100_0101,
    0x0000_0101_0101_ff00,
    0x0001_ffff_ffff_0000,
    0x0001_ffff_ff00_00ff,
    0x0001_ffff_ff00_0001,
    0x0001_ffff_ff00_0100,
    0x0001_ffff_ff01_0000,
    0x0001_ffff_00ff_00ff,
    0x0001_ffff_0000_ffff,
    0x0001_ffff_0000_0000,
    0x0001_ffff_0000_0001,
    0x0001_ffff_0000_01ff,
    0x0001_ffff_0000_0101,
    0x0001_ffff_0001_ff00,
    0x0001_ffff_0001_00ff,
    0x0001_ffff_0001_0001,
    0x0001_ffff_0001_0100,
    0x0001_ffff_01ff_ff00,
    0x0001_ffff_0100_0001,
    0x0001_ffff_0101_0000,
    0x0001_ff00_ffff_ff00,
    0x0001_ff00_ffff_00ff,
    0x0001_ff00_ffff_0001,
    0x0001_ff00_ffff_0100,
    0x0001_ff00_ff00_ff01,
    0x0001_ff00_ff00_0000,
    0x0001_ff00_ff01_ff00,
    0x0001_ff00_ff01_ff01,
    0x0001_ff00_ff01_0001,
    0x0001_ff00_ff01_0100,
    0x0001_ff00_00ff_0000,
    0x0001_ff00_00ff_0100,
    0x0001_ff00_0000_ff00,
    0x0001_ff00_0000_0000,
    0x0001_ff00_0000_0001,
    0x0001_ff00_0000_0100,
    0x0001_ff00_0001_0000,
    0x0001_ff00_0001_0001,
    0x0001_ff00_0001_0101,
    0x0001_ff00_01ff_00ff,
    0x0001_ff00_01ff_0101,
    0x0001_ff00_0100_ff01,
    0x0001_ff00_0100_0000,
    0x0001_ff00_0101_ff00,
    0x0001_ff00_0101_0001,
    0x0001_ff00_0101_0100,
    0x0001_ff01_ff00_ff00,
    0x0001_ff01_ff00_0001,
    0x0001_ff01_ff00_0100,
    0x0001_ff01_00ff_ffff,
    0x0001_ff01_00ff_ff00,
    0x0001_ff01_00ff_0001,
    0x0001_ff01_0000_0000,
    0x0001_ff01_0000_0001,
    0x0001_ff01_0000_01ff,
    0x0001_ff01_0001_ffff,
    0x0001_ff01_01ff_0000,
    0x0001_ff01_0100_ff00,
    0x0001_ff01_0100_0001,
    0x0001_ff01_0101_0000,
    0x0001_00ff_ff00_ff00,
    0x0001_00ff_ff00_ff01,
    0x0001_00ff_ff00_0000,
    0x0001_00ff_ff00_0001,
    0x0001_00ff_ff00_0101,
    0x0001_00ff_ff01_ff00,
    0x0001_00ff_ff01_0001,
    0x0001_00ff_ff01_0100,
    0x0001_00ff_00ff_ffff,
    0x0001_00ff_00ff_ff01,
    0x0001_00ff_00ff_0000,
    0x0001_00ff_00ff_01ff,
    0x0001_00ff_00ff_0101,
    0x0001_00ff_0000_ff00,
    0x0001_00ff_0000_00ff,
    0x0001_00ff_0000_0000,
    0x0001_00ff_0000_0001,
    0x0001_00ff_0000_0100,
    0x0001_00ff_0000_0101,
    0x0001_00ff_0001_ffff,
    0x0001_00ff_0001_ff01,
    0x0001_00ff_0001_0000,
    0x0001_00ff_01ff_00ff,
    0x0001_00ff_01ff_0000,
    0x0001_00ff_01ff_0100,
    0x0001_00ff_0100_ffff,
    0x0001_00ff_0100_ff01,
    0x0001_00ff_0100_00ff,
    0x0001_00ff_0100_0000,
    0x0001_00ff_0100_0001,
    0x0001_00ff_0100_01ff,
    0x0001_00ff_0100_0101,
    0x0001_00ff_0101_ff00,
    0x0001_00ff_0101_00ff,
    0x0001_00ff_0101_0100,
    0x0001_0000_ffff_0000,
    0x0001_0000_ffff_01ff,
    0x0001_0000_ffff_0101,
    0x0001_0000_ff00_ff00,
    0x0001_0000_ff00_0000,
    0x0001_0000_ff00_0001,
    0x0001_0000_ff00_0100,
    0x0001_0000_00ff_00ff,
    0x0001_0000_00ff_0000,
    0x0001_0000_00ff_0001,
    0x0001_0000_00ff_0100,
    0x0001_0000_0000_ffff,
    0x0001_0000_0000_ff00,
    0x0001_0000_0000_00ff,
    0x0001_0000_0000_0000,
    0x0001_0000_0000_0001,
    0x0001_0000_0000_0100,
    0x0001_0000_0001_ff00,
    0x0001_0000_0001_00ff,
    0x0001_0000_0001_0000,
    0x0001_0000_0001_0001,
    0x0001_0000_0001_0100,
    0x0001_0000_01ff_0001,
    0x0001_0000_01ff_0100,
    0x0001_0000_01ff_0101,
    0x0001_0000_0100_ff00,
    0x0001_0000_0100_0000,
    0x0001_0000_0100_0001,
    0x0001_0000_0100_0100,
    0x0001_0000_0100_0101,
    0x0001_0000_0101_ff01,
    0x0001_0000_0101_0000,
    0x0001_0000_0101_0001,
    0x0001_0000_0101_01ff,
    0x0001_0001_ffff_ff01,
    0x0001_0001_ffff_0100,
    0x0001_0001_ff00_0000,
    0x0001_0001_ff01_ffff,
    0x0001_0001_ff01_0001,
    0x0001_0001_ff01_01ff,
    0x0001_0001_ff01_0100,
    0x0001_0001_00ff_ffff,
    0x0001_0001_00ff_0000,
    0x0001_0001_00ff_01ff,
    0x0001_0001_00ff_0101,
    0x0001_0001_0000_ff00,
    0x0001_0001_0000_00ff,
    0x0001_0001_0000_0000,
    0x0001_0001_0000_0001,
    0x0001_0001_0000_01ff,
    0x0001_0001_0000_0101,
    0x0001_0001_0001_ffff,
    0x0001_0001_0001_0000,
    0x0001_0001_0001_01ff,
    0x0001_0001_01ff_ffff,
    0x0001_0001_01ff_ff01,
    0x0001_0001_01ff_0000,
    0x0001_0001_01ff_0101,
    0x0001_0001_0100_00ff,
    0x0001_0001_0100_0001,
    0x0001_0001_0100_01ff,
    0x0001_0001_0100_0100,
    0x0001_0001_0101_ffff,
    0x0001_0001_0101_00ff,
    0x0001_0001_0101_0001,
    0x0001_0001_0101_0101,
    0x0001_01ff_ff00_0001,
    0x0001_01ff_ff00_0100,
    0x0001_01ff_ff01_0000,
    0x0001_01ff_00ff_ff00,
    0x0001_01ff_0000_ff01,
    0x0001_01ff_0000_0000,
    0x0001_01ff_0000_0101,
    0x0001_01ff_0001_ff00,
    0x0001_01ff_0001_0100,
    0x0001_01ff_01ff_0000,
    0x0001_01ff_0100_ff00,
    0x0001_01ff_0100_01ff,
    0x0001_01ff_0101_0001,
    0x0001_0100_ffff_ff00,
    0x0001_0100_ffff_00ff,
    0x0001_0100_ff00_ffff,
    0x0001_0100_ff00_0000,
    0x0001_0100_ff01_ff00,
    0x0001_0100_ff01_00ff,
    0x0001_0100_ff01_0001,
    0x0001_0100_ff01_0100,
    0x0001_0100_00ff_ffff,
    0x0001_0100_00ff_ff00,
    0x0001_0100_00ff_0000,
    0x0001_0100_00ff_0001,
    0x0001_0100_00ff_01ff,
    0x0001_0100_0000_ff00,
    0x0001_0100_0000_00ff,
    0x0001_0100_0000_0000,
    0x0001_0100_0000_0001,
    0x0001_0100_0000_0100,
    0x0001_0100_0001_ffff,
    0x0001_0100_0001_0000,
    0x0001_0100_0001_0101,
    0x0001_0100_01ff_ff01,
    0x0001_0100_01ff_00ff,
    0x0001_0100_01ff_0101,
    0x0001_0100_0100_0000,
    0x0001_0100_0101_ff00,
    0x0001_0100_0101_00ff,
    0x0001_0100_0101_0000,
    0x0001_0100_0101_0100,
    0x0001_0101_ff00_ff00,
    0x0001_0101_ff00_0001,
    0x0001_0101_ff00_01ff,
    0x0001_0101_00ff_ff00,
    0x0001_0101_00ff_00ff,
    0x0001_0101_00ff_0100,
    0x0001_0101_0000_ffff,
    0x0001_0101_0000_0000,
    0x0001_0101_0000_01ff,
    0x0001_0101_0000_0101,
    0x0001_0101_0001_00ff,
    0x0001_0101_0001_0000,
    0x0001_0101_0001_0100,
    0x0001_0101_01ff_0001,
    0x0001_0101_0100_00ff,
    0x0001_0101_0100_01ff,
    0x0001_0101_0100_0101,
    0x0001_0101_0101_0001,
    0x01ff_ffff_ffff_ffff,
    0x01ff_ffff_ffff_ff01,
    0x01ff_ffff_ffff_01ff,
    0x01ff_ffff_ffff_0101,
    0x01ff_ffff_ff01_ffff,
    0x01ff_ffff_ff01_ff01,
    0x01ff_ffff_ff01_01ff,
    0x01ff_ffff_ff01_0101,
    0x01ff_ffff_00ff_0000,
    0x01ff_ffff_0000_ffff,
    0x01ff_ffff_0000_ff00,
    0x01ff_ffff_0000_00ff,
    0x01ff_ffff_0000_0001,
    0x01ff_ffff_0000_0100,
    0x01ff_ffff_0001_0000,
    0x01ff_ffff_01ff_ffff,
    0x01ff_ffff_01ff_ff01,
    0x01ff_ffff_01ff_01ff,
    0x01ff_ffff_01ff_0101,
    0x01ff_ffff_0100_0000,
    0x01ff_ffff_0101_ffff,
    0x01ff_ffff_0101_ff01,
    0x01ff_ffff_0101_01ff,
    0x01ff_ffff_0101_0101,
    0x01ff_ff00_ffff_0000,
    0x01ff_ff00_ff00_ff00,
    0x01ff_ff00_ff00_00ff,
    0x01ff_ff00_ff00_0001,
    0x01ff_ff00_ff00_0100,
    0x01ff_ff00_ff01_0000,
    0x01ff_ff00_00ff_ff00,
    0x01ff_ff00_00ff_00ff,
    0x01ff_ff00_00ff_0100,
    0x01ff_ff00_0000_ffff,
    0x01ff_ff00_0000_ff01,
    0x01ff_ff00_0000_0000,
    0x01ff_ff00_0000_0001,
    0x01ff_ff00_0000_01ff,
    0x01ff_ff00_0000_0100,
    0x01ff_ff00_0001_00ff,
    0x01ff_ff00_0001_0001,
    0x01ff_ff00_0001_0100,
    0x01ff_ff00_01ff_0000,
    0x01ff_ff00_01ff_0100,
    0x01ff_ff00_0100_00ff,
    0x01ff_ff00_0100_0001,
    0x01ff_ff00_0100_0100,
    0x01ff_ff00_0101_0000,
    0x01ff_ff01_ffff_ffff,
    0x01ff_ff01_ffff_ff01,
    0x01ff_ff01_ffff_01ff,
    0x01ff_ff01_ffff_0101,
    0x01ff_ff01_ff00_0000,
    0x01ff_ff01_ff01_ffff,
    0x01ff_ff01_ff01_ff01,
    0x01ff_ff01_ff01_01ff,
    0x01ff_ff01_ff01_0101,
    0x01ff_ff01_0000_ff00,
    0x01ff_ff01_0000_00ff,
    0x01ff_ff01_0000_0100,
    0x01ff_ff01_0001_0000,
    0x01ff_ff01_01ff_ffff,
    0x01ff_ff01_01ff_ff01,
    0x01ff_ff01_01ff_01ff,
    0x01ff_ff01_01ff_0101,
    0x01ff_ff01_0100_0000,
    0x01ff_ff01_0101_ffff,
    0x01ff_ff01_0101_ff01,
    0x01ff_ff01_0101_01ff,
    0x01ff_ff01_0101_0101,
    0x01ff_00ff_ff00_00ff,
    0x01ff_00ff_ff00_0100,
    0x01ff_00ff_00ff_ff00,
    0x01ff_00ff_00ff_00ff,
    0x01ff_00ff_0000_ff00,
    0x01ff_00ff_0000_0000,
    0x01ff_00ff_0000_0101,
    0x01ff_00ff_0001_ff00,
    0x01ff_00ff_0001_00ff,
    0x01ff_00ff_0001_0100,
    0x01ff_00ff_0100_00ff,
    0x01ff_00ff_0100_0100,
    0x01ff_0000_ffff_ff00,
    0x01ff_0000_ffff_0100,
    0x01ff_0000_ff00_ff01,
    0x01ff_0000_ff00_0000,
    0x01ff_0000_ff00_0101,
    0x01ff_0000_ff01_0001,
    0x01ff_0000_ff01_0100,
    0x01ff_0000_00ff_ffff,
    0x01ff_0000_00ff_ff00,
    0x01ff_0000_00ff_0000,
    0x01ff_0000_00ff_01ff,
    0x01ff_0000_0000_ff00,
    0x01ff_0000_0000_00ff,
    0x01ff_0000_0000_0000,
    0x01ff_0000_0000_0001,
    0x01ff_0000_0000_0100,
    0x01ff_0000_0000_0101,
    0x01ff_0000_0001_0000,
    0x01ff_0000_0001_0001,
    0x01ff_0000_0001_01ff,
    0x01ff_0000_0001_0101,
    0x01ff_0000_01ff_ff00,
    0x01ff_0000_01ff_00ff,
    0x01ff_0000_01ff_0001,
    0x01ff_0000_01ff_0100,
    0x01ff_0000_0100_ffff,
    0x01ff_0000_0100_ff01,
    0x01ff_0000_0100_0000,
    0x01ff_0000_0100_01ff,
    0x01ff_0000_0101_0001,
    0x01ff_0001_ff00_ff00,
    0x01ff_0001_ff00_0001,
    0x01ff_0001_ff00_0100,
    0x01ff_0001_ff01_0000,
    0x01ff_0001_00ff_ff00,
    0x01ff_0001_00ff_00ff,
    0x01ff_0001_00ff_0100,
    0x01ff_0001_00ff_0101,
    0x01ff_0001_0000_ffff,
    0x01ff_0001_0000_0000,
    0x01ff_0001_0000_0100,
    0x01ff_0001_0000_0101,
    0x01ff_0001_0001_ff00,
    0x01ff_0001_0001_0001,
    0x01ff_0001_0001_0101,
    0x01ff_0001_01ff_0000,
    0x01ff_0001_0100_ff00,
    0x01ff_0001_0100_0101,
    0x01ff_0001_0101_00ff,
    0x01ff_01ff_ffff_ffff,
    0x01ff_01ff_ffff_ff01,
    0x01ff_01ff_ffff_01ff,
    0x01ff_01ff_ffff_0101,
    0x01ff_01ff_ff00_0000,
    0x01ff_01ff_ff01_ffff,
    0x01ff_01ff_ff01_ff01,
    0x01ff_01ff_ff01_01ff,
    0x01ff_01ff_ff01_0101,
    0x01ff_01ff_00ff_ff00,
    0x01ff_01ff_00ff_0000,
    0x01ff_01ff_0000_ff00,
    0x01ff_01ff_0000_00ff,
    0x01ff_01ff_0000_0100,
    0x01ff_01ff_0001_0000,
    0x01ff_01ff_0001_0100,
    0x01ff_01ff_01ff_ffff,
    0x01ff_01ff_01ff_ff01,
    0x01ff_01ff_01ff_01ff,
    0x01ff_01ff_01ff_0101,
    0x01ff_01ff_0100_0000,
    0x01ff_01ff_0101_ffff,
    0x01ff_01ff_0101_ff01,
    0x01ff_01ff_0101_01ff,
    0x01ff_01ff_0101_0101,
    0x01ff_0100_ffff_0000,
    0x01ff_0100_ffff_0001,
    0x01ff_0100_ff00_ff00,
    0x01ff_0100_ff00_00ff,
    0x01ff_0100_ff00_0001,
    0x01ff_0100_ff01_0000,
    0x01ff_0100_00ff_ff00,
    0x01ff_0100_00ff_00ff,
    0x01ff_0100_00ff_0001,
    0x01ff_0100_00ff_0100,
    0x01ff_0100_0000_ffff,
    0x01ff_0100_0000_ff01,
    0x01ff_0100_0000_0000,
    0x01ff_0100_0000_0101,
    0x01ff_0100_0001_ff00,
    0x01ff_0100_0001_00ff,
    0x01ff_0100_01ff_0000,
    0x01ff_0100_0100_0001,
    0x01ff_0100_0100_0100,
    0x01ff_0100_0101_0000,
    0x01ff_0101_ffff_ffff,
    0x01ff_0101_ffff_ff01,
    0x01ff_0101_ffff_01ff,
    0x01ff_0101_ffff_0101,
    0x01ff_0101_ff00_0000,
    0x01ff_0101_ff01_ffff,
    0x01ff_0101_ff01_ff01,
    0x01ff_0101_ff01_01ff,
    0x01ff_0101_ff01_0101,
    0x01ff_0101_00ff_0000,
    0x01ff_0101_0000_ff00,
    0x01ff_0101_0000_00ff,
    0x01ff_0101_0000_0001,
    0x01ff_0101_01ff_ffff,
    0x01ff_0101_01ff_ff01,
    0x01ff_0101_01ff_01ff,
    0x01ff_0101_01ff_0101,
    0x01ff_0101_0100_0000,
    0x01ff_0101_0101_ffff,
    0x01ff_0101_0101_ff01,
    0x01ff_0101_0101_01ff,
    0x01ff_0101_0101_0101,
    0x0100_ffff_ffff_0000,
    0x0100_ffff_ff00_ff00,
    0x0100_ffff_ff00_0001,
    0x0100_ffff_ff00_01ff,
    0x0100_ffff_ff00_0100,
    0x0100_ffff_ff01_0000,
    0x0100_ffff_00ff_ff00,
    0x0100_ffff_00ff_0001,
    0x0100_ffff_00ff_0100,
    0x0100_ffff_0000_0000,
    0x0100_ffff_0000_01ff,
    0x0100_ffff_0000_0101,
    0x0100_ffff_0001_0100,
    0x0100_ffff_0001_0101,
    0x0100_ffff_01ff_0000,
    0x0100_ffff_0100_ff00,
    0x0100_ffff_0100_00ff,
    0x0100_ffff_0100_0001,
    0x0100_ffff_0100_0100,
    0x0100_ffff_0101_0000,
    0x0100_ff00_ffff_ff00,
    0x0100_ff00_ffff_00ff,
    0x0100_ff00_ffff_0001,
    0x0100_ff00_ffff_0100,
    0x0100_ff00_ff00_ffff,
    0x0100_ff00_ff00_0000,
    0x0100_ff00_ff00_01ff,
    0x0100_ff00_ff00_0101,
    0x0100_ff00_ff01_ff00,
    0x0100_ff00_ff01_00ff,
    0x0100_ff00_ff01_0001,
    0x0100_ff00_ff01_0100,
    0x0100_ff00_00ff_ffff,
    0x0100_ff00_00ff_0000,
    0x0100_ff00_0000_ffff,
    0x0100_ff00_0000_ff00,
    0x0100_ff00_0000_00ff,
    0x0100_ff00_0000_0000,
    0x0100_ff00_0000_0001,
    0x0100_ff00_0000_0100,
    0x0100_ff00_0001_ff01,
    0x0100_ff00_0001_0000,
    0x0100_ff00_01ff_00ff,
    0x0100_ff00_01ff_0001,
    0x0100_ff00_0100_ff01,
    0x0100_ff00_0100_0000,
    0x0100_ff00_0100_01ff,
    0x0100_ff00_0101_ff00,
    0x0100_ff00_0101_00ff,
    0x0100_ff00_0101_0001,
    0x0100_ff00_0101_0100,
    0x0100_ff01_ffff_0000,
    0x0100_ff01_ff00_ff00,
    0x0100_ff01_ff00_00ff,
    0x0100_ff01_ff00_0100,
    0x0100_ff01_ff01_0000,
    0x0100_ff01_00ff_00ff,
    0x0100_ff01_00ff_0001,
    0x0100_ff01_00ff_0100,
    0x0100_ff01_0000_ffff,
    0x0100_ff01_0000_ff01,
    0x0100_ff01_0000_0000,
    0x0100_ff01_0000_01ff,
    0x0100_ff01_0001_0001,
    0x0100_ff01_0001_0100,
    0x0100_ff01_01ff_0000,
    0x0100_ff01_0100_00ff,
    0x0100_ff01_0100_0001,
    0x0100_ff01_0101_0100,
    0x0100_00ff_ffff_ff00,
    0x0100_00ff_ffff_00ff,
    0x0100_00ff_ffff_0001,
    0x0100_00ff_ff00_ffff,
    0x0100_00ff_ff00_0000,
    0x0100_00ff_ff00_01ff,
    0x0100_00ff_ff01_0001,
    0x0100_00ff_00ff_ffff,
    0x0100_00ff_00ff_0101,
    0x0100_00ff_0000_ff00,
    0x0100_00ff_0000_00ff,
    0x0100_00ff_0000_0000,
    0x0100_00ff_0000_0001,
    0x0100_00ff_0000_01ff,
    0x0100_00ff_0000_0100,
    0x0100_00ff_0001_ffff,
    0x0100_00ff_0001_ff00,
    0x0100_00ff_0001_ff01,
    0x0100_00ff_0001_0000,
    0x0100_00ff_01ff_00ff,
    0x0100_00ff_01ff_0001,
    0x0100_00ff_0100_ff01,
    0x0100_00ff_0100_00ff,
    0x0100_00ff_0100_0000,
    0x0100_00ff_0100_01ff,
    0x0100_00ff_0101_ff00,
    0x0100_00ff_0101_0100,
    0x0100_0000_ffff_ffff,
    0x0100_0000_ffff_0000,
    0x0100_0000_ffff_01ff,
    0x0100_0000_ffff_0101,
    0x0100_0000_ff00_ffff,
    0x0100_0000_ff00_ff00,
    0x0100_0000_ff00_00ff,
    0x0100_0000_ff00_0000,
    0x0100_0000_ff00_0001,
    0x0100_0000_ff00_0100,
    0x0100_0000_ff01_ff00,
    0x0100_0000_ff01_0000,
    0x0100_0000_ff01_0100,
    0x0100_0000_ff01_0101,
    0x0100_0000_00ff_ff00,
    0x0100_0000_00ff_00ff,
    0x0100_0000_00ff_0000,
    0x0100_0000_00ff_0001,
    0x0100_0000_00ff_0100,
    0x0100_0000_0000_ffff,
    0x0100_0000_0000_ff00,
    0x0100_0000_0000_ff01,
    0x0100_0000_0000_00ff,
    0x0100_0000_0000_0000,
    0x0100_0000_0000_0001,
    0x0100_0000_0000_01ff,
    0x0100_0000_0000_0100,
    0x0100_0000_0000_0101,
    0x0100_0000_0001_ff00,
    0x0100_0000_0001_00ff,
    0x0100_0000_0001_0000,
    0x0100_0000_0001_0001,
    0x0100_0000_0001_0100,
    0x0100_0000_01ff_ff00,
    0x0100_0000_01ff_0000,
    0x0100_0000_01ff_01ff,
    0x0100_0000_0100_ff00,
    0x0100_0000_0100_ff01,
    0x0100_0000_0100_00ff,
    0x0100_0000_0100_0000,
    0x0100_0000_0100_0001,
    0x0100_0000_0100_0100,
    0x0100_0000_0100_0101,
    0x0100_0000_0101_ffff,
    0x0100_0000_0101_ff01,
    0x0100_0000_0101_0000,
    0x0100_0000_0101_01ff,
    0x0100_0000_0101_0101,
    0x0100_0001_ffff_ff00,
    0x0100_0001_ffff_00ff,
    0x0100_0001_ff00_ffff,
    0x0100_0001_ff00_0000,
    0x0100_0001_ff00_0100,
    0x0100_0001_ff01_ffff,
    0x0100_0001_ff01_0001,
    0x0100_0001_ff01_0100,
    0x0100_0001_00ff_0000,
    0x0100_0001_00ff_01ff,
    0x0100_0001_00ff_0100,
    0x0100_0001_0000_ff00,
    0x0100_0001_0000_ff01,
    0x0100_0001_0000_0000,
    0x0100_0001_0000_0001,
    0x0100_0001_0000_0100,
    0x0100_0001_0001_0000,
    0x0100_0001_0001_01ff,
    0x0100_0001_01ff_ff01,
    0x0100_0001_01ff_00ff,
    0x0100_0001_01ff_0100,
    0x0100_0001_01ff_0101,
    0x0100_0001_0100_ff01,
    0x0100_0001_0100_00ff,
    0x0100_0001_0100_0000,
    0x0100_0001_0101_00ff,
    0x0100_0001_0101_0001,
    0x0100_0001_0101_0100,
    0x0100_01ff_ffff_0000,
    0x0100_01ff_ff00_0001,
    0x0100_01ff_ff00_0100,
    0x0100_01ff_ff01_0000,
    0x0100_01ff_00ff_ff00,
    0x0100_01ff_00ff_0001,
    0x0100_01ff_0000_ffff,
    0x0100_01ff_0000_ff01,
    0x0100_01ff_0000_0000,
    0x0100_01ff_0000_0001,
    0x0100_01ff_0000_0101,
    0x0100_01ff_0001_00ff,
    0x0100_01ff_0001_0000,
    0x0100_01ff_01ff_0000,
    0x0100_01ff_0100_ff00,
    0x0100_01ff_0100_0001,
    0x0100_01ff_0100_0100,
    0x0100_01ff_0101_0000,
    0x0100_0100_ffff_00ff,
    0x0100_0100_ffff_0001,
    0x0100_0100_ffff_0100,
    0x0100_0100_ff00_ffff,
    0x0100_0100_ff00_ff01,
    0x0100_0100_ff00_0000,
    0x0100_0100_ff00_01ff,
    0x0100_0100_ff00_0101,
    0x0100_0100_ff01_ffff,
    0x0100_0100_ff01_ff00,
    0x0100_0100_ff01_00ff,
    0x0100_0100_ff01_0001,
    0x0100_0100_00ff_ffff,
    0x0100_0100_00ff_ff01,
    0x0100_0100_00ff_0000,
    0x0100_0100_00ff_01ff,
    0x0100_0100_00ff_0101,
    0x0100_0100_0000_ff00,
    0x0100_0100_0000_00ff,
    0x0100_0100_0000_0000,
    0x0100_0100_0000_0001,
    0x0100_0100_0000_0100,
    0x0100_0100_0001_ff01,
    0x0100_0100_0001_0000,
    0x0100_0100_0001_0001,
    0x0100_0100_0001_0101,
    0x0100_0100_01ff_ff00,
    0x0100_0100_01ff_00ff,
    0x0100_0100_0100_ffff,
    0x0100_0100_0100_ff01,
    0x0100_0100_0100_0000,
    0x0100_0100_0100_0101,
    0x0100_0100_0101_ff00,
    0x0100_0100_0101_0001,
    0x0100_0101_ffff_0000,
    0x0100_0101_ff00_0000,
    0x0100_0101_ff01_0000,
    0x0100_0101_00ff_00ff,
    0x0100_0101_00ff_0001,
    0x0100_0101_00ff_0100,
    0x0100_0101_0000_ffff,
    0x0100_0101_0000_0000,
    0x0100_0101_0000_01ff,
    0x0100_0101_0001_ff00,
    0x0100_0101_01ff_0000,
    0x0100_0101_0100_ff00,
    0x0100_0101_0100_00ff,
    0x0100_0101_0100_0000,
    0x0100_0101_0100_0001,
    0x0101_ffff_ffff_ffff,
    0x0101_ffff_ffff_ff01,
    0x0101_ffff_ffff_01ff,
    0x0101_ffff_ffff_0101,
    0x0101_ffff_ff00_0000,
    0x0101_ffff_ff01_ffff,
    0x0101_ffff_ff01_ff01,
    0x0101_ffff_ff01_01ff,
    0x0101_ffff_ff01_0101,
    0x0101_ffff_00ff_0000,
    0x0101_ffff_0000_ff00,
    0x0101_ffff_0000_00ff,
    0x0101_ffff_0000_0001,
    0x0101_ffff_0000_0100,
    0x0101_ffff_01ff_ffff,
    0x0101_ffff_01ff_ff01,
    0x0101_ffff_01ff_01ff,
    0x0101_ffff_01ff_0101,
    0x0101_ffff_0100_0000,
    0x0101_ffff_0101_ffff,
    0x0101_ffff_0101_ff01,
    0x0101_ffff_0101_01ff,
    0x0101_ffff_0101_0101,
    0x0101_ff00_ffff_0000,
    0x0101_ff00_ffff_0100,
    0x0101_ff00_ff00_ff00,
    0x0101_ff00_ff00_00ff,
    0x0101_ff00_ff00_0001,
    0x0101_ff00_ff00_0100,
    0x0101_ff00_ff00_0101,
    0x0101_ff00_00ff_0001,
    0x0101_ff00_00ff_0100,
    0x0101_ff00_0000_ff00,
    0x0101_ff00_0000_0000,
    0x0101_ff00_0000_01ff,
    0x0101_ff00_0000_0101,
    0x0101_ff00_0001_ff00,
    0x0101_ff00_0001_00ff,
    0x0101_ff00_01ff_0000,
    0x0101_ff00_0100_ffff,
    0x0101_ff00_0100_ff01,
    0x0101_ff00_0100_0001,
    0x0101_ff00_0100_0100,
    0x0101_ff01_ffff_ff01,
    0x0101_ff01_ffff_01ff,
    0x0101_ff01_ffff_0101,
    0x0101_ff01_ff00_ffff,
    0x0101_ff01_ff00_0100,
    0x0101_ff01_ff01_ff01,
    0x0101_ff01_ff01_01ff,
    0x0101_ff01_ff01_0101,
    0x0101_ff01_00ff_0000,
    0x0101_ff01_0000_ff00,
    0x0101_ff01_0000_0001,
    0x0101_ff01_0000_0100,
    0x0101_ff01_0001_0000,
    0x0101_ff01_01ff_ffff,
    0x0101_ff01_01ff_ff01,
    0x0101_ff01_01ff_01ff,
    0x0101_ff01_01ff_0101,
    0x0101_ff01_0100_0000,
    0x0101_ff01_0101_ffff,
    0x0101_ff01_0101_ff01,
    0x0101_ff01_0101_01ff,
    0x0101_ff01_0101_0101,
    0x0101_00ff_ff00_0100,
    0x0101_00ff_ff01_0000,
    0x0101_00ff_00ff_ff00,
    0x0101_00ff_00ff_00ff,
    0x0101_00ff_0000_ffff,
    0x0101_00ff_0000_00ff,
    0x0101_00ff_0000_0000,
    0x0101_00ff_0000_01ff,
    0x0101_00ff_0000_0101,
    0x0101_00ff_0001_ff00,
    0x0101_00ff_0001_0000,
    0x0101_00ff_0001_0001,
    0x0101_00ff_0001_01ff,
    0x0101_00ff_0001_0100,
    0x0101_00ff_01ff_0000,
    0x0101_0000_ffff_0001,
    0x0101_0000_ffff_0100,
    0x0101_0000_ff00_ffff,
    0x0101_0000_ff00_ff01,
    0x0101_0000_ff00_0000,
    0x0101_0000_ff00_01ff,
    0x0101_0000_ff01_0001,
    0x0101_0000_ff01_0100,
    0x0101_0000_00ff_ff01,
    0x0101_0000_00ff_0000,
    0x0101_0000_0000_ff00,
    0x0101_0000_0000_00ff,
    0x0101_0000_0000_0000,
    0x0101_0000_0000_0001,
    0x0101_0000_0000_0100,
    0x0101_0000_0001_0000,
    0x0101_0000_0001_0101,
    0x0101_0000_01ff_ff00,
    0x0101_0000_01ff_00ff,
    0x0101_0000_01ff_0000,
    0x0101_0000_01ff_0001,
    0x0101_0000_01ff_0100,
    0x0101_0000_0100_ff01,
    0x0101_0000_0100_0000,
    0x0101_0000_0100_01ff,
    0x0101_0001_ffff_0000,
    0x0101_0001_ff00_ff00,
    0x0101_0001_ff00_0001,
    0x0101_0001_ff00_0101,
    0x0101_0001_ff01_ff00,
    0x0101_0001_ff01_0000,
    0x0101_0001_00ff_00ff,
    0x0101_0001_00ff_0001,
    0x0101_0001_00ff_0101,
    0x0101_0001_0000_ff01,
    0x0101_0001_0000_0000,
    0x0101_0001_0000_0001,
    0x0101_0001_0000_01ff,
    0x0101_0001_0001_ffff,
    0x0101_0001_0001_ff01,
    0x0101_0001_01ff_0001,
    0x0101_0001_0100_ffff,
    0x0101_0001_0100_0000,
    0x0101_0001_0100_0001,
    0x0101_0001_0100_0100,
    0x0101_0001_0101_ff00,
    0x0101_0001_0101_00ff,
    0x0101_0001_0101_0001,
    0x0101_01ff_ffff_ffff,
    0x0101_01ff_ffff_ff01,
    0x0101_01ff_ffff_01ff,
    0x0101_01ff_ffff_0101,
    0x0101_01ff_ff01_ffff,
    0x0101_01ff_ff01_ff01,
    0x0101_01ff_ff01_01ff,
    0x0101_01ff_ff01_0101,
    0x0101_01ff_0000_ff00,
    0x0101_01ff_0000_00ff,
    0x0101_01ff_0000_0001,
    0x0101_01ff_0000_0100,
    0x0101_01ff_01ff_ffff,
    0x0101_01ff_01ff_ff01,
    0x0101_01ff_01ff_01ff,
    0x0101_01ff_01ff_0101,
    0x0101_01ff_0100_0000,
    0x0101_01ff_0101_ffff,
    0x0101_01ff_0101_ff01,
    0x0101_01ff_0101_01ff,
    0x0101_01ff_0101_0101,
    0x0101_0100_ffff_0000,
    0x0101_0100_ff00_00ff,
    0x0101_0100_ff00_0100,
    0x0101_0100_ff01_ff00,
    0x0101_0100_ff01_0000,
    0x0101_0100_00ff_ff00,
    0x0101_0100_0000_ffff,
    0x0101_0100_0000_0000,
    0x0101_0100_0000_0101,
    0x0101_0100_0001_ff00,
    0x0101_0100_0001_0001,
    0x0101_0100_0001_0100,
    0x0101_0100_0100_ffff,
    0x0101_0100_0100_0001,
    0x0101_0101_ffff_ffff,
    0x0101_0101_ffff_ff01,
    0x0101_0101_ffff_01ff,
    0x0101_0101_ffff_0101,
    0x0101_0101_ff01_ffff,
    0x0101_0101_ff01_ff01,
    0x0101_0101_ff01_01ff,
    0x0101_0101_ff01_0101,
    0x0101_0101_0000_ff00,
    0x0101_0101_0000_00ff,
    0x0101_0101_0000_0001,
    0x0101_0101_01ff_ffff,
    0x0101_0101_01ff_ff01,
    0x0101_0101_01ff_01ff,
    0x0101_0101_01ff_0101,
    0x0101_0101_0100_0000,
    0x0101_0101_0101_ffff,
    0x0101_0101_0101_ff01,
    0x0101_0101_0101_01ff,
    0x0101_0101_0101_0101,
];

#[expect(
    clippy::unreadable_literal,
    reason = "Preserve pinned ggml-common.h IQ2 codebook literals."
)]
static IQ2XXS_GRID: [u64; 256] = [
    0x0808080808080808,
    0x080808080808082b,
    0x0808080808081919,
    0x0808080808082b08,
    0x0808080808082b2b,
    0x0808080808190819,
    0x0808080808191908,
    0x08080808082b0808,
    0x08080808082b082b,
    0x08080808082b2b08,
    0x08080808082b2b2b,
    0x0808080819080819,
    0x0808080819081908,
    0x0808080819190808,
    0x0808080819192b08,
    0x08080808192b0819,
    0x08080808192b1908,
    0x080808082b080808,
    0x080808082b08082b,
    0x080808082b082b2b,
    0x080808082b2b082b,
    0x0808081908080819,
    0x0808081908081908,
    0x0808081908190808,
    0x0808081908191919,
    0x0808081919080808,
    0x080808192b081908,
    0x080808192b192b08,
    0x0808082b08080808,
    0x0808082b0808082b,
    0x0808082b082b082b,
    0x0808082b2b08082b,
    0x0808190808080819,
    0x0808190808081908,
    0x0808190808190808,
    0x08081908082b0819,
    0x08081908082b1908,
    0x0808190819080808,
    0x080819081908082b,
    0x0808190819082b08,
    0x08081908192b0808,
    0x080819082b080819,
    0x080819082b081908,
    0x080819082b190808,
    0x080819082b2b1908,
    0x0808191908080808,
    0x080819190808082b,
    0x0808191908082b08,
    0x08081919082b0808,
    0x080819191908192b,
    0x08081919192b2b19,
    0x080819192b080808,
    0x080819192b190819,
    0x0808192b08082b19,
    0x0808192b08190808,
    0x0808192b19080808,
    0x0808192b2b081908,
    0x0808192b2b2b1908,
    0x08082b0808080808,
    0x08082b0808081919,
    0x08082b0808082b08,
    0x08082b0808191908,
    0x08082b08082b2b08,
    0x08082b0819080819,
    0x08082b0819081908,
    0x08082b0819190808,
    0x08082b081919082b,
    0x08082b082b082b08,
    0x08082b1908081908,
    0x08082b1919080808,
    0x08082b2b0808082b,
    0x08082b2b08191908,
    0x0819080808080819,
    0x0819080808081908,
    0x0819080808190808,
    0x08190808082b0819,
    0x0819080819080808,
    0x08190808192b0808,
    0x081908082b081908,
    0x081908082b190808,
    0x081908082b191919,
    0x0819081908080808,
    0x0819081908082b08,
    0x08190819082b0808,
    0x0819081919190808,
    0x0819081919192b2b,
    0x081908192b080808,
    0x0819082b082b1908,
    0x0819082b19081919,
    0x0819190808080808,
    0x0819190808082b08,
    0x08191908082b0808,
    0x08191908082b1919,
    0x0819190819082b19,
    0x081919082b080808,
    0x0819191908192b08,
    0x08191919192b082b,
    0x0819192b08080808,
    0x0819192b0819192b,
    0x08192b0808080819,
    0x08192b0808081908,
    0x08192b0808190808,
    0x08192b0819080808,
    0x08192b082b080819,
    0x08192b1908080808,
    0x08192b1908081919,
    0x08192b192b2b0808,
    0x08192b2b19190819,
    0x082b080808080808,
    0x082b08080808082b,
    0x082b080808082b2b,
    0x082b080819081908,
    0x082b0808192b0819,
    0x082b08082b080808,
    0x082b08082b08082b,
    0x082b0819082b2b19,
    0x082b081919082b08,
    0x082b082b08080808,
    0x082b082b0808082b,
    0x082b190808080819,
    0x082b190808081908,
    0x082b190808190808,
    0x082b190819080808,
    0x082b19081919192b,
    0x082b191908080808,
    0x082b191919080819,
    0x082b1919192b1908,
    0x082b192b2b190808,
    0x082b2b0808082b08,
    0x082b2b08082b0808,
    0x082b2b082b191908,
    0x082b2b2b19081908,
    0x1908080808080819,
    0x1908080808081908,
    0x1908080808190808,
    0x1908080808192b08,
    0x19080808082b0819,
    0x19080808082b1908,
    0x1908080819080808,
    0x1908080819082b08,
    0x190808081919192b,
    0x19080808192b0808,
    0x190808082b080819,
    0x190808082b081908,
    0x190808082b190808,
    0x1908081908080808,
    0x19080819082b0808,
    0x19080819192b0819,
    0x190808192b080808,
    0x190808192b081919,
    0x1908082b08080819,
    0x1908082b08190808,
    0x1908082b19082b08,
    0x1908082b1919192b,
    0x1908082b192b2b08,
    0x1908190808080808,
    0x1908190808082b08,
    0x19081908082b0808,
    0x190819082b080808,
    0x190819082b192b19,
    0x190819190819082b,
    0x19081919082b1908,
    0x1908192b08080808,
    0x19082b0808080819,
    0x19082b0808081908,
    0x19082b0808190808,
    0x19082b0819080808,
    0x19082b0819081919,
    0x19082b1908080808,
    0x19082b1919192b08,
    0x19082b19192b0819,
    0x19082b192b08082b,
    0x19082b2b19081919,
    0x19082b2b2b190808,
    0x1919080808080808,
    0x1919080808082b08,
    0x1919080808190819,
    0x1919080808192b19,
    0x19190808082b0808,
    0x191908082b080808,
    0x191908082b082b08,
    0x1919081908081908,
    0x191908191908082b,
    0x191908192b2b1908,
    0x1919082b2b190819,
    0x191919082b190808,
    0x191919082b19082b,
    0x1919191908082b2b,
    0x1919192b08080819,
    0x1919192b19191908,
    0x19192b0808080808,
    0x19192b0808190819,
    0x19192b0808192b19,
    0x19192b08192b1908,
    0x19192b1919080808,
    0x19192b2b08082b08,
    0x192b080808081908,
    0x192b080808190808,
    0x192b080819080808,
    0x192b0808192b2b08,
    0x192b081908080808,
    0x192b081919191919,
    0x192b082b08192b08,
    0x192b082b192b0808,
    0x192b190808080808,
    0x192b190808081919,
    0x192b191908190808,
    0x192b19190819082b,
    0x192b19192b081908,
    0x192b2b081908082b,
    0x2b08080808080808,
    0x2b0808080808082b,
    0x2b08080808082b2b,
    0x2b08080819080819,
    0x2b0808082b08082b,
    0x2b08081908081908,
    0x2b08081908192b08,
    0x2b08081919080808,
    0x2b08082b08190819,
    0x2b08190808080819,
    0x2b08190808081908,
    0x2b08190808190808,
    0x2b08190808191919,
    0x2b08190819080808,
    0x2b081908192b0808,
    0x2b08191908080808,
    0x2b0819191908192b,
    0x2b0819192b191908,
    0x2b08192b08082b19,
    0x2b08192b19080808,
    0x2b08192b192b0808,
    0x2b082b080808082b,
    0x2b082b1908081908,
    0x2b082b2b08190819,
    0x2b19080808081908,
    0x2b19080808190808,
    0x2b190808082b1908,
    0x2b19080819080808,
    0x2b1908082b2b0819,
    0x2b1908190819192b,
    0x2b1908192b080808,
    0x2b19082b19081919,
    0x2b19190808080808,
    0x2b191908082b082b,
    0x2b19190819081908,
    0x2b19191919190819,
    0x2b192b082b080819,
    0x2b192b19082b0808,
    0x2b2b08080808082b,
    0x2b2b080819190808,
    0x2b2b08082b081919,
    0x2b2b081908082b19,
    0x2b2b082b08080808,
    0x2b2b190808192b08,
    0x2b2b2b0819190808,
    0x2b2b2b1908081908,
];

#[expect(
    clippy::unreadable_literal,
    reason = "Preserve pinned ggml-common.h IQ2 codebook literals."
)]
static IQ2XS_GRID: [u64; 512] = [
    0x0808080808080808,
    0x080808080808082b,
    0x0808080808081919,
    0x0808080808082b08,
    0x0808080808082b2b,
    0x0808080808190819,
    0x0808080808191908,
    0x080808080819192b,
    0x0808080808192b19,
    0x08080808082b0808,
    0x08080808082b082b,
    0x08080808082b1919,
    0x08080808082b2b08,
    0x0808080819080819,
    0x0808080819081908,
    0x080808081908192b,
    0x0808080819082b19,
    0x0808080819190808,
    0x080808081919082b,
    0x0808080819191919,
    0x0808080819192b08,
    0x08080808192b0819,
    0x08080808192b1908,
    0x080808082b080808,
    0x080808082b08082b,
    0x080808082b081919,
    0x080808082b082b08,
    0x080808082b190819,
    0x080808082b191908,
    0x080808082b192b19,
    0x080808082b2b0808,
    0x0808081908080819,
    0x0808081908081908,
    0x080808190808192b,
    0x0808081908082b19,
    0x0808081908190808,
    0x080808190819082b,
    0x0808081908191919,
    0x0808081908192b08,
    0x0808081908192b2b,
    0x08080819082b0819,
    0x08080819082b1908,
    0x0808081919080808,
    0x080808191908082b,
    0x0808081919081919,
    0x0808081919082b08,
    0x0808081919190819,
    0x0808081919191908,
    0x08080819192b0808,
    0x08080819192b2b08,
    0x080808192b080819,
    0x080808192b081908,
    0x080808192b190808,
    0x0808082b08080808,
    0x0808082b0808082b,
    0x0808082b08081919,
    0x0808082b08082b08,
    0x0808082b08190819,
    0x0808082b08191908,
    0x0808082b082b0808,
    0x0808082b19080819,
    0x0808082b19081908,
    0x0808082b19190808,
    0x0808082b19191919,
    0x0808082b2b080808,
    0x0808082b2b082b2b,
    0x0808190808080819,
    0x0808190808081908,
    0x080819080808192b,
    0x0808190808082b19,
    0x0808190808190808,
    0x080819080819082b,
    0x0808190808191919,
    0x0808190808192b08,
    0x08081908082b0819,
    0x08081908082b1908,
    0x0808190819080808,
    0x080819081908082b,
    0x0808190819081919,
    0x0808190819082b08,
    0x0808190819190819,
    0x0808190819191908,
    0x080819081919192b,
    0x08081908192b0808,
    0x080819082b080819,
    0x080819082b081908,
    0x080819082b190808,
    0x0808191908080808,
    0x080819190808082b,
    0x0808191908081919,
    0x0808191908082b08,
    0x0808191908190819,
    0x0808191908191908,
    0x08081919082b0808,
    0x0808191919080819,
    0x0808191919081908,
    0x0808191919190808,
    0x08081919192b0819,
    0x080819192b080808,
    0x0808192b08080819,
    0x0808192b08081908,
    0x0808192b08190808,
    0x0808192b082b192b,
    0x0808192b19080808,
    0x0808192b1908082b,
    0x0808192b2b081908,
    0x08082b0808080808,
    0x08082b080808082b,
    0x08082b0808081919,
    0x08082b0808082b08,
    0x08082b0808082b2b,
    0x08082b0808190819,
    0x08082b0808191908,
    0x08082b08082b0808,
    0x08082b08082b1919,
    0x08082b0819080819,
    0x08082b0819081908,
    0x08082b0819190808,
    0x08082b0819192b08,
    0x08082b082b080808,
    0x08082b082b2b0808,
    0x08082b082b2b2b2b,
    0x08082b1908080819,
    0x08082b1908081908,
    0x08082b1908190808,
    0x08082b1919080808,
    0x08082b192b080819,
    0x08082b192b082b19,
    0x08082b2b08080808,
    0x08082b2b082b0808,
    0x08082b2b082b2b08,
    0x08082b2b2b19192b,
    0x08082b2b2b2b0808,
    0x0819080808080819,
    0x0819080808081908,
    0x081908080808192b,
    0x0819080808082b19,
    0x0819080808190808,
    0x081908080819082b,
    0x0819080808191919,
    0x0819080808192b08,
    0x08190808082b0819,
    0x08190808082b1908,
    0x0819080819080808,
    0x081908081908082b,
    0x0819080819081919,
    0x0819080819082b08,
    0x0819080819190819,
    0x0819080819191908,
    0x08190808192b0808,
    0x08190808192b2b2b,
    0x081908082b080819,
    0x081908082b081908,
    0x081908082b190808,
    0x0819081908080808,
    0x081908190808082b,
    0x0819081908081919,
    0x0819081908082b08,
    0x0819081908190819,
    0x0819081908191908,
    0x08190819082b0808,
    0x0819081919080819,
    0x0819081919081908,
    0x0819081919190808,
    0x081908192b080808,
    0x081908192b191908,
    0x081908192b19192b,
    0x0819082b08080819,
    0x0819082b08081908,
    0x0819082b0808192b,
    0x0819082b08190808,
    0x0819082b19080808,
    0x0819082b192b0808,
    0x0819190808080808,
    0x081919080808082b,
    0x0819190808081919,
    0x0819190808082b08,
    0x0819190808190819,
    0x0819190808191908,
    0x08191908082b0808,
    0x0819190819080819,
    0x0819190819081908,
    0x0819190819082b19,
    0x0819190819190808,
    0x08191908192b1908,
    0x081919082b080808,
    0x0819191908080819,
    0x0819191908081908,
    0x0819191908190808,
    0x0819191919080808,
    0x0819192b08080808,
    0x0819192b08191908,
    0x0819192b19082b19,
    0x08192b0808080819,
    0x08192b0808081908,
    0x08192b0808190808,
    0x08192b080819082b,
    0x08192b0819080808,
    0x08192b0819191908,
    0x08192b082b08192b,
    0x08192b1908080808,
    0x08192b1908081919,
    0x08192b19192b192b,
    0x08192b2b19190819,
    0x08192b2b2b2b2b19,
    0x082b080808080808,
    0x082b08080808082b,
    0x082b080808081919,
    0x082b080808082b08,
    0x082b080808082b2b,
    0x082b080808190819,
    0x082b080808191908,
    0x082b0808082b0808,
    0x082b080819080819,
    0x082b080819081908,
    0x082b080819190808,
    0x082b08082b080808,
    0x082b08082b2b0808,
    0x082b081908080819,
    0x082b081908081908,
    0x082b081908190808,
    0x082b081919080808,
    0x082b081919082b08,
    0x082b0819192b1919,
    0x082b082b08080808,
    0x082b082b082b082b,
    0x082b082b2b080808,
    0x082b082b2b2b2b08,
    0x082b190808080819,
    0x082b190808081908,
    0x082b190808190808,
    0x082b1908082b2b19,
    0x082b190819080808,
    0x082b191908080808,
    0x082b191919080819,
    0x082b19191919082b,
    0x082b19192b192b19,
    0x082b192b08080819,
    0x082b192b08192b2b,
    0x082b192b2b2b192b,
    0x082b2b0808080808,
    0x082b2b0808082b08,
    0x082b2b0808082b2b,
    0x082b2b08082b0808,
    0x082b2b0819191919,
    0x082b2b082b082b08,
    0x082b2b082b2b082b,
    0x082b2b19192b2b08,
    0x082b2b192b190808,
    0x082b2b2b08082b08,
    0x082b2b2b082b0808,
    0x082b2b2b2b08082b,
    0x082b2b2b2b082b08,
    0x082b2b2b2b082b2b,
    0x1908080808080819,
    0x1908080808081908,
    0x190808080808192b,
    0x1908080808082b19,
    0x1908080808190808,
    0x190808080819082b,
    0x1908080808191919,
    0x1908080808192b08,
    0x19080808082b0819,
    0x19080808082b1908,
    0x1908080819080808,
    0x190808081908082b,
    0x1908080819081919,
    0x1908080819082b08,
    0x1908080819082b2b,
    0x1908080819190819,
    0x1908080819191908,
    0x19080808192b0808,
    0x19080808192b1919,
    0x190808082b080819,
    0x190808082b081908,
    0x190808082b190808,
    0x1908081908080808,
    0x190808190808082b,
    0x1908081908081919,
    0x1908081908082b08,
    0x1908081908190819,
    0x1908081908191908,
    0x19080819082b0808,
    0x1908081919080819,
    0x1908081919081908,
    0x1908081919190808,
    0x190808192b080808,
    0x190808192b081919,
    0x190808192b2b082b,
    0x1908082b08080819,
    0x1908082b08081908,
    0x1908082b08190808,
    0x1908082b0819082b,
    0x1908082b082b2b19,
    0x1908082b19080808,
    0x1908190808080808,
    0x190819080808082b,
    0x1908190808081919,
    0x1908190808082b08,
    0x1908190808190819,
    0x1908190808191908,
    0x1908190808192b19,
    0x19081908082b0808,
    0x1908190819080819,
    0x1908190819081908,
    0x1908190819190808,
    0x190819082b080808,
    0x190819082b191908,
    0x1908191908080819,
    0x1908191908081908,
    0x1908191908190808,
    0x19081919082b1908,
    0x1908191919080808,
    0x190819192b192b2b,
    0x1908192b08080808,
    0x1908192b08082b2b,
    0x1908192b19081908,
    0x1908192b19190808,
    0x19082b0808080819,
    0x19082b0808081908,
    0x19082b0808190808,
    0x19082b0819080808,
    0x19082b0819081919,
    0x19082b0819191908,
    0x19082b08192b082b,
    0x19082b1908080808,
    0x19082b1908190819,
    0x19082b1919081908,
    0x19082b1919190808,
    0x19082b19192b2b19,
    0x19082b2b08081908,
    0x1919080808080808,
    0x191908080808082b,
    0x1919080808081919,
    0x1919080808082b08,
    0x1919080808190819,
    0x1919080808191908,
    0x19190808082b0808,
    0x19190808082b2b08,
    0x1919080819080819,
    0x1919080819081908,
    0x1919080819190808,
    0x191908082b080808,
    0x1919081908080819,
    0x1919081908081908,
    0x1919081908190808,
    0x1919081908191919,
    0x1919081919080808,
    0x191908191908082b,
    0x1919082b08080808,
    0x1919082b19081908,
    0x1919082b2b2b2b2b,
    0x1919190808080819,
    0x1919190808081908,
    0x1919190808190808,
    0x19191908082b0819,
    0x1919190819080808,
    0x19191908192b0808,
    0x191919082b080819,
    0x191919082b2b0819,
    0x1919191908080808,
    0x1919191908082b08,
    0x191919192b080808,
    0x191919192b082b08,
    0x1919192b082b0819,
    0x1919192b192b2b08,
    0x1919192b2b2b0819,
    0x19192b0808080808,
    0x19192b0808191908,
    0x19192b0819080819,
    0x19192b0819190808,
    0x19192b082b192b19,
    0x19192b1908192b2b,
    0x19192b1919080808,
    0x19192b191908082b,
    0x19192b2b2b081919,
    0x192b080808080819,
    0x192b080808081908,
    0x192b080808190808,
    0x192b080819080808,
    0x192b080819191908,
    0x192b0808192b082b,
    0x192b08082b08192b,
    0x192b08082b2b2b19,
    0x192b081908080808,
    0x192b082b082b1908,
    0x192b082b19082b2b,
    0x192b082b2b19082b,
    0x192b190808080808,
    0x192b19080819192b,
    0x192b191908190808,
    0x192b191919080808,
    0x192b191919081919,
    0x192b19192b2b1908,
    0x192b2b0808080819,
    0x192b2b08192b2b2b,
    0x192b2b19082b1919,
    0x192b2b2b0808192b,
    0x192b2b2b19191908,
    0x192b2b2b192b082b,
    0x2b08080808080808,
    0x2b0808080808082b,
    0x2b08080808081919,
    0x2b08080808082b08,
    0x2b08080808190819,
    0x2b08080808191908,
    0x2b080808082b0808,
    0x2b080808082b2b2b,
    0x2b08080819080819,
    0x2b08080819081908,
    0x2b08080819190808,
    0x2b0808082b080808,
    0x2b0808082b08082b,
    0x2b0808082b2b2b08,
    0x2b0808082b2b2b2b,
    0x2b08081908080819,
    0x2b08081908081908,
    0x2b0808190808192b,
    0x2b08081908190808,
    0x2b08081919080808,
    0x2b08081919190819,
    0x2b08081919192b19,
    0x2b08082b08080808,
    0x2b08082b082b0808,
    0x2b08082b2b080808,
    0x2b08082b2b08082b,
    0x2b08082b2b2b0808,
    0x2b08082b2b2b2b08,
    0x2b08190808080819,
    0x2b08190808081908,
    0x2b08190808190808,
    0x2b0819080819082b,
    0x2b08190808191919,
    0x2b08190819080808,
    0x2b081908192b0808,
    0x2b0819082b082b19,
    0x2b08191908080808,
    0x2b08191919081908,
    0x2b0819192b2b1919,
    0x2b08192b08192b08,
    0x2b08192b192b2b2b,
    0x2b082b0808080808,
    0x2b082b0808082b08,
    0x2b082b08082b1919,
    0x2b082b0819192b2b,
    0x2b082b082b080808,
    0x2b082b082b08082b,
    0x2b082b082b2b2b08,
    0x2b082b190808192b,
    0x2b082b2b082b082b,
    0x2b082b2b2b080808,
    0x2b082b2b2b082b08,
    0x2b082b2b2b19192b,
    0x2b082b2b2b2b2b08,
    0x2b19080808080819,
    0x2b19080808081908,
    0x2b19080808190808,
    0x2b19080819080808,
    0x2b1908081919192b,
    0x2b1908082b081908,
    0x2b19081908080808,
    0x2b190819082b082b,
    0x2b190819192b1908,
    0x2b19082b1919192b,
    0x2b19082b2b082b19,
    0x2b19190808080808,
    0x2b19190808081919,
    0x2b19190819081908,
    0x2b19190819190808,
    0x2b19190819192b08,
    0x2b191919082b2b19,
    0x2b1919192b190808,
    0x2b1919192b19082b,
    0x2b19192b19080819,
    0x2b192b0819190819,
    0x2b192b082b2b192b,
    0x2b192b1919082b19,
    0x2b192b2b08191919,
    0x2b192b2b192b0808,
    0x2b2b080808080808,
    0x2b2b08080808082b,
    0x2b2b080808082b08,
    0x2b2b080808082b2b,
    0x2b2b0808082b0808,
    0x2b2b0808082b2b2b,
    0x2b2b08082b2b0808,
    0x2b2b081919190819,
    0x2b2b081919192b19,
    0x2b2b08192b2b192b,
    0x2b2b082b08080808,
    0x2b2b082b0808082b,
    0x2b2b082b08082b08,
    0x2b2b082b082b2b2b,
    0x2b2b082b2b080808,
    0x2b2b082b2b2b0808,
    0x2b2b190819080808,
    0x2b2b19082b191919,
    0x2b2b192b192b1919,
    0x2b2b192b2b192b08,
    0x2b2b2b0808082b2b,
    0x2b2b2b08082b0808,
    0x2b2b2b08082b082b,
    0x2b2b2b08082b2b08,
    0x2b2b2b082b2b0808,
    0x2b2b2b082b2b2b08,
    0x2b2b2b1908081908,
    0x2b2b2b192b081908,
    0x2b2b2b192b08192b,
    0x2b2b2b2b082b2b08,
    0x2b2b2b2b082b2b2b,
    0x2b2b2b2b2b190819,
    0x2b2b2b2b2b2b2b2b,
];

#[expect(
    clippy::unreadable_literal,
    reason = "Preserve pinned ggml-common.h IQ2 codebook literals."
)]
static IQ2S_GRID: [u64; 1024] = [
    0x0808080808080808,
    0x080808080808082b,
    0x0808080808081919,
    0x0808080808082b08,
    0x0808080808082b2b,
    0x0808080808190819,
    0x0808080808191908,
    0x080808080819192b,
    0x0808080808192b19,
    0x08080808082b0808,
    0x08080808082b082b,
    0x08080808082b1919,
    0x08080808082b2b08,
    0x0808080819080819,
    0x0808080819081908,
    0x080808081908192b,
    0x0808080819082b19,
    0x0808080819190808,
    0x080808081919082b,
    0x0808080819191919,
    0x0808080819192b08,
    0x08080808192b0819,
    0x08080808192b1908,
    0x08080808192b192b,
    0x08080808192b2b19,
    0x080808082b080808,
    0x080808082b08082b,
    0x080808082b081919,
    0x080808082b082b08,
    0x080808082b190819,
    0x080808082b191908,
    0x080808082b2b0808,
    0x080808082b2b1919,
    0x080808082b2b2b2b,
    0x0808081908080819,
    0x0808081908081908,
    0x080808190808192b,
    0x0808081908082b19,
    0x0808081908190808,
    0x080808190819082b,
    0x0808081908191919,
    0x0808081908192b08,
    0x08080819082b0819,
    0x08080819082b1908,
    0x0808081919080808,
    0x080808191908082b,
    0x0808081919081919,
    0x0808081919082b08,
    0x0808081919190819,
    0x0808081919191908,
    0x080808191919192b,
    0x0808081919192b19,
    0x08080819192b0808,
    0x08080819192b1919,
    0x08080819192b2b08,
    0x080808192b080819,
    0x080808192b081908,
    0x080808192b190808,
    0x080808192b19082b,
    0x080808192b191919,
    0x080808192b2b0819,
    0x080808192b2b1908,
    0x0808082b08080808,
    0x0808082b0808082b,
    0x0808082b08081919,
    0x0808082b08082b08,
    0x0808082b08190819,
    0x0808082b08191908,
    0x0808082b082b0808,
    0x0808082b082b2b2b,
    0x0808082b19080819,
    0x0808082b19081908,
    0x0808082b1908192b,
    0x0808082b19082b19,
    0x0808082b19190808,
    0x0808082b19191919,
    0x0808082b2b080808,
    0x0808082b2b081919,
    0x0808082b2b082b2b,
    0x0808082b2b191908,
    0x0808082b2b2b082b,
    0x0808190808080819,
    0x0808190808081908,
    0x080819080808192b,
    0x0808190808082b19,
    0x0808190808190808,
    0x080819080819082b,
    0x0808190808191919,
    0x0808190808192b08,
    0x08081908082b0819,
    0x08081908082b1908,
    0x08081908082b192b,
    0x08081908082b2b19,
    0x0808190819080808,
    0x080819081908082b,
    0x0808190819081919,
    0x0808190819082b08,
    0x0808190819082b2b,
    0x0808190819190819,
    0x0808190819191908,
    0x080819081919192b,
    0x0808190819192b19,
    0x08081908192b0808,
    0x08081908192b082b,
    0x08081908192b1919,
    0x080819082b080819,
    0x080819082b081908,
    0x080819082b08192b,
    0x080819082b082b19,
    0x080819082b190808,
    0x080819082b191919,
    0x080819082b192b08,
    0x080819082b2b0819,
    0x080819082b2b1908,
    0x0808191908080808,
    0x080819190808082b,
    0x0808191908081919,
    0x0808191908082b08,
    0x0808191908082b2b,
    0x0808191908190819,
    0x0808191908191908,
    0x080819190819192b,
    0x0808191908192b19,
    0x08081919082b0808,
    0x08081919082b1919,
    0x08081919082b2b08,
    0x0808191919080819,
    0x0808191919081908,
    0x080819191908192b,
    0x0808191919082b19,
    0x0808191919190808,
    0x080819191919082b,
    0x0808191919191919,
    0x0808191919192b08,
    0x08081919192b0819,
    0x08081919192b1908,
    0x080819192b080808,
    0x080819192b08082b,
    0x080819192b081919,
    0x080819192b082b08,
    0x080819192b190819,
    0x080819192b191908,
    0x080819192b2b0808,
    0x0808192b08080819,
    0x0808192b08081908,
    0x0808192b0808192b,
    0x0808192b08082b19,
    0x0808192b08190808,
    0x0808192b08191919,
    0x0808192b19080808,
    0x0808192b19081919,
    0x0808192b19082b08,
    0x0808192b19190819,
    0x0808192b19191908,
    0x0808192b192b0808,
    0x0808192b2b080819,
    0x0808192b2b081908,
    0x0808192b2b190808,
    0x08082b0808080808,
    0x08082b080808082b,
    0x08082b0808081919,
    0x08082b0808082b08,
    0x08082b0808190819,
    0x08082b0808191908,
    0x08082b080819192b,
    0x08082b0808192b19,
    0x08082b08082b0808,
    0x08082b08082b1919,
    0x08082b08082b2b2b,
    0x08082b0819080819,
    0x08082b0819081908,
    0x08082b081908192b,
    0x08082b0819082b19,
    0x08082b0819190808,
    0x08082b081919082b,
    0x08082b0819191919,
    0x08082b0819192b08,
    0x08082b08192b0819,
    0x08082b08192b1908,
    0x08082b082b080808,
    0x08082b082b081919,
    0x08082b082b191908,
    0x08082b082b2b2b2b,
    0x08082b1908080819,
    0x08082b1908081908,
    0x08082b1908190808,
    0x08082b190819082b,
    0x08082b1908191919,
    0x08082b1908192b08,
    0x08082b19082b0819,
    0x08082b1919080808,
    0x08082b1919081919,
    0x08082b1919082b08,
    0x08082b1919190819,
    0x08082b1919191908,
    0x08082b19192b0808,
    0x08082b192b080819,
    0x08082b192b190808,
    0x08082b2b08080808,
    0x08082b2b08190819,
    0x08082b2b08191908,
    0x08082b2b082b082b,
    0x08082b2b082b2b08,
    0x08082b2b082b2b2b,
    0x08082b2b19190808,
    0x08082b2b2b192b19,
    0x0819080808080819,
    0x0819080808081908,
    0x081908080808192b,
    0x0819080808082b19,
    0x0819080808190808,
    0x081908080819082b,
    0x0819080808191919,
    0x0819080808192b08,
    0x08190808082b0819,
    0x08190808082b1908,
    0x08190808082b192b,
    0x0819080819080808,
    0x081908081908082b,
    0x0819080819081919,
    0x0819080819082b08,
    0x0819080819190819,
    0x0819080819191908,
    0x081908081919192b,
    0x0819080819192b19,
    0x08190808192b0808,
    0x08190808192b082b,
    0x08190808192b1919,
    0x08190808192b2b08,
    0x081908082b080819,
    0x081908082b081908,
    0x081908082b08192b,
    0x081908082b190808,
    0x081908082b191919,
    0x081908082b192b08,
    0x081908082b2b0819,
    0x081908082b2b1908,
    0x0819081908080808,
    0x081908190808082b,
    0x0819081908081919,
    0x0819081908082b08,
    0x0819081908082b2b,
    0x0819081908190819,
    0x0819081908191908,
    0x081908190819192b,
    0x0819081908192b19,
    0x08190819082b0808,
    0x08190819082b082b,
    0x08190819082b1919,
    0x08190819082b2b08,
    0x0819081919080819,
    0x0819081919081908,
    0x081908191908192b,
    0x0819081919082b19,
    0x0819081919190808,
    0x081908191919082b,
    0x0819081919191919,
    0x0819081919192b08,
    0x08190819192b0819,
    0x08190819192b1908,
    0x081908192b080808,
    0x081908192b08082b,
    0x081908192b081919,
    0x081908192b082b08,
    0x081908192b190819,
    0x081908192b191908,
    0x0819082b08080819,
    0x0819082b08081908,
    0x0819082b08082b19,
    0x0819082b08190808,
    0x0819082b08191919,
    0x0819082b082b0819,
    0x0819082b082b1908,
    0x0819082b19080808,
    0x0819082b19081919,
    0x0819082b19190819,
    0x0819082b19191908,
    0x0819082b2b080819,
    0x0819082b2b081908,
    0x0819082b2b190808,
    0x0819190808080808,
    0x081919080808082b,
    0x0819190808081919,
    0x0819190808082b08,
    0x0819190808190819,
    0x0819190808191908,
    0x081919080819192b,
    0x0819190808192b19,
    0x08191908082b0808,
    0x08191908082b1919,
    0x08191908082b2b08,
    0x0819190819080819,
    0x0819190819081908,
    0x081919081908192b,
    0x0819190819082b19,
    0x0819190819190808,
    0x081919081919082b,
    0x0819190819191919,
    0x0819190819192b08,
    0x08191908192b0819,
    0x08191908192b1908,
    0x081919082b080808,
    0x081919082b08082b,
    0x081919082b081919,
    0x081919082b082b08,
    0x081919082b190819,
    0x081919082b191908,
    0x081919082b2b0808,
    0x0819191908080819,
    0x0819191908081908,
    0x081919190808192b,
    0x0819191908082b19,
    0x0819191908190808,
    0x081919190819082b,
    0x0819191908191919,
    0x0819191908192b08,
    0x08191919082b0819,
    0x08191919082b1908,
    0x0819191919080808,
    0x081919191908082b,
    0x0819191919081919,
    0x0819191919082b08,
    0x0819191919190819,
    0x0819191919191908,
    0x08191919192b0808,
    0x081919192b080819,
    0x081919192b081908,
    0x081919192b190808,
    0x0819192b08080808,
    0x0819192b08081919,
    0x0819192b08082b08,
    0x0819192b08190819,
    0x0819192b08191908,
    0x0819192b082b0808,
    0x0819192b19080819,
    0x0819192b19081908,
    0x0819192b19190808,
    0x0819192b2b080808,
    0x0819192b2b2b2b2b,
    0x08192b0808080819,
    0x08192b0808081908,
    0x08192b080808192b,
    0x08192b0808082b19,
    0x08192b0808190808,
    0x08192b0808191919,
    0x08192b0808192b08,
    0x08192b08082b0819,
    0x08192b0819080808,
    0x08192b081908082b,
    0x08192b0819081919,
    0x08192b0819082b08,
    0x08192b0819190819,
    0x08192b0819191908,
    0x08192b08192b0808,
    0x08192b082b080819,
    0x08192b082b081908,
    0x08192b1908080808,
    0x08192b190808082b,
    0x08192b1908081919,
    0x08192b1908082b08,
    0x08192b1908190819,
    0x08192b1908191908,
    0x08192b19082b0808,
    0x08192b1919080819,
    0x08192b1919081908,
    0x08192b1919190808,
    0x08192b19192b2b19,
    0x08192b192b2b082b,
    0x08192b2b08081908,
    0x08192b2b08190808,
    0x08192b2b19080808,
    0x08192b2b1919192b,
    0x082b080808080808,
    0x082b08080808082b,
    0x082b080808081919,
    0x082b080808082b08,
    0x082b080808190819,
    0x082b080808191908,
    0x082b08080819192b,
    0x082b080808192b19,
    0x082b0808082b0808,
    0x082b0808082b1919,
    0x082b0808082b2b2b,
    0x082b080819080819,
    0x082b080819081908,
    0x082b080819190808,
    0x082b08081919082b,
    0x082b080819191919,
    0x082b0808192b1908,
    0x082b08082b080808,
    0x082b08082b082b2b,
    0x082b08082b191908,
    0x082b08082b2b2b2b,
    0x082b081908080819,
    0x082b081908081908,
    0x082b081908190808,
    0x082b08190819082b,
    0x082b081908191919,
    0x082b0819082b0819,
    0x082b081919080808,
    0x082b08191908082b,
    0x082b081919081919,
    0x082b081919190819,
    0x082b081919191908,
    0x082b0819192b0808,
    0x082b08192b080819,
    0x082b08192b081908,
    0x082b08192b190808,
    0x082b082b08080808,
    0x082b082b08082b2b,
    0x082b082b082b082b,
    0x082b082b082b2b08,
    0x082b082b082b2b2b,
    0x082b082b19081908,
    0x082b082b19190808,
    0x082b082b2b082b08,
    0x082b082b2b082b2b,
    0x082b082b2b2b2b08,
    0x082b190808080819,
    0x082b190808081908,
    0x082b19080808192b,
    0x082b190808082b19,
    0x082b190808190808,
    0x082b190808191919,
    0x082b190808192b08,
    0x082b1908082b0819,
    0x082b1908082b1908,
    0x082b190819080808,
    0x082b19081908082b,
    0x082b190819081919,
    0x082b190819082b08,
    0x082b190819190819,
    0x082b190819191908,
    0x082b1908192b0808,
    0x082b19082b080819,
    0x082b19082b081908,
    0x082b19082b190808,
    0x082b191908080808,
    0x082b191908081919,
    0x082b191908082b08,
    0x082b191908190819,
    0x082b191908191908,
    0x082b1919082b0808,
    0x082b191919080819,
    0x082b191919081908,
    0x082b191919190808,
    0x082b1919192b192b,
    0x082b19192b080808,
    0x082b192b08080819,
    0x082b192b08081908,
    0x082b192b08190808,
    0x082b192b19080808,
    0x082b192b19192b19,
    0x082b2b0808080808,
    0x082b2b0808081919,
    0x082b2b0808190819,
    0x082b2b0808191908,
    0x082b2b0819080819,
    0x082b2b0819081908,
    0x082b2b0819190808,
    0x082b2b082b082b2b,
    0x082b2b082b2b2b2b,
    0x082b2b1908080819,
    0x082b2b1908081908,
    0x082b2b1908190808,
    0x082b2b192b191919,
    0x082b2b2b08082b2b,
    0x082b2b2b082b082b,
    0x082b2b2b192b1908,
    0x082b2b2b2b082b08,
    0x082b2b2b2b082b2b,
    0x1908080808080819,
    0x1908080808081908,
    0x190808080808192b,
    0x1908080808082b19,
    0x1908080808190808,
    0x190808080819082b,
    0x1908080808191919,
    0x1908080808192b08,
    0x1908080808192b2b,
    0x19080808082b0819,
    0x19080808082b1908,
    0x19080808082b192b,
    0x1908080819080808,
    0x190808081908082b,
    0x1908080819081919,
    0x1908080819082b08,
    0x1908080819082b2b,
    0x1908080819190819,
    0x1908080819191908,
    0x190808081919192b,
    0x1908080819192b19,
    0x19080808192b0808,
    0x19080808192b082b,
    0x19080808192b1919,
    0x190808082b080819,
    0x190808082b081908,
    0x190808082b190808,
    0x190808082b191919,
    0x190808082b192b08,
    0x190808082b2b0819,
    0x190808082b2b1908,
    0x1908081908080808,
    0x190808190808082b,
    0x1908081908081919,
    0x1908081908082b08,
    0x1908081908190819,
    0x1908081908191908,
    0x190808190819192b,
    0x1908081908192b19,
    0x19080819082b0808,
    0x19080819082b082b,
    0x19080819082b1919,
    0x1908081919080819,
    0x1908081919081908,
    0x190808191908192b,
    0x1908081919082b19,
    0x1908081919190808,
    0x190808191919082b,
    0x1908081919191919,
    0x1908081919192b08,
    0x19080819192b0819,
    0x19080819192b1908,
    0x190808192b080808,
    0x190808192b08082b,
    0x190808192b081919,
    0x190808192b082b08,
    0x190808192b190819,
    0x190808192b191908,
    0x190808192b2b0808,
    0x1908082b08080819,
    0x1908082b08081908,
    0x1908082b08190808,
    0x1908082b0819082b,
    0x1908082b08191919,
    0x1908082b08192b08,
    0x1908082b082b1908,
    0x1908082b19080808,
    0x1908082b19081919,
    0x1908082b19082b08,
    0x1908082b19190819,
    0x1908082b19191908,
    0x1908082b192b0808,
    0x1908082b2b080819,
    0x1908082b2b081908,
    0x1908190808080808,
    0x190819080808082b,
    0x1908190808081919,
    0x1908190808082b08,
    0x1908190808082b2b,
    0x1908190808190819,
    0x1908190808191908,
    0x190819080819192b,
    0x1908190808192b19,
    0x19081908082b0808,
    0x19081908082b082b,
    0x19081908082b1919,
    0x19081908082b2b08,
    0x1908190819080819,
    0x1908190819081908,
    0x190819081908192b,
    0x1908190819082b19,
    0x1908190819190808,
    0x190819081919082b,
    0x1908190819191919,
    0x1908190819192b08,
    0x19081908192b0819,
    0x19081908192b1908,
    0x190819082b080808,
    0x190819082b08082b,
    0x190819082b081919,
    0x190819082b082b08,
    0x190819082b190819,
    0x190819082b191908,
    0x190819082b2b0808,
    0x1908191908080819,
    0x1908191908081908,
    0x190819190808192b,
    0x1908191908082b19,
    0x1908191908190808,
    0x190819190819082b,
    0x1908191908191919,
    0x1908191908192b08,
    0x19081919082b0819,
    0x19081919082b1908,
    0x1908191919080808,
    0x190819191908082b,
    0x1908191919081919,
    0x1908191919082b08,
    0x1908191919190819,
    0x1908191919191908,
    0x19081919192b0808,
    0x19081919192b2b2b,
    0x190819192b080819,
    0x190819192b081908,
    0x190819192b190808,
    0x1908192b08080808,
    0x1908192b0808082b,
    0x1908192b08081919,
    0x1908192b08082b08,
    0x1908192b08190819,
    0x1908192b08191908,
    0x1908192b082b0808,
    0x1908192b19080819,
    0x1908192b19081908,
    0x1908192b19190808,
    0x1908192b2b080808,
    0x1908192b2b2b1919,
    0x19082b0808080819,
    0x19082b0808081908,
    0x19082b0808082b19,
    0x19082b0808190808,
    0x19082b080819082b,
    0x19082b0808191919,
    0x19082b0808192b08,
    0x19082b08082b0819,
    0x19082b08082b1908,
    0x19082b0819080808,
    0x19082b081908082b,
    0x19082b0819081919,
    0x19082b0819082b08,
    0x19082b0819190819,
    0x19082b0819191908,
    0x19082b08192b0808,
    0x19082b082b081908,
    0x19082b082b190808,
    0x19082b1908080808,
    0x19082b190808082b,
    0x19082b1908081919,
    0x19082b1908082b08,
    0x19082b1908190819,
    0x19082b1908191908,
    0x19082b19082b0808,
    0x19082b1919080819,
    0x19082b1919081908,
    0x19082b1919190808,
    0x19082b192b080808,
    0x19082b192b19192b,
    0x19082b2b08080819,
    0x19082b2b08081908,
    0x19082b2b08190808,
    0x19082b2b19080808,
    0x1919080808080808,
    0x191908080808082b,
    0x1919080808081919,
    0x1919080808082b08,
    0x1919080808190819,
    0x1919080808191908,
    0x191908080819192b,
    0x1919080808192b19,
    0x19190808082b0808,
    0x19190808082b082b,
    0x19190808082b1919,
    0x19190808082b2b08,
    0x1919080819080819,
    0x1919080819081908,
    0x191908081908192b,
    0x1919080819082b19,
    0x1919080819190808,
    0x191908081919082b,
    0x1919080819191919,
    0x1919080819192b08,
    0x19190808192b0819,
    0x19190808192b1908,
    0x191908082b080808,
    0x191908082b08082b,
    0x191908082b081919,
    0x191908082b082b08,
    0x191908082b190819,
    0x191908082b191908,
    0x1919081908080819,
    0x1919081908081908,
    0x191908190808192b,
    0x1919081908082b19,
    0x1919081908190808,
    0x191908190819082b,
    0x1919081908191919,
    0x1919081908192b08,
    0x19190819082b0819,
    0x19190819082b1908,
    0x1919081919080808,
    0x191908191908082b,
    0x1919081919081919,
    0x1919081919082b08,
    0x1919081919190819,
    0x1919081919191908,
    0x19190819192b0808,
    0x191908192b080819,
    0x191908192b081908,
    0x191908192b190808,
    0x1919082b08080808,
    0x1919082b08081919,
    0x1919082b08082b08,
    0x1919082b08190819,
    0x1919082b08191908,
    0x1919082b082b0808,
    0x1919082b19080819,
    0x1919082b19081908,
    0x1919082b19190808,
    0x1919082b192b2b19,
    0x1919082b2b080808,
    0x1919190808080819,
    0x1919190808081908,
    0x191919080808192b,
    0x1919190808082b19,
    0x1919190808190808,
    0x191919080819082b,
    0x1919190808191919,
    0x1919190808192b08,
    0x19191908082b0819,
    0x19191908082b1908,
    0x1919190819080808,
    0x191919081908082b,
    0x1919190819081919,
    0x1919190819082b08,
    0x1919190819190819,
    0x1919190819191908,
    0x19191908192b0808,
    0x191919082b080819,
    0x191919082b081908,
    0x191919082b190808,
    0x1919191908080808,
    0x191919190808082b,
    0x1919191908081919,
    0x1919191908082b08,
    0x1919191908190819,
    0x1919191908191908,
    0x19191919082b0808,
    0x1919191919080819,
    0x1919191919081908,
    0x1919191919190808,
    0x191919192b080808,
    0x1919192b08080819,
    0x1919192b08081908,
    0x1919192b08190808,
    0x1919192b082b192b,
    0x1919192b19080808,
    0x19192b0808080808,
    0x19192b080808082b,
    0x19192b0808081919,
    0x19192b0808082b08,
    0x19192b0808190819,
    0x19192b0808191908,
    0x19192b08082b0808,
    0x19192b0819080819,
    0x19192b0819081908,
    0x19192b0819190808,
    0x19192b0819192b2b,
    0x19192b082b080808,
    0x19192b1908080819,
    0x19192b1908081908,
    0x19192b1908190808,
    0x19192b1919080808,
    0x19192b2b08080808,
    0x19192b2b08192b19,
    0x19192b2b2b081919,
    0x19192b2b2b2b2b08,
    0x192b080808080819,
    0x192b080808081908,
    0x192b08080808192b,
    0x192b080808190808,
    0x192b08080819082b,
    0x192b080808191919,
    0x192b080808192b08,
    0x192b0808082b0819,
    0x192b0808082b1908,
    0x192b080819080808,
    0x192b080819081919,
    0x192b080819082b08,
    0x192b080819190819,
    0x192b080819191908,
    0x192b0808192b0808,
    0x192b08082b081908,
    0x192b08082b190808,
    0x192b081908080808,
    0x192b08190808082b,
    0x192b081908081919,
    0x192b081908082b08,
    0x192b081908190819,
    0x192b081908191908,
    0x192b0819082b0808,
    0x192b081919080819,
    0x192b081919081908,
    0x192b081919190808,
    0x192b08192b080808,
    0x192b08192b192b19,
    0x192b082b08081908,
    0x192b082b08190808,
    0x192b082b19080808,
    0x192b082b1919192b,
    0x192b082b2b2b0819,
    0x192b190808080808,
    0x192b190808081919,
    0x192b190808082b08,
    0x192b190808190819,
    0x192b190808191908,
    0x192b1908082b0808,
    0x192b190819080819,
    0x192b190819081908,
    0x192b190819190808,
    0x192b19082b080808,
    0x192b191908080819,
    0x192b191908081908,
    0x192b191908190808,
    0x192b191919080808,
    0x192b191919082b2b,
    0x192b1919192b2b08,
    0x192b19192b19082b,
    0x192b192b08080808,
    0x192b192b2b191908,
    0x192b2b0808080819,
    0x192b2b0808081908,
    0x192b2b0808190808,
    0x192b2b08192b1919,
    0x192b2b082b192b08,
    0x192b2b1908080808,
    0x192b2b19082b2b2b,
    0x192b2b2b1908082b,
    0x192b2b2b2b2b0819,
    0x2b08080808080808,
    0x2b0808080808082b,
    0x2b08080808081919,
    0x2b08080808082b08,
    0x2b08080808190819,
    0x2b08080808191908,
    0x2b08080808192b19,
    0x2b080808082b0808,
    0x2b080808082b1919,
    0x2b08080819080819,
    0x2b08080819081908,
    0x2b08080819190808,
    0x2b0808081919082b,
    0x2b08080819191919,
    0x2b08080819192b08,
    0x2b080808192b0819,
    0x2b0808082b080808,
    0x2b0808082b081919,
    0x2b0808082b190819,
    0x2b0808082b191908,
    0x2b08081908080819,
    0x2b08081908081908,
    0x2b08081908082b19,
    0x2b08081908190808,
    0x2b0808190819082b,
    0x2b08081908191919,
    0x2b08081908192b08,
    0x2b080819082b0819,
    0x2b080819082b1908,
    0x2b08081919080808,
    0x2b0808191908082b,
    0x2b08081919081919,
    0x2b08081919082b08,
    0x2b08081919190819,
    0x2b08081919191908,
    0x2b0808192b080819,
    0x2b0808192b081908,
    0x2b0808192b190808,
    0x2b0808192b2b2b19,
    0x2b08082b08080808,
    0x2b08082b08081919,
    0x2b08082b08082b2b,
    0x2b08082b08190819,
    0x2b08082b08191908,
    0x2b08082b19080819,
    0x2b08082b19081908,
    0x2b08082b19190808,
    0x2b08190808080819,
    0x2b08190808081908,
    0x2b0819080808192b,
    0x2b08190808082b19,
    0x2b08190808190808,
    0x2b0819080819082b,
    0x2b08190808191919,
    0x2b08190808192b08,
    0x2b081908082b0819,
    0x2b08190819080808,
    0x2b0819081908082b,
    0x2b08190819081919,
    0x2b08190819082b08,
    0x2b08190819190819,
    0x2b08190819191908,
    0x2b081908192b0808,
    0x2b0819082b080819,
    0x2b0819082b081908,
    0x2b0819082b190808,
    0x2b08191908080808,
    0x2b0819190808082b,
    0x2b08191908081919,
    0x2b08191908082b08,
    0x2b08191908190819,
    0x2b08191908191908,
    0x2b081919082b0808,
    0x2b08191919080819,
    0x2b08191919081908,
    0x2b08191919190808,
    0x2b0819192b080808,
    0x2b0819192b082b2b,
    0x2b08192b08080819,
    0x2b08192b08081908,
    0x2b08192b08190808,
    0x2b08192b082b2b19,
    0x2b08192b19080808,
    0x2b082b0808080808,
    0x2b082b0808081919,
    0x2b082b0808190819,
    0x2b082b0808191908,
    0x2b082b0819080819,
    0x2b082b0819081908,
    0x2b082b0819190808,
    0x2b082b082b2b082b,
    0x2b082b1908080819,
    0x2b082b1908081908,
    0x2b082b1919080808,
    0x2b082b19192b1919,
    0x2b082b2b082b082b,
    0x2b082b2b19192b08,
    0x2b082b2b19192b2b,
    0x2b082b2b2b08082b,
    0x2b082b2b2b2b082b,
    0x2b19080808080819,
    0x2b19080808081908,
    0x2b19080808082b19,
    0x2b19080808190808,
    0x2b1908080819082b,
    0x2b19080808191919,
    0x2b19080808192b08,
    0x2b190808082b1908,
    0x2b19080819080808,
    0x2b1908081908082b,
    0x2b19080819081919,
    0x2b19080819082b08,
    0x2b19080819190819,
    0x2b19080819191908,
    0x2b190808192b0808,
    0x2b1908082b080819,
    0x2b1908082b081908,
    0x2b1908082b190808,
    0x2b19081908080808,
    0x2b19081908081919,
    0x2b19081908190819,
    0x2b19081908191908,
    0x2b19081919080819,
    0x2b19081919081908,
    0x2b19081919190808,
    0x2b19081919192b2b,
    0x2b19082b08080819,
    0x2b19082b08081908,
    0x2b19082b08190808,
    0x2b19082b19080808,
    0x2b19082b2b2b192b,
    0x2b19190808080808,
    0x2b1919080808082b,
    0x2b19190808081919,
    0x2b19190808082b08,
    0x2b19190808190819,
    0x2b19190808191908,
    0x2b191908082b0808,
    0x2b19190819080819,
    0x2b19190819081908,
    0x2b19190819190808,
    0x2b1919082b080808,
    0x2b1919082b19192b,
    0x2b19191908080819,
    0x2b19191908081908,
    0x2b19191908190808,
    0x2b19191919080808,
    0x2b1919192b192b08,
    0x2b1919192b2b0819,
    0x2b19192b08080808,
    0x2b19192b1908192b,
    0x2b19192b192b1908,
    0x2b192b0808080819,
    0x2b192b0808081908,
    0x2b192b0808190808,
    0x2b192b08082b192b,
    0x2b192b0819080808,
    0x2b192b082b2b2b19,
    0x2b192b1908080808,
    0x2b192b1919082b19,
    0x2b192b191919082b,
    0x2b192b2b2b190808,
    0x2b2b080808080808,
    0x2b2b080808081919,
    0x2b2b080808082b2b,
    0x2b2b080808191908,
    0x2b2b0808082b082b,
    0x2b2b0808082b2b2b,
    0x2b2b080819080819,
    0x2b2b080819081908,
    0x2b2b080819190808,
    0x2b2b08082b2b082b,
    0x2b2b08082b2b2b2b,
    0x2b2b081919080808,
    0x2b2b0819192b1919,
    0x2b2b082b0808082b,
    0x2b2b082b08082b2b,
    0x2b2b082b082b082b,
    0x2b2b082b082b2b08,
    0x2b2b082b082b2b2b,
    0x2b2b082b2b08082b,
    0x2b2b082b2b082b08,
    0x2b2b082b2b082b2b,
    0x2b2b082b2b2b2b08,
    0x2b2b190808080819,
    0x2b2b190808081908,
    0x2b2b190808190808,
    0x2b2b190819080808,
    0x2b2b19082b082b19,
    0x2b2b19082b2b1908,
    0x2b2b191908080808,
    0x2b2b191908192b19,
    0x2b2b192b19190819,
    0x2b2b2b0808082b2b,
    0x2b2b2b08082b2b08,
    0x2b2b2b082b2b082b,
    0x2b2b2b1919191908,
    0x2b2b2b192b08192b,
    0x2b2b2b2b08082b08,
    0x2b2b2b2b08082b2b,
    0x2b2b2b2b082b0808,
    0x2b2b2b2b082b082b,
    0x2b2b2b2b082b2b08,
    0x2b2b2b2b2b082b08,
    0x2b2b2b2b2b2b2b2b,
];

static IQ3XXS_GRID: [u32; 256] = [
    0x0404_0404,
    0x0404_0414,
    0x0404_0424,
    0x0404_0c0c,
    0x0404_0c1c,
    0x0404_0c3e,
    0x0404_1404,
    0x0404_1414,
    0x0404_1c0c,
    0x0404_2414,
    0x0404_3e1c,
    0x0404_3e2c,
    0x040c_040c,
    0x040c_041c,
    0x040c_0c04,
    0x040c_0c14,
    0x040c_140c,
    0x040c_142c,
    0x040c_1c04,
    0x040c_1c14,
    0x040c_240c,
    0x040c_2c24,
    0x040c_3e04,
    0x0414_0404,
    0x0414_0414,
    0x0414_0424,
    0x0414_0c0c,
    0x0414_1404,
    0x0414_1414,
    0x0414_1c0c,
    0x0414_1c1c,
    0x0414_1c3e,
    0x0414_2c0c,
    0x0414_2c3e,
    0x0414_3e2c,
    0x041c_040c,
    0x041c_043e,
    0x041c_0c04,
    0x041c_0c14,
    0x041c_142c,
    0x041c_3e04,
    0x0424_0c1c,
    0x0424_1c3e,
    0x0424_2424,
    0x0424_2c3e,
    0x0424_3e1c,
    0x0424_3e2c,
    0x042c_040c,
    0x042c_043e,
    0x042c_1c14,
    0x042c_2c14,
    0x0434_1c2c,
    0x0434_3424,
    0x043e_0c04,
    0x043e_0c24,
    0x043e_0c34,
    0x043e_241c,
    0x043e_340c,
    0x0c04_040c,
    0x0c04_041c,
    0x0c04_0c04,
    0x0c04_0c14,
    0x0c04_140c,
    0x0c04_141c,
    0x0c04_1c04,
    0x0c04_1c14,
    0x0c04_1c24,
    0x0c04_243e,
    0x0c04_2c04,
    0x0c0c_0404,
    0x0c0c_0414,
    0x0c0c_0c0c,
    0x0c0c_1404,
    0x0c0c_1414,
    0x0c14_040c,
    0x0c14_041c,
    0x0c14_0c04,
    0x0c14_0c14,
    0x0c14_140c,
    0x0c14_1c04,
    0x0c14_3e14,
    0x0c1c_0404,
    0x0c1c_0414,
    0x0c1c_1404,
    0x0c1c_1c0c,
    0x0c1c_2434,
    0x0c1c_3434,
    0x0c24_040c,
    0x0c24_042c,
    0x0c24_2c04,
    0x0c2c_1404,
    0x0c2c_1424,
    0x0c2c_2434,
    0x0c2c_3e0c,
    0x0c34_042c,
    0x0c3e_1414,
    0x0c3e_2404,
    0x1404_0404,
    0x1404_0414,
    0x1404_0c0c,
    0x1404_0c1c,
    0x1404_1404,
    0x1404_1414,
    0x1404_1434,
    0x1404_1c0c,
    0x1404_2414,
    0x140c_040c,
    0x140c_041c,
    0x140c_042c,
    0x140c_0c04,
    0x140c_0c14,
    0x140c_140c,
    0x140c_1c04,
    0x140c_341c,
    0x140c_343e,
    0x140c_3e04,
    0x1414_0404,
    0x1414_0414,
    0x1414_0c0c,
    0x1414_0c3e,
    0x1414_1404,
    0x1414_1414,
    0x1414_1c3e,
    0x1414_2404,
    0x1414_2c2c,
    0x141c_040c,
    0x141c_0c04,
    0x141c_0c24,
    0x141c_3e04,
    0x141c_3e24,
    0x1424_1c2c,
    0x1424_2c1c,
    0x142c_041c,
    0x142c_143e,
    0x142c_240c,
    0x142c_3e24,
    0x143e_040c,
    0x143e_041c,
    0x143e_0c34,
    0x143e_242c,
    0x1c04_040c,
    0x1c04_0c04,
    0x1c04_0c14,
    0x1c04_140c,
    0x1c04_141c,
    0x1c04_2c04,
    0x1c04_342c,
    0x1c04_3e14,
    0x1c0c_0404,
    0x1c0c_0414,
    0x1c0c_1404,
    0x1c0c_1c0c,
    0x1c0c_2424,
    0x1c0c_2434,
    0x1c14_040c,
    0x1c14_041c,
    0x1c14_0c04,
    0x1c14_142c,
    0x1c14_2c14,
    0x1c14_3e14,
    0x1c1c_0c0c,
    0x1c1c_1c1c,
    0x1c24_1c04,
    0x1c24_243e,
    0x1c24_3e14,
    0x1c2c_0404,
    0x1c2c_0434,
    0x1c2c_1414,
    0x1c2c_2c2c,
    0x1c34_0c24,
    0x1c34_1c34,
    0x1c34_341c,
    0x1c3e_1c1c,
    0x1c3e_3404,
    0x2404_0424,
    0x2404_0c3e,
    0x2404_1c2c,
    0x2404_1c3e,
    0x2404_2c1c,
    0x2404_2c3e,
    0x240c_3e24,
    0x2414_1404,
    0x2414_1c3e,
    0x2414_2404,
    0x2414_3404,
    0x2414_3434,
    0x241c_043e,
    0x241c_242c,
    0x2424_0424,
    0x2424_2c0c,
    0x2424_3424,
    0x242c_142c,
    0x242c_241c,
    0x242c_3e04,
    0x243e_042c,
    0x243e_0c04,
    0x243e_0c14,
    0x243e_1c04,
    0x2c04_0c14,
    0x2c04_240c,
    0x2c04_3e04,
    0x2c0c_0404,
    0x2c0c_0434,
    0x2c0c_1434,
    0x2c0c_2c2c,
    0x2c14_0c24,
    0x2c14_1c14,
    0x2c14_3e14,
    0x2c1c_0414,
    0x2c1c_2c1c,
    0x2c24_0c04,
    0x2c24_141c,
    0x2c24_143e,
    0x2c24_3e14,
    0x2c2c_0414,
    0x2c2c_1c0c,
    0x2c34_2c04,
    0x2c3e_1424,
    0x2c3e_2414,
    0x3404_1424,
    0x3404_2424,
    0x3404_2434,
    0x3404_3424,
    0x340c_140c,
    0x340c_340c,
    0x3414_0c3e,
    0x3414_3424,
    0x341c_1c04,
    0x341c_1c34,
    0x3424_2424,
    0x342c_042c,
    0x342c_2c14,
    0x3434_1c1c,
    0x343e_041c,
    0x343e_140c,
    0x3e04_041c,
    0x3e04_042c,
    0x3e04_043e,
    0x3e04_0c04,
    0x3e04_1c14,
    0x3e04_2c14,
    0x3e0c_1434,
    0x3e0c_2404,
    0x3e14_0c14,
    0x3e14_242c,
    0x3e14_2c14,
    0x3e1c_0404,
    0x3e1c_0c2c,
    0x3e1c_1c1c,
    0x3e1c_3404,
    0x3e24_140c,
    0x3e24_240c,
    0x3e2c_0404,
    0x3e2c_0414,
    0x3e2c_1424,
    0x3e34_1c04,
];

static IQ3S_GRID: [u32; 512] = [
    0x0101_0101,
    0x0101_0103,
    0x0101_0105,
    0x0101_010b,
    0x0101_010f,
    0x0101_0301,
    0x0101_0303,
    0x0101_0305,
    0x0101_0309,
    0x0101_030d,
    0x0101_0501,
    0x0101_0503,
    0x0101_050b,
    0x0101_0707,
    0x0101_0901,
    0x0101_0905,
    0x0101_090b,
    0x0101_090f,
    0x0101_0b03,
    0x0101_0b07,
    0x0101_0d01,
    0x0101_0d05,
    0x0101_0f03,
    0x0101_0f09,
    0x0101_0f0f,
    0x0103_0101,
    0x0103_0103,
    0x0103_0105,
    0x0103_0109,
    0x0103_0301,
    0x0103_0303,
    0x0103_030b,
    0x0103_0501,
    0x0103_0507,
    0x0103_050f,
    0x0103_0703,
    0x0103_070b,
    0x0103_0909,
    0x0103_0d03,
    0x0103_0d0b,
    0x0103_0f05,
    0x0105_0101,
    0x0105_0103,
    0x0105_010b,
    0x0105_010f,
    0x0105_0301,
    0x0105_0307,
    0x0105_030d,
    0x0105_0503,
    0x0105_050b,
    0x0105_0701,
    0x0105_0709,
    0x0105_0905,
    0x0105_090b,
    0x0105_090f,
    0x0105_0b03,
    0x0105_0b07,
    0x0105_0f01,
    0x0105_0f07,
    0x0107_0107,
    0x0107_0303,
    0x0107_030b,
    0x0107_0501,
    0x0107_0505,
    0x0107_0703,
    0x0107_0707,
    0x0107_070d,
    0x0107_0909,
    0x0107_0b01,
    0x0107_0b05,
    0x0107_0d0f,
    0x0107_0f03,
    0x0107_0f0b,
    0x0109_0101,
    0x0109_0307,
    0x0109_030f,
    0x0109_0503,
    0x0109_0509,
    0x0109_0705,
    0x0109_0901,
    0x0109_0907,
    0x0109_0b03,
    0x0109_0f01,
    0x010b_0105,
    0x010b_0109,
    0x010b_0501,
    0x010b_0505,
    0x010b_050d,
    0x010b_0707,
    0x010b_0903,
    0x010b_090b,
    0x010b_090f,
    0x010b_0d0d,
    0x010b_0f07,
    0x010d_010d,
    0x010d_0303,
    0x010d_0307,
    0x010d_0703,
    0x010d_0b05,
    0x010d_0f03,
    0x010f_0101,
    0x010f_0105,
    0x010f_0109,
    0x010f_0501,
    0x010f_0505,
    0x010f_050d,
    0x010f_0707,
    0x010f_0b01,
    0x010f_0b09,
    0x0301_0101,
    0x0301_0103,
    0x0301_0105,
    0x0301_0109,
    0x0301_0301,
    0x0301_0303,
    0x0301_0307,
    0x0301_030b,
    0x0301_030f,
    0x0301_0501,
    0x0301_0505,
    0x0301_0703,
    0x0301_0709,
    0x0301_070d,
    0x0301_0b09,
    0x0301_0b0d,
    0x0301_0d03,
    0x0301_0f05,
    0x0303_0101,
    0x0303_0103,
    0x0303_0107,
    0x0303_010d,
    0x0303_0301,
    0x0303_0309,
    0x0303_0503,
    0x0303_0701,
    0x0303_0707,
    0x0303_0903,
    0x0303_0b01,
    0x0303_0b05,
    0x0303_0f01,
    0x0303_0f0d,
    0x0305_0101,
    0x0305_0305,
    0x0305_030b,
    0x0305_030f,
    0x0305_0501,
    0x0305_0509,
    0x0305_0705,
    0x0305_0901,
    0x0305_0907,
    0x0305_0b0b,
    0x0305_0d01,
    0x0305_0f05,
    0x0307_0103,
    0x0307_0109,
    0x0307_010f,
    0x0307_0301,
    0x0307_0307,
    0x0307_0503,
    0x0307_050f,
    0x0307_0701,
    0x0307_0709,
    0x0307_0903,
    0x0307_0d05,
    0x0307_0f01,
    0x0309_0107,
    0x0309_010b,
    0x0309_0305,
    0x0309_0309,
    0x0309_0703,
    0x0309_0707,
    0x0309_0905,
    0x0309_090d,
    0x0309_0b01,
    0x0309_0b09,
    0x030b_0103,
    0x030b_0301,
    0x030b_0307,
    0x030b_0503,
    0x030b_0701,
    0x030b_0705,
    0x030b_0b03,
    0x030d_0501,
    0x030d_0509,
    0x030d_050f,
    0x030d_0909,
    0x030d_090d,
    0x030f_0103,
    0x030f_0107,
    0x030f_0301,
    0x030f_0305,
    0x030f_0503,
    0x030f_070b,
    0x030f_0903,
    0x030f_0d05,
    0x030f_0f01,
    0x0501_0101,
    0x0501_0103,
    0x0501_0107,
    0x0501_010b,
    0x0501_010f,
    0x0501_0301,
    0x0501_0305,
    0x0501_0309,
    0x0501_030d,
    0x0501_0503,
    0x0501_0507,
    0x0501_050f,
    0x0501_0701,
    0x0501_0705,
    0x0501_0903,
    0x0501_0907,
    0x0501_090b,
    0x0501_0b01,
    0x0501_0b05,
    0x0501_0d0f,
    0x0501_0f01,
    0x0501_0f07,
    0x0501_0f0b,
    0x0503_0101,
    0x0503_0105,
    0x0503_0301,
    0x0503_0307,
    0x0503_030f,
    0x0503_0505,
    0x0503_050b,
    0x0503_0703,
    0x0503_0709,
    0x0503_0905,
    0x0503_0b03,
    0x0505_0103,
    0x0505_0109,
    0x0505_010f,
    0x0505_0503,
    0x0505_0507,
    0x0505_0701,
    0x0505_070f,
    0x0505_0903,
    0x0505_0b07,
    0x0505_0b0f,
    0x0505_0f03,
    0x0505_0f09,
    0x0507_0101,
    0x0507_0105,
    0x0507_010b,
    0x0507_0303,
    0x0507_0505,
    0x0507_0509,
    0x0507_0703,
    0x0507_0707,
    0x0507_0905,
    0x0507_0b01,
    0x0507_0d0d,
    0x0509_0103,
    0x0509_010f,
    0x0509_0501,
    0x0509_0507,
    0x0509_0705,
    0x0509_070b,
    0x0509_0903,
    0x0509_0f05,
    0x0509_0f0b,
    0x050b_0109,
    0x050b_0303,
    0x050b_0505,
    0x050b_070f,
    0x050b_0901,
    0x050b_0b07,
    0x050b_0f01,
    0x050d_0101,
    0x050d_0105,
    0x050d_010f,
    0x050d_0503,
    0x050d_0b0b,
    0x050d_0d03,
    0x050f_010b,
    0x050f_0303,
    0x050f_050d,
    0x050f_0701,
    0x050f_0907,
    0x050f_0b01,
    0x0701_0105,
    0x0701_0303,
    0x0701_0307,
    0x0701_030b,
    0x0701_030f,
    0x0701_0505,
    0x0701_0703,
    0x0701_0707,
    0x0701_070b,
    0x0701_0905,
    0x0701_0909,
    0x0701_090f,
    0x0701_0b03,
    0x0701_0d07,
    0x0701_0f03,
    0x0703_0103,
    0x0703_0107,
    0x0703_010b,
    0x0703_0309,
    0x0703_0503,
    0x0703_0507,
    0x0703_0901,
    0x0703_0d01,
    0x0703_0f05,
    0x0703_0f0d,
    0x0705_0101,
    0x0705_0305,
    0x0705_0501,
    0x0705_0705,
    0x0705_0709,
    0x0705_0b01,
    0x0707_0103,
    0x0707_0301,
    0x0707_0309,
    0x0707_0503,
    0x0707_0507,
    0x0707_050f,
    0x0707_0701,
    0x0707_0903,
    0x0707_0907,
    0x0707_090f,
    0x0707_0b0b,
    0x0707_0f07,
    0x0709_0107,
    0x0709_0303,
    0x0709_030d,
    0x0709_0505,
    0x0709_0703,
    0x0709_0b05,
    0x0709_0d01,
    0x0709_0d09,
    0x070b_0103,
    0x070b_0301,
    0x070b_0305,
    0x070b_050b,
    0x070b_0705,
    0x070b_0909,
    0x070b_0b0d,
    0x070b_0f07,
    0x070d_030d,
    0x070d_0903,
    0x070f_0103,
    0x070f_0107,
    0x070f_0501,
    0x070f_0505,
    0x070f_070b,
    0x0901_0101,
    0x0901_0109,
    0x0901_0305,
    0x0901_0501,
    0x0901_0509,
    0x0901_050f,
    0x0901_0705,
    0x0901_0903,
    0x0901_0b01,
    0x0901_0f01,
    0x0903_0105,
    0x0903_010f,
    0x0903_0303,
    0x0903_0307,
    0x0903_0505,
    0x0903_0701,
    0x0903_070b,
    0x0903_0907,
    0x0903_0b03,
    0x0903_0b0b,
    0x0905_0103,
    0x0905_0107,
    0x0905_0301,
    0x0905_030b,
    0x0905_0503,
    0x0905_0707,
    0x0905_0901,
    0x0905_0b0f,
    0x0905_0d05,
    0x0905_0f01,
    0x0907_0109,
    0x0907_0303,
    0x0907_0307,
    0x0907_0501,
    0x0907_0505,
    0x0907_0703,
    0x0907_070b,
    0x0909_0101,
    0x0909_0105,
    0x0909_0509,
    0x0909_070f,
    0x0909_0901,
    0x0909_0f03,
    0x090b_010b,
    0x090b_010f,
    0x090b_0503,
    0x090b_0d05,
    0x090d_0307,
    0x090d_0709,
    0x090d_0d01,
    0x090f_0301,
    0x090f_030b,
    0x090f_0701,
    0x090f_0907,
    0x090f_0b03,
    0x0b01_0105,
    0x0b01_0301,
    0x0b01_0309,
    0x0b01_0505,
    0x0b01_0901,
    0x0b01_0909,
    0x0b01_090f,
    0x0b01_0b05,
    0x0b01_0d0d,
    0x0b01_0f09,
    0x0b03_0103,
    0x0b03_0107,
    0x0b03_010b,
    0x0b03_0305,
    0x0b03_0503,
    0x0b03_0705,
    0x0b03_0f05,
    0x0b05_0101,
    0x0b05_0303,
    0x0b05_0507,
    0x0b05_0701,
    0x0b05_070d,
    0x0b05_0b07,
    0x0b07_0105,
    0x0b07_010f,
    0x0b07_0301,
    0x0b07_050f,
    0x0b07_0909,
    0x0b07_0b03,
    0x0b07_0d0b,
    0x0b07_0f07,
    0x0b09_0103,
    0x0b09_0109,
    0x0b09_0501,
    0x0b09_0705,
    0x0b09_090d,
    0x0b0b_0305,
    0x0b0b_050d,
    0x0b0b_0b03,
    0x0b0b_0b07,
    0x0b0d_0905,
    0x0b0f_0105,
    0x0b0f_0109,
    0x0b0f_0505,
    0x0d01_0303,
    0x0d01_0307,
    0x0d01_030b,
    0x0d01_0703,
    0x0d01_0707,
    0x0d01_0d01,
    0x0d03_0101,
    0x0d03_0501,
    0x0d03_050f,
    0x0d03_0d09,
    0x0d05_0305,
    0x0d05_0709,
    0x0d05_0905,
    0x0d05_0b0b,
    0x0d05_0d05,
    0x0d05_0f01,
    0x0d07_0101,
    0x0d07_0309,
    0x0d07_0503,
    0x0d07_0901,
    0x0d09_050b,
    0x0d09_0907,
    0x0d09_0d05,
    0x0d0b_0101,
    0x0d0b_0107,
    0x0d0b_0709,
    0x0d0b_0d01,
    0x0d0d_010b,
    0x0d0d_0901,
    0x0d0f_0303,
    0x0d0f_0307,
    0x0f01_0101,
    0x0f01_0109,
    0x0f01_010f,
    0x0f01_0501,
    0x0f01_0505,
    0x0f01_070d,
    0x0f01_0901,
    0x0f01_0b09,
    0x0f01_0d05,
    0x0f03_0105,
    0x0f03_0303,
    0x0f03_0509,
    0x0f03_0907,
    0x0f03_090b,
    0x0f05_0103,
    0x0f05_0109,
    0x0f05_0301,
    0x0f05_030d,
    0x0f05_0503,
    0x0f05_0701,
    0x0f05_0b03,
    0x0f07_0105,
    0x0f07_0705,
    0x0f07_070b,
    0x0f07_0b07,
    0x0f09_0103,
    0x0f09_010b,
    0x0f09_0307,
    0x0f09_0501,
    0x0f09_0b01,
    0x0f0b_0505,
    0x0f0b_0905,
    0x0f0d_0105,
    0x0f0d_0703,
    0x0f0f_0101,
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, Target};

    fn wrap(op: &Value) -> Value {
        json!({"SchemaVersion": 1, "Operations": [op]})
    }

    fn floats_hex(values: &[f32]) -> String {
        hex::encode(
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn independent_codec_passes_quantize_fns_and_ptq_cuda_across_targets() -> Result<()> {
        for target in [Target::RustReference, Target::Cpp] {
            let engine = Engine::new(target, None, None, None, None, None)?;
            let quantize_report = crate::codec_assertions::run_quantize(&engine)?;
            assert_eq!(quantize_report["Id"], IDS[0]);
            assert_eq!(quantize_report["Status"], "pass", "{quantize_report}");

            let ptq_report = crate::codec_assertions::run_ptq_cuda(&engine)?;
            assert_eq!(ptq_report["Id"], IDS[1]);
            assert_eq!(ptq_report["Status"], "pass", "{ptq_report}");
        }
        for id in IDS {
            assert!(crate::reference_provider::supports(id));
        }
        let info = identity();
        assert_eq!(info["SupportedSourceIds"], json!(IDS));
        assert_eq!(info["NativeReferenceReplay"], false);
        assert_eq!(info["ProductionRuntimeCalled"], false);
        assert_eq!(info["CompleteSourceGate"], false);
        assert_eq!(info["NativeRetirementQualified"], false);
        Ok(())
    }

    #[test]
    fn independent_codec_rejects_malformed_requests() {
        let valid_f16 = json!({
            "Kind": "codec-quantize",
            "TypeId": 1,
            "Mode": "cpu",
            "Count": 1,
            "DataHex": floats_hex(&[1.0])
        });
        assert!(handles(&wrap(&valid_f16)));

        // Non-finite floats in quantize input (NaN, +Inf, -Inf)
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let req = wrap(&json!({
                "Kind": "codec-quantize",
                "TypeId": 1,
                "Mode": "cpu",
                "Count": 1,
                "DataHex": floats_hex(&[bad])
            }));
            assert!(execute(&req).is_err());
        }

        // Bad hex: uppercase, non-hex char, odd/wrong length
        for bad_hex in ["0000803F", "0000803g", "000080", "0000803f00"] {
            assert!(
                execute(&wrap(&json!({
                    "Kind": "codec-quantize",
                    "TypeId": 1,
                    "Mode": "cpu",
                    "Count": 1,
                    "DataHex": bad_hex
                })))
                .is_err()
            );
            assert!(
                execute(&wrap(&json!({
                    "Kind": "codec-dequantize",
                    "TypeId": 1,
                    "Count": 1,
                    "DataHex": bad_hex
                })))
                .is_err()
            );
        }

        // Partial block counts, zero count, oversized count, and mismatched activation block
        assert!(
            execute(&wrap(&json!({
                "Kind": "codec-quantize",
                "TypeId": 2,
                "Mode": "cpu",
                "Count": 16,
                "DataHex": floats_hex(&[0.0; 16])
            })))
            .is_err()
        );
        assert!(
            execute(&wrap(&json!({
                "Kind": "codec-quantize",
                "TypeId": 1,
                "Mode": "cpu",
                "Count": 0,
                "DataHex": ""
            })))
            .is_err()
        );
        assert!(
            execute(&wrap(&json!({
                "Kind": "codec-quantize",
                "TypeId": 1,
                "Mode": "cpu",
                "Count": 8193,
                "DataHex": ""
            })))
            .is_err()
        );
        // PQ2_0 dot activation is Q8_K (block 256), so count=128 must be rejected
        assert!(
            execute(&wrap(&json!({
                "Kind": "codec-vec-dot",
                "TypeId": 142,
                "Count": 128,
                "WeightsHex": "00".repeat(34),
                "ActivationsHex": "00".repeat(292)
            })))
            .is_err()
        );
    }

    #[test]
    fn independent_codec_rejects_unsupported_capabilities_and_envelope() {
        let valid_f16 = json!({
            "Kind": "codec-quantize",
            "TypeId": 1,
            "Mode": "cpu",
            "Count": 1,
            "DataHex": floats_hex(&[1.0])
        });

        // Unknown op, bad SchemaVersion, operation count bounds, extra/missing fields
        assert!(execute(&wrap(&json!({"Kind": "codec-unknown"}))).is_err());
        assert!(execute(&json!({"SchemaVersion": 2, "Operations": [valid_f16]})).is_err());
        assert!(execute(&json!({"SchemaVersion": 1.0, "Operations": [valid_f16]})).is_err());
        assert!(execute(&json!({"SchemaVersion": 1, "Operations": []})).is_err());
        assert!(
            execute(&json!({"SchemaVersion": 1, "Operations": vec![valid_f16.clone(); 129]}))
                .is_err()
        );
        assert!(execute(&wrap(&json!({"Kind": "codec-inventory", "Extra": 1}))).is_err());
        assert!(
            execute(&json!({"SchemaVersion": 1, "Operations": [valid_f16], "Extra": 1})).is_err()
        );

        // Unsupported mode/to_float/vec_dot capabilities
        assert!(
            execute(&wrap(&json!({
                "Kind": "codec-quantize",
                "TypeId": 0,
                "Mode": "reference",
                "Count": 1,
                "DataHex": floats_hex(&[1.0])
            })))
            .is_err()
        );
        assert!(
            execute(&wrap(&json!({
                "Kind": "codec-quantize",
                "TypeId": 18,
                "Mode": "cpu",
                "Count": 256,
                "DataHex": floats_hex(&[0.0; 256])
            })))
            .is_err()
        );
        assert!(
            execute(&wrap(&json!({
                "Kind": "codec-dequantize",
                "TypeId": 0,
                "Count": 1,
                "DataHex": floats_hex(&[1.0])
            })))
            .is_err()
        );
        assert!(
            execute(&wrap(&json!({
                "Kind": "codec-vec-dot",
                "TypeId": 15,
                "Count": 256,
                "WeightsHex": "00".repeat(292),
                "ActivationsHex": "00".repeat(292)
            })))
            .is_err()
        );
    }

    #[test]
    fn independent_codec_supports_all_iq_and_ptq_byte_domain_edge_cases() -> Result<()> {
        // Verify PTQ1_0 accepts every byte 0..=255 including > 242 without rejecting.
        for b in 0_u8..=255 {
            let mut block = vec![b; 26];
            block.extend(0x3c00_u16.to_le_bytes());
            let decoded = execute(&wrap(&json!({
                "Kind": "codec-dequantize",
                "TypeId": 143,
                "Count": 128,
                "DataHex": hex::encode(block)
            })))?;
            let hex_out = decoded["Results"][0]["DataHex"]
                .as_str()
                .ok_or_else(|| invalid("missing DataHex"))?;
            assert_eq!(hex_out.len(), 128 * 4 * 2);
        }

        // Verify reference quantize -> dequantize -> vec_dot on IQ2_S (22), IQ3_XXS (18), IQ3_S (21),
        // and dequantize + vec_dot on IQ2_XXS (16), IQ2_XS (17), IQ1_S (19), IQ1_M (29).
        let input: Vec<f32> = (0..256)
            .map(|i| 2.0_f32.mul_add(f32::from(u16::try_from(i).unwrap_or(0)).cos(), 0.1))
            .collect();
        let act_res = execute(&wrap(&json!({
            "Kind": "codec-quantize",
            "TypeId": 15,
            "Mode": "cpu",
            "Count": 256,
            "DataHex": floats_hex(&input)
        })))?;
        let act_hex = act_res["Results"][0]["DataHex"]
            .as_str()
            .ok_or_else(|| invalid("missing Q8_K hex"))?;
        for id in [18_usize, 21, 22] {
            let q = execute(&wrap(&json!({
                "Kind": "codec-quantize",
                "TypeId": id,
                "Mode": "reference",
                "Count": 256,
                "DataHex": floats_hex(&input)
            })))?;
            let q_hex = q["Results"][0]["DataHex"]
                .as_str()
                .ok_or_else(|| invalid("missing quantized hex"))?;
            let dec = execute(&wrap(&json!({
                "Kind": "codec-dequantize",
                "TypeId": id,
                "Count": 256,
                "DataHex": q_hex
            })))?;
            assert!(dec["Results"][0]["DataHex"].is_string());
            let dot = execute(&wrap(&json!({
                "Kind": "codec-vec-dot",
                "TypeId": id,
                "Count": 256,
                "WeightsHex": q_hex,
                "ActivationsHex": act_hex
            })))?;
            let dot_bytes = hex::decode(
                dot["Results"][0]["DataHex"]
                    .as_str()
                    .ok_or_else(|| invalid("missing dot hex"))?,
            )?;
            let dot_val = f32::from_le_bytes(dot_bytes.try_into().map_err(|_| invalid("len"))?);
            assert!(dot_val > 100.0);
        }
        for (id, bytes) in [(16_usize, 66_usize), (17, 74), (19, 50), (29, 56)] {
            let mut block = vec![0x15_u8; bytes];
            if id == 29 {
                block[48..56].copy_from_slice(&[0x00, 0x00, 0x00, 0xc0, 0x00, 0x00, 0x00, 0x30]);
            } else {
                block[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
            }
            let w_hex = hex::encode(block);
            let dec = execute(&wrap(&json!({
                "Kind": "codec-dequantize",
                "TypeId": id,
                "Count": 256,
                "DataHex": w_hex
            })))?;
            assert!(dec["Results"][0]["DataHex"].is_string());
            let dot = execute(&wrap(&json!({
                "Kind": "codec-vec-dot",
                "TypeId": id,
                "Count": 256,
                "WeightsHex": w_hex,
                "ActivationsHex": act_hex
            })))?;
            assert!(dot["Results"][0]["DataHex"].is_string());
        }
        Ok(())
    }
}
