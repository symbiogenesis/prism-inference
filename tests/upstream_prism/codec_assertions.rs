//! Complete safe Rust ports of the pinned quantizer and PTQ CUDA-host contracts.
//!
//! Inputs, host intrinsic transcriptions, scalar references and assertions are
//! Rust. Production codecs remain the target. The CUDA-named source itself uses
//! host stubs; this port makes no claim about GPU compilation or execution.
//! Source attribution: tools/prism/upstream/LICENSE (upstream MIT).
#![allow(
    clippy::suboptimal_flops,
    clippy::needless_range_loop,
    reason = "Preserve source scalar arithmetic association and explicit packed lane indexing."
)]

use serde_json::{Value, json};

use super::engine::Engine;
use super::{Result, invalid, result};

const QUANTIZE_ID: &str = "tests/test-quantize-fns.cpp";
const CUDA_ID: &str = "tests/test-ptq1_0-cuda-dot.cpp";
const QUANTIZE_SHA: &str = "830e197b34efc57030cef3b83449264f8b2d4db76121796ae962602cdf675d6b";
const CUDA_SHA: &str = "17d72e91495a7ffe4e726e90ebe99bf9d7b9acbe1d451a38c865352c3559092a";
// From the pinned ggml.c and ggml-cpu.c trait initializers, never an observed
// passing run. Pinning eligibility prevents a smaller SUT inventory from hiding
// unsupported production quantizers behind the original capability skip.
const ELIGIBLE: [usize; 22] = [
    1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 20, 23, 30, 34, 35, 39, 40, 41, 42, 142, 143,
];
const ACTIVE: [usize; 37] = [
    0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 34, 35, 39, 40, 41, 42, 142, 143,
];

#[derive(Default)]
struct Checks {
    count: usize,
    failed: usize,
    examples: Vec<String>,
}
impl Checks {
    fn check(&mut self, passed: bool, detail: String) {
        self.count += 1;
        if !passed {
            self.failed += 1;
            if self.examples.len() < 12 {
                self.examples.push(detail);
            }
        }
    }
    fn finish(&self, id: &str, detail: &str) -> Value {
        result(
            id,
            self.failed == 0,
            format!(
                "{detail}; checks={}; failures={}; examples={:?}",
                self.count, self.failed, self.examples
            ),
        )
    }
}

const UNSUPPORTED: &str =
    "unsupported: required production codec/CPU-dot API is missing; no reference fallback";

enum CodecResponse {
    Observations(Vec<Value>),
    Rejected(String),
}
fn fields(value: &Value, names: &[&str]) -> Result<()> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid("codec value must be an object"))?;
    if object.len() != names.len() || names.iter().any(|name| !object.contains_key(*name)) {
        return Err(invalid("codec object fields mismatch"));
    }
    Ok(())
}
fn array(value: &Value) -> Result<&[Value]> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| invalid("codec value must be an array"))
}
fn number(value: &Value) -> Result<usize> {
    usize::try_from(
        value
            .as_u64()
            .ok_or_else(|| invalid("codec integer must be unsigned"))?,
    )
    .map_err(Into::into)
}
fn boolean(value: &Value) -> Result<bool> {
    value
        .as_bool()
        .ok_or_else(|| invalid("codec capability must be boolean"))
}
fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8] = b"0123456789abcdef";
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        text.push(char::from(DIGITS[usize::from(byte >> 4)]));
        text.push(char::from(DIGITS[usize::from(byte & 15)]));
    }
    text
}
fn unhex(value: &Value, expected: usize) -> Result<Vec<u8>> {
    let text = value
        .as_str()
        .ok_or_else(|| invalid("codec DataHex must be text"))?;
    if expected > 1024 * 1024
        || text.len() != expected * 2
        || !text
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(invalid("codec DataHex extent or digits mismatch"));
    }
    text.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| Ok(u8::from_str_radix(std::str::from_utf8(pair)?, 16)?))
        .collect()
}
fn raw(value: &Value, expected: usize) -> Result<Vec<u8>> {
    fields(value, &["DataHex"])?;
    unhex(&value["DataHex"], expected)
}
fn rejection(value: &Value) -> Result<Option<&str>> {
    if value.get("Error").is_none() {
        return Ok(None);
    }
    fields(value, &["Error"])?;
    let message = value["Error"]
        .as_str()
        .ok_or_else(|| invalid("codec Error must be text"))?;
    if message.is_empty() || message.len() > 8192 {
        return Err(invalid("codec Error outside text bound"));
    }
    Ok(Some(message))
}
fn floats(value: &Value, count: usize) -> Result<Vec<f32>> {
    raw(value, count * 4)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| {
            let number = f32::from_le_bytes(*bytes);
            if !number.is_finite() {
                return Err(invalid("nonfinite production codec output"));
            }
            Ok(number)
        })
        .collect()
}
fn float_hex(values: &[f32]) -> String {
    hex(&values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>())
}
fn operation_rejections(
    name: &str,
    operations: &[Value],
    observations: &[Value],
) -> Result<Option<String>> {
    let mut detail = None;
    for (index, (operation, observation)) in operations.iter().zip(observations).enumerate() {
        if let Some(error) = rejection(observation)? {
            detail.get_or_insert_with(|| {
                format!(
                    "{name} operation={index} kind={} type={} mode={}: production codec rejected: {error}",
                    operation["Kind"], operation["TypeId"], operation["Mode"]
                )
            });
        }
    }
    Ok(detail)
}

fn exchange(engine: &Engine, name: &str, operations: &[Value]) -> Result<CodecResponse> {
    let mut results = Vec::with_capacity(operations.len());
    for (batch, operations) in operations.chunks(128).enumerate() {
        let request = json!({"SchemaVersion":1,"Operations":operations});
        let Some(response) = engine.execute_codec(&format!("{name}-{batch}"), &request)? else {
            return Ok(CodecResponse::Rejected(UNSUPPORTED.to_owned()));
        };
        fields(&response, &["SchemaVersion", "Results"])?;
        if response["SchemaVersion"] != 1 {
            return Err(invalid("codec response version mismatch"));
        }
        let observations = array(&response["Results"])?;
        if observations.len() != operations.len() {
            return Err(invalid("codec result count mismatch"));
        }
        // A valid target rejection is a failed source contract. It must not be
        // parsed as DataHex and promoted to an infrastructure/schema failure.
        // Validate every Error in the batch, retaining malformed responses as
        // infrastructure failures even after a preceding legitimate rejection.
        if let Some(detail) = operation_rejections(name, operations, observations)? {
            return Ok(CodecResponse::Rejected(detail));
        }
        results.extend_from_slice(observations);
    }
    Ok(CodecResponse::Observations(results))
}
fn small(value: usize) -> Result<f32> {
    Ok(f32::from(u16::try_from(value)?))
}
fn half(bits: u16) -> f32 {
    let exponent = i32::from((bits >> 10) & 31);
    let fraction = f32::from(bits & 1023);
    let value = if exponent == 0 {
        fraction * 2.0_f32.powi(-24)
    } else {
        (1.0 + fraction / 1024.0) * 2.0_f32.powi(exponent - 15)
    };
    if bits & 0x8000 == 0 { value } else { -value }
}

#[allow(
    clippy::struct_excessive_bools,
    reason = "Mirror independent production function-pointer capabilities."
)]
struct Traits {
    block: usize,
    size: usize,
    from_cpu: bool,
    from_reference: bool,
    to_float: bool,
    vec_dot: bool,
    activation: usize,
}
impl Traits {
    fn extent(&self, count: usize) -> Result<usize> {
        if self.block == 0 || !count.is_multiple_of(self.block) {
            return Err(invalid(
                "codec source count must contain whole production blocks",
            ));
        }
        count
            .checked_div(self.block)
            .and_then(|n| n.checked_mul(self.size))
            .filter(|n| *n <= 1024 * 1024)
            .ok_or_else(|| invalid("codec extent exceeds bound"))
    }
}
fn inventory(response: &Value) -> Result<Vec<Traits>> {
    fields(response, &["Types"])?;
    let values = array(&response["Types"])?;
    if values.len() != 144 {
        return Err(invalid("pinned codec inventory must contain all 144 IDs"));
    }
    let mut traits = Vec::with_capacity(values.len());
    for (id, value) in values.iter().enumerate() {
        fields(
            value,
            &[
                "TypeId",
                "Name",
                "BlockSize",
                "TypeSize",
                "HasFromFloatCpu",
                "HasFromFloatRef",
                "HasToFloat",
                "HasVecDot",
                "VecDotType",
            ],
        )?;
        if number(&value["TypeId"])? != id
            || (!value["Name"].is_null() && !value["Name"].is_string())
        {
            return Err(invalid("codec inventory ID/name mismatch"));
        }
        let item = Traits {
            block: number(&value["BlockSize"])?,
            size: number(&value["TypeSize"])?,
            from_cpu: boolean(&value["HasFromFloatCpu"])?,
            from_reference: boolean(&value["HasFromFloatRef"])?,
            to_float: boolean(&value["HasToFloat"])?,
            vec_dot: boolean(&value["HasVecDot"])?,
            activation: number(&value["VecDotType"])?,
        };
        if item.block > 4096
            || item.size > 4096
            || item.activation >= 144
            || (item.block > 0) != ACTIVE.contains(&id)
            || (item.block > 0 && item.from_cpu && item.to_float) != ELIGIBLE.contains(&id)
        {
            return Err(invalid(format!(
                "production codec capabilities differ from pinned source at ID {id}"
            )));
        }
        if ELIGIBLE.contains(&id) && (!item.from_reference || !item.vec_dot) {
            return Err(invalid(
                "eligible source codec lacks reference quantizer or CPU dot",
            ));
        }
        traits.push(item);
    }
    if !traits[0].vec_dot || traits[0].activation != 0 {
        return Err(invalid("production F32 tail-dot API is unavailable"));
    }
    Ok(traits)
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::as_conversions,
    reason = "The C++ source narrows the double literal sum back to float after cosf."
)]
fn generate_data(offset: f32, count: usize) -> Result<Vec<f32>> {
    (0..count)
        .map(|i| Ok((0.1_f64 + f64::from(2.0 * (small(i)? + offset).cos())) as f32))
        .collect()
}
#[allow(
    clippy::cast_possible_truncation,
    clippy::as_conversions,
    reason = "Preserve C++ double accumulation followed by float return/sqrtf conversion."
)]
fn source_rmse(left: &[f32], right: &[f32]) -> Result<f32> {
    if left.len() != right.len() || left.is_empty() {
        return Err(invalid("RMSE extents mismatch"));
    }
    let sum: f64 = left
        .iter()
        .zip(right)
        .map(|(a, b)| {
            let diff = f64::from(a - b);
            diff * diff
        })
        .sum();
    Ok((sum as f32).sqrt() / small(left.len())?)
}
#[allow(
    clippy::cast_possible_truncation,
    clippy::as_conversions,
    reason = "Original dot_product multiplies in float, sums in double, and returns float."
)]
fn source_dot(left: &[f32], right: &[f32]) -> Result<f32> {
    if left.len() != right.len() {
        return Err(invalid("dot extents mismatch"));
    }
    Ok(left
        .iter()
        .zip(right)
        .map(|(a, b)| f64::from(a * b))
        .sum::<f64>() as f32)
}
const fn quantization_limit(id: usize) -> f32 {
    match id {
        41 => 0.025,
        34 | 35 | 42 | 142 | 143 => 0.01,
        10 | 22 => 0.0075,
        11 | 21 => 0.0040,
        18 => 0.0050,
        40 => 0.0030,
        _ => 0.002,
    }
}
const fn dot_limit(id: usize) -> f32 {
    match id {
        10 | 16 | 17 | 18 | 21 | 22 => 0.04,
        41 => 0.40,
        34 | 35 | 42 | 142 | 143 => 0.15,
        40 => 0.03,
        _ => 0.02,
    }
}
fn quantize(id: usize, mode: &str, values: &[f32]) -> Value {
    json!({"Kind":"codec-quantize","TypeId":id,"Mode":mode,"Count":values.len(),"DataHex":float_hex(values)})
}
fn decode(id: usize, count: usize, bytes: &[u8]) -> Value {
    json!({"Kind":"codec-dequantize","TypeId":id,"Count":count,"DataHex":hex(bytes)})
}
fn dot(id: usize, count: usize, weights: &[u8], activations: &[u8]) -> Value {
    json!({"Kind":"codec-vec-dot","TypeId":id,"Count":count,"WeightsHex":hex(weights),"ActivationsHex":hex(activations)})
}

pub(super) fn run_quantize(engine: &Engine) -> Result<Value> {
    let metadata = match exchange(
        engine,
        "codec-inventory",
        &[json!({"Kind":"codec-inventory"})],
    )? {
        CodecResponse::Observations(values) => values,
        CodecResponse::Rejected(detail) => return Ok(result(QUANTIZE_ID, false, detail)),
    };
    let traits = inventory(&metadata[0])?;
    let mut checks = Checks::default();
    let tails = [
        1, 2, 3, 5, 7, 8, 15, 16, 17, 31, 33, 63, 67, 127, 129, 193, 255, 1023,
    ];
    let mut operations = Vec::new();
    for count in tails {
        let left = generate_data(0.0, count)?;
        let right = generate_data(1.0, count)?;
        operations.push(dot(0, count, &raw_floats(&left), &raw_floats(&right)));
    }
    let data = generate_data(0.0, 4096)?;
    let data2 = generate_data(1.0, 4096)?;
    for id in ELIGIBLE {
        // Preserve each source call, including the repeated CPU quantization
        // in reference_quantization_error and the third weight quantization.
        operations.push(quantize(id, "cpu", &data));
        operations.push(quantize(id, "cpu", &data));
        operations.push(quantize(id, "reference", &data));
        operations.push(quantize(id, "cpu", &data));
        let activation = traits[id].activation;
        if !traits[activation].from_cpu {
            return Err(invalid("dot activation quantizer is unavailable"));
        }
        operations.push(quantize(activation, "cpu", &data2));
    }
    let encoded = match exchange(engine, "codec-quantize", &operations)? {
        CodecResponse::Observations(values) => values,
        CodecResponse::Rejected(detail) => return Ok(result(QUANTIZE_ID, false, detail)),
    };
    for (index, count) in tails.into_iter().enumerate() {
        let target = floats(&encoded[index], 1)?[0];
        let reference = source_dot(&generate_data(0.0, count)?, &generate_data(1.0, count)?)?;
        let error = (target - reference).abs() / small(count)?;
        checks.check(error < 0.0001, format!("F32 tail n={count}: error={error}"));
    }
    operations.clear();
    for (index, id) in ELIGIBLE.into_iter().enumerate() {
        let entries = &encoded[tails.len() + index * 5..tails.len() + index * 5 + 5];
        let packed = entries[..4]
            .iter()
            .map(|value| raw(value, traits[id].extent(4096)?))
            .collect::<Result<Vec<_>>>()?;
        let activations = raw(&entries[4], traits[traits[id].activation].extent(4096)?)?;
        for bytes in &packed[..3] {
            operations.push(decode(id, 4096, bytes));
        }
        operations.push(dot(id, 4096, &packed[3], &activations));
    }
    let decoded = match exchange(engine, "codec-errors", &operations)? {
        CodecResponse::Observations(values) => values,
        CodecResponse::Rejected(detail) => return Ok(result(QUANTIZE_ID, false, detail)),
    };
    for (index, id) in ELIGIBLE.into_iter().enumerate() {
        let values = &decoded[index * 4..index * 4 + 4];
        let total_error = source_rmse(&data, &floats(&values[0], 4096)?)?;
        checks.check(
            total_error < quantization_limit(id),
            format!("type={id}: total error={total_error}"),
        );
        let reference_error = source_rmse(&floats(&values[1], 4096)?, &floats(&values[2], 4096)?)?;
        checks.check(
            reference_error < 0.0001,
            format!("type={id}: reference error={reference_error}"),
        );
        let dot_error = (floats(&values[3], 1)?[0] - source_dot(&data, &data2)?).abs() / 4096.0;
        checks.check(
            dot_error < dot_limit(id),
            format!("type={id}: dot error={dot_error}"),
        );
    }
    if let Some(detail) = packed_patterns(engine, &mut checks)? {
        return Ok(result(QUANTIZE_ID, false, detail));
    }
    if checks.count != 1108 {
        return Err(invalid("quantize source assertion count mismatch"));
    }
    Ok(checks.finish(QUANTIZE_ID, &format!("source={QUANTIZE_SHA}; 18 F32 tails, 22 eligible types x3, 1024 exact packed dots; original skips:107 deprecated,15 active without both CPU quantizer/decoder")))
}

fn raw_floats(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

struct Pattern {
    id: usize,
    blocks: usize,
    seed_byte: usize,
    weights: Vec<u8>,
    activations: Vec<u8>,
    expanded: Vec<f32>,
}
fn pattern(id: usize, blocks: usize, value: usize) -> Result<Pattern> {
    let mut weights = Vec::new();
    let mut activations = Vec::new();
    let mut expanded = Vec::new();
    for i in 0..blocks {
        // All scales used here are exact half-precision powers of two.
        let bits = match i {
            0 => 0x3400_u16,
            1 => 0x3800,
            2 => 0x3a00,
            3 => 0x3c00,
            _ => return Err(invalid("packed source block count out of range")),
        };
        if id == 142 {
            weights.extend(bits.to_le_bytes());
            for j in 0..32 {
                weights.push(u8::try_from((value + 17 * j + i) % 256)?);
            }
        } else {
            for j in 0..24 {
                weights.push(u8::try_from((value + 17 * j + i) % 256)?);
            }
            for j in 0..2 {
                weights.push(u8::try_from((value + 37 * j + i) % 256)?);
            }
            weights.extend(bits.to_le_bytes());
        }
    }
    if id == 142 {
        // The source allocates nb/2+1 Q8_K blocks. The final spare block is
        // deliberately not sent to the production API, which consumes n only.
        for i in 0..=blocks / 2 {
            let scale = 0.125 * small(i % 4 + 1)?;
            let mut block = scale.to_le_bytes().to_vec();
            let mut quantized = Vec::new();
            for j in 0..256 {
                let q = i8::try_from(i16::try_from((value + 13 * j + i) % 256)? - 128)?;
                block.extend(q.to_le_bytes());
                quantized.push(q);
                if i < blocks / 2 {
                    expanded.push(scale * f32::from(q));
                }
            }
            for group in quantized.as_chunks::<16>().0 {
                let sum: i16 = group.iter().map(|q| i16::from(*q)).sum();
                block.extend(sum.to_le_bytes());
            }
            if i < blocks / 2 {
                activations.extend(block);
            }
        }
    } else {
        for i in 0..blocks * 4 {
            let bits = [0x3000_u16, 0x3400, 0x3600, 0x3800][i % 4];
            activations.extend(bits.to_le_bytes());
            for j in 0..32 {
                let q = i8::try_from(i16::try_from((value + 13 * j + i) % 256)? - 128)?;
                activations.extend(q.to_le_bytes());
                expanded.push(half(bits) * f32::from(q));
            }
        }
    }
    Ok(Pattern {
        id,
        blocks,
        seed_byte: value,
        weights,
        activations,
        expanded,
    })
}
#[allow(
    clippy::float_cmp,
    reason = "The original packed-dot contract requires exact equality using power-of-two scales."
)]
fn packed_patterns(engine: &Engine, checks: &mut Checks) -> Result<Option<String>> {
    let mut cases = Vec::new();
    for id in [142, 143] {
        for blocks in if id == 142 { [2, 4] } else { [1, 3] } {
            for value in 0..256 {
                cases.push(pattern(id, blocks, value)?);
            }
        }
    }
    for (batch, cases) in cases.chunks(32).enumerate() {
        let mut operations = Vec::new();
        for case in cases {
            operations.push(decode(case.id, case.blocks * 128, &case.weights));
            if case.id == 143 {
                operations.push(decode(8, case.blocks * 128, &case.activations));
            }
            operations.push(dot(
                case.id,
                case.blocks * 128,
                &case.weights,
                &case.activations,
            ));
        }
        let observations = match exchange(engine, &format!("codec-patterns-{batch}"), &operations)?
        {
            CodecResponse::Observations(values) => values,
            CodecResponse::Rejected(detail) => return Ok(Some(detail)),
        };
        let mut index = 0;
        for case in cases {
            let x = floats(&observations[index], case.blocks * 128)?;
            index += 1;
            let y = if case.id == 143 {
                let values = floats(&observations[index], case.blocks * 128)?;
                index += 1;
                values
            } else {
                case.expanded.clone()
            };
            let reference = source_dot(&x, &y)?;
            let actual = floats(&observations[index], 1)?[0];
            index += 1;
            checks.check(
                actual == reference,
                format!(
                    "type={} blocks={} pattern={}: ref={reference}, got={actual}",
                    case.id, case.blocks, case.seed_byte
                ),
            );
        }
    }
    Ok(None)
}

// Host-only C++ CUDA-intrinsic stubs are transcribed below in safe integer Rust.
// They do not implement the SUT. Every generated block separately targets the
// production decoder and CPU dot through the codec adapter.
#[derive(Clone)]
struct Q8Host {
    scale: f32,
    values: [i8; 32],
}
struct HostCase {
    packed: [u8; 26],
    scale: f32,
    activations: [Q8Host; 4],
}
struct Lcg(u32);
impl Lcg {
    const fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        (self.0 >> 16) & 0xffff
    }
}
fn reference_trits(packed: &[u8; 26]) -> [i8; 128] {
    let powers = [1_u8, 3, 9, 27, 81];
    let mut output = [0_i8; 128];
    let mut cursor = 0;
    let mut start = 0;
    for width in [32, 16, 8] {
        while start + width <= 24 {
            for power in powers {
                for m in 0..width {
                    let q = packed[start + m].wrapping_mul(power);
                    // The high byte of 3*q is always 0,1,2.
                    output[cursor] = i8::from_ne_bytes([(u16::from(q) * 3).to_le_bytes()[1]]) - 1;
                    cursor += 1;
                }
            }
            start += width;
        }
    }
    for power in &powers[..4] {
        for h in 0..2 {
            let q = packed[24 + h].wrapping_mul(*power);
            output[cursor] = i8::from_ne_bytes([(u16::from(q) * 3).to_le_bytes()[1]]) - 1;
            cursor += 1;
        }
    }
    output
}
fn scalar_trit(packed: &[u8; 26], element: usize) -> i32 {
    let (byte, digit) = if element < 80 {
        (packed[element & 15], element >> 4)
    } else if element < 120 {
        let t = element - 80;
        (packed[16 + (t & 7)], t >> 3)
    } else {
        let t = element - 120;
        (packed[24 + (t & 1)], t >> 1)
    };
    let mut value = u32::from(byte);
    for i in 0..4 {
        if i < digit {
            value = (value * 3) & 255;
        }
    }
    i32::from((value * 3).to_le_bytes()[1]) - 1
}
fn dp4a(left: u32, right: u32, initial: i32) -> i32 {
    left.to_le_bytes()
        .into_iter()
        .zip(right.to_le_bytes())
        .fold(initial, |sum, (a, b)| {
            sum + i32::from(i8::from_ne_bytes([a])) * i32::from(i8::from_ne_bytes([b]))
        })
}
const fn word(bytes: &[i8; 32], index: usize) -> u32 {
    let start = index * 4;
    u32::from_le_bytes([
        bytes[start].to_ne_bytes()[0],
        bytes[start + 1].to_ne_bytes()[0],
        bytes[start + 2].to_ne_bytes()[0],
        bytes[start + 3].to_ne_bytes()[0],
    ])
}
fn hip_perm(a: u32, b: u32, selectors: u32) -> u32 {
    let mut result = 0;
    for i in 0..4 {
        let selector = (selectors >> (8 * i)) & 7;
        let byte = if selector & 4 == 0 {
            b >> (8 * (selector & 3))
        } else {
            a >> (8 * (selector & 3))
        };
        result |= (byte & 255) << (8 * i);
    }
    result
}
fn scalar_integer(case: &HostCase, chunk: usize) -> i32 {
    let mut sum = 0;
    for j in 0..8 {
        let base = chunk * 32 + j * 4;
        let mut bytes = [0; 4];
        for lane in 0..4 {
            bytes[lane] = scalar_trit(&case.packed, base + lane).to_le_bytes()[0];
        }
        sum = dp4a(
            word(&case.activations[chunk].values, j),
            u32::from_le_bytes(bytes),
            sum,
        );
    }
    sum
}
fn vector_integers(case: &HostCase) -> [i32; 4] {
    let mut sums = [0; 4];
    let mut unsigned_sums = [0; 4];
    for (start_word, words, base, width) in [(0, 4, 0, 16), (4, 2, 80, 8)] {
        for w in 0..words {
            let offset = (start_word + w) * 4;
            let packed = u32::from_le_bytes([
                case.packed[offset],
                case.packed[offset + 1],
                case.packed[offset + 2],
                case.packed[offset + 3],
            ]);
            let mut low = hip_perm(0, packed, 0x0c01_0c00);
            let mut high = hip_perm(0, packed, 0x0c03_0c02);
            for t in 0..5 {
                let low_digits = low.wrapping_mul(3);
                let high_digits = high.wrapping_mul(3);
                low = low_digits & 0x00ff_00ff;
                high = high_digits & 0x00ff_00ff;
                let element = base + t * width + 4 * w;
                let activation = word(&case.activations[element >> 5].values, (element & 31) >> 2);
                let digits = hip_perm(high_digits, low_digits, 0x0705_0301);
                sums[element >> 5] += dp4a(digits, activation, 0);
                unsigned_sums[element >> 5] += dp4a(0x0101_0101, activation, 0);
            }
        }
    }
    for h in 0..2 {
        let mut value = u32::from(case.packed[24 + h]);
        for t in 0..4 {
            let product = value * 3;
            let q = i32::from(product.to_le_bytes()[1]);
            value = product & 255;
            let element = 120 + t * 2 + h;
            let activation = i32::from(case.activations[element >> 5].values[element & 31]);
            sums[element >> 5] += q * activation;
            unsigned_sums[element >> 5] += activation;
        }
    }
    std::array::from_fn(|i| sums[i] - unsigned_sums[i])
}
fn reference_sums(case: &HostCase, trits: &[i8; 128]) -> [i32; 4] {
    std::array::from_fn(|chunk| {
        (0..32)
            .map(|i| {
                i32::from(trits[chunk * 32 + i]) * i32::from(case.activations[chunk].values[i])
            })
            .sum()
    })
}
#[allow(
    clippy::cast_precision_loss,
    clippy::as_conversions,
    reason = "The source explicitly converts bounded integer dot accumulators to float."
)]
const fn integer_float(value: i32) -> f32 {
    value as f32
}
#[allow(
    clippy::cast_possible_truncation,
    clippy::as_conversions,
    reason = "Preserve llround after dividing the original float kernel result by its scales."
)]
fn rounded_integer(value: f64) -> Result<i32> {
    let rounded = value.round();
    if !rounded.is_finite() || rounded.abs() > 32768.0 {
        return Err(invalid("host dot integer recovery outside bound"));
    }
    Ok(rounded as i32)
}
fn trial(rng: &mut Lcg) -> Result<HostCase> {
    let mut packed = [0; 26];
    for byte in &mut packed {
        *byte = rng.next().to_le_bytes()[0];
    }
    let scale = 0.01 + f32::from(u16::try_from(rng.next() % 1000)?) / 50000.0;
    let mut activations = std::array::from_fn(|_| Q8Host {
        scale: 0.0,
        values: [0; 32],
    });
    for block in &mut activations {
        block.scale = 0.005 + f32::from(u16::try_from(rng.next() % 1000)?) / 80000.0;
        for value in &mut block.values {
            *value = i8::try_from(i16::try_from(rng.next() % 255)? - 127)?;
        }
    }
    Ok(HostCase {
        packed,
        scale,
        activations,
    })
}
fn exhaustive(position: usize, value: usize) -> Result<HostCase> {
    let mut packed = [0xa5; 26];
    if position < 6 {
        packed[position * 4..position * 4 + 4].fill(u8::try_from(value)?);
    } else {
        packed[24 + position - 6] = u8::try_from(value)?;
    }
    let mut activations = std::array::from_fn(|_| Q8Host {
        scale: 0.02,
        values: [0; 32],
    });
    for (b, block) in activations.iter_mut().enumerate() {
        for (i, target) in block.values.iter_mut().enumerate() {
            *target = i8::try_from(i16::try_from((i * 7 + b * 13 + value) % 251)? - 125)?;
        }
    }
    Ok(HostCase {
        packed,
        scale: 0.03,
        activations,
    })
}
// Supplement the unchanged source schedule with the former packed_codec
// regression's isolated byte domain and original signed activation bits. The
// source exhaustive groups change four lanes at once, so keep these stronger
// independent checks in the upstream port.
fn native_byte_domain(position: usize, value: u8) -> HostCase {
    let mut packed = [0; 26];
    packed[position] = value;
    HostCase {
        packed,
        scale: 1.0,
        activations: std::array::from_fn(|group| Q8Host {
            scale: 1.0,
            values: std::array::from_fn(|element| {
                let index = group * 32 + element;
                i8::from_ne_bytes([index.to_le_bytes()[0].wrapping_mul(197).wrapping_add(13)])
            }),
        }),
    }
}

fn production_host_ops(case: &HostCase) -> [Value; 2] {
    let mut weights = case.packed.to_vec();
    weights.extend(0x3c00_u16.to_le_bytes());
    let mut activations = Vec::with_capacity(136);
    for block in &case.activations {
        activations.extend(0x3c00_u16.to_le_bytes());
        activations.extend(block.values.iter().flat_map(|value| value.to_le_bytes()));
    }
    [
        decode(143, 128, &weights),
        dot(143, 128, &weights, &activations),
    ]
}

#[derive(Default)]
struct Metrics {
    worst_relative: f64,
    worst_scaled: f64,
}
fn check_trial(
    case: &HostCase,
    index: usize,
    checks: &mut Checks,
    metrics: &mut Metrics,
) -> Result<()> {
    let trits = reference_trits(&case.packed);
    let references = reference_sums(case, &trits);
    let vector = vector_integers(case);
    let mut scalar_total = 0.0_f64;
    for chunk in 0..4 {
        let actual =
            case.scale * case.activations[chunk].scale * integer_float(scalar_integer(case, chunk));
        let recovered = rounded_integer(
            f64::from(actual) / (f64::from(case.scale) * f64::from(case.activations[chunk].scale)),
        )?;
        checks.check(
            recovered == references[chunk],
            format!("trial={index} chunk={chunk}: scalar integer mismatch"),
        );
        scalar_total += f64::from(actual);
    }
    let mut reference_total = 0.0_f64;
    let mut magnitude = 0.0_f64;
    for chunk in 0..4 {
        for i in 0..32 {
            let decoded = f32::from(trits[chunk * 32 + i]) * case.scale;
            reference_total += f64::from(decoded)
                * (f64::from(case.activations[chunk].values[i])
                    * f64::from(case.activations[chunk].scale));
            magnitude += (f64::from(decoded)
                * f64::from(case.activations[chunk].values[i])
                * f64::from(case.activations[chunk].scale))
            .abs();
        }
    }
    let denominator = if reference_total.abs() > 1e-9 {
        reference_total.abs()
    } else {
        1.0
    };
    metrics.worst_relative = metrics
        .worst_relative
        .max((scalar_total - reference_total).abs() / denominator);
    if magnitude > 0.0 {
        metrics.worst_scaled = metrics
            .worst_scaled
            .max((scalar_total - reference_total).abs() / magnitude);
    }
    let mut vector_total = 0.0_f32;
    for chunk in 0..4 {
        checks.check(
            vector[chunk] == references[chunk],
            format!("trial={index} chunk={chunk}: HIP vector integer mismatch"),
        );
        vector_total += case.activations[chunk].scale * integer_float(vector[chunk]);
    }
    vector_total *= case.scale;
    let scalar_float = narrow_float(scalar_total);
    checks.check(
        (vector_total - scalar_float).abs() <= 1e-6 * scalar_float.abs().max(1.0),
        format!("trial={index}: HIP float association mismatch"),
    );
    Ok(())
}
#[allow(
    clippy::cast_possible_truncation,
    clippy::as_conversions,
    reason = "The original HIP float comparison explicitly casts the double scalar total to float."
)]
const fn narrow_float(value: f64) -> f32 {
    value as f32
}
#[allow(
    clippy::float_cmp,
    reason = "Unit-scale trits and bounded integer dot sums are exactly representable in f32."
)]
fn check_production(
    case: &HostCase,
    observation: &[Value],
    index: usize,
    checks: &mut Checks,
) -> Result<()> {
    let trits = reference_trits(&case.packed);
    if let Some(error) = rejection(&observation[0])? {
        checks.check(
            false,
            format!("case={index}: production PTQ decoder rejected: {error}"),
        );
    } else {
        let decoded = floats(&observation[0], 128)?;
        checks.check(
            decoded
                .iter()
                .zip(trits)
                .all(|(value, trit)| *value == f32::from(trit)),
            format!("case={index}: production PTQ decoder mismatch"),
        );
    }
    let reference: i32 = reference_sums(case, &trits).iter().sum();
    if let Some(error) = rejection(&observation[1])? {
        checks.check(
            false,
            format!("case={index}: production PTQ CPU dot rejected: {error}"),
        );
    } else {
        checks.check(
            floats(&observation[1], 1)?[0] == integer_float(reference),
            format!("case={index}: production PTQ CPU dot mismatch"),
        );
    }
    Ok(())
}

pub(super) fn run_ptq_cuda(engine: &Engine) -> Result<Value> {
    let mut original = Checks::default();
    let mut production = Checks::default();
    let mut metrics = Metrics::default();
    let mut rng = Lcg(99);
    let mut cases = Vec::with_capacity(7048);
    for index in 0..5000 {
        let case = trial(&mut rng)?;
        check_trial(&case, index, &mut original, &mut metrics)?;
        cases.push(case);
    }
    for position in 0..8 {
        for value in 0..256 {
            let case = exhaustive(position, value)?;
            let references = reference_sums(&case, &reference_trits(&case.packed));
            let actual = vector_integers(&case);
            for chunk in 0..4 {
                original.check(actual[chunk] == references[chunk], format!("exhaustive position={position} value={value} chunk={chunk}: HIP integer mismatch"));
            }
            cases.push(case);
        }
    }
    for (batch, cases) in cases.chunks(64).enumerate() {
        let operations: Vec<Value> = cases.iter().flat_map(production_host_ops).collect();
        let observations = match exchange(engine, &format!("codec-ptq-host-{batch}"), &operations)?
        {
            CodecResponse::Observations(values) => values,
            CodecResponse::Rejected(detail) => return Ok(result(CUDA_ID, false, detail)),
        };
        for (index, case) in cases.iter().enumerate() {
            check_production(
                case,
                &observations[index * 2..index * 2 + 2],
                batch * 64 + index,
                &mut production,
            )?;
        }
    }
    if original.count != 53192 || production.count != 14096 {
        return Err(invalid("PTQ CUDA-host source assertion count mismatch"));
    }
    let mut supplemental = Checks::default();
    for position in 0..26 {
        let cases: Vec<_> = (0..=255)
            .map(|value| native_byte_domain(position, value))
            .collect();
        let operations: Vec<Value> = cases.iter().flat_map(production_host_ops).collect();
        let observations = match exchange(
            engine,
            &format!("codec-ptq-byte-domain-{position}"),
            &operations,
        )? {
            CodecResponse::Observations(values) => values,
            CodecResponse::Rejected(detail) => return Ok(result(CUDA_ID, false, detail)),
        };
        for (value, case) in cases.iter().enumerate() {
            check_production(
                case,
                &observations[value * 2..value * 2 + 2],
                position * 256 + value,
                &mut supplemental,
            )?;
        }
    }
    if supplemental.count != 13312 {
        return Err(invalid(
            "PTQ supplemental byte-domain assertion count mismatch",
        ));
    }
    Ok(result(
        CUDA_ID,
        original.failed == 0 && production.failed == 0 && supplemental.failed == 0,
        format!(
            "source={CUDA_SHA}; host transcription, no GPU execution; 5000 trials:20000 scalar integer,20000 HIP integer,5000 HIP float; exhaustive:8192 HIP integer; original failures={}; production decode/CPU dot checks=14096 failures={}; supplemental 6656 isolated lane bytes with signed i8 activations: decode/CPU dot checks=13312 failures={}; worst relative={:.3e}, worst scaled={:.3e}; examples={:?}/{:?}/{:?}",
            original.failed,
            production.failed,
            supplemental.failed,
            metrics.worst_relative,
            metrics.worst_scaled,
            original.examples,
            production.examples,
            supplemental.examples
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn available_inventory_and_unavailable_cpu_quantizer_report_failed_source() -> Result<()> {
        let engine = Engine::new(
            super::super::engine::Target::Rust,
            None,
            None,
            None,
            None,
            None,
        )?;
        let report = run_quantize(&engine)?;
        assert_eq!(report["Id"], QUANTIZE_ID);
        assert_eq!(report["Status"], "fail");
        let detail = report["Detail"]
            .as_str()
            .ok_or_else(|| invalid("codec source report detail missing"))?;
        assert!(detail.contains("codec-quantize operation=18"));
        assert!(detail.contains("type=1 mode=\"cpu\""));
        assert!(detail.contains("requested Rust codec operation is unavailable"));
        Ok(())
    }

    #[test]
    fn target_rejections_retain_operation_context_and_malformed_errors_reject() -> Result<()> {
        let operations = [quantize(2, "cpu", &[0.0; 32]), decode(2, 32, &[0; 18])];
        let observations = [
            json!({"Error":"CPU quantizer unavailable"}),
            json!({"Error":"decoder unavailable"}),
        ];
        let detail = operation_rejections("focused", &operations, &observations)?
            .ok_or_else(|| invalid("codec target rejection was lost"))?;
        assert!(detail.contains("operation=0 kind=\"codec-quantize\" type=2 mode=\"cpu\""));
        assert!(detail.contains("CPU quantizer unavailable"));
        for malformed in [
            json!({"Error":""}),
            json!({"Error":5}),
            json!({"Error":"unavailable","DataHex":"00"}),
            json!({"Error":"x".repeat(8193)}),
        ] {
            assert!(
                operation_rejections(
                    "focused",
                    &operations,
                    &[observations[0].clone(), malformed],
                )
                .is_err()
            );
        }
        assert!(
            operation_rejections(
                "focused",
                &operations[..1],
                &[json!({"DataHex":"00000000"})],
            )?
            .is_none()
        );
        Ok(())
    }

    #[test]
    fn rejects_malformed_codec_and_detects_host_lane_error() -> Result<()> {
        assert!(floats(&json!({"DataHex":"00000000","Extra":1}), 1).is_err());
        assert!(floats(&json!({"DataHex":"0000807f"}), 1).is_err());
        assert!(floats(&json!({"DataHex":"000000"}), 1).is_err());
        let case = exhaustive(6, 255)?;
        let trits = reference_trits(&case.packed);
        let mut observations = vec![
            json!({"DataHex":float_hex(&trits.map(f32::from))}),
            json!({"DataHex":float_hex(&[integer_float(reference_sums(&case,&trits).iter().sum())])}),
        ];
        // This focused guard does not execute a source corpus. It demonstrates
        // that altered target bytes fail even when reference transcriptions agree.
        let mut checks = Checks::default();
        check_production(&case, &observations, 0, &mut checks)?;
        assert_eq!(checks.failed, 0);
        observations[0]["DataHex"] = Value::String(float_hex(&[0.0; 128]));
        check_production(&case, &observations, 0, &mut checks)?;
        assert_eq!(checks.failed, 1);
        Ok(())
    }
}
