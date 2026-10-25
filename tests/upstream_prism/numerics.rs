//! Complete safe Rust ports of col2im, `DFly` fusion and `RoPE` source contracts.
//!
//! PrismML-Eng/llama.cpp b87ca7225456b06b0975b8239c66b5aecf07e37c;
//! upstream MIT attribution is retained in tools/prism/upstream/LICENSE.
#![allow(
    clippy::suboptimal_flops,
    clippy::needless_range_loop,
    reason = "Preserve the original scalar arithmetic order and explicit tensor index formulas."
)]

use super::engine::Engine;
use super::{Result, invalid, result};
use serde_json::{Value, json};

const COL2IM_ID: &str = "tests/test-col2im-1d.cpp";
const DFLY_ID: &str = "tests/test-dfly-fusion.cpp";
const ROPE_ID: &str = "tests/test-rope.cpp";

#[derive(Default)]
struct Graph {
    nodes: Vec<Value>,
}

impl Graph {
    fn node(&mut self, value: Value) -> usize {
        let index = self.nodes.len();
        self.nodes.push(value);
        index
    }
    fn f32(&mut self, dimensions: &[usize], values: &[f32]) -> usize {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        self.node(json!({"Op":"input","Type":"F32","Dimensions":dimensions,"DataHex":hex(&bytes)}))
    }
    fn i32(&mut self, values: &[i32]) -> usize {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        self.node(
            json!({"Op":"input","Type":"I32","Dimensions":[values.len()],"DataHex":hex(&bytes)}),
        )
    }
    fn unary(&mut self, op: &str, input: usize) -> usize {
        self.node(json!({"Op":op,"Inputs":[input]}))
    }
    fn binary(&mut self, op: &str, left: usize, right: usize) -> usize {
        self.node(json!({"Op":op,"Inputs":[left,right]}))
    }
    fn reshape(&mut self, input: usize, dimensions: &[usize]) -> usize {
        self.node(json!({"Op":"reshape","Inputs":[input],"Dimensions":dimensions}))
    }
    fn permute(&mut self, input: usize, axes: [usize; 4]) -> usize {
        self.node(json!({"Op":"permute","Inputs":[input],"Axes":axes}))
    }
    fn cast(&mut self, input: usize, kind: &str) -> usize {
        self.node(json!({"Op":"cast","Inputs":[input],"Type":kind}))
    }
    fn exchange(
        &self,
        engine: &Engine,
        name: &str,
        outputs: &[usize],
        threads: u32,
    ) -> Result<Option<Vec<Tensor>>> {
        let request = json!({"SchemaVersion":1,"Operations":[{"Kind":"tensor-graph","Threads":threads,"Nodes":self.nodes,"Outputs":outputs}]});
        let Some(response) = engine.execute_common(name, &request)? else {
            return Ok(None);
        };
        exact_fields(&response, &["SchemaVersion", "Results"])?;
        if response["SchemaVersion"] != 1 {
            return Err(invalid("tensor graph response version mismatch"));
        }
        let results = array(&response["Results"])?;
        if results.len() != 1 {
            return Err(invalid("tensor graph response operation count mismatch"));
        }
        exact_fields(&results[0], &["Outputs"])?;
        let values = array(&results[0]["Outputs"])?;
        if values.len() != outputs.len() {
            return Err(invalid("tensor graph output count mismatch"));
        }
        Ok(Some(values.iter().map(tensor).collect::<Result<Vec<_>>>()?))
    }
}

struct Tensor {
    dimensions: [usize; 4],
    values: Vec<f32>,
}

fn exact_fields(value: &Value, names: &[&str]) -> Result<()> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid("graph response must be an object"))?;
    if object.len() != names.len() || names.iter().any(|name| !object.contains_key(*name)) {
        return Err(invalid("graph response object fields mismatch"));
    }
    Ok(())
}
fn array(value: &Value) -> Result<&[Value]> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| invalid("graph response must be an array"))
}
fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        result.push(char::from(DIGITS[usize::from(byte >> 4)]));
        result.push(char::from(DIGITS[usize::from(byte & 15)]));
    }
    result
}
fn unhex(value: &str) -> Result<Vec<u8>> {
    if value.len() > 16 * 1024 * 1024
        || !value.len().is_multiple_of(2)
        || !value.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(invalid("invalid graph output hex extent/digits"));
    }
    value
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| Ok(u8::from_str_radix(std::str::from_utf8(pair)?, 16)?))
        .collect()
}
fn half(bits: u16) -> f32 {
    let exponent = i32::from((bits >> 10) & 31);
    let fraction = f32::from(bits & 1023);
    let value = if exponent == 0 {
        fraction * 2.0_f32.powi(-24)
    } else if exponent == 31 {
        f32::NAN
    } else {
        (1.0 + fraction / 1024.0) * 2.0_f32.powi(exponent - 15)
    };
    if bits & 0x8000 == 0 { value } else { -value }
}
fn tensor(value: &Value) -> Result<Tensor> {
    exact_fields(value, &["Type", "Dimensions", "DataHex"])?;
    let shape = array(&value["Dimensions"])?;
    if shape.len() != 4 {
        return Err(invalid("graph output must have four dimensions"));
    }
    let mut dimensions = [0; 4];
    let mut count = 1_usize;
    for (target, source) in dimensions.iter_mut().zip(shape) {
        *target = usize::try_from(
            source
                .as_u64()
                .ok_or_else(|| invalid("graph dimension must be unsigned"))?,
        )?;
        if *target == 0 || *target > 8192 {
            return Err(invalid("graph dimension outside bound"));
        }
        count = count
            .checked_mul(*target)
            .ok_or_else(|| invalid("graph output element overflow"))?;
    }
    if count > 2 * 1024 * 1024 {
        return Err(invalid("graph output element bound exceeded"));
    }
    let bytes = unhex(
        value["DataHex"]
            .as_str()
            .ok_or_else(|| invalid("graph DataHex must be text"))?,
    )?;
    let kind = value["Type"]
        .as_str()
        .ok_or_else(|| invalid("graph Type must be text"))?;
    let width = match kind {
        "F32" => 4,
        "F16" | "BF16" => 2,
        _ => return Err(invalid("unsupported floating graph output type")),
    };
    if bytes.len() != count * width {
        return Err(invalid("graph output bytes do not match dimensions/type"));
    }
    let values = bytes
        .chunks_exact(width)
        .map(|chunk| {
            let value = if width == 4 {
                f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
            } else {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                if kind == "F16" {
                    half(bits)
                } else {
                    f32::from_bits(u32::from(bits) << 16)
                }
            };
            if value.is_finite() {
                Ok(value)
            } else {
                Err(invalid("nonfinite production graph output"))
            }
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Tensor { dimensions, values })
}
fn unsupported(id: &str, contract: &str) -> Value {
    result(
        id,
        false,
        format!("unsupported production tensor-graph API; {contract} required"),
    )
}

// Windows UCRT rand(): the original separate executables start at seed 1;
// DFly explicitly resets to 1234. RAND_MAX is 32767 on the scoped platform.
struct WindowsRandom(u32);
impl WindowsRandom {
    fn unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(214_013).wrapping_add(2_531_011);
        let bytes = self.0.to_le_bytes();
        f32::from(u16::from_le_bytes([bytes[2], bytes[3] & 127])) / 32767.0
    }
    fn signed(&mut self) -> f32 {
        2.0 * self.unit() - 1.0
    }
}
struct ColRandom(u64);
impl ColRandom {
    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "The original cast converts a masked 24-bit integer, exactly representable as f32."
    )]
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) & 0xff_ffff) as f32 / 8_388_608.0 - 1.0
    }
}

const COL_CASES: [(usize, usize, usize, usize, usize); 11] = [
    (16, 32, 197, 8, 0),
    (4, 3, 7, 2, 0),
    (1, 5, 13, 1, 0),
    (6, 4, 11, 3, 1),
    (2, 3, 9, 3, 0),
    (5, 4, 11, 2, 0),
    (8, 4, 13, 4, 2),
    (4, 3, 1, 2, 0),
    (16, 1, 197, 8, 0),
    (1, 5, 13, 3, 0),
    (8, 2, 3, 2, 5),
];
fn col_graph(
    random: &mut ColRandom,
    shape: (usize, usize, usize, usize, usize),
) -> (Graph, [usize; 4]) {
    let (kernel, channels, length, stride, padding) = shape;
    let mut graph = Graph::default();
    let weights: Vec<f32> = (0..kernel * channels * 7).map(|_| random.next()).collect();
    let inputs: Vec<f32> = (0..length * 7).map(|_| random.next()).collect();
    let weights = graph.f32(&[kernel, channels, 7], &weights);
    let inputs = graph.f32(&[length, 7], &inputs);
    let reference = graph.node(json!({"Op":"conv_transpose_1d","Inputs":[weights,inputs],"Stride":stride,"Padding":0,"Dilation":1}));
    let permuted = graph.permute(weights, [1, 2, 0, 3]);
    let contiguous = graph.unary("cont", permuted);
    let weights = graph.reshape(contiguous, &[7, kernel * channels]);
    let transposed = graph.unary("transpose", inputs);
    let inputs = graph.unary("cont", transposed);
    let columns = graph.binary("mul_mat", weights, inputs);
    let mut outputs = [reference, 0, 0, 0];
    for (index, kind) in ["F32", "F16", "BF16"].iter().enumerate() {
        let column = if *kind == "F32" {
            columns
        } else {
            graph.cast(columns, kind)
        };
        outputs[index+1] = graph.node(json!({"Op":"col2im_1d","Inputs":[column],"Stride":stride,"Channels":channels,"Padding":padding}));
    }
    (graph, outputs)
}
fn cropped_nmse(
    actual: &[f32],
    reference: &[f32],
    length: usize,
    reference_length: usize,
    channels: usize,
    padding: usize,
) -> f64 {
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for channel in 0..channels {
        for time in 0..length {
            let actual = f64::from(actual[time + channel * length]);
            let reference = f64::from(reference[time + padding + channel * reference_length]);
            numerator += (actual - reference) * (actual - reference);
            denominator += reference * reference;
        }
    }
    numerator / (denominator + 1e-30)
}
pub(super) fn run_col2im(engine: &Engine) -> Result<Value> {
    let mut random = ColRandom(0x1234_5678);
    let mut failures = Vec::new();
    for shape in COL_CASES {
        let (kernel, channels, length, stride, padding) = shape;
        let reference_length = (length - 1) * stride + kernel;
        let output_length = reference_length - 2 * padding;
        let (graph, ids) = col_graph(&mut random, shape);
        let Some(outputs) = graph.exchange(engine, "col2im-1d", &ids, 4)? else {
            return Ok(unsupported(
                COL2IM_ID,
                "11 original geometries, 22 shape contracts and 33 NMSE thresholds",
            ));
        };
        if outputs[0].dimensions != [reference_length, channels, 1, 1] {
            failures.push(format!("{shape:?}: native reference shape mismatch"));
            continue;
        }
        for (index, threshold) in [1e-7, 5e-4, 5e-4].iter().enumerate() {
            if outputs[index + 1].dimensions != [output_length, channels, 1, 1] {
                failures.push(format!(
                    "{shape:?} precision {index}: output shape mismatch"
                ));
                continue;
            }
            let error = cropped_nmse(
                &outputs[index + 1].values,
                &outputs[0].values,
                output_length,
                reference_length,
                channels,
                padding,
            );
            if !error.is_finite() || error > *threshold {
                failures.push(format!("{shape:?} precision {index}: NMSE={error}"));
            }
        }
    }
    Ok(result(
        COL2IM_ID,
        failures.is_empty(),
        format!(
            "11 original geometries x F32/F16/BF16; cropped NMSE <=1e-7/5e-4/5e-4; failed={}; {}",
            failures.len(),
            failures.join("; ")
        ),
    ))
}

fn dfly_graph(
    input: &[f32],
    projection: &[f32],
    fusion: &[f32],
    norm: &[f32],
) -> (Graph, [usize; 2]) {
    let mut graph = Graph::default();
    let input = graph.f32(&[40, 4], input);
    let projection = graph.f32(&[40, 8], projection);
    let fusion = graph.f32(&[5, 3], fusion);
    let norm = graph.f32(&[8], norm);
    let base = graph.binary("mul_mat", projection, input);
    let probabilities = graph.unary("soft_max", fusion);
    let features = graph.reshape(input, &[8, 5, 4]);
    let features = graph.permute(features, [1, 0, 2, 3]);
    let features = graph.unary("cont", features);
    let features = graph.reshape(features, &[5, 32]);
    let residual = graph.binary("mul_mat", probabilities, features);
    let residual = graph.reshape(residual, &[3, 8, 4]);
    let residual = graph.permute(residual, [1, 0, 2, 3]);
    let residual = graph.unary("cont", residual);
    let base = graph.reshape(base, &[8, 1, 4]);
    let current = graph.binary("add", residual, base);
    let current = graph.node(json!({"Op":"rms_norm","Inputs":[current],"Epsilon":1e-6}));
    let current = graph.binary("mul", current, norm);
    let current = graph.reshape(current, &[24, 4]);
    let slice = graph.node(json!({"Op":"view_2d","Inputs":[current],"Dimensions":[8,4],"RowStrideBytes":96,"OffsetBytes":32}));
    let slice = graph.unary("cont", slice);
    (graph, [current, slice])
}
fn dfly_reference(input: &[f32], projection: &[f32], fusion: &[f32], norm: &[f32]) -> Vec<f32> {
    let mut reference = vec![0.0; 96];
    for token in 0..4 {
        let mut base = [0.0_f32; 8];
        for hidden in 0..8 {
            for source in 0..40 {
                base[hidden] += projection[source + hidden * 40] * input[source + token * 40];
            }
        }
        for layer in 0..3 {
            let mut maximum = f32::NEG_INFINITY;
            for feature in 0..5 {
                maximum = maximum.max(fusion[feature + layer * 5]);
            }
            let mut weights = [0.0_f32; 5];
            let mut sum = 0.0_f32;
            for feature in 0..5 {
                weights[feature] = (fusion[feature + layer * 5] - maximum).exp();
                sum += weights[feature];
            }
            for weight in &mut weights {
                *weight /= sum;
            }
            let mut values = [0.0_f32; 8];
            for hidden in 0..8 {
                let mut accumulator = 0.0_f32;
                for feature in 0..5 {
                    accumulator += weights[feature] * input[hidden + feature * 8 + token * 40];
                }
                values[hidden] = base[hidden] + accumulator;
            }
            let mut squared = 0.0_f32;
            for value in values {
                squared += value * value;
            }
            let scale = 1.0_f32 / (squared / 8.0 + 1e-6).sqrt();
            for hidden in 0..8 {
                reference[hidden + layer * 8 + token * 24] = values[hidden] * scale * norm[hidden];
            }
        }
    }
    reference
}
pub(super) fn run_dfly(engine: &Engine) -> Result<Value> {
    let mut random = WindowsRandom(1234);
    let input: Vec<f32> = (0..160).map(|_| random.signed()).collect();
    let projection: Vec<f32> = (0..320).map(|_| random.signed()).collect();
    let fusion: Vec<f32> = (0..15).map(|_| random.signed()).collect();
    let norm: Vec<f32> = (0..8).map(|_| random.signed()).collect();
    let (graph, ids) = dfly_graph(&input, &projection, &fusion, &norm);
    let Some(outputs) = graph.exchange(engine, "dfly-fusion", &ids, 2)? else {
        return Ok(unsupported(
            DFLY_ID,
            "all 3 original non-square numerical/layout/separation contracts",
        ));
    };
    if outputs[0].dimensions != [24, 4, 1, 1] || outputs[1].dimensions != [8, 4, 1, 1] {
        return Ok(result(
            DFLY_ID,
            false,
            "DFly output/slice shape mismatch".to_owned(),
        ));
    }
    let reference = dfly_reference(&input, &projection, &fusion, &norm);
    let maximum = outputs[0]
        .values
        .iter()
        .zip(&reference)
        .map(|(&actual, &reference)| (f64::from(actual) - f64::from(reference)).abs())
        .fold(0.0_f64, f64::max);
    let mut slice_error = 0.0_f64;
    let mut minimum_separation = f64::INFINITY;
    for token in 0..4 {
        for hidden in 0..8 {
            slice_error = slice_error.max(
                (f64::from(outputs[1].values[hidden + token * 8])
                    - f64::from(reference[hidden + 8 + token * 24]))
                .abs(),
            );
        }
        for layer in 1..3 {
            let mut squared = 0.0;
            for hidden in 0..8 {
                let delta = f64::from(outputs[0].values[hidden + token * 24])
                    - f64::from(outputs[0].values[hidden + layer * 8 + token * 24]);
                squared += delta * delta;
            }
            minimum_separation = minimum_separation.min(squared.sqrt());
        }
    }
    Ok(result(
        DFLY_ID,
        maximum < 1e-4 && slice_error < 1e-4 && minimum_separation > 1e-3,
        format!(
            "3 original contracts; embd8/feat5/layer3/token4; max_abs={maximum}; layer1_slice_abs={slice_error}; min_layer_separation={minimum_separation}"
        ),
    ))
}

fn rope_graph(values: &[f32], mode: u32) -> Result<(Graph, [usize; 2])> {
    let mut graph = Graph::default();
    let input = graph.f32(&[256, 32, 73, 1], values);
    let sections = if mode < 8 { 1 } else { 4 };
    let mut positions = [Vec::new(), Vec::new(), Vec::new()];
    for section in 0..sections {
        for token in 0_usize..73 {
            positions[0].push(100 + i32::try_from(token)? + section);
            positions[1].push(-67);
            positions[2].push(33 + i32::try_from(token)? + section);
        }
    }
    let positions = [
        graph.i32(&positions[0]),
        graph.i32(&positions[1]),
        graph.i32(&positions[2]),
    ];
    let mut rotate = |source, position| {
        if mode < 8 {
            graph.node(json!({"Op":"rope","Inputs":[source,position],"Rotations":128,"Mode":mode}))
        } else {
            graph.node(json!({"Op":"rope_multi","Inputs":[source,position],"Rotations":128,"Mode":mode,"Sections":[16,24,24,0],"Context":32768,"FrequencyBase":1_000_000.0,"FrequencyScale":1.0,"ExtensionFactor":0.0,"AttentionFactor":1.0,"BetaFast":32.0,"BetaSlow":1.0}))
        }
    };
    let first = rotate(input, positions[0]);
    let shifted = rotate(first, positions[1]);
    let direct = rotate(input, positions[2]);
    Ok((graph, [shifted, direct]))
}
pub(super) fn run_rope(engine: &Engine) -> Result<Value> {
    let mut random = WindowsRandom(1);
    let mut failures = Vec::new();
    for mode in [0, 2, 8, 24, 40] {
        let values: Vec<f32> = (0..256 * 32 * 73)
            .map(|_| random.unit() * 2.0 - 1.0)
            .collect();
        let (graph, ids) = rope_graph(&values, mode)?;
        let Some(outputs) = graph.exchange(engine, "rope", &ids, 4)? else {
            return Ok(unsupported(
                ROPE_ID,
                "all 5 original modes, full 256x32x73 tensors and 10 relative-L1 assertions",
            ));
        };
        if outputs
            .iter()
            .any(|output| output.dimensions != [256, 32, 73, 1])
        {
            failures.push(format!("mode {mode}: RoPE output shape mismatch"));
            continue;
        }
        let mut sum0 = 0.0;
        let mut sum1 = 0.0;
        let mut difference = 0.0;
        for (&left, &right) in outputs[0].values.iter().zip(&outputs[1].values) {
            sum0 += f64::from(left.abs());
            sum1 += f64::from(right.abs());
            // Original subtraction rounds in f32 before fabs/double accumulation.
            difference += f64::from((left - right).abs());
        }
        for (index, sum) in [sum0, sum1].iter().enumerate() {
            let error = difference / sum;
            if !error.is_finite() || error >= f64::from(0.0001_f32) {
                failures.push(format!(
                    "mode {mode} denominator {index}: relative L1={error}"
                ));
            }
        }
    }
    Ok(result(
        ROPE_ID,
        failures.is_empty(),
        format!(
            "5 original modes 0/2/8/24/40; full 256x32x73x1; 10 relative-L1 assertions <0.0001f; failed={}; {}",
            failures.len(),
            failures.join("; ")
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn malformed_graph_response_and_nonfinite_values_are_rejected() -> Result<()> {
        assert!(tensor(&json!({"Type":"F32","Dimensions":[1,1,1,1],"DataHex":"00"})).is_err());
        assert!(
            tensor(
                &json!({"Type":"F32","Dimensions":[1,1,1,1],"DataHex":hex(&f32::NAN.to_le_bytes())})
            )
            .is_err()
        );
        assert!(tensor(&json!({"Type":"F32","Dimensions":[0,1,1,1],"DataHex":""})).is_err());
        assert!(unhex("gg").is_err());
        let value = tensor(&json!({"Type":"F16","Dimensions":[1,1,1,1],"DataHex":"003c"}))?;
        assert_eq!(value.values, [1.0]);
        Ok(())
    }
}
