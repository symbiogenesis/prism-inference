//! Independent safe Rust tensor-graph evaluator for `col2im-1d`, `dfly-fusion` and `rope`.
//!
//! No production `bonsai_prism_runtime` code, native `CPU` backend or expected-answer
//! fixture is called here. Immutable upstream `MIT` source identities are disclosed by
//! `identity`.

use std::sync::Arc;

use serde::Deserialize;
use serde_json::{Value, json};

use super::{Result, invalid};

pub(super) const IDS: &[&str] = &[
    "tests/test-col2im-1d.cpp",
    "tests/test-dfly-fusion.cpp",
    "tests/test-rope.cpp",
];

const MAX_ELEMENTS: usize = 2 * 1024 * 1024;
const MAX_STORAGE: usize = 96 * 1024 * 1024;
const MAX_HEX_CHARS: usize = 16 * 1024 * 1024;
const MAX_OUTPUT_HEX_CHARS: usize = 15 * 1024 * 1024;
const TAU: f64 = std::f64::consts::TAU;

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
struct Request {
    schema_version: u32,
    operations: Vec<TensorGraphOp>,
}

#[derive(Deserialize)]
#[serde(tag = "Kind", rename_all_fields = "PascalCase", deny_unknown_fields)]
enum TensorGraphOp {
    #[serde(rename = "tensor-graph")]
    Graph {
        threads: u64,
        nodes: Vec<RawNode>,
        outputs: Vec<u64>,
    },
}

#[derive(Deserialize)]
#[serde(tag = "Op", rename_all_fields = "PascalCase", deny_unknown_fields)]
enum RawNode {
    #[serde(rename = "input")]
    Input {
        #[serde(rename = "Type")]
        kind: String,
        dimensions: Vec<u64>,
        data_hex: String,
    },
    #[serde(rename = "cont")]
    Cont { inputs: Vec<u64> },
    #[serde(rename = "transpose")]
    Transpose { inputs: Vec<u64> },
    #[serde(rename = "soft_max")]
    SoftMax { inputs: Vec<u64> },
    #[serde(rename = "permute")]
    Permute { inputs: Vec<u64>, axes: Vec<u64> },
    #[serde(rename = "reshape")]
    Reshape {
        inputs: Vec<u64>,
        dimensions: Vec<u64>,
    },
    #[serde(rename = "cast")]
    Cast {
        inputs: Vec<u64>,
        #[serde(rename = "Type")]
        kind: String,
    },
    #[serde(rename = "mul_mat")]
    MulMat { inputs: Vec<u64> },
    #[serde(rename = "add")]
    Add { inputs: Vec<u64> },
    #[serde(rename = "mul")]
    Mul { inputs: Vec<u64> },
    #[serde(rename = "rms_norm")]
    RmsNorm { inputs: Vec<u64>, epsilon: f64 },
    #[serde(rename = "view_2d")]
    View2d {
        inputs: Vec<u64>,
        dimensions: Vec<u64>,
        row_stride_bytes: u64,
        offset_bytes: u64,
    },
    #[serde(rename = "conv_transpose_1d")]
    ConvTranspose1d {
        inputs: Vec<u64>,
        stride: u64,
        padding: u64,
        dilation: u64,
    },
    #[serde(rename = "col2im_1d")]
    Col2im1d {
        inputs: Vec<u64>,
        stride: u64,
        padding: u64,
        #[serde(default = "default_one_u64")]
        dilation: u64,
        channels: u64,
    },
    #[serde(rename = "rope")]
    Rope {
        inputs: Vec<u64>,
        rotations: u64,
        mode: u64,
        #[serde(default = "default_rope_context")]
        context: u64,
        #[serde(default = "default_rope_freq_base")]
        frequency_base: f64,
        #[serde(default = "default_one_f64")]
        frequency_scale: f64,
        #[serde(default)]
        extension_factor: f64,
        #[serde(default = "default_one_f64")]
        attention_factor: f64,
        #[serde(default = "default_rope_beta_fast")]
        beta_fast: f64,
        #[serde(default = "default_one_f64")]
        beta_slow: f64,
    },
    #[serde(rename = "rope_multi")]
    RopeMulti {
        inputs: Vec<u64>,
        rotations: u64,
        mode: u64,
        context: u64,
        frequency_base: f64,
        frequency_scale: f64,
        extension_factor: f64,
        attention_factor: f64,
        beta_fast: f64,
        beta_slow: f64,
        sections: Vec<u64>,
    },
}

const fn default_one_u64() -> u64 {
    1
}

const fn default_rope_context() -> u64 {
    32_768
}

const fn default_rope_freq_base() -> f64 {
    10_000.0
}

const fn default_one_f64() -> f64 {
    1.0
}

const fn default_rope_beta_fast() -> f64 {
    32.0
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DType {
    F32,
    F16,
    Bf16,
    I32,
}

impl DType {
    fn parse(raw: &str) -> Result<Self> {
        match raw {
            "F32" => Ok(Self::F32),
            "F16" => Ok(Self::F16),
            "BF16" => Ok(Self::Bf16),
            "I32" => Ok(Self::I32),
            _ => Err(invalid("unsupported tensor data type")),
        }
    }

    const fn byte_size(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::Bf16 => 2,
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::I32 => "I32",
        }
    }
}

#[derive(Clone)]
struct Tensor {
    dtype: DType,
    ne: [usize; 4],
    strides: [usize; 4],
    contiguous: bool,
    offset: usize,
    data: Arc<Vec<u8>>,
}

impl Tensor {
    fn from_contiguous(dtype: DType, ne: [usize; 4], bytes: Vec<u8>) -> Result<Self> {
        let count = validated_count(ne)?;
        let expected = count
            .checked_mul(dtype.byte_size())
            .ok_or_else(|| invalid("tensor byte size overflow"))?;
        require(
            bytes.len() == expected,
            "tensor byte buffer length mismatch",
        )?;
        Ok(Self {
            dtype,
            ne,
            strides: contiguous_strides(dtype.byte_size(), ne),
            contiguous: true,
            offset: 0,
            data: Arc::new(bytes),
        })
    }

    const fn elem_bytes(&self) -> usize {
        self.dtype.byte_size()
    }

    const fn count(&self) -> usize {
        self.ne[0] * self.ne[1] * self.ne[2] * self.ne[3]
    }

    const fn byte_len(&self) -> usize {
        self.count() * self.elem_bytes()
    }

    const fn elem_byte_offset(&self, idx: [usize; 4]) -> usize {
        self.offset
            + idx[0] * self.strides[0]
            + idx[1] * self.strides[1]
            + idx[2] * self.strides[2]
            + idx[3] * self.strides[3]
    }

    fn read_f32_at(&self, byte_off: usize) -> Result<f32> {
        let slice = self
            .data
            .get(byte_off..byte_off + 4)
            .ok_or_else(|| invalid("f32 tensor read out of bounds"))?;
        Ok(f32::from_le_bytes(slice.try_into()?))
    }

    fn read_scalar_f32_flat(&self, elem_idx: usize) -> Result<f32> {
        let byte_off = self.offset + elem_idx * self.elem_bytes();
        match self.dtype {
            DType::F32 => self.read_f32_at(byte_off),
            DType::F16 => {
                let slice = self
                    .data
                    .get(byte_off..byte_off + 2)
                    .ok_or_else(|| invalid("f16 tensor read out of bounds"))?;
                Ok(f16_to_f32(u16::from_le_bytes(slice.try_into()?)))
            }
            DType::Bf16 => {
                let slice = self
                    .data
                    .get(byte_off..byte_off + 2)
                    .ok_or_else(|| invalid("bf16 tensor read out of bounds"))?;
                Ok(bf16_to_f32(u16::from_le_bytes(slice.try_into()?)))
            }
            DType::I32 => Err(invalid("cannot read I32 tensor as floating scalar")),
        }
    }

    fn read_i32_flat(&self, elem_idx: usize) -> Result<i32> {
        let byte_off = self.offset + elem_idx * 4;
        let slice = self
            .data
            .get(byte_off..byte_off + 4)
            .ok_or_else(|| invalid("i32 tensor read out of bounds"))?;
        Ok(i32::from_le_bytes(slice.try_into()?))
    }
}

struct RopeParams {
    rotations: usize,
    mode: usize,
    context: f64,
    freq_base: f64,
    freq_scale: f64,
    ext_factor: f64,
    attn_factor: f64,
    beta_fast: f64,
    beta_slow: f64,
    sections: [usize; 4],
    multi: bool,
}

const fn widen_u16(value: u16) -> u32 {
    let [lo, hi] = value.to_le_bytes();
    u32::from_le_bytes([lo, hi, 0, 0])
}

const fn narrow_u32_to_u16(value: u32) -> u16 {
    let [b0, b1, _, _] = value.to_le_bytes();
    u16::from_le_bytes([b0, b1])
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "IEEE-754 round-to-nearest narrowing from f64 to f32 matches C++ double-to-float cast"
)]
const fn f64_to_f32(value: f64) -> f32 {
    value as f32
}

const fn f16_to_f32(bits: u16) -> f32 {
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

const fn contiguous_strides(elem_bytes: usize, ne: [usize; 4]) -> [usize; 4] {
    let s1 = elem_bytes * ne[0];
    let s2 = s1 * ne[1];
    let s3 = s2 * ne[2];
    [elem_bytes, s1, s2, s3]
}

const fn is_contiguous_strides(elem_bytes: usize, ne: [usize; 4], strides: [usize; 4]) -> bool {
    strides[0] == elem_bytes
        && strides[1] == strides[0] * ne[0]
        && strides[2] == strides[1] * ne[1]
        && strides[3] == strides[2] * ne[2]
}

fn require(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(invalid(message))
    }
}

fn validated_count(ne: [usize; 4]) -> Result<usize> {
    let mut count = 1_usize;
    for dim in ne {
        require((1..=8192).contains(&dim), "tensor dimension outside bound")?;
        count = count
            .checked_mul(dim)
            .ok_or_else(|| invalid("tensor element count overflow"))?;
        require(count <= MAX_ELEMENTS, "tensor exceeds element bound")?;
    }
    Ok(count)
}

fn parse_dimensions(raw: &[u64], expected_rank: Option<usize>) -> Result<[usize; 4]> {
    require(
        !raw.is_empty() && raw.len() <= 4,
        "tensor requires 1..4 dimensions",
    )?;
    if let Some(rank) = expected_rank {
        require(raw.len() == rank, "tensor rank mismatch")?;
    }
    let mut ne = [1_usize; 4];
    for (target, &dim) in ne.iter_mut().zip(raw) {
        *target = usize::try_from(dim)?;
    }
    validated_count(ne)?;
    Ok(ne)
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(char::from(DIGITS[usize::from(byte >> 4)]));
        output.push(char::from(DIGITS[usize::from(byte & 15)]));
    }
    output
}

const fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn unhex(text: &str) -> Result<Vec<u8>> {
    require(
        text.len() <= MAX_HEX_CHARS && text.len().is_multiple_of(2),
        "tensor hex extent outside bound",
    )?;
    text.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&[hi, lo]| {
            let high = hex_nibble(hi).ok_or_else(|| invalid("invalid hex digit"))?;
            let low = hex_nibble(lo).ok_or_else(|| invalid("invalid hex digit"))?;
            Ok((high << 4) | low)
        })
        .collect()
}

fn unary_input<'a>(inputs: &[u64], nodes: &'a [Tensor]) -> Result<&'a Tensor> {
    require(inputs.len() == 1, "unary op requires 1 input")?;
    let idx = usize::try_from(inputs[0])?;
    nodes
        .get(idx)
        .ok_or_else(|| invalid("tensor input index out of range"))
}

fn binary_inputs<'a>(inputs: &[u64], nodes: &'a [Tensor]) -> Result<(&'a Tensor, &'a Tensor)> {
    require(inputs.len() == 2, "binary op requires 2 inputs")?;
    let a = nodes
        .get(usize::try_from(inputs[0])?)
        .ok_or_else(|| invalid("tensor first input out of range"))?;
    let b = nodes
        .get(usize::try_from(inputs[1])?)
        .ok_or_else(|| invalid("tensor second input out of range"))?;
    Ok((a, b))
}

fn eval_input(kind: &str, dimensions: &[u64], data_hex: &str) -> Result<Tensor> {
    let dtype = DType::parse(kind)?;
    let ne = parse_dimensions(dimensions, None)?;
    let bytes = unhex(data_hex)?;
    if dtype == DType::F32 {
        for chunk in bytes.as_chunks::<4>().0 {
            require(
                f32::from_le_bytes(*chunk).is_finite(),
                "nonfinite F32 tensor input",
            )?;
        }
    }
    Tensor::from_contiguous(dtype, ne, bytes)
}

fn eval_cont(src: &Tensor) -> Result<Tensor> {
    let elem = src.elem_bytes();
    let count = validated_count(src.ne)?;
    let mut out = Vec::with_capacity(count * elem);
    for i3 in 0..src.ne[3] {
        for i2 in 0..src.ne[2] {
            for i1 in 0..src.ne[1] {
                for i0 in 0..src.ne[0] {
                    let off = src.elem_byte_offset([i0, i1, i2, i3]);
                    let slice = src
                        .data
                        .get(off..off + elem)
                        .ok_or_else(|| invalid("cont source offset out of bounds"))?;
                    out.extend_from_slice(slice);
                }
            }
        }
    }
    Tensor::from_contiguous(src.dtype, src.ne, out)
}

fn eval_transpose(src: &Tensor) -> Tensor {
    let mut ne = src.ne;
    let mut strides = src.strides;
    ne.swap(0, 1);
    strides.swap(0, 1);
    Tensor {
        dtype: src.dtype,
        ne,
        strides,
        contiguous: is_contiguous_strides(src.elem_bytes(), ne, strides),
        offset: src.offset,
        data: Arc::clone(&src.data),
    }
}

fn eval_permute(src: &Tensor, axes_raw: &[u64]) -> Result<Tensor> {
    require(axes_raw.len() == 4, "permute requires 4 axes")?;
    let mut seen = [false; 4];
    let mut perm_axes = [0_usize; 4];
    for (slot, &raw) in perm_axes.iter_mut().zip(axes_raw) {
        let axis = usize::try_from(raw)?;
        require(
            axis < 4 && !seen[axis],
            "permute requires valid 0..4 axis permutation",
        )?;
        seen[axis] = true;
        *slot = axis;
    }
    let mut ne = [0_usize; 4];
    let mut strides = [0_usize; 4];
    for (src_axis, &dst_axis) in perm_axes.iter().enumerate() {
        ne[dst_axis] = src.ne[src_axis];
        strides[dst_axis] = src.strides[src_axis];
    }
    Ok(Tensor {
        dtype: src.dtype,
        ne,
        strides,
        contiguous: is_contiguous_strides(src.elem_bytes(), ne, strides),
        offset: src.offset,
        data: Arc::clone(&src.data),
    })
}

fn eval_reshape(src: &Tensor, dimensions: &[u64]) -> Result<Tensor> {
    require(src.contiguous, "reshape requires contiguous source")?;
    let ne = parse_dimensions(dimensions, None)?;
    let count = validated_count(ne)?;
    require(count == src.count(), "reshape element count mismatch")?;
    Ok(Tensor {
        dtype: src.dtype,
        ne,
        strides: contiguous_strides(src.elem_bytes(), ne),
        contiguous: true,
        offset: src.offset,
        data: Arc::clone(&src.data),
    })
}

fn eval_cast(src: &Tensor, kind: &str) -> Result<Tensor> {
    require(src.dtype == DType::F32, "cast source must be F32")?;
    let target = DType::parse(kind)?;
    require(target != DType::I32, "cast target cannot be I32")?;
    let count = validated_count(src.ne)?;
    let mut out = Vec::with_capacity(count * target.byte_size());
    for i3 in 0..src.ne[3] {
        for i2 in 0..src.ne[2] {
            for i1 in 0..src.ne[1] {
                for i0 in 0..src.ne[0] {
                    let val = src.read_f32_at(src.elem_byte_offset([i0, i1, i2, i3]))?;
                    match target {
                        DType::F32 => out.extend_from_slice(&val.to_le_bytes()),
                        DType::F16 => out.extend_from_slice(&f32_to_f16(val).to_le_bytes()),
                        DType::Bf16 => out.extend_from_slice(&f32_to_bf16(val).to_le_bytes()),
                        DType::I32 => return Err(invalid("cast target cannot be I32")),
                    }
                }
            }
        }
    }
    Tensor::from_contiguous(target, src.ne, out)
}

fn eval_mul_mat(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    require(
        a.dtype == DType::F32 && b.dtype == DType::F32,
        "mul_mat inputs must be F32",
    )?;
    require(a.contiguous, "mul_mat left operand must be contiguous")?;
    require(a.ne[0] == b.ne[0], "mul_mat inner dimension K mismatch")?;
    require(
        b.ne[2].is_multiple_of(a.ne[2]) && b.ne[3].is_multiple_of(a.ne[3]),
        "mul_mat batch broadcast mismatch",
    )?;
    let (k_len, m_len) = (a.ne[0], a.ne[1]);
    let (n_len, b2_len, b3_len) = (b.ne[1], b.ne[2], b.ne[3]);
    let r2 = b2_len / a.ne[2];
    let r3 = b3_len / a.ne[3];
    let out_ne = [m_len, n_len, b2_len, b3_len];
    let count = validated_count(out_ne)?;
    let mut out = Vec::with_capacity(count * 4);
    for i3 in 0..b3_len {
        let a3 = i3 / r3;
        for i2 in 0..b2_len {
            let a2 = i2 / r2;
            for n in 0..n_len {
                for m in 0..m_len {
                    let mut acc = 0.0_f64;
                    for k in 0..k_len {
                        let va = a.read_f32_at(a.elem_byte_offset([k, m, a2, a3]))?;
                        let vb = b.read_f32_at(b.elem_byte_offset([k, n, i2, i3]))?;
                        let prod = f64::from(va) * f64::from(vb);
                        acc += prod;
                    }
                    out.extend_from_slice(&f64_to_f32(acc).to_le_bytes());
                }
            }
        }
    }
    Tensor::from_contiguous(DType::F32, out_ne, out)
}

fn eval_binary(a: &Tensor, b: &Tensor, is_mul: bool) -> Result<Tensor> {
    require(
        a.dtype == DType::F32 && b.dtype == DType::F32,
        "binary elementwise ops require F32",
    )?;
    for (da, db) in a.ne.into_iter().zip(b.ne) {
        require(da.is_multiple_of(db), "incompatible broadcast shapes")?;
    }
    let count = validated_count(a.ne)?;
    let mut out = Vec::with_capacity(count * 4);
    for i3 in 0..a.ne[3] {
        for i2 in 0..a.ne[2] {
            for i1 in 0..a.ne[1] {
                for i0 in 0..a.ne[0] {
                    let va = a.read_f32_at(a.elem_byte_offset([i0, i1, i2, i3]))?;
                    let vb = b.read_f32_at(b.elem_byte_offset([
                        i0 % b.ne[0],
                        i1 % b.ne[1],
                        i2 % b.ne[2],
                        i3 % b.ne[3],
                    ]))?;
                    let val = if is_mul { va * vb } else { va + vb };
                    out.extend_from_slice(&val.to_le_bytes());
                }
            }
        }
    }
    Tensor::from_contiguous(DType::F32, a.ne, out)
}

fn eval_soft_max(src: &Tensor) -> Result<Tensor> {
    require(
        src.dtype == DType::F32 && src.contiguous,
        "soft_max requires contiguous F32 input",
    )?;
    let row_len = src.ne[0];
    let rows = src.count() / row_len;
    let mut out = Vec::with_capacity(src.count() * 4);
    let mut exps = Vec::with_capacity(row_len);
    for row in 0..rows {
        let base = row * row_len;
        let mut max_val = f32::NEG_INFINITY;
        for col in 0..row_len {
            let val = src.read_f32_at(src.offset + (base + col) * 4)?;
            if val > max_val {
                max_val = val;
            }
        }
        exps.clear();
        let mut sum = 0.0_f64;
        for col in 0..row_len {
            let val = src.read_f32_at(src.offset + (base + col) * 4)?;
            let e = (val - max_val).exp();
            exps.push(e);
            sum += f64::from(e);
        }
        let inv = f64_to_f32(1.0_f64 / sum);
        for &e in &exps {
            out.extend_from_slice(&(e * inv).to_le_bytes());
        }
    }
    Tensor::from_contiguous(DType::F32, src.ne, out)
}

fn eval_rms_norm(src: &Tensor, epsilon: f64) -> Result<Tensor> {
    require(
        src.dtype == DType::F32 && src.contiguous,
        "rms_norm requires contiguous F32 input",
    )?;
    require(
        epsilon.is_finite() && (0.0..=1.0).contains(&epsilon),
        "rms_norm epsilon outside 0..=1",
    )?;
    let row_len = src.ne[0];
    let rows = src.count() / row_len;
    let row_len_f64 = f64::from(u32::try_from(row_len)?);
    let mut out = Vec::with_capacity(src.count() * 4);
    for row in 0..rows {
        let base = row * row_len;
        let mut sum_sq = 0.0_f64;
        for col in 0..row_len {
            let val = f64::from(src.read_f32_at(src.offset + (base + col) * 4)?);
            let sq = val * val;
            sum_sq += sq;
        }
        let scale = f64_to_f32(1.0_f64 / (sum_sq / row_len_f64 + epsilon).sqrt());
        for col in 0..row_len {
            let val = src.read_f32_at(src.offset + (base + col) * 4)?;
            out.extend_from_slice(&(val * scale).to_le_bytes());
        }
    }
    Tensor::from_contiguous(DType::F32, src.ne, out)
}

fn eval_view_2d(
    src: &Tensor,
    dimensions: &[u64],
    row_stride_bytes: u64,
    offset_bytes: u64,
) -> Result<Tensor> {
    require(
        src.contiguous && src.dtype == DType::F32,
        "view_2d requires contiguous F32 source",
    )?;
    let ne = parse_dimensions(dimensions, Some(2))?;
    let stride = usize::try_from(row_stride_bytes)?;
    let offset = usize::try_from(offset_bytes)?;
    require(
        (1..=MAX_STORAGE).contains(&stride) && stride.is_multiple_of(4) && stride >= ne[0] * 4,
        "view_2d invalid row stride",
    )?;
    require(
        offset <= MAX_STORAGE && offset.is_multiple_of(4),
        "view_2d invalid offset",
    )?;
    let extent = (ne[1] - 1)
        .checked_mul(stride)
        .and_then(|rows| rows.checked_add(ne[0] * 4))
        .and_then(|span| offset.checked_add(span))
        .ok_or_else(|| invalid("view_2d extent overflow"))?;
    require(extent <= src.byte_len(), "view_2d exceeds source extent")?;
    let strides = [4, stride, stride * ne[1], stride * ne[1]];
    Ok(Tensor {
        dtype: src.dtype,
        ne,
        strides,
        contiguous: stride == ne[0] * 4,
        offset: src.offset + offset,
        data: Arc::clone(&src.data),
    })
}

fn eval_conv_transpose_1d(
    a: &Tensor,
    b: &Tensor,
    stride_u64: u64,
    padding_u64: u64,
    dilation_u64: u64,
) -> Result<Tensor> {
    require(
        a.dtype == DType::F32 && b.dtype == DType::F32 && a.contiguous && b.contiguous,
        "conv_transpose_1d requires contiguous F32 inputs",
    )?;
    let stride = usize::try_from(stride_u64)?;
    require(
        (1..=4096).contains(&stride),
        "conv_transpose_1d stride outside bound",
    )?;
    require(
        padding_u64 == 0 && dilation_u64 == 1,
        "conv_transpose_1d requires padding 0 dilation 1",
    )?;
    require(
        a.ne[3] == 1 && b.ne[2] == 1 && b.ne[3] == 1 && a.ne[2] == b.ne[1],
        "conv_transpose_1d tensor dimension mismatch",
    )?;
    let (kernel_len, filters, depth, steps) = (a.ne[0], a.ne[1], a.ne[2], b.ne[0]);
    let out_steps = (steps - 1)
        .checked_mul(stride)
        .and_then(|base| base.checked_add(kernel_len))
        .ok_or_else(|| invalid("conv_transpose_1d output length overflow"))?;
    let out_ne = [out_steps, filters, 1, 1];
    let count = validated_count(out_ne)?;
    let mut acc = vec![0.0_f32; count];
    for ic in 0..depth {
        for il in 0..steps {
            let vb = b.read_f32_at(b.offset + (il + ic * steps) * 4)?;
            for oc in 0..filters {
                for k in 0..kernel_len {
                    let va =
                        a.read_f32_at(a.offset + (k + (oc + ic * filters) * kernel_len) * 4)?;
                    let prod = va * vb;
                    acc[il * stride + k + oc * out_steps] += prod;
                }
            }
        }
    }
    let bytes = acc.into_iter().flat_map(f32::to_le_bytes).collect();
    Tensor::from_contiguous(DType::F32, out_ne, bytes)
}

fn eval_col2im_1d(
    src: &Tensor,
    raw_stride: u64,
    padding_u64: u64,
    dilation_u64: u64,
    channels_u64: u64,
) -> Result<Tensor> {
    require(
        src.contiguous && src.dtype != DType::I32,
        "col2im_1d requires contiguous non-I32 input",
    )?;
    require(src.ne[3] == 1, "col2im_1d requires ne[3] == 1")?;
    let stride = usize::try_from(raw_stride)?;
    let padding = usize::try_from(padding_u64)?;
    let dilation = usize::try_from(dilation_u64)?;
    let channels = usize::try_from(channels_u64)?;
    require(
        (1..=4096).contains(&stride),
        "col2im_1d stride outside bound",
    )?;
    require(padding <= 4096, "col2im_1d padding outside bound")?;
    require(
        (1..=4096).contains(&dilation),
        "col2im_1d dilation outside bound",
    )?;
    require(
        (1..=8192).contains(&channels) && src.ne[0].is_multiple_of(channels),
        "col2im_1d channel count mismatch",
    )?;
    let kernel = src.ne[0] / channels;
    let il = src.ne[1];
    let stride_signed = i64::try_from(stride)?;
    let pad_i64 = i64::try_from(padding)?;
    let dil_i64 = i64::try_from(dilation)?;
    let ol_i64 = i64::try_from(il - 1)? * stride_signed + dil_i64 * i64::try_from(kernel - 1)? + 1
        - 2 * pad_i64;
    require(ol_i64 > 0, "col2im_1d output length must be positive")?;
    let ol = usize::try_from(ol_i64)?;
    let out_ne = [ol, channels, src.ne[2], 1];
    let count = validated_count(out_ne)?;
    let mut acc = vec![0.0_f32; count];
    for batch in 0..src.ne[2] {
        for il_idx in 0..il {
            let base_pos = i64::try_from(il_idx)? * stride_signed - pad_i64;
            for oc in 0..channels {
                for k_idx in 0..kernel {
                    let pos = base_pos + i64::try_from(k_idx)? * dil_i64;
                    if (0..ol_i64).contains(&pos) {
                        let flat = k_idx + oc * kernel + (il_idx + batch * il) * src.ne[0];
                        let val = src.read_scalar_f32_flat(flat)?;
                        let dst_idx = usize::try_from(pos)? + (oc + batch * channels) * ol;
                        acc[dst_idx] += val;
                    }
                }
            }
        }
    }
    encode_f32_vec(src.dtype, out_ne, &acc)
}

fn encode_f32_vec(dtype: DType, ne: [usize; 4], values: &[f32]) -> Result<Tensor> {
    let mut bytes = Vec::with_capacity(values.len() * dtype.byte_size());
    for &val in values {
        match dtype {
            DType::F32 => bytes.extend_from_slice(&val.to_le_bytes()),
            DType::F16 => bytes.extend_from_slice(&f32_to_f16(val).to_le_bytes()),
            DType::Bf16 => bytes.extend_from_slice(&f32_to_bf16(val).to_le_bytes()),
            DType::I32 => return Err(invalid("cannot encode float vector as I32")),
        }
    }
    Tensor::from_contiguous(dtype, ne, bytes)
}

fn validate_rope_params(a: &Tensor, pos: &Tensor, params: &RopeParams) -> Result<()> {
    require(
        a.contiguous && a.dtype == DType::F32,
        "rope requires contiguous F32 data input",
    )?;
    require(
        pos.contiguous && pos.dtype == DType::I32,
        "rope requires contiguous I32 positions input",
    )?;
    if params.multi {
        require(
            matches!(params.mode, 8 | 24 | 40),
            "rope_multi mode must be 8, 24 or 40",
        )?;
        require(
            pos.count() == a.ne[2] * 4,
            "rope_multi positions length must equal 4 * ne[2]",
        )?;
        let sect_sum: usize = params.sections.iter().sum();
        require(
            sect_sum == params.rotations / 2,
            "rope_multi sections must sum to rotations / 2",
        )?;
        if params.mode == 24 {
            require(
                a.ne[0] == 2 * params.rotations,
                "vision rope requires ne[0] == 2 * rotations",
            )?;
        }
    } else {
        require(matches!(params.mode, 0 | 2), "rope mode must be 0 or 2")?;
        require(
            pos.ne[0] == a.ne[2] && pos.count() == a.ne[2],
            "rope positions must match ne[2]",
        )?;
    }
    require(
        params.rotations > 0 && params.rotations.is_multiple_of(2) && params.rotations <= a.ne[0],
        "rope rotations must be positive, even and <= ne[0]",
    )?;
    require(
        params.context >= 1.0
            && params.freq_base.is_finite()
            && params.freq_base > 0.0
            && params.freq_scale.is_finite()
            && params.freq_scale > 0.0
            && params.ext_factor.is_finite()
            && params.ext_factor >= 0.0
            && params.attn_factor.is_finite()
            && params.beta_fast.is_finite()
            && params.beta_slow.is_finite(),
        "rope numeric parameter invalid or out of range",
    )
}

struct RopeCache<'a> {
    cos: &'a [f32],
    sin: &'a [f32],
    is_normal: bool,
}

fn yarn_corr_dims(params: &RopeParams) -> Result<(f64, f64)> {
    if params.ext_factor == 0.0 {
        return Ok((0.0, 0.0));
    }
    let ln_base = params.freq_base.ln();
    require(ln_base.abs() > 0.0, "YaRN requires frequency_base != 1.0")?;
    let n_dims_f = f64::from(u32::try_from(params.rotations)?);
    let max_dim = f64::from(u32::try_from(params.rotations - 1)?);
    let corr_dim = |n_rot: f64| -> Result<f64> {
        let ratio = params.context / (n_rot * TAU);
        require(
            ratio.is_finite() && ratio > 0.0,
            "invalid YaRN correction domain",
        )?;
        Ok(n_dims_f * ratio.ln() / (2.0 * ln_base))
    };
    let low = corr_dim(params.beta_fast)?.floor().max(0.0);
    let high = corr_dim(params.beta_slow)?.ceil().min(max_dim);
    Ok((low, high))
}

fn rope_cos_sin(
    params: &RopeParams,
    corr: (f64, f64),
    pos_val: i32,
    pair_idx: usize,
) -> Result<(f32, f32)> {
    let n_dims_f = f64::from(u32::try_from(params.rotations)?);
    let pair_f = f64::from(u32::try_from(pair_idx)?);
    let exponent = -(2.0 * pair_f) / n_dims_f;
    let freq = params.freq_base.powf(exponent);
    let theta_extrap = f64::from(pos_val) * freq;
    let theta_interp = params.freq_scale * theta_extrap;
    let (theta, mscale) = if params.ext_factor == 0.0 {
        (theta_interp, params.attn_factor)
    } else {
        let (low, high) = corr;
        let span = (high - low).max(0.001);
        let y = (pair_f - low) / span;
        let ramp = (1.0 - y.clamp(0.0, 1.0)) * params.ext_factor;
        let scaled = theta_interp * (1.0 - ramp);
        let extra = theta_extrap * ramp;
        let log_scale = (1.0_f64 / params.freq_scale).ln();
        let mscale_corr = 0.1_f64.mul_add(log_scale, 1.0_f64);
        (scaled + extra, params.attn_factor * mscale_corr)
    };
    Ok((
        f64_to_f32(theta.cos() * mscale),
        f64_to_f32(theta.sin() * mscale),
    ))
}

const fn mrope_section_pos(params: &RopeParams, sector: usize, pos_4: [i32; 4]) -> i32 {
    if params.mode == 40 {
        let s1 = params.sections[1];
        let s2 = params.sections[2];
        if sector % 3 == 1 && sector < 3 * s1 {
            return pos_4[1];
        }
        if sector % 3 == 2 && sector < 3 * s2 {
            return pos_4[2];
        }
        return pos_4[0];
    }
    let s0 = params.sections[0];
    let s1 = params.sections[1];
    let s2 = params.sections[2];
    if sector < s0 {
        pos_4[0]
    } else if sector < s0 + s1 {
        pos_4[1]
    } else if sector < s0 + s1 + s2 {
        pos_4[2]
    } else {
        pos_4[3]
    }
}

fn read_token_pos4(pos: &Tensor, i2: usize, ne2: usize, multi: bool) -> Result<[i32; 4]> {
    if multi {
        Ok([
            pos.read_i32_flat(i2)?,
            pos.read_i32_flat(i2 + ne2)?,
            pos.read_i32_flat(i2 + ne2 * 2)?,
            pos.read_i32_flat(i2 + ne2 * 3)?,
        ])
    } else {
        Ok([pos.read_i32_flat(i2)?, 0, 0, 0])
    }
}

fn rotate_rope_row(
    a: &Tensor,
    row: usize,
    ne0: usize,
    n_pairs: usize,
    cache: &RopeCache<'_>,
    out: &mut [f32],
) -> Result<()> {
    for pair in 0..n_pairs {
        let (idx0, idx1) = if cache.is_normal {
            (pair * 2, pair * 2 + 1)
        } else {
            (pair, pair + n_pairs)
        };
        let x0 = a.read_f32_at(a.offset + (row + idx0) * 4)?;
        let x1 = a.read_f32_at(a.offset + (row + idx1) * 4)?;
        let (c, s) = (cache.cos[pair], cache.sin[pair]);
        let rx = x0 * c;
        let ry = x1 * s;
        let sx = x0 * s;
        let sy = x1 * c;
        out[row + idx0] = rx - ry;
        out[row + idx1] = sx + sy;
    }
    for col in (n_pairs * 2)..ne0 {
        out[row + col] = a.read_f32_at(a.offset + (row + col) * 4)?;
    }
    Ok(())
}

fn eval_rope(a: &Tensor, pos: &Tensor, params: &RopeParams) -> Result<Tensor> {
    validate_rope_params(a, pos, params)?;
    let corr = yarn_corr_dims(params)?;
    let [ne0, ne1, ne2, ne3] = a.ne;
    let n_pairs = if params.mode == 24 {
        params.rotations
    } else {
        params.rotations / 2
    };
    let sect_dims: usize = params.sections.iter().sum();
    let mut out = vec![0.0_f32; a.count()];
    let mut cos_cache = Vec::with_capacity(n_pairs);
    let mut sin_cache = Vec::with_capacity(n_pairs);
    for i3 in 0..ne3 {
        for i2 in 0..ne2 {
            let pos_4 = read_token_pos4(pos, i2, ne2, params.multi)?;
            cos_cache.clear();
            sin_cache.clear();
            for pair in 0..n_pairs {
                let p = if params.multi {
                    mrope_section_pos(params, pair % sect_dims, pos_4)
                } else {
                    pos_4[0]
                };
                let (c, s) = rope_cos_sin(params, corr, p, pair)?;
                cos_cache.push(c);
                sin_cache.push(s);
            }
            let cache = RopeCache {
                cos: &cos_cache,
                sin: &sin_cache,
                is_normal: params.mode == 0,
            };
            for i1 in 0..ne1 {
                let row = ((i3 * ne2 + i2) * ne1 + i1) * ne0;
                rotate_rope_row(a, row, ne0, n_pairs, &cache, &mut out)?;
            }
        }
    }
    let bytes = out.into_iter().flat_map(f32::to_le_bytes).collect();
    Tensor::from_contiguous(DType::F32, a.ne, bytes)
}

fn eval_rope_node(node: &RawNode, nodes: &[Tensor]) -> Result<Tensor> {
    match node {
        RawNode::Rope {
            inputs,
            rotations,
            mode,
            context,
            frequency_base,
            frequency_scale,
            extension_factor,
            attention_factor,
            beta_fast,
            beta_slow,
        } => {
            let (a, pos) = binary_inputs(inputs, nodes)?;
            let params = RopeParams {
                rotations: usize::try_from(*rotations)?,
                mode: usize::try_from(*mode)?,
                context: f64::from(u32::try_from(*context)?),
                freq_base: *frequency_base,
                freq_scale: *frequency_scale,
                ext_factor: *extension_factor,
                attn_factor: *attention_factor,
                beta_fast: *beta_fast,
                beta_slow: *beta_slow,
                sections: [0, 0, 0, 0],
                multi: false,
            };
            eval_rope(a, pos, &params)
        }
        RawNode::RopeMulti {
            inputs,
            rotations,
            mode,
            context,
            frequency_base,
            frequency_scale,
            extension_factor,
            attention_factor,
            beta_fast,
            beta_slow,
            sections,
        } => {
            require(sections.len() == 4, "rope_multi requires 4 sections")?;
            let (a, pos) = binary_inputs(inputs, nodes)?;
            let params = RopeParams {
                rotations: usize::try_from(*rotations)?,
                mode: usize::try_from(*mode)?,
                context: f64::from(u32::try_from(*context)?),
                freq_base: *frequency_base,
                freq_scale: *frequency_scale,
                ext_factor: *extension_factor,
                attn_factor: *attention_factor,
                beta_fast: *beta_fast,
                beta_slow: *beta_slow,
                sections: [
                    usize::try_from(sections[0])?,
                    usize::try_from(sections[1])?,
                    usize::try_from(sections[2])?,
                    usize::try_from(sections[3])?,
                ],
                multi: true,
            };
            eval_rope(a, pos, &params)
        }
        _ => Err(invalid("expected rope or rope_multi node")),
    }
}

fn eval_node(node: &RawNode, nodes: &[Tensor]) -> Result<Tensor> {
    match node {
        RawNode::Input {
            kind,
            dimensions,
            data_hex,
        } => eval_input(kind, dimensions, data_hex),
        RawNode::Cont { inputs } => eval_cont(unary_input(inputs, nodes)?),
        RawNode::Transpose { inputs } => Ok(eval_transpose(unary_input(inputs, nodes)?)),
        RawNode::Permute { inputs, axes } => eval_permute(unary_input(inputs, nodes)?, axes),
        RawNode::Reshape { inputs, dimensions } => {
            eval_reshape(unary_input(inputs, nodes)?, dimensions)
        }
        RawNode::Cast { inputs, kind } => eval_cast(unary_input(inputs, nodes)?, kind),
        RawNode::MulMat { inputs } => {
            let (a, b) = binary_inputs(inputs, nodes)?;
            eval_mul_mat(a, b)
        }
        RawNode::Add { inputs } | RawNode::Mul { inputs } => {
            let (a, b) = binary_inputs(inputs, nodes)?;
            eval_binary(a, b, matches!(node, RawNode::Mul { .. }))
        }
        RawNode::SoftMax { inputs } => eval_soft_max(unary_input(inputs, nodes)?),
        RawNode::RmsNorm { inputs, epsilon } => {
            eval_rms_norm(unary_input(inputs, nodes)?, *epsilon)
        }
        RawNode::View2d {
            inputs,
            dimensions,
            row_stride_bytes,
            offset_bytes,
        } => eval_view_2d(
            unary_input(inputs, nodes)?,
            dimensions,
            *row_stride_bytes,
            *offset_bytes,
        ),
        RawNode::ConvTranspose1d {
            inputs,
            stride,
            padding,
            dilation,
        } => {
            let (a, b) = binary_inputs(inputs, nodes)?;
            eval_conv_transpose_1d(a, b, *stride, *padding, *dilation)
        }
        RawNode::Col2im1d {
            inputs,
            stride,
            padding,
            dilation,
            channels,
        } => eval_col2im_1d(
            unary_input(inputs, nodes)?,
            *stride,
            *padding,
            *dilation,
            *channels,
        ),
        RawNode::Rope { .. } | RawNode::RopeMulti { .. } => eval_rope_node(node, nodes),
    }
}

fn execute_graph(threads: u64, raw_nodes: &[RawNode], raw_outputs: &[u64]) -> Result<Value> {
    require(
        (1..=16).contains(&threads),
        "tensor-graph Threads outside bound",
    )?;
    require(
        (1..=96).contains(&raw_nodes.len()),
        "tensor-graph requires 1..96 nodes",
    )?;
    require(
        (1..=16).contains(&raw_outputs.len()),
        "tensor-graph requires 1..16 outputs",
    )?;
    let mut nodes = Vec::with_capacity(raw_nodes.len());
    let mut cumulative_storage = 0_usize;
    for raw_node in raw_nodes {
        let tensor = eval_node(raw_node, &nodes)?;
        cumulative_storage = cumulative_storage
            .checked_add(tensor.byte_len())
            .ok_or_else(|| invalid("tensor-graph cumulative storage overflow"))?;
        require(
            cumulative_storage <= MAX_STORAGE,
            "tensor-graph cumulative storage exceeds bound",
        )?;
        nodes.push(tensor);
    }
    let mut total_output_bytes = 0_usize;
    let mut outputs = Vec::with_capacity(raw_outputs.len());
    for &raw_idx in raw_outputs {
        let idx = usize::try_from(raw_idx)?;
        let node = nodes
            .get(idx)
            .ok_or_else(|| invalid("tensor-graph output index out of range"))?;
        require(node.contiguous, "tensor-graph outputs must be contiguous")?;
        total_output_bytes = total_output_bytes
            .checked_add(node.byte_len())
            .ok_or_else(|| invalid("tensor-graph output size overflow"))?;
        require(
            total_output_bytes
                .checked_mul(2)
                .is_some_and(|chars| chars <= MAX_OUTPUT_HEX_CHARS),
            "tensor-graph response hex exceeds bound",
        )?;
        let slice = node
            .data
            .get(node.offset..node.offset + node.byte_len())
            .ok_or_else(|| invalid("tensor-graph output slice out of bounds"))?;
        outputs.push(json!({
            "Type": node.dtype.as_str(),
            "Dimensions": node.ne,
            "DataHex": hex(slice),
        }));
    }
    Ok(json!({"Outputs": outputs}))
}

pub(super) fn handles(value: &Value) -> bool {
    value["Operations"].as_array().is_some_and(|operations| {
        !operations.is_empty()
            && operations
                .iter()
                .all(|operation| operation["Kind"].as_str() == Some("tensor-graph"))
    })
}

pub(super) fn execute(value: &Value) -> Result<Value> {
    let request: Request = serde_json::from_value(value.clone())?;
    require(
        request.schema_version == 1 && (1..=128).contains(&request.operations.len()),
        "tensor-graph schema or operation count outside bound",
    )?;
    let results = request
        .operations
        .iter()
        .map(|op| match op {
            TensorGraphOp::Graph {
                threads,
                nodes,
                outputs,
            } => execute_graph(*threads, nodes, outputs),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(json!({"SchemaVersion": 1, "Results": results}))
}

pub(super) fn identity() -> Value {
    json!({
        "Kind": "independent-rust-tensor-graph",
        "SupportedSourceIds": IDS,
        "UpstreamCommit": super::UPSTREAM,
        "AssertionSources": [
            {
                "Path": IDS[0],
                "Sha256": "8b0b9fec8a5f7177722060fccfa67ddcd4b1073c833e1511bc4106b03f6c8323"
            },
            {
                "Path": IDS[1],
                "Sha256": "504161d9c3ec6f5ca621b12dc952b5a9cc2e631edb4ad225132b219057eb1fdd"
            },
            {
                "Path": IDS[2],
                "Sha256": "cca4b2a3acbce6c410ec808f1cf263b191aea2c7a3b5df6d14b17c6be13871df"
            }
        ],
        "Architecture": "strided-arc-storage-f16-bf16-rne-tensor-graph-evaluator-v1",
        "NativeReferenceReplay": false,
        "ProductionRuntimeCalled": false,
        "CompleteSourceGate": false,
        "NativeRetirementQualified": false
    })
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::super::{Engine, Target};
    use super::{
        IDS, Result, bf16_to_f32, execute, f16_to_f32, f32_to_bf16, f32_to_f16, handles, hex,
        identity, invalid, unhex,
    };

    fn f32_hex(values: &[f32]) -> String {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        hex(&bytes)
    }

    fn i32_hex(values: &[i32]) -> String {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        hex(&bytes)
    }

    fn decode_f32_hex(hex_str: &str) -> Result<Vec<f32>> {
        let bytes = unhex(hex_str)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect())
    }

    fn decode_u16_hex(hex_str: &str) -> Result<Vec<u16>> {
        let bytes = unhex(hex_str)?;
        Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| u16::from_le_bytes(*chunk))
            .collect())
    }

    fn wrap_graph(threads: u64, nodes: &[Value], outputs: &[usize]) -> Value {
        json!({
            "SchemaVersion": 1,
            "Operations": [{
                "Kind": "tensor-graph",
                "Threads": threads,
                "Nodes": nodes,
                "Outputs": outputs
            }]
        })
    }

    fn assert_close(actual: &[f32], expected: &[f32], tol: f32) -> Result<()> {
        require_eq_len(actual.len(), expected.len())?;
        for (&a, &e) in actual.iter().zip(expected) {
            let diff = (a - e).abs();
            assert!(
                diff <= tol,
                "mismatch: actual={a}, expected={e}, diff={diff}"
            );
        }
        Ok(())
    }

    fn require_eq_len(left: usize, right: usize) -> Result<()> {
        if left == right {
            Ok(())
        } else {
            Err(invalid("test slice length mismatch"))
        }
    }

    #[test]
    fn independent_tensor_graph_preserves_col2im_dfly_and_rope_suites() -> Result<()> {
        for target in [Target::RustReference, Target::Cpp, Target::Rust] {
            let engine = Engine::new(target, None, None, None, None, None)?;
            for runner in [
                crate::numerics::run_col2im,
                crate::numerics::run_dfly,
                crate::numerics::run_rope,
            ] {
                let report = runner(&engine)?;
                assert_eq!(report["Status"], "pass", "{report}");
            }
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
    fn independent_f16_and_bf16_conversions_match_ieee754_rne() {
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert_eq!(f32_to_f16(-0.0), 0x8000);
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0_f32).to_bits());
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(-2.5), 0xc100);
        assert_eq!(f32_to_f16(65_504.0), 0x7bff);
        assert_eq!(f16_to_f32(0x7bff), 65_504.0);
        assert_eq!(f32_to_f16(f32::from_bits(0x3380_0000)), 0x0001);
        assert_eq!(f16_to_f32(0x0001).to_bits(), 0x3380_0000);
        assert_eq!(f32_to_f16(f32::from_bits(0x3300_0000)), 0x0000);
        assert_eq!(f32_to_f16(f32::from_bits(0x3300_0001)), 0x0001);
        assert_eq!(f32_to_f16(f32::from_bits(0x387f_ffff)), 0x0400);
        assert_eq!(f32_to_f16(f32::from_bits(0x3f80_1000)), 0x3c00);
        assert_eq!(f32_to_f16(f32::from_bits(0x3f80_3000)), 0x3c02);
        assert_eq!(f32_to_f16(65_520.0), 0x7c00);
        assert_eq!(f32_to_f16(f32::NEG_INFINITY), 0xfc00);
        let q16 = f32_to_f16(f32::from_bits(0x7f80_0001));
        assert_eq!(q16 & 0x7e00, 0x7e00);
        assert!(f16_to_f32(q16).is_nan());

        assert_eq!(f32_to_bf16(0.0), 0x0000);
        assert_eq!(f32_to_bf16(-0.0), 0x8000);
        assert_eq!(f32_to_bf16(1.0), 0x3f80);
        assert_eq!(f32_to_bf16(-2.5), 0xc020);
        assert_eq!(f16_to_f32(0x03ff), 1023.0 * f32::from_bits(0x3380_0000));
        assert_eq!(bf16_to_f32(0xc020), -2.5);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f81_8000)), 0x3f82);
        assert_eq!(f32_to_bf16(f32::MAX), 0x7f80);
        assert_eq!(f32_to_bf16(f32::NEG_INFINITY), 0xff80);
        let qbf = f32_to_bf16(f32::from_bits(0x7f80_0001));
        assert_eq!(qbf, 0x7fc0);
        assert!(bf16_to_f32(qbf).is_nan());
    }

    #[test]
    fn independent_col2im_and_conv_transpose_match_across_types_padding_and_dilation() -> Result<()>
    {
        let a = [
            1.0_f32, -2.0, 0.5, 3.0, -1.5, 2.0, 0.25, -0.5, 1.5, -1.0, 0.75, 2.5,
        ];
        let b = [1.0_f32, 2.0, -1.0, 0.5, -2.0, 1.5];
        let req = wrap_graph(
            2,
            &[
                json!({"Op": "input", "Type": "F32", "Dimensions": [3, 2, 2], "DataHex": f32_hex(&a)}),
                json!({"Op": "input", "Type": "F32", "Dimensions": [3, 2], "DataHex": f32_hex(&b)}),
                json!({"Op": "conv_transpose_1d", "Inputs": [0, 1], "Stride": 2, "Padding": 0, "Dilation": 1}),
                json!({"Op": "permute", "Inputs": [0], "Axes": [1, 2, 0, 3]}),
                json!({"Op": "cont", "Inputs": [3]}),
                json!({"Op": "reshape", "Inputs": [4], "Dimensions": [2, 6]}),
                json!({"Op": "transpose", "Inputs": [1]}),
                json!({"Op": "mul_mat", "Inputs": [5, 6]}),
                json!({"Op": "col2im_1d", "Inputs": [7], "Stride": 2, "Padding": 0, "Dilation": 1, "Channels": 2}),
                json!({"Op": "cast", "Inputs": [7], "Type": "F16"}),
                json!({"Op": "col2im_1d", "Inputs": [9], "Stride": 2, "Padding": 0, "Dilation": 1, "Channels": 2}),
                json!({"Op": "cast", "Inputs": [7], "Type": "BF16"}),
                json!({"Op": "col2im_1d", "Inputs": [11], "Stride": 2, "Padding": 0, "Dilation": 1, "Channels": 2}),
            ],
            &[2, 8, 10, 12],
        );
        assert!(handles(&req));
        let res = execute(&req)?;
        let outs = res["Results"][0]["Outputs"]
            .as_array()
            .ok_or_else(|| invalid("missing Outputs"))?;
        assert_eq!(outs[0]["Type"], "F32");
        assert_eq!(outs[1]["Type"], "F32");
        assert_eq!(outs[2]["Type"], "F16");
        assert_eq!(outs[3]["Type"], "BF16");
        assert_eq!(outs[0]["Dimensions"], json!([7, 2, 1, 1]));

        let conv = decode_f32_hex(outs[0]["DataHex"].as_str().ok_or_else(|| invalid("hex"))?)?;
        let col_f32 = decode_f32_hex(outs[1]["DataHex"].as_str().ok_or_else(|| invalid("hex"))?)?;
        assert_close(&col_f32, &conv, 1e-5)?;
        let half_out: Vec<f32> =
            decode_u16_hex(outs[2]["DataHex"].as_str().ok_or_else(|| invalid("hex"))?)?
                .into_iter()
                .map(f16_to_f32)
                .collect();
        let bfloat_out: Vec<f32> =
            decode_u16_hex(outs[3]["DataHex"].as_str().ok_or_else(|| invalid("hex"))?)?
                .into_iter()
                .map(bf16_to_f32)
                .collect();
        assert_close(&half_out, &conv, 0.05)?;
        assert_close(&bfloat_out, &conv, 0.2)?;

        let dilated_request = wrap_graph(
            1,
            &[
                json!({"Op": "input", "Type": "F32", "Dimensions": [4, 2], "DataHex": f32_hex(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])}),
                json!({"Op": "col2im_1d", "Inputs": [0], "Stride": 2, "Padding": 1, "Dilation": 2, "Channels": 2}),
            ],
            &[1],
        );
        let dilated_response = execute(&dilated_request)?;
        let dil_out = &dilated_response["Results"][0]["Outputs"][0];
        assert_eq!(dil_out["Dimensions"], json!([3, 2, 1, 1]));
        let dil_vals = decode_f32_hex(dil_out["DataHex"].as_str().ok_or_else(|| invalid("hex"))?)?;
        assert_close(&dil_vals, &[0.0, 7.0, 0.0, 0.0, 11.0, 0.0], 1e-6)
    }

    #[test]
    fn independent_dfly_ops_verify_gqa_softmax_rmsnorm_and_strided_views() -> Result<()> {
        let a = [1.0_f32, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0];
        let b = [3.0_f32, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let norm_in = [3.0_f32, 4.0, 0.0, 2.0];
        let view_src = [1.0_f32, 2.0, 99.0, 3.0, 4.0, 99.0];
        let req = wrap_graph(
            4,
            &[
                json!({"Op": "input", "Type": "F32", "Dimensions": [2, 2, 2], "DataHex": f32_hex(&a)}),
                json!({"Op": "input", "Type": "F32", "Dimensions": [1, 2, 4], "DataHex": f32_hex(&b)}),
                json!({"Op": "transpose", "Inputs": [1]}),
                json!({"Op": "mul_mat", "Inputs": [0, 2]}),
                json!({"Op": "input", "Type": "F32", "Dimensions": [2, 2], "DataHex": f32_hex(&norm_in)}),
                json!({"Op": "rms_norm", "Inputs": [4], "Epsilon": 0.0}),
                json!({"Op": "soft_max", "Inputs": [4]}),
                json!({"Op": "input", "Type": "F32", "Dimensions": [3, 2], "DataHex": f32_hex(&view_src)}),
                json!({"Op": "view_2d", "Inputs": [7], "Dimensions": [2, 2], "RowStrideBytes": 12, "OffsetBytes": 0}),
                json!({"Op": "permute", "Inputs": [8], "Axes": [1, 0, 2, 3]}),
                json!({"Op": "cont", "Inputs": [9]}),
                json!({"Op": "add", "Inputs": [10, 4]}),
                json!({"Op": "mul", "Inputs": [11, 5]}),
            ],
            &[3, 5, 6, 10, 12],
        );
        let res = execute(&req)?;
        let outs = res["Results"][0]["Outputs"]
            .as_array()
            .ok_or_else(|| invalid("missing Outputs"))?;

        let gqa = decode_f32_hex(outs[0]["DataHex"].as_str().ok_or_else(|| invalid("hex"))?)?;
        assert_close(&gqa, &[3.0, 4.0, 5.0, 6.0, 14.0, 16.0, 18.0, 20.0], 1e-6)?;

        let rms = decode_f32_hex(outs[1]["DataHex"].as_str().ok_or_else(|| invalid("hex"))?)?;
        let r0 = (12.5_f32).sqrt();
        let r1 = (2.0_f32).sqrt();
        assert_close(&rms, &[3.0 / r0, 4.0 / r0, 0.0, 2.0 / r1], 1e-5)?;

        let sm = decode_f32_hex(outs[2]["DataHex"].as_str().ok_or_else(|| invalid("hex"))?)?;
        let e0 = (-1.0_f32).exp();
        let inv0 = 1.0 / (e0 + 1.0);
        let e1 = (-2.0_f32).exp();
        let inv1 = 1.0 / (e1 + 1.0);
        assert_close(&sm, &[e0 * inv0, inv0, e1 * inv1, inv1], 1e-5)?;

        let cont = decode_f32_hex(outs[3]["DataHex"].as_str().ok_or_else(|| invalid("hex"))?)?;
        assert_close(&cont, &[1.0, 3.0, 2.0, 4.0], 1e-6)?;
        let combined = decode_f32_hex(outs[4]["DataHex"].as_str().ok_or_else(|| invalid("hex"))?)?;
        assert_close(
            &combined,
            &[4.0 * 3.0 / r0, 7.0 * 4.0 / r0, 0.0, 6.0 * 2.0 / r1],
            1e-5,
        )
    }

    #[test]
    fn independent_rope_covers_all_five_modes_and_composition() -> Result<()> {
        let data = [
            1.0_f32, 0.5, -0.25, 1.5, 0.75, -1.0, 2.0, -0.5, 0.25, 1.25, -0.75, 0.5,
        ];
        for mode in [0_u64, 2, 8, 24, 40] {
            let (dims, rots, tail) = if mode == 24 {
                ([8_u64, 1, 1, 1], 4_u64, 8_usize)
            } else {
                ([6_u64, 2, 1, 1], 4_u64, 12_usize)
            };
            let (pos_forward, pos_rewind, pos_net) = if mode >= 8 {
                ([100, 50, 20, 0], [-67, -30, -10, 0], [33, 20, 10, 0])
            } else {
                ([100, 0, 0, 0], [-67, 0, 0, 0], [33, 0, 0, 0])
            };
            let pos_len = if mode >= 8 { 4_usize } else { 1_usize };
            let op_name = if mode >= 8 { "rope_multi" } else { "rope" };
            let make_rope = |inp: usize, pid: usize| -> Value {
                let mut v = json!({
                    "Op": op_name,
                    "Inputs": [inp, pid],
                    "Rotations": rots,
                    "Mode": mode,
                    "Context": 32_768,
                    "FrequencyBase": 10_000.0,
                    "FrequencyScale": 1.0,
                    "ExtensionFactor": 0.0,
                    "AttentionFactor": 1.0,
                    "BetaFast": 32.0,
                    "BetaSlow": 1.0
                });
                if mode >= 8 {
                    v["Sections"] = json!([1, 1, 0, 0]);
                }
                v
            };
            let req = wrap_graph(
                2,
                &[
                    json!({"Op": "input", "Type": "F32", "Dimensions": dims, "DataHex": f32_hex(&data[..tail])}),
                    json!({"Op": "input", "Type": "I32", "Dimensions": [pos_len], "DataHex": i32_hex(&pos_forward[..pos_len])}),
                    json!({"Op": "input", "Type": "I32", "Dimensions": [pos_len], "DataHex": i32_hex(&pos_rewind[..pos_len])}),
                    json!({"Op": "input", "Type": "I32", "Dimensions": [pos_len], "DataHex": i32_hex(&pos_net[..pos_len])}),
                    make_rope(0, 1),
                    make_rope(4, 2),
                    make_rope(0, 3),
                ],
                &[5, 6],
            );
            let res = execute(&req)?;
            let outs = res["Results"][0]["Outputs"]
                .as_array()
                .ok_or_else(|| invalid("missing Outputs"))?;
            let comp = decode_f32_hex(outs[0]["DataHex"].as_str().ok_or_else(|| invalid("hex"))?)?;
            let single =
                decode_f32_hex(outs[1]["DataHex"].as_str().ok_or_else(|| invalid("hex"))?)?;
            assert_close(&comp, &single, 1e-4)?;
            if mode < 24 {
                assert_close(&single[4..6], &data[4..6], 1e-6)?;
                assert_close(&single[10..12], &data[10..12], 1e-6)?;
            }
        }
        Ok(())
    }

    #[test]
    fn independent_rope_yarn_matches_magnitude_and_ramp_interpolation() -> Result<()> {
        let data = [1.0_f32, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let req = wrap_graph(
            1,
            &[
                json!({"Op": "input", "Type": "F32", "Dimensions": [8], "DataHex": f32_hex(&data)}),
                json!({"Op": "input", "Type": "I32", "Dimensions": [1], "DataHex": i32_hex(&[10])}),
                json!({
                    "Op": "rope",
                    "Inputs": [0, 1],
                    "Rotations": 8,
                    "Mode": 0,
                    "Context": 32_768,
                    "FrequencyBase": 10_000.0,
                    "FrequencyScale": 0.5,
                    "ExtensionFactor": 1.0,
                    "AttentionFactor": 1.2,
                    "BetaFast": 32.0,
                    "BetaSlow": 1.0
                }),
            ],
            &[2],
        );
        let res = execute(&req)?;
        let out = decode_f32_hex(
            res["Results"][0]["Outputs"][0]["DataHex"]
                .as_str()
                .ok_or_else(|| invalid("hex"))?,
        )?;
        let mscale = 1.2_f64 * 0.1_f64.mul_add(2.0_f64.ln(), 1.0_f64);
        assert!((f64::from(out[0].hypot(out[1])) - mscale).abs() < 1e-5);
        assert!(mscale.mul_add(-10.0_f64.cos(), f64::from(out[0])).abs() < 1e-5);
        assert!(mscale.mul_add(-10.0_f64.sin(), f64::from(out[1])).abs() < 1e-5);
        let freq3 = 10_000.0_f64.powf(-0.75);
        let mid_theta = 7.5_f64 * freq3;
        assert!(mscale.mul_add(-mid_theta.cos(), f64::from(out[6])).abs() < 1e-5);
        assert!(mscale.mul_add(-mid_theta.sin(), f64::from(out[7])).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn independent_tensor_graph_refuses_envelope_bounds_cycles_and_shapes() {
        let base_in = json!({"Op": "input", "Type": "F32", "Dimensions": [2, 2], "DataHex": f32_hex(&[1.0, 2.0, 3.0, 4.0])});
        assert!(execute(&json!({"SchemaVersion": 2, "Operations": [{"Kind": "tensor-graph", "Threads": 1, "Nodes": [base_in], "Outputs": [0]}]})).is_err());
        assert!(execute(&json!({"SchemaVersion": 1, "Operations": []})).is_err());
        assert!(execute(&wrap_graph(0, std::slice::from_ref(&base_in), &[0])).is_err());
        assert!(execute(&wrap_graph(17, std::slice::from_ref(&base_in), &[0])).is_err());
        assert!(execute(&wrap_graph(1, &[], &[0])).is_err());
        assert!(execute(&wrap_graph(1, std::slice::from_ref(&base_in), &[])).is_err());
        assert!(execute(&wrap_graph(1, std::slice::from_ref(&base_in), &[1])).is_err());

        let bad_nan = json!({"Op": "input", "Type": "F32", "Dimensions": [2], "DataHex": f32_hex(&[1.0, f32::NAN])});
        assert!(execute(&wrap_graph(1, &[bad_nan], &[0])).is_err());
        assert!(
            execute(&wrap_graph(
                1,
                &[base_in.clone(), json!({"Op": "add", "Inputs": [0, 1]})],
                &[1]
            ))
            .is_err()
        );
        assert!(
            execute(&wrap_graph(
                1,
                &[
                    base_in.clone(),
                    json!({"Op": "reshape", "Inputs": [0], "Dimensions": [3]})
                ],
                &[1]
            ))
            .is_err()
        );
        assert!(
            execute(&wrap_graph(
                1,
                &[base_in.clone(), json!({"Op": "transpose", "Inputs": [0]})],
                &[1]
            ))
            .is_err()
        );
        assert!(
            execute(&wrap_graph(
                1,
                &[
                    base_in.clone(),
                    json!({"Op": "transpose", "Inputs": [0]}),
                    json!({"Op": "reshape", "Inputs": [1], "Dimensions": [4]})
                ],
                &[2]
            ))
            .is_err()
        );
        assert!(
            execute(&wrap_graph(
                1,
                &[
                    base_in.clone(),
                    json!({"Op": "permute", "Inputs": [0], "Axes": [0, 0, 1, 2]})
                ],
                &[1]
            ))
            .is_err()
        );
        assert!(execute(&wrap_graph(1, &[base_in.clone(), json!({"Op": "view_2d", "Inputs": [0], "Dimensions": [2, 2], "RowStrideBytes": 4, "OffsetBytes": 0})], &[1])).is_err());
        assert!(
            execute(&wrap_graph(
                1,
                &[
                    base_in.clone(),
                    json!({"Op": "transpose", "Inputs": [0]}),
                    json!({"Op": "mul_mat", "Inputs": [1, 0]})
                ],
                &[2]
            ))
            .is_err()
        );
        assert!(execute(&wrap_graph(1, &[base_in.clone(), json!({"Op": "input", "Type": "F32", "Dimensions": [3], "DataHex": f32_hex(&[1.0, 2.0, 3.0])}), json!({"Op": "add", "Inputs": [0, 1]})], &[2])).is_err());
        assert!(execute(&wrap_graph(1, &[base_in, json!({"Op": "input", "Type": "F32", "Dimensions": [3, 2], "DataHex": f32_hex(&[1.0; 6])}), json!({"Op": "mul_mat", "Inputs": [0, 1]})], &[2])).is_err());
    }

    #[test]
    fn independent_tensor_graph_refuses_op_specific_domain_errors() {
        let f32_in = json!({"Op": "input", "Type": "F32", "Dimensions": [4, 2], "DataHex": f32_hex(&[1.0; 8])});
        let i32_in =
            json!({"Op": "input", "Type": "I32", "Dimensions": [1], "DataHex": i32_hex(&[5])});
        assert!(
            execute(&wrap_graph(
                1,
                &[
                    f32_in.clone(),
                    json!({"Op": "cast", "Inputs": [0], "Type": "I32"})
                ],
                &[1]
            ))
            .is_err()
        );
        assert!(
            execute(&wrap_graph(
                1,
                &[
                    i32_in.clone(),
                    json!({"Op": "cast", "Inputs": [0], "Type": "F16"})
                ],
                &[1]
            ))
            .is_err()
        );
        assert!(
            execute(&wrap_graph(
                1,
                &[
                    f32_in.clone(),
                    json!({"Op": "rms_norm", "Inputs": [0], "Epsilon": -0.1})
                ],
                &[1]
            ))
            .is_err()
        );
        assert!(
            execute(&wrap_graph(
                1,
                &[
                    f32_in.clone(),
                    json!({"Op": "rms_norm", "Inputs": [0], "Epsilon": 1.5})
                ],
                &[1]
            ))
            .is_err()
        );
        assert!(execute(&wrap_graph(1, &[i32_in, json!({"Op": "col2im_1d", "Inputs": [0], "Stride": 1, "Padding": 0, "Dilation": 1, "Channels": 1})], &[1])).is_err());
        assert!(execute(&wrap_graph(1, &[f32_in.clone(), json!({"Op": "col2im_1d", "Inputs": [0], "Stride": 1, "Padding": 100, "Dilation": 1, "Channels": 2})], &[1])).is_err());

        let k_in = json!({"Op": "input", "Type": "F32", "Dimensions": [2, 1, 1], "DataHex": f32_hex(&[1.0, 2.0])});
        assert!(execute(&wrap_graph(1, &[k_in, f32_in, json!({"Op": "conv_transpose_1d", "Inputs": [0, 1], "Stride": 1, "Padding": 1, "Dilation": 1})], &[2])).is_err());
        assert!(!handles(
            &json!({"Operations": [{"Kind": "cpu-arena-graph"}]})
        ));
    }

    #[test]
    fn independent_tensor_graph_refuses_rope_domain_errors() {
        let f32_in = json!({"Op": "input", "Type": "F32", "Dimensions": [4, 2], "DataHex": f32_hex(&[1.0; 8])});
        let i32_in =
            json!({"Op": "input", "Type": "I32", "Dimensions": [1], "DataHex": i32_hex(&[5])});
        let bad_rope = |mode: u64, op_name: &str, rots: u64, sections: Option<Value>| -> Value {
            let mut n = json!({
                "Op": op_name, "Inputs": [0, 1], "Rotations": rots, "Mode": mode, "Context": 4096,
                "FrequencyBase": 10_000.0, "FrequencyScale": 1.0, "ExtensionFactor": 0.0,
                "AttentionFactor": 1.0, "BetaFast": 32.0, "BetaSlow": 1.0
            });
            if let Some(secs) = sections {
                n["Sections"] = secs;
            }
            n
        };
        let r_in = json!({"Op": "input", "Type": "F32", "Dimensions": [6, 1, 1], "DataHex": f32_hex(&[1.0; 6])});
        let p4 = json!({"Op": "input", "Type": "I32", "Dimensions": [4], "DataHex": i32_hex(&[1, 2, 3, 4])});
        assert!(
            execute(&wrap_graph(
                1,
                &[r_in.clone(), i32_in.clone(), bad_rope(8, "rope", 4, None)],
                &[2]
            ))
            .is_err()
        );
        assert!(
            execute(&wrap_graph(
                1,
                &[
                    r_in.clone(),
                    p4.clone(),
                    bad_rope(2, "rope_multi", 4, Some(json!([1, 1, 0, 0])))
                ],
                &[2]
            ))
            .is_err()
        );
        assert!(
            execute(&wrap_graph(
                1,
                &[
                    r_in.clone(),
                    p4.clone(),
                    bad_rope(24, "rope_multi", 4, Some(json!([1, 1, 0, 0])))
                ],
                &[2]
            ))
            .is_err()
        );
        assert!(
            execute(&wrap_graph(
                1,
                &[
                    r_in.clone(),
                    p4,
                    bad_rope(8, "rope_multi", 4, Some(json!([1, 2, 0, 0])))
                ],
                &[2]
            ))
            .is_err()
        );
        assert!(
            execute(&wrap_graph(
                1,
                &[
                    r_in,
                    i32_in.clone(),
                    bad_rope(8, "rope_multi", 4, Some(json!([1, 1, 0, 0])))
                ],
                &[2]
            ))
            .is_err()
        );
        assert!(
            execute(&wrap_graph(
                1,
                &[f32_in, i32_in, bad_rope(0, "rope", 3, None)],
                &[2]
            ))
            .is_err()
        );
    }
}
