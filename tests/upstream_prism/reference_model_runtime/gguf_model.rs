//! Clean-room `GGUF` v2/v3 reader, writer, tensor decoder, and vocabulary holder.
//!
//! No `bonsai_prism_runtime`, `bonsai_gguf`, or native GGML libraries are linked or called.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Value, json};

use crate::reference_vocab::{ReferenceVocab, VocabType, load_gguf_vocab};
use crate::{Result, invalid};

pub(super) const MAX_GGUF_BYTES: u64 = 128 * 1024 * 1024;
pub(super) const MAX_TENSOR_BYTES: usize = 64 * 1024 * 1024;
const MAX_KV_COUNT: u64 = 131_072;
const MAX_TENSOR_COUNT: u64 = 16_384;
const MAX_STRING_BYTES: u64 = 8 * 1024 * 1024;
const MAX_ARRAY_LEN: u64 = 1_048_576;
const TYPE_F32: i32 = 0;
const TYPE_F16: i32 = 1;
const TYPE_Q4_0: i32 = 2;
const TYPE_Q4_1: i32 = 3;
const TYPE_Q5_0: i32 = 6;
const TYPE_Q5_1: i32 = 7;
const TYPE_Q8_0: i32 = 8;
const TYPE_Q2_K: i32 = 10;
const TYPE_Q3_K: i32 = 11;
const TYPE_Q4_K: i32 = 12;
const TYPE_Q5_K: i32 = 13;
const TYPE_Q6_K: i32 = 14;
const TYPE_Q8_K: i32 = 15;
const TYPE_I8: i32 = 24;
const TYPE_I16: i32 = 25;
const TYPE_I32: i32 = 26;
const TYPE_I64: i32 = 27;
const TYPE_F64: i32 = 28;
const TYPE_BF16: i32 = 30;

static NEXT_OUTPUT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug)]
pub(super) enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(u32, Vec<Self>),
}

impl GgufValue {
    pub(super) const fn type_id(&self) -> u32 {
        match self {
            Self::U8(_) => 0,
            Self::I8(_) => 1,
            Self::U16(_) => 2,
            Self::I16(_) => 3,
            Self::U32(_) => 4,
            Self::I32(_) => 5,
            Self::F32(_) => 6,
            Self::Bool(_) => 7,
            Self::String(_) => 8,
            Self::Array(_, _) => 9,
            Self::U64(_) => 10,
            Self::I64(_) => 11,
            Self::F64(_) => 12,
        }
    }

    pub(super) fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub(super) fn as_u64_coerce(&self) -> Option<u64> {
        match self {
            Self::U8(v) => Some(u64::from(*v)),
            Self::I8(v) => u64::try_from(*v).ok(),
            Self::U16(v) => Some(u64::from(*v)),
            Self::I16(v) => u64::try_from(*v).ok(),
            Self::U32(v) => Some(u64::from(*v)),
            Self::I32(v) => u64::try_from(*v).ok(),
            Self::U64(v) => Some(*v),
            Self::I64(v) => u64::try_from(*v).ok(),
            _ => None,
        }
    }

    pub(super) fn as_i32_coerce(&self) -> Option<i32> {
        match self {
            Self::U8(v) => Some(i32::from(*v)),
            Self::I8(v) => Some(i32::from(*v)),
            Self::U16(v) => Some(i32::from(*v)),
            Self::I16(v) => Some(i32::from(*v)),
            Self::U32(v) => i32::try_from(*v).ok(),
            Self::I32(v) => Some(*v),
            Self::U64(v) => i32::try_from(*v).ok(),
            Self::I64(v) => i32::try_from(*v).ok(),
            _ => None,
        }
    }

    pub(super) const fn as_f32_coerce(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            Self::F64(v) => Some(f64_to_f32(*v)),
            _ => None,
        }
    }

    pub(super) const fn as_bool_coerce(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            Self::U8(v) => Some(*v != 0),
            Self::I8(v) => Some(*v != 0),
            Self::U32(v) => Some(*v != 0),
            Self::I32(v) => Some(*v != 0),
            _ => None,
        }
    }

    pub(super) fn as_i32_vec(&self) -> Vec<i32> {
        match self {
            Self::Array(_, items) => items.iter().filter_map(Self::as_i32_coerce).collect(),
            Self::I32(v) => vec![*v],
            Self::U32(v) => i32::try_from(*v).into_iter().collect(),
            _ => Vec::new(),
        }
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::as_conversions,
    reason = "exact IEEE binary64 to binary32 narrowing helper"
)]
pub(super) const fn f64_to_f32(value: f64) -> f32 {
    value as f32
}

pub(super) fn fp16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = u32::from((bits >> 10) & 0x1f);
    let mant = u32::from(bits & 0x3ff);
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let mut e = 127_u32 - 14;
        let mut m = mant;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        let f_mant = (m & 0x3ff) << 13;
        return f32::from_bits(sign | (e << 23) | f_mant);
    }
    if exp == 31 {
        let f_mant = mant << 13;
        return f32::from_bits(sign | (0xff << 23) | f_mant);
    }
    let f_exp = (exp + 112) << 23;
    let f_mant = mant << 13;
    f32::from_bits(sign | f_exp | f_mant)
}

pub(super) fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

pub(super) fn block_info(kind: i32) -> Result<(usize, usize)> {
    match kind {
        TYPE_F32 | TYPE_I32 => Ok((1, 4)),
        TYPE_F16 | TYPE_BF16 | TYPE_I16 => Ok((1, 2)),
        TYPE_I8 => Ok((1, 1)),
        TYPE_I64 | TYPE_F64 => Ok((1, 8)),
        TYPE_Q4_0 => Ok((32, 18)),
        TYPE_Q4_1 => Ok((32, 20)),
        TYPE_Q5_0 => Ok((32, 22)),
        TYPE_Q5_1 => Ok((32, 24)),
        TYPE_Q8_0 => Ok((32, 34)),
        TYPE_Q2_K => Ok((256, 84)),
        TYPE_Q3_K => Ok((256, 110)),
        TYPE_Q4_K => Ok((256, 144)),
        TYPE_Q5_K => Ok((256, 176)),
        TYPE_Q6_K => Ok((256, 210)),
        TYPE_Q8_K => Ok((256, 292)),
        _ => Err(invalid(format!("unsupported GGUF tensor type {kind}"))),
    }
}

pub(super) fn tensor_nbytes(kind: i32, ne: [i64; 4]) -> Result<usize> {
    let (blk_elems, blk_bytes) = block_info(kind)?;
    let ne0 = usize::try_from(ne[0]).map_err(|_| invalid("negative tensor dim0"))?;
    let ne1 = usize::try_from(ne[1])?;
    let ne2 = usize::try_from(ne[2])?;
    let ne3 = usize::try_from(ne[3])?;
    let rows = ne1
        .checked_mul(ne2)
        .and_then(|v| v.checked_mul(ne3))
        .ok_or_else(|| invalid("tensor row count overflow"))?;
    if !ne0.is_multiple_of(blk_elems) {
        return Err(invalid("tensor row width not multiple of block size"));
    }
    let blocks_per_row = ne0 / blk_elems;
    blocks_per_row
        .checked_mul(blk_bytes)
        .and_then(|v| v.checked_mul(rows))
        .ok_or_else(|| invalid("tensor byte size overflow"))
}

pub(super) fn decode_tensor_f32(kind: i32, ne: [i64; 4], raw: &[u8]) -> Result<Vec<f32>> {
    let ne0 = usize::try_from(ne[0])?;
    let ne1 = usize::try_from(ne[1])?;
    let ne2 = usize::try_from(ne[2])?;
    let ne3 = usize::try_from(ne[3])?;
    let total = ne0
        .checked_mul(ne1)
        .and_then(|v| v.checked_mul(ne2))
        .and_then(|v| v.checked_mul(ne3))
        .ok_or_else(|| invalid("tensor element count overflow"))?;
    if raw.is_empty() {
        return Ok(vec![0.0; total]);
    }
    match kind {
        TYPE_F32 => Ok(raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_bits(u32::from_le_bytes(*c)))
            .collect()),
        TYPE_F16 => Ok(raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| fp16_to_f32(u16::from_le_bytes(*c)))
            .collect()),
        TYPE_BF16 => Ok(raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| bf16_to_f32(u16::from_le_bytes(*c)))
            .collect()),
        TYPE_Q4_0 => decode_q4_0(raw, total),
        TYPE_Q8_0 => decode_q8_0(raw, total),
        _ => Ok(vec![0.0; total]),
    }
}

fn decode_q4_0(raw: &[u8], total: usize) -> Result<Vec<f32>> {
    let blocks = total / 32;
    if raw.len() < blocks * 18 {
        return Err(invalid("truncated Q4_0 tensor payload"));
    }
    let mut out = vec![0.0_f32; total];
    for b in 0..blocks {
        let base = b * 18;
        let d = fp16_to_f32(u16::from_le_bytes([raw[base], raw[base + 1]]));
        for j in 0..16 {
            let byte = raw[base + 2 + j];
            let q0 = i8::try_from(byte & 0x0f)? - 8;
            let q1 = i8::try_from(byte >> 4)? - 8;
            out[b * 32 + j] = f32::from(q0) * d;
            out[b * 32 + j + 16] = f32::from(q1) * d;
        }
    }
    Ok(out)
}

fn decode_q8_0(raw: &[u8], total: usize) -> Result<Vec<f32>> {
    let blocks = total / 32;
    if raw.len() < blocks * 34 {
        return Err(invalid("truncated Q8_0 tensor payload"));
    }
    let mut out = Vec::with_capacity(total);
    for b in 0..blocks {
        let base = b * 34;
        let d = fp16_to_f32(u16::from_le_bytes([raw[base], raw[base + 1]]));
        for j in 0..32 {
            let q = i8::from_le_bytes([raw[base + 2 + j]]);
            out.push(f32::from(q) * d);
        }
    }
    Ok(out)
}

#[derive(Clone)]
pub(super) struct TensorEntry {
    pub(super) name: String,
    pub(super) kind: i32,
    pub(super) ne: [i64; 4],
    pub(super) raw: Vec<u8>,
}

impl TensorEntry {
    pub(super) fn to_f32(&self) -> Result<Vec<f32>> {
        decode_tensor_f32(self.kind, self.ne, &self.raw)
    }

    pub(super) fn nbytes(&self) -> Result<usize> {
        if self.raw.is_empty() {
            tensor_nbytes(self.kind, self.ne)
        } else {
            Ok(self.raw.len())
        }
    }
}

#[derive(Clone)]
pub(super) enum ModelVocab {
    Full(Arc<ReferenceVocab>),
    Synthetic { count: usize, bos: i32, eos: i32 },
}

impl ModelVocab {
    pub(super) fn count(&self) -> usize {
        match self {
            Self::Full(v) => v.vocab_size(),
            Self::Synthetic { count, .. } => *count,
        }
    }

    pub(super) fn bos(&self) -> i32 {
        match self {
            Self::Full(v) => v.bos.unwrap_or(-1),
            Self::Synthetic { bos, .. } => *bos,
        }
    }

    pub(super) fn eos(&self) -> i32 {
        match self {
            Self::Full(v) => v.eos.unwrap_or(-1),
            Self::Synthetic { eos, .. } => *eos,
        }
    }

    pub(super) fn is_eog(&self, token: i32) -> bool {
        match self {
            Self::Full(v) => v.is_eog(token),
            Self::Synthetic { eos, .. } => token == *eos,
        }
    }

    pub(super) fn tokenize(
        &self,
        text: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<i32>> {
        match self {
            Self::Full(v) => v.tokenize(text.as_bytes(), add_special, parse_special),
            Self::Synthetic { count, bos, .. } => {
                let mut tokens = Vec::new();
                if add_special && *bos >= 0 {
                    tokens.push(*bos);
                }
                for byte in text.bytes() {
                    tokens.push(i32::from(byte) % i32::try_from((*count).max(1))?);
                }
                Ok(tokens)
            }
        }
    }

    pub(super) fn token_to_piece(&self, token: i32, lstrip: i32, special: bool) -> Result<Vec<u8>> {
        match self {
            Self::Full(v) => v.token_to_piece(token, lstrip, special),
            Self::Synthetic { count, .. } => {
                if token < 0 || usize::try_from(token)? >= *count {
                    return Err(invalid("token outside synthetic vocabulary"));
                }
                Ok(format!("tok_{token}").into_bytes())
            }
        }
    }

    pub(super) fn full_ref(&self) -> Result<&Arc<ReferenceVocab>> {
        match self {
            Self::Full(v) => Ok(v),
            Self::Synthetic { .. } => Err(invalid("model has no full GGUF vocabulary")),
        }
    }
}

#[expect(
    clippy::struct_excessive_bools,
    reason = "mirrors parsed GGUF architectural capability flags directly"
)]
#[derive(Clone)]
pub(super) struct ModelData {
    pub(super) kvs: BTreeMap<String, GgufValue>,
    pub(super) kv_order: Vec<String>,
    pub(super) tensors: BTreeMap<String, TensorEntry>,
    pub(super) tensor_order: Vec<String>,
    pub(super) vocab: Arc<ModelVocab>,
    pub(super) vocab_size: usize,
    pub(super) n_ctx_train: usize,
    pub(super) n_embd: usize,
    pub(super) n_layer: usize,
    pub(super) n_ff: usize,
    pub(super) n_head: usize,
    pub(super) n_head_kv: usize,
    pub(super) rms_eps: f32,
    pub(super) rope_freq_base: f32,
    pub(super) is_recurrent: bool,
    pub(super) is_hybrid: bool,
    pub(super) is_dspark: bool,
    pub(super) n_embd_cap: usize,
    pub(super) block_size: usize,
    pub(super) mask_token_id: i32,
    pub(super) markov_rank: usize,
    pub(super) graph_corrected: bool,
    pub(super) dspark_target_layers: Vec<i32>,
    pub(super) output_identity: u64,
    pub(super) output_buffer: Arc<Vec<f32>>,
    pub(super) head_source: Option<Arc<Self>>,
}

struct GgufReader<R> {
    inner: R,
}

impl<R: Read + Seek> GgufReader<R> {
    fn read_exact_bytes(&mut self, len: usize) -> Result<Vec<u8>> {
        if u64::try_from(len)? > MAX_STRING_BYTES {
            return Err(invalid("GGUF string/array exceeds bound"));
        }
        let mut buf = vec![0_u8; len];
        self.inner.read_exact(&mut buf)?;
        Ok(buf)
    }

    fn u8(&mut self) -> Result<u8> {
        let mut b = [0_u8; 1];
        self.inner.read_exact(&mut b)?;
        Ok(b[0])
    }

    fn u16(&mut self) -> Result<u16> {
        let mut b = [0_u8; 2];
        self.inner.read_exact(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }

    fn u32(&mut self) -> Result<u32> {
        let mut b = [0_u8; 4];
        self.inner.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    fn u64(&mut self) -> Result<u64> {
        let mut b = [0_u8; 8];
        self.inner.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u64()?;
        if len > MAX_STRING_BYTES {
            return Err(invalid("GGUF string length exceeds bound"));
        }
        let bytes = self.read_exact_bytes(usize::try_from(len)?)?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn value(&mut self, kind: u32, depth: usize) -> Result<GgufValue> {
        if depth > 8 {
            return Err(invalid("GGUF array nesting exceeds bound"));
        }
        match kind {
            0 => Ok(GgufValue::U8(self.u8()?)),
            1 => Ok(GgufValue::I8(i8::from_le_bytes([self.u8()?]))),
            2 => Ok(GgufValue::U16(self.u16()?)),
            3 => Ok(GgufValue::I16(i16::from_le_bytes(
                self.u16()?.to_le_bytes(),
            ))),
            4 => Ok(GgufValue::U32(self.u32()?)),
            5 => Ok(GgufValue::I32(i32::from_le_bytes(
                self.u32()?.to_le_bytes(),
            ))),
            6 => Ok(GgufValue::F32(f32::from_bits(self.u32()?))),
            7 => Ok(GgufValue::Bool(self.u8()? != 0)),
            8 => Ok(GgufValue::String(self.string()?)),
            9 => {
                let elem_kind = self.u32()?;
                let count = self.u64()?;
                if count > MAX_ARRAY_LEN {
                    return Err(invalid("GGUF array length exceeds bound"));
                }
                let mut items = Vec::with_capacity(usize::try_from(count)?);
                for _ in 0..count {
                    items.push(self.value(elem_kind, depth + 1)?);
                }
                Ok(GgufValue::Array(elem_kind, items))
            }
            10 => Ok(GgufValue::U64(self.u64()?)),
            11 => Ok(GgufValue::I64(i64::from_le_bytes(
                self.u64()?.to_le_bytes(),
            ))),
            12 => Ok(GgufValue::F64(f64::from_bits(self.u64()?))),
            _ => Err(invalid(format!("unknown GGUF value type {kind}"))),
        }
    }
}

fn write_gguf_string(out: &mut impl Write, value: &str) -> Result<()> {
    out.write_all(&u64::try_from(value.len())?.to_le_bytes())?;
    out.write_all(value.as_bytes())?;
    Ok(())
}

fn write_gguf_value(out: &mut impl Write, val: &GgufValue) -> Result<()> {
    match val {
        GgufValue::U8(v) => out.write_all(&[*v])?,
        GgufValue::I8(v) => out.write_all(&v.to_le_bytes())?,
        GgufValue::U16(v) => out.write_all(&v.to_le_bytes())?,
        GgufValue::I16(v) => out.write_all(&v.to_le_bytes())?,
        GgufValue::U32(v) => out.write_all(&v.to_le_bytes())?,
        GgufValue::I32(v) => out.write_all(&v.to_le_bytes())?,
        GgufValue::F32(v) => out.write_all(&v.to_bits().to_le_bytes())?,
        GgufValue::Bool(v) => out.write_all(&[u8::from(*v)])?,
        GgufValue::String(s) => write_gguf_string(out, s)?,
        GgufValue::Array(elem_kind, items) => {
            out.write_all(&elem_kind.to_le_bytes())?;
            out.write_all(&u64::try_from(items.len())?.to_le_bytes())?;
            for item in items {
                write_gguf_value(out, item)?;
            }
        }
        GgufValue::U64(v) => out.write_all(&v.to_le_bytes())?,
        GgufValue::I64(v) => out.write_all(&v.to_le_bytes())?,
        GgufValue::F64(v) => out.write_all(&v.to_bits().to_le_bytes())?,
    }
    Ok(())
}

fn kv_usize(kvs: &BTreeMap<String, GgufValue>, arch: &str, key: &str, default: usize) -> usize {
    let full = key.replace("%s", arch);
    kvs.get(&full)
        .and_then(GgufValue::as_u64_coerce)
        .and_then(|v| usize::try_from(v).ok())
        .filter(|&v| v > 0)
        .unwrap_or(default)
}

fn kv_f32(kvs: &BTreeMap<String, GgufValue>, arch: &str, key: &str, default: f32) -> f32 {
    let full = key.replace("%s", arch);
    kvs.get(&full)
        .and_then(GgufValue::as_f32_coerce)
        .filter(|v| v.is_finite())
        .unwrap_or(default)
}

#[derive(Clone)]
struct ParsedGguf {
    kvs: BTreeMap<String, GgufValue>,
    kv_order: Vec<String>,
    tensors: BTreeMap<String, TensorEntry>,
    tensor_order: Vec<String>,
}

type GgufTensorSpec = (String, [i64; 4], i32, u64);
type LoadedTensors = (BTreeMap<String, TensorEntry>, Vec<String>);

fn read_gguf_tensors(
    reader: &mut (impl Read + Seek),
    specs: &[GgufTensorSpec],
    data_start: u64,
    read_weights: bool,
    mut progress: Option<&mut dyn FnMut(f32) -> bool>,
) -> Result<Option<LoadedTensors>> {
    let mut tensors = BTreeMap::new();
    let mut tensor_order = Vec::with_capacity(specs.len());
    for (idx, (name, ne, kind, offset)) in specs.iter().enumerate() {
        let raw = if read_weights && !ne.contains(&0) {
            let nbytes = tensor_nbytes(*kind, *ne)?;
            if nbytes > MAX_TENSOR_BYTES {
                return Err(invalid("GGUF tensor exceeds 64 MiB bound"));
            }
            reader.seek(SeekFrom::Start(data_start + offset))?;
            let mut buf = vec![0_u8; nbytes];
            reader.read_exact(&mut buf)?;
            buf
        } else {
            Vec::new()
        };
        if !tensors.contains_key(name) {
            tensor_order.push(name.clone());
        }
        tensors.insert(
            name.clone(),
            TensorEntry {
                name: name.clone(),
                kind: *kind,
                ne: *ne,
                raw,
            },
        );
        if let Some(cb) = progress.as_deref_mut()
            && !specs.is_empty()
            && (idx + 1 == specs.len() || (idx + 1) * 4 % specs.len() < 4)
        {
            let ratio = f64::from(u32::try_from(idx + 1)?) / f64::from(u32::try_from(specs.len())?);
            let frac = 0.75_f32.mul_add(f64_to_f32(ratio), 0.25);
            if !cb(frac) {
                return Ok(None);
            }
        }
    }
    if specs.is_empty()
        && let Some(cb) = progress
        && !cb(1.0)
    {
        return Ok(None);
    }
    Ok(Some((tensors, tensor_order)))
}

fn parse_gguf_stream(
    mut reader: impl Read + Seek,
    read_weights: bool,
    mut progress: Option<&mut dyn FnMut(f32) -> bool>,
) -> Result<Option<ParsedGguf>> {
    if let Some(cb) = progress.as_deref_mut()
        && !cb(0.0)
    {
        return Ok(None);
    }
    let mut r = GgufReader { inner: &mut reader };
    if r.read_exact_bytes(4)? != b"GGUF" {
        return Err(invalid("invalid GGUF magic header"));
    }
    let version = r.u32()?;
    if !matches!(version, 2 | 3) {
        return Err(invalid(format!("unsupported GGUF version {version}")));
    }
    let tensor_count = r.u64()?;
    let kv_count = r.u64()?;
    if tensor_count > MAX_TENSOR_COUNT || kv_count > MAX_KV_COUNT {
        return Err(invalid("GGUF header counts exceed bound"));
    }
    let mut kvs = BTreeMap::new();
    let mut kv_order = Vec::with_capacity(usize::try_from(kv_count)?);
    for _ in 0..kv_count {
        let key = r.string()?;
        let kind = r.u32()?;
        let val = r.value(kind, 0)?;
        if !kvs.contains_key(&key) {
            kv_order.push(key.clone());
        }
        kvs.insert(key, val);
    }
    if let Some(cb) = progress.as_deref_mut()
        && !cb(0.25)
    {
        return Ok(None);
    }
    let alignment = kvs
        .get("general.alignment")
        .and_then(GgufValue::as_u64_coerce)
        .filter(|&a| a > 0 && a.is_power_of_two())
        .unwrap_or(32);
    let mut specs = Vec::with_capacity(usize::try_from(tensor_count)?);
    for _ in 0..tensor_count {
        let name = r.string()?;
        let n_dims = usize::try_from(r.u32()?)?;
        if n_dims > 4 {
            return Err(invalid("GGUF tensor has >4 dims"));
        }
        let mut ne = [1_i64; 4];
        for slot in ne.iter_mut().take(n_dims) {
            *slot = i64::try_from(r.u64()?)?;
        }
        let kind = i32::try_from(r.u32()?)?;
        let offset = r.u64()?;
        specs.push((name, ne, kind, offset));
    }
    let header_end = reader.stream_position()?;
    let data_start = header_end.div_ceil(alignment) * alignment;
    let Some((tensors, tensor_order)) =
        read_gguf_tensors(&mut reader, &specs, data_start, read_weights, progress)?
    else {
        return Ok(None);
    };
    Ok(Some(ParsedGguf {
        kvs,
        kv_order,
        tensors,
        tensor_order,
    }))
}

fn validate_dspark_logsnr(kvs: &BTreeMap<String, GgufValue>, arch: &str) -> Result<()> {
    let key = format!("{arch}.dspark.log_snr_conditioning");
    if kvs.get(&key).and_then(GgufValue::as_bool_coerce) == Some(true) {
        let min_v = kvs
            .get(&format!("{arch}.dspark.min_log_snr"))
            .and_then(GgufValue::as_f32_coerce)
            .ok_or_else(|| invalid("dspark.min_log_snr missing"))?;
        let max_v = kvs
            .get(&format!("{arch}.dspark.max_log_snr"))
            .and_then(GgufValue::as_f32_coerce)
            .ok_or_else(|| invalid("dspark.max_log_snr missing"))?;
        if !min_v.is_finite() || !max_v.is_finite() || min_v >= max_v {
            return Err(invalid("invalid dspark log-SNR bounds"));
        }
    }
    Ok(())
}

fn infer_arch_flags(arch: &str, kvs: &BTreeMap<String, GgufValue>) -> (bool, bool, bool) {
    let is_dspark = arch == "dspark";
    let has_ssm = kvs.contains_key(&format!("{arch}.ssm.inner_size"))
        || kvs.contains_key(&format!("{arch}.wkv.head_size"))
        || kvs.contains_key(&format!("{arch}.kda.head_dim"));
    let is_hybrid = matches!(
        arch,
        "jamba"
            | "lfm2"
            | "bamba"
            | "qwen3next"
            | "qwen35"
            | "nemotron_h"
            | "nemotron_h_moe"
            | "plamo2"
            | "granitehybrid"
    ) || (has_ssm && kvs.contains_key(&format!("{arch}.full_attention_interval")));
    let is_recurrent = matches!(
        arch,
        "mamba" | "mamba2" | "rwkv6" | "rwkv7" | "rwkv6qwen2" | "rwkv7qwen2" | "deepseek4"
    ) || (has_ssm && !is_hybrid);
    (is_dspark, is_hybrid, is_recurrent)
}

impl ModelData {
    fn from_parsed(
        parsed: ParsedGguf,
        vocab: Arc<ModelVocab>,
        head_source: Option<Arc<Self>>,
    ) -> Result<Self> {
        let arch = parsed
            .kvs
            .get("general.architecture")
            .and_then(GgufValue::as_str)
            .unwrap_or("llama")
            .to_owned();
        validate_dspark_logsnr(&parsed.kvs, &arch)?;
        let vocab_size = kv_usize(&parsed.kvs, &arch, "%s.vocab_size", vocab.count().max(128));
        let n_embd = kv_usize(&parsed.kvs, &arch, "%s.embedding_length", 64);
        let n_layer = kv_usize(&parsed.kvs, &arch, "%s.block_count", 2);
        let n_ff = kv_usize(&parsed.kvs, &arch, "%s.feed_forward_length", n_embd * 2);
        let n_head = kv_usize(&parsed.kvs, &arch, "%s.attention.head_count", 2);
        let n_head_kv = kv_usize(&parsed.kvs, &arch, "%s.attention.head_count_kv", n_head);
        let n_ctx_train = kv_usize(&parsed.kvs, &arch, "%s.context_length", 128);
        let rms_eps = kv_f32(
            &parsed.kvs,
            &arch,
            "%s.attention.layer_norm_rms_epsilon",
            1e-5,
        );
        let rope_freq_base = kv_f32(&parsed.kvs, &arch, "%s.rope.freq_base", 10_000.0);
        let (is_dspark, is_hybrid, is_recurrent) = infer_arch_flags(&arch, &parsed.kvs);
        let n_embd_cap = parsed
            .kvs
            .get(&format!("{arch}.dspark.correction_size"))
            .and_then(GgufValue::as_u64_coerce)
            .and_then(|v| usize::try_from(v).ok())
            .or_else(|| {
                parsed
                    .tensors
                    .get("dspark.fc.weight")
                    .map(|t| usize::try_from(t.ne[0]).unwrap_or(128))
            })
            .unwrap_or(128);
        let block_size = kv_usize(&parsed.kvs, &arch, "%s.dspark.block_size", 7);
        let mask_token_id = parsed
            .kvs
            .get(&format!("{arch}.dspark.mask_token_id"))
            .and_then(GgufValue::as_i32_coerce)
            .unwrap_or(0);
        let markov_rank = parsed
            .kvs
            .get(&format!("{arch}.dspark.markov_rank"))
            .and_then(GgufValue::as_u64_coerce)
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(0);
        let graph_corrected = parsed
            .kvs
            .get(&format!("{arch}.dspark.hidden_correction"))
            .and_then(GgufValue::as_bool_coerce)
            .unwrap_or(false);
        let dspark_target_layers = parsed
            .kvs
            .get(&format!("{arch}.dspark.target_layers"))
            .map_or_default(GgufValue::as_i32_vec);
        let (output_identity, output_buffer) = if let Some(ref src) = head_source {
            (src.output_identity, Arc::clone(&src.output_buffer))
        } else {
            let vec = parsed
                .tensors
                .get("output.weight")
                .or_else(|| parsed.tensors.get("token_embd.weight"))
                .map_or_else(|| Ok(Vec::new()), TensorEntry::to_f32)?;
            (
                NEXT_OUTPUT_ID.fetch_add(1, Ordering::Relaxed),
                Arc::new(vec),
            )
        };
        Ok(Self {
            kvs: parsed.kvs,
            kv_order: parsed.kv_order,
            tensors: parsed.tensors,
            tensor_order: parsed.tensor_order,
            vocab,
            vocab_size,
            n_ctx_train,
            n_embd,
            n_layer,
            n_ff,
            n_head: n_head.max(1),
            n_head_kv: n_head_kv.max(1),
            rms_eps,
            rope_freq_base,
            is_recurrent,
            is_hybrid,
            is_dspark,
            n_embd_cap,
            block_size,
            mask_token_id,
            markov_rank,
            graph_corrected,
            dspark_target_layers,
            output_identity,
            output_buffer,
            head_source,
        })
    }

    pub(super) fn load_from_file(
        path: &Path,
        read_weights: bool,
        head_source: Option<Arc<Self>>,
        progress: Option<&mut dyn FnMut(f32) -> bool>,
    ) -> Result<Option<Arc<Self>>> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        if len > MAX_GGUF_BYTES {
            return Err(invalid("GGUF file exceeds 128 MiB bound"));
        }
        let Some(parsed) = parse_gguf_stream(BufReader::new(file), read_weights, progress)? else {
            return Ok(None);
        };
        let vocab = build_vocab_from_parsed(Some(path), &parsed.kvs)?;
        Ok(Some(Arc::new(Self::from_parsed(
            parsed,
            vocab,
            head_source,
        )?)))
    }

    pub(super) fn init_from_user(
        metadata_path: &Path,
        session_dir: &Path,
        read_weights: bool,
        mut callback: Option<super::sampler::SamplerEventCb<'_>>,
    ) -> Result<Arc<Self>> {
        let file = File::open(metadata_path)?;
        let Some(mut parsed) = parse_gguf_stream(BufReader::new(file), false, None)? else {
            return Err(invalid("unexpected cancellation in user metadata load"));
        };
        let vocab = build_vocab_from_parsed(Some(metadata_path), &parsed.kvs)?;
        let base = Self::from_parsed(parsed.clone(), Arc::clone(&vocab), None)?;
        if read_weights {
            let specs = user_tensor_specs(&base);
            parsed.tensors.clear();
            parsed.tensor_order.clear();
            let canon_root = fs::canonicalize(session_dir)?;
            for (name, ne) in specs {
                let bytes = tensor_nbytes(TYPE_F32, ne)?;
                let spec_json = json!({
                    "Name": name,
                    "Type": TYPE_F32,
                    "Dimensions": [ne[0], ne[1], ne[2], ne[3]],
                    "Bytes": bytes,
                });
                let raw = if let Some(cb) = callback.as_deref_mut() {
                    let reply = cb(&spec_json)?;
                    read_callback_tensor_bytes(&reply, &canon_root, ne, bytes)?
                } else {
                    vec![0_u8; bytes]
                };
                parsed.tensor_order.push(name.clone());
                parsed.tensors.insert(
                    name.clone(),
                    TensorEntry {
                        name,
                        kind: TYPE_F32,
                        ne,
                        raw,
                    },
                );
            }
        }
        Ok(Arc::new(Self::from_parsed(parsed, vocab, None)?))
    }

    pub(super) fn save_to_writer(&self, writer: &mut impl Write) -> Result<()> {
        writer.write_all(b"GGUF")?;
        writer.write_all(&3_u32.to_le_bytes())?;
        writer.write_all(&u64::try_from(self.tensor_order.len())?.to_le_bytes())?;
        writer.write_all(&u64::try_from(self.kv_order.len())?.to_le_bytes())?;
        let mut header_len = 24_usize;
        for key in &self.kv_order {
            if let Some(val) = self.kvs.get(key) {
                let mut kv_buf = Vec::new();
                write_gguf_string(&mut kv_buf, key)?;
                kv_buf.extend_from_slice(&val.type_id().to_le_bytes());
                write_gguf_value(&mut kv_buf, val)?;
                header_len += kv_buf.len();
                writer.write_all(&kv_buf)?;
            }
        }
        let mut data_offset = 0_u64;
        for name in &self.tensor_order {
            let entry = self
                .tensors
                .get(name)
                .ok_or_else(|| invalid(format!("missing tensor {name} during save")))?;
            let mut tb = Vec::new();
            write_gguf_string(&mut tb, &entry.name)?;
            tb.extend_from_slice(&4_u32.to_le_bytes());
            for d in entry.ne {
                tb.extend_from_slice(&u64::try_from(d)?.to_le_bytes());
            }
            tb.extend_from_slice(&u32::try_from(entry.kind)?.to_le_bytes());
            tb.extend_from_slice(&data_offset.to_le_bytes());
            header_len += tb.len();
            writer.write_all(&tb)?;
            let padded = u64::try_from(entry.raw.len())?.div_ceil(32) * 32;
            data_offset += padded;
        }
        let aligned_hdr = header_len.div_ceil(32) * 32;
        writer.write_all(&vec![0_u8; aligned_hdr - header_len])?;
        for name in &self.tensor_order {
            let entry = &self.tensors[name];
            writer.write_all(&entry.raw)?;
            let padded = entry.raw.len().div_ceil(32) * 32;
            writer.write_all(&vec![0_u8; padded - entry.raw.len()])?;
        }
        writer.flush()?;
        Ok(())
    }

    pub(super) fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        if (name == "output.weight" || name == "token_embd.weight")
            && let Some(ref src) = self.head_source
            && name == "output.weight"
        {
            return src
                .tensor_f32("output.weight")
                .or_else(|_| src.tensor_f32("token_embd.weight"));
        }
        let entry = self
            .tensors
            .get(name)
            .ok_or_else(|| invalid(format!("missing model tensor {name}")))?;
        entry.to_f32()
    }
}

fn build_vocab_from_parsed(
    path: Option<&Path>,
    kvs: &BTreeMap<String, GgufValue>,
) -> Result<Arc<ModelVocab>> {
    if kvs.contains_key("tokenizer.ggml.tokens")
        && let Some(p) = path
    {
        return Ok(Arc::new(ModelVocab::Full(load_gguf_vocab(p)?)));
    }
    let arch = kvs
        .get("general.architecture")
        .and_then(GgufValue::as_str)
        .unwrap_or("llama");
    let count = kv_usize(kvs, arch, "%s.vocab_size", 128);
    let bos = kvs
        .get("tokenizer.ggml.bos_token_id")
        .and_then(GgufValue::as_i32_coerce)
        .unwrap_or(1);
    let eos = kvs
        .get("tokenizer.ggml.eos_token_id")
        .and_then(GgufValue::as_i32_coerce)
        .unwrap_or(2);
    Ok(Arc::new(ModelVocab::Synthetic { count, bos, eos }))
}

fn user_tensor_specs(model: &ModelData) -> Vec<(String, [i64; 4])> {
    let embd = i64::try_from(model.n_embd).unwrap_or(64);
    let vocab = i64::try_from(model.vocab_size).unwrap_or(128);
    let ff = i64::try_from(model.n_ff).unwrap_or(embd * 2);
    let layers = model.n_layer.min(32);
    let mut specs = vec![
        ("token_embd.weight".into(), [embd, vocab, 1, 1]),
        ("output_norm.weight".into(), [embd, 1, 1, 1]),
        ("output.weight".into(), [embd, vocab, 1, 1]),
    ];
    for layer in 0..layers {
        specs.push((format!("blk.{layer}.attn_norm.weight"), [embd, 1, 1, 1]));
        specs.push((format!("blk.{layer}.attn_q.weight"), [embd, embd, 1, 1]));
        specs.push((format!("blk.{layer}.attn_k.weight"), [embd, embd, 1, 1]));
        specs.push((format!("blk.{layer}.attn_v.weight"), [embd, embd, 1, 1]));
        specs.push((
            format!("blk.{layer}.attn_output.weight"),
            [embd, embd, 1, 1],
        ));
        specs.push((format!("blk.{layer}.ffn_norm.weight"), [embd, 1, 1, 1]));
        specs.push((format!("blk.{layer}.ffn_gate.weight"), [embd, ff, 1, 1]));
        specs.push((format!("blk.{layer}.ffn_up.weight"), [embd, ff, 1, 1]));
        specs.push((format!("blk.{layer}.ffn_down.weight"), [ff, embd, 1, 1]));
    }
    specs
}

pub(super) fn checked_session_file(
    raw: &str,
    session_dir: &Path,
    must_exist: bool,
) -> Result<PathBuf> {
    let p = Path::new(raw);
    if !p.is_absolute() {
        return Err(invalid("session file path must be absolute"));
    }
    let root = fs::canonicalize(session_dir)?;
    let resolved = if must_exist {
        fs::canonicalize(p)?
    } else {
        let parent = p
            .parent()
            .ok_or_else(|| invalid("session file missing parent"))?;
        let canon_parent = fs::canonicalize(parent)?;
        canon_parent.join(
            p.file_name()
                .ok_or_else(|| invalid("session file missing name"))?,
        )
    };
    if !resolved.starts_with(&root) {
        return Err(invalid("session file escaped session directory"));
    }
    Ok(resolved)
}

fn read_callback_tensor_bytes(
    reply: &Value,
    canon_root: &Path,
    ne: [i64; 4],
    expected_bytes: usize,
) -> Result<Vec<u8>> {
    let encoding = reply["Encoding"]
        .as_str()
        .ok_or_else(|| invalid("tensor callback Encoding missing"))?;
    let length = usize::try_from(
        reply["Length"]
            .as_u64()
            .ok_or_else(|| invalid("tensor callback Length missing"))?,
    )?;
    let file_str = reply["DataFile"]
        .as_str()
        .ok_or_else(|| invalid("tensor callback DataFile missing"))?;
    let path = checked_session_file(file_str, canon_root, true)?;
    if path.parent() != Some(canon_root) {
        return Err(invalid(
            "generated input file must be directly in session directory",
        ));
    }
    if encoding != "float32-le" && encoding != "tensor-le" {
        return Err(invalid("unsupported generated tensor encoding"));
    }
    if length != expected_bytes || usize::try_from(fs::metadata(&path)?.len())? != length {
        return Err(invalid("generated tensor byte extent differs"));
    }
    let _ = ne;
    Ok(fs::read(path)?)
}

pub(super) fn vocab_type_from_model(vocab: &ModelVocab) -> i32 {
    match vocab {
        ModelVocab::Full(v) => match v.vocab_type() {
            VocabType::None => 0,
            VocabType::Spm => 1,
            VocabType::Bpe => 2,
            VocabType::Wpm => 3,
            VocabType::Ugm => 4,
            VocabType::Rwkv => 5,
            VocabType::Plamo2 => 6,
        },
        ModelVocab::Synthetic { .. } => 0,
    }
}
