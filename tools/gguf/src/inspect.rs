// Keep the private contract explicit if the parent is reorganized later.
#![allow(
    clippy::redundant_pub_crate,
    reason = "Keep explicit internal visibility if the parent module is reorganized."
)]

use crate::{Metadata, MetadataArray, MetadataValue};
use crate::{PRISM_DIALECT, Result};
#[cfg(feature = "report")]
use serde_json::{Map, Value, json};
#[cfg(feature = "report")]
use sha2::{Digest, Sha256};
#[cfg(feature = "report")]
use std::collections::BTreeMap;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

const HEADER_LIMIT: u64 = 256 * 1024 * 1024;
const STRING_LIMIT: u64 = 8 * 1024 * 1024;
const ARRAY_LIMIT: u64 = 2 * 1024 * 1024;
const TOTAL_ELEMENT_LIMIT: u64 = 8 * 1024 * 1024;
const BUFFER_BYTES: usize = 65_536;

fn bad(message: impl Into<String>) -> crate::Error {
    crate::bad(format!("GGUF: {}", message.into()))
}

/// A validated tensor descriptor in GGUF/ggml dimension order.
#[derive(Debug)]
pub struct TensorDescriptor {
    /// Unique tensor name from the GGUF descriptor.
    pub name: String,
    /// Nonzero extents, with the contiguous row dimension first.
    pub dimensions: Vec<u64>,
    /// GGUF tensor encoding ID, including explicitly pinned Prism IDs.
    pub type_id: u32,
    /// Byte offset relative to the validated tensor data region.
    pub offset: u64,
    /// Exact serialized payload length implied by dimensions and encoding.
    pub bytes: u64,
    /// Product of every tensor dimension.
    pub elements: u64,
    #[cfg(feature = "report")]
    pub(crate) type_position: u64,
}

pub(crate) struct Inspection {
    #[cfg(feature = "report")]
    pub report: Value,
    pub tensors: Vec<TensorDescriptor>,
    #[cfg(feature = "report")]
    pub file_type_position: Option<u64>,
    pub metadata: Metadata,
    pub retained_bytes: usize,
    pub data_offset: u64,
}

#[derive(Clone, Copy)]
struct Kind {
    #[cfg(feature = "report")]
    name: &'static str,
    block: u64,
    bytes: u64,
}

const fn kind(id: u32, prism: bool) -> Option<Kind> {
    let (name, block, bytes) = match id {
        0 => ("F32", 1, 4),
        1 => ("F16", 1, 2),
        2 => ("Q4_0", 32, 18),
        3 => ("Q4_1", 32, 20),
        6 => ("Q5_0", 32, 22),
        7 => ("Q5_1", 32, 24),
        8 => ("Q8_0", 32, 34),
        9 => ("Q8_1", 32, 36),
        10 => ("Q2_K", 256, 84),
        11 => ("Q3_K", 256, 110),
        12 => ("Q4_K", 256, 144),
        13 => ("Q5_K", 256, 176),
        14 => ("Q6_K", 256, 210),
        15 => ("Q8_K", 256, 292),
        16 => ("IQ2_XXS", 256, 66),
        17 => ("IQ2_XS", 256, 74),
        18 => ("IQ3_XXS", 256, 98),
        19 => ("IQ1_S", 256, 50),
        20 => ("IQ4_NL", 32, 18),
        21 => ("IQ3_S", 256, 110),
        22 => ("IQ2_S", 256, 82),
        23 => ("IQ4_XS", 256, 136),
        24 => ("I8", 1, 1),
        25 => ("I16", 1, 2),
        26 => ("I32", 1, 4),
        27 => ("I64", 1, 8),
        28 => ("F64", 1, 8),
        29 => ("IQ1_M", 256, 56),
        30 => ("BF16", 1, 2),
        34 => ("TQ1_0", 256, 54),
        35 => ("TQ2_0", 256, 66),
        39 => ("MXFP4", 32, 17),
        40 => ("NVFP4", 64, 36),
        41 => ("Q1_0", 128, 18),
        42 => ("Q2_0", 64, 18),
        142 if prism => ("PQ2_0", 128, 34),
        143 if prism => ("PTQ1_0", 128, 28),
        _ => return None,
    };
    #[cfg(not(feature = "report"))]
    let _ = name;
    Some(Kind {
        #[cfg(feature = "report")]
        name,
        block,
        bytes,
    })
}

fn overflow() -> crate::Error {
    bad("descriptor/count arithmetic overflow.")
}

fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).ok_or_else(overflow)
}

fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b).ok_or_else(overflow)
}

const fn width(type_id: u32) -> Option<u64> {
    match type_id {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

struct Parser<'a> {
    // One fixed buffer serves scalar reads, skipped booleans and report hashes.
    // Every seek goes through this reader, preserving its logical file cursor.
    reader: BufReader<&'a mut File>,
    // Reads and skips advance this cursor; hashing restores the file cursor.
    // Tracking it avoids an OS position query for every scalar/limit check.
    position: u64,
    file_bytes: u64,
    elements: u64,
    array_summary: Option<(u32, u64)>,
    retain_budget: Option<usize>,
}

impl Parser<'_> {
    fn charge(&mut self, bytes: usize) -> Result<()> {
        if let Some(remaining) = &mut self.retain_budget {
            *remaining = remaining
                .checked_sub(bytes)
                .ok_or_else(|| bad("retained model metadata exceeds memory budget."))?;
        }
        Ok(())
    }

    const fn position(&self) -> u64 {
        self.position
    }

    fn need(&self, bytes: u64) -> Result<()> {
        let position = self.position();
        if bytes > self.file_bytes.saturating_sub(position)
            || bytes > HEADER_LIMIT.saturating_sub(position)
        {
            return Err(bad(
                "truncated or over-limit metadata/tensor descriptor region.",
            ));
        }
        Ok(())
    }

    fn read<const N: usize>(&mut self) -> Result<[u8; N]> {
        self.need(u64::try_from(N)?)?;
        let mut bytes = [0; N];
        self.reader.read_exact(&mut bytes)?;
        self.position = add(self.position, u64::try_from(N)?)?;
        Ok(bytes)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.read()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.read()?))
    }

    fn text(&mut self, limit: u64) -> Result<String> {
        let size = self.u64()?;
        if size > limit {
            return Err(bad("name/string exceeds inspection limit."));
        }
        self.need(size)?;
        self.read_text(usize::try_from(size)?)
    }

    fn read_text(&mut self, size: usize) -> Result<String> {
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(size)?;
        let available = self.reader.fill_buf()?;
        if let Some(text) = available.get(..size) {
            bytes.extend_from_slice(text);
            self.reader.consume(size);
        } else {
            bytes.resize(size, 0);
            self.reader.read_exact(&mut bytes)?;
        }
        self.position = add(self.position, u64::try_from(size)?)?;
        Ok(String::from_utf8(bytes)?)
    }

    fn skip(&mut self, bytes: u64) -> Result<()> {
        let position = add(self.position(), bytes)?;
        self.reader.seek_relative(i64::try_from(bytes)?)?;
        self.position = position;
        Ok(())
    }

    fn scalar(&mut self, type_id: u32) -> Result<MetadataValue> {
        Ok(match type_id {
            0 => MetadataValue::U8(self.read::<1>()?[0]),
            1 => MetadataValue::I8(i8::from_le_bytes(self.read()?)),
            2 => MetadataValue::U16(u16::from_le_bytes(self.read()?)),
            3 => MetadataValue::I16(i16::from_le_bytes(self.read()?)),
            4 => MetadataValue::U32(self.u32()?),
            5 => MetadataValue::I32(i32::from_le_bytes(self.read()?)),
            6 => MetadataValue::F32(self.f32()?),
            7 => MetadataValue::Bool(self.boolean()?),
            10 => MetadataValue::U64(self.u64()?),
            11 => MetadataValue::I64(i64::from_le_bytes(self.read()?)),
            12 => MetadataValue::F64(self.f64()?),
            _ => return Err(bad("unknown scalar type.")),
        })
    }

    fn f32(&mut self) -> Result<f32> {
        let value = f32::from_le_bytes(self.read()?);
        if !value.is_finite() {
            return Err(bad("non-finite float metadata."));
        }
        Ok(value)
    }

    fn f64(&mut self) -> Result<f64> {
        let value = f64::from_le_bytes(self.read()?);
        if !value.is_finite() {
            return Err(bad("non-finite float metadata."));
        }
        Ok(value)
    }

    fn boolean(&mut self) -> Result<bool> {
        match self.read::<1>()?[0] {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(bad("invalid boolean.")),
        }
    }

    fn array_values<T>(
        &mut self,
        count: u64,
        mut read: impl FnMut(&mut Self) -> Result<T>,
    ) -> Result<Vec<T>> {
        let count = usize::try_from(count)?;
        self.charge(count.checked_mul(size_of::<T>()).ok_or_else(overflow)?)?;
        let mut values = Vec::new();
        values.try_reserve_exact(count)?;
        for _ in 0..count {
            values.push(read(self)?);
        }
        Ok(values)
    }

    fn booleans(&mut self, mut count: u64) -> Result<()> {
        self.need(count)?;
        while count > 0 {
            let available = self.reader.fill_buf()?;
            if available.is_empty() {
                return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof).into());
            }
            let size = available.len().min(usize::try_from(count)?);
            if available[..size].iter().any(|value| *value > 1) {
                return Err(bad("invalid boolean."));
            }
            self.reader.consume(size);
            self.position = add(self.position, u64::try_from(size)?)?;
            count -= u64::try_from(size)?;
        }
        Ok(())
    }

    fn value(&mut self, type_id: u32, depth: u32, retain: bool) -> Result<Option<MetadataValue>> {
        if depth > 4 {
            return Err(bad("array nesting exceeds inspection limit."));
        }
        if type_id == 8 {
            let size = self.u64()?;
            if size > STRING_LIMIT {
                return Err(bad("string exceeds 8 MiB inspection limit."));
            }
            self.need(size)?;
            if retain && self.retain_budget.is_some() {
                let size = usize::try_from(size)?;
                self.charge(size)?;
                return Ok(Some(MetadataValue::String(self.read_text(size)?)));
            }
            if retain && size <= u64::try_from(BUFFER_BYTES)? {
                return Ok(Some(MetadataValue::String(
                    self.read_text(usize::try_from(size)?)?,
                )));
            }
            self.skip(size)?;
            return Ok(None);
        }
        if type_id != 9 {
            return Ok(Some(self.scalar(type_id)?));
        }
        let item_type = self.u32()?;
        if item_type > 12 {
            return Err(bad("unknown array element type."));
        }
        let count = self.u64()?;
        if count > ARRAY_LIMIT || count > TOTAL_ELEMENT_LIMIT - self.elements {
            return Err(bad("array element limit exceeded."));
        }
        self.elements += count;
        let keep = retain
            && (self.retain_budget.is_some()
                || (count <= 4096 && item_type != 8 && item_type != 9));
        if depth == 0 {
            self.array_summary = Some((item_type, count));
        }
        if let Some(width) = width(item_type)
            && !keep
            && item_type != 7
        {
            let bytes = mul(count, width)?;
            self.need(bytes)?;
            self.skip(bytes)?;
            return Ok(None);
        }
        if !keep && item_type == 7 {
            if count > 0 && depth == 4 {
                return Err(bad("array nesting exceeds inspection limit."));
            }
            self.booleans(count)?;
            return Ok(None);
        }
        if count > 0 && depth == 4 {
            return Err(bad("array nesting exceeds inspection limit."));
        }
        if !keep {
            for _ in 0..count {
                self.value(item_type, depth + 1, false)?;
            }
            return Ok(None);
        }
        self.retained_array(item_type, count, depth)
    }

    fn retained_array(
        &mut self,
        item_type: u32,
        count: u64,
        depth: u32,
    ) -> Result<Option<MetadataValue>> {
        let values = match item_type {
            0 => MetadataArray::U8(self.array_values(count, |parser| Ok(parser.read::<1>()?[0]))?),
            1 => MetadataArray::I8(
                self.array_values(count, |parser| Ok(i8::from_le_bytes(parser.read()?)))?,
            ),
            2 => MetadataArray::U16(
                self.array_values(count, |parser| Ok(u16::from_le_bytes(parser.read()?)))?,
            ),
            3 => MetadataArray::I16(
                self.array_values(count, |parser| Ok(i16::from_le_bytes(parser.read()?)))?,
            ),
            4 => MetadataArray::U32(self.array_values(count, Self::u32)?),
            5 => MetadataArray::I32(
                self.array_values(count, |parser| Ok(i32::from_le_bytes(parser.read()?)))?,
            ),
            6 => MetadataArray::F32(self.array_values(count, Self::f32)?),
            7 => MetadataArray::Bool(self.array_values(count, Self::boolean)?),
            10 => MetadataArray::U64(self.array_values(count, Self::u64)?),
            11 => MetadataArray::I64(
                self.array_values(count, |parser| Ok(i64::from_le_bytes(parser.read()?)))?,
            ),
            12 => MetadataArray::F64(self.array_values(count, Self::f64)?),
            8 => {
                MetadataArray::String(self.array_values(count, |parser| {
                    match parser.value(8, depth + 1, true)? {
                        Some(MetadataValue::String(value)) => Ok(value),
                        _ => Err(bad("retained string is required.")),
                    }
                })?)
            }
            9 => MetadataArray::Array(self.array_values(count, |parser| {
                match parser.value(9, depth + 1, true)? {
                    Some(MetadataValue::Array(value)) => Ok(value),
                    _ => Err(bad("retained array is required.")),
                }
            })?),
            _ => return Err(bad("unknown array element type.")),
        };
        Ok(Some(MetadataValue::Array(values)))
    }

    #[cfg(feature = "report")]
    fn hash_record(&mut self, hash: &mut Sha256, start: u64, length: u64) -> Result<()> {
        let restore = self.position();
        self.reader.seek(SeekFrom::Start(start))?;
        let result = (|| -> Result<()> {
            let mut remaining = length;
            while remaining > 0 {
                let available = self.reader.fill_buf()?;
                if available.is_empty() {
                    return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof).into());
                }
                let size = available.len().min(usize::try_from(remaining)?);
                hash.update(&available[..size]);
                self.reader.consume(size);
                remaining -= u64::try_from(size)?;
            }
            Ok(())
        })();
        self.reader.seek(SeekFrom::Start(restore))?;
        result
    }
}

#[cfg(feature = "report")]
fn report_value(value: &MetadataValue) -> Result<Value> {
    Ok(match value {
        // Preserve the established JSON schema's shortest binary32 decimal,
        // rather than exposing binary64's longer exact promotion.
        MetadataValue::F32(value) => Value::Number(value.to_string().parse()?),
        MetadataValue::Array(values) => report_array(values)?,
        MetadataValue::U8(value) => json!(value),
        MetadataValue::I8(value) => json!(value),
        MetadataValue::U16(value) => json!(value),
        MetadataValue::I16(value) => json!(value),
        MetadataValue::U32(value) => json!(value),
        MetadataValue::I32(value) => json!(value),
        MetadataValue::Bool(value) => json!(value),
        MetadataValue::String(value) => json!(value),
        MetadataValue::U64(value) => json!(value),
        MetadataValue::I64(value) => json!(value),
        MetadataValue::F64(value) => json!(value),
    })
}

#[cfg(feature = "report")]
fn report_array(values: &MetadataArray) -> Result<Value> {
    Ok(match values {
        MetadataArray::F32(values) => Value::Array(
            values
                .iter()
                .map(|value| Ok(Value::Number(value.to_string().parse()?)))
                .collect::<Result<_>>()?,
        ),
        MetadataArray::Array(values) => {
            Value::Array(values.iter().map(report_array).collect::<Result<_>>()?)
        }
        MetadataArray::U8(values) => json!(values),
        MetadataArray::I8(values) => json!(values),
        MetadataArray::U16(values) => json!(values),
        MetadataArray::I16(values) => json!(values),
        MetadataArray::U32(values) => json!(values),
        MetadataArray::I32(values) => json!(values),
        MetadataArray::Bool(values) => json!(values),
        MetadataArray::String(values) => json!(values),
        MetadataArray::U64(values) => json!(values),
        MetadataArray::I64(values) => json!(values),
        MetadataArray::F64(values) => json!(values),
    })
}

// Match Convert.ToUInt64 for the architecture's count metadata, including its
// accepted booleans, numeric strings and rounded finite floating point values.
fn float_count(value: f64) -> Result<u64> {
    let rounded = value.round_ties_even();
    if rounded == 0.0 {
        return Ok(0);
    }
    if !(0.0..18_446_744_073_709_551_616.0).contains(&rounded) {
        return Err(overflow());
    }
    Ok(format!("{rounded:.0}").parse()?)
}

fn count_value(value: Option<&MetadataValue>) -> Result<u64> {
    let Some(value) = value else { return Ok(0) };
    if let Some(value) = value.as_u64() {
        return Ok(value);
    }
    match value {
        MetadataValue::Bool(value) => Ok(u64::from(*value)),
        MetadataValue::String(value) => {
            let value = value.trim_matches([' ', '\t', '\r', '\n', '\x0b', '\x0c']);
            if value
                .strip_prefix('-')
                .is_some_and(|digits| !digits.is_empty() && digits.bytes().all(|byte| byte == b'0'))
            {
                return Ok(0);
            }
            value.parse().map_err(Into::into)
        }
        MetadataValue::F32(value) => float_count(f64::from(*value)),
        MetadataValue::F64(value) => float_count(*value),
        MetadataValue::I8(_)
        | MetadataValue::I16(_)
        | MetadataValue::I32(_)
        | MetadataValue::I64(_) => Err(bad("architecture count cannot be negative.")),
        _ => Err(bad("invalid architecture count.")),
    }
}

// Keep validation and report assembly adjacent in serialized file order.
#[cfg(feature = "report")]
pub(crate) fn inspect(file: &mut File, path: &Path, dialect: &str) -> Result<Inspection> {
    inspect_retained(file, path, dialect, None)
}

// One parser serves inspection and execution; retention changes only what is
// kept, never descriptor, format, identity, or extent validation.
#[allow(
    clippy::too_many_lines,
    reason = "Keep descriptor validation and report assembly together in serialized order."
)]
pub(crate) fn inspect_retained(
    file: &mut File,
    path: &Path,
    dialect: &str,
    memory_limit: Option<usize>,
) -> Result<Inspection> {
    inspect_with_vocabulary_version(file, path, dialect, memory_limit, false)
}

pub(crate) fn inspect_vocabulary(
    file: &mut File,
    path: &Path,
    memory_limit: usize,
) -> Result<Inspection> {
    inspect_with_vocabulary_version(file, path, "Stock", Some(memory_limit), true)
}

#[allow(
    clippy::too_many_lines,
    reason = "Keep descriptor validation and report assembly together in serialized order."
)]
fn inspect_with_vocabulary_version(
    file: &mut File,
    path: &Path,
    dialect: &str,
    memory_limit: Option<usize>,
    legacy_vocabulary: bool,
) -> Result<Inspection> {
    if dialect != "Stock" && dialect != PRISM_DIALECT {
        return Err(bad(
            "unknown dialect; use Stock or the explicitly pinned Prism dialect.",
        ));
    }
    file.seek(SeekFrom::Start(0))?;
    let file_bytes = file.metadata()?.len();
    let mut parser = Parser {
        reader: BufReader::with_capacity(BUFFER_BYTES, file),
        position: 0,
        file_bytes,
        elements: 0,
        array_summary: None,
        retain_budget: memory_limit,
    };
    if parser.u32()? != 0x4655_4747 {
        return Err(bad("invalid magic."));
    }
    let version = parser.u32()?;
    if version != 3 && !(legacy_vocabulary && version == 2) {
        return Err(bad("only little-endian version 3 is supported."));
    }
    let tensor_count = parser.u64()?;
    if version == 2 && tensor_count != 0 {
        return Err(bad(
            "legacy vocabulary files must not contain model tensors.",
        ));
    }
    let metadata_count = parser.u64()?;
    if tensor_count > 100_000 || metadata_count > 10_000 {
        return Err(bad("tensor/KV count exceeds inspection limit."));
    }
    let mut keys = HashSet::new();
    let mut scalars = Metadata::new();
    #[cfg(feature = "report")]
    let mut scalar_report = Map::new();
    #[cfg(feature = "report")]
    let mut arrays = Map::new();
    #[cfg(feature = "report")]
    let mut templates = Map::new();
    #[cfg(feature = "report")]
    let mut special = Map::new();
    let mut metadata = Metadata::new();
    let mut alignment = 32u32;
    #[cfg(feature = "report")]
    let mut file_type_position = None;
    #[cfg(feature = "report")]
    let mut tokenizer_hash = Sha256::new();
    #[cfg(feature = "report")]
    let mut tokenizer_bytes = 0;
    #[cfg(feature = "report")]
    let mut tokenizer_records = 0u64;
    for _ in 0..metadata_count {
        #[cfg(feature = "report")]
        let start = parser.position();
        let key = parser.text(1024)?;
        #[cfg(feature = "report")]
        let tokenizer = key.starts_with("tokenizer.");
        // Covers tree nodes, duplicated validation keys and native value storage.
        // Payload charges below are checked before reserving string/array space.
        parser.charge(key.len().checked_add(256).ok_or_else(overflow)?)?;
        if key.is_empty() || !keys.insert(key.clone()) {
            return Err(bad("empty or duplicate metadata key."));
        }
        let type_id = parser.u32()?;
        #[cfg(feature = "report")]
        if key == "general.file_type" {
            file_type_position = Some(parser.position());
        }
        let template =
            key == "tokenizer.chat_template" || key.starts_with("tokenizer.chat_template.");
        if template && type_id == 8 && memory_limit.is_none() {
            #[cfg(feature = "report")]
            let value_start = parser.position();
            let size = parser.u64()?;
            if size > STRING_LIMIT {
                return Err(bad("template exceeds inspection limit."));
            }
            parser.need(size)?;
            parser.skip(size)?;
            #[cfg(feature = "report")]
            {
                let mut hash = Sha256::new();
                parser.hash_record(&mut hash, add(value_start, 8)?, size)?;
                let digest = crate::digest_hex(hash);
                templates.insert(key, json!({"Utf8Bytes": size, "Sha256": digest}));
            }
        } else {
            let value = parser.value(type_id, 0, true)?;
            #[cfg(feature = "report")]
            if memory_limit.is_none() {
                if type_id == 9 {
                    let report = if let Some(value) = &value {
                        report_value(value)?
                    } else {
                        let (item_type, count) = parser
                            .array_summary
                            .ok_or_else(|| bad("array descriptor is required."))?;
                        json!({"ElementTypeId": item_type, "Count": count})
                    };
                    arrays.insert(key.clone(), report);
                } else if let Some(value) = &value {
                    scalar_report.insert(key.clone(), report_value(value)?);
                }
            }
            if memory_limit.is_none()
                && type_id != 9
                && (key == "general.architecture"
                    || key.ends_with(".block_count")
                    || key.ends_with(".nextn_predict_layers"))
                && let Some(value) = &value
            {
                scalars.insert(key.clone(), value.clone());
            }
            if key.starts_with("tokenizer.") && key.ends_with("_token_id") {
                if !matches!(type_id, 4 | 5 | 10 | 11) {
                    return Err(bad("special token ID must be an integer."));
                }
                #[cfg(feature = "report")]
                if memory_limit.is_none() {
                    special.insert(
                        key.clone(),
                        report_value(
                            value
                                .as_ref()
                                .ok_or_else(|| bad("special token ID is required."))?,
                        )?,
                    );
                }
            }
            if key == "general.alignment" {
                if type_id != 4 {
                    return Err(bad("general.alignment must be UINT32."));
                }
                alignment = u32::try_from(
                    value
                        .as_ref()
                        .and_then(MetadataValue::as_u64)
                        .ok_or_else(|| bad("general.alignment must be UINT32."))?,
                )?;
                if alignment == 0 || !alignment.is_power_of_two() || alignment > 1_048_576 {
                    return Err(bad("invalid/over-limit alignment."));
                }
            }
            if memory_limit.is_some() {
                metadata.insert(
                    key,
                    value.ok_or_else(|| bad("retained model metadata value is required."))?,
                );
            }
        }
        #[cfg(feature = "report")]
        if memory_limit.is_none() && tokenizer {
            let bytes = parser.position() - start;
            parser.hash_record(&mut tokenizer_hash, start, bytes)?;
            tokenizer_bytes = add(tokenizer_bytes, bytes)?;
            tokenizer_records += 1;
        }
    }
    #[cfg(feature = "report")]
    let metadata_end = parser.position();
    let scalar_metadata = if memory_limit.is_some() {
        &metadata
    } else {
        &scalars
    };
    let architecture = scalar_metadata
        .get("general.architecture")
        .and_then(MetadataValue::as_str)
        .filter(|arch| !arch.is_empty())
        .ok_or_else(|| bad("general.architecture string is required."))?
        .to_owned();
    let mut names = HashSet::new();
    let tensor_count = usize::try_from(tensor_count)?;
    parser.charge(
        tensor_count
            .checked_mul(size_of::<TensorDescriptor>())
            .ok_or_else(overflow)?,
    )?;
    let mut tensors = Vec::new();
    tensors.try_reserve_exact(tensor_count)?;
    for _ in 0..tensor_count {
        let name = parser.text(1024)?;
        if name.is_empty() || !names.insert(name.clone()) {
            return Err(bad("empty or duplicate tensor name."));
        }
        let dimension_count = parser.u32()?;
        if !(1..=4).contains(&dimension_count) {
            return Err(bad("invalid tensor dimension count."));
        }
        let mut elements = 1;
        let mut first = 0;
        let mut dimensions = Vec::with_capacity(usize::try_from(dimension_count)?);
        for dimension in 0..dimension_count {
            let size = parser.u64()?;
            if size == 0 || i64::try_from(size).is_err() {
                return Err(bad("invalid tensor dimension."));
            }
            if dimension == 0 {
                first = size;
            }
            elements = mul(elements, size)?;
            dimensions.push(size);
        }
        #[cfg(feature = "report")]
        let type_position = parser.position();
        let type_id = parser.u32()?;
        let kind = kind(type_id, dialect == PRISM_DIALECT)
            .ok_or_else(|| bad(format!("unknown/unsupported tensor type ID {type_id}.")))?;
        if first % kind.block != 0 {
            return Err(bad(
                "tensor row is not divisible by quantization block size.",
            ));
        }
        let bytes = mul(elements / kind.block, kind.bytes)?;
        let offset = parser.u64()?;
        if offset % u64::from(alignment) != 0 {
            return Err(bad("unaligned tensor offset."));
        }
        parser.charge(
            name.len()
                // A native Model retains a map key and distinct tensor name.
                .checked_mul(2)
                .and_then(|bytes| bytes.checked_add(dimensions.len() * size_of::<u64>()))
                .and_then(|bytes| bytes.checked_add(128))
                .ok_or_else(overflow)?,
        )?;
        tensors.push(TensorDescriptor {
            name,
            dimensions,
            type_id,
            offset,
            bytes,
            elements,
            #[cfg(feature = "report")]
            type_position,
        });
    }
    let descriptors_end = parser.position();
    let data_offset = if tensor_count > 0 {
        mul(
            add(descriptors_end, u64::from(alignment) - 1)? / u64::from(alignment),
            u64::from(alignment),
        )?
    } else {
        descriptors_end
    };
    if data_offset > file_bytes {
        return Err(bad("tensor data start exceeds file length."));
    }
    let mut ordered: Vec<_> = tensors.iter().collect();
    // Valid nonempty payloads have distinct offsets, so stable tie ordering
    // cannot affect an accepted file. Sort without an extra scratch allocation.
    ordered.sort_unstable_by_key(|tensor| tensor.offset);
    let mut previous_end = 0;
    for tensor in ordered {
        let end = add(tensor.offset, tensor.bytes)?;
        if end > file_bytes - data_offset || tensor.offset < previous_end {
            return Err(bad("tensor payload extent is truncated or overlapping."));
        }
        previous_end = end;
    }
    let block_key = format!("{architecture}.block_count");
    let nextn_key = format!("{architecture}.nextn_predict_layers");
    let blocks = count_value(
        scalar_metadata
            .get(&block_key)
            .filter(|value| value.as_array().is_none()),
    )?;
    let nextn = count_value(
        scalar_metadata
            .get(&nextn_key)
            .filter(|value| value.as_array().is_none()),
    )?;
    if nextn > blocks {
        return Err(bad("MTP block count exceeds block count."));
    }
    #[cfg(feature = "report")]
    let report = if memory_limit.is_some() {
        Value::Null
    } else {
        let mut groups: BTreeMap<u32, (u64, u64, u64)> = BTreeMap::new();
        for tensor in &tensors {
            let group = groups.entry(tensor.type_id).or_default();
            group.0 += 1;
            group.1 = add(group.1, tensor.bytes)?;
            group.2 = add(group.2, tensor.elements)?;
        }
        let mut histogram = Vec::with_capacity(groups.len());
        for (type_id, (count, bytes, elements)) in groups {
            let kind = kind(type_id, dialect == PRISM_DIALECT)
                .ok_or_else(|| bad("unknown tensor kind."))?;
            histogram.push(json!({"TypeId": type_id, "Name": kind.name, "Count": count, "Bytes": bytes, "Elements": elements}));
        }
        let mtp: Vec<_> = tensors
            .iter()
            .filter(|tensor| {
                if tensor.name.starts_with("nextn.") || tensor.name.contains(".nextn.") {
                    return true;
                }
                if nextn > 0
                    && let Some(rest) = tensor.name.strip_prefix("blk.")
                    && let Some((block, _)) = rest.split_once('.')
                    && !block.is_empty()
                    && block.bytes().all(|byte| byte.is_ascii_digit())
                    && let Ok(block) = block.parse::<u64>()
                {
                    return block >= blocks - nextn && block < blocks;
                }
                false
            })
            .map(|tensor| tensor.name.clone())
            .collect();
        json!({
                "SchemaVersion": 1, "Format": "GGUF", "Version": 3, "ByteOrder": "little", "ModelPath": path.to_string_lossy(),
                "FileBytes": file_bytes, "Architecture": architecture, "Alignment": alignment, "MetadataCount": metadata_count,
                "MetadataEndOffset": metadata_end, "TensorDescriptorsEndOffset": descriptors_end, "TensorDataOffset": data_offset,
                "TensorCount": tensor_count, "TensorTypeHistogram": histogram, "Scalars": scalar_report, "Arrays": arrays,
                "SpecialTokenIds": special, "EmbeddedTemplates": templates, "MtpTensorCount": mtp.len(), "MtpTensorNames": mtp,
                "TokenizerMetadataSha256": crate::digest_hex(tokenizer_hash), "TokenizerMetadataBytes": tokenizer_bytes,
                "TokenizerMetadataRecords": tokenizer_records,
                "TokenizerDigestDefinition": "SHA-256 of complete raw serialized tokenizer.* KV records concatenated in file order, including chat templates",
                "InspectionLimits": {"HeaderBytes": HEADER_LIMIT, "StringBytes": STRING_LIMIT, "ArrayElements": ARRAY_LIMIT,
                    "TotalArrayElements": TOTAL_ELEMENT_LIMIT, "ArrayNesting": 4, "MetadataRecords": 10_000, "Tensors": 100_000},
                "TensorPayloadRead": false, "Dialect": dialect
        })
    };
    #[cfg(not(feature = "report"))]
    let _ = path;
    let retained_bytes = match (memory_limit, parser.retain_budget) {
        (Some(limit), Some(remaining)) => limit - remaining,
        _ => 0,
    };
    // Release buffered read-ahead before returning the verified file owner.
    drop(parser);
    Ok(Inspection {
        #[cfg(feature = "report")]
        report,
        tensors,
        #[cfg(feature = "report")]
        file_type_position,
        metadata,
        retained_bytes,
        data_offset,
    })
}
