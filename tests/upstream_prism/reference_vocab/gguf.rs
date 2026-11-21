//! Clean-room little-endian GGUF v2/v3 metadata-only reader for vocabulary extraction.
//!
//! No `bonsai_prism_runtime` or `bonsai_gguf` code is imported or called.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::Path;

use super::super::{Result, invalid};

const MAX_GGUF_BYTES: u64 = 64 * 1024 * 1024;
const MAX_KV_COUNT: u64 = 131_072;
const MAX_ARRAY_LEN: u64 = 2_000_000;
const MAX_STRING_LEN: u64 = 16 * 1024 * 1024;
const MAX_ARRAY_DEPTH: usize = 16;

#[derive(Clone, Debug)]
pub(super) enum MetaValue {
    UInt(u64),
    Int(i64),
    Bool(bool),
    Bytes(Vec<u8>),
    UIntArray(Vec<u64>),
    IntArray(Vec<i64>),
    FloatArray(Vec<f32>),
    BytesArray(Vec<Vec<u8>>),
    Other,
}

impl MetaValue {
    pub(super) fn as_str_lossy(&self) -> Option<String> {
        match self {
            Self::Bytes(bytes) => Some(String::from_utf8_lossy(bytes).into_owned()),
            _ => None,
        }
    }

    pub(super) const fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(value) => Some(*value),
            Self::UInt(value) => Some(*value != 0),
            Self::Int(value) => Some(*value != 0),
            _ => None,
        }
    }

    pub(super) fn as_token_id(&self) -> Option<u32> {
        match self {
            Self::UInt(value) => u32::try_from(*value).ok(),
            Self::Int(value) if *value >= 0 => u32::try_from(*value).ok(),
            _ => None,
        }
    }
}

struct Reader {
    inner: BufReader<File>,
}

impl Reader {
    fn read_exact_bytes(&mut self, len: usize) -> Result<Vec<u8>> {
        if u64::try_from(len)? > MAX_STRING_LEN {
            return Err(invalid("GGUF field exceeds maximum byte bound"));
        }
        let mut buf = vec![0_u8; len];
        self.inner.read_exact(&mut buf)?;
        Ok(buf)
    }

    fn u8(&mut self) -> Result<u8> {
        let mut buf = [0_u8; 1];
        self.inner.read_exact(&mut buf)?;
        Ok(buf[0])
    }

    fn u16(&mut self) -> Result<u16> {
        let mut buf = [0_u8; 2];
        self.inner.read_exact(&mut buf)?;
        Ok(u16::from_le_bytes(buf))
    }

    fn i16(&mut self) -> Result<i16> {
        let mut buf = [0_u8; 2];
        self.inner.read_exact(&mut buf)?;
        Ok(i16::from_le_bytes(buf))
    }

    fn u32(&mut self) -> Result<u32> {
        let mut buf = [0_u8; 4];
        self.inner.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn i32(&mut self) -> Result<i32> {
        let mut buf = [0_u8; 4];
        self.inner.read_exact(&mut buf)?;
        Ok(i32::from_le_bytes(buf))
    }

    fn f32(&mut self) -> Result<f32> {
        let mut buf = [0_u8; 4];
        self.inner.read_exact(&mut buf)?;
        Ok(f32::from_le_bytes(buf))
    }

    fn u64(&mut self) -> Result<u64> {
        let mut buf = [0_u8; 8];
        self.inner.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    fn i64(&mut self) -> Result<i64> {
        let mut buf = [0_u8; 8];
        self.inner.read_exact(&mut buf)?;
        Ok(i64::from_le_bytes(buf))
    }

    fn f64(&mut self) -> Result<f64> {
        let mut buf = [0_u8; 8];
        self.inner.read_exact(&mut buf)?;
        Ok(f64::from_le_bytes(buf))
    }

    fn string_bytes(&mut self) -> Result<Vec<u8>> {
        let len = usize::try_from(self.u64()?)?;
        self.read_exact_bytes(len)
    }

    fn skip_bytes(&mut self, len: usize) -> Result<()> {
        let mut remaining = len;
        let mut buf = [0_u8; 4096];
        while remaining > 0 {
            let chunk = remaining.min(buf.len());
            self.inner.read_exact(&mut buf[..chunk])?;
            remaining -= chunk;
        }
        Ok(())
    }

    fn read_value(&mut self, kind: u32, retain: bool, depth: usize) -> Result<MetaValue> {
        if depth > MAX_ARRAY_DEPTH {
            return Err(invalid("GGUF metadata nesting exceeds bound"));
        }
        match kind {
            0..=7 | 10..=12 => self.read_scalar(kind, retain),
            8 => {
                if retain {
                    Ok(MetaValue::Bytes(self.string_bytes()?))
                } else {
                    let len = usize::try_from(self.u64()?)?;
                    if u64::try_from(len)? > MAX_STRING_LEN {
                        return Err(invalid("GGUF string exceeds maximum byte bound"));
                    }
                    self.skip_bytes(len)?;
                    Ok(MetaValue::Other)
                }
            }
            9 => self.read_array(retain, depth),
            _ => Err(invalid("unsupported GGUF metadata value type")),
        }
    }

    fn read_scalar(&mut self, kind: u32, retain: bool) -> Result<MetaValue> {
        let value = match kind {
            0 => MetaValue::UInt(u64::from(self.u8()?)),
            1 => {
                let mut buf = [0_u8; 1];
                self.inner.read_exact(&mut buf)?;
                MetaValue::Int(i64::from(i8::from_le_bytes(buf)))
            }
            2 => MetaValue::UInt(u64::from(self.u16()?)),
            3 => MetaValue::Int(i64::from(self.i16()?)),
            4 => MetaValue::UInt(u64::from(self.u32()?)),
            5 => MetaValue::Int(i64::from(self.i32()?)),
            6 => {
                let _ = self.f32()?;
                MetaValue::Other
            }
            7 => MetaValue::Bool(self.u8()? != 0),
            10 => MetaValue::UInt(self.u64()?),
            11 => MetaValue::Int(self.i64()?),
            12 => {
                let _ = self.f64()?;
                MetaValue::Other
            }
            _ => return Err(invalid("unsupported GGUF scalar metadata type")),
        };
        Ok(if retain { value } else { MetaValue::Other })
    }

    fn read_array(&mut self, retain: bool, depth: usize) -> Result<MetaValue> {
        let elem_kind = self.u32()?;
        let count_u64 = self.u64()?;
        if count_u64 > MAX_ARRAY_LEN {
            return Err(invalid("GGUF array length exceeds bound"));
        }
        let count = usize::try_from(count_u64)?;
        if !retain {
            if let Some(fixed) = fixed_size(elem_kind) {
                let bytes = count
                    .checked_mul(fixed)
                    .ok_or_else(|| invalid("GGUF skipped array size overflow"))?;
                self.skip_bytes(bytes)?;
                return Ok(MetaValue::Other);
            }
            for _ in 0..count {
                self.read_value(elem_kind, false, depth + 1)?;
            }
            return Ok(MetaValue::Other);
        }
        match elem_kind {
            8 => {
                let mut items = Vec::with_capacity(count.min(4096));
                for _ in 0..count {
                    items.push(self.string_bytes()?);
                }
                Ok(MetaValue::BytesArray(items))
            }
            6 => {
                let mut items = Vec::with_capacity(count.min(4096));
                for _ in 0..count {
                    items.push(self.f32()?);
                }
                Ok(MetaValue::FloatArray(items))
            }
            4 | 2 | 0 | 10 => {
                let mut items = Vec::with_capacity(count.min(4096));
                for _ in 0..count {
                    let MetaValue::UInt(v) = self.read_value(elem_kind, true, depth + 1)? else {
                        return Err(invalid("GGUF array unsigned integer mismatch"));
                    };
                    items.push(v);
                }
                Ok(MetaValue::UIntArray(items))
            }
            5 | 3 | 1 | 11 => {
                let mut items = Vec::with_capacity(count.min(4096));
                for _ in 0..count {
                    let MetaValue::Int(v) = self.read_value(elem_kind, true, depth + 1)? else {
                        return Err(invalid("GGUF array signed integer mismatch"));
                    };
                    items.push(v);
                }
                Ok(MetaValue::IntArray(items))
            }
            7 => {
                self.skip_bytes(count)?;
                Ok(MetaValue::Other)
            }
            _ => {
                for _ in 0..count {
                    self.read_value(elem_kind, false, depth + 1)?;
                }
                Ok(MetaValue::Other)
            }
        }
    }
}

const fn fixed_size(kind: u32) -> Option<usize> {
    match kind {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

fn retain_key(key: &str) -> bool {
    key.starts_with("tokenizer.") || key == "general.architecture" || key == "general.name"
}

pub(super) fn read_vocabulary_metadata(
    path: &Path,
    max_bytes: u64,
) -> Result<BTreeMap<String, MetaValue>> {
    let meta = fs::metadata(path)?;
    if !meta.is_file() || meta.len() > max_bytes.min(MAX_GGUF_BYTES) {
        return Err(invalid("GGUF file is not a bounded regular file"));
    }
    let file = File::open(path)?;
    let mut reader = Reader {
        inner: BufReader::new(file),
    };
    let magic = reader.read_exact_bytes(4)?;
    if magic != b"GGUF" {
        return Err(invalid("invalid GGUF magic header"));
    }
    let version = reader.u32()?;
    if !matches!(version, 2 | 3) {
        return Err(invalid("unsupported GGUF container version"));
    }
    let _tensor_count = reader.u64()?;
    let kv_count = reader.u64()?;
    if kv_count > MAX_KV_COUNT {
        return Err(invalid("GGUF metadata key count exceeds bound"));
    }
    let mut entries = BTreeMap::new();
    for _ in 0..kv_count {
        let key_bytes = reader.string_bytes()?;
        let key = String::from_utf8(key_bytes)
            .map_err(|_| invalid("GGUF metadata key is not valid UTF-8"))?;
        let kind = reader.u32()?;
        let retain = retain_key(&key);
        let value = reader.read_value(kind, retain, 0)?;
        if retain {
            entries.insert(key, value);
        }
    }
    Ok(entries)
}
