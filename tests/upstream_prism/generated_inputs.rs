//! Independent safe Rust metadata/weight inputs for generated source models.

use super::windows_random::{Normal, Random, hash_name64};
use super::{Result, invalid};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub(super) struct Entry {
    name: String,
    kind: u32,
    data: Vec<u8>,
}
fn entry(name: &str, kind: u32, data: Vec<u8>) -> Entry {
    Entry {
        name: name.into(),
        kind,
        data,
    }
}
fn string_bytes(value: &str) -> Result<Vec<u8>> {
    let mut bytes = u64::try_from(value.len())?.to_le_bytes().to_vec();
    bytes.extend_from_slice(value.as_bytes());
    Ok(bytes)
}
pub(super) fn string(name: &str, value: &str) -> Result<Entry> {
    Ok(entry(name, 8, string_bytes(value)?))
}
pub(super) fn u32_value(name: &str, value: u32) -> Entry {
    entry(name, 4, value.to_le_bytes().to_vec())
}
pub(super) fn f32_value(name: &str, value: f32) -> Entry {
    entry(name, 6, value.to_bits().to_le_bytes().to_vec())
}
pub(super) fn boolean(name: &str, value: bool) -> Entry {
    entry(name, 7, vec![u8::from(value)])
}
pub(super) fn u32_array(name: &str, value: &[u32]) -> Result<Entry> {
    let mut bytes = 4_u32.to_le_bytes().to_vec();
    bytes.extend_from_slice(&u64::try_from(value.len())?.to_le_bytes());
    for item in value {
        bytes.extend_from_slice(&item.to_le_bytes());
    }
    Ok(entry(name, 9, bytes))
}
pub(super) fn f32_array(name: &str, value: &[f32]) -> Result<Entry> {
    let mut bytes = 6_u32.to_le_bytes().to_vec();
    bytes.extend_from_slice(&u64::try_from(value.len())?.to_le_bytes());
    for item in value {
        bytes.extend_from_slice(&item.to_bits().to_le_bytes());
    }
    Ok(entry(name, 9, bytes))
}
pub(super) fn metadata(path: &Path, entries: &[Entry], hints: &[(String, u32)]) -> Result<()> {
    if entries.len() > 1024 || hints.len() > 1024 {
        return Err(invalid("generated metadata inventory exceeds bound"));
    }
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    bytes.extend_from_slice(&u64::try_from(hints.len())?.to_le_bytes());
    bytes.extend_from_slice(&u64::try_from(entries.len())?.to_le_bytes());
    let mut names = std::collections::BTreeSet::new();
    for item in entries {
        if !names.insert(&item.name) {
            return Err(invalid("duplicate generated metadata key"));
        }
        bytes.extend(string_bytes(&item.name)?);
        bytes.extend_from_slice(&item.kind.to_le_bytes());
        bytes.extend_from_slice(&item.data);
    }
    names.clear();
    for (name, kind) in hints {
        if !names.insert(name) {
            return Err(invalid("duplicate generated tensor hint"));
        }
        bytes.extend(string_bytes(name)?);
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&[0; 32]);
        bytes.extend_from_slice(&kind.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
    }
    bytes.resize(
        bytes
            .len()
            .checked_add(31)
            .ok_or_else(|| invalid("metadata padding overflow"))?
            & !31,
        0,
    );
    if bytes.len() > 1024 * 1024 {
        return Err(invalid("generated metadata exceeds1MiB"));
    }
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)?;
    file.write_all(&bytes)?;
    Ok(())
}
pub(super) fn hex(bytes: &[u8]) -> Result<String> {
    let mut value = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut value, "{byte:02x}")?;
    }
    Ok(value)
}
pub(super) fn floats(value: &Value) -> Result<Vec<f32>> {
    let text = value
        .as_str()
        .ok_or_else(|| invalid("missing float bytes"))?;
    if text.len() % 8 != 0 || text.len() > 16 * 1024 * 1024 {
        return Err(invalid("invalid float buffer extent"));
    }
    let raw = text
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| {
            let digit = |v: u8| match v {
                b'0'..=b'9' => Ok(v - b'0'),
                b'a'..=b'f' => Ok(v - b'a' + 10),
                _ => Err(invalid("invalid float byte digit")),
            };
            Ok(digit(pair[0])? * 16 + digit(pair[1])?)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&bytes| f32::from_bits(u32::from_le_bytes(bytes)))
        .collect())
}
pub(super) type TensorFunction = Arc<dyn Fn(&Value) -> Result<Value> + Send + Sync>;
#[derive(Default)]
pub(super) struct WeightEvidence {
    pub(super) tensors: Vec<Value>,
    pub(super) bytes: u64,
    pub(super) source_type_checks: usize,
}
pub(super) fn weights(
    directory: PathBuf,
    seed: u64,
    xor: bool,
    require_f32: bool,
) -> (TensorFunction, Arc<Mutex<WeightEvidence>>) {
    let evidence = Arc::new(Mutex::new(WeightEvidence::default()));
    let shared = Arc::clone(&evidence);
    let function: TensorFunction = Arc::new(move |spec| {
        let mut state = shared
            .lock()
            .map_err(|_| invalid("generated input receipt poisoned"))?;
        let name = spec["Name"]
            .as_str()
            .ok_or_else(|| invalid("tensor input has no name"))?;
        let kind = spec["Type"]
            .as_u64()
            .ok_or_else(|| invalid("tensor input has no type"))?;
        if require_f32 {
            state.source_type_checks += 1;
            if kind != 0 {
                return Err(invalid("source initializer requires F32 tensor"));
            }
        }
        if name.is_empty() || name.len() > 128 || kind > 1 || state.tensors.len() >= 2048 {
            return Err(invalid("generated tensor descriptor outside source bounds"));
        }
        let dimensions = spec["Dimensions"]
            .as_array()
            .ok_or_else(|| invalid("tensor dimensions absent"))?;
        if dimensions.len() != 4 {
            return Err(invalid("tensor dimensions not padded to4"));
        }
        let elements = dimensions.iter().try_fold(1_u64, |n, v| {
            n.checked_mul(
                v.as_u64()
                    .filter(|&n| n > 0)
                    .ok_or_else(|| invalid("invalid tensor extent"))?,
            )
            .ok_or_else(|| invalid("tensor extent overflow"))
        })?;
        let length = elements
            .checked_mul(4)
            .ok_or_else(|| invalid("generated tensor byte overflow"))?;
        if length > 64 * 1024 * 1024
            || spec["Bytes"].as_u64() != Some(elements * if kind == 0 { 4 } else { 2 })
            || state.bytes + length > 512 * 1024 * 1024
        {
            return Err(invalid("generated tensor bytes exceed checked bounds"));
        }
        let filename = format!("generated-weight-{seed}-{:04}.bin", state.tensors.len());
        let path = directory.join(filename);
        let hash = hash_name64(name);
        let combined = if xor {
            hash ^ seed
        } else {
            hash.wrapping_add(seed)
        };
        let seed32 = u32::from_le_bytes(combined.to_le_bytes()[..4].try_into()?);
        let mut random = Random::new(seed32);
        let mut normal = Normal::new(0.0, 1e-2);
        let mut hasher = Sha256::new();
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)?;
        let mut bytes = Vec::with_capacity(8192);
        for _ in 0..elements {
            let value = normal.sample(&mut random);
            if !value.is_finite() {
                return Err(invalid("source weight generator produced nonfinite value"));
            }
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            if bytes.len() == 8192 {
                file.write_all(&bytes)?;
                hasher.update(&bytes);
                bytes.clear();
            }
        }
        file.write_all(&bytes)?;
        hasher.update(&bytes);
        drop(file);
        state.bytes += length;
        state.tensors.push(json!({"Spec":spec,"Seed32":seed32,"InputSHA256":hex(&hasher.finalize())?,"FloatInputBytes":length}));
        drop(state);
        Ok(json!({"DataFile":path,"Length":length,"Encoding":"float32-le"}))
    });
    (function, evidence)
}
