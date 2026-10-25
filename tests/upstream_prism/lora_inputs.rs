//! Immutable original HF inputs; no converted models or observed outputs.
use super::engine::Engine;
use super::{Result, invalid};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
pub(super) struct Inputs {
    paths: BTreeMap<String, PathBuf>,
    pub(super) receipt: Value,
}
pub(super) fn hash_bytes(data: &[u8]) -> String {
    encoded_hash(&Sha256::digest(data))
}
fn encoded_hash(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    bytes
        .iter()
        .flat_map(|&v| {
            [
                char::from(DIGITS[usize::from(v >> 4)]),
                char::from(DIGITS[usize::from(v & 15)]),
            ]
        })
        .collect()
}
pub(super) fn digest(path: &Path) -> Result<(u64, String)> {
    let mut input = File::open(path)?;
    let mut sha = Sha256::new();
    let mut length = 0_u64;
    let mut buffer = vec![0_u8; 65536];
    loop {
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        length = length
            .checked_add(u64::try_from(count)?)
            .ok_or_else(|| invalid("LoRA input length overflow"))?;
        if length > 256 * 1024 * 1024 {
            return Err(invalid("LoRA file exceeds256MiB"));
        }
        sha.update(&buffer[..count]);
    }
    Ok((length, encoded_hash(&sha.finalize())))
}
impl Inputs {
    pub(super) fn verify(engine: &Engine) -> Result<Self> {
        let lock: Value = serde_json::from_str(include_str!("lora-assets.json"))?;
        if lock["Revision"] != "c26d5fb85b4070a9e9c4e65d132c783b98086890" {
            return Err(invalid("unexpected LoRA input revision"));
        }
        let entries = lock["Files"]
            .as_array()
            .filter(|v| v.len() == 28)
            .ok_or_else(|| invalid("LoRA input lock incomplete"))?;
        let mut paths = BTreeMap::new();
        let mut receipts = Vec::new();
        for entry in entries {
            let relative = entry["Path"]
                .as_str()
                .ok_or_else(|| invalid("input path absent"))?;
            if relative.is_empty()
                || relative.contains("..")
                || relative.contains('\\')
                || Path::new(relative).is_absolute()
            {
                return Err(invalid("invalid locked LoRA input path"));
            }
            let path = engine.asset_path(&format!("lora-tests/{relative}"))?;
            if Some(path.metadata()?.len()) != entry["Bytes"].as_u64() {
                return Err(invalid(format!(
                    "original LoRA input size mismatch: {relative}"
                )));
            }
            let (length, sha) = digest(&path)?;
            if entry["Bytes"].as_u64() != Some(length) || entry["Sha256"] != sha {
                return Err(invalid(format!(
                    "original LoRA input digest mismatch: {relative}"
                )));
            }
            if paths.insert(relative.into(), path).is_some() {
                return Err(invalid("duplicate locked LoRA input path"));
            }
            receipts.push(json!({"Path":relative,"Bytes":length,"Sha256":sha}));
        }
        Ok(Self {
            paths,
            receipt: json!({"Revision":lock["Revision"],"Files":receipts}),
        })
    }
    pub(super) fn path(&self, name: &str) -> Result<&Path> {
        self.paths
            .get(name)
            .map(PathBuf::as_path)
            .ok_or_else(|| invalid(format!("LoRA input not locked: {name}")))
    }
    pub(super) fn json(&self, name: &str) -> Result<Value> {
        let path = self.path(name)?;
        if path.metadata()?.len() > 20 * 1024 * 1024 {
            return Err(invalid("LoRA JSON input exceeds20MiB"));
        }
        Ok(serde_json::from_reader(File::open(path)?)?)
    }
}
