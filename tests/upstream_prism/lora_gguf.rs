//! Independent bounded F32 GGUF writer for original conversion inputs.
use super::{Result, invalid};
use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

pub(super) struct Metadata(BTreeMap<String, (u32, Vec<u8>)>);
fn string(output: &mut Vec<u8>, value: &str) -> Result<()> {
    output.extend_from_slice(&u64::try_from(value.len())?.to_le_bytes());
    output.extend_from_slice(value.as_bytes());
    Ok(())
}
impl Metadata {
    pub(super) const fn new() -> Self {
        Self(BTreeMap::new())
    }
    fn put(&mut self, name: &str, kind: u32, data: Vec<u8>) {
        self.0.insert(name.into(), (kind, data));
    }
    pub(super) fn text(&mut self, name: &str, value: &str) -> Result<()> {
        let mut data = Vec::new();
        string(&mut data, value)?;
        self.put(name, 8, data);
        Ok(())
    }
    pub(super) fn uint(&mut self, name: &str, value: u32) {
        self.put(name, 4, value.to_le_bytes().to_vec());
    }
    pub(super) fn float(&mut self, name: &str, value: f32) {
        self.put(name, 6, value.to_le_bytes().to_vec());
    }
    pub(super) fn boolean(&mut self, name: &str, value: bool) {
        self.put(name, 7, vec![u8::from(value)]);
    }
    fn array(&mut self, name: &str, kind: u32, count: usize, data: &[u8]) -> Result<()> {
        let mut bytes = kind.to_le_bytes().to_vec();
        bytes.extend_from_slice(&u64::try_from(count)?.to_le_bytes());
        bytes.extend_from_slice(data);
        self.put(name, 9, bytes);
        Ok(())
    }
    pub(super) fn strings(&mut self, name: &str, values: &[String]) -> Result<()> {
        let mut data = Vec::new();
        for value in values {
            string(&mut data, value)?;
        }
        self.array(name, 8, values.len(), &data)
    }
    pub(super) fn floats(&mut self, name: &str, values: &[f32]) -> Result<()> {
        self.array(
            name,
            6,
            values.len(),
            &values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
    }
    pub(super) fn integers(&mut self, name: &str, values: &[i32]) -> Result<()> {
        self.array(
            name,
            5,
            values.len(),
            &values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
    }
}
pub(super) struct Tensor {
    pub(super) name: String,
    /// `PyTorch` order; GGUF records the reversed dimensions.
    pub(super) shape: Vec<u64>,
}
impl Tensor {
    pub(super) fn bytes(&self) -> Result<u64> {
        if self.shape.is_empty() || self.shape.len() > 4 || self.shape.contains(&0) {
            return Err(invalid("invalid conversion tensor dimensions"));
        }
        self.shape.iter().try_fold(4_u64, |n, &v| {
            n.checked_mul(v)
                .filter(|&n| n <= 128 * 1024 * 1024)
                .ok_or_else(|| invalid("conversion tensor exceeds128MiB"))
        })
    }
}
fn pad(size: u64) -> Result<u64> {
    size.checked_add(31)
        .map(|n| n / 32 * 32)
        .ok_or_else(|| invalid("GGUF extent overflow"))
}
pub(super) fn write(
    path: &Path,
    metadata: &Metadata,
    tensors: &[Tensor],
    mut data: impl FnMut(usize) -> Result<Vec<u8>>,
) -> Result<u64> {
    if tensors.is_empty() || tensors.len() > 1024 {
        return Err(invalid("GGUF tensor count outside conversion bound"));
    }
    let mut header = b"GGUF".to_vec();
    header.extend_from_slice(&3_u32.to_le_bytes());
    header.extend_from_slice(&u64::try_from(tensors.len())?.to_le_bytes());
    header.extend_from_slice(&u64::try_from(metadata.0.len())?.to_le_bytes());
    for (name, (kind, bytes)) in &metadata.0 {
        string(&mut header, name)?;
        header.extend_from_slice(&kind.to_le_bytes());
        header.extend_from_slice(bytes);
    }
    let mut offset = 0_u64;
    let mut names = std::collections::BTreeSet::new();
    for tensor in tensors {
        if tensor.name.len() > 127 || !names.insert(&tensor.name) {
            return Err(invalid("GGUF duplicate/oversize tensor name"));
        }
        string(&mut header, &tensor.name)?;
        header.extend_from_slice(&u32::try_from(tensor.shape.len())?.to_le_bytes());
        for dim in tensor.shape.iter().rev() {
            header.extend_from_slice(&dim.to_le_bytes());
        }
        header.extend_from_slice(&0_u32.to_le_bytes());
        header.extend_from_slice(&offset.to_le_bytes());
        offset = offset
            .checked_add(pad(tensor.bytes()?)?)
            .ok_or_else(|| invalid("GGUF data overflow"))?;
    }
    if header.len() > 32 * 1024 * 1024 || offset > 256 * 1024 * 1024 {
        return Err(invalid("converted GGUF exceeds source resource bound"));
    }
    let aligned = usize::try_from(pad(u64::try_from(header.len())?)?)?;
    header.resize(aligned, 0);
    let mut output = OpenOptions::new().write(true).create_new(true).open(path)?;
    output.write_all(&header)?;
    for (i, tensor) in tensors.iter().enumerate() {
        let bytes = data(i)?;
        let expected = tensor.bytes()?;
        if u64::try_from(bytes.len())? != expected {
            return Err(invalid("converted tensor data extent mismatch"));
        }
        output.write_all(&bytes)?;
        let padding = usize::try_from(pad(expected)? - expected)?;
        output.write_all(&[0_u8; 32][..padding])?;
    }
    output.flush()?;
    Ok(u64::try_from(header.len())? + offset)
}

/// Write a bounded vocabulary-only test input, without tensors or model weights.
pub(super) fn write_vocabulary(path: &Path, metadata: &Metadata) -> Result<u64> {
    if !path.is_absolute() || metadata.0.len() > 256 {
        return Err(invalid("vocabulary GGUF path/metadata count outside bound"));
    }
    let mut header = b"GGUF".to_vec();
    header.extend_from_slice(&3_u32.to_le_bytes());
    header.extend_from_slice(&0_u64.to_le_bytes());
    header.extend_from_slice(&u64::try_from(metadata.0.len())?.to_le_bytes());
    for (name, (kind, bytes)) in &metadata.0 {
        string(&mut header, name)?;
        header.extend_from_slice(&kind.to_le_bytes());
        header.extend_from_slice(bytes);
        if header.len() > 32 * 1024 * 1024 {
            return Err(invalid("vocabulary GGUF metadata exceeds32MiB"));
        }
    }
    let aligned = usize::try_from(pad(u64::try_from(header.len())?)?)?;
    header.resize(aligned, 0);
    let mut output = OpenOptions::new().write(true).create_new(true).open(path)?;
    output.write_all(&header)?;
    output.flush()?;
    Ok(u64::try_from(header.len())?)
}
