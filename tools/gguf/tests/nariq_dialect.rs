#![forbid(unsafe_code)]
//! Dialect isolation and layout checks for `NARIQ_DIALECT` (`TRL1`–`TRL6`, `I32`).

use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use bonsai_gguf::{NARIQ_DIALECT, PRISM_DIALECT, Result, VerifiedModel, inspect};

mod support;
use support::ScratchDirectory;

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend(value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend(value.to_le_bytes());
}

fn push_string(out: &mut Vec<u8>, value: &str) -> Result<()> {
    push_u64(out, u64::try_from(value.len())?);
    out.extend(value.as_bytes());
    Ok(())
}

fn push_string_kv(out: &mut Vec<u8>, key: &str, value: &str) -> Result<()> {
    push_string(out, key)?;
    push_u32(out, 8);
    push_string(out, value)
}

fn push_tensor_info(
    out: &mut Vec<u8>,
    name: &str,
    dims: &[u64],
    type_id: u32,
    offset: usize,
) -> Result<()> {
    push_string(out, name)?;
    push_u32(out, u32::try_from(dims.len())?);
    for &dim in dims {
        push_u64(out, dim);
    }
    push_u32(out, type_id);
    push_u64(out, u64::try_from(offset)?);
    Ok(())
}

fn build_gguf(specs: &[(&str, &[u64], u32, usize)]) -> Result<Vec<u8>> {
    let mut header = Vec::new();
    header.extend(b"GGUF");
    push_u32(&mut header, 3);
    push_u64(&mut header, u64::try_from(specs.len())?);
    push_u64(&mut header, 1);
    push_string_kv(&mut header, "general.architecture", "qwen35")?;

    let mut offset = 0_usize;
    for &(name, dims, type_id, payload_len) in specs {
        let aligned = offset.next_multiple_of(32);
        push_tensor_info(&mut header, name, dims, type_id, aligned)?;
        offset = aligned + payload_len;
    }

    let data_start = header.len().next_multiple_of(32);
    header.resize(data_start, 0);
    header.resize(data_start + offset, 0);
    Ok(header)
}

fn sha256_hex(bytes: &[u8]) -> Result<String> {
    let digest = bonsai_native_hash::hash_bytes(bytes)?;
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = write!(out, "{byte:02x}");
    }
    Ok(out)
}

fn write_fixture(
    dir: &ScratchDirectory,
    name: &str,
    bytes: &[u8],
) -> Result<(PathBuf, u64, String)> {
    let path = dir.path().join(name);
    fs::write(&path, bytes)?;
    Ok((path, u64::try_from(bytes.len())?, sha256_hex(bytes)?))
}

#[test]
fn nariq_dialect_accepts_trellis_types_and_i32_with_exact_payload_sizes() -> Result<()> {
    let dir = ScratchDirectory::new()?;
    let bytes = build_gguf(&[
        ("t1.weight", &[16, 16], 44, 32),
        ("t2.weight", &[16, 16], 45, 64),
        ("t3.weight", &[16, 16], 46, 96),
        ("t4.weight", &[16, 16], 47, 128),
        ("t5.weight", &[16, 16], 48, 160),
        ("t6.weight", &[16, 16], 49, 192),
        ("t.trellis_pin", &[16], 26, 64),
    ])?;
    let (path, len, hash) = write_fixture(&dir, "synthetic-nariq.gguf", &bytes)?;

    let report = inspect(&path, NARIQ_DIALECT, false, len, &hash)?;
    assert_eq!(report["Dialect"], NARIQ_DIALECT);
    assert_eq!(report["TensorCount"], 7);

    let reader = VerifiedModel::open(&path, NARIQ_DIALECT, len, &hash, usize::MAX)?;
    let expected = [
        ("t1.weight", 44_u32, 32_u64),
        ("t2.weight", 45_u32, 64_u64),
        ("t3.weight", 46_u32, 96_u64),
        ("t4.weight", 47_u32, 128_u64),
        ("t5.weight", 48_u32, 160_u64),
        ("t6.weight", 49_u32, 192_u64),
        ("t.trellis_pin", 26_u32, 64_u64),
    ];
    assert_eq!(reader.tensors().len(), expected.len());
    for (descriptor, (name, kind, byte_len)) in reader.tensors().iter().zip(expected) {
        assert_eq!(descriptor.name, name);
        assert_eq!(descriptor.type_id, kind);
        assert_eq!(descriptor.bytes, byte_len);
    }
    drop(reader);
    dir.close()?;
    Ok(())
}

#[test]
fn nariq_and_other_dialects_stay_mutually_isolated() -> Result<()> {
    let dir = ScratchDirectory::new()?;
    for type_id in 44_u32..=49 {
        let len = 32 * usize::try_from(type_id - 43)?;
        let bytes = build_gguf(&[("tensor", &[16, 16], type_id, len)])?;
        let (path, total, hash) = write_fixture(&dir, &format!("trl-{type_id}.gguf"), &bytes)?;
        assert!(inspect(&path, "Stock", false, total, &hash).is_err());
        assert!(inspect(&path, PRISM_DIALECT, false, total, &hash).is_err());
    }

    for prism_only in [(142_u32, 34_usize), (143_u32, 28_usize)] {
        let bytes = build_gguf(&[("prism_tensor", &[128], prism_only.0, prism_only.1)])?;
        let (path, total, hash) =
            write_fixture(&dir, &format!("prism-{}.gguf", prism_only.0), &bytes)?;
        assert!(inspect(&path, NARIQ_DIALECT, false, total, &hash).is_err());
    }
    dir.close()?;
    Ok(())
}
