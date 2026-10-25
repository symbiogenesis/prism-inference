//! The legacy vocabulary reader preserves the strict model format boundary.
#![forbid(unsafe_code)]

use bonsai_gguf::{MetadataArray, MetadataValue, Result, VerifiedModel, read_vocabulary};
use sha2::{Digest, Sha256};
use std::fs;

mod support;
use support::ScratchDirectory;

fn text(output: &mut Vec<u8>, value: &str) -> Result<()> {
    output.extend(u64::try_from(value.len())?.to_le_bytes());
    output.extend(value.as_bytes());
    Ok(())
}

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .flat_map(|byte| [byte >> 4, byte & 15])
        .map(|nibble| char::from(b"0123456789abcdef"[usize::from(nibble)]))
        .collect()
}

#[test]
fn legacy_vocabulary_metadata_is_verified_without_relaxing_model_execution() -> Result<()> {
    let scratch = ScratchDirectory::new()?;
    let path = scratch.path().join("legacy-vocabulary.gguf");
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(2_u32.to_le_bytes());
    bytes.extend(0_u64.to_le_bytes());
    bytes.extend(2_u64.to_le_bytes());
    text(&mut bytes, "general.architecture")?;
    bytes.extend(8_u32.to_le_bytes());
    text(&mut bytes, "synthetic")?;
    text(&mut bytes, "tokenizer.ggml.tokens")?;
    bytes.extend(9_u32.to_le_bytes());
    bytes.extend(8_u32.to_le_bytes());
    bytes.extend(3_u64.to_le_bytes());
    for piece in ["hello", "世界", "<eos>"] {
        text(&mut bytes, piece)?;
    }
    fs::write(&path, &bytes)?;
    let size = u64::try_from(bytes.len())?;
    let hash = digest(&bytes);
    let metadata = read_vocabulary(&path, size, &hash, 16_384)?;
    assert_eq!(
        metadata.get("tokenizer.ggml.tokens"),
        Some(&MetadataValue::Array(MetadataArray::String(vec![
            "hello".to_owned(),
            "世界".to_owned(),
            "<eos>".to_owned()
        ])))
    );
    assert!(VerifiedModel::open(&path, "Stock", size, &hash, 16_384).is_err());
    assert!(read_vocabulary(&path, size, &hash, 1).is_err());
    assert!(read_vocabulary(&path, size + 1, &hash, 16_384).is_err());
    assert!(read_vocabulary(&path, size, &"0".repeat(64), 16_384).is_err());
    for version in [1_u32, 3, 4, 0x0200_0000] {
        bytes[4..8].copy_from_slice(&version.to_le_bytes());
        fs::write(&path, &bytes)?;
        assert_eq!(
            read_vocabulary(&path, size, &digest(&bytes), 16_384).is_ok(),
            version == 3
        );
    }
    bytes[4..8].copy_from_slice(&2_u32.to_le_bytes());
    bytes[8..16].copy_from_slice(&1_u64.to_le_bytes());
    fs::write(&path, &bytes)?;
    assert!(read_vocabulary(&path, size, &digest(&bytes), 16_384).is_err());
    scratch.close()?;
    Ok(())
}
