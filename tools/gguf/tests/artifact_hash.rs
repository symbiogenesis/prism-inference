//! Full-file verification must include bytes beyond the parsed GGUF header.
#![forbid(unsafe_code)]

use std::fs;

use bonsai_gguf::{Result, VerifiedModel};
use sha2::{Digest, Sha256};

mod support;
use support::ScratchDirectory;

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .flat_map(|byte| [byte >> 4, byte & 15])
        .map(|nibble| char::from(b"0123456789abcdef"[usize::from(nibble)]))
        .collect()
}

#[test]
fn verification_hashes_the_complete_file_before_header_parsing() -> Result<()> {
    let scratch = ScratchDirectory::new()?;
    let path = scratch.path().join("verified.gguf");
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3_u32.to_le_bytes());
    bytes.extend(0_u64.to_le_bytes()); // No tensors.
    bytes.extend(1_u64.to_le_bytes());
    bytes.extend(20_u64.to_le_bytes());
    bytes.extend(b"general.architecture");
    bytes.extend(8_u32.to_le_bytes());
    bytes.extend(9_u64.to_le_bytes());
    bytes.extend(b"synthetic");
    bytes.resize(131_073, 0xa5); // Include multiple hash buffers and a final tail.
    fs::write(&path, &bytes)?;
    let size = u64::try_from(bytes.len())?;
    let hash = digest(&bytes);
    let model = VerifiedModel::open(&path, "Stock", size, &hash.to_uppercase(), 16_384)?;
    assert!(model.tensors().is_empty());
    drop(model);

    // Header and size are unchanged; corruption only in the final payload byte.
    if let Some(last) = bytes.last_mut() {
        *last ^= 1;
    }
    fs::write(&path, &bytes)?;
    let error = VerifiedModel::open(&path, "Stock", size, &hash, 16_384).err();
    assert!(error.is_some_and(|error| error.to_string().contains("SHA-256 mismatch")));
    assert!(VerifiedModel::open(&path, "Stock", size + 1, &digest(&bytes), 16_384).is_err());
    assert!(VerifiedModel::open(&path, "Stock", size - 1, &digest(&bytes), 16_384).is_err());
    assert!(VerifiedModel::open(&path, "Stock", 0, &digest(&bytes), 16_384).is_err());
    scratch.close()?;
    Ok(())
}
