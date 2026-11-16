//! Full-file verification must include bytes beyond the parsed GGUF header.
#![forbid(unsafe_code)]

use std::fs;

use bonsai_gguf::{Result, VerifiedModel};
use sha2::{Digest, Sha256};

mod support;
use support::ScratchDirectory;

fn digest(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
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

#[test]
fn read_tensor_into_reuses_staging_and_clears_stale_bytes_on_error() -> Result<()> {
    let scratch = ScratchDirectory::new()?;
    let path = scratch.path().join("tensors.gguf");
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3_u32.to_le_bytes());
    bytes.extend(2_u64.to_le_bytes()); // Two tensors.
    bytes.extend(1_u64.to_le_bytes());
    bytes.extend(20_u64.to_le_bytes());
    bytes.extend(b"general.architecture");
    bytes.extend(8_u32.to_le_bytes());
    bytes.extend(9_u64.to_le_bytes());
    bytes.extend(b"synthetic");

    // Tensor 0: 8 bytes of F32 (2 elements) at offset 0.
    bytes.extend(1_u64.to_le_bytes());
    bytes.extend(b"a");
    bytes.extend(1_u32.to_le_bytes());
    bytes.extend(2_u64.to_le_bytes());
    bytes.extend(0_u32.to_le_bytes());
    bytes.extend(0_u64.to_le_bytes());

    // Tensor 1: 4 bytes of F32 (1 element) at offset 32.
    bytes.extend(1_u64.to_le_bytes());
    bytes.extend(b"b");
    bytes.extend(1_u32.to_le_bytes());
    bytes.extend(1_u64.to_le_bytes());
    bytes.extend(0_u32.to_le_bytes());
    bytes.extend(32_u64.to_le_bytes());

    let data_offset = bytes.len().div_ceil(32) * 32;
    bytes.resize(data_offset + 36, 0);
    let payload_a = [1_u8, 2, 3, 4, 5, 6, 7, 8];
    let payload_b = [9_u8, 10, 11, 12];
    bytes[data_offset..data_offset + 8].copy_from_slice(&payload_a);
    bytes[data_offset + 32..data_offset + 36].copy_from_slice(&payload_b);

    // Also include a >1 MiB padding tail so the 1 MiB hash_range buffer spans multiple reads.
    bytes.resize(1_200_000, 0x5a);
    fs::write(&path, &bytes)?;
    let size = u64::try_from(bytes.len())?;
    let hash = digest(&bytes);

    let mut model = VerifiedModel::open(&path, "Stock", size, &hash, 16_384)?;
    let mut staging = Vec::new();
    model.read_tensor_into(0, &mut staging, 16)?;
    assert_eq!(staging, payload_a);
    let cap_after_a = staging.capacity();

    // Smaller tensor reuses existing capacity without shrinking or leaving tail bytes.
    model.read_tensor_into(1, &mut staging, 16)?;
    assert_eq!(staging, payload_b);
    assert_eq!(staging.capacity(), cap_after_a);

    // Budget and out-of-bounds failures clear stale bytes from staging.
    assert!(model.read_tensor_into(0, &mut staging, 7).is_err());
    assert_eq!(staging, [0_u8; 0]);

    staging.extend_from_slice(b"stale");
    assert!(model.read_tensor_into(2, &mut staging, 16).is_err());
    assert_eq!(staging, [0_u8; 0]);

    assert_eq!(model.read_tensor(0, 8)?, payload_a);
    assert_eq!(model.read_tensor(1, 4)?, payload_b);
    drop(model);
    scratch.close()?;
    Ok(())
}
