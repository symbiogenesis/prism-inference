//! Shared independent GGUF bytes for tokenizer, CLI and worker fixtures.
#![expect(
    clippy::redundant_pub_crate,
    reason = "Shared fixture helpers stay visible only inside their owning integration test."
)]

use bonsai_prism_runtime::Result;
use sha2::{Digest, Sha256};

pub(crate) fn model_bytes(vocab_size: u64, eos: u32, pre: &str) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    u32_bytes(&mut bytes, 0x4655_4747);
    u32_bytes(&mut bytes, 3);
    u64_bytes(&mut bytes, 1);
    u64_bytes(&mut bytes, 10);
    string_metadata(
        &mut bytes,
        "general.architecture",
        "unsupported-for-generation",
    )?;
    string_metadata(&mut bytes, "tokenizer.ggml.model", "gpt2")?;
    string_metadata(&mut bytes, "tokenizer.ggml.pre", pre)?;
    text(&mut bytes, "tokenizer.ggml.tokens")?;
    u32_bytes(&mut bytes, 9);
    u32_bytes(&mut bytes, 8);
    u64_bytes(&mut bytes, 260);
    let mut next = 256_u32;
    for byte in u8::MIN..=u8::MAX {
        let scalar = if (b'!'..=b'~').contains(&byte)
            || (0xa1..=0xac).contains(&byte)
            || (0xae..=0xff).contains(&byte)
        {
            u32::from(byte)
        } else {
            let scalar = next;
            next += 1;
            scalar
        };
        let character = char::from_u32(scalar).ok_or("invalid fixture byte alphabet")?;
        text(&mut bytes, &character.to_string())?;
    }
    for token in ["ab", "a!", "<stop>", "<tag>"] {
        text(&mut bytes, token)?;
    }
    text(&mut bytes, "tokenizer.ggml.token_type")?;
    u32_bytes(&mut bytes, 9);
    u32_bytes(&mut bytes, 5);
    u64_bytes(&mut bytes, 260);
    for _ in 0..258 {
        bytes.extend(1_i32.to_le_bytes());
    }
    for kind in [3_i32, 4] {
        bytes.extend(kind.to_le_bytes());
    }
    text(&mut bytes, "tokenizer.ggml.merges")?;
    u32_bytes(&mut bytes, 9);
    u32_bytes(&mut bytes, 8);
    u64_bytes(&mut bytes, 2);
    text(&mut bytes, "a b")?;
    text(&mut bytes, "a !")?;
    for (key, token) in [
        ("tokenizer.ggml.eos_token_id", eos),
        ("tokenizer.ggml.bos_token_id", 259),
    ] {
        text(&mut bytes, key)?;
        u32_bytes(&mut bytes, 4);
        u32_bytes(&mut bytes, token);
    }
    for key in [
        "tokenizer.ggml.add_bos_token",
        "tokenizer.ggml.add_eos_token",
    ] {
        text(&mut bytes, key)?;
        u32_bytes(&mut bytes, 7);
        bytes.push(1);
    }
    text(&mut bytes, "token_embd.weight")?;
    u32_bytes(&mut bytes, 2);
    u64_bytes(&mut bytes, 1024);
    u64_bytes(&mut bytes, vocab_size);
    u32_bytes(&mut bytes, 0);
    u64_bytes(&mut bytes, 0);
    let data_offset = bytes.len().div_ceil(32) * 32;
    let payload_bytes = usize::try_from(vocab_size)? * 1024 * 4;
    bytes.resize(data_offset + payload_bytes, 0);
    bytes[data_offset..data_offset + 4].copy_from_slice(&f32::NAN.to_le_bytes());
    Ok(bytes)
}

pub(crate) fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .into_iter()
        .flat_map(|byte| [byte >> 4, byte & 15])
        .map(|nibble| char::from(b"0123456789abcdef"[usize::from(nibble)]))
        .collect()
}

fn u32_bytes(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend(value.to_le_bytes());
}

fn u64_bytes(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend(value.to_le_bytes());
}

fn text(bytes: &mut Vec<u8>, value: &str) -> Result<()> {
    u64_bytes(bytes, u64::try_from(value.len())?);
    bytes.extend(value.as_bytes());
    Ok(())
}

fn string_metadata(bytes: &mut Vec<u8>, key: &str, value: &str) -> Result<()> {
    text(bytes, key)?;
    u32_bytes(bytes, 8);
    text(bytes, value)
}
