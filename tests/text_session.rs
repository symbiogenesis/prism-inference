#![forbid(unsafe_code)]
//! Focused verified-artifact output-session and caller-owned stream regressions.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use bonsai_prism_runtime::Result;
use bonsai_prism_runtime::model::Model;
use bonsai_prism_runtime::text_output::{MAX_TEXT_OUTPUT_BYTES, Utf8Output};
use bonsai_prism_runtime::text_session::TextOutputSession;
use sha2::{Digest, Sha256};

const HEADER_BUDGET: usize = 64 * 1024;
const RECORDS: &[(&str, i32)] = &[
    ("Ġhello", 1),
    ("<0xCE>", 6),
    ("<0xBB>", 6),
    ("<stop>", 3),
    ("<stop>", 1),
    ("λ", 4),
    ("<0x00>", 6),
    ("<0xFF>", 6),
    ("<unknown>", 2),
    ("", 5),
];
static NEXT_ARTIFACT: AtomicU64 = AtomicU64::new(0);

struct Artifact {
    root: PathBuf,
    path: PathBuf,
    bytes: Vec<u8>,
}

impl Artifact {
    fn new(records: &[(&str, i32)], vocab_size: u64, pre: &str) -> Result<Self> {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let sequence = NEXT_ARTIFACT.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "SwiftBonsai-TextOutputSession-{}-{nonce}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&root)?;
        let path = root.join("model.gguf");
        let mut bytes = Vec::new();
        u32_bytes(&mut bytes, 0x4655_4747);
        u32_bytes(&mut bytes, 3);
        u64_bytes(&mut bytes, 1);
        u64_bytes(&mut bytes, 6);
        string_metadata(
            &mut bytes,
            "general.architecture",
            "unsupported-for-generation",
        )?;
        string_metadata(&mut bytes, "tokenizer.ggml.model", "gpt2")?;
        string_metadata(&mut bytes, "tokenizer.ggml.pre", pre)?;
        string_metadata(
            &mut bytes,
            "tokenizer.chat_template",
            "PRIVATE_TEMPLATE_NOT_RETAINED_BY_SESSION",
        )?;
        text(&mut bytes, "tokenizer.ggml.tokens")?;
        u32_bytes(&mut bytes, 9);
        u32_bytes(&mut bytes, 8);
        u64_bytes(&mut bytes, u64::try_from(records.len())?);
        for (piece, _) in records {
            text(&mut bytes, piece)?;
        }
        text(&mut bytes, "tokenizer.ggml.token_type")?;
        u32_bytes(&mut bytes, 9);
        u32_bytes(&mut bytes, 5);
        u64_bytes(&mut bytes, u64::try_from(records.len())?);
        for (_, kind) in records {
            bytes.extend(kind.to_le_bytes());
        }
        text(&mut bytes, "token_embd.weight")?;
        u32_bytes(&mut bytes, 2);
        u64_bytes(&mut bytes, 4096);
        u64_bytes(&mut bytes, vocab_size);
        u32_bytes(&mut bytes, 0);
        u64_bytes(&mut bytes, 0);
        let data_offset = bytes.len().div_ceil(32) * 32;
        let payload_bytes = usize::try_from(vocab_size)? * 4096 * 4;
        bytes.resize(data_offset + payload_bytes, 0);
        bytes[data_offset..data_offset + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        fs::write(&path, &bytes)?;
        Ok(Self { root, path, bytes })
    }

    fn valid() -> Result<Self> {
        Self::new(RECORDS, u64::try_from(RECORDS.len())?, "qwen35")
    }

    fn hash(&self) -> String {
        Sha256::digest(&self.bytes)
            .into_iter()
            .flat_map(|byte| [byte >> 4, byte & 15])
            .map(|nibble| char::from(b"0123456789abcdef"[usize::from(nibble)]))
            .collect()
    }

    fn load(&self) -> Result<TextOutputSession> {
        TextOutputSession::load(
            &self.path,
            u64::try_from(self.bytes.len())?,
            &self.hash(),
            HEADER_BUDGET,
        )
    }
}

impl Drop for Artifact {
    fn drop(&mut self) {
        // Remove only the exact paths created and owned by this fixture.
        let _ = fs::remove_file(&self.path);
        let _ = fs::remove_dir(&self.root);
    }
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

fn push(session: &TextOutputSession, output: &mut Utf8Output, id: u32) -> Result<String> {
    output.push(session.piece(id)?)
}

#[test]
fn session_releases_artifact_and_metadata_without_loading_executable_weights() -> Result<()> {
    let artifact = Artifact::valid()?;
    assert!(artifact.bytes.len() > HEADER_BUDGET);
    let session = TextOutputSession::load(
        &artifact.path,
        u64::try_from(artifact.bytes.len())?,
        &artifact.hash().to_uppercase(),
        HEADER_BUDGET,
    )?;
    assert_eq!(session.identity(), artifact.hash());
    assert_eq!(session.vocab_size(), RECORDS.len());
    assert!(
        Model::load(
            &artifact.path,
            u64::try_from(artifact.bytes.len())?,
            &artifact.hash(),
            1024 * 1024,
        )
        .is_err()
    );
    // Later lookups must use owned pieces without reopening the removed artifact.
    fs::remove_file(&artifact.path)?;
    assert_eq!(session.piece(0)?, b" hello");
    assert_eq!(session.piece(0)?.as_ptr(), session.piece(0)?.as_ptr());
    let debug = format!("{session:?}");
    assert!(debug.contains(session.identity()));
    assert!(!debug.contains("PRIVATE_TEMPLATE"));
    assert!(!debug.contains("hello"));
    assert!(!debug.contains("<stop>"));
    Ok(())
}

#[test]
fn session_preserves_split_utf8_control_spellings_and_arbitrary_bytes() -> Result<()> {
    let session = Artifact::valid()?.load()?;
    assert_eq!(session.piece(1)?, &[0xce]);
    assert_eq!(session.piece(2)?, &[0xbb]);
    assert_eq!(session.piece(3)?, session.piece(4)?);
    assert!(session.is_control(3)?);
    assert!(!session.is_control(4)?);
    assert_eq!(session.piece(5)?, "λ".as_bytes());
    assert_eq!(session.piece(6)?, &[0]);
    assert_eq!(session.piece(7)?, &[0xff]);

    let mut output = Utf8Output::new();
    let mut raw = Vec::new();
    let mut display = String::new();
    for id in [0, 1, 2, 3, 4, 5, 6, 7, 1] {
        raw.extend_from_slice(session.piece(id)?);
        display.push_str(&push(&session, &mut output, id)?);
    }
    display.push_str(output.finish());
    assert_eq!(raw, b" hello\xce\xbb<stop><stop>\xce\xbb\0\xff\xce");
    assert_eq!(display, String::from_utf8_lossy(&raw));
    assert_eq!(output.pending_bytes(), [0_u8; 0]);
    Ok(())
}

#[test]
fn shared_session_supports_independent_caller_owned_output_streams() -> Result<()> {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<TextOutputSession>();
    let session = Artifact::valid()?.load()?;
    let mut first = Utf8Output::new();
    let mut second = Utf8Output::new();
    assert_eq!(push(&session, &mut first, 1)?, "");
    assert_eq!(push(&session, &mut second, 2)?, "\u{fffd}");
    assert_eq!(first.pending_bytes(), &[0xce]);
    assert_eq!(push(&session, &mut second, 1)?, "");
    first.reset();
    assert_eq!(second.pending_bytes(), &[0xce]);
    assert_eq!(push(&session, &mut second, 2)?, "λ");
    assert_eq!(push(&session, &mut first, 0)?, " hello");
    assert_eq!(first.finish(), "");
    assert_eq!(second.finish(), "");
    Ok(())
}

#[test]
fn failed_piece_lookup_and_display_push_preserve_other_requests_and_pending_suffix() -> Result<()> {
    let session = Artifact::valid()?.load()?;
    let held = session.piece(0)?;
    let mut output = Utf8Output::new();
    assert_eq!(push(&session, &mut output, 1)?, "");
    for id in [8, 9, 10, u32::MAX] {
        let error = push(&session, &mut output, id)
            .err()
            .ok_or("unrenderable output ID was accepted")?;
        assert!(!error.to_string().contains("<unknown>"));
        assert_eq!(output.pending_bytes(), &[0xce]);
    }
    assert!(!session.is_control(8)?);
    assert!(!session.is_control(9)?);
    assert!(session.is_control(10).is_err());
    assert!(output.push(&vec![b'x'; MAX_TEXT_OUTPUT_BYTES + 1]).is_err());
    assert_eq!(output.pending_bytes(), &[0xce]);
    assert_eq!(push(&session, &mut output, 2)?, "λ");
    assert_eq!(held, b" hello");
    assert_eq!(session.piece(0)?, held);
    Ok(())
}

#[test]
fn session_rejects_identity_mismatches_and_verifies_complete_payload() -> Result<()> {
    let mut artifact = Artifact::valid()?;
    let expected_bytes = u64::try_from(artifact.bytes.len())?;
    let expected_hash = artifact.hash();
    for (bytes, hash) in [
        (expected_bytes + 1, expected_hash.clone()),
        (expected_bytes, "0".repeat(64)),
        (expected_bytes, "not-a-sha256".to_owned()),
    ] {
        assert!(TextOutputSession::load(&artifact.path, bytes, &hash, HEADER_BUDGET).is_err());
    }
    assert!(TextOutputSession::load(&artifact.path, expected_bytes, &expected_hash, 0).is_err());
    let last = artifact.bytes.last_mut().ok_or("missing fixture payload")?;
    *last ^= 1;
    fs::write(&artifact.path, &artifact.bytes)?;
    assert!(
        TextOutputSession::load(
            &artifact.path,
            expected_bytes,
            &expected_hash,
            HEADER_BUDGET,
        )
        .is_err()
    );
    // The changed payload remains valid metadata and can establish a new identity.
    assert_eq!(artifact.load()?.identity(), artifact.hash());
    Ok(())
}

#[test]
fn session_rejects_malformed_or_unsupported_output_metadata() -> Result<()> {
    for (records, vocab_size, pre) in [
        (&[("a", 1)][..], 2, "qwen35"),
        (&[("a", 1)][..], 1, "qwen2"),
        (&[("λ", 1)][..], 1, "qwen35"),
        (&[("<0xGG>", 6)][..], 1, "qwen35"),
        (&[("secret-piece", 7)][..], 1, "qwen35"),
    ] {
        let artifact = Artifact::new(records, vocab_size, pre)?;
        let error = artifact
            .load()
            .err()
            .ok_or("malformed output vocabulary was accepted")?;
        assert!(!error.to_string().contains("secret-piece"));
    }
    Ok(())
}
