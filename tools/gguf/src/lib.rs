//! Bounded GGUF v3 tooling for verified artifacts, with opt-in pinned Prism types.
//!
//! Inspection skips tensor data unless a PQ2 census is requested. PTQ1 export
//! preserves raw weight symbols and scales; inference equivalence and performance
//! remain unqualified. PowerShell retains provisioning and external-path policy.

#![forbid(unsafe_code)]

#[cfg(feature = "report")]
mod census;
#[cfg(feature = "report")]
mod export;
mod inspect;
mod metadata;

#[cfg(feature = "report")]
use std::fs;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
#[cfg(feature = "report")]
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[cfg(feature = "report")]
use serde_json::Value;
use sha2::{Digest, Sha256};

pub use inspect::TensorDescriptor;
pub use metadata::{Metadata, MetadataArray, MetadataValue, MetadataValueRef};

/// The only supported private GGUF tensor dialect.
pub const PRISM_DIALECT: &str = "Prism-73bfe6cc485554822f80c473a3c0c4f25e80f41b";
/// Immutable Prism codec revision used by the private dialect.
pub const CODEC_COMMIT: &str = "73bfe6cc485554822f80c473a3c0c4f25e80f41b";
/// A failed I/O, artifact verification, or bounded format operation.
pub type Error = Box<dyn std::error::Error + Send + Sync>;
/// A checked tooling result.
pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn bad(message: impl Into<String>) -> Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()).into()
}

fn open_read(path: &Path) -> Result<File> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.share_mode(1); // FILE_SHARE_READ: deny write/delete during use.
    }
    let file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err(bad("GGUF: a regular input file is required."));
    }
    Ok(file)
}

pub(crate) fn digest_hex(hash: Sha256) -> String {
    hash.finalize()
        .into_iter()
        .flat_map(|byte| [byte >> 4, byte & 0x0f])
        .map(|nibble| char::from(b"0123456789abcdef"[usize::from(nibble)]))
        .collect()
}
pub(crate) fn hash_range(file: &mut File, start: u64, mut length: u64) -> Result<String> {
    let restore = file.stream_position()?;
    let result = (|| {
        file.seek(SeekFrom::Start(start))?;
        let mut hash = Sha256::new();
        let mut buffer = vec![0_u8; 65_536];
        while length != 0 {
            let count = usize::try_from(length.min(65_536))?;
            file.read_exact(&mut buffer[..count])?;
            hash.update(&buffer[..count]);
            length -= u64::try_from(count)?;
        }
        Ok(digest_hex(hash))
    })();
    file.seek(SeekFrom::Start(restore))?;
    result
}

fn verify(file: &mut File, expected_bytes: u64, expected_sha256: &str) -> Result<()> {
    if expected_bytes == 0
        || expected_sha256.len() != 64
        || !expected_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(bad(
            "GGUF: a positive byte count and 64-digit SHA-256 are required.",
        ));
    }
    if file.metadata()?.len() != expected_bytes {
        return Err(bad("GGUF: artifact size mismatch."));
    }
    if !hash_range(file, 0, expected_bytes)?.eq_ignore_ascii_case(expected_sha256) {
        return Err(bad("GGUF: artifact SHA-256 mismatch."));
    }
    Ok(())
}

/// Verifies and retains bounded vocabulary metadata, including legacy GGUF v2.
///
/// Version 2 is admitted only for files without model tensors. Version 3 retains
/// the ordinary stock descriptor checks; no tensor payload is read. Verified
/// model execution and artifact inspection retain their strict v3 contract.
///
/// # Errors
/// Rejects identity mismatches, unsupported versions, v2 model tensors, malformed
/// metadata, invalid extents and metadata exceeding `memory_limit`.
pub fn read_vocabulary(
    path: &Path,
    expected_bytes: u64,
    expected_sha256: &str,
    memory_limit: usize,
) -> Result<Metadata> {
    let mut file = open_read(path)?;
    verify(&mut file, expected_bytes, expected_sha256)?;
    Ok(inspect::inspect_vocabulary(&mut file, path, memory_limit)?.metadata)
}

/// A bounded model header and the exact file handle whose identity was checked.
///
/// Unlike the inspection report, this retains tokenizer strings and complete
/// arrays. Header retention is charged conservatively against the supplied
/// budget before allocating string/array payloads. Parsing additionally uses a
/// fixed 64 KiB input buffer and bounded descriptor-validation bookkeeping.
/// Buffered reads can include bytes beyond the header; no tensor payload is
/// materialized or retained by header parsing. Later payload reads seek to
/// their validated absolute offsets through the same verified file handle.
/// On Windows the handle denies concurrent writes and deletion for its lifetime.
pub struct VerifiedModel {
    file: File,
    inspection: inspect::Inspection,
}

impl VerifiedModel {
    /// Verifies file identity and parses the existing bounded GGUF v3 dialect.
    ///
    /// # Errors
    /// Rejects identity mismatches, malformed metadata/descriptors, unsupported
    /// dialects/encodings, invalid extents, allocation failures, and metadata
    /// exceeding `memory_limit`. No tensor payload is retained by this operation.
    pub fn open(
        path: &Path,
        dialect: &str,
        expected_bytes: u64,
        expected_sha256: &str,
        memory_limit: usize,
    ) -> Result<Self> {
        let mut file = open_read(path)?;
        verify(&mut file, expected_bytes, expected_sha256)?;
        let inspection = inspect::inspect_retained(&mut file, path, dialect, Some(memory_limit))?;
        Ok(Self { file, inspection })
    }

    /// Returns exact retained scalar, tokenizer, array and template metadata.
    #[must_use]
    pub const fn metadata(&self) -> &Metadata {
        &self.inspection.metadata
    }

    /// Returns validated tensor descriptors in file descriptor order.
    #[must_use]
    pub fn tensors(&self) -> &[TensorDescriptor] {
        &self.inspection.tensors
    }

    /// Returns the conservative retained-header charge against the model budget.
    #[must_use]
    pub const fn retained_bytes(&self) -> usize {
        self.inspection.retained_bytes
    }

    /// Reads one tensor from the verified handle within a caller's byte budget.
    ///
    /// `max_bytes` is the remaining payload budget, rather than a fresh budget
    /// for every tensor. Callers retaining multiple tensors must subtract each
    /// successful payload length before making another call.
    ///
    /// # Errors
    /// Rejects missing tensors, payloads exceeding `max_bytes`, allocation
    /// failures, checked-offset overflow, and truncated or unreadable payloads.
    pub fn read_tensor(&mut self, index: usize, max_bytes: usize) -> Result<Vec<u8>> {
        let tensor = self
            .inspection
            .tensors
            .get(index)
            .ok_or_else(|| bad("GGUF: tensor index is out of bounds."))?;
        let length = usize::try_from(tensor.bytes)?;
        if length > max_bytes {
            return Err(bad("GGUF: tensor payload exceeds remaining memory budget."));
        }
        let offset = self
            .inspection
            .data_offset
            .checked_add(tensor.offset)
            .ok_or_else(|| bad("GGUF: tensor payload offset overflow."))?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(length)?;
        bytes.resize(length, 0);
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    /// Consumes the verified reader and transfers its retained metadata.
    #[must_use]
    pub fn into_metadata(self) -> Metadata {
        self.inspection.metadata
    }
}

/// Verifies an artifact and inspects metadata, optionally streaming PQ2 payloads.
///
/// # Errors
/// Rejects identity mismatches, unsupported dialects, malformed or over-limit
/// metadata/descriptors, invalid extents, and I/O failures.
#[cfg(feature = "report")]
pub fn inspect(
    path: &Path,
    dialect: &str,
    census: bool,
    expected_bytes: u64,
    expected_sha256: &str,
) -> Result<Value> {
    if census && dialect != PRISM_DIALECT {
        return Err(bad(
            "GGUF: PQ2 census requires the explicitly pinned Prism dialect.",
        ));
    }
    let mut file = open_read(path)?;
    verify(&mut file, expected_bytes, expected_sha256)?;
    let mut inspection = inspect::inspect(&mut file, path, dialect)?;
    if census {
        inspection.report["Pq2Census"] = census::census(&mut file, &inspection)?;
        inspection.report["TensorPayloadRead"] = true.into();
    }
    inspection.report["ArtifactSha256"] = expected_sha256.to_ascii_lowercase().into();
    inspection.report["ArtifactVerification"] = "verified-size-and-sha256".into();
    Ok(inspection.report)
}

/// Verifies and exports an experimental PTQ1 GGUF to a new output name.
///
/// Writes a unique sibling partial, validates its identity and metadata, then
/// publishes using a hard link that cannot replace an existing destination.
/// Failed partials remain available for diagnosis. Publication requires a
/// filesystem supporting hard links, such as NTFS.
///
/// # Errors
/// Rejects artifact mismatches, invalid tensors/scales/symbols, deadlines outside
/// one second through one hour, existing output paths, and filesystem failures.
#[cfg(feature = "report")]
pub fn export_ptq1(
    source: &Path,
    output: &Path,
    expected_bytes: u64,
    expected_sha256: &str,
    timeout: Duration,
) -> Result<Value> {
    if timeout < Duration::from_secs(1) || timeout > Duration::from_secs(3600) {
        return Err(bad(
            "GGUF: export deadline must be between one second and one hour.",
        ));
    }
    if output.try_exists()? {
        return Err(bad("GGUF: output must name a new file."));
    }
    let mut file = open_read(source)?;
    verify(&mut file, expected_bytes, expected_sha256)?;
    let inspection = inspect::inspect(&mut file, source, PRISM_DIALECT)?;
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let mut partial_name = output.as_os_str().to_os_string();
    partial_name.push(format!(".partial-{}-{nonce}", std::process::id()));
    let partial = std::path::PathBuf::from(partial_name);
    let result = (|| {
        let mut report = export::export(&mut file, &partial, &inspection, timeout)?;
        let mut exported = open_read(&partial)?;
        let output_bytes = report["OutputBytes"]
            .as_u64()
            .ok_or_else(|| bad("missing output size"))?;
        let output_hash = report["OutputSha256"]
            .as_str()
            .ok_or_else(|| bad("missing output hash"))?;
        verify(&mut exported, output_bytes, output_hash)?;
        let checked = inspect::inspect(&mut exported, &partial, PRISM_DIALECT)?;
        if checked.report["TokenizerMetadataSha256"] != inspection.report["TokenizerMetadataSha256"]
            || checked.report["TensorCount"] != inspection.report["TensorCount"]
        {
            return Err(bad("GGUF: export metadata identity differs."));
        }
        report["SourceSha256"] = expected_sha256.to_ascii_lowercase().into();
        report["TokenizerMetadataSha256"] = checked.report["TokenizerMetadataSha256"].clone();
        report["OutputPath"] = output.to_string_lossy().into_owned().into();
        fs::hard_link(&partial, output)?;
        drop(exported);
        if let Err(error) = fs::remove_file(&partial) {
            // Publication already succeeded. Report a retained alias instead of
            // claiming that the validated final output failed to publish.
            report["RetainedPartialPath"] = partial.to_string_lossy().into_owned().into();
            report["PartialCleanupError"] = error.to_string().into();
        }
        Ok(report)
    })();
    result.map_err(|error: Error| {
        bad(format!(
            "{error} Partial path (retained if created): {}",
            partial.display()
        ))
    })
}
