//! Pinned Swift process admission and guarded private key reading.
#![allow(
    clippy::redundant_pub_crate,
    reason = "Explicit parent scope keeps these implementation helpers private."
)]
#[cfg(windows)]
use std::ffi::OsString;
#[cfg(windows)]
use std::{io, path::Path};

/// Available physical memory required for the fixed 12 GiB Swift process limit
/// plus the separate 4 GiB host reserve. The launcher enforces the Job limit.
#[cfg(windows)]
pub(super) const REQUIRED_AVAILABLE_BYTES: u64 = 16 * 1024 * 1024 * 1024;

pub(super) fn valid_key(bytes: &[u8]) -> bool {
    bytes.len() == 64
        && bytes
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
}

#[cfg(windows)]
pub(super) fn validate_environment(names: impl IntoIterator<Item = OsString>) -> io::Result<()> {
    for name in names {
        let name = name.to_str().ok_or_else(invalid)?.to_ascii_uppercase();
        if ["LLAMA_", "LLAMACPP_", "GGML_", "AIP_", "MTMD_"]
            .iter()
            .any(|prefix| name.starts_with(prefix))
            || ["HF_TOKEN", "HF_ENDPOINT", "MODEL_ENDPOINT"].contains(&name.as_str())
        {
            return Err(invalid());
        }
    }
    Ok(())
}

#[cfg(windows)]
fn invalid() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        "private inference configuration rejected",
    )
}

#[cfg(windows)]
pub(super) fn read_key(
    path: &Path,
) -> io::Result<(bonsai_windows_credentials::RetainedCredential, String)> {
    let credential = bonsai_windows_credentials::RetainedCredential::open(path)?;
    let key = credential.as_hex().to_owned();
    Ok((credential, key))
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    #[test]
    fn read_key_retains_the_verified_private_reader_and_rejects_invalid_files() -> io::Result<()> {
        let credentials =
            bonsai_windows_credentials::PrivateCredentials::create(&std::env::temp_dir())?;
        let (_reader, key) = read_key(credentials.backend_path())?;
        // Equality failures must not print either credential.
        assert!(key.bytes().eq(credentials.backend_key().bytes()));
        assert!(
            std::fs::OpenOptions::new()
                .write(true)
                .open(credentials.backend_path())
                .is_err()
        );
        let invalid = credentials.directory().join("empty.key");
        std::fs::write(&invalid, [])?;
        assert!(read_key(&invalid).is_err());
        assert!(read_key(credentials.directory()).is_err());
        Ok(())
    }
}
