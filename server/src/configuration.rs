//! Pinned Swift process admission and guarded private key reading.
#![allow(
    clippy::redundant_pub_crate,
    reason = "Explicit parent scope keeps these implementation helpers private."
)]
#[cfg(windows)]
use std::ffi::OsString;
#[cfg(windows)]
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read},
    path::Path,
};

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
pub(super) fn read_key(path: &Path) -> io::Result<(File, String)> {
    use std::os::windows::fs::{MetadataExt as _, OpenOptionsExt as _};
    for ancestor in path.ancestors() {
        let metadata = fs::symlink_metadata(ancestor)?;
        if metadata.file_type().is_symlink()
            || metadata.file_attributes() & 0x400 != 0
            || (ancestor == path && !metadata.is_file())
            || (ancestor != path && !metadata.is_dir())
        {
            return Err(invalid());
        }
    }
    let mut file = OpenOptions::new()
        .read(true)
        .share_mode(1)
        .custom_flags(0x0020_0000)
        .open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.file_attributes() & 0x400 != 0 || metadata.len() != 64 {
        return Err(invalid());
    }
    let mut bytes = [0; 64];
    file.read_exact(&mut bytes)?;
    let mut surplus = [0];
    if file.read(&mut surplus)? != 0 || !valid_key(&bytes) {
        return Err(invalid());
    }
    let key = String::from_utf8(bytes.to_vec()).map_err(|_| invalid())?;
    Ok((file, key))
}
