//! Complete, bounded SHA-256 streams without retaining their payloads.
//!
//! Windows uses the operating system's CPU-dispatched CNG provider through an
//! isolated audited boundary. Other targets retain the portable Rust baseline.
//! Provider, read, length and cancellation failures never return a digest.

#![deny(unsafe_code)]

use std::io::{self, Read};

#[cfg(windows)]
#[allow(
    unsafe_code,
    reason = "The private audited CNG owner retains its handle and borrows input only during synchronous calls."
)]
mod windows;
#[cfg(windows)]
use windows::Sha256;

#[cfg(not(windows))]
mod portable;
#[cfg(not(windows))]
use portable::Sha256;

/// Hashes exactly `expected` bytes from the current position through EOF.
///
/// The caller owns a reusable, nonempty read buffer and a cancellation checkpoint
/// invoked before every read, including retries after interruption. No payload is
/// retained and no input is decoded or normalized. This function does not seek,
/// open files or change their sharing/identity protections. Use a bounded reader
/// when hashing one range within a larger stream.
///
/// # Errors
/// Rejects an empty buffer, short/long streams, overflow, failed reads or native
/// hashing, and checkpoint failure. No partial digest is returned on any error.
pub fn hash_reader(
    reader: &mut impl Read,
    expected: u64,
    buffer: &mut [u8],
    checkpoint: &mut impl FnMut() -> io::Result<()>,
) -> io::Result<[u8; 32]> {
    if buffer.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "SHA-256 requires a nonempty read buffer.",
        ));
    }
    let mut hash = Sha256::new()?;
    let mut total = 0_u64;
    loop {
        checkpoint()?;
        let count = match reader.read(buffer) {
            Ok(0) => break,
            Ok(count) => count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        };
        total = total
            .checked_add(u64::try_from(count).map_err(io::Error::other)?)
            .ok_or_else(|| io::Error::other("SHA-256 stream length overflow."))?;
        if total > expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "SHA-256 stream exceeds the expected length.",
            ));
        }
        hash.update(&buffer[..count])?;
    }
    if total != expected {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "SHA-256 stream is shorter than the expected length.",
        ));
    }
    hash.finish()
}
