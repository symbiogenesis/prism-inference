//! Portable Rust baseline for non-Windows targets.
#![allow(
    clippy::redundant_pub_crate,
    reason = "Explicit parent-only visibility keeps the platform owner private."
)]

use std::io;

use sha2::Digest;

pub(super) struct Sha256(sha2::Sha256);

impl Sha256 {
    #[allow(
        clippy::unnecessary_wraps,
        reason = "The platform-independent owner shares the fallible CNG boundary's API."
    )]
    pub(super) fn new() -> io::Result<Self> {
        Ok(Self(sha2::Sha256::new()))
    }

    #[allow(
        clippy::unnecessary_wraps,
        reason = "The platform-independent owner shares the fallible CNG boundary's API."
    )]
    pub(super) fn update(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.0.update(bytes);
        Ok(())
    }

    #[allow(
        clippy::unnecessary_wraps,
        reason = "The platform-independent owner shares the fallible CNG boundary's API."
    )]
    pub(super) fn finish(self) -> io::Result<[u8; 32]> {
        Ok(self.0.finalize().into())
    }
}
