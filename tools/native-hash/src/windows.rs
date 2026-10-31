//! Thread-confined owner of a Windows CNG SHA-256 hash.
#![allow(
    clippy::redundant_pub_crate,
    reason = "Explicit parent-only visibility keeps the platform owner inside its audited boundary."
)]

use std::{ffi::c_void, io, ptr, ptr::NonNull};

use windows_sys::Win32::Security::Cryptography::{
    BCRYPT_SHA256_ALG_HANDLE, BCryptCreateHash, BCryptDestroyHash, BCryptFinishHash, BCryptHashData,
};

// NonNull prevents Send/Sync; no handle is transferred or used concurrently.
pub(super) struct Sha256(NonNull<c_void>);

impl Sha256 {
    pub(super) fn new() -> io::Result<Self> {
        let mut handle = ptr::null_mut();
        // SAFETY: the Windows 10 SHA-256 pseudo-handle needs no close. CNG
        // allocates hash storage with null/zero object arguments. The writable
        // output slot is live; no secret or reusable flag is supplied.
        let status = unsafe {
            BCryptCreateHash(
                BCRYPT_SHA256_ALG_HANDLE,
                &raw mut handle,
                ptr::null_mut(),
                0,
                ptr::null(),
                0,
                0,
            )
        };
        check(status)?;
        NonNull::new(handle)
            .map(Self)
            .ok_or_else(|| io::Error::other("Windows returned an invalid SHA-256 state."))
    }

    pub(super) fn update(&mut self, bytes: &[u8]) -> io::Result<()> {
        // A chunk never exceeds u32::MAX bytes, even for a larger caller buffer.
        for chunk in bytes.chunks(usize::try_from(u32::MAX).map_err(io::Error::other)?) {
            let length = u32::try_from(chunk.len()).map_err(io::Error::other)?;
            // SAFETY: this owner exclusively holds a live hash. The immutable
            // input has exactly length bytes; CNG retains no input pointer.
            check(unsafe { BCryptHashData(self.0.as_ptr(), chunk.as_ptr(), length, 0) })?;
        }
        Ok(())
    }

    pub(super) fn finish(self) -> io::Result<[u8; 32]> {
        let mut digest = [0; 32];
        // SAFETY: the live SHA-256 handle has not been finalized; the output is
        // writable for precisely the algorithm's 32-byte digest size.
        check(unsafe { BCryptFinishHash(self.0.as_ptr(), digest.as_mut_ptr(), 32, 0) })?;
        Ok(digest)
    }
}

impl Drop for Sha256 {
    fn drop(&mut self) {
        // SAFETY: destroy the uniquely owned CNG hash and provider-allocated
        // storage exactly once, after every synchronous operation has returned.
        unsafe { BCryptDestroyHash(self.0.as_ptr()) };
    }
}

fn check(status: i32) -> io::Result<()> {
    if status >= 0 {
        Ok(())
    } else {
        Err(io::Error::other("Windows SHA-256 operation failed."))
    }
}
