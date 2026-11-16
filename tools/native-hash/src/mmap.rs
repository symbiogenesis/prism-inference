//! Safe read-only file-mapping views and chunked prefetched SHA-256 verification.

use std::fs::File;
use std::io;
use std::ops::Deref;
#[cfg(windows)]
use std::sync::{Arc, Weak};

#[cfg(windows)]
use crate::Sha256;
#[cfg(windows)]
use crate::mmap_windows::MappingOwner;

const HASH_CHUNK_BYTES: usize = 4 * 1024 * 1024;
#[cfg(windows)]
const PREFETCH_WINDOW_BYTES: usize = 16 * 1024 * 1024;

/// Returns whether read-only file mapping is available on this platform.
#[must_use]
pub const fn is_supported() -> bool {
    cfg!(windows)
}

/// An immutable whole-file read-only memory mapping backed by an open shared file lock.
#[derive(Clone, Debug)]
pub struct MappedReadOnlyFile {
    #[cfg(windows)]
    owner: Arc<MappingOwner>,
    #[cfg(not(windows))]
    _private: (),
}

/// An immutable reference-counted byte range backed by a [`MappedReadOnlyFile`].
#[derive(Clone, Debug)]
pub struct MappedSlice {
    #[cfg(windows)]
    owner: Arc<MappingOwner>,
    #[cfg(windows)]
    offset: usize,
    #[cfg(windows)]
    len: usize,
    #[cfg(not(windows))]
    _private: (),
}

/// A weak non-owning handle to a [`MappedSlice`].
#[derive(Clone, Debug)]
pub struct WeakMappedSlice {
    #[cfg(windows)]
    owner: Weak<MappingOwner>,
    #[cfg(windows)]
    offset: usize,
    #[cfg(windows)]
    len: usize,
    #[cfg(not(windows))]
    _private: (),
}

impl MappedReadOnlyFile {
    /// Maps an already opened read-only `file` covering `expected_bytes`.
    ///
    /// Returns `Ok(None)` on targets where native read-only mapping is not enabled.
    ///
    /// # Errors
    /// Returns an I/O error if the file length differs from `expected_bytes`,
    /// exceeds the address space, or fails OS section/view creation.
    pub fn try_map(file: &File, expected_bytes: u64) -> io::Result<Option<Self>> {
        #[cfg(windows)]
        {
            MappingOwner::map_file(file, expected_bytes).map(|owner| Some(Self { owner }))
        }
        #[cfg(not(windows))]
        {
            let _ = (file, expected_bytes);
            Ok(None)
        }
    }

    /// Returns the total byte length of the mapped file.
    #[must_use]
    pub fn len(&self) -> usize {
        #[cfg(windows)]
        {
            self.owner.as_bytes().len()
        }
        #[cfg(not(windows))]
        {
            0
        }
    }

    /// Returns `true` when the mapped file has zero bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the entire mapped file as an immutable byte slice.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        #[cfg(windows)]
        {
            self.owner.as_bytes()
        }
        #[cfg(not(windows))]
        {
            &[]
        }
    }

    /// Issues an advisory OS prefetch hint for `offset..offset + len` when in bounds.
    pub fn prefetch_range(&self, offset: usize, len: usize) {
        #[cfg(windows)]
        {
            self.owner.prefetch_range(offset, len);
        }
        #[cfg(not(windows))]
        {
            let _ = (offset, len);
        }
    }

    /// Creates a cheap reference-counted zero-copy slice view covering `offset..offset + len`.
    ///
    /// # Errors
    /// Returns `InvalidInput` if `offset + len` overflows or exceeds the mapped file length.
    pub fn slice(&self, offset: usize, len: usize) -> io::Result<MappedSlice> {
        let in_bounds = offset
            .checked_add(len)
            .is_some_and(|end| end <= self.as_bytes().len());
        if !in_bounds {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Mapped tensor slice falls outside the verified file range.",
            ));
        }
        #[cfg(windows)]
        {
            Ok(MappedSlice {
                owner: Arc::clone(&self.owner),
                offset,
                len,
            })
        }
        #[cfg(not(windows))]
        {
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Read-only file mapping is unsupported on this target.",
            ))
        }
    }

    /// Computes the SHA-256 digest of the mapped file in 4 MiB chunks with
    /// OS virtual-memory readahead hints and a per-chunk checkpoint.
    ///
    /// # Errors
    /// Returns any checkpoint cancellation error or OS SHA-256 error.
    pub fn sha256_with_checkpoint(
        &self,
        checkpoint: &mut impl FnMut(u64) -> io::Result<()>,
    ) -> io::Result<[u8; 32]> {
        #[cfg(not(windows))]
        {
            let _ = (checkpoint, HASH_CHUNK_BYTES);
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Read-only file mapping is unsupported on this target.",
            ))
        }
        #[cfg(windows)]
        {
            let bytes = self.as_bytes();
            let total = bytes.len();
            let mut hash = Sha256::new()?;
            let mut offset = 0_usize;
            while offset < total {
                let end = offset.saturating_add(HASH_CHUNK_BYTES).min(total);
                let ahead_end = offset.saturating_add(PREFETCH_WINDOW_BYTES).min(total);
                self.owner.prefetch_range(offset, ahead_end - offset);
                checkpoint(u64::try_from(offset).map_err(io::Error::other)?)?;
                hash.update(&bytes[offset..end])?;
                offset = end;
            }
            checkpoint(u64::try_from(total).map_err(io::Error::other)?)?;
            hash.finish()
        }
    }

    /// Computes the SHA-256 digest of the entire mapped file in a single pass.
    ///
    /// # Errors
    /// Returns an OS SHA-256 error if hashing fails.
    pub fn sha256(&self) -> io::Result<[u8; 32]> {
        self.sha256_with_checkpoint(&mut |_| Ok(()))
    }
}

impl MappedSlice {
    /// Returns the immutable byte slice covered by this view.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        #[cfg(windows)]
        {
            let end = self.offset.saturating_add(self.len);
            &self.owner.as_bytes()[self.offset..end]
        }
        #[cfg(not(windows))]
        {
            &[]
        }
    }

    /// Returns `true` when `this` and `other` reference the same byte range of
    /// the same underlying file mapping.
    #[must_use]
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        #[cfg(windows)]
        {
            Arc::ptr_eq(&this.owner, &other.owner)
                && this.offset == other.offset
                && this.len == other.len
        }
        #[cfg(not(windows))]
        {
            let _ = (this, other);
            false
        }
    }

    /// Returns the strong reference count of the underlying mapping owner.
    #[must_use]
    pub fn strong_count(this: &Self) -> usize {
        #[cfg(windows)]
        {
            Arc::strong_count(&this.owner)
        }
        #[cfg(not(windows))]
        {
            let _ = this;
            0
        }
    }

    /// Creates a weak non-owning handle to this mapped slice.
    #[must_use]
    pub fn downgrade(&self) -> WeakMappedSlice {
        #[cfg(windows)]
        {
            WeakMappedSlice {
                owner: Arc::downgrade(&self.owner),
                offset: self.offset,
                len: self.len,
            }
        }
        #[cfg(not(windows))]
        {
            WeakMappedSlice { _private: () }
        }
    }
}

impl Deref for MappedSlice {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsRef<[u8]> for MappedSlice {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl WeakMappedSlice {
    /// Attempts to upgrade this weak handle back into a [`MappedSlice`].
    #[must_use]
    pub fn upgrade(&self) -> Option<MappedSlice> {
        #[cfg(windows)]
        {
            self.owner.upgrade().map(|owner| MappedSlice {
                owner,
                offset: self.offset,
                len: self.len,
            })
        }
        #[cfg(not(windows))]
        {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash_bytes;
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_ID: AtomicU64 = AtomicU64::new(0);

    struct TempFile(std::path::PathBuf);

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    #[cfg(windows)]
    fn mapped_sha256_and_slices_match_contiguous_bytes() -> io::Result<()> {
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "bonsai-native-hash-mmap-{}-{id}.bin",
            std::process::id()
        ));
        let _guard = TempFile(path.clone());
        let mut payload = vec![0_u8; 5 * 1024 * 1024 + 123];
        for (i, byte) in payload.iter_mut().enumerate() {
            *byte = u8::try_from((i * 131 + 17) & 0xff).map_err(io::Error::other)?;
        }
        {
            let mut file = File::create(&path)?;
            file.write_all(&payload)?;
            file.flush()?;
        }

        let file = File::open(&path)?;
        let expected_len = u64::try_from(payload.len()).map_err(io::Error::other)?;
        let mapped = MappedReadOnlyFile::try_map(&file, expected_len)?
            .ok_or_else(|| io::Error::other("expected Windows file mapping"))?;
        assert_eq!(mapped.as_bytes().len(), payload.len());
        assert_eq!(mapped.sha256()?, hash_bytes(&payload)?);

        let a = mapped.slice(1024, 4096)?;
        let a_clone = a.clone();
        let b = mapped.slice(1024 + 4096, 4096)?;
        assert_eq!(&*a, &payload[1024..5120]);
        assert_eq!(&*b, &payload[5120..9216]);
        assert!(MappedSlice::ptr_eq(&a, &a_clone));
        assert!(!MappedSlice::ptr_eq(&a, &b));

        let weak = a.downgrade();
        assert!(weak.upgrade().is_some());
        drop(mapped);
        drop(a);
        drop(a_clone);
        assert!(weak.upgrade().is_some());
        drop(b);
        assert!(weak.upgrade().is_none());

        let mut recheck = File::open(&path)?;
        recheck.seek(SeekFrom::Start(0))?;
        let mut head = [0_u8; 4];
        recheck.read_exact(&mut head)?;
        assert_eq!(head, payload[..4]);
        Ok(())
    }
}
