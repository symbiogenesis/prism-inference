//! Audited Windows read-only file-mapping owner.
#![expect(
    clippy::redundant_pub_crate,
    reason = "Explicit crate-only visibility keeps raw OS mapping handles behind the safe wrapper."
)]

use std::ffi::c_void;
use std::fs::File;
use std::io;
use std::os::windows::io::AsRawHandle;
use std::ptr::{self, NonNull};
use std::sync::Arc;

use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, FILE_MAP_READ, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, PAGE_READONLY,
    PrefetchVirtualMemory, UnmapViewOfFile, WIN32_MEMORY_RANGE_ENTRY,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

struct MappingHandle(HANDLE);

// SAFETY: a read-only Windows section handle is not thread-affine and is closed once on Drop.
unsafe impl Send for MappingHandle {}

// SAFETY: shared references never mutate or close `MappingHandle` before Drop.
unsafe impl Sync for MappingHandle {}

impl Drop for MappingHandle {
    fn drop(&mut self) {
        // SAFETY: `self.0` was returned by `CreateFileMappingW` and is closed once here
        // after any mapped view derived from it has already been unmapped.
        unsafe { CloseHandle(self.0) };
    }
}

struct MappedView(NonNull<u8>);

// SAFETY: a read-only mapped view address is process-wide valid and only unmapped on Drop.
unsafe impl Send for MappedView {}

// SAFETY: concurrent shared reads from `&MappedView` observe immutable bytes until Drop.
unsafe impl Sync for MappedView {}

impl Drop for MappedView {
    fn drop(&mut self) {
        let address = MEMORY_MAPPED_VIEW_ADDRESS {
            Value: self.0.as_ptr().cast(),
        };
        // SAFETY: `self.0` is the live view base returned by `MapViewOfFile` and is
        // unmapped exactly once when all references to `MappingOwner` have dropped.
        unsafe { UnmapViewOfFile(address) };
    }
}

pub(crate) struct MappingOwner {
    len: usize,
    view: Option<MappedView>,
    _mapping: Option<MappingHandle>,
    _file: File,
}

impl std::fmt::Debug for MappingOwner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MappingOwner")
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl MappingOwner {
    pub(crate) fn map_file(file: &File, expected_bytes: u64) -> io::Result<Arc<Self>> {
        let owned = file.try_clone()?;
        let actual_len = owned.metadata()?.len();
        if actual_len != expected_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Mapped file length does not match expected size.",
            ));
        }
        let len = usize::try_from(expected_bytes)
            .map_err(|_| io::Error::other("Mapped file length exceeds address space."))?;
        if len == 0 {
            return Ok(Arc::new(Self {
                len: 0,
                view: None,
                _mapping: None,
                _file: owned,
            }));
        }

        // SAFETY: `owned` is an open read-only file handle whose size was verified to be
        // nonzero; `PAGE_READONLY` with zero max size creates an immutable section for
        // the file's entire length without security attributes or global name.
        let handle = unsafe {
            CreateFileMappingW(
                owned.as_raw_handle(),
                ptr::null(),
                PAGE_READONLY,
                0,
                0,
                ptr::null(),
            )
        };
        if handle.is_null() {
            return Err(io::Error::last_os_error());
        }
        let mapping = MappingHandle(handle);

        // SAFETY: `mapping.0` is a valid read-only section object covering `len` bytes;
        // requesting a read-only view from offset 0 returns either null or a valid
        // view base at least `len` bytes long.
        let view_addr = unsafe { MapViewOfFile(mapping.0, FILE_MAP_READ, 0, 0, len) };
        let Some(base) = NonNull::new(view_addr.Value.cast::<u8>()) else {
            return Err(io::Error::last_os_error());
        };
        let view = MappedView(base);

        Ok(Arc::new(Self {
            len,
            view: Some(view),
            _mapping: Some(mapping),
            _file: owned,
        }))
    }

    pub(crate) const fn as_bytes(&self) -> &[u8] {
        let Some(view) = &self.view else {
            return &[];
        };
        // SAFETY: when `self.view` is `Some`, `self.len > 0` and `view.0` points to
        // `self.len` immutable mapped bytes kept alive for writes until `Drop`.
        unsafe { std::slice::from_raw_parts(view.0.as_ptr(), self.len) }
    }

    pub(crate) fn prefetch_range(&self, offset: usize, len: usize) {
        let Some(slice) = offset
            .checked_add(len)
            .and_then(|end| self.as_bytes().get(offset..end))
        else {
            return;
        };
        if slice.is_empty() {
            return;
        }
        let mut entry = WIN32_MEMORY_RANGE_ENTRY {
            VirtualAddress: slice.as_ptr().cast_mut().cast::<c_void>(),
            NumberOfBytes: slice.len(),
        };
        // SAFETY: `GetCurrentProcess` returns the pseudo-handle for the calling process.
        let process = unsafe { GetCurrentProcess() };
        // SAFETY: `process` is the current process pseudo-handle and `entry` points to
        // a single live `WIN32_MEMORY_RANGE_ENTRY` describing a valid mapped slice.
        let _ = unsafe { PrefetchVirtualMemory(process, 1, &raw mut entry, 0) };
    }
}
