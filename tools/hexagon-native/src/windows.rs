//! Native calls are deliberately limited to SCM, signature, loader and query.

use super::{CDSP_DOMAIN, Capability, CapabilityRequest, Error, GET_DSP_INFO};
use std::ffi::{c_char, c_void};
use std::fs::{File, OpenOptions};
use std::marker::PhantomData;
use std::os::windows::ffi::OsStrExt as _;
use std::os::windows::fs::{MetadataExt as _, OpenOptionsExt as _};
use std::os::windows::io::AsRawHandle as _;
use std::path::{Component, Path, PathBuf, Prefix};
use std::ptr::{NonNull, null, null_mut};
use std::rc::Rc;
use std::sync::{Mutex, MutexGuard};

#[path = "mapping.rs"]
mod mapping;

type Handle = *mut c_void;
type Control = unsafe extern "C" fn(u32, *mut c_void, u32) -> i32;

static DRIVER_LEASE: Mutex<()> = Mutex::new(());

#[repr(C)]
struct ServiceConfig {
    service_type: u32,
    start_type: u32,
    error_control: u32,
    binary_path: *mut u16,
    load_order: *mut u16,
    tag_id: u32,
    dependencies: *mut u16,
    start_name: *mut u16,
    display_name: *mut u16,
}

#[repr(C)]
struct Guid {
    data1: u32,
    data2: u16,
    data3: u16,
    data4: [u8; 8],
}

#[repr(C)]
struct TrustFile {
    size: u32,
    path: *const u16,
    file: Handle,
    subject: *mut Guid,
}

#[repr(C)]
struct TrustData {
    size: u32,
    policy: *mut c_void,
    sip: *mut c_void,
    ui: u32,
    revocation: u32,
    choice: u32,
    file: *mut TrustFile,
    action: u32,
    state: Handle,
    url: *mut u16,
    flags: u32,
    context: u32,
    signature_settings: *mut c_void,
}

#[link(name = "advapi32")]
unsafe extern "system" {
    fn OpenSCManagerW(machine: *const u16, database: *const u16, access: u32) -> Handle;
    fn OpenServiceW(manager: Handle, name: *const u16, access: u32) -> Handle;
    fn QueryServiceConfigW(
        service: Handle,
        config: *mut ServiceConfig,
        size: u32,
        needed: *mut u32,
    ) -> i32;
    fn CloseServiceHandle(handle: Handle) -> i32;
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetLastError() -> u32;
    fn GetWindowsDirectoryW(buffer: *mut u16, size: u32) -> u32;
    fn LoadLibraryExW(path: *const u16, file: Handle, flags: u32) -> Handle;
    fn GetProcAddress(module: Handle, name: *const c_char) -> *mut c_void;
    fn FreeLibrary(module: Handle) -> i32;
}

#[link(name = "wintrust")]
unsafe extern "system" {
    fn WinVerifyTrust(window: Handle, action: *mut Guid, data: *mut c_void) -> i32;
}

fn last_error(stage: &'static str) -> Error {
    // SAFETY: GetLastError takes no pointers and preserves the last native code.
    Error::Native(stage, i64::from(unsafe { GetLastError() }))
}

struct ServiceHandle(NonNull<c_void>);

impl Drop for ServiceHandle {
    fn drop(&mut self) {
        // SAFETY: each successful SCM/service handle has one owner here.
        unsafe {
            CloseServiceHandle(self.0.as_ptr());
        }
    }
}

fn wide(path: &Path) -> Result<Vec<u16>, Error> {
    let mut value: Vec<u16> = path.as_os_str().encode_wide().collect();
    if value.contains(&0) || value.len() >= 32767 {
        return Err(Error::InvalidDriverPath);
    }
    value.push(0);
    Ok(value)
}

fn windows_directory() -> Result<PathBuf, Error> {
    let mut output = vec![0_u16; 32768];
    // SAFETY: the output buffer has the full declared capacity and stays live.
    let length = unsafe { GetWindowsDirectoryW(output.as_mut_ptr(), 32768) };
    if length == 0 {
        return Err(last_error("windows-directory"));
    }
    let length = usize::try_from(length).map_err(|_| Error::InvalidDriverPath)?;
    if length >= output.len() {
        return Err(Error::InvalidDriverPath);
    }
    let value = String::from_utf16(&output[..length]).map_err(|_| Error::InvalidDriverPath)?;
    Ok(PathBuf::from(value))
}

fn service_binary() -> Result<String, Error> {
    // SAFETY: null machine/database select the local active SCM; access is query-only.
    let manager = unsafe { OpenSCManagerW(null(), null(), 1) };
    let manager = ServiceHandle(NonNull::new(manager).ok_or_else(|| last_error("scm-open"))?);
    let name: Vec<u16> = "qcnspmcdm\0".encode_utf16().collect();
    // SAFETY: manager is live and the service name is a terminated UTF-16 string.
    let service = unsafe { OpenServiceW(manager.0.as_ptr(), name.as_ptr(), 1) };
    let service = ServiceHandle(NonNull::new(service).ok_or_else(|| last_error("service-open"))?);
    let mut needed = 0;
    // SAFETY: this is the documented zero-buffer size query; needed is writable.
    let status = unsafe { QueryServiceConfigW(service.0.as_ptr(), null_mut(), 0, &raw mut needed) };
    if status != 0 || !(64..=65536).contains(&needed) {
        return Err(Error::InvalidDriverPath);
    }
    // A usize vector supplies the alignment required by QUERY_SERVICE_CONFIGW.
    let words = usize::try_from(needed)
        .map_err(|_| Error::InvalidDriverPath)?
        .div_ceil(size_of::<usize>());
    let mut storage = vec![0_usize; words];
    let config = storage.as_mut_ptr().cast::<ServiceConfig>();
    // SAFETY: storage is aligned, writable and at least needed bytes long.
    if unsafe { QueryServiceConfigW(service.0.as_ptr(), config, needed, &raw mut needed) } == 0 {
        return Err(last_error("service-config"));
    }
    // SAFETY: successful QueryServiceConfigW initializes the header in storage.
    let binary = unsafe { (*config).binary_path };
    let start = storage.as_ptr().addr();
    let end = start + storage.len() * size_of::<usize>();
    if binary.is_null() || binary.addr() < start || binary.addr() >= end || binary.addr() % 2 != 0 {
        return Err(Error::InvalidDriverPath);
    }
    let count = (end - binary.addr()) / 2;
    // SAFETY: the pointer was checked inside initialized zeroed allocation bounds.
    let buffer = unsafe { std::slice::from_raw_parts(binary, count) };
    let length = buffer
        .iter()
        .position(|value| *value == 0)
        .ok_or(Error::InvalidDriverPath)?;
    String::from_utf16(&buffer[..length]).map_err(|_| Error::InvalidDriverPath)
}

fn no_reparse(path: &Path) -> Result<(), Error> {
    for ancestor in path.ancestors() {
        let metadata = std::fs::symlink_metadata(ancestor).map_err(Error::File)?;
        if metadata.file_attributes() & 0x400 != 0 {
            return Err(Error::InvalidDriverPath);
        }
    }
    Ok(())
}

#[allow(
    clippy::redundant_pub_crate,
    reason = "Native helpers stay restricted to their safe parent rather than becoming unreachable public API"
)]
pub(super) fn inspect_system() -> Result<PathBuf, Error> {
    let system = windows_directory()?;
    let service = service_binary()?;
    if service.chars().any(char::is_control) || service.contains('"') || service.contains('/') {
        return Err(Error::InvalidDriverPath);
    }
    let prefix = "\\SystemRoot\\";
    let normalized = if service
        .to_ascii_lowercase()
        .starts_with(&prefix.to_ascii_lowercase())
    {
        system.join(&service[prefix.len()..])
    } else {
        PathBuf::from(service)
    };
    if !normalized.is_absolute()
        || normalized.components().any(|component| match component {
            Component::Prefix(prefix) => !matches!(prefix.kind(), Prefix::Disk(_)),
            Component::ParentDir | Component::CurDir => true,
            Component::Normal(value) => value.to_string_lossy().contains(':'),
            Component::RootDir => false,
        })
    {
        return Err(Error::InvalidDriverPath);
    }
    // Normalize repeated interior separators using Windows path components,
    // after refusing UNC/device prefixes and traversal. Do not canonicalize to
    // a verbatim namespace or follow a reparse point.
    let normalized: PathBuf = normalized.components().collect();
    let dll = normalized
        .parent()
        .ok_or(Error::InvalidDriverPath)?
        .join("libcdsprpc.dll");
    let allowed = system.join("System32/DriverStore/FileRepository");
    let allowed = allowed
        .to_string_lossy()
        .replace('/', "\\")
        .to_ascii_lowercase()
        + "\\";
    if !dll
        .to_string_lossy()
        .to_ascii_lowercase()
        .starts_with(&allowed)
    {
        return Err(Error::InvalidDriverPath);
    }
    no_reparse(&dll)?;
    if !dll.is_file() {
        return Err(Error::InvalidDriverPath);
    }
    Ok(dll)
}

fn verify_signature(path: &Path, file: &File) -> Result<(), Error> {
    let wide = wide(path)?;
    let mut info = TrustFile {
        size: 32,
        path: wide.as_ptr(),
        file: file.as_raw_handle(),
        subject: null_mut(),
    };
    let mut data = TrustData {
        size: 88,
        policy: null_mut(),
        sip: null_mut(),
        ui: 2,
        revocation: 0,
        choice: 1,
        file: &raw mut info,
        action: 1,
        state: null_mut(),
        url: null_mut(),
        flags: 0x1000,
        context: 0,
        signature_settings: null_mut(),
    };
    let mut action = Guid {
        data1: 0x00aa_c56b,
        data2: 0xcd44,
        data3: 0x11d0,
        data4: [0x8c, 0xc2, 0, 0xc0, 0x4f, 0xc2, 0x95, 0xee],
    };
    // SAFETY: all documented WINTRUST_DATA/FILE_INFO pointers stay live; no UI
    // or network retrieval is permitted and the file handle denies mutation.
    let result = unsafe {
        WinVerifyTrust(
            std::ptr::without_provenance_mut(usize::MAX),
            &raw mut action,
            std::ptr::from_mut(&mut data).cast(),
        )
    };
    data.action = 2;
    // SAFETY: close uses the same live state and input objects after verification.
    let close = unsafe {
        WinVerifyTrust(
            std::ptr::without_provenance_mut(usize::MAX),
            &raw mut action,
            std::ptr::from_mut(&mut data).cast(),
        )
    };
    if result != 0 {
        return Err(Error::Native(
            "authenticode-cache-policy",
            i64::from(result),
        ));
    }
    if close != 0 {
        return Err(Error::Native("authenticode-state-close", i64::from(close)));
    }
    Ok(())
}

#[allow(
    clippy::redundant_pub_crate,
    reason = "The raw native owner is intentionally restricted to its safe parent API"
)]
pub(super) struct Driver {
    library: NonNull<c_void>,
    control: Control,
    _file: File,
    // The vendor control API has no established concurrent-call contract here.
    _single_thread: PhantomData<Rc<()>>,
    // One native owner across all threads/instances; held through unload/file
    // close, or permanently pinned by a terminal mapping owner's ManuallyDrop.
    _lease: MutexGuard<'static, ()>,
}

impl Driver {
    pub(super) fn open(path: &Path) -> Result<Self, Error> {
        let lease = DRIVER_LEASE.try_lock().map_err(|_| Error::Contract)?;
        if mapping::terminal() { return Err(Error::Contract); }
        no_reparse(path)?;
        let file = OpenOptions::new()
            .read(true)
            .share_mode(1)
            .open(path)
            .map_err(Error::File)?;
        if file.metadata().map_err(Error::File)?.len() > 16 * 1024 * 1024 {
            return Err(Error::InvalidDriverPath);
        }
        verify_signature(path, &file)?;
        let wide = wide(path)?;
        // SAFETY: this absolute, validated, signed DriverStore DLL is retained
        // deny-write/delete. Dependencies search only its directory and System32.
        let library = unsafe { LoadLibraryExW(wide.as_ptr(), null_mut(), 0x100 | 0x800) };
        let library = NonNull::new(library).ok_or_else(|| last_error("vendor-driver-load"))?;
        // SAFETY: library is live; the fixed ASCII export name is NUL terminated.
        let symbol = unsafe { GetProcAddress(library.as_ptr(), c"remote_handle_control".as_ptr()) };
        if symbol.is_null() {
            let error = last_error("capability-export");
            // SAFETY: no pointers escaped and this owns the one successful load.
            unsafe {
                FreeLibrary(library.as_ptr());
            }
            return Err(error);
        }
        // SAFETY: pinned vendor/ggml sources establish the exact C function ABI.
        let control = unsafe { std::mem::transmute::<*mut c_void, Control>(symbol) };
        Ok(Self {
            library,
            control,
            _file: file,
            _single_thread: PhantomData,
            _lease: lease,
        })
    }

    pub(super) fn capability(&mut self, capability: Capability) -> Result<u32, Error> {
        if mapping::terminal() { return Err(Error::Contract); }
        let mut query = CapabilityRequest {
            domain: CDSP_DOMAIN,
            attribute_id: capability.id(),
            capability: 0,
        };
        // SAFETY: the only exposed request writes the fixed 12-byte C structure;
        // the library and deny-mutation file guard remain owned throughout.
        let result =
            unsafe { (self.control)(GET_DSP_INFO, std::ptr::from_mut(&mut query).cast(), 12) };
        if result != 0 {
            return Err(Error::Native("capability-query", i64::from(result)));
        }
        if query.domain != CDSP_DOMAIN || query.attribute_id != capability.id() {
            return Err(Error::Contract);
        }
        Ok(query.capability)
    }
}

#[allow(clippy::redundant_pub_crate, reason = "Only the safe parent may consume the private native owner")]
pub(super) fn mapping_probe(driver: Driver) -> Result<crate::MappingProbeReport, Error> {
    mapping::probe(driver)
}

impl Drop for Driver {
    fn drop(&mut self) {
        // SAFETY: this owner completed only fixed queries or pre-mapping cleanup.
        // Terminal mapping owners never reach Drop after a mapping attempt;
        // their DLL/file lease remains pinned until supervised process death.
        unsafe {
            FreeLibrary(self.library.as_ptr());
        }
    }
}

const _: () = assert!(size_of::<TrustFile>() == 32);
const _: () = assert!(size_of::<TrustData>() == 88);
const _: () = assert!(size_of::<ServiceConfig>() == 64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_lease_rejects_another_owner_without_waiting() -> Result<(), Error> {
        let first = DRIVER_LEASE.try_lock().map_err(|_| Error::Contract)?;
        assert!(DRIVER_LEASE.try_lock().is_err());
        drop(first);
        let second = DRIVER_LEASE.try_lock().map_err(|_| Error::Contract)?;
        drop(second);
        Ok(())
    }
}
