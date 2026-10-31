//! Fixed signed-module open followed by one established-session FD-map cycle.

use super::{Driver, GetProcAddress, last_error, mapping, no_reparse, session_native};
use crate::{Error, MAP_FD, SessionMappingReport, SessionPreparationReport};
use std::ffi::{CStr, c_char, c_void};
use std::fs::{File, OpenOptions};
use std::mem::ManuallyDrop;
use std::os::windows::fs::OpenOptionsExt as _;
use std::path::{Component, Path, Prefix};

type Open = unsafe extern "C" fn(*const c_char, *mut u64) -> i32;
// Use the authoritative vendor remote_handle64 typedef (uint64), not the
// reference host wrapper's inconsistent remote_handle close-pointer typedef.
type Close = unsafe extern "C" fn(u64) -> i32;
type Alloc = unsafe extern "C" fn(i32, u32, i32) -> *mut c_void;
type Free = unsafe extern "C" fn(*mut c_void);
type ToFd = unsafe extern "C" fn(*mut c_void) -> i32;
type Map = unsafe extern "C" fn(i32, i32, *mut c_void, i32, usize, i32) -> i32;
type Unmap = unsafe extern "C" fn(i32, i32, *mut c_void, usize) -> i32;

struct Functions {
    open: Open,
    close: Close,
    alloc: Alloc,
    free: Free,
    to_fd: ToFd,
    map: Map,
    unmap: Unmap,
}

fn symbol(driver: &Driver, name: &CStr) -> Result<*mut c_void, Error> {
    // SAFETY: the retained driver owns the live DLL and fixed terminated name.
    let pointer = unsafe { GetProcAddress(driver.library.as_ptr(), name.as_ptr()) };
    if pointer.is_null() {
        Err(last_error("session-mapping-export"))
    } else {
        Ok(pointer)
    }
}

fn functions(driver: &Driver) -> Result<Functions, Error> {
    let open = symbol(driver, c"remote_handle64_open")?;
    let close = symbol(driver, c"remote_handle64_close")?;
    let alloc = symbol(driver, c"rpcmem_alloc")?;
    let free = symbol(driver, c"rpcmem_free")?;
    let to_fd = symbol(driver, c"rpcmem_to_fd")?;
    let map = symbol(driver, c"fastrpc_mmap")?;
    let unmap = symbol(driver, c"fastrpc_munmap")?;
    // SAFETY: the pinned public remote.h declares const char*, uint64* -> int.
    let open = unsafe { std::mem::transmute::<*mut c_void, Open>(open) };
    // SAFETY: the public vendor header declares uint64 handle -> int.
    let close = unsafe { std::mem::transmute::<*mut c_void, Close>(close) };
    // SAFETY: the pinned rpcmem.h declares int,uint32,int -> void*.
    let alloc = unsafe { std::mem::transmute::<*mut c_void, Alloc>(alloc) };
    // SAFETY: only the pointer returned by this same allocator is freed.
    let free = unsafe { std::mem::transmute::<*mut c_void, Free>(free) };
    // SAFETY: rpcmem_to_fd accepts an allocation from the same retained library.
    let to_fd = unsafe { std::mem::transmute::<*mut c_void, ToFd>(to_fd) };
    // SAFETY: remote.h pins domain,fd,void*,offset,size_t,32-bit enum -> int.
    let map = unsafe { std::mem::transmute::<*mut c_void, Map>(map) };
    // SAFETY: remote.h pins domain,fd,void*,size_t -> int.
    let unmap = unsafe { std::mem::transmute::<*mut c_void, Unmap>(unmap) };
    Ok(Functions {
        open,
        close,
        alloc,
        free,
        to_fd,
        map,
        unmap,
    })
}

fn package_guards(package: &Path, architecture: u32) -> Result<Vec<File>, Error> {
    if !package.is_absolute()
        || package.components().any(|part| match part {
            Component::Prefix(prefix) => !matches!(prefix.kind(), Prefix::Disk(_)),
            Component::Normal(name) => {
                name.to_string_lossy().contains([':', ';'])
                    || name.to_string_lossy().chars().any(char::is_control)
            }
            Component::ParentDir | Component::CurDir => true,
            Component::RootDir => false,
        })
        || std::env::var_os("ADSP_LIBRARY_PATH").as_deref() != Some(package.as_os_str())
    {
        return Err(Error::InheritedOverride);
    }
    no_reparse(package)?;
    if !package.is_dir() {
        return Err(Error::InvalidDriverPath);
    }
    let module = match architecture {
        73 => "libggml-htp-v73.so",
        81 => "libggml-htp-v81.so",
        _ => return Err(Error::Contract),
    };
    [module, "libggml-htp.cat", "libggml-htp.inf"]
        .into_iter()
        .map(|name| {
            let path = package.join(name);
            no_reparse(&path)?;
            let file = OpenOptions::new()
                .read(true)
                .share_mode(1)
                .open(path)
                .map_err(Error::File)?;
            let metadata = file.metadata().map_err(Error::File)?;
            if !metadata.is_file() || metadata.len() == 0 || metadata.len() > 16 * 1024 * 1024 {
                return Err(Error::InvalidDriverPath);
            }
            Ok(file)
        })
        .collect()
}

#[allow(
    clippy::redundant_pub_crate,
    reason = "Only the safe parent can consume the private native driver"
)]
pub(super) fn probe(
    mut driver: Driver,
    package: &Path,
    mut progress: impl FnMut(&SessionMappingReport),
) -> Result<SessionMappingReport, Error> {
    let mut uri = [0xa5; 256];
    let metadata = session_native::prepare_with_uri(&mut driver, &mut uri)?;
    if !metadata.metadata_prepared() {
        return Ok(SessionMappingReport::new(metadata));
    }
    let architecture = metadata.architecture.ok_or(Error::Contract)?;
    let guards = package_guards(package, architecture)?;
    let calls = functions(&driver)?;
    mapping::admit_terminal()?;
    // All owners are pinned before the first remote open attempt. A callback
    // panic, native refusal, uncertain allocation, or failed close cannot unload
    // the driver or release the checked package files. Process supervision owns
    // the remaining lifetime; process death is not a qualified vendor teardown.
    let _owner = ManuallyDrop::new((driver, guards));
    Ok(exercise(metadata, &uri, &calls, &mut progress))
}

fn entered(
    report: &mut SessionMappingReport,
    stage: &'static str,
    progress: &mut impl FnMut(&SessionMappingReport),
) {
    report.stage = stage;
    progress(report);
}

fn close_module(
    report: &mut SessionMappingReport,
    handle: u64,
    calls: &Functions,
    progress: &mut impl FnMut(&SessionMappingReport),
) {
    entered(report, "module-close-entered", progress);
    // SAFETY: the handle came from the successful fixed open, no compute was
    // submitted, and every scratch mapping was explicitly removed before close.
    let result = unsafe { (calls.close)(handle) };
    report.module_close_result = Some(result);
    report.module_close_completed = result == 0;
}

fn exercise(
    metadata: SessionPreparationReport,
    uri: &[u8; 256],
    calls: &Functions,
    progress: &mut impl FnMut(&SessionMappingReport),
) -> SessionMappingReport {
    let mut report = SessionMappingReport::new(metadata);
    let Some(domain) = metadata.effective_domain else {
        return report;
    };
    if !metadata.metadata_prepared()
        || !metadata
            .architecture
            .is_some_and(|architecture| crate::session::checked_uri(architecture, uri))
    {
        return report;
    }
    report.owner_pinned_until_exit = true;
    let mut handle = u64::MAX;
    entered(&mut report, "module-open-entered", progress);
    // SAFETY: the exact driver-returned URI was validated as the fixed signed
    // family/session module. It is terminated and live; output storage is u64.
    let result = unsafe { (calls.open)(uri.as_ptr().cast(), &raw mut handle) };
    report.module_open_result = Some(result);
    report.handle_observed = handle != u64::MAX;
    if result != 0 || !report.handle_observed {
        report.stage = if result != 0 {
            "module-open-refused-pinned"
        } else {
            "module-handle-not-written-pinned"
        };
        return report;
    }
    report.remote_module_opened = true;
    entered(&mut report, "allocation-entered", progress);
    // SAFETY: one fixed cached system-heap allocation, with documented arguments.
    let buffer = unsafe { (calls.alloc)(25, 1, 4096) };
    if buffer.is_null() {
        close_module(&mut report, handle, calls, progress);
        report.stage = "allocation-refused";
        return report;
    }
    report.mapping.allocation_succeeded = true;
    entered(&mut report, "descriptor-entered", progress);
    // SAFETY: this is the allocation returned by the same retained driver.
    let fd = unsafe { (calls.to_fd)(buffer) };
    report.mapping.descriptor_observed = fd >= 0;
    report.mapping.descriptor_error = (fd < 0).then_some(fd);
    if fd < 0 || buffer.addr() % 4096 != 0 {
        // SAFETY: this scratch has not been mapped or submitted to any consumer.
        unsafe { (calls.free)(buffer) };
        report.mapping.free_completed = true;
        close_module(&mut report, handle, calls, progress);
        report.stage = if fd < 0 {
            "descriptor-refused"
        } else {
            "allocation-not-page-aligned"
        };
        return report;
    }
    // SAFETY: the exclusive allocation is 4096 bytes and no device has seen it.
    unsafe { buffer.cast::<u8>().write_bytes(0xa5, 4096) };
    // SAFETY: the same unregistered allocation is live and CPU accessible.
    let view = unsafe { std::slice::from_raw_parts(buffer.cast::<u8>(), 4096) };
    report.mapping.cpu_roundtrip_verified = view.iter().all(|byte| *byte == 0xa5);
    if !report.mapping.cpu_roundtrip_verified {
        // SAFETY: no mapping or asynchronous submission occurred; view is no longer used.
        unsafe { (calls.free)(buffer) };
        report.mapping.free_completed = true;
        close_module(&mut report, handle, calls, progress);
        report.stage = "cpu-roundtrip-failed";
        return report;
    }
    report.mapping.mapping_attempted = true;
    report.mapping.owner_pinned_until_exit = true;
    entered(&mut report, "fd-mapping-entered", progress);

    // SAFETY: domain came from validated default-session metadata; the module is
    // open in that session. This exact owned fd/address is page aligned, offset0,
    // length4096; FD=2 is public ABI. No compute or cache-coherence claim follows.
    let result = unsafe { (calls.map)(domain, fd, buffer, 0, 4096, MAP_FD) };
    report.mapping.mapping_result = Some(result);
    if result != 0 {
        report.mapping.poisoned = true;
        report.mapping.stage = "fd-mapping-failed-pinned";
        report.stage = "fd-mapping-failed-pinned";
        return report;
    }
    entered(&mut report, "fd-unmapping-entered", progress);
    // SAFETY: the successful FD mapping is removed with its identical tuple;
    // there has been no DSP invocation or asynchronous buffer consumption.
    let result = unsafe { (calls.unmap)(domain, fd, buffer, 4096) };
    report.mapping.unmapping_result = Some(result);
    if result != 0 {
        report.mapping.poisoned = true;
        report.mapping.stage = "fd-unmapping-failed-pinned";
        report.stage = "fd-unmapping-failed-pinned";
        return report;
    }
    entered(&mut report, "free-entered", progress);
    // SAFETY: explicit unmap completed and no other native consumer exists.
    unsafe { (calls.free)(buffer) };
    report.mapping.free_completed = true;
    report.mapping.stage = "mapped-unmapped-freed";
    close_module(&mut report, handle, calls, progress);
    report.stage = if report.module_close_completed {
        "session-fd-cycle-completed-owner-pinned"
    } else {
        "module-close-refused-pinned"
    };
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, MutexGuard};

    #[repr(align(4096))]
    struct Page([u8; 4096]);

    struct State {
        fail: &'static str,
        calls: Vec<&'static str>,
        page: Option<Box<Page>>,
    }

    static STATE: Mutex<State> = Mutex::new(State {
        fail: "",
        calls: Vec::new(),
        page: None,
    });
    static TEST_LEASE: Mutex<()> = Mutex::new(());

    fn state() -> MutexGuard<'static, State> {
        STATE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    unsafe extern "C" fn open(uri: *const c_char, handle: *mut u64) -> i32 {
        let mut state = state();
        state.calls.push("open");
        // SAFETY: exercise supplies its live fixed terminated 256-byte URI.
        let uri = unsafe { CStr::from_ptr(uri) };
        if uri.to_bytes() != b"file:///libggml-htp-v81.so?htp_iface_skel_handle_invoke&_modver=1.0&_dom=cdsp&_session=0" {
            return 14;
        }
        if state.fail == "open" {
            return 512;
        }
        if state.fail != "handle" {
            // SAFETY: exercise supplies one live writable u64 output slot.
            unsafe { handle.write(42) };
        }
        drop(state);
        0
    }

    unsafe extern "C" fn close(handle: u64) -> i32 {
        let mut state = state();
        state.calls.push("close");
        if handle != 42 {
            return 14;
        }
        if state.fail == "close" { 512 } else { 0 }
    }

    unsafe extern "C" fn alloc(heap: i32, flags: u32, bytes: i32) -> *mut c_void {
        let mut state = state();
        state.calls.push("alloc");
        if (heap, flags, bytes) != (25, 1, 4096) || state.fail == "alloc" {
            return std::ptr::null_mut();
        }
        let page = state.page.insert(Box::new(Page([0; 4096])));
        let pointer = page.0.as_mut_ptr().cast();
        drop(state);
        pointer
    }

    unsafe extern "C" fn free(_: *mut c_void) {
        let mut state = state();
        state.calls.push("free");
        state.page = None;
    }

    unsafe extern "C" fn to_fd(_: *mut c_void) -> i32 {
        let mut state = state();
        state.calls.push("fd");
        if state.fail == "fd" { -4 } else { 7 }
    }

    unsafe extern "C" fn map(
        domain: i32,
        fd: i32,
        _: *mut c_void,
        offset: i32,
        bytes: usize,
        flags: i32,
    ) -> i32 {
        let mut state = state();
        state.calls.push("map");
        if (domain, fd, offset, bytes, flags) != (3, 7, 0, 4096, 2) {
            return 14;
        }
        if state.fail == "map" { 512 } else { 0 }
    }

    unsafe extern "C" fn unmap(domain: i32, fd: i32, _: *mut c_void, bytes: usize) -> i32 {
        let mut state = state();
        state.calls.push("unmap");
        if (domain, fd, bytes) != (3, 7, 4096) {
            return 14;
        }
        if state.fail == "unmap" { 512 } else { 0 }
    }

    #[test]
    fn fixed_session_fd_cycle_and_uncertain_lifetimes() {
        let _lease = TEST_LEASE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let calls = Functions {
            open,
            close,
            alloc,
            free,
            to_fd,
            map,
            unmap,
        };
        let mut metadata = SessionPreparationReport::new(0x81);
        metadata.domain_result = Some(0);
        metadata.effective_domain = Some(3);
        metadata.uri_result = Some(0);
        metadata.uri_verified = true;
        let mut uri = [0; 256];
        let bytes = b"file:///libggml-htp-v81.so?htp_iface_skel_handle_invoke&_modver=1.0&_dom=cdsp&_session=0\0";
        uri[..bytes.len()].copy_from_slice(bytes);
        for fail in ["", "open", "handle", "alloc", "fd", "map", "unmap", "close"] {
            *state() = State {
                fail,
                calls: Vec::new(),
                page: None,
            };
            let mut stages = Vec::new();
            let report = exercise(metadata, &uri, &calls, &mut |snapshot| {
                stages.push(snapshot.stage);
            });
            assert_eq!(report.completed_cycle(), fail.is_empty());
            assert_eq!(stages.first(), Some(&"module-open-entered"));
            assert!(report.owner_pinned_until_exit);
            let state = state();
            match fail {
                "open" | "handle" => assert_eq!(state.calls, ["open"]),
                "alloc" => assert_eq!(state.calls, ["open", "alloc", "close"]),
                "fd" => assert_eq!(state.calls, ["open", "alloc", "fd", "free", "close"]),
                "map" => {
                    assert_eq!(state.calls, ["open", "alloc", "fd", "map"]);
                    assert!(report.mapping.poisoned);
                    assert!(state.page.is_some());
                    assert_eq!(report.module_close_result, None);
                }
                "unmap" => {
                    assert_eq!(state.calls, ["open", "alloc", "fd", "map", "unmap"]);
                    assert!(report.mapping.poisoned);
                    assert!(state.page.is_some());
                    assert_eq!(report.module_close_result, None);
                }
                _ => assert_eq!(
                    state.calls,
                    ["open", "alloc", "fd", "map", "unmap", "free", "close"]
                ),
            }
        }
        *state() = State {
            fail: "",
            calls: Vec::new(),
            page: None,
        };
        metadata.uri_verified = false;
        assert!(!exercise(metadata, &uri, &calls, &mut |_| {}).remote_module_opened);
        let no_calls: [&str; 0] = [];
        assert_eq!(state().calls, no_calls);
    }
}
