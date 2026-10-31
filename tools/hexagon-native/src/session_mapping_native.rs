//! Fixed signed-module open followed by one established-session FD-map cycle.

use super::{Driver, GetProcAddress, last_error, mapping, no_reparse, session_native};
use crate::{Capability, Error, MAP_FD, SessionMappingReport, SessionPreparationReport};
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
    Ok(exercise(
        SessionMappingReport::new(metadata),
        &uri,
        &calls,
        &mut progress,
    ))
}

// Public remote.h session_control_req_id starts at zero: this is its third
// fixed entry; the matching two-int C structure is remote_rpc_control_unsigned_module.
const REFERENCE_PROCESS_REQUEST: u32 = 2;

#[repr(C)]
struct ReferenceProcessRequest {
    domain: i32,
    enable: i32,
}

const _: () = assert!(size_of::<ReferenceProcessRequest>() == 8);
const _: () = assert!(align_of::<ReferenceProcessRequest>() == 4);
const _: () = assert!(std::mem::offset_of!(ReferenceProcessRequest, enable) == 4);

#[allow(
    clippy::redundant_pub_crate,
    reason = "The safe API exposes a separately named fixed reference initialization only"
)]
pub(super) fn reference_probe(
    mut driver: Driver,
    package: &Path,
    mut progress: impl FnMut(&SessionMappingReport),
) -> Result<SessionMappingReport, Error> {
    let raw = driver.capability(Capability::Architecture)?;
    let initial = SessionPreparationReport::new(raw);
    let Some(architecture) = initial.architecture else {
        let mut report = SessionMappingReport::new(initial);
        report.reference_process_initialization_selected = true;
        return Ok(report);
    };
    let guards = package_guards(package, architecture)?;
    let calls = functions(&driver)?;
    let control = session_native::session_control(&driver)?;
    let metadata = session_native::prepare_effective_domain(raw, control);
    let mut report = SessionMappingReport::new(metadata);
    report.reference_process_initialization_selected = true;
    if metadata.effective_domain.is_none() || metadata.domain_result != Some(0) {
        return Ok(report);
    }
    mapping::admit_terminal()?;
    // Reference process-class selection is a native session-affecting operation.
    // Pin all owners before attempting it, even when no module is later opened.
    let _owner = ManuallyDrop::new((driver, guards));
    Ok(reference_after_domain(
        report,
        control,
        &calls,
        &mut progress,
    ))
}

fn reference_after_domain(
    mut report: SessionMappingReport,
    control: session_native::SessionControl,
    calls: &Functions,
    progress: &mut impl FnMut(&SessionMappingReport),
) -> SessionMappingReport {
    let Some(domain) = report.metadata.effective_domain else {
        return report;
    };
    if !report.reference_process_initialization_selected
        || report.metadata.architecture.is_none()
        || report.metadata.domain_result != Some(0)
        || domain < 0
        || report.metadata.uri_result.is_some()
    {
        return report;
    }
    report.owner_pinned_until_exit = true;
    entered(&mut report, "reference-process-init-entered", progress);
    let mut request = ReferenceProcessRequest { domain, enable: 1 };
    report.unsigned_process_requested = true;
    // SAFETY: the pinned eight-byte/two-int ABI selects the returned effective
    // default-CDSP domain, enable=1, before any module open. This is the exact
    // reference initializer, not an OS trust/certificate/boot-policy operation.
    let result = unsafe {
        control(
            REFERENCE_PROCESS_REQUEST,
            std::ptr::from_mut(&mut request).cast(),
            8,
        )
    };
    report.process_initialization_result = Some(result);
    report.process_initialization_domain_after = Some(request.domain);
    report.process_initialization_enable_after = Some(request.enable);
    if result != 0 {
        report.stage = "reference-process-init-refused-pinned";
        return report;
    }
    // The public ABI permits a writable payload and the reference caller does
    // not require input preservation. Record its raw returned bytes only; all
    // subsequent URI/mapping configuration uses original validated metadata.
    let mut uri = [0xa5; 256];
    let Ok(metadata) = session_native::prepare_fixed_uri(report.metadata, control, &mut uri) else {
        report.stage = "session-metadata-incomplete";
        return report;
    };
    report.metadata = metadata;
    if !report.metadata.metadata_prepared() {
        report.stage = "session-metadata-incomplete";
        return report;
    }
    exercise(report, &uri, calls, progress)
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
    mut report: SessionMappingReport,
    uri: &[u8; 256],
    calls: &Functions,
    progress: &mut impl FnMut(&SessionMappingReport),
) -> SessionMappingReport {
    let metadata = report.metadata;
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
            let report = exercise(
                SessionMappingReport::new(metadata),
                &uri,
                &calls,
                &mut |snapshot| {
                    stages.push(snapshot.stage);
                },
            );
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
        assert!(
            !exercise(
                SessionMappingReport::new(metadata),
                &uri,
                &calls,
                &mut |_| {}
            )
            .remote_module_opened
        );
        let no_calls: [&str; 0] = [];
        assert_eq!(state().calls, no_calls);
    }
    #[repr(C)]
    struct TestUri {
        domain: *mut c_char,
        domain_len: u32,
        session: u32,
        module: *mut c_char,
        module_len: u32,
        uri: *mut c_char,
        uri_len: u32,
    }

    unsafe extern "C" fn reference_control(request: u32, data: *mut c_void, bytes: u32) -> i32 {
        let mut state = state();
        if request == 2 && bytes == 8 {
            state.calls.push("init");
            let fail = state.fail;
            drop(state);
            // SAFETY: reference_after_domain passes this exact live two-int layout.
            let request = unsafe { &mut *data.cast::<ReferenceProcessRequest>() };
            if (request.domain, request.enable) != (3, 1) {
                return 14;
            }
            if fail == "init" {
                return 512;
            }
            if fail == "init-contract" {
                request.domain = -33;
                request.enable = 17;
            }
            0
        } else if request == 15 && bytes == 48 {
            state.calls.push("uri");
            let fail = state.fail;
            drop(state);
            if fail == "uri" {
                return 512;
            }
            // SAFETY: prepare_fixed_uri passes the exact tested 48-byte C request.
            let request = unsafe { &mut *data.cast::<TestUri>() };
            if request.domain_len != 4 || request.session != 0 {
                return 14;
            }
            // SAFETY: the fixed caller owns these exact terminated input arrays.
            let domain = unsafe { CStr::from_ptr(request.domain) };
            // SAFETY: the fixed caller owns its full terminated module URI array.
            let module = unsafe { CStr::from_ptr(request.module) };
            if domain.to_bytes() != b"cdsp"
                || module.to_bytes()
                    != b"file:///libggml-htp-v81.so?htp_iface_skel_handle_invoke&_modver=1.0"
                || usize::try_from(request.module_len).unwrap_or_default()
                    != module.to_bytes().len()
            {
                return 14;
            }
            let uri = b"file:///libggml-htp-v81.so?htp_iface_skel_handle_invoke&_modver=1.0&_dom=cdsp&_session=0\0";
            if request.uri_len != 256 {
                return 14;
            }
            // SAFETY: the fixed caller owns its 256-byte writable output array.
            unsafe {
                std::ptr::copy_nonoverlapping(uri.as_ptr(), request.uri.cast::<u8>(), uri.len());
            };
            0
        } else {
            14
        }
    }

    #[test]
    fn reference_initialization_precedes_uri_and_open_and_never_retries() {
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
        metadata.stage = "effective-domain-prepared";
        for fail in ["", "init", "init-contract", "uri", "open", "map"] {
            *state() = State {
                fail,
                calls: Vec::new(),
                page: None,
            };
            let mut initial = SessionMappingReport::new(metadata);
            initial.reference_process_initialization_selected = true;
            let mut stages = Vec::new();
            let report =
                reference_after_domain(initial, reference_control, &calls, &mut |snapshot| {
                    stages.push(snapshot.stage);
                });
            assert!(report.reference_process_initialization_selected);
            assert!(report.unsigned_process_requested);
            assert!(report.owner_pinned_until_exit);
            assert_eq!(
                report.completed_cycle(),
                fail.is_empty() || fail == "init-contract"
            );
            assert_eq!(stages.first(), Some(&"reference-process-init-entered"));
            if fail != "init-contract" {
                assert_eq!(report.process_initialization_domain_after, Some(3));
                assert_eq!(report.process_initialization_enable_after, Some(1));
            }
            let state = state();
            match fail {
                "init" => {
                    assert_eq!(state.calls, ["init"]);
                    assert_eq!(report.process_initialization_result, Some(512));
                    assert_eq!(report.stage, "reference-process-init-refused-pinned");
                }
                "init-contract" => {
                    assert_eq!(
                        state.calls,
                        [
                            "init", "uri", "open", "alloc", "fd", "map", "unmap", "free", "close"
                        ]
                    );
                    assert_eq!(report.process_initialization_result, Some(0));
                    assert_eq!(report.process_initialization_domain_after, Some(-33));
                    assert_eq!(report.process_initialization_enable_after, Some(17));
                    assert_eq!(report.metadata.effective_domain, Some(3));
                    assert_eq!(report.stage, "session-fd-cycle-completed-owner-pinned");
                }
                "uri" => {
                    assert_eq!(state.calls, ["init", "uri"]);
                    assert_eq!(report.metadata.uri_result, Some(512));
                    assert_eq!(report.module_open_result, None);
                }
                "open" => assert_eq!(state.calls, ["init", "uri", "open"]),
                "map" => {
                    assert_eq!(state.calls, ["init", "uri", "open", "alloc", "fd", "map"]);
                    assert!(state.page.is_some());
                }
                _ => assert_eq!(
                    state.calls,
                    [
                        "init", "uri", "open", "alloc", "fd", "map", "unmap", "free", "close"
                    ]
                ),
            }
        }
        *state() = State {
            fail: "",
            calls: Vec::new(),
            page: None,
        };
        let mut invalid = SessionMappingReport::new(metadata);
        invalid.reference_process_initialization_selected = true;
        invalid.metadata.domain_result = Some(512);
        let refused = reference_after_domain(invalid, reference_control, &calls, &mut |_| {});
        assert_eq!(refused.process_initialization_result, None);
        assert!(!refused.unsigned_process_requested);
        let no_calls: [&str; 0] = [];
        assert_eq!(state().calls, no_calls);
    }
}
