//! One terminal scratch registration attempt; no DSP module or compute call.

use super::{Driver, GetProcAddress, last_error};
use crate::{Error, MappingProbeReport};
use std::ffi::{CStr, c_void};
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicBool, Ordering};

type Alloc = unsafe extern "C" fn(i32, u32, i32) -> *mut c_void;
type Free = unsafe extern "C" fn(*mut c_void);
type ToFd = unsafe extern "C" fn(*mut c_void) -> i32;
type Map = unsafe extern "C" fn(i32, i32, *mut c_void, i32, usize, i32) -> i32;
type Unmap = unsafe extern "C" fn(i32, i32, *mut c_void, usize) -> i32;

static USED: AtomicBool = AtomicBool::new(false);

struct MemoryFunctions {
    alloc: Alloc,
    free: Free,
    to_fd: ToFd,
    map: Map,
    unmap: Unmap,
}

#[allow(
    clippy::redundant_pub_crate,
    reason = "Private native helpers remain accessible only to their safe parent"
)]
#[must_use]
pub(super) fn terminal() -> bool {
    USED.load(Ordering::Acquire)
}

fn symbol(driver: &Driver, name: &CStr) -> Result<*mut c_void, Error> {
    // SAFETY: the live retained driver owns its loader handle and the fixed
    // CStr export names are NUL terminated. No function pointer escapes here.
    let pointer = unsafe { GetProcAddress(driver.library.as_ptr(), name.as_ptr()) };
    if pointer.is_null() {
        Err(last_error("mapping-export"))
    } else {
        Ok(pointer)
    }
}

fn memory_functions(driver: &Driver) -> Result<MemoryFunctions, Error> {
    let alloc = symbol(driver, c"rpcmem_alloc")?;
    let free = symbol(driver, c"rpcmem_free")?;
    let to_fd = symbol(driver, c"rpcmem_to_fd")?;
    let map = symbol(driver, c"fastrpc_mmap")?;
    let unmap = symbol(driver, c"fastrpc_munmap")?;
    // SAFETY: each fixed export has the exact C signature in the pinned public
    // Qualcomm headers and the matching signed-reference host-driver source.
    let alloc = unsafe { std::mem::transmute::<*mut c_void, Alloc>(alloc) };
    // SAFETY: rpcmem_free accepts a pointer from the same live rpcmem allocator.
    let free = unsafe { std::mem::transmute::<*mut c_void, Free>(free) };
    // SAFETY: rpcmem_to_fd accepts a pointer from the same live rpcmem allocator.
    let to_fd = unsafe { std::mem::transmute::<*mut c_void, ToFd>(to_fd) };
    // SAFETY: fastrpc_mmap's pinned C ABI includes size_t and a 32-bit enum.
    let map = unsafe { std::mem::transmute::<*mut c_void, Map>(map) };
    // SAFETY: fastrpc_munmap uses the exact domain/fd/address/length tuple.
    let unmap = unsafe { std::mem::transmute::<*mut c_void, Unmap>(unmap) };
    Ok(MemoryFunctions {
        alloc,
        free,
        to_fd,
        map,
        unmap,
    })
}

#[allow(
    clippy::redundant_pub_crate,
    reason = "Private native mapping owner must not become public API"
)]
pub(super) fn probe(driver: Driver) -> Result<MappingProbeReport, Error> {
    if USED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return Err(Error::Contract);
    }
    let MemoryFunctions {
        alloc,
        free,
        to_fd,
        map,
        unmap,
    } = memory_functions(&driver)?;

    // Once native allocation begins, only an explicitly known-safe cleanup
    // releases the owner. A map/unmap failure leaves its DLL/file handles and
    // allocation pinned until the supervising process is actually terminated.
    let owner = ManuallyDrop::new(driver);
    let mut report = MappingProbeReport::new();
    // SAFETY: one fixed 4KiB system-heap allocation with documented cached flags.
    let buffer = unsafe { alloc(25, 1, 4096) };
    if buffer.is_null() {
        drop(ManuallyDrop::into_inner(owner));
        report.stage = "allocation-refused";
        return Ok(report);
    }
    report.allocation_succeeded = true;
    // SAFETY: only the exact returned allocation is passed to its own allocator.
    let fd = unsafe { to_fd(buffer) };
    report.descriptor_observed = fd >= 0;
    report.descriptor_error = (fd < 0).then_some(fd);
    if fd < 0 || buffer.addr() % 4096 != 0 {
        // SAFETY: no mapping or submission was attempted; this is our allocation.
        unsafe {
            free(buffer);
        }
        report.free_completed = true;
        drop(ManuallyDrop::into_inner(owner));
        report.stage = if fd < 0 {
            "descriptor-refused"
        } else {
            "allocation-not-page-aligned"
        };
        return Ok(report);
    }
    // SAFETY: rpcmem_alloc returned a CPU-accessible exclusive 4096-byte extent;
    // no registration, kernel or device operation has yet seen this pointer.
    unsafe {
        buffer.cast::<u8>().write_bytes(0xa5, 4096);
    }
    // SAFETY: the same valid allocation remains owned, unregistered and live.
    let view = unsafe { std::slice::from_raw_parts(buffer.cast::<u8>(), 4096) };
    report.cpu_roundtrip_verified = view.iter().all(|byte| *byte == 0xa5);
    if !report.cpu_roundtrip_verified {
        // SAFETY: no mapping or submission occurred and no borrowed view is used again.
        unsafe {
            free(buffer);
        }
        report.free_completed = true;
        drop(ManuallyDrop::into_inner(owner));
        report.stage = "cpu-roundtrip-failed";
        return Ok(report);
    }
    report.mapping_attempted = true;
    report.owner_pinned_until_exit = true;
    // SAFETY: domain3, fd/address from the same owned allocation, aligned offset0
    // and length4096, and pinned FASTRPC_MAP_STATIC=0; no arbitrary flags exist.
    let result = unsafe { map(3, fd, buffer, 0, 4096, 0) };
    report.mapping_result = Some(result);
    if result != 0 {
        report.poisoned = true;
        report.stage = "mapping-failed-pinned";
        return Ok(report);
    }
    // SAFETY: the successful mapping is unregistered with the identical tuple;
    // there is no DSP compute submission or asynchronous buffer consumer.
    let result = unsafe { unmap(3, fd, buffer, 4096) };
    report.unmapping_result = Some(result);
    if result != 0 {
        report.poisoned = true;
        report.stage = "unmapping-failed-pinned";
        return Ok(report);
    }
    // SAFETY: explicit unmap completed and the scratch has no remaining native
    // consumer. The implicit domain-session DLL/file owner remains pinned.
    unsafe {
        free(buffer);
    }
    report.free_completed = true;
    report.stage = "mapped-unmapped-freed";
    Ok(report)
}
