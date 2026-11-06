//! Consuming fixed X2 matrix queue controls; no inference-library dependency.
//!
//! The supervisor must qualify the independently compiled Windows ARM64 queue
//! layouts/prototypes and public seven-argument start frame before execution.
//! No failed/expired submission frees or reuses its allocation. Only a complete
//! correlated numeric reply followed by stop/close/unmap permits normal free.

use super::{Driver, GetProcAddress, last_error, mapping, session_mapping_native, session_native};
use crate::{Capability, Error, QueueComputeReport, SessionPreparationReport, compute};
use std::ffi::{CStr, c_char, c_void};
use std::mem::ManuallyDrop;
use std::path::Path;
use std::ptr::{from_mut, null_mut};
use std::time::{Duration, Instant};

#[path = "projection_native.rs"]
mod projection_native;

pub(super) fn projection(
    driver: Driver,
    package: &Path,
    limits: crate::Q2ProjectionLimits,
    progress: impl FnMut(&crate::Q2ProjectionReport),
    work: impl FnOnce(&mut dyn crate::GuardedQ2Submitter) -> Result<(), Error>,
) -> Result<crate::Q2ProjectionReport, Error> {
    projection_native::run(driver, package, limits, progress, work)
}

pub(super) fn hmx_projection(
    driver: Driver,
    package: &Path,
    limits: crate::HmxQ2ProjectionLimits,
    progress: impl FnMut(&crate::HmxQ2ProjectionReport),
    work: impl FnOnce(&mut dyn crate::HmxQ2Submitter) -> Result<(), Error>,
) -> Result<crate::HmxQ2ProjectionReport, Error> {
    projection_native::run_hmx(driver, package, limits, progress, work)
}

#[allow(
    clippy::redundant_pub_crate,
    reason = "Safe crate root wraps the native session handle."
)]
pub(crate) use projection_native::NativeQ2Session;

pub(super) fn native_q2_projection(
    driver: Driver,
    package: &Path,
    limits: crate::NativeQ2ProjectionLimits,
    progress: impl FnMut(&crate::NativeQ2ProjectionReport),
    work: impl FnOnce(&mut dyn crate::NativeQ2Submitter) -> Result<(), Error>,
) -> Result<crate::NativeQ2ProjectionReport, Error> {
    projection_native::run_native_q2(driver, package, limits, progress, work)
}

pub(super) fn open_native_q2_session(
    driver: Driver,
    package: &Path,
    limits: crate::NativeQ2ProjectionLimits,
) -> Result<NativeQ2Session, Error> {
    projection_native::open_native_q2(driver, package, limits)
}

type Open = unsafe extern "C" fn(*const c_char, *mut u64) -> i32;
type Close = unsafe extern "C" fn(u64) -> i32;
type Invoke = unsafe extern "C" fn(u64, u32, *mut RemoteBuffer) -> i32;
type Alloc = unsafe extern "C" fn(i32, u32, i32) -> *mut c_void;
type Free = unsafe extern "C" fn(*mut c_void);
type ToFd = unsafe extern "C" fn(*mut c_void) -> i32;
type Map = unsafe extern "C" fn(i32, i32, *mut c_void, i32, usize, i32) -> i32;
type Unmap = unsafe extern "C" fn(i32, i32, *mut c_void, usize) -> i32;
type Callback = Option<unsafe extern "C" fn(*mut c_void, i32, *mut c_void)>;
type QueueCreate = unsafe extern "C" fn(
    i32,
    u32,
    u32,
    u32,
    Callback,
    Callback,
    *mut c_void,
    *mut *mut c_void,
) -> i32;
type QueueExport = unsafe extern "C" fn(*mut c_void, *mut u64) -> i32;
type QueueClose = unsafe extern "C" fn(*mut c_void) -> i32;
type QueueWrite =
    unsafe extern "C" fn(*mut c_void, u32, u32, *mut QueueBuffer, u32, *const u8, u32) -> i32;
type QueueWriteNoBlock =
    unsafe extern "C" fn(*mut c_void, u32, u32, *mut QueueBuffer, u32, *const u8) -> i32;
type QueueRead = unsafe extern "C" fn(
    *mut c_void,
    *mut u32,
    u32,
    *mut u32,
    *mut QueueBuffer,
    u32,
    *mut u32,
    *mut u8,
) -> i32;

#[repr(C)]
struct RemoteBuffer {
    pointer: *mut c_void,
    bytes: usize,
}

// The only argument is the remote_buf member of remote_arg. On Windows ARM64
// the full public remote_arg union has this same size/alignment; no handle or
// DMA member is used. Independent C controls must check this before invocation.
const _: () = assert!(size_of::<RemoteBuffer>() == 16);
const _: () = assert!(align_of::<RemoteBuffer>() == 8);
const _: () = assert!(std::mem::offset_of!(RemoteBuffer, bytes) == 8);

#[repr(C)]
#[derive(Clone, Copy)]
struct QueueBuffer {
    fd: u32,
    size: u32,
    offset: u32,
    flags: u32,
    pointer: *mut c_void,
}
const _: () = assert!(size_of::<QueueBuffer>() == 24);
const _: () = assert!(align_of::<QueueBuffer>() == 8);
const _: () = assert!(std::mem::offset_of!(QueueBuffer, fd) == 0);
const _: () = assert!(std::mem::offset_of!(QueueBuffer, size) == 4);
const _: () = assert!(std::mem::offset_of!(QueueBuffer, offset) == 8);
const _: () = assert!(std::mem::offset_of!(QueueBuffer, flags) == 12);
const _: () = assert!(std::mem::offset_of!(QueueBuffer, pointer) == 16);

#[repr(C)]
struct ReferenceProcessRequest {
    domain: i32,
    enable: i32,
}
const _: () = assert!(size_of::<ReferenceProcessRequest>() == 8);
const _: () = assert!(align_of::<ReferenceProcessRequest>() == 4);

enum WriteCall {
    Timed(QueueWrite),
    Nonblocking(QueueWriteNoBlock),
}

struct Calls {
    open: Open,
    close: Close,
    invoke: Invoke,
    alloc: Alloc,
    free: Free,
    to_fd: ToFd,
    map: Map,
    unmap: Unmap,
    queue_create: QueueCreate,
    queue_export: QueueExport,
    queue_close: QueueClose,
    queue_write: WriteCall,
    queue_read: QueueRead,
}

struct CleanupCalls {
    invoke: Invoke,
    queue_close: QueueClose,
    unmap: Unmap,
    close: Close,
    free: Free,
}

impl Calls {
    const fn cleanup_calls(&self) -> CleanupCalls {
        CleanupCalls {
            invoke: self.invoke,
            queue_close: self.queue_close,
            unmap: self.unmap,
            close: self.close,
            free: self.free,
        }
    }
}

fn symbol(driver: &Driver, name: &CStr) -> Result<*mut c_void, Error> {
    // SAFETY: the driver retains its live module and the fixed export is terminated.
    let pointer = unsafe { GetProcAddress(driver.library.as_ptr(), name.as_ptr()) };
    if pointer.is_null() {
        Err(last_error("compute-export"))
    } else {
        Ok(pointer)
    }
}

fn calls(driver: &Driver, fixture: compute::ComputeFixture) -> Result<Calls, Error> {
    let open = symbol(driver, c"remote_handle64_open")?;
    let close = symbol(driver, c"remote_handle64_close")?;
    let invoke = symbol(driver, c"remote_handle64_invoke")?;
    let alloc = symbol(driver, c"rpcmem_alloc")?;
    let free = symbol(driver, c"rpcmem_free")?;
    let to_fd = symbol(driver, c"rpcmem_to_fd")?;
    let map = symbol(driver, c"fastrpc_mmap")?;
    let unmap = symbol(driver, c"fastrpc_munmap")?;
    let create = symbol(driver, c"dspqueue_create")?;
    let export = symbol(driver, c"dspqueue_export")?;
    let qclose = symbol(driver, c"dspqueue_close")?;
    let queue_write = if matches!(
        fixture,
        compute::ComputeFixture::F32NoBlock | compute::ComputeFixture::GuardedQ2NoBlock
    ) {
        let write = symbol(driver, c"dspqueue_write_noblock")?;
        // SAFETY: the pinned public header and independent C prototype mock
        // specify exactly six arguments and no timeout for this separate API.
        WriteCall::Nonblocking(unsafe {
            std::mem::transmute::<*mut c_void, QueueWriteNoBlock>(write)
        })
    } else {
        let write = symbol(driver, c"dspqueue_write")?;
        // SAFETY: the independently checked timed-write API has seven arguments.
        WriteCall::Timed(unsafe { std::mem::transmute::<*mut c_void, QueueWrite>(write) })
    };
    let read = symbol(driver, c"dspqueue_read_noblock")?;
    // SAFETY: public remote.h pins const char*, uint64* -> int.
    let open = unsafe { std::mem::transmute::<*mut c_void, Open>(open) };
    // SAFETY: authoritative remote_handle64_close takes uint64, not uint32.
    let close = unsafe { std::mem::transmute::<*mut c_void, Close>(close) };
    // SAFETY: only remote_arg.buf is used with independently checked explicit
    // pointer/size_t/u64 widths. Windows ARM64 is LLP64, not LP64.
    let invoke = unsafe { std::mem::transmute::<*mut c_void, Invoke>(invoke) };
    // SAFETY: public rpcmem_alloc takes int, uint32, int and returns void*.
    let alloc = unsafe { std::mem::transmute::<*mut c_void, Alloc>(alloc) };
    // SAFETY: public rpcmem_free accepts its own live allocation.
    let free = unsafe { std::mem::transmute::<*mut c_void, Free>(free) };
    // SAFETY: public rpcmem_to_fd accepts its own live allocation.
    let to_fd = unsafe { std::mem::transmute::<*mut c_void, ToFd>(to_fd) };
    // SAFETY: pinned fastrpc_mmap takes int,int,void*,int,size_t,int.
    let map = unsafe { std::mem::transmute::<*mut c_void, Map>(map) };
    // SAFETY: pinned fastrpc_munmap takes int,int,void*,size_t.
    let unmap = unsafe { std::mem::transmute::<*mut c_void, Unmap>(unmap) };
    // SAFETY: pinned public dspqueue.h and independent Windows prototype control
    // specify these eight arguments; callbacks/context are always null here.
    let queue_create = unsafe { std::mem::transmute::<*mut c_void, QueueCreate>(create) };
    // SAFETY: public dspqueue_export takes queue pointer and writable uint64*.
    let queue_export = unsafe { std::mem::transmute::<*mut c_void, QueueExport>(export) };
    // SAFETY: public dspqueue_close takes one queue pointer.
    let queue_close = unsafe { std::mem::transmute::<*mut c_void, QueueClose>(qclose) };
    // SAFETY: the no-block read has eight arguments and no timeout argument.
    let queue_read = unsafe { std::mem::transmute::<*mut c_void, QueueRead>(read) };
    Ok(Calls {
        open,
        close,
        invoke,
        alloc,
        free,
        to_fd,
        map,
        unmap,
        queue_create,
        queue_export,
        queue_close,
        queue_write,
        queue_read,
    })
}

fn entered(
    report: &mut QueueComputeReport,
    stage: &'static str,
    progress: &mut impl FnMut(&QueueComputeReport),
) {
    report.stage = stage;
    progress(report);
}

#[allow(
    clippy::redundant_pub_crate,
    reason = "Only the safe parent consumes the native driver"
)]
pub(super) fn probe(
    driver: Driver,
    package: &Path,
    progress: impl FnMut(&QueueComputeReport),
) -> Result<QueueComputeReport, Error> {
    probe_fixture(driver, package, progress, compute::ComputeFixture::F32)
}

#[allow(
    clippy::redundant_pub_crate,
    reason = "Only the safe parent selects one of four private fixed compute controls"
)]
pub(super) fn probe_fixture(
    driver: Driver,
    package: &Path,
    mut progress: impl FnMut(&QueueComputeReport),
    fixture: compute::ComputeFixture,
) -> Result<QueueComputeReport, Error> {
    let (report, opened) = open_module(driver, package, &mut progress, fixture)?;
    let Some((handle, domain, calls)) = opened else {
        return Ok(report);
    };
    Ok(exercise(report, handle, domain, &calls, &mut progress))
}

type OpenedModule = (u64, i32, Calls);

fn open_module(
    mut driver: Driver,
    package: &Path,
    progress: &mut impl FnMut(&QueueComputeReport),
    fixture: compute::ComputeFixture,
) -> Result<(QueueComputeReport, Option<OpenedModule>), Error> {
    crate::self_check()?;
    let raw = driver.capability(Capability::Architecture)?;
    let metadata = SessionPreparationReport::new(raw);
    if metadata.architecture != Some(81) {
        return Ok((QueueComputeReport::new_fixture(metadata, fixture), None));
    }
    let guards = session_mapping_native::package_guards(package, 81)?;
    let calls = calls(&driver, fixture)?;
    let control = session_native::session_control(&driver)?;
    let metadata = session_native::prepare_effective_domain(raw, control);
    let mut report = QueueComputeReport::new_fixture(metadata, fixture);
    let Some(domain) = metadata.effective_domain else {
        return Ok((report, None));
    };
    if metadata.domain_result != Some(0) {
        return Ok((report, None));
    }
    mapping::admit_terminal()?;
    // Pin the driver/lease/package files before process initialization. All later
    // errors, callback unwinding or native stalls keep these owners live until
    // supervised process death; no native buffer owner implements implicit free.
    let _owner = ManuallyDrop::new((driver, guards));
    report.owner_pinned_until_exit = true;
    report.poisoned = true;
    report.unsigned_process_requested = true;
    entered(
        &mut report,
        "compute-reference-process-init-entered",
        progress,
    );
    let mut request = ReferenceProcessRequest { domain, enable: 1 };
    // SAFETY: exact existing fixed two-int reference initializer, before module
    // open. It makes no OS certificate, boot, test-signing or trust-store change.
    let result = unsafe { control(2, from_mut(&mut request).cast(), 8) };
    report.process_initialization_result = Some(result);
    report.process_initialization_domain_after = Some(request.domain);
    report.process_initialization_enable_after = Some(request.enable);
    if result != 0 {
        report.stage = "compute-reference-process-init-refused-pinned";
        return Ok((report, None));
    }
    let mut uri = [0xa5; 256];
    report.metadata = session_native::prepare_fixed_uri(metadata, control, &mut uri)?;
    if !report.metadata.metadata_prepared() {
        return Ok((report, None));
    }
    let mut handle = u64::MAX;
    entered(&mut report, "compute-module-open-entered", progress);
    // SAFETY: URI is the exact validated, terminated v81/default-CDSP module URI.
    let result = unsafe { (calls.open)(uri.as_ptr().cast(), &raw mut handle) };
    report.module_open_result = Some(result);
    report.handle_observed = handle != u64::MAX;
    if result != 0 || !report.handle_observed {
        report.stage = "compute-module-open-refused-pinned";
        return Ok((report, None));
    }
    Ok((report, Some((handle, domain, calls))))
}

#[allow(
    clippy::too_many_lines,
    reason = "Keep one fixed operation's registration, submission and completion sequence linear for ownership review"
)]
fn exercise(
    mut report: QueueComputeReport,
    handle: u64,
    domain: i32,
    calls: &Calls,
    progress: &mut impl FnMut(&QueueComputeReport),
) -> QueueComputeReport {
    entered(&mut report, "compute-allocation-entered", progress);
    // SAFETY: one fixed cached system-heap 16KiB allocation.
    let memory = unsafe { (calls.alloc)(25, 1, 16_384) };
    if memory.is_null() {
        report.stage = "compute-allocation-refused-pinned";
        return report;
    }
    // SAFETY: only the allocation from this live rpcmem allocator is passed back.
    let fd = unsafe { (calls.to_fd)(memory) };
    let Ok(fd_u32) = u32::try_from(fd) else {
        report.stage = "compute-fd-refused-pinned";
        return report;
    };
    if memory.addr() % 4096 != 0 {
        report.stage = "compute-allocation-not-aligned-pinned";
        return report;
    }
    let Ok(base) = u64::try_from(memory.addr()) else {
        report.stage = "compute-address-out-of-range-pinned";
        return report;
    };
    report.allocation_succeeded = true;
    let image = compute::fixture_image_for(report.fixture, base, fd_u32);
    // SAFETY: rpcmem returned an exclusive CPU-accessible extent of exactly the
    // requested length. No registration or device consumer exists yet.
    unsafe {
        std::ptr::copy_nonoverlapping(image.as_ptr(), memory.cast::<u8>(), image.len());
    }
    entered(&mut report, "compute-fd-map-entered", progress);
    // SAFETY: same validated effective domain and exact aligned allocation/FD;
    // public MAP_FD=2 is the previously successful terminal session map mode.
    let result = unsafe { (calls.map)(domain, fd, memory, 0, 16_384, 2) };
    report.mapping_result = Some(result);
    if result != 0 {
        report.stage = "compute-fd-map-refused-pinned";
        return report;
    }
    let mut queue = null_mut();
    entered(&mut report, "compute-queue-create-entered", progress);
    // SAFETY: exact public signature, finite 4KiB queues, no callback threads or
    // callback context. The writable queue handle is poisoned as null first.
    let result = unsafe {
        (calls.queue_create)(
            domain,
            0,
            4096,
            4096,
            None,
            None,
            null_mut(),
            &raw mut queue,
        )
    };
    report.queue_create_result = Some(result);
    if result != 0 || queue.is_null() {
        report.stage = "compute-queue-create-refused-pinned";
        return report;
    }
    let mut queue_id = u64::MAX;
    entered(&mut report, "compute-queue-export-entered", progress);
    // SAFETY: queue came from this live driver; output is a writable uint64.
    let result = unsafe { (calls.queue_export)(queue, &raw mut queue_id) };
    report.queue_export_result = Some(result);
    if result != 0 || queue_id == u64::MAX {
        report.stage = "compute-queue-export-refused-pinned";
        return report;
    }
    let mut start = compute::start_wire(queue_id);
    let mut argument = RemoteBuffer {
        pointer: start.as_mut_ptr().cast(),
        bytes: start.len(),
    };
    entered(&mut report, "compute-queue-start-entered", progress);
    // SAFETY: independently captured public seven-argument start frame, exactly
    // one 48byte remote_arg.buf, method2. The call is synchronous; stack buffers
    // remain live until it returns. No arbitrary fields or operation are exposed.
    let result = unsafe { (calls.invoke)(handle, compute::START_SCALARS, &raw mut argument) };
    report.queue_start_result = Some(result);
    if result != 0 {
        report.stage = "compute-queue-start-refused-pinned";
        return report;
    }
    let request = compute::request_wire();
    // Include the entire allocation in the cache-maintained queue descriptor,
    // including output poison/data. Limiting this to command bytes would leave
    // dirty/stale host output lines outside the recipient invalidation extent.
    let expected = QueueBuffer {
        fd: fd_u32,
        size: 16_384,
        offset: 0,
        flags: 0x90,
        pointer: memory,
    };
    let mut descriptor = expected;
    report.compute_submitted = true;
    entered(&mut report, "compute-queue-write-entered", progress);
    // Both selected APIs copy the exact same request/descriptor. There is one
    // attempt only; any nonzero result pins all uncertain owners without retry.
    let result = write_packet(&calls.queue_write, queue, &mut descriptor, &request);
    report.queue_write_result = Some(result);
    // A write may return a signaling error after advancing the packet queue.
    // Observe at most one reply under the existing absolute deadline; never resubmit or
    // change the write API. Any write error still forbids all normal cleanup.
    let read_stage = if result == 0 {
        "compute-queue-read-entered"
    } else {
        "compute-write-error-drain-entered"
    };
    entered(&mut report, read_stage, progress);
    if !read_reply(&mut report, calls, queue, expected) {
        return report;
    }
    // SAFETY: response validated exact same FD/extent/pointer and cache flags;
    // the completed operation globally flushed its writes before replying and
    // the queue invalidated the complete host extent. No borrowed slice existed
    // during asynchronous device use. Only the fixed output subsection is read.
    let output = unsafe {
        std::slice::from_raw_parts(
            memory
                .cast::<u8>()
                .wrapping_add(report.fixture.output_offset()),
            compute::OUTPUT_BYTES,
        )
    };
    if !report.record_numeric_reply(compute::output_valid_for(report.fixture, output)) {
        return report;
    }
    cleanup(
        report,
        &calls.cleanup_calls(),
        handle,
        queue,
        domain,
        fd,
        memory,
        progress,
    )
}

fn write_packet(
    call: &WriteCall,
    queue: *mut c_void,
    descriptor: &mut QueueBuffer,
    request: &[u8; compute::COMPUTE_REQUEST_BYTES],
) -> i32 {
    match *call {
        WriteCall::Timed(write) => {
            // SAFETY: one fully prepared mapped packet and exact24byte request;
            // the existing timed modes retain their fixed100ms timeout.
            unsafe { write(queue, 0, 1, descriptor, 24, request.as_ptr(), 100_000) }
        }
        WriteCall::Nonblocking(write) => {
            // SAFETY: same packet/flags/FD/cache extent with the separately
            // checked six-argument nonblocking export. No timeout or retry.
            unsafe { write(queue, 0, 1, descriptor, 24, request.as_ptr()) }
        }
    }
}

fn read_reply(
    report: &mut QueueComputeReport,
    calls: &Calls,
    queue: *mut c_void,
    expected: QueueBuffer,
) -> bool {
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        if Instant::now() >= deadline {
            report.stage = "compute-queue-read-expired-pinned";
            return false;
        }
        let mut flags = u32::MAX;
        let mut count = u32::MAX;
        let mut size = u32::MAX;
        let mut descriptor = QueueBuffer {
            fd: u32::MAX,
            size: u32::MAX,
            offset: u32::MAX,
            flags: u32::MAX,
            pointer: null_mut(),
        };
        let mut reply = [0xa5; compute::COMPUTE_RESPONSE_BYTES];
        // SAFETY: independent Windows ABI validates the exact no-block prototype.
        // Every writable output has its full declared capacity. Callback-free
        // single-slot queue permits reads; no returned extent is trusted yet.
        let result = unsafe {
            (calls.queue_read)(
                queue,
                &raw mut flags,
                1,
                &raw mut count,
                &raw mut descriptor,
                88,
                &raw mut size,
                reply.as_mut_ptr(),
            )
        };
        report.queue_read_result = Some(result);
        if result == 516 {
            // Pinned public AEE_EWOULDBLOCK, host return value.
            std::thread::sleep(Duration::from_millis(1));
            continue;
        }
        if result != 0 {
            report.stage = "compute-queue-read-refused-pinned";
            return false;
        }
        report.response_validated = count == 1
            && size == 88
            && flags & 3 == 3
            && flags & !0x1f == 0
            && descriptor.fd == expected.fd
            && descriptor.size == expected.size
            && descriptor.offset == expected.offset
            && descriptor.pointer == expected.pointer
            && descriptor.flags == expected.flags
            && compute::response_valid(&reply);
        if !report.response_validated {
            report.stage = "compute-response-contract-failed-pinned";
        }
        return report.response_validated;
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "Cleanup consumes one fixed exact domain/FD/address tuple and its live owners"
)]
fn cleanup(
    report: QueueComputeReport,
    calls: &CleanupCalls,
    handle: u64,
    queue: *mut c_void,
    domain: i32,
    fd: i32,
    memory: *mut c_void,
    progress: &mut impl FnMut(&QueueComputeReport),
) -> QueueComputeReport {
    cleanup_extent(
        report, calls, handle, queue, domain, fd, memory, 16_384, progress,
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "Explicit cleanup retains one exact mapped allocation tuple and its native owners"
)]
fn cleanup_extent(
    mut report: QueueComputeReport,
    calls: &CleanupCalls,
    handle: u64,
    queue: *mut c_void,
    domain: i32,
    fd: i32,
    memory: *mut c_void,
    allocation_bytes: usize,
    progress: &mut impl FnMut(&QueueComputeReport),
) -> QueueComputeReport {
    entered(&mut report, "compute-queue-stop-entered", progress);
    // SAFETY: method3 has no buffers/handles and null args, as in the independent
    // wire control. One fully validated numeric response preceded stop; the DSP
    // stop joins its processing thread and closes its imported queue on success.
    let result = unsafe { (calls.invoke)(handle, compute::STOP_SCALARS, null_mut()) };
    report.queue_stop_result = Some(result);
    if result != 0 {
        report.stage = "compute-queue-stop-refused-pinned";
        return report;
    }
    entered(&mut report, "compute-queue-close-entered", progress);
    // SAFETY: successful DSP stop closed its import; no work remains in flight.
    let result = unsafe { (calls.queue_close)(queue) };
    report.queue_close_result = Some(result);
    if result != 0 {
        report.stage = "compute-queue-close-refused-pinned";
        return report;
    }
    entered(&mut report, "compute-fd-unmap-entered", progress);
    // SAFETY: original same-domain FD/address/extent tuple; the operation and
    // imported queue were drained/stopped and the host queue closed. Keep the
    // remote module/session open through this release, as in the earlier FD
    // mapping control. The allocation remains retained through module close.
    let result = unsafe { (calls.unmap)(domain, fd, memory, allocation_bytes) };
    report.unmapping_result = Some(result);
    if result != 0 {
        report.stage = "compute-fd-unmap-refused-pinned";
        return report;
    }
    entered(&mut report, "compute-module-close-entered", progress);
    // SAFETY: after successful stop, host queue close and explicit host FD unmap,
    // close the exact original u64 handle. No queue or operation remains active;
    // the allocation is still retained if this final handle release is refused.
    let result = unsafe { (calls.close)(handle) };
    report.module_close_result = Some(result);
    if result != 0 {
        report.stage = "compute-module-close-refused-pinned";
        return report;
    }
    entered(&mut report, "compute-free-entered", progress);
    // SAFETY: successful stop, queue close, module close and explicit FD unmap
    // establish that this exact allocation has no remaining known consumer.
    unsafe {
        (calls.free)(memory);
    }
    report.free_completed = true;
    report.poisoned = false;
    report.stage = "compute-numeric-verified-stopped-unmapped-freed";
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU32, Ordering};

    static TIMED_CALLS: AtomicU32 = AtomicU32::new(0);
    static NONBLOCKING_CALLS: AtomicU32 = AtomicU32::new(0);

    type CleanupFailure = Option<(&'static str, i32)>;
    type CleanupCase = (CleanupFailure, &'static [&'static str]);

    struct CleanupState {
        steps: Vec<&'static str>,
        failure: CleanupFailure,
    }

    static CLEANUP_STATE: Mutex<CleanupState> = Mutex::new(CleanupState {
        steps: Vec::new(),
        failure: None,
    });

    fn cleanup_step(name: &'static str) -> i32 {
        let Ok(mut state) = CLEANUP_STATE.lock() else {
            return -77;
        };
        state.steps.push(name);
        match state.failure {
            Some((failure_point, code)) if failure_point == name => code,
            _ => 0,
        }
    }

    extern "C" fn cleanup_stop(handle: u64, scalars: u32, arguments: *mut RemoteBuffer) -> i32 {
        if handle != 42 || scalars != compute::STOP_SCALARS || !arguments.is_null() {
            return -77;
        }
        cleanup_step("stop")
    }

    extern "C" fn cleanup_queue_close(queue: *mut c_void) -> i32 {
        if queue.addr() != 0x1000 {
            return -77;
        }
        cleanup_step("queue-close")
    }

    extern "C" fn cleanup_unmap(domain: i32, fd: i32, memory: *mut c_void, bytes: usize) -> i32 {
        if domain != 3 || fd != 7 || memory.addr() != 0x2000 || bytes != 16_384 {
            return -77;
        }
        cleanup_step("unmap")
    }

    extern "C" fn cleanup_module_close(handle: u64) -> i32 {
        if handle != 42 {
            return -77;
        }
        cleanup_step("module-close")
    }

    extern "C" fn cleanup_free(memory: *mut c_void) {
        let _ = cleanup_step(if memory.addr() == 0x2000 {
            "free"
        } else {
            "invalid-free"
        });
    }

    const fn numeric_report() -> QueueComputeReport {
        let mut metadata = SessionPreparationReport::new(0x81);
        metadata.domain_result = Some(0);
        metadata.effective_domain = Some(3);
        metadata.uri_result = Some(0);
        metadata.uri_verified = true;
        let mut report = QueueComputeReport::new(metadata);
        report.process_initialization_result = Some(0);
        report.process_initialization_domain_after = Some(3);
        report.process_initialization_enable_after = Some(1);
        report.unsigned_process_requested = true;
        report.module_open_result = Some(0);
        report.handle_observed = true;
        report.allocation_succeeded = true;
        report.mapping_result = Some(0);
        report.queue_create_result = Some(0);
        report.queue_export_result = Some(0);
        report.queue_start_result = Some(0);
        report.queue_write_result = Some(0);
        report.queue_read_result = Some(0);
        report.compute_submitted = true;
        report.response_validated = true;
        report.numeric_validated = true;
        report.owner_pinned_until_exit = true;
        report.poisoned = true;
        report
    }

    #[test]
    fn actual_cleanup_unmaps_before_module_close_and_retains_on_any_failure() -> Result<(), Error> {
        let calls = CleanupCalls {
            invoke: cleanup_stop,
            queue_close: cleanup_queue_close,
            unmap: cleanup_unmap,
            close: cleanup_module_close,
            free: cleanup_free,
        };
        let cases: [CleanupCase; 5] = [
            (
                None,
                &["stop", "queue-close", "unmap", "module-close", "free"],
            ),
            (Some(("stop", 14)), &["stop"]),
            (Some(("queue-close", 14)), &["stop", "queue-close"]),
            (Some(("unmap", 512)), &["stop", "queue-close", "unmap"]),
            (
                Some(("module-close", 14)),
                &["stop", "queue-close", "unmap", "module-close"],
            ),
        ];
        for (failure, expected) in cases {
            {
                let mut state = CLEANUP_STATE.lock().map_err(|_| Error::Contract)?;
                state.steps.clear();
                state.failure = failure;
            }
            // Opaque markers are compared/passed back by mocks, never dereferenced.
            let report = cleanup(
                numeric_report(),
                &calls,
                42,
                std::ptr::without_provenance_mut(0x1000),
                3,
                7,
                std::ptr::without_provenance_mut(0x2000),
                &mut |_| {},
            );
            let state = CLEANUP_STATE.lock().map_err(|_| Error::Contract)?;
            assert_eq!(state.steps.as_slice(), expected);
            assert!(report.compute_executed());
            assert_eq!(report.free_completed, failure.is_none());
            assert_eq!(report.poisoned, failure.is_some());
            assert_eq!(report.completed_cycle(), failure.is_none());
            assert!(report.owner_pinned_until_exit);
            if failure == Some(("unmap", 512)) {
                assert_eq!(report.unmapping_result, Some(512));
                assert_eq!(report.module_close_result, None);
                assert!(!report.free_completed);
            }
            if failure == Some(("module-close", 14)) {
                assert_eq!(report.unmapping_result, Some(0));
                assert_eq!(report.module_close_result, Some(14));
                assert!(!report.free_completed);
            }
        }
        Ok(())
    }

    extern "C" fn timed_refusal(
        queue: *mut c_void,
        flags: u32,
        count: u32,
        buffers: *mut QueueBuffer,
        bytes: u32,
        message: *const u8,
        timeout: u32,
    ) -> i32 {
        TIMED_CALLS.fetch_add(1, Ordering::Relaxed);
        if queue.is_null()
            && flags == 0
            && count == 1
            && !buffers.is_null()
            && bytes == 24
            && !message.is_null()
            && timeout == 100_000
        {
            14
        } else {
            -77
        }
    }

    extern "C" fn nonblocking_refusal(
        queue: *mut c_void,
        flags: u32,
        count: u32,
        buffers: *mut QueueBuffer,
        bytes: u32,
        message: *const u8,
    ) -> i32 {
        NONBLOCKING_CALLS.fetch_add(1, Ordering::Relaxed);
        if queue.is_null()
            && flags == 0
            && count == 1
            && !buffers.is_null()
            && bytes == 24
            && !message.is_null()
        {
            14
        } else {
            -77
        }
    }

    #[test]
    fn refused_nonblocking_write_never_retries_or_calls_timed_api() {
        TIMED_CALLS.store(0, Ordering::Relaxed);
        NONBLOCKING_CALLS.store(0, Ordering::Relaxed);
        let mut descriptor = QueueBuffer {
            fd: 7,
            size: 16_384,
            offset: 0,
            flags: 0x90,
            pointer: null_mut(),
        };
        let request = compute::request_wire();
        assert_eq!(
            write_packet(
                &WriteCall::Nonblocking(nonblocking_refusal),
                null_mut(),
                &mut descriptor,
                &request
            ),
            14
        );
        assert_eq!(NONBLOCKING_CALLS.load(Ordering::Relaxed), 1);
        assert_eq!(TIMED_CALLS.load(Ordering::Relaxed), 0);
        assert_eq!(
            write_packet(
                &WriteCall::Timed(timed_refusal),
                null_mut(),
                &mut descriptor,
                &request
            ),
            14
        );
        assert_eq!(NONBLOCKING_CALLS.load(Ordering::Relaxed), 1);
        assert_eq!(TIMED_CALLS.load(Ordering::Relaxed), 1);
    }
}
