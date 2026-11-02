//! One bounded reusable queue for fixed-K/N guarded-Q2 projection operations.

use super::{
    Driver, QueueBuffer, QueueRead, RemoteBuffer, WriteCall, cleanup_extent, compute, null_mut,
    open_module, write_packet,
};
use crate::{
    Error, GuardedQ2Submitter, Q2ProjectionLimits, Q2ProjectionReport, SessionPreparationReport,
    projection,
};
use std::ffi::c_void;
use std::path::Path;
use std::time::{Duration, Instant};

fn entered(
    report: &mut Q2ProjectionReport,
    stage: &'static str,
    progress: &mut impl FnMut(&Q2ProjectionReport),
) {
    report.queue.stage = stage;
    progress(report);
}

#[allow(
    clippy::too_many_lines,
    reason = "Keep terminal queue ownership and ordered native setup explicit for review"
)]
pub(super) fn run(
    driver: Driver,
    package: &Path,
    limits: Q2ProjectionLimits,
    mut progress: impl FnMut(&Q2ProjectionReport),
    work: impl FnOnce(&mut dyn GuardedQ2Submitter) -> Result<(), Error>,
) -> Result<Q2ProjectionReport, Error> {
    let started = Instant::now();
    let deadline = started + limits.duration;
    let work_deadline = started
        + limits
            .duration
            .checked_sub(Duration::from_secs(3))
            .ok_or(Error::Contract)?;
    let mut report = Q2ProjectionReport::new(SessionPreparationReport::new(0));
    let (queue_report, opened) = open_module(
        driver,
        package,
        &mut |queue| {
            report.queue = *queue;
            progress(&report);
        },
        compute::ComputeFixture::GuardedQ2NoBlock,
    )?;
    report.queue = queue_report;
    let Some((handle, domain, calls)) = opened else {
        return Ok(report);
    };
    if Instant::now() >= work_deadline {
        report.deadline_expired = true;
        report.queue.stage = "projection-initialization-expired-pinned";
        return Ok(report);
    }
    let allocation_bytes = limits.allocation_bytes();
    let allocator_request = i32::try_from(allocation_bytes).map_err(|_| Error::Contract)?;
    let descriptor_size = u32::try_from(allocation_bytes).map_err(|_| Error::Contract)?;
    entered(&mut report, "projection-allocation-entered", &mut progress);
    // SAFETY: one explicitly bounded cached system-heap allocation, under i32 limits.
    let memory = unsafe { (calls.alloc)(25, 1, allocator_request) };
    if memory.is_null() || memory.addr() % 4096 != 0 {
        report.queue.stage = "projection-allocation-refused-pinned";
        return Ok(report);
    }
    // SAFETY: fresh, exclusive, correctly aligned allocator-owned extent.
    unsafe {
        std::ptr::write_bytes(memory.cast::<u8>(), 0, allocation_bytes);
    }
    // SAFETY: only this driver's fresh allocator-owned pointer is passed back.
    let fd = unsafe { (calls.to_fd)(memory) };
    let Ok(fd_u32) = u32::try_from(fd) else {
        report.queue.stage = "projection-fd-refused-pinned";
        return Ok(report);
    };
    report.queue.allocation_succeeded = true;
    entered(&mut report, "projection-fd-map-entered", &mut progress);
    // SAFETY: exact aligned pointer/FD/length and original validated effective domain.
    let result = unsafe { (calls.map)(domain, fd, memory, 0, allocation_bytes, 2) };
    report.queue.mapping_result = Some(result);
    if result != 0 {
        report.queue.stage = "projection-fd-map-refused-pinned";
        return Ok(report);
    }
    let mut queue = null_mut();
    entered(
        &mut report,
        "projection-queue-create-entered",
        &mut progress,
    );
    // SAFETY: fixed bounded queues, no callbacks/context; exact checked public ABI.
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
    report.queue.queue_create_result = Some(result);
    if result != 0 || queue.is_null() {
        report.queue.stage = "projection-queue-create-refused-pinned";
        return Ok(report);
    }
    let mut queue_id = u64::MAX;
    entered(
        &mut report,
        "projection-queue-export-entered",
        &mut progress,
    );
    // SAFETY: one live driver-owned queue and poisoned writable u64 export ID.
    let result = unsafe { (calls.queue_export)(queue, &raw mut queue_id) };
    report.queue.queue_export_result = Some(result);
    if result != 0 || queue_id == u64::MAX {
        report.queue.stage = "projection-queue-export-refused-pinned";
        return Ok(report);
    }
    let mut start = compute::start_wire(queue_id);
    start[32..40].copy_from_slice(&u64::from(descriptor_size).to_le_bytes());
    let mut argument = RemoteBuffer {
        pointer: start.as_mut_ptr().cast(),
        bytes: start.len(),
    };
    entered(&mut report, "projection-queue-start-entered", &mut progress);
    // SAFETY: same independently checked 48-byte seven-argument start frame;
    // only its bounded mapped-memory allowance follows this one allocation.
    let result = unsafe { (calls.invoke)(handle, compute::START_SCALARS, &raw mut argument) };
    report.queue.queue_start_result = Some(result);
    if result != 0 {
        report.queue.stage = "projection-queue-start-refused-pinned";
        return Ok(report);
    }
    let expected = QueueBuffer {
        fd: fd_u32,
        size: descriptor_size,
        offset: 0,
        flags: 0x90,
        pointer: memory,
    };
    let (work_result, session_failed) = {
        let mut session = Session {
            report: &mut report,
            queue_write: &calls.queue_write,
            queue_read: calls.queue_read,
            queue,
            memory,
            expected,
            limits,
            work_deadline,
            progress: &mut progress,
            failed: false,
        };
        let result = work(&mut session);
        (result.is_ok(), session.failed)
    };
    if !work_result || session_failed || report.operations_completed == 0 {
        if !session_failed {
            report.queue.stage = "projection-caller-rejected-pinned";
        }
        return Ok(report);
    }
    if Instant::now() >= work_deadline {
        report.deadline_expired = true;
        report.queue.stage = "projection-work-expired-pinned";
        return Ok(report);
    }
    report.work_completed = true;
    report.queue = cleanup_extent(
        report.queue,
        &calls.cleanup_calls(),
        handle,
        queue,
        domain,
        fd,
        memory,
        allocation_bytes,
        &mut |queue| {
            let mut snapshot = report;
            snapshot.queue = *queue;
            progress(&snapshot);
        },
    );
    if Instant::now() >= deadline {
        report.deadline_expired = true;
    }
    progress(&report);
    Ok(report)
}

struct Session<'a, P> {
    report: &'a mut Q2ProjectionReport,
    queue_write: &'a WriteCall,
    queue_read: QueueRead,
    queue: *mut c_void,
    memory: *mut c_void,
    expected: QueueBuffer,
    limits: Q2ProjectionLimits,
    work_deadline: Instant,
    progress: &'a mut P,
    failed: bool,
}

impl<P: FnMut(&Q2ProjectionReport)> Session<'_, P> {
    fn fail(&mut self, stage: &'static str) -> Error {
        self.failed = true;
        self.report.queue.poisoned = true;
        self.report.queue.stage = stage;
        (self.progress)(self.report);
        Error::Contract
    }

    fn within_deadline(&mut self) -> bool {
        if Instant::now() < self.work_deadline {
            return true;
        }
        self.report.deadline_expired = true;
        let _ = self.fail("projection-work-expired-pinned");
        false
    }

    fn read(&mut self, sequence: u64) -> bool {
        let deadline = (Instant::now() + Duration::from_secs(3)).min(self.work_deadline);
        loop {
            if Instant::now() >= deadline {
                if Instant::now() >= self.work_deadline {
                    self.report.deadline_expired = true;
                }
                let _ = self.fail("projection-read-expired-pinned");
                return false;
            }
            let mut flags = u32::MAX;
            let mut count = u32::MAX;
            let mut bytes = u32::MAX;
            let mut descriptor = QueueBuffer {
                fd: u32::MAX,
                size: u32::MAX,
                offset: u32::MAX,
                flags: u32::MAX,
                pointer: null_mut(),
            };
            let mut reply = [0xa5; compute::COMPUTE_RESPONSE_BYTES];
            // SAFETY: exact checked no-block queue ABI; all writable outputs
            // have their full declared capacity and no callback accesses them.
            let result = unsafe {
                (self.queue_read)(
                    self.queue,
                    &raw mut flags,
                    1,
                    &raw mut count,
                    &raw mut descriptor,
                    88,
                    &raw mut bytes,
                    reply.as_mut_ptr(),
                )
            };
            self.report.queue.queue_read_result = Some(result);
            if result == 516 {
                std::thread::sleep(Duration::from_millis(1));
                continue;
            }
            self.report.queue.response_validated = result == 0
                && count == 1
                && bytes == 88
                && flags & 3 == 3
                && flags & !0x1f == 0
                && descriptor.fd == self.expected.fd
                && descriptor.size == self.expected.size
                && descriptor.offset == self.expected.offset
                && descriptor.flags == self.expected.flags
                && descriptor.pointer == self.expected.pointer
                && projection::response_valid(&reply, sequence);
            if !self.report.queue.response_validated {
                let _ = self.fail("projection-response-refused-pinned");
            }
            return self.report.queue.response_validated;
        }
    }
}

impl<P: FnMut(&Q2ProjectionReport)> GuardedQ2Submitter for Session<'_, P> {
    fn submit(
        &mut self,
        weights: &[u8],
        activation: &[[f32; 256]; 2],
        output: &mut [f32],
    ) -> Result<(), Error> {
        output.fill(f32::NAN);
        if self.failed || self.report.operations_submitted >= self.limits.operations {
            return Err(self.fail("projection-session-bound-refused-pinned"));
        }
        if !self.within_deadline() {
            return Err(Error::Contract);
        }
        let Ok(rows) =
            projection::validate_inputs(self.limits.rows, weights, activation, output.len())
        else {
            return Err(self.fail("projection-input-refused-pinned"));
        };
        let Ok(base) = u64::try_from(self.memory.addr()) else {
            return Err(self.fail("projection-address-refused-pinned"));
        };
        // SAFETY: the single previous operation completed and cache invalidation
        // was validated before any allocation reuse. There is no DSP consumer
        // before the first write and no mutable view survives this preparation.
        let image = unsafe {
            std::slice::from_raw_parts_mut(self.memory.cast::<u8>(), self.limits.allocation_bytes())
        };
        let Ok(active) =
            projection::image(image, base, self.expected.fd, rows, weights, activation)
        else {
            return Err(self.fail("projection-frame-refused-pinned"));
        };
        if !self.within_deadline() {
            return Err(Error::Contract);
        }
        self.report.queue.response_validated = false;
        self.report.queue.numeric_validated = false;
        self.report.operations_submitted += 1;
        self.report.queue.compute_submitted = true;
        let sequence = u64::from(self.report.operations_submitted);
        let request = projection::request(sequence);
        let mut descriptor = self.expected;
        entered(self.report, "projection-queue-write-entered", self.progress);
        let result = write_packet(self.queue_write, self.queue, &mut descriptor, &request);
        self.report.queue.queue_write_result = Some(result);
        if result != 0 {
            return Err(self.fail("projection-write-refused-pinned"));
        }
        entered(self.report, "projection-queue-read-entered", self.progress);
        if !self.read(sequence) {
            return Err(Error::Contract);
        }
        // SAFETY: exact correlated response validated same FD, full allocation,
        // pointer and cache flags. No asynchronous consumer remains, and only
        // the checked active output subsection is borrowed until copying ends.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                self.memory.cast::<u8>().wrapping_add(active.output),
                rows * 8,
            )
        };
        if !projection::output_valid(bytes) {
            return Err(self.fail("projection-numeric-refused-pinned"));
        }
        if !self.within_deadline() {
            return Err(Error::Contract);
        }
        for (destination, value) in output.iter_mut().zip(bytes.as_chunks::<4>().0) {
            *destination = f32::from_le_bytes(*value);
        }
        self.report.queue.numeric_validated = true;
        self.report.operations_completed += 1;
        entered(self.report, "projection-operation-complete", self.progress);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct MockState {
        writes: u32,
        sequence: u64,
        descriptor: [u32; 4],
        pointer: usize,
        write_error: bool,
        stale_reply: bool,
        bad_numeric: bool,
    }

    static SERIAL: Mutex<()> = Mutex::new(());
    static STATE: Mutex<MockState> = Mutex::new(MockState {
        writes: 0,
        sequence: 0,
        descriptor: [0; 4],
        pointer: 0,
        write_error: false,
        stale_reply: false,
        bad_numeric: false,
    });

    unsafe extern "C" fn write(
        _queue: *mut c_void,
        _flags: u32,
        _count: u32,
        descriptor: *mut QueueBuffer,
        _bytes: u32,
        message: *const u8,
    ) -> i32 {
        let Ok(mut state) = STATE.lock() else {
            return -77;
        };
        state.writes += 1;
        // SAFETY: only the session test calls this mock, with its live full
        // descriptor and exact 24-byte request, as checked by the safe submitter.
        let descriptor = unsafe { *descriptor };
        // SAFETY: the request has the full fixed public request extent.
        let request = unsafe { std::slice::from_raw_parts(message, 24) };
        let Ok(sequence) = request[..8].try_into() else {
            return -77;
        };
        state.sequence = u64::from_le_bytes(sequence);
        state.descriptor = [
            descriptor.fd,
            descriptor.size,
            descriptor.offset,
            descriptor.flags,
        ];
        state.pointer = descriptor.pointer.expose_provenance();
        let Ok(allocation_bytes) = usize::try_from(descriptor.size) else {
            return -77;
        };
        // SAFETY: the test's unique Vec owns this exact allocation; submit has
        // relinquished its mutable preparation slice before mock consumption.
        let memory = unsafe {
            std::slice::from_raw_parts_mut(descriptor.pointer.cast::<u8>(), allocation_bytes)
        };
        let Ok(offset) = memory[136..144].try_into() else {
            return -77;
        };
        let Ok(offset) = usize::try_from(u64::from_le_bytes(offset)) else {
            return -77;
        };
        let Ok(rows) = memory[160..164].try_into() else {
            return -77;
        };
        let Ok(rows) = usize::try_from(u32::from_le_bytes(rows)) else {
            return -77;
        };
        let value = if state.bad_numeric { 7.5_f32 } else { 7.0_f32 };
        for slot in memory[offset..offset + rows * 8].as_chunks_mut::<4>().0 {
            *slot = value.to_le_bytes();
        }
        if state.write_error { 14 } else { 0 }
    }

    unsafe extern "C" fn read(
        _queue: *mut c_void,
        flags: *mut u32,
        _capacity: u32,
        count: *mut u32,
        descriptor: *mut QueueBuffer,
        _maximum: u32,
        bytes: *mut u32,
        reply: *mut u8,
    ) -> i32 {
        let Ok(state) = STATE.lock() else {
            return -77;
        };
        let [fd, size, offset, buffer_flags] = state.descriptor;
        let mut response = [0; 88];
        let sequence = if state.stale_reply {
            state.sequence - 1
        } else {
            state.sequence
        };
        response[..8].copy_from_slice(&sequence.to_le_bytes());
        for (offset, value) in [(24, 1_u32), (28, 1), (32, 3), (36, 1)] {
            response[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        // SAFETY: safe session supplies writable capacities for each output;
        // the pointer was originally recorded from that test-owned allocation.
        unsafe {
            *flags = 3;
        }
        // SAFETY: safe session supplies one writable u32 count.
        unsafe {
            *count = 1;
        }
        // SAFETY: safe session supplies one writable u32 byte extent.
        unsafe {
            *bytes = 88;
        }
        // SAFETY: safe session supplies one writable complete descriptor.
        unsafe {
            *descriptor = QueueBuffer {
                fd,
                size,
                offset,
                flags: buffer_flags,
                pointer: std::ptr::with_exposed_provenance_mut(state.pointer),
            };
        }
        // SAFETY: safe session supplies a nonoverlapping writable 88-byte reply.
        unsafe {
            std::ptr::copy_nonoverlapping(response.as_ptr(), reply, 88);
        }
        0
    }

    #[test]
    fn actual_session_reuses_only_correlated_results_and_never_retries_failure() -> Result<(), Error>
    {
        let _serial = SERIAL.lock().map_err(|_| Error::Contract)?;
        let fixed = compute::q2_fixture_image(0, 7);
        let weights = &fixed[1024..3328];
        let mut activation = [[0.0; 256]; 2];
        for (index, value) in activation.iter_mut().flatten().enumerate() {
            *value = f32::from_le_bytes(
                fixed[5120 + index * 4..5124 + index * 4]
                    .try_into()
                    .map_err(|_| Error::Contract)?,
            );
        }
        for case in 0..5 {
            {
                let mut state = STATE.lock().map_err(|_| Error::Contract)?;
                state.writes = 0;
                state.sequence = 0;
                state.write_error = case == 1;
                state.stale_reply = case == 2;
                state.bad_numeric = case == 3;
            }
            let limits = Q2ProjectionLimits::new(32, 2, Duration::from_secs(4))?;
            let mut memory = vec![0; limits.allocation_bytes()];
            let pointer = memory.as_mut_ptr().cast::<c_void>();
            let expected = QueueBuffer {
                fd: 7,
                size: u32::try_from(memory.len()).map_err(|_| Error::Contract)?,
                offset: 0,
                flags: 0x90,
                pointer,
            };
            let mut report = Q2ProjectionReport::new(SessionPreparationReport::new(0x81));
            report.queue.poisoned = true;
            let queue_write = WriteCall::Nonblocking(write);
            let mut progress = |_: &Q2ProjectionReport| {};
            let mut session = Session {
                report: &mut report,
                queue_write: &queue_write,
                queue_read: read,
                queue: null_mut(),
                memory: pointer,
                expected,
                limits,
                work_deadline: if case == 4 {
                    Instant::now()
                } else {
                    Instant::now() + Duration::from_secs(1)
                },
                progress: &mut progress,
                failed: false,
            };
            let mut output = [99.0; 64];
            let first = session.submit(weights, &activation, &mut output);
            if case == 0 {
                assert!(first.is_ok());
                assert_eq!(output, [7.0; 64]);
                assert!(session.submit(weights, &activation, &mut output).is_ok());
                assert_eq!(session.report.operations_completed, 2);
                assert!(session.submit(weights, &activation, &mut output).is_err());
                assert!(session.failed);
                assert_eq!(STATE.lock().map_err(|_| Error::Contract)?.writes, 2);
            } else {
                assert!(first.is_err());
                assert!(output.iter().all(|value| value.is_nan()));
                assert!(session.submit(weights, &activation, &mut output).is_err());
                assert_eq!(session.report.operations_completed, 0);
                assert_eq!(
                    STATE.lock().map_err(|_| Error::Contract)?.writes,
                    u32::from(case != 4)
                );
            }
            assert!(session.report.queue.poisoned);
            assert_eq!(session.report.queue.queue_stop_result, None);
            assert_eq!(session.report.queue.unmapping_result, None);
            assert!(!session.report.completed_cycle());
        }
        Ok(())
    }
}
