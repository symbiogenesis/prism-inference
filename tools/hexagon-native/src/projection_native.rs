//! One bounded reusable queue for fixed-K/N guarded-Q2 projection operations.

use super::{
    Driver, QueueBuffer, QueueRead, RemoteBuffer, WriteCall, cleanup_extent, compute, null_mut,
    open_module, write_packet,
};
use crate::{
    Error, GuardedQ2Submitter, HMX_Q2_BATCH_OPERATIONS, HMX_Q2_BATCH10_OPERATIONS,
    HmxQ2ProjectionLimits, HmxQ2Submitter, HmxQ2WeightHandle, HmxQ2WeightRegistry,
    NativeQ2ProjectionLimits, NativeQ2Submitter, NativeQ2WeightHandle, NativeQ2WeightRegistry,
    Q2ProjectionLimits, Q2ProjectionReport, SessionPreparationReport, hmx_projection, native_q2,
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
    progress: impl FnMut(&Q2ProjectionReport),
    work: impl FnOnce(&mut dyn GuardedQ2Submitter) -> Result<(), Error>,
) -> Result<Q2ProjectionReport, Error> {
    run_mode(
        driver,
        package,
        limits,
        Mode::Guarded,
        0,
        progress,
        |session| work(session),
    )
}

pub(super) fn run_hmx(
    driver: Driver,
    package: &Path,
    limits: HmxQ2ProjectionLimits,
    progress: impl FnMut(&Q2ProjectionReport),
    work: impl FnOnce(&mut dyn HmxQ2Submitter) -> Result<(), Error>,
) -> Result<Q2ProjectionReport, Error> {
    let mode = match limits.batch_operations {
        HMX_Q2_BATCH_OPERATIONS => Mode::HmxBatch4,
        HMX_Q2_BATCH10_OPERATIONS => Mode::HmxBatch10,
        _ => Mode::Hmx,
    };
    run_mode(
        driver,
        package,
        limits.common,
        mode,
        limits.resident_weight_bytes,
        progress,
        |session| work(session),
    )
}

pub(super) fn run_native_q2(
    driver: Driver,
    package: &Path,
    limits: NativeQ2ProjectionLimits,
    progress: impl FnMut(&Q2ProjectionReport),
    work: impl FnOnce(&mut dyn NativeQ2Submitter) -> Result<(), Error>,
) -> Result<Q2ProjectionReport, Error> {
    run_mode(
        driver,
        package,
        limits.common,
        Mode::NativeQ2 {
            columns: limits.columns,
        },
        limits.resident_weight_bytes,
        progress,
        |session| work(session),
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Guarded,
    Hmx,
    HmxBatch4,
    HmxBatch10,
    NativeQ2 { columns: usize },
}

impl Mode {
    const fn is_hmx(self) -> bool {
        matches!(self, Self::Hmx | Self::HmxBatch4 | Self::HmxBatch10)
    }

    const fn is_native_q2(self) -> bool {
        matches!(self, Self::NativeQ2 { .. })
    }

    const fn batch_operations(self) -> u32 {
        match self {
            Self::Guarded | Self::Hmx | Self::NativeQ2 { .. } => 1,
            Self::HmxBatch4 => 4,
            Self::HmxBatch10 => 10,
        }
    }

    const fn allocation_bytes(self, rows: usize) -> usize {
        match self {
            Self::Guarded => projection::layout(rows).allocation,
            Self::Hmx => hmx_projection::layout(rows).allocation,
            Self::HmxBatch4 => hmx_projection::batch_layout(rows).allocation,
            Self::HmxBatch10 => hmx_projection::batch10_layout(rows).allocation,
            Self::NativeQ2 { columns } => native_q2::working_allocation(columns, rows),
        }
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "One shared terminal ownership protocol covers both fixed signed-source arithmetic modes"
)]
fn run_mode<P: FnMut(&Q2ProjectionReport)>(
    driver: Driver,
    package: &Path,
    limits: Q2ProjectionLimits,
    mode: Mode,
    resident_weight_bytes: usize,
    mut progress: P,
    work: impl FnOnce(&mut Session<'_, P>) -> Result<(), Error>,
) -> Result<Q2ProjectionReport, Error> {
    let started = Instant::now();
    let deadline = started + limits.duration;
    let work_deadline = started
        + limits
            .duration
            .checked_sub(Duration::from_secs(3))
            .ok_or(Error::Contract)?;
    let mut report = Q2ProjectionReport::new(SessionPreparationReport::new(0));
    report.batch_operations = mode.batch_operations();
    report.queue.hmx_bitplane = mode.is_hmx();
    let (queue_report, opened) = open_module(
        driver,
        package,
        &mut |queue| {
            report.queue = *queue;
            report.queue.hmx_bitplane = mode.is_hmx();
            progress(&report);
        },
        compute::ComputeFixture::GuardedQ2NoBlock,
    )?;
    report.queue = queue_report;
    report.queue.hmx_bitplane = mode.is_hmx();
    let Some((handle, domain, calls)) = opened else {
        return Ok(report);
    };
    if Instant::now() >= work_deadline {
        report.deadline_expired = true;
        report.queue.stage = "projection-initialization-expired-pinned";
        return Ok(report);
    }
    let allocation_bytes =
        (mode.allocation_bytes(limits.rows) + resident_weight_bytes).next_multiple_of(4096);
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
    report.mapped_allocation_bytes = allocation_bytes;
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
    let mut start = match mode {
        Mode::Guarded => {
            let mut start = compute::start_wire(queue_id);
            start[32..40].copy_from_slice(&u64::from(descriptor_size).to_le_bytes());
            start
        }
        Mode::Hmx | Mode::HmxBatch4 | Mode::HmxBatch10 => {
            hmx_projection::start_wire(queue_id, allocation_bytes)?
        }
        Mode::NativeQ2 { .. } => native_q2::start_wire(queue_id, allocation_bytes)?,
    };
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
            mode,
            resident_weights: HmxQ2WeightRegistry::new(resident_weight_bytes)?,
            native_resident_weights: NativeQ2WeightRegistry::new(resident_weight_bytes)?,
            work_deadline,
            progress: &mut progress,
            failed: false,
        };
        let result = work(&mut session);
        (result.is_ok(), session.failed)
    };
    if !work_result
        || session_failed
        || report.operations_completed == 0
        || report.queue_writes_completed == 0
        || report.operations_completed
            != report
                .queue_writes_completed
                .saturating_mul(report.batch_operations)
    {
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
    mode: Mode,
    resident_weights: HmxQ2WeightRegistry,
    native_resident_weights: NativeQ2WeightRegistry,
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

    fn read(&mut self, sequence: u64, batch_operations: u32) -> bool {
        let started = Instant::now();
        let deadline = (started + Duration::from_secs(3)).min(self.work_deadline);
        let spin_deadline = (started + Duration::from_millis(15)).min(deadline);
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
                if Instant::now() < spin_deadline {
                    std::hint::spin_loop();
                } else {
                    std::thread::sleep(Duration::from_millis(1));
                }
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
                && projection::response_batch_valid(&reply, sequence, batch_operations);
            if !self.report.queue.response_validated {
                let _ = self.fail("projection-response-refused-pinned");
            }
            return self.report.queue.response_validated;
        }
    }
}

#[derive(Clone, Copy)]
enum Activation<'a> {
    Guarded(&'a [[f32; 256]; 2]),
    Hmx(&'a [[f32; 128]; 8]),
    NativeQ2(&'a [f32]),
}

#[derive(Clone, Copy)]
enum Weights<'a> {
    Transient(&'a [u8]),
    Resident(&'a HmxQ2WeightHandle),
    NativeTransient {
        columns: usize,
        rows: usize,
        bytes: &'a [u8],
    },
    NativeResident(&'a NativeQ2WeightHandle),
}

#[derive(Clone, Copy)]
enum BatchWeights<'a, const B: usize> {
    Transient([&'a [u8]; B]),
    Resident([&'a HmxQ2WeightHandle; B]),
}

impl<P: FnMut(&Q2ProjectionReport)> Session<'_, P> {
    #[allow(
        clippy::too_many_lines,
        reason = "Keep both transient and immutable arena validation beside the shared correlated native submission"
    )]
    fn submit_inner(
        &mut self,
        weights: Weights<'_>,
        activation: Activation<'_>,
        output: &mut [f32],
    ) -> Result<(), Error> {
        output.fill(f32::NAN);
        if self.failed
            || self.report.operations_submitted >= self.limits.operations
            || self.report.queue_writes_submitted >= self.limits.operations
        {
            return Err(self.fail("projection-session-bound-refused-pinned"));
        }
        if !self.within_deadline() {
            return Err(Error::Contract);
        }
        let rows = match (weights, &activation, self.mode) {
            (Weights::Transient(weights), Activation::Guarded(values), Mode::Guarded) => {
                projection::validate_inputs(self.limits.rows, weights, values, output.len())
            }
            (Weights::Transient(weights), Activation::Hmx(values), Mode::Hmx) => {
                hmx_projection::validate_inputs(self.limits.rows, weights, values, output.len())
            }
            (Weights::Resident(handle), Activation::Hmx(values), Mode::Hmx) => self
                .resident_weights
                .range(handle)
                .and_then(|_| {
                    hmx_projection::validate_activation(
                        self.limits.rows,
                        handle.rows(),
                        values,
                        output.len(),
                    )
                })
                .map(|()| handle.rows()),
            (
                Weights::NativeTransient {
                    columns,
                    rows,
                    bytes,
                },
                Activation::NativeQ2(values),
                Mode::NativeQ2 {
                    columns: max_columns,
                },
            ) => native_q2::validate_inputs(
                max_columns,
                self.limits.rows,
                columns,
                rows,
                bytes,
                values,
                output.len(),
            )
            .map(|()| rows),
            (
                Weights::NativeResident(handle),
                Activation::NativeQ2(values),
                Mode::NativeQ2 {
                    columns: max_columns,
                },
            ) => self
                .native_resident_weights
                .range(handle)
                .and_then(|_| {
                    native_q2::validate_activation(
                        max_columns,
                        self.limits.rows,
                        handle.columns(),
                        handle.rows(),
                        values,
                        output.len(),
                    )
                })
                .map(|()| handle.rows()),
            _ => Err(Error::Contract),
        };
        let Ok(rows) = rows else {
            return Err(self.fail("projection-input-refused-pinned"));
        };
        let Ok(base) = u64::try_from(self.memory.addr()) else {
            return Err(self.fail("projection-address-refused-pinned"));
        };
        let Ok(allocation_bytes) = usize::try_from(self.expected.size) else {
            return Err(self.fail("projection-allocation-size-refused-pinned"));
        };
        let working_bytes = self.mode.allocation_bytes(self.limits.rows);
        // SAFETY: the single previous operation completed and cache invalidation
        // was validated before any allocation reuse. There is no DSP consumer
        // before the first write and no mutable view survives this preparation.
        let image =
            unsafe { std::slice::from_raw_parts_mut(self.memory.cast::<u8>(), allocation_bytes) };
        let active = match (weights, activation) {
            (Weights::Transient(weights), Activation::Guarded(values)) => projection::image(
                &mut image[..working_bytes],
                base,
                self.expected.fd,
                rows,
                weights,
                values,
            ),
            (Weights::Transient(weights), Activation::Hmx(values)) => hmx_projection::image(
                &mut image[..working_bytes],
                base,
                self.expected.fd,
                rows,
                weights,
                values,
            ),
            (Weights::Resident(handle), Activation::Hmx(values)) => hmx_projection::image_resident(
                image,
                base,
                self.expected.fd,
                working_bytes,
                handle,
                values,
            ),
            (
                Weights::NativeTransient {
                    columns,
                    rows: active_rows,
                    bytes,
                },
                Activation::NativeQ2(values),
            ) => native_q2::image(
                &mut image[..working_bytes],
                base,
                self.expected.fd,
                columns,
                active_rows,
                bytes,
                values,
            ),
            (Weights::NativeResident(handle), Activation::NativeQ2(values)) => {
                native_q2::image_resident(
                    image,
                    base,
                    self.expected.fd,
                    working_bytes,
                    handle,
                    values,
                )
            }
            _ => Err(Error::Contract),
        };
        let Ok(active) = active else {
            return Err(self.fail("projection-frame-refused-pinned"));
        };
        image[8..16].copy_from_slice(&u64::from(self.expected.size).to_le_bytes());
        if !self.within_deadline() {
            return Err(Error::Contract);
        }
        self.report.queue.response_validated = false;
        self.report.queue.numeric_validated = false;
        self.report.operations_submitted += 1;
        self.report.queue_writes_submitted += 1;
        if matches!(weights, Weights::Resident(_) | Weights::NativeResident(_)) {
            self.report.resident_submissions += 1;
        }
        self.report.queue.compute_submitted = true;
        let sequence = u64::from(self.report.queue_writes_submitted);
        let request = projection::request(sequence);
        let mut descriptor = self.expected;
        entered(self.report, "projection-queue-write-entered", self.progress);
        let result = write_packet(self.queue_write, self.queue, &mut descriptor, &request);
        self.report.queue.queue_write_result = Some(result);
        if result != 0 {
            return Err(self.fail("projection-write-refused-pinned"));
        }
        entered(self.report, "projection-queue-read-entered", self.progress);
        if !self.read(sequence, 1) {
            return Err(Error::Contract);
        }
        // SAFETY: exact correlated response validated same FD, full allocation,
        // pointer and cache flags. No asynchronous consumer remains, and only
        // the checked active output subsection is borrowed until copying ends.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                self.memory.cast::<u8>().wrapping_add(active.output),
                size_of_val(output),
            )
        };
        let output_valid = match self.mode {
            Mode::Guarded => projection::output_valid(bytes),
            Mode::Hmx | Mode::HmxBatch4 | Mode::HmxBatch10 => hmx_projection::output_valid(bytes),
            Mode::NativeQ2 { .. } => native_q2::output_valid(bytes),
        };
        if !output_valid {
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
        self.report.queue_writes_completed += 1;
        entered(self.report, "projection-operation-complete", self.progress);
        Ok(())
    }

    #[allow(
        clippy::too_many_lines,
        reason = "Keep batch capacity reservation, descriptor regeneration and full-batch output checks together"
    )]
    fn submit_batch_inner<const B: usize, const CMD_PAD: usize>(
        &mut self,
        expected_mode: Mode,
        weights: BatchWeights<'_, B>,
        activations: [&[[f32; 128]; 8]; B],
        mut outputs: [&mut [f32]; B],
    ) -> Result<(), Error> {
        for output in &mut outputs {
            output.fill(f32::NAN);
        }
        let Ok(batch_u32) = u32::try_from(B) else {
            return Err(self.fail("projection-session-bound-refused-pinned"));
        };
        let Some(next_operations) = self
            .report
            .operations_submitted
            .checked_add(batch_u32)
            .filter(|&next| !self.failed && next <= self.limits.operations)
        else {
            return Err(self.fail("projection-session-bound-refused-pinned"));
        };
        let Some(next_writes) = self
            .report
            .queue_writes_submitted
            .checked_add(1)
            .filter(|&next| next <= self.limits.operations)
        else {
            return Err(self.fail("projection-session-bound-refused-pinned"));
        };
        if !self.within_deadline() {
            return Err(Error::Contract);
        }
        if self.mode != expected_mode {
            return Err(self.fail("projection-input-refused-pinned"));
        }
        let output_lens = std::array::from_fn(|member| outputs[member].len());
        let rows = match weights {
            BatchWeights::Transient(weights) => hmx_projection::validate_batch_inputs::<B>(
                self.limits.rows,
                weights,
                activations,
                output_lens,
            ),
            BatchWeights::Resident(handles) => (|| {
                for handle in handles {
                    self.resident_weights.range(handle)?;
                }
                hmx_projection::validate_batch_resident::<B>(
                    self.limits.rows,
                    handles,
                    activations,
                    output_lens,
                )
            })(),
        };
        let Ok(rows) = rows else {
            return Err(self.fail("projection-input-refused-pinned"));
        };
        let Ok(base) = u64::try_from(self.memory.addr()) else {
            return Err(self.fail("projection-address-refused-pinned"));
        };
        let Ok(allocation_bytes) = usize::try_from(self.expected.size) else {
            return Err(self.fail("projection-allocation-size-refused-pinned"));
        };
        let working_bytes = self.mode.allocation_bytes(self.limits.rows);
        // SAFETY: the single previous batch completed and cache invalidation
        // was validated before any allocation reuse. There is no DSP consumer
        // before the write and no mutable view survives this preparation.
        let image =
            unsafe { std::slice::from_raw_parts_mut(self.memory.cast::<u8>(), allocation_bytes) };
        let active = match weights {
            BatchWeights::Transient(weights) => hmx_projection::image_batch::<B, CMD_PAD>(
                &mut image[..working_bytes],
                base,
                self.expected.fd,
                rows,
                weights,
                activations,
            ),
            BatchWeights::Resident(handles) => hmx_projection::image_batch_resident::<B, CMD_PAD>(
                image,
                base,
                self.expected.fd,
                working_bytes,
                handles,
                activations,
            ),
        };
        let Ok(active) = active else {
            return Err(self.fail("projection-frame-refused-pinned"));
        };
        image[8..16].copy_from_slice(&u64::from(self.expected.size).to_le_bytes());
        if !self.within_deadline() {
            return Err(Error::Contract);
        }
        let sequence = u64::from(next_writes);
        let Ok(request) = projection::request_batch(sequence, batch_u32) else {
            return Err(self.fail("projection-frame-refused-pinned"));
        };
        self.report.queue.response_validated = false;
        self.report.queue.numeric_validated = false;
        self.report.operations_submitted = next_operations;
        self.report.queue_writes_submitted = next_writes;
        if matches!(weights, BatchWeights::Resident(_)) {
            self.report.resident_submissions += batch_u32;
        }
        self.report.queue.compute_submitted = true;
        let mut descriptor = self.expected;
        entered(self.report, "projection-queue-write-entered", self.progress);
        let result = write_packet(self.queue_write, self.queue, &mut descriptor, &request);
        self.report.queue.queue_write_result = Some(result);
        if result != 0 {
            return Err(self.fail("projection-write-refused-pinned"));
        }
        entered(self.report, "projection-queue-read-entered", self.progress);
        if !self.read(sequence, batch_u32) {
            return Err(Error::Contract);
        }
        let member_bytes = rows * 32;
        let total_output_bytes = B * member_bytes;
        // SAFETY: exact correlated response validated same FD, full allocation,
        // pointer, n_tensors=3*B and n_ops=B. All B output regions are checked
        // before any member is copied to the caller.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                self.memory.cast::<u8>().wrapping_add(active.output),
                total_output_bytes,
            )
        };
        if !hmx_projection::output_valid(bytes) {
            return Err(self.fail("projection-numeric-refused-pinned"));
        }
        if !self.within_deadline() {
            return Err(Error::Contract);
        }
        for (member, output) in outputs.iter_mut().enumerate() {
            let slice = &bytes[member * member_bytes..(member + 1) * member_bytes];
            for (destination, value) in output.iter_mut().zip(slice.as_chunks::<4>().0) {
                *destination = f32::from_le_bytes(*value);
            }
        }
        self.report.queue.numeric_validated = true;
        self.report.operations_completed = next_operations;
        self.report.queue_writes_completed = next_writes;
        entered(self.report, "projection-operation-complete", self.progress);
        Ok(())
    }
}

impl<P: FnMut(&Q2ProjectionReport)> GuardedQ2Submitter for Session<'_, P> {
    fn remaining_operations(&self) -> Option<u64> {
        Some(if self.failed {
            0
        } else {
            u64::from(
                self.limits
                    .operations
                    .saturating_sub(self.report.operations_submitted),
            )
        })
    }

    fn submit(
        &mut self,
        weights: &[u8],
        activation: &[[f32; 256]; 2],
        output: &mut [f32],
    ) -> Result<(), Error> {
        self.submit_inner(
            Weights::Transient(weights),
            Activation::Guarded(activation),
            output,
        )
    }
}

impl<P: FnMut(&Q2ProjectionReport)> HmxQ2Submitter for Session<'_, P> {
    fn remaining_operations(&self) -> Option<u64> {
        Some(if self.failed {
            0
        } else {
            u64::from(
                self.limits
                    .operations
                    .saturating_sub(self.report.operations_submitted),
            )
        })
    }

    fn submit(
        &mut self,
        weights: &[u8],
        activation: &[[f32; 128]; 8],
        output: &mut [f32],
    ) -> Result<(), Error> {
        self.submit_inner(
            Weights::Transient(weights),
            Activation::Hmx(activation),
            output,
        )
    }

    fn residency_capacity_bytes(&self) -> usize {
        if self.failed || !self.mode.is_hmx() {
            0
        } else {
            self.resident_weights.remaining_bytes()
        }
    }

    fn prepare_weights(&mut self, weights: &[u8]) -> Result<Option<HmxQ2WeightHandle>, Error> {
        if self.failed || !self.mode.is_hmx() {
            return Err(self.fail("projection-resident-owner-refused-pinned"));
        }
        if !self.within_deadline() {
            return Err(Error::Contract);
        }
        let rows = weights.len() / 36;
        if hmx_projection::validate_inputs(self.limits.rows, weights, &[[0.0; 128]; 8], rows * 8)
            .is_err()
        {
            return Err(self.fail("projection-resident-input-refused-pinned"));
        }
        let handle = match self.resident_weights.prepare(weights) {
            Ok(Some(handle)) => handle,
            Ok(None) => return Ok(None),
            Err(_) => return Err(self.fail("projection-resident-reservation-refused-pinned")),
        };
        let range = self.resident_weights.range(&handle)?;
        let offset = self.mode.allocation_bytes(self.limits.rows) + range.start;
        // SAFETY: this fresh append-only reservation is inside the explicitly
        // admitted allocation. No queued consumer remains after a validated
        // previous operation. No existing immutable reservation is overwritten,
        // and the temporary exclusive slice ends before exposing its handle.
        let destination = unsafe {
            std::slice::from_raw_parts_mut(
                self.memory.cast::<u8>().wrapping_add(offset),
                weights.len(),
            )
        };
        destination.copy_from_slice(weights);
        self.report.resident_weight_bytes_prepared += weights.len();
        self.report.resident_weight_preparations += 1;
        if !self.within_deadline() {
            return Err(Error::Contract);
        }
        Ok(Some(handle))
    }

    fn submit_resident(
        &mut self,
        handle: &HmxQ2WeightHandle,
        activation: &[[f32; 128]; 8],
        output: &mut [f32],
    ) -> Result<(), Error> {
        self.submit_inner(
            Weights::Resident(handle),
            Activation::Hmx(activation),
            output,
        )
    }

    fn submit_batch4(
        &mut self,
        weights: [&[u8]; HMX_Q2_BATCH_OPERATIONS],
        activations: [&[[f32; 128]; 8]; HMX_Q2_BATCH_OPERATIONS],
        outputs: [&mut [f32]; HMX_Q2_BATCH_OPERATIONS],
    ) -> Result<(), Error> {
        self.submit_batch_inner::<HMX_Q2_BATCH_OPERATIONS, 2048>(
            Mode::HmxBatch4,
            BatchWeights::Transient(weights),
            activations,
            outputs,
        )
    }

    fn submit_resident_batch4(
        &mut self,
        handles: [&HmxQ2WeightHandle; HMX_Q2_BATCH_OPERATIONS],
        activations: [&[[f32; 128]; 8]; HMX_Q2_BATCH_OPERATIONS],
        outputs: [&mut [f32]; HMX_Q2_BATCH_OPERATIONS],
    ) -> Result<(), Error> {
        self.submit_batch_inner::<HMX_Q2_BATCH_OPERATIONS, 2048>(
            Mode::HmxBatch4,
            BatchWeights::Resident(handles),
            activations,
            outputs,
        )
    }

    fn submit_batch10(
        &mut self,
        weights: [&[u8]; HMX_Q2_BATCH10_OPERATIONS],
        activations: [&[[f32; 128]; 8]; HMX_Q2_BATCH10_OPERATIONS],
        outputs: [&mut [f32]; HMX_Q2_BATCH10_OPERATIONS],
    ) -> Result<(), Error> {
        self.submit_batch_inner::<HMX_Q2_BATCH10_OPERATIONS, 8192>(
            Mode::HmxBatch10,
            BatchWeights::Transient(weights),
            activations,
            outputs,
        )
    }

    fn submit_resident_batch10(
        &mut self,
        handles: [&HmxQ2WeightHandle; HMX_Q2_BATCH10_OPERATIONS],
        activations: [&[[f32; 128]; 8]; HMX_Q2_BATCH10_OPERATIONS],
        outputs: [&mut [f32]; HMX_Q2_BATCH10_OPERATIONS],
    ) -> Result<(), Error> {
        self.submit_batch_inner::<HMX_Q2_BATCH10_OPERATIONS, 8192>(
            Mode::HmxBatch10,
            BatchWeights::Resident(handles),
            activations,
            outputs,
        )
    }
}

impl<P: FnMut(&Q2ProjectionReport)> NativeQ2Submitter for Session<'_, P> {
    fn remaining_operations(&self) -> Option<u64> {
        Some(if self.failed {
            0
        } else {
            u64::from(
                self.limits
                    .operations
                    .saturating_sub(self.report.operations_submitted),
            )
        })
    }

    fn residency_capacity_bytes(&self) -> usize {
        if self.failed || !self.mode.is_native_q2() {
            0
        } else {
            self.native_resident_weights.remaining_bytes()
        }
    }

    fn prepare_weights(
        &mut self,
        columns: usize,
        rows: usize,
        weights: &[u8],
    ) -> Result<Option<NativeQ2WeightHandle>, Error> {
        let Mode::NativeQ2 {
            columns: max_columns,
        } = self.mode
        else {
            return Err(self.fail("projection-resident-owner-refused-pinned"));
        };
        if self.failed {
            return Err(self.fail("projection-resident-owner-refused-pinned"));
        }
        if !self.within_deadline() {
            return Err(Error::Contract);
        }
        if native_q2::validate_weights(max_columns, self.limits.rows, columns, rows, weights)
            .is_err()
        {
            return Err(self.fail("projection-resident-input-refused-pinned"));
        }
        let handle = match self.native_resident_weights.prepare(columns, rows, weights) {
            Ok(Some(handle)) => handle,
            Ok(None) => return Ok(None),
            Err(_) => return Err(self.fail("projection-resident-reservation-refused-pinned")),
        };
        let range = self.native_resident_weights.range(&handle)?;
        let offset = self.mode.allocation_bytes(self.limits.rows) + range.start;
        // SAFETY: this fresh append-only reservation is inside the explicitly
        // admitted allocation. No queued consumer remains after a validated
        // previous operation. No existing immutable reservation is overwritten,
        // and the temporary exclusive slice ends before exposing its handle.
        let destination = unsafe {
            std::slice::from_raw_parts_mut(
                self.memory.cast::<u8>().wrapping_add(offset),
                weights.len(),
            )
        };
        destination.copy_from_slice(weights);
        self.report.resident_weight_bytes_prepared += weights.len();
        self.report.resident_weight_preparations += 1;
        if !self.within_deadline() {
            return Err(Error::Contract);
        }
        Ok(Some(handle))
    }

    fn submit_resident(
        &mut self,
        handle: &NativeQ2WeightHandle,
        activation: &[f32],
        output: &mut [f32],
    ) -> Result<(), Error> {
        self.submit_inner(
            Weights::NativeResident(handle),
            Activation::NativeQ2(activation),
            output,
        )
    }

    fn submit(
        &mut self,
        columns: usize,
        rows: usize,
        weights: &[u8],
        activation: &[f32],
        output: &mut [f32],
    ) -> Result<(), Error> {
        self.submit_inner(
            Weights::NativeTransient {
                columns,
                rows,
                bytes: weights,
            },
            Activation::NativeQ2(activation),
            output,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct MockState {
        writes: u32,
        sequence: u64,
        batch_operations: u32,
        descriptor: [u32; 4],
        pointer: usize,
        write_error: bool,
        stale_reply: bool,
        bad_numeric: bool,
        bad_member: Option<usize>,
    }

    static SERIAL: Mutex<()> = Mutex::new(());
    static STATE: Mutex<MockState> = Mutex::new(MockState {
        writes: 0,
        sequence: 0,
        batch_operations: 1,
        descriptor: [0; 4],
        pointer: 0,
        write_error: false,
        stale_reply: false,
        bad_numeric: false,
        bad_member: None,
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
        let Ok(n_ops) = request[16..20].try_into() else {
            return -77;
        };
        let n_ops = u32::from_le_bytes(n_ops);
        state.sequence = u64::from_le_bytes(sequence);
        state.batch_operations = n_ops;
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
        let Ok(members) = usize::try_from(n_ops) else {
            return -77;
        };
        for member in 0..members {
            let desc_base = 24 + (3 * member + 2) * 56;
            let Ok(offset) = memory[desc_base..desc_base + 8].try_into() else {
                return -77;
            };
            let Ok(offset) = usize::try_from(u64::from_le_bytes(offset)) else {
                return -77;
            };
            let Ok(output_bytes) = memory[desc_base + 8..desc_base + 12].try_into() else {
                return -77;
            };
            let Ok(output_bytes) = usize::try_from(u32::from_le_bytes(output_bytes)) else {
                return -77;
            };
            let value = if state.bad_numeric || state.bad_member == Some(member) {
                7.5_f32
            } else {
                let Ok(step) = u8::try_from(member) else {
                    return -77;
                };
                7.0_f32 + f32::from(step)
            };
            for slot in memory[offset..offset + output_bytes].as_chunks_mut::<4>().0 {
                *slot = value.to_le_bytes();
            }
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
        let n_ops = state.batch_operations.max(1);
        let n_tensors = 3 * n_ops;
        response[..8].copy_from_slice(&sequence.to_le_bytes());
        for (offset, value) in [(24, 1_u32), (28, 1), (32, n_tensors), (36, n_ops)] {
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
    #[allow(
        clippy::too_many_lines,
        reason = "Keep correlated single-op session assertions and queue-write checks together"
    )]
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
                state.batch_operations = 1;
                state.write_error = case == 1;
                state.stale_reply = case == 2;
                state.bad_numeric = case == 3;
                state.bad_member = None;
            }
            let limits = Q2ProjectionLimits::new(32, 24_000, Duration::from_secs(4))?;
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
            // Seed prior accepted work to exercise the real 24,000-write edge
            // without submitting a diagnostic-sized workload in a unit test.
            report.operations_submitted = 23_998;
            report.operations_completed = 23_998;
            report.queue_writes_submitted = 23_998;
            report.queue_writes_completed = 23_998;
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
                mode: Mode::Guarded,
                resident_weights: HmxQ2WeightRegistry::new(0)?,
                native_resident_weights: NativeQ2WeightRegistry::new(0)?,
                work_deadline: if case == 4 {
                    Instant::now()
                } else {
                    Instant::now() + Duration::from_secs(1)
                },
                progress: &mut progress,
                failed: false,
            };
            let mut output = [99.0; 64];
            assert_eq!(GuardedQ2Submitter::remaining_operations(&session), Some(2));
            let first = GuardedQ2Submitter::submit(&mut session, weights, &activation, &mut output);
            if case == 0 {
                assert!(first.is_ok());
                assert_eq!(GuardedQ2Submitter::remaining_operations(&session), Some(1));
                assert_eq!(output, [7.0; 64]);
                assert!(
                    GuardedQ2Submitter::submit(&mut session, weights, &activation, &mut output)
                        .is_ok()
                );
                assert_eq!(session.report.operations_completed, 24_000);
                assert_eq!(session.report.operations_submitted, 24_000);
                assert_eq!(GuardedQ2Submitter::remaining_operations(&session), Some(0));
                assert!(
                    GuardedQ2Submitter::submit(&mut session, weights, &activation, &mut output)
                        .is_err()
                );
                assert!(session.failed);
                assert_eq!(STATE.lock().map_err(|_| Error::Contract)?.writes, 2);
            } else {
                assert!(first.is_err());
                assert_eq!(GuardedQ2Submitter::remaining_operations(&session), Some(0));
                assert!(output.iter().all(|value| value.is_nan()));
                assert!(
                    GuardedQ2Submitter::submit(&mut session, weights, &activation, &mut output)
                        .is_err()
                );
                assert_eq!(session.report.operations_completed, 23_998);
                assert_eq!(
                    session.report.operations_submitted,
                    23_998 + u32::from(case != 4)
                );
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

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "Keep eight-plane HMX session assertions and queue-write checks together"
    )]
    fn hmx_session_checks_all_eight_planes_and_retains_uncertain_owners() -> Result<(), Error> {
        let _serial = SERIAL.lock().map_err(|_| Error::Contract)?;
        let mut weights = vec![0x55; 1152];
        for tile in weights.as_chunks_mut::<576>().0 {
            for scale in tile[512..].as_chunks_mut::<2>().0 {
                *scale = [0, 0x3c];
            }
        }
        let activation = [[1.0; 128]; 8];
        for case in 0..5 {
            {
                let mut state = STATE.lock().map_err(|_| Error::Contract)?;
                state.writes = 0;
                state.sequence = 0;
                state.batch_operations = 1;
                state.write_error = case == 1;
                state.stale_reply = case == 2;
                state.bad_numeric = case == 3;
                state.bad_member = None;
            }
            let limits = HmxQ2ProjectionLimits::new(32, 24_000, Duration::from_secs(4))?;
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
            report.queue.hmx_bitplane = true;
            report.operations_submitted = 23_998;
            report.operations_completed = 23_998;
            report.queue_writes_submitted = 23_998;
            report.queue_writes_completed = 23_998;
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
                limits: limits.common,
                mode: Mode::Hmx,
                resident_weights: HmxQ2WeightRegistry::new(0)?,
                native_resident_weights: NativeQ2WeightRegistry::new(0)?,
                work_deadline: if case == 4 {
                    Instant::now()
                } else {
                    Instant::now() + Duration::from_secs(1)
                },
                progress: &mut progress,
                failed: false,
            };
            let mut output = [99.0; 256];
            assert_eq!(HmxQ2Submitter::remaining_operations(&session), Some(2));
            let first = HmxQ2Submitter::submit(&mut session, &weights, &activation, &mut output);
            if case == 0 {
                assert!(first.is_ok());
                assert_eq!(HmxQ2Submitter::remaining_operations(&session), Some(1));
                assert_eq!(output, [7.0; 256]);
                assert!(
                    HmxQ2Submitter::submit(&mut session, &weights, &activation, &mut output)
                        .is_ok()
                );
                assert_eq!(session.report.operations_completed, 24_000);
                assert_eq!(session.report.operations_submitted, 24_000);
                assert_eq!(HmxQ2Submitter::remaining_operations(&session), Some(0));
                assert!(
                    HmxQ2Submitter::submit(&mut session, &weights, &activation, &mut output)
                        .is_err()
                );
                assert_eq!(STATE.lock().map_err(|_| Error::Contract)?.writes, 2);
            } else {
                assert!(first.is_err());
                assert_eq!(HmxQ2Submitter::remaining_operations(&session), Some(0));
                assert!(output.iter().all(|value| value.is_nan()));
                assert!(
                    HmxQ2Submitter::submit(&mut session, &weights, &activation, &mut output)
                        .is_err()
                );
                assert_eq!(session.report.operations_completed, 23_998);
                assert_eq!(
                    session.report.operations_submitted,
                    23_998 + u32::from(case != 4)
                );
                assert_eq!(
                    STATE.lock().map_err(|_| Error::Contract)?.writes,
                    u32::from(case != 4)
                );
            }
            assert!(session.report.is_hmx_bitplane_control());
            assert!(!session.report.queue.is_guarded_q2_control());
            assert!(session.report.queue.is_nonblocking_write_control());
            assert!(session.report.queue.poisoned);
            assert_eq!(session.report.queue.queue_stop_result, None);
            assert_eq!(session.report.queue.unmapping_result, None);
            assert!(!session.report.completed_cycle());
        }
        Ok(())
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "Exercise one complete native owner across independent queue and identity failure cases"
    )]
    fn resident_hmx_reuses_immutable_arena_and_never_reuses_a_faulted_owner() -> Result<(), Error> {
        let _serial = SERIAL.lock().map_err(|_| Error::Contract)?;
        let mut weights = vec![0x55; 1152];
        for tile in weights.as_chunks_mut::<576>().0 {
            for scale in tile[512..].as_chunks_mut::<2>().0 {
                *scale = [0, 0x3c];
            }
        }
        let activation = [[1.0; 128]; 8];
        for case in 0..6 {
            {
                let mut state = STATE.lock().map_err(|_| Error::Contract)?;
                state.writes = 0;
                state.sequence = 0;
                state.batch_operations = 1;
                state.write_error = case == 1;
                state.stale_reply = case == 2;
                state.bad_numeric = case == 3;
                state.bad_member = None;
            }
            let limits = HmxQ2ProjectionLimits::new(32, 3, Duration::from_secs(4))?
                .with_resident_weights(2304)?;
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
            report.queue.hmx_bitplane = true;
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
                limits: limits.common,
                mode: Mode::Hmx,
                resident_weights: HmxQ2WeightRegistry::new(limits.resident_weight_bytes())?,
                native_resident_weights: NativeQ2WeightRegistry::new(0)?,
                work_deadline: if case == 4 {
                    Instant::now()
                } else {
                    Instant::now() + Duration::from_secs(1)
                },
                progress: &mut progress,
                failed: false,
            };
            if case == 4 {
                assert!(HmxQ2Submitter::prepare_weights(&mut session, &weights).is_err());
                assert_eq!(HmxQ2Submitter::remaining_operations(&session), Some(0));
                assert_eq!(HmxQ2Submitter::residency_capacity_bytes(&session), 0);
                assert_eq!(STATE.lock().map_err(|_| Error::Contract)?.writes, 0);
                continue;
            }
            let first =
                HmxQ2Submitter::prepare_weights(&mut session, &weights)?.ok_or(Error::Contract)?;
            let second =
                HmxQ2Submitter::prepare_weights(&mut session, &weights)?.ok_or(Error::Contract)?;
            assert!(HmxQ2Submitter::prepare_weights(&mut session, &weights)?.is_none());
            assert_eq!(HmxQ2Submitter::remaining_operations(&session), Some(3));
            assert_eq!(HmxQ2Submitter::residency_capacity_bytes(&session), 0);
            let arena = hmx_projection::layout(32).allocation;
            assert_eq!(&memory[arena..arena + 1152], &weights);
            assert_eq!(&memory[arena + 1152..arena + 2304], &weights);
            let mut foreign = HmxQ2WeightRegistry::new(1152)?;
            let foreign = foreign.prepare(&weights)?.ok_or(Error::Contract)?;
            let handle = if case == 5 { &foreign } else { &first };
            let mut output = [99.0; 256];
            let result =
                HmxQ2Submitter::submit_resident(&mut session, handle, &activation, &mut output);
            if case == 0 {
                assert!(result.is_ok());
                assert_eq!(output, [7.0; 256]);
                HmxQ2Submitter::submit(&mut session, &weights, &activation, &mut output)?;
                HmxQ2Submitter::submit_resident(&mut session, &second, &activation, &mut output)?;
                assert_eq!(session.report.operations_completed, 3);
                assert_eq!(&memory[arena..arena + 1152], &weights);
                assert_eq!(&memory[arena + 1152..arena + 2304], &weights);
                assert_eq!(STATE.lock().map_err(|_| Error::Contract)?.writes, 3);
                assert!(
                    HmxQ2Submitter::submit_resident(&mut session, &first, &activation, &mut output)
                        .is_err()
                );
            } else {
                assert!(result.is_err());
                assert!(output.iter().all(|value| value.is_nan()));
                assert_eq!(session.report.operations_completed, 0);
                assert!(
                    HmxQ2Submitter::submit_resident(&mut session, &first, &activation, &mut output)
                        .is_err()
                );
                assert!(HmxQ2Submitter::prepare_weights(&mut session, &weights).is_err());
                assert_eq!(
                    STATE.lock().map_err(|_| Error::Contract)?.writes,
                    u32::from(case != 5)
                );
            }
            assert!(session.failed);
            assert_eq!(HmxQ2Submitter::remaining_operations(&session), Some(0));
            assert!(session.report.queue.poisoned);
            assert_eq!(session.report.queue.queue_stop_result, None);
            assert_eq!(session.report.queue.unmapping_result, None);
            assert!(!session.report.completed_cycle());
        }
        Ok(())
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "Exercise batch-four streaming, resident, capacity, and per-member failure cases on one mock queue"
    )]
    fn batch4_hmx_session_checks_all_twelve_tensors_four_outputs_and_capacity() -> Result<(), Error>
    {
        let _serial = SERIAL.lock().map_err(|_| Error::Contract)?;
        let mut weights = vec![0x55; 1152];
        for tile in weights.as_chunks_mut::<576>().0 {
            for scale in tile[512..].as_chunks_mut::<2>().0 {
                *scale = [0, 0x3c];
            }
        }
        let activation = [[1.0; 128]; 8];
        for case in 0..8 {
            {
                let mut state = STATE.lock().map_err(|_| Error::Contract)?;
                state.writes = 0;
                state.sequence = 0;
                state.batch_operations = 4;
                state.write_error = case == 1;
                state.stale_reply = case == 2;
                state.bad_numeric = false;
                state.bad_member = if (3..7).contains(&case) {
                    Some(case - 3)
                } else {
                    None
                };
            }
            let limits = HmxQ2ProjectionLimits::new(32, 24_000, Duration::from_secs(4))?
                .with_batch_four()?
                .with_resident_weights(4 * 1152)?;
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
            report.queue.hmx_bitplane = true;
            report.batch_operations = 4;
            report.operations_submitted = if case == 7 { 23_998 } else { 23_992 };
            report.operations_completed = report.operations_submitted;
            report.queue_writes_submitted = report.operations_submitted / 4;
            report.queue_writes_completed = report.operations_completed / 4;
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
                limits: limits.common,
                mode: Mode::HmxBatch4,
                resident_weights: HmxQ2WeightRegistry::new(limits.resident_weight_bytes())?,
                native_resident_weights: NativeQ2WeightRegistry::new(0)?,
                work_deadline: Instant::now() + Duration::from_secs(1),
                progress: &mut progress,
                failed: false,
            };
            let first =
                HmxQ2Submitter::prepare_weights(&mut session, &weights)?.ok_or(Error::Contract)?;
            let second =
                HmxQ2Submitter::prepare_weights(&mut session, &weights)?.ok_or(Error::Contract)?;
            let third =
                HmxQ2Submitter::prepare_weights(&mut session, &weights)?.ok_or(Error::Contract)?;
            let fourth =
                HmxQ2Submitter::prepare_weights(&mut session, &weights)?.ok_or(Error::Contract)?;
            let mut out0 = [99.0; 256];
            let mut out1 = [99.0; 256];
            let mut out2 = [99.0; 256];
            let mut out3 = [99.0; 256];
            let res = HmxQ2Submitter::submit_batch4(
                &mut session,
                [&weights, &weights, &weights, &weights],
                [&activation, &activation, &activation, &activation],
                [&mut out0, &mut out1, &mut out2, &mut out3],
            );
            if case == 0 {
                assert!(res.is_ok());
                assert_eq!(out0, [7.0; 256]);
                assert_eq!(out1, [8.0; 256]);
                assert_eq!(out2, [9.0; 256]);
                assert_eq!(out3, [10.0; 256]);
                assert_eq!(session.report.operations_submitted, 23_996);
                assert_eq!(session.report.operations_completed, 23_996);
                assert_eq!(session.report.queue_writes_submitted, 5999);
                assert_eq!(session.report.queue_writes_completed, 5999);
                let res2 = HmxQ2Submitter::submit_resident_batch4(
                    &mut session,
                    [&first, &second, &third, &fourth],
                    [&activation, &activation, &activation, &activation],
                    [&mut out0, &mut out1, &mut out2, &mut out3],
                );
                assert!(res2.is_ok());
                assert_eq!(session.report.operations_submitted, 24_000);
                assert_eq!(session.report.operations_completed, 24_000);
                assert_eq!(session.report.queue_writes_submitted, 6000);
                assert_eq!(session.report.queue_writes_completed, 6000);
                assert_eq!(HmxQ2Submitter::remaining_operations(&session), Some(0));
                assert!(
                    HmxQ2Submitter::submit_resident_batch4(
                        &mut session,
                        [&first, &second, &third, &fourth],
                        [&activation, &activation, &activation, &activation],
                        [&mut out0, &mut out1, &mut out2, &mut out3],
                    )
                    .is_err()
                );
                assert_eq!(STATE.lock().map_err(|_| Error::Contract)?.writes, 2);
            } else {
                assert!(res.is_err());
                assert!(out0.iter().all(|value| value.is_nan()));
                assert!(out1.iter().all(|value| value.is_nan()));
                assert!(out2.iter().all(|value| value.is_nan()));
                assert!(out3.iter().all(|value| value.is_nan()));
                assert_eq!(HmxQ2Submitter::remaining_operations(&session), Some(0));
                assert_eq!(
                    STATE.lock().map_err(|_| Error::Contract)?.writes,
                    u32::from(case != 7)
                );
            }
        }
        Ok(())
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "Exercise batch-ten streaming, resident, capacity, and per-member failure cases on one mock queue"
    )]
    fn batch10_hmx_session_checks_all_thirty_tensors_ten_outputs_and_capacity() -> Result<(), Error>
    {
        let _serial = SERIAL.lock().map_err(|_| Error::Contract)?;
        let mut weights = vec![0x55; 1152];
        for tile in weights.as_chunks_mut::<576>().0 {
            for scale in tile[512..].as_chunks_mut::<2>().0 {
                *scale = [0, 0x3c];
            }
        }
        let activation = [[1.0; 128]; 8];
        for case in 0..4 {
            {
                let mut state = STATE.lock().map_err(|_| Error::Contract)?;
                state.writes = 0;
                state.sequence = 0;
                state.batch_operations = 10;
                state.write_error = case == 1;
                state.stale_reply = false;
                state.bad_numeric = false;
                state.bad_member = if case == 2 { Some(9) } else { None };
            }
            let limits = HmxQ2ProjectionLimits::new(32, 24_000, Duration::from_secs(4))?
                .with_batch_ten()?
                .with_resident_weights(10 * 1152)?;
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
            report.queue.hmx_bitplane = true;
            report.batch_operations = 10;
            report.operations_submitted = if case == 3 { 23_995 } else { 23_980 };
            report.operations_completed = report.operations_submitted;
            report.queue_writes_submitted = report.operations_submitted / 10;
            report.queue_writes_completed = report.operations_completed / 10;
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
                limits: limits.common,
                mode: Mode::HmxBatch10,
                resident_weights: HmxQ2WeightRegistry::new(limits.resident_weight_bytes())?,
                native_resident_weights: NativeQ2WeightRegistry::new(0)?,
                work_deadline: Instant::now() + Duration::from_secs(1),
                progress: &mut progress,
                failed: false,
            };
            let mut handles = Vec::new();
            for _ in 0..10 {
                handles.push(
                    HmxQ2Submitter::prepare_weights(&mut session, &weights)?
                        .ok_or(Error::Contract)?,
                );
            }
            let handle_refs: [&HmxQ2WeightHandle; 10] =
                std::array::from_fn(|member| &handles[member]);
            let mut outs = [[99.0_f32; 256]; 10];
            let res = {
                let out_refs: [&mut [f32]; 10] = outs.each_mut().map(<[f32; 256]>::as_mut_slice);
                HmxQ2Submitter::submit_batch10(
                    &mut session,
                    [&weights; 10],
                    [&activation; 10],
                    out_refs,
                )
            };
            if case == 0 {
                assert!(res.is_ok());
                for (member, out) in outs.iter().enumerate() {
                    let step = u8::try_from(member).map_err(|_| Error::Contract)?;
                    assert_eq!(*out, [7.0 + f32::from(step); 256]);
                }
                assert_eq!(session.report.operations_submitted, 23_990);
                assert_eq!(session.report.operations_completed, 23_990);
                assert_eq!(session.report.queue_writes_submitted, 2399);
                assert_eq!(session.report.queue_writes_completed, 2399);
                let res2 = {
                    let out_refs: [&mut [f32]; 10] =
                        outs.each_mut().map(<[f32; 256]>::as_mut_slice);
                    HmxQ2Submitter::submit_resident_batch10(
                        &mut session,
                        handle_refs,
                        [&activation; 10],
                        out_refs,
                    )
                };
                assert!(res2.is_ok());
                assert_eq!(session.report.operations_submitted, 24_000);
                assert_eq!(session.report.operations_completed, 24_000);
                assert_eq!(session.report.queue_writes_submitted, 2400);
                assert_eq!(session.report.queue_writes_completed, 2400);
            } else {
                assert!(res.is_err());
                for out in &outs {
                    assert!(out.iter().all(|value| value.is_nan()));
                }
            }
            assert_eq!(HmxQ2Submitter::remaining_operations(&session), Some(0));
        }
        Ok(())
    }
}
