//! Fixed newer signed-package matrix controls and strict queue reply checks.
//!
//! All packet fields are encoded explicitly from ggml's pinned `htp-ops.h` at
//! `19cce6919992fdffd92e0ac152caaf1e2b8f254e`. This is an access diagnostic,
//! never a Swift arithmetic, performance, deployment or admission qualification.
#![allow(
    clippy::redundant_pub_crate,
    reason = "Private packet helpers remain accessible only to their safe parent and native sibling"
)]

use crate::SessionPreparationReport;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ComputeFixture {
    F32,
    F32NoBlock,
    GuardedQ2,
    GuardedQ2NoBlock,
}

impl ComputeFixture {
    pub(crate) const fn output_offset(self) -> usize {
        match self {
            Self::F32 | Self::F32NoBlock => OUTPUT_OFFSET,
            Self::GuardedQ2 | Self::GuardedQ2NoBlock => Q2_OUTPUT_OFFSET,
        }
    }
}

/// Signed-package protocol revision for the separate seven-argument start.
pub const COMPUTE_GGML_SOURCE: &str = "19cce6919992fdffd92e0ac152caaf1e2b8f254e";
/// Bounded CPU/DSP allocation shared by command, weights, activations and output.
pub const COMPUTE_ALLOCATION_BYTES: usize = 16_384;
/// Fixed inner dimension of the matrix access control.
pub const COMPUTE_K: usize = 32;
/// Fixed padded K of the separate guarded integer Q2 access control.
pub const COMPUTE_Q2_K: usize = 256;
/// Fixed output rows of the matrix access control.
pub const COMPUTE_M: usize = 32;
/// Two activation rows exercise the HVX matrix path with mixed signed outputs.
pub const COMPUTE_N: usize = 2;
/// Fixed scalar request extent from the pinned HTP operation-batch declaration.
pub const COMPUTE_REQUEST_BYTES: usize = 24;
/// Fixed reply extent from the pinned HTP operation-batch declaration.
pub const COMPUTE_RESPONSE_BYTES: usize = 88;

pub(crate) const WEIGHT_OFFSET: usize = 1024;
pub(crate) const ACTIVATION_OFFSET: usize = 5120;
pub(crate) const OUTPUT_OFFSET: usize = 5504;
pub(crate) const OUTPUT_BYTES: usize = COMPUTE_M * COMPUTE_N * 4;
const Q2_OUTPUT_OFFSET: usize = 7296;
const Q2_WEIGHT_BYTES: u16 = 2304;
const Q2_HOST_TILE_BYTES: usize = 576;
const Q2_VTCM_TILE_BYTES: u32 = 640;
const Q2_VTCM_BYTES: u32 = 61_440;
pub(crate) const START_SCALARS: u32 = 0x0201_0000;
pub(crate) const STOP_SCALARS: u32 = 0x0300_0000;

/// Independent observations of one consuming terminal queue operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "Keep independently completed native operations separate in evidence"
)]
pub struct QueueComputeReport {
    pub(crate) fixture: ComputeFixture,
    /// Last entered or completed fixed stage.
    pub stage: &'static str,
    /// Session metadata, including original validated effective domain.
    pub metadata: SessionPreparationReport,
    /// Reference initialization return; absent means not called.
    pub process_initialization_result: Option<i32>,
    /// Raw returned domain; never used as later configuration.
    pub process_initialization_domain_after: Option<i32>,
    /// Raw returned enable; never an effective process-class assertion.
    pub process_initialization_enable_after: Option<i32>,
    /// Explicit reference process-class request was attempted.
    pub unsigned_process_requested: bool,
    /// Fixed signed module open result.
    pub module_open_result: Option<i32>,
    /// Open replaced the poisoned 64-bit handle sentinel.
    pub handle_observed: bool,
    /// Fixed cached allocation succeeded with a usable descriptor.
    pub allocation_succeeded: bool,
    /// Explicit same-domain FD registration return.
    pub mapping_result: Option<i32>,
    /// Queue creation return.
    pub queue_create_result: Option<i32>,
    /// Queue export return.
    pub queue_export_result: Option<i32>,
    /// Seven-argument start invocation return.
    pub queue_start_result: Option<i32>,
    /// Fixed packet write return.
    pub queue_write_result: Option<i32>,
    /// Latest nonblocking read return, including would-block on expiry.
    pub queue_read_result: Option<i32>,
    /// A write was attempted, so its allocation must be treated as in use.
    pub compute_submitted: bool,
    /// Exact response, sequence, counts, descriptor and cache flags passed.
    pub response_validated: bool,
    /// Every poisoned output was replaced by its independent exact result.
    pub numeric_validated: bool,
    /// Stop completed after a fully validated reply and numeric output.
    pub queue_stop_result: Option<i32>,
    /// Host queue close after successful DSP stop.
    pub queue_close_result: Option<i32>,
    /// Host FD unregistration after successful stop and host queue close.
    pub unmapping_result: Option<i32>,
    /// Normal free happened only after known-safe cleanup.
    pub free_completed: bool,
    /// Explicit module handle close result.
    pub module_close_result: Option<i32>,
    /// All driver/module/catalog owners remain pinned through process exit.
    pub owner_pinned_until_exit: bool,
    /// Uncertain native consumers prevent allocation/module release.
    pub poisoned: bool,
}

impl QueueComputeReport {
    pub(crate) const fn new(metadata: SessionPreparationReport) -> Self {
        Self::new_fixture(metadata, ComputeFixture::F32)
    }

    pub(crate) const fn new_fixture(
        metadata: SessionPreparationReport,
        fixture: ComputeFixture,
    ) -> Self {
        Self {
            fixture,
            stage: "compute-metadata-incomplete",
            metadata,
            process_initialization_result: None,
            process_initialization_domain_after: None,
            process_initialization_enable_after: None,
            unsigned_process_requested: false,
            module_open_result: None,
            handle_observed: false,
            allocation_succeeded: false,
            mapping_result: None,
            queue_create_result: None,
            queue_export_result: None,
            queue_start_result: None,
            queue_write_result: None,
            queue_read_result: None,
            compute_submitted: false,
            response_validated: false,
            numeric_validated: false,
            queue_stop_result: None,
            queue_close_result: None,
            unmapping_result: None,
            free_completed: false,
            module_close_result: None,
            owner_pinned_until_exit: false,
            poisoned: false,
        }
    }

    /// This receipt belongs to the separate fixed guarded-Q2 integer control.
    #[must_use]
    pub const fn is_guarded_q2_control(self) -> bool {
        matches!(
            self.fixture,
            ComputeFixture::GuardedQ2 | ComputeFixture::GuardedQ2NoBlock
        )
    }

    /// This receipt selects exactly one direct nonblocking packet write.
    #[must_use]
    pub const fn is_nonblocking_write_control(self) -> bool {
        matches!(
            self.fixture,
            ComputeFixture::F32NoBlock | ComputeFixture::GuardedQ2NoBlock
        )
    }

    /// An exact matrix reply and all numeric outputs were independently checked.
    ///
    /// A write error can occur after a packet enters the queue. This observation
    /// may therefore be true while the original write result remains nonzero;
    /// it never authorizes cleanup, admission or a complete-cycle claim.
    #[must_use]
    pub const fn compute_executed(self) -> bool {
        self.compute_submitted
            && self.response_validated
            && self.numeric_validated
            && matches!(self.queue_read_result, Some(0))
    }

    /// All fixed access, numeric and explicit cleanup steps completed.
    ///
    /// This does not establish complete vendor-session teardown or eligibility.
    #[must_use]
    pub const fn completed_cycle(self) -> bool {
        self.metadata.metadata_prepared()
            && matches!(self.metadata.architecture, Some(81))
            && self.unsigned_process_requested
            && matches!(self.process_initialization_result, Some(0))
            && self.process_initialization_domain_after.is_some()
            && self.process_initialization_enable_after.is_some()
            && matches!(self.module_open_result, Some(0))
            && self.handle_observed
            && self.allocation_succeeded
            && matches!(self.mapping_result, Some(0))
            && matches!(self.queue_create_result, Some(0))
            && matches!(self.queue_export_result, Some(0))
            && matches!(self.queue_start_result, Some(0))
            && matches!(self.queue_write_result, Some(0))
            && self.compute_executed()
            && matches!(self.queue_stop_result, Some(0))
            && matches!(self.queue_close_result, Some(0))
            && matches!(self.unmapping_result, Some(0))
            && self.free_completed
            && matches!(self.module_close_result, Some(0))
            && self.owner_pinned_until_exit
            && !self.poisoned
    }

    // Shared by the native reply path and model-free failure-state controls.
    // True permits the caller to enter its explicit cleanup sequence, never an
    // inference/admission decision. Write errors preserve every uncertain owner
    // even after an independently validated numeric result has been observed.
    #[cfg(any(test, all(windows, target_arch = "aarch64")))]
    pub(crate) fn record_numeric_reply(&mut self, numeric_valid: bool) -> bool {
        self.numeric_validated =
            numeric_valid && self.response_validated && matches!(self.queue_read_result, Some(0));
        if !matches!(self.queue_write_result, Some(0)) {
            self.poisoned = true;
            self.stage = if self.numeric_validated {
                "compute-write-error-numeric-reply-pinned"
            } else {
                "compute-write-error-numeric-mismatch-pinned"
            };
            return false;
        }
        if !self.compute_executed() {
            self.stage = "compute-numeric-mismatch-pinned";
            return false;
        }
        true
    }
}

fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}
fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}
fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}
fn get_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(offset..offset + 4)?.try_into().ok()?,
    ))
}
fn get_u64(bytes: &[u8], offset: usize) -> Option<u64> {
    Some(u64::from_le_bytes(
        bytes.get(offset..offset + 8)?.try_into().ok()?,
    ))
}

pub(crate) fn start_wire(queue_id: u64) -> [u8; 48] {
    let mut bytes = [0; 48];
    // sess_id=0, one HVX, HMX disabled, default VTCM request, 64KiB mapping
    // budget and explicit balanced power mode. No inherited runtime override.
    put_u64(&mut bytes, 8, queue_id);
    put_u32(&mut bytes, 16, 1);
    put_u64(&mut bytes, 32, 65_536);
    put_u32(&mut bytes, 40, 4);
    bytes
}

pub(crate) fn request_wire() -> [u8; COMPUTE_REQUEST_BYTES] {
    let mut bytes = [0; COMPUTE_REQUEST_BYTES];
    put_u64(&mut bytes, 0, 1);
    put_u32(&mut bytes, 8, 1);
    put_u32(&mut bytes, 12, 3);
    put_u32(&mut bytes, 16, 1);
    bytes
}

fn tensor(bytes: &mut [u8], offset: usize, data: u64, size: u32, ti: u16, rows: u32, flags: u32) {
    put_u64(bytes, offset, data);
    put_u32(bytes, offset + 8, size);
    put_u32(bytes, offset + 12, flags);
    put_u16(bytes, offset + 22, ti);
    for (i, value) in [32, rows, 1, 1].into_iter().enumerate() {
        put_u32(bytes, offset + 24 + i * 4, value);
    }
    for (i, value) in [4, 128, rows * 128, rows * 128].into_iter().enumerate() {
        put_u32(bytes, offset + 40 + i * 4, value);
    }
}

/// Construct the only supported packet image before the device can see it.
pub(crate) fn fixture_image(base: u64, fd: u32) -> [u8; COMPUTE_ALLOCATION_BYTES] {
    const OP: usize = 192;
    const KP: usize = OP + 72;
    let mut bytes = [0; COMPUTE_ALLOCATION_BYTES];
    // One htp_buf_desc (24), three htp_tensor (56), one htp_op_desc
    // (232), one reserved htp_prof_desc (48), total minimum command 472.
    put_u64(&mut bytes, 0, base);
    put_u64(&mut bytes, 8, 16_384);
    put_u32(&mut bytes, 20, fd);
    tensor(&mut bytes, 24, 1024, 4096, 0, 32, 1);
    tensor(&mut bytes, 80, 5120, 256, 1, 2, 0);
    tensor(&mut bytes, 136, 5504, 256, 2, 2, 0);
    put_u32(&mut bytes, OP, 4); // HTP_OP_MUL_MAT, no stub or fusion flags.
    bytes[KP] = 4; // HTP_MM_KERNEL_HVX_F32_F32_VTCM.
    bytes[KP + 4] = 1; // one compute thread.
    bytes[KP + 6] = 16; // prefetch rows; source-derived F32 solver choice.
    put_u32(&mut bytes, KP + 24, 128); // activation row.
    put_u32(&mut bytes, KP + 28, 2304); // 256 activation + 2048 weight scratch.
    // init_fastdiv_values(2)=(1,1), init_fastdiv_values(1)=(1,0).
    for (i, shift) in [1, 1, 0, 0, 0].into_iter().enumerate() {
        put_u32(&mut bytes, KP + 36 + i * 8, 1);
        put_u32(&mut bytes, KP + 40 + i * 8, shift);
    }
    for index in 0..10 {
        put_u16(&mut bytes, OP + 200 + index * 2, u16::MAX);
    }
    put_u16(&mut bytes, OP + 200, 0);
    put_u16(&mut bytes, OP + 202, 1);
    for index in 0..4 {
        put_u16(&mut bytes, OP + 220 + index * 2, u16::MAX);
    }
    put_u16(&mut bytes, OP + 220, 2);
    for index in 0_u16..32 {
        let index_usize = usize::from(index);
        let weight = f32::from(index + 1);
        put_u32(
            &mut bytes,
            WEIGHT_OFFSET + (index_usize * 32 + index_usize) * 4,
            weight.to_bits(),
        );
        put_u32(
            &mut bytes,
            ACTIVATION_OFFSET + index_usize * 4,
            weight.to_bits(),
        );
        let second = f32::from(i16::try_from(index * 2).unwrap_or_default() - 17);
        put_u32(
            &mut bytes,
            ACTIVATION_OFFSET + 128 + index_usize * 4,
            second.to_bits(),
        );
    }
    for index in 0..64 {
        put_u32(&mut bytes, OUTPUT_OFFSET + index * 4, 0x7fc0_ae51);
    }
    bytes
}

fn q2_code(lane: u16) -> i16 {
    // Three whole 31-lane runs force both guard signs and maximum FP16-exact
    // corrected sums (+/-1857); the tail supplies nonconstant mixed codes.
    match lane {
        0..=30 => -128,
        31..=61 => 127,
        62..=92 => -127,
        _ => i16::try_from((lane * 73 + 19) & 255).unwrap_or_default() - 128,
    }
}

fn q2_symbol(row: u16, lane: u16) -> u8 {
    match row {
        0 => 1, // Exact zero weights test offset cancellation, including guards.
        1 => 3, // The fourth symbol is +2, never a ternary-only interpretation.
        2 => 0,
        3 => {
            if q2_code(lane) < 0 {
                0
            } else {
                3
            }
        }
        4 => {
            if q2_code(lane) < 0 {
                3
            } else {
                0
            }
        }
        _ => u8::try_from(((lane / 4 + row) ^ (lane + row * row + lane / (row + 1))) & 3)
            .unwrap_or_default(),
    }
}

// Independently derived as sum((original_symbol-1)*original_code_limb) over
// original lanes 0..128, without guards, padding, tile packing or Q8_1 code.
// Odd results above 2048 reject an accidental FP16-output/HMX route.
const Q2_EXPECTED_DOTS: [i16; 64] = [
    0, -4490, 2245, 9406, -11651, -1043, -1393, -871, -976, -1444, -1179, -1918, -391, -1410,
    -1132, -635, -1175, -1258, -1239, -783, -1265, -1219, -1020, -867, -909, -1647, -779, -739,
    -1369, -1011, -1161, -1137, 0, -4496, 2248, 9523, -11771, -1040, -1398, -869, -979, -1455,
    -1177, -1935, -374, -1417, -1128, -623, -1186, -1273, -1243, -781, -1266, -1212, -1021, -871,
    -907, -1651, -782, -743, -1378, -1011, -1162, -1140,
];

pub(crate) fn q2_fixture_image(base: u64, fd: u32) -> [u8; COMPUTE_ALLOCATION_BYTES] {
    const OP: usize = 192;
    const KP: usize = OP + 72;
    let mut bytes = fixture_image(base, fd);
    // Remove the F32 fixture, preserving only common descriptor/op framing.
    bytes[WEIGHT_OFFSET..].fill(0);
    tensor(&mut bytes, 24, 1024, u32::from(Q2_WEIGHT_BYTES), 0, 32, 3);
    put_u32(&mut bytes, 24 + 16, 42); // HTP_TYPE_Q2_0, WEIGHT|TILED.
    for (index, value) in [256, 32, 1, 1].into_iter().enumerate() {
        put_u32(&mut bytes, 24 + 24 + index * 4, value);
    }
    for (index, value) in [18, 72, 2304, 2304].into_iter().enumerate() {
        put_u32(&mut bytes, 24 + 40 + index * 4, value);
    }
    tensor(&mut bytes, 80, 5120, 2048, 1, 2, 0);
    put_u32(&mut bytes, 80 + 24, 256);
    for (index, value) in [4, 1024, 2048, 2048].into_iter().enumerate() {
        put_u32(&mut bytes, 80 + 40 + index * 4, value);
    }
    tensor(&mut bytes, 136, 7296, 256, 2, 2, 0);
    bytes[KP] = 6; // HTP_MM_KERNEL_HVX_QUANT_ROW: N=2 >= one HVX thread.
    put_u32(&mut bytes, KP + 16, 576);
    put_u32(&mut bytes, KP + 20, Q2_VTCM_TILE_BYTES);
    put_u32(&mut bytes, KP + 24, 10_240); // (round_up(K,128)/32)*1280 Q8_1.
    // Pinned solver: 2*10240 activation + max(16*4*640 weight,2048 raw).
    // N>1 has no destination scratch; the raw activation stage overlaps weights.
    put_u32(&mut bytes, KP + 28, Q2_VTCM_BYTES);
    for row in 0_u16..32 {
        let row_index = usize::from(row);
        for block in 0..4 {
            let tile = WEIGHT_OFFSET + block * Q2_HOST_TILE_BYTES;
            // Every lane starts with symbol one: zero weight, including guards.
            for plane in 0..4 {
                let index = tile + plane * 128 + row_index * 4;
                bytes[index..index + 4].fill(0x55);
            }
            put_u16(&mut bytes, tile + 512 + row_index * 2, 0x3c00);
        }
        for lane in 0_u16..128 {
            let destination = lane / 31 * 32 + lane % 31;
            let local = destination % 64;
            let tile = WEIGHT_OFFSET + usize::from(destination) / 64 * Q2_HOST_TILE_BYTES;
            let index =
                tile + usize::from(local) / 16 * 128 + row_index * 4 + usize::from(local) % 4;
            let shift = local / 4 % 4 * 2;
            bytes[index] = (bytes[index] & !(3 << shift)) | q2_symbol(row, lane) << shift;
        }
    }
    let mut activation = [[0_i16; COMPUTE_Q2_K]; 2];
    for lane in 0_u16..128 {
        let destination = usize::from(lane / 31 * 32 + lane % 31);
        let code = q2_code(lane);
        activation[0][destination] = code / 2;
        activation[1][destination] = code - code / 2;
    }
    for (limb_index, limb) in activation.iter_mut().enumerate() {
        for group in limb.as_chunks_mut::<32>().0 {
            let sum: i16 = group[..31].iter().sum();
            group[31] = if sum >= 0 { -127 } else { 127 };
        }
        for (lane, value) in limb.iter().enumerate() {
            put_u32(
                &mut bytes,
                ACTIVATION_OFFSET + (limb_index * 256 + lane) * 4,
                f32::from(*value).to_bits(),
            );
        }
    }
    for index in 0..64 {
        put_u32(&mut bytes, Q2_OUTPUT_OFFSET + index * 4, 0x7fc0_ae51);
    }
    bytes
}

pub(crate) fn fixture_image_for(
    fixture: ComputeFixture,
    base: u64,
    fd: u32,
) -> [u8; COMPUTE_ALLOCATION_BYTES] {
    match fixture {
        ComputeFixture::F32 | ComputeFixture::F32NoBlock => fixture_image(base, fd),
        ComputeFixture::GuardedQ2 | ComputeFixture::GuardedQ2NoBlock => q2_fixture_image(base, fd),
    }
}

pub(crate) fn output_valid_for(fixture: ComputeFixture, bytes: &[u8]) -> bool {
    match fixture {
        ComputeFixture::F32 | ComputeFixture::F32NoBlock => output_valid(bytes),
        ComputeFixture::GuardedQ2 | ComputeFixture::GuardedQ2NoBlock => {
            bytes.len() == OUTPUT_BYTES
                && Q2_EXPECTED_DOTS
                    .iter()
                    .enumerate()
                    .all(|(index, &expected)| {
                        get_u32(bytes, index * 4) == Some(f32::from(expected).to_bits())
                    })
        }
    }
}

pub(crate) fn response_valid(bytes: &[u8]) -> bool {
    bytes.len() == COMPUTE_RESPONSE_BYTES
        && get_u64(bytes, 0) == Some(1)
        && get_u32(bytes, 24) == Some(1)
        && get_u32(bytes, 28) == Some(1)
        && get_u32(bytes, 32) == Some(3)
        && get_u32(bytes, 36) == Some(1)
        && bytes
            .get(44..88)
            .is_some_and(|traces| traces.iter().all(|byte| *byte == 0))
}

pub(crate) fn output_valid(bytes: &[u8]) -> bool {
    if bytes.len() != OUTPUT_BYTES {
        return false;
    }
    for index in 0_u16..32 {
        let index_usize = usize::from(index);
        let row = f32::from(index + 1);
        let second = f32::from(i16::try_from(index * 2).unwrap_or_default() - 17);
        if get_u32(bytes, index_usize * 4) != Some((row * row).to_bits())
            || get_u32(bytes, (32 + index_usize) * 4) != Some((row * second).to_bits())
        {
            return false;
        }
    }
    true
}

pub(crate) fn self_check() -> bool {
    let image = fixture_image(0x1234_5600, 19);
    let nonblocking_image = fixture_image_for(ComputeFixture::F32NoBlock, 0x1234_5600, 19);
    let start = start_wire(0x2132_4354_6576_0718);
    let mut response = [0; COMPUTE_RESPONSE_BYTES];
    put_u64(&mut response, 0, 1);
    for (offset, value) in [(24, 1), (28, 1), (32, 3), (36, 1)] {
        put_u32(&mut response, offset, value);
    }
    let q2_image = fixture_image_for(ComputeFixture::GuardedQ2, 0x1234_5600, 19);
    let q2_nonblocking_image = fixture_image_for(ComputeFixture::GuardedQ2NoBlock, 0x1234_5600, 19);
    let q2_offset = ComputeFixture::GuardedQ2.output_offset();
    image == nonblocking_image
        && q2_image == q2_nonblocking_image
        && response_valid(&response)
        && !output_valid(&image[OUTPUT_OFFSET..OUTPUT_OFFSET + OUTPUT_BYTES])
        && !output_valid_for(
            ComputeFixture::GuardedQ2,
            &q2_image[q2_offset..q2_offset + OUTPUT_BYTES],
        )
        && get_u32(&q2_image, 24 + 16) == Some(42)
        && get_u32(&q2_image, 264 + 24) == Some(10_240)
        && get_u32(&q2_image, 264 + 28) == Some(Q2_VTCM_BYTES)
        && request_wire()[..8] == 1_u64.to_le_bytes()
        && start[4..8] == [0; 4]
        && start[44..48] == [0; 4]
        && START_SCALARS == 0x0201_0000
        && STOP_SCALARS == 0x0300_0000
        && !QueueComputeReport::new(SessionPreparationReport::new(0x81)).completed_cycle()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn submitted_report(fixture: ComputeFixture, write_result: i32) -> QueueComputeReport {
        let mut metadata = SessionPreparationReport::new(0x81);
        metadata.domain_result = Some(0);
        metadata.effective_domain = Some(3);
        metadata.uri_result = Some(0);
        metadata.uri_verified = true;
        let mut report = QueueComputeReport::new_fixture(metadata, fixture);
        report.unsigned_process_requested = true;
        report.process_initialization_result = Some(0);
        report.process_initialization_domain_after = Some(3);
        report.process_initialization_enable_after = Some(1);
        report.module_open_result = Some(0);
        report.handle_observed = true;
        report.allocation_succeeded = true;
        report.mapping_result = Some(0);
        report.queue_create_result = Some(0);
        report.queue_export_result = Some(0);
        report.queue_start_result = Some(0);
        report.queue_write_result = Some(write_result);
        report.compute_submitted = true;
        report.queue_read_result = Some(0);
        report.response_validated = true;
        report.owner_pinned_until_exit = true;
        report.poisoned = true;
        report
    }

    #[test]
    fn numeric_reply_after_write_error_preserves_error_and_never_permits_cleanup() {
        for fixture in [
            ComputeFixture::F32,
            ComputeFixture::GuardedQ2,
            ComputeFixture::F32NoBlock,
            ComputeFixture::GuardedQ2NoBlock,
        ] {
            let mut report = submitted_report(fixture, 14);
            assert!(!report.record_numeric_reply(true));
            assert!(report.compute_executed());
            assert_eq!(report.queue_write_result, Some(14));
            assert_eq!(report.stage, "compute-write-error-numeric-reply-pinned");
            assert!(report.owner_pinned_until_exit && report.poisoned);
            assert_eq!(report.queue_stop_result, None);
            assert_eq!(report.queue_close_result, None);
            assert_eq!(report.module_close_result, None);
            assert_eq!(report.unmapping_result, None);
            assert!(!report.free_completed);
            assert!(!report.completed_cycle());
            // A stale or fabricated successful-cleanup receipt cannot mask the
            // original write error. The positive control isolates that gate.
            report.queue_stop_result = Some(0);
            report.queue_close_result = Some(0);
            report.module_close_result = Some(0);
            report.unmapping_result = Some(0);
            report.free_completed = true;
            report.poisoned = false;
            assert!(!report.completed_cycle());
            report.queue_write_result = Some(0);
            assert!(report.completed_cycle());
        }
    }

    #[test]
    fn failed_or_uncorrelated_reply_never_becomes_numeric_evidence_or_cleanup() {
        let mut report = submitted_report(ComputeFixture::F32NoBlock, 14);
        report.queue_read_result = Some(516);
        report.response_validated = false;
        assert!(!report.record_numeric_reply(true));
        assert!(!report.compute_executed());
        assert_eq!(report.queue_write_result, Some(14));
        assert!(report.poisoned && report.owner_pinned_until_exit);
        assert!(!report.free_completed);
        assert!(!report.completed_cycle());
        let mut successful_write = submitted_report(ComputeFixture::F32, 0);
        assert!(!successful_write.record_numeric_reply(false));
        assert!(!successful_write.compute_executed());
        assert!(!successful_write.free_completed);
        let mut unsubmitted = submitted_report(ComputeFixture::F32, 0);
        unsubmitted.compute_submitted = false;
        assert!(!unsubmitted.record_numeric_reply(true));
        assert!(!unsubmitted.compute_executed());
    }

    #[test]
    fn fixed_wire_shapes_padding_and_numeric_poison() {
        assert!(self_check());
        let image = fixture_image(0x1234_5600, 19);
        assert_eq!(get_u64(&image, 8), Some(16_384));
        assert_eq!(get_u32(&image, 20), Some(19));
        assert_eq!(get_u32(&image, 192), Some(4));
        assert_eq!(get_u32(&image, 292), Some(2304));
        assert_eq!(image[472..1024], [0; 552]);
        assert_eq!(
            fixture_image_for(ComputeFixture::F32NoBlock, 0x1234_5600, 19),
            image
        );
        assert!(
            !QueueComputeReport::new(SessionPreparationReport::new(0x81))
                .is_nonblocking_write_control()
        );
        assert!(
            QueueComputeReport::new_fixture(
                SessionPreparationReport::new(0x81),
                ComputeFixture::F32NoBlock
            )
            .is_nonblocking_write_control()
        );
    }

    #[test]
    fn response_rejects_every_correlated_field_change_and_wrong_extent() {
        let mut valid = [0; COMPUTE_RESPONSE_BYTES];
        put_u64(&mut valid, 0, 1);
        for (offset, value) in [(24, 1), (28, 1), (32, 3), (36, 1)] {
            put_u32(&mut valid, offset, value);
        }
        assert!(response_valid(&valid));
        for offset in [0, 7, 24, 28, 32, 36, 44, 87] {
            let mut wrong = valid;
            wrong[offset] ^= 0x80;
            assert!(!response_valid(&wrong));
        }
        assert!(!response_valid(&valid[..87]));
    }

    #[test]
    fn numeric_control_requires_all_sixty_four_exact_outputs() {
        let mut output = [0; OUTPUT_BYTES];
        for index in 0_u16..32 {
            let row = f32::from(index + 1);
            let second = f32::from(i16::try_from(index * 2).unwrap_or_default() - 17);
            put_u32(&mut output, usize::from(index) * 4, (row * row).to_bits());
            put_u32(
                &mut output,
                (32 + usize::from(index)) * 4,
                (row * second).to_bits(),
            );
        }
        assert!(output_valid(&output));
        for index in 0..64 {
            let mut wrong = output;
            put_u32(&mut wrong, index * 4, 0x7fc0_ae51);
            assert!(!output_valid(&wrong));
        }
    }

    #[test]
    fn guarded_q2_image_matches_original_integer_math_and_host_tile_strides()
    -> Result<(), crate::Error> {
        let image = fixture_image_for(ComputeFixture::GuardedQ2, 0x1234_5600, 19);
        assert_eq!(get_u32(&image, 24 + 8), Some(2304));
        assert_eq!(get_u32(&image, 24 + 12), Some(3));
        assert_eq!(get_u32(&image, 24 + 16), Some(42));
        assert_eq!(get_u32(&image, 24 + 24), Some(256));
        assert_eq!(get_u32(&image, 24 + 44), Some(72));
        assert_eq!(get_u32(&image, 264 + 16), Some(576));
        assert_eq!(get_u32(&image, 264 + 20), Some(640));
        assert_eq!(get_u32(&image, 264 + 24), Some(10_240));
        assert_eq!(get_u32(&image, 264 + 28), Some(61_440));
        assert_eq!(image[264], 6);
        assert_eq!(image[268], 1);
        assert_eq!(image[270], 16);
        // Last scale of host tile 0, then tile 1's row-zero symbols (zero weight).
        assert_eq!(&image[1024 + 574..1024 + 576], &[0x00, 0x3c]);
        assert_eq!(&image[1600..1604], &[0x55; 4]);
        for tile in image[WEIGHT_OFFSET..WEIGHT_OFFSET + usize::from(Q2_WEIGHT_BYTES)]
            .as_chunks::<576>()
            .0
        {
            for scale in tile[512..576].as_chunks::<2>().0 {
                assert_eq!(*scale, [0x00, 0x3c]);
            }
        }
        let mut output = [0; OUTPUT_BYTES];
        let mut observed_guard_sums = Vec::new();
        for limb in 0..2 {
            let mut values = [0_i16; COMPUTE_Q2_K];
            for (lane, value) in values.iter_mut().enumerate() {
                let bits = get_u32(&image, ACTIVATION_OFFSET + (limb * 256 + lane) * 4)
                    .ok_or(crate::Error::Contract)?;
                *value = (-127_i16..=127)
                    .find(|&integer| f32::from(integer).to_bits() == bits)
                    .ok_or(crate::Error::Contract)?;
            }
            for group in values.as_chunks::<32>().0 {
                assert_eq!(group.iter().map(|v| v.abs()).max(), Some(127));
                let sum: i16 = group.iter().sum();
                assert!(sum.abs() <= 1857);
                observed_guard_sums.push(sum);
            }
            for row in 0_u16..32 {
                let row_index = usize::from(row);
                let mut tiled_integer = 0_i32;
                for (lane, &value) in values.iter().enumerate() {
                    let tile = WEIGHT_OFFSET + lane / 64 * 576;
                    let local = lane % 64;
                    let packed = image[tile + local / 16 * 128 + row_index * 4 + local % 4];
                    let code = (packed >> (local / 4 % 4 * 2)) & 3;
                    tiled_integer += (i32::from(code) - 1) * i32::from(value);
                }
                let mut original_integer = 0_i32;
                for lane in 0_u16..128 {
                    let code = q2_code(lane);
                    let value = if limb == 0 { code / 2 } else { code - code / 2 };
                    original_integer += (i32::from(q2_symbol(row, lane)) - 1) * i32::from(value);
                }
                assert_eq!(tiled_integer, original_integer);
                assert_eq!(
                    original_integer,
                    i32::from(Q2_EXPECTED_DOTS[limb * 32 + row_index])
                );
                let integer = i16::try_from(tiled_integer).map_err(|_| crate::Error::Contract)?;
                put_u32(
                    &mut output,
                    (limb * 32 + row_index) * 4,
                    f32::from(integer).to_bits(),
                );
            }
        }
        assert!(observed_guard_sums.contains(&1857));
        assert!(observed_guard_sums.contains(&-1857));
        assert!(output_valid_for(ComputeFixture::GuardedQ2, &output));
        assert!(!output_valid_for(ComputeFixture::F32, &output));
        for index in 0..64 {
            let mut wrong = output;
            put_u32(&mut wrong, index * 4, 0x7fc0_ae51);
            assert!(!output_valid_for(ComputeFixture::GuardedQ2, &wrong));
        }
        assert!(!output_valid_for(
            ComputeFixture::GuardedQ2,
            &output[..OUTPUT_BYTES - 1]
        ));
        Ok(())
    }
}
