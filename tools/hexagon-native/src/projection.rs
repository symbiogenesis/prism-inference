//! Bounded signed-Q2 projection requests; no profile or model qualification.
#![allow(
    clippy::redundant_pub_crate,
    reason = "Private packet helpers are shared only with the audited native sibling"
)]
#![cfg_attr(
    not(all(windows, target_arch = "aarch64")),
    allow(
        dead_code,
        reason = "Packet implementation is used by the native Windows ARM64 boundary and hosted model-free controls"
    )
)]

use crate::{Error, QueueComputeReport, SessionPreparationReport, compute};
use std::time::Duration;

/// Largest weight-row extent of the independently bound Swift up projection.
pub const Q2_PROJECTION_MAX_ROWS: usize = 17_408;
/// Owner-authorized upper write-attempt bound shared by HVX and HMX sessions.
///
/// This admits one all-eligible original-PQ2 evaluation token (23,256 writes),
/// without extending session duration or establishing complete-request capacity.
pub const Q2_PROJECTION_MAX_OPERATIONS: u32 = 24_000;

/// One terminal session's fixed allocation, operation and absolute-time bounds.
#[derive(Clone, Copy, Debug)]
pub struct Q2ProjectionLimits {
    pub(crate) rows: usize,
    pub(crate) operations: u32,
    pub(crate) duration: Duration,
    pub(crate) persistent: bool,
}

impl Q2ProjectionLimits {
    /// Validate bounds before opening a remote module or allocating memory.
    ///
    /// Only 32-row multiples are admitted. K=256, N=2, one HVX thread and the
    /// pinned 61,440-byte quant-row solver stay fixed. The deadline covers
    /// initialization, packing and every operation, with three seconds reserved
    /// inside it for explicit cleanup. An external process deadline is required.
    ///
    /// # Errors
    /// Rejects unsupported rows, empty/excessive work or durations outside
    /// 4..=120 seconds. This does not establish available VTCM or device support.
    pub fn new(rows: usize, operations: u32, duration: Duration) -> Result<Self, Error> {
        if rows == 0
            || rows > Q2_PROJECTION_MAX_ROWS
            || !rows.is_multiple_of(32)
            || operations == 0
            || operations > Q2_PROJECTION_MAX_OPERATIONS
            || duration < Duration::from_secs(4)
            || duration > Duration::from_secs(120)
        {
            return Err(Error::Contract);
        }
        Ok(Self {
            rows,
            operations,
            duration,
            persistent: false,
        })
    }

    /// Maximum active weight rows in any operation.
    #[must_use]
    pub const fn maximum_rows(self) -> usize {
        self.rows
    }

    /// Maximum mapped host allocation, including command, inputs and output.
    #[must_use]
    pub const fn allocation_bytes(self) -> usize {
        layout(self.rows).allocation
    }

    /// Maximum writes across the complete session, with no implicit retry.
    #[must_use]
    pub const fn maximum_operations(self) -> u32 {
        self.operations
    }

    /// Complete absolute session allowance, including its cleanup reserve.
    #[must_use]
    pub const fn maximum_duration(self) -> Duration {
        self.duration
    }
}

/// Safe synchronous submission of one original-128-block integer operation.
pub trait GuardedQ2Submitter {
    /// Remaining writes in the actual borrowed native session, when known.
    ///
    /// Custom submitters returning `None` require independently matched limits.
    /// Native sessions include all earlier work, including previous executors.
    fn remaining_operations(&self) -> Option<u64> {
        None
    }

    /// Submit tightly packed 576-byte host tiles and two guarded F32 K=256 rows.
    ///
    /// `output` has two contiguous M-row limbs, M a multiple of 32. No pointer,
    /// operation, solver, map or power field is caller-controlled. Every result
    /// must be finite, integral and bounded by 16,384 before publication. The
    /// trusted caller must additionally compare exact dots with an independent
    /// oracle: integer-shaped wrong results do not qualify numerical parity.
    ///
    /// # Errors
    /// Rejects malformed inputs, exhausted bounds, refusals and uncorrelated or
    /// invalid outputs. A failed session cannot submit again or release uncertain
    /// native consumers. No partial output is published by this method.
    fn submit(
        &mut self,
        weights: &[u8],
        activation: &[[f32; 256]; 2],
        output: &mut [f32],
    ) -> Result<(), Error>;
}

/// Separate observations of native lifetime and complete caller-owned work.
#[derive(Clone, Copy, Debug)]
pub struct Q2ProjectionReport {
    /// Exact native setup, latest queue operation and explicit cleanup receipts.
    pub queue: QueueComputeReport,
    /// Logical operations carried in each queue packet (1, 4 or 10).
    pub batch_operations: u32,
    /// Number of queue packet writes attempted, including a refused/uncertain final write.
    pub queue_writes_submitted: u32,
    /// Correlated queue packet replies validated synchronously.
    pub queue_writes_completed: u32,
    /// Number of logical operations attempted, including a refused/uncertain final write.
    pub operations_submitted: u32,
    /// Correlated, integer-validated operations returned synchronously.
    pub operations_completed: u32,
    /// Successful mapped allocation extent also covered by each queue descriptor.
    /// This is not a measured cache-maintenance or bus-transfer byte count.
    pub mapped_allocation_bytes: usize,
    /// Immutable packed payload copied into the host arena, excluding padding.
    pub resident_weight_bytes_prepared: usize,
    /// Successfully copied immutable weight extents; preparation submits no kernel.
    pub resident_weight_preparations: u32,
    /// Attempted kernel writes using immutable arena handles, including failed writes.
    pub resident_submissions: u32,
    /// Caller completed the entire projection and independently accepted output.
    pub work_completed: bool,
    /// The one absolute session/work deadline expired.
    pub deadline_expired: bool,
}

impl Q2ProjectionReport {
    pub(crate) const fn new(metadata: SessionPreparationReport) -> Self {
        Self {
            queue: QueueComputeReport::new_fixture(
                metadata,
                compute::ComputeFixture::GuardedQ2NoBlock,
            ),
            batch_operations: 1,
            queue_writes_submitted: 0,
            queue_writes_completed: 0,
            operations_submitted: 0,
            operations_completed: 0,
            mapped_allocation_bytes: 0,
            resident_weight_bytes_prepared: 0,
            resident_weight_preparations: 0,
            resident_submissions: 0,
            work_completed: false,
            deadline_expired: false,
        }
    }

    /// Whether this terminal report belongs to the distinct HMX bitplane path.
    /// This distinguishes requested kernels, without proving effective placement.
    #[must_use]
    pub const fn is_hmx_bitplane_control(self) -> bool {
        self.queue.is_hmx_bitplane_control()
    }

    /// All requested work and explicit cleanup passed; no eligibility follows.
    #[must_use]
    pub const fn completed_cycle(self) -> bool {
        self.work_completed
            && !self.deadline_expired
            && matches!(self.batch_operations, 1 | 4 | 10)
            && self.queue_writes_submitted > 0
            && self.queue_writes_submitted == self.queue_writes_completed
            && self.operations_submitted > 0
            && self.operations_submitted == self.operations_completed
            && (self.operations_submitted
                == self
                    .queue_writes_submitted
                    .saturating_mul(self.batch_operations)
                || (!self.queue.hmx_bitplane
                    && self.batch_operations == 1
                    && self.resident_submissions > 0
                    && self.operations_submitted >= self.queue_writes_submitted
                    && self.operations_submitted <= self.queue_writes_submitted.saturating_mul(3)))
            && self.queue.completed_cycle()
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Layout {
    pub(crate) weights: usize,
    pub(crate) activation: usize,
    pub(crate) output: usize,
    pub(crate) allocation: usize,
}

pub(crate) const fn layout(rows: usize) -> Layout {
    let weights = rows / 32 * 2304;
    let activation = (1024 + weights).next_multiple_of(128);
    let output = activation + 2048;
    Layout {
        weights,
        activation,
        output,
        allocation: (output + rows * 8).next_multiple_of(4096),
    }
}

pub(crate) fn request(sequence: u64) -> [u8; compute::COMPUTE_REQUEST_BYTES] {
    let mut result = compute::request_wire();
    result[..8].copy_from_slice(&sequence.to_le_bytes());
    result
}

pub(crate) fn request_batch(
    sequence: u64,
    batch_operations: u32,
) -> Result<[u8; compute::COMPUTE_REQUEST_BYTES], Error> {
    if !matches!(batch_operations, 1 | 2 | 3 | 4 | 10) {
        return Err(Error::Contract);
    }
    let mut result = request(sequence);
    put32(&mut result, 12, batch_operations * 3);
    put32(&mut result, 16, batch_operations);
    Ok(result)
}

#[cfg(test)]
pub(crate) fn response_valid(bytes: &[u8], sequence: u64) -> bool {
    response_batch_valid(bytes, sequence, 1)
}

pub(crate) fn response_batch_valid(bytes: &[u8], sequence: u64, batch_operations: u32) -> bool {
    if !matches!(batch_operations, 1 | 2 | 3 | 4 | 10)
        || sequence == 0
        || bytes.len() != compute::COMPUTE_RESPONSE_BYTES
    {
        return false;
    }
    let mut fixed = [0; compute::COMPUTE_RESPONSE_BYTES];
    fixed.copy_from_slice(bytes);
    let correlated = fixed[..8] == sequence.to_le_bytes()
        && fixed[32..36] == (batch_operations * 3).to_le_bytes()
        && fixed[36..40] == batch_operations.to_le_bytes();
    fixed[..8].copy_from_slice(&1_u64.to_le_bytes());
    fixed[32..36].copy_from_slice(&3_u32.to_le_bytes());
    fixed[36..40].copy_from_slice(&1_u32.to_le_bytes());
    correlated && compute::response_valid(&fixed)
}

pub(crate) fn validate_inputs(
    maximum_rows: usize,
    weights: &[u8],
    activation: &[[f32; 256]; 2],
    output_count: usize,
) -> Result<usize, Error> {
    let rows = output_count / 2;
    if output_count != rows * 2
        || rows == 0
        || rows > maximum_rows
        || !rows.is_multiple_of(32)
        || weights.len() != layout(rows).weights
    {
        return Err(Error::Contract);
    }
    for limb in activation {
        for group in limb.as_chunks::<32>().0 {
            let mut sum = 0.0;
            for &value in &group[..31] {
                if !value.is_finite() || value.fract() != 0.0 || value.abs() > 64.0 {
                    return Err(Error::Contract);
                }
                sum += value;
            }
            let expected: f32 = if sum >= 0.0 { -127.0 } else { 127.0 };
            if group[31].to_bits() != expected.to_bits() {
                return Err(Error::Contract);
            }
        }
    }
    // Every real row is scale-one; guard lanes encode zero weight (symbol one).
    for tile in weights.as_chunks::<576>().0 {
        if tile[512..]
            .as_chunks::<2>()
            .0
            .iter()
            .any(|scale| *scale != [0, 0x3c])
        {
            return Err(Error::Contract);
        }
        for row in 0..32 {
            for lane in [31, 63] {
                let byte = tile[lane / 16 * 128 + row * 4 + lane % 4];
                if (byte >> (lane / 4 % 4 * 2)) & 3 != 1 {
                    return Err(Error::Contract);
                }
            }
        }
    }
    Ok(rows)
}

pub(crate) fn image(
    bytes: &mut [u8],
    base: u64,
    fd: u32,
    rows: usize,
    weights: &[u8],
    activation: &[[f32; 256]; 2],
) -> Result<Layout, Error> {
    let active = layout(rows);
    if bytes.len() < active.allocation || bytes.len() > layout(Q2_PROJECTION_MAX_ROWS).allocation {
        return Err(Error::Contract);
    }
    bytes.fill(0);
    let fixed = compute::q2_framing_header(base, fd);
    bytes[..1024].copy_from_slice(&fixed);
    put64(
        bytes,
        8,
        u64::try_from(bytes.len()).map_err(|_| Error::Contract)?,
    );
    for (offset, value) in [
        (32, active.weights),
        (52, rows),
        (72, active.weights),
        (76, active.weights),
        (144, rows * 8),
        (160, rows),
        (180, rows * 4),
        (184, rows * 8),
        (188, rows * 8),
    ] {
        put32(
            bytes,
            offset,
            u32::try_from(value).map_err(|_| Error::Contract)?,
        );
    }
    put64(
        bytes,
        80,
        u64::try_from(active.activation).map_err(|_| Error::Contract)?,
    );
    put64(
        bytes,
        136,
        u64::try_from(active.output).map_err(|_| Error::Contract)?,
    );
    bytes[1024..1024 + weights.len()].copy_from_slice(weights);
    for (index, value) in activation.iter().flatten().enumerate() {
        put32(bytes, active.activation + index * 4, value.to_bits());
    }
    for index in 0..rows * 2 {
        put32(bytes, active.output + index * 4, 0x7fc0_ae51);
    }
    Ok(active)
}

fn put32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}
fn put64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

pub(crate) fn output_valid(bytes: &[u8]) -> bool {
    bytes.len().is_multiple_of(4)
        && bytes.as_chunks::<4>().0.iter().all(|value| {
            let value = f32::from_le_bytes(*value);
            value.is_finite() && value.fract() == 0.0 && value.abs() <= 16_384.0
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limits_admit_full_m_without_changing_vtcm_or_accepting_tails() -> Result<(), Error> {
        let limits = Q2ProjectionLimits::new(17_408, 40, Duration::from_secs(20))?;
        assert_eq!(limits.allocation_bytes(), 1_396_736);
        let maximum = Q2ProjectionLimits::new(17_408, 24_000, Duration::from_secs(120))?;
        assert_eq!(maximum.maximum_operations(), 24_000);
        assert_eq!(maximum.maximum_duration(), Duration::from_secs(120));
        assert_eq!(maximum.allocation_bytes(), limits.allocation_bytes());
        for (rows, operations, seconds) in [
            (0, 40, 20),
            (33, 40, 20),
            (17_440, 40, 20),
            (32, 0, 20),
            (32, 24_001, 20),
            (32, 1, 3),
            (32, 1, 121),
        ] {
            assert!(
                Q2ProjectionLimits::new(rows, operations, Duration::from_secs(seconds)).is_err()
            );
        }
        Ok(())
    }

    #[test]
    fn sequence_validation_never_accepts_stale_reply() -> Result<(), Error> {
        let mut bytes = [0; compute::COMPUTE_RESPONSE_BYTES];
        bytes[..8].copy_from_slice(&42_u64.to_le_bytes());
        for (offset, value) in [(24, 1_u32), (28, 1), (32, 3), (36, 1)] {
            put32(&mut bytes, offset, value);
        }
        assert!(response_valid(&bytes, 42));
        assert!(!response_valid(&bytes, 41));
        assert!(!response_valid(&bytes, 0));
        assert!(!response_batch_valid(&bytes, 42, 4));
        assert!(!response_batch_valid(&bytes, 42, 2));
        assert_eq!(&request(u64::MAX)[..8], &u64::MAX.to_le_bytes());
        let batch_req = request_batch(7, 4)?;
        assert_eq!(&batch_req[..8], &7_u64.to_le_bytes());
        assert_eq!(
            &batch_req[8..24],
            &[1, 0, 0, 0, 12, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0]
        );
        assert!(request_batch(7, 5).is_err());
        let two_op_req = request_batch(7, 2)?;
        assert_eq!(
            &two_op_req[8..24],
            &[1, 0, 0, 0, 6, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]
        );
        let three_op_req = request_batch(7, 3)?;
        assert_eq!(
            &three_op_req[8..24],
            &[1, 0, 0, 0, 9, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0]
        );
        put32(&mut bytes, 32, 12);
        put32(&mut bytes, 36, 4);
        assert!(response_batch_valid(&bytes, 42, 4));
        assert!(!response_batch_valid(&bytes, 41, 4));
        assert!(!response_valid(&bytes, 42));
        let batch10_req = request_batch(9, 10)?;
        assert_eq!(&batch10_req[..8], &9_u64.to_le_bytes());
        assert_eq!(
            &batch10_req[8..24],
            &[1, 0, 0, 0, 30, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0]
        );
        put32(&mut bytes, 32, 30);
        put32(&mut bytes, 36, 10);
        assert!(response_batch_valid(&bytes, 42, 10));
        assert!(!response_batch_valid(&bytes, 42, 4));
        Ok(())
    }

    #[test]
    fn dynamic_frame_keeps_known_m32_bytes_and_full_m_strides() -> Result<(), Error> {
        let fixed = compute::q2_fixture_image(0x1234_5600, 19);
        let mut activation = [[0.0; 256]; 2];
        for (index, value) in activation.iter_mut().flatten().enumerate() {
            let start = 5120 + index * 4;
            *value = f32::from_le_bytes(
                fixed[start..start + 4]
                    .try_into()
                    .map_err(|_| Error::Contract)?,
            );
        }
        let weights = &fixed[1024..3328];
        assert_eq!(validate_inputs(32, weights, &activation, 64)?, 32);
        let mut bytes = [0; compute::COMPUTE_ALLOCATION_BYTES];
        let active = image(&mut bytes, 0x1234_5600, 19, 32, weights, &activation)?;
        // The dynamic compact placement is separate from the known fixed gaps.
        assert_eq!(&bytes[..80], &fixed[..80]);
        assert_eq!(&bytes[192..472], &fixed[192..472]);
        assert_eq!(
            &bytes[active.activation..active.activation + 2048],
            &fixed[5120..7168]
        );
        assert!(!output_valid(&bytes[active.output..active.output + 256]));
        let mut wrong = activation;
        wrong[0][31] = 0.0;
        assert!(validate_inputs(32, weights, &wrong, 64).is_err());
        let mut wrong_weights = weights.to_vec();
        wrong_weights[512] = 1;
        assert!(validate_inputs(32, &wrong_weights, &activation, 64).is_err());
        let full_rows = Q2_PROJECTION_MAX_ROWS;
        let full_layout = layout(full_rows);
        let mut full = vec![0; full_layout.allocation];
        let full_weights = weights.repeat(full_rows / 32);
        image(
            &mut full,
            0x1234_5600,
            19,
            full_rows,
            &full_weights,
            &activation,
        )?;
        let field = |offset| -> Result<u32, Error> {
            Ok(u32::from_le_bytes(
                full[offset..offset + 4]
                    .try_into()
                    .map_err(|_| Error::Contract)?,
            ))
        };
        assert_eq!(field(32)?, 1_253_376);
        assert_eq!(field(52)?, 17_408);
        assert_eq!(field(144)?, 139_264);
        assert_eq!(field(160)?, 17_408);
        assert_eq!(field(180)?, 69_632);
        assert_eq!(field(264 + 16)?, 576);
        assert_eq!(field(264 + 20)?, 640);
        assert_eq!(field(264 + 28)?, 61_440);
        assert!(full_layout.output + full_rows * 8 <= full.len());
        Ok(())
    }
}
