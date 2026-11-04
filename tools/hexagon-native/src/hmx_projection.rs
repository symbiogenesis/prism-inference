//! Fixed signed-source HMX bitplane packet and bounded terminal-session inputs.
#![allow(
    clippy::redundant_pub_crate,
    reason = "Packet helpers are shared only with the audited native sibling"
)]
#![cfg_attr(
    not(all(windows, target_arch = "aarch64")),
    allow(
        dead_code,
        reason = "Packet helpers are exercised by hosted controls and native Windows ARM64 sessions"
    )
)]

use crate::{Error, Q2ProjectionLimits, Q2ProjectionReport, compute, projection};
use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

/// Fixed VTCM request used to derive the HMX chunks; refusal never falls back.
pub const HMX_Q2_VTCM_REQUEST_BYTES: u32 = 65_536;
/// Maximum opt-in host-mapped immutable packed-weight arena; this is not VTCM.
pub const HMX_Q2_MAX_RESIDENT_WEIGHT_BYTES: usize = 64 * 1024 * 1024;

/// One terminal K=128/N=8 bitplane session's explicit lifetime and work limits.
#[derive(Clone, Copy, Debug)]
pub struct HmxQ2ProjectionLimits {
    pub(crate) common: Q2ProjectionLimits,
    pub(crate) resident_weight_bytes: usize,
}

impl HmxQ2ProjectionLimits {
    /// Validate 32-row multiples up to 17,408, at most 24,000 write attempts and
    /// one 4..=120-second allowance including initialization and cleanup.
    ///
    /// # Errors
    /// Rejects unsupported extents or bounds before any native operation.
    pub fn new(rows: usize, operations: u32, duration: Duration) -> Result<Self, Error> {
        Ok(Self {
            common: Q2ProjectionLimits::new(rows, operations, duration)?,
            resident_weight_bytes: 0,
        })
    }

    /// Reserve an optional immutable host-mapped weight arena in the same FD.
    ///
    /// This increases the allocation bound, without changing VTCM, operation or
    /// lifetime limits. The caller separately admits these bytes. Existing
    /// whole-allocation cache flags remain unchanged; no reduced DMA or cache
    /// traffic follows from avoiding repeated host tile copies.
    ///
    /// # Errors
    /// Rejects more than 64 MiB before opening a module or allocating memory.
    pub const fn with_resident_weights(mut self, bytes: usize) -> Result<Self, Error> {
        if bytes > HMX_Q2_MAX_RESIDENT_WEIGHT_BYTES {
            return Err(Error::Contract);
        }
        self.resident_weight_bytes = bytes;
        Ok(self)
    }

    /// Explicit opt-in immutable packed-weight payload capacity, excluding padding.
    #[must_use]
    pub const fn resident_weight_bytes(self) -> usize {
        self.resident_weight_bytes
    }

    /// Largest active output-row extent for one operation.
    #[must_use]
    pub const fn maximum_rows(self) -> usize {
        self.common.maximum_rows()
    }
    /// Maximum mapped host allocation, including the command and eight outputs.
    #[must_use]
    pub const fn allocation_bytes(self) -> usize {
        (layout(self.common.rows).allocation + self.resident_weight_bytes).next_multiple_of(4096)
    }
    /// Maximum single-attempt writes in the complete session.
    #[must_use]
    pub const fn maximum_operations(self) -> u32 {
        self.common.maximum_operations()
    }
    /// Absolute session allowance including the three-second cleanup reserve.
    #[must_use]
    pub const fn maximum_duration(self) -> Duration {
        self.common.maximum_duration()
    }
}

/// An immutable packed-weight extent owned by one submitter's registry.
///
/// The private owner identity prevents a handle from another session from
/// addressing native memory. Cloning a handle never extends a native session's
/// lifetime, frees storage or changes its append-only contents.
#[derive(Clone, Debug)]
pub struct HmxQ2WeightHandle {
    owner: Arc<()>,
    range: Range<usize>,
    rows: usize,
}

impl HmxQ2WeightHandle {
    /// Exact active weight-row count, always a positive multiple of 32.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Exact packed payload size, excluding alignment gaps.
    #[must_use]
    pub const fn packed_bytes(&self) -> usize {
        self.range.end - self.range.start
    }
}

/// Metadata-only append-only allocator for one submitter's packed-weight arena.
///
/// Custom model-free submitters may use this to issue and validate opaque
/// handles. The registry stores no payload or native pointer. The submitter
/// copies a successful reservation's bytes exactly once before exposing its
/// handle, and never changes or frees them while the session can consume them.
pub struct HmxQ2WeightRegistry {
    owner: Arc<()>,
    capacity: usize,
    used: usize,
}

impl HmxQ2WeightRegistry {
    /// Create one independent owner with an explicitly admitted payload bound.
    ///
    /// # Errors
    /// Rejects capacity above the same 64 MiB native bound.
    pub fn new(capacity_bytes: usize) -> Result<Self, Error> {
        if capacity_bytes > HMX_Q2_MAX_RESIDENT_WEIGHT_BYTES {
            return Err(Error::Contract);
        }
        Ok(Self {
            owner: Arc::new(()),
            capacity: capacity_bytes,
            used: 0,
        })
    }

    /// Remaining payload extent, including alignment charged by prior reservations.
    #[must_use]
    pub const fn remaining_bytes(&self) -> usize {
        self.capacity - self.used
    }

    /// Validate and reserve one immutable, 128-byte-aligned packed-weight extent.
    ///
    /// `None` means insufficient capacity and leaves the registry unchanged.
    /// This consumes no native operation. No reservation is ever evicted/reused.
    ///
    /// # Errors
    /// Rejects malformed geometry or non-unit packed scales before reservation.
    pub fn prepare(&mut self, weights: &[u8]) -> Result<Option<HmxQ2WeightHandle>, Error> {
        let rows = validate_weights(projection::Q2_PROJECTION_MAX_ROWS, weights)?;
        let start = self.used.next_multiple_of(128);
        let end = start.checked_add(weights.len()).ok_or(Error::Contract)?;
        if end > self.capacity {
            return Ok(None);
        }
        self.used = end;
        Ok(Some(HmxQ2WeightHandle {
            owner: Arc::clone(&self.owner),
            range: start..end,
            rows,
        }))
    }

    /// Recover an owner-checked byte range inside this registry's arena.
    ///
    /// # Errors
    /// Rejects handles issued by any other registry, including earlier sessions.
    pub fn range(&self, handle: &HmxQ2WeightHandle) -> Result<Range<usize>, Error> {
        if !Arc::ptr_eq(&self.owner, &handle.owner)
            || handle.range.end > self.used
            || handle.range.end > self.capacity
        {
            return Err(Error::Contract);
        }
        Ok(handle.range.clone())
    }
}

/// Terminal HMX session report with the same explicit lifetime observations.
/// Check [`Q2ProjectionReport::is_hmx_bitplane_control`] before consuming it as HMX evidence.
pub type HmxQ2ProjectionReport = Q2ProjectionReport;

/// Synchronous exact integer bitplane submission to the fixed source HMX path.
pub trait HmxQ2Submitter {
    /// Remaining writes in the actual borrowed native session, when known.
    ///
    /// Custom submitters returning `None` require independently matched limits.
    /// Native sessions include all earlier work, including previous executors.
    fn remaining_operations(&self) -> Option<u64> {
        None
    }

    /// Remaining optional immutable host-mapped weight capacity, including padding.
    /// The default submitter supports no residency and preserves ordinary submission.
    fn residency_capacity_bytes(&self) -> usize {
        0
    }

    /// Copy immutable packed weights into an optional session-owned arena once.
    ///
    /// Capacity refusal returns `None` without consuming an operation or
    /// faulting a healthy native session. Custom submitters may use
    /// [`HmxQ2WeightRegistry`] to issue owner-checked handles.
    ///
    /// # Errors
    /// Malformed weights or an expired/faulted native session reject explicitly.
    fn prepare_weights(&mut self, _weights: &[u8]) -> Result<Option<HmxQ2WeightHandle>, Error> {
        Ok(None)
    }

    /// Submit a previously prepared immutable extent without copying its weights.
    ///
    /// Geometry, activation, queue correlation, work and deadline checks match
    /// ordinary submission. The handle must belong to this exact session.
    ///
    /// # Errors
    /// Unsupported residency, foreign handles and submission failures reject;
    /// native failures permanently close the session with no CPU replay.
    fn submit_resident(
        &mut self,
        _handle: &HmxQ2WeightHandle,
        _activation: &[[f32; 128]; 8],
        output: &mut [f32],
    ) -> Result<(), Error> {
        output.fill(f32::NAN);
        Err(Error::Contract)
    }

    /// Submit unit-scale tiled Q2 weights and eight binary F32 K=128 planes.
    /// `output` is contiguous `[plane][M-row]`, with M a multiple of 32.
    /// All correlated outputs must be integral in -128..=256; the caller must
    /// additionally compare exact dots against an independent oracle.
    ///
    /// # Errors
    /// Any refusal, malformed output or deadline permanently closes this
    /// session and retains uncertain native owners until supervised exit.
    fn submit(
        &mut self,
        weights: &[u8],
        activation: &[[f32; 128]; 8],
        output: &mut [f32],
    ) -> Result<(), Error>;
}

pub(crate) const fn layout(rows: usize) -> projection::Layout {
    let weights = rows / 32 * 1152;
    let activation = (1024 + weights).next_multiple_of(128);
    let output = activation + 4096;
    projection::Layout {
        weights,
        activation,
        output,
        allocation: (output + rows * 32).next_multiple_of(4096),
    }
}

pub(crate) fn start_wire(queue_id: u64, allocation: usize) -> Result<[u8; 48], Error> {
    let mut result = compute::start_wire(queue_id);
    put32(&mut result, 20, 1);
    put32(&mut result, 24, HMX_Q2_VTCM_REQUEST_BYTES);
    put64(
        &mut result,
        32,
        u64::try_from(allocation).map_err(|_| Error::Contract)?,
    );
    Ok(result)
}

// Specialization of htp_mm_hmx_compute_chunks and solve_2d_params at pinned
// 19cce691: K128, m8 padded32, Q2_0, one thread, no pipeline/bias, 64KiB.
// Per-N=2*640/32+128*2=296, per-M=256, per-MN=2, overhead=5*2048+256.
// Cost ties prefer least tail waste then largest mc*nc, as the source does.
pub(crate) fn chunks(rows: usize) -> Result<(u32, u32), Error> {
    let usable = 65_536 - 10_496;
    let mut best: Option<(usize, usize, usize)> = None;
    for columns in (32..=rows.min(usable / 296)).step_by(32) {
        let remaining = usable - columns * 296;
        let mc = (remaining / (256 + columns * 2) / 32 * 32).min(32);
        if mc == 0 {
            continue;
        }
        let blocks = rows.div_ceil(columns);
        let waste = if rows.is_multiple_of(columns) {
            0
        } else {
            columns - rows % columns
        };
        if best.is_none_or(|(best_blocks, best_waste, best_columns)| {
            (blocks, waste, std::cmp::Reverse(columns))
                < (best_blocks, best_waste, std::cmp::Reverse(best_columns))
        }) {
            best = Some((blocks, waste, columns));
        }
    }
    let (_, _, columns) = best.ok_or(Error::Contract)?;
    // Exact htp_mm_hmx_vtcm_layout_build, not its conservative cost estimate.
    let weight = (columns / 32 * 2 * 640).next_multiple_of(2048);
    let output = (32 * columns * 2).next_multiple_of(2048);
    let scratch = (columns * 256).next_multiple_of(2048);
    let compute_group = weight + output + scratch;
    let activation_temp = compute_group.clamp(2048, 32_768).next_multiple_of(128);
    let bytes = 2048 + 8192 + compute_group.max(activation_temp);
    if bytes > 65_536 {
        return Err(Error::Contract);
    }
    Ok((
        u32::try_from(columns).map_err(|_| Error::Contract)?,
        u32::try_from(bytes).map_err(|_| Error::Contract)?,
    ))
}

pub(crate) fn validate_inputs(
    maximum_rows: usize,
    weights: &[u8],
    activation: &[[f32; 128]; 8],
    outputs: usize,
) -> Result<usize, Error> {
    let rows = validate_weights(maximum_rows, weights)?;
    validate_activation(maximum_rows, rows, activation, outputs)?;
    Ok(rows)
}

pub(crate) fn validate_activation(
    maximum_rows: usize,
    rows: usize,
    activation: &[[f32; 128]; 8],
    outputs: usize,
) -> Result<(), Error> {
    if outputs / 8 != rows
        || !outputs.is_multiple_of(8)
        || rows == 0
        || rows > maximum_rows
        || !rows.is_multiple_of(32)
    {
        return Err(Error::Contract);
    }
    if activation
        .iter()
        .flatten()
        .any(|value| !matches!(value.to_bits(), 0 | 0x3f80_0000))
    {
        return Err(Error::Contract);
    }
    Ok(())
}

fn validate_weights(maximum_rows: usize, weights: &[u8]) -> Result<usize, Error> {
    let rows = weights.len() / 36;
    if rows == 0
        || rows > maximum_rows
        || !rows.is_multiple_of(32)
        || weights.len() != layout(rows).weights
        || weights.as_chunks::<576>().0.iter().any(|tile| {
            tile[512..]
                .as_chunks::<2>()
                .0
                .iter()
                .any(|scale| *scale != [0, 0x3c])
        })
    {
        return Err(Error::Contract);
    }
    Ok(rows)
}

pub(crate) fn image(
    bytes: &mut [u8],
    base: u64,
    fd: u32,
    rows: usize,
    weights: &[u8],
    activation: &[[f32; 128]; 8],
) -> Result<projection::Layout, Error> {
    let active = frame(bytes, base, fd, rows, activation)?;
    if weights.len() != active.weights {
        return Err(Error::Contract);
    }
    bytes[1024..1024 + weights.len()].copy_from_slice(weights);
    Ok(active)
}

pub(crate) fn image_resident(
    bytes: &mut [u8],
    base: u64,
    fd: u32,
    arena_start: usize,
    handle: &HmxQ2WeightHandle,
    activation: &[[f32; 128]; 8],
) -> Result<projection::Layout, Error> {
    let allocation = bytes.len();
    let weight_offset = arena_start
        .checked_add(handle.range.start)
        .ok_or(Error::Contract)?;
    let weight_end = arena_start
        .checked_add(handle.range.end)
        .ok_or(Error::Contract)?;
    if weight_end > allocation || handle.packed_bytes() != layout(handle.rows).weights {
        return Err(Error::Contract);
    }
    let working = bytes.get_mut(..arena_start).ok_or(Error::Contract)?;
    let active = frame(working, base, fd, handle.rows, activation)?;
    put64(
        bytes,
        8,
        u64::try_from(allocation).map_err(|_| Error::Contract)?,
    );
    put64(
        bytes,
        24,
        u64::try_from(weight_offset).map_err(|_| Error::Contract)?,
    );
    Ok(active)
}

fn frame(
    bytes: &mut [u8],
    base: u64,
    fd: u32,
    rows: usize,
    activation: &[[f32; 128]; 8],
) -> Result<projection::Layout, Error> {
    const KP: usize = 264;
    let active = layout(rows);
    if bytes.len() < active.allocation
        || bytes.len() > layout(projection::Q2_PROJECTION_MAX_ROWS).allocation
    {
        return Err(Error::Contract);
    }
    let (n_chunk, vtcm_size) = chunks(rows)?;
    bytes.fill(0);
    let framing = compute::q2_fixture_image(base, fd);
    bytes[..1024].copy_from_slice(&framing[..1024]);
    put64(
        bytes,
        8,
        u64::try_from(bytes.len()).map_err(|_| Error::Contract)?,
    );
    // Weight tensor: Q2_0 K128/M rows, WEIGHT|TILED, 36 logical bytes/row.
    for (offset, value) in [
        (32, active.weights),
        (48, 128),
        (52, rows),
        (64, 18),
        (68, 36),
        (72, active.weights),
        (76, active.weights),
        (88, 4096),
        (104, 128),
        (108, 8),
        (120, 4),
        (124, 512),
        (128, 4096),
        (132, 4096),
        (144, rows * 32),
        (160, rows),
        (164, 8),
        (176, 4),
        (180, rows * 4),
        (184, rows * 32),
        (188, rows * 32),
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
    bytes[KP..KP + 128].fill(0);
    bytes[KP] = 1; // HTP_MM_KERNEL_HMX_2D.
    bytes[KP + 3] = 1; // n_hmx; no pipeline for N8.
    bytes[KP + 4] = 1; // one HVX worker.
    bytes[KP + 5] = 1; // one F32->F16 activation worker.
    for (offset, value) in [
        (8, 32),
        (12, n_chunk),
        (16, 576),
        (20, 640),
        (24, 5120),
        (28, vtcm_size),
    ] {
        put32(bytes, KP + offset, value);
    }
    // fastdiv(8)=(1,3), (1)=(1,0), (128)=(1,7).
    for (index, shift) in [3, 3, 0, 0, 0, 0, 7].into_iter().enumerate() {
        put32(bytes, KP + 36 + index * 8, 1);
        put32(bytes, KP + 40 + index * 8, shift);
    }
    for (index, value) in activation.iter().flatten().enumerate() {
        put32(bytes, active.activation + index * 4, value.to_bits());
    }
    for index in 0..rows * 8 {
        put32(bytes, active.output + index * 4, 0x7fc0_ae51);
    }
    Ok(active)
}

pub(crate) fn output_valid(bytes: &[u8]) -> bool {
    !bytes.is_empty()
        && bytes.len().is_multiple_of(4)
        && bytes.as_chunks::<4>().0.iter().all(|value| {
            let value = f32::from_le_bytes(*value);
            value.is_finite() && value.fract() == 0.0 && (-128.0..=256.0).contains(&value)
        })
}

fn put32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}
fn put64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hmx_limits_share_the_authorized_cap_without_extending_time_or_allocation() -> Result<(), Error>
    {
        let limits = HmxQ2ProjectionLimits::new(17_408, 24_000, Duration::from_secs(120))?;
        assert_eq!(
            limits.maximum_operations(),
            projection::Q2_PROJECTION_MAX_OPERATIONS
        );
        assert_eq!(limits.maximum_duration(), Duration::from_secs(120));
        assert_eq!(limits.allocation_bytes(), 1_191_936);
        for operations in [0, 24_001, u32::MAX] {
            assert!(HmxQ2ProjectionLimits::new(32, operations, Duration::from_secs(4)).is_err());
        }
        assert!(HmxQ2ProjectionLimits::new(32, 24_000, Duration::from_secs(121)).is_err());
        Ok(())
    }

    #[test]
    fn fixed_hmx_solver_and_frame_keep_eight_planes_and_exact_bounded_extents() -> Result<(), Error>
    {
        assert_eq!(chunks(32)?, (32, 22528));
        assert_eq!(chunks(64)?, (64, 34816));
        assert_eq!(chunks(17_408)?, (128, 57344));
        let limits = HmxQ2ProjectionLimits::new(17_408, 40, Duration::from_secs(20))?;
        assert_eq!(limits.allocation_bytes(), 1_191_936);
        let start = start_wire(0x1234, limits.allocation_bytes())?;
        assert_eq!(&start[16..28], &[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]);
        let mut weights = vec![0x55; 1152];
        for tile in weights.as_chunks_mut::<576>().0 {
            for scale in tile[512..].as_chunks_mut::<2>().0 {
                *scale = [0, 0x3c];
            }
        }
        let activation = [[1.0; 128]; 8];
        assert_eq!(validate_inputs(32, &weights, &activation, 256)?, 32);
        let mut bytes = vec![0; layout(32).allocation];
        let active = image(&mut bytes, 0x1234_5600, 7, 32, &weights, &activation)?;
        assert_eq!(&bytes[264..270], &[1, 0, 0, 1, 1, 1]);
        assert_eq!(&bytes[264 + 8..264 + 16], &[32, 0, 0, 0, 32, 0, 0, 0]);
        assert_eq!(&bytes[264 + 28..264 + 32], &22528_u32.to_le_bytes());
        assert!(!output_valid(&bytes[active.output..active.output + 1024]));
        let mut malformed = activation;
        malformed[0][0] = -0.0;
        assert!(validate_inputs(32, &weights, &malformed, 256).is_err());
        for value in [f32::NAN, f32::INFINITY, -129.0, 257.0, 0.5] {
            assert!(!output_valid(&value.to_le_bytes()));
        }
        for value in [-128.0_f32, 0.0, 256.0] {
            assert!(output_valid(&value.to_le_bytes()));
        }
        Ok(())
    }

    #[test]
    fn resident_registry_has_owner_identity_and_transactional_capacity_refusal() -> Result<(), Error>
    {
        let mut weights = vec![0x55; 1152];
        for tile in weights.as_chunks_mut::<576>().0 {
            for scale in tile[512..].as_chunks_mut::<2>().0 {
                *scale = [0, 0x3c];
            }
        }
        let mut owner = HmxQ2WeightRegistry::new(2304)?;
        let mut other = HmxQ2WeightRegistry::new(2304)?;
        let first = owner.prepare(&weights)?.ok_or(Error::Contract)?;
        let second = owner.prepare(&weights)?.ok_or(Error::Contract)?;
        assert_eq!(first.rows(), 32);
        assert_eq!(first.packed_bytes(), 1152);
        assert_eq!(owner.range(&first)?, 0..1152);
        assert_eq!(owner.range(&second)?, 1152..2304);
        assert_eq!(owner.remaining_bytes(), 0);
        assert!(owner.prepare(&weights)?.is_none());
        assert_eq!(owner.remaining_bytes(), 0);
        assert!(other.range(&first).is_err());
        let foreign = other.prepare(&weights)?.ok_or(Error::Contract)?;
        assert!(owner.range(&foreign).is_err());
        weights[512] = 1;
        assert!(other.prepare(&weights).is_err());
        assert_eq!(other.remaining_bytes(), 1152);
        assert!(HmxQ2WeightRegistry::new(HMX_Q2_MAX_RESIDENT_WEIGHT_BYTES + 1).is_err());
        let limits = HmxQ2ProjectionLimits::new(32, 3, Duration::from_secs(4))?;
        assert_eq!(limits.resident_weight_bytes(), 0);
        assert_eq!(limits.allocation_bytes(), 8192);
        let resident = limits.with_resident_weights(2304)?;
        assert_eq!(resident.resident_weight_bytes(), 2304);
        assert_eq!(resident.allocation_bytes(), 12288);
        assert_eq!(resident.maximum_operations(), limits.maximum_operations());
        assert_eq!(resident.maximum_duration(), limits.maximum_duration());
        assert!(
            limits
                .with_resident_weights(HMX_Q2_MAX_RESIDENT_WEIGHT_BYTES + 1)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn resident_frame_preserves_immutable_arena_and_signed_solver_fields() -> Result<(), Error> {
        let mut weights = vec![0x55; 1152];
        for tile in weights.as_chunks_mut::<576>().0 {
            for scale in tile[512..].as_chunks_mut::<2>().0 {
                *scale = [0, 0x3c];
            }
        }
        let mut registry = HmxQ2WeightRegistry::new(2304)?;
        let first = registry.prepare(&weights)?.ok_or(Error::Contract)?;
        let second = registry.prepare(&weights)?.ok_or(Error::Contract)?;
        let working_bytes = layout(32).allocation;
        let mut bytes = vec![0xa5; 12288];
        bytes[working_bytes..working_bytes + 1152].copy_from_slice(&weights);
        bytes[working_bytes + 1152..working_bytes + 2304].copy_from_slice(&weights);
        let retained = bytes[working_bytes..].to_vec();
        let activation = [[1.0; 128]; 8];
        let mut ordinary = vec![0; working_bytes];
        image(&mut ordinary, 0x1234, 7, 32, &weights, &activation)?;
        for handle in [&first, &second, &first] {
            let active = image_resident(&mut bytes, 0x1234, 7, working_bytes, handle, &activation)?;
            assert_eq!(&bytes[working_bytes..], &retained);
            assert_eq!(&bytes[8..16], &12288_u64.to_le_bytes());
            let offset = working_bytes + registry.range(handle)?.start;
            assert_eq!(
                &bytes[24..32],
                &u64::try_from(offset)
                    .map_err(|_| Error::Contract)?
                    .to_le_bytes()
            );
            assert_eq!(&bytes[32..1024], &ordinary[32..1024]);
            assert_eq!(
                &bytes[active.activation..active.output],
                &ordinary[active.activation..active.output]
            );
            assert!(!output_valid(&bytes[active.output..active.output + 1024]));
        }
        Ok(())
    }
}
