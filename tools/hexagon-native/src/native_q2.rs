//! Full-`K` native `HTP_TYPE_Q2_0` 1-op projection session and packet builder.
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

use crate::{
    Error, HMX_Q2_HVX_WORKERS, HMX_Q2_POWER_MODE_BURST, HMX_Q2_VTCM_REQUEST_BYTES,
    Q2_PROJECTION_MAX_ROWS, Q2ProjectionLimits, Q2ProjectionReport, compute, hmx_projection,
    projection,
};
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

static NEXT_NATIVE_REGISTRY_OWNER: AtomicU64 = AtomicU64::new(1);

/// Largest input-column extent (`K`) for a single 1-op native `Q2_0` projection.
pub const NATIVE_Q2_MAX_COLUMNS: usize = Q2_PROJECTION_MAX_ROWS;
/// Largest weight-row extent (`M`) for a single 1-op native `Q2_0` projection.
pub const NATIVE_Q2_MAX_ROWS: usize = Q2_PROJECTION_MAX_ROWS;
/// Largest packed native `HTP_TYPE_Q2_0` weight payload for a single projection (`[5120, 17408]` or `[17408, 5120]`).
pub const NATIVE_Q2_MAX_WEIGHT_BYTES: usize = (Q2_PROJECTION_MAX_ROWS / 32) * (5120 / 64) * 576;
/// Largest working host allocation for any single native `Q2_0` projection (`[5120, 17408]` or `[17408, 5120]`).
pub const NATIVE_Q2_MAX_WORKING_BYTES: usize = layout(5120, Q2_PROJECTION_MAX_ROWS).allocation;
/// Largest immutable host-mapped weight arena in a single native `Q2_0` session (`480 MiB`, within the `512 MiB` `rpcmem` `m-size` limit).
pub const NATIVE_Q2_MAX_RESIDENT_WEIGHT_BYTES: usize = 480 * 1024 * 1024;
/// Fixed 4 MiB VTCM request used by the 8-HVX full-`K` `Q2_0` session (`v81` hardware has 8 MiB VTCM).
pub const NATIVE_Q2_VTCM_REQUEST_BYTES: u32 = 4 * 1024 * 1024;

/// One terminal full-`K` native `HTP_TYPE_Q2_0` session's lifetime and work bounds.
#[derive(Clone, Copy, Debug)]
pub struct NativeQ2ProjectionLimits {
    pub(crate) columns: usize,
    pub(crate) common: Q2ProjectionLimits,
    pub(crate) resident_weight_bytes: usize,
}

impl NativeQ2ProjectionLimits {
    /// Validate 128-column multiples up to 17,408, 32-row multiples up to 17,408,
    /// at most 24,000 write attempts and a 4..=120-second session allowance.
    ///
    /// # Errors
    /// Rejects unsupported extents or bounds before any native operation.
    pub fn new(
        columns: usize,
        rows: usize,
        operations: u32,
        duration: Duration,
    ) -> Result<Self, Error> {
        if columns == 0 || columns > NATIVE_Q2_MAX_COLUMNS || !columns.is_multiple_of(128) {
            return Err(Error::Contract);
        }
        Ok(Self {
            columns,
            common: Q2ProjectionLimits::new(rows, operations, duration)?,
            resident_weight_bytes: 0,
        })
    }

    /// Reserve an optional immutable host-mapped weight arena in the same FD.
    ///
    /// # Errors
    /// Rejects more than 480 MiB before opening a module or allocating memory.
    pub const fn with_resident_weights(mut self, bytes: usize) -> Result<Self, Error> {
        if bytes > NATIVE_Q2_MAX_RESIDENT_WEIGHT_BYTES {
            return Err(Error::Contract);
        }
        self.resident_weight_bytes = bytes;
        Ok(self)
    }

    /// Enable persistent service-lifetime operation without idle wall-clock expiration.
    ///
    /// Individual operations retain their 3-second hardware response deadline.
    #[must_use]
    pub const fn with_persistent_session(mut self) -> Self {
        self.common.persistent = true;
        self.common.operations = u32::MAX;
        self
    }

    /// Explicit opt-in immutable packed-weight payload capacity, excluding padding.
    #[must_use]
    pub const fn resident_weight_bytes(self) -> usize {
        self.resident_weight_bytes
    }

    /// Largest active input-column extent (`K`) for one operation.
    #[must_use]
    pub const fn maximum_columns(self) -> usize {
        self.columns
    }

    /// Largest active output-row extent (`M`) for one operation.
    #[must_use]
    pub const fn maximum_rows(self) -> usize {
        self.common.maximum_rows()
    }

    /// Maximum mapped host allocation, including command, activation, output and weights.
    #[must_use]
    pub const fn allocation_bytes(self) -> usize {
        let working = working_allocation(self.columns, self.common.rows);
        (working + self.resident_weight_bytes).next_multiple_of(4096)
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

/// An immutable full-`K` tiled `Q2_0` weight extent owned by one submitter's registry.
#[derive(Clone, Debug)]
pub struct NativeQ2WeightHandle {
    owner: u64,
    range: Range<usize>,
    columns: usize,
    rows: usize,
}

impl NativeQ2WeightHandle {
    /// Exact active input-column count (`K`), always a positive multiple of 128.
    #[must_use]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// Exact active weight-row count (`M`), always a positive multiple of 32.
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

/// Metadata-only append-only allocator for one submitter's native `Q2_0` weight arena.
#[derive(Clone, Copy)]
pub struct NativeQ2WeightRegistry {
    owner: u64,
    capacity: usize,
    used: usize,
}

impl NativeQ2WeightRegistry {
    /// Create one independent owner with an explicitly admitted payload bound.
    ///
    /// # Errors
    /// Rejects capacity above the 256 MiB native bound.
    pub fn new(capacity_bytes: usize) -> Result<Self, Error> {
        if capacity_bytes > NATIVE_Q2_MAX_RESIDENT_WEIGHT_BYTES {
            return Err(Error::Contract);
        }
        Ok(Self {
            owner: NEXT_NATIVE_REGISTRY_OWNER.fetch_add(1, Ordering::Relaxed),
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
    /// # Errors
    /// Rejects malformed geometry or nonfinite packed `FP16` scales before reservation.
    pub fn prepare(
        &mut self,
        columns: usize,
        rows: usize,
        weights: &[u8],
    ) -> Result<Option<NativeQ2WeightHandle>, Error> {
        validate_weights(
            NATIVE_Q2_MAX_COLUMNS,
            NATIVE_Q2_MAX_ROWS,
            columns,
            rows,
            weights,
        )?;
        self.prepare_validated(columns, rows, weights.len())
    }

    /// Reserve one immutable, 128-byte-aligned packed-weight extent whose
    /// geometry and `FP16` scales have already been validated.
    pub(crate) fn prepare_validated(
        &mut self,
        columns: usize,
        rows: usize,
        weight_bytes: usize,
    ) -> Result<Option<NativeQ2WeightHandle>, Error> {
        let start = self.used.next_multiple_of(128);
        let end = start.checked_add(weight_bytes).ok_or(Error::Contract)?;
        if end > self.capacity {
            return Ok(None);
        }
        self.used = end;
        Ok(Some(NativeQ2WeightHandle {
            owner: self.owner,
            range: start..end,
            columns,
            rows,
        }))
    }

    /// Bytes of the immutable arena currently populated (including internal alignment padding).
    #[must_use]
    pub const fn used_bytes(&self) -> usize {
        self.used
    }

    /// Recover an owner-checked byte range inside this registry's arena.
    ///
    /// # Errors
    /// Rejects handles issued by any other registry, including earlier sessions.
    pub fn range(&self, handle: &NativeQ2WeightHandle) -> Result<Range<usize>, Error> {
        if self.owner != handle.owner
            || handle.range.end > self.used
            || handle.range.end > self.capacity
        {
            return Err(Error::Contract);
        }
        Ok(handle.range.clone())
    }
}

/// Terminal native `Q2_0` session report with the same explicit lifetime observations.
pub type NativeQ2ProjectionReport = Q2ProjectionReport;

/// Synchronous full-`K` 1-op native `HTP_TYPE_Q2_0` projection submission.
pub trait NativeQ2Submitter {
    /// Remaining writes in the actual borrowed native session, when known.
    fn remaining_operations(&self) -> Option<u64> {
        None
    }

    /// Remaining optional immutable host-mapped weight capacity, including padding.
    fn residency_capacity_bytes(&self) -> usize {
        0
    }

    /// Copy immutable tiled `Q2_0` weights into an optional session-owned arena once.
    ///
    /// # Errors
    /// Malformed weights or an expired/faulted native session reject explicitly.
    fn prepare_weights(
        &mut self,
        _columns: usize,
        _rows: usize,
        _weights: &[u8],
    ) -> Result<Option<NativeQ2WeightHandle>, Error> {
        Ok(None)
    }

    /// Submit a previously prepared immutable extent without copying its weights.
    ///
    /// # Errors
    /// Unsupported residency, foreign handles and submission failures reject;
    /// native failures permanently close the session with no CPU replay.
    fn submit_resident(
        &mut self,
        _handle: &NativeQ2WeightHandle,
        _activation: &[f32],
        output: &mut [f32],
    ) -> Result<(), Error> {
        output.fill(f32::NAN);
        Err(Error::Contract)
    }

    /// Submit two resident `HTP_TYPE_Q2_0` projections sharing one `F32`
    /// activation vector in a single 2-op DSP queue submission.
    ///
    /// # Errors
    /// Unsupported residency, foreign handles, mismatched input columns or
    /// submission failures reject and close the session with no CPU replay.
    fn submit_resident_batch2(
        &mut self,
        _handles: [&NativeQ2WeightHandle; 2],
        _activation: &[f32],
        mut outputs: [&mut [f32]; 2],
    ) -> Result<(), Error> {
        for output in &mut outputs {
            output.fill(f32::NAN);
        }
        Err(Error::Contract)
    }

    /// Submit three resident `HTP_TYPE_Q2_0` projections sharing one `F32`
    /// activation vector in a single 3-op DSP queue submission.
    ///
    /// # Errors
    /// Unsupported residency, foreign handles, mismatched input columns or
    /// submission failures reject and close the session with no CPU replay.
    fn submit_resident_batch3(
        &mut self,
        _handles: [&NativeQ2WeightHandle; 3],
        _activation: &[f32],
        mut outputs: [&mut [f32]; 3],
    ) -> Result<(), Error> {
        for output in &mut outputs {
            output.fill(f32::NAN);
        }
        Err(Error::Contract)
    }

    /// Submit tiled `HTP_TYPE_Q2_0` weights (`[columns, rows]`) and one `F32`
    /// activation vector (`[columns]`) in a single DSP operation.
    ///
    /// # Errors
    /// Any refusal, malformed output or deadline permanently closes this
    /// session and retains uncertain native owners until supervised exit.
    fn submit(
        &mut self,
        columns: usize,
        rows: usize,
        weights: &[u8],
        activation: &[f32],
        output: &mut [f32],
    ) -> Result<(), Error>;
}

impl<S: NativeQ2Submitter + ?Sized> NativeQ2Submitter for &mut S {
    fn remaining_operations(&self) -> Option<u64> {
        (**self).remaining_operations()
    }

    fn residency_capacity_bytes(&self) -> usize {
        (**self).residency_capacity_bytes()
    }

    fn prepare_weights(
        &mut self,
        columns: usize,
        rows: usize,
        weights: &[u8],
    ) -> Result<Option<NativeQ2WeightHandle>, Error> {
        (**self).prepare_weights(columns, rows, weights)
    }

    fn submit_resident(
        &mut self,
        handle: &NativeQ2WeightHandle,
        activation: &[f32],
        output: &mut [f32],
    ) -> Result<(), Error> {
        (**self).submit_resident(handle, activation, output)
    }

    fn submit_resident_batch2(
        &mut self,
        handles: [&NativeQ2WeightHandle; 2],
        activation: &[f32],
        outputs: [&mut [f32]; 2],
    ) -> Result<(), Error> {
        (**self).submit_resident_batch2(handles, activation, outputs)
    }

    fn submit_resident_batch3(
        &mut self,
        handles: [&NativeQ2WeightHandle; 3],
        activation: &[f32],
        outputs: [&mut [f32]; 3],
    ) -> Result<(), Error> {
        (**self).submit_resident_batch3(handles, activation, outputs)
    }

    fn submit(
        &mut self,
        columns: usize,
        rows: usize,
        weights: &[u8],
        activation: &[f32],
        output: &mut [f32],
    ) -> Result<(), Error> {
        (**self).submit(columns, rows, weights, activation, output)
    }
}

pub(crate) const fn layout(columns: usize, rows: usize) -> projection::Layout {
    let weights = (rows / 32) * (columns / 64) * 576;
    let activation = 1024;
    let output = (activation + columns * 4).next_multiple_of(128);
    let weights_offset = (output + rows * 4).next_multiple_of(128);
    projection::Layout {
        weights,
        activation,
        output,
        allocation: (weights_offset + weights).next_multiple_of(4096),
    }
}

pub(crate) const fn working_allocation(columns: usize, rows: usize) -> usize {
    let raw = layout(columns, rows).allocation;
    if raw > NATIVE_Q2_MAX_WORKING_BYTES {
        NATIVE_Q2_MAX_WORKING_BYTES
    } else {
        raw
    }
}

pub(crate) fn start_wire(queue_id: u64, allocation: usize) -> Result<[u8; 48], Error> {
    let mut result = compute::start_wire(queue_id);
    put32(&mut result, 16, HMX_Q2_HVX_WORKERS);
    put32(&mut result, 20, 1);
    put32(&mut result, 24, NATIVE_Q2_VTCM_REQUEST_BYTES);
    put32(&mut result, 40, HMX_Q2_POWER_MODE_BURST);
    put64(
        &mut result,
        32,
        u64::try_from(allocation).map_err(|_| Error::Contract)?,
    );
    Ok(result)
}

/// Specialization of `htp_kparams_solve_matmul` (`0x1800282F0`) and
/// `htp_kparams_solve_hvx_mv` (`0x1800322DC` + `0x180029EB0`) at pinned
/// `19cce691` for `src0 = HTP_TYPE_Q2_0 (42)`, `src1 = F32`, `dst = F32`,
/// `N = 1`, `n_hvx = 8` and `2 MiB` / `4 MiB` VTCM.
///
/// Returns `(n_prefetch, src1_row_size, vtcm_size)`.
pub(crate) fn hvx_mv_params(columns: usize, rows: usize) -> Result<(u8, u32, u32), Error> {
    if columns == 0
        || columns > NATIVE_Q2_MAX_COLUMNS
        || !columns.is_multiple_of(128)
        || rows == 0
        || rows > NATIVE_Q2_MAX_ROWS
        || !rows.is_multiple_of(32)
    {
        return Err(Error::Contract);
    }
    let vtcm_request = if columns > 6144 {
        NATIVE_Q2_VTCM_REQUEST_BYTES
    } else {
        HMX_Q2_VTCM_REQUEST_BYTES
    };
    let vtcm_budget = usize::try_from(vtcm_request).map_err(|_| Error::Contract)?;
    let n_threads = usize::try_from(HMX_Q2_HVX_WORKERS).map_err(|_| Error::Contract)?;
    let k_groups = ((columns + 127) >> 5) & !3;
    let src1_row_size = k_groups * 1280;
    let k_tiles = (columns + 63) >> 6;
    let dst_row_bytes = rows * 4;
    let src1_vtcm = src1_row_size.next_multiple_of(256);
    let dst_vtcm = dst_row_bytes.div_ceil(n_threads).next_multiple_of(128) * n_threads;
    let act_vtcm = (columns * 4).next_multiple_of(512);
    let mut n_prefetch = 16_usize;
    while n_prefetch > 0 {
        let src0_vtcm = (n_prefetch * k_tiles * 640).next_multiple_of(256) * n_threads;
        let total_vtcm = src1_vtcm + (dst_vtcm + src0_vtcm).max(act_vtcm);
        if total_vtcm <= vtcm_budget {
            return Ok((
                u8::try_from(n_prefetch).map_err(|_| Error::Contract)?,
                u32::try_from(src1_row_size).map_err(|_| Error::Contract)?,
                u32::try_from(total_vtcm).map_err(|_| Error::Contract)?,
            ));
        }
        n_prefetch >>= 1;
    }
    Err(Error::Contract)
}

pub(crate) fn validate_inputs(
    maximum_columns: usize,
    maximum_rows: usize,
    columns: usize,
    rows: usize,
    weights: &[u8],
    activation: &[f32],
    outputs: usize,
) -> Result<(), Error> {
    validate_weights(maximum_columns, maximum_rows, columns, rows, weights)?;
    validate_activation(
        maximum_columns,
        maximum_rows,
        columns,
        rows,
        activation,
        outputs,
    )
}

#[inline]
pub(crate) fn is_finite_f32_slice(values: &[f32]) -> bool {
    const EXP_MASK: u32 = 0x7f80_0000;
    const EXP_STEP: u32 = 0x0080_0000;
    const SIGN_BIT: u32 = 0x8000_0000;

    let (chunks, tail) = values.as_chunks::<16>();
    for chunk in chunks {
        let mut mask = 0_u32;
        for &value in chunk {
            mask |= (value.to_bits() & EXP_MASK).wrapping_add(EXP_STEP);
        }
        if (mask & SIGN_BIT) != 0 {
            return false;
        }
    }
    let mut tail_mask = 0_u32;
    for &value in tail {
        tail_mask |= (value.to_bits() & EXP_MASK).wrapping_add(EXP_STEP);
    }
    (tail_mask & SIGN_BIT) == 0
}

pub(crate) const fn validate_activation_dimensions(
    maximum_columns: usize,
    maximum_rows: usize,
    columns: usize,
    rows: usize,
    activation_len: usize,
    outputs: usize,
) -> Result<(), Error> {
    if columns == 0
        || columns > maximum_columns
        || !columns.is_multiple_of(128)
        || rows == 0
        || rows > maximum_rows
        || !rows.is_multiple_of(32)
        || activation_len != columns
        || outputs != rows
    {
        return Err(Error::Contract);
    }
    Ok(())
}

pub(crate) fn validate_activation(
    maximum_columns: usize,
    maximum_rows: usize,
    columns: usize,
    rows: usize,
    activation: &[f32],
    outputs: usize,
) -> Result<(), Error> {
    validate_activation_dimensions(
        maximum_columns,
        maximum_rows,
        columns,
        rows,
        activation.len(),
        outputs,
    )?;
    if !is_finite_f32_slice(activation) {
        return Err(Error::Contract);
    }
    Ok(())
}

#[inline]
fn tile_scales_valid(scales: &[u8]) -> bool {
    const EXP_MASK: u64 = 0x7c00_7c00_7c00_7c00;
    const EXP_STEP: u64 = 0x0400_0400_0400_0400;
    const SIGN_BITS: u64 = 0x8000_8000_8000_8000;

    let mut mask = 0_u64;
    for chunk in scales.as_chunks::<8>().0 {
        let word = u64::from_le_bytes(*chunk);
        mask |= (word & EXP_MASK).wrapping_add(EXP_STEP);
    }
    (mask & SIGN_BITS) == 0
}

pub(crate) fn validate_weights(
    maximum_columns: usize,
    maximum_rows: usize,
    columns: usize,
    rows: usize,
    weights: &[u8],
) -> Result<(), Error> {
    if columns == 0
        || columns > maximum_columns
        || !columns.is_multiple_of(128)
        || rows == 0
        || rows > maximum_rows
        || !rows.is_multiple_of(32)
        || weights.len() != layout(columns, rows).weights
        || weights.len() > NATIVE_Q2_MAX_WEIGHT_BYTES
        || weights
            .as_chunks::<576>()
            .0
            .iter()
            .any(|tile| !tile_scales_valid(&tile[512..]))
    {
        return Err(Error::Contract);
    }
    Ok(())
}

pub(crate) fn image(
    bytes: &mut [u8],
    base: u64,
    fd: u32,
    columns: usize,
    rows: usize,
    weights: &[u8],
    activation: &[f32],
) -> Result<projection::Layout, Error> {
    let active = layout(columns, rows);
    if weights.len() != active.weights || bytes.len() < active.allocation {
        return Err(Error::Contract);
    }
    let weights_offset = (active.output + rows * 4).next_multiple_of(128);
    write_frame(
        bytes,
        base,
        bytes.len(),
        fd,
        columns,
        rows,
        weights_offset,
        active,
        activation,
    )?;
    bytes[weights_offset..weights_offset + weights.len()].copy_from_slice(weights);
    Ok(active)
}

pub(crate) fn image_resident(
    bytes: &mut [u8],
    base: u64,
    fd: u32,
    arena_start: usize,
    handle: &NativeQ2WeightHandle,
    activation: &[f32],
) -> Result<projection::Layout, Error> {
    let allocation = bytes.len();
    let active = layout(handle.columns, handle.rows);
    let weight_offset = arena_start
        .checked_add(handle.range.start)
        .ok_or(Error::Contract)?;
    let weight_end = arena_start
        .checked_add(handle.range.end)
        .ok_or(Error::Contract)?;
    if weight_end > allocation || handle.packed_bytes() != active.weights {
        return Err(Error::Contract);
    }
    let working = bytes.get_mut(..arena_start).ok_or(Error::Contract)?;
    write_frame(
        working,
        base,
        allocation,
        fd,
        handle.columns,
        handle.rows,
        weight_offset,
        active,
        activation,
    )?;
    Ok(active)
}

#[derive(Clone, Copy)]
pub(crate) struct BatchLayout<const B: usize> {
    pub(crate) output_offsets: [usize; B],
    pub(crate) output_end: usize,
}

pub(crate) fn image_batch_resident<const B: usize>(
    bytes: &mut [u8],
    base: u64,
    fd: u32,
    arena_start: usize,
    handles: [&NativeQ2WeightHandle; B],
    activation: &[f32],
) -> Result<BatchLayout<B>, Error> {
    const CMD_PAD: usize = 2048;
    if !matches!(B, 2 | 3) {
        return Err(Error::Contract);
    }
    let allocation = bytes.len();
    let columns = handles[0].columns;
    if activation.len() != columns {
        return Err(Error::Contract);
    }
    let act_offset = CMD_PAD;
    let act_bytes = columns * 4;
    let mut next_out = (act_offset + act_bytes).next_multiple_of(128);
    let mut weight_offsets = [0_usize; B];
    let mut output_offsets = [0_usize; B];
    for (member, handle) in handles.into_iter().enumerate() {
        let expected_weights = layout(columns, handle.rows).weights;
        let start = arena_start
            .checked_add(handle.range.start)
            .ok_or(Error::Contract)?;
        let end = arena_start
            .checked_add(handle.range.end)
            .ok_or(Error::Contract)?;
        if handle.columns != columns
            || end > allocation
            || handle.packed_bytes() != expected_weights
        {
            return Err(Error::Contract);
        }
        weight_offsets[member] = start;
        output_offsets[member] = next_out;
        next_out = (next_out + handle.rows * 4).next_multiple_of(128);
    }
    let output_end = output_offsets[B - 1] + handles[B - 1].rows * 4;
    let working = bytes.get_mut(..arena_start).ok_or(Error::Contract)?;
    if output_end > working.len() {
        return Err(Error::Contract);
    }
    working[..CMD_PAD].fill(0);
    put64(working, 0, base);
    put64(
        working,
        8,
        u64::try_from(allocation).map_err(|_| Error::Contract)?,
    );
    put32(working, 20, fd);
    let operations_base = 24 + B * 3 * 56;
    for (member, handle) in handles.into_iter().enumerate() {
        let ti_base = u16::try_from(member * 3).map_err(|_| Error::Contract)?;
        let weight_desc = 24 + (member * 3) * 56;
        let operation_desc = operations_base + member * 232;
        write_member_descriptors(
            working,
            ti_base,
            weight_desc,
            operation_desc,
            columns,
            handle.rows,
            weight_offsets[member],
            layout(columns, handle.rows).weights,
            act_offset,
            output_offsets[member],
        )?;
    }
    for (dst, &value) in working[act_offset..act_offset + act_bytes]
        .as_chunks_mut::<4>()
        .0[..activation.len()]
        .iter_mut()
        .zip(activation)
    {
        *dst = value.to_bits().to_le_bytes();
    }
    working[output_offsets[0]..output_end]
        .as_chunks_mut::<4>()
        .0
        .fill(0x7fc0_ae51_u32.to_le_bytes());
    Ok(BatchLayout {
        output_offsets,
        output_end,
    })
}

#[allow(
    clippy::too_many_arguments,
    reason = "Keep full-K Q2_0 tensor descriptors, HVX kparams and output poison region together"
)]
fn write_frame(
    bytes: &mut [u8],
    base: u64,
    allocation_bytes: usize,
    fd: u32,
    columns: usize,
    rows: usize,
    weight_offset: usize,
    active: projection::Layout,
    activation: &[f32],
) -> Result<(), Error> {
    if activation.len() != columns {
        return Err(Error::Contract);
    }
    write_descriptors(
        bytes,
        base,
        allocation_bytes,
        fd,
        columns,
        rows,
        weight_offset,
        active.weights,
        active.activation,
        active.output,
    )?;
    let act_bytes = columns * 4;
    let out_bytes = rows * 4;
    for (dst, &value) in bytes[active.activation..active.activation + act_bytes]
        .as_chunks_mut::<4>()
        .0[..activation.len()]
        .iter_mut()
        .zip(activation)
    {
        *dst = value.to_bits().to_le_bytes();
    }
    bytes[active.output..active.output + out_bytes]
        .as_chunks_mut::<4>()
        .0
        .fill(0x7fc0_ae51_u32.to_le_bytes());
    Ok(())
}

#[allow(
    clippy::too_many_arguments,
    reason = "Keep 1-op Q2_0 tensor descriptors and HVX kparams together"
)]
fn write_descriptors(
    bytes: &mut [u8],
    base: u64,
    allocation_bytes: usize,
    fd: u32,
    columns: usize,
    rows: usize,
    weight_offset: usize,
    weight_bytes: usize,
    act_offset: usize,
    out_offset: usize,
) -> Result<(), Error> {
    bytes[..1024].fill(0);
    put64(bytes, 0, base);
    put64(
        bytes,
        8,
        u64::try_from(allocation_bytes).map_err(|_| Error::Contract)?,
    );
    put32(bytes, 20, fd);
    write_member_descriptors(
        bytes,
        0,
        24,
        192,
        columns,
        rows,
        weight_offset,
        weight_bytes,
        act_offset,
        out_offset,
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "Keep member Q2_0 tensor descriptors and HVX kparams together"
)]
fn write_member_descriptors(
    bytes: &mut [u8],
    ti_base: u16,
    weight_desc: usize,
    op_desc: usize,
    columns: usize,
    rows: usize,
    weight_offset: usize,
    weight_bytes: usize,
    act_offset: usize,
    out_offset: usize,
) -> Result<(), Error> {
    let (n_prefetch, src1_row_size, vtcm_size) = hvx_mv_params(columns, rows)?;
    let weight_row_bytes = 36 * (columns / 128);
    let act_bytes = columns * 4;
    let out_bytes = rows * 4;
    let act_desc = weight_desc + 56;
    let out_desc = act_desc + 56;
    put64(
        bytes,
        weight_desc,
        u64::try_from(weight_offset).map_err(|_| Error::Contract)?,
    );
    put16(bytes, weight_desc + 22, ti_base);
    put64(
        bytes,
        act_desc,
        u64::try_from(act_offset).map_err(|_| Error::Contract)?,
    );
    put16(bytes, act_desc + 22, ti_base + 1);
    put64(
        bytes,
        out_desc,
        u64::try_from(out_offset).map_err(|_| Error::Contract)?,
    );
    put16(bytes, out_desc + 22, ti_base + 2);
    for (offset, value) in [
        (weight_desc + 8, weight_bytes),
        (weight_desc + 12, 3),
        (weight_desc + 16, 42),
        (weight_desc + 24, columns),
        (weight_desc + 28, rows),
        (weight_desc + 32, 1),
        (weight_desc + 36, 1),
        (weight_desc + 40, 18),
        (weight_desc + 44, weight_row_bytes),
        (weight_desc + 48, weight_bytes),
        (weight_desc + 52, weight_bytes),
        (act_desc + 8, act_bytes),
        (act_desc + 24, columns),
        (act_desc + 28, 1),
        (act_desc + 32, 1),
        (act_desc + 36, 1),
        (act_desc + 40, 4),
        (act_desc + 44, act_bytes),
        (act_desc + 48, act_bytes),
        (act_desc + 52, act_bytes),
        (out_desc + 8, out_bytes),
        (out_desc + 24, rows),
        (out_desc + 28, 1),
        (out_desc + 32, 1),
        (out_desc + 36, 1),
        (out_desc + 40, 4),
        (out_desc + 44, out_bytes),
        (out_desc + 48, out_bytes),
        (out_desc + 52, out_bytes),
    ] {
        put32(
            bytes,
            offset,
            u32::try_from(value).map_err(|_| Error::Contract)?,
        );
    }
    let kp = op_desc + 72;
    put32(bytes, op_desc, 4); // HTP_OP_MUL_MAT, no stub or fusion flags.
    bytes[kp] = 7; // HTP_MM_KERNEL_HVX_QUANT_ROW_N1 (N = 1 < n_threads = 8).
    bytes[kp + 4] = u8::try_from(HMX_Q2_HVX_WORKERS).map_err(|_| Error::Contract)?;
    bytes[kp + 6] = n_prefetch;
    for (offset, value) in [(16, 576), (20, 640), (24, src1_row_size), (28, vtcm_size)] {
        put32(bytes, kp + offset, value);
    }
    for index in 0..5 {
        put32(bytes, kp + 36 + index * 8, 1);
    }
    bytes[op_desc + 200..op_desc + 228].fill(0xff);
    put16(bytes, op_desc + 200, ti_base);
    put16(bytes, op_desc + 202, ti_base + 1);
    put16(bytes, op_desc + 220, ti_base + 2);
    Ok(())
}

#[cfg(test)]
fn output_valid(bytes: &[u8]) -> bool {
    const EXP_MASK: u32 = 0x7f80_0000;
    const EXP_STEP: u32 = 0x0080_0000;
    const SIGN_BIT: u32 = 0x8000_0000;

    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return false;
    }
    let words = bytes.as_chunks::<4>().0;
    let (chunks, tail) = words.as_chunks::<16>();
    for chunk in chunks {
        let mut mask = 0_u32;
        for word in chunk {
            let bits = u32::from_le_bytes(*word);
            mask |= (bits & EXP_MASK).wrapping_add(EXP_STEP);
        }
        if (mask & SIGN_BIT) != 0 {
            return false;
        }
    }
    let mut tail_mask = 0_u32;
    for word in tail {
        let bits = u32::from_le_bytes(*word);
        tail_mask |= (bits & EXP_MASK).wrapping_add(EXP_STEP);
    }
    (tail_mask & SIGN_BIT) == 0
}

pub(crate) fn copy_and_validate_output(bytes: &[u8], output: &mut [f32]) -> bool {
    const EXP_MASK: u32 = 0x7f80_0000;
    const EXP_STEP: u32 = 0x0080_0000;
    const SIGN_BIT: u32 = 0x8000_0000;

    if output.is_empty() || bytes.len() != output.len().saturating_mul(4) {
        output.fill(f32::NAN);
        return false;
    }
    let words = bytes.as_chunks::<4>().0;
    let (in_chunks, in_tail) = words.as_chunks::<16>();
    let (out_chunks, out_tail) = output.as_chunks_mut::<16>();
    for (in_chunk, out_chunk) in in_chunks.iter().zip(out_chunks) {
        let mut mask = 0_u32;
        for (word, dst) in in_chunk.iter().zip(out_chunk) {
            let bits = u32::from_le_bytes(*word);
            mask |= (bits & EXP_MASK).wrapping_add(EXP_STEP);
            *dst = f32::from_bits(bits);
        }
        if (mask & SIGN_BIT) != 0 {
            output.fill(f32::NAN);
            return false;
        }
    }
    let mut tail_mask = 0_u32;
    for (word, dst) in in_tail.iter().zip(out_tail) {
        let bits = u32::from_le_bytes(*word);
        tail_mask |= (bits & EXP_MASK).wrapping_add(EXP_STEP);
        *dst = f32::from_bits(bits);
    }
    if (tail_mask & SIGN_BIT) != 0 {
        output.fill(f32::NAN);
        return false;
    }
    true
}

fn put16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}
fn put32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}
fn put64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

impl From<NativeQ2ProjectionLimits> for hmx_projection::HmxQ2ProjectionLimits {
    fn from(limits: NativeQ2ProjectionLimits) -> Self {
        Self {
            common: limits.common,
            resident_weight_bytes: limits.resident_weight_bytes,
            batch_operations: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hvx_mv_params_matches_signed_ggml_hexagon_solver() -> Result<(), Error> {
        let (prefetch_128, row_128, vtcm_128) = hvx_mv_params(128, 17_408)?;
        assert_eq!(prefetch_128, 16);
        assert_eq!(row_128, 5_120);
        assert_eq!(vtcm_128, 238_592);

        let (prefetch_5120, row_5120, vtcm_5120) = hvx_mv_params(5120, 17_408)?;
        assert_eq!(prefetch_5120, 4);
        assert_eq!(row_5120, 204_800);
        assert_eq!(vtcm_5120, 1_912_832);
        assert!(vtcm_5120 <= HMX_Q2_VTCM_REQUEST_BYTES);

        let (prefetch_6144, row_6144, vtcm_6144) = hvx_mv_params(6144, 5120)?;
        assert_eq!(prefetch_6144, 2);
        assert_eq!(row_6144, 245_760);
        assert_eq!(vtcm_6144, 1_249_280);
        assert!(vtcm_6144 <= HMX_Q2_VTCM_REQUEST_BYTES);

        let (prefetch_17408, row_17408, vtcm_17408) = hvx_mv_params(17_408, 5120)?;
        assert_eq!(prefetch_17408, 2);
        assert_eq!(row_17408, 696_320);
        assert_eq!(vtcm_17408, 3_502_080);
        assert!(vtcm_17408 <= NATIVE_Q2_VTCM_REQUEST_BYTES);
        Ok(())
    }

    #[test]
    fn native_q2_limits_and_resident_registry_validate_scales_and_extents() -> Result<(), Error> {
        let limits =
            NativeQ2ProjectionLimits::new(17_408, 17_408, 24_000, Duration::from_secs(60))?
                .with_resident_weights(NATIVE_Q2_MAX_RESIDENT_WEIGHT_BYTES)?;
        assert_eq!(limits.maximum_columns(), 17_408);
        assert_eq!(limits.maximum_rows(), 17_408);
        assert_eq!(
            limits.resident_weight_bytes(),
            NATIVE_Q2_MAX_RESIDENT_WEIGHT_BYTES
        );
        assert_eq!(
            limits.allocation_bytes(),
            NATIVE_Q2_MAX_WORKING_BYTES + NATIVE_Q2_MAX_RESIDENT_WEIGHT_BYTES
        );

        let mut registry = NativeQ2WeightRegistry::new(4096)?;
        let mut weights = vec![0x55_u8; layout(128, 32).weights];
        for tile in weights.as_chunks_mut::<576>().0 {
            for scale in tile[512..].as_chunks_mut::<2>().0 {
                *scale = 0x3800_u16.to_le_bytes(); // 0.5 in FP16 (non-unit finite scale).
            }
        }
        let handle = registry
            .prepare(128, 32, &weights)?
            .ok_or(Error::Contract)?;
        assert_eq!(handle.columns(), 128);
        assert_eq!(handle.rows(), 32);
        assert_eq!(handle.packed_bytes(), 1152);

        // NaN FP16 scale must reject in every lane of the 64-byte scale block.
        for lane in 0..32 {
            let mut bad = weights.clone();
            bad[512 + lane * 2..514 + lane * 2].copy_from_slice(&0x7e00_u16.to_le_bytes());
            assert!(registry.prepare(128, 32, &bad).is_err());
            bad[512 + lane * 2..514 + lane * 2].copy_from_slice(&0xfc00_u16.to_le_bytes());
            assert!(registry.prepare(128, 32, &bad).is_err());
        }
        Ok(())
    }

    #[test]
    fn coalesced_finiteness_and_batch_resident_descriptors_preserve_contract() -> Result<(), Error>
    {
        for len in [1, 15, 16, 17, 32, 128] {
            let mut values = vec![-1.25_f32; len];
            assert!(is_finite_f32_slice(&values));
            let mut bytes = Vec::with_capacity(len * 4);
            for &v in &values {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            assert!(output_valid(&bytes));
            let mut copied = vec![0.0_f32; len];
            assert!(copy_and_validate_output(&bytes, &mut copied));
            assert_eq!(copied, values);
            for pos in [0, len / 2, len - 1] {
                values[pos] = f32::NAN;
                assert!(!is_finite_f32_slice(&values));
                values[pos] = f32::NEG_INFINITY;
                assert!(!is_finite_f32_slice(&values));
                values[pos] = 0.5;
                let mut bad_bytes = bytes.clone();
                bad_bytes[pos * 4..pos * 4 + 4].copy_from_slice(&f32::INFINITY.to_le_bytes());
                assert!(!output_valid(&bad_bytes));
                assert!(!copy_and_validate_output(&bad_bytes, &mut copied));
            }
        }

        let mut registry = NativeQ2WeightRegistry::new(16_384)?;
        let mut weights = vec![0x55_u8; layout(128, 32).weights];
        for tile in weights.as_chunks_mut::<576>().0 {
            for scale in tile[512..].as_chunks_mut::<2>().0 {
                *scale = 0x3c00_u16.to_le_bytes();
            }
        }
        let h0 = registry
            .prepare(128, 32, &weights)?
            .ok_or(Error::Contract)?;
        let h1 = registry
            .prepare(128, 32, &weights)?
            .ok_or(Error::Contract)?;
        let arena_start = 8192;
        let mut arena = vec![0_u8; arena_start + 4096];
        let activation = vec![1.5_f32; 128];
        let batch = image_batch_resident(
            &mut arena,
            0x1000_0000,
            7,
            arena_start,
            [&h0, &h1],
            &activation,
        )?;
        assert_eq!(batch.output_offsets, [2560, 2688]);
        assert_eq!(batch.output_end, 2816);
        assert_eq!(
            u64::from_le_bytes(arena[0..8].try_into().map_err(|_| Error::Contract)?),
            0x1000_0000
        );
        assert_eq!(
            u32::from_le_bytes(arena[20..24].try_into().map_err(|_| Error::Contract)?),
            7
        );
        // Member 1 tensor indices are 3, 4, 5 and op_desc starts at 24 + 6*56 + 232 = 592.
        assert_eq!(
            u16::from_le_bytes(
                arena[24 + 3 * 56 + 22..24 + 3 * 56 + 24]
                    .try_into()
                    .map_err(|_| Error::Contract)?
            ),
            3
        );
        assert_eq!(
            u16::from_le_bytes(
                arena[592 + 200..592 + 202]
                    .try_into()
                    .map_err(|_| Error::Contract)?
            ),
            3
        );
        assert_eq!(
            u16::from_le_bytes(
                arena[592 + 202..592 + 204]
                    .try_into()
                    .map_err(|_| Error::Contract)?
            ),
            4
        );
        assert_eq!(
            u16::from_le_bytes(
                arena[592 + 220..592 + 222]
                    .try_into()
                    .map_err(|_| Error::Contract)?
            ),
            5
        );
        Ok(())
    }
}
