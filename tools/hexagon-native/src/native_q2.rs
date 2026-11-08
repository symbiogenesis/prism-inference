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
/// Largest immutable weight payload in a single `FastRPC` buffer (`480 MiB`, within the `512 MiB` `rpcmem` `m-size` limit).
pub const NATIVE_Q2_MAX_ARENA_WEIGHT_BYTES: usize = 480 * 1024 * 1024;
/// Exact even-arena 28-projection weight step (`479.53125 MiB`, page-aligned) used when partitioning multi-arena resident weights.
pub const NATIVE_Q2_ARENA_WEIGHT_STEP_BYTES: usize = 502_824_960;
/// Exact odd-arena 28-projection weight step (`438.75 MiB`, page-aligned) completing each 56-projection (9-layer) pair.
pub const NATIVE_Q2_ARENA_ODD_WEIGHT_STEP_BYTES: usize = 460_062_720;
/// Maximum number of host-mapped `FastRPC` weight arenas in a single native `Q2_0` session.
pub const NATIVE_Q2_MAX_WEIGHT_ARENAS: usize = 6;
/// Largest aggregate immutable host-mapped weight capacity across all arenas in a single native `Q2_0` session (`2,880 MiB`).
pub const NATIVE_Q2_MAX_RESIDENT_WEIGHT_BYTES: usize =
    NATIVE_Q2_MAX_WEIGHT_ARENAS * NATIVE_Q2_MAX_ARENA_WEIGHT_BYTES;
/// Fixed 4 MiB VTCM request used by the 8-HVX full-`K` `Q2_0` session (`v81` hardware has 8 MiB VTCM).
pub const NATIVE_Q2_VTCM_REQUEST_BYTES: u32 = 4 * 1024 * 1024;

pub(crate) const fn arena_capacities(
    resident_weight_bytes: usize,
) -> Option<([usize; NATIVE_Q2_MAX_WEIGHT_ARENAS], usize)> {
    if resident_weight_bytes > NATIVE_Q2_MAX_RESIDENT_WEIGHT_BYTES {
        return None;
    }
    if resident_weight_bytes <= NATIVE_Q2_MAX_ARENA_WEIGHT_BYTES {
        let mut capacities = [0_usize; NATIVE_Q2_MAX_WEIGHT_ARENAS];
        capacities[0] = resident_weight_bytes;
        return Some((capacities, 1));
    }
    let use_alternating_steps = resident_weight_bytes
        <= (NATIVE_Q2_MAX_WEIGHT_ARENAS / 2)
            * (NATIVE_Q2_ARENA_WEIGHT_STEP_BYTES + NATIVE_Q2_ARENA_ODD_WEIGHT_STEP_BYTES);
    let mut capacities = [0_usize; NATIVE_Q2_MAX_WEIGHT_ARENAS];
    let mut remaining = resident_weight_bytes;
    let mut count = 0_usize;
    while remaining > 0 {
        if count >= NATIVE_Q2_MAX_WEIGHT_ARENAS {
            return None;
        }
        let step = if use_alternating_steps {
            if count.is_multiple_of(2) {
                NATIVE_Q2_ARENA_WEIGHT_STEP_BYTES
            } else {
                NATIVE_Q2_ARENA_ODD_WEIGHT_STEP_BYTES
            }
        } else {
            NATIVE_Q2_MAX_ARENA_WEIGHT_BYTES
        };
        let chunk = if remaining > step { step } else { remaining };
        capacities[count] = chunk;
        remaining -= chunk;
        count += 1;
    }
    Some((capacities, count))
}

const fn arena_allocation_bytes_opt(
    columns: usize,
    rows: usize,
    resident_weight_bytes: usize,
) -> Option<([usize; NATIVE_Q2_MAX_WEIGHT_ARENAS], usize)> {
    let Some((capacities, count)) = arena_capacities(resident_weight_bytes) else {
        return None;
    };
    let working = working_allocation(columns, rows);
    let mut allocations = [0_usize; NATIVE_Q2_MAX_WEIGHT_ARENAS];
    allocations[0] = (working + capacities[0]).next_multiple_of(4096);
    let mut index = 1_usize;
    while index < count {
        allocations[index] = capacities[index].next_multiple_of(4096);
        index += 1;
    }
    Some((allocations, count))
}

pub(crate) fn arena_allocation_bytes(
    columns: usize,
    rows: usize,
    resident_weight_bytes: usize,
) -> Result<([usize; NATIVE_Q2_MAX_WEIGHT_ARENAS], usize), Error> {
    arena_allocation_bytes_opt(columns, rows, resident_weight_bytes).ok_or(Error::Contract)
}

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

    /// Reserve optional immutable host-mapped weight arenas in the session.
    ///
    /// # Errors
    /// Rejects more than `NATIVE_Q2_MAX_RESIDENT_WEIGHT_BYTES` before opening a module or allocating memory.
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

    /// Maximum mapped host allocation across all session arenas, including command, activation, output and weights.
    #[must_use]
    pub const fn allocation_bytes(self) -> usize {
        if let Some((allocations, count)) =
            arena_allocation_bytes_opt(self.columns, self.common.rows, self.resident_weight_bytes)
        {
            let mut total = 0_usize;
            let mut index = 0_usize;
            while index < count {
                total += allocations[index];
                index += 1;
            }
            total
        } else {
            let working = working_allocation(self.columns, self.common.rows);
            (working + self.resident_weight_bytes).next_multiple_of(4096)
        }
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
    arena_index: u16,
    range: Range<usize>,
    columns: usize,
    rows: usize,
}

impl NativeQ2WeightHandle {
    /// Zero-based index of the session weight arena (`0..4`) holding this weight extent.
    #[must_use]
    pub const fn arena_index(&self) -> u16 {
        self.arena_index
    }

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

/// Metadata-only append-only allocator for one submitter's native `Q2_0` weight arenas.
#[derive(Clone, Copy)]
pub struct NativeQ2WeightRegistry {
    owner: u64,
    capacities: [usize; NATIVE_Q2_MAX_WEIGHT_ARENAS],
    used: [usize; NATIVE_Q2_MAX_WEIGHT_ARENAS],
    arena_count: usize,
    active_arena: usize,
    total_used: usize,
}

impl NativeQ2WeightRegistry {
    /// Create one independent owner with an explicitly admitted payload bound.
    ///
    /// # Errors
    /// Rejects capacity above `NATIVE_Q2_MAX_RESIDENT_WEIGHT_BYTES`.
    pub fn new(capacity_bytes: usize) -> Result<Self, Error> {
        let (capacities, arena_count) = arena_capacities(capacity_bytes).ok_or(Error::Contract)?;
        Ok(Self {
            owner: NEXT_NATIVE_REGISTRY_OWNER.fetch_add(1, Ordering::Relaxed),
            capacities,
            used: [0; NATIVE_Q2_MAX_WEIGHT_ARENAS],
            arena_count,
            active_arena: 0,
            total_used: 0,
        })
    }

    /// Remaining payload extent from the active arena onward, including alignment charged by prior reservations.
    #[must_use]
    pub const fn remaining_bytes(&self) -> usize {
        let mut remaining = 0_usize;
        let mut index = self.active_arena;
        while index < self.arena_count {
            remaining += self.capacities[index].saturating_sub(self.used[index]);
            index += 1;
        }
        remaining
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
        let mut arena_idx = self.active_arena;
        while arena_idx < self.arena_count {
            let start = self.used[arena_idx].next_multiple_of(128);
            let end = start.checked_add(weight_bytes).ok_or(Error::Contract)?;
            if end <= self.capacities[arena_idx] {
                let arena_index = u16::try_from(arena_idx).map_err(|_| Error::Contract)?;
                self.active_arena = arena_idx;
                self.used[arena_idx] = end;
                self.total_used = self.used.iter().sum();
                return Ok(Some(NativeQ2WeightHandle {
                    owner: self.owner,
                    arena_index,
                    range: start..end,
                    columns,
                    rows,
                }));
            }
            arena_idx += 1;
        }
        Ok(None)
    }

    /// Total bytes across all immutable arenas currently populated (including internal alignment padding).
    #[must_use]
    pub const fn used_bytes(&self) -> usize {
        self.total_used
    }

    /// Bytes of the specified arena (`0..NATIVE_Q2_MAX_WEIGHT_ARENAS`) currently populated.
    #[must_use]
    pub const fn arena_used_bytes(&self, arena_index: usize) -> usize {
        if arena_index < NATIVE_Q2_MAX_WEIGHT_ARENAS {
            self.used[arena_index]
        } else {
            0
        }
    }

    /// Recover an owner-checked byte range inside this registry's arena.
    ///
    /// # Errors
    /// Rejects handles issued by any other registry, including earlier sessions.
    pub fn range(&self, handle: &NativeQ2WeightHandle) -> Result<Range<usize>, Error> {
        let arena_idx = usize::from(handle.arena_index);
        if self.owner != handle.owner
            || arena_idx >= self.arena_count
            || handle.range.end > self.used[arena_idx]
            || handle.range.end > self.capacities[arena_idx]
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

    /// Submit two resident `HTP_TYPE_Q2_0` projections with per-operation `F32`
    /// activation vectors in a single 2-op DSP queue submission.
    ///
    /// # Errors
    /// Unsupported residency, foreign handles, mismatched input columns or
    /// submission failures reject and close the session with no CPU replay.
    fn submit_resident_multi2(
        &mut self,
        _handles: [&NativeQ2WeightHandle; 2],
        _activations: [&[f32]; 2],
        mut outputs: [&mut [f32]; 2],
    ) -> Result<(), Error> {
        for output in &mut outputs {
            output.fill(f32::NAN);
        }
        Err(Error::Contract)
    }

    /// Submit three resident `HTP_TYPE_Q2_0` projections with per-operation `F32`
    /// activation vectors in a single 3-op DSP queue submission.
    ///
    /// # Errors
    /// Unsupported residency, foreign handles, mismatched input columns or
    /// submission failures reject and close the session with no CPU replay.
    fn submit_resident_multi3(
        &mut self,
        _handles: [&NativeQ2WeightHandle; 3],
        _activations: [&[f32]; 3],
        mut outputs: [&mut [f32]; 3],
    ) -> Result<(), Error> {
        for output in &mut outputs {
            output.fill(f32::NAN);
        }
        Err(Error::Contract)
    }

    /// Submit four resident `HTP_TYPE_Q2_0` projections with per-operation `F32`
    /// activation vectors in a single 4-op DSP queue submission.
    ///
    /// # Errors
    /// Unsupported residency, foreign handles, mismatched input columns or
    /// submission failures reject and close the session with no CPU replay.
    fn submit_resident_multi4(
        &mut self,
        _handles: [&NativeQ2WeightHandle; 4],
        _activations: [&[f32]; 4],
        mut outputs: [&mut [f32]; 4],
    ) -> Result<(), Error> {
        for output in &mut outputs {
            output.fill(f32::NAN);
        }
        Err(Error::Contract)
    }

    /// Submit eight resident `HTP_TYPE_Q2_0` projections with per-operation `F32`
    /// activation vectors in a single 8-op DSP queue submission (or two 4-op
    /// submissions when the submitter only implements 4-op batching).
    ///
    /// # Errors
    /// Unsupported residency, foreign handles, mismatched input columns or
    /// submission failures reject and close the session with no CPU replay.
    fn submit_resident_multi8(
        &mut self,
        handles: [&NativeQ2WeightHandle; 8],
        activations: [&[f32]; 8],
        outputs: [&mut [f32]; 8],
    ) -> Result<(), Error> {
        let [o0, o1, o2, o3, o4, o5, o6, o7] = outputs;
        self.submit_resident_multi4(
            [handles[0], handles[1], handles[2], handles[3]],
            [
                activations[0],
                activations[1],
                activations[2],
                activations[3],
            ],
            [o0, o1, o2, o3],
        )?;
        self.submit_resident_multi4(
            [handles[4], handles[5], handles[6], handles[7]],
            [
                activations[4],
                activations[5],
                activations[6],
                activations[7],
            ],
            [o4, o5, o6, o7],
        )
    }

    /// Submit a fused 3-operation `ffn_gate` + `ffn_up` + `HTP_OP_GLU_SWIGLU`
    /// chain in a single DSP queue submission with VTCM intermediate residency.
    ///
    /// # Errors
    /// Unsupported residency, foreign handles, mismatched dimensions or
    /// submission failures reject and close the session with no CPU replay.
    fn submit_resident_ffn_swiglu(
        &mut self,
        _gate_handle: &NativeQ2WeightHandle,
        _up_handle: &NativeQ2WeightHandle,
        _activation: &[f32],
        output: &mut [f32],
    ) -> Result<(), Error> {
        output.fill(f32::NAN);
        Err(Error::Contract)
    }

    /// Submit two tokens of fused `ffn_gate` + `ffn_up` + `HTP_OP_GLU_SWIGLU`
    /// chains in a single 6-operation DSP queue submission with per-token VTCM
    /// intermediate residency (or two 3-op submissions when the submitter only
    /// implements single-token fused FFN `SwiGLU`).
    ///
    /// # Errors
    /// Unsupported residency, foreign handles, mismatched dimensions or
    /// submission failures reject and close the session with no CPU replay.
    fn submit_resident_ffn_swiglu_multi2(
        &mut self,
        gate_handle: &NativeQ2WeightHandle,
        up_handle: &NativeQ2WeightHandle,
        activations: [&[f32]; 2],
        outputs: [&mut [f32]; 2],
    ) -> Result<(), Error> {
        let [o0, o1] = outputs;
        self.submit_resident_ffn_swiglu(gate_handle, up_handle, activations[0], o0)?;
        self.submit_resident_ffn_swiglu(gate_handle, up_handle, activations[1], o1)
    }

    /// Submit a full intra-layer FFN chain (`[gate_handle, up_handle, down_handle]`:
    /// `ffn_gate` + `ffn_up` + `HTP_OP_GLU_SWIGLU` + optional `HTP_OP_HADAMARD` +
    /// `ffn_down` + optional `HTP_OP_ADD` residual) with VTCM intermediate residency chaining.
    ///
    /// # Errors
    /// Unsupported residency, foreign handles, mismatched dimensions or
    /// submission failures reject and close the session with no CPU replay.
    fn submit_resident_ffn_chain(
        &mut self,
        _handles: [&NativeQ2WeightHandle; 3],
        _activation: &[f32],
        _residual: Option<&[f32]>,
        _hadamard_block_size: Option<u32>,
        output: &mut [f32],
    ) -> Result<(), Error> {
        output.fill(f32::NAN);
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

    fn submit_resident_multi2(
        &mut self,
        handles: [&NativeQ2WeightHandle; 2],
        activations: [&[f32]; 2],
        outputs: [&mut [f32]; 2],
    ) -> Result<(), Error> {
        (**self).submit_resident_multi2(handles, activations, outputs)
    }

    fn submit_resident_multi3(
        &mut self,
        handles: [&NativeQ2WeightHandle; 3],
        activations: [&[f32]; 3],
        outputs: [&mut [f32]; 3],
    ) -> Result<(), Error> {
        (**self).submit_resident_multi3(handles, activations, outputs)
    }

    fn submit_resident_multi4(
        &mut self,
        handles: [&NativeQ2WeightHandle; 4],
        activations: [&[f32]; 4],
        outputs: [&mut [f32]; 4],
    ) -> Result<(), Error> {
        (**self).submit_resident_multi4(handles, activations, outputs)
    }

    fn submit_resident_multi8(
        &mut self,
        handles: [&NativeQ2WeightHandle; 8],
        activations: [&[f32]; 8],
        outputs: [&mut [f32]; 8],
    ) -> Result<(), Error> {
        (**self).submit_resident_multi8(handles, activations, outputs)
    }

    fn submit_resident_ffn_swiglu(
        &mut self,
        gate_handle: &NativeQ2WeightHandle,
        up_handle: &NativeQ2WeightHandle,
        activation: &[f32],
        output: &mut [f32],
    ) -> Result<(), Error> {
        (**self).submit_resident_ffn_swiglu(gate_handle, up_handle, activation, output)
    }

    fn submit_resident_ffn_swiglu_multi2(
        &mut self,
        gate_handle: &NativeQ2WeightHandle,
        up_handle: &NativeQ2WeightHandle,
        activations: [&[f32]; 2],
        outputs: [&mut [f32]; 2],
    ) -> Result<(), Error> {
        (**self).submit_resident_ffn_swiglu_multi2(gate_handle, up_handle, activations, outputs)
    }

    fn submit_resident_ffn_chain(
        &mut self,
        handles: [&NativeQ2WeightHandle; 3],
        activation: &[f32],
        residual: Option<&[f32]>,
        hadamard_block_size: Option<u32>,
        output: &mut [f32],
    ) -> Result<(), Error> {
        (**self).submit_resident_ffn_chain(
            handles,
            activation,
            residual,
            hadamard_block_size,
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
    let mut m0 = 0_u32;
    let mut m1 = 0_u32;
    let mut m2 = 0_u32;
    let mut m3 = 0_u32;
    for chunk in chunks {
        m0 |= (chunk[0].to_bits() & EXP_MASK).wrapping_add(EXP_STEP)
            | (chunk[1].to_bits() & EXP_MASK).wrapping_add(EXP_STEP)
            | (chunk[2].to_bits() & EXP_MASK).wrapping_add(EXP_STEP)
            | (chunk[3].to_bits() & EXP_MASK).wrapping_add(EXP_STEP);
        m1 |= (chunk[4].to_bits() & EXP_MASK).wrapping_add(EXP_STEP)
            | (chunk[5].to_bits() & EXP_MASK).wrapping_add(EXP_STEP)
            | (chunk[6].to_bits() & EXP_MASK).wrapping_add(EXP_STEP)
            | (chunk[7].to_bits() & EXP_MASK).wrapping_add(EXP_STEP);
        m2 |= (chunk[8].to_bits() & EXP_MASK).wrapping_add(EXP_STEP)
            | (chunk[9].to_bits() & EXP_MASK).wrapping_add(EXP_STEP)
            | (chunk[10].to_bits() & EXP_MASK).wrapping_add(EXP_STEP)
            | (chunk[11].to_bits() & EXP_MASK).wrapping_add(EXP_STEP);
        m3 |= (chunk[12].to_bits() & EXP_MASK).wrapping_add(EXP_STEP)
            | (chunk[13].to_bits() & EXP_MASK).wrapping_add(EXP_STEP)
            | (chunk[14].to_bits() & EXP_MASK).wrapping_add(EXP_STEP)
            | (chunk[15].to_bits() & EXP_MASK).wrapping_add(EXP_STEP);
    }
    let mut tail_mask = (m0 | m1) | (m2 | m3);
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HtpBufferDesc {
    pub(crate) base: u64,
    pub(crate) size: usize,
    pub(crate) fd: u32,
}

#[derive(Clone, Copy)]
pub(crate) struct ResidentLayout {
    pub(crate) output: usize,
    pub(crate) n_bufs: u32,
}

pub(crate) fn image_resident(
    bytes: &mut [u8],
    base: u64,
    fd: u32,
    arena_start: usize,
    extra_arenas: &[HtpBufferDesc],
    handle: &NativeQ2WeightHandle,
    activation: &[f32],
) -> Result<ResidentLayout, Error> {
    let allocation = bytes.len();
    let active = layout(handle.columns, handle.rows);
    if handle.packed_bytes() != active.weights || activation.len() != handle.columns {
        return Err(Error::Contract);
    }
    if handle.arena_index == 0 {
        let weight_offset = arena_start
            .checked_add(handle.range.start)
            .ok_or(Error::Contract)?;
        let weight_end = arena_start
            .checked_add(handle.range.end)
            .ok_or(Error::Contract)?;
        if weight_end > allocation {
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
        return Ok(ResidentLayout {
            output: active.output,
            n_bufs: 1,
        });
    }
    let extra_idx = usize::from(handle.arena_index - 1);
    let extra = extra_arenas
        .get(extra_idx)
        .copied()
        .ok_or(Error::Contract)?;
    if handle.range.end > extra.size {
        return Err(Error::Contract);
    }
    let working = bytes.get_mut(..arena_start).ok_or(Error::Contract)?;
    let act_bytes = handle.columns * 4;
    let out_bytes = handle.rows * 4;
    if active.output + out_bytes > working.len() {
        return Err(Error::Contract);
    }
    working[..1024].fill(0);
    put64(working, 0, base);
    put64(
        working,
        8,
        u64::try_from(allocation).map_err(|_| Error::Contract)?,
    );
    put32(working, 20, fd);
    put64(working, 24, extra.base);
    put64(
        working,
        32,
        u64::try_from(extra.size).map_err(|_| Error::Contract)?,
    );
    put32(working, 44, extra.fd);
    write_member_descriptors(
        working,
        0,
        1,
        48,
        216,
        handle.columns,
        handle.rows,
        handle.range.start,
        active.weights,
        active.activation,
        active.output,
    )?;
    for (dst, &value) in working[active.activation..active.activation + act_bytes]
        .as_chunks_mut::<4>()
        .0[..activation.len()]
        .iter_mut()
        .zip(activation)
    {
        *dst = value.to_bits().to_le_bytes();
    }
    working[active.output..active.output + out_bytes]
        .as_chunks_mut::<4>()
        .0
        .fill(0x7fc0_ae51_u32.to_le_bytes());
    Ok(ResidentLayout {
        output: active.output,
        n_bufs: 2,
    })
}

#[derive(Clone, Copy)]
pub(crate) struct BatchLayout<const B: usize> {
    pub(crate) output_offsets: [usize; B],
    pub(crate) output_end: usize,
    pub(crate) n_bufs: u32,
}

#[cfg(test)]
pub(crate) fn image_batch_resident<const B: usize>(
    bytes: &mut [u8],
    base: u64,
    fd: u32,
    arena_start: usize,
    extra_arenas: &[HtpBufferDesc],
    handles: [&NativeQ2WeightHandle; B],
    activation: &[f32],
) -> Result<BatchLayout<B>, Error> {
    if !matches!(B, 2 | 3) {
        return Err(Error::Contract);
    }
    image_multi_resident(
        bytes,
        base,
        fd,
        arena_start,
        extra_arenas,
        handles,
        [activation; B],
    )
}

pub(crate) fn image_multi_resident<const B: usize>(
    bytes: &mut [u8],
    base: u64,
    fd: u32,
    arena_start: usize,
    extra_arenas: &[HtpBufferDesc],
    handles: [&NativeQ2WeightHandle; B],
    activations: [&[f32]; B],
) -> Result<BatchLayout<B>, Error> {
    if !matches!(B, 2..=4 | 8) {
        return Err(Error::Contract);
    }
    let cmd_pad = if B > 4 { 4096 } else { 2048 };
    image_multi_resident_with_pad(
        bytes,
        base,
        fd,
        arena_start,
        extra_arenas,
        handles,
        activations,
        cmd_pad,
    )
}

#[allow(
    clippy::too_many_lines,
    clippy::too_many_arguments,
    reason = "Keep multi-arena batch descriptor assembly and poison region together"
)]
fn image_multi_resident_with_pad<const B: usize>(
    bytes: &mut [u8],
    base: u64,
    fd: u32,
    arena_start: usize,
    extra_arenas: &[HtpBufferDesc],
    handles: [&NativeQ2WeightHandle; B],
    activations: [&[f32]; B],
    cmd_pad: usize,
) -> Result<BatchLayout<B>, Error> {
    if !matches!(B, 2..=4 | 8) || !matches!(cmd_pad, 2048 | 4096) {
        return Err(Error::Contract);
    }
    let allocation = bytes.len();
    let columns = handles[0].columns;
    let act_bytes = columns * 4;
    let mut next_offset = cmd_pad;
    let mut act_offsets = [0_usize; B];
    let mut act_copy_member = [false; B];
    for (member, &activation) in activations.iter().enumerate() {
        if activation.len() != columns {
            return Err(Error::Contract);
        }
        if let Some(prev) =
            (0..member).find(|&idx| std::ptr::eq(activations[idx].as_ptr(), activation.as_ptr()))
        {
            act_offsets[member] = act_offsets[prev];
        } else {
            act_offsets[member] = next_offset;
            act_copy_member[member] = true;
            next_offset = (next_offset + act_bytes).next_multiple_of(128);
        }
    }
    let mut next_out = next_offset;
    let mut weight_offsets = [0_usize; B];
    let mut weight_bis = [0_u16; B];
    let mut output_offsets = [0_usize; B];
    let mut batch_extra_arena_indices = [0_u16; B];
    let mut batch_extra_descs = [HtpBufferDesc {
        base: 0,
        size: 0,
        fd: 0,
    }; B];
    let mut extra_buf_count = 0_usize;
    for (member, handle) in handles.into_iter().enumerate() {
        let expected_weights = layout(columns, handle.rows).weights;
        if handle.columns != columns || handle.packed_bytes() != expected_weights {
            return Err(Error::Contract);
        }
        if handle.arena_index == 0 {
            let start = arena_start
                .checked_add(handle.range.start)
                .ok_or(Error::Contract)?;
            let end = arena_start
                .checked_add(handle.range.end)
                .ok_or(Error::Contract)?;
            if end > allocation {
                return Err(Error::Contract);
            }
            weight_offsets[member] = start;
            weight_bis[member] = 0;
        } else {
            let extra_idx = usize::from(handle.arena_index - 1);
            let extra = extra_arenas
                .get(extra_idx)
                .copied()
                .ok_or(Error::Contract)?;
            if handle.range.end > extra.size {
                return Err(Error::Contract);
            }
            let bi = if let Some(pos) = batch_extra_arena_indices[..extra_buf_count]
                .iter()
                .position(|&idx| idx == handle.arena_index)
            {
                u16::try_from(pos + 1).map_err(|_| Error::Contract)?
            } else {
                batch_extra_arena_indices[extra_buf_count] = handle.arena_index;
                batch_extra_descs[extra_buf_count] = extra;
                extra_buf_count += 1;
                u16::try_from(extra_buf_count).map_err(|_| Error::Contract)?
            };
            weight_offsets[member] = handle.range.start;
            weight_bis[member] = bi;
        }
        output_offsets[member] = next_out;
        next_out = (next_out + handle.rows * 4).next_multiple_of(128);
    }
    let output_end = output_offsets[B - 1] + handles[B - 1].rows * 4;
    let working = bytes.get_mut(..arena_start).ok_or(Error::Contract)?;
    if output_end > working.len() {
        return Err(Error::Contract);
    }
    let n_bufs_usize = 1 + extra_buf_count;
    let n_bufs = u32::try_from(n_bufs_usize).map_err(|_| Error::Contract)?;
    working[..cmd_pad].fill(0);
    put64(working, 0, base);
    put64(
        working,
        8,
        u64::try_from(allocation).map_err(|_| Error::Contract)?,
    );
    put32(working, 20, fd);
    for (idx, extra) in batch_extra_descs[..extra_buf_count].iter().enumerate() {
        let buf_offset = (idx + 1) * 24;
        put64(working, buf_offset, extra.base);
        put64(
            working,
            buf_offset + 8,
            u64::try_from(extra.size).map_err(|_| Error::Contract)?,
        );
        put32(working, buf_offset + 20, extra.fd);
    }
    let tensors_base = n_bufs_usize * 24;
    let operations_base = tensors_base + B * 3 * 56;
    for (member, handle) in handles.into_iter().enumerate() {
        let ti_base = u16::try_from(member * 3).map_err(|_| Error::Contract)?;
        let weight_desc = tensors_base + (member * 3) * 56;
        let operation_desc = operations_base + member * 232;
        write_member_descriptors(
            working,
            ti_base,
            weight_bis[member],
            weight_desc,
            operation_desc,
            columns,
            handle.rows,
            weight_offsets[member],
            layout(columns, handle.rows).weights,
            act_offsets[member],
            output_offsets[member],
        )?;
    }
    for (member, &activation) in activations.iter().enumerate() {
        if !act_copy_member[member] {
            continue;
        }
        let act_offset = act_offsets[member];
        for (dst, &value) in working[act_offset..act_offset + act_bytes]
            .as_chunks_mut::<4>()
            .0[..activation.len()]
            .iter_mut()
            .zip(activation)
        {
            *dst = value.to_bits().to_le_bytes();
        }
    }
    working[output_offsets[0]..output_end]
        .as_chunks_mut::<4>()
        .0
        .fill(0x7fc0_ae51_u32.to_le_bytes());
    Ok(BatchLayout {
        output_offsets,
        output_end,
        n_bufs,
    })
}

#[derive(Clone, Copy)]
pub(crate) struct FfnSwigluLayout {
    pub(crate) output_offset: usize,
    pub(crate) output_end: usize,
    pub(crate) n_bufs: u32,
}

/// Build a fused 3-operation `dspqueue` packet (`OP_MUL_MAT(gate)` +
/// `OP_MUL_MAT(up)` + `HTP_OP_GLU_SWIGLU(gate_out, up_out) -> swiglu_out`)
/// with VTCM intermediate residency chaining (`KEEP_DST_IN_VTCM` +
/// `READ_SRC0_FROM_VTCM | READ_SRC1_FROM_VTCM`).
#[allow(
    clippy::too_many_lines,
    clippy::too_many_arguments,
    reason = "Keep 2x Q2_0 MulMat + 1x GluSwiglu descriptor assembly and VTCM pass flags together"
)]
pub(crate) fn image_ffn_swiglu_resident(
    bytes: &mut [u8],
    base: u64,
    fd: u32,
    arena_start: usize,
    extra_arenas: &[HtpBufferDesc],
    gate_handle: &NativeQ2WeightHandle,
    up_handle: &NativeQ2WeightHandle,
    activation: &[f32],
) -> Result<FfnSwigluLayout, Error> {
    let columns = gate_handle.columns;
    let rows = gate_handle.rows;
    if up_handle.columns != columns
        || up_handle.rows != rows
        || activation.len() != columns
        || !rows.is_multiple_of(32)
    {
        return Err(Error::Contract);
    }
    let batch = image_multi_resident::<2>(
        bytes,
        base,
        fd,
        arena_start,
        extra_arenas,
        [gate_handle, up_handle],
        [activation, activation],
    )?;
    let out_bytes = rows * 4;
    let gate_out_offset = batch.output_offsets[0];
    let up_out_offset = batch.output_offsets[1];
    let swiglu_out_offset = batch.output_end.next_multiple_of(128);
    let output_end = swiglu_out_offset + out_bytes;
    let working = bytes.get_mut(..arena_start).ok_or(Error::Contract)?;
    if output_end > working.len() {
        return Err(Error::Contract);
    }
    let n_bufs_usize = usize::try_from(batch.n_bufs).map_err(|_| Error::Contract)?;
    // Re-layout tensors (9 descriptors = 504 bytes) and operations (3 descriptors = 696 bytes)
    // inside CMD_PAD (2048 bytes) so n_tensors = 9, n_ops = 3.
    let tensors_base = n_bufs_usize * 24;
    let old_ops_base = tensors_base + 6 * 56;
    let new_ops_base = tensors_base + 9 * 56;
    working.copy_within(old_ops_base..old_ops_base + 2 * 232, new_ops_base);
    working[tensors_base + 6 * 56..new_ops_base].fill(0);

    // Chained VTCM flags on Op 0 (gate) and Op 1 (up) when 3 * out_bytes fits inside 1 MiB of VTCM.
    let vtcm_chain = out_bytes.saturating_mul(3) <= 1_048_576;
    let matmul_pass = if vtcm_chain {
        crate::htp_ops::HtpOpPassFlags::KEEP_DST_IN_VTCM
    } else {
        crate::htp_ops::HtpOpPassFlags::NONE
    };
    let glu_pass = if vtcm_chain {
        crate::htp_ops::HtpOpPassFlags::READ_SRC0_FROM_VTCM
            .union(crate::htp_ops::HtpOpPassFlags::READ_SRC1_FROM_VTCM)
    } else {
        crate::htp_ops::HtpOpPassFlags::NONE
    };
    put32(working, new_ops_base + 4, matmul_pass.bits());
    put32(working, new_ops_base + 232 + 4, matmul_pass.bits());

    // Shape 1D [rows] F32 vectors as 2D [glu_cols, glu_rows] with glu_rows >= 2 for solve_glu_kparams.
    let (glu_cols, glu_rows) = if rows >= 64 {
        (32_usize, rows / 32)
    } else {
        (rows / 2, 2_usize)
    };
    let glu_cols_u32 = u32::try_from(glu_cols).map_err(|_| Error::Contract)?;
    let glu_rows_u32 = u32::try_from(glu_rows).map_err(|_| Error::Contract)?;
    let glu_kp = crate::htp_ops::solve_glu_kparams(glu_cols_u32, glu_rows_u32)?;
    let glu_row_bytes = glu_cols * 4;

    for (ti_offset, buf_offset) in [
        (6_u16, gate_out_offset),
        (7_u16, up_out_offset),
        (8_u16, swiglu_out_offset),
    ] {
        let desc = tensors_base + usize::from(ti_offset) * 56;
        put64(
            working,
            desc,
            u64::try_from(buf_offset).map_err(|_| Error::Contract)?,
        );
        put16(working, desc + 20, 0);
        put16(working, desc + 22, ti_offset);
        for (off, val) in [
            (desc + 8, out_bytes),
            (desc + 12, 0),
            (desc + 16, 0), // F32
            (desc + 24, glu_cols),
            (desc + 28, glu_rows),
            (desc + 32, 1),
            (desc + 36, 1),
            (desc + 40, 4),
            (desc + 44, glu_row_bytes),
            (desc + 48, out_bytes),
            (desc + 52, out_bytes),
        ] {
            put32(
                working,
                off,
                u32::try_from(val).map_err(|_| Error::Contract)?,
            );
        }
    }

    let glu_op_desc = new_ops_base + 2 * 232;
    working[glu_op_desc..glu_op_desc + 232].fill(0);
    put32(
        working,
        glu_op_desc,
        crate::htp_ops::HtpOpCode::GluSwiglu.id(),
    );
    put32(working, glu_op_desc + 4, glu_pass.bits());
    working[glu_op_desc + 72..glu_op_desc + 84].copy_from_slice(&glu_kp);
    working[glu_op_desc + 200..glu_op_desc + 228].fill(0xff);
    put16(working, glu_op_desc + 200, 6);
    put16(working, glu_op_desc + 202, 7);
    put16(working, glu_op_desc + 220, 8);

    working[swiglu_out_offset..output_end]
        .as_chunks_mut::<4>()
        .0
        .fill(0x7fc0_ae51_u32.to_le_bytes());

    Ok(FfnSwigluLayout {
        output_offset: swiglu_out_offset,
        output_end,
        n_bufs: batch.n_bufs,
    })
}

#[derive(Clone, Copy)]
pub(crate) struct FfnSwigluMulti2Layout {
    pub(crate) output_offsets: [usize; 2],
    pub(crate) output_end: usize,
    pub(crate) n_bufs: u32,
}

/// Build a fused 6-operation `dspqueue` packet for two tokens:
/// Token 0 (`OP_MUL_MAT(gate)` + `OP_MUL_MAT(up)` + `HTP_OP_GLU_SWIGLU`) followed
/// by Token 1 (`OP_MUL_MAT(gate)` + `OP_MUL_MAT(up)` + `HTP_OP_GLU_SWIGLU`),
/// with per-token VTCM intermediate residency chaining.
#[allow(
    clippy::too_many_lines,
    clippy::too_many_arguments,
    reason = "Keep 2-token 6-op FFN SwiGLU descriptor assembly and VTCM pass flags together"
)]
pub(crate) fn image_ffn_swiglu_multi2_resident(
    bytes: &mut [u8],
    base: u64,
    fd: u32,
    arena_start: usize,
    extra_arenas: &[HtpBufferDesc],
    gate_handle: &NativeQ2WeightHandle,
    up_handle: &NativeQ2WeightHandle,
    activations: [&[f32]; 2],
) -> Result<FfnSwigluMulti2Layout, Error> {
    let columns = gate_handle.columns;
    let rows = gate_handle.rows;
    if up_handle.columns != columns
        || up_handle.rows != rows
        || activations[0].len() != columns
        || activations[1].len() != columns
        || !rows.is_multiple_of(32)
    {
        return Err(Error::Contract);
    }
    let batch = image_multi_resident_with_pad::<4>(
        bytes,
        base,
        fd,
        arena_start,
        extra_arenas,
        [gate_handle, up_handle, gate_handle, up_handle],
        [
            activations[0],
            activations[0],
            activations[1],
            activations[1],
        ],
        4096,
    )?;
    let out_bytes = rows * 4;
    let gate_out_0 = batch.output_offsets[0];
    let up_out_0 = batch.output_offsets[1];
    let gate_out_1 = batch.output_offsets[2];
    let up_out_1 = batch.output_offsets[3];
    let swiglu_out_0 = batch.output_end.next_multiple_of(128);
    let swiglu_out_1 = (swiglu_out_0 + out_bytes).next_multiple_of(128);
    let output_end = swiglu_out_1 + out_bytes;
    let working = bytes.get_mut(..arena_start).ok_or(Error::Contract)?;
    if output_end > working.len() {
        return Err(Error::Contract);
    }
    let n_bufs_usize = usize::try_from(batch.n_bufs).map_err(|_| Error::Contract)?;
    // Re-layout 18 tensor descriptors (1008 bytes) and 6 operation descriptors (1392 bytes)
    // inside cmd_pad (4096 bytes) so n_tensors = 18, n_ops = 6.
    let tensors_base = n_bufs_usize * 24;
    let old_ops_base = tensors_base + 12 * 56;
    let new_ops_base = tensors_base + 18 * 56;
    let mut old_ops = [0_u8; 4 * 232];
    old_ops.copy_from_slice(&working[old_ops_base..old_ops_base + 4 * 232]);
    // Shift Token 1 MulMat tensor descriptors from indices 6..12 to 9..15.
    working.copy_within(
        tensors_base + 6 * 56..tensors_base + 12 * 56,
        tensors_base + 9 * 56,
    );
    for ti in 9_u16..15_u16 {
        let desc = tensors_base + usize::from(ti) * 56;
        put16(working, desc + 22, ti);
    }
    working[tensors_base + 6 * 56..tensors_base + 9 * 56].fill(0);
    working[tensors_base + 15 * 56..new_ops_base + 6 * 232].fill(0);
    working[new_ops_base..new_ops_base + 2 * 232].copy_from_slice(&old_ops[..2 * 232]);
    working[new_ops_base + 3 * 232..new_ops_base + 5 * 232]
        .copy_from_slice(&old_ops[2 * 232..4 * 232]);
    // Update Token 1 MulMat operation tensor indices (Op 3 -> 9, 10, 11; Op 4 -> 12, 13, 14).
    let op3_desc = new_ops_base + 3 * 232;
    put16(working, op3_desc + 200, 9);
    put16(working, op3_desc + 202, 10);
    put16(working, op3_desc + 220, 11);
    let op4_desc = new_ops_base + 4 * 232;
    put16(working, op4_desc + 200, 12);
    put16(working, op4_desc + 202, 13);
    put16(working, op4_desc + 220, 14);

    let vtcm_chain = out_bytes.saturating_mul(3) <= 1_048_576;
    let matmul_pass = if vtcm_chain {
        crate::htp_ops::HtpOpPassFlags::KEEP_DST_IN_VTCM
    } else {
        crate::htp_ops::HtpOpPassFlags::NONE
    };
    let glu_pass = if vtcm_chain {
        crate::htp_ops::HtpOpPassFlags::READ_SRC0_FROM_VTCM
            .union(crate::htp_ops::HtpOpPassFlags::READ_SRC1_FROM_VTCM)
    } else {
        crate::htp_ops::HtpOpPassFlags::NONE
    };
    for op_idx in [0_usize, 1, 3, 4] {
        put32(working, new_ops_base + op_idx * 232 + 4, matmul_pass.bits());
    }

    let (glu_cols, glu_rows) = if rows >= 64 {
        (32_usize, rows / 32)
    } else {
        (rows / 2, 2_usize)
    };
    let glu_cols_u32 = u32::try_from(glu_cols).map_err(|_| Error::Contract)?;
    let glu_rows_u32 = u32::try_from(glu_rows).map_err(|_| Error::Contract)?;
    let glu_kp = crate::htp_ops::solve_glu_kparams(glu_cols_u32, glu_rows_u32)?;
    let glu_row_bytes = glu_cols * 4;

    for (ti_offset, buf_offset) in [
        (6_u16, gate_out_0),
        (7_u16, up_out_0),
        (8_u16, swiglu_out_0),
        (15_u16, gate_out_1),
        (16_u16, up_out_1),
        (17_u16, swiglu_out_1),
    ] {
        let desc = tensors_base + usize::from(ti_offset) * 56;
        put64(
            working,
            desc,
            u64::try_from(buf_offset).map_err(|_| Error::Contract)?,
        );
        put16(working, desc + 20, 0);
        put16(working, desc + 22, ti_offset);
        for (off, val) in [
            (desc + 8, out_bytes),
            (desc + 12, 0),
            (desc + 16, 0), // F32
            (desc + 24, glu_cols),
            (desc + 28, glu_rows),
            (desc + 32, 1),
            (desc + 36, 1),
            (desc + 40, 4),
            (desc + 44, glu_row_bytes),
            (desc + 48, out_bytes),
            (desc + 52, out_bytes),
        ] {
            put32(
                working,
                off,
                u32::try_from(val).map_err(|_| Error::Contract)?,
            );
        }
    }

    for (op_idx, src0_ti, src1_ti, dst_ti) in [(2_usize, 6_u16, 7_u16, 8_u16), (5, 15, 16, 17)] {
        let glu_op_desc = new_ops_base + op_idx * 232;
        working[glu_op_desc..glu_op_desc + 232].fill(0);
        put32(
            working,
            glu_op_desc,
            crate::htp_ops::HtpOpCode::GluSwiglu.id(),
        );
        put32(working, glu_op_desc + 4, glu_pass.bits());
        working[glu_op_desc + 72..glu_op_desc + 84].copy_from_slice(&glu_kp);
        working[glu_op_desc + 200..glu_op_desc + 228].fill(0xff);
        put16(working, glu_op_desc + 200, src0_ti);
        put16(working, glu_op_desc + 202, src1_ti);
        put16(working, glu_op_desc + 220, dst_ti);
    }

    working[swiglu_out_0..output_end]
        .as_chunks_mut::<4>()
        .0
        .fill(0x7fc0_ae51_u32.to_le_bytes());

    Ok(FfnSwigluMulti2Layout {
        output_offsets: [swiglu_out_0, swiglu_out_1],
        output_end,
        n_bufs: batch.n_bufs,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(
    dead_code,
    reason = "Used by full intra-layer FFN VTCM chain packet construction and verification"
)]
pub(crate) struct FfnFullChainLayout {
    pub(crate) output_offset: usize,
    pub(crate) output_end: usize,
    pub(crate) n_bufs: u32,
    pub(crate) n_tensors: u32,
    pub(crate) n_ops: u32,
}

#[allow(
    dead_code,
    clippy::too_many_arguments,
    reason = "Keep F32 2D tensor descriptor fields together for intra-layer VTCM chains"
)]
fn write_f32_2d_tensor_desc(
    working: &mut [u8],
    tensors_base: usize,
    ti: u16,
    buf_offset: usize,
    total_bytes: usize,
    cols: usize,
    rows: usize,
) -> Result<(), Error> {
    let desc = tensors_base + usize::from(ti) * 56;
    let row_bytes = cols.checked_mul(4).ok_or(Error::Contract)?;
    put64(
        working,
        desc,
        u64::try_from(buf_offset).map_err(|_| Error::Contract)?,
    );
    put16(working, desc + 20, 0);
    put16(working, desc + 22, ti);
    for (off, val) in [
        (desc + 8, total_bytes),
        (desc + 12, 0),
        (desc + 16, 0),
        (desc + 24, cols),
        (desc + 28, rows),
        (desc + 32, 1),
        (desc + 36, 1),
        (desc + 40, 4),
        (desc + 44, row_bytes),
        (desc + 48, total_bytes),
        (desc + 52, total_bytes),
    ] {
        put32(
            working,
            off,
            u32::try_from(val).map_err(|_| Error::Contract)?,
        );
    }
    Ok(())
}

/// Build a full intra-layer FFN `dspqueue` packet:
/// `OP_MUL_MAT(gate)` + `OP_MUL_MAT(up)` + `HTP_OP_GLU_SWIGLU` +
/// optional `HTP_OP_HADAMARD` + `OP_MUL_MAT(down)` + optional `HTP_OP_ADD(residual)`
/// with VTCM intermediate residency chaining across all stages.
#[allow(
    dead_code,
    clippy::too_many_lines,
    clippy::too_many_arguments,
    reason = "Keep full intra-layer FFN 4/5/6-op descriptor assembly and VTCM pass flags together"
)]
pub(crate) fn image_ffn_full_chain_resident(
    bytes: &mut [u8],
    base: u64,
    fd: u32,
    arena_start: usize,
    extra_arenas: &[HtpBufferDesc],
    gate_handle: &NativeQ2WeightHandle,
    up_handle: &NativeQ2WeightHandle,
    down_handle: &NativeQ2WeightHandle,
    activation: &[f32],
    residual: Option<&[f32]>,
    hadamard_block_size: Option<u32>,
) -> Result<FfnFullChainLayout, Error> {
    let hidden = gate_handle.columns;
    let ffn_dim = gate_handle.rows;
    if up_handle.columns != hidden
        || up_handle.rows != ffn_dim
        || down_handle.columns != ffn_dim
        || down_handle.rows != hidden
        || activation.len() != hidden
        || !hidden.is_multiple_of(128)
        || !ffn_dim.is_multiple_of(128)
        || residual.is_some_and(|res| res.len() != hidden || !is_finite_f32_slice(res))
        || !is_finite_f32_slice(activation)
    {
        return Err(Error::Contract);
    }
    if let Some(block_size) = hadamard_block_size {
        let block_usize = usize::try_from(block_size).map_err(|_| Error::Contract)?;
        if block_size < 32 || !block_size.is_power_of_two() || !ffn_dim.is_multiple_of(block_usize)
        {
            return Err(Error::Contract);
        }
    }
    let allocation = bytes.len();
    if arena_start > allocation || arena_start < 4096 {
        return Err(Error::Contract);
    }
    let handles = [gate_handle, up_handle, down_handle];
    let mut extra_buf_count = 0_usize;
    let mut batch_extra_arena_indices = [0_u16; 3];
    let mut batch_extra_descs = [HtpBufferDesc {
        base: 0,
        size: 0,
        fd: 0,
    }; 3];
    let mut weight_offsets = [0_usize; 3];
    let mut weight_bis = [0_u16; 3];
    for (member, handle) in handles.into_iter().enumerate() {
        let expected_weights = layout(handle.columns, handle.rows).weights;
        if handle.packed_bytes() != expected_weights {
            return Err(Error::Contract);
        }
        if handle.arena_index == 0 {
            let start = arena_start
                .checked_add(handle.range.start)
                .ok_or(Error::Contract)?;
            let end = arena_start
                .checked_add(handle.range.end)
                .ok_or(Error::Contract)?;
            if end > allocation {
                return Err(Error::Contract);
            }
            weight_offsets[member] = start;
            weight_bis[member] = 0;
        } else {
            let extra_idx = usize::from(handle.arena_index - 1);
            let extra = extra_arenas
                .get(extra_idx)
                .copied()
                .ok_or(Error::Contract)?;
            if handle.range.end > extra.size {
                return Err(Error::Contract);
            }
            let bi = if let Some(pos) = batch_extra_arena_indices[..extra_buf_count]
                .iter()
                .position(|&idx| idx == handle.arena_index)
            {
                u16::try_from(pos + 1).map_err(|_| Error::Contract)?
            } else {
                batch_extra_arena_indices[extra_buf_count] = handle.arena_index;
                batch_extra_descs[extra_buf_count] = extra;
                extra_buf_count += 1;
                u16::try_from(extra_buf_count).map_err(|_| Error::Contract)?
            };
            weight_offsets[member] = handle.range.start;
            weight_bis[member] = bi;
        }
    }

    let cmd_pad = 4096_usize;
    let hidden_bytes = hidden * 4;
    let ffn_bytes = ffn_dim * 4;
    let act_offset = cmd_pad;
    let mut next_offset = (act_offset + hidden_bytes).next_multiple_of(128);
    let res_offset = if residual.is_some() {
        let off = next_offset;
        next_offset = (off + hidden_bytes).next_multiple_of(128);
        off
    } else {
        0
    };
    let gate_out_offset = next_offset;
    let up_out_offset = (gate_out_offset + ffn_bytes).next_multiple_of(128);
    let swiglu_out_offset = (up_out_offset + ffn_bytes).next_multiple_of(128);
    let hadamard_out_offset = if hadamard_block_size.is_some() {
        (swiglu_out_offset + ffn_bytes).next_multiple_of(128)
    } else {
        swiglu_out_offset
    };
    let down_out_offset = (hadamard_out_offset + ffn_bytes).next_multiple_of(128);
    let final_out_offset = if residual.is_some() {
        (down_out_offset + hidden_bytes).next_multiple_of(128)
    } else {
        down_out_offset
    };
    let output_end = final_out_offset + hidden_bytes;
    let working = bytes.get_mut(..arena_start).ok_or(Error::Contract)?;
    if output_end > working.len() {
        return Err(Error::Contract);
    }

    let n_bufs_usize = 1 + extra_buf_count;
    let n_bufs = u32::try_from(n_bufs_usize).map_err(|_| Error::Contract)?;
    let n_tensors_usize =
        12 + usize::from(hadamard_block_size.is_some()) * 2 + usize::from(residual.is_some()) * 3;
    let n_ops_usize =
        4 + usize::from(hadamard_block_size.is_some()) + usize::from(residual.is_some());
    let tensors_base = n_bufs_usize * 24;
    let ops_base = tensors_base + n_tensors_usize * 56;
    if ops_base + n_ops_usize * 232 > cmd_pad {
        return Err(Error::Contract);
    }
    working[..cmd_pad].fill(0);
    put64(working, 0, base);
    put64(
        working,
        8,
        u64::try_from(allocation).map_err(|_| Error::Contract)?,
    );
    put32(working, 20, fd);
    for (idx, extra) in batch_extra_descs[..extra_buf_count].iter().enumerate() {
        let buf_offset = (idx + 1) * 24;
        put64(working, buf_offset, extra.base);
        put64(
            working,
            buf_offset + 8,
            u64::try_from(extra.size).map_err(|_| Error::Contract)?,
        );
        put32(working, buf_offset + 20, extra.fd);
    }

    let vtcm_chain = ffn_bytes.saturating_mul(3) <= 1_048_576;
    let keep_vtcm = if vtcm_chain {
        crate::htp_ops::HtpOpPassFlags::KEEP_DST_IN_VTCM
    } else {
        crate::htp_ops::HtpOpPassFlags::NONE
    };

    // Op 0 (gate: ti 0, 1 -> 2) and Op 1 (up: ti 3, 4 -> 5).
    write_member_descriptors(
        working,
        0,
        weight_bis[0],
        tensors_base,
        ops_base,
        hidden,
        ffn_dim,
        weight_offsets[0],
        layout(hidden, ffn_dim).weights,
        act_offset,
        gate_out_offset,
    )?;
    put32(working, ops_base + 4, keep_vtcm.bits());

    write_member_descriptors(
        working,
        3,
        weight_bis[1],
        tensors_base + 3 * 56,
        ops_base + 232,
        hidden,
        ffn_dim,
        weight_offsets[1],
        layout(hidden, ffn_dim).weights,
        act_offset,
        up_out_offset,
    )?;
    put32(working, ops_base + 232 + 4, keep_vtcm.bits());

    // Op 2: GluSwiglu (ti 6, 7 -> 8).
    let (glu_cols, glu_rows) = if ffn_dim >= 64 {
        (32_usize, ffn_dim / 32)
    } else {
        (ffn_dim / 2, 2_usize)
    };
    let glu_kp = crate::htp_ops::solve_glu_kparams(
        u32::try_from(glu_cols).map_err(|_| Error::Contract)?,
        u32::try_from(glu_rows).map_err(|_| Error::Contract)?,
    )?;
    for (ti, buf_off) in [
        (6_u16, gate_out_offset),
        (7_u16, up_out_offset),
        (8_u16, swiglu_out_offset),
    ] {
        write_f32_2d_tensor_desc(
            working,
            tensors_base,
            ti,
            buf_off,
            ffn_bytes,
            glu_cols,
            glu_rows,
        )?;
    }
    let glu_op_desc = ops_base + 2 * 232;
    let glu_flags = if vtcm_chain {
        crate::htp_ops::HtpOpPassFlags::READ_SRC0_FROM_VTCM
            .union(crate::htp_ops::HtpOpPassFlags::READ_SRC1_FROM_VTCM)
            .union(crate::htp_ops::HtpOpPassFlags::KEEP_DST_IN_VTCM)
    } else {
        crate::htp_ops::HtpOpPassFlags::NONE
    };
    put32(
        working,
        glu_op_desc,
        crate::htp_ops::HtpOpCode::GluSwiglu.id(),
    );
    put32(working, glu_op_desc + 4, glu_flags.bits());
    working[glu_op_desc + 72..glu_op_desc + 84].copy_from_slice(&glu_kp);
    working[glu_op_desc + 200..glu_op_desc + 228].fill(0xff);
    put16(working, glu_op_desc + 200, 6);
    put16(working, glu_op_desc + 202, 7);
    put16(working, glu_op_desc + 220, 8);

    let mut next_ti = 9_u16;
    let mut next_op = 3_usize;

    // Optional Op 3: HTP_OP_HADAMARD (69) over [block_size, ffn_dim / block_size].
    if let Some(block_size) = hadamard_block_size {
        let block_cols = usize::try_from(block_size).map_err(|_| Error::Contract)?;
        let block_rows = ffn_dim / block_cols;
        let had_kp = crate::htp_ops::solve_unary_kparams(
            crate::htp_ops::HtpOpCode::Hadamard,
            block_size,
            u32::try_from(block_rows).map_err(|_| Error::Contract)?,
            1,
        )?;
        write_f32_2d_tensor_desc(
            working,
            tensors_base,
            next_ti,
            swiglu_out_offset,
            ffn_bytes,
            block_cols,
            block_rows,
        )?;
        write_f32_2d_tensor_desc(
            working,
            tensors_base,
            next_ti + 1,
            hadamard_out_offset,
            ffn_bytes,
            block_cols,
            block_rows,
        )?;
        let had_op_desc = ops_base + next_op * 232;
        let had_flags = if vtcm_chain {
            crate::htp_ops::HtpOpPassFlags::READ_SRC0_FROM_VTCM
                .union(crate::htp_ops::HtpOpPassFlags::KEEP_DST_IN_VTCM)
        } else {
            crate::htp_ops::HtpOpPassFlags::NONE
        };
        put32(
            working,
            had_op_desc,
            crate::htp_ops::HtpOpCode::Hadamard.id(),
        );
        put32(working, had_op_desc + 4, had_flags.bits());
        working[had_op_desc + 72..had_op_desc + 136].copy_from_slice(&had_kp);
        working[had_op_desc + 200..had_op_desc + 228].fill(0xff);
        put16(working, had_op_desc + 200, next_ti);
        put16(working, had_op_desc + 220, next_ti + 1);
        next_ti += 2;
        next_op += 1;
    }

    // Next Op: MUL_MAT(down) reading activation from VTCM.
    let down_op_desc = ops_base + next_op * 232;
    write_member_descriptors(
        working,
        next_ti,
        weight_bis[2],
        tensors_base + usize::from(next_ti) * 56,
        down_op_desc,
        ffn_dim,
        hidden,
        weight_offsets[2],
        layout(ffn_dim, hidden).weights,
        hadamard_out_offset,
        down_out_offset,
    )?;
    let down_flags = if vtcm_chain {
        let base_flags = crate::htp_ops::HtpOpPassFlags::READ_SRC1_FROM_VTCM;
        if residual.is_some() {
            base_flags.union(crate::htp_ops::HtpOpPassFlags::KEEP_DST_IN_VTCM)
        } else {
            base_flags
        }
    } else {
        crate::htp_ops::HtpOpPassFlags::NONE
    };
    put32(working, down_op_desc + 4, down_flags.bits());
    next_ti += 3;
    next_op += 1;

    // Optional Final Op: HTP_OP_ADD(down_out, residual) -> final_out.
    if residual.is_some() {
        let add_kp = crate::htp_ops::solve_binary_kparams(
            crate::htp_ops::HtpOpCode::Add,
            u32::try_from(hidden).map_err(|_| Error::Contract)?,
            1,
        )?;
        write_f32_2d_tensor_desc(
            working,
            tensors_base,
            next_ti,
            down_out_offset,
            hidden_bytes,
            hidden,
            1,
        )?;
        write_f32_2d_tensor_desc(
            working,
            tensors_base,
            next_ti + 1,
            res_offset,
            hidden_bytes,
            hidden,
            1,
        )?;
        write_f32_2d_tensor_desc(
            working,
            tensors_base,
            next_ti + 2,
            final_out_offset,
            hidden_bytes,
            hidden,
            1,
        )?;
        let add_op_desc = ops_base + next_op * 232;
        let add_flags = if vtcm_chain {
            crate::htp_ops::HtpOpPassFlags::READ_SRC0_FROM_VTCM
        } else {
            crate::htp_ops::HtpOpPassFlags::NONE
        };
        put32(working, add_op_desc, crate::htp_ops::HtpOpCode::Add.id());
        put32(working, add_op_desc + 4, add_flags.bits());
        working[add_op_desc + 72..add_op_desc + 168].copy_from_slice(&add_kp);
        working[add_op_desc + 200..add_op_desc + 228].fill(0xff);
        put16(working, add_op_desc + 200, next_ti);
        put16(working, add_op_desc + 202, next_ti + 1);
        put16(working, add_op_desc + 220, next_ti + 2);
    }

    for (dst, &val) in working[act_offset..act_offset + hidden_bytes]
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(activation)
    {
        *dst = val.to_bits().to_le_bytes();
    }
    if let Some(res_slice) = residual {
        for (dst, &val) in working[res_offset..res_offset + hidden_bytes]
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(res_slice)
        {
            *dst = val.to_bits().to_le_bytes();
        }
    }
    working[gate_out_offset..output_end]
        .as_chunks_mut::<4>()
        .0
        .fill(0x7fc0_ae51_u32.to_le_bytes());

    Ok(FfnFullChainLayout {
        output_offset: final_out_offset,
        output_end,
        n_bufs,
        n_tensors: u32::try_from(n_tensors_usize).map_err(|_| Error::Contract)?,
        n_ops: u32::try_from(n_ops_usize).map_err(|_| Error::Contract)?,
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
    weight_bi: u16,
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
    put16(bytes, weight_desc + 20, weight_bi);
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
    let mut m0 = 0_u32;
    let mut m1 = 0_u32;
    let mut m2 = 0_u32;
    let mut m3 = 0_u32;
    for (in_chunk, out_chunk) in in_chunks.iter().zip(out_chunks) {
        let mut local0 = 0_u32;
        let mut local1 = 0_u32;
        let mut local2 = 0_u32;
        let mut local3 = 0_u32;
        for i in 0..4 {
            let b0 = u32::from_le_bytes(in_chunk[i]);
            let b1 = u32::from_le_bytes(in_chunk[i + 4]);
            let b2 = u32::from_le_bytes(in_chunk[i + 8]);
            let b3 = u32::from_le_bytes(in_chunk[i + 12]);
            local0 |= (b0 & EXP_MASK).wrapping_add(EXP_STEP);
            local1 |= (b1 & EXP_MASK).wrapping_add(EXP_STEP);
            local2 |= (b2 & EXP_MASK).wrapping_add(EXP_STEP);
            local3 |= (b3 & EXP_MASK).wrapping_add(EXP_STEP);
            out_chunk[i] = f32::from_bits(b0);
            out_chunk[i + 4] = f32::from_bits(b1);
            out_chunk[i + 8] = f32::from_bits(b2);
            out_chunk[i + 12] = f32::from_bits(b3);
        }
        m0 |= local0;
        m1 |= local1;
        m2 |= local2;
        m3 |= local3;
    }
    let mut tail_mask = (m0 | m1) | (m2 | m3);
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

        // Verify 56-projection multi-arena split (962,887,680 bytes -> [502,824,960, 460,062,720]).
        let multi_limits =
            NativeQ2ProjectionLimits::new(17_408, 17_408, 24_000, Duration::from_secs(60))?
                .with_resident_weights(962_887_680)?;
        let (caps, count) = arena_capacities(962_887_680).ok_or(Error::Contract)?;
        assert_eq!(count, 2);
        assert_eq!(caps, [502_824_960, 460_062_720, 0, 0, 0, 0]);
        assert_eq!(
            multi_limits.allocation_bytes(),
            NATIVE_Q2_MAX_WORKING_BYTES + 962_887_680
        );

        // Verify 112-projection 4-arena split (1,925,775,360 bytes -> alternating [502,824,960, 460,062,720, 502,824,960, 460,062,720]).
        let quad_limits =
            NativeQ2ProjectionLimits::new(17_408, 17_408, 24_000, Duration::from_secs(60))?
                .with_resident_weights(1_925_775_360)?;
        let (quad_caps, quad_count) = arena_capacities(1_925_775_360).ok_or(Error::Contract)?;
        assert_eq!(quad_count, 4);
        assert_eq!(
            quad_caps,
            [502_824_960, 460_062_720, 502_824_960, 460_062_720, 0, 0]
        );
        assert_eq!(
            quad_limits.allocation_bytes(),
            NATIVE_Q2_MAX_WORKING_BYTES + 1_925_775_360
        );

        // Verify 168-projection 6-arena split (2,888,663,040 bytes -> alternating 6-arena steps).
        let six_limits =
            NativeQ2ProjectionLimits::new(17_408, 17_408, 24_000, Duration::from_secs(60))?
                .with_resident_weights(2_888_663_040)?;
        let (six_caps, six_count) = arena_capacities(2_888_663_040).ok_or(Error::Contract)?;
        assert_eq!(six_count, 6);
        assert_eq!(
            six_caps,
            [
                502_824_960,
                460_062_720,
                502_824_960,
                460_062_720,
                502_824_960,
                460_062_720
            ]
        );
        assert_eq!(
            six_limits.allocation_bytes(),
            NATIVE_Q2_MAX_WORKING_BYTES + 2_888_663_040
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
        assert_eq!(handle.arena_index(), 0);
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
            &[],
            [&h0, &h1],
            &activation,
        )?;
        assert_eq!(batch.n_bufs, 1);
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

    #[test]
    #[allow(
        clippy::too_many_lines,
        clippy::similar_names,
        reason = "Keep multi-arena deduplication, 4/8-op batching and 3/6-op FFN SwiGLU checks together"
    )]
    fn multi_arena_resident_descriptors_deduplicate_buffers_and_route_bi() -> Result<(), Error> {
        let mut multi_reg = NativeQ2WeightRegistry::new(962_887_680)?;
        let h_arena0 = multi_reg
            .prepare_validated(5120, 17_408, 502_824_960)?
            .ok_or(Error::Contract)?;
        assert_eq!(h_arena0.arena_index(), 0);
        let h_arena1 = multi_reg
            .prepare_validated(128, 32, 1152)?
            .ok_or(Error::Contract)?;
        assert_eq!(h_arena1.arena_index(), 1);
        assert_eq!(multi_reg.arena_used_bytes(0), 502_824_960);
        assert_eq!(multi_reg.arena_used_bytes(1), 1152);
        let arena_start = 4096_usize;
        let mut arena = vec![0_u8; arena_start + 4096];
        let activation = vec![1.0_f32; 128];
        let extra = [HtpBufferDesc {
            base: 0x2000_0000,
            size: 4096,
            fd: 9,
        }];
        let single_multi = image_resident(
            &mut arena,
            0x1000_0000,
            7,
            arena_start,
            &extra,
            &h_arena1,
            &activation,
        )?;
        assert_eq!(single_multi.n_bufs, 2);
        // bufs[1] is at 24..48, weight_desc is at 48..104 with bi = 1 at 48 + 20 = 68.
        assert_eq!(
            u64::from_le_bytes(arena[24..32].try_into().map_err(|_| Error::Contract)?),
            0x2000_0000
        );
        assert_eq!(
            u32::from_le_bytes(arena[44..48].try_into().map_err(|_| Error::Contract)?),
            9
        );
        assert_eq!(
            u16::from_le_bytes(arena[68..70].try_into().map_err(|_| Error::Contract)?),
            1
        );

        let batch_multi = image_batch_resident(
            &mut arena,
            0x1000_0000,
            7,
            arena_start,
            &extra,
            [&h_arena1, &h_arena1],
            &activation,
        )?;
        assert_eq!(batch_multi.n_bufs, 2);
        // Both members share Arena 1 -> deduplicated into bufs[1] (bi = 1).
        assert_eq!(
            u16::from_le_bytes(
                arena[48 + 20..48 + 22]
                    .try_into()
                    .map_err(|_| Error::Contract)?
            ),
            1
        );
        assert_eq!(
            u16::from_le_bytes(
                arena[48 + 3 * 56 + 20..48 + 3 * 56 + 22]
                    .try_into()
                    .map_err(|_| Error::Contract)?
            ),
            1
        );

        let act1 = vec![2.5_f32; 128];
        let multi4 = image_multi_resident(
            &mut arena,
            0x1000_0000,
            7,
            arena_start,
            &extra,
            [&h_arena1, &h_arena1, &h_arena1, &h_arena1],
            [&activation, &activation, &act1, &act1],
        )?;
        assert_eq!(multi4.n_bufs, 2);
        // Two unique 512-byte activations at 2048 and 2560; 4 outputs at 3072, 3200, 3328, 3456.
        assert_eq!(multi4.output_offsets, [3072, 3200, 3328, 3456]);
        assert_eq!(multi4.output_end, 3584);

        // Verify 8-op multi-resident batch and fused FFN SwiGLU 3-op packet.
        let arena8_start = 16_384_usize;
        let mut arena8 = vec![0_u8; arena8_start + 4096];
        let multi8 = image_multi_resident(
            &mut arena8,
            0x1000_0000,
            7,
            arena8_start,
            &extra,
            [
                &h_arena1, &h_arena1, &h_arena1, &h_arena1, &h_arena1, &h_arena1, &h_arena1,
                &h_arena1,
            ],
            [
                &activation,
                &activation,
                &act1,
                &act1,
                &activation,
                &activation,
                &act1,
                &act1,
            ],
        )?;
        assert_eq!(multi8.n_bufs, 2);
        assert_eq!(multi8.output_offsets[0], 5120);
        assert_eq!(multi8.output_end, 5120 + 8 * 128);

        let swiglu = image_ffn_swiglu_resident(
            &mut arena,
            0x1000_0000,
            7,
            arena_start,
            &extra,
            &h_arena1,
            &h_arena1,
            &activation,
        )?;
        assert_eq!(swiglu.n_bufs, 2);
        assert_eq!(swiglu.output_offset, 2816);
        assert_eq!(swiglu.output_end, 2944);
        // Verify GluSwiglu opcode (22) at Op 2 and KEEP_DST_IN_VTCM (1) on Op 0 & Op 1.
        let ops_base = 2 * 24 + 9 * 56;
        assert_eq!(
            u32::from_le_bytes(
                arena[ops_base + 4..ops_base + 8]
                    .try_into()
                    .map_err(|_| Error::Contract)?
            ),
            crate::htp_ops::HtpOpPassFlags::KEEP_DST_IN_VTCM.bits()
        );
        assert_eq!(
            u32::from_le_bytes(
                arena[ops_base + 232 + 4..ops_base + 232 + 8]
                    .try_into()
                    .map_err(|_| Error::Contract)?
            ),
            crate::htp_ops::HtpOpPassFlags::KEEP_DST_IN_VTCM.bits()
        );
        assert_eq!(
            u32::from_le_bytes(
                arena[ops_base + 2 * 232..ops_base + 2 * 232 + 4]
                    .try_into()
                    .map_err(|_| Error::Contract)?
            ),
            crate::htp_ops::HtpOpCode::GluSwiglu.id()
        );
        assert_eq!(
            u32::from_le_bytes(
                arena[ops_base + 2 * 232 + 4..ops_base + 2 * 232 + 8]
                    .try_into()
                    .map_err(|_| Error::Contract)?
            ),
            crate::htp_ops::HtpOpPassFlags::READ_SRC0_FROM_VTCM
                .union(crate::htp_ops::HtpOpPassFlags::READ_SRC1_FROM_VTCM)
                .bits()
        );

        let swiglu_multi2 = image_ffn_swiglu_multi2_resident(
            &mut arena8,
            0x1000_0000,
            7,
            arena8_start,
            &extra,
            &h_arena1,
            &h_arena1,
            [&activation, &act1],
        )?;
        assert_eq!(swiglu_multi2.n_bufs, 2);
        // cmd_pad = 4096; two 512B activations at 4096 and 4608; four intermediate
        // 128B outputs at 5120, 5248, 5376, 5504; two SwiGLU outputs at 5632 and 5760.
        assert_eq!(swiglu_multi2.output_offsets, [5632, 5760]);
        assert_eq!(swiglu_multi2.output_end, 5888);
        let ops6_base = 2 * 24 + 18 * 56;
        for glu_idx in [2_usize, 5] {
            assert_eq!(
                u32::from_le_bytes(
                    arena8[ops6_base + glu_idx * 232..ops6_base + glu_idx * 232 + 4]
                        .try_into()
                        .map_err(|_| Error::Contract)?
                ),
                crate::htp_ops::HtpOpCode::GluSwiglu.id()
            );
        }

        // Verify 6-op full intra-layer FFN VTCM chain (gate + up + GluSwiglu + Hadamard + down + Add).
        let h_down = multi_reg
            .prepare_validated(128, 128, 4608)?
            .ok_or(Error::Contract)?;
        let h_gate128 = multi_reg
            .prepare_validated(128, 128, 4608)?
            .ok_or(Error::Contract)?;
        let extra_large = [HtpBufferDesc {
            base: 0x2000_0000,
            size: 32_768,
            fd: 9,
        }];
        let chain = image_ffn_full_chain_resident(
            &mut arena8,
            0x1000_0000,
            7,
            arena8_start,
            &extra_large,
            &h_gate128,
            &h_gate128,
            &h_down,
            &activation,
            Some(&act1),
            Some(64),
        )?;
        assert_eq!(chain.n_bufs, 2);
        assert_eq!(chain.n_tensors, 17);
        assert_eq!(chain.n_ops, 6);
        let chain_ops_base = 2 * 24 + 17 * 56;
        let expected_opcodes = [
            crate::htp_ops::HtpOpCode::MulMat.id(),
            crate::htp_ops::HtpOpCode::MulMat.id(),
            crate::htp_ops::HtpOpCode::GluSwiglu.id(),
            crate::htp_ops::HtpOpCode::Hadamard.id(),
            crate::htp_ops::HtpOpCode::MulMat.id(),
            crate::htp_ops::HtpOpCode::Add.id(),
        ];
        for (op_idx, expected_op) in expected_opcodes.into_iter().enumerate() {
            let op_off = chain_ops_base + op_idx * 232;
            assert_eq!(
                u32::from_le_bytes(
                    arena8[op_off..op_off + 4]
                        .try_into()
                        .map_err(|_| Error::Contract)?
                ),
                expected_op
            );
        }

        // Verify 6-arena total_used accounting across arenas 4 and 5.
        let mut six_reg = NativeQ2WeightRegistry::new(2_888_663_040)?;
        for expected_arena in 0..6_u16 {
            let step = if expected_arena % 2 == 0 {
                NATIVE_Q2_ARENA_WEIGHT_STEP_BYTES
            } else {
                NATIVE_Q2_ARENA_ODD_WEIGHT_STEP_BYTES
            };
            let h = six_reg
                .prepare_validated(5120, 17_408, step)?
                .ok_or(Error::Contract)?;
            assert_eq!(h.arena_index(), expected_arena);
        }
        assert_eq!(six_reg.used_bytes(), 2_888_663_040);
        assert_eq!(six_reg.remaining_bytes(), 0);
        Ok(())
    }
}
