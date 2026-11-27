//! Complete-row arithmetic shared by the safe detected boundary and controls.

#![expect(
    clippy::redundant_pub_crate,
    reason = "Parent-only checks and portable controls remain inside the safe scaled-row boundary."
)]

/// Exact prepared activation for one original 128-column PQ2 block.
///
/// Byte planes follow two 64-column halves, each with four 16-byte digit
/// planes. `sum` is the sum of all signed bytes used for the unsigned-code
/// correction. The runtime retains its existing 136-byte block layout.
#[repr(C)]
pub struct Pq2Activation {
    /// Original `Q8_K` scale, shared by both halves of its quantizer block.
    pub scale: f32,
    /// Original signed codes rearranged into consumption order.
    pub values: [i8; 128],
    /// Exact signed-byte sum subtracted from each unsigned-code dot.
    pub sum: i32,
}

const _: () = assert!(size_of::<Pq2Activation>() <= 136);
const _: () = assert!(core::mem::offset_of!(Pq2Activation, scale) == 0);
const _: () = assert!(core::mem::offset_of!(Pq2Activation, values) == 4);
const _: () = assert!(core::mem::offset_of!(Pq2Activation, sum) == 132);

/// Pre-packed two-token `I8MM` (`SMMLA`) activation panel for one 128-column PQ2 block.
///
/// Interleaves token pairs `(2p, 2p+1)` once per projection tile via `pair_s8_lo_hi`
/// so inner 4x8 `SMMLA` loops consume contiguous 16-byte halves with zero permutes.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct Pq2I8mmPairBlock {
    /// Pre-interleaved `(lo, hi)` `i8x16` panels for each of the 8 PQ2 consumption planes.
    pub planes: [([i8; 16], [i8; 16]); 8],
    /// Signed sums `[t0.sum, t1.sum]` for the token pair.
    pub sums: [i32; 2],
    /// Activation scales `[t0.scale, t1.scale]` for the token pair.
    pub scales: [f32; 2],
}

const _: () = assert!(size_of::<Pq2I8mmPairBlock>() == 272);
const _: () = assert!(core::mem::offset_of!(Pq2I8mmPairBlock, planes) == 0);
const _: () = assert!(core::mem::offset_of!(Pq2I8mmPairBlock, sums) == 256);
const _: () = assert!(core::mem::offset_of!(Pq2I8mmPairBlock, scales) == 264);

impl Pq2I8mmPairBlock {
    /// Returns a zeroed [`Pq2I8mmPairBlock`].
    #[inline]
    #[must_use]
    pub const fn zeroed() -> Self {
        Self {
            planes: [([0_i8; 16], [0_i8; 16]); 8],
            sums: [0; 2],
            scales: [0.0; 2],
        }
    }
}

/// One 32-element quantized `Q8_0` activation group with its original finite scale.
#[derive(Clone, Copy, Debug)]
pub struct Q8ActivationBlock {
    /// Original `f32` activation scale, preserving zero sign.
    pub scale: f32,
    /// Signed 8-bit activation codes in source lane order.
    pub values: [i8; 32],
}

/// Packed 128-element `Q8_0` activation block for four consecutive `PTQ1_0` groups.
///
/// Layout matches one 28-byte `PTQ1_0` block (4 groups of 32 activations):
/// `scales` holds the four original `f32` activation scales, `values` holds the
/// 128 signed quantization codes contiguously, and `neg_lane_sums` holds the
/// precomputed negated 8-element lane sums `[-(sum_k0..3 + sum_k16..19); 4]`
/// for each of the four groups.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct Ptq1Activation {
    /// Original `f32` scales for groups `0..4`.
    pub(crate) scales: [f32; 4],
    /// Contiguous signed 8-bit activation codes for groups `0..4`.
    pub(crate) values: [i8; 128],
    /// Precomputed negated 8-lane sums per group for unsigned ternary SVE decoding.
    pub(crate) neg_lane_sums: [[i32; 4]; 4],
}

const _: () = assert!(size_of::<Ptq1Activation>() == 208);
const _: () = assert!(core::mem::offset_of!(Ptq1Activation, values) == 16);
const _: () = assert!(core::mem::offset_of!(Ptq1Activation, neg_lane_sums) == 144);

/// Pre-packed two-token `I8MM` (`SMMLA`) activation panel for one 32-element `Q8_0` group
/// under `PTQ1_0` pinned four-lane accumulation.
///
/// Each of the four 16-byte `planes[l]` holds the 8-element lane `l` from token `t0`
/// (`values[4*l..4*l+4]`, `values[16+4*l..16+4*l+4]`) followed by the 8-element lane `l`
/// from token `t1`.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct Ptq1PinnedI8mmPairBlock {
    /// Four lane-deinterleaved `(t0, t1)` 16-byte `SMMLA` panels for lanes `0..4`.
    pub planes: [[i8; 16]; 4],
    /// Original `f32` activation scales `[t0.scale(), t1.scale()]` for the token pair.
    pub scales: [f32; 2],
}

const _: () = assert!(size_of::<Ptq1PinnedI8mmPairBlock>() == 80);
const _: () = assert!(core::mem::offset_of!(Ptq1PinnedI8mmPairBlock, planes) == 0);
const _: () = assert!(core::mem::offset_of!(Ptq1PinnedI8mmPairBlock, scales) == 64);

impl Ptq1PinnedI8mmPairBlock {
    /// Returns a zeroed [`Ptq1PinnedI8mmPairBlock`].
    #[inline]
    #[must_use]
    pub const fn zeroed() -> Self {
        Self {
            planes: [[0_i8; 16]; 4],
            scales: [0.0; 2],
        }
    }
}

impl Ptq1Activation {
    /// Returns a zeroed [`Ptq1Activation`] block.
    #[inline]
    #[must_use]
    pub const fn zeroed() -> Self {
        Self {
            scales: [0.0; 4],
            values: [0; 128],
            neg_lane_sums: [[0; 4]; 4],
        }
    }

    /// Constructs a [`Ptq1Activation`] from contiguous group scales and codes,
    /// precomputing the negated lane sums once for all row tiles.
    #[inline]
    #[must_use]
    pub fn new(scales: [f32; 4], values: [i8; 128]) -> Self {
        #[cfg(target_arch = "aarch64")]
        let neg_lane_sums = crate::aarch64::ptq1_neg_lane_sums(&values);
        #[cfg(not(target_arch = "aarch64"))]
        let neg_lane_sums = core::array::from_fn(|group| {
            let base = group * 32;
            core::array::from_fn(|lane| {
                let off = lane * 4;
                let mut sum = 0_i32;
                for k in 0..4 {
                    sum +=
                        i32::from(values[base + off + k]) + i32::from(values[base + 16 + off + k]);
                }
                -sum
            })
        });
        Self {
            scales,
            values,
            neg_lane_sums,
        }
    }

    /// Constructs a [`Ptq1Activation`] without computing `neg_lane_sums`.
    ///
    /// Intended for non-SVE paths (`!Ptq1Dotprod::has_sve()`) where the 64-byte
    /// SVE correction table is unused, preserving the exact 208-byte layout.
    #[inline]
    #[must_use]
    pub const fn new_without_lane_sums(scales: [f32; 4], values: [i8; 128]) -> Self {
        Self {
            scales,
            values,
            neg_lane_sums: [[0; 4]; 4],
        }
    }

    /// Constructs a [`Ptq1Activation`] from four 32-element group slices,
    /// precomputing the negated lane sums once for SVE consumption.
    #[inline]
    #[must_use]
    pub fn from_groups(scales: [f32; 4], groups: [&[i8; 32]; 4]) -> Self {
        Self::new(scales, Self::copy_groups(groups))
    }

    /// Constructs a [`Ptq1Activation`] from four 32-element group slices without
    /// computing SVE `neg_lane_sums`, for non-SVE targets.
    #[inline]
    #[must_use]
    pub const fn from_groups_without_lane_sums(scales: [f32; 4], groups: [&[i8; 32]; 4]) -> Self {
        Self::new_without_lane_sums(scales, Self::copy_groups(groups))
    }

    #[inline]
    const fn copy_groups(groups: [&[i8; 32]; 4]) -> [i8; 128] {
        let mut values = [0_i8; 128];
        let chunks = values.as_chunks_mut::<32>().0;
        chunks[0].copy_from_slice(groups[0]);
        chunks[1].copy_from_slice(groups[1]);
        chunks[2].copy_from_slice(groups[2]);
        chunks[3].copy_from_slice(groups[3]);
        values
    }

    /// Returns the original `f32` activation scales for groups `0..4`.
    #[inline]
    #[must_use]
    pub const fn scales(&self) -> &[f32; 4] {
        &self.scales
    }

    /// Returns the contiguous 128 signed 8-bit activation codes.
    #[inline]
    #[must_use]
    pub const fn values(&self) -> &[i8; 128] {
        &self.values
    }
}

/// Safe read-only view of one 32-element `Q8_0` activation group (`scale` and `32` `i8` codes).
pub trait Q8ActivationView {
    /// Returns the group's original `f32` scale.
    fn scale(&self) -> f32;
    /// Returns the group's 32 signed quantization codes in source order.
    fn values(&self) -> &[i8; 32];
    /// Returns a direct reference to the group's `f32` scale when stored in-struct.
    #[inline]
    fn scale_ref(&self) -> Option<&f32> {
        None
    }
}

impl Q8ActivationView for Q8ActivationBlock {
    #[inline]
    fn scale(&self) -> f32 {
        self.scale
    }

    #[inline]
    fn values(&self) -> &[i8; 32] {
        &self.values
    }

    #[inline]
    fn scale_ref(&self) -> Option<&f32> {
        Some(&self.scale)
    }
}

impl Q8ActivationView for (&[i8; 32], f32) {
    #[inline]
    fn scale(&self) -> f32 {
        self.1
    }

    #[inline]
    fn values(&self) -> &[i8; 32] {
        self.0
    }

    #[inline]
    fn scale_ref(&self) -> Option<&f32> {
        Some(&self.1)
    }
}

impl<T: Q8ActivationView + ?Sized> Q8ActivationView for &T {
    #[inline]
    fn scale(&self) -> f32 {
        (*self).scale()
    }

    #[inline]
    fn values(&self) -> &[i8; 32] {
        (*self).values()
    }

    #[inline]
    fn scale_ref(&self) -> Option<&f32> {
        (*self).scale_ref()
    }
}

pub(super) fn valid<const ROWS: usize>(rows: [&[u8]; ROWS], activation: &[Pq2Activation]) -> bool {
    valid_rows(rows, activation.len()) && activation.iter().all(|block| block.scale.is_finite())
}

pub(super) fn valid_rows<const ROWS: usize>(rows: [&[u8]; ROWS], count: usize) -> bool {
    matches!(ROWS, 1 | 4 | 8 | 16)
        && count != 0
        && count
            .checked_mul(34)
            .is_some_and(|bytes| rows.iter().all(|row| row.len() == bytes))
}

pub(super) fn valid_scaled_8x8_chunks(
    rows: [&[u8]; 8],
    chunks: &[[&[Pq2Activation]; 8]],
    outputs_len: usize,
) -> bool {
    if chunks.is_empty() || chunks.len() > 4 || outputs_len != chunks.len() {
        return false;
    }
    let count = chunks[0][0].len();
    valid_rows(rows, count)
        && chunks.iter().flatten().all(|activation| {
            activation.len() == count && activation.iter().all(|block| block.scale.is_finite())
        })
}

pub(super) fn valid_scaled_8x_tokens(rows: [&[u8]; 8], tokens: &[&[Pq2Activation]]) -> bool {
    if tokens.is_empty() || tokens.len() > 8 {
        return false;
    }
    let count = tokens[0].len();
    valid_rows(rows, count)
        && tokens.iter().all(|activation| {
            activation.len() == count && activation.iter().all(|block| block.scale.is_finite())
        })
}

pub(super) fn valid_pq2_chunks_only(chunks: &[[&[Pq2Activation]; 8]]) -> Option<usize> {
    if chunks.is_empty() || chunks.len() > 8 {
        return None;
    }
    let count = chunks[0][0].len();
    if count == 0 || count.checked_mul(34).is_none() {
        return None;
    }
    chunks
        .iter()
        .flatten()
        .all(|activation| {
            activation.len() == count && activation.iter().all(|block| block.scale.is_finite())
        })
        .then_some(count)
}

pub(super) fn valid_pq2_tokens_only(tokens: &[&[Pq2Activation]]) -> Option<usize> {
    if tokens.is_empty() || tokens.len() > 8 {
        return None;
    }
    let count = tokens[0].len();
    if count == 0 || count.checked_mul(34).is_none() {
        return None;
    }
    tokens
        .iter()
        .all(|activation| {
            activation.len() == count && activation.iter().all(|block| block.scale.is_finite())
        })
        .then_some(count)
}

#[cfg(not(target_arch = "aarch64"))]
fn portable_pack_pq2_i8mm_pair_block(b0: &Pq2Activation, b1: &Pq2Activation) -> Pq2I8mmPairBlock {
    let mut planes = [([0_i8; 16], [0_i8; 16]); 8];
    for (plane_idx, dst) in planes.iter_mut().enumerate() {
        let off = plane_idx * 16;
        let p0 = &b0.values[off..off + 16];
        let p1 = &b1.values[off..off + 16];
        dst.0[..8].copy_from_slice(&p0[..8]);
        dst.0[8..].copy_from_slice(&p1[..8]);
        dst.1[..8].copy_from_slice(&p0[8..]);
        dst.1[8..].copy_from_slice(&p1[8..]);
    }
    Pq2I8mmPairBlock {
        planes,
        sums: [b0.sum, b1.sum],
        scales: [b0.scale, b1.scale],
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub(super) fn portable_append_i8mm_pair_blocks(
    t0: &[Pq2Activation],
    t1: &[Pq2Activation],
    out: &mut Vec<Pq2I8mmPairBlock>,
) {
    out.reserve(t0.len());
    for (b0, b1) in t0.iter().zip(t1.iter()) {
        out.push(portable_pack_pq2_i8mm_pair_block(b0, b1));
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub(super) fn portable_pack_i8mm_pair_blocks_into(
    t0: &[Pq2Activation],
    t1: &[Pq2Activation],
    out: &mut [Pq2I8mmPairBlock],
) {
    for (dst, (b0, b1)) in out.iter_mut().zip(t0.iter().zip(t1.iter())) {
        *dst = portable_pack_pq2_i8mm_pair_block(b0, b1);
    }
}

#[cfg(any(test, not(target_arch = "aarch64")))]
pub(super) fn portable_scaled_8x8_chunks_up_to_8<const FUSED: bool>(
    rows: [&[u8]; 8],
    chunks: &[[&[Pq2Activation]; 8]],
    outputs: &mut [[[f32; 8]; 8]],
) -> Option<()> {
    for (c, o) in chunks.chunks(4).zip(outputs.chunks_mut(4)) {
        portable_scaled_8x8_chunks::<FUSED>(rows, c, o)?;
    }
    Some(())
}

#[cfg(any(test, not(target_arch = "aarch64")))]
pub(super) fn portable_scaled_8x8_chunks<const FUSED: bool>(
    rows: [&[u8]; 8],
    chunks: &[[&[Pq2Activation]; 8]],
    outputs: &mut [[[f32; 8]; 8]],
) -> Option<()> {
    if !valid_scaled_8x8_chunks(rows, chunks, outputs.len()) {
        return None;
    }
    for (chunk, out_chunk) in chunks.iter().zip(outputs.iter_mut()) {
        for (tok, activation) in chunk.iter().enumerate() {
            let col = portable_validated::<8, FUSED>(rows, activation)?;
            for (row, &val) in col.iter().enumerate() {
                out_chunk[row][tok] = val;
            }
        }
    }
    Some(())
}

#[cfg(any(test, not(target_arch = "aarch64")))]
pub(super) fn portable_scaled_8x_tokens<const FUSED: bool>(
    rows: [&[u8]; 8],
    tokens: &[&[Pq2Activation]],
    out: &mut [[f32; 8]; 8],
) -> Option<()> {
    if !valid_scaled_8x_tokens(rows, tokens) {
        return None;
    }
    *out = [[0.0_f32; 8]; 8];
    for (tok, activation) in tokens.iter().enumerate() {
        let col = portable_validated::<8, FUSED>(rows, activation)?;
        for (row, &val) in col.iter().enumerate() {
            out[row][tok] = val;
        }
    }
    Some(())
}

pub(super) fn valid_ptq1_activation<A: Q8ActivationView>(activation: &[A]) -> Option<usize> {
    if activation.is_empty() || !activation.len().is_multiple_of(4) {
        return None;
    }
    let block_count = activation.len() / 4;
    block_count.checked_mul(28)?;
    activation
        .iter()
        .all(|block| block.scale().is_finite())
        .then_some(block_count)
}

pub(super) fn valid_ptq1_packed_activation(activation: &[Ptq1Activation]) -> Option<usize> {
    let block_count = activation.len();
    if block_count == 0 {
        return None;
    }
    block_count.checked_mul(28)?;
    activation
        .iter()
        .all(|block| block.scales.iter().all(|scale| scale.is_finite()))
        .then_some(block_count)
}

pub(super) fn valid_ptq1_row_extents<const ROWS: usize>(
    rows: [&[u8]; ROWS],
    block_count: usize,
) -> bool {
    matches!(ROWS, 1 | 4 | 8 | 16)
        && block_count
            .checked_mul(28)
            .is_some_and(|bytes| rows.iter().all(|row| row.len() == bytes))
}

pub(super) fn valid_ptq1_rows<const ROWS: usize, A: Q8ActivationView>(
    rows: [&[u8]; ROWS],
    activation: &[A],
) -> bool {
    valid_ptq1_activation(activation)
        .is_some_and(|block_count| valid_ptq1_row_extents(rows, block_count))
}

pub(super) fn valid_ptq1_8x_tokens<A: Q8ActivationView>(rows: [&[u8]; 8], tokens: &[&[A]]) -> bool {
    if tokens.is_empty() || tokens.len() > 8 {
        return false;
    }
    let len = tokens[0].len();
    if len == 0 || !len.is_multiple_of(4) {
        return false;
    }
    let block_count = len / 4;
    block_count
        .checked_mul(28)
        .is_some_and(|bytes| rows.iter().all(|row| row.len() == bytes))
        && tokens
            .iter()
            .all(|tok| tok.len() == len && tok.iter().all(|block| block.scale().is_finite()))
}

pub(super) fn valid_ptq1_8x8_chunks<A: Q8ActivationView>(
    rows: [&[u8]; 8],
    chunks: &[[&[A]; 8]],
    outputs_len: usize,
) -> bool {
    if chunks.is_empty() || chunks.len() > 4 || outputs_len != chunks.len() {
        return false;
    }
    let len = chunks[0][0].len();
    if len == 0 || !len.is_multiple_of(4) {
        return false;
    }
    let block_count = len / 4;
    block_count
        .checked_mul(28)
        .is_some_and(|bytes| rows.iter().all(|row| row.len() == bytes))
        && chunks
            .iter()
            .flatten()
            .all(|tok| tok.len() == len && tok.iter().all(|block| block.scale().is_finite()))
}

pub(super) fn valid_ptq1_8x8_chunks_up_to_8<A: Q8ActivationView>(
    rows: [&[u8]; 8],
    chunks: &[[&[A]; 8]],
    outputs_len: usize,
) -> Option<usize> {
    let first_len = chunks.first().map(|chunk| chunk[0].len())?;
    if chunks.len() > 8
        || outputs_len != chunks.len()
        || !chunks
            .chunks(4)
            .all(|c| c[0][0].len() == first_len && valid_ptq1_8x8_chunks(rows, c, c.len()))
    {
        return None;
    }
    Some(first_len)
}

#[cfg(not(target_arch = "aarch64"))]
fn portable_pack_ptq1_pinned_i8mm_pair_block<A: Q8ActivationView>(
    b0: &A,
    b1: &A,
) -> Ptq1PinnedI8mmPairBlock {
    let v0 = b0.values();
    let v1 = b1.values();
    let mut planes = [[0_i8; 16]; 4];
    for (lane, dst) in planes.iter_mut().enumerate() {
        let off = lane * 4;
        dst[0..4].copy_from_slice(&v0[off..off + 4]);
        dst[4..8].copy_from_slice(&v0[16 + off..16 + off + 4]);
        dst[8..12].copy_from_slice(&v1[off..off + 4]);
        dst[12..16].copy_from_slice(&v1[16 + off..16 + off + 4]);
    }
    Ptq1PinnedI8mmPairBlock {
        planes,
        scales: [b0.scale(), b1.scale()],
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub(super) fn portable_append_ptq1_pinned_i8mm_pair_blocks<A: Q8ActivationView>(
    t0: &[A],
    t1: &[A],
    out: &mut Vec<Ptq1PinnedI8mmPairBlock>,
) {
    out.reserve(t0.len());
    for (b0, b1) in t0.iter().zip(t1.iter()) {
        out.push(portable_pack_ptq1_pinned_i8mm_pair_block(b0, b1));
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub(super) fn portable_pack_ptq1_pinned_i8mm_pair_blocks_into<A: Q8ActivationView>(
    t0: &[A],
    t1: &[A],
    out: &mut [Ptq1PinnedI8mmPairBlock],
) {
    for (dst, (b0, b1)) in out.iter_mut().zip(t0.iter().zip(t1.iter())) {
        *dst = portable_pack_ptq1_pinned_i8mm_pair_block(b0, b1);
    }
}

pub(super) fn valid_q8_0_activation<A: Q8ActivationView>(activation: &[A]) -> Option<usize> {
    let block_count = activation.len();
    if block_count == 0
        || block_count.checked_mul(34).is_none()
        || !activation.iter().all(|block| block.scale().is_finite())
    {
        return None;
    }
    Some(block_count)
}

pub(super) fn valid_q8_0_rows_count<const ROWS: usize>(
    rows: [&[u8]; ROWS],
    block_count: usize,
) -> bool {
    matches!(ROWS, 1 | 4 | 8 | 16)
        && block_count != 0
        && block_count
            .checked_mul(34)
            .is_some_and(|bytes| rows.iter().all(|row| row.len() == bytes))
}

pub(super) fn valid_q8_0_rows<const ROWS: usize, A: Q8ActivationView>(
    rows: [&[u8]; ROWS],
    activation: &[A],
) -> bool {
    valid_q8_0_rows_count(rows, activation.len())
        && activation.iter().all(|block| block.scale().is_finite())
}

pub(super) fn valid_q8_0_8x_tokens<A: Q8ActivationView>(rows: [&[u8]; 8], tokens: &[&[A]]) -> bool {
    if tokens.is_empty() || tokens.len() > 8 {
        return false;
    }
    let block_count = tokens[0].len();
    block_count != 0
        && block_count
            .checked_mul(34)
            .is_some_and(|bytes| rows.iter().all(|row| row.len() == bytes))
        && tokens.iter().all(|tok| {
            tok.len() == block_count && tok.iter().all(|block| block.scale().is_finite())
        })
}

pub(super) fn valid_q8_0_8x8_chunks<A: Q8ActivationView>(
    rows: [&[u8]; 8],
    chunks: &[[&[A]; 8]],
    outputs_len: usize,
) -> bool {
    if chunks.is_empty() || chunks.len() > 4 || outputs_len != chunks.len() {
        return false;
    }
    let block_count = chunks[0][0].len();
    block_count != 0
        && block_count
            .checked_mul(34)
            .is_some_and(|bytes| rows.iter().all(|row| row.len() == bytes))
        && chunks.iter().flatten().all(|tok| {
            tok.len() == block_count && tok.iter().all(|block| block.scale().is_finite())
        })
}

#[cfg(any(test, not(target_arch = "aarch64")))]
#[expect(
    clippy::suboptimal_flops,
    reason = "PTQ1 SOURCE mode requires separate multiply and addition rounding."
)]
fn portable_ptq1_rows_with_groups<'a, const ROWS: usize, const PINNED: bool>(
    rows: [&[u8]; ROWS],
    group_at: impl Fn(usize, usize) -> (f32, &'a [i8; 32]),
) -> Option<[f32; ROWS]> {
    let mut output = [0.0_f32; ROWS];
    for (row, dst) in rows.into_iter().zip(&mut output) {
        let mut lanes = [0.0_f32; 4];
        let mut sum = 0.0_f32;
        for (block_idx, block) in row.as_chunks::<28>().0.iter().enumerate() {
            let weight_scale =
                bonsai_ternary::half_to_float(u16::from_le_bytes([block[26], block[27]]));
            if !weight_scale.is_finite() {
                return None;
            }
            let trits = crate::decode_ptq1_trits_i8(block);
            if PINNED {
                for group in 0..4 {
                    let (act_scale, act_vals) = group_at(block_idx, group);
                    let w0 = &trits[group * 2];
                    let w1 = &trits[group * 2 + 1];
                    let d = weight_scale * act_scale;
                    for (lane, lane_acc) in lanes.iter_mut().enumerate() {
                        let base = lane * 4;
                        let integer: i32 = (0..4)
                            .map(|k| {
                                i32::from(w0[base + k]) * i32::from(act_vals[base + k])
                                    + i32::from(w1[base + k]) * i32::from(act_vals[16 + base + k])
                            })
                            .sum();
                        let f = f32::from(i16::try_from(integer).ok()?);
                        *lane_acc = f.mul_add(d, *lane_acc);
                    }
                }
            } else {
                let mut dot_sum = 0.0_f32;
                for group in 0..4 {
                    let (act_scale, act_vals) = group_at(block_idx, group);
                    let w0 = &trits[group * 2];
                    let w1 = &trits[group * 2 + 1];
                    let integer: i32 = (0..16)
                        .map(|k| {
                            i32::from(w0[k]) * i32::from(act_vals[k])
                                + i32::from(w1[k]) * i32::from(act_vals[16 + k])
                        })
                        .sum();
                    let prod = f32::from(i16::try_from(integer).ok()?) * act_scale;
                    dot_sum += prod;
                }
                sum += weight_scale * dot_sum;
            }
        }
        let value = if PINNED {
            (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
        } else {
            sum
        };
        if !value.is_finite() {
            return None;
        }
        *dst = value;
    }
    Some(output)
}

#[cfg(any(test, not(target_arch = "aarch64")))]
pub(super) fn portable_ptq1_rows_validated<
    const ROWS: usize,
    const PINNED: bool,
    A: Q8ActivationView,
>(
    rows: [&[u8]; ROWS],
    activation: &[A],
) -> Option<[f32; ROWS]> {
    portable_ptq1_rows_with_groups::<ROWS, PINNED>(rows, |block_idx, group| {
        let act = &activation[block_idx * 4 + group];
        (act.scale(), act.values())
    })
}

#[cfg(any(test, not(target_arch = "aarch64")))]
pub(super) fn portable_ptq1_packed_rows_validated<const ROWS: usize, const PINNED: bool>(
    rows: [&[u8]; ROWS],
    activation: &[Ptq1Activation],
) -> Option<[f32; ROWS]> {
    portable_ptq1_rows_with_groups::<ROWS, PINNED>(rows, |block_idx, group| {
        let act = &activation[block_idx];
        let groups = act.values.as_chunks::<32>().0;
        (act.scales[group], &groups[group])
    })
}

#[cfg(any(test, not(target_arch = "aarch64")))]
pub(super) fn portable_ptq1_8x_tokens_validated<const PINNED: bool, A: Q8ActivationView>(
    rows: [&[u8]; 8],
    tokens: &[&[A]],
    out: &mut [[f32; 8]; 8],
) -> Option<()> {
    *out = [[0.0_f32; 8]; 8];
    for (tok, activation) in tokens.iter().enumerate() {
        let col = portable_ptq1_rows_validated::<8, PINNED, A>(rows, activation)?;
        for (row, &val) in col.iter().enumerate() {
            out[row][tok] = val;
        }
    }
    Some(())
}

#[cfg(any(test, not(target_arch = "aarch64")))]
pub(super) fn portable_ptq1_8x8_chunks_validated<const PINNED: bool, A: Q8ActivationView>(
    rows: [&[u8]; 8],
    chunks: &[[&[A]; 8]],
    outputs: &mut [[[f32; 8]; 8]],
) -> Option<()> {
    let mut out_buf = [[[0.0_f32; 8]; 8]; 4];
    for (chunk, out_chunk) in chunks.iter().zip(out_buf.iter_mut()) {
        portable_ptq1_8x_tokens_validated::<PINNED, A>(rows, chunk, out_chunk)?;
    }
    for (dst, src) in outputs.iter_mut().zip(out_buf.iter()) {
        *dst = *src;
    }
    Some(())
}

#[expect(
    clippy::cast_precision_loss,
    clippy::as_conversions,
    clippy::suboptimal_flops,
    reason = "32-element i8*i8 dot fits in f32 mantissa and non-FMA multiply-then-add matches scalar Q8_0 rounding."
)]
pub(super) fn portable_q8_0_rows_validated<const ROWS: usize, A: Q8ActivationView>(
    rows: [&[u8]; ROWS],
    activation: &[A],
) -> Option<[f32; ROWS]> {
    let mut output = [0.0_f32; ROWS];
    for (row, dst) in rows.into_iter().zip(&mut output) {
        let mut sum = 0.0_f32;
        for (block, act) in row.as_chunks::<34>().0.iter().zip(activation) {
            let row_scale = bonsai_ternary::half_to_float(u16::from_le_bytes([block[0], block[1]]));
            if !row_scale.is_finite() {
                return None;
            }
            let act_vals = act.values();
            let int_dot: i32 = block[2..34]
                .iter()
                .zip(act_vals)
                .map(|(&w, &a)| i32::from(i8::from_ne_bytes([w])) * i32::from(a))
                .sum();
            sum += (row_scale * act.scale()) * (int_dot as f32);
        }
        if !sum.is_finite() {
            return None;
        }
        *dst = sum;
    }
    Some(output)
}

#[cfg(any(test, not(target_arch = "aarch64")))]
#[expect(
    clippy::cast_precision_loss,
    clippy::as_conversions,
    clippy::suboptimal_flops,
    reason = "32-element i8*i8 dot fits in f32 mantissa and non-FMA multiply-then-add matches scalar Q8_0 rounding."
)]
pub(super) fn portable_q8_0_8x_tokens_validated<A: Q8ActivationView>(
    rows: [&[u8]; 8],
    tokens: &[&[A]],
    out: &mut [[f32; 8]; 8],
) -> Option<()> {
    let block_count = tokens[0].len();
    let row_chunks = rows.map(|row| row.as_chunks::<34>().0);
    let mut acc = [[0.0_f32; 8]; 8];
    for block_idx in 0..block_count {
        let row_scales: [f32; 8] = std::array::from_fn(|r| {
            let block = &row_chunks[r][block_idx];
            bonsai_ternary::half_to_float(u16::from_le_bytes([block[0], block[1]]))
        });
        if !row_scales.iter().all(|scale| scale.is_finite()) {
            return None;
        }
        for (t, token) in tokens.iter().enumerate() {
            let act = &token[block_idx];
            let act_scale = act.scale();
            let act_vals = act.values();
            for r in 0..8 {
                let w_bytes = &row_chunks[r][block_idx][2..34];
                let int_dot: i32 = w_bytes
                    .iter()
                    .zip(act_vals)
                    .map(|(&w, &a)| i32::from(i8::from_ne_bytes([w])) * i32::from(a))
                    .sum();
                acc[r][t] += (row_scales[r] * act_scale) * (int_dot as f32);
            }
        }
    }
    if acc.iter().flatten().any(|sum| !sum.is_finite()) {
        return None;
    }
    *out = acc;
    Some(())
}

#[cfg(any(test, not(target_arch = "aarch64")))]
pub(super) fn portable_q8_0_8x8_chunks_validated<A: Q8ActivationView>(
    rows: [&[u8]; 8],
    chunks: &[[&[A]; 8]],
    outputs: &mut [[[f32; 8]; 8]],
) -> Option<()> {
    let mut out_buf = [[[0.0_f32; 8]; 8]; 4];
    for (chunk, out_chunk) in chunks.iter().zip(out_buf.iter_mut()) {
        portable_q8_0_8x_tokens_validated::<A>(rows, chunk, out_chunk)?;
    }
    for (dst, src) in outputs.iter_mut().zip(out_buf.iter()) {
        *dst = *src;
    }
    Some(())
}

#[cfg(any(test, not(target_arch = "aarch64")))]
pub(super) fn portable<const ROWS: usize, const FUSED: bool>(
    rows: [&[u8]; ROWS],
    activation: &[Pq2Activation],
) -> Option<[f32; ROWS]> {
    if !valid(rows, activation) {
        return None;
    }
    portable_validated::<ROWS, FUSED>(rows, activation)
}

#[cfg(any(test, not(target_arch = "aarch64")))]
#[expect(
    clippy::suboptimal_flops,
    reason = "The nonfused contract requires separate product and addition rounding."
)]
pub(super) fn portable_validated<const ROWS: usize, const FUSED: bool>(
    rows: [&[u8]; ROWS],
    activation: &[Pq2Activation],
) -> Option<[f32; ROWS]> {
    let mut output = [0.0_f32; ROWS];
    for (row, output) in rows.into_iter().zip(&mut output) {
        for (bytes, activation) in row.as_chunks::<34>().0.iter().zip(activation) {
            let scale = bonsai_ternary::half_to_float(u16::from_le_bytes([bytes[0], bytes[1]]));
            if !scale.is_finite() {
                return None;
            }
            let mut dot = 0_i32;
            for (half, bytes) in bytes[2..].as_chunks::<16>().0.iter().enumerate() {
                for (index, &byte) in bytes.iter().enumerate() {
                    for digit in 0..4 {
                        dot += i32::from((byte >> (digit * 2)) & 3)
                            * i32::from(activation.values[half * 64 + digit * 16 + index]);
                    }
                }
            }
            // Public callers supply the correction; wrapping matches SIMD
            // subtraction even for an inconsistent but memory-safe input.
            let dot = dot.wrapping_sub(activation.sum);
            #[expect(
                clippy::cast_precision_loss,
                clippy::as_conversions,
                reason = "The exact PQ2 dot bound fits binary32; arbitrary safe caller corrections retain the existing integer-to-float contract."
            )]
            let dot = dot as f32;
            let scale = scale * activation.scale;
            *output = if FUSED {
                scale.mul_add(dot, *output)
            } else {
                *output + scale * dot
            };
        }
        if !output.is_finite() {
            return None;
        }
    }
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::{Pq2Activation, portable, portable_q8_0_8x_tokens_validated};

    #[test]
    fn x1_scoped_pq2_matches_reference_and_revalidates_replaced_activations() {
        #[cfg(target_arch = "aarch64")]
        let Some(detected) = crate::Pq2Dotprod::detect() else {
            return;
        };
        // This test-only capability selects exclusively portable arithmetic on
        // non-ARM targets. It never reaches a target-feature instruction.
        #[cfg(not(target_arch = "aarch64"))]
        let detected = crate::Pq2Dotprod { _detected: () };
        let mut activation = [
            Pq2Activation {
                scale: f32::from_bits(0x3da0_0001),
                values: [3; 128],
                sum: 384,
            },
            Pq2Activation {
                scale: f32::from_bits(0x3da0_0001),
                values: [-13; 128],
                sum: -1664,
            },
        ];
        let mut row = [0xaa; 68];
        row[..2].copy_from_slice(&0x3c01_u16.to_le_bytes());
        row[34..36].copy_from_slice(&0x3787_u16.to_le_bytes());
        {
            let prepared = detected.prepare_scaled(&activation);
            assert!(prepared.is_some());
            if let Some(prepared) = prepared {
                for _ in 0..3 {
                    assert_eq!(
                        prepared
                            .dot_scaled_rows::<1, true>([&row])
                            .map(|v| v.map(f32::to_bits)),
                        Some([0xc1f9_0b01])
                    );
                    assert_eq!(
                        prepared
                            .dot_scaled_rows::<4, false>([&row; 4])
                            .map(|v| v.map(f32::to_bits)),
                        Some([0xc1f9_0b02; 4])
                    );
                    assert_eq!(
                        prepared
                            .dot_scaled_rows::<8, true>([&row; 8])
                            .map(|v| v.map(f32::to_bits)),
                        portable::<8, true>([&row; 8], &activation).map(|v| v.map(f32::to_bits))
                    );
                }
                assert!(prepared.dot_scaled_rows::<2, true>([&row; 2]).is_none());
                assert!(
                    prepared
                        .dot_scaled_rows::<8, true>([&row[..67]; 8])
                        .is_none()
                );
                row[34..36].copy_from_slice(&0x7c01_u16.to_le_bytes());
                assert!(prepared.dot_scaled_rows::<8, true>([&row; 8]).is_none());
            }
        }
        activation[1].scale = f32::NAN;
        assert!(detected.prepare_scaled(&activation).is_none());
        assert!(detected.prepare_scaled(&[]).is_none());
        activation[1].scale = -0.125;
        row[34..36].copy_from_slice(&0x3787_u16.to_le_bytes());
        let expected = portable::<8, true>([&row; 8], &activation).map(|v| v.map(f32::to_bits));
        assert_eq!(
            detected
                .prepare_scaled(&activation)
                .and_then(|scope| scope.dot_scaled_rows::<8, true>([&row; 8]))
                .map(|v| v.map(f32::to_bits)),
            expected
        );
    }

    #[test]
    fn x1_scaled_pq2_rows_preserve_independent_fma_control_and_reject_bad_inputs() {
        let activation = [
            Pq2Activation {
                scale: f32::from_bits(0x3da0_0001),
                values: [3; 128],
                sum: 384,
            },
            Pq2Activation {
                scale: f32::from_bits(0x3da0_0001),
                values: [-13; 128],
                sum: -1664,
            },
        ];
        let mut row = [0xaa; 68];
        row[..2].copy_from_slice(&0x3c01_u16.to_le_bytes());
        row[34..36].copy_from_slice(&0x3787_u16.to_le_bytes());
        assert_eq!(
            portable::<1, true>([&row], &activation).map(|v| v.map(f32::to_bits)),
            Some([0xc1f9_0b01])
        );
        assert_eq!(
            portable::<1, false>([&row], &activation).map(|v| v.map(f32::to_bits)),
            Some([0xc1f9_0b02])
        );
        if let Some(detected) = crate::Pq2Dotprod::detect() {
            assert_eq!(
                detected
                    .dot_scaled_rows::<1, true>([&row], &activation)
                    .map(|v| v.map(f32::to_bits)),
                Some([0xc1f9_0b01])
            );
            assert_eq!(
                detected
                    .dot_scaled_rows::<4, false>([&row; 4], &activation)
                    .map(|v| v.map(f32::to_bits)),
                Some([0xc1f9_0b02; 4])
            );
            assert_eq!(
                detected
                    .dot_scaled_rows::<8, true>([&row; 8], &activation)
                    .map(|v| v.map(f32::to_bits)),
                Some([0xc1f9_0b01; 8])
            );
            assert!(
                detected
                    .dot_scaled_rows::<1, true>([&row[..67]], &activation)
                    .is_none()
            );
            assert!(
                detected
                    .dot_scaled_rows::<2, true>([&row; 2], &activation)
                    .is_none()
            );
            row[34..36].copy_from_slice(&0x7c00_u16.to_le_bytes());
            assert!(
                detected
                    .dot_scaled_rows::<8, true>([&row; 8], &activation)
                    .is_none()
            );
        }
        assert!(portable::<1, true>([&[]], &[]).is_none());
        assert!(portable::<1, true>([&row[..34]], &activation).is_none());
    }

    #[test]
    fn x1_scaled_pq2_rows_match_all_symbols_extremes_and_scale_orders() {
        for count in [1, 2, 3, 40] {
            let activation: Vec<_> = (0..count)
                .map(|block: usize| {
                    let values: [i8; 128] = std::array::from_fn(|lane: usize| {
                        i8::from_ne_bytes([lane.to_le_bytes()[0]
                            .wrapping_mul(71)
                            .wrapping_add(block.to_le_bytes()[0])])
                    });
                    Pq2Activation {
                        scale: [0.0, -0.0, 0.125, -0.375][block % 4],
                        sum: values.iter().map(|&v| i32::from(v)).sum(),
                        values,
                    }
                })
                .collect();
            let rows: [Vec<u8>; 8] = std::array::from_fn(|row: usize| {
                let mut bytes: Vec<u8> = (0..count * 34)
                    .map(|index: usize| {
                        index.to_le_bytes()[0]
                            .wrapping_mul(73)
                            .wrapping_add(row.to_le_bytes()[0])
                    })
                    .collect();
                for (index, block) in bytes.as_chunks_mut::<34>().0.iter_mut().enumerate() {
                    block[..2].copy_from_slice(
                        &[0x0001_u16, 0x8000, 0x3555, 0xb787][(index + row) % 4].to_le_bytes(),
                    );
                }
                bytes
            });
            let refs = rows.each_ref().map(Vec::as_slice);
            let expected_fused =
                portable::<8, true>(refs, &activation).map(|v| v.map(f32::to_bits));
            let expected_separate =
                portable::<8, false>(refs, &activation).map(|v| v.map(f32::to_bits));
            if let Some(detected) = crate::Pq2Dotprod::detect() {
                assert_eq!(
                    detected
                        .dot_scaled_rows::<8, true>(refs, &activation)
                        .map(|v| v.map(f32::to_bits)),
                    expected_fused
                );
                assert_eq!(
                    detected
                        .dot_scaled_rows::<8, false>(refs, &activation)
                        .map(|v| v.map(f32::to_bits)),
                    expected_separate
                );
                for (row, &bytes) in refs.iter().enumerate() {
                    assert_eq!(
                        detected
                            .dot_scaled_rows::<1, true>([bytes], &activation)
                            .map(|v| v[0].to_bits()),
                        expected_fused.map(|v| v[row])
                    );
                }
            }
        }
    }

    #[test]
    fn x1_scaled_pq2_rows_preserve_16_row_parity_and_errors() {
        let mut raw_values = [[0_i8; 128]; 2];
        for (block_idx, block) in raw_values.iter_mut().enumerate() {
            for (elem_idx, slot) in block.iter_mut().enumerate() {
                let raw = i8::try_from((block_idx * 43 + elem_idx * 17) % 199).unwrap_or(0);
                *slot = raw.wrapping_sub(99);
            }
        }
        let activation = [
            Pq2Activation {
                values: raw_values[0],
                scale: 0.125,
                sum: raw_values[0].iter().map(|&v| i32::from(v)).sum(),
            },
            Pq2Activation {
                values: raw_values[1],
                scale: -0.0625,
                sum: raw_values[1].iter().map(|&v| i32::from(v)).sum(),
            },
        ];
        let scale_pool: [u16; 16] = [
            0x3c00, 0xbc00, 0x3800, 0xb800, 0x4000, 0xc000, 0x3400, 0xb400, 0x8000, 0x0000, 0x3e00,
            0xbe00, 0x4200, 0xc200, 0x3600, 0xb600,
        ];
        let mut rows = [[0_u8; 68]; 16];
        for (row_idx, row) in rows.iter_mut().enumerate() {
            for (block_idx, scale_bits) in [scale_pool[row_idx], scale_pool[(row_idx + 7) % 16]]
                .into_iter()
                .enumerate()
            {
                let start = block_idx * 34;
                let le = scale_bits.to_le_bytes();
                row[start] = le[0];
                row[start + 1] = le[1];
                for byte_idx in 0..32 {
                    row[start + 2 + byte_idx] =
                        u8::try_from((row_idx * 29 + block_idx * 53 + byte_idx * 13 + 3) % 256)
                            .unwrap_or(0);
                }
            }
        }
        let refs: [&[u8]; 16] = std::array::from_fn(|row| rows[row].as_slice());
        let expected_fused = portable::<16, true>(refs, &activation);
        let expected_unfused = portable::<16, false>(refs, &activation);
        assert!(expected_fused.is_some());
        assert!(expected_unfused.is_some());

        if let Some(detected) = crate::Pq2Dotprod::detect() {
            assert_eq!(
                detected
                    .dot_scaled_rows::<16, true>(refs, &activation)
                    .map(|values| values.map(f32::to_bits)),
                expected_fused.map(|values| values.map(f32::to_bits))
            );
            assert_eq!(
                detected
                    .dot_scaled_rows::<16, false>(refs, &activation)
                    .map(|values| values.map(f32::to_bits)),
                expected_unfused.map(|values| values.map(f32::to_bits))
            );
            assert_eq!(
                detected
                    .prepare_scaled(&activation)
                    .and_then(|prepared| prepared.dot_scaled_rows::<16, true>(refs))
                    .map(|values| values.map(f32::to_bits)),
                expected_fused.map(|values| values.map(f32::to_bits))
            );
            for (row, &bytes) in refs.iter().enumerate() {
                assert_eq!(
                    detected
                        .dot_scaled_rows::<1, true>([bytes], &activation)
                        .map(|value| value[0].to_bits()),
                    expected_fused.map(|values| values[row].to_bits())
                );
            }

            let mut short_refs = refs;
            short_refs[15] = &rows[15][..67];
            assert!(
                detected
                    .dot_scaled_rows::<16, true>(short_refs, &activation)
                    .is_none()
            );

            let mut bad_scale_rows = rows;
            let nan_bits = 0x7e00_u16.to_le_bytes();
            bad_scale_rows[11][34] = nan_bits[0];
            bad_scale_rows[11][35] = nan_bits[1];
            let bad_refs: [&[u8]; 16] = std::array::from_fn(|row| bad_scale_rows[row].as_slice());
            assert!(
                detected
                    .dot_scaled_rows::<16, true>(bad_refs, &activation)
                    .is_none()
            );

            let mut inf_act = activation;
            inf_act[1].scale = f32::INFINITY;
            assert!(
                detected
                    .dot_scaled_rows::<16, true>(refs, &inf_act)
                    .is_none()
            );
        }
    }

    fn sample_scaled_8x8_fixture(
        block_count: usize,
        seed: usize,
    ) -> ([Vec<u8>; 8], Vec<Vec<Pq2Activation>>) {
        let f16_pool: [u16; 8] = [
            0x0000, 0x8000, 0x0001, 0x8001, 0x3c00, 0xbc00, 0x3555, 0xb787,
        ];
        let act_scale_pool: [f32; 8] = [
            0.0_f32,
            -0.0_f32,
            f32::from_bits(1),
            -f32::from_bits(1),
            0.0625_f32,
            -0.125_f32,
            0.375_f32,
            -0.25_f32,
        ];
        let owned_rows: [Vec<u8>; 8] = std::array::from_fn(|row_idx| {
            let mut row = vec![0_u8; block_count * 34];
            for blk in 0..block_count {
                let start = blk * 34;
                let s = f16_pool[(row_idx + blk * 3 + seed) % f16_pool.len()].to_le_bytes();
                row[start] = s[0];
                row[start + 1] = s[1];
                for i in 0..32 {
                    row[start + 2 + i] =
                        u8::try_from((row_idx * 41 + blk * 59 + i * 19 + seed * 13 + 5) % 256)
                            .unwrap_or(0);
                }
            }
            row
        });
        let act_storage: Vec<Vec<Pq2Activation>> = (0..32_usize)
            .map(|tok| {
                (0..block_count)
                    .map(|blk| {
                        let values: [i8; 128] = std::array::from_fn(|j| {
                            let raw =
                                i16::try_from((tok * 13 + blk * 29 + j * 7 + seed * 11) % 120)
                                    .unwrap_or(0);
                            i8::try_from(raw - 60).unwrap_or(0)
                        });
                        let sum = values.iter().map(|&v| i32::from(v)).sum();
                        let scale = act_scale_pool[(tok + blk * 3 + seed) % act_scale_pool.len()];
                        Pq2Activation { scale, values, sum }
                    })
                    .collect()
            })
            .collect();
        (owned_rows, act_storage)
    }

    #[test]
    fn x1_scaled_8x8_chunks_matches_portable() {
        use super::portable_scaled_8x8_chunks;

        #[cfg(target_arch = "aarch64")]
        let Some(detected) = crate::Pq2Dotprod::detect() else {
            return;
        };
        #[cfg(not(target_arch = "aarch64"))]
        let detected = crate::Pq2Dotprod { _detected: () };

        #[cfg(target_arch = "aarch64")]
        let caps = [
            detected,
            crate::Pq2Dotprod {
                i8mm: false,
                ..detected
            },
        ];
        #[cfg(not(target_arch = "aarch64"))]
        let caps = [detected];

        for (seed, block_count) in [1_usize, 2, 3, 4, 5, 7, 8, 9, 16, 17, 24, 32]
            .into_iter()
            .enumerate()
        {
            let (owned_rows, act_storage) = sample_scaled_8x8_fixture(block_count, seed + 1);
            let row_refs: [&[u8]; 8] = std::array::from_fn(|r| owned_rows[r].as_slice());
            let chunk_slices: [[&[Pq2Activation]; 8]; 4] =
                std::array::from_fn(|c| std::array::from_fn(|t| act_storage[c * 8 + t].as_slice()));

            for count in 1..=4 {
                let active_chunks = &chunk_slices[..count];
                let mut expected_fused = vec![[[0.0_f32; 8]; 8]; count];
                let mut expected_unfused = vec![[[0.0_f32; 8]; 8]; count];
                assert!(
                    portable_scaled_8x8_chunks::<true>(
                        row_refs,
                        active_chunks,
                        &mut expected_fused
                    )
                    .is_some()
                );
                assert!(
                    portable_scaled_8x8_chunks::<false>(
                        row_refs,
                        active_chunks,
                        &mut expected_unfused
                    )
                    .is_some()
                );

                for &cap in &caps {
                    let mut actual_fused = vec![[[123.0_f32; 8]; 8]; count];
                    assert!(
                        cap.dot_scaled_8x8_chunks::<true>(
                            row_refs,
                            active_chunks,
                            &mut actual_fused
                        )
                        .is_some()
                    );
                    for (e, a) in expected_fused.iter().zip(actual_fused.iter()) {
                        for r in 0..8 {
                            assert_eq!(e[r].map(f32::to_bits), a[r].map(f32::to_bits));
                        }
                    }

                    let mut actual_unfused = vec![[[-1.0_f32; 8]; 8]; count];
                    assert!(
                        cap.dot_scaled_8x8_chunks::<false>(
                            row_refs,
                            active_chunks,
                            &mut actual_unfused
                        )
                        .is_some()
                    );
                    for (e, a) in expected_unfused.iter().zip(actual_unfused.iter()) {
                        for r in 0..8 {
                            assert_eq!(e[r].map(f32::to_bits), a[r].map(f32::to_bits));
                        }
                    }
                }
            }
        }
    }

    fn assert_chunk_bits_eq(expected: &[[[f32; 8]; 8]], actual: &[[[f32; 8]; 8]]) {
        for (e, a) in expected.iter().zip(actual.iter()) {
            for r in 0..8 {
                assert_eq!(e[r].map(f32::to_bits), a[r].map(f32::to_bits));
            }
        }
    }

    fn check_pq2_prepared_cap(
        cap: crate::Pq2Dotprod,
        row_refs: [&[u8]; 8],
        active_chunks: &[[&[Pq2Activation]; 8]],
        expected_fused: &[[[f32; 8]; 8]],
        expected_unfused: &[[[f32; 8]; 8]],
        block_count: usize,
    ) -> Result<(), &'static str> {
        let count = active_chunks.len();
        let prepared = cap
            .prepare_scaled_chunks(active_chunks)
            .ok_or("valid PQ2 chunks must prepare")?;
        assert_eq!(prepared.block_count(), block_count);
        let mut panel_storage = Vec::new();
        for chunk in active_chunks {
            for p in 0..4 {
                assert!(
                    cap.append_i8mm_pair_blocks(chunk[2 * p], chunk[2 * p + 1], &mut panel_storage)
                        .is_some()
                );
            }
        }
        let panel_chunks: Vec<[&[super::Pq2I8mmPairBlock]; 4]> = (0..count)
            .map(|c| {
                std::array::from_fn(|p| {
                    let start = (c * 4 + p) * block_count;
                    &panel_storage[start..start + block_count]
                })
            })
            .collect();

        let mut actual = vec![[[0.0_f32; 8]; 8]; count];
        assert!(
            prepared
                .dot_scaled_8x8_chunks::<true>(row_refs, &mut actual)
                .is_some()
        );
        assert_chunk_bits_eq(expected_fused, &actual);

        assert!(
            prepared
                .dot_scaled_8x8_chunks_with_panels::<true>(row_refs, &panel_chunks, &mut actual)
                .is_some()
        );
        assert_chunk_bits_eq(expected_fused, &actual);

        assert!(
            prepared
                .dot_scaled_8x8_chunks_with_panels::<false>(row_refs, &panel_chunks, &mut actual)
                .is_some()
        );
        assert_chunk_bits_eq(expected_unfused, &actual);

        let prepared_toks = cap
            .prepare_scaled_tokens(&active_chunks[0][..3])
            .ok_or("valid PQ2 remainder tokens must prepare")?;
        let mut actual_rem = [[0.0_f32; 8]; 8];
        assert!(
            prepared_toks
                .dot_scaled_8x_tokens::<true>(row_refs, &mut actual_rem)
                .is_some()
        );
        for r in 0..8 {
            for t in 0..3 {
                assert_eq!(
                    expected_fused[0][r][t].to_bits(),
                    actual_rem[r][t].to_bits()
                );
            }
        }
        Ok(())
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Verifies 1..=8 chunk PQ2 I8MM panel parity, slice-fill packing, and NaN/Inf scale rejection."
    )]
    #[test]
    fn x1_pq2_prepared_chunks_and_i8mm_panels_match_portable() -> Result<(), &'static str> {
        let Some(detected) = crate::Pq2Dotprod::detect() else {
            return Ok(());
        };
        #[cfg(target_arch = "aarch64")]
        let caps = [
            detected,
            crate::Pq2Dotprod {
                i8mm: false,
                ..detected
            },
        ];
        #[cfg(not(target_arch = "aarch64"))]
        let caps = [detected];

        for (seed, block_count) in [1_usize, 3, 4, 5, 7, 8, 9, 40, 136].into_iter().enumerate() {
            let (owned_rows, act_storage) = sample_scaled_8x8_fixture(block_count, seed + 19);
            let row_refs: [&[u8]; 8] = std::array::from_fn(|r| owned_rows[r].as_slice());
            let chunk_slices: [[&[Pq2Activation]; 8]; 8] = std::array::from_fn(|c| {
                std::array::from_fn(|t| act_storage[(c * 8 + t) % act_storage.len()].as_slice())
            });
            for count in 1..=8_usize {
                let active_chunks = &chunk_slices[..count];
                let mut expected_fused = vec![[[0.0_f32; 8]; 8]; count];
                let mut expected_unfused = vec![[[0.0_f32; 8]; 8]; count];
                assert!(
                    super::portable_scaled_8x8_chunks_up_to_8::<true>(
                        row_refs,
                        active_chunks,
                        &mut expected_fused
                    )
                    .is_some()
                );
                assert!(
                    super::portable_scaled_8x8_chunks_up_to_8::<false>(
                        row_refs,
                        active_chunks,
                        &mut expected_unfused
                    )
                    .is_some()
                );

                for &cap in &caps {
                    check_pq2_prepared_cap(
                        cap,
                        row_refs,
                        active_chunks,
                        &expected_fused,
                        &expected_unfused,
                        block_count,
                    )?;
                }
            }
            // Verify slice-fill packing parity and NaN/Inf weight scale rejection at first/middle/last block.
            let mut appended = Vec::new();
            let mut into_buf = vec![super::Pq2I8mmPairBlock::zeroed(); block_count];
            assert!(
                detected
                    .append_i8mm_pair_blocks(chunk_slices[0][0], chunk_slices[0][1], &mut appended)
                    .is_some()
            );
            assert!(
                detected
                    .pack_i8mm_pair_blocks_into(
                        chunk_slices[0][0],
                        chunk_slices[0][1],
                        &mut into_buf
                    )
                    .is_some()
            );
            assert_eq!(appended.len(), into_buf.len());
            for (a, b) in appended.iter().zip(&into_buf) {
                assert_eq!(a.planes, b.planes);
                assert_eq!(a.sums, b.sums);
                assert_eq!(a.scales.map(f32::to_bits), b.scales.map(f32::to_bits));
            }
            if block_count >= 1 {
                let mut panel_storage = Vec::new();
                for chunk in &chunk_slices[..2] {
                    for p in 0..4 {
                        assert!(
                            detected
                                .append_i8mm_pair_blocks(
                                    chunk[2 * p],
                                    chunk[2 * p + 1],
                                    &mut panel_storage
                                )
                                .is_some()
                        );
                    }
                }
                let panel_chunks: [[&[super::Pq2I8mmPairBlock]; 4]; 2] = std::array::from_fn(|c| {
                    std::array::from_fn(|p| {
                        let start = (c * 4 + p) * block_count;
                        &panel_storage[start..start + block_count]
                    })
                });
                let prepared = detected
                    .prepare_scaled_chunks(&chunk_slices[..2])
                    .ok_or("valid PQ2 chunks must prepare")?;
                for bad_f16 in [0x7e00_u16, 0x7c00, 0xfc00] {
                    for bad_blk in [0_usize, block_count / 2, block_count - 1] {
                        let mut bad_rows = owned_rows.clone();
                        let row_slot = (bad_blk + 3) % 8;
                        bad_rows[row_slot][bad_blk * 34..bad_blk * 34 + 2]
                            .copy_from_slice(&bad_f16.to_le_bytes());
                        let bad_refs: [&[u8]; 8] = std::array::from_fn(|r| bad_rows[r].as_slice());
                        let mut out = [[[0.0_f32; 8]; 8]; 2];
                        assert!(
                            prepared
                                .dot_scaled_8x8_chunks_with_panels::<true>(
                                    bad_refs,
                                    &panel_chunks,
                                    &mut out
                                )
                                .is_none()
                        );
                        assert!(
                            prepared
                                .dot_scaled_8x8_chunks_with_panels::<false>(
                                    bad_refs,
                                    &panel_chunks,
                                    &mut out
                                )
                                .is_none()
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn x1_scaled_8x8_chunks_rejects_bad_inputs() {
        #[cfg(target_arch = "aarch64")]
        let Some(detected) = crate::Pq2Dotprod::detect() else {
            return;
        };
        #[cfg(not(target_arch = "aarch64"))]
        let detected = crate::Pq2Dotprod { _detected: () };

        #[cfg(target_arch = "aarch64")]
        let caps = [
            detected,
            crate::Pq2Dotprod {
                i8mm: false,
                ..detected
            },
        ];
        #[cfg(not(target_arch = "aarch64"))]
        let caps = [detected];

        let (owned_rows, act_storage) = sample_scaled_8x8_fixture(2, 99);
        let row_refs: [&[u8]; 8] = std::array::from_fn(|r| owned_rows[r].as_slice());
        let chunk_slices: [[&[Pq2Activation]; 8]; 4] =
            std::array::from_fn(|c| std::array::from_fn(|t| act_storage[c * 8 + t].as_slice()));

        assert!(
            detected
                .dot_scaled_8x8_chunks::<true>(row_refs, &[], &mut [])
                .is_none()
        );
        let mut wrong_len = vec![[[0.0_f32; 8]; 8]; 2];
        assert!(
            detected
                .dot_scaled_8x8_chunks::<true>(row_refs, &chunk_slices[..1], &mut wrong_len)
                .is_none()
        );
        let mut out_one = [[[0.0_f32; 8]; 8]; 1];
        for bad_f16 in [0x7e00_u16, 0x7c00, 0xfc00] {
            let mut bad_rows = owned_rows.clone();
            bad_rows[5][34..36].copy_from_slice(&bad_f16.to_le_bytes());
            let bad_row_refs: [&[u8]; 8] = std::array::from_fn(|r| bad_rows[r].as_slice());
            for &cap in &caps {
                assert!(
                    cap.dot_scaled_8x8_chunks::<true>(
                        bad_row_refs,
                        &chunk_slices[..1],
                        &mut out_one
                    )
                    .is_none()
                );
                assert!(
                    cap.dot_scaled_8x8_chunks::<false>(
                        bad_row_refs,
                        &chunk_slices[..1],
                        &mut out_one
                    )
                    .is_none()
                );
            }
        }
        let bad_act = [
            Pq2Activation {
                scale: f32::NAN,
                values: act_storage[0][0].values,
                sum: act_storage[0][0].sum,
            },
            Pq2Activation {
                scale: act_storage[0][1].scale,
                values: act_storage[0][1].values,
                sum: act_storage[0][1].sum,
            },
        ];
        let mut bad_chunk = chunk_slices[0];
        bad_chunk[3] = &bad_act;
        assert!(
            detected
                .dot_scaled_8x8_chunks::<true>(row_refs, &[bad_chunk], &mut out_one)
                .is_none()
        );
    }

    fn sample_scaled_8x_tokens_fixture() -> ([Vec<u8>; 8], Vec<Vec<Pq2Activation>>) {
        let block_count = 3_usize;
        let f16_pool: [u16; 8] = [
            0x0000, 0x8000, 0x0001, 0x8001, 0x3c01, 0xbc01, 0x3555, 0xb787,
        ];
        let act_scale_pool: [f32; 8] = [
            0.0_f32,
            -0.0_f32,
            f32::from_bits(1),
            -f32::from_bits(1),
            f32::from_bits(0x3da0_0001),
            -0.125_f32,
            0.375_f32,
            -0.0625_f32,
        ];
        let rows: [Vec<u8>; 8] = std::array::from_fn(|row_idx| {
            let mut row = vec![0_u8; block_count * 34];
            for blk in 0..block_count {
                let s_bits = f16_pool[(row_idx + blk * 3) % f16_pool.len()].to_le_bytes();
                let start = blk * 34;
                row[start] = s_bits[0];
                row[start + 1] = s_bits[1];
                for i in 0..32 {
                    row[start + 2 + i] =
                        u8::try_from((row_idx * 41 + blk * 59 + i * 19 + 5) % 256).unwrap_or(0);
                }
            }
            row
        });
        let act_storage: Vec<Vec<Pq2Activation>> = (0..9_usize)
            .map(|tok| {
                (0..block_count)
                    .map(|blk| {
                        let values: [i8; 128] = match (tok + blk) % 4 {
                            0 => [i8::MIN; 128],
                            1 => [i8::MAX; 128],
                            2 => std::array::from_fn(|j| {
                                if (j + tok).is_multiple_of(2) {
                                    i8::MIN
                                } else {
                                    i8::MAX
                                }
                            }),
                            _ => std::array::from_fn(|j| {
                                i8::from_ne_bytes([j.to_le_bytes()[0]
                                    .wrapping_mul(71)
                                    .wrapping_add(tok.to_le_bytes()[0].wrapping_mul(17))
                                    .wrapping_add(blk.to_le_bytes()[0])])
                            }),
                        };
                        let sum = values.iter().map(|&v| i32::from(v)).sum();
                        let scale = act_scale_pool[(tok + blk * 2) % act_scale_pool.len()];
                        Pq2Activation { scale, values, sum }
                    })
                    .collect()
            })
            .collect();
        (rows, act_storage)
    }

    #[test]
    fn x1_scaled_8x_tokens_matches_portable_extremes_signed_zeros_and_errors() {
        use super::portable_scaled_8x_tokens;

        #[cfg(target_arch = "aarch64")]
        let Some(detected) = crate::Pq2Dotprod::detect() else {
            return;
        };
        #[cfg(not(target_arch = "aarch64"))]
        let detected = crate::Pq2Dotprod { _detected: () };

        #[cfg(target_arch = "aarch64")]
        let sdot_only = crate::Pq2Dotprod {
            i8mm: false,
            ..detected
        };
        #[cfg(not(target_arch = "aarch64"))]
        let sdot_only = detected;

        let (rows, act_storage) = sample_scaled_8x_tokens_fixture();
        let row_refs: [&[u8]; 8] = std::array::from_fn(|r| rows[r].as_slice());
        let act_refs: Vec<&[Pq2Activation]> = act_storage.iter().map(Vec::as_slice).collect();

        for cap in [detected, sdot_only] {
            for tok_count in 1..=8_usize {
                let slice = &act_refs[..tok_count];

                let mut exp_fused = [[0.0_f32; 8]; 8];
                let mut act_fused = [[f32::NAN; 8]; 8];
                assert!(
                    portable_scaled_8x_tokens::<true>(row_refs, slice, &mut exp_fused).is_some()
                );
                assert!(
                    cap.dot_scaled_8x_tokens::<true>(row_refs, slice, &mut act_fused)
                        .is_some()
                );
                for r in 0..8 {
                    assert_eq!(
                        exp_fused[r].map(f32::to_bits),
                        act_fused[r].map(f32::to_bits)
                    );
                    assert!(act_fused[r][tok_count..].iter().all(|v| v.to_bits() == 0));
                }

                let mut exp_sep = [[0.0_f32; 8]; 8];
                let mut act_sep = [[f32::NEG_INFINITY; 8]; 8];
                assert!(
                    portable_scaled_8x_tokens::<false>(row_refs, slice, &mut exp_sep).is_some()
                );
                assert!(
                    cap.dot_scaled_8x_tokens::<false>(row_refs, slice, &mut act_sep)
                        .is_some()
                );
                for r in 0..8 {
                    assert_eq!(exp_sep[r].map(f32::to_bits), act_sep[r].map(f32::to_bits));
                    assert!(act_sep[r][tok_count..].iter().all(|v| v.to_bits() == 0));
                }
            }
        }
    }

    #[test]
    fn x1_scaled_8x_tokens_rejects_bad_inputs_and_overflow() {
        #[cfg(target_arch = "aarch64")]
        let Some(detected) = crate::Pq2Dotprod::detect() else {
            return;
        };
        #[cfg(not(target_arch = "aarch64"))]
        let detected = crate::Pq2Dotprod { _detected: () };

        #[cfg(target_arch = "aarch64")]
        let sdot_only = crate::Pq2Dotprod {
            i8mm: false,
            ..detected
        };
        #[cfg(not(target_arch = "aarch64"))]
        let sdot_only = detected;

        let (rows, act_storage) = sample_scaled_8x_tokens_fixture();
        let row_refs: [&[u8]; 8] = std::array::from_fn(|r| rows[r].as_slice());
        let act_refs: Vec<&[Pq2Activation]> = act_storage.iter().map(Vec::as_slice).collect();
        let block_count = act_storage[0].len();

        let mut out = [[0.0_f32; 8]; 8];
        assert!(
            detected
                .dot_scaled_8x_tokens::<true>(row_refs, &[], &mut out)
                .is_none()
        );
        assert!(
            detected
                .dot_scaled_8x_tokens::<true>(row_refs, &act_refs[..9], &mut out)
                .is_none()
        );

        let mut short_rows = row_refs;
        short_rows[7] = &rows[7][..rows[7].len() - 1];
        assert!(
            detected
                .dot_scaled_8x_tokens::<true>(short_rows, &act_refs[..3], &mut out)
                .is_none()
        );

        let mut short_acts = act_refs[..3].to_vec();
        short_acts[2] = &act_storage[2][..block_count - 1];
        assert!(
            detected
                .dot_scaled_8x_tokens::<true>(row_refs, &short_acts, &mut out)
                .is_none()
        );

        for bad_f16 in [0x7c00_u16, 0x7e00_u16, 0xfc00_u16] {
            let mut bad_rows = rows.clone();
            bad_rows[6][34..36].copy_from_slice(&bad_f16.to_le_bytes());
            let bad_refs: [&[u8]; 8] = std::array::from_fn(|r| bad_rows[r].as_slice());
            for cap in [detected, sdot_only] {
                assert!(
                    cap.dot_scaled_8x_tokens::<true>(bad_refs, &act_refs[..3], &mut out)
                        .is_none()
                );
                assert!(
                    cap.dot_scaled_8x_tokens::<false>(bad_refs, &act_refs[..3], &mut out)
                        .is_none()
                );
            }
        }

        let bad_tok: Vec<Pq2Activation> = act_storage[0]
            .iter()
            .enumerate()
            .map(|(idx, b)| Pq2Activation {
                scale: if idx == 1 { f32::NAN } else { b.scale },
                values: b.values,
                sum: b.sum,
            })
            .collect();
        assert!(
            detected
                .dot_scaled_8x_tokens::<true>(
                    row_refs,
                    &[act_refs[0], bad_tok.as_slice()],
                    &mut out
                )
                .is_none()
        );

        let mut ovf_row = vec![0xff_u8; 34];
        ovf_row[..2].copy_from_slice(&0x7bff_u16.to_le_bytes());
        let ovf_refs: [&[u8]; 8] = [ovf_row.as_slice(); 8];
        let ovf_act = [Pq2Activation {
            scale: f32::MAX,
            values: [127; 128],
            sum: 128 * 127,
        }];
        for cap in [detected, sdot_only] {
            assert!(
                cap.dot_scaled_8x_tokens::<true>(ovf_refs, &[&ovf_act], &mut out)
                    .is_none()
            );
            assert!(
                cap.dot_scaled_8x_tokens::<false>(ovf_refs, &[&ovf_act], &mut out)
                    .is_none()
            );
        }
    }

    fn sample_ptq1_fixture_blocks(
        block_count: usize,
        seed: usize,
    ) -> ([Vec<u8>; 16], Vec<Vec<super::Q8ActivationBlock>>) {
        let f16_pool: [u16; 8] = [
            0x0000, 0x8000, 0x0001, 0x8001, 0x3c01, 0xbc01, 0x3555, 0xb787,
        ];
        let act_scale_pool: [f32; 8] = [
            0.0_f32,
            -0.0_f32,
            f32::from_bits(1),
            -f32::from_bits(1),
            f32::from_bits(0x3da0_0001),
            -0.125_f32,
            0.375_f32,
            -0.0625_f32,
        ];
        let rows: [Vec<u8>; 16] = std::array::from_fn(|row_idx| {
            let mut row = vec![0_u8; block_count * 28];
            for blk in 0..block_count {
                let start = blk * 28;
                for i in 0..26 {
                    row[start + i] =
                        u8::try_from((row_idx * 41 + blk * 59 + i * 19 + seed * 13 + 5) % 256)
                            .unwrap_or(0);
                }
                let s_bits = f16_pool[(row_idx + blk * 3 + seed) % f16_pool.len()].to_le_bytes();
                row[start + 26] = s_bits[0];
                row[start + 27] = s_bits[1];
            }
            row
        });
        let acts: Vec<Vec<super::Q8ActivationBlock>> = (0..9_usize)
            .map(|tok| {
                (0..block_count * 4)
                    .map(|grp| {
                        let values: [i8; 32] = match (tok + grp + seed) % 4 {
                            0 => [i8::MIN; 32],
                            1 => [i8::MAX; 32],
                            2 => std::array::from_fn(|j| {
                                if (j + tok).is_multiple_of(2) {
                                    i8::MIN
                                } else {
                                    i8::MAX
                                }
                            }),
                            _ => std::array::from_fn(|j| {
                                i8::from_ne_bytes([j.to_le_bytes()[0]
                                    .wrapping_mul(71)
                                    .wrapping_add(tok.to_le_bytes()[0].wrapping_mul(17))
                                    .wrapping_add(grp.to_le_bytes()[0])
                                    .wrapping_add(seed.to_le_bytes()[0])])
                            }),
                        };
                        super::Q8ActivationBlock {
                            scale: act_scale_pool[(tok + grp + seed) % act_scale_pool.len()],
                            values,
                        }
                    })
                    .collect()
            })
            .collect();
        (rows, acts)
    }

    fn sample_ptq1_fixture() -> ([Vec<u8>; 16], Vec<Vec<super::Q8ActivationBlock>>) {
        sample_ptq1_fixture_blocks(3, 0)
    }

    #[test]
    fn x1_ptq1_scaled_rows_match_portable_across_pinned_and_source() {
        use super::portable_ptq1_rows_validated;

        #[cfg(target_arch = "aarch64")]
        let Some(detected) = crate::Ptq1Dotprod::detect() else {
            return;
        };
        #[cfg(not(target_arch = "aarch64"))]
        let detected = crate::Ptq1Dotprod { _detected: () };

        for (seed, block_count) in [1_usize, 2, 3, 4, 5, 7, 8, 9, 16, 17, 24, 32]
            .into_iter()
            .enumerate()
        {
            let (rows, acts) = sample_ptq1_fixture_blocks(block_count, seed);
            let act = &acts[0];
            let r1 = [rows[0].as_slice()];
            let r4: [&[u8]; 4] = std::array::from_fn(|r| rows[r].as_slice());
            let r8: [&[u8]; 8] = std::array::from_fn(|r| rows[r].as_slice());
            let r16: [&[u8]; 16] = std::array::from_fn(|r| rows[r].as_slice());

            assert_eq!(
                detected
                    .dot_scaled_rows::<1, true>(r1, act)
                    .map(|v| v.map(f32::to_bits)),
                portable_ptq1_rows_validated::<1, true, _>(r1, act).map(|v| v.map(f32::to_bits))
            );
            assert_eq!(
                detected
                    .dot_scaled_rows::<1, false>(r1, act)
                    .map(|v| v.map(f32::to_bits)),
                portable_ptq1_rows_validated::<1, false, _>(r1, act).map(|v| v.map(f32::to_bits))
            );
            assert_eq!(
                detected
                    .dot_scaled_rows::<4, true>(r4, act)
                    .map(|v| v.map(f32::to_bits)),
                portable_ptq1_rows_validated::<4, true, _>(r4, act).map(|v| v.map(f32::to_bits))
            );
            assert_eq!(
                detected
                    .dot_scaled_rows::<4, false>(r4, act)
                    .map(|v| v.map(f32::to_bits)),
                portable_ptq1_rows_validated::<4, false, _>(r4, act).map(|v| v.map(f32::to_bits))
            );
            assert_eq!(
                detected
                    .dot_scaled_rows::<8, true>(r8, act)
                    .map(|v| v.map(f32::to_bits)),
                portable_ptq1_rows_validated::<8, true, _>(r8, act).map(|v| v.map(f32::to_bits))
            );
            assert_eq!(
                detected
                    .dot_scaled_rows::<8, false>(r8, act)
                    .map(|v| v.map(f32::to_bits)),
                portable_ptq1_rows_validated::<8, false, _>(r8, act).map(|v| v.map(f32::to_bits))
            );
            assert_eq!(
                detected
                    .dot_scaled_rows::<16, true>(r16, act)
                    .map(|v| v.map(f32::to_bits)),
                portable_ptq1_rows_validated::<16, true, _>(r16, act).map(|v| v.map(f32::to_bits))
            );
            assert_eq!(
                detected
                    .dot_scaled_rows::<16, false>(r16, act)
                    .map(|v| v.map(f32::to_bits)),
                portable_ptq1_rows_validated::<16, false, _>(r16, act).map(|v| v.map(f32::to_bits))
            );
        }

        let (rows, acts) = sample_ptq1_fixture_blocks(2, 55);
        for bad_f16 in [0x7e00_u16, 0x7c00] {
            let mut bad_rows = rows.clone();
            bad_rows[13][28 + 26..28 + 28].copy_from_slice(&bad_f16.to_le_bytes());
            let bad16: [&[u8]; 16] = std::array::from_fn(|r| bad_rows[r].as_slice());
            assert!(
                detected
                    .dot_scaled_rows::<16, true>(bad16, &acts[0])
                    .is_none()
            );
            assert!(
                detected
                    .dot_scaled_rows::<16, false>(bad16, &acts[0])
                    .is_none()
            );
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Covers pinned/source modes across row counts, extremes, and error recovery."
    )]
    #[test]
    fn x1_ptq1_packed_prepared_matches_portable_across_pinned_and_source() {
        use super::portable_ptq1_packed_rows_validated;

        #[cfg(target_arch = "aarch64")]
        let Some(detected) = crate::Ptq1Dotprod::detect() else {
            return;
        };
        #[cfg(not(target_arch = "aarch64"))]
        let detected = crate::Ptq1Dotprod { _detected: () };

        let to_packed = |acts: &[super::Q8ActivationBlock]| -> Vec<super::Ptq1Activation> {
            acts.as_chunks::<4>()
                .0
                .iter()
                .map(|g| {
                    super::Ptq1Activation::from_groups(
                        [g[0].scale, g[1].scale, g[2].scale, g[3].scale],
                        [&g[0].values, &g[1].values, &g[2].values, &g[3].values],
                    )
                })
                .collect()
        };

        for (seed, block_count) in [1_usize, 2, 3, 4, 5, 7, 8, 9, 16, 17, 24, 32]
            .into_iter()
            .enumerate()
        {
            let (rows, acts) = sample_ptq1_fixture_blocks(block_count, seed);
            let packed = to_packed(&acts[0]);
            let Some(prepared) = detected.prepare_packed(&packed) else {
                assert!(detected.prepare_packed(&packed).is_some());
                return;
            };
            let r1 = [rows[0].as_slice()];
            let r4: [&[u8]; 4] = std::array::from_fn(|r| rows[r].as_slice());
            let r8: [&[u8]; 8] = std::array::from_fn(|r| rows[r].as_slice());
            let r16: [&[u8]; 16] = std::array::from_fn(|r| rows[r].as_slice());

            assert_eq!(
                prepared
                    .dot_scaled_rows::<1, true>(r1)
                    .map(|v| v.map(f32::to_bits)),
                portable_ptq1_packed_rows_validated::<1, true>(r1, &packed)
                    .map(|v| v.map(f32::to_bits))
            );
            assert_eq!(
                prepared
                    .dot_scaled_rows::<1, false>(r1)
                    .map(|v| v.map(f32::to_bits)),
                portable_ptq1_packed_rows_validated::<1, false>(r1, &packed)
                    .map(|v| v.map(f32::to_bits))
            );
            assert_eq!(
                prepared
                    .dot_scaled_rows::<4, true>(r4)
                    .map(|v| v.map(f32::to_bits)),
                portable_ptq1_packed_rows_validated::<4, true>(r4, &packed)
                    .map(|v| v.map(f32::to_bits))
            );
            assert_eq!(
                prepared
                    .dot_scaled_rows::<4, false>(r4)
                    .map(|v| v.map(f32::to_bits)),
                portable_ptq1_packed_rows_validated::<4, false>(r4, &packed)
                    .map(|v| v.map(f32::to_bits))
            );
            assert_eq!(
                prepared
                    .dot_scaled_rows::<8, true>(r8)
                    .map(|v| v.map(f32::to_bits)),
                portable_ptq1_packed_rows_validated::<8, true>(r8, &packed)
                    .map(|v| v.map(f32::to_bits))
            );
            assert_eq!(
                prepared
                    .dot_scaled_rows::<8, false>(r8)
                    .map(|v| v.map(f32::to_bits)),
                portable_ptq1_packed_rows_validated::<8, false>(r8, &packed)
                    .map(|v| v.map(f32::to_bits))
            );
            assert_eq!(
                prepared
                    .dot_scaled_rows::<16, true>(r16)
                    .map(|v| v.map(f32::to_bits)),
                portable_ptq1_packed_rows_validated::<16, true>(r16, &packed)
                    .map(|v| v.map(f32::to_bits))
            );
            assert_eq!(
                prepared
                    .dot_scaled_rows::<16, false>(r16)
                    .map(|v| v.map(f32::to_bits)),
                portable_ptq1_packed_rows_validated::<16, false>(r16, &packed)
                    .map(|v| v.map(f32::to_bits))
            );
        }

        assert!(detected.prepare_packed(&[]).is_none());
        let (rows, acts) = sample_ptq1_fixture_blocks(2, 55);
        let mut packed = to_packed(&acts[0]);
        for bad_f16 in [0x7e00_u16, 0x7c00, 0xfc00] {
            for bad_row in [0_usize, 3, 7, 13] {
                let mut bad_rows = rows.clone();
                bad_rows[bad_row][28 + 26..28 + 28].copy_from_slice(&bad_f16.to_le_bytes());
                let bad16: [&[u8]; 16] = std::array::from_fn(|r| bad_rows[r].as_slice());
                let Some(prepared) = detected.prepare_packed(&packed) else {
                    assert!(detected.prepare_packed(&packed).is_some());
                    return;
                };
                assert!(prepared.dot_scaled_rows::<16, true>(bad16).is_none());
                assert!(prepared.dot_scaled_rows::<16, false>(bad16).is_none());
                if bad_row < 8 {
                    let bad8: [&[u8]; 8] = std::array::from_fn(|r| bad_rows[r].as_slice());
                    assert!(prepared.dot_scaled_rows::<8, true>(bad8).is_none());
                    assert!(prepared.dot_scaled_rows::<8, false>(bad8).is_none());
                }
                if bad_row < 4 {
                    let bad4: [&[u8]; 4] = std::array::from_fn(|r| bad_rows[r].as_slice());
                    assert!(prepared.dot_scaled_rows::<4, true>(bad4).is_none());
                    assert!(prepared.dot_scaled_rows::<4, false>(bad4).is_none());
                }
                if bad_row == 0 {
                    assert!(prepared.dot_scaled_rows::<1, true>([bad16[0]]).is_none());
                    assert!(prepared.dot_scaled_rows::<1, false>([bad16[0]]).is_none());
                }
            }
        }
        packed[1].scales[2] = f32::NAN;
        assert!(detected.prepare_packed(&packed).is_none());

        let mut ovf_row = vec![0xff_u8; 28];
        ovf_row[26..28].copy_from_slice(&0x7bff_u16.to_le_bytes());
        let ovf_act = [super::Ptq1Activation::new([f32::MAX; 4], [127; 128])];
        let Some(ovf_prep) = detected.prepare_packed(&ovf_act) else {
            assert!(detected.prepare_packed(&ovf_act).is_some());
            return;
        };
        assert!(
            ovf_prep
                .dot_scaled_rows::<1, true>([ovf_row.as_slice()])
                .is_none()
        );
        assert!(
            ovf_prep
                .dot_scaled_rows::<4, true>([ovf_row.as_slice(); 4])
                .is_none()
        );
        assert!(
            ovf_prep
                .dot_scaled_rows::<8, true>([ovf_row.as_slice(); 8])
                .is_none()
        );
        assert!(
            ovf_prep
                .dot_scaled_rows::<16, true>([ovf_row.as_slice(); 16])
                .is_none()
        );
        assert!(
            ovf_prep
                .dot_scaled_rows::<1, false>([ovf_row.as_slice()])
                .is_none()
        );
        assert!(
            ovf_prep
                .dot_scaled_rows::<4, false>([ovf_row.as_slice(); 4])
                .is_none()
        );
        assert!(
            ovf_prep
                .dot_scaled_rows::<8, false>([ovf_row.as_slice(); 8])
                .is_none()
        );
        assert!(
            ovf_prep
                .dot_scaled_rows::<16, false>([ovf_row.as_slice(); 16])
                .is_none()
        );
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Verifies 4-row and 8-row non-SVE PTQ1 decode parity vs 1-row, all 243 trit bytes, signed zeros, and NaN/Inf in every row slot."
    )]
    #[test]
    fn x1_ptq1_packed_4_and_8rows_non_sve_parity_and_extremes() {
        use super::portable_ptq1_packed_rows_validated;

        #[cfg(target_arch = "aarch64")]
        let Some(detected) = crate::Ptq1Dotprod::detect() else {
            return;
        };
        #[cfg(target_arch = "aarch64")]
        let non_sve = crate::Ptq1Dotprod {
            sve: false,
            ..detected
        };
        #[cfg(not(target_arch = "aarch64"))]
        let non_sve = crate::Ptq1Dotprod { _detected: () };

        let to_packed_no_sve = |acts: &[super::Q8ActivationBlock]| -> Vec<super::Ptq1Activation> {
            acts.as_chunks::<4>()
                .0
                .iter()
                .map(|g| {
                    super::Ptq1Activation::from_groups_without_lane_sums(
                        [g[0].scale, g[1].scale, g[2].scale, g[3].scale],
                        [&g[0].values, &g[1].values, &g[2].values, &g[3].values],
                    )
                })
                .collect()
        };

        // Sweep all 243 valid 5-trit bytes across positions and all block counts.
        let valid_bytes: Vec<u8> = (0..=242_u8)
            .map(|idx| {
                // Map 5-trit index 0..243 into canonical PTQ1 byte (round-up base-3 encoding).
                let v = u32::from(idx);
                u8::try_from((v * 256 + 242) / 243).unwrap_or(0)
            })
            .collect();

        for (seed, block_count) in [1_usize, 2, 3, 4, 5, 40, 136].into_iter().enumerate() {
            let (mut rows, mut acts) = sample_ptq1_fixture_blocks(block_count, seed + 71);
            // Inject all 243 valid 5-trit bytes and signed zeros across the 8 rows.
            for (blk_idx, row) in rows[..8].iter_mut().enumerate() {
                for b in 0..block_count {
                    let off = b * 28;
                    for pos in 0..26 {
                        row[off + pos] =
                            valid_bytes[(blk_idx * 37 + b * 29 + pos * 11 + seed) % 243];
                    }
                    if (blk_idx + b) % 9 == 0 {
                        let sign_zero: u16 = if (blk_idx + b).is_multiple_of(2) {
                            0x0000
                        } else {
                            0x8000
                        };
                        row[off + 26..off + 28].copy_from_slice(&sign_zero.to_le_bytes());
                    }
                }
            }
            for (g_idx, blk) in acts[0].iter_mut().enumerate() {
                if g_idx % 11 == 0 {
                    blk.scale = if g_idx.is_multiple_of(2) { 0.0 } else { -0.0 };
                }
            }
            let packed = to_packed_no_sve(&acts[0]);
            let Some(prepared) = non_sve.prepare_packed(&packed) else {
                assert!(non_sve.prepare_packed(&packed).is_some());
                return;
            };
            let r4: [&[u8]; 4] = std::array::from_fn(|r| rows[r].as_slice());
            let r8: [&[u8]; 8] = std::array::from_fn(|r| rows[r].as_slice());

            let got4 = prepared.dot_scaled_rows::<4, true>(r4);
            let got8 = prepared.dot_scaled_rows::<8, true>(r8);
            let exp4 = portable_ptq1_packed_rows_validated::<4, true>(r4, &packed);
            let exp8 = portable_ptq1_packed_rows_validated::<8, true>(r8, &packed);
            assert_eq!(
                got4.map(|v| v.map(f32::to_bits)),
                exp4.map(|v| v.map(f32::to_bits))
            );
            assert_eq!(
                got8.map(|v| v.map(f32::to_bits)),
                exp8.map(|v| v.map(f32::to_bits))
            );

            // Compare 0-ULP bit-for-bit against 1-row pinned kernel (`ptq1_packed_1row_pinned_accum`).
            for (r, row) in r8.iter().enumerate() {
                let single = prepared.dot_scaled_rows::<1, true>([*row]);
                assert_eq!(
                    single.map(|v| v[0].to_bits()),
                    got8.map(|v| v[r].to_bits()),
                    "block_count={block_count} row={r}"
                );
            }

            // Inject NaN/Inf scales in every row slot 0..7 at first, middle, and last block.
            for bad_f16 in [0x7e00_u16, 0x7c00, 0xfc00] {
                for bad_blk in [0_usize, block_count / 2, block_count - 1] {
                    for bad_row in 0..8_usize {
                        let mut bad_rows = rows[..8].to_vec();
                        let off = bad_blk * 28 + 26;
                        bad_rows[bad_row][off..off + 2].copy_from_slice(&bad_f16.to_le_bytes());
                        let bad8: [&[u8]; 8] = std::array::from_fn(|r| bad_rows[r].as_slice());
                        assert!(
                            prepared.dot_scaled_rows::<8, true>(bad8).is_none(),
                            "expected None for bad_f16={bad_f16:#x} blk={bad_blk} row={bad_row}"
                        );
                        if bad_row < 4 {
                            let bad4: [&[u8]; 4] = std::array::from_fn(|r| bad_rows[r].as_slice());
                            assert!(
                                prepared.dot_scaled_rows::<4, true>(bad4).is_none(),
                                "expected None for 4-row bad_f16={bad_f16:#x} blk={bad_blk} row={bad_row}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn x1_ptq1_scaled_8x_tokens_matches_portable_and_rejects_bad_inputs() {
        use super::portable_ptq1_8x_tokens_validated;

        #[cfg(target_arch = "aarch64")]
        let Some(detected) = crate::Ptq1Dotprod::detect() else {
            return;
        };
        #[cfg(not(target_arch = "aarch64"))]
        let detected = crate::Ptq1Dotprod { _detected: () };

        #[cfg(target_arch = "aarch64")]
        let sdot_only = crate::Ptq1Dotprod {
            i8mm: false,
            ..detected
        };
        #[cfg(not(target_arch = "aarch64"))]
        let sdot_only = detected;

        let (rows, acts) = sample_ptq1_fixture();
        let row_refs: [&[u8]; 8] = std::array::from_fn(|r| rows[r].as_slice());
        let act_refs: Vec<&[super::Q8ActivationBlock]> = acts.iter().map(Vec::as_slice).collect();

        for cap in [detected, sdot_only] {
            for tok_count in 1..=8_usize {
                let slice = &act_refs[..tok_count];
                for pinned in [false, true] {
                    let mut exp = [[0.0_f32; 8]; 8];
                    let mut got = [[f32::NAN; 8]; 8];
                    if pinned {
                        assert!(
                            portable_ptq1_8x_tokens_validated::<true, _>(row_refs, slice, &mut exp)
                                .is_some()
                        );
                        assert!(
                            cap.dot_scaled_8x_tokens::<true>(row_refs, slice, &mut got)
                                .is_some()
                        );
                    } else {
                        assert!(
                            portable_ptq1_8x_tokens_validated::<false, _>(
                                row_refs, slice, &mut exp
                            )
                            .is_some()
                        );
                        assert!(
                            cap.dot_scaled_8x_tokens::<false>(row_refs, slice, &mut got)
                                .is_some()
                        );
                    }
                    for r in 0..8 {
                        assert_eq!(exp[r].map(f32::to_bits), got[r].map(f32::to_bits));
                        assert!(got[r][tok_count..].iter().all(|v| v.to_bits() == 0));
                    }
                }
            }
        }

        let mut out = [[0.0_f32; 8]; 8];
        assert!(
            detected
                .dot_scaled_8x_tokens::<true>(row_refs, &[], &mut out)
                .is_none()
        );
        assert!(
            detected
                .dot_scaled_8x_tokens::<true>(row_refs, &act_refs[..9], &mut out)
                .is_none()
        );
        let mut bad_rows = rows[..8].to_vec();
        bad_rows[5][28 + 26..28 + 28].copy_from_slice(&0x7e00_u16.to_le_bytes());
        let bad_refs: [&[u8]; 8] = std::array::from_fn(|r| bad_rows[r].as_slice());
        for cap in [detected, sdot_only] {
            assert!(
                cap.dot_scaled_8x_tokens::<true>(bad_refs, &act_refs[..3], &mut out)
                    .is_none()
            );
            assert!(
                cap.dot_scaled_8x_tokens::<false>(bad_refs, &act_refs[..3], &mut out)
                    .is_none()
            );
        }
        assert!(
            detected
                .dot_scaled_rows::<16, true>(
                    std::array::from_fn(|r| if r == 5 {
                        bad_rows[5].as_slice()
                    } else {
                        rows[r].as_slice()
                    }),
                    act_refs[0]
                )
                .is_none()
        );
    }

    #[test]
    fn x1_ptq1_prepared_matches_portable_and_rejects_bad_inputs() {
        use super::portable_ptq1_rows_validated;

        #[cfg(target_arch = "aarch64")]
        let Some(detected) = crate::Ptq1Dotprod::detect() else {
            return;
        };
        #[cfg(not(target_arch = "aarch64"))]
        let detected = crate::Ptq1Dotprod { _detected: () };

        assert!(detected.prepare_scaled(&[]).is_none());
        let (rows, mut acts) = sample_ptq1_fixture_blocks(5, 19);
        assert!(detected.prepare_scaled(&acts[0][..7]).is_none());
        acts[0][2].scale = f32::NAN;
        assert!(detected.prepare_scaled(&acts[0]).is_none());
        acts[0][2].scale = 0.25;

        let prepared = detected.prepare_scaled(&acts[0]);
        assert!(prepared.is_some());
        let Some(prepared) = prepared else {
            return;
        };
        let r1 = [rows[0].as_slice()];
        let r4: [&[u8]; 4] = std::array::from_fn(|r| rows[r].as_slice());
        let r8: [&[u8]; 8] = std::array::from_fn(|r| rows[r].as_slice());
        let r16: [&[u8]; 16] = std::array::from_fn(|r| rows[r].as_slice());

        assert_eq!(
            prepared
                .dot_scaled_rows::<1, true>(r1)
                .map(|v| v.map(f32::to_bits)),
            portable_ptq1_rows_validated::<1, true, _>(r1, &acts[0]).map(|v| v.map(f32::to_bits))
        );
        assert_eq!(
            prepared
                .dot_scaled_rows::<1, false>(r1)
                .map(|v| v.map(f32::to_bits)),
            portable_ptq1_rows_validated::<1, false, _>(r1, &acts[0]).map(|v| v.map(f32::to_bits))
        );
        assert_eq!(
            prepared
                .dot_scaled_rows::<4, true>(r4)
                .map(|v| v.map(f32::to_bits)),
            portable_ptq1_rows_validated::<4, true, _>(r4, &acts[0]).map(|v| v.map(f32::to_bits))
        );
        assert_eq!(
            prepared
                .dot_scaled_rows::<4, false>(r4)
                .map(|v| v.map(f32::to_bits)),
            portable_ptq1_rows_validated::<4, false, _>(r4, &acts[0]).map(|v| v.map(f32::to_bits))
        );
        assert_eq!(
            prepared
                .dot_scaled_rows::<8, true>(r8)
                .map(|v| v.map(f32::to_bits)),
            portable_ptq1_rows_validated::<8, true, _>(r8, &acts[0]).map(|v| v.map(f32::to_bits))
        );
        assert_eq!(
            prepared
                .dot_scaled_rows::<8, false>(r8)
                .map(|v| v.map(f32::to_bits)),
            portable_ptq1_rows_validated::<8, false, _>(r8, &acts[0]).map(|v| v.map(f32::to_bits))
        );
        assert_eq!(
            prepared
                .dot_scaled_rows::<16, true>(r16)
                .map(|v| v.map(f32::to_bits)),
            portable_ptq1_rows_validated::<16, true, _>(r16, &acts[0]).map(|v| v.map(f32::to_bits))
        );
        assert_eq!(
            prepared
                .dot_scaled_rows::<16, false>(r16)
                .map(|v| v.map(f32::to_bits)),
            portable_ptq1_rows_validated::<16, false, _>(r16, &acts[0])
                .map(|v| v.map(f32::to_bits))
        );

        assert!(
            prepared
                .dot_scaled_rows::<2, true>([rows[0].as_slice(), rows[1].as_slice()])
                .is_none()
        );
        assert!(
            prepared
                .dot_scaled_rows::<1, true>([&rows[0][..28]])
                .is_none()
        );
        for bad_f16 in [0x7c00_u16, 0x7e00] {
            let mut bad_rows = rows.clone();
            bad_rows[7][28 + 26..28 + 28].copy_from_slice(&bad_f16.to_le_bytes());
            let bad16: [&[u8]; 16] = std::array::from_fn(|r| bad_rows[r].as_slice());
            assert!(prepared.dot_scaled_rows::<16, true>(bad16).is_none());
            assert!(prepared.dot_scaled_rows::<16, false>(bad16).is_none());
        }
    }

    fn sample_q8_0_fixture(
        blocks_per_row: usize,
        tokens_count: usize,
        seed: usize,
    ) -> ([Vec<u8>; 8], Vec<Vec<super::Q8ActivationBlock>>) {
        let f16_scales = [
            0x0000_u16, 0x8000, 0x0001, 0x8001, 0x3555, 0xb787, 0x3c00, 0xbc00,
        ];
        let act_scales = [0.0_f32, -0.0, 0.125, -0.375, 1.5, -2.25, 0.0625, -0.5];
        let rows: [Vec<u8>; 8] = std::array::from_fn(|row| {
            let mut bytes: Vec<u8> = (0..blocks_per_row * 34)
                .map(|i| {
                    i.wrapping_mul(61 + row)
                        .wrapping_add(seed * 13 + row * 7)
                        .to_le_bytes()[0]
                })
                .collect();
            for (b, block) in bytes.as_chunks_mut::<34>().0.iter_mut().enumerate() {
                let s = f16_scales[(b * 3 + row + seed) % f16_scales.len()];
                block[..2].copy_from_slice(&s.to_le_bytes());
                if (b + row) % 5 == 0 {
                    block[2..34].fill(0x80);
                } else if (b + row) % 7 == 0 {
                    block[2..34].fill(0x7f);
                }
            }
            bytes
        });
        let acts: Vec<Vec<super::Q8ActivationBlock>> = (0..tokens_count)
            .map(|t| {
                (0..blocks_per_row)
                    .map(|b| super::Q8ActivationBlock {
                        scale: act_scales[(t * 5 + b + seed) % act_scales.len()],
                        values: std::array::from_fn(|lane| {
                            if (t + b) % 6 == 0 {
                                if (lane + t) % 2 == 0 { -128 } else { 127 }
                            } else {
                                i8::from_ne_bytes([lane
                                    .wrapping_mul(43 + t)
                                    .wrapping_add(b * 17 + seed)
                                    .to_le_bytes()[0]])
                            }
                        }),
                    })
                    .collect()
            })
            .collect();
        (rows, acts)
    }

    #[test]
    fn x1_q8_0_scaled_8x_tokens_matches_portable_across_modes_and_extremes() {
        let Some(detected) = crate::Q8Dotprod::detect() else {
            return;
        };
        #[cfg(target_arch = "aarch64")]
        let caps = [
            detected,
            crate::Q8Dotprod {
                i8mm: false,
                ..detected
            },
        ];
        #[cfg(not(target_arch = "aarch64"))]
        let caps = [detected];

        for (seed, block_count) in [1_usize, 3, 16, 40].into_iter().enumerate() {
            let (rows, acts) = sample_q8_0_fixture(block_count, 8, seed + 1);
            let row_refs: [&[u8]; 8] = std::array::from_fn(|r| rows[r].as_slice());
            let act_refs: [&[super::Q8ActivationBlock]; 8] =
                std::array::from_fn(|t| acts[t].as_slice());

            for tok_count in 1..=8_usize {
                let slice = &act_refs[..tok_count];
                let mut exp = [[0.0_f32; 8]; 8];
                assert!(portable_q8_0_8x_tokens_validated(row_refs, slice, &mut exp).is_some());
                for cap in caps {
                    let mut got = [[f32::NAN; 8]; 8];
                    assert!(
                        cap.dot_scaled_8x_tokens(row_refs, slice, &mut got)
                            .is_some()
                    );
                    for r in 0..8 {
                        assert_eq!(exp[r].map(f32::to_bits), got[r].map(f32::to_bits));
                        assert!(got[r][tok_count..].iter().all(|v| v.to_bits() == 0));
                    }
                }
                let mut via_top = [[f32::NAN; 8]; 8];
                assert!(crate::dot_q8_0_scaled_8x_tokens(row_refs, slice, &mut via_top).is_some());
                for r in 0..8 {
                    assert_eq!(exp[r].map(f32::to_bits), via_top[r].map(f32::to_bits));
                }
            }
        }
    }

    #[test]
    fn x1_q8_0_scaled_8x_tokens_rejects_bad_inputs_and_overflow() {
        let Some(detected) = crate::Q8Dotprod::detect() else {
            return;
        };
        let (rows, mut acts) = sample_q8_0_fixture(2, 9, 99);
        let row_refs: [&[u8]; 8] = std::array::from_fn(|r| rows[r].as_slice());
        let act_refs: Vec<&[super::Q8ActivationBlock]> = acts.iter().map(Vec::as_slice).collect();
        let sentinel = [[123.5_f32; 8]; 8];
        let mut out = sentinel;

        assert!(
            detected
                .dot_scaled_8x_tokens(row_refs, &[], &mut out)
                .is_none()
        );
        assert!(
            detected
                .dot_scaled_8x_tokens(row_refs, &act_refs[..9], &mut out)
                .is_none()
        );
        let mut short_rows = row_refs;
        short_rows[4] = &row_refs[4][..33];
        assert!(
            detected
                .dot_scaled_8x_tokens(short_rows, &act_refs[..2], &mut out)
                .is_none()
        );
        assert!(crate::dot_q8_0_scaled_8x_tokens(short_rows, &[], &mut out).is_none());

        for bad_f16 in [0x7c00_u16, 0x7e00, 0xfc00] {
            let mut bad_rows = rows.clone();
            bad_rows[6][34..36].copy_from_slice(&bad_f16.to_le_bytes());
            let bad_refs: [&[u8]; 8] = std::array::from_fn(|r| bad_rows[r].as_slice());
            assert!(
                detected
                    .dot_scaled_8x_tokens(bad_refs, &act_refs[..3], &mut out)
                    .is_none()
            );
        }
        acts[1][1].scale = f32::NAN;
        let bad_act_refs: Vec<&[super::Q8ActivationBlock]> =
            acts[..3].iter().map(Vec::as_slice).collect();
        assert!(
            detected
                .dot_scaled_8x_tokens(row_refs, &bad_act_refs, &mut out)
                .is_none()
        );

        let mut big_rows = rows;
        for row in &mut big_rows {
            row[..2].copy_from_slice(&0x7bff_u16.to_le_bytes());
            row[2..34].fill(0x7f);
            row[34..36].copy_from_slice(&0x7bff_u16.to_le_bytes());
            row[36..68].fill(0x7f);
        }
        let big_acts = vec![
            super::Q8ActivationBlock {
                scale: 1.0e32,
                values: [127; 32],
            };
            2
        ];
        let big_refs: [&[u8]; 8] = std::array::from_fn(|r| big_rows[r].as_slice());
        assert!(
            detected
                .dot_scaled_8x_tokens(big_refs, &[big_acts.as_slice()], &mut out)
                .is_none()
        );
        for r in 0..8 {
            assert_eq!(out[r].map(f32::to_bits), sentinel[r].map(f32::to_bits));
        }
    }

    fn sample_q8_0_rows16_fixture(
        blocks_per_row: usize,
        seed: usize,
    ) -> ([Vec<u8>; 16], Vec<super::Q8ActivationBlock>) {
        let f16_scales = [
            0x0000_u16, 0x8000, 0x0001, 0x8001, 0x3555, 0xb787, 0x3c00, 0xbc00,
        ];
        let act_scales = [0.0_f32, -0.0, 0.125, -0.375, 1.5, -2.25, 0.0625, -0.5];
        let rows: [Vec<u8>; 16] = std::array::from_fn(|row| {
            let mut bytes: Vec<u8> = (0..blocks_per_row * 34)
                .map(|i| {
                    i.wrapping_mul(61 + row)
                        .wrapping_add(seed * 13 + row * 7)
                        .to_le_bytes()[0]
                })
                .collect();
            for (b, block) in bytes.as_chunks_mut::<34>().0.iter_mut().enumerate() {
                let s = f16_scales[(b * 3 + row + seed) % f16_scales.len()];
                block[..2].copy_from_slice(&s.to_le_bytes());
                if (b + row) % 5 == 0 {
                    block[2..34].fill(0x80);
                } else if (b + row) % 7 == 0 {
                    block[2..34].fill(0x7f);
                }
            }
            bytes
        });
        let acts: Vec<super::Q8ActivationBlock> = (0..blocks_per_row)
            .map(|b| super::Q8ActivationBlock {
                scale: act_scales[(b * 5 + seed) % act_scales.len()],
                values: std::array::from_fn(|lane| {
                    if b % 6 == 0 {
                        if lane % 2 == 0 { -128 } else { 127 }
                    } else {
                        i8::from_ne_bytes([lane
                            .wrapping_mul(43 + b)
                            .wrapping_add(b * 17 + seed)
                            .to_le_bytes()[0]])
                    }
                }),
            })
            .collect();
        (rows, acts)
    }

    #[test]
    fn x1_q8_0_scaled_rows_1_4_8_16_match_portable_zero_ulp() {
        use super::portable_q8_0_rows_validated;

        let Some(detected) = crate::Q8Dotprod::detect() else {
            return;
        };
        #[cfg(target_arch = "aarch64")]
        let caps = [
            detected,
            crate::Q8Dotprod {
                i8mm: false,
                ..detected
            },
            crate::Q8Dotprod {
                dotprod: false,
                ..detected
            },
        ];
        #[cfg(not(target_arch = "aarch64"))]
        let caps = [detected];

        for (seed, block_count) in [1_usize, 2, 3, 4, 5, 7, 16, 64].into_iter().enumerate() {
            let (rows, acts) = sample_q8_0_rows16_fixture(block_count, seed + 11);
            let act_views: Vec<(&[i8; 32], f32)> =
                acts.iter().map(|b| (&b.values, b.scale)).collect();

            let r1 = [rows[0].as_slice()];
            let r4: [&[u8]; 4] = std::array::from_fn(|r| rows[r].as_slice());
            let r8: [&[u8]; 8] = std::array::from_fn(|r| rows[r].as_slice());
            let r16: [&[u8]; 16] = std::array::from_fn(|r| rows[r].as_slice());

            let exp1 = portable_q8_0_rows_validated(r1, &acts).map(|v| v.map(f32::to_bits));
            let exp4 = portable_q8_0_rows_validated(r4, &acts).map(|v| v.map(f32::to_bits));
            let exp8 = portable_q8_0_rows_validated(r8, &acts).map(|v| v.map(f32::to_bits));
            let exp16 = portable_q8_0_rows_validated(r16, &acts).map(|v| v.map(f32::to_bits));
            assert!(exp16.is_some());

            for cap in caps {
                assert_eq!(
                    cap.dot_scaled_rows(r1, &acts).map(|v| v.map(f32::to_bits)),
                    exp1
                );
                assert_eq!(
                    cap.dot_scaled_rows(r4, &acts).map(|v| v.map(f32::to_bits)),
                    exp4
                );
                assert_eq!(
                    cap.dot_scaled_rows(r8, &acts).map(|v| v.map(f32::to_bits)),
                    exp8
                );
                assert_eq!(
                    cap.dot_scaled_rows(r16, &acts).map(|v| v.map(f32::to_bits)),
                    exp16
                );
                assert_eq!(
                    cap.dot_scaled_rows_view(r16, &act_views)
                        .map(|v| v.map(f32::to_bits)),
                    exp16
                );
            }
            assert_eq!(
                crate::dot_q8_0_scaled_rows(r1, &acts).map(|v| v.map(f32::to_bits)),
                exp1
            );
            assert_eq!(
                crate::dot_q8_0_scaled_rows(r4, &acts).map(|v| v.map(f32::to_bits)),
                exp4
            );
            assert_eq!(
                crate::dot_q8_0_scaled_rows(r8, &acts).map(|v| v.map(f32::to_bits)),
                exp8
            );
            assert_eq!(
                crate::dot_q8_0_scaled_rows(r16, &acts).map(|v| v.map(f32::to_bits)),
                exp16
            );
            assert_eq!(
                crate::dot_q8_0_scaled_rows_view(r16, &act_views).map(|v| v.map(f32::to_bits)),
                exp16
            );
        }
    }

    #[test]
    fn x1_q8_0_prepared_matches_portable_zero_ulp() -> Result<(), &'static str> {
        use super::portable_q8_0_rows_validated;

        let Some(detected) = crate::Q8Dotprod::detect() else {
            return Ok(());
        };
        for (seed, block_count) in [1_usize, 2, 3, 17, 32].into_iter().enumerate() {
            let (rows, mut acts) = sample_q8_0_rows16_fixture(block_count, seed + 31);
            let act_views: Vec<(&[i8; 32], f32)> =
                acts.iter().map(|b| (&b.values, b.scale)).collect();
            let r1 = [rows[0].as_slice()];
            let r4: [&[u8]; 4] = std::array::from_fn(|r| rows[r].as_slice());
            let r8: [&[u8]; 8] = std::array::from_fn(|r| rows[r].as_slice());
            let r16: [&[u8]; 16] = std::array::from_fn(|r| rows[r].as_slice());
            let prep = detected
                .prepare_scaled(&acts)
                .ok_or("valid Q8_0 acts must prepare")?;
            assert_eq!(
                prep.dot_scaled_rows(r1).map(|v| v.map(f32::to_bits)),
                portable_q8_0_rows_validated(r1, &acts).map(|v| v.map(f32::to_bits))
            );
            assert_eq!(
                prep.dot_scaled_rows(r4).map(|v| v.map(f32::to_bits)),
                portable_q8_0_rows_validated(r4, &acts).map(|v| v.map(f32::to_bits))
            );
            assert_eq!(
                prep.dot_scaled_rows(r8).map(|v| v.map(f32::to_bits)),
                portable_q8_0_rows_validated(r8, &acts).map(|v| v.map(f32::to_bits))
            );
            assert_eq!(
                prep.dot_scaled_rows(r16).map(|v| v.map(f32::to_bits)),
                portable_q8_0_rows_validated(r16, &acts).map(|v| v.map(f32::to_bits))
            );
            let prep_view = detected
                .prepare_scaled_view(&act_views)
                .ok_or("valid Q8_0 view must prepare")?;
            assert_eq!(
                prep_view.dot_scaled_rows(r16).map(|v| v.map(f32::to_bits)),
                portable_q8_0_rows_validated(r16, &act_views).map(|v| v.map(f32::to_bits))
            );
            acts[block_count / 2].scale = f32::NAN;
            assert!(detected.prepare_scaled(&acts).is_none());
        }
        Ok(())
    }

    #[test]
    fn x1_q8_0_scaled_rows_reject_bad_inputs_and_overflow() {
        let Some(detected) = crate::Q8Dotprod::detect() else {
            return;
        };
        let (rows, mut acts) = sample_q8_0_rows16_fixture(5, 77);
        let r16: [&[u8]; 16] = std::array::from_fn(|r| rows[r].as_slice());

        assert!(detected.dot_scaled_rows([r16[0]], &[]).is_none());
        assert!(detected.dot_scaled_rows([r16[0], r16[1]], &acts).is_none());
        assert!(
            detected
                .dot_scaled_rows([r16[0], r16[1], r16[2]], &acts)
                .is_none()
        );
        assert!(
            detected
                .dot_scaled_rows([&rows[0][..5 * 34 - 1]], &acts)
                .is_none()
        );
        let mut mismatched16 = r16;
        mismatched16[15] = &rows[15][..4 * 34];
        assert!(detected.dot_scaled_rows(mismatched16, &acts).is_none());

        for bad_f16 in [0x7c00_u16, 0x7e00, 0xfc00] {
            for (bad_row, bad_blk) in [(0, 0), (0, 4), (3, 1), (7, 4), (15, 4)] {
                let mut bad_rows = rows.clone();
                bad_rows[bad_row][bad_blk * 34..bad_blk * 34 + 2]
                    .copy_from_slice(&bad_f16.to_le_bytes());
                let bad16: [&[u8]; 16] = std::array::from_fn(|r| bad_rows[r].as_slice());
                assert!(detected.dot_scaled_rows(bad16, &acts).is_none());
                if bad_row < 8 {
                    let bad8: [&[u8]; 8] = std::array::from_fn(|r| bad_rows[r].as_slice());
                    assert!(detected.dot_scaled_rows(bad8, &acts).is_none());
                }
                if bad_row < 4 {
                    let bad4: [&[u8]; 4] = std::array::from_fn(|r| bad_rows[r].as_slice());
                    assert!(detected.dot_scaled_rows(bad4, &acts).is_none());
                }
                if bad_row == 0 {
                    assert!(
                        detected
                            .dot_scaled_rows([bad_rows[0].as_slice()], &acts)
                            .is_none()
                    );
                }
            }
        }

        let orig_scale = acts[4].scale;
        acts[4].scale = f32::NAN;
        assert!(detected.dot_scaled_rows(r16, &acts).is_none());
        acts[4].scale = f32::INFINITY;
        assert!(detected.dot_scaled_rows(r16, &acts).is_none());
        acts[4].scale = orig_scale;

        let mut ovf_row = vec![0x7f_u8; 5 * 34];
        for blk in ovf_row.as_chunks_mut::<34>().0 {
            blk[..2].copy_from_slice(&0x7bff_u16.to_le_bytes());
        }
        let ovf_acts = vec![
            super::Q8ActivationBlock {
                scale: 1.0e32,
                values: [127; 32],
            };
            5
        ];
        assert!(
            detected
                .dot_scaled_rows([ovf_row.as_slice()], &ovf_acts)
                .is_none()
        );
        assert!(
            detected
                .dot_scaled_rows([ovf_row.as_slice(); 4], &ovf_acts)
                .is_none()
        );
        assert!(
            detected
                .dot_scaled_rows([ovf_row.as_slice(); 8], &ovf_acts)
                .is_none()
        );
        assert!(
            detected
                .dot_scaled_rows([ovf_row.as_slice(); 16], &ovf_acts)
                .is_none()
        );
    }

    fn sample_ptq1_32tok_fixture(
        block_count: usize,
        seed: usize,
    ) -> ([Vec<u8>; 8], Vec<Vec<super::Q8ActivationBlock>>) {
        let f16_pool: [u16; 8] = [
            0x0000, 0x8000, 0x0001, 0x8001, 0x3c01, 0xbc01, 0x3555, 0xb787,
        ];
        let act_scale_pool: [f32; 8] = [
            0.0_f32,
            -0.0_f32,
            f32::from_bits(1),
            -f32::from_bits(1),
            f32::from_bits(0x3da0_0001),
            -0.125_f32,
            0.375_f32,
            -0.0625_f32,
        ];
        let rows: [Vec<u8>; 8] = std::array::from_fn(|row_idx| {
            let mut row = vec![0_u8; block_count * 28];
            for blk in 0..block_count {
                let start = blk * 28;
                for i in 0..26 {
                    row[start + i] =
                        u8::try_from((row_idx * 41 + blk * 59 + i * 19 + seed * 13 + 5) % 256)
                            .unwrap_or(0);
                }
                let s_bits = f16_pool[(row_idx + blk * 3 + seed) % f16_pool.len()].to_le_bytes();
                row[start + 26] = s_bits[0];
                row[start + 27] = s_bits[1];
            }
            row
        });
        let acts: Vec<Vec<super::Q8ActivationBlock>> = (0..32_usize)
            .map(|tok| {
                (0..block_count * 4)
                    .map(|grp| {
                        let values: [i8; 32] = match (tok + grp + seed) % 4 {
                            0 => [i8::MIN; 32],
                            1 => [i8::MAX; 32],
                            2 => std::array::from_fn(|j| {
                                if (j + tok).is_multiple_of(2) {
                                    i8::MIN
                                } else {
                                    i8::MAX
                                }
                            }),
                            _ => std::array::from_fn(|j| {
                                i8::from_ne_bytes([j.to_le_bytes()[0]
                                    .wrapping_mul(71)
                                    .wrapping_add(tok.to_le_bytes()[0].wrapping_mul(17))
                                    .wrapping_add(grp.to_le_bytes()[0])
                                    .wrapping_add(seed.to_le_bytes()[0])])
                            }),
                        };
                        super::Q8ActivationBlock {
                            scale: act_scale_pool[(tok * 3 + grp + seed) % act_scale_pool.len()],
                            values,
                        }
                    })
                    .collect()
            })
            .collect();
        (rows, acts)
    }

    fn ptq1_test_caps() -> Option<Vec<crate::Ptq1Dotprod>> {
        let detected = crate::Ptq1Dotprod::detect()?;
        #[cfg(target_arch = "aarch64")]
        {
            let mut caps = vec![
                detected,
                crate::Ptq1Dotprod {
                    i8mm: false,
                    sve: false,
                    ..detected
                },
            ];
            if detected.sve {
                caps.push(crate::Ptq1Dotprod {
                    sve: false,
                    ..detected
                });
            }
            Some(caps)
        }
        #[cfg(not(target_arch = "aarch64"))]
        Some(vec![detected])
    }

    fn q8_0_test_caps() -> Option<Vec<crate::Q8Dotprod>> {
        let detected = crate::Q8Dotprod::detect()?;
        #[cfg(target_arch = "aarch64")]
        return Some(vec![
            detected,
            crate::Q8Dotprod {
                i8mm: false,
                ..detected
            },
        ]);
        #[cfg(not(target_arch = "aarch64"))]
        Some(vec![detected])
    }

    #[test]
    fn x1_ptq1_scaled_8x8_chunks_matches_portable() {
        use super::portable_ptq1_8x8_chunks_validated;

        let Some(caps) = ptq1_test_caps() else {
            return;
        };

        for (seed, block_count) in [1_usize, 2, 3, 4, 5, 7, 8, 9, 16, 17, 24, 32]
            .into_iter()
            .enumerate()
        {
            let (rows, acts) = sample_ptq1_32tok_fixture(block_count, seed + 1);
            let row_refs: [&[u8]; 8] = std::array::from_fn(|r| rows[r].as_slice());
            let chunk_slices: [[&[super::Q8ActivationBlock]; 8]; 4] =
                std::array::from_fn(|c| std::array::from_fn(|t| acts[c * 8 + t].as_slice()));

            for count in 1..=4_usize {
                let active_chunks = &chunk_slices[..count];
                for &cap in &caps {
                    for pinned in [false, true] {
                        let mut expected = vec![[[0.0_f32; 8]; 8]; count];
                        let mut actual = vec![[[123.5_f32; 8]; 8]; count];
                        let mut per_chunk = [[[0.0_f32; 8]; 8]; 4];
                        if pinned {
                            assert!(
                                portable_ptq1_8x8_chunks_validated::<true, _>(
                                    row_refs,
                                    active_chunks,
                                    &mut expected,
                                )
                                .is_some()
                            );
                            assert!(
                                cap.dot_scaled_8x8_chunks::<true, _>(
                                    row_refs,
                                    active_chunks,
                                    &mut actual,
                                )
                                .is_some()
                            );
                            for (c, chunk) in active_chunks.iter().enumerate() {
                                assert!(
                                    cap.dot_scaled_8x_tokens::<true>(
                                        row_refs,
                                        chunk,
                                        &mut per_chunk[c]
                                    )
                                    .is_some()
                                );
                            }
                        } else {
                            assert!(
                                portable_ptq1_8x8_chunks_validated::<false, _>(
                                    row_refs,
                                    active_chunks,
                                    &mut expected,
                                )
                                .is_some()
                            );
                            assert!(
                                cap.dot_scaled_8x8_chunks::<false, _>(
                                    row_refs,
                                    active_chunks,
                                    &mut actual,
                                )
                                .is_some()
                            );
                            for (c, chunk) in active_chunks.iter().enumerate() {
                                assert!(
                                    cap.dot_scaled_8x_tokens::<false>(
                                        row_refs,
                                        chunk,
                                        &mut per_chunk[c]
                                    )
                                    .is_some()
                                );
                            }
                        }
                        for ((e, a), single) in
                            expected.iter().zip(actual.iter()).zip(per_chunk.iter())
                        {
                            for r in 0..8 {
                                assert_eq!(e[r].map(f32::to_bits), a[r].map(f32::to_bits));
                                assert_eq!(single[r].map(f32::to_bits), a[r].map(f32::to_bits));
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn x1_ptq1_scaled_8x8_chunks_rejects_bad_inputs() {
        let Some(caps) = ptq1_test_caps() else {
            return;
        };
        let detected = caps[0];
        let (rows, acts) = sample_ptq1_32tok_fixture(10, 7);
        let row_refs: [&[u8]; 8] = std::array::from_fn(|r| rows[r].as_slice());
        let chunk_slices: [[&[super::Q8ActivationBlock]; 8]; 4] =
            std::array::from_fn(|c| std::array::from_fn(|t| acts[c * 8 + t].as_slice()));
        let sentinel = [[[99.25_f32; 8]; 8]; 4];
        let mut out = sentinel;

        assert!(
            detected
                .dot_scaled_8x8_chunks::<true, super::Q8ActivationBlock>(row_refs, &[], &mut [])
                .is_none()
        );
        let mut five_out = [[[0.0_f32; 8]; 8]; 5];
        let five_chunks = [
            chunk_slices[0],
            chunk_slices[1],
            chunk_slices[2],
            chunk_slices[3],
            chunk_slices[0],
        ];
        assert!(
            detected
                .dot_scaled_8x8_chunks::<true, _>(row_refs, &five_chunks, &mut five_out)
                .is_none()
        );
        assert!(
            detected
                .dot_scaled_8x8_chunks::<true, _>(row_refs, &chunk_slices[..1], &mut out[..2])
                .is_none()
        );

        let mut short_rows = row_refs;
        short_rows[7] = &rows[7][..rows[7].len() - 1];
        assert!(
            detected
                .dot_scaled_8x8_chunks::<false, _>(short_rows, &chunk_slices[..2], &mut out[..2])
                .is_none()
        );

        for bad_f16 in [0x7e00_u16, 0x7c00, 0xfc00] {
            for (bad_row, bad_blk) in [(0, 0), (5, 1), (3, 7), (7, 8), (6, 9)] {
                let mut bad_rows = rows.clone();
                let off = bad_blk * 28 + 26;
                bad_rows[bad_row][off..off + 2].copy_from_slice(&bad_f16.to_le_bytes());
                let bad_row_refs: [&[u8]; 8] = std::array::from_fn(|r| bad_rows[r].as_slice());
                for &cap in &caps {
                    assert!(
                        cap.dot_scaled_8x8_chunks::<true, _>(
                            bad_row_refs,
                            &chunk_slices[..2],
                            &mut out[..2]
                        )
                        .is_none()
                    );
                    assert!(
                        cap.dot_scaled_8x8_chunks::<false, _>(
                            bad_row_refs,
                            &chunk_slices[..2],
                            &mut out[..2]
                        )
                        .is_none()
                    );
                }
            }
        }
    }

    #[test]
    fn x1_ptq1_scaled_8x8_chunks_rejects_bad_scales_and_overflow() {
        let Some(caps) = ptq1_test_caps() else {
            return;
        };
        let detected = caps[0];
        let (rows, mut acts) = sample_ptq1_32tok_fixture(2, 7);
        let row_refs: [&[u8]; 8] = std::array::from_fn(|r| rows[r].as_slice());
        let sentinel = [[[99.25_f32; 8]; 8]; 4];
        let mut out = sentinel;

        for bad_scale in [f32::NAN, f32::INFINITY] {
            acts[13][3].scale = bad_scale;
            let bad_chunks: [[&[super::Q8ActivationBlock]; 8]; 2] =
                std::array::from_fn(|c| std::array::from_fn(|t| acts[c * 8 + t].as_slice()));
            assert!(
                detected
                    .dot_scaled_8x8_chunks::<true, _>(row_refs, &bad_chunks, &mut out[..2])
                    .is_none()
            );
            assert!(
                detected
                    .dot_scaled_8x8_chunks::<false, _>(row_refs, &bad_chunks, &mut out[..2])
                    .is_none()
            );
        }

        let mut ovf_row = vec![0xff_u8; 28];
        ovf_row[26..28].copy_from_slice(&0x7bff_u16.to_le_bytes());
        let ovf_refs = [ovf_row.as_slice(); 8];
        let ovf_act = vec![
            super::Q8ActivationBlock {
                scale: f32::MAX,
                values: [127; 32]
            };
            4
        ];
        let ovf_chunk = [ovf_act.as_slice(); 8];
        for &cap in &caps {
            assert!(
                cap.dot_scaled_8x8_chunks::<true, _>(ovf_refs, &[ovf_chunk], &mut out[..1])
                    .is_none()
            );
            assert!(
                cap.dot_scaled_8x8_chunks::<false, _>(ovf_refs, &[ovf_chunk], &mut out[..1])
                    .is_none()
            );
        }
        for (out_c, sen_c) in out.iter().zip(sentinel.iter()) {
            for (out_r, sen_r) in out_c.iter().zip(sen_c.iter()) {
                assert_eq!(out_r.map(f32::to_bits), sen_r.map(f32::to_bits));
            }
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Verifies 1..=8 chunk PTQ1 pinned I8MM panel parity, slice-fill packing, and NaN/Inf scale rejection."
    )]
    #[test]
    fn x1_ptq1_pinned_i8mm_panels_match_portable_and_reject_bad_inputs() {
        use super::portable_ptq1_8x8_chunks_validated;

        let Some(caps) = ptq1_test_caps() else {
            return;
        };
        for block_count in [1_usize, 3, 4, 5, 7, 8, 9, 40, 136] {
            let q8_len = block_count * 4;
            let (rows, acts) = sample_ptq1_32tok_fixture(block_count, 13);
            let row_refs: [&[u8]; 8] = std::array::from_fn(|r| rows[r].as_slice());
            let chunk_slices: [[&[super::Q8ActivationBlock]; 8]; 8] = std::array::from_fn(|c| {
                std::array::from_fn(|t| acts[(c * 8 + t) % acts.len()].as_slice())
            });

            for count in 1..=8_usize {
                let active_chunks = &chunk_slices[..count];
                let mut expected = vec![[[0.0_f32; 8]; 8]; count];
                for (c_chunk, e_chunk) in active_chunks.chunks(4).zip(expected.chunks_mut(4)) {
                    assert!(
                        portable_ptq1_8x8_chunks_validated::<true, _>(row_refs, c_chunk, e_chunk)
                            .is_some()
                    );
                }

                for &cap in &caps {
                    let mut packed = Vec::new();
                    for chunk in active_chunks {
                        for p in 0..4 {
                            assert!(
                                cap.append_pinned_i8mm_pair_blocks(
                                    chunk[2 * p],
                                    chunk[2 * p + 1],
                                    &mut packed,
                                )
                                .is_some()
                            );
                        }
                    }
                    let mut packed_into =
                        vec![super::Ptq1PinnedI8mmPairBlock::zeroed(); count * 4 * q8_len];
                    for (c, chunk) in active_chunks.iter().enumerate() {
                        for p in 0..4 {
                            let start = (c * 4 + p) * q8_len;
                            assert!(
                                cap.pack_pinned_i8mm_pair_blocks_into(
                                    chunk[2 * p],
                                    chunk[2 * p + 1],
                                    &mut packed_into[start..start + q8_len],
                                )
                                .is_some()
                            );
                        }
                    }
                    for (a, b) in packed.iter().zip(&packed_into) {
                        assert_eq!(a.planes, b.planes);
                        assert_eq!(a.scales.map(f32::to_bits), b.scales.map(f32::to_bits));
                    }

                    let panel_chunks: Vec<[&[super::Ptq1PinnedI8mmPairBlock]; 4]> = (0..count)
                        .map(|c| {
                            std::array::from_fn(|p| {
                                let start = (c * 4 + p) * q8_len;
                                &packed_into[start..start + q8_len]
                            })
                        })
                        .collect();

                    let mut actual = vec![[[f32::NAN; 8]; 8]; count];
                    assert!(
                        cap.dot_scaled_8x8_chunks_with_pinned_panels(
                            row_refs,
                            active_chunks,
                            &panel_chunks,
                            &mut actual,
                        )
                        .is_some()
                    );
                    for (e, a) in expected.iter().zip(actual.iter()) {
                        for r in 0..8 {
                            assert_eq!(e[r].map(f32::to_bits), a[r].map(f32::to_bits));
                        }
                    }
                }
            }

            let mut packed_bad = Vec::new();
            for chunk in &chunk_slices[..2] {
                for p in 0..4 {
                    assert!(
                        caps[0]
                            .append_pinned_i8mm_pair_blocks(
                                chunk[2 * p],
                                chunk[2 * p + 1],
                                &mut packed_bad,
                            )
                            .is_some()
                    );
                }
            }
            let panel_chunks_bad: [[&[super::Ptq1PinnedI8mmPairBlock]; 4]; 2] =
                std::array::from_fn(|c| {
                    std::array::from_fn(|p| {
                        let start = (c * 4 + p) * q8_len;
                        &packed_bad[start..start + q8_len]
                    })
                });
            for bad_f16 in [0x7e00_u16, 0x7c00, 0xfc00] {
                for bad_blk in [0_usize, block_count / 2, block_count - 1] {
                    let mut bad_rows = rows.clone();
                    let row_slot = (bad_blk + 5) % 8;
                    let off = bad_blk * 28 + 26;
                    bad_rows[row_slot][off..off + 2].copy_from_slice(&bad_f16.to_le_bytes());
                    let bad_refs: [&[u8]; 8] = std::array::from_fn(|r| bad_rows[r].as_slice());
                    for &cap in &caps {
                        let mut out = [[[0.0_f32; 8]; 8]; 2];
                        assert!(
                            cap.dot_scaled_8x8_chunks_with_pinned_panels(
                                bad_refs,
                                &chunk_slices[..2],
                                &panel_chunks_bad,
                                &mut out,
                            )
                            .is_none()
                        );
                    }
                }
            }
        }

        // Mismatched panel length must return None without touching sentinel outputs.
        let q8_len = 3 * 4;
        let (rows, acts) = sample_ptq1_32tok_fixture(3, 13);
        let row_refs: [&[u8]; 8] = std::array::from_fn(|r| rows[r].as_slice());
        let chunk_slices: [[&[super::Q8ActivationBlock]; 8]; 4] =
            std::array::from_fn(|c| std::array::from_fn(|t| acts[c * 8 + t].as_slice()));
        for &cap in &caps {
            let mut packed = Vec::new();
            for chunk in &chunk_slices {
                for p in 0..4 {
                    assert!(
                        cap.append_pinned_i8mm_pair_blocks(
                            chunk[2 * p],
                            chunk[2 * p + 1],
                            &mut packed,
                        )
                        .is_some()
                    );
                }
            }
            let panel_chunks: [[&[super::Ptq1PinnedI8mmPairBlock]; 4]; 4] =
                std::array::from_fn(|c| {
                    std::array::from_fn(|p| {
                        let start = (c * 4 + p) * q8_len;
                        &packed[start..start + q8_len]
                    })
                });
            let sentinel = [[[77.5_f32; 8]; 8]; 4];
            let mut bad_out = sentinel;
            let mut bad_panels = panel_chunks;
            bad_panels[1][2] = &panel_chunks[1][2][..q8_len - 1];
            assert!(
                cap.dot_scaled_8x8_chunks_with_pinned_panels(
                    row_refs,
                    &chunk_slices,
                    &bad_panels,
                    &mut bad_out,
                )
                .is_none()
            );
            assert_eq!(
                bad_out.map(|c| c.map(|r| r.map(f32::to_bits))),
                sentinel.map(|c| c.map(|r| r.map(f32::to_bits)))
            );
        }
    }

    #[test]
    fn x1_q8_0_scaled_8x8_chunks_matches_portable_across_modes() {
        use super::portable_q8_0_8x8_chunks_validated;

        let Some(caps) = q8_0_test_caps() else {
            return;
        };

        for (seed, block_count) in [1_usize, 3, 16, 40].into_iter().enumerate() {
            let (rows, acts) = sample_q8_0_fixture(block_count, 32, seed + 5);
            let row_refs: [&[u8]; 8] = std::array::from_fn(|r| rows[r].as_slice());
            let chunk_slices: [[&[super::Q8ActivationBlock]; 8]; 4] =
                std::array::from_fn(|c| std::array::from_fn(|t| acts[c * 8 + t].as_slice()));

            for count in 1..=4_usize {
                let active_chunks = &chunk_slices[..count];
                let mut expected = vec![[[0.0_f32; 8]; 8]; count];
                assert!(
                    portable_q8_0_8x8_chunks_validated(row_refs, active_chunks, &mut expected)
                        .is_some()
                );
                for &cap in &caps {
                    let mut actual = vec![[[f32::NAN; 8]; 8]; count];
                    assert!(
                        cap.dot_scaled_8x8_chunks(row_refs, active_chunks, &mut actual)
                            .is_some()
                    );
                    for (c, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
                        let mut per_chunk = [[f32::NAN; 8]; 8];
                        assert!(
                            cap.dot_scaled_8x_tokens(row_refs, &active_chunks[c], &mut per_chunk)
                                .is_some()
                        );
                        for r in 0..8 {
                            assert_eq!(e[r].map(f32::to_bits), a[r].map(f32::to_bits));
                            assert_eq!(per_chunk[r].map(f32::to_bits), a[r].map(f32::to_bits));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn x1_q8_0_scaled_8x8_chunks_rejects_bad_inputs() {
        let Some(caps) = q8_0_test_caps() else {
            return;
        };
        let detected = caps[0];
        let (rows, mut acts) = sample_q8_0_fixture(2, 32, 99);
        let row_refs: [&[u8]; 8] = std::array::from_fn(|r| rows[r].as_slice());
        let chunk_slices: [[&[super::Q8ActivationBlock]; 8]; 4] =
            std::array::from_fn(|c| std::array::from_fn(|t| acts[c * 8 + t].as_slice()));
        let sentinel = [[[123.5_f32; 8]; 8]; 4];
        let mut out = sentinel;

        assert!(
            detected
                .dot_scaled_8x8_chunks::<super::Q8ActivationBlock>(row_refs, &[], &mut [])
                .is_none()
        );
        let mut five_out = [[[0.0_f32; 8]; 8]; 5];
        let five_chunks = [
            chunk_slices[0],
            chunk_slices[1],
            chunk_slices[2],
            chunk_slices[3],
            chunk_slices[0],
        ];
        assert!(
            detected
                .dot_scaled_8x8_chunks(row_refs, &five_chunks, &mut five_out)
                .is_none()
        );
        assert!(
            detected
                .dot_scaled_8x8_chunks(row_refs, &chunk_slices[..1], &mut out[..2])
                .is_none()
        );

        let mut short_rows = row_refs;
        short_rows[4] = &row_refs[4][..33];
        assert!(
            detected
                .dot_scaled_8x8_chunks(short_rows, &chunk_slices[..2], &mut out[..2])
                .is_none()
        );

        for bad_f16 in [0x7c00_u16, 0x7e00, 0xfc00] {
            let mut bad_rows = rows.clone();
            bad_rows[6][34..36].copy_from_slice(&bad_f16.to_le_bytes());
            let bad_refs: [&[u8]; 8] = std::array::from_fn(|r| bad_rows[r].as_slice());
            for &cap in &caps {
                assert!(
                    cap.dot_scaled_8x8_chunks(bad_refs, &chunk_slices[..2], &mut out[..2])
                        .is_none()
                );
            }
        }

        acts[21][1].scale = f32::NAN;
        let bad_chunks: [[&[super::Q8ActivationBlock]; 8]; 3] =
            std::array::from_fn(|c| std::array::from_fn(|t| acts[c * 8 + t].as_slice()));
        assert!(
            detected
                .dot_scaled_8x8_chunks(row_refs, &bad_chunks, &mut out[..3])
                .is_none()
        );

        let mut big_rows = rows;
        for row in &mut big_rows {
            row[..2].copy_from_slice(&0x7bff_u16.to_le_bytes());
            row[2..34].fill(0x7f);
            row[34..36].copy_from_slice(&0x7bff_u16.to_le_bytes());
            row[36..68].fill(0x7f);
        }
        let big_acts = vec![
            super::Q8ActivationBlock {
                scale: 1.0e32,
                values: [127; 32]
            };
            2
        ];
        let big_refs: [&[u8]; 8] = std::array::from_fn(|r| big_rows[r].as_slice());
        let big_chunk = [big_acts.as_slice(); 8];
        for &cap in &caps {
            assert!(
                cap.dot_scaled_8x8_chunks(big_refs, &[big_chunk], &mut out[..1])
                    .is_none()
            );
        }
        for (out_c, sen_c) in out.iter().zip(sentinel.iter()) {
            for (out_r, sen_r) in out_c.iter().zip(sen_c.iter()) {
                assert_eq!(out_r.map(f32::to_bits), sen_r.map(f32::to_bits));
            }
        }
    }
}
