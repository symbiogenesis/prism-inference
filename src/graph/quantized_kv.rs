//! Explicit lossy, packed 256-channel KV storage for Snapdragon CPU evaluation.
//!
//! The codec is independent of model-weight packing. Only full-attention K/V
//! change; recurrent/convolution state and checkpoint ownership stay exact.

use super::{KvCache, checked_product, checked_sum};
use crate::kernels::{MAX_ROW_WORKERS, RowParallelism};
use crate::kv_quant::{Bits, HEAD_DIM, Quantizer};
use crate::{Result, invalid};
use std::sync::Arc;

#[cfg(test)]
#[path = "quantized_kv_tests.rs"]
mod tests;

/// Immutable full-attention cache precision selected before graph construction.
///
/// Rotated profiles are experimental approximations, not FP16-equivalent paths.
/// Both Snapdragon X1 Elite and X2 Elite Extreme use the same portable/NEON
/// contract; selection does not claim measured device benefit or quality.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum KvCachePolicy {
    /// Existing exact binary16 cache and attention contract.
    #[default]
    F16,
    /// Four-bit rotated scalar keys and values, with one F32 scale per head.
    Rotated4,
    /// Four-bit keys with version-2 guarded fitted scales on four-bit values.
    Rotated4FittedValues,
    /// Four-bit rotated keys and values, both encoded with version-3 scale-code co-optimization.
    Rotated4FittedKv,
    /// Four-bit keys and three-bit values, with one F32 scale per head.
    Rotated4Keys3Values,
}

impl KvCachePolicy {
    /// Stable configuration name, independent of CPU dispatch and model packing.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Rotated4 => "rsq4",
            Self::Rotated4FittedValues => "rsq4vf",
            Self::Rotated4FittedKv => "rsq4kvf",
            Self::Rotated4Keys3Values => "rsq4k3v",
        }
    }

    /// Conservative packed K/V element payload for one layer at full context.
    ///
    /// Includes F32 vector scales. Page tables, codec state, scratch and allocator
    /// overhead remain separate; this is not a process-memory projection.
    ///
    /// # Errors
    /// Rejects unsupported dimensions, zero geometry and arithmetic overflow.
    pub fn layer_payload_bytes(self, head: usize, heads: usize, context: usize) -> Result<usize> {
        if head == 0 || heads == 0 || context == 0 {
            return Err(invalid("invalid KV payload geometry"));
        }
        let record = if self == Self::F16 {
            checked_product(&[2, head, size_of::<u16>()])?
        } else {
            if head != HEAD_DIM {
                return Err(invalid("rotated KV requires 256-channel heads"));
            }
            let (key, value) = self.bits().ok_or_else(|| invalid("missing KV precision"))?;
            checked_sum(
                Quantizer::new(key, KEY_SEED).encoded_len(),
                Quantizer::new(value, VALUE_SEED).encoded_len(),
            )?
        };
        checked_product(&[record, heads, context])
    }

    const fn bits(self) -> Option<(Bits, Bits)> {
        match self {
            Self::F16 => None,
            Self::Rotated4 | Self::Rotated4FittedValues | Self::Rotated4FittedKv => {
                Some((Bits::Four, Bits::Four))
            }
            Self::Rotated4Keys3Values => Some((Bits::Four, Bits::Three)),
        }
    }
}

// Fixed, distinct transforms prevent accidental key/value basis reuse. They are
// numerical profile identities, unrelated to the sampler's request seed.
const KEY_SEED: u64 = 0x5357_4946_544b_5631;
const VALUE_SEED: u64 = 0x5357_4946_5456_5631;

fn same_quantizer(left: &Quantizer, right: &Quantizer) -> bool {
    left.identity() == right.identity()
        && left.uses_neon() == right.uses_neon()
        && left.encoded_len() == right.encoded_len()
}

pub(super) struct PackedBatchReservation {
    add_bytes: usize,
    extra_pages: Vec<(Vec<u8>, usize)>,
    trailing_tail: Option<Vec<u8>>,
}

#[derive(Clone, Copy, Default)]
pub(super) struct ScheduledWorkItem {
    pub(super) token_start: usize,
    pub(super) token_len: usize,
    pub(super) kv_head: usize,
    pub(super) out_len: usize,
}

/// Reusable scheduling buffers for 2D `(token_pair, kv_head)` causal-cost-balanced prefill.
#[derive(Default)]
pub(super) struct CausalScheduleScratch {
    costs: Vec<usize>,
    token_starts: Vec<usize>,
    kv_heads_idx: Vec<usize>,
    order: Vec<usize>,
    assigned: Vec<usize>,
    pub(super) item_offsets: Vec<usize>,
    pub(super) ordered: Vec<ScheduledWorkItem>,
}

/// Reusable scratch buffers for parallel rotated-KV tile encoding and 2D causal-balanced attention.
#[derive(Default)]
pub(super) struct PackedTileScratch {
    encoded_keys: Vec<u8>,
    encoded_values: Vec<u8>,
    kv_major_outputs: Vec<f32>,
    worker_scores: Vec<f32>,
    schedule: CausalScheduleScratch,
}

pub(super) struct PackedCache {
    pages: Vec<Arc<Vec<u8>>>,
    tail: Vec<u8>,
    prepared: Vec<u8>,
    prepared_ready: bool,
    quantizer: Quantizer,
    heads: usize,
    row_bytes: usize,
    page_bytes: usize,
    maximum: usize,
    len: usize,
}

impl PackedCache {
    fn new(width: usize, context: usize, bits: Bits, seed: u64, scalar: bool) -> Result<Self> {
        if !width.is_multiple_of(HEAD_DIM) || width == 0 || context == 0 {
            return Err(invalid("invalid packed KV geometry"));
        }
        let quantizer = if scalar {
            Quantizer::scalar(bits, seed)
        } else {
            Quantizer::new(bits, seed)
        };
        let heads = width / HEAD_DIM;
        let row_bytes = checked_product(&[heads, quantizer.encoded_len()])?;
        Ok(Self {
            pages: Vec::new(),
            tail: Vec::new(),
            prepared: Vec::new(),
            prepared_ready: false,
            quantizer,
            heads,
            row_bytes,
            page_bytes: checked_product(&[row_bytes, context.min(super::KV_PAGE_TOKENS)])?,
            maximum: checked_product(&[row_bytes, context])?,
            len: 0,
        })
    }

    fn prepare(&mut self, values: &[f32]) -> Result<()> {
        self.prepared_ready = false;
        if values.len() != self.heads * HEAD_DIM {
            return Err(invalid("packed KV row width mismatch"));
        }
        self.reserve()?;
        resize_bytes(self.row_bytes, &mut self.prepared)?;
        for (head, output) in values
            .as_chunks::<HEAD_DIM>()
            .0
            .iter()
            .zip(self.prepared.chunks_exact_mut(self.quantizer.encoded_len()))
        {
            self.quantizer.encode(head, output)?;
        }
        Ok(())
    }

    fn reserve(&mut self) -> Result<()> {
        let next = checked_sum(self.len, self.row_bytes)?;
        if next > self.maximum {
            return Err(invalid("packed KV append exceeds context bound"));
        }
        let required = checked_sum(self.tail.len(), self.row_bytes)?;
        let remaining = self.maximum - (self.len - self.tail.len());
        let limit = self.page_bytes.min(remaining);
        if required > self.tail.capacity() {
            let target = self
                .tail
                .capacity()
                .saturating_mul(2)
                .min(limit)
                .max(required);
            self.tail.try_reserve_exact(target - self.tail.len())?;
        }
        if required == self.page_bytes || next == self.maximum {
            self.pages.try_reserve_exact(1)?;
        }
        Ok(())
    }

    pub(super) fn reserve_batch(&mut self, count: usize) -> Result<PackedBatchReservation> {
        let add_bytes = checked_product(&[count, self.row_bytes])?;
        let next = checked_sum(self.len, add_bytes)?;
        if next > self.maximum {
            return Err(invalid("packed KV append exceeds context bound"));
        }
        if count == 0 {
            return Ok(PackedBatchReservation {
                add_bytes: 0,
                extra_pages: Vec::new(),
                trailing_tail: None,
            });
        }
        let first_bytes = (self.page_bytes - self.tail.len())
            .min(self.maximum - self.len)
            .min(add_bytes);
        let first_required = checked_sum(self.tail.len(), first_bytes)?;
        let first_remaining = self.maximum - (self.len - self.tail.len());
        let first_limit = self.page_bytes.min(first_remaining);
        let first_tail_reserve = if first_required > self.tail.capacity() {
            let target = self
                .tail
                .capacity()
                .saturating_mul(2)
                .min(first_limit)
                .max(first_required);
            target - self.tail.len()
        } else {
            0
        };
        let mut simulated_len = self.len + first_bytes;
        let mut sealed_pages = usize::from(
            self.tail.len() + first_bytes == self.page_bytes || simulated_len == self.maximum,
        );
        let mut extra_pages = Vec::new();
        let mut trailing_tail = None;
        while simulated_len < next {
            let chunk = self
                .page_bytes
                .min(self.maximum - simulated_len)
                .min(next - simulated_len);
            let mut buf = Vec::new();
            buf.try_reserve_exact(chunk)?;
            simulated_len += chunk;
            if chunk == self.page_bytes || simulated_len == self.maximum {
                sealed_pages += 1;
                extra_pages.push((buf, chunk));
            } else {
                trailing_tail = Some(buf);
            }
        }
        self.pages.try_reserve_exact(sealed_pages)?;
        if first_tail_reserve > 0 {
            self.tail.try_reserve_exact(first_tail_reserve)?;
        }
        Ok(PackedBatchReservation {
            add_bytes,
            extra_pages,
            trailing_tail,
        })
    }

    pub(super) fn commit_batch(&mut self, encoded: &[u8], reservation: PackedBatchReservation) {
        debug_assert_eq!(encoded.len(), reservation.add_bytes);
        if encoded.is_empty() {
            return;
        }
        let first_bytes = (self.page_bytes - self.tail.len())
            .min(self.maximum - self.len)
            .min(encoded.len());
        let (first_slice, mut rest) = encoded.split_at(first_bytes);
        self.tail.extend_from_slice(first_slice);
        self.len += first_bytes;
        self.prepared_ready = false;
        if self.tail.len() == self.page_bytes || self.len == self.maximum {
            self.pages.push(Arc::new(std::mem::take(&mut self.tail)));
        }
        for (mut buf, chunk_len) in reservation.extra_pages {
            let (chunk, next_rest) = rest.split_at(chunk_len);
            rest = next_rest;
            buf.extend_from_slice(chunk);
            self.len += chunk.len();
            self.pages.push(Arc::new(buf));
        }
        if let Some(mut buf) = reservation.trailing_tail {
            buf.extend_from_slice(rest);
            self.len += rest.len();
            self.tail = buf;
        }
    }

    pub(super) fn commit(&mut self) {
        debug_assert_eq!(self.prepared.len(), self.row_bytes);
        self.tail.extend_from_slice(&self.prepared);
        self.len += self.row_bytes;
        self.prepared_ready = false;
        if self.tail.len() == self.page_bytes || self.len == self.maximum {
            self.pages.push(Arc::new(std::mem::take(&mut self.tail)));
        }
    }

    fn sealed_rows(&self) -> usize {
        self.len
            .saturating_sub(self.tail.len())
            .checked_div(self.row_bytes)
            .unwrap_or(0)
    }

    fn page_rows(&self) -> impl Iterator<Item = &[u8]> {
        self.pages
            .iter()
            .flat_map(|page| page.chunks_exact(self.row_bytes))
    }

    fn tail_rows(&self) -> impl Iterator<Item = &[u8]> {
        self.tail.chunks_exact(self.row_bytes)
    }

    fn rows(&self) -> impl Iterator<Item = &[u8]> {
        self.page_rows().chain(self.tail_rows())
    }

    fn row(&self, position: usize) -> Result<&[u8]> {
        if position >= self.len / self.row_bytes {
            return Err(invalid("packed KV position exceeds retained history"));
        }
        let page_rows = self.page_bytes / self.row_bytes;
        let page = position / page_rows;
        let offset = position % page_rows * self.row_bytes;
        let bytes = self
            .pages
            .get(page)
            .map_or(self.tail.as_slice(), |page| page.as_slice());
        bytes
            .get(offset..offset + self.row_bytes)
            .ok_or_else(|| invalid("invalid packed KV page extent"))
    }

    fn record<'a>(&self, row: &'a [u8], head: usize) -> &'a [u8] {
        let width = self.quantizer.encoded_len();
        &row[head * width..(head + 1) * width]
    }

    pub(super) fn same_representation(&self, other: &Self) -> bool {
        self.heads == other.heads
            && self.row_bytes == other.row_bytes
            && self.page_bytes == other.page_bytes
            && self.maximum == other.maximum
            && same_quantizer(&self.quantizer, &other.quantizer)
    }

    pub(super) fn validate_restore(
        &self,
        live: &Self,
        expected_tokens: Option<usize>,
    ) -> Result<()> {
        if !self.same_representation(live) {
            return Err(invalid("packed KV cache representation mismatch"));
        }
        if self.heads == 0
            || self.row_bytes == 0
            || self.page_bytes == 0
            || !self.maximum.is_multiple_of(self.row_bytes)
            || !self.page_bytes.is_multiple_of(self.row_bytes)
            || self.len > self.maximum
            || !self.len.is_multiple_of(self.row_bytes)
        {
            return Err(invalid("malformed packed KV cache geometry"));
        }
        let tokens = self.len / self.row_bytes;
        if expected_tokens.is_some_and(|expected| tokens != expected) {
            return Err(invalid(
                "packed KV cache token length does not match checkpoint position",
            ));
        }
        let max_tokens = self.maximum / self.row_bytes;
        let page_tokens = self.page_bytes / self.row_bytes;
        if page_tokens != max_tokens.min(super::KV_PAGE_TOKENS) {
            return Err(invalid("malformed packed KV page geometry"));
        }
        if tokens == max_tokens {
            if !self.tail.is_empty() {
                return Err(invalid("full packed KV cache must have an empty tail"));
            }
            let expected_pages = tokens.div_ceil(page_tokens);
            if self.pages.len() != expected_pages {
                return Err(invalid("malformed packed KV page count"));
            }
            for (index, page) in self.pages.iter().enumerate() {
                let expected_bytes =
                    if index + 1 < expected_pages || tokens.is_multiple_of(page_tokens) {
                        self.page_bytes
                    } else {
                        (tokens % page_tokens) * self.row_bytes
                    };
                if page.len() != expected_bytes {
                    return Err(invalid("malformed packed KV sealed page length"));
                }
            }
        } else if self.pages.len() != tokens / super::KV_PAGE_TOKENS
            || self.pages.iter().any(|page| page.len() != self.page_bytes)
            || self.tail.len() != (tokens % super::KV_PAGE_TOKENS) * self.row_bytes
        {
            return Err(invalid("malformed packed KV page or tail partition"));
        }
        Ok(())
    }

    pub(super) fn copy(&self) -> Result<Self> {
        let mut pages = super::capacity(self.pages.len())?;
        pages.extend(self.pages.iter().map(Arc::clone));
        Ok(Self {
            pages,
            tail: super::copy_values(&self.tail)?,
            prepared: Vec::new(),
            prepared_ready: false,
            quantizer: self.quantizer.clone(),
            heads: self.heads,
            row_bytes: self.row_bytes,
            page_bytes: self.page_bytes,
            maximum: self.maximum,
            len: self.len,
        })
    }

    pub(super) fn copy_into(&self, target: &mut Self) -> Result<()> {
        if !self.same_representation(target) {
            *target = self.copy()?;
            return Ok(());
        }
        if self.pages.len() > target.pages.capacity() {
            target
                .pages
                .try_reserve_exact(self.pages.len() - target.pages.len())?;
        }
        if self.tail.len() > target.tail.capacity() {
            target
                .tail
                .try_reserve_exact(self.tail.len() - target.tail.len())?;
        }
        target.pages.clear();
        target.pages.extend(self.pages.iter().map(Arc::clone));
        target.tail.resize(self.tail.len(), 0);
        target.tail.copy_from_slice(&self.tail);
        target.prepared.clear();
        target.prepared_ready = false;
        target.len = self.len;
        Ok(())
    }

    pub(super) fn reserve_restore(
        &mut self,
        saved: &Self,
        expected_tokens: Option<usize>,
    ) -> Result<()> {
        saved.validate_restore(self, expected_tokens)?;
        if saved.pages.len() > self.pages.capacity() {
            self.pages
                .try_reserve_exact(saved.pages.len() - self.pages.len())?;
        }
        if saved.tail.len() > self.tail.capacity() {
            self.tail
                .try_reserve_exact(saved.tail.len() - self.tail.len())?;
        }
        Ok(())
    }

    pub(super) fn apply_restore_in_place(&mut self, saved: &Self) {
        self.pages.clear();
        self.pages.extend(saved.pages.iter().map(Arc::clone));
        self.tail.resize(saved.tail.len(), 0);
        self.tail.copy_from_slice(&saved.tail);
        self.prepared.clear();
        self.prepared_ready = false;
        self.len = saved.len;
    }

    #[allow(
        dead_code,
        reason = "Direct in-place restoration contract tested and available alongside two-phase restore."
    )]
    pub(super) fn restore_in_place(&mut self, saved: &Self) -> Result<()> {
        self.reserve_restore(saved, None)?;
        self.apply_restore_in_place(saved);
        Ok(())
    }

    pub(super) fn clear(&mut self) {
        self.pages.clear();
        self.tail.clear();
        self.prepared.clear();
        self.prepared_ready = false;
        self.len = 0;
    }

    pub(super) fn truncate_to(&mut self, position: usize) -> Result<()> {
        if self.heads == 0
            || self.row_bytes == 0
            || self.page_bytes == 0
            || !self.page_bytes.is_multiple_of(self.row_bytes)
            || self.len > self.maximum
            || !self.len.is_multiple_of(self.row_bytes)
        {
            return Err(invalid("malformed packed KV cache geometry"));
        }
        let tokens = self.len / self.row_bytes;
        if position > tokens {
            return Err(invalid(
                "packed KV truncation position exceeds cached length",
            ));
        }
        if position == tokens {
            self.prepared.clear();
            self.prepared_ready = false;
            return Ok(());
        }
        let page_tokens = self.page_bytes / self.row_bytes;
        let keep_pages = position / page_tokens;
        let tail_tokens = position % page_tokens;
        let tail_bytes = checked_product(&[tail_tokens, self.row_bytes])?;
        if keep_pages < self.pages.len() {
            if self.pages[keep_pages].len() < tail_bytes {
                return Err(invalid("malformed packed KV page extent during truncation"));
            }
            if tail_bytes > self.tail.capacity() {
                self.tail.try_reserve_exact(tail_bytes - self.tail.len())?;
            }
            if tail_bytes == 0 {
                self.tail.clear();
            } else {
                self.tail.resize(tail_bytes, 0);
                self.tail
                    .copy_from_slice(&self.pages[keep_pages][..tail_bytes]);
            }
            self.pages.truncate(keep_pages);
        } else {
            self.tail.truncate(tail_bytes);
        }
        self.prepared.clear();
        self.prepared_ready = false;
        self.len = checked_product(&[position, self.row_bytes])?;
        Ok(())
    }

    pub(super) fn bytes(&self) -> usize {
        size_of::<Self>()
            + self.pages.capacity() * size_of::<Arc<Vec<u8>>>()
            + self.tail.capacity()
            + self.prepared.capacity()
            + self
                .pages
                .iter()
                .map(|page| page.capacity() + size_of::<Vec<u8>>() + 2 * size_of::<usize>())
                .sum::<usize>()
    }
}

impl KvCache {
    pub(super) fn invalidate_prepared(&mut self) {
        if let Some(packed) = &mut self.packed {
            packed.prepared_ready = false;
        }
    }

    // Only ordinary attention in this evaluation attempt can establish this
    // private staged row. Narrowed KV-only layers retain normal preparation.
    pub(super) fn prepare_staged_append(&mut self, values: &[f32]) -> Result<()> {
        if let Some(packed) = &mut self.packed
            && packed.prepared_ready
        {
            return packed.reserve();
        }
        self.prepare_append(values)
    }

    pub(super) fn with_policy(
        width: usize,
        context: usize,
        policy: KvCachePolicy,
        key: bool,
        scalar: bool,
    ) -> Result<Self> {
        let Some((key_bits, value_bits)) = policy.bits() else {
            return Self::new(width, context);
        };
        let mut cache = Self::new(width, context)?;
        let mut packed = PackedCache::new(
            width,
            context,
            if key { key_bits } else { value_bits },
            if key { KEY_SEED } else { VALUE_SEED },
            scalar,
        )?;
        match policy {
            KvCachePolicy::Rotated4FittedValues if !key => {
                packed.quantizer = packed.quantizer.with_fitted_scale();
            }
            KvCachePolicy::Rotated4FittedKv => {
                packed.quantizer = packed.quantizer.with_cooptimized_scale();
            }
            _ => {}
        }
        cache.packed = Some(Box::new(packed));
        Ok(cache)
    }

    pub(super) fn prepare_append(&mut self, values: &[f32]) -> Result<()> {
        if let Some(packed) = &mut self.packed {
            packed.prepare(values)
        } else {
            self.reserve_append()
        }
    }

    pub(super) fn observe_packed(
        &self,
        mut observer: impl FnMut(usize, &[f32]) -> Result<()>,
    ) -> Result<()> {
        let packed = self
            .packed
            .as_ref()
            .ok_or_else(|| invalid("missing packed KV"))?;
        let mut row = super::zeros(self.width)?;
        for (position, bytes) in packed.rows().enumerate() {
            for head in 0..packed.heads {
                let mut decoded = [0.0; HEAD_DIM];
                packed
                    .quantizer
                    .decode(packed.record(bytes, head), &mut decoded)?;
                row[head * HEAD_DIM..(head + 1) * HEAD_DIM].copy_from_slice(&decoded);
            }
            observer(position, &row)?;
        }
        Ok(())
    }
}

// Match the balanced contiguous head partitions. Each worker consumes its
// heads sequentially, so their probability rows have disjoint lifetimes.
pub(super) fn score_rows(heads: usize, parallelism: &RowParallelism) -> usize {
    heads.min(parallelism.workers().max(1))
}

const fn grouped_head_unit(heads_per_kv: usize, heads: usize) -> usize {
    if heads_per_kv.is_multiple_of(6) && heads.is_multiple_of(6) {
        6
    } else if heads_per_kv.is_multiple_of(4) && heads.is_multiple_of(4) {
        4
    } else if heads_per_kv.is_multiple_of(3) && heads.is_multiple_of(3) {
        3
    } else if heads_per_kv.is_multiple_of(2) && heads.is_multiple_of(2) {
        2
    } else {
        1
    }
}

#[cfg(test)]
const fn grouped_unit_with_capacity(heads_per_kv: usize, heads: usize, max_rows: usize) -> usize {
    if max_rows >= 6 && heads_per_kv.is_multiple_of(6) && heads.is_multiple_of(6) {
        6
    } else if max_rows >= 4 && heads_per_kv.is_multiple_of(4) && heads.is_multiple_of(4) {
        4
    } else if max_rows >= 3 && heads_per_kv.is_multiple_of(3) && heads.is_multiple_of(3) {
        3
    } else if max_rows >= 2 && heads_per_kv.is_multiple_of(2) && heads.is_multiple_of(2) {
        2
    } else {
        1
    }
}

#[derive(Clone, Copy)]
struct HeadPartition {
    base_head: usize,
    chunk_heads: usize,
    unit: usize,
}

fn partition_heads(
    heads: usize,
    kv_heads: usize,
    workers: usize,
) -> ([HeadPartition; MAX_ROW_WORKERS], usize) {
    let mut partitions = [HeadPartition {
        base_head: 0,
        chunk_heads: 0,
        unit: 1,
    }; MAX_ROW_WORKERS];
    if heads == 0 {
        return (partitions, 0);
    }
    let task_count = heads.min(workers.max(1)).min(MAX_ROW_WORKERS);
    if kv_heads == 0 || !heads.is_multiple_of(kv_heads) {
        let base = heads / task_count;
        let rem = heads % task_count;
        let mut cursor = 0;
        for (index, slot) in partitions[..task_count].iter_mut().enumerate() {
            let chunk_heads = base + usize::from(index < rem);
            *slot = HeadPartition {
                base_head: cursor,
                chunk_heads,
                unit: 1,
            };
            cursor += chunk_heads;
        }
        return (partitions, task_count);
    }
    let heads_per_kv = heads / kv_heads;
    if task_count <= kv_heads {
        let kv_base = kv_heads / task_count;
        let kv_rem = kv_heads % task_count;
        let kv_unit = grouped_head_unit(heads_per_kv, heads_per_kv);
        let mut cursor = 0;
        for (index, slot) in partitions[..task_count].iter_mut().enumerate() {
            let chunk_kv = kv_base + usize::from(index < kv_rem);
            let chunk_heads = chunk_kv * heads_per_kv;
            *slot = HeadPartition {
                base_head: cursor,
                chunk_heads,
                unit: kv_unit,
            };
            cursor += chunk_heads;
        }
        return (partitions, task_count);
    }
    // Distribute workers evenly across KV heads, then split each KV head's
    // query heads by quotient+remainder so no worker spans a KV boundary.
    let w_base = task_count / kv_heads;
    let w_rem = task_count % kv_heads;
    let mut task_idx = 0;
    let mut cursor = 0;
    for kv in 0..kv_heads {
        let w_k = w_base + usize::from(kv < w_rem);
        let h_base = heads_per_kv / w_k;
        let h_rem = heads_per_kv % w_k;
        for sub in 0..w_k {
            let chunk_heads = h_base + usize::from(sub < h_rem);
            partitions[task_idx] = HeadPartition {
                base_head: cursor,
                chunk_heads,
                unit: grouped_head_unit(heads_per_kv, chunk_heads),
            };
            cursor += chunk_heads;
            task_idx += 1;
        }
    }
    (partitions, task_idx)
}

pub(super) fn group_score_rows(
    keys: &KvCache,
    values: &KvCache,
    heads: usize,
    parallelism: &RowParallelism,
) -> usize {
    let tasks = score_rows(heads, parallelism);
    let Some(kv_heads) = keys
        .packed
        .as_ref()
        .map(|cache| cache.heads)
        .or_else(|| values.packed.as_ref().map(|cache| cache.heads))
    else {
        return tasks;
    };
    if kv_heads == 0 || !heads.is_multiple_of(kv_heads) {
        return tasks;
    }
    let (partitions, count) = partition_heads(heads, kv_heads, parallelism.workers());
    partitions[..count]
        .iter()
        .map(|part| part.unit)
        .fold(0_usize, usize::saturating_add)
}

struct AttentionTask<'a> {
    base_head: usize,
    unit: usize,
    attended: &'a mut [f32],
    scores: &'a mut [f32],
}

// Stage exactly the bytes attention will read and a successful token will
// publish. Keep the original per-head key/value failure order. Tail/page
// reservation remains at the graph's later all-layer publication boundary.
fn prepare_current(
    keys: &mut PackedCache,
    values: &mut PackedCache,
    current_key: &[f32],
    current_value: &[f32],
) -> Result<()> {
    resize_bytes(keys.row_bytes, &mut keys.prepared)?;
    resize_bytes(values.row_bytes, &mut values.prepared)?;
    for ((key, output), (value, value_output)) in current_key
        .as_chunks::<HEAD_DIM>()
        .0
        .iter()
        .zip(keys.prepared.chunks_exact_mut(keys.quantizer.encoded_len()))
        .zip(
            current_value.as_chunks::<HEAD_DIM>().0.iter().zip(
                values
                    .prepared
                    .chunks_exact_mut(values.quantizer.encoded_len()),
            ),
        )
    {
        keys.quantizer.encode(key, output)?;
        values.quantizer.encode(value, value_output)?;
    }
    Ok(())
}

/// Caller-owned disjoint staging slices for fusing current-token KV encoding into head-parallel normalization.
pub(super) struct PackedPrepareTargets<'a> {
    pub(super) key_quantizer: &'a Quantizer,
    pub(super) value_quantizer: &'a Quantizer,
    pub(super) key_prepared: &'a mut [u8],
    pub(super) value_prepared: &'a mut [u8],
    pub(super) key_record_bytes: usize,
    pub(super) value_record_bytes: usize,
}

/// Resets `prepared_ready`, resizes current-token staging buffers, and returns disjoint borrows
/// so head-parallel normalization can encode K/V records without an extra barrier.
pub(super) fn begin_prepare_current<'a>(
    keys: &'a mut KvCache,
    values: &'a mut KvCache,
    kv_heads: usize,
) -> Result<Option<PackedPrepareTargets<'a>>> {
    let (Some(key_cache), Some(value_cache)) = (keys.packed.as_mut(), values.packed.as_mut())
    else {
        return Ok(None);
    };
    key_cache.prepared_ready = false;
    value_cache.prepared_ready = false;
    if kv_heads == 0
        || key_cache.heads != kv_heads
        || value_cache.heads != kv_heads
        || keys.width != kv_heads * HEAD_DIM
        || values.width != kv_heads * HEAD_DIM
        || keys.len != values.len
    {
        return Ok(None);
    }
    resize_bytes(key_cache.row_bytes, &mut key_cache.prepared)?;
    resize_bytes(value_cache.row_bytes, &mut value_cache.prepared)?;
    let key_record_bytes = key_cache.quantizer.encoded_len();
    let value_record_bytes = value_cache.quantizer.encoded_len();
    Ok(Some(PackedPrepareTargets {
        key_quantizer: &key_cache.quantizer,
        value_quantizer: &value_cache.quantizer,
        key_prepared: &mut key_cache.prepared,
        value_prepared: &mut value_cache.prepared,
        key_record_bytes,
        value_record_bytes,
    }))
}

// Tile attention consumes the committed form of its current row, which must
// have exactly the same interleaved per-head encoding as ordinary attention.
// Reserve both append destinations before publishing either row.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "Retained serial packed-KV append reference tested directly alongside parallel tile append."
    )
)]
pub(super) fn prepare_tile_append(
    keys: &mut KvCache,
    values: &mut KvCache,
    current_key: &[f32],
    current_value: &[f32],
) -> Result<()> {
    let key_cache = keys
        .packed
        .as_mut()
        .ok_or_else(|| invalid("missing packed tile keys"))?;
    let value_cache = values
        .packed
        .as_mut()
        .ok_or_else(|| invalid("missing packed tile values"))?;
    key_cache.prepared_ready = false;
    value_cache.prepared_ready = false;
    if current_key.len() != keys.width
        || current_value.len() != values.width
        || key_cache.heads != value_cache.heads
        || keys.len != values.len
    {
        return Err(invalid("invalid packed tile append geometry"));
    }
    prepare_current(key_cache, value_cache, current_key, current_value)?;
    key_cache.reserve()?;
    value_cache.reserve()
}

struct EncodeRowTask<'a> {
    quantizer: &'a Quantizer,
    src: &'a [f32],
    dst: &'a mut [u8],
}

/// Encode and commit a tile of rotated K/V rows in parallel across `parallelism`
/// while preserving exact token-order pages and deterministic failure priority.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "Wired from prefill tile execution during integration and exercised in tests."
    )
)]
pub(super) fn prepare_tile_append_parallel(
    keys: &mut KvCache,
    values: &mut KvCache,
    current_keys: &[Vec<f32>],
    current_values: &[Vec<f32>],
    parallelism: &RowParallelism,
) -> Result<()> {
    let mut scratch = PackedTileScratch::default();
    prepare_tile_append_parallel_with_scratch(
        keys,
        values,
        current_keys,
        current_values,
        &mut scratch,
        parallelism,
    )
}

pub(super) fn prepare_tile_append_parallel_with_scratch(
    keys: &mut KvCache,
    values: &mut KvCache,
    current_keys: &[Vec<f32>],
    current_values: &[Vec<f32>],
    scratch: &mut PackedTileScratch,
    parallelism: &RowParallelism,
) -> Result<()> {
    let key_cache = keys
        .packed
        .as_mut()
        .ok_or_else(|| invalid("missing packed tile keys"))?;
    let value_cache = values
        .packed
        .as_mut()
        .ok_or_else(|| invalid("missing packed tile values"))?;
    key_cache.prepared_ready = false;
    value_cache.prepared_ready = false;
    let count = current_keys.len();
    if count == 0
        || count > crate::MAX_PREFILL_BLOCK_TOKENS
        || current_values.len() != count
        || key_cache.heads == 0
        || key_cache.heads != value_cache.heads
        || keys.width != key_cache.heads * HEAD_DIM
        || values.width != value_cache.heads * HEAD_DIM
        || keys.len != values.len
        || !keys.len.is_multiple_of(keys.width)
        || !key_cache.len.is_multiple_of(key_cache.row_bytes)
        || !value_cache.len.is_multiple_of(value_cache.row_bytes)
        || key_cache.len / key_cache.row_bytes != value_cache.len / value_cache.row_bytes
        || key_cache.len / key_cache.row_bytes != keys.len / keys.width
        || current_keys.iter().any(|row| row.len() != keys.width)
        || current_values.iter().any(|row| row.len() != values.width)
    {
        return Err(invalid("invalid packed tile append geometry"));
    }
    let add_width = checked_product(&[count, keys.width])?;
    if checked_sum(keys.len, add_width)? > keys.maximum
        || checked_sum(values.len, add_width)? > values.maximum
    {
        return Err(invalid("packed KV append exceeds context bound"));
    }
    let key_res = key_cache.reserve_batch(count)?;
    let val_res = value_cache.reserve_batch(count)?;
    let total_key_bytes = checked_product(&[count, key_cache.row_bytes])?;
    let total_val_bytes = checked_product(&[count, value_cache.row_bytes])?;
    resize_bytes(total_key_bytes, &mut scratch.encoded_keys)?;
    resize_bytes(total_val_bytes, &mut scratch.encoded_values)?;

    encode_packed_kv_rows_parallel(
        key_cache,
        value_cache,
        current_keys,
        current_values,
        &mut scratch.encoded_keys[..total_key_bytes],
        &mut scratch.encoded_values[..total_val_bytes],
        parallelism,
    )?;

    key_cache.commit_batch(&scratch.encoded_keys[..total_key_bytes], key_res);
    value_cache.commit_batch(&scratch.encoded_values[..total_val_bytes], val_res);
    keys.len += add_width;
    values.len += add_width;
    Ok(())
}

fn encode_packed_kv_rows_parallel(
    key_cache: &PackedCache,
    value_cache: &PackedCache,
    current_keys: &[Vec<f32>],
    current_values: &[Vec<f32>],
    encoded_keys: &mut [u8],
    encoded_values: &mut [u8],
    parallelism: &RowParallelism,
) -> Result<()> {
    let count = current_keys.len();
    let kv_heads = key_cache.heads;
    let key_enc = key_cache.quantizer.encoded_len();
    let val_enc = value_cache.quantizer.encoded_len();
    let total_pairs = checked_product(&[count, kv_heads])?;
    let workers = parallelism.workers().max(1);
    let key_quantizer = &key_cache.quantizer;
    let val_quantizer = &value_cache.quantizer;

    if workers <= 1 {
        for (token, (k_row, v_row)) in current_keys.iter().zip(current_values).enumerate() {
            let k_base = token * key_cache.row_bytes;
            let v_base = token * value_cache.row_bytes;
            for ((k_src, k_dst), (v_src, v_dst)) in k_row
                .as_chunks::<HEAD_DIM>()
                .0
                .iter()
                .zip(encoded_keys[k_base..k_base + key_cache.row_bytes].chunks_exact_mut(key_enc))
                .zip(
                    v_row.as_chunks::<HEAD_DIM>().0.iter().zip(
                        encoded_values[v_base..v_base + value_cache.row_bytes]
                            .chunks_exact_mut(val_enc),
                    ),
                )
            {
                key_quantizer.encode(k_src, k_dst)?;
                val_quantizer.encode(v_src, v_dst)?;
            }
        }
        return Ok(());
    }
    if total_pairs >= workers {
        return parallelism.update_paired_chunks(
            encoded_keys,
            key_enc,
            encoded_values,
            val_enc,
            |base_unit, k_chunk, v_chunk| {
                for (local, (k_dst, v_dst)) in k_chunk
                    .chunks_exact_mut(key_enc)
                    .zip(v_chunk.chunks_exact_mut(val_enc))
                    .enumerate()
                {
                    let unit = base_unit + local;
                    let token = unit / kv_heads;
                    let head = unit % kv_heads;
                    let src_range = head * HEAD_DIM..(head + 1) * HEAD_DIM;
                    key_quantizer.encode(&current_keys[token][src_range.clone()], k_dst)?;
                    val_quantizer.encode(&current_values[token][src_range], v_dst)?;
                }
                Ok(())
            },
        );
    }
    let total_tasks = 2 * total_pairs;
    let mut slots: [Option<EncodeRowTask<'_>>; 2 * MAX_ROW_WORKERS] = std::array::from_fn(|_| None);
    let mut k_rem = encoded_keys;
    let mut v_rem = encoded_values;
    let mut slot_idx = 0;
    for (k_row, v_row) in current_keys.iter().zip(current_values) {
        for head in 0..kv_heads {
            let src_range = head * HEAD_DIM..(head + 1) * HEAD_DIM;
            let (k_dst, k_tail) = std::mem::take(&mut k_rem).split_at_mut(key_enc);
            k_rem = k_tail;
            let (v_dst, v_tail) = std::mem::take(&mut v_rem).split_at_mut(val_enc);
            v_rem = v_tail;
            slots[slot_idx] = Some(EncodeRowTask {
                quantizer: key_quantizer,
                src: &k_row[src_range.clone()],
                dst: k_dst,
            });
            slots[slot_idx + 1] = Some(EncodeRowTask {
                quantizer: val_quantizer,
                src: &v_row[src_range],
                dst: v_dst,
            });
            slot_idx += 2;
        }
    }
    parallelism.update_chunks(&mut slots[..total_tasks], 1, |_, chunk| {
        for task in chunk.iter_mut().flatten() {
            task.quantizer.encode(task.src, task.dst)?;
        }
        Ok(())
    })
}

/// Stream packed history directly: query rotation once, one score row per task,
/// and one bounded rotated value accumulator per active worker. Independent
/// query heads use the existing private row pool, with each head retaining its
/// original score, softmax, value and inverse-transform order. Dispatch joins
/// every worker before returning the earliest head-chunk failure; only the
/// caller's provisional scratch can change. Cancellation retains the ordinary
/// graph's synchronous-token boundary. Never expands a context-sized FP16/F32
/// key or value buffer or creates a second worker pool.
#[cfg(test)]
#[expect(
    clippy::too_many_arguments,
    reason = "Preserve legacy single-row score layout contract for existing unit tests."
)]
pub(super) fn attend(
    keys: &mut KvCache,
    values: &mut KvCache,
    queries: &[f32],
    projected: &[f32],
    current_key: &[f32],
    current_value: &[f32],
    attended: &mut [f32],
    scores: &mut [f32],
    heads: usize,
    scale: f32,
    arithmetic: super::Arithmetic,
    parallelism: &RowParallelism,
) -> Result<()> {
    attend_with_layout(
        keys,
        values,
        queries,
        projected,
        current_key,
        current_value,
        attended,
        scores,
        heads,
        scale,
        arithmetic,
        parallelism,
        false,
        false,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "Borrow immutable history and provisional current projections separately."
)]
pub(super) fn attend_with_group_scores(
    keys: &mut KvCache,
    values: &mut KvCache,
    queries: &[f32],
    projected: &[f32],
    current_key: &[f32],
    current_value: &[f32],
    attended: &mut [f32],
    scores: &mut [f32],
    heads: usize,
    scale: f32,
    arithmetic: super::Arithmetic,
    parallelism: &RowParallelism,
) -> Result<()> {
    attend_with_layout(
        keys,
        values,
        queries,
        projected,
        current_key,
        current_value,
        attended,
        scores,
        heads,
        scale,
        arithmetic,
        parallelism,
        true,
        false,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "Matches attend_with_group_scores while skipping redundant current-row encoding when pre-encoded in head-parallel prep."
)]
pub(super) fn attend_with_preencoded_group_scores(
    keys: &mut KvCache,
    values: &mut KvCache,
    queries: &[f32],
    projected: &[f32],
    current_key: &[f32],
    current_value: &[f32],
    attended: &mut [f32],
    scores: &mut [f32],
    heads: usize,
    scale: f32,
    arithmetic: super::Arithmetic,
    parallelism: &RowParallelism,
) -> Result<()> {
    attend_with_layout(
        keys,
        values,
        queries,
        projected,
        current_key,
        current_value,
        attended,
        scores,
        heads,
        scale,
        arithmetic,
        parallelism,
        true,
        true,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "Share decode packed-KV validation and worker partitioning between single-row and group score layouts."
)]
fn attend_with_layout(
    keys: &mut KvCache,
    values: &mut KvCache,
    queries: &[f32],
    projected: &[f32],
    current_key: &[f32],
    current_value: &[f32],
    attended: &mut [f32],
    scores: &mut [f32],
    heads: usize,
    scale: f32,
    arithmetic: super::Arithmetic,
    parallelism: &RowParallelism,
    grouped_scores: bool,
    pre_encoded_current: bool,
) -> Result<()> {
    let key_cache = keys
        .packed
        .as_mut()
        .ok_or_else(|| invalid("missing packed keys"))?;
    let value_cache = values
        .packed
        .as_mut()
        .ok_or_else(|| invalid("missing packed values"))?;
    key_cache.prepared_ready = false;
    value_cache.prepared_ready = false;
    let tokens = keys.len / keys.width;
    let stride = checked_sum(tokens, 1)?;
    if heads == 0
        || key_cache.heads == 0
        || !heads.is_multiple_of(key_cache.heads)
        || key_cache.heads != value_cache.heads
        || keys.len != values.len
        || queries.len() != heads * HEAD_DIM
        || attended.len() != queries.len()
        || projected.len() != 2 * queries.len()
        || current_key.len() != keys.width
        || current_value.len() != values.width
    {
        return Err(invalid("invalid packed attention geometry"));
    }
    let heads_per_kv = heads / key_cache.heads;
    let (partitions, task_count) = partition_heads(heads, key_cache.heads, parallelism.workers());
    let expected_score_rows = if grouped_scores {
        partitions[..task_count]
            .iter()
            .map(|part| part.unit)
            .fold(0_usize, usize::saturating_add)
    } else {
        task_count
    };
    if scores.len() != checked_product(&[expected_score_rows, stride])? {
        return Err(invalid("invalid packed attention geometry"));
    }
    if pre_encoded_current {
        if key_cache.prepared.len() != key_cache.row_bytes
            || value_cache.prepared.len() != value_cache.row_bytes
        {
            return Err(invalid("missing pre-encoded packed current KV row"));
        }
    } else {
        prepare_current(key_cache, value_cache, current_key, current_value)?;
    }
    let input = PackedAttention {
        keys: key_cache,
        values: value_cache,
        queries,
        projected,
        current_keys: &key_cache.prepared,
        current_values: &value_cache.prepared,
        history_tokens: tokens,
        heads_per_kv,
        scale,
        arithmetic,
    };
    dispatch_attend_tasks(
        &input,
        attended,
        scores,
        stride,
        &partitions[..task_count],
        grouped_scores,
        parallelism,
    )?;
    key_cache.prepared_ready = true;
    value_cache.prepared_ready = true;
    Ok(())
}

fn dispatch_attend_tasks(
    input: &PackedAttention<'_>,
    attended: &mut [f32],
    scores: &mut [f32],
    stride: usize,
    partitions: &[HeadPartition],
    grouped_scores: bool,
    parallelism: &RowParallelism,
) -> Result<()> {
    if partitions.len() == 1 {
        let part = partitions[0];
        if grouped_scores {
            return input.attend_chunk_with_scores(
                part.base_head,
                attended,
                scores,
                stride,
                part.unit,
            );
        }
        return input.attend_chunk_stack_buffered(
            part.base_head,
            attended,
            scores,
            stride,
            part.unit,
        );
    }
    // Keep the bounded borrowed task descriptors on the stack. This preserves
    // balanced worker partitions and earliest-chunk failure order without
    // allocating a task vector or extra score matrices across worker boundaries.
    let mut tasks: [Option<AttentionTask<'_>>; MAX_ROW_WORKERS] = std::array::from_fn(|_| None);
    let mut rem_attended = attended;
    let mut rem_scores = scores;
    for (part, task) in partitions.iter().zip(&mut tasks) {
        let (att_chunk, att_tail) = rem_attended.split_at_mut(part.chunk_heads * HEAD_DIM);
        rem_attended = att_tail;
        let score_rows_for_task = if grouped_scores { part.unit } else { 1 };
        let (score_chunk, score_tail) = rem_scores.split_at_mut(score_rows_for_task * stride);
        rem_scores = score_tail;
        *task = Some(AttentionTask {
            base_head: part.base_head,
            unit: part.unit,
            attended: att_chunk,
            scores: score_chunk,
        });
    }
    parallelism.update_chunks(&mut tasks[..partitions.len()], 1, |_, tasks| {
        for task in tasks.iter_mut().flatten() {
            if grouped_scores {
                input.attend_chunk_with_scores(
                    task.base_head,
                    task.attended,
                    task.scores,
                    stride,
                    task.unit,
                )?;
            } else {
                input.attend_chunk_stack_buffered(
                    task.base_head,
                    task.attended,
                    task.scores,
                    stride,
                    task.unit,
                )?;
            }
        }
        Ok(())
    })
}

struct TileContext<'a> {
    keys: &'a PackedCache,
    values: &'a PackedCache,
    queries: &'a [Vec<f32>],
    projected: &'a [Vec<f32>],
    base: usize,
    heads_per_kv: usize,
    scale: f32,
    arithmetic: super::Arithmetic,
}

impl<'a> TileContext<'a> {
    fn make_input(&self, token: usize) -> Result<PackedAttention<'a>> {
        let position = self.base + token;
        Ok(PackedAttention {
            keys: self.keys,
            values: self.values,
            queries: &self.queries[token],
            projected: &self.projected[token],
            current_keys: self.keys.row(position)?,
            current_values: self.values.row(position)?,
            history_tokens: position,
            heads_per_kv: self.heads_per_kv,
            scale: self.scale,
            arithmetic: self.arithmetic,
        })
    }

    fn run_single_kv<const G: usize>(
        &self,
        token: usize,
        kv_head: usize,
        output: &mut [f32],
        score_scratch: &mut [f32],
    ) -> std::result::Result<(), (usize, usize, crate::Error)> {
        let input = self
            .make_input(token)
            .map_err(|err| (token, kv_head, err))?;
        let stride = input.history_tokens + 1;
        let base_kv_head = kv_head * self.heads_per_kv;
        let group_elems = G * HEAD_DIM;
        for (group_idx, out_group) in output.chunks_exact_mut(group_elems).enumerate() {
            let group_head = base_kv_head + group_idx * G;
            let res = if G == 1 {
                input.attend_head(group_head, out_group, &mut score_scratch[..stride])
            } else {
                input.attend_chunk_fixed_scores::<G>(
                    group_head,
                    out_group,
                    &mut score_scratch[..G * stride],
                    stride,
                )
            };
            if let Err(err) = res {
                return Err((token, kv_head, err));
            }
        }
        Ok(())
    }

    fn run_pair_kv<const G: usize>(
        &self,
        token0: usize,
        kv_head: usize,
        output: &mut [f32],
        score_scratch: &mut [f32],
    ) -> std::result::Result<(), (usize, usize, crate::Error)> {
        let token1 = token0 + 1;
        let kv_out_len = self.heads_per_kv * HEAD_DIM;
        let (out0, out1) = output.split_at_mut(kv_out_len);
        let pair_res = (|| -> Result<()> {
            let input0 = self.make_input(token0)?;
            let input1 = self.make_input(token1)?;
            let stride0 = input0.history_tokens + 1;
            let stride1 = input1.history_tokens + 1;
            let needed0 = checked_product(&[G, stride0])?;
            let needed1 = checked_product(&[G, stride1])?;
            if score_scratch.len() < checked_sum(needed0, needed1)? {
                return Err(invalid("grouped attention slice length underflow"));
            }
            let (scores0_flat, tail) = score_scratch.split_at_mut(needed0);
            let scores1_flat = &mut tail[..needed1];
            let mut score_rows0 = split_mut_slices::<G>(scores0_flat, stride0)?;
            let mut score_rows1 = split_mut_slices::<G>(scores1_flat, stride1)?;
            let base_kv_head = kv_head * self.heads_per_kv;
            let group_elems = G * HEAD_DIM;
            for (group_idx, (out0_group, out1_group)) in out0
                .chunks_exact_mut(group_elems)
                .zip(out1.chunks_exact_mut(group_elems))
                .enumerate()
            {
                let mut head_outputs0 = split_mut_slices::<G>(out0_group, HEAD_DIM)?;
                let mut head_outputs1 = split_mut_slices::<G>(out1_group, HEAD_DIM)?;
                attend_group_pair::<G>(
                    [&input0, &input1],
                    base_kv_head + group_idx * G,
                    [&mut head_outputs0, &mut head_outputs1],
                    [&mut score_rows0, &mut score_rows1],
                )?;
            }
            Ok(())
        })();
        match pair_res {
            Ok(()) => Ok(()),
            Err(pair_err) => {
                // Re-run tokens singly so an earlier failure in token0 cannot be
                // shadowed by a later-token failure within the shared pair.
                self.run_single_kv::<G>(token0, kv_head, out0, score_scratch)?;
                self.run_single_kv::<G>(token1, kv_head, out1, score_scratch)?;
                Err((token0, kv_head, pair_err))
            }
        }
    }

    fn run_item_fixed<const G: usize>(
        &self,
        item: ScheduledWorkItem,
        output: &mut [f32],
        score_scratch: &mut [f32],
    ) -> std::result::Result<(), (usize, usize, crate::Error)> {
        if item.token_len == 2 {
            self.run_pair_kv::<G>(item.token_start, item.kv_head, output, score_scratch)
        } else {
            self.run_single_kv::<G>(item.token_start, item.kv_head, output, score_scratch)
        }
    }

    fn run_item(
        &self,
        item: ScheduledWorkItem,
        output: &mut [f32],
        score_scratch: &mut [f32],
        unit: usize,
    ) -> std::result::Result<(), (usize, usize, crate::Error)> {
        match unit {
            6 => self.run_item_fixed::<6>(item, output, score_scratch),
            4 => self.run_item_fixed::<4>(item, output, score_scratch),
            3 => self.run_item_fixed::<3>(item, output, score_scratch),
            2 => self.run_item_fixed::<2>(item, output, score_scratch),
            _ => self.run_item_fixed::<1>(item, output, score_scratch),
        }
    }
}

struct TileWorkerTask<'a> {
    items: &'a [ScheduledWorkItem],
    outputs: &'a mut [f32],
    scores: &'a mut [f32],
    error: Option<(usize, usize, crate::Error)>,
}

fn reserve_usize_vec(vec: &mut Vec<usize>, needed: usize) -> Result<()> {
    vec.clear();
    if needed > vec.capacity() {
        vec.try_reserve_exact(needed - vec.capacity())?;
    }
    Ok(())
}

pub(super) fn assign_causal_balanced_items_with_order(
    costs: &[usize],
    token_starts: &[usize],
    kv_heads_idx: &[usize],
    workers: usize,
    assigned: &mut [usize],
    counts: &mut [usize; MAX_ROW_WORKERS],
    order: &mut Vec<usize>,
) -> Result<(usize, [usize; MAX_ROW_WORKERS])> {
    let active_workers = costs.len().min(workers.max(1)).min(MAX_ROW_WORKERS);
    counts.fill(0);
    let mut loads = [0_usize; MAX_ROW_WORKERS];
    if active_workers == 0 {
        return Ok((0, loads));
    }
    reserve_usize_vec(order, costs.len())?;
    order.extend(0..costs.len());
    order.sort_by(|&lhs, &rhs| {
        costs[rhs]
            .cmp(&costs[lhs])
            .then_with(|| token_starts[lhs].cmp(&token_starts[rhs]))
            .then_with(|| kv_heads_idx[lhs].cmp(&kv_heads_idx[rhs]))
    });
    for &idx in order.iter() {
        let mut best_worker = 0_usize;
        let mut best_load = loads[0];
        for (worker, &load) in loads[..active_workers].iter().enumerate().skip(1) {
            if load < best_load {
                best_load = load;
                best_worker = worker;
            }
        }
        loads[best_worker] = loads[best_worker].saturating_add(costs[idx]);
        counts[best_worker] += 1;
        assigned[idx] = best_worker;
    }
    Ok((active_workers, loads))
}

#[cfg(test)]
fn assign_causal_balanced_items(
    costs: &[usize],
    token_starts: &[usize],
    kv_heads_idx: &[usize],
    workers: usize,
    assigned: &mut [usize],
    counts: &mut [usize; MAX_ROW_WORKERS],
) -> Result<(usize, [usize; MAX_ROW_WORKERS])> {
    let mut order = Vec::new();
    assign_causal_balanced_items_with_order(
        costs,
        token_starts,
        kv_heads_idx,
        workers,
        assigned,
        counts,
        &mut order,
    )
}

pub(super) struct BalancedScheduleSummary {
    pub(super) used_workers: usize,
    pub(super) counts: [usize; MAX_ROW_WORKERS],
    pub(super) worker_out_elems: [usize; MAX_ROW_WORKERS],
}

pub(super) fn prepare_causal_balanced_schedule(
    schedule: &mut CausalScheduleScratch,
    count: usize,
    first_position: usize,
    kv_heads: usize,
    group_out_stride: usize,
    active_workers: usize,
) -> Result<BalancedScheduleSummary> {
    let num_token_units = count.div_ceil(2);
    let total_items = checked_product(&[num_token_units, kv_heads])?;
    reserve_usize_vec(&mut schedule.costs, total_items)?;
    reserve_usize_vec(&mut schedule.token_starts, total_items)?;
    reserve_usize_vec(&mut schedule.kv_heads_idx, total_items)?;
    reserve_usize_vec(&mut schedule.assigned, total_items)?;
    schedule.assigned.resize(total_items, 0);
    reserve_usize_vec(&mut schedule.item_offsets, total_items)?;
    schedule.item_offsets.resize(total_items, 0);
    schedule.ordered.clear();
    if total_items > schedule.ordered.capacity() {
        schedule
            .ordered
            .try_reserve_exact(total_items - schedule.ordered.capacity())?;
    }
    schedule
        .ordered
        .resize(total_items, ScheduledWorkItem::default());

    for u in 0..num_token_units {
        let token_start = u * 2;
        let token_len = (count - token_start).min(2);
        let causal_cost = if token_len == 2 {
            2_usize
                .saturating_mul(first_position + token_start)
                .saturating_add(3)
        } else {
            (first_position + token_start).saturating_add(1)
        };
        for kv_head in 0..kv_heads {
            schedule.costs.push(causal_cost);
            schedule.token_starts.push(token_start);
            schedule.kv_heads_idx.push(kv_head);
        }
    }

    let mut counts = [0_usize; MAX_ROW_WORKERS];
    let (used_workers, _) = assign_causal_balanced_items_with_order(
        &schedule.costs,
        &schedule.token_starts,
        &schedule.kv_heads_idx,
        active_workers,
        &mut schedule.assigned,
        &mut counts,
        &mut schedule.order,
    )?;

    let mut worker_out_elems = [0_usize; MAX_ROW_WORKERS];
    for (&worker, &token_start) in schedule.assigned.iter().zip(&schedule.token_starts) {
        let token_len = (count - token_start).min(2);
        worker_out_elems[worker] += token_len * group_out_stride;
    }

    let mut item_cursor = [0_usize; MAX_ROW_WORKERS];
    let mut out_cursor = [0_usize; MAX_ROW_WORKERS];
    let mut running_items = 0_usize;
    let mut running_out = 0_usize;
    for worker in 0..used_workers {
        item_cursor[worker] = running_items;
        running_items += counts[worker];
        out_cursor[worker] = running_out;
        running_out += worker_out_elems[worker];
    }

    for (idx, ((&worker, &token_start), &kv_head)) in schedule
        .assigned
        .iter()
        .zip(&schedule.token_starts)
        .zip(&schedule.kv_heads_idx)
        .enumerate()
    {
        let token_len = (count - token_start).min(2);
        let out_len = token_len * group_out_stride;
        let slot = item_cursor[worker];
        item_cursor[worker] += 1;
        let out_offset = out_cursor[worker];
        out_cursor[worker] += out_len;
        schedule.item_offsets[idx] = out_offset;
        schedule.ordered[slot] = ScheduledWorkItem {
            token_start,
            token_len,
            kv_head,
            out_len,
        };
    }

    Ok(BalancedScheduleSummary {
        used_workers,
        counts,
        worker_out_elems,
    })
}

/// Attend already encoded tile rows against their own causal prefixes using
/// 2D `(token_pair, kv_head)` causal-cost-balanced partitioning and per-worker
/// `2 * unit`-row score scratch.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "Retained convenience entry point tested alongside attend_tile_with_scratch."
    )
)]
#[expect(
    clippy::too_many_arguments,
    reason = "Keep causal history, immutable prepared inputs and bounded tile scratch explicit."
)]
pub(super) fn attend_tile(
    keys: &KvCache,
    values: &KvCache,
    queries: &[Vec<f32>],
    projected: &[Vec<f32>],
    attended: &mut [Vec<f32>],
    scores: &mut [f32],
    base: usize,
    heads: usize,
    scale: f32,
    arithmetic: super::Arithmetic,
    parallelism: &RowParallelism,
) -> Result<()> {
    let mut scratch = PackedTileScratch::default();
    attend_tile_with_scratch(
        keys,
        values,
        queries,
        projected,
        attended,
        scores,
        base,
        heads,
        scale,
        arithmetic,
        &mut scratch,
        parallelism,
    )
}

#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "Coordinates validation, 2D causal-balanced scheduling, worker execution, and scatter."
)]
pub(super) fn attend_tile_with_scratch(
    keys: &KvCache,
    values: &KvCache,
    queries: &[Vec<f32>],
    projected: &[Vec<f32>],
    attended: &mut [Vec<f32>],
    scores: &mut [f32],
    base: usize,
    heads: usize,
    scale: f32,
    arithmetic: super::Arithmetic,
    tile_scratch: &mut PackedTileScratch,
    parallelism: &RowParallelism,
) -> Result<()> {
    const MAX_WORKER_SCORE_SCRATCH_BYTES: usize = 16 * 1024 * 1024;
    let result = (|| {
        let keys = keys
            .packed
            .as_ref()
            .ok_or_else(|| invalid("missing packed tile keys"))?;
        let values = values
            .packed
            .as_ref()
            .ok_or_else(|| invalid("missing packed tile values"))?;
        let count = queries.len();
        let end = checked_sum(base, count)?;
        let width = checked_product(&[heads, HEAD_DIM])?;
        let projected_width = checked_product(&[2, width])?;
        if count == 0
            || count > crate::MAX_PREFILL_BLOCK_TOKENS
            || heads == 0
            || !heads.is_multiple_of(keys.heads)
            || keys.heads != values.heads
            || keys.len / keys.row_bytes != end
            || values.len / values.row_bytes != end
            || projected.len() != count
            || attended.len() != count
            || queries.iter().any(|row| row.len() != width)
            || projected.iter().any(|row| row.len() != projected_width)
            || attended.iter().any(|row| row.len() != width)
            || scores.len() != checked_product(&[count, end])?
        {
            return Err(invalid("invalid packed attention tile geometry"));
        }
        let kv_heads = keys.heads;
        let heads_per_kv = heads / kv_heads;
        let ctx = TileContext {
            keys,
            values,
            queries,
            projected,
            base,
            heads_per_kv,
            scale,
            arithmetic,
        };
        let num_token_units = count.div_ceil(2);
        let total_items = checked_product(&[num_token_units, kv_heads])?;
        let kv_out_len = checked_product(&[heads_per_kv, HEAD_DIM])?;
        let total_out_elems = checked_product(&[count, width])?;
        resize_f32s(total_out_elems, &mut tile_scratch.kv_major_outputs)?;

        let active_workers = total_items
            .min(parallelism.workers().max(1))
            .min(MAX_ROW_WORKERS);
        let pair_toks = if count >= 2 { 2 } else { 1 };
        let max_score_elems = MAX_WORKER_SCORE_SCRATCH_BYTES / size_of::<f32>();
        let row_denom = active_workers
            .saturating_mul(pair_toks)
            .saturating_mul(end)
            .max(1);
        let max_unit_rows = (max_score_elems / row_denom).clamp(1, heads_per_kv);
        let unit = grouped_head_unit(heads_per_kv, max_unit_rows);
        let score_rows_per_worker = checked_product(&[pair_toks, unit])?;
        let worker_score_len = checked_product(&[score_rows_per_worker, end])?;
        let reuse_caller_scores = scores.len() >= worker_score_len;
        let scratch_workers = if reuse_caller_scores {
            active_workers - 1
        } else {
            active_workers
        };
        resize_f32s(
            checked_product(&[scratch_workers, worker_score_len])?,
            &mut tile_scratch.worker_scores,
        )?;

        if active_workers == 1 {
            let mut rem_out = &mut tile_scratch.kv_major_outputs[..total_out_elems];
            let worker_scores = if reuse_caller_scores {
                &mut scores[..worker_score_len]
            } else {
                &mut tile_scratch.worker_scores[..worker_score_len]
            };
            let mut earliest: Option<(usize, usize, crate::Error)> = None;
            for u in 0..num_token_units {
                let token_start = u * 2;
                let token_len = (count - token_start).min(2);
                let chunk_len = token_len * kv_out_len;
                for kv_head in 0..kv_heads {
                    let (out_slice, tail) = std::mem::take(&mut rem_out).split_at_mut(chunk_len);
                    rem_out = tail;
                    let item = ScheduledWorkItem {
                        token_start,
                        token_len,
                        kv_head,
                        out_len: chunk_len,
                    };
                    if let Err(err) = ctx.run_item(item, out_slice, worker_scores, unit) {
                        let key = (err.0, err.1);
                        if earliest
                            .as_ref()
                            .is_none_or(|(tok, kh, _)| key < (*tok, *kh))
                        {
                            earliest = Some(err);
                        }
                    }
                }
            }
            if let Some((_, _, err)) = earliest {
                return Err(err);
            }
            let mut rem_out = &tile_scratch.kv_major_outputs[..total_out_elems];
            for u in 0..num_token_units {
                let token_start = u * 2;
                let token_len = (count - token_start).min(2);
                let chunk_len = token_len * kv_out_len;
                for kv_head in 0..kv_heads {
                    let (kv_block, tail) = rem_out.split_at(chunk_len);
                    rem_out = tail;
                    let dst_offset = kv_head * kv_out_len;
                    for local_tok in 0..token_len {
                        let src = &kv_block[local_tok * kv_out_len..(local_tok + 1) * kv_out_len];
                        attended[token_start + local_tok][dst_offset..dst_offset + kv_out_len]
                            .copy_from_slice(src);
                    }
                }
            }
        } else {
            let summary = prepare_causal_balanced_schedule(
                &mut tile_scratch.schedule,
                count,
                base,
                kv_heads,
                kv_out_len,
                active_workers,
            )?;
            let used_workers = summary.used_workers;
            let mut tasks: [Option<TileWorkerTask<'_>>; MAX_ROW_WORKERS] =
                std::array::from_fn(|_| None);
            let mut rem_ordered = &tile_scratch.schedule.ordered[..total_items];
            let mut rem_out = &mut tile_scratch.kv_major_outputs[..total_out_elems];
            let scratch_slice_workers = if reuse_caller_scores {
                used_workers - 1
            } else {
                used_workers
            };
            let mut rem_scores =
                &mut tile_scratch.worker_scores[..scratch_slice_workers * worker_score_len];
            let mut caller_scores_opt =
                reuse_caller_scores.then(|| &mut scores[..worker_score_len]);
            for (worker, task) in tasks[..used_workers].iter_mut().enumerate() {
                let (item_chunk, item_tail) = rem_ordered.split_at(summary.counts[worker]);
                rem_ordered = item_tail;
                let (out_chunk, out_tail) =
                    std::mem::take(&mut rem_out).split_at_mut(summary.worker_out_elems[worker]);
                rem_out = out_tail;
                let score_chunk = caller_scores_opt.take().map_or_else(
                    || {
                        let (chunk, score_tail) =
                            std::mem::take(&mut rem_scores).split_at_mut(worker_score_len);
                        rem_scores = score_tail;
                        chunk
                    },
                    |caller_chunk| caller_chunk,
                );
                *task = Some(TileWorkerTask {
                    items: item_chunk,
                    outputs: out_chunk,
                    scores: score_chunk,
                    error: None,
                });
            }

            parallelism.update_chunks(&mut tasks[..used_workers], 1, |_, worker_slice| {
                for task in worker_slice.iter_mut().flatten() {
                    let mut rem_task_out = std::mem::take(&mut task.outputs);
                    for &item in task.items {
                        let (out_slice, tail) =
                            std::mem::take(&mut rem_task_out).split_at_mut(item.out_len);
                        rem_task_out = tail;
                        if let Err(err) = ctx.run_item(item, out_slice, task.scores, unit) {
                            let key = (err.0, err.1);
                            if task
                                .error
                                .as_ref()
                                .is_none_or(|(tok, kh, _)| key < (*tok, *kh))
                            {
                                task.error = Some(err);
                            }
                        }
                    }
                }
                Ok(())
            })?;

            let earliest = tasks[..used_workers]
                .iter_mut()
                .filter_map(|task| task.as_mut().and_then(|t| t.error.take()))
                .min_by_key(|(tok, kh, _)| (*tok, *kh));
            if let Some((_, _, err)) = earliest {
                return Err(err);
            }

            let mut idx = 0_usize;
            for u in 0..num_token_units {
                let token_start = u * 2;
                let token_len = (count - token_start).min(2);
                let chunk_len = token_len * kv_out_len;
                for kv_head in 0..kv_heads {
                    let out_offset = tile_scratch.schedule.item_offsets[idx];
                    idx += 1;
                    let kv_block =
                        &tile_scratch.kv_major_outputs[out_offset..out_offset + chunk_len];
                    let dst_offset = kv_head * kv_out_len;
                    for local_tok in 0..token_len {
                        let src = &kv_block[local_tok * kv_out_len..(local_tok + 1) * kv_out_len];
                        attended[token_start + local_tok][dst_offset..dst_offset + kv_out_len]
                            .copy_from_slice(src);
                    }
                }
            }
        }
        Ok(())
    })();
    if result.is_err() {
        for row in attended {
            row.clear();
        }
    }
    result
}

struct PackedAttention<'a> {
    keys: &'a PackedCache,
    values: &'a PackedCache,
    queries: &'a [f32],
    projected: &'a [f32],
    current_keys: &'a [u8],
    current_values: &'a [u8],
    history_tokens: usize,
    heads_per_kv: usize,
    scale: f32,
    arithmetic: super::Arithmetic,
}

const STACK_GROUP_SCORES_LEN: usize = 4_096;

fn split_mut_slices<const G: usize>(slice: &mut [f32], stride: usize) -> Result<[&mut [f32]; G]> {
    if slice.len() < G * stride {
        return Err(invalid("grouped attention slice length underflow"));
    }
    let mut rest = slice;
    let mut out: [&mut [f32]; G] = std::array::from_fn(|_| &mut [][..]);
    for slot in &mut out {
        let (head, tail) = std::mem::take(&mut rest).split_at_mut(stride);
        *slot = head;
        rest = tail;
    }
    Ok(out)
}

impl PackedAttention<'_> {
    fn attend_chunk_with_scores(
        &self,
        base_head: usize,
        attended: &mut [f32],
        group_scores: &mut [f32],
        stride: usize,
        unit: usize,
    ) -> Result<()> {
        match unit {
            6 => self.attend_chunk_fixed_scores::<6>(base_head, attended, group_scores, stride),
            4 => self.attend_chunk_fixed_scores::<4>(base_head, attended, group_scores, stride),
            3 => self.attend_chunk_fixed_scores::<3>(base_head, attended, group_scores, stride),
            2 => self.attend_chunk_fixed_scores::<2>(base_head, attended, group_scores, stride),
            _ => {
                for (offset, result) in attended
                    .as_chunks_mut::<HEAD_DIM>()
                    .0
                    .iter_mut()
                    .enumerate()
                {
                    self.attend_head(base_head + offset, result, group_scores)?;
                }
                Ok(())
            }
        }
    }

    fn attend_chunk_fixed_scores<const G: usize>(
        &self,
        base_head: usize,
        attended: &mut [f32],
        group_scores: &mut [f32],
        stride: usize,
    ) -> Result<()> {
        let mut score_rows = split_mut_slices::<G>(group_scores, stride)?;
        self.attend_chunk_with_score_rows::<G>(base_head, attended, &mut score_rows)
    }

    fn attend_chunk_with_score_rows<const G: usize>(
        &self,
        base_head: usize,
        attended: &mut [f32],
        score_rows: &mut [&mut [f32]; G],
    ) -> Result<()> {
        let (groups, remainder) =
            attended.split_at_mut((attended.len() / (G * HEAD_DIM)) * G * HEAD_DIM);
        for (group_index, out_chunk) in groups.chunks_exact_mut(G * HEAD_DIM).enumerate() {
            let group_base = base_head + group_index * G;
            if group_base / self.heads_per_kv == (group_base + G - 1) / self.heads_per_kv {
                let mut results = split_mut_slices::<G>(out_chunk, HEAD_DIM)?;
                self.attend_group::<G>(group_base, &mut results, score_rows)?;
            } else {
                for (offset, result) in out_chunk
                    .as_chunks_mut::<HEAD_DIM>()
                    .0
                    .iter_mut()
                    .enumerate()
                {
                    self.attend_head(group_base + offset, result, score_rows[0])?;
                }
            }
        }
        let rem_base = base_head + (groups.len() / HEAD_DIM);
        for (offset, result) in remainder
            .as_chunks_mut::<HEAD_DIM>()
            .0
            .iter_mut()
            .enumerate()
        {
            self.attend_head(rem_base + offset, result, score_rows[0])?;
        }
        Ok(())
    }

    fn attend_chunk_stack_buffered(
        &self,
        base_head: usize,
        attended: &mut [f32],
        first_scores: &mut [f32],
        stride: usize,
        unit: usize,
    ) -> Result<()> {
        match unit {
            6 if 5 * stride <= STACK_GROUP_SCORES_LEN => {
                self.attend_chunk_stack_fixed::<6>(base_head, attended, first_scores, stride)
            }
            4 if 3 * stride <= STACK_GROUP_SCORES_LEN => {
                self.attend_chunk_stack_fixed::<4>(base_head, attended, first_scores, stride)
            }
            6 | 3 if 2 * stride <= STACK_GROUP_SCORES_LEN => {
                self.attend_chunk_stack_fixed::<3>(base_head, attended, first_scores, stride)
            }
            6 | 4 | 2 if stride <= STACK_GROUP_SCORES_LEN => {
                self.attend_chunk_stack_fixed::<2>(base_head, attended, first_scores, stride)
            }
            _ => {
                for (offset, result) in attended
                    .as_chunks_mut::<HEAD_DIM>()
                    .0
                    .iter_mut()
                    .enumerate()
                {
                    self.attend_head(base_head + offset, result, first_scores)?;
                }
                Ok(())
            }
        }
    }

    fn attend_chunk_stack_fixed<const G: usize>(
        &self,
        base_head: usize,
        attended: &mut [f32],
        first_scores: &mut [f32],
        stride: usize,
    ) -> Result<()> {
        let mut stack_buf = [0.0_f32; STACK_GROUP_SCORES_LEN];
        let mut rest = &mut stack_buf[..(G - 1) * stride];
        let mut score_rows: [&mut [f32]; G] = std::array::from_fn(|_| &mut [][..]);
        for slot in &mut score_rows[..G - 1] {
            let (head, tail) = std::mem::take(&mut rest).split_at_mut(stride);
            *slot = head;
            rest = tail;
        }
        score_rows[G - 1] = first_scores;
        self.attend_chunk_with_score_rows::<G>(base_head, attended, &mut score_rows)
    }

    fn attend_group<const G: usize>(
        &self,
        base_head: usize,
        results: &mut [&mut [f32]; G],
        score_rows: &mut [&mut [f32]; G],
    ) -> Result<()> {
        const PROB_PACK_CHUNK: usize = 64;
        let kv_head = base_head / self.heads_per_kv;
        let mut rotated_queries = [[0.0; HEAD_DIM]; G];
        for (offset, rotated) in rotated_queries.iter_mut().enumerate() {
            let head = base_head + offset;
            self.keys.quantizer.rotate(
                &self.queries[head * HEAD_DIM..(head + 1) * HEAD_DIM],
                rotated,
            )?;
        }
        let query_refs: [&[f32; HEAD_DIM]; G] =
            std::array::from_fn(|offset| &rotated_queries[offset]);
        let prepared_query = self.keys.quantizer.prepare_query_group(query_refs)?;
        let key_sealed_take = self.history_tokens.min(self.keys.sealed_rows());
        let key_tail_take = self.history_tokens - key_sealed_take;
        let mut token = 0_usize;
        for row in self.keys.page_rows().take(key_sealed_take) {
            let scores = prepared_query.score_group_trusted(self.keys.record(row, kv_head))?;
            for (probabilities, score) in score_rows.iter_mut().zip(scores) {
                probabilities[token] = score * self.scale;
            }
            token += 1;
        }
        for row in self
            .keys
            .tail_rows()
            .take(key_tail_take)
            .chain(std::iter::once(self.current_keys))
        {
            let scores = prepared_query.score_group(self.keys.record(row, kv_head))?;
            for (probabilities, score) in score_rows.iter_mut().zip(scores) {
                probabilities[token] = score * self.scale;
            }
            token += 1;
        }
        for probabilities in score_rows.iter_mut() {
            match self.arithmetic {
                super::Arithmetic::Source => crate::kernels::softmax_round_f16(probabilities)?,
                super::Arithmetic::PinnedSwift => {
                    crate::kernels::pinned_attention::softmax_rotated_kv(probabilities)?;
                }
            }
        }
        let total_tokens = self.history_tokens + 1;
        if score_rows.iter().any(|row| row.len() != total_tokens) {
            return Err(invalid("packed attention score row length mismatch"));
        }
        let mut accumulator = self.values.quantizer.value_group_accumulator::<G>()?;
        let mut packed_weights = [[0.0_f32; G]; PROB_PACK_CHUNK];
        let val_sealed_take = self.history_tokens.min(self.values.sealed_rows());
        let val_tail_take = self.history_tokens - val_sealed_take;
        let mut page_value_rows = self.values.page_rows().take(val_sealed_take);
        let mut tail_value_rows = self
            .values
            .tail_rows()
            .take(val_tail_take)
            .chain(std::iter::once(self.current_values));
        for chunk_start in (0..total_tokens).step_by(PROB_PACK_CHUNK) {
            let chunk_len = (total_tokens - chunk_start).min(PROB_PACK_CHUNK);
            for (head, probabilities) in score_rows.iter().enumerate() {
                for (slot, &weight) in packed_weights[..chunk_len]
                    .iter_mut()
                    .zip(&probabilities[chunk_start..chunk_start + chunk_len])
                {
                    slot[head] = weight;
                }
            }
            let sealed_in_chunk = val_sealed_take.saturating_sub(chunk_start).min(chunk_len);
            for &weights in &packed_weights[..sealed_in_chunk] {
                let row = page_value_rows
                    .next()
                    .ok_or_else(|| invalid("packed value cache row is short"))?;
                accumulator.accumulate_group_trusted(self.values.record(row, kv_head), weights)?;
            }
            for &weights in &packed_weights[sealed_in_chunk..chunk_len] {
                let row = tail_value_rows
                    .next()
                    .ok_or_else(|| invalid("packed value cache row is short"))?;
                accumulator.accumulate_group(self.values.record(row, kv_head), weights)?;
            }
        }
        let mut outputs = accumulator.finish()?;
        for (offset, (output, result)) in outputs.iter_mut().zip(results.iter_mut()).enumerate() {
            let head = base_head + offset;
            self.values.quantizer.inverse(output)?;
            super::mul_sigmoid_in_place(
                output,
                super::attention_gate(self.projected, head, HEAD_DIM),
            );
            super::finite(output, "packed attention output")?;
            result.copy_from_slice(output);
        }
        Ok(())
    }

    fn attend_head(
        &self,
        head: usize,
        result: &mut [f32],
        probabilities: &mut [f32],
    ) -> Result<()> {
        let kv_head = head / self.heads_per_kv;
        let mut query = [0.0; HEAD_DIM];
        self.keys.quantizer.rotate(
            &self.queries[head * HEAD_DIM..(head + 1) * HEAD_DIM],
            &mut query,
        )?;
        let prepared_query = self.keys.quantizer.prepare_query(&query)?;
        let key_sealed_take = self.history_tokens.min(self.keys.sealed_rows());
        let key_tail_take = self.history_tokens - key_sealed_take;
        let (prob_sealed, prob_tail) =
            probabilities.split_at_mut(key_sealed_take.min(probabilities.len()));
        for (score, row) in prob_sealed
            .iter_mut()
            .zip(self.keys.page_rows().take(key_sealed_take))
        {
            *score = prepared_query.score_trusted(self.keys.record(row, kv_head))? * self.scale;
        }
        for (score, row) in prob_tail.iter_mut().zip(
            self.keys
                .tail_rows()
                .take(key_tail_take)
                .chain(std::iter::once(self.current_keys)),
        ) {
            *score = prepared_query.score(self.keys.record(row, kv_head))? * self.scale;
        }
        match self.arithmetic {
            super::Arithmetic::Source => crate::kernels::softmax_round_f16(probabilities)?,
            super::Arithmetic::PinnedSwift => {
                crate::kernels::pinned_attention::softmax_rotated_kv(probabilities)?;
            }
        }
        let mut accumulator = self.values.quantizer.value_accumulator()?;
        let val_sealed_take = self.history_tokens.min(self.values.sealed_rows());
        let val_tail_take = self.history_tokens - val_sealed_take;
        let (w_sealed, w_tail) = probabilities.split_at(val_sealed_take.min(probabilities.len()));
        for (&weight, row) in w_sealed
            .iter()
            .zip(self.values.page_rows().take(val_sealed_take))
        {
            accumulator.accumulate_trusted(self.values.record(row, kv_head), weight)?;
        }
        for (&weight, row) in w_tail.iter().zip(
            self.values
                .tail_rows()
                .take(val_tail_take)
                .chain(std::iter::once(self.current_values)),
        ) {
            accumulator.accumulate(self.values.record(row, kv_head), weight)?;
        }
        let mut output = accumulator.finish()?;
        self.values.quantizer.inverse(&mut output)?;
        super::mul_sigmoid_in_place(
            &mut output,
            super::attention_gate(self.projected, head, HEAD_DIM),
        );
        super::finite(&output, "packed attention output")?;
        result.copy_from_slice(&output);
        Ok(())
    }
}

fn score_pair_group<const G: usize>(
    pair: [&PackedAttention<'_>; 2],
    base_head: usize,
    kv_head: usize,
    score_rows: &mut [&mut [&mut [f32]; G]; 2],
) -> Result<()> {
    let pos0 = pair[0].history_tokens;
    let pos1 = pair[1].history_tokens;
    let mut rotated0 = [[0.0; HEAD_DIM]; G];
    let mut rotated1 = [[0.0; HEAD_DIM]; G];
    for offset in 0..G {
        let head = base_head + offset;
        let range = head * HEAD_DIM..(head + 1) * HEAD_DIM;
        pair[0]
            .keys
            .quantizer
            .rotate(&pair[0].queries[range.clone()], &mut rotated0[offset])?;
        pair[1]
            .keys
            .quantizer
            .rotate(&pair[1].queries[range], &mut rotated1[offset])?;
    }
    let refs0: [&[f32; HEAD_DIM]; G] = std::array::from_fn(|offset| &rotated0[offset]);
    let refs1: [&[f32; HEAD_DIM]; G] = std::array::from_fn(|offset| &rotated1[offset]);
    let prepared_pair = pair[0].keys.quantizer.prepare_query_pair(refs0, refs1)?;
    let key_sealed_take = pos0.min(pair[0].keys.sealed_rows());
    let key_tail_take = pos0 - key_sealed_take;
    let mut token = 0_usize;
    for row in pair[0].keys.page_rows().take(key_sealed_take) {
        let (s0, s1) = prepared_pair.score_pair_trusted(pair[0].keys.record(row, kv_head))?;
        for (probabilities, score) in score_rows[0].iter_mut().zip(s0) {
            probabilities[token] = score * pair[0].scale;
        }
        for (probabilities, score) in score_rows[1].iter_mut().zip(s1) {
            probabilities[token] = score * pair[1].scale;
        }
        token += 1;
    }
    for row in pair[0]
        .keys
        .tail_rows()
        .take(key_tail_take)
        .chain(std::iter::once(pair[0].current_keys))
    {
        let (s0, s1) = prepared_pair.score_pair(pair[0].keys.record(row, kv_head))?;
        for (probabilities, score) in score_rows[0].iter_mut().zip(s0) {
            probabilities[token] = score * pair[0].scale;
        }
        for (probabilities, score) in score_rows[1].iter_mut().zip(s1) {
            probabilities[token] = score * pair[1].scale;
        }
        token += 1;
    }
    let s1_last = prepared_pair.score_second(pair[1].keys.record(pair[1].current_keys, kv_head))?;
    for (probabilities, score) in score_rows[1].iter_mut().zip(s1_last) {
        probabilities[pos1] = score * pair[1].scale;
    }
    for token_rows in score_rows.iter_mut() {
        for probabilities in token_rows.iter_mut() {
            match pair[0].arithmetic {
                super::Arithmetic::Source => crate::kernels::softmax_round_f16(probabilities)?,
                super::Arithmetic::PinnedSwift => {
                    crate::kernels::pinned_attention::softmax_rotated_kv(probabilities)?;
                }
            }
        }
    }
    Ok(())
}

fn attend_group_pair<const G: usize>(
    pair: [&PackedAttention<'_>; 2],
    base_head: usize,
    results: [&mut [&mut [f32]; G]; 2],
    mut score_rows: [&mut [&mut [f32]; G]; 2],
) -> Result<()> {
    const PROB_PACK_CHUNK: usize = 64;
    let kv_head = base_head / pair[0].heads_per_kv;
    let pos0 = pair[0].history_tokens;
    let total0 = pos0 + 1;
    let total1 = pair[1].history_tokens + 1;
    if score_rows[0].iter().any(|row| row.len() != total0)
        || score_rows[1].iter().any(|row| row.len() != total1)
    {
        return Err(invalid("packed attention score row length mismatch"));
    }
    score_pair_group::<G>(pair, base_head, kv_head, &mut score_rows)?;
    let mut accumulator = pair[0]
        .values
        .quantizer
        .value_group_pair_accumulator::<G>()?;
    let mut packed_w0 = [[0.0_f32; G]; PROB_PACK_CHUNK];
    let mut packed_w1 = [[0.0_f32; G]; PROB_PACK_CHUNK];
    let val_sealed_take = pos0.min(pair[0].values.sealed_rows());
    let val_tail_take = pos0 - val_sealed_take;
    let mut page_value_rows = pair[0].values.page_rows().take(val_sealed_take);
    let mut tail_value_rows = pair[0]
        .values
        .tail_rows()
        .take(val_tail_take)
        .chain(std::iter::once(pair[0].current_values));
    for chunk_start in (0..total0).step_by(PROB_PACK_CHUNK) {
        let chunk_len = (total0 - chunk_start).min(PROB_PACK_CHUNK);
        for (head, (probs0, probs1)) in score_rows[0].iter().zip(score_rows[1].iter()).enumerate() {
            for (slot, &w0) in packed_w0[..chunk_len]
                .iter_mut()
                .zip(&probs0[chunk_start..chunk_start + chunk_len])
            {
                slot[head] = w0;
            }
            for (slot, &w1) in packed_w1[..chunk_len]
                .iter_mut()
                .zip(&probs1[chunk_start..chunk_start + chunk_len])
            {
                slot[head] = w1;
            }
        }
        let sealed_in_chunk = val_sealed_take.saturating_sub(chunk_start).min(chunk_len);
        for (&w0, &w1) in packed_w0[..sealed_in_chunk]
            .iter()
            .zip(&packed_w1[..sealed_in_chunk])
        {
            let row = page_value_rows
                .next()
                .ok_or_else(|| invalid("packed value cache row is short"))?;
            accumulator.accumulate_pair_trusted(pair[0].values.record(row, kv_head), w0, w1)?;
        }
        for (&w0, &w1) in packed_w0[sealed_in_chunk..chunk_len]
            .iter()
            .zip(&packed_w1[sealed_in_chunk..chunk_len])
        {
            let row = tail_value_rows
                .next()
                .ok_or_else(|| invalid("packed value cache row is short"))?;
            accumulator.accumulate_pair(pair[0].values.record(row, kv_head), w0, w1)?;
        }
    }
    let w1_last: [f32; G] = std::array::from_fn(|head| score_rows[1][head][total0]);
    accumulator.accumulate_second(
        pair[1].values.record(pair[1].current_values, kv_head),
        w1_last,
    )?;
    let mut outputs = accumulator.finish()?;
    for (token, (token_outputs, token_results)) in outputs.iter_mut().zip(results).enumerate() {
        for (offset, (output, result)) in token_outputs
            .iter_mut()
            .zip(token_results.iter_mut())
            .enumerate()
        {
            let head = base_head + offset;
            pair[token].values.quantizer.inverse(output)?;
            super::mul_sigmoid_in_place(
                output,
                super::attention_gate(pair[token].projected, head, HEAD_DIM),
            );
            super::finite(output, "packed attention output")?;
            result.copy_from_slice(output);
        }
    }
    Ok(())
}

fn resize_bytes(count: usize, output: &mut Vec<u8>) -> Result<()> {
    if count > output.len() {
        output.try_reserve_exact(count - output.len())?;
    }
    output.resize(count, 0);
    Ok(())
}

fn resize_f32s(count: usize, output: &mut Vec<f32>) -> Result<()> {
    if count > output.len() {
        output.try_reserve_exact(count - output.len())?;
    }
    output.resize(count, 0.0);
    Ok(())
}
