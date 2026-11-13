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
            Self::Rotated4 | Self::Rotated4FittedValues => Some((Bits::Four, Bits::Four)),
            Self::Rotated4Keys3Values => Some((Bits::Four, Bits::Three)),
        }
    }
}

// Fixed, distinct transforms prevent accidental key/value basis reuse. They are
// numerical profile identities, unrelated to the sampler's request seed.
const KEY_SEED: u64 = 0x5357_4946_544b_5631;
const VALUE_SEED: u64 = 0x5357_4946_5456_5631;

pub(super) struct PackedCache {
    pages: Vec<Arc<Vec<u8>>>,
    tail: Vec<u8>,
    prepared: Vec<u8>,
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
            quantizer,
            heads,
            row_bytes,
            page_bytes: checked_product(&[row_bytes, context.min(super::KV_PAGE_TOKENS)])?,
            maximum: checked_product(&[row_bytes, context])?,
            len: 0,
        })
    }

    fn prepare(&mut self, values: &[f32]) -> Result<()> {
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

    pub(super) fn commit(&mut self) {
        debug_assert_eq!(self.prepared.len(), self.row_bytes);
        self.tail.extend_from_slice(&self.prepared);
        self.len += self.row_bytes;
        if self.tail.len() == self.page_bytes || self.len == self.maximum {
            self.pages.push(Arc::new(std::mem::take(&mut self.tail)));
        }
    }

    fn rows(&self) -> impl Iterator<Item = &[u8]> {
        self.pages
            .iter()
            .flat_map(|page| page.chunks_exact(self.row_bytes))
            .chain(self.tail.chunks_exact(self.row_bytes))
    }

    fn record<'a>(&self, row: &'a [u8], head: usize) -> &'a [u8] {
        let width = self.quantizer.encoded_len();
        &row[head * width..(head + 1) * width]
    }

    pub(super) fn copy(&self) -> Result<Self> {
        let mut pages = super::capacity(self.pages.len())?;
        pages.extend(self.pages.iter().map(Arc::clone));
        Ok(Self {
            pages,
            tail: super::copy_values(&self.tail)?,
            prepared: Vec::new(),
            quantizer: self.quantizer.clone(),
            heads: self.heads,
            row_bytes: self.row_bytes,
            page_bytes: self.page_bytes,
            maximum: self.maximum,
            len: self.len,
        })
    }

    pub(super) fn clear(&mut self) {
        self.pages.clear();
        self.tail.clear();
        self.prepared.clear();
        self.len = 0;
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
        if !key && policy == KvCachePolicy::Rotated4FittedValues {
            packed.quantizer = packed.quantizer.with_fitted_scale();
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

// Match the existing contiguous head partitions exactly. Each worker consumes
// its heads sequentially, so their probability rows have disjoint lifetimes.
pub(super) fn score_rows(heads: usize, parallelism: &RowParallelism) -> usize {
    heads.div_ceil(heads.div_ceil(parallelism.workers()).max(1))
}

struct AttentionTask<'a> {
    base_head: usize,
    attended: &'a mut [f32],
    scores: &'a mut [f32],
}

/// Stream packed history directly: query rotation once, one score row per task,
/// and one bounded rotated value accumulator per active worker. Independent
/// query heads use the existing private row pool, with each head retaining its
/// original score, softmax, value and inverse-transform order. Dispatch joins
/// every worker before returning the earliest head-chunk failure; only the
/// caller's provisional scratch can change. Cancellation retains the ordinary
/// graph's synchronous-token boundary. Never expands a context-sized FP16/F32
/// key or value buffer or creates a second worker pool.
#[expect(
    clippy::too_many_arguments,
    reason = "Borrow immutable history and provisional current projections separately."
)]
pub(super) fn attend(
    keys: &KvCache,
    values: &KvCache,
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
    let key_cache = keys
        .packed
        .as_ref()
        .ok_or_else(|| invalid("missing packed keys"))?;
    let value_cache = values
        .packed
        .as_ref()
        .ok_or_else(|| invalid("missing packed values"))?;
    let tokens = keys.len / keys.width;
    let stride = checked_sum(tokens, 1)?;
    if heads == 0
        || !heads.is_multiple_of(key_cache.heads)
        || key_cache.heads != value_cache.heads
        || keys.len != values.len
        || queries.len() != heads * HEAD_DIM
        || attended.len() != queries.len()
        || projected.len() != 2 * queries.len()
        || scores.len() != score_rows(heads, parallelism) * stride
        || current_key.len() != keys.width
        || current_value.len() != values.width
    {
        return Err(invalid("invalid packed attention geometry"));
    }
    let mut key_records = Vec::new();
    let mut value_records = Vec::new();
    resize_bytes(key_cache.row_bytes, &mut key_records)?;
    resize_bytes(value_cache.row_bytes, &mut value_records)?;
    for ((key, output), (value, value_output)) in current_key
        .as_chunks::<HEAD_DIM>()
        .0
        .iter()
        .zip(key_records.chunks_exact_mut(key_cache.quantizer.encoded_len()))
        .zip(
            current_value
                .as_chunks::<HEAD_DIM>()
                .0
                .iter()
                .zip(value_records.chunks_exact_mut(value_cache.quantizer.encoded_len())),
        )
    {
        key_cache.quantizer.encode(key, output)?;
        value_cache.quantizer.encode(value, value_output)?;
    }
    let input = PackedAttention {
        keys: key_cache,
        values: value_cache,
        queries,
        projected,
        current_keys: &key_records,
        current_values: &value_records,
        heads_per_kv: heads / key_cache.heads,
        scale,
        arithmetic,
    };
    let chunk_heads = heads.div_ceil(parallelism.workers());
    let task_count = score_rows(heads, parallelism);
    if task_count == 1 {
        for (head, result) in attended
            .as_chunks_mut::<HEAD_DIM>()
            .0
            .iter_mut()
            .enumerate()
        {
            input.attend_head(head, result, scores)?;
        }
        return Ok(());
    }
    // Keep the bounded borrowed task descriptors on the stack. This preserves
    // the original worker partitions and earliest-chunk failure order without
    // allocating a task vector or a probability row for each serial head.
    let mut tasks: [Option<AttentionTask<'_>>; MAX_ROW_WORKERS] = std::array::from_fn(|_| None);
    for (index, ((attended, scores), task)) in attended
        .chunks_mut(chunk_heads * HEAD_DIM)
        .zip(scores.chunks_exact_mut(stride))
        .zip(&mut tasks)
        .enumerate()
    {
        *task = Some(AttentionTask {
            base_head: index * chunk_heads,
            attended,
            scores,
        });
    }
    parallelism.update_chunks(&mut tasks[..task_count], 1, |_, tasks| {
        for task in tasks.iter_mut().flatten() {
            for (offset, result) in task
                .attended
                .as_chunks_mut::<HEAD_DIM>()
                .0
                .iter_mut()
                .enumerate()
            {
                input.attend_head(task.base_head + offset, result, task.scores)?;
            }
        }
        Ok(())
    })
}

struct PackedAttention<'a> {
    keys: &'a PackedCache,
    values: &'a PackedCache,
    queries: &'a [f32],
    projected: &'a [f32],
    current_keys: &'a [u8],
    current_values: &'a [u8],
    heads_per_kv: usize,
    scale: f32,
    arithmetic: super::Arithmetic,
}

impl PackedAttention<'_> {
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
        for (score, row) in probabilities
            .iter_mut()
            .zip(self.keys.rows().chain(std::iter::once(self.current_keys)))
        {
            *score = prepared_query.score(self.keys.record(row, kv_head))? * self.scale;
        }
        match self.arithmetic {
            super::Arithmetic::Source => crate::kernels::softmax_round_f16(probabilities)?,
            super::Arithmetic::PinnedSwift => {
                crate::kernels::pinned_attention::softmax_rotated_kv(probabilities)?;
            }
        }
        let mut accumulator = self.values.quantizer.value_accumulator()?;
        for (&weight, row) in probabilities.iter().zip(
            self.values
                .rows()
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

fn resize_bytes(count: usize, output: &mut Vec<u8>) -> Result<()> {
    if count > output.len() {
        output.try_reserve_exact(count - output.len())?;
    }
    output.resize(count, 0);
    Ok(())
}
