//! Explicit layer-major CPU prompt tiles with unchanged token arithmetic.

use std::fmt::Write as _;

use super::{
    Arithmetic, Attention, Evaluation, FullAttention, Geometry, KvCache, LayerState, Qwen35,
    Transforms, add, checked_product, finite, resize_into, rope_frequencies, row_into, small_float,
};
use crate::kernels::{
    MatvecBackend, RowParallelism,
    prefill::{ProjectionTile, ROW_TASK_SCRATCH_BYTES},
    round_f16_slice,
};
use crate::model::Tensor;
use crate::model::lora::{RankOneAdapter, Scratch as AdapterScratch};
use crate::{MAX_PREFILL_BLOCK_TOKENS, Result, invalid};

#[path = "prefill_gdn.rs"]
mod gdn;

#[derive(Clone, Copy, Default)]
pub(super) struct TileReuse<'a> {
    basis: Option<&'a Tensor>,
    prepared: Option<&'a Tensor>,
}

const fn reborrow_executor<'a, 'b: 'a>(
    executor: &'a mut Option<&'b mut dyn crate::kernels::Pq2ProjectionExecutor>,
) -> Option<&'a mut (dyn crate::kernels::Pq2ProjectionExecutor + 'a)> {
    match executor {
        Some(exec) => Some(*exec),
        None => None,
    }
}

/// Lifetime counters for explicitly selected CPU tiled prompt ingestion.
///
/// Counts include provisional work later cancelled or rolled back. They are not
/// performance measurements, committed-token counters or worker/ISA attribution.
#[derive(Clone, Copy, Default, Debug, Eq, PartialEq)]
pub struct TiledPrefillStatistics {
    /// Successfully completed causal token tiles.
    pub completed_tiles: u64,
    /// Tokens in successfully completed tiles, including later request rollback.
    pub completed_tokens: u64,
    /// Packed matrix projections dispatched across multiple tile tokens.
    pub packed_projections: u64,
    /// Packed weight blocks decoded once and shared across tile tokens.
    pub decoded_weight_blocks: u64,
    /// Recurrent heads evaluated by the separately selected FP32 chunk algebra.
    pub chunkwise_gdn_heads: u64,
}

pub(super) struct TiledPrefill {
    tokens: usize,
    scratch_bytes: usize,
    hidden: Vec<Vec<f32>>,
    normalized: Vec<Vec<f32>>,
    up: Vec<Vec<f32>>,
    gate: Vec<Vec<f32>>,
    transformed: Vec<Vec<f32>>,
    mixed: Vec<Vec<f32>>,
    recurrent_gate: Vec<Vec<f32>>,
    beta: Vec<Vec<f32>>,
    alpha: Vec<Vec<f32>>,
    attended: Vec<Vec<f32>>,
    attention: Vec<Vec<f32>>,
    frequencies: Vec<Vec<(f32, f32)>>,
    gdn_conv_outputs: Vec<f32>,
    gdn_head_outputs: Vec<f32>,
    gdn_key_dots: Vec<f32>,
    chunkwise_gdn: bool,
    projector: ProjectionTile,
    statistics: TiledPrefillStatistics,
}

impl TiledPrefill {
    #[cfg_attr(
        not(feature = "parallel-rows"),
        allow(
            clippy::missing_const_for_fn,
            reason = "The delegated setter drops the replaced pool owner in parallel builds; retain one signature across feature sets."
        )
    )]
    pub(super) fn set_row_parallelism(&mut self, parallelism: RowParallelism) {
        self.projector.set_row_parallelism(parallelism);
    }

    fn new(
        tokens: usize,
        width: usize,
        scratch_bytes: usize,
        backend: MatvecBackend,
        parallelism: RowParallelism,
    ) -> Result<Self> {
        let create = || -> Result<Vec<Vec<f32>>> {
            let mut rows = Vec::new();
            rows.try_reserve_exact(tokens)?;
            for _ in 0..tokens {
                let mut row = Vec::new();
                row.try_reserve_exact(width)?;
                rows.push(row);
            }
            Ok(rows)
        };
        let mut frequencies = Vec::new();
        frequencies.try_reserve_exact(tokens)?;
        for _ in 0..tokens {
            frequencies.push(Vec::new());
        }
        let mut projector = ProjectionTile::with_backend(backend);
        projector.set_row_parallelism(parallelism);
        Ok(Self {
            tokens,
            scratch_bytes,
            hidden: create()?,
            normalized: create()?,
            up: create()?,
            gate: create()?,
            transformed: create()?,
            mixed: create()?,
            recurrent_gate: create()?,
            beta: create()?,
            alpha: create()?,
            attended: create()?,
            attention: create()?,
            frequencies,
            gdn_conv_outputs: Vec::new(),
            gdn_head_outputs: Vec::new(),
            gdn_key_dots: Vec::new(),
            chunkwise_gdn: false,
            projector,
            statistics: TiledPrefillStatistics::default(),
        })
    }
}

impl Qwen35 {
    // Rebuild execution identity from the selected typed policy whenever an
    // adapter reset replaces its trailing scale contract. Never infer a live
    // numerical policy by parsing an old identity string.
    pub(super) fn append_tiled_prefill_identity(&self, identity: &mut String) -> Result<()> {
        if let Some(tile) = &self.tiled_prefill {
            identity.try_reserve_exact(64)?;
            write!(identity, ":cpu-tiled-ffn-v1:{}", tile.tokens)?;
            if tile.chunkwise_gdn {
                identity.push_str(":gdn-wy-f32-v1");
            }
        }
        Ok(())
    }

    /// Conservative extra payload/metadata reservation for a CPU prompt tile.
    ///
    /// This is additional to live state, ordinary graph scratch, checkpoints,
    /// worker stacks and allocator overhead. Both packed activation contracts
    /// and floating fallback high-water storage are charged simultaneously.
    ///
    /// # Errors
    /// Rejects tile sizes other than 8, 16 or 32 and size overflow.
    pub fn tiled_prefill_scratch_bytes(&self, tokens: usize) -> Result<usize> {
        if !matches!(tokens, 8 | 16 | 32) {
            return Err(invalid("CPU prefill tiles must contain 8, 16 or 32 tokens"));
        }
        let width = self
            .geometry
            .embedding
            .max(self.geometry.feed_forward)
            .max(self.geometry.convolution_width())
            .max(self.geometry.value_width())
            .max(self.geometry.query_gate_width())
            .max(self.geometry.kv_width());
        checked_product(&[tokens, width, 80])?
            .checked_add(checked_product(&[tokens, 2048])?)
            // The four fixed 32x32 coefficient/product tables and their small
            // temporary vectors exist even for an eight-token, narrow graph.
            .and_then(|bytes| bytes.checked_add(64 * 1024))
            // Fixed task slice/error descriptors stay bounded at 16 workers
            // and 32 token outputs, including later worker-policy changes.
            .and_then(|bytes| bytes.checked_add(ROW_TASK_SCRATCH_BYTES))
            .ok_or_else(|| invalid("CPU prefill scratch reservation overflow"))
    }

    /// Select experimental CPU tiled state-only prefill before the first token.
    ///
    /// This consumes no new weights and leaves ordinary one-token decode and
    /// default block ingestion unchanged. FFN packed blocks are decoded once
    /// across a tile; attention/GDN/convolution remain strictly causal in their
    /// existing token arithmetic order. This is not a chunkwise WY GDN kernel.
    /// Separate identity prevents mixing runtime prefix/cache plans. No private
    /// service or application default selects this unqualified source path.
    ///
    /// The explicit reservation covers additional tile storage, and must be
    /// charged by a model owner without reducing existing reserves. Cancellation
    /// checks occur between tokens, layers and packed matrix projections. An
    /// error may leave provisional layer state; as for [`Decoder::ingest_block`],
    /// the owner must restore the entire request checkpoint or reset before reuse.
    ///
    /// # Errors
    /// Rejects nonempty/already-configured graphs, unsupported sizes, insufficient
    /// reservation and failed allocation before changing selection or identity.
    pub fn with_tiled_prefill(mut self, tokens: usize, scratch_bytes: usize) -> Result<Self> {
        let required = self.tiled_prefill_scratch_bytes(tokens)?;
        if self.position != 0 || self.tiled_prefill.is_some() || scratch_bytes < required {
            return Err(invalid(
                "CPU tiled prefill requires an empty graph and sufficient scratch",
            ));
        }
        let tile = TiledPrefill::new(
            tokens,
            self.geometry
                .embedding
                .max(self.geometry.feed_forward)
                .max(self.geometry.convolution_width())
                .max(self.geometry.value_width())
                .max(self.geometry.query_gate_width())
                .max(self.geometry.kv_width()),
            required,
            self.requested_backend(),
            self.workspace.projection.matvec.clone_row_parallelism(),
        )?;
        let suffix = format!(":cpu-tiled-ffn-v1:{tokens}");
        self.identity.try_reserve_exact(suffix.len())?;
        self.identity.push_str(&suffix);
        self.provenance = std::sync::Arc::new(());
        self.tiled_prefill = Some(tile);
        Ok(self)
    }

    /// Select a separately identified FP32 triangular GDN chunk algorithm.
    ///
    /// Requires explicit tiled prefill selection on an empty graph. This changes
    /// floating evaluation order; it does not promise bit-identical recurrence.
    /// The initial qualification tolerance is `1e-3 + 1e-3 * abs(reference)` for
    /// complete logits and state, including long-prefix continuation and restore.
    /// This source API establishes no actual-model acceptance or speed benefit.
    /// Decay interval products avoid division and handle exactly zero decay.
    ///
    /// # Errors
    /// Rejects nonempty, untiled or already selected graphs and failed identity allocation.
    pub fn with_chunkwise_gdn_prefill(mut self) -> Result<Self> {
        let Some(tile) = self.tiled_prefill.as_mut() else {
            return Err(invalid("chunkwise GDN requires explicit tiled prefill"));
        };
        if self.position != 0 || tile.chunkwise_gdn {
            return Err(invalid("chunkwise GDN requires an empty, unselected graph"));
        }
        let suffix = ":gdn-wy-f32-v1";
        self.identity.try_reserve_exact(suffix.len())?;
        self.identity.push_str(suffix);
        self.provenance = std::sync::Arc::new(());
        tile.chunkwise_gdn = true;
        Ok(self)
    }

    /// Extra reserved tile payload/metadata, zero for ordinary ingestion.
    #[must_use]
    pub fn tiled_prefill_reserved_bytes(&self) -> usize {
        self.tiled_prefill
            .as_ref()
            .map_or(0, |tile| tile.scratch_bytes)
    }

    /// Provisional tile work, separate from one-vector backend diagnostics.
    #[must_use]
    pub fn tiled_prefill_statistics(&self) -> TiledPrefillStatistics {
        self.tiled_prefill
            .as_ref()
            .map_or_else(TiledPrefillStatistics::default, |tile| {
                let mut statistics = tile.statistics;
                statistics.packed_projections = tile.projector.packed_projections;
                statistics.decoded_weight_blocks = tile.projector.decoded_blocks;
                statistics
            })
    }

    pub(super) fn ingest_prefill_block(
        &mut self,
        tokens: &[u32],
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        const fn coerce_executor<'a>(
            executor: &'a mut Box<dyn crate::kernels::Pq2ProjectionExecutor + Send>,
        ) -> &'a mut (dyn crate::kernels::Pq2ProjectionExecutor + 'a) {
            &mut **executor
        }
        let mut executor = self.projection_executor.take();
        let result = self.ingest_prefill_block_with_executor(
            tokens,
            check,
            executor.as_mut().map(coerce_executor),
        );
        self.projection_executor = executor;
        result
    }

    pub(super) fn ingest_prefill_block_with_executor(
        &mut self,
        tokens: &[u32],
        check: &dyn Fn() -> Result<()>,
        mut executor: Option<&mut dyn crate::kernels::Pq2ProjectionExecutor>,
    ) -> Result<()> {
        if tokens.is_empty()
            || tokens.len() > MAX_PREFILL_BLOCK_TOKENS
            || tokens
                .iter()
                .any(|&token| usize::try_from(token).map_or(true, |value| value >= self.vocabulary))
        {
            return Err(invalid("invalid native prefill block bounds or tokens"));
        }
        if self.tiled_prefill.is_some()
            && self
                .position
                .checked_add(tokens.len())
                .is_none_or(|end| end > self.context_limit)
        {
            return Err(invalid("CPU prefill block exceeds remaining context"));
        }
        let Some(mut tile) = self.tiled_prefill.take() else {
            for &token in tokens {
                check()?;
                self.evaluate_token_with_executor(token, None, reborrow_executor(&mut executor))?;
                check()?;
            }
            return Ok(());
        };
        let result = (|| {
            for tokens in tokens.chunks(tile.tokens) {
                check()?;
                self.ingest_cpu_tile(tokens, check, &mut tile, reborrow_executor(&mut executor))?;
                tile.statistics.completed_tiles = tile.statistics.completed_tiles.saturating_add(1);
                tile.statistics.completed_tokens = tile
                    .statistics
                    .completed_tokens
                    .saturating_add(u64::try_from(tokens.len())?);
                check()?;
            }
            Ok(())
        })();
        self.tiled_prefill = Some(tile);
        result
    }

    #[allow(
        clippy::too_many_lines,
        reason = "Keep causal attention commits, FFN tile consumption and cancellation visibly ordered."
    )]
    fn ingest_cpu_tile(
        &mut self,
        tokens: &[u32],
        check: &dyn Fn() -> Result<()>,
        tile: &mut TiledPrefill,
        mut executor: Option<&mut dyn crate::kernels::Pq2ProjectionExecutor>,
    ) -> Result<()> {
        let count = tokens.len();
        let base = self.position;
        for (token, hidden) in tokens.iter().zip(&mut tile.hidden) {
            check()?;
            row_into(&self.embedding, usize::try_from(*token)?, hidden)?;
            if self.transforms.inverse_embedding {
                self.transforms
                    .rotate_with_parallelism(hidden, tile.projector.row_parallelism())?;
                self.transforms.sign(hidden);
            }
            if let Some(adapter) = &self.adapter {
                adapter.embedding(
                    usize::try_from(*token)?,
                    self.adapter_scale,
                    hidden,
                    &mut self.workspace.adapter,
                )?;
            }
        }
        // A partial layer-major attempt is not a committed token prefix. Mark
        // position nonempty before any state mutation so explicit reset always
        // clears provisional state, including failure during the first tile.
        self.position = base + count;
        let mut frequencies_ready = false;
        let total_layers = self.layers.len();
        let narrow_final = self.narrow_last_layer;
        for (layer_idx, ((layer, state), delta)) in self
            .layers
            .iter()
            .zip(&mut self.state)
            .zip(&mut self.workspace.deltas)
            .enumerate()
        {
            let first_attn_target = match &layer.attention {
                Attention::Recurrent(weights) => &weights.qkv,
                Attention::Full(weights) if narrow_final && layer_idx + 1 == total_layers => {
                    &weights.key
                }
                Attention::Full(weights) => &weights.query,
            };
            let norm_reuse = normalize_for_tile_into(
                &self.transforms,
                &self.geometry,
                self.arithmetic,
                &layer.norm,
                first_attn_target,
                &tile.hidden[..count],
                &mut tile.normalized[..count],
                &mut tile.transformed[..count],
                tile.projector.row_parallelism(),
            )?;
            if narrow_final && layer_idx + 1 == total_layers {
                match (&layer.attention, &mut *state) {
                    (Attention::Recurrent(weights), _) => {
                        gdn::recurrent_tile(
                            weights,
                            state,
                            delta,
                            &self.geometry,
                            self.arithmetic,
                            &self.transforms,
                            self.adapter.as_ref(),
                            self.adapter_scale,
                            &mut self.workspace.adapter,
                            &mut self.workspace.projection.matvec,
                            norm_reuse,
                            count,
                            check,
                            tile,
                            reborrow_executor(&mut executor),
                        )?;
                    }
                    (Attention::Full(weights), LayerState::Full { keys, values }) => {
                        check()?;
                        let _ = project_tile_pair(
                            &self.transforms,
                            &self.geometry,
                            self.arithmetic,
                            self.adapter.as_ref(),
                            self.adapter_scale,
                            &mut self.workspace.adapter,
                            &mut self.workspace.projection.matvec,
                            norm_reuse,
                            [&weights.key, &weights.value],
                            &tile.normalized[..count],
                            &mut tile.recurrent_gate[..count],
                            &mut tile.beta[..count],
                            &mut tile.transformed[..count],
                            &mut tile.projector,
                            reborrow_executor(&mut executor),
                            check,
                        )?;
                        full_attention_kv_only_tile(
                            weights,
                            &self.geometry,
                            self.arithmetic,
                            base,
                            count,
                            &mut frequencies_ready,
                            keys,
                            values,
                            tile,
                            check,
                        )?;
                    }
                    _ => return Err(invalid("CPU tiled prefill layer/cache mismatch")),
                }
                check()?;
                continue;
            }
            match (&layer.attention, &mut *state) {
                (Attention::Recurrent(weights), _) => {
                    gdn::recurrent_tile(
                        weights,
                        state,
                        delta,
                        &self.geometry,
                        self.arithmetic,
                        &self.transforms,
                        self.adapter.as_ref(),
                        self.adapter_scale,
                        &mut self.workspace.adapter,
                        &mut self.workspace.projection.matvec,
                        norm_reuse,
                        count,
                        check,
                        tile,
                        reborrow_executor(&mut executor),
                    )?;
                }
                (Attention::Full(weights), LayerState::Full { keys, values }) => {
                    check()?;
                    let _ = project_tile_triple(
                        &self.transforms,
                        &self.geometry,
                        self.arithmetic,
                        self.adapter.as_ref(),
                        self.adapter_scale,
                        &mut self.workspace.adapter,
                        &mut self.workspace.projection.matvec,
                        norm_reuse,
                        [&weights.query, &weights.key, &weights.value],
                        &tile.normalized[..count],
                        &mut tile.mixed[..count],
                        &mut tile.recurrent_gate[..count],
                        &mut tile.beta[..count],
                        &mut tile.transformed[..count],
                        &mut tile.projector,
                        reborrow_executor(&mut executor),
                        check,
                    )?;
                    full_attention_tile(
                        weights,
                        &self.geometry,
                        self.arithmetic,
                        &self.transforms,
                        base,
                        count,
                        &mut frequencies_ready,
                        keys,
                        values,
                        &mut self.workspace.projection,
                        self.adapter.as_ref(),
                        self.adapter_scale,
                        &mut self.workspace.adapter,
                        &mut self.workspace.attention_scratch,
                        tile,
                        check,
                    )?;
                    check()?;
                    let _ = project_tile(
                        &self.transforms,
                        &self.geometry,
                        self.arithmetic,
                        self.adapter.as_ref(),
                        self.adapter_scale,
                        &mut self.workspace.adapter,
                        &mut self.workspace.projection.matvec,
                        TileReuse::default(),
                        &weights.output,
                        &tile.attended[..count],
                        &mut tile.attention[..count],
                        &mut tile.transformed[..count],
                        &mut tile.projector,
                        reborrow_executor(&mut executor),
                        check,
                    )?;
                }
                _ => return Err(invalid("CPU tiled prefill layer/cache mismatch")),
            }
            for index in 0..count {
                check()?;
                add(&mut tile.hidden[index], &tile.attention[index]);
            }
            let post_norm_reuse = normalize_for_tile_into(
                &self.transforms,
                &self.geometry,
                self.arithmetic,
                &layer.post_norm,
                &layer.gate,
                &tile.hidden[..count],
                &mut tile.normalized[..count],
                &mut tile.transformed[..count],
                tile.projector.row_parallelism(),
            )?;
            check()?;
            if try_project_tile_ffn_chain(
                &self.transforms,
                &self.geometry,
                self.arithmetic,
                self.adapter.as_ref(),
                post_norm_reuse,
                [&layer.gate, &layer.up, &layer.down],
                &tile.normalized[..count],
                Some(&tile.hidden[..count]),
                &mut tile.attention[..count],
                &mut tile.transformed[..count],
                &tile.projector,
                reborrow_executor(&mut executor),
                check,
            )? {
                for (hidden, output) in tile.hidden[..count]
                    .iter_mut()
                    .zip(&mut tile.attention[..count])
                {
                    std::mem::swap(hidden, output);
                    finite(hidden, "CPU tiled decoder layer output")?;
                }
            } else {
                let ffn_reuse = project_tile_ffn_swiglu(
                    &self.transforms,
                    &self.geometry,
                    self.arithmetic,
                    self.adapter.as_ref(),
                    self.adapter_scale,
                    &mut self.workspace.adapter,
                    &mut self.workspace.projection.matvec,
                    post_norm_reuse,
                    &layer.gate,
                    &layer.up,
                    &layer.down,
                    &tile.normalized[..count],
                    &mut tile.gate[..count],
                    &mut tile.up[..count],
                    &mut tile.transformed[..count],
                    &mut tile.projector,
                    reborrow_executor(&mut executor),
                    check,
                )?;
                check()?;
                let _ = project_tile(
                    &self.transforms,
                    &self.geometry,
                    self.arithmetic,
                    self.adapter.as_ref(),
                    self.adapter_scale,
                    &mut self.workspace.adapter,
                    &mut self.workspace.projection.matvec,
                    ffn_reuse,
                    &layer.down,
                    &tile.up[..count],
                    &mut tile.normalized[..count],
                    &mut tile.transformed[..count],
                    &mut tile.projector,
                    reborrow_executor(&mut executor),
                    check,
                )?;
                for (hidden, output) in tile.hidden[..count]
                    .iter_mut()
                    .zip(&tile.normalized[..count])
                {
                    add(hidden, output);
                    finite(hidden, "CPU tiled decoder layer output")?;
                }
            }
            check()?;
        }
        Ok(())
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "Keep causal KV append ordering and tile scratch explicit."
)]
fn full_attention_kv_only_tile(
    weights: &FullAttention,
    g: &Geometry,
    arithmetic: Arithmetic,
    base: usize,
    count: usize,
    frequencies_ready: &mut bool,
    keys: &mut KvCache,
    values: &mut KvCache,
    tile: &mut TiledPrefill,
    check: &dyn Fn() -> Result<()>,
) -> Result<()> {
    check()?;
    if !*frequencies_ready {
        for (index, frequencies) in tile.frequencies.iter_mut().enumerate().take(count) {
            rope_frequencies(g, base + index, frequencies)?;
        }
        *frequencies_ready = true;
    }
    let frequencies_ref: &[Vec<(f32, f32)>] = &tile.frequencies[..count];
    let key_norm: &[f32] = &weights.key_norm;
    if count > 1 && tile.projector.row_parallelism().workers() > 1 {
        tile.projector.row_parallelism().update_paired_chunks(
            &mut tile.recurrent_gate[..count],
            1,
            &mut tile.beta[..count],
            1,
            |base_tok, key_chunk, value_chunk| {
                for (offset, (key_tok, value_tok)) in
                    key_chunk.iter_mut().zip(value_chunk.iter_mut()).enumerate()
                {
                    let tok = base_tok + offset;
                    let frequencies = &frequencies_ref[tok];
                    let mut key_finite = true;
                    let mut value_finite = true;
                    for (key_head, value_head) in key_tok
                        .chunks_exact_mut(g.head)
                        .zip(value_tok.chunks_exact_mut(g.head))
                    {
                        key_finite &= arithmetic.normalize_rope_round_f16_in_place(
                            key_head,
                            key_norm,
                            g.epsilon,
                            g,
                            frequencies,
                        )?;
                        value_finite &= round_f16_slice(value_head);
                    }
                    if !key_finite {
                        return Err(invalid("nonfinite values in binary16 key cache"));
                    }
                    if !value_finite {
                        return Err(invalid("nonfinite values in binary16 value cache"));
                    }
                }
                Ok(())
            },
        )?;
    } else {
        for tok in 0..count {
            let frequencies = &frequencies_ref[tok];
            let key_tok = &mut tile.recurrent_gate[tok];
            let value_tok = &mut tile.beta[tok];
            let mut key_finite = true;
            let mut value_finite = true;
            for (key_head, value_head) in key_tok
                .chunks_exact_mut(g.head)
                .zip(value_tok.chunks_exact_mut(g.head))
            {
                key_finite &= arithmetic.normalize_rope_round_f16_in_place(
                    key_head,
                    key_norm,
                    g.epsilon,
                    g,
                    frequencies,
                )?;
                value_finite &= round_f16_slice(value_head);
            }
            if !key_finite {
                return Err(invalid("nonfinite values in binary16 key cache"));
            }
            if !value_finite {
                return Err(invalid("nonfinite values in binary16 value cache"));
            }
        }
    }
    for index in 0..count {
        if index > 0 {
            check()?;
        }
        keys.reserve_append()?;
        values.reserve_append()?;
        keys.append(&tile.recurrent_gate[index]);
        values.append(&tile.beta[index]);
        check()?;
    }
    Ok(())
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::suboptimal_flops,
    reason = "Keep causal KV append ordering, exact FP16 score/value rounding and tile scratch explicit."
)]
fn full_attention_tile(
    weights: &FullAttention,
    g: &Geometry,
    arithmetic: Arithmetic,
    transforms: &Transforms,
    base: usize,
    count: usize,
    frequencies_ready: &mut bool,
    keys: &mut KvCache,
    values: &mut KvCache,
    projection: &mut super::ProjectionScratch,
    adapter: Option<&RankOneAdapter>,
    adapter_scale: f32,
    adapter_scratch: &mut AdapterScratch,
    attention_scratch: &mut super::AttentionScratch,
    tile: &mut TiledPrefill,
    check: &dyn Fn() -> Result<()>,
) -> Result<()> {
    if count <= 1 || tile.projector.row_parallelism().workers() <= 1 {
        for index in 0..count {
            check()?;
            if !*frequencies_ready {
                rope_frequencies(g, base + index, &mut tile.frequencies[index])?;
            }
            std::mem::swap(
                &mut attention_scratch.frequencies,
                &mut tile.frequencies[index],
            );
            let mut evaluation = Evaluation {
                geometry: g,
                arithmetic,
                transforms,
                position: base + index,
                frequencies_ready: true,
                projection,
                adapter,
                adapter_scale,
                adapter_scratch,
                executor: None,
            };
            let step_result = evaluation.full_attention_state(
                weights,
                &tile.mixed[index],
                &mut tile.recurrent_gate[index],
                &mut tile.beta[index],
                keys,
                values,
                &mut attention_scratch.query,
                &mut attention_scratch.gate,
                &mut attention_scratch.scores,
                &mut attention_scratch.frequencies,
                &mut tile.attended[index],
            );
            std::mem::swap(
                &mut attention_scratch.frequencies,
                &mut tile.frequencies[index],
            );
            step_result?;
            keys.reserve_append()?;
            values.reserve_append()?;
            keys.append(&tile.recurrent_gate[index]);
            values.append(&tile.beta[index]);
            check()?;
        }
        *frequencies_ready = true;
        return Ok(());
    }

    check()?;
    if !*frequencies_ready {
        for (index, frequencies) in tile.frequencies.iter_mut().enumerate().take(count) {
            rope_frequencies(g, base + index, frequencies)?;
        }
        *frequencies_ready = true;
    }
    let query_gate_len = g.head * g.heads;
    for index in 0..count {
        resize_into(query_gate_len, &mut tile.up[index])?;
        resize_into(query_gate_len, &mut tile.gate[index])?;
        resize_into(query_gate_len, &mut tile.attended[index])?;
    }
    let mixed_ref: &[Vec<f32>] = &tile.mixed[..count];
    let frequencies_ref: &[Vec<(f32, f32)>] = &tile.frequencies[..count];
    let query_norm: &[f32] = &weights.query_norm;
    tile.projector.row_parallelism().update_paired_chunks(
        &mut tile.up[..count],
        1,
        &mut tile.gate[..count],
        1,
        |base_tok, query_chunk, gate_chunk| {
            for (offset, (query_tok, gate_tok)) in query_chunk
                .iter_mut()
                .zip(gate_chunk.iter_mut())
                .enumerate()
            {
                let tok = base_tok + offset;
                let projected = &mixed_ref[tok];
                let frequencies = &frequencies_ref[tok];
                let mut query_finite = true;
                for (head_idx, (query_head, gate_head)) in query_tok
                    .chunks_exact_mut(g.head)
                    .zip(gate_tok.chunks_exact_mut(g.head))
                    .enumerate()
                {
                    let src = &projected[head_idx * 2 * g.head..(head_idx + 1) * 2 * g.head];
                    gate_head.copy_from_slice(&src[g.head..]);
                    query_finite &= arithmetic.normalize_rope_round_f16_into(
                        &src[..g.head],
                        query_norm,
                        g.epsilon,
                        g,
                        frequencies,
                        query_head,
                    )?;
                }
                if !query_finite {
                    return Err(invalid("nonfinite values in binary16 attention query"));
                }
            }
            Ok(())
        },
    )?;
    let key_norm: &[f32] = &weights.key_norm;
    tile.projector.row_parallelism().update_paired_chunks(
        &mut tile.recurrent_gate[..count],
        1,
        &mut tile.beta[..count],
        1,
        |base_tok, key_chunk, value_chunk| {
            for (offset, (key_tok, value_tok)) in
                key_chunk.iter_mut().zip(value_chunk.iter_mut()).enumerate()
            {
                let tok = base_tok + offset;
                let frequencies = &frequencies_ref[tok];
                let mut key_finite = true;
                let mut value_finite = true;
                for (key_head, value_head) in key_tok
                    .chunks_exact_mut(g.head)
                    .zip(value_tok.chunks_exact_mut(g.head))
                {
                    key_finite &= arithmetic.normalize_rope_round_f16_in_place(
                        key_head,
                        key_norm,
                        g.epsilon,
                        g,
                        frequencies,
                    )?;
                    value_finite &= round_f16_slice(value_head);
                }
                if !key_finite {
                    return Err(invalid("nonfinite values in binary16 key cache"));
                }
                if !value_finite {
                    return Err(invalid("nonfinite values in binary16 value cache"));
                }
            }
            Ok(())
        },
    )?;
    for index in 0..count {
        if index > 0 {
            check()?;
        }
        keys.reserve_append()?;
        values.reserve_append()?;
        keys.append(&tile.recurrent_gate[index]);
        values.append(&tile.beta[index]);
        check()?;
    }
    let scale = 1.0 / small_float(g.head)?.sqrt();
    let max_score_len = base + count;
    let token_score_stride = 4 * max_score_len;
    resize_into(count * token_score_stride, &mut attention_scratch.scores)?;
    let query_ref: &[Vec<f32>] = &tile.up[..count];
    let gate_ref: &[Vec<f32>] = &tile.gate[..count];
    let current_keys: &[Vec<f32>] = &tile.recurrent_gate[..count];
    let current_values: &[Vec<f32>] = &tile.beta[..count];
    let key_cache: &KvCache = keys;
    let value_cache: &KvCache = values;
    let heads_per_kv = g.heads / g.kv_heads;
    tile.projector.row_parallelism().update_paired_chunks(
        &mut tile.attended[..count],
        1,
        &mut attention_scratch.scores,
        token_score_stride,
        |base_tok, attended_chunk, scores_chunk| {
            let chunk_tokens = attended_chunk.len();
            let mut head = 0;
            while head < g.heads {
                let kv_head = head / heads_per_kv;
                let offset = kv_head * g.head;
                if head + 4 <= g.heads && kv_head == (head + 3) / heads_per_kv {
                    let shared_prefix_len = base + base_tok;
                    let total_past = shared_prefix_len + chunk_tokens.saturating_sub(1);
                    let mut past_iter = key_cache.tokens().take(total_past).enumerate();
                    while let Some((pos0, p0)) = past_iter.next() {
                        if pos0 + 4 <= shared_prefix_len
                            && let Some((_, p1)) = past_iter.next()
                            && let Some((_, p2)) = past_iter.next()
                            && let Some((_, p3)) = past_iter.next()
                        {
                            let keys4 = [
                                &p0[offset..offset + g.head],
                                &p1[offset..offset + g.head],
                                &p2[offset..offset + g.head],
                                &p3[offset..offset + g.head],
                            ];
                            let mut t = 0;
                            while t + 2 <= chunk_tokens {
                                let tok0 = base_tok + t;
                                let tok1 = tok0 + 1;
                                let queries0 = std::array::from_fn(|i| {
                                    let h = head + i;
                                    &query_ref[tok0][h * g.head..(h + 1) * g.head]
                                });
                                let queries1 = std::array::from_fn(|i| {
                                    let h = head + i;
                                    &query_ref[tok1][h * g.head..(h + 1) * g.head]
                                });
                                let (dots0_4, dots1_4) = arithmetic
                                    .attention_dot_four_keys_two_token_four_queries(
                                        queries0, queries1, keys4,
                                    )?;
                                let (scores_pair, _) = scores_chunk
                                    [t * token_score_stride..(t + 2) * token_score_stride]
                                    .split_at_mut(token_score_stride);
                                let tok0_scores = scores_pair;
                                for (k_off, dots) in dots0_4.into_iter().enumerate() {
                                    let pos = pos0 + k_off;
                                    tok0_scores[pos] = dots[0] * scale;
                                    tok0_scores[max_score_len + pos] = dots[1] * scale;
                                    tok0_scores[2 * max_score_len + pos] = dots[2] * scale;
                                    tok0_scores[3 * max_score_len + pos] = dots[3] * scale;
                                }
                                let tok1_scores = &mut scores_chunk
                                    [(t + 1) * token_score_stride..(t + 2) * token_score_stride];
                                for (k_off, dots) in dots1_4.into_iter().enumerate() {
                                    let pos = pos0 + k_off;
                                    tok1_scores[pos] = dots[0] * scale;
                                    tok1_scores[max_score_len + pos] = dots[1] * scale;
                                    tok1_scores[2 * max_score_len + pos] = dots[2] * scale;
                                    tok1_scores[3 * max_score_len + pos] = dots[3] * scale;
                                }
                                t += 2;
                            }
                            if t < chunk_tokens {
                                let tok = base_tok + t;
                                let queries = std::array::from_fn(|i| {
                                    let h = head + i;
                                    &query_ref[tok][h * g.head..(h + 1) * g.head]
                                });
                                let dots4 = arithmetic
                                    .attention_dot_four_keys_four_queries(queries, keys4)?;
                                let tok_scores = &mut scores_chunk
                                    [t * token_score_stride..(t + 1) * token_score_stride];
                                for (k_off, dots) in dots4.into_iter().enumerate() {
                                    let pos = pos0 + k_off;
                                    tok_scores[pos] = dots[0] * scale;
                                    tok_scores[max_score_len + pos] = dots[1] * scale;
                                    tok_scores[2 * max_score_len + pos] = dots[2] * scale;
                                    tok_scores[3 * max_score_len + pos] = dots[3] * scale;
                                }
                            }
                            continue;
                        }
                        let past_key = &p0[offset..offset + g.head];
                        let first_t = (pos0 + 1).saturating_sub(shared_prefix_len);
                        let mut t = first_t;
                        while t + 2 <= chunk_tokens {
                            let tok0 = base_tok + t;
                            let tok1 = tok0 + 1;
                            let queries0 = std::array::from_fn(|i| {
                                let h = head + i;
                                &query_ref[tok0][h * g.head..(h + 1) * g.head]
                            });
                            let queries1 = std::array::from_fn(|i| {
                                let h = head + i;
                                &query_ref[tok1][h * g.head..(h + 1) * g.head]
                            });
                            let (dots0, dots1) = arithmetic.attention_dot_two_token_four_queries(
                                queries0, queries1, past_key,
                            )?;
                            {
                                let tok0_scores = &mut scores_chunk
                                    [t * token_score_stride..(t + 1) * token_score_stride];
                                tok0_scores[pos0] = dots0[0] * scale;
                                tok0_scores[max_score_len + pos0] = dots0[1] * scale;
                                tok0_scores[2 * max_score_len + pos0] = dots0[2] * scale;
                                tok0_scores[3 * max_score_len + pos0] = dots0[3] * scale;
                            }
                            {
                                let tok1_scores = &mut scores_chunk
                                    [(t + 1) * token_score_stride..(t + 2) * token_score_stride];
                                tok1_scores[pos0] = dots1[0] * scale;
                                tok1_scores[max_score_len + pos0] = dots1[1] * scale;
                                tok1_scores[2 * max_score_len + pos0] = dots1[2] * scale;
                                tok1_scores[3 * max_score_len + pos0] = dots1[3] * scale;
                            }
                            t += 2;
                        }
                        for rem_t in t..chunk_tokens {
                            let tok = base_tok + rem_t;
                            let queries = std::array::from_fn(|i| {
                                let h = head + i;
                                &query_ref[tok][h * g.head..(h + 1) * g.head]
                            });
                            let dots = arithmetic.attention_dot_four(queries, past_key)?;
                            let tok_scores = &mut scores_chunk
                                [rem_t * token_score_stride..(rem_t + 1) * token_score_stride];
                            tok_scores[pos0] = dots[0] * scale;
                            tok_scores[max_score_len + pos0] = dots[1] * scale;
                            tok_scores[2 * max_score_len + pos0] = dots[2] * scale;
                            tok_scores[3 * max_score_len + pos0] = dots[3] * scale;
                        }
                    }
                    let (attended_pairs, attended_odd) = attended_chunk.as_chunks_mut::<2>();
                    for (pair_idx, [attended_tok0, attended_tok1]) in
                        attended_pairs.iter_mut().enumerate()
                    {
                        let t0 = pair_idx * 2;
                        let t1 = t0 + 1;
                        let tok0 = base_tok + t0;
                        let tok1 = base_tok + t1;
                        let score_len0 = base + tok0 + 1;
                        let score_len1 = base + tok1 + 1;
                        let queries0 = std::array::from_fn(|i| {
                            let h = head + i;
                            &query_ref[tok0][h * g.head..(h + 1) * g.head]
                        });
                        let queries1 = std::array::from_fn(|i| {
                            let h = head + i;
                            &query_ref[tok1][h * g.head..(h + 1) * g.head]
                        });
                        let current_dots0 = arithmetic.attention_current_dot_four(
                            queries0,
                            &current_keys[tok0][offset..offset + g.head],
                        )?;
                        let current_dots1 = arithmetic.attention_current_dot_four(
                            queries1,
                            &current_keys[tok1][offset..offset + g.head],
                        )?;
                        let pair_scores = &mut scores_chunk
                            [t0 * token_score_stride..(t1 + 1) * token_score_stride];
                        let (tok0_scores, tok1_scores) =
                            pair_scores.split_at_mut(token_score_stride);
                        let (s0_left, s0_right) = tok0_scores.split_at_mut(2 * max_score_len);
                        let (s0_0, s0_1) = s0_left.split_at_mut(max_score_len);
                        let (s0_2, s0_3) = s0_right.split_at_mut(max_score_len);
                        let s0_0 = &mut s0_0[..score_len0];
                        let s0_1 = &mut s0_1[..score_len0];
                        let s0_2 = &mut s0_2[..score_len0];
                        let s0_3 = &mut s0_3[..score_len0];
                        s0_0[score_len0 - 1] = current_dots0[0] * scale;
                        s0_1[score_len0 - 1] = current_dots0[1] * scale;
                        s0_2[score_len0 - 1] = current_dots0[2] * scale;
                        s0_3[score_len0 - 1] = current_dots0[3] * scale;
                        for scores in [&mut *s0_0, &mut *s0_1, &mut *s0_2, &mut *s0_3] {
                            arithmetic.attention_softmax(scores)?;
                            round_f16_slice(scores);
                        }
                        let (s1_left, s1_right) = tok1_scores.split_at_mut(2 * max_score_len);
                        let (s1_0, s1_1) = s1_left.split_at_mut(max_score_len);
                        let (s1_2, s1_3) = s1_right.split_at_mut(max_score_len);
                        let s1_0 = &mut s1_0[..score_len1];
                        let s1_1 = &mut s1_1[..score_len1];
                        let s1_2 = &mut s1_2[..score_len1];
                        let s1_3 = &mut s1_3[..score_len1];
                        s1_0[score_len1 - 1] = current_dots1[0] * scale;
                        s1_1[score_len1 - 1] = current_dots1[1] * scale;
                        s1_2[score_len1 - 1] = current_dots1[2] * scale;
                        s1_3[score_len1 - 1] = current_dots1[3] * scale;
                        for scores in [&mut *s1_0, &mut *s1_1, &mut *s1_2, &mut *s1_3] {
                            arithmetic.attention_softmax(scores)?;
                            round_f16_slice(scores);
                        }
                        let probs0: [&[f32]; 4] = [s0_0, s0_1, s0_2, s0_3];
                        let probs1: [&[f32]; 4] = [s1_0, s1_1, s1_2, s1_3];
                        let (r0_left, r0_right) = attended_tok0[head * g.head..(head + 4) * g.head]
                            .split_at_mut(2 * g.head);
                        let (r0_0, r0_1) = r0_left.split_at_mut(g.head);
                        let (r0_2, r0_3) = r0_right.split_at_mut(g.head);
                        let (r1_left, r1_right) = attended_tok1[head * g.head..(head + 4) * g.head]
                            .split_at_mut(2 * g.head);
                        let (r1_0, r1_1) = r1_left.split_at_mut(g.head);
                        let (r1_2, r1_3) = r1_right.split_at_mut(g.head);
                        for (
                            tile_idx,
                            (
                                (((out0_0, out0_1), out0_2), out0_3),
                                (((out1_0, out1_1), out1_2), out1_3),
                            ),
                        ) in r0_0
                            .chunks_mut(
                                crate::kernels::pinned_attention::VALUE_TILE_FOUR_TWO_TOKENS,
                            )
                            .zip(r0_1.chunks_mut(
                                crate::kernels::pinned_attention::VALUE_TILE_FOUR_TWO_TOKENS,
                            ))
                            .zip(r0_2.chunks_mut(
                                crate::kernels::pinned_attention::VALUE_TILE_FOUR_TWO_TOKENS,
                            ))
                            .zip(r0_3.chunks_mut(
                                crate::kernels::pinned_attention::VALUE_TILE_FOUR_TWO_TOKENS,
                            ))
                            .zip(
                                r1_0.chunks_mut(
                                    crate::kernels::pinned_attention::VALUE_TILE_FOUR_TWO_TOKENS,
                                )
                                .zip(r1_1.chunks_mut(
                                    crate::kernels::pinned_attention::VALUE_TILE_FOUR_TWO_TOKENS,
                                ))
                                .zip(r1_2.chunks_mut(
                                    crate::kernels::pinned_attention::VALUE_TILE_FOUR_TWO_TOKENS,
                                ))
                                .zip(r1_3.chunks_mut(
                                    crate::kernels::pinned_attention::VALUE_TILE_FOUR_TWO_TOKENS,
                                )),
                            )
                            .enumerate()
                        {
                            let dimension = tile_idx
                                * crate::kernels::pinned_attention::VALUE_TILE_FOUR_TWO_TOKENS;
                            let len = out0_0.len();
                            arithmetic.attention_value_tile_four_two_tokens(
                                value_cache,
                                [
                                    &current_values[tok0]
                                        [offset + dimension..offset + dimension + len],
                                    &current_values[tok1]
                                        [offset + dimension..offset + dimension + len],
                                ],
                                offset + dimension,
                                probs0,
                                probs1,
                                [out0_0, out0_1, out0_2, out0_3],
                                [out1_0, out1_1, out1_2, out1_3],
                            )?;
                        }
                        crate::kernels::mul_sigmoid_in_place(
                            &mut attended_tok0[head * g.head..(head + 4) * g.head],
                            &gate_ref[tok0][head * g.head..(head + 4) * g.head],
                        );
                        crate::kernels::mul_sigmoid_in_place(
                            &mut attended_tok1[head * g.head..(head + 4) * g.head],
                            &gate_ref[tok1][head * g.head..(head + 4) * g.head],
                        );
                    }
                    if let Some(attended_tok) = attended_odd.first_mut() {
                        let t = attended_pairs.len() * 2;
                        let tok = base_tok + t;
                        let score_len = base + tok + 1;
                        let queries = std::array::from_fn(|i| {
                            let h = head + i;
                            &query_ref[tok][h * g.head..(h + 1) * g.head]
                        });
                        let current_dots = arithmetic.attention_current_dot_four(
                            queries,
                            &current_keys[tok][offset..offset + g.head],
                        )?;
                        let tok_scores =
                            &mut scores_chunk[t * token_score_stride..(t + 1) * token_score_stride];
                        let (s01, s23) = tok_scores.split_at_mut(2 * max_score_len);
                        let (s0, s1) = s01.split_at_mut(max_score_len);
                        let (s2, s3) = s23.split_at_mut(max_score_len);
                        let s0 = &mut s0[..score_len];
                        let s1 = &mut s1[..score_len];
                        let s2 = &mut s2[..score_len];
                        let s3 = &mut s3[..score_len];
                        s0[score_len - 1] = current_dots[0] * scale;
                        s1[score_len - 1] = current_dots[1] * scale;
                        s2[score_len - 1] = current_dots[2] * scale;
                        s3[score_len - 1] = current_dots[3] * scale;
                        for scores in [&mut *s0, &mut *s1, &mut *s2, &mut *s3] {
                            arithmetic.attention_softmax(scores)?;
                            round_f16_slice(scores);
                        }
                        let probs: [&[f32]; 4] = [s0, s1, s2, s3];
                        let (r01, r23) = attended_tok[head * g.head..(head + 4) * g.head]
                            .split_at_mut(2 * g.head);
                        let (r0, r1) = r01.split_at_mut(g.head);
                        let (r2, r3) = r23.split_at_mut(g.head);
                        for (tile_idx, (((out0, out1), out2), out3)) in r0
                            .chunks_mut(crate::kernels::pinned_attention::VALUE_TILE_FOUR)
                            .zip(r1.chunks_mut(crate::kernels::pinned_attention::VALUE_TILE_FOUR))
                            .zip(r2.chunks_mut(crate::kernels::pinned_attention::VALUE_TILE_FOUR))
                            .zip(r3.chunks_mut(crate::kernels::pinned_attention::VALUE_TILE_FOUR))
                            .enumerate()
                        {
                            let dimension =
                                tile_idx * crate::kernels::pinned_attention::VALUE_TILE_FOUR;
                            arithmetic.attention_value_tile_four(
                                value_cache,
                                &current_values[tok]
                                    [offset + dimension..offset + dimension + out0.len()],
                                offset + dimension,
                                probs,
                                [out0, out1, out2, out3],
                            )?;
                        }
                        crate::kernels::mul_sigmoid_in_place(
                            &mut attended_tok[head * g.head..(head + 4) * g.head],
                            &gate_ref[tok][head * g.head..(head + 4) * g.head],
                        );
                    }
                    head += 4;
                    continue;
                }
                for (pos, past) in key_cache
                    .tokens()
                    .take(base + base_tok + chunk_tokens.saturating_sub(1))
                    .enumerate()
                {
                    let past_key = &past[offset..offset + g.head];
                    let first_t = (pos + 1).saturating_sub(base + base_tok);
                    for t in first_t..chunk_tokens {
                        let tok = base_tok + t;
                        let q = &query_ref[tok][head * g.head..(head + 1) * g.head];
                        scores_chunk[t * token_score_stride + pos] =
                            arithmetic.attention_dot(q, past_key)? * scale;
                    }
                }
                for (t, attended_tok) in attended_chunk.iter_mut().enumerate() {
                    let tok = base_tok + t;
                    let score_len = base + tok + 1;
                    let q = &query_ref[tok][head * g.head..(head + 1) * g.head];
                    let scores = &mut scores_chunk
                        [t * token_score_stride..t * token_score_stride + score_len];
                    scores[score_len - 1] = arithmetic
                        .attention_current_dot(q, &current_keys[tok][offset..offset + g.head])?
                        * scale;
                    arithmetic.attention_softmax(scores)?;
                    round_f16_slice(scores);
                    let result = &mut attended_tok[head * g.head..(head + 1) * g.head];
                    for (tile_idx, out) in result
                        .chunks_mut(crate::kernels::pinned_attention::VALUE_TILE)
                        .enumerate()
                    {
                        let dimension = tile_idx * crate::kernels::pinned_attention::VALUE_TILE;
                        arithmetic.attention_value_tile(
                            value_cache,
                            &current_values[tok]
                                [offset + dimension..offset + dimension + out.len()],
                            offset + dimension,
                            scores,
                            out,
                        )?;
                    }
                    crate::kernels::mul_sigmoid_in_place(
                        result,
                        &gate_ref[tok][head * g.head..(head + 1) * g.head],
                    );
                }
                head += 1;
            }
            Ok(())
        },
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "Keep tile RMSNorm, fused Hadamard rotation and bounded tile scratch explicit."
)]
fn normalize_for_tile_into<'a>(
    transforms: &Transforms,
    geometry: &Geometry,
    arithmetic: Arithmetic,
    weight: &[f32],
    first_target: &'a Tensor,
    hidden: &[Vec<f32>],
    normalized: &mut [Vec<f32>],
    transformed: &mut [Vec<f32>],
    parallelism: &RowParallelism,
) -> Result<TileReuse<'a>> {
    let count = hidden.len();
    let len = hidden.first().map_or(0, Vec::len);
    if count > 0
        && len > 0
        && transforms.basis(first_target) == (true, false)
        && transforms.can_block_hadamard(len)
        && first_target.dimensions.first().copied() == Some(len)
    {
        let signs = transforms.signs.get(&len).map(Vec::as_slice);
        let block = transforms.block;
        let epsilon = geometry.epsilon;
        let fused_ok = if count > 1 && parallelism.workers() > 1 {
            let single_worker = RowParallelism::default();
            parallelism
                .update_paired_chunks(
                    normalized,
                    1,
                    transformed,
                    1,
                    |base_tok, norm_chunk, trans_chunk| {
                        for (offset, (norm_out, rot_out)) in norm_chunk
                            .iter_mut()
                            .zip(trans_chunk.iter_mut())
                            .enumerate()
                        {
                            arithmetic.normalize_hadamard_into(
                                &hidden[base_tok + offset],
                                weight,
                                epsilon,
                                block,
                                signs,
                                &single_worker,
                                norm_out,
                                rot_out,
                            )?;
                        }
                        Ok(())
                    },
                )
                .is_ok()
        } else {
            let mut ok = true;
            for ((h, norm_out), rot_out) in hidden
                .iter()
                .zip(normalized.iter_mut())
                .zip(transformed.iter_mut())
            {
                if arithmetic
                    .normalize_hadamard_into(
                        h,
                        weight,
                        epsilon,
                        block,
                        signs,
                        parallelism,
                        norm_out,
                        rot_out,
                    )
                    .is_err()
                {
                    ok = false;
                    break;
                }
            }
            ok
        };
        if fused_ok {
            return Ok(TileReuse {
                basis: Some(first_target),
                prepared: None,
            });
        }
    }
    if count > 1 && parallelism.workers() > 1 {
        let epsilon = geometry.epsilon;
        parallelism.update_chunks(normalized, 1, |base_tok, chunk| {
            for (offset, norm_out) in chunk.iter_mut().enumerate() {
                arithmetic.normalize(&hidden[base_tok + offset], weight, epsilon, norm_out)?;
            }
            Ok(())
        })?;
    } else {
        for (h, norm_out) in hidden.iter().zip(normalized.iter_mut()) {
            arithmetic.normalize(h, weight, geometry.epsilon, norm_out)?;
        }
    }
    Ok(TileReuse::default())
}

fn transform_tile_inputs<'a>(
    transforms: &Transforms,
    geometry: &Geometry,
    tensor: &Tensor,
    reuse_basis: bool,
    inputs: &'a [Vec<f32>],
    transformed: &'a mut [Vec<f32>],
    parallelism: &RowParallelism,
) -> Result<&'a [Vec<f32>]> {
    let basis = transforms.basis(tensor);
    if basis.0 {
        if !reuse_basis {
            if inputs.len() > 1 && parallelism.workers() > 1 {
                parallelism.update_chunks(transformed, 1, |base, chunk| {
                    for (offset, target) in chunk.iter_mut().enumerate() {
                        transforms.transform_into(
                            tensor,
                            &inputs[base + offset],
                            geometry,
                            target,
                        )?;
                    }
                    Ok(())
                })?;
            } else {
                for (input, transformed) in inputs.iter().zip(transformed.iter_mut()) {
                    transforms.transform_into_with_parallelism(
                        tensor,
                        input,
                        geometry,
                        parallelism,
                        transformed,
                    )?;
                }
            }
        }
        Ok(&*transformed)
    } else {
        Ok(inputs)
    }
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "Keep exact transforms, original-basis adapter and bounded tile storage explicit."
)]
fn project_tile_pair<'a>(
    transforms: &Transforms,
    geometry: &Geometry,
    arithmetic: Arithmetic,
    adapter: Option<&RankOneAdapter>,
    adapter_scale: f32,
    adapter_scratch: &mut AdapterScratch,
    projection_scratch: &mut crate::kernels::MatvecScratch,
    previous: TileReuse<'a>,
    tensors: [&'a Tensor; 2],
    inputs: &[Vec<f32>],
    outputs_a: &mut [Vec<f32>],
    outputs_b: &mut [Vec<f32>],
    transformed: &mut [Vec<f32>],
    projector: &mut ProjectionTile,
    mut executor: Option<&mut dyn crate::kernels::Pq2ProjectionExecutor>,
    check: &dyn Fn() -> Result<()>,
) -> Result<TileReuse<'a>> {
    if let Some(exec) = reborrow_executor(&mut executor)
        && tensors[0].kind == 142
        && tensors[1].kind == 142
        && transforms.basis(tensors[0]) == transforms.basis(tensors[1])
        && exec.supports_f32_group(&tensors)
    {
        let reuse_basis = previous
            .basis
            .is_some_and(|prev| transforms.basis(prev) == transforms.basis(tensors[0]));
        let activation = transform_tile_inputs(
            transforms,
            geometry,
            tensors[0],
            reuse_basis,
            inputs,
            transformed,
            projector.row_parallelism(),
        )?;
        check()?;
        exec.project_f32_pair_tokens(
            tensors,
            activation,
            matches!(arithmetic, Arithmetic::PinnedSwift),
            outputs_a,
            outputs_b,
        )?;
        let rows_a = tensors[0]
            .dimensions
            .get(1..)
            .map_or(0, |dims| dims.iter().product::<usize>());
        let rows_b = tensors[1]
            .dimensions
            .get(1..)
            .map_or(0, |dims| dims.iter().product::<usize>());
        for (out_a, out_b) in outputs_a.iter().zip(outputs_b.iter()) {
            if out_a.len() != rows_a || out_b.len() != rows_b {
                return Err(invalid("projection executor returned incomplete output"));
            }
            finite(out_a, "projection executor output")?;
            finite(out_b, "projection executor output")?;
        }
        if let Some(adapter) = adapter {
            adapter.linear_tokens(
                &tensors[0].name,
                inputs,
                adapter_scale,
                outputs_a,
                adapter_scratch,
            )?;
            adapter.linear_tokens(
                &tensors[1].name,
                inputs,
                adapter_scale,
                outputs_b,
                adapter_scratch,
            )?;
        }
        return Ok(TileReuse {
            basis: Some(tensors[1]),
            prepared: None,
        });
    }
    let reuse = project_tile(
        transforms,
        geometry,
        arithmetic,
        adapter,
        adapter_scale,
        adapter_scratch,
        projection_scratch,
        previous,
        tensors[0],
        inputs,
        outputs_a,
        transformed,
        projector,
        reborrow_executor(&mut executor),
        check,
    )?;
    check()?;
    project_tile(
        transforms,
        geometry,
        arithmetic,
        adapter,
        adapter_scale,
        adapter_scratch,
        projection_scratch,
        reuse,
        tensors[1],
        inputs,
        outputs_b,
        transformed,
        projector,
        reborrow_executor(&mut executor),
        check,
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "Keep exact transforms, residual chaining and bounded tile storage explicit."
)]
fn try_project_tile_ffn_chain(
    transforms: &Transforms,
    geometry: &Geometry,
    arithmetic: Arithmetic,
    adapter: Option<&RankOneAdapter>,
    previous: TileReuse<'_>,
    tensors: [&Tensor; 3],
    inputs: &[Vec<f32>],
    residuals: Option<&[Vec<f32>]>,
    outputs: &mut [Vec<f32>],
    transformed: &mut [Vec<f32>],
    projector: &ProjectionTile,
    mut executor: Option<&mut dyn crate::kernels::Pq2ProjectionExecutor>,
    check: &dyn Fn() -> Result<()>,
) -> Result<bool> {
    let [gate, up, down] = tensors;
    let Some(hadamard_block) = transforms.ffn_chain_hadamard_block_for(down) else {
        return Ok(false);
    };
    if adapter.is_some()
        || gate.kind != 142
        || up.kind != 142
        || down.kind != 142
        || transforms.basis(gate) != transforms.basis(up)
    {
        return Ok(false);
    }
    let Some(exec) = reborrow_executor(&mut executor) else {
        return Ok(false);
    };
    if !exec.supports_f32_ffn_chain(tensors) {
        return Ok(false);
    }
    let reuse_basis = previous
        .basis
        .is_some_and(|prev| transforms.basis(prev) == transforms.basis(gate));
    let activation = transform_tile_inputs(
        transforms,
        geometry,
        gate,
        reuse_basis,
        inputs,
        transformed,
        projector.row_parallelism(),
    )?;
    check()?;
    exec.project_f32_ffn_chain_tokens(
        tensors,
        activation,
        residuals,
        hadamard_block,
        matches!(arithmetic, Arithmetic::PinnedSwift),
        outputs,
    )?;
    let rows = down
        .dimensions
        .get(1..)
        .map_or(0, |dims| dims.iter().product::<usize>());
    for out in outputs.iter() {
        if out.len() != rows {
            return Err(invalid("projection executor returned incomplete output"));
        }
        finite(out, "projection executor output")?;
    }
    Ok(true)
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "Keep exact transforms, original-basis adapter and bounded tile storage explicit."
)]
fn project_tile_ffn_swiglu<'a>(
    transforms: &Transforms,
    geometry: &Geometry,
    arithmetic: Arithmetic,
    adapter: Option<&RankOneAdapter>,
    adapter_scale: f32,
    adapter_scratch: &mut AdapterScratch,
    projection_scratch: &mut crate::kernels::MatvecScratch,
    previous: TileReuse<'a>,
    gate: &'a Tensor,
    up: &'a Tensor,
    down: &'a Tensor,
    inputs: &[Vec<f32>],
    gate_outputs: &mut [Vec<f32>],
    up_outputs: &mut [Vec<f32>],
    transformed: &mut [Vec<f32>],
    projector: &mut ProjectionTile,
    mut executor: Option<&mut dyn crate::kernels::Pq2ProjectionExecutor>,
    check: &dyn Fn() -> Result<()>,
) -> Result<TileReuse<'a>> {
    if adapter.is_none()
        && let Some(exec) = reborrow_executor(&mut executor)
        && gate.kind == 142
        && up.kind == 142
        && transforms.basis(gate) == transforms.basis(up)
        && exec.supports_f32_ffn_swiglu(gate, up)
    {
        let reuse_basis = previous
            .basis
            .is_some_and(|prev| transforms.basis(prev) == transforms.basis(gate));
        let activation = transform_tile_inputs(
            transforms,
            geometry,
            gate,
            reuse_basis,
            inputs,
            transformed,
            projector.row_parallelism(),
        )?;
        check()?;
        exec.project_f32_ffn_swiglu_tokens(
            gate,
            up,
            activation,
            matches!(arithmetic, Arithmetic::PinnedSwift),
            up_outputs,
        )?;
        let rows = up
            .dimensions
            .get(1..)
            .map_or(0, |dims| dims.iter().product::<usize>());
        for out in up_outputs.iter() {
            if out.len() != rows {
                return Err(invalid("projection executor returned incomplete output"));
            }
            finite(out, "projection executor output")?;
        }
        return Ok(TileReuse::default());
    }
    let _ = project_tile_pair(
        transforms,
        geometry,
        arithmetic,
        adapter,
        adapter_scale,
        adapter_scratch,
        projection_scratch,
        previous,
        [up, gate],
        inputs,
        up_outputs,
        gate_outputs,
        transformed,
        projector,
        executor,
        check,
    )?;
    let count = inputs.len();
    let up_len = up_outputs.first().map_or(0, Vec::len);
    if adapter.is_none()
        && up_len > 0
        && transforms.basis(down) == (true, false)
        && transforms.can_block_hadamard(up_len)
        && down.dimensions.first().copied() == Some(up_len)
    {
        let signs = transforms.signs.get(&up_len).map(Vec::as_slice);
        let block = transforms.block;
        let fused_ok = if count > 1 && projector.row_parallelism().workers() > 1 {
            let gate_ref = &gate_outputs[..count];
            let up_ref = &up_outputs[..count];
            let single_worker = RowParallelism::default();
            projector
                .row_parallelism()
                .update_chunks(&mut transformed[..count], 1, |base_tok, chunk| {
                    for (offset, target) in chunk.iter_mut().enumerate() {
                        let tok = base_tok + offset;
                        arithmetic.swiglu_hadamard_read_into_with_parallelism(
                            &gate_ref[tok],
                            &up_ref[tok],
                            block,
                            signs,
                            &single_worker,
                            target,
                        )?;
                    }
                    Ok(())
                })
                .is_ok()
        } else {
            let parallelism = projector.row_parallelism();
            let mut ok = true;
            for ((up_out, gate_out), target) in up_outputs[..count]
                .iter()
                .zip(&gate_outputs[..count])
                .zip(&mut transformed[..count])
            {
                if arithmetic
                    .swiglu_hadamard_read_into_with_parallelism(
                        gate_out,
                        up_out,
                        block,
                        signs,
                        parallelism,
                        target,
                    )
                    .is_err()
                {
                    ok = false;
                    break;
                }
            }
            ok
        };
        if fused_ok {
            return Ok(TileReuse {
                basis: Some(down),
                prepared: None,
            });
        }
    }
    if count > 1 && projector.row_parallelism().workers() > 1 {
        let gate_ref = &gate_outputs[..count];
        projector.row_parallelism().update_chunks(
            &mut up_outputs[..count],
            1,
            |base_tok, chunk| {
                for (offset, up_out) in chunk.iter_mut().enumerate() {
                    arithmetic.swiglu_in_place(&gate_ref[base_tok + offset], up_out)?;
                }
                Ok(())
            },
        )?;
    } else {
        for (up_out, gate_out) in up_outputs[..count].iter_mut().zip(&gate_outputs[..count]) {
            arithmetic.swiglu_in_place_with_parallelism(
                gate_out,
                up_out,
                projector.row_parallelism(),
            )?;
        }
    }
    Ok(TileReuse::default())
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "Keep exact transforms, original-basis adapter and bounded tile storage explicit."
)]
fn project_tile_triple<'a>(
    transforms: &Transforms,
    geometry: &Geometry,
    arithmetic: Arithmetic,
    adapter: Option<&RankOneAdapter>,
    adapter_scale: f32,
    adapter_scratch: &mut AdapterScratch,
    projection_scratch: &mut crate::kernels::MatvecScratch,
    previous: TileReuse<'a>,
    tensors: [&'a Tensor; 3],
    inputs: &[Vec<f32>],
    outputs_a: &mut [Vec<f32>],
    outputs_b: &mut [Vec<f32>],
    outputs_c: &mut [Vec<f32>],
    transformed: &mut [Vec<f32>],
    projector: &mut ProjectionTile,
    mut executor: Option<&mut dyn crate::kernels::Pq2ProjectionExecutor>,
    check: &dyn Fn() -> Result<()>,
) -> Result<TileReuse<'a>> {
    if let Some(exec) = reborrow_executor(&mut executor)
        && tensors[0].kind == 142
        && tensors[1].kind == 142
        && tensors[2].kind == 142
        && transforms.basis(tensors[0]) == transforms.basis(tensors[1])
        && transforms.basis(tensors[0]) == transforms.basis(tensors[2])
        && exec.supports_f32_group(&tensors)
    {
        let reuse_basis = previous
            .basis
            .is_some_and(|prev| transforms.basis(prev) == transforms.basis(tensors[0]));
        let activation = transform_tile_inputs(
            transforms,
            geometry,
            tensors[0],
            reuse_basis,
            inputs,
            transformed,
            projector.row_parallelism(),
        )?;
        check()?;
        exec.project_f32_triple_tokens(
            tensors,
            activation,
            matches!(arithmetic, Arithmetic::PinnedSwift),
            outputs_a,
            outputs_b,
            outputs_c,
        )?;
        let rows_a = tensors[0]
            .dimensions
            .get(1..)
            .map_or(0, |dims| dims.iter().product::<usize>());
        let rows_b = tensors[1]
            .dimensions
            .get(1..)
            .map_or(0, |dims| dims.iter().product::<usize>());
        let rows_c = tensors[2]
            .dimensions
            .get(1..)
            .map_or(0, |dims| dims.iter().product::<usize>());
        for ((out_a, out_b), out_c) in outputs_a.iter().zip(outputs_b.iter()).zip(outputs_c.iter())
        {
            if out_a.len() != rows_a || out_b.len() != rows_b || out_c.len() != rows_c {
                return Err(invalid("projection executor returned incomplete output"));
            }
            finite(out_a, "projection executor output")?;
            finite(out_b, "projection executor output")?;
            finite(out_c, "projection executor output")?;
        }
        if let Some(adapter) = adapter {
            for (tensor, outputs) in [
                (tensors[0], &mut *outputs_a),
                (tensors[1], &mut *outputs_b),
                (tensors[2], &mut *outputs_c),
            ] {
                adapter.linear_tokens(
                    &tensor.name,
                    inputs,
                    adapter_scale,
                    outputs,
                    adapter_scratch,
                )?;
            }
        }
        return Ok(TileReuse {
            basis: Some(tensors[2]),
            prepared: None,
        });
    }
    let reuse = project_tile_pair(
        transforms,
        geometry,
        arithmetic,
        adapter,
        adapter_scale,
        adapter_scratch,
        projection_scratch,
        previous,
        [tensors[0], tensors[1]],
        inputs,
        outputs_a,
        outputs_b,
        transformed,
        projector,
        reborrow_executor(&mut executor),
        check,
    )?;
    check()?;
    project_tile(
        transforms,
        geometry,
        arithmetic,
        adapter,
        adapter_scale,
        adapter_scratch,
        projection_scratch,
        reuse,
        tensors[2],
        inputs,
        outputs_c,
        transformed,
        projector,
        reborrow_executor(&mut executor),
        check,
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "Keep exact transforms, original-basis adapter and bounded tile storage explicit."
)]
fn project_tile<'a>(
    transforms: &Transforms,
    geometry: &Geometry,
    arithmetic: Arithmetic,
    adapter: Option<&RankOneAdapter>,
    adapter_scale: f32,
    adapter_scratch: &mut AdapterScratch,
    projection_scratch: &mut crate::kernels::MatvecScratch,
    previous: TileReuse<'a>,
    tensor: &'a Tensor,
    inputs: &[Vec<f32>],
    outputs: &mut [Vec<f32>],
    transformed: &mut [Vec<f32>],
    projector: &mut ProjectionTile,
    mut executor: Option<&mut dyn crate::kernels::Pq2ProjectionExecutor>,
    check: &dyn Fn() -> Result<()>,
) -> Result<TileReuse<'a>> {
    let basis = transforms.basis(tensor);
    let reuse_basis = previous
        .basis
        .is_some_and(|prev| transforms.basis(prev) == basis);
    let reuse_prepared = reuse_basis
        && previous
            .prepared
            .is_some_and(|prev| transforms.basis(prev) == basis);
    let activation = transform_tile_inputs(
        transforms,
        geometry,
        tensor,
        reuse_basis,
        inputs,
        transformed,
        projector.row_parallelism(),
    )?;
    let prepared = if let Some(exec) = reborrow_executor(&mut executor) {
        let cols = tensor.dimensions.first().copied().unwrap_or(0);
        let rows = tensor
            .dimensions
            .get(1..)
            .map_or(0, |dims| dims.iter().product::<usize>());
        if tensor.kind == 142 && cols.is_multiple_of(128) && exec.supports_f32(tensor) {
            check()?;
            exec.project_f32_tokens(
                tensor,
                activation,
                matches!(arithmetic, Arithmetic::PinnedSwift),
                outputs,
            )?;
            for out in outputs.iter() {
                if out.len() != rows {
                    return Err(invalid("projection executor returned incomplete output"));
                }
                finite(out, "projection executor output")?;
            }
            None
        } else if (tensor.kind == 142 && cols.is_multiple_of(256) && exec.supports(tensor))
            || ((tensor.kind == 143 || (tensor.kind == 142 && !cols.is_multiple_of(256)))
                && exec.supports_q8(tensor))
        {
            for (input, output) in activation.iter().zip(outputs.iter_mut()) {
                check()?;
                projection_scratch
                    .prepare(tensor, input)?
                    .matvec_with_executor(tensor, output, exec)?;
            }
            None
        } else {
            projector.project_with_previous(
                previous.prepared.filter(|_| reuse_prepared),
                tensor,
                activation,
                outputs,
                matches!(arithmetic, Arithmetic::PinnedSwift),
                check,
            )?;
            Some(tensor)
        }
    } else {
        projector.project_with_previous(
            previous.prepared.filter(|_| reuse_prepared),
            tensor,
            activation,
            outputs,
            matches!(arithmetic, Arithmetic::PinnedSwift),
            check,
        )?;
        Some(tensor)
    };
    if let Some(adapter) = adapter {
        adapter.linear_tokens(
            &tensor.name,
            inputs,
            adapter_scale,
            outputs,
            adapter_scratch,
        )?;
    }
    Ok(TileReuse {
        basis: Some(tensor),
        prepared,
    })
}
