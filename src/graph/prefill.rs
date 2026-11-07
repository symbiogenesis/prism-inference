//! Explicit layer-major CPU prompt tiles with unchanged token arithmetic.

use std::fmt::Write as _;

use super::{
    Arithmetic, Attention, Evaluation, LayerDelta, LayerState, Qwen35, Transforms, add,
    checked_product, finite, row_into,
};
use crate::kernels::{
    MatvecBackend, RowParallelism,
    prefill::{ProjectionTile, ROW_TASK_SCRATCH_BYTES},
};
use crate::model::Tensor;
use crate::model::lora::{RankOneAdapter, Scratch as AdapterScratch};
use crate::{Decoder, MAX_PREFILL_BLOCK_TOKENS, Result, invalid};

#[path = "prefill_gdn.rs"]
mod gdn;

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
            .max(self.geometry.value_width());
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
                .max(self.geometry.value_width()),
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
                self.ingest(token)?;
                check()?;
            }
            return Ok(());
        };
        let result = (|| {
            for tokens in tokens.chunks(tile.tokens) {
                check()?;
                self.ingest_cpu_tile(tokens, check, &mut tile)?;
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
        for ((layer, state), delta) in self
            .layers
            .iter()
            .zip(&mut self.state)
            .zip(&mut self.workspace.deltas)
        {
            for (hidden, normalized) in tile.hidden[..count]
                .iter()
                .zip(&mut tile.normalized[..count])
            {
                self.arithmetic.normalize(
                    hidden,
                    &layer.norm,
                    self.geometry.epsilon,
                    normalized,
                )?;
            }
            let chunked = if let (true, Attention::Recurrent(weights)) =
                (tile.chunkwise_gdn, &layer.attention)
            {
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
                    count,
                    check,
                    tile,
                )?;
                true
            } else {
                false
            };
            for index in 0..count {
                check()?;
                let hidden = &mut tile.hidden[index];
                let normalized = &mut tile.normalized[index];
                if chunked {
                    add(hidden, &tile.attention[index]);
                    self.arithmetic.normalize(
                        hidden,
                        &layer.post_norm,
                        self.geometry.epsilon,
                        normalized,
                    )?;
                    check()?;
                    continue;
                }
                let mut evaluation = Evaluation {
                    geometry: &self.geometry,
                    arithmetic: self.arithmetic,
                    transforms: &self.transforms,
                    position: base + index,
                    frequencies_ready: false,
                    projection: &mut self.workspace.projection,
                    adapter: self.adapter.as_ref(),
                    adapter_scale: self.adapter_scale,
                    adapter_scratch: &mut self.workspace.adapter,
                    executor: None,
                };
                match (&layer.attention, &mut *state) {
                    (Attention::Full(weights), LayerState::Full { keys, values }) => {
                        evaluation.full_attention(
                            weights,
                            normalized,
                            keys,
                            values,
                            &mut self.workspace.attention_scratch,
                            delta,
                            &mut self.workspace.attention,
                        )?;
                    }
                    (
                        Attention::Recurrent(weights),
                        LayerState::Recurrent {
                            convolution,
                            matrix,
                        },
                    ) => {
                        evaluation.recurrent_attention(
                            weights,
                            normalized,
                            convolution,
                            matrix,
                            &mut self.workspace.attention_scratch,
                            delta,
                            &mut self.workspace.attention,
                        )?;
                    }
                    _ => return Err(invalid("CPU tiled prefill layer/cache mismatch")),
                }
                match (&mut *state, &mut *delta) {
                    (LayerState::Full { keys, values }, LayerDelta::Full { key, value }) => {
                        keys.reserve_append()?;
                        values.reserve_append()?;
                        keys.append(key);
                        values.append(value);
                    }
                    (
                        LayerState::Recurrent {
                            convolution,
                            matrix,
                        },
                        LayerDelta::Recurrent {
                            convolution: next,
                            matrix: next_matrix,
                        },
                    ) => {
                        std::mem::swap(convolution, next);
                        std::mem::swap(matrix, next_matrix);
                    }
                    _ => return Err(invalid("CPU tiled prefill delta mismatch")),
                }
                add(hidden, &self.workspace.attention);
                self.arithmetic.normalize(
                    hidden,
                    &layer.post_norm,
                    self.geometry.epsilon,
                    normalized,
                )?;
                check()?;
            }
            let mut previous = None;
            for (weight, output) in [(&layer.up, &mut tile.up), (&layer.gate, &mut tile.gate)] {
                check()?;
                project_tile(
                    &self.transforms,
                    &self.geometry,
                    self.arithmetic,
                    self.adapter.as_ref(),
                    self.adapter_scale,
                    &mut self.workspace.adapter,
                    previous,
                    weight,
                    &tile.normalized[..count],
                    &mut output[..count],
                    &mut tile.transformed[..count],
                    &mut tile.projector,
                    check,
                )?;
                previous = Some(weight);
            }
            for (up, gate) in tile.up[..count].iter_mut().zip(&tile.gate[..count]) {
                self.arithmetic.swiglu_in_place_with_parallelism(
                    gate,
                    up,
                    tile.projector.row_parallelism(),
                )?;
            }
            check()?;
            project_tile(
                &self.transforms,
                &self.geometry,
                self.arithmetic,
                self.adapter.as_ref(),
                self.adapter_scale,
                &mut self.workspace.adapter,
                None,
                &layer.down,
                &tile.up[..count],
                &mut tile.normalized[..count],
                &mut tile.transformed[..count],
                &mut tile.projector,
                check,
            )?;
            for (hidden, output) in tile.hidden[..count]
                .iter_mut()
                .zip(&tile.normalized[..count])
            {
                add(hidden, output);
                finite(hidden, "CPU tiled decoder layer output")?;
            }
            check()?;
        }
        Ok(())
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "Keep exact transforms, original-basis adapter and bounded tile storage explicit."
)]
fn project_tile(
    transforms: &Transforms,
    geometry: &super::Geometry,
    arithmetic: Arithmetic,
    adapter: Option<&RankOneAdapter>,
    adapter_scale: f32,
    adapter_scratch: &mut AdapterScratch,
    previous: Option<&Tensor>,
    tensor: &Tensor,
    inputs: &[Vec<f32>],
    outputs: &mut [Vec<f32>],
    transformed: &mut [Vec<f32>],
    projector: &mut ProjectionTile,
    check: &dyn Fn() -> Result<()>,
) -> Result<()> {
    let basis = transforms.basis(tensor);
    let reuse_basis = previous.is_some_and(|prev| transforms.basis(prev) == basis);
    let activation = if basis.0 {
        if !reuse_basis {
            let parallelism = projector.row_parallelism();
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
        &*transformed
    } else {
        inputs
    };
    projector.project_with_previous(
        previous.filter(|_| reuse_basis),
        tensor,
        activation,
        outputs,
        matches!(arithmetic, Arithmetic::PinnedSwift),
        check,
    )?;
    if let Some(adapter) = adapter {
        for (input, output) in inputs.iter().zip(outputs) {
            adapter.linear(&tensor.name, input, adapter_scale, output, adapter_scratch)?;
        }
    }
    Ok(())
}
