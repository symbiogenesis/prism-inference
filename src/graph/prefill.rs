//! Explicit layer-major CPU prompt tiles with unchanged token arithmetic.

use std::fmt::Write as _;

use super::{
    ActiveAdapters, Arithmetic, Attention, Evaluation, FullAttention, Geometry, KvCache, Layer,
    LayerDelta, LayerState, Qwen35, RecurrentAttention, Transforms, add_finite, attention_gate,
    checked_product, finite, l2_norm, recurrent_decay, resize_attention_scores, resize_into,
    rope_frequencies, sigmoid, small_float,
};
use crate::kernels::{
    MatvecBackend, RowParallelism,
    pinned_gdn::{StateStepInput, step_four_rows_state_only_tokens_from},
    prefill::{ProjectionTile, ROW_TASK_SCRATCH_BYTES},
    round_f16_slice,
};
use crate::model::Tensor;
use crate::model::lora::Scratch as AdapterScratch;
use crate::{MAX_PREFILL_BLOCK_TOKENS, Result, invalid};

#[path = "prefill_gdn.rs"]
mod gdn;

#[cfg(test)]
#[path = "prefill_six_tests.rs"]
mod six_tests;

#[cfg(test)]
#[path = "prefill_ffn_tests.rs"]
mod ffn_tests;

#[cfg(test)]
#[path = "prefill_sibling_tests.rs"]
mod sibling_tests;

fn tiled_attention_scratch_bytes(
    geometry: &Geometry,
    arithmetic: Arithmetic,
    policy: super::KvCachePolicy,
    context: usize,
    tokens: usize,
) -> Result<usize> {
    if policy != super::KvCachePolicy::F16 {
        // Packed tokens retain one probability row each while their heads
        // execute serially inside a token task. History stays packed.
        return checked_product(&[tokens, context, size_of::<f32>()]);
    }
    let heads = if matches!(arithmetic, Arithmetic::PinnedSwift)
        && geometry.heads / geometry.kv_heads == 6
    {
        6
    } else {
        4
    };
    checked_product(&[tokens, context, heads, size_of::<f32>()])
}

#[derive(Clone, Copy, Default)]
pub(super) struct TileReuse<'a> {
    basis: Option<&'a Tensor>,
    prepared: Option<&'a Tensor>,
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
    attention_task_outputs: Vec<f32>,
    attention_task_scores: Vec<f32>,
    attention: Vec<Vec<f32>>,
    frequencies: Vec<Vec<(f32, f32)>>,
    gdn_conv_outputs: Vec<f32>,
    gdn_head_outputs: Vec<f32>,
    gdn_key_dots: Vec<f32>,
    packed_kv: super::quantized_kv::PackedTileScratch,
    six_head: super::attention_six::SixHeadTileScratch,
    chunkwise_gdn: bool,
    verification: bool,
    folded: Option<FoldedVerificationState>,
    projector: ProjectionTile,
    statistics: TiledPrefillStatistics,
}

#[derive(Clone, Debug, Default)]
struct FoldedVerificationState {
    pending_tokens: Vec<u32>,
    carried_tokens: Vec<u32>,
    layers: Vec<gdn::FoldedRecurrentLayerTape>,
}

impl FoldedVerificationState {
    fn bytes_for_model(max_tokens: usize, layers: &[Layer], geometry: &Geometry) -> Result<usize> {
        let recurrent_layers = layers
            .iter()
            .filter(|layer| matches!(layer.attention, Attention::Recurrent(_)))
            .count();
        let per_layer = gdn::FoldedRecurrentLayerTape::bytes_for_capacity(max_tokens, geometry)?;
        let token_bytes = max_tokens
            .checked_mul(2)
            .and_then(|v| v.checked_mul(size_of::<u32>()))
            .ok_or_else(|| invalid("folded verification token buffer size overflow"))?;
        recurrent_layers
            .checked_mul(per_layer)
            .and_then(|v| v.checked_add(token_bytes))
            .and_then(|v| v.checked_add(size_of::<Self>()))
            .ok_or_else(|| invalid("folded verification state size overflow"))
    }

    fn new(max_tokens: usize, layers: &[Layer], geometry: &Geometry) -> Result<Self> {
        let recurrent_layers = layers
            .iter()
            .filter(|layer| matches!(layer.attention, Attention::Recurrent(_)))
            .count();
        let mut tapes = Vec::new();
        tapes.try_reserve_exact(recurrent_layers)?;
        for layer in layers {
            if matches!(layer.attention, Attention::Recurrent(_)) {
                tapes.push(gdn::FoldedRecurrentLayerTape::new(max_tokens, geometry)?);
            }
        }
        let mut pending_tokens = Vec::new();
        pending_tokens.try_reserve_exact(max_tokens)?;
        let mut carried_tokens = Vec::new();
        carried_tokens.try_reserve_exact(max_tokens)?;
        Ok(Self {
            pending_tokens,
            carried_tokens,
            layers: tapes,
        })
    }

    fn is_active(&self) -> bool {
        !self.pending_tokens.is_empty()
            || !self.carried_tokens.is_empty()
            || self
                .layers
                .iter()
                .any(|tape| tape.has_pending() || tape.has_carried())
    }

    fn clear_all(&mut self) {
        self.pending_tokens.clear();
        self.carried_tokens.clear();
        for tape in &mut self.layers {
            tape.clear_all();
        }
    }
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
            attention_task_outputs: Vec::new(),
            attention_task_scores: Vec::new(),
            attention: create()?,
            frequencies,
            gdn_conv_outputs: Vec::new(),
            gdn_head_outputs: Vec::new(),
            gdn_key_dots: Vec::new(),
            packed_kv: super::quantized_kv::PackedTileScratch::default(),
            six_head: super::attention_six::SixHeadTileScratch::default(),
            chunkwise_gdn: false,
            verification: false,
            folded: None,
            projector,
            statistics: TiledPrefillStatistics::default(),
        })
    }

    #[cfg(test)]
    pub(super) fn poison_gate_and_attended(&mut self, width: usize) {
        for slot in &mut self.attended {
            slot.resize(width, f32::NAN);
            slot.fill(f32::NAN);
        }
        for slot in &mut self.recurrent_gate {
            slot.resize(width, f32::NAN);
            slot.fill(f32::NAN);
        }
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
            if tile.verification {
                identity.push_str(":cpu-tiled-verify-v1");
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
    /// Rejects unsupported tile token counts (outside 8, 16, 32, or 64 when
    /// enabled by `crate::kernels::prefill::MAX_TILE_TOKENS`) and size overflow.
    pub fn tiled_prefill_scratch_bytes(&self, tokens: usize) -> Result<usize> {
        if !matches!(tokens, 8 | 16 | 32 | 64) || tokens > crate::kernels::prefill::MAX_TILE_TOKENS
        {
            return Err(invalid(
                "CPU prefill tiles must contain 8, 16, 32 or supported 64 tokens",
            ));
        }
        let width = self
            .geometry
            .embedding
            .max(self.geometry.feed_forward)
            .max(self.geometry.convolution_width())
            .max(self.geometry.value_width())
            .max(self.geometry.query_gate_width())
            .max(self.geometry.kv_width());
        // FP16 tiles retain four or six full-context score rows per tile
        // token. Reserve the complete high water independently of the ordinary
        // sequential score allowance and any later worker-policy selection.
        // Rotated tiles retain one full-context probability row per token,
        // with no expanded K/V cache and no nested head-worker submissions.
        let tiled_scores = tiled_attention_scratch_bytes(
            &self.geometry,
            self.arithmetic,
            self.kv_cache_policy,
            self.context_limit,
            tokens,
        )?;
        checked_product(&[tokens, width, 80])?
            .checked_add(checked_product(&[tokens, 2048])?)
            .and_then(|bytes| bytes.checked_add(tiled_scores))
            // The four fixed 32x32 coefficient/product tables and their small
            // temporary vectors exist even for an eight-token, narrow graph.
            .and_then(|bytes| bytes.checked_add(64 * 1024))
            // Fixed task slice/error descriptors stay bounded at 16 workers
            // and MAX_TILE_TOKENS token outputs, including later worker-policy changes.
            .and_then(|bytes| bytes.checked_add(ROW_TASK_SCRATCH_BYTES))
            .ok_or_else(|| invalid("CPU prefill scratch reservation overflow"))
    }

    /// Select experimental CPU tiled state-only prefill before the first token.
    ///
    /// This consumes no new weights and leaves ordinary one-token decode and
    /// default block ingestion unchanged. FFN packed blocks are decoded once
    /// across a tile; attention/GDN/convolution remain strictly causal in their
    /// existing token arithmetic order. This is not a chunkwise WY GDN kernel.
    /// Rotated KV retains its selected codec and ordinary per-token attention;
    /// surrounding projections still share packed weights across the tile.
    /// Separate identity prevents mixing runtime prefix/cache plans. No private
    /// service or application default selects this unqualified source path.
    ///
    /// The explicit reservation covers additional tile storage, and must be
    /// charged by a model owner without reducing existing reserves. Cancellation
    /// checks occur between tokens, layers and packed matrix projections. An
    /// error may leave provisional layer state; as for [`crate::Decoder::ingest_block`],
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
            self.workspace
                .projection
                .matvec
                .clone_prefill_row_parallelism(),
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
    /// Requires explicit tiled prefill selection on an empty graph. Recurrent
    /// GDN layers carry no KV cache, so this is valid on both FP16 and rotated
    /// KV cache policies. This changes floating evaluation order; it does not
    /// promise bit-identical recurrence.
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
        if self.position != 0 || tile.chunkwise_gdn || tile.verification {
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

    /// Additional reservation for at most eight per-position vocabulary rows.
    ///
    /// This is additional to [`Self::tiled_prefill_scratch_bytes`], live state,
    /// checkpoints, worker stacks and ordinary logits. The caller-owned output
    /// rows are charged even when allocation reuse later makes them cheaper.
    ///
    /// # Errors
    /// Rejects absent tiles, triangular GDN and arithmetic overflow.
    pub fn tiled_verification_scratch_bytes(&self) -> Result<usize> {
        let tile = self
            .tiled_prefill
            .as_ref()
            .ok_or_else(|| invalid("verification requires CPU tiled prefill"))?;
        if tile.chunkwise_gdn {
            return Err(invalid(
                "verification requires exact sequential GDN arithmetic",
            ));
        }
        let rows = tile
            .tokens
            .min(crate::speculative::MAX_SPECULATIVE_DRAFT_TOKENS);
        checked_product(&[rows, self.vocabulary, size_of::<f32>()])?
            .checked_add(checked_product(&[rows, size_of::<Vec<f32>>()])?)
            .ok_or_else(|| invalid("verification logits reservation overflow"))
    }

    fn has_released_host_weight(&self) -> bool {
        let executor = self.projection_executor.as_deref();
        let released = |tensor: &Tensor| {
            tensor.data.is_empty()
                || executor.is_some_and(|exec| exec.has_released_host_weight(tensor))
        };
        self.layers.iter().any(|layer| {
            let attn_released = match &layer.attention {
                Attention::Full(full) => {
                    released(&full.query)
                        || released(&full.key)
                        || released(&full.value)
                        || released(&full.output)
                }
                Attention::Recurrent(rec) => {
                    released(&rec.qkv)
                        || released(&rec.gate)
                        || released(&rec.beta)
                        || released(&rec.alpha)
                        || released(&rec.output)
                }
            };
            attn_released || released(&layer.up) || released(&layer.gate) || released(&layer.down)
        })
    }

    fn lacks_tile_weights_for_cpu_fallback(&self) -> bool {
        self.projection_executor
            .as_ref()
            .is_none_or(|exec| !exec.supports_tile_tokens())
            && self.has_released_host_weight()
    }

    /// Explicitly enable tiled target verification on an empty CPU graph.
    ///
    /// Eligible packed layer projections and the vocabulary head share weights
    /// across tokens; GDN/convolution retain causal, original arithmetic order.
    /// Rotated KV retains its selected lossy codec and causal per-token attention;
    /// this does not imply equivalence to the separate FP16 numerical profile.
    /// This selects no drafter, sampler correction or serving default. Rejection
    /// recovery still requires a complete checkpoint and accepted-prefix replay.
    ///
    /// # Errors
    /// Rejects nonempty/already selected graphs, triangular GDN, insufficient
    /// additional reservation and identity-allocation failure.
    pub fn with_tiled_verification(mut self, additional_bytes: usize) -> Result<Self> {
        if self.lacks_tile_weights_for_cpu_fallback() {
            return Err(invalid(
                "tiled verification with non-tile executor requires retained host weights",
            ));
        }
        let required = self.tiled_verification_scratch_bytes()?;
        let verify_capacity = {
            let tile = self
                .tiled_prefill
                .as_ref()
                .ok_or_else(|| invalid("verification requires CPU tiled prefill"))?;
            if self.position != 0 || tile.verification || additional_bytes < required {
                return Err(invalid(
                    "tiled verification requires an empty graph and admitted logits",
                ));
            }
            tile.tokens
                .min(crate::speculative::MAX_SPECULATIVE_DRAFT_TOKENS)
        };
        let folded_state =
            FoldedVerificationState::new(verify_capacity, &self.layers, &self.geometry)?;
        let tile = self
            .tiled_prefill
            .as_mut()
            .ok_or_else(|| invalid("verification requires CPU tiled prefill"))?;
        let total = tile
            .scratch_bytes
            .checked_add(required)
            .ok_or_else(|| invalid("verification scratch reservation overflow"))?;
        let suffix = ":cpu-tiled-verify-v1";
        self.identity.try_reserve_exact(suffix.len())?;
        self.identity.push_str(suffix);
        self.provenance = std::sync::Arc::new(());
        tile.scratch_bytes = total;
        tile.folded = Some(folded_state);
        tile.verification = true;
        Ok(self)
    }

    /// Preallocated heap byte footprint for folded verification recurrent replay tapes.
    ///
    /// # Errors
    /// Rejects models without CPU tiled prefill configured or on arithmetic overflow.
    pub fn folded_verification_tape_bytes(&self) -> Result<usize> {
        let tile = self
            .tiled_prefill
            .as_ref()
            .ok_or_else(|| invalid("folded verification requires CPU tiled prefill"))?;
        let cap = tile
            .tokens
            .min(crate::speculative::MAX_SPECULATIVE_DRAFT_TOKENS);
        FoldedVerificationState::bytes_for_model(cap, &self.layers, &self.geometry)
    }

    pub(super) fn verification_capacity(&self) -> usize {
        self.tiled_prefill
            .as_ref()
            .filter(|tile| tile.verification)
            .map_or(0, |tile| {
                tile.tokens
                    .min(crate::speculative::MAX_SPECULATIVE_DRAFT_TOKENS)
            })
    }

    pub(super) fn supports_folded_verification(&self) -> bool {
        self.tiled_prefill
            .as_ref()
            .is_some_and(|tile| tile.verification && tile.folded.is_some())
    }

    pub(super) fn has_pending_folded_verification(&self) -> bool {
        self.tiled_prefill
            .as_ref()
            .and_then(|tile| tile.folded.as_ref())
            .is_some_and(FoldedVerificationState::is_active)
    }

    pub(super) fn discard_folded_verification(&mut self) {
        if let Some(tile) = self.tiled_prefill.as_mut()
            && let Some(folded) = tile.folded.as_mut()
        {
            folded.clear_all();
        }
    }

    fn prepare_verification_output(
        vocabulary: usize,
        count: usize,
        output: &mut Vec<Vec<f32>>,
    ) -> Result<()> {
        let allocation = (|| -> Result<()> {
            if output.len() > count {
                output.truncate(count);
            }
            for row in &mut *output {
                row.clear();
                if vocabulary > row.capacity() {
                    row.try_reserve_exact(vocabulary)?;
                }
            }
            if count > output.capacity() {
                output.try_reserve_exact(count - output.len())?;
            }
            while output.len() < count {
                let mut row = Vec::new();
                row.try_reserve_exact(vocabulary)?;
                output.push(row);
            }
            Ok(())
        })();
        if let Err(error) = allocation {
            output.clear();
            return Err(error);
        }
        Ok(())
    }

    pub(super) fn evaluate_verification_block(
        &mut self,
        tokens: &[u32],
        output: &mut Vec<Vec<f32>>,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        if self.has_pending_folded_verification() {
            output.clear();
            return Err(invalid(
                "folded verification state must be committed or flushed before standard verification",
            ));
        }
        if !self.state_consistent {
            output.clear();
            return Err(invalid(
                "decoder state is inconsistent after failed evaluation and requires reset or restore",
            ));
        }
        if tokens.is_empty()
            || tokens.len() > self.verification_capacity()
            || self.lacks_tile_weights_for_cpu_fallback()
            || tokens
                .iter()
                .any(|&token| !usize::try_from(token).is_ok_and(|value| value < self.vocabulary))
            || self
                .position
                .checked_add(tokens.len())
                .is_none_or(|end| end > self.context_limit)
        {
            output.clear();
            return Err(invalid(
                "invalid tiled verification bounds, tokens or context",
            ));
        }
        Self::prepare_verification_output(self.vocabulary, tokens.len(), output)?;
        let mut tile = self
            .tiled_prefill
            .take()
            .ok_or_else(|| invalid("missing verification tile"))?;
        let mut executor = self.projection_executor.take();
        let result = (|| {
            check()?;
            let prefill_executor = executor
                .as_mut()
                .map(Self::coerce_executor)
                .filter(|exec| exec.supports_tile_tokens());
            self.ingest_cpu_tile_with_folded(
                tokens,
                check,
                &mut tile,
                None,
                prefill_executor,
                Some(output.as_mut_slice()),
            )?;
            check()
        })();
        self.projection_executor = executor;
        self.tiled_prefill = Some(tile);
        if result.is_err() {
            output.clear();
        }
        result
    }

    fn validate_folded_block_inputs(
        &self,
        carried_tokens: &[u32],
        tokens: &[u32],
        output: &mut Vec<Vec<f32>>,
    ) -> Result<bool> {
        if !self.state_consistent {
            output.clear();
            return Err(invalid(
                "decoder state is inconsistent after failed evaluation and requires reset or restore",
            ));
        }
        if !self.supports_folded_verification() {
            output.clear();
            return Err(invalid(
                "folded verification requires CPU tiled verification support",
            ));
        }
        if tokens.is_empty()
            || tokens.len() > self.verification_capacity()
            || self.lacks_tile_weights_for_cpu_fallback()
            || tokens
                .iter()
                .any(|&token| !usize::try_from(token).is_ok_and(|value| value < self.vocabulary))
            || carried_tokens
                .iter()
                .any(|&token| !usize::try_from(token).is_ok_and(|value| value < self.vocabulary))
        {
            output.clear();
            return Err(invalid(
                "invalid folded verification bounds, carried tokens, or vocabulary",
            ));
        }
        let folded = self
            .tiled_prefill
            .as_ref()
            .and_then(|tile| tile.folded.as_ref())
            .ok_or_else(|| invalid("missing folded verification state"))?;
        if !folded.pending_tokens.is_empty() {
            output.clear();
            return Err(invalid(
                "folded verification block already awaits commit_folded_verification",
            ));
        }
        if folded.carried_tokens.is_empty() {
            return Ok(false);
        }
        if folded.carried_tokens.as_slice() != carried_tokens {
            output.clear();
            return Err(invalid(
                "folded verification carried tokens do not match deferred replay tape",
            ));
        }
        Ok(true)
    }

    fn rollback_failed_folded_block(
        &mut self,
        folded: &mut FoldedVerificationState,
        tile: &mut TiledPrefill,
        start_position: usize,
        start_mrope: usize,
    ) {
        let rollback = (|| -> Result<()> {
            let mut rec_idx = 0_usize;
            for ((layer, state), delta) in self
                .layers
                .iter()
                .zip(&mut self.state)
                .zip(&mut self.workspace.deltas)
            {
                match (&layer.attention, state) {
                    (Attention::Full(_), LayerState::Full { keys, values }) => {
                        keys.truncate_to(start_position)?;
                        values.truncate_to(start_position)?;
                    }
                    (Attention::Recurrent(_), rec_state @ LayerState::Recurrent { .. }) => {
                        if let Some(layer_tape) = folded.layers.get_mut(rec_idx) {
                            gdn::flush_recurrent_carry_layer(
                                layer_tape,
                                rec_state,
                                delta,
                                &self.geometry,
                                self.arithmetic,
                                &tile.projector,
                                &mut tile.gdn_head_outputs,
                            )?;
                        }
                        rec_idx += 1;
                    }
                    _ => {}
                }
            }
            folded.clear_all();
            self.position = start_position;
            self.mrope_position = start_mrope;
            self.state_consistent = true;
            Ok(())
        })();
        if rollback.is_err() {
            folded.clear_all();
            self.state_consistent = false;
        }
    }

    pub(super) fn evaluate_verification_block_folded(
        &mut self,
        carried_tokens: &[u32],
        tokens: &[u32],
        output: &mut Vec<Vec<f32>>,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        let has_carried_tape = self.validate_folded_block_inputs(carried_tokens, tokens, output)?;
        if !has_carried_tape && !carried_tokens.is_empty() {
            if self
                .position
                .checked_add(carried_tokens.len() + tokens.len())
                .is_none_or(|end| end > self.context_limit)
            {
                output.clear();
                return Err(invalid("folded verification exceeds remaining context"));
            }
            self.ingest_prefill_block(carried_tokens, check)?;
        } else if self
            .position
            .checked_add(tokens.len())
            .is_none_or(|end| end > self.context_limit)
        {
            output.clear();
            return Err(invalid("folded verification exceeds remaining context"));
        }

        Self::prepare_verification_output(self.vocabulary, tokens.len(), output)?;
        let start_position = self.position;
        let start_mrope = self.mrope_position;
        let mut tile = self
            .tiled_prefill
            .take()
            .ok_or_else(|| invalid("missing verification tile"))?;
        let mut folded = tile
            .folded
            .take()
            .ok_or_else(|| invalid("missing folded verification state"))?;
        let mut executor = self.projection_executor.take();
        let result = (|| {
            check()?;
            let prefill_executor = executor
                .as_mut()
                .map(Self::coerce_executor)
                .filter(|exec| exec.supports_tile_tokens());
            self.ingest_cpu_tile_with_folded(
                tokens,
                check,
                &mut tile,
                Some(&mut folded),
                prefill_executor,
                Some(output.as_mut_slice()),
            )?;
            check()?;
            folded.carried_tokens.clear();
            folded.pending_tokens.clear();
            folded.pending_tokens.extend_from_slice(tokens);
            Ok(())
        })();
        if result.is_err() {
            output.clear();
            self.rollback_failed_folded_block(&mut folded, &mut tile, start_position, start_mrope);
        }
        tile.folded = Some(folded);
        self.projection_executor = executor;
        self.tiled_prefill = Some(tile);
        result
    }

    pub(super) fn commit_folded_verification(&mut self, kept: usize) -> Result<()> {
        if !self.state_consistent {
            return Err(invalid(
                "decoder state is inconsistent after failed evaluation and requires reset or restore",
            ));
        }
        let rec_count = self
            .layers
            .iter()
            .filter(|layer| matches!(layer.attention, Attention::Recurrent(_)))
            .count();
        let tile = self
            .tiled_prefill
            .as_mut()
            .ok_or_else(|| invalid("missing verification tile"))?;
        let folded = tile
            .folded
            .as_mut()
            .ok_or_else(|| invalid("missing folded verification state"))?;
        let pending = folded.pending_tokens.len();
        if pending == 0 {
            return Err(invalid("no pending folded verification block to commit"));
        }
        if kept > pending {
            return Err(invalid(
                "folded verification kept token count exceeds pending block length",
            ));
        }
        if !folded.carried_tokens.is_empty() {
            return Err(invalid(
                "folded verification carry tape was not consumed before commit",
            ));
        }
        if folded.layers.len() != rec_count {
            return Err(invalid(
                "folded verification recurrent layer count mismatch",
            ));
        }
        if folded.carried_tokens.capacity() < kept {
            return Err(invalid(
                "folded verification carried token buffer lacks required capacity",
            ));
        }
        for layer_tape in &folded.layers {
            layer_tape.validate_pending_promotion(kept)?;
        }
        let start_position = self
            .position
            .checked_sub(pending)
            .ok_or_else(|| invalid("folded verification position underflow"))?;
        let start_mrope = self
            .mrope_position
            .checked_sub(pending)
            .ok_or_else(|| invalid("folded verification MRoPE underflow"))?;
        let target_position = start_position + kept;
        let target_mrope = start_mrope + kept;

        self.state_consistent = false;
        let result = (|| -> Result<()> {
            for state in &mut self.state {
                if let LayerState::Full { keys, values } = state {
                    keys.truncate_to(target_position)?;
                    values.truncate_to(target_position)?;
                }
            }
            self.position = target_position;
            self.mrope_position = target_mrope;
            for layer_tape in &mut folded.layers {
                layer_tape.promote_pending_to_carried(kept)?;
            }
            folded.carried_tokens.clear();
            if kept > 0 {
                folded
                    .carried_tokens
                    .extend_from_slice(&folded.pending_tokens[..kept]);
            }
            folded.pending_tokens.clear();
            Ok(())
        })();
        if result.is_err() {
            folded.clear_all();
            return result;
        }
        self.state_consistent = true;
        Ok(())
    }

    pub(super) fn flush_folded_verification(
        &mut self,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        if !self.state_consistent {
            return Err(invalid(
                "decoder state is inconsistent after failed evaluation and requires reset or restore",
            ));
        }
        if !self.has_pending_folded_verification() {
            return Ok(());
        }
        if self
            .tiled_prefill
            .as_ref()
            .and_then(|tile| tile.folded.as_ref())
            .is_some_and(|folded| !folded.pending_tokens.is_empty())
        {
            return Err(invalid(
                "cannot flush uncommitted folded verification block before commit",
            ));
        }
        check()?;
        let mut tile = self
            .tiled_prefill
            .take()
            .ok_or_else(|| invalid("missing verification tile"))?;
        let mut folded = tile
            .folded
            .take()
            .ok_or_else(|| invalid("missing folded verification state"))?;
        let result = Self::flush_folded_recurrent_layers(
            &self.layers,
            &mut self.state,
            &mut self.workspace.deltas,
            &self.geometry,
            self.arithmetic,
            &mut tile,
            &mut folded,
        );
        if result.is_err() {
            folded.clear_all();
            self.state_consistent = false;
        }
        tile.folded = Some(folded);
        self.tiled_prefill = Some(tile);
        result
    }

    fn flush_folded_recurrent_layers(
        layers: &[Layer],
        states: &mut [LayerState],
        deltas: &mut [LayerDelta],
        geometry: &Geometry,
        arithmetic: Arithmetic,
        tile: &mut TiledPrefill,
        folded: &mut FoldedVerificationState,
    ) -> Result<()> {
        let rec_count = layers
            .iter()
            .filter(|layer| matches!(layer.attention, Attention::Recurrent(_)))
            .count();
        if tile.projector.row_parallelism().workers() > 1
            && folded.layers.len() >= 4
            && folded.layers.len() == rec_count
        {
            let mut tasks = super::capacity(rec_count)?;
            let mut layer_tapes = folded.layers.iter_mut();
            for ((layer, state), delta) in layers.iter().zip(states).zip(deltas) {
                if matches!(layer.attention, Attention::Recurrent(_)) {
                    let layer_tape = layer_tapes.next().ok_or_else(|| {
                        invalid("folded verification recurrent layer tape missing")
                    })?;
                    tasks.push((state, delta, layer_tape));
                }
            }
            tile.projector
                .row_parallelism()
                .update_chunks(&mut tasks, 1, |_, chunk| {
                    let serial_projector = ProjectionTile::default();
                    let mut unused_outputs = Vec::new();
                    for (state, delta, layer_tape) in chunk {
                        gdn::flush_recurrent_carry_layer(
                            layer_tape,
                            state,
                            delta,
                            geometry,
                            arithmetic,
                            &serial_projector,
                            &mut unused_outputs,
                        )?;
                    }
                    Ok(())
                })?;
        } else {
            let mut rec_idx = 0_usize;
            for ((layer, state), delta) in layers.iter().zip(states).zip(deltas) {
                if matches!(layer.attention, Attention::Recurrent(_)) {
                    let layer_tape = folded.layers.get_mut(rec_idx).ok_or_else(|| {
                        invalid("folded verification recurrent layer tape missing")
                    })?;
                    rec_idx += 1;
                    gdn::flush_recurrent_carry_layer(
                        layer_tape,
                        state,
                        delta,
                        geometry,
                        arithmetic,
                        &tile.projector,
                        &mut tile.gdn_head_outputs,
                    )?;
                }
            }
        }
        folded.carried_tokens.clear();
        Ok(())
    }

    pub(super) fn ingest_prefill_block(
        &mut self,
        tokens: &[u32],
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        let mut executor = self.projection_executor.take();
        let prefill_executor = if self.tiled_prefill.is_some() {
            executor
                .as_mut()
                .map(Self::coerce_executor)
                .filter(|exec| exec.supports_tile_tokens())
        } else {
            executor.as_mut().map(Self::coerce_executor)
        };
        let result = self.ingest_prefill_block_with_executor(tokens, check, prefill_executor);
        self.projection_executor = executor;
        result
    }

    fn validate_prefill_block_bounds(&self, tokens: &[u32]) -> Result<()> {
        if self.has_pending_folded_verification() {
            return Err(invalid(
                "folded verification state must be committed or flushed before prefill",
            ));
        }
        if !self.state_consistent {
            return Err(invalid(
                "decoder state is inconsistent after failed evaluation and requires reset or restore",
            ));
        }
        if tokens.is_empty()
            || tokens.len() > MAX_PREFILL_BLOCK_TOKENS
            || tokens
                .iter()
                .any(|&token| !usize::try_from(token).is_ok_and(|value| value < self.vocabulary))
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
        Ok(())
    }

    pub(super) fn ingest_prefill_block_and_evaluate_last_into(
        &mut self,
        tokens: &[u32],
        output: &mut Vec<f32>,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        let mut executor = self.projection_executor.take();
        let use_executor_on_tiles = executor
            .as_deref()
            .is_some_and(crate::kernels::Pq2ProjectionExecutor::supports_tile_tokens);
        let result = self.ingest_prefill_block_and_evaluate_last_with_executor(
            tokens,
            output,
            check,
            executor.as_mut().map(Self::coerce_executor),
            use_executor_on_tiles,
        );
        self.projection_executor = executor;
        result
    }

    pub(super) fn ingest_prefill_block_and_evaluate_last_with_executor(
        &mut self,
        tokens: &[u32],
        output: &mut Vec<f32>,
        check: &dyn Fn() -> Result<()>,
        mut executor: Option<&mut (dyn crate::kernels::Pq2ProjectionExecutor + '_)>,
        use_executor_on_tiles: bool,
    ) -> Result<()> {
        if let Err(error) = self.validate_prefill_block_bounds(tokens) {
            output.clear();
            return Err(error);
        }
        let Some(mut tile) = self.tiled_prefill.take() else {
            let Some((&last, prefix)) = tokens.split_last() else {
                output.clear();
                return Err(invalid("invalid native prefill block bounds or tokens"));
            };
            let result = (|| {
                for &token in prefix {
                    check()?;
                    self.evaluate_token_with_executor(token, None, executor.as_deref_mut(), None)?;
                    check()?;
                }
                check()?;
                self.evaluate_token_with_executor(last, Some(output), executor, None)?;
                check()
            })();
            if result.is_err() {
                output.clear();
            }
            return result;
        };
        let result = (|| {
            let mut chunks = tokens.chunks(tile.tokens).peekable();
            while let Some(chunk) = chunks.next() {
                let is_last_chunk = chunks.peek().is_none();
                check()?;
                if !is_last_chunk {
                    let tile_executor = if use_executor_on_tiles {
                        executor.as_deref_mut()
                    } else {
                        None
                    };
                    self.ingest_cpu_tile_with_folded(
                        chunk,
                        check,
                        &mut tile,
                        None,
                        tile_executor,
                        None,
                    )?;
                    tile.statistics.completed_tiles =
                        tile.statistics.completed_tiles.saturating_add(1);
                    tile.statistics.completed_tokens = tile
                        .statistics
                        .completed_tokens
                        .saturating_add(u64::try_from(chunk.len())?);
                } else if chunk.len() == 1 {
                    self.evaluate_token_with_executor(chunk[0], Some(output), executor, None)?;
                    check()?;
                    break;
                } else {
                    self.ingest_cpu_tile_and_evaluate_last(
                        chunk,
                        check,
                        &mut tile,
                        executor,
                        use_executor_on_tiles,
                        output,
                    )?;
                    tile.statistics.completed_tiles =
                        tile.statistics.completed_tiles.saturating_add(1);
                    tile.statistics.completed_tokens = tile
                        .statistics
                        .completed_tokens
                        .saturating_add(u64::try_from(chunk.len())?);
                    check()?;
                    break;
                }
                check()?;
            }
            Ok(())
        })();
        self.tiled_prefill = Some(tile);
        if result.is_err() {
            output.clear();
        }
        result
    }

    pub(super) fn ingest_prefill_block_with_executor(
        &mut self,
        tokens: &[u32],
        check: &dyn Fn() -> Result<()>,
        mut executor: Option<&mut (dyn crate::kernels::Pq2ProjectionExecutor + '_)>,
    ) -> Result<()> {
        self.validate_prefill_block_bounds(tokens)?;
        let Some(mut tile) = self.tiled_prefill.take() else {
            for &token in tokens {
                check()?;
                self.evaluate_token_with_executor(token, None, executor.as_deref_mut(), None)?;
                check()?;
            }
            return Ok(());
        };
        let result = (|| {
            for tokens in tokens.chunks(tile.tokens) {
                check()?;
                self.ingest_cpu_tile_with_folded(
                    tokens,
                    check,
                    &mut tile,
                    None,
                    executor.as_deref_mut(),
                    None,
                )?;
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

    fn embed_tile_tokens(
        &self,
        tokens: &[u32],
        tile: &mut TiledPrefill,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        let count = tokens.len();
        if count > 1 && tile.projector.row_parallelism().workers() > 1 {
            let single_worker = RowParallelism::default();
            let transforms = &self.transforms;
            let inverse = transforms.inverse_embedding;
            match &self.embedding {
                super::EmbeddingSource::Plain(tensor) => {
                    for _ in tokens {
                        check()?;
                    }
                    tile.projector.row_parallelism().update_chunks(
                        &mut tile.hidden[..count],
                        1,
                        |base_tok, hidden_chunk| {
                            for (offset, hidden) in hidden_chunk.iter_mut().enumerate() {
                                super::row_into(
                                    tensor,
                                    usize::try_from(tokens[base_tok + offset])?,
                                    hidden,
                                )?;
                                if inverse {
                                    transforms.rotate_with_parallelism(hidden, &single_worker)?;
                                    transforms.sign(hidden);
                                }
                            }
                            Ok(())
                        },
                    )?;
                }
                super::EmbeddingSource::Trellis(_) | super::EmbeddingSource::TrellisBanded(_) => {
                    for (token, hidden) in tokens.iter().zip(&mut tile.hidden[..count]) {
                        check()?;
                        self.embedding.row_into(
                            usize::try_from(*token)?,
                            hidden,
                            &mut tile.projector.trellis.tmp,
                        )?;
                    }
                    if inverse {
                        tile.projector.row_parallelism().update_chunks(
                            &mut tile.hidden[..count],
                            1,
                            |_, hidden_chunk| {
                                for hidden in hidden_chunk {
                                    transforms.rotate_with_parallelism(hidden, &single_worker)?;
                                    transforms.sign(hidden);
                                }
                                Ok(())
                            },
                        )?;
                    }
                }
            }
            return Ok(());
        }
        for (token, hidden) in tokens.iter().zip(&mut tile.hidden) {
            check()?;
            self.embedding.row_into(
                usize::try_from(*token)?,
                hidden,
                &mut tile.projector.trellis.tmp,
            )?;
            if self.transforms.inverse_embedding {
                self.transforms
                    .rotate_with_parallelism(hidden, tile.projector.row_parallelism())?;
                self.transforms.sign(hidden);
            }
        }
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Keep causal FFN range execution, compaction compensation and deferred residual flag explicit."
    )]
    fn run_tile_ffn_range(
        transforms: &Transforms,
        geometry: &Geometry,
        arithmetic: Arithmetic,
        adapters: ActiveAdapters<'_>,
        adapter_scratch: &mut AdapterScratch,
        projection_scratch: &mut crate::kernels::MatvecScratch,
        layer_idx: usize,
        layer: &Layer,
        tile: &mut TiledPrefill,
        start: usize,
        end: usize,
        mut executor: Option<&mut (dyn crate::kernels::Pq2ProjectionExecutor + '_)>,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<bool> {
        let post_norm_reuse = add_normalize_for_tile_into(
            transforms,
            geometry,
            arithmetic,
            &layer.post_norm,
            &layer.gate,
            &mut tile.hidden[start..end],
            &tile.attention[start..end],
            &mut tile.normalized[start..end],
            &mut tile.transformed[start..end],
            tile.projector.row_parallelism(),
        )?;
        check()?;
        let compacted_layer = transforms.is_compacted_layer(layer_idx);
        let ffn_compensation = transforms.compaction_compensation(layer_idx);
        let chain_ran = !compacted_layer
            && ffn_compensation.is_none()
            && try_project_tile_ffn_chain(
                transforms,
                geometry,
                arithmetic,
                adapters,
                post_norm_reuse,
                [&layer.gate, &layer.up, &layer.down],
                &tile.normalized[start..end],
                Some(&tile.hidden[start..end]),
                &mut tile.attention[start..end],
                &mut tile.transformed[start..end],
                &tile.projector,
                executor.as_deref_mut(),
                check,
            )?;
        if chain_ran {
            for (hidden, output) in tile.hidden[start..end]
                .iter_mut()
                .zip(&mut tile.attention[start..end])
            {
                std::mem::swap(hidden, output);
                finite(hidden, "CPU tiled decoder layer output")?;
            }
            return Ok(false);
        }
        let ffn_reuse = project_tile_ffn_swiglu(
            transforms,
            geometry,
            arithmetic,
            adapters,
            adapter_scratch,
            projection_scratch,
            post_norm_reuse,
            &layer.gate,
            &layer.up,
            &layer.down,
            &tile.normalized[start..end],
            &mut tile.gate[start..end],
            &mut tile.up[start..end],
            &mut tile.transformed[start..end],
            &mut tile.projector,
            executor.as_deref_mut(),
            check,
        )?;
        check()?;
        let _ = project_tile(
            transforms,
            geometry,
            arithmetic,
            adapters,
            adapter_scratch,
            projection_scratch,
            ffn_reuse,
            &layer.down,
            &tile.up[start..end],
            &mut tile.attention[start..end],
            &mut tile.transformed[start..end],
            &mut tile.projector,
            executor,
            check,
        )?;
        if let Some(compensation) = ffn_compensation {
            for (norm_in, branch) in tile.normalized[start..end]
                .iter()
                .zip(&mut tile.attention[start..end])
            {
                compensation.apply(norm_in, branch)?;
            }
        }
        Ok(true)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Keep full causal layer attention execution explicit across leading and final tiles."
    )]
    #[allow(
        clippy::too_many_lines,
        reason = "Keep full causal layer attention execution unified across recurrent and full layers."
    )]
    fn run_full_tile_attention(
        geometry: &Geometry,
        arithmetic: Arithmetic,
        transforms: &Transforms,
        adapters: ActiveAdapters<'_>,
        projection: &mut super::ProjectionScratch,
        adapter_scratch: &mut AdapterScratch,
        attention_scratch: &mut super::AttentionScratch,
        layer: &Layer,
        state: &mut LayerState,
        delta: &mut LayerDelta,
        base: usize,
        mrope_base: usize,
        count: usize,
        frequencies_ready: &mut bool,
        pending_residual: &mut bool,
        tile: &mut TiledPrefill,
        folded_tape: Option<&mut gdn::FoldedRecurrentLayerTape>,
        mut executor: Option<&mut (dyn crate::kernels::Pq2ProjectionExecutor + '_)>,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        let first_attn_target = match &layer.attention {
            Attention::Recurrent(weights) => &weights.qkv,
            Attention::Full(weights) => &weights.query,
        };
        let norm_reuse = normalize_or_add_tile_into(
            *pending_residual,
            transforms,
            geometry,
            arithmetic,
            &layer.norm,
            first_attn_target,
            &mut tile.hidden[..count],
            &tile.attention[..count],
            &mut tile.normalized[..count],
            &mut tile.transformed[..count],
            tile.projector.row_parallelism(),
        )?;
        *pending_residual = false;
        match (&layer.attention, state) {
            (Attention::Recurrent(weights), state @ LayerState::Recurrent { .. }) => {
                if let Some(layer_tape) = folded_tape {
                    gdn::flush_recurrent_carry_layer(
                        layer_tape,
                        state,
                        delta,
                        geometry,
                        arithmetic,
                        &tile.projector,
                        &mut tile.gdn_head_outputs,
                    )?;
                    gdn::recurrent_tile_folded(
                        weights,
                        state,
                        delta,
                        geometry,
                        arithmetic,
                        transforms,
                        adapters,
                        adapter_scratch,
                        &mut projection.matvec,
                        norm_reuse,
                        count,
                        check,
                        tile,
                        executor,
                        layer_tape,
                    )?;
                } else {
                    gdn::recurrent_tile(
                        weights,
                        state,
                        delta,
                        geometry,
                        arithmetic,
                        transforms,
                        adapters,
                        adapter_scratch,
                        &mut projection.matvec,
                        norm_reuse,
                        count,
                        check,
                        tile,
                        executor,
                    )?;
                }
            }
            (Attention::Full(weights), LayerState::Full { keys, values }) => {
                check()?;
                let _ = project_tile_triple(
                    transforms,
                    geometry,
                    arithmetic,
                    adapters,
                    adapter_scratch,
                    &mut projection.matvec,
                    norm_reuse,
                    [&weights.query, &weights.key, &weights.value],
                    &tile.normalized[..count],
                    &mut tile.mixed[..count],
                    &mut tile.recurrent_gate[..count],
                    &mut tile.beta[..count],
                    &mut tile.transformed[..count],
                    &mut tile.projector,
                    executor.as_deref_mut(),
                    check,
                )?;
                full_attention_tile(
                    weights,
                    geometry,
                    arithmetic,
                    transforms,
                    base,
                    mrope_base,
                    count,
                    frequencies_ready,
                    keys,
                    values,
                    projection,
                    adapters,
                    adapter_scratch,
                    attention_scratch,
                    tile,
                    check,
                )?;
                check()?;
                let _ = project_tile(
                    transforms,
                    geometry,
                    arithmetic,
                    adapters,
                    adapter_scratch,
                    &mut projection.matvec,
                    TileReuse::default(),
                    &weights.output,
                    &tile.attended[..count],
                    &mut tile.attention[..count],
                    &mut tile.transformed[..count],
                    &mut tile.projector,
                    executor,
                    check,
                )?;
            }
            _ => return Err(invalid("CPU tiled prefill layer/cache mismatch")),
        }
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Keep full causal layer attention + FFN execution explicit across leading and final tiles."
    )]
    fn run_full_tile_layer(
        geometry: &Geometry,
        arithmetic: Arithmetic,
        transforms: &Transforms,
        adapters: ActiveAdapters<'_>,
        projection: &mut super::ProjectionScratch,
        adapter_scratch: &mut AdapterScratch,
        attention_scratch: &mut super::AttentionScratch,
        layer_idx: usize,
        layer: &Layer,
        state: &mut LayerState,
        delta: &mut LayerDelta,
        base: usize,
        mrope_base: usize,
        count: usize,
        frequencies_ready: &mut bool,
        pending_residual: &mut bool,
        tile: &mut TiledPrefill,
        folded_tape: Option<&mut gdn::FoldedRecurrentLayerTape>,
        mut executor: Option<&mut (dyn crate::kernels::Pq2ProjectionExecutor + '_)>,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        Self::run_full_tile_attention(
            geometry,
            arithmetic,
            transforms,
            adapters,
            projection,
            adapter_scratch,
            attention_scratch,
            layer,
            state,
            delta,
            base,
            mrope_base,
            count,
            frequencies_ready,
            pending_residual,
            tile,
            folded_tape,
            executor.as_deref_mut(),
            check,
        )?;
        for _ in 0..count {
            check()?;
        }
        *pending_residual = Self::run_tile_ffn_range(
            transforms,
            geometry,
            arithmetic,
            adapters,
            adapter_scratch,
            &mut projection.matvec,
            layer_idx,
            layer,
            tile,
            0,
            count,
            executor,
            check,
        )?;
        check()
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Keep causal output-head normalization, projection and finite verification explicit."
    )]
    #[allow(
        clippy::too_many_lines,
        reason = "Keep plain and banded output-head range evaluation in one explicit path."
    )]
    fn evaluate_tile_output_head_range(
        output_head: &super::OutputHead,
        output_norm: &[f32],
        vocabulary: usize,
        transforms: &Transforms,
        geometry: &Geometry,
        arithmetic: Arithmetic,
        adapters: ActiveAdapters<'_>,
        adapter_scratch: &mut AdapterScratch,
        projection_scratch: &mut crate::kernels::MatvecScratch,
        pending_residual: bool,
        tile: &mut TiledPrefill,
        start: usize,
        end: usize,
        logits: &mut [Vec<f32>],
        executor: Option<&mut (dyn crate::kernels::Pq2ProjectionExecutor + '_)>,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        if let super::OutputHead::Plain(output_tensor) = output_head {
            let reuse = normalize_or_add_tile_into(
                pending_residual,
                transforms,
                geometry,
                arithmetic,
                output_norm,
                output_tensor,
                &mut tile.hidden[start..end],
                &tile.attention[start..end],
                &mut tile.normalized[start..end],
                &mut tile.transformed[start..end],
                tile.projector.row_parallelism(),
            )?;
            check()?;
            let _ = project_tile(
                transforms,
                geometry,
                arithmetic,
                adapters,
                adapter_scratch,
                projection_scratch,
                reuse,
                output_tensor,
                &tile.normalized[start..end],
                logits,
                &mut tile.transformed[start..end],
                &mut tile.projector,
                executor,
                check,
            )?;
        } else {
            if pending_residual {
                for (hidden, branch) in tile.hidden[start..end]
                    .iter_mut()
                    .zip(&tile.attention[start..end])
                {
                    add_finite(hidden, branch, "CPU tiled decoder layer output")?;
                }
            }
            super::normalize_tile_plain_into(
                geometry,
                arithmetic,
                output_norm,
                &tile.hidden[start..end],
                &mut tile.normalized[start..end],
                tile.projector.row_parallelism(),
            )?;
            check()?;
            let backend = tile.projector.requested_backend();
            let (row_parallelism, trellis) = tile.projector.row_parallelism_and_trellis();
            match output_head {
                super::OutputHead::Trellis(weight) if logits.len() == 1 => {
                    crate::kernels::trellis::trellis_matvec_into(
                        weight,
                        &tile.normalized[start],
                        &mut logits[0],
                        backend,
                        row_parallelism,
                        trellis,
                    )?;
                }
                super::OutputHead::Trellis(weight) => {
                    crate::kernels::trellis::trellis_matmul_tokens_into(
                        weight,
                        &tile.normalized[start..end],
                        logits,
                        backend,
                        row_parallelism,
                        trellis,
                    )?;
                }
                super::OutputHead::TrellisBanded(head) if logits.len() == 1 => {
                    head.matvec_into(
                        &tile.normalized[start],
                        &mut logits[0],
                        backend,
                        row_parallelism,
                        trellis,
                    )?;
                }
                super::OutputHead::TrellisBanded(head) => {
                    head.matmul_tokens_into(
                        &tile.normalized[start..end],
                        logits,
                        backend,
                        row_parallelism,
                        trellis,
                    )?;
                }
                super::OutputHead::Plain(_) => {}
            }
        }
        for row in logits {
            if row.len() != vocabulary {
                return Err(invalid("invalid tiled verification vocabulary extent"));
            }
            finite(row, "tiled verification output logits")?;
            check()?;
        }
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "Keep last-layer narrow 0..T-1 KV/GDN update and full T-1 evaluation ordered."
    )]
    fn run_last_layer_and_evaluate_last(
        geometry: &Geometry,
        arithmetic: Arithmetic,
        transforms: &Transforms,
        adapters: ActiveAdapters<'_>,
        projection: &mut super::ProjectionScratch,
        adapter_scratch: &mut AdapterScratch,
        attention_scratch: &mut super::AttentionScratch,
        last_idx: usize,
        layer: &Layer,
        state: &mut LayerState,
        delta: &mut LayerDelta,
        base: usize,
        mrope_base: usize,
        count: usize,
        frequencies_ready: &mut bool,
        pending_residual: &mut bool,
        tile: &mut TiledPrefill,
        mut executor: Option<&mut (dyn crate::kernels::Pq2ProjectionExecutor + '_)>,
        use_executor_on_tiles: bool,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        let last_tok = count - 1;
        let first_attn_target = match &layer.attention {
            Attention::Full(weights) => &weights.key,
            Attention::Recurrent(weights) => &weights.qkv,
        };
        let norm_reuse = normalize_or_add_tile_into(
            *pending_residual,
            transforms,
            geometry,
            arithmetic,
            &layer.norm,
            first_attn_target,
            &mut tile.hidden[..count],
            &tile.attention[..count],
            &mut tile.normalized[..count],
            &mut tile.transformed[..count],
            tile.projector.row_parallelism(),
        )?;
        *pending_residual = false;
        match (&layer.attention, state) {
            (Attention::Full(weights), LayerState::Full { keys, values }) => {
                keys.invalidate_prepared();
                values.invalidate_prepared();
                check()?;
                let kv_reuse = project_tile_pair(
                    transforms,
                    geometry,
                    arithmetic,
                    adapters,
                    adapter_scratch,
                    &mut projection.matvec,
                    norm_reuse,
                    [&weights.key, &weights.value],
                    &tile.normalized[..count],
                    &mut tile.recurrent_gate[..count],
                    &mut tile.beta[..count],
                    &mut tile.transformed[..count],
                    &mut tile.projector,
                    if use_executor_on_tiles {
                        executor.as_deref_mut()
                    } else {
                        None
                    },
                    check,
                )?;
                check()?;
                let query_reuse = if use_executor_on_tiles || executor.is_none() {
                    kv_reuse
                } else {
                    TileReuse {
                        basis: kv_reuse.basis,
                        prepared: None,
                    }
                };
                let _ = project_tile(
                    transforms,
                    geometry,
                    arithmetic,
                    adapters,
                    adapter_scratch,
                    &mut projection.matvec,
                    query_reuse,
                    &weights.query,
                    &tile.normalized[last_tok..count],
                    &mut tile.mixed[last_tok..count],
                    &mut tile.transformed[last_tok..count],
                    &mut tile.projector,
                    executor.as_deref_mut(),
                    check,
                )?;
                check()?;
                if !*frequencies_ready {
                    for (index, frequencies) in tile.frequencies.iter_mut().enumerate().take(count)
                    {
                        rope_frequencies(geometry, mrope_base + index, frequencies)?;
                    }
                    *frequencies_ready = true;
                }
                full_attention_kv_only_tile(
                    weights,
                    geometry,
                    arithmetic,
                    mrope_base,
                    last_tok,
                    frequencies_ready,
                    keys,
                    values,
                    tile,
                    check,
                )?;
                std::mem::swap(
                    &mut attention_scratch.frequencies,
                    &mut tile.frequencies[last_tok],
                );
                let step_result = {
                    let mut evaluation = Evaluation {
                        geometry,
                        arithmetic,
                        transforms,
                        position: base + last_tok,
                        mrope_positions: [
                            mrope_base + last_tok,
                            mrope_base + last_tok,
                            mrope_base + last_tok,
                            0,
                        ],
                        frequencies_ready: true,
                        projection,
                        adapters,
                        adapter_scratch,
                        executor: None,
                    };
                    evaluation.full_attention_state(
                        weights,
                        &tile.mixed[last_tok],
                        &mut tile.recurrent_gate[last_tok],
                        &mut tile.beta[last_tok],
                        keys,
                        values,
                        &mut attention_scratch.query,
                        &mut attention_scratch.scores,
                        &mut attention_scratch.packed_probs,
                        &mut attention_scratch.frequencies,
                        &mut tile.attended[last_tok],
                    )
                };
                std::mem::swap(
                    &mut attention_scratch.frequencies,
                    &mut tile.frequencies[last_tok],
                );
                step_result?;
                keys.prepare_staged_append(&tile.recurrent_gate[last_tok])?;
                values.prepare_staged_append(&tile.beta[last_tok])?;
                keys.append(&tile.recurrent_gate[last_tok]);
                values.append(&tile.beta[last_tok]);
                check()?;
                let _ = project_tile(
                    transforms,
                    geometry,
                    arithmetic,
                    adapters,
                    adapter_scratch,
                    &mut projection.matvec,
                    TileReuse::default(),
                    &weights.output,
                    &tile.attended[last_tok..count],
                    &mut tile.attention[last_tok..count],
                    &mut tile.transformed[last_tok..count],
                    &mut tile.projector,
                    executor.as_deref_mut(),
                    check,
                )?;
            }
            (
                Attention::Recurrent(weights),
                LayerState::Recurrent {
                    convolution: history,
                    matrix: previous,
                },
            ) => {
                check()?;
                let qkv_reuse = project_tile(
                    transforms,
                    geometry,
                    arithmetic,
                    adapters,
                    adapter_scratch,
                    &mut projection.matvec,
                    norm_reuse,
                    &weights.qkv,
                    &tile.normalized[..count],
                    &mut tile.mixed[..count],
                    &mut tile.transformed[..count],
                    &mut tile.projector,
                    if use_executor_on_tiles {
                        executor.as_deref_mut()
                    } else {
                        None
                    },
                    check,
                )?;
                check()?;
                let ba_reuse = project_tile_pair(
                    transforms,
                    geometry,
                    arithmetic,
                    adapters,
                    adapter_scratch,
                    &mut projection.matvec,
                    qkv_reuse,
                    [&weights.beta, &weights.alpha],
                    &tile.normalized[..count],
                    &mut tile.beta[..count],
                    &mut tile.alpha[..count],
                    &mut tile.transformed[..count],
                    &mut tile.projector,
                    if use_executor_on_tiles {
                        executor.as_deref_mut()
                    } else {
                        None
                    },
                    check,
                )?;
                check()?;
                let gate_reuse = if use_executor_on_tiles || executor.is_none() {
                    ba_reuse
                } else {
                    TileReuse {
                        basis: ba_reuse.basis,
                        prepared: None,
                    }
                };
                let _ = project_tile(
                    transforms,
                    geometry,
                    arithmetic,
                    adapters,
                    adapter_scratch,
                    &mut projection.matvec,
                    gate_reuse,
                    &weights.gate,
                    &tile.normalized[last_tok..count],
                    &mut tile.recurrent_gate[last_tok..count],
                    &mut tile.transformed[last_tok..count],
                    &mut tile.projector,
                    executor.as_deref_mut(),
                    check,
                )?;
                check()?;
                {
                    let LayerDelta::Recurrent {
                        convolution,
                        matrix,
                    } = delta
                    else {
                        return Err(invalid("CPU tiled prefill layer/cache mismatch"));
                    };
                    resize_into(history.len(), convolution)?;
                    resize_into(previous.len(), matrix)?;
                    let conv_head_major = step_recurrent_state_only_conv(
                        weights,
                        geometry,
                        arithmetic,
                        history,
                        convolution,
                        tile,
                        last_tok,
                        check,
                    )?;
                    step_recurrent_state_only_matrix(
                        geometry,
                        arithmetic,
                        tile,
                        last_tok,
                        conv_head_major,
                        previous,
                        matrix,
                    )?;
                    check()?;
                    std::mem::swap(history, convolution);
                    std::mem::swap(previous, matrix);
                }
                {
                    let evaluation = Evaluation {
                        geometry,
                        arithmetic,
                        transforms,
                        position: base + last_tok,
                        mrope_positions: [
                            mrope_base + last_tok,
                            mrope_base + last_tok,
                            mrope_base + last_tok,
                            0,
                        ],
                        frequencies_ready: *frequencies_ready,
                        projection,
                        adapters,
                        adapter_scratch,
                        executor: None,
                    };
                    evaluation.recurrent_attention_state(
                        weights,
                        &mut tile.mixed[last_tok],
                        Some(&tile.recurrent_gate[last_tok]),
                        &tile.beta[last_tok],
                        &tile.alpha[last_tok],
                        history,
                        previous,
                        delta,
                        Some(&mut tile.attended[last_tok]),
                    )?;
                    let LayerDelta::Recurrent {
                        convolution,
                        matrix,
                    } = delta
                    else {
                        return Err(invalid("CPU tiled prefill layer/cache mismatch"));
                    };
                    std::mem::swap(history, convolution);
                    std::mem::swap(previous, matrix);
                }
                check()?;
                let _ = project_tile(
                    transforms,
                    geometry,
                    arithmetic,
                    adapters,
                    adapter_scratch,
                    &mut projection.matvec,
                    TileReuse::default(),
                    &weights.output,
                    &tile.attended[last_tok..count],
                    &mut tile.attention[last_tok..count],
                    &mut tile.transformed[last_tok..count],
                    &mut tile.projector,
                    executor.as_deref_mut(),
                    check,
                )?;
            }
            _ => return Err(invalid("CPU tiled prefill layer/cache mismatch")),
        }
        check()?;
        *pending_residual = Self::run_tile_ffn_range(
            transforms,
            geometry,
            arithmetic,
            adapters,
            adapter_scratch,
            &mut projection.matvec,
            last_idx,
            layer,
            tile,
            last_tok,
            count,
            executor,
            check,
        )?;
        check()
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Keep narrow last-layer KV/GDN state update explicit across leading tiles."
    )]
    fn run_narrow_last_layer_tile(
        geometry: &Geometry,
        arithmetic: Arithmetic,
        transforms: &Transforms,
        adapters: ActiveAdapters<'_>,
        adapter_scratch: &mut AdapterScratch,
        projection_scratch: &mut crate::kernels::MatvecScratch,
        layer: &Layer,
        state: &mut LayerState,
        delta: &mut LayerDelta,
        mrope_base: usize,
        count: usize,
        frequencies_ready: &mut bool,
        pending_residual: &mut bool,
        tile: &mut TiledPrefill,
        executor: Option<&mut (dyn crate::kernels::Pq2ProjectionExecutor + '_)>,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        let first_attn_target = match &layer.attention {
            Attention::Recurrent(weights) => &weights.qkv,
            Attention::Full(weights) => &weights.key,
        };
        let norm_reuse = normalize_or_add_tile_into(
            *pending_residual,
            transforms,
            geometry,
            arithmetic,
            &layer.norm,
            first_attn_target,
            &mut tile.hidden[..count],
            &tile.attention[..count],
            &mut tile.normalized[..count],
            &mut tile.transformed[..count],
            tile.projector.row_parallelism(),
        )?;
        *pending_residual = false;
        match (&layer.attention, state) {
            (Attention::Recurrent(weights), state @ LayerState::Recurrent { .. }) => {
                recurrent_state_only_tile(
                    weights,
                    state,
                    delta,
                    geometry,
                    arithmetic,
                    transforms,
                    adapters,
                    adapter_scratch,
                    projection_scratch,
                    norm_reuse,
                    count,
                    check,
                    tile,
                    executor,
                )?;
            }
            (Attention::Full(weights), LayerState::Full { keys, values }) => {
                check()?;
                let _ = project_tile_pair(
                    transforms,
                    geometry,
                    arithmetic,
                    adapters,
                    adapter_scratch,
                    projection_scratch,
                    norm_reuse,
                    [&weights.key, &weights.value],
                    &tile.normalized[..count],
                    &mut tile.recurrent_gate[..count],
                    &mut tile.beta[..count],
                    &mut tile.transformed[..count],
                    &mut tile.projector,
                    executor,
                    check,
                )?;
                full_attention_kv_only_tile(
                    weights,
                    geometry,
                    arithmetic,
                    mrope_base,
                    count,
                    frequencies_ready,
                    keys,
                    values,
                    tile,
                    check,
                )?;
            }
            _ => return Err(invalid("CPU tiled prefill layer/cache mismatch")),
        }
        check()
    }

    #[allow(
        clippy::too_many_lines,
        reason = "Keep tiled prefill with fused last-token evaluation in a single linear pass."
    )]
    fn ingest_cpu_tile_and_evaluate_last(
        &mut self,
        tokens: &[u32],
        check: &dyn Fn() -> Result<()>,
        tile: &mut TiledPrefill,
        mut executor: Option<&mut (dyn crate::kernels::Pq2ProjectionExecutor + '_)>,
        use_executor_on_tiles: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        let count = tokens.len();
        let base = self.position;
        let mrope_base = self.mrope_position;
        self.embed_tile_tokens(tokens, tile, check)?;
        let adapters = ActiveAdapters {
            multi_rank: self.multi_rank_adapter.as_ref(),
            multi_rank_scale: self.multi_rank_adapter_scale,
        };
        let Some((last_layer, leading_layers)) = self.layers.split_last() else {
            return Err(invalid("CPU tiled prefill requires at least one layer"));
        };
        let last_idx = self.layers.len() - 1;
        let (leading_states, last_states) = self.state.split_at_mut(last_idx);
        let (leading_deltas, last_deltas) = self.workspace.deltas.split_at_mut(last_idx);
        let (Some(last_state), Some(last_delta)) =
            (last_states.first_mut(), last_deltas.first_mut())
        else {
            return Err(invalid("CPU tiled prefill layer/state count mismatch"));
        };
        self.state_consistent = false;
        let mut frequencies_ready = false;
        let mut pending_residual = false;
        for (layer_idx, ((layer, state), delta)) in leading_layers
            .iter()
            .zip(leading_states.iter_mut())
            .zip(leading_deltas.iter_mut())
            .enumerate()
        {
            let tile_executor = if use_executor_on_tiles {
                executor.as_deref_mut()
            } else {
                None
            };
            Self::run_full_tile_layer(
                &self.geometry,
                self.arithmetic,
                &self.transforms,
                adapters,
                &mut self.workspace.projection,
                &mut self.workspace.adapter,
                &mut self.workspace.attention_scratch,
                layer_idx,
                layer,
                state,
                delta,
                base,
                mrope_base,
                count,
                &mut frequencies_ready,
                &mut pending_residual,
                tile,
                None,
                tile_executor,
                check,
            )?;
        }
        Self::run_last_layer_and_evaluate_last(
            &self.geometry,
            self.arithmetic,
            &self.transforms,
            adapters,
            &mut self.workspace.projection,
            &mut self.workspace.adapter,
            &mut self.workspace.attention_scratch,
            last_idx,
            last_layer,
            last_state,
            last_delta,
            base,
            mrope_base,
            count,
            &mut frequencies_ready,
            &mut pending_residual,
            tile,
            executor.as_deref_mut(),
            use_executor_on_tiles,
            check,
        )?;
        let last_tok = count - 1;
        Self::evaluate_tile_output_head_range(
            &self.output,
            &self.output_norm,
            self.vocabulary,
            &self.transforms,
            &self.geometry,
            self.arithmetic,
            adapters,
            &mut self.workspace.adapter,
            &mut self.workspace.projection.matvec,
            pending_residual,
            tile,
            last_tok,
            count,
            std::slice::from_mut(output),
            executor,
            check,
        )?;
        self.position = base + count;
        self.mrope_position = mrope_base + count;
        self.state_consistent = true;
        Ok(())
    }

    #[allow(
        clippy::too_many_lines,
        reason = "Keep tiled prefill with folded verification and logits capture in one pass."
    )]
    fn ingest_cpu_tile_with_folded(
        &mut self,
        tokens: &[u32],
        check: &dyn Fn() -> Result<()>,
        tile: &mut TiledPrefill,
        mut folded: Option<&mut FoldedVerificationState>,
        mut executor: Option<&mut (dyn crate::kernels::Pq2ProjectionExecutor + '_)>,
        logits: Option<&mut [Vec<f32>]>,
    ) -> Result<()> {
        let count = tokens.len();
        let base = self.position;
        let mrope_base = self.mrope_position;
        self.embed_tile_tokens(tokens, tile, check)?;
        // Mark live state inconsistent before any layer-major mutation so any
        // mid-tile error or cancellation leaves state unavailable until explicit
        // reset or restore, and keep `position` at the last committed prefix.
        let adapters = ActiveAdapters {
            multi_rank: self.multi_rank_adapter.as_ref(),
            multi_rank_scale: self.multi_rank_adapter_scale,
        };
        self.state_consistent = false;
        let mut frequencies_ready = false;
        let mut pending_residual = false;
        let total_layers = self.layers.len();
        let narrow_final = logits.is_none() && self.narrow_last_layer;
        let mut recurrent_layer_idx = 0_usize;
        for (layer_idx, ((layer, state), delta)) in self
            .layers
            .iter()
            .zip(&mut self.state)
            .zip(&mut self.workspace.deltas)
            .enumerate()
        {
            if narrow_final && layer_idx + 1 == total_layers {
                Self::run_narrow_last_layer_tile(
                    &self.geometry,
                    self.arithmetic,
                    &self.transforms,
                    adapters,
                    &mut self.workspace.adapter,
                    &mut self.workspace.projection.matvec,
                    layer,
                    state,
                    delta,
                    mrope_base,
                    count,
                    &mut frequencies_ready,
                    &mut pending_residual,
                    tile,
                    executor.as_deref_mut(),
                    check,
                )?;
                continue;
            }
            let folded_tape = if let Some(folded_state) = folded.as_deref_mut()
                && matches!(layer.attention, Attention::Recurrent(_))
            {
                let layer_tape = folded_state
                    .layers
                    .get_mut(recurrent_layer_idx)
                    .ok_or_else(|| invalid("folded verification layer tape missing"))?;
                recurrent_layer_idx += 1;
                Some(layer_tape)
            } else {
                None
            };
            Self::run_full_tile_layer(
                &self.geometry,
                self.arithmetic,
                &self.transforms,
                adapters,
                &mut self.workspace.projection,
                &mut self.workspace.adapter,
                &mut self.workspace.attention_scratch,
                layer_idx,
                layer,
                state,
                delta,
                base,
                mrope_base,
                count,
                &mut frequencies_ready,
                &mut pending_residual,
                tile,
                folded_tape,
                executor.as_deref_mut(),
                check,
            )?;
        }
        if let Some(logits) = logits {
            Self::evaluate_tile_output_head_range(
                &self.output,
                &self.output_norm,
                self.vocabulary,
                &self.transforms,
                &self.geometry,
                self.arithmetic,
                adapters,
                &mut self.workspace.adapter,
                &mut self.workspace.projection.matvec,
                pending_residual,
                tile,
                0,
                count,
                logits,
                executor,
                check,
            )?;
        } else if pending_residual {
            for (hidden, branch) in tile.hidden[..count]
                .iter_mut()
                .zip(&tile.attention[..count])
            {
                add_finite(hidden, branch, "CPU tiled decoder layer output")?;
            }
        }
        self.position = base + count;
        self.mrope_position = mrope_base + count;
        self.state_consistent = true;
        Ok(())
    }
}

#[cfg(test)]
#[path = "prefill_folded_tests.rs"]
mod folded_tests;

fn apply_state_only_gates(
    layer: &RecurrentAttention,
    tile: &mut TiledPrefill,
    count: usize,
) -> Result<()> {
    for (alpha, beta) in tile.alpha[..count].iter_mut().zip(&mut tile.beta[..count]) {
        for (((alpha_out, beta_out), &time_bias), &decay_weight) in alpha
            .iter_mut()
            .zip(beta.iter_mut())
            .zip(&layer.time_bias)
            .zip(&layer.decay)
        {
            *alpha_out = recurrent_decay(*alpha_out, time_bias, decay_weight)?;
            *beta_out = sigmoid(*beta_out);
        }
    }
    Ok(())
}

fn step_recurrent_state_only_conv_head_major(
    layer: &RecurrentAttention,
    geometry: &Geometry,
    arithmetic: Arithmetic,
    history: &[f32],
    convolution: &mut [f32],
    tile: &mut TiledPrefill,
    count: usize,
) -> Result<()> {
    let width = geometry.ssm_head;
    let hist_width = geometry.convolution - 1;
    let key_width = geometry.key_width();
    let head_hist = width * hist_width;
    let head_coeff = width * geometry.convolution;
    let head_out_len = count * width;
    let num_conv_heads = geometry.convolution_width() / width;
    let q_heads = key_width / width;
    let k_heads_end = 2 * q_heads;
    if !matches!(arithmetic, Arithmetic::PinnedSwift) {
        convolution.copy_from_slice(history);
    }
    resize_into(num_conv_heads * head_out_len, &mut tile.gdn_conv_outputs)?;
    let mixed_ref: &[Vec<f32>] = &tile.mixed[..count];
    tile.projector.row_parallelism().update_paired_chunks(
        convolution,
        head_hist,
        &mut tile.gdn_conv_outputs,
        head_out_len,
        |base_head, conv_chunk, out_chunk| {
            for (offset, (head_conv, head_out)) in conv_chunk
                .chunks_exact_mut(head_hist)
                .zip(out_chunk.chunks_exact_mut(head_out_len))
                .enumerate()
            {
                let head = base_head + offset;
                let c_slice = layer
                    .convolution
                    .range(head * head_coeff..(head + 1) * head_coeff);
                if matches!(arithmetic, Arithmetic::PinnedSwift) {
                    if head >= q_heads && head < k_heads_end {
                        crate::kernels::pinned_convolution::step_silu_clamped_l2_token_rows_from_coefficients(
                            &history[head * head_hist..(head + 1) * head_hist],
                            head_conv,
                            mixed_ref,
                            &(head * width..(head + 1) * width),
                            c_slice,
                            geometry.convolution,
                            geometry.epsilon,
                            head_out,
                        )?;
                    } else {
                        crate::kernels::pinned_convolution::step_silu_token_rows_from_coefficients(
                            &history[head * head_hist..(head + 1) * head_hist],
                            head_conv,
                            mixed_ref,
                            &(head * width..(head + 1) * width),
                            c_slice,
                            geometry.convolution,
                            head_out,
                        )?;
                    }
                } else {
                    for (token, token_mixed) in mixed_ref.iter().enumerate() {
                        head_out[token * width..(token + 1) * width]
                            .copy_from_slice(&token_mixed[head * width..(head + 1) * width]);
                    }
                    for head_mixed in head_out.chunks_exact_mut(width) {
                        for (channel, value) in head_mixed.iter_mut().enumerate() {
                            let projected = *value;
                            let start = channel * hist_width;
                            let coefficients = &c_slice.source()[channel * geometry.convolution
                                ..(channel + 1) * geometry.convolution];
                            *value = arithmetic.convolution_row(
                                &head_conv[start..start + hist_width],
                                projected,
                                coefficients,
                            )?;
                            head_conv[start..start + hist_width].rotate_left(1);
                            head_conv[start + hist_width - 1] = projected;
                        }
                        arithmetic.silu(head_mixed);
                    }
                    if head >= q_heads && head < k_heads_end {
                        for head_mixed in head_out.chunks_exact_mut(width) {
                            l2_norm(head_mixed, geometry.epsilon);
                        }
                    }
                }
            }
            Ok(())
        },
    )?;
    apply_state_only_gates(layer, tile, count)?;
    finite(convolution, "convolution state")
}

fn step_recurrent_state_only_conv_single_parallel(
    layer: &RecurrentAttention,
    geometry: &Geometry,
    arithmetic: Arithmetic,
    convolution: &mut [f32],
    tile: &mut TiledPrefill,
) -> Result<()> {
    let width = geometry.ssm_head;
    let hist_width = geometry.convolution - 1;
    let head_hist = width * hist_width;
    let head_coeff = width * geometry.convolution;
    let q_heads = geometry.key_width() / width;
    let k_heads_end = 2 * q_heads;
    let mixed = &mut tile.mixed[0];
    tile.projector.row_parallelism().update_paired_chunks(
        mixed,
        width,
        convolution,
        head_hist,
        |base_head, mixed_chunk, conv_chunk| {
            for (offset, (head_mixed, head_conv)) in mixed_chunk
                .chunks_exact_mut(width)
                .zip(conv_chunk.chunks_exact_mut(head_hist))
                .enumerate()
            {
                let head = base_head + offset;
                let c_slice = layer
                    .convolution
                    .range(head * head_coeff..(head + 1) * head_coeff);
                if matches!(arithmetic, Arithmetic::PinnedSwift) {
                    crate::kernels::pinned_convolution::step_silu_in_place_coefficients(
                        head_conv,
                        head_mixed,
                        c_slice,
                        geometry.convolution,
                    )?;
                } else {
                    for (channel, value) in head_mixed.iter_mut().enumerate() {
                        let projected = *value;
                        let start = channel * hist_width;
                        let coefficients = &c_slice.source()
                            [channel * geometry.convolution..(channel + 1) * geometry.convolution];
                        *value = arithmetic.convolution_row(
                            &head_conv[start..start + hist_width],
                            projected,
                            coefficients,
                        )?;
                        head_conv[start..start + hist_width].rotate_left(1);
                        head_conv[start + hist_width - 1] = projected;
                    }
                    arithmetic.silu(head_mixed);
                }
                if head >= q_heads && head < k_heads_end {
                    l2_norm(head_mixed, geometry.epsilon);
                }
            }
            Ok(())
        },
    )?;
    apply_state_only_gates(layer, tile, 1)?;
    finite(convolution, "convolution state")
}

#[expect(
    clippy::too_many_arguments,
    reason = "Keep causal state-only convolution and head-major dispatch explicit."
)]
fn step_recurrent_state_only_conv(
    layer: &RecurrentAttention,
    geometry: &Geometry,
    arithmetic: Arithmetic,
    history: &[f32],
    convolution: &mut [f32],
    tile: &mut TiledPrefill,
    count: usize,
    check: &dyn Fn() -> Result<()>,
) -> Result<bool> {
    let width = geometry.ssm_head;
    let hist_width = geometry.convolution - 1;
    let key_width = geometry.key_width();
    let conv_aligned = hist_width > 0
        && geometry.convolution_width().is_multiple_of(width)
        && width.is_multiple_of(4);
    if conv_aligned && count > 1 {
        for _ in 0..count {
            check()?;
        }
        step_recurrent_state_only_conv_head_major(
            layer,
            geometry,
            arithmetic,
            history,
            convolution,
            tile,
            count,
        )?;
        return Ok(true);
    }
    convolution.copy_from_slice(history);
    if conv_aligned && count == 1 && tile.projector.row_parallelism().workers() > 1 {
        check()?;
        step_recurrent_state_only_conv_single_parallel(
            layer,
            geometry,
            arithmetic,
            convolution,
            tile,
        )?;
        return Ok(false);
    }
    for ((mixed, alpha), beta) in tile.mixed[..count]
        .iter_mut()
        .zip(&mut tile.alpha[..count])
        .zip(&mut tile.beta[..count])
    {
        check()?;
        if matches!(arithmetic, Arithmetic::PinnedSwift) {
            crate::kernels::pinned_convolution::step_silu_in_place_coefficients(
                convolution,
                mixed,
                layer.convolution.view(),
                geometry.convolution,
            )?;
        } else {
            for (channel, value) in mixed.iter_mut().enumerate() {
                let projected = *value;
                let start = channel * hist_width;
                let coefficients = &layer.convolution.view().source()
                    [channel * geometry.convolution..(channel + 1) * geometry.convolution];
                *value = arithmetic.convolution_row(
                    &convolution[start..start + hist_width],
                    projected,
                    coefficients,
                )?;
                if hist_width > 0 {
                    convolution[start..start + hist_width].rotate_left(1);
                    convolution[start + hist_width - 1] = projected;
                }
            }
            arithmetic.silu(mixed);
        }
        // Only keys feed the state recurrence; queries and output gates are skipped.
        for head in mixed[key_width..2 * key_width].chunks_exact_mut(width) {
            l2_norm(head, geometry.epsilon);
        }
        for (((alpha_out, beta_out), &time_bias), &decay_weight) in alpha
            .iter_mut()
            .zip(beta.iter_mut())
            .zip(&layer.time_bias)
            .zip(&layer.decay)
        {
            *alpha_out = recurrent_decay(*alpha_out, time_bias, decay_weight)?;
            *beta_out = sigmoid(*beta_out);
        }
    }
    finite(convolution, "convolution state")?;
    Ok(false)
}

fn step_recurrent_state_only_rows_single(
    arithmetic: Arithmetic,
    step_inputs: &[StateStepInput<'_>],
    row_start: usize,
    width: usize,
    prev_head: &[f32],
    state_head: &mut [f32],
) -> Result<()> {
    for (token, input) in step_inputs.iter().enumerate() {
        for (offset, &v_val) in input.value[row_start..width].iter().enumerate() {
            let row_idx = row_start + offset;
            let state_row = &mut state_head[row_idx * width..(row_idx + 1) * width];
            let dot_k = if token == 0 {
                let prev_row = &prev_head[row_idx * width..(row_idx + 1) * width];
                arithmetic.gdn_decay_and_dot_from(prev_row, state_row, input.key, input.decay)?
            } else {
                arithmetic.gdn_decay_and_dot(state_row, input.key, input.decay)?
            };
            let step_delta = (v_val - dot_k) * input.beta;
            arithmetic.gdn_add_scaled(state_row, input.key, step_delta)?;
            finite(state_row, "GDN state")?;
        }
    }
    Ok(())
}

fn step_recurrent_state_only_head_pinned(
    arithmetic: Arithmetic,
    step_inputs: &[StateStepInput<'_>],
    width: usize,
    prev_head: &[f32],
    state_head: &mut [f32],
) -> Result<()> {
    let row_groups = width / 4;
    for group_idx in 0..row_groups {
        let row_base = group_idx * 4;
        let prev_rows: [&[f32]; 4] =
            std::array::from_fn(|r| &prev_head[(row_base + r) * width..(row_base + r + 1) * width]);
        let state_group = &mut state_head[row_base * width..(row_base + 4) * width];
        let (s01, s23) = state_group.split_at_mut(2 * width);
        let (s0, s1) = s01.split_at_mut(width);
        let (s2, s3) = s23.split_at_mut(width);
        step_four_rows_state_only_tokens_from(
            prev_rows,
            [s0, s1, s2, s3],
            step_inputs,
            row_base,
            width,
        )?;
    }
    let tail_start = row_groups * 4;
    if tail_start < width {
        step_recurrent_state_only_rows_single(
            arithmetic,
            step_inputs,
            tail_start,
            width,
            prev_head,
            state_head,
        )?;
    }
    Ok(())
}

fn step_recurrent_state_only_matrix(
    geometry: &Geometry,
    arithmetic: Arithmetic,
    tile: &TiledPrefill,
    count: usize,
    conv_head_major: bool,
    previous: &[f32],
    matrix: &mut [f32],
) -> Result<()> {
    let width = geometry.ssm_head;
    let key_width = geometry.key_width();
    let head_state_len = width * width;
    let head_out_len = count * width;
    let q_heads = key_width / width;
    let mixed_ref: &[Vec<f32>] = &tile.mixed[..count];
    let conv_ref: &[f32] = &tile.gdn_conv_outputs;
    let alpha_ref: &[Vec<f32>] = &tile.alpha[..count];
    let beta_ref: &[Vec<f32>] = &tile.beta[..count];
    let use_pinned_four_row =
        matches!(arithmetic, Arithmetic::PinnedSwift) && width >= 4 && count > 0;
    tile.projector.row_parallelism().update_chunks(
        matrix,
        head_state_len,
        |base_head, state_chunk| {
            let mut step_inputs = Vec::with_capacity(count);
            for (offset, state_head) in state_chunk.chunks_exact_mut(head_state_len).enumerate() {
                let head = base_head + offset;
                let prev_head = &previous[head * head_state_len..(head + 1) * head_state_len];
                let key_head = head % geometry.ssm_keys;
                step_inputs.clear();
                if conv_head_major {
                    let k_head_idx = q_heads + key_head;
                    let k_head =
                        &conv_ref[k_head_idx * head_out_len..(k_head_idx + 1) * head_out_len];
                    let v_head_idx = 2 * q_heads + head;
                    let v_head =
                        &conv_ref[v_head_idx * head_out_len..(v_head_idx + 1) * head_out_len];
                    for (token, (alpha, beta)) in alpha_ref.iter().zip(beta_ref).enumerate() {
                        step_inputs.push(StateStepInput {
                            key: &k_head[token * width..(token + 1) * width],
                            value: &v_head[token * width..(token + 1) * width],
                            decay: alpha[head],
                            beta: beta[head],
                        });
                    }
                } else {
                    for ((mixed, alpha), beta) in mixed_ref.iter().zip(alpha_ref).zip(beta_ref) {
                        step_inputs.push(StateStepInput {
                            key: &mixed
                                [key_width + key_head * width..key_width + (key_head + 1) * width],
                            value: &mixed
                                [2 * key_width + head * width..2 * key_width + (head + 1) * width],
                            decay: alpha[head],
                            beta: beta[head],
                        });
                    }
                }
                if use_pinned_four_row {
                    step_recurrent_state_only_head_pinned(
                        arithmetic,
                        &step_inputs,
                        width,
                        prev_head,
                        state_head,
                    )?;
                } else {
                    step_recurrent_state_only_rows_single(
                        arithmetic,
                        &step_inputs,
                        0,
                        width,
                        prev_head,
                        state_head,
                    )?;
                }
            }
            Ok(())
        },
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "Keep causal recurrent state updates, projection reuse and tile scratch explicit."
)]
fn recurrent_state_only_tile<'a>(
    layer: &'a RecurrentAttention,
    state: &mut LayerState,
    delta: &mut LayerDelta,
    geometry: &Geometry,
    arithmetic: Arithmetic,
    transforms: &Transforms,
    adapters: ActiveAdapters<'_>,
    adapter_scratch: &mut AdapterScratch,
    projection_scratch: &mut crate::kernels::MatvecScratch,
    previous_norm: TileReuse<'a>,
    count: usize,
    check: &dyn Fn() -> Result<()>,
    tile: &mut TiledPrefill,
    mut executor: Option<&mut (dyn crate::kernels::Pq2ProjectionExecutor + '_)>,
) -> Result<()> {
    let (
        LayerState::Recurrent {
            convolution: history,
            matrix: previous,
        },
        LayerDelta::Recurrent {
            convolution,
            matrix,
        },
    ) = (state, delta)
    else {
        return Err(invalid("state-only tile requires recurrent state"));
    };
    check()?;
    let previous_qkv = project_tile(
        transforms,
        geometry,
        arithmetic,
        adapters,
        adapter_scratch,
        projection_scratch,
        previous_norm,
        &layer.qkv,
        &tile.normalized[..count],
        &mut tile.mixed[..count],
        &mut tile.transformed[..count],
        &mut tile.projector,
        executor.as_deref_mut(),
        check,
    )?;
    check()?;
    let _ = project_tile_pair(
        transforms,
        geometry,
        arithmetic,
        adapters,
        adapter_scratch,
        projection_scratch,
        previous_qkv,
        [&layer.beta, &layer.alpha],
        &tile.normalized[..count],
        &mut tile.beta[..count],
        &mut tile.alpha[..count],
        &mut tile.transformed[..count],
        &mut tile.projector,
        executor,
        check,
    )?;
    resize_into(history.len(), convolution)?;
    resize_into(previous.len(), matrix)?;
    let conv_head_major = step_recurrent_state_only_conv(
        layer,
        geometry,
        arithmetic,
        history,
        convolution,
        tile,
        count,
        check,
    )?;
    step_recurrent_state_only_matrix(
        geometry,
        arithmetic,
        tile,
        count,
        conv_head_major,
        previous,
        matrix,
    )?;
    check()?;
    std::mem::swap(history, convolution);
    std::mem::swap(previous, matrix);
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "Keep causal KV append ordering and tile scratch explicit."
)]
fn full_attention_kv_only_tile(
    weights: &FullAttention,
    g: &Geometry,
    arithmetic: Arithmetic,
    mrope_base: usize,
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
            rope_frequencies(g, mrope_base + index, frequencies)?;
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
        keys.prepare_append(&tile.recurrent_gate[index])?;
        values.prepare_append(&tile.beta[index])?;
        keys.append(&tile.recurrent_gate[index]);
        values.append(&tile.beta[index]);
        check()?;
    }
    Ok(())
}

const fn select_2d_attention_chunks(count: usize, kv_heads: usize, workers: usize) -> usize {
    if count <= 1 || kv_heads <= 1 || workers <= 1 {
        return 1;
    }
    let mut best_chunks = 1;
    let mut best_active = 0;
    let mut best_units = usize::MAX;
    let mut n_chunks = 1;
    while n_chunks <= count {
        let units = n_chunks * kv_heads;
        let task_units = units.div_ceil(workers);
        let active = units.div_ceil(task_units);
        if active > best_active || (active == best_active && units < best_units) {
            best_active = active;
            best_units = units;
            best_chunks = n_chunks;
        }
        n_chunks += 1;
    }
    best_chunks
}

const fn token_chunk_range(count: usize, n_chunks: usize, chunk_idx: usize) -> (usize, usize) {
    let base_len = count / n_chunks;
    let rem = count % n_chunks;
    let start = chunk_idx * base_len + if chunk_idx < rem { chunk_idx } else { rem };
    let len = base_len + if chunk_idx < rem { 1 } else { 0 };
    (start, len)
}

enum AttendedSlice<'a> {
    TokenRows(&'a mut [Vec<f32>]),
    FlatKvGroup {
        rows: &'a mut [f32],
        base_head: usize,
        row_stride: usize,
        head_width: usize,
    },
}

type PairFourHeadSlices<'a> = ([&'a mut [f32]; 4], [&'a mut [f32]; 4]);

impl AttendedSlice<'_> {
    fn head_slice_mut(&mut self, tok_offset: usize, head: usize, width: usize) -> &mut [f32] {
        match self {
            Self::TokenRows(rows) => &mut rows[tok_offset][head * width..(head + 1) * width],
            Self::FlatKvGroup {
                rows,
                base_head,
                row_stride,
                head_width,
            } => {
                let start = tok_offset * *row_stride + (head - *base_head) * *head_width;
                &mut rows[start..start + width]
            }
        }
    }

    fn pair_four_heads_mut(
        &mut self,
        t0: usize,
        t1: usize,
        head: usize,
        width: usize,
    ) -> Result<PairFourHeadSlices<'_>> {
        let (tok0, tok1) = match self {
            Self::TokenRows(rows) => {
                let (head_rows, tail_rows) = rows.split_at_mut(t1);
                let start = head * width;
                let end = (head + 4) * width;
                (
                    &mut head_rows[t0][start..end],
                    &mut tail_rows[0][start..end],
                )
            }
            Self::FlatKvGroup {
                rows,
                base_head,
                row_stride,
                head_width,
            } => {
                let stride = *row_stride;
                let start = (head - *base_head) * *head_width;
                let end = start + 4 * width;
                let (_, tail) = rows.split_at_mut(t0 * stride);
                let (row0, row1_and_more) = tail.split_at_mut(stride);
                (&mut row0[start..end], &mut row1_and_more[start..end])
            }
        };
        let (r0_left, r0_right) = tok0.split_at_mut(2 * width);
        let (r0_0, r0_1) = r0_left.split_at_mut(width);
        let (r0_2, r0_3) = r0_right.split_at_mut(width);
        let (r1_left, r1_right) = tok1.split_at_mut(2 * width);
        let (r1_0, r1_1) = r1_left.split_at_mut(width);
        let (r1_2, r1_3) = r1_right.split_at_mut(width);
        if r0_0.len() != width || r1_0.len() != width {
            return Err(invalid("invalid generic attention output slice length"));
        }
        Ok(([r0_0, r0_1, r0_2, r0_3], [r1_0, r1_1, r1_2, r1_3]))
    }

    fn four_heads_mut(&mut self, t: usize, head: usize, width: usize) -> Result<[&mut [f32]; 4]> {
        let row = match self {
            Self::TokenRows(rows) => &mut rows[t][head * width..(head + 4) * width],
            Self::FlatKvGroup {
                rows,
                base_head,
                row_stride,
                head_width,
            } => {
                let start = t * *row_stride + (head - *base_head) * *head_width;
                &mut rows[start..start + 4 * width]
            }
        };
        let (r01, r23) = row.split_at_mut(2 * width);
        let (r0, r1) = r01.split_at_mut(width);
        let (r2, r3) = r23.split_at_mut(width);
        if r0.len() != width {
            return Err(invalid("invalid generic attention output slice length"));
        }
        Ok([r0, r1, r2, r3])
    }
}

#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "Evaluates one KV-head range across a causal token chunk for both 1D token chunks and 2D (token_chunk, kv_head) tasks."
)]
fn attend_generic_head_range(
    g: &Geometry,
    arithmetic: Arithmetic,
    base: usize,
    base_tok: usize,
    chunk_tokens: usize,
    head_start: usize,
    head_end: usize,
    heads_per_kv: usize,
    max_score_len: usize,
    token_score_stride: usize,
    scale: f32,
    key_cache: &KvCache,
    value_cache: &KvCache,
    query_ref: &[Vec<f32>],
    projected_ref: &[Vec<f32>],
    current_keys: &[Vec<f32>],
    current_values: &[Vec<f32>],
    mut attended_out: AttendedSlice<'_>,
    scores_chunk: &mut [f32],
) -> Result<()> {
    let mut head = head_start;
    while head < head_end {
        let kv_head = head / heads_per_kv;
        let offset = kv_head * g.head;
        if head + 4 <= head_end && kv_head == (head + 3) / heads_per_kv {
            let shared_prefix_len = base + base_tok;
            let total_past = shared_prefix_len + chunk_tokens.saturating_sub(1);
            let score_one_key = |scores_chunk: &mut [f32],
                                 pos: usize,
                                 past_key: &[u16],
                                 start_t: usize,
                                 end_t: usize|
             -> Result<()> {
                let mut t = start_t;
                while t + 2 <= end_t {
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
                    let (dots0, dots1) = arithmetic
                        .attention_dot_two_token_four_queries(queries0, queries1, past_key)?;
                    {
                        let tok0_scores =
                            &mut scores_chunk[t * token_score_stride..(t + 1) * token_score_stride];
                        tok0_scores[pos] = dots0[0] * scale;
                        tok0_scores[max_score_len + pos] = dots0[1] * scale;
                        tok0_scores[2 * max_score_len + pos] = dots0[2] * scale;
                        tok0_scores[3 * max_score_len + pos] = dots0[3] * scale;
                    }
                    {
                        let tok1_scores = &mut scores_chunk
                            [(t + 1) * token_score_stride..(t + 2) * token_score_stride];
                        tok1_scores[pos] = dots1[0] * scale;
                        tok1_scores[max_score_len + pos] = dots1[1] * scale;
                        tok1_scores[2 * max_score_len + pos] = dots1[2] * scale;
                        tok1_scores[3 * max_score_len + pos] = dots1[3] * scale;
                    }
                    t += 2;
                }
                for rem_t in t..end_t {
                    let tok = base_tok + rem_t;
                    let queries = std::array::from_fn(|i| {
                        let h = head + i;
                        &query_ref[tok][h * g.head..(h + 1) * g.head]
                    });
                    let dots = arithmetic.attention_dot_four(queries, past_key)?;
                    let tok_scores = &mut scores_chunk
                        [rem_t * token_score_stride..(rem_t + 1) * token_score_stride];
                    tok_scores[pos] = dots[0] * scale;
                    tok_scores[max_score_len + pos] = dots[1] * scale;
                    tok_scores[2 * max_score_len + pos] = dots[2] * scale;
                    tok_scores[3 * max_score_len + pos] = dots[3] * scale;
                }
                Ok(())
            };
            let score_two_keys = |scores_chunk: &mut [f32],
                                  pos0: usize,
                                  keys2: [&[u16]; 2],
                                  start_t: usize,
                                  end_t: usize|
             -> Result<()> {
                let mut t = start_t;
                while t + 2 <= end_t {
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
                    let (dots0_2, dots1_2) = arithmetic
                        .attention_dot_two_keys_two_token_four_queries(queries0, queries1, keys2)?;
                    {
                        let tok0_scores =
                            &mut scores_chunk[t * token_score_stride..(t + 1) * token_score_stride];
                        for (k_off, dots) in dots0_2.into_iter().enumerate() {
                            let pos = pos0 + k_off;
                            tok0_scores[pos] = dots[0] * scale;
                            tok0_scores[max_score_len + pos] = dots[1] * scale;
                            tok0_scores[2 * max_score_len + pos] = dots[2] * scale;
                            tok0_scores[3 * max_score_len + pos] = dots[3] * scale;
                        }
                    }
                    {
                        let tok1_scores = &mut scores_chunk
                            [(t + 1) * token_score_stride..(t + 2) * token_score_stride];
                        for (k_off, dots) in dots1_2.into_iter().enumerate() {
                            let pos = pos0 + k_off;
                            tok1_scores[pos] = dots[0] * scale;
                            tok1_scores[max_score_len + pos] = dots[1] * scale;
                            tok1_scores[2 * max_score_len + pos] = dots[2] * scale;
                            tok1_scores[3 * max_score_len + pos] = dots[3] * scale;
                        }
                    }
                    t += 2;
                }
                if t < end_t {
                    let tok = base_tok + t;
                    let queries = std::array::from_fn(|i| {
                        let h = head + i;
                        &query_ref[tok][h * g.head..(h + 1) * g.head]
                    });
                    let dots2 = arithmetic.attention_dot_two_keys_four_queries(queries, keys2)?;
                    let tok_scores =
                        &mut scores_chunk[t * token_score_stride..(t + 1) * token_score_stride];
                    for (k_off, dots) in dots2.into_iter().enumerate() {
                        let pos = pos0 + k_off;
                        tok_scores[pos] = dots[0] * scale;
                        tok_scores[max_score_len + pos] = dots[1] * scale;
                        tok_scores[2 * max_score_len + pos] = dots[2] * scale;
                        tok_scores[3 * max_score_len + pos] = dots[3] * scale;
                    }
                }
                Ok(())
            };
            let mut past_iter = key_cache.tokens().take(total_past).enumerate();
            while let Some((pos0, p0)) = past_iter.next() {
                if pos0 + 4 <= total_past
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
                    let common_t = (pos0 + 4)
                        .saturating_sub(shared_prefix_len)
                        .min(chunk_tokens);
                    if common_t > 0 {
                        let t0 = (pos0 + 1).saturating_sub(shared_prefix_len).min(common_t);
                        let t1 = (pos0 + 2).saturating_sub(shared_prefix_len).min(common_t);
                        if t0 < t1 {
                            score_one_key(scores_chunk, pos0, keys4[0], t0, t1)?;
                        }
                        if t1 < common_t {
                            score_two_keys(scores_chunk, pos0, [keys4[0], keys4[1]], t1, common_t)?;
                        }
                        let t2 = (pos0 + 3).saturating_sub(shared_prefix_len).min(common_t);
                        if t2 < common_t {
                            score_one_key(scores_chunk, pos0 + 2, keys4[2], t2, common_t)?;
                        }
                    }
                    let mut t = common_t;
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
                        let dots4 =
                            arithmetic.attention_dot_four_keys_four_queries(queries, keys4)?;
                        let tok_scores =
                            &mut scores_chunk[t * token_score_stride..(t + 1) * token_score_stride];
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
                if pos0 + 2 <= total_past
                    && let Some((_, p1)) = past_iter.next()
                {
                    let keys2 = [&p0[offset..offset + g.head], &p1[offset..offset + g.head]];
                    let common_t = (pos0 + 2)
                        .saturating_sub(shared_prefix_len)
                        .min(chunk_tokens);
                    let t0 = (pos0 + 1).saturating_sub(shared_prefix_len).min(common_t);
                    if t0 < common_t {
                        score_one_key(scores_chunk, pos0, keys2[0], t0, common_t)?;
                    }
                    score_two_keys(scores_chunk, pos0, keys2, common_t, chunk_tokens)?;
                    continue;
                }
                let past_key = &p0[offset..offset + g.head];
                let first_t = (pos0 + 1).saturating_sub(shared_prefix_len);
                score_one_key(scores_chunk, pos0, past_key, first_t, chunk_tokens)?;
            }
            let pairs = chunk_tokens / 2;
            for pair_idx in 0..pairs {
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
                let pair_scores =
                    &mut scores_chunk[t0 * token_score_stride..(t1 + 1) * token_score_stride];
                let (tok0_scores, tok1_scores) = pair_scores.split_at_mut(token_score_stride);
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
                    arithmetic.attention_softmax_round_f16(scores)?;
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
                    arithmetic.attention_softmax_round_f16(scores)?;
                }
                let probs0: [&[f32]; 4] = [s0_0, s0_1, s0_2, s0_3];
                let probs1: [&[f32]; 4] = [s1_0, s1_1, s1_2, s1_3];
                let ([r0_0, r0_1, r0_2, r0_3], [r1_0, r1_1, r1_2, r1_3]) =
                    attended_out.pair_four_heads_mut(t0, t1, head, g.head)?;
                let g0_0 = attention_gate(&projected_ref[tok0], head, g.head);
                let g0_1 = attention_gate(&projected_ref[tok0], head + 1, g.head);
                let g0_2 = attention_gate(&projected_ref[tok0], head + 2, g.head);
                let g0_3 = attention_gate(&projected_ref[tok0], head + 3, g.head);
                let g1_0 = attention_gate(&projected_ref[tok1], head, g.head);
                let g1_1 = attention_gate(&projected_ref[tok1], head + 1, g.head);
                let g1_2 = attention_gate(&projected_ref[tok1], head + 2, g.head);
                let g1_3 = attention_gate(&projected_ref[tok1], head + 3, g.head);
                for (
                    tile_idx,
                    ((((out0_0, out0_1), out0_2), out0_3), (((out1_0, out1_1), out1_2), out1_3)),
                ) in
                    r0_0.chunks_mut(crate::kernels::pinned_attention::VALUE_TILE_FOUR_TWO_TOKENS)
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
                    let dimension =
                        tile_idx * crate::kernels::pinned_attention::VALUE_TILE_FOUR_TWO_TOKENS;
                    let len = out0_0.len();
                    arithmetic.attention_value_tile_four_two_tokens(
                        value_cache,
                        [
                            &current_values[tok0][offset + dimension..offset + dimension + len],
                            &current_values[tok1][offset + dimension..offset + dimension + len],
                        ],
                        offset + dimension,
                        probs0,
                        probs1,
                        [out0_0, out0_1, out0_2, out0_3],
                        [out1_0, out1_1, out1_2, out1_3],
                        [
                            [
                                &g0_0[dimension..dimension + len],
                                &g0_1[dimension..dimension + len],
                                &g0_2[dimension..dimension + len],
                                &g0_3[dimension..dimension + len],
                            ],
                            [
                                &g1_0[dimension..dimension + len],
                                &g1_1[dimension..dimension + len],
                                &g1_2[dimension..dimension + len],
                                &g1_3[dimension..dimension + len],
                            ],
                        ],
                    )?;
                }
            }
            if !chunk_tokens.is_multiple_of(2) {
                let t = pairs * 2;
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
                    arithmetic.attention_softmax_round_f16(scores)?;
                }
                let probs: [&[f32]; 4] = [s0, s1, s2, s3];
                let [r0, r1, r2, r3] = attended_out.four_heads_mut(t, head, g.head)?;
                let g0 = attention_gate(&projected_ref[tok], head, g.head);
                let g1 = attention_gate(&projected_ref[tok], head + 1, g.head);
                let g2 = attention_gate(&projected_ref[tok], head + 2, g.head);
                let g3 = attention_gate(&projected_ref[tok], head + 3, g.head);
                for (tile_idx, (((out0, out1), out2), out3)) in r0
                    .chunks_mut(crate::kernels::pinned_attention::VALUE_TILE_FOUR)
                    .zip(r1.chunks_mut(crate::kernels::pinned_attention::VALUE_TILE_FOUR))
                    .zip(r2.chunks_mut(crate::kernels::pinned_attention::VALUE_TILE_FOUR))
                    .zip(r3.chunks_mut(crate::kernels::pinned_attention::VALUE_TILE_FOUR))
                    .enumerate()
                {
                    let dimension = tile_idx * crate::kernels::pinned_attention::VALUE_TILE_FOUR;
                    let len = out0.len();
                    arithmetic.attention_value_tile_four(
                        value_cache,
                        &current_values[tok][offset + dimension..offset + dimension + len],
                        offset + dimension,
                        probs,
                        [out0, out1, out2, out3],
                        [
                            &g0[dimension..dimension + len],
                            &g1[dimension..dimension + len],
                            &g2[dimension..dimension + len],
                            &g3[dimension..dimension + len],
                        ],
                    )?;
                }
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
        for t in 0..chunk_tokens {
            let tok = base_tok + t;
            let score_len = base + tok + 1;
            let q = &query_ref[tok][head * g.head..(head + 1) * g.head];
            let scores =
                &mut scores_chunk[t * token_score_stride..t * token_score_stride + score_len];
            scores[score_len - 1] = arithmetic
                .attention_current_dot(q, &current_keys[tok][offset..offset + g.head])?
                * scale;
            arithmetic.attention_softmax_round_f16(scores)?;
            let result = attended_out.head_slice_mut(t, head, g.head);
            let g_head = attention_gate(&projected_ref[tok], head, g.head);
            for (tile_idx, out) in result
                .chunks_mut(crate::kernels::pinned_attention::VALUE_TILE)
                .enumerate()
            {
                let dimension = tile_idx * crate::kernels::pinned_attention::VALUE_TILE;
                let len = out.len();
                arithmetic.attention_value_tile(
                    value_cache,
                    &current_values[tok][offset + dimension..offset + dimension + len],
                    offset + dimension,
                    scores,
                    out,
                )?;
                crate::kernels::mul_sigmoid_in_place(out, &g_head[dimension..dimension + len]);
            }
        }
        head += 1;
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "Keep causal KV append ordering, exact FP16 score/value rounding and tile scratch explicit."
)]
fn full_attention_tile(
    weights: &FullAttention,
    g: &Geometry,
    arithmetic: Arithmetic,
    transforms: &Transforms,
    base: usize,
    mrope_base: usize,
    count: usize,
    frequencies_ready: &mut bool,
    keys: &mut KvCache,
    values: &mut KvCache,
    projection: &mut super::ProjectionScratch,
    adapters: ActiveAdapters<'_>,
    adapter_scratch: &mut AdapterScratch,
    attention_scratch: &mut super::AttentionScratch,
    tile: &mut TiledPrefill,
    check: &dyn Fn() -> Result<()>,
) -> Result<()> {
    check()?;
    if !*frequencies_ready {
        for (index, frequencies) in tile.frequencies.iter_mut().enumerate().take(count) {
            rope_frequencies(g, mrope_base + index, frequencies)?;
        }
        *frequencies_ready = true;
    }
    // A one-token tile and the single-worker policy retain the ordinary
    // attention path. Multi-token packed history uses immutable causal views
    // after the common normalization/RoPE stage below.
    if count <= 1 || tile.projector.row_parallelism().workers() <= 1 {
        for index in 0..count {
            if index > 0 {
                check()?;
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
                mrope_positions: [
                    mrope_base + index,
                    mrope_base + index,
                    mrope_base + index,
                    0,
                ],
                frequencies_ready: true,
                projection,
                adapters,
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
                &mut attention_scratch.scores,
                &mut attention_scratch.packed_probs,
                &mut attention_scratch.frequencies,
                &mut tile.attended[index],
            );
            std::mem::swap(
                &mut attention_scratch.frequencies,
                &mut tile.frequencies[index],
            );
            step_result?;
            keys.prepare_staged_append(&tile.recurrent_gate[index])?;
            values.prepare_staged_append(&tile.beta[index])?;
            keys.append(&tile.recurrent_gate[index]);
            values.append(&tile.beta[index]);
            check()?;
        }
        return Ok(());
    }
    let query_gate_len = g.head * g.heads;
    for index in 0..count {
        resize_into(query_gate_len, &mut tile.up[index])?;
        resize_into(query_gate_len, &mut tile.attended[index])?;
    }
    let mixed_ref: &[Vec<f32>] = &tile.mixed[..count];
    let frequencies_ref: &[Vec<(f32, f32)>] = &tile.frequencies[..count];
    let query_norm: &[f32] = &weights.query_norm;
    let normalize_queries = |base_tok: usize, query_chunk: &mut [Vec<f32>]| {
        for (offset, query_tok) in query_chunk.iter_mut().enumerate() {
            let tok = base_tok + offset;
            let projected = &mixed_ref[tok];
            let frequencies = &frequencies_ref[tok];
            let mut query_finite = true;
            for (head_idx, query_head) in query_tok.chunks_exact_mut(g.head).enumerate() {
                let src = &projected[head_idx * 2 * g.head..(head_idx + 1) * 2 * g.head];
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
    };
    let key_norm: &[f32] = &weights.key_norm;
    let normalize_key_values =
        |base_tok: usize, key_chunk: &mut [Vec<f32>], value_chunk: &mut [Vec<f32>]| {
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
        };
    // All three buffers partition by token, so every worker retains the exact
    // chunks of the two original submissions, including partial final chunks.
    // Query failures must outrank every KV failure. Keep KV failures on the cold
    // path until all queries succeed; only provisional scratch changes here.
    let key_value_failure = std::sync::Mutex::new(None);
    tile.projector.row_parallelism().update_triple_chunks(
        &mut tile.up[..count],
        1,
        &mut tile.recurrent_gate[..count],
        1,
        &mut tile.beta[..count],
        1,
        |base_tok, query_chunk, key_chunk, value_chunk| {
            normalize_queries(base_tok, query_chunk)?;
            if let Err(error) = normalize_key_values(base_tok, key_chunk, value_chunk) {
                let mut failure = key_value_failure
                    .lock()
                    .map_err(|_| invalid("prefill attention preparation failure lock poisoned"))?;
                if failure.as_ref().is_none_or(|&(base, _)| base_tok < base) {
                    *failure = Some((base_tok, error));
                }
            }
            Ok(())
        },
    )?;
    if let Some((_, error)) = key_value_failure
        .into_inner()
        .map_err(|_| invalid("prefill attention preparation failure lock poisoned"))?
    {
        return Err(error);
    }
    if keys.packed.is_some() || values.packed.is_some() {
        // Encode rotated/quantized K/V rows in parallel and commit pages in
        // exact token order. Every append remains provisional to the
        // layer-major attempt and whole-request checkpoint/rollback contract.
        check()?;
        super::quantized_kv::prepare_tile_append_parallel_with_scratch(
            keys,
            values,
            &tile.recurrent_gate[..count],
            &tile.beta[..count],
            &mut tile.packed_kv,
            tile.projector.row_parallelism(),
        )?;
        check()?;
        let score_stride = base + count;
        resize_attention_scores(
            count,
            score_stride,
            keys.maximum / keys.width,
            &mut attention_scratch.scores,
        )?;
        super::quantized_kv::attend_tile_with_scratch(
            keys,
            values,
            &tile.up[..count],
            &tile.mixed[..count],
            &mut tile.attended[..count],
            &mut attention_scratch.scores,
            base,
            g.heads,
            1.0 / small_float(g.head)?.sqrt(),
            arithmetic,
            &mut tile.packed_kv,
            tile.projector.row_parallelism(),
        )?;
        check()?;
        return Ok(());
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
    let scale = 1.0 / small_float(g.head)?.sqrt();
    let max_score_len = base + count;
    let six_head_groups =
        matches!(arithmetic, Arithmetic::PinnedSwift) && g.heads / g.kv_heads == 6;
    let score_heads = if six_head_groups { 6 } else { 4 };
    let token_score_stride = score_heads * max_score_len;
    let heads_per_kv = g.heads / g.kv_heads;
    let workers = tile.projector.row_parallelism().workers();
    resize_attention_scores(
        count * score_heads,
        max_score_len,
        keys.maximum / keys.width,
        &mut attention_scratch.scores,
    )?;
    if six_head_groups && count > 1 && workers > 1 {
        return super::attention_six::attend_tile_balanced_with_scratch(
            g,
            base,
            &tile.up[..count],
            &tile.mixed[..count],
            keys,
            values,
            &tile.recurrent_gate[..count],
            &tile.beta[..count],
            &mut tile.attended[..count],
            &mut attention_scratch.scores,
            scale,
            &mut tile.six_head,
            tile.projector.row_parallelism(),
        );
    }
    if count < workers && g.kv_heads > 1 {
        let n_chunks = select_2d_attention_chunks(count, g.kv_heads, workers);
        let max_chunk_tokens = count.div_ceil(n_chunks);
        let kv_out_len = heads_per_kv * g.head;
        let chunk_out_unit = checked_product(&[max_chunk_tokens, kv_out_len])?;
        let chunk_score_unit = checked_product(&[max_chunk_tokens, token_score_stride])?;
        let total_tasks = n_chunks * g.kv_heads;
        let total_score_rows = checked_product(&[total_tasks, max_chunk_tokens, score_heads])?;
        let total_out_elems = checked_product(&[total_tasks, chunk_out_unit])?;
        resize_attention_scores(
            total_score_rows,
            max_score_len,
            keys.maximum / keys.width,
            &mut tile.attention_task_scores,
        )?;
        resize_into(total_out_elems, &mut tile.attention_task_outputs)?;
        let query_ref: &[Vec<f32>] = &tile.up[..count];
        let projected_ref: &[Vec<f32>] = &tile.mixed[..count];
        let current_keys: &[Vec<f32>] = &tile.recurrent_gate[..count];
        let current_values: &[Vec<f32>] = &tile.beta[..count];
        let key_cache: &KvCache = keys;
        let value_cache: &KvCache = values;
        tile.projector.row_parallelism().update_paired_chunks(
            &mut tile.attention_task_outputs[..total_out_elems],
            chunk_out_unit,
            &mut tile.attention_task_scores[..total_tasks * chunk_score_unit],
            chunk_score_unit,
            |base_task, out_chunk, scores_chunk| {
                for (local, (task_out, task_scores)) in out_chunk
                    .chunks_exact_mut(chunk_out_unit)
                    .zip(scores_chunk.chunks_exact_mut(chunk_score_unit))
                    .enumerate()
                {
                    let task_idx = base_task + local;
                    let chunk_idx = task_idx / g.kv_heads;
                    let kv_head = task_idx % g.kv_heads;
                    let (tok_start, chunk_tokens) = token_chunk_range(count, n_chunks, chunk_idx);
                    let tok_end = tok_start + chunk_tokens;
                    let active_out = &mut task_out[..chunk_tokens * kv_out_len];
                    let active_scores = &mut task_scores[..chunk_tokens * token_score_stride];
                    let head_start = kv_head * heads_per_kv;
                    if six_head_groups {
                        super::attention_six::attend_tile_kv_group(
                            g,
                            head_start,
                            base + tok_start,
                            &query_ref[tok_start..tok_end],
                            &projected_ref[tok_start..tok_end],
                            key_cache,
                            value_cache,
                            &current_keys[tok_start..tok_end],
                            &current_values[tok_start..tok_end],
                            active_out,
                            active_scores,
                            token_score_stride,
                            scale,
                        )?;
                    } else {
                        attend_generic_head_range(
                            g,
                            arithmetic,
                            base,
                            tok_start,
                            chunk_tokens,
                            head_start,
                            head_start + heads_per_kv,
                            heads_per_kv,
                            max_score_len,
                            token_score_stride,
                            scale,
                            key_cache,
                            value_cache,
                            query_ref,
                            projected_ref,
                            current_keys,
                            current_values,
                            AttendedSlice::FlatKvGroup {
                                rows: active_out,
                                base_head: head_start,
                                row_stride: kv_out_len,
                                head_width: g.head,
                            },
                            active_scores,
                        )?;
                    }
                }
                Ok(())
            },
        )?;
        for chunk_idx in 0..n_chunks {
            let (tok_start, chunk_tokens) = token_chunk_range(count, n_chunks, chunk_idx);
            for kv_head in 0..g.kv_heads {
                let task_idx = chunk_idx * g.kv_heads + kv_head;
                let task_out = &tile.attention_task_outputs[task_idx * chunk_out_unit
                    ..task_idx * chunk_out_unit + chunk_tokens * kv_out_len];
                let out_offset = kv_head * kv_out_len;
                for local_tok in 0..chunk_tokens {
                    let src = &task_out[local_tok * kv_out_len..(local_tok + 1) * kv_out_len];
                    tile.attended[tok_start + local_tok][out_offset..out_offset + kv_out_len]
                        .copy_from_slice(src);
                }
            }
        }
        return Ok(());
    }
    let query_ref: &[Vec<f32>] = &tile.up[..count];
    // The projected query/gate rows stay immutable through score/value work.
    let projected_ref: &[Vec<f32>] = &tile.mixed[..count];
    let current_keys: &[Vec<f32>] = &tile.recurrent_gate[..count];
    let current_values: &[Vec<f32>] = &tile.beta[..count];
    let key_cache: &KvCache = keys;
    let value_cache: &KvCache = values;
    tile.projector.row_parallelism().update_paired_chunks(
        &mut tile.attended[..count],
        1,
        &mut attention_scratch.scores,
        token_score_stride,
        |base_tok, attended_chunk, scores_chunk| {
            let chunk_tokens = attended_chunk.len();
            if six_head_groups {
                let end = base_tok + chunk_tokens;
                return super::attention_six::attend_tile_chunk(
                    g,
                    base + base_tok,
                    &query_ref[base_tok..end],
                    &projected_ref[base_tok..end],
                    key_cache,
                    value_cache,
                    &current_keys[base_tok..end],
                    &current_values[base_tok..end],
                    attended_chunk,
                    scores_chunk,
                    token_score_stride,
                    scale,
                );
            }
            attend_generic_head_range(
                g,
                arithmetic,
                base,
                base_tok,
                chunk_tokens,
                0,
                g.heads,
                heads_per_kv,
                max_score_len,
                token_score_stride,
                scale,
                key_cache,
                value_cache,
                query_ref,
                projected_ref,
                current_keys,
                current_values,
                AttendedSlice::TokenRows(attended_chunk),
                scores_chunk,
            )
        },
    )
}

#[expect(
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
        && transforms.can_block_hadamard_for(first_target, len)
        && first_target.dimensions.first().copied() == Some(len)
    {
        let signs = transforms.signs_for(first_target, len);
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

#[expect(
    clippy::too_many_arguments,
    reason = "Keep fused tile residual-add, RMSNorm and Hadamard rotation explicit."
)]
fn add_normalize_hadamard_tile_into<'a>(
    transforms: &Transforms,
    geometry: &Geometry,
    arithmetic: Arithmetic,
    weight: &[f32],
    first_target: &'a Tensor,
    hidden: &mut [Vec<f32>],
    delta: &[Vec<f32>],
    normalized: &mut [Vec<f32>],
    transformed: &mut [Vec<f32>],
    parallelism: &RowParallelism,
) -> Result<TileReuse<'a>> {
    let count = hidden.len();
    let len = hidden.first().map_or(0, Vec::len);
    let signs = transforms.signs_for(first_target, len);
    let block = transforms.block;
    let epsilon = geometry.epsilon;
    if count > 1 && parallelism.workers() > 1 {
        let single_worker = RowParallelism::default();
        parallelism.update_triple_chunks(
            hidden,
            1,
            normalized,
            1,
            transformed,
            1,
            |base_tok, hidden_chunk, norm_chunk, trans_chunk| {
                for (offset, ((h, norm_out), rot_out)) in hidden_chunk
                    .iter_mut()
                    .zip(norm_chunk.iter_mut())
                    .zip(trans_chunk.iter_mut())
                    .enumerate()
                {
                    arithmetic.add_normalize_hadamard_into(
                        h,
                        &delta[base_tok + offset],
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
        )?;
    } else {
        for (((h, d), norm_out), rot_out) in hidden
            .iter_mut()
            .zip(delta.iter())
            .zip(normalized.iter_mut())
            .zip(transformed.iter_mut())
        {
            arithmetic.add_normalize_hadamard_into(
                h,
                d,
                weight,
                epsilon,
                block,
                signs,
                parallelism,
                norm_out,
                rot_out,
            )?;
        }
    }
    Ok(TileReuse {
        basis: Some(first_target),
        prepared: None,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "Keep fused tile residual-add, RMSNorm, Hadamard rotation and bounded tile scratch explicit."
)]
fn add_normalize_for_tile_into<'a>(
    transforms: &Transforms,
    geometry: &Geometry,
    arithmetic: Arithmetic,
    weight: &[f32],
    first_target: &'a Tensor,
    hidden: &mut [Vec<f32>],
    delta: &[Vec<f32>],
    normalized: &mut [Vec<f32>],
    transformed: &mut [Vec<f32>],
    parallelism: &RowParallelism,
) -> Result<TileReuse<'a>> {
    let count = hidden.len();
    let len = hidden.first().map_or(0, Vec::len);
    let can_hadamard = count > 0
        && len > 0
        && transforms.basis(first_target) == (true, false)
        && transforms.can_block_hadamard_for(first_target, len)
        && first_target.dimensions.first().copied() == Some(len);
    if can_hadamard {
        return add_normalize_hadamard_tile_into(
            transforms,
            geometry,
            arithmetic,
            weight,
            first_target,
            hidden,
            delta,
            normalized,
            transformed,
            parallelism,
        );
    }
    if count > 1 && parallelism.workers() > 1 {
        let epsilon = geometry.epsilon;
        parallelism.update_paired_chunks(
            hidden,
            1,
            normalized,
            1,
            |base_tok, hidden_chunk, norm_chunk| {
                for (offset, (h, norm_out)) in hidden_chunk
                    .iter_mut()
                    .zip(norm_chunk.iter_mut())
                    .enumerate()
                {
                    arithmetic.add_normalize(
                        h,
                        &delta[base_tok + offset],
                        weight,
                        epsilon,
                        norm_out,
                    )?;
                }
                Ok(())
            },
        )?;
    } else {
        for ((h, d), norm_out) in hidden
            .iter_mut()
            .zip(delta.iter())
            .zip(normalized.iter_mut())
        {
            arithmetic.add_normalize(h, d, weight, geometry.epsilon, norm_out)?;
        }
    }
    Ok(TileReuse::default())
}

#[expect(
    clippy::too_many_arguments,
    reason = "Dispatch fused residual-add RMSNorm when previous FFN delta is still pending."
)]
fn normalize_or_add_tile_into<'a>(
    pending_residual: bool,
    transforms: &Transforms,
    geometry: &Geometry,
    arithmetic: Arithmetic,
    weight: &[f32],
    first_target: &'a Tensor,
    hidden: &mut [Vec<f32>],
    delta: &[Vec<f32>],
    normalized: &mut [Vec<f32>],
    transformed: &mut [Vec<f32>],
    parallelism: &RowParallelism,
) -> Result<TileReuse<'a>> {
    if pending_residual {
        add_normalize_for_tile_into(
            transforms,
            geometry,
            arithmetic,
            weight,
            first_target,
            hidden,
            delta,
            normalized,
            transformed,
            parallelism,
        )
    } else {
        normalize_for_tile_into(
            transforms,
            geometry,
            arithmetic,
            weight,
            first_target,
            hidden,
            normalized,
            transformed,
            parallelism,
        )
    }
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

#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "Keep exact transforms, original-basis adapter and bounded tile storage explicit."
)]
fn project_tile_pair<'a>(
    transforms: &Transforms,
    geometry: &Geometry,
    arithmetic: Arithmetic,
    adapters: ActiveAdapters<'_>,
    adapter_scratch: &mut AdapterScratch,
    projection_scratch: &mut crate::kernels::MatvecScratch,
    previous: TileReuse<'a>,
    tensors: [&'a Tensor; 2],
    inputs: &[Vec<f32>],
    outputs_a: &mut [Vec<f32>],
    outputs_b: &mut [Vec<f32>],
    transformed: &mut [Vec<f32>],
    projector: &mut ProjectionTile,
    mut executor: Option<&mut (dyn crate::kernels::Pq2ProjectionExecutor + '_)>,
    check: &dyn Fn() -> Result<()>,
) -> Result<TileReuse<'a>> {
    if let Some(exec) = executor.as_deref_mut()
        && matches!(tensors[0].kind, 142 | 143)
        && matches!(tensors[1].kind, 142 | 143)
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
        if adapters.is_any_loaded() {
            adapters.apply_linear_tokens(&tensors[0].name, inputs, outputs_a, adapter_scratch)?;
            adapters.apply_linear_tokens(&tensors[1].name, inputs, outputs_b, adapter_scratch)?;
        }
        return Ok(TileReuse {
            basis: Some(tensors[1]),
            prepared: None,
        });
    }
    // Untargeted adapter calls are guaranteed no-ops. Apply LoRA adapters
    // immediately after the fused sibling base projections.
    if executor.is_none()
        && transforms.basis(tensors[0]) == transforms.basis(tensors[1])
        && ProjectionTile::supports_siblings(&tensors)
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
        let prepared = previous
            .prepared
            .filter(|prev| reuse_basis && transforms.basis(prev) == transforms.basis(tensors[0]));
        projector.project_siblings_with_previous(
            prepared,
            &tensors,
            activation,
            &mut [outputs_a, outputs_b],
            matches!(arithmetic, Arithmetic::PinnedSwift),
            check,
        )?;
        if adapters.is_any_loaded() {
            adapters.apply_linear_tokens(&tensors[0].name, inputs, outputs_a, adapter_scratch)?;
            adapters.apply_linear_tokens(&tensors[1].name, inputs, outputs_b, adapter_scratch)?;
        }
        return Ok(TileReuse {
            basis: Some(tensors[1]),
            prepared: Some(tensors[1]),
        });
    }
    let reuse = project_tile(
        transforms,
        geometry,
        arithmetic,
        adapters,
        adapter_scratch,
        projection_scratch,
        previous,
        tensors[0],
        inputs,
        outputs_a,
        transformed,
        projector,
        executor.as_deref_mut(),
        check,
    )?;
    check()?;
    project_tile(
        transforms,
        geometry,
        arithmetic,
        adapters,
        adapter_scratch,
        projection_scratch,
        reuse,
        tensors[1],
        inputs,
        outputs_b,
        transformed,
        projector,
        executor,
        check,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "Keep exact transforms, residual chaining and bounded tile storage explicit."
)]
fn try_project_tile_ffn_chain(
    transforms: &Transforms,
    geometry: &Geometry,
    arithmetic: Arithmetic,
    adapters: ActiveAdapters<'_>,
    previous: TileReuse<'_>,
    tensors: [&Tensor; 3],
    inputs: &[Vec<f32>],
    residuals: Option<&[Vec<f32>]>,
    outputs: &mut [Vec<f32>],
    transformed: &mut [Vec<f32>],
    projector: &ProjectionTile,
    executor: Option<&mut (dyn crate::kernels::Pq2ProjectionExecutor + '_)>,
    check: &dyn Fn() -> Result<()>,
) -> Result<bool> {
    let [gate, up, down] = tensors;
    let Some(hadamard_block) = transforms.ffn_chain_hadamard_block_for(down) else {
        return Ok(false);
    };
    if adapters.affects_any(&tensors)
        || !matches!(gate.kind, 142 | 143)
        || !matches!(up.kind, 142 | 143)
        || !matches!(down.kind, 142 | 143)
        || transforms.basis(gate) != transforms.basis(up)
    {
        return Ok(false);
    }
    let Some(exec) = executor else {
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

#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "Keep exact transforms, original-basis adapter and bounded tile storage explicit."
)]
fn project_tile_ffn_swiglu<'a>(
    transforms: &Transforms,
    geometry: &Geometry,
    arithmetic: Arithmetic,
    adapters: ActiveAdapters<'_>,
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
    mut executor: Option<&mut (dyn crate::kernels::Pq2ProjectionExecutor + '_)>,
    check: &dyn Fn() -> Result<()>,
) -> Result<TileReuse<'a>> {
    if !adapters.affects_any(&[gate, up])
        && !transforms.is_compacted_down(down)
        && let Some(exec) = executor.as_deref_mut()
        && matches!(gate.kind, 142 | 143)
        && matches!(up.kind, 142 | 143)
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
        adapters,
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
    if adapters.has_target(&down.name)
        && up_len > 0
        && transforms.basis(down) == (true, false)
        && transforms.can_block_hadamard_for(down, up_len)
        && down.dimensions.first().copied() == Some(up_len)
    {
        let signs = transforms.signs_for(down, up_len);
        let block = transforms.block;
        if count > 1 && projector.row_parallelism().workers() > 1 {
            let single_worker = RowParallelism::default();
            projector.row_parallelism().update_paired_chunks(
                &mut up_outputs[..count],
                1,
                &mut transformed[..count],
                1,
                |base_tok, up_chunk, out_chunk| {
                    for (offset, (up_out, target)) in up_chunk.iter_mut().zip(out_chunk).enumerate()
                    {
                        arithmetic.swiglu_hadamard_preserving_up_into(
                            &gate_outputs[base_tok + offset],
                            up_out,
                            block,
                            signs,
                            &single_worker,
                            target,
                        )?;
                    }
                    Ok(())
                },
            )?;
        } else {
            for ((up_out, gate_out), target) in up_outputs[..count]
                .iter_mut()
                .zip(&gate_outputs[..count])
                .zip(&mut transformed[..count])
            {
                arithmetic.swiglu_hadamard_preserving_up_into(
                    gate_out,
                    up_out,
                    block,
                    signs,
                    projector.row_parallelism(),
                    target,
                )?;
            }
        }
        return Ok(TileReuse {
            basis: Some(down),
            prepared: None,
        });
    }
    if !adapters.has_target(&down.name)
        && up_len > 0
        && transforms.basis(down) == (true, false)
        && transforms.can_block_hadamard_for(down, up_len)
        && down.dimensions.first().copied() == Some(up_len)
    {
        let signs = transforms.signs_for(down, up_len);
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

#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "Keep exact transforms, original-basis adapter and bounded tile storage explicit."
)]
fn project_tile_triple<'a>(
    transforms: &Transforms,
    geometry: &Geometry,
    arithmetic: Arithmetic,
    adapters: ActiveAdapters<'_>,
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
    mut executor: Option<&mut (dyn crate::kernels::Pq2ProjectionExecutor + '_)>,
    check: &dyn Fn() -> Result<()>,
) -> Result<TileReuse<'a>> {
    if let Some(exec) = executor.as_deref_mut()
        && matches!(tensors[0].kind, 142 | 143)
        && matches!(tensors[1].kind, 142 | 143)
        && matches!(tensors[2].kind, 142 | 143)
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
        if adapters.is_any_loaded() {
            for (tensor, outputs) in [
                (tensors[0], &mut *outputs_a),
                (tensors[1], &mut *outputs_b),
                (tensors[2], &mut *outputs_c),
            ] {
                adapters.apply_linear_tokens(&tensor.name, inputs, outputs, adapter_scratch)?;
            }
        }
        return Ok(TileReuse {
            basis: Some(tensors[2]),
            prepared: None,
        });
    }
    // Untargeted adapter calls are guaranteed no-ops. Apply LoRA adapters
    // immediately after the fused sibling base projections.
    if executor.is_none()
        && transforms.basis(tensors[0]) == transforms.basis(tensors[1])
        && transforms.basis(tensors[0]) == transforms.basis(tensors[2])
        && ProjectionTile::supports_siblings(&tensors)
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
        let prepared = previous
            .prepared
            .filter(|prev| reuse_basis && transforms.basis(prev) == transforms.basis(tensors[0]));
        projector.project_siblings_with_previous(
            prepared,
            &tensors,
            activation,
            &mut [outputs_a, outputs_b, outputs_c],
            matches!(arithmetic, Arithmetic::PinnedSwift),
            check,
        )?;
        if adapters.is_any_loaded() {
            for (tensor, outputs) in [
                (tensors[0], &mut *outputs_a),
                (tensors[1], &mut *outputs_b),
                (tensors[2], &mut *outputs_c),
            ] {
                adapters.apply_linear_tokens(&tensor.name, inputs, outputs, adapter_scratch)?;
            }
        }
        return Ok(TileReuse {
            basis: Some(tensors[2]),
            prepared: Some(tensors[2]),
        });
    }
    let reuse = project_tile_pair(
        transforms,
        geometry,
        arithmetic,
        adapters,
        adapter_scratch,
        projection_scratch,
        previous,
        [tensors[0], tensors[1]],
        inputs,
        outputs_a,
        outputs_b,
        transformed,
        projector,
        executor.as_deref_mut(),
        check,
    )?;
    check()?;
    project_tile(
        transforms,
        geometry,
        arithmetic,
        adapters,
        adapter_scratch,
        projection_scratch,
        reuse,
        tensors[2],
        inputs,
        outputs_c,
        transformed,
        projector,
        executor,
        check,
    )
}

#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "Keep exact transforms, trellis branch, original-basis adapter and bounded tile storage explicit."
)]
fn project_tile<'a>(
    transforms: &Transforms,
    geometry: &Geometry,
    arithmetic: Arithmetic,
    adapters: ActiveAdapters<'_>,
    adapter_scratch: &mut AdapterScratch,
    projection_scratch: &mut crate::kernels::MatvecScratch,
    previous: TileReuse<'a>,
    tensor: &'a Tensor,
    inputs: &[Vec<f32>],
    outputs: &mut [Vec<f32>],
    transformed: &mut [Vec<f32>],
    projector: &mut ProjectionTile,
    mut executor: Option<&mut (dyn crate::kernels::Pq2ProjectionExecutor + '_)>,
    check: &dyn Fn() -> Result<()>,
) -> Result<TileReuse<'a>> {
    if let Some(tw) = transforms.trellis_weight(&tensor.name) {
        if let Some(exec) = executor.as_deref_mut()
            && Transforms::executor_claims_weight(Some(&*exec), tensor)
        {
            return Err(invalid(
                "trellis weight cannot be offloaded to an external executor",
            ));
        }
        let backend = projector.requested_backend();
        let (row_parallelism, trellis) = projector.row_parallelism_and_trellis();
        crate::kernels::trellis::trellis_matmul_tokens_into(
            tw,
            inputs,
            outputs,
            backend,
            row_parallelism,
            trellis,
        )?;
        if adapters.is_any_loaded() {
            adapters.apply_linear_tokens(&tensor.name, inputs, outputs, adapter_scratch)?;
        }
        return Ok(TileReuse::default());
    }
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
    let prepared = if let Some(exec) = executor {
        let cols = tensor.dimensions.first().copied().unwrap_or(0);
        let rows = tensor
            .dimensions
            .get(1..)
            .map_or(0, |dims| dims.iter().product::<usize>());
        if matches!(tensor.kind, 142 | 143) && cols.is_multiple_of(128) && exec.supports_f32(tensor)
        {
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
    if adapters.is_any_loaded() {
        adapters.apply_linear_tokens(&tensor.name, inputs, outputs, adapter_scratch)?;
    }
    Ok(TileReuse {
        basis: Some(tensor),
        prepared,
    })
}

#[cfg(test)]
mod chunkwise_kv_tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use bonsai_gguf::MetadataValue;

    use super::super::{KvCachePolicy, MAX_STATE_BYTES, Qwen35};
    use crate::kernels::MatvecBackend;
    use crate::model::{Model, Tensor};
    use crate::{Decoder, Result};

    fn f32_tensor(name: &str, dims: &[usize], values: &[f32]) -> Tensor {
        Tensor {
            name: name.to_owned(),
            dimensions: dims.to_vec(),
            kind: 0,
            data: Arc::<[u8]>::from(
                values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .into(),
        }
    }

    fn insert_hybrid_gdn_layer(
        tensors: &mut BTreeMap<String, Tensor>,
        emb: usize,
        head: usize,
        pat: &dyn Fn(usize, usize) -> Vec<f32>,
        ones: &dyn Fn(usize) -> Vec<f32>,
    ) {
        tensors.extend([
            (
                "blk.0.attn_norm.weight".into(),
                f32_tensor("blk.0.attn_norm.weight", &[emb], &ones(emb)),
            ),
            (
                "blk.0.post_attention_norm.weight".into(),
                f32_tensor("blk.0.post_attention_norm.weight", &[emb], &ones(emb)),
            ),
            (
                "blk.0.attn_qkv.weight".into(),
                f32_tensor(
                    "blk.0.attn_qkv.weight",
                    &[emb, 3 * head],
                    &pat(emb * 3 * head, 3),
                ),
            ),
            (
                "blk.0.attn_gate.weight".into(),
                f32_tensor("blk.0.attn_gate.weight", &[emb, head], &pat(emb * head, 4)),
            ),
            (
                "blk.0.ssm_alpha.weight".into(),
                f32_tensor("blk.0.ssm_alpha.weight", &[emb, 1], &pat(emb, 5)),
            ),
            (
                "blk.0.ssm_beta.weight".into(),
                f32_tensor("blk.0.ssm_beta.weight", &[emb, 1], &pat(emb, 6)),
            ),
            (
                "blk.0.ssm_conv1d.weight".into(),
                f32_tensor(
                    "blk.0.ssm_conv1d.weight",
                    &[2, 3 * head],
                    &pat(2 * 3 * head, 7),
                ),
            ),
            (
                "blk.0.ssm_dt.bias".into(),
                f32_tensor("blk.0.ssm_dt.bias", &[1], &[0.125]),
            ),
            (
                "blk.0.ssm_a".into(),
                f32_tensor("blk.0.ssm_a", &[1], &[-0.25]),
            ),
            (
                "blk.0.ssm_norm.weight".into(),
                f32_tensor("blk.0.ssm_norm.weight", &[head], &ones(head)),
            ),
            (
                "blk.0.ssm_out.weight".into(),
                f32_tensor("blk.0.ssm_out.weight", &[head, emb], &pat(head * emb, 8)),
            ),
        ]);
    }

    fn hybrid_gdn_rotated_kv_metadata() -> BTreeMap<String, MetadataValue> {
        BTreeMap::from([
            ("general.architecture".into(), MetadataValue::from("qwen35")),
            ("qwen35.block_count".into(), MetadataValue::from(2_u32)),
            ("qwen35.context_length".into(), MetadataValue::from(64_u32)),
            (
                "qwen35.embedding_length".into(),
                MetadataValue::from(64_u32),
            ),
            (
                "qwen35.feed_forward_length".into(),
                MetadataValue::from(128_u32),
            ),
            (
                "qwen35.attention.head_count".into(),
                MetadataValue::from(1_u32),
            ),
            (
                "qwen35.attention.head_count_kv".into(),
                MetadataValue::from(1_u32),
            ),
            (
                "qwen35.attention.key_length".into(),
                MetadataValue::from(256_u32),
            ),
            (
                "qwen35.attention.value_length".into(),
                MetadataValue::from(256_u32),
            ),
            (
                "qwen35.rope.dimension_count".into(),
                MetadataValue::from(64_u32),
            ),
            (
                "qwen35.rope.freq_base".into(),
                MetadataValue::from(10_000.0_f32),
            ),
            (
                "qwen35.attention.layer_norm_rms_epsilon".into(),
                MetadataValue::from(1e-6_f32),
            ),
            (
                "qwen35.full_attention_interval".into(),
                MetadataValue::from(2_u32),
            ),
            ("qwen35.ssm.state_size".into(), MetadataValue::from(64_u32)),
            ("qwen35.ssm.group_count".into(), MetadataValue::from(1_u32)),
            (
                "qwen35.ssm.time_step_rank".into(),
                MetadataValue::from(1_u32),
            ),
            ("qwen35.ssm.inner_size".into(), MetadataValue::from(64_u32)),
            ("qwen35.ssm.conv_kernel".into(), MetadataValue::from(2_u32)),
            (
                "qwen35.rope.dimension_sections".into(),
                MetadataValue::from(vec![16_u32, 16, 0, 0]),
            ),
            (
                "qwen35.attention.recurrent_layers".into(),
                MetadataValue::from(vec![true, false]),
            ),
        ])
    }

    fn insert_hybrid_full_attn_and_ffn(
        tensors: &mut BTreeMap<String, Tensor>,
        emb: usize,
        ffn: usize,
        full_head: usize,
        pat: &dyn Fn(usize, usize) -> Vec<f32>,
        ones: &dyn Fn(usize) -> Vec<f32>,
    ) {
        tensors.extend([
            (
                "blk.1.attn_norm.weight".into(),
                f32_tensor("blk.1.attn_norm.weight", &[emb], &ones(emb)),
            ),
            (
                "blk.1.post_attention_norm.weight".into(),
                f32_tensor("blk.1.post_attention_norm.weight", &[emb], &ones(emb)),
            ),
            (
                "blk.1.attn_q.weight".into(),
                f32_tensor(
                    "blk.1.attn_q.weight",
                    &[emb, 2 * full_head],
                    &pat(emb * 2 * full_head, 9),
                ),
            ),
            (
                "blk.1.attn_k.weight".into(),
                f32_tensor(
                    "blk.1.attn_k.weight",
                    &[emb, full_head],
                    &pat(emb * full_head, 10),
                ),
            ),
            (
                "blk.1.attn_v.weight".into(),
                f32_tensor(
                    "blk.1.attn_v.weight",
                    &[emb, full_head],
                    &pat(emb * full_head, 11),
                ),
            ),
            (
                "blk.1.attn_output.weight".into(),
                f32_tensor(
                    "blk.1.attn_output.weight",
                    &[full_head, emb],
                    &pat(full_head * emb, 12),
                ),
            ),
            (
                "blk.1.attn_q_norm.weight".into(),
                f32_tensor("blk.1.attn_q_norm.weight", &[full_head], &ones(full_head)),
            ),
            (
                "blk.1.attn_k_norm.weight".into(),
                f32_tensor("blk.1.attn_k_norm.weight", &[full_head], &ones(full_head)),
            ),
        ]);
        for layer in 0..2_usize {
            let prefix = format!("blk.{layer}.");
            tensors.extend([
                (
                    format!("{prefix}ffn_up.weight"),
                    f32_tensor(
                        &format!("{prefix}ffn_up.weight"),
                        &[emb, ffn],
                        &pat(emb * ffn, 13 + layer),
                    ),
                ),
                (
                    format!("{prefix}ffn_gate.weight"),
                    f32_tensor(
                        &format!("{prefix}ffn_gate.weight"),
                        &[emb, ffn],
                        &pat(emb * ffn, 15 + layer),
                    ),
                ),
                (
                    format!("{prefix}ffn_down.weight"),
                    f32_tensor(
                        &format!("{prefix}ffn_down.weight"),
                        &[ffn, emb],
                        &pat(ffn * emb, 17 + layer),
                    ),
                ),
            ]);
        }
    }

    fn hybrid_gdn_rotated_kv_fixture() -> Model {
        let emb = 64_usize;
        let ffn = 128_usize;
        let ssm_head = 64_usize;
        let full_head = 256_usize;
        let vocab = 6_usize;
        let metadata = hybrid_gdn_rotated_kv_metadata();
        let pat = |len: usize, seed: usize| -> Vec<f32> {
            (0..len)
                .map(|i| [0.0625_f32, -0.03125, 0.015_625, -0.0625, 0.03125][(i + seed) % 5])
                .collect()
        };
        let ones = |len: usize| vec![1.0_f32; len];
        let mut tensors = BTreeMap::from([
            (
                "token_embd.weight".into(),
                f32_tensor("token_embd.weight", &[emb, vocab], &pat(emb * vocab, 1)),
            ),
            (
                "output.weight".into(),
                f32_tensor("output.weight", &[emb, vocab], &pat(emb * vocab, 2)),
            ),
            (
                "output_norm.weight".into(),
                f32_tensor("output_norm.weight", &[emb], &ones(emb)),
            ),
        ]);
        insert_hybrid_gdn_layer(&mut tensors, emb, ssm_head, &pat, &ones);
        insert_hybrid_full_attn_and_ffn(&mut tensors, emb, ffn, full_head, &pat, &ones);
        Model {
            metadata,
            tensors,
            identity: "hybrid-gdn-rotated-kv-fixture".into(),
        }
    }

    #[test]
    fn chunkwise_gdn_works_on_fp16_and_all_rotated_kv_policies() -> Result<()> {
        let prompt = [0_u32, 1, 2, 3, 1, 2];
        for policy in [
            KvCachePolicy::F16,
            KvCachePolicy::Rotated4,
            KvCachePolicy::Rotated4FittedValues,
            KvCachePolicy::Rotated4Keys3Values,
        ] {
            let seq_base = Qwen35::with_adapters_and_kv(
                hybrid_gdn_rotated_kv_fixture(),
                16,
                MAX_STATE_BYTES,
                MatvecBackend::Scalar,
                None,
                policy,
            )?;
            let scratch = seq_base.tiled_prefill_scratch_bytes(8)?;
            let mut sequential = seq_base.with_tiled_prefill(8, scratch)?;
            sequential.ingest_block(&prompt, &|| Ok(()))?;
            let mut expected = Vec::new();
            sequential.evaluate_into(4, &mut expected)?;

            let wy_base = Qwen35::with_adapters_and_kv(
                hybrid_gdn_rotated_kv_fixture(),
                16,
                MAX_STATE_BYTES,
                MatvecBackend::Scalar,
                None,
                policy,
            )?;
            let scratch_wy = wy_base.tiled_prefill_scratch_bytes(8)?;
            let mut chunkwise = wy_base
                .with_tiled_prefill(8, scratch_wy)?
                .with_chunkwise_gdn_prefill()?;
            assert!(chunkwise.identity().ends_with(":gdn-wy-f32-v1"));
            assert_ne!(chunkwise.identity(), sequential.identity());
            chunkwise.ingest_block(&prompt, &|| Ok(()))?;
            assert!(chunkwise.tiled_prefill_statistics().chunkwise_gdn_heads > 0);
            let mut actual = Vec::new();
            chunkwise.evaluate_into(4, &mut actual)?;
            for (&a, &e) in actual.iter().zip(&expected) {
                let tol = 1e-3_f32.mul_add(e.abs(), 1e-3_f32);
                assert!(
                    (a - e).abs() <= tol,
                    "policy={policy:?} |{a} - {e}| > {tol}"
                );
            }

            let verify_base = Qwen35::with_adapters_and_kv(
                hybrid_gdn_rotated_kv_fixture(),
                16,
                MAX_STATE_BYTES,
                MatvecBackend::Scalar,
                None,
                policy,
            )?;
            let scratch_v = verify_base.tiled_prefill_scratch_bytes(8)?;
            let wy_for_verify = verify_base
                .with_tiled_prefill(8, scratch_v)?
                .with_chunkwise_gdn_prefill()?;
            assert!(wy_for_verify.tiled_verification_scratch_bytes().is_err());
        }
        Ok(())
    }
}

#[cfg(test)]
mod ptq1_executor_tests {
    use std::sync::Arc;

    use super::{
        ActiveAdapters, AdapterScratch, Arithmetic, Geometry, MatvecBackend, ProjectionTile,
        Tensor, TileReuse, Transforms, project_tile, project_tile_ffn_swiglu, project_tile_pair,
        project_tile_triple, try_project_tile_ffn_chain,
    };
    use crate::Result;
    use crate::kernels::{MatvecScratch, Pq2ProjectionExecutor, Q8Block, Q8KBlock, matvec_into};

    struct TiledPtq1Exec {
        allow_f32: bool,
        allow_q8: bool,
        f32_tokens_calls: usize,
        pair_calls: usize,
        triple_calls: usize,
        swiglu_tokens_calls: usize,
        chain_tokens_calls: usize,
        q8_calls: usize,
    }

    impl Pq2ProjectionExecutor for TiledPtq1Exec {
        fn identity(&self) -> &'static str {
            "tiled-ptq1-test"
        }

        fn supports(&self, _tensor: &Tensor) -> bool {
            true
        }

        fn supports_q8(&self, _tensor: &Tensor) -> bool {
            self.allow_q8
        }

        fn supports_f32(&self, _tensor: &Tensor) -> bool {
            self.allow_f32
        }

        fn supports_f32_group(&self, _tensors: &[&Tensor]) -> bool {
            self.allow_f32
        }

        fn supports_f32_ffn_swiglu(&self, _gate: &Tensor, _up: &Tensor) -> bool {
            self.allow_f32
        }

        fn supports_f32_ffn_chain(&self, _tensors: [&Tensor; 3]) -> bool {
            self.allow_f32
        }

        fn project(
            &mut self,
            _tensor: &Tensor,
            _activation: &[Q8KBlock],
            _fused: bool,
            _output: &mut Vec<f32>,
        ) -> Result<()> {
            Err(crate::invalid("PTQ1 must never dispatch to Q8_K"))
        }

        fn project_q8(
            &mut self,
            tensor: &Tensor,
            _activation: &[Q8Block],
            _pinned: bool,
            output: &mut Vec<f32>,
        ) -> Result<()> {
            self.q8_calls += 1;
            output.clear();
            output.resize(tensor.dimensions[1], 6.5);
            Ok(())
        }

        fn project_f32_tokens(
            &mut self,
            tensor: &Tensor,
            activations: &[Vec<f32>],
            _pinned: bool,
            outputs: &mut [Vec<f32>],
        ) -> Result<()> {
            self.f32_tokens_calls += 1;
            assert_eq!(activations.len(), outputs.len());
            for out in outputs.iter_mut() {
                out.clear();
                out.resize(tensor.dimensions[1], 1.5);
            }
            Ok(())
        }

        fn project_f32_pair_tokens(
            &mut self,
            tensors: [&Tensor; 2],
            activations: &[Vec<f32>],
            _pinned: bool,
            outputs_a: &mut [Vec<f32>],
            outputs_b: &mut [Vec<f32>],
        ) -> Result<()> {
            self.pair_calls += 1;
            assert_eq!(activations.len(), outputs_a.len());
            for (a, b) in outputs_a.iter_mut().zip(outputs_b.iter_mut()) {
                a.clear();
                a.resize(tensors[0].dimensions[1], 2.5);
                b.clear();
                b.resize(tensors[1].dimensions[1], 3.5);
            }
            Ok(())
        }

        fn project_f32_triple_tokens(
            &mut self,
            tensors: [&Tensor; 3],
            activations: &[Vec<f32>],
            _pinned: bool,
            outputs_a: &mut [Vec<f32>],
            outputs_b: &mut [Vec<f32>],
            outputs_c: &mut [Vec<f32>],
        ) -> Result<()> {
            self.triple_calls += 1;
            assert_eq!(activations.len(), outputs_a.len());
            for ((a, b), c) in outputs_a
                .iter_mut()
                .zip(outputs_b.iter_mut())
                .zip(outputs_c.iter_mut())
            {
                a.clear();
                a.resize(tensors[0].dimensions[1], 4.5);
                b.clear();
                b.resize(tensors[1].dimensions[1], 5.5);
                c.clear();
                c.resize(tensors[2].dimensions[1], 6.5);
            }
            Ok(())
        }

        fn project_f32_ffn_swiglu_tokens(
            &mut self,
            gate: &Tensor,
            _up: &Tensor,
            activations: &[Vec<f32>],
            _pinned: bool,
            outputs: &mut [Vec<f32>],
        ) -> Result<()> {
            self.swiglu_tokens_calls += 1;
            assert_eq!(activations.len(), outputs.len());
            for out in outputs.iter_mut() {
                out.clear();
                out.resize(gate.dimensions[1], 7.5);
            }
            Ok(())
        }

        fn project_f32_ffn_chain_tokens(
            &mut self,
            tensors: [&Tensor; 3],
            activations: &[Vec<f32>],
            _residuals: Option<&[Vec<f32>]>,
            _hadamard_block_size: Option<u32>,
            _pinned: bool,
            outputs: &mut [Vec<f32>],
        ) -> Result<()> {
            self.chain_tokens_calls += 1;
            assert_eq!(activations.len(), outputs.len());
            for out in outputs.iter_mut() {
                out.clear();
                out.resize(tensors[2].dimensions[1], 8.5);
            }
            Ok(())
        }
    }

    fn ptq1_tensor(name: &str, cols: usize, rows: usize) -> Result<Tensor> {
        let mut pq2_block = [0x66_u8; 34];
        pq2_block[..2].copy_from_slice(&0x3555_u16.to_le_bytes());
        let block = bonsai_ternary::pack_ptq(&pq2_block)?;
        Ok(Tensor {
            name: name.into(),
            dimensions: vec![cols, rows],
            kind: 143,
            data: Arc::<[u8]>::from(block.repeat((cols / 128) * rows)).into(),
        })
    }

    fn pq2_tensor(name: &str, cols: usize, rows: usize) -> Tensor {
        let mut block = [0x55_u8; 34];
        block[..2].copy_from_slice(&0x3555_u16.to_le_bytes());
        Tensor {
            name: name.into(),
            dimensions: vec![cols, rows],
            kind: 142,
            data: Arc::<[u8]>::from(block.repeat((cols / 128) * rows)).into(),
        }
    }

    fn test_geometry() -> Geometry {
        Geometry {
            embedding: 256,
            feed_forward: 128,
            heads: 2,
            kv_heads: 1,
            head: 128,
            rope: 64,
            rope_sections: [32, 0, 0, 0],
            rope_base: 10_000.0,
            epsilon: 1e-6,
            ssm_head: 128,
            ssm_keys: 1,
            ssm_values: 1,
            convolution: 2,
        }
    }

    fn new_exec(allow_f32: bool, allow_q8: bool) -> TiledPtq1Exec {
        TiledPtq1Exec {
            allow_f32,
            allow_q8,
            f32_tokens_calls: 0,
            pair_calls: 0,
            triple_calls: 0,
            swiglu_tokens_calls: 0,
            chain_tokens_calls: 0,
            q8_calls: 0,
        }
    }

    #[test]
    fn ptq1_tiled_prefill_dispatches_supports_f32_tile_and_group_paths() -> Result<()> {
        let transforms = Transforms::default();
        let geometry = test_geometry();
        let q_pq2 = pq2_tensor("q_pq2", 256, 128);
        let q_ptq1 = ptq1_tensor("q_ptq1", 256, 128)?;
        let k_ptq1 = ptq1_tensor("k_ptq1", 256, 128)?;
        let v_ptq1 = ptq1_tensor("v_ptq1", 256, 128)?;
        let inputs = vec![vec![0.5_f32; 256], vec![-0.25_f32; 256]];
        let mut out_a = vec![Vec::new(); 2];
        let mut out_b = vec![Vec::new(); 2];
        let mut out_c = vec![Vec::new(); 2];
        let mut transformed = vec![Vec::new(); 2];
        let mut adapter_scratch = AdapterScratch::default();
        let mut proj_scratch = MatvecScratch::default();
        let mut projector = ProjectionTile::with_backend(MatvecBackend::Scalar);
        let mut exec = new_exec(true, false);

        let _ = project_tile(
            &transforms,
            &geometry,
            Arithmetic::Source,
            ActiveAdapters::default(),
            &mut adapter_scratch,
            &mut proj_scratch,
            TileReuse::default(),
            &q_ptq1,
            &inputs,
            &mut out_a,
            &mut transformed,
            &mut projector,
            Some(&mut exec),
            &|| Ok(()),
        )?;
        assert_eq!(out_a[0][0], 1.5);

        let _ = project_tile_pair(
            &transforms,
            &geometry,
            Arithmetic::Source,
            ActiveAdapters::default(),
            &mut adapter_scratch,
            &mut proj_scratch,
            TileReuse::default(),
            [&q_pq2, &k_ptq1],
            &inputs,
            &mut out_a,
            &mut out_b,
            &mut transformed,
            &mut projector,
            Some(&mut exec),
            &|| Ok(()),
        )?;
        assert_eq!((out_a[0][0], out_b[1][0]), (2.5, 3.5));

        let _ = project_tile_triple(
            &transforms,
            &geometry,
            Arithmetic::Source,
            ActiveAdapters::default(),
            &mut adapter_scratch,
            &mut proj_scratch,
            TileReuse::default(),
            [&q_pq2, &k_ptq1, &v_ptq1],
            &inputs,
            &mut out_a,
            &mut out_b,
            &mut out_c,
            &mut transformed,
            &mut projector,
            Some(&mut exec),
            &|| Ok(()),
        )?;
        assert_eq!((out_a[0][0], out_b[0][0], out_c[1][0]), (4.5, 5.5, 6.5));
        assert_eq!(
            (exec.f32_tokens_calls, exec.pair_calls, exec.triple_calls),
            (1, 1, 1)
        );
        Ok(())
    }

    #[test]
    fn ptq1_tiled_prefill_dispatches_supports_f32_ffn_paths() -> Result<()> {
        let transforms = Transforms::default();
        let geometry = test_geometry();
        let gate_ptq1 = ptq1_tensor("gate_ptq1", 256, 128)?;
        let up_ptq1 = ptq1_tensor("up_ptq1", 256, 128)?;
        let down_ptq1 = ptq1_tensor("down_ptq1", 128, 256)?;
        let inputs = vec![vec![0.5_f32; 256], vec![-0.25_f32; 256]];
        let mut out_a = vec![Vec::new(); 2];
        let mut out_b = vec![Vec::new(); 2];
        let mut out_c = vec![Vec::new(); 2];
        let mut transformed = vec![Vec::new(); 2];
        let mut adapter_scratch = AdapterScratch::default();
        let mut proj_scratch = MatvecScratch::default();
        let mut projector = ProjectionTile::with_backend(MatvecBackend::Scalar);
        let mut exec = new_exec(true, false);

        let _ = project_tile_ffn_swiglu(
            &transforms,
            &geometry,
            Arithmetic::Source,
            ActiveAdapters::default(),
            &mut adapter_scratch,
            &mut proj_scratch,
            TileReuse::default(),
            &gate_ptq1,
            &up_ptq1,
            &down_ptq1,
            &inputs,
            &mut out_a,
            &mut out_b,
            &mut transformed,
            &mut projector,
            Some(&mut exec),
            &|| Ok(()),
        )?;
        assert_eq!(out_b[1][0], 7.5);

        let chained = try_project_tile_ffn_chain(
            &transforms,
            &geometry,
            Arithmetic::Source,
            ActiveAdapters::default(),
            TileReuse::default(),
            [&gate_ptq1, &up_ptq1, &down_ptq1],
            &inputs,
            None,
            &mut out_c,
            &mut transformed,
            &projector,
            Some(&mut exec),
            &|| Ok(()),
        )?;
        assert!(chained);
        assert_eq!(out_c[1][0], 8.5);
        assert_eq!((exec.swiglu_tokens_calls, exec.chain_tokens_calls), (1, 1));
        Ok(())
    }

    #[test]
    fn ptq1_tiled_prefill_falls_back_to_q8_and_cpu_when_supports_f32_is_false() -> Result<()> {
        let transforms = Transforms::default();
        let geometry = test_geometry();
        let ptq1 = ptq1_tensor("q_ptq1", 256, 4)?;
        let inputs = vec![vec![0.5_f32; 256], vec![-0.25_f32; 256]];
        let mut expected = vec![Vec::new(); 2];
        let mut cpu_scratch = MatvecScratch::default();
        for (inp, exp) in inputs.iter().zip(&mut expected) {
            matvec_into(&ptq1, inp, exp, &mut cpu_scratch)?;
        }

        let mut outputs = vec![Vec::new(); 2];
        let mut transformed = vec![Vec::new(); 2];
        let mut adapter_scratch = AdapterScratch::default();
        let mut proj_scratch = MatvecScratch::default();
        let mut projector = ProjectionTile::with_backend(MatvecBackend::Scalar);

        let mut q8_exec = TiledPtq1Exec {
            allow_f32: false,
            allow_q8: true,
            f32_tokens_calls: 0,
            pair_calls: 0,
            triple_calls: 0,
            swiglu_tokens_calls: 0,
            chain_tokens_calls: 0,
            q8_calls: 0,
        };
        let _ = project_tile(
            &transforms,
            &geometry,
            Arithmetic::Source,
            ActiveAdapters::default(),
            &mut adapter_scratch,
            &mut proj_scratch,
            TileReuse::default(),
            &ptq1,
            &inputs,
            &mut outputs,
            &mut transformed,
            &mut projector,
            Some(&mut q8_exec),
            &|| Ok(()),
        )?;
        assert_eq!(q8_exec.f32_tokens_calls, 0);
        assert_eq!(q8_exec.q8_calls, 2);
        assert_eq!(outputs, vec![vec![6.5_f32; 4], vec![6.5_f32; 4]]);

        let mut cpu_exec = TiledPtq1Exec {
            allow_f32: false,
            allow_q8: false,
            f32_tokens_calls: 0,
            pair_calls: 0,
            triple_calls: 0,
            swiglu_tokens_calls: 0,
            chain_tokens_calls: 0,
            q8_calls: 0,
        };
        let _ = project_tile(
            &transforms,
            &geometry,
            Arithmetic::Source,
            ActiveAdapters::default(),
            &mut adapter_scratch,
            &mut proj_scratch,
            TileReuse::default(),
            &ptq1,
            &inputs,
            &mut outputs,
            &mut transformed,
            &mut projector,
            Some(&mut cpu_exec),
            &|| Ok(()),
        )?;
        assert_eq!(cpu_exec.f32_tokens_calls, 0);
        assert_eq!(cpu_exec.q8_calls, 0);
        assert_eq!(outputs, expected);
        Ok(())
    }
}

#[cfg(test)]
mod state_only_gdn_tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use bonsai_gguf::MetadataValue;

    use super::super::{
        ActiveAdapters, AdapterScratch, Attention, LayerState, MAX_STATE_BYTES, Qwen35,
    };
    use super::{
        Arithmetic, StateStepInput, TileReuse, recurrent_state_only_tile,
        step_recurrent_state_only_head_pinned, step_recurrent_state_only_rows_single,
    };
    use crate::kernels::{MatvecBackend, MatvecScratch};
    use crate::model::{Model, Tensor};
    use crate::{Result, invalid};

    fn values(len: usize, mut seed: u32) -> Vec<f32> {
        (0..len)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let mag = f32::from_bits((seed & 0x007f_ffff) | 0x3d00_0000);
                if (seed >> 31) != 0 { -mag } else { mag }
            })
            .collect()
    }

    fn f32_tensor(name: &str, dims: &[usize], data: &[f32]) -> Tensor {
        Tensor {
            name: name.to_owned(),
            dimensions: dims.to_vec(),
            kind: 0,
            data: Arc::<[u8]>::from(
                data.iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .into(),
        }
    }

    fn single_recurrent_metadata(head_dim: usize) -> BTreeMap<String, MetadataValue> {
        let dim = u32::try_from(head_dim).unwrap_or(32);
        BTreeMap::from([
            ("general.architecture".into(), MetadataValue::from("qwen35")),
            ("qwen35.block_count".into(), MetadataValue::from(1_u32)),
            ("qwen35.context_length".into(), MetadataValue::from(32_u32)),
            ("qwen35.embedding_length".into(), MetadataValue::from(dim)),
            (
                "qwen35.feed_forward_length".into(),
                MetadataValue::from(dim),
            ),
            (
                "qwen35.attention.head_count".into(),
                MetadataValue::from(1_u32),
            ),
            (
                "qwen35.attention.head_count_kv".into(),
                MetadataValue::from(1_u32),
            ),
            (
                "qwen35.attention.key_length".into(),
                MetadataValue::from(dim),
            ),
            (
                "qwen35.attention.value_length".into(),
                MetadataValue::from(dim),
            ),
            (
                "qwen35.attention.layer_norm_rms_epsilon".into(),
                MetadataValue::from(1e-6_f32),
            ),
            (
                "qwen35.rope.dimension_count".into(),
                MetadataValue::from(dim),
            ),
            (
                "qwen35.rope.dimension_sections".into(),
                MetadataValue::from(vec![8_u32, 8, 0, 0]),
            ),
            (
                "qwen35.rope.freq_base".into(),
                MetadataValue::from(10_000.0_f32),
            ),
            ("qwen35.ssm.state_size".into(), MetadataValue::from(dim)),
            ("qwen35.ssm.group_count".into(), MetadataValue::from(1_u32)),
            (
                "qwen35.ssm.time_step_rank".into(),
                MetadataValue::from(1_u32),
            ),
            ("qwen35.ssm.inner_size".into(), MetadataValue::from(dim)),
            ("qwen35.ssm.conv_kernel".into(), MetadataValue::from(2_u32)),
            (
                "qwen35.full_attention_interval".into(),
                MetadataValue::from(1_u32),
            ),
            (
                "qwen35.attention.recurrent_layers".into(),
                MetadataValue::from(vec![true]),
            ),
        ])
    }

    fn single_recurrent_layer_fixture(head_dim: usize) -> Model {
        let mut seed = 0x1357_9bdf_u32;
        let mut pat = |len: usize| -> Vec<f32> {
            seed = seed.wrapping_add(1);
            values(len, seed)
        };
        let ones = |len: usize| vec![1.0_f32; len];
        let vocab = 6_usize;
        let metadata = single_recurrent_metadata(head_dim);
        let mut tensors = BTreeMap::new();
        let mut add = |name: &str, dims: &[usize], data: Vec<f32>| {
            tensors.insert(name.into(), f32_tensor(name, dims, &data));
        };
        add(
            "token_embd.weight",
            &[head_dim, vocab],
            pat(head_dim * vocab),
        );
        add("output.weight", &[head_dim, vocab], pat(head_dim * vocab));
        add("output_norm.weight", &[head_dim], ones(head_dim));
        add("blk.0.attn_norm.weight", &[head_dim], ones(head_dim));
        add(
            "blk.0.post_attention_norm.weight",
            &[head_dim],
            ones(head_dim),
        );
        add(
            "blk.0.ffn_up.weight",
            &[head_dim, head_dim],
            pat(head_dim * head_dim),
        );
        add(
            "blk.0.ffn_gate.weight",
            &[head_dim, head_dim],
            pat(head_dim * head_dim),
        );
        add(
            "blk.0.ffn_down.weight",
            &[head_dim, head_dim],
            pat(head_dim * head_dim),
        );
        add(
            "blk.0.attn_qkv.weight",
            &[head_dim, 3 * head_dim],
            pat(head_dim * 3 * head_dim),
        );
        add(
            "blk.0.attn_gate.weight",
            &[head_dim, head_dim],
            pat(head_dim * head_dim),
        );
        add(
            "blk.0.ssm_conv1d.weight",
            &[2, 3 * head_dim],
            pat(2 * 3 * head_dim),
        );
        add("blk.0.ssm_dt.bias", &[1], vec![0.125]);
        add("blk.0.ssm_a", &[1], vec![-0.25]);
        add("blk.0.ssm_beta.weight", &[head_dim, 1], pat(head_dim));
        add("blk.0.ssm_alpha.weight", &[head_dim, 1], pat(head_dim));
        add("blk.0.ssm_norm.weight", &[head_dim], ones(head_dim));
        add(
            "blk.0.ssm_out.weight",
            &[head_dim, head_dim],
            pat(head_dim * head_dim),
        );
        Model {
            metadata,
            tensors,
            identity: "state-only-gdn-test-fixture".into(),
        }
    }

    #[test]
    fn state_only_gdn_pinned_four_row_matches_single_row_zero_ulp() -> Result<()> {
        for width in [4_usize, 32, 64, 128, 131] {
            let mut prev = values(width * width, 0x9e37_79b9);
            prev[0] = -0.0;
            prev[width - 1] = f32::from_bits(1);
            for count in [1_usize, 2, 4, 16, 32] {
                let mut mixed = Vec::with_capacity(count);
                let mut alpha = Vec::with_capacity(count);
                let mut beta = Vec::with_capacity(count);
                let mut step_inputs = Vec::with_capacity(count);
                let key_width = width;
                for token in 0..count {
                    let row = values(3 * width, 0x1000_0001 + u32::try_from(token).unwrap_or(0));
                    mixed.push(row);
                    alpha.push(vec![[0.875_f32, 0.0, -0.0, -0.25, 1.0][token % 5]]);
                    beta.push(vec![[0.625_f32, 0.375, 0.0, -0.0, 0.5][token % 5]]);
                }
                for ((m, a), b) in mixed.iter().zip(&alpha).zip(&beta) {
                    step_inputs.push(StateStepInput {
                        key: &m[key_width..2 * key_width],
                        value: &m[2 * key_width..3 * key_width],
                        decay: a[0],
                        beta: b[0],
                    });
                }

                let mut pinned_state = vec![f32::NAN; width * width];
                step_recurrent_state_only_head_pinned(
                    Arithmetic::PinnedSwift,
                    &step_inputs,
                    width,
                    &prev,
                    &mut pinned_state,
                )?;

                let mut ref_state = vec![f32::NAN; width * width];
                step_recurrent_state_only_rows_single(
                    Arithmetic::PinnedSwift,
                    &step_inputs,
                    0,
                    width,
                    &prev,
                    &mut ref_state,
                )?;
                assert_eq!(
                    pinned_state.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    ref_state.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "width={width} count={count}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn state_only_gdn_pinned_rejects_nonfinite_without_masking() {
        for width in [32_usize, 131] {
            let prev = vec![0.0_f32; width * width];
            let mut k0 = vec![0.0_f32; width];
            k0[width - 1] = f32::MAX;
            let k1 = vec![0.0_f32; width];
            let mut v0 = vec![0.0_f32; width];
            v0[2] = 4.0;
            let inputs = [
                StateStepInput {
                    key: &k0,
                    value: &v0,
                    decay: 1.0,
                    beta: 1.0,
                },
                StateStepInput {
                    key: &k1,
                    value: &v0,
                    decay: 0.0,
                    beta: 1.0,
                },
            ];
            let mut state = vec![-1.0_f32; width * width];
            assert!(
                step_recurrent_state_only_head_pinned(
                    Arithmetic::PinnedSwift,
                    &inputs,
                    width,
                    &prev,
                    &mut state,
                )
                .is_err()
            );

            let mut bad_val = vec![0.0_f32; width];
            bad_val[width - 1] = f32::NAN;
            let bad_inputs = [StateStepInput {
                key: &k1,
                value: &bad_val,
                decay: 1.0,
                beta: 1.0,
            }];
            assert!(
                step_recurrent_state_only_head_pinned(
                    Arithmetic::PinnedSwift,
                    &bad_inputs,
                    width,
                    &prev,
                    &mut state,
                )
                .is_err()
            );
        }
    }

    fn build_single_recurrent_tiled_session(head_dim: usize, tile_tokens: usize) -> Result<Qwen35> {
        Qwen35::with_adapters_and_kv(
            single_recurrent_layer_fixture(head_dim),
            32,
            MAX_STATE_BYTES,
            MatvecBackend::Auto,
            None,
            super::super::KvCachePolicy::F16,
        )?
        .with_tiled_prefill(tile_tokens, 1 << 20)
    }

    #[test]
    fn recurrent_state_only_tile_matches_sequential_pinned_and_source_zero_ulp() -> Result<()> {
        let count = 8_usize;
        let head_dim = 32_usize;
        let norm_inputs: Vec<Vec<f32>> = (0..count)
            .map(|t| values(head_dim, 0x5555_0001 + u32::try_from(t).unwrap_or(0)))
            .collect();
        for arithmetic in [Arithmetic::Source, Arithmetic::PinnedSwift] {
            let mut single = build_single_recurrent_tiled_session(head_dim, 8)?;
            let mut tiled = build_single_recurrent_tiled_session(head_dim, count)?;

            let Attention::Recurrent(ref single_layer) = single.layers[0].attention else {
                return Err(invalid("expected recurrent layer"));
            };
            let Attention::Recurrent(ref tiled_layer) = tiled.layers[0].attention else {
                return Err(invalid("expected recurrent layer"));
            };
            let mut single_tile = single
                .tiled_prefill
                .take()
                .ok_or_else(|| invalid("missing single tile"))?;
            let mut multi_tile = tiled
                .tiled_prefill
                .take()
                .ok_or_else(|| invalid("missing multi tile"))?;
            let mut adapter_scratch = AdapterScratch::default();
            let mut proj_scratch = MatvecScratch::default();

            for input in &norm_inputs {
                single_tile.normalized[0].resize(head_dim, 0.0);
                single_tile.normalized[0].copy_from_slice(input);
                recurrent_state_only_tile(
                    single_layer,
                    &mut single.state[0],
                    &mut single.workspace.deltas[0],
                    &single.geometry,
                    arithmetic,
                    &single.transforms,
                    ActiveAdapters::default(),
                    &mut adapter_scratch,
                    &mut proj_scratch,
                    TileReuse::default(),
                    1,
                    &|| Ok(()),
                    &mut single_tile,
                    None,
                )?;
            }

            for (t, input) in norm_inputs.iter().enumerate() {
                multi_tile.normalized[t].resize(head_dim, 0.0);
                multi_tile.normalized[t].copy_from_slice(input);
            }
            recurrent_state_only_tile(
                tiled_layer,
                &mut tiled.state[0],
                &mut tiled.workspace.deltas[0],
                &tiled.geometry,
                arithmetic,
                &tiled.transforms,
                ActiveAdapters::default(),
                &mut adapter_scratch,
                &mut proj_scratch,
                TileReuse::default(),
                count,
                &|| Ok(()),
                &mut multi_tile,
                None,
            )?;

            match (&single.state[0], &tiled.state[0]) {
                (
                    LayerState::Recurrent {
                        convolution: sc,
                        matrix: sm,
                    },
                    LayerState::Recurrent {
                        convolution: tc,
                        matrix: tm,
                    },
                ) => {
                    assert_eq!(
                        tc.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        sc.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                    assert_eq!(
                        tm.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        sm.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                }
                _ => return Err(invalid("expected recurrent layer state")),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod attention_2d_tests {
    use std::sync::Arc;

    use super::super::{
        ActiveAdapters, AdapterScratch, FullAttention, Geometry, KvCache, KvCachePolicy, Transforms,
    };
    use super::{Arithmetic, RowParallelism, TiledPrefill, full_attention_tile};
    use crate::Result;
    use crate::kernels::MatvecBackend;
    use crate::model::Tensor;

    fn sample(index: usize) -> f32 {
        [0.0, -0.0, 0.33337, -0.251, 0.501, -1.001, 2.001][index % 7]
    }

    fn test_geometry() -> Geometry {
        Geometry {
            embedding: 256,
            feed_forward: 256,
            heads: 24,
            kv_heads: 4,
            head: 256,
            rope: 64,
            rope_sections: [11, 11, 10, 0],
            rope_base: 10_000_000.0,
            epsilon: 1e-6,
            ssm_head: 128,
            ssm_keys: 16,
            ssm_values: 48,
            convolution: 4,
        }
    }

    fn run_attention_tile_once(
        count: usize,
        workers: usize,
        arithmetic: Arithmetic,
    ) -> Result<Vec<Vec<f32>>> {
        let geometry = test_geometry();
        let unused = Tensor {
            name: String::from("unused-2d-attn-test"),
            dimensions: vec![256, 256],
            kind: 0,
            data: Arc::<[u8]>::from([]).into(),
        };
        let weights = FullAttention {
            query: unused.clone(),
            key: unused.clone(),
            value: unused.clone(),
            output: unused,
            query_norm: vec![1.0; geometry.head],
            key_norm: vec![1.0; geometry.head],
        };
        let base = 63;
        let mut keys =
            KvCache::with_policy(geometry.kv_width(), 128, KvCachePolicy::F16, true, true)?;
        let mut values =
            KvCache::with_policy(geometry.kv_width(), 128, KvCachePolicy::F16, false, true)?;
        for position in 0..base {
            let key: Vec<f32> = (0..geometry.kv_width())
                .map(|i| sample(i + position))
                .collect();
            let value: Vec<f32> = (0..geometry.kv_width())
                .map(|i| sample(i + 3 * position + 1))
                .collect();
            keys.prepare_append(&key)?;
            values.prepare_append(&value)?;
            keys.append(&key);
            values.append(&value);
        }
        let parallelism = RowParallelism::new(workers)?;
        let mut tile = TiledPrefill::new(
            count,
            geometry.query_gate_width(),
            0,
            MatvecBackend::Scalar,
            parallelism.clone(),
        )?;
        for token in 0..count {
            tile.mixed[token] = (0..geometry.query_gate_width())
                .map(|i| sample(i + 2 * token + 2))
                .collect();
            tile.recurrent_gate[token] = (0..geometry.kv_width())
                .map(|i| sample(i + 3 * token + 3))
                .collect();
            tile.beta[token] = (0..geometry.kv_width())
                .map(|i| sample(i + 4 * token + 4))
                .collect();
        }
        let mut projection = super::super::ProjectionScratch::default();
        projection.matvec.set_row_parallelism(parallelism);
        let mut adapter_scratch = AdapterScratch::default();
        let mut attention_scratch = super::super::AttentionScratch::default();
        let mut frequencies_ready = false;
        full_attention_tile(
            &weights,
            &geometry,
            arithmetic,
            &Transforms::default(),
            base,
            base,
            count,
            &mut frequencies_ready,
            &mut keys,
            &mut values,
            &mut projection,
            ActiveAdapters::default(),
            &mut adapter_scratch,
            &mut attention_scratch,
            &mut tile,
            &|| Ok(()),
        )?;
        Ok(tile.attended[..count].to_vec())
    }

    #[test]
    fn full_attention_tile_2d_short_blocks_match_single_worker_zero_ulp() -> Result<()> {
        #[cfg(feature = "parallel-rows")]
        let worker_counts = [1, 2, 4, 12, 18];
        #[cfg(not(feature = "parallel-rows"))]
        let worker_counts = [1];
        for count in [1, 2, 3, 4, 5, 8] {
            for arithmetic in [Arithmetic::PinnedSwift, Arithmetic::Source] {
                let expected = run_attention_tile_once(count, 1, arithmetic)?;
                for workers in worker_counts {
                    let actual = run_attention_tile_once(count, workers, arithmetic)?;
                    assert_eq!(
                        actual
                            .iter()
                            .flatten()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        expected
                            .iter()
                            .flatten()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        "2D full_attention_tile mismatch count={count}, workers={workers}"
                    );
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod fused_last_token_tests {
    use std::cell::Cell;
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use bonsai_gguf::MetadataValue;

    use super::super::{Arithmetic, Geometry, KvCachePolicy, MAX_STATE_BYTES, Qwen35, Transforms};
    use super::add_normalize_for_tile_into;
    use crate::kernels::{MatvecBackend, Pq2ProjectionExecutor, Q8KBlock, RowParallelism};
    use crate::model::{Model, Tensor};
    use crate::{Decoder, MAX_PREFILL_BLOCK_TOKENS, Result, invalid};

    fn f32_tensor(name: &str, dims: &[usize], values: &[f32]) -> Tensor {
        Tensor {
            name: name.to_owned(),
            dimensions: dims.to_vec(),
            kind: 0,
            data: Arc::<[u8]>::from(
                values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .into(),
        }
    }

    fn pat(len: usize, seed: usize) -> Vec<f32> {
        (0..len)
            .map(|i| [0.0625_f32, -0.03125, 0.015_625, -0.0625, 0.03125][(i + seed) % 5])
            .collect()
    }

    fn insert_layer_tensors(
        tensors: &mut BTreeMap<String, Tensor>,
        layer: usize,
        is_rec: bool,
        emb: usize,
        ffn: usize,
        ssm_head: usize,
        full_head: usize,
    ) {
        let ones = |len: usize| vec![1.0_f32; len];
        let p = format!("blk.{layer}.");
        for (suffix, dims, vals) in [
            ("attn_norm.weight", vec![emb], ones(emb)),
            ("post_attention_norm.weight", vec![emb], ones(emb)),
            ("ffn_up.weight", vec![emb, ffn], pat(emb * ffn, 13 + layer)),
            (
                "ffn_gate.weight",
                vec![emb, ffn],
                pat(emb * ffn, 15 + layer),
            ),
            (
                "ffn_down.weight",
                vec![ffn, emb],
                pat(ffn * emb, 17 + layer),
            ),
        ] {
            let name = format!("{p}{suffix}");
            tensors.insert(name.clone(), f32_tensor(&name, &dims, &vals));
        }
        let attn_specs: Vec<(&str, Vec<usize>, Vec<f32>)> = if is_rec {
            vec![
                (
                    "attn_qkv.weight",
                    vec![emb, 3 * ssm_head],
                    pat(emb * 3 * ssm_head, 3 + layer),
                ),
                (
                    "attn_gate.weight",
                    vec![emb, ssm_head],
                    pat(emb * ssm_head, 4 + layer),
                ),
                ("ssm_alpha.weight", vec![emb, 1], pat(emb, 5)),
                ("ssm_beta.weight", vec![emb, 1], pat(emb, 6)),
                (
                    "ssm_conv1d.weight",
                    vec![2, 3 * ssm_head],
                    pat(2 * 3 * ssm_head, 7),
                ),
                ("ssm_dt.bias", vec![1], vec![0.125]),
                ("ssm_a", vec![1], vec![-0.25]),
                ("ssm_norm.weight", vec![ssm_head], ones(ssm_head)),
                (
                    "ssm_out.weight",
                    vec![ssm_head, emb],
                    pat(ssm_head * emb, 8),
                ),
            ]
        } else {
            vec![
                (
                    "attn_q.weight",
                    vec![emb, 2 * full_head],
                    pat(emb * 2 * full_head, 9 + layer),
                ),
                (
                    "attn_k.weight",
                    vec![emb, full_head],
                    pat(emb * full_head, 10 + layer),
                ),
                (
                    "attn_v.weight",
                    vec![emb, full_head],
                    pat(emb * full_head, 11 + layer),
                ),
                (
                    "attn_output.weight",
                    vec![full_head, emb],
                    pat(full_head * emb, 12 + layer),
                ),
                ("attn_q_norm.weight", vec![full_head], ones(full_head)),
                ("attn_k_norm.weight", vec![full_head], ones(full_head)),
            ]
        };
        for (suffix, dims, vals) in attn_specs {
            let name = format!("{p}{suffix}");
            tensors.insert(name.clone(), f32_tensor(&name, &dims, &vals));
        }
    }

    fn hybrid_two_layer_fixture(last_recurrent: bool) -> Model {
        let emb = 64_usize;
        let ffn = 128_usize;
        let ssm_head = 64_usize;
        let full_head = 256_usize;
        let vocab = 6_usize;
        let ones = |len: usize| vec![1.0_f32; len];
        let recurrent_flags = if last_recurrent {
            vec![false, true]
        } else {
            vec![true, false]
        };
        let metadata = BTreeMap::from([
            ("general.architecture".into(), MetadataValue::from("qwen35")),
            ("qwen35.block_count".into(), MetadataValue::from(2_u32)),
            ("qwen35.context_length".into(), MetadataValue::from(64_u32)),
            (
                "qwen35.embedding_length".into(),
                MetadataValue::from(64_u32),
            ),
            (
                "qwen35.feed_forward_length".into(),
                MetadataValue::from(128_u32),
            ),
            (
                "qwen35.attention.head_count".into(),
                MetadataValue::from(1_u32),
            ),
            (
                "qwen35.attention.head_count_kv".into(),
                MetadataValue::from(1_u32),
            ),
            (
                "qwen35.attention.key_length".into(),
                MetadataValue::from(256_u32),
            ),
            (
                "qwen35.attention.value_length".into(),
                MetadataValue::from(256_u32),
            ),
            (
                "qwen35.rope.dimension_count".into(),
                MetadataValue::from(64_u32),
            ),
            (
                "qwen35.rope.freq_base".into(),
                MetadataValue::from(10_000.0_f32),
            ),
            (
                "qwen35.attention.layer_norm_rms_epsilon".into(),
                MetadataValue::from(1e-6_f32),
            ),
            (
                "qwen35.full_attention_interval".into(),
                MetadataValue::from(2_u32),
            ),
            ("qwen35.ssm.state_size".into(), MetadataValue::from(64_u32)),
            ("qwen35.ssm.group_count".into(), MetadataValue::from(1_u32)),
            (
                "qwen35.ssm.time_step_rank".into(),
                MetadataValue::from(1_u32),
            ),
            ("qwen35.ssm.inner_size".into(), MetadataValue::from(64_u32)),
            ("qwen35.ssm.conv_kernel".into(), MetadataValue::from(2_u32)),
            (
                "qwen35.rope.dimension_sections".into(),
                MetadataValue::from(vec![16_u32, 16, 0, 0]),
            ),
            (
                "qwen35.attention.recurrent_layers".into(),
                MetadataValue::from(recurrent_flags.clone()),
            ),
        ]);
        let mut tensors = BTreeMap::from([
            (
                "token_embd.weight".into(),
                f32_tensor("token_embd.weight", &[emb, vocab], &pat(emb * vocab, 1)),
            ),
            (
                "output.weight".into(),
                f32_tensor("output.weight", &[emb, vocab], &pat(emb * vocab, 2)),
            ),
            (
                "output_norm.weight".into(),
                f32_tensor("output_norm.weight", &[emb], &ones(emb)),
            ),
        ]);
        for (layer, &is_rec) in recurrent_flags.iter().enumerate() {
            insert_layer_tensors(&mut tensors, layer, is_rec, emb, ffn, ssm_head, full_head);
        }
        Model {
            metadata,
            tensors,
            identity: format!("fused-last-token-fixture-{last_recurrent}"),
        }
    }

    fn state_bits(graph: &Qwen35) -> Result<Vec<u32>> {
        let mut bits = Vec::new();
        graph.visit_state(&mut |_, _, values| {
            bits.extend(values.iter().map(|v| v.to_bits()));
            Ok(())
        })?;
        Ok(bits)
    }

    fn state_values(graph: &Qwen35) -> Result<Vec<f32>> {
        let mut out = Vec::new();
        graph.visit_state(&mut |_, _, values| {
            out.extend_from_slice(values);
            Ok(())
        })?;
        Ok(out)
    }

    fn build_tiled(
        last_recurrent: bool,
        policy: KvCachePolicy,
        tile_tokens: usize,
        chunkwise: bool,
        narrow_last: bool,
    ) -> Result<Qwen35> {
        let mut graph = Qwen35::with_adapters_and_kv(
            hybrid_two_layer_fixture(last_recurrent),
            32,
            MAX_STATE_BYTES,
            MatvecBackend::Scalar,
            None,
            policy,
        )?;
        graph.set_last_layer_prefill_narrowing(narrow_last);
        let scratch = graph.tiled_prefill_scratch_bytes(tile_tokens)?;
        let graph = graph.with_tiled_prefill(tile_tokens, scratch)?;
        if chunkwise {
            graph.with_chunkwise_gdn_prefill()
        } else {
            Ok(graph)
        }
    }

    #[test]
    fn fused_last_token_matches_ingest_plus_evaluate_across_kv_policies_and_last_layer_kinds()
    -> Result<()> {
        for last_recurrent in [false, true] {
            for policy in [
                KvCachePolicy::F16,
                KvCachePolicy::Rotated4,
                KvCachePolicy::Rotated4FittedValues,
                KvCachePolicy::Rotated4Keys3Values,
            ] {
                for narrow_last in [false, true] {
                    for length in [1_usize, 2, 5, 8, 9, 11] {
                        let prompt: Vec<u32> = (0..length)
                            .map(|i| u32::try_from(i % 6).unwrap_or(0))
                            .collect();
                        let mut expected =
                            build_tiled(last_recurrent, policy, 8, false, narrow_last)?;
                        let mut expected_logits = Vec::new();
                        let Some((&last, prefix)) = prompt.split_last() else {
                            return Err(invalid("empty test prompt"));
                        };
                        if !prefix.is_empty() {
                            expected.ingest_block(prefix, &|| Ok(()))?;
                        }
                        expected.evaluate_into(last, &mut expected_logits)?;

                        let mut actual =
                            build_tiled(last_recurrent, policy, 8, false, narrow_last)?;
                        let mut actual_logits = Vec::new();
                        actual.ingest_block_and_evaluate_last_into(
                            &prompt,
                            &mut actual_logits,
                            &|| Ok(()),
                        )?;

                        assert_eq!(
                            actual_logits
                                .iter()
                                .map(|v| v.to_bits())
                                .collect::<Vec<_>>(),
                            expected_logits
                                .iter()
                                .map(|v| v.to_bits())
                                .collect::<Vec<_>>(),
                            "logits mismatch last_recurrent={last_recurrent} policy={policy:?} narrow={narrow_last} len={length}"
                        );
                        assert_eq!(state_bits(&actual)?, state_bits(&expected)?);
                        assert_eq!(actual.evaluate(3)?, expected.evaluate(3)?);
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    #[expect(
        clippy::suboptimal_flops,
        reason = "Retain declared separately rounded chunkwise tolerance check."
    )]
    fn fused_last_token_chunkwise_gdn_stays_within_declared_tolerance() -> Result<()> {
        for last_recurrent in [false, true] {
            let prompt: Vec<u32> = (0..11).map(|i| u32::try_from(i % 6).unwrap_or(0)).collect();
            let mut expected = build_tiled(last_recurrent, KvCachePolicy::F16, 8, true, true)?;
            let mut expected_logits = Vec::new();
            let Some((&last, prefix)) = prompt.split_last() else {
                return Err(invalid("empty test prompt"));
            };
            expected.ingest_block(prefix, &|| Ok(()))?;
            expected.evaluate_into(last, &mut expected_logits)?;

            let mut actual = build_tiled(last_recurrent, KvCachePolicy::F16, 8, true, true)?;
            let mut actual_logits = Vec::new();
            actual.ingest_block_and_evaluate_last_into(&prompt, &mut actual_logits, &|| Ok(()))?;
            for (&a, &e) in actual_logits.iter().zip(&expected_logits) {
                let tol = 1e-3 + 1e-3 * e.abs();
                assert!((a - e).abs() <= tol, "|{a} - {e}| > {tol}");
            }
            for (&a, &e) in state_values(&actual)?.iter().zip(&state_values(&expected)?) {
                let tol = 1e-3 + 1e-3 * e.abs();
                assert!((a - e).abs() <= tol, "state |{a} - {e}| > {tol}");
            }
        }
        Ok(())
    }

    #[test]
    fn fused_last_token_preserves_bounds_cancellation_and_rollback_safety() -> Result<()> {
        let mut graph = build_tiled(false, KvCachePolicy::F16, 8, false, true)?;
        let initial = graph.checkpoint()?;
        let initial_state = state_bits(&graph)?;
        let mut output = vec![1.0_f32; 4];

        for bad_block in [
            vec![],
            vec![0; MAX_PREFILL_BLOCK_TOKENS + 1],
            vec![0, 99],
            vec![0; 33],
        ] {
            output.push(2.0);
            assert!(
                graph
                    .ingest_block_and_evaluate_last_into(&bad_block, &mut output, &|| Ok(()))
                    .is_err()
            );
            assert_eq!(output, Vec::<f32>::new());
            assert_eq!(state_bits(&graph)?, initial_state);
        }

        // Mid-tile cancellation marks state inconsistent until restore, and clears output.
        let checks = Cell::new(0_usize);
        output = vec![1.0_f32; 3];
        let cancelled =
            graph.ingest_block_and_evaluate_last_into(&[0, 1, 2, 3], &mut output, &|| {
                checks.set(checks.get() + 1);
                if checks.get() == 8 {
                    Err(invalid("mid-tile cancel"))
                } else {
                    Ok(())
                }
            });
        assert!(cancelled.is_err());
        assert_eq!(output, Vec::<f32>::new());
        assert!(graph.evaluate(0).is_err());
        assert!(graph.checkpoint().is_err());

        graph.restore(&initial)?;
        assert_eq!(
            graph.ingest_block_and_evaluate_last_into(&[0, 1, 2, 3], &mut output, &|| Ok(()))?,
            ()
        );
        assert_eq!(output.len(), 6);

        // Post-evaluation cancellation must still clear output.
        let after_restore = graph.checkpoint()?;
        let total_checks = Cell::new(0_usize);
        graph.ingest_block_and_evaluate_last_into(&[1, 2], &mut output, &|| {
            total_checks.set(total_checks.get() + 1);
            Ok(())
        })?;
        let target_check = total_checks.get();
        graph.restore(&after_restore)?;
        let post_cancel = Cell::new(0_usize);
        assert!(
            graph
                .ingest_block_and_evaluate_last_into(&[1, 2], &mut output, &|| {
                    post_cancel.set(post_cancel.get() + 1);
                    if post_cancel.get() == target_check {
                        Err(invalid("post-eval cancel"))
                    } else {
                        Ok(())
                    }
                })
                .is_err()
        );
        assert_eq!(output, Vec::<f32>::new());
        Ok(())
    }

    struct DecodeGateExecutor {
        supports_tiles: bool,
        calls: Arc<AtomicUsize>,
    }

    impl Pq2ProjectionExecutor for DecodeGateExecutor {
        fn identity(&self) -> &'static str {
            "fused-last-decode-gate-test"
        }

        fn supports(&self, _tensor: &Tensor) -> bool {
            false
        }

        fn supports_f32(&self, tensor: &Tensor) -> bool {
            tensor.kind == 142
        }

        fn supports_tile_tokens(&self) -> bool {
            self.supports_tiles
        }

        fn project(
            &mut self,
            _tensor: &Tensor,
            _activation: &[Q8KBlock],
            _fused: bool,
            _output: &mut Vec<f32>,
        ) -> Result<()> {
            Err(invalid("unexpected Q8_K call"))
        }

        fn project_f32(
            &mut self,
            tensor: &Tensor,
            activation: &[f32],
            pinned: bool,
            output: &mut Vec<f32>,
        ) -> Result<()> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let mut scratch = if pinned {
                crate::kernels::MatvecScratch::with_pinned_backend(MatvecBackend::Scalar)
            } else {
                crate::kernels::MatvecScratch::with_backend(MatvecBackend::Scalar)
            };
            scratch
                .prepare(tensor, activation)?
                .matvec_into(tensor, output)
        }
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "Fixture setup and assertions for fused last-token decode executor."
    )]
    fn fused_last_token_decode_executor_skips_multi_token_tiles_and_runs_last_token() -> Result<()>
    {
        let mut model = hybrid_two_layer_fixture(false);
        model.tensors.clear();
        let pq2_rect = |cols: usize, rows: usize, seed: u8| -> Tensor {
            let mut block = [0_u8; 34];
            block[..2].copy_from_slice(&0x3800_u16.to_le_bytes());
            for (idx, byte) in block[2..].iter_mut().enumerate() {
                *byte = u8::try_from((idx * 13 + usize::from(seed)) % 256).unwrap_or(0);
            }
            Tensor {
                name: String::new(),
                dimensions: vec![cols, rows],
                kind: 142,
                data: Arc::<[u8]>::from(block.repeat((cols / 128) * rows)).into(),
            }
        };
        for (key, value) in [
            ("qwen35.block_count", MetadataValue::from(1_u32)),
            ("qwen35.embedding_length", MetadataValue::from(128_u32)),
            ("qwen35.feed_forward_length", MetadataValue::from(128_u32)),
            ("qwen35.attention.key_length", MetadataValue::from(128_u32)),
            (
                "qwen35.attention.value_length",
                MetadataValue::from(128_u32),
            ),
            (
                "qwen35.rope.dimension_sections",
                MetadataValue::from(vec![32_u32, 0, 0, 0]),
            ),
            (
                "qwen35.attention.recurrent_layers",
                MetadataValue::from(vec![false]),
            ),
        ] {
            model.metadata.insert(key.into(), value);
        }
        let ones128 = vec![1.0_f32; 128];
        for (name, dims, data) in [
            ("token_embd.weight", &[128, 6][..], pat(128 * 6, 1)),
            ("output.weight", &[128, 6][..], pat(128 * 6, 2)),
            ("output_norm.weight", &[128][..], ones128.clone()),
            ("blk.0.attn_norm.weight", &[128][..], ones128.clone()),
            (
                "blk.0.post_attention_norm.weight",
                &[128][..],
                ones128.clone(),
            ),
            ("blk.0.attn_q_norm.weight", &[128][..], ones128.clone()),
            ("blk.0.attn_k_norm.weight", &[128][..], ones128),
        ] {
            model
                .tensors
                .insert(name.into(), f32_tensor(name, dims, &data));
        }
        for (name, cols, rows, seed) in [
            ("blk.0.attn_q.weight", 128, 256, 1_u8),
            ("blk.0.attn_k.weight", 128, 128, 2),
            ("blk.0.attn_v.weight", 128, 128, 3),
            ("blk.0.attn_output.weight", 128, 128, 4),
            ("blk.0.ffn_up.weight", 128, 128, 5),
            ("blk.0.ffn_gate.weight", 128, 128, 6),
            ("blk.0.ffn_down.weight", 128, 128, 7),
        ] {
            let mut tensor = pq2_rect(cols, rows, seed);
            tensor.name = name.to_owned();
            model.tensors.insert(name.to_owned(), tensor);
        }

        let cloned_model = Model {
            metadata: model.metadata.clone(),
            tensors: model.tensors.clone(),
            identity: model.identity.clone(),
        };
        let mut cpu_ref = Qwen35::new(cloned_model, 32)?;
        let cpu_scratch = cpu_ref.tiled_prefill_scratch_bytes(8)?;
        cpu_ref = cpu_ref.with_tiled_prefill(8, cpu_scratch)?;
        let mut expected_logits = Vec::new();
        cpu_ref.ingest_block_and_evaluate_last_into(
            &[0, 1, 2, 3],
            &mut expected_logits,
            &|| Ok(()),
        )?;

        let mut decode_gated = Qwen35::new(model, 32)?;
        let gated_scratch = decode_gated.tiled_prefill_scratch_bytes(8)?;
        decode_gated = decode_gated.with_tiled_prefill(8, gated_scratch)?;
        let calls = Arc::new(AtomicUsize::new(0));
        decode_gated.set_projection_executor(Some(Box::new(DecodeGateExecutor {
            supports_tiles: false,
            calls: Arc::clone(&calls),
        })));
        let mut actual_logits = Vec::new();
        decode_gated.ingest_block_and_evaluate_last_into(
            &[0, 1, 2, 3],
            &mut actual_logits,
            &|| Ok(()),
        )?;
        // Only the 5 single-token projections on token T-1 (query, output, up, gate, down) hit executor.
        assert_eq!(calls.load(Ordering::Relaxed), 5);
        assert_eq!(
            actual_logits
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected_logits
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn add_normalize_for_tile_rejects_nonfinite_delta_without_hadamard_fallback() -> Result<()> {
        let len = 32_usize;
        let block = 32_usize;
        let ones_mat = vec![1.0_f32; len * len];
        let tensor = f32_tensor("blk.0.ffn_gate.weight", &[len, len], &ones_mat);
        let mut weights = BTreeSet::new();
        weights.insert("blk.0.ffn_gate.weight".to_owned());
        let transforms = Transforms {
            block,
            weights,
            inverse_embedding: false,
            grouped_values: false,
            signs: BTreeMap::new(),
            tensor_signs: BTreeMap::new(),
            compacted_ffn: super::super::weight_patch::CompactedFfnRuntime::default(),
            trellis: None,
        };
        let geometry = Geometry {
            embedding: len,
            feed_forward: len,
            heads: 1,
            kv_heads: 1,
            head: len,
            rope: 4,
            rope_sections: [2, 0, 0, 0],
            rope_base: 10_000.0,
            epsilon: 1e-6,
            ssm_head: len,
            ssm_keys: 1,
            ssm_values: 1,
            convolution: 2,
        };
        let norm_weight = vec![1.0_f32; len];
        let parallelism = if cfg!(feature = "parallel-rows") {
            RowParallelism::new(4)?
        } else {
            RowParallelism::default()
        };
        let mut hidden = vec![vec![1.0_f32; len]; 2];
        let mut delta = vec![vec![0.5_f32; len]; 2];
        delta[0][3] = f32::NAN;
        let mut normalized = vec![Vec::new(); 2];
        let mut transformed = vec![Vec::new(); 2];
        for arithmetic in [Arithmetic::Source, Arithmetic::PinnedSwift] {
            let err = add_normalize_for_tile_into(
                &transforms,
                &geometry,
                arithmetic,
                &norm_weight,
                &tensor,
                &mut hidden,
                &delta,
                &mut normalized,
                &mut transformed,
                &parallelism,
            );
            assert!(err.is_err());
        }
        Ok(())
    }

    #[test]
    fn tiled_prefill_fusions_match_decode_with_hadamard_inverse_embedding_and_workers() -> Result<()>
    {
        let mut model = hybrid_two_layer_fixture(false);
        insert_layer_tensors(&mut model.tensors, 2, false, 64, 128, 64, 256);
        insert_layer_tensors(&mut model.tensors, 3, true, 64, 128, 64, 256);
        model
            .metadata
            .insert("qwen35.block_count".into(), MetadataValue::from(4_u32));
        model.metadata.insert(
            "qwen35.attention.recurrent_layers".into(),
            MetadataValue::from(vec![true, false, false, true]),
        );
        for (key, value) in [
            ("prism.hadamard.version", MetadataValue::from(1_u32)),
            ("prism.hadamard.block_size", MetadataValue::from(16_u32)),
            (
                "prism.hadamard.transform",
                MetadataValue::from("normalized-sylvester-walsh-hadamard"),
            ),
            (
                "prism.hadamard.axis",
                MetadataValue::from("input-last-dimension"),
            ),
            (
                "prism.hadamard.weight_names",
                MetadataValue::from(vec![
                    "blk.0.attn_qkv.weight",
                    "blk.0.ffn_gate.weight",
                    "blk.0.ffn_up.weight",
                    "blk.1.attn_q.weight",
                    "blk.1.attn_k.weight",
                    "blk.1.attn_v.weight",
                    "blk.1.ffn_gate.weight",
                    "blk.1.ffn_up.weight",
                    "blk.2.attn_q.weight",
                    "blk.2.attn_k.weight",
                    "blk.2.attn_v.weight",
                    "blk.2.ffn_gate.weight",
                    "blk.2.ffn_up.weight",
                    "blk.3.attn_qkv.weight",
                    "blk.3.ffn_gate.weight",
                    "blk.3.ffn_up.weight",
                    "output.weight",
                ]),
            ),
            (
                "prism.hadamard.inverse_weight_names",
                MetadataValue::from(vec!["token_embd.weight"]),
            ),
            ("prism.hadamard.sign_mode", MetadataValue::from("identity")),
        ] {
            model.metadata.insert(key.into(), value);
        }
        let cloned = Model {
            metadata: model.metadata.clone(),
            tensors: model.tensors.clone(),
            identity: model.identity.clone(),
        };
        #[cfg(feature = "parallel-rows")]
        let workers = 4;
        #[cfg(not(feature = "parallel-rows"))]
        let workers = 1;

        let mut sequential = Qwen35::new(cloned, 32)?;
        sequential.set_row_parallelism(RowParallelism::new(workers)?);
        for &token in &[0_u32, 1, 2, 3, 2, 0, 1, 2, 3] {
            sequential.ingest(token)?;
        }
        let expected_logits = sequential.evaluate(1)?;

        let mut tiled = Qwen35::new(model, 32)?;
        let scratch = tiled.tiled_prefill_scratch_bytes(8)?;
        tiled = tiled.with_tiled_prefill(8, scratch)?;
        tiled.set_row_parallelism(RowParallelism::new(workers)?);
        let mut actual_logits = Vec::new();
        // Two tiles: leading 8-token tile runs narrow_last_layer (parallel state-only conv/matrix);
        // final 2-token tile exercises fused residual add + last-layer evaluation.
        tiled.ingest_block(&[0_u32, 1, 2, 3, 2, 0, 1, 2], &|| Ok(()))?;
        tiled.ingest_block_and_evaluate_last_into(&[3_u32, 1], &mut actual_logits, &|| Ok(()))?;
        assert_eq!(
            actual_logits
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected_logits
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(state_bits(&tiled)?, state_bits(&sequential)?);

        // Verify 64-token reservation mirrors MAX_TILE_TOKENS cleanly.
        let r64 = tiled.tiled_prefill_scratch_bytes(64);
        assert_eq!(r64.is_ok(), 64 <= crate::kernels::prefill::MAX_TILE_TOKENS);
        Ok(())
    }
}
