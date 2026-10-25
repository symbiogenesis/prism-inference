//! Bounded Qwen3.5 text decoder with full attention and Gated Delta Net.
//!
//! The operation ordering follows Prism's `src/models/qwen35.cpp`. KV values
//! are stored as IEEE binary16 bits; convolution and recurrent state
//! remain binary32. This correctness path does not establish device parity.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use bonsai_gguf::{Metadata, MetadataArray, MetadataValue, MetadataValueRef};
use bonsai_ternary::half_to_float;

use crate::kernels::{
    MatvecBackend, MatvecDiagnostics, MatvecScratch, MatvecStatistics, half_bits, rms_norm_into,
    round_f16, row, row_into, silu, softmax,
};
use crate::model::{Model, Tensor};
use crate::{Decoder, Result, invalid};

/// Maximum admitted live KV/recurrent element payload, also the default budget.
///
/// This excludes model storage, scratch, checkpoints, page-table/Arc metadata
/// and allocator overhead.
pub const MAX_STATE_BYTES: usize = 4 * 1024 * 1024 * 1024;
const MAX_WIDTH: usize = 1 << 20;
const MAX_LAYERS: usize = 256;
const KV_PAGE_TOKENS: usize = 64;

/// Completed pages never mutate. Checkpoints share them and independently copy
/// only the bounded tail; appending and restoring never copy a sealed payload.
struct KvCache {
    pages: Vec<Arc<Vec<u16>>>,
    tail: Vec<u16>,
    width: usize,
    page_elements: usize,
    maximum: usize,
    len: usize,
}

impl KvCache {
    fn new(width: usize, context: usize) -> Result<Self> {
        let maximum = checked_product(&[width, context])?;
        if width == 0
            || context == 0
            || checked_product(&[maximum, size_of::<u16>()])? > MAX_STATE_BYTES
        {
            return Err(invalid("invalid paged KV cache bounds"));
        }
        Ok(Self {
            pages: Vec::new(),
            tail: Vec::new(),
            width,
            page_elements: checked_product(&[width, context.min(KV_PAGE_TOKENS)])?,
            maximum,
            len: 0,
        })
    }

    fn tokens(&self) -> impl Iterator<Item = &[u16]> {
        self.pages
            .iter()
            .flat_map(|page| page.chunks_exact(self.width))
            .chain(self.tail.chunks_exact(self.width))
    }

    // Reserve every fallible payload/table growth before any token commits.
    // Sealing moves the tail allocation into an Arc; there is no payload COW
    // or second page-sized allocation, even while snapshots retain old pages.
    // The standard Arc constructor still allocates its fixed-size header.
    fn reserve_append(&mut self) -> Result<()> {
        let next = checked_sum(self.len, self.width)?;
        if next > self.maximum {
            return Err(invalid("KV cache append exceeds context bound"));
        }
        let required = checked_sum(self.tail.len(), self.width)?;
        let remaining = self.maximum - (self.len - self.tail.len());
        let limit = self.page_elements.min(remaining);
        if required > self.tail.capacity() {
            let target = self
                .tail
                .capacity()
                .saturating_mul(2)
                .min(limit)
                .max(required);
            self.tail.try_reserve_exact(target - self.tail.len())?;
        }
        if required == self.page_elements || next == self.maximum {
            self.pages.try_reserve_exact(1)?;
        }
        Ok(())
    }

    fn append(&mut self, values: &[f32]) {
        debug_assert_eq!(values.len(), self.width);
        self.tail.extend(values.iter().copied().map(half_bits));
        self.len += self.width;
        if self.tail.len() == self.page_elements || self.len == self.maximum {
            self.pages.push(Arc::new(std::mem::take(&mut self.tail)));
        }
    }

    fn copy(&self) -> Result<Self> {
        let mut pages = capacity(self.pages.len())?;
        pages.extend(self.pages.iter().map(Arc::clone));
        Ok(Self {
            pages,
            tail: copy_values(&self.tail)?,
            width: self.width,
            page_elements: self.page_elements,
            maximum: self.maximum,
            len: self.len,
        })
    }

    fn clear(&mut self) {
        self.pages.clear();
        self.tail.clear();
        self.len = 0;
    }

    fn bytes(&self) -> usize {
        self.pages.capacity() * size_of::<Arc<Vec<u16>>>()
            + self.tail.capacity() * size_of::<u16>()
            + self
                .pages
                .iter()
                .map(|page| {
                    page.capacity() * size_of::<u16>()
                        + size_of::<Vec<u16>>()
                        + 2 * size_of::<usize>()
                })
                .sum::<usize>()
    }
}

#[derive(Clone)]
struct Geometry {
    embedding: usize,
    feed_forward: usize,
    heads: usize,
    kv_heads: usize,
    head: usize,
    rope: usize,
    rope_sections: [usize; 4],
    rope_base: f32,
    epsilon: f32,
    ssm_head: usize,
    ssm_keys: usize,
    ssm_values: usize,
    convolution: usize,
}

impl Geometry {
    const fn key_width(&self) -> usize {
        self.ssm_head * self.ssm_keys
    }

    const fn value_width(&self) -> usize {
        self.ssm_head * self.ssm_values
    }

    const fn convolution_width(&self) -> usize {
        2 * self.key_width() + self.value_width()
    }
}

struct FullAttention {
    query: Tensor,
    key: Tensor,
    value: Tensor,
    output: Tensor,
    query_norm: Vec<f32>,
    key_norm: Vec<f32>,
}

struct RecurrentAttention {
    qkv: Tensor,
    gate: Tensor,
    beta: Tensor,
    alpha: Tensor,
    convolution: Vec<f32>,
    time_bias: Vec<f32>,
    decay: Vec<f32>,
    norm: Vec<f32>,
    output: Tensor,
}

enum Attention {
    Full(FullAttention),
    Recurrent(RecurrentAttention),
}

struct Layer {
    norm: Vec<f32>,
    post_norm: Vec<f32>,
    attention: Attention,
    up: Tensor,
    gate: Tensor,
    down: Tensor,
}

enum LayerState {
    Full {
        keys: KvCache,
        values: KvCache,
    },
    Recurrent {
        convolution: Vec<f32>,
        matrix: Vec<f32>,
    },
}

impl LayerState {
    fn bytes(&self) -> usize {
        match self {
            Self::Full { keys, values } => keys.bytes() + values.bytes(),
            Self::Recurrent {
                convolution,
                matrix,
            } => (convolution.capacity() + matrix.capacity()) * size_of::<f32>(),
        }
    }
}

enum LayerDelta {
    Full {
        // Keep the current token in reusable binary32 scratch while the graph
        // runs; only a successful evaluation appends its rounded binary16 bits.
        key: Vec<f32>,
        value: Vec<f32>,
    },
    Recurrent {
        convolution: Vec<f32>,
        matrix: Vec<f32>,
    },
}

#[derive(Default)]
struct ProjectionScratch {
    transformed: Vec<f32>,
    matvec: MatvecScratch,
    #[cfg(test)]
    transforms: usize,
}

#[derive(Default)]
struct AttentionScratch {
    joint_query: Vec<f32>,
    query: Vec<f32>,
    gate: Vec<f32>,
    key_projection: Vec<f32>,
    normalized: Vec<f32>,
    attended: Vec<f32>,
    scores: Vec<f32>,
    frequencies: Vec<(f32, f32)>,
    qkv: Vec<f32>,
    beta: Vec<f32>,
    alpha: Vec<f32>,
    mixed: Vec<f32>,
}

#[derive(Default)]
struct Workspace {
    hidden: Vec<f32>,
    normalized: Vec<f32>,
    up: Vec<f32>,
    gate: Vec<f32>,
    attention: Vec<f32>,
    projection: ProjectionScratch,
    attention_scratch: AttentionScratch,
    deltas: Vec<LayerDelta>,
}

struct Evaluation<'a> {
    geometry: &'a Geometry,
    transforms: &'a Transforms,
    position: usize,
    frequencies_ready: bool,
    projection: &'a mut ProjectionScratch,
}

#[derive(Default)]
struct Transforms {
    block: usize,
    weights: BTreeSet<String>,
    inverse_embedding: bool,
    grouped_values: bool,
    signs: BTreeMap<usize, Vec<f32>>,
}

impl Transforms {
    #[allow(
        clippy::too_many_lines,
        reason = "Keep validation and graph operation order auditable together."
    )]
    fn read(model: &Model, geometry: &Geometry) -> Result<Self> {
        let metadata = &model.metadata;
        let Some(version) = metadata.get("prism.hadamard.version") else {
            if metadata
                .keys()
                .any(|key| key.starts_with("prism.hadamard."))
            {
                return Err(invalid("Hadamard metadata requires prism.hadamard.version"));
            }
            return Ok(Self::default());
        };
        let version = version
            .as_u64()
            .ok_or_else(|| invalid("invalid Hadamard version"))?;
        let tied = optional_bool(metadata, "prism.hadamard.tied_output", false)?;
        if !matches!(version, 1 | 2) || (version == 2) != tied {
            return Err(invalid(
                "Hadamard version 2 requires tied_output; version 1 forbids it",
            ));
        }
        if tied && model.tensors.contains_key("output.weight") {
            return Err(invalid(
                "Hadamard tied output requires output.weight to be absent",
            ));
        }
        let block = integer(metadata, "prism.hadamard.block_size")?;
        if !block.is_power_of_two() || block > 65_536 {
            return Err(invalid(
                "Hadamard block size must be a bounded power of two",
            ));
        }
        if string(metadata, "prism.hadamard.transform")? != "normalized-sylvester-walsh-hadamard"
            || string(metadata, "prism.hadamard.axis")? != "input-last-dimension"
        {
            return Err(invalid("unsupported Hadamard transform or axis"));
        }
        let mut weights = BTreeSet::new();
        for name in array(metadata, "prism.hadamard.weight_names")?.iter() {
            let name = name
                .as_str()
                .ok_or_else(|| invalid("Hadamard weight names must be strings"))?;
            if !foldable(name) || !weights.insert(name.to_owned()) {
                return Err(invalid(format!(
                    "unsupported or duplicate Hadamard weight: {name}"
                )));
            }
        }
        if weights.is_empty() {
            return Err(invalid("Hadamard weight_names must not be empty"));
        }
        let mut inverse_embedding = false;
        if metadata.contains_key("prism.hadamard.inverse_weight_names") {
            for name in array(metadata, "prism.hadamard.inverse_weight_names")?.iter() {
                if name.as_str() != Some("token_embd.weight") || inverse_embedding {
                    return Err(invalid(
                        "only one token embedding inverse transform is supported",
                    ));
                }
                inverse_embedding = true;
            }
        }
        if tied {
            if !inverse_embedding {
                return Err(invalid(
                    "Hadamard tied output requires inverse token embedding transform",
                ));
            }
            weights.insert("token_embd.weight".to_owned());
        } else if inverse_embedding && !model.tensors.contains_key("output.weight") {
            return Err(invalid("latent tied output requires Hadamard version 2"));
        }
        let mut signs = BTreeMap::new();
        match string(metadata, "prism.hadamard.sign_mode")? {
            "identity" => {}
            "explicit" => {
                let widths = array(metadata, "prism.hadamard.sign_widths")?;
                let values = array(metadata, "prism.hadamard.sign_values")?;
                if widths.is_empty() {
                    return Err(invalid("explicit Hadamard sign widths must not be empty"));
                }
                let mut offset = 0_usize;
                for width in widths.iter() {
                    let width = metadata_usize(width, "Hadamard sign width")?;
                    let end = offset
                        .checked_add(width)
                        .ok_or_else(|| invalid("sign length overflow"))?;
                    if width == 0 || width > MAX_WIDTH || width % block != 0 || end > values.len() {
                        return Err(invalid("invalid Hadamard sign width"));
                    }
                    let vector = (offset..end)
                        .map(
                            |index| match values.get(index).and_then(MetadataValueRef::as_i64) {
                                Some(-1) => Ok(-1.0),
                                Some(1) => Ok(1.0),
                                _ => Err(invalid("Hadamard signs must be +1 or -1")),
                            },
                        )
                        .collect::<Result<Vec<_>>>()?;
                    if signs.insert(width, vector).is_some() {
                        return Err(invalid("duplicate Hadamard sign width"));
                    }
                    offset = end;
                }
                if offset != values.len() {
                    return Err(invalid("Hadamard sign_values length mismatch"));
                }
            }
            _ => return Err(invalid("unsupported Hadamard sign mode")),
        }
        for name in weights
            .iter()
            .map(String::as_str)
            .chain(inverse_embedding.then_some("token_embd.weight"))
        {
            let tensor = model
                .tensors
                .get(name)
                .ok_or_else(|| invalid(format!("Hadamard tensor absent: {name}")))?;
            let width = tensor
                .dimensions
                .first()
                .copied()
                .ok_or_else(|| invalid("Hadamard tensor has no dimensions"))?;
            if width % block != 0 || (!signs.is_empty() && !signs.contains_key(&width)) {
                return Err(invalid(format!(
                    "Hadamard width or signs do not match {name}"
                )));
            }
        }
        let grouped_values = optional_bool(metadata, "prism.hadamard.gdn_v_grouped", false)?;
        if grouped_values && !geometry.ssm_values.is_multiple_of(geometry.ssm_keys) {
            return Err(invalid(
                "grouped GDN transform has incompatible head geometry",
            ));
        }
        Ok(Self {
            block,
            weights,
            inverse_embedding,
            grouped_values,
            signs,
        })
    }

    fn project_into(
        &self,
        weight: &Tensor,
        input: &[f32],
        geometry: &Geometry,
        output: &mut Vec<f32>,
        scratch: &mut ProjectionScratch,
    ) -> Result<()> {
        self.project_group_into([weight], input, geometry, [output], scratch)
    }

    fn basis(&self, weight: &Tensor) -> (bool, bool) {
        let rotated = self.weights.contains(&weight.name);
        (
            rotated,
            rotated && self.grouped_values && weight.name.ends_with(".ssm_out.weight"),
        )
    }

    // The fixed-size group borrows one exact input slice for its whole scope.
    // Nothing is reused across calls, tokens, layers or failed evaluations.
    fn project_group_into<const N: usize>(
        &self,
        weights: [&Tensor; N],
        input: &[f32],
        geometry: &Geometry,
        mut outputs: [&mut Vec<f32>; N],
        scratch: &mut ProjectionScratch,
    ) -> Result<()> {
        for output in &mut outputs {
            output.clear();
        }
        let result = (|| {
            if N > 4 {
                return Err(invalid("projection group exceeds four siblings"));
            }
            let mut completed = [false; N];
            for index in 0..N {
                if completed[index] {
                    continue;
                }
                let basis = self.basis(weights[index]);
                let activation = if basis.0 {
                    self.transform_into(weights[index], input, geometry, &mut scratch.transformed)?;
                    #[cfg(test)]
                    {
                        scratch.transforms += 1;
                    }
                    scratch.transformed.as_slice()
                } else {
                    input
                };
                let mut prepared = scratch.matvec.prepare(weights[index], activation)?;
                for (candidate, (weight, output)) in
                    weights.iter().zip(outputs.iter_mut()).enumerate()
                {
                    if !completed[candidate]
                        && self.basis(weight) == basis
                        && prepared.is_compatible(weight)?
                    {
                        prepared.matvec_into(weight, output)?;
                        completed[candidate] = true;
                    }
                }
            }
            Ok(())
        })();
        if result.is_err() {
            for output in &mut outputs {
                output.clear();
            }
        }
        result
    }

    fn transform_into(
        &self,
        weight: &Tensor,
        input: &[f32],
        geometry: &Geometry,
        transformed: &mut Vec<f32>,
    ) -> Result<()> {
        copy_into(input, transformed)?;
        if self.grouped_values && weight.name.ends_with(".ssm_out.weight") {
            let repetitions = geometry.ssm_values / geometry.ssm_keys;
            for key in 0..geometry.ssm_keys {
                for repeat in 0..repetitions {
                    let source = (repeat * geometry.ssm_keys + key) * geometry.ssm_head;
                    let target = (key * repetitions + repeat) * geometry.ssm_head;
                    transformed[target..target + geometry.ssm_head]
                        .copy_from_slice(&input[source..source + geometry.ssm_head]);
                }
            }
        }
        self.sign(transformed);
        self.rotate(transformed)?;
        Ok(())
    }

    fn sign(&self, input: &mut [f32]) {
        if let Some(signs) = self.signs.get(&input.len()) {
            for (value, sign) in input.iter_mut().zip(signs) {
                *value *= sign;
            }
        }
    }

    fn rotate(&self, input: &mut [f32]) -> Result<()> {
        for block in input.chunks_exact_mut(self.block) {
            crate::kernels::hadamard(block)?;
        }
        Ok(())
    }
}

/// Exact conversation state with shared immutable KV pages and an owned tail.
/// Convolution and recurrent payloads are copied exactly, without F16 rounding.
pub struct Qwen35Checkpoint {
    provenance: Arc<()>,
    identity: String,
    context_limit: usize,
    position: usize,
    layers: Vec<LayerState>,
}

/// One logical state component observed for full-model numerical qualification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StateComponent {
    /// One cached key token, promoted exactly from stored binary16 bits.
    Key(usize),
    /// One cached value token, promoted exactly from stored binary16 bits.
    Value(usize),
    /// Channel-major convolution history, oldest value first per channel.
    Convolution,
    /// GDN state in value-head, value-row, contiguous-key-column order.
    Recurrent,
}

/// Native decoder for the supported Qwen3.5 dense text graph.
pub struct Qwen35 {
    provenance: Arc<()>,
    identity: String,
    geometry: Geometry,
    embedding: Tensor,
    output: Tensor,
    output_norm: Vec<f32>,
    transforms: Transforms,
    layers: Vec<Layer>,
    state: Vec<LayerState>,
    workspace: Workspace,
    context_limit: usize,
    position: usize,
    vocabulary: usize,
}

impl Qwen35 {
    /// Borrows each logical state component for numerical qualification.
    ///
    /// The callback receives the block index and component. Recurrent and
    /// convolution slices borrow committed state directly. Each cached token
    /// uses one reusable, fallibly allocated row to promote FP16 bits to F32.
    /// No observation can mutate state, and callback errors leave state intact.
    /// This supplies no state-import or portable checkpoint contract.
    ///
    /// # Errors
    /// Returns allocation or observer errors without modifying decoder state.
    pub fn visit_state(
        &self,
        observer: &mut impl FnMut(usize, StateComponent, &[f32]) -> Result<()>,
    ) -> Result<()> {
        let mut row = Vec::new();
        for (index, state) in self.state.iter().enumerate() {
            match state {
                LayerState::Full { keys, values } => {
                    for (cache, key) in [(keys, true), (values, false)] {
                        for (position, values) in cache.tokens().enumerate() {
                            row.clear();
                            row.try_reserve_exact(values.len())?;
                            row.extend(values.iter().copied().map(half_to_float));
                            let component = if key {
                                StateComponent::Key(position)
                            } else {
                                StateComponent::Value(position)
                            };
                            observer(index, component, &row)?;
                        }
                    }
                }
                LayerState::Recurrent {
                    convolution,
                    matrix,
                } => {
                    observer(index, StateComponent::Convolution, convolution)?;
                    observer(index, StateComponent::Recurrent, matrix)?;
                }
            }
        }
        Ok(())
    }

    /// Validates model contracts and cache bounds before allocating decoder state.
    ///
    /// Uses [`MatvecBackend::Auto`] with per-operation scalar fallback.
    ///
    /// # Errors
    /// Returns an error for malformed tensors, unsupported graph variants, or
    /// a context/state allocation beyond the native runtime's explicit bounds.
    pub fn new(model: Model, context_limit: usize) -> Result<Self> {
        Self::with_backend(model, context_limit, MatvecBackend::Auto)
    }

    /// Validates and creates a decoder with an explicit packed-matrix kernel request.
    ///
    /// Selection applies to every projection, including grouped siblings and the
    /// output head. Unsupported builds, targets and tensor formats fall back to
    /// scalar for each operation, matching [`MatvecScratch::with_backend`]. Other
    /// graph operations retain their reference arithmetic. The request does not
    /// establish whole-model SIMD parity or performance qualification.
    ///
    /// # Errors
    /// Returns the same model validation and allocation errors as [`Self::new`].
    pub fn with_backend(
        model: Model,
        context_limit: usize,
        backend: MatvecBackend,
    ) -> Result<Self> {
        Self::with_state_budget(model, context_limit, MAX_STATE_BYTES, backend)
    }

    /// Creates a decoder within an explicit maximum live-state payload budget.
    ///
    /// Admission charges full-context FP16 K/V plus F32 convolution/recurrent
    /// elements before allocating decoder state. KV allocation remains lazy;
    /// completed pages hold up to 64 tokens and never copy their payload when
    /// sealed or restored. Checkpoints share those immutable pages and copy the
    /// bounded tail plus exact F32 recurrent state. Snapshot admission charges
    /// each referenced page in full, conservatively overcounting shared pages.
    /// The live-state budget excludes model storage, graph scratch, checkpoints,
    /// page-table/Arc metadata and allocator overhead; it does not bound process
    /// memory. Kernel selection follows [`Self::with_backend`].
    ///
    /// # Errors
    /// Rejects zero budgets, budgets above [`MAX_STATE_BYTES`], a graph whose
    /// maximum live-state element payload exceeds the budget, and the same
    /// malformed model/allocation errors as [`Self::new`].
    #[allow(
        clippy::too_many_lines,
        reason = "Keep validation and graph operation order auditable together."
    )]
    pub fn with_state_budget(
        model: Model,
        context_limit: usize,
        state_budget_bytes: usize,
        backend: MatvecBackend,
    ) -> Result<Self> {
        if state_budget_bytes == 0 || state_budget_bytes > MAX_STATE_BYTES {
            return Err(invalid("invalid native live-state payload budget"));
        }
        let metadata = &model.metadata;
        if string(metadata, "general.architecture")? != "qwen35" {
            return Err(invalid(
                "the native graph supports general.architecture=qwen35 only",
            ));
        }
        reject_unsupported(metadata)?;
        let block_count = integer(metadata, "qwen35.block_count")?;
        let mtp = optional_integer(metadata, "qwen35.nextn_predict_layers", 0)?;
        if block_count == 0 || block_count > MAX_LAYERS || mtp >= block_count {
            return Err(invalid("invalid Qwen35 block or MTP count"));
        }
        let layer_count = block_count - mtp;
        let train_context = integer(metadata, "qwen35.context_length")?;
        if context_limit == 0 || context_limit > train_context || context_limit > 1_048_576 {
            return Err(invalid(
                "context limit must be positive and within the unscaled training context",
            ));
        }
        let mut geometry = Geometry {
            embedding: integer(metadata, "qwen35.embedding_length")?,
            feed_forward: integer(metadata, "qwen35.feed_forward_length")?,
            heads: integer(metadata, "qwen35.attention.head_count")?,
            kv_heads: integer(metadata, "qwen35.attention.head_count_kv")?,
            head: integer(metadata, "qwen35.attention.key_length")?,
            rope: integer(metadata, "qwen35.rope.dimension_count")?,
            rope_sections: [0; 4],
            rope_base: float(metadata, "qwen35.rope.freq_base")?,
            epsilon: float(metadata, "qwen35.attention.layer_norm_rms_epsilon")?,
            ssm_head: integer(metadata, "qwen35.ssm.state_size")?,
            ssm_keys: integer(metadata, "qwen35.ssm.group_count")?,
            ssm_values: integer(metadata, "qwen35.ssm.time_step_rank")?,
            convolution: integer(metadata, "qwen35.ssm.conv_kernel")?,
        };
        let sizes = [
            geometry.embedding,
            geometry.feed_forward,
            geometry.heads,
            geometry.kv_heads,
            geometry.head,
            geometry.rope,
            geometry.ssm_head,
            geometry.ssm_keys,
            geometry.ssm_values,
            geometry.convolution,
        ];
        if sizes.iter().any(|&value| value == 0 || value > 65_535)
            || !geometry.heads.is_multiple_of(geometry.kv_heads)
            || !geometry.ssm_values.is_multiple_of(geometry.ssm_keys)
            || geometry.rope > geometry.head
            || !geometry.rope.is_multiple_of(2)
            || geometry.rope_base <= 0.0
            || geometry.epsilon <= 0.0
            || integer(metadata, "qwen35.attention.value_length")? != geometry.head
            || integer(metadata, "qwen35.ssm.inner_size")? != geometry.value_width()
            || geometry.convolution_width() > MAX_WIDTH
            || geometry.convolution_width() * geometry.convolution > 16 * MAX_WIDTH
            || geometry.heads * geometry.head > MAX_WIDTH
        {
            return Err(invalid(
                "unsupported or invalid Qwen35 head, RoPE, normalization, or convolution geometry",
            ));
        }
        let sections = array(metadata, "qwen35.rope.dimension_sections")?;
        if sections.len() != 4
            || sections.iter().try_fold(0_usize, |sum, value| {
                checked_sum(sum, metadata_usize(value, "RoPE section")?)
            })? != geometry.rope / 2
        {
            return Err(invalid(
                "RoPE sections must describe exactly half the rotated dimensions",
            ));
        }
        for (destination, value) in geometry.rope_sections.iter_mut().zip(sections.iter()) {
            *destination = metadata_usize(value, "RoPE section")?;
        }
        let embedding = tensor(&model, "token_embd.weight", &[geometry.embedding, 0])?;
        let vocabulary = embedding.dimensions[1];
        if vocabulary == 0 || vocabulary > 2_097_152 {
            return Err(invalid("vocabulary size exceeds decoder bounds"));
        }
        let output = if model.tensors.contains_key("output.weight") {
            tensor(&model, "output.weight", &[geometry.embedding, vocabulary])?
        } else {
            embedding.clone()
        };
        let output_norm = vector(&model, "output_norm.weight", geometry.embedding)?;
        let transforms = Transforms::read(&model, &geometry)?;
        let full_interval = optional_integer(metadata, "qwen35.full_attention_interval", 4)?;
        if full_interval == 0 {
            return Err(invalid("full attention interval must be positive"));
        }
        let recurrent_flags = if metadata.contains_key("qwen35.attention.recurrent_layers") {
            let flags = array(metadata, "qwen35.attention.recurrent_layers")?;
            if flags.len() != block_count {
                return Err(invalid("recurrent layer flags must cover all blocks"));
            }
            flags
                .iter()
                .map(|value| {
                    value
                        .as_bool()
                        .ok_or_else(|| invalid("recurrent layer flags must be boolean"))
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            (0..block_count)
                .map(|index| index < layer_count && (index + 1) % full_interval != 0)
                .collect()
        };
        let mut state_bytes = 0_usize;
        for &recurrent in recurrent_flags.iter().take(layer_count) {
            let elements = if recurrent {
                checked_sum(
                    checked_product(&[geometry.ssm_head, geometry.ssm_head, geometry.ssm_values])?,
                    checked_product(&[geometry.convolution - 1, geometry.convolution_width()])?,
                )?
            } else {
                checked_product(&[2, geometry.head, geometry.kv_heads, context_limit])?
            };
            let element_bytes = if recurrent {
                size_of::<f32>()
            } else {
                size_of::<u16>()
            };
            state_bytes = checked_sum(state_bytes, checked_product(&[elements, element_bytes])?)?;
        }
        if state_bytes > state_budget_bytes {
            return Err(invalid(
                "KV and recurrent state exceed the native state budget",
            ));
        }
        for tensor in model.tensors.values() {
            crate::kernels::validate_tensor(tensor)?;
        }
        let mut layers = capacity(layer_count)?;
        for (index, &recurrent) in recurrent_flags.iter().take(layer_count).enumerate() {
            let prefix = format!("blk.{index}.");
            let attention = if recurrent {
                let channels = geometry.convolution_width();
                let convolution = flatten(&tensor(
                    &model,
                    &format!("{prefix}ssm_conv1d.weight"),
                    &[geometry.convolution, channels],
                )?)?;
                Attention::Recurrent(RecurrentAttention {
                    qkv: tensor(
                        &model,
                        &format!("{prefix}attn_qkv.weight"),
                        &[geometry.embedding, channels],
                    )?,
                    gate: tensor(
                        &model,
                        &format!("{prefix}attn_gate.weight"),
                        &[geometry.embedding, geometry.value_width()],
                    )?,
                    beta: tensor(
                        &model,
                        &format!("{prefix}ssm_beta.weight"),
                        &[geometry.embedding, geometry.ssm_values],
                    )?,
                    alpha: tensor(
                        &model,
                        &format!("{prefix}ssm_alpha.weight"),
                        &[geometry.embedding, geometry.ssm_values],
                    )?,
                    convolution,
                    time_bias: vector(
                        &model,
                        &format!("{prefix}ssm_dt.bias"),
                        geometry.ssm_values,
                    )?,
                    decay: vector(&model, &format!("{prefix}ssm_a"), geometry.ssm_values)?,
                    norm: vector(
                        &model,
                        &format!("{prefix}ssm_norm.weight"),
                        geometry.ssm_head,
                    )?,
                    output: tensor(
                        &model,
                        &format!("{prefix}ssm_out.weight"),
                        &[geometry.value_width(), geometry.embedding],
                    )?,
                })
            } else {
                Attention::Full(FullAttention {
                    query: tensor(
                        &model,
                        &format!("{prefix}attn_q.weight"),
                        &[geometry.embedding, geometry.head * geometry.heads * 2],
                    )?,
                    key: tensor(
                        &model,
                        &format!("{prefix}attn_k.weight"),
                        &[geometry.embedding, geometry.head * geometry.kv_heads],
                    )?,
                    value: tensor(
                        &model,
                        &format!("{prefix}attn_v.weight"),
                        &[geometry.embedding, geometry.head * geometry.kv_heads],
                    )?,
                    output: tensor(
                        &model,
                        &format!("{prefix}attn_output.weight"),
                        &[geometry.head * geometry.heads, geometry.embedding],
                    )?,
                    query_norm: vector(
                        &model,
                        &format!("{prefix}attn_q_norm.weight"),
                        geometry.head,
                    )?,
                    key_norm: vector(
                        &model,
                        &format!("{prefix}attn_k_norm.weight"),
                        geometry.head,
                    )?,
                })
            };
            layers.push(Layer {
                norm: vector(
                    &model,
                    &format!("{prefix}attn_norm.weight"),
                    geometry.embedding,
                )?,
                post_norm: vector(
                    &model,
                    &format!("{prefix}post_attention_norm.weight"),
                    geometry.embedding,
                )?,
                attention,
                up: tensor(
                    &model,
                    &format!("{prefix}ffn_up.weight"),
                    &[geometry.embedding, geometry.feed_forward],
                )?,
                gate: tensor(
                    &model,
                    &format!("{prefix}ffn_gate.weight"),
                    &[geometry.embedding, geometry.feed_forward],
                )?,
                down: tensor(
                    &model,
                    &format!("{prefix}ffn_down.weight"),
                    &[geometry.feed_forward, geometry.embedding],
                )?,
            });
        }
        reject_unconsumed_tensors(&model, &layers, block_count, mtp)?;
        let mut state = capacity(layer_count)?;
        for layer in &layers {
            state.push(match &layer.attention {
                Attention::Full(_) => LayerState::Full {
                    keys: KvCache::new(geometry.head * geometry.kv_heads, context_limit)?,
                    values: KvCache::new(geometry.head * geometry.kv_heads, context_limit)?,
                },
                Attention::Recurrent(_) => LayerState::Recurrent {
                    convolution: zeros((geometry.convolution - 1) * geometry.convolution_width())?,
                    matrix: zeros(geometry.ssm_head * geometry.ssm_head * geometry.ssm_values)?,
                },
            });
        }
        let mut deltas = capacity(layer_count)?;
        for state in &state {
            deltas.push(match state {
                LayerState::Full { .. } => LayerDelta::Full {
                    key: Vec::new(),
                    value: Vec::new(),
                },
                LayerState::Recurrent { .. } => LayerDelta::Recurrent {
                    convolution: Vec::new(),
                    matrix: Vec::new(),
                },
            });
        }
        Ok(Self {
            provenance: Arc::new(()),
            identity: model.identity + ":qwen35-text-v1:nonflash-f16:kv-f16:gdn-f32",
            geometry,
            embedding,
            output,
            output_norm,
            transforms,
            layers,
            state,
            workspace: Workspace {
                deltas,
                projection: ProjectionScratch {
                    matvec: MatvecScratch::with_backend(backend),
                    ..ProjectionScratch::default()
                },
                ..Workspace::default()
            },
            context_limit,
            position: 0,
            vocabulary,
        })
    }

    /// Returns the configured packed-matrix kernel request.
    ///
    /// This is a dispatch policy, not a whole-graph effective backend. Individual
    /// projections may fall back to scalar even after a NEON projection succeeds.
    /// Checkpoint restoration and reset preserve this request.
    #[must_use]
    pub const fn requested_backend(&self) -> MatvecBackend {
        self.workspace.projection.matvec.requested_backend()
    }

    /// Diagnostics for the latest matrix attempt, including provisional work.
    ///
    /// This describes one projection, not every projection in a token or graph.
    /// Failure before matrix preparation, reset and checkpoint restoration leave
    /// the record unchanged. A successful matrix may precede a later graph
    /// failure; this record does not establish committed conversation state.
    #[must_use]
    pub const fn backend_diagnostics(&self) -> MatvecDiagnostics {
        self.workspace.projection.matvec.diagnostics()
    }

    /// Aggregate matrix dispatch and failure counts for this decoder's lifetime.
    ///
    /// Reset and checkpoint restoration preserve these counts, including failed
    /// and provisional graph work. They do not describe committed token counts
    /// or establish whole-model backend coverage or performance.
    #[must_use]
    pub const fn backend_statistics(&self) -> MatvecStatistics {
        self.workspace.projection.matvec.statistics()
    }

    #[allow(
        clippy::too_many_lines,
        reason = "Keep validation and transactional graph operation order auditable together."
    )]
    fn evaluate_token_into(&mut self, token: u32, logits: Option<&mut Vec<f32>>) -> Result<()> {
        let token = usize::try_from(token)?;
        if token >= self.vocabulary || self.position >= self.context_limit {
            return Err(invalid(
                "token is outside the vocabulary or context is exhausted",
            ));
        }
        let Workspace {
            hidden,
            normalized,
            up,
            gate,
            attention,
            projection,
            attention_scratch,
            deltas,
        } = &mut self.workspace;
        let mut evaluation = Evaluation {
            geometry: &self.geometry,
            transforms: &self.transforms,
            position: self.position,
            frequencies_ready: false,
            projection,
        };
        row_into(&self.embedding, token, hidden)?;
        if self.transforms.inverse_embedding {
            self.transforms.rotate(hidden)?;
            self.transforms.sign(hidden);
        }
        for ((layer, state), delta) in self.layers.iter().zip(&self.state).zip(deltas.iter_mut()) {
            rms_norm_into(hidden, &layer.norm, self.geometry.epsilon, normalized)?;
            match (&layer.attention, state) {
                (Attention::Full(weights), LayerState::Full { keys, values }) => {
                    evaluation.full_attention(
                        weights,
                        normalized,
                        keys,
                        values,
                        attention_scratch,
                        delta,
                        attention,
                    )?;
                }
                (
                    Attention::Recurrent(weights),
                    LayerState::Recurrent {
                        convolution,
                        matrix,
                    },
                ) => evaluation.recurrent_attention(
                    weights,
                    normalized,
                    convolution,
                    matrix,
                    attention_scratch,
                    delta,
                    attention,
                )?,
                _ => return Err(invalid("decoder layer/cache kind mismatch")),
            }
            add(hidden, attention);
            rms_norm_into(hidden, &layer.post_norm, self.geometry.epsilon, normalized)?;
            evaluation.project_group_into([&layer.up, &layer.gate], normalized, [up, gate])?;
            for (up, &gate) in up.iter_mut().zip(gate.iter()) {
                *up *= silu(gate);
            }
            evaluation.project_into(&layer.down, up, attention)?;
            add(hidden, attention);
            finite(hidden, "decoder layer output")?;
        }
        if let Some(logits) = logits {
            rms_norm_into(hidden, &self.output_norm, self.geometry.epsilon, normalized)?;
            evaluation.project_into(&self.output, normalized, logits)?;
            finite(logits, "output logits")?;
        }
        // Reserve every append before changing numerical state. Deltas retain
        // their allocations across tokens but never alias checkpoint storage.
        for state in &mut self.state {
            if let LayerState::Full { keys, values } = state {
                keys.reserve_append()?;
                values.reserve_append()?;
            }
        }
        for (state, delta) in self.state.iter_mut().zip(deltas.iter_mut()) {
            match (state, delta) {
                (LayerState::Full { keys, values }, LayerDelta::Full { key, value }) => {
                    keys.append(key);
                    values.append(value);
                }
                (
                    LayerState::Recurrent {
                        convolution,
                        matrix,
                    },
                    LayerDelta::Recurrent {
                        convolution: new_convolution,
                        matrix: new_matrix,
                    },
                ) => {
                    std::mem::swap(convolution, new_convolution);
                    std::mem::swap(matrix, new_matrix);
                }
                // Every kind was checked by the corresponding attention call.
                _ => {}
            }
        }
        self.position += 1;
        Ok(())
    }
}

impl Evaluation<'_> {
    fn project_group_into<const N: usize>(
        &mut self,
        tensors: [&Tensor; N],
        input: &[f32],
        outputs: [&mut Vec<f32>; N],
    ) -> Result<()> {
        self.transforms
            .project_group_into(tensors, input, self.geometry, outputs, self.projection)
    }

    fn project_into(
        &mut self,
        tensor: &Tensor,
        input: &[f32],
        output: &mut Vec<f32>,
    ) -> Result<()> {
        self.transforms
            .project_into(tensor, input, self.geometry, output, self.projection)
    }

    #[allow(
        clippy::too_many_arguments,
        clippy::suboptimal_flops,
        clippy::cast_possible_truncation,
        clippy::as_conversions,
        reason = "Preserve scalar nonflash F16 matmul inputs and binary64 accumulation rounded to binary32."
    )]
    fn full_attention(
        &mut self,
        layer: &FullAttention,
        input: &[f32],
        keys: &KvCache,
        values: &KvCache,
        scratch: &mut AttentionScratch,
        delta: &mut LayerDelta,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        let LayerDelta::Full { key, value } = delta else {
            return Err(invalid("decoder full-attention delta kind mismatch"));
        };
        let g = self.geometry;
        self.project_group_into(
            [&layer.query, &layer.key, &layer.value],
            input,
            [&mut scratch.joint_query, &mut scratch.key_projection, value],
        )?;
        prepare(&mut scratch.query, g.head * g.heads)?;
        prepare(&mut scratch.gate, g.head * g.heads)?;
        for head in scratch.joint_query.chunks_exact(2 * g.head) {
            rms_norm_into(
                &head[..g.head],
                &layer.query_norm,
                g.epsilon,
                &mut scratch.normalized,
            )?;
            scratch.query.extend_from_slice(&scratch.normalized);
            scratch.gate.extend_from_slice(&head[g.head..]);
        }
        prepare(key, scratch.key_projection.len())?;
        for head in scratch.key_projection.chunks_exact(g.head) {
            rms_norm_into(head, &layer.key_norm, g.epsilon, &mut scratch.normalized)?;
            key.extend_from_slice(&scratch.normalized);
        }
        // Geometry and position are shared by every full-attention layer in
        // this token evaluation. Reuse the exact scalar frequencies for Q/K
        // and subsequent layers; the flag resets for every evaluation attempt.
        if !self.frequencies_ready {
            rope_frequencies(g, self.position, &mut scratch.frequencies)?;
            self.frequencies_ready = true;
        }
        rope(&mut scratch.query, g, &scratch.frequencies)?;
        rope(key, g, &scratch.frequencies)?;
        // F16 K x Q converts the F32 query to F16 even with F32 precision.
        for element in scratch
            .query
            .iter_mut()
            .chain(key.iter_mut())
            .chain(value.iter_mut())
        {
            *element = round_f16(*element);
        }
        finite(&scratch.query, "binary16 attention query")?;
        finite(key, "binary16 key cache")?;
        finite(value, "binary16 value cache")?;
        zeros_into(g.head * g.heads, &mut scratch.attended)?;
        let scale = 1.0 / small_float(g.head)?.sqrt();
        for head in 0..g.heads {
            let kv_head = head / (g.heads / g.kv_heads);
            let offset = kv_head * g.head;
            let q = &scratch.query[head * g.head..(head + 1) * g.head];
            prepare(&mut scratch.scores, self.position + 1)?;
            let scores = &mut scratch.scores;
            for past in keys.tokens() {
                scores.push(dot_f16(q, &past[offset..offset + g.head]) * scale);
            }
            scores.push(dot(q, &key[offset..offset + g.head]) * scale);
            softmax(scores)?;
            // Nonflash V(F16) x softmax(F32) also converts its RHS to F16.
            for probability in scores.iter_mut() {
                *probability = round_f16(*probability);
            }
            let result = &mut scratch.attended[head * g.head..(head + 1) * g.head];
            for (dimension, out) in result.iter_mut().enumerate() {
                let past_sum = values
                    .tokens()
                    .zip(scores.iter())
                    .map(|(past, &probability)| {
                        f64::from(probability * half_to_float(past[offset + dimension]))
                    })
                    .sum::<f64>();
                *out = (past_sum + f64::from(scores[self.position] * value[offset + dimension]))
                    as f32;
            }
        }
        for (out, &gate) in scratch.attended.iter_mut().zip(&scratch.gate) {
            *out *= sigmoid(gate);
        }
        self.project_into(&layer.output, &scratch.attended, output)
    }

    #[allow(
        clippy::too_many_lines,
        reason = "Keep validation and graph operation order auditable together."
    )]
    #[allow(
        clippy::suboptimal_flops,
        clippy::imprecise_flops,
        reason = "Match the pinned scalar GDN operations, including logf(1+expf(x))."
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "Separate immutable committed state from reusable transactional scratch."
    )]
    fn recurrent_attention(
        &mut self,
        layer: &RecurrentAttention,
        input: &[f32],
        history: &[f32],
        previous: &[f32],
        scratch: &mut AttentionScratch,
        delta: &mut LayerDelta,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        let LayerDelta::Recurrent {
            convolution,
            matrix,
        } = delta
        else {
            return Err(invalid("decoder recurrent-attention delta kind mismatch"));
        };
        let g = self.geometry;
        self.project_group_into(
            [&layer.qkv, &layer.gate, &layer.beta, &layer.alpha],
            input,
            [
                &mut scratch.qkv,
                &mut scratch.gate,
                &mut scratch.beta,
                &mut scratch.alpha,
            ],
        )?;
        copy_into(history, convolution)?;
        zeros_into(g.convolution_width(), &mut scratch.mixed)?;
        let mixed = &mut scratch.mixed;
        for (channel, mixed_value) in mixed.iter_mut().enumerate() {
            let start = channel * (g.convolution - 1);
            let coefficients =
                &layer.convolution[channel * g.convolution..(channel + 1) * g.convolution];
            let mut sum = 0.0;
            for (old, coefficient) in history[start..start + g.convolution - 1]
                .iter()
                .zip(coefficients)
            {
                sum += old * coefficient;
            }
            sum += scratch.qkv[channel] * coefficients[g.convolution - 1];
            *mixed_value = silu(sum);
            if g.convolution > 1 {
                convolution[start..start + g.convolution - 1].rotate_left(1);
                convolution[start + g.convolution - 2] = scratch.qkv[channel];
            }
        }
        let key_width = g.key_width();
        for head in mixed[..2 * key_width].chunks_exact_mut(g.ssm_head) {
            l2_norm(head, g.epsilon);
        }
        copy_into(previous, matrix)?;
        zeros_into(g.value_width(), &mut scratch.attended)?;
        let scale = 1.0 / small_float(g.ssm_head)?.sqrt();
        for head in 0..g.ssm_values {
            // ggml repeat tiles key groups; it does not repeat each key head contiguously.
            let key_head = head % g.ssm_keys;
            let q = &mixed[key_head * g.ssm_head..(key_head + 1) * g.ssm_head];
            let k =
                &mixed[key_width + key_head * g.ssm_head..key_width + (key_head + 1) * g.ssm_head];
            let v =
                &mixed[2 * key_width + head * g.ssm_head..2 * key_width + (head + 1) * g.ssm_head];
            let activation = scratch.alpha[head] + layer.time_bias[head];
            let softplus = if activation > 20.0 {
                activation
            } else {
                (1.0 + activation.exp()).ln()
            };
            let decay = (softplus * layer.decay[head]).exp();
            let update = sigmoid(scratch.beta[head]);
            let state =
                &mut matrix[head * g.ssm_head * g.ssm_head..(head + 1) * g.ssm_head * g.ssm_head];
            for element in state.iter_mut() {
                *element *= decay;
            }
            let output = &mut scratch.attended[head * g.ssm_head..(head + 1) * g.ssm_head];
            for ((state_row, &value), out) in state
                .chunks_exact_mut(g.ssm_head)
                .zip(v)
                .zip(output.iter_mut())
            {
                let delta = (value - dot(state_row, k)) * update;
                for (element, &key) in state_row.iter_mut().zip(k) {
                    *element += delta * key;
                }
                *out = dot(state_row, q) * scale;
            }
            rms_norm_into(output, &layer.norm, g.epsilon, &mut scratch.normalized)?;
            for ((out, &normalized), &z) in output
                .iter_mut()
                .zip(&scratch.normalized)
                .zip(&scratch.gate[head * g.ssm_head..(head + 1) * g.ssm_head])
            {
                *out = normalized * silu(z);
            }
        }
        finite(matrix, "GDN state")?;
        finite(convolution, "convolution state")?;
        self.project_into(&layer.output, &scratch.attended, output)
    }
}

impl Decoder for Qwen35 {
    type Checkpoint = Qwen35Checkpoint;

    fn identity(&self) -> &str {
        &self.identity
    }

    fn vocab_size(&self) -> usize {
        self.vocabulary
    }

    fn reset(&mut self) -> Result<()> {
        for state in &mut self.state {
            match state {
                LayerState::Full { keys, values } => {
                    keys.clear();
                    values.clear();
                }
                LayerState::Recurrent {
                    convolution,
                    matrix,
                } => {
                    convolution.fill(0.0);
                    matrix.fill(0.0);
                }
            }
        }
        self.position = 0;
        Ok(())
    }

    fn evaluate(&mut self, token: u32) -> Result<Vec<f32>> {
        let mut logits = Vec::new();
        self.evaluate_into(token, &mut logits)?;
        Ok(logits)
    }

    fn evaluate_into(&mut self, token: u32, logits: &mut Vec<f32>) -> Result<()> {
        logits.clear();
        let result = self.evaluate_token_into(token, Some(logits));
        if result.is_err() {
            logits.clear();
        }
        result
    }
    fn ingest(&mut self, token: u32) -> Result<()> {
        self.evaluate_token_into(token, None)
    }
    fn checkpoint_bytes(&self) -> usize {
        // Deliberately charge each referenced page in full for every snapshot,
        // including its Arc header and page table. Runtime admission therefore
        // overcounts shared pages rather than undercounting unique pages or
        // retained recurrent snapshots. Tails are private: subsequent append
        // needs no payload COW. Allocator overhead and graph scratch remain
        // outside this element/metadata storage bound, as for live state.
        size_of::<Qwen35Checkpoint>()
            + self.identity.capacity()
            + self.state.capacity() * size_of::<LayerState>()
            + self.state.iter().map(LayerState::bytes).sum::<usize>()
    }

    fn checkpoint(&self) -> Result<Self::Checkpoint> {
        Ok(Qwen35Checkpoint {
            provenance: self.provenance.clone(),
            identity: copy_string(&self.identity)?,
            context_limit: self.context_limit,
            position: self.position,
            layers: copy_state(&self.state)?,
        })
    }

    fn restore(&mut self, checkpoint: &Self::Checkpoint) -> Result<()> {
        if !Arc::ptr_eq(&checkpoint.provenance, &self.provenance)
            || checkpoint.identity != self.identity
            || checkpoint.context_limit != self.context_limit
            || checkpoint.layers.len() != self.layers.len()
            || checkpoint.position > self.context_limit
        {
            return Err(invalid(
                "checkpoint belongs to a different model or context",
            ));
        }
        self.state = copy_state(&checkpoint.layers)?;
        self.position = checkpoint.position;
        Ok(())
    }
}

fn foldable(name: &str) -> bool {
    if name == "output.weight" {
        return true;
    }
    let Some(rest) = name.strip_prefix("blk.") else {
        return false;
    };
    let Some((index, suffix)) = rest.split_once('.') else {
        return false;
    };
    !index.is_empty()
        && index.bytes().all(|byte| byte.is_ascii_digit())
        && matches!(
            suffix,
            "attn_q.weight"
                | "attn_k.weight"
                | "attn_v.weight"
                | "attn_qkv.weight"
                | "attn_gate.weight"
                | "attn_output.weight"
                | "ffn_gate.weight"
                | "ffn_up.weight"
                | "ffn_down.weight"
                | "ssm_out.weight"
        )
}

fn reject_unsupported(metadata: &Metadata) -> Result<()> {
    for key in metadata.keys() {
        if let Some(key) = key.strip_prefix("qwen35.")
            && !matches!(
                key,
                "block_count"
                    | "nextn_predict_layers"
                    | "context_length"
                    | "embedding_length"
                    | "feed_forward_length"
                    | "attention.head_count"
                    | "attention.head_count_kv"
                    | "attention.key_length"
                    | "attention.value_length"
                    | "attention.layer_norm_rms_epsilon"
                    | "attention.recurrent_layers"
                    | "attention.causal"
                    | "rope.dimension_count"
                    | "rope.dimension_sections"
                    | "rope.freq_base"
                    | "rope.scaling.type"
                    | "ssm.state_size"
                    | "ssm.group_count"
                    | "ssm.time_step_rank"
                    | "ssm.inner_size"
                    | "ssm.conv_kernel"
                    | "full_attention_interval"
                    | "expert_count"
                    | "expert_used_count"
            )
        {
            return Err(invalid(format!("unsupported Qwen35 metadata: {key}")));
        }
        if let Some(key) = key.strip_prefix("prism.hadamard.")
            && !matches!(
                key,
                "version"
                    | "tied_output"
                    | "block_size"
                    | "transform"
                    | "axis"
                    | "sign_mode"
                    | "weight_names"
                    | "inverse_weight_names"
                    | "sign_widths"
                    | "sign_values"
                    | "gdn_v_grouped"
            )
        {
            return Err(invalid(format!("unsupported Hadamard metadata: {key}")));
        }
    }
    for key in ["qwen35.expert_count", "qwen35.expert_used_count"] {
        if optional_integer(metadata, key, 0)? != 0 {
            return Err(invalid("MoE graph is not supported"));
        }
    }
    for key in [
        "qwen35.rope.scaling.factor",
        "qwen35.rope.scaling.attn_factor",
        "qwen35.attention.scale",
    ] {
        if metadata.contains_key(key) {
            return Err(invalid(format!("unsupported graph override: {key}")));
        }
    }
    if let Some(scaling) = metadata.get("qwen35.rope.scaling.type")
        && scaling.as_str() != Some("none")
    {
        return Err(invalid("scaled RoPE is not supported"));
    }
    if !optional_bool(metadata, "qwen35.attention.causal", true)? {
        return Err(invalid("noncausal Qwen35 attention is not supported"));
    }
    Ok(())
}

fn reject_unconsumed_tensors(
    model: &Model,
    layers: &[Layer],
    block_count: usize,
    mtp: usize,
) -> Result<()> {
    let mut consumed: BTreeSet<String> =
        ["token_embd.weight", "output.weight", "output_norm.weight"]
            .into_iter()
            .map(str::to_owned)
            .collect();
    for (index, layer) in layers.iter().enumerate() {
        for suffix in [
            "attn_norm.weight",
            "post_attention_norm.weight",
            "ffn_up.weight",
            "ffn_gate.weight",
            "ffn_down.weight",
        ] {
            consumed.insert(format!("blk.{index}.{suffix}"));
        }
        let attention: &[&str] = match &layer.attention {
            Attention::Full(_) => &[
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "attn_q_norm.weight",
                "attn_k_norm.weight",
            ],
            Attention::Recurrent(_) => &[
                "attn_qkv.weight",
                "attn_gate.weight",
                "ssm_beta.weight",
                "ssm_alpha.weight",
                "ssm_conv1d.weight",
                "ssm_dt.bias",
                "ssm_a",
                "ssm_norm.weight",
                "ssm_out.weight",
            ],
        };
        for suffix in attention {
            consumed.insert(format!("blk.{index}.{suffix}"));
        }
    }
    for (name, tensor) in &model.tensors {
        let mtp_tensor = mtp != 0
            && (name.starts_with("nextn.")
                || name.contains(".nextn.")
                || name
                    .strip_prefix("blk.")
                    .and_then(|rest| rest.split_once('.'))
                    .and_then(|(index, _)| index.parse::<usize>().ok())
                    .is_some_and(|index| index >= layers.len() && index < block_count));
        if tensor.name != *name || (!consumed.contains(name) && !mtp_tensor) {
            return Err(invalid(format!("unsupported or unconsumed tensor: {name}")));
        }
    }
    Ok(())
}

fn tensor(model: &Model, name: &str, shape: &[usize]) -> Result<Tensor> {
    let tensor = model
        .tensors
        .get(name)
        .ok_or_else(|| invalid(format!("required tensor absent: {name}")))?;
    if tensor.name != name
        || tensor.dimensions.len() != shape.len()
        || tensor
            .dimensions
            .iter()
            .zip(shape)
            .any(|(&actual, &expected)| actual == 0 || (expected != 0 && actual != expected))
    {
        return Err(invalid(format!("tensor shape mismatch: {name}")));
    }
    // Validate the packed layout without expanding large weight payloads.
    crate::kernels::validate_tensor(tensor)?;
    Ok(tensor.clone())
}

fn vector(model: &Model, name: &str, width: usize) -> Result<Vec<f32>> {
    let tensor = tensor(model, name, &[width])?;
    if !matches!(tensor.kind, 0 | 1 | 30) {
        return Err(invalid(format!(
            "normalization/state tensor must be floating point: {name}"
        )));
    }
    let result = row(&tensor, 0)?;
    finite(&result, name)?;
    Ok(result)
}

fn flatten(tensor: &Tensor) -> Result<Vec<f32>> {
    if !matches!(tensor.kind, 0 | 1 | 30) {
        return Err(invalid("convolution weights must be floating point"));
    }
    let mut values = Vec::new();
    for index in 0..tensor.dimensions[1] {
        values.extend(row(tensor, index)?);
    }
    finite(&values, &tensor.name)?;
    Ok(values)
}

fn integer(metadata: &Metadata, key: &str) -> Result<usize> {
    metadata_usize(
        metadata
            .get(key)
            .ok_or_else(|| invalid(format!("required metadata absent: {key}")))?
            .as_ref(),
        key,
    )
}

fn optional_integer(metadata: &Metadata, key: &str, default: usize) -> Result<usize> {
    metadata
        .get(key)
        .map_or(Ok(default), |value| metadata_usize(value.as_ref(), key))
}

fn metadata_usize(value: MetadataValueRef<'_>, name: &str) -> Result<usize> {
    usize::try_from(
        value
            .as_u64()
            .ok_or_else(|| invalid(format!("metadata must be an unsigned integer: {name}")))?,
    )
    .map_err(Into::into)
}

fn optional_bool(metadata: &Metadata, key: &str, default: bool) -> Result<bool> {
    metadata.get(key).map_or(Ok(default), |value| {
        value
            .as_bool()
            .ok_or_else(|| invalid(format!("metadata must be boolean: {key}")))
    })
}

fn string<'a>(metadata: &'a Metadata, key: &str) -> Result<&'a str> {
    metadata
        .get(key)
        .and_then(MetadataValue::as_str)
        .ok_or_else(|| invalid(format!("metadata must be a string: {key}")))
}

fn array<'a>(metadata: &'a Metadata, key: &str) -> Result<&'a MetadataArray> {
    metadata
        .get(key)
        .and_then(MetadataValue::as_array)
        .ok_or_else(|| invalid(format!("metadata must be an array: {key}")))
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::as_conversions,
    reason = "The pinned scalar numerical contract explicitly rounds this value to binary32."
)]
fn float(metadata: &Metadata, key: &str) -> Result<f32> {
    let value = metadata
        .get(key)
        .and_then(MetadataValue::as_f64)
        .ok_or_else(|| invalid(format!("metadata must be numeric: {key}")))?;
    let result = value as f32;
    if !result.is_finite() {
        return Err(invalid(format!("nonfinite metadata: {key}")));
    }
    Ok(result)
}

fn checked_product(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1_usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or_else(|| invalid("decoder allocation size overflow"))
    })
}

fn checked_sum(first: usize, second: usize) -> Result<usize> {
    first
        .checked_add(second)
        .ok_or_else(|| invalid("decoder allocation size overflow"))
}

fn zeros(count: usize) -> Result<Vec<f32>> {
    if count > MAX_STATE_BYTES / size_of::<f32>() {
        return Err(invalid("decoder allocation exceeds state budget"));
    }
    let mut result = Vec::new();
    result.try_reserve_exact(count)?;
    result.resize(count, 0.0);
    Ok(result)
}

fn prepare<T>(output: &mut Vec<T>, count: usize) -> Result<()> {
    output.clear();
    if count
        .checked_mul(size_of::<T>())
        .is_none_or(|bytes| bytes > MAX_STATE_BYTES)
    {
        return Err(invalid("decoder scratch exceeds allocation budget"));
    }
    output.try_reserve_exact(count)?;
    Ok(())
}

fn zeros_into(count: usize, output: &mut Vec<f32>) -> Result<()> {
    prepare(output, count)?;
    output.resize(count, 0.0);
    Ok(())
}

fn copy_into(input: &[f32], output: &mut Vec<f32>) -> Result<()> {
    prepare(output, input.len())?;
    output.extend_from_slice(input);
    Ok(())
}

fn capacity<T>(count: usize) -> Result<Vec<T>> {
    if count
        .checked_mul(size_of::<T>())
        .is_none_or(|bytes| bytes > MAX_STATE_BYTES)
    {
        return Err(invalid("decoder vector exceeds allocation budget"));
    }
    let mut result = Vec::new();
    result.try_reserve_exact(count)?;
    Ok(result)
}

fn copy_values<T: Copy>(values: &[T]) -> Result<Vec<T>> {
    let mut result = capacity(values.len())?;
    result.extend_from_slice(values);
    Ok(result)
}

fn copy_string(value: &str) -> Result<String> {
    let mut result = String::new();
    result.try_reserve_exact(value.len())?;
    result.push_str(value);
    Ok(result)
}

fn copy_state(states: &[LayerState]) -> Result<Vec<LayerState>> {
    let mut result = capacity(states.len())?;
    for state in states {
        result.push(match state {
            LayerState::Full { keys, values } => LayerState::Full {
                keys: keys.copy()?,
                values: values.copy()?,
            },
            LayerState::Recurrent {
                convolution,
                matrix,
            } => LayerState::Recurrent {
                convolution: copy_values(convolution)?,
                matrix: copy_values(matrix)?,
            },
        });
    }
    Ok(result)
}

fn small_float(value: usize) -> Result<f32> {
    Ok(f32::from(u16::try_from(value)?))
}

fn finite(values: &[f32], operation: &str) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(invalid(format!("nonfinite values in {operation}")));
    }
    Ok(())
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::as_conversions,
    reason = "The pinned scalar numerical contract explicitly rounds this value to binary32."
)]
fn dot(first: &[f32], second: &[f32]) -> f32 {
    first
        .iter()
        .zip(second)
        .map(|(&first, &second)| f64::from(first * second))
        .sum::<f64>() as f32
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::as_conversions,
    reason = "Preserve binary32 products and binary64 accumulation when reading the binary16 cache."
)]
fn dot_f16(first: &[f32], second: &[u16]) -> f32 {
    // Widen each cached operand before the binary32 multiply, then accumulate
    // its rounded product in binary64 exactly as the former float cache did.
    first
        .iter()
        .zip(second)
        .map(|(&first, &second)| f64::from(first * half_to_float(second)))
        .sum::<f64>() as f32
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::as_conversions,
    reason = "The pinned scalar numerical contract explicitly rounds this value to binary32."
)]
fn l2_norm(input: &mut [f32], epsilon: f32) {
    let squares = input
        .iter()
        .map(|&value| f64::from(value * value))
        .sum::<f64>();
    let scale = 1.0 / (squares as f32).sqrt().max(epsilon);
    for value in input {
        *value *= scale;
    }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn add(target: &mut [f32], other: &[f32]) {
    for (target, other) in target.iter_mut().zip(other) {
        *target += other;
    }
}

#[allow(
    clippy::cast_precision_loss,
    clippy::as_conversions,
    reason = "Bounded token positions are converted to the same binary32 RoPE input as Prism."
)]
fn rope_frequencies(
    geometry: &Geometry,
    position: usize,
    frequencies: &mut Vec<(f32, f32)>,
) -> Result<()> {
    let theta_scale = geometry.rope_base.powf(-2.0 / small_float(geometry.rope)?);
    prepare(frequencies, geometry.rope / 2)?;
    let mut theta = position as f32;
    for index in 0..geometry.rope / 2 {
        // Text positions are [p,p,p,0]. Interleaved MRoPE falls back to the
        // fourth (zero) position for sectors outside their declared section.
        let angle = if index < 3 * geometry.rope_sections[index % 3] {
            theta
        } else {
            0.0
        };
        frequencies.push((angle.cos(), angle.sin()));
        theta *= theta_scale;
    }
    Ok(())
}

#[allow(
    clippy::suboptimal_flops,
    reason = "Preserve separate multiply/add rounding in the scalar RoPE reference."
)]
fn rope(input: &mut [f32], geometry: &Geometry, frequencies: &[(f32, f32)]) -> Result<()> {
    for head in input.chunks_exact_mut(geometry.head) {
        for (index, &(cosine, sine)) in frequencies.iter().enumerate() {
            let first = head[index];
            let second = head[index + geometry.rope / 2];
            head[index] = first * cosine - second * sine;
            head[index + geometry.rope / 2] = first * sine + second * cosine;
        }
    }
    finite(input, "RoPE output")
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::Arc;

    use super::{
        Geometry, KV_PAGE_TOKENS, KvCache, LayerState, MAX_STATE_BYTES, ProjectionScratch,
        Transforms, copy_state, dot, dot_f16, half_bits, half_to_float, rope, rope_frequencies,
    };
    use crate::Result;
    use crate::model::Tensor;

    fn projection_geometry(width: usize) -> Geometry {
        Geometry {
            embedding: width,
            feed_forward: width,
            heads: 1,
            kv_heads: 1,
            head: width,
            rope: width,
            rope_sections: [width / 2, 0, 0, 0],
            rope_base: 10_000.0,
            epsilon: 0.000_001,
            ssm_head: width / 4,
            ssm_keys: 2,
            ssm_values: 4,
            convolution: 1,
        }
    }

    fn projection_weight(name: &str, kind: u32, width: usize) -> Tensor {
        let data: Vec<u8> = match kind {
            142 => {
                let mut block = [0x66; 34];
                block[..2].copy_from_slice(&0x3555_u16.to_le_bytes());
                block.repeat(width / 128)
            }
            8 => {
                let mut block = [1; 34];
                block[..2].copy_from_slice(&0x3555_u16.to_le_bytes());
                block.repeat(width / 32)
            }
            _ => (0..width)
                .flat_map(|index| {
                    [0.125_f32, -0.25, 0.5, -1.0, 2.0, -4.0, 0.0, -0.0][index % 8].to_le_bytes()
                })
                .collect(),
        };
        Tensor {
            name: name.to_owned(),
            dimensions: vec![width],
            kind,
            data: Arc::from(data),
        }
    }

    #[test]
    fn grouped_projection_shares_only_matching_basis_permutation_and_quantizer() -> Result<()> {
        for width in [8, 256] {
            let geometry = projection_geometry(width);
            let kind = if width == 256 { 142 } else { 0 };
            let weights = [
                projection_weight("up.weight", kind, width),
                projection_weight("gate.weight", kind, width),
                projection_weight("blk.0.ssm_out.weight", kind, width),
                projection_weight("plain.weight", kind, width),
            ];
            let mut transforms = Transforms {
                block: 4,
                weights: weights[..3]
                    .iter()
                    .map(|weight| weight.name.clone())
                    .collect::<BTreeSet<_>>(),
                inverse_embedding: false,
                grouped_values: true,
                signs: BTreeMap::from([(
                    width,
                    (0..width)
                        .map(|index| if index % 3 == 0 { -1.0 } else { 1.0 })
                        .collect(),
                )]),
            };
            let input: Vec<f32> = (0..width)
                .map(|index| [127.0, 0.5, 1.5, -63.5, 2.5, -0.5, 4.0, -16.0][index % 8])
                .collect();
            let mut expected = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
            let mut independent = ProjectionScratch::default();
            for (weight, output) in weights.iter().zip(&mut expected) {
                transforms.project_into(weight, &input, &geometry, output, &mut independent)?;
            }
            let mut actual = [
                vec![f32::NAN],
                vec![f32::NAN],
                vec![f32::NAN],
                vec![f32::NAN],
            ];
            let mut scratch = ProjectionScratch::default();
            transforms.project_group_into(
                weights.each_ref(),
                &input,
                &geometry,
                actual.each_mut(),
                &mut scratch,
            )?;
            for (actual, expected) in actual.iter().zip(&expected) {
                assert!(
                    actual
                        .iter()
                        .map(|value| value.to_bits())
                        .eq(expected.iter().map(|value| value.to_bits()))
                );
            }
            assert_eq!(scratch.transforms, 2);
            assert_eq!(scratch.matvec.preparation_count(), 3);
            assert_eq!(independent.transforms, 3);
            assert_eq!(independent.matvec.preparation_count(), 4);

            // Same basis is still insufficient for PQ2 Q8_K vs legacy Q8_0.
            if width == 256 {
                transforms.grouped_values = false;
                let legacy = projection_weight("gate.weight", 8, width);
                let before = scratch.matvec.preparation_count();
                let transforms_before = scratch.transforms;
                let [first, second, ..] = &mut actual;
                transforms.project_group_into(
                    [&weights[0], &legacy],
                    &input,
                    &geometry,
                    [first, second],
                    &mut scratch,
                )?;
                assert_eq!(scratch.matvec.preparation_count() - before, 2);
                assert_eq!(scratch.transforms - transforms_before, 2);
                transforms.project_into(
                    &legacy,
                    &input,
                    &geometry,
                    &mut expected[1],
                    &mut independent,
                )?;
                assert!(
                    actual[1]
                        .iter()
                        .map(|value| value.to_bits())
                        .eq(expected[1].iter().map(|value| value.to_bits()))
                );
            }
        }
        Ok(())
    }

    #[test]
    fn grouped_projection_failure_clears_all_siblings_and_reprepares() -> Result<()> {
        let geometry = projection_geometry(8);
        let first = projection_weight("first", 0, 8);
        let mut second = projection_weight("second", 0, 8);
        second.data = Arc::from(f32::NAN.to_le_bytes().repeat(8));
        let transforms = Transforms::default();
        let mut scratch = ProjectionScratch::default();
        let mut output = [vec![1.0], vec![2.0]];
        assert!(
            transforms
                .project_group_into(
                    [&first, &second],
                    &[1.0; 8],
                    &geometry,
                    output.each_mut(),
                    &mut scratch
                )
                .is_err()
        );
        assert!(output.iter().all(Vec::is_empty));
        assert_eq!(scratch.matvec.preparation_count(), 1);
        transforms.project_group_into(
            [&first, &first],
            &[-1.0; 8],
            &geometry,
            output.each_mut(),
            &mut scratch,
        )?;
        assert_eq!(scratch.matvec.preparation_count(), 2);
        assert_eq!(output[0], output[1]);
        assert_eq!(output[0].len(), 1);
        Ok(())
    }

    fn assert_state_bits_eq(expected: &[LayerState], actual: &[LayerState]) {
        assert_eq!(expected.len(), actual.len());
        for (expected, actual) in expected.iter().zip(actual) {
            match (expected, actual) {
                (
                    LayerState::Full { keys, values },
                    LayerState::Full {
                        keys: actual_keys,
                        values: actual_values,
                    },
                ) => {
                    assert!(keys.tokens().eq(actual_keys.tokens()));
                    assert!(values.tokens().eq(actual_values.tokens()));
                }
                (
                    LayerState::Recurrent {
                        convolution,
                        matrix,
                    },
                    LayerState::Recurrent {
                        convolution: actual_convolution,
                        matrix: actual_matrix,
                    },
                ) => {
                    assert!(
                        convolution
                            .iter()
                            .map(|value| value.to_bits())
                            .eq(actual_convolution.iter().map(|value| value.to_bits()))
                    );
                    assert!(
                        matrix
                            .iter()
                            .map(|value| value.to_bits())
                            .eq(actual_matrix.iter().map(|value| value.to_bits()))
                    );
                }
                _ => assert_eq!(
                    std::mem::discriminant(expected),
                    std::mem::discriminant(actual)
                ),
            }
        }
    }

    #[test]
    fn packed_cache_roundtrip_preserves_every_finite_binary16_value() {
        for bits in 0..=u16::MAX {
            if bits & 0x7c00 != 0x7c00 {
                assert_eq!(half_bits(half_to_float(bits)), bits);
            }
        }
    }

    #[test]
    fn packed_cache_dot_preserves_product_rounding_and_wide_accumulation() {
        // Promoting operands before multiplication changes the first result;
        // accumulating products in f32 loses the middle unit in the second.
        assert_eq!(
            dot_f16(&[f32::from_bits(0x3f80_0001), -1.0], &[0x3c01; 2]).to_bits(),
            0x3400_0000
        );
        assert_eq!(
            dot_f16(&[16_777_216.0, 1.0, -16_777_216.0], &[0x3c00; 3]).to_bits(),
            1.0_f32.to_bits()
        );
        let bits = [0x8000, 0x0001, 0x03ff, 0x0400, 0x7bff, 0xfbff];
        let expanded = bits.map(half_to_float);
        let query = [-0.0, 0.25, -0.5, 1.0, 0.5, -0.5];
        assert_eq!(
            dot_f16(&query, &bits).to_bits(),
            dot(&query, &expanded).to_bits()
        );
    }

    #[test]
    fn paged_cache_copy_shares_only_sealed_payload_and_copies_exact_recurrent_bits() -> Result<()> {
        let bits = [0x8000, 0x0001, 0x03ff, 0x0400, 0x7bff, 0xfbff];
        let values = bits.map(half_to_float);
        let mut cache = KvCache::new(bits.len(), 3 * KV_PAGE_TOKENS)?;
        for _ in 0..=KV_PAGE_TOKENS {
            cache.reserve_append()?;
            cache.append(&values);
        }
        assert_eq!(cache.pages.len(), 1);
        assert_eq!(cache.tail, bits);
        let mut states = vec![
            LayerState::Full {
                keys: cache.copy()?,
                values: cache,
            },
            LayerState::Recurrent {
                convolution: vec![-0.0, f32::from_bits(1)],
                matrix: vec![f32::MIN_POSITIVE, f32::MAX],
            },
        ];
        let checkpoint = copy_state(&states)?;
        let expected = copy_state(&states)?;
        assert_state_bits_eq(&states, &checkpoint);
        if let (
            LayerState::Full { keys, values },
            LayerState::Full {
                keys: saved_keys,
                values: saved_values,
            },
        ) = (&mut states[0], &checkpoint[0])
        {
            assert!(Arc::ptr_eq(&keys.pages[0], &saved_keys.pages[0]));
            assert!(Arc::ptr_eq(&values.pages[0], &saved_values.pages[0]));
            assert_ne!(keys.tail.as_ptr(), saved_keys.tail.as_ptr());
            assert_ne!(values.tail.as_ptr(), saved_values.tail.as_ptr());
            // Each snapshot charges all shared payloads conservatively.
            assert_eq!(keys.bytes(), saved_keys.bytes());
            assert!(keys.bytes() >= (KV_PAGE_TOKENS + 1) * bits.len() * size_of::<u16>());
            keys.tail[0] = 0;
            values.tail[1] = 0;
            keys.clear();
        }
        if let LayerState::Recurrent {
            convolution,
            matrix,
        } = &mut states[1]
        {
            convolution[0] = 1.0;
            matrix[0] = 0.0;
        }
        assert_state_bits_eq(&expected, &checkpoint);
        Ok(())
    }

    #[test]
    fn text_imrope_uses_zero_fourth_position_for_unbalanced_sections() -> Result<()> {
        let geometry = Geometry {
            embedding: 12,
            feed_forward: 12,
            heads: 1,
            kv_heads: 1,
            head: 12,
            rope: 12,
            rope_sections: [4, 1, 1, 0],
            rope_base: 10_000.0,
            epsilon: 0.000_001,
            ssm_head: 1,
            ssm_keys: 1,
            ssm_values: 1,
            convolution: 1,
        };
        let mut input = [0.0; 12];
        input[0] = 1.0;
        input[4] = 1.0;
        input[10] = 2.0;
        let mut frequencies = Vec::new();
        rope_frequencies(&geometry, 3, &mut frequencies)?;
        rope(&mut input, &geometry, &frequencies)?;
        // Sector four is outside the height section, so [p,p,p,0] selects 0.
        assert_eq!(input[4].to_bits(), 1.0_f32.to_bits());
        assert_eq!(input[10].to_bits(), 2.0_f32.to_bits());
        assert!((input[0] - 3.0_f32.cos()).abs() < 0.000_001);
        assert!((input[6] - 3.0_f32.sin()).abs() < 0.000_001);
        Ok(())
    }

    #[test]
    fn paged_cache_iteration_keeps_flat_f16_dot_and_weighted_sum_order() -> Result<()> {
        let mut cache = KvCache::new(3, 3 * KV_PAGE_TOKENS)?;
        let mut flat = Vec::new();
        for position in 0..=2 * KV_PAGE_TOKENS {
            let values = match position % 3 {
                0 => [65_504.0, -0.0, 1.0],
                1 => [0.000_061_035_156, 0.5, -65_504.0],
                _ => [-65_504.0, -0.25, 0.000_061_035_156],
            };
            cache.reserve_append()?;
            cache.append(&values);
            flat.extend(values.map(half_bits));
        }
        let query = [1.000_000_1, -0.5, 0.25];
        assert!(
            cache
                .tokens()
                .map(|token| dot_f16(&query, token).to_bits())
                .eq(flat
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|token| dot_f16(&query, token).to_bits()))
        );
        for dimension in 0..3 {
            let weighted = |token: &[u16], index: usize| {
                let probability = [0.5, -0.25, 0.125][index % 3];
                f64::from(probability * half_to_float(token[dimension]))
            };
            let paged = cache
                .tokens()
                .enumerate()
                .map(|(index, token)| weighted(token, index))
                .sum::<f64>();
            let contiguous = flat
                .as_chunks::<3>()
                .0
                .iter()
                .enumerate()
                .map(|(index, token)| weighted(token, index))
                .sum::<f64>();
            assert_eq!(paged.to_bits(), contiguous.to_bits());
        }
        Ok(())
    }

    #[test]
    fn paged_cache_growth_seals_without_payload_copy_and_remains_context_bounded() -> Result<()> {
        let width = 3;
        let context = 2 * KV_PAGE_TOKENS + 5;
        let mut cache = KvCache::new(width, context)?;
        for position in 0..context {
            cache.reserve_append()?;
            let prepared_pointer = cache.tail.as_ptr();
            cache.append(&[1.0; 3]);
            assert_eq!(cache.len, (position + 1) * width);
            assert_eq!(cache.tokens().count(), position + 1);
            assert!(cache.tokens().all(|token| token == [0x3c00; 3]));
            assert!(cache.tail.capacity() <= width * KV_PAGE_TOKENS);
            let allocated = cache
                .pages
                .iter()
                .map(|page| page.capacity())
                .sum::<usize>()
                + cache.tail.capacity();
            assert!(allocated <= width * context);
            if cache.tail.is_empty() {
                let sealed = cache
                    .pages
                    .last()
                    .ok_or_else(|| crate::invalid("missing sealed page"))?;
                assert_eq!(sealed.as_ptr(), prepared_pointer);
            }
        }
        let saved = cache.copy()?;
        assert!(cache.reserve_append().is_err());
        assert!(cache.tokens().eq(saved.tokens()));
        assert_eq!(cache.pages.len(), 3);
        assert_eq!(cache.pages[2].len(), 5 * width);
        assert_eq!(cache.tail, Vec::<u16>::new());
        cache.clear();
        cache.reserve_append()?;
        cache.append(&[-1.0; 3]);
        assert_eq!(cache.tokens().collect::<Vec<_>>(), vec![&[0xbc00; 3][..]]);
        assert!(saved.tokens().all(|token| token == [0x3c00; 3]));
        assert!(KvCache::new(usize::MAX, 2).is_err());
        assert!(KvCache::new(1, MAX_STATE_BYTES).is_err());
        assert!(KvCache::new(0, 1).is_err());
        Ok(())
    }
}
