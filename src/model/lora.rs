//! Explicit rank-one residual-writer and embedding corrections.
//!
//! These operations preserve the supplied adapter's original activation basis.
//! They do not establish that its authoring basis matches a particular model,
//! and do not enable adapters in the private Swift HTTP service.

use std::{collections::BTreeMap, path::Path, sync::Arc};

use bonsai_gguf::MetadataValue;

use super::{Model, Tensor};
use crate::{
    Result, invalid,
    kernels::{
        MatvecBackend, MatvecScratch, lora_a_dot, lora_a_dot_four, lora_a_dot_pair,
        lora_b_scale_and_finish, lora_b_scale_and_finish_tokens, row_into,
    },
};

mod multi_rank;
pub use multi_rank::{MultiRankAdapter, MultiRankPolicy};

/// Exact approved Orca adapter file length.
pub const SWIFT_ADAPTER_BYTES: u64 = 9_682_464;
/// Exact approved Orca adapter file SHA-256.
pub const SWIFT_ADAPTER_SHA256: &str =
    "f1669534803d340a496015f5c45125f3437b4d13ec764f40e34488ce83967f42";
/// Conservative retained header and tensor budget for the approved adapter.
pub const SWIFT_ADAPTER_MEMORY_BYTES: usize = 16 * 1024 * 1024;
const MAX_PAIRS: usize = 513;

/// Explicit identity of one edited, source-only experimental adapter artifact.
///
/// The caller selects the candidate; these fields authenticate its bytes, not
/// its authoring history or quality. Released loaders still admit only Orca.
#[derive(Clone, Copy)]
pub struct EditedAdapter<'a> {
    /// Absolute path of the candidate GGUF, separate from the approved file.
    pub path: &'a Path,
    /// Exact candidate length; the fixed Orca factor inventory is required.
    pub bytes: u64,
    /// Exact lowercase candidate SHA-256.
    pub sha256: &'a str,
}

impl EditedAdapter<'_> {
    pub(crate) fn validate_identity(self) -> Result<()> {
        if !self.path.is_absolute()
            || self.bytes != SWIFT_ADAPTER_BYTES
            || self.sha256 == SWIFT_ADAPTER_SHA256
            || self.sha256.len() != 64
            || !self
                .sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(invalid("edited Swift adapter identity rejected"));
        }
        Ok(())
    }
}

/// Retained factor payload and coefficient counts, excluding allocator overhead.
///
/// B vectors are counted once per shared allocation. Dense serialized linear A
/// tensors are released; embedding lookup retains its original serialized A.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AdapterStorage {
    /// Retained dense or sparse F32 linear A coefficients, excluding indices.
    pub linear_a_bytes: usize,
    /// Retained serialized embedding A bytes.
    pub embedding_a_bytes: usize,
    /// Retained decoded B bytes, counting shared allocations once.
    pub unique_b_bytes: usize,
    /// Retained U16 sparse input indices.
    pub index_bytes: usize,
    /// Sum of the four retained payload byte counts.
    pub total_payload_bytes: usize,
    /// Dense linear A extent before sparse storage conversion.
    pub linear_a_coefficients: usize,
    /// Linear A coefficients retained in dense or sparse storage.
    pub stored_linear_a_coefficients: usize,
    /// Linear sites using the explicitly selected sparse arithmetic profile.
    pub sparse_sites: usize,
}

#[derive(Clone)]
struct SparseA {
    width: usize,
    indices: Vec<u16>,
    values: Vec<f32>,
}

#[derive(Clone)]
struct Factors {
    a_values: Vec<f32>,
    sparse_a: Option<SparseA>,
    b_values: Arc<Vec<f32>>,
}

impl Factors {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::as_conversions,
        reason = "The experimental profile explicitly rounds its finite F64 dot to F32 before B multiplication."
    )]
    fn dot(&self, input: &[f32], scratch: &mut Scratch) -> Result<()> {
        let Some(sparse) = &self.sparse_a else {
            return lora_a_dot(&self.a_values, input, &mut scratch.rank, &scratch.matvec);
        };
        scratch.rank.clear();
        if input.len() != sparse.width || input.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "sparse A dot requires a matching finite activation vector",
            ));
        }
        let sum: f64 = sparse
            .indices
            .iter()
            .zip(&sparse.values)
            .map(|(&index, &value)| f64::from(value) * f64::from(input[usize::from(index)]))
            .sum();
        let rank = sum as f32;
        if !rank.is_finite() {
            return Err(invalid("nonfinite sparse rank-one A product"));
        }
        scratch.rank.try_reserve_exact(1)?;
        scratch.rank.push(rank);
        Ok(())
    }
}

/// An immutable, explicitly admitted rank-one adapter for one base model.
///
/// Only embedding and residual-writing attention/SSM/FFN factors are supported.
/// The numerical implementation follows the two matrix products, scale and add
/// stages separately. Structural admission is not full-model qualification.
pub struct RankOneAdapter {
    identity: String,
    execution_identity: String,
    base_identity: String,
    alpha: f32,
    embedding: Option<(Tensor, Arc<Vec<f32>>)>,
    factors: BTreeMap<String, Factors>,
}

/// Reusable bounded correction storage, independent of base projection scratch.
#[derive(Default)]
pub(crate) struct Scratch {
    rank: Vec<f32>,
    multi_ranks: Vec<f32>,
    correction: Vec<f32>,
    matvec: MatvecScratch,
}

impl Scratch {
    pub(crate) fn new(backend: MatvecBackend, pinned: bool) -> Self {
        Self {
            rank: Vec::new(),
            multi_ranks: Vec::new(),
            correction: Vec::new(),
            matvec: if pinned {
                MatvecScratch::with_pinned_backend(backend)
            } else {
                MatvecScratch::with_backend(backend)
            },
        }
    }

    pub(crate) fn set_row_parallelism(&mut self, parallelism: crate::kernels::RowParallelism) {
        self.matvec.set_row_parallelism(parallelism);
    }

    #[cfg(all(test, feature = "parallel-rows"))]
    pub(crate) const fn row_workers(&self) -> usize {
        self.matvec.row_workers()
    }

    pub(crate) fn apply_dense_b_dot(
        &mut self,
        lora_b: &[f32],
        lora_dot: f32,
        effective_scale: f32,
        output: &mut [f32],
    ) -> Result<()> {
        lora_b_scale_and_finish(
            lora_b,
            &[lora_dot],
            effective_scale,
            output,
            &mut self.correction,
        )
    }
}

impl RankOneAdapter {
    /// Load a bounded edited Orca candidate with explicit sparse A execution.
    ///
    /// The selected base remains strictly pinned. The full 129-site F32 rank-one
    /// inventory and alpha one are required. Only profitable sparse linear A
    /// vectors use U16 indices; their ascending-index F64 dot rounds once to
    /// F32. This experimental numerical profile differs from the dense kernels
    /// and is bound into graph identity. No production loader calls this method.
    ///
    /// # Errors
    /// Rejects invalid identity, changed length, unpinned base, malformed factors,
    /// missing sites, zero linear corrections or no useful sparse representation.
    pub fn load_edited_swift(
        candidate: EditedAdapter<'_>,
        base: &Model,
        representation: super::swift::Representation,
    ) -> Result<Self> {
        candidate.validate_identity()?;
        if base.identity != representation.model_sha256() {
            return Err(invalid("edited Swift adapter base rejected"));
        }
        let mut model = Model::load(
            candidate.path,
            candidate.bytes,
            candidate.sha256,
            SWIFT_ADAPTER_MEMORY_BYTES,
        )?;
        model.share_lora_b();
        let mut adapter = Self::from_model(model, base)?;
        if adapter.factors.len() != 129 || adapter.alpha.to_bits() != 1.0_f32.to_bits() {
            return Err(invalid("edited Swift adapter inventory or alpha mismatch"));
        }
        for layer in 0..64 {
            for family in [
                "ffn_down",
                if layer % 4 == 3 {
                    "attn_output"
                } else {
                    "ssm_out"
                },
            ] {
                if !adapter
                    .factors
                    .contains_key(&format!("blk.{layer}.{family}.weight"))
                {
                    return Err(invalid("edited Swift adapter lacks a required linear site"));
                }
            }
        }
        if adapter.embedding.is_none() {
            return Err(invalid("edited Swift adapter lacks its embedding site"));
        }
        adapter.enable_sparse_a()?;
        Ok(adapter)
    }

    /// Loads the exact approved artifact for the exact original Swift PQ2 model.
    ///
    /// File verification precedes payload publication. Identical immutable B
    /// tensors share storage only after kind, layout and bytes match. This does
    /// not qualify grouped SSM authoring basis, outputs, quality or resources.
    ///
    /// # Errors
    /// Rejects identity/metadata/shape drift, nonfinite payloads, missing/extra
    /// factors, allocation failures and non-PQ2 base identity.
    pub fn load_swift(path: &Path, base: &Model) -> Result<Self> {
        Self::load_swift_representation(path, base, super::swift::Representation::Pq2)
    }

    /// Loads the approved adapter for one explicitly selected current Swift artifact.
    ///
    /// PQ2 and its exact current-weight PTQ1 export use the same factor inventory
    /// and original activation basis. This checks the selected base identity;
    /// callers must obtain its executable weights through the strict Swift loader.
    /// The older published PTQ1 checkpoint is never substituted for this export.
    ///
    /// # Errors
    /// Rejects a base identity outside the selected representation and the same
    /// artifact, inventory, shape and resource errors as [`Self::load_swift`].
    pub fn load_swift_representation(
        path: &Path,
        base: &Model,
        representation: super::swift::Representation,
    ) -> Result<Self> {
        if base.identity != representation.model_sha256() {
            return Err(invalid(
                "approved Swift adapter requires the exact selected current-weight base",
            ));
        }
        let mut model = Model::load(
            path,
            SWIFT_ADAPTER_BYTES,
            SWIFT_ADAPTER_SHA256,
            SWIFT_ADAPTER_MEMORY_BYTES,
        )?;
        model.share_lora_b();
        let adapter = Self::from_model(model, base)?;
        if adapter.factors.len() != 129 || adapter.alpha.to_bits() != 1.0_f32.to_bits() {
            return Err(invalid(
                "approved Swift adapter inventory or alpha mismatch",
            ));
        }
        Ok(adapter)
    }

    /// Validates already loaded factors against their exact base tensor shapes.
    ///
    /// Callers own file authenticity. The returned identity binds the supplied
    /// model identities; it does not attest that a manually constructed `Model`
    /// was verified. No permutation or transform is applied to `LoRA` A/B.
    ///
    /// # Errors
    /// Rejects other architectures/types, non-F32 or non-rank-one factors,
    /// unsupported/absent targets, incomplete pairs, nonfinite values, invalid
    /// alpha, more than 513 pairs, or empty identities.
    pub fn from_model(model: Model, base: &Model) -> Result<Self> {
        for (metadata, key, expected) in [
            (&model.metadata, "general.architecture", "qwen35"),
            (&base.metadata, "general.architecture", "qwen35"),
            (&model.metadata, "general.type", "adapter"),
            (&model.metadata, "adapter.type", "lora"),
        ] {
            if metadata.get(key).and_then(MetadataValue::as_str) != Some(expected) {
                return Err(invalid("unsupported rank-one adapter metadata"));
            }
        }
        let Some(MetadataValue::F32(alpha)) = model.metadata.get("adapter.lora_alpha") else {
            return Err(invalid("rank-one adapter requires an F32 alpha"));
        };
        if !alpha.is_finite()
            || *alpha <= 0.0
            || model.identity.is_empty()
            || base.identity.is_empty()
            || model.tensors.is_empty()
            || model.tensors.len() > MAX_PAIRS * 2
            || !model.tensors.len().is_multiple_of(2)
        {
            return Err(invalid(
                "invalid rank-one adapter alpha, identity or inventory",
            ));
        }
        let mut factors = BTreeMap::new();
        let mut decoded_b: Vec<(&Tensor, Arc<Vec<f32>>)> = Vec::new();
        let mut embedding = None;
        for (name, a) in &model.tensors {
            if let Some(target) = name.strip_suffix(".lora_a") {
                let base_tensor = base
                    .tensors
                    .get(target)
                    .filter(|_| supported_target(target))
                    .ok_or_else(|| invalid("unsupported rank-one adapter target"))?;
                let b = model
                    .tensors
                    .get(&format!("{target}.lora_b"))
                    .ok_or_else(|| invalid("rank-one adapter B factor missing"))?;
                let [input, output] = base_tensor.dimensions.as_slice() else {
                    return Err(invalid("rank-one adapter target must be a matrix"));
                };
                let a_shape = if target == "token_embd.weight" {
                    [1, *output]
                } else {
                    [*input, 1]
                };
                validate_factor(a, name, &a_shape)?;
                let a_values = if target == "token_embd.weight" {
                    Vec::new()
                } else {
                    decode_factor(a)?
                };
                let b_shape = [
                    1,
                    if target == "token_embd.weight" {
                        *input
                    } else {
                        *output
                    },
                ];
                validate_factor(b, &format!("{target}.lora_b"), &b_shape)?;
                let b_values = decode_shared_factor(b, &mut decoded_b)?;
                if target == "token_embd.weight" {
                    // Embedding row lookup alone needs serialized A. Keep its
                    // already shared B nearby to avoid another target lookup.
                    embedding = Some((a.clone(), Arc::clone(&b_values)));
                }
                factors.insert(
                    target.to_owned(),
                    Factors {
                        a_values,
                        sparse_a: None,
                        b_values,
                    },
                );
            } else if let Some(target) = name.strip_suffix(".lora_b") {
                if !model.tensors.contains_key(&format!("{target}.lora_a")) {
                    return Err(invalid("rank-one adapter A factor missing"));
                }
            } else {
                return Err(invalid("unsupported rank-one adapter tensor name"));
            }
        }
        if factors.len() * 2 != model.tensors.len() {
            return Err(invalid("incomplete rank-one adapter inventory"));
        }
        let execution_identity = adapter_execution_identity(
            &model.identity,
            &base.identity,
            *alpha,
            embedding.as_ref().map(|(tensor, _)| tensor),
            &factors,
        )?;
        Ok(Self {
            execution_identity,
            identity: model.identity,
            base_identity: base.identity.clone(),
            alpha: *alpha,
            // Linear sites consume only decoded A values. Retaining their raw
            // tensors would keep an unused second complete factor payload.
            embedding,
            factors,
        })
    }

    /// Identity of the supplied adapter model or its verified GGUF file.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    pub(crate) fn execution_identity(&self) -> &str {
        &self.execution_identity
    }

    /// Count retained factor payload, without process or peak-memory claims.
    #[must_use]
    pub fn storage(&self) -> AdapterStorage {
        let mut storage = AdapterStorage::default();
        let mut seen_b = Vec::new();
        for factors in self.factors.values() {
            if let Some(sparse) = &factors.sparse_a {
                storage.linear_a_bytes += sparse.values.capacity() * size_of::<f32>();
                storage.index_bytes += sparse.indices.capacity() * size_of::<u16>();
                storage.linear_a_coefficients += sparse.width;
                storage.stored_linear_a_coefficients += sparse.values.len();
                storage.sparse_sites += 1;
            } else {
                storage.linear_a_bytes += factors.a_values.capacity() * size_of::<f32>();
                storage.linear_a_coefficients += factors.a_values.len();
                storage.stored_linear_a_coefficients += factors.a_values.len();
            }
            if !seen_b
                .iter()
                .any(|other| Arc::ptr_eq(other, &factors.b_values))
            {
                storage.unique_b_bytes += factors.b_values.capacity() * size_of::<f32>();
                seen_b.push(Arc::clone(&factors.b_values));
            }
        }
        storage.embedding_a_bytes = self.embedding.as_ref().map_or(0, |(a, _)| a.data.len());
        storage.total_payload_bytes = storage.linear_a_bytes
            + storage.index_bytes
            + storage.embedding_a_bytes
            + storage.unique_b_bytes;
        storage
    }

    pub(crate) fn enable_sparse_a(&mut self) -> Result<()> {
        let mut sparse_count = 0;
        for (target, factors) in &mut self.factors {
            if target == "token_embd.weight" {
                continue;
            }
            let nonzero = factors
                .a_values
                .iter()
                .filter(|value| **value != 0.0)
                .count();
            if nonzero == 0 {
                return Err(invalid(
                    "edited adapter must retain every linear correction",
                ));
            }
            if factors.a_values.len() > usize::from(u16::MAX) + 1
                || nonzero * (size_of::<f32>() + size_of::<u16>())
                    >= factors.a_values.len() * size_of::<f32>()
            {
                continue;
            }
            let mut sparse = SparseA {
                width: factors.a_values.len(),
                indices: Vec::new(),
                values: Vec::new(),
            };
            sparse.indices.try_reserve_exact(nonzero)?;
            sparse.values.try_reserve_exact(nonzero)?;
            for (index, &value) in factors
                .a_values
                .iter()
                .enumerate()
                .filter(|(_, value)| **value != 0.0)
            {
                sparse.indices.push(u16::try_from(index)?);
                sparse.values.push(value);
            }
            factors.a_values = Vec::new();
            factors.sparse_a = Some(sparse);
            sparse_count += 1;
        }
        if sparse_count == 0 {
            return Err(invalid(
                "edited adapter has no profitable sparse linear A vectors",
            ));
        }
        self.execution_identity
            .push_str(":edited-sparse-a-u16-f64-v1");
        Ok(())
    }

    /// Number of complete admitted factor pairs.
    #[must_use]
    pub const fn target_count(&self) -> usize {
        self.factors.len()
    }

    pub(crate) fn base_identity(&self) -> &str {
        &self.base_identity
    }

    pub(crate) fn targets(&self) -> impl Iterator<Item = &str> {
        self.factors.keys().map(String::as_str)
    }

    pub(crate) fn compact_ffn_down_layers(&self, layers: &[(usize, &[usize])]) -> Result<Self> {
        const GROUPS: usize = 17;
        let mut factors = self.factors.clone();
        for &(layer, omitted_groups) in layers {
            if omitted_groups.is_empty()
                || omitted_groups.len() >= GROUPS
                || omitted_groups.iter().any(|&group| group >= GROUPS)
                || omitted_groups.windows(2).any(|pair| pair[0] >= pair[1])
            {
                return Err(invalid(
                    "FFN compaction requires 1..=16 sorted distinct omitted groups in 0..17",
                ));
            }
            if factors.contains_key(&format!("blk.{layer}.ffn_gate.weight"))
                || factors.contains_key(&format!("blk.{layer}.ffn_up.weight"))
            {
                return Err(invalid(
                    "rank-one gate/up adapters are unsupported on compacted FFN layers",
                ));
            }
            let target = format!("blk.{layer}.ffn_down.weight");
            if let Some(entry) = factors.get_mut(&target) {
                compact_rank_one_ffn_down_entry(entry, omitted_groups)?;
            }
        }
        let mut execution_identity = adapter_execution_identity(
            &self.identity,
            &self.base_identity,
            self.alpha,
            self.embedding.as_ref().map(|(tensor, _)| tensor),
            &factors,
        )?;
        if factors.values().any(|factor| factor.sparse_a.is_some()) {
            let mut hasher = bonsai_native_hash::Hasher::new()?;
            hasher.update(b"prism-rank-one-sparse-compacted-v1\0")?;
            for (target, factor) in &factors {
                if let Some(sparse) = &factor.sparse_a {
                    hash_len_bytes(&mut hasher, target.as_bytes())?;
                    hasher.update(&u64::try_from(sparse.width)?.to_le_bytes())?;
                    hasher.update(&u64::try_from(sparse.indices.len())?.to_le_bytes())?;
                    for (&index, &value) in sparse.indices.iter().zip(&sparse.values) {
                        hasher.update(&index.to_le_bytes())?;
                        hasher.update(&value.to_bits().to_le_bytes())?;
                    }
                }
            }
            execution_identity.push_str(":edited-sparse-a-u16-f64-v1:compacted=");
            execution_identity.push_str(&hex::encode(hasher.finish()?));
        }
        Ok(Self {
            identity: self.identity.clone(),
            execution_identity,
            base_identity: self.base_identity.clone(),
            alpha: self.alpha,
            embedding: self.embedding.clone(),
            factors,
        })
    }

    pub(crate) fn validate_scale(scale: f32) -> Result<()> {
        if scale.is_finite() && (0.0..=2.0).contains(&scale) {
            Ok(())
        } else {
            Err(invalid("rank-one adapter scale must be finite and in 0..2"))
        }
    }

    /// Whether a linear correction can do work for this exact target and scale.
    /// Keep this identical to the no-op admission in `linear_tokens`.
    pub(crate) fn affects_linear(&self, target: &str, scale: f32) -> bool {
        scale != 0.0 && possible_factor_target(target) && self.factors.contains_key(target)
    }

    /// Returns `(a_values, effective_scale_alpha, b_values)` for an active
    /// dense rank-one target when `scale` is nonzero and `sparse_a` is absent.
    pub(crate) fn dense_rank_one_factor(
        &self,
        name: &str,
        scale: Option<f32>,
    ) -> Option<(&[f32], f32, &[f32])> {
        let scale = scale.filter(|&value| value.is_finite() && value != 0.0)?;
        if !possible_factor_target(name) {
            return None;
        }
        let factors = self.factors.get(name)?;
        if factors.sparse_a.is_some() {
            return None;
        }
        let effective_scale = scale * self.alpha;
        if !effective_scale.is_finite() || effective_scale == 0.0 {
            return None;
        }
        Some((
            factors.a_values.as_slice(),
            effective_scale,
            factors.b_values.as_slice(),
        ))
    }

    pub(crate) fn linear(
        &self,
        target: &str,
        input: &[f32],
        scale: f32,
        output: &mut [f32],
        scratch: &mut Scratch,
    ) -> Result<()> {
        if scale == 0.0 || !possible_factor_target(target) {
            return Ok(());
        }
        let Some(factors) = self.factors.get(target) else {
            return Ok(());
        };
        factors.dot(input, scratch)?;
        self.scale_and_finish(&factors.b_values, scale, output, scratch)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Sparse execution precedes the existing four-, two- and single-token dense paths without changing their arithmetic."
    )]
    pub(crate) fn linear_tokens(
        &self,
        target: &str,
        inputs: &[Vec<f32>],
        scale: f32,
        outputs: &mut [Vec<f32>],
        scratch: &mut Scratch,
    ) -> Result<()> {
        if scale == 0.0 || !possible_factor_target(target) {
            return Ok(());
        }
        let Some(factors) = self.factors.get(target) else {
            return Ok(());
        };
        let count = inputs.len().min(outputs.len());
        if factors.sparse_a.is_some() {
            for (input, output) in inputs[..count].iter().zip(&mut outputs[..count]) {
                factors.dot(input, scratch)?;
                self.scale_and_finish(&factors.b_values, scale, output, scratch)?;
            }
            return Ok(());
        }
        let mut idx = 0;
        while idx + 4 <= count {
            if let Some(ranks) = lora_a_dot_four(
                &factors.a_values,
                [
                    &inputs[idx],
                    &inputs[idx + 1],
                    &inputs[idx + 2],
                    &inputs[idx + 3],
                ],
                &scratch.matvec,
            ) {
                scratch.rank.clear();
                scratch.rank.try_reserve_exact(1)?;
                scratch.rank.push(ranks[3]);
                let (first_pair, second_pair) = outputs[idx..idx + 4].split_at_mut(2);
                let (out0, out1) = first_pair.split_at_mut(1);
                let (out2, out3) = second_pair.split_at_mut(1);
                if !lora_b_scale_and_finish_tokens(
                    &factors.b_values,
                    ranks,
                    scale * self.alpha,
                    [&mut out0[0], &mut out1[0], &mut out2[0], &mut out3[0]],
                    &mut scratch.correction,
                ) {
                    for (tok_offset, &r) in ranks.iter().enumerate() {
                        scratch.rank[0] = r;
                        self.scale_and_finish(
                            &factors.b_values,
                            scale,
                            &mut outputs[idx + tok_offset],
                            scratch,
                        )?;
                    }
                }
                idx += 4;
                continue;
            }
            break;
        }
        while idx + 2 <= count {
            if let Some((r0, r1)) = lora_a_dot_pair(
                &factors.a_values,
                &inputs[idx],
                &inputs[idx + 1],
                &scratch.matvec,
            ) {
                scratch.rank.clear();
                scratch.rank.try_reserve_exact(1)?;
                scratch.rank.push(r1);
                let (out0, out1) = outputs[idx..idx + 2].split_at_mut(1);
                if lora_b_scale_and_finish_tokens(
                    &factors.b_values,
                    [r0, r1],
                    scale * self.alpha,
                    [&mut out0[0], &mut out1[0]],
                    &mut scratch.correction,
                ) {
                    idx += 2;
                    continue;
                }
                scratch.rank[0] = r0;
                self.scale_and_finish(&factors.b_values, scale, &mut outputs[idx], scratch)?;
                scratch.rank[0] = r1;
            } else {
                lora_a_dot(
                    &factors.a_values,
                    &inputs[idx],
                    &mut scratch.rank,
                    &scratch.matvec,
                )?;
                self.scale_and_finish(&factors.b_values, scale, &mut outputs[idx], scratch)?;
                lora_a_dot(
                    &factors.a_values,
                    &inputs[idx + 1],
                    &mut scratch.rank,
                    &scratch.matvec,
                )?;
            }
            self.scale_and_finish(&factors.b_values, scale, &mut outputs[idx + 1], scratch)?;
            idx += 2;
        }
        while idx < count {
            lora_a_dot(
                &factors.a_values,
                &inputs[idx],
                &mut scratch.rank,
                &scratch.matvec,
            )?;
            self.scale_and_finish(&factors.b_values, scale, &mut outputs[idx], scratch)?;
            idx += 1;
        }
        Ok(())
    }

    pub(crate) fn embedding(
        &self,
        token: usize,
        scale: f32,
        output: &mut [f32],
        scratch: &mut Scratch,
    ) -> Result<()> {
        let Some((a, b_values)) = self.embedding.as_ref().filter(|_| scale != 0.0) else {
            return Ok(());
        };
        row_into(a, token, &mut scratch.rank)?;
        self.scale_and_finish(b_values, scale, output, scratch)
    }

    fn scale_and_finish(
        &self,
        b_values: &[f32],
        scale: f32,
        output: &mut [f32],
        scratch: &mut Scratch,
    ) -> Result<()> {
        lora_b_scale_and_finish(
            b_values,
            &scratch.rank,
            scale * self.alpha,
            output,
            &mut scratch.correction,
        )
    }
}

fn compact_rank_one_ffn_down_entry(entry: &mut Factors, omitted_groups: &[usize]) -> Result<()> {
    const GROUP_WIDTH: usize = 1024;
    const GROUPS: usize = 17;
    const FULL_WIDTH: usize = GROUP_WIDTH * GROUPS;
    let retained = GROUPS - omitted_groups.len();
    let new_width = retained * GROUP_WIDTH;
    if let Some(sparse) = &mut entry.sparse_a {
        if sparse.width != FULL_WIDTH {
            return Err(invalid(
                "rank-one FFN down sparse A width does not match Swift intermediate width",
            ));
        }
        let mut remap = [None; GROUPS];
        let mut next_group = 0_usize;
        for (group, slot) in remap.iter_mut().enumerate() {
            if !omitted_groups.contains(&group) {
                *slot = Some(next_group);
                next_group += 1;
            }
        }
        let mut new_indices = Vec::new();
        let mut new_values = Vec::new();
        for (&index, &value) in sparse.indices.iter().zip(&sparse.values) {
            let coord = usize::from(index);
            let group = coord / GROUP_WIDTH;
            let offset = coord % GROUP_WIDTH;
            if let Some(Some(mapped_group)) = remap.get(group).copied() {
                new_indices.try_reserve(1)?;
                new_values.try_reserve(1)?;
                new_indices.push(u16::try_from(mapped_group * GROUP_WIDTH + offset)?);
                new_values.push(value);
            }
        }
        sparse.width = new_width;
        sparse.indices = new_indices;
        sparse.values = new_values;
    } else {
        if entry.a_values.len() != FULL_WIDTH {
            return Err(invalid(
                "rank-one FFN down A width does not match Swift intermediate width",
            ));
        }
        let mut new_a = Vec::new();
        new_a.try_reserve_exact(new_width)?;
        for group in 0..GROUPS {
            if !omitted_groups.contains(&group) {
                new_a.extend_from_slice(
                    &entry.a_values[group * GROUP_WIDTH..(group + 1) * GROUP_WIDTH],
                );
            }
        }
        entry.a_values = new_a;
    }
    Ok(())
}

#[inline]
fn possible_factor_target(name: &str) -> bool {
    name.ends_with(".ffn_down.weight")
        || name.ends_with(".ssm_out.weight")
        || name.ends_with(".attn_output.weight")
        || name == "token_embd.weight"
}

fn supported_target(name: &str) -> bool {
    if name == "token_embd.weight" {
        return true;
    }
    let Some((layer, suffix)) = name
        .strip_prefix("blk.")
        .and_then(|name| name.split_once('.'))
    else {
        return false;
    };
    !layer.is_empty()
        && layer.bytes().all(|byte| byte.is_ascii_digit())
        && matches!(
            suffix,
            "attn_output.weight" | "ssm_out.weight" | "ffn_down.weight"
        )
}

fn validate_factor(tensor: &Tensor, name: &str, shape: &[usize]) -> Result<()> {
    let bytes = shape
        .iter()
        .try_fold(4_usize, |bytes, dimension| bytes.checked_mul(*dimension));
    if tensor.name != name
        || tensor.kind != 0
        || tensor.dimensions != shape
        || shape.contains(&0)
        || bytes != Some(tensor.data.len())
        || tensor
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .any(|bytes| !f32::from_le_bytes(*bytes).is_finite())
    {
        return Err(invalid("invalid rank-one adapter factor"));
    }
    Ok(())
}

fn decode_factor(tensor: &Tensor) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    values.try_reserve_exact(tensor.data.len() / 4)?;
    for chunk in tensor.data.as_chunks::<4>().0 {
        values.push(f32::from_le_bytes(*chunk));
    }
    Ok(values)
}

fn decode_shared_factor<'a>(
    tensor: &'a Tensor,
    decoded: &mut Vec<(&'a Tensor, Arc<Vec<f32>>)>,
) -> Result<Arc<Vec<f32>>> {
    // The approved loader already proves identical B storage. Keep that sharing
    // after decoding without float equality, which collapses signed-zero bits.
    if let Some((_, values)) = decoded.iter().find(|(other, _)| {
        other.kind == tensor.kind
            && other.dimensions == tensor.dimensions
            && crate::model::TensorBytes::ptr_eq(&other.data, &tensor.data)
    }) {
        return Ok(Arc::clone(values));
    }
    let values = Arc::new(decode_factor(tensor)?);
    decoded.try_reserve_exact(1)?;
    decoded.push((tensor, Arc::clone(&values)));
    Ok(values)
}

fn hash_len_bytes(hasher: &mut bonsai_native_hash::Hasher, bytes: &[u8]) -> Result<()> {
    hasher.update(&u64::try_from(bytes.len())?.to_le_bytes())?;
    hasher.update(bytes)?;
    Ok(())
}

fn adapter_execution_identity(
    model_identity: &str,
    base_identity: &str,
    alpha: f32,
    embedding: Option<&Tensor>,
    factors: &BTreeMap<String, Factors>,
) -> Result<String> {
    let mut hasher = bonsai_native_hash::Hasher::new()?;
    hasher.update(b"prism-rank-one-adapter-v1\0")?;
    hash_len_bytes(&mut hasher, base_identity.as_bytes())?;
    hasher.update(&alpha.to_bits().to_le_bytes())?;
    if let Some(embed) = embedding {
        hasher.update(&[1_u8])?;
        hash_len_bytes(&mut hasher, embed.name.as_bytes())?;
        hasher.update(&embed.kind.to_le_bytes())?;
        hasher.update(&u64::try_from(embed.dimensions.len())?.to_le_bytes())?;
        for &dim in &embed.dimensions {
            hasher.update(&u64::try_from(dim)?.to_le_bytes())?;
        }
        hash_len_bytes(&mut hasher, &embed.data)?;
    } else {
        hasher.update(&[0_u8])?;
    }
    hasher.update(&u64::try_from(factors.len())?.to_le_bytes())?;
    for (target, factor) in factors {
        hash_len_bytes(&mut hasher, target.as_bytes())?;
        hasher.update(&u64::try_from(factor.a_values.len())?.to_le_bytes())?;
        for value in &factor.a_values {
            hasher.update(&value.to_bits().to_le_bytes())?;
        }
        hasher.update(&u64::try_from(factor.b_values.len())?.to_le_bytes())?;
        for value in factor.b_values.as_slice() {
            hasher.update(&value.to_bits().to_le_bytes())?;
        }
    }
    let digest = hasher.finish()?;
    let hex = hex::encode(digest);
    Ok(format!("{model_identity}:sha256={hex}"))
}

#[cfg(test)]
mod storage_tests;

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, sync::Arc};

    use bonsai_gguf::MetadataValue;

    use super::{Model, RankOneAdapter, Result, Scratch, Tensor};
    use crate::kernels::MatvecBackend;
    use crate::model::swift::Representation;

    fn tensor(name: &str, values: &[f32]) -> Tensor {
        Tensor {
            name: name.into(),
            dimensions: vec![1, values.len()],
            kind: 0,
            data: Arc::<[u8]>::from(
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .into(),
        }
    }

    fn adapter(a: f32, b: f32) -> Result<RankOneAdapter> {
        let target = "blk.0.ffn_down.weight";
        let base = Model {
            identity: "base".into(),
            metadata: BTreeMap::from([(
                "general.architecture".into(),
                MetadataValue::from("qwen35"),
            )]),
            tensors: BTreeMap::from([(target.into(), tensor(target, &[0.0]))]),
        };
        let factors = Model {
            identity: "adapter".into(),
            metadata: BTreeMap::from([
                ("general.architecture".into(), MetadataValue::from("qwen35")),
                ("general.type".into(), MetadataValue::from("adapter")),
                ("adapter.type".into(), MetadataValue::from("lora")),
                ("adapter.lora_alpha".into(), MetadataValue::from(1.0_f32)),
            ]),
            tensors: BTreeMap::from([
                (
                    format!("{target}.lora_a"),
                    tensor(&format!("{target}.lora_a"), &[a]),
                ),
                (
                    format!("{target}.lora_b"),
                    tensor(&format!("{target}.lora_b"), &[b]),
                ),
            ]),
        };
        RankOneAdapter::from_model(factors, &base)
    }

    #[test]
    fn x1_cpu_sibling_adapter_eligibility_matches_exact_noop_contract() -> Result<()> {
        let adapter = adapter(0.0, 0.0)?;
        let target = "blk.0.ffn_down.weight";
        // Numerically zero factors still execute their correction and may
        // change signed zero or fail; only actual no-op admission permits grouping.
        assert!(adapter.affects_linear(target, 1.3));
        for scale in [0.0, -0.0] {
            assert!(!adapter.affects_linear(target, scale));
        }
        let mut scratch = Scratch::default();
        for name in [
            "blk.0.ssm_in.weight",
            "blk.0.ffn_up.weight",
            "blk.1.ffn_down.weight",
        ] {
            assert!(!adapter.affects_linear(name, 1.3));
            let mut output = vec![vec![f32::from_bits(0x8000_0000)]];
            adapter.linear_tokens(name, &[vec![f32::NAN]], 1.3, &mut output, &mut scratch)?;
            assert_eq!(output[0][0].to_bits(), 0x8000_0000);
        }
        Ok(())
    }

    #[test]
    fn rank_one_selected_current_weight_base_keeps_default_and_adapter_verification_strict()
    -> Result<()> {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "PrismAdapterIdentity-{}-{nonce}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, b"unapproved adapter bytes")?;
        let mut base = Model {
            identity: String::new(),
            metadata: BTreeMap::new(),
            tensors: BTreeMap::new(),
        };
        for representation in [Representation::Pq2, Representation::CurrentRepackedPtq1] {
            base.identity = representation.model_sha256().into();
            let error = RankOneAdapter::load_swift_representation(&path, &base, representation)
                .err()
                .ok_or("unapproved adapter was admitted")?;
            assert!(error.to_string().contains("artifact size mismatch"));
            let other = if representation == Representation::Pq2 {
                Representation::CurrentRepackedPtq1
            } else {
                Representation::Pq2
            };
            let error = RankOneAdapter::load_swift_representation(&path, &base, other)
                .err()
                .ok_or("mismatched base representation was admitted")?;
            assert!(error.to_string().contains("selected current-weight base"));
        }
        assert!(RankOneAdapter::load_swift(&path, &base).is_err());
        base.identity = "older-published-PTQ1-checkpoint".into();
        assert!(
            RankOneAdapter::load_swift_representation(
                &path,
                &base,
                Representation::CurrentRepackedPtq1
            )
            .is_err()
        );
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn rank_one_products_round_before_scale_and_base_addition() -> Result<()> {
        let adapter = adapter(4097.0, 1.0)?;
        let mut scratch = Scratch::new(MatvecBackend::Scalar, true);
        let mut output = [-16_785_408.0];
        adapter.linear(
            "blk.0.ffn_down.weight",
            &[4097.0],
            1.0,
            &mut output,
            &mut scratch,
        )?;
        // The unrounded product is 16_785_409; fusion would incorrectly give 1.
        assert_eq!(output, [0.0]);
        assert_eq!(scratch.rank.len(), 1);
        assert_eq!(scratch.correction.len(), 1);
        Ok(())
    }

    #[test]
    fn rank_one_zero_skips_nonfinite_branch_and_failures_preserve_output() -> Result<()> {
        let adapter = adapter(f32::MAX, 1.0)?;
        let mut scratch = Scratch::new(MatvecBackend::Scalar, false);
        for scale in [0.0, -0.0] {
            let mut output = [-0.0];
            adapter.linear(
                "blk.0.ffn_down.weight",
                &[f32::MAX],
                scale,
                &mut output,
                &mut scratch,
            )?;
            assert_eq!(output[0].to_bits(), (-0.0_f32).to_bits());
            assert_eq!(scratch.rank, Vec::<f32>::new());
            assert_eq!(scratch.correction, Vec::<f32>::new());
        }
        let mut output = [3.0];
        assert!(
            adapter
                .linear(
                    "blk.0.ffn_down.weight",
                    &[1.0],
                    2.0,
                    &mut output,
                    &mut scratch
                )
                .is_err()
        );
        assert_eq!(output, [3.0]);
        Ok(())
    }

    #[test]
    fn rank_one_linear_tokens_matches_single_token_bits() -> Result<()> {
        let target = "blk.0.ffn_down.weight";
        let a_vals: Vec<f32> = (0..32_usize)
            .map(|i| f32::from(i8::try_from((i * 5) % 13).unwrap_or(0) - 6) * 0.125)
            .collect();
        let b_vals: Vec<f32> = (0..23_usize)
            .map(|i| f32::from(i8::try_from((i * 3) % 11).unwrap_or(0) - 5) * 0.25)
            .collect();
        let base_tensor = Tensor {
            name: target.into(),
            dimensions: vec![a_vals.len(), b_vals.len()],
            kind: 0,
            data: Arc::<[u8]>::from(vec![0_u8; a_vals.len() * b_vals.len() * 4]).into(),
        };
        let a_tensor = Tensor {
            name: format!("{target}.lora_a"),
            dimensions: vec![a_vals.len(), 1],
            kind: 0,
            data: Arc::<[u8]>::from(
                a_vals
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .into(),
        };
        let b_tensor = tensor(&format!("{target}.lora_b"), &b_vals);
        let base = Model {
            identity: "base".into(),
            metadata: BTreeMap::from([(
                "general.architecture".into(),
                MetadataValue::from("qwen35"),
            )]),
            tensors: BTreeMap::from([(target.into(), base_tensor)]),
        };
        let factors = Model {
            identity: "adapter".into(),
            metadata: BTreeMap::from([
                ("general.architecture".into(), MetadataValue::from("qwen35")),
                ("general.type".into(), MetadataValue::from("adapter")),
                ("adapter.type".into(), MetadataValue::from("lora")),
                ("adapter.lora_alpha".into(), MetadataValue::from(1.0_f32)),
            ]),
            tensors: BTreeMap::from([
                (format!("{target}.lora_a"), a_tensor),
                (format!("{target}.lora_b"), b_tensor),
            ]),
        };
        let adapter = RankOneAdapter::from_model(factors, &base)?;
        for token_count in 1..=7 {
            let inputs: Vec<Vec<f32>> = (0..token_count)
                .map(|tok| {
                    (0..32_usize)
                        .map(|i| {
                            f32::from(i8::try_from((i + tok * 7) % 17).unwrap_or(0) - 8) * 0.0625
                        })
                        .collect()
                })
                .collect();
            for pinned in [false, true] {
                let mut single_outputs: Vec<Vec<f32>> = (0..token_count)
                    .map(|tok| {
                        (0..b_vals.len())
                            .map(|i| f32::from(u8::try_from(i + tok).unwrap_or(0)))
                            .collect()
                    })
                    .collect();
                let mut batch_outputs = single_outputs.clone();
                let mut scratch_single = Scratch::new(MatvecBackend::ExperimentalNeon, pinned);
                let mut scratch_batch = Scratch::new(MatvecBackend::ExperimentalNeon, pinned);
                for (inp, out) in inputs.iter().zip(&mut single_outputs) {
                    adapter.linear(target, inp, 0.75, out, &mut scratch_single)?;
                }
                adapter.linear_tokens(
                    target,
                    &inputs,
                    0.75,
                    &mut batch_outputs,
                    &mut scratch_batch,
                )?;
                for (actual, expected) in batch_outputs.iter().zip(&single_outputs) {
                    let actual_bits: Vec<u32> = actual.iter().copied().map(f32::to_bits).collect();
                    let expected_bits: Vec<u32> =
                        expected.iter().copied().map(f32::to_bits).collect();
                    assert_eq!(actual_bits, expected_bits);
                }
                let to_bits = |s: &[f32]| s.iter().copied().map(f32::to_bits).collect::<Vec<_>>();
                assert_eq!(to_bits(&scratch_batch.rank), to_bits(&scratch_single.rank));
                assert_eq!(
                    to_bits(&scratch_batch.correction),
                    to_bits(&scratch_single.correction)
                );
            }
        }
        Ok(())
    }

    #[test]
    fn rank_one_execution_identity_binds_weights_alpha_signed_zero_embedding_and_base() -> Result<()>
    {
        let build = |base_id: &str,
                     alpha: f32,
                     a_val: f32,
                     b_val: f32,
                     emb_a_val: Option<f32>|
         -> Result<RankOneAdapter> {
            let linear = "blk.0.ffn_down.weight";
            let emb = "token_embd.weight";
            let mut base = Model {
                identity: base_id.into(),
                metadata: BTreeMap::from([(
                    "general.architecture".into(),
                    MetadataValue::from("qwen35"),
                )]),
                tensors: BTreeMap::from([(
                    linear.into(),
                    Tensor {
                        name: linear.into(),
                        dimensions: vec![1, 1],
                        kind: 0,
                        data: Arc::<[u8]>::from(vec![0_u8; 4]).into(),
                    },
                )]),
            };
            let mut tensors = BTreeMap::from([
                (
                    format!("{linear}.lora_a"),
                    tensor(&format!("{linear}.lora_a"), &[a_val]),
                ),
                (
                    format!("{linear}.lora_b"),
                    tensor(&format!("{linear}.lora_b"), &[b_val]),
                ),
            ]);
            if let Some(ev) = emb_a_val {
                base.tensors.insert(
                    emb.into(),
                    Tensor {
                        name: emb.into(),
                        dimensions: vec![1, 1],
                        kind: 0,
                        data: Arc::<[u8]>::from(vec![0_u8; 4]).into(),
                    },
                );
                tensors.insert(
                    format!("{emb}.lora_a"),
                    tensor(&format!("{emb}.lora_a"), &[ev]),
                );
                tensors.insert(
                    format!("{emb}.lora_b"),
                    tensor(&format!("{emb}.lora_b"), &[1.0]),
                );
            }
            let model = Model {
                identity: "reused-general-name".into(),
                metadata: BTreeMap::from([
                    ("general.architecture".into(), MetadataValue::from("qwen35")),
                    ("general.type".into(), MetadataValue::from("adapter")),
                    ("adapter.type".into(), MetadataValue::from("lora")),
                    ("adapter.lora_alpha".into(), MetadataValue::from(alpha)),
                ]),
                tensors,
            };
            RankOneAdapter::from_model(model, &base)
        };
        let baseline = build("base-a", 1.0, 0.0, 2.0, Some(0.5))?;
        let repeated = build("base-a", 1.0, 0.0, 2.0, Some(0.5))?;
        assert_eq!(baseline.identity(), "reused-general-name");
        assert_eq!(baseline.execution_identity(), repeated.execution_identity());

        let diff_signed_zero = build("base-a", 1.0, -0.0, 2.0, Some(0.5))?;
        let diff_weights = build("base-a", 1.0, 0.0, 3.0, Some(0.5))?;
        let diff_alpha = build("base-a", 2.0, 0.0, 2.0, Some(0.5))?;
        let diff_embed = build("base-a", 1.0, 0.0, 2.0, Some(-0.5))?;
        let diff_base = build("base-b", 1.0, 0.0, 2.0, Some(0.5))?;
        for other in [
            diff_signed_zero,
            diff_weights,
            diff_alpha,
            diff_embed,
            diff_base,
        ] {
            assert_eq!(other.identity(), "reused-general-name");
            assert_ne!(other.execution_identity(), baseline.execution_identity());
        }
        Ok(())
    }

    #[test]
    fn dense_rank_one_factor_gates_scale_target_and_sparse_mode() -> Result<()> {
        let linear = "blk.0.ffn_down.weight";
        let base = Model {
            identity: "dense-factor-base".into(),
            metadata: BTreeMap::from([(
                "general.architecture".into(),
                MetadataValue::from("qwen35"),
            )]),
            tensors: BTreeMap::from([(
                linear.into(),
                Tensor {
                    name: linear.into(),
                    dimensions: vec![4, 2],
                    kind: 0,
                    data: Arc::<[u8]>::from(vec![0_u8; 32]).into(),
                },
            )]),
        };
        let a_vals = [1.0_f32, 0.0, -0.5, 0.0];
        let b_vals = [0.25_f32, -0.75];
        let model = Model {
            identity: "dense-factor-adapter".into(),
            metadata: BTreeMap::from([
                ("general.architecture".into(), MetadataValue::from("qwen35")),
                ("general.type".into(), MetadataValue::from("adapter")),
                ("adapter.type".into(), MetadataValue::from("lora")),
                ("adapter.lora_alpha".into(), MetadataValue::from(2.0_f32)),
            ]),
            tensors: BTreeMap::from([
                (
                    format!("{linear}.lora_a"),
                    Tensor {
                        name: format!("{linear}.lora_a"),
                        dimensions: vec![4, 1],
                        kind: 0,
                        data: Arc::<[u8]>::from(
                            a_vals
                                .iter()
                                .flat_map(|v| v.to_le_bytes())
                                .collect::<Vec<_>>(),
                        )
                        .into(),
                    },
                ),
                (
                    format!("{linear}.lora_b"),
                    Tensor {
                        name: format!("{linear}.lora_b"),
                        dimensions: vec![1, 2],
                        kind: 0,
                        data: Arc::<[u8]>::from(
                            b_vals
                                .iter()
                                .flat_map(|v| v.to_le_bytes())
                                .collect::<Vec<_>>(),
                        )
                        .into(),
                    },
                ),
            ]),
        };
        let mut adapter = RankOneAdapter::from_model(model, &base)?;
        assert!(adapter.dense_rank_one_factor(linear, None).is_none());
        assert!(adapter.dense_rank_one_factor(linear, Some(0.0)).is_none());
        assert!(adapter.dense_rank_one_factor(linear, Some(-0.0)).is_none());
        assert!(
            adapter
                .dense_rank_one_factor("blk.1.ffn_down.weight", Some(1.3))
                .is_none()
        );
        let (a, scale_alpha, b) = adapter
            .dense_rank_one_factor(linear, Some(1.3))
            .ok_or_else(|| crate::invalid("expected dense rank-one factor"))?;
        assert_eq!(a, &a_vals);
        assert_eq!(scale_alpha.to_bits(), (1.3_f32 * 2.0_f32).to_bits());
        assert_eq!(b, &b_vals);
        adapter.enable_sparse_a()?;
        assert!(adapter.dense_rank_one_factor(linear, Some(1.3)).is_none());
        Ok(())
    }

    fn build_compact_rank_one_adapters(
        target: &str,
        width: usize,
        out_width: usize,
    ) -> Result<(RankOneAdapter, RankOneAdapter)> {
        let base = Model {
            identity: "swift-r1-compact-base".into(),
            metadata: BTreeMap::from([(
                "general.architecture".into(),
                MetadataValue::from("qwen35"),
            )]),
            tensors: BTreeMap::from([(
                target.into(),
                Tensor {
                    name: target.into(),
                    dimensions: vec![width, out_width],
                    kind: 0,
                    data: Arc::<[u8]>::from(vec![0_u8; width * out_width * 4]).into(),
                },
            )]),
        };
        let a_vals: Vec<f32> = (0..width)
            .map(|i| match i % 17 {
                0 => 0.125,
                5 => -0.0625,
                _ => 0.0,
            })
            .collect();
        let b_vals = [0.5_f32, -0.25, 0.75, -0.125];
        let model = Model {
            identity: "swift-r1-compact-adapter".into(),
            metadata: BTreeMap::from([
                ("general.architecture".into(), MetadataValue::from("qwen35")),
                ("general.type".into(), MetadataValue::from("adapter")),
                ("adapter.type".into(), MetadataValue::from("lora")),
                ("adapter.lora_alpha".into(), MetadataValue::from(1.0_f32)),
            ]),
            tensors: BTreeMap::from([
                (
                    format!("{target}.lora_a"),
                    Tensor {
                        name: format!("{target}.lora_a"),
                        dimensions: vec![width, 1],
                        kind: 0,
                        data: Arc::<[u8]>::from(
                            a_vals
                                .iter()
                                .flat_map(|v| v.to_le_bytes())
                                .collect::<Vec<_>>(),
                        )
                        .into(),
                    },
                ),
                (
                    format!("{target}.lora_b"),
                    tensor(&format!("{target}.lora_b"), &b_vals),
                ),
            ]),
        };
        let dense = RankOneAdapter::from_model(model, &base)?;
        let mut sparse = RankOneAdapter {
            identity: dense.identity.clone(),
            execution_identity: dense.execution_identity.clone(),
            base_identity: dense.base_identity.clone(),
            alpha: dense.alpha,
            embedding: None,
            factors: dense.factors.clone(),
        };
        sparse.enable_sparse_a()?;
        Ok((dense, sparse))
    }

    #[test]
    fn compact_ffn_down_layers_preserves_dense_and_sparse_zero_masked_dots() -> Result<()> {
        let target = "blk.0.ffn_down.weight";
        let width = 17_408_usize;
        let (dense, sparse) = build_compact_rank_one_adapters(target, width, 4)?;

        let omitted = [0_usize, 3, 8, 16];
        let dense_compacted = dense.compact_ffn_down_layers(&[(0, &omitted)])?;
        let sparse_compacted = sparse.compact_ffn_down_layers(&[(0, &omitted)])?;
        assert_ne!(
            dense.execution_identity(),
            dense_compacted.execution_identity()
        );
        assert_ne!(
            sparse.execution_identity(),
            sparse_compacted.execution_identity()
        );
        assert!(dense.compact_ffn_down_layers(&[(0, &[])]).is_err());
        assert!(
            dense
                .compact_ffn_down_layers(&[(0, &(0..17).collect::<Vec<_>>())])
                .is_err()
        );
        assert!(dense.compact_ffn_down_layers(&[(0, &[2, 1])]).is_err());

        let mut full_input: Vec<f32> = (0..width)
            .map(|i| [-0.5_f32, 0.25, 1.0, -0.125, 0.75][i % 5])
            .collect();
        let mut compacted_input = Vec::new();
        for g in 0..17 {
            if omitted.contains(&g) {
                full_input[g * 1024..(g + 1) * 1024].fill(0.0);
            } else {
                compacted_input.extend_from_slice(&full_input[g * 1024..(g + 1) * 1024]);
            }
        }
        for (orig, comp) in [(&dense, &dense_compacted), (&sparse, &sparse_compacted)] {
            let mut scratch_a = Scratch::new(MatvecBackend::Scalar, true);
            let mut scratch_b = Scratch::new(MatvecBackend::Scalar, true);
            let mut out_orig = [0.25_f32, -0.5, 1.0, -0.125];
            let mut out_comp = out_orig;
            orig.linear(target, &full_input, 1.3, &mut out_orig, &mut scratch_a)?;
            comp.linear(target, &compacted_input, 1.3, &mut out_comp, &mut scratch_b)?;
            assert_eq!(
                out_orig.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                out_comp.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
        Ok(())
    }
}
