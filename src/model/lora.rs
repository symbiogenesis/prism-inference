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

struct SparseA {
    width: usize,
    indices: Vec<u16>,
    values: Vec<f32>,
}

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
    correction: Vec<f32>,
    matvec: MatvecScratch,
}

impl Scratch {
    pub(crate) fn new(backend: MatvecBackend, pinned: bool) -> Self {
        Self {
            rank: Vec::new(),
            correction: Vec::new(),
            matvec: if pinned {
                MatvecScratch::with_pinned_backend(backend)
            } else {
                MatvecScratch::with_backend(backend)
            },
        }
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
        Ok(Self {
            execution_identity: model.identity.clone(),
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

    fn enable_sparse_a(&mut self) -> Result<()> {
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
            && Arc::ptr_eq(&other.data, &tensor.data)
    }) {
        return Ok(Arc::clone(values));
    }
    let values = Arc::new(decode_factor(tensor)?);
    decoded.try_reserve_exact(1)?;
    decoded.push((tensor, Arc::clone(&values)));
    Ok(values)
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
            data: Arc::from(
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
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
            data: Arc::from(vec![0_u8; a_vals.len() * b_vals.len() * 4]),
        };
        let a_tensor = Tensor {
            name: format!("{target}.lora_a"),
            dimensions: vec![a_vals.len(), 1],
            kind: 0,
            data: Arc::from(
                a_vals
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
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
}
