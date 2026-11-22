//! Stackable multi-rank `LoRA` adapters (`F16`/`F32`) enforcing NPU-friendly target envelopes.

use std::{collections::BTreeMap, path::Path};

use bonsai_gguf::MetadataValue;
use bonsai_ternary::half_to_float;

use super::{Model, Scratch, Tensor, hash_len_bytes};
use crate::{
    Result, invalid,
    kernels::{
        lora_multi_a_f16, lora_multi_a_f32, lora_multi_a_tokens_f16, lora_multi_a_tokens_f32,
        lora_multi_b_scale_and_finish_f16, lora_multi_b_scale_and_finish_f32,
        lora_multi_b_scale_and_finish_tokens_f16, lora_multi_b_scale_and_finish_tokens_f32,
        prefill::MAX_TILE_TOKENS,
    },
};

const MAX_MULTI_RANK: usize = 256;
const MAX_MULTI_RANK_MEMORY_BYTES: usize = 256 * 1024 * 1024;
const LORA_A_SUFFIX: &str = ".lora_a";
const LORA_B_SUFFIX: &str = ".lora_b";
const IDENTITY_DOMAIN: &[u8] = b"prism.multi_rank_adapter.v1";

/// Target admission policy for multi-rank adapters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MultiRankPolicy {
    /// Admits linear block weights excluding recurrent `ssm_out.weight`,
    /// `token_embd.weight`, `output.weight`, and FFN `ffn_gate.weight` /
    /// `ffn_up.weight` so fused Hexagon `SwiGLU` projections remain eligible.
    NpuFriendly,
    /// Admits linear block weights including `ffn_gate.weight` and
    /// `ffn_up.weight`, while still forbidding `ssm_out.weight`,
    /// `token_embd.weight`, and `output.weight`.
    AnyLinear,
}

impl MultiRankPolicy {
    const fn identity_tag(self) -> &'static [u8] {
        match self {
            Self::NpuFriendly => b"npu-friendly",
            Self::AnyLinear => b"any-linear",
        }
    }
}

#[derive(Clone)]
enum MultiRankStorage {
    F16 { a: Box<[u16]>, b: Box<[u16]> },
    F32 { a: Box<[f32]>, b: Box<[f32]> },
}

impl MultiRankStorage {
    const fn kind_tag(&self) -> u8 {
        match self {
            Self::F16 { .. } => 1,
            Self::F32 { .. } => 0,
        }
    }

    const fn retained_bytes(&self) -> usize {
        match self {
            Self::F16 { a, b } => (a.len() + b.len()) * size_of::<u16>(),
            Self::F32 { a, b } => (a.len() + b.len()) * size_of::<f32>(),
        }
    }
}

#[derive(Clone)]
struct MultiRankTarget {
    in_features: usize,
    out_features: usize,
    rank: usize,
    scaling: f32,
    storage: MultiRankStorage,
}

/// Verified stackable multi-rank `LoRA` adapter compatible with runtime scale control.
#[derive(Clone)]
pub struct MultiRankAdapter {
    identity: String,
    execution_identity: String,
    base_identity: String,
    policy: MultiRankPolicy,
    alpha: f32,
    max_rank: usize,
    storage_bytes: usize,
    targets: BTreeMap<String, MultiRankTarget>,
}

impl MultiRankAdapter {
    /// Default memory ceiling for verified multi-rank adapter loading.
    pub const DEFAULT_MEMORY_LIMIT: usize = MAX_MULTI_RANK_MEMORY_BYTES;

    /// Loads and verifies a GGUF multi-rank adapter from disk against `base`.
    ///
    /// # Errors
    /// Rejects relative paths, zero byte counts, invalid hex SHA-256 digests,
    /// file size/digest mismatches, malformed GGUF metadata or shapes,
    /// nonfinite or overflowed F16/F32 weights, base shape mismatches, and
    /// targets disallowed by `policy`.
    pub fn load_verified(
        base: &Model,
        path: &Path,
        expected_bytes: u64,
        expected_sha256: &str,
        memory_limit: usize,
        policy: MultiRankPolicy,
    ) -> Result<Self> {
        if !path.is_absolute() || expected_bytes == 0 || memory_limit == 0 {
            return Err(invalid(
                "multi-rank adapter requires an absolute path, positive byte length, and memory limit",
            ));
        }
        let model = Model::load(path, expected_bytes, expected_sha256, memory_limit)?;
        Self::from_model(base, &model, policy)
    }

    /// Validates and builds an immutable `MultiRankAdapter` from loaded GGUF models.
    ///
    /// # Errors
    /// Rejects non-adapter GGUFs, architecture mismatches, missing/mismatched
    /// alpha metadata, unpaired or unknown tensors, disallowed targets,
    /// shape/rank violations, mixed encodings within a target, and nonfinite weights.
    pub fn from_model(base: &Model, adapter: &Model, policy: MultiRankPolicy) -> Result<Self> {
        validate_metadata(base, adapter)?;
        let alpha = resolve_lora_alpha(adapter)?;
        let pairs = collect_lora_pairs(adapter)?;

        let mut targets = BTreeMap::new();
        let mut max_rank = 0_usize;
        let mut storage_bytes = 0_usize;
        for (target_name, (a_tensor, b_tensor)) in pairs {
            validate_target_name(target_name, policy)?;
            let target = parse_target_pair(base, target_name, a_tensor, b_tensor, alpha)?;
            max_rank = max_rank.max(target.rank);
            storage_bytes = storage_bytes
                .checked_add(target.storage.retained_bytes())
                .ok_or_else(|| invalid("multi-rank adapter retained byte overflow"))?;
            targets.insert(target_name.to_owned(), target);
        }

        let execution_identity =
            compute_execution_identity(&base.identity, policy, alpha, &targets)?;

        Ok(Self {
            identity: adapter.identity.clone(),
            execution_identity,
            base_identity: base.identity.clone(),
            policy,
            alpha,
            max_rank,
            storage_bytes,
            targets,
        })
    }

    /// Validates an inference-time multiplier (`0.0..=2.0`, finite).
    ///
    /// # Errors
    /// Rejects `NaN`, infinities, negative scales, or values above `2.0`.
    pub fn validate_scale(scale: f32) -> Result<()> {
        if !scale.is_finite() || !(0.0..=2.0).contains(&scale) {
            return Err(invalid(
                "multi-rank adapter scale must be finite and within 0..=2",
            ));
        }
        Ok(())
    }

    /// Normalizes signed zero scale to `+0.0`.
    #[must_use]
    pub fn normalize_scale(scale: f32) -> f32 {
        if scale == 0.0 { 0.0 } else { scale }
    }

    /// Returns the admission policy enforced for this adapter.
    #[must_use]
    pub const fn policy(&self) -> MultiRankPolicy {
        self.policy
    }

    /// Returns the verified artifact identity string (lowercase hex SHA-256).
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Canonical content-addressed execution identity bound into graph checkpoints.
    #[must_use]
    pub(crate) fn execution_identity(&self) -> &str {
        &self.execution_identity
    }

    /// Base model identity validated against this adapter.
    #[must_use]
    pub(crate) fn base_identity(&self) -> &str {
        &self.base_identity
    }

    /// Configured global `adapter.lora.alpha` value.
    #[must_use]
    pub const fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Maximum `LoRA` rank across all loaded target matrices.
    #[must_use]
    pub const fn max_rank(&self) -> usize {
        self.max_rank
    }

    /// Retained adapter factor bytes in host memory.
    #[must_use]
    pub const fn storage_bytes(&self) -> usize {
        self.storage_bytes
    }

    /// Number of target linear matrices in this adapter.
    #[must_use]
    pub fn target_count(&self) -> usize {
        self.targets.len()
    }

    /// Iterates target tensor names in sorted order.
    pub fn targets(&self) -> impl Iterator<Item = &str> {
        self.targets.keys().map(String::as_str)
    }

    /// Returns `true` when this adapter contains weights for `target`.
    #[must_use]
    pub fn has_target(&self, target: &str) -> bool {
        self.targets.contains_key(target)
    }

    /// Returns `true` if `target` will be modified when run at `scale`.
    #[must_use]
    pub(crate) fn affects_linear(&self, target: &str, scale: f32) -> bool {
        Self::validate_scale(scale).is_ok() && scale != 0.0 && self.targets.contains_key(target)
    }

    pub(crate) fn compact_ffn_down_layers(&self, layers: &[(usize, &[usize])]) -> Result<Self> {
        const GROUP_WIDTH: usize = 1024;
        const GROUPS: usize = 17;
        const FULL_WIDTH: usize = GROUP_WIDTH * GROUPS;
        let mut compacted = self.clone();
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
            if compacted
                .targets
                .contains_key(&format!("blk.{layer}.ffn_gate.weight"))
                || compacted
                    .targets
                    .contains_key(&format!("blk.{layer}.ffn_up.weight"))
            {
                return Err(invalid(
                    "multi-rank gate/up adapters are unsupported on compacted FFN layers",
                ));
            }
            let target_name = format!("blk.{layer}.ffn_down.weight");
            let Some(entry) = compacted.targets.get_mut(&target_name) else {
                continue;
            };
            if entry.in_features != FULL_WIDTH {
                return Err(invalid(format!(
                    "multi-rank adapter {target_name} input width does not match Swift FFN"
                )));
            }
            let new_in_features = (GROUPS - omitted_groups.len()) * GROUP_WIDTH;
            match &mut entry.storage {
                MultiRankStorage::F16 { a, .. } => {
                    *a = compact_multi_a_rows(a, entry.rank, omitted_groups)?;
                }
                MultiRankStorage::F32 { a, .. } => {
                    *a = compact_multi_a_rows(a, entry.rank, omitted_groups)?;
                }
            }
            entry.in_features = new_in_features;
        }
        compacted.storage_bytes = compacted
            .targets
            .values()
            .try_fold(0_usize, |acc, target| {
                acc.checked_add(target.storage.retained_bytes())
            })
            .ok_or_else(|| invalid("multi-rank adapter retained byte overflow"))?;
        compacted.execution_identity = compute_execution_identity(
            &compacted.base_identity,
            compacted.policy,
            compacted.alpha,
            &compacted.targets,
        )?;
        Ok(compacted)
    }

    /// Applies this adapter in place onto a single token's projection `output`.
    ///
    /// Preserves `output` verbatim when `scale == 0.0` or when `target` is absent.
    ///
    /// # Errors
    /// Rejects invalid scales, mismatched input/output shapes, or nonfinite
    /// intermediate products without partially mutating `output`.
    pub(crate) fn linear(
        &self,
        target: &str,
        input: &[f32],
        scale: f32,
        output: &mut [f32],
        scratch: &mut Scratch,
    ) -> Result<()> {
        Self::validate_scale(scale)?;
        let Some(entry) = self.targets.get(target) else {
            return Ok(());
        };
        if input.len() != entry.in_features || output.len() != entry.out_features {
            return Err(invalid(format!(
                "multi-rank adapter dimension mismatch for {target}"
            )));
        }
        if scale == 0.0 {
            return Ok(());
        }
        let effective_scale = scale * entry.scaling;
        if !effective_scale.is_finite() {
            return Err(invalid(format!(
                "nonfinite effective LoRA scale for {target}"
            )));
        }
        let parallelism = scratch.matvec.clone_row_parallelism();
        match &entry.storage {
            MultiRankStorage::F16 { a, b } => {
                lora_multi_a_f16(
                    a,
                    entry.in_features,
                    input,
                    &mut scratch.multi_ranks,
                    &parallelism,
                )?;
                lora_multi_b_scale_and_finish_f16(
                    b,
                    &scratch.multi_ranks,
                    effective_scale,
                    output,
                    &mut scratch.correction,
                    &parallelism,
                )?;
            }
            MultiRankStorage::F32 { a, b } => {
                lora_multi_a_f32(
                    a,
                    entry.in_features,
                    input,
                    &mut scratch.multi_ranks,
                    &parallelism,
                )?;
                lora_multi_b_scale_and_finish_f32(
                    b,
                    &scratch.multi_ranks,
                    effective_scale,
                    output,
                    &mut scratch.correction,
                    &parallelism,
                )?;
            }
        }
        Ok(())
    }

    /// Applies this adapter across a batched prefill token tile (`inputs.len() <= outputs.len()`).
    ///
    /// # Errors
    /// Rejects mismatched token batch counts or extents, invalid scales, or
    /// nonfinite outputs without partially mutating `outputs`.
    pub(crate) fn linear_tokens(
        &self,
        target: &str,
        inputs: &[Vec<f32>],
        scale: f32,
        outputs: &mut [Vec<f32>],
        scratch: &mut Scratch,
    ) -> Result<()> {
        Self::validate_scale(scale)?;
        if inputs.len() != outputs.len() {
            return Err(invalid("multi-rank adapter token batch count mismatch"));
        }
        if inputs.is_empty() {
            return Ok(());
        }
        let Some(entry) = self.targets.get(target) else {
            return Ok(());
        };
        if inputs.iter().any(|input| input.len() != entry.in_features)
            || outputs
                .iter()
                .any(|output| output.len() != entry.out_features)
        {
            return Err(invalid(format!(
                "multi-rank adapter batched dimension mismatch for {target}"
            )));
        }
        if scale == 0.0 {
            return Ok(());
        }
        let effective_scale = scale * entry.scaling;
        if !effective_scale.is_finite() {
            return Err(invalid(format!(
                "nonfinite effective LoRA scale for {target}"
            )));
        }
        let mut stack_inputs: [&[f32]; MAX_TILE_TOKENS] = [&[]; MAX_TILE_TOKENS];
        let mut heap_inputs = Vec::new();
        let input_refs: &[&[f32]] = if inputs.len() <= MAX_TILE_TOKENS {
            for (slot, input) in stack_inputs[..inputs.len()].iter_mut().zip(inputs.iter()) {
                *slot = input.as_slice();
            }
            &stack_inputs[..inputs.len()]
        } else {
            heap_inputs.try_reserve_exact(inputs.len())?;
            for input in inputs {
                heap_inputs.push(input.as_slice());
            }
            &heap_inputs
        };
        let parallelism = scratch.matvec.clone_row_parallelism();
        match &entry.storage {
            MultiRankStorage::F16 { a, b } => {
                lora_multi_a_tokens_f16(
                    a,
                    entry.in_features,
                    entry.rank,
                    input_refs,
                    &mut scratch.multi_ranks,
                    &parallelism,
                )?;
                lora_multi_b_scale_and_finish_tokens_f16(
                    b,
                    &scratch.multi_ranks,
                    entry.rank,
                    effective_scale,
                    outputs,
                    &mut scratch.correction,
                    &parallelism,
                )?;
            }
            MultiRankStorage::F32 { a, b } => {
                lora_multi_a_tokens_f32(
                    a,
                    entry.in_features,
                    entry.rank,
                    input_refs,
                    &mut scratch.multi_ranks,
                    &parallelism,
                )?;
                lora_multi_b_scale_and_finish_tokens_f32(
                    b,
                    &scratch.multi_ranks,
                    entry.rank,
                    effective_scale,
                    outputs,
                    &mut scratch.correction,
                    &parallelism,
                )?;
            }
        }
        Ok(())
    }
}

fn compact_multi_a_rows<T: Copy>(
    old_a: &[T],
    rank: usize,
    omitted_groups: &[usize],
) -> Result<Box<[T]>> {
    const GROUP_WIDTH: usize = 1024;
    const GROUPS: usize = 17;
    const FULL_WIDTH: usize = GROUP_WIDTH * GROUPS;
    let new_in = (GROUPS - omitted_groups.len()) * GROUP_WIDTH;
    let total = rank
        .checked_mul(new_in)
        .ok_or_else(|| invalid("multi-rank LoRA compacted A size overflow"))?;
    let mut compacted = Vec::new();
    compacted.try_reserve_exact(total)?;
    for row in old_a.as_chunks::<FULL_WIDTH>().0 {
        for group in 0..GROUPS {
            if !omitted_groups.contains(&group) {
                compacted.extend_from_slice(&row[group * GROUP_WIDTH..(group + 1) * GROUP_WIDTH]);
            }
        }
    }
    Ok(compacted.into_boxed_slice())
}

fn collect_lora_pairs(adapter: &Model) -> Result<BTreeMap<&str, (&Tensor, &Tensor)>> {
    let mut partial: BTreeMap<&str, (Option<&Tensor>, Option<&Tensor>)> = BTreeMap::new();
    for (name, tensor) in &adapter.tensors {
        if let Some(target) = name.strip_suffix(LORA_A_SUFFIX) {
            if target.is_empty() {
                return Err(invalid("empty target name in multi-rank LoRA A tensor"));
            }
            let entry = partial.entry(target).or_insert((None, None));
            if entry.0.is_some() {
                return Err(invalid(format!("duplicate lora_a tensor for {target}")));
            }
            entry.0 = Some(tensor);
        } else if let Some(target) = name.strip_suffix(LORA_B_SUFFIX) {
            if target.is_empty() {
                return Err(invalid("empty target name in multi-rank LoRA B tensor"));
            }
            let entry = partial.entry(target).or_insert((None, None));
            if entry.1.is_some() {
                return Err(invalid(format!("duplicate lora_b tensor for {target}")));
            }
            entry.1 = Some(tensor);
        } else {
            return Err(invalid(format!(
                "unexpected multi-rank adapter tensor {name}"
            )));
        }
    }
    if partial.is_empty() {
        return Err(invalid("multi-rank adapter contains no LoRA pairs"));
    }
    let mut pairs = BTreeMap::new();
    for (target_name, (a_opt, b_opt)) in partial {
        let a_tensor = a_opt.ok_or_else(|| {
            invalid(format!(
                "multi-rank adapter target {target_name} missing lora_a"
            ))
        })?;
        let b_tensor = b_opt.ok_or_else(|| {
            invalid(format!(
                "multi-rank adapter target {target_name} missing lora_b"
            ))
        })?;
        pairs.insert(target_name, (a_tensor, b_tensor));
    }
    Ok(pairs)
}

fn parse_target_pair(
    base: &Model,
    target_name: &str,
    a_tensor: &Tensor,
    b_tensor: &Tensor,
    alpha: f32,
) -> Result<MultiRankTarget> {
    let base_tensor = base.tensors.get(target_name).ok_or_else(|| {
        invalid(format!(
            "multi-rank adapter target {target_name} missing from base model"
        ))
    })?;
    if base_tensor.dimensions.len() != 2 {
        return Err(invalid(format!(
            "multi-rank adapter target {target_name} requires a 2D base matrix"
        )));
    }
    let in_features = base_tensor.dimensions[0];
    let out_features = base_tensor.dimensions[1];
    if in_features == 0 || out_features == 0 {
        return Err(invalid(format!(
            "multi-rank adapter target {target_name} has zero base extent"
        )));
    }
    if a_tensor.dimensions.len() != 2 || b_tensor.dimensions.len() != 2 {
        return Err(invalid(format!(
            "multi-rank adapter {target_name} factors must be 2D"
        )));
    }
    if a_tensor.dimensions[0] != in_features {
        return Err(invalid(format!(
            "multi-rank adapter {target_name}.lora_a input extent mismatch"
        )));
    }
    let rank = a_tensor.dimensions[1];
    if rank == 0 || rank > MAX_MULTI_RANK {
        return Err(invalid(format!(
            "multi-rank adapter {target_name} rank {rank} outside 1..={MAX_MULTI_RANK}"
        )));
    }
    if b_tensor.dimensions[0] != rank || b_tensor.dimensions[1] != out_features {
        return Err(invalid(format!(
            "multi-rank adapter {target_name}.lora_b shape mismatch"
        )));
    }
    if a_tensor.kind != b_tensor.kind {
        return Err(invalid(format!(
            "multi-rank adapter {target_name} mixes factor encodings"
        )));
    }
    let a_elems = rank
        .checked_mul(in_features)
        .ok_or_else(|| invalid("multi-rank LoRA A element count overflow"))?;
    let b_elems = out_features
        .checked_mul(rank)
        .ok_or_else(|| invalid("multi-rank LoRA B element count overflow"))?;
    let storage = match a_tensor.kind {
        1 => MultiRankStorage::F16 {
            a: decode_f16_slice(a_tensor, a_elems, target_name)?,
            b: decode_f16_slice(b_tensor, b_elems, target_name)?,
        },
        0 => MultiRankStorage::F32 {
            a: decode_f32_slice(a_tensor, a_elems, target_name)?,
            b: decode_f32_slice(b_tensor, b_elems, target_name)?,
        },
        kind => {
            return Err(invalid(format!(
                "unsupported multi-rank adapter encoding {kind} for {target_name}"
            )));
        }
    };
    let rank_u16 = u16::try_from(rank)?;
    let scaling = alpha / f32::from(rank_u16);
    if !scaling.is_finite() || scaling <= 0.0 {
        return Err(invalid(format!(
            "nonfinite or nonpositive LoRA scaling for {target_name}"
        )));
    }
    Ok(MultiRankTarget {
        in_features,
        out_features,
        rank,
        scaling,
        storage,
    })
}

fn validate_metadata(base: &Model, adapter: &Model) -> Result<()> {
    match adapter.metadata.get("general.type") {
        Some(MetadataValue::String(kind)) if kind == "adapter" => {}
        _ => {
            return Err(invalid(
                "multi-rank adapter requires general.type = adapter",
            ));
        }
    }
    match adapter.metadata.get("adapter.type") {
        Some(MetadataValue::String(kind)) if kind == "lora" => {}
        _ => return Err(invalid("multi-rank adapter requires adapter.type = lora")),
    }
    if let (Some(MetadataValue::String(base_arch)), Some(MetadataValue::String(adapter_arch))) = (
        base.metadata.get("general.architecture"),
        adapter.metadata.get("general.architecture"),
    ) && base_arch != adapter_arch
    {
        return Err(invalid(format!(
            "multi-rank adapter architecture {adapter_arch} mismatches base {base_arch}"
        )));
    } else if matches!(
        adapter.metadata.get("general.architecture"),
        Some(other) if !matches!(other, MetadataValue::String(_))
    ) {
        return Err(invalid("malformed general.architecture in adapter"));
    }
    Ok(())
}

fn resolve_lora_alpha(adapter: &Model) -> Result<f32> {
    let parse_alpha = |value: &MetadataValue| -> Result<f32> {
        match *value {
            MetadataValue::F32(alpha) if alpha.is_finite() && alpha > 0.0 => Ok(alpha),
            _ => Err(invalid("adapter LoRA alpha must be finite and positive")),
        }
    };
    let primary = adapter
        .metadata
        .get("adapter.lora.alpha")
        .map(parse_alpha)
        .transpose()?;
    let legacy = adapter
        .metadata
        .get("adapter.lora_alpha")
        .map(parse_alpha)
        .transpose()?;
    match (primary, legacy) {
        (Some(a), Some(b)) => {
            if a.to_bits() != b.to_bits() {
                return Err(invalid(
                    "adapter.lora.alpha disagrees with legacy adapter.lora_alpha",
                ));
            }
            Ok(a)
        }
        (Some(a), None) | (None, Some(a)) => Ok(a),
        (None, None) => Err(invalid("missing adapter.lora.alpha metadata")),
    }
}

fn validate_target_name(target: &str, policy: MultiRankPolicy) -> Result<()> {
    if !target.ends_with(".weight") {
        return Err(invalid(format!(
            "multi-rank adapter target {target} must reference a .weight matrix"
        )));
    }
    if target == "token_embd.weight"
        || target == "output.weight"
        || target.ends_with(".ssm_out.weight")
    {
        return Err(invalid(format!(
            "multi-rank adapter forbids target {target}"
        )));
    }
    if policy == MultiRankPolicy::NpuFriendly
        && (target.ends_with(".ffn_gate.weight") || target.ends_with(".ffn_up.weight"))
    {
        return Err(invalid(format!(
            "npu-friendly multi-rank adapter forbids gating target {target}"
        )));
    }
    Ok(())
}

fn decode_f16_slice(tensor: &Tensor, expected_elems: usize, target: &str) -> Result<Box<[u16]>> {
    let expected_bytes = expected_elems
        .checked_mul(2)
        .ok_or_else(|| invalid("F16 byte length overflow"))?;
    if tensor.data.len() != expected_bytes {
        return Err(invalid(format!(
            "multi-rank F16 tensor length mismatch for {target}"
        )));
    }
    let mut values = Vec::new();
    values.try_reserve_exact(expected_elems)?;
    for &chunk in tensor.data.as_chunks::<2>().0 {
        let bits = u16::from_le_bytes(chunk);
        let value = half_to_float(bits);
        if !value.is_finite() || value.abs() >= 65_520.0 {
            return Err(invalid(format!(
                "nonfinite or overflowed F16 weight in multi-rank adapter {target}"
            )));
        }
        values.push(bits);
    }
    Ok(values.into_boxed_slice())
}

fn decode_f32_slice(tensor: &Tensor, expected_elems: usize, target: &str) -> Result<Box<[f32]>> {
    let expected_bytes = expected_elems
        .checked_mul(4)
        .ok_or_else(|| invalid("F32 byte length overflow"))?;
    if tensor.data.len() != expected_bytes {
        return Err(invalid(format!(
            "multi-rank F32 tensor length mismatch for {target}"
        )));
    }
    let mut values = Vec::new();
    values.try_reserve_exact(expected_elems)?;
    for &chunk in tensor.data.as_chunks::<4>().0 {
        let value = f32::from_le_bytes(chunk);
        if !value.is_finite() {
            return Err(invalid(format!(
                "nonfinite F32 weight in multi-rank adapter {target}"
            )));
        }
        values.push(value);
    }
    Ok(values.into_boxed_slice())
}

fn compute_execution_identity(
    base_identity: &str,
    policy: MultiRankPolicy,
    alpha: f32,
    targets: &BTreeMap<String, MultiRankTarget>,
) -> Result<String> {
    let mut hasher = bonsai_native_hash::Hasher::new()?;
    hash_len_bytes(&mut hasher, IDENTITY_DOMAIN)?;
    hash_len_bytes(&mut hasher, base_identity.as_bytes())?;
    hash_len_bytes(&mut hasher, policy.identity_tag())?;
    hasher.update(&alpha.to_bits().to_le_bytes())?;
    hasher.update(&u64::try_from(targets.len())?.to_le_bytes())?;
    for (name, target) in targets {
        hash_len_bytes(&mut hasher, name.as_bytes())?;
        hasher.update(&u64::try_from(target.in_features)?.to_le_bytes())?;
        hasher.update(&u64::try_from(target.out_features)?.to_le_bytes())?;
        hasher.update(&u32::try_from(target.rank)?.to_le_bytes())?;
        hasher.update(&[target.storage.kind_tag()])?;
        match &target.storage {
            MultiRankStorage::F16 { a, b } => {
                hasher.update(&u64::try_from(a.len())?.to_le_bytes())?;
                for &bits in a {
                    hasher.update(&bits.to_le_bytes())?;
                }
                hasher.update(&u64::try_from(b.len())?.to_le_bytes())?;
                for &bits in b {
                    hasher.update(&bits.to_le_bytes())?;
                }
            }
            MultiRankStorage::F32 { a, b } => {
                hasher.update(&u64::try_from(a.len())?.to_le_bytes())?;
                for &val in a {
                    hasher.update(&val.to_bits().to_le_bytes())?;
                }
                hasher.update(&u64::try_from(b.len())?.to_le_bytes())?;
                for &val in b {
                    hasher.update(&val.to_bits().to_le_bytes())?;
                }
            }
        }
    }
    Ok(hex::encode(hasher.finish()?))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use bonsai_gguf::MetadataValue;

    use super::{MultiRankAdapter, MultiRankPolicy};
    use crate::{
        Result,
        kernels::{MatvecBackend, half_bits},
        model::{Model, Tensor, TensorBytes, lora::Scratch},
    };

    fn f32_tensor(name: &str, dimensions: &[usize], values: &[f32]) -> Tensor {
        Tensor {
            name: name.to_owned(),
            dimensions: dimensions.to_vec(),
            kind: 0,
            data: TensorBytes::from(
                values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
        }
    }

    fn f16_tensor(name: &str, dimensions: &[usize], values: &[f32]) -> Tensor {
        Tensor {
            name: name.to_owned(),
            dimensions: dimensions.to_vec(),
            kind: 1,
            data: TensorBytes::from(
                values
                    .iter()
                    .flat_map(|v| half_bits(*v).to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
        }
    }

    fn base_model(targets: &[(&str, [usize; 2])]) -> Model {
        let mut tensors = BTreeMap::new();
        for &(name, dims) in targets {
            tensors.insert(
                name.to_owned(),
                f32_tensor(name, &dims, &vec![0.0; dims[0] * dims[1]]),
            );
        }
        Model {
            identity: "base-id".into(),
            metadata: BTreeMap::from([(
                "general.architecture".into(),
                MetadataValue::from("qwen35"),
            )]),
            tensors,
        }
    }

    fn adapter_metadata(alpha: f32) -> BTreeMap<String, MetadataValue> {
        BTreeMap::from([
            ("general.architecture".into(), MetadataValue::from("qwen35")),
            ("general.type".into(), MetadataValue::from("adapter")),
            ("adapter.type".into(), MetadataValue::from("lora")),
            ("adapter.lora.alpha".into(), MetadataValue::from(alpha)),
            ("adapter.lora_alpha".into(), MetadataValue::from(alpha)),
        ])
    }

    #[test]
    fn npu_friendly_adapter_enforces_envelope_and_computes_exact_f16_rank2_delta() -> Result<()> {
        let base = base_model(&[
            ("blk.0.ffn_down.weight", [4, 3]),
            ("blk.0.attn_q.weight", [4, 4]),
            ("blk.0.ffn_gate.weight", [4, 3]),
            ("blk.0.ssm_out.weight", [4, 4]),
        ]);
        // A = [[1, 0, -1, 2], [0.5, 1.5, 0, -0.5]] (dims [in=4, rank=2])
        // B = [[2, -1], [0, 4], [1, 1]] (dims [rank=2, out=3])
        // alpha = 4.0 => scaling = alpha / rank = 2.0
        let a_vals = [1.0, 0.0, -1.0, 2.0, 0.5, 1.5, 0.0, -0.5];
        let b_vals = [2.0, -1.0, 0.0, 4.0, 1.0, 1.0];
        let adapter_model = Model {
            identity: "a".repeat(64),
            metadata: adapter_metadata(4.0),
            tensors: BTreeMap::from([
                (
                    "blk.0.ffn_down.weight.lora_a".into(),
                    f16_tensor("blk.0.ffn_down.weight.lora_a", &[4, 2], &a_vals),
                ),
                (
                    "blk.0.ffn_down.weight.lora_b".into(),
                    f16_tensor("blk.0.ffn_down.weight.lora_b", &[2, 3], &b_vals),
                ),
            ]),
        };
        let adapter =
            MultiRankAdapter::from_model(&base, &adapter_model, MultiRankPolicy::NpuFriendly)?;
        assert_eq!(adapter.max_rank(), 2);
        assert_eq!(adapter.target_count(), 1);
        assert_eq!(adapter.storage_bytes(), (8 + 6) * 2);
        assert!(adapter.affects_linear("blk.0.ffn_down.weight", 0.5));
        assert!(!adapter.affects_linear("blk.0.ffn_down.weight", 0.0));

        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut scratch = Scratch::new(MatvecBackend::ExperimentalNeon, false);

        // Zero scale preserves output verbatim.
        let mut out_zero = [10.0_f32, -2.0, 5.0];
        adapter.linear(
            "blk.0.ffn_down.weight",
            &input,
            0.0,
            &mut out_zero,
            &mut scratch,
        )?;
        assert_eq!(out_zero, [10.0, -2.0, 5.0]);

        // r = [1*1 - 1*3 + 2*4, 0.5*1 + 1.5*2 - 0.5*4] = [6.0, 1.5]
        // B * r = [2*6 - 1.5 = 10.5, 4*1.5 = 6.0, 6 + 1.5 = 7.5]
        // At scale 0.5, effective_scale = 0.5 * (4.0 / 2) = 1.0
        let mut out = [10.0_f32, -2.0, 5.0];
        adapter.linear("blk.0.ffn_down.weight", &input, 0.5, &mut out, &mut scratch)?;
        assert_eq!(out, [20.5, 4.0, 12.5]);

        // Batched matches single-token bits identically.
        let mut batched = vec![vec![10.0_f32, -2.0, 5.0], vec![0.0_f32, 1.0, -1.0]];
        let inputs = vec![input.to_vec(), vec![2.0_f32, 0.0, -1.0, 1.0]];
        adapter.linear_tokens(
            "blk.0.ffn_down.weight",
            &inputs,
            0.5,
            &mut batched,
            &mut scratch,
        )?;
        assert_eq!(
            batched[0]
                .iter()
                .copied()
                .map(f32::to_bits)
                .collect::<Vec<_>>(),
            out.iter().copied().map(f32::to_bits).collect::<Vec<_>>()
        );
        let mut second_single = vec![0.0_f32, 1.0, -1.0];
        adapter.linear(
            "blk.0.ffn_down.weight",
            &inputs[1],
            0.5,
            &mut second_single,
            &mut scratch,
        )?;
        assert_eq!(
            batched[1]
                .iter()
                .copied()
                .map(f32::to_bits)
                .collect::<Vec<_>>(),
            second_single
                .iter()
                .copied()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn multi_rank_rejects_disallowed_targets_and_mismatched_alpha_or_dtypes() {
        let base = base_model(&[
            ("token_embd.weight", [4, 4]),
            ("output.weight", [4, 4]),
            ("blk.0.ssm_out.weight", [4, 4]),
            ("blk.0.ffn_gate.weight", [4, 4]),
            ("blk.0.ffn_up.weight", [4, 4]),
            ("blk.0.ffn_down.weight", [4, 4]),
        ]);
        let pair_model =
            |target: &str, mut meta: BTreeMap<String, MetadataValue>, b_f32: bool| Model {
                identity: "b".repeat(64),
                metadata: {
                    meta.entry("general.architecture".into())
                        .or_insert_with(|| MetadataValue::from("qwen35"));
                    meta
                },
                tensors: BTreeMap::from([
                    (
                        format!("{target}.lora_a"),
                        f16_tensor(&format!("{target}.lora_a"), &[4, 2], &[0.1; 8]),
                    ),
                    (
                        format!("{target}.lora_b"),
                        if b_f32 {
                            f32_tensor(&format!("{target}.lora_b"), &[2, 4], &[0.1; 8])
                        } else {
                            f16_tensor(&format!("{target}.lora_b"), &[2, 4], &[0.1; 8])
                        },
                    ),
                ]),
            };

        for forbidden in [
            "token_embd.weight",
            "output.weight",
            "blk.0.ssm_out.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
        ] {
            let model = pair_model(forbidden, adapter_metadata(16.0), false);
            assert!(
                MultiRankAdapter::from_model(&base, &model, MultiRankPolicy::NpuFriendly).is_err(),
                "{forbidden} must be rejected under NpuFriendly"
            );
        }
        // AnyLinear permits ffn_gate/ffn_up, still rejects ssm_out/token_embd/output.
        assert!(
            MultiRankAdapter::from_model(
                &base,
                &pair_model("blk.0.ffn_gate.weight", adapter_metadata(16.0), false),
                MultiRankPolicy::AnyLinear
            )
            .is_ok()
        );
        assert!(
            MultiRankAdapter::from_model(
                &base,
                &pair_model("blk.0.ssm_out.weight", adapter_metadata(16.0), false),
                MultiRankPolicy::AnyLinear
            )
            .is_err()
        );

        // Mixed encodings (F16 A with F32 B) are rejected.
        assert!(
            MultiRankAdapter::from_model(
                &base,
                &pair_model("blk.0.ffn_down.weight", adapter_metadata(16.0), true),
                MultiRankPolicy::NpuFriendly
            )
            .is_err()
        );

        // Disagreeing adapter.lora.alpha and adapter.lora_alpha are rejected.
        let mut mismatch_meta = adapter_metadata(16.0);
        mismatch_meta.insert("adapter.lora_alpha".into(), MetadataValue::from(32.0_f32));
        assert!(
            MultiRankAdapter::from_model(
                &base,
                &pair_model("blk.0.ffn_down.weight", mismatch_meta, false),
                MultiRankPolicy::NpuFriendly
            )
            .is_err()
        );
    }

    #[cfg(feature = "parallel-rows")]
    #[test]
    fn multi_rank_parallel_rows_preserve_single_and_batched_f16_bits() -> Result<()> {
        // Dimensions match Swift attn_k / attn_v (5120 x 1024) at rank 16 (81,920 A elems >= 65,536
        // and 1,024 B rows >= 1,024) so both A and B kernels dispatch onto the row worker pool.
        let target = "blk.3.attn_k.weight";
        let in_features = 5120_usize;
        let out_features = 1024_usize;
        let rank = 16_usize;
        let base = base_model(&[(target, [in_features, out_features])]);
        let a_vals: Vec<f32> = (0..in_features * rank)
            .map(|i| [-0.03125_f32, 0.0625, -0.015_625, 0.046_875, 0.0][(i * 7 + i / 13) % 5])
            .collect();
        let b_vals: Vec<f32> = (0..rank * out_features)
            .map(|i| {
                [0.015_625_f32, -0.03125, 0.007_812_5, -0.023_437_5, 0.03125][(i * 5 + i / 11) % 5]
            })
            .collect();
        let adapter_model = Model {
            identity: "c".repeat(64),
            metadata: adapter_metadata(16.0),
            tensors: BTreeMap::from([
                (
                    format!("{target}.lora_a"),
                    f16_tensor(&format!("{target}.lora_a"), &[in_features, rank], &a_vals),
                ),
                (
                    format!("{target}.lora_b"),
                    f16_tensor(&format!("{target}.lora_b"), &[rank, out_features], &b_vals),
                ),
            ]),
        };
        let adapter =
            MultiRankAdapter::from_model(&base, &adapter_model, MultiRankPolicy::NpuFriendly)?;

        let inputs: Vec<Vec<f32>> = (0..4_usize)
            .map(|tok| {
                (0..in_features)
                    .map(|i| [-1.25_f32, 0.5, 2.0, -0.75, 0.125][(i + tok * 3) % 5])
                    .collect()
            })
            .collect();
        let initial_outputs: Vec<Vec<f32>> = (0..4_usize)
            .map(|tok| {
                (0..out_features)
                    .map(|i| [1.5_f32, -0.25, 0.75, -2.0][(i + tok) % 4])
                    .collect()
            })
            .collect();

        let mut serial_scratch = Scratch::new(MatvecBackend::ExperimentalNeon, true);
        let mut parallel_scratch = Scratch::new(MatvecBackend::ExperimentalNeon, true);
        parallel_scratch.set_row_parallelism(crate::kernels::RowParallelism::new(4)?);

        let mut serial_single = initial_outputs[0].clone();
        let mut parallel_single = initial_outputs[0].clone();
        adapter.linear(
            target,
            &inputs[0],
            1.0,
            &mut serial_single,
            &mut serial_scratch,
        )?;
        adapter.linear(
            target,
            &inputs[0],
            1.0,
            &mut parallel_single,
            &mut parallel_scratch,
        )?;
        let to_bits = |slice: &[f32]| slice.iter().copied().map(f32::to_bits).collect::<Vec<_>>();
        assert_eq!(to_bits(&parallel_single), to_bits(&serial_single));

        let mut serial_batch = initial_outputs.clone();
        let mut parallel_batch = initial_outputs;
        adapter.linear_tokens(target, &inputs, 1.0, &mut serial_batch, &mut serial_scratch)?;
        adapter.linear_tokens(
            target,
            &inputs,
            1.0,
            &mut parallel_batch,
            &mut parallel_scratch,
        )?;
        for (actual, expected) in parallel_batch.iter().zip(&serial_batch) {
            assert_eq!(to_bits(actual), to_bits(expected));
        }
        Ok(())
    }

    fn make_compact_f32_adapter(
        base: &Model,
        dims: [usize; 3],
        a_vals: &[f32],
        b_vals: &[f32],
        with_gate_on_layer1: bool,
        with_up_on_layer0: bool,
    ) -> Result<MultiRankAdapter> {
        let [in_features, out_features, rank] = dims;
        let down_name = "blk.0.ffn_down.weight";
        let gate_name = "blk.1.ffn_gate.weight";
        let up_name = "blk.0.ffn_up.weight";
        let mut tensors = BTreeMap::from([
            (
                format!("{down_name}.lora_a"),
                f32_tensor(&format!("{down_name}.lora_a"), &[in_features, rank], a_vals),
            ),
            (
                format!("{down_name}.lora_b"),
                f32_tensor(
                    &format!("{down_name}.lora_b"),
                    &[rank, out_features],
                    b_vals,
                ),
            ),
        ]);
        for (enabled, target) in [
            (with_gate_on_layer1, gate_name),
            (with_up_on_layer0, up_name),
        ] {
            if enabled {
                tensors.insert(
                    format!("{target}.lora_a"),
                    f32_tensor(
                        &format!("{target}.lora_a"),
                        &[out_features, rank],
                        &vec![0.25; out_features * rank],
                    ),
                );
                tensors.insert(
                    format!("{target}.lora_b"),
                    f32_tensor(
                        &format!("{target}.lora_b"),
                        &[rank, in_features],
                        &vec![0.125; rank * in_features],
                    ),
                );
            }
        }
        let mut metadata = adapter_metadata(2.0);
        metadata.insert("general.architecture".into(), MetadataValue::from("qwen35"));
        let adapter_model = Model {
            metadata,
            tensors,
            identity: "mr-compact-f32".into(),
        };
        MultiRankAdapter::from_model(base, &adapter_model, MultiRankPolicy::AnyLinear)
    }

    fn make_compact_f16_adapter(
        base: &Model,
        dims: [usize; 3],
        a_vals: &[f32],
        b_vals: &[f32],
    ) -> Result<MultiRankAdapter> {
        let [in_features, out_features, rank] = dims;
        let down_name = "blk.0.ffn_down.weight";
        let f16_model = Model {
            metadata: adapter_metadata(2.0),
            tensors: BTreeMap::from([
                (
                    format!("{down_name}.lora_a"),
                    f16_tensor(&format!("{down_name}.lora_a"), &[in_features, rank], a_vals),
                ),
                (
                    format!("{down_name}.lora_b"),
                    f16_tensor(
                        &format!("{down_name}.lora_b"),
                        &[rank, out_features],
                        b_vals,
                    ),
                ),
            ]),
            identity: "mr-compact-f16".into(),
        };
        MultiRankAdapter::from_model(base, &f16_model, MultiRankPolicy::NpuFriendly)
    }

    #[test]
    fn compact_ffn_down_layers_preserves_f16_f32_parity_and_rejects_gate_up() -> Result<()> {
        let in_features = 17_408_usize;
        let out_features = 8_usize;
        let rank = 2_usize;
        let down_name = "blk.0.ffn_down.weight";
        let gate_name = "blk.1.ffn_gate.weight";
        let up_name = "blk.0.ffn_up.weight";

        let base = base_model(&[
            (down_name, [in_features, out_features]),
            (gate_name, [out_features, in_features]),
            (up_name, [out_features, in_features]),
        ]);

        let a_vals: Vec<f32> = (0..rank * in_features)
            .map(|i| [-0.125_f32, 0.25, -0.0625, 0.5][i % 4])
            .collect();
        let b_vals: Vec<f32> = (0..rank * out_features)
            .map(|i| [0.5_f32, -0.25, 0.125, -0.375][i % 4])
            .collect();

        let dims = [in_features, out_features, rank];
        let f16_adapter = make_compact_f16_adapter(&base, dims, &a_vals, &b_vals)?;
        let f32_adapter = make_compact_f32_adapter(&base, dims, &a_vals, &b_vals, true, false)?;
        let gate_up_adapter = make_compact_f32_adapter(&base, dims, &a_vals, &b_vals, false, true)?;

        let omitted = [0_usize, 4, 10, 16];
        assert!(
            gate_up_adapter
                .compact_ffn_down_layers(&[(0, &omitted)])
                .is_err()
        );
        let f32_compacted = f32_adapter.compact_ffn_down_layers(&[(0, &omitted)])?;
        let f16_compacted = f16_adapter.compact_ffn_down_layers(&[(0, &omitted)])?;
        assert!(f32_compacted.storage_bytes() < f32_adapter.storage_bytes());
        assert!(f16_compacted.storage_bytes() < f16_adapter.storage_bytes());
        assert_ne!(
            f32_compacted.execution_identity(),
            f32_adapter.execution_identity()
        );
        assert_ne!(
            f16_compacted.execution_identity(),
            f16_adapter.execution_identity()
        );

        let mut full_input: Vec<f32> = (0..in_features)
            .map(|i| [-0.5_f32, 1.25, -0.75, 0.25, 0.125][i % 5])
            .collect();
        let mut comp_input = Vec::new();
        for g in 0..17 {
            if omitted.contains(&g) {
                full_input[g * 1024..(g + 1) * 1024].fill(0.0);
            } else {
                comp_input.extend_from_slice(&full_input[g * 1024..(g + 1) * 1024]);
            }
        }
        for (orig, comp) in [
            (&f32_adapter, &f32_compacted),
            (&f16_adapter, &f16_compacted),
        ] {
            let mut s0 = Scratch::new(MatvecBackend::Scalar, true);
            let mut s1 = Scratch::new(MatvecBackend::Scalar, true);
            let mut out0 = vec![0.5_f32; out_features];
            let mut out1 = out0.clone();
            orig.linear(down_name, &full_input, 1.0, &mut out0, &mut s0)?;
            comp.linear(down_name, &comp_input, 1.0, &mut out1, &mut s1)?;
            assert_eq!(
                out0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                out1.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
        Ok(())
    }
}
