//! Opt-in W8.2 physical FFN group compaction with surviving adapters and recovery.

use std::collections::BTreeMap;

use bonsai_native_hash::Hasher;

use super::FfnCompensation;
use crate::{
    Result, invalid,
    kernels::{PQ2, PTQ1},
    model::{Tensor, TensorBytes},
};

/// Element width of each Swift FFN Hadamard/omission group.
pub const COMPACTED_FFN_GROUP_WIDTH: usize = 1024;
/// Total number of 1,024-element groups in the Swift FFN intermediate dimension.
pub const COMPACTED_FFN_GROUP_COUNT: usize = 17;
/// Full uncompacted Swift FFN intermediate width (`17 * 1,024 = 17,408`).
pub const COMPACTED_FFN_ORIGINAL_INTERMEDIATE_WIDTH: usize =
    COMPACTED_FFN_GROUP_WIDTH * COMPACTED_FFN_GROUP_COUNT;
/// Hidden residual width of the pinned Swift architecture (`5,120`).
pub const COMPACTED_FFN_HIDDEN_WIDTH: usize = 5_120;
/// Graph identity segment prefix for W8.2 physically compacted FFN layers.
pub const COMPACTED_FFN_PROFILE: &str = "w8-compacted-ffn-v1";

const F32_TENSOR_KIND: u32 = 0;

/// Opt-in W8.2 FFN group compaction specification for a single decoder layer.
pub struct LayerFfnCompaction {
    /// Zero-based decoder layer index.
    pub layer: usize,
    /// Sorted distinct omitted group indices in `0..17` (1 to 16 omitted groups).
    pub omitted_groups: Vec<usize>,
    /// Optional normalized-input low-rank recovery applied before residual addition.
    pub compensation: Option<FfnCompensation>,
}

/// Validated per-layer W8.2 FFN group compaction plan for a block-1024 Swift decoder.
pub struct CompactedFfnPlan {
    layers: Vec<LayerFfnCompaction>,
}

impl CompactedFfnPlan {
    /// Validates and normalizes a per-layer FFN compaction plan.
    ///
    /// Layers are normalized into ascending layer order and checked for uniqueness.
    /// Each layer must specify `1..=16` sorted distinct omitted group indices in `0..17`,
    /// leaving `1..=16` retained groups of `1,024` elements each.
    ///
    /// # Errors
    /// Rejects empty plans, duplicate layer indices, and noncanonical or out-of-range
    /// omitted group selections.
    pub fn new(mut layers: Vec<LayerFfnCompaction>) -> Result<Self> {
        if layers.is_empty() {
            return Err(invalid("compacted FFN plan requires at least one layer"));
        }
        for entry in &layers {
            validate_omitted_groups(&entry.omitted_groups)?;
        }
        layers.sort_by_key(|entry| entry.layer);
        if layers.windows(2).any(|pair| pair[0].layer == pair[1].layer) {
            return Err(invalid("duplicate layer in compacted FFN plan"));
        }
        Ok(Self { layers })
    }

    /// Validated per-layer compaction entries in ascending layer order.
    #[must_use]
    pub fn layers(&self) -> &[LayerFfnCompaction] {
        &self.layers
    }

    pub(in crate::graph) fn into_layers(self) -> Vec<LayerFfnCompaction> {
        self.layers
    }
}

#[derive(Default)]
pub(in crate::graph) struct CompactedFfnRuntime {
    pub(in crate::graph) saved_bytes: usize,
    pub(in crate::graph) identity_suffix: Option<String>,
    pub(in crate::graph) compensations: BTreeMap<usize, Option<FfnCompensation>>,
}

pub(in crate::graph) fn validate_omitted_groups(omitted_groups: &[usize]) -> Result<()> {
    if omitted_groups.is_empty()
        || omitted_groups.len() >= COMPACTED_FFN_GROUP_COUNT
        || omitted_groups
            .iter()
            .any(|&group| group >= COMPACTED_FFN_GROUP_COUNT)
        || omitted_groups.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(invalid(
            "FFN compaction requires 1..=16 sorted distinct omitted groups in 0..17",
        ));
    }
    Ok(())
}

fn ffn_tensor_block_layout(kind: u32) -> Result<(usize, usize)> {
    match kind {
        PQ2 => Ok((128, 34)),
        PTQ1 => Ok((128, 28)),
        F32_TENSOR_KIND => Ok((1, 4)),
        _ => Err(invalid(
            "FFN compaction supports PQ2, PTQ1, and F32 tensors only",
        )),
    }
}

pub(in crate::graph) fn compact_gate_up_rows(
    tensor: &Tensor,
    omitted_groups: &[usize],
) -> Result<Tensor> {
    validate_omitted_groups(omitted_groups)?;
    let [cols, rows] = tensor.dimensions.as_slice() else {
        return Err(invalid("compacted gate/up tensor must be 2D"));
    };
    if *cols == 0 || *rows != COMPACTED_FFN_ORIGINAL_INTERMEDIATE_WIDTH {
        return Err(invalid(
            "compacted gate/up tensor dimensions do not match Swift FFN",
        ));
    }
    let (group_elems, block_bytes) = ffn_tensor_block_layout(tensor.kind)?;
    if !cols.is_multiple_of(group_elems) {
        return Err(invalid(
            "compacted gate/up column count is not aligned to block size",
        ));
    }
    let row_bytes = (*cols / group_elems)
        .checked_mul(block_bytes)
        .ok_or_else(|| invalid("compacted gate/up row byte overflow"))?;
    if row_bytes.checked_mul(*rows) != Some(tensor.data.len()) {
        return Err(invalid("compacted gate/up byte extent mismatch"));
    }
    let retained_groups = COMPACTED_FFN_GROUP_COUNT - omitted_groups.len();
    let group_row_bytes = COMPACTED_FFN_GROUP_WIDTH
        .checked_mul(row_bytes)
        .ok_or_else(|| invalid("compacted gate/up group byte overflow"))?;
    let total_bytes = retained_groups
        .checked_mul(group_row_bytes)
        .ok_or_else(|| invalid("compacted gate/up total byte overflow"))?;
    let mut compacted = Vec::new();
    compacted.try_reserve_exact(total_bytes)?;
    for group in 0..COMPACTED_FFN_GROUP_COUNT {
        if !omitted_groups.contains(&group) {
            let start = group * group_row_bytes;
            compacted.extend_from_slice(&tensor.data[start..start + group_row_bytes]);
        }
    }
    Ok(Tensor {
        name: tensor.name.clone(),
        dimensions: vec![*cols, retained_groups * COMPACTED_FFN_GROUP_WIDTH],
        kind: tensor.kind,
        data: TensorBytes::from(compacted),
    })
}

pub(in crate::graph) fn compact_down_cols(
    tensor: &Tensor,
    omitted_groups: &[usize],
) -> Result<Tensor> {
    validate_omitted_groups(omitted_groups)?;
    let [cols, rows] = tensor.dimensions.as_slice() else {
        return Err(invalid("compacted down tensor must be 2D"));
    };
    if *cols != COMPACTED_FFN_ORIGINAL_INTERMEDIATE_WIDTH || *rows == 0 {
        return Err(invalid(
            "compacted down tensor dimensions do not match Swift FFN",
        ));
    }
    let (group_elems, block_bytes) = ffn_tensor_block_layout(tensor.kind)?;
    let blocks_per_group = COMPACTED_FFN_GROUP_WIDTH / group_elems;
    let group_bytes = blocks_per_group
        .checked_mul(block_bytes)
        .ok_or_else(|| invalid("compacted down group byte overflow"))?;
    let old_row_bytes = COMPACTED_FFN_GROUP_COUNT
        .checked_mul(group_bytes)
        .ok_or_else(|| invalid("compacted down row byte overflow"))?;
    if old_row_bytes.checked_mul(*rows) != Some(tensor.data.len()) {
        return Err(invalid("compacted down byte extent mismatch"));
    }
    let retained_groups = COMPACTED_FFN_GROUP_COUNT - omitted_groups.len();
    let new_row_bytes = retained_groups
        .checked_mul(group_bytes)
        .ok_or_else(|| invalid("compacted down compacted row byte overflow"))?;
    let total_bytes = new_row_bytes
        .checked_mul(*rows)
        .ok_or_else(|| invalid("compacted down total byte overflow"))?;
    let mut compacted = Vec::new();
    compacted.try_reserve_exact(total_bytes)?;
    for row in tensor.data.chunks_exact(old_row_bytes) {
        for group in 0..COMPACTED_FFN_GROUP_COUNT {
            if !omitted_groups.contains(&group) {
                let start = group * group_bytes;
                compacted.extend_from_slice(&row[start..start + group_bytes]);
            }
        }
    }
    Ok(Tensor {
        name: tensor.name.clone(),
        dimensions: vec![retained_groups * COMPACTED_FFN_GROUP_WIDTH, *rows],
        kind: tensor.kind,
        data: TensorBytes::from(compacted),
    })
}

pub(in crate::graph) fn compact_signs(
    full_signs: &[f32],
    omitted_groups: &[usize],
) -> Result<Vec<f32>> {
    validate_omitted_groups(omitted_groups)?;
    if full_signs.len() != COMPACTED_FFN_ORIGINAL_INTERMEDIATE_WIDTH {
        return Err(invalid(
            "compacted FFN signs require Swift 17,408 intermediate width",
        ));
    }
    let retained_groups = COMPACTED_FFN_GROUP_COUNT - omitted_groups.len();
    let mut compacted = Vec::new();
    compacted.try_reserve_exact(retained_groups * COMPACTED_FFN_GROUP_WIDTH)?;
    for group in 0..COMPACTED_FFN_GROUP_COUNT {
        if !omitted_groups.contains(&group) {
            let start = group * COMPACTED_FFN_GROUP_WIDTH;
            compacted.extend_from_slice(&full_signs[start..start + COMPACTED_FFN_GROUP_WIDTH]);
        }
    }
    Ok(compacted)
}

pub(in crate::graph) fn plan_identity_sha256(
    plan: &CompactedFfnPlan,
    signs_sha256: &str,
) -> Result<String> {
    let mut hash = Hasher::new()?;
    hash.update(b"prism-w8-compacted-ffn-v1\0")?;
    hash.update(&u64::try_from(signs_sha256.len())?.to_le_bytes())?;
    hash.update(signs_sha256.as_bytes())?;
    hash.update(&u64::try_from(plan.layers.len())?.to_le_bytes())?;
    for entry in &plan.layers {
        hash.update(&u64::try_from(entry.layer)?.to_le_bytes())?;
        hash.update(&u64::try_from(entry.omitted_groups.len())?.to_le_bytes())?;
        for &group in &entry.omitted_groups {
            hash.update(&u64::try_from(group)?.to_le_bytes())?;
        }
        match &entry.compensation {
            Some(comp) => {
                hash.update(&[1_u8])?;
                hash.update(&u64::try_from(comp.identity().len())?.to_le_bytes())?;
                hash.update(comp.identity().as_bytes())?;
            }
            None => {
                hash.update(&[0_u8])?;
            }
        }
    }
    let mut digest = String::new();
    super::push_sha256_hex(hash, &mut digest)?;
    Ok(digest)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use bonsai_gguf::MetadataValue;

    use super::{
        COMPACTED_FFN_GROUP_WIDTH, COMPACTED_FFN_HIDDEN_WIDTH,
        COMPACTED_FFN_ORIGINAL_INTERMEDIATE_WIDTH, CompactedFfnPlan, LayerFfnCompaction,
        compact_down_cols, compact_gate_up_rows, compact_signs, plan_identity_sha256,
    };
    use crate::graph::weight_patch::{FfnCompensation, FfnProbe};
    use crate::graph::{KvCachePolicy, Qwen35};
    use crate::kernels::{MatvecBackend, PQ2, PTQ1};
    use crate::model::{
        Model, Tensor,
        lora::{MultiRankAdapter, MultiRankPolicy, RankOneAdapter},
    };
    use crate::{Decoder, Result};

    fn f32_tensor(name: &str, dims: &[usize], values: &[f32]) -> Tensor {
        Tensor {
            name: name.to_owned(),
            dimensions: dims.to_vec(),
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

    fn build_packed_block(kind: u32, r: usize, b: usize, seed: usize) -> Result<Vec<u8>> {
        let scale_bits = match (r + b + seed) % 5 {
            0 => 0x3555_u16,
            1 => 0xb400_u16,
            2 => 0x3400_u16,
            3 => 0xb555_u16,
            _ => 0x3200_u16,
        };
        let mut pq2_block = [0_u8; 34];
        pq2_block[..2].copy_from_slice(&scale_bits.to_le_bytes());
        for (byte_idx, byte) in pq2_block[2..].iter_mut().enumerate() {
            let v = (r * 31 + b * 17 + byte_idx * 13 + seed * 7) % 256;
            *byte = u8::try_from(v)?;
        }
        if kind == PQ2 {
            return Ok(pq2_block.to_vec());
        }
        for byte in &mut pq2_block[2..] {
            // Restrict 2-bit symbols to {0 (-1), 1 (0), 2 (+1)} (no code 3) for pack_ptq.
            let mut masked = 0_u8;
            for lane in 0..4 {
                let sym = ((*byte >> (lane * 2)) & 3) % 3;
                masked |= sym << (lane * 2);
            }
            *byte = masked;
        }
        Ok(bonsai_ternary::pack_ptq(&pq2_block)?.to_vec())
    }

    fn packed_ffn_tensor(
        name: &str,
        kind: u32,
        cols: usize,
        rows: usize,
        seed: usize,
    ) -> Result<Tensor> {
        const TEMPLATE_PHASES: usize = 17;
        let blocks_per_row = cols / 128;
        let block_bytes = if kind == PQ2 { 34 } else { 28 };
        let row_bytes = blocks_per_row * block_bytes;
        let mut row_templates = Vec::new();
        row_templates.try_reserve_exact(TEMPLATE_PHASES)?;
        for phase in 0..TEMPLATE_PHASES {
            let mut row = Vec::new();
            row.try_reserve_exact(row_bytes)?;
            for b in 0..blocks_per_row {
                row.extend_from_slice(&build_packed_block(kind, phase, b, seed)?);
            }
            row_templates.push(row);
        }
        let mut raw = Vec::new();
        raw.try_reserve_exact(rows * row_bytes)?;
        for r in 0..rows {
            let phase = ((r / COMPACTED_FFN_GROUP_WIDTH) * 3 + r) % TEMPLATE_PHASES;
            raw.extend_from_slice(&row_templates[phase]);
        }
        Ok(Tensor {
            name: name.to_owned(),
            dimensions: vec![cols, rows],
            kind,
            data: Arc::<[u8]>::from(raw).into(),
        })
    }

    fn insert_swift_layer_tensors(
        tensors: &mut BTreeMap<String, Tensor>,
        layer: usize,
        ffn_kind: u32,
        norm_vals: &[f32],
    ) -> Result<()> {
        let hidden = COMPACTED_FFN_HIDDEN_WIDTH;
        let ffn = COMPACTED_FFN_ORIGINAL_INTERMEDIATE_WIDTH;
        let head = 64_usize;
        let prefix = format!("blk.{layer}.");
        let attn_q_vals: Vec<f32> = (0..hidden * (2 * head))
            .map(|i| [0.03125_f32, -0.015_625, 0.0625, -0.03125][(i + layer) % 4])
            .collect();
        let attn_kv_vals: Vec<f32> = (0..hidden * head)
            .map(|i| [-0.015_625_f32, 0.03125, 0.0625, -0.0625][(i + layer) % 4])
            .collect();
        let attn_out_vals: Vec<f32> = (0..head * hidden)
            .map(|i| [0.0625_f32, -0.03125, 0.015_625, -0.0625][(i + layer) % 4])
            .collect();
        let head_ones = vec![1.0_f32; head];

        for (suffix, dims, vals) in [
            ("attn_norm.weight", &[hidden][..], norm_vals),
            ("post_attention_norm.weight", &[hidden][..], norm_vals),
            ("attn_q.weight", &[hidden, 2 * head][..], &attn_q_vals),
            ("attn_k.weight", &[hidden, head][..], &attn_kv_vals),
            ("attn_v.weight", &[hidden, head][..], &attn_kv_vals),
            ("attn_output.weight", &[head, hidden][..], &attn_out_vals),
            ("attn_q_norm.weight", &[head][..], &head_ones),
            ("attn_k_norm.weight", &[head][..], &head_ones),
        ] {
            let name = format!("{prefix}{suffix}");
            tensors.insert(name.clone(), f32_tensor(&name, dims, vals));
        }

        for (suffix, rows, cols, seed_base) in [
            ("ffn_up.weight", hidden, ffn, 11),
            ("ffn_gate.weight", hidden, ffn, 29),
            ("ffn_down.weight", ffn, hidden, 47),
        ] {
            let name = format!("{prefix}{suffix}");
            let tensor = packed_ffn_tensor(&name, ffn_kind, rows, cols, seed_base + layer * 13)?;
            tensors.insert(name, tensor);
        }
        Ok(())
    }

    fn insert_swift_hadamard_metadata(
        metadata: &mut BTreeMap<String, MetadataValue>,
        block_count: usize,
    ) -> Result<()> {
        let hidden = COMPACTED_FFN_HIDDEN_WIDTH;
        let ffn = COMPACTED_FFN_ORIGINAL_INTERMEDIATE_WIDTH;
        let sign_values: Vec<i32> = [hidden, 6_144, ffn]
            .into_iter()
            .flat_map(|w| {
                (0..w).map(|i| {
                    if (i * 7 + i / 1024 * 3) % 5 == 0 {
                        -1
                    } else {
                        1
                    }
                })
            })
            .collect();
        let weight_names: Vec<String> = (0..block_count)
            .map(|layer| format!("blk.{layer}.ffn_down.weight"))
            .collect();
        metadata.insert("prism.hadamard.version".into(), MetadataValue::from(1_u32));
        metadata.insert(
            "prism.hadamard.block_size".into(),
            MetadataValue::from(u32::try_from(COMPACTED_FFN_GROUP_WIDTH)?),
        );
        metadata.insert(
            "prism.hadamard.transform".into(),
            MetadataValue::from("normalized-sylvester-walsh-hadamard"),
        );
        metadata.insert(
            "prism.hadamard.axis".into(),
            MetadataValue::from("input-last-dimension"),
        );
        metadata.insert(
            "prism.hadamard.sign_mode".into(),
            MetadataValue::from("explicit"),
        );
        metadata.insert(
            "prism.hadamard.weight_names".into(),
            MetadataValue::from(weight_names),
        );
        metadata.insert(
            "prism.hadamard.sign_widths".into(),
            MetadataValue::from(vec![5_120_u32, 6_144_u32, 17_408_u32]),
        );
        metadata.insert(
            "prism.hadamard.sign_values".into(),
            MetadataValue::from(sign_values),
        );
        Ok(())
    }

    fn swift_ffn_model_metadata(block_count: usize) -> Result<BTreeMap<String, MetadataValue>> {
        let hidden = COMPACTED_FFN_HIDDEN_WIDTH;
        let ffn = COMPACTED_FFN_ORIGINAL_INTERMEDIATE_WIDTH;
        let head = 64_usize;
        let mut metadata = BTreeMap::from([
            ("general.architecture".into(), MetadataValue::from("qwen35")),
            (
                "qwen35.block_count".into(),
                MetadataValue::from(u32::try_from(block_count)?),
            ),
            ("qwen35.context_length".into(), MetadataValue::from(16_u32)),
            (
                "qwen35.embedding_length".into(),
                MetadataValue::from(u32::try_from(hidden)?),
            ),
            (
                "qwen35.feed_forward_length".into(),
                MetadataValue::from(u32::try_from(ffn)?),
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
                MetadataValue::from(u32::try_from(head)?),
            ),
            (
                "qwen35.attention.value_length".into(),
                MetadataValue::from(u32::try_from(head)?),
            ),
            (
                "qwen35.rope.dimension_count".into(),
                MetadataValue::from(32_u32),
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
                MetadataValue::from(1_u32),
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
                MetadataValue::from(vec![16_u32, 0, 0, 0]),
            ),
            (
                "qwen35.attention.recurrent_layers".into(),
                MetadataValue::from(vec![false; block_count]),
            ),
        ]);
        insert_swift_hadamard_metadata(&mut metadata, block_count)?;
        Ok(metadata)
    }

    fn swift_ffn_model_fixture(ffn_kind: u32, block_count: usize) -> Result<Model> {
        let hidden = COMPACTED_FFN_HIDDEN_WIDTH;
        let vocab = 4_usize;
        let metadata = swift_ffn_model_metadata(block_count)?;

        let emb_vals: Vec<f32> = (0..hidden * vocab)
            .map(|i| [-0.5_f32, 0.25, -0.125, 0.75, 0.375][i % 5])
            .collect();
        let out_vals: Vec<f32> = (0..hidden * vocab)
            .map(|i| [0.125_f32, -0.25, 0.5, -0.0625, 0.25][i % 5])
            .collect();
        let norm_vals: Vec<f32> = (0..hidden)
            .map(|i| [1.0_f32, 0.875, 1.125, 0.75][i % 4])
            .collect();

        let mut tensors = BTreeMap::from([
            (
                "token_embd.weight".into(),
                f32_tensor("token_embd.weight", &[hidden, vocab], &emb_vals),
            ),
            (
                "output.weight".into(),
                f32_tensor("output.weight", &[hidden, vocab], &out_vals),
            ),
            (
                "output_norm.weight".into(),
                f32_tensor("output_norm.weight", &[hidden], &norm_vals),
            ),
        ]);
        for layer in 0..block_count {
            insert_swift_layer_tensors(&mut tensors, layer, ffn_kind, &norm_vals)?;
        }

        Ok(Model {
            metadata,
            tensors,
            identity: format!("swift-compaction-fixture-{ffn_kind}-{block_count}"),
        })
    }

    fn down_rank_one_adapter(base: &Model, layer: usize) -> Result<RankOneAdapter> {
        let target = format!("blk.{layer}.ffn_down.weight");
        let a_vals: Vec<f32> = (0..COMPACTED_FFN_ORIGINAL_INTERMEDIATE_WIDTH)
            .map(|i| [-0.0625_f32, 0.125, -0.03125, 0.0625][i % 4])
            .collect();
        let b_vals: Vec<f32> = (0..COMPACTED_FFN_HIDDEN_WIDTH)
            .map(|i| [0.03125_f32, -0.015_625, 0.0625, -0.03125][i % 4])
            .collect();
        let factors = Model {
            metadata: BTreeMap::from([
                ("general.architecture".into(), MetadataValue::from("qwen35")),
                ("general.type".into(), MetadataValue::from("adapter")),
                ("adapter.type".into(), MetadataValue::from("lora")),
                ("adapter.lora_alpha".into(), MetadataValue::from(1.0_f32)),
            ]),
            tensors: BTreeMap::from([
                (
                    format!("{target}.lora_a"),
                    f32_tensor(
                        &format!("{target}.lora_a"),
                        &[COMPACTED_FFN_ORIGINAL_INTERMEDIATE_WIDTH, 1],
                        &a_vals,
                    ),
                ),
                (
                    format!("{target}.lora_b"),
                    f32_tensor(
                        &format!("{target}.lora_b"),
                        &[1, COMPACTED_FFN_HIDDEN_WIDTH],
                        &b_vals,
                    ),
                ),
            ]),
            identity: "swift-compaction-r1-adapter".into(),
        };
        RankOneAdapter::from_model(factors, base)
    }

    fn ffn_compensation() -> Result<FfnCompensation> {
        let a = vec![
            (0..COMPACTED_FFN_HIDDEN_WIDTH)
                .map(|i| [0.03125_f32, -0.015_625, 0.0625][i % 3])
                .collect(),
            (0..COMPACTED_FFN_HIDDEN_WIDTH)
                .map(|i| [-0.015_625_f32, 0.03125, -0.0625][i % 3])
                .collect(),
        ];
        let b = vec![
            (0..COMPACTED_FFN_HIDDEN_WIDTH)
                .map(|i| [0.0625_f32, -0.03125, 0.015_625][i % 3])
                .collect(),
            (0..COMPACTED_FFN_HIDDEN_WIDTH)
                .map(|i| [-0.03125_f32, 0.015_625, 0.0625][i % 3])
                .collect(),
        ];
        FfnCompensation::new(a, b)
    }

    #[test]
    fn plan_and_tensor_compaction_primitives_validate_and_normalize() -> Result<()> {
        assert!(CompactedFfnPlan::new(Vec::new()).is_err());
        assert!(
            CompactedFfnPlan::new(vec![LayerFfnCompaction {
                layer: 0,
                omitted_groups: Vec::new(),
                compensation: None,
            }])
            .is_err()
        );
        assert!(
            CompactedFfnPlan::new(vec![LayerFfnCompaction {
                layer: 0,
                omitted_groups: (0..17).collect(),
                compensation: None,
            }])
            .is_err()
        );
        assert!(
            CompactedFfnPlan::new(vec![LayerFfnCompaction {
                layer: 0,
                omitted_groups: vec![3, 1],
                compensation: None,
            }])
            .is_err()
        );
        assert!(
            CompactedFfnPlan::new(vec![LayerFfnCompaction {
                layer: 0,
                omitted_groups: vec![2, 2],
                compensation: None,
            }])
            .is_err()
        );
        assert!(
            CompactedFfnPlan::new(vec![LayerFfnCompaction {
                layer: 0,
                omitted_groups: vec![17],
                compensation: None,
            }])
            .is_err()
        );
        assert!(
            CompactedFfnPlan::new(vec![
                LayerFfnCompaction {
                    layer: 1,
                    omitted_groups: vec![0],
                    compensation: None,
                },
                LayerFfnCompaction {
                    layer: 1,
                    omitted_groups: vec![2],
                    compensation: None,
                },
            ])
            .is_err()
        );

        let plan_a = CompactedFfnPlan::new(vec![
            LayerFfnCompaction {
                layer: 3,
                omitted_groups: vec![1, 5],
                compensation: None,
            },
            LayerFfnCompaction {
                layer: 0,
                omitted_groups: vec![0, 16],
                compensation: Some(ffn_compensation()?),
            },
        ])?;
        let plan_b = CompactedFfnPlan::new(vec![
            LayerFfnCompaction {
                layer: 0,
                omitted_groups: vec![0, 16],
                compensation: Some(ffn_compensation()?),
            },
            LayerFfnCompaction {
                layer: 3,
                omitted_groups: vec![1, 5],
                compensation: None,
            },
        ])?;
        assert_eq!(plan_a.layers()[0].layer, 0);
        assert_eq!(plan_a.layers()[1].layer, 3);
        assert_eq!(
            plan_identity_sha256(&plan_a, "abc")?,
            plan_identity_sha256(&plan_b, "abc")?
        );
        assert_ne!(
            plan_identity_sha256(&plan_a, "abc")?,
            plan_identity_sha256(&plan_b, "def")?
        );
        Ok(())
    }

    #[test]
    fn tensor_compaction_preserves_surviving_bits_and_signs() -> Result<()> {
        // Verify F32 tensor compaction preserves exact bits on surviving rows/cols.
        let omitted = [0_usize, 5, 16];
        let mut up_vals = Vec::with_capacity(256 * 17_408);
        for row in 0..17_408_usize {
            for col in 0..256_usize {
                up_vals.push(f32::from(u16::try_from((row * 7 + col) % 1000)?));
            }
        }
        let up_f32 = f32_tensor("up", &[256, 17_408], &up_vals);
        let up_comp = compact_gate_up_rows(&up_f32, &omitted)?;
        assert_eq!(up_comp.dimensions, [256, 14 * 1024]);
        let up_bytes = up_comp.data.as_chunks::<4>().0;
        let mut retained_row = 0_usize;
        for group in 0..17 {
            if !omitted.contains(&group) {
                for r in group * 1024..(group + 1) * 1024 {
                    for c in 0..256 {
                        let actual = f32::from_le_bytes(up_bytes[retained_row * 256 + c]);
                        assert_eq!(actual.to_bits(), up_vals[r * 256 + c].to_bits());
                    }
                    retained_row += 1;
                }
            }
        }

        let mut down_vals = Vec::with_capacity(17_408 * 4);
        for row in 0..4_usize {
            for col in 0..17_408_usize {
                down_vals.push(f32::from(u16::try_from((row * 13 + col) % 1000)?));
            }
        }
        let down_f32 = f32_tensor("down", &[17_408, 4], &down_vals);
        let down_comp = compact_down_cols(&down_f32, &omitted)?;
        assert_eq!(down_comp.dimensions, [14 * 1024, 4]);
        let down_bytes = down_comp.data.as_chunks::<4>().0;
        for row in 0..4 {
            let mut retained_col = 0_usize;
            for group in 0..17 {
                if !omitted.contains(&group) {
                    for c in group * 1024..(group + 1) * 1024 {
                        let actual =
                            f32::from_le_bytes(down_bytes[row * (14 * 1024) + retained_col]);
                        assert_eq!(actual.to_bits(), down_vals[row * 17_408 + c].to_bits());
                        retained_col += 1;
                    }
                }
            }
        }

        let signs: Vec<f32> = (0..17_408)
            .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
            .collect();
        let comp_signs = compact_signs(&signs, &omitted)?;
        assert_eq!(comp_signs.len(), 14 * 1024);
        Ok(())
    }

    fn assert_compacted_parity_case(
        ffn_kind: u32,
        omitted: Vec<usize>,
        with_compensation: bool,
    ) -> Result<()> {
        let comp_probe = with_compensation.then(ffn_compensation).transpose()?;
        let comp_compact = with_compensation.then(ffn_compensation).transpose()?;

        let probe_model = swift_ffn_model_fixture(ffn_kind, 1)?;
        let probe_adapter = down_rank_one_adapter(&probe_model, 0)?;
        let probe_graph = Qwen35::with_rank_one_adapter(
            probe_model,
            8,
            crate::graph::MAX_STATE_BYTES,
            MatvecBackend::Scalar,
            probe_adapter,
            1.3,
        )?;
        let mut probe =
            FfnProbe::new(probe_graph, 0, None)?.with_group_omission(omitted.clone())?;
        if let Some(comp) = comp_probe {
            probe = probe.with_group_compensation(comp)?;
        }

        let compact_model = swift_ffn_model_fixture(ffn_kind, 1)?;
        let compact_adapter = down_rank_one_adapter(&compact_model, 0)?;
        let compact_base = Qwen35::with_rank_one_adapter_and_kv(
            compact_model,
            8,
            crate::graph::MAX_STATE_BYTES,
            MatvecBackend::Scalar,
            compact_adapter,
            1.3,
            KvCachePolicy::F16,
        )?;
        let pre_compact_ckpt = compact_base.checkpoint()?;
        let plan = CompactedFfnPlan::new(vec![LayerFfnCompaction {
            layer: 0,
            omitted_groups: omitted,
            compensation: comp_compact,
        }])?;
        let mut compacted = compact_base.with_compacted_ffn(plan)?;
        assert!(compacted.compacted_ffn_saved_bytes() > 0);
        assert!(compacted.identity().contains(":w8-compacted-ffn-v1:"));
        assert!(compacted.restore(&pre_compact_ckpt).is_err());

        let mut expected_logits_0 = Vec::new();
        probe.evaluate_into(0, &mut expected_logits_0)?;
        let mut actual_logits_0 = Vec::new();
        compacted.evaluate_into(0, &mut actual_logits_0)?;
        assert_bits_eq(&actual_logits_0, &expected_logits_0, "token 0");

        let ckpt = compacted.checkpoint()?;
        let mut decode_1 = Vec::new();
        compacted.evaluate_into(1, &mut decode_1)?;
        let mut decode_2 = Vec::new();
        compacted.evaluate_into(2, &mut decode_2)?;
        compacted.restore(&ckpt)?;
        let mut replayed_1 = Vec::new();
        compacted.evaluate_into(1, &mut replayed_1)?;
        assert_bits_eq(&replayed_1, &decode_1, "restore replay token 1");

        // Multi-token tiled prefill matches sequential decode at 0-ULP.
        compacted.reset()?;
        let scratch_bytes = compacted.tiled_prefill_scratch_bytes(8)?;
        let mut tiled = compacted.with_tiled_prefill(8, scratch_bytes)?;
        assert!(tiled.restore(&ckpt).is_err());
        tiled.ingest_block(&[0, 1], &|| Ok(()))?;
        let mut tiled_2 = Vec::new();
        tiled.evaluate_into(2, &mut tiled_2)?;
        assert_bits_eq(&tiled_2, &decode_2, "tiled ingest [0,1] -> token 2");
        Ok(())
    }

    fn assert_bits_eq(actual: &[f32], expected: &[f32], label: &str) {
        let actual_bits: Vec<u32> = actual.iter().copied().map(f32::to_bits).collect();
        let expected_bits: Vec<u32> = expected.iter().copied().map(f32::to_bits).collect();
        assert_eq!(actual_bits, expected_bits, "0-ULP mismatch at {label}");
    }

    #[test]
    fn compacted_ffn_pq2_with_compensation_and_adapter_matches_probe_at_zero_ulp() -> Result<()> {
        // Leaving 5 groups (5,120) also verifies no width collision with global 5,120 signs.
        assert_compacted_parity_case(PQ2, (0..12).collect(), true)
    }

    #[test]
    fn compacted_ffn_ptq1_without_compensation_and_adapter_matches_probe_at_zero_ulp() -> Result<()>
    {
        assert_compacted_parity_case(
            PTQ1,
            vec![0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16],
            false,
        )
    }

    #[test]
    fn compacted_ffn_rejects_cross_profile_checkpoints_multi_rank_and_rollback() -> Result<()> {
        let model_a = swift_ffn_model_fixture(PQ2, 2)?;
        let adapter_a = down_rank_one_adapter(&model_a, 0)?;
        let model_b = swift_ffn_model_fixture(PQ2, 2)?;
        let adapter_b = down_rank_one_adapter(&model_b, 0)?;

        // Fail-closed: MultiRankAdapter on ffn_gate of compacted layer 0 must reject.
        let gate_target = "blk.0.ffn_gate.weight";
        let gate_mr_model = Model {
            metadata: BTreeMap::from([
                ("general.architecture".into(), MetadataValue::from("qwen35")),
                ("general.type".into(), MetadataValue::from("adapter")),
                ("adapter.type".into(), MetadataValue::from("lora")),
                ("adapter.lora_alpha".into(), MetadataValue::from(1.0_f32)),
            ]),
            tensors: BTreeMap::from([
                (
                    format!("{gate_target}.lora_a"),
                    f32_tensor(
                        &format!("{gate_target}.lora_a"),
                        &[COMPACTED_FFN_HIDDEN_WIDTH, 2],
                        &vec![0.0625_f32; COMPACTED_FFN_HIDDEN_WIDTH * 2],
                    ),
                ),
                (
                    format!("{gate_target}.lora_b"),
                    f32_tensor(
                        &format!("{gate_target}.lora_b"),
                        &[2, COMPACTED_FFN_ORIGINAL_INTERMEDIATE_WIDTH],
                        &vec![0.03125_f32; 2 * COMPACTED_FFN_ORIGINAL_INTERMEDIATE_WIDTH],
                    ),
                ),
            ]),
            identity: "swift-compaction-mr-gate-reject".into(),
        };
        let gate_mr =
            MultiRankAdapter::from_model(&model_a, &gate_mr_model, MultiRankPolicy::AnyLinear)?;
        let plan_gate_reject = CompactedFfnPlan::new(vec![LayerFfnCompaction {
            layer: 0,
            omitted_groups: vec![0, 2],
            compensation: None,
        }])?;
        assert!(
            Qwen35::with_adapters_and_kv(
                swift_ffn_model_fixture(PQ2, 2)?,
                8,
                crate::graph::MAX_STATE_BYTES,
                MatvecBackend::Scalar,
                None,
                Some((gate_mr, 1.0)),
                KvCachePolicy::F16,
            )?
            .with_compacted_ffn(plan_gate_reject)
            .is_err()
        );

        let plan_a = CompactedFfnPlan::new(vec![LayerFfnCompaction {
            layer: 0,
            omitted_groups: (0..15).collect(),
            compensation: None,
        }])?;
        let plan_b = CompactedFfnPlan::new(vec![LayerFfnCompaction {
            layer: 0,
            omitted_groups: vec![0, 3],
            compensation: None,
        }])?;

        let mut graph_a = Qwen35::with_rank_one_adapter(
            model_a,
            8,
            crate::graph::MAX_STATE_BYTES,
            MatvecBackend::Scalar,
            adapter_a,
            1.3,
        )?
        .with_compacted_ffn(plan_a)?;
        let mut graph_b = Qwen35::with_rank_one_adapter(
            model_b,
            8,
            crate::graph::MAX_STATE_BYTES,
            MatvecBackend::Scalar,
            adapter_b,
            1.3,
        )?
        .with_compacted_ffn(plan_b)?;
        assert_ne!(graph_a.identity(), graph_b.identity());

        let ckpt_empty = graph_a.checkpoint()?;
        assert!(graph_b.restore(&ckpt_empty).is_err());
        let original_id = graph_a.identity().to_owned();
        graph_a.reset_with_adapter_scale(0.5)?;
        assert!(graph_a.identity().contains(":w8-compacted-ffn-v1:"));
        assert!(graph_a.restore(&ckpt_empty).is_err());
        graph_a.reset_with_adapter_scale(1.3)?;
        assert_eq!(graph_a.identity(), original_id);
        assert!(graph_a.restore(&ckpt_empty).is_err());
        Ok(())
    }
}
