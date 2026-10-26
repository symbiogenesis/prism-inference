//! Exact Swift model admission and retained-storage accounting.
//!
//! PQ2 remains the service MVP default. The explicit current-weight PTQ1 lane
//! permits correctness captures; it does not qualify the service or adapters.
//! Full-model numerical and process-memory qualification remain separate.

use std::collections::BTreeMap;
use std::path::Path;

use bonsai_gguf::{
    Metadata, MetadataArray, MetadataValue, MetadataValueRef, PRISM_DIALECT, TensorDescriptor,
    VerifiedModel,
};
use sha2::{Digest, Sha256};

use super::Model;
use crate::{Result, invalid};

/// Exact byte length from `config/swift-assessment.lock.json`.
pub const MODEL_BYTES: u64 = 7_206_168_928;
/// Immutable pinned Swift PQ2 artifact identity.
pub const MODEL_SHA256: &str = "5912bb739217cf25b4283a098c7b893d9baff36bdf2e72d3dbf4cb999d5633d5";
/// Exact verified current-weight PTQ1 export length, not the older publisher model.
pub const REPACKED_PTQ1_BYTES: u64 = 5_946_648_928;
/// Immutable current-weight export identity from Runner's PTQ1 export receipt.
pub const REPACKED_PTQ1_SHA256: &str =
    "e658f0f0736a03a4549535a439f9b07254ebf8ab88ae27cbe6e52d098ca5d1f5";
/// Exact embedded UTF-8 template identity in the verified artifact.
pub const TEMPLATE_SHA256: &str =
    "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041";
/// Number of declared input and output vocabulary entries.
pub const VOCAB_SIZE: usize = 248_320;
/// Total conversation context admitted by the Swift MVP.
pub const CONTEXT_TOKENS: usize = 8_192;
/// Maximum complete output allowance, including reasoning.
pub const MAX_OUTPUT_TOKENS: usize = 4_096;
/// Exact serialized tensor payload bytes, excluding header and alignment gaps.
pub const TENSOR_PAYLOAD_BYTES: usize = 7_195_047_936;
/// Maximum live FP16 KV and F32 recurrent/convolution payload at 8K context.
pub const STATE_PAYLOAD_BYTES: usize = 693_764_096;

const SIGN_SHA256: &str = "3d1ea9136a6044cd680cc1e92cde65e9dd7238e1288fe2effabec5d201c48bed";

/// Exact current-weight artifact selected for an offline correctness capture.
///
/// Both choices retain the same decoded weights. Their native activation
/// quantizers differ, so this selection does not promise equal model outputs.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Representation {
    /// Original pinned current-weight PQ2 artifact and service default.
    #[default]
    Pq2,
    /// Verified lossless repack of the current PQ2 weights, not older PTQ weights.
    CurrentRepackedPtq1,
}

impl Representation {
    /// Stable evidence identity distinguishing the source from its exact export.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Pq2 => "PQ2_0",
            Self::CurrentRepackedPtq1 => "current-repacked-PTQ1_0",
        }
    }

    /// Complete immutable artifact length, including GGUF metadata and alignment.
    #[must_use]
    pub const fn model_bytes(self) -> u64 {
        match self {
            Self::Pq2 => MODEL_BYTES,
            Self::CurrentRepackedPtq1 => REPACKED_PTQ1_BYTES,
        }
    }

    /// SHA-256 of the only artifact admitted by this representation.
    #[must_use]
    pub const fn model_sha256(self) -> &'static str {
        match self {
            Self::Pq2 => MODEL_SHA256,
            Self::CurrentRepackedPtq1 => REPACKED_PTQ1_SHA256,
        }
    }

    const fn tensor_kind(self) -> u32 {
        match self {
            Self::Pq2 => 142,
            Self::CurrentRepackedPtq1 => 143,
        }
    }

    const fn file_type(self) -> u64 {
        match self {
            Self::Pq2 => 141,
            Self::CurrentRepackedPtq1 => 143,
        }
    }

    const fn tensor_payload_bytes(self) -> usize {
        match self {
            Self::Pq2 => TENSOR_PAYLOAD_BYTES,
            Self::CurrentRepackedPtq1 => 5_935_527_936,
        }
    }
}

/// Conservative loader accounting; this is not a process resident-set verdict.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoadReceipt {
    /// Conservative parser charge for retained metadata and descriptors.
    pub header_charge_bytes: usize,
    /// Exact immutable serialized tensor payload retained by the model.
    pub tensor_payload_bytes: usize,
    /// Header charge plus all retained serialized tensor payloads.
    pub retained_storage_bytes: usize,
    /// Largest tensor; shared-storage conversion can transiently copy this much.
    pub largest_tensor_bytes: usize,
    /// Counts in F32, BF16 and selected packed-representation order.
    pub tensor_counts: [usize; 3],
}

/// Loads only the pinned Swift artifact through one immutable verified handle.
///
/// Complete inventory, geometry, tokenizer/template and transform declarations
/// are checked before allocating tensor payloads. `memory_limit` covers the
/// conservative retained-header charge and all payloads. Conversion of one
/// tensor to immutable shared storage can transiently allocate a second copy;
/// graph state, scratch, checkpoints and allocator overhead are separate.
///
/// # Errors
/// Rejects artifact drift, unsupported/malformed metadata, any extra or missing
/// tensor, encoding/dimension drift, insufficient storage budgets and I/O errors.
pub fn load(path: &Path, memory_limit: usize) -> Result<(Model, LoadReceipt)> {
    load_representation(path, memory_limit, Representation::Pq2)
}

/// Loads the exact selected current-weight representation for correctness checks.
///
/// This retains the same strict metadata, tensor geometry and storage budget as
/// [`load`]. PTQ1 selection admits only the verified current-weight export and
/// does not enable a service profile, adapters or performance qualification.
///
/// # Errors
/// Rejects changed artifacts, mixed representations, inventory/metadata drift,
/// insufficient retained-storage budgets and invalid executable tensor values.
pub fn load_representation(
    path: &Path,
    memory_limit: usize,
    representation: Representation,
) -> Result<(Model, LoadReceipt)> {
    let (reader, receipt) = verify(path, memory_limit, representation)?;
    let model = Model::load_verified(
        reader,
        representation.model_sha256(),
        memory_limit,
        &mut |_| Ok(true),
    )?;
    Ok((model, receipt))
}

/// Verifies the full artifact and its strict header without retaining weights.
///
/// The returned receipt describes storage a subsequent load would retain. It is
/// not evidence that tensor payloads have been loaded or executable values are
/// valid. Hashing still reads the entire pinned file.
///
/// # Errors
/// Rejects changed identity, declarations, inventory or the model storage budget.
pub fn inspect(path: &Path, memory_limit: usize) -> Result<LoadReceipt> {
    inspect_representation(path, memory_limit, Representation::Pq2)
}

/// Verifies only the pinned artifact and strict header of an explicit representation.
///
/// As with [`inspect`], hashing reads the complete file without retaining weights;
/// a successful receipt is not a numerical, service or memory-residency verdict.
///
/// # Errors
/// Rejects changed identity, declarations, inventory or the model storage budget.
pub fn inspect_representation(
    path: &Path,
    memory_limit: usize,
    representation: Representation,
) -> Result<LoadReceipt> {
    Ok(verify(path, memory_limit, representation)?.1)
}

fn verify(
    path: &Path,
    memory_limit: usize,
    representation: Representation,
) -> Result<(VerifiedModel, LoadReceipt)> {
    let reader = VerifiedModel::open(
        path,
        PRISM_DIALECT,
        representation.model_bytes(),
        representation.model_sha256(),
        memory_limit,
    )?;
    validate_metadata_for(reader.metadata(), representation)?;
    let (tensor_counts, tensor_payload_bytes, largest_tensor_bytes) =
        validate_inventory(reader.tensors(), representation)?;
    let header_charge_bytes = reader.retained_bytes();
    let retained_storage_bytes = header_charge_bytes
        .checked_add(tensor_payload_bytes)
        .ok_or_else(|| invalid("Swift retained storage accounting overflow"))?;
    if retained_storage_bytes > memory_limit {
        return Err(invalid("Swift retained model storage exceeds its budget"));
    }
    Ok((
        reader,
        LoadReceipt {
            header_charge_bytes,
            tensor_payload_bytes,
            retained_storage_bytes,
            largest_tensor_bytes,
            tensor_counts,
        },
    ))
}

/// Checks the pinned Swift metadata contract without executing or loading weights.
///
/// This checks declarations, not artifact authenticity. Callers must obtain the
/// metadata through a loader that verifies [`MODEL_BYTES`] and [`MODEL_SHA256`].
/// That complete file hash also binds every tokenizer string/merge and metadata
/// record to the raw-tokenizer digest recorded in `config/swift.metadata.json`.
///
/// # Errors
/// Rejects missing, extra or changed pinned metadata declarations.
pub fn validate_metadata(metadata: &Metadata) -> Result<()> {
    validate_metadata_for(metadata, Representation::Pq2)
}

#[allow(
    clippy::too_many_lines,
    reason = "Keep the exact shared metadata contract and representation discriminator together."
)]
fn validate_metadata_for(metadata: &Metadata, representation: Representation) -> Result<()> {
    if metadata.len() != 49 {
        return Err(invalid("Swift metadata inventory mismatch"));
    }
    for (key, value) in [
        ("general.architecture", "qwen35"),
        ("general.type", "model"),
        ("general.name", "Hf"),
        ("general.version", "v5"),
        ("general.basename", "folded"),
        ("general.size_label", "27B"),
        (
            "prism.hadamard.transform",
            "normalized-sylvester-walsh-hadamard",
        ),
        ("prism.hadamard.axis", "input-last-dimension"),
        ("prism.hadamard.sign_mode", "explicit"),
        ("tokenizer.ggml.model", "gpt2"),
        ("tokenizer.ggml.pre", "qwen35"),
    ] {
        if metadata.get(key).and_then(MetadataValue::as_str) != Some(value) {
            return Err(invalid(format!("Swift metadata mismatch: {key}")));
        }
    }
    for (key, value) in [
        ("general.sampling.top_k", 20),
        ("qwen35.block_count", 64),
        ("qwen35.context_length", 262_144),
        ("qwen35.embedding_length", 5_120),
        ("qwen35.feed_forward_length", 17_408),
        ("qwen35.attention.head_count", 24),
        ("qwen35.attention.head_count_kv", 4),
        ("qwen35.attention.key_length", 256),
        ("qwen35.attention.value_length", 256),
        ("qwen35.ssm.conv_kernel", 4),
        ("qwen35.ssm.state_size", 128),
        ("qwen35.ssm.group_count", 16),
        ("qwen35.ssm.time_step_rank", 48),
        ("qwen35.ssm.inner_size", 6_144),
        ("qwen35.full_attention_interval", 4),
        ("qwen35.rope.dimension_count", 64),
        ("prism.hadamard.version", 1),
        ("prism.hadamard.block_size", 1_024),
        ("tokenizer.ggml.eos_token_id", 248_046),
        ("tokenizer.ggml.padding_token_id", 248_044),
        ("tokenizer.ggml.bos_token_id", 248_044),
        ("general.quantization_version", 2),
        ("general.file_type", representation.file_type()),
    ] {
        if metadata.get(key).and_then(MetadataValue::as_u64) != Some(value) {
            return Err(invalid(format!("Swift metadata mismatch: {key}")));
        }
    }
    for (key, value) in [
        ("general.sampling.top_p", 0.95_f32),
        ("general.sampling.temp", 1.0),
        ("qwen35.rope.freq_base", 10_000_000.0),
        ("qwen35.attention.layer_norm_rms_epsilon", 0.000_001),
    ] {
        if metadata
            .get(key)
            .and_then(MetadataValue::as_f64)
            .map(f64::to_bits)
            != Some(f64::from(value).to_bits())
        {
            return Err(invalid(format!("Swift metadata mismatch: {key}")));
        }
    }
    for (key, value) in [
        ("prism.hadamard.gdn_v_grouped", true),
        ("tokenizer.ggml.add_bos_token", false),
    ] {
        if metadata.get(key).and_then(MetadataValue::as_bool) != Some(value) {
            return Err(invalid(format!("Swift metadata mismatch: {key}")));
        }
    }
    let template = metadata
        .get("tokenizer.chat_template")
        .and_then(MetadataValue::as_str)
        .ok_or_else(|| invalid("Swift embedded template is missing"))?;
    if template.len() != 8_952 || digest([template.as_bytes()]) != TEMPLATE_SHA256 {
        return Err(invalid("Swift embedded template identity mismatch"));
    }
    check_integers(metadata, "qwen35.rope.dimension_sections", &[11, 11, 10, 0])?;
    check_integers(
        metadata,
        "prism.hadamard.sign_widths",
        &[5_120, 6_144, 17_408],
    )?;
    let signs = metadata
        .get("prism.hadamard.sign_values")
        .and_then(MetadataValue::as_array)
        .ok_or_else(|| invalid("Swift Hadamard signs are missing"))?;
    let mut sign_hash = Sha256::new();
    if signs.len() != 28_672 {
        return Err(invalid("Swift Hadamard sign count mismatch"));
    }
    for value in signs.iter() {
        let value = value
            .as_i64()
            .ok_or_else(|| invalid("Swift Hadamard signs must be integers"))?;
        if value != -1 && value != 1 {
            return Err(invalid("Swift Hadamard signs must be +1 or -1"));
        }
        sign_hash.update(i32::try_from(value)?.to_le_bytes());
    }
    if hex(sign_hash) != SIGN_SHA256 {
        return Err(invalid("Swift Hadamard sign identity mismatch"));
    }
    let expected = expected_tensors(representation);
    let transformed = metadata
        .get("prism.hadamard.weight_names")
        .and_then(MetadataValue::as_array)
        .ok_or_else(|| invalid("Swift Hadamard weight inventory missing"))?;
    let mut names = std::collections::BTreeSet::new();
    for value in transformed.iter() {
        let name = value
            .as_str()
            .ok_or_else(|| invalid("Swift transform names must be strings"))?;
        if name == "token_embd.weight"
            || expected
                .get(name)
                .is_none_or(|spec| spec.0 != representation.tensor_kind())
            || !names.insert(name)
        {
            return Err(invalid("Swift Hadamard weight inventory mismatch"));
        }
    }
    if names.len() != 401 {
        return Err(invalid("Swift Hadamard weight count mismatch"));
    }
    let inverse = metadata
        .get("prism.hadamard.inverse_weight_names")
        .and_then(MetadataValue::as_array)
        .ok_or_else(|| invalid("Swift inverse transform missing"))?;
    if inverse.len() != 1
        || inverse.get(0).and_then(MetadataValueRef::as_str) != Some("token_embd.weight")
    {
        return Err(invalid("Swift inverse embedding declaration mismatch"));
    }
    for (key, count, strings) in [
        ("tokenizer.ggml.tokens", VOCAB_SIZE, true),
        ("tokenizer.ggml.token_type", VOCAB_SIZE, false),
        ("tokenizer.ggml.merges", 247_587, true),
    ] {
        let array = metadata
            .get(key)
            .and_then(MetadataValue::as_array)
            .ok_or_else(|| invalid(format!("Swift array missing: {key}")))?;
        if array.len() != count
            || (strings && !matches!(array, MetadataArray::String(_)))
            || (!strings && !matches!(array, MetadataArray::I32(_)))
        {
            return Err(invalid(format!("Swift tokenizer array mismatch: {key}")));
        }
    }
    Ok(())
}

fn check_integers(metadata: &Metadata, key: &str, expected: &[u64]) -> Result<()> {
    let values = metadata
        .get(key)
        .and_then(MetadataValue::as_array)
        .ok_or_else(|| invalid(format!("Swift array missing: {key}")))?;
    if values.len() != expected.len()
        || !values
            .iter()
            .zip(expected)
            .all(|(a, b)| a.as_u64() == Some(*b))
    {
        return Err(invalid(format!("Swift metadata array mismatch: {key}")));
    }
    Ok(())
}

fn digest<'a>(parts: impl IntoIterator<Item = &'a [u8]>) -> String {
    let mut hash = Sha256::new();
    for part in parts {
        hash.update(part);
    }
    hex(hash)
}

fn hex(hash: Sha256) -> String {
    hash.finalize()
        .iter()
        .flat_map(|byte| [byte >> 4, byte & 15])
        .map(|nibble| char::from(b"0123456789abcdef"[usize::from(nibble)]))
        .collect()
}

fn expected_tensors(representation: Representation) -> BTreeMap<String, (u32, Vec<u64>)> {
    let mut tensors = BTreeMap::new();
    let packed = representation.tensor_kind();
    for name in ["token_embd.weight", "output.weight"] {
        tensors.insert(name.to_owned(), (packed, vec![5_120, 248_320]));
    }
    tensors.insert("output_norm.weight".to_owned(), (0, vec![5_120]));
    for layer in 0..64 {
        let mut add = |name: &str, kind, dimensions: &[u64]| {
            tensors.insert(format!("blk.{layer}.{name}"), (kind, dimensions.to_vec()));
        };
        for name in ["attn_norm.weight", "post_attention_norm.weight"] {
            add(name, 0, &[5_120]);
        }
        for name in ["ffn_up.weight", "ffn_gate.weight"] {
            add(name, packed, &[5_120, 17_408]);
        }
        add("ffn_down.weight", packed, &[17_408, 5_120]);
        if (layer + 1) % 4 == 0 {
            add("attn_q.weight", packed, &[5_120, 12_288]);
            for name in ["attn_k.weight", "attn_v.weight"] {
                add(name, packed, &[5_120, 1_024]);
            }
            add("attn_output.weight", packed, &[6_144, 5_120]);
            for name in ["attn_q_norm.weight", "attn_k_norm.weight"] {
                add(name, 0, &[256]);
            }
        } else {
            add("attn_qkv.weight", packed, &[5_120, 10_240]);
            add("attn_gate.weight", packed, &[5_120, 6_144]);
            add("ssm_out.weight", packed, &[6_144, 5_120]);
            for name in ["ssm_alpha.weight", "ssm_beta.weight"] {
                add(name, 30, &[5_120, 48]);
            }
            add("ssm_conv1d.weight", 0, &[4, 10_240]);
            add("ssm_norm.weight", 0, &[128]);
            for name in ["ssm_a", "ssm_dt.bias"] {
                add(name, 0, &[48]);
            }
        }
    }
    tensors
}

fn validate_inventory(
    tensors: &[TensorDescriptor],
    representation: Representation,
) -> Result<([usize; 3], usize, usize)> {
    validate_layouts(
        representation,
        tensors.len(),
        tensors.iter().map(|tensor| {
            (
                tensor.name.as_str(),
                tensor.dimensions.as_slice(),
                tensor.type_id,
                tensor.bytes,
            )
        }),
    )
}

fn validate_layouts<'a>(
    representation: Representation,
    count: usize,
    tensors: impl IntoIterator<Item = (&'a str, &'a [u64], u32, u64)>,
) -> Result<([usize; 3], usize, usize)> {
    let expected = expected_tensors(representation);
    if count != expected.len() {
        return Err(invalid("Swift tensor inventory count mismatch"));
    }
    let mut counts = [0; 3];
    let mut bytes = 0_usize;
    let mut largest = 0;
    for (name, dimensions, kind, payload_bytes) in tensors {
        if expected
            .get(name)
            .is_none_or(|(expected_kind, expected_dimensions)| {
                *expected_kind != kind || expected_dimensions != dimensions
            })
        {
            return Err(invalid(format!("Swift tensor layout mismatch: {name}")));
        }
        let slot = match kind {
            0 => 0,
            30 => 1,
            kind if kind == representation.tensor_kind() => 2,
            _ => return Err(invalid("Swift tensor encoding mismatch")),
        };
        counts[slot] += 1;
        let length = usize::try_from(payload_bytes)?;
        bytes = bytes
            .checked_add(length)
            .ok_or_else(|| invalid("Swift tensor accounting overflow"))?;
        largest = largest.max(length);
    }
    if counts != [353, 96, 402] || bytes != representation.tensor_payload_bytes() {
        return Err(invalid("Swift tensor payload inventory mismatch"));
    }
    Ok((counts, bytes, largest))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptors(representation: Representation) -> Vec<(String, Vec<u64>, u32, u64)> {
        expected_tensors(representation)
            .into_iter()
            .map(|(name, (type_id, dimensions))| {
                let elements = dimensions.iter().product::<u64>();
                let bytes = match type_id {
                    0 => elements * 4,
                    30 => elements * 2,
                    142 => elements / 256 * 68,
                    143 => elements / 128 * 28,
                    _ => 0,
                };
                (name, dimensions, type_id, bytes)
            })
            .collect()
    }

    #[test]
    fn inventory_covers_every_hybrid_block_and_exact_payload() -> Result<()> {
        let descriptors = descriptors(Representation::Pq2);
        assert_eq!(
            validate_layouts(
                Representation::Pq2,
                descriptors.len(),
                descriptors.iter().map(|(name, dimensions, kind, bytes)| (
                    name.as_str(),
                    dimensions.as_slice(),
                    *kind,
                    *bytes
                ))
            )?,
            ([353, 96, 402], TENSOR_PAYLOAD_BYTES, 337_715_200)
        );
        let mut descriptors = descriptors;
        descriptors[0].1[0] += 1;
        assert!(
            validate_layouts(
                Representation::Pq2,
                descriptors.len(),
                descriptors.iter().map(|(name, dimensions, kind, bytes)| (
                    name.as_str(),
                    dimensions.as_slice(),
                    *kind,
                    *bytes
                ))
            )
            .is_err()
        );
        descriptors.pop();
        assert!(
            validate_layouts(
                Representation::Pq2,
                descriptors.len(),
                descriptors.iter().map(|(name, dimensions, kind, bytes)| (
                    name.as_str(),
                    dimensions.as_slice(),
                    *kind,
                    *bytes
                ))
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn current_weight_ptq1_inventory_is_exact_and_never_substitutes_for_pq2() -> Result<()> {
        let representation = Representation::CurrentRepackedPtq1;
        let descriptors = descriptors(representation);
        let layouts = || {
            descriptors.iter().map(|(name, dimensions, kind, bytes)| {
                (name.as_str(), dimensions.as_slice(), *kind, *bytes)
            })
        };
        assert_eq!(
            validate_layouts(representation, descriptors.len(), layouts())?,
            ([353, 96, 402], 5_935_527_936, 278_118_400)
        );
        assert!(validate_layouts(Representation::Pq2, descriptors.len(), layouts()).is_err());
        assert!(
            validate_layouts(
                representation,
                descriptors.len(),
                layouts().map(|(name, dimensions, kind, bytes)| {
                    (
                        name,
                        dimensions,
                        if kind == 143 { 142 } else { kind },
                        bytes,
                    )
                })
            )
            .is_err()
        );
        assert!(
            validate_layouts(
                representation,
                descriptors.len(),
                layouts().map(|(name, dimensions, kind, bytes)| {
                    (name, dimensions, kind, bytes + u64::from(kind == 143))
                })
            )
            .is_err()
        );
        assert_eq!(Representation::default(), Representation::Pq2);
        assert_eq!(representation.model_bytes(), 5_946_648_928);
        assert_eq!(
            representation.model_sha256(),
            "e658f0f0736a03a4549535a439f9b07254ebf8ab88ae27cbe6e52d098ca5d1f5"
        );
        assert_eq!(representation.name(), "current-repacked-PTQ1_0");
        assert_eq!(representation.file_type(), 143);
        Ok(())
    }

    #[test]
    fn metadata_cannot_admit_generic_qwen_or_missing_template() {
        let mut metadata = Metadata::new();
        metadata.insert(
            "general.architecture".to_owned(),
            MetadataValue::from("qwen35"),
        );
        assert!(validate_metadata(&metadata).is_err());
    }
}
