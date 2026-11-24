//! Exact Swift model admission and retained-storage accounting.
//!
//! PQ2 remains the service MVP default. The explicit current-weight PTQ1 lane
//! permits correctness captures; it does not qualify the service or adapters.
//! Full-model numerical and process-memory qualification remain separate.

use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use bonsai_gguf::{
    Metadata, MetadataArray, MetadataValue, MetadataValueRef, NARIQ_DIALECT, PRISM_DIALECT,
    TensorDescriptor, VerifiedModel,
};
use bonsai_native_hash::{Hasher, hash_bytes};

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
/// Exact length of the pinned `Qwen3.8-27B-NariQ-2.5bpw.gguf` artifact.
pub const NARIQ_2_5BPW_BYTES: u64 = 8_443_119_712;
/// Immutable pinned `Qwen3.8-27B-NariQ-2.5bpw.gguf` SHA-256 identity.
pub const NARIQ_2_5BPW_SHA256: &str =
    "2445372e6de20620b940fbfab53aa718af9ba741f202da30cd2f02b5df78d05c";
/// Exact length of the pinned `Qwen3.8-27B-NariQ-3.0bpw.gguf` artifact.
pub const NARIQ_3_0BPW_BYTES: u64 = 10_123_626_592;
/// Immutable pinned `Qwen3.8-27B-NariQ-3.0bpw.gguf` SHA-256 identity.
pub const NARIQ_3_0BPW_SHA256: &str =
    "1eaad934aa05579b8eba47ed9b26450392f9875fc2354efe8cfa8b29be8f4419";
/// Exact length of the pinned `Qwen3.8-27B-NariQ-2.0bpw.gguf` artifact.
pub const NARIQ_2_0BPW_BYTES: u64 = 6_753_954_304;
/// Immutable pinned `Qwen3.8-27B-NariQ-2.0bpw.gguf` SHA-256 identity.
pub const NARIQ_2_0BPW_SHA256: &str =
    "3a9e4955329dba8ba0be5509a9552220e5ccdae2f66bcc72c21c4580cd49bbb2";
/// Exact embedded UTF-8 template identity in the verified artifact.
pub const TEMPLATE_SHA256: &str =
    "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041";
/// Number of declared input and output vocabulary entries.
pub const VOCAB_SIZE: usize = 248_320;
/// Legacy default context for fixed Swift qualification callers.
pub const CONTEXT_TOKENS: usize = 8_192;
/// Unscaled training context declared by the pinned Swift artifact.
pub const TRAINING_CONTEXT_TOKENS: usize = 262_144;
/// Maximum complete output allowance, including reasoning.
pub const MAX_OUTPUT_TOKENS: usize = 4_096;
/// Exact serialized tensor payload bytes, excluding header and alignment gaps.
pub const TENSOR_PAYLOAD_BYTES: usize = 7_195_047_936;
/// Maximum live FP16 KV and F32 recurrent/convolution payload at 8K context.
pub const STATE_PAYLOAD_BYTES: usize = 693_764_096;
/// Complete immutable multimodal projector GGUF byte length.
pub const MMPROJ_BYTES: u64 = 629_246_976;
/// SHA-256 digest of the pinned `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` artifact.
pub const MMPROJ_SHA256: &str = "6807ede61d570bb86ba34b756a0fa109edc33668604de867c6ea6d8f1d631903";
/// Exact serialized tensor payload bytes, excluding header and alignment gaps.
pub const MMPROJ_TENSOR_PAYLOAD_BYTES: usize = 629_227_328;

const SIGN_SHA256: &str = "3d1ea9136a6044cd680cc1e92cde65e9dd7238e1288fe2effabec5d201c48bed";
const EXPECTED_MMPROJ_TENSORS: usize = 334;
const EXPECTED_SWIFT_TENSORS: usize = 851;
const EXPECTED_NARIQ_UNBANDED_TENSORS: usize = 2_710;
const EXPECTED_NARIQ_BANDED_TENSORS: usize = 2_771;

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
    /// Verified `NariQ` 2.5bpw trellis artifact (`Qwen3.8-27B-NariQ-2.5bpw.gguf`).
    NariQ2_5Bpw,
    /// Verified `NariQ` 3.0bpw trellis artifact (`Qwen3.8-27B-NariQ-3.0bpw.gguf`).
    NariQ3_0Bpw,
    /// Verified `NariQ` 2.0bpw trellis artifact (`Qwen3.8-27B-NariQ-2.0bpw.gguf`).
    NariQ2_0Bpw,
}

impl Representation {
    /// Stable evidence identity distinguishing the source from its exact export or `NariQ` pack.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Pq2 => "PQ2_0",
            Self::CurrentRepackedPtq1 => "current-repacked-PTQ1_0",
            Self::NariQ2_5Bpw => "NariQ-2.5bpw",
            Self::NariQ3_0Bpw => "NariQ-3.0bpw",
            Self::NariQ2_0Bpw => "NariQ-2.0bpw",
        }
    }

    /// Stable model identifier surfaced on readiness responses.
    #[must_use]
    pub const fn model_id(self) -> &'static str {
        match self {
            Self::Pq2 => "Swift-Bonsai-2-PQ2_0",
            Self::CurrentRepackedPtq1 => "Swift-Bonsai-2-current-repacked-PTQ1_0",
            Self::NariQ2_5Bpw => "Qwen3.8-27B-NariQ-2.5bpw",
            Self::NariQ3_0Bpw => "Qwen3.8-27B-NariQ-3.0bpw",
            Self::NariQ2_0Bpw => "Qwen3.8-27B-NariQ-2.0bpw",
        }
    }

    /// Complete immutable artifact length, including GGUF metadata and alignment.
    #[must_use]
    pub const fn model_bytes(self) -> u64 {
        match self {
            Self::Pq2 => MODEL_BYTES,
            Self::CurrentRepackedPtq1 => REPACKED_PTQ1_BYTES,
            Self::NariQ2_5Bpw => NARIQ_2_5BPW_BYTES,
            Self::NariQ3_0Bpw => NARIQ_3_0BPW_BYTES,
            Self::NariQ2_0Bpw => NARIQ_2_0BPW_BYTES,
        }
    }

    /// SHA-256 of the only artifact admitted by this representation.
    #[must_use]
    pub const fn model_sha256(self) -> &'static str {
        match self {
            Self::Pq2 => MODEL_SHA256,
            Self::CurrentRepackedPtq1 => REPACKED_PTQ1_SHA256,
            Self::NariQ2_5Bpw => NARIQ_2_5BPW_SHA256,
            Self::NariQ3_0Bpw => NARIQ_3_0BPW_SHA256,
            Self::NariQ2_0Bpw => NARIQ_2_0BPW_SHA256,
        }
    }

    /// Whether this representation selects an opt-in `NariQ` trellis artifact.
    #[must_use]
    pub const fn is_nariq(self) -> bool {
        matches!(
            self,
            Self::NariQ2_5Bpw | Self::NariQ3_0Bpw | Self::NariQ2_0Bpw
        )
    }

    /// GGUF inspection dialect required when opening this artifact.
    #[must_use]
    pub const fn dialect(self) -> &'static str {
        if self.is_nariq() {
            NARIQ_DIALECT
        } else {
            PRISM_DIALECT
        }
    }

    /// Maximum retained host weight-loading budget appropriate for this representation.
    #[must_use]
    pub const fn model_memory_limit_bytes(self) -> usize {
        match self {
            Self::Pq2 | Self::CurrentRepackedPtq1 | Self::NariQ2_0Bpw => 8 * 1024 * 1024 * 1024,
            Self::NariQ2_5Bpw => 9 * 1024 * 1024 * 1024,
            Self::NariQ3_0Bpw => 11 * 1024 * 1024 * 1024,
        }
    }

    const fn tensor_kind(self) -> u32 {
        match self {
            Self::Pq2 => 142,
            Self::CurrentRepackedPtq1 => 143,
            Self::NariQ2_5Bpw | Self::NariQ2_0Bpw => 45,
            Self::NariQ3_0Bpw => 46,
        }
    }

    const fn file_type(self) -> u64 {
        match self {
            Self::Pq2 => 141,
            Self::CurrentRepackedPtq1 => 143,
            Self::NariQ2_5Bpw | Self::NariQ3_0Bpw | Self::NariQ2_0Bpw => 32,
        }
    }

    const fn tensor_payload_bytes(self) -> usize {
        match self {
            Self::Pq2 => TENSOR_PAYLOAD_BYTES,
            Self::CurrentRepackedPtq1 => 5_935_527_936,
            Self::NariQ2_5Bpw | Self::NariQ3_0Bpw | Self::NariQ2_0Bpw => 0,
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

/// Conservative loader accounting for the pinned `mmproj` artifact.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MmprojLoadReceipt {
    /// Conservative parser charge for retained metadata and descriptors.
    pub header_charge_bytes: usize,
    /// Exact immutable serialized tensor payload retained by the projector.
    pub tensor_payload_bytes: usize,
    /// Header charge plus all retained serialized tensor payloads.
    pub retained_storage_bytes: usize,
    /// Largest tensor payload in bytes.
    pub largest_tensor_bytes: usize,
    /// Counts in `F32`, `F16`, and `Q8_0` order.
    pub tensor_counts: [usize; 3],
}

/// Loads only the pinned Swift multimodal projector through one verified handle.
///
/// # Errors
/// Rejects artifact drift, malformed metadata, any extra or missing tensor,
/// encoding/dimension drift, insufficient storage budgets, and I/O errors.
pub fn load_mmproj(path: &Path, memory_limit: usize) -> Result<(Model, MmprojLoadReceipt)> {
    let (reader, receipt) = verify_mmproj(path, memory_limit)?;
    let model = Model::load_verified(reader, MMPROJ_SHA256, memory_limit, &mut |_| Ok(true))?;
    Ok((model, receipt))
}

/// Verifies the full pinned `mmproj` artifact and header without retaining weights.
///
/// # Errors
/// Rejects changed identity, declarations, inventory, or the storage budget.
pub fn inspect_mmproj(path: &Path, memory_limit: usize) -> Result<MmprojLoadReceipt> {
    Ok(verify_mmproj(path, memory_limit)?.1)
}

fn verify_mmproj(path: &Path, memory_limit: usize) -> Result<(VerifiedModel, MmprojLoadReceipt)> {
    let reader = VerifiedModel::open(path, "Stock", MMPROJ_BYTES, MMPROJ_SHA256, memory_limit)?;
    validate_mmproj_metadata(reader.metadata())?;
    let (tensor_counts, tensor_payload_bytes, largest_tensor_bytes) = validate_mmproj_layouts(
        reader.tensors().len(),
        reader.tensors().iter().map(|tensor| {
            (
                tensor.name.as_str(),
                tensor.dimensions.as_slice(),
                tensor.type_id,
                tensor.bytes,
            )
        }),
    )?;
    let header_charge_bytes = reader.retained_bytes();
    let retained_storage_bytes = header_charge_bytes
        .checked_add(tensor_payload_bytes)
        .ok_or_else(|| invalid("Swift mmproj retained storage accounting overflow"))?;
    if retained_storage_bytes > memory_limit {
        return Err(invalid(
            "Swift mmproj retained storage exceeds its memory budget",
        ));
    }
    Ok((
        reader,
        MmprojLoadReceipt {
            header_charge_bytes,
            tensor_payload_bytes,
            retained_storage_bytes,
            largest_tensor_bytes,
            tensor_counts,
        },
    ))
}

/// Checks the pinned `qwen3vl` `mmproj` metadata contract.
///
/// # Errors
/// Rejects missing, incompatible, or drifted vision projector declarations.
pub fn validate_mmproj_metadata(metadata: &Metadata) -> Result<()> {
    if metadata.len() < 22 || metadata.len() > 64 {
        return Err(invalid("Swift mmproj metadata inventory mismatch"));
    }
    for (key, value) in [
        ("general.architecture", "clip"),
        ("general.type", "mmproj"),
        ("general.name", "Hf"),
        ("general.size_label", "461M"),
        ("general.license", "apache-2.0"),
        ("clip.projector_type", "qwen3vl_merger"),
    ] {
        if metadata.get(key).and_then(MetadataValue::as_str) != Some(value) {
            return Err(invalid(format!("Swift mmproj metadata mismatch: {key}")));
        }
    }
    for (key, value) in [("clip.has_vision_encoder", true), ("clip.use_gelu", true)] {
        if metadata.get(key).and_then(MetadataValue::as_bool) != Some(value) {
            return Err(invalid(format!("Swift mmproj boolean mismatch: {key}")));
        }
    }
    for (key, forbidden) in [("clip.has_audio_encoder", true), ("clip.use_silu", true)] {
        if metadata.get(key).and_then(MetadataValue::as_bool) == Some(forbidden) {
            return Err(invalid(format!("Swift mmproj unsupported flag: {key}")));
        }
    }
    for (key, value) in [
        ("general.file_type", 7_u64),
        ("general.quantization_version", 2),
        ("clip.vision.projection_dim", 5_120),
        ("clip.vision.image_size", 768),
        ("clip.vision.patch_size", 16),
        ("clip.vision.embedding_length", 1_152),
        ("clip.vision.feed_forward_length", 4_304),
        ("clip.vision.block_count", 27),
        ("clip.vision.attention.head_count", 16),
        ("clip.vision.spatial_merge_size", 2),
    ] {
        if metadata.get(key).and_then(MetadataValue::as_u64) != Some(value) {
            return Err(invalid(format!("Swift mmproj integer mismatch: {key}")));
        }
    }
    if let Some(alignment) = metadata.get("general.alignment")
        && alignment.as_u64() != Some(32)
    {
        return Err(invalid("Swift mmproj alignment mismatch"));
    }
    let eps = metadata
        .get("clip.vision.attention.layer_norm_epsilon")
        .and_then(MetadataValue::as_f64)
        .ok_or_else(|| invalid("Swift mmproj layer norm epsilon missing"))?;
    if !eps.is_finite() || (eps - f64::from(1e-6_f32)).abs() > 1e-12 {
        return Err(invalid("Swift mmproj layer norm epsilon mismatch"));
    }
    match metadata
        .get("clip.vision.is_deepstack_layers")
        .and_then(MetadataValue::as_array)
    {
        Some(MetadataArray::Bool(values))
            if values.len() == 27 && values.iter().all(|enabled| !*enabled) => {}
        _ => return Err(invalid("Swift mmproj deepstack array mismatch")),
    }
    for key in ["clip.vision.image_mean", "clip.vision.image_std"] {
        match metadata.get(key).and_then(MetadataValue::as_array) {
            Some(MetadataArray::F32(values))
                if values.len() == 3
                    && values.iter().all(|&v| v.to_bits() == 0.5_f32.to_bits()) => {}
            _ => {
                return Err(invalid(format!(
                    "Swift mmproj normalization mismatch: {key}"
                )));
            }
        }
    }
    Ok(())
}

/// Returns the expected tensor inventory `(gguf_type_id, dimensions)` for the pinned `mmproj`.
#[must_use]
pub fn expected_mmproj_tensors() -> BTreeMap<String, (u32, Vec<u64>)> {
    let mut expected = BTreeMap::new();
    expected.insert(
        "v.patch_embd.weight".to_owned(),
        (0, vec![16, 16, 3, 1_152]),
    );
    expected.insert(
        "v.patch_embd.weight.1".to_owned(),
        (0, vec![16, 16, 3, 1_152]),
    );
    expected.insert("v.patch_embd.bias".to_owned(), (0, vec![1_152]));
    expected.insert("v.position_embd.weight".to_owned(), (0, vec![1_152, 2_304]));
    expected.insert("v.post_ln.weight".to_owned(), (0, vec![1_152]));
    expected.insert("v.post_ln.bias".to_owned(), (0, vec![1_152]));
    expected.insert("mm.0.weight".to_owned(), (8, vec![4_608, 4_608]));
    expected.insert("mm.0.bias".to_owned(), (0, vec![4_608]));
    expected.insert("mm.2.weight".to_owned(), (8, vec![4_608, 5_120]));
    expected.insert("mm.2.bias".to_owned(), (0, vec![5_120]));
    for block in 0..27 {
        expected.insert(format!("v.blk.{block}.ln1.weight"), (0, vec![1_152]));
        expected.insert(format!("v.blk.{block}.ln1.bias"), (0, vec![1_152]));
        expected.insert(
            format!("v.blk.{block}.attn_qkv.weight"),
            (8, vec![1_152, 3_456]),
        );
        expected.insert(format!("v.blk.{block}.attn_qkv.bias"), (0, vec![3_456]));
        expected.insert(
            format!("v.blk.{block}.attn_out.weight"),
            (8, vec![1_152, 1_152]),
        );
        expected.insert(format!("v.blk.{block}.attn_out.bias"), (0, vec![1_152]));
        expected.insert(format!("v.blk.{block}.ln2.weight"), (0, vec![1_152]));
        expected.insert(format!("v.blk.{block}.ln2.bias"), (0, vec![1_152]));
        expected.insert(
            format!("v.blk.{block}.ffn_up.weight"),
            (8, vec![1_152, 4_304]),
        );
        expected.insert(format!("v.blk.{block}.ffn_up.bias"), (0, vec![4_304]));
        expected.insert(
            format!("v.blk.{block}.ffn_down.weight"),
            (1, vec![4_304, 1_152]),
        );
        expected.insert(format!("v.blk.{block}.ffn_down.bias"), (0, vec![1_152]));
    }
    expected
}

fn parse_block_index(segment: &str, limit: usize) -> Option<usize> {
    if segment.is_empty()
        || (segment.len() > 1 && segment.starts_with('0'))
        || !segment.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    let index = segment.parse::<usize>().ok()?;
    (index < limit).then_some(index)
}

fn expected_mmproj_spec(name: &str) -> Option<(u32, &'static [u64])> {
    match name {
        "v.patch_embd.weight" | "v.patch_embd.weight.1" => Some((0, &[16, 16, 3, 1_152])),
        "v.patch_embd.bias" | "v.post_ln.weight" | "v.post_ln.bias" => Some((0, &[1_152])),
        "v.position_embd.weight" => Some((0, &[1_152, 2_304])),
        "mm.0.weight" => Some((8, &[4_608, 4_608])),
        "mm.0.bias" => Some((0, &[4_608])),
        "mm.2.weight" => Some((8, &[4_608, 5_120])),
        "mm.2.bias" => Some((0, &[5_120])),
        _ => {
            let rest = name.strip_prefix("v.blk.")?;
            let (block, suffix) = rest.split_once('.')?;
            parse_block_index(block, 27)?;
            match suffix {
                "ln1.weight" | "ln1.bias" | "ln2.weight" | "ln2.bias" | "attn_out.bias"
                | "ffn_down.bias" => Some((0, &[1_152])),
                "attn_qkv.weight" => Some((8, &[1_152, 3_456])),
                "attn_qkv.bias" => Some((0, &[3_456])),
                "attn_out.weight" => Some((8, &[1_152, 1_152])),
                "ffn_up.weight" => Some((8, &[1_152, 4_304])),
                "ffn_up.bias" => Some((0, &[4_304])),
                "ffn_down.weight" => Some((1, &[4_304, 1_152])),
                _ => None,
            }
        }
    }
}

/// Validates tensor names, dimensions, encodings, and total byte size for the pinned `mmproj`.
///
/// # Errors
/// Rejects missing, extra, renamed, reshaped, or re-typed tensors or mismatched byte counts.
pub fn validate_mmproj_layouts<'a>(
    count: usize,
    tensors: impl IntoIterator<Item = (&'a str, &'a [u64], u32, u64)>,
) -> Result<([usize; 3], usize, usize)> {
    if count != EXPECTED_MMPROJ_TENSORS {
        return Err(invalid("Swift mmproj tensor inventory count mismatch"));
    }
    let mut counts = [0; 3];
    let mut bytes = 0_usize;
    let mut largest = 0_usize;
    let mut seen = HashSet::new();
    seen.try_reserve(EXPECTED_MMPROJ_TENSORS)?;
    for (name, dimensions, kind, payload_bytes) in tensors {
        if !seen.insert(name)
            || expected_mmproj_spec(name).is_none_or(|(expected_kind, expected_dimensions)| {
                expected_kind != kind || expected_dimensions != dimensions
            })
        {
            return Err(invalid(format!(
                "Swift mmproj tensor layout mismatch: {name}"
            )));
        }
        let slot = match kind {
            0 => 0,
            1 => 1,
            8 => 2,
            _ => return Err(invalid("Swift mmproj tensor encoding mismatch")),
        };
        counts[slot] += 1;
        let length = usize::try_from(payload_bytes)?;
        bytes = bytes
            .checked_add(length)
            .ok_or_else(|| invalid("Swift mmproj tensor accounting overflow"))?;
        largest = largest.max(length);
    }
    if counts != [224, 27, 83] || bytes != MMPROJ_TENSOR_PAYLOAD_BYTES {
        return Err(invalid("Swift mmproj tensor payload inventory mismatch"));
    }
    Ok((counts, bytes, largest))
}

fn verify(
    path: &Path,
    memory_limit: usize,
    representation: Representation,
) -> Result<(VerifiedModel, LoadReceipt)> {
    let reader = VerifiedModel::open(
        path,
        representation.dialect(),
        representation.model_bytes(),
        representation.model_sha256(),
        memory_limit,
    )?;
    if representation.is_nariq() {
        validate_nariq_metadata(reader.metadata(), representation)?;
    } else {
        validate_metadata_for(reader.metadata(), representation)?;
    }
    let (tensor_counts, tensor_payload_bytes, largest_tensor_bytes) = if representation.is_nariq() {
        validate_nariq_inventory(reader.tensors(), representation)?
    } else {
        validate_inventory(reader.tensors(), representation)?
    };
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

#[expect(
    clippy::too_many_lines,
    reason = "Keep the exact NariQ metadata contract auditable in one helper."
)]
fn validate_nariq_metadata(metadata: &Metadata, representation: Representation) -> Result<()> {
    if !representation.is_nariq() {
        return Err(invalid(
            "non-NariQ representation passed to NariQ metadata validator",
        ));
    }
    if metadata
        .keys()
        .any(|key| key.starts_with("prism.hadamard."))
    {
        return Err(invalid("NariQ metadata forbids prism.hadamard transforms"));
    }
    for (key, value) in [
        ("general.architecture", "qwen35"),
        ("tokenizer.ggml.model", "gpt2"),
        ("tokenizer.ggml.pre", "qwen35"),
    ] {
        if metadata.get(key).and_then(MetadataValue::as_str) != Some(value) {
            return Err(invalid(format!("NariQ metadata mismatch: {key}")));
        }
    }
    for (key, value) in [
        ("general.file_type", representation.file_type()),
        ("trellis.version", 1),
        ("trellis.lm_head_bands", 16),
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
        ("tokenizer.ggml.eos_token_id", 248_046),
        ("tokenizer.ggml.padding_token_id", 248_044),
        ("tokenizer.ggml.bos_token_id", 248_044),
    ] {
        if metadata.get(key).and_then(MetadataValue::as_u64) != Some(value) {
            return Err(invalid(format!("NariQ metadata mismatch: {key}")));
        }
    }
    match representation {
        Representation::NariQ2_0Bpw => {
            if metadata
                .get("trellis.embed_bands")
                .and_then(MetadataValue::as_u64)
                != Some(16)
            {
                return Err(invalid("NariQ 2.0bpw requires trellis.embed_bands = 16"));
            }
        }
        Representation::NariQ2_5Bpw | Representation::NariQ3_0Bpw => {
            if metadata.contains_key("trellis.embed_bands") {
                return Err(invalid(
                    "unbanded NariQ artifact forbids trellis.embed_bands",
                ));
            }
        }
        Representation::Pq2 | Representation::CurrentRepackedPtq1 => {}
    }
    for (key, value) in [
        ("qwen35.rope.freq_base", 10_000_000.0_f32),
        ("qwen35.attention.layer_norm_rms_epsilon", 0.000_001_f32),
    ] {
        if metadata
            .get(key)
            .and_then(MetadataValue::as_f64)
            .map(f64::to_bits)
            != Some(f64::from(value).to_bits())
        {
            return Err(invalid(format!("NariQ metadata mismatch: {key}")));
        }
    }
    if metadata
        .get("tokenizer.ggml.add_bos_token")
        .and_then(MetadataValue::as_bool)
        != Some(false)
    {
        return Err(invalid(
            "NariQ metadata mismatch: tokenizer.ggml.add_bos_token",
        ));
    }
    if let Some(add_eos) = metadata.get("tokenizer.ggml.add_eos_token")
        && add_eos.as_bool() != Some(false)
    {
        return Err(invalid(
            "NariQ metadata mismatch: tokenizer.ggml.add_eos_token",
        ));
    }
    let template = metadata
        .get("tokenizer.chat_template")
        .and_then(MetadataValue::as_str)
        .ok_or_else(|| invalid("NariQ embedded template is missing"))?;
    if template.len() != 8_952 || !digest_matches(hash_bytes(template.as_bytes())?, TEMPLATE_SHA256)
    {
        return Err(invalid("NariQ embedded template identity mismatch"));
    }
    check_integers(metadata, "qwen35.rope.dimension_sections", &[11, 11, 10, 0])?;
    for (key, count, strings) in [
        ("tokenizer.ggml.tokens", VOCAB_SIZE, true),
        ("tokenizer.ggml.token_type", VOCAB_SIZE, false),
        ("tokenizer.ggml.merges", 247_587, true),
    ] {
        let array = metadata
            .get(key)
            .and_then(MetadataValue::as_array)
            .ok_or_else(|| invalid(format!("NariQ array missing: {key}")))?;
        if array.len() != count
            || (strings && !matches!(array, MetadataArray::String(_)))
            || (!strings && !matches!(array, MetadataArray::I32(_)))
        {
            return Err(invalid(format!("NariQ tokenizer array mismatch: {key}")));
        }
    }
    Ok(())
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

#[expect(
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
    if template.len() != 8_952 || !digest_matches(hash_bytes(template.as_bytes())?, TEMPLATE_SHA256)
    {
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
    let mut sign_hash = Hasher::new()?;
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
        sign_hash.update(&i32::try_from(value)?.to_le_bytes())?;
    }
    if !digest_matches(sign_hash.finish()?, SIGN_SHA256) {
        return Err(invalid("Swift Hadamard sign identity mismatch"));
    }
    let transformed = metadata
        .get("prism.hadamard.weight_names")
        .and_then(MetadataValue::as_array)
        .ok_or_else(|| invalid("Swift Hadamard weight inventory missing"))?;
    let mut names = HashSet::new();
    names.try_reserve(401)?;
    for value in transformed.iter() {
        let name = value
            .as_str()
            .ok_or_else(|| invalid("Swift transform names must be strings"))?;
        if name == "token_embd.weight"
            || expected_tensor_spec(name, representation)
                .is_none_or(|(kind, _)| kind != representation.tensor_kind())
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

fn digest_matches(digest: [u8; 32], expected_hex: &str) -> bool {
    let mut actual = [0_u8; 64];
    hex::encode_to_slice(digest, &mut actual).is_ok() && actual == expected_hex.as_bytes()
}

fn expected_tensor_spec(
    name: &str,
    representation: Representation,
) -> Option<(u32, &'static [u64])> {
    let packed = representation.tensor_kind();
    match name {
        "token_embd.weight" | "output.weight" => Some((packed, &[5_120, 248_320])),
        "output_norm.weight" => Some((0, &[5_120])),
        _ => {
            let rest = name.strip_prefix("blk.")?;
            let (layer_str, suffix) = rest.split_once('.')?;
            let layer = parse_block_index(layer_str, 64)?;
            match suffix {
                "attn_norm.weight" | "post_attention_norm.weight" => Some((0, &[5_120])),
                "ffn_up.weight" | "ffn_gate.weight" => Some((packed, &[5_120, 17_408])),
                "ffn_down.weight" => Some((packed, &[17_408, 5_120])),
                _ if (layer + 1).is_multiple_of(4) => match suffix {
                    "attn_q.weight" => Some((packed, &[5_120, 12_288])),
                    "attn_k.weight" | "attn_v.weight" => Some((packed, &[5_120, 1_024])),
                    "attn_output.weight" => Some((packed, &[6_144, 5_120])),
                    "attn_q_norm.weight" | "attn_k_norm.weight" => Some((0, &[256])),
                    _ => None,
                },
                _ => match suffix {
                    "attn_qkv.weight" => Some((packed, &[5_120, 10_240])),
                    "attn_gate.weight" => Some((packed, &[5_120, 6_144])),
                    "ssm_out.weight" => Some((packed, &[6_144, 5_120])),
                    "ssm_alpha.weight" | "ssm_beta.weight" => Some((30, &[5_120, 48])),
                    "ssm_conv1d.weight" => Some((0, &[4, 10_240])),
                    "ssm_norm.weight" => Some((0, &[128])),
                    "ssm_a" | "ssm_dt.bias" => Some((0, &[48])),
                    _ => None,
                },
            }
        }
    }
}

#[cfg(test)]
fn expected_tensors(representation: Representation) -> BTreeMap<String, (u32, Vec<u64>)> {
    let mut tensors = BTreeMap::new();
    let packed = representation.tensor_kind();
    for name in ["token_embd.weight", "output.weight"] {
        tensors.insert(name.to_owned(), (packed, vec![5_120, 248_320]));
    }
    tensors.insert("output_norm.weight".to_owned(), (0, vec![5_120]));
    for layer in 0_usize..64 {
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
        if (layer + 1).is_multiple_of(4) {
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

fn validate_nariq_band_name(name: &str, prefix: &str) -> Option<usize> {
    let rest = name.strip_prefix(prefix)?;
    let band_str = rest.strip_suffix(".weight")?;
    if band_str.len() != 2 || !band_str.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let idx = band_str.parse::<usize>().ok()?;
    (idx < 16).then_some(idx)
}

#[expect(
    clippy::too_many_lines,
    reason = "Validate complete NariQ tensor inventory and band geometry in one pass."
)]
fn validate_nariq_inventory(
    tensors: &[TensorDescriptor],
    representation: Representation,
) -> Result<([usize; 3], usize, usize)> {
    let expected_count = match representation {
        Representation::NariQ2_0Bpw => EXPECTED_NARIQ_BANDED_TENSORS,
        Representation::NariQ2_5Bpw | Representation::NariQ3_0Bpw => {
            EXPECTED_NARIQ_UNBANDED_TENSORS
        }
        Representation::Pq2 | Representation::CurrentRepackedPtq1 => {
            return Err(invalid(
                "non-NariQ representation passed to NariQ inventory validator",
            ));
        }
    };
    if tensors.len() != expected_count {
        return Err(invalid("NariQ tensor inventory count mismatch"));
    }
    let vocab_u64 = u64::try_from(VOCAB_SIZE)?;
    let mut counts = [0_usize; 3];
    let mut bytes = 0_usize;
    let mut largest = 0_usize;
    let mut seen = HashSet::new();
    seen.try_reserve(expected_count)?;

    let mut saw_output_gather = false;
    let mut output_band_cols = [0_u64; 16];
    let mut saw_embed_weight = false;
    let mut saw_embed_gather = false;
    let mut embed_band_cols = [0_u64; 16];

    for tensor in tensors {
        let name = tensor.name.as_str();
        if !seen.insert(name) {
            return Err(invalid(format!("duplicate NariQ tensor: {name}")));
        }
        let dims = tensor.dimensions.as_slice();
        let (slot, expected_bytes) = match tensor.type_id {
            0 => {
                let elems = dims
                    .iter()
                    .try_fold(1_u64, |acc, &d| {
                        (d > 0).then_some(()).and_then(|()| acc.checked_mul(d))
                    })
                    .ok_or_else(|| {
                        invalid(format!("NariQ F32 tensor dimension overflow: {name}"))
                    })?;
                (
                    0_usize,
                    elems
                        .checked_mul(4)
                        .ok_or_else(|| invalid("NariQ tensor byte overflow"))?,
                )
            }
            1 | 30 => {
                let elems = dims
                    .iter()
                    .try_fold(1_u64, |acc, &d| {
                        (d > 0).then_some(()).and_then(|()| acc.checked_mul(d))
                    })
                    .ok_or_else(|| {
                        invalid(format!("NariQ 16-bit tensor dimension overflow: {name}"))
                    })?;
                (
                    0_usize,
                    elems
                        .checked_mul(2)
                        .ok_or_else(|| invalid("NariQ tensor byte overflow"))?,
                )
            }
            26 => {
                let elems = dims
                    .iter()
                    .try_fold(1_u64, |acc, &d| {
                        (d > 0).then_some(()).and_then(|()| acc.checked_mul(d))
                    })
                    .ok_or_else(|| {
                        invalid(format!("NariQ I32 tensor dimension overflow: {name}"))
                    })?;
                (
                    1_usize,
                    elems
                        .checked_mul(4)
                        .ok_or_else(|| invalid("NariQ tensor byte overflow"))?,
                )
            }
            44..=49 => {
                if dims.len() != 2
                    || !dims[0].is_multiple_of(16)
                    || !dims[1].is_multiple_of(16)
                    || dims[0] == 0
                    || dims[1] == 0
                {
                    return Err(invalid(format!(
                        "NariQ trellis tensor dimensions misaligned: {name}"
                    )));
                }
                let bits = u64::from(tensor.type_id - 43);
                let expected = (dims[0] / 16)
                    .checked_mul(dims[1] / 16)
                    .and_then(|tiles| tiles.checked_mul(32 * bits))
                    .ok_or_else(|| invalid("NariQ trellis payload overflow"))?;
                (2_usize, expected)
            }
            _ => {
                return Err(invalid(format!(
                    "unsupported NariQ tensor encoding in {name}"
                )));
            }
        };
        if tensor.bytes != expected_bytes {
            return Err(invalid(format!(
                "NariQ tensor payload byte mismatch: {name}"
            )));
        }

        if name == "output.vocab_gather" {
            if tensor.type_id != 26 || dims != [vocab_u64] {
                return Err(invalid("invalid NariQ output.vocab_gather layout"));
            }
            saw_output_gather = true;
        } else if let Some(idx) = validate_nariq_band_name(name, "output.band") {
            if slot != 2 || dims.len() != 2 || dims[0] != 5_120 {
                return Err(invalid(format!("invalid NariQ output band layout: {name}")));
            }
            output_band_cols[idx] = dims[1];
        } else if name == "token_embd.vocab_gather" {
            if representation != Representation::NariQ2_0Bpw
                || tensor.type_id != 26
                || dims != [vocab_u64]
            {
                return Err(invalid(
                    "unexpected or malformed NariQ token_embd.vocab_gather",
                ));
            }
            saw_embed_gather = true;
        } else if let Some(idx) = validate_nariq_band_name(name, "token_embd.band") {
            if representation != Representation::NariQ2_0Bpw
                || slot != 2
                || dims.len() != 2
                || dims[0] != 5_120
            {
                return Err(invalid(format!(
                    "unexpected or malformed NariQ embed band: {name}"
                )));
            }
            embed_band_cols[idx] = dims[1];
        } else if name.starts_with("token_embd.band") || name.starts_with("output.band") {
            if name.ends_with(".weight") {
                return Err(invalid(format!("unexpected NariQ head band index: {name}")));
            }
        } else if name == "token_embd.weight" {
            if representation == Representation::NariQ2_0Bpw || dims != [5_120, vocab_u64] {
                return Err(invalid("unexpected or malformed NariQ token_embd.weight"));
            }
            saw_embed_weight = true;
        }

        counts[slot] += 1;
        let length = usize::try_from(tensor.bytes)?;
        bytes = bytes
            .checked_add(length)
            .ok_or_else(|| invalid("NariQ tensor accounting overflow"))?;
        largest = largest.max(length);
    }

    if !saw_output_gather
        || output_band_cols.contains(&0)
        || output_band_cols.iter().sum::<u64>() != vocab_u64
    {
        return Err(invalid("incomplete NariQ banded output head inventory"));
    }
    match representation {
        Representation::NariQ2_0Bpw => {
            if saw_embed_weight
                || !saw_embed_gather
                || embed_band_cols.contains(&0)
                || embed_band_cols.iter().sum::<u64>() != vocab_u64
            {
                return Err(invalid(
                    "incomplete NariQ 2.0bpw banded embedding inventory",
                ));
            }
        }
        Representation::NariQ2_5Bpw | Representation::NariQ3_0Bpw => {
            if !saw_embed_weight || saw_embed_gather || embed_band_cols.iter().any(|&c| c != 0) {
                return Err(invalid("invalid unbanded NariQ embedding inventory"));
            }
        }
        Representation::Pq2 | Representation::CurrentRepackedPtq1 => {}
    }
    Ok((counts, bytes, largest))
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
    if count != EXPECTED_SWIFT_TENSORS {
        return Err(invalid("Swift tensor inventory count mismatch"));
    }
    let mut counts = [0; 3];
    let mut bytes = 0_usize;
    let mut largest = 0;
    let mut seen = HashSet::new();
    seen.try_reserve(EXPECTED_SWIFT_TENSORS)?;
    for (name, dimensions, kind, payload_bytes) in tensors {
        if !seen.insert(name)
            || expected_tensor_spec(name, representation).is_none_or(
                |(expected_kind, expected_dimensions)| {
                    expected_kind != kind || expected_dimensions != dimensions
                },
            )
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

    #[test]
    fn mmproj_inventory_and_metadata_match_pinned_q8_artifact() -> Result<()> {
        let mut metadata = Metadata::new();
        for (key, value) in [
            ("general.architecture", "clip"),
            ("general.type", "mmproj"),
            ("general.name", "Hf"),
            ("general.size_label", "461M"),
            ("general.license", "apache-2.0"),
            ("clip.projector_type", "qwen3vl_merger"),
        ] {
            metadata.insert(key.to_owned(), MetadataValue::from(value));
        }
        metadata.insert(
            "clip.has_vision_encoder".to_owned(),
            MetadataValue::from(true),
        );
        metadata.insert("clip.use_gelu".to_owned(), MetadataValue::from(true));
        for (key, value) in [
            ("general.file_type", 7_u32),
            ("general.quantization_version", 2),
            ("clip.vision.projection_dim", 5_120),
            ("clip.vision.image_size", 768),
            ("clip.vision.patch_size", 16),
            ("clip.vision.embedding_length", 1_152),
            ("clip.vision.feed_forward_length", 4_304),
            ("clip.vision.block_count", 27),
            ("clip.vision.attention.head_count", 16),
            ("clip.vision.spatial_merge_size", 2),
        ] {
            metadata.insert(key.to_owned(), MetadataValue::from(value));
        }
        metadata.insert(
            "clip.vision.attention.layer_norm_epsilon".to_owned(),
            MetadataValue::F32(1e-6_f32),
        );
        metadata.insert(
            "clip.vision.is_deepstack_layers".to_owned(),
            MetadataValue::Array(MetadataArray::Bool(vec![false; 27])),
        );
        metadata.insert(
            "clip.vision.image_mean".to_owned(),
            MetadataValue::Array(MetadataArray::F32(vec![0.5, 0.5, 0.5])),
        );
        metadata.insert(
            "clip.vision.image_std".to_owned(),
            MetadataValue::Array(MetadataArray::F32(vec![0.5, 0.5, 0.5])),
        );
        validate_mmproj_metadata(&metadata)?;

        let expected = expected_mmproj_tensors();
        let descriptors: Vec<(String, Vec<u64>, u32, u64)> = expected
            .into_iter()
            .map(|(name, (kind, dims))| {
                let elements = dims.iter().product::<u64>();
                let bytes = match kind {
                    0 => elements * 4,
                    1 => elements * 2,
                    8 => (elements / 32) * 34,
                    _ => 0,
                };
                (name, dims, kind, bytes)
            })
            .collect();
        assert_eq!(
            validate_mmproj_layouts(
                descriptors.len(),
                descriptors
                    .iter()
                    .map(|(n, d, k, b)| (n.as_str(), d.as_slice(), *k, *b)),
            )?,
            ([224, 27, 83], MMPROJ_TENSOR_PAYLOAD_BYTES, 25_067_520)
        );

        let mut drifted = metadata;
        drifted.insert(
            "clip.projector_type".to_owned(),
            MetadataValue::from("qwen2vl_merger"),
        );
        assert!(validate_mmproj_metadata(&drifted).is_err());
        Ok(())
    }

    #[test]
    fn zero_alloc_tensor_specs_match_full_inventories_and_reject_duplicates_and_bad_indices()
    -> Result<()> {
        for representation in [Representation::Pq2, Representation::CurrentRepackedPtq1] {
            let map = expected_tensors(representation);
            assert_eq!(map.len(), EXPECTED_SWIFT_TENSORS);
            for (name, (kind, dims)) in &map {
                let spec = expected_tensor_spec(name, representation)
                    .ok_or_else(|| invalid("missing spec"))?;
                assert_eq!(spec.0, *kind);
                assert_eq!(spec.1, dims.as_slice());
            }
            for bad_name in [
                "blk.00.attn_norm.weight",
                "blk.+0.attn_norm.weight",
                "blk.64.attn_norm.weight",
                "blk.3.attn_qkv.weight",
                "blk.0.attn_q.weight",
                "blk.0.unknown",
            ] {
                assert!(expected_tensor_spec(bad_name, representation).is_none());
            }

            let mut dup = descriptors(representation);
            let dupe_entry = dup[0].clone();
            let last = dup.len() - 1;
            dup[last] = dupe_entry;
            assert!(
                validate_layouts(
                    representation,
                    dup.len(),
                    dup.iter()
                        .map(|(n, d, k, b)| (n.as_str(), d.as_slice(), *k, *b)),
                )
                .is_err()
            );
        }

        let mmproj_map = expected_mmproj_tensors();
        assert_eq!(mmproj_map.len(), EXPECTED_MMPROJ_TENSORS);
        for (name, (kind, dims)) in &mmproj_map {
            let spec = expected_mmproj_spec(name).ok_or_else(|| invalid("missing mmproj spec"))?;
            assert_eq!(spec.0, *kind);
            assert_eq!(spec.1, dims.as_slice());
        }
        for bad_name in [
            "v.blk.00.ln1.weight",
            "v.blk.+1.ln1.weight",
            "v.blk.27.ln1.weight",
            "v.blk.0.unknown",
        ] {
            assert!(expected_mmproj_spec(bad_name).is_none());
        }
        Ok(())
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "Checks NariQ identity, limit, metadata, and tensor inventory tables in one cohesive suite."
    )]
    fn nariq_representations_verify_identities_limits_metadata_and_inventories() -> Result<()> {
        let cases = [
            (
                Representation::NariQ2_5Bpw,
                "NariQ-2.5bpw",
                "Qwen3.8-27B-NariQ-2.5bpw",
                NARIQ_2_5BPW_BYTES,
                NARIQ_2_5BPW_SHA256,
                9 * 1024 * 1024 * 1024,
                false,
                EXPECTED_NARIQ_UNBANDED_TENSORS,
            ),
            (
                Representation::NariQ3_0Bpw,
                "NariQ-3.0bpw",
                "Qwen3.8-27B-NariQ-3.0bpw",
                NARIQ_3_0BPW_BYTES,
                NARIQ_3_0BPW_SHA256,
                11 * 1024 * 1024 * 1024,
                false,
                EXPECTED_NARIQ_UNBANDED_TENSORS,
            ),
            (
                Representation::NariQ2_0Bpw,
                "NariQ-2.0bpw",
                "Qwen3.8-27B-NariQ-2.0bpw",
                NARIQ_2_0BPW_BYTES,
                NARIQ_2_0BPW_SHA256,
                8 * 1024 * 1024 * 1024,
                true,
                EXPECTED_NARIQ_BANDED_TENSORS,
            ),
        ];

        for (repr, name, model_id, bytes, sha, mem_limit, banded, expected_count) in cases {
            assert!(repr.is_nariq());
            assert_eq!(repr.name(), name);
            assert_eq!(repr.model_id(), model_id);
            assert_eq!(repr.model_bytes(), bytes);
            assert_eq!(repr.model_sha256(), sha);
            assert_eq!(repr.dialect(), NARIQ_DIALECT);
            assert_eq!(repr.model_memory_limit_bytes(), mem_limit);
            assert_eq!(repr.file_type(), 32);

            let mut descs = Vec::with_capacity(expected_count);
            descs.push(TensorDescriptor {
                name: "output.vocab_gather".to_owned(),
                type_id: 26,
                dimensions: vec![u64::try_from(VOCAB_SIZE)?],
                elements: u64::try_from(VOCAB_SIZE)?,
                offset: 0,
                bytes: u64::try_from(VOCAB_SIZE)? * 4,
            });
            let cols = [15_520_u64; 16];
            for (idx, &band_n) in cols.iter().enumerate() {
                descs.push(TensorDescriptor {
                    name: format!("output.band{idx:02}.weight"),
                    type_id: 45,
                    dimensions: vec![5_120, band_n],
                    elements: 5_120 * band_n,
                    offset: 0,
                    bytes: (5_120 / 16) * (band_n / 16) * 32 * 2,
                });
            }
            if banded {
                descs.push(TensorDescriptor {
                    name: "token_embd.vocab_gather".to_owned(),
                    type_id: 26,
                    dimensions: vec![u64::try_from(VOCAB_SIZE)?],
                    elements: u64::try_from(VOCAB_SIZE)?,
                    offset: 0,
                    bytes: u64::try_from(VOCAB_SIZE)? * 4,
                });
                for (idx, &band_n) in cols.iter().enumerate() {
                    descs.push(TensorDescriptor {
                        name: format!("token_embd.band{idx:02}.weight"),
                        type_id: 44,
                        dimensions: vec![5_120, band_n],
                        elements: 5_120 * band_n,
                        offset: 0,
                        bytes: (5_120 / 16) * (band_n / 16) * 32,
                    });
                }
            } else {
                descs.push(TensorDescriptor {
                    name: "token_embd.weight".to_owned(),
                    type_id: 1,
                    dimensions: vec![5_120, u64::try_from(VOCAB_SIZE)?],
                    elements: 5_120 * u64::try_from(VOCAB_SIZE)?,
                    offset: 0,
                    bytes: 5_120 * u64::try_from(VOCAB_SIZE)? * 2,
                });
            }
            while descs.len() < expected_count {
                let extra_idx = descs.len();
                descs.push(TensorDescriptor {
                    name: format!("blk.{extra_idx}.filler_norm.weight"),
                    type_id: 0,
                    dimensions: vec![5_120],
                    elements: 5_120,
                    offset: 0,
                    bytes: 5_120 * 4,
                });
            }
            let (counts, total_bytes, _) = validate_nariq_inventory(&descs, repr)?;
            assert!(total_bytes > 0);
            assert_eq!(counts[0] + counts[1] + counts[2], expected_count);
            // Cross-representation inventory must fail closed between banded 2.0bpw and unbanded 2.5/3.0bpw.
            let other = if banded {
                Representation::NariQ2_5Bpw
            } else {
                Representation::NariQ2_0Bpw
            };
            assert!(validate_nariq_inventory(&descs, other).is_err());
        }
        Ok(())
    }
}
