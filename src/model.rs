//! Verified, bounded GGUF storage for the native Rust runtime.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use bonsai_gguf::{Metadata, PRISM_DIALECT, VerifiedModel};
use bonsai_prism::{Layout, Pool};

use crate::{Result, invalid};

/// An immutable serialized tensor in GGUF/ggml dimension order.
#[derive(Clone, Debug)]
pub struct Tensor {
    /// Unique tensor name from the source descriptor.
    pub name: String,
    /// Extents with the contiguous row dimension first.
    pub dimensions: Vec<usize>,
    /// Validated GGUF encoding ID.
    pub kind: u32,
    /// Exact serialized bytes, shareable without mutable aliases.
    pub data: Arc<[u8]>,
}

/// A verified GGUF model with exact metadata and immutable tensor storage.
#[derive(Debug)]
pub struct Model {
    /// Complete retained scalar, tokenizer, array and template metadata.
    pub metadata: Metadata,
    /// Tensors indexed by their unique GGUF names.
    pub tensors: BTreeMap<String, Tensor>,
    /// Lowercase SHA-256 of the verified file or canonical generated GGUF.
    pub identity: String,
}

/// Verified, bounded GGUF metadata for tokenization without executable weights.
///
/// The embedding descriptor supplies the vocabulary size, but no tensor payload
/// is materialized or checked for executable values. This is not graph or
/// inference validation; unsupported graph architectures and executable tensor
/// encodings can still provide a usable tokenizer.
#[derive(Debug)]
pub struct ModelMetadata {
    /// Complete retained scalar, tokenizer, array and template metadata.
    pub metadata: Metadata,
    /// Vocabulary extent from the rank-two `token_embd.weight` descriptor.
    pub vocab_size: usize,
    /// Lowercase SHA-256 identity verified before parsing the model header.
    pub identity: String,
}

impl ModelMetadata {
    /// Verifies the complete artifact and retains only its bounded GGUF header.
    ///
    /// Hashing streams through the complete file, then the same verified handle
    /// parses metadata and validates tensor descriptors. `memory_limit` bounds
    /// conservative retained-header storage, excluding bounded parser scratch
    /// and subsequent tokenizer allocations. Artifact hashing reads the whole
    /// file, and the bounded header buffer may read ahead into tensor data;
    /// neither materializes tensors, constructs a graph, nor validates inference.
    ///
    /// # Errors
    /// Rejects identity mismatches, malformed or over-budget GGUF headers,
    /// unsupported dialect encodings, a missing/rank-mismatched token embedding,
    /// a vocabulary extent outside `usize`, allocation failures and I/O errors.
    pub fn load(
        path: &Path,
        expected_bytes: u64,
        expected_sha256: &str,
        memory_limit: usize,
    ) -> Result<Self> {
        let reader = VerifiedModel::open(
            path,
            PRISM_DIALECT,
            expected_bytes,
            expected_sha256,
            memory_limit,
        )?;
        let vocab_size = reader
            .tensors()
            .iter()
            .find(|tensor| tensor.name == "token_embd.weight")
            .filter(|tensor| tensor.dimensions.len() == 2)
            .and_then(|tensor| tensor.dimensions.get(1))
            .copied()
            .ok_or_else(|| invalid("missing token embedding vocabulary dimension"))?;
        Ok(Self {
            metadata: reader.into_metadata(),
            vocab_size: usize::try_from(vocab_size)?,
            identity: expected_sha256.to_ascii_lowercase(),
        })
    }
}

impl Model {
    /// Loads the explicitly pinned Prism GGUF dialect within a retained budget.
    ///
    /// `memory_limit` covers a conservative header charge and all serialized
    /// payloads retained by the returned model. The parser uses fixed bounded
    /// scratch, and converting vectors to shared storage can transiently copy
    /// one tensor. This is a storage budget, not a process resident-set limit.
    /// Payloads support every currently implemented production row decoder.
    /// Finite native PQ2/PTQ1 scales include both signed zeroes.
    /// Native PQ2 supports all four two-bit codes, decoding to -1, 0, 1, 2;
    /// native PTQ1 supports every fractional trit byte, including aliases.
    ///
    /// # Errors
    /// Rejects identity mismatches, malformed GGUF, unsupported encodings,
    /// nonfinite values, insufficient
    /// memory budgets, allocation failures, and I/O failures.
    pub fn load(
        path: &Path,
        expected_bytes: u64,
        expected_sha256: &str,
        memory_limit: usize,
    ) -> Result<Self> {
        Self::load_with_progress(
            path,
            expected_bytes,
            expected_sha256,
            memory_limit,
            &mut |_| Ok(true),
        )
    }

    /// Loads a verified model with cancellable, byte-weighted progress.
    ///
    /// The callback runs before each tensor read and once at `1.0` after all
    /// payload validation, matching the pinned file loader's ordering. Returning
    /// false or an error drops all partial storage without publishing a model.
    /// Hash verification and header parsing precede the first callback.
    ///
    /// # Errors
    /// Returns the same errors as [`Self::load`], plus callback failure or
    /// cancellation, including cancellation at the final callback.
    #[allow(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "The pinned loader reports byte-weighted progress as binary32."
    )]
    pub fn load_with_progress(
        path: &Path,
        expected_bytes: u64,
        expected_sha256: &str,
        memory_limit: usize,
        progress: &mut impl FnMut(f32) -> Result<bool>,
    ) -> Result<Self> {
        let reader = VerifiedModel::open(
            path,
            PRISM_DIALECT,
            expected_bytes,
            expected_sha256,
            memory_limit,
        )?;
        Self::load_verified(reader, expected_sha256, memory_limit, progress)
    }

    #[allow(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "The pinned loader reports byte-weighted progress as binary32."
    )]
    fn load_verified(
        mut reader: VerifiedModel,
        expected_sha256: &str,
        memory_limit: usize,
        progress: &mut impl FnMut(f32) -> Result<bool>,
    ) -> Result<Self> {
        let mut remaining = memory_limit
            .checked_sub(reader.retained_bytes())
            .ok_or_else(|| invalid("model header exceeds memory budget"))?;
        let mut required = 0_usize;
        for tensor in reader.tensors() {
            if !owned::supported_kind(tensor.type_id) {
                return Err(invalid(format!(
                    "unsupported executable tensor encoding {} for {}",
                    tensor.type_id, tensor.name
                )));
            }
            required = required
                .checked_add(usize::try_from(tensor.bytes)?)
                .ok_or_else(|| invalid("model payload byte count overflow"))?;
        }
        if required > remaining {
            return Err(invalid(
                "model tensor payloads exceed remaining memory budget",
            ));
        }
        let mut tensors = BTreeMap::new();
        let mut completed = 0_usize;
        for index in 0..reader.tensors().len() {
            if !progress(completed as f32 / required as f32)? {
                return Err(invalid("model loading cancelled"));
            }
            let descriptor = &reader.tensors()[index];
            let name = descriptor.name.clone();
            let dimensions = descriptor
                .dimensions
                .iter()
                .map(|value| usize::try_from(*value))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let kind = descriptor.type_id;
            let bytes = reader.read_tensor(index, remaining)?;
            validate_payload(kind, &bytes)?;
            remaining -= bytes.len();
            completed += bytes.len();
            tensors.insert(
                name.clone(),
                Tensor {
                    name,
                    dimensions,
                    kind,
                    data: bytes.into(),
                },
            );
        }
        if !progress(1.0)? {
            return Err(invalid("model loading cancelled"));
        }
        Ok(Self {
            metadata: reader.into_metadata(),
            tensors,
            identity: expected_sha256.to_ascii_lowercase(),
        })
    }

    /// Shares identical immutable F32 rank-one `LoRA` B tensors within a model.
    ///
    /// Only names ending in `.lora_b` or `.lora_b.weight` enter the existing
    /// bounded Prism pool. Layouts include all extents and strides; a successful
    /// byte-exact hit receives an `Arc` to the retained owner's storage. The
    /// pool and its bounded copied comparison keys are dropped before return.
    /// Returns the number of tensors whose payload allocation was shared.
    pub fn share_lora_b(&mut self) -> usize {
        let mut pool = Pool::new();
        let mut owners: Vec<Arc<[u8]>> = Vec::new();
        for tensor in self.tensors.values_mut() {
            if tensor.kind != 0
                || !(tensor.name.ends_with(".lora_b") || tensor.name.ends_with(".lora_b.weight"))
            {
                continue;
            }
            let Some(layout) = lora_layout(tensor) else {
                continue;
            };
            if let Some(owner) = pool.find(&layout, &tensor.data) {
                if let Some(storage) = owners.get(owner - 1) {
                    tensor.data = Arc::clone(storage);
                }
            } else if pool.remember(&layout, &tensor.data, owners.len() + 1) {
                owners.push(Arc::clone(&tensor.data));
            }
        }
        pool.shared_tensors()
    }
}

fn lora_layout(tensor: &Tensor) -> Option<Layout> {
    if tensor.dimensions.is_empty() || tensor.dimensions.len() > 4 {
        return None;
    }
    let mut dimensions = [1_i64; 4];
    for (destination, source) in dimensions.iter_mut().zip(&tensor.dimensions) {
        *destination = i64::try_from(*source).ok()?;
    }
    let mut strides = [4_usize; 4];
    for index in 1..4 {
        strides[index] =
            strides[index - 1].checked_mul(usize::try_from(dimensions[index - 1]).ok()?)?;
    }
    let layout = Layout {
        context: 1,
        buffer_type: 1,
        cpu: true,
        f32: true,
        contiguous: true,
        dimensions,
        strides,
        byte_len: tensor.data.len(),
    };
    (layout.eligible()
        && strides[3].checked_mul(usize::try_from(dimensions[3]).ok()?)? == tensor.data.len())
    .then_some(layout)
}

fn validate_payload(kind: u32, bytes: &[u8]) -> Result<()> {
    let finite = match kind {
        0 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .all(|value| f32::from_le_bytes(*value).is_finite()),
        1 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .all(|value| u16::from_le_bytes(*value) & 0x7c00 != 0x7c00),
        30 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .all(|value| u16::from_le_bytes(*value) & 0x7f80 != 0x7f80),
        142 => {
            for block in bytes.as_chunks::<34>().0 {
                validate_prism_scale([block[0], block[1]])?;
                // Every two-bit symbol is valid for native PQ2. The ternary
                // export codec deliberately accepts a narrower symbol set.
            }
            true
        }
        143 => {
            for block in bytes.as_chunks::<28>().0 {
                validate_prism_scale([block[26], block[27]])?;
                // Every fractional trit byte has native decoder semantics;
                // canonical conversion constraints belong to the exporter.
            }
            true
        }
        _ if owned::supported_kind(kind) => {
            let (elements, width) = gguf_context::tensor_layout(kind)
                .ok_or_else(|| invalid("unsupported executable tensor encoding"))?;
            let mut decoded = Vec::new();
            for block in bytes.chunks_exact(usize::try_from(width)?) {
                crate::kernels::decode_row_into(
                    kind,
                    usize::try_from(elements)?,
                    block,
                    &mut decoded,
                )?;
                if decoded.iter().any(|value| !value.is_finite()) {
                    return Err(invalid("nonfinite executable tensor payload"));
                }
            }
            true
        }
        _ => return Err(invalid("unsupported executable tensor encoding")),
    };
    if !finite {
        return Err(invalid("nonfinite executable tensor payload"));
    }
    Ok(())
}

fn validate_prism_scale(bytes: [u8; 2]) -> Result<()> {
    if u16::from_le_bytes(bytes) & 0x7c00 == 0x7c00 {
        return Err(invalid("nonfinite executable tensor payload"));
    }
    Ok(())
}
pub mod batch_allocator;
pub mod gguf_context;
pub mod owned;
pub mod quantization_policy;
pub mod swift;
