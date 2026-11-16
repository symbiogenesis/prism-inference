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

    #[expect(
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
        let mut staging = Vec::new();
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
            reader.read_tensor_into(index, &mut staging, remaining)?;
            let len = staging.len();
            validate_payload(kind, &staging[..len])?;
            remaining -= len;
            completed += len;
            let data = Arc::<[u8]>::from(&staging[..len]);
            tensors.insert(
                name.clone(),
                Tensor {
                    name,
                    dimensions,
                    kind,
                    data,
                },
            );
        }
        drop(staging);
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

fn validate_block_scale<const N: usize>(bytes: &[u8], scale_offset: usize) -> Result<()> {
    let (blocks, remainder) = bytes.as_chunks::<N>();
    if !remainder.is_empty() {
        return Err(invalid("invalid executable tensor block length"));
    }
    for block in blocks {
        validate_prism_scale([block[scale_offset], block[scale_offset + 1]])?;
    }
    Ok(())
}

fn validate_payload(kind: u32, bytes: &[u8]) -> Result<()> {
    let finite = match kind {
        0 => {
            let (values, remainder) = bytes.as_chunks::<4>();
            if !remainder.is_empty() {
                return Err(invalid("invalid F32 executable tensor payload length"));
            }
            values
                .iter()
                .all(|value| f32::from_le_bytes(*value).is_finite())
        }
        1 => {
            let (values, remainder) = bytes.as_chunks::<2>();
            if !remainder.is_empty() {
                return Err(invalid("invalid F16 executable tensor payload length"));
            }
            values
                .iter()
                .all(|value| u16::from_le_bytes(*value) & 0x7c00 != 0x7c00)
        }
        30 => {
            let (values, remainder) = bytes.as_chunks::<2>();
            if !remainder.is_empty() {
                return Err(invalid("invalid BF16 executable tensor payload length"));
            }
            values
                .iter()
                .all(|value| u16::from_le_bytes(*value) & 0x7f80 != 0x7f80)
        }
        2 => {
            validate_block_scale::<18>(bytes, 0)?;
            true
        }
        8 => {
            validate_block_scale::<34>(bytes, 0)?;
            true
        }
        142 => {
            validate_block_scale::<34>(bytes, 0)?;
            // Every two-bit symbol is valid for native PQ2. The ternary
            // export codec deliberately accepts a narrower symbol set.
            true
        }
        143 => {
            validate_block_scale::<28>(bytes, 26)?;
            // Every fractional trit byte has native decoder semantics;
            // canonical conversion constraints belong to the exporter.
            true
        }
        _ if owned::supported_kind(kind) => {
            let (elements, width) = gguf_context::tensor_layout(kind)
                .ok_or_else(|| invalid("unsupported executable tensor encoding"))?;
            let blocks = bytes.chunks_exact(usize::try_from(width)?);
            if !blocks.remainder().is_empty() {
                return Err(invalid("invalid executable tensor block length"));
            }
            let mut decoded = Vec::new();
            for block in blocks {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEST_GGUF: AtomicU64 = AtomicU64::new(0);

    struct TempGguf {
        root: PathBuf,
        path: PathBuf,
        bytes: Vec<u8>,
    }

    impl TempGguf {
        fn create(tensors: &[(&str, &[u64], u32, &[u8])]) -> Result<Self> {
            let id = NEXT_TEST_GGUF.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir()
                .join(format!("SwiftBonsai-ModelUnit-{}-{id}", std::process::id()));
            fs::create_dir(&root)?;
            let path = root.join("model.gguf");
            let mut bytes = Vec::new();
            bytes.extend(b"GGUF");
            bytes.extend(3_u32.to_le_bytes());
            bytes.extend(u64::try_from(tensors.len())?.to_le_bytes());
            bytes.extend(1_u64.to_le_bytes());
            let key = "general.architecture";
            bytes.extend(u64::try_from(key.len())?.to_le_bytes());
            bytes.extend(key.as_bytes());
            bytes.extend(8_u32.to_le_bytes());
            let arch = "synthetic";
            bytes.extend(u64::try_from(arch.len())?.to_le_bytes());
            bytes.extend(arch.as_bytes());

            let mut offsets = Vec::with_capacity(tensors.len());
            let mut end = 0_usize;
            for (name, dims, kind, data) in tensors {
                let offset = end.div_ceil(32) * 32;
                offsets.push(offset);
                end = offset + data.len();
                bytes.extend(u64::try_from(name.len())?.to_le_bytes());
                bytes.extend(name.as_bytes());
                bytes.extend(u32::try_from(dims.len())?.to_le_bytes());
                for &dim in *dims {
                    bytes.extend(dim.to_le_bytes());
                }
                bytes.extend(kind.to_le_bytes());
                bytes.extend(u64::try_from(offset)?.to_le_bytes());
            }
            let data_offset = bytes.len().div_ceil(32) * 32;
            bytes.resize(data_offset + end, 0);
            for ((_, _, _, data), offset) in tensors.iter().zip(offsets) {
                bytes[data_offset + offset..data_offset + offset + data.len()]
                    .copy_from_slice(data);
            }
            fs::write(&path, &bytes)?;
            Ok(Self { root, path, bytes })
        }

        fn sha256_hex(&self) -> Result<String> {
            const HEX: &[u8; 16] = b"0123456789abcdef";
            let digest = bonsai_native_hash::hash_bytes(&self.bytes)?;
            let mut out = String::with_capacity(64);
            for byte in digest {
                out.push(char::from(HEX[usize::from(byte >> 4)]));
                out.push(char::from(HEX[usize::from(byte & 15)]));
            }
            Ok(out)
        }
    }

    impl Drop for TempGguf {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.path);
            let _ = fs::remove_dir(&self.root);
        }
    }

    #[test]
    fn validate_payload_fast_paths_q4_0_and_q8_0_scales_and_lengths() -> Result<()> {
        let mut q4_valid = [0x5a_u8; 36];
        q4_valid[0..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        q4_valid[18..20].copy_from_slice(&0x8000_u16.to_le_bytes());
        validate_payload(2, &q4_valid)?;

        let mut q8_valid = [0x7f_u8; 68];
        q8_valid[0..2].copy_from_slice(&0x0000_u16.to_le_bytes());
        q8_valid[34..36].copy_from_slice(&0xbc00_u16.to_le_bytes());
        validate_payload(8, &q8_valid)?;

        for bad_scale in [0x7c00_u16, 0xfc00, 0x7e01, 0xfe01] {
            let mut q4_bad = q4_valid;
            q4_bad[18..20].copy_from_slice(&bad_scale.to_le_bytes());
            assert!(validate_payload(2, &q4_bad).is_err());

            let mut q8_bad = q8_valid;
            q8_bad[34..36].copy_from_slice(&bad_scale.to_le_bytes());
            assert!(validate_payload(8, &q8_bad).is_err());
        }

        for bad_len in [1, 17, 19, 35] {
            assert!(validate_payload(2, &vec![0; bad_len]).is_err());
        }
        for bad_len in [1, 33, 35, 67] {
            assert!(validate_payload(8, &vec![0; bad_len]).is_err());
        }
        Ok(())
    }

    #[test]
    fn load_verified_single_staging_handles_growing_and_shrinking_tensors() -> Result<()> {
        let mut q4 = [0x12_u8; 18];
        q4[0..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        let mut q8_large = [0x01_u8; 68];
        q8_large[0..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        q8_large[34..36].copy_from_slice(&0x4000_u16.to_le_bytes());
        let f32_small = 1.5_f32.to_le_bytes();

        let fixture = TempGguf::create(&[
            ("a.q4", &[32], 2, &q4),
            ("b.q8", &[64], 8, &q8_large),
            ("c.f32", &[1], 0, &f32_small),
        ])?;
        let sha = fixture.sha256_hex()?;
        let mut progress_steps = Vec::new();
        let reader = VerifiedModel::open(
            &fixture.path,
            PRISM_DIALECT,
            u64::try_from(fixture.bytes.len())?,
            &sha,
            1024 * 1024,
        )?;
        let model = Model::load_verified(reader, &sha, 1024 * 1024, &mut |fraction| {
            progress_steps.push(fraction);
            Ok(true)
        })?;
        assert_eq!(&*model.tensors["a.q4"].data, &q4);
        assert_eq!(&*model.tensors["b.q8"].data, &q8_large);
        assert_eq!(&*model.tensors["c.f32"].data, &f32_small);
        assert_eq!(progress_steps.len(), 4);
        assert_eq!(progress_steps.last().copied(), Some(1.0));
        Ok(())
    }
}
pub mod batch_allocator;
pub mod gguf_context;
pub mod lora;
pub mod owned;
pub mod quantization_policy;
pub mod swift;
