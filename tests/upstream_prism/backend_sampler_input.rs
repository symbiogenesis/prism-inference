//! Source-specific input selection for the backend-sampler mechanism tests.
//!
//! The source accepts an arbitrary model. Its fixed token 3834 needs at least
//! 3835 entries; the next power of two is 4096. Preserve that prefix of the
//! already locked Llama vocabulary and generate a small, ordinary F32 Llama.
//! No production observations or expected tokens enter this constructor.
use super::super::lora_gguf::{self, Metadata, Tensor};
use super::super::windows_random::{Random, hash_name64};
use super::{Result, invalid};
use bonsai_gguf::{Metadata as SourceMetadata, MetadataArray, MetadataValue, VerifiedModel};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

const SOURCE_BYTES: u64 = 19_077_344;
const SOURCE_SHA: &str = "66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739";
const MODEL_SHA: &str = "ec82660cf6446ef5b842128ed4a0a6ba985fc77c0a0df0d3e884e8558e7caed2";
const VOCABULARY: usize = 4096;
const EMBEDDING: u64 = 64;
const FEED_FORWARD: u64 = 128;
const SEED: u64 = 12345;

fn sha256(path: &Path, bytes: u64) -> Result<String> {
    let mut input = File::open(path)?.take(bytes);
    let mut hash = Sha256::new();
    let mut buffer = [0; 16384];
    loop {
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    if input.limit() != 0 {
        return Err(invalid(
            "generated backend sampler hash extent is truncated",
        ));
    }
    Ok(super::hex(&hash.finalize()))
}

fn vocabulary(source: &SourceMetadata, output: &mut Metadata) -> Result<()> {
    if source
        .get("tokenizer.ggml.model")
        .and_then(MetadataValue::as_str)
        != Some("llama")
    {
        return Err(invalid(
            "backend sampler input needs the locked Llama vocabulary",
        ));
    }
    for name in ["bos", "eos", "unknown"] {
        let id = source
            .get(&format!("tokenizer.ggml.{name}_token_id"))
            .and_then(MetadataValue::as_u64)
            .ok_or_else(|| invalid("locked backend sampler special token is absent"))?;
        if id >= u64::try_from(VOCABULARY)? {
            return Err(invalid("backend sampler prefix omits a special token"));
        }
    }
    for (name, value) in source {
        if !name.starts_with("tokenizer.ggml.") {
            continue;
        }
        match value {
            MetadataValue::String(value) => output.text(name, value)?,
            MetadataValue::Bool(value) => output.boolean(name, *value),
            MetadataValue::U32(value) => output.uint(name, *value),
            MetadataValue::Array(MetadataArray::String(values))
                if name == "tokenizer.ggml.tokens" =>
            {
                output.strings(
                    name,
                    values
                        .get(..VOCABULARY)
                        .ok_or_else(|| invalid("locked token prefix is truncated"))?,
                )?;
            }
            MetadataValue::Array(MetadataArray::F32(values)) if name == "tokenizer.ggml.scores" => {
                output.floats(
                    name,
                    values
                        .get(..VOCABULARY)
                        .ok_or_else(|| invalid("locked score prefix is truncated"))?,
                )?;
            }
            MetadataValue::Array(MetadataArray::I32(values))
                if name == "tokenizer.ggml.token_type" =>
            {
                output.integers(
                    name,
                    values
                        .get(..VOCABULARY)
                        .ok_or_else(|| invalid("locked type prefix is truncated"))?,
                )?;
            }
            _ => {
                return Err(invalid(format!(
                    "unsupported locked backend sampler vocabulary field {name}"
                )));
            }
        }
    }
    for name in ["tokens", "scores", "token_type"] {
        if source
            .get(&format!("tokenizer.ggml.{name}"))
            .and_then(MetadataValue::as_array)
            .is_none()
        {
            return Err(invalid(
                "locked backend sampler vocabulary arrays are incomplete",
            ));
        }
    }
    Ok(())
}

fn tensors() -> Vec<Tensor> {
    let mut result = vec![
        Tensor {
            name: "token_embd.weight".into(),
            shape: vec![4096, EMBEDDING],
        },
        Tensor {
            name: "output_norm.weight".into(),
            shape: vec![EMBEDDING],
        },
        Tensor {
            name: "output.weight".into(),
            shape: vec![4096, EMBEDDING],
        },
    ];
    for layer in 0..2 {
        for (name, shape) in [
            ("attn_norm", vec![EMBEDDING]),
            ("attn_q", vec![EMBEDDING, EMBEDDING]),
            ("attn_k", vec![EMBEDDING, EMBEDDING]),
            ("attn_v", vec![EMBEDDING, EMBEDDING]),
            ("attn_output", vec![EMBEDDING, EMBEDDING]),
            ("ffn_norm", vec![EMBEDDING]),
            ("ffn_gate", vec![FEED_FORWARD, EMBEDDING]),
            ("ffn_down", vec![EMBEDDING, FEED_FORWARD]),
            ("ffn_up", vec![FEED_FORWARD, EMBEDDING]),
        ] {
            result.push(Tensor {
                name: format!("blk.{layer}.{name}.weight"),
                shape,
            });
        }
    }
    result
}

fn weights(tensor: &Tensor) -> Result<Vec<u8>> {
    let count = usize::try_from(tensor.bytes()? / 4)?;
    let seed = (hash_name64(&tensor.name) ^ SEED).to_le_bytes();
    let mut random = Random::new(u32::from_le_bytes(seed[..4].try_into()?));
    // Standard linear-layer uniform initialization: +/- 1/sqrt(fan_in).
    // Binary32 constants fix the input bytes independently of host sqrt/libm.
    let bound = match tensor.shape.last() {
        Some(&EMBEDDING) => 0.125_f32,
        Some(&FEED_FORWARD) => f32::from_bits(0x3db5_04f3),
        _ => return Err(invalid("backend sampler tensor fan-in is unsupported")),
    };
    let norm = tensor.name.ends_with("_norm.weight");
    Ok((0..count)
        .flat_map(|_| {
            let mut value = if norm {
                1.0
            } else {
                random.uniform_f32(-bound, bound)
            };
            if value == 0.0 {
                value = f32::EPSILON;
            }
            value.to_le_bytes()
        })
        .collect())
}

pub(super) fn create(source: &Path, directory: &Path) -> Result<(PathBuf, Value)> {
    let source = VerifiedModel::open(source, "Stock", SOURCE_BYTES, SOURCE_SHA, 16 * 1024 * 1024)?;
    let mut metadata = Metadata::new();
    vocabulary(source.metadata(), &mut metadata)?;
    metadata.text("general.architecture", "llama")?;
    metadata.text("general.name", "backend-sampler-input-v1")?;
    metadata.uint("general.file_type", 0);
    metadata.uint("llama.vocab_size", 4096);
    metadata.uint("llama.context_length", 512);
    metadata.uint("llama.embedding_length", 64);
    metadata.uint("llama.feed_forward_length", 128);
    metadata.uint("llama.block_count", 2);
    metadata.uint("llama.attention.head_count", 4);
    metadata.uint("llama.attention.head_count_kv", 4);
    metadata.uint("llama.rope.dimension_count", 16);
    metadata.float("llama.rope.freq_base", 10_000.0);
    metadata.float("llama.attention.layer_norm_rms_epsilon", 1e-5);
    let tensors = tensors();
    let mut observations = Vec::new();
    let path = directory.join("backend-sampler-input-v1.gguf");
    let bytes = lora_gguf::write(&path, &metadata, &tensors, |index| {
        let tensor = &tensors[index];
        let data = weights(tensor)?;
        observations.push(json!({"Name":tensor.name,"Shape":tensor.shape,"Bytes":data.len(),"Sha256":super::hex(&Sha256::digest(&data))}));
        Ok(data)
    })?;
    let digest = sha256(&path, bytes)?;
    if digest != MODEL_SHA || bytes != 2_514_944 {
        return Err(invalid(
            "generated backend sampler input differs from its version1 pin",
        ));
    }
    let mut payload = 0_u64;
    for tensor in &tensors {
        payload = payload
            .checked_add(tensor.bytes()?.next_multiple_of(32))
            .ok_or_else(|| invalid("backend sampler payload extent overflow"))?;
    }
    let header = bytes
        .checked_sub(payload)
        .ok_or_else(|| invalid("backend sampler header extent is invalid"))?;
    let receipt = json!({
        "SchemaVersion":1,"Kind":"GeneratedBackendSamplerInput","SourceModelSha256":SOURCE_SHA,"SourceModelBytes":SOURCE_BYTES,
        "SourceVocabularyPrefix":4096,"SourceMinimumVocabulary":3835,
        "Architecture":"llama","Embedding":64,"FeedForward":128,"Layers":2,"Heads":4,"Context":512,
        "Initializer":"MT19937/name-FNV64-xor-seed; linear uniform +/-1/sqrt(fan-in); RMS norm ones; replace exact zero with f32::EPSILON",
        "Seed":SEED,"Tensors":observations,"Bytes":bytes,"Sha256":digest,
        "MetadataAndTensorDirectoryBytes":header,"MetadataAndTensorDirectorySha256":sha256(&path,header)?
    });
    fs::write(
        directory.join("backend-sampler-input.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    Ok((path, receipt))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixture_preserves_source_token_domain_and_has_nonzero_reproducible_weights() -> Result<()> {
        let tensors = tensors();
        assert_eq!(tensors.len(), 21);
        assert_eq!(tensors[0].shape, [4096, 64]);
        assert!(tensors[0].shape[0] > 3834);
        let tensor = &tensors[6];
        let first = weights(tensor)?;
        assert_eq!(first, weights(tensor)?);
        assert_ne!(first, weights(&tensors[7])?);
        assert!(
            first
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| f32::from_le_bytes(*word))
                .all(|value| value.is_finite() && value != 0.0)
        );
        let mut metadata = SourceMetadata::new();
        metadata.insert(
            "tokenizer.ggml.model".into(),
            MetadataValue::String("llama".into()),
        );
        assert!(vocabulary(&metadata, &mut Metadata::new()).is_err());
        Ok(())
    }
}
