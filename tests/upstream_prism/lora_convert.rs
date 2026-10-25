//! Safe Rust subset of the pinned HF/PEFT converters covering every original
//! input in test-lora-conversion-inference.sh. No production observations.
use super::lora_gguf::{self, Metadata, Tensor};
use super::lora_inputs::Inputs;
use super::{Result, invalid};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

#[derive(Clone)]
struct SourceTensor {
    shape: Vec<u64>,
    begin: u64,
    length: u64,
}
pub(super) struct Safetensors {
    path: PathBuf,
    offset: u64,
    tensors: BTreeMap<String, SourceTensor>,
}
impl Safetensors {
    pub(super) fn open(path: &Path) -> Result<Self> {
        let mut input = File::open(path)?;
        let length = input.metadata()?.len();
        let mut word = [0; 8];
        input.read_exact(&mut word)?;
        let header_length = u64::from_le_bytes(word);
        if header_length == 0 || header_length > 1024 * 1024 || length > 128 * 1024 * 1024 {
            return Err(invalid("safetensors input extent invalid"));
        }
        let offset = 8_u64
            .checked_add(header_length)
            .filter(|&v| v <= length)
            .ok_or_else(|| invalid("safetensors header truncated"))?;
        let mut header = vec![0; usize::try_from(header_length)?];
        input.read_exact(&mut header)?;
        let header: Value = serde_json::from_slice(&header)?;
        let mut tensors = BTreeMap::new();
        let mut spans = Vec::new();
        for (name, value) in header
            .as_object()
            .ok_or_else(|| invalid("safetensors header not object"))?
        {
            if name == "__metadata__" {
                continue;
            }
            if value["dtype"] != "F32" || name.len() > 256 {
                return Err(invalid(
                    "source conversion supports only original F32 tensors",
                ));
            }
            let shape = value["shape"]
                .as_array()
                .ok_or_else(|| invalid("safetensors shape absent"))?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .filter(|&n| n > 0)
                        .ok_or_else(|| invalid("invalid safetensors dimension"))
                })
                .collect::<Result<Vec<_>>>()?;
            let spans_value = value["data_offsets"]
                .as_array()
                .filter(|v| v.len() == 2)
                .ok_or_else(|| invalid("safetensors offsets absent"))?;
            let begin = spans_value[0]
                .as_u64()
                .ok_or_else(|| invalid("invalid safetensors begin"))?;
            let end = spans_value[1]
                .as_u64()
                .filter(|&end| end >= begin && end <= length - offset)
                .ok_or_else(|| invalid("invalid safetensors end"))?;
            if (Tensor {
                name: name.clone(),
                shape: shape.clone(),
            })
            .bytes()?
                != end - begin
            {
                return Err(invalid("safetensors shape/data extent differs"));
            }
            tensors.insert(
                name.clone(),
                SourceTensor {
                    shape,
                    begin,
                    length: end - begin,
                },
            );
            spans.push((begin, end));
        }
        spans.sort_unstable();
        let mut cursor = 0;
        for (begin, end) in spans {
            if begin != cursor {
                return Err(invalid("safetensors data has gap or overlap"));
            }
            cursor = end;
        }
        if cursor != length - offset || tensors.is_empty() || tensors.len() > 1024 {
            return Err(invalid("safetensors extent/count differs"));
        }
        Ok(Self {
            path: path.into(),
            offset,
            tensors,
        })
    }
    fn read(&self, name: &str) -> Result<Vec<u8>> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| invalid("conversion source tensor absent"))?;
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(self.offset + tensor.begin))?;
        let mut data = vec![0; usize::try_from(tensor.length)?];
        file.read_exact(&mut data)?;
        if data
            .as_chunks::<4>()
            .0
            .iter()
            .any(|v| !f32::from_le_bytes(*v).is_finite())
        {
            return Err(invalid("source F32 tensor contains nonfinite input"));
        }
        Ok(data)
    }
}
fn uint(config: &Value, key: &str) -> Result<u32> {
    Ok(u32::try_from(config[key].as_u64().ok_or_else(|| {
        invalid(format!("HF integer missing: {key}"))
    })?)?)
}
fn float(config: &Value, key: &str) -> Result<f32> {
    let value = config[key]
        .as_f64()
        .ok_or_else(|| invalid(format!("HF float missing: {key}")))?;
    #[allow(
        clippy::cast_possible_truncation,
        clippy::as_conversions,
        reason = "GGUF F32 metadata matches the original converter's explicit float32 serialization"
    )]
    let value = value as f32;
    if !value.is_finite() {
        return Err(invalid("HF float nonfinite"));
    }
    Ok(value)
}
fn arch(architecture: &str) -> Result<&'static str> {
    match architecture {
        "Gemma2ForCausalLM" => Ok("gemma2"),
        "LlamaForCausalLM" => Ok("llama"),
        "Phi3ForCausalLM" => Ok("phi3"),
        _ => Err(invalid("unsupported original LoRA architecture")),
    }
}
pub(super) fn mapped(architecture: &str, name: &str) -> Result<String> {
    let global = match name {
        "model.embed_tokens.weight" => Some("token_embd.weight"),
        "model.norm.weight" => Some("output_norm.weight"),
        "lm_head.weight" => Some("output.weight"),
        _ => None,
    };
    if let Some(name) = global {
        return Ok(name.into());
    }
    let rest = name
        .strip_prefix("model.layers.")
        .ok_or_else(|| invalid(format!("unmapped HF tensor: {name}")))?;
    let (layer, rest) = rest
        .split_once('.')
        .ok_or_else(|| invalid("HF layer index absent"))?;
    let layer = layer.parse::<u32>()?;
    let component = match rest {
        "input_layernorm.weight" => "attn_norm",
        "post_attention_layernorm.weight" => {
            if architecture == "Gemma2ForCausalLM" {
                "post_attention_norm"
            } else {
                "ffn_norm"
            }
        }
        "pre_feedforward_layernorm.weight" => "ffn_norm",
        "post_feedforward_layernorm.weight" => "post_ffw_norm",
        "self_attn.q_proj.weight" => "attn_q",
        "self_attn.k_proj.weight" => "attn_k",
        "self_attn.v_proj.weight" => "attn_v",
        "self_attn.o_proj.weight" => "attn_output",
        "self_attn.qkv_proj.weight" => "attn_qkv",
        "mlp.gate_proj.weight" => "ffn_gate",
        "mlp.down_proj.weight" => "ffn_down",
        "mlp.up_proj.weight" | "mlp.gate_up_proj.weight" => "ffn_up",
        _ => return Err(invalid(format!("unmapped HF layer tensor: {name}"))),
    };
    Ok(format!("blk.{layer}.{component}.weight"))
}
/// Equivalent to reshape(heads,2,rows/heads/2,columns).swapaxes(1,2).
pub(super) fn permute_rows(data: &[u8], shape: &[u64], heads: u32) -> Result<Vec<u8>> {
    if shape.len() != 2 || heads == 0 || !shape[0].is_multiple_of(u64::from(heads) * 2) {
        return Err(invalid("invalid Llama Q/K permutation geometry"));
    }
    let width = usize::try_from(
        shape[1]
            .checked_mul(4)
            .ok_or_else(|| invalid("permutation width overflow"))?,
    )?;
    let rows = usize::try_from(shape[0])?;
    let heads = usize::try_from(heads)?;
    let half = rows / heads / 2;
    if rows.checked_mul(width) != Some(data.len()) {
        return Err(invalid("permutation buffer extent differs"));
    }
    let mut result = vec![0; data.len()];
    for head in 0..heads {
        for pair in 0..half {
            for part in 0..2 {
                let from = (head * 2 * half + part * half + pair) * width;
                let to = (head * 2 * half + pair * 2 + part) * width;
                result[to..to + width].copy_from_slice(&data[from..from + width]);
            }
        }
    }
    Ok(result)
}
fn transform(
    architecture: &str,
    config: &Value,
    name: &str,
    shape: &[u64],
    mut data: Vec<u8>,
    adapter_a: bool,
) -> Result<Vec<u8>> {
    if architecture == "Gemma2ForCausalLM" && name.ends_with("norm.weight") {
        for value in data.as_chunks_mut::<4>().0 {
            let float = f32::from_le_bytes(*value) + 1.0;
            value.copy_from_slice(&float.to_le_bytes());
        }
    }
    if architecture == "LlamaForCausalLM" && !adapter_a {
        let heads = if name.ends_with("q_proj.weight") {
            Some(uint(config, "num_attention_heads")?)
        } else if name.ends_with("k_proj.weight") {
            Some(uint(config, "num_key_value_heads")?)
        } else {
            None
        };
        if let Some(heads) = heads {
            data = permute_rows(&data, shape, heads)?;
        }
    }
    Ok(data)
}
fn metadata(
    inputs: &Inputs,
    architecture: &str,
    config: &Value,
    adapter: bool,
) -> Result<Metadata> {
    let architecture_name = arch(architecture)?;
    let mut result = Metadata::new();
    result.text("general.architecture", architecture_name)?;
    result.uint("general.file_type", 0);
    result.uint("general.quantization_version", 2);
    if adapter {
        result.text("general.type", "adapter")?;
        result.text("adapter.type", "lora")?;
        let lora = inputs.json(&format!(
            "{architecture}/hidden_size=64/lora/adapter_config.json"
        ))?;
        if lora["r"] != 32
            || lora["peft_type"] != "LORA"
            || lora["fan_in_fan_out"] != false
            || lora["use_dora"] != false
            || lora["use_rslora"] != false
            || lora["bias"] != "none"
            || !lora["rank_pattern"]
                .as_object()
                .is_some_and(serde_json::Map::is_empty)
            || !lora["alpha_pattern"]
                .as_object()
                .is_some_and(serde_json::Map::is_empty)
        {
            return Err(invalid("LoRA config outside source conversion contract"));
        }
        result.float("adapter.lora.alpha", float(&lora, "lora_alpha")?);
        return Ok(result);
    }
    result.text("general.type", "model")?;
    for (key, hf) in [
        ("context_length", "max_position_embeddings"),
        ("embedding_length", "hidden_size"),
        ("feed_forward_length", "intermediate_size"),
        ("block_count", "num_hidden_layers"),
        ("attention.head_count", "num_attention_heads"),
        ("attention.head_count_kv", "num_key_value_heads"),
    ] {
        result.uint(&format!("{architecture_name}.{key}"), uint(config, hf)?);
    }
    result.float(
        &format!("{architecture_name}.attention.layer_norm_rms_epsilon"),
        float(config, "rms_norm_eps")?,
    );
    if architecture == "Gemma2ForCausalLM" {
        let head = uint(config, "head_dim")?;
        result.uint("gemma2.attention.key_length", head);
        result.uint("gemma2.attention.value_length", head);
        result.float(
            "gemma2.attn_logit_softcapping",
            float(config, "attn_logit_softcapping")?,
        );
        result.float(
            "gemma2.final_logit_softcapping",
            float(config, "final_logit_softcapping")?,
        );
        result.uint(
            "gemma2.attention.sliding_window",
            uint(config, "sliding_window")?,
        );
    } else {
        result.uint(
            &format!("{architecture_name}.rope.dimension_count"),
            uint(config, "hidden_size")? / uint(config, "num_attention_heads")?,
        );
        result.float(
            &format!("{architecture_name}.rope.freq_base"),
            float(config, "rope_theta")?,
        );
        if architecture == "LlamaForCausalLM" {
            result.uint("llama.vocab_size", uint(config, "vocab_size")?);
        } else {
            result.uint(
                "phi3.rope.scaling.original_context_length",
                uint(config, "original_max_position_embeddings")?,
            );
            result.uint(
                "phi3.attention.sliding_window",
                uint(config, "sliding_window")?,
            );
        }
    }
    super::lora_vocab::convert(inputs, architecture, config, &mut result)?;
    Ok(result)
}
pub(super) struct Converted {
    pub(super) base: PathBuf,
    pub(super) adapter: PathBuf,
    pub(super) receipt: Value,
}
#[allow(
    clippy::too_many_lines,
    reason = "keep original base and paired adapter transformations together with full tensor coverage checks"
)]
pub(super) fn convert(inputs: &Inputs, architecture: &str, directory: &Path) -> Result<Converted> {
    let prefix = format!("{architecture}/hidden_size=64");
    let config = inputs.json(&format!("{prefix}/base/config.json"))?;
    if config["architectures"][0] != architecture
        || config["hidden_size"] != 64
        || config["torch_dtype"] != "float32"
    {
        return Err(invalid("original LoRA architecture input differs"));
    }
    let base = Safetensors::open(inputs.path(&format!("{prefix}/base/model.safetensors"))?)?;
    let adapter =
        Safetensors::open(inputs.path(&format!("{prefix}/lora/adapter_model.safetensors"))?)?;
    let mut base_names = Vec::new();
    let mut base_tensors = Vec::new();
    for (name, tensor) in &base.tensors {
        if architecture == "Gemma2ForCausalLM" && name == "lm_head.weight" {
            continue;
        }
        base_names.push(name.clone());
        base_tensors.push(Tensor {
            name: mapped(architecture, name)?,
            shape: tensor.shape.clone(),
        });
    }
    let mut pairs = BTreeMap::<String, (Option<String>, Option<String>)>::new();
    let mut ignored = 0;
    for name in adapter.tensors.keys() {
        let name_without_prefix = name
            .strip_prefix("base_model.model.")
            .ok_or_else(|| invalid("LoRA base prefix missing"))?;
        if name_without_prefix.ends_with(".base_layer.weight") {
            ignored += 1;
            continue;
        }
        let (tensor, a) = if let Some(base) = name_without_prefix.strip_suffix(".lora_A.weight") {
            (format!("{base}.weight"), true)
        } else if let Some(base) = name_without_prefix.strip_suffix(".lora_B.weight") {
            (format!("{base}.weight"), false)
        } else {
            return Err(invalid("unexpected original LoRA tensor"));
        };
        let pair = pairs.entry(tensor).or_default();
        let slot = if a { &mut pair.0 } else { &mut pair.1 };
        if slot.replace(name.clone()).is_some() {
            return Err(invalid("duplicate LoRA factor"));
        }
    }
    let mut factor_names = Vec::new();
    let mut factor_tensors = Vec::new();
    let mut source_names = Vec::new();
    for (name, (a, b)) in pairs {
        let a = a.ok_or_else(|| invalid("LoRA factor A absent"))?;
        let b = b.ok_or_else(|| invalid("LoRA factor B absent"))?;
        let shape_a = &adapter.tensors[&a].shape;
        let shape_b = &adapter.tensors[&b].shape;
        let source = &base
            .tensors
            .get(&name)
            .ok_or_else(|| invalid("LoRA destination missing in base"))?
            .shape;
        if shape_a.len() != 2
            || shape_b.len() != 2
            || source.len() != 2
            || shape_a[0] != shape_b[1]
            || shape_b[0] != source[0]
            || shape_a[1] != source[1]
        {
            return Err(invalid("LoRA A/B/base geometry differs"));
        }
        let destination = mapped(architecture, &name)?;
        for (factor, suffix, shape) in [(a, "a", shape_a), (b, "b", shape_b)] {
            factor_names.push(factor);
            source_names.push((name.clone(), suffix == "a"));
            factor_tensors.push(Tensor {
                name: format!("{destination}.lora_{suffix}"),
                shape: shape.clone(),
            });
        }
    }
    let expected = match architecture {
        "Gemma2ForCausalLM" => (288, 312, 0),
        "LlamaForCausalLM" => (291, 386, 1),
        "Phi3ForCausalLM" => (195, 66, 1),
        _ => return Err(invalid("unrecognized LoRA architecture")),
    };
    if (base_tensors.len(), factor_tensors.len(), ignored) != expected {
        return Err(invalid("original LoRA tensor coverage differs"));
    }
    let base_path = directory.join(format!("{architecture}-Base-F32.gguf"));
    let adapter_path = directory.join(format!("{architecture}-Lora-F32-LoRA.gguf"));
    let base_bytes = lora_gguf::write(
        &base_path,
        &metadata(inputs, architecture, &config, false)?,
        &base_tensors,
        |index| {
            let name = &base_names[index];
            transform(
                architecture,
                &config,
                name,
                &base.tensors[name].shape,
                base.read(name)?,
                false,
            )
        },
    )?;
    let adapter_bytes = lora_gguf::write(
        &adapter_path,
        &metadata(inputs, architecture, &config, true)?,
        &factor_tensors,
        |index| {
            let factor = &factor_names[index];
            let (name, a) = &source_names[index];
            transform(
                architecture,
                &config,
                name,
                &adapter.tensors[factor].shape,
                adapter.read(factor)?,
                *a,
            )
        },
    )?;
    Ok(Converted {
        base: base_path,
        adapter: adapter_path,
        receipt: json!({"Architecture":architecture,"BaseTensors":base_tensors.len(),"AdapterTensors":factor_tensors.len(),"IgnoredBaseLayerTensors":ignored,"BaseBytes":base_bytes,"AdapterBytes":adapter_bytes,"Converter":"safe-rust-original-f32-inputs","GemmaNormShift":architecture=="Gemma2ForCausalLM","LlamaQKPermutation":architecture=="LlamaForCausalLM"}),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn focused_lora_conversion_input_guard() -> Result<()> {
        let directory = std::env::temp_dir().join(format!(
            "SwiftBonsai-LoRA-input-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        std::fs::create_dir(&directory)?;
        let path = directory.join("input.safetensors");
        let mut header = serde_json::to_vec(
            &json!({"model.layers.0.self_attn.q_proj.weight":{"dtype":"F32","shape":[8,1],"data_offsets":[0,32]}}),
        )?;
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let raw = (0_u8..8)
            .flat_map(|v| f32::from(v).to_le_bytes())
            .collect::<Vec<_>>();
        let mut file = u64::try_from(header.len())?.to_le_bytes().to_vec();
        file.extend(&header);
        file.extend(&raw);
        std::fs::write(&path, &file)?;
        let source = Safetensors::open(&path)?;
        if source.read("model.layers.0.self_attn.q_proj.weight")? != raw {
            return Err(invalid("F32 transport changed input bytes"));
        }
        let permuted = permute_rows(&raw, &[8, 1], 2)?;
        let expected = [0_u8, 2, 1, 3, 4, 6, 5, 7]
            .into_iter()
            .flat_map(|v| f32::from(v).to_le_bytes())
            .collect::<Vec<_>>();
        if permuted != expected || permute_rows(&raw, &[7, 1], 2).is_ok() {
            return Err(invalid("Q/K row permutation or extent rejection differs"));
        }
        let shifted = transform(
            "Gemma2ForCausalLM",
            &Value::Null,
            "model.norm.weight",
            &[8],
            raw.clone(),
            false,
        )?;
        let expected = (1_u8..9)
            .flat_map(|v| f32::from(v).to_le_bytes())
            .collect::<Vec<_>>();
        if shifted != expected
            || mapped(
                "Phi3ForCausalLM",
                "model.layers.0.self_attn.qkv_proj.weight",
            )? != "blk.0.attn_qkv.weight"
        {
            return Err(invalid("Gemma norm shift or Phi mapping differs"));
        }
        let piece = [0x0a, 10, 0x0a, 1, b'a', 0x15, 0, 0, 0, 0x3f, 0x18, 4];
        let (tokens, scores, kinds) = super::super::lora_vocab::sentencepiece(&piece, 2)?;
        if tokens != ["a", "[PAD1]"]
            || scores != [0.5, -10000.0]
            || kinds != [1, 5]
            || super::super::lora_vocab::sentencepiece(&piece[..11], 2).is_ok()
        {
            return Err(invalid("SP protobuf vocabulary/guard differs"));
        }
        let output = directory.join("input.gguf");
        let mut metadata = Metadata::new();
        metadata.text("general.architecture", "llama")?;
        let tensors = [Tensor {
            name: "x".into(),
            shape: vec![8, 1],
        }];
        lora_gguf::write(&output, &metadata, &tensors, |_| Ok(raw.clone()))?;
        let result = std::fs::read(&output)?;
        if !result.starts_with(b"GGUF") || result[result.len() - 32..] != raw {
            return Err(invalid("independent F32 GGUF writing differs"));
        }
        file.pop();
        std::fs::write(&path, file)?;
        if Safetensors::open(&path).is_ok() {
            return Err(invalid("truncated safetensors accepted"));
        }
        if super::super::lora_conversion::normalized(" \t abc\n\n") != "abc"
            || super::super::lora_conversion::source_prefix("aé\nz", "xy") != "aé"
        {
            return Err(invalid(
                "source shell normalization/character length differs",
            ));
        }
        for name in ["input.safetensors", "input.gguf"] {
            std::fs::remove_file(directory.join(name))?;
        }
        std::fs::remove_dir(directory)?;
        Ok(())
    }
}
