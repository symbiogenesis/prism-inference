//! Independent small compatible GGUF inputs for `DSpark` source schedules.
//! This is a synthetic protocol fixture, not a model quality measurement.
use super::engine::Engine;
use super::windows_random::{Normal, Random, hash_name64};
use super::{Result, invalid};
use serde_json::{Value, json};
use std::io::Write;
use std::path::{Path, PathBuf};
#[derive(Clone)]
struct Tensor {
    name: String,
    dimensions: Vec<u64>,
    data: Vec<u8>,
}
struct Entry {
    name: String,
    kind: u32,
    data: Vec<u8>,
}
pub(super) struct Pair {
    pub(super) target: PathBuf,
    pub(super) drafter: PathBuf,
    pub(super) receipt: Value,
}
fn string(bytes: &mut Vec<u8>, value: &str) -> Result<()> {
    bytes.extend_from_slice(&u64::try_from(value.len())?.to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
    Ok(())
}
fn scalar(name: &str, kind: u32, data: Vec<u8>) -> Entry {
    Entry {
        name: name.into(),
        kind,
        data,
    }
}
fn text(name: &str, value: &str) -> Result<Entry> {
    let mut bytes = Vec::new();
    string(&mut bytes, value)?;
    Ok(scalar(name, 8, bytes))
}
fn u(name: &str, value: u32) -> Entry {
    scalar(name, 4, value.to_le_bytes().to_vec())
}
fn f(name: &str, value: f32) -> Entry {
    scalar(name, 6, value.to_bits().to_le_bytes().to_vec())
}
fn b(name: &str, value: bool) -> Entry {
    scalar(name, 7, vec![u8::from(value)])
}
fn array(name: &str, kind: u32, count: usize, raw: &[u8]) -> Result<Entry> {
    let mut bytes = kind.to_le_bytes().to_vec();
    bytes.extend_from_slice(&u64::try_from(count)?.to_le_bytes());
    bytes.extend_from_slice(raw);
    Ok(scalar(name, 9, bytes))
}
fn ua(name: &str, values: &[u32]) -> Result<Entry> {
    array(
        name,
        4,
        values.len(),
        &values
            .iter()
            .flat_map(|n| n.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}
fn strings(name: &str, values: &[String]) -> Result<Entry> {
    let mut bytes = Vec::new();
    for value in values {
        string(&mut bytes, value)?;
    }
    array(name, 8, values.len(), &bytes)
}
fn align(bytes: &mut Vec<u8>) -> Result<()> {
    bytes.resize(
        bytes
            .len()
            .checked_add(31)
            .ok_or_else(|| invalid("pair alignment overflow"))?
            & !31,
        0,
    );
    Ok(())
}
fn write(path: &Path, entries: Vec<Entry>, tensors: &[Tensor]) -> Result<u64> {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    bytes.extend_from_slice(&u64::try_from(tensors.len())?.to_le_bytes());
    bytes.extend_from_slice(&u64::try_from(entries.len())?.to_le_bytes());
    let mut keys = std::collections::BTreeSet::new();
    for entry in entries {
        if !keys.insert(entry.name.clone()) {
            return Err(invalid("duplicate pair input metadata key"));
        }
        string(&mut bytes, &entry.name)?;
        bytes.extend_from_slice(&entry.kind.to_le_bytes());
        bytes.extend(entry.data);
    }
    let mut data = Vec::new();
    keys.clear();
    for tensor in tensors {
        if !keys.insert(tensor.name.clone())
            || tensor.dimensions.is_empty()
            || tensor.dimensions.len() > 2
        {
            return Err(invalid("invalid pair input tensor name/rank"));
        }
        let expected = tensor.dimensions.iter().try_fold(4_u64, |n, &d| {
            n.checked_mul(d)
                .ok_or_else(|| invalid("pair tensor size overflow"))
        })?;
        if expected != u64::try_from(tensor.data.len())? {
            return Err(invalid("pair tensor byte extent differs"));
        }
        align(&mut data)?;
        string(&mut bytes, &tensor.name)?;
        bytes.extend_from_slice(&u32::try_from(tensor.dimensions.len())?.to_le_bytes());
        for &n in &tensor.dimensions {
            bytes.extend_from_slice(&n.to_le_bytes());
        }
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        bytes.extend_from_slice(&u64::try_from(data.len())?.to_le_bytes());
        data.extend_from_slice(&tensor.data);
    }
    align(&mut bytes)?;
    bytes.extend(data);
    if bytes.len() > 32 * 1024 * 1024 {
        return Err(invalid("synthetic pair model exceeds32MiB"));
    }
    let length = u64::try_from(bytes.len())?;
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)?;
    file.write_all(&bytes)?;
    Ok(length)
}
fn weights(name: &str, dimensions: Vec<u64>) -> Result<Tensor> {
    let count = dimensions.iter().try_fold(1_u64, |n, &d| {
        n.checked_mul(d)
            .ok_or_else(|| invalid("synthetic weight extent overflow"))
    })?;
    if count > 1024 * 1024 {
        return Err(invalid("synthetic target tensor exceeds1M elements"));
    }
    let bits = (hash_name64(name) ^ 0x04d2).to_le_bytes();
    let mut random = Random::new(u32::from_le_bytes(bits[..4].try_into()?));
    let mut normal = Normal::new(0.0, 0.02);
    let mut bytes = Vec::with_capacity(usize::try_from(count * 4)?);
    for _ in 0..count {
        let v = if name.ends_with("norm.weight") {
            1.0
        } else if name.ends_with("ssm_a") {
            -1.0
        } else if name.ends_with("ssm_dt.bias") {
            0.0
        } else {
            normal.sample(&mut random)
        };
        bytes.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    Ok(Tensor {
        name: name.into(),
        dimensions,
        data: bytes,
    })
}
fn byte_tokens() -> Result<Vec<String>> {
    let mut next = 256_u32;
    (0_u32..256)
        .map(|byte| {
            let code = if (33..=126).contains(&byte)
                || (161..=172).contains(&byte)
                || (174..=255).contains(&byte)
            {
                byte
            } else {
                let value = next;
                next += 1;
                value
            };
            Ok(char::from_u32(code)
                .ok_or_else(|| invalid("invalid GPT2 byte alphabet"))?
                .to_string())
        })
        .collect()
}
#[allow(
    clippy::too_many_lines,
    reason = "explicit synthetic target shapes stay alongside the pinned Qwen35 geometry"
)]
fn target() -> Result<(Vec<Entry>, Vec<Tensor>)> {
    let mut entries = vec![
        text("general.architecture", "qwen35")?,
        text("general.name", "safe-rust-tiny-dspark-target")?,
        u("general.file_type", 0),
        u("qwen35.vocab_size", 256),
        u("qwen35.context_length", 4096),
        u("qwen35.embedding_length", 64),
        u("qwen35.block_count", 17),
        u("qwen35.feed_forward_length", 96),
        u("qwen35.attention.head_count", 2),
        u("qwen35.attention.head_count_kv", 1),
        f("qwen35.attention.layer_norm_rms_epsilon", 1e-6),
        u("qwen35.full_attention_interval", 2),
        u("qwen35.ssm.conv_kernel", 4),
        u("qwen35.ssm.inner_size", 256),
        u("qwen35.ssm.state_size", 128),
        u("qwen35.ssm.time_step_rank", 2),
        u("qwen35.ssm.group_count", 2),
        ua("qwen35.rope.dimension_sections", &[8; 4])?,
        text("tokenizer.ggml.model", "gpt2")?,
        text("tokenizer.ggml.pre", "gpt-2")?,
        strings("tokenizer.ggml.tokens", &byte_tokens()?)?,
        strings("tokenizer.ggml.merges", &[])?,
        u("tokenizer.ggml.bos_token_id", 254),
        u("tokenizer.ggml.eos_token_id", 255),
        b("tokenizer.ggml.add_bos_token", false),
        b("tokenizer.ggml.add_eos_token", false),
    ];
    let types: Vec<_> = (0..256)
        .flat_map(|i| if i >= 254 { 3_i32 } else { 1_i32 }.to_le_bytes())
        .collect();
    entries.push(array("tokenizer.ggml.token_type", 5, 256, &types)?);
    let mut tensors = vec![
        weights("token_embd.weight", vec![64, 256])?,
        weights("output_norm.weight", vec![64])?,
        weights("output.weight", vec![64, 256])?,
    ];
    for layer in 0..17 {
        let prefix = format!("blk.{layer}");
        for suffix in ["attn_norm.weight", "post_attention_norm.weight"] {
            tensors.push(weights(&format!("{prefix}.{suffix}"), vec![64])?);
        }
        if (layer + 1) % 2 == 0 {
            for (suffix, dims) in [
                ("attn_q.weight", vec![64, 128]),
                ("attn_k.weight", vec![64, 32]),
                ("attn_v.weight", vec![64, 32]),
                ("attn_output.weight", vec![64, 64]),
                ("attn_q_norm.weight", vec![32]),
                ("attn_k_norm.weight", vec![32]),
            ] {
                tensors.push(weights(&format!("{prefix}.{suffix}"), dims)?);
            }
        } else {
            for (suffix, dims) in [
                ("attn_qkv.weight", vec![64, 768]),
                ("attn_gate.weight", vec![64, 256]),
                ("ssm_conv1d.weight", vec![4, 768]),
                ("ssm_dt.bias", vec![2]),
                ("ssm_a", vec![2]),
                ("ssm_beta.weight", vec![64, 2]),
                ("ssm_alpha.weight", vec![64, 2]),
                ("ssm_norm.weight", vec![128]),
                ("ssm_out.weight", vec![256, 64]),
            ] {
                tensors.push(weights(&format!("{prefix}.{suffix}"), dims)?);
            }
        }
        for (suffix, dims) in [
            ("ffn_gate.weight", vec![64, 96]),
            ("ffn_down.weight", vec![96, 64]),
            ("ffn_up.weight", vec![64, 96]),
        ] {
            tensors.push(weights(&format!("{prefix}.{suffix}"), dims)?);
        }
    }
    Ok((entries, tensors))
}
fn raw_tensor(layout: &str, name: &str) -> Result<Tensor> {
    let manifest: Value = serde_json::from_str(include_str!("dspark-inputs.json"))?;
    let layouts = manifest["Layouts"]
        .as_array()
        .ok_or_else(|| invalid("pair input layout absent"))?;
    let tensors = layouts
        .iter()
        .find(|v| v["Name"] == layout)
        .and_then(|v| v["Tensors"].as_array())
        .ok_or_else(|| invalid("pair input tensor table absent"))?;
    let tensor = tensors
        .iter()
        .find(|v| v["Name"] == name)
        .ok_or_else(|| invalid(format!("pair frozen tensor absent {name}")))?;
    let number = |key: &str| -> Result<usize> {
        usize::try_from(
            tensor[key]
                .as_u64()
                .ok_or_else(|| invalid("pair input tensor number absent"))?,
        )
        .map_err(Into::into)
    };
    let start: usize = number("Offset")?;
    let length: usize = number("Bytes")?;
    let end = start
        .checked_add(length)
        .ok_or_else(|| invalid("pair frozen input range overflow"))?;
    let data = include_bytes!("dspark-inputs.bin")
        .get(start..end)
        .ok_or_else(|| invalid("pair frozen input outside bytes"))?
        .to_vec();
    let dimensions = tensor["Dimensions"]
        .as_array()
        .ok_or_else(|| invalid("pair tensor dimensions absent"))?
        .iter()
        .map(|v| {
            v.as_u64()
                .ok_or_else(|| invalid("pair tensor extent absent"))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Tensor {
        name: name.into(),
        dimensions,
        data,
    })
}
fn draft(corrected: bool) -> Result<(Vec<Entry>, Vec<Tensor>)> {
    let block = if corrected { 8 } else { 7 };
    let mut entries = vec![
        text("general.architecture", "dspark")?,
        text("general.name", "safe-rust-tiny-dspark-drafter")?,
        u("general.file_type", 0),
        u("dspark.block_count", 2),
        u("dspark.context_length", 4096),
        u("dspark.embedding_length", 64),
        u("dspark.feed_forward_length", 96),
        u("dspark.attention.head_count", 2),
        u("dspark.attention.head_count_kv", 1),
        f("dspark.attention.layer_norm_rms_epsilon", 1e-6),
        f("dspark.rope.freq_base", 10000.0),
        u("dspark.dspark.block_size", block),
        u("dspark.dspark.mask_token_id", 255),
        ua("dspark.dspark.target_layers", &[1, 16])?,
        u("dspark.dspark.markov_rank", 8),
        text("tokenizer.ggml.model", "none")?,
        u("dspark.vocab_size", 256),
        b("dspark.dspark.log_snr_conditioning", corrected),
    ];
    if corrected {
        entries.extend([
            f("dspark.dspark.min_log_snr", -9.0),
            f("dspark.dspark.max_log_snr", 6.0),
            b("dspark.dspark.hidden_correction", true),
            b("dspark.dspark.markov_deploy_exposure", true),
            u("dspark.dspark.correction_size", 64),
        ]);
    }
    let manifest: Value = serde_json::from_str(include_str!("dspark-inputs.json"))?;
    let layout = if corrected {
        "corrected"
    } else {
        "unconditioned"
    };
    let table = manifest["Layouts"]
        .as_array()
        .and_then(|rows| rows.iter().find(|r| r["Name"] == layout))
        .and_then(|r| r["Tensors"].as_array())
        .ok_or_else(|| invalid("pair frozen tensor list absent"))?;
    let mut tensors = Vec::new();
    for tensor in table {
        tensors.push(raw_tensor(
            layout,
            tensor["Name"]
                .as_str()
                .ok_or_else(|| invalid("pair tensor name absent"))?,
        )?);
    }
    if !corrected {
        for name in ["dspark.markov_head_a.weight", "dspark.markov_head_b.weight"] {
            tensors.push(raw_tensor("corrected", name)?);
        }
    }
    Ok((entries, tensors))
}
pub(super) fn create(engine: &Engine, kind: &str) -> Result<Pair> {
    let corrected = match kind {
        "corrected" => true,
        "reference" => false,
        _ => return Err(invalid("unknown synthetic DSpark pair kind")),
    };
    super::dspark_inputs::Corpus::load()?;
    let directory = engine.evidence_path(&format!("dspark-pair-{kind}"))?;
    let target_path = directory.join("tiny-qwen35-target.gguf");
    let draft_path = directory.join("tiny-dspark-drafter.gguf");
    let (target_entries, target_tensors) = target()?;
    let (target_count, draft_block) = (target_tensors.len(), if corrected { 8 } else { 7 });
    let target_bytes = write(&target_path, target_entries, &target_tensors)?;
    let (draft_entries, draft_tensors) = draft(corrected)?;
    let draft_bytes = write(&draft_path, draft_entries, &draft_tensors)?;
    Ok(Pair {
        target: target_path,
        drafter: draft_path,
        receipt: json!({"SyntheticProtocolFixture":true,"QualityMeasurement":false,"Kind":kind,"Seed":1234,"TargetArchitecture":"qwen35","TargetLayers":17,"Embedding":64,"Vocabulary":256,"Tokenizer":"GPT2 byte alphabet256, no merges","CaptureLayers":[1,16],"DrafterLayers":2,"BlockSize":draft_block,"Mask":255,"MarkovRank":8,"GraphCorrected":corrected,"TargetTensors":target_count,"DraftTensors":draft_tensors.len(),"TargetBytes":target_bytes,"DraftBytes":draft_bytes}),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn focused_pair_input_shape_and_extent_guard() -> Result<()> {
        super::super::dspark_inputs::Corpus::load()?;
        let tokens = byte_tokens()?;
        let unique: std::collections::BTreeSet<_> = tokens.iter().collect();
        if tokens.len() != 256
            || unique.len() != 256
            || !tokens.iter().all(|v| v.chars().count() == 1)
        {
            return Err(invalid("byte tokenizer alphabet is not a bijection"));
        }
        let (_, target_tensors) = target()?;
        let (_, corrected) = draft(true)?;
        let (_, reference) = draft(false)?;
        let target_names: std::collections::BTreeSet<_> =
            target_tensors.iter().map(|t| &t.name).collect();
        if target_names.len() != target_tensors.len()
            || corrected.len() != 37
            || reference.len() != 28
        {
            return Err(invalid("pair architecture tensor inventory changed"));
        }
        for tensors in [&target_tensors, &corrected, &reference] {
            for tensor in tensors {
                let elements = tensor.dimensions.iter().try_fold(1_u64, |n, &d| {
                    n.checked_mul(d)
                        .ok_or_else(|| invalid("input extent overflow"))
                })?;
                if tensor.dimensions.contains(&0)
                    || elements * 4 != u64::try_from(tensor.data.len())?
                {
                    return Err(invalid("pair tensor input extent mismatch"));
                }
                if tensor
                    .data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .any(|v| !f32::from_le_bytes([v[0], v[1], v[2], v[3]]).is_finite())
                {
                    return Err(invalid("pair input is nonfinite"));
                }
            }
        }
        let directory = std::env::temp_dir().join(format!(
            "prism-pair-extent-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        std::fs::create_dir(&directory)?;
        let bad = [Tensor {
            name: "bad".into(),
            dimensions: vec![2],
            data: vec![0; 4],
        }];
        let rejected = write(&directory.join("bad.gguf"), Vec::new(), &bad).is_err();
        std::fs::remove_dir(&directory)?;
        if !rejected {
            return Err(invalid("truncated pair tensor was accepted"));
        }
        Ok(())
    }
}
