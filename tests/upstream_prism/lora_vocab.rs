//! Safe tokenizer metadata conversion from original JSON/SP `ModelProto` inputs.
use super::lora_gguf::Metadata;
use super::lora_inputs::Inputs;
use super::{Result, invalid};
use serde_json::Value;
use std::collections::BTreeMap;

fn varint(bytes: &[u8], offset: &mut usize) -> Result<u64> {
    let mut result = 0_u64;
    for shift in (0..70).step_by(7) {
        let byte = *bytes
            .get(*offset)
            .ok_or_else(|| invalid("truncated SP varint"))?;
        *offset += 1;
        if shift == 63 && byte > 1 {
            return Err(invalid("SP varint overflow"));
        }
        result |= u64::from(byte & 127) << shift;
        if byte & 128 == 0 {
            return Ok(result);
        }
    }
    Err(invalid("SP varint overlong"))
}
fn bytes<'a>(input: &'a [u8], offset: &mut usize, length: usize) -> Result<&'a [u8]> {
    let end = offset
        .checked_add(length)
        .filter(|&v| v <= input.len())
        .ok_or_else(|| invalid("SP field out of bounds"))?;
    let value = &input[*offset..end];
    *offset = end;
    Ok(value)
}
fn field<'a>(input: &'a [u8], offset: &mut usize) -> Result<(u64, u64, &'a [u8])> {
    let key = varint(input, offset)?;
    let wire = key & 7;
    let value = match wire {
        0 => {
            let begin = *offset;
            varint(input, offset)?;
            &input[begin..*offset]
        }
        1 => bytes(input, offset, 8)?,
        2 => {
            let length = usize::try_from(varint(input, offset)?)?;
            bytes(input, offset, length)?
        }
        5 => bytes(input, offset, 4)?,
        _ => return Err(invalid("unsupported SP wire type")),
    };
    if key >> 3 == 0 {
        return Err(invalid("SP field zero"));
    }
    Ok((key >> 3, wire, value))
}
pub(super) fn sentencepiece(
    input: &[u8],
    count: usize,
) -> Result<(Vec<String>, Vec<f32>, Vec<i32>)> {
    if input.len() > 8 * 1024 * 1024 || count > 300_000 {
        return Err(invalid("SP input bound exceeded"));
    }
    let mut tokens = (0..count).map(|i| format!("[PAD{i}]")).collect::<Vec<_>>();
    let mut scores = vec![-10_000.0; count];
    let mut kinds = vec![5; count];
    let mut offset = 0;
    let mut id = 0;
    while offset < input.len() {
        let (tag, wire, message) = field(input, &mut offset)?;
        if tag != 1 {
            continue;
        }
        if wire != 2 {
            return Err(invalid("SP piece wire type differs"));
        }
        let (mut text, mut score, mut kind) = (None, 0.0, 1);
        let mut p = 0;
        while p < message.len() {
            let (tag, wire, value) = field(message, &mut p)?;
            match (tag, wire) {
                (1, 2) => text = Some(std::str::from_utf8(value)?.to_owned()),
                (2, 5) => score = f32::from_le_bytes(value.try_into()?),
                (3, 0) => {
                    let mut index = 0;
                    kind = i32::try_from(varint(value, &mut index)?)?;
                }
                _ => {}
            }
        }
        if !(1..=6).contains(&kind) || !score.is_finite() {
            return Err(invalid("invalid SP score/type"));
        }
        if id < count {
            tokens[id] = text.ok_or_else(|| invalid("SP piece text absent"))?;
            scores[id] = score;
            kinds[id] = if kind == 4 { 1 } else { kind };
        }
        id += 1;
    }
    if id == 0 {
        return Err(invalid("SP vocabulary empty"));
    }
    Ok((tokens, scores, kinds))
}
fn looks_special(token: &str) -> bool {
    matches!(token, "<pad>" | "<mask>" | "<2mass>" | "[@BOS@]")
        || token.starts_with("<|") && token.ends_with("|>")
        || token.starts_with("<｜") && token.ends_with("｜>")
        || token.starts_with("<unused") && token.ends_with('>')
}
fn added(token: &Value, phi: bool) -> Result<(String, i32)> {
    let text = token["content"]
        .as_str()
        .ok_or_else(|| invalid("added token text absent"))?;
    let special = token["special"].as_bool().unwrap_or(false) || (!phi && looks_special(text));
    Ok((
        if special || phi {
            text.into()
        } else {
            text.replace('▁', " ")
        },
        if special { 3 } else { 4 },
    ))
}
fn token_text(value: &Value) -> Option<&str> {
    value.as_str().or_else(|| value["content"].as_str())
}
#[allow(
    clippy::too_many_lines,
    reason = "keep the ordered original special-token precedence in one auditable conversion"
)]
fn special(
    metadata: &mut Metadata,
    config: &Value,
    tokenizer: &Value,
    tc: &Value,
    count: usize,
    merges: bool,
) -> Result<()> {
    let mut ids = BTreeMap::<String, u32>::new();
    let mut flags = BTreeMap::<String, bool>::new();
    let mut names = BTreeMap::<String, String>::new();
    let types = ["bos", "eos", "unk", "sep", "pad", "cls", "mask"];
    for kind in types {
        if let Some(s) = token_text(&tc[format!("{kind}_token")]) {
            names.insert(kind.into(), s.into());
        }
    }
    if !names.contains_key("bos")
        && let Some(v) = names.get("cls").cloned()
    {
        names.insert("bos".into(), v);
    }
    if !names.contains_key("eos")
        && let Some(v) = names.get("sep").cloned()
    {
        names.insert("eos".into(), v);
    }
    let post = &tokenizer["post_processor"];
    let processors = post["processors"]
        .as_array()
        .cloned()
        .unwrap_or_else(|| vec![post.clone()]);
    for processor in processors {
        if processor["type"] == "TemplateProcessing" {
            if let Some(single) = processor["single"].as_array().filter(|v| v.len() > 1) {
                if let Some(first) = single[0]["SpecialToken"]["id"].as_str() {
                    if !names.contains_key("bos") {
                        names.insert("bos".into(), first.into());
                    }
                    flags.insert(
                        "bos".into(),
                        names.get("bos").is_some_and(|s| s == first)
                            || names.get("cls").is_some_and(|s| s == first),
                    );
                }
                if let Some(last) = single.last().and_then(|v| v["SpecialToken"]["id"].as_str()) {
                    if let Some(previous) = names
                        .insert("eos".into(), last.into())
                        .filter(|s| s != last)
                    {
                        names.insert("eot".into(), previous);
                    }
                    flags.insert("eos".into(), true);
                }
            }
        } else if !processor.is_null()
            && processor["type"] != "ByteLevel"
            && processor["type"] != "Sequence"
        {
            return Err(invalid("source tokenizer postprocessor not supported"));
        }
    }
    let added = tokenizer["added_tokens"]
        .as_array()
        .ok_or_else(|| invalid("original tokenizer added tokens absent"))?;
    for (kind, text) in &names {
        if let Some(id) = added
            .iter()
            .find(|t| t["content"] == *text)
            .and_then(|v| v["id"].as_u64())
            .filter(|&id| usize::try_from(id).is_ok_and(|id| id < count))
        {
            ids.insert(kind.clone(), u32::try_from(id)?);
        }
    }
    for kind in types {
        if let Some(value) = tc[format!("add_{kind}_token")].as_bool() {
            flags.insert(kind.into(), value);
        }
        if !ids.contains_key(kind)
            && let Some(id) = config[format!("{kind}_token_id")]
                .as_u64()
                .filter(|&id| usize::try_from(id).is_ok_and(|id| id < count))
        {
            ids.insert(kind.into(), u32::try_from(id)?);
        }
    }
    for (kind, id) in ids {
        let key = match kind.as_str() {
            "unk" => "unknown",
            "pad" => "padding",
            "sep" => "seperator",
            "cls" => continue,
            other => other,
        };
        metadata.uint(&format!("tokenizer.ggml.{key}_token_id"), id);
    }
    for (kind, value) in flags {
        metadata.boolean(&format!("tokenizer.ggml.add_{kind}_token"), value);
    }
    if let Some(template) = tc["chat_template"].as_str() {
        metadata.text("tokenizer.chat_template", template)?;
    }
    if merges {
        let values = tokenizer["model"]["merges"]
            .as_array()
            .ok_or_else(|| invalid("BPE merges absent"))?;
        let mut output = Vec::with_capacity(values.len());
        for value in values {
            if let Some(s) = value.as_str() {
                output.push(s.into());
            } else {
                let pair = value
                    .as_array()
                    .filter(|v| v.len() == 2)
                    .ok_or_else(|| invalid("invalid BPE merge pair"))?;
                let a = pair[0]
                    .as_str()
                    .ok_or_else(|| invalid("invalid merge text"))?
                    .replace(' ', "Ġ");
                let b = pair[1]
                    .as_str()
                    .ok_or_else(|| invalid("invalid merge text"))?
                    .replace(' ', "Ġ");
                output.push(format!("{a} {b}"));
            }
        }
        metadata.strings("tokenizer.ggml.merges", &output)?;
    }
    Ok(())
}
#[allow(
    clippy::too_many_lines,
    reason = "three source vocabulary paths share the final production metadata serialization"
)]
pub(super) fn convert(
    inputs: &Inputs,
    architecture: &str,
    config: &Value,
    metadata: &mut Metadata,
) -> Result<()> {
    let base = format!("{architecture}/hidden_size=64/base");
    let tokenizer = inputs.json(&format!("{base}/tokenizer.json"))?;
    let tc = inputs.json(&format!("{base}/tokenizer_config.json"))?;
    let count = usize::try_from(
        config["vocab_size"]
            .as_u64()
            .filter(|&v| v <= 300_000)
            .ok_or_else(|| invalid("invalid vocabulary size"))?,
    )?;
    let (mut tokens, mut scores, mut kinds) = if architecture == "LlamaForCausalLM" {
        let vocabulary = tokenizer["model"]["vocab"]
            .as_object()
            .ok_or_else(|| invalid("BPE vocabulary absent"))?;
        let mut tokens = (0..count).map(|i| format!("[PAD{i}]")).collect::<Vec<_>>();
        let mut kinds = vec![5; count];
        for (text, id) in vocabulary {
            let index =
                usize::try_from(id.as_u64().ok_or_else(|| invalid("invalid BPE token id"))?)?;
            if index >= count {
                return Err(invalid("BPE token outside vocabulary"));
            }
            tokens[index].clone_from(text);
            kinds[index] = 1;
        }
        for token in tokenizer["added_tokens"]
            .as_array()
            .ok_or_else(|| invalid("BPE added tokens absent"))?
        {
            let index = usize::try_from(
                token["id"]
                    .as_u64()
                    .ok_or_else(|| invalid("invalid BPE added id"))?,
            )?;
            if index >= count {
                return Err(invalid("added BPE token outside vocabulary"));
            }
            // The pinned Llama3 fixture's added tokens are all special and
            // encode/decode identically; no tokenizer implementation substitutes SUT.
            if token["special"] != true {
                return Err(invalid("unqualified non-special Llama3 added token"));
            }
            let (text, kind) = added(token, false)?;
            tokens[index] = text;
            kinds[index] = kind;
        }
        metadata.text("tokenizer.ggml.model", "gpt2")?;
        metadata.text("tokenizer.ggml.pre", "llama-bpe")?;
        (tokens, Vec::new(), kinds)
    } else {
        let data = std::fs::read(inputs.path(&format!("{base}/tokenizer.model"))?)?;
        metadata.text("tokenizer.ggml.model", "llama")?;
        metadata.text("tokenizer.ggml.pre", "default")?;
        sentencepiece(&data, count)?
    };
    if architecture != "LlamaForCausalLM" {
        let phi = architecture == "Phi3ForCausalLM";
        if let Some(decoder) = tc["added_tokens_decoder"].as_object() {
            for (id, value) in decoder {
                let id = id.parse::<usize>()?;
                if id >= count {
                    continue;
                }
                let (text, kind) = added(value, phi)?;
                tokens[id] = text;
                scores[id] = -1000.0;
                kinds[id] = kind;
            }
        }
        if phi {
            for token in tokenizer["added_tokens"]
                .as_array()
                .ok_or_else(|| invalid("Phi added tokens absent"))?
            {
                let id = usize::try_from(
                    token["id"]
                        .as_u64()
                        .ok_or_else(|| invalid("Phi token id absent"))?,
                )?;
                if id >= count {
                    return Err(invalid("Phi added token outside vocabulary"));
                }
                let (text, kind) = added(token, true)?;
                tokens[id] = text;
                scores[id] = -1000.0;
                kinds[id] = kind;
            }
        }
        metadata.floats("tokenizer.ggml.scores", &scores)?;
    }
    metadata.strings("tokenizer.ggml.tokens", &tokens)?;
    metadata.integers("tokenizer.ggml.token_type", &kinds)?;
    special(
        metadata,
        config,
        &tokenizer,
        &tc,
        count,
        architecture == "LlamaForCausalLM",
    )?;
    if architecture == "Gemma2ForCausalLM" {
        metadata.boolean("tokenizer.ggml.add_space_prefix", false);
    } else if let Some(value) = tc["add_prefix_space"].as_bool() {
        metadata.boolean("tokenizer.ggml.add_space_prefix", value);
    }
    Ok(())
}
