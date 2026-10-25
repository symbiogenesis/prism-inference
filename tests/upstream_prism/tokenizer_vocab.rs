//! Input-only vocabulary fixture conversion from pinned publisher JSON/SP files.
//! The rules are transcribed from the pinned conversion/base.py, bert.py, gemma.py
//! and gguf-py/gguf/vocab.py. Production token IDs or decoded text are never inputs.
use super::lora_gguf::{self, Metadata};
use super::tokenizer_reference::Reference;
use super::{Result, invalid};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::Path;

const PRE_INPUTS: &str = include_str!("tokenizer-pre-inputs.json");
fn content(value: &Value) -> Option<&str> {
    value
        .as_str()
        .or_else(|| value.get("content").and_then(Value::as_str))
}
fn read(path: &Path) -> Result<Value> {
    if std::fs::metadata(path)?.len() > 32 * 1024 * 1024 {
        return Err(invalid("Tokenizer input JSON exceeds32MiB"));
    }
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}
fn looks_special(token: &str) -> bool {
    matches!(token, "<pad>" | "<mask>" | "<2mass>" | "[@BOS@]")
        || token.starts_with("<|") && token.ends_with("|>")
        || token.starts_with("<｜") && token.ends_with("｜>")
        || token.starts_with("<unused") && token.ends_with('>')
}
fn normalizer(metadata: &mut Metadata, value: &Value, depth: usize) -> Result<()> {
    if depth > 16 {
        return Err(invalid("Tokenizer normalizer nesting exceeds16"));
    }
    match value.get("type").and_then(Value::as_str) {
        Some("Lowercase") => metadata.boolean("tokenizer.ggml.normalizer.lowercase", true),
        Some("StripAccents") => metadata.boolean("tokenizer.ggml.normalizer.strip_accents", true),
        Some("BertNormalizer") => {
            for (json_name, gguf_name) in [
                ("lowercase", "normalizer.lowercase"),
                ("strip_accents", "normalizer.strip_accents"),
            ] {
                if let Some(flag) = value.get(json_name).and_then(Value::as_bool) {
                    metadata.boolean(&format!("tokenizer.ggml.{gguf_name}"), flag);
                }
            }
        }
        Some("Sequence") => {
            for value in value
                .get("normalizers")
                .and_then(Value::as_array)
                .ok_or_else(|| invalid("Normalizer sequence missing"))?
            {
                normalizer(metadata, value, depth + 1)?;
            }
        }
        _ => {}
    }
    Ok(())
}
fn pre(reference: &Reference) -> Result<(String, String)> {
    let inputs: Value = serde_json::from_str(PRE_INPUTS)?;
    let text = inputs["CheckText"]
        .as_str()
        .ok_or_else(|| invalid("Pinned pre-tokenizer check text missing"))?;
    let ids = reference.encode(text, true)?;
    let representation = format!(
        "[{}]",
        ids.iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    );
    let hash = super::lora_inputs::hash_bytes(representation.as_bytes());
    // The original generated function uses ordered if statements; duplicate
    // fingerprints intentionally retain the final assignment.
    let label = inputs["HashPre"]
        .as_array()
        .ok_or_else(|| invalid("Pinned pre-tokenizer fingerprints missing"))?
        .iter()
        .rev()
        .find(|row| row["Sha256"] == hash)
        .and_then(|row| row["Pre"].as_str())
        .ok_or_else(|| {
            invalid(format!(
                "Publisher tokenizer fingerprint {hash} is not recognized by the pinned converter"
            ))
        })?;
    Ok((label.to_owned(), hash))
}
fn merge_list(tokenizer: &Value, directory: &Path) -> Result<Vec<String>> {
    if let Some(merges) = tokenizer["model"]["merges"]
        .as_array()
        .filter(|values| !values.is_empty())
    {
        return merges
            .iter()
            .map(|value| {
                if let Some(text) = value.as_str() {
                    Ok(text.to_owned())
                } else {
                    let pair = value
                        .as_array()
                        .filter(|v| v.len() == 2)
                        .ok_or_else(|| invalid("Tokenizer merge pair malformed"))?;
                    Ok(format!(
                        "{} {}",
                        pair[0]
                            .as_str()
                            .ok_or_else(|| invalid("Tokenizer merge text missing"))?
                            .replace(' ', "Ġ"),
                        pair[1]
                            .as_str()
                            .ok_or_else(|| invalid("Tokenizer merge text missing"))?
                            .replace(' ', "Ġ")
                    ))
                }
            })
            .collect();
    }
    let path = directory.join("merges.txt");
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let input = std::fs::read_to_string(path)?;
    Ok(input
        .lines()
        .enumerate()
        .filter(|(index, line)| *index != 0 || !line.trim().starts_with('#'))
        .filter_map(|(_, line)| {
            let pieces = line.split_whitespace().collect::<Vec<_>>();
            (pieces.len() == 2).then(|| format!("{} {}", pieces[0], pieces[1]))
        })
        .collect())
}
#[allow(
    clippy::too_many_lines,
    reason = "Ordered source SpecialVocab metadata precedence is kept together"
)]
fn special(
    metadata: &mut Metadata,
    config: &Value,
    tokenizer: &Value,
    tc: &Value,
    count: usize,
    directory: &Path,
    load_merges: bool,
) -> Result<()> {
    normalizer(metadata, &tokenizer["normalizer"], 0)?;
    let mut names = BTreeMap::<String, String>::new();
    let mut flags = BTreeMap::<String, bool>::new();
    let mut kinds = vec!["bos", "eos", "unk", "sep", "pad", "cls", "mask"];
    for kind in &kinds {
        if let Some(text) = content(&tc[format!("{kind}_token")]) {
            names.insert((*kind).to_owned(), text.to_owned());
        }
    }
    if !names.contains_key("bos")
        && let Some(text) = names.get("cls").cloned()
    {
        names.insert("bos".to_owned(), text);
    }
    if !names.contains_key("eos")
        && let Some(text) = names.get("sep").cloned()
    {
        names.insert("eos".to_owned(), text);
    }
    let post = &tokenizer["post_processor"];
    let processors = post["processors"]
        .as_array()
        .cloned()
        .unwrap_or_else(|| vec![post.clone()]);
    for processor in processors {
        if processor["type"] == "RobertaProcessing" {
            for kind in ["bos", "eos", "sep"] {
                flags.insert(kind.to_owned(), true);
            }
            for kind in ["cls", "sep"] {
                if !names.contains_key(kind)
                    && let Some(text) = processor[kind][0].as_str()
                {
                    names.insert(kind.to_owned(), text.to_owned());
                }
            }
        } else if processor["type"] == "TemplateProcessing" {
            let single = processor["single"]
                .as_array()
                .ok_or_else(|| invalid("Tokenizer single template missing"))?;
            let mut first = None;
            let mut last = None;
            if single.len() > 1 {
                if let Some(text) = single[0]["SpecialToken"]["id"].as_str() {
                    first = Some(text);
                    if names.get("bos").map(String::as_str) != Some(text)
                        && names.get("cls").map(String::as_str) != Some(text)
                    {
                        names
                            .entry("bos".to_owned())
                            .or_insert_with(|| text.to_owned());
                        names
                            .entry("cls".to_owned())
                            .or_insert_with(|| text.to_owned());
                    }
                    flags.insert(
                        "bos".to_owned(),
                        names.get("bos").map(String::as_str) == Some(text)
                            || names.get("cls").map(String::as_str) == Some(text),
                    );
                }
                if let Some(text) = single.last().and_then(|v| v["SpecialToken"]["id"].as_str()) {
                    last = Some(text);
                    if names.get("eos").map(String::as_str) != Some(text) {
                        if !kinds.contains(&"eot") {
                            kinds.push("eot");
                            if let Some(previous) = names.get("eos").cloned() {
                                names.insert("eot".to_owned(), previous);
                            }
                        } else if !kinds.contains(&"eom") {
                            kinds.push("eom");
                            if let Some(previous) = names.get("eos").cloned() {
                                names.insert("eom".to_owned(), previous);
                            }
                        }
                        names.insert("eos".to_owned(), text.to_owned());
                    }
                    flags.insert("eos".to_owned(), true);
                }
            }
            if let Some(pair) = processor["pair"].as_array().filter(|v| !v.is_empty()) {
                let begin =
                    usize::from(first.is_some() && pair[0]["SpecialToken"]["id"].as_str() == first);
                let end = pair.len()
                    - usize::from(
                        last.is_some()
                            && pair.last().and_then(|v| v["SpecialToken"]["id"].as_str()) == last,
                    );
                let pair = pair
                    .get(begin..end)
                    .ok_or_else(|| invalid("Tokenizer pair template bounds invalid"))?;
                if pair.len() >= 2
                    && pair[0]["Sequence"]["id"] == "A"
                    && pair.last().is_some_and(|v| v["Sequence"]["id"] == "B")
                {
                    let middle = &pair[1..pair.len() - 1];
                    if !middle.is_empty() {
                        let matches = |value: &Value| {
                            value["SpecialToken"]["id"].as_str().is_some_and(|text| {
                                names.get("sep").map(String::as_str) == Some(text)
                                    || names.get("eos").map(String::as_str) == Some(text)
                            })
                        };
                        let add_sep = (matches(&middle[0]) && last.is_none())
                            || (middle.len() == 2 && matches(&middle[1]));
                        flags.insert("sep".to_owned(), add_sep);
                        if add_sep
                            && !names.contains_key("sep")
                            && let Some(text) = names.get("eos").cloned()
                        {
                            names.insert("sep".to_owned(), text);
                        }
                    }
                }
            }
        }
    }
    let added = tokenizer["added_tokens"]
        .as_array()
        .ok_or_else(|| invalid("Tokenizer added descriptors missing"))?;
    for kind in kinds {
        if let Some(flag) = tc[format!("add_{kind}_token")].as_bool() {
            flags.insert(kind.to_owned(), flag);
        }
        let id = names
            .get(kind)
            .and_then(|text| added.iter().find(|token| token["content"] == *text))
            .and_then(|token| token["id"].as_u64())
            .or_else(|| config[format!("{kind}_token_id")].as_u64());
        if let Some(id) = id.filter(|id| usize::try_from(*id).is_ok_and(|id| id < count)) {
            let key = match kind {
                "unk" => "unknown",
                "pad" => "padding",
                "sep" => "seperator",
                "cls" => continue,
                other => other,
            };
            metadata.uint(
                &format!("tokenizer.ggml.{key}_token_id"),
                u32::try_from(id)?,
            );
        }
    }
    for (kind, flag) in flags {
        metadata.boolean(&format!("tokenizer.ggml.add_{kind}_token"), flag);
    }
    if let Some(template) = tc["chat_template"].as_str() {
        metadata.text("tokenizer.chat_template", template)?;
    }
    if load_merges {
        let values = merge_list(tokenizer, directory)?;
        if !values.is_empty() {
            metadata.strings("tokenizer.ggml.merges", &values)?;
        }
    }
    Ok(())
}
fn added_descriptions(
    tokenizer: &Value,
    tc: &Value,
    directory: &Path,
) -> Result<BTreeMap<usize, Value>> {
    let mut added = BTreeMap::new();
    for token in tokenizer["added_tokens"]
        .as_array()
        .ok_or_else(|| invalid("Tokenizer added descriptors absent"))?
    {
        let id = usize::try_from(
            token["id"]
                .as_u64()
                .ok_or_else(|| invalid("Added token ID absent"))?,
        )?;
        added.insert(id, token.clone());
    }
    if let Some(values) = tc["added_tokens_decoder"].as_object() {
        for (id, value) in values {
            added.insert(id.parse()?, value.clone());
        }
    }
    let path = directory.join("added_tokens.json");
    if path.is_file() {
        for (text, id) in read(&path)?
            .as_object()
            .ok_or_else(|| invalid("Legacy added-token map malformed"))?
        {
            let id = usize::try_from(
                id.as_u64()
                    .ok_or_else(|| invalid("Legacy token ID invalid"))?,
            )?;
            added.entry(id).or_insert_with(
                || json!({"content":text,"special":false,"normalized":true,"SourceLegacy":true}),
            );
        }
    }
    Ok(added)
}
#[allow(
    clippy::too_many_lines,
    reason = "The three original vocabulary conversion branches share metadata serialization"
)]
pub(super) fn write(name: &str, directory: &Path, output: &Path) -> Result<Value> {
    let config_path = directory.join("config.json");
    let config = if config_path.is_file() {
        read(&config_path)?
    } else {
        json!({})
    };
    let tokenizer = read(&directory.join("tokenizer.json"))?;
    let tc = read(&directory.join("tokenizer_config.json"))?;
    let reference = Reference::open(directory)?;
    let mut metadata = Metadata::new();
    let architecture = match name {
        "gemma" => "gemma",
        "gemma-2" => "gemma2",
        "jina-v2-en" | "jina-v2-es" | "jina-v2-de" | "jina-v2-code" => "jina-bert-v2",
        "phi-2" => "phi2",
        "stablelm2" => "stablelm",
        "olmo" => "olmo",
        "smaug-bpe" | "viking" => "llama",
        "poro-chat" => "bloom",
        "jais" => "jais",
        _ => {
            return Err(invalid(
                "Vocabulary fixture is not one of the original missing13",
            ));
        }
    };
    metadata.text("general.architecture", architecture)?;
    metadata.text(
        "general.name",
        &format!("Original HF tokenizer input {name}"),
    )?;
    let sentencepiece = matches!(name, "gemma" | "gemma-2");
    let no_added_json = json!({"added_tokens":[]});
    let added = added_descriptions(
        if sentencepiece {
            &no_added_json
        } else {
            &tokenizer
        },
        &tc,
        directory,
    )?;
    let largest_model = if let Some(values) = tokenizer["model"]["vocab"].as_object() {
        values.values().filter_map(Value::as_u64).max().unwrap_or(0)
    } else {
        u64::try_from(
            tokenizer["model"]["vocab"]
                .as_array()
                .map_or(0, Vec::len)
                .saturating_sub(1),
        )?
    };
    let largest_added = added.keys().next_back().copied().unwrap_or(0);
    let inferred = usize::try_from(largest_model)?.max(largest_added) + 1;
    let count = usize::try_from(
        config["vocab_size"]
            .as_u64()
            .unwrap_or(u64::try_from(inferred)?),
    )?;
    if count == 0 || count > 300_000 || (!sentencepiece && inferred > count) {
        return Err(invalid("Original vocabulary extent outside bounds"));
    }
    let wordpiece = tokenizer["model"]["type"] == "WordPiece";
    let (mut tokens, mut scores, mut types) = if sentencepiece {
        super::lora_vocab::sentencepiece(&std::fs::read(directory.join("tokenizer.model"))?, count)?
    } else {
        let mut tokens = (0..count).map(|i| format!("[PAD{i}]")).collect::<Vec<_>>();
        let mut types = vec![5; count];
        for (text, id) in tokenizer["model"]["vocab"]
            .as_object()
            .ok_or_else(|| invalid("Original base vocabulary is not a map"))?
        {
            let id = usize::try_from(
                id.as_u64()
                    .ok_or_else(|| invalid("Original vocabulary ID invalid"))?,
            )?;
            if id >= count {
                return Err(invalid("Base vocabulary ID exceeds configured count"));
            }
            tokens[id].clone_from(text);
            types[id] = 1;
        }
        (tokens, Vec::new(), types)
    };
    for (id, description) in added {
        if id >= count {
            continue;
        }
        let mut text = content(&description)
            .ok_or_else(|| invalid("Added-token content absent"))?
            .to_owned();
        if !sentencepiece && !description["normalized"].as_bool().unwrap_or(true) {
            text = reference.decode(&reference.encode(&text, false)?, false)?;
        }
        let special = description["special"].as_bool().unwrap_or(false)
            || (!(sentencepiece && description["SourceLegacy"] == true) && looks_special(&text));
        if !special {
            text = text.replace('▁', " ");
        }
        tokens[id] = text;
        types[id] = if special { 3 } else { 4 };
        if sentencepiece {
            scores[id] = -1000.0;
        }
    }
    let fingerprint = if sentencepiece {
        metadata.text("tokenizer.ggml.model", "llama")?;
        metadata.text("tokenizer.ggml.pre", "default")?;
        metadata.floats("tokenizer.ggml.scores", &scores)?;
        None
    } else {
        let (label, hash) = pre(&reference)?;
        metadata.text(
            "tokenizer.ggml.model",
            if wordpiece { "bert" } else { "gpt2" },
        )?;
        metadata.text("tokenizer.ggml.pre", &label)?;
        if wordpiece {
            for (token, kind) in tokens.iter_mut().zip(&types) {
                if *kind != 3 {
                    *token = token
                        .strip_prefix("##")
                        .map_or_else(|| format!("▁{token}"), str::to_owned);
                }
            }
        }
        Some(json!({"Pre":label,"CheckTokenIdsSha256":hash}))
    };
    if name.starts_with("jina-v2-") {
        metadata.uint(
            "tokenizer.ggml.token_type_count",
            if wordpiece {
                u32::try_from(config["type_vocab_size"].as_u64().unwrap_or(1))?
            } else {
                2
            },
        );
    }
    metadata.strings("tokenizer.ggml.tokens", &tokens)?;
    metadata.integers("tokenizer.ggml.token_type", &types)?;
    special(
        &mut metadata,
        &config,
        &tokenizer,
        &tc,
        count,
        directory,
        !sentencepiece && !wordpiece,
    )?;
    if name == "phi-2" {
        metadata.boolean("tokenizer.ggml.add_bos_token", false);
    }
    // The pinned writer has no deprecated prefix/suffix/middle/fsep handlers;
    // source SpecialVocab logs those and only emits its eot override.
    if name == "gemma" {
        metadata.uint("tokenizer.ggml.eot_token_id", 107);
    }
    if name == "gemma-2" {
        metadata.boolean("tokenizer.ggml.add_space_prefix", false);
    }
    let length = lora_gguf::write_vocabulary(output, &metadata)?;
    Ok(
        json!({"Name":name,"GeneratedVocabFixture":true,"SourceCorrection":"Missing pinned source GGUF is generated in safe Rust from immutable original publisher tokenizer inputs using pinned converter input rules","VocabularySize":count,"Architecture":architecture,"SentencePiece":sentencepiece,"WordPiece":wordpiece,"Fingerprint":fingerprint,"Length":length,"Sha256":super::lora_inputs::digest(output)?.1,"NoWeights":true,"NoProductionObservations":true}),
    )
}
