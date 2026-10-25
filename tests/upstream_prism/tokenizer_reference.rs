//! Independent Hugging Face 4.57.6 wrapper semantics over safe Rust tokenizers.
//!
//! Inputs are immutable publisher tokenizer files. No production token observations
//! are used by this reference. `BaiChuan`'s original custom slow class is transcribed
//! separately because its publisher does not provide a fast tokenizer JSON.
use super::tokenizer_generators::Vocabulary;
use super::{Result, invalid};
use sentencepiece_rs::SentencePieceProcessor;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use tokenizers::{AddedToken, Tokenizer};

const SPECIAL_KEYS: [&str; 7] = [
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
];
enum Backend {
    Fast(Box<Tokenizer>),
    BaiChuan(Box<SentencePieceProcessor>),
}
#[allow(
    clippy::struct_excessive_bools,
    reason = "Independent upstream tokenizer configuration flags"
)]
pub(super) struct Reference {
    backend: Backend,
    specials: Vec<String>,
    special_ids: BTreeSet<u32>,
    valid_ids: Vec<u32>,
    added_ids: Vec<u32>,
    bos: Option<String>,
    eos: Option<String>,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    add_bos: bool,
    add_eos: bool,
    manual_boundaries: bool,
    cleanup: bool,
    boundary_properties: (Option<bool>, Option<bool>),
    pub(super) class: String,
    pub(super) source_correction: Option<&'static str>,
}
fn read_json(path: &Path) -> Result<Value> {
    let bytes = std::fs::read(path)?;
    if bytes.len() > 32 * 1024 * 1024 {
        return Err(invalid("HF tokenizer JSON exceeds 32 MiB"));
    }
    Ok(serde_json::from_slice(&bytes)?)
}
fn token_content(value: &Value) -> Option<&str> {
    value
        .as_str()
        .or_else(|| value.get("content").and_then(Value::as_str))
}
fn defaults(class: &str) -> Vec<(&'static str, &'static str)> {
    if class.starts_with("Bert") {
        vec![
            ("unk_token", "[UNK]"),
            ("sep_token", "[SEP]"),
            ("pad_token", "[PAD]"),
            ("cls_token", "[CLS]"),
            ("mask_token", "[MASK]"),
        ]
    } else if class.starts_with("Roberta") {
        vec![
            ("bos_token", "<s>"),
            ("eos_token", "</s>"),
            ("unk_token", "<unk>"),
            ("sep_token", "</s>"),
            ("pad_token", "<pad>"),
            ("cls_token", "<s>"),
            ("mask_token", "<mask>"),
        ]
    } else if class.starts_with("Gemma") {
        vec![
            ("bos_token", "<bos>"),
            ("eos_token", "<eos>"),
            ("unk_token", "<unk>"),
            ("pad_token", "<pad>"),
        ]
    } else if class.starts_with("Llama") || class == "BaiChuanTokenizer" {
        vec![
            ("bos_token", "<s>"),
            ("eos_token", "</s>"),
            ("unk_token", "<unk>"),
        ]
    } else if class.starts_with("Bloom") {
        vec![
            ("bos_token", "<s>"),
            ("eos_token", "</s>"),
            ("unk_token", "<unk>"),
            ("pad_token", "<pad>"),
        ]
    } else if class.starts_with("GPT2")
        || class.starts_with("GPTNeoX")
        || class.starts_with("CodeGen")
    {
        vec![
            ("bos_token", "<|endoftext|>"),
            ("eos_token", "<|endoftext|>"),
            ("unk_token", "<|endoftext|>"),
        ]
    } else {
        Vec::new()
    }
}
fn added_token(value: &Value, special: bool) -> Result<AddedToken> {
    let content = token_content(value).ok_or_else(|| invalid("HF added token has no content"))?;
    let mut token = AddedToken::from(content, special);
    for (key, field) in [
        ("single_word", &mut token.single_word),
        ("lstrip", &mut token.lstrip),
        ("rstrip", &mut token.rstrip),
        ("normalized", &mut token.normalized),
    ] {
        if let Some(value) = value.get(key) {
            *field = value
                .as_bool()
                .ok_or_else(|| invalid("HF added-token flag is not boolean"))?;
        }
    }
    token.special = value
        .get("special")
        .and_then(Value::as_bool)
        .unwrap_or(special)
        || special;
    Ok(token)
}
fn add_descriptions(
    tokenizer: &mut Tokenizer,
    descriptions: &BTreeMap<u32, Value>,
    specials: &[String],
) -> Result<()> {
    if descriptions.is_empty() {
        return Ok(());
    }
    let mut contents = BTreeSet::new();
    let tokens = descriptions
        .values()
        .map(|value| {
            let content =
                token_content(value).ok_or_else(|| invalid("HF added-token content missing"))?;
            if content.is_empty() || !contents.insert(content) {
                return Err(invalid("HF added-token overlay needs ordered updates"));
            }
            added_token(value, specials.iter().any(|token| token == content))
        })
        .collect::<Result<Vec<_>>>();
    if let Ok(tokens) = tokens {
        // Each add_tokens call clones the existing descriptors and rebuilds two
        // Aho-Corasick tries, even when a publisher descriptor is unchanged.
        // Unique contents have independent ID/flag overlays, so construct the
        // identical ordered token tables and matching tries only once.
        tokenizer.add_tokens(&tokens);
        for ((id, _), token) in descriptions.iter().zip(tokens) {
            if tokenizer.token_to_id(&token.content) != Some(*id) {
                return Err(invalid(
                    "HF added-token ID differs from publisher descriptor",
                ));
            }
        }
    } else {
        // Duplicate/empty contents and malformed flags retain the original
        // sequential overlay behavior, including which validation fails first.
        for (id, value) in descriptions {
            let content =
                token_content(value).ok_or_else(|| invalid("HF added-token content missing"))?;
            tokenizer.add_tokens(&[added_token(
                value,
                specials.iter().any(|token| token == content),
            )?]);
            if tokenizer.token_to_id(content) != Some(*id) {
                return Err(invalid(
                    "HF added-token ID differs from publisher descriptor",
                ));
            }
        }
    }
    Ok(())
}
impl Reference {
    #[allow(
        clippy::too_many_lines,
        clippy::similar_names,
        clippy::collapsible_if,
        reason = "The upstream wrapper initialization has ordered configuration overlays"
    )]
    pub(super) fn open(directory: &Path) -> Result<Self> {
        let config = read_json(&directory.join("tokenizer_config.json"))?;
        let class = config
            .get("tokenizer_class")
            .and_then(Value::as_str)
            .unwrap_or("PreTrainedTokenizerFast")
            .to_owned();
        let mut named: BTreeMap<String, Value> = defaults(&class)
            .into_iter()
            .map(|(key, value)| (key.to_owned(), Value::from(value)))
            .collect();
        let special_map = directory.join("special_tokens_map.json");
        if special_map.is_file() {
            for (key, value) in read_json(&special_map)?
                .as_object()
                .ok_or_else(|| invalid("HF special map is not an object"))?
            {
                named.insert(key.clone(), value.clone());
            }
        }
        for key in SPECIAL_KEYS
            .into_iter()
            .chain(["additional_special_tokens"])
        {
            if let Some(value) = config.get(key) {
                named.insert(key.to_owned(), value.clone());
            }
        }
        let mut specials = Vec::new();
        for key in SPECIAL_KEYS {
            if let Some(value) = named.get(key).and_then(token_content) {
                if !specials.iter().any(|token| token == value) {
                    specials.push(value.to_owned());
                }
            }
        }
        if let Some(values) = named
            .get("additional_special_tokens")
            .and_then(Value::as_array)
        {
            for value in values {
                let text = token_content(value)
                    .ok_or_else(|| invalid("HF additional special token invalid"))?;
                if !specials.iter().any(|token| token == text) {
                    specials.push(text.to_owned());
                }
            }
        }
        let bos = named
            .get("bos_token")
            .and_then(token_content)
            .map(str::to_owned);
        let eos = named
            .get("eos_token")
            .and_then(token_content)
            .map(str::to_owned);
        let manual_boundaries = class.starts_with("Llama")
            || class.starts_with("Gemma")
            || class == "BaiChuanTokenizer";
        let add_bos = config
            .get("add_bos_token")
            .and_then(Value::as_bool)
            .unwrap_or(manual_boundaries);
        let add_eos = config
            .get("add_eos_token")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let (backend, added_ids) = if class == "BaiChuanTokenizer" {
            let processor = SentencePieceProcessor::open(directory.join("tokenizer.model"))?;
            let mut ids = specials
                .iter()
                .map(|text| u32::try_from(processor.model().piece_to_id(text)).map_err(Into::into))
                .collect::<Result<Vec<_>>>()?;
            ids.sort_unstable();
            (Backend::BaiChuan(Box::new(processor)), ids)
        } else {
            let mut json = read_json(&directory.join("tokenizer.json"))?;
            // PreTrainedTokenizerFast only patches a top-level pre-tokenizer's flag.
            if let (Some(prefix), Some(pre)) = (
                config.get("add_prefix_space").and_then(Value::as_bool),
                json.get_mut("pre_tokenizer").and_then(Value::as_object_mut),
            ) {
                if pre.contains_key("add_prefix_space") {
                    pre.insert("add_prefix_space".to_owned(), Value::Bool(prefix));
                }
            }
            let mut tokenizer = Tokenizer::from_bytes(serde_json::to_vec(&json)?)?;
            tokenizer.with_padding(None);
            tokenizer.with_truncation(None)?;
            let mut descriptions = BTreeMap::new();
            if let Some(values) = config
                .get("added_tokens_decoder")
                .and_then(Value::as_object)
            {
                for (id, value) in values {
                    descriptions.insert(id.parse::<u32>()?, value.clone());
                }
            }
            let legacy = directory.join("added_tokens.json");
            if legacy.is_file() {
                for (content, id) in read_json(&legacy)?
                    .as_object()
                    .ok_or_else(|| invalid("HF legacy added tokens invalid"))?
                {
                    let id = u32::try_from(
                        id.as_u64()
                            .ok_or_else(|| invalid("HF added-token ID invalid"))?,
                    )?;
                    descriptions
                        .entry(id)
                        .or_insert_with(|| Value::from(content.clone()));
                }
            }
            add_descriptions(&mut tokenizer, &descriptions, &specials)?;
            let encoder = tokenizer
                .get_added_tokens_decoder()
                .into_values()
                .map(|token| token.content)
                .collect::<BTreeSet<_>>();
            let missing = specials
                .iter()
                .filter(|token| !encoder.contains(*token))
                .map(|token| AddedToken::from(token.as_str(), true))
                .collect::<Vec<_>>();
            if !missing.is_empty() {
                tokenizer.add_special_tokens(&missing);
            }
            let mut ids = tokenizer
                .get_added_tokens_decoder()
                .into_keys()
                .collect::<Vec<_>>();
            ids.sort_unstable();
            (Backend::Fast(Box::new(tokenizer)), ids)
        };
        let mut valid_ids = match &backend {
            // Query both maps: a supplied added spelling can hide a model
            // spelling in get_vocab(true), while its original ID still decodes.
            Backend::Fast(tokenizer) => tokenizer
                .get_vocab(false)
                .into_values()
                .chain(tokenizer.get_added_tokens_decoder().into_keys())
                .collect::<Vec<_>>(),
            Backend::BaiChuan(_) => Vec::new(),
        };
        valid_ids.sort_unstable();
        valid_ids.dedup();
        let mut reference = Self {
            backend, specials, special_ids: BTreeSet::new(), valid_ids, added_ids, bos, eos,
            bos_id: None, eos_id: None, add_bos, add_eos, manual_boundaries,
            boundary_properties: (config.get("add_bos_token").and_then(Value::as_bool).or_else(|| manual_boundaries.then_some(add_bos)), config.get("add_eos_token").and_then(Value::as_bool).or_else(|| manual_boundaries.then_some(add_eos))),
            cleanup: config.get("clean_up_tokenization_spaces").and_then(Value::as_bool).unwrap_or(false),
            source_correction: (class == "BaiChuanTokenizer").then_some("Original AutoTokenizer needs publisher remote Python; its slow SentencePiece wrapper is transcribed in safe Rust"), class,
        };
        reference.bos_id = reference
            .bos
            .as_deref()
            .and_then(|text| reference.token_id(text));
        reference.eos_id = reference
            .eos
            .as_deref()
            .and_then(|text| reference.token_id(text));
        reference.special_ids = reference
            .specials
            .iter()
            .filter_map(|text| reference.token_id(text))
            .collect();
        if let Backend::Fast(tokenizer) = &reference.backend {
            reference.special_ids = tokenizer
                .get_added_tokens_decoder()
                .into_iter()
                .filter_map(|(id, token)| token.special.then_some(id))
                .collect();
        }
        if (add_bos && reference.bos_id.is_none()) || (add_eos && reference.eos_id.is_none()) {
            return Err(invalid("HF configured BOS/EOS token has no vocabulary ID"));
        }
        Ok(reference)
    }
    pub(super) fn token_id(&self, text: &str) -> Option<u32> {
        match &self.backend {
            Backend::Fast(tokenizer) => tokenizer.token_to_id(text),
            Backend::BaiChuan(processor) => processor
                .model()
                .pieces()
                .iter()
                .position(|piece| piece.piece == text)
                .and_then(|id| u32::try_from(id).ok()),
        }
    }
    pub(super) fn encode(&self, text: &str, special: bool) -> Result<Vec<u32>> {
        let mut ids = match &self.backend {
            Backend::Fast(tokenizer) => tokenizer
                .encode_fast(text, special && !self.manual_boundaries)?
                .get_ids()
                .to_vec(),
            Backend::BaiChuan(processor) => {
                let mut ids = Vec::new();
                let mut rest = text;
                while !rest.is_empty() {
                    let found = self
                        .specials
                        .iter()
                        .filter_map(|token| rest.find(token).map(|offset| (offset, token)))
                        .min_by(|(a, x), (b, y)| a.cmp(b).then_with(|| y.len().cmp(&x.len())));
                    let Some((offset, token)) = found else {
                        ids.extend(
                            processor
                                .encode_to_ids(rest)?
                                .into_iter()
                                .map(u32::try_from)
                                .collect::<std::result::Result<Vec<_>, _>>()?,
                        );
                        break;
                    };
                    ids.extend(
                        processor
                            .encode_to_ids(&rest[..offset])?
                            .into_iter()
                            .map(u32::try_from)
                            .collect::<std::result::Result<Vec<_>, _>>()?,
                    );
                    ids.push(
                        self.token_id(token)
                            .ok_or_else(|| invalid("BaiChuan special token has no ID"))?,
                    );
                    rest = &rest[offset + token.len()..];
                }
                ids
            }
        };
        self.add_boundaries(&mut ids, special)?;
        Ok(ids)
    }
    fn add_boundaries(&self, ids: &mut Vec<u32>, special: bool) -> Result<()> {
        if special && self.manual_boundaries {
            if self.add_bos {
                ids.insert(0, self.bos_id.ok_or_else(|| invalid("HF BOS ID missing"))?);
            }
            if self.add_eos {
                ids.push(self.eos_id.ok_or_else(|| invalid("HF EOS ID missing"))?);
            }
        }
        Ok(())
    }
    pub(super) fn encode_batch(&self, texts: &[String], special: bool) -> Vec<Result<Vec<u32>>> {
        if let Backend::Fast(tokenizer) = &self.backend
            && let Ok(encodings) = tokenizer.encode_batch_fast(
                texts.iter().map(String::as_str).collect(),
                special && !self.manual_boundaries,
            )
        {
            return encodings
                .into_iter()
                .map(|encoding| {
                    let mut ids = encoding.get_ids().to_vec();
                    self.add_boundaries(&mut ids, special)?;
                    Ok(ids)
                })
                .collect();
        }
        // Batch APIs return only the first failure. Recover individual errors so
        // an unconsumed prefetched input cannot change the source stop prefix.
        texts
            .iter()
            .map(|text| self.encode(text, special))
            .collect()
    }
    pub(super) fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        let text = match &self.backend {
            Backend::Fast(tokenizer) => {
                // Python's release backend returns an empty string here; an explicit
                // guard also avoids its BPEDecoder's empty-input usize subtraction.
                if ids.iter().all(|id| {
                    self.valid_ids.binary_search(id).is_err()
                        || (skip_special && self.special_ids.contains(id))
                }) {
                    String::new()
                } else {
                    tokenizer.decode(ids, skip_special)?
                }
            }
            Backend::BaiChuan(processor) => {
                let mut text = String::new();
                let mut current = Vec::<String>::new();
                let mut previous_special = false;
                let mut index = 0;
                for id in ids {
                    if skip_special && self.special_ids.contains(id) {
                        continue;
                    }
                    let piece = processor.model().id_to_piece(usize::try_from(*id)?)?;
                    if self.special_ids.contains(id) {
                        if !previous_special && index != 0 {
                            text.push(' ');
                        }
                        text.push_str(&processor.decode(&current)?);
                        text.push_str(piece);
                        current.clear();
                        previous_special = true;
                    } else {
                        current.push(piece.to_owned());
                        previous_special = false;
                    }
                    index += 1;
                }
                text.push_str(&processor.decode(&current)?);
                text
            }
        };
        Ok(self.clean_text(text))
    }
    fn clean_text(&self, mut text: String) -> String {
        // All upstream replacements start with ASCII space. Most single-token
        // and Unicode observations need no cleanup; keep their existing String.
        if self.cleanup && text.contains(' ') {
            for (from, to) in [
                (" .", "."),
                (" ?", "?"),
                (" !", "!"),
                (" ,", ","),
                (" ' ", "'"),
                (" n't", "n't"),
                (" 'm", "'m"),
                (" 's", "'s"),
                (" 've", "'ve"),
                (" 're", "'re"),
            ] {
                if text.contains(from) {
                    text = text.replace(from, to);
                }
            }
        }
        text
    }
    pub(super) fn decode_batch(&self, ids: &[&[u32]], skip_special: bool) -> Vec<Result<String>> {
        if let Backend::Fast(tokenizer) = &self.backend {
            // BPEDecoder cannot process an empty surviving token sequence in
            // debug builds. Preserve the scalar wrapper's empty-result guard.
            let active = ids
                .iter()
                .map(|ids| {
                    ids.iter().any(|id| {
                        self.valid_ids.binary_search(id).is_ok()
                            && !(skip_special && self.special_ids.contains(id))
                    })
                })
                .collect::<Vec<_>>();
            let nonempty = ids
                .iter()
                .zip(&active)
                .filter_map(|(ids, active)| active.then_some(*ids))
                .collect::<Vec<_>>();
            if let Ok(decoded) = tokenizer.decode_batch(&nonempty, skip_special) {
                let mut decoded = decoded.into_iter();
                return active
                    .into_iter()
                    .map(|active| {
                        if active {
                            decoded
                                .next()
                                .ok_or_else(|| invalid("HF batch decode extent differs"))
                                .map(|text| self.clean_text(text))
                        } else {
                            Ok(String::new())
                        }
                    })
                    .collect();
            }
        }
        ids.iter()
            .map(|ids| self.decode(ids, skip_special))
            .collect()
    }
    pub(super) fn vocabulary(&self) -> Result<Vocabulary> {
        // Groundtruth checks this only for the random harness, never the file script.
        let probe = self.encode("a", true)?;
        if !(1..=3).contains(&probe.len()) {
            return Err(invalid("Original HF encode(a) length predicate failed"));
        }
        let mut ids = match &self.backend {
            Backend::Fast(tokenizer) => tokenizer.get_vocab(true).into_values().collect::<Vec<_>>(),
            Backend::BaiChuan(processor) => (0..processor.model().pieces().len())
                .map(u32::try_from)
                .collect::<std::result::Result<Vec<_>, _>>()?,
        };
        ids.sort_unstable();
        let mut words = Vec::with_capacity(ids.len());
        for batch in ids.chunks(1024) {
            let batch = batch.iter().map(std::slice::from_ref).collect::<Vec<_>>();
            words.extend(
                self.decode_batch(&batch, true)
                    .into_iter()
                    .collect::<Result<Vec<_>>>()?,
            );
        }
        words.sort();
        let added = self
            .added_ids
            .iter()
            .map(|id| self.decode(&[*id], false))
            .collect::<Result<Vec<_>>>()?;
        Ok(Vocabulary {
            words,
            added,
            special: self.specials.clone(),
            bos: self.bos.clone(),
            eos: self.eos.clone(),
            add_bos: self
                .boundary_properties
                .0
                .unwrap_or_else(|| probe.len() > 1 && probe.first().copied() == self.bos_id),
            add_eos: self
                .boundary_properties
                .1
                .unwrap_or_else(|| probe.len() > 1 && probe.last().copied() == self.eos_id),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn cleanup_equivalence_guard(reference: &Reference) {
        let fragments = [
            "", " ", " .", " ?", " !", " ,", " ' ", " n't", " 'm", " 's", " 've", " 're", " ' t",
            "  '  ", "é𝕒", "\0\n\r\t", "plain",
        ];
        for left in fragments {
            for right in fragments {
                let input = format!("{left}{right}");
                let mut expected = input.clone();
                for (from, to) in [
                    (" .", "."),
                    (" ?", "?"),
                    (" !", "!"),
                    (" ,", ","),
                    (" ' ", "'"),
                    (" n't", "n't"),
                    (" 'm", "'m"),
                    (" 's", "'s"),
                    (" 've", "'ve"),
                    (" 're", "'re"),
                ] {
                    expected = expected.replace(from, to);
                }
                assert_eq!(reference.clean_text(input), expected);
            }
        }
        for input in ["a", "é𝕒", "unchanged whitespace", "\0\n\r\t"] {
            let text = input.to_owned();
            let allocation = text.as_ptr();
            let cleaned = reference.clean_text(text);
            assert_eq!(cleaned, input);
            assert_eq!(cleaned.as_ptr(), allocation);
        }
    }
    #[test]
    fn focused_added_token_overlay_equivalence_guard() -> Result<()> {
        let original = Tokenizer::from_bytes(serde_json::to_vec(
            &json!({"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":1,"content":"a","single_word":false,"lstrip":false,"rstrip":false,"normalized":true,"special":false}],"normalizer":{"type":"Lowercase"},"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<unk>":0,"a":1,"x":2,".":3,"":4},"unk_token":"<unk>"}}),
        )?)?;
        let specials = vec!["<s>".to_owned(), "<t>".to_owned()];
        let unique = BTreeMap::from([
            (
                1,
                json!({"content":"a","single_word":true,"normalized":false}),
            ),
            (2, json!({"content":"x","lstrip":true,"rstrip":true})),
            (3, json!({"content":".","special":true})),
            (5, json!({"content":"<s>","normalized":false})),
            (
                6,
                json!({"content":"X","normalized":true,"single_word":true}),
            ),
            (
                7,
                json!({"content":"<t>","normalized":false,"lstrip":true,"rstrip":true}),
            ),
            (8, json!("NEW")),
        ]);
        // Cover the fast path and every reason to retain ordered scalar updates,
        // including an earlier ID error before a later malformed descriptor.
        for descriptions in [
            BTreeMap::new(),
            unique,
            BTreeMap::from([(5, json!("X")), (6, json!({"content":"X","special":true}))]),
            BTreeMap::from([(4, json!("")), (5, json!("X"))]),
            BTreeMap::from([
                (5, json!("X")),
                (6, json!({"content":"Y","lstrip":"invalid"})),
            ]),
            BTreeMap::from([
                (9, json!("X")),
                (10, json!({"content":"Y","normalized":null})),
            ]),
            BTreeMap::from([(9, json!("X")), (10, json!("Y"))]),
            BTreeMap::from([(5, json!(null))]),
        ] {
            let mut scalar = original.clone();
            // Keep the previous implementation independent from the optimized
            // helper so IDs, descriptor flags and error precedence are checked.
            let expected = (|| -> Result<()> {
                for (id, value) in &descriptions {
                    let content = token_content(value)
                        .ok_or_else(|| invalid("HF added-token content missing"))?;
                    scalar.add_tokens(&[added_token(
                        value,
                        specials.iter().any(|token| token == content),
                    )?]);
                    if scalar.token_to_id(content) != Some(*id) {
                        return Err(invalid(
                            "HF added-token ID differs from publisher descriptor",
                        ));
                    }
                }
                Ok(())
            })();
            let mut batched = original.clone();
            let actual = add_descriptions(&mut batched, &descriptions, &specials);
            let succeeded = expected.is_ok();
            assert_eq!(
                actual.map_err(|error| error.to_string()),
                expected.map_err(|error| error.to_string())
            );
            if !succeeded {
                continue;
            }
            assert_eq!(batched.get_vocab(true), scalar.get_vocab(true));
            assert_eq!(
                batched.get_added_tokens_decoder(),
                scalar.get_added_tokens_decoder()
            );
            let fragments = [
                "", "a", "Aa", "x", "X", "XX", "NEW", "<s>", "<t>", ".", "é", "\0", "\t", "\n",
                " ", "  ",
            ];
            for left in fragments {
                for right in fragments {
                    let input = format!("{left}{right}");
                    for special in [false, true] {
                        assert_eq!(
                            batched.encode(input.as_str(), special)?.get_ids(),
                            scalar.encode(input.as_str(), special)?.get_ids(),
                            "{input:?}"
                        );
                    }
                }
            }
            for skip_special in [false, true] {
                for id in 0..=9 {
                    assert_eq!(
                        batched.decode(&[id], skip_special)?,
                        scalar.decode(&[id], skip_special)?
                    );
                }
            }
        }
        Ok(())
    }
    #[test]
    fn focused_hf_wrapper_configuration_guard() -> Result<()> {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "SwiftBonsai-HfReference-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&directory)?;
        let test = (|| -> Result<()> {
            let json = json!({"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":4,"content":".","single_word":false,"lstrip":false,"rstrip":false,"normalized":true,"special":false}],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":{"type":"WordPiece","prefix":"##","cleanup":false},"model":{"type":"WordLevel","vocab":{"<unk>":0,"a":1,"<s>":2,"</s>":3,".":4},"unk_token":"<unk>"}});
            std::fs::write(directory.join("tokenizer.json"), serde_json::to_vec(&json)?)?;
            std::fs::write(
                directory.join("tokenizer_config.json"),
                serde_json::to_vec(
                    &json!({"tokenizer_class":"LlamaTokenizer","mask_token":".","add_bos_token":true,"add_eos_token":false,"clean_up_tokenization_spaces":true,"added_tokens_decoder":{"2":{"content":"<s>","normalized":false,"special":true},"3":{"content":"</s>","normalized":false,"special":true},"0":{"content":"<unk>","normalized":false,"special":true}}}),
                )?,
            )?;
            let reference = Reference::open(&directory)?;
            cleanup_equivalence_guard(&reference);
            assert_eq!(reference.encode("a", true)?, [2, 1]);
            assert_eq!(reference.encode("a", false)?, [1]);
            assert_eq!(reference.decode(&[2, 1, 4], false)?, "<s> a.");
            assert_eq!(reference.decode(&[2, 1, 4], true)?, "a.");
            assert_eq!(reference.decode(&[2, 3], true)?, "");
            // A legacy configured name does not override an existing non-special backend added token.
            assert_eq!(reference.decode(&[4], true)?, ".");
            let vocab = reference.vocabulary()?;
            assert!(vocab.add_bos);
            assert!(!vocab.add_eos);
            assert_eq!(vocab.words.iter().filter(|word| word.is_empty()).count(), 3);
            let inputs = ["", "a", "a .", "<s>a</s>", "é𝕒", "a\n\0", "\r\n"].map(str::to_owned);
            for special in [false, true] {
                let batch = reference.encode_batch(&inputs, special);
                let Backend::Fast(tokenizer) = &reference.backend else {
                    return Err(invalid("synthetic reference must use fast backend"));
                };
                for (input, actual) in inputs.iter().zip(batch) {
                    // The previous offset-producing API is the independent
                    // baseline for the IDs-only scalar and parallel paths.
                    let mut expected = tokenizer
                        .encode(input.as_str(), special && !reference.manual_boundaries)?
                        .get_ids()
                        .to_vec();
                    reference.add_boundaries(&mut expected, special)?;
                    assert_eq!(actual?, expected);
                    assert_eq!(reference.encode(input, special)?, expected);
                }
            }
            let ids: [&[u32]; 6] = [&[], &[2, 3], &[2, 1, 4], &[4], &[u32::MAX], &[u32::MAX, 1]];
            for skip_special in [false, true] {
                for (ids, actual) in ids.iter().zip(reference.decode_batch(&ids, skip_special)) {
                    assert_eq!(actual?, reference.decode(ids, skip_special)?);
                }
            }
            // One failed prefetched input must preserve successful neighbors,
            // not turn the complete batch into a source failure.
            std::fs::write(
                directory.join("tokenizer.json"),
                serde_json::to_vec(
                    &json!({"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"a":0,"b":1},"unk_token":"<missing>"}}),
                )?,
            )?;
            std::fs::write(
                directory.join("tokenizer_config.json"),
                serde_json::to_vec(&json!({"tokenizer_class":"PreTrainedTokenizerFast"}))?,
            )?;
            let reference = Reference::open(&directory)?;
            let inputs = ["a", "absent", "b"].map(str::to_owned);
            let batch = reference.encode_batch(&inputs, true);
            assert_eq!(
                batch.iter().map(Result::is_ok).collect::<Vec<_>>(),
                [true, false, true]
            );
            for (input, actual) in inputs.iter().zip(batch) {
                assert_eq!(
                    actual.map_err(|error| error.to_string()),
                    reference
                        .encode(input, true)
                        .map_err(|error| error.to_string())
                );
            }
            Ok(())
        })();
        for name in ["tokenizer.json", "tokenizer_config.json"] {
            let _ = std::fs::remove_file(directory.join(name));
        }
        let _ = std::fs::remove_dir(directory);
        test
    }

    #[test]
    fn cached_decode_ids_keep_sparse_vocabulary_and_added_tokens() -> Result<()> {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "SwiftBonsai-HfSparseReference-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&directory)?;
        let test = (|| -> Result<()> {
            // Sparse model IDs and an added special must all follow the
            // backend lookup, without copying token strings to test existence.
            std::fs::write(
                directory.join("tokenizer.json"),
                serde_json::to_vec(
                    &json!({"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":{"type":"WordPiece","prefix":"##","cleanup":false},"model":{"type":"WordLevel","vocab":{"a":0,"b":7,"<special>":8},"unk_token":"a"}}),
                )?,
            )?;
            std::fs::write(
                directory.join("tokenizer_config.json"),
                serde_json::to_vec(
                    &json!({"tokenizer_class":"PreTrainedTokenizerFast","additional_special_tokens":["<special>"]}),
                )?,
            )?;
            let reference = Reference::open(&directory)?;
            let Backend::Fast(tokenizer) = &reference.backend else {
                return Err(invalid("sparse reference must use fast backend"));
            };
            assert_eq!(reference.valid_ids, [0, 7, 8]);
            for id in (0..=9).chain([u32::MAX]) {
                assert_eq!(
                    reference.valid_ids.binary_search(&id).is_ok(),
                    tokenizer.id_to_token(id).is_some()
                );
            }
            let ids: [&[u32]; 5] = [&[0], &[1], &[7], &[8], &[1, 7, 8, u32::MAX]];
            for skip_special in [false, true] {
                for (ids, actual) in ids.iter().zip(reference.decode_batch(&ids, skip_special)) {
                    assert_eq!(actual?, tokenizer.decode(ids, skip_special)?);
                    assert_eq!(
                        reference.decode(ids, skip_special)?,
                        tokenizer.decode(ids, skip_special)?
                    );
                }
            }
            Ok(())
        })();
        for name in ["tokenizer.json", "tokenizer_config.json"] {
            std::fs::remove_file(directory.join(name))?;
        }
        std::fs::remove_dir(&directory)?;
        test
    }
}
