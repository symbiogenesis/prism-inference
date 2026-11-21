//! Clean-room `LLGuidance` and vocabulary/Unicode session op dispatchers.
//!
//! Delegates to `reference_llguidance::LlgRegistry` and `reference_vocab::ReferenceVocab`
//! without importing production runtimes or native bridges.

use std::sync::Arc;

use serde_json::{Value, json};

use super::gguf_model::ModelData;
use crate::reference_llguidance::LlgRegistry;
use crate::reference_vocab::ReferenceVocab;
use crate::{Result, invalid};

const MAX_ITEMS: usize = 512;
const MAX_SUPPLIED_BYTES: usize = 1_048_576;
const MAX_RETURNED_BYTES: usize = 4 * 1_048_576;

pub(super) fn handles_llg(op: &str) -> bool {
    op.starts_with("llg-")
}

pub(super) fn handles_vocab(op: &str) -> bool {
    matches!(
        op,
        "vocab-tokenize"
            | "vocab-detokenize"
            | "raw-vocab-detokenize"
            | "unicode-encode"
            | "unicode-decode"
    )
}

pub(super) fn execute_llg(
    registry: &mut LlgRegistry,
    operation: &Value,
    resolve_model: impl Fn(&Value) -> Result<Arc<ModelData>>,
) -> Result<Value> {
    let op = operation["Op"]
        .as_str()
        .ok_or_else(|| invalid("LLG Op must be string"))?;
    match op {
        "llg-available" => {
            exact_obj(operation, &["Op"])?;
            Ok(json!({ "Enabled": true }))
        }
        "llg-vocabulary" => {
            exact_obj(operation, &["Op", "Model"])?;
            let model = resolve_model(&operation["Model"])?;
            Ok(LlgRegistry::vocabulary_info(model.vocab.full_ref()?))
        }
        "llg-new" => {
            let model = resolve_model(&operation["Model"])?;
            registry.sampler_new_grammar(operation, model.vocab.full_ref()?)
        }
        "llg-chain-new" => registry.sampler_new_chain(operation),
        "llg-dist-new" => registry.sampler_new_dist(operation),
        "llg-chain-add" => registry.sampler_chain_add(operation),
        "llg-candidates-new" => registry.candidates_new(operation),
        "llg-candidates-inspect" => registry.candidates_inspect(operation),
        "llg-candidates-clone" => registry.candidates_clone(operation),
        "llg-candidates-copy" => registry.candidates_copy(operation),
        "llg-candidates-load" => registry.candidates_load(operation),
        "llg-candidates-patch" => registry.candidates_patch(operation),
        "llg-apply" => registry.sampler_apply(operation),
        "llg-reset" => registry.sampler_reset(operation),
        "llg-accept" => registry.sampler_accept(operation),
        "llg-free" => registry.sampler_free(operation),
        "llg-candidates-free" => registry.candidates_free(operation),
        _ => Err(invalid(format!("unknown LLG operation {op}"))),
    }
}

pub(super) fn execute_vocab(
    operation: &Value,
    vocab: Option<&Arc<ReferenceVocab>>,
) -> Result<Value> {
    let op = operation["Op"]
        .as_str()
        .ok_or_else(|| invalid("vocab session Op must be string"))?;
    match op {
        "unicode-encode" => exec_unicode_encode(operation),
        "unicode-decode" => exec_unicode_decode(operation),
        "vocab-tokenize" => exec_vocab_tokenize(
            operation,
            vocab.ok_or_else(|| invalid("vocab-tokenize requires context"))?,
        ),
        "vocab-detokenize" | "raw-vocab-detokenize" => exec_vocab_detokenize(
            op,
            operation,
            vocab.ok_or_else(|| invalid("vocab detokenize requires context"))?,
        ),
        _ => Err(invalid(format!("unknown vocab session op {op}"))),
    }
}

fn exact_obj(val: &Value, keys: &[&str]) -> Result<()> {
    let obj = val
        .as_object()
        .ok_or_else(|| invalid("operation must be JSON object"))?;
    if obj.len() != keys.len() || keys.iter().any(|k| !obj.contains_key(*k)) {
        return Err(invalid("vocab/LLG operation fields mismatch"));
    }
    Ok(())
}

fn decode_hex_bounded(val: &Value) -> Result<Vec<u8>> {
    let s = val
        .as_str()
        .ok_or_else(|| invalid("vocab hex must be string"))?;
    if s.len() > 2 * MAX_SUPPLIED_BYTES || !s.len().is_multiple_of(2) {
        return Err(invalid("vocab hex extent outside bound"));
    }
    if s.bytes()
        .any(|b| !b.is_ascii_digit() && !(b'a'..=b'f').contains(&b))
    {
        return Err(invalid("vocab hex must be lowercase hex"));
    }
    Ok(hex::decode(s)?)
}

fn exec_unicode_encode(operation: &Value) -> Result<Value> {
    exact_obj(operation, &["Op", "CodePoints"])?;
    let items = operation["CodePoints"]
        .as_array()
        .ok_or_else(|| invalid("CodePoints must be array"))?;
    if items.len() > MAX_ITEMS {
        return Err(invalid("Unicode batch exceeds 512"));
    }
    let mut values = Vec::with_capacity(items.len());
    let mut errors = Vec::with_capacity(items.len());
    let mut returned = 0_usize;
    for item in items {
        let cp = u32::try_from(
            item.as_u64()
                .ok_or_else(|| invalid("codepoint must be unsigned int"))?,
        )?;
        if let Some(ch) = char::from_u32(cp) {
            let mut buf = [0_u8; 4];
            let bytes = ch.encode_utf8(&mut buf).as_bytes();
            returned += bytes.len();
            if returned > MAX_RETURNED_BYTES {
                return Err(invalid("unicode-encode payload exceeds 4 MiB"));
            }
            values.push(Value::String(hex::encode(bytes)));
            errors.push(Value::Null);
        } else {
            values.push(Value::Null);
            errors.push(
                json!({ "Kind": "invalid-argument", "Message": "invalid Unicode codepoint" }),
            );
        }
    }
    Ok(json!({ "Values": values, "Errors": errors }))
}

fn exec_unicode_decode(operation: &Value) -> Result<Value> {
    exact_obj(operation, &["Op", "InputsHex"])?;
    let items = operation["InputsHex"]
        .as_array()
        .ok_or_else(|| invalid("InputsHex must be array"))?;
    if items.len() > MAX_ITEMS {
        return Err(invalid("Unicode batch exceeds 512"));
    }
    let mut supplied = 0_usize;
    let mut decoded_inputs = Vec::with_capacity(items.len());
    for item in items {
        let b = decode_hex_bounded(item)?;
        supplied += b.len();
        if supplied > MAX_SUPPLIED_BYTES {
            return Err(invalid("Unicode input exceeds 1 MiB"));
        }
        decoded_inputs.push(b);
    }
    let mut values = Vec::with_capacity(items.len());
    let mut errors = Vec::with_capacity(items.len());
    let mut returned = 0_usize;
    for bytes in decoded_inputs {
        if let Ok(s) = std::str::from_utf8(&bytes) {
            let cpts: Vec<u32> = s.chars().map(u32::from).collect();
            returned += cpts.len() * 4;
            if returned > MAX_RETURNED_BYTES {
                return Err(invalid("unicode-decode payload exceeds 4 MiB"));
            }
            values.push(json!(cpts));
            errors.push(Value::Null);
        } else {
            values.push(Value::Null);
            errors.push(json!({ "Kind": "invalid-argument", "Message": "invalid UTF-8 input" }));
        }
    }
    Ok(json!({ "Values": values, "Errors": errors }))
}

fn exec_vocab_tokenize(operation: &Value, vocab: &Arc<ReferenceVocab>) -> Result<Value> {
    exact_obj(
        operation,
        &["Op", "Context", "InputsHex", "AddSpecial", "ParseSpecial"],
    )?;
    let add_special = operation["AddSpecial"]
        .as_bool()
        .ok_or_else(|| invalid("AddSpecial must be boolean"))?;
    let parse_special = operation["ParseSpecial"]
        .as_bool()
        .ok_or_else(|| invalid("ParseSpecial must be boolean"))?;
    let items = operation["InputsHex"]
        .as_array()
        .ok_or_else(|| invalid("InputsHex must be array"))?;
    if items.len() > MAX_ITEMS {
        return Err(invalid("vocab-tokenize batch exceeds 512"));
    }
    let mut inputs = Vec::with_capacity(items.len());
    let mut supplied = 0_usize;
    for item in items {
        let b = decode_hex_bounded(item)?;
        supplied += b.len();
        if supplied > MAX_SUPPLIED_BYTES {
            return Err(invalid("vocab-tokenize supplied bytes exceed 1 MiB"));
        }
        inputs.push(b);
    }
    let mut values = Vec::with_capacity(inputs.len());
    let mut errors = Vec::with_capacity(inputs.len());
    let mut returned = 0_usize;
    for bytes in inputs {
        match vocab.tokenize(&bytes, add_special, parse_special) {
            Ok(toks) => {
                returned += toks.len() * 4;
                if returned > MAX_RETURNED_BYTES {
                    return Err(invalid("vocab-tokenize payload exceeds 4 MiB"));
                }
                values.push(json!(toks));
                errors.push(Value::Null);
            }
            Err(err) => {
                values.push(Value::Null);
                errors.push(json!({ "Kind": "standard", "Message": err.to_string() }));
            }
        }
    }
    Ok(json!({ "Values": values, "Errors": errors }))
}

fn exec_vocab_detokenize(
    op: &str,
    operation: &Value,
    vocab: &Arc<ReferenceVocab>,
) -> Result<Value> {
    let (remove_special, unparse_special) = if op == "vocab-detokenize" {
        exact_obj(operation, &["Op", "Context", "TokenLists", "Special"])?;
        let special = operation["Special"]
            .as_bool()
            .ok_or_else(|| invalid("Special must be boolean"))?;
        (!special, special)
    } else {
        exact_obj(
            operation,
            &[
                "Op",
                "Context",
                "TokenLists",
                "RemoveSpecial",
                "UnparseSpecial",
            ],
        )?;
        let rm = operation["RemoveSpecial"]
            .as_bool()
            .ok_or_else(|| invalid("RemoveSpecial must be boolean"))?;
        let unparse = operation["UnparseSpecial"]
            .as_bool()
            .ok_or_else(|| invalid("UnparseSpecial must be boolean"))?;
        (rm, unparse)
    };
    let lists = operation["TokenLists"]
        .as_array()
        .ok_or_else(|| invalid("TokenLists must be array"))?;
    if lists.len() > MAX_ITEMS {
        return Err(invalid("vocab detokenizer batch exceeds 512"));
    }
    let mut token_lists = Vec::with_capacity(lists.len());
    let mut supplied = 0_usize;
    let vocab_count = i32::try_from(vocab.vocab_size())?;
    for list in lists {
        let arr = list
            .as_array()
            .ok_or_else(|| invalid("token list must be array"))?;
        if arr.len() > 65_536 {
            return Err(invalid("vocab detokenizer token extent exceeds bound"));
        }
        supplied += arr.len() * 4;
        if supplied > MAX_SUPPLIED_BYTES {
            return Err(invalid("vocab detokenizer supplied bytes exceed 1 MiB"));
        }
        let mut toks = Vec::with_capacity(arr.len());
        for t in arr {
            let id = i32::try_from(
                t.as_u64()
                    .ok_or_else(|| invalid("detokenizer token must be u32"))?,
            )?;
            if id < 0 || id >= vocab_count {
                return Err(invalid("detokenizer token outside vocabulary"));
            }
            toks.push(id);
        }
        token_lists.push(toks);
    }
    let mut values = Vec::with_capacity(token_lists.len());
    let mut errors = Vec::with_capacity(token_lists.len());
    let mut returned = 0_usize;
    for toks in token_lists {
        match vocab.detokenize(&toks, remove_special, unparse_special) {
            Ok(bytes) => {
                returned += bytes.len();
                if returned > MAX_RETURNED_BYTES {
                    return Err(invalid("vocab detokenizer payload exceeds 4 MiB"));
                }
                values.push(Value::String(hex::encode(&bytes)));
                errors.push(Value::Null);
            }
            Err(err) => {
                values.push(Value::Null);
                errors.push(json!({ "Kind": "standard", "Message": err.to_string() }));
            }
        }
    }
    Ok(json!({ "Values": values, "Errors": errors }))
}
