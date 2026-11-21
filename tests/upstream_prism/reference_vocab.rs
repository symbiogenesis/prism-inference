//! Clean-room independent safe Rust GGUF vocabulary loader and tokenizer provider.
//!
//! No production `bonsai_prism_runtime`, `bonsai_gguf`, `tokenizers`, or native
//! bridge code is imported or called here. Parses GGUF v2/v3 vocabulary metadata
//! directly and implements `SPM`, `BPE` (with pretokenizer regex tables,
//! `ignore_merges`, and `llama3` cleanup semantics), `WPM`, `UGM`, `RWKV`, and
//! `PLaMo2` tokenization/detokenization for `vocab-runtime-info` and
//! `vocab-evaluate`, plus shared helpers for persistent reference sessions.

#[expect(
    dead_code,
    reason = "Submodule exposes full vocabulary methods for both file-mode and session-mode consumers."
)]
#[path = "reference_vocab/algorithms.rs"]
mod algorithms;
#[path = "reference_vocab/gguf.rs"]
mod gguf;
#[path = "reference_vocab/pretokenizer.rs"]
mod pretokenizer;

use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use serde_json::{Value, json};

use super::llama_grammar::{array, exact_fields, field, uint};
use super::{Result, invalid};
pub(super) use algorithms::ReferenceVocab;

/// Upstream source routes backed by the file-mode vocabulary provider.
pub(super) const IDS: &[&str] = &[
    "tests/test-tokenizer-0.cpp",
    "tests/test-tokenizer-0.sh",
    "tests/test-tokenizers-repo.sh",
];

const SOURCE_SHA256: &str = "3ab4be406c0bd04f0ddf951f3dc6a1cc594bb3fe1f56ae7da08ba077bd5f9d18";
const MAX_GGUF_FILE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_OPERATIONS: usize = 128;
const MAX_INPUTS: usize = 1024;
const MAX_INPUT_BYTES: usize = 65_536;
const MAX_TOTAL_INPUT_BYTES: usize = 1024 * 1024;
const MAX_WORKERS: usize = 256;
const MAX_WORKER_ACTIONS: usize = 1024;
const MAX_TOTAL_ACTIONS: usize = 262_144;
const MAX_CAPTURES: usize = 4096;
const MAX_CAPTURED_TOKENS_PER_WORKER: usize = 1_048_576;
const MAX_RETURNED_TOKENS: usize = 262_144;
const MAX_DETOKENIZED_BYTES: usize = 1024 * 1024;

/// Loads a bounded GGUF v2/v3 vocabulary independently from disk.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "Also consumed directly by upcoming persistent session reference modules."
    )
)]
pub(super) fn load_gguf_vocab(path: &Path) -> Result<Arc<ReferenceVocab>> {
    let meta = gguf::read_vocabulary_metadata(path, MAX_GGUF_FILE_BYTES)?;
    Ok(Arc::new(ReferenceVocab::from_metadata(&meta)?))
}

fn kind(value: &Value) -> Result<&str> {
    field(value, "Kind")?
        .as_str()
        .ok_or_else(|| invalid("vocabulary Kind must be text"))
}

fn boolean(value: &Value, name: &str) -> Result<bool> {
    field(value, name)?
        .as_bool()
        .ok_or_else(|| invalid(format!("vocabulary {name} must be boolean")))
}

fn validate_evaluate(operation: &Value) -> Result<()> {
    exact_fields(
        operation,
        &["Kind", "Path", "CreateContext", "Inputs", "Workers"],
    )?;
    let path = field(operation, "Path")?
        .as_str()
        .ok_or_else(|| invalid("vocabulary Path must be text"))?;
    if path.len() > 32_768 || path.contains('\0') || !Path::new(path).is_absolute() {
        return Err(invalid("vocabulary Path must be absolute and bounded"));
    }
    let _create_context = boolean(operation, "CreateContext")?;
    let inputs = array(operation, "Inputs")?;
    let workers = array(operation, "Workers")?;
    if inputs.len() > MAX_INPUTS || workers.len() > MAX_WORKERS {
        return Err(invalid("vocabulary request extent exceeds bound"));
    }
    let mut supplied = 0_usize;
    for input in inputs {
        exact_fields(input, &["Bytes", "AddSpecial", "ParseSpecial"])?;
        let _ = boolean(input, "AddSpecial")?;
        let _ = boolean(input, "ParseSpecial")?;
        let bytes = array(input, "Bytes")?;
        if bytes.len() > MAX_INPUT_BYTES {
            return Err(invalid("vocabulary input bytes exceed bound"));
        }
        supplied += bytes.len();
        if supplied > MAX_TOTAL_INPUT_BYTES {
            return Err(invalid("vocabulary total input bytes exceed bound"));
        }
        for byte in bytes {
            let value = byte
                .as_u64()
                .ok_or_else(|| invalid("vocabulary byte must be unsigned"))?;
            u8::try_from(value)?;
        }
    }
    let mut scheduled = 0_usize;
    let mut captured = 0_usize;
    for worker in workers {
        exact_fields(worker, &["Actions"])?;
        let schedule = array(worker, "Actions")?;
        if schedule.len() > MAX_WORKER_ACTIONS {
            return Err(invalid("vocabulary worker schedule exceeds bound"));
        }
        scheduled += schedule.len();
        if scheduled > MAX_TOTAL_ACTIONS {
            return Err(invalid("vocabulary total schedule exceeds bound"));
        }
        for action in schedule {
            match kind(action)? {
                "clock" => exact_fields(action, &["Kind"])?,
                "tokenize" => {
                    exact_fields(action, &["Kind", "Input", "Capture", "Detokenize"])?;
                    if inputs.is_empty() {
                        return Err(invalid("vocabulary tokenize action requires input"));
                    }
                    let idx = usize::try_from(uint(action, "Input")?)?;
                    if idx >= inputs.len() {
                        return Err(invalid("vocabulary input index exceeds bound"));
                    }
                    let _ = boolean(action, "Detokenize")?;
                    if boolean(action, "Capture")? {
                        captured += 1;
                        if captured > MAX_CAPTURES {
                            return Err(invalid("vocabulary capture count exceeds bound"));
                        }
                    }
                }
                _ => return Err(invalid("unknown vocabulary worker action")),
            }
        }
    }
    Ok(())
}

fn validate_operation(operation: &Value) -> Result<()> {
    match kind(operation)? {
        "vocab-runtime-info" => exact_fields(operation, &["Kind"]),
        "vocab-evaluate" => validate_evaluate(operation),
        _ => Err(invalid("unknown vocabulary operation")),
    }
}

struct PreparedInput {
    bytes: Vec<u8>,
    add_special: bool,
    parse_special: bool,
}

fn run_worker(
    vocab: &ReferenceVocab,
    inputs: &[PreparedInput],
    schedule: &Value,
    started: Instant,
) -> Result<Value> {
    let actions = array(schedule, "Actions")?;
    let mut results = Vec::with_capacity(actions.len());
    let mut captured_tokens = 0_usize;
    for action in actions {
        if kind(action)? == "clock" {
            results.push(json!({"Microseconds": i64::try_from(started.elapsed().as_micros())?}));
            continue;
        }
        let input_idx = usize::try_from(uint(action, "Input")?)?;
        let input = &inputs[input_idx];
        let tokens = vocab.tokenize(&input.bytes, input.add_special, input.parse_special)?;
        if tokens.len() > MAX_RETURNED_TOKENS {
            return Err(invalid("vocabulary tokenized output exceeds bound"));
        }
        let text = if boolean(action, "Detokenize")? {
            let bytes = vocab.detokenize(&tokens, false, true)?;
            if bytes.len() > MAX_DETOKENIZED_BYTES {
                return Err(invalid("vocabulary detokenized bytes exceed bound"));
            }
            Some(hex::encode(bytes))
        } else {
            None
        };
        if boolean(action, "Capture")? {
            captured_tokens += tokens.len();
            if captured_tokens > MAX_CAPTURED_TOKENS_PER_WORKER {
                return Err(invalid("vocabulary captured tokens exceed bound"));
            }
            results.push(json!({"Tokens": tokens, "TextHex": text}));
        } else {
            results.push(json!({}));
        }
    }
    Ok(json!({"Completed": actions.len(), "Results": results}))
}

fn execute_evaluate(operation: &Value) -> Result<Value> {
    let path_str = field(operation, "Path")?
        .as_str()
        .ok_or_else(|| invalid("vocabulary Path must be text"))?;
    let path = Path::new(path_str);
    let file_meta = fs::metadata(path)?;
    if !file_meta.is_file() || file_meta.len() > MAX_GGUF_FILE_BYTES {
        return Err(invalid("vocabulary file missing or exceeds 64 MiB"));
    }
    let create_context = boolean(operation, "CreateContext")?;
    let mut response = json!({
        "ModelLoaded": false,
        "ContextLoaded": if create_context { Value::Bool(false) } else { Value::Null },
        "Workers": [],
        "Error": null,
    });
    let meta = match gguf::read_vocabulary_metadata(path, MAX_GGUF_FILE_BYTES) {
        Ok(value) => value,
        Err(error) => {
            response["Error"] = json!(error.to_string());
            return Ok(response);
        }
    };
    response["ModelLoaded"] = Value::Bool(true);
    let vocab = match ReferenceVocab::from_metadata(&meta) {
        Ok(value) => Arc::new(value),
        Err(error) => {
            response["Error"] = json!(error.to_string());
            return Ok(response);
        }
    };
    if create_context {
        response["ContextLoaded"] = Value::Bool(true);
    }
    let mut prepared = Vec::new();
    for input in array(operation, "Inputs")? {
        let raw_bytes = array(input, "Bytes")?;
        let mut bytes = Vec::with_capacity(raw_bytes.len());
        for byte in raw_bytes {
            let value = byte
                .as_u64()
                .ok_or_else(|| invalid("vocabulary byte must be unsigned"))?;
            bytes.push(u8::try_from(value)?);
        }
        prepared.push(PreparedInput {
            bytes,
            add_special: boolean(input, "AddSpecial")?,
            parse_special: boolean(input, "ParseSpecial")?,
        });
    }
    let started = Instant::now();
    let workers = array(operation, "Workers")?;
    let outcomes = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers.len());
        for schedule in workers {
            let vocab = Arc::clone(&vocab);
            let prepared = &prepared;
            handles.push(scope.spawn(move || run_worker(&vocab, prepared, schedule, started)));
        }
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .map_err(|_| invalid("independent reference vocabulary worker panicked"))?
            })
            .collect::<Result<Vec<_>>>()
    });
    match outcomes {
        Ok(observations) => response["Workers"] = Value::Array(observations),
        Err(error) => response["Error"] = json!(error.to_string()),
    }
    Ok(response)
}

pub(super) fn handles(value: &Value) -> bool {
    value["Operations"].as_array().is_some_and(|operations| {
        !operations.is_empty()
            && operations.iter().all(|operation| {
                matches!(
                    operation["Kind"].as_str(),
                    Some("vocab-runtime-info" | "vocab-evaluate")
                )
            })
    })
}

pub(super) fn execute(value: &Value) -> Result<Value> {
    exact_fields(value, &["SchemaVersion", "Operations"])?;
    if uint(value, "SchemaVersion")? != 1 {
        return Err(invalid("unsupported vocabulary request schema version"));
    }
    let operations = array(value, "Operations")?;
    if operations.is_empty() || operations.len() > MAX_OPERATIONS {
        return Err(invalid("vocabulary operation count outside 1..=128"));
    }
    for operation in operations {
        validate_operation(operation)?;
    }
    let results = operations
        .iter()
        .map(|operation| {
            if kind(operation)? == "vocab-runtime-info" {
                Ok(json!({"HardwareConcurrency": thread::available_parallelism()?.get()}))
            } else {
                execute_evaluate(operation)
            }
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(json!({"SchemaVersion": 1, "Results": results}))
}

pub(super) fn identity() -> Value {
    json!({
        "Kind": "independent-rust-vocab",
        "SupportedSourceIds": IDS,
        "UpstreamCommit": super::UPSTREAM,
        "AssertionSource": {"Path": "tests/test-tokenizer-0.cpp", "Sha256": SOURCE_SHA256},
        "Architecture": "clean-room-gguf-spm-bpe-wpm-ugm-rwkv-plamo2-v1",
        "NativeReferenceReplay": false,
        "ProductionRuntimeCalled": false,
        "CompleteSourceGate": false,
        "NativeRetirementQualified": false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::io::Write as _;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn meta_str(text: &str) -> gguf::MetaValue {
        gguf::MetaValue::Bytes(text.as_bytes().to_vec())
    }

    fn meta_strings(values: &[&str]) -> gguf::MetaValue {
        gguf::MetaValue::BytesArray(values.iter().map(|s| s.as_bytes().to_vec()).collect())
    }

    fn encode_gguf_string(out: &mut Vec<u8>, bytes: &[u8]) -> Result<()> {
        out.extend_from_slice(&u64::try_from(bytes.len())?.to_le_bytes());
        out.extend_from_slice(bytes);
        Ok(())
    }

    struct SyntheticGgufSpec<'a> {
        model: &'a str,
        tokens: &'a [&'a str],
        scores: Option<&'a [f32]>,
        types: Option<&'a [i32]>,
        extra_bools: &'a [(&'a str, bool)],
        extra_u32s: &'a [(&'a str, u32)],
    }

    fn write_synthetic_gguf(path: &Path, spec: &SyntheticGgufSpec<'_>) -> Result<()> {
        let mut kvs: Vec<(String, u32, Vec<u8>)> = Vec::new();
        {
            let mut b = Vec::new();
            encode_gguf_string(&mut b, spec.model.as_bytes())?;
            kvs.push(("tokenizer.ggml.model".into(), 8, b));
        }
        {
            let mut b = Vec::new();
            b.extend_from_slice(&8_u32.to_le_bytes());
            b.extend_from_slice(&u64::try_from(spec.tokens.len())?.to_le_bytes());
            for tok in spec.tokens {
                encode_gguf_string(&mut b, tok.as_bytes())?;
            }
            kvs.push(("tokenizer.ggml.tokens".into(), 9, b));
        }
        if let Some(scores) = spec.scores {
            let mut b = Vec::new();
            b.extend_from_slice(&6_u32.to_le_bytes());
            b.extend_from_slice(&u64::try_from(scores.len())?.to_le_bytes());
            for &score in scores {
                b.extend_from_slice(&score.to_le_bytes());
            }
            kvs.push(("tokenizer.ggml.scores".into(), 9, b));
        }
        if let Some(types) = spec.types {
            let mut b = Vec::new();
            b.extend_from_slice(&5_u32.to_le_bytes());
            b.extend_from_slice(&u64::try_from(types.len())?.to_le_bytes());
            for &ty in types {
                b.extend_from_slice(&ty.to_le_bytes());
            }
            kvs.push(("tokenizer.ggml.token_type".into(), 9, b));
        }
        for &(k, v) in spec.extra_bools {
            kvs.push((k.into(), 7, vec![u8::from(v)]));
        }
        for &(k, v) in spec.extra_u32s {
            kvs.push((k.into(), 4, v.to_le_bytes().to_vec()));
        }
        let mut raw = b"GGUF".to_vec();
        raw.extend_from_slice(&3_u32.to_le_bytes());
        raw.extend_from_slice(&0_u64.to_le_bytes());
        raw.extend_from_slice(&u64::try_from(kvs.len())?.to_le_bytes());
        for (key, kind, data) in kvs {
            encode_gguf_string(&mut raw, key.as_bytes())?;
            raw.extend_from_slice(&kind.to_le_bytes());
            raw.extend_from_slice(&data);
        }
        let mut file = fs::File::create(path)?;
        file.write_all(&raw)?;
        Ok(())
    }

    #[test]
    fn independent_bpe_pretokenizer_and_ignore_merges_cleanup() -> Result<()> {
        for pre in [
            "default",
            "gpt-2",
            "llama3",
            "llama-bpe",
            "llama4",
            "qwen2",
            "qwen35",
            "command-r",
            "deepseek-llm",
            "deepseek-coder",
            "deepseek-v3",
            "falcon",
            "mpt",
            "refact",
            "starcoder",
            "gpt-neox",
            "tekken",
            "gpt-4o",
            "superbpe",
        ] {
            let compiled = pretokenizer::compile_pretokenizer(pre)?;
            assert!(!compiled.is_empty());
        }
        let mut meta = BTreeMap::new();
        meta.insert("tokenizer.ggml.model".into(), meta_str("gpt2"));
        meta.insert("tokenizer.ggml.pre".into(), meta_str("llama-bpe"));
        meta.insert(
            "tokenizer.ggml.tokens".into(),
            meta_strings(&["<|begin_of_text|>", "Ġ", "!", "-", "$", "Ġ!-$", "Ġ!", "a"]),
        );
        meta.insert(
            "tokenizer.ggml.token_type".into(),
            gguf::MetaValue::IntArray(vec![3, 1, 1, 1, 1, 1, 1, 1]),
        );
        meta.insert("tokenizer.ggml.merges".into(), meta_strings(&["Ġ !"]));
        meta.insert(
            "tokenizer.ggml.bos_token_id".into(),
            gguf::MetaValue::UInt(0),
        );
        let vocab = ReferenceVocab::from_metadata(&meta)?;
        assert!(vocab.add_bos);
        assert_eq!(vocab.tokenize(b" !-$", false, false)?, vec![5]);
        assert_eq!(vocab.detokenize(&[5], false, true)?, b" !-$");
        assert_eq!(vocab.tokenize(b" !", true, false)?, vec![0, 6]);
        Ok(())
    }

    #[test]
    fn independent_spm_evaluate_and_workers() -> Result<()> {
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let dir =
            std::env::temp_dir().join(format!("prism-ref-vocab-{}-{stamp}", std::process::id()));
        fs::create_dir(&dir)?;

        let spm_path = dir.join("spm.gguf");
        write_synthetic_gguf(
            &spm_path,
            &SyntheticGgufSpec {
                model: "llama",
                tokens: &[
                    "<unk>",
                    "<s>",
                    "</s>",
                    "<0x80>",
                    "▁",
                    "▁foo",
                    "bar",
                    "▁foobar",
                    "f",
                    "o",
                    "b",
                    "a",
                    "r",
                    "▁f",
                    "▁fo",
                    "ba",
                ],
                scores: Some(&[
                    0.0, 0.0, 0.0, 0.0, 1.0, 5.0, 2.0, 9.0, 0.5, 0.5, 0.5, 0.5, 0.5, 3.0, 4.0, 1.5,
                ]),
                types: Some(&[2, 3, 3, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                extra_bools: &[("tokenizer.ggml.add_bos_token", true)],
                extra_u32s: &[
                    ("tokenizer.ggml.unknown_token_id", 0),
                    ("tokenizer.ggml.bos_token_id", 1),
                    ("tokenizer.ggml.eos_token_id", 2),
                ],
            },
        )?;
        let spm = load_gguf_vocab(&spm_path)?;
        assert_eq!(spm.tokenize(b"foobar", false, false)?, vec![7]);
        assert_eq!(spm.detokenize(&[7], false, true)?, b"foobar");
        assert_eq!(spm.tokenize(b"\x80", false, true)?, vec![4, 3]);
        assert_eq!(spm.detokenize(&[4, 3], false, true)?, b"\x80");
        assert_eq!(spm.tokenize(b"foobar", true, false)?, vec![1, 7]);
        assert_eq!(spm.detokenize(&[1, 7], false, true)?, b"<s>foobar");
        assert!(spm.is_control(1) && spm.is_eog(2));
        assert_eq!(spm.token_attr_flags(3), Some(32));
        assert_eq!(spm.token_type_raw(6), Some(1));
        assert_eq!(spm.token_text_raw(5), Some("▁foo".as_bytes()));
        assert_eq!(spm.token_to_piece(5, 1, false)?, b"foo");

        let info = execute(&json!({
            "SchemaVersion": 1,
            "Operations": [{"Kind": "vocab-runtime-info"}],
        }))?;
        assert!(info["Results"][0]["HardwareConcurrency"].as_u64() >= Some(1));

        let eval = execute(&json!({
            "SchemaVersion": 1,
            "Operations": [{
                "Kind": "vocab-evaluate",
                "Path": spm_path,
                "CreateContext": true,
                "Inputs": [{"Bytes": b"foobar".to_vec(), "AddSpecial": false, "ParseSpecial": false}],
                "Workers": [
                    {"Actions": [{"Kind": "clock"}, {"Kind": "tokenize", "Input": 0, "Capture": true, "Detokenize": true}, {"Kind": "clock"}]},
                    {"Actions": [{"Kind": "tokenize", "Input": 0, "Capture": false, "Detokenize": false}]},
                ]
            }]
        }))?;
        let r = &eval["Results"][0];
        assert_eq!(r["ModelLoaded"], true);
        assert_eq!(r["ContextLoaded"], true);
        assert!(r["Error"].is_null());
        assert_eq!(r["Workers"][0]["Results"][1]["Tokens"], json!([7]));
        assert_eq!(
            r["Workers"][0]["Results"][1]["TextHex"],
            hex::encode(b"foobar")
        );
        assert_eq!(r["Workers"][1]["Results"][0], json!({}));

        let corrupt_path = dir.join("corrupt.gguf");
        fs::write(&corrupt_path, b"NOT_GGUF")?;
        let bad = execute(&json!({
            "SchemaVersion": 1,
            "Operations": [{
                "Kind": "vocab-evaluate",
                "Path": corrupt_path,
                "CreateContext": true,
                "Inputs": [],
                "Workers": []
            }]
        }))?;
        assert_eq!(bad["Results"][0]["ModelLoaded"], false);
        assert_eq!(bad["Results"][0]["ContextLoaded"], false);
        assert!(bad["Results"][0]["Error"].is_string());

        fs::remove_dir_all(&dir)?;
        assert_eq!(identity()["NativeReferenceReplay"], false);
        Ok(())
    }

    #[test]
    fn independent_wpm_ugm_and_rwkv_algorithms() -> Result<()> {
        let mut wpm_meta = BTreeMap::new();
        wpm_meta.insert("tokenizer.ggml.model".into(), meta_str("bert"));
        wpm_meta.insert(
            "tokenizer.ggml.tokens".into(),
            meta_strings(&["[UNK]", "[CLS]", "[SEP]", "▁play", "ing"]),
        );
        wpm_meta.insert(
            "tokenizer.ggml.token_type".into(),
            gguf::MetaValue::IntArray(vec![2, 3, 3, 1, 1]),
        );
        wpm_meta.insert(
            "tokenizer.ggml.unknown_token_id".into(),
            gguf::MetaValue::UInt(0),
        );
        wpm_meta.insert(
            "tokenizer.ggml.cls_token_id".into(),
            gguf::MetaValue::UInt(1),
        );
        wpm_meta.insert(
            "tokenizer.ggml.seperator_token_id".into(),
            gguf::MetaValue::UInt(2),
        );
        let wpm = ReferenceVocab::from_metadata(&wpm_meta)?;
        assert_eq!(wpm.tokenize(b"Playing", true, false)?, vec![1, 3, 4, 2]);
        assert_eq!(wpm.tokenize(b"unknown", false, false)?, vec![0]);

        let mut ugm_meta = BTreeMap::new();
        ugm_meta.insert("tokenizer.ggml.model".into(), meta_str("ugm"));
        ugm_meta.insert(
            "tokenizer.ggml.tokens".into(),
            meta_strings(&["<unk>", "▁", "a", "b", "ab", "▁ab"]),
        );
        ugm_meta.insert(
            "tokenizer.ggml.scores".into(),
            gguf::MetaValue::FloatArray(vec![-100.0, -2.0, -5.0, -5.0, -3.0, -1.0]),
        );
        let ugm = ReferenceVocab::from_metadata(&ugm_meta)?;
        assert_eq!(ugm.tokenize(b"ab", false, false)?, vec![5]);

        let mut rwkv_meta = BTreeMap::new();
        rwkv_meta.insert("tokenizer.ggml.model".into(), meta_str("rwkv"));
        rwkv_meta.insert(
            "tokenizer.ggml.tokens".into(),
            meta_strings(&["<pad>", "he", "llo"]),
        );
        let rwkv = ReferenceVocab::from_metadata(&rwkv_meta)?;
        assert_eq!(rwkv.tokenize(b"hello", false, false)?, vec![1, 2]);
        Ok(())
    }
}
