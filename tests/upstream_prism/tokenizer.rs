//! Complete safe Rust default corpus checks from test-tokenizer-0.cpp.
//!
//! The fifteen `CMake` vocabulary fixtures stay in the verified external source
//! archive. Their .inp/.out bytes are independent token references, never native
//! engine output. Rust reproduces Windows text-mode input and `std::map` ordering,
//! supplies every worker schedule, and compares only worker zero as the source
//! does. Detokenization is diagnostic; it is not an added roundtrip assertion.
//! The optional text-file export is a separate CI-only diagnostic entry.
//! Upstream MIT attribution: tools/prism/upstream/LICENSE.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::engine::Engine;
use super::{Result, invalid, result};

const ID: &str = "tests/test-tokenizer-0.cpp";
const SOURCE_SHA256: &str = "3ab4be406c0bd04f0ddf951f3dc6a1cc594bb3fe1f56ae7da08ba077bd5f9d18";
const ASSETS: &str = include_str!("tokenizer-assets.json");
const NAMES: [&str; 15] = [
    "bert-bge",
    "command-r",
    "deepseek-coder",
    "deepseek-llm",
    "falcon",
    "gemma-4",
    "gpt-2",
    "llama-bpe",
    "llama-spm",
    "mpt",
    "phi-3",
    "qwen2",
    "qwen35",
    "refact",
    "starcoder",
];
const ORIGINAL_PREDICATES: usize = 694;
type Cases = BTreeMap<Vec<u8>, Vec<i32>>;

fn field<'a>(value: &'a Value, name: &str) -> Result<&'a Value> {
    value
        .get(name)
        .ok_or_else(|| invalid(format!("tokenizer field {name} missing")))
}

fn array<'a>(value: &'a Value, name: &str) -> Result<&'a [Value]> {
    field(value, name)?
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| invalid(format!("tokenizer {name} must be array")))
}

fn text<'a>(value: &'a Value, name: &str) -> Result<&'a str> {
    field(value, name)?
        .as_str()
        .ok_or_else(|| invalid(format!("tokenizer {name} must be text")))
}

fn uint(value: &Value, name: &str) -> Result<u64> {
    field(value, name)?
        .as_u64()
        .ok_or_else(|| invalid(format!("tokenizer {name} must be unsigned")))
}

fn boolean(value: &Value, name: &str) -> Result<bool> {
    field(value, name)?
        .as_bool()
        .ok_or_else(|| invalid(format!("tokenizer {name} must be boolean")))
}

fn exact_fields(value: &Value, names: &[&str]) -> Result<()> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid("tokenizer object required"))?;
    if object.len() != names.len() || names.iter().any(|name| !object.contains_key(*name)) {
        return Err(invalid("tokenizer response fields mismatch"));
    }
    Ok(())
}

fn exchange(engine: &Engine, operation: &Value) -> Result<Option<Value>> {
    let Some(value) = engine.execute_common(
        "tokenizer",
        &json!({"SchemaVersion":1,"Operations":[operation]}),
    )?
    else {
        return Ok(None);
    };
    exact_fields(&value, &["SchemaVersion", "Results"])?;
    let results = array(&value, "Results")?;
    if uint(&value, "SchemaVersion")? != 1 || results.len() != 1 {
        return Err(invalid("tokenizer response framing mismatch"));
    }
    Ok(Some(results[0].clone()))
}

fn manifest() -> Result<Value> {
    let value: Value = serde_json::from_str(ASSETS)?;
    if text(&value, "Source")? != ID || text(&value, "SourceSha256")? != SOURCE_SHA256 {
        return Err(invalid("tokenizer source identity mismatch"));
    }
    let assets = array(&value, "Assets")?;
    if assets.len() != NAMES.len() {
        return Err(invalid(
            "tokenizer must retain all fifteen CMake vocabularies",
        ));
    }
    let mut predicates = 0;
    for (asset, name) in assets.iter().zip(NAMES) {
        if text(asset, "Name")? != name || array(asset, "Files")?.len() != 3 {
            return Err(invalid("tokenizer CMake fixture order/extent mismatch"));
        }
        predicates += usize::try_from(uint(asset, "UniqueCases")?)?;
    }
    if predicates != ORIGINAL_PREDICATES {
        return Err(invalid("tokenizer original predicate extent mismatch"));
    }
    Ok(value)
}

fn verify_file(path: &Path, identity: &Value) -> Result<()> {
    if !path.is_absolute() || fs::metadata(path)?.len() != uint(identity, "Length")? {
        return Err(invalid(format!(
            "tokenizer asset length mismatch: {}",
            path.display()
        )));
    }
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0; 16 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    let mut digest = String::with_capacity(64);
    for byte in hash.finalize() {
        write!(digest, "{byte:02x}")?;
    }
    if digest != text(identity, "Sha256")? {
        return Err(invalid(format!(
            "tokenizer asset digest mismatch: {}",
            path.display()
        )));
    }
    Ok(())
}

fn asset_paths(engine: &Engine, asset: &Value) -> Result<[PathBuf; 3]> {
    let name = text(asset, "Name")?;
    let identities = array(asset, "Files")?;
    let mut paths = Vec::new();
    for (identity, suffix) in identities.iter().zip(["", ".inp", ".out"]) {
        if text(identity, "Suffix")? != suffix {
            return Err(invalid("tokenizer asset suffix mismatch"));
        }
        let path = engine.source_path(&format!("models/ggml-vocab-{name}.gguf{suffix}"))?;
        verify_file(&path, identity)?;
        paths.push(path);
    }
    paths
        .try_into()
        .map_err(|_| invalid("tokenizer asset path extent mismatch"))
}

// Windows CRT text streams collapse CRLF and stop at Ctrl-Z. Byte strings stay
// byte strings: tokenizer inputs may contain NUL or invalid UTF-8.
fn windows_text(raw: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(raw.len());
    for (index, byte) in raw.iter().copied().enumerate() {
        if byte == 0x1a {
            break;
        }
        if byte != b'\r' || raw.get(index + 1) != Some(&b'\n') {
            bytes.push(byte);
        }
    }
    bytes
}

fn read_cases(input: &[u8], output: &[u8]) -> Result<(usize, Cases)> {
    let input = windows_text(input);
    let output = windows_text(output);
    let separator = b"\n__ggml_vocab_test__\n";
    let mut inputs = Vec::new();
    let mut position = 0;
    while position < input.len() {
        if let Some(next) = input[position..]
            .windows(separator.len())
            .position(|bytes| bytes == separator)
        {
            inputs.push(&input[position..position + next]);
            position += next + separator.len();
        } else {
            inputs.push(&input[position..]);
            break;
        }
    }
    let mut lines: Vec<_> = if output.is_empty() {
        Vec::new()
    } else {
        output.split(|byte| *byte == b'\n').collect()
    };
    if output.last() == Some(&b'\n') {
        lines.pop();
    }
    if inputs.len() != lines.len() {
        return Err(invalid("tokenizer input/output row counts differ"));
    }
    let rows = inputs.len();
    let mut cases = BTreeMap::new();
    for (bytes, line) in inputs.into_iter().zip(lines) {
        let line = std::str::from_utf8(line)?
            .trim_matches(|character: char| character.is_ascii_whitespace());
        let tokens = if line.is_empty() {
            Vec::new()
        } else {
            line.split(' ')
                .map(str::parse::<i32>)
                .collect::<std::result::Result<Vec<_>, _>>()?
        };
        // Assignment matches std::map: a duplicate input replaces its reference.
        cases.insert(bytes.to_vec(), tokens);
    }
    if cases.is_empty() {
        return Err(invalid("tokenizer corpus has no tests"));
    }
    Ok((rows, cases))
}

fn inputs(cases: &Cases) -> Vec<Value> {
    cases
        .keys()
        .map(|bytes| json!({"Bytes":bytes,"AddSpecial":false,"ParseSpecial":false}))
        .collect()
}

fn token_action(index: usize, capture: bool, detokenize: bool) -> Value {
    json!({"Kind":"tokenize","Input":index,"Capture":capture,"Detokenize":detokenize})
}

fn tokens(value: &Value) -> Result<Vec<i32>> {
    array(value, "Tokens")?
        .iter()
        .map(|token| {
            Ok(i32::try_from(token.as_i64().ok_or_else(|| {
                invalid("tokenizer token must be int32")
            })?)?)
        })
        .collect()
}

fn validate_text_hex(value: &Value) -> Result<()> {
    let bytes = text(value, "TextHex")?.as_bytes();
    if bytes.len() > 2 * 1024 * 1024
        || bytes.len() % 2 != 0
        || !bytes.iter().all(u8::is_ascii_hexdigit)
    {
        return Err(invalid("tokenizer diagnostic byte encoding malformed"));
    }
    Ok(())
}

fn evaluate(engine: &Engine, path: &Path, inputs: &[Value], workers: &[Value]) -> Result<Value> {
    exchange(engine, &json!({"Kind":"vocab-evaluate","Path":path.to_str().ok_or_else(|| invalid("vocabulary path is not UTF-8"))?,"CreateContext":true,"Inputs":inputs,"Workers":workers}))?
        .ok_or_else(|| invalid("production vocabulary API disappeared"))
}

fn worker_results<'a>(worker: &'a Value, actions: &[Value]) -> Result<&'a [Value]> {
    exact_fields(worker, &["Completed", "Results"])?;
    let results = array(worker, "Results")?;
    if usize::try_from(uint(worker, "Completed")?)? != actions.len()
        || results.len() != actions.len()
    {
        return Err(invalid("tokenizer worker completion extent mismatch"));
    }
    Ok(results)
}

pub(super) fn run(engine: &Engine) -> Result<Value> {
    let Some(runtime) = exchange(engine, &json!({"Kind":"vocab-runtime-info"}))? else {
        return Ok(result(ID, false, "Rust production vocabulary API unavailable; all fifteen corpus contracts remain unsupported".into()));
    };
    exact_fields(&runtime, &["HardwareConcurrency"])?;
    let workers = usize::try_from(uint(&runtime, "HardwareConcurrency")?)?;
    if workers == 0 || workers > 256 {
        return Err(invalid(
            "source hardware worker count outside bounded transport",
        ));
    }
    let manifest = manifest()?;
    let mut checked = 0;
    let mut failures = Vec::new();
    let mut loader_failures = Vec::new();
    for asset in array(&manifest, "Assets")? {
        let name = text(asset, "Name")?;
        let [vocab, input, output] = asset_paths(engine, asset)?;
        let (rows, cases) = read_cases(&fs::read(input)?, &fs::read(output)?)?;
        if rows != usize::try_from(uint(asset, "Rows")?)?
            || cases.len() != usize::try_from(uint(asset, "UniqueCases")?)?
        {
            return Err(invalid("tokenizer reference corpus extent mismatch"));
        }
        let schedules: Vec<_> = (0..workers).map(|worker| json!({"Actions":(0..cases.len()).map(|index| token_action(index,worker==0,worker==0)).collect::<Vec<_>>()})).collect();
        let response = evaluate(engine, &vocab, &inputs(&cases), &schedules)?;
        exact_fields(
            &response,
            &["ModelLoaded", "ContextLoaded", "Workers", "Error"],
        )?;
        let model_loaded = boolean(&response, "ModelLoaded")?;
        let context_loaded = boolean(&response, "ContextLoaded")?;
        if let Some(error) = response.get("Error").filter(|error| !error.is_null()) {
            loader_failures.push(format!(
                "{name}: {}",
                error
                    .as_str()
                    .ok_or_else(|| invalid("vocabulary error must be null or text"))?
            ));
            continue;
        }
        if !model_loaded || !context_loaded {
            loader_failures.push(format!("{name}: vocabulary model/context loading failed"));
            continue;
        }
        let observations = array(&response, "Workers")?;
        if observations.len() != workers {
            return Err(invalid("tokenizer worker count mismatch"));
        }
        for (worker, observation) in observations.iter().enumerate() {
            let events = worker_results(observation, array(&schedules[worker], "Actions")?)?;
            if worker == 0 {
                for (index, (event, expected)) in events.iter().zip(cases.values()).enumerate() {
                    exact_fields(event, &["Tokens", "TextHex"])?;
                    validate_text_hex(event)?; // Printed diagnostic in source; no equality assertion.
                    let actual = tokens(event)?;
                    checked += 1;
                    if actual != *expected {
                        failures.push(format!(
                            "{name} case {index}: expected {expected:?}, observed {actual:?}"
                        ));
                    }
                }
            } else {
                for event in events {
                    exact_fields(event, &[])?;
                }
            }
        }
    }
    let passed =
        loader_failures.is_empty() && failures.is_empty() && checked == ORIGINAL_PREDICATES;
    let detail = format!(
        "15 bundled vocabularies; {checked}/{ORIGINAL_PREDICATES} source token-vector predicates; {workers} shared-context workers per vocabulary, worker0 comparisons only; loader failures={}; token failures={}; {}",
        loader_failures.len(),
        failures.len(),
        loader_failures
            .iter()
            .chain(&failures)
            .take(8)
            .cloned()
            .collect::<Vec<_>>()
            .join("; ")
    );
    Ok(result(ID, passed, detail))
}

/// Reuse the original corpus parser and shared-context worker semantics for an
/// independently verified external vocabulary repository fixture.
pub(super) fn check_corpus(
    engine: &Engine,
    name: &str,
    vocabulary: &Path,
    input: &[u8],
    output: &[u8],
) -> Result<Value> {
    let runtime = exchange(engine, &json!({"Kind":"vocab-runtime-info"}))?
        .ok_or_else(|| invalid("target lacks production vocabulary API"))?;
    exact_fields(&runtime, &["HardwareConcurrency"])?;
    let workers = usize::try_from(uint(&runtime, "HardwareConcurrency")?)?;
    if workers == 0 || workers > 256 || !vocabulary.is_absolute() {
        return Err(invalid(
            "external corpus workers/path outside source bounds",
        ));
    }
    let (rows, cases) = read_cases(input, output)?;
    let schedules: Vec<_> = (0..workers).map(|worker| json!({"Actions":(0..cases.len()).map(|index| token_action(index,worker==0,worker==0)).collect::<Vec<_>>()})).collect();
    let response = evaluate(engine, vocabulary, &inputs(&cases), &schedules)?;
    exact_fields(
        &response,
        &["ModelLoaded", "ContextLoaded", "Workers", "Error"],
    )?;
    if !boolean(&response, "ModelLoaded")?
        || !boolean(&response, "ContextLoaded")?
        || !response["Error"].is_null()
    {
        return Err(invalid(format!(
            "external vocabulary {name} failed production load: {}",
            response["Error"]
        )));
    }
    let observations = array(&response, "Workers")?;
    if observations.len() != workers {
        return Err(invalid("external corpus worker extent differs"));
    }
    let mut failures = Vec::new();
    for (worker, observation) in observations.iter().enumerate() {
        let events = worker_results(observation, array(&schedules[worker], "Actions")?)?;
        if worker == 0 {
            for (index, (event, expected)) in events.iter().zip(cases.values()).enumerate() {
                exact_fields(event, &["Tokens", "TextHex"])?;
                validate_text_hex(event)?;
                let actual = tokens(event)?;
                if actual != *expected {
                    failures.push(json!({"Case":index,"Expected":expected,"Actual":actual}));
                }
            }
        } else {
            for event in events {
                exact_fields(event, &[])?;
            }
        }
    }
    Ok(
        json!({"Name":name,"Rows":rows,"SourcePredicates":cases.len(),"Workers":workers,"Passed":failures.is_empty(),"Failures":failures}),
    )
}

/// Optional source text-file path: a bounded export and timing diagnostic.
/// It adds no corpus predicates and must only be executed in CI.
pub(super) fn export_text(
    engine: &Engine,
    name: &str,
    input_path: &Path,
    output_path: &Path,
) -> Result<Value> {
    if !NAMES.contains(&name)
        || !input_path.is_absolute()
        || !output_path.is_absolute()
        || input_path == output_path
        || fs::metadata(input_path)?.len() > 65536
    {
        return Err(invalid(
            "tokenizer diagnostic requires a bundled vocabulary, distinct absolute paths and at most 65536 input bytes",
        ));
    }
    let manifest = manifest()?;
    let asset = array(&manifest, "Assets")?
        .iter()
        .find(|asset| asset["Name"] == name)
        .ok_or_else(|| invalid("tokenizer diagnostic asset missing"))?;
    let [vocab, _, _] = asset_paths(engine, asset)?;
    let bytes = windows_text(&fs::read(input_path)?);
    let actions = vec![
        json!({"Kind":"clock"}),
        token_action(0, true, false),
        json!({"Kind":"clock"}),
    ];
    let response = evaluate(
        engine,
        &vocab,
        &[json!({"Bytes":bytes,"AddSpecial":false,"ParseSpecial":false})],
        &[json!({"Actions":actions})],
    )?;
    exact_fields(
        &response,
        &["ModelLoaded", "ContextLoaded", "Workers", "Error"],
    )?;
    if let Some(error) = response.get("Error").filter(|error| !error.is_null()) {
        return Err(invalid(format!(
            "tokenizer diagnostic: {}",
            error
                .as_str()
                .ok_or_else(|| invalid("vocabulary error must be null or text"))?
        )));
    }
    if !boolean(&response, "ModelLoaded")? || !boolean(&response, "ContextLoaded")? {
        return Err(invalid("tokenizer diagnostic vocabulary loading failed"));
    }
    let workers = array(&response, "Workers")?;
    if workers.len() != 1 {
        return Err(invalid("tokenizer diagnostic must have one worker"));
    }
    let events = worker_results(&workers[0], &actions)?;
    exact_fields(&events[0], &["Microseconds"])?;
    exact_fields(&events[2], &["Microseconds"])?;
    exact_fields(&events[1], &["Tokens", "TextHex"])?;
    if !events[1]["TextHex"].is_null() {
        return Err(invalid("tokenizer export unexpectedly detokenized"));
    }
    let start = events[0]["Microseconds"]
        .as_i64()
        .ok_or_else(|| invalid("tokenizer clock must be int64"))?;
    let end = events[2]["Microseconds"]
        .as_i64()
        .ok_or_else(|| invalid("tokenizer clock must be int64"))?;
    let elapsed = end
        .checked_sub(start)
        .filter(|value| *value >= 0)
        .ok_or_else(|| invalid("tokenizer clock moved backwards"))?;
    let tokens = tokens(&events[1])?;
    // std::ofstream text output on the accepted Windows target uses CRLF.
    let mut output = String::new();
    for token in &tokens {
        write!(output, "{token}\r\n")?;
    }
    fs::write(output_path, output)?;
    Ok(
        json!({"Source":ID,"Diagnostic":"text-file-export","Vocabulary":name,"InputBytes":bytes.len(),"TokenCount":tokens.len(),"Microseconds":elapsed,"Output":output_path,"TimingIncludes":"action dispatch and raw token serialization; excludes request parsing, loading and thread creation"}),
    )
}

#[cfg(test)]
mod tests {
    use super::{read_cases, windows_text};

    #[test]
    fn windows_text_and_map_reference_semantics() -> super::Result<()> {
        if windows_text(b"a\r\nb\rc\x1aignored") != b"a\nb\rc" {
            return Err(super::invalid("Windows text translation mismatch"));
        }
        let (rows, cases) = read_cases(
            b"z\n__ggml_vocab_test__\na\n__ggml_vocab_test__\nz\n__ggml_vocab_test__\n",
            b"1\r\n2\r\n3\r\n",
        )?;
        if rows != 3
            || cases.len() != 2
            || cases.get(b"z".as_slice()) != Some(&vec![3])
            || cases.first_key_value().map(|(key, _)| key.as_slice()) != Some(b"a".as_slice())
        {
            return Err(super::invalid(
                "source duplicate overwrite/map ordering mismatch",
            ));
        }
        if read_cases(b"a", b"1\n2\n").is_ok() {
            return Err(super::invalid("mismatched reference rows accepted"));
        }
        Ok(())
    }
}
