//! Full safe Rust ports of the SPM and BPE tokenizer roundtrip sources.
//!
//! Rust enumerates every vocabulary token and the exact Unicode domain, owns
//! exception skips, ignore-merges checks, worker strides and shared stopping.
//! The native session performs only supplied production batch primitives. Pure
//! operations within the current batch may finish after another worker fails;
//! Rust ignores those observations and schedules no further batches.
//! The approved BPE correction restores its intended invalid-UTF8 exception skip:
//! the pinned bulk Unicode decoder replaces invalid bytes instead of throwing.
//! Strict Rust validity checks reject only malformed raw pieces before encoding.
//! Upstream MIT attribution: tools/prism/upstream/LICENSE.

use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::engine::Engine;
use super::llama_grammar::{array, exact_fields, field, uint};
use super::tokenizer_session::{Client, ModelSession};
use super::{Result, invalid, result};

const SPM_ID: &str = "tests/test-tokenizer-1-spm.cpp";
const BPE_ID: &str = "tests/test-tokenizer-1-bpe.cpp";
const FIXTURE: &str = include_str!("tokenizer-roundtrip-assets.json");
const BATCH: usize = 512;
const UNICODE_LIMIT: u32 = 0x0011_0000;
const UNICODE_CALLS: usize = 456_703;
const UNICODE_PREDICATES: usize = 456_702;
const NAMES: [&str; 10] = [
    "llama-spm",
    "baichuan",
    "aquila",
    "falcon",
    "gpt-2",
    "gpt-neox",
    "llama-bpe",
    "mpt",
    "refact",
    "starcoder",
];
const DEFAULT_BPE: [&str; 7] = [
    "aquila",
    "gpt-2",
    "gpt-neox",
    "llama-bpe",
    "mpt",
    "refact",
    "starcoder",
];
const ORIGINAL_BPE_DIAGNOSTICS: [&str; 1] = ["falcon"];

fn text<'a>(value: &'a Value, name: &str) -> Result<&'a str> {
    field(value, name)?
        .as_str()
        .ok_or_else(|| invalid(format!("tokenizer roundtrip {name} must be text")))
}

fn manifest() -> Result<Value> {
    let value: Value = serde_json::from_str(FIXTURE)?;
    let sources = array(&value, "Sources")?;
    if sources.len() != 2
        || text(&sources[0], "Source")? != SPM_ID
        || text(&sources[1], "Source")? != BPE_ID
        || text(&sources[0], "Sha256")?
            != "87ca4367b6acd9ddf1e3fc694f2ae02b569070eb66faa5f88068f81aaed35bfb"
        || text(&sources[1], "Sha256")?
            != "2de134d10ff27590c714cfa70a75beb07be4dbeb971f78f2d9969ed0daaee36a"
    {
        return Err(invalid("tokenizer roundtrip source identity mismatch"));
    }
    let assets = array(&value, "Vocabularies")?;
    if assets.len() != NAMES.len() {
        return Err(invalid("roundtrip documented fixture extent mismatch"));
    }
    for (index, (asset, name)) in assets.iter().zip(NAMES).enumerate() {
        if text(asset, "Name")? != name
            || uint(asset, "ExpectedType")? != if index < 2 { 1 } else { 2 }
            || field(asset, "IgnoreMerges")?.as_bool() != Some(name == "llama-bpe")
        {
            return Err(invalid("roundtrip documented fixture variant mismatch"));
        }
    }
    if uint(&value, "InputSelectionVersion")? != 3 {
        return Err(invalid("BPE acceptance input selection version differs"));
    }
    for (key, expected) in [
        ("DefaultBpeAcceptance", DEFAULT_BPE.as_slice()),
        (
            "OriginalBpeDiagnostics",
            ORIGINAL_BPE_DIAGNOSTICS.as_slice(),
        ),
    ] {
        let actual = array(&value, key)?;
        if actual.len() != expected.len()
            || actual
                .iter()
                .zip(expected)
                .any(|(a, b)| a.as_str() != Some(*b))
        {
            return Err(invalid("BPE acceptance/diagnostic input selection differs"));
        }
    }
    Ok(value)
}

fn verify_asset(engine: &Engine, asset: &Value) -> Result<std::path::PathBuf> {
    let path = engine.source_path(&format!("models/ggml-vocab-{}.gguf", text(asset, "Name")?))?;
    if fs::metadata(&path)?.len() != uint(asset, "Length")? {
        return Err(invalid("roundtrip vocabulary length mismatch"));
    }
    let mut file = File::open(&path)?;
    let mut buffer = [0; 16 * 1024];
    let mut hash = Sha256::new();
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
    if digest != text(asset, "Sha256")? {
        return Err(invalid("roundtrip vocabulary digest mismatch"));
    }
    Ok(path)
}

fn call(client: &Client, worker: u32, operation: Value) -> Result<Value> {
    Ok(client.call(worker, operation, None)?.value)
}

#[derive(Clone)]
struct ProductionError {
    kind: String,
    message: String,
}
enum Sample {
    Value(Value),
    Error(ProductionError),
}

fn samples(client: &Client, worker: u32, operation: Value, count: usize) -> Result<Vec<Sample>> {
    let response = call(client, worker, operation)?;
    exact_fields(&response, &["Values", "Errors"])?;
    let values = array(&response, "Values")?;
    let errors = array(&response, "Errors")?;
    if count > BATCH || values.len() != count || errors.len() != count {
        return Err(invalid("roundtrip primitive batch extent mismatch"));
    }
    values
        .iter()
        .zip(errors)
        .map(|(value, error)| {
            if error.is_null() {
                if value.is_null() {
                    return Err(invalid("roundtrip successful primitive returned null"));
                }
                Ok(Sample::Value(value.clone()))
            } else {
                exact_fields(error, &["Kind", "Message"])?;
                let kind = text(error, "Kind")?;
                if !matches!(kind, "invalid-argument" | "standard" | "unknown") || !value.is_null()
                {
                    return Err(invalid("roundtrip exception metadata malformed"));
                }
                Ok(Sample::Error(ProductionError {
                    kind: kind.into(),
                    message: text(error, "Message")?.into(),
                }))
            }
        })
        .collect()
}

fn hex(value: &Value) -> Result<String> {
    let value = value
        .as_str()
        .ok_or_else(|| invalid("roundtrip raw bytes must be hex text"))?;
    if value.len() > 2 * 1024 * 1024
        || value.len() % 2 != 0
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(invalid("roundtrip raw byte encoding malformed"));
    }
    Ok(value.into())
}

fn token_list(value: &Value, vocabulary: usize) -> Result<Vec<i32>> {
    let list = value
        .as_array()
        .ok_or_else(|| invalid("roundtrip tokens must be array"))?;
    if list.len() > 65536 {
        return Err(invalid("roundtrip token list exceeds bound"));
    }
    list.iter()
        .map(|value| {
            let id = i32::try_from(
                value
                    .as_i64()
                    .ok_or_else(|| invalid("roundtrip token must be int32"))?,
            )?;
            if id < 0 || usize::try_from(id)? >= vocabulary {
                return Err(invalid(
                    "production tokenizer returned token outside vocabulary",
                ));
            }
            Ok(id)
        })
        .collect()
}

#[derive(Default)]
struct Row {
    id: u32,
    original: Option<String>,
    tokens: Option<Vec<i32>>,
    check: Option<String>,
    skipped: bool,
    restored_utf8_skip: bool,
    failure: Option<String>,
    merge: Option<bool>,
}

impl Row {
    fn active(&self) -> bool {
        !self.skipped && self.failure.is_none() && self.merge != Some(false)
    }
    fn error(&mut self, stage: &str, error: &ProductionError, catches_invalid: bool) {
        if catches_invalid && error.kind == "invalid-argument" {
            self.skipped = true;
        } else {
            self.failure = Some(format!("{stage}: {}: {}", error.kind, error.message));
        }
    }
}

fn active(rows: &[Row]) -> Vec<usize> {
    rows.iter()
        .enumerate()
        .filter_map(|(i, row)| row.active().then_some(i))
        .collect()
}

fn validate_utf8(client: &Client, rows: &mut [Row]) -> Result<()> {
    restore_utf8_piece_skips(rows)?;
    let indices = active(rows);
    let inputs: Vec<_> = indices
        .iter()
        .map(|index| rows[*index].original.clone())
        .collect();
    let outputs = samples(
        client,
        0,
        json!({"Op":"unicode-decode","InputsHex":inputs}),
        indices.len(),
    )?;
    for (index, output) in indices.into_iter().zip(outputs) {
        match output {
            Sample::Value(value) => {
                let values = value
                    .as_array()
                    .ok_or_else(|| invalid("production Unicode decoder returned non-array"))?;
                for point in values {
                    u32::try_from(
                        point
                            .as_u64()
                            .ok_or_else(|| invalid("Unicode codepoint must be unsigned"))?,
                    )?;
                }
            }
            Sample::Error(error) => rows[index].error("UTF8 validation", &error, true),
        }
    }
    Ok(())
}

fn restore_utf8_piece_skips(rows: &mut [Row]) -> Result<()> {
    for row in rows.iter_mut().filter(|row| row.active()) {
        let encoded = row
            .original
            .as_ref()
            .ok_or_else(|| invalid("BPE initial piece missing"))?;
        let encoded = hex(&json!(encoded))?;
        let bytes = encoded
            .as_bytes()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| {
                let digit = |value: u8| {
                    if value <= b'9' {
                        value - b'0'
                    } else {
                        value - b'a' + 10
                    }
                };
                (digit(pair[0]) << 4) | digit(pair[1])
            })
            .collect::<Vec<_>>();
        if std::str::from_utf8(&bytes).is_err() {
            row.skipped = true;
            row.restored_utf8_skip = true;
        }
    }
    Ok(())
}

fn retokenize(
    client: &Client,
    worker: u32,
    context: u64,
    rows: &mut [Row],
    parse_special: bool,
    catches_invalid: bool,
    vocabulary: usize,
) -> Result<()> {
    let indices = active(rows);
    let inputs: Vec<_> = indices
        .iter()
        .map(|index| rows[*index].original.clone())
        .collect();
    let outputs = samples(
        client,
        worker,
        json!({"Op":"vocab-tokenize","Context":context,"InputsHex":inputs,"AddSpecial":false,"ParseSpecial":parse_special}),
        indices.len(),
    )?;
    for (index, output) in indices.into_iter().zip(outputs) {
        match output {
            Sample::Value(value) => rows[index].tokens = Some(token_list(&value, vocabulary)?),
            Sample::Error(error) => rows[index].error("tokenization", &error, catches_invalid),
        }
    }
    Ok(())
}

fn detokenize(
    client: &Client,
    worker: u32,
    context: u64,
    rows: &mut [Row],
    catches_invalid: bool,
) -> Result<()> {
    let indices = active(rows);
    let tokens: Vec<_> = indices
        .iter()
        .map(|index| rows[*index].tokens.clone())
        .collect();
    let outputs = samples(
        client,
        worker,
        json!({"Op":"vocab-detokenize","Context":context,"TokenLists":tokens,"Special":true}),
        indices.len(),
    )?;
    for (index, output) in indices.into_iter().zip(outputs) {
        match output {
            Sample::Value(value) => rows[index].check = Some(hex(&value)?),
            Sample::Error(error) => {
                rows[index].error("check detokenization", &error, catches_invalid);
            }
        }
    }
    Ok(())
}

#[derive(Default)]
struct Counts {
    tokens_visited: usize,
    token_checks: usize,
    merge_checks: usize,
    invalid_skips: usize,
    restored_utf8_skips: usize,
    unicode_visited: usize,
    unicode_checks: usize,
    exempt_9601: usize,
    failures: Vec<String>,
}

impl Counts {
    fn merge(&mut self, other: Self) {
        self.tokens_visited += other.tokens_visited;
        self.token_checks += other.token_checks;
        self.merge_checks += other.merge_checks;
        self.invalid_skips += other.invalid_skips;
        self.restored_utf8_skips += other.restored_utf8_skips;
        self.unicode_visited += other.unicode_visited;
        self.unicode_checks += other.unicode_checks;
        self.exempt_9601 += other.exempt_9601;
        self.failures.extend(other.failures);
    }
    fn json(&self, asset: &Value) -> Value {
        let mut arguments = vec![format!(
            "models/ggml-vocab-{}.gguf",
            asset["Name"].as_str().unwrap_or("")
        )];
        if asset["IgnoreMerges"] == true {
            arguments.push("--ignore-merges".into());
        }
        json!({"Name":asset["Name"],"IgnoreMerges":asset["IgnoreMerges"],"InputSha256":asset["Sha256"],"InputLength":asset["Length"],"SourceArguments":arguments,"TokenMetadataCount":asset["TokenMetadataCount"],"TokenRowsVisited":self.tokens_visited,"TokenRoundtripPredicates":self.token_checks,"IgnoreMergePredicates":self.merge_checks,"CaughtInvalidArgumentSkips":self.invalid_skips-self.restored_utf8_skips,"RestoredInvalidUtf8Skips":self.restored_utf8_skips,"TotalSkippedRows":self.invalid_skips,"UnicodeRowsVisited":self.unicode_visited,"UnicodeEqualityPredicates":self.unicode_checks,"Exempt9601":self.exempt_9601,"Failures":self.failures})
    }
}

fn observe_tokens(rows: Vec<Row>, counts: &mut Counts) -> Result<bool> {
    for row in rows {
        counts.tokens_visited += 1;
        if let Some(merge) = row.merge {
            counts.merge_checks += 1;
            if !merge {
                counts
                    .failures
                    .push(format!("token{} retokenized to multiple tokens", row.id));
                return Ok(false);
            }
        }
        if row.skipped {
            counts.invalid_skips += 1;
            counts.restored_utf8_skips += usize::from(row.restored_utf8_skip);
            continue;
        }
        if let Some(failure) = row.failure {
            counts.failures.push(format!("token{}: {failure}", row.id));
            return Ok(false);
        }
        let original = row
            .original
            .ok_or_else(|| invalid("roundtrip initial observation missing"))?;
        let check = row
            .check
            .ok_or_else(|| invalid("roundtrip check observation missing"))?;
        counts.token_checks += 1;
        if original != check {
            counts.failures.push(format!(
                "token{} byte roundtrip mismatch: {original} -> {check}",
                row.id
            ));
            return Ok(false);
        }
    }
    Ok(true)
}

fn vocabulary_phase(
    client: &Client,
    context: u64,
    vocabulary: usize,
    bpe: bool,
    ignore_merges: bool,
) -> Result<Counts> {
    let mut counts = Counts::default();
    for start in (0..vocabulary).step_by(BATCH) {
        let end = (start + BATCH).min(vocabulary);
        let ids: Vec<_> = (start..end).map(|id| vec![id]).collect();
        let outputs = samples(
            client,
            0,
            json!({"Op":"vocab-detokenize","Context":context,"TokenLists":ids,"Special":true}),
            end - start,
        )?;
        let mut rows = Vec::new();
        for (id, output) in (start..end).zip(outputs) {
            let mut row = Row {
                id: u32::try_from(id)?,
                ..Row::default()
            };
            match output {
                Sample::Value(value) => row.original = Some(hex(&value)?),
                Sample::Error(error) => row.error("initial detokenization", &error, false),
            }
            rows.push(row);
        }
        if bpe {
            validate_utf8(client, &mut rows)?;
        }
        retokenize(client, 0, context, &mut rows, true, bpe, vocabulary)?;
        if ignore_merges {
            for row in &mut rows {
                if row.active() {
                    row.merge = Some(
                        row.tokens
                            .as_ref()
                            .ok_or_else(|| invalid("merge token observation missing"))?
                            .len()
                            <= 1,
                    );
                }
            }
        }
        detokenize(client, 0, context, &mut rows, bpe)?;
        if !observe_tokens(rows, &mut counts)? {
            break;
        }
    }
    Ok(counts)
}

fn excluded(point: u32) -> bool {
    (0x0000_d800..=0x0000_dfff).contains(&point) || (0x0004_0000..=0x000e_0000).contains(&point)
}

fn unicode_batch(
    client: &Client,
    worker: u32,
    context: u64,
    points: &[u32],
    bpe: bool,
    vocabulary: usize,
) -> Result<Vec<Row>> {
    let outputs = samples(
        client,
        worker,
        json!({"Op":"unicode-encode","CodePoints":points}),
        points.len(),
    )?;
    let mut rows = Vec::new();
    for (point, output) in points.iter().copied().zip(outputs) {
        let mut row = Row {
            id: point,
            ..Row::default()
        };
        match output {
            Sample::Value(value) => row.original = Some(hex(&value)?),
            Sample::Error(error) => row.error("codepoint encoding", &error, false),
        }
        rows.push(row);
    }
    retokenize(client, worker, context, &mut rows, !bpe, false, vocabulary)?;
    detokenize(client, worker, context, &mut rows, false)?;
    Ok(rows)
}

fn observe_unicode(rows: Vec<Row>, counts: &mut Counts, stop: &AtomicBool) -> Result<()> {
    for row in rows {
        if stop.load(Ordering::SeqCst) {
            break;
        }
        counts.unicode_visited += 1;
        if let Some(failure) = row.failure {
            counts
                .failures
                .push(format!("codepoint{:x}: {failure}", row.id));
            stop.store(true, Ordering::SeqCst);
            break;
        }
        let original = row
            .original
            .ok_or_else(|| invalid("Unicode encoding observation missing"))?;
        let check = row
            .check
            .ok_or_else(|| invalid("Unicode detokenization observation missing"))?;
        if row.id == 9601 {
            counts.exempt_9601 += 1;
            continue;
        }
        counts.unicode_checks += 1;
        if original != check {
            counts.failures.push(format!(
                "codepoint{:x} byte roundtrip mismatch: {original} -> {check}",
                row.id
            ));
            stop.store(true, Ordering::SeqCst);
            break;
        }
    }
    Ok(())
}

struct UnicodeWork {
    context: u64,
    workers: u32,
    vocabulary: usize,
    bpe: bool,
}

fn unicode_worker(
    client: &Client,
    worker: u32,
    work: &UnicodeWork,
    stop: &AtomicBool,
) -> Result<Counts> {
    let mut counts = Counts::default();
    let mut point = worker;
    while point < UNICODE_LIMIT && !stop.load(Ordering::SeqCst) {
        let mut points = Vec::with_capacity(BATCH);
        while point < UNICODE_LIMIT && points.len() < BATCH && !stop.load(Ordering::SeqCst) {
            if !excluded(point) {
                points.push(point);
            }
            point += work.workers;
        }
        if points.is_empty() {
            continue;
        }
        let rows = match unicode_batch(
            client,
            worker,
            work.context,
            &points,
            work.bpe,
            work.vocabulary,
        ) {
            Ok(rows) => rows,
            Err(error) => {
                stop.store(true, Ordering::SeqCst);
                return Err(error);
            }
        };
        if let Err(error) = observe_unicode(rows, &mut counts, stop) {
            stop.store(true, Ordering::SeqCst);
            return Err(error);
        }
    }
    Ok(counts)
}

fn unicode_phase(client: &Client, work: &UnicodeWork) -> Result<Counts> {
    let stop = AtomicBool::new(false);
    thread::scope(|scope| {
        let mut threads = Vec::new();
        for worker in 0..work.workers {
            let client = client.clone();
            let stop = &stop;
            threads.push(scope.spawn(move || unicode_worker(&client, worker, work, stop)));
        }
        let mut counts = Counts::default();
        for worker in threads {
            counts.merge(
                worker
                    .join()
                    .map_err(|_| invalid("Rust Unicode worker panicked"))??,
            );
        }
        Ok(counts)
    })
}

fn release(client: &Client, kind: &str, handle: u64) -> Result<()> {
    call(
        client,
        0,
        json!({"Op":"release","Kind":kind,"Handle":handle}),
    )?;
    Ok(())
}

fn loaded_checks(
    client: &Client,
    context: u64,
    model: u64,
    asset: &Value,
    bpe: bool,
) -> Result<Counts> {
    let metadata = call(client, 0, json!({"Op":"vocab","Model":model}))?;
    exact_fields(&metadata, &["Count", "Bos", "Type"])?;
    let vocabulary = usize::try_from(uint(&metadata, "Count")?)?;
    let mut counts = Counts::default();
    if uint(&metadata, "Type")? != uint(asset, "ExpectedType")? {
        counts
            .failures
            .push("source vocabulary-type guard failed".into());
        return Ok(counts);
    }
    if vocabulary != usize::try_from(uint(asset, "TokenMetadataCount")?)? {
        counts
            .failures
            .push("vocabulary extent differs from pinned input metadata".into());
        return Ok(counts);
    }
    counts = vocabulary_phase(
        client,
        context,
        vocabulary,
        bpe,
        asset["IgnoreMerges"] == true,
    )?;
    if !counts.failures.is_empty() {
        return Ok(counts);
    }
    if bpe && asset["IgnoreMerges"] == true && counts.merge_checks == 0 {
        return Err(invalid(
            "complete BPE source did not exercise its ignore-merges predicate",
        ));
    }
    let runtime = call(client, 0, json!({"Op":"hardware-concurrency"}))?;
    exact_fields(&runtime, &["Count"])?;
    let workers = u32::try_from(uint(&runtime, "Count")?)?;
    if workers == 0 || workers > 16 {
        return Err(invalid(
            "source hardware workers exceed persistent session domain",
        ));
    }
    counts.merge(unicode_phase(
        client,
        &UnicodeWork {
            context,
            workers,
            vocabulary,
            bpe,
        },
    )?);
    if counts.failures.is_empty()
        && (counts.tokens_visited != vocabulary
            || counts.unicode_visited != UNICODE_CALLS
            || counts.unicode_checks != UNICODE_PREDICATES
            || counts.exempt_9601 != 1)
    {
        return Err(invalid("complete tokenizer source extent mismatch"));
    }
    Ok(counts)
}

fn run_asset(engine: &Engine, asset: &Value, bpe: bool) -> Result<Option<Counts>> {
    let Some(session) = ModelSession::open(
        engine,
        &format!("tokenizer-roundtrip-{}", text(asset, "Name")?),
    )?
    else {
        return Ok(None);
    };
    let client = session.client();
    let path = verify_asset(engine, asset)?;
    call(&client, 0, json!({"Op":"backend-init"}))?;
    let model = call(
        &client,
        0,
        json!({"Op":"model-load","Path":path,"Base":"default","Params":null,"Overrides":{"VocabOnly":true},"Progress":false}),
    )?;
    exact_fields(&model, &["Model"])?;
    let mut counts = Counts::default();
    if field(&model, "Model")?.is_null() {
        counts
            .failures
            .push("source vocabulary model loading failed".into());
    } else {
        let model = uint(&model, "Model")?;
        let context = call(
            &client,
            0,
            json!({"Op":"context","Model":model,"Params":null,"Overrides":{}}),
        )?;
        exact_fields(&context, &["Context"])?;
        if field(&context, "Context")?.is_null() {
            counts.failures.push("source context loading failed".into());
        } else {
            let context = uint(&context, "Context")?;
            counts = loaded_checks(&client, context, model, asset, bpe)?;
            release(&client, "context", context)?;
        }
        release(&client, "model", model)?;
    }
    call(&client, 0, json!({"Op":"backend-free"}))?;
    session.finish()?;
    Ok(Some(counts))
}

fn run_source(engine: &Engine, bpe: bool) -> Result<Value> {
    let id = if bpe { BPE_ID } else { SPM_ID };
    let manifest = manifest()?;
    let mut observations = Vec::new();
    let mut passed = true;
    for asset in array(&manifest, "Vocabularies")?.iter().filter(|asset| {
        if bpe {
            asset["Name"]
                .as_str()
                .is_some_and(|name| DEFAULT_BPE.contains(&name))
        } else {
            asset["ExpectedType"] == 1
        }
    }) {
        let Some(counts) = run_asset(engine, asset, bpe)? else {
            return Ok(result(id,false,"Rust production persistent vocabulary API unavailable; exhaustive source contract unsupported".into()));
        };
        passed &= counts.failures.is_empty();
        observations.push(counts.json(asset));
    }
    if observations.len() != if bpe { DEFAULT_BPE.len() } else { 2 } {
        return Err(invalid("roundtrip full fixture variant count mismatch"));
    }
    let receipt = json!({"Vocabularies":observations,"UnicodeDomainCallsPerVocabulary":UNICODE_CALLS,"UnicodePredicatesPerVocabulary":UNICODE_PREDICATES,"BatchSize":BATCH,"InputSelectionVersion":manifest["InputSelectionVersion"],"DefaultBpeAcceptance":manifest["DefaultBpeAcceptance"],"OriginalBpeDiagnostics":manifest["OriginalBpeDiagnostics"],"InputSelectionPolicy":manifest["InputSelectionPolicy"],"ProductionCorrection":bpe.then_some(&manifest["ProductionCorrection"]),"SourceCorrection":if bpe{Some("Approved restoration of intended invalid-UTF8 skip: strict Rust validity before retokenization, because pinned unicode_cpts_from_utf8 replaces invalid bytes instead of throwing") }else{None},"StopPolicy":"Rust checks shared stop before batches and each returned-item assertion; speculative pure operations in an in-flight batch are ignored after failure"});
    Ok(result(id, passed, receipt.to_string()))
}

pub(super) fn run_spm(engine: &Engine) -> Result<Value> {
    run_source(engine, false)
}
pub(super) fn run_bpe(engine: &Engine) -> Result<Value> {
    run_source(engine, true)
}

/// The source's vocabulary argument and optional flag remain independently
/// selectable. Diagnostics retain actual failures and cannot count as a source.
pub(super) fn diagnostic(engine: &Engine, configuration: &Value) -> Result<Value> {
    exact_fields(configuration, &["Kind", "Vocabulary", "IgnoreMerges"])?;
    if text(configuration, "Kind")? != "tokenizer-bpe-roundtrip" {
        return Err(invalid("unknown tokenizer roundtrip diagnostic"));
    }
    let name = text(configuration, "Vocabulary")?;
    let ignore = field(configuration, "IgnoreMerges")?
        .as_bool()
        .ok_or_else(|| invalid("BPE diagnostic flag must be boolean"))?;
    let fixture = manifest()?;
    let mut asset = array(&fixture, "Vocabularies")?
        .iter()
        .find(|asset| asset["Name"] == name && asset["ExpectedType"] == 2)
        .ok_or_else(|| invalid("diagnostic requires a named pinned BPE vocabulary"))?
        .clone();
    asset["IgnoreMerges"] = json!(ignore);
    let counts = run_asset(engine, &asset, true)?;
    diagnostic_output(
        &asset,
        counts.as_ref(),
        uint(&fixture, "InputSelectionVersion")?,
    )
}

fn diagnostic_output(
    asset: &Value,
    counts: Option<&Counts>,
    selection_version: u64,
) -> Result<Value> {
    let name = text(asset, "Name")?;
    let (passed,receipt)=counts.map_or_else(
        ||(false,json!({"Name":name,"IgnoreMerges":asset["IgnoreMerges"],"Failure":"Runtime lacks the production vocabulary session APIs"})),
        |counts|(counts.failures.is_empty(),counts.json(asset)),
    );
    let mut output = result(
        &format!("diagnostic/tokenizer-bpe-roundtrip/{name}"),
        passed,
        serde_json::to_string(&receipt)?,
    );
    output["SourceId"] = json!(BPE_ID);
    output["SourceCount"] = json!(0);
    output["Measure"] = json!("diagnostic-workloads");
    output["InputSelectionVersion"] = json!(selection_version);
    output["InputSha256"] = asset["Sha256"].clone();
    output["IgnoreMerges"] = asset["IgnoreMerges"].clone();
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::{Row, excluded, hex, restore_utf8_piece_skips};
    use serde_json::json;

    #[test]
    fn explicit_acceptance_keeps_original_diagnostics_out_of_source_count() -> super::Result<()> {
        let fixture = super::manifest()?;
        let assets = super::array(&fixture, "Vocabularies")?;
        let llama = assets
            .iter()
            .find(|a| a["Name"] == "llama-bpe")
            .ok_or_else(|| super::invalid("Original Llama-BPE input missing"))?;
        if llama["IgnoreMerges"] != true
            || llama["TokenMetadataCount"] != 128_256
            || llama["Sha256"] != "97272e430d53bc7688f52d5e0ad8ea8f163ede9f1bbd1694feaa504797d5d96e"
            || super::DEFAULT_BPE.contains(&"falcon")
            || !super::DEFAULT_BPE.contains(&"llama-bpe")
            || assets.iter().any(|a| a["Name"] == "qwen2")
            || fixture["ProductionCorrection"]["CanonicalPatchSha256"]
                != "b52f1cceb6a36f9f511503c2e550a9b18a718c4b10272dbdf64cbfae8ae5fd01"
        {
            return Err(super::invalid("explicit BPE acceptance selection changed"));
        }
        for name in super::ORIGINAL_BPE_DIAGNOSTICS {
            let asset = assets
                .iter()
                .find(|a| a["Name"] == name)
                .ok_or_else(|| super::invalid("original BPE input was removed"))?;
            let counts = super::Counts {
                failures: vec!["independent failing-source receipt guard".into()],
                ..super::Counts::default()
            };
            let output = super::diagnostic_output(asset, Some(&counts), 3)?;
            if output["Status"] != "fail"
                || output["SourceCount"] != 0
                || output["SourceId"] != super::BPE_ID
                || output["Id"] == super::BPE_ID
            {
                return Err(super::invalid(
                    "diagnostic failure was counted as a source pass",
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn source_unicode_boundaries_and_raw_bytes() -> super::Result<()> {
        for point in [0x0000_d800, 0x0000_dfff, 0x0004_0000, 0x000e_0000] {
            if !excluded(point) {
                return Err(super::invalid("source exclusion endpoint lost"));
            }
        }
        for point in [
            0,
            9601,
            0x0000_d7ff,
            0x0000_e000,
            0x0003_ffff,
            0x000e_0001,
            0x0010_ffff,
        ] {
            if excluded(point) {
                return Err(super::invalid("source codepoint incorrectly excluded"));
            }
        }
        if hex(&json!("00ff"))? != "00ff" || hex(&json!("0x")).is_ok() {
            return Err(super::invalid("raw NUL/invalidUTF8 transport guard failed"));
        }
        Ok(())
    }

    #[test]
    fn restored_bpe_guard_keeps_valid_replacement_nul_and_unicode() -> super::Result<()> {
        let cases = [
            ("a1", true),
            ("ff", true),
            ("f09f", true),
            ("eda080", true),
            ("c080", true),
            ("efbfbd", false),
            ("00", false),
            ("c2a1", false),
            ("f09f9a80", false),
        ];
        let mut rows = cases
            .iter()
            .map(|(bytes, _)| Row {
                original: Some((*bytes).into()),
                ..Row::default()
            })
            .collect::<Vec<_>>();
        restore_utf8_piece_skips(&mut rows)?;
        for (row, (_, expected_skip)) in rows.iter().zip(cases) {
            if row.skipped != expected_skip
                || row.restored_utf8_skip != expected_skip
                || row.active() == expected_skip
            {
                return Err(super::invalid(
                    "restored BPE invalid-piece guard changed the valid UTF8 domain",
                ));
            }
        }
        Ok(())
    }
}
