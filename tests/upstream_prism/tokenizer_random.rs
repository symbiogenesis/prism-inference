//! Full original tokenizer-random schedules and diagnostic verdicts, in safe Rust.
use super::engine::Engine;
use super::tokenizer_generators::{self as generators, Kind, Vocabulary};
use super::tokenizer_reference::Reference;
use super::tokenizer_session::VocabularySession;
use super::{Result, invalid, result};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Instant;

const ID: &str = "tests/test-tokenizer-random.py";
const BATCH: usize = 256;
const MAX_ERRORS: usize = 10;
// Each case owns a vocabulary, reference and production session. Keep memory
// bounded while sharing the existing inner tokenizer batch worker pool.
const CASE_WORKERS: usize = 2;
pub(super) const NAMES: [&str; 26] = [
    "llama-spm",
    "phi-3",
    "gemma",
    "gemma-2",
    "baichuan",
    "bert-bge",
    "jina-v2-en",
    "llama-bpe",
    "phi-2",
    "deepseek-llm",
    "deepseek-coder",
    "falcon",
    "mpt",
    "starcoder",
    "gpt-2",
    "stablelm2",
    "refact",
    "qwen2",
    "olmo",
    "jina-v2-es",
    "jina-v2-de",
    "smaug-bpe",
    "poro-chat",
    "jina-v2-code",
    "viking",
    "jais",
];

pub(super) struct Options {
    pub(super) generators: Vec<Kind>,
    pub(super) random_iterations: usize,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            generators: vec![
                Kind::Ascii,
                Kind::Apostrophe,
                Kind::Unicode,
                Kind::Vocab,
                Kind::Added,
            ],
            random_iterations: 100,
        }
    }
}
const fn generator_name(kind: Kind) -> &'static str {
    match kind {
        Kind::Custom => "generator_custom_text",
        Kind::Edges => "generator_custom_text_edge_cases",
        Kind::Ascii => "generator_ascii_lr_strip",
        Kind::Apostrophe => "generator_apostrophe",
        Kind::Unicode => "generator_unicodes",
        Kind::Vocab => "generator_vocab_words",
        Kind::Added => "generator_added_lr_strip",
        Kind::RandomAdded => "generator_random_added_tokens",
        Kind::RandomChars => "generator_random_chars",
        Kind::RandomUnicode => "generator_random_unicodes",
        Kind::RandomVocabChars => "generator_random_vocab_chars",
        Kind::RandomVocabWords => "generator_random_vocab_words",
    }
}
fn hex(bytes: &[u8]) -> Result<String> {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(output, "{byte:02x}")?;
    }
    Ok(output)
}
fn unhex(value: &Value) -> Result<Vec<u8>> {
    let text = value
        .as_str()
        .ok_or_else(|| invalid("Tokenizer native text is not hex"))?;
    if text.len() % 2 != 0 || text.len() > 2 * 1024 * 1024 {
        return Err(invalid("Tokenizer native text extent malformed"));
    }
    text.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|bytes| Ok(u8::from_str_radix(std::str::from_utf8(bytes)?, 16)?))
        .collect()
}
fn observations(value: &Value, count: usize) -> Result<Vec<Result<&Value>>> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid("Tokenizer observation is not an object"))?;
    let values = value
        .get("Values")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("Tokenizer native values missing"))?;
    let errors = value
        .get("Errors")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("Tokenizer native errors missing"))?;
    if object.len() != 2 || values.len() != count || errors.len() != count {
        return Err(invalid("Tokenizer native observation extent mismatch"));
    }
    Ok(values
        .iter()
        .zip(errors)
        .map(|(value, error)| {
            if error.is_null() {
                if value.is_null() {
                    Err(invalid("Tokenizer native null result without error"))
                } else {
                    Ok(value)
                }
            } else {
                Err(invalid(format!(
                    "Tokenizer production operation rejected input: {error}"
                )))
            }
        })
        .collect())
}
fn token_ids(value: &Value) -> Result<Vec<u32>> {
    value
        .as_array()
        .ok_or_else(|| invalid("Tokenizer native token list missing"))?
        .iter()
        .map(|id| {
            Ok(u32::try_from(id.as_u64().ok_or_else(|| {
                invalid("Tokenizer token is not nonnegative")
            })?)?)
        })
        .collect()
}
fn source_decode_matches(
    input: &str,
    reference: &str,
    production: &str,
    vocab: &Vocabulary,
) -> bool {
    if reference == production {
        return true;
    }
    let mut production = production;
    if vocab.add_bos
        && let Some(bos) = vocab.bos.as_deref()
        && let Some(rest) = production.strip_prefix(bos)
    {
        production = rest;
    }
    if vocab.add_eos
        && let Some(eos) = vocab.eos.as_deref()
        && let Some(rest) = production.strip_suffix(eos)
    {
        production = rest;
    }
    input == production
}
#[derive(Default)]
struct Comparison {
    compared: usize,
    prefetched: usize,
    encode_errors: usize,
    decode_errors: usize,
    encode_differences: usize,
    decode_differences: usize,
    input_hash: Sha256,
    diagnostics: Vec<Value>,
    reference_encode_ns: u128,
    production_encode_ns: u128,
    reference_decode_ns: u128,
    production_decode_ns: u128,
}
impl Comparison {
    const fn stopped(&self) -> bool {
        self.encode_errors >= MAX_ERRORS && self.decode_errors >= MAX_ERRORS
    }
    fn batch(
        &mut self,
        session: &mut VocabularySession<'_>,
        reference: &Reference,
        vocab: &Vocabulary,
        inputs: &[String],
    ) -> Result<bool> {
        if inputs.is_empty() {
            return Ok(!self.stopped());
        }
        self.prefetched += inputs.len();
        let start = Instant::now();
        // Store per-item reference errors so observations beyond the exact source
        // early-stop prefix cannot create an extra source failure.
        let expected = reference.encode_batch(inputs, true);
        self.reference_encode_ns += start.elapsed().as_nanos();
        let start = Instant::now();
        let supplied = expected
            .iter()
            .map(|ids| ids.as_ref().map_or(&[][..], Vec::as_slice))
            .collect::<Vec<_>>();
        let decoded = reference.decode_batch(&supplied, false);
        self.reference_decode_ns += start.elapsed().as_nanos();
        let observation = session.observe(inputs, &supplied)?;
        self.production_encode_ns += observation.encode_ns;
        self.production_decode_ns += observation.decode_ns;
        let actual = observations(&observation.tokens, inputs.len())?;
        let decoded_actual = observations(&observation.texts, inputs.len())?;
        for (((input, expected), actual), (decoded, decoded_actual)) in inputs
            .iter()
            .zip(expected)
            .zip(actual)
            .zip(decoded.into_iter().zip(decoded_actual))
        {
            let expected = expected?;
            let actual = token_ids(actual?)?;
            let decoded = decoded?;
            // Python's original CFFI wrapper explicitly uses UTF-8 errors=replace.
            let decoded_bytes = unhex(decoded_actual?)?;
            let decoded_actual = String::from_utf8_lossy(&decoded_bytes);
            self.compared += 1;
            self.input_hash
                .update(u64::try_from(input.len())?.to_le_bytes());
            self.input_hash.update(input.as_bytes());
            let encode_differs = expected != actual;
            let decode_differs = decoded != decoded_actual;
            self.encode_differences += usize::from(encode_differs);
            self.decode_differences += usize::from(decode_differs);
            let source_decode_error =
                !source_decode_matches(input, &decoded, &decoded_actual, vocab);
            let record_encode = encode_differs && self.encode_errors < MAX_ERRORS;
            let record_decode = source_decode_error && self.decode_errors < MAX_ERRORS;
            self.encode_errors += usize::from(record_encode);
            self.decode_errors += usize::from(record_decode);
            if record_encode || record_decode {
                let first_token = expected
                    .iter()
                    .zip(&actual)
                    .position(|(a, b)| a != b)
                    .unwrap_or_else(|| expected.len().min(actual.len()));
                let start = first_token.saturating_sub(2);
                let end_expected = (first_token + 6).min(expected.len());
                let end_actual = (first_token + 6).min(actual.len());
                self.diagnostics.push(json!({"ComparedIndex":self.compared-1,"EncodeError":record_encode,"DecodeError":record_decode,"ExpectedTokenWindow":expected.get(start..end_expected).unwrap_or(&[]),"ActualTokenWindow":actual.get(start..end_actual).unwrap_or(&[]),"InputSha256":super::lora_inputs::hash_bytes(input.as_bytes()),"ExpectedTextPrefix":decoded.chars().take(32).collect::<String>(),"ActualTextPrefix":decoded_actual.chars().take(32).collect::<String>()}));
            }
            if self.stopped() {
                break;
            }
        }
        Ok(!self.stopped())
    }
    fn report(&self, kind: Kind, elapsed_ns: u128) -> Result<Value> {
        Ok(
            json!({"Generator":generator_name(kind),"Compared":self.compared,"Prefetched":self.prefetched,"UncomparedPrefetch":self.prefetched-self.compared,"SourceEarlyStop":self.stopped(),"GeneratorExhausted":!self.stopped(),"SourceAssertions":0,"SourceEncodeErrors":self.encode_errors,"SourceDecodeErrors":self.decode_errors,"EncodeDifferences":self.encode_differences,"StrictDecodeDifferences":self.decode_differences,"StrictReferenceMatch":self.encode_differences==0&&self.decode_differences==0,"ComparedInputSha256":hex(&self.input_hash.clone().finalize())?,"DiagnosticTimingsNs":{"ReferenceEncode":self.reference_encode_ns.to_string(),"ProductionEncode":self.production_encode_ns.to_string(),"ReferenceDecode":self.reference_decode_ns.to_string(),"ProductionDecode":self.production_decode_ns.to_string(),"Total":elapsed_ns.to_string()},"Diagnostics":self.diagnostics}),
        )
    }
}
fn compare(
    session: &mut VocabularySession<'_>,
    reference: &Reference,
    vocab: &Vocabulary,
    kind: Kind,
    iterations: usize,
) -> Result<Value> {
    let start = Instant::now();
    let mut comparison = Comparison::default();
    let mut pending = Vec::with_capacity(BATCH);
    generators::visit(kind, vocab, iterations, |input| {
        pending.push(input);
        if pending.len() < BATCH {
            return Ok(true);
        }
        let keep_going = comparison.batch(session, reference, vocab, &pending)?;
        pending.clear();
        Ok(keep_going)
    })?;
    if !pending.is_empty() {
        comparison.batch(session, reference, vocab, &pending)?;
    }
    comparison.report(kind, start.elapsed().as_nanos())
}
pub(super) fn run_case(
    engine: &Engine,
    name: &str,
    vocabulary_file: &Path,
    reference_directory: &Path,
    options: &Options,
) -> Result<Option<Value>> {
    require_ci()?;
    if options.generators.is_empty()
        || options.generators.len() > 12
        || options.random_iterations > 10000
    {
        return Err(invalid("Tokenizer generator options outside source bounds"));
    }
    let reference = Reference::open(reference_directory)?;
    let vocab = reference.vocabulary()?;
    let Some(mut session) = VocabularySession::open(engine, name, vocabulary_file)? else {
        return Ok(None);
    };
    let mut groups = Vec::new();
    for kind in &options.generators {
        groups.push(compare(
            &mut session,
            &reference,
            &vocab,
            *kind,
            options.random_iterations,
        )?);
    }
    let leases = session.finish()?;
    let strict = groups
        .iter()
        .all(|group| group.get("StrictReferenceMatch").and_then(Value::as_bool) == Some(true));
    Ok(Some(
        json!({"Name":name,"ReferenceClass":reference.class,"ReferenceSourceCorrection":reference.source_correction,"SourceAssertions":0,"SetupPredicates":1,"Completed":true,"StrictReferenceMatch":strict,"ProcessLeases":leases,"Groups":groups}),
    ))
}

fn ordered_cases<T: Send>(count: usize, run: impl Fn(usize) -> T + Sync) -> Result<Vec<T>> {
    let next = AtomicUsize::new(0);
    thread::scope(|scope| {
        let mut workers = Vec::new();
        for _ in 0..count.min(CASE_WORKERS) {
            workers.push(scope.spawn(|| {
                let mut completed = Vec::new();
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    if index >= count {
                        break;
                    }
                    completed.push((index, run(index)));
                }
                completed
            }));
        }
        // Join every worker before interpreting outcomes, then restore original
        // case order so errors and unsupported verdicts keep their source order.
        let joined = workers
            .into_iter()
            .map(thread::ScopedJoinHandle::join)
            .collect::<Vec<_>>();
        let mut completed = Vec::with_capacity(count);
        for worker in joined {
            completed.extend(worker.map_err(|_| invalid("Tokenizer case worker panicked"))?);
        }
        completed.sort_unstable_by_key(|(index, _)| *index);
        Ok(completed.into_iter().map(|(_, value)| value).collect())
    })
}

pub(super) fn run(engine: &Engine) -> Result<Value> {
    require_ci()?;
    let prepared = super::tokenizer_inputs::prepare(engine)?;
    if prepared.cases.len() != NAMES.len()
        || prepared
            .cases
            .iter()
            .zip(NAMES)
            .any(|(case, name)| case.name != name)
    {
        return Err(invalid(
            "Tokenizer original26 default identities/order changed",
        ));
    }
    let observations = ordered_cases(prepared.cases.len(), |index| {
        let case = &prepared.cases[index];
        run_case(
            engine,
            &case.name,
            &case.vocabulary,
            &case.reference_directory,
            &Options::default(),
        )
    })?;
    let mut cases = Vec::with_capacity(observations.len());
    for observation in observations {
        let Some(observation) = observation? else {
            return Ok(result(
                ID,
                false,
                "production vocabulary/model API is unavailable".to_owned(),
            ));
        };
        cases.push(observation);
    }
    let strict = cases
        .iter()
        .all(|case| case.get("StrictReferenceMatch").and_then(Value::as_bool) == Some(true));
    Ok(result(ID,true,json!({"SourceAssertions":0,"SetupPredicates":26,"WorkloadCompleted":true,"StrictReferenceMatch":strict,"NativeObservationBatch":BATCH,"OriginalMaxErrors":MAX_ERRORS,"OriginalGeneratorsPerCase":5,"OptionalGeneratorsRetained":7,"SourceCorrection":"The stale Python CFFI/header preprocessing is replaced by typed production vocabulary APIs; mismatch diagnostics retain the original non-failing verdict","Inputs":prepared.receipt,"Cases":cases}).to_string()))
}

fn require_ci() -> Result<()> {
    if std::env::var("GITHUB_ACTIONS").as_deref() != Ok("true")
        || std::env::var("GITHUB_REF").as_deref() != Ok("refs/heads/main")
    {
        return Err(invalid(
            "Original exhaustive tokenizer schedules run only on main CI",
        ));
    }
    Ok(())
}

pub(super) fn diagnostic(engine: &Engine, value: &Value) -> Result<Value> {
    super::llama_grammar::exact_fields(value, &["Kind", "Name", "Generators", "RandomIterations"])?;
    require_ci()?;
    let name = value["Name"]
        .as_str()
        .ok_or_else(|| invalid("Tokenizer name must be text"))?;
    let generators = value["Generators"]
        .as_array()
        .ok_or_else(|| invalid("Generators must be an array"))?
        .iter()
        .map(|value| match value.as_str() {
            Some("generator_custom_text") => Ok(Kind::Custom),
            Some("generator_custom_text_edge_cases") => Ok(Kind::Edges),
            Some("generator_ascii_lr_strip") => Ok(Kind::Ascii),
            Some("generator_apostrophe") => Ok(Kind::Apostrophe),
            Some("generator_unicodes") => Ok(Kind::Unicode),
            Some("generator_vocab_words") => Ok(Kind::Vocab),
            Some("generator_added_lr_strip") => Ok(Kind::Added),
            Some("generator_random_added_tokens") => Ok(Kind::RandomAdded),
            Some("generator_random_chars") => Ok(Kind::RandomChars),
            Some("generator_random_unicodes") => Ok(Kind::RandomUnicode),
            Some("generator_random_vocab_chars") => Ok(Kind::RandomVocabChars),
            Some("generator_random_vocab_words") => Ok(Kind::RandomVocabWords),
            _ => Err(invalid("Unknown original tokenizer generator")),
        })
        .collect::<Result<Vec<_>>>()?;
    let options = Options {
        generators,
        random_iterations: usize::try_from(
            value["RandomIterations"]
                .as_u64()
                .ok_or_else(|| invalid("RandomIterations must be unsigned"))?,
        )?,
    };
    let prepared = super::tokenizer_inputs::prepare(engine)?;
    let case = prepared
        .cases
        .iter()
        .find(|case| case.name == name)
        .ok_or_else(|| invalid("Unknown original tokenizer family"))?;
    let observation = run_case(
        engine,
        name,
        &case.vocabulary,
        &case.reference_directory,
        &options,
    )?;
    Ok(
        json!({"Diagnostic":"tokenizer-random-options","CorrectnessSourceCount":0,"Status":if observation.is_some(){"completed"}else{"unsupported"},"Observation":observation}),
    )
}

#[cfg(test)]
mod tests {
    use super::{CASE_WORKERS, NAMES, Result, invalid, ordered_cases};
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::thread;

    #[test]
    fn whole_case_workers_preserve_order_and_bound_live_sessions() -> Result<()> {
        let first_pair = Barrier::new(CASE_WORKERS);
        let second_completed = AtomicBool::new(false);
        let active = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let visits = NAMES.map(|_| AtomicUsize::new(0));
        let cases = ordered_cases(NAMES.len(), |index| {
            let live = active.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(live, Ordering::SeqCst);
            if index < CASE_WORKERS {
                first_pair.wait();
                if index == 0 {
                    while !second_completed.load(Ordering::Acquire) {
                        thread::yield_now();
                    }
                } else {
                    second_completed.store(true, Ordering::Release);
                }
            }
            visits[index].fetch_add(1, Ordering::Relaxed);
            active.fetch_sub(1, Ordering::SeqCst);
            NAMES[index]
        })?;
        assert_eq!(cases, NAMES);
        assert_eq!(peak.load(Ordering::SeqCst), CASE_WORKERS);
        assert_eq!(active.load(Ordering::SeqCst), 0);
        assert!(
            visits
                .iter()
                .all(|count| count.load(Ordering::Relaxed) == 1)
        );
        Ok(())
    }

    #[test]
    fn whole_case_workers_retain_original_error_and_unsupported_order() -> Result<()> {
        let cases = ordered_cases(3, |index| match index {
            0 => Err(invalid("first source error")),
            1 => Err(invalid("second source error")),
            _ => Ok(Some(index)),
        })?;
        let error = cases
            .into_iter()
            .collect::<Result<Vec<_>>>()
            .err()
            .ok_or_else(|| invalid("Tokenizer case errors were lost"))?;
        assert_eq!(error.to_string(), "first source error");
        let cases = ordered_cases(3, |index| if index == 1 { None } else { Some(index) })?;
        assert_eq!(cases, [Some(0), None, Some(2)]);
        assert_eq!(ordered_cases(1, |index| index)?, [0]);
        assert_eq!(ordered_cases(0, |index| index)?, [0_usize; 0]);
        Ok(())
    }
}
