//! Complete pinned `LLGuidance` grammar/schema and sampler-chain contracts.
//! MIT attribution: tools/prism/upstream/LICENSE. All match decisions are Rust.

use super::engine::Engine;
use super::llama_grammar::{array, exact_fields, field, uint};
use super::model_session::{Client, ModelSession};
use super::{Result, invalid, result};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

const ID: &str = "tests/test-grammar-llguidance.cpp";
const SHA: &str = "050d42e1577bdb537bd8c133e3a71354d5b64a9ea5cc8606962531095e28d56f";
const VOCAB_SHA: &str = "97272e430d53bc7688f52d5e0ad8ea8f163ede9f1bbd1694feaa504797d5d96e";
const CHAIN_GRAMMAR: &str = "%llguidance {}\nstart: /[A-Z ]*/";
const CHAIN_INPUT: &str = "ALL YOUR BASE ARE BELONG TO US";

fn text<'a>(v: &'a Value, name: &str) -> Result<&'a str> {
    field(v, name)?
        .as_str()
        .ok_or_else(|| invalid(format!("LLG {name} must be text")))
}
fn hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .flat_map(|b| [b >> 4, b & 15])
        .map(|n| char::from(b"0123456789abcdef"[usize::from(n)]))
        .collect()
}
fn call(client: &Client, value: Value) -> Result<Value> {
    let reply = client.call(0, value, None)?;
    if !reply.progress.is_empty() {
        return Err(invalid("LLG unexpected progress callback"));
    }
    Ok(reply.value)
}
fn null(client: &Client, value: Value) -> Result<()> {
    if !call(client, value)?.is_null() {
        return Err(invalid("LLG void response differs"));
    }
    Ok(())
}
fn handle(client: &Client, operation: Value, name: &str) -> Result<u64> {
    let value = call(client, operation)?;
    exact_fields(&value, &[name])?;
    uint(&value, name)
}
fn fixtures() -> Result<Value> {
    let v: Value = serde_json::from_str(include_str!("llguidance-cases.json"))?;
    exact_fields(&v, &["Source", "SourceSha256", "Purpose", "Cases"])?;
    let cases = array(&v, "Cases")?;
    if text(&v, "Source")? != ID || text(&v, "SourceSha256")? != SHA || cases.len() != 42 {
        return Err(invalid("LLG source fixture identity or extent differs"));
    }
    let mut predicates = 0;
    for case in cases {
        exact_fields(
            case,
            &[
                "Group",
                "Description",
                "Kind",
                "Literal",
                "Passing",
                "Failing",
            ],
        )?;
        for key in ["Group", "Description", "Literal"] {
            text(case, key)?;
        }
        if !matches!(text(case, "Kind")?, "grammar" | "schema") {
            return Err(invalid("LLG fixture kind unknown"));
        }
        for key in ["Passing", "Failing"] {
            for input in array(case, key)? {
                input
                    .as_str()
                    .ok_or_else(|| invalid("LLG literal input must be text"))?;
                predicates += 1;
            }
        }
    }
    if predicates != 344 {
        return Err(invalid("LLG complete literal predicate extent differs"));
    }
    Ok(v)
}
fn verify_vocab(engine: &Engine) -> Result<std::path::PathBuf> {
    let path = engine.source_path("models/ggml-vocab-llama-bpe.gguf")?;
    if fs::metadata(&path)?.len() != 7_818_140 {
        return Err(invalid("LLG pinned vocabulary length differs"));
    }
    let digest = Sha256::digest(fs::read(&path)?);
    if hex(&digest) != VOCAB_SHA {
        return Err(invalid("LLG pinned vocabulary digest differs"));
    }
    Ok(path)
}
struct Loaded {
    model: u64,
    context: u64,
    count: usize,
    eos: i32,
}
fn load(client: &Client, path: &Path) -> Result<Loaded> {
    null(client, json!({"Op":"backend-init"}))?;
    let model = handle(
        client,
        json!({"Op":"model-load","Path":path,"Base":"default","Params":null,"Overrides":{"VocabOnly":true},"Progress":false}),
        "Model",
    )?;
    let context = handle(
        client,
        json!({"Op":"context","Model":model,"Params":null,"Overrides":{}}),
        "Context",
    )?;
    let vocab = call(client, json!({"Op":"llg-vocabulary","Model":model}))?;
    exact_fields(&vocab, &["Count", "Eot", "Eos"])?;
    let count = usize::try_from(uint(&vocab, "Count")?)?;
    let token = |name: &str| -> Result<i32> {
        i32::try_from(
            field(&vocab, name)?
                .as_i64()
                .ok_or_else(|| invalid("LLG end-token must be signed integer"))?,
        )
        .map_err(Into::into)
    };
    let eot = token("Eot")?;
    let eos = if eot == -1 { token("Eos")? } else { eot };
    if count == 0 || count > 1_048_576 || eos < 0 || usize::try_from(eos)? >= count {
        return Err(invalid("LLG vocabulary extent/end-token invalid"));
    }
    Ok(Loaded {
        model,
        context,
        count,
        eos,
    })
}
fn cleanup(client: &Client, loaded: &Loaded) -> Result<()> {
    for (kind, id) in [("context", loaded.context), ("model", loaded.model)] {
        null(client, json!({"Op":"release","Kind":kind,"Handle":id}))?;
    }
    null(client, json!({"Op":"backend-free"}))
}
fn tokenize(client: &Client, context: u64, input: &str, count: usize) -> Result<Vec<i32>> {
    let v = call(
        client,
        json!({"Op":"vocab-tokenize","Context":context,"InputsHex":[hex(input.as_bytes())],"AddSpecial":false,"ParseSpecial":false}),
    )?;
    exact_fields(&v, &["Values", "Errors"])?;
    let values = array(&v, "Values")?;
    let errors = array(&v, "Errors")?;
    if values.len() != 1 || errors.len() != 1 || !errors[0].is_null() {
        return Err(invalid("LLG source tokenization failed"));
    }
    values[0]
        .as_array()
        .ok_or_else(|| invalid("LLG tokens must be array"))?
        .iter()
        .map(|v| {
            let token = i32::try_from(
                v.as_i64()
                    .ok_or_else(|| invalid("LLG token must be integer"))?,
            )?;
            if token < 0 || usize::try_from(token)? >= count {
                return Err(invalid("LLG token outside vocabulary"));
            }
            Ok(token)
        })
        .collect()
}
fn candidate_bytes(count: usize) -> Result<Vec<u8>> {
    let mut full = Vec::with_capacity(count * 12);
    for index in 0..count {
        let id = i32::try_from(index)?.to_le_bytes();
        full.extend(id);
        full.extend(0_u32.to_le_bytes());
        full.extend(0_u32.to_le_bytes());
    }
    Ok(full)
}
fn candidate_template(client: &Client, directory: &Path, count: usize) -> Result<u64> {
    let full = directory.join("candidates.bin");
    fs::write(&full, candidate_bytes(count)?)?;
    // Retain the original caller-supplied fields once. Every reset still copies
    // all vocabulary entries, preserving exactly the fields each source loop
    // assigns, without repeating the file reads and byte decoding.
    handle(
        client,
        json!({"Op":"llg-candidates-new","Count":count,"File":full,"Selected":-1,"Sorted":false}),
        "Candidates",
    )
}
fn new_candidates(client: &Client, template: u64) -> Result<u64> {
    handle(
        client,
        json!({"Op":"llg-candidates-clone","Source":template}),
        "Candidates",
    )
}
fn patch(client: &Client, candidates: u64, index: i32, bits: u32) -> Result<()> {
    null(
        client,
        json!({"Op":"llg-candidates-patch","Candidates":candidates,"Patches":[{"Index":index,"Fields":{"LogitBits":bits}}]}),
    )
}
fn apply(
    client: &Client,
    sampler: u64,
    candidates: u64,
    probe: Option<i32>,
    count: usize,
) -> Result<Value> {
    let indices: Vec<i32> = probe.into_iter().collect();
    let v = call(
        client,
        json!({"Op":"llg-apply","Sampler":sampler,"Candidates":candidates,"Indices":indices}),
    )?;
    exact_fields(
        &v,
        &[
            "BufferSize",
            "Size",
            "Offset",
            "Selected",
            "Sorted",
            "SelectedBacking",
            "Probes",
        ],
    )?;
    let size = usize::try_from(uint(&v, "Size")?)?;
    let offset = usize::try_from(uint(&v, "Offset")?)?;
    if usize::try_from(uint(&v, "BufferSize")?)? != count
        || offset > count
        || size > count - offset
        || !field(&v, "Sorted")?.is_boolean()
        || !field(&v, "Selected")?.is_i64()
    {
        return Err(invalid("LLG candidate response metadata invalid"));
    }
    let probes = array(&v, "Probes")?;
    if probes.len() != indices.len() {
        return Err(invalid("LLG probe response extent differs"));
    }
    for (row, index) in probes.iter().zip(indices) {
        exact_fields(row, &["Index", "Candidate"])?;
        if uint(row, "Index")? != u64::try_from(index)? {
            return Err(invalid("LLG probe index differs"));
        }
        validate_candidate(field(row, "Candidate")?)?;
    }
    if !v["SelectedBacking"].is_null() {
        validate_candidate(&v["SelectedBacking"])?;
    }
    Ok(v)
}
fn validate_candidate(v: &Value) -> Result<()> {
    exact_fields(v, &["Id", "LogitBits", "ProbabilityBits"])?;
    i32::try_from(
        field(v, "Id")?
            .as_i64()
            .ok_or_else(|| invalid("LLG candidate id invalid"))?,
    )?;
    u32::try_from(uint(v, "LogitBits")?)?;
    u32::try_from(uint(v, "ProbabilityBits")?)?;
    Ok(())
}
fn probed_logit(v: &Value) -> Result<f32> {
    let rows = array(v, "Probes")?;
    if rows.len() != 1 {
        return Err(invalid("LLG single probe missing"));
    }
    Ok(f32::from_bits(u32::try_from(uint(
        field(&rows[0], "Candidate")?,
        "LogitBits",
    )?)?))
}
fn matches(
    client: &Client,
    loaded: &Loaded,
    template: u64,
    sampler: u64,
    input: &str,
) -> Result<bool> {
    null(client, json!({"Op":"llg-reset","Sampler":sampler}))?;
    let tokens = tokenize(client, loaded.context, input, loaded.count)?;
    let candidates = new_candidates(client, template)?;
    for token in tokens {
        null(
            client,
            json!({"Op":"llg-candidates-copy","Candidates":candidates,"Source":template,"Encoding":"logits-le","Metadata":null}),
        )?;
        let observed = apply(client, sampler, candidates, Some(token), loaded.count)?;
        if probed_logit(&observed)? < 0.0 {
            null(
                client,
                json!({"Op":"llg-candidates-free","Candidates":candidates}),
            )?;
            return Ok(false);
        }
        null(
            client,
            json!({"Op":"llg-accept","Sampler":sampler,"Token":token}),
        )?;
    }
    patch(client, candidates, loaded.eos, 0)?;
    let observed = apply(client, sampler, candidates, Some(loaded.eos), loaded.count)?;
    null(
        client,
        json!({"Op":"llg-candidates-free","Candidates":candidates}),
    )?;
    Ok(probed_logit(&observed)? >= 0.0)
}
fn selected_id(v: &Value) -> Result<i32> {
    let selected = field(v, "SelectedBacking")?;
    validate_candidate(selected)?;
    i32::try_from(
        field(selected, "Id")?
            .as_i64()
            .ok_or_else(|| invalid("LLG selected id invalid"))?,
    )
    .map_err(Into::into)
}
fn debug_text(value: &str) -> Vec<u8> {
    // Original diagnostics use fprintf("%s") on a Windows text-mode stream.
    let mut bytes = Vec::new();
    for byte in value.split('\0').next().unwrap_or_default().bytes() {
        if byte == b'\n' {
            bytes.push(b'\r');
        }
        bytes.push(byte);
    }
    bytes
}
fn chain(client: &Client, loaded: &Loaded, template: u64) -> Result<(usize, Vec<String>)> {
    let sampler = handle(
        client,
        json!({"Op":"llg-chain-new","NoPerf":false}),
        "Sampler",
    )?;
    let grammar = handle(
        client,
        json!({"Op":"llg-new","Model":loaded.model,"GrammarKind":"lark","GrammarData":CHAIN_GRAMMAR}),
        "Sampler",
    )?;
    null(
        client,
        json!({"Op":"llg-chain-add","Chain":sampler,"Sampler":grammar}),
    )?;
    let dist = handle(client, json!({"Op":"llg-dist-new","Seed":42}), "Sampler")?;
    null(
        client,
        json!({"Op":"llg-chain-add","Chain":sampler,"Sampler":dist}),
    )?;
    let tokens = tokenize(client, loaded.context, CHAIN_INPUT, loaded.count)?;
    let candidates = new_candidates(client, template)?;
    let mut checks = 0;
    let mut failures = Vec::new();
    for token in tokens.iter().copied().chain(std::iter::once(loaded.eos)) {
        null(
            client,
            json!({"Op":"llg-candidates-copy","Candidates":candidates,"Source":template,"Encoding":"id-logit-le","Metadata":{"Offset":0,"Size":loaded.count,"Selected":-1,"Sorted":false}}),
        )?;
        patch(client, candidates, token, 100.0_f32.to_bits())?;
        let value = apply(client, sampler, candidates, None, loaded.count)?;
        checks += 1;
        if selected_id(&value)? != token {
            failures.push(format!(
                "sampler-chain selected token mismatch at condition {checks}"
            ));
            break;
        }
        // The original final EOS condition is observed but not accepted.
        if checks <= tokens.len() {
            null(
                client,
                json!({"Op":"llg-accept","Sampler":sampler,"Token":token}),
            )?;
        }
    }
    null(
        client,
        json!({"Op":"llg-candidates-free","Candidates":candidates}),
    )?;
    null(client, json!({"Op":"llg-free","Sampler":sampler}))?;
    if failures.is_empty() && checks != tokens.len() + 1 {
        return Err(invalid("LLG chain condition extent differs"));
    }
    Ok((checks, failures))
}
pub(super) fn run(engine: &Engine) -> Result<Value> {
    let fixture = fixtures()?;
    let mut checks = 0;
    let mut chain_checks = 0;
    let mut failures = Vec::new();
    let mut summaries = Vec::new();
    // The source main owns one vocabulary/model/context across every group.
    // The production LLGuidance sampler caches its tokenizer by vocabulary
    // pointer, so retaining that scope also avoids rebuilding the 128K-token
    // index for every grammar. Samplers and candidate vectors remain fresh at
    // exactly the original boundaries; no matching decisions are cached.
    let Some(session) = ModelSession::open(engine, "llguidance")? else {
        return Ok(result(
            ID,
            false,
            "Production LLGuidance/model-session API unavailable; full source unsupported".into(),
        ));
    };
    let client = session.client();
    let available = call(&client, json!({"Op":"llg-available"}))?;
    exact_fields(&available, &["Enabled"])?;
    if field(&available, "Enabled")?.as_bool() != Some(true) {
        session.finish()?;
        return Ok(result(ID,false,"Pinned production LLGuidance feature/dependency is required; disabled builds do not count as passing coverage".into()));
    }
    let loaded = load(&client, &verify_vocab(engine)?)?;
    let template = candidate_template(&client, session.evidence_directory(), loaded.count)?;
    for group in 0..=42 {
        if group == 42 {
            let (count, failed) = chain(&client, &loaded, template)?;
            chain_checks = count;
            failures.extend(failed);
        } else {
            let case = &array(&fixture, "Cases")?[group];
            let literal = text(case, "Literal")?;
            let grammar = if text(case, "Kind")? == "schema" {
                format!("%llguidance {{}}\nstart: %json {literal}")
            } else {
                literal.to_owned()
            };
            let sampler = handle(
                &client,
                json!({"Op":"llg-new","Model":loaded.model,"GrammarKind":"lark","GrammarData":grammar}),
                "Sampler",
            )?;
            let before = checks;
            for (name, expected) in [("Passing", true), ("Failing", false)] {
                for input in array(case, name)? {
                    let input = input
                        .as_str()
                        .ok_or_else(|| invalid("LLG input must be text"))?;
                    let matched = matches(&client, &loaded, template, sampler, input)?;
                    checks += 1;
                    if matched != expected {
                        if expected {
                            // fopen failures are also nonfatal in the source.
                            let _ = fs::write(
                                session
                                    .evidence_directory()
                                    .join("test-grammar-integration.grammar.gbnf"),
                                debug_text(&grammar),
                            );
                            let _ = fs::write(
                                session
                                    .evidence_directory()
                                    .join("test-grammar-integration.string.txt"),
                                debug_text(input),
                            );
                        }
                        failures.push(format!("group {group} {} predicate {checks}: expected {expected}, observed {matched}",text(case,"Description")?));
                    }
                }
            }
            null(&client, json!({"Op":"llg-free","Sampler":sampler}))?;
            summaries.push(
                json!({"Group":group,"Description":case["Description"],"Predicates":checks-before}),
            );
        }
    }
    null(
        &client,
        json!({"Op":"llg-candidates-free","Candidates":template}),
    )?;
    cleanup(&client, &loaded)?;
    session.finish()?;
    if failures.is_empty() && (checks != 344 || chain_checks == 0 || summaries.len() != 42) {
        return Err(invalid("LLG full source condition extent differs"));
    }
    let mut output = result(
        ID,
        failures.is_empty() && checks == 344 && chain_checks > 0,
        format!(
            "{checks}/344 literal acceptance predicates; {chain_checks} exact token/EOS chain conditions; failures={failures:?}"
        ),
    );
    output["Groups"] = json!(summaries);
    output["LiteralPredicates"] = json!(checks);
    output["SamplerChainPredicates"] = json!(chain_checks);
    output["DependencyRevision"] = json!("d795912fedc7d393de740177ea9ea761e7905774");
    output["VocabularySha256"] = json!(VOCAB_SHA);
    output["ModelLoads"] = json!(1);
    output["VocabularyScope"] = json!(
        "One source model/context across all 42 grammar groups and the sampler chain; fresh source samplers and candidate vectors, with the production vocabulary tokenizer cache retained"
    );
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::{candidate_bytes, fixtures, probed_logit, validate_candidate};
    use serde_json::json;

    #[test]
    #[ignore = "requires BONSAI_BACKEND_TRANSPORT_BRIDGE naming the checked native common bridge"]
    #[allow(
        clippy::too_many_lines,
        reason = "Keeps the complete native field-import equivalence and malformed-input schedule together."
    )]
    fn native_candidate_templates_match_original_file_imports() -> super::Result<()> {
        use super::super::engine::{Engine, Target};
        use super::{ModelSession, call, handle, null, patch};
        use std::{fs, path::PathBuf, time::SystemTime};

        let bridge = PathBuf::from(std::env::var("BONSAI_BACKEND_TRANSPORT_BRIDGE")?);
        let evidence = std::env::temp_dir().join(format!(
            "prism-llg-candidate-equivalence-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        let engine = Engine::new(Target::Cpp, None, Some(bridge), Some(evidence), None, None)?;
        let session = ModelSession::open(&engine, "llguidance-candidate-equivalence")?
            .ok_or_else(|| super::invalid("native candidate session unavailable"))?;
        let client = session.client();
        let count = 257;
        let patterns = [
            0_u32,
            0x8000_0000,
            0x7f80_0000,
            0xff80_0000,
            0x7fc1_2345,
            0x7fa1_2345,
            0x3f80_0000,
            0xbf80_0000,
        ];
        let mut source = Vec::new();
        let mut initial = Vec::new();
        let mut logits = Vec::new();
        let mut id_logits = Vec::new();
        for index in 0..count {
            let id = i32::try_from(count - index)?.to_le_bytes();
            let logit = patterns[index % patterns.len()].to_le_bytes();
            let probability = patterns[(index + 3) % patterns.len()].to_le_bytes();
            source.extend([id, logit, probability].concat());
            logits.extend(logit);
            id_logits.extend([id, logit].concat());
            initial.extend(
                [
                    i32::try_from(index)?.to_le_bytes(),
                    patterns[(index + 1) % patterns.len()].to_le_bytes(),
                    patterns[(index + 2) % patterns.len()].to_le_bytes(),
                ]
                .concat(),
            );
        }
        let directory = session.evidence_directory();
        let source_file = directory.join("source.bin");
        let initial_file = directory.join("initial.bin");
        fs::write(&source_file, &source)?;
        fs::write(&initial_file, &initial)?;
        let new = |file: &std::path::Path, count| {
            handle(
                &client,
                json!({"Op":"llg-candidates-new","Count":count,"File":file,"Selected":-1,"Sorted":false}),
                "Candidates",
            )
        };
        let inspect = |candidates| {
            call(
                &client,
                json!({"Op":"llg-candidates-inspect","Candidates":candidates,"Indices":(0..count).collect::<Vec<_>>()}),
            )
        };
        let template = new(&source_file, count)?;
        let untouched = inspect(template)?;
        for (encoding, bytes) in [
            ("candidate-le", &source),
            ("id-logit-le", &id_logits),
            ("logits-le", &logits),
        ] {
            let file = directory.join(format!("{encoding}.bin"));
            fs::write(&file, bytes)?;
            let expected = new(&initial_file, count)?;
            let actual = new(&initial_file, count)?;
            // Preserve a partial view and non-default selected/sorted metadata.
            // A field-only reset must leave these and untouched P/ID bits intact.
            for candidates in [expected, actual] {
                null(
                    &client,
                    json!({"Op":"llg-candidates-load","Candidates":candidates,"File":initial_file,"Encoding":"candidate-le","Metadata":{"Offset":2,"Size":count-4,"Selected":5,"Sorted":true}}),
                )?;
            }
            null(
                &client,
                json!({"Op":"llg-candidates-load","Candidates":expected,"File":file,"Encoding":encoding,"Metadata":null}),
            )?;
            null(
                &client,
                json!({"Op":"llg-candidates-copy","Candidates":actual,"Source":template,"Encoding":encoding,"Metadata":null}),
            )?;
            let observed = inspect(actual)?;
            assert_eq!(observed, inspect(expected)?);
            assert_eq!(inspect(template)?, untouched);
            for (index, row) in super::array(&observed, "Probes")?.iter().enumerate() {
                let value = &row["Candidate"];
                assert_eq!(
                    value["Id"],
                    json!(if encoding == "logits-le" {
                        index
                    } else {
                        count - index
                    })
                );
                assert_eq!(value["LogitBits"], json!(patterns[index % patterns.len()]));
                assert_eq!(
                    value["ProbabilityBits"],
                    json!(
                        patterns[(index + if encoding == "candidate-le" { 3 } else { 2 })
                            % patterns.len()]
                    )
                );
            }
            let cloned = handle(
                &client,
                json!({"Op":"llg-candidates-clone","Source":actual}),
                "Candidates",
            )?;
            assert_eq!(inspect(cloned)?, observed);
            patch(&client, cloned, 3, 100_f32.to_bits())?;
            assert_eq!(inspect(actual)?, observed);
            assert_ne!(inspect(cloned)?, observed);
            null(
                &client,
                json!({"Op":"llg-candidates-copy","Candidates":actual,"Source":template,"Encoding":encoding,"Metadata":{"Offset":0,"Size":count,"Selected":-1,"Sorted":false}}),
            )?;
            let reset = inspect(actual)?;
            assert_eq!(reset["Offset"], 0);
            assert_eq!(reset["Size"], count);
            assert_eq!(reset["Selected"], -1);
            assert_eq!(reset["Sorted"], false);
            assert!(call(&client, json!({"Op":"llg-candidates-copy","Candidates":actual,"Source":template,"Encoding":"invalid","Metadata":null})).is_err());
            assert!(call(&client, json!({"Op":"llg-candidates-copy","Candidates":actual,"Source":template,"Encoding":encoding,"Metadata":{"Offset":count,"Size":1,"Selected":-1,"Sorted":false}})).is_err());
            for candidates in [expected, actual, cloned] {
                null(
                    &client,
                    json!({"Op":"llg-candidates-free","Candidates":candidates}),
                )?;
            }
        }
        let short_file = directory.join("short.bin");
        fs::write(&short_file, &source[..12])?;
        assert!(new(&short_file, count).is_err());
        let short = new(&short_file, 1)?;
        assert!(call(&client, json!({"Op":"llg-candidates-copy","Candidates":short,"Source":template,"Encoding":"logits-le","Metadata":null})).is_err());
        let changed = new(&initial_file, count)?;
        patch(&client, template, 7, 0x7fc0_dead)?;
        null(
            &client,
            json!({"Op":"llg-candidates-copy","Candidates":changed,"Source":template,"Encoding":"candidate-le","Metadata":null}),
        )?;
        assert_eq!(inspect(changed)?, inspect(template)?);
        assert_eq!(
            inspect(changed)?["Probes"][7]["Candidate"]["LogitBits"],
            0x7fc0_dead_u32
        );
        assert!(
            call(
                &client,
                json!({"Op":"llg-candidates-clone","Source":999_999})
            )
            .is_err()
        );
        for candidates in [short, template, changed] {
            null(
                &client,
                json!({"Op":"llg-candidates-free","Candidates":candidates}),
            )?;
        }
        session.finish()
    }

    #[test]
    fn retained_template_contains_every_original_reset_field() -> super::Result<()> {
        // Initial candidates assign all three fields. The literal matcher then
        // restores only logits; the chain restores IDs and logits, retaining P.
        // All three source byte streams must be exact projections of this one
        // immutable, Rust-supplied template, including the entire BPE vocabulary.
        for count in [1, 257, 128_256] {
            let bytes = candidate_bytes(count)?;
            assert_eq!(bytes.len(), count * 12);
            for (index, row) in bytes.as_chunks::<12>().0.iter().enumerate() {
                assert_eq!(&row[..4], &i32::try_from(index)?.to_le_bytes());
                assert_eq!(&row[4..8], &0_f32.to_le_bytes());
                assert_eq!(&row[8..], &0_f32.to_le_bytes());
            }
        }
        // The fixture validator independently pins all 42 original groups and
        // all 344 passing/failing predicates, beyond the sampler-chain workload.
        let fixture = fixtures()?;
        let mut literal_bytes = 0;
        for case in super::array(&fixture, "Cases")? {
            for key in ["Passing", "Failing"] {
                for input in super::array(case, key)? {
                    literal_bytes += input
                        .as_str()
                        .ok_or_else(|| super::invalid("LLG fixture input is not text"))?
                        .len();
                }
            }
        }
        assert_eq!(literal_bytes, 4430);
        // Even a byte-fallback token for every byte requires fewer than 16K
        // primitives including reset/tokenize/clone/EOS/free, group sampler
        // construction/destruction and the chain. The bounded shared process
        // retains its ordinary 65,536-request and 64MB stream limits.
        assert!(3 * literal_bytes + 7 * 344 + 2 * 42 + 100 < 16_000);
        Ok(())
    }

    #[test]
    fn raw_float_probes_preserve_sign_and_reject_overflow() -> super::Result<()> {
        for value in [f32::NEG_INFINITY, -0.0, 0.0, f32::NAN] {
            let row = json!({"Probes":[{"Candidate":{"Id":1,"LogitBits":value.to_bits(),"ProbabilityBits":0}}]});
            assert_eq!(probed_logit(&row)?.to_bits(), value.to_bits());
        }
        assert!(
            validate_candidate(&json!({"Id":1,"LogitBits":0,"ProbabilityBits":4_294_967_296_u64}))
                .is_err()
        );
        assert!(
            validate_candidate(
                &json!({"Id":1,"LogitBits":0,"ProbabilityBits":0,"Unexpected":true})
            )
            .is_err()
        );
        Ok(())
    }
}
