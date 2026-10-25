//! Complete Windows source contracts for test-arg-parser.cpp.
//!
//! Every argument, expected value and option-inventory check belongs to Rust;
//! the adapter only invokes production parser/speculative/download APIs.
//! Pinned source b87ca7225456b06b0975b8239c66b5aecf07e37c (MIT).

use super::engine::Engine;
use super::{Result, invalid, result};
use serde_json::{Value, json};
use std::collections::BTreeSet;

const ID: &str = "tests/test-arg-parser.cpp";
const SOURCE_SHA: &str = "935b2cd08b63799a26fb804991380c541a45406c76dceb38589d78c8b28befc1";
const GOOD_URL: &str = "http://ggml.ai/";
const BAD_URL: &str = "http://ggml.ai/404";
const COMMON: u32 = 2;
const SPECULATIVE: u32 = 3;
const EMBEDDING: u32 = 6;
const EXAMPLES: u32 = 27;

#[derive(Default)]
struct Checks {
    source: u64,
    failed: Vec<String>,
}
impl Checks {
    fn check(&mut self, passed: bool, message: impl Into<String>) {
        self.source += 1;
        if !passed && self.failed.len() < 32 {
            self.failed.push(message.into());
        }
    }
}
enum Contract {
    Equal(&'static str, Value),
    Greater(&'static str, i64),
    Length(&'static str, usize),
}
impl Contract {
    fn check(&self, value: &Value) -> Result<bool> {
        let field = match self {
            Self::Equal(field, _) | Self::Greater(field, _) | Self::Length(field, _) => field,
        };
        let mut actual = value;
        for part in field.split('.') {
            actual = part
                .parse::<usize>()
                .map_or_else(|_| actual.get(part), |index| actual.get(index))
                .ok_or_else(|| invalid(format!("missing observed argument field {field}")))?;
        }
        Ok(match self {
            Self::Equal(_, expected) => actual == expected,
            Self::Greater(_, minimum) => actual.as_i64().is_some_and(|n| n > *minimum),
            Self::Length(_, count) => actual.as_array().is_some_and(|v| v.len() == *count),
        })
    }
}
struct Step {
    operation: Value,
    contracts: Vec<Contract>,
}
const fn step(operation: Value, contracts: Vec<Contract>) -> Step {
    Step {
        operation,
        contracts,
    }
}
fn parse(
    slot: u32,
    example: u32,
    arguments: &[&str],
    expected: bool,
    mut fields: Vec<Contract>,
) -> Step {
    fields.insert(0, Contract::Equal("Parsed", json!(expected)));
    step(
        json!({"Action":"parse","Slot":slot,"Example":example,"Arguments":arguments}),
        fields,
    )
}
const fn equal(field: &'static str, value: Value) -> Contract {
    Contract::Equal(field, value)
}
fn source_steps() -> Vec<Step> {
    let mut steps = Vec::new();
    for (batch, parallel, draft, total, per_seq) in [
        (16, 2, 3, 8, 4),
        (16, 2, -1, 2, 1),
        (6, 2, 3, 6, 4),
        (2, 1, 3, 2, 2),
        (i32::MAX, i32::MAX, i32::MAX, i32::MAX, i32::MAX),
    ] {
        steps.push(step(
            json!({"Action":"limits","Batch":batch,"Parallel":parallel,"Draft":draft}),
            vec![
                equal("Total", json!(total)),
                equal("PerSequence", json!(per_seq)),
            ],
        ));
    }
    steps.push(step(
        json!({"Action":"speculative","Slot":2}),
        vec![
            equal("Params.OutputsMax", json!(4)),
            equal("Params.OutputsMaxPerSequence", json!(1)),
        ],
    ));
    for example in 0..EXAMPLES {
        steps.push(step(
            json!({"Action":"inventory","Slot":0,"Example":example}),
            Vec::new(),
        ));
    }
    for args in [
        &["binary_name", "-m"][..],
        &["binary_name", "-ngl", "hello"],
        &["binary_name", "-sm", "hello"],
    ] {
        steps.push(parse(0, COMMON, args, false, Vec::new()));
    }
    steps.push(step(
        json!({"Action":"snapshot","Slot":1}),
        vec![
            equal("Params.PenaltyLast", json!(64)),
            equal("Params.DryPenaltyLast", json!(64)),
        ],
    ));
    for args in [
        &["binary_name", "--repeat-last-n", "-1"][..],
        &["binary_name", "--dry-penalty-last-n", "-1"],
        &["binary_name", "--repeat-penalty", "0"],
        &["binary_name", "--repeat-penalty", "-1"],
        &["binary_name", "--repeat-penalty", "nan"],
        &["binary_name", "--repeat-penalty", "inf"],
        &["binary_name", "--repeat-penalty", "-inf"],
    ] {
        steps.push(parse(1, COMMON, args, false, Vec::new()));
    }
    for option in ["--frequency-penalty", "--presence-penalty"] {
        for value in ["nan", "inf", "-inf"] {
            steps.push(parse(
                1,
                COMMON,
                &["binary_name", option, value],
                false,
                Vec::new(),
            ));
        }
    }
    steps.push(parse(
        0,
        EMBEDDING,
        &["binary_name", "--draft", "123"],
        false,
        Vec::new(),
    ));
    steps.push(parse(
        0,
        COMMON,
        &["binary_name", "-lm", "hello"],
        false,
        Vec::new(),
    ));
    steps.extend(valid_steps());
    steps
}
fn valid_steps() -> Vec<Step> {
    let mut steps = vec![
        parse(
            0,
            COMMON,
            &["binary_name", "-m", "model_file.gguf"],
            true,
            vec![equal("Params.ModelPath", json!("model_file.gguf"))],
        ),
        parse(
            0,
            COMMON,
            &["binary_name", "-t", "1234"],
            true,
            vec![equal("Params.Threads", json!(1234))],
        ),
        parse(
            0,
            COMMON,
            &["binary_name", "--verbose"],
            true,
            vec![Contract::Greater("Params.Verbosity", 1)],
        ),
        parse(
            0,
            COMMON,
            &[
                "binary_name",
                "-m",
                "abc.gguf",
                "--predict",
                "6789",
                "--batch-size",
                "9090",
            ],
            true,
            vec![
                equal("Params.ModelPath", json!("abc.gguf")),
                equal("Params.Predict", json!(6789)),
                equal("Params.Batch", json!(9090)),
            ],
        ),
        parse(
            0,
            SPECULATIVE,
            &["binary_name", "--spec-draft-n-max", "123"],
            true,
            vec![equal("Params.DraftMax", json!(123))],
        ),
    ];
    for (name, value) in [
        ("none", 0),
        ("mmap", 1),
        ("mlock", 2),
        ("mmap+mlock", 3),
        ("dio", 4),
    ] {
        steps.push(parse(
            0,
            COMMON,
            &["binary_name", "-lm", name],
            true,
            vec![equal("Params.LoadMode", json!(value))],
        ));
    }
    steps.push(parse(
        0,
        COMMON,
        &[
            "binary_name",
            "--lora",
            r#"file1.gguf,"file2,2.gguf","file3""3"".gguf",file4".gguf"#,
        ],
        true,
        vec![
            Contract::Length("Params.LoraPaths", 4),
            equal("Params.LoraPaths.0", json!("file1.gguf")),
            equal("Params.LoraPaths.1", json!("file2,2.gguf")),
            equal("Params.LoraPaths.2", json!("file3\"3\".gguf")),
            equal("Params.LoraPaths.3", json!("file4\".gguf")),
        ],
    ));
    steps
}
fn verify(passed: bool, message: &str) -> Result<()> {
    if passed {
        Ok(())
    } else {
        Err(invalid(message))
    }
}
fn strings(value: &Value) -> Result<Vec<&str>> {
    value
        .as_array()
        .ok_or_else(|| invalid("missing argument inventory list"))?
        .iter()
        .map(|v| {
            v.as_str()
                .ok_or_else(|| invalid("nonstring argument inventory entry"))
        })
        .collect()
}
fn inventory(value: &Value, checks: &mut Checks) -> Result<usize> {
    let options = value["Options"]
        .as_array()
        .ok_or_else(|| invalid("missing parser option inventory"))?;
    verify(
        !options.is_empty(),
        "production parser option inventory is empty",
    )?;
    let mut args = BTreeSet::new();
    let mut environment = BTreeSet::new();
    for option in options {
        for arg in strings(&option["AllArgs"])? {
            checks.check(
                args.insert(arg),
                format!("duplicate argument handler: {arg}"),
            );
        }
        for env in strings(&option["Environment"])? {
            checks.check(
                environment.insert(env),
                format!("duplicate environment handler: {env}"),
            );
        }
        let speculative = option["Speculative"]
            .as_bool()
            .ok_or_else(|| invalid("missing speculative option flag"))?;
        for (key, skip) in [("Args", speculative), ("NegatedArgs", false)] {
            let ordered = strings(&option[key])?;
            if !skip && ordered.len() > 1 {
                checks.check(
                    ordered[0].len() <= ordered[ordered.len() - 1].len(),
                    format!("shorter argument does not precede longer {key}"),
                );
            }
        }
    }
    Ok(options.len())
}
fn body(value: &Value) -> Result<Option<Vec<u8>>> {
    let Some(hex) = value.as_str() else {
        return Ok(None);
    };
    verify(
        hex.len() % 2 == 0 && hex.len() <= 16 * 1024 * 1024,
        "invalid remote content hex length",
    )?;
    let bytes = hex
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| {
            let nibble = |b: u8| match b {
                b'0'..=b'9' => Ok(b - b'0'),
                b'a'..=b'f' => Ok(b - b'a' + 10),
                _ => Err(invalid("invalid remote content hex digit")),
            };
            Ok(nibble(pair[0])? * 16 + nibble(pair[1])?)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(bytes))
}
fn download_checks(results: &[Value], checks: &mut Checks) -> Result<()> {
    verify(results.len() == 3, "remote content result count changed")?;
    for value in results {
        if !value["Exception"].is_null() {
            verify(
                value["Exception"]["Kind"].as_str() == Some("standard")
                    && value["Exception"]["Message"].is_string(),
                "invalid remote exception observation",
            )?;
        }
    }
    checks.check(
        results[0]["HttpCode"].as_i64() == Some(200),
        "original good URL did not return200",
    );
    let good = body(&results[0]["BodyHex"])?;
    checks.check(
        good.as_ref().is_some_and(|b| !b.is_empty()),
        "original good URL returned no body",
    );
    checks.check(
        good.as_ref()
            .is_some_and(|b| b.windows(b"llama.cpp".len()).any(|w| w == b"llama.cpp")),
        "good URL body omitted llama.cpp",
    );
    checks.check(
        results[1]["HttpCode"].as_i64() == Some(404),
        "original bad URL did not return404",
    );
    checks.check(
        !results[2]["Exception"].is_null(),
        "max_size1 did not throw a production exception",
    );
    Ok(())
}
fn exchange(engine: &Engine, name: &str, operations: Vec<Value>) -> Result<Option<Vec<Value>>> {
    let request = json!({"SchemaVersion":1,"Operations":Value::Array(operations)});
    let Some(response) = engine.execute_common(name, &request)? else {
        return Ok(None);
    };
    verify(
        response["SchemaVersion"] == 1,
        "argument response version changed",
    )?;
    Ok(Some(
        response["Results"]
            .as_array()
            .ok_or_else(|| invalid("missing argument response results"))?
            .clone(),
    ))
}
pub(super) fn run(engine: &Engine) -> Result<Value> {
    let Some(enums) = exchange(
        engine,
        "argument-enums",
        vec![json!({"Kind":"argument-enums"})],
    )?
    else {
        return Ok(result(
            ID,
            false,
            "production argument/parser/download API unavailable".into(),
        ));
    };
    verify(
        enums.len() == 1
            && enums[0]["Count"] == EXAMPLES
            && enums[0]["Common"] == COMMON
            && enums[0]["Speculative"] == SPECULATIVE
            && enums[0]["Embedding"] == EMBEDDING,
        "pinned example enum inventory changed",
    )?;
    let steps = source_steps();
    verify(
        steps
            .iter()
            .filter(|s| s.operation["Action"] == "parse")
            .count()
            == 29,
        "source argv schedule changed",
    )?;
    verify(
        steps.iter().map(|s| s.contracts.len()).sum::<usize>() == 60,
        "fixed source contract inventory changed",
    )?;
    let actions: Vec<_> = steps.iter().map(|s| s.operation.clone()).collect();
    let op = json!({"Kind":"argument-parser-trace","Initials":[{},{},{"Parallel":4,"OutputsMaxPerSequence":8}],"Actions":actions});
    let response = exchange(engine, "argument-original-trace", vec![op])?
        .ok_or_else(|| invalid("production argument API disappeared"))?;
    verify(response.len() == 1, "argument trace result count changed")?;
    let trace = response[0]["Trace"]
        .as_array()
        .ok_or_else(|| invalid("missing argument trace"))?;
    verify(
        trace.len() == steps.len(),
        "argument trace truncated after production exception",
    )?;
    let mut checks = Checks::default();
    let mut inventories = Vec::new();
    for (index, (step, value)) in steps.iter().zip(trace).enumerate() {
        verify(
            value["Exception"].is_null(),
            "production argument operation threw",
        )?;
        if step.operation["Action"] == "inventory" {
            inventories.push(inventory(value, &mut checks)?);
        }
        for contract in &step.contracts {
            checks.check(
                contract.check(value)?,
                format!("original argument predicate failed at action{index}"),
            );
        }
    }
    let operations=[(GOOD_URL,0),(BAD_URL,0),(GOOD_URL,1)].into_iter().map(|(url,max_size)|json!({"Kind":"remote-get-content","Url":url,"Headers":[],"TimeoutSeconds":0,"MaxSize":max_size})).collect();
    let downloads = exchange(engine, "argument-original-downloads", operations)?
        .ok_or_else(|| invalid("production downloader API disappeared"))?;
    download_checks(&downloads, &mut checks)?;
    Ok(result(ID,checks.failed.is_empty(),json!({"SourceSHA256":SOURCE_SHA,"SourceChecks":checks.source,"FixedSourceChecks":65,"ExampleInventories":inventories,"ParseCalls":29,"SharedParamsSlots":3,"WindowsSourceExclusions":["setenv-only valid/invalid/negated/override environment cases"],"DownloadUrls":[GOOD_URL,BAD_URL],"DownloadObservations":downloads,"Failures":checks.failed}).to_string()))
}

#[cfg(test)]
mod focused_guards {
    use super::*;
    #[test]
    fn duplicate_options_and_download_mutations_are_measured_failures() -> Result<()> {
        let steps = source_steps();
        assert_eq!(steps.len(), 63);
        assert_eq!(
            steps
                .iter()
                .filter(|s| s.operation["Action"] == "parse")
                .count(),
            29
        );
        assert_eq!(steps.iter().map(|s| s.contracts.len()).sum::<usize>(), 60);
        let option = json!({"Args":["--long","-s"],"NegatedArgs":["--no-long","-n"],"AllArgs":["-s"],"Environment":[],"Speculative":true});
        let mut checks = Checks::default();
        inventory(&json!({"Options":[option.clone(),option]}), &mut checks)?;
        assert_eq!(checks.failed.len(), 3); // duplicate plus both negated-order checks
        let mut checks = Checks::default();
        download_checks(
            &[
                json!({"HttpCode":200,"BodyHex":"6c6c616d612e637070","Exception":null}),
                json!({"HttpCode":404,"BodyHex":"","Exception":null}),
                json!({"HttpCode":null,"BodyHex":null,"Exception":{"Kind":"standard","Message":"cancelled"}}),
            ],
            &mut checks,
        )?;
        assert_eq!(checks.source, 5);
        assert_eq!(checks.failed, Vec::<String>::new());
        let mut checks = Checks::default();
        download_checks(
            &[
                json!({"HttpCode":200,"BodyHex":"00","Exception":null}),
                json!({"HttpCode":200,"BodyHex":"","Exception":null}),
                json!({"HttpCode":200,"BodyHex":"","Exception":null}),
            ],
            &mut checks,
        )?;
        assert_eq!(checks.failed.len(), 3);
        assert!(body(&json!("gg")).is_err());
        Ok(())
    }
}
