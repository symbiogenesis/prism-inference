//! Safe Rust ownership of all six pinned tests/peg-parser source contracts.
//!
//! Fixtures contain source-derived builder graphs, byte inputs and assertions;
//! no native answers are frozen. Native code only dispatches production PEG
//! builder, parser, grammar generation and arena serialization operations.
//! See tools/prism/upstream/LICENSE for the upstream MIT license.

use std::ffi::OsStr;
use std::time::Instant;

use serde_json::{Value, json};

use super::engine::Engine;
use super::llama_grammar::{array, exact_fields, exchange, field, uint};
use super::{Result, invalid, result};

struct Source {
    id: &'static str,
    hash: &'static str,
    fixture: &'static str,
    operations: usize,
    assertions: usize,
    expected_executed: usize,
}

const SOURCES: [Source; 6] = [
    Source {
        id: "tests/peg-parser/test-basic.cpp",
        hash: "24c5c31c65b2f5fc88919a4b5d06506ae644f8f0460c14c0134cd5439c5a7694",
        fixture: include_str!("peg-basic.json"),
        operations: 42,
        assertions: 44,
        expected_executed: 44,
    },
    Source {
        id: "tests/peg-parser/test-gbnf-generation.cpp",
        hash: "d14973bb98faaa87d1e17af30b76d413e2ac6541ab800c20eb55d48c8fa1952a",
        fixture: include_str!("peg-gbnf-generation.json"),
        operations: 29,
        assertions: 29,
        expected_executed: 29,
    },
    Source {
        id: "tests/peg-parser/test-json-parser.cpp",
        hash: "2731298bf7f28047852e4f420243fc533bdb1408611257f47a5fff2adacb36fc",
        fixture: include_str!("peg-json-parser.json"),
        operations: 12,
        assertions: 18,
        expected_executed: 18,
    },
    Source {
        id: "tests/peg-parser/test-json-serialization.cpp",
        hash: "276f50faa0f1f50da7d5289b3e13110c26cf9120ff23bbdb4a6a8c12c384a028",
        fixture: include_str!("peg-json-serialization.json"),
        operations: 1,
        assertions: 2,
        expected_executed: 2,
    },
    Source {
        id: "tests/peg-parser/test-python-dict-parser.cpp",
        hash: "ab6b20f141fed193bc93e810608c1b6800415d14ff832ad3ff84671594d18d4f",
        fixture: include_str!("peg-python-dict-parser.json"),
        operations: 27,
        assertions: 47,
        expected_executed: 47,
    },
    Source {
        id: "tests/peg-parser/test-unicode.cpp",
        hash: "5eda6c6457af1ff1a62081dadac597d7706bb656e9941bb828de6421d9c563cf",
        fixture: include_str!("peg-unicode.json"),
        operations: 59,
        assertions: 112,
        expected_executed: 101,
    },
];

/// Thin observations over the reusable production PEG arena API.
pub(super) fn execute(request: &Value) -> Result<Option<Value>> {
    use bonsai_prism_runtime::peg::Arena;
    exact_fields(request, &["SchemaVersion", "Operations"])?;
    let operations = array(request, "Operations")?;
    if uint(request, "SchemaVersion")? != 1 || operations.len() > 128 {
        return Err(invalid("PEG request framing exceeds bound"));
    }
    if operations.iter().any(|operation| {
        !matches!(
            operation["Kind"].as_str(),
            Some("peg-evaluate" | "peg-serialize" | "peg-deserialize")
        )
    }) {
        return Ok(None);
    }
    let mut results = Vec::new();
    for operation in operations {
        let kind = text(operation, "Kind")?;
        if kind == "peg-deserialize" {
            exact_fields(operation, &["Kind", "SerializedJson"])?;
            results.push(json!({"Root":Arena::load(text(operation,"SerializedJson")?)?.root()}));
            continue;
        }
        let arena = Arena::from_graph(field(operation, "Graph")?)?;
        if kind == "peg-serialize" {
            exact_fields(operation, &["Kind", "Graph"])?;
            results.push(json!({"SerializedJson":arena.save()?}));
            continue;
        }
        exact_fields(
            operation,
            &["Kind", "Graph", "Inputs", "GenerateGrammar", "RoundTrip"],
        )?;
        let parse = |arena: &Arena| -> Result<Vec<Value>> {
            array(operation, "Inputs")?
                .iter()
                .map(|input| {
                    exact_fields(input, &["Bytes", "Flags"])?;
                    let observation = arena.parse(
                        &bytes(field(input, "Bytes")?)?,
                        u8::try_from(uint(input, "Flags")?)?,
                    )?;
                    Ok(serde_json::to_value(observation)?)
                })
                .collect()
        };
        let observations = parse(&arena)?;
        let grammars = array(operation, "GenerateGrammar")?
            .iter()
            .map(|value| {
                arena.grammar(
                    value
                        .as_bool()
                        .ok_or_else(|| invalid("PEG grammar lazy flag missing"))?,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let serialized = boolean(operation, "RoundTrip")?
            .then(|| arena.save())
            .transpose()?;
        let roundtrip = serialized
            .as_deref()
            .map(|value| Arena::load(value).and_then(|arena| parse(&arena)))
            .transpose()?
            .unwrap_or_default();
        results.push(json!({"Results":observations,"GrammarTexts":grammars,"SerializedJson":serialized,"RoundTripResults":roundtrip}));
    }
    Ok(Some(json!({"SchemaVersion":1,"Results":results})))
}

fn text<'a>(value: &'a Value, name: &str) -> Result<&'a str> {
    field(value, name)?
        .as_str()
        .ok_or_else(|| invalid(format!("PEG {name} must be text")))
}

fn boolean(value: &Value, name: &str) -> Result<bool> {
    field(value, name)?
        .as_bool()
        .ok_or_else(|| invalid(format!("PEG {name} must be boolean")))
}

fn bytes(value: &Value) -> Result<Vec<u8>> {
    value
        .as_array()
        .ok_or_else(|| invalid("PEG byte string must be an array"))?
        .iter()
        .map(|value| {
            Ok(u8::try_from(
                value
                    .as_u64()
                    .ok_or_else(|| invalid("PEG byte must be unsigned"))?,
            )?)
        })
        .collect()
}

fn fixture(source: &Source) -> Result<Value> {
    let fixture: Value = serde_json::from_str(source.fixture)?;
    if uint(&fixture, "SchemaVersion")? != 1
        || text(&fixture, "Source")? != source.id
        || text(&fixture, "SourceSha256")? != source.hash
        || array(&fixture, "Operations")?.len() != source.operations
        || array(&fixture, "Assertions")?.len() != source.assertions
    {
        return Err(invalid(
            "PEG source provenance or complete contract count mismatch",
        ));
    }
    Ok(fixture)
}

fn parse_result(value: &Value) -> Result<()> {
    exact_fields(value, &["Type", "Start", "End"])?;
    if uint(value, "Type")? > 2 {
        return Err(invalid("unknown PEG parse result type"));
    }
    let _ = uint(value, "Start")?;
    let _ = uint(value, "End")?;
    Ok(())
}

fn validate_observation(operation: &Value, value: &Value) -> Result<()> {
    exact_fields(
        value,
        &[
            "Results",
            "GrammarTexts",
            "SerializedJson",
            "RoundTripResults",
        ],
    )?;
    let results = array(value, "Results")?;
    let grammars = array(value, "GrammarTexts")?;
    let roundtrip = array(value, "RoundTripResults")?;
    let serialized = field(value, "SerializedJson")?;
    let expected_roundtrip = if boolean(operation, "RoundTrip")? {
        array(operation, "Inputs")?.len()
    } else {
        0
    };
    if results.len() != array(operation, "Inputs")?.len()
        || grammars.len() != array(operation, "GenerateGrammar")?.len()
        || roundtrip.len() != expected_roundtrip
    {
        return Err(invalid("PEG native observation count mismatch"));
    }
    if boolean(operation, "RoundTrip")? {
        if !serialized.is_string() {
            return Err(invalid("PEG roundtrip must retain serialized arena JSON"));
        }
    } else if !serialized.is_null() {
        return Err(invalid("unexpected PEG arena serialization"));
    }
    for value in results.iter().chain(roundtrip) {
        parse_result(value)?;
    }
    for grammar in grammars {
        if !grammar.is_string() {
            return Err(invalid("PEG generated grammar must be text"));
        }
    }
    Ok(())
}

#[derive(Debug, PartialEq)]
enum Evaluated {
    Boolean(bool),
    Number(i64),
    Bytes(Vec<u8>),
}

impl Evaluated {
    fn number(self) -> Result<i64> {
        if let Self::Number(value) = self {
            Ok(value)
        } else {
            Err(invalid("PEG assertion requires an integer"))
        }
    }
    fn boolean(self) -> Result<bool> {
        if let Self::Boolean(value) = self {
            Ok(value)
        } else {
            Err(invalid("PEG assertion requires a boolean"))
        }
    }
    fn bytes(self) -> Result<Vec<u8>> {
        if let Self::Bytes(value) = self {
            Ok(value)
        } else {
            Err(invalid("PEG assertion requires a byte string"))
        }
    }
}

// Equivalent to upstream regex_replace(s, (^|\n)\s+, "$1") in its C locale.
fn trim_leading_space(input: &[u8]) -> Vec<u8> {
    let whitespace = |byte: u8| matches!(byte, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c);
    let mut output = Vec::with_capacity(input.len());
    let mut index = 0;
    while index < input.len() && whitespace(input[index]) {
        index += 1;
    }
    while index < input.len() {
        let byte = input[index];
        output.push(byte);
        index += 1;
        if byte == b'\n' {
            while index < input.len() && whitespace(input[index]) {
                index += 1;
            }
        }
    }
    output
}

fn operation_result<'a>(expression: &Value, values: &'a [Value]) -> Result<&'a Value> {
    values
        .get(usize::try_from(uint(expression, "Operation")?)?)
        .ok_or_else(|| invalid("PEG assertion operation index out of range"))
}

fn referenced(expression: &Value, values: &[Value], grammar: bool) -> Result<Evaluated> {
    let value = operation_result(expression, values)?;
    let collection = if grammar {
        "GrammarTexts"
    } else {
        text(expression, "Collection")?
    };
    if !matches!(collection, "GrammarTexts" | "Results" | "RoundTripResults") {
        return Err(invalid("unknown PEG assertion result collection"));
    }
    let value = array(value, collection)?
        .get(usize::try_from(uint(expression, "Index")?)?)
        .ok_or_else(|| invalid("PEG assertion result index out of range"))?;
    if grammar {
        return Ok(Evaluated::Bytes(
            value
                .as_str()
                .ok_or_else(|| invalid("PEG grammar observation must be text"))?
                .as_bytes()
                .to_vec(),
        ));
    }
    let field_name = text(expression, "Field")?;
    if !matches!(field_name, "Type" | "Start" | "End") {
        return Err(invalid("unknown PEG assertion result field"));
    }
    Ok(Evaluated::Number(i64::try_from(uint(value, field_name)?)?))
}

fn arguments(expression: &Value, expected: usize) -> Result<&[Value]> {
    let arguments = array(expression, "Args")?;
    if arguments.len() != expected {
        return Err(invalid("PEG assertion expression arity mismatch"));
    }
    Ok(arguments)
}

fn evaluate(expression: &Value, values: &[Value], depth: usize) -> Result<Evaluated> {
    if depth > 64 {
        return Err(invalid("PEG assertion expression exceeds depth bound"));
    }
    if let Some(value) = expression.as_bool() {
        return Ok(Evaluated::Boolean(value));
    }
    if let Some(value) = expression.as_i64() {
        return Ok(Evaluated::Number(value));
    }
    if let Some(value) = expression.get("Bytes") {
        return Ok(Evaluated::Bytes(bytes(value)?));
    }
    let child = |expression: &Value| evaluate(expression, values, depth + 1);
    match text(expression, "Op")? {
        "result" => referenced(expression, values, false),
        "grammar" => referenced(expression, values, true),
        "equal" => {
            let args = arguments(expression, 2)?;
            Ok(Evaluated::Boolean(child(&args[0])? == child(&args[1])?))
        }
        "subtract" => {
            let args = arguments(expression, 2)?;
            Ok(Evaluated::Number(
                child(&args[0])?
                    .number()?
                    .checked_sub(child(&args[1])?.number()?)
                    .ok_or_else(|| invalid("PEG assertion arithmetic overflow"))?,
            ))
        }
        "or" => {
            let args = arguments(expression, 2)?;
            Ok(Evaluated::Boolean(
                child(&args[0])?.boolean()? || child(&args[1])?.boolean()?,
            ))
        }
        "slice" => {
            let args = arguments(expression, 3)?;
            let bytes = child(&args[0])?.bytes()?;
            let start = usize::try_from(child(&args[1])?.number()?)?;
            let length = usize::try_from(child(&args[2])?.number()?)?;
            let end = start
                .checked_add(length)
                .ok_or_else(|| invalid("PEG assertion slice overflow"))?;
            Ok(Evaluated::Bytes(
                bytes
                    .get(start..end)
                    .ok_or_else(|| invalid("PEG matched-text extent exceeds input"))?
                    .to_vec(),
            ))
        }
        "trim-leading-space" => {
            let args = arguments(expression, 1)?;
            Ok(Evaluated::Bytes(trim_leading_space(
                &child(&args[0])?.bytes()?,
            )))
        }
        _ => Err(invalid("unknown PEG source assertion expression")),
    }
}

pub(super) fn run(engine: &Engine, id: &str) -> Result<Value> {
    let source = SOURCES
        .iter()
        .find(|source| source.id == id)
        .ok_or_else(|| invalid("unknown PEG source contract"))?;
    let fixture = fixture(source)?;
    let operations = array(&fixture, "Operations")?;
    let Some(values) = exchange(engine, operations)? else {
        return Ok(result(
            source.id,
            false,
            format!(
                "unsupported production API: PEG builders/parser/grammar/arena serialization; {} source assertion records required",
                source.assertions
            ),
        ));
    };
    for (operation, value) in operations.iter().zip(&values) {
        validate_observation(operation, value)?;
    }
    let mut failures = Vec::new();
    let mut executed = 0;
    for assertion in array(&fixture, "Assertions")? {
        if !evaluate(field(assertion, "When")?, &values, 0)?.boolean()? {
            continue;
        }
        executed += 1;
        let expected = evaluate(field(assertion, "Left")?, &values, 0)?;
        let actual = evaluate(field(assertion, "Right")?, &values, 0)?;
        if expected != actual {
            failures.push(format!(
                "{}: {expected:?} != {actual:?}",
                text(assertion, "Name")?
            ));
        }
    }
    if executed != source.expected_executed {
        failures.push(format!(
            "source assertion extent mismatch: {executed} != {}",
            source.expected_executed
        ));
    }
    Ok(result(
        source.id,
        failures.is_empty(),
        format!(
            "all original builder graphs and byte inputs; operations={}; assertion_records={}; executed={executed}; expected_executed_on_original_results={}; source runtime guards preserved; CI-only benchmark paths={}; failed={}; {}",
            source.operations,
            source.assertions,
            source.expected_executed,
            array(&fixture, "Benchmarks")?.len(),
            failures.len(),
            failures.join("; ")
        ),
    ))
}

pub(super) fn run_basic(engine: &Engine) -> Result<Value> {
    run(engine, SOURCES[0].id)
}

pub(super) fn run_gbnf(engine: &Engine) -> Result<Value> {
    run(engine, SOURCES[1].id)
}

pub(super) fn run_json(engine: &Engine) -> Result<Value> {
    run(engine, SOURCES[2].id)
}

pub(super) fn run_serialization(engine: &Engine) -> Result<Value> {
    run(engine, SOURCES[3].id)
}

pub(super) fn run_python_dict(engine: &Engine) -> Result<Value> {
    run(engine, SOURCES[4].id)
}

pub(super) fn run_unicode(engine: &Engine) -> Result<Value> {
    run(engine, SOURCES[5].id)
}

pub(super) fn run_all(engine: &Engine) -> Result<Value> {
    // Preserve test-peg-parser.cpp's original group order. The inherited
    // deserialize timing path is dispatched separately by the explicit CI
    // benchmark entry and never folded into correctness observations.
    let groups = [
        run_basic(engine)?,
        run_unicode(engine)?,
        run_json(engine)?,
        run_gbnf(engine)?,
        run_serialization(engine)?,
        run_python_dict(engine)?,
    ];
    let failed = groups
        .iter()
        .filter(|group| group.get("Status").and_then(Value::as_str) != Some("pass"))
        .count();
    let mut report = result(
        "tests/test-peg-parser.cpp",
        failed == 0,
        format!(
            "6 original groups in basic/unicode/json/gbnf/serialization/python-dict order; 241 inherited assertions; failed_groups={failed}; deserialize100 benchmark dispatched separately in CI"
        ),
    );
    report["Groups"] = json!(groups);
    Ok(report)
}

/// Runs the original 100-deserialize workload in an explicit CI-only path.
/// Wall time includes the bridge transport; it is not a native-kernel timing.
pub(super) fn benchmark_deserialization(engine: &Engine) -> Result<Value> {
    if std::env::var_os("GITHUB_ACTIONS").as_deref() != Some(OsStr::new("true")) {
        return Err(invalid(
            "PEG deserialize benchmark runs only on GitHub Actions",
        ));
    }
    let fixture = fixture(&SOURCES[3])?;
    let benchmark = &array(&fixture, "Benchmarks")?[0];
    if uint(benchmark, "Iterations")? != 100 || text(benchmark, "Operation")? != "deserialize" {
        return Err(invalid("original PEG benchmark workload mismatch"));
    }
    let operation = json!({"Kind":"peg-serialize","Graph":field(benchmark,"Graph")?});
    let Some(values) = exchange(engine, &[operation])? else {
        return Ok(json!({
            "Id":"tests/peg-parser/test-json-serialization.cpp#deserialize",
            "Status":"unsupported",
            "Iterations":100,
            "Detail":"unsupported production API: PEG arena serialization/deserialization",
        }));
    };
    exact_fields(&values[0], &["SerializedJson"])?;
    let serialized = text(&values[0], "SerializedJson")?;
    let operations = vec![json!({"Kind":"peg-deserialize","SerializedJson":serialized}); 100];
    let started = Instant::now();
    let values = exchange(engine, &operations)?
        .ok_or_else(|| invalid("PEG benchmark production API is unsupported"))?;
    let nanoseconds = u64::try_from(started.elapsed().as_nanos())?;
    for value in values {
        exact_fields(&value, &["Root"])?;
        let _ = uint(&value, "Root")?;
    }
    Ok(
        json!({"Id":"tests/peg-parser/test-json-serialization.cpp#deserialize","Status":"measured","Iterations":100,"WallNanoseconds":nanoseconds,"MeasurementScope":"100 production JSON parse and arena deserialization operations including native bridge transport","CorrectnessThreshold":null}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn original_source(index: usize) -> Result<()> {
        let source = &SOURCES[index];
        let fixture = fixture(source)?;
        let response = execute(&json!({"SchemaVersion":1,"Operations":fixture["Operations"]}))?
            .ok_or("PEG production response missing")?;
        let values = array(&response, "Results")?;
        let mut executed = 0;
        let mut failures = Vec::new();
        for (operation, value) in array(&fixture, "Operations")?.iter().zip(values) {
            validate_observation(operation, value)?;
        }
        for assertion in array(&fixture, "Assertions")? {
            if !evaluate(field(assertion, "When")?, values, 0)?.boolean()? {
                continue;
            }
            executed += 1;
            let left = evaluate(field(assertion, "Left")?, values, 0)?;
            let right = evaluate(field(assertion, "Right")?, values, 0)?;
            if left != right {
                failures.push(format!(
                    "{}: {left:?} != {right:?}",
                    text(assertion, "Name")?
                ));
            }
        }
        assert_eq!(executed, source.expected_executed);
        assert!(failures.is_empty(), "{}", failures.join("\n"));
        Ok(())
    }
    #[test]
    fn production_original_peg_basic() -> Result<()> {
        original_source(0)
    }
    #[test]
    fn production_original_peg_gbnf() -> Result<()> {
        original_source(1)
    }
    #[test]
    fn production_original_peg_json() -> Result<()> {
        original_source(2)
    }
    #[test]
    fn production_original_peg_serialization() -> Result<()> {
        original_source(3)
    }
    #[test]
    fn production_original_peg_python() -> Result<()> {
        original_source(4)
    }
    #[test]
    fn production_original_peg_unicode() -> Result<()> {
        original_source(5)
    }

    #[test]
    fn assertion_slices_and_native_types_are_bounded_and_byte_exact() -> Result<()> {
        let expression = json!({"Op":"slice","Args":[{"Bytes":[0xff,0xc3,0xa9]},1,2]});
        assert_eq!(
            evaluate(&expression, &[], 0)?,
            Evaluated::Bytes(vec![0xc3, 0xa9])
        );
        let invalid = json!({"Op":"slice","Args":[{"Bytes":[0xff]},0,2]});
        assert!(evaluate(&invalid, &[], 0).is_err());
        assert!(parse_result(&json!({"Type":3,"Start":0,"End":0})).is_err());
        assert_eq!(
            trim_leading_space(b"\n  root\n\n \tother\n  "),
            b"root\nother\n"
        );
        Ok(())
    }
}
