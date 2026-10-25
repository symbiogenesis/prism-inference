//! Safe Rust ownership of every default test-json-schema-to-grammar.cpp assertion.
//!
//! Source: PrismML-Eng/llama.cpp b87ca7225456b06b0975b8239c66b5aecf07e37c;
//! see tools/prism/upstream/LICENSE for MIT terms. Raw schema JSON preserves
//! insertion order and number lexemes; expected grammars are source literals.
//! The optional independent Python implementation follows the original skip
//! rules. Expected-grammar validation deliberately checks the root symbol only:
//! the original source ignores the production parser's boolean return value.

#[cfg(test)]
use std::time::Duration;

use serde_json::{Value, json};

use super::engine::Engine;
use super::llama_grammar::{array, exact_fields, field, uint};
use super::schema_python::{Availability, PythonReference};
use super::{Result, invalid, result};

const ID: &str = "tests/test-json-schema-to-grammar.cpp";
const SOURCE_SHA256: &str = "409361847bab0707ee02749c4e6978a63f9cc9f119ffff40de8c69b0c948e99d";
const CASES: &str = include_str!("json-schema-cases.json");
const MANDATORY_ASSERTIONS: usize = 269;

#[derive(Default)]
struct Assertions {
    count: usize,
    failures: Vec<String>,
}
impl Assertions {
    fn check(&mut self, name: impl Into<String>, passed: bool) {
        self.count += 1;
        if !passed {
            self.failures.push(name.into());
        }
    }
}

fn text<'a>(value: &'a Value, name: &str) -> Result<&'a str> {
    field(value, name)?
        .as_str()
        .ok_or_else(|| invalid(format!("schema {name} must be text")))
}

fn fixture() -> Result<Value> {
    let fixture: Value = serde_json::from_str(CASES)?;
    let shared = array(&fixture, "Shared")?;
    let cpp = array(&fixture, "CppOnly")?;
    let successful = |cases: &[Value]| -> Result<usize> {
        cases.iter().try_fold(0, |count, case| {
            let status = text(case, "ExpectedStatus")?;
            if !["SUCCESS", "FAILURE"].contains(&status) {
                return Err(invalid("invalid source schema status"));
            }
            Ok(count + usize::from(status == "SUCCESS"))
        })
    };
    if text(&fixture, "Source")? != ID
        || text(&fixture, "SourceSha256")? != SOURCE_SHA256
        || shared.len() != 78
        || successful(shared)? != 76
        || cpp.len() != 6
        || successful(cpp)? != 5
        || array(&fixture, "ResolvesToString")?.len() != 28
    {
        return Err(invalid("schema source fixture extent mismatch"));
    }
    Ok(fixture)
}

fn trim(source: &str) -> String {
    let source = source.trim_matches([' ', '\n', '\r', '\t']);
    let mut output = String::new();
    let mut line_start = true;
    for character in source.chars() {
        if line_start && matches!(character, ' ' | '\t') {
            continue;
        }
        output.push(character);
        line_start = character == '\n';
    }
    output
}

fn exchange(engine: &Engine, operations: &[Value]) -> Result<Option<Vec<Value>>> {
    let mut output = Vec::new();
    for chunk in operations.chunks(8) {
        let Some(value) = engine.execute_common(
            "json-schema",
            &json!({"SchemaVersion":1,"Operations":chunk}),
        )?
        else {
            return Ok(None);
        };
        exact_fields(&value, &["SchemaVersion", "Results"])?;
        let results = array(&value, "Results")?;
        if uint(&value, "SchemaVersion")? != 1 || results.len() != chunk.len() {
            return Err(invalid("schema native response framing mismatch"));
        }
        output.extend_from_slice(results);
    }
    Ok(Some(output))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ConversionStatus {
    Success,
    Failure,
}

fn conversion_status(value: &Value) -> Result<ConversionStatus> {
    exact_fields(value, &["Grammar", "ExceptionKind", "Message"])?;
    let _ = text(value, "Message")?;
    match field(value, "ExceptionKind")? {
        Value::Null => {
            let _ = text(value, "Grammar")?;
            Ok(ConversionStatus::Success)
        }
        Value::String(kind) if kind == "invalid-argument" && field(value, "Grammar")?.is_null() => {
            Ok(ConversionStatus::Failure)
        }
        _ => Err(invalid(format!(
            "unexpected production schema exception: {}",
            text(value, "Message")?
        ))),
    }
}

fn expected_status(case: &Value) -> Result<ConversionStatus> {
    match text(case, "ExpectedStatus")? {
        "SUCCESS" => Ok(ConversionStatus::Success),
        "FAILURE" => Ok(ConversionStatus::Failure),
        _ => Err(invalid("invalid source schema status")),
    }
}

fn check_probe(case: &Value, actual: bool, assertions: &mut Assertions) -> Result<()> {
    let expected = field(case, "Expected")?
        .as_bool()
        .ok_or_else(|| invalid("source string expectation must be boolean"))?;
    assertions.check(
        format!(
            "string probe: {}: expected {expected}, actual {actual}",
            text(case, "Name")?
        ),
        actual == expected,
    );
    Ok(())
}

fn check_conversion(
    case: &Value,
    status: ConversionStatus,
    grammar: Option<&str>,
    message: &str,
    assertions: &mut Assertions,
) -> Result<()> {
    let name = text(case, "Name")?;
    if status == ConversionStatus::Success {
        let actual = trim(grammar.ok_or_else(|| invalid("successful schema grammar missing"))?);
        let expected = trim(text(case, "ExpectedGrammar")?);
        assertions.check(
            format!("C++ grammar: {name}:\nExpected:\n{expected}\nActual:\n{actual}"),
            actual == expected,
        );
    }
    assertions.check(
        format!(
            "C++ status: {name}: expected {}, actual {}; {message}",
            text(case, "ExpectedStatus")?,
            if status == ConversionStatus::Success {
                "SUCCESS"
            } else {
                "FAILURE"
            }
        ),
        status == expected_status(case)?,
    );
    Ok(())
}

fn run_probes(engine: &Engine, fixture: &Value, assertions: &mut Assertions) -> Result<bool> {
    let cases = array(fixture, "ResolvesToString")?;
    let operations=cases.iter().map(|case|Ok(json!({"Kind":"json-schema-info","SchemaJson":text(case,"SchemaJson")?,"Actions":["resolve-refs","resolves-to-string"]}))).collect::<Result<Vec<_>>>()?;
    let Some(values) = exchange(engine, &operations)? else {
        return Ok(false);
    };
    for (case, value) in cases.iter().zip(values) {
        exact_fields(&value, &["Trace"])?;
        let trace = array(&value, "Trace")?;
        if trace.len() != 2 {
            return Err(invalid("schema-info trace extent mismatch"));
        }
        exact_fields(&trace[0], &["SchemaJson"])?;
        let _ = text(&trace[0], "SchemaJson")?;
        exact_fields(&trace[1], &["ResolvesToString"])?;
        let actual = field(&trace[1], "ResolvesToString")?
            .as_bool()
            .ok_or_else(|| invalid("resolves-to-string result must be boolean"))?;
        check_probe(case, actual, assertions)?;
    }
    Ok(true)
}

fn run_conversions(engine: &Engine, cases: &[Value], assertions: &mut Assertions) -> Result<()> {
    let operations=cases.iter().map(|case|Ok(json!({"Kind":"json-schema-convert","SchemaJson":text(case,"SchemaJson")?,"ForceGbnf":true}))).collect::<Result<Vec<_>>>()?;
    let values = exchange(engine, &operations)?
        .ok_or_else(|| invalid("schema conversion production API disappeared"))?;
    for (case, value) in cases.iter().zip(values) {
        let status = conversion_status(&value)?;
        check_conversion(
            case,
            status,
            field(&value, "Grammar")?.as_str(),
            text(&value, "Message")?,
            assertions,
        )?;
    }
    Ok(())
}

fn expectation_has_root(value: &Value) -> Result<bool> {
    exact_fields(value, &["Parsed", "SymbolIds", "Rules"])?;
    if field(value, "Parsed")?.as_bool().is_none() || !field(value, "Rules")?.is_array() {
        return Err(invalid("grammar-parser result type mismatch"));
    }
    let mut found = false;
    for symbol in array(value, "SymbolIds")? {
        exact_fields(symbol, &["Name", "Id"])?;
        let _ = uint(symbol, "Id")?;
        found |= text(symbol, "Name")? == "root";
    }
    Ok(found)
}

fn run_expectations(engine: &Engine, cases: &[Value], assertions: &mut Assertions) -> Result<()> {
    let successful = cases
        .iter()
        .filter_map(|case| match expected_status(case) {
            Ok(ConversionStatus::Success) => Some(Ok(case)),
            Ok(ConversionStatus::Failure) => None,
            Err(error) => Some(Err(error)),
        })
        .collect::<Result<Vec<_>>>()?;
    let operations = successful
        .iter()
        .map(|case| {
            Ok(json!({"Kind":"grammar-parser-parse","Grammar":text(case,"ExpectedGrammar")?}))
        })
        .collect::<Result<Vec<_>>>()?;
    let values = exchange(engine, &operations)?
        .ok_or_else(|| invalid("schema expectation grammar parser production API disappeared"))?;
    for (case, value) in successful.iter().zip(values) {
        assertions.check(
            format!("expected grammar root: {}", text(case, "Name")?),
            expectation_has_root(&value)?,
        );
    }
    Ok(())
}

fn run_python(engine: &Engine, cases: &[Value], assertions: &mut Assertions) -> Result<Value> {
    match PythonReference::prepare(engine)? {
        Availability::Skipped(reason) => {
            Ok(json!({"Status":"skip","Detail":reason,"Assertions":0}))
        }
        Availability::Ready(reference) => {
            let before = assertions.count;
            let failures = assertions.failures.len();
            for (index, case) in cases.iter().enumerate() {
                let output = reference.convert(index, text(case, "SchemaJson")?)?;
                let name = text(case, "Name")?;
                assertions.check(
                    format!("Python status: {name}"),
                    output.success == (expected_status(case)? == ConversionStatus::Success),
                );
                // Unlike the C++ path, the original Python path compares stdout
                // even for expected failures; those cases expect an empty grammar.
                assertions.check(
                    format!("Python grammar: {name}"),
                    trim(&output.grammar) == trim(text(case, "ExpectedGrammar")?),
                );
            }
            let executed = assertions.count - before;
            if executed != 156 {
                return Err(invalid("Python schema assertion extent mismatch"));
            }
            Ok(
                json!({"Status":if failures==assertions.failures.len(){"pass"}else{"fail"},"Cases":78,"Assertions":executed,"Implementation":"pinned examples/json_schema_to_grammar.py"}),
            )
        }
    }
}

/// Runs all mandatory C++ contracts and the original conditional independent Python path.
pub(super) fn run(engine: &Engine) -> Result<Value> {
    let fixture = fixture()?;
    let mut assertions = Assertions::default();
    if !run_probes(engine, &fixture, &mut assertions)? {
        return Ok(result(
            ID,
            false,
            "unsupported production API: schema conversion/type inspection and grammar parser"
                .to_owned(),
        ));
    }
    let shared = array(&fixture, "Shared")?;
    run_conversions(engine, shared, &mut assertions)?;
    run_conversions(engine, array(&fixture, "CppOnly")?, &mut assertions)?;
    let before_python = assertions.count;
    let python = run_python(engine, shared, &mut assertions)?;
    let python_count = assertions.count - before_python;
    run_expectations(engine, shared, &mut assertions)?;
    let mandatory = assertions.count - python_count;
    if mandatory != MANDATORY_ASSERTIONS {
        assertions.failures.push(format!(
            "mandatory schema success extent mismatch: {mandatory}"
        ));
    }
    let mut report = result(
        ID,
        assertions.failures.is_empty(),
        format!(
            "78shared/6C++only conversions (3negative),28string probes,76shared expected-grammar root checks; mandatory={mandatory}; Python assertions={python_count}; failures={:?}; no Node execution or benchmark in source",
            assertions.failures
        ),
    );
    report["PythonReference"] = python;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::super::schema_python::synthetic_limits;
    use super::*;

    #[cfg(feature = "cli")]
    #[test]
    fn rust_schema_contract_uses_original_cases() -> Result<()> {
        use bonsai_prism_runtime::schema::{self, SchemaInfo};

        // Exercise the public Rust API without requiring native bridge wiring.
        // The source runner and this entry point share all acceptance predicates.
        let fixture = fixture()?;
        let mut assertions = Assertions::default();
        for section in ["Shared", "CppOnly"] {
            for case in array(&fixture, section)? {
                match schema::to_grammar(text(case, "SchemaJson")?, true) {
                    Ok(grammar) => check_conversion(
                        case,
                        ConversionStatus::Success,
                        Some(&grammar),
                        "",
                        &mut assertions,
                    )?,
                    Err(error) => check_conversion(
                        case,
                        ConversionStatus::Failure,
                        None,
                        &error.to_string(),
                        &mut assertions,
                    )?,
                }
            }
        }
        for case in array(&fixture, "ResolvesToString")? {
            let mut info = SchemaInfo::new(text(case, "SchemaJson")?)?;
            info.resolve_refs()?;
            check_probe(case, info.resolves_to_string(), &mut assertions)?;
        }
        assert!(
            assertions.failures.is_empty(),
            "{}",
            assertions.failures.join("\n")
        );
        assert_eq!(assertions.count, 193);
        Ok(())
    }

    #[test]
    fn source_trim_exception_and_root_guards_are_exact() -> Result<()> {
        let _ = fixture()?;
        assert_eq!(
            trim(" \r\n\troot ::= \"x\"\n  \n\tchild ::= \"y\"\n "),
            "root ::= \"x\"\n\nchild ::= \"y\""
        );
        assert_eq!(trim("\u{2003}x\u{2003}"), "\u{2003}x\u{2003}");
        assert!(matches!(
            conversion_status(
                &json!({"Grammar":null,"ExceptionKind":"invalid-argument","Message":"input rejection"})
            )?,
            ConversionStatus::Failure
        ));
        assert!(
            conversion_status(
                &json!({"Grammar":null,"ExceptionKind":"runtime","Message":"unexpected"})
            )
            .is_err()
        );
        assert!(expectation_has_root(
            &json!({"Parsed":false,"SymbolIds":[{"Name":"root","Id":0}],"Rules":[]})
        )?);
        assert!(!synthetic_limits(
            Duration::from_secs(30),
            16 * 1024 * 1024,
            1024 * 1024
        ));
        assert!(synthetic_limits(
            Duration::from_secs(30) + Duration::from_nanos(1),
            0,
            0
        ));
        assert!(synthetic_limits(Duration::ZERO, 16 * 1024 * 1024 + 1, 0));
        Ok(())
    }
}
