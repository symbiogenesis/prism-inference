//! Complete safe Rust port of the default test-chat-auto-parser.cpp groups.
//!
//! Source: PrismML-Eng/llama.cpp b87ca7225456b06b0975b8239c66b5aecf07e37c;
//! see tools/prism/upstream/LICENSE for MIT terms. Every literal, template input,
//! declarative graph, callback variant and expected condition is source-derived.
//! Native adapters expose production observations only. The isolated quote
//! normalization helper from the source is independently implemented in Rust.

use serde_json::{Map, Value, json};

use super::chat_peg::{exchange, text};
use super::engine::Engine;
use super::llama_grammar::{array, exact_fields, field, uint};
use super::{Result, invalid, result};

const ID: &str = "tests/test-chat-auto-parser.cpp";
const SOURCE_SHA256: &str = "d69af3154f3ed738e47a74280bcd63d2e90cd60b05dff7f5f1f7a5b26c59478c";
const CASES: &str = include_str!("chat-auto-cases.json");
const NATIVE_ASSERTIONS: usize = 513;
const NORMALIZATION_ASSERTIONS: usize = 11;

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

fn fixture() -> Result<Value> {
    let value: Value = serde_json::from_str(CASES)?;
    let scenarios = array(&value, "Scenarios")?;
    let conditions = scenarios
        .iter()
        .try_fold(0, |count, scenario| -> Result<usize> {
            Ok(count + array(scenario, "Checks")?.len())
        })?;
    let operations = scenarios
        .iter()
        .try_fold(0, |count, scenario| -> Result<usize> {
            Ok(count + array(scenario, "Operations")?.len())
        })?;
    if text(&value, "Source")? != ID
        || text(&value, "SourceSha256")? != SOURCE_SHA256
        || array(&value, "Groups")?.len() != 17
        || uint(&value, "SourceSubgroups")? != 96
        || uint(&value, "RoleTemplates")? != 42
        || scenarios.len() != 85
        || conditions != NATIVE_ASSERTIONS
        || operations != 136
        || field(&value, "Assets")?.as_object().map(Map::len) != Some(45)
        || array(&value, "Normalization")?.len() != 7
        || !array(&value, "Benchmarks")?.is_empty()
        || array(&value, "ExcludedUnregisteredFunctions")? != [json!("test_tool_format_cohere")]
    {
        return Err(invalid("auto-parser source fixture extent mismatch"));
    }
    Ok(value)
}

fn asset<'a>(assets: &'a Map<String, Value>, key: &str) -> Result<&'a str> {
    assets
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(format!("missing source chat template {key}")))
}

fn boolean(value: &Value) -> Result<bool> {
    value
        .as_bool()
        .ok_or_else(|| invalid("auto-parser expression must be boolean"))
}
fn integer(value: &Value) -> Result<i64> {
    value
        .as_i64()
        .ok_or_else(|| invalid("auto-parser expression must be int64"))
}
fn string(value: &Value) -> Result<&str> {
    value
        .as_str()
        .ok_or_else(|| invalid("auto-parser expression must be text"))
}

fn result_path(expression: &Value, results: &[Value]) -> Result<Value> {
    exact_fields(expression, &["Result", "Path"])?;
    let index_value = field(expression, "Result")?;
    let index = usize::try_from(
        index_value
            .as_u64()
            .ok_or_else(|| invalid("auto-parser result index must be unsigned"))?,
    )?;
    let mut value = results
        .get(index)
        .ok_or_else(|| invalid("auto-parser result index exceeds extent"))?;
    for segment in array(expression, "Path")? {
        value = if let Some(key) = segment.as_str() {
            field(value, key)?
        } else {
            value
                .get(usize::try_from(segment.as_u64().ok_or_else(|| {
                    invalid("auto-parser path index must be unsigned")
                })?)?)
                .ok_or_else(|| invalid("auto-parser path index exceeds extent"))?
        };
    }
    Ok(value.clone())
}

fn evaluate(
    expression: &Value,
    results: &[Value],
    assets: &Map<String, Value>,
    depth: usize,
) -> Result<Value> {
    if depth > 64 {
        return Err(invalid("auto-parser assertion exceeds expression depth"));
    }
    if expression.get("Result").is_some() {
        return result_path(expression, results);
    }
    let Some(op) = expression.get("Op") else {
        return Ok(expression.clone());
    };
    exact_fields(expression, &["Op", "Args"])?;
    let args = array(expression, "Args")?;
    let operand = |index| -> Result<Value> {
        evaluate(
            args.get(index)
                .ok_or_else(|| invalid("missing auto-parser assertion operand"))?,
            results,
            assets,
            depth + 1,
        )
    };
    match op.as_str() {
        Some("equal") if args.len() == 2 => Ok(json!(operand(0)? == operand(1)?)),
        Some("not-equal") if args.len() == 2 => Ok(json!(operand(0)? != operand(1)?)),
        Some("not") if args.len() == 1 => Ok(json!(!boolean(&operand(0)?)?)),
        Some("and") if args.len() == 2 => {
            Ok(json!(boolean(&operand(0)?)? && boolean(&operand(1)?)?))
        }
        Some("or") if args.len() == 2 => {
            Ok(json!(boolean(&operand(0)?)? || boolean(&operand(1)?)?))
        }
        Some("greater") if args.len() == 2 => {
            Ok(json!(integer(&operand(0)?)? > integer(&operand(1)?)?))
        }
        Some("subtract") if args.len() == 2 => Ok(json!(
            integer(&operand(0)?)?
                .checked_sub(integer(&operand(1)?)?)
                .ok_or_else(|| invalid("auto-parser reference subtraction overflow"))?
        )),
        Some("length") if args.len() == 1 => {
            let value = operand(0)?;
            let len = value
                .as_str()
                .map(str::len)
                .or_else(|| value.as_array().map(Vec::len))
                .or_else(|| value.as_object().map(Map::len))
                .ok_or_else(|| invalid("auto-parser length requires container"))?;
            Ok(json!(len))
        }
        Some("asset-length") if args.len() == 1 => {
            Ok(json!(asset(assets, string(&operand(0)?)?)?.len()))
        }
        Some("find") if args.len() == 2 => {
            let haystack = operand(0)?;
            let needle = operand(1)?;
            Ok(json!(
                string(&haystack)?
                    .find(string(&needle)?)
                    .map_or(Ok(-1), i64::try_from)?
            ))
        }
        Some("contains") if args.len() == 2 => {
            let list = operand(0)?;
            let needle = operand(1)?;
            let list = list
                .as_array()
                .ok_or_else(|| invalid("auto-parser contains requires array"))?;
            Ok(json!(list.contains(&needle)))
        }
        Some("substr") if (2..=3).contains(&args.len()) => {
            let value = operand(0)?;
            let bytes = string(&value)?.as_bytes();
            let start = usize::try_from(integer(&operand(1)?)?)?;
            let remaining = bytes
                .len()
                .checked_sub(start)
                .ok_or_else(|| invalid("auto-parser substring starts beyond extent"))?;
            let length = if args.len() == 3 {
                usize::try_from(integer(&operand(2)?)?)?.min(remaining)
            } else {
                remaining
            };
            Ok(json!(std::str::from_utf8(&bytes[start..start + length])?))
        }
        Some("json-valid") if args.len() == 1 => Ok(json!(
            serde_json::from_str::<Value>(string(&operand(0)?)?).is_ok()
        )),
        Some("parse-json") if args.len() == 1 => Ok(serde_json::from_str(string(&operand(0)?)?)?),
        Some("json-value") if args.len() == 3 => {
            let object = operand(0)?;
            let key = operand(1)?;
            let default = operand(2)?;
            let value = object.get(string(&key)?).unwrap_or(&default);
            let _ = string(value)?;
            Ok(value.clone())
        }
        _ => Err(invalid("unknown auto-parser assertion operator or arity")),
    }
}

fn prepare_operation(operation: &Value, assets: &Map<String, Value>) -> Result<Value> {
    let mut operation = operation.clone();
    if let Some(key) = operation.get("TemplateRef") {
        let template = asset(assets, string(key)?)?.to_owned();
        let object = operation
            .as_object_mut()
            .ok_or_else(|| invalid("auto-parser operation must be object"))?;
        object.remove("TemplateRef");
        object.insert("Template".to_owned(), json!(template));
    }
    Ok(operation)
}

fn validate_observation(operation: &Value, value: &Value) -> Result<()> {
    match text(operation, "Kind")? {
        "chat-helper" => match text(operation, "Function")? {
            "calculate-diff-split" => {
                exact_fields(value, &["prefix", "suffix", "left", "right"])?;
                for key in ["prefix", "suffix", "left", "right"] {
                    let _ = text(value, key)?;
                }
            }
            "until-common-prefix" | "after-common-suffix" => {
                exact_fields(value, &["value"])?;
                let _ = text(value, "value")?;
            }
            "segmentize-markers" => {
                exact_fields(value, &["segments"])?;
                for segment in array(value, "segments")? {
                    exact_fields(segment, &["type", "value"])?;
                    if uint(segment, "type")? > 1 {
                        return Err(invalid("invalid auto-parser segment type"));
                    }
                    let _ = text(segment, "value")?;
                }
            }
            _ => return Err(invalid("unknown auto-parser helper")),
        },
        "chat-compare-variants" => {
            exact_fields(value, &["has_value", "diff", "output_A", "output_B"])?;
            if boolean(field(value, "has_value")?)? {
                exact_fields(
                    field(value, "diff")?,
                    &["prefix", "suffix", "left", "right"],
                )?;
                for key in ["output_A", "output_B"] {
                    let _ = text(value, key)?;
                }
            } else if ["diff", "output_A", "output_B"]
                .iter()
                .any(|key| !value[*key].is_null())
            {
                return Err(invalid("empty optional chat variant has payload"));
            }
        }
        "chat-auto-trace" | "chat-peg-trace" => {
            exact_fields(value, &["Trace"])?;
            let trace = array(value, "Trace")?;
            let actions = array(operation, "Actions")?;
            if trace.len() != actions.len() {
                return Err(invalid("auto-parser action/result extent mismatch"));
            }
            for (action, value) in actions.iter().zip(trace) {
                match text(action, "Kind")? {
                    "analyze" => {
                        exact_fields(value, &["analysis"])?;
                        if !field(value, "analysis")?.is_object() {
                            return Err(invalid("auto-parser analysis must be object"));
                        }
                    }
                    "original-caps" => {
                        exact_fields(value, &["caps"])?;
                        if !field(value, "caps")?.is_object() {
                            return Err(invalid("auto-parser caps must be object"));
                        }
                    }
                    "build-parser" => exact_fields(value, &[])?,
                    "parse" => {
                        if text(operation, "Kind")? == "chat-peg-trace" {
                            exact_fields(
                                value,
                                &[
                                    "Result",
                                    "Message",
                                    "Tags",
                                    "ExceptionKind",
                                    "ExceptionMessage",
                                ],
                            )?;
                            if !field(value, "ExceptionKind")?.is_null() {
                                return Err(invalid(format!(
                                    "production chat parse exception: {}",
                                    text(value, "ExceptionMessage")?
                                )));
                            }
                        } else {
                            exact_fields(value, &["result", "message"])?;
                        }
                    }
                    _ => return Err(invalid("unknown auto-parser trace action")),
                }
            }
        }
        _ => return Err(invalid("unknown source auto-parser operation")),
    }
    Ok(())
}

fn run_scenario(
    engine: &Engine,
    scenario: &Value,
    assets: &Map<String, Value>,
    assertions: &mut Assertions,
) -> Result<bool> {
    let mut results = Vec::new();
    for operation in array(scenario, "Operations")? {
        let operation = prepare_operation(operation, assets)?;
        let Some(value) = exchange(engine, &operation)? else {
            return Ok(false);
        };
        validate_observation(&operation, &value)?;
        results.push(value);
    }
    for check in array(scenario, "Checks")? {
        if boolean(&evaluate(field(check, "When")?, &results, assets, 0)?)? {
            assertions.check(
                format!("{}: {}", text(scenario, "Name")?, text(check, "Name")?),
                evaluate(field(check, "Left")?, &results, assets, 0)?
                    == evaluate(field(check, "Right")?, &results, assets, 0)?,
            );
        }
    }
    Ok(true)
}

fn normalize_quotes(input: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(input.len() + 16);
    let mut single = false;
    let mut double = false;
    let mut index = 0;
    while index < input.len() {
        let byte = input[index];
        if byte == b'\\' && index + 1 < input.len() {
            let next = input[index + 1];
            if single {
                if next == b'\'' {
                    output.push(b'\'');
                    index += 2;
                    continue;
                }
                if next == b'"' {
                    output.extend_from_slice(b"\\\"");
                    index += 2;
                    continue;
                }
                output.extend_from_slice(&[byte, next]);
                index += 2;
                continue;
            }
            if double {
                output.extend_from_slice(&[byte, next]);
                index += 2;
                continue;
            }
            output.push(byte);
            index += 1;
            continue;
        }
        match byte {
            b'"' => {
                if single {
                    output.extend_from_slice(b"\\\"");
                } else {
                    double = !double;
                    output.push(byte);
                }
            }
            b'\'' => {
                if double {
                    output.push(byte);
                } else {
                    single = !single;
                    output.push(b'"');
                }
            }
            _ => output.push(byte),
        }
        index += 1;
    }
    output
}

fn run_normalization(fixture: &Value, assertions: &mut Assertions) -> Result<()> {
    for case in array(fixture, "Normalization")? {
        let name = text(case, "Name")?;
        let output = normalize_quotes(text(case, "Input")?.as_bytes());
        assertions.check(
            format!("{name}: source expected text"),
            output == text(case, "Expected")?.as_bytes(),
        );
        if boolean(field(case, "Embedded")?)? {
            let parsed = serde_json::from_slice::<Value>(&output);
            assertions.check(format!("{name}: valid JSON"), parsed.is_ok());
            if let Ok(parsed) = parsed {
                assertions.check(
                    format!("{name}: filename field"),
                    text(&parsed, "filename")? == "foo.cpp",
                );
                assertions.check(
                    format!("{name}: oldString embedded quotes"),
                    text(&parsed, "oldString")?.contains("\"14\""),
                );
                assertions.check(
                    format!("{name}: newString embedded quotes"),
                    text(&parsed, "newString")?.contains("\"15\""),
                );
            }
        }
    }
    Ok(())
}

/// Executes all17 default groups and their original runtime guards; the source contains no benchmark.
pub(super) fn run(engine: &Engine) -> Result<Value> {
    let fixture = fixture()?;
    let assets = field(&fixture, "Assets")?
        .as_object()
        .ok_or_else(|| invalid("auto-parser assets must be object"))?;
    let mut assertions = Assertions::default();
    let mut normalized = false;
    let mut roles = 0;
    let mut groups = Vec::new();
    for scenario in array(&fixture, "Scenarios")? {
        let group = text(scenario, "Group")?;
        if group == "tagged_args_embedded_quotes" && !normalized {
            run_normalization(&fixture, &mut assertions)?;
            normalized = true;
            groups.push("normalize_quotes_to_json");
        }
        if groups.last().copied() != Some(group) {
            groups.push(group);
        }
        if group == "role_markers_all_templates" {
            roles += 1;
        }
        if !run_scenario(engine, scenario, assets, &mut assertions)? {
            return Ok(result(ID,false,"unsupported production API: chat differential helpers, template analysis and generated parsers".to_owned()));
        }
    }
    if !normalized
        || roles != 42
        || groups.iter().copied().ne(array(&fixture, "Groups")?
            .iter()
            .map(|v| v.as_str().unwrap_or_default()))
        || assertions.count != NATIVE_ASSERTIONS + NORMALIZATION_ASSERTIONS
    {
        assertions.failures.push(format!("source success extent mismatch: normalized={normalized}, roles={roles}, groups={}, assertions={}",groups.len(),assertions.count));
    }
    Ok(result(
        ID,
        assertions.failures.is_empty(),
        format!(
            "all17groups/96subgroups;42role templates/45rawtemplate inputs;136production operations;513native+11source-local quote assertions, executed={}; failures={:?}; original unregistered Cohere helper remains excluded; no benchmarks in source",
            assertions.count, assertions.failures
        ),
    ))
}
