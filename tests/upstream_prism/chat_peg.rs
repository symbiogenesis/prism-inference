//! Safe Rust port of all seven default test-chat-peg-parser.cpp groups.
//!
//! Source: PrismML-Eng/llama.cpp b87ca7225456b06b0975b8239c66b5aecf07e37c;
//! see tools/prism/upstream/LICENSE for MIT terms. Source-derived declarative
//! graphs and literals contain no recorded native answers. Rust owns incremental
//! token boundaries, message references, grammar comparisons and every assertion.

use std::ffi::OsStr;

use bonsai_prism_runtime::peg::Capture;
use serde_json::{Value, json};
use std::collections::BTreeMap;

use super::engine::Engine;
use super::llama_grammar::{array, exact_fields, field, uint};
use super::{Result, invalid, result};

const ID: &str = "tests/test-chat-peg-parser.cpp";
const SOURCE_SHA256: &str = "c0583c716c65fa7c06fc9b282fc8d24aecde29fa55271536e8c23dfbbc0429e0";
const CASES: &str = include_str!("chat-peg-cases.json");

pub(super) fn execute_operations(request: &Value) -> Result<Option<Value>> {
    let Some(operations) = request.get("Operations").and_then(Value::as_array) else {
        return Ok(None);
    };
    if !operations
        .iter()
        .all(|operation| operation.get("Kind").and_then(Value::as_str) == Some("chat-peg-trace"))
    {
        return Ok(None);
    }
    let results = operations
        .iter()
        .map(observe_trace)
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(json!({"SchemaVersion":1,"Results":results})))
}
fn wire_message(message: &bonsai_prism_runtime::chat::ChatMessage) -> Value {
    json!({"Role":message.role,"Content":message.content,"ReasoningContent":message.reasoning_content,"ToolCalls":message.tool_calls.iter().map(|call|json!({"Name":call.name,"Arguments":call.arguments,"Id":call.id})).collect::<Vec<_>>()})
}
fn bytes(value: &Value) -> Result<Vec<u8>> {
    value
        .as_array()
        .ok_or("chat input must be byte array")?
        .iter()
        .map(|byte| {
            u8::try_from(byte.as_u64().ok_or("chat input byte must be unsigned")?)
                .map_err(Into::into)
        })
        .collect()
}
fn tags(nodes: &[Capture], input: &[u8], output: &mut BTreeMap<String, String>) -> Result<()> {
    for node in nodes {
        if !node.tag.is_empty() {
            output.insert(
                node.tag.clone(),
                String::from_utf8(
                    input
                        .get(node.start..node.end)
                        .ok_or("chat tag extent invalid")?
                        .to_vec(),
                )?,
            );
        }
        tags(&node.children, input, output)?;
    }
    Ok(())
}

fn wire_diffs(diffs: Vec<bonsai_prism_runtime::chat::MessageDiff>) -> Value {
    json!(diffs.into_iter().map(|diff|json!({
        "ReasoningContentDelta":diff.reasoning_content_delta,
        "ContentDelta":diff.content_delta,
        "ToolCallIndex":diff.tool_call_index.map_or(u64::MAX,|index|u64::try_from(index).unwrap_or(u64::MAX)),
        "ToolCallDelta":{"Name":diff.tool_call_delta.name,"Arguments":diff.tool_call_delta.arguments,"Id":diff.tool_call_delta.id}
    })).collect::<Vec<_>>())
}
fn parse_action(
    arena: &bonsai_prism_runtime::peg::Arena,
    action: &Value,
    input: &[u8],
) -> Result<(Value, Option<bonsai_prism_runtime::chat::ChatMessage>)> {
    let mapper = text(action, "Mapper")?;
    let flags = if mapper == "tags-anywhere" {
        0
    } else {
        u8::try_from(uint(action, "Flags")?)?
    };
    let mut parsed = arena.parse_captured(input, flags, 0)?;
    if mapper == "tags-anywhere" && !input.is_empty() {
        for start in 0..input.len() {
            parsed = arena.parse_captured(input, 0, start)?;
            if parsed.result.result_type == 1 {
                break;
            }
        }
    }
    let mut message = Value::Null;
    let mut tagged = Value::Null;
    let mut retained = None;
    if matches!(mapper, "chat" | "chat-unretained") {
        let parsed_message = bonsai_prism_runtime::chat::map_captures(&parsed.nodes, input)?;
        message = wire_message(&parsed_message);
        if mapper == "chat" {
            retained = Some(parsed_message);
        }
    } else if matches!(mapper, "tags" | "tags-anywhere") {
        let mut values = BTreeMap::new();
        tags(&parsed.nodes, input, &mut values)?;
        tagged = json!(values);
    }
    Ok((
        json!({"Result":parsed.result,"Message":message,"Tags":tagged,"ExceptionKind":null,"ExceptionMessage":""}),
        retained,
    ))
}
fn observe_trace(operation: &Value) -> Result<Value> {
    use bonsai_prism_runtime::{chat::ChatMessage, peg::Arena};
    let arena = Arena::from_graph(field(operation, "Graph")?)?;
    let references = operation
        .get("ResolveRefs")
        .and_then(Value::as_array)
        .map_or_else(
            || Ok(Vec::new()),
            |values| {
                values
                    .iter()
                    .map(|value| {
                        value
                            .as_str()
                            .map(str::to_owned)
                            .ok_or_else(|| invalid("chat schema reference must be JSON text"))
                    })
                    .collect::<Result<Vec<_>>>()
            },
        )?;
    let mut messages: BTreeMap<usize, ChatMessage> = BTreeMap::new();
    let mut inputs: BTreeMap<usize, Vec<u8>> = BTreeMap::new();
    let mut trace = Vec::new();
    let clock = std::time::Instant::now();
    for action in array(operation, "Actions")? {
        let index = trace.len();
        match text(action, "Kind")? {
            "grammar" => {
                let lazy = field(action, "Lazy")?
                    .as_bool()
                    .ok_or("chat grammar Lazy must be boolean")?;
                trace.push(json!({"GrammarText":arena.grammar_with_refs(lazy,&references)?}));
            }
            "diff" => {
                let before = if action["Before"].is_null() {
                    ChatMessage::default()
                } else {
                    messages
                        .get(&usize::try_from(uint(action, "Before")?)?)
                        .ok_or("chat previous parse missing")?
                        .clone()
                };
                let after = messages
                    .get(&usize::try_from(uint(action, "After")?)?)
                    .ok_or("chat current parse missing")?;
                trace.push(match before.diff(after) {
                    Ok(diffs) => {
                        json!({"ExceptionKind":null,"Message":"","Diffs":wire_diffs(diffs)})
                    }
                    Err(error) => {
                        json!({"ExceptionKind":"standard","Message":error.to_string(),"Diffs":[]})
                    }
                });
            }
            "parse" => {
                let input = if action.get("Input").is_some() {
                    inputs
                        .get(&usize::try_from(uint(action, "Input")?)?)
                        .ok_or("chat input handle missing")?
                        .clone()
                } else {
                    bytes(field(action, "Bytes")?)?
                };
                let value = match parse_action(&arena, action, &input) {
                    Ok((value, retained)) => {
                        if let Some(message) = retained {
                            messages.insert(index, message);
                        }
                        value
                    }
                    Err(error) => {
                        json!({"Result":null,"Message":null,"Tags":null,"ExceptionKind":"standard","ExceptionMessage":error.to_string()})
                    }
                };
                trace.push(value);
            }
            "init-input" | "append-input" => {
                let handle = usize::try_from(uint(action, "Input")?)?;
                if text(action, "Kind")? == "init-input" {
                    inputs.insert(handle, Vec::new());
                }
                let input = inputs.get_mut(&handle).ok_or("chat input handle missing")?;
                input.extend(bytes(field(action, "Bytes")?)?);
                if input.len() > 1024 * 1024 {
                    return Err(invalid("chat assembled input exceeds byte bound"));
                }
                trace.push(json!({}));
            }
            "free-input" => {
                inputs
                    .remove(&usize::try_from(uint(action, "Input")?)?)
                    .ok_or("chat input handle missing")?;
                trace.push(json!({}));
            }
            "clock" => {
                trace.push(json!({"Microseconds":u64::try_from(clock.elapsed().as_micros())?}));
            }
            _ => return Err(invalid("unsupported chat PEG trace action")),
        }
    }
    Ok(json!({"Trace":trace}))
}
pub(super) fn text<'a>(value: &'a Value, name: &str) -> Result<&'a str> {
    field(value, name)?
        .as_str()
        .ok_or_else(|| invalid(format!("chat {name} must be text")))
}

#[derive(Default)]
struct Assertions {
    count: usize,
    exception_checks: usize,
    failures: Vec<String>,
}
impl Assertions {
    fn check(&mut self, name: impl Into<String>, passed: bool) -> bool {
        self.count += 1;
        if !passed {
            self.failures.push(name.into());
        }
        passed
    }
    fn no_exception(&mut self, name: impl Into<String>, value: &Value) -> Result<bool> {
        exact_fields(value, &["ExceptionKind", "Message", "Diffs"])?;
        let _ = text(value, "Message")?;
        let _ = array(value, "Diffs")?;
        self.exception_checks += 1;
        let passed = field(value, "ExceptionKind")?.is_null();
        if !passed {
            self.failures.push(name.into());
        }
        Ok(passed)
    }
}

fn fixture() -> Result<Value> {
    let value: Value = serde_json::from_str(CASES)?;
    if text(&value, "Source")? != ID
        || text(&value, "SourceSha256")? != SOURCE_SHA256
        || array(&value, "Groups")?
            != [
                json!("native"),
                json!("qwen3 coder"),
                json!("qwen3 non-coder"),
                json!("comparison"),
                json!("prefix tool names"),
                json!("tagged peg parser"),
                json!("permute"),
            ]
        || array(&value, "Native")?.len() != 9
        || array(&value, "Tagged")?.len() != 8
        || array(&value, "Permute")?.len() != 6
    {
        return Err(invalid("chat PEG source fixture extent mismatch"));
    }
    Ok(value)
}

/// Reproduces tests/peg-parser/simple-tokenize.cpp byte for byte.
pub(super) fn simple_tokenize(input: &[u8]) -> Vec<&[u8]> {
    let mut tokens = Vec::new();
    let mut start = 0;
    for (index, byte) in input.iter().enumerate() {
        if b" \n\t{},[\"].<>=/".contains(byte) && index > start {
            tokens.push(&input[start..index]);
            start = index;
        }
    }
    if start < input.len() {
        tokens.push(&input[start..]);
    }
    tokens
}

fn parse(actions: &mut Vec<Value>, bytes: &[u8], flags: u32, mapper: &str) -> usize {
    let index = actions.len();
    actions.push(json!({"Kind":"parse","Bytes":bytes,"Flags":flags,"Mapper":mapper}));
    index
}

fn grammars(scenario: &Value) -> Result<Vec<Value>> {
    if scenario.get("GrammarModes").is_none() {
        return Ok(Vec::new());
    }
    let mut actions = Vec::new();
    for lazy in array(scenario, "GrammarModes")? {
        if lazy.as_bool().is_none() {
            return Err(invalid("chat grammar mode must be boolean"));
        }
        actions.push(json!({"Kind":"grammar","Lazy":lazy}));
    }
    Ok(actions)
}

pub(super) fn exchange(engine: &Engine, operation: &Value) -> Result<Option<Value>> {
    let Some(response) = engine.execute_common(
        "chat-parser",
        &json!({"SchemaVersion":1,"Operations":[operation]}),
    )?
    else {
        return Ok(None);
    };
    exact_fields(&response, &["SchemaVersion", "Results"])?;
    if uint(&response, "SchemaVersion")? != 1 || array(&response, "Results")?.len() != 1 {
        return Err(invalid("chat parser response framing mismatch"));
    }
    Ok(Some(array(&response, "Results")?[0].clone()))
}

fn execute(engine: &Engine, scenario: &Value, actions: &[Value]) -> Result<Option<Vec<Value>>> {
    let refs = scenario
        .get("ResolveRefs")
        .cloned()
        .unwrap_or_else(|| json!([]));
    let Some(value) = exchange(
        engine,
        &json!({"Kind":"chat-peg-trace","Graph":field(scenario,"Graph")?,"ResolveRefs":refs,"Actions":actions}),
    )?
    else {
        return Ok(None);
    };
    exact_fields(&value, &["Trace"])?;
    let trace = array(&value, "Trace")?;
    if trace.len() != actions.len() {
        return Err(invalid("chat PEG trace extent mismatch"));
    }
    for (action, value) in actions.iter().zip(trace) {
        match text(action, "Kind")? {
            "grammar" => {
                exact_fields(value, &["GrammarText"])?;
                let _ = text(value, "GrammarText")?;
            }
            "parse" => {
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
                let _ = text(value, "ExceptionMessage")?;
            }
            "diff" => exact_fields(value, &["ExceptionKind", "Message", "Diffs"])?,
            "clock" => exact_fields(value, &["Microseconds"])?,
            "init-input" | "append-input" | "free-input" => exact_fields(value, &[])?,
            _ => return Err(invalid("chat PEG unknown Rust action")),
        }
    }
    Ok(Some(trace.to_vec()))
}

fn parsed(value: &Value) -> Result<&Value> {
    if !field(value, "ExceptionKind")?.is_null() {
        return Err(invalid(format!(
            "production chat parse/mapper exception: {}",
            text(value, "ExceptionMessage")?
        )));
    }
    let result = field(value, "Result")?;
    exact_fields(result, &["Type", "Start", "End"])?;
    if uint(result, "Type")? > 2 {
        return Err(invalid("invalid chat PEG parse type"));
    }
    let _ = uint(result, "Start")?;
    let _ = uint(result, "End")?;
    Ok(result)
}

fn message(value: &Value) -> Result<&Value> {
    let _ = parsed(value)?;
    let msg = field(value, "Message")?;
    exact_fields(msg, &["Role", "Content", "ReasoningContent", "ToolCalls"])?;
    for key in ["Role", "Content", "ReasoningContent"] {
        let _ = text(msg, key)?;
    }
    for tool in array(msg, "ToolCalls")? {
        exact_fields(tool, &["Name", "Arguments", "Id"])?;
        for key in ["Name", "Arguments", "Id"] {
            let _ = text(tool, key)?;
        }
    }
    Ok(msg)
}

fn check_message(
    expected: &Value,
    actual: &Value,
    name: &str,
    assertions: &mut Assertions,
) -> Result<()> {
    for (source, target) in [("Content", "Content"), ("Reasoning", "ReasoningContent")] {
        if let Some(expected) = expected.get(source) {
            assertions.check(
                format!("{name} {source}"),
                expected == field(actual, target)?,
            );
        }
    }
    let expected = array(expected, "Tools")?;
    let actual = array(actual, "ToolCalls")?;
    assertions.check(format!("{name} tool count"), expected.len() == actual.len());
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        assertions.check(
            format!("{name} tool[{index}] name"),
            text(expected, "Name")? == text(actual, "Name")?,
        );
        if expected.get("Arguments").is_some() {
            assertions.check(
                format!("{name} tool[{index}] arguments"),
                text(expected, "Arguments")? == text(actual, "Arguments")?,
            );
        }
    }
    Ok(())
}

fn prefixes(actions: &mut Vec<Value>, input: &[u8]) -> Vec<(usize, usize)> {
    let tokens = simple_tokenize(input);
    let mut accumulated = Vec::new();
    let mut previous = None;
    let mut observations = Vec::new();
    for (index, token) in tokens.iter().enumerate() {
        accumulated.extend_from_slice(token);
        let parsed = parse(
            actions,
            &accumulated,
            u32::from(index + 1 < tokens.len()),
            "chat",
        );
        let diff = actions.len();
        actions.push(json!({"Kind":"diff","Before":previous,"After":parsed}));
        observations.push((parsed, diff));
        previous = Some(parsed);
    }
    observations
}

fn check_prefixes(
    trace: &[Value],
    observations: &[(usize, usize)],
    forbidden: Option<&str>,
    name: &str,
    assertions: &mut Assertions,
) -> Result<Option<usize>> {
    let mut final_parse = None;
    for (step, (index, diff)) in observations.iter().enumerate() {
        let not_fail = assertions.check(
            format!("{name} prefix[{step}] not fail"),
            uint(parsed(&trace[*index])?, "Type")? != 0,
        );
        if forbidden.is_some() && !not_fail {
            return Ok(None);
        }
        let msg = message(&trace[*index])?;
        if let Some(forbidden) = forbidden {
            for tool in array(msg, "ToolCalls")? {
                if !assertions.check(
                    format!("{name} prefix[{step}] no short tool name"),
                    text(tool, "Name")? != forbidden,
                ) {
                    return Ok(None);
                }
            }
        }
        let no_exception = assertions.no_exception(
            format!("{name} prefix[{step}] diff exception"),
            &trace[*diff],
        )?;
        if forbidden.is_some() && !no_exception {
            return Ok(None);
        }
        final_parse = Some(*index);
    }
    Ok(final_parse)
}

fn evaluate(expression: &Value, actual: &Value, depth: usize) -> Result<Value> {
    if depth > 32 {
        return Err(invalid("chat assertion expression exceeds depth"));
    }
    if let Some(path) = expression.get("Path") {
        let mut value = actual;
        for segment in path
            .as_array()
            .ok_or_else(|| invalid("chat assertion path must be array"))?
        {
            value = if let Some(key) = segment.as_str() {
                field(value, key)?
            } else {
                value
                    .get(usize::try_from(
                        segment
                            .as_u64()
                            .ok_or_else(|| invalid("invalid chat assertion index"))?,
                    )?)
                    .ok_or_else(|| invalid("chat assertion index outside result"))?
            };
        }
        return Ok(value.clone());
    }
    let Some(op) = expression.get("Op") else {
        return Ok(expression.clone());
    };
    let args = array(expression, "Args")?;
    let value = |index| -> Result<Value> {
        evaluate(
            args.get(index)
                .ok_or_else(|| invalid("missing chat assertion operand"))?,
            actual,
            depth + 1,
        )
    };
    match op.as_str() {
        Some("equal") if args.len() == 2 => Ok(json!(value(0)? == value(1)?)),
        Some("not") if args.len() == 1 => Ok(json!(
            !value(0)?
                .as_bool()
                .ok_or_else(|| invalid("chat not requires boolean"))?
        )),
        Some("length") if args.len() == 1 => {
            let object = value(0)?;
            let len = object
                .as_object()
                .map(serde_json::Map::len)
                .or_else(|| object.as_array().map(Vec::len))
                .or_else(|| object.as_str().map(str::len))
                .ok_or_else(|| invalid("chat length requires container"))?;
            Ok(json!(len))
        }
        _ => Err(invalid("invalid chat assertion expression")),
    }
}

fn trim_leading_space(input: &str) -> String {
    let mut output = String::new();
    let mut trimming = true;
    for character in input.chars() {
        if trimming && matches!(character, ' ' | '\t' | '\r' | '\n' | '\x0b' | '\x0c') {
            continue;
        }
        output.push(character);
        trimming = character == '\n';
    }
    output
}

fn comparison_tokens(comparison: &Value) -> Result<Vec<Vec<u8>>> {
    let mut tokens = vec![b"<|START_THINKING|>".to_vec()];
    tokens.extend(
        simple_tokenize(text(comparison, "Reasoning")?.as_bytes())
            .into_iter()
            .map(<[u8]>::to_vec),
    );
    tokens.push(b"<|END_THINKING|>".to_vec());
    tokens.push(b"<|START_ACTION|>".to_vec());
    tokens.extend(
        simple_tokenize(text(comparison, "ToolCallsJson")?.as_bytes())
            .into_iter()
            .map(<[u8]>::to_vec),
    );
    tokens.push(b"<|END_ACTION|>".to_vec());
    Ok(tokens)
}

/// Runs every default correctness group; both embedded benchmarks use a separate CI entry.
fn run_native(engine: &Engine, fixture: &Value, assertions: &mut Assertions) -> Result<bool> {
    for case in array(fixture, "Native")? {
        let mut actions = grammars(case)?;
        let index = parse(&mut actions, text(case, "Input")?.as_bytes(), 0, "chat");
        let Some(trace) = execute(engine, case, &actions)? else {
            return Ok(false);
        };
        let name = text(case, "Name")?;
        assertions.check(
            format!("{name} success"),
            uint(parsed(&trace[index])?, "Type")? == 1,
        );
        check_message(
            field(case, "Expected")?,
            message(&trace[index])?,
            name,
            assertions,
        )?;
    }
    Ok(true)
}

fn run_coder(engine: &Engine, fixture: &Value, assertions: &mut Assertions) -> Result<()> {
    let coder = field(fixture, "Coder")?;
    let mut actions = grammars(coder)?;
    let observations = prefixes(&mut actions, text(coder, "Input")?.as_bytes());
    if observations.len() != 39 {
        return Err(invalid("coder source prefix extent mismatch"));
    }
    let trace =
        execute(engine, coder, &actions)?.ok_or_else(|| invalid("chat PEG backend disappeared"))?;
    let _ = check_prefixes(&trace, &observations, None, "coder", assertions)?;
    Ok(())
}

fn run_noncoder(engine: &Engine, fixture: &Value, assertions: &mut Assertions) -> Result<()> {
    let noncoder = field(fixture, "NonCoder")?;
    let mut actions = grammars(noncoder)?;
    let complete = parse(&mut actions, text(noncoder, "Input")?.as_bytes(), 0, "chat");
    let observations = prefixes(&mut actions, text(noncoder, "Input")?.as_bytes());
    if observations.len() != 43 {
        return Err(invalid("noncoder source prefix extent mismatch"));
    }
    let trace = execute(engine, noncoder, &actions)?
        .ok_or_else(|| invalid("chat PEG backend disappeared"))?;
    assertions.check(
        "noncoder complete success",
        uint(parsed(&trace[complete])?, "Type")? == 1,
    );
    check_message(
        field(noncoder, "Expected")?,
        message(&trace[complete])?,
        "noncoder complete",
        assertions,
    )?;
    let _ = check_prefixes(&trace, &observations, None, "noncoder", assertions)?;
    Ok(())
}

fn run_comparison(engine: &Engine, fixture: &Value) -> Result<()> {
    let comparison = field(fixture, "Comparison")?;
    let input = comparison_tokens(comparison)?.concat();
    let mut actions = Vec::new();
    let index = parse(&mut actions, &input, 0, "chat");
    let trace = execute(engine, comparison, &actions)?
        .ok_or_else(|| invalid("chat PEG backend disappeared"))?;
    let _ = message(&trace[index])?;
    Ok(())
}

fn run_prefix(engine: &Engine, fixture: &Value, assertions: &mut Assertions) -> Result<()> {
    let prefix = field(fixture, "Prefix")?;
    let mut actions = Vec::new();
    let long = parse(&mut actions, text(prefix, "Input")?.as_bytes(), 0, "chat");
    let observations = prefixes(&mut actions, text(prefix, "Input")?.as_bytes());
    let short = parse(
        &mut actions,
        text(prefix, "ShortInput")?.as_bytes(),
        0,
        "chat",
    );
    if observations.len() != 23 {
        return Err(invalid("tool-name source prefix extent mismatch"));
    }
    let trace = execute(engine, prefix, &actions)?
        .ok_or_else(|| invalid("chat PEG backend disappeared"))?;
    let expected_content = field(prefix, "ExpectedContent")?;
    let expected = |name| json!({"Content":expected_content,"Tools":[{"Name":name}]});
    assertions.check(
        "long tool success",
        uint(parsed(&trace[long])?, "Type")? == 1,
    );
    check_message(
        &expected(text(prefix, "ExpectedLongName")?),
        message(&trace[long])?,
        "long tool",
        assertions,
    )?;
    if let Some(last) = check_prefixes(
        &trace,
        &observations,
        Some(text(prefix, "ExpectedShortName")?),
        "long tool",
        assertions,
    )? {
        let msg = message(&trace[last])?;
        let tools = array(msg, "ToolCalls")?;
        assertions.check("final long tool count", tools.len() == 1);
        if let Some(tool) = tools.first() {
            assertions.check(
                "final long tool name",
                text(tool, "Name")? == text(prefix, "ExpectedLongName")?,
            );
        }
    }
    assertions.check(
        "short tool success",
        uint(parsed(&trace[short])?, "Type")? == 1,
    );
    check_message(
        &expected(text(prefix, "ExpectedShortName")?),
        message(&trace[short])?,
        "short tool",
        assertions,
    )?;
    Ok(())
}

fn run_tagged(engine: &Engine, fixture: &Value, assertions: &mut Assertions) -> Result<()> {
    for case in array(fixture, "Tagged")? {
        let mut actions = Vec::new();
        let index = parse(
            &mut actions,
            text(case, "Input")?.as_bytes(),
            u32::try_from(uint(case, "Flags")?)?,
            text(case, "Mapper")?,
        );
        let trace = execute(engine, case, &actions)?
            .ok_or_else(|| invalid("chat PEG backend disappeared"))?;
        let _ = parsed(&trace[index])?;
        for check in array(case, "Checks")? {
            assertions.check(
                format!("{}: {}", text(case, "Name")?, text(check, "Name")?),
                evaluate(field(check, "Left")?, &trace[index], 0)?
                    == evaluate(field(check, "Right")?, &trace[index], 0)?,
            );
        }
    }
    Ok(())
}

fn run_permute(engine: &Engine, fixture: &Value, assertions: &mut Assertions) -> Result<()> {
    for case in array(fixture, "Permute")? {
        let mut actions = Vec::new();
        let mode = text(case, "Mode")?;
        let mut indices = Vec::new();
        if mode == "permute-inputs" {
            for input in array(case, "Inputs")? {
                indices.push(parse(
                    &mut actions,
                    input
                        .as_str()
                        .ok_or_else(|| invalid("permute input must be text"))?
                        .as_bytes(),
                    0,
                    "none",
                ));
            }
        } else {
            actions.push(json!({"Kind":"grammar","Lazy":false}));
        }
        let trace = execute(engine, case, &actions)?
            .ok_or_else(|| invalid("chat PEG backend disappeared"))?;
        let name = text(case, "Name")?;
        if mode == "permute-inputs" {
            for (index, expected) in indices.iter().zip(array(case, "Expected")?) {
                assertions.check(
                    name,
                    (uint(parsed(&trace[*index])?, "Type")? == 1)
                        == expected
                            .as_bool()
                            .ok_or_else(|| invalid("permute expected must be boolean"))?,
                );
            }
        } else if mode == "permute-count" {
            let prefix = text(case, "Prefix")?;
            assertions.check(
                name,
                text(&trace[0], "GrammarText")?
                    .split('\n')
                    .filter(|line| line.starts_with(prefix))
                    .count()
                    == usize::try_from(uint(case, "ExpectedCount")?)?,
            );
        } else if mode == "permute-grammar" {
            assertions.check(
                name,
                trim_leading_space(text(case, "ExpectedGrammar")?)
                    == text(&trace[0], "GrammarText")?,
            );
        } else {
            return Err(invalid("unknown source permutation mode"));
        }
    }
    Ok(())
}

/// Runs every original default correctness group.
pub(super) fn run(engine: &Engine) -> Result<Value> {
    let fixture = fixture()?;
    let mut assertions = Assertions::default();
    if !run_native(engine, &fixture, &mut assertions)? {
        return Ok(result(
            ID,
            false,
            "unsupported production API: chat PEG builder, AST mapper and message diffs".to_owned(),
        ));
    }
    run_coder(engine, &fixture, &mut assertions)?;
    run_noncoder(engine, &fixture, &mut assertions)?;
    run_comparison(engine, &fixture)?;
    run_prefix(engine, &fixture, &mut assertions)?;
    run_tagged(engine, &fixture, &mut assertions)?;
    run_permute(engine, &fixture, &mut assertions)?;
    if assertions.count < 197 || assertions.exception_checks != 105 {
        assertions
            .failures
            .push("source assertion/exception extent mismatch".to_owned());
    }
    Ok(result(
        ID,
        assertions.failures.is_empty(),
        format!(
            "all7groups/30subgroups;9native,105incremental prefixes,8tagged,6permute; original assertions={} (197 fixed plus observed conditional tool-name checks); diff no-exception checks={}; failures={:?};100complete/20incremental benchmarks dispatched separately in CI",
            assertions.count, assertions.exception_checks, assertions.failures
        ),
    ))
}

/// Evaluates both original embedded workloads through the explicit CI benchmark path.
pub(super) fn benchmark(engine: &Engine) -> Result<Value> {
    if std::env::var_os("GITHUB_ACTIONS").as_deref() != Some(OsStr::new("true")) {
        return Err(invalid("chat PEG benchmarks run only on GitHub Actions"));
    }
    let fixture = fixture()?;
    let scenario = field(&fixture, "Comparison")?;
    let tokens = comparison_tokens(scenario)?;
    if tokens.len() != 170
        || uint(scenario, "CompleteIterations")? != 100
        || uint(scenario, "IncrementalIterations")? != 20
    {
        return Err(invalid("chat PEG benchmark workload mismatch"));
    }
    let mut measurements = Vec::new();
    for (name, iterations, incremental) in [("complete", 100, false), ("incremental", 20, true)] {
        let mut actions = Vec::new();
        let mut intervals = Vec::new();
        for _ in 0..iterations {
            let start = actions.len();
            actions.push(json!({"Kind":"clock"}));
            if incremental {
                actions.push(json!({"Kind":"init-input","Input":0,"Bytes":[]}));
                for (index, token) in tokens.iter().enumerate() {
                    actions.push(json!({"Kind":"append-input","Input":0,"Bytes":token}));
                    actions.push(json!({"Kind":"parse","Input":0,"Flags":u32::from(index+1<tokens.len()),"Mapper":"chat-unretained"}));
                }
                actions.push(json!({"Kind":"free-input","Input":0}));
            } else {
                parse(&mut actions, &tokens.concat(), 0, "chat-unretained");
            }
            let end = actions.len();
            actions.push(json!({"Kind":"clock"}));
            intervals.push((start, end));
        }
        let Some(trace) = execute(engine, scenario, &actions)? else {
            return Ok(
                json!({"Id":"tests/test-chat-peg-parser.cpp#comparison","Status":"unsupported","Detail":"unsupported production API: chat PEG parse and AST mapper"}),
            );
        };
        let mut microseconds = 0_u64;
        for (start, end) in intervals {
            let start = field(&trace[start], "Microseconds")?
                .as_i64()
                .ok_or_else(|| invalid("chat clock must be int64"))?;
            let end = field(&trace[end], "Microseconds")?
                .as_i64()
                .ok_or_else(|| invalid("chat clock must be int64"))?;
            microseconds = microseconds
                .checked_add(u64::try_from(
                    end.checked_sub(start)
                        .ok_or_else(|| invalid("chat timing subtraction overflow"))?,
                )?)
                .ok_or_else(|| invalid("chat timing sum overflow"))?;
        }
        for (action, value) in actions.iter().zip(&trace) {
            if text(action, "Kind")? == "parse" {
                let _ = message(value)?;
            }
        }
        measurements.push(json!({"Workload":name,"Iterations":iterations,"ParsesPerIteration":if incremental{170}else{1},"WallNanoseconds":microseconds.checked_mul(1000).ok_or_else(||invalid("chat timing conversion overflow"))?}));
    }
    Ok(
        json!({"Id":"tests/test-chat-peg-parser.cpp#comparison","Status":"measured","Measurements":measurements,"MeasurementScope":"native production parse and AST mapping, explicit source token-prefix appends, action dispatch and metadata handling; excludes bridge parsing and graph construction","CorrectnessThreshold":null}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn production_original_chat_peg() -> Result<()> {
        let engine = Engine::new(
            super::super::engine::Target::Rust,
            None,
            None,
            None,
            None,
            None,
        )?;
        let report = run(&engine)?;
        assert_eq!(report["Status"], "pass", "{report:?}");
        Ok(())
    }
    #[test]
    fn production_original_chat_tags_and_permutations() -> Result<()> {
        let engine = Engine::new(
            super::super::engine::Target::Rust,
            None,
            None,
            None,
            None,
            None,
        )?;
        let fixture = fixture()?;
        let mut assertions = Assertions::default();
        run_tagged(&engine, &fixture, &mut assertions)?;
        run_permute(&engine, &fixture, &mut assertions)?;
        assert!(
            assertions.failures.is_empty(),
            "{}",
            assertions.failures.join("\n")
        );
        Ok(())
    }
    #[test]
    fn production_original_chat_native_text() -> Result<()> {
        let fixture = fixture()?;
        let mut assertions = Assertions::default();
        for case in array(&fixture, "Native")?.iter().take(6) {
            let operation = json!({"Kind":"chat-peg-trace","Graph":case["Graph"],"ResolveRefs":[],"Actions":[{"Kind":"parse","Bytes":text(case,"Input")?.as_bytes(),"Flags":0,"Mapper":"chat"}]});
            let response = execute_operations(&json!({"Operations":[operation]}))?
                .ok_or("chat response missing")?;
            let value = &response["Results"][0]["Trace"][0];
            assert_eq!(parsed(value)?["Type"], 1);
            check_message(
                field(case, "Expected")?,
                message(value)?,
                text(case, "Name")?,
                &mut assertions,
            )?;
        }
        assert!(
            assertions.failures.is_empty(),
            "{}",
            assertions.failures.join("\n")
        );
        Ok(())
    }
    #[test]
    fn byte_token_boundaries_and_source_assertion_paths_are_exact() -> Result<()> {
        assert_eq!(
            simple_tokenize(b" a<b/c=\"d\" >"),
            [
                b" a".as_slice(),
                b"<b",
                b"/c",
                b"=",
                b"\"d",
                b"\"",
                b" ",
                b">"
            ]
        );
        let fixture = fixture()?;
        assert_eq!(
            simple_tokenize(text(field(&fixture, "Coder")?, "Input")?.as_bytes()).len(),
            39
        );
        assert_eq!(
            comparison_tokens(field(&fixture, "Comparison")?)?.len(),
            170
        );
        assert_eq!(
            evaluate(
                &json!({"Op":"not","Args":[{"Op":"equal","Args":[{"Path":["Result","Type"]},0]}]}),
                &json!({"Result":{"Type":2}}),
                0
            )?,
            json!(true)
        );
        assert!(evaluate(&json!({"Path":["missing"]}), &json!({}), 0).is_err());
        assert_eq!(trim_leading_space("\n  abc\n\n \tdef\n  "), "abc\ndef\n");
        Ok(())
    }
}
