//! Full Rust schedule for the pinned chat-analysis diagnostic executable.
//!
//! The source has no expected-output assertions and accepts analysis exceptions.
//! Additional Rust guards compare actual production diffs with a byte reference.
//! Pinned source b87ca7225456b06b0975b8239c66b5aecf07e37c, MIT.

use super::engine::Engine;
use super::{Result, invalid, result};
use serde_json::{Value, json};

const ID: &str = "tests/test-chat-analysis.cpp";
const SOURCE_SHA: &str = "d895ae1736845640b8dab486c68189726821abfbbfc0ca96f98cd419633ad1a1";
const INPUTS: &str = include_str!("chat-analysis-inputs.json");
const TOOLS: &str = r#"[{"type":"function","function":{"name":"test_function_name","description":"A test function for debugging","parameters":{"type":"object","properties":{"param1":{"type":"string","description":"First parameter"},"param2":{"type":"string","description":"Second parameter"}},"required":["param1","param2"]}}}]"#;
const USER: &str = r#"{"role":"user","content":"Hello, please help me."}"#;
const THANKS: &str = r#"{"role":"user","content":"Thank you."}"#;
const CONTINUE: &str = r#"{"role":"user","content":"Continue."}"#;
const PLAIN: &str = r#"{"role":"assistant","content":"Let me help you."}"#;
const NO_REASON: &str = r#"{"role":"assistant","content":"I can help you with that."}"#;
const REASON: &str = r#"{"role":"assistant","content":"I can help you with that.","reasoning_content":"The user is asking for help. I should respond positively."}"#;
const ONE_TOOL: &str = r#"{"role":"assistant","content":null,"tool_calls":[{"id":"call_001","type":"function","function":{"name":"test_function_name","arguments":{"param1":"value1","param2":"value2"}}}]}"#;
const TWO_TOOLS: &str = r#"{"role":"assistant","content":null,"tool_calls":[{"id":"call_001","type":"function","function":{"name":"test_function_name","arguments":{"param1":"value1","param2":"value2"}}},{"id":"call_002","type":"function","function":{"name":"test_function_name","arguments":{"param1":"value3","param2":"value4"}}}]}"#;
const REASON_TOOL: &str = r#"{"role":"assistant","content":null,"tool_calls":[{"id":"call_001","type":"function","function":{"name":"test_function_name","arguments":{"param1":"value1","param2":"value2"}}}],"reasoning_content":"I need to call the tool first."}"#;
const VARIABLES: &str = r#"{"messages":[{"role":"user","content":"Test message"},{"role":"assistant","content":"Response","reasoning_content":"Some reasoning"}],"tools":[],"bos_token":"","eos_token":"","add_generation_prompt":false,"enable_thinking":true}"#;

fn verify(value: bool, message: &str) -> Result<()> {
    if value { Ok(()) } else { Err(invalid(message)) }
}
fn strings(value: &Value) -> Result<Vec<&str>> {
    value
        .as_array()
        .ok_or_else(|| invalid("missing string list"))?
        .iter()
        .map(|v| v.as_str().ok_or_else(|| invalid("nonstring list entry")))
        .collect()
}

// Safe replacement for the source's options parser, including its default-all
// mode, case-insensitive substring selection, file paths and rejection branches.
fn select<'a>(arguments: &'a [&str], all: &[&'a str]) -> Result<Vec<&'a str>> {
    let mut analyze_all = arguments.is_empty();
    let mut selected = Vec::new();
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index] {
            "--all" => analyze_all = true,
            "--template" => {
                index += 1;
                let pattern = arguments
                    .get(index)
                    .ok_or_else(|| invalid("--template requires an argument"))?
                    .to_ascii_lowercase();
                let matches: Vec<_> = all
                    .iter()
                    .copied()
                    .filter(|s| s.to_ascii_lowercase().contains(&pattern))
                    .collect();
                verify(!matches.is_empty(), "no templates match selection")?;
                selected.extend(matches);
            }
            "--template-file" => {
                index += 1;
                selected.push(
                    *arguments
                        .get(index)
                        .ok_or_else(|| invalid("--template-file requires an argument"))?,
                );
            }
            "--help" | "-h" => return Err(invalid("help requested")),
            _ => return Err(invalid("unknown chat-analysis option")),
        }
        index += 1;
    }
    if analyze_all {
        selected = all.to_vec();
    }
    verify(!selected.is_empty(), "no templates specified")?;
    Ok(selected)
}
fn parameters(
    messages: &[&str],
    tools: Option<&str>,
    prompt: bool,
    thinking: Option<bool>,
) -> String {
    // Preserve common_json source insertion order, including nested tool calls.
    let mut value = format!(
        r#"{{"messages":[{}],"add_generation_prompt":{prompt}"#,
        messages.join(",")
    );
    if let Some(tools) = tools {
        value.push_str(",\"tools\":");
        value.push_str(tools);
    }
    if let Some(thinking) = thinking {
        value.push_str(",\"enable_thinking\":");
        value.push_str(if thinking { "true" } else { "false" });
    }
    value.push('}');
    value
}
fn actions(names: &Value) -> Result<Vec<Value>> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    verify(
        now <= 4_102_444_800,
        "chat timestamp exceeds protocol bound",
    )?;
    let pairs = [
        (
            parameters(&[USER], Some("[]"), false, None),
            parameters(&[USER], Some(TOOLS), false, None),
        ),
        (
            parameters(&[USER], Some("[]"), false, None),
            parameters(&[USER], Some("[]"), true, None),
        ),
        (
            parameters(&[USER, NO_REASON], None, false, Some(true)),
            parameters(&[USER, REASON], None, false, Some(true)),
        ),
        (
            parameters(&[USER, NO_REASON, THANKS], None, false, Some(true)),
            parameters(&[USER, REASON, THANKS], None, false, Some(true)),
        ),
        (
            parameters(&[USER, PLAIN], Some(TOOLS), false, None),
            parameters(&[USER, ONE_TOOL], Some(TOOLS), false, None),
        ),
        (
            parameters(&[USER, PLAIN, CONTINUE], Some(TOOLS), false, None),
            parameters(&[USER, ONE_TOOL, CONTINUE], Some(TOOLS), false, None),
        ),
        (
            parameters(&[USER, ONE_TOOL], Some(TOOLS), false, None),
            parameters(&[USER, TWO_TOOLS], Some(TOOLS), false, None),
        ),
        (
            parameters(&[USER, ONE_TOOL, CONTINUE], Some(TOOLS), false, None),
            parameters(&[USER, TWO_TOOLS, CONTINUE], Some(TOOLS), false, None),
        ),
        (
            parameters(&[USER, ONE_TOOL], Some(TOOLS), false, Some(true)),
            parameters(&[USER, REASON_TOOL], Some(TOOLS), false, Some(true)),
        ),
    ];
    let mut actions = vec![json!({"Action":"caps"})];
    for (left, right) in pairs {
        let index = actions.len();
        actions.push(json!({"Action":"apply","ParametersJson":left,"NowUnixSeconds":now}));
        actions.push(json!({"Action":"apply","ParametersJson":right,"NowUnixSeconds":now}));
        actions.push(json!({"Action":"diff","Left":index,"Right":index+1}));
    }
    actions.push(json!({"Action":"undefined-probe","VariablesJson":VARIABLES,"Names":names}));
    Ok(actions)
}
fn bytes(value: &Value) -> Result<Vec<u8>> {
    let text = value.as_str().ok_or_else(|| invalid("missing hex bytes"))?;
    verify(
        text.len() % 2 == 0 && text.len() <= 2 * 1024 * 1024,
        "invalid hex byte extent",
    )?;
    text.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| {
            let nibble = |b: u8| match b {
                b'0'..=b'9' => Ok(b - b'0'),
                b'a'..=b'f' => Ok(b - b'a' + 10),
                _ => Err(invalid("invalid hex digit")),
            };
            Ok(nibble(pair[0])? * 16 + nibble(pair[1])?)
        })
        .collect()
}
#[derive(PartialEq, Debug)]
struct Diff {
    prefix: Vec<u8>,
    suffix: Vec<u8>,
    left: Vec<u8>,
    right: Vec<u8>,
}
fn segments(input: &[u8]) -> Vec<(bool, &[u8])> {
    let mut output = Vec::new();
    let mut opener = None;
    let mut start = 0;
    for (index, &byte) in input.iter().enumerate() {
        if opener.is_none() && (byte == b'<' || byte == b'[') {
            if start < index {
                output.push((false, &input[start..index]));
            }
            start = index;
            opener = Some(byte);
        } else if (opener == Some(b'<') && byte == b'>') || (opener == Some(b'[') && byte == b']') {
            output.push((true, &input[start..=index]));
            start = index + 1;
            opener = None;
        }
    }
    if start < input.len() {
        output.push((false, &input[start..]));
    }
    output
}
fn concat(parts: &[(bool, &[u8])]) -> Vec<u8> {
    parts.iter().flat_map(|(_, s)| s.iter().copied()).collect()
}
fn prepend(prefix: &[u8], value: &[u8]) -> Vec<u8> {
    prefix.iter().chain(value).copied().collect()
}
fn reference(left: &[u8], right: &[u8]) -> Diff {
    let a = segments(left);
    let b = segments(right);
    let mut result = Diff {
        prefix: Vec::new(),
        suffix: Vec::new(),
        left: Vec::new(),
        right: Vec::new(),
    };
    if a.is_empty() {
        result.right = right.to_vec();
        return result;
    }
    if b.is_empty() {
        result.left = left.to_vec();
        return result;
    }
    let (mut a_start, mut b_start, mut a_end, mut b_end) = (0, 0, a.len() - 1, b.len() - 1);
    let (mut a_consumed, mut b_consumed) = (false, false);
    while a_start != a_end && b_start != b_end {
        let prefix_advanced = if a[a_start] == b[b_start] {
            result.prefix.extend(a[a_start].1);
            a_start += 1;
            b_start += 1;
            true
        } else {
            false
        };
        let suffix_advanced = if a[a_end] == b[b_end] {
            result.suffix = prepend(a[a_end].1, &result.suffix);
            if a_start == a_end {
                a_consumed = true;
            } else {
                a_end -= 1;
            }
            if b_start == b_end {
                b_consumed = true;
            } else {
                b_end -= 1;
            }
            true
        } else {
            false
        };
        if !prefix_advanced && !suffix_advanced {
            break;
        }
    }
    let same_middle = a[a_start] == b[b_start];
    if a_start == a_end && b_start != b_end {
        if a[a_start] == b[b_end] {
            result.suffix = prepend(b[b_end].1, &result.suffix);
            b_end -= 1;
            a_consumed = true;
        } else if a[a_start] == b[b_start] {
            result.prefix.extend(b[b_start].1);
            b_start += 1;
            a_consumed = true;
        }
    } else if b_start == b_end && a_start != a_end {
        if a[a_end] == b[b_start] {
            result.suffix = prepend(a[a_end].1, &result.suffix);
            a_end -= 1;
            b_consumed = true;
        } else if a[a_start] == b[b_start] {
            result.prefix.extend(a[a_start].1);
            a_start += 1;
            b_consumed = true;
        }
    } else if a_start == a_end && b_start == b_end && same_middle && a[a_start].0 {
        result.prefix.extend(b[b_start].1);
        a_consumed = true;
        b_consumed = true;
    }
    let text_suffix = !a[a_end].0 && !b[b_end].0;
    let text_prefix = !a[a_start].0 && !b[b_start].0;
    let a_rest = concat(&a[a_start..a_end + usize::from(!a_consumed)]);
    let b_rest = concat(&b[b_start..b_end + usize::from(!b_consumed)]);
    finish_diff(
        result,
        left,
        right,
        &a_rest,
        &b_rest,
        text_suffix,
        text_prefix,
    )
}
fn finish_diff(
    mut result: Diff,
    left: &[u8],
    right: &[u8],
    a_rest: &[u8],
    b_rest: &[u8],
    text_suffix: bool,
    text_prefix: bool,
) -> Diff {
    let suffix = if text_suffix {
        a_rest
            .iter()
            .rev()
            .zip(b_rest.iter().rev())
            .take_while(|(a, b)| a == b)
            .count()
    } else {
        0
    };
    let prefix = if text_prefix {
        a_rest[..a_rest.len() - suffix]
            .iter()
            .zip(&b_rest[..b_rest.len() - suffix])
            .take_while(|(a, b)| a == b)
            .count()
    } else {
        0
    };
    result.prefix.extend(&a_rest[..prefix]);
    result.suffix = prepend(&a_rest[a_rest.len() - suffix..], &result.suffix);
    result.left = a_rest[prefix..a_rest.len() - suffix].to_vec();
    result.right = b_rest[prefix..b_rest.len() - suffix].to_vec();
    if result.left.is_empty() && result.right.is_empty() {
        result.prefix = left.to_vec();
        result.suffix.clear();
    }
    if result.left.is_empty() && !result.right.is_empty() && right.starts_with(left) {
        result.prefix = left.to_vec();
        result.suffix.clear();
        result.right = right[left.len()..].to_vec();
    }
    result
}
fn observed_diff(value: &Value) -> Result<Diff> {
    Ok(Diff {
        prefix: bytes(&value["PrefixHex"])?,
        suffix: bytes(&value["SuffixHex"])?,
        left: bytes(&value["LeftHex"])?,
        right: bytes(&value["RightHex"])?,
    })
}
fn error(value: &Value) -> Result<()> {
    verify(
        value["Kind"].as_str() == Some("standard") && value["Message"].is_string(),
        "invalid production exception observation",
    )
}
fn check_probe(observed: &Value, names: &[&str]) -> Result<usize> {
    let variables = observed["Variables"]
        .as_array()
        .ok_or_else(|| invalid("missing undefined-variable observations"))?;
    verify(
        variables.len() == names.len(),
        "undefined-variable count changed",
    )?;
    for (variable, &name) in variables.iter().zip(names) {
        verify(
            variable["Name"].as_str() == Some(name)
                && variable["Undefined"].is_boolean()
                && variable["Used"].is_boolean(),
            "undefined-variable observation lost name/order/stats",
        )?;
    }
    let runtime_error = !observed["RuntimeError"].is_null();
    if runtime_error {
        error(&observed["RuntimeError"])?;
    }
    Ok(1 + names.len() + usize::from(runtime_error))
}
fn check_trace(
    value: &Value,
    actions: &[Value],
    names: &[&str],
) -> Result<(usize, usize, Option<Value>)> {
    let trace = value["Trace"]
        .as_array()
        .ok_or_else(|| invalid("missing chat observation trace"))?;
    verify(
        trace.len() <= actions.len(),
        "extra chat observation actions",
    )?;
    let mut guards = 0;
    let mut pairs = 0;
    if !value["TemplateError"].is_null() {
        error(&value["TemplateError"])?;
        verify(
            trace.is_empty(),
            "template construction failed after executing actions",
        )?;
        return Ok((
            1,
            0,
            Some(json!({"Stage":"template","Error":value["TemplateError"]})),
        ));
    }
    for (index, (action, observed)) in actions.iter().zip(trace).enumerate() {
        if !observed["ActionError"].is_null() {
            error(&observed["ActionError"])?;
            verify(
                index + 1 == trace.len(),
                "actions continued after source's outer catch",
            )?;
            return Ok((
                guards + 1,
                pairs,
                Some(json!({"Stage":index,"Error":observed["ActionError"]})),
            ));
        }
        match action["Action"].as_str() {
            Some("caps") => {
                for key in [
                    "supports_tools",
                    "supports_tool_calls",
                    "supports_system_role",
                    "supports_parallel_tool_calls",
                    "supports_typed_content",
                    "supports_string_content",
                ] {
                    verify(
                        observed["Caps"][key].is_boolean(),
                        "missing template capability",
                    )?;
                    guards += 1;
                }
            }
            Some("apply") => {
                bytes(&observed["RenderedHex"])?;
                guards += 1;
            }
            Some("diff") => {
                let left = usize::try_from(
                    action["Left"]
                        .as_u64()
                        .ok_or_else(|| invalid("missing diff left index"))?,
                )?;
                let right = usize::try_from(
                    action["Right"]
                        .as_u64()
                        .ok_or_else(|| invalid("missing diff right index"))?,
                )?;
                let expected = reference(
                    &bytes(&trace[left]["RenderedHex"])?,
                    &bytes(&trace[right]["RenderedHex"])?,
                );
                let observed = observed_diff(&observed["Diff"])?;
                verify(
                    observed == expected,
                    "production diff differs from Rust byte/marker reference",
                )?;
                guards += 1;
                pairs += 1;
            }
            Some("undefined-probe") => {
                guards += check_probe(observed, names)?;
            }
            _ => return Err(invalid("unknown chat trace action")),
        }
    }
    verify(
        trace.len() == actions.len(),
        "truncated chat observation trace without exception",
    )?;
    Ok((guards, pairs, None))
}
pub(super) fn run(engine: &Engine) -> Result<Value> {
    let fixture: Value = serde_json::from_str(INPUTS)?;
    verify(
        fixture["SourceSHA256"].as_str() == Some(SOURCE_SHA),
        "chat-analysis source provenance changed",
    )?;
    let all = strings(&fixture["TemplatePaths"])?;
    let names = strings(&fixture["CandidateVariables"])?;
    verify(
        all.len() == 37 && names.len() == 19,
        "chat-analysis source input inventory changed",
    )?;
    let paths = select(&[], &all)?;
    let actions = actions(&fixture["CandidateVariables"])?;
    let mut guards = 0;
    let mut pairs = 0;
    let mut accepted = Vec::new();
    let mut probes = 0;
    for (index, path) in paths.iter().enumerate() {
        let source = engine.source_path(path)?;
        verify(
            std::fs::metadata(&source)?.len() <= 1024 * 1024,
            "template source exceeds bound",
        )?;
        let text = std::fs::read_to_string(source)?;
        let request = json!({"SchemaVersion":1,"Operations":[{"Kind":"chat-template-trace","Template":text,"Bos":"","Eos":"","StopOnError":true,"Actions":actions}]});
        let Some(response) =
            engine.execute_common(&format!("chat-analysis-{index:02}"), &request)?
        else {
            return Ok(result(
                ID,
                false,
                "production chat template/direct-apply/statistics API unavailable".into(),
            ));
        };
        verify(
            response["SchemaVersion"] == 1,
            "chat observation response version changed",
        )?;
        let results = response["Results"]
            .as_array()
            .ok_or_else(|| invalid("missing chat result list"))?;
        verify(results.len() == 1, "chat observation result count changed")?;
        let (count, complete, exception) = check_trace(&results[0], &actions, &names)?;
        guards += count;
        pairs += complete;
        if let Some(exception) = exception {
            accepted.push(json!({"Template":path,"Exception":exception}));
        } else {
            probes += 1;
        }
    }
    Ok(result(ID,true,json!({"SourceSHA256":SOURCE_SHA,"SourceAssertions":0,"DiagnosticOnly":true,"Templates":paths.len(),"DifferentialPairsCompleted":pairs,"ReasoningVariableProbesCompleted":probes,"CandidatesPerProbe":names.len(),"AdditionalRustGuards":guards,"SourceAcceptedExceptions":accepted}).to_string()))
}

#[cfg(test)]
mod focused_guards {
    use super::*;
    #[test]
    fn original_options_and_mutated_observations_fail_closed() -> Result<()> {
        let all = [
            "models/templates/DeepSeek.jinja",
            "models/templates/Other.jinja",
        ];
        assert_eq!(select(&[], &all)?, all);
        assert_eq!(select(&["--template", "deepseek"], &all)?, [all[0]]);
        assert_eq!(select(&["--template", "other", "--all"], &all)?, all);
        assert_eq!(
            select(&["--template-file", "mine.jinja"], &all)?,
            ["mine.jinja"]
        );
        for args in [
            &["--template"][..],
            &["--template-file"],
            &["--template", "absent"],
            &["--help"],
            &["-h"],
            &["--unknown"],
        ] {
            assert!(select(args, &all).is_err());
        }
        assert!(
            check_trace(
                &json!({"Trace":[],"TemplateError":null}),
                &[json!({"Action":"caps"})],
                &[]
            )
            .is_err()
        );
        assert!(bytes(&json!("0g")).is_err());
        assert!(bytes(&json!("0")).is_err());
        Ok(())
    }
    #[test]
    fn marker_diff_handles_utf8_bytes_and_prefix_rotation() {
        let left = b"<x>hello</x>\n";
        let right = b"<x>hello</x>\n<assistant>\n";
        let diff = reference(left, right);
        assert_eq!(diff.prefix, left);
        assert_eq!(diff.suffix, Vec::<u8>::new());
        assert_eq!(diff.right, b"<assistant>\n");
        let diff = reference(&[0xc3, 0xa9], &[0xc3, 0xa8]);
        assert_eq!(diff.prefix, [0xc3]);
        assert_eq!(diff.left, [0xa9]);
        assert_eq!(diff.right, [0xa8]);
    }
}
