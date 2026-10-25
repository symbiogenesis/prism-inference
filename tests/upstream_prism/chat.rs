//! Safe Rust ownership of the complete default tests/test-chat.cpp contract.
//!
//! Source: PrismML-Eng/llama.cpp b87ca7225456b06b0975b8239c66b5aecf07e37c;
//! see tools/prism/upstream/LICENSE for MIT terms. Fixtures contain source inputs,
//! expected literals and original condition guards, including all template bytes.
//! Every byte-prefix parse, diff accumulation, grammar decision and reconstruction
//! assertion is Rust-owned. Native code only exposes generic production calls.

use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::ser::{SerializeMap, SerializeSeq};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Number, Value, json};

use super::engine::Engine;
use super::llama_grammar::{array, exact_fields, field, uint};
use super::{Result, invalid, result};

const ID: &str = "tests/test-chat.cpp";
const SHA: &str = "9920d484dfe35d8ccec4e34f3dd6a23e1788a479a419cab231c361d3cad4e8b4";
const CASES: &str = include_str!("chat-cases.json");
const GROUPS: &[&str] = &[
    "test_msg_diffs_compute",
    "test_msgs_oaicompat_json_conversion",
    "test_msg_token_delimiters_split",
    "test_tools_oaicompat_json_conversion",
    "test_convert_responses_to_chatcmpl",
    "test_developer_role_to_system_workaround",
    "test_deepseek_v4_thinking_retention",
    "test_deepseek_v4_tool_result_ordering",
    "test_template_generation_prompt",
    "test_reasoning_effort_caps",
    "test_reasoning_budget_tokens_per_request",
    "test_reasoning_budget_message_per_request",
    "test_template_output_peg_parsers",
    "test_chat_malformed_utf8",
];

#[derive(Default)]
struct Assertions {
    fixed: usize,
    cases: usize,
    prefixes: usize,
    final_messages: usize,
    reconstructions: usize,
    grammar_builds: usize,
    grammar_matches: usize,
    grammar_triggers: usize,
    forced_mismatches: usize,
    failures: Vec<String>,
}
impl Assertions {
    fn check(&mut self, name: &str, passed: bool) {
        if !passed && self.failures.len() < 64 {
            self.failures.push(name.to_owned());
        }
    }
}
fn text<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    field(v, key)?
        .as_str()
        .ok_or_else(|| invalid(format!("chat {key} must be text")))
}
fn boolean(v: &Value) -> Result<bool> {
    v.as_bool()
        .ok_or_else(|| invalid("chat value must be boolean"))
}
fn byte_array(v: &Value) -> Result<Vec<u8>> {
    v.as_array()
        .ok_or_else(|| invalid("chat bytes must be array"))?
        .iter()
        .map(|x| {
            Ok(u8::try_from(
                x.as_u64().ok_or_else(|| invalid("invalid chat byte"))?,
            )?)
        })
        .collect()
}
fn bytes(v: &Value) -> Result<Vec<u8>> {
    if let Some(s) = v.as_str() {
        return Ok(s.as_bytes().to_vec());
    }
    byte_array(field(v, "Bytes")?)
}
fn validate_template_kwargs(value: &Value) -> Result<()> {
    if let Some(encoded) = value.get("BodyJson") {
        let body: Value = serde_json::from_str(
            encoded
                .as_str()
                .ok_or_else(|| invalid("chat BodyJson must be a string"))?,
        )?;
        if !body.is_object() || !body["messages"].is_array() {
            return Err(invalid(
                "chat server body must preserve the source JSON object",
            ));
        }
    }
    if let Some(encoded) = value.get("ParametersJson") {
        validate_template_kwargs(&serde_json::from_str::<Value>(
            encoded
                .as_str()
                .ok_or_else(|| invalid("chat ParametersJson must be a string"))?,
        )?)?;
    }
    if let Some(kwargs) = value.get("chat_template_kwargs") {
        let kwargs = kwargs
            .as_object()
            .ok_or_else(|| invalid("chat template kwargs must be an object"))?;
        if kwargs.values().any(|value| !value.is_string()) {
            return Err(invalid(
                "chat template kwargs must preserve the source map<string,string> values",
            ));
        }
    }
    match value {
        Value::Array(values) => {
            for value in values {
                validate_template_kwargs(value)?;
            }
        }
        Value::Object(values) => {
            for value in values.values() {
                validate_template_kwargs(value)?;
            }
        }
        _ => {}
    }
    Ok(())
}
fn fixture() -> Result<Value> {
    let v: Value = serde_json::from_str(CASES)?;
    let cases = array(&v, "Cases")?;
    let groups = array(&v, "Groups")?;
    if text(&v, "Source")? != ID
        || text(&v, "SourceSha256")? != SHA
        || array(&v, "DefaultGroups")? != GROUPS.iter().map(|s| json!(s)).collect::<Vec<_>>()
        || groups.len() != 14
        || cases.len() != 431
        || field(&v, "Templates")?
            .as_object()
            .map(serde_json::Map::len)
            != Some(59)
        || cases.iter().try_fold(0usize, |n, c| {
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>(n + array(c, "InputBytes")?.len())
        })? != 44162
        || groups.iter().try_fold(0usize, |n, g| {
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>(n + array(g, "Checks")?.len())
        })? != 363
        || cases.iter().filter(|c| c["IsPartial"] == true).count() != 9
        || cases
            .iter()
            .filter(|c| c["ExpectReconstruction"] == true && c["IsPartial"] == false)
            .count()
            != 82
    {
        return Err(invalid("chat source fixture extent mismatch"));
    }
    for (name, group) in GROUPS.iter().zip(groups) {
        if text(group, "Name")? != *name {
            return Err(invalid("chat group order mismatch"));
        }
    }
    validate_template_kwargs(&v)?;
    Ok(v)
}
fn exchange(engine: &Engine, operation: &Value) -> Result<Option<Value>> {
    let Some(v) =
        engine.execute_common("chat", &json!({"SchemaVersion":1,"Operations":[operation]}))?
    else {
        return Ok(None);
    };
    exact_fields(&v, &["SchemaVersion", "Results"])?;
    if uint(&v, "SchemaVersion")? != 1 || array(&v, "Results")?.len() != 1 {
        return Err(invalid("chat response framing mismatch"));
    }
    Ok(Some(array(&v, "Results")?[0].clone()))
}

// common_json keeps insertion order. A separate ordered representation preserves
// the source's JSON dump comparison, rather than sorting object keys in Value.
#[derive(Clone, Debug)]
enum Ordered {
    Scalar(Value),
    Array(Vec<Self>),
    Object(Vec<(String, Self)>),
}
impl Serialize for Ordered {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        match self {
            Self::Scalar(v) => v.serialize(s),
            Self::Array(v) => {
                let mut out = s.serialize_seq(Some(v.len()))?;
                for x in v {
                    out.serialize_element(x)?;
                }
                out.end()
            }
            Self::Object(v) => {
                let mut out = s.serialize_map(Some(v.len()))?;
                for (k, x) in v {
                    out.serialize_entry(k, x)?;
                }
                out.end()
            }
        }
    }
}
struct OrderedVisitor;
impl<'de> Visitor<'de> for OrderedVisitor {
    type Value = Ordered;
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("JSON value")
    }
    fn visit_bool<E: serde::de::Error>(self, v: bool) -> std::result::Result<Ordered, E> {
        Ok(Ordered::Scalar(json!(v)))
    }
    fn visit_i64<E: serde::de::Error>(self, v: i64) -> std::result::Result<Ordered, E> {
        Ok(Ordered::Scalar(json!(v)))
    }
    fn visit_u64<E: serde::de::Error>(self, v: u64) -> std::result::Result<Ordered, E> {
        Ok(Ordered::Scalar(json!(v)))
    }
    fn visit_f64<E: serde::de::Error>(self, v: f64) -> std::result::Result<Ordered, E> {
        Number::from_f64(v)
            .map(|n| Ordered::Scalar(Value::Number(n)))
            .ok_or_else(|| E::custom("nonfinite JSON number"))
    }
    fn visit_str<E: serde::de::Error>(self, v: &str) -> std::result::Result<Ordered, E> {
        Ok(Ordered::Scalar(json!(v)))
    }
    fn visit_string<E: serde::de::Error>(self, v: String) -> std::result::Result<Ordered, E> {
        Ok(Ordered::Scalar(json!(v)))
    }
    fn visit_none<E: serde::de::Error>(self) -> std::result::Result<Ordered, E> {
        Ok(Ordered::Scalar(Value::Null))
    }
    fn visit_unit<E: serde::de::Error>(self) -> std::result::Result<Ordered, E> {
        self.visit_none()
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> std::result::Result<Ordered, A::Error> {
        let mut out = Vec::new();
        while let Some(v) = a.next_element()? {
            out.push(v);
        }
        Ok(Ordered::Array(out))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> std::result::Result<Ordered, A::Error> {
        let mut out: Vec<(String, Ordered)> = Vec::new();
        while let Some((key, v)) = a.next_entry::<String, Ordered>()? {
            if let Some((_, old)) = out.iter_mut().find(|(k, _)| *k == key) {
                *old = v;
            } else {
                out.push((key, v));
            }
        }
        Ok(Ordered::Object(out))
    }
}
impl<'de> Deserialize<'de> for Ordered {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        d.deserialize_any(OrderedVisitor)
    }
}
fn normalized_json(v: &str) -> String {
    serde_json::from_str::<Ordered>(v)
        .ok()
        .and_then(|x| serde_json::to_string(&x).ok())
        .unwrap_or_default()
}
fn strip(v: &str) -> &str {
    v.trim_matches(|c: char| matches!(c, ' ' | '\t' | '\n' | '\r' | '\x0b' | '\x0c'))
}
fn message_equal(expected: &Value, actual: &Value, ignore: bool) -> Result<bool> {
    let same = |key: &str| -> Result<bool> {
        let e = text(expected, key)?;
        let a = text(actual, key)?;
        Ok(if ignore { strip(e) == strip(a) } else { e == a })
    };
    if text(expected, "role")? != text(actual, "role")? || !same("content")? {
        return Ok(false);
    }
    let ep = array(expected, "content_parts")?;
    let ap = array(actual, "content_parts")?;
    if ep.len() != ap.len() {
        return Ok(false);
    }
    for (e, a) in ep.iter().zip(ap) {
        if text(e, "type")? != text(a, "type")?
            || (if ignore {
                strip(text(e, "text")?) != strip(text(a, "text")?)
            } else {
                text(e, "text")? != text(a, "text")?
            })
        {
            return Ok(false);
        }
    }
    if !same("reasoning_content")? {
        return Ok(false);
    }
    let et = array(expected, "tool_calls")?;
    let at = array(actual, "tool_calls")?;
    if et.len() != at.len() {
        return Ok(false);
    }
    for (e, a) in et.iter().zip(at) {
        if text(e, "name")? != text(a, "name")?
            || text(e, "id")? != text(a, "id")?
            || normalized_json(text(e, "arguments")?) != normalized_json(text(a, "arguments")?)
        {
            return Ok(false);
        }
    }
    Ok(true)
}
fn diff_equal(e: &Value, a: &Value) -> Result<bool> {
    let e = e
        .as_array()
        .ok_or_else(|| invalid("expected diffs must be array"))?;
    let a = a
        .as_array()
        .ok_or_else(|| invalid("actual diffs must be array"))?;
    if e.len() != a.len() {
        return Ok(false);
    }
    for (e, a) in e.iter().zip(a) {
        // The original common_chat_msg_diff::operator== omits reasoning_content_delta.
        // Source literals may be byte wrappers while the native wire uses strings.
        if bytes(field(e, "content_delta")?)? != bytes(field(a, "content_delta")?)?
            || field(e, "tool_call_index")? != field(a, "tool_call_index")?
        {
            return Ok(false);
        }
        let expected_call = field(e, "tool_call_delta")?;
        let actual_call = field(a, "tool_call_delta")?;
        for key in ["name", "arguments", "id"] {
            if bytes(field(expected_call, key)?)? != bytes(field(actual_call, key)?)? {
                return Ok(false);
            }
        }
    }
    Ok(true)
}
#[allow(
    clippy::too_many_lines,
    reason = "Keep the source helper control flow together for coverage review."
)]
fn eval(v: &Value, results: &[Value]) -> Result<Value> {
    if let Some(index) = v.get("Result") {
        let index = usize::try_from(
            index
                .as_u64()
                .ok_or_else(|| invalid("chat result index invalid"))?,
        )?;
        let mut x = results
            .get(index)
            .ok_or_else(|| invalid("chat result reference unavailable"))?;
        for key in array(v, "Path")? {
            x = if let Some(k) = key.as_str() {
                field(x, k)?
            } else {
                x.as_array()
                    .and_then(|a| {
                        key.as_u64()
                            .and_then(|i| usize::try_from(i).ok())
                            .and_then(|i| a.get(i))
                    })
                    .ok_or_else(|| invalid("chat result path invalid"))?
            };
        }
        return Ok(x.clone());
    }
    if let Some(name) = v.get("Op") {
        let name = name
            .as_str()
            .ok_or_else(|| invalid("chat expression opcode invalid"))?;
        let args = array(v, "Args")?;
        let arg = |i: usize| -> Result<Value> {
            eval(
                args.get(i)
                    .ok_or_else(|| invalid("chat expression argument missing"))?,
                results,
            )
        };
        if name == "and" {
            return Ok(json!(boolean(&arg(0)?)? && boolean(&arg(1)?)?));
        }
        if name == "or" {
            return Ok(json!(boolean(&arg(0)?)? || boolean(&arg(1)?)?));
        }
        if name == "not" {
            return Ok(json!(!boolean(&arg(0)?)?));
        }
        let a = arg(0)?;
        return Ok(match name {
            "equal" | "not-equal" => {
                let b = arg(1)?;
                let same = if a.get("Bytes").is_some() || b.get("Bytes").is_some() {
                    bytes(&a)? == bytes(&b)?
                } else {
                    a == b
                };
                json!(same == (name == "equal"))
            }
            "diff-equal" => json!(diff_equal(&a, &arg(1)?)?),
            "message-equal" => json!(message_equal(&a, &arg(1)?, boolean(&arg(2)?)?)?),
            "length" => json!(if let Some(a) = a.as_array() {
                a.len()
            } else if let Some(o) = a.as_object() {
                if o.contains_key("Bytes") {
                    bytes(&a)?.len()
                } else {
                    o.len()
                }
            } else {
                bytes(&a)?.len()
            }),
            "contains" | "not-contains" | "ends-with" | "find" => {
                let a = bytes(&a)?;
                let b = bytes(&arg(1)?)?;
                let pos = find(&a, &b);
                match name {
                    "contains" => json!(pos.is_some()),
                    "not-contains" => json!(pos.is_none()),
                    "ends-with" => json!(a.ends_with(&b)),
                    _ => json!(pos.and_then(|i| i64::try_from(i).ok()).unwrap_or(-1)),
                }
            }
            "greater" => json!(
                a.as_i64()
                    .ok_or_else(|| invalid("comparison requires integer"))?
                    > arg(1)?
                        .as_i64()
                        .ok_or_else(|| invalid("comparison requires integer"))?
            ),
            "has-field" => json!(a.get(String::from_utf8(bytes(&arg(1)?)?)?).is_some()),
            "is-array" => json!(a.is_array()),
            "is-user-start" => {
                let position = arg(1)?;
                json!(
                    a.as_array()
                        .ok_or_else(|| invalid("spans must be array"))?
                        .iter()
                        .any(|s| s["role"] == 3 && s["pos"] == position)
                )
            }
            "field" => {
                let k = arg(1)?;
                field(&a, k.as_str().ok_or_else(|| invalid("field name invalid"))?)?.clone()
            }
            "parse-json" => json!({"ParsedJson":String::from_utf8(bytes(&a)?)?}),
            "dump" => {
                let raw = text(&a, "ParsedJson")?;
                let ordered: Ordered = serde_json::from_str(raw)?;
                if args.len() == 2 && arg(1)? == 2 {
                    json!(serde_json::to_string_pretty(&ordered)?)
                } else {
                    json!(serde_json::to_string(&ordered)?)
                }
            }
            _ => return Err(invalid(format!("unknown chat expression {name}"))),
        });
    }
    if let Some(values) = v.as_array() {
        return values
            .iter()
            .map(|x| eval(x, results))
            .collect::<Result<Vec<_>>>()
            .map(Value::Array);
    }
    if let Some(values) = v.as_object() {
        let mut output = serde_json::Map::new();
        for (k, x) in values {
            output.insert(k.clone(), eval(x, results)?);
        }
        return Ok(Value::Object(output));
    }
    Ok(v.clone())
}
fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        Some(0)
    } else {
        haystack.windows(needle.len()).position(|w| w == needle)
    }
}
fn materialize(operation: &Value, fixture: &Value, results: &[Value]) -> Result<Value> {
    let mut operation = eval(operation, results)?;
    if let Some(path) = operation
        .get("TemplateRef")
        .and_then(Value::as_str)
        .map(str::to_owned)
    {
        let template = field(field(fixture, "Templates")?, &path)?.clone();
        operation
            .as_object_mut()
            .ok_or_else(|| invalid("chat operation must be object"))?
            .remove("TemplateRef");
        operation["Template"] = template;
    }
    Ok(operation)
}
fn empty_message() -> Value {
    json!({"role":"assistant","content":"","content_parts":[],"tool_calls":[],"reasoning_content":"","tool_name":"","tool_call_id":""})
}
fn accumulate(message: &mut Value, diffs: &[Value]) -> Result<()> {
    for diff in diffs {
        for (key, delta) in [
            ("reasoning_content", "reasoning_content_delta"),
            ("content", "content_delta"),
        ] {
            let mut s = text(message, key)?.to_owned();
            s.push_str(text(diff, delta)?);
            message[key] = json!(s);
        }
        if let Some(index) = field(diff, "tool_call_index")?.as_u64() {
            let index = usize::try_from(index)?;
            if index > 65536 {
                return Err(invalid("chat diff index exceeds bound"));
            }
            let calls = message["tool_calls"]
                .as_array_mut()
                .ok_or_else(|| invalid("accumulated calls must be array"))?;
            while calls.len() <= index {
                calls.push(json!({"name":"","arguments":"","id":""}));
            }
            let call = &mut calls[index];
            let delta = field(diff, "tool_call_delta")?;
            for key in ["name", "id"] {
                let value = text(delta, key)?;
                if !value.is_empty() {
                    call[key] = json!(value);
                }
            }
            let mut arguments = text(call, "arguments")?.to_owned();
            arguments.push_str(text(delta, "arguments")?);
            call["arguments"] = json!(arguments);
        }
    }
    Ok(())
}
// Exact source byte helper, including malformed trailing sequences.
fn safe_prefix_len(s: &[u8]) -> usize {
    let len = s.len();
    if len == 0 {
        return 0;
    }
    let mut i = len;
    for _ in 0..4 {
        if i == 0 {
            break;
        }
        i -= 1;
        let c = s[i];
        if c & 0x80 == 0 {
            return len;
        }
        if c & 0xc0 == 0xc0 {
            let expected = if c & 0xe0 == 0xc0 {
                2
            } else if c & 0xf0 == 0xe0 {
                3
            } else if c & 0xf8 == 0xf0 {
                4
            } else {
                return i;
            };
            return if len - i >= expected { len } else { i };
        }
    }
    len - len.min(3)
}
fn trace<'a>(v: &'a Value, actions: &[Value]) -> Result<&'a [Value]> {
    exact_fields(v, &["Trace"])?;
    let out = array(v, "Trace")?;
    if out.len() != actions.len() {
        return Err(invalid("chat native trace extent mismatch"));
    }
    Ok(out)
}
fn successful_message(v: &Value) -> Result<&Value> {
    if !field(v, "ExceptionKind")?.is_null() {
        return Err(invalid(format!(
            "chat parse raised {}: {}",
            v["ExceptionKind"], v["Message"]
        )));
    }
    let message = field(v, "Message")?;
    for key in [
        "role",
        "content",
        "reasoning_content",
        "tool_name",
        "tool_call_id",
    ] {
        let _ = text(message, key)?;
    }
    let _ = array(message, "content_parts")?;
    let _ = array(message, "tool_calls")?;
    Ok(message)
}

#[allow(
    clippy::too_many_lines,
    reason = "Keep the source helper control flow together for coverage review."
)]
fn trigger_position(
    engine: &Engine,
    params: &Value,
    input: &[u8],
    expected_tools: bool,
) -> Result<Option<usize>> {
    let prompt = text(params, "generation_prompt")?.as_bytes();
    let mut full = prompt.to_vec();
    full.extend_from_slice(input);
    let triggers = array(params, "grammar_triggers")?;
    let ends = array(params, "thinking_end_tags")?;
    let mut budget = boolean(field(params, "grammar_lazy")?)?
        && !ends.is_empty()
        && triggers.iter().all(|t| t["type"] == 1);
    let mut earliest = None;
    if budget {
        let start = text(params, "thinking_start_tag")?.as_bytes();
        let mut thinking = false;
        let mut i = 0;
        while i < full.len() {
            if !thinking && !start.is_empty() && full[i..].starts_with(start) {
                thinking = true;
                i += start.len();
                continue;
            }
            if thinking {
                let mut end_width = 0;
                for end in ends {
                    let end = end
                        .as_str()
                        .ok_or_else(|| invalid("thinking end must be string"))?
                        .as_bytes();
                    if full[i..].starts_with(end) {
                        thinking = false;
                        end_width = end.len();
                        break;
                    }
                }
                i += if end_width == 0 { 1 } else { end_width };
                continue;
            }
            if triggers.iter().any(|t| {
                t["value"]
                    .as_str()
                    .is_some_and(|s| full[i..].starts_with(s.as_bytes()))
            }) {
                earliest = Some(i);
                break;
            }
            i += 1;
        }
        if earliest.is_none() && expected_tools {
            budget = false;
        }
    }
    if budget {
        return Ok(earliest);
    }
    let mut patterns = Vec::new();
    let mut actions = Vec::new();
    let mut ranges = Vec::new();
    for trigger in triggers {
        let kind = uint(trigger, "type")?;
        let value = text(trigger, "value")?;
        match kind {
            1 => {
                if let Some(p) = find(&full, value.as_bytes()) {
                    earliest = Some(earliest.map_or(p, |e| e.min(p)));
                }
            }
            2 | 3 => {
                let index = patterns.len();
                patterns.push(json!(value));
                let start = actions.len();
                if kind == 2 {
                    actions.push(json!({"Pattern":index,"Function":"search","Bytes":full}));
                } else {
                    for end in prompt.len()..=full.len() {
                        actions
                            .push(json!({"Pattern":index,"Function":"match","Bytes":&full[..end]}));
                    }
                }
                ranges.push((kind, start, actions.len()));
            }
            _ => return Err(invalid("unknown source chat grammar trigger type")),
        }
    }
    if !actions.is_empty() {
        let Some(response) = exchange(
            engine,
            &json!({"Kind":"chat-regex","Patterns":patterns,"Actions":actions}),
        )?
        else {
            return Err(invalid("chat regex production API disappeared"));
        };
        let observations = trace(&response, &actions)?;
        for (kind, start, end) in ranges {
            for value in &observations[start..end] {
                if !boolean(field(value, "Matched")?)? {
                    continue;
                }
                let captures = array(value, "Captures")?;
                let capture = if kind == 2 {
                    captures
                        .get(usize::try_from(uint(value, "MarkCount")?)?)
                        .ok_or_else(|| invalid("regex last capture unavailable"))?
                } else {
                    captures
                        .iter()
                        .skip(1)
                        .find(|c| c["Length"].as_i64().is_some_and(|v| v > 0))
                        .or_else(|| captures.first())
                        .ok_or_else(|| invalid("regex match has no capture"))?
                };
                if let Some(position) = field(capture, "Position")?.as_i64().filter(|p| *p >= 0) {
                    let position = usize::try_from(position)?;
                    earliest = Some(earliest.map_or(position, |e| e.min(position)));
                }
                break;
            }
        }
    }
    Ok(earliest)
}
// Same legacy decoder as unicode_cpts_from_utf8: an invalid byte advances one
// position and yields U+FFFD; overlong sequences are decoded without extra checks.
fn codepoints(bytes: &[u8]) -> Vec<u32> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        let first = bytes[at];
        let (width, mask) = if first & 0x80 == 0 {
            (1, 0x7f)
        } else if first & 0xe0 == 0xc0 {
            (2, 0x1f)
        } else if first & 0xf0 == 0xe0 {
            (3, 0x0f)
        } else if first & 0xf8 == 0xf0 {
            (4, 0x07)
        } else {
            (0, 0)
        };
        if width == 0
            || at + width > bytes.len()
            || bytes[at + 1..at + width].iter().any(|b| b & 0xc0 != 0x80)
        {
            out.push(0xfffd);
            at += 1;
            continue;
        }
        let mut point = u32::from(first & mask);
        for b in &bytes[at + 1..at + width] {
            point = (point << 6) | u32::from(b & 0x3f);
        }
        out.push(point);
        at += width;
    }
    out
}
fn check_grammar(
    engine: &Engine,
    case: &Value,
    params: &Value,
    stats: &mut Assertions,
) -> Result<()> {
    let grammar = text(params, "grammar")?;
    if grammar.is_empty() {
        return Ok(());
    }
    let input = byte_array(field(case, "InputBytes")?)?;
    let expected_tools = !array(field(case, "Expect")?, "tool_calls")?.is_empty();
    let partial = boolean(field(case, "IsPartial")?)?;
    let earliest = trigger_position(engine, params, &input, expected_tools)?;
    let lazy = boolean(field(params, "grammar_lazy")?)?;
    if lazy && expected_tools && !partial {
        stats.grammar_triggers += 1;
        stats.check("lazy tool grammar trigger must fire", earliest.is_some());
    }
    let prompt = text(params, "generation_prompt")?.as_bytes();
    let mut full = prompt.to_vec();
    full.extend_from_slice(&input);
    let constrained = earliest.map_or(full.as_slice(), |position| {
        &full[position.max(prompt.len())..]
    });
    let match_grammar = (earliest.is_some() || !lazy) && !partial;
    let mut actions = vec![
        json!({"Kind":"init-text","Grammar":0,"GrammarText":grammar,"Root":"root"}),
        json!({"Kind":"state","Grammar":0}),
    ];
    if match_grammar {
        for point in codepoints(constrained) {
            actions.push(json!({"Kind":"accept-codepoint","Grammar":0,"CodePoint":point}));
        }
    }
    actions.push(json!({"Kind":"free","Grammar":0}));
    let Some(response) = exchange(engine, &json!({"Kind":"grammar-trace","Actions":actions}))?
    else {
        return Err(invalid("chat grammar production API disappeared"));
    };
    let events = trace(&response, &actions)?;
    stats.grammar_builds += 1;
    let created = boolean(field(&events[0], "Created")?)?;
    stats.check("generated grammar must build", created);
    if match_grammar {
        stats.grammar_matches += 1;
        let mut sizes = array(&events[1], "StackSizes")?;
        let mut matched = true;
        // The source stops at the first empty stack set. Later pure observations
        // already present in the transport response cannot turn a failure green.
        for event in &events[2..events.len() - 1] {
            if !field(event, "ExceptionKind")?.is_null() {
                return Err(invalid("direct grammar accept raised exception"));
            }
            sizes = array(event, "StackSizes")?;
            if sizes.is_empty() {
                matched = false;
                break;
            }
        }
        stats.check(
            "constrained generated text must complete grammar",
            matched && sizes.iter().any(|s| s.as_u64() == Some(0)),
        );
    }
    Ok(())
}
#[allow(
    clippy::too_many_lines,
    reason = "Keep the source helper control flow together for coverage review."
)]
fn run_case(
    engine: &Engine,
    fixture: &Value,
    case: &Value,
    stats: &mut Assertions,
    force_reconstruction: bool,
) -> Result<bool> {
    let input = byte_array(field(case, "InputBytes")?)?;
    let partial = boolean(field(case, "IsPartial")?)?;
    let mut parameters = field(case, "Parameters")?.clone();
    let milliseconds = u64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis())?;
    parameters["now_milliseconds"] = json!(milliseconds);
    let template = field(field(fixture, "Templates")?, text(case, "TemplateRef")?)?;
    let mut actions =
        vec![json!({"Kind":"apply","ParametersJson":serde_json::to_string(&parameters)?})];
    let mut previous = Value::Null;
    let mut pairs = Vec::new();
    for end in 1..=input.len() {
        let safe = safe_prefix_len(&input[..end]);
        let parse = actions.len();
        actions.push(json!({"Kind":"parse","Bytes":&input[..safe],"Partial":end<input.len()||partial,"Store":parse}));
        let diff = actions.len();
        actions.push(json!({"Kind":"diff","Before":previous,"After":parse}));
        previous = json!(parse);
        pairs.push((parse, diff));
    }
    let final_parse = if partial {
        None
    } else {
        let index = actions.len();
        actions.push(json!({"Kind":"parse","Bytes":input,"Partial":false,"Store":index}));
        Some(index)
    };
    let reconstruction_parse =
        if (boolean(field(case, "ExpectReconstruction")?)? || force_reconstruction) && !partial {
            let index = actions.len();
            actions.push(json!({"Kind":"parse","Bytes":input,"Partial":false,"Store":index}));
            Some(index)
        } else {
            None
        };
    let Some(response) = exchange(
        engine,
        &json!({"Kind":"chat-trace","Template":template,"Actions":actions}),
    )?
    else {
        return Ok(false);
    };
    let observations = trace(&response, &actions)?;
    let mut accumulated = empty_message();
    for (parse, diff) in pairs {
        let current = successful_message(&observations[parse])?;
        accumulate(&mut accumulated, array(&observations[diff], "Diffs")?)?;
        stats.prefixes += 1;
        stats.check(
            "streaming diff accumulation equals current message",
            message_equal(current, &accumulated, true)?,
        );
    }
    let expected = field(case, "Expect")?;
    if let Some(index) = final_parse {
        stats.final_messages += 1;
        stats.check(
            "complete final parse equals source expectation",
            message_equal(expected, successful_message(&observations[index])?, true)?,
        );
    }
    stats.final_messages += 1;
    stats.check(
        "final accumulated message equals source expectation",
        message_equal(expected, &accumulated, true)?,
    );
    check_grammar(engine, case, &observations[0], stats)?;
    if (boolean(field(case, "ExpectReconstruction")?)? || force_reconstruction) && !partial {
        // Source copies the expected message, replacing only argument spelling
        // from actual parser output; all control flow remains in Rust.
        let parsed = successful_message(
            &observations
                [reconstruction_parse.ok_or_else(|| invalid("reconstruction missing parse"))?],
        )?;
        let mut message = expected.clone();
        let actual_calls = array(parsed, "tool_calls")?;
        for (i, call) in message["tool_calls"]
            .as_array_mut()
            .ok_or_else(|| invalid("expected calls must be array"))?
            .iter_mut()
            .enumerate()
        {
            if let Some(actual) = actual_calls.get(i) {
                call["arguments"] = field(actual, "arguments")?.clone();
            }
        }
        parameters["messages"]
            .as_array_mut()
            .ok_or_else(|| invalid("case messages must be array"))?
            .push(message);
        parameters["add_generation_prompt"] = json!(false);
        let reconstruct_actions =
            vec![json!({"Kind":"apply","ParametersJson":serde_json::to_string(&parameters)?})];
        let Some(reconstructed) = exchange(
            engine,
            &json!({"Kind":"chat-trace","Template":template,"Actions":reconstruct_actions}),
        )?
        else {
            return Err(invalid("chat reconstruction API disappeared"));
        };
        let reconstructed = trace(&reconstructed, &reconstruct_actions)?;
        let mut expected_text = text(&observations[0], "prompt")?.as_bytes().to_vec();
        expected_text.extend_from_slice(&input);
        stats.reconstructions += 1;
        let matched = text(&reconstructed[0], "prompt")?
            .as_bytes()
            .starts_with(&expected_text);
        if !matched && force_reconstruction && !boolean(field(case, "ExpectReconstruction")?)? {
            stats.forced_mismatches += 1; // Original forced-only failures are diagnostic.
        } else {
            stats.check("prompt reconstruction equals or extends original", matched);
        }
    }
    stats.cases += 1;
    Ok(true)
}
fn filtered_regression(engine: &Engine, fixture: &Value, stats: &mut Assertions) -> Result<bool> {
    let case = field(fixture, "Filtered")?;
    let template = field(field(fixture, "Templates")?, text(case, "TemplateRef")?)?;
    let actions = vec![
        json!({"Kind":"apply","ParametersJson":field(case,"ParametersJson")?}),
        json!({"Kind":"parse","Bytes":field(case,"InputBytes")?,"Partial":false,"Store":1}),
    ];
    let Some(v) = exchange(
        engine,
        &json!({"Kind":"chat-trace","Template":template,"Actions":actions}),
    )?
    else {
        return Ok(false);
    };
    let events = trace(&v, &actions)?;
    stats.fixed += 3;
    stats.check(
        "filtered generation prompt must be nonempty",
        !text(&events[0], "generation_prompt")?.is_empty(),
    );
    stats.check(
        "filtered parse must avoid out-of-range",
        events[1]["ExceptionKind"] != "out-of-range",
    );
    stats.check(
        "filtered parse must throw runtime error",
        events[1]["ExceptionKind"] == "runtime",
    );
    Ok(true)
}
/// Runs every default test-chat.cpp group, in its original order.
pub(super) fn run(engine: &Engine) -> Result<Value> {
    run_with_options(engine, None, false)
}
/// Retains the source's optional template-filter and forced reconstruction modes.
#[allow(
    clippy::too_many_lines,
    reason = "Keep the original ordered group dispatch and source condition guards together."
)]
pub(super) fn run_with_options(
    engine: &Engine,
    filter: Option<&str>,
    force_reconstruction: bool,
) -> Result<Value> {
    let fixture = fixture()?;
    let mut stats = Assertions::default();
    let filtered_only = filter.is_some() || force_reconstruction;
    let mut expected_cases = 0;
    let mut expected_prefixes = 0;
    let mut expected_final = 0;
    let mut expected_reconstruction = 0;
    for group in array(&fixture, "Groups")? {
        if filtered_only && text(group, "Name")? != "test_template_output_peg_parsers" {
            continue;
        }
        let mut results = Vec::new();
        for entry in array(group, "Sequence")? {
            if let Some(index) = entry.get("Operation") {
                let index = usize::try_from(
                    index
                        .as_u64()
                        .ok_or_else(|| invalid("chat operation index invalid"))?,
                )?;
                if index != results.len() {
                    return Err(invalid("chat operation order mismatch"));
                }
                let operation = materialize(
                    array(group, "Operations")?
                        .get(index)
                        .ok_or_else(|| invalid("chat operation unavailable"))?,
                    &fixture,
                    &results,
                )?;
                let Some(value) = exchange(engine, &operation)? else {
                    return Ok(result(ID,false,"unsupported production API: common chat parsing/templates/diffs, server request conversion, delimiters and grammar validation".to_owned()));
                };
                results.push(value);
            } else if let Some(index) = entry.get("Check") {
                let index = usize::try_from(
                    index
                        .as_u64()
                        .ok_or_else(|| invalid("chat check index invalid"))?,
                )?;
                let check = array(group, "Checks")?
                    .get(index)
                    .ok_or_else(|| invalid("chat check unavailable"))?;
                if boolean(&eval(field(check, "When")?, &results)?)? {
                    stats.fixed += 1;
                    stats.check(
                        &format!(
                            "{} check {index}: {}",
                            text(group, "Name")?,
                            text(check, "Name")?
                        ),
                        boolean(&eval(field(check, "Value")?, &results)?)?,
                    );
                }
            } else if let Some(index) = entry.get("Case") {
                let index = usize::try_from(
                    index
                        .as_u64()
                        .ok_or_else(|| invalid("chat case index invalid"))?,
                )?;
                let case = array(&fixture, "Cases")?
                    .get(index)
                    .ok_or_else(|| invalid("chat case unavailable"))?;
                if filter.is_some_and(|filter| {
                    !case["TemplateRef"].as_str().is_some_and(|path| {
                        path.to_ascii_lowercase()
                            .contains(&filter.to_ascii_lowercase())
                    })
                }) {
                    continue;
                }
                expected_cases += 1;
                expected_prefixes += array(case, "InputBytes")?.len();
                let partial = boolean(field(case, "IsPartial")?)?;
                expected_final += 1 + usize::from(!partial);
                expected_reconstruction += usize::from(
                    !partial
                        && (boolean(field(case, "ExpectReconstruction")?)? || force_reconstruction),
                );
                if !run_case(engine, &fixture, case, &mut stats, force_reconstruction)? {
                    return Ok(result(ID,false,"unsupported production API: complete common chat parser/template contracts".to_owned()));
                }
            } else if entry.get("Filtered") == Some(&json!(0)) {
                // Source condition is deliberately case-sensitive, unlike its builder filter.
                if filter.is_some_and(|f| {
                    !f.is_empty() && "models/templates/GLM-4.7-Flash.jinja".contains(f)
                }) && !filtered_regression(engine, &fixture, &mut stats)?
                {
                    return Ok(result(
                        ID,
                        false,
                        "unsupported production API: filtered chat parse regression".to_owned(),
                    ));
                }
            } else {
                return Err(invalid("unknown chat source sequence event"));
            }
        }
    }
    if !filtered_only {
        stats.check("all fixed source conditions executed", stats.fixed == 363);
        stats.check(
            "full case fixture coverage",
            expected_cases == 431
                && expected_prefixes == 44162
                && expected_final == 853
                && expected_reconstruction == 82,
        );
    }
    stats.check(
        "all selected source cases executed",
        stats.cases == expected_cases,
    );
    stats.check(
        "all selected source byte prefixes executed",
        stats.prefixes == expected_prefixes,
    );
    stats.check(
        "all selected final message comparisons executed",
        stats.final_messages == expected_final,
    );
    stats.check(
        "all selected source reconstructions executed",
        stats.reconstructions == expected_reconstruction,
    );
    Ok(result(
        ID,
        stats.failures.is_empty(),
        format!(
            "14default groups/59template assets; {}cases,{}fixed conditions,{}byte-prefix message comparisons,{}final-message comparisons,{}reconstructions; generated grammar builds={},trigger checks={},matches={}; forced-only reconstruction mismatches={} (diagnostic as source); filter={filter:?}; failures={:?}; no source benchmark",
            stats.cases,
            stats.fixed,
            stats.prefixes,
            stats.final_messages,
            stats.reconstructions,
            stats.grammar_builds,
            stats.grammar_triggers,
            stats.grammar_matches,
            stats.forced_mismatches,
            stats.failures
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn diff_equality_compares_source_string_bytes_and_rejects_changed_fields() -> Result<()> {
        let expected = json!([{
            "reasoning_content_delta":"ignored source field",
            "content_delta":{"Bytes":[104,101,108,108,111]},
            "tool_call_index":0,
            "tool_call_delta":{
                "name":{"Bytes":[102]},
                "arguments":{"Bytes":[123,125]},
                "id":{"Bytes":[105,100]}
            }
        }]);
        let actual = json!([{
            "reasoning_content_delta":"different ignored field",
            "content_delta":"hello",
            "tool_call_index":0,
            "tool_call_delta":{"name":"f","arguments":"{}","id":"id"}
        }]);
        assert!(diff_equal(&expected, &actual)?);
        let mut changed = actual.clone();
        changed[0]["content_delta"] = json!("Hello");
        assert!(!diff_equal(&expected, &changed)?);
        changed = actual.clone();
        changed[0]["tool_call_index"] = json!(1);
        assert!(!diff_equal(&expected, &changed)?);
        for key in ["name", "arguments", "id"] {
            changed = actual.clone();
            changed[0]["tool_call_delta"][key] = json!("different");
            assert!(!diff_equal(&expected, &changed)?);
        }
        changed = actual.clone();
        changed[0]["tool_call_delta"]["arguments"] = json!("{ }");
        assert!(!diff_equal(&expected, &changed)?);
        assert!(!diff_equal(&expected, &json!([]))?);
        assert!(diff_equal(&actual, &expected)?);
        Ok(())
    }

    #[test]
    fn source_chat_helper_guards_preserve_bytes_and_json_order() -> Result<()> {
        let _ = fixture()?;
        validate_template_kwargs(
            &json!({"chat_template_kwargs":{"drop_thinking":"false","thinking":"true"}}),
        )?;
        assert!(
            validate_template_kwargs(&json!({"chat_template_kwargs":{"drop_thinking":false}}))
                .is_err()
        );
        assert_eq!(safe_prefix_len(&[0xeb, 0x82]), 0);
        assert_eq!(safe_prefix_len(&[0xeb, 0x82, 0xac]), 3);
        assert_eq!(safe_prefix_len(&[b'x', 0xeb, 0x82]), 1);
        assert_eq!(normalized_json(r#"{ "z":1,"a":2 }"#), r#"{"z":1,"a":2}"#);
        assert_ne!(
            normalized_json(r#"{"a":2,"z":1}"#),
            normalized_json(r#"{"z":1,"a":2}"#)
        );
        assert_eq!(normalized_json("{\"partial\":"), "");
        let mut message = empty_message();
        accumulate(
            &mut message,
            &[
                json!({"reasoning_content_delta":"r","content_delta":"c","tool_call_index":0,"tool_call_delta":{"name":"f","id":"id","arguments":"{"}}),
            ],
        )?;
        accumulate(
            &mut message,
            &[
                json!({"reasoning_content_delta":"","content_delta":"","tool_call_index":0,"tool_call_delta":{"name":"","id":"","arguments":"}"}}),
            ],
        )?;
        assert_eq!(
            message["tool_calls"][0],
            json!({"name":"f","id":"id","arguments":"{}"})
        );
        assert!(diff_equal(
            &json!([{"content_delta":"x","tool_call_index":null,"tool_call_delta":{"name":"","arguments":"","id":""},"reasoning_content_delta":"a"}]),
            &json!([{"content_delta":"x","tool_call_index":null,"tool_call_delta":{"name":"","arguments":"","id":""},"reasoning_content_delta":"b"}])
        )?);
        assert_eq!(codepoints(&[0xeb, 0x82, b'x']), vec![0xfffd, 0xfffd, 120]);
        Ok(())
    }
}
