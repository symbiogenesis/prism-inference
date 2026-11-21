//! Clean-room independent safe Rust chat provider replacing `tools/prism/upstream/common/chat.hpp`.
//!
//! Implements `chat-trace`, `chat-regex`, `chat-generic-parse`, `chat-server-params`, and
//! `chat-value` directly from `tools/prism/upstream/common/chat.hpp`, `PROTOCOL.md`,
//! `tests/upstream_prism/chat.rs`, and the sibling `reference_*` modules (`reference_jinja`,
//! `reference_chat_templates`, `reference_chat_auto`, `reference_chat_parser`, `reference_peg`,
//! `reference_json_schema`), without importing `bonsai_prism_runtime`, invoking a native bridge,
//! or reading expected-answer fixtures.

use std::collections::{BTreeMap, BTreeSet};

use fancy_regex::Regex;
use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::ser::{SerializeMap, SerializeSeq};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Number, Value, json};

use super::llama_grammar::{array, exact_fields, field};
use super::reference_chat_auto::{
    AutoAnalysis, AutoParser, ChatTemplate, GenerationParams, ReasoningMode, TemplateParams,
    ToolFormatMode, caps_to_json,
};
use super::reference_chat_parser::{
    ChatArena, ChatContentPart, ChatMessage, ChatMessageDiff, ChatPegBuilder, ChatToolCall,
    JsonToolsOptions, compute_message_diffs, map_chat_ast, schema_resolves_to_string,
};
use super::reference_chat_templates::{
    CHATML_JINJA_TEMPLATE, ChatMessage as LegacyChatMessage, current_unix_seconds, legacy_apply,
};
use super::reference_jinja::{JVal, parse_json_jval_bounded};
use super::reference_json_schema::{JsonParser, JsonVal};
use super::reference_peg::{ParseContext, ParseResultType, ParserId, parse_flags};
use super::{Result, invalid};

/// Upstream source identifier covered by this independent reference provider.
pub(super) const ID: &str = "tests/test-chat.cpp";
const SOURCE_SHA256: &str = "9920d484dfe35d8ccec4e34f3dd6a23e1788a479a419cab231c361d3cad4e8b4";

const MAX_OPERATIONS: usize = 128;
const MAX_STRING_BYTES: usize = 65_536;
const MAX_TEMPLATE_BYTES: usize = 1024 * 1024;
const MAX_INPUT_BYTES: usize = 1024 * 1024;
const MAX_RAW_JSON_BYTES: usize = 4 * 1024 * 1024;
const MAX_ARRAY_LEN: usize = 1024;
const MAX_ACTIONS: usize = 32_768;
const MAX_PATTERNS: usize = 128;
const MAX_TOKENS: usize = 65_536;
const MAX_SKIP_POS: i64 = 65_536;
const MAX_STORE_HANDLE: i64 = 32_767;
const MAX_NOW_MS: i64 = 4_102_444_800_000;

// ---------------------------------------------------------------------------
// Insertion-order JSON DOM matching nlohmann::ordered_json semantics
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum OrdJson {
    Null,
    Bool(bool),
    Int(i64),
    Uint(u64),
    Float(Number),
    Str(String),
    Array(Vec<Self>),
    Object(Vec<(String, Self)>),
}

impl OrdJson {
    fn parse(raw: &str) -> std::result::Result<Self, String> {
        serde_json::from_str::<Self>(raw).map_err(|err| err.to_string())
    }

    fn dump_compact(&self) -> Result<String> {
        serde_json::to_string(self).map_err(Into::into)
    }

    fn dump_pretty_2(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(Into::into)
    }

    fn to_serde_value(&self) -> Result<Value> {
        serde_json::to_value(self).map_err(Into::into)
    }

    fn as_str(&self) -> Option<&str> {
        if let Self::Str(s) = self {
            Some(s.as_str())
        } else {
            None
        }
    }

    const fn as_bool(&self) -> Option<bool> {
        if let Self::Bool(b) = self {
            Some(*b)
        } else {
            None
        }
    }

    fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            Self::Uint(v) => i64::try_from(*v).ok(),
            _ => None,
        }
    }

    fn as_array(&self) -> Option<&[Self]> {
        if let Self::Array(items) = self {
            Some(items.as_slice())
        } else {
            None
        }
    }

    fn as_object(&self) -> Option<&[(String, Self)]> {
        if let Self::Object(entries) = self {
            Some(entries.as_slice())
        } else {
            None
        }
    }

    fn get(&self, key: &str) -> Option<&Self> {
        self.as_object()?
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    fn insert(&mut self, key: String, val: Self) {
        if let Self::Object(entries) = self {
            if let Some((_, existing)) = entries.iter_mut().find(|(k, _)| k == &key) {
                *existing = val;
            } else {
                entries.push((key, val));
            }
        }
    }
}

struct OrdJsonVisitor;

impl<'de> Visitor<'de> for OrdJsonVisitor {
    type Value = OrdJson;

    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("JSON value")
    }

    fn visit_bool<E: serde::de::Error>(self, v: bool) -> std::result::Result<OrdJson, E> {
        Ok(OrdJson::Bool(v))
    }

    fn visit_i64<E: serde::de::Error>(self, v: i64) -> std::result::Result<OrdJson, E> {
        Ok(OrdJson::Int(v))
    }

    fn visit_u64<E: serde::de::Error>(self, v: u64) -> std::result::Result<OrdJson, E> {
        Ok(OrdJson::Uint(v))
    }

    fn visit_f64<E: serde::de::Error>(self, v: f64) -> std::result::Result<OrdJson, E> {
        Number::from_f64(v)
            .map(OrdJson::Float)
            .ok_or_else(|| E::custom("nonfinite JSON number"))
    }

    fn visit_str<E: serde::de::Error>(self, v: &str) -> std::result::Result<OrdJson, E> {
        Ok(OrdJson::Str(v.to_owned()))
    }

    fn visit_string<E: serde::de::Error>(self, v: String) -> std::result::Result<OrdJson, E> {
        Ok(OrdJson::Str(v))
    }

    fn visit_none<E: serde::de::Error>(self) -> std::result::Result<OrdJson, E> {
        Ok(OrdJson::Null)
    }

    fn visit_unit<E: serde::de::Error>(self) -> std::result::Result<OrdJson, E> {
        Ok(OrdJson::Null)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<OrdJson, A::Error> {
        let mut out = Vec::new();
        while let Some(v) = seq.next_element()? {
            out.push(v);
        }
        Ok(OrdJson::Array(out))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> std::result::Result<OrdJson, A::Error> {
        let mut out: Vec<(String, OrdJson)> = Vec::new();
        while let Some((key, val)) = map.next_entry::<String, OrdJson>()? {
            if let Some((_, existing)) = out.iter_mut().find(|(k, _)| *k == key) {
                *existing = val;
            } else {
                out.push((key, val));
            }
        }
        Ok(OrdJson::Object(out))
    }
}

impl<'de> Deserialize<'de> for OrdJson {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        deserializer.deserialize_any(OrdJsonVisitor)
    }
}

impl Serialize for OrdJson {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        match self {
            Self::Null => serializer.serialize_unit(),
            Self::Bool(b) => serializer.serialize_bool(*b),
            Self::Int(v) => serializer.serialize_i64(*v),
            Self::Uint(v) => serializer.serialize_u64(*v),
            Self::Float(n) => n.serialize(serializer),
            Self::Str(s) => serializer.serialize_str(s),
            Self::Array(items) => {
                let mut seq = serializer.serialize_seq(Some(items.len()))?;
                for item in items {
                    seq.serialize_element(item)?;
                }
                seq.end()
            }
            Self::Object(entries) => {
                let mut map = serializer.serialize_map(Some(entries.len()))?;
                for (k, v) in entries {
                    map.serialize_entry(k, v)?;
                }
                map.end()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Wire validation helpers mirroring tools/prism/upstream/common/chat.hpp
// ---------------------------------------------------------------------------

fn require(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(invalid(message))
    }
}

fn int_bound(value: &Value, low: i64, high: i64) -> Result<i64> {
    if let Some(u) = value.as_u64() {
        let high_u = u64::try_from(high).map_err(|_| invalid("PEG integer outside bound"))?;
        require(u <= high_u, "PEG integer exceeds bound")?;
        let signed = i64::try_from(u).map_err(|_| invalid("PEG integer outside bound"))?;
        require(signed >= low && signed <= high, "PEG integer outside bound")?;
        return Ok(signed);
    }
    let signed = value
        .as_i64()
        .ok_or_else(|| invalid("PEG argument must be an integer"))?;
    require(signed >= low && signed <= high, "PEG integer outside bound")?;
    Ok(signed)
}

fn string_bound(value: &Value, maximum: usize) -> Result<&str> {
    let s = value
        .as_str()
        .ok_or_else(|| invalid("PEG string must be a string"))?;
    require(s.len() <= maximum, "PEG string exceeds bound")?;
    Ok(s)
}

fn decode_bytes(value: &Value) -> Result<Vec<u8>> {
    let arr = value
        .as_array()
        .ok_or_else(|| invalid("chat byte input must be array"))?;
    require(
        arr.len() <= MAX_INPUT_BYTES,
        "chat byte input exceeds bound",
    )?;
    let mut out = Vec::with_capacity(arr.len());
    for entry in arr {
        let byte = int_bound(entry, 0, 255)?;
        out.push(u8::try_from(byte)?);
    }
    Ok(out)
}

fn parse_raw_ord(value: &Value) -> Result<OrdJson> {
    let s = string_bound(value, MAX_RAW_JSON_BYTES)?;
    OrdJson::parse(s).map_err(invalid)
}

fn wire_call_from_json(value: &Value) -> Result<ChatToolCall> {
    let name = field(value, "name")?
        .as_str()
        .ok_or_else(|| invalid("tool call name must be string"))?
        .to_owned();
    let arguments = field(value, "arguments")?
        .as_str()
        .ok_or_else(|| invalid("tool call arguments must be string"))?
        .to_owned();
    let id = field(value, "id")?
        .as_str()
        .ok_or_else(|| invalid("tool call id must be string"))?
        .to_owned();
    Ok(ChatToolCall {
        name,
        arguments,
        id,
    })
}

fn wire_message_from_json(value: &Value) -> Result<ChatMessage> {
    let role = field(value, "role")?
        .as_str()
        .ok_or_else(|| invalid("message role must be string"))?
        .to_owned();
    let content = field(value, "content")?
        .as_str()
        .ok_or_else(|| invalid("message content must be string"))?
        .to_owned();
    let reasoning_content = field(value, "reasoning_content")?
        .as_str()
        .ok_or_else(|| invalid("message reasoning_content must be string"))?
        .to_owned();
    let tool_name = field(value, "tool_name")?
        .as_str()
        .ok_or_else(|| invalid("message tool_name must be string"))?
        .to_owned();
    let tool_call_id = field(value, "tool_call_id")?
        .as_str()
        .ok_or_else(|| invalid("message tool_call_id must be string"))?
        .to_owned();

    let mut content_parts = Vec::new();
    for p in array(value, "content_parts")? {
        let part_type = field(p, "type")?
            .as_str()
            .ok_or_else(|| invalid("content part type must be string"))?
            .to_owned();
        let text = field(p, "text")?
            .as_str()
            .ok_or_else(|| invalid("content part text must be string"))?
            .to_owned();
        content_parts.push(ChatContentPart { part_type, text });
    }

    let mut tool_calls = Vec::new();
    for c in array(value, "tool_calls")? {
        tool_calls.push(wire_call_from_json(c)?);
    }

    Ok(ChatMessage {
        role,
        content,
        reasoning_content,
        tool_calls,
        content_parts,
        tool_name,
        tool_call_id,
    })
}

fn wire_message_to_json(msg: &ChatMessage) -> Value {
    let parts: Vec<Value> = msg
        .content_parts
        .iter()
        .map(|p| json!({"type": p.part_type, "text": p.text}))
        .collect();
    let calls: Vec<Value> = msg
        .tool_calls
        .iter()
        .map(|c| json!({"name": c.name, "arguments": c.arguments, "id": c.id}))
        .collect();
    json!({
        "role": msg.role,
        "content": msg.content,
        "content_parts": parts,
        "tool_calls": calls,
        "reasoning_content": msg.reasoning_content,
        "tool_name": msg.tool_name,
        "tool_call_id": msg.tool_call_id,
    })
}

fn wire_messages_from_json(value: &Value) -> Result<Vec<ChatMessage>> {
    let arr = value
        .as_array()
        .ok_or_else(|| invalid("chat messages must be array"))?;
    require(
        arr.len() <= MAX_ARRAY_LEN,
        "chat message count exceeds bound",
    )?;
    arr.iter().map(wire_message_from_json).collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ChatTool {
    name: String,
    description: String,
    parameters: String,
}

fn wire_tools_from_json(value: &Value) -> Result<Vec<ChatTool>> {
    let arr = value
        .as_array()
        .ok_or_else(|| invalid("chat tools must be array"))?;
    require(arr.len() <= MAX_ARRAY_LEN, "chat tool count exceeds bound")?;
    let mut out = Vec::with_capacity(arr.len());
    for t in arr {
        let name = field(t, "name")?
            .as_str()
            .ok_or_else(|| invalid("tool name must be string"))?
            .to_owned();
        let description = field(t, "description")?
            .as_str()
            .ok_or_else(|| invalid("tool description must be string"))?
            .to_owned();
        let parameters = field(t, "parameters")?
            .as_str()
            .ok_or_else(|| invalid("tool parameters must be string"))?
            .to_owned();
        out.push(ChatTool {
            name,
            description,
            parameters,
        });
    }
    Ok(out)
}

fn wire_diffs_to_json(values: &[ChatMessageDiff]) -> Value {
    let out: Vec<Value> = values
        .iter()
        .map(|d| {
            json!({
                "reasoning_content_delta": d.reasoning_content_delta,
                "content_delta": d.content_delta,
                "tool_call_index": d.tool_call_index,
                "tool_call_delta": {
                    "name": d.tool_call_delta.name,
                    "arguments": d.tool_call_delta.arguments,
                    "id": d.tool_call_delta.id,
                }
            })
        })
        .collect();
    Value::Array(out)
}

fn wire_error(kind: &str, message: &str) -> Value {
    json!({
        "ExceptionKind": kind,
        "Message": message,
    })
}

#[derive(Clone, Debug)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "Mirrors upstream common_chat_templates_inputs boolean fields"
)]
struct ChatApplyInputs {
    messages: Vec<ChatMessage>,
    tools: Vec<ChatTool>,
    add_generation_prompt: bool,
    continue_final_message: i64,
    use_jinja: bool,
    tool_choice: i64,
    parallel_tool_calls: bool,
    reasoning_format: i64,
    enable_thinking: bool,
    grammar: String,
    json_schema: String,
    chat_template_kwargs: BTreeMap<String, String>,
    now_milliseconds: Option<i64>,
}

impl Default for ChatApplyInputs {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            tools: Vec::new(),
            add_generation_prompt: true,
            continue_final_message: 0,
            use_jinja: true,
            tool_choice: 0,
            parallel_tool_calls: false,
            reasoning_format: 0,
            enable_thinking: true,
            grammar: String::new(),
            json_schema: String::new(),
            chat_template_kwargs: BTreeMap::new(),
            now_milliseconds: None,
        }
    }
}

fn parse_apply_inputs(encoded: &Value) -> Result<ChatApplyInputs> {
    let ord = parse_raw_ord(encoded)?;
    let entries = ord
        .as_object()
        .ok_or_else(|| invalid("chat inputs must be object"))?;
    let mut out = ChatApplyInputs::default();
    for (k, v) in entries {
        let sv = v.to_serde_value()?;
        match k.as_str() {
            "messages" => out.messages = wire_messages_from_json(&sv)?,
            "tools" => out.tools = wire_tools_from_json(&sv)?,
            "add_generation_prompt" => {
                out.add_generation_prompt = sv
                    .as_bool()
                    .ok_or_else(|| invalid("add_generation_prompt must be bool"))?;
            }
            "continue_final_message" => {
                out.continue_final_message = int_bound(&sv, 0, 3)?;
            }
            "use_jinja" => {
                out.use_jinja = sv
                    .as_bool()
                    .ok_or_else(|| invalid("use_jinja must be bool"))?;
            }
            "tool_choice" => {
                out.tool_choice = int_bound(&sv, 0, 2)?;
            }
            "parallel_tool_calls" => {
                out.parallel_tool_calls = sv
                    .as_bool()
                    .ok_or_else(|| invalid("parallel_tool_calls must be bool"))?;
            }
            "reasoning_format" => {
                out.reasoning_format = int_bound(&sv, 0, 3)?;
            }
            "enable_thinking" => {
                out.enable_thinking = sv
                    .as_bool()
                    .ok_or_else(|| invalid("enable_thinking must be bool"))?;
            }
            "grammar" => {
                out.grammar = sv
                    .as_str()
                    .ok_or_else(|| invalid("grammar must be string"))?
                    .to_owned();
            }
            "json_schema" => {
                out.json_schema = sv
                    .as_str()
                    .ok_or_else(|| invalid("json_schema must be string"))?
                    .to_owned();
            }
            "chat_template_kwargs" => {
                let obj = v
                    .as_object()
                    .ok_or_else(|| invalid("chat_template_kwargs must be object"))?;
                for (kw_k, kw_v) in obj {
                    let s = kw_v
                        .as_str()
                        .ok_or_else(|| invalid("chat_template_kwargs value must be string"))?;
                    out.chat_template_kwargs.insert(kw_k.clone(), s.to_owned());
                }
            }
            "now_milliseconds" => {
                out.now_milliseconds = Some(int_bound(&sv, 0, MAX_NOW_MS)?);
            }
            _ => return Err(invalid("unknown chat input parameter")),
        }
    }
    Ok(out)
}

fn validate_chat_trace(op: &Value) -> Result<()> {
    exact_fields(op, &["Kind", "Template", "Actions"])?;
    let _ = string_bound(field(op, "Template")?, MAX_TEMPLATE_BYTES)?;
    let actions = array(op, "Actions")?;
    require(
        actions.len() <= MAX_ACTIONS,
        "chat action count exceeds bound",
    )?;
    let mut applied = false;
    let mut stored = BTreeSet::new();
    for a in actions {
        let k = field(a, "Kind")?
            .as_str()
            .ok_or_else(|| invalid("action Kind must be string"))?;
        match k {
            "apply" => {
                exact_fields(a, &["Kind", "ParametersJson"])?;
                let _ = parse_apply_inputs(field(a, "ParametersJson")?)?;
                applied = true;
            }
            "caps" => {
                exact_fields(a, &["Kind"])?;
            }
            "parse" => {
                exact_fields(a, &["Kind", "Bytes", "Partial", "Store"])?;
                require(applied, "chat parse requires apply")?;
                let _ = decode_bytes(field(a, "Bytes")?)?;
                require(field(a, "Partial")?.is_boolean(), "partial must be boolean")?;
                let handle = int_bound(field(a, "Store")?, 0, MAX_STORE_HANDLE)?;
                require(stored.insert(handle), "duplicate chat message handle")?;
            }
            "diff" => {
                exact_fields(a, &["Kind", "Before", "After"])?;
                let after = int_bound(field(a, "After")?, 0, MAX_STORE_HANDLE)?;
                require(stored.contains(&after), "unknown after message handle")?;
                let before_val = field(a, "Before")?;
                if !before_val.is_null() {
                    let before = int_bound(before_val, 0, MAX_STORE_HANDLE)?;
                    require(stored.contains(&before), "unknown before message handle")?;
                }
            }
            _ => return Err(invalid("unknown chat action")),
        }
    }
    Ok(())
}

fn validate_chat_regex(op: &Value) -> Result<()> {
    exact_fields(op, &["Kind", "Patterns", "Actions"])?;
    let patterns = array(op, "Patterns")?;
    require(
        patterns.len() <= MAX_PATTERNS,
        "regex pattern count exceeds bound",
    )?;
    for p in patterns {
        let _ = string_bound(p, MAX_STRING_BYTES)?;
    }
    let actions = array(op, "Actions")?;
    require(
        actions.len() <= MAX_ACTIONS,
        "regex action count exceeds bound",
    )?;
    for a in actions {
        exact_fields(a, &["Pattern", "Function", "Bytes"])?;
        let idx = usize::try_from(int_bound(field(a, "Pattern")?, 0, 127)?)?;
        require(idx < patterns.len(), "regex pattern index invalid")?;
        let fun = field(a, "Function")?
            .as_str()
            .ok_or_else(|| invalid("regex function must be string"))?;
        require(fun == "search" || fun == "match", "unknown regex function")?;
        let _ = decode_bytes(field(a, "Bytes")?)?;
    }
    Ok(())
}

fn validate_chat_value(op: &Value) -> Result<()> {
    let fn_name = field(op, "Function")?
        .as_str()
        .ok_or_else(|| invalid("chat value Function must be string"))?;
    match fn_name {
        "messages-to-oai" => {
            exact_fields(op, &["Kind", "Function", "Values"])?;
            let _ = wire_messages_from_json(field(op, "Values")?)?;
        }
        "tools-to-oai" => {
            exact_fields(op, &["Kind", "Function", "Values"])?;
            let _ = wire_tools_from_json(field(op, "Values")?)?;
        }
        "messages-from-oai" | "tools-from-oai" | "convert-responses" => {
            exact_fields(op, &["Kind", "Function", "Json"])?;
            let _ = parse_raw_ord(field(op, "Json")?)?;
        }
        "diff" => {
            exact_fields(op, &["Kind", "Function", "Before", "After"])?;
            let _ = wire_message_from_json(field(op, "Before")?)?;
            let _ = wire_message_from_json(field(op, "After")?)?;
        }
        "delimiter-split" => {
            exact_fields(op, &["Kind", "Function", "Delimiters", "Tokens", "Skips"])?;
            let delims = array(op, "Delimiters")?;
            require(
                delims.len() <= MAX_ARRAY_LEN,
                "delimiter count exceeds bound",
            )?;
            let tokens = array(op, "Tokens")?;
            require(tokens.len() <= MAX_TOKENS, "token count exceeds bound")?;
            for t in tokens {
                let _ = int_bound(t, -1, i64::from(i32::MAX))?;
            }
            for d in delims {
                exact_fields(d, &["Role", "Delimiter", "Tokens"])?;
                let _ = int_bound(field(d, "Role")?, 0, 4)?;
                let _ = string_bound(field(d, "Delimiter")?, MAX_STRING_BYTES)?;
                let d_toks = array(d, "Tokens")?;
                require(
                    d_toks.len() <= MAX_ARRAY_LEN,
                    "delimiter tokens exceed bound",
                )?;
                for t in d_toks {
                    let _ = int_bound(t, -1, i64::from(i32::MAX))?;
                }
            }
            let skips = array(op, "Skips")?;
            require(skips.len() <= MAX_ARRAY_LEN, "skip count exceeds bound")?;
            for s in skips {
                let pair = s.as_array().ok_or_else(|| invalid("invalid skip pair"))?;
                require(pair.len() == 2, "invalid skip pair")?;
                let _ = int_bound(&pair[0], 0, MAX_SKIP_POS)?;
                let _ = int_bound(&pair[1], 0, MAX_SKIP_POS)?;
            }
        }
        _ => return Err(invalid("unknown chat value function")),
    }
    Ok(())
}

fn validate_operation(op: &Value) -> Result<()> {
    let kind = field(op, "Kind")?
        .as_str()
        .ok_or_else(|| invalid("operation Kind must be string"))?;
    match kind {
        "chat-trace" => validate_chat_trace(op),
        "chat-regex" => validate_chat_regex(op),
        "chat-generic-parse" => {
            exact_fields(op, &["Kind", "Bytes", "Partial"])?;
            let _ = decode_bytes(field(op, "Bytes")?)?;
            require(
                field(op, "Partial")?.is_boolean(),
                "partial must be boolean",
            )?;
            Ok(())
        }
        "chat-server-params" => {
            exact_fields(op, &["Kind", "Template", "Options", "BodyJson"])?;
            let _ = string_bound(field(op, "Template")?, MAX_TEMPLATE_BYTES)?;
            let _ = parse_raw_ord(field(op, "BodyJson")?)?;
            let opts = field(op, "Options")?
                .as_object()
                .ok_or_else(|| invalid("server options must be object"))?;
            require(opts.len() <= 5, "server options exceed bound")?;
            for key in opts.keys() {
                require(
                    matches!(
                        key.as_str(),
                        "use_jinja"
                            | "enable_thinking"
                            | "reasoning_budget"
                            | "reasoning_format"
                            | "reasoning_budget_message"
                    ),
                    "unknown server option",
                )?;
            }
            Ok(())
        }
        "chat-value" => validate_chat_value(op),
        _ => Err(invalid("unknown chat operation")),
    }
}

// ---------------------------------------------------------------------------
// chat-regex: ECMAScript-compatible regex translation & execution
// ---------------------------------------------------------------------------

fn ecmascript_to_fancy_regex(pattern: &str) -> String {
    let chars: Vec<char> = pattern.chars().collect();
    let mut out = String::with_capacity(pattern.len() + 16);
    let mut in_class = false;
    let mut class_start = false;
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];
        if ch == '\\' {
            out.push('\\');
            i += 1;
            if i < chars.len() {
                let esc = chars[i];
                if esc == '0' && !chars.get(i + 1).is_some_and(char::is_ascii_digit) {
                    out.push_str("x00");
                } else {
                    out.push(esc);
                }
                i += 1;
            }
            class_start = false;
            continue;
        }
        if !in_class {
            if ch == '[' {
                if chars.get(i + 1) == Some(&'^') && chars.get(i + 2) == Some(&']') {
                    out.push_str(r"[\s\S]");
                    i += 3;
                    continue;
                }
                if chars.get(i + 1) == Some(&']') {
                    out.push_str("(?!)");
                    i += 2;
                    continue;
                }
                out.push('[');
                in_class = true;
                class_start = true;
                i += 1;
                if chars.get(i) == Some(&'^') {
                    out.push('^');
                    i += 1;
                }
                continue;
            }
            out.push(ch);
            i += 1;
            continue;
        }
        if class_start && ch == ']' {
            out.push_str(r"\]");
            class_start = false;
            i += 1;
            continue;
        }
        class_start = false;
        if ch == '[' {
            out.push_str(r"\[");
        } else if ch == ']' {
            out.push(']');
            in_class = false;
        } else {
            out.push(ch);
        }
        i += 1;
    }

    out
}

struct CompiledRegex {
    search: Regex,
    full_match: Option<Regex>,
    raw_translated: String,
    mark_count: usize,
}

impl CompiledRegex {
    fn compile(pattern: &str) -> Result<Self> {
        let translated = ecmascript_to_fancy_regex(pattern);
        let search = Regex::new(&translated)
            .map_err(|err| invalid(format!("invalid ECMAScript regex '{pattern}': {err}")))?;
        let mark_count = search.captures_len().saturating_sub(1);
        Ok(Self {
            search,
            full_match: None,
            raw_translated: translated,
            mark_count,
        })
    }

    fn match_regex(&mut self) -> Result<&Regex> {
        if self.full_match.is_none() {
            let anchored = format!(r"\A(?:{})\z", self.raw_translated);
            let compiled = Regex::new(&anchored)
                .map_err(|err| invalid(format!("invalid anchored regex: {err}")))?;
            self.full_match = Some(compiled);
        }
        self.full_match
            .as_ref()
            .ok_or_else(|| invalid("missing compiled full-match regex"))
    }
}

fn execute_chat_regex(op: &Value) -> Result<Value> {
    let raw_patterns = array(op, "Patterns")?;
    let mut compiled: Vec<Option<CompiledRegex>> = Vec::with_capacity(raw_patterns.len());
    compiled.resize_with(raw_patterns.len(), || None);

    let mut trace = Vec::new();
    for action in array(op, "Actions")? {
        let pat_idx = usize::try_from(int_bound(field(action, "Pattern")?, 0, 127)?)?;
        if compiled[pat_idx].is_none() {
            let pat_str = raw_patterns[pat_idx]
                .as_str()
                .ok_or_else(|| invalid("regex pattern must be string"))?;
            compiled[pat_idx] = Some(CompiledRegex::compile(pat_str)?);
        }
        let re = compiled[pat_idx]
            .as_mut()
            .ok_or_else(|| invalid("regex pattern missing"))?;
        let input_bytes = decode_bytes(field(action, "Bytes")?)?;
        let func = field(action, "Function")?
            .as_str()
            .ok_or_else(|| invalid("regex Function must be string"))?;

        let (utf8_ok, haystack) = std::str::from_utf8(&input_bytes).map_or_else(
            |_| {
                let mapped: String = input_bytes.iter().map(|&b| char::from(b)).collect();
                (false, mapped)
            },
            |s| (true, s.to_owned()),
        );

        let regex_ref = if func == "search" {
            &re.search
        } else {
            re.match_regex()?
        };
        let caps_opt = regex_ref
            .captures(&haystack)
            .map_err(|err| invalid(format!("regex execution error: {err}")))?;

        let mut captures_json = Vec::new();
        let matched = if let Some(caps) = caps_opt {
            let char_offsets: Vec<usize> = if utf8_ok {
                Vec::new()
            } else {
                let mut map = vec![0_usize; haystack.len() + 1];
                let mut char_idx = 0_usize;
                for (byte_idx, ch) in haystack.char_indices() {
                    map[byte_idx] = char_idx;
                    char_idx += 1;
                    map[byte_idx + ch.len_utf8()] = char_idx;
                }
                map
            };
            for i in 0..=re.mark_count {
                if let Some(m) = caps.get(i) {
                    let (start, len) = if utf8_ok {
                        (m.start(), m.end().saturating_sub(m.start()))
                    } else {
                        let s = char_offsets[m.start()];
                        let e = char_offsets[m.end()];
                        (s, e.saturating_sub(s))
                    };
                    captures_json.push(json!({
                        "Position": isize::try_from(start)?,
                        "Length": len,
                    }));
                } else {
                    captures_json.push(json!({"Position": -1, "Length": 0}));
                }
            }
            true
        } else {
            false
        };

        trace.push(json!({
            "Matched": matched,
            "MarkCount": re.mark_count,
            "Captures": captures_json,
        }));
    }

    Ok(json!({ "Trace": trace }))
}

// ---------------------------------------------------------------------------
// chat-value: OAI conversions, Responses conversion, diff, delimiter-split
// ---------------------------------------------------------------------------

fn messages_to_oai_ord(msgs: &[ChatMessage]) -> OrdJson {
    let mut out = Vec::with_capacity(msgs.len());
    for msg in msgs {
        let mut entries = Vec::new();
        entries.push(("role".to_owned(), OrdJson::Str(msg.role.clone())));
        if msg.content_parts.is_empty() {
            entries.push(("content".to_owned(), OrdJson::Str(msg.content.clone())));
        } else {
            let parts = msg
                .content_parts
                .iter()
                .map(|p| {
                    OrdJson::Object(vec![
                        ("type".to_owned(), OrdJson::Str(p.part_type.clone())),
                        ("text".to_owned(), OrdJson::Str(p.text.clone())),
                    ])
                })
                .collect();
            entries.push(("content".to_owned(), OrdJson::Array(parts)));
        }
        if !msg.reasoning_content.is_empty() {
            entries.push((
                "reasoning_content".to_owned(),
                OrdJson::Str(msg.reasoning_content.clone()),
            ));
        }
        if !msg.tool_name.is_empty() {
            entries.push(("name".to_owned(), OrdJson::Str(msg.tool_name.clone())));
        }
        if !msg.tool_call_id.is_empty() {
            entries.push((
                "tool_call_id".to_owned(),
                OrdJson::Str(msg.tool_call_id.clone()),
            ));
        }
        if !msg.tool_calls.is_empty() {
            let calls = msg
                .tool_calls
                .iter()
                .map(|tc| {
                    let mut tc_entries = Vec::new();
                    if !tc.id.is_empty() {
                        tc_entries.push(("id".to_owned(), OrdJson::Str(tc.id.clone())));
                    }
                    tc_entries.push(("type".to_owned(), OrdJson::Str("function".to_owned())));
                    tc_entries.push((
                        "function".to_owned(),
                        OrdJson::Object(vec![
                            ("name".to_owned(), OrdJson::Str(tc.name.clone())),
                            ("arguments".to_owned(), OrdJson::Str(tc.arguments.clone())),
                        ]),
                    ));
                    OrdJson::Object(tc_entries)
                })
                .collect();
            entries.push(("tool_calls".to_owned(), OrdJson::Array(calls)));
        }
        out.push(OrdJson::Object(entries));
    }
    OrdJson::Array(out)
}

fn tools_to_oai_ord(tools: &[ChatTool]) -> Result<OrdJson> {
    let mut out = Vec::with_capacity(tools.len());
    for t in tools {
        let mut fn_entries = vec![("name".to_owned(), OrdJson::Str(t.name.clone()))];
        if !t.description.is_empty() {
            fn_entries.push((
                "description".to_owned(),
                OrdJson::Str(t.description.clone()),
            ));
        }
        if !t.parameters.is_empty() {
            let parsed = OrdJson::parse(&t.parameters).map_err(invalid)?;
            fn_entries.push(("parameters".to_owned(), parsed));
        }
        out.push(OrdJson::Object(vec![
            ("type".to_owned(), OrdJson::Str("function".to_owned())),
            ("function".to_owned(), OrdJson::Object(fn_entries)),
        ]));
    }
    Ok(OrdJson::Array(out))
}

#[expect(
    clippy::too_many_lines,
    reason = "Direct field-by-field OpenAI message validation matching upstream"
)]
fn parse_oai_messages(raw: &OrdJson) -> std::result::Result<Vec<ChatMessage>, String> {
    let arr = raw
        .as_array()
        .ok_or_else(|| "messages must be a JSON array".to_owned())?;
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let _ = item
            .as_object()
            .ok_or_else(|| "message must be a JSON object".to_owned())?;
        let role = item
            .get("role")
            .and_then(OrdJson::as_str)
            .ok_or_else(|| "message missing role string".to_owned())?
            .to_owned();
        let mut msg = ChatMessage {
            role,
            ..ChatMessage::default()
        };
        if let Some(rc) = item.get("reasoning_content")
            && !matches!(rc, OrdJson::Null)
        {
            msg.reasoning_content = rc
                .as_str()
                .ok_or_else(|| "reasoning_content must be a string".to_owned())?
                .to_owned();
        }
        if let Some(name) = item.get("name")
            && !matches!(name, OrdJson::Null)
        {
            msg.tool_name = name
                .as_str()
                .ok_or_else(|| "message name must be a string".to_owned())?
                .to_owned();
        }
        if let Some(tcid) = item.get("tool_call_id")
            && !matches!(tcid, OrdJson::Null)
        {
            msg.tool_call_id = tcid
                .as_str()
                .ok_or_else(|| "tool_call_id must be a string".to_owned())?
                .to_owned();
        }
        if let Some(content) = item.get("content") {
            match content {
                OrdJson::Null => {}
                OrdJson::Str(s) => msg.content.clone_from(s),
                OrdJson::Array(parts) => {
                    for part in parts {
                        let p_type = part
                            .get("type")
                            .and_then(OrdJson::as_str)
                            .ok_or_else(|| "content part missing type".to_owned())?;
                        if p_type == "text" {
                            let text = part
                                .get("text")
                                .and_then(OrdJson::as_str)
                                .ok_or_else(|| "text content part missing text".to_owned())?;
                            msg.content_parts.push(ChatContentPart {
                                part_type: "text".to_owned(),
                                text: text.to_owned(),
                            });
                        } else {
                            return Err(format!("Unsupported content part type: {p_type}"));
                        }
                    }
                }
                _ => {
                    return Err("message content must be string, array, or null".to_owned());
                }
            }
        }
        if item.get("content").is_none() && item.get("tool_calls").is_none() {
            return Err("message must contain 'content' or 'tool_calls'".to_owned());
        }
        if let Some(calls) = item.get("tool_calls")
            && !matches!(calls, OrdJson::Null)
        {
            let call_arr = calls
                .as_array()
                .ok_or_else(|| "tool_calls must be a JSON array".to_owned())?;
            for tc in call_arr {
                let id = tc
                    .get("id")
                    .and_then(OrdJson::as_str)
                    .unwrap_or_default()
                    .to_owned();
                if let Some(t) = tc.get("type").and_then(OrdJson::as_str)
                    && t != "function"
                {
                    return Err(format!("Unsupported tool_call type: {t}"));
                }
                let func = tc
                    .get("function")
                    .ok_or_else(|| "tool_call missing function".to_owned())?;
                let name = func
                    .get("name")
                    .and_then(OrdJson::as_str)
                    .ok_or_else(|| "tool_call function missing name".to_owned())?
                    .to_owned();
                let arguments = match func.get("arguments") {
                    Some(OrdJson::Str(s)) => s.clone(),
                    Some(other) if !matches!(other, OrdJson::Null) => {
                        other.dump_compact().map_err(|e| e.to_string())?
                    }
                    _ => String::new(),
                };
                msg.tool_calls.push(ChatToolCall {
                    name,
                    arguments,
                    id,
                });
            }
        }
        out.push(msg);
    }
    Ok(out)
}

fn parse_oai_tools(raw: &OrdJson) -> Result<Vec<ChatTool>> {
    let Some(arr) = raw.as_array() else {
        return Err(invalid("tools must be a JSON array"));
    };
    let mut out = Vec::with_capacity(arr.len());
    for t in arr {
        let fn_obj = t.get("function").unwrap_or(t);
        let name = fn_obj
            .get("name")
            .and_then(OrdJson::as_str)
            .ok_or_else(|| invalid("tool missing function name"))?
            .to_owned();
        let description = fn_obj
            .get("description")
            .and_then(OrdJson::as_str)
            .unwrap_or_default()
            .to_owned();
        let parameters = match fn_obj.get("parameters") {
            Some(OrdJson::Str(s)) => s.clone(),
            Some(v) if !matches!(v, OrdJson::Null) => v.dump_compact()?,
            _ => String::new(),
        };
        out.push(ChatTool {
            name,
            description,
            parameters,
        });
    }
    Ok(out)
}

fn responses_extract_text(content: &OrdJson) -> OrdJson {
    match content {
        OrdJson::Str(s) => OrdJson::Array(vec![OrdJson::Object(vec![
            ("type".to_owned(), OrdJson::Str("text".to_owned())),
            ("text".to_owned(), OrdJson::Str(s.clone())),
        ])]),
        OrdJson::Array(parts) => {
            let mut mapped = Vec::with_capacity(parts.len());
            for part in parts {
                let ptype = part.get("type").and_then(OrdJson::as_str).unwrap_or("");
                if matches!(ptype, "input_text" | "output_text" | "text")
                    && let Some(text) = part.get("text").and_then(OrdJson::as_str)
                {
                    mapped.push(OrdJson::Object(vec![
                        ("type".to_owned(), OrdJson::Str("text".to_owned())),
                        ("text".to_owned(), OrdJson::Str(text.to_owned())),
                    ]));
                } else if ptype == "input_image" {
                    let img_val = match part.get("image_url") {
                        Some(OrdJson::Str(url)) => {
                            OrdJson::Object(vec![("url".to_owned(), OrdJson::Str(url.clone()))])
                        }
                        Some(other) => other.clone(),
                        None => OrdJson::Object(Vec::new()),
                    };
                    mapped.push(OrdJson::Object(vec![
                        ("type".to_owned(), OrdJson::Str("image_url".to_owned())),
                        ("image_url".to_owned(), img_val),
                    ]));
                } else {
                    mapped.push(part.clone());
                }
            }
            OrdJson::Array(mapped)
        }
        OrdJson::Null => OrdJson::Str(String::new()),
        other => other.clone(),
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "Maps OpenAI Responses items and parameters uniformly in one pass"
)]
fn convert_responses_to_chatcmpl(body: &OrdJson) -> Result<OrdJson> {
    let entries = body
        .as_object()
        .ok_or_else(|| invalid("responses body must be JSON object"))?;
    let mut out = OrdJson::Object(Vec::new());
    let mut messages: Vec<OrdJson> = Vec::new();

    if let Some(instr) = body.get("instructions").and_then(OrdJson::as_str)
        && !instr.is_empty()
    {
        messages.push(OrdJson::Object(vec![
            ("role".to_owned(), OrdJson::Str("system".to_owned())),
            ("content".to_owned(), OrdJson::Str(instr.to_owned())),
        ]));
    }

    if let Some(input) = body.get("input") {
        match input {
            OrdJson::Str(s) => {
                messages.push(OrdJson::Object(vec![
                    ("role".to_owned(), OrdJson::Str("user".to_owned())),
                    ("content".to_owned(), OrdJson::Str(s.clone())),
                ]));
            }
            OrdJson::Array(items) => {
                for item in items {
                    let itype = item
                        .get("type")
                        .and_then(OrdJson::as_str)
                        .unwrap_or("message");
                    match itype {
                        "message" => {
                            let role = item.get("role").and_then(OrdJson::as_str).unwrap_or("user");
                            let content = item.get("content").map_or_else(
                                || OrdJson::Str(String::new()),
                                responses_extract_text,
                            );
                            messages.push(OrdJson::Object(vec![
                                ("role".to_owned(), OrdJson::Str(role.to_owned())),
                                ("content".to_owned(), content),
                            ]));
                        }
                        "function_call" => {
                            let call_id = item
                                .get("call_id")
                                .or_else(|| item.get("id"))
                                .and_then(OrdJson::as_str)
                                .unwrap_or_default();
                            let name = item
                                .get("name")
                                .and_then(OrdJson::as_str)
                                .unwrap_or_default();
                            let arguments = match item.get("arguments") {
                                Some(OrdJson::Str(s)) => s.clone(),
                                Some(other) => other.dump_compact()?,
                                None => "{}".to_owned(),
                            };
                            let tc = OrdJson::Object(vec![
                                ("id".to_owned(), OrdJson::Str(call_id.to_owned())),
                                ("type".to_owned(), OrdJson::Str("function".to_owned())),
                                (
                                    "function".to_owned(),
                                    OrdJson::Object(vec![
                                        ("name".to_owned(), OrdJson::Str(name.to_owned())),
                                        ("arguments".to_owned(), OrdJson::Str(arguments)),
                                    ]),
                                ),
                            ]);
                            let merged = if let Some(last) = messages.last_mut()
                                && last.get("role").and_then(OrdJson::as_str) == Some("assistant")
                                && let OrdJson::Object(last_entries) = last
                            {
                                if let Some((_, OrdJson::Array(calls))) =
                                    last_entries.iter_mut().find(|(k, _)| k == "tool_calls")
                                {
                                    calls.push(tc.clone());
                                } else {
                                    last_entries.push((
                                        "tool_calls".to_owned(),
                                        OrdJson::Array(vec![tc.clone()]),
                                    ));
                                }
                                true
                            } else {
                                false
                            };
                            if !merged {
                                messages.push(OrdJson::Object(vec![
                                    ("role".to_owned(), OrdJson::Str("assistant".to_owned())),
                                    ("content".to_owned(), OrdJson::Null),
                                    ("tool_calls".to_owned(), OrdJson::Array(vec![tc])),
                                ]));
                            }
                        }
                        "function_call_output" => {
                            let call_id = item
                                .get("call_id")
                                .and_then(OrdJson::as_str)
                                .unwrap_or_default();
                            let output = match item.get("output") {
                                Some(OrdJson::Str(s)) => s.clone(),
                                Some(other) => other.dump_compact()?,
                                None => String::new(),
                            };
                            messages.push(OrdJson::Object(vec![
                                ("role".to_owned(), OrdJson::Str("tool".to_owned())),
                                ("tool_call_id".to_owned(), OrdJson::Str(call_id.to_owned())),
                                ("content".to_owned(), OrdJson::Str(output)),
                            ]));
                        }
                        "reasoning" => {
                            let mut text = String::new();
                            for key in ["summary", "content"] {
                                if let Some(OrdJson::Array(parts)) = item.get(key) {
                                    for part in parts {
                                        if let Some(t) = part.get("text").and_then(OrdJson::as_str)
                                        {
                                            text.push_str(t);
                                        }
                                    }
                                }
                            }
                            messages.push(OrdJson::Object(vec![
                                ("role".to_owned(), OrdJson::Str("assistant".to_owned())),
                                ("content".to_owned(), OrdJson::Str(String::new())),
                                ("reasoning_content".to_owned(), OrdJson::Str(text)),
                            ]));
                        }
                        _ => {}
                    }
                }
            }
            _ => return Err(invalid("responses input must be string or array")),
        }
    }

    if !messages.is_empty() || body.get("input").is_some() {
        out.insert("messages".to_owned(), OrdJson::Array(messages));
    }

    for (k, v) in entries {
        match k.as_str() {
            "input" | "instructions" => {}
            "max_output_tokens" => out.insert("max_tokens".to_owned(), v.clone()),
            "tools" => {
                if let Some(arr) = v.as_array() {
                    let mut mapped = Vec::new();
                    for t in arr {
                        let ttype = t.get("type").and_then(OrdJson::as_str).unwrap_or("");
                        if t.get("function").is_some() {
                            mapped.push(t.clone());
                        } else if ttype == "function" {
                            let mut fn_entries = Vec::new();
                            for fk in ["name", "description", "parameters"] {
                                if let Some(fv) = t.get(fk) {
                                    fn_entries.push((fk.to_owned(), fv.clone()));
                                }
                            }
                            let strict = t.get("strict").cloned().unwrap_or(OrdJson::Bool(true));
                            fn_entries.push(("strict".to_owned(), strict));
                            mapped.push(OrdJson::Object(vec![
                                ("type".to_owned(), OrdJson::Str("function".to_owned())),
                                ("function".to_owned(), OrdJson::Object(fn_entries)),
                            ]));
                        }
                    }
                    if !mapped.is_empty() {
                        out.insert("tools".to_owned(), OrdJson::Array(mapped));
                    }
                }
            }
            "tool_choice" => {
                if v.get("function").is_none()
                    && v.get("type").and_then(OrdJson::as_str) == Some("function")
                    && let Some(name) = v.get("name").and_then(OrdJson::as_str)
                {
                    out.insert(
                        "tool_choice".to_owned(),
                        OrdJson::Object(vec![
                            ("type".to_owned(), OrdJson::Str("function".to_owned())),
                            (
                                "function".to_owned(),
                                OrdJson::Object(vec![(
                                    "name".to_owned(),
                                    OrdJson::Str(name.to_owned()),
                                )]),
                            ),
                        ]),
                    );
                } else {
                    out.insert("tool_choice".to_owned(), v.clone());
                }
            }
            "reasoning" => {
                if let Some(effort) = v.get("effort") {
                    out.insert("reasoning_effort".to_owned(), effort.clone());
                }
            }
            "text" => {
                if let Some(fmt) = v.get("format") {
                    let ftype = fmt.get("type").and_then(OrdJson::as_str).unwrap_or("");
                    if ftype == "json_schema" {
                        let mut schema_obj = Vec::new();
                        for sk in ["name", "schema", "strict", "description"] {
                            if let Some(sv) = fmt.get(sk) {
                                schema_obj.push((sk.to_owned(), sv.clone()));
                            }
                        }
                        out.insert(
                            "response_format".to_owned(),
                            OrdJson::Object(vec![
                                ("type".to_owned(), OrdJson::Str("json_schema".to_owned())),
                                ("json_schema".to_owned(), OrdJson::Object(schema_obj)),
                            ]),
                        );
                    } else if ftype == "json_object" {
                        out.insert(
                            "response_format".to_owned(),
                            OrdJson::Object(vec![(
                                "type".to_owned(),
                                OrdJson::Str("json_object".to_owned()),
                            )]),
                        );
                    }
                }
            }
            _ => out.insert(k.clone(), v.clone()),
        }
    }

    Ok(out)
}

#[derive(Clone, Debug)]
struct DelimDef {
    role: i64,
    tokens: Vec<i32>,
}

fn execute_delimiter_split(op: &Value) -> Result<Value> {
    let mut delims = Vec::new();
    for d in array(op, "Delimiters")? {
        let role = int_bound(field(d, "Role")?, 0, 4)?;
        let d_toks_val = array(d, "Tokens")?;
        let mut d_toks = Vec::with_capacity(d_toks_val.len());
        for t in d_toks_val {
            d_toks.push(i32::try_from(int_bound(t, -1, i64::from(i32::MAX))?)?);
        }
        if !d_toks.is_empty() {
            delims.push(DelimDef {
                role,
                tokens: d_toks,
            });
        }
    }
    delims.sort_by_key(|d| std::cmp::Reverse(d.tokens.len()));

    let toks_val = array(op, "Tokens")?;
    let mut tokens = Vec::with_capacity(toks_val.len());
    for t in toks_val {
        tokens.push(i32::try_from(int_bound(t, -1, i64::from(i32::MAX))?)?);
    }

    let mut skips = BTreeMap::new();
    for s in array(op, "Skips")? {
        let pair = s.as_array().ok_or_else(|| invalid("invalid skip pair"))?;
        let pos = usize::try_from(int_bound(&pair[0], 0, MAX_SKIP_POS)?)?;
        let len = usize::try_from(int_bound(&pair[1], 0, MAX_SKIP_POS)?)?;
        skips.insert(pos, len);
    }

    let mut spans: Vec<Value> = Vec::new();
    let mut cur_role: Option<i64> = None;
    let mut cur_start: usize = 0;
    let mut pos: usize = 0;

    while pos < tokens.len() {
        if let Some(&skip_len) = skips.get(&pos)
            && skip_len > 0
        {
            pos = (pos + skip_len).min(tokens.len());
            continue;
        }
        let matched = delims.iter().find(|d| {
            pos + d.tokens.len() <= tokens.len() && tokens[pos..pos + d.tokens.len()] == *d.tokens
        });
        if let Some(d) = matched {
            if let Some(role) = cur_role
                && pos > cur_start
            {
                spans.push(json!({
                    "role": role,
                    "pos": cur_start,
                    "len": pos - cur_start,
                }));
            }
            cur_role = Some(d.role);
            cur_start = pos;
            pos += d.tokens.len();
        } else {
            pos += 1;
        }
    }
    if let Some(role) = cur_role
        && tokens.len() > cur_start
    {
        spans.push(json!({
            "role": role,
            "pos": cur_start,
            "len": tokens.len() - cur_start,
        }));
    }

    Ok(json!({ "spans": spans }))
}

fn execute_chat_value(op: &Value) -> Result<Value> {
    let fn_name = field(op, "Function")?
        .as_str()
        .ok_or_else(|| invalid("chat value Function must be string"))?;
    match fn_name {
        "messages-to-oai" => {
            let msgs = wire_messages_from_json(field(op, "Values")?)?;
            let dump = messages_to_oai_ord(&msgs).dump_pretty_2()?;
            Ok(json!({ "JsonDump": dump }))
        }
        "tools-to-oai" => {
            let tools = wire_tools_from_json(field(op, "Values")?)?;
            let dump = tools_to_oai_ord(&tools)?.dump_pretty_2()?;
            Ok(json!({ "JsonDump": dump }))
        }
        "messages-from-oai" => {
            let raw = parse_raw_ord(field(op, "Json")?)?;
            match parse_oai_messages(&raw) {
                Ok(msgs) => {
                    let wire: Vec<Value> = msgs.iter().map(wire_message_to_json).collect();
                    Ok(json!({
                        "ExceptionKind": null,
                        "Message": "",
                        "Messages": wire,
                    }))
                }
                Err(msg) => Ok(wire_error("standard", &msg)),
            }
        }
        "tools-from-oai" => {
            let raw = parse_raw_ord(field(op, "Json")?)?;
            let tools = parse_oai_tools(&raw)?;
            let wire: Vec<Value> = tools
                .iter()
                .map(|t| {
                    json!({
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    })
                })
                .collect();
            Ok(json!({ "Tools": wire }))
        }
        "convert-responses" => {
            let raw = parse_raw_ord(field(op, "Json")?)?;
            let converted = convert_responses_to_chatcmpl(&raw)?;
            Ok(json!({ "Value": converted.to_serde_value()? }))
        }
        "diff" => {
            let before = wire_message_from_json(field(op, "Before")?)?;
            let after = wire_message_from_json(field(op, "After")?)?;
            let diffs = compute_message_diffs(&before, &after)?;
            Ok(json!({ "Diffs": wire_diffs_to_json(&diffs) }))
        }
        "delimiter-split" => execute_delimiter_split(op),
        _ => Err(invalid("unknown chat value function")),
    }
}

// ---------------------------------------------------------------------------
// Format detection, generation_prompt, special PEG builders, and chat-trace
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[expect(
    dead_code,
    reason = "Documents complete upstream common_chat_format enum discriminants"
)]
enum ChatFormat {
    ContentOnly = 0,
    Generic = 1,
    MistralNemo = 2,
    Magistral = 3,
    Llama3X = 4,
    Llama3XBuiltin = 5,
    DeepseekR1 = 6,
    FireFunctionV2 = 7,
    FunctionaryV32 = 8,
    FunctionaryV31 = 9,
    DeepseekV31 = 10,
    Hermes2Pro = 11,
    CommandR7b = 12,
    Granite = 13,
    GptOss = 14,
    SeedOss = 15,
    NemotronV2 = 16,
    Apriel15 = 17,
    XiaomiMimo = 18,
    SolarOpen = 19,
    ExaoneMoe = 20,
    MinimaxM2 = 21,
    KimiK2 = 22,
    Qwen3CoderXml = 23,
    Glm45 = 24,
    DeepseekV32 = 25,
    PegSimple = 26,
    PegNative = 27,
    PegConstructed = 28,
    Gemma4 = 29,
    KimiK3 = 30,
    Lfm2 = 31,
    MinimaxM3 = 32,
    MiniCpm5 = 33,
    MuseGlimmer = 34,
    GigaChat3 = 35,
    DeepseekV4 = 36,
}

impl ChatFormat {
    const fn as_i32(self) -> i32 {
        match self {
            Self::ContentOnly => 0,
            Self::Generic => 1,
            Self::MistralNemo => 2,
            Self::Magistral => 3,
            Self::Llama3X => 4,
            Self::Llama3XBuiltin => 5,
            Self::DeepseekR1 => 6,
            Self::FireFunctionV2 => 7,
            Self::FunctionaryV32 => 8,
            Self::FunctionaryV31 => 9,
            Self::DeepseekV31 => 10,
            Self::Hermes2Pro => 11,
            Self::CommandR7b => 12,
            Self::Granite => 13,
            Self::GptOss => 14,
            Self::SeedOss => 15,
            Self::NemotronV2 => 16,
            Self::Apriel15 => 17,
            Self::XiaomiMimo => 18,
            Self::SolarOpen => 19,
            Self::ExaoneMoe => 20,
            Self::MinimaxM2 => 21,
            Self::KimiK2 => 22,
            Self::Qwen3CoderXml => 23,
            Self::Glm45 => 24,
            Self::DeepseekV32 => 25,
            Self::PegSimple => 26,
            Self::PegNative => 27,
            Self::PegConstructed => 28,
            Self::Gemma4 => 29,
            Self::KimiK3 => 30,
            Self::Lfm2 => 31,
            Self::MinimaxM3 => 32,
            Self::MiniCpm5 => 33,
            Self::MuseGlimmer => 34,
            Self::GigaChat3 => 35,
            Self::DeepseekV4 => 36,
        }
    }
}

#[derive(Clone, Debug)]
struct TriggerDef {
    kind: i32,
    value: String,
}

#[derive(Clone, Debug)]
struct AppliedChatParams {
    format: i32,
    prompt: String,
    grammar: String,
    grammar_lazy: bool,
    generation_prompt: String,
    supports_thinking: bool,
    thinking_start_tag: String,
    thinking_end_tags: Vec<String>,
    grammar_triggers: Vec<TriggerDef>,
    parser: String,
    reasoning_format: i64,
    format_kind: ChatFormat,
    continue_mode: i64,
    prefill_reasoning: String,
    prefill_content: String,
    has_tools_active: bool,
    rf0_think_prefix: String,
    apriel_think_prefix: String,
    tool_param_schemas: BTreeMap<String, JsonVal>,
}

impl AppliedChatParams {
    fn to_wire_json(&self) -> Value {
        let triggers: Vec<Value> = self
            .grammar_triggers
            .iter()
            .map(|t| json!({"type": t.kind, "value": t.value}))
            .collect();
        json!({
            "format": self.format,
            "prompt": self.prompt,
            "grammar": self.grammar,
            "grammar_lazy": self.grammar_lazy,
            "generation_prompt": self.generation_prompt,
            "supports_thinking": self.supports_thinking,
            "thinking_start_tag": self.thinking_start_tag,
            "thinking_end_tags": self.thinking_end_tags,
            "grammar_triggers": triggers,
            "parser": self.parser,
        })
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "Classifies template sources across all special chat formats in priority order"
)]
fn detect_special_format(src: &str, analysis: Option<&AutoAnalysis>) -> ChatFormat {
    if src.contains("<|channel|>") && src.contains("<|message|>") {
        return ChatFormat::GptOss;
    }
    if src.contains("<|channel>thought") || src.contains("<|tool_call>call:") {
        return ChatFormat::Gemma4;
    }
    if src.contains("｜DSML｜tool_calls")
        || (src.contains("DSML") && (src.contains("tool_calls>") || src.contains("tool_calls\"")))
    {
        return ChatFormat::DeepseekV4;
    }
    if src.contains("｜DSML｜function_calls") || src.contains("DSML") {
        return ChatFormat::DeepseekV32;
    }
    if src.contains("<｜tool▁calls▁begin｜>") {
        return ChatFormat::DeepseekV31;
    }
    if src.contains("<|open|>think<|sep|>") || src.contains("open_tag('think')") {
        return ChatFormat::KimiK3;
    }
    if src.contains("<|tool_calls_section_begin|>") {
        return ChatFormat::KimiK2;
    }
    if src.contains("<|tool_call_start|>")
        || src.contains("<|tool_list_start|>")
        || src.contains("List of tools: [")
    {
        return ChatFormat::Lfm2;
    }
    if src.contains("]<]minimax[>[") {
        return ChatFormat::MinimaxM3;
    }
    if src.contains("<minimax:tool_call>") {
        return ChatFormat::MinimaxM2;
    }
    if src.contains("<arg_key>") && src.contains("<arg_value>") {
        return ChatFormat::Glm45;
    }
    if src.contains("<TOOLCALL>") && src.contains("</TOOLCALL>") {
        return ChatFormat::NemotronV2;
    }
    if src.contains("[BEGIN FINAL RESPONSE]") {
        return ChatFormat::Apriel15;
    }
    if src.contains("<function name=\"") && src.contains("<param name=\"") {
        return ChatFormat::MiniCpm5;
    }
    if src.contains("<atem:function_calls>") {
        return ChatFormat::MuseGlimmer;
    }
    if src.contains("<|role_sep|>")
        && (src.contains("function call") || src.contains("<|function_call|>"))
    {
        return ChatFormat::GigaChat3;
    }
    if src.contains("[TOOL_CALLS]") {
        if src.contains("[ARGS]") {
            return if src.contains("[THINK]") {
                ChatFormat::Magistral
            } else {
                ChatFormat::MistralNemo
            };
        }
        return ChatFormat::MistralNemo;
    }
    if (src.contains("from=assistant") || src.contains("recipient=")) && src.contains(">>>") {
        return ChatFormat::FunctionaryV32;
    }
    if src.contains("<function=")
        && !src.contains("<parameter=")
        && !src.contains("<tool_call>")
        && !src.contains("<arg_key>")
        && !src.contains("<invoke")
    {
        return ChatFormat::FunctionaryV31;
    }
    if src.contains("functools") {
        return ChatFormat::FireFunctionV2;
    }
    if (src.contains("<|python_tag|>")
        || src.contains("Environment: ipython")
        || src.contains("message.role == \"ipython\""))
        && !src.contains("<tool_call>")
    {
        return ChatFormat::Llama3X;
    }
    if src.contains("<|START_THINKING|>") && src.contains("<|START_ACTION|>") {
        return ChatFormat::CommandR7b;
    }
    if src.contains("Action: ```json")
        || src.contains("Action:\n```json")
        || src.contains(r"Action:\n```json")
    {
        return ChatFormat::CommandR7b;
    }
    if src.contains("<|start_of_role|>")
        && (src.contains("<|tool_call|>") || src.contains("Granite"))
        && !src.contains("<tool_call>")
    {
        return ChatFormat::Granite;
    }
    if let Some(a) = analysis {
        if a.tools.format.mode == ToolFormatMode::TagWithTagged {
            return ChatFormat::PegConstructed;
        }
        if matches!(
            a.tools.format.mode,
            ToolFormatMode::Json | ToolFormatMode::TagWithJsonArgs
        ) {
            return ChatFormat::PegNative;
        }
        if a.reasoning.mode != ReasoningMode::None {
            return ChatFormat::DeepseekR1;
        }
    }
    ChatFormat::PegSimple
}

fn json_val_from_str(raw: &str) -> Result<JsonVal> {
    JsonParser::parse_with_limit(raw, MAX_RAW_JSON_BYTES).map_err(invalid)
}

fn jval_from_json_str(raw: &str) -> Result<JVal> {
    parse_json_jval_bounded(raw, MAX_RAW_JSON_BYTES, 128)
}

#[derive(Clone, Copy)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "Tracks orthogonal template message serialization quirks"
)]
struct MsgAdaptMode {
    use_thinking_field: bool,
    avoid_content_with_tools_and_thinking: bool,
    ministral_thinking_parts: bool,
    ds_v31_inline_think: bool,
    strip_trailing_newlines: bool,
}

fn adapt_mode_for_src(src: &str, fmt: ChatFormat) -> MsgAdaptMode {
    MsgAdaptMode {
        use_thinking_field: matches!(fmt, ChatFormat::GptOss | ChatFormat::Lfm2)
            || src.contains("message.thinking")
            || src.contains("message['thinking']"),
        avoid_content_with_tools_and_thinking: fmt == ChatFormat::GptOss,
        ministral_thinking_parts: src.contains("block['type'] == 'thinking'"),
        ds_v31_inline_think: fmt == ChatFormat::DeepseekV31,
        strip_trailing_newlines: src.contains("render_content") && src.contains("<im_patch>"),
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "Converts ChatMessage fields into template-adapted JVal entries"
)]
fn messages_to_jval_for_template(
    msgs: &[ChatMessage],
    mode: MsgAdaptMode,
    requires_nonnull: bool,
) -> JVal {
    let mut out = Vec::with_capacity(msgs.len());
    for msg in msgs {
        let trim_nl = |s: &str| -> String {
            if mode.strip_trailing_newlines {
                s.trim_end_matches('\n').to_owned()
            } else {
                s.to_owned()
            }
        };
        let reasoning_text = trim_nl(&msg.reasoning_content);
        let msg_content_text = trim_nl(&msg.content);
        let mut pairs = Vec::new();
        pairs.push((
            JVal::text_str("role"),
            JVal::bytes(msg.role.as_bytes().to_vec(), true),
        ));

        if mode.ministral_thinking_parts && msg.role == "assistant" && !reasoning_text.is_empty() {
            let mut parts = vec![JVal::object(vec![
                (JVal::text_str("type"), JVal::text_str("thinking")),
                (
                    JVal::text_str("thinking"),
                    JVal::bytes(reasoning_text.as_bytes().to_vec(), true),
                ),
            ])];
            if !msg_content_text.is_empty() {
                parts.push(JVal::object(vec![
                    (JVal::text_str("type"), JVal::text_str("text")),
                    (
                        JVal::text_str("text"),
                        JVal::bytes(msg_content_text.as_bytes().to_vec(), true),
                    ),
                ]));
            }
            pairs.push((JVal::text_str("content"), JVal::array(parts)));
        } else if !msg.content_parts.is_empty() {
            let parts: Vec<JVal> = msg
                .content_parts
                .iter()
                .map(|p| {
                    let pt = if p.part_type == "text" {
                        trim_nl(&p.text)
                    } else {
                        p.text.clone()
                    };
                    JVal::object(vec![
                        (
                            JVal::text_str("type"),
                            JVal::bytes(p.part_type.as_bytes().to_vec(), true),
                        ),
                        (JVal::text_str("text"), JVal::bytes(pt.into_bytes(), true)),
                    ])
                })
                .collect();
            pairs.push((JVal::text_str("content"), JVal::array(parts)));
        } else {
            let mut content_str = msg_content_text;
            if mode.ds_v31_inline_think && msg.role == "assistant" && !reasoning_text.is_empty() {
                content_str = format!("<think>{reasoning_text}</think>{content_str}");
            }
            if content_str.is_empty()
                && !msg.tool_calls.is_empty()
                && !requires_nonnull
                && !mode.avoid_content_with_tools_and_thinking
            {
                pairs.push((JVal::text_str("content"), JVal::none()));
            } else {
                pairs.push((
                    JVal::text_str("content"),
                    JVal::bytes(content_str.into_bytes(), true),
                ));
            }
        }

        if !reasoning_text.is_empty() && !mode.ministral_thinking_parts {
            if mode.use_thinking_field {
                pairs.push((
                    JVal::text_str("thinking"),
                    JVal::bytes(reasoning_text.as_bytes().to_vec(), true),
                ));
            }
            if !mode.avoid_content_with_tools_and_thinking {
                pairs.push((
                    JVal::text_str("reasoning_content"),
                    JVal::bytes(reasoning_text.as_bytes().to_vec(), true),
                ));
            }
        }

        if !msg.tool_name.is_empty() {
            pairs.push((
                JVal::text_str("name"),
                JVal::bytes(msg.tool_name.as_bytes().to_vec(), true),
            ));
        }
        if !msg.tool_call_id.is_empty() {
            pairs.push((
                JVal::text_str("tool_call_id"),
                JVal::bytes(msg.tool_call_id.as_bytes().to_vec(), true),
            ));
        }

        if !msg.tool_calls.is_empty() {
            let mut calls = Vec::with_capacity(msg.tool_calls.len());
            for (idx, tc) in msg.tool_calls.iter().enumerate() {
                let mut tc_pairs = Vec::new();
                let id_str = if tc.id.is_empty() {
                    format!("call_{idx}")
                } else {
                    tc.id.clone()
                };
                tc_pairs.push((JVal::text_str("id"), JVal::bytes(id_str.into_bytes(), true)));
                tc_pairs.push((JVal::text_str("type"), JVal::text_str("function")));
                let args_val = if tc.arguments.is_empty() {
                    JVal::text_str("{}")
                } else {
                    JVal::bytes(tc.arguments.as_bytes().to_vec(), true)
                };
                let fn_obj = JVal::object(vec![
                    (
                        JVal::text_str("name"),
                        JVal::bytes(tc.name.as_bytes().to_vec(), true),
                    ),
                    (JVal::text_str("arguments"), args_val),
                ]);
                tc_pairs.push((JVal::text_str("function"), fn_obj));
                calls.push(JVal::object(tc_pairs));
            }
            pairs.push((JVal::text_str("tool_calls"), JVal::array(calls)));
        }

        out.push(JVal::object(pairs));
    }
    JVal::array(out)
}

fn tools_to_jval(tools: &[ChatTool]) -> Result<Option<JVal>> {
    if tools.is_empty() {
        return Ok(None);
    }
    let mut out = Vec::with_capacity(tools.len());
    for t in tools {
        let params = if t.parameters.is_empty() {
            jval_from_json_str("{}")?
        } else {
            jval_from_json_str(&t.parameters)?
        };
        let fn_obj = JVal::object(vec![
            (
                JVal::text_str("name"),
                JVal::bytes(t.name.as_bytes().to_vec(), true),
            ),
            (
                JVal::text_str("description"),
                JVal::bytes(t.description.as_bytes().to_vec(), true),
            ),
            (JVal::text_str("parameters"), params),
        ]);
        out.push(JVal::object(vec![
            (JVal::text_str("type"), JVal::text_str("function")),
            (JVal::text_str("function"), fn_obj),
        ]));
    }
    Ok(Some(JVal::array(out)))
}

fn tools_to_json_val(tools: &[ChatTool]) -> Result<Option<JsonVal>> {
    if tools.is_empty() {
        return Ok(None);
    }
    let mut arr = Vec::with_capacity(tools.len());
    for t in tools {
        let params = if t.parameters.is_empty() {
            JsonVal::Object(vec![("type".to_owned(), JsonVal::Str("object".to_owned()))])
        } else {
            json_val_from_str(&t.parameters)?
        };
        arr.push(JsonVal::Object(vec![
            ("type".to_owned(), JsonVal::Str("function".to_owned())),
            (
                "function".to_owned(),
                JsonVal::Object(vec![
                    ("name".to_owned(), JsonVal::Str(t.name.clone())),
                    (
                        "description".to_owned(),
                        JsonVal::Str(t.description.clone()),
                    ),
                    ("parameters".to_owned(), params),
                ]),
            ),
        ]));
    }
    Ok(Some(JsonVal::Array(arr)))
}

fn reorder_tool_results_for_assistant(msgs: &mut [ChatMessage]) {
    let mut i = 0;
    while i < msgs.len() {
        if msgs[i].role == "assistant" && !msgs[i].tool_calls.is_empty() {
            let start = i + 1;
            let mut end = start;
            while end < msgs.len() && msgs[end].role == "tool" {
                end += 1;
            }
            if end - start > 1 && msgs[start..end].iter().any(|m| !m.tool_call_id.is_empty()) {
                let call_ids: Vec<String> =
                    msgs[i].tool_calls.iter().map(|c| c.id.clone()).collect();
                msgs[start..end].sort_by_key(|m| {
                    if m.tool_call_id.is_empty() {
                        usize::MAX
                    } else {
                        call_ids
                            .iter()
                            .position(|id| id == &m.tool_call_id)
                            .unwrap_or(usize::MAX)
                    }
                });
            }
            i = end;
        } else {
            i += 1;
        }
    }
}

fn build_effective_extra(
    template_src: &str,
    initial_kind: ChatFormat,
    inputs: &ChatApplyInputs,
) -> Result<Option<JVal>> {
    let mut pairs = Vec::with_capacity(inputs.chat_template_kwargs.len() + 2);
    for (k, v) in &inputs.chat_template_kwargs {
        let jval = parse_json_jval_bounded(v, MAX_RAW_JSON_BYTES, 64)
            .unwrap_or_else(|_| JVal::bytes(v.as_bytes().to_vec(), true));
        pairs.push((JVal::bytes(k.as_bytes().to_vec(), true), jval));
    }
    if template_src.contains("drop_thinking")
        && let Some(raw_pr) = inputs.chat_template_kwargs.get("preserve_reasoning")
        && !inputs.chat_template_kwargs.contains_key("drop_thinking")
    {
        match raw_pr.trim() {
            "true" => pairs.push((JVal::text_str("drop_thinking"), JVal::boolean(false))),
            "false" => pairs.push((JVal::text_str("drop_thinking"), JVal::boolean(true))),
            _ => {}
        }
    }
    if initial_kind == ChatFormat::DeepseekV4
        && !inputs.json_schema.trim().is_empty()
        && !inputs.chat_template_kwargs.contains_key("response_format")
    {
        let schema_jval = jval_from_json_str(&inputs.json_schema)?;
        pairs.push((JVal::text_str("response_format"), schema_jval));
    }
    if pairs.is_empty() {
        Ok(None)
    } else {
        Ok(Some(JVal::object(pairs)))
    }
}

fn longest_common_prefix_len(a: &str, b: &str) -> usize {
    let mut len = 0;
    for (ca, cb) in a.chars().zip(b.chars()) {
        if ca != cb {
            break;
        }
        len += ca.len_utf8();
    }
    len
}

const CONTINUE_SENTINEL: &str = "CONTINUE_PROBE_SENTINEL_9f3a";

#[expect(
    clippy::too_many_arguments,
    reason = "Mirrors continuation probe parameters"
)]
fn detect_generation_prompt(
    tmpl: &ChatTemplate,
    inputs: &ChatApplyInputs,
    effective_messages: &[ChatMessage],
    mode: MsgAdaptMode,
    requires_nonnull: bool,
    now_secs: Option<i64>,
    force_object_args: bool,
    extra: Option<&JVal>,
    prompt_with_gen: &str,
) -> String {
    if !inputs.add_generation_prompt {
        return String::new();
    }
    let extra_cloned = extra.map(JVal::deep_clone_fresh);
    let tools_jval = tools_to_jval(&inputs.tools).ok().flatten();
    let msgs_jval = messages_to_jval_for_template(effective_messages, mode, requires_nonnull);
    let no_gen_params = TemplateParams {
        messages: Some(msgs_jval),
        tools: tools_jval.clone(),
        add_generation_prompt: false,
        enable_thinking: inputs.enable_thinking,
        extra_context: extra_cloned.clone(),
    };
    if let Ok(rendered_no_gen) =
        tmpl.apply_with_now_object_args(&no_gen_params, now_secs, force_object_args)
    {
        if let Some(suffix) = prompt_with_gen.strip_prefix(&rendered_no_gen) {
            return suffix.to_owned();
        }
        let lcp = longest_common_prefix_len(prompt_with_gen, &rendered_no_gen);
        return prompt_with_gen[lcp..].to_owned();
    }

    let mut sentinel_msgs = effective_messages.to_vec();
    sentinel_msgs.push(ChatMessage {
        role: "assistant".to_owned(),
        content: "SENTINEL_GENERATION_PROBE".to_owned(),
        ..ChatMessage::default()
    });
    let sentinel_jval = messages_to_jval_for_template(&sentinel_msgs, mode, requires_nonnull);
    let probe_params = TemplateParams {
        messages: Some(sentinel_jval),
        tools: tools_jval,
        add_generation_prompt: false,
        enable_thinking: inputs.enable_thinking,
        extra_context: extra_cloned,
    };
    if let Ok(rendered_with_sentinel) =
        tmpl.apply_with_now_object_args(&probe_params, now_secs, force_object_args)
        && let Some(sentinel_pos) = rendered_with_sentinel.find("SENTINEL_GENERATION_PROBE")
    {
        let before_sentinel = &rendered_with_sentinel[..sentinel_pos];
        let lcp = longest_common_prefix_len(prompt_with_gen, before_sentinel);
        if lcp < prompt_with_gen.len() {
            return prompt_with_gen[lcp..].to_owned();
        }
    }
    String::new()
}

#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "Continuation probing coordinates template rendering across format variants"
)]
fn apply_continuation_and_gen_prompt(
    tmpl: &ChatTemplate,
    inputs: &ChatApplyInputs,
    effective_messages: &[ChatMessage],
    format_kind: ChatFormat,
    mode: MsgAdaptMode,
    requires_nonnull: bool,
    now_secs: Option<i64>,
    force_object_args: bool,
    extra: Option<&JVal>,
    prompt: &mut String,
) -> String {
    if inputs.continue_final_message == 0 {
        return detect_generation_prompt(
            tmpl,
            inputs,
            effective_messages,
            mode,
            requires_nonnull,
            now_secs,
            force_object_args,
            extra,
            prompt,
        );
    }
    let tools_jval = tools_to_jval(&inputs.tools).ok().flatten();

    if let Some(last) = effective_messages.last() {
        let target_reasoning = inputs.continue_final_message == 2
            || (inputs.continue_final_message == 1
                && last.content.is_empty()
                && !last.reasoning_content.is_empty());

        let synthetic_tags = match format_kind {
            ChatFormat::Gemma4 => Some(("<|channel>thought\n", "<channel|>")),
            ChatFormat::KimiK2 | ChatFormat::Lfm2 => Some(("<think>", "</think>")),
            _ => None,
        };

        let mut handled = false;
        if last.role == "assistant" && !last.reasoning_content.is_empty() {
            let mut r_msgs = effective_messages.to_vec();
            if let Some(edited) = r_msgs.last_mut() {
                edited.reasoning_content.push_str(CONTINUE_SENTINEL);
            }
            let r_jval = messages_to_jval_for_template(&r_msgs, mode, requires_nonnull);
            let r_params = TemplateParams {
                messages: Some(r_jval),
                tools: tools_jval.clone(),
                add_generation_prompt: false,
                enable_thinking: inputs.enable_thinking,
                extra_context: extra.map(JVal::deep_clone_fresh),
            };
            let native_pos = tmpl
                .apply_with_now_object_args(&r_params, now_secs, force_object_args)
                .ok()
                .and_then(|rendered_r| {
                    rendered_r
                        .find(CONTINUE_SENTINEL)
                        .map(|pos| (rendered_r, pos))
                });

            if let Some((rendered_r, pos)) = native_pos {
                if target_reasoning {
                    *prompt = rendered_r[..pos].to_owned();
                    handled = true;
                }
            } else if let Some((open_tag, close_tag)) = synthetic_tags
                && effective_messages.len() > 1
            {
                let prefix_msgs = &effective_messages[..effective_messages.len() - 1];
                let pfx_jval = messages_to_jval_for_template(prefix_msgs, mode, requires_nonnull);
                let pfx_gen_params = TemplateParams {
                    messages: Some(pfx_jval),
                    tools: tools_jval.clone(),
                    add_generation_prompt: true,
                    enable_thinking: true,
                    extra_context: extra.map(JVal::deep_clone_fresh),
                };
                if let Ok(mut rendered_pfx) =
                    tmpl.apply_with_now_object_args(&pfx_gen_params, now_secs, force_object_args)
                {
                    if format_kind == ChatFormat::Gemma4 && rendered_pfx.ends_with("<turn|>\n") {
                        rendered_pfx.push_str("<|turn>model\n");
                    }
                    rendered_pfx.push_str(open_tag);
                    rendered_pfx.push_str(&last.reasoning_content);
                    if !target_reasoning {
                        rendered_pfx.push_str(close_tag);
                        rendered_pfx.push_str(&last.content);
                    }
                    *prompt = rendered_pfx;
                    handled = true;
                }
            }
        }

        if !handled {
            let mut sentinel_msgs = effective_messages.to_vec();
            if let Some(edited) = sentinel_msgs.last_mut() {
                if target_reasoning {
                    edited.reasoning_content.push_str(CONTINUE_SENTINEL);
                } else {
                    edited.content.push_str(CONTINUE_SENTINEL);
                }
            }
            let s_jval = messages_to_jval_for_template(&sentinel_msgs, mode, requires_nonnull);
            let s_params = TemplateParams {
                messages: Some(s_jval),
                tools: tools_jval.clone(),
                add_generation_prompt: false,
                enable_thinking: inputs.enable_thinking,
                extra_context: extra.map(JVal::deep_clone_fresh),
            };
            if let Ok(rendered_s) =
                tmpl.apply_with_now_object_args(&s_params, now_secs, force_object_args)
                && let Some(pos) = rendered_s.find(CONTINUE_SENTINEL)
            {
                *prompt = rendered_s[..pos].to_owned();
            } else {
                let mut probe_msgs = effective_messages.to_vec();
                probe_msgs.pop();
                probe_msgs.push(ChatMessage {
                    role: last.role.clone(),
                    content: "CONTINUE_PROBE_MARK".to_owned(),
                    ..ChatMessage::default()
                });
                let p_jval = messages_to_jval_for_template(&probe_msgs, mode, requires_nonnull);
                let p_params = TemplateParams {
                    messages: Some(p_jval),
                    tools: tools_jval.clone(),
                    add_generation_prompt: false,
                    enable_thinking: inputs.enable_thinking,
                    extra_context: extra.map(JVal::deep_clone_fresh),
                };
                if let Ok(rendered_p) =
                    tmpl.apply_with_now_object_args(&p_params, now_secs, force_object_args)
                    && let Some(pos) = rendered_p.find("CONTINUE_PROBE_MARK")
                {
                    if last.content.is_empty() {
                        *prompt = rendered_p[..pos].to_owned();
                    } else {
                        let suffix = &rendered_p[pos + "CONTINUE_PROBE_MARK".len()..];
                        if !suffix.is_empty()
                            && let Some(stripped) = prompt.strip_suffix(suffix)
                        {
                            *prompt = stripped.to_owned();
                        }
                    }
                }
            }
        }
    }

    if effective_messages.len() > 1 {
        let prefix_msgs = &effective_messages[..effective_messages.len() - 1];
        let pfx_jval = messages_to_jval_for_template(prefix_msgs, mode, requires_nonnull);
        let pfx_params = TemplateParams {
            messages: Some(pfx_jval),
            tools: tools_jval,
            add_generation_prompt: false,
            enable_thinking: inputs.enable_thinking,
            extra_context: extra.map(JVal::deep_clone_fresh),
        };
        if let Ok(prefix_rendered) =
            tmpl.apply_with_now_object_args(&pfx_params, now_secs, force_object_args)
        {
            let lcp = longest_common_prefix_len(prompt, &prefix_rendered);
            return prompt[lcp..].to_owned();
        }
    }
    String::new()
}

fn safe_utf8_prefix(bytes: &[u8]) -> usize {
    let n = bytes.len();
    for back in 1..=3.min(n) {
        let start = n - back;
        let b0 = bytes[start];
        let expected_len = if b0 & 0xE0 == 0xC0 {
            2
        } else if b0 & 0xF0 == 0xE0 {
            3
        } else if b0 & 0xF8 == 0xF0 {
            4
        } else {
            continue;
        };
        if back < expected_len && bytes[start + 1..].iter().all(|b| b & 0xC0 == 0x80) {
            return start;
        }
        break;
    }
    n
}

// ---------------------------------------------------------------------------
// Special-format PEG parser builders
// ---------------------------------------------------------------------------

fn build_reasoning_segment(
    builder: &mut ChatPegBuilder,
    start_tag: &str,
    end_tags: &[&str],
    generation_prompt: &str,
    reasoning_format: i64,
    emit_trailing_space: bool,
) -> Option<ParserId> {
    if end_tags.is_empty() {
        return None;
    }
    let prompt_trim = generation_prompt.trim_end();
    let already_open = !start_tag.is_empty()
        && prompt_trim.ends_with(start_tag.trim_end())
        && !end_tags.iter().any(|e| prompt_trim.ends_with(e.trim_end()));
    let already_closed = end_tags.iter().any(|e| prompt_trim.ends_with(e.trim_end()));

    let end_choice = if end_tags.len() == 1 {
        builder.inner_mut().literal(end_tags[0])
    } else {
        let lits: Vec<ParserId> = end_tags
            .iter()
            .map(|t| builder.inner_mut().literal(*t))
            .collect();
        builder.inner_mut().choice(lits)
    };
    let end_vec: Vec<String> = end_tags.iter().map(|t| (*t).to_owned()).collect();
    let until_end = if end_vec.len() == 1 {
        builder.inner_mut().until(&end_vec[0])
    } else {
        builder.inner_mut().until_one_of(end_vec)
    };

    if reasoning_format == 0 {
        if already_open {
            let seq = builder.inner_mut().sequence(vec![until_end, end_choice]);
            let content_p = builder.content(seq);
            return Some(builder.inner_mut().optional(content_p));
        }
        if already_closed || start_tag.is_empty() {
            return None;
        }
        let open = builder.inner_mut().literal(start_tag);
        let seq = builder
            .inner_mut()
            .sequence(vec![open, until_end, end_choice]);
        let block = builder.content(seq);
        return Some(builder.inner_mut().optional(block));
    }

    let body = builder.reasoning(until_end);

    if already_open {
        let mut seq = vec![body, end_choice];
        if emit_trailing_space {
            seq.push(builder.inner_mut().space());
        }
        let block = builder.inner_mut().sequence(seq);
        return Some(builder.reasoning_block(block));
    }
    if already_closed || start_tag.is_empty() {
        return None;
    }
    let open = builder.inner_mut().literal(start_tag);
    let mut seq = vec![open, body, end_choice];
    if emit_trailing_space {
        seq.push(builder.inner_mut().space());
    }
    let inner = builder.inner_mut().sequence(seq);
    let block = builder.reasoning_block(inner);
    Some(builder.inner_mut().optional(block))
}

fn build_tool_schema_json(
    builder: &mut ChatPegBuilder,
    tool_name: &str,
    params: &JsonVal,
) -> Result<ParserId> {
    let json_p = builder.inner_mut().json()?;
    Ok(builder.schema(
        json_p,
        format!("tool-{tool_name}-args"),
        params.clone(),
        false,
    ))
}

fn extract_tool_defs(tools: &JsonVal) -> Vec<(String, JsonVal)> {
    let mut out = Vec::new();
    for entry in tools.array() {
        let fun = if entry.has("function") {
            entry.get("function")
        } else {
            entry
        };
        let Some(name) = fun.get("name").text() else {
            continue;
        };
        let params = if fun.has("parameters") {
            fun.get("parameters").clone()
        } else {
            JsonVal::Object(vec![("type".to_owned(), JsonVal::Str("object".to_owned()))])
        };
        out.push((name.to_owned(), params));
    }
    out.sort_by_key(|a| std::cmp::Reverse(a.0.len()));
    out
}

fn content_or_schema_parser(
    builder: &mut ChatPegBuilder,
    json_schema: Option<&JsonVal>,
) -> Result<ParserId> {
    if let Some(schema) = json_schema
        && !matches!(schema, JsonVal::Null)
    {
        return builder.schema_content_parser(schema);
    }
    let rest = builder.inner_mut().rest();
    Ok(builder.content(rest))
}

fn lit_tool_name(b: &mut ChatPegBuilder, name: &str) -> ParserId {
    let lit = b.inner_mut().literal(name);
    b.tool_name(lit)
}

fn lit_tool_open(b: &mut ChatPegBuilder, s: &str) -> ParserId {
    let lit = b.inner_mut().literal(s);
    b.tool_open(lit)
}

fn lit_tool_close(b: &mut ChatPegBuilder, s: &str) -> ParserId {
    let lit = b.inner_mut().literal(s);
    b.tool_close(lit)
}

fn seq_tool_open(b: &mut ChatPegBuilder, items: Vec<ParserId>) -> ParserId {
    let s = b.inner_mut().sequence(items);
    b.tool_open(s)
}

fn seq_tool(b: &mut ChatPegBuilder, items: Vec<ParserId>) -> ParserId {
    let s = b.inner_mut().sequence(items);
    b.tool(s)
}

fn seq_tool_arg(b: &mut ChatPegBuilder, items: Vec<ParserId>) -> ParserId {
    let s = b.inner_mut().sequence(items);
    b.tool_arg(s)
}

fn until_tool_arg_str(b: &mut ChatPegBuilder, delim: &str) -> ParserId {
    let u = b.inner_mut().until(delim);
    b.tool_arg_string_value(u)
}

fn opt_lit(b: &mut ChatPegBuilder, s: &str) -> ParserId {
    let lit = b.inner_mut().literal(s);
    b.inner_mut().optional(lit)
}

fn or_lits(b: &mut ChatPegBuilder, a: &str, c: &str) -> ParserId {
    let pa = b.inner_mut().literal(a);
    let pc = b.inner_mut().literal(c);
    b.inner_mut().or(pa, pc)
}

#[expect(
    clippy::too_many_lines,
    reason = "Builds GPT-OSS Harmony multi-channel PEG rules in one coherent pass"
)]
fn build_gpt_oss_parser(
    mut builder: ChatPegBuilder,
    tools: Option<&JsonVal>,
    json_schema: Option<&JsonVal>,
    inputs: &ChatApplyInputs,
    generation_prompt: &str,
) -> Result<(ChatArena, Vec<TriggerDef>, bool)> {
    let tool_defs = tools.map_or_else(Vec::new, extract_tool_defs);
    let has_tools = !tool_defs.is_empty() && inputs.tool_choice != 2;
    let force_tools = has_tools && inputs.tool_choice == 1;

    let mut chan_alts = Vec::new();

    let analysis_hdr = builder
        .inner_mut()
        .literal("<|channel|>analysis<|message|>");
    let until_end = builder.inner_mut().until("<|end|>");
    if inputs.reasoning_format == 0 {
        let end_tok = builder.inner_mut().literal("<|end|>");
        let a_seq = builder
            .inner_mut()
            .sequence(vec![analysis_hdr, until_end, end_tok]);
        chan_alts.push(builder.content(a_seq));
    } else {
        let r_body = builder.reasoning(until_end);
        let end_tok = builder.inner_mut().literal("<|end|>");
        let a_seq = builder
            .inner_mut()
            .sequence(vec![analysis_hdr, r_body, end_tok]);
        chan_alts.push(builder.reasoning_block(a_seq));
    }

    let mut triggers = Vec::new();
    let mut lazy = false;

    if has_tools {
        let mut call_alts = Vec::with_capacity(tool_defs.len() * 2);
        for (name, params) in &tool_defs {
            let schema_p = build_tool_schema_json(&mut builder, name, params)?;
            let tname_1 = lit_tool_name(&mut builder, name);
            let tname_2 = lit_tool_name(&mut builder, name);

            let opt_lead_1 = opt_lit(&mut builder, " ");
            let pfx_to = builder.inner_mut().literal("to=functions.");
            let mid_chan = or_lits(&mut builder, "<|channel|>commentary", "<|channel|>analysis");
            let opt_sp = opt_lit(&mut builder, " ");
            let opt_constrain = opt_lit(&mut builder, "<|constrain|>");
            let opt_json = opt_lit(&mut builder, "json");
            let msg_tok_1 = builder.inner_mut().literal("<|message|>");
            let args_1 = builder.tool_args(schema_p);
            let call_end_1 = or_lits(&mut builder, "<|call|>", "<|end|>");
            let opt_end_1 = builder.inner_mut().optional(call_end_1);
            let open_1_raw = seq_tool_open(
                &mut builder,
                vec![
                    opt_lead_1,
                    pfx_to,
                    tname_1,
                    mid_chan,
                    opt_sp,
                    opt_constrain,
                    opt_json,
                    msg_tok_1,
                ],
            );
            let open_1 = builder.inner_mut().atomic(open_1_raw);
            let close_1 = builder.tool_close(opt_end_1);
            call_alts.push(seq_tool(&mut builder, vec![open_1, args_1, close_1]));

            let chan_pfx = or_lits(
                &mut builder,
                "<|channel|>commentary to=functions.",
                "<|channel|>analysis to=functions.",
            );
            let opt_sp_2 = opt_lit(&mut builder, " ");
            let opt_constrain_2 = opt_lit(&mut builder, "<|constrain|>");
            let opt_json_2 = opt_lit(&mut builder, "json");
            let msg_tok_2 = builder.inner_mut().literal("<|message|>");
            let args_2 = builder.tool_args(schema_p);
            let call_end_2 = or_lits(&mut builder, "<|call|>", "<|end|>");
            let opt_end_2 = builder.inner_mut().optional(call_end_2);
            let open_2_raw = seq_tool_open(
                &mut builder,
                vec![
                    chan_pfx,
                    tname_2,
                    opt_sp_2,
                    opt_constrain_2,
                    opt_json_2,
                    msg_tok_2,
                ],
            );
            let open_2 = builder.inner_mut().atomic(open_2_raw);
            let close_2 = builder.tool_close(opt_end_2);
            call_alts.push(seq_tool(&mut builder, vec![open_2, args_2, close_2]));
        }
        let tool_choice_p = builder.inner_mut().choice(call_alts);
        let trig_rule = builder.trigger_rule("tool-call", tool_choice_p);
        chan_alts.push(trig_rule);
        if !force_tools && json_schema.is_none() {
            lazy = true;
            triggers.push(TriggerDef {
                kind: 1,
                value: "to=functions.".to_owned(),
            });
            triggers.push(TriggerDef {
                kind: 1,
                value: "<|channel|>commentary to=functions.".to_owned(),
            });
            triggers.push(TriggerDef {
                kind: 1,
                value: "<|channel|>analysis to=functions.".to_owned(),
            });
        }
    }

    if !force_tools {
        let final_pfx = builder.inner_mut().literal("<|channel|>final");
        let opt_final_sp = opt_lit(&mut builder, " ");
        let opt_final_constrain = opt_lit(&mut builder, "<|constrain|>json");
        let final_msg_tok = builder.inner_mut().literal("<|message|>");
        let final_hdr = builder.inner_mut().sequence(vec![
            final_pfx,
            opt_final_sp,
            opt_final_constrain,
            final_msg_tok,
        ]);
        let f_body = if let Some(schema) = json_schema
            && !matches!(schema, JsonVal::Null)
        {
            builder.schema_content_parser(schema)?
        } else {
            let until_final_end = builder
                .inner_mut()
                .until_one_of(vec!["<|return|>".to_owned(), "<|end|>".to_owned()]);
            builder.content(until_final_end)
        };
        let end_or = or_lits(&mut builder, "<|return|>", "<|end|>");
        let end_opt = builder.inner_mut().optional(end_or);
        let f_seq = builder
            .inner_mut()
            .sequence(vec![final_hdr, f_body, end_opt]);
        chan_alts.push(f_seq);

        let comm_hdr = builder
            .inner_mut()
            .literal("<|channel|>commentary<|message|>");
        let comm_until = builder
            .inner_mut()
            .until_one_of(vec!["<|end|>".to_owned(), "<|return|>".to_owned()]);
        let comm_body = builder.content(comm_until);
        let comm_end_or = or_lits(&mut builder, "<|end|>", "<|return|>");
        let comm_end = builder.inner_mut().optional(comm_end_or);
        let c_seq = builder
            .inner_mut()
            .sequence(vec![comm_hdr, comm_body, comm_end]);
        chan_alts.push(c_seq);

        // Built-in non-function tool channels (e.g., to=container.exec, commentary to=python)
        let builtin_to_pfx = builder.inner_mut().literal("to=");
        let builtin_to_neg = builder.inner_mut().literal("functions.");
        let builtin_to_guard = builder.inner_mut().negate(builtin_to_neg);
        let builtin_to_mid = builder.inner_mut().until("<|message|>");
        let builtin_to_msg = builder.inner_mut().literal("<|message|>");
        let builtin_to_body = builder.inner_mut().until_one_of(vec![
            "<|end|>".to_owned(),
            "<|call|>".to_owned(),
            "<|return|>".to_owned(),
        ]);
        let builtin_end_alts = vec![
            builder.inner_mut().literal("<|end|>"),
            builder.inner_mut().literal("<|call|>"),
            builder.inner_mut().literal("<|return|>"),
        ];
        let builtin_end_choice = builder.inner_mut().choice(builtin_end_alts);
        let builtin_to_end = builder.inner_mut().optional(builtin_end_choice);
        chan_alts.push(builder.inner_mut().sequence(vec![
            builtin_to_pfx,
            builtin_to_guard,
            builtin_to_mid,
            builtin_to_msg,
            builtin_to_body,
            builtin_to_end,
        ]));

        let comm_to_pfx = builder.inner_mut().literal("<|channel|>commentary to=");
        let comm_to_neg_fn = builder.inner_mut().literal("functions.");
        let comm_to_neg_asst = builder.inner_mut().literal("assistant");
        let comm_to_neg = builder.inner_mut().or(comm_to_neg_fn, comm_to_neg_asst);
        let comm_to_guard = builder.inner_mut().negate(comm_to_neg);
        let comm_to_mid = builder.inner_mut().until("<|message|>");
        let comm_to_msg = builder.inner_mut().literal("<|message|>");
        let comm_to_body = builder.inner_mut().until_one_of(vec![
            "<|end|>".to_owned(),
            "<|call|>".to_owned(),
            "<|return|>".to_owned(),
        ]);
        let comm_end_alts = vec![
            builder.inner_mut().literal("<|end|>"),
            builder.inner_mut().literal("<|call|>"),
            builder.inner_mut().literal("<|return|>"),
        ];
        let comm_end_choice = builder.inner_mut().choice(comm_end_alts);
        let comm_to_end = builder.inner_mut().optional(comm_end_choice);
        chan_alts.push(builder.inner_mut().sequence(vec![
            comm_to_pfx,
            comm_to_guard,
            comm_to_mid,
            comm_to_msg,
            comm_to_body,
            comm_to_end,
        ]));
    }

    let core_chan = builder.inner_mut().choice(chan_alts);
    let opt_assist_wrap = opt_lit(&mut builder, "<|channel|>commentary to=assistant");
    let one_chan = builder
        .inner_mut()
        .sequence(vec![opt_assist_wrap, core_chan]);
    let start_asst = builder.inner_mut().literal("<|start|>assistant");
    let opt_start = builder.inner_mut().optional(start_asst);
    let sp = builder.inner_mut().space();
    let chan_with_sep = builder.inner_mut().sequence(vec![sp, opt_start, one_chan]);
    let rest_chans = builder.inner_mut().zero_or_more(chan_with_sep);
    let first_chan = if generation_prompt.contains("analysis") {
        let until_end = builder.inner_mut().until("<|end|>");
        let end_tok = builder.inner_mut().literal("<|end|>");
        if inputs.reasoning_format == 0 {
            let seq = builder.inner_mut().sequence(vec![until_end, end_tok]);
            builder.content(seq)
        } else {
            let r_body = builder.reasoning(until_end);
            let seq = builder.inner_mut().sequence(vec![r_body, end_tok]);
            builder.reasoning_block(seq)
        }
    } else {
        one_chan
    };

    let lead_sp = if generation_prompt.contains("analysis") {
        builder.inner_mut().eps()
    } else {
        builder.inner_mut().space()
    };
    let full_seq = if force_tools {
        builder
            .inner_mut()
            .sequence(vec![lead_sp, first_chan, rest_chans])
    } else {
        let fallback_rest = content_or_schema_parser(&mut builder, json_schema)?;
        let multi = builder
            .inner_mut()
            .sequence(vec![lead_sp, first_chan, rest_chans]);
        builder.inner_mut().or(multi, fallback_rest)
    };
    builder.set_root(full_seq);
    Ok((builder.build()?, triggers, lazy))
}

fn build_gemma4_value(builder: &mut ChatPegBuilder, prefix: &str) -> Result<ParserId> {
    let q_delim = builder.inner_mut().literal("<|\"|>");
    let str_inner = builder.inner_mut().until("<|\"|>");
    let q_end = builder.inner_mut().literal("<|\"|>");
    let str_val = builder
        .inner_mut()
        .sequence(vec![q_delim, str_inner, q_end]);

    let t_lit = builder.inner_mut().literal("true");
    let f_lit = builder.inner_mut().literal("false");
    let n_lit = builder.inner_mut().literal("null");
    let num = builder.inner_mut().chars("[0-9.+eE-]", 1, -1)?;

    let val_ref_name = format!("{prefix}-val");
    let placeholder = builder.inner_mut().begin_rule_lazy(&val_ref_name, false);
    let val_ref = builder.inner_mut().ref_rule(&val_ref_name);

    let arr_open = builder.inner_mut().literal("[");
    let arr_close = builder.inner_mut().literal("]");
    let comma = builder.inner_mut().literal(",");
    let arr_sep = builder.inner_mut().space_add(comma, val_ref);
    let arr_tail = builder.inner_mut().zero_or_more(arr_sep);
    let arr_items = builder.inner_mut().add(val_ref, arr_tail);
    let arr_inner = builder.inner_mut().optional(arr_items);
    let arr_sp_open = builder.inner_mut().space();
    let arr_sp_close = builder.inner_mut().space();
    let arr = builder.inner_mut().sequence(vec![
        arr_open,
        arr_sp_open,
        arr_inner,
        arr_sp_close,
        arr_close,
    ]);

    let obj_open = builder.inner_mut().literal("{");
    let obj_close = builder.inner_mut().literal("}");
    let key_unq = builder.inner_mut().chars("[a-zA-Z0-9_.-]", 1, -1)?;
    let key_p = builder.inner_mut().or(str_val, key_unq);
    let colon = builder.inner_mut().literal(":");
    let colon_ws_before = builder.inner_mut().space();
    let key_colon_seq = builder
        .inner_mut()
        .sequence(vec![key_p, colon_ws_before, colon]);
    let key_colon = builder.inner_mut().atomic(key_colon_seq);
    let colon_ws_after = builder.inner_mut().space();
    let kv = builder
        .inner_mut()
        .sequence(vec![key_colon, colon_ws_after, val_ref]);
    let obj_comma = builder.inner_mut().literal(",");
    let obj_sep = builder.inner_mut().space_add(obj_comma, kv);
    let obj_tail = builder.inner_mut().zero_or_more(obj_sep);
    let obj_items = builder.inner_mut().add(kv, obj_tail);
    let obj_inner = builder.inner_mut().optional(obj_items);
    let obj_sp_open = builder.inner_mut().space();
    let obj_sp_close = builder.inner_mut().space();
    let obj = builder.inner_mut().sequence(vec![
        obj_open,
        obj_sp_open,
        obj_inner,
        obj_sp_close,
        obj_close,
    ]);

    let val_choice = builder
        .inner_mut()
        .choice(vec![str_val, obj, arr, t_lit, f_lit, n_lit, num]);
    builder
        .inner_mut()
        .finish_rule_lazy(placeholder, val_ref_name, val_choice, false);
    Ok(placeholder)
}

#[expect(
    clippy::too_many_lines,
    reason = "Builds Gemma4 thought channel and structured tool-call PEG rules together"
)]
fn build_gemma4_parser(
    mut builder: ChatPegBuilder,
    tools: Option<&JsonVal>,
    json_schema: Option<&JsonVal>,
    inputs: &ChatApplyInputs,
    generation_prompt: &str,
) -> Result<(ChatArena, Vec<TriggerDef>, bool)> {
    let mut parts = Vec::new();
    let prompt_trim = generation_prompt.trim_end();
    if prompt_trim.ends_with("<|channel>thought")
        || prompt_trim.ends_with("<channel|>")
        || inputs.reasoning_format == 0
    {
        if let Some(r) = build_reasoning_segment(
            &mut builder,
            "<|channel>thought\n",
            &["\n<channel|>", "<channel|>"],
            generation_prompt,
            inputs.reasoning_format,
            false,
        ) {
            parts.push(r);
        }
    } else {
        let open_thought = or_lits(
            &mut builder,
            "<|channel><|channel>thought",
            "<|channel>thought",
        );
        let opt_nl = opt_lit(&mut builder, "\n");
        let until_ch = builder
            .inner_mut()
            .until_one_of(vec!["\n<channel|>".to_owned(), "<channel|>".to_owned()]);
        let r_body = builder.reasoning(until_ch);
        let close_ch = or_lits(&mut builder, "\n<channel|>", "<channel|>");
        let seq = builder
            .inner_mut()
            .sequence(vec![open_thought, opt_nl, r_body, close_ch]);
        let blk = builder.reasoning_block(seq);
        parts.push(builder.inner_mut().optional(blk));
    }

    let make_gemma4_noise = |b: &mut ChatPegBuilder| {
        let empty_t_open = b.inner_mut().literal("<|channel>thought");
        let empty_t_nl = opt_lit(b, "\n");
        let empty_t_close = b.inner_mut().literal("<channel|>");
        let empty_t = b
            .inner_mut()
            .sequence(vec![empty_t_open, empty_t_nl, empty_t_close]);
        let stray_c = b.inner_mut().literal("<channel|>");
        let noise_alt = b.inner_mut().or(empty_t, stray_c);
        b.inner_mut().zero_or_more(noise_alt)
    };

    let tool_defs = tools.map_or_else(Vec::new, extract_tool_defs);
    let has_tools = !tool_defs.is_empty() && inputs.tool_choice != 2;
    let force_tools = has_tools && inputs.tool_choice == 1;
    let mut triggers = Vec::new();
    let mut lazy = false;

    if has_tools {
        let g4_val = build_gemma4_value(&mut builder, "g4")?;
        let mut call_parsers = Vec::with_capacity(tool_defs.len());
        for (name, _) in &tool_defs {
            let open = lit_tool_open(&mut builder, "<|tool_call>call:");
            let tname = lit_tool_name(&mut builder, name);
            let obj_open = builder.inner_mut().literal("{");

            let arg_name_raw = builder.inner_mut().chars("[a-zA-Z0-9_.-]", 1, -1)?;
            let arg_name = builder.tool_arg_name(arg_name_raw);
            let colon = builder.inner_mut().literal(":");
            let arg_key_seq = builder.inner_mut().sequence(vec![arg_name, colon]);
            let arg_key = builder.inner_mut().atomic(arg_key_seq);
            let q1 = builder.inner_mut().literal("<|\"|>");
            let s_val = until_tool_arg_str(&mut builder, "<|\"|>");
            let q2 = builder.inner_mut().literal("<|\"|>");
            let str_arg = builder.inner_mut().sequence(vec![q1, s_val, q2]);
            let j_arg = builder.tool_arg_json_value(g4_val);
            let val_choice = builder.inner_mut().or(str_arg, j_arg);
            let one_arg = seq_tool_arg(&mut builder, vec![arg_key, val_choice]);

            let comma = builder.inner_mut().literal(",");
            let next_arg = builder.inner_mut().sequence(vec![comma, one_arg]);
            let tail_args = builder.inner_mut().zero_or_more(next_arg);
            let all_args = builder.inner_mut().add(one_arg, tail_args);
            let opt_args = builder.inner_mut().optional(all_args);
            let close = lit_tool_close(&mut builder, "}<tool_call|>");
            call_parsers.push(seq_tool(
                &mut builder,
                vec![open, tname, obj_open, opt_args, close],
            ));
        }
        let single_call = builder.inner_mut().choice(call_parsers);
        let calls = if inputs.parallel_tool_calls {
            let rep = builder.inner_mut().zero_or_more(single_call);
            builder.inner_mut().add(single_call, rep)
        } else {
            single_call
        };
        let trig = builder.trigger_rule("tool-call", calls);
        if force_tools {
            parts.push(trig);
        } else if let Some(schema) = json_schema
            && !matches!(schema, JsonVal::Null)
        {
            let opt_content = builder.schema_content_parser(schema)?;
            parts.push(opt_content);
            parts.push(builder.inner_mut().optional(trig));
        } else {
            let until_tc = builder.inner_mut().until_one_of(vec![
                "<|tool_call>".to_owned(),
                "<|channel>thought".to_owned(),
                "<channel|>".to_owned(),
            ]);
            let opt_content = builder.content(until_tc);
            let noise = make_gemma4_noise(&mut builder);
            parts.push(opt_content);
            parts.push(noise);
            parts.push(builder.inner_mut().optional(trig));
            lazy = true;
            triggers.push(TriggerDef {
                kind: 1,
                value: "<|tool_call>".to_owned(),
            });
        }
    } else if json_schema.is_some_and(|s| !matches!(s, JsonVal::Null)) {
        parts.push(content_or_schema_parser(&mut builder, json_schema)?);
    } else {
        let u = builder.inner_mut().until_one_of(vec![
            "<|channel>thought".to_owned(),
            "<channel|>".to_owned(),
        ]);
        parts.push(builder.content(u));
        parts.push(make_gemma4_noise(&mut builder));
    }

    let root = if parts.len() == 1 {
        parts[0]
    } else {
        builder.inner_mut().sequence(parts)
    };
    builder.set_root(root);
    Ok((builder.build()?, triggers, lazy))
}

fn convert_gemma4_object_to_json(raw: &str, partial: bool) -> Option<String> {
    if !raw.contains("<|\"|>") && serde_json::from_str::<Value>(raw).is_ok() {
        return None;
    }
    let bytes = raw.as_bytes();
    let mut out = String::with_capacity(raw.len() + 16);
    let mut i = 0;
    while i < bytes.len() {
        if partial && "<|\"|>".starts_with(&raw[i..]) && raw[i..].len() < 5 {
            break;
        }
        if raw[i..].starts_with("<|\"|>") {
            i += 5;
            let start = i;
            while i < bytes.len() && !raw[i..].starts_with("<|\"|>") {
                if let Some(ch) = raw[i..].chars().next() {
                    i += ch.len_utf8();
                } else {
                    break;
                }
            }
            let mut inner = &raw[start..i];
            if i == bytes.len() && partial {
                for k in (1..5).rev() {
                    if inner.ends_with(&"<|\"|>"[..k]) {
                        inner = &inner[..inner.len() - k];
                        break;
                    }
                }
            }
            if let Ok(quoted) = serde_json::to_string(inner) {
                if i < bytes.len() && raw[i..].starts_with("<|\"|>") {
                    out.push_str(&quoted);
                    i += 5;
                } else if partial {
                    // Omit closing quote while streaming an unterminated Gemma4 string.
                    out.push_str(&quoted[..quoted.len() - 1]);
                } else {
                    out.push_str(&quoted);
                }
            } else if i < bytes.len() && raw[i..].starts_with("<|\"|>") {
                i += 5;
            }
            continue;
        }
        let b = bytes[i];
        if (b == b'{' || b == b',') && i + 1 < bytes.len() {
            out.push(char::from(b));
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                out.push(char::from(bytes[i]));
                i += 1;
            }
            if i < bytes.len()
                && !raw[i..].starts_with("<|\"|>")
                && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_')
            {
                let key_start = i;
                while i < bytes.len()
                    && (bytes[i].is_ascii_alphanumeric() || matches!(bytes[i], b'_' | b'-' | b'.'))
                {
                    i += 1;
                }
                let mut look = i;
                while look < bytes.len() && bytes[look].is_ascii_whitespace() {
                    look += 1;
                }
                if look < bytes.len() && bytes[look] == b':' {
                    if let Ok(quoted) = serde_json::to_string(&raw[key_start..i]) {
                        out.push_str(&quoted);
                    }
                    continue;
                }
                if partial && look >= bytes.len() {
                    break;
                }
                out.push_str(&raw[key_start..i]);
                continue;
            }
            continue;
        }
        if let Some(ch) = raw[i..].chars().next() {
            out.push(ch);
            i += ch.len_utf8();
        } else {
            break;
        }
    }
    Some(out)
}

#[expect(
    clippy::too_many_lines,
    reason = "PEG grammar definition for upstream chat format"
)]
fn build_dsml_parser(
    mut builder: ChatPegBuilder,
    v4: bool,
    tools: Option<&JsonVal>,
    json_schema: Option<&JsonVal>,
    inputs: &ChatApplyInputs,
    generation_prompt: &str,
) -> Result<(ChatArena, Vec<TriggerDef>, bool)> {
    let dsml_tag = if v4 {
        "<｜DSML｜tool_calls>"
    } else {
        "<｜DSML｜function_calls>"
    };
    let dsml_close = if v4 {
        "</｜DSML｜tool_calls>"
    } else {
        "</｜DSML｜function_calls>"
    };

    let tool_defs = tools.map_or_else(Vec::new, extract_tool_defs);
    let has_tools = !tool_defs.is_empty() && inputs.tool_choice != 2;
    let force_tools = has_tools && inputs.tool_choice == 1;
    let mut triggers = Vec::new();
    let mut lazy = false;

    let mut parts = Vec::new();
    let prompt_trim = generation_prompt.trim_end();
    let dsml_already_open = inputs.reasoning_format != 0
        && prompt_trim.ends_with("<think>")
        && !prompt_trim.ends_with("</think>");
    if dsml_already_open && has_tools {
        let u = builder
            .inner_mut()
            .until_one_of(vec!["</think>".to_owned(), dsml_tag.to_owned()]);
        let body = builder.reasoning(u);
        let r_block = builder.reasoning_block(body);
        let opt_close = opt_lit(&mut builder, "</think>");
        let sp = builder.inner_mut().space();
        parts.push(builder.inner_mut().sequence(vec![r_block, opt_close, sp]));
    } else if let Some(r) = build_reasoning_segment(
        &mut builder,
        "<think>",
        &["</think>"],
        generation_prompt,
        inputs.reasoning_format,
        false,
    ) {
        parts.push(r);
    }

    if has_tools {
        let mut call_parsers = Vec::with_capacity(tool_defs.len());
        for (name, _) in &tool_defs {
            let open_p1 = builder.inner_mut().literal("<｜DSML｜invoke name=\"");
            let tname = lit_tool_name(&mut builder, name);
            let open_p2 = builder.inner_mut().literal("\">");
            let sp0 = builder.inner_mut().space();
            let open = seq_tool_open(&mut builder, vec![open_p1, tname, open_p2, sp0]);

            let p_open = builder.inner_mut().literal("<｜DSML｜parameter name=\"");
            let pname_raw = builder.inner_mut().until("\"");
            let pname = builder.tool_arg_name(pname_raw);
            let str_true_mid = builder.inner_mut().literal("\" string=\"true\">");
            let p_val_str = until_tool_arg_str(&mut builder, "</｜DSML｜parameter>");
            let p_close = builder.inner_mut().literal("</｜DSML｜parameter>");
            let str_param = builder
                .inner_mut()
                .sequence(vec![str_true_mid, p_val_str, p_close]);

            let str_false_mid = builder.inner_mut().literal("\" string=\"false\">");
            let json_v = builder.inner_mut().json()?;
            let p_val_json = builder.tool_arg_json_value(json_v);
            let p_close_2 = builder.inner_mut().literal("</｜DSML｜parameter>");
            let json_param =
                builder
                    .inner_mut()
                    .sequence(vec![str_false_mid, p_val_json, p_close_2]);

            let param_body = builder.inner_mut().or(str_param, json_param);
            let sp_arg = builder.inner_mut().space();
            let one_arg = seq_tool_arg(&mut builder, vec![p_open, pname, param_body, sp_arg]);
            let arg_rep = builder.inner_mut().zero_or_more(one_arg);
            let args = builder.tool_args(arg_rep);
            let close = lit_tool_close(&mut builder, "</｜DSML｜invoke>");
            let sp_after = builder.inner_mut().space();
            call_parsers.push(seq_tool(&mut builder, vec![open, args, close, sp_after]));
        }
        let single_call = builder.inner_mut().choice(call_parsers);
        let calls = if inputs.parallel_tool_calls {
            let rep = builder.inner_mut().zero_or_more(single_call);
            builder.inner_mut().add(single_call, rep)
        } else {
            single_call
        };
        let sec_open = builder.inner_mut().literal(dsml_tag);
        let sec_sp = builder.inner_mut().space();
        let sec_close = builder.inner_mut().literal(dsml_close);
        let section = builder
            .inner_mut()
            .sequence(vec![sec_open, sec_sp, calls, sec_close]);
        let trig = builder.trigger_rule("tool-call", section);
        if force_tools {
            parts.push(trig);
        } else {
            let until_tc = builder.inner_mut().until(dsml_tag);
            let c = builder.content(until_tc);
            let opt_trig = builder.inner_mut().optional(trig);
            parts.push(c);
            parts.push(opt_trig);
            if json_schema.is_none() {
                lazy = true;
                triggers.push(TriggerDef {
                    kind: 1,
                    value: dsml_tag.to_owned(),
                });
            }
        }
    } else {
        parts.push(content_or_schema_parser(&mut builder, json_schema)?);
    }

    let root = if parts.len() == 1 {
        parts[0]
    } else {
        builder.inner_mut().sequence(parts)
    };
    builder.set_root(root);
    Ok((builder.build()?, triggers, lazy))
}

#[expect(
    clippy::too_many_lines,
    reason = "PEG grammar definition for upstream chat format"
)]
fn build_deepseek_v31_r1_parser(
    mut builder: ChatPegBuilder,
    template_src: &str,
    tools: Option<&JsonVal>,
    json_schema: Option<&JsonVal>,
    inputs: &ChatApplyInputs,
    generation_prompt: &str,
) -> Result<(ChatArena, Vec<TriggerDef>, bool)> {
    let tool_defs = tools.map_or_else(Vec::new, extract_tool_defs);
    let has_tools = !tool_defs.is_empty() && inputs.tool_choice != 2;
    let force_tools = has_tools && inputs.tool_choice == 1;
    let is_r1 = template_src.contains("tool['type']")
        || template_src.contains("tc['type']")
        || template_src.contains("function<｜tool▁sep｜>");

    let mut parts = Vec::new();
    let prompt_trim = generation_prompt.trim_end();
    let already_open = prompt_trim.ends_with("<think>") && !prompt_trim.ends_with("</think>");
    let already_closed = prompt_trim.ends_with("</think>");
    if inputs.reasoning_format == 0 {
        if let Some(r) = build_reasoning_segment(
            &mut builder,
            "<think>",
            &["</think>"],
            generation_prompt,
            inputs.reasoning_format,
            false,
        ) {
            parts.push(r);
        }
    } else if already_open {
        let u = builder.inner_mut().until("</think>");
        let body = builder.reasoning(u);
        let close = builder.inner_mut().literal("</think>");
        let sp = builder.inner_mut().space();
        let r_sp = builder.reasoning(sp);
        let inner = builder.inner_mut().sequence(vec![body, close, r_sp]);
        let r_blk = builder.reasoning_block(inner);
        let opt_r = if !force_tools && json_schema.is_none() {
            builder.inner_mut().optional(r_blk)
        } else {
            r_blk
        };
        parts.push(opt_r);
    } else if !already_closed {
        let open = builder.inner_mut().literal("<think>");
        let u = builder.inner_mut().until("</think>");
        let body = builder.reasoning(u);
        let close = builder.inner_mut().literal("</think>");
        let sp = builder.inner_mut().space();
        let r_sp = builder.reasoning(sp);
        let inner = builder.inner_mut().sequence(vec![open, body, close, r_sp]);
        let r_blk = builder.reasoning_block(inner);
        parts.push(builder.inner_mut().optional(r_blk));
    }

    let mut triggers = Vec::new();
    let mut lazy = false;

    if has_tools {
        let mut call_parsers = Vec::with_capacity(tool_defs.len());
        for (name, params) in &tool_defs {
            let schema_p = build_tool_schema_json(&mut builder, name, params)?;
            if is_r1 {
                let p1 = builder
                    .inner_mut()
                    .literal("<｜tool▁call▁begin｜>function<｜tool▁sep｜>");
                let tname = lit_tool_name(&mut builder, name);
                let sp1 = builder.inner_mut().space();
                let opt_fence = opt_lit(&mut builder, "```json");
                let sp2 = builder.inner_mut().space();
                let open_raw = seq_tool_open(&mut builder, vec![p1, tname, sp1, opt_fence, sp2]);
                let open = builder.inner_mut().atomic(open_raw);
                let args = builder.tool_args(schema_p);
                let sp_c = builder.inner_mut().space();
                let opt_fence_end = opt_lit(&mut builder, "```");
                let close_lit = builder.inner_mut().literal("<｜tool▁call▁end｜>");
                let close_seq = builder
                    .inner_mut()
                    .sequence(vec![sp_c, opt_fence_end, close_lit]);
                let close_blk = builder.tool_close(close_seq);
                call_parsers.push(seq_tool(&mut builder, vec![open, args, close_blk]));
            } else {
                let p1 = builder.inner_mut().literal("<｜tool▁call▁begin｜>");
                let tname = lit_tool_name(&mut builder, name);
                let p2 = builder.inner_mut().literal("<｜tool▁sep｜>");
                let open_raw = seq_tool_open(&mut builder, vec![p1, tname, p2]);
                let open = builder.inner_mut().atomic(open_raw);
                let args = builder.tool_args(schema_p);
                let close = lit_tool_close(&mut builder, "<｜tool▁call▁end｜>");
                call_parsers.push(seq_tool(&mut builder, vec![open, args, close]));
            }
        }
        let single_call = builder.inner_mut().choice(call_parsers);
        let calls = if inputs.parallel_tool_calls {
            let sp = builder.inner_mut().space();
            let step = builder.inner_mut().add(sp, single_call);
            let tail = builder.inner_mut().zero_or_more(step);
            builder.inner_mut().add(single_call, tail)
        } else {
            single_call
        };
        let sec_open = builder.inner_mut().literal("<｜tool▁calls▁begin｜>");
        let sec_sp1 = builder.inner_mut().space();
        let sec_sp2 = builder.inner_mut().space();
        let sec_close = builder.inner_mut().literal("<｜tool▁calls▁end｜>");
        let section = builder
            .inner_mut()
            .sequence(vec![sec_open, sec_sp1, calls, sec_sp2, sec_close]);
        let trig = builder.trigger_rule("tool-call", section);
        if force_tools {
            parts.push(trig);
        } else {
            let until_tc = builder.inner_mut().until("<｜tool▁calls▁begin｜>");
            let c = builder.content(until_tc);
            let opt_t = builder.inner_mut().optional(trig);
            parts.push(c);
            parts.push(opt_t);
            if json_schema.is_none() {
                lazy = true;
                triggers.push(TriggerDef {
                    kind: 1,
                    value: "<｜tool▁calls▁begin｜>".to_owned(),
                });
            }
        }
    } else {
        parts.push(content_or_schema_parser(&mut builder, json_schema)?);
    }

    let root = if parts.len() == 1 {
        parts[0]
    } else {
        builder.inner_mut().sequence(parts)
    };
    builder.set_root(root);
    Ok((builder.build()?, triggers, lazy))
}

#[expect(
    clippy::too_many_lines,
    reason = "PEG grammar definition for upstream chat format"
)]
fn build_kimi_k2_parser(
    mut builder: ChatPegBuilder,
    tools: Option<&JsonVal>,
    json_schema: Option<&JsonVal>,
    inputs: &ChatApplyInputs,
    generation_prompt: &str,
) -> Result<(ChatArena, Vec<TriggerDef>, bool)> {
    let tool_defs = tools.map_or_else(Vec::new, extract_tool_defs);
    let has_tools = !tool_defs.is_empty() && inputs.tool_choice != 2;
    let force_tools = has_tools && inputs.tool_choice == 1;
    let mut triggers = Vec::new();
    let mut lazy = false;

    let mut parts = Vec::new();
    if inputs.reasoning_format == 0 {
        if let Some(r) = build_reasoning_segment(
            &mut builder,
            "<think>",
            &["</think>"],
            generation_prompt,
            inputs.reasoning_format,
            false,
        ) {
            parts.push(r);
        }
    } else {
        let prompt_trim = generation_prompt.trim_end();
        let already_open = prompt_trim.ends_with("<think>") && !prompt_trim.ends_with("</think>");
        let already_closed = prompt_trim.ends_with("</think>");
        let until_r = if has_tools {
            builder.inner_mut().until_one_of(vec![
                "</think>".to_owned(),
                "<|tool_calls_section_begin|>".to_owned(),
                "<|tool_call_begin|>".to_owned(),
            ])
        } else {
            builder.inner_mut().until("</think>")
        };
        let r_body = builder.reasoning(until_r);
        let close_tok = if has_tools {
            opt_lit(&mut builder, "</think>")
        } else {
            builder.inner_mut().literal("</think>")
        };
        if already_open {
            let block_inner = builder.inner_mut().sequence(vec![r_body, close_tok]);
            let r_blk = builder.reasoning_block(block_inner);
            parts.push(if !force_tools && json_schema.is_none() {
                builder.inner_mut().optional(r_blk)
            } else {
                r_blk
            });
        } else if !already_closed {
            let open_tok = builder.inner_mut().literal("<think>");
            let block_inner = builder
                .inner_mut()
                .sequence(vec![open_tok, r_body, close_tok]);
            let block = builder.reasoning_block(block_inner);
            parts.push(builder.inner_mut().optional(block));
        }
    }

    if has_tools {
        let mut call_parsers = Vec::with_capacity(tool_defs.len());
        for (name, params) in &tool_defs {
            let schema_p = build_tool_schema_json(&mut builder, name, params)?;
            let open_pfx = builder.inner_mut().literal("<|tool_call_begin|>");
            let opt_fn_pfx = opt_lit(&mut builder, "functions.");
            let name_lit = builder.inner_mut().literal(name);
            let colon_lit = builder.inner_mut().literal(":");
            let digits = builder.inner_mut().chars("[0-9]", 1, -1)?;
            let colon_digits = builder.inner_mut().sequence(vec![colon_lit, digits]);
            let opt_idx = builder.inner_mut().optional(colon_digits);
            let id_body = builder
                .inner_mut()
                .sequence(vec![opt_fn_pfx, name_lit, opt_idx]);
            let id_tagged = builder.tool_id(id_body);
            let arg_beg = builder.inner_mut().literal("<|tool_call_argument_begin|>");
            let open_seq = seq_tool_open(&mut builder, vec![open_pfx, id_tagged, arg_beg]);
            let open = builder.inner_mut().atomic(open_seq);
            let args = builder.tool_args(schema_p);
            let close_raw = opt_lit(&mut builder, "<|tool_call_end|>");
            let close = builder.tool_close(close_raw);
            call_parsers.push(seq_tool(&mut builder, vec![open, args, close]));
        }
        let single_call = builder.inner_mut().choice(call_parsers);
        let calls = if inputs.parallel_tool_calls {
            let rep = builder.inner_mut().zero_or_more(single_call);
            builder.inner_mut().add(single_call, rep)
        } else {
            single_call
        };
        let sec_open = opt_lit(&mut builder, "<|tool_calls_section_begin|>");
        let sec_close = opt_lit(&mut builder, "<|tool_calls_section_end|>");
        let section = builder
            .inner_mut()
            .sequence(vec![sec_open, calls, sec_close]);
        let trig = builder.trigger_rule("tool-call", section);
        if force_tools {
            parts.push(trig);
        } else {
            let until_tc = builder.inner_mut().until_one_of(vec![
                "<|tool_calls_section_begin|>".to_owned(),
                "<|tool_call_begin|>".to_owned(),
            ]);
            let c = builder.content(until_tc);
            let opt_trig = builder.inner_mut().optional(trig);
            parts.push(c);
            parts.push(opt_trig);
            if json_schema.is_none() {
                lazy = true;
                triggers.push(TriggerDef {
                    kind: 1,
                    value: "<|tool_calls_section_begin|>".to_owned(),
                });
                triggers.push(TriggerDef {
                    kind: 1,
                    value: "<|tool_call_begin|>".to_owned(),
                });
            }
        }
    } else {
        parts.push(content_or_schema_parser(&mut builder, json_schema)?);
    }

    let root = if parts.len() == 1 {
        parts[0]
    } else {
        builder.inner_mut().sequence(parts)
    };
    builder.set_root(root);
    Ok((builder.build()?, triggers, lazy))
}

#[expect(
    clippy::too_many_lines,
    reason = "Builds Kimi K3 XTML reasoning/response/tool PEG grammar together"
)]
fn build_kimi_k3_parser(
    mut builder: ChatPegBuilder,
    tools: Option<&JsonVal>,
    json_schema: Option<&JsonVal>,
    inputs: &ChatApplyInputs,
    generation_prompt: &str,
) -> Result<(ChatArena, Vec<TriggerDef>, bool)> {
    let prompt_trim = generation_prompt.trim_end();
    let think_open_tag = "<|open|>think<|sep|>";
    let think_close_tag = "<|close|>think<|sep|>";
    let think_already_open = inputs.reasoning_format != 0
        && prompt_trim.ends_with(think_open_tag)
        && !prompt_trim.ends_with(think_close_tag);

    let mut post_think_parts = Vec::new();
    let resp_already_open = prompt_trim.ends_with("<|open|>response<|sep|>");
    if resp_already_open {
        let u = builder.inner_mut().until("<|close|>response<|sep|>");
        let resp_body = builder.content(u);
        let resp_close = builder.inner_mut().literal("<|close|>response<|sep|>");
        let seq = builder.inner_mut().sequence(vec![resp_body, resp_close]);
        post_think_parts.push(seq);
    } else {
        let resp_open = builder.inner_mut().literal("<|open|>response<|sep|>");
        let resp_body = if let Some(schema) = json_schema
            && !matches!(schema, JsonVal::Null)
        {
            builder.schema_content_parser(schema)?
        } else {
            let u = builder.inner_mut().until("<|close|>response<|sep|>");
            builder.content(u)
        };
        let resp_close = builder.inner_mut().literal("<|close|>response<|sep|>");
        let resp_block = builder
            .inner_mut()
            .sequence(vec![resp_open, resp_body, resp_close]);
        let opt_resp = builder.inner_mut().optional(resp_block);
        post_think_parts.push(opt_resp);
    }

    let tool_defs = tools.map_or_else(Vec::new, extract_tool_defs);
    let has_tools = !tool_defs.is_empty() && inputs.tool_choice != 2;
    let force_tools = has_tools && inputs.tool_choice == 1;
    let mut triggers = Vec::new();
    let mut lazy = false;

    if has_tools {
        let mut call_parsers = Vec::with_capacity(tool_defs.len());
        for (name, params) in &tool_defs {
            let p1 = builder.inner_mut().literal("<|open|>call tool=\"");
            let tname = lit_tool_name(&mut builder, name);
            let p2 = builder.inner_mut().until("<|sep|>");
            let p3 = builder.inner_mut().literal("<|sep|>");
            let open = seq_tool_open(&mut builder, vec![p1, tname, p2, p3]);

            let schema_p = build_tool_schema_json(&mut builder, name, params)?;
            let json_open = builder
                .inner_mut()
                .literal("<|open|>json type=\"object\"<|sep|>");
            let json_close = builder.inner_mut().literal("<|close|>json<|sep|>");
            let json_block = builder.tool_args(schema_p);
            let json_alt = builder
                .inner_mut()
                .sequence(vec![json_open, json_block, json_close]);

            let arg_open_p = builder.inner_mut().literal("<|open|>argument key=\"");
            let arg_key_raw = builder.inner_mut().until("\"");
            let arg_key = builder.tool_arg_name(arg_key_raw);
            let str_mid = builder.inner_mut().literal("\" type=\"string\"<|sep|>");
            let str_val = until_tool_arg_str(&mut builder, "<|close|>argument<|sep|>");
            let arg_end_str = builder.inner_mut().literal("<|close|>argument<|sep|>");
            let str_arg = builder
                .inner_mut()
                .sequence(vec![str_mid, str_val, arg_end_str]);

            let t_pfx = builder.inner_mut().literal("\" type=\"");
            let t_mid = builder.inner_mut().until("<|sep|>");
            let t_sep = builder.inner_mut().literal("<|sep|>");
            let other_mid = builder.inner_mut().sequence(vec![t_pfx, t_mid, t_sep]);
            let j_raw = builder.inner_mut().json()?;
            let other_val = builder.tool_arg_json_value(j_raw);
            let arg_end_oth = builder.inner_mut().literal("<|close|>argument<|sep|>");
            let oth_arg = builder
                .inner_mut()
                .sequence(vec![other_mid, other_val, arg_end_oth]);

            let arg_val = builder.inner_mut().or(str_arg, oth_arg);
            let one_arg = seq_tool_arg(&mut builder, vec![arg_open_p, arg_key, arg_val]);
            let arg_rep = builder.inner_mut().zero_or_more(one_arg);
            let constructed_args = builder.tool_args(arg_rep);
            let args_alt = builder.inner_mut().or(json_alt, constructed_args);
            let close = lit_tool_close(&mut builder, "<|close|>call<|sep|>");
            call_parsers.push(seq_tool(&mut builder, vec![open, args_alt, close]));
        }
        let single_call = builder.inner_mut().choice(call_parsers);
        let calls = if inputs.parallel_tool_calls {
            let rep = builder.inner_mut().zero_or_more(single_call);
            builder.inner_mut().add(single_call, rep)
        } else {
            single_call
        };
        let sec_open = builder.inner_mut().literal("<|open|>tools<|sep|>");
        let sec_close = builder.inner_mut().literal("<|close|>tools<|sep|>");
        let opt_msg_close = opt_lit(&mut builder, "<|close|>message<|sep|>");
        let section = builder
            .inner_mut()
            .sequence(vec![sec_open, calls, sec_close, opt_msg_close]);
        let trig = builder.trigger_rule("tool-call", section);
        if force_tools {
            post_think_parts.push(trig);
        } else {
            let opt_t = builder.inner_mut().optional(trig);
            post_think_parts.push(opt_t);
            if json_schema.is_none() {
                lazy = true;
                triggers.push(TriggerDef {
                    kind: 1,
                    value: "<|open|>tools<|sep|>".to_owned(),
                });
            }
        }
    } else {
        post_think_parts.push(opt_lit(&mut builder, "<|close|>message<|sep|>"));
    }

    let post_think = if post_think_parts.len() == 1 {
        post_think_parts[0]
    } else {
        builder.inner_mut().sequence(post_think_parts)
    };

    let root = if think_already_open {
        let u = builder.inner_mut().until(think_close_tag);
        let body = builder.reasoning(u);
        let r_block = builder.reasoning_block(body);
        let close_t = builder.inner_mut().literal(think_close_tag);
        let closed = builder.inner_mut().sequence(vec![close_t, post_think]);
        let after_think = if !force_tools && json_schema.is_none() {
            builder.inner_mut().optional(closed)
        } else {
            closed
        };
        builder.inner_mut().sequence(vec![r_block, after_think])
    } else if inputs.reasoning_format == 0
        && prompt_trim.ends_with(think_open_tag)
        && !prompt_trim.ends_with(think_close_tag)
    {
        let start_post = or_lits(
            &mut builder,
            "<|open|>response<|sep|>",
            "<|open|>tools<|sep|>",
        );
        let peek_post = builder.inner_mut().peek(start_post);
        let direct_post = builder.inner_mut().sequence(vec![peek_post, post_think]);
        let u = builder.inner_mut().until(think_close_tag);
        let close_t = builder.inner_mut().literal(think_close_tag);
        let r_seq = builder.inner_mut().sequence(vec![u, close_t]);
        let r_content = builder.content(r_seq);
        let with_think = builder.inner_mut().sequence(vec![r_content, post_think]);
        builder.inner_mut().or(direct_post, with_think)
    } else if let Some(r) = build_reasoning_segment(
        &mut builder,
        think_open_tag,
        &[think_close_tag],
        generation_prompt,
        inputs.reasoning_format,
        false,
    ) {
        builder.inner_mut().sequence(vec![r, post_think])
    } else {
        post_think
    };
    builder.set_root(root);
    Ok((builder.build()?, triggers, lazy))
}

#[expect(
    clippy::too_many_lines,
    reason = "PEG grammar definition for upstream chat format"
)]
fn build_lfm2_parser(
    mut builder: ChatPegBuilder,
    tools: Option<&JsonVal>,
    json_schema: Option<&JsonVal>,
    inputs: &ChatApplyInputs,
    generation_prompt: &str,
) -> Result<(ChatArena, Vec<TriggerDef>, bool)> {
    let mut parts = Vec::new();
    if let Some(r) = build_reasoning_segment(
        &mut builder,
        "<think>",
        &["</think>"],
        generation_prompt,
        inputs.reasoning_format,
        false,
    ) {
        parts.push(r);
    }

    let tool_defs = tools.map_or_else(Vec::new, extract_tool_defs);
    let has_tools = !tool_defs.is_empty() && inputs.tool_choice != 2;
    let force_tools = has_tools && inputs.tool_choice == 1;
    let mut triggers = Vec::new();
    let mut lazy = false;

    if has_tools {
        let mut call_parsers = Vec::with_capacity(tool_defs.len());
        for (name, _) in &tool_defs {
            let eps = builder.inner_mut().eps();
            let open = builder.tool_open(eps);
            let tname = lit_tool_name(&mut builder, name);
            let lparen = builder.inner_mut().literal("(");

            let arg_name_raw = builder.inner_mut().chars("[a-zA-Z0-9_]", 1, -1)?;
            let arg_name = builder.tool_arg_name(arg_name_raw);
            let eq = builder.inner_mut().literal("=");
            let arg_key_seq = builder.inner_mut().sequence(vec![arg_name, eq]);
            let arg_key = builder.inner_mut().atomic(arg_key_seq);

            let ch_tab = builder.inner_mut().literal("\t");
            let ch_lf = builder.inner_mut().literal("\n");
            let ch_cr = builder.inner_mut().literal("\r");
            let ws_ctrl = builder.inner_mut().choice(vec![ch_tab, ch_lf, ch_cr]);
            let sq_1 = builder.inner_mut().literal("'");
            let sq_sc_1 = builder.inner_mut().string_content(b'\'');
            let sq_sc_2 = builder.inner_mut().string_content(b'\'');
            let sq_ws_pair = builder.inner_mut().sequence(vec![ws_ctrl, sq_sc_2]);
            let sq_ws_tail = builder.inner_mut().zero_or_more(sq_ws_pair);
            let sq_raw = builder.inner_mut().sequence(vec![sq_sc_1, sq_ws_tail]);
            let sq_body = builder.tool_arg_string_value(sq_raw);
            let sq_2 = builder.inner_mut().literal("'");
            let sq_str = builder.inner_mut().sequence(vec![sq_1, sq_body, sq_2]);

            let dq_1 = builder.inner_mut().literal("\"");
            let dq_sc_1 = builder.inner_mut().string_content(b'"');
            let dq_sc_2 = builder.inner_mut().string_content(b'"');
            let dq_ws_pair = builder.inner_mut().sequence(vec![ws_ctrl, dq_sc_2]);
            let dq_ws_tail = builder.inner_mut().zero_or_more(dq_ws_pair);
            let dq_raw = builder.inner_mut().sequence(vec![dq_sc_1, dq_ws_tail]);
            let dq_2 = builder.inner_mut().literal("\"");
            let dq_seq = builder.inner_mut().sequence(vec![dq_1, dq_raw, dq_2]);
            let dq_str = builder.tool_arg_json_value(dq_seq);

            let j_raw = builder.inner_mut().json()?;
            let j_val = builder.tool_arg_json_value(j_raw);
            let py_raw = builder.inner_mut().python_value()?;
            let py_val = builder.tool_arg_json_value(py_raw);
            let val_choice = builder
                .inner_mut()
                .choice(vec![dq_str, sq_str, py_val, j_val]);
            let one_arg = seq_tool_arg(&mut builder, vec![arg_key, val_choice]);

            let comma = builder.inner_mut().literal(",");
            let sep_sp = builder.inner_mut().space_add(comma, one_arg);
            let arg_tail = builder.inner_mut().zero_or_more(sep_sp);
            let all_args = builder.inner_mut().add(one_arg, arg_tail);
            let opt_args = builder.inner_mut().optional(all_args);
            let close = lit_tool_close(&mut builder, ")");
            call_parsers.push(seq_tool(
                &mut builder,
                vec![open, tname, lparen, opt_args, close],
            ));
        }
        let single_call = builder.inner_mut().choice(call_parsers);
        let calls = if inputs.parallel_tool_calls {
            let comma = builder.inner_mut().literal(",");
            let sep_call = builder.inner_mut().space_add(comma, single_call);
            let tail = builder.inner_mut().zero_or_more(sep_call);
            builder.inner_mut().add(single_call, tail)
        } else {
            single_call
        };
        let sec_open = builder.inner_mut().literal("<|tool_call_start|>[");
        let sec_close = builder.inner_mut().literal("]<|tool_call_end|>");
        let section = builder
            .inner_mut()
            .sequence(vec![sec_open, calls, sec_close]);
        let trig = builder.trigger_rule("tool-call", section);
        if force_tools {
            parts.push(trig);
        } else {
            let until_tc = builder.inner_mut().until("<|tool_call_start|>");
            let c = builder.content(until_tc);
            let opt_t = builder.inner_mut().optional(trig);
            parts.push(c);
            parts.push(opt_t);
            if json_schema.is_none() {
                lazy = true;
                triggers.push(TriggerDef {
                    kind: 1,
                    value: "<|tool_call_start|>".to_owned(),
                });
            }
        }
    } else {
        parts.push(content_or_schema_parser(&mut builder, json_schema)?);
    }

    let root = if parts.len() == 1 {
        parts[0]
    } else {
        builder.inner_mut().sequence(parts)
    };
    builder.set_root(root);
    Ok((builder.build()?, triggers, lazy))
}

fn schema_is_pure_scalar(schema: &JsonVal) -> bool {
    if schema_resolves_to_string(schema) {
        return false;
    }
    if let Some(t) = schema.get("type").text() {
        return matches!(t, "integer" | "number" | "boolean" | "null");
    }
    for key in ["anyOf", "oneOf"] {
        let branches = schema.get(key).array();
        if !branches.is_empty() {
            return branches.iter().all(schema_is_pure_scalar);
        }
    }
    false
}

#[expect(
    clippy::too_many_lines,
    reason = "PEG grammar definition for upstream chat format"
)]
fn build_minimax_m3_parser(
    mut builder: ChatPegBuilder,
    tools: Option<&JsonVal>,
    json_schema: Option<&JsonVal>,
    inputs: &ChatApplyInputs,
    generation_prompt: &str,
) -> Result<(ChatArena, Vec<TriggerDef>, bool)> {
    let mut parts = Vec::new();
    let prompt_trim = generation_prompt.trim_end();
    let already_open = prompt_trim.ends_with("<mm:think>") && !prompt_trim.ends_with("</mm:think>");
    let already_closed = prompt_trim.ends_with("</mm:think>");
    if already_open || already_closed || inputs.reasoning_format == 0 {
        if let Some(r) = build_reasoning_segment(
            &mut builder,
            "<mm:think>",
            &["</mm:think>"],
            generation_prompt,
            inputs.reasoning_format,
            false,
        ) {
            parts.push(r);
        }
    } else {
        let open_r = builder.inner_mut().literal("<mm:think>");
        let until_r = builder.inner_mut().until("</mm:think>");
        let r_body = builder.reasoning(until_r);
        let close_r = builder.inner_mut().literal("</mm:think>");
        let full_r = builder.inner_mut().sequence(vec![open_r, r_body, close_r]);
        let empty_close = builder.inner_mut().literal("</mm:think>");
        let either_r = builder.inner_mut().or(full_r, empty_close);
        let blk = builder.reasoning_block(either_r);
        parts.push(builder.inner_mut().optional(blk));
    }

    let tool_defs = tools.map_or_else(Vec::new, extract_tool_defs);
    let has_tools = !tool_defs.is_empty() && inputs.tool_choice != 2;
    let force_tools = has_tools && inputs.tool_choice == 1;
    let mut triggers = Vec::new();
    let mut lazy = false;

    if has_tools {
        let mut call_parsers = Vec::with_capacity(tool_defs.len());
        for (name, params) in &tool_defs {
            let p1 = builder.inner_mut().literal("]<]minimax[>[<invoke name=\"");
            let tname = lit_tool_name(&mut builder, name);
            let p2 = builder.inner_mut().literal("\">");
            let sp0 = builder.inner_mut().space();
            let open_raw = seq_tool_open(&mut builder, vec![p1, tname, p2, sp0]);
            let open = builder.inner_mut().atomic(open_raw);

            let props = params.get("properties").object();
            let arg_rep = if props.is_empty() {
                let a_open_lit = builder.inner_mut().literal("]<]minimax[>[<");
                let a_name_raw = builder.inner_mut().chars("[a-zA-Z0-9_-]", 1, -1)?;
                let a_name = builder.tool_arg_name(a_name_raw);
                let a_gt = builder.inner_mut().literal(">");
                let a_key_seq = builder.inner_mut().sequence(vec![a_open_lit, a_name, a_gt]);
                let a_key = builder.inner_mut().atomic(a_key_seq);
                let a_val = until_tool_arg_str(&mut builder, "]<]minimax[>[</");
                let a_close_pfx = builder.inner_mut().literal("]<]minimax[>[</");
                let a_close_name = builder.inner_mut().chars("[a-zA-Z0-9_-]", 1, -1)?;
                let a_close_gt = builder.inner_mut().literal(">");
                let sp_arg = builder.inner_mut().space();
                let one_arg = seq_tool_arg(
                    &mut builder,
                    vec![a_key, a_val, a_close_pfx, a_close_name, a_close_gt, sp_arg],
                );
                builder.inner_mut().zero_or_more(one_arg)
            } else {
                let mut prop_parsers = Vec::with_capacity(props.len());
                for (prop_name, prop_schema) in props {
                    let close_tag = format!("]<]minimax[>[</{prop_name}>");
                    let a_open_lit = builder.inner_mut().literal("]<]minimax[>[<");
                    let a_name_lit = builder.inner_mut().literal(prop_name);
                    let a_name = builder.tool_arg_name(a_name_lit);
                    let a_gt = builder.inner_mut().literal(">");
                    let a_key_seq = builder.inner_mut().sequence(vec![a_open_lit, a_name, a_gt]);
                    let a_key = builder.inner_mut().atomic(a_key_seq);
                    let a_val = if schema_is_pure_scalar(prop_schema) {
                        let jp = builder.inner_mut().json()?;
                        builder.tool_arg_json_value(jp)
                    } else {
                        until_tool_arg_str(&mut builder, &close_tag)
                    };
                    let a_close = builder.inner_mut().literal(&close_tag);
                    let sp_arg = builder.inner_mut().space();
                    prop_parsers.push(seq_tool_arg(
                        &mut builder,
                        vec![a_key, a_val, a_close, sp_arg],
                    ));
                }
                let one_arg = builder.inner_mut().choice(prop_parsers);
                builder.inner_mut().zero_or_more(one_arg)
            };
            let args = builder.tool_args(arg_rep);
            let close = lit_tool_close(&mut builder, "]<]minimax[>[</invoke>");
            let sp_after = builder.inner_mut().space();
            call_parsers.push(seq_tool(&mut builder, vec![open, args, close, sp_after]));
        }
        let single_call = builder.inner_mut().choice(call_parsers);
        let calls = if inputs.parallel_tool_calls {
            let rep = builder.inner_mut().zero_or_more(single_call);
            builder.inner_mut().add(single_call, rep)
        } else {
            single_call
        };
        let sec_open = builder.inner_mut().literal("]<]minimax[>[<tool_call>");
        let sec_sp = builder.inner_mut().space();
        let sec_close = builder.inner_mut().literal("]<]minimax[>[</tool_call>");
        let section = builder
            .inner_mut()
            .sequence(vec![sec_open, sec_sp, calls, sec_close]);
        let trig = builder.trigger_rule("tool-call", section);
        if force_tools {
            parts.push(trig);
        } else {
            let until_tc = builder.inner_mut().until("]<]minimax[>[<tool_call>");
            let c = builder.content(until_tc);
            let opt_t = builder.inner_mut().optional(trig);
            parts.push(c);
            parts.push(opt_t);
            if json_schema.is_none() {
                lazy = true;
                triggers.push(TriggerDef {
                    kind: 1,
                    value: "]<]minimax[>[<tool_call>".to_owned(),
                });
            }
        }
    } else {
        parts.push(content_or_schema_parser(&mut builder, json_schema)?);
    }

    let root = if parts.len() == 1 {
        parts[0]
    } else {
        builder.inner_mut().sequence(parts)
    };
    builder.set_root(root);
    Ok((builder.build()?, triggers, lazy))
}

fn build_minicpm5_parser(
    mut builder: ChatPegBuilder,
    tools: Option<&JsonVal>,
    json_schema: Option<&JsonVal>,
    inputs: &ChatApplyInputs,
    generation_prompt: &str,
) -> Result<(ChatArena, Vec<TriggerDef>, bool)> {
    let mut parts = Vec::new();
    if let Some(r) = build_reasoning_segment(
        &mut builder,
        "<think>",
        &["\n</think>\n\n", "</think>"],
        generation_prompt,
        inputs.reasoning_format,
        true,
    ) {
        parts.push(r);
    }

    let tool_defs = tools.map_or_else(Vec::new, extract_tool_defs);
    let has_tools = !tool_defs.is_empty() && inputs.tool_choice != 2;
    let force_tools = has_tools && inputs.tool_choice == 1;
    let mut triggers = Vec::new();
    let mut lazy = false;

    if has_tools {
        let mut call_parsers = Vec::with_capacity(tool_defs.len());
        for (name, _) in &tool_defs {
            let p1 = builder.inner_mut().literal("<function name=\"");
            let tname = lit_tool_name(&mut builder, name);
            let p2 = builder.inner_mut().literal("\">");
            let open_seq = seq_tool_open(&mut builder, vec![p1, tname, p2]);
            let open = builder.inner_mut().atomic(open_seq);

            let a_open = builder.inner_mut().literal("<param name=\"");
            let a_name_raw = builder.inner_mut().until("\">");
            let a_name = builder.tool_arg_name(a_name_raw);
            let a_mid = builder.inner_mut().literal("\">");
            let a_key_seq = builder.inner_mut().sequence(vec![a_open, a_name, a_mid]);
            let a_key = builder.inner_mut().atomic(a_key_seq);
            let cdata_open = builder.inner_mut().literal("<![CDATA[");
            let cdata_body = until_tool_arg_str(&mut builder, "]]>");
            let cdata_close = builder.inner_mut().literal("]]>");
            let cdata_alt = builder
                .inner_mut()
                .sequence(vec![cdata_open, cdata_body, cdata_close]);
            let plain_body = until_tool_arg_str(&mut builder, "</param>");
            let val_p = builder.inner_mut().or(cdata_alt, plain_body);
            let a_close = builder.inner_mut().literal("</param>");
            let one_arg = seq_tool_arg(&mut builder, vec![a_key, val_p, a_close]);
            let arg_rep = builder.inner_mut().zero_or_more(one_arg);
            let args = builder.tool_args(arg_rep);
            let close = lit_tool_close(&mut builder, "</function>");
            call_parsers.push(seq_tool(&mut builder, vec![open, args, close]));
        }
        let single_call = builder.inner_mut().choice(call_parsers);
        let calls_core = if inputs.parallel_tool_calls {
            let sep_sp = builder.inner_mut().space();
            let step = builder.inner_mut().add(sep_sp, single_call);
            let tail = builder.inner_mut().zero_or_more(step);
            builder.inner_mut().add(single_call, tail)
        } else {
            single_call
        };
        let end_sp = builder.inner_mut().space();
        let calls = builder.inner_mut().add(calls_core, end_sp);
        let trig = builder.trigger_rule("tool-call", calls);
        if force_tools {
            parts.push(trig);
        } else {
            let until_tc = builder.inner_mut().until("<function name=\"");
            let c = builder.content(until_tc);
            let opt_t = builder.inner_mut().optional(trig);
            parts.push(c);
            parts.push(opt_t);
            if json_schema.is_none() {
                lazy = true;
                triggers.push(TriggerDef {
                    kind: 1,
                    value: "<function name=\"".to_owned(),
                });
            }
        }
    } else {
        parts.push(content_or_schema_parser(&mut builder, json_schema)?);
    }

    let root = if parts.len() == 1 {
        parts[0]
    } else {
        builder.inner_mut().sequence(parts)
    };
    builder.set_root(root);
    Ok((builder.build()?, triggers, lazy))
}

#[expect(
    clippy::too_many_lines,
    reason = "PEG grammar definition for upstream chat format"
)]
fn build_muse_glimmer_parser(
    mut builder: ChatPegBuilder,
    tools: Option<&JsonVal>,
    json_schema: Option<&JsonVal>,
    inputs: &ChatApplyInputs,
) -> Result<(ChatArena, Vec<TriggerDef>, bool)> {
    let tool_defs = tools.map_or_else(Vec::new, extract_tool_defs);
    let has_tools = !tool_defs.is_empty() && inputs.tool_choice != 2;
    let force_tools = has_tools && inputs.tool_choice == 1;
    let mut triggers = Vec::new();
    let mut lazy = false;

    let mut chan_alts = Vec::new();
    if inputs.reasoning_format != 0 {
        let self_hdr = builder.inner_mut().literal("to=self<|message|>");
        let r_until = builder
            .inner_mut()
            .until_one_of(vec!["<|eom|>".to_owned(), "<|eot|>".to_owned()]);
        let r_body = builder.reasoning(r_until);
        let r_end = or_lits(&mut builder, "<|eom|>", "<|eot|>");
        let seq = builder.inner_mut().sequence(vec![self_hdr, r_body, r_end]);
        chan_alts.push(builder.reasoning_block(seq));
    }

    if has_tools {
        let mut call_parsers = Vec::with_capacity(tool_defs.len());
        for (name, params) in &tool_defs {
            let p1 = builder.inner_mut().literal("to=");
            let tname_outer = builder.inner_mut().literal(name);
            let p2 = builder
                .inner_mut()
                .literal("<|message|><atem:function_calls>\n<atem:invoke name=\"");
            let tname = lit_tool_name(&mut builder, name);
            let p3 = builder.inner_mut().literal("\">\n");
            let open_raw = seq_tool_open(&mut builder, vec![p1, tname_outer, p2, tname, p3]);
            let open = builder.inner_mut().atomic(open_raw);

            let props = params.get("properties").object();
            let arg_rep = if props.is_empty() {
                let a_open = builder.inner_mut().literal("<atem:parameter name=\"");
                let a_name_raw = builder.inner_mut().until("\">");
                let a_name = builder.tool_arg_name(a_name_raw);
                let a_mid = builder.inner_mut().literal("\">");
                let a_key_seq = builder.inner_mut().sequence(vec![a_open, a_name, a_mid]);
                let a_key = builder.inner_mut().atomic(a_key_seq);
                let a_val = until_tool_arg_str(&mut builder, "</atem:parameter>");
                let a_close = builder.inner_mut().literal("</atem:parameter>\n");
                let one_arg = seq_tool_arg(&mut builder, vec![a_key, a_val, a_close]);
                builder.inner_mut().zero_or_more(one_arg)
            } else {
                let mut prop_parsers = Vec::with_capacity(props.len());
                for (prop_name, prop_schema) in props {
                    let a_open = builder.inner_mut().literal("<atem:parameter name=\"");
                    let a_name_lit = builder.inner_mut().literal(prop_name);
                    let a_name = builder.tool_arg_name(a_name_lit);
                    let a_mid = builder.inner_mut().literal("\">");
                    let a_key_seq = builder.inner_mut().sequence(vec![a_open, a_name, a_mid]);
                    let a_key = builder.inner_mut().atomic(a_key_seq);
                    let a_val = if schema_resolves_to_string(prop_schema) {
                        until_tool_arg_str(&mut builder, "</atem:parameter>")
                    } else {
                        let jp = builder.inner_mut().json()?;
                        builder.tool_arg_json_value(jp)
                    };
                    let a_close = builder.inner_mut().literal("</atem:parameter>\n");
                    prop_parsers.push(seq_tool_arg(&mut builder, vec![a_key, a_val, a_close]));
                }
                let one_arg = builder.inner_mut().choice(prop_parsers);
                builder.inner_mut().zero_or_more(one_arg)
            };
            let args = builder.tool_args(arg_rep);
            let close = lit_tool_close(&mut builder, "</atem:invoke>\n</atem:function_calls>");
            let end_or = or_lits(&mut builder, "<|eom|>", "<|eot|>");
            let turn_end = builder.inner_mut().optional(end_or);
            call_parsers.push(seq_tool(&mut builder, vec![open, args, close, turn_end]));
        }
        let call_choice = builder.inner_mut().choice(call_parsers);
        let trig = builder.trigger_rule("tool-call", call_choice);
        chan_alts.push(trig);
        if !force_tools && json_schema.is_none() {
            lazy = true;
            for (name, _) in &tool_defs {
                triggers.push(TriggerDef {
                    kind: 1,
                    value: format!("to={name}<|message|><atem:function_calls>"),
                });
            }
        }
    }

    if !force_tools {
        let user_hdr = builder.inner_mut().literal("to=user<|message|>");
        let c_body = if let Some(schema) = json_schema
            && !matches!(schema, JsonVal::Null)
        {
            builder.schema_content_parser(schema)?
        } else {
            let u = builder
                .inner_mut()
                .until_one_of(vec!["<|eom|>".to_owned(), "<|eot|>".to_owned()]);
            builder.content(u)
        };
        let end_or = or_lits(&mut builder, "<|eom|>", "<|eot|>");
        let end_tok = builder.inner_mut().optional(end_or);
        let seq = builder
            .inner_mut()
            .sequence(vec![user_hdr, c_body, end_tok]);
        chan_alts.push(seq);
    }

    let one_chan = builder.inner_mut().choice(chan_alts);
    let next_start = builder.inner_mut().literal("<|start|>assistant ");
    let next_chan = builder.inner_mut().sequence(vec![next_start, one_chan]);
    let rest_chans = builder.inner_mut().zero_or_more(next_chan);
    let lead_sp = builder.inner_mut().space();
    let multi = builder
        .inner_mut()
        .sequence(vec![lead_sp, one_chan, rest_chans]);
    let root = if force_tools {
        multi
    } else {
        let fallback = content_or_schema_parser(&mut builder, json_schema)?;
        builder.inner_mut().or(multi, fallback)
    };
    builder.set_root(root);
    Ok((builder.build()?, triggers, lazy))
}

#[expect(
    clippy::too_many_lines,
    reason = "Builds Mistral/Ministral/Devstral [THINK]/[TOOL_CALLS]/[ARGS] PEG grammar together"
)]
fn build_mistral_parser(
    mut builder: ChatPegBuilder,
    src: &str,
    tools: Option<&JsonVal>,
    json_schema: Option<&JsonVal>,
    inputs: &ChatApplyInputs,
    generation_prompt: &str,
) -> Result<(ChatArena, Vec<TriggerDef>, bool)> {
    let mut parts = Vec::new();
    if src.contains("[THINK]")
        && let Some(r) = build_reasoning_segment(
            &mut builder,
            "[THINK]",
            &["[/THINK]"],
            generation_prompt,
            inputs.reasoning_format,
            false,
        )
    {
        parts.push(r);
    }

    let tool_defs = tools.map_or_else(Vec::new, extract_tool_defs);
    let has_tools = !tool_defs.is_empty() && inputs.tool_choice != 2;
    let force_tools = has_tools && inputs.tool_choice == 1;
    let mut triggers = Vec::new();
    let mut lazy = false;

    if has_tools {
        let mut tool_section = if src.contains("[ARGS]") {
            let mut call_parsers = Vec::with_capacity(tool_defs.len());
            for (name, params) in &tool_defs {
                let schema_p = build_tool_schema_json(&mut builder, name, params)?;
                let tc_open = lit_tool_open(&mut builder, "[TOOL_CALLS]");
                let tname = lit_tool_name(&mut builder, name);
                let opt_id = if src.contains("[CALL_ID]") {
                    let cid_pfx = builder.inner_mut().literal("[CALL_ID]");
                    let cid_raw = builder.inner_mut().chars("[^\\[]", 1, -1)?;
                    let cid_body = builder.tool_id(cid_raw);
                    let args_ahead_lit = builder.inner_mut().literal("[ARGS]");
                    let args_ahead = builder.inner_mut().peek(args_ahead_lit);
                    let cid_inner = builder
                        .inner_mut()
                        .sequence(vec![cid_pfx, cid_body, args_ahead]);
                    let cid_seq = builder.inner_mut().atomic(cid_inner);
                    builder.inner_mut().optional(cid_seq)
                } else {
                    builder.inner_mut().eps()
                };
                let args_pfx = builder.inner_mut().literal("[ARGS]");
                let args_val = builder.tool_args(schema_p);
                let eps = builder.inner_mut().eps();
                let close = builder.tool_close(eps);
                call_parsers.push(seq_tool(
                    &mut builder,
                    vec![tc_open, tname, opt_id, args_pfx, args_val, close],
                ));
            }
            let single = builder.inner_mut().choice(call_parsers);
            if inputs.parallel_tool_calls {
                let rep = builder.inner_mut().zero_or_more(single);
                builder.inner_mut().add(single, rep)
            } else {
                single
            }
        } else {
            let tools_val = tools.ok_or_else(|| invalid("missing mistral tools"))?;
            builder.standard_json_tools(&JsonToolsOptions {
                section_start: "[TOOL_CALLS]",
                section_end: "",
                tools: tools_val,
                parallel_tool_calls: inputs.parallel_tool_calls,
                force_tool_calls: force_tools,
                per_call_start: "",
                per_call_end: "",
                fun_name_is_key: false,
                tools_array_wrapped: true,
                name_field: "name",
                args_field: "arguments",
                id_field: if src.contains("\"id\"") { "id" } else { "" },
                gen_id_field: "",
                parameter_order: &[],
                allow_raw_arguments: false,
            })?
        };
        if src.contains("[ARGS]") {
            tool_section = builder.trigger_rule("tool-call", tool_section);
        }
        if force_tools {
            parts.push(tool_section);
        } else {
            let until_tc = builder.inner_mut().until("[TOOL_CALLS]");
            let c = builder.content(until_tc);
            parts.push(c);
            if src.contains("[ARGS]") {
                let opt_s = builder.inner_mut().optional(tool_section);
                parts.push(opt_s);
            } else {
                parts.push(tool_section);
            }
            if json_schema.is_none() {
                lazy = true;
                triggers.push(TriggerDef {
                    kind: 1,
                    value: "[TOOL_CALLS]".to_owned(),
                });
            }
        }
    } else {
        parts.push(content_or_schema_parser(&mut builder, json_schema)?);
    }

    let root = if parts.len() == 1 {
        parts[0]
    } else {
        builder.inner_mut().sequence(parts)
    };
    builder.set_root(root);
    Ok((builder.build()?, triggers, lazy))
}

struct TaggedJsonSpec<'a> {
    section_start: &'a str,
    section_end: &'a str,
    name_prefix: &'a str,
    name_suffix: &'a str,
    call_close: &'a str,
}

#[expect(
    clippy::too_many_lines,
    reason = "Builds GLM-4.5/4.6/4.7 XML key-value tool PEG rules in one pass"
)]
fn build_glm45_parser(
    mut builder: ChatPegBuilder,
    tools: Option<&JsonVal>,
    json_schema: Option<&JsonVal>,
    inputs: &ChatApplyInputs,
    generation_prompt: &str,
) -> Result<(ChatArena, Vec<TriggerDef>, bool)> {
    let mut parts = Vec::new();
    if let Some(r) = build_reasoning_segment(
        &mut builder,
        "<think>",
        &["</think>"],
        generation_prompt,
        inputs.reasoning_format,
        false,
    ) {
        parts.push(r);
    }

    let tool_defs = tools.map_or_else(Vec::new, extract_tool_defs);
    let has_tools = !tool_defs.is_empty() && inputs.tool_choice != 2;
    let force_tools = has_tools && inputs.tool_choice == 1;
    let mut triggers = Vec::new();
    let mut lazy = false;

    if has_tools {
        let mut call_parsers = Vec::with_capacity(tool_defs.len());
        for (name, params) in &tool_defs {
            let p_open = builder.inner_mut().literal("<tool_call>");
            let tname = lit_tool_name(&mut builder, name);
            let peek_alts = vec![
                builder.inner_mut().literal("\n"),
                builder.inner_mut().literal("<arg_key>"),
                builder.inner_mut().literal("</tool_call>"),
            ];
            let peek_choice = builder.inner_mut().choice(peek_alts);
            let peek_delim = builder.inner_mut().peek(peek_choice);
            let opt_nl_open = opt_lit(&mut builder, "\n");
            let open_raw =
                seq_tool_open(&mut builder, vec![p_open, tname, peek_delim, opt_nl_open]);
            let open = builder.inner_mut().atomic(open_raw);

            let props = params.get("properties").object();
            let arg_rep = if props.is_empty() {
                let ak_open = builder.inner_mut().literal("<arg_key>");
                let ak_raw = builder.inner_mut().until("</arg_key>");
                let ak_name = builder.tool_arg_name(ak_raw);
                let ak_close = builder.inner_mut().literal("</arg_key>");
                let opt_mid = opt_lit(&mut builder, "\n");
                let val_open = builder.inner_mut().literal("<arg_value>");
                let a_key_seq = builder
                    .inner_mut()
                    .sequence(vec![ak_open, ak_name, ak_close, opt_mid, val_open]);
                let a_key = builder.inner_mut().atomic(a_key_seq);
                let val_body = until_tool_arg_str(&mut builder, "</arg_value>");
                let val_close = builder.inner_mut().literal("</arg_value>");
                let opt_trail = opt_lit(&mut builder, "\n");
                let one_arg =
                    seq_tool_arg(&mut builder, vec![a_key, val_body, val_close, opt_trail]);
                builder.inner_mut().zero_or_more(one_arg)
            } else {
                let mut prop_parsers = Vec::with_capacity(props.len());
                for (prop_name, prop_schema) in props {
                    let ak_open = builder.inner_mut().literal("<arg_key>");
                    let ak_lit = builder.inner_mut().literal(prop_name);
                    let ak_name = builder.tool_arg_name(ak_lit);
                    let ak_close = builder.inner_mut().literal("</arg_key>");
                    let opt_mid = opt_lit(&mut builder, "\n");
                    let val_open = builder.inner_mut().literal("<arg_value>");
                    let a_key_seq = builder
                        .inner_mut()
                        .sequence(vec![ak_open, ak_name, ak_close, opt_mid, val_open]);
                    let a_key = builder.inner_mut().atomic(a_key_seq);
                    let val_body = if schema_resolves_to_string(prop_schema) {
                        until_tool_arg_str(&mut builder, "</arg_value>")
                    } else {
                        let jp = builder.inner_mut().json()?;
                        builder.tool_arg_json_value(jp)
                    };
                    let val_close = builder.inner_mut().literal("</arg_value>");
                    let opt_trail = opt_lit(&mut builder, "\n");
                    prop_parsers.push(seq_tool_arg(
                        &mut builder,
                        vec![a_key, val_body, val_close, opt_trail],
                    ));
                }
                let one_arg = builder.inner_mut().choice(prop_parsers);
                builder.inner_mut().zero_or_more(one_arg)
            };
            let args = builder.tool_args(arg_rep);
            let close = lit_tool_close(&mut builder, "</tool_call>");
            call_parsers.push(seq_tool(&mut builder, vec![open, args, close]));
        }
        let single_call = builder.inner_mut().choice(call_parsers);
        let calls = if inputs.parallel_tool_calls {
            let opt_sep = opt_lit(&mut builder, "\n");
            let step = builder.inner_mut().add(opt_sep, single_call);
            let tail = builder.inner_mut().zero_or_more(step);
            builder.inner_mut().add(single_call, tail)
        } else {
            single_call
        };
        let trig = builder.trigger_rule("tool-call", calls);
        if force_tools {
            parts.push(trig);
        } else {
            let until_tc = builder.inner_mut().until("<tool_call>");
            let c = builder.content(until_tc);
            let opt_t = builder.inner_mut().optional(trig);
            parts.push(c);
            parts.push(opt_t);
            if json_schema.is_none() {
                lazy = true;
                triggers.push(TriggerDef {
                    kind: 1,
                    value: "<tool_call>".to_owned(),
                });
            }
        }
        parts.push(builder.inner_mut().end());
    } else {
        parts.push(content_or_schema_parser(&mut builder, json_schema)?);
    }

    let root = if parts.len() == 1 {
        parts[0]
    } else {
        builder.inner_mut().sequence(parts)
    };
    builder.set_root(root);
    Ok((builder.build()?, triggers, lazy))
}

#[expect(
    clippy::too_many_lines,
    reason = "Builds MiniMax-M2 XML parameter tool PEG rules in one pass"
)]
fn build_minimax_m2_parser(
    mut builder: ChatPegBuilder,
    tools: Option<&JsonVal>,
    json_schema: Option<&JsonVal>,
    inputs: &ChatApplyInputs,
    generation_prompt: &str,
) -> Result<(ChatArena, Vec<TriggerDef>, bool)> {
    let mut parts = Vec::new();
    if let Some(r) = build_reasoning_segment(
        &mut builder,
        "<think>",
        &["\n</think>\n\n", "\n</think>", "</think>"],
        generation_prompt,
        inputs.reasoning_format,
        true,
    ) {
        parts.push(r);
    }

    let tool_defs = tools.map_or_else(Vec::new, extract_tool_defs);
    let has_tools = !tool_defs.is_empty() && inputs.tool_choice != 2;
    let force_tools = has_tools && inputs.tool_choice == 1;
    let mut triggers = Vec::new();
    let mut lazy = false;

    if has_tools {
        let mut call_parsers = Vec::with_capacity(tool_defs.len());
        for (name, params) in &tool_defs {
            let p1 = builder.inner_mut().literal("<invoke name=\"");
            let tname = lit_tool_name(&mut builder, name);
            let p2 = builder.inner_mut().literal("\">");
            let sp0 = builder.inner_mut().space();
            let open_raw = seq_tool_open(&mut builder, vec![p1, tname, p2, sp0]);
            let open = builder.inner_mut().atomic(open_raw);

            let props = params.get("properties").object();
            let arg_rep = if props.is_empty() {
                let p_open = builder.inner_mut().literal("<parameter name=\"");
                let p_raw = builder.inner_mut().until("\">");
                let p_name = builder.tool_arg_name(p_raw);
                let p_mid = builder.inner_mut().literal("\">");
                let a_key_seq = builder.inner_mut().sequence(vec![p_open, p_name, p_mid]);
                let a_key = builder.inner_mut().atomic(a_key_seq);
                let a_val = until_tool_arg_str(&mut builder, "</parameter>");
                let p_close = builder.inner_mut().literal("</parameter>");
                let sp_a = builder.inner_mut().space();
                let one = seq_tool_arg(&mut builder, vec![a_key, a_val, p_close, sp_a]);
                builder.inner_mut().zero_or_more(one)
            } else {
                let mut prop_parsers = Vec::with_capacity(props.len());
                for (prop_name, prop_schema) in props {
                    let p_open = builder.inner_mut().literal("<parameter name=\"");
                    let p_lit = builder.inner_mut().literal(prop_name);
                    let p_name = builder.tool_arg_name(p_lit);
                    let p_mid = builder.inner_mut().literal("\">");
                    let a_key_seq = builder.inner_mut().sequence(vec![p_open, p_name, p_mid]);
                    let a_key = builder.inner_mut().atomic(a_key_seq);
                    let a_val = if schema_resolves_to_string(prop_schema) {
                        until_tool_arg_str(&mut builder, "</parameter>")
                    } else {
                        let jp = builder.inner_mut().json()?;
                        builder.tool_arg_json_value(jp)
                    };
                    let p_close = builder.inner_mut().literal("</parameter>");
                    let sp_a = builder.inner_mut().space();
                    prop_parsers.push(seq_tool_arg(
                        &mut builder,
                        vec![a_key, a_val, p_close, sp_a],
                    ));
                }
                let one = builder.inner_mut().choice(prop_parsers);
                builder.inner_mut().zero_or_more(one)
            };
            let args = builder.tool_args(arg_rep);
            let close = lit_tool_close(&mut builder, "</invoke>");
            let sp_after = builder.inner_mut().space();
            call_parsers.push(seq_tool(&mut builder, vec![open, args, close, sp_after]));
        }
        let single = builder.inner_mut().choice(call_parsers);
        let calls = if inputs.parallel_tool_calls {
            let eps = builder.inner_mut().eps();
            let step = builder.inner_mut().space_add(eps, single);
            let tail = builder.inner_mut().zero_or_more(step);
            builder.inner_mut().add(single, tail)
        } else {
            single
        };
        let sec_open = builder.inner_mut().literal("<minimax:tool_call>");
        let sec_sp = builder.inner_mut().space();
        let sec_close = builder.inner_mut().literal("</minimax:tool_call>");
        let end_sp = builder.inner_mut().space();
        let section = builder
            .inner_mut()
            .sequence(vec![sec_open, sec_sp, calls, sec_close, end_sp]);
        let trig = builder.trigger_rule("tool-call", section);
        if force_tools {
            parts.push(trig);
        } else {
            let until_tc = builder.inner_mut().until("<minimax:tool_call>");
            let c = builder.content(until_tc);
            let opt_t = builder.inner_mut().optional(trig);
            parts.push(c);
            parts.push(opt_t);
            if json_schema.is_none() {
                lazy = true;
                triggers.push(TriggerDef {
                    kind: 1,
                    value: "<minimax:tool_call>".to_owned(),
                });
            }
        }
    } else {
        parts.push(content_or_schema_parser(&mut builder, json_schema)?);
    }

    let root = if parts.len() == 1 {
        parts[0]
    } else {
        builder.inner_mut().sequence(parts)
    };
    builder.set_root(root);
    Ok((builder.build()?, triggers, lazy))
}

#[expect(
    clippy::too_many_lines,
    reason = "Builds Nemotron-v2 JSON tool-call grammar including trailing whitespace"
)]
fn build_nemotron_v2_parser(
    mut builder: ChatPegBuilder,
    tools: Option<&JsonVal>,
    json_schema: Option<&JsonVal>,
    inputs: &ChatApplyInputs,
    generation_prompt: &str,
) -> Result<(ChatArena, Vec<TriggerDef>, bool)> {
    let mut parts = Vec::new();
    if let Some(r) = build_reasoning_segment(
        &mut builder,
        "<think>",
        &["\n</think>", "</think>"],
        generation_prompt,
        inputs.reasoning_format,
        true,
    ) {
        parts.push(r);
    }

    let tool_defs = tools.map_or_else(Vec::new, extract_tool_defs);
    let has_tools = !tool_defs.is_empty() && inputs.tool_choice != 2;
    let force_tools = has_tools && inputs.tool_choice == 1;
    let mut triggers = Vec::new();
    let mut lazy = false;

    if has_tools {
        let mut call_parsers = Vec::with_capacity(tool_defs.len());
        for (name, params) in &tool_defs {
            let lbrace = builder.inner_mut().literal("{");
            let open = builder.tool_open(lbrace);
            let rbrace = builder.inner_mut().literal("}");
            let close = builder.tool_close(rbrace);

            let k_name = builder.inner_mut().literal("\"name\"");
            let c1 = builder.inner_mut().literal(":");
            let q1 = builder.inner_mut().literal("\"");
            let tname = lit_tool_name(&mut builder, name);
            let q2 = builder.inner_mut().literal("\"");
            let v_name = builder.inner_mut().sequence(vec![q1, tname, q2]);
            let nc1 = builder.inner_mut().space_add(k_name, c1);
            let name_kv = builder.inner_mut().space_add(nc1, v_name);

            let schema_p = build_tool_schema_json(&mut builder, name, params)?;
            let args_val = builder.tool_args(schema_p);
            let k_args = builder.inner_mut().literal("\"arguments\"");
            let c2 = builder.inner_mut().literal(":");
            let ac2 = builder.inner_mut().space_add(k_args, c2);
            let args_kv = builder.inner_mut().space_add(ac2, args_val);

            let comma = builder.inner_mut().literal(",");
            let sp_b = builder.inner_mut().space();
            let with_comma = builder.inner_mut().add(sp_b, comma);
            let sp_a = builder.inner_mut().space();
            let comma_sep = builder.inner_mut().add(with_comma, sp_a);

            let pair_na = builder
                .inner_mut()
                .sequence(vec![name_kv, comma_sep, args_kv]);
            let pair_an = builder
                .inner_mut()
                .sequence(vec![args_kv, comma_sep, name_kv]);
            let mut body_alts = vec![pair_na, pair_an];
            if params.get("required").array().is_empty() {
                body_alts.push(name_kv);
            }
            let body = builder.inner_mut().choice(body_alts);
            let left = builder.inner_mut().space_add(open, body);
            let full = builder.inner_mut().space_add(left, close);
            call_parsers.push(builder.tool(full));
        }
        let single_call = builder.inner_mut().choice(call_parsers);
        let calls = if inputs.parallel_tool_calls {
            let comma = builder.inner_mut().literal(",");
            let next_call = builder.inner_mut().space_add(comma, single_call);
            let eps = builder.inner_mut().eps();
            let step = builder.inner_mut().space_add(eps, next_call);
            let tail = builder.inner_mut().zero_or_more(step);
            builder.inner_mut().add(single_call, tail)
        } else {
            single_call
        };
        let lbrack = builder.inner_mut().literal("[");
        let rbrack = builder.inner_mut().literal("]");
        let arr_left = builder.inner_mut().space_add(lbrack, calls);
        let arr = builder.inner_mut().space_add(arr_left, rbrack);
        let sec_open = builder.inner_mut().literal("<TOOLCALL>");
        let sec_close = builder.inner_mut().literal("</TOOLCALL>");
        let sec_left = builder.inner_mut().space_add(sec_open, arr);
        let sec_body = builder.inner_mut().space_add(sec_left, sec_close);
        let tail_sp = builder.inner_mut().space();
        let section = builder.inner_mut().add(sec_body, tail_sp);
        let trig = builder.trigger_rule("tool-call", section);
        if force_tools {
            parts.push(trig);
        } else {
            let until_tc = builder.inner_mut().until("<TOOLCALL>");
            let opt_trig = builder.inner_mut().optional(trig);
            parts.push(builder.content(until_tc));
            parts.push(opt_trig);
            if json_schema.is_none() {
                lazy = true;
                triggers.push(TriggerDef {
                    kind: 1,
                    value: "<TOOLCALL>".to_owned(),
                });
            }
        }
    } else {
        parts.push(content_or_schema_parser(&mut builder, json_schema)?);
    }

    let root = if parts.len() == 1 {
        parts[0]
    } else {
        builder.inner_mut().sequence(parts)
    };
    builder.set_root(root);
    Ok((builder.build()?, triggers, lazy))
}

fn build_granite_parser(
    mut builder: ChatPegBuilder,
    tools: Option<&JsonVal>,
    json_schema: Option<&JsonVal>,
    inputs: &ChatApplyInputs,
    generation_prompt: &str,
) -> Result<(ChatArena, Vec<TriggerDef>, bool)> {
    let mut parts = Vec::new();
    if let Some(r) = build_reasoning_segment(
        &mut builder,
        "<think>",
        &["</think>"],
        generation_prompt,
        inputs.reasoning_format,
        false,
    ) {
        parts.push(r);
    }
    let tool_defs = tools.map_or_else(Vec::new, extract_tool_defs);
    let has_tools = !tool_defs.is_empty() && inputs.tool_choice != 2;
    let force_tools = has_tools && inputs.tool_choice == 1;
    let mut triggers = Vec::new();
    let mut lazy = false;
    if has_tools {
        let tv = tools.ok_or_else(|| invalid("missing tools"))?;
        let tp = builder.standard_json_tools(&JsonToolsOptions {
            section_start: "<|tool_call|>",
            section_end: "",
            tools: tv,
            parallel_tool_calls: inputs.parallel_tool_calls,
            force_tool_calls: force_tools,
            per_call_start: "",
            per_call_end: "",
            fun_name_is_key: false,
            tools_array_wrapped: true,
            name_field: "name",
            args_field: "arguments",
            id_field: "",
            gen_id_field: "",
            parameter_order: &[],
            allow_raw_arguments: false,
        })?;
        if force_tools {
            parts.push(tp);
        } else {
            let u = builder.inner_mut().until("<|tool_call|>");
            parts.push(builder.content(u));
            parts.push(tp);
            if json_schema.is_none() {
                lazy = true;
                triggers.push(TriggerDef {
                    kind: 1,
                    value: "<|tool_call|>".to_owned(),
                });
            }
        }
    } else {
        parts.push(content_or_schema_parser(&mut builder, json_schema)?);
    }
    let root = if parts.len() == 1 {
        parts[0]
    } else {
        builder.inner_mut().sequence(parts)
    };
    builder.set_root(root);
    Ok((builder.build()?, triggers, lazy))
}

fn build_apriel15_parser(
    mut builder: ChatPegBuilder,
    tools: Option<&JsonVal>,
    json_schema: Option<&JsonVal>,
    inputs: &ChatApplyInputs,
    generation_prompt: &str,
) -> Result<(ChatArena, Vec<TriggerDef>, bool)> {
    let tool_defs = tools.map_or_else(Vec::new, extract_tool_defs);
    let has_tools = !tool_defs.is_empty() && inputs.tool_choice != 2;
    let force_tools = has_tools && inputs.tool_choice == 1;
    let mut triggers = Vec::new();
    let mut lazy = false;

    let body_p = if has_tools {
        let tv = tools.ok_or_else(|| invalid("missing tools"))?;
        let tp = builder.standard_json_tools(&JsonToolsOptions {
            section_start: "<tool_calls>",
            section_end: "</tool_calls>",
            tools: tv,
            parallel_tool_calls: inputs.parallel_tool_calls,
            force_tool_calls: force_tools,
            per_call_start: "",
            per_call_end: "",
            fun_name_is_key: false,
            tools_array_wrapped: true,
            name_field: "name",
            args_field: "arguments",
            id_field: "",
            gen_id_field: "",
            parameter_order: &[],
            allow_raw_arguments: false,
        })?;
        if force_tools {
            tp
        } else {
            let u = builder.inner_mut().until("<tool_calls>");
            let c = builder.content(u);
            if json_schema.is_none() {
                lazy = true;
                triggers.push(TriggerDef {
                    kind: 1,
                    value: "<tool_calls>".to_owned(),
                });
            }
            builder.inner_mut().sequence(vec![c, tp])
        }
    } else {
        content_or_schema_parser(&mut builder, json_schema)?
    };

    let already_closed = generation_prompt
        .trim_end()
        .ends_with("[BEGIN FINAL RESPONSE]");
    let think_open = !already_closed
        && (generation_prompt.ends_with("Here are my reasoning steps:\n")
            || inputs.continue_final_message == 2);
    let root = if think_open {
        let until_r = builder.inner_mut().until_one_of(vec![
            "\n[BEGIN FINAL RESPONSE]".to_owned(),
            "[BEGIN FINAL RESPONSE]".to_owned(),
        ]);
        let sep_lit = or_lits(
            &mut builder,
            "\n[BEGIN FINAL RESPONSE]",
            "[BEGIN FINAL RESPONSE]",
        );
        let sp = builder.inner_mut().space();
        if inputs.reasoning_format == 0 {
            let r_seq = builder.inner_mut().sequence(vec![until_r, sep_lit]);
            let r_c = builder.content(r_seq);
            let tail = builder.inner_mut().sequence(vec![sp, body_p]);
            let opt_tail = builder.inner_mut().optional(tail);
            builder.inner_mut().sequence(vec![r_c, opt_tail])
        } else {
            let r_body = builder.reasoning(until_r);
            let r_blk = builder.reasoning_block(r_body);
            let closed = builder.inner_mut().sequence(vec![sep_lit, sp, body_p]);
            let after = if !force_tools && json_schema.is_none() {
                builder.inner_mut().optional(closed)
            } else {
                closed
            };
            builder.inner_mut().sequence(vec![r_blk, after])
        }
    } else {
        body_p
    };
    builder.set_root(root);
    Ok((builder.build()?, triggers, lazy))
}

#[expect(
    clippy::too_many_lines,
    reason = "PEG grammar definition for upstream chat format"
)]
fn build_simple_tagged_json_tools(
    mut builder: ChatPegBuilder,
    spec: &TaggedJsonSpec<'_>,
    tools: &JsonVal,
    inputs: &ChatApplyInputs,
    json_schema: Option<&JsonVal>,
    generation_prompt: &str,
) -> Result<(ChatArena, Vec<TriggerDef>, bool)> {
    let force = inputs.tool_choice == 1;
    let tool_defs = extract_tool_defs(tools);
    let open_first_without_pfx = spec.name_prefix == ">>>" && generation_prompt.ends_with(">>>");
    let mut call_parsers = Vec::with_capacity(tool_defs.len());
    let mut first_call_parsers = Vec::with_capacity(tool_defs.len());
    for (name, params) in &tool_defs {
        let schema_p = build_tool_schema_json(&mut builder, name, params)?;
        let open = lit_tool_open(&mut builder, spec.name_prefix);
        let tname = lit_tool_name(&mut builder, name);
        let mid = builder.inner_mut().literal(spec.name_suffix);
        let sp = builder.inner_mut().space();
        let args = builder.tool_args(schema_p);
        let close = lit_tool_close(&mut builder, spec.call_close);
        call_parsers.push(seq_tool(
            &mut builder,
            vec![open, tname, mid, sp, args, close],
        ));

        if open_first_without_pfx {
            let schema_p_first = build_tool_schema_json(&mut builder, name, params)?;
            let opt_pfx = opt_lit(&mut builder, ">>>");
            let first_open = builder.tool_open(opt_pfx);
            let tname_f = lit_tool_name(&mut builder, name);
            let mid_f = builder.inner_mut().literal(spec.name_suffix);
            let sp_f = builder.inner_mut().space();
            let args_f = builder.tool_args(schema_p_first);
            let close_f = lit_tool_close(&mut builder, spec.call_close);
            first_call_parsers.push(seq_tool(
                &mut builder,
                vec![first_open, tname_f, mid_f, sp_f, args_f, close_f],
            ));
        }
    }
    let single = builder.inner_mut().choice(call_parsers);
    let first_single = if open_first_without_pfx {
        builder.inner_mut().choice(first_call_parsers)
    } else {
        single
    };
    let calls = if inputs.parallel_tool_calls {
        let eps = builder.inner_mut().eps();
        let sep_call = builder.inner_mut().space_add(eps, single);
        let tail = builder.inner_mut().zero_or_more(sep_call);
        builder.inner_mut().add(first_single, tail)
    } else {
        first_single
    };
    let section = if spec.section_start.is_empty() && spec.section_end.is_empty() {
        calls
    } else {
        let so = builder.inner_mut().literal(spec.section_start);
        let sc = builder.inner_mut().literal(spec.section_end);
        builder.inner_mut().sequence(vec![so, calls, sc])
    };
    let trig = builder.trigger_rule("tool-call", section);
    let trigger_lit = if spec.section_start.is_empty() {
        spec.name_prefix
    } else {
        spec.section_start
    };
    let mut triggers = Vec::new();
    let mut lazy = false;

    let root = if force {
        if !generation_prompt.is_empty() && generation_prompt.ends_with(trigger_lit) {
            let opt_trig = builder.inner_mut().optional(trig);
            builder.inner_mut().or(trig, opt_trig)
        } else {
            trig
        }
    } else if open_first_without_pfx {
        let mut delims = vec![">>>".to_owned()];
        for (name, _) in &tool_defs {
            delims.push(format!("{name}\n"));
        }
        if json_schema.is_none() {
            lazy = true;
            for d in &delims {
                triggers.push(TriggerDef {
                    kind: 1,
                    value: d.clone(),
                });
            }
        }
        let all_opt = opt_lit(&mut builder, "all\n");
        let until_tc = builder.inner_mut().until_one_of(delims);
        let c = builder.content(until_tc);
        let opt_tc = builder.inner_mut().optional(trig);
        builder.inner_mut().sequence(vec![all_opt, c, opt_tc])
    } else {
        let until_tc = builder.inner_mut().until(trigger_lit);
        let c = builder.content(until_tc);
        let opt_tc = builder.inner_mut().optional(trig);
        if json_schema.is_none() {
            lazy = true;
            triggers.push(TriggerDef {
                kind: 1,
                value: trigger_lit.to_owned(),
            });
        }
        builder.inner_mut().sequence(vec![c, opt_tc])
    };
    builder.set_root(root);
    Ok((builder.build()?, triggers, lazy))
}

struct JsonOnlySpec<'a> {
    section_start: &'a str,
    section_end: &'a str,
    name_field: &'a str,
    args_field: &'a str,
    array_wrapped: bool,
}

fn build_json_only_tools_parser(
    mut builder: ChatPegBuilder,
    spec: &JsonOnlySpec<'_>,
    tools: &JsonVal,
    inputs: &ChatApplyInputs,
    json_schema: Option<&JsonVal>,
) -> Result<(ChatArena, Vec<TriggerDef>, bool)> {
    let force = inputs.tool_choice == 1;
    let bare_json = spec.section_start.is_empty() && !force;
    let tools_p = builder.standard_json_tools(&JsonToolsOptions {
        section_start: spec.section_start,
        section_end: spec.section_end,
        tools,
        parallel_tool_calls: inputs.parallel_tool_calls,
        force_tool_calls: force || bare_json,
        per_call_start: "",
        per_call_end: "",
        fun_name_is_key: false,
        tools_array_wrapped: spec.array_wrapped,
        name_field: spec.name_field,
        args_field: spec.args_field,
        id_field: "",
        gen_id_field: "",
        parameter_order: &[],
        allow_raw_arguments: false,
    })?;

    let mut triggers = Vec::new();
    let mut lazy = false;
    let root = if force {
        tools_p
    } else {
        if json_schema.is_none() {
            lazy = true;
            if spec.section_start.is_empty() {
                triggers.push(TriggerDef {
                    kind: 2,
                    value: if spec.array_wrapped {
                        r"(\[)".to_owned()
                    } else {
                        r"(\{)".to_owned()
                    },
                });
            } else {
                triggers.push(TriggerDef {
                    kind: 1,
                    value: spec.section_start.to_owned(),
                });
            }
        }
        if spec.section_start.is_empty() {
            let rest = content_or_schema_parser(&mut builder, json_schema)?;
            builder.inner_mut().or(tools_p, rest)
        } else if spec.section_start == "function call<|role_sep|>\n" {
            let u = builder.inner_mut().until_one_of(vec![
                "<|message_sep|>".to_owned(),
                spec.section_start.to_owned(),
            ]);
            let c = builder.content(u);
            let msep = builder.inner_mut().literal("<|message_sep|>");
            let sp = builder.inner_mut().space();
            let msep_sp = builder.inner_mut().sequence(vec![msep, sp]);
            let opt_msep = builder.inner_mut().optional(msep_sp);
            builder.inner_mut().sequence(vec![c, opt_msep, tools_p])
        } else {
            let u = builder.inner_mut().until(spec.section_start);
            let c = builder.content(u);
            builder.inner_mut().sequence(vec![c, tools_p])
        }
    };
    builder.set_root(root);
    Ok((builder.build()?, triggers, lazy))
}

fn ensure_grammar_covers_gen_prompt(
    grammar: String,
    generation_prompt: &str,
    lazy: bool,
) -> String {
    if lazy || generation_prompt.is_empty() || grammar.is_empty() {
        return grammar;
    }
    let Ok(quoted) = serde_json::to_string(generation_prompt) else {
        return grammar;
    };
    let mut lines: Vec<String> = grammar.lines().map(str::to_owned).collect();
    if let Some(root_line) = lines.iter_mut().find(|l| l.starts_with("root ::= ")) {
        let body = root_line.trim_start_matches("root ::= ").to_owned();
        *root_line = format!("root ::= ({quoted})? ({body})");
        let mut joined = lines.join("\n");
        if grammar.ends_with('\n') {
            joined.push('\n');
        }
        return joined;
    }
    grammar
}

struct ChatTraceState {
    template_src: String,
    tmpl: ChatTemplate,
    auto: AutoParser,
    auto_analyzed: bool,
    applied: Option<AppliedChatParams>,
    cached_arena: Option<ChatArena>,
    stored: BTreeMap<i64, ChatMessage>,
}

impl ChatTraceState {
    fn new(raw_template: &str, now_secs: i64) -> Self {
        let template_src = if raw_template.is_empty() {
            CHATML_JINJA_TEMPLATE.to_owned()
        } else {
            raw_template.to_owned()
        };
        let tmpl = ChatTemplate::new_with_now(&template_src, "", "", now_secs);
        Self {
            template_src,
            tmpl,
            auto: AutoParser::default(),
            auto_analyzed: false,
            applied: None,
            cached_arena: None,
            stored: BTreeMap::new(),
        }
    }

    fn ensure_auto(&mut self) -> &AutoAnalysis {
        if !self.auto_analyzed {
            self.auto.analyze_template(&self.tmpl);
            self.auto_analyzed = true;
        }
        self.auto.analysis()
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Coordinates prompt rendering, generation_prompt inference, and format-specific PEG/grammar assembly."
    )]
    fn apply(&mut self, inputs: &ChatApplyInputs) -> Result<AppliedChatParams> {
        self.cached_arena = None;

        if !inputs.use_jinja {
            let legacy_msgs: Vec<LegacyChatMessage<'_>> = inputs
                .messages
                .iter()
                .map(|m| LegacyChatMessage {
                    role: &m.role,
                    content: &m.content,
                })
                .collect();
            let prompt = legacy_apply(
                &self.template_src,
                &legacy_msgs,
                inputs.add_generation_prompt,
            )
            .ok_or_else(|| invalid("legacy chat template apply failed"))?;

            let mut builder = ChatPegBuilder::new();
            let rest = builder.inner_mut().rest();
            let content_p = builder.content(rest);
            builder.set_root(content_p);
            let arena = builder.build()?;
            let parser_json = arena.to_json_string()?;
            let grammar = if !inputs.grammar.is_empty() {
                inputs.grammar.clone()
            } else if !inputs.json_schema.is_empty() {
                let s = json_val_from_str(&inputs.json_schema)?;
                let mut sb = ChatPegBuilder::new();
                let jp = sb.inner_mut().json()?;
                let sp = sb.schema(jp, "response".to_owned(), s, false);
                sb.set_root(sp);
                sb.build()?.build_grammar(false, &[])?
            } else {
                String::new()
            };
            let res = AppliedChatParams {
                format: ChatFormat::ContentOnly.as_i32(),
                prompt,
                grammar,
                grammar_lazy: false,
                generation_prompt: String::new(),
                supports_thinking: false,
                thinking_start_tag: String::new(),
                thinking_end_tags: Vec::new(),
                grammar_triggers: Vec::new(),
                parser: parser_json,
                reasoning_format: inputs.reasoning_format,
                format_kind: ChatFormat::ContentOnly,
                continue_mode: inputs.continue_final_message,
                prefill_reasoning: String::new(),
                prefill_content: String::new(),
                has_tools_active: false,
                rf0_think_prefix: String::new(),
                apriel_think_prefix: String::new(),
                tool_param_schemas: BTreeMap::new(),
            };
            self.applied = Some(res.clone());
            return Ok(res);
        }

        let now_secs = inputs.now_milliseconds.map(|ms| ms / 1000);
        let initial_kind = detect_special_format(&self.template_src, None);
        let mut mode = adapt_mode_for_src(&self.template_src, initial_kind);
        mode.strip_trailing_newlines &= inputs.continue_final_message == 0;
        let requires_nonnull = if matches!(
            initial_kind,
            ChatFormat::PegSimple
                | ChatFormat::PegNative
                | ChatFormat::PegConstructed
                | ChatFormat::DeepseekR1
        ) {
            self.ensure_auto().content.requires_nonnull_content
        } else {
            true
        };
        let force_object_args = matches!(
            initial_kind,
            ChatFormat::FireFunctionV2
                | ChatFormat::Glm45
                | ChatFormat::MinimaxM2
                | ChatFormat::MinimaxM3
        ) || self.template_src.contains(".arguments|tojson")
            || self.template_src.contains(".arguments | tojson")
            || self.template_src.contains("['arguments']|tojson")
            || self.template_src.contains("[\"arguments\"]|tojson")
            || self.template_src.contains(".arguments.items()")
            || self.template_src.contains("_args.items()");

        let format_kind = if matches!(
            initial_kind,
            ChatFormat::PegSimple
                | ChatFormat::PegNative
                | ChatFormat::PegConstructed
                | ChatFormat::DeepseekR1
        ) {
            let _ = self.ensure_auto();
            detect_special_format(&self.template_src, Some(self.auto.analysis()))
        } else {
            initial_kind
        };

        let mut effective_messages = inputs.messages.clone();
        let supports_developer = self.template_src.contains("'developer'")
            || self.template_src.contains("\"developer\"");
        if !supports_developer {
            for msg in &mut effective_messages {
                if msg.role == "developer" {
                    msg.role = "system".to_owned();
                }
            }
        }
        if initial_kind == ChatFormat::DeepseekV4 {
            reorder_tool_results_for_assistant(&mut effective_messages);
        }

        let add_gen = if inputs.continue_final_message != 0 {
            false
        } else {
            inputs.add_generation_prompt
        };

        let msgs_jval = messages_to_jval_for_template(&effective_messages, mode, requires_nonnull);
        let tools_jval = tools_to_jval(&inputs.tools)?;
        let extra = build_effective_extra(&self.template_src, initial_kind, inputs)?;

        let template_params = TemplateParams {
            messages: Some(msgs_jval),
            tools: tools_jval,
            add_generation_prompt: add_gen,
            enable_thinking: inputs.enable_thinking,
            extra_context: extra.as_ref().map(JVal::deep_clone_fresh),
        };
        let mut prompt = self
            .tmpl
            .apply_with_now_object_args(&template_params, now_secs, force_object_args)
            .map_err(invalid)?;
        if initial_kind == ChatFormat::Gemma4 && add_gen && prompt.ends_with("<turn|>\n") {
            prompt.push_str("<|turn>model\n");
            if !inputs.enable_thinking {
                prompt.push_str("<|channel>thought\n<channel|>");
            }
        }

        let generation_prompt = apply_continuation_and_gen_prompt(
            &self.tmpl,
            inputs,
            &effective_messages,
            format_kind,
            mode,
            requires_nonnull,
            now_secs,
            force_object_args,
            extra.as_ref(),
            &mut prompt,
        );

        let (prefill_reasoning, prefill_content, parser_gen_prompt) = if inputs
            .continue_final_message
            != 0
            && let Some(last) = effective_messages.last().filter(|m| m.role == "assistant")
        {
            let target_reasoning = inputs.continue_final_message == 2
                || (inputs.continue_final_message == 1
                    && last.content.is_empty()
                    && !last.reasoning_content.is_empty());
            let (open_tag, close_tag) = match format_kind {
                ChatFormat::GptOss => (
                    "<|channel|>analysis<|message|>".to_owned(),
                    "<|end|>".to_owned(),
                ),
                ChatFormat::Gemma4 => ("<|channel>thought\n".to_owned(), "<channel|>".to_owned()),
                ChatFormat::KimiK3 => (
                    "<|open|>think<|sep|>".to_owned(),
                    "<|close|>think<|sep|>".to_owned(),
                ),
                ChatFormat::MinimaxM3 => ("<mm:think>".to_owned(), "</mm:think>".to_owned()),
                ChatFormat::Apriel15 => (
                    "Here are my reasoning steps:\n".to_owned(),
                    "[BEGIN FINAL RESPONSE]".to_owned(),
                ),
                ChatFormat::MistralNemo | ChatFormat::Magistral => {
                    ("[THINK]".to_owned(), "[/THINK]".to_owned())
                }
                ChatFormat::CommandR7b => (
                    "<|START_THINKING|>".to_owned(),
                    "<|END_THINKING|>".to_owned(),
                ),
                ChatFormat::PegSimple
                | ChatFormat::PegNative
                | ChatFormat::PegConstructed
                | ChatFormat::DeepseekR1 => {
                    let a = self.auto.analysis();
                    let s = if a.reasoning.start.trim().is_empty() {
                        "<think>".to_owned()
                    } else {
                        a.reasoning.start.trim().to_owned()
                    };
                    let e = if a.reasoning.end.trim().is_empty() {
                        "</think>".to_owned()
                    } else {
                        a.reasoning.end.trim().to_owned()
                    };
                    (s, e)
                }
                _ => ("<think>".to_owned(), "</think>".to_owned()),
            };
            let eff_gp = if target_reasoning {
                open_tag
            } else {
                close_tag
            };
            (last.reasoning_content.clone(), last.content.clone(), eff_gp)
        } else {
            (String::new(), String::new(), generation_prompt.clone())
        };

        let tools_json = tools_to_json_val(&inputs.tools)?;
        let json_schema = if inputs.json_schema.trim().is_empty() {
            None
        } else {
            Some(json_val_from_str(&inputs.json_schema)?)
        };

        let mut builder = ChatPegBuilder::new();
        let mut supports_thinking = self.template_src.contains("<think>")
            || self.template_src.contains("thought")
            || self.template_src.contains("reasoning_content")
            || self.template_src.contains("START_THINKING")
            || self.template_src.contains("[THINK]");
        let mut thinking_start_tag = if supports_thinking {
            "<think>".to_owned()
        } else {
            String::new()
        };
        let mut thinking_end_tags: Vec<String> = if supports_thinking {
            vec!["</think>".to_owned()]
        } else {
            Vec::new()
        };

        let (arena, grammar_triggers, grammar_lazy) = match format_kind {
            ChatFormat::GptOss => {
                supports_thinking = true;
                thinking_start_tag = "<|channel|>analysis<|message|>".to_owned();
                thinking_end_tags = vec!["<|end|>".to_owned()];
                build_gpt_oss_parser(
                    builder,
                    tools_json.as_ref(),
                    json_schema.as_ref(),
                    inputs,
                    &parser_gen_prompt,
                )?
            }
            ChatFormat::Gemma4 => {
                supports_thinking = true;
                thinking_start_tag = "<|channel>thought\n".to_owned();
                thinking_end_tags = vec!["<channel|>".to_owned()];
                build_gemma4_parser(
                    builder,
                    tools_json.as_ref(),
                    json_schema.as_ref(),
                    inputs,
                    &parser_gen_prompt,
                )?
            }
            ChatFormat::DeepseekV4 | ChatFormat::DeepseekV32 => {
                supports_thinking = true;
                thinking_start_tag = "<think>".to_owned();
                thinking_end_tags = vec!["</think>".to_owned()];
                build_dsml_parser(
                    builder,
                    format_kind == ChatFormat::DeepseekV4,
                    tools_json.as_ref(),
                    json_schema.as_ref(),
                    inputs,
                    &parser_gen_prompt,
                )?
            }
            ChatFormat::DeepseekV31 => {
                supports_thinking = true;
                thinking_start_tag = "<think>".to_owned();
                thinking_end_tags = vec!["</think>".to_owned()];
                build_deepseek_v31_r1_parser(
                    builder,
                    &self.template_src,
                    tools_json.as_ref(),
                    json_schema.as_ref(),
                    inputs,
                    &parser_gen_prompt,
                )?
            }
            ChatFormat::Glm45 => {
                supports_thinking = true;
                thinking_start_tag = "<think>".to_owned();
                thinking_end_tags = vec!["</think>".to_owned()];
                build_glm45_parser(
                    builder,
                    tools_json.as_ref(),
                    json_schema.as_ref(),
                    inputs,
                    &parser_gen_prompt,
                )?
            }
            ChatFormat::MinimaxM2 => {
                supports_thinking = true;
                thinking_start_tag = "<think>".to_owned();
                thinking_end_tags = vec!["</think>".to_owned()];
                build_minimax_m2_parser(
                    builder,
                    tools_json.as_ref(),
                    json_schema.as_ref(),
                    inputs,
                    &parser_gen_prompt,
                )?
            }
            ChatFormat::NemotronV2 => {
                supports_thinking = true;
                thinking_start_tag = "<think>".to_owned();
                thinking_end_tags = vec!["</think>".to_owned()];
                build_nemotron_v2_parser(
                    builder,
                    tools_json.as_ref(),
                    json_schema.as_ref(),
                    inputs,
                    &parser_gen_prompt,
                )?
            }
            ChatFormat::Granite => {
                supports_thinking = true;
                thinking_start_tag = "<think>".to_owned();
                thinking_end_tags = vec!["</think>".to_owned()];
                build_granite_parser(
                    builder,
                    tools_json.as_ref(),
                    json_schema.as_ref(),
                    inputs,
                    &parser_gen_prompt,
                )?
            }
            ChatFormat::Apriel15 => {
                supports_thinking = true;
                thinking_start_tag = "Here are my reasoning steps:\n".to_owned();
                thinking_end_tags = vec!["[BEGIN FINAL RESPONSE]".to_owned()];
                build_apriel15_parser(
                    builder,
                    tools_json.as_ref(),
                    json_schema.as_ref(),
                    inputs,
                    &parser_gen_prompt,
                )?
            }
            ChatFormat::KimiK2 => {
                supports_thinking = self.template_src.contains("<think>");
                build_kimi_k2_parser(
                    builder,
                    tools_json.as_ref(),
                    json_schema.as_ref(),
                    inputs,
                    &parser_gen_prompt,
                )?
            }
            ChatFormat::KimiK3 => {
                supports_thinking = true;
                thinking_start_tag = "<|open|>think<|sep|>".to_owned();
                thinking_end_tags = vec!["<|close|>think<|sep|>".to_owned()];
                build_kimi_k3_parser(
                    builder,
                    tools_json.as_ref(),
                    json_schema.as_ref(),
                    inputs,
                    &parser_gen_prompt,
                )?
            }
            ChatFormat::Lfm2 => {
                supports_thinking = self.template_src.contains("<think>");
                build_lfm2_parser(
                    builder,
                    tools_json.as_ref(),
                    json_schema.as_ref(),
                    inputs,
                    &parser_gen_prompt,
                )?
            }
            ChatFormat::MinimaxM3 => {
                supports_thinking = true;
                thinking_start_tag = "<mm:think>".to_owned();
                thinking_end_tags = vec!["</mm:think>".to_owned()];
                build_minimax_m3_parser(
                    builder,
                    tools_json.as_ref(),
                    json_schema.as_ref(),
                    inputs,
                    &parser_gen_prompt,
                )?
            }
            ChatFormat::MiniCpm5 => {
                supports_thinking = true;
                thinking_start_tag = "<think>".to_owned();
                thinking_end_tags = vec!["</think>".to_owned()];
                build_minicpm5_parser(
                    builder,
                    tools_json.as_ref(),
                    json_schema.as_ref(),
                    inputs,
                    &parser_gen_prompt,
                )?
            }
            ChatFormat::MuseGlimmer => {
                supports_thinking = true;
                thinking_start_tag = "to=self<|message|>".to_owned();
                thinking_end_tags = vec!["<|eom|>".to_owned(), "<|eot|>".to_owned()];
                build_muse_glimmer_parser(
                    builder,
                    tools_json.as_ref(),
                    json_schema.as_ref(),
                    inputs,
                )?
            }
            ChatFormat::MistralNemo | ChatFormat::Magistral => {
                supports_thinking = self.template_src.contains("[THINK]");
                if supports_thinking {
                    thinking_start_tag = "[THINK]".to_owned();
                    thinking_end_tags = vec!["[/THINK]".to_owned()];
                }
                build_mistral_parser(
                    builder,
                    &self.template_src,
                    tools_json.as_ref(),
                    json_schema.as_ref(),
                    inputs,
                    &parser_gen_prompt,
                )?
            }
            ChatFormat::FunctionaryV32 => {
                if tools_json.is_some() && inputs.tool_choice != 2 {
                    let tv = tools_json
                        .as_ref()
                        .ok_or_else(|| invalid("missing tools"))?;
                    build_simple_tagged_json_tools(
                        builder,
                        &TaggedJsonSpec {
                            section_start: "",
                            section_end: "",
                            name_prefix: ">>>",
                            name_suffix: "\n",
                            call_close: "",
                        },
                        tv,
                        inputs,
                        json_schema.as_ref(),
                        &parser_gen_prompt,
                    )?
                } else {
                    let body = content_or_schema_parser(&mut builder, json_schema.as_ref())?;
                    let root = if inputs.continue_final_message == 0 {
                        let all_pfx = opt_lit(&mut builder, "all\n");
                        builder.inner_mut().sequence(vec![all_pfx, body])
                    } else {
                        body
                    };
                    builder.set_root(root);
                    (builder.build()?, Vec::new(), false)
                }
            }
            ChatFormat::FunctionaryV31 if tools_json.is_some() && inputs.tool_choice != 2 => {
                let tv = tools_json
                    .as_ref()
                    .ok_or_else(|| invalid("missing tools"))?;
                build_simple_tagged_json_tools(
                    builder,
                    &TaggedJsonSpec {
                        section_start: "",
                        section_end: "",
                        name_prefix: "<function=",
                        name_suffix: ">",
                        call_close: "</function>",
                    },
                    tv,
                    inputs,
                    json_schema.as_ref(),
                    &parser_gen_prompt,
                )?
            }
            ChatFormat::FireFunctionV2 if tools_json.is_some() && inputs.tool_choice != 2 => {
                let tv = tools_json
                    .as_ref()
                    .ok_or_else(|| invalid("missing tools"))?;
                build_json_only_tools_parser(
                    builder,
                    &JsonOnlySpec {
                        section_start: "functools",
                        section_end: "",
                        name_field: "name",
                        args_field: "arguments",
                        array_wrapped: true,
                    },
                    tv,
                    inputs,
                    json_schema.as_ref(),
                )?
            }
            ChatFormat::Llama3X if tools_json.is_some() && inputs.tool_choice != 2 => {
                let tv = tools_json
                    .as_ref()
                    .ok_or_else(|| invalid("missing tools"))?;
                let sec_start = if self.template_src.contains("<|python_tag|>") {
                    "<|python_tag|>"
                } else {
                    ""
                };
                build_json_only_tools_parser(
                    builder,
                    &JsonOnlySpec {
                        section_start: sec_start,
                        section_end: "",
                        name_field: "name",
                        args_field: "parameters",
                        array_wrapped: false,
                    },
                    tv,
                    inputs,
                    json_schema.as_ref(),
                )?
            }
            ChatFormat::GigaChat3 if tools_json.is_some() && inputs.tool_choice != 2 => {
                let tv = tools_json
                    .as_ref()
                    .ok_or_else(|| invalid("missing tools"))?;
                let sec_start = if self.template_src.contains("<|function_call|>") {
                    "<|function_call|>"
                } else {
                    "function call<|role_sep|>\n"
                };
                build_json_only_tools_parser(
                    builder,
                    &JsonOnlySpec {
                        section_start: sec_start,
                        section_end: "",
                        name_field: "name",
                        args_field: "arguments",
                        array_wrapped: false,
                    },
                    tv,
                    inputs,
                    json_schema.as_ref(),
                )?
            }
            ChatFormat::CommandR7b => {
                let has_cohere_tags = self.template_src.contains("<|START_THINKING|>");
                if has_cohere_tags {
                    supports_thinking = true;
                    thinking_start_tag = "<|START_THINKING|>".to_owned();
                    thinking_end_tags = vec!["<|END_THINKING|>".to_owned()];
                }
                let (resp_open_tag, resp_close_tag) =
                    if self.template_src.contains("<|START_RESPONSE|>") {
                        ("<|START_RESPONSE|>", "<|END_RESPONSE|>")
                    } else {
                        ("<|START_TEXT|>", "<|END_TEXT|>")
                    };
                if tools_json.is_some() && inputs.tool_choice != 2 && has_cohere_tags {
                    let tv = tools_json
                        .as_ref()
                        .ok_or_else(|| invalid("missing tools"))?;
                    let mut p = Vec::new();
                    if let Some(r) = build_reasoning_segment(
                        &mut builder,
                        "<|START_THINKING|>",
                        &["<|END_THINKING|>"],
                        &parser_gen_prompt,
                        inputs.reasoning_format,
                        false,
                    ) {
                        p.push(r);
                    }
                    let tp = builder.standard_json_tools(&JsonToolsOptions {
                        section_start: "<|START_ACTION|>",
                        section_end: "<|END_ACTION|>",
                        tools: tv,
                        parallel_tool_calls: inputs.parallel_tool_calls,
                        force_tool_calls: inputs.tool_choice == 1,
                        per_call_start: "",
                        per_call_end: "",
                        fun_name_is_key: false,
                        tools_array_wrapped: true,
                        name_field: "tool_name",
                        args_field: "parameters",
                        id_field: "tool_call_id",
                        gen_id_field: "",
                        parameter_order: &[],
                        allow_raw_arguments: false,
                    })?;
                    let mut triggers = Vec::new();
                    let mut lazy = false;
                    if inputs.tool_choice == 1 {
                        p.push(tp);
                    } else {
                        let c_open = builder.inner_mut().literal(resp_open_tag);
                        let c_body = if let Some(schema) = json_schema.as_ref()
                            && !matches!(schema, JsonVal::Null)
                        {
                            builder.schema_content_parser(schema)?
                        } else {
                            let u_text = builder.inner_mut().until(resp_close_tag);
                            builder.content(u_text)
                        };
                        let c_close = builder.inner_mut().literal(resp_close_tag);
                        let text_blk = builder.inner_mut().sequence(vec![c_open, c_body, c_close]);
                        let opt_blk = builder.inner_mut().optional(text_blk);
                        p.push(opt_blk);
                        p.push(tp);
                        if json_schema.is_none() {
                            lazy = true;
                            triggers.push(TriggerDef {
                                kind: 1,
                                value: "<|START_ACTION|>".to_owned(),
                            });
                        }
                    }
                    let root = if p.len() == 1 {
                        p[0]
                    } else {
                        builder.inner_mut().sequence(p)
                    };
                    builder.set_root(root);
                    (builder.build()?, triggers, lazy)
                } else if tools_json.is_some() && inputs.tool_choice != 2 {
                    let tv = tools_json
                        .as_ref()
                        .ok_or_else(|| invalid("missing tools"))?;
                    build_json_only_tools_parser(
                        builder,
                        &JsonOnlySpec {
                            section_start: "Action:\n```json",
                            section_end: "```",
                            name_field: "tool_name",
                            args_field: "parameters",
                            array_wrapped: true,
                        },
                        tv,
                        inputs,
                        json_schema.as_ref(),
                    )?
                } else if has_cohere_tags {
                    let mut p = Vec::new();
                    if let Some(r) = build_reasoning_segment(
                        &mut builder,
                        "<|START_THINKING|>",
                        &["<|END_THINKING|>"],
                        &parser_gen_prompt,
                        inputs.reasoning_format,
                        false,
                    ) {
                        p.push(r);
                    }
                    let c_open = builder.inner_mut().literal(resp_open_tag);
                    let c_body = if let Some(schema) = json_schema.as_ref()
                        && !matches!(schema, JsonVal::Null)
                    {
                        builder.schema_content_parser(schema)?
                    } else {
                        let u_text = builder.inner_mut().until(resp_close_tag);
                        builder.content(u_text)
                    };
                    let c_close = builder.inner_mut().literal(resp_close_tag);
                    let text_blk = builder.inner_mut().sequence(vec![c_open, c_body, c_close]);
                    let fallback = content_or_schema_parser(&mut builder, json_schema.as_ref())?;
                    p.push(builder.inner_mut().or(text_blk, fallback));
                    let root = if p.len() == 1 {
                        p[0]
                    } else {
                        builder.inner_mut().sequence(p)
                    };
                    builder.set_root(root);
                    (builder.build()?, Vec::new(), false)
                } else {
                    let opt_open = opt_lit(&mut builder, "<|CHATBOT_TOKEN|>");
                    let body = if let Some(schema) = json_schema.as_ref()
                        && !matches!(schema, JsonVal::Null)
                    {
                        builder.schema_content_parser(schema)?
                    } else {
                        let u = builder.inner_mut().until("<|END_OF_TURN_TOKEN|>");
                        builder.content(u)
                    };
                    let opt_close = opt_lit(&mut builder, "<|END_OF_TURN_TOKEN|>");
                    let root = builder
                        .inner_mut()
                        .sequence(vec![opt_open, body, opt_close]);
                    builder.set_root(root);
                    (builder.build()?, Vec::new(), false)
                }
            }
            _ => {
                let analysis = self.ensure_auto().clone();
                if analysis.reasoning.mode != ReasoningMode::None {
                    supports_thinking = true;
                    thinking_start_tag = analysis.reasoning.start.trim().to_owned();
                    if !analysis.reasoning.end.trim().is_empty() {
                        thinking_end_tags = vec![analysis.reasoning.end.trim().to_owned()];
                    }
                }
                let has_tools = tools_json.is_some()
                    && inputs.tool_choice != 2
                    && analysis.tools.format.mode != ToolFormatMode::None;
                let force_tools = has_tools && inputs.tool_choice == 1;

                let gen_params = GenerationParams {
                    tools: tools_json.clone(),
                    messages: None,
                    json_schema: json_schema.clone(),
                    reasoning_format: i32::try_from(inputs.reasoning_format)?,
                    tool_choice: i32::try_from(inputs.tool_choice)?,
                    parallel_tool_calls: inputs.parallel_tool_calls,
                    enable_thinking: inputs.enable_thinking,
                    add_generation_prompt: inputs.add_generation_prompt,
                };
                let arena = self.auto.build_parser(&gen_params, &parser_gen_prompt)?;
                let mut triggers = Vec::new();
                let mut lazy = false;
                if has_tools && !force_tools && json_schema.is_none() {
                    let f = &analysis.tools.format;
                    if !f.section_start.trim().is_empty() {
                        lazy = true;
                        triggers.push(TriggerDef {
                            kind: 1,
                            value: f.section_start.trim().to_owned(),
                        });
                    } else if !f.per_call_start.trim().is_empty() {
                        lazy = true;
                        triggers.push(TriggerDef {
                            kind: 1,
                            value: f.per_call_start.trim().to_owned(),
                        });
                        if f.per_call_start.trim() == "<tool_call>"
                            && analysis.tools.function.name_prefix.trim() == "<function="
                        {
                            triggers.push(TriggerDef {
                                kind: 1,
                                value: "<function=".to_owned(),
                            });
                        }
                    } else if !analysis.tools.function.name_prefix.trim().is_empty() {
                        lazy = true;
                        triggers.push(TriggerDef {
                            kind: 1,
                            value: analysis.tools.function.name_prefix.trim().to_owned(),
                        });
                    } else if f.mode == ToolFormatMode::Json {
                        lazy = true;
                        triggers.push(TriggerDef {
                            kind: 2,
                            value: if f.tools_array_wrapped {
                                r"(\[)".to_owned()
                            } else {
                                r"(\{)".to_owned()
                            },
                        });
                    }
                }
                (arena, triggers, lazy)
            }
        };

        let parser_json = arena.to_json_string()?;
        let has_tools_active = !inputs.tools.is_empty() && inputs.tool_choice != 2;
        let need_grammar = !inputs.grammar.is_empty() || has_tools_active || json_schema.is_some();
        let grammar = if !inputs.grammar.is_empty() {
            inputs.grammar.clone()
        } else if need_grammar {
            let resolve_refs = Vec::new();
            let raw_g = arena.build_grammar(grammar_lazy, &resolve_refs)?;
            ensure_grammar_covers_gen_prompt(raw_g, &generation_prompt, grammar_lazy)
        } else {
            String::new()
        };

        let rf0_think_prefix = if format_kind != ChatFormat::Apriel15
            && format_kind != ChatFormat::KimiK3
            && inputs.reasoning_format == 0
            && inputs.continue_final_message == 0
            && supports_thinking
            && !thinking_start_tag.is_empty()
            && let Some(pos) = parser_gen_prompt.rfind(thinking_start_tag.trim())
        {
            let tail = &parser_gen_prompt[pos..];
            let s_trim = thinking_start_tag.trim();
            let after_start = tail.strip_prefix(s_trim).unwrap_or("");
            let is_open = after_start.trim().is_empty()
                && !thinking_end_tags
                    .iter()
                    .any(|e| parser_gen_prompt.trim_end().ends_with(e.trim()));
            let is_closed = !has_tools_active
                && thinking_end_tags.iter().any(|e| {
                    let mid = after_start.trim();
                    mid.strip_prefix(e.trim())
                        .is_some_and(|rest| rest.trim().is_empty())
                });
            if is_open || is_closed {
                tail.to_owned()
            } else {
                String::new()
            }
        } else {
            String::new()
        };
        let apriel_think_prefix = if format_kind == ChatFormat::Apriel15
            && inputs.continue_final_message == 0
            && parser_gen_prompt.ends_with("Here are my reasoning steps:\n")
        {
            "Here are my reasoning steps:\n".to_owned()
        } else {
            String::new()
        };
        let tool_param_schemas = if format_kind == ChatFormat::MinimaxM3 {
            tools_json
                .as_ref()
                .map_or_else(Vec::new, extract_tool_defs)
                .into_iter()
                .collect()
        } else {
            BTreeMap::new()
        };

        let res = AppliedChatParams {
            format: format_kind.as_i32(),
            prompt,
            grammar,
            grammar_lazy,
            generation_prompt,
            supports_thinking,
            thinking_start_tag,
            thinking_end_tags,
            grammar_triggers,
            parser: parser_json,
            reasoning_format: inputs.reasoning_format,
            format_kind,
            continue_mode: inputs.continue_final_message,
            prefill_reasoning,
            prefill_content,
            has_tools_active,
            rf0_think_prefix,
            apriel_think_prefix,
            tool_param_schemas,
        };
        self.cached_arena = Some(arena);
        self.applied = Some(res.clone());
        Ok(res)
    }

    fn parse_with_cache(
        &mut self,
        input: &[u8],
        partial: bool,
        store_handle: i64,
    ) -> std::result::Result<ChatMessage, (&'static str, String)> {
        let applied = self
            .applied
            .as_ref()
            .ok_or_else(|| ("runtime", "chat parse requires apply".to_owned()))?
            .clone();

        if self.cached_arena.is_none() {
            let loaded = ChatArena::from_json_str(&applied.parser)
                .map_err(|e| ("runtime", e.to_string()))?;
            self.cached_arena = Some(loaded);
        }
        let arena = self
            .cached_arena
            .as_ref()
            .ok_or_else(|| ("runtime", "missing cached chat arena".to_owned()))?;

        let safe_len = safe_utf8_prefix(input);
        let invalid_mid_utf8 = std::str::from_utf8(&input[..safe_len]).is_err();
        let flags = if partial {
            parse_flags::IS_PARTIAL
        } else {
            parse_flags::NONE
        };
        let mut msg = ChatMessage {
            role: "assistant".to_owned(),
            ..ChatMessage::default()
        };
        let mut ctx = ParseContext::new(&input[..safe_len], flags);
        match arena.parse_with_context(&mut ctx) {
            Ok(parsed) if parsed.result_type != ParseResultType::Fail && !invalid_mid_utf8 => {
                map_chat_ast(&ctx.ast, &parsed, &mut msg);
            }
            _ if invalid_mid_utf8 && applied.has_tools_active => {
                return Err(("runtime", "Invalid UTF-8 in chat parse input".to_owned()));
            }
            _ if invalid_mid_utf8 => {
                let lossy = String::from_utf8_lossy(&input[..safe_len])
                    .into_owned()
                    .into_bytes();
                let mut lossy_ctx = ParseContext::new(&lossy, flags);
                let parsed = arena
                    .parse_with_context(&mut lossy_ctx)
                    .map_err(|e| ("runtime", e.to_string()))?;
                if parsed.result_type == ParseResultType::Fail {
                    return Err(("runtime", "Failed to parse chat message".to_owned()));
                }
                map_chat_ast(&lossy_ctx.ast, &parsed, &mut msg);
            }
            Ok(_) => return Err(("runtime", "Failed to parse chat message".to_owned())),
            Err(e) => return Err(("runtime", e.to_string())),
        }

        if applied.format_kind == ChatFormat::Gemma4 {
            for tc in &mut msg.tool_calls {
                if let Some(norm) = convert_gemma4_object_to_json(&tc.arguments, partial) {
                    tc.arguments = norm;
                }
            }
        } else if applied.format_kind == ChatFormat::Lfm2 {
            for tc in &mut msg.tool_calls {
                tc.arguments = normalize_lfm2_python_to_json(&tc.arguments, partial);
            }
        } else if applied.format_kind == ChatFormat::MinimaxM3 {
            for tc in &mut msg.tool_calls {
                if let Some(params_schema) = applied.tool_param_schemas.get(&tc.name) {
                    tc.arguments =
                        convert_minimax_m3_args_to_json(&tc.arguments, params_schema, partial);
                }
            }
        } else if applied.format_kind == ChatFormat::KimiK2 {
            for tc in &mut msg.tool_calls {
                if tc.name.is_empty() && !tc.id.is_empty() {
                    let stripped = tc.id.strip_prefix("functions.").unwrap_or(&tc.id);
                    if let Some((raw_name, _)) = stripped.split_once(':') {
                        tc.name = raw_name.to_owned();
                    } else {
                        tc.name = stripped.to_owned();
                    }
                }
            }
        }
        if !applied.apriel_think_prefix.is_empty() {
            if applied.reasoning_format == 0 {
                msg.content = format!("{}{}", applied.apriel_think_prefix, msg.content);
            } else {
                msg.reasoning_content =
                    format!("{}{}", applied.apriel_think_prefix, msg.reasoning_content);
            }
        }
        if applied.reasoning_format == 0 && !applied.rf0_think_prefix.is_empty() {
            msg.content = format!("{}{}", applied.rf0_think_prefix, msg.content);
        }
        if applied.continue_mode != 0 {
            if !applied.prefill_reasoning.is_empty() && applied.reasoning_format != 0 {
                msg.reasoning_content =
                    format!("{}{}", applied.prefill_reasoning, msg.reasoning_content);
            }
            if !applied.prefill_content.is_empty() {
                msg.content = format!("{}{}", applied.prefill_content, msg.content);
            }
        }
        self.stored.insert(store_handle, msg.clone());
        Ok(msg)
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "Single-pass state machine normalizes Python string/bool/null syntax to JSON"
)]
fn normalize_lfm2_python_to_json(raw: &str, partial: bool) -> String {
    let bytes = raw.as_bytes();
    let mut out = String::with_capacity(raw.len() + 8);
    let mut i = 0_usize;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'"' {
            out.push('"');
            i += 1;
            while i < bytes.len() {
                let c = bytes[i];
                if c == b'\\' {
                    if i + 1 < bytes.len() {
                        out.push('\\');
                        if let Some(ch) = raw[i + 1..].chars().next() {
                            out.push(ch);
                            i += 1 + ch.len_utf8();
                        } else {
                            i += 1;
                        }
                        continue;
                    }
                    if partial {
                        return out;
                    }
                    out.push_str("\\\\");
                    i += 1;
                    continue;
                }
                if c == b'\n' {
                    out.push_str("\\n");
                    i += 1;
                    continue;
                }
                if c == b'\r' {
                    out.push_str("\\r");
                    i += 1;
                    continue;
                }
                if c == b'\t' {
                    out.push_str("\\t");
                    i += 1;
                    continue;
                }
                if let Some(ch) = raw[i..].chars().next() {
                    out.push(ch);
                    i += ch.len_utf8();
                    if ch == '"' {
                        break;
                    }
                } else {
                    break;
                }
            }
            continue;
        }
        if b == b'\'' {
            out.push('"');
            i += 1;
            while i < bytes.len() {
                let c = bytes[i];
                if c == b'\\' {
                    if i + 1 < bytes.len() {
                        let next = bytes[i + 1];
                        if next == b'\'' {
                            out.push('\'');
                            i += 2;
                        } else {
                            out.push('\\');
                            if let Some(ch) = raw[i + 1..].chars().next() {
                                out.push(ch);
                                i += 1 + ch.len_utf8();
                            } else {
                                i += 1;
                            }
                        }
                        continue;
                    }
                    if partial {
                        return out;
                    }
                    out.push_str("\\\\");
                    i += 1;
                    continue;
                }
                if c == b'\'' {
                    out.push('"');
                    i += 1;
                    break;
                }
                if c == b'"' {
                    out.push_str("\\\"");
                    i += 1;
                    continue;
                }
                if c == b'\n' {
                    out.push_str("\\n");
                    i += 1;
                    continue;
                }
                if c == b'\r' {
                    out.push_str("\\r");
                    i += 1;
                    continue;
                }
                if c == b'\t' {
                    out.push_str("\\t");
                    i += 1;
                    continue;
                }
                if let Some(ch) = raw[i..].chars().next() {
                    out.push(ch);
                    i += ch.len_utf8();
                } else {
                    break;
                }
            }
            continue;
        }
        let prev_is_ident = i > 0 && (bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
        if !prev_is_ident {
            let rem = &raw[i..];
            if partial
                && ["True", "False", "None"]
                    .iter()
                    .any(|kw| kw.starts_with(rem) && rem.len() < kw.len())
            {
                break;
            }
            let mut replaced = false;
            for (kw, rep) in [("True", "true"), ("False", "false"), ("None", "null")] {
                if rem.starts_with(kw) {
                    let after = i + kw.len();
                    let next_is_ident = after < bytes.len()
                        && (bytes[after].is_ascii_alphanumeric() || bytes[after] == b'_');
                    if !next_is_ident {
                        out.push_str(rep);
                        i = after;
                        replaced = true;
                        break;
                    }
                }
            }
            if replaced {
                continue;
            }
        }
        if let Some(ch) = raw[i..].chars().next() {
            out.push(ch);
            i += ch.len_utf8();
        } else {
            break;
        }
    }
    out
}

const MINIMAX_DELIM: &str = "]<]minimax[>[";

fn trim_minimax_delim_prefix(s: &str, closed: bool) -> &str {
    if closed {
        return s;
    }
    let s = s.find(MINIMAX_DELIM).map_or(s, |pos| &s[..pos]);
    for k in (1..MINIMAX_DELIM.len()).rev() {
        if s.ends_with(&MINIMAX_DELIM[..k]) {
            return &s[..s.len() - k];
        }
    }
    s
}

fn schema_has_string_branch(schema: &JsonVal) -> bool {
    if schema_resolves_to_string(schema) {
        return true;
    }
    for key in ["anyOf", "oneOf"] {
        let branches = schema.get(key).array();
        if !branches.is_empty() && branches.iter().any(schema_has_string_branch) {
            return true;
        }
    }
    false
}

fn find_minimax_close_tag<'a>(s: &'a str, tag: &str) -> Option<(&'a str, &'a str)> {
    let needle = format!("]<]minimax[>[</{tag}>");
    s.find(&needle)
        .map(|pos| (&s[..pos], &s[pos + needle.len()..]))
}

fn next_minimax_open_tag(s: &str) -> Option<(&str, &str)> {
    let trimmed = s.trim_start();
    let after_pfx = trimmed.strip_prefix("]<]minimax[>[<")?;
    if after_pfx.starts_with('/') {
        return None;
    }
    let gt = after_pfx.find('>')?;
    Some((&after_pfx[..gt], &after_pfx[gt + 1..]))
}

fn select_union_branch<'a>(
    schema: &'a JsonVal,
    raw_xml: &str,
    closed: bool,
) -> Option<&'a JsonVal> {
    let branches = if !schema.get("anyOf").array().is_empty() {
        schema.get("anyOf").array()
    } else if !schema.get("oneOf").array().is_empty() {
        schema.get("oneOf").array()
    } else {
        return Some(schema);
    };
    let trimmed = raw_xml.trim_start();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.starts_with(']') {
        let Some((tag, _)) = next_minimax_open_tag(trimmed) else {
            if !closed {
                return None;
            }
            return branches.first();
        };
        if tag == "item" {
            return branches
                .iter()
                .find(|b| b.get("type").text() == Some("array"));
        }
        return branches
            .iter()
            .find(|b| b.get("type").text() == Some("object") && b.get("properties").has(tag))
            .or_else(|| {
                branches
                    .iter()
                    .find(|b| b.get("type").text() == Some("object"))
            });
    }
    branches.iter().find(|b| {
        matches!(
            b.get("type").text(),
            Some("integer" | "number" | "boolean" | "string")
        )
    })
}

fn minimax_xml_to_json(xml: &str, schema: &JsonVal, closed: bool) -> String {
    let Some(resolved) = select_union_branch(schema, xml, closed) else {
        return String::new();
    };
    let kind = resolved.get("type").text().unwrap_or("");
    match kind {
        "array" => {
            let item_schema = resolved.get("items");
            let mut out = String::from("[");
            let mut rest = xml.trim_start();
            let mut idx = 0_usize;
            while let Some((tag, after_open)) = next_minimax_open_tag(rest) {
                if tag != "item" {
                    break;
                }
                if idx > 0 {
                    out.push_str(", ");
                }
                idx += 1;
                if let Some((inner, after_close)) = find_minimax_close_tag(after_open, "item") {
                    out.push_str(&minimax_xml_to_json(inner, item_schema, true));
                    rest = after_close.trim_start();
                } else {
                    out.push_str(&minimax_xml_to_json(after_open, item_schema, false));
                    break;
                }
            }
            if closed {
                out.push(']');
            }
            out
        }
        "object" => {
            let props = resolved.get("properties");
            let mut out = String::from("{");
            let mut rest = xml.trim_start();
            let mut idx = 0_usize;
            while let Some((key, after_open)) = next_minimax_open_tag(rest) {
                if idx > 0 {
                    out.push_str(", ");
                }
                idx += 1;
                if let Ok(k_json) = serde_json::to_string(key) {
                    out.push_str(&k_json);
                    out.push_str(": ");
                }
                let k_schema = props.get(key);
                if let Some((inner, after_close)) = find_minimax_close_tag(after_open, key) {
                    out.push_str(&minimax_xml_to_json(inner, k_schema, true));
                    rest = after_close.trim_start();
                } else {
                    out.push_str(&minimax_xml_to_json(after_open, k_schema, false));
                    break;
                }
            }
            if closed {
                out.push('}');
            }
            out
        }
        "string" => {
            let clean = trim_minimax_delim_prefix(xml, closed);
            let serialized = serde_json::to_string(clean).unwrap_or_else(|_| "\"\"".to_owned());
            if closed {
                serialized
            } else {
                serialized[..serialized.len().saturating_sub(1)].to_owned()
            }
        }
        _ => {
            let trimmed = trim_minimax_delim_prefix(xml, closed).trim();
            trimmed.to_owned()
        }
    }
}

fn unescape_partial_json_string(inner: &str, closed: bool) -> String {
    if closed {
        let wrapped = format!("\"{inner}\"");
        if let Ok(s) = serde_json::from_str::<String>(&wrapped) {
            return s;
        }
    }
    let bytes = inner.as_bytes();
    let mut i = 0_usize;
    let mut out = String::with_capacity(inner.len());
    while i < bytes.len() {
        if bytes[i] == b'\\' {
            if i + 1 >= bytes.len() {
                break;
            }
            match bytes[i + 1] {
                b'"' => out.push('"'),
                b'\\' => out.push('\\'),
                b'/' => out.push('/'),
                b'b' => out.push('\u{0008}'),
                b'f' => out.push('\u{000C}'),
                b'n' => out.push('\n'),
                b'r' => out.push('\r'),
                b't' => out.push('\t'),
                b'u' if i + 6 <= bytes.len() => {
                    if let Ok(code) = u32::from_str_radix(&inner[i + 2..i + 6], 16)
                        && let Some(ch) = char::from_u32(code)
                    {
                        out.push(ch);
                    }
                    i += 6;
                    continue;
                }
                b'u' => break,
                other => out.push(char::from(other)),
            }
            i += 2;
        } else if let Some(ch) = inner[i..].chars().next() {
            out.push(ch);
            i += ch.len_utf8();
        } else {
            break;
        }
    }
    out
}

fn convert_minimax_m3_args_to_json(raw: &str, params_schema: &JsonVal, partial: bool) -> String {
    let props = params_schema.get("properties");
    let bytes = raw.as_bytes();
    let mut i = 0_usize;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    if i >= bytes.len() || bytes[i] != b'{' {
        return raw.to_owned();
    }
    let mut out = String::from("{");
    i += 1;
    let mut first = true;
    while i < bytes.len() {
        while i < bytes.len() && (bytes[i].is_ascii_whitespace() || bytes[i] == b',') {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        if bytes[i] == b'}' {
            out.push('}');
            break;
        }
        if bytes[i] != b'"' {
            return raw.to_owned();
        }
        let k_start = i + 1;
        let mut j = k_start;
        while j < bytes.len() {
            if bytes[j] == b'\\' {
                j += 2;
            } else if bytes[j] == b'"' {
                break;
            } else {
                j += 1;
            }
        }
        if j >= bytes.len() {
            out.push_str(&raw[i..]);
            break;
        }
        let key_raw = &raw[i..=j];
        let key_name = serde_json::from_str::<String>(key_raw).unwrap_or_default();
        i = j + 1;
        while i < bytes.len() && (bytes[i].is_ascii_whitespace() || bytes[i] == b':') {
            i += 1;
        }
        if !first {
            out.push_str(", ");
        }
        first = false;
        out.push_str(key_raw);
        out.push_str(": ");
        if i >= bytes.len() {
            break;
        }
        let prop_schema = props.get(&key_name);
        if bytes[i] == b'"' {
            let v_start = i + 1;
            let mut k = v_start;
            let mut closed = false;
            while k < bytes.len() {
                if bytes[k] == b'\\' {
                    k += if k + 1 < bytes.len() { 2 } else { 1 };
                } else if bytes[k] == b'"' {
                    closed = true;
                    break;
                } else {
                    k += 1;
                }
            }
            let val_slice = &raw[i..k.min(bytes.len())];
            if schema_is_pure_scalar(prop_schema) || schema_has_string_branch(prop_schema) {
                out.push_str(val_slice);
                if closed {
                    out.push('"');
                }
            } else {
                let decoded =
                    unescape_partial_json_string(&raw[v_start..k.min(bytes.len())], closed);
                let conv = minimax_xml_to_json(&decoded, prop_schema, closed || !partial);
                out.push_str(&conv);
            }
            i = if closed { k + 1 } else { k };
        } else {
            let v_start = i;
            while i < bytes.len() && bytes[i] != b',' && bytes[i] != b'}' {
                i += 1;
            }
            out.push_str(raw[v_start..i].trim_end());
        }
    }
    out
}

fn execute_chat_trace(op: &Value) -> Result<Value> {
    let tmpl_src = string_bound(field(op, "Template")?, MAX_TEMPLATE_BYTES)?;
    let now_secs = current_unix_seconds();
    let mut state = ChatTraceState::new(tmpl_src, now_secs);
    let mut trace = Vec::new();

    for action in array(op, "Actions")? {
        let k = field(action, "Kind")?
            .as_str()
            .ok_or_else(|| invalid("action Kind must be string"))?;
        match k {
            "apply" => {
                let pjson = field(action, "ParametersJson")?;
                let inputs = parse_apply_inputs(pjson)?;
                let applied = state.apply(&inputs)?;
                trace.push(applied.to_wire_json());
            }
            "caps" => {
                let caps = state.tmpl.original_caps();
                trace.push(json!({ "Caps": caps_to_json(caps) }));
            }
            "parse" => {
                let bytes = decode_bytes(field(action, "Bytes")?)?;
                let partial = field(action, "Partial")?
                    .as_bool()
                    .ok_or_else(|| invalid("partial must be boolean"))?;
                let store = int_bound(field(action, "Store")?, 0, MAX_STORE_HANDLE)?;
                match state.parse_with_cache(&bytes, partial, store) {
                    Ok(msg) => {
                        trace.push(json!({
                            "ExceptionKind": null,
                            "Message": wire_message_to_json(&msg),
                        }));
                    }
                    Err((kind, msg)) => {
                        trace.push(wire_error(kind, &msg));
                    }
                }
            }
            "diff" => {
                let before = if field(action, "Before")?.is_null() {
                    ChatMessage {
                        role: "assistant".to_owned(),
                        ..ChatMessage::default()
                    }
                } else {
                    let b_idx = int_bound(field(action, "Before")?, 0, MAX_STORE_HANDLE)?;
                    state
                        .stored
                        .get(&b_idx)
                        .ok_or_else(|| invalid("unknown before handle"))?
                        .clone()
                };
                let a_idx = int_bound(field(action, "After")?, 0, MAX_STORE_HANDLE)?;
                let after = state
                    .stored
                    .get(&a_idx)
                    .ok_or_else(|| invalid("unknown after handle"))?;
                let diffs = compute_message_diffs(&before, after)?;
                trace.push(json!({ "Diffs": wire_diffs_to_json(&diffs) }));
            }
            _ => return Err(invalid("unknown chat action")),
        }
    }

    Ok(json!({ "Trace": trace }))
}

// ---------------------------------------------------------------------------
// chat-generic-parse
// ---------------------------------------------------------------------------

fn execute_generic_parse(op: &Value) -> Result<Value> {
    let input = decode_bytes(field(op, "Bytes")?)?;
    let partial = field(op, "Partial")?
        .as_bool()
        .ok_or_else(|| invalid("partial must be boolean"))?;
    let safe_len = if partial {
        safe_utf8_prefix(&input)
    } else {
        input.len()
    };
    let content = String::from_utf8(input[..safe_len].to_vec())
        .unwrap_or_else(|_| String::from_utf8_lossy(&input[..safe_len]).into_owned());
    let msg = ChatMessage {
        role: "assistant".to_owned(),
        content,
        ..ChatMessage::default()
    };
    Ok(json!({ "Message": wire_message_to_json(&msg) }))
}

// ---------------------------------------------------------------------------
// chat-server-params (oaicompat_chat_params_parse)
// ---------------------------------------------------------------------------

fn reasoning_format_name(code: i64) -> &'static str {
    match code {
        1 => "auto",
        2 => "deepseek-legacy",
        3 => "deepseek",
        _ => "none",
    }
}

fn parse_reasoning_format_str(name: &str) -> Option<i64> {
    match name {
        "none" => Some(0),
        "auto" => Some(1),
        "deepseek-legacy" => Some(2),
        "deepseek" => Some(3),
        _ => None,
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "Direct translation of oaicompat_chat_params_parse option and body mapping"
)]
fn execute_server_params(op: &Value) -> Result<Value> {
    let tmpl_src = string_bound(field(op, "Template")?, MAX_TEMPLATE_BYTES)?;
    let opts = field(op, "Options")?;
    let body = parse_raw_ord(field(op, "BodyJson")?)?;

    let mut inputs = ChatApplyInputs::default();
    if let Some(use_jinja) = opts.get("use_jinja").and_then(Value::as_bool) {
        inputs.use_jinja = use_jinja;
    }
    let mut enable_thinking_opt = opts.get("enable_thinking").and_then(Value::as_bool);
    if let Some(et) = enable_thinking_opt {
        inputs.enable_thinking = et;
    }
    if let Some(rf) = opts.get("reasoning_format").and_then(Value::as_i64) {
        inputs.reasoning_format = rf;
    }
    let reasoning_budget = body
        .get("reasoning_budget_tokens")
        .or_else(|| body.get("thinking_budget_tokens"))
        .and_then(OrdJson::as_i64)
        .or_else(|| opts.get("reasoning_budget").and_then(Value::as_i64))
        .unwrap_or(-1);
    let reasoning_budget_msg = body
        .get("reasoning_budget_message")
        .and_then(OrdJson::as_str)
        .or_else(|| opts.get("reasoning_budget_message").and_then(Value::as_str))
        .unwrap_or_default();

    let tmpl = ChatTemplate::new_with_now(
        if tmpl_src.is_empty() {
            CHATML_JINJA_TEMPLATE
        } else {
            tmpl_src
        },
        "",
        "",
        current_unix_seconds(),
    );
    let caps = tmpl.original_caps();

    if let Some(msgs_ord) = body.get("messages") {
        let mut parsed_msgs = parse_oai_messages(msgs_ord).map_err(invalid)?;
        let supports_developer =
            tmpl_src.contains("'developer'") || tmpl_src.contains("\"developer\"");
        if !supports_developer {
            for msg in &mut parsed_msgs {
                if msg.role == "developer" {
                    msg.role = "system".to_owned();
                }
            }
        }
        inputs.messages = parsed_msgs;
    }
    if let Some(tools_ord) = body.get("tools")
        && !matches!(tools_ord, OrdJson::Null)
    {
        inputs.tools = parse_oai_tools(tools_ord)?;
    }
    if let Some(ptc) = body.get("parallel_tool_calls").and_then(OrdJson::as_bool) {
        inputs.parallel_tool_calls = ptc;
    }
    if let Some(tc) = body.get("tool_choice") {
        inputs.tool_choice = match tc.as_str() {
            Some("required") => 1,
            Some("none") => 2,
            _ => 0,
        };
    }
    if let Some(rf_str) = body.get("reasoning_format").and_then(OrdJson::as_str)
        && let Some(rf) = parse_reasoning_format_str(rf_str)
    {
        inputs.reasoning_format = rf;
    }

    if let Some(kwargs) = body
        .get("chat_template_kwargs")
        .and_then(OrdJson::as_object)
    {
        for (k, v) in kwargs {
            if k == "enable_thinking"
                && let Some(b) = v.as_bool()
            {
                enable_thinking_opt = Some(b);
                inputs.enable_thinking = b;
            }
            let s = if let Some(sv) = v.as_str() {
                sv.to_owned()
            } else {
                v.dump_compact()?
            };
            inputs.chat_template_kwargs.insert(k.clone(), s);
        }
    }

    if let Some(effort) = body.get("reasoning_effort").and_then(OrdJson::as_str) {
        if effort == "none" {
            inputs.enable_thinking = false;
            enable_thinking_opt = Some(false);
        } else if caps.supports_reasoning_effort {
            inputs
                .chat_template_kwargs
                .insert("reasoning_effort".to_owned(), effort.to_owned());
        }
    }

    if reasoning_budget == 0 {
        inputs.enable_thinking = false;
        enable_thinking_opt = Some(false);
    }
    let _ = enable_thinking_opt;

    if let Some(rf) = body.get("response_format") {
        let rtype = rf.get("type").and_then(OrdJson::as_str).unwrap_or("");
        if rtype == "json_object" {
            inputs.json_schema = r#"{"type":"object"}"#.to_owned();
        } else if rtype == "json_schema"
            && let Some(schema_wrap) = rf.get("json_schema")
        {
            let inner_schema = schema_wrap.get("schema").unwrap_or(schema_wrap);
            inputs.json_schema = inner_schema.dump_compact()?;
        }
    }
    if let Some(g) = body.get("grammar").and_then(OrdJson::as_str) {
        inputs.grammar = g.to_owned();
    }

    let mut state = ChatTraceState::new(tmpl_src, current_unix_seconds());
    let applied = state.apply(&inputs)?;

    let mut result = Map::new();
    result.insert("prompt".to_owned(), json!(applied.prompt));
    result.insert("chat_format".to_owned(), json!(applied.format));
    result.insert(
        "generation_prompt".to_owned(),
        json!(applied.generation_prompt),
    );
    result.insert(
        "reasoning_format".to_owned(),
        json!(reasoning_format_name(inputs.reasoning_format)),
    );
    result.insert("chat_parser".to_owned(), json!(applied.parser));
    if !applied.grammar.is_empty() {
        result.insert("grammar".to_owned(), json!(applied.grammar));
        result.insert("grammar_lazy".to_owned(), json!(applied.grammar_lazy));
        let trigs: Vec<Value> = applied
            .grammar_triggers
            .iter()
            .map(|t| json!({"type": t.kind, "value": t.value}))
            .collect();
        result.insert("grammar_triggers".to_owned(), Value::Array(trigs));
    }
    if reasoning_budget >= 0 {
        result.insert(
            "reasoning_budget_tokens".to_owned(),
            json!(reasoning_budget),
        );
    }
    if !reasoning_budget_msg.is_empty() {
        result.insert(
            "reasoning_budget_message".to_owned(),
            json!(reasoning_budget_msg),
        );
    }
    let thinking_forced_open = !applied.generation_prompt.is_empty()
        && !applied.thinking_start_tag.is_empty()
        && applied
            .generation_prompt
            .trim_end()
            .ends_with(applied.thinking_start_tag.trim_end());
    result.insert(
        "thinking_forced_open".to_owned(),
        json!(thinking_forced_open),
    );
    let mut stops: Vec<Value> = state
        .auto
        .analysis()
        .additional_stops
        .iter()
        .map(|s| json!(s))
        .collect();
    if let Some(body_stop) = body.get("stop") {
        match body_stop {
            OrdJson::Str(s) => stops.push(json!(s)),
            OrdJson::Array(arr) => {
                for s in arr {
                    if let Some(txt) = s.as_str() {
                        stops.push(json!(txt));
                    }
                }
            }
            _ => {}
        }
    }
    if !stops.is_empty() {
        result.insert("stop".to_owned(), Value::Array(stops));
    }
    let preserved: Vec<Value> = state
        .auto
        .analysis()
        .preserved_tokens
        .iter()
        .map(|s| json!(s))
        .collect();
    if !preserved.is_empty() {
        result.insert("preserved_tokens".to_owned(), Value::Array(preserved));
    }
    if let Some(max_tok) = body
        .get("max_completion_tokens")
        .or_else(|| body.get("max_tokens"))
        .and_then(OrdJson::as_i64)
    {
        result.insert("n_predict".to_owned(), json!(max_tok));
    }
    if let Some(stream) = body.get("stream").and_then(OrdJson::as_bool) {
        result.insert("stream".to_owned(), json!(stream));
    }
    for pass_key in ["temperature", "top_p", "top_k", "seed"] {
        if let Some(v) = body.get(pass_key) {
            result.insert(pass_key.to_owned(), v.to_serde_value()?);
        }
    }

    Ok(json!({ "Value": Value::Object(result) }))
}

fn execute_operation(op: &Value) -> Result<Value> {
    let kind = field(op, "Kind")?
        .as_str()
        .ok_or_else(|| invalid("operation Kind must be string"))?;
    match kind {
        "chat-trace" => execute_chat_trace(op),
        "chat-regex" => execute_chat_regex(op),
        "chat-generic-parse" => execute_generic_parse(op),
        "chat-server-params" => execute_server_params(op),
        "chat-value" => execute_chat_value(op),
        _ => Err(invalid("unknown chat operation")),
    }
}

/// Returns `true` when every operation in `request` belongs to `chat.hpp` (`chat-*`).
pub(super) fn handles(request: &Value) -> bool {
    request["Operations"].as_array().is_some_and(|operations| {
        !operations.is_empty()
            && operations.iter().all(|op| {
                matches!(
                    op["Kind"].as_str(),
                    Some(
                        "chat-trace"
                            | "chat-value"
                            | "chat-server-params"
                            | "chat-generic-parse"
                            | "chat-regex"
                    )
                )
            })
    })
}

/// Validates and executes a `chat-*` wire request.
pub(super) fn execute(request: &Value) -> Result<Value> {
    exact_fields(request, &["SchemaVersion", "Operations"])?;
    if int_bound(field(request, "SchemaVersion")?, 1, 1)? != 1 {
        return Err(invalid("chat request SchemaVersion must equal 1"));
    }
    let operations = array(request, "Operations")?;
    require(
        !operations.is_empty() && operations.len() <= MAX_OPERATIONS,
        "chat Operations must contain 1 through 128 operations",
    )?;
    for operation in operations {
        validate_operation(operation)?;
    }
    let results = operations
        .iter()
        .map(execute_operation)
        .collect::<Result<Vec<_>>>()?;
    Ok(json!({ "SchemaVersion": 1, "Results": results }))
}

/// Provider identity metadata for the independent chat provider reference.
pub(super) fn identity() -> Value {
    json!({
        "Kind": "independent-rust-chat",
        "SupportedSourceIds": [ID],
        "UpstreamCommit": super::UPSTREAM,
        "AssertionSource": {
            "Path": ID,
            "Sha256": SOURCE_SHA256,
        },
        "NativeReferenceReplay": false,
        "ProductionRuntimeCalled": false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, Target};

    #[test]
    fn independent_chat_suite_passes_on_rust_and_cpp_targets() -> Result<()> {
        for target in [Target::RustReference, Target::Cpp] {
            let engine = Engine::new(target, None, None, None, None)?;
            let outcome = crate::chat::run(&engine)?;
            assert_eq!(outcome["Id"], ID);
            assert_eq!(outcome["Status"], "pass", "chat suite failed: {outcome}");
        }
        assert!(crate::reference_provider::supports(ID));
        let meta = identity();
        assert_eq!(meta["NativeReferenceReplay"], false);
        assert_eq!(meta["ProductionRuntimeCalled"], false);
        Ok(())
    }

    #[test]
    fn chat_regex_handles_search_match_optional_captures_and_empty_inverted_class() -> Result<()> {
        let req = json!({
            "SchemaVersion": 1,
            "Operations": [{
                "Kind": "chat-regex",
                "Patterns": [
                    r"(?:^|\n)(?:(foo)|(bar))baz([^]*)",
                    r"layer\.(\d+)\.attn",
                ],
                "Actions": [
                    {
                        "Pattern": 0,
                        "Function": "match",
                        "Bytes": b"barbaz\nx".to_vec(),
                    },
                    {
                        "Pattern": 1,
                        "Function": "search",
                        "Bytes": b"blk.12.layer.7.attn.q".to_vec(),
                    }
                ]
            }]
        });
        let resp = execute(&req)?;
        let trace = &resp["Results"][0]["Trace"];
        assert_eq!(trace[0]["Matched"], json!(true));
        assert_eq!(trace[0]["MarkCount"], json!(3));
        assert_eq!(
            trace[0]["Captures"][1],
            json!({"Position": -1, "Length": 0})
        );
        assert_eq!(trace[0]["Captures"][2], json!({"Position": 0, "Length": 3}));
        assert_eq!(trace[1]["Matched"], json!(true));
        assert_eq!(
            trace[1]["Captures"][1],
            json!({"Position": 13, "Length": 1})
        );
        Ok(())
    }

    #[test]
    fn chat_value_oai_diff_and_delimiter_split_roundtrip() -> Result<()> {
        let req = json!({
            "SchemaVersion": 1,
            "Operations": [
                {
                    "Kind": "chat-value",
                    "Function": "messages-from-oai",
                    "Json": serde_json::to_string(&json!([
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": null, "tool_calls": [
                            {"id": "c1", "type": "function", "function": {"name": "get", "arguments": "{\"x\":1}"}}
                        ]}
                    ]))?
                },
                {
                    "Kind": "chat-value",
                    "Function": "delimiter-split",
                    "Delimiters": [
                        {"Role": 3, "Delimiter": "<user>", "Tokens": [10, 11]},
                        {"Role": 0, "Delimiter": "<end>", "Tokens": [12]}
                    ],
                    "Tokens": [1, 10, 11, 5, 10, 11, 6, 12],
                    "Skips": [[4, 2]]
                }
            ]
        });
        let resp = execute(&req)?;
        assert!(resp["Results"][0]["ExceptionKind"].is_null());
        assert_eq!(
            resp["Results"][0]["Messages"][1]["tool_calls"][0]["name"],
            json!("get")
        );
        assert_eq!(
            resp["Results"][1]["spans"],
            json!([
                {"role": 3, "pos": 1, "len": 6},
                {"role": 0, "pos": 7, "len": 1}
            ])
        );
        Ok(())
    }

    #[test]
    fn chat_generic_parse_and_server_params_work_cleanly() -> Result<()> {
        let req = json!({
            "SchemaVersion": 1,
            "Operations": [
                {
                    "Kind": "chat-generic-parse",
                    "Bytes": b"Hello".to_vec(),
                    "Partial": false,
                },
                {
                    "Kind": "chat-server-params",
                    "Template": "",
                    "Options": {
                        "use_jinja": true,
                        "reasoning_budget": 128,
                    },
                    "BodyJson": serde_json::to_string(&json!({
                        "messages": [{"role": "developer", "content": "be concise"}, {"role": "user", "content": "hi"}],
                        "response_format": {"type": "json_object"}
                    }))?
                }
            ]
        });
        let resp = execute(&req)?;
        assert_eq!(resp["Results"][0]["Message"]["content"], json!("Hello"));
        let server_val = &resp["Results"][1]["Value"];
        assert!(
            server_val["prompt"]
                .as_str()
                .is_some_and(|p| p.contains("system\nbe concise"))
        );
        assert!(
            server_val["grammar"]
                .as_str()
                .is_some_and(|g| !g.is_empty())
        );
        assert_eq!(server_val["reasoning_budget_tokens"], json!(128));
        Ok(())
    }

    #[test]
    fn chat_validation_rejects_out_of_bounds_and_invalid_handles() {
        let bad_regex = json!({
            "SchemaVersion": 1,
            "Operations": [{
                "Kind": "chat-regex",
                "Patterns": ["a"],
                "Actions": [{"Pattern": 1, "Function": "search", "Bytes": [97]}]
            }]
        });
        assert!(execute(&bad_regex).is_err());

        let missing_apply = json!({
            "SchemaVersion": 1,
            "Operations": [{
                "Kind": "chat-trace",
                "Template": "",
                "Actions": [{"Kind": "parse", "Bytes": [97], "Partial": false, "Store": 0}]
            }]
        });
        assert!(execute(&missing_apply).is_err());
    }
}
