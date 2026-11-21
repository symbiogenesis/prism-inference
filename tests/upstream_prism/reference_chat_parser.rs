//! Clean-room independent safe Rust chat PEG builder, AST mapper, and message diff provider.
//!
//! Implements the `chat-peg-trace` protocol directly from
//! `tools/prism/upstream/common/chat_parser.hpp`, `PROTOCOL.md`, and
//! `tests/upstream_prism/chat_peg.rs`, layered on `reference_peg` and
//! `reference_json_schema`. It never imports `bonsai_prism_runtime`, invokes a
//! native bridge, or reads expected-answer fixtures.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use serde_json::{Value, json};

use super::llama_grammar::{array, exact_fields, field};
use super::reference_json_schema::{JsonParser, JsonVal, SchemaGrammarContext};
use super::reference_peg::{
    Arena, AstArena, AstId, AstNode, ParseContext, ParseResult, ParseResultType, ParserBuilder,
    ParserId, parse_flags,
};
use super::{Result, invalid};

/// Upstream source identifier covered by this independent reference provider.
pub(super) const ID: &str = "tests/test-chat-peg-parser.cpp";
const SOURCE_SHA256: &str = "c0583c716c65fa7c06fc9b282fc8d24aecde29fa55271536e8c23dfbbc0429e0";

const MAX_OPERATIONS: usize = 128;
const MAX_GRAPH_DEPTH: usize = 48;
const MAX_EXPR_NODES: usize = 16_384;
const MAX_STATEMENTS: usize = 4096;
const MAX_LOCAL_NAME: usize = 128;
const MAX_STRING_BYTES: usize = 65_536;
const MAX_SCHEMA_JSON_BYTES: usize = 1024 * 1024;
const MAX_INPUT_BYTES: usize = 1024 * 1024;
const MAX_ARGS: usize = 64;
const MAX_LIST_LEN: usize = 1024;
const MAX_PARAM_ORDER: usize = 128;
const MAX_RESOLVE_REFS: usize = 128;
const MAX_ACTIONS: usize = 32_768;
const MAX_INPUT_HANDLES: i64 = 31;
const MAX_PERMUTE_EXACT: usize = 4;

const TAG_REASONING_BLOCK: &str = "reasoning-block";
const TAG_REASONING: &str = "reasoning";
const TAG_CONTENT: &str = "content";
const TAG_TOOL: &str = "tool";
const TAG_TOOL_OPEN: &str = "tool-open";
const TAG_TOOL_CLOSE: &str = "tool-close";
const TAG_TOOL_ID: &str = "tool-id";
const TAG_TOOL_NAME: &str = "tool-name";
const TAG_TOOL_ARGS: &str = "tool-args";
const TAG_TOOL_ARG: &str = "tool-arg";
const TAG_TOOL_ARG_OPEN: &str = "tool-arg-open";
const TAG_TOOL_ARG_CLOSE: &str = "tool-arg-close";
const TAG_TOOL_ARG_NAME: &str = "tool-arg-name";
const TAG_TOOL_ARG_STRING_VALUE: &str = "tool-arg-string-value";
const TAG_TOOL_ARG_JSON_VALUE: &str = "tool-arg-json-value";

// ---------------------------------------------------------------------------
// Chat message, tool call, diff, and AST mapper domain types
// ---------------------------------------------------------------------------

/// Parsed or accumulated tool call fields.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct ChatToolCall {
    /// Function/tool name.
    pub(super) name: String,
    /// Raw or serialized JSON arguments text.
    pub(super) arguments: String,
    /// Optional tool call identifier.
    pub(super) id: String,
}

/// Content part descriptor retained for downstream chat providers.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[allow(dead_code, reason = "Exposed for Stage 3-4 chat_auto/chat providers")]
pub(super) struct ChatContentPart {
    /// Part type (e.g. `"text"`).
    pub(super) part_type: String,
    /// Part text body.
    pub(super) text: String,
}

/// Parsed chat message structure matching `common_chat_msg`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct ChatMessage {
    /// Message role (e.g. `""` from raw mapper, `"assistant"` from chat parser).
    pub(super) role: String,
    /// Assistant text content.
    pub(super) content: String,
    /// Reasoning/thinking text content.
    pub(super) reasoning_content: String,
    /// Parsed tool calls in order.
    pub(super) tool_calls: Vec<ChatToolCall>,
    /// Optional structured content parts.
    pub(super) content_parts: Vec<ChatContentPart>,
    /// Optional tool response name.
    pub(super) tool_name: String,
    /// Optional tool response call ID.
    pub(super) tool_call_id: String,
}

/// Incremental diff record matching `common_chat_msg_diff`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct ChatMessageDiff {
    /// Incremental delta appended to `reasoning_content`.
    pub(super) reasoning_content_delta: String,
    /// Incremental delta appended to `content`.
    pub(super) content_delta: String,
    /// Index of modified/created tool call, or `None` (`npos`).
    pub(super) tool_call_index: Option<usize>,
    /// Incremental tool call delta fields.
    pub(super) tool_call_delta: ChatToolCall,
}

/// Computes streaming message deltas from `before` to `after`.
pub(super) fn compute_message_diffs(
    before: &ChatMessage,
    after: &ChatMessage,
) -> Result<Vec<ChatMessageDiff>> {
    let mut diffs = Vec::new();
    if after.reasoning_content != before.reasoning_content {
        let delta = after
            .reasoning_content
            .strip_prefix(&before.reasoning_content)
            .ok_or_else(|| {
                invalid(
                    "Cannot compute diff for reasoning_content: new_msg does not start with previous_msg",
                )
            })?;
        diffs.push(ChatMessageDiff {
            reasoning_content_delta: delta.to_owned(),
            ..ChatMessageDiff::default()
        });
    }
    if after.content != before.content {
        let delta = after.content.strip_prefix(&before.content).ok_or_else(|| {
            invalid("Cannot compute diff for content: new_msg does not start with previous_msg")
        })?;
        diffs.push(ChatMessageDiff {
            content_delta: delta.to_owned(),
            ..ChatMessageDiff::default()
        });
    }
    if after.tool_calls.len() < before.tool_calls.len() {
        return Err(invalid(
            "Cannot compute diff for tool_calls: fewer tool calls than previous message",
        ));
    }
    for (idx, curr) in after.tool_calls.iter().enumerate() {
        if let Some(prev) = before.tool_calls.get(idx) {
            if prev == curr {
                continue;
            }
            let mut delta = ChatToolCall::default();
            if prev.name != curr.name {
                if !prev.name.is_empty() {
                    return Err(invalid(
                        "Cannot compute diff for tool_calls: tool call name changed",
                    ));
                }
                delta.name.clone_from(&curr.name);
            }
            if prev.id != curr.id {
                if !prev.id.is_empty() {
                    return Err(invalid(
                        "Cannot compute diff for tool_calls: tool call id changed",
                    ));
                }
                delta.id.clone_from(&curr.id);
            }
            if prev.arguments != curr.arguments {
                let arg_delta = curr.arguments.strip_prefix(&prev.arguments).ok_or_else(|| {
                    invalid(
                        "Cannot compute diff for tool_calls: arguments do not start with previous",
                    )
                })?;
                delta.arguments = arg_delta.to_owned();
            }
            diffs.push(ChatMessageDiff {
                tool_call_index: Some(idx),
                tool_call_delta: delta,
                ..ChatMessageDiff::default()
            });
        } else {
            diffs.push(ChatMessageDiff {
                tool_call_index: Some(idx),
                tool_call_delta: curr.clone(),
                ..ChatMessageDiff::default()
            });
        }
    }
    Ok(diffs)
}

#[derive(Default)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "Tracks orthogonal incremental constructed tool AST state flags"
)]
struct ToolBuildState {
    call: ChatToolCall,
    constructed: bool,
    has_args: bool,
    in_string_value: bool,
    closed_object: bool,
}

impl ToolBuildState {
    fn open_arg_name(&mut self, name: &str) {
        self.constructed = true;
        if self.in_string_value {
            self.call.arguments.push('"');
            self.in_string_value = false;
        }
        if self.has_args {
            self.call.arguments.push(',');
        } else {
            self.call.arguments.push('{');
            self.has_args = true;
        }
        push_json_escaped_quoted(&mut self.call.arguments, name);
        self.call.arguments.push(':');
    }

    fn push_string_value(&mut self, text: &str) {
        self.constructed = true;
        if !self.in_string_value {
            self.call.arguments.push('"');
            self.in_string_value = true;
        }
        push_json_escaped_body(&mut self.call.arguments, text);
    }

    fn close_arg(&mut self) {
        if self.in_string_value {
            self.call.arguments.push('"');
            self.in_string_value = false;
        }
    }

    fn push_json_value(&mut self, text: &str) {
        self.constructed = true;
        self.in_string_value = false;
        self.call.arguments.push_str(text);
    }

    fn close_tool(&mut self) {
        self.close_arg();
        if !self.closed_object {
            if self.has_args {
                self.call.arguments.push('}');
                self.closed_object = true;
            } else if self.call.arguments.is_empty() {
                self.call.arguments.push_str("{}");
                self.closed_object = true;
            }
        }
    }

    fn has_content(&self) -> bool {
        !self.call.name.is_empty() || !self.call.id.is_empty() || !self.call.arguments.is_empty()
    }
}

fn push_json_escaped_body(out: &mut String, text: &str) {
    use std::fmt::Write as _;
    for ch in text.chars() {
        match ch {
            '"' => out.push_str(r#"\""#),
            '\\' => out.push_str(r"\\"),
            '\u{08}' => out.push_str(r"\b"),
            '\u{0C}' => out.push_str(r"\f"),
            '\n' => out.push_str(r"\n"),
            '\r' => out.push_str(r"\r"),
            '\t' => out.push_str(r"\t"),
            c if u32::from(c) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            c => out.push(c),
        }
    }
}

fn push_json_escaped_quoted(out: &mut String, text: &str) {
    out.push('"');
    push_json_escaped_body(out, text);
    out.push('"');
}

struct ChatAstMapper<'a> {
    ast: &'a AstArena,
    is_complete: bool,
    msg: &'a mut ChatMessage,
    current_tool: Option<ToolBuildState>,
}

impl ChatAstMapper<'_> {
    fn subtree_has_constructed_arg(&self, ids: &[AstId]) -> bool {
        for &id in ids {
            if let Some(node) = self.ast.get(id) {
                if matches!(
                    node.tag.as_str(),
                    TAG_TOOL_ARG
                        | TAG_TOOL_ARG_OPEN
                        | TAG_TOOL_ARG_CLOSE
                        | TAG_TOOL_ARG_NAME
                        | TAG_TOOL_ARG_STRING_VALUE
                        | TAG_TOOL_ARG_JSON_VALUE
                ) {
                    return true;
                }
                if self.subtree_has_constructed_arg(&node.children) {
                    return true;
                }
            }
        }
        false
    }

    fn flush_tool(&mut self) {
        if let Some(mut tool) = self.current_tool.take() {
            if self.is_complete && tool.constructed && !tool.closed_object {
                tool.close_tool();
            }
            if tool.has_content() {
                self.msg.tool_calls.push(tool.call);
            }
        }
    }

    fn visit_list(&mut self, ids: &[AstId]) {
        for &id in ids {
            if let Some(node) = self.ast.get(id) {
                self.visit_node(node);
            }
        }
    }

    fn visit_node(&mut self, node: &AstNode) {
        match node.tag.as_str() {
            TAG_REASONING => {
                self.msg.reasoning_content.push_str(&node.text);
            }
            TAG_CONTENT => {
                self.msg.content.push_str(&node.text);
            }
            TAG_TOOL => {
                self.flush_tool();
                self.current_tool = Some(ToolBuildState::default());
                self.visit_list(&node.children);
                self.flush_tool();
            }
            TAG_TOOL_OPEN => {
                if self.current_tool.is_none() {
                    self.current_tool = Some(ToolBuildState::default());
                }
                self.visit_list(&node.children);
            }
            TAG_TOOL_CLOSE => {
                if let Some(tool) = self.current_tool.as_mut() {
                    tool.close_tool();
                }
            }
            TAG_TOOL_ID => {
                let tool = self
                    .current_tool
                    .get_or_insert_with(ToolBuildState::default);
                tool.call.id.clone_from(&node.text);
            }
            TAG_TOOL_NAME => {
                let tool = self
                    .current_tool
                    .get_or_insert_with(ToolBuildState::default);
                tool.call.name.clone_from(&node.text);
            }
            TAG_TOOL_ARGS => {
                if self.subtree_has_constructed_arg(&node.children) {
                    let _ = self
                        .current_tool
                        .get_or_insert_with(ToolBuildState::default);
                    self.visit_list(&node.children);
                } else {
                    let tool = self
                        .current_tool
                        .get_or_insert_with(ToolBuildState::default);
                    tool.call.arguments.clone_from(&node.text);
                }
            }
            TAG_TOOL_ARG => {
                let _ = self
                    .current_tool
                    .get_or_insert_with(ToolBuildState::default);
                self.visit_list(&node.children);
                if self.is_complete
                    && let Some(tool) = self.current_tool.as_mut()
                {
                    tool.close_arg();
                }
            }
            TAG_TOOL_ARG_NAME => {
                let tool = self
                    .current_tool
                    .get_or_insert_with(ToolBuildState::default);
                tool.open_arg_name(&node.text);
            }
            TAG_TOOL_ARG_STRING_VALUE => {
                let tool = self
                    .current_tool
                    .get_or_insert_with(ToolBuildState::default);
                tool.push_string_value(&node.text);
            }
            TAG_TOOL_ARG_JSON_VALUE => {
                let tool = self
                    .current_tool
                    .get_or_insert_with(ToolBuildState::default);
                tool.push_json_value(&node.text);
            }
            TAG_TOOL_ARG_CLOSE => {
                if let Some(tool) = self.current_tool.as_mut() {
                    tool.close_arg();
                }
            }
            _ => {
                self.visit_list(&node.children);
            }
        }
    }
}

/// Maps captured AST nodes from `parsed` into `msg`.
pub(super) fn map_chat_ast(ast: &AstArena, parsed: &ParseResult, msg: &mut ChatMessage) {
    let mut mapper = ChatAstMapper {
        ast,
        is_complete: parsed.result_type == ParseResultType::Success,
        msg,
        current_tool: None,
    };
    mapper.visit_list(&parsed.nodes);
    mapper.flush_tool();
}

/// Extracts all tagged spans into a pre-order `tag -> UTF-8 text` map.
pub(super) fn extract_tags(
    ast: &AstArena,
    nodes: &[AstId],
    input: &[u8],
) -> Result<BTreeMap<String, String>> {
    fn walk(
        ast: &AstArena,
        nodes: &[AstId],
        input: &[u8],
        out: &mut BTreeMap<String, String>,
    ) -> Result<()> {
        for &id in nodes {
            if let Some(node) = ast.get(id) {
                if !node.tag.is_empty() {
                    let slice = input
                        .get(node.start..node.end)
                        .ok_or_else(|| invalid("chat tag extent invalid"))?;
                    let text = String::from_utf8(slice.to_vec())?;
                    out.insert(node.tag.clone(), text);
                }
                walk(ast, &node.children, input, out)?;
            }
        }
        Ok(())
    }
    let mut out = BTreeMap::new();
    walk(ast, nodes, input, &mut out)?;
    Ok(out)
}

/// Formats `ParseResult` as `PascalCase` wire JSON (`{"Type","Start","End"}`).
pub(super) fn wire_parse_result(result: &ParseResult) -> Value {
    json!({
        "Type": result.result_type.as_u8(),
        "Start": result.start,
        "End": result.end,
    })
}

/// Formats `ChatToolCall` as `PascalCase` wire JSON (`{"Name","Arguments","Id"}`).
pub(super) fn wire_chat_tool_call(call: &ChatToolCall) -> Value {
    json!({
        "Name": call.name,
        "Arguments": call.arguments,
        "Id": call.id,
    })
}

/// Formats `ChatMessage` as `PascalCase` wire JSON (`{"Role","Content","ReasoningContent","ToolCalls"}`).
pub(super) fn wire_chat_message(msg: &ChatMessage) -> Value {
    let calls: Vec<Value> = msg.tool_calls.iter().map(wire_chat_tool_call).collect();
    json!({
        "Role": msg.role,
        "Content": msg.content,
        "ReasoningContent": msg.reasoning_content,
        "ToolCalls": calls,
    })
}

// ---------------------------------------------------------------------------
// Chat PEG builder & schema-aware Arena
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct SchemaSpec {
    name: String,
    schema: JsonVal,
    raw: bool,
}

/// Options controlling `ChatPegBuilder::standard_json_tools`.
#[expect(
    clippy::struct_excessive_bools,
    reason = "Mirrors upstream standard_json_tools boolean parameters"
)]
pub(super) struct JsonToolsOptions<'a> {
    /// Prefix string introducing the tool-call section.
    pub(super) section_start: &'a str,
    /// Suffix string closing the tool-call section.
    pub(super) section_end: &'a str,
    /// Parsed JSON array of tool definitions.
    pub(super) tools: &'a JsonVal,
    /// Whether multiple tool calls may be emitted.
    pub(super) parallel_tool_calls: bool,
    /// Whether at least one tool call is mandatory.
    pub(super) force_tool_calls: bool,
    /// Optional per-call prefix wrapper.
    pub(super) per_call_start: &'a str,
    /// Optional per-call suffix wrapper.
    pub(super) per_call_end: &'a str,
    /// Whether tool function name is the outer JSON object key.
    pub(super) fun_name_is_key: bool,
    /// Whether calls are wrapped in a JSON array `[...]`.
    pub(super) tools_array_wrapped: bool,
    /// Optional JSON key/path for function name (defaults to `"name"` when empty).
    pub(super) name_field: &'a str,
    /// Optional JSON key/path for arguments (defaults to `"arguments"` when empty).
    pub(super) args_field: &'a str,
    /// Optional JSON key for string tool call id (omitted when empty).
    pub(super) id_field: &'a str,
    /// Optional JSON key for generated integer/string tool call id (omitted when empty).
    pub(super) gen_id_field: &'a str,
    /// Explicit parameter ordering list for tool schemas.
    pub(super) parameter_order: &'a [String],
    /// Optional trailing boolean flag from upstream signature.
    pub(super) allow_raw_arguments: bool,
}

struct JsonIdFields<'a> {
    pairs: Vec<(&'a str, ParserId)>,
    optional_keys: Vec<&'a str>,
}

/// Clean-room `common_chat_peg_builder` layered on [`ParserBuilder`].
pub(super) struct ChatPegBuilder {
    inner: ParserBuilder,
    schemas: BTreeMap<ParserId, SchemaSpec>,
    permute_cache: BTreeMap<String, BTreeMap<usize, ParserId>>,
}

impl ChatPegBuilder {
    /// Creates an empty chat PEG builder.
    pub(super) fn new() -> Self {
        Self {
            inner: ParserBuilder::new(),
            schemas: BTreeMap::new(),
            permute_cache: BTreeMap::new(),
        }
    }

    /// Sets the root parser of the arena.
    pub(super) const fn set_root(&mut self, id: ParserId) {
        self.inner.set_root(id);
    }

    /// Mutable access to the underlying [`ParserBuilder`].
    pub(super) const fn inner_mut(&mut self) -> &mut ParserBuilder {
        &mut self.inner
    }

    /// Wraps `child` as `reasoning-block`.
    pub(super) fn reasoning_block(&mut self, child: ParserId) -> ParserId {
        self.inner.tag(TAG_REASONING_BLOCK, child)
    }

    /// Wraps `child` as `reasoning`.
    pub(super) fn reasoning(&mut self, child: ParserId) -> ParserId {
        self.inner.tag(TAG_REASONING, child)
    }

    /// Wraps `child` as `content`.
    pub(super) fn content(&mut self, child: ParserId) -> ParserId {
        self.inner.tag(TAG_CONTENT, child)
    }

    /// Wraps `child` as `tool`.
    pub(super) fn tool(&mut self, child: ParserId) -> ParserId {
        self.inner.tag(TAG_TOOL, child)
    }

    /// Atomically wraps `child` as `tool-open`.
    pub(super) fn tool_open(&mut self, child: ParserId) -> ParserId {
        let tagged = self.inner.tag(TAG_TOOL_OPEN, child);
        self.inner.atomic(tagged)
    }

    /// Atomically wraps `child` as `tool-close`.
    pub(super) fn tool_close(&mut self, child: ParserId) -> ParserId {
        let tagged = self.inner.tag(TAG_TOOL_CLOSE, child);
        self.inner.atomic(tagged)
    }

    /// Atomically wraps `child` as `tool-id`.
    pub(super) fn tool_id(&mut self, child: ParserId) -> ParserId {
        let tagged = self.inner.tag(TAG_TOOL_ID, child);
        self.inner.atomic(tagged)
    }

    /// Atomically wraps `child` as `tool-name`.
    pub(super) fn tool_name(&mut self, child: ParserId) -> ParserId {
        let tagged = self.inner.tag(TAG_TOOL_NAME, child);
        self.inner.atomic(tagged)
    }

    /// Wraps `child` as `tool-args`.
    pub(super) fn tool_args(&mut self, child: ParserId) -> ParserId {
        self.inner.tag(TAG_TOOL_ARGS, child)
    }

    /// Wraps `child` as `tool-arg`.
    pub(super) fn tool_arg(&mut self, child: ParserId) -> ParserId {
        self.inner.tag(TAG_TOOL_ARG, child)
    }

    /// Atomically wraps `child` as `tool-arg-open`.
    pub(super) fn tool_arg_open(&mut self, child: ParserId) -> ParserId {
        let tagged = self.inner.tag(TAG_TOOL_ARG_OPEN, child);
        self.inner.atomic(tagged)
    }

    /// Atomically wraps `child` as `tool-arg-close`.
    pub(super) fn tool_arg_close(&mut self, child: ParserId) -> ParserId {
        let tagged = self.inner.tag(TAG_TOOL_ARG_CLOSE, child);
        self.inner.atomic(tagged)
    }

    /// Atomically wraps `child` as `tool-arg-name`.
    pub(super) fn tool_arg_name(&mut self, child: ParserId) -> ParserId {
        let tagged = self.inner.tag(TAG_TOOL_ARG_NAME, child);
        self.inner.atomic(tagged)
    }

    /// Wraps `child` as `tool-arg-string-value`.
    pub(super) fn tool_arg_string_value(&mut self, child: ParserId) -> ParserId {
        self.inner.tag(TAG_TOOL_ARG_STRING_VALUE, child)
    }

    /// Wraps `child` as `tool-arg-json-value`.
    pub(super) fn tool_arg_json_value(&mut self, child: ParserId) -> ParserId {
        self.inner.tag(TAG_TOOL_ARG_JSON_VALUE, child)
    }

    /// Defines a trigger rule (`rule(name, child, true)`).
    pub(super) fn trigger_rule(&mut self, name: &str, child: ParserId) -> ParserId {
        self.inner.rule(name, child, true)
    }

    /// Associates a JSON Schema with `child` for GBNF emission while parsing with `child`.
    pub(super) fn schema(
        &mut self,
        child: ParserId,
        name: String,
        schema: JsonVal,
        raw: bool,
    ) -> ParserId {
        let wrapper = self.inner.sequence(vec![child]);
        self.schemas
            .insert(wrapper, SchemaSpec { name, schema, raw });
        wrapper
    }

    /// Builds a content parser for a JSON Schema allowing optional markdown fences and whitespace.
    pub(super) fn schema_content_parser(&mut self, schema: &JsonVal) -> Result<ParserId> {
        let json_p = self.inner.json()?;
        let schema_p = self.schema(json_p, "response".to_owned(), schema.clone(), false);
        let content_schema = self.content(schema_p);
        let fence_open = self.inner.literal("```");
        let fence_lang = self.inner.literal("json");
        let opt_lang = self.inner.optional(fence_lang);
        let fence_close = self.inner.literal("```");
        let sp_lead = self.inner.space();
        let sp_open = self.inner.space();
        let sp_close = self.inner.space();
        let sp_tail_fenced = self.inner.space();
        let sp_tail_raw = self.inner.space();
        let fenced = self.inner.sequence(vec![
            fence_open,
            opt_lang,
            sp_open,
            content_schema,
            sp_close,
            fence_close,
            sp_tail_fenced,
        ]);
        let raw = self.inner.sequence(vec![content_schema, sp_tail_raw]);
        let body = self.inner.choice(vec![fenced, raw]);
        Ok(self.inner.sequence(vec![sp_lead, body]))
    }

    /// Generates an order-independent permutation parser/grammar over `children`.
    pub(super) fn permute(&mut self, prefix: &str, children: &[ParserId]) -> ParserId {
        match children.len() {
            0 => self.inner.eps(),
            1 => children[0],
            n if n > MAX_PERMUTE_EXACT => self.inner.sequence(children.to_vec()),
            n => {
                let full_mask = (1_usize << n) - 1;
                for mask in 1..full_mask {
                    let _ = self.build_permute_mask(prefix, children, mask);
                }
                self.build_permute_mask(prefix, children, full_mask)
            }
        }
    }

    fn build_permute_mask(&mut self, prefix: &str, children: &[ParserId], mask: usize) -> ParserId {
        if let Some(&cached) = self
            .permute_cache
            .get(prefix)
            .and_then(|by_mask| by_mask.get(&mask))
        {
            return cached;
        }
        let body = if mask.is_power_of_two() {
            let idx = usize::try_from(mask.trailing_zeros()).unwrap_or(0);
            children[idx]
        } else {
            let mut alts = Vec::with_capacity(usize::try_from(mask.count_ones()).unwrap_or(0));
            for (idx, &item) in children.iter().enumerate() {
                let bit = 1_usize << idx;
                if (mask & bit) != 0 {
                    let rem = mask & !bit;
                    let alt = if rem == 0 {
                        item
                    } else {
                        let rest = self.build_permute_mask(prefix, children, rem);
                        self.inner.add(item, rest)
                    };
                    alts.push(alt);
                }
            }
            self.inner.choice(alts)
        };
        let rule_name = format!("{prefix}-{mask}");
        let id = self.inner.rule(&rule_name, body, false);
        if let Some(by_mask) = self.permute_cache.get_mut(prefix) {
            by_mask.insert(mask, id);
        } else {
            self.permute_cache
                .insert(prefix.to_owned(), BTreeMap::from([(mask, id)]));
        }
        id
    }

    /// Builds a JSON tool-calling parser from `opts`.
    pub(super) fn standard_json_tools(&mut self, opts: &JsonToolsOptions<'_>) -> Result<ParserId> {
        let mut parsed_tools = extract_tool_defs(opts.tools, opts.parameter_order);
        if parsed_tools.is_empty() {
            return Ok(if opts.force_tool_calls {
                self.inner.choice(Vec::new())
            } else {
                self.inner.eps()
            });
        }
        // Sort longest tool name first so shorter prefixes do not match eagerly during streaming.
        parsed_tools.sort_by_key(|a| std::cmp::Reverse(a.0.len()));

        let name_key = if opts.name_field.is_empty() {
            "name"
        } else {
            opts.name_field
        };
        let args_key = if opts.args_field.is_empty() {
            "arguments"
        } else {
            opts.args_field
        };

        let mut tool_parsers = Vec::with_capacity(parsed_tools.len());
        for (name, params_schema) in &parsed_tools {
            let call_parser =
                self.build_single_json_tool(name, params_schema, name_key, args_key, opts)?;
            tool_parsers.push(call_parser);
        }

        let array_delimited =
            opts.section_start.ends_with('[') && opts.section_end.starts_with(']');
        let single_call = self.inner.choice(tool_parsers);
        let calls_seq = if opts.tools_array_wrapped {
            let lbrack = self.inner.literal("[");
            let rbrack = self.inner.literal("]");
            let inner_calls = self.parallel_json_calls(single_call, opts.parallel_tool_calls, true);
            let left = self.inner.space_add(lbrack, inner_calls);
            self.inner.space_add(left, rbrack)
        } else {
            self.parallel_json_calls(single_call, opts.parallel_tool_calls, array_delimited)
        };

        let section = if opts.parallel_tool_calls
            && !opts.tools_array_wrapped
            && !array_delimited
            && opts.per_call_start.is_empty()
            && opts.per_call_end.is_empty()
            && (!opts.section_start.is_empty() || !opts.section_end.is_empty())
        {
            let one_section = wrap_section(
                &mut self.inner,
                opts.section_start,
                calls_seq,
                opts.section_end,
            );
            let single_wrapped = wrap_section(
                &mut self.inner,
                opts.section_start,
                single_call,
                opts.section_end,
            );
            let eps = self.inner.eps();
            let rep_step = self.inner.space_add(eps, single_wrapped);
            let rep_tail = self.inner.zero_or_more(rep_step);
            let repeated_sections = self.inner.add(single_wrapped, rep_tail);
            self.inner.or(repeated_sections, one_section)
        } else {
            wrap_section(
                &mut self.inner,
                opts.section_start,
                calls_seq,
                opts.section_end,
            )
        };
        let triggered = self.trigger_rule("tool-call", section);
        Ok(if opts.force_tool_calls {
            triggered
        } else {
            self.inner.optional(triggered)
        })
    }

    fn parallel_json_calls(
        &mut self,
        single_call: ParserId,
        parallel: bool,
        comma_separated: bool,
    ) -> ParserId {
        if !parallel {
            return single_call;
        }
        let next_call = if comma_separated {
            let comma = self.inner.literal(",");
            self.inner.space_add(comma, single_call)
        } else {
            single_call
        };
        let eps = self.inner.eps();
        let next_item = self.inner.space_add(eps, next_call);
        let tail = self.inner.zero_or_more(next_item);
        self.inner.add(single_call, tail)
    }

    fn json_kv(&mut self, key: &str, value_p: ParserId) -> ParserId {
        let k = self.inner.literal(json_quoted(key));
        let colon = self.inner.literal(":");
        let k_colon = self.inner.space_add(k, colon);
        self.inner.space_add(k_colon, value_p)
    }

    fn quoted_tool_name(&mut self, name: &str) -> ParserId {
        let q1 = self.inner.literal("\"");
        let name_lit = self.inner.literal(name);
        let tagged_name = self.tool_name(name_lit);
        let q2 = self.inner.literal("\"");
        self.inner.sequence(vec![q1, tagged_name, q2])
    }

    fn join_comma_seq(&mut self, comma_sep: ParserId, seq_items: &[ParserId]) -> ParserId {
        let mut acc = seq_items[0];
        for &next in &seq_items[1..] {
            let with_sep = self.inner.add(acc, comma_sep);
            acc = self.inner.add(with_sep, next);
        }
        acc
    }

    fn json_object_wrap(
        &mut self,
        fields: &[(&str, ParserId)],
        order: &[String],
        optional_keys: &[&str],
    ) -> ParserId {
        let comma_lit = self.inner.literal(",");
        let sp_before = self.inner.space();
        let with_comma = self.inner.add(sp_before, comma_lit);
        let sp_after = self.inner.space();
        let comma_sep = self.inner.add(with_comma, sp_after);

        let mut ordered: Vec<(&str, ParserId)> = Vec::with_capacity(fields.len());
        for wanted in order {
            if let Some(&pair) = fields.iter().find(|(k, _)| *k == wanted.as_str()) {
                ordered.push(pair);
            }
        }
        for &pair in fields {
            if !ordered.iter().any(|(k, _)| *k == pair.0) {
                ordered.push(pair);
            }
        }

        let required_only: Vec<ParserId> = ordered
            .iter()
            .filter(|(k, _)| !optional_keys.contains(k))
            .map(|(_, p)| *p)
            .collect();
        let all_fields: Vec<ParserId> = ordered.iter().map(|(_, p)| *p).collect();

        let mut variants = Vec::new();
        if !all_fields.is_empty() {
            variants.push(self.join_comma_seq(comma_sep, &all_fields));
        }
        if all_fields.len() == 2 {
            variants.push(self.join_comma_seq(comma_sep, &[all_fields[1], all_fields[0]]));
        } else if all_fields.len() == 3 {
            for first_idx in 0..3_usize {
                for second_idx in 0..3_usize {
                    if second_idx == first_idx {
                        continue;
                    }
                    let third_idx = 3 - first_idx - second_idx;
                    variants.push(self.join_comma_seq(
                        comma_sep,
                        &[
                            all_fields[first_idx],
                            all_fields[second_idx],
                            all_fields[third_idx],
                        ],
                    ));
                }
            }
        }
        if !required_only.is_empty() && required_only.len() < all_fields.len() {
            variants.push(self.join_comma_seq(comma_sep, &required_only));
            if required_only.len() == 2 {
                variants
                    .push(self.join_comma_seq(comma_sep, &[required_only[1], required_only[0]]));
            }
        }
        if variants.len() == 1 {
            variants[0]
        } else {
            self.inner.choice(variants)
        }
    }

    fn build_id_fields<'a>(&mut self, opts: &JsonToolsOptions<'a>) -> Result<JsonIdFields<'a>> {
        let mut pairs = Vec::new();
        let mut optional_keys = Vec::new();
        if !opts.id_field.is_empty() {
            let q1 = self.inner.literal("\"");
            let sc = self.inner.string_content(b'"');
            let tagged = self.tool_id(sc);
            let q2 = self.inner.literal("\"");
            let val = self.inner.sequence(vec![q1, tagged, q2]);
            pairs.push((opts.id_field, self.json_kv(opts.id_field, val)));
            optional_keys.push(opts.id_field);
        }
        if !opts.gen_id_field.is_empty() {
            let q1 = self.inner.literal("\"");
            let sc = self.inner.string_content(b'"');
            let tagged = self.tool_id(sc);
            let q2 = self.inner.literal("\"");
            let str_id = self.inner.sequence(vec![q1, tagged, q2]);
            let num_raw = self.inner.chars("[0-9]", 1, -1)?;
            let num_id = self.tool_id(num_raw);
            let val = self.inner.or(str_id, num_id);
            pairs.push((opts.gen_id_field, self.json_kv(opts.gen_id_field, val)));
            optional_keys.push(opts.gen_id_field);
        }
        Ok(JsonIdFields {
            pairs,
            optional_keys,
        })
    }

    fn build_single_json_tool(
        &mut self,
        name: &str,
        params_schema: &JsonVal,
        name_key: &str,
        args_key: &str,
        opts: &JsonToolsOptions<'_>,
    ) -> Result<ParserId> {
        let json_p = self.inner.json()?;
        let schema_name = format!("tool-{name}-args");
        let schema_p = self.schema(json_p, schema_name, params_schema.clone(), false);
        let args_val = if opts.allow_raw_arguments {
            let str_p = self.inner.double_quoted_string();
            let choice = self.inner.or(schema_p, str_p);
            self.tool_args(choice)
        } else {
            self.tool_args(schema_p)
        };

        let lbrace = self.inner.literal("{");
        let open_inner = if opts.per_call_start.is_empty() {
            lbrace
        } else {
            let p_start = self.inner.literal(opts.per_call_start);
            self.inner.space_add(p_start, lbrace)
        };
        let open = self.tool_open(open_inner);

        let rbrace = self.inner.literal("}");
        let close_inner = if opts.per_call_end.is_empty() {
            rbrace
        } else {
            let p_end = self.inner.literal(opts.per_call_end);
            self.inner.space_add(rbrace, p_end)
        };
        let close = self.tool_close(close_inner);

        let JsonIdFields {
            pairs: id_pairs,
            optional_keys: mut opt_keys,
        } = self.build_id_fields(opts)?;
        let empty_args = schema_allows_empty_object(params_schema);
        let name_val = self.quoted_tool_name(name);

        let body = if opts.fun_name_is_key {
            let colon = self.inner.literal(":");
            let left = self.inner.space_add(name_val, colon);
            let inner_val = if opts.args_field.is_empty() && id_pairs.is_empty() {
                args_val
            } else {
                let mut inner_fields = id_pairs;
                let args_kv = self.json_kv(args_key, args_val);
                inner_fields.push((args_key, args_kv));
                if empty_args {
                    opt_keys.push(args_key);
                }
                let inner_body =
                    self.json_object_wrap(&inner_fields, opts.parameter_order, &opt_keys);
                let lb = self.inner.literal("{");
                let rb = self.inner.literal("}");
                let l_inner = self.inner.space_add(lb, inner_body);
                self.inner.space_add(l_inner, rb)
            };
            self.inner.space_add(left, inner_val)
        } else if let (Some((np, nk)), Some((ap, ak))) =
            (name_key.split_once('.'), args_key.split_once('.'))
            && np == ap
        {
            let name_kv = self.json_kv(nk, name_val);
            let args_kv = self.json_kv(ak, args_val);
            let nested_opt: Vec<&str> = if empty_args { vec![ak] } else { Vec::new() };
            let nested_inner =
                self.json_object_wrap(&[(nk, name_kv), (ak, args_kv)], &[], &nested_opt);
            let lb = self.inner.literal("{");
            let rb = self.inner.literal("}");
            let l_nested = self.inner.space_add(lb, nested_inner);
            let nested_obj = self.inner.space_add(l_nested, rb);
            let fn_kv = self.json_kv(np, nested_obj);
            let mut outer_fields = id_pairs;
            outer_fields.push((np, fn_kv));
            self.json_object_wrap(&outer_fields, opts.parameter_order, &opt_keys)
        } else {
            let name_kv = self.json_kv(name_key, name_val);
            let args_kv = self.json_kv(args_key, args_val);
            if empty_args {
                opt_keys.push(args_key);
            }
            let mut outer_fields = id_pairs;
            outer_fields.push((name_key, name_kv));
            outer_fields.push((args_key, args_kv));
            self.json_object_wrap(&outer_fields, opts.parameter_order, &opt_keys)
        };

        let left = self.inner.space_add(open, body);
        let full = self.inner.space_add(left, close);
        Ok(self.tool(full))
    }

    /// Builds a constructed (XML / tag-delimited) tool-calling parser from `markers` and `tools`.
    pub(super) fn standard_constructed_tools(
        &mut self,
        markers: &BTreeMap<String, String>,
        tools: &JsonVal,
        parallel_tool_calls: bool,
        force_tool_calls: bool,
    ) -> Result<ParserId> {
        self.standard_constructed_tools_with_ws(
            markers,
            tools,
            parallel_tool_calls,
            force_tool_calls,
            false,
        )
    }

    /// Builds a constructed tool-calling parser with optional inter-tag whitespace tolerance.
    pub(super) fn standard_constructed_tools_with_ws(
        &mut self,
        markers: &BTreeMap<String, String>,
        tools: &JsonVal,
        parallel_tool_calls: bool,
        force_tool_calls: bool,
        tolerate_intertag_whitespace: bool,
    ) -> Result<ParserId> {
        let mut parsed_tools = extract_tool_defs(tools, &[]);
        if parsed_tools.is_empty() {
            return Ok(if force_tool_calls {
                self.inner.choice(Vec::new())
            } else {
                self.inner.eps()
            });
        }
        parsed_tools.sort_by_key(|a| std::cmp::Reverse(a.0.len()));

        let mut m = ConstructedMarkers::from_map(markers);
        m.tolerate_intertag_whitespace = tolerate_intertag_whitespace;
        let mut tool_parsers = Vec::with_capacity(parsed_tools.len());
        for (name, params_schema) in &parsed_tools {
            let p = self.build_single_constructed_tool(name, params_schema, &m)?;
            tool_parsers.push(p);
        }

        let single_call = self.inner.choice(tool_parsers);
        let calls_seq = if parallel_tool_calls {
            let separator = if m.tool_separator.is_empty() {
                self.inner.space()
            } else {
                let s_lit = self.inner.literal(&m.tool_separator);
                let sp1 = self.inner.space();
                let sp2 = self.inner.space();
                self.inner.sequence(vec![sp1, s_lit, sp2])
            };
            let step = self.inner.add(separator, single_call);
            let tail = self.inner.zero_or_more(step);
            self.inner.add(single_call, tail)
        } else {
            single_call
        };

        let section = wrap_section(&mut self.inner, &m.section_start, calls_seq, &m.section_end);
        let section_with_ws = if m.section_end.is_empty() && m.tool_end.is_empty() {
            section
        } else {
            let sp = self.inner.space();
            self.inner.add(section, sp)
        };
        let triggered = self.trigger_rule("tool-call", section_with_ws);
        Ok(if force_tool_calls {
            triggered
        } else {
            self.inner.optional(triggered)
        })
    }

    fn build_single_constructed_tool(
        &mut self,
        name: &str,
        params_schema: &JsonVal,
        m: &ConstructedMarkers,
    ) -> Result<ParserId> {
        let open = match (m.tool_start.is_empty(), m.func_name_prefix.is_empty()) {
            (true, true) => self.inner.eps(),
            (false, true) => {
                let l = self.inner.literal(&m.tool_start);
                self.tool_open(l)
            }
            (true, false) => {
                let l = self.inner.literal(&m.func_name_prefix);
                self.tool_open(l)
            }
            (false, false) => {
                let l1 = self.inner.literal(&m.tool_start);
                let first = if m.tool_start.trim() == "<tool_call>"
                    && m.func_name_prefix.trim() == "<function="
                {
                    self.inner.optional(l1)
                } else {
                    l1
                };
                let l2 = self.inner.literal(&m.func_name_prefix);
                let seq = self.inner.space_add(first, l2);
                self.tool_open(seq)
            }
        };

        let name_lit = self.inner.literal(name);
        let name_p = self.tool_name(name_lit);
        let name_suffix = if m.func_name_suffix.is_empty() {
            self.inner.eps()
        } else {
            self.inner.literal(&m.func_name_suffix)
        };
        let args_sep = if m.args_separator.is_empty() {
            self.inner.space()
        } else {
            let sep_lit = self.inner.literal(&m.args_separator);
            let sp1 = self.inner.space();
            let sp2 = self.inner.space();
            self.inner.sequence(vec![sp1, sep_lit, sp2])
        };

        let args_p = self.build_constructed_args(name, params_schema, m)?;
        let close = match (m.func_close.is_empty(), m.tool_end.is_empty()) {
            (true, true) => self.inner.eps(),
            (false, true) => {
                let l = self.inner.literal(&m.func_close);
                self.tool_close(l)
            }
            (true, false) => {
                let l = self.inner.literal(&m.tool_end);
                self.tool_close(l)
            }
            (false, false) => {
                let l1 = self.inner.literal(&m.func_close);
                let l2 = self.inner.literal(&m.tool_end);
                let seq = self.inner.space_add(l1, l2);
                self.tool_close(seq)
            }
        };

        let sp_close = self.inner.space();
        let full = self.inner.sequence(vec![
            open,
            name_p,
            name_suffix,
            args_sep,
            args_p,
            sp_close,
            close,
        ]);
        Ok(self.tool(full))
    }

    fn build_constructed_args(
        &mut self,
        tool_name: &str,
        params_schema: &JsonVal,
        m: &ConstructedMarkers,
    ) -> Result<ParserId> {
        let props = params_schema.get("properties").object();
        if props.is_empty() {
            let empty = self.inner.eps();
            return Ok(self.tool_args(empty));
        }
        let required: BTreeSet<&str> = params_schema
            .get("required")
            .array()
            .iter()
            .filter_map(JsonVal::text)
            .collect();

        let schema_ctx = SchemaGrammarContext::new();
        let mut arg_parsers = Vec::with_capacity(props.len());
        let mut any_arg_parsers = Vec::with_capacity(props.len());

        for (arg_name, arg_schema) in props {
            let arg_p =
                self.build_single_constructed_arg(tool_name, arg_name, arg_schema, m, &schema_ctx)?;
            any_arg_parsers.push(arg_p);
            if required.contains(arg_name.as_str()) {
                arg_parsers.push(arg_p);
            } else {
                arg_parsers.push(self.inner.optional(arg_p));
            }
        }

        let body = if props.len() <= MAX_PERMUTE_EXACT && m.arg_separator.is_empty() {
            let spaced_args: Vec<ParserId> = arg_parsers
                .into_iter()
                .map(|p| {
                    let sp = self.inner.space();
                    self.inner.add(p, sp)
                })
                .collect();
            self.permute(&format!("tool-{tool_name}-args"), &spaced_args)
        } else {
            let single = self.inner.choice(any_arg_parsers);
            let sep = if m.arg_separator.is_empty() {
                self.inner.space()
            } else {
                let lit = self.inner.literal(&m.arg_separator);
                let sp1 = self.inner.space();
                let sp2 = self.inner.space();
                self.inner.sequence(vec![sp1, lit, sp2])
            };
            let rest = self.inner.add(sep, single);
            let tail = self.inner.zero_or_more(rest);
            let one_or_more = self.inner.add(single, tail);
            if required.is_empty() {
                self.inner.optional(one_or_more)
            } else {
                one_or_more
            }
        };

        let wrapped = wrap_section(&mut self.inner, &m.args_start, body, &m.args_end);
        Ok(self.tool_args(wrapped))
    }

    fn build_single_constructed_arg(
        &mut self,
        tool_name: &str,
        arg_name: &str,
        arg_schema: &JsonVal,
        m: &ConstructedMarkers,
        schema_ctx: &SchemaGrammarContext,
    ) -> Result<ParserId> {
        let prefix = self.inner.literal(&m.arg_name_prefix);
        let name_lit = self.inner.literal(arg_name);
        let arg_name_p = self.tool_arg_name(name_lit);
        let suffix = self.inner.literal(&m.arg_name_suffix);
        let arg_open_seq = self.inner.sequence(vec![prefix, arg_name_p, suffix]);
        let arg_open = self.tool_arg_open(arg_open_seq);

        let val_prefix = if m.arg_value_prefix.is_empty() {
            self.inner.eps()
        } else if m.tolerate_intertag_whitespace {
            let sp = self.inner.space();
            let lit = self.inner.literal(&m.arg_value_prefix);
            self.inner.add(sp, lit)
        } else {
            self.inner.literal(&m.arg_value_prefix)
        };

        let close_delim = if !m.arg_value_suffix.is_empty() {
            m.arg_value_suffix.clone()
        } else if !m.args_end.is_empty() {
            m.args_end.clone()
        } else {
            format!("{}{}", m.func_close, m.tool_end)
        };

        let val_p = if schema_ctx.resolves_to_string(arg_schema) {
            let until_p = if close_delim.is_empty() {
                self.inner.rest()
            } else {
                self.inner.until(&close_delim)
            };
            let tagged = self.tool_arg_string_value(until_p);
            self.schema(
                tagged,
                format!("tool-{tool_name}-arg-{arg_name}"),
                arg_schema.clone(),
                true,
            )
        } else {
            let json_p = self.inner.json()?;
            let bounded = if close_delim.is_empty() {
                json_p
            } else {
                self.inner.ac(json_p, vec![close_delim])
            };
            let tagged = self.tool_arg_json_value(bounded);
            self.schema(
                tagged,
                format!("tool-{tool_name}-arg-{arg_name}"),
                arg_schema.clone(),
                false,
            )
        };

        let arg_close = if m.arg_value_suffix.is_empty() {
            self.inner.eps()
        } else {
            let close_lit = self.inner.literal(&m.arg_value_suffix);
            self.tool_arg_close(close_lit)
        };

        let seq = self
            .inner
            .sequence(vec![arg_open, val_prefix, val_p, arg_close]);
        Ok(self.tool_arg(seq))
    }

    /// Builds an immutable [`ChatArena`].
    pub(super) fn build(self) -> Result<ChatArena> {
        let arena = self.inner.build()?;
        Ok(ChatArena {
            arena,
            schemas: self.schemas,
        })
    }
}

/// Returns whether `schema` permits string values directly or within unions.
pub(super) fn schema_resolves_to_string(schema: &JsonVal) -> bool {
    SchemaGrammarContext::new().resolves_to_string(schema)
}

struct ConstructedMarkers {
    section_start: String,
    section_end: String,
    tool_start: String,
    tool_end: String,
    tool_separator: String,
    func_name_prefix: String,
    func_name_suffix: String,
    args_separator: String,
    func_close: String,
    args_start: String,
    args_end: String,
    arg_name_prefix: String,
    arg_name_suffix: String,
    arg_value_prefix: String,
    arg_value_suffix: String,
    arg_separator: String,
    tolerate_intertag_whitespace: bool,
}

impl ConstructedMarkers {
    fn is_known_key(key: &str) -> bool {
        matches!(
            key,
            "section_start"
                | "section_end"
                | "tool_call_start_marker"
                | "tool_call_end_marker"
                | "tool_call_separator"
                | "tool_start"
                | "per_call_start"
                | "call_start"
                | "tool_end"
                | "per_call_end"
                | "call_end"
                | "tool_separator"
                | "call_separator"
                | "function_opener"
                | "function_closer"
                | "function_name_suffix"
                | "func_call_start_marker"
                | "func_call_end_marker"
                | "func_name_prefix"
                | "function_name_prefix"
                | "name_prefix"
                | "func_name_suffix"
                | "name_suffix"
                | "args_separator"
                | "func_args_sep"
                | "func_close"
                | "function_close"
                | "close"
                | "args_start"
                | "arg_start"
                | "args_end"
                | "arg_end"
                | "arg_start_marker"
                | "arg_end_marker"
                | "parameter_key_prefix"
                | "parameter_key_suffix"
                | "parameter_value_prefix"
                | "parameter_closer"
                | "parameter_separator"
                | "arg_name_prefix"
                | "arg_name_suffix"
                | "arg_value_prefix"
                | "arg_value_suffix"
                | "arg_name_start_marker"
                | "arg_name_end_marker"
                | "arg_value_start_marker"
                | "arg_value_end_marker"
                | "arg_separator"
        )
    }

    fn from_map(map: &BTreeMap<String, String>) -> Self {
        let get = |keys: &[&str]| -> String {
            for &key in keys {
                if let Some(val) = map.get(key) {
                    return val.clone();
                }
            }
            String::new()
        };
        let func_name_prefix = get(&[
            "function_opener",
            "func_call_start_marker",
            "func_name_prefix",
            "function_name_prefix",
            "name_prefix",
        ]);
        let mut func_name_suffix =
            get(&["func_name_suffix", "function_name_suffix", "name_suffix"]);
        if func_name_suffix.is_empty()
            && func_name_prefix.contains('<')
            && !func_name_prefix.ends_with('>')
        {
            func_name_suffix = ">".to_owned();
        }
        Self {
            section_start: get(&["section_start"]),
            section_end: get(&["section_end"]),
            tool_start: get(&[
                "tool_call_start_marker",
                "tool_start",
                "per_call_start",
                "call_start",
            ]),
            tool_end: get(&[
                "tool_call_end_marker",
                "tool_end",
                "per_call_end",
                "call_end",
            ]),
            tool_separator: get(&["tool_call_separator", "tool_separator", "call_separator"]),
            func_name_prefix,
            func_name_suffix,
            args_separator: get(&["args_separator", "func_args_sep"]),
            func_close: get(&[
                "function_closer",
                "func_call_end_marker",
                "func_close",
                "function_close",
                "close",
            ]),
            args_start: get(&["args_start", "arg_start"]),
            args_end: get(&["args_end", "arg_end"]),
            arg_name_prefix: get(&[
                "parameter_key_prefix",
                "arg_start_marker",
                "arg_name_start_marker",
                "arg_name_prefix",
            ]),
            arg_name_suffix: get(&[
                "parameter_key_suffix",
                "arg_name_end_marker",
                "arg_name_suffix",
            ]),
            arg_value_prefix: get(&[
                "parameter_value_prefix",
                "arg_value_start_marker",
                "arg_value_prefix",
            ]),
            arg_value_suffix: get(&[
                "parameter_closer",
                "arg_end_marker",
                "arg_value_end_marker",
                "arg_value_suffix",
            ]),
            arg_separator: get(&["parameter_separator", "arg_separator"]),
            tolerate_intertag_whitespace: false,
        }
    }
}

fn wrap_section(builder: &mut ParserBuilder, start: &str, inner: ParserId, end: &str) -> ParserId {
    if start.is_empty() && end.is_empty() {
        return inner;
    }
    let mut seq = Vec::new();
    if !start.is_empty() {
        seq.push(builder.literal(start));
        seq.push(builder.space());
    }
    seq.push(inner);
    if !end.is_empty() {
        seq.push(builder.space());
        seq.push(builder.literal(end));
    }
    builder.sequence(seq)
}

fn json_quoted(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    push_json_escaped_quoted(&mut out, text);
    out
}

fn schema_allows_empty_object(schema: &JsonVal) -> bool {
    schema.get("required").array().is_empty()
}

fn reorder_properties(schema: &JsonVal, parameter_order: &[String]) -> JsonVal {
    if parameter_order.is_empty() || !schema.has("properties") {
        return schema.clone();
    }
    let JsonVal::Object(entries) = schema else {
        return schema.clone();
    };
    let mut updated = Vec::with_capacity(entries.len());
    for (k, v) in entries {
        if k == "properties"
            && let JsonVal::Object(props) = v
        {
            let mut reordered = Vec::with_capacity(props.len());
            for ordered_key in parameter_order {
                if let Some(pair) = props.iter().find(|(pk, _)| pk == ordered_key) {
                    reordered.push(pair.clone());
                }
            }
            for pair in props {
                if !parameter_order.iter().any(|ok| ok == &pair.0) {
                    reordered.push(pair.clone());
                }
            }
            updated.push((k.clone(), JsonVal::Object(reordered)));
        } else {
            updated.push((k.clone(), v.clone()));
        }
    }
    JsonVal::Object(updated)
}

fn extract_tool_defs(tools: &JsonVal, parameter_order: &[String]) -> Vec<(String, JsonVal)> {
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
            reorder_properties(fun.get("parameters"), parameter_order)
        } else {
            JsonVal::Object(vec![("type".to_owned(), JsonVal::Str("object".to_owned()))])
        };
        out.push((name.to_owned(), params));
    }
    out
}

fn is_unconstrained_string_schema(schema: &JsonVal) -> bool {
    let JsonVal::Object(entries) = schema else {
        return false;
    };
    let has_string_type = schema.get("type").is("string");
    let has_constraint = entries.iter().any(|(key, _)| {
        matches!(
            key.as_str(),
            "$ref"
                | "enum"
                | "const"
                | "pattern"
                | "format"
                | "minLength"
                | "maxLength"
                | "oneOf"
                | "anyOf"
                | "allOf"
        )
    });
    has_string_type && !has_constraint
}

/// Compiled immutable chat PEG arena retaining schema bindings for GBNF emission.
#[derive(Clone, Debug)]
pub(super) struct ChatArena {
    arena: Arena,
    schemas: BTreeMap<ParserId, SchemaSpec>,
}

impl ChatArena {
    /// Serializes the underlying PEG arena to compact JSON.
    pub(super) fn to_json_string(&self) -> Result<String> {
        self.arena.to_json_string()
    }

    /// Deserializes a runtime-only [`ChatArena`] from serialized PEG JSON.
    pub(super) fn from_json_str(json_str: &str) -> Result<Self> {
        let arena = Arena::from_json_str(json_str)?;
        Ok(Self {
            arena,
            schemas: BTreeMap::new(),
        })
    }

    /// Parses `ctx` from byte offset 0.
    pub(super) fn parse_with_context(&self, ctx: &mut ParseContext<'_>) -> Result<ParseResult> {
        self.arena.parse_with_context(ctx)
    }

    /// Searches across `ctx.input` for the first matching start offset.
    pub(super) fn parse_anywhere(&self, ctx: &mut ParseContext<'_>) -> Result<ParseResult> {
        self.arena.parse_anywhere(ctx)
    }

    /// Generates a GBNF grammar string resolving `ResolveRefs` schemas.
    pub(super) fn build_grammar(&self, lazy: bool, resolve_refs: &[JsonVal]) -> Result<String> {
        let mut schema_ctx = SchemaGrammarContext::new();
        for ref_schema in resolve_refs {
            schema_ctx.resolve_refs(ref_schema).map_err(invalid)?;
        }
        self.arena
            .build_grammar_with_hook(lazy, |id, existing_rules| {
                let Some(spec) = self.schemas.get(&id) else {
                    return Ok(None);
                };
                if spec.raw && is_unconstrained_string_schema(&spec.schema) {
                    return Ok(None);
                }
                schema_ctx.merge_rules(existing_rules);
                let root_symbol = schema_ctx
                    .add_schema(&spec.name, &spec.schema, spec.raw)
                    .map_err(invalid)?;
                for (name, body) in schema_ctx.rules() {
                    if !existing_rules.contains_key(name) {
                        existing_rules.insert(name.clone(), body.clone());
                    }
                }
                Ok(Some(root_symbol))
            })
    }
}

// ---------------------------------------------------------------------------
// Validation & declarative graph compilation for `chat-peg-trace`
// ---------------------------------------------------------------------------

fn require(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(invalid(message))
    }
}

fn text_field<'a>(value: &'a Value, name: &str, maximum: usize) -> Result<&'a str> {
    let text = field(value, name)?
        .as_str()
        .ok_or_else(|| invalid(format!("PEG {name} must be a string")))?;
    require(text.len() <= maximum, "PEG string exceeds bound")?;
    Ok(text)
}

fn text_value(value: &Value, maximum: usize) -> Result<&str> {
    let text = value
        .as_str()
        .ok_or_else(|| invalid("PEG argument must be a string"))?;
    require(text.len() <= maximum, "PEG string exceeds bound")?;
    Ok(text)
}

fn int_value(value: &Value, low: i64, high: i64) -> Result<i64> {
    let num = if let Some(u) = value.as_u64() {
        i64::try_from(u).map_err(|_| invalid("PEG integer exceeds bound"))?
    } else if let Some(i) = value.as_i64() {
        i
    } else {
        return Err(invalid("PEG argument must be an integer"));
    };
    require(num >= low && num <= high, "PEG integer outside bound")?;
    Ok(num)
}

fn bool_value(value: &Value) -> Result<bool> {
    value
        .as_bool()
        .ok_or_else(|| invalid("PEG argument must be boolean"))
}

fn parse_json_wrapper(value: &Value) -> Result<JsonVal> {
    exact_fields(value, &["Json"])?;
    let raw = text_field(value, "Json", MAX_SCHEMA_JSON_BYTES)?;
    JsonParser::parse_with_limit(raw, MAX_SCHEMA_JSON_BYTES).map_err(invalid)
}

fn validate_json_wrapper(value: &Value) -> Result<()> {
    let _ = parse_json_wrapper(value)?;
    Ok(())
}

fn validate_expr(
    expression: &Value,
    locals: &BTreeSet<String>,
    depth: usize,
    count: &mut usize,
) -> Result<()> {
    *count += 1;
    require(
        depth <= MAX_GRAPH_DEPTH && *count <= MAX_EXPR_NODES && expression.is_object(),
        "PEG expression depth/count exceeds bound",
    )?;
    if expression.get("Ref").is_some() {
        exact_fields(expression, &["Ref"])?;
        let name = text_field(expression, "Ref", MAX_LOCAL_NAME)?;
        require(locals.contains(name), "unknown PEG local Ref")?;
        return Ok(());
    }
    exact_fields(expression, &["Op", "Args"])?;
    let op = text_field(expression, "Op", MAX_LOCAL_NAME)?;
    let args = array(expression, "Args")?;
    require(
        args.len() <= MAX_ARGS,
        "PEG Args must be an array of at most 64 values",
    )?;
    validate_op(op, args, locals, depth, count)
}

fn validate_op(
    op: &str,
    args: &[Value],
    locals: &BTreeSet<String>,
    depth: usize,
    count: &mut usize,
) -> Result<()> {
    let arity = |low: usize, high: usize| -> Result<()> {
        require(
            args.len() >= low && args.len() <= high,
            "PEG argument count mismatch",
        )
    };
    match op {
        "eps"
        | "start"
        | "end"
        | "any"
        | "space"
        | "rest"
        | "double_quoted_string"
        | "single_quoted_string"
        | "json"
        | "python_value"
        | "marker"
        | "choice-empty" => arity(0, 0),
        "literal" | "ref" | "until" => {
            arity(1, 1)?;
            let _ = text_value(&args[0], MAX_STRING_BYTES)?;
            Ok(())
        }
        "chars" => {
            arity(1, 3)?;
            let _ = text_value(&args[0], MAX_STRING_BYTES)?;
            for (idx, arg) in args.iter().enumerate().skip(1) {
                let low = if idx == 1 { 0 } else { -1 };
                let _ = int_value(arg, low, 65_536)?;
            }
            Ok(())
        }
        "or-assign" => {
            arity(2, 2)?;
            validate_expr(&args[0], locals, depth + 1, count)?;
            validate_expr(&args[1], locals, depth + 1, count)
        }
        "sequence" | "choice" => {
            arity(1, 1)?;
            let items = args[0]
                .as_array()
                .ok_or_else(|| invalid("PEG parser list must be an array"))?;
            require(items.len() <= MAX_LIST_LEN, "PEG parser list exceeds bound")?;
            for item in items {
                validate_expr(item, locals, depth + 1, count)?;
            }
            Ok(())
        }
        "until_one_of" => {
            arity(1, 1)?;
            let items = args[0]
                .as_array()
                .ok_or_else(|| invalid("PEG delimiter list must be an array"))?;
            require(
                items.len() <= MAX_LIST_LEN,
                "PEG delimiter list exceeds bound",
            )?;
            for item in items {
                let _ = text_value(item, MAX_STRING_BYTES)?;
            }
            Ok(())
        }
        "add" | "or" | "space-add" => {
            arity(2, 2)?;
            require(
                !(args[0].is_string() && args[1].is_string()),
                "PEG operator requires a parser operand",
            )?;
            for arg in args {
                if arg.is_string() {
                    let _ = text_value(arg, MAX_STRING_BYTES)?;
                } else {
                    validate_expr(arg, locals, depth + 1, count)?;
                }
            }
            Ok(())
        }
        _ if is_unary_op(op) => {
            arity(1, 1)?;
            validate_expr(&args[0], locals, depth + 1, count)
        }
        _ => validate_op_chat_and_composite(op, args, locals, depth, count),
    }
}

fn is_unary_op(op: &str) -> bool {
    matches!(
        op,
        "one_or_more"
            | "zero_or_more"
            | "optional"
            | "atomic"
            | "peek"
            | "negate"
            | "reasoning_block"
            | "reasoning"
            | "content"
            | "tool"
            | "tool_open"
            | "tool_close"
            | "tool_id"
            | "tool_name"
            | "tool_args"
            | "tool_arg"
            | "tool_arg_open"
            | "tool_arg_close"
            | "tool_arg_name"
            | "tool_arg_string_value"
            | "tool_arg_json_value"
    )
}

fn validate_op_chat_and_composite(
    op: &str,
    args: &[Value],
    locals: &BTreeSet<String>,
    depth: usize,
    count: &mut usize,
) -> Result<()> {
    let arity = |low: usize, high: usize| -> Result<()> {
        require(
            args.len() >= low && args.len() <= high,
            "PEG argument count mismatch",
        )
    };
    match op {
        "rule" | "rule-lazy" => {
            arity(2, 3)?;
            let _ = text_value(&args[0], MAX_LOCAL_NAME)?;
            if op == "rule-lazy" {
                validate_graph(&args[1], locals.clone(), depth + 1, count)?;
            } else {
                validate_expr(&args[1], locals, depth + 1, count)?;
            }
            if args.len() == 3 {
                let _ = bool_value(&args[2])?;
            }
            Ok(())
        }
        "trigger_rule" => {
            arity(2, 2)?;
            let _ = text_value(&args[0], MAX_LOCAL_NAME)?;
            validate_expr(&args[1], locals, depth + 1, count)
        }
        "tag" | "json_member" => {
            arity(2, 2)?;
            let _ = text_value(&args[0], MAX_STRING_BYTES)?;
            validate_expr(&args[1], locals, depth + 1, count)
        }
        "gbnf" => {
            arity(2, 2)?;
            validate_expr(&args[0], locals, depth + 1, count)?;
            let _ = text_value(&args[1], MAX_STRING_BYTES)?;
            Ok(())
        }
        "ac" => validate_ac_op(args, locals, depth, count),
        "string_content" => {
            arity(1, 1)?;
            let _ = int_value(&args[0], 0, 255)?;
            Ok(())
        }
        "repeat" => {
            arity(2, 3)?;
            validate_expr(&args[0], locals, depth + 1, count)?;
            let _ = int_value(&args[1], 0, 65_536)?;
            if args.len() == 3 {
                let _ = int_value(&args[2], -1, 65_536)?;
            }
            Ok(())
        }
        "permute" => {
            arity(2, 2)?;
            let _ = text_value(&args[0], MAX_LOCAL_NAME)?;
            let items = args[1]
                .as_array()
                .ok_or_else(|| invalid("chat permute list must be array"))?;
            require(items.len() <= MAX_ARGS, "chat permute list exceeds bound")?;
            for item in items {
                validate_expr(item, locals, depth + 1, count)?;
            }
            Ok(())
        }
        "schema" => {
            arity(3, 4)?;
            validate_expr(&args[0], locals, depth + 1, count)?;
            let _ = text_value(&args[1], MAX_STRING_BYTES)?;
            validate_json_wrapper(&args[2])?;
            if args.len() == 4 {
                let _ = bool_value(&args[3])?;
            }
            Ok(())
        }
        "standard_json_tools" => validate_standard_json_tools_args(args),
        "standard_constructed_tools" => {
            arity(4, 4)?;
            validate_json_wrapper(&args[0])?;
            validate_json_wrapper(&args[1])?;
            require(
                args[2].is_boolean() && args[3].is_boolean(),
                "chat tool options must be boolean",
            )?;
            Ok(())
        }
        _ => Err(invalid("unknown PEG builder Op")),
    }
}

fn validate_ac_op(
    args: &[Value],
    locals: &BTreeSet<String>,
    depth: usize,
    count: &mut usize,
) -> Result<()> {
    require(args.len() == 2, "PEG argument count mismatch")?;
    validate_expr(&args[0], locals, depth + 1, count)?;
    if args[1].is_string() {
        let _ = text_value(&args[1], MAX_STRING_BYTES)?;
    } else {
        let items = args[1]
            .as_array()
            .ok_or_else(|| invalid("PEG AC delimiters must be string or array"))?;
        require(
            items.len() <= MAX_LIST_LEN,
            "PEG AC delimiter list exceeds bound",
        )?;
        for item in items {
            let _ = text_value(item, MAX_STRING_BYTES)?;
        }
    }
    Ok(())
}

fn validate_standard_json_tools_args(args: &[Value]) -> Result<()> {
    require(
        (5..=13).contains(&args.len()),
        "PEG argument count mismatch",
    )?;
    let _ = text_value(&args[0], MAX_STRING_BYTES)?;
    let _ = text_value(&args[1], MAX_STRING_BYTES)?;
    validate_json_wrapper(&args[2])?;
    require(
        args[3].is_boolean() && args[4].is_boolean(),
        "chat tool options must be boolean",
    )?;
    for (idx, arg) in args.iter().enumerate().skip(5) {
        if matches!(idx, 7 | 8 | 12) {
            require(arg.is_boolean(), "chat tool option must be boolean")?;
        } else if idx == 11 {
            let list = arg
                .as_array()
                .ok_or_else(|| invalid("chat parameter order must be array"))?;
            require(
                list.len() <= MAX_PARAM_ORDER,
                "chat parameter order exceeds bound",
            )?;
            for item in list {
                let _ = text_value(item, MAX_STRING_BYTES)?;
            }
        } else {
            let _ = text_value(arg, MAX_STRING_BYTES)?;
        }
    }
    Ok(())
}

fn validate_graph(
    graph: &Value,
    mut locals: BTreeSet<String>,
    depth: usize,
    count: &mut usize,
) -> Result<()> {
    require(depth <= MAX_GRAPH_DEPTH, "PEG graph depth exceeds bound")?;
    exact_fields(graph, &["Statements", "Root"])?;
    let statements = array(graph, "Statements")?;
    require(
        statements.len() <= MAX_STATEMENTS,
        "PEG statements exceed bound",
    )?;
    for statement in statements {
        if statement.get("Let").is_some() {
            exact_fields(statement, &["Let", "Expr"])?;
            let _ = text_field(statement, "Let", MAX_LOCAL_NAME)?;
        } else {
            exact_fields(statement, &["Expr"])?;
        }
        validate_expr(field(statement, "Expr")?, &locals, depth + 1, count)?;
        if let Some(let_name) = statement.get("Let").and_then(Value::as_str) {
            locals.insert(let_name.to_owned());
        }
    }
    validate_expr(field(graph, "Root")?, &locals, depth + 1, count)
}

fn validate_byte_array(value: &Value) -> Result<Vec<u8>> {
    let arr = value
        .as_array()
        .ok_or_else(|| invalid("chat byte input must be array"))?;
    require(
        arr.len() <= MAX_INPUT_BYTES,
        "chat input chunk exceeds bound",
    )?;
    arr.iter()
        .map(|b| Ok(u8::try_from(int_value(b, 0, 255)?)?))
        .collect()
}

fn validate_operation(operation: &Value) -> Result<()> {
    exact_fields(operation, &["Kind", "Graph", "ResolveRefs", "Actions"])?;
    if text_field(operation, "Kind", 64)? != "chat-peg-trace" {
        return Err(invalid("unknown chat PEG operation Kind"));
    }
    let mut count = 0;
    validate_graph(field(operation, "Graph")?, BTreeSet::new(), 0, &mut count)?;
    let refs = array(operation, "ResolveRefs")?;
    require(
        refs.len() <= MAX_RESOLVE_REFS,
        "chat reference list exceeds bound",
    )?;
    for r in refs {
        let s = text_value(r, MAX_SCHEMA_JSON_BYTES)?;
        let _ = JsonParser::parse_with_limit(s, MAX_SCHEMA_JSON_BYTES).map_err(invalid)?;
    }
    let actions = array(operation, "Actions")?;
    require(
        actions.len() <= MAX_ACTIONS,
        "chat action list exceeds bound",
    )?;
    let mut input_handles = BTreeSet::new();
    for (idx, action) in actions.iter().enumerate() {
        require(action.is_object(), "chat action must be object")?;
        let kind = text_field(action, "Kind", 64)?;
        validate_action(kind, action, idx, actions, &mut input_handles)?;
    }
    Ok(())
}

fn validate_action(
    kind: &str,
    action: &Value,
    idx: usize,
    actions: &[Value],
    input_handles: &mut BTreeSet<i64>,
) -> Result<()> {
    match kind {
        "grammar" => {
            exact_fields(action, &["Kind", "Lazy"])?;
            let _ = bool_value(field(action, "Lazy")?)?;
            Ok(())
        }
        "init-input" | "append-input" => {
            exact_fields(action, &["Kind", "Input", "Bytes"])?;
            let handle = int_value(field(action, "Input")?, 0, MAX_INPUT_HANDLES)?;
            if kind == "append-input" {
                require(input_handles.contains(&handle), "unknown chat input handle")?;
            } else {
                input_handles.insert(handle);
            }
            let _ = validate_byte_array(field(action, "Bytes")?)?;
            Ok(())
        }
        "free-input" => {
            exact_fields(action, &["Kind", "Input"])?;
            let handle = int_value(field(action, "Input")?, 0, MAX_INPUT_HANDLES)?;
            require(input_handles.remove(&handle), "unknown chat input handle")?;
            Ok(())
        }
        "parse" => {
            if action.get("Input").is_some() {
                exact_fields(action, &["Kind", "Input", "Flags", "Mapper"])?;
                let handle = int_value(field(action, "Input")?, 0, MAX_INPUT_HANDLES)?;
                require(
                    input_handles.contains(&handle),
                    "unknown chat parse input handle",
                )?;
            } else {
                exact_fields(action, &["Kind", "Bytes", "Flags", "Mapper"])?;
                let _ = validate_byte_array(field(action, "Bytes")?)?;
            }
            let _ = int_value(field(action, "Flags")?, 0, 3)?;
            let mapper = text_field(action, "Mapper", 32)?;
            require(
                matches!(
                    mapper,
                    "chat" | "chat-unretained" | "tags" | "tags-anywhere" | "none"
                ),
                "unknown chat mapper",
            )?;
            Ok(())
        }
        "diff" => {
            exact_fields(action, &["Kind", "Before", "After"])?;
            for key in ["Before", "After"] {
                let val = field(action, key)?;
                if val.is_null() {
                    require(key == "Before", "null chat After")?;
                    continue;
                }
                let n = usize::try_from(int_value(val, 0, 32_767)?)?;
                require(
                    n < idx
                        && actions[n].get("Kind").and_then(Value::as_str) == Some("parse")
                        && actions[n].get("Mapper").and_then(Value::as_str) == Some("chat"),
                    "chat diff must reference prior chat parses",
                )?;
            }
            Ok(())
        }
        "clock" => exact_fields(action, &["Kind"]),
        _ => Err(invalid("unknown chat trace action")),
    }
}

// ---------------------------------------------------------------------------
// Declarative graph construction
// ---------------------------------------------------------------------------

fn build_expr(
    builder: &mut ChatPegBuilder,
    expression: &Value,
    locals: &BTreeMap<String, ParserId>,
) -> Result<ParserId> {
    if let Some(ref_val) = expression.get("Ref") {
        let name = ref_val
            .as_str()
            .ok_or_else(|| invalid("PEG Ref must be string"))?;
        return locals
            .get(name)
            .copied()
            .ok_or_else(|| invalid("unknown PEG local Ref"));
    }
    let op = text_field(expression, "Op", MAX_LOCAL_NAME)?;
    let args = array(expression, "Args")?;
    build_op(builder, op, args, locals)
}

fn build_op(
    builder: &mut ChatPegBuilder,
    op: &str,
    args: &[Value],
    locals: &BTreeMap<String, ParserId>,
) -> Result<ParserId> {
    match op {
        "eps" => Ok(builder.inner_mut().eps()),
        "start" => Ok(builder.inner_mut().start()),
        "end" => Ok(builder.inner_mut().end()),
        "any" => Ok(builder.inner_mut().any()),
        "space" => Ok(builder.inner_mut().space()),
        "rest" => Ok(builder.inner_mut().rest()),
        "marker" => Ok(builder.inner_mut().marker()),
        "choice-empty" => Ok(builder.inner_mut().choice(Vec::new())),
        "double_quoted_string" => Ok(builder.inner_mut().double_quoted_string()),
        "single_quoted_string" => Ok(builder.inner_mut().single_quoted_string()),
        "json" => builder.inner_mut().json(),
        "python_value" => builder.inner_mut().python_value(),
        "literal" => Ok(builder
            .inner_mut()
            .literal(text_value(&args[0], MAX_STRING_BYTES)?)),
        "ref" => Ok(builder
            .inner_mut()
            .ref_rule(text_value(&args[0], MAX_STRING_BYTES)?)),
        "until" => Ok(builder
            .inner_mut()
            .until(text_value(&args[0], MAX_STRING_BYTES)?)),
        "chars" => {
            let pattern = text_value(&args[0], MAX_STRING_BYTES)?;
            let min = if let Some(val) = args.get(1) {
                i32::try_from(int_value(val, 0, 65_536)?)?
            } else {
                1
            };
            let max = if let Some(val) = args.get(2) {
                i32::try_from(int_value(val, -1, 65_536)?)?
            } else {
                -1
            };
            builder.inner_mut().chars(pattern, min, max)
        }
        "string_content" => {
            let quote = u8::try_from(int_value(&args[0], 0, 255)?)?;
            Ok(builder.inner_mut().string_content(quote))
        }
        "until_one_of" => Ok(builder.inner_mut().until_one_of(string_list(&args[0])?)),
        "sequence" | "choice" => {
            let items = args[0]
                .as_array()
                .ok_or_else(|| invalid("PEG parser list must be array"))?;
            let mut children = Vec::with_capacity(items.len());
            for item in items {
                children.push(build_expr(builder, item, locals)?);
            }
            Ok(if op == "sequence" {
                builder.inner_mut().sequence(children)
            } else {
                builder.inner_mut().choice(children)
            })
        }
        "or-assign" => {
            let left = build_expr(builder, &args[0], locals)?;
            let right = build_expr(builder, &args[1], locals)?;
            Ok(builder.inner_mut().or_assign(left, right))
        }
        "add" | "or" | "space-add" => {
            let left = if args[0].is_string() {
                builder
                    .inner_mut()
                    .literal(text_value(&args[0], MAX_STRING_BYTES)?)
            } else {
                build_expr(builder, &args[0], locals)?
            };
            let right = if args[1].is_string() {
                builder
                    .inner_mut()
                    .literal(text_value(&args[1], MAX_STRING_BYTES)?)
            } else {
                build_expr(builder, &args[1], locals)?
            };
            Ok(match op {
                "add" => builder.inner_mut().add(left, right),
                "or" => builder.inner_mut().or(left, right),
                _ => builder.inner_mut().space_add(left, right),
            })
        }
        _ => build_op_chat_and_composite(builder, op, args, locals),
    }
}

fn build_op_chat_and_composite(
    builder: &mut ChatPegBuilder,
    op: &str,
    args: &[Value],
    locals: &BTreeMap<String, ParserId>,
) -> Result<ParserId> {
    match op {
        "rule" => {
            let name = text_value(&args[0], MAX_LOCAL_NAME)?;
            let child = build_expr(builder, &args[1], locals)?;
            let trigger = args.get(2).map(bool_value).transpose()?.unwrap_or(false);
            Ok(builder.inner_mut().rule(name, child, trigger))
        }
        "rule-lazy" => {
            let name = text_value(&args[0], MAX_LOCAL_NAME)?.to_owned();
            let trigger = args.get(2).map(bool_value).transpose()?.unwrap_or(false);
            let placeholder = builder.inner_mut().begin_rule_lazy(name.clone(), trigger);
            let body = build_graph(builder, &args[1], locals.clone())?;
            builder
                .inner_mut()
                .finish_rule_lazy(placeholder, name, body, trigger);
            Ok(placeholder)
        }
        "trigger_rule" => {
            let name = text_value(&args[0], MAX_LOCAL_NAME)?;
            let child = build_expr(builder, &args[1], locals)?;
            Ok(builder.trigger_rule(name, child))
        }
        "tag" | "json_member" => {
            let name = text_value(&args[0], MAX_STRING_BYTES)?;
            let child = build_expr(builder, &args[1], locals)?;
            Ok(if op == "tag" {
                builder.inner_mut().tag(name, child)
            } else {
                builder.inner_mut().json_member(name, child)
            })
        }
        "gbnf" => {
            let child = build_expr(builder, &args[0], locals)?;
            let grammar = text_value(&args[1], MAX_STRING_BYTES)?;
            Ok(builder.inner_mut().gbnf(child, grammar))
        }
        "ac" => {
            let child = build_expr(builder, &args[0], locals)?;
            let delims = if args[1].is_string() {
                vec![text_value(&args[1], MAX_STRING_BYTES)?.to_owned()]
            } else {
                string_list(&args[1])?
            };
            Ok(builder.inner_mut().ac(child, delims))
        }
        "repeat" => {
            let child = build_expr(builder, &args[0], locals)?;
            let min = i32::try_from(int_value(&args[1], 0, 65_536)?)?;
            let max = if let Some(val) = args.get(2) {
                i32::try_from(int_value(val, -1, 65_536)?)?
            } else {
                -1
            };
            builder.inner_mut().repeat(child, min, max)
        }
        "permute" => {
            let prefix = text_value(&args[0], MAX_LOCAL_NAME)?;
            let items = args[1]
                .as_array()
                .ok_or_else(|| invalid("chat permute list must be array"))?;
            let mut children = Vec::with_capacity(items.len());
            for item in items {
                children.push(build_expr(builder, item, locals)?);
            }
            Ok(builder.permute(prefix, &children))
        }
        "schema" => {
            let child = build_expr(builder, &args[0], locals)?;
            let name = text_value(&args[1], MAX_STRING_BYTES)?.to_owned();
            let schema_val = parse_json_wrapper(&args[2])?;
            let raw = args.get(3).map(bool_value).transpose()?.unwrap_or(false);
            Ok(builder.schema(child, name, schema_val, raw))
        }
        "standard_json_tools" => build_standard_json_tools(builder, args),
        "standard_constructed_tools" => build_standard_constructed_tools(builder, args),
        _ => build_op_unary(builder, op, args, locals),
    }
}

fn build_op_unary(
    builder: &mut ChatPegBuilder,
    op: &str,
    args: &[Value],
    locals: &BTreeMap<String, ParserId>,
) -> Result<ParserId> {
    let child = build_expr(builder, &args[0], locals)?;
    Ok(match op {
        "one_or_more" => builder.inner_mut().one_or_more(child),
        "zero_or_more" => builder.inner_mut().zero_or_more(child),
        "optional" => builder.inner_mut().optional(child),
        "atomic" => builder.inner_mut().atomic(child),
        "peek" => builder.inner_mut().peek(child),
        "negate" => builder.inner_mut().negate(child),
        "reasoning_block" => builder.reasoning_block(child),
        "reasoning" => builder.reasoning(child),
        "content" => builder.content(child),
        "tool" => builder.tool(child),
        "tool_open" => builder.tool_open(child),
        "tool_close" => builder.tool_close(child),
        "tool_id" => builder.tool_id(child),
        "tool_name" => builder.tool_name(child),
        "tool_args" => builder.tool_args(child),
        "tool_arg" => builder.tool_arg(child),
        "tool_arg_open" => builder.tool_arg_open(child),
        "tool_arg_close" => builder.tool_arg_close(child),
        "tool_arg_name" => builder.tool_arg_name(child),
        "tool_arg_string_value" => builder.tool_arg_string_value(child),
        "tool_arg_json_value" => builder.tool_arg_json_value(child),
        _ => return Err(invalid("unknown PEG builder Op")),
    })
}

fn build_standard_json_tools(builder: &mut ChatPegBuilder, args: &[Value]) -> Result<ParserId> {
    let section_start = text_value(&args[0], MAX_STRING_BYTES)?;
    let section_end = text_value(&args[1], MAX_STRING_BYTES)?;
    let tools = parse_json_wrapper(&args[2])?;
    let parallel_tool_calls = bool_value(&args[3])?;
    let force_tool_calls = bool_value(&args[4])?;
    let name_field = args
        .get(5)
        .map(|v| text_value(v, MAX_STRING_BYTES))
        .transpose()?
        .unwrap_or("");
    let args_field = args
        .get(6)
        .map(|v| text_value(v, MAX_STRING_BYTES))
        .transpose()?
        .unwrap_or("");
    let args_are_raw = args.get(7).map(bool_value).transpose()?.unwrap_or(false);
    let fun_name_is_key = args.get(8).map(bool_value).transpose()?.unwrap_or(false);
    let id_field = args
        .get(9)
        .map(|v| text_value(v, MAX_STRING_BYTES))
        .transpose()?
        .unwrap_or("");
    let gen_id_field = args
        .get(10)
        .map(|v| text_value(v, MAX_STRING_BYTES))
        .transpose()?
        .unwrap_or("");
    let param_order = if let Some(list) = args.get(11) {
        string_list(list)?
    } else {
        Vec::new()
    };
    let trailing_raw = args.get(12).map(bool_value).transpose()?.unwrap_or(false);
    builder.standard_json_tools(&JsonToolsOptions {
        section_start,
        section_end,
        tools: &tools,
        parallel_tool_calls,
        force_tool_calls,
        per_call_start: "",
        per_call_end: "",
        fun_name_is_key,
        tools_array_wrapped: false,
        name_field,
        args_field,
        id_field,
        gen_id_field,
        parameter_order: &param_order,
        allow_raw_arguments: args_are_raw || trailing_raw,
    })
}

fn build_standard_constructed_tools(
    builder: &mut ChatPegBuilder,
    args: &[Value],
) -> Result<ParserId> {
    let markers_val = parse_json_wrapper(&args[0])?;
    let mut markers = BTreeMap::new();
    let mut unknown = Vec::new();
    for (key, val) in markers_val.object() {
        if !ConstructedMarkers::is_known_key(key) {
            unknown.push(key.clone());
        }
        let s = val
            .text()
            .ok_or_else(|| invalid("constructed tool marker value must be string"))?;
        markers.insert(key.clone(), s.to_owned());
    }
    if !unknown.is_empty() {
        return Err(invalid(format!(
            "unknown constructed tool marker keys: {}",
            unknown.join(", ")
        )));
    }
    let tools = parse_json_wrapper(&args[1])?;
    let parallel_tool_calls = bool_value(&args[2])?;
    let force_tool_calls = bool_value(&args[3])?;
    builder.standard_constructed_tools(&markers, &tools, parallel_tool_calls, force_tool_calls)
}

fn string_list(value: &Value) -> Result<Vec<String>> {
    let arr = value
        .as_array()
        .ok_or_else(|| invalid("PEG string list must be array"))?;
    arr.iter()
        .map(|entry| Ok(text_value(entry, MAX_STRING_BYTES)?.to_owned()))
        .collect()
}

fn build_graph(
    builder: &mut ChatPegBuilder,
    graph: &Value,
    mut locals: BTreeMap<String, ParserId>,
) -> Result<ParserId> {
    for statement in array(graph, "Statements")? {
        let value = build_expr(builder, field(statement, "Expr")?, &locals)?;
        if let Some(let_name) = statement.get("Let").and_then(Value::as_str) {
            locals.insert(let_name.to_owned(), value);
        }
    }
    build_expr(builder, field(graph, "Root")?, &locals)
}

// ---------------------------------------------------------------------------
// Trace execution
// ---------------------------------------------------------------------------

fn execute_parse_action(
    arena: &ChatArena,
    action: &Value,
    inputs: &BTreeMap<i64, Vec<u8>>,
) -> Result<(Value, Option<ChatMessage>)> {
    let input_owned;
    let input = if let Some(handle_val) = action.get("Input") {
        let handle = int_value(handle_val, 0, MAX_INPUT_HANDLES)?;
        inputs
            .get(&handle)
            .ok_or_else(|| invalid("unknown chat parse input handle"))?
            .as_slice()
    } else {
        input_owned = validate_byte_array(field(action, "Bytes")?)?;
        &input_owned
    };
    let flags = u8::try_from(int_value(field(action, "Flags")?, 0, 3)?)?;
    let mapper = text_field(action, "Mapper", 32)?;

    let mut ctx = ParseContext::new(
        input,
        if mapper == "tags-anywhere" {
            parse_flags::NONE
        } else {
            flags
        },
    );
    let parsed = if mapper == "tags-anywhere" {
        arena.parse_anywhere(&mut ctx)?
    } else {
        arena.parse_with_context(&mut ctx)?
    };
    let result_json = wire_parse_result(&parsed);
    let mut message_json = Value::Null;
    let mut tags_json = Value::Null;
    let mut retained = None;

    if matches!(mapper, "chat" | "chat-unretained") {
        let mut msg = ChatMessage::default();
        map_chat_ast(&ctx.ast, &parsed, &mut msg);
        message_json = wire_chat_message(&msg);
        if mapper == "chat" {
            retained = Some(msg);
        }
    } else if matches!(mapper, "tags" | "tags-anywhere") {
        let extracted = extract_tags(&ctx.ast, &parsed.nodes, input)?;
        tags_json = json!(extracted);
    }

    Ok((
        json!({
            "Result": result_json,
            "Message": message_json,
            "Tags": tags_json,
            "ExceptionKind": null,
            "ExceptionMessage": "",
        }),
        retained,
    ))
}

fn execute_diff_action(action: &Value, messages: &BTreeMap<usize, ChatMessage>) -> Result<Value> {
    let before_val = field(action, "Before")?;
    let before = if before_val.is_null() {
        ChatMessage::default()
    } else {
        let b_idx = usize::try_from(int_value(before_val, 0, 32_767)?)?;
        messages
            .get(&b_idx)
            .ok_or_else(|| invalid("missing prior chat parse"))?
            .clone()
    };
    let a_idx = usize::try_from(int_value(field(action, "After")?, 0, 32_767)?)?;
    let after = messages
        .get(&a_idx)
        .ok_or_else(|| invalid("missing after chat parse"))?;
    Ok(match compute_message_diffs(&before, after) {
        Ok(diffs) => {
            let wire_diffs: Vec<Value> = diffs
                .iter()
                .map(|d| {
                    json!({
                        "ReasoningContentDelta": d.reasoning_content_delta,
                        "ContentDelta": d.content_delta,
                        "ToolCallIndex": d.tool_call_index.map_or(u64::MAX, |i| u64::try_from(i).unwrap_or(u64::MAX)),
                        "ToolCallDelta": wire_chat_tool_call(&d.tool_call_delta),
                    })
                })
                .collect();
            json!({
                "ExceptionKind": null,
                "Message": "",
                "Diffs": wire_diffs,
            })
        }
        Err(err) => json!({
            "ExceptionKind": "standard",
            "Message": err.to_string(),
            "Diffs": [],
        }),
    })
}

fn execute_operation(operation: &Value) -> Result<Value> {
    let mut builder = ChatPegBuilder::new();
    let root = build_graph(&mut builder, field(operation, "Graph")?, BTreeMap::new())?;
    builder.set_root(root);
    let arena = builder.build()?;

    let mut resolve_refs = Vec::new();
    for ref_val in array(operation, "ResolveRefs")? {
        let raw = text_value(ref_val, MAX_SCHEMA_JSON_BYTES)?;
        resolve_refs
            .push(JsonParser::parse_with_limit(raw, MAX_SCHEMA_JSON_BYTES).map_err(invalid)?);
    }

    let mut messages: BTreeMap<usize, ChatMessage> = BTreeMap::new();
    let mut inputs: BTreeMap<i64, Vec<u8>> = BTreeMap::new();
    let mut trace = Vec::new();
    let clock = Instant::now();

    for action in array(operation, "Actions")? {
        let index = trace.len();
        let kind = text_field(action, "Kind", 64)?;
        match kind {
            "init-input" | "append-input" => {
                let handle = int_value(field(action, "Input")?, 0, MAX_INPUT_HANDLES)?;
                let chunk = validate_byte_array(field(action, "Bytes")?)?;
                let buf = inputs.entry(handle).or_default();
                if kind == "init-input" {
                    buf.clear();
                }
                buf.extend_from_slice(&chunk);
                require(
                    buf.len() <= MAX_INPUT_BYTES,
                    "assembled chat input exceeds bound",
                )?;
                trace.push(json!({}));
            }
            "free-input" => {
                let handle = int_value(field(action, "Input")?, 0, MAX_INPUT_HANDLES)?;
                inputs.remove(&handle);
                trace.push(json!({}));
            }
            "clock" => {
                let us = i64::try_from(clock.elapsed().as_micros())
                    .map_err(|_| invalid("chat clock overflow"))?;
                trace.push(json!({ "Microseconds": us }));
            }
            "grammar" => {
                let lazy = bool_value(field(action, "Lazy")?)?;
                let grammar = arena.build_grammar(lazy, &resolve_refs)?;
                trace.push(json!({ "GrammarText": grammar }));
            }
            "diff" => trace.push(execute_diff_action(action, &messages)?),
            "parse" => match execute_parse_action(&arena, action, &inputs) {
                Ok((val, retained)) => {
                    if let Some(msg) = retained {
                        messages.insert(index, msg);
                    }
                    trace.push(val);
                }
                Err(err) => {
                    trace.push(json!({
                        "Result": null,
                        "Message": null,
                        "Tags": null,
                        "ExceptionKind": "standard",
                        "ExceptionMessage": err.to_string(),
                    }));
                }
            },
            _ => return Err(invalid("unknown chat trace action")),
        }
    }
    Ok(json!({ "Trace": trace }))
}

/// Returns `true` when every operation in `request` is `"chat-peg-trace"`.
pub(super) fn handles(request: &Value) -> bool {
    request["Operations"].as_array().is_some_and(|operations| {
        !operations.is_empty()
            && operations
                .iter()
                .all(|operation| operation["Kind"].as_str() == Some("chat-peg-trace"))
    })
}

/// Validates and executes a `chat-peg-trace` wire request.
pub(super) fn execute(request: &Value) -> Result<Value> {
    exact_fields(request, &["SchemaVersion", "Operations"])?;
    if int_value(field(request, "SchemaVersion")?, 1, 1)? != 1 {
        return Err(invalid("chat PEG request SchemaVersion must equal 1"));
    }
    let operations = array(request, "Operations")?;
    require(
        !operations.is_empty() && operations.len() <= MAX_OPERATIONS,
        "chat PEG Operations must contain 1 through 128 operations",
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

/// Provider identity metadata for the independent chat PEG parser reference.
pub(super) fn identity() -> Value {
    json!({
        "Kind": "independent-rust-chat-peg-parser",
        "SupportedSourceIds": [ID],
        "UpstreamCommit": super::UPSTREAM,
        "AssertionSource": {"Path": ID, "Sha256": SOURCE_SHA256},
        "Architecture": "clean-room-chat-peg-builder-ast-mapper-and-msg-diff-v1",
        "NativeReferenceReplay": false,
        "ProductionRuntimeCalled": false,
        "CompleteSourceGate": false,
        "NativeRetirementQualified": false
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, Target};

    #[test]
    fn independent_chat_peg_parser_passes_all_seven_groups_on_rust_reference_and_cpp() -> Result<()>
    {
        for target in [Target::RustReference, Target::Cpp] {
            let engine = Engine::new(target, None, None, None, None)?;
            let report = crate::chat_peg::run(&engine)?;
            assert_eq!(report["Id"], ID);
            assert_eq!(report["Status"], "pass", "{report}");
        }
        assert!(crate::reference_provider::supports(ID));
        let meta = identity();
        assert_eq!(meta["NativeReferenceReplay"], false);
        assert_eq!(meta["ProductionRuntimeCalled"], false);
        Ok(())
    }

    #[test]
    fn independent_chat_peg_parser_rejects_invalid_ops_mappers_handles_and_bounds() -> Result<()> {
        let valid = json!({
            "SchemaVersion": 1,
            "Operations": [{
                "Kind": "chat-peg-trace",
                "Graph": {"Statements": [], "Root": {"Op": "content", "Args": [{"Op": "rest", "Args": []}]}},
                "ResolveRefs": [],
                "Actions": [
                    {"Kind": "init-input", "Input": 0, "Bytes": [104, 105]},
                    {"Kind": "parse", "Input": 0, "Flags": 0, "Mapper": "chat"},
                    {"Kind": "diff", "Before": null, "After": 1},
                    {"Kind": "clock"},
                    {"Kind": "grammar", "Lazy": false},
                    {"Kind": "free-input", "Input": 0}
                ]
            }]
        });
        let ok = execute(&valid)?;
        assert_eq!(ok["Results"][0]["Trace"][1]["Message"]["Content"], "hi");
        assert_eq!(
            ok["Results"][0]["Trace"][2]["Diffs"][0]["ContentDelta"],
            "hi"
        );
        assert!(ok["Results"][0]["Trace"][3]["Microseconds"].is_i64());

        // Unknown action kind rejected.
        let mut bad_action = valid.clone();
        bad_action["Operations"][0]["Actions"][0] = json!({"Kind": "bogus"});
        assert!(execute(&bad_action).is_err());

        // Unknown mapper rejected.
        let mut bad_mapper = valid.clone();
        bad_mapper["Operations"][0]["Actions"][1]["Mapper"] = json!("xml");
        assert!(execute(&bad_mapper).is_err());

        // Missing/uninitialized input handle rejected.
        let mut bad_handle = valid.clone();
        bad_handle["Operations"][0]["Actions"][1]["Input"] = json!(7);
        assert!(execute(&bad_handle).is_err());

        // Out-of-range input handle (> 31) rejected.
        let mut oob_handle = valid.clone();
        oob_handle["Operations"][0]["Actions"][0]["Input"] = json!(32);
        assert!(execute(&oob_handle).is_err());

        // Oversized input bytes rejected.
        let mut huge_bytes = valid.clone();
        huge_bytes["Operations"][0]["Actions"][0]["Bytes"] =
            json!(vec![65_u8; MAX_INPUT_BYTES + 1]);
        assert!(execute(&huge_bytes).is_err());

        // Excessive action count (> 32768) rejected.
        let mut too_many_actions = valid.clone();
        too_many_actions["Operations"][0]["Actions"] =
            json!(vec![json!({"Kind": "clock"}); MAX_ACTIONS + 1]);
        assert!(execute(&too_many_actions).is_err());

        // Unknown PEG builder op rejected.
        let mut bad_op = valid;
        bad_op["Operations"][0]["Graph"]["Root"] = json!({"Op": "no_such_chat_op", "Args": []});
        assert!(execute(&bad_op).is_err());

        Ok(())
    }
}
