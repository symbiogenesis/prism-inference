//! Clean-room independent safe Rust chat auto-parser, differential template analyzer,
//! and parser builder reference provider.
//!
//! Implements the `chat-helper`, `chat-compare-variants`, and `chat-auto-trace`
//! protocols directly from `tools/prism/upstream/common/chat_auto.hpp`,
//! `PROTOCOL.md`, and `tests/upstream_prism/chat_auto.rs`, layered on
//! `reference_jinja`, `reference_chat_analysis`, `reference_chat_templates`,
//! `reference_chat_parser`, `reference_json_schema`, and `reference_peg`.
//! It never imports `bonsai_prism_runtime`, invokes a native bridge, or reads
//! expected-answer fixtures.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value, json};

use super::llama_grammar::{array, exact_fields, field};
use super::reference_chat_analysis::{DiffSplit, calculate_diff_split, segment_markers};
use super::reference_chat_parser::{
    ChatArena, ChatMessage, ChatPegBuilder, JsonToolsOptions, map_chat_ast, wire_chat_message,
    wire_parse_result,
};
use super::reference_chat_templates::{adapt_messages_for_caps, current_unix_seconds};
use super::reference_jinja::{
    JVal, JinjaCaps, Program, detect_jinja_caps, eval_program_chat_fallback, eval_program_with_now,
    parse_json_jval_bounded, parse_template_with_limit,
};
use super::reference_json_schema::{JsonParser, JsonVal};
use super::reference_peg::{ParseContext, ParserId};
use super::{Result, invalid};

/// Upstream source identifier covered by this independent reference provider.
pub(super) const ID: &str = "tests/test-chat-auto-parser.cpp";
const SOURCE_SHA256: &str = "d69af3154f3ed738e47a74280bcd63d2e90cd60b05dff7f5f1f7a5b26c59478c";

const MAX_OPERATIONS: usize = 128;
const MAX_ACTIONS: usize = 128;
const MAX_TEMPLATE_BYTES: usize = 1024 * 1024;
const MAX_TOKEN_BYTES: usize = 65_536;
const MAX_HELPER_ARG_BYTES: usize = 1024 * 1024;
const MAX_PARAMS_JSON_BYTES: usize = 4 * 1024 * 1024;
const MAX_PARAMS_JSON_DEPTH: usize = 64;
const MAX_PARSE_BYTES: usize = 1024 * 1024;

// ---------------------------------------------------------------------------
// Helper utilities: diff split, common prefix/suffix, segmentize markers
// ---------------------------------------------------------------------------

/// Segment classification emitted by [`segmentize_markers`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SegmentType {
    /// Ordinary text segment (`0`).
    Text = 0,
    /// Bracketed `<...>` or `[...]` marker token (`1`).
    Marker = 1,
}

impl SegmentType {
    const fn as_u8(self) -> u8 {
        match self {
            Self::Text => 0,
            Self::Marker => 1,
        }
    }
}

/// Segment emitted by [`segmentize_markers`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Segment {
    /// Segment kind (`Text` = `0`, `Marker` = `1`).
    pub(super) seg_type: SegmentType,
    /// Segment text payload.
    pub(super) value: String,
}

/// String-typed differential split matching `chat-helper` / `chat-compare-variants`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct StringDiffSplit {
    /// Common prefix before divergence.
    pub(super) prefix: String,
    /// Common suffix after divergence.
    pub(super) suffix: String,
    /// Left divergent region.
    pub(super) left: String,
    /// Right divergent region.
    pub(super) right: String,
}

impl StringDiffSplit {
    fn to_json(&self) -> Value {
        json!({
            "prefix": self.prefix,
            "suffix": self.suffix,
            "left": self.left,
            "right": self.right,
        })
    }
}

fn adjust_utf8_split(left: &str, right: &str, split: &DiffSplit) -> StringDiffSplit {
    let mut prefix_len = split.prefix.len().min(left.len()).min(right.len());
    while prefix_len > 0
        && (!left.is_char_boundary(prefix_len) || !right.is_char_boundary(prefix_len))
    {
        prefix_len -= 1;
    }
    let mut suffix_len = split
        .suffix
        .len()
        .min(left.len().saturating_sub(prefix_len))
        .min(right.len().saturating_sub(prefix_len));
    while suffix_len > 0
        && (!left.is_char_boundary(left.len() - suffix_len)
            || !right.is_char_boundary(right.len() - suffix_len))
    {
        suffix_len -= 1;
    }
    let left_end = left.len() - suffix_len;
    let right_end = right.len() - suffix_len;
    StringDiffSplit {
        prefix: left[..prefix_len].to_owned(),
        suffix: left[left_end..].to_owned(),
        left: left[prefix_len..left_end].to_owned(),
        right: right[prefix_len..right_end].to_owned(),
    }
}

/// Computes `calculate_diff_split` over valid UTF-8 strings preserving char boundaries.
pub(super) fn diff_split_str(left: &str, right: &str) -> StringDiffSplit {
    let raw = calculate_diff_split(left.as_bytes(), right.as_bytes());
    adjust_utf8_split(left, right, &raw)
}

fn exact_byte_diff(left: &str, right: &str) -> StringDiffSplit {
    let mut prefix_len = left
        .as_bytes()
        .iter()
        .zip(right.as_bytes())
        .take_while(|(a, b)| a == b)
        .count();
    while prefix_len > 0
        && (!left.is_char_boundary(prefix_len) || !right.is_char_boundary(prefix_len))
    {
        prefix_len -= 1;
    }
    // Snap back if prefix split inside an unclosed `<...>` or `[...]` marker on either side.
    let mut scan = prefix_len;
    while scan > 0 {
        let prev = left.as_bytes()[scan - 1];
        if prev == b'>' || prev == b']' || prev == b'\n' {
            break;
        }
        if prev == b'<' || prev == b'[' {
            let next_l = left.as_bytes().get(prefix_len).copied();
            let next_r = right.as_bytes().get(prefix_len).copied();
            let closed_at_diff_l = left[scan - 1..].find(if prev == b'<' { '>' } else { ']' });
            let closed_at_diff_r = right[scan - 1..].find(if prev == b'<' { '>' } else { ']' });
            if closed_at_diff_l.is_some()
                && closed_at_diff_r.is_some()
                && (next_l.is_some() || next_r.is_some())
            {
                prefix_len = scan - 1;
            }
            break;
        }
        scan -= 1;
    }

    let rem_l = &left.as_bytes()[prefix_len..];
    let rem_r = &right.as_bytes()[prefix_len..];
    let mut suffix_len = rem_l
        .iter()
        .rev()
        .zip(rem_r.iter().rev())
        .take_while(|(a, b)| a == b)
        .count();
    while suffix_len > 0
        && (!left.is_char_boundary(left.len() - suffix_len)
            || !right.is_char_boundary(right.len() - suffix_len))
    {
        suffix_len -= 1;
    }
    // Snap suffix forward if suffix starts mid-tag (after `<` / `[` consumed on the left/right).
    if suffix_len > 0 {
        let suf_bytes = &left.as_bytes()[left.len() - suffix_len..];
        for (i, &b) in suf_bytes.iter().enumerate() {
            if b == b'<' || b == b'[' || b == b'\n' {
                break;
            }
            if b == b'>' || b == b']' {
                suffix_len -= i + 1;
                break;
            }
        }
    }
    StringDiffSplit {
        prefix: left[..prefix_len].to_owned(),
        suffix: left[left.len() - suffix_len..].to_owned(),
        left: left[prefix_len..left.len() - suffix_len].to_owned(),
        right: right[prefix_len..right.len() - suffix_len].to_owned(),
    }
}

/// Splits `input` into alternating text (`0`) and `<...>`/`[...]` marker (`1`) segments.
pub(super) fn segmentize_markers(input: &str) -> Vec<Segment> {
    let raw = segment_markers(input.as_bytes());
    let mut out: Vec<Segment> = Vec::with_capacity(raw.len());
    for (is_marker, bytes) in raw {
        let seg_type = if is_marker {
            SegmentType::Marker
        } else {
            SegmentType::Text
        };
        let text = String::from_utf8_lossy(bytes).into_owned();
        if seg_type == SegmentType::Text
            && let Some(prev) = out.last_mut()
            && prev.seg_type == SegmentType::Text
        {
            prev.value.push_str(&text);
            continue;
        }
        out.push(Segment {
            seg_type,
            value: text,
        });
    }
    out
}

/// Returns the substring of `input` preceding the common prefix of `left` and `right`.
pub(super) fn until_common_prefix(input: &str, left: &str, right: &str) -> String {
    let mut common_len = left
        .as_bytes()
        .iter()
        .zip(right.as_bytes())
        .take_while(|(a, b)| a == b)
        .count();
    while common_len > 0 && !left.is_char_boundary(common_len) {
        common_len -= 1;
    }
    if common_len == 0 {
        return String::new();
    }
    let common = &left[..common_len];
    input
        .find(common)
        .map_or_else(String::new, |pos| input[..pos].to_owned())
}

/// Returns the substring of `input` following the common suffix of `left` and `right`.
pub(super) fn after_common_suffix(input: &str, left: &str, right: &str) -> String {
    let mut common_len = left
        .as_bytes()
        .iter()
        .rev()
        .zip(right.as_bytes().iter().rev())
        .take_while(|(a, b)| a == b)
        .count();
    while common_len > 0 && !left.is_char_boundary(left.len() - common_len) {
        common_len -= 1;
    }
    if common_len == 0 {
        return String::new();
    }
    let common = &left[left.len() - common_len..];
    input
        .rfind(common)
        .map_or_else(String::new, |pos| input[pos + common.len()..].to_owned())
}

// ---------------------------------------------------------------------------
// ChatTemplate wrapper, TemplateParams, original_caps, and compare_variants
// ---------------------------------------------------------------------------

/// Parameters supplied to template rendering / differential comparison.
#[derive(Clone, Debug)]
pub(super) struct TemplateParams {
    /// Optional `messages` JSON value (`JVal`).
    pub(super) messages: Option<JVal>,
    /// Optional `tools` JSON value (`JVal`).
    pub(super) tools: Option<JVal>,
    /// Whether to append the assistant generation prompt.
    pub(super) add_generation_prompt: bool,
    /// Whether thinking/reasoning mode is enabled.
    pub(super) enable_thinking: bool,
    /// Optional extra Jinja template globals.
    pub(super) extra_context: Option<JVal>,
}

impl Default for TemplateParams {
    fn default() -> Self {
        Self {
            messages: None,
            tools: None,
            add_generation_prompt: false,
            enable_thinking: true,
            extra_context: None,
        }
    }
}

/// Differential rendering comparison returned by [`compare_variants`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct CompareVariantsResult {
    /// Marker-aware split between `output_a` and `output_b`.
    pub(super) diff: StringDiffSplit,
    /// Rendered output from the base parameters.
    pub(super) output_a: String,
    /// Rendered output from the modified parameters.
    pub(super) output_b: String,
}

/// Parsed chat template wrapper retaining source tokens and probed capabilities.
pub(super) struct ChatTemplate {
    bos_token: String,
    eos_token: String,
    program: Option<Program>,
    caps: JinjaCaps,
    now_unix_seconds: i64,
}

impl ChatTemplate {
    /// Compiles `source` with `bos_token` / `eos_token` and probes [`JinjaCaps`].
    pub(super) fn new(source: &str, bos_token: &str, eos_token: &str) -> Self {
        Self::new_with_now(source, bos_token, eos_token, current_unix_seconds())
    }

    /// Compiles `source` with explicit UTC `now_unix_seconds` and probes [`JinjaCaps`].
    pub(super) fn new_with_now(
        source: &str,
        bos_token: &str,
        eos_token: &str,
        now_unix_seconds: i64,
    ) -> Self {
        let program = parse_template_with_limit(source, MAX_TEMPLATE_BYTES).ok();
        let caps = program.as_ref().map_or_else(JinjaCaps::default, |prog| {
            detect_jinja_caps(prog, now_unix_seconds)
        });
        Self {
            bos_token: bos_token.to_owned(),
            eos_token: eos_token.to_owned(),
            program,
            caps,
            now_unix_seconds,
        }
    }

    /// Returns the probed 9-field [`JinjaCaps`].
    pub(super) const fn original_caps(&self) -> JinjaCaps {
        self.caps
    }

    /// Evaluates the template with `params`, returning rendered UTF-8 text or error string.
    pub(super) fn apply(&self, params: &TemplateParams) -> std::result::Result<String, String> {
        self.apply_with_now(params, None)
    }

    /// Evaluates the template with `params` and optional `now_unix_seconds` override.
    pub(super) fn apply_with_now(
        &self,
        params: &TemplateParams,
        now_unix_seconds: Option<i64>,
    ) -> std::result::Result<String, String> {
        self.apply_with_now_object_args(params, now_unix_seconds, false)
    }

    /// Evaluates the template with `params`, optional `now_unix_seconds` override, and optional
    /// forced object-typed tool arguments during message adaptation.
    pub(super) fn apply_with_now_object_args(
        &self,
        params: &TemplateParams,
        now_unix_seconds: Option<i64>,
        force_object_args: bool,
    ) -> std::result::Result<String, String> {
        let program = self
            .program
            .as_ref()
            .ok_or_else(|| "invalid Jinja template".to_owned())?;
        let now = now_unix_seconds.unwrap_or(self.now_unix_seconds);
        let mut caps = self.caps;
        if force_object_args {
            caps.supports_object_arguments = true;
        }
        let mut pairs = Vec::new();
        if let Some(ref msgs) = params.messages {
            let adapted = adapt_messages_for_template(msgs, caps);
            pairs.push((JVal::text_str("messages"), adapted));
        }
        if let Some(ref tools_val) = params.tools
            && !tools_val.is_none()
            && tools_val
                .as_array_elems()
                .is_none_or(|elems| !elems.is_empty())
        {
            pairs.push((JVal::text_str("tools"), tools_val.deep_clone_fresh()));
        }
        pairs.push((
            JVal::text_str("bos_token"),
            JVal::bytes(self.bos_token.as_bytes().to_vec(), true),
        ));
        pairs.push((
            JVal::text_str("eos_token"),
            JVal::bytes(self.eos_token.as_bytes().to_vec(), true),
        ));
        pairs.push((
            JVal::text_str("add_generation_prompt"),
            JVal::boolean(params.add_generation_prompt),
        ));
        pairs.push((
            JVal::text_str("enable_thinking"),
            JVal::boolean(params.enable_thinking),
        ));
        if let Some(ref extra) = params.extra_context
            && let Some(extra_pairs) = extra.as_object_pairs()
        {
            for (k, v) in extra_pairs {
                let key_bytes = k.raw_bytes();
                pairs.retain(|(existing_k, _)| existing_k.raw_bytes() != key_bytes);
                pairs.push((k, v.deep_clone_fresh()));
            }
        }
        let globals = JVal::object(pairs);
        let spans = match eval_program_with_now(program, &globals, Some(now)) {
            Ok(s) => s,
            Err(first_err) if params.extra_context.is_none() => {
                eval_program_chat_fallback(program, &globals, Some(now))
                    .map_err(|_| first_err.message().to_owned())?
            }
            Err(err) => return Err(err.message().to_owned()),
        };
        let bytes: Vec<u8> = spans.into_iter().flat_map(|s| s.bytes).collect();
        String::from_utf8(bytes).map_err(|_| "rendered chat output is not valid UTF-8".to_owned())
    }
}

/// Computes the 9-field `JinjaCaps` of a chat template source string.
#[allow(dead_code, reason = "Exposed for Stage 4 chat provider")]
pub(super) fn original_caps(template: &str, bos_token: &str, eos_token: &str) -> JinjaCaps {
    ChatTemplate::new(template, bos_token, eos_token).original_caps()
}

/// Renders `tmpl` under `base` and optional `modifier`, returning `None` if either fails.
pub(super) fn compare_variants(
    tmpl: &ChatTemplate,
    base: &TemplateParams,
    modifier: Option<&dyn Fn(&mut TemplateParams)>,
) -> Option<CompareVariantsResult> {
    let output_a = tmpl.apply(base).ok()?;
    let mut updated = base.clone();
    if let Some(f) = modifier {
        f(&mut updated);
    }
    let output_b = tmpl.apply(&updated).ok()?;
    let diff = diff_split_str(&output_a, &output_b);
    Some(CompareVariantsResult {
        diff,
        output_a,
        output_b,
    })
}

fn jval_to_compact_json(val: &JVal) -> String {
    if val.is_string() {
        return String::from_utf8_lossy(&val.raw_bytes()).into_owned();
    }
    if val.is_none() || val.is_undefined() {
        return "{}".to_owned();
    }
    if let Some(pairs) = val.as_object_pairs() {
        let mut out = String::from("{");
        for (idx, (k, v)) in pairs.iter().enumerate() {
            if idx > 0 {
                out.push(',');
            }
            let key_bytes = k.raw_bytes();
            let key = String::from_utf8_lossy(&key_bytes);
            out.push_str(&serde_json::to_string(key.as_ref()).unwrap_or_default());
            out.push(':');
            out.push_str(&jval_scalar_or_json(v));
        }
        out.push('}');
        return out;
    }
    "{}".to_owned()
}

fn jval_scalar_or_json(val: &JVal) -> String {
    if val.is_none() || val.is_undefined() {
        return "null".to_owned();
    }
    if val.is_string() {
        let text_bytes = val.raw_bytes();
        let text = String::from_utf8_lossy(&text_bytes);
        return serde_json::to_string(text.as_ref()).unwrap_or_else(|_| "\"\"".to_owned());
    }
    if let Some(elems) = val.as_array_elems() {
        let items: Vec<String> = elems.iter().map(jval_scalar_or_json).collect();
        return format!("[{}]", items.join(","));
    }
    if val.as_object_pairs().is_some() {
        return jval_to_compact_json(val);
    }
    let raw_bytes = val.raw_bytes();
    let raw = String::from_utf8_lossy(&raw_bytes);
    match raw.as_ref() {
        "True" => "true".to_owned(),
        "False" => "false".to_owned(),
        "None" => "null".to_owned(),
        other => other.to_owned(),
    }
}

fn adapt_tool_call_entry(tc: JVal, want_object: bool) -> JVal {
    let Some(pairs) = tc.as_object_pairs() else {
        return tc;
    };
    let adapted: Vec<(JVal, JVal)> = pairs
        .into_iter()
        .map(|(k, v)| {
            let key = k.raw_bytes();
            if key == b"function" {
                (k, adapt_tool_call_entry(v, want_object))
            } else if key == b"arguments" {
                let new_v = if !want_object && v.as_object_pairs().is_some() {
                    JVal::bytes(jval_to_compact_json(&v).into_bytes(), true)
                } else if want_object && v.is_string() {
                    let raw = v.raw_bytes();
                    std::str::from_utf8(&raw)
                        .ok()
                        .and_then(|s| parse_json_jval_bounded(s, MAX_PARAMS_JSON_BYTES, 32).ok())
                        .unwrap_or(v)
                } else {
                    v
                };
                (k, new_v)
            } else {
                (k, v)
            }
        })
        .collect();
    JVal::object(adapted)
}

#[allow(dead_code, reason = "Exposed for Stage 4 chat provider")]
pub(super) fn adapt_messages_for_template(messages: &JVal, caps: JinjaCaps) -> JVal {
    let basic_caps = super::reference_chat_templates::TemplateCaps {
        supports_tools: caps.supports_tools,
        supports_tool_calls: caps.supports_tool_calls,
        supports_system_role: caps.supports_system_role,
        supports_parallel_tool_calls: caps.supports_parallel_tool_calls,
        supports_typed_content: caps.supports_typed_content,
        supports_string_content: caps.supports_string_content,
    };
    let content_adapted = adapt_messages_for_caps(messages, basic_caps);
    let Some(elems) = content_adapted.as_array_elems() else {
        return content_adapted;
    };
    let out = elems
        .into_iter()
        .map(|msg| {
            let Some(pairs) = msg.as_object_pairs() else {
                return msg;
            };
            let new_pairs = pairs
                .into_iter()
                .map(|(k, v)| {
                    if k.raw_bytes() == b"tool_calls"
                        && let Some(calls) = v.as_array_elems()
                    {
                        let mapped = calls
                            .into_iter()
                            .map(|c| adapt_tool_call_entry(c, caps.supports_object_arguments))
                            .collect();
                        (k, JVal::array(mapped))
                    } else {
                        (k, v)
                    }
                })
                .collect();
            JVal::object(new_pairs)
        })
        .collect();
    JVal::array(out)
}

// ---------------------------------------------------------------------------
// AutoAnalysis data model & JSON serialization
// ---------------------------------------------------------------------------

/// Reasoning detection mode (`0` = `None`, `1` = `TagBased`, `2` = `ToolsOnly`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum ReasoningMode {
    /// No reasoning tags detected (`0`).
    #[default]
    None = 0,
    /// Tag-delimited reasoning (`1`).
    TagBased = 1,
    /// Reasoning emitted only alongside tool calls (`2`).
    ToolsOnly = 2,
}

impl ReasoningMode {
    const fn as_u8(self) -> u8 {
        match self {
            Self::None => 0,
            Self::TagBased => 1,
            Self::ToolsOnly => 2,
        }
    }
}

/// Content wrapping mode (`0` = `Plain`/`None`, `1` = `TagBased`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum ContentMode {
    /// Plain untagged content (`0`).
    #[default]
    Plain = 0,
    /// Explicitly tag-delimited content (`1`).
    TagBased = 1,
}

impl ContentMode {
    const fn as_u8(self) -> u8 {
        match self {
            Self::Plain => 0,
            Self::TagBased => 1,
        }
    }
}

/// Tool call format mode (`0` = `None`, `1` = `Json`, `2` = `TagWithJsonArgs`, `3` = `TagWithTagged`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum ToolFormatMode {
    /// No tool call format detected (`0`).
    #[default]
    None = 0,
    /// Native JSON object/array tool calls (`1`).
    Json = 1,
    /// Tag-based function wrapper with JSON arguments (`2`).
    #[expect(dead_code, reason = "Retained for TagWithJsonArgs mode parity")]
    TagWithJsonArgs = 2,
    /// Tag-constructed function and tagged argument calls (`3`).
    TagWithTagged = 3,
}

impl ToolFormatMode {
    const fn as_u8(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Json => 1,
            Self::TagWithJsonArgs => 2,
            Self::TagWithTagged => 3,
        }
    }
}

/// Position of a non-JSON tool call ID (`0` = `None`, `1` = `BeforeName`, `2` = `AfterName`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum CallIdPos {
    /// No call ID marker detected (`0`).
    #[default]
    None = 0,
    /// Call ID precedes function name (`1`).
    BeforeName = 1,
    /// Call ID follows function name (`2`).
    #[allow(dead_code, reason = "Exposed for Stage 4 chat provider")]
    AfterName = 2,
}

impl CallIdPos {
    const fn as_u8(self) -> u8 {
        match self {
            Self::None => 0,
            Self::BeforeName => 1,
            Self::AfterName => 2,
        }
    }
}

/// Reasoning tag delimiters extracted from differential probes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct ReasoningInfo {
    /// Reasoning extraction mode.
    pub(super) mode: ReasoningMode,
    /// Opening reasoning marker.
    pub(super) start: String,
    /// Closing reasoning marker.
    pub(super) end: String,
}

/// Content delimiter metadata extracted from differential probes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct ContentInfo {
    /// Content extraction mode.
    pub(super) mode: ContentMode,
    /// Opening content tag (empty if plain).
    pub(super) start: String,
    /// Closing content tag (empty if plain).
    pub(super) end: String,
    /// True if template fails when `assistant` `content` is `null` alongside `tool_calls`.
    pub(super) requires_nonnull_content: bool,
}

/// Tool section and JSON-key format metadata.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct ToolFormatInfo {
    /// Tool format mode (`None`, `Json`, `Constructed`, `TagWithJsonArgs`).
    pub(super) mode: ToolFormatMode,
    /// Outer section opening delimiter.
    pub(super) section_start: String,
    /// Outer section closing delimiter.
    pub(super) section_end: String,
    /// Per-call opening wrapper.
    pub(super) per_call_start: String,
    /// Per-call closing wrapper.
    pub(super) per_call_end: String,
    /// True when the JSON key itself is the function name.
    pub(super) fun_name_is_key: bool,
    /// True when JSON tool calls are wrapped in `[...]`.
    pub(super) tools_array_wrapped: bool,
    /// Optional wrapper trigger prefix.
    pub(super) openai_wrapper_trigger: String,
    /// Optional nested function field name.
    pub(super) function_field: String,
    /// JSON key/path for function name.
    pub(super) name_field: String,
    /// JSON key/path for arguments.
    pub(super) args_field: String,
    /// JSON key for caller-supplied call ID.
    pub(super) id_field: String,
    /// JSON key for template-generated sequential call ID.
    pub(super) gen_id_field: String,
    /// Observed JSON field key order.
    pub(super) parameter_order: Vec<String>,
}

/// Function tag delimiters for constructed / tag-wrapped tool calls.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct ToolFunctionInfo {
    /// Prefix before function name.
    pub(super) name_prefix: String,
    /// Suffix after function name.
    pub(super) name_suffix: String,
    /// Separator between function name suffix and arguments.
    pub(super) args_separator: String,
    /// Closing tag after function arguments.
    pub(super) close: String,
}

/// Argument tag delimiters for constructed tool calls.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct ToolArgumentsInfo {
    /// Prefix opening the argument block.
    pub(super) start: String,
    /// Suffix closing the argument block.
    pub(super) end: String,
    /// Prefix before each parameter name.
    pub(super) name_prefix: String,
    /// Suffix after each parameter name.
    pub(super) name_suffix: String,
    /// Prefix before each parameter value.
    pub(super) value_prefix: String,
    /// Suffix after each parameter value.
    pub(super) value_suffix: String,
    /// Separator between consecutive arguments.
    pub(super) separator: String,
    /// Whether whitespace is permitted between argument tags.
    pub(super) tolerate_intertag_whitespace: bool,
}

/// Non-JSON tool call ID marker metadata.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct ToolCallIdInfo {
    /// Call ID placement relative to function name.
    pub(super) pos: CallIdPos,
    /// Prefix preceding the call ID.
    pub(super) prefix: String,
    /// Suffix following the call ID.
    pub(super) suffix: String,
}

/// Aggregate tool-calling analysis metadata.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct ToolsInfo {
    /// Section and JSON format fields.
    pub(super) format: ToolFormatInfo,
    /// Function wrapper delimiters.
    pub(super) function: ToolFunctionInfo,
    /// Constructed argument delimiters.
    pub(super) arguments: ToolArgumentsInfo,
    /// Call ID delimiters.
    pub(super) call_id: ToolCallIdInfo,
}

/// Complete differential analysis output produced by [`AutoParser::analyze_template`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct AutoAnalysis {
    /// Detected user turn start marker.
    pub(super) user_start: String,
    /// Detected assistant turn start marker.
    pub(super) assistant_start: String,
    /// Whether template analysis completed cleanly.
    pub(super) analysis_complete: bool,
    /// Special marker tokens to preserve across tokenization.
    pub(super) preserved_tokens: Vec<String>,
    /// Additional assistant turn stop strings not covered by `eos_token`.
    pub(super) additional_stops: Vec<String>,
    /// Probed Jinja capabilities.
    pub(super) jinja_caps: JinjaCaps,
    /// Extracted reasoning configuration.
    pub(super) reasoning: ReasoningInfo,
    /// Extracted content configuration.
    pub(super) content: ContentInfo,
    /// Extracted tool-calling configuration.
    pub(super) tools: ToolsInfo,
}

impl AutoAnalysis {
    fn to_json(&self) -> Value {
        let f = &self.tools.format;
        let fun = &self.tools.function;
        let arg = &self.tools.arguments;
        let id = &self.tools.call_id;
        json!({
            "user_start": self.user_start,
            "assistant_start": self.assistant_start,
            "analysis_complete": self.analysis_complete,
            "preserved_tokens": self.preserved_tokens,
            "additional_stops": self.additional_stops,
            "jinja_caps": caps_to_json(self.jinja_caps),
            "reasoning": {
                "mode": self.reasoning.mode.as_u8(),
                "start": self.reasoning.start,
                "end": self.reasoning.end,
            },
            "content": {
                "mode": self.content.mode.as_u8(),
                "start": self.content.start,
                "end": self.content.end,
                "requires_nonnull_content": self.content.requires_nonnull_content,
            },
            "tools": {
                "format": {
                    "mode": f.mode.as_u8(),
                    "section_start": f.section_start,
                    "section_end": f.section_end,
                    "per_call_start": f.per_call_start,
                    "per_call_end": f.per_call_end,
                    "fun_name_is_key": f.fun_name_is_key,
                    "tools_array_wrapped": f.tools_array_wrapped,
                    "openai_wrapper_trigger": f.openai_wrapper_trigger,
                    "function_field": f.function_field,
                    "name_field": f.name_field,
                    "args_field": f.args_field,
                    "id_field": f.id_field,
                    "gen_id_field": f.gen_id_field,
                    "parameter_order": f.parameter_order,
                },
                "function": {
                    "name_prefix": fun.name_prefix,
                    "name_suffix": fun.name_suffix,
                    "args_separator": fun.args_separator,
                    "close": fun.close,
                },
                "arguments": {
                    "start": arg.start,
                    "end": arg.end,
                    "name_prefix": arg.name_prefix,
                    "name_suffix": arg.name_suffix,
                    "value_prefix": arg.value_prefix,
                    "value_suffix": arg.value_suffix,
                    "separator": arg.separator,
                    "tolerate_intertag_whitespace": arg.tolerate_intertag_whitespace,
                },
                "call_id": {
                    "pos": id.pos.as_u8(),
                    "prefix": id.prefix,
                    "suffix": id.suffix,
                },
            },
        })
    }
}

pub(super) fn caps_to_json(c: JinjaCaps) -> Value {
    json!({
        "supports_tools": c.supports_tools,
        "supports_tool_calls": c.supports_tool_calls,
        "supports_system_role": c.supports_system_role,
        "supports_parallel_tool_calls": c.supports_parallel_tool_calls,
        "supports_preserve_reasoning": c.supports_preserve_reasoning,
        "supports_reasoning_effort": c.supports_reasoning_effort,
        "supports_string_content": c.supports_string_content,
        "supports_typed_content": c.supports_typed_content,
        "supports_object_arguments": c.supports_object_arguments,
    })
}

// ---------------------------------------------------------------------------
// Differential template analysis implementation
// ---------------------------------------------------------------------------

const PROBE_USER_TEXT: &str = "Hello, please help me.";
const PROBE_USER_A: &str = "ALPHA_USER_FIRST";
const PROBE_USER_B: &str = "BETA_USR_SECOND";
const PROBE_REASON_TEXT: &str = "ALPHA_REASON_FIRST";
const PROBE_REASON_ALT: &str = "BETA_THOUGHT_TWO";
const PROBE_CONTENT_TEXT: &str = "ALPHA_CONTENT_FIRST";
const PROBE_CONTENT_ALT: &str = "BETA_TEXT_SECOND";
const PROBE_TOOL_REASON_TEXT: &str = "GAMMA_TOOL_REASON";
const PROBE_TOOL_REASON_ALT: &str = "DELTA_CALL_THINK";
const PROBE_FN_1: &str = "alpha_probe_fn";
const PROBE_FN_2: &str = "gamma_other_call";
const PROBE_PARAM_1: &str = "alpha_key";
const PROBE_PARAM_2: &str = "beta_arg";
const PROBE_VAL_1: &str = "first_val";
const PROBE_VAL_2: &str = "second_str";
const PROBE_VAL_3: &str = "third_item";
const PROBE_VAL_4: &str = "fourth_data";
const PROBE_ID_1: &str = "a1b2c3d4e";
const PROBE_ID_2: &str = "z9y8x7w6v";
const PROBE_ID_ALT: &str = "m5n6p7q8r";

fn parse_jval(raw: &str) -> JVal {
    parse_json_jval_bounded(raw, MAX_PARAMS_JSON_BYTES, MAX_PARAMS_JSON_DEPTH)
        .unwrap_or_else(|_| JVal::none())
}

fn probe_tools_def() -> JVal {
    parse_jval(
        r#"[{"type":"function","function":{"name":"alpha_probe_fn","description":"A test function for debugging","parameters":{"type":"object","properties":{"alpha_key":{"type":"string","description":"First parameter"},"beta_arg":{"type":"string","description":"Second parameter"}},"required":["alpha_key","beta_arg"]}}},{"type":"function","function":{"name":"gamma_other_call","description":"Another test function","parameters":{"type":"object","properties":{"alpha_key":{"type":"string","description":"First parameter"},"beta_arg":{"type":"string","description":"Second parameter"}},"required":["alpha_key","beta_arg"]}}}]"#,
    )
}

fn msg_user(content: &str) -> JVal {
    JVal::object(vec![
        (JVal::text_str("role"), JVal::bytes(b"user".to_vec(), true)),
        (
            JVal::text_str("content"),
            JVal::bytes(content.as_bytes().to_vec(), true),
        ),
    ])
}

fn msg_assistant(content: Option<&str>, reasoning: Option<&str>, tool_calls: Option<JVal>) -> JVal {
    let mut pairs = vec![(
        JVal::text_str("role"),
        JVal::bytes(b"assistant".to_vec(), true),
    )];
    pairs.push((
        JVal::text_str("content"),
        content.map_or_else(JVal::none, |s| JVal::bytes(s.as_bytes().to_vec(), true)),
    ));
    if let Some(reason) = reasoning {
        pairs.push((
            JVal::text_str("reasoning_content"),
            JVal::bytes(reason.as_bytes().to_vec(), true),
        ));
    }
    if let Some(calls) = tool_calls {
        pairs.push((JVal::text_str("tool_calls"), calls));
    }
    JVal::object(pairs)
}

fn make_call(id: &str, name: &str, args: &[(&str, &str)]) -> JVal {
    let arg_pairs: Vec<(JVal, JVal)> = args
        .iter()
        .map(|(k, v)| (JVal::text_str(*k), JVal::bytes(v.as_bytes().to_vec(), true)))
        .collect();
    JVal::object(vec![
        (
            JVal::text_str("id"),
            JVal::bytes(id.as_bytes().to_vec(), true),
        ),
        (
            JVal::text_str("type"),
            JVal::bytes(b"function".to_vec(), true),
        ),
        (
            JVal::text_str("function"),
            JVal::object(vec![
                (
                    JVal::text_str("name"),
                    JVal::bytes(name.as_bytes().to_vec(), true),
                ),
                (JVal::text_str("arguments"), JVal::object(arg_pairs)),
            ]),
        ),
    ])
}

fn first_marker(text: &str) -> Option<String> {
    segmentize_markers(text)
        .into_iter()
        .find(|s| s.seg_type == SegmentType::Marker)
        .map(|s| s.value)
}

fn last_marker(text: &str) -> Option<String> {
    segmentize_markers(text)
        .into_iter()
        .rev()
        .find(|s| s.seg_type == SegmentType::Marker)
        .map(|s| s.value)
}

fn strip_trailing_reasoning_from_prompt(prompt: &str, reasoning: &ReasoningInfo) -> String {
    let mut s = prompt.to_owned();
    for candidate in [reasoning.start.as_str(), reasoning.start.trim_end()] {
        if !candidate.is_empty()
            && let Some(pos) = s.rfind(candidate)
        {
            let tail = &s[pos + candidate.len()..];
            let trimmed_tail = tail.trim_start_matches(['\n', '\r', ' ']);
            if trimmed_tail.is_empty()
                || (!reasoning.end.is_empty()
                    && trimmed_tail
                        .trim_start_matches(&reasoning.end)
                        .trim()
                        .is_empty())
            {
                s.truncate(pos);
                return s;
            }
        }
    }
    if !reasoning.end.is_empty()
        && let Some(pos) = s.rfind(&reasoning.end)
    {
        let tail = &s[pos + reasoning.end.len()..];
        if tail.trim().is_empty() {
            s.truncate(pos);
            return s;
        }
    }
    s
}

fn detect_user_start(tmpl: &ChatTemplate) -> (String, bool) {
    let turn1_a = TemplateParams {
        messages: Some(JVal::array(vec![msg_user(PROBE_USER_A)])),
        ..TemplateParams::default()
    };
    let Some(d1) = compare_variants(
        tmpl,
        &turn1_a,
        Some(&|p| {
            p.messages = Some(JVal::array(vec![msg_user(PROBE_USER_B)]));
        }),
    ) else {
        return (String::new(), false);
    };
    let turn1_prefix = d1.diff.prefix;

    let base2 = TemplateParams {
        messages: Some(JVal::array(vec![
            msg_user(PROBE_USER_TEXT),
            msg_assistant(Some("Hi"), None, None),
        ])),
        ..TemplateParams::default()
    };
    let turn2_a = TemplateParams {
        messages: Some(JVal::array(vec![
            msg_user(PROBE_USER_TEXT),
            msg_assistant(Some("Hi"), None, None),
            msg_user(PROBE_USER_A),
        ])),
        ..TemplateParams::default()
    };
    let turn2_b = TemplateParams {
        messages: Some(JVal::array(vec![
            msg_user(PROBE_USER_TEXT),
            msg_assistant(Some("Hi"), None, None),
            msg_user(PROBE_USER_B),
        ])),
        ..TemplateParams::default()
    };

    if let (Ok(base_out), Ok(out_a), Ok(out_b)) = (
        tmpl.apply(&base2),
        tmpl.apply(&turn2_a),
        tmpl.apply(&turn2_b),
    ) {
        let d2_prefix = if let (Some(ta), Some(tb)) =
            (out_a.strip_prefix(&base_out), out_b.strip_prefix(&base_out))
        {
            diff_split_str(ta, tb).prefix
        } else {
            let da = diff_split_str(&base_out, &out_a);
            let db = diff_split_str(&base_out, &out_b);
            diff_split_str(&da.right, &db.right).prefix
        };
        if !d2_prefix.is_empty() {
            if turn1_prefix == d2_prefix || turn1_prefix.ends_with(&d2_prefix) {
                return (d2_prefix, true);
            }
            if d2_prefix.ends_with(&turn1_prefix) && !turn1_prefix.is_empty() {
                return (turn1_prefix, true);
            }
            let suffix = diff_split_str(&turn1_prefix, &d2_prefix).suffix;
            if !suffix.is_empty() {
                return (suffix, true);
            }
            return (d2_prefix, true);
        }
    }
    (turn1_prefix, true)
}

fn extract_reasoning_markers_from_pair(
    out_no: &str,
    out_a: &str,
    out_b: &str,
    reason_a: &str,
    content_needle: &str,
    gen_prompt_t: Option<&str>,
) -> Option<(String, String)> {
    let ab = diff_split_str(out_a, out_b);
    if ab.left == reason_a {
        let before = &ab.prefix;
        let open_m = last_marker(before)?;
        let open_pos = before.rfind(&open_m)?;
        let start_with_tail = before[open_pos..].to_owned();

        let after = ab.suffix.as_str();
        let close_area = after
            .find(content_needle)
            .map_or(after, |c_pos| &after[..c_pos]);
        let end_m = first_marker(close_area)?;
        if open_m == end_m {
            return None;
        }

        // Preserve newlines surrounding `<think>\n` / `\n</think>\n` only when the no-reasoning
        // turn already emitted empty `<think></think>` and the generation prompt itself appends `\n`.
        let no_vs_a = diff_split_str(out_no, out_a);
        let has_empty_pair_in_no = no_vs_a.left == end_m;
        let prompt_has_open_with_ws = start_with_tail != open_m
            && gen_prompt_t.is_some_and(|gp| gp.ends_with(&start_with_tail));
        if has_empty_pair_in_no && prompt_has_open_with_ws {
            return Some((start_with_tail, close_area.to_owned()));
        }
        return Some((open_m, end_m));
    }
    extract_reasoning_markers_around_text(out_a, reason_a)
}

fn extract_reasoning_markers_around_text(full: &str, needle: &str) -> Option<(String, String)> {
    let pos = full.find(needle)?;
    let before = &full[..pos];
    let after = &full[pos + needle.len()..];
    let start = last_marker(before)?;
    let end = first_marker(after)?;
    if start == end {
        return None;
    }
    Some((start, end))
}

fn detect_tools_only_reasoning(
    tmpl: &ChatTemplate,
    requires_nonnull: bool,
    gen_prompt_think: Option<&str>,
) -> Option<ReasoningInfo> {
    let empty = if requires_nonnull { Some("") } else { None };
    let one_tool = TemplateParams {
        messages: Some(JVal::array(vec![
            msg_user(PROBE_USER_TEXT),
            msg_assistant(
                empty,
                None,
                Some(JVal::array(vec![make_call(
                    PROBE_ID_1,
                    PROBE_FN_1,
                    &[(PROBE_PARAM_1, PROBE_VAL_1), (PROBE_PARAM_2, PROBE_VAL_2)],
                )])),
            ),
        ])),
        tools: Some(probe_tools_def()),
        ..TemplateParams::default()
    };
    let mut tool_reason_a = one_tool.clone();
    tool_reason_a.messages = Some(JVal::array(vec![
        msg_user(PROBE_USER_TEXT),
        msg_assistant(
            empty,
            Some(PROBE_TOOL_REASON_TEXT),
            Some(JVal::array(vec![make_call(
                PROBE_ID_1,
                PROBE_FN_1,
                &[(PROBE_PARAM_1, PROBE_VAL_1), (PROBE_PARAM_2, PROBE_VAL_2)],
            )])),
        ),
    ]));
    let mut tool_reason_b = one_tool.clone();
    tool_reason_b.messages = Some(JVal::array(vec![
        msg_user(PROBE_USER_TEXT),
        msg_assistant(
            empty,
            Some(PROBE_TOOL_REASON_ALT),
            Some(JVal::array(vec![make_call(
                PROBE_ID_1,
                PROBE_FN_1,
                &[(PROBE_PARAM_1, PROBE_VAL_1), (PROBE_PARAM_2, PROBE_VAL_2)],
            )])),
        ),
    ]));
    let (Ok(out_no), Ok(out_a), Ok(out_b)) = (
        tmpl.apply(&one_tool),
        tmpl.apply(&tool_reason_a),
        tmpl.apply(&tool_reason_b),
    ) else {
        return None;
    };
    if out_a == out_no {
        return None;
    }
    let (start, end) = extract_reasoning_markers_from_pair(
        &out_no,
        &out_a,
        &out_b,
        PROBE_TOOL_REASON_TEXT,
        PROBE_FN_1,
        gen_prompt_think,
    )?;
    Some(ReasoningInfo {
        mode: ReasoningMode::ToolsOnly,
        start,
        end,
    })
}

fn detect_nothink_user_tag_reasoning(tmpl: &ChatTemplate) -> Option<ReasoningInfo> {
    let nothink_user_probe = TemplateParams {
        messages: Some(JVal::array(vec![msg_user(&format!(
            "/no_think {PROBE_USER_TEXT}"
        ))])),
        add_generation_prompt: true,
        enable_thinking: true,
        ..TemplateParams::default()
    };
    if let (Ok(out_t), Ok(out_nt)) = (
        tmpl.apply(&TemplateParams {
            messages: Some(JVal::array(vec![msg_user(PROBE_USER_TEXT)])),
            add_generation_prompt: true,
            enable_thinking: true,
            ..TemplateParams::default()
        }),
        tmpl.apply(&nothink_user_probe),
    ) && out_t != out_nt
    {
        let udiff = exact_byte_diff(&out_t, &out_nt);
        if udiff.right.starts_with("</")
            && udiff.right.ends_with('>')
            && let Some(inner) = udiff
                .right
                .strip_prefix("</")
                .and_then(|s| s.strip_suffix('>'))
        {
            let open_tag = format!("<{inner}>");
            if udiff.prefix.ends_with(&open_tag) {
                return Some(ReasoningInfo {
                    mode: ReasoningMode::TagBased,
                    start: open_tag,
                    end: udiff.right,
                });
            }
        }
    }

    let gen_false = TemplateParams {
        messages: Some(JVal::array(vec![msg_user(PROBE_USER_TEXT)])),
        add_generation_prompt: false,
        enable_thinking: true,
        ..TemplateParams::default()
    };
    let plain_probe = TemplateParams {
        messages: Some(JVal::array(vec![
            msg_user(PROBE_USER_TEXT),
            msg_assistant(Some(PROBE_CONTENT_TEXT), None, None),
        ])),
        add_generation_prompt: false,
        enable_thinking: true,
        ..TemplateParams::default()
    };
    let (Some(gen_cmp), Ok(plain_out)) = (
        compare_variants(
            tmpl,
            &gen_false,
            Some(&|p| {
                p.add_generation_prompt = true;
            }),
        ),
        tmpl.apply(&plain_probe),
    ) else {
        return None;
    };
    let gen_prompt = gen_cmp.diff.right;
    let plain_pos = plain_out.find(PROBE_CONTENT_TEXT)?;
    let plain_before = &plain_out[..plain_pos];
    let plain_asst_prefix = diff_split_str(&gen_cmp.output_a, plain_before).right;
    let extra = diff_split_str(&plain_asst_prefix, &gen_prompt).right;
    let tag = first_marker(&extra)?;
    let inner = tag.strip_prefix('<')?.strip_suffix('>')?;
    if tag.starts_with("</") {
        return None;
    }
    Some(ReasoningInfo {
        mode: ReasoningMode::TagBased,
        start: tag.clone(),
        end: format!("</{inner}>"),
    })
}

fn detect_gen_prompt_delta_reasoning(
    tmpl: &ChatTemplate,
    gen_prompt_think: Option<&str>,
) -> Option<ReasoningInfo> {
    let gen_nothink_off_prompt = TemplateParams {
        messages: Some(JVal::array(vec![msg_user(PROBE_USER_TEXT)])),
        add_generation_prompt: false,
        enable_thinking: false,
        ..TemplateParams::default()
    };
    if let (Some(gp_t), Some(cmp_nt)) = (
        gen_prompt_think,
        compare_variants(
            tmpl,
            &gen_nothink_off_prompt,
            Some(&|p| p.add_generation_prompt = true),
        ),
    ) && gp_t != cmp_nt.diff.right
    {
        let tail_diff = diff_split_str(gp_t, &cmp_nt.diff.right);
        let region = if tail_diff.right.is_empty() {
            &tail_diff.left
        } else {
            &tail_diff.right
        };
        let markers: Vec<String> = segmentize_markers(region)
            .into_iter()
            .filter(|s| s.seg_type == SegmentType::Marker)
            .map(|s| s.value)
            .collect();
        if markers.len() >= 2 && markers[0] != markers[1] {
            return Some(ReasoningInfo {
                mode: ReasoningMode::TagBased,
                start: markers[0].clone(),
                end: markers[1].clone(),
            });
        }
        if markers.len() == 1
            && let Some(close_inner) = markers[0]
                .strip_prefix("</")
                .and_then(|s| s.strip_suffix('>'))
        {
            let open_tag = format!("<{close_inner}>");
            if tail_diff.prefix.ends_with(&open_tag) {
                return Some(ReasoningInfo {
                    mode: ReasoningMode::TagBased,
                    start: open_tag,
                    end: markers[0].clone(),
                });
            }
        }
    }

    detect_nothink_user_tag_reasoning(tmpl)
}

fn detect_reasoning(tmpl: &ChatTemplate, requires_nonnull: bool, has_tools: bool) -> ReasoningInfo {
    let gen_think_base = TemplateParams {
        messages: Some(JVal::array(vec![msg_user(PROBE_USER_TEXT)])),
        add_generation_prompt: false,
        enable_thinking: true,
        ..TemplateParams::default()
    };
    let gen_prompt_think = compare_variants(
        tmpl,
        &gen_think_base,
        Some(&|p| p.add_generation_prompt = true),
    )
    .map(|c| c.diff.right);

    let no_reason = TemplateParams {
        messages: Some(JVal::array(vec![
            msg_user(PROBE_USER_TEXT),
            msg_assistant(Some(PROBE_CONTENT_TEXT), None, None),
        ])),
        ..TemplateParams::default()
    };
    let mut reason_a = no_reason.clone();
    reason_a.messages = Some(JVal::array(vec![
        msg_user(PROBE_USER_TEXT),
        msg_assistant(Some(PROBE_CONTENT_TEXT), Some(PROBE_REASON_TEXT), None),
    ]));
    let mut reason_b = no_reason.clone();
    reason_b.messages = Some(JVal::array(vec![
        msg_user(PROBE_USER_TEXT),
        msg_assistant(Some(PROBE_CONTENT_TEXT), Some(PROBE_REASON_ALT), None),
    ]));
    if let (Ok(out_no), Ok(out_a), Ok(out_b)) = (
        tmpl.apply(&no_reason),
        tmpl.apply(&reason_a),
        tmpl.apply(&reason_b),
    ) && out_a != out_no
        && let Some((start, end)) = extract_reasoning_markers_from_pair(
            &out_no,
            &out_a,
            &out_b,
            PROBE_REASON_TEXT,
            PROBE_CONTENT_TEXT,
            gen_prompt_think.as_deref(),
        )
    {
        return ReasoningInfo {
            mode: ReasoningMode::TagBased,
            start,
            end,
        };
    }

    if has_tools
        && let Some(info) =
            detect_tools_only_reasoning(tmpl, requires_nonnull, gen_prompt_think.as_deref())
    {
        return info;
    }

    detect_gen_prompt_delta_reasoning(tmpl, gen_prompt_think.as_deref()).unwrap_or_default()
}

fn detect_unconditional_assistant_turn(
    tmpl: &ChatTemplate,
    base_no_gen: &TemplateParams,
    tail: &str,
) -> Option<(String, String)> {
    let plain_next = TemplateParams {
        messages: Some(JVal::array(vec![
            msg_user(PROBE_USER_TEXT),
            msg_assistant(Some("ASST_STOP_ALPHA"), None, None),
            msg_user("NEXT_TURN_PROBE"),
        ])),
        ..TemplateParams::default()
    };
    let (Ok(user_only), Ok(with_next)) = (tmpl.apply(base_no_gen), tmpl.apply(&plain_next)) else {
        return None;
    };
    let u_pos = user_only.find(PROBE_USER_TEXT)?;
    let user_tail_segs: Vec<String> =
        segmentize_markers(&user_only[u_pos + PROBE_USER_TEXT.len()..])
            .into_iter()
            .filter(|s| s.seg_type == SegmentType::Marker)
            .map(|s| s.value)
            .collect();
    let asst_tail_segs: Vec<String> = segmentize_markers(tail)
        .into_iter()
        .filter(|s| s.seg_type == SegmentType::Marker)
        .map(|s| s.value)
        .collect();
    if asst_tail_segs.len() < 2
        || user_tail_segs.is_empty()
        || user_tail_segs != asst_tail_segs[asst_tail_segs.len() - user_tail_segs.len()..]
        || !with_next.contains("NEXT_TURN_PROBE")
    {
        return None;
    }
    let turn_stop = first_marker(&user_only[u_pos + PROBE_USER_TEXT.len()..])?;
    if turn_stop.is_empty() || !tail.contains(&turn_stop) {
        return None;
    }
    let after_stop_pos = user_only.rfind(&turn_stop)?;
    let trailing = user_only[after_stop_pos + turn_stop.len()..].trim();
    if trailing.is_empty() {
        return None;
    }
    Some((trailing.to_owned(), turn_stop))
}

fn extract_gen_prompt_assistant_start(
    tmpl: &ChatTemplate,
    base_no_gen: &TemplateParams,
    reasoning: &ReasoningInfo,
) -> String {
    let think_cmp = compare_variants(
        tmpl,
        base_no_gen,
        Some(&|p| {
            p.add_generation_prompt = true;
            p.enable_thinking = true;
        }),
    );
    let nothink_cmp = compare_variants(
        tmpl,
        base_no_gen,
        Some(&|p| {
            p.add_generation_prompt = true;
            p.enable_thinking = false;
        }),
    );

    match (think_cmp, nothink_cmp) {
        (Some(t), Some(nt)) => {
            let prompt_think = t
                .output_b
                .strip_prefix(&t.output_a)
                .map_or(t.diff.right, str::to_owned);
            let prompt_nothink = nt
                .output_b
                .strip_prefix(&nt.output_a)
                .map_or(nt.diff.right, str::to_owned);
            let raw = if prompt_think == prompt_nothink {
                prompt_think
            } else {
                let common = diff_split_str(&prompt_think, &prompt_nothink).prefix;
                if common.is_empty() {
                    prompt_think
                } else {
                    common
                }
            };
            strip_trailing_reasoning_from_prompt(&raw, reasoning)
        }
        (Some(t), None) => {
            let prompt_think = t
                .output_b
                .strip_prefix(&t.output_a)
                .map_or(t.diff.right, str::to_owned);
            strip_trailing_reasoning_from_prompt(&prompt_think, reasoning)
        }
        _ => String::new(),
    }
}

fn detect_assistant_start_and_stops(
    tmpl: &ChatTemplate,
    reasoning: &ReasoningInfo,
) -> (String, Vec<String>) {
    let base_no_gen = TemplateParams {
        messages: Some(JVal::array(vec![msg_user(PROBE_USER_TEXT)])),
        add_generation_prompt: false,
        enable_thinking: true,
        ..TemplateParams::default()
    };
    let mut assistant_start = extract_gen_prompt_assistant_start(tmpl, &base_no_gen, reasoning);

    let plain_probe = TemplateParams {
        messages: Some(JVal::array(vec![
            msg_user(PROBE_USER_TEXT),
            msg_assistant(Some(PROBE_CONTENT_TEXT), None, None),
        ])),
        ..TemplateParams::default()
    };
    if let (Ok(user_out), Ok(plain_out)) = (tmpl.apply(&base_no_gen), tmpl.apply(&plain_probe))
        && let Some(c_pos) = plain_out.find(PROBE_CONTENT_TEXT)
    {
        let plain_before = &plain_out[..c_pos];
        if let Some(plain_asst_header) = plain_before.strip_prefix(&user_out) {
            let trimmed_header = plain_asst_header.trim();
            if assistant_start.is_empty() {
                if !trimmed_header.is_empty() {
                    assistant_start = trimmed_header.to_owned();
                }
            } else if !trimmed_header.is_empty()
                && assistant_start.starts_with(plain_asst_header)
                && assistant_start != plain_asst_header
            {
                assistant_start = plain_asst_header.to_owned();
            }
        }
    }
    assistant_start = assistant_start.trim().to_owned();

    let mut additional_stops = Vec::new();
    let plain_a = TemplateParams {
        messages: Some(JVal::array(vec![
            msg_user(PROBE_USER_TEXT),
            msg_assistant(Some("ASST_STOP_ALPHA"), None, None),
        ])),
        ..TemplateParams::default()
    };
    if let Ok(rendered) = tmpl.apply(&plain_a)
        && let Some(pos) = rendered.find("ASST_STOP_ALPHA")
    {
        let tail = &rendered[pos + "ASST_STOP_ALPHA".len()..];
        let mut stop_candidate = last_marker(tail);
        if !assistant_start.is_empty() && tail.ends_with(&assistant_start) {
            let before_asst = tail.strip_suffix(&assistant_start).unwrap_or(tail);
            stop_candidate = last_marker(before_asst);
        } else if let Some((trailing, turn_stop)) =
            detect_unconditional_assistant_turn(tmpl, &base_no_gen, tail)
        {
            if assistant_start.is_empty() {
                assistant_start = trailing;
            }
            stop_candidate = Some(turn_stop);
        }
        if let Some(stop_tok) = stop_candidate
            && !stop_tok.is_empty()
            && stop_tok != tmpl.eos_token
        {
            additional_stops.push(stop_tok);
        }
    }

    (assistant_start, additional_stops)
}

fn detect_content_info(
    tmpl: &ChatTemplate,
    assistant_start: &str,
    reasoning: &ReasoningInfo,
    additional_stops: &[String],
    requires_nonnull_content: bool,
) -> ContentInfo {
    let probe_a = TemplateParams {
        messages: Some(JVal::array(vec![
            msg_user(PROBE_USER_TEXT),
            msg_assistant(Some(PROBE_CONTENT_TEXT), None, None),
        ])),
        ..TemplateParams::default()
    };
    let Some(cmp) = compare_variants(
        tmpl,
        &probe_a,
        Some(&|p| {
            p.messages = Some(JVal::array(vec![
                msg_user(PROBE_USER_TEXT),
                msg_assistant(Some(PROBE_CONTENT_ALT), None, None),
            ]));
        }),
    ) else {
        return ContentInfo {
            requires_nonnull_content,
            ..ContentInfo::default()
        };
    };

    let before_content_raw = cmp.diff.prefix;
    let mut before_content = before_content_raw.clone();
    if !assistant_start.is_empty()
        && let Some(idx) = before_content.rfind(assistant_start)
    {
        before_content = before_content[idx + assistant_start.len()..].to_owned();
    }
    if !reasoning.end.is_empty()
        && let Some(idx) = before_content.rfind(&reasoning.end)
    {
        before_content = before_content[idx + reasoning.end.len()..].to_owned();
    }
    let start_trimmed = before_content.trim();
    let mut start_marker = last_marker(start_trimmed).unwrap_or_default();

    let mut after_content = cmp.diff.suffix;
    if let Some(stop) = additional_stops.first()
        && let Some(idx) = after_content.rfind(stop)
    {
        after_content = after_content[..idx].to_owned();
    }
    let end_marker = first_marker(after_content.trim()).unwrap_or_default();
    if start_marker.is_empty()
        && !end_marker.is_empty()
        && let Some(raw_m) = last_marker(before_content_raw.trim())
        && raw_m != end_marker
        && raw_m != reasoning.end
        && !additional_stops.contains(&raw_m)
    {
        start_marker = raw_m;
    }

    if !start_marker.is_empty() && !end_marker.is_empty() && start_marker != end_marker {
        ContentInfo {
            mode: ContentMode::TagBased,
            start: start_marker,
            end: end_marker,
            requires_nonnull_content,
        }
    } else {
        ContentInfo {
            mode: ContentMode::Plain,
            start: String::new(),
            end: String::new(),
            requires_nonnull_content,
        }
    }
}

fn check_requires_nonnull_content(tmpl: &ChatTemplate) -> bool {
    if !tmpl.caps.supports_tool_calls {
        return false;
    }
    let call = make_call(PROBE_ID_1, PROBE_FN_1, &[(PROBE_PARAM_1, PROBE_VAL_1)]);
    let null_params = TemplateParams {
        messages: Some(JVal::array(vec![
            msg_user(PROBE_USER_TEXT),
            msg_assistant(None, None, Some(JVal::array(vec![call.clone()]))),
        ])),
        tools: Some(probe_tools_def()),
        ..TemplateParams::default()
    };
    if tmpl.apply(&null_params).is_ok() {
        return false;
    }
    let empty_params = TemplateParams {
        messages: Some(JVal::array(vec![
            msg_user(PROBE_USER_TEXT),
            msg_assistant(Some(""), None, Some(JVal::array(vec![call]))),
        ])),
        tools: Some(probe_tools_def()),
        ..TemplateParams::default()
    };
    tmpl.apply(&empty_params).is_ok()
}

fn detect_json_call_fields(
    obj: &JsonVal,
    call_two_obj: Option<&JsonVal>,
    fmt: &mut ToolFormatInfo,
) {
    let JsonVal::Object(entries) = obj else {
        return;
    };
    for (k, v) in entries {
        fmt.parameter_order.push(k.clone());
        if v.text() == Some(PROBE_FN_1) {
            fmt.name_field.clone_from(k);
        } else if k == PROBE_FN_1 {
            fmt.fun_name_is_key = true;
            if let JsonVal::Object(inner) = v {
                for (ik, iv) in inner {
                    if iv.text() == Some(PROBE_ID_1) {
                        fmt.id_field.clone_from(ik);
                    } else if iv.get("param1").text() == Some("value1") {
                        fmt.args_field.clone_from(ik);
                    }
                }
            }
        } else if v.text() == Some(PROBE_ID_1) {
            fmt.id_field.clone_from(k);
        } else if v.get("param1").text() == Some("value1")
            || v.text()
                .is_some_and(|s| s.contains("param1") && s.contains("value1"))
        {
            fmt.args_field.clone_from(k);
        } else if let JsonVal::Object(nested) = v {
            let mut found_fn = false;
            for (nk, nv) in nested {
                if nv.text() == Some(PROBE_FN_1) {
                    fmt.function_field.clone_from(k);
                    fmt.name_field = format!("{k}.{nk}");
                    found_fn = true;
                } else if nv.get("param1").text() == Some("value1")
                    || nv
                        .text()
                        .is_some_and(|s| s.contains("param1") && s.contains("value1"))
                {
                    fmt.args_field = format!("{k}.{nk}");
                }
            }
            let _ = found_fn;
        }
        if let Some(two_v) = call_two_obj.map(|o| o.get(k))
            && v.text() == Some("0")
            && two_v.text() == Some("1")
            && fmt.id_field != *k
        {
            fmt.gen_id_field.clone_from(k);
        }
    }
}

fn extract_json_object_span(text: &str) -> Option<(usize, usize, JsonVal)> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end <= start {
        return None;
    }
    let slice = &text[start..=end];
    let parsed = JsonParser::parse_with_limit(slice, MAX_PARAMS_JSON_BYTES).ok()?;
    if matches!(parsed, JsonVal::Object(_)) {
        Some((start, end + 1, parsed))
    } else {
        None
    }
}

fn split_outer_per_call_markers(
    before_fn: &str,
    after_args: &str,
) -> (String, String, String, String) {
    let trimmed_before = before_fn.trim_start_matches(['\n', '\r', ' ', '\t']);
    let trimmed_after = after_args.trim_end_matches(['\n', '\r', ' ', '\t']);
    let before_segs = segmentize_markers(trimmed_before);
    let after_segs = segmentize_markers(trimmed_after);

    let before_markers: Vec<&str> = before_segs
        .iter()
        .filter(|s| s.seg_type == SegmentType::Marker)
        .map(|s| s.value.as_str())
        .collect();
    let after_markers: Vec<&str> = after_segs
        .iter()
        .filter(|s| s.seg_type == SegmentType::Marker)
        .map(|s| s.value.as_str())
        .collect();

    if before_markers.len() >= 2 && !after_markers.is_empty() {
        let last_open_m = before_markers[before_markers.len() - 1];
        let per_end = after_markers[after_markers.len() - 1].to_owned();
        if let Some(split_open) = trimmed_before.rfind(last_open_m)
            && let Some(split_end) = trimmed_after.rfind(&per_end)
        {
            let per_start = trimmed_before[..split_open].to_owned();
            let rem_before = trimmed_before[split_open..].to_owned();
            let rem_after = trimmed_after[..split_end].to_owned();
            return (per_start, per_end, rem_before, rem_after);
        }
    }

    if before_segs.len() >= 2
        && before_segs[0].seg_type == SegmentType::Marker
        && after_markers.len() >= 2
    {
        let first_m_len = before_segs[0].value.len();
        let tail_before = &trimmed_before[first_m_len..];
        let ws_len = tail_before.len()
            - tail_before
                .trim_start_matches(['\n', '\r', ' ', '\t'])
                .len();
        let split_open = first_m_len + ws_len;
        let per_end = after_markers[after_markers.len() - 1].to_owned();
        if let Some(split_end) = trimmed_after.rfind(&per_end) {
            let per_start = trimmed_before[..split_open].to_owned();
            let rem_before = trimmed_before[split_open..].to_owned();
            let rem_after = trimmed_after[..split_end].to_owned();
            return (per_start, per_end, rem_before, rem_after);
        }
    }

    if before_markers.len() == 1
        && trimmed_before.trim() == before_markers[0]
        && after_markers.len() == 1
        && trimmed_after.trim() == after_markers[0]
    {
        return (
            trimmed_before.to_owned(),
            trimmed_after.to_owned(),
            String::new(),
            String::new(),
        );
    }

    (
        String::new(),
        String::new(),
        trimmed_before.to_owned(),
        trimmed_after.to_owned(),
    )
}

fn split_intertag_markers(between: &str) -> (String, String, bool) {
    let segs = segmentize_markers(between);
    let markers: Vec<(usize, &str)> = segs
        .iter()
        .enumerate()
        .filter(|(_, s)| s.seg_type == SegmentType::Marker)
        .map(|(i, s)| (i, s.value.as_str()))
        .collect();
    if markers.len() >= 2 {
        let (first_idx, first_m) = markers[0];
        let (second_idx, second_m) = markers[1];
        let has_ws = (first_idx + 1..second_idx).any(|i| {
            segs[i].seg_type == SegmentType::Text && segs[i].value.chars().any(char::is_whitespace)
        });
        return (first_m.to_owned(), second_m.to_owned(), has_ws);
    }
    if markers.len() == 1 {
        let (idx, m) = markers[0];
        let trailing = segs[idx + 1..]
            .iter()
            .map(|s| s.value.as_str())
            .collect::<String>();
        if trailing.trim().is_empty() {
            return (between.to_owned(), String::new(), false);
        }
        return (m.to_owned(), trailing, false);
    }
    (between.to_owned(), String::new(), false)
}

#[expect(
    clippy::too_many_lines,
    reason = "Keeps cohesive differential tool probes together without fragmenting probe state."
)]
fn detect_tools_info(
    tmpl: &ChatTemplate,
    reasoning: &ReasoningInfo,
    additional_stops: &[String],
    requires_nonnull: bool,
) -> ToolsInfo {
    if !tmpl.caps.supports_tool_calls {
        return ToolsInfo::default();
    }
    let tools_def = probe_tools_def();
    let empty = if requires_nonnull { Some("") } else { None };

    let base_params = TemplateParams {
        messages: Some(JVal::array(vec![
            msg_user(PROBE_USER_TEXT),
            msg_assistant(Some(""), None, None),
        ])),
        tools: Some(tools_def.clone()),
        ..TemplateParams::default()
    };
    let one_call_params = |id: &str, name: &str, args: &[(&str, &str)]| TemplateParams {
        messages: Some(JVal::array(vec![
            msg_user(PROBE_USER_TEXT),
            msg_assistant(
                empty,
                None,
                Some(JVal::array(vec![make_call(id, name, args)])),
            ),
        ])),
        tools: Some(tools_def.clone()),
        ..TemplateParams::default()
    };

    let p_one = one_call_params(
        PROBE_ID_1,
        PROBE_FN_1,
        &[(PROBE_PARAM_1, PROBE_VAL_1), (PROBE_PARAM_2, PROBE_VAL_2)],
    );
    let p_alt_fn = one_call_params(
        PROBE_ID_1,
        PROBE_FN_2,
        &[(PROBE_PARAM_1, PROBE_VAL_1), (PROBE_PARAM_2, PROBE_VAL_2)],
    );
    let p_arg1 = one_call_params(PROBE_ID_1, PROBE_FN_1, &[(PROBE_PARAM_1, PROBE_VAL_1)]);
    let p_arg2 = one_call_params(PROBE_ID_1, PROBE_FN_1, &[(PROBE_PARAM_2, PROBE_VAL_2)]);
    let p_noargs = one_call_params(PROBE_ID_1, PROBE_FN_1, &[]);
    let p_diff_id = one_call_params(
        PROBE_ID_ALT,
        PROBE_FN_1,
        &[(PROBE_PARAM_1, PROBE_VAL_1), (PROBE_PARAM_2, PROBE_VAL_2)],
    );

    let Ok(out_one) = tmpl.apply(&p_one) else {
        return ToolsInfo::default();
    };
    let Ok(out_base) = tmpl.apply(&base_params) else {
        return ToolsInfo::default();
    };
    let Some(fn_diff) = compare_variants(
        tmpl,
        &p_one,
        Some(&|p| {
            p.messages = p_alt_fn.messages.clone();
        }),
    ) else {
        return ToolsInfo::default();
    };
    if !fn_diff.diff.left.contains(PROBE_FN_1) || !fn_diff.diff.right.contains(PROBE_FN_2) {
        return ToolsInfo::default();
    }

    let base_vs_one = diff_split_str(&out_base, &out_one);
    let mut region_one = base_vs_one.right;
    if !reasoning.start.is_empty() && !reasoning.end.is_empty() {
        let empty_reason = format!("{}{}", reasoning.start, reasoning.end);
        if let Some(stripped) = region_one.strip_prefix(&empty_reason) {
            region_one = stripped.to_owned();
        }
    }
    for stop in additional_stops {
        if let Some(stripped) = region_one.strip_suffix(stop) {
            region_one = stripped.to_owned();
        } else if let Some(pos) = region_one.rfind(stop)
            && region_one[pos + stop.len()..].trim().is_empty()
        {
            region_one.truncate(pos);
        }
    }

    let mut info = ToolsInfo::default();
    let mut call_two_region: Option<String> = None;

    if tmpl.caps.supports_parallel_tool_calls {
        let p_two = TemplateParams {
            messages: Some(JVal::array(vec![
                msg_user(PROBE_USER_TEXT),
                msg_assistant(
                    empty,
                    None,
                    Some(JVal::array(vec![
                        make_call(
                            PROBE_ID_1,
                            PROBE_FN_1,
                            &[(PROBE_PARAM_1, PROBE_VAL_1), (PROBE_PARAM_2, PROBE_VAL_2)],
                        ),
                        make_call(
                            PROBE_ID_2,
                            PROBE_FN_2,
                            &[(PROBE_PARAM_1, PROBE_VAL_3), (PROBE_PARAM_2, PROBE_VAL_4)],
                        ),
                    ])),
                ),
            ])),
            tools: Some(tools_def),
            ..TemplateParams::default()
        };
        if let Ok(out_two) = tmpl.apply(&p_two) {
            let par_diff = exact_byte_diff(&out_one, &out_two);
            let before_first_fn = &fn_diff.diff.prefix;
            let base_prefix = &base_vs_one.prefix;
            let between_base_and_fn = before_first_fn
                .strip_prefix(base_prefix)
                .unwrap_or(before_first_fn);
            if let Some(first_fn_pos) = par_diff.prefix.find(PROBE_FN_1) {
                let _ = first_fn_pos;
            }
            let _ = between_base_and_fn;
            call_two_region = Some(par_diff.right);
        }
    }

    // Check if single call region contains a JSON object representing the tool call.
    let single_body_candidate = region_one.clone();
    if let Some((j_start, j_end, parsed_call)) = extract_json_object_span(&single_body_candidate) {
        let before_brace = &single_body_candidate[..j_start];
        let after_brace = &single_body_candidate[j_end..];
        let trimmed_before = before_brace.trim_end();
        if trimmed_before.ends_with('[') && after_brace.trim_start().starts_with(']') {
            info.format.tools_array_wrapped = true;
            let sec_s = trimmed_before.strip_suffix('[').unwrap_or_default().trim();
            let sec_e = after_brace
                .trim_start()
                .strip_prefix(']')
                .unwrap_or_default()
                .trim();
            info.format.section_start = sec_s.to_owned();
            info.format.section_end = sec_e.to_owned();
        } else {
            info.format.per_call_start = before_brace.to_owned();
            info.format.per_call_end = after_brace.to_owned();
        }

        let two_obj = call_two_region
            .as_deref()
            .and_then(extract_json_object_span)
            .map(|(_, _, val)| val);
        detect_json_call_fields(&parsed_call, two_obj.as_ref(), &mut info.format);
        if !info.format.name_field.is_empty() || info.format.fun_name_is_key {
            info.format.mode = ToolFormatMode::Json;
            return info;
        }
    }

    // Constructed / tag-based tool format.
    let Some(fn_pos_in_region) = region_one.find(PROBE_FN_1) else {
        return info;
    };
    let raw_before_fn = &region_one[..fn_pos_in_region];
    let after_fn = &region_one[fn_pos_in_region + PROBE_FN_1.len()..];

    // Check section wrappers when parallel calls exist.
    let mut per_call_before = raw_before_fn.to_owned();
    let mut after_call_area = after_fn.to_owned();
    if let Some(ref second_call_diff) = call_two_region {
        // If `region_one` starts with a marker that is NOT in `second_call_diff`, it's `section_start`.
        let reg_segs = segmentize_markers(raw_before_fn);
        if let Some(first_seg) = reg_segs.first()
            && first_seg.seg_type == SegmentType::Marker
            && !second_call_diff.contains(&first_seg.value)
        {
            info.format.section_start.clone_from(&first_seg.value);
            per_call_before = raw_before_fn[first_seg.value.len()..]
                .trim_start_matches(['\n', '\r'])
                .to_owned();
        }
        let end_segs = segmentize_markers(&after_call_area);
        if let Some(last_seg) = end_segs.last()
            && last_seg.seg_type == SegmentType::Marker
            && !second_call_diff.contains(&last_seg.value)
            && let Some(pos) = after_call_area.rfind(&last_seg.value)
        {
            info.format.section_end.clone_from(&last_seg.value);
            after_call_area = after_call_area[..pos]
                .trim_end_matches(['\n', '\r'])
                .to_owned();
        }
    }

    // Check non-JSON call_id before or after function name.
    if let Ok(out_diff_id) = tmpl.apply(&p_diff_id)
        && out_diff_id != out_one
        && let Some(id_pos) = per_call_before.find(PROBE_ID_1)
    {
        info.call_id.pos = CallIdPos::BeforeName;
        let id_before = &per_call_before[..id_pos];
        let id_after = &per_call_before[id_pos + PROBE_ID_1.len()..];
        if let Some(m) = last_marker(id_before) {
            info.call_id.prefix = m;
        }
        if let Some(m) = first_marker(id_after) {
            info.call_id.suffix = m;
        }
        per_call_before = id_before.to_owned();
    }

    // Analyze arguments using 0-arg, arg1, arg2 probes.
    let (Ok(out_arg1), Ok(out_arg2), Ok(out_noargs)) = (
        tmpl.apply(&p_arg1),
        tmpl.apply(&p_arg2),
        tmpl.apply(&p_noargs),
    ) else {
        return info;
    };

    let arg1_only = exact_byte_diff(&out_noargs, &out_arg1).right;
    let arg2_only = exact_byte_diff(&out_noargs, &out_arg2).right;
    let arg1_vs_2 = exact_byte_diff(&out_arg1, &out_arg2);
    let _ = arg1_vs_2;

    let before_args = until_common_prefix(&after_call_area, &arg1_only, &arg2_only);
    let after_args = after_common_suffix(&after_call_area, &arg1_only, &arg2_only);

    // Separate function name suffix and args_separator.
    let mut name_suffix = String::new();
    let mut args_separator = String::new();
    let before_segs = segmentize_markers(&before_args);
    if !before_segs.is_empty() {
        if before_segs[0].seg_type == SegmentType::Text && before_segs[0].value.starts_with('>') {
            if before_args[1..].trim().is_empty() {
                name_suffix.clone_from(&before_args);
            } else {
                name_suffix = ">".to_owned();
                args_separator = before_args[1..].to_owned();
            }
        } else if before_segs[0].seg_type == SegmentType::Marker {
            args_separator.clone_from(&before_segs[0].value);
        } else {
            args_separator.clone_from(&before_args);
        }
    }
    if name_suffix.is_empty() && per_call_before.contains('<') && !per_call_before.ends_with('>') {
        if before_args
            .strip_prefix('>')
            .is_some_and(|rest| rest.trim().is_empty())
        {
            name_suffix.clone_from(&before_args);
            args_separator.clear();
        } else {
            name_suffix = ">".to_owned();
            args_separator = before_args
                .strip_prefix('>')
                .unwrap_or(&before_args)
                .to_owned();
        }
    }
    let (per_call_start, per_call_end, fn_name_prefix, fn_close) =
        split_outer_per_call_markers(&per_call_before, &after_args);

    info.format.mode = ToolFormatMode::TagWithTagged;
    info.format.per_call_start = per_call_start;
    info.format.per_call_end = per_call_end;
    info.function.name_prefix = fn_name_prefix;
    info.function.name_suffix = name_suffix;
    info.function.args_separator = args_separator.trim().to_owned();
    info.function.close = fn_close;

    // Extract argument name/value prefixes and suffixes from `arg1_only`.
    if let (Some(p1_pos), Some(v1_pos)) =
        (arg1_only.find(PROBE_PARAM_1), arg1_only.find(PROBE_VAL_1))
        && v1_pos > p1_pos
    {
        info.arguments.name_prefix = arg1_only[..p1_pos]
            .trim_start_matches(['\n', '\r'])
            .to_owned();
        let between_name_and_val = &arg1_only[p1_pos + PROBE_PARAM_1.len()..v1_pos];
        let (n_suf, v_pre, intertag_ws) = split_intertag_markers(between_name_and_val);
        info.arguments.name_suffix = n_suf;
        info.arguments.value_prefix = v_pre;
        info.arguments.tolerate_intertag_whitespace = intertag_ws;

        let after_val = &arg1_only[v1_pos + PROBE_VAL_1.len()..];
        let first_m = first_marker(after_val).unwrap_or_else(|| after_val.to_owned());
        if !after_val.starts_with(&first_m)
            && info.arguments.value_prefix.is_empty()
            && after_val
                .find(&first_m)
                .is_some_and(|p| after_val[..p].trim().is_empty())
        {
            info.arguments.value_suffix = after_val.to_owned();
        } else {
            info.arguments.value_suffix = first_m;
        }
    }

    // Detect argument separator between arg1 and arg2.
    let arg_diff = exact_byte_diff(&out_arg1, &out_one);
    if !info.arguments.name_prefix.is_empty()
        && let Some(next_arg_pos) = arg_diff.right.find(&info.arguments.name_prefix)
    {
        let sep_candidate = arg_diff.right[..next_arg_pos].trim();
        if !sep_candidate.is_empty() {
            info.arguments.separator = sep_candidate.to_owned();
        }
    }

    info
}

fn collect_preserved_tokens(analysis: &AutoAnalysis, eos_token: &str) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    let mut push_markers = |text: &str| {
        for seg in segmentize_markers(text) {
            if seg.seg_type == SegmentType::Marker
                && !seg.value.is_empty()
                && seg.value != eos_token
                && seen.insert(seg.value.clone())
            {
                out.push(seg.value);
            }
        }
    };
    push_markers(&analysis.user_start);
    push_markers(&analysis.assistant_start);
    push_markers(&analysis.reasoning.start);
    push_markers(&analysis.reasoning.end);
    push_markers(&analysis.content.start);
    push_markers(&analysis.content.end);
    push_markers(&analysis.tools.format.section_start);
    push_markers(&analysis.tools.format.section_end);
    push_markers(&analysis.tools.format.per_call_start);
    push_markers(&analysis.tools.format.per_call_end);
    push_markers(&analysis.tools.function.name_prefix);
    push_markers(&analysis.tools.function.name_suffix);
    push_markers(&analysis.tools.function.args_separator);
    push_markers(&analysis.tools.function.close);
    push_markers(&analysis.tools.arguments.start);
    push_markers(&analysis.tools.arguments.end);
    push_markers(&analysis.tools.arguments.name_prefix);
    push_markers(&analysis.tools.arguments.name_suffix);
    push_markers(&analysis.tools.arguments.value_prefix);
    push_markers(&analysis.tools.arguments.value_suffix);
    push_markers(&analysis.tools.arguments.separator);
    push_markers(&analysis.tools.call_id.prefix);
    push_markers(&analysis.tools.call_id.suffix);
    for stop in &analysis.additional_stops {
        push_markers(stop);
    }
    out
}

// ---------------------------------------------------------------------------
// AutoParser: analyze_template & build_parser
// ---------------------------------------------------------------------------

/// Generation parameters used by [`AutoParser::build_parser`].
#[derive(Clone, Debug)]
pub(super) struct GenerationParams {
    /// Tools JSON array (`JsonVal`).
    pub(super) tools: Option<JsonVal>,
    /// Optional conversation messages (`JsonVal`).
    pub(super) messages: Option<JsonVal>,
    /// Optional structured output JSON schema (`JsonVal`).
    pub(super) json_schema: Option<JsonVal>,
    /// Reasoning extraction format (`0` = None, `1..=3` = Auto/DeepSeek).
    pub(super) reasoning_format: i32,
    /// Tool choice constraint (`0` = Auto, `1` = Required, `2` = None).
    pub(super) tool_choice: i32,
    /// Whether parallel tool calls are allowed.
    pub(super) parallel_tool_calls: bool,
    /// Whether thinking is enabled.
    pub(super) enable_thinking: bool,
    /// Whether generation prompt was appended.
    pub(super) add_generation_prompt: bool,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            tools: None,
            messages: None,
            json_schema: None,
            reasoning_format: 1,
            tool_choice: 0,
            parallel_tool_calls: false,
            enable_thinking: true,
            add_generation_prompt: true,
        }
    }
}

/// Stateful auto-parser analyzer and PEG parser builder mirroring `autoparser::autoparser`.
#[derive(Clone, Debug, Default)]
pub(super) struct AutoParser {
    analysis: AutoAnalysis,
}

impl AutoParser {
    /// Returns the last analyzed [`AutoAnalysis`].
    pub(super) const fn analysis(&self) -> &AutoAnalysis {
        &self.analysis
    }

    /// Runs differential template analysis on `tmpl` and stores the resulting [`AutoAnalysis`].
    pub(super) fn analyze_template(&mut self, tmpl: &ChatTemplate) {
        let caps = tmpl.original_caps();
        let (raw_user_start, user_ok) = detect_user_start(tmpl);
        let requires_nonnull = check_requires_nonnull_content(tmpl);
        let reasoning = detect_reasoning(tmpl, requires_nonnull, caps.supports_tool_calls);
        let (assistant_start, additional_stops) =
            detect_assistant_start_and_stops(tmpl, &reasoning);
        let mut user_start = raw_user_start;
        for stop in &additional_stops {
            if let Some(stripped) = user_start.strip_prefix(stop) {
                user_start = stripped.to_owned();
            }
        }
        user_start = user_start.trim().to_owned();
        let content = detect_content_info(
            tmpl,
            &assistant_start,
            &reasoning,
            &additional_stops,
            requires_nonnull,
        );
        let tools = detect_tools_info(tmpl, &reasoning, &additional_stops, requires_nonnull);

        let mut analysis = AutoAnalysis {
            user_start,
            assistant_start,
            analysis_complete: user_ok
                && (!caps.supports_tool_calls || tools.format.mode != ToolFormatMode::None),
            preserved_tokens: Vec::new(),
            additional_stops,
            jinja_caps: caps,
            reasoning,
            content,
            tools,
        };
        analysis.preserved_tokens = collect_preserved_tokens(&analysis, &tmpl.eos_token);
        self.analysis = analysis;
    }

    /// Builds a [`ChatArena`] parser matching `self.analysis`, `inputs`, and `generation_prompt`.
    #[expect(
        clippy::too_many_lines,
        reason = "Coordinates reasoning, tools, schema, and open-thinking fallback assembly"
    )]
    pub(super) fn build_parser(
        &self,
        inputs: &GenerationParams,
        generation_prompt: &str,
    ) -> Result<ChatArena> {
        let mut builder = ChatPegBuilder::new();
        let mut parts: Vec<ParserId> = Vec::new();

        // Prepare tool and post-reasoning body parsers first so already-open reasoning
        // can branch cleanly on `</think>` closing tags, direct tool triggers, or EOF.
        let has_tools = inputs.tools.as_ref().is_some_and(|t| !t.array().is_empty())
            && inputs.tool_choice != 2
            && self.analysis.tools.format.mode != ToolFormatMode::None;
        let force_tools = inputs.tool_choice == 1;

        let mut tool_triggers = Vec::new();
        let f = &self.analysis.tools.format;
        let sec_start_trig = f.section_start.trim();
        let per_call_trig = f.per_call_start.trim();
        let fn_pfx_trig = self.analysis.tools.function.name_prefix.trim();
        if !sec_start_trig.is_empty() {
            tool_triggers.push(sec_start_trig.to_owned());
        } else if !per_call_trig.is_empty() {
            tool_triggers.push(per_call_trig.to_owned());
            if per_call_trig == "<tool_call>" && fn_pfx_trig == "<function=" {
                tool_triggers.push("<function=".to_owned());
            }
        } else if !fn_pfx_trig.is_empty() {
            tool_triggers.push(fn_pfx_trig.to_owned());
        }

        let bare_json_tools = has_tools
            && tool_triggers.is_empty()
            && !force_tools
            && self.analysis.content.start.is_empty();
        let tool_parser = if has_tools {
            let tools_val = inputs
                .tools
                .as_ref()
                .ok_or_else(|| invalid("missing tools value"))?;
            Some(self.build_tools_parser(
                &mut builder,
                tools_val,
                inputs.parallel_tool_calls,
                force_tools || bare_json_tools,
            )?)
        } else {
            None
        };

        let mut body_parts: Vec<ParserId> = Vec::new();
        if let Some(tools_p) = tool_parser {
            if !self.analysis.content.start.is_empty() && !self.analysis.content.end.is_empty() {
                let c_open = builder.inner_mut().literal(&self.analysis.content.start);
                let until_end = builder.inner_mut().until(&self.analysis.content.end);
                let c_body = builder.content(until_end);
                let c_close = builder.inner_mut().literal(&self.analysis.content.end);
                let wrapped = builder.inner_mut().sequence(vec![c_open, c_body, c_close]);
                let opt_content = builder.inner_mut().optional(wrapped);
                body_parts.push(opt_content);
                body_parts.push(tools_p);
            } else if bare_json_tools {
                let fallback = if let Some(ref schema) = inputs.json_schema
                    && !matches!(schema, JsonVal::Null)
                {
                    builder.schema_content_parser(schema)?
                } else {
                    let rest = builder.inner_mut().rest();
                    builder.content(rest)
                };
                body_parts.push(builder.inner_mut().or(tools_p, fallback));
            } else if !tool_triggers.is_empty() {
                let until_trig = builder.inner_mut().until_one_of(tool_triggers.clone());
                let c_body = builder.content(until_trig);
                body_parts.push(c_body);
                body_parts.push(tools_p);
            } else {
                body_parts.push(tools_p);
            }
        } else if !self.analysis.content.start.is_empty() && !self.analysis.content.end.is_empty() {
            let c_open = builder.inner_mut().literal(&self.analysis.content.start);
            let until_end = builder.inner_mut().until(&self.analysis.content.end);
            let c_body = builder.content(until_end);
            let c_close = builder.inner_mut().literal(&self.analysis.content.end);
            body_parts.push(builder.inner_mut().sequence(vec![c_open, c_body, c_close]));
        } else if let Some(ref schema) = inputs.json_schema
            && !matches!(schema, JsonVal::Null)
        {
            body_parts.push(builder.schema_content_parser(schema)?);
        } else {
            let rest = builder.inner_mut().rest();
            body_parts.push(builder.content(rest));
        }

        let after_body = if body_parts.len() == 1 {
            body_parts[0]
        } else {
            builder.inner_mut().sequence(body_parts.clone())
        };

        // 1. Optional or prefilled reasoning segment (kept in `content` when format == 0).
        if self.analysis.reasoning.mode != ReasoningMode::None
            && !self.analysis.reasoning.end.trim().is_empty()
        {
            let start_tag = self.analysis.reasoning.start.trim();
            let end_tag = self.analysis.reasoning.end.trim();
            let prompt_trimmed = generation_prompt.trim_end();
            let already_open = !start_tag.is_empty()
                && prompt_trimmed.ends_with(start_tag)
                && !prompt_trimmed.ends_with(end_tag);
            let already_closed = prompt_trimmed.ends_with(end_tag);
            let constructed_tools = has_tools
                && self.analysis.tools.format.mode == ToolFormatMode::TagWithTagged
                && !tool_triggers.is_empty();

            if inputs.reasoning_format == 0 {
                if already_open {
                    let until_end = if constructed_tools {
                        let mut delims = vec![end_tag.to_owned()];
                        delims.extend(tool_triggers);
                        builder.inner_mut().until_one_of(delims)
                    } else {
                        builder.inner_mut().until(end_tag)
                    };
                    let close = builder.inner_mut().literal(end_tag);
                    let closed_head = builder.inner_mut().sequence(vec![until_end, close]);
                    let closed_content = builder.content(closed_head);
                    let opt_closed = if !force_tools && inputs.json_schema.is_none() {
                        builder.inner_mut().optional(closed_content)
                    } else {
                        closed_content
                    };
                    parts.push(opt_closed);
                } else if !already_closed && !start_tag.is_empty() {
                    let open = builder.inner_mut().literal(start_tag);
                    let until_end = builder.inner_mut().until(end_tag);
                    let close = builder.inner_mut().literal(end_tag);
                    let inner = builder.inner_mut().sequence(vec![open, until_end, close]);
                    let block = builder.content(inner);
                    parts.push(builder.inner_mut().optional(block));
                }
            } else if already_open {
                let until_end = if constructed_tools {
                    let mut delims = vec![end_tag.to_owned()];
                    delims.extend(tool_triggers);
                    builder.inner_mut().until_one_of(delims)
                } else {
                    builder.inner_mut().until(end_tag)
                };
                let body = builder.reasoning(until_end);
                let r_block = builder.reasoning_block(body);

                let close = builder.inner_mut().literal(end_tag);
                let sp = builder.inner_mut().space();
                let closed_branch = builder.inner_mut().sequence(vec![close, sp, after_body]);

                let after_reason = if constructed_tools && let Some(tools_p) = tool_parser {
                    let branch = builder.inner_mut().choice(vec![closed_branch, tools_p]);
                    if !force_tools && inputs.json_schema.is_none() {
                        builder.inner_mut().optional(branch)
                    } else {
                        branch
                    }
                } else if !force_tools && inputs.json_schema.is_none() {
                    builder.inner_mut().optional(closed_branch)
                } else {
                    closed_branch
                };

                let full = builder.inner_mut().sequence(vec![r_block, after_reason]);
                builder.set_root(full);
                return builder.build();
            } else if !already_closed && !start_tag.is_empty() {
                let open = builder.inner_mut().literal(start_tag);
                let sp1 = builder.inner_mut().space();
                let until_end = builder.inner_mut().until(end_tag);
                let body = builder.reasoning(until_end);
                let close = builder.inner_mut().literal(end_tag);
                let sp2 = builder.inner_mut().space();
                let inner = builder
                    .inner_mut()
                    .sequence(vec![open, sp1, body, close, sp2]);
                let block = builder.reasoning_block(inner);
                parts.push(builder.inner_mut().optional(block));
            }
        }

        parts.extend(body_parts);

        let root = if parts.len() == 1 {
            parts[0]
        } else {
            builder.inner_mut().sequence(parts)
        };
        builder.set_root(root);
        builder.build()
    }

    fn build_tools_parser(
        &self,
        builder: &mut ChatPegBuilder,
        tools_val: &JsonVal,
        parallel: bool,
        force: bool,
    ) -> Result<ParserId> {
        let f = &self.analysis.tools.format;
        match f.mode {
            ToolFormatMode::Json => builder.standard_json_tools(&JsonToolsOptions {
                section_start: f.section_start.trim(),
                section_end: f.section_end.trim(),
                tools: tools_val,
                parallel_tool_calls: parallel,
                force_tool_calls: force,
                per_call_start: f.per_call_start.trim(),
                per_call_end: f.per_call_end.trim(),
                fun_name_is_key: f.fun_name_is_key,
                tools_array_wrapped: f.tools_array_wrapped,
                name_field: &f.name_field,
                args_field: &f.args_field,
                id_field: &f.id_field,
                gen_id_field: &f.gen_id_field,
                parameter_order: &f.parameter_order,
                allow_raw_arguments: false,
            }),
            ToolFormatMode::TagWithTagged | ToolFormatMode::TagWithJsonArgs => {
                let fun = &self.analysis.tools.function;
                let arg = &self.analysis.tools.arguments;
                let mut markers = BTreeMap::new();
                markers.insert("section_start".to_owned(), f.section_start.clone());
                markers.insert("section_end".to_owned(), f.section_end.clone());
                markers.insert("per_call_start".to_owned(), f.per_call_start.clone());
                markers.insert("per_call_end".to_owned(), f.per_call_end.clone());
                markers.insert("func_name_prefix".to_owned(), fun.name_prefix.clone());
                markers.insert("func_name_suffix".to_owned(), fun.name_suffix.clone());
                markers.insert("args_separator".to_owned(), fun.args_separator.clone());
                markers.insert("func_close".to_owned(), fun.close.clone());
                markers.insert("args_start".to_owned(), arg.start.clone());
                markers.insert("args_end".to_owned(), arg.end.clone());
                markers.insert("arg_name_prefix".to_owned(), arg.name_prefix.clone());
                markers.insert("arg_name_suffix".to_owned(), arg.name_suffix.clone());
                markers.insert("arg_value_prefix".to_owned(), arg.value_prefix.clone());
                markers.insert("arg_value_suffix".to_owned(), arg.value_suffix.clone());
                markers.insert("arg_separator".to_owned(), arg.separator.clone());
                builder.standard_constructed_tools_with_ws(
                    &markers,
                    tools_val,
                    parallel,
                    force,
                    arg.tolerate_intertag_whitespace,
                )
            }
            ToolFormatMode::None => Ok(builder.inner_mut().eps()),
        }
    }
}

// ---------------------------------------------------------------------------
// Wire protocol validation & execution
// ---------------------------------------------------------------------------

fn require(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(invalid(message))
    }
}

fn bounded_str<'a>(val: &'a Value, max_bytes: usize, label: &str) -> Result<&'a str> {
    let s = val
        .as_str()
        .ok_or_else(|| invalid(format!("{label} must be string")))?;
    require(s.len() <= max_bytes, &format!("{label} exceeds bound"))?;
    Ok(s)
}

fn bounded_int(val: &Value, low: i64, high: i64) -> Result<i64> {
    let n = if let Some(u) = val.as_u64() {
        i64::try_from(u).map_err(|_| invalid("integer exceeds bound"))?
    } else if let Some(i) = val.as_i64() {
        i
    } else {
        return Err(invalid("argument must be integer"));
    };
    require(n >= low && n <= high, "integer outside bound")?;
    Ok(n)
}

fn parse_template_params(val: &Value) -> Result<TemplateParams> {
    let raw = bounded_str(val, MAX_PARAMS_JSON_BYTES, "chat template params JSON")?;
    let parsed = parse_json_jval_bounded(raw, MAX_PARAMS_JSON_BYTES, MAX_PARAMS_JSON_DEPTH)
        .map_err(|err| invalid(format!("invalid chat template params JSON: {err}")))?;
    let Some(pairs) = parsed.as_object_pairs() else {
        return Err(invalid("chat template params must be object"));
    };
    let mut out = TemplateParams::default();
    let json_val: Value = serde_json::from_str(raw)?;
    let json_obj = json_val
        .as_object()
        .ok_or_else(|| invalid("chat template params must be object"))?;
    for (key, _) in &pairs {
        let key_bytes = key.raw_bytes();
        let key_str = std::str::from_utf8(&key_bytes)?;
        match key_str {
            "messages" | "tools" | "extra_context" => {}
            "add_generation_prompt" => {
                let flag = json_obj
                    .get("add_generation_prompt")
                    .and_then(Value::as_bool)
                    .ok_or_else(|| invalid("chat template flag must be boolean"))?;
                out.add_generation_prompt = flag;
            }
            "enable_thinking" => {
                let flag = json_obj
                    .get("enable_thinking")
                    .and_then(Value::as_bool)
                    .ok_or_else(|| invalid("chat template flag must be boolean"))?;
                out.enable_thinking = flag;
            }
            _ => return Err(invalid("unknown chat template parameter")),
        }
    }
    out.messages = parsed.lookup_exact_key(b"messages");
    out.tools = parsed.lookup_exact_key(b"tools");
    out.extra_context = parsed.lookup_exact_key(b"extra_context");
    Ok(out)
}

fn parse_generation_params(val: &Value) -> Result<GenerationParams> {
    let raw = bounded_str(val, MAX_PARAMS_JSON_BYTES, "chat generation params JSON")?;
    let _ = parse_json_jval_bounded(raw, MAX_PARAMS_JSON_BYTES, MAX_PARAMS_JSON_DEPTH)
        .map_err(|err| invalid(format!("invalid chat generation params JSON: {err}")))?;
    let parsed: Value = serde_json::from_str(raw)?;
    let obj = parsed
        .as_object()
        .ok_or_else(|| invalid("chat generation params must be object"))?;
    let mut params = GenerationParams::default();
    for (k, v) in obj {
        match k.as_str() {
            "tools" => {
                let text = serde_json::to_string(v)?;
                params.tools = Some(
                    JsonParser::parse_with_limit(&text, MAX_PARAMS_JSON_BYTES).map_err(invalid)?,
                );
            }
            "messages" => {
                let text = serde_json::to_string(v)?;
                params.messages = Some(
                    JsonParser::parse_with_limit(&text, MAX_PARAMS_JSON_BYTES).map_err(invalid)?,
                );
            }
            "json_schema" => {
                let text = serde_json::to_string(v)?;
                params.json_schema = Some(
                    JsonParser::parse_with_limit(&text, MAX_PARAMS_JSON_BYTES).map_err(invalid)?,
                );
            }
            "reasoning_format" => {
                params.reasoning_format = i32::try_from(bounded_int(v, 0, 3)?)?;
            }
            "tool_choice" => {
                params.tool_choice = i32::try_from(bounded_int(v, 0, 2)?)?;
            }
            "parallel_tool_calls" => {
                params.parallel_tool_calls = v
                    .as_bool()
                    .ok_or_else(|| invalid("generation flag must be boolean"))?;
            }
            "enable_thinking" => {
                params.enable_thinking = v
                    .as_bool()
                    .ok_or_else(|| invalid("generation flag must be boolean"))?;
            }
            "add_generation_prompt" => {
                params.add_generation_prompt = v
                    .as_bool()
                    .ok_or_else(|| invalid("generation flag must be boolean"))?;
            }
            _ => return Err(invalid("unknown chat generation parameter")),
        }
    }
    Ok(params)
}

fn validate_helper_op(obj: &Map<String, Value>) -> Result<()> {
    let val = Value::Object(obj.clone());
    exact_fields(&val, &["Kind", "Function", "Arguments"])?;
    let func = bounded_str(field(&val, "Function")?, 64, "Function")?;
    let args = array(&val, "Arguments")?;
    let valid_arity = match func {
        "calculate-diff-split" => args.len() == 2,
        "until-common-prefix" | "after-common-suffix" => args.len() == 3,
        "segmentize-markers" => args.len() == 1,
        _ => false,
    };
    require(valid_arity, "unknown chat helper or arity")?;
    for arg in args {
        let _ = bounded_str(arg, MAX_HELPER_ARG_BYTES, "chat helper argument")?;
    }
    Ok(())
}

fn validate_compare_variants_op(obj: &Map<String, Value>) -> Result<()> {
    let val = Value::Object(obj.clone());
    exact_fields(
        &val,
        &[
            "Kind",
            "Template",
            "BosToken",
            "EosToken",
            "ParametersJson",
            "ModifierJson",
        ],
    )?;
    let _ = bounded_str(field(&val, "Template")?, MAX_TEMPLATE_BYTES, "Template")?;
    let _ = bounded_str(field(&val, "BosToken")?, MAX_TOKEN_BYTES, "BosToken")?;
    let _ = bounded_str(field(&val, "EosToken")?, MAX_TOKEN_BYTES, "EosToken")?;
    let _ = parse_template_params(field(&val, "ParametersJson")?)?;
    let mod_val = field(&val, "ModifierJson")?;
    if !mod_val.is_null() {
        let _ = parse_template_params(mod_val)?;
    }
    Ok(())
}

fn validate_auto_trace_op(obj: &Map<String, Value>) -> Result<()> {
    let val = Value::Object(obj.clone());
    exact_fields(
        &val,
        &["Kind", "Template", "BosToken", "EosToken", "Actions"],
    )?;
    let _ = bounded_str(field(&val, "Template")?, MAX_TEMPLATE_BYTES, "Template")?;
    let _ = bounded_str(field(&val, "BosToken")?, MAX_TOKEN_BYTES, "BosToken")?;
    let _ = bounded_str(field(&val, "EosToken")?, MAX_TOKEN_BYTES, "EosToken")?;
    let actions = array(&val, "Actions")?;
    require(
        actions.len() <= MAX_ACTIONS,
        "chat auto actions exceed bound",
    )?;

    let mut analyzed = false;
    let mut built = false;
    for action in actions {
        let kind = action
            .get("Kind")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("chat auto action Kind must be string"))?;
        match kind {
            "analyze" => {
                exact_fields(action, &["Kind"])?;
                analyzed = true;
            }
            "original-caps" => {
                exact_fields(action, &["Kind"])?;
            }
            "build-parser" => {
                exact_fields(action, &["Kind", "ParametersJson", "GenerationPrompt"])?;
                require(analyzed, "chat parser requires prior analysis")?;
                let _ = parse_generation_params(field(action, "ParametersJson")?)?;
                let _ = bounded_str(
                    field(action, "GenerationPrompt")?,
                    MAX_TOKEN_BYTES,
                    "GenerationPrompt",
                )?;
                built = true;
            }
            "parse" => {
                exact_fields(action, &["Kind", "Bytes", "Flags", "Map"])?;
                require(built, "chat auto parse requires prior parser")?;
                let bytes = array(action, "Bytes")?;
                require(
                    bytes.len() <= MAX_PARSE_BYTES,
                    "chat auto parse input exceeds bound",
                )?;
                for byte in bytes {
                    let _ = bounded_int(byte, 0, 255)?;
                }
                let _ = bounded_int(field(action, "Flags")?, 0, 3)?;
                require(
                    field(action, "Map")?.is_boolean(),
                    "chat mapper flag must be boolean",
                )?;
            }
            _ => return Err(invalid("unknown chat auto action")),
        }
    }
    Ok(())
}

fn validate_operation(op: &Value) -> Result<()> {
    let obj = op
        .as_object()
        .ok_or_else(|| invalid("chat auto operation must be object"))?;
    let kind = obj
        .get("Kind")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("chat auto Kind must be string"))?;
    match kind {
        "chat-helper" => validate_helper_op(obj),
        "chat-compare-variants" => validate_compare_variants_op(obj),
        "chat-auto-trace" => validate_auto_trace_op(obj),
        _ => Err(invalid("unknown chat auto operation Kind")),
    }
}

fn execute_helper_op(op: &Value) -> Result<Value> {
    let func = op["Function"]
        .as_str()
        .ok_or_else(|| invalid("missing Function"))?;
    let args = array(op, "Arguments")?;
    let arg_str = |idx: usize| -> Result<&str> {
        args.get(idx)
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("missing helper string arg"))
    };
    match func {
        "calculate-diff-split" => Ok(diff_split_str(arg_str(0)?, arg_str(1)?).to_json()),
        "until-common-prefix" => {
            Ok(json!({"value": until_common_prefix(arg_str(0)?, arg_str(1)?, arg_str(2)?)}))
        }
        "after-common-suffix" => {
            Ok(json!({"value": after_common_suffix(arg_str(0)?, arg_str(1)?, arg_str(2)?)}))
        }
        "segmentize-markers" => {
            let segments: Vec<Value> = segmentize_markers(arg_str(0)?)
                .into_iter()
                .map(|seg| {
                    json!({
                        "type": seg.seg_type.as_u8(),
                        "value": seg.value,
                    })
                })
                .collect();
            Ok(json!({"segments": segments}))
        }
        _ => Err(invalid("unknown chat helper")),
    }
}

fn execute_compare_variants_op(op: &Value) -> Result<Value> {
    let tmpl = ChatTemplate::new(
        op["Template"].as_str().unwrap_or_default(),
        op["BosToken"].as_str().unwrap_or_default(),
        op["EosToken"].as_str().unwrap_or_default(),
    );
    let base = parse_template_params(&op["ParametersJson"])?;
    let mod_opt = if op["ModifierJson"].is_null() {
        None
    } else {
        Some(parse_template_params(&op["ModifierJson"])?)
    };
    let result = mod_opt.map_or_else(
        || compare_variants(&tmpl, &base, None),
        |variant| {
            compare_variants(
                &tmpl,
                &base,
                Some(&|p| {
                    *p = variant.clone();
                }),
            )
        },
    );
    Ok(match result {
        Some(v) => json!({
            "has_value": true,
            "diff": v.diff.to_json(),
            "output_A": v.output_a,
            "output_B": v.output_b,
        }),
        None => json!({
            "has_value": false,
            "diff": null,
            "output_A": null,
            "output_B": null,
        }),
    })
}

fn execute_auto_trace_op(op: &Value) -> Result<Value> {
    let tmpl = ChatTemplate::new(
        op["Template"].as_str().unwrap_or_default(),
        op["BosToken"].as_str().unwrap_or_default(),
        op["EosToken"].as_str().unwrap_or_default(),
    );
    let mut analyzer = AutoParser::default();
    let mut parser: Option<ChatArena> = None;
    let mut trace = Vec::new();
    for action in array(op, "Actions")? {
        match action["Kind"].as_str() {
            Some("analyze") => {
                analyzer.analyze_template(&tmpl);
                trace.push(json!({"analysis": analyzer.analysis().to_json()}));
            }
            Some("original-caps") => {
                trace.push(json!({"caps": caps_to_json(tmpl.original_caps())}));
            }
            Some("build-parser") => {
                let gen_params = parse_generation_params(&action["ParametersJson"])?;
                let gen_prompt = action["GenerationPrompt"]
                    .as_str()
                    .ok_or_else(|| invalid("GenerationPrompt must be string"))?;
                parser = Some(analyzer.build_parser(&gen_params, gen_prompt)?);
                trace.push(json!({}));
            }
            Some("parse") => {
                let arena = parser
                    .as_ref()
                    .ok_or_else(|| invalid("chat auto parse requires prior parser"))?;
                let mut bytes = Vec::new();
                for b in array(action, "Bytes")? {
                    bytes.push(u8::try_from(bounded_int(b, 0, 255)?)?);
                }
                let flags = u8::try_from(bounded_int(&action["Flags"], 0, 3)?)?;
                let mut ctx = ParseContext::new(&bytes, flags);
                let res = arena.parse_with_context(&mut ctx)?;
                let msg = if action["Map"].as_bool() == Some(true) {
                    let mut chat_msg = ChatMessage::default();
                    map_chat_ast(&ctx.ast, &res, &mut chat_msg);
                    wire_chat_message(&chat_msg)
                } else {
                    Value::Null
                };
                trace.push(json!({
                    "result": wire_parse_result(&res),
                    "message": msg,
                }));
            }
            _ => return Err(invalid("unknown chat auto action")),
        }
    }
    Ok(json!({"Trace": trace}))
}

fn execute_operation(op: &Value) -> Result<Value> {
    let kind = op["Kind"].as_str().ok_or_else(|| invalid("missing Kind"))?;
    match kind {
        "chat-helper" => execute_helper_op(op),
        "chat-compare-variants" => execute_compare_variants_op(op),
        "chat-auto-trace" => execute_auto_trace_op(op),
        _ => Err(invalid("unknown chat auto operation Kind")),
    }
}

/// Returns `true` when every operation in `request` belongs to `reference_chat_auto`.
pub(super) fn handles(request: &Value) -> bool {
    request["Operations"].as_array().is_some_and(|operations| {
        !operations.is_empty()
            && operations.iter().all(|operation| {
                matches!(
                    operation["Kind"].as_str(),
                    Some("chat-helper" | "chat-compare-variants" | "chat-auto-trace")
                )
            })
    })
}

/// Validates and executes a bounded batch of `chat-helper` / `chat-compare-variants` / `chat-auto-trace` operations.
pub(super) fn execute(request: &Value) -> Result<Value> {
    exact_fields(request, &["SchemaVersion", "Operations"])?;
    if bounded_int(field(request, "SchemaVersion")?, 1, 1)? != 1 {
        return Err(invalid("unsupported SchemaVersion"));
    }
    let operations = array(request, "Operations")?;
    require(
        !operations.is_empty() && operations.len() <= MAX_OPERATIONS,
        "Operations must contain 1 through 128 operations",
    )?;
    for operation in operations {
        validate_operation(operation)?;
    }
    let results = operations
        .iter()
        .map(execute_operation)
        .collect::<Result<Vec<_>>>()?;
    Ok(json!({"SchemaVersion": 1, "Results": results}))
}

/// Provider identity metadata for `tests/test-chat-auto-parser.cpp`.
pub(super) fn identity() -> Value {
    json!({
        "Kind": "independent-rust-chat-auto-parser",
        "SupportedSourceIds": [ID],
        "UpstreamCommit": super::UPSTREAM,
        "AssertionSource": {"Path": ID, "Sha256": SOURCE_SHA256},
        "Architecture": "clean-room-differential-template-analyzer-and-peg-builder-v1",
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
    fn independent_chat_auto_parser_passes_all_groups_on_rust_reference_and_cpp() -> Result<()> {
        for target in [Target::RustReference, Target::Cpp] {
            let engine = Engine::new(target, None, None, None, None)?;
            let report = crate::chat_auto::run(&engine)?;
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
    fn independent_chat_auto_parser_rejects_invalid_ops_prerequisites_json_and_bounds() -> Result<()>
    {
        let valid = json!({
            "SchemaVersion": 1,
            "Operations": [{
                "Kind": "chat-auto-trace",
                "Template": "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}",
                "BosToken": "",
                "EosToken": "<|end|>",
                "Actions": [
                    {"Kind": "original-caps"},
                    {"Kind": "analyze"},
                    {"Kind": "build-parser", "ParametersJson": "{\"reasoning_format\":0,\"tools\":null}", "GenerationPrompt": ""},
                    {"Kind": "parse", "Bytes": [104, 101, 108, 108, 111], "Flags": 0, "Map": true}
                ]
            }]
        });
        let ok = execute(&valid)?;
        assert_eq!(ok["Results"][0]["Trace"][3]["message"]["Content"], "hello");

        // Reject unknown action kind.
        let mut bad_action = valid.clone();
        bad_action["Operations"][0]["Actions"][0] = json!({"Kind": "bogus"});
        assert!(execute(&bad_action).is_err());

        // Reject build-parser before analyze.
        let no_analyze = json!({
            "SchemaVersion": 1,
            "Operations": [{
                "Kind": "chat-auto-trace",
                "Template": "x",
                "BosToken": "",
                "EosToken": "",
                "Actions": [
                    {"Kind": "build-parser", "ParametersJson": "{}", "GenerationPrompt": ""}
                ]
            }]
        });
        assert!(execute(&no_analyze).is_err());

        // Reject parse before build-parser.
        let no_parser = json!({
            "SchemaVersion": 1,
            "Operations": [{
                "Kind": "chat-auto-trace",
                "Template": "x",
                "BosToken": "",
                "EosToken": "",
                "Actions": [
                    {"Kind": "analyze"},
                    {"Kind": "parse", "Bytes": [65], "Flags": 0, "Map": false}
                ]
            }]
        });
        assert!(execute(&no_parser).is_err());

        // Reject invalid ParametersJson inside build-parser & compare-variants.
        let mut bad_params = valid.clone();
        bad_params["Operations"][0]["Actions"][2]["ParametersJson"] = json!("{not json");
        assert!(execute(&bad_params).is_err());

        let bad_compare = json!({
            "SchemaVersion": 1,
            "Operations": [{
                "Kind": "chat-compare-variants",
                "Template": "x",
                "BosToken": "",
                "EosToken": "",
                "ParametersJson": "{bad",
                "ModifierJson": null
            }]
        });
        assert!(execute(&bad_compare).is_err());

        // Reject unknown helper and malformed arg counts/types.
        let bad_helper = json!({
            "SchemaVersion": 1,
            "Operations": [{
                "Kind": "chat-helper",
                "Function": "bogus-helper",
                "Arguments": ["a"]
            }]
        });
        assert!(execute(&bad_helper).is_err());

        let bad_helper_args = json!({
            "SchemaVersion": 1,
            "Operations": [{
                "Kind": "chat-helper",
                "Function": "until-common-prefix",
                "Arguments": ["a", "b"]
            }]
        });
        assert!(execute(&bad_helper_args).is_err());

        // Reject out-of-range parse byte and flags.
        let mut bad_byte = valid.clone();
        bad_byte["Operations"][0]["Actions"][3]["Bytes"] = json!([256]);
        assert!(execute(&bad_byte).is_err());

        let mut bad_flags = valid;
        bad_flags["Operations"][0]["Actions"][3]["Flags"] = json!(9);
        assert!(execute(&bad_flags).is_err());

        Ok(())
    }
}
