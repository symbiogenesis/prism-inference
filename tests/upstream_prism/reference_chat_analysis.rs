//! Clean-room independent safe Rust chat template trace & analysis provider.
//!
//! Implements the `chat-template-trace` diagnostic protocol (`caps`, `apply`,
//! `diff`, `undefined-probe`) using `reference_jinja` and
//! `reference_chat_templates`, without importing `bonsai_prism_runtime`,
//! invoking a native bridge, or reading expected-answer fixtures.

use std::collections::BTreeSet;

use serde_json::{Map, Value, json};

use super::reference_chat_templates::{
    TemplateCaps, common_direct_apply, current_unix_seconds, template_caps_from_program,
};
use super::reference_jinja::{
    JVal, eval_program_with_context, parse_json_jval_bounded, parse_template_with_limit,
};
use super::{Result, invalid};

/// Upstream source identifier covered by this independent reference provider.
pub(super) const ID: &str = "tests/test-chat-analysis.cpp";
const SOURCE_SHA: &str = "d895ae1736845640b8dab486c68189726821abfbbfc0ca96f98cd419633ad1a1";

const MAX_OPERATIONS: usize = 128;
const MAX_TEMPLATE_BYTES: usize = 1024 * 1024;
const MAX_TOKEN_BYTES: usize = 65_536;
const MAX_ACTIONS: usize = 128;
const MAX_ACTION_NAME_BYTES: usize = 64;
const MAX_NOW_UNIX_SECONDS: i64 = 4_102_444_800;
const MAX_JSON_BYTES: usize = 256 * 1024;
const MAX_JSON_DEPTH: usize = 64;
const MAX_PROBE_NAMES: usize = 128;
const MAX_PROBE_NAME_BYTES: usize = 256;
const MAX_OUTPUT_BYTES: usize = 1024 * 1024;
const MAX_ERROR_BYTES: usize = 4096;

/// Marker- and byte-level split between two rendered chat prompts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct DiffSplit {
    /// Common prefix bytes before the divergence region.
    pub(super) prefix: Vec<u8>,
    /// Common suffix bytes after the divergence region.
    pub(super) suffix: Vec<u8>,
    /// Divergent bytes present in the left prompt.
    pub(super) left: Vec<u8>,
    /// Divergent bytes present in the right prompt.
    pub(super) right: Vec<u8>,
}

enum ValidatedAction {
    Caps,
    Apply {
        parameters_json: String,
        now_unix_seconds: i64,
        add_generation_prompt: bool,
        enable_thinking: Option<bool>,
    },
    Diff {
        left: usize,
        right: usize,
    },
    UndefinedProbe {
        variables_json: String,
        names: Vec<String>,
    },
}

struct ValidatedTraceOp {
    template: String,
    bos: String,
    eos: String,
    stop_on_error: bool,
    actions: Vec<ValidatedAction>,
}

fn segment_markers(input: &[u8]) -> Vec<(bool, &[u8])> {
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

fn concat_segments(parts: &[(bool, &[u8])]) -> Vec<u8> {
    parts
        .iter()
        .flat_map(|(_, seg)| seg.iter().copied())
        .collect()
}

fn prepend_bytes(prefix: &[u8], tail: &[u8]) -> Vec<u8> {
    prefix.iter().chain(tail).copied().collect()
}

fn finish_diff_split(
    mut result: DiffSplit,
    left: &[u8],
    right: &[u8],
    a_rest: &[u8],
    b_rest: &[u8],
    text_suffix: bool,
    text_prefix: bool,
) -> DiffSplit {
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
    result.prefix.extend_from_slice(&a_rest[..prefix]);
    result.suffix = prepend_bytes(&a_rest[a_rest.len() - suffix..], &result.suffix);
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

/// Computes upstream marker-aware `calculate_diff_split` over two byte strings.
pub(super) fn calculate_diff_split(left: &[u8], right: &[u8]) -> DiffSplit {
    let a = segment_markers(left);
    let b = segment_markers(right);
    let mut result = DiffSplit {
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
            result.prefix.extend_from_slice(a[a_start].1);
            a_start += 1;
            b_start += 1;
            true
        } else {
            false
        };
        let suffix_advanced = if a[a_end] == b[b_end] {
            result.suffix = prepend_bytes(a[a_end].1, &result.suffix);
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
            result.suffix = prepend_bytes(b[b_end].1, &result.suffix);
            b_end -= 1;
            a_consumed = true;
        } else if a[a_start] == b[b_start] {
            result.prefix.extend_from_slice(b[b_start].1);
            b_start += 1;
            a_consumed = true;
        }
    } else if b_start == b_end && a_start != a_end {
        if a[a_end] == b[b_start] {
            result.suffix = prepend_bytes(a[a_end].1, &result.suffix);
            a_end -= 1;
            b_consumed = true;
        } else if a[a_start] == b[b_start] {
            result.prefix.extend_from_slice(a[a_start].1);
            a_start += 1;
            b_consumed = true;
        }
    } else if a_start == a_end && b_start == b_end && same_middle && a[a_start].0 {
        result.prefix.extend_from_slice(b[b_start].1);
        a_consumed = true;
        b_consumed = true;
    }
    let text_suffix = !a[a_end].0 && !b[b_end].0;
    let text_prefix = !a[a_start].0 && !b[b_start].0;
    let a_rest = concat_segments(&a[a_start..a_end + usize::from(!a_consumed)]);
    let b_rest = concat_segments(&b[b_start..b_end + usize::from(!b_consumed)]);
    finish_diff_split(
        result,
        left,
        right,
        &a_rest,
        &b_rest,
        text_suffix,
        text_prefix,
    )
}

fn require_exact_fields(obj: &Map<String, Value>, names: &[&str]) -> Result<()> {
    if obj.len() != names.len() || names.iter().any(|name| !obj.contains_key(*name)) {
        return Err(invalid("chat observation field mismatch"));
    }
    Ok(())
}

fn bounded_text(val: &Value, maximum: usize) -> Result<&str> {
    let s = val
        .as_str()
        .ok_or_else(|| invalid("chat observation string required"))?;
    if s.len() > maximum || s.contains('\0') {
        return Err(invalid("chat observation string bound/NUL violation"));
    }
    Ok(s)
}

fn bounded_bool(val: &Value) -> Result<bool> {
    val.as_bool()
        .ok_or_else(|| invalid("chat observation boolean required"))
}

fn bounded_index(val: &Value, limit_exclusive: usize) -> Result<usize> {
    let raw = val
        .as_i64()
        .ok_or_else(|| invalid("chat observation index must be integer"))?;
    if raw < 0 {
        return Err(invalid("chat observation index must be nonnegative"));
    }
    let idx = usize::try_from(raw)?;
    if idx >= limit_exclusive {
        return Err(invalid("chat observation index outside prior trace"));
    }
    Ok(idx)
}

fn bounded_timestamp(val: &Value) -> Result<i64> {
    let raw = val
        .as_i64()
        .ok_or_else(|| invalid("chat observation timestamp must be integer"))?;
    if !(0..=MAX_NOW_UNIX_SECONDS).contains(&raw) {
        return Err(invalid("chat observation timestamp outside protocol bound"));
    }
    Ok(raw)
}

fn parse_json_object(val: &Value) -> Result<(String, Map<String, Value>)> {
    let raw = bounded_text(val, MAX_JSON_BYTES)?;
    let parsed: Value =
        serde_json::from_str(raw).map_err(|err| invalid(format!("invalid JSON object: {err}")))?;
    let obj = parsed
        .as_object()
        .ok_or_else(|| invalid("chat parameters/globals must be object"))?
        .clone();
    // Validate structural depth & duplicate keys via the strict JSON parser.
    let _ = parse_json_jval_bounded(raw, MAX_JSON_BYTES, MAX_JSON_DEPTH)
        .map_err(|err| invalid(format!("invalid JSON value bounds: {err}")))?;
    Ok((raw.to_owned(), obj))
}

fn validate_action(
    idx: usize,
    action_val: &Value,
    raw_actions: &[Value],
) -> Result<ValidatedAction> {
    let obj = action_val
        .as_object()
        .ok_or_else(|| invalid("chat observation action must be object"))?;
    let name = bounded_text(
        obj.get("Action")
            .ok_or_else(|| invalid("chat observation Action required"))?,
        MAX_ACTION_NAME_BYTES,
    )?;
    match name {
        "caps" => {
            require_exact_fields(obj, &["Action"])?;
            Ok(ValidatedAction::Caps)
        }
        "apply" => {
            require_exact_fields(obj, &["Action", "ParametersJson", "NowUnixSeconds"])?;
            let now_unix_seconds = bounded_timestamp(
                obj.get("NowUnixSeconds")
                    .ok_or_else(|| invalid("missing NowUnixSeconds"))?,
            )?;
            let (parameters_json, params_obj) = parse_json_object(
                obj.get("ParametersJson")
                    .ok_or_else(|| invalid("missing ParametersJson"))?,
            )?;
            let mut add_generation_prompt = true;
            let mut enable_thinking = None;
            for (key, val) in &params_obj {
                match key.as_str() {
                    "messages" | "tools" => {}
                    "add_generation_prompt" => {
                        add_generation_prompt = val
                            .as_bool()
                            .ok_or_else(|| invalid("direct-apply flag must be boolean"))?;
                    }
                    "enable_thinking" => {
                        enable_thinking = Some(
                            val.as_bool()
                                .ok_or_else(|| invalid("direct-apply flag must be boolean"))?,
                        );
                    }
                    _ => return Err(invalid("unknown direct-apply parameter")),
                }
            }
            Ok(ValidatedAction::Apply {
                parameters_json,
                now_unix_seconds,
                add_generation_prompt,
                enable_thinking,
            })
        }
        "diff" => {
            require_exact_fields(obj, &["Action", "Left", "Right"])?;
            let left = bounded_index(obj.get("Left").ok_or_else(|| invalid("missing Left"))?, idx)?;
            let right = bounded_index(
                obj.get("Right").ok_or_else(|| invalid("missing Right"))?,
                idx,
            )?;
            for prior in [left, right] {
                if raw_actions[prior].get("Action").and_then(Value::as_str) != Some("apply") {
                    return Err(invalid("diff reference must name prior apply"));
                }
            }
            Ok(ValidatedAction::Diff { left, right })
        }
        "undefined-probe" => {
            require_exact_fields(obj, &["Action", "VariablesJson", "Names"])?;
            let (variables_json, _) = parse_json_object(
                obj.get("VariablesJson")
                    .ok_or_else(|| invalid("missing VariablesJson"))?,
            )?;
            let names_arr = obj
                .get("Names")
                .and_then(Value::as_array)
                .ok_or_else(|| invalid("Names must be an array"))?;
            if names_arr.len() > MAX_PROBE_NAMES {
                return Err(invalid("undefined probe names exceed 128"));
            }
            let mut seen = BTreeSet::new();
            let mut names = Vec::with_capacity(names_arr.len());
            for entry in names_arr {
                let s = bounded_text(entry, MAX_PROBE_NAME_BYTES)?;
                if !seen.insert(s) {
                    return Err(invalid("duplicate undefined probe name"));
                }
                names.push(s.to_owned());
            }
            Ok(ValidatedAction::UndefinedProbe {
                variables_json,
                names,
            })
        }
        _ => Err(invalid("unknown chat observation action")),
    }
}

fn validate_operation(op: &Value) -> Result<ValidatedTraceOp> {
    let obj = op
        .as_object()
        .ok_or_else(|| invalid("chat observation operation must be object"))?;
    require_exact_fields(
        obj,
        &["Kind", "Template", "Bos", "Eos", "StopOnError", "Actions"],
    )?;
    if obj.get("Kind").and_then(Value::as_str) != Some("chat-template-trace") {
        return Err(invalid("unknown chat observation kind"));
    }
    let template = bounded_text(
        obj.get("Template")
            .ok_or_else(|| invalid("missing Template"))?,
        MAX_TEMPLATE_BYTES,
    )?
    .to_owned();
    let bos = bounded_text(
        obj.get("Bos").ok_or_else(|| invalid("missing Bos"))?,
        MAX_TOKEN_BYTES,
    )?
    .to_owned();
    let eos = bounded_text(
        obj.get("Eos").ok_or_else(|| invalid("missing Eos"))?,
        MAX_TOKEN_BYTES,
    )?
    .to_owned();
    let stop_on_error = bounded_bool(
        obj.get("StopOnError")
            .ok_or_else(|| invalid("missing StopOnError"))?,
    )?;
    let actions_arr = obj
        .get("Actions")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("Actions must be an array"))?;
    if actions_arr.len() > MAX_ACTIONS {
        return Err(invalid("chat observation actions exceed 128"));
    }
    let mut actions = Vec::with_capacity(actions_arr.len());
    for (idx, action_val) in actions_arr.iter().enumerate() {
        actions.push(validate_action(idx, action_val, actions_arr)?);
    }
    Ok(ValidatedTraceOp {
        template,
        bos,
        eos,
        stop_on_error,
        actions,
    })
}

fn truncate_utf8_bytes(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut end = max_bytes;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

fn format_error(message: &str) -> Value {
    json!({
        "Kind": "standard",
        "Message": truncate_utf8_bytes(message, MAX_ERROR_BYTES),
    })
}

fn encode_bounded_hex(bytes: &[u8]) -> std::result::Result<String, String> {
    if bytes.len() > MAX_OUTPUT_BYTES {
        return Err("chat observation output exceeds 1MiB".to_owned());
    }
    Ok(hex::encode(bytes))
}

fn caps_json(caps: TemplateCaps) -> Value {
    json!({
        "supports_tools": caps.supports_tools,
        "supports_tool_calls": caps.supports_tool_calls,
        "supports_system_role": caps.supports_system_role,
        "supports_parallel_tool_calls": caps.supports_parallel_tool_calls,
        "supports_typed_content": caps.supports_typed_content,
        "supports_string_content": caps.supports_string_content,
    })
}

fn execute_single_action(
    program: &super::reference_jinja::Program,
    bos: &str,
    eos: &str,
    idx: usize,
    action: ValidatedAction,
    rendered: &mut [Option<Vec<u8>>],
    cached_caps: &mut Option<TemplateCaps>,
) -> std::result::Result<Value, String> {
    match action {
        ValidatedAction::Caps => {
            let caps = *cached_caps.get_or_insert_with(|| template_caps_from_program(program));
            Ok(json!({"Caps": caps_json(caps)}))
        }
        ValidatedAction::Apply {
            parameters_json,
            now_unix_seconds,
            add_generation_prompt,
            enable_thinking,
        } => {
            let caps = *cached_caps.get_or_insert_with(|| template_caps_from_program(program));
            let params_val =
                parse_json_jval_bounded(&parameters_json, MAX_JSON_BYTES, MAX_JSON_DEPTH)
                    .map_err(|err| err.to_string())?;
            let messages = params_val
                .lookup_exact_key(b"messages")
                .unwrap_or_else(|| JVal::array(Vec::new()));
            let tools = params_val.lookup_exact_key(b"tools");
            let bytes = common_direct_apply(
                program,
                caps,
                bos,
                eos,
                &messages,
                tools,
                add_generation_prompt,
                enable_thinking,
                now_unix_seconds,
            )?;
            let rendered_hex = encode_bounded_hex(&bytes)?;
            rendered[idx] = Some(bytes);
            Ok(json!({"RenderedHex": rendered_hex}))
        }
        ValidatedAction::Diff { left, right } => {
            let left_bytes = rendered[left].as_deref().unwrap_or(&[]);
            let right_bytes = rendered[right].as_deref().unwrap_or(&[]);
            let split = calculate_diff_split(left_bytes, right_bytes);
            Ok(json!({
                "Diff": {
                    "PrefixHex": encode_bounded_hex(&split.prefix)?,
                    "SuffixHex": encode_bounded_hex(&split.suffix)?,
                    "LeftHex": encode_bounded_hex(&split.left)?,
                    "RightHex": encode_bounded_hex(&split.right)?,
                }
            }))
        }
        ValidatedAction::UndefinedProbe {
            variables_json,
            names,
        } => {
            let vars = parse_json_jval_bounded(&variables_json, MAX_JSON_BYTES, MAX_JSON_DEPTH)
                .map_err(|err| err.to_string())?;
            for name in &names {
                vars.set_member(JVal::text_str(name), JVal::undefined())
                    .map_err(|err| err.message().to_owned())?;
            }
            let now = current_unix_seconds();
            let (eval_res, root_scope) = eval_program_with_context(program, &vars, Some(now));
            let runtime_error = match eval_res {
                Ok(_) => Value::Null,
                Err(err) => format_error(err.message()),
            };
            let mut variables = Vec::with_capacity(names.len());
            for name in names {
                let observed = root_scope.get(&name);
                variables.push(json!({
                    "Name": name,
                    "Undefined": observed.is_none_or(JVal::is_undefined),
                    "Used": observed.is_some_and(JVal::is_used),
                }));
            }
            Ok(json!({
                "Variables": variables,
                "RuntimeError": runtime_error,
            }))
        }
    }
}

fn execute_trace_op(op: ValidatedTraceOp) -> Value {
    let program = match parse_template_with_limit(&op.template, MAX_TEMPLATE_BYTES) {
        Ok(program) => program,
        Err(err) => {
            return json!({
                "Trace": [],
                "TemplateError": format_error(err.message()),
            });
        }
    };

    let mut trace = Vec::with_capacity(op.actions.len());
    let mut rendered: Vec<Option<Vec<u8>>> = vec![None; op.actions.len()];
    let mut cached_caps: Option<TemplateCaps> = None;

    for (idx, action) in op.actions.into_iter().enumerate() {
        match execute_single_action(
            &program,
            &op.bos,
            &op.eos,
            idx,
            action,
            &mut rendered,
            &mut cached_caps,
        ) {
            Ok(item) => trace.push(item),
            Err(message) => {
                trace.push(json!({"ActionError": format_error(&message)}));
                if op.stop_on_error {
                    break;
                }
            }
        }
    }

    json!({
        "Trace": trace,
        "TemplateError": Value::Null,
    })
}

/// Returns `true` when every operation in `request` belongs to `chat_analysis`.
pub(super) fn handles(request: &Value) -> bool {
    request["Operations"].as_array().is_some_and(|operations| {
        !operations.is_empty()
            && operations
                .iter()
                .all(|operation| operation["Kind"].as_str() == Some("chat-template-trace"))
    })
}

/// Validates and executes a bounded batch of `chat-template-trace` operations.
pub(super) fn execute(request: &Value) -> Result<Value> {
    let obj = request
        .as_object()
        .ok_or_else(|| invalid("chat analysis request must be a JSON object"))?;
    require_exact_fields(obj, &["SchemaVersion", "Operations"])?;
    let schema = obj
        .get("SchemaVersion")
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid("missing or non-integer SchemaVersion"))?;
    if schema != 1 {
        return Err(invalid("unsupported SchemaVersion"));
    }
    let ops_arr = obj
        .get("Operations")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("Operations must be an array"))?;
    if ops_arr.is_empty() || ops_arr.len() > MAX_OPERATIONS {
        return Err(invalid("Operations must contain 1 through 128 operations"));
    }
    let validated = ops_arr
        .iter()
        .map(validate_operation)
        .collect::<Result<Vec<_>>>()?;
    let results: Vec<Value> = validated.into_iter().map(execute_trace_op).collect();
    Ok(json!({"SchemaVersion": 1, "Results": results}))
}

/// Independent reference provider metadata for `tests/test-chat-analysis.cpp`.
pub(super) fn identity() -> Value {
    json!({
        "Kind": "independent-rust-chat-analysis",
        "SupportedSourceIds": [ID],
        "UpstreamCommit": super::UPSTREAM,
        "AssertionSource": {"Path": ID, "Sha256": SOURCE_SHA},
        "Architecture": "clean-room-template-caps-direct-apply-marker-diff-and-undefined-probe-v1",
        "NativeReferenceReplay": false,
        "ProductionRuntimeCalled": false,
        "CompleteSourceGate": false,
        "NativeRetirementQualified": false
    })
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use crate::engine::{Engine, Target};

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(prefix: &str) -> Result<Self> {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
            let seq = NEXT.fetch_add(1, Ordering::Relaxed);
            let dir =
                std::env::temp_dir().join(format!("{prefix}-{}-{nonce}-{seq}", std::process::id()));
            if !dir.is_absolute() {
                return Err(invalid("temp directory must be absolute"));
            }
            fs::create_dir(&dir)?;
            Ok(Self(dir))
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn wrap(op: &Value) -> Value {
        json!({"SchemaVersion": 1, "Operations": [op]})
    }

    #[test]
    fn independent_chat_analysis_passes_full_suite_across_reference_arms_with_synthetic_templates()
    -> Result<()> {
        let inputs: Value = serde_json::from_str(include_str!("chat-analysis-inputs.json"))?;
        let template_paths = inputs["TemplatePaths"]
            .as_array()
            .ok_or_else(|| invalid("missing TemplatePaths"))?;
        assert_eq!(template_paths.len(), 37);

        let dir = TempDir::new("SwiftBonsai-RefChatAnalysis")?;
        let variants = [
            // Variant 0: reasoning, tools, tool calls, system role, strftime_now, thinking flag
            "{% if enable_thinking is defined and enable_thinking %}<think/>{% endif %}{% if tools %}[TOOLS:{{ tools|length }}]{% endif %}{% for m in messages %}<|{{ m.role }}|>{{ m.content|default('') }}{% if m.reasoning_content %}<thought>{{ m.reasoning_content }}</thought>{% endif %}{% if m.tool_calls %}[CALLS:{{ m.tool_calls|length }}]{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}",
            // Variant 1: typed content only (no string content), parallel tool calls
            "{% for m in messages %}<|{{ m.role }}|>{% for part in m.content | selectattr('type', 'equalto', 'text') %}{{ part.text }}{% endfor %}{% if m.tool_calls %}{% for tc in m.tool_calls %}[TC:{{ tc.function.name }}]{% endfor %}{% endif %}{% endfor %}{% if add_generation_prompt %}<|gen|>{% endif %}",
            // Variant 2: syntax error (verifies accepted TemplateError handling in suite)
            "{{ unclosed_template",
            // Variant 3: runtime exception when thinking enabled
            "{% if enable_thinking %}{{ raise_exception('thinking unsupported') }}{% endif %}{% for m in messages %}{{ m.role }}:{{ m.content }}{% endfor %}",
        ];

        for (idx, rel) in template_paths.iter().enumerate() {
            let rel_str = rel
                .as_str()
                .ok_or_else(|| invalid("TemplatePaths entry must be string"))?;
            let full = dir.path().join(rel_str);
            if let Some(parent) = full.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(full, variants[idx % variants.len()])?;
        }

        for target in [Target::RustReference, Target::Cpp] {
            let engine = Engine::new(
                target,
                None,
                None,
                None,
                Some(dir.path().to_path_buf()),
                None,
            )?;
            let report = crate::chat_analysis::run(&engine)?;
            assert_eq!(report["Id"], ID);
            assert_eq!(report["Status"], "pass", "{report}");
        }

        assert!(crate::reference_provider::supports(ID));
        let info = identity();
        assert_eq!(info["SupportedSourceIds"], json!([ID]));
        assert_eq!(info["AssertionSource"]["Sha256"], SOURCE_SHA);
        assert_eq!(info["NativeReferenceReplay"], false);
        assert_eq!(info["ProductionRuntimeCalled"], false);
        assert_eq!(info["CompleteSourceGate"], false);
        assert_eq!(info["NativeRetirementQualified"], false);
        Ok(())
    }

    #[test]
    fn independent_chat_analysis_trace_actions_errors_and_diff_behave_per_protocol() -> Result<()> {
        let tmpl = "{% if enable_thinking is defined %}{{ enable_thinking }}{% endif %}{% for m in messages %}<|{{ m.role }}|>{{ m.content|default('') }}{% if m.reasoning_content is defined and m.reasoning_content %}<think>{{ m.reasoning_content }}</think>{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}";
        let req = wrap(&json!({
            "Kind": "chat-template-trace",
            "Template": tmpl,
            "Bos": "",
            "Eos": "",
            "StopOnError": false,
            "Actions": [
                {"Action": "caps"},
                {
                    "Action": "apply",
                    "ParametersJson": r#"{"messages":[{"role":"user","content":"hi"}],"add_generation_prompt":false}"#,
                    "NowUnixSeconds": 0
                },
                {
                    "Action": "apply",
                    "ParametersJson": r#"{"messages":[{"role":"user","content":"hi"}],"add_generation_prompt":true}"#,
                    "NowUnixSeconds": 0
                },
                {"Action": "diff", "Left": 1, "Right": 2},
                {
                    "Action": "undefined-probe",
                    "VariablesJson": r#"{"messages":[{"role":"user","content":"hi"}],"enable_thinking":true,"unused_override":false}"#,
                    "Names": ["enable_thinking", "reasoning_content", "unused_var"]
                },
                {"Action": "undefined-probe", "VariablesJson": r#"{"messages":[{"role":"user","content":"hi"}]}"#, "Names": ["enable_thinking"]}
            ]
        }));
        let res = execute(&req)?;
        assert_eq!(res["SchemaVersion"], 1);
        let item = &res["Results"][0];
        assert!(item["TemplateError"].is_null());
        let trace = item["Trace"]
            .as_array()
            .ok_or_else(|| invalid("missing trace"))?;
        assert_eq!(trace.len(), 6);
        assert_eq!(trace[0]["Caps"]["supports_string_content"], true);
        assert_eq!(trace[0]["Caps"]["supports_system_role"], true);
        assert_eq!(trace[3]["Diff"]["PrefixHex"], trace[1]["RenderedHex"]);
        assert_eq!(trace[3]["Diff"]["LeftHex"], "");
        assert_eq!(trace[3]["Diff"]["RightHex"], hex::encode("<|assistant|>"));
        // Probe checks enable_thinking & reasoning_content were used while unused_var stayed unused.
        let vars = trace[4]["Variables"]
            .as_array()
            .ok_or_else(|| invalid("missing Variables"))?;
        assert_eq!(vars[0]["Name"], "enable_thinking");
        assert_eq!(vars[0]["Undefined"], true);
        assert_eq!(vars[0]["Used"], true);
        assert_eq!(vars[1]["Name"], "reasoning_content");
        assert_eq!(vars[1]["Undefined"], true);
        assert_eq!(vars[1]["Used"], false);
        assert_eq!(vars[2]["Name"], "unused_var");
        assert_eq!(vars[2]["Undefined"], true);
        assert_eq!(vars[2]["Used"], false);
        Ok(())
    }

    #[test]
    fn independent_chat_analysis_stop_on_error_and_template_error_behave_per_protocol() -> Result<()>
    {
        // TemplateError short-circuits with empty Trace.
        let bad_tpl_res = execute(&wrap(&json!({
            "Kind": "chat-template-trace",
            "Template": "{{ bad",
            "Bos": "",
            "Eos": "",
            "StopOnError": true,
            "Actions": [{"Action": "caps"}]
        })))?;
        assert_eq!(bad_tpl_res["Results"][0]["Trace"], json!([]));
        assert_eq!(
            bad_tpl_res["Results"][0]["TemplateError"]["Kind"],
            "standard"
        );

        // StopOnError=true halts on ActionError, whereas StopOnError=false continues.
        let err_tpl = "{% if add_generation_prompt %}{{ raise_exception('boom') }}{% endif %}ok";
        for (stop, expected_len) in [(true, 1), (false, 2)] {
            let stop_res = execute(&wrap(&json!({
                "Kind": "chat-template-trace",
                "Template": err_tpl,
                "Bos": "",
                "Eos": "",
                "StopOnError": stop,
                "Actions": [
                    {
                        "Action": "apply",
                        "ParametersJson": r#"{"messages":[],"add_generation_prompt":true}"#,
                        "NowUnixSeconds": 0
                    },
                    {
                        "Action": "apply",
                        "ParametersJson": r#"{"messages":[],"add_generation_prompt":false}"#,
                        "NowUnixSeconds": 0
                    }
                ]
            })))?;
            let err_trace = stop_res["Results"][0]["Trace"]
                .as_array()
                .ok_or_else(|| invalid("missing Trace"))?;
            assert_eq!(err_trace.len(), expected_len);
            assert_eq!(err_trace[0]["ActionError"]["Kind"], "standard");
        }
        Ok(())
    }

    #[test]
    fn independent_chat_analysis_rejects_invalid_protocol_fields_and_bounds() {
        assert!(!handles(&json!({"Operations": []})));
        assert!(!handles(&wrap(&json!({"Kind": "chat-builtins"}))));

        let valid_action = json!({"Action": "caps"});
        for bad_req in [
            json!({"SchemaVersion": 2, "Operations": [{"Kind": "chat-template-trace", "Template": "x", "Bos": "", "Eos": "", "StopOnError": true, "Actions": []}]}),
            json!({"SchemaVersion": 1, "Operations": []}),
            json!({"SchemaVersion": 1, "Operations": vec![json!({"Kind": "chat-template-trace", "Template": "x", "Bos": "", "Eos": "", "StopOnError": true, "Actions": []}); 129]}),
            wrap(
                &json!({"Kind": "chat-template-trace", "Template": "x\0", "Bos": "", "Eos": "", "StopOnError": true, "Actions": []}),
            ),
            wrap(
                &json!({"Kind": "chat-template-trace", "Template": "x", "Bos": "", "Eos": "", "StopOnError": "yes", "Actions": []}),
            ),
            wrap(
                &json!({"Kind": "chat-template-trace", "Template": "x", "Bos": "", "Eos": "", "StopOnError": true, "Actions": vec![valid_action; 129]}),
            ),
            wrap(
                &json!({"Kind": "chat-template-trace", "Template": "x", "Bos": "", "Eos": "", "StopOnError": true, "Actions": [{"Action": "caps", "Extra": 1}]}),
            ),
            wrap(
                &json!({"Kind": "chat-template-trace", "Template": "x", "Bos": "", "Eos": "", "StopOnError": true, "Actions": [{"Action": "apply", "ParametersJson": "{}", "NowUnixSeconds": 4_102_444_801_i64}]}),
            ),
            wrap(
                &json!({"Kind": "chat-template-trace", "Template": "x", "Bos": "", "Eos": "", "StopOnError": true, "Actions": [{"Action": "apply", "ParametersJson": r#"{"unknown":1}"#, "NowUnixSeconds": 0}]}),
            ),
            wrap(
                &json!({"Kind": "chat-template-trace", "Template": "x", "Bos": "", "Eos": "", "StopOnError": true, "Actions": [{"Action": "apply", "ParametersJson": r#"{"enable_thinking":1}"#, "NowUnixSeconds": 0}]}),
            ),
            wrap(
                &json!({"Kind": "chat-template-trace", "Template": "x", "Bos": "", "Eos": "", "StopOnError": true, "Actions": [{"Action": "caps"}, {"Action": "diff", "Left": 0, "Right": 0}]}),
            ),
            wrap(
                &json!({"Kind": "chat-template-trace", "Template": "x", "Bos": "", "Eos": "", "StopOnError": true, "Actions": [{"Action": "diff", "Left": 0, "Right": 0}]}),
            ),
            wrap(
                &json!({"Kind": "chat-template-trace", "Template": "x", "Bos": "", "Eos": "", "StopOnError": true, "Actions": [{"Action": "undefined-probe", "VariablesJson": "[]", "Names": []}]}),
            ),
            wrap(
                &json!({"Kind": "chat-template-trace", "Template": "x", "Bos": "", "Eos": "", "StopOnError": true, "Actions": [{"Action": "undefined-probe", "VariablesJson": "{}", "Names": ["a", "a"]}]}),
            ),
            wrap(
                &json!({"Kind": "chat-template-trace", "Template": "x", "Bos": "", "Eos": "", "StopOnError": true, "Actions": [{"Action": "undefined-probe", "VariablesJson": "{}", "Names": (0..129).map(|i| format!("v{i}")).collect::<Vec<_>>()} ]}),
            ),
        ] {
            assert!(execute(&bad_req).is_err(), "{bad_req}");
        }
    }
}
