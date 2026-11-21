//! Clean-room architecture inventory, CPU device discovery, and chat template rendering.
//!
//! No `bonsai_prism_runtime`, `bonsai_gguf`, or native GGML libraries are linked or called.

use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

use super::gguf_model::{GgufValue, ModelData};
use crate::reference_jinja::{JVal, eval_program_chat_fallback, parse_template_with_limit};
use crate::{Result, invalid};

const ARCH_FIXTURE: &str = include_str!("../architecture-inputs.json");

pub(super) fn execute_devices(operation: &Value) -> Result<Value> {
    if operation
        .as_object()
        .is_none_or(|o| o.len() != 1 || o.get("Op").and_then(Value::as_str) != Some("devices"))
    {
        return Err(invalid("invalid devices fields"));
    }
    Ok(json!({
        "Devices": [{
            "Index": 0,
            "Type": 0,
            "Name": "CPU",
            "Description": "Clean-room Safe Rust CPU",
            "CpuBuffer": true,
        }]
    }))
}

pub(super) fn execute_architecture_inventory(operation: &Value) -> Result<Value> {
    if operation.as_object().is_none_or(|o| {
        o.len() != 1 || o.get("Op").and_then(Value::as_str) != Some("architecture-inventory")
    }) {
        return Err(invalid("invalid architecture-inventory fields"));
    }
    let fixture: Value = serde_json::from_str(ARCH_FIXTURE)?;
    let rows = fixture["Architectures"]
        .as_array()
        .ok_or_else(|| invalid("architecture fixture array missing"))?;
    let mut arches = Vec::with_capacity(rows.len());
    for (idx, row) in rows.iter().enumerate() {
        let name = row["Name"]
            .as_str()
            .ok_or_else(|| invalid("architecture fixture Name missing"))?;
        let supported = row["Supported"].as_bool().unwrap_or(false);
        arches.push(json!({
            "Id": idx,
            "Name": name,
            "ParsedId": idx,
            "Saver": supported,
            "TensorSplit": false,
        }));
    }
    Ok(json!({
        "Architectures": arches,
        "WebGpu": false,
        "Hip": false,
    }))
}

pub(super) fn execute_model_chat_render(operation: &Value, model: &ModelData) -> Result<Value> {
    let msgs_arr = operation["Messages"]
        .as_array()
        .ok_or_else(|| invalid("model-chat-render Messages must be array"))?;
    if msgs_arr.len() > 256 {
        return Err(invalid("model-chat-render messages exceed 256"));
    }
    let enable_thinking = operation["EnableThinking"]
        .as_bool()
        .ok_or_else(|| invalid("EnableThinking must be boolean"))?;
    let add_generation = operation["AddGeneration"]
        .as_bool()
        .ok_or_else(|| invalid("AddGeneration must be boolean"))?;
    let mut messages = Vec::with_capacity(msgs_arr.len());
    let mut supplied = 0_usize;
    for msg in msgs_arr {
        let role = msg["role"]
            .as_str()
            .ok_or_else(|| invalid("chat message missing role"))?;
        let content = msg["content"]
            .as_str()
            .ok_or_else(|| invalid("chat message missing content"))?;
        supplied += role.len() + content.len();
        if role.len() > 128 || content.len() > 1_048_576 || supplied > 1_048_576 {
            return Err(invalid("model chat messages exceed 1 MiB"));
        }
        messages.push((role.to_owned(), content.to_owned()));
    }
    let template_str = model
        .kvs
        .get("tokenizer.chat_template")
        .and_then(GgufValue::as_str);
    let rendered = if let Some(src) = template_str
        && let Ok(prompt) = render_jinja_chat(src, &messages, enable_thinking, add_generation)
    {
        prompt
    } else {
        render_chatml_fallback(&messages, add_generation)
    };
    Ok(json!({ "Prompt": rendered }))
}

fn render_jinja_chat(
    template: &str,
    messages: &[(String, String)],
    enable_thinking: bool,
    add_generation: bool,
) -> Result<String> {
    let program = parse_template_with_limit(template, 256 * 1024)
        .map_err(|e| invalid(format!("chat template parse failed: {}", e.message())))?;
    let j_msgs: Vec<JVal> = messages
        .iter()
        .map(|(role, content)| {
            JVal::object(vec![
                (JVal::text_str("role"), JVal::text_str(role)),
                (JVal::text_str("content"), JVal::text_str(content)),
            ])
        })
        .collect();
    let globals = JVal::object(vec![
        (JVal::text_str("messages"), JVal::array(j_msgs)),
        (
            JVal::text_str("add_generation_prompt"),
            JVal::boolean(add_generation),
        ),
        (
            JVal::text_str("enable_thinking"),
            JVal::boolean(enable_thinking),
        ),
        (JVal::text_str("bos_token"), JVal::text_str("")),
        (JVal::text_str("eos_token"), JVal::text_str("")),
    ]);
    let now = i64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    )
    .unwrap_or(0);
    let spans = eval_program_chat_fallback(&program, &globals, Some(now))
        .map_err(|e| invalid(format!("chat template eval failed: {}", e.message())))?;
    let mut bytes = Vec::new();
    for span in spans {
        bytes.extend_from_slice(&span.bytes);
    }
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn render_chatml_fallback(messages: &[(String, String)], add_generation: bool) -> String {
    let mut out = String::new();
    for (role, content) in messages {
        out.push_str("<|im_start|>");
        out.push_str(role);
        out.push('\n');
        out.push_str(content);
        out.push_str("<|im_end|>\n");
    }
    if add_generation {
        out.push_str("<|im_start|>assistant\n");
    }
    out
}
