//! Chat capabilities from pinned production marker and value-usage probes.
//!
//! Ported from MIT-licensed Prism `common/jinja/caps.cpp` at
//! `b87ca7225456b06b0975b8239c66b5aecf07e37c`; attribution is retained in
//! `tools/prism/upstream/LICENSE`.

use super::{Execution, PathSegment, Template};
use crate::Result;

/// The six chat capabilities exposed by the pinned template analysis API.
///
/// These are observations of the source's fixed probes, including its fallback
/// and exception rules. They do not establish general support for every input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "Expose the six independent capability flags of the native API"
)]
pub struct ChatCapabilities {
    /// Whether the tool definition's function name was consumed.
    pub supports_tools: bool,
    /// Whether the single-call probe consumed the assistant's tool calls.
    pub supports_tool_calls: bool,
    /// Whether the system probe consumed the first message's content.
    pub supports_system_role: bool,
    /// Whether a successful parallel probe consumed the second call's function.
    pub supports_parallel_tool_calls: bool,
    /// Whether the content probe accepted its supplied string.
    pub supports_string_content: bool,
    /// Whether the content probe attempted array access or `selectattr`.
    pub supports_typed_content: bool,
}

const TOOLS: &str = r#"[{"name":"tool","type":"function","function":{"name":"tool1","description":"Tool description","parameters":{"type":"object","properties":{"arg":{"type":"string","description":"Arg description"}},"required":["arg"]}}}]"#;
const OBJECT_ARGUMENTS: &str = r#"{"arg":"value"}"#;
const STRING_ARGUMENTS: &str = r#""{\"arg\": \"value\"}""#;

fn probe(template: &Template, messages: &str, tools: &str) -> Result<Execution> {
    // Retain the native JSON insertion order, including nested tool definitions.
    template.execute(&format!(
        r#"{{"messages":{messages},"tools":{tools},"bos_token":"","eos_token":"","add_generation_prompt":true}}"#
    ))
}

fn tool_probe(template: &Template, object_arguments: bool, parallel: bool) -> Result<Execution> {
    let arguments = if object_arguments {
        OBJECT_ARGUMENTS
    } else {
        STRING_ARGUMENTS
    };
    let call = |id: &str| {
        format!(
            r#"{{"id":"{id}","type":"function","function":{{"name":"tool1","arguments":{arguments}}}}}"#
        )
    };
    let first = call("call00001");
    let second = if parallel {
        format!(",{}", call("call00002"))
    } else {
        String::new()
    };
    probe(
        template,
        &format!(
            r#"[{{"role":"user","content":"User message"}},{{"role":"assistant","content":"","tool_calls":[{first}{second}]}},{{"role":"tool","content":"Tool response","tool_call_id":"call00001"}},{{"role":"assistant","content":"The tool response was 'tool response'"}},{{"role":"user","content":"User message"}}]"#
        ),
        TOOLS,
    )
}

fn message_path(index: usize, member: &str) -> Vec<PathSegment> {
    vec![
        PathSegment::Key("messages".into()),
        PathSegment::Index(index),
        PathSegment::Key(member.into()),
    ]
}

fn tool_name_used(execution: &Execution) -> bool {
    execution.used(&[
        PathSegment::Key("tools".into()),
        PathSegment::Index(0),
        PathSegment::Key("function".into()),
        PathSegment::Key("name".into()),
    ])
}

/// Execute the fixed native probes for the six exposed chat flags.
pub(super) fn chat_capabilities(template: &Template) -> Result<ChatCapabilities> {
    let (supports_string_content, supports_typed_content) = template.content_capabilities()?;
    let system = probe(
        template,
        r#"[{"role":"system","content":"System message"},{"role":"user","content":"User message"}]"#,
        "[]",
    )?;
    let mut result = ChatCapabilities {
        supports_tools: true,
        supports_tool_calls: true,
        supports_system_role: system.used(&message_path(0, "content")),
        supports_parallel_tool_calls: false,
        supports_string_content,
        supports_typed_content,
    };

    // A failed object probe infers nothing. The fallback must use string
    // arguments unless successful execution consumed the object's actual arg.
    let object = tool_probe(template, true, false)?;
    let calls = message_path(1, "tool_calls");
    let mut object_arguments = false;
    if object.error.is_none() {
        result.supports_tools = tool_name_used(&object);
        result.supports_tool_calls = object.used(&calls);
        if result.supports_tool_calls {
            let mut argument = calls.clone();
            argument.extend([
                PathSegment::Index(0),
                PathSegment::Key("function".into()),
                PathSegment::Key("arguments".into()),
                PathSegment::Key("arg".into()),
            ]);
            object_arguments = object.used(&argument);
        }
    }
    if !object_arguments {
        let string = tool_probe(template, false, false)?;
        if string.error.is_some() {
            result.supports_tools = false;
            result.supports_tool_calls = false;
        } else {
            result.supports_tools &= tool_name_used(&string);
            result.supports_tool_calls &= string.used(&calls);
        }
    }
    let parallel = tool_probe(template, object_arguments, true)?;
    let mut second = calls;
    second.extend([PathSegment::Index(1), PathSegment::Key("function".into())]);
    result.supports_parallel_tool_calls = parallel.error.is_none() && parallel.used(&second);
    Ok(result)
}

/// Raw observations from one production rendering with statistics enabled.
pub(super) struct Observation {
    /// Rendered bytes, including any non-UTF-8 bytes accepted by the runtime.
    pub(super) rendered: Vec<u8>,
    /// Whether rendering completed without a runtime exception.
    pub(super) success: bool,
    /// Whether `messages[0].content` recorded `selectattr` or `array_access`.
    pub(super) content_array_access: bool,
}

/// Returns string and typed-content support using the exact upstream probe.
/// Other upstream capability fields are not exposed by this API.
pub(super) fn detect(mut render: impl FnMut(&str) -> Result<Observation>) -> Result<(bool, bool)> {
    const VARIABLES: &str = r#"{"messages":[{"role":"user","content":"STRING_MARKER"}],"tools":[],"bos_token":"","eos_token":"","add_generation_prompt":true}"#;
    let observation = render(VARIABLES)?;
    let typed = observation.content_array_access;
    let marker = observation
        .rendered
        .windows(b"STRING_MARKER".len())
        .any(|bytes| bytes == b"STRING_MARKER");
    let string = observation.success && (!typed || marker);
    Ok((string, typed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_probe_distinguishes_iteration_from_rendered_marker() -> Result<()> {
        for (rendered, success, array, expected) in [
            (b"STRING_MARKER".as_slice(), true, false, (true, false)),
            (b"STRING_MARKER".as_slice(), true, true, (true, true)),
            (b"".as_slice(), true, true, (false, true)),
            (b"".as_slice(), false, false, (false, false)),
        ] {
            assert_eq!(
                detect(|variables| {
                    assert!(variables.contains("STRING_MARKER"));
                    Ok(Observation {
                        rendered: rendered.to_vec(),
                        success,
                        content_array_access: array,
                    })
                })?,
                expected
            );
        }
        Ok(())
    }
}
