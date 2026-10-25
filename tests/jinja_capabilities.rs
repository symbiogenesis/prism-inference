#![forbid(unsafe_code)]
#![cfg(feature = "cli")]
//! Focused chat capability probes and opt-in pinned production differentials.

use bonsai_prism_runtime::Result;
use bonsai_prism_runtime::jinja::{ChatCapabilities, Template};
use serde_json::{Value, json};

fn observed(source: &str) -> Result<ChatCapabilities> {
    Template::parse(source)?.chat_capabilities()
}

#[test]
fn plain_and_typed_content_keep_the_existing_api() -> Result<()> {
    let template =
        Template::parse("{% for message in messages %}{{ message.content }}{% endfor %}")?;
    let actual = template.chat_capabilities()?;
    assert_eq!(
        actual,
        ChatCapabilities {
            supports_tools: false,
            supports_tool_calls: false,
            supports_system_role: true,
            supports_parallel_tool_calls: false,
            supports_string_content: true,
            supports_typed_content: false,
        }
    );
    assert_eq!(template.content_capabilities()?, (true, false));
    let typed = Template::parse("{{ messages[0].content[0].text }}")?;
    assert_eq!(typed.content_capabilities()?, (false, true));
    let actual = typed.chat_capabilities()?;
    assert!(!actual.supports_string_content);
    assert!(actual.supports_typed_content);
    Ok(())
}

#[test]
fn object_arguments_and_parallel_calls_follow_actual_value_usage() -> Result<()> {
    let all = observed(
        "{% for tool in tools %}{{ tool.function.name }}{% endfor %}{% for message in messages %}{{ message.content }}{% for call in message.tool_calls %}{{ call.function.arguments.arg }}{% endfor %}{% endfor %}",
    )?;
    assert!(all.supports_tools && all.supports_tool_calls && all.supports_parallel_tool_calls);
    let first = observed(
        "{% for tool in tools %}{{ tool.function.name }}{% endfor %}{% for message in messages %}{% if message.tool_calls %}{{ message.tool_calls[0].function.arguments.arg }}{% endif %}{% endfor %}",
    )?;
    assert!(first.supports_tools && first.supports_tool_calls);
    assert!(!first.supports_parallel_tool_calls);
    let tools_only = observed("{% for tool in tools %}{{ tool.function.name }}{% endfor %}")?;
    assert!(tools_only.supports_tools);
    assert!(!tools_only.supports_tool_calls && !tools_only.supports_parallel_tool_calls);
    Ok(())
}

#[test]
fn string_fallback_recovers_from_object_rejection() -> Result<()> {
    let actual = observed(
        "{% for message in messages %}{% for call in message.tool_calls %}{% if call.function.arguments is not string %}{{ raise_exception('string required') }}{% endif %}{{ call.function.arguments }}{% endfor %}{% endfor %}{% for tool in tools %}{{ tool.function.name }}{% endfor %}",
    )?;
    assert!(
        actual.supports_tools && actual.supports_tool_calls && actual.supports_parallel_tool_calls
    );
    Ok(())
}

#[test]
fn object_probe_must_consume_the_argument_before_skipping_fallback() -> Result<()> {
    // Looking at the argument's type does not consume its nested `arg` value.
    // The native contract still attempts string fallback, which this rejects.
    let actual = observed(
        "{% for tool in tools %}{{ tool.function.name }}{% endfor %}{% for message in messages %}{% for call in message.tool_calls %}{% if call.function.arguments is string %}{{ raise_exception('object required') }}{% endif %}{% endfor %}{% endfor %}",
    )?;
    assert!(!actual.supports_tools && !actual.supports_tool_calls);
    assert!(!actual.supports_parallel_tool_calls);
    Ok(())
}

#[test]
fn system_usage_survives_an_exception_but_parallel_usage_does_not() -> Result<()> {
    let system = observed(
        "{% if messages[0].role == 'system' %}{{ messages[0].content }}{{ raise_exception('after system') }}{% endif %}",
    )?;
    assert!(system.supports_system_role);
    let rejected = observed(
        "{% if messages[0].role == 'system' %}{{ raise_exception('before system') }}{% endif %}",
    )?;
    assert!(!rejected.supports_system_role);
    let parallel = observed(
        "{% for tool in tools %}{{ tool.function.name }}{% endfor %}{% for message in messages %}{% for call in message.tool_calls %}{{ call.function.arguments.arg }}{% if loop.index == 2 %}{{ raise_exception('after second') }}{% endif %}{% endfor %}{% endfor %}",
    )?;
    assert!(parallel.supports_tools && parallel.supports_tool_calls);
    assert!(!parallel.supports_parallel_tool_calls);
    Ok(())
}

fn wire(caps: ChatCapabilities) -> Value {
    json!({
        "supports_tools":caps.supports_tools,
        "supports_tool_calls":caps.supports_tool_calls,
        "supports_system_role":caps.supports_system_role,
        "supports_parallel_tool_calls":caps.supports_parallel_tool_calls,
        "supports_string_content":caps.supports_string_content,
        "supports_typed_content":caps.supports_typed_content,
    })
}

#[test]
#[ignore = "requires BONSAI_PRISM_CAPS_REFERENCE, BONSAI_PRISM_CAPS_SOURCE and BONSAI_PRISM_CAPS_EVIDENCE absolute paths"]
fn native_chat_capabilities_match_pinned_templates() -> Result<()> {
    let reference = std::path::PathBuf::from(std::env::var("BONSAI_PRISM_CAPS_REFERENCE")?);
    let source = std::path::PathBuf::from(std::env::var("BONSAI_PRISM_CAPS_SOURCE")?);
    let evidence = std::path::PathBuf::from(std::env::var("BONSAI_PRISM_CAPS_EVIDENCE")?);
    assert!(reference.is_absolute() && source.is_absolute() && evidence.is_absolute());
    std::fs::create_dir_all(&evidence)?;
    let fixture: Value =
        serde_json::from_str(include_str!("upstream_prism/chat-analysis-inputs.json"))?;
    let paths = fixture["TemplatePaths"]
        .as_array()
        .ok_or("missing template paths")?;
    assert_eq!(paths.len(), 37);
    let mut operations = Vec::new();
    let mut actual = Vec::new();
    for path in paths {
        let relative = path.as_str().ok_or("template path must be text")?;
        let text = std::fs::read_to_string(source.join(relative))?;
        operations.push(json!({"Kind":"chat-template-trace","Template":text,"Bos":"","Eos":"","StopOnError":true,"Actions":[{"Action":"caps"}]}));
        let template = Template::parse(&text)?;
        let content = template.content_capabilities()?;
        let content_error = template.execute(r#"{"messages":[{"role":"user","content":"STRING_MARKER"}],"tools":[],"bos_token":"","eos_token":"","add_generation_prompt":true}"#)?.error;
        actual.push((wire(template.chat_capabilities()?), content, content_error));
    }
    let input = evidence.join("caps-request.json");
    let output = evidence.join("caps-reference.json");
    std::fs::write(
        &input,
        serde_json::to_vec(&json!({"SchemaVersion":1,"Operations":operations}))?,
    )?;
    let status = std::process::Command::new(reference)
        .args([
            "--input",
            input.to_str().ok_or("non-UTF-8 input path")?,
            "--output",
            output.to_str().ok_or("non-UTF-8 output path")?,
        ])
        .current_dir(&evidence)
        .output()?;
    std::fs::write(evidence.join("caps-reference.stdout"), status.stdout)?;
    std::fs::write(evidence.join("caps-reference.stderr"), status.stderr)?;
    assert!(status.status.success());
    let native: Value = serde_json::from_slice(&std::fs::read(output)?)?;
    assert_eq!(native["SchemaVersion"], 1);
    let results = native["Results"]
        .as_array()
        .ok_or("missing native results")?;
    assert_eq!(results.len(), paths.len());
    let mut mismatches = Vec::new();
    for ((path, (actual, content, content_error)), native) in paths.iter().zip(actual).zip(results)
    {
        assert!(native["TemplateError"].is_null(), "{path}");
        let expected = &native["Trace"][0]["Caps"];
        if &actual != expected {
            mismatches.push(json!({"Template":path,"Rust":actual,"Native":expected,
                "PreexistingContent":{"String":content.0,"Typed":content.1,"ProbeError":content_error}}));
        }
    }
    std::fs::write(
        evidence.join("caps-mismatches.json"),
        serde_json::to_vec_pretty(&mismatches)?,
    )?;
    assert!(
        mismatches.is_empty(),
        "capability mismatches: {mismatches:?}"
    );
    Ok(())
}
