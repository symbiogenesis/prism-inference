//! Production chat template rendering built on the reusable Jinja runtime.
//!
//! MIT-licensed upstream contracts: `common/chat.cpp` at
//! `b87ca7225456b06b0975b8239c66b5aecf07e37c`, attribution in
//! `tools/prism/upstream/LICENSE`.
use crate::{
    Result, invalid,
    jinja::{PathSegment, Template},
};
use serde_json::{Value, json};
#[path = "chat_legacy.rs"]
mod legacy;
#[path = "chat_message.rs"]
mod message;
pub use message::{ChatMessage, ContentPart, MessageDiff, ToolCall};
#[path = "chat_mapper.rs"]
mod mapper;
pub use mapper::map_captures;
#[path = "chat_json.rs"]
pub(crate) mod json;

/// Registered legacy formatter names in the source map's lexical order.
#[must_use]
pub const fn builtin_template_names() -> &'static [&'static str] {
    legacy::NAMES
}
/// Render a recognized legacy template name or detected template source.
///
/// # Errors
/// Returns an error for unrecognized formats or message/output bound violations.
pub fn render_legacy(source: &str, messages: &[Message], add_generation: bool) -> Result<String> {
    legacy::render(source, messages, add_generation)
}

/// A plain text chat turn.
#[derive(Clone, Debug)]
pub struct Message {
    /// Source role, such as system, user or assistant.
    pub role: String,
    /// Source content, preserving whitespace.
    pub content: String,
}
/// Parsed chat source and its observed role capability.
pub struct ChatTemplate {
    program: Template,
    bos: String,
    eos: String,
    supports_system: bool,
    map_developer: bool,
}
impl ChatTemplate {
    /// Parse the template and observe whether it consumes a system message.
    ///
    /// # Errors
    /// Returns an error for malformed or oversized template syntax.
    pub fn new(source: &str, bos: &str, eos: &str) -> Result<Self> {
        let program = Template::parse(source)?;
        let probe=program.execute(r#"{"messages":[{"role":"system","content":"System message"},{"role":"user","content":"User message"}],"tools":[],"bos_token":"","eos_token":"","add_generation_prompt":true}"#)?;
        let supports_system = probe.used(&[
            PathSegment::Key("messages".into()),
            PathSegment::Index(0),
            PathSegment::Key("content".into()),
        ]);
        Ok(Self {
            program,
            bos: bos.into(),
            eos: eos.into(),
            supports_system,
            map_developer: !source.contains("<|channel|>"),
        })
    }
    /// Render plain text turns using the source role workarounds.
    ///
    /// # Errors
    /// Returns an error for invalid values, a template exception, or resource bounds.
    pub fn render(&self, messages: &[Message], add_generation: bool) -> Result<String> {
        let messages = messages
            .iter()
            .map(|message| json!({"role":message.role,"content":message.content}))
            .collect();
        self.render_json(
            Value::Array(messages),
            add_generation,
            &Value::Object(serde_json::Map::new()),
        )
    }
    /// Render structured messages with additional template variables.
    ///
    /// # Errors
    /// Returns an error for invalid message shapes, a template exception or bounds.
    pub fn render_json(
        &self,
        mut messages: Value,
        add_generation: bool,
        extra: &Value,
    ) -> Result<String> {
        let values = messages
            .as_array_mut()
            .ok_or_else(|| invalid("chat messages must be an array"))?;
        if values.len() > 1024 {
            return Err(invalid("chat message count bound exceeded"));
        }
        if self.map_developer {
            for message in &mut *values {
                if message.get("role").and_then(Value::as_str) == Some("developer") {
                    message["role"] = json!("system");
                }
            }
        }
        if !self.supports_system
            && values
                .first()
                .and_then(|message| message.get("role"))
                .and_then(Value::as_str)
                == Some("system")
        {
            let system = values.remove(0);
            if let Some(next) = values.first_mut() {
                let before = system
                    .get("content")
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid("chat system content must be a string"))?;
                let after = next
                    .get("content")
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid("chat content must be a string"))?;
                next["content"] = json!(format!("{before}\n{after}"));
            }
        }
        let mut variables = json!({"messages":messages,"bos_token":self.bos,"eos_token":self.eos,"add_generation_prompt":add_generation,"enable_thinking":true});
        if !add_generation {
            variables
                .as_object_mut()
                .ok_or_else(|| invalid("chat variable object missing"))?
                .remove("add_generation_prompt");
        }
        let extra = extra
            .as_object()
            .ok_or_else(|| invalid("chat extra variables must be an object"))?;
        for (key, value) in extra {
            variables[key] = value.clone();
        }
        let result = self.program.execute(&serde_json::to_string(&variables)?)?;
        if let Some(error) = result.error {
            return Err(invalid(error));
        }
        String::from_utf8(result.bytes()).map_err(Into::into)
    }
}
