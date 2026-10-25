//! Owned common chat messages and monotonic streaming deltas.
use crate::{Result, invalid};
use serde::{Deserialize, Serialize};

/// A typed message content range.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContentPart {
    /// Source content category.
    #[serde(rename = "type")]
    pub kind: String,
    /// Raw text of this range.
    pub text: String,
}
/// A generated function invocation.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCall {
    /// Function name.
    pub name: String,
    /// Streaming JSON argument text.
    pub arguments: String,
    /// Provider call identifier.
    pub id: String,
}
/// Structured message shared by formatting and incremental parsing.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct ChatMessage {
    /// Source message role.
    pub role: String,
    /// Plain message text.
    pub content: String,
    /// Typed content alternatives.
    pub content_parts: Vec<ContentPart>,
    /// Function calls in generation order.
    pub tool_calls: Vec<ToolCall>,
    /// Separate reasoning text.
    pub reasoning_content: String,
    /// Tool result function name.
    pub tool_name: String,
    /// Tool result call identifier.
    pub tool_call_id: String,
}
/// One incremental change in the source observation order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MessageDiff {
    /// Added reasoning bytes.
    pub reasoning_content_delta: String,
    /// Added message bytes.
    pub content_delta: String,
    /// Call affected by the change, absent for ordinary text deltas.
    pub tool_call_index: Option<usize>,
    /// Added argument bytes and updated full name/id when they change.
    pub tool_call_delta: ToolCall,
}
fn string_diff(previous: &str, current: &str) -> Result<String> {
    current.strip_prefix(previous).map_or_else(
        || {
            if previous.starts_with(current) {
                Ok(String::new())
            } else {
                Err(invalid("chat delta is not a prefix extension"))
            }
        },
        |suffix| Ok(suffix.to_owned()),
    )
}
impl ChatMessage {
    /// Compute source-compatible deltas for growing messages and call arguments.
    ///
    /// # Errors
    /// Rejects non-prefix text/argument rewrites, removed calls and changed call names.
    pub fn diff(&self, current: &Self) -> Result<Vec<MessageDiff>> {
        let mut diffs = Vec::new();
        if self.reasoning_content != current.reasoning_content {
            diffs.push(MessageDiff {
                reasoning_content_delta: string_diff(
                    &self.reasoning_content,
                    &current.reasoning_content,
                )?,
                ..MessageDiff::default()
            });
        }
        if self.content != current.content {
            diffs.push(MessageDiff {
                content_delta: string_diff(&self.content, &current.content)?,
                ..MessageDiff::default()
            });
        }
        if current.tool_calls.len() < self.tool_calls.len() {
            return Err(invalid("chat delta removes tool calls"));
        }
        if let Some(index) = self.tool_calls.len().checked_sub(1) {
            let before = &self.tool_calls[index];
            let after = &current.tool_calls[index];
            if before.name != after.name
                && !before.name.is_empty()
                && !after.name.is_empty()
                && !after.name.starts_with(&before.name)
            {
                return Err(invalid("chat delta tool name mismatch"));
            }
            let arguments = string_diff(&before.arguments, &after.arguments)?;
            if !arguments.is_empty() || before.id != after.id || before.name != after.name {
                diffs.push(MessageDiff {
                    tool_call_index: Some(index),
                    tool_call_delta: ToolCall {
                        name: if before.id != after.id || before.name != after.name {
                            after.name.clone()
                        } else {
                            String::new()
                        },
                        id: if before.id != after.id || before.name != after.name {
                            after.id.clone()
                        } else {
                            String::new()
                        },
                        arguments,
                    },
                    ..MessageDiff::default()
                });
            }
        }
        for (index, call) in current
            .tool_calls
            .iter()
            .enumerate()
            .skip(self.tool_calls.len())
        {
            diffs.push(MessageDiff {
                tool_call_index: Some(index),
                tool_call_delta: call.clone(),
                ..MessageDiff::default()
            });
        }
        Ok(diffs)
    }
}
