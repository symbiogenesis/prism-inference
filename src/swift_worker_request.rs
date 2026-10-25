//! Private one-model Swift JSONL request schema shared by the worker and its tests.
#![allow(
    clippy::redundant_pub_crate,
    reason = "Keep the shared worker/test schema explicitly crate scoped rather than publicly reachable"
)]

use std::fmt;

use bonsai_prism_runtime::swift_chat::{MAX_MESSAGE_BYTES, MAX_MESSAGES};
use serde::{
    Deserialize, Deserializer,
    de::{self, MapAccess, SeqAccess, Visitor},
};
use serde_json::{Value, json};

/// Largest complete request frame, including its LF terminator.
pub(crate) const MAX_FRAME_BYTES: usize = 1024 * 1024;

/// Strict request shape; message objects are validated before conversion to JSON values.
#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Request {
    Count {
        id: u64,
        #[serde(deserialize_with = "messages")]
        messages: Value,
        max_tokens: usize,
    },
    Complete {
        id: u64,
        #[serde(deserialize_with = "messages")]
        messages: Value,
        max_tokens: usize,
        timeout_seconds: u64,
    },
    Reset {
        id: u64,
    },
}

/// Parse one bounded complete frame using the exact worker request schema.
pub(crate) fn parse(frame: &[u8]) -> Result<Request, serde_json::Error> {
    if frame.len() > MAX_FRAME_BYTES
        || frame.last() != Some(&b'\n')
        || frame.trim_ascii_start().first() != Some(&b'{')
    {
        return Err(de::Error::custom("Swift request exceeds frame bound"));
    }
    serde_json::from_slice(frame)
}

struct TextMessage {
    role: String,
    content: String,
    reasoning_content: Option<String>,
}

impl<'de> Deserialize<'de> for TextMessage {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct MessageVisitor;
        impl<'de> Visitor<'de> for MessageVisitor {
            type Value = TextMessage;
            fn expecting(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
                output.write_str("a Swift text message object")
            }
            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<TextMessage, M::Error> {
                let mut role = None;
                let mut content = None;
                let mut reasoning_content = None;
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "role" if role.is_none() => role = Some(map.next_value::<String>()?),
                        "content" if content.is_none() => {
                            content = Some(map.next_value::<String>()?);
                        }
                        "reasoning_content" if reasoning_content.is_none() => {
                            reasoning_content = Some(map.next_value::<String>()?);
                        }
                        "role" | "content" | "reasoning_content" => {
                            return Err(de::Error::custom("duplicate Swift message field"));
                        }
                        _ => return Err(de::Error::custom("unknown Swift message field")),
                    }
                }
                Ok(TextMessage {
                    role: role.ok_or_else(|| de::Error::missing_field("role"))?,
                    content: content.ok_or_else(|| de::Error::missing_field("content"))?,
                    reasoning_content,
                })
            }
        }
        // deserialize_map deliberately excludes positional/tuple message arrays.
        deserializer.deserialize_map(MessageVisitor)
    }
}

fn messages<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Value, D::Error> {
    struct MessagesVisitor;
    impl<'de> Visitor<'de> for MessagesVisitor {
        type Value = Vec<TextMessage>;
        fn expecting(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
            output.write_str("a bounded array of Swift text message objects")
        }
        fn visit_seq<S: SeqAccess<'de>>(self, mut sequence: S) -> Result<Self::Value, S::Error> {
            let mut messages = Vec::new();
            messages
                .try_reserve_exact(MAX_MESSAGES)
                .map_err(|_| de::Error::custom("Swift message allocation failed"))?;
            let mut supplied_bytes = 0usize;
            while let Some(message) = sequence.next_element::<TextMessage>()? {
                if messages.len() == MAX_MESSAGES {
                    return Err(de::Error::custom("Swift message count exceeds bound"));
                }
                supplied_bytes = supplied_bytes
                    .checked_add(message.content.len())
                    .and_then(|bytes| {
                        bytes.checked_add(message.reasoning_content.as_ref().map_or(0, String::len))
                    })
                    .filter(|&bytes| bytes <= MAX_MESSAGE_BYTES)
                    .ok_or_else(|| de::Error::custom("Swift message bytes exceed bound"))?;
                messages.push(message);
            }
            if messages.is_empty() {
                return Err(de::Error::custom("Swift messages must be nonempty"));
            }
            Ok(messages)
        }
    }
    let messages = deserializer.deserialize_seq(MessagesVisitor)?;
    Ok(Value::Array(
        messages
            .into_iter()
            .map(|message| {
                let mut value = json!({"role":message.role,"content":message.content});
                if let Some(reasoning) = message.reasoning_content {
                    value["reasoning_content"] = Value::String(reasoning);
                }
                value
            })
            .collect(),
    ))
}
