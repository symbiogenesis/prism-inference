//! Production semantic mapping over generic PEG captures.
use super::{ChatMessage, ToolCall};
use crate::{Result, invalid, peg::Capture};

pub(super) fn normalize_quotes(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut output = Vec::new();
    let mut index = 0;
    let mut single = false;
    let mut double = false;
    let word = |byte: u8| byte.is_ascii_alphanumeric() || byte == b'_';
    while index < bytes.len() {
        let byte = bytes[index];
        if byte == b'\\' && index + 1 < bytes.len() {
            let next = bytes[index + 1];
            if single {
                if next == b'\'' {
                    output.push(b'\'');
                    index += 2;
                    continue;
                }
                if next == b'"' {
                    output.extend(b"\\\"");
                    index += 2;
                    continue;
                }
                output.extend([byte, next]);
                index += 2;
                continue;
            }
            if double {
                output.extend([byte, next]);
                index += 2;
                continue;
            }
            output.push(byte);
            index += 1;
            continue;
        }
        match byte {
            b'"' => {
                if single {
                    output.extend(b"\\\"");
                } else {
                    double = !double;
                    output.push(byte);
                }
            }
            b'\'' => {
                if double {
                    output.push(byte);
                } else {
                    single = !single;
                    output.push(b'"');
                }
            }
            b'T' | b'F' | b'N' if !single && !double && (index == 0 || !word(bytes[index - 1])) => {
                let length = bytes[index..]
                    .iter()
                    .take_while(|&&byte| word(byte))
                    .count();
                let token = &bytes[index..index + length];
                let mut found = false;
                for (python, json) in [
                    (b"True".as_slice(), b"true".as_slice()),
                    (b"False", b"false"),
                    (b"None", b"null"),
                ] {
                    if python.starts_with(token) {
                        output.extend(&json[..length]);
                        index += length;
                        found = true;
                        break;
                    }
                }
                if found {
                    continue;
                }
                output.push(byte);
            }
            _ => output.push(byte),
        }
        index += 1;
    }
    String::from_utf8_lossy(&output).into_owned()
}
fn brace_depth(text: &str) -> usize {
    let mut depth = 0_usize;
    let mut string = false;
    let mut escaped = false;
    for byte in text.bytes() {
        if escaped {
            escaped = false;
            continue;
        }
        if byte == b'\\' && string {
            escaped = true;
            continue;
        }
        if byte == b'"' {
            string = !string;
            continue;
        }
        if !string {
            if byte == b'{' {
                depth += 1;
            } else if byte == b'}' {
                depth = depth.saturating_sub(1);
            }
        }
    }
    depth
}
fn trim_end(text: &str) -> &str {
    text.trim_end_matches(|ch: char| ch.is_ascii_whitespace())
}
struct Mapper {
    message: ChatMessage,
    current: Option<ToolCall>,
    index: Option<usize>,
    buffer: String,
    count: usize,
    quote: bool,
}
impl Mapper {
    const fn target(&mut self) -> &mut String {
        match &mut self.current {
            Some(call) if !call.name.is_empty() => &mut call.arguments,
            _ => &mut self.buffer,
        }
    }
    #[allow(
        clippy::too_many_lines,
        reason = "Preserve source semantic tag order and tool/argument streaming transitions together"
    )]
    fn visit(&mut self, node: &Capture, input: &[u8]) -> Result<()> {
        let text = std::str::from_utf8(
            input
                .get(node.start..node.end)
                .ok_or_else(|| invalid("chat capture byte extent invalid"))?,
        )?;
        match node.tag.as_str() {
            "reasoning" => self.message.reasoning_content.push_str(text),
            "content" => self.message.content.push_str(text),
            "tool-open" => {
                self.current = Some(ToolCall::default());
                self.index = None;
                self.buffer.clear();
                self.count = 0;
                self.quote = false;
            }
            "tool-id" => {
                if let Some(call) = &mut self.current {
                    let text = trim_end(text);
                    call.id = text
                        .strip_prefix('"')
                        .and_then(|value| value.strip_suffix('"'))
                        .unwrap_or(text)
                        .into();
                }
            }
            "tool-name" => {
                if let Some(call) = &mut self.current {
                    call.name = trim_end(text).into();
                    if !self.buffer.is_empty() {
                        call.arguments = std::mem::take(&mut self.buffer);
                    } else if call.arguments.is_empty() {
                        call.arguments = "{".into();
                    }
                    if self.index.is_none() {
                        self.index = Some(self.message.tool_calls.len());
                        self.message.tool_calls.push(call.clone());
                    }
                }
            }
            "tool-args" => {
                let text = trim_end(text);
                if self.current.is_some() && text.starts_with('{') {
                    *self.target() = text.into();
                }
            }
            "tool-arg-open" => self.quote = false,
            "tool-arg-name" => {
                if self.current.is_some() {
                    let mut text = trim_end(text);
                    if text.as_bytes().first().is_some_and(u8::is_ascii_whitespace) {
                        text = &text[1..];
                    }
                    let mut entry = if self.count == 0 {
                        String::new()
                    } else {
                        ",".into()
                    };
                    entry.push_str(&serde_json::to_string(text)?);
                    entry.push(':');
                    self.count += 1;
                    let target = self.target();
                    if target.is_empty() {
                        target.push('{');
                    }
                    target.push_str(&entry);
                }
            }
            "tool-arg-value" | "tool-arg-string-value" => {
                if self.current.is_some() {
                    let string = node.tag == "tool-arg-string-value";
                    let mut entry = String::new();
                    if string {
                        if !self.quote {
                            entry.push('"');
                            self.quote = true;
                        }
                        let escaped = serde_json::to_string(text)?;
                        entry.push_str(&escaped[1..escaped.len() - 1]);
                    } else if !text.is_empty() {
                        entry = normalize_quotes(text);
                    }
                    self.target().push_str(&entry);
                }
            }
            "tool-arg-close" => {
                if self.current.is_some() && self.quote {
                    self.target().push('"');
                    self.quote = false;
                }
            }
            "tool-close" => {
                if let Some(call) = &mut self.current {
                    if call.name.is_empty() && !self.buffer.is_empty() {
                        call.arguments = std::mem::take(&mut self.buffer);
                    }
                    if self.quote {
                        call.arguments.push('"');
                        self.quote = false;
                    }
                    let depth = brace_depth(&call.arguments);
                    call.arguments.extend(std::iter::repeat_n('}', depth));
                    if self.index.is_none() && !call.name.is_empty() {
                        self.index = Some(self.message.tool_calls.len());
                        self.message.tool_calls.push(call.clone());
                    }
                }
            }
            _ => {}
        }
        if let (Some(index), Some(call)) = (self.index, &self.current) {
            self.message.tool_calls[index] = call.clone();
        }
        for child in &node.children {
            self.visit(child, input)?;
        }
        Ok(())
    }
}
/// Map streaming semantic captures into a structured assistant message.
///
/// # Errors
/// Returns an error for invalid capture offsets or non-UTF-8 semantic text.
pub fn map_captures(nodes: &[Capture], input: &[u8]) -> Result<ChatMessage> {
    let mut mapper = Mapper {
        message: ChatMessage::default(),
        current: None,
        index: None,
        buffer: String::new(),
        count: 0,
        quote: false,
    };
    for node in nodes {
        mapper.visit(node, input)?;
    }
    if mapper.index.is_none()
        && let Some(mut call) = mapper.current
        && !call.name.is_empty()
    {
        if !mapper.buffer.is_empty() {
            call.arguments = mapper.buffer;
        }
        if mapper.quote && !call.arguments.is_empty() {
            call.arguments.push('"');
        }
        mapper.message.tool_calls.push(call);
    }
    if mapper
        .message
        .reasoning_content
        .bytes()
        .all(|byte| matches!(byte, b' ' | b'\n' | b'\r' | b'\t'))
    {
        mapper.message.reasoning_content.clear();
    }
    Ok(mapper.message)
}
