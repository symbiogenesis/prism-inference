//! Bounded private HTTP service for the independently released Prism runtime.
//!
//! The owning launcher retains process, Job, artifact and memory protections.
#![forbid(unsafe_code)]

mod broker_pipe;
mod configuration;
pub mod native_backend;
#[cfg(windows)]
pub mod native_child;
pub mod native_response;
pub mod native_swift;
mod reception;
pub mod response;
#[cfg(windows)]
mod service;
pub mod startup;
mod wire;

use serde::{Deserialize, Deserializer, Serialize, de};
use std::{borrow::Cow, fmt, marker::PhantomData};
use subtle::ConstantTimeEq;

/// Largest accepted text-only client request body in bytes.
pub const MAX_BODY_BYTES: usize = 1024 * 1024;
/// Largest accepted multimodal client request body in bytes.
pub const MAX_MULTIMODAL_BODY_BYTES: usize = 16 * 1024 * 1024;
/// Largest accepted HTTP header block in bytes.
pub const MAX_HEADER_BYTES: usize = 16 * 1024;
/// Largest accepted number of HTTP header fields, including duplicates.
pub const MAX_HEADERS: usize = 64;
/// Largest accepted number of messages in one chat history.
pub const MAX_MESSAGES: usize = 128;
/// Largest accepted number of content parts inside one user message.
pub const MAX_CONTENT_PARTS: usize = 32;
/// Largest allowed completion token budget.
pub const MAX_COMPLETION_TOKENS: u32 = 4096;
/// Total prompt and completion context budget of the pinned model.
pub const CONTEXT_TOKENS: usize = 8192;
/// Context reserved for template accounting differences.
pub const TEMPLATE_MARGIN_TOKENS: usize = 64;

/// Deliberately contains no input values, parser diagnostics, keys or prompts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rejection {
    /// The private endpoint port or credential format is invalid.
    Configuration,
    /// Header count, size, multiplicity or forwarding metadata is invalid.
    Headers,
    /// The private authorization credential is missing or invalid.
    Unauthorized,
    /// The Host header does not identify the private endpoint.
    Host,
    /// The supplied Origin header does not match the private endpoint.
    Origin,
    /// The HTTP method, version or request target is unsupported.
    Route,
    /// The request has unsupported or ambiguous HTTP framing.
    Framing,
    /// The request does not declare the supported JSON content type.
    ContentType,
    /// The declared or actual body length is invalid or exceeds the cap.
    BodySize,
    /// The body does not match the strict chat schema or role sequence.
    ChatSchema,
    /// The requested completion token budget is outside the allowed range.
    CompletionLimit,
    /// The trusted counter could not provide a positive prompt token count.
    TokenCountUnavailable,
    /// The prompt, completion and template margin exceed the context budget.
    ContextLimit,
}

/// No Debug/Serialize implementation: credentials must not enter diagnostics.
pub struct RequestPolicy {
    authority: String,
    origin: String,
    authorization: [u8; 71], // "Bearer " + 64 lowercase hex characters
    vision: bool,
}

impl RequestPolicy {
    /// The launcher must supply 32 cryptographically random bytes encoded as
    /// lowercase hex. Format validation here cannot establish randomness or ACLs.
    ///
    /// # Errors
    /// Returns [`Rejection::Configuration`] for a privileged port or invalid key.
    pub fn new(port: u16, key: &str) -> Result<Self, Rejection> {
        if port < 1024 || !configuration::valid_key(key.as_bytes()) {
            return Err(Rejection::Configuration);
        }
        let authority = format!("127.0.0.1:{port}");
        let mut authorization = [0; 71];
        authorization[..7].copy_from_slice(b"Bearer ");
        authorization[7..].copy_from_slice(key.as_bytes());
        Ok(Self {
            origin: format!("http://{authority}"),
            authority,
            authorization,
            vision: false,
        })
    }

    /// Enable or disable multimodal request framing and schema admission.
    #[must_use]
    pub const fn with_vision(mut self, vision: bool) -> Self {
        self.vision = vision;
        self
    }

    const fn max_body_bytes(&self) -> usize {
        if self.vision {
            MAX_MULTIMODAL_BODY_BYTES
        } else {
            MAX_BODY_BYTES
        }
    }

    /// Check the parsed head before the transport accepts a bounded body.
    /// Only HTTP/1.1 origin-form POST /v1/chat/completions is supported.
    /// OPTIONS, readiness, UI, tokenization and administrative routes are closed.
    ///
    /// # Errors
    /// Returns a value-free [`Rejection`] for invalid credentials, headers, route,
    /// content type, framing or body length.
    pub fn check_head(
        &self,
        method: &str,
        target: &str,
        version: u8,
        headers: &[httparse::Header<'_>],
    ) -> Result<AuthorizedHead, Rejection> {
        self.check_common_head(method, target, version, headers)?;
        let content_type = singleton(headers, "content-type")?.ok_or(Rejection::ContentType)?;
        if !content_type.eq_ignore_ascii_case(b"application/json")
            && !content_type.eq_ignore_ascii_case(b"application/json; charset=utf-8")
        {
            return Err(Rejection::ContentType);
        }
        let length = singleton(headers, "content-length")?.ok_or(Rejection::Framing)?;
        if length.is_empty() || !length.iter().all(u8::is_ascii_digit) {
            return Err(Rejection::Framing);
        }
        let length = length
            .iter()
            .try_fold(0usize, |n, b| {
                n.checked_mul(10)?.checked_add(usize::from(b - b'0'))
            })
            .ok_or(Rejection::BodySize)?;
        if length == 0 || length > self.max_body_bytes() {
            return Err(Rejection::BodySize);
        }
        Ok(AuthorizedHead {
            length,
            vision: self.vision,
        })
    }

    pub(crate) fn check_common_head(
        &self,
        method: &str,
        target: &str,
        version: u8,
        headers: &[httparse::Header<'_>],
    ) -> Result<(), Rejection> {
        if headers.len() > MAX_HEADERS || !wire::valid_headers(headers) {
            return Err(Rejection::Headers);
        }
        let bytes = headers.iter().try_fold(0usize, |total, header| {
            total.checked_add(header.name.len() + header.value.len() + 4)
        });
        if bytes.is_none_or(|n| n > MAX_HEADER_BYTES) {
            return Err(Rejection::Headers);
        }
        let auth = singleton(headers, "authorization")?.ok_or(Rejection::Unauthorized)?;
        if auth.len() != self.authorization.len()
            || !bool::from(auth.ct_eq(self.authorization.as_slice()))
        {
            return Err(Rejection::Unauthorized);
        }
        if singleton(headers, "host")? != Some(self.authority.as_bytes()) {
            return Err(Rejection::Host);
        }
        if let Some(origin) = singleton(headers, "origin")?
            && origin != self.origin.as_bytes()
        {
            return Err(Rejection::Origin);
        }
        if method != "POST" || version != 1 || target != "/v1/chat/completions" {
            return Err(Rejection::Route);
        }
        for name in [
            "transfer-encoding",
            "content-encoding",
            "trailer",
            "expect",
            "upgrade",
        ] {
            if contains_header(headers, name) {
                return Err(Rejection::Framing);
            }
        }
        if headers.iter().any(|header| {
            header.name.eq_ignore_ascii_case("forwarded")
                || header
                    .name
                    .get(..12)
                    .is_some_and(|prefix| prefix.eq_ignore_ascii_case("x-forwarded-"))
        }) {
            return Err(Rejection::Headers);
        }
        if let Some(connection) = singleton(headers, "connection")?
            && !connection.eq_ignore_ascii_case(b"close")
            && !connection.eq_ignore_ascii_case(b"keep-alive")
        {
            return Err(Rejection::Framing);
        }
        Ok(())
    }
}

pub(crate) fn singleton<'a>(
    headers: &'a [httparse::Header<'a>],
    name: &str,
) -> Result<Option<&'a [u8]>, Rejection> {
    let mut values = headers
        .iter()
        .filter(|header| header.name.eq_ignore_ascii_case(name))
        .map(|header| header.value);
    let first = values.next();
    if values.next().is_some() {
        return Err(Rejection::Headers);
    }
    Ok(first)
}

pub(crate) fn contains_header(headers: &[httparse::Header<'_>], name: &str) -> bool {
    headers
        .iter()
        .any(|header| header.name.eq_ignore_ascii_case(name))
}

/// Constructed only by successful head validation; does not retain the key.
pub struct AuthorizedHead {
    length: usize,
    vision: bool,
}

impl AuthorizedHead {
    /// Validate a bounded JSON chat body and construct canonical backend bytes.
    ///
    /// # Errors
    /// Returns [`Rejection`] for a mismatched length, invalid schema, unsupported
    /// message history, completion limit or canonical serialization failure.
    pub fn prepare(self, body: &[u8]) -> Result<PendingChat, Rejection> {
        let max_body_bytes = if self.vision {
            MAX_MULTIMODAL_BODY_BYTES
        } else {
            MAX_BODY_BYTES
        };
        if body.len() != self.length || body.len() > max_body_bytes {
            return Err(Rejection::BodySize);
        }
        // Deserialize directly: maps/Value would silently erase duplicate keys.
        // Parser error strings can include private values; never propagate them.
        let Object(chat): Object<ClientChat<'_>> =
            serde_json::from_slice(body).map_err(|_| Rejection::ChatSchema)?;
        if !valid_history_for_vision(&chat.messages, self.vision) {
            return Err(Rejection::ChatSchema);
        }
        if !(1..=MAX_COMPLETION_TOKENS).contains(&chat.max_completion_tokens) {
            return Err(Rejection::CompletionLimit);
        }
        let max_completion_tokens = chat.max_completion_tokens;
        let controls = validate_cognitive_controls(CognitiveControlInput {
            temperature: chat.temperature.unwrap_or(1.0),
            top_p: chat.top_p.unwrap_or(0.95),
            top_k: chat.top_k.unwrap_or(20),
            min_p: chat.min_p.unwrap_or(0.0),
            presence_penalty: chat.presence_penalty.unwrap_or(0.0),
            frequency_penalty: chat.frequency_penalty.unwrap_or(0.0),
            repeat_penalty: chat.repeat_penalty.unwrap_or(1.0),
            enable_thinking: chat
                .chat_template_kwargs
                .as_ref()
                .and_then(|Object(kwargs)| kwargs.enable_thinking)
                .unwrap_or(true),
            reasoning_effort: resolve_client_reasoning_effort(
                chat.reasoning_effort.as_deref(),
                chat.chat_template_kwargs
                    .as_ref()
                    .and_then(|Object(kwargs)| kwargs.reasoning_effort.as_deref()),
            )?,
            preserve_thinking: chat
                .chat_template_kwargs
                .as_ref()
                .and_then(|Object(kwargs)| kwargs.preserve_thinking)
                .unwrap_or(true),
            reasoning_budget: chat.reasoning_budget,
            dry_multiplier: chat.dry_multiplier,
            dry_base: chat.dry_base,
            dry_allowed_length: chat.dry_allowed_length,
            dry_penalty_last_n: chat.dry_penalty_last_n,
            logprobs: chat.logprobs,
            top_logprobs: chat.top_logprobs,
        })
        .ok_or(Rejection::ChatSchema)?;
        let body = serde_json::to_vec(&controls.to_backend_chat(
            &chat.messages,
            max_completion_tokens,
            chat.stream,
        ))
        .map_err(|_| Rejection::ChatSchema)?;
        Ok(PendingChat {
            body,
            max_completion_tokens,
            stream: chat.stream,
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ClientTemplateKwargs<'a> {
    enable_thinking: Option<bool>,
    #[serde(borrow)]
    reasoning_effort: Option<Cow<'a, str>>,
    preserve_thinking: Option<bool>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ClientChat<'a> {
    #[serde(borrow, deserialize_with = "deserialize_messages")]
    messages: Vec<Message<'a>>,
    #[serde(default = "default_cap")]
    max_completion_tokens: u32,
    #[serde(default = "default_stream")]
    stream: bool,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<u32>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    repeat_penalty: Option<f32>,
    #[serde(default, borrow)]
    reasoning_effort: Option<Cow<'a, str>>,
    #[serde(default)]
    reasoning_budget: Option<u32>,
    #[serde(default)]
    dry_multiplier: Option<f32>,
    #[serde(default)]
    dry_base: Option<f32>,
    #[serde(default)]
    dry_allowed_length: Option<u32>,
    #[serde(default)]
    dry_penalty_last_n: Option<u32>,
    #[serde(default)]
    logprobs: Option<bool>,
    #[serde(default)]
    top_logprobs: Option<u8>,
    #[serde(default, borrow)]
    chat_template_kwargs: Option<Object<ClientTemplateKwargs<'a>>>,
}

const fn default_cap() -> u32 {
    MAX_COMPLETION_TOKENS
}
const fn default_stream() -> bool {
    true
}

/// Text or user multimodal parts (`text` + inline base64 image data URLs only).
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Message<'a> {
    role: Role,
    // JSON text without escapes can be borrowed until canonical serialization;
    // escaped text still uses Serde's owned decoding without changing its bytes.
    #[serde(borrow)]
    content: MessageContent<'a>,
    #[serde(
        default,
        borrow,
        deserialize_with = "deserialize_reasoning_content",
        skip_serializing_if = "Option::is_none"
    )]
    reasoning_content: Option<Cow<'a, str>>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum MessageContent<'a> {
    Text(Cow<'a, str>),
    Parts(Vec<ContentPart<'a>>),
}

impl MessageContent<'_> {
    fn image_count(&self) -> usize {
        match self {
            Self::Text(_) => 0,
            Self::Parts(parts) => parts
                .iter()
                .filter(|part| matches!(part, ContentPart::ImageUrl { .. }))
                .count(),
        }
    }
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentPart<'a> {
    Text { text: Cow<'a, str> },
    ImageUrl { image_url: ImageUrl<'a> },
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ImageUrl<'a> {
    #[serde(borrow)]
    url: Cow<'a, str>,
}

enum ContentPartField {
    Type,
    Text,
    ImageUrl,
}

impl<'de> Deserialize<'de> for ContentPartField {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct FieldVisitor;
        impl de::Visitor<'_> for FieldVisitor {
            type Value = ContentPartField;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a supported content part field")
            }
            fn visit_str<E: de::Error>(self, value: &str) -> Result<ContentPartField, E> {
                match value {
                    "type" => Ok(ContentPartField::Type),
                    "text" => Ok(ContentPartField::Text),
                    "image_url" => Ok(ContentPartField::ImageUrl),
                    other => Err(E::unknown_field(other, &["type", "text", "image_url"])),
                }
            }
        }
        deserializer.deserialize_identifier(FieldVisitor)
    }
}

enum ContentPartKind {
    Text,
    ImageUrl,
}

impl<'de> Deserialize<'de> for ContentPartKind {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct KindVisitor;
        impl de::Visitor<'_> for KindVisitor {
            type Value = ContentPartKind;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a content part type")
            }
            fn visit_str<E: de::Error>(self, value: &str) -> Result<ContentPartKind, E> {
                match value {
                    "text" => Ok(ContentPartKind::Text),
                    "image_url" => Ok(ContentPartKind::ImageUrl),
                    _ => Err(E::custom("unsupported content part type")),
                }
            }
        }
        deserializer.deserialize_str(KindVisitor)
    }
}

impl<'de: 'a, 'a> Deserialize<'de> for ContentPart<'a> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct BorrowedStr<'a>(#[serde(borrow)] Cow<'a, str>);
        struct PartVisitor<'a>(PhantomData<&'a ()>);
        impl<'de: 'a, 'a> de::Visitor<'de> for PartVisitor<'a> {
            type Value = ContentPart<'a>;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a text or image_url content part object")
            }
            fn visit_map<A: de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut kind = None;
                let mut text = None;
                let mut image_url = None;
                while let Some(key) = map.next_key::<ContentPartField>()? {
                    match key {
                        ContentPartField::Type => {
                            if kind.is_some() {
                                return Err(de::Error::duplicate_field("type"));
                            }
                            kind = Some(map.next_value::<ContentPartKind>()?);
                        }
                        ContentPartField::Text => {
                            if text.is_some() {
                                return Err(de::Error::duplicate_field("text"));
                            }
                            let BorrowedStr(value) = map.next_value::<BorrowedStr<'a>>()?;
                            text = Some(value);
                        }
                        ContentPartField::ImageUrl => {
                            if image_url.is_some() {
                                return Err(de::Error::duplicate_field("image_url"));
                            }
                            let Object(value) = map.next_value::<Object<ImageUrl<'a>>>()?;
                            image_url = Some(value);
                        }
                    }
                }
                match kind.ok_or_else(|| de::Error::missing_field("type"))? {
                    ContentPartKind::Text => match (text, image_url) {
                        (Some(text), None) => Ok(ContentPart::Text { text }),
                        (None, _) => Err(de::Error::missing_field("text")),
                        (_, Some(_)) => {
                            Err(de::Error::unknown_field("image_url", &["type", "text"]))
                        }
                    },
                    ContentPartKind::ImageUrl => match (image_url, text) {
                        (Some(image_url), None) => {
                            if !valid_image_data_url(&image_url.url) {
                                return Err(de::Error::custom("unsupported image data url"));
                            }
                            Ok(ContentPart::ImageUrl { image_url })
                        }
                        (None, _) => Err(de::Error::missing_field("image_url")),
                        (_, Some(_)) => {
                            Err(de::Error::unknown_field("text", &["type", "image_url"]))
                        }
                    },
                }
            }
        }
        deserializer.deserialize_map(PartVisitor(PhantomData))
    }
}

impl<'de: 'a, 'a> Deserialize<'de> for MessageContent<'a> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ContentVisitor<'a>(PhantomData<&'a ()>);
        impl<'de: 'a, 'a> de::Visitor<'de> for ContentVisitor<'a> {
            type Value = MessageContent<'a>;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a message content string or bounded content part array")
            }
            fn visit_borrowed_str<E: de::Error>(self, value: &'de str) -> Result<Self::Value, E> {
                Ok(MessageContent::Text(Cow::Borrowed(value)))
            }
            fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
                Ok(MessageContent::Text(Cow::Owned(value.to_owned())))
            }
            fn visit_string<E: de::Error>(self, value: String) -> Result<Self::Value, E> {
                Ok(MessageContent::Text(Cow::Owned(value)))
            }
            fn visit_seq<A: de::SeqAccess<'de>>(
                self,
                mut sequence: A,
            ) -> Result<Self::Value, A::Error> {
                let mut parts = Vec::new();
                while let Some(Object(part)) = sequence.next_element::<Object<ContentPart<'a>>>()? {
                    if parts.len() == MAX_CONTENT_PARTS {
                        return Err(de::Error::custom("too many content parts"));
                    }
                    parts.push(part);
                }
                if parts.is_empty() {
                    return Err(de::Error::custom("content parts must not be empty"));
                }
                Ok(MessageContent::Parts(parts))
            }
        }
        deserializer.deserialize_any(ContentVisitor(PhantomData))
    }
}

fn valid_image_data_url(url: &str) -> bool {
    if url.len() > bonsai_prism_runtime::vision::decode::MAX_DATA_URL_BYTES {
        return false;
    }
    let Some(payload) = url
        .strip_prefix("data:image/png;base64,")
        .or_else(|| url.strip_prefix("data:image/x-portable-pixmap;base64,"))
    else {
        return false;
    };
    let bytes = payload.as_bytes();
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return false;
    }
    let pad = if bytes.ends_with(b"==") {
        2
    } else {
        usize::from(bytes.ends_with(b"="))
    };
    let Some(decoded_len) = (bytes.len() / 4)
        .checked_mul(3)
        .and_then(|n| n.checked_sub(pad))
    else {
        return false;
    };
    if decoded_len > bonsai_prism_runtime::vision::decode::MAX_ENCODED_IMAGE_BYTES {
        return false;
    }
    let prefix_len = bytes.len() - pad;
    bytes[..prefix_len]
        .iter()
        .all(|b| matches!(b, b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'+' | b'/'))
}

fn deserialize_reasoning_content<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Cow<'de, str>>, D::Error> {
    // An absent field defaults to None; a present field must be text, even when
    // empty. Deserializing Option directly would also accept an unsupported null.
    // The derived wrapper enables Serde's borrowing support for Cow while
    // retaining owned decoding for escaped JSON strings.
    #[derive(Deserialize)]
    struct Text<'a>(#[serde(borrow)] Cow<'a, str>);

    Text::deserialize(deserializer).map(|Text(text)| Some(text))
}

#[derive(Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Role {
    System,
    User,
    Assistant,
}

impl<'de> Deserialize<'de> for Role {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // A derived unit enum also accepts externally tagged objects in JSON.
        struct RoleVisitor;
        impl de::Visitor<'_> for RoleVisitor {
            type Value = Role;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a role string")
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<Role, E> {
                match value {
                    "system" => Ok(Role::System),
                    "user" => Ok(Role::User),
                    "assistant" => Ok(Role::Assistant),
                    _ => Err(E::custom("unsupported role")),
                }
            }
        }
        deserializer.deserialize_str(RoleVisitor)
    }
}

// Serde's derived structs accept positional arrays. Demand an object, then
// delegate its streaming map directly to the derive without losing duplicate keys.
struct Object<T>(T);

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Object<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ObjectVisitor<T>(PhantomData<T>);
        impl<'de, T: Deserialize<'de>> de::Visitor<'de> for ObjectVisitor<T> {
            type Value = Object<T>;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a JSON object")
            }
            fn visit_map<A: de::MapAccess<'de>>(self, map: A) -> Result<Self::Value, A::Error> {
                T::deserialize(de::value::MapAccessDeserializer::new(map)).map(Object)
            }
        }
        deserializer.deserialize_map(ObjectVisitor(PhantomData))
    }
}

fn deserialize_messages<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<Message<'de>>, D::Error> {
    struct MessagesVisitor;

    impl<'de> de::Visitor<'de> for MessagesVisitor {
        type Value = Vec<Message<'de>>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a bounded array of message objects")
        }

        fn visit_seq<A: de::SeqAccess<'de>>(
            self,
            mut sequence: A,
        ) -> Result<Self::Value, A::Error> {
            let mut messages = Vec::new();
            while let Some(Object(message)) = sequence.next_element::<Object<Message<'de>>>()? {
                // Stop on the first excess message instead of allocating and
                // parsing every remaining message in an oversized history.
                if messages.len() == MAX_MESSAGES {
                    return Err(de::Error::custom("too many messages"));
                }
                messages.push(message);
            }
            Ok(messages)
        }
    }

    deserializer.deserialize_seq(MessagesVisitor)
}

fn valid_history(messages: &[Message<'_>]) -> bool {
    if messages.is_empty()
        || messages.len() > MAX_MESSAGES
        || messages.iter().any(|message| {
            (message.reasoning_content.is_some() && message.role != Role::Assistant)
                || (matches!(message.content, MessageContent::Parts(_))
                    && message.role != Role::User)
        })
    {
        return false;
    }
    let turns = if messages[0].role == Role::System {
        &messages[1..]
    } else {
        messages
    };
    !turns.is_empty()
        && turns.len() % 2 == 1
        && turns.iter().enumerate().all(|(i, m)| {
            m.role
                == if i % 2 == 0 {
                    Role::User
                } else {
                    Role::Assistant
                }
        })
}

fn valid_history_for_vision(messages: &[Message<'_>], vision: bool) -> bool {
    if !valid_history(messages) {
        return false;
    }
    if !vision {
        return messages
            .iter()
            .all(|message| matches!(message.content, MessageContent::Text(_)));
    }
    let mut total_images = 0usize;
    for message in messages {
        total_images = total_images.saturating_add(message.content.image_count());
        if total_images > bonsai_prism_runtime::swift_session::MAX_PROMPT_IMAGES {
            return false;
        }
    }
    true
}

/// Exposes canonical bytes only to a trusted counter before context admission.
pub struct PendingChat {
    body: Vec<u8>,
    max_completion_tokens: u32,
    stream: bool,
}

impl PendingChat {
    /// The trusted counter must send these exact canonical bytes to the pinned
    /// runtime's internal `/v1/chat/completions/input_tokens` endpoint. Generation
    /// must use the same immutable model/runtime/template and explicit thinking
    /// settings. `the private HTTP adapter` implements the supplied-stream exchange;
    /// bytes/characters are never tokens. The supervisor must prevent identity drift.
    ///
    /// # Errors
    /// Propagates the trusted counter's rejection, or returns
    /// [`Rejection::TokenCountUnavailable`] for zero tokens and
    /// [`Rejection::ContextLimit`] when the checked context budget is exceeded.
    pub fn admit(
        self,
        count_rendered_prompt: impl FnOnce(&[u8]) -> Result<usize, Rejection>,
    ) -> Result<AdmittedChat, Rejection> {
        self.admit_with_context(CONTEXT_TOKENS, count_rendered_prompt)
    }

    /// Admit against the immutable context selected by the owning native session.
    ///
    /// The legacy [`Self::admit`] entry point retains its fixed qualification
    /// default. Only the verified session owner may supply this capacity.
    ///
    /// # Errors
    /// Returns the same counting/admission errors as [`Self::admit`].
    pub fn admit_with_context(
        self,
        context_tokens: usize,
        count_rendered_prompt: impl FnOnce(&[u8]) -> Result<usize, Rejection>,
    ) -> Result<AdmittedChat, Rejection> {
        let prompt = count_rendered_prompt(&self.body)?;
        if prompt == 0 {
            return Err(Rejection::TokenCountUnavailable);
        }
        let completion =
            usize::try_from(self.max_completion_tokens).map_err(|_| Rejection::ContextLimit)?;
        let total = prompt
            .checked_add(completion)
            .and_then(|n| n.checked_add(TEMPLATE_MARGIN_TOKENS));
        if total.is_none_or(|n| n > context_tokens) {
            return Err(Rejection::ContextLimit);
        }
        Ok(AdmittedChat {
            body: self.body,
            stream: self.stream,
        })
    }
}

/// Canonical body only: caller headers, credentials and JSON are never forwarded.
/// The backend adapter supplies its own fixed route/Host/key/Content-Length.
pub struct AdmittedChat {
    body: Vec<u8>,
    stream: bool,
}

impl AdmittedChat {
    /// Borrow the canonical request body admitted by the trusted token counter.
    #[must_use]
    pub fn backend_body(&self) -> &[u8] {
        &self.body
    }

    /// Retained from the canonical request, never inferred from backend headers.
    #[must_use]
    pub const fn streams_response(&self) -> bool {
        self.stream
    }
}

#[derive(Serialize)]
struct BackendChat<'a, 'b> {
    messages: &'a [Message<'b>],
    max_completion_tokens: u32,
    stream: bool,
    n: u8,
    cache_prompt: bool,
    n_cache_reuse: u8,
    ignore_eos: bool,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    min_p: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    repeat_penalty: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_budget: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dry_multiplier: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dry_base: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dry_allowed_length: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dry_penalty_last_n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_logprobs: Option<u8>,
    chat_template_kwargs: ChatTemplateKwargs,
}

#[derive(Serialize)]
struct ChatTemplateKwargs {
    enable_thinking: bool,
    reasoning_effort: &'static str,
    preserve_thinking: bool,
}

#[derive(Clone, Copy)]
struct CognitiveControlInput<'a> {
    temperature: f32,
    top_p: f32,
    top_k: u32,
    min_p: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    repeat_penalty: f32,
    enable_thinking: bool,
    reasoning_effort: &'a str,
    preserve_thinking: bool,
    reasoning_budget: Option<u32>,
    dry_multiplier: Option<f32>,
    dry_base: Option<f32>,
    dry_allowed_length: Option<u32>,
    dry_penalty_last_n: Option<u32>,
    logprobs: Option<bool>,
    top_logprobs: Option<u8>,
}

struct ValidatedCognitiveControls {
    temperature: f32,
    top_p: f32,
    top_k: u32,
    min_p: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    repeat_penalty: f32,
    enable_thinking: bool,
    reasoning_effort: &'static str,
    reasoning_budget: Option<u32>,
    dry_multiplier: Option<f32>,
    dry_base: Option<f32>,
    dry_allowed_length: Option<u32>,
    dry_penalty_last_n: Option<u32>,
    logprobs: Option<bool>,
    top_logprobs: Option<u8>,
}

impl ValidatedCognitiveControls {
    const fn to_backend_chat<'a, 'b>(
        &self,
        messages: &'a [Message<'b>],
        max_completion_tokens: u32,
        stream: bool,
    ) -> BackendChat<'a, 'b> {
        BackendChat {
            messages,
            max_completion_tokens,
            stream,
            n: 1,
            cache_prompt: false,
            n_cache_reuse: 0,
            ignore_eos: false,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            repeat_penalty: self.repeat_penalty,
            reasoning_budget: self.reasoning_budget,
            dry_multiplier: self.dry_multiplier,
            dry_base: self.dry_base,
            dry_allowed_length: self.dry_allowed_length,
            dry_penalty_last_n: self.dry_penalty_last_n,
            logprobs: self.logprobs,
            top_logprobs: self.top_logprobs,
            chat_template_kwargs: ChatTemplateKwargs {
                enable_thinking: self.enable_thinking,
                reasoning_effort: self.reasoning_effort,
                preserve_thinking: true,
            },
        }
    }
}

fn resolve_client_reasoning_effort<'a>(
    top_level: Option<&'a str>,
    kwarg: Option<&'a str>,
) -> Result<&'a str, Rejection> {
    match (top_level, kwarg) {
        (Some(a), Some(b)) if a != b => Err(Rejection::ChatSchema),
        (Some(a), _) | (None, Some(a)) => Ok(a),
        (None, None) => Ok("xhigh"),
    }
}

const fn is_non_neg_zero_finite(value: f32) -> bool {
    value.is_finite() && !(value == 0.0 && value.is_sign_negative())
}

fn validate_cognitive_controls(
    input: CognitiveControlInput<'_>,
) -> Option<ValidatedCognitiveControls> {
    let effort = match input.reasoning_effort {
        "low" => "low",
        "medium" => "medium",
        "xhigh" => "xhigh",
        _ => return None,
    };
    let dry_active = input.dry_multiplier.is_some_and(|mult| mult > 0.0);
    if !(input.preserve_thinking
        && is_non_neg_zero_finite(input.temperature)
        && 0.0 < input.temperature
        && input.temperature <= 2.0
        && is_non_neg_zero_finite(input.top_p)
        && 0.0 < input.top_p
        && input.top_p <= 1.0
        && input.top_k <= 248_320
        && is_non_neg_zero_finite(input.min_p)
        && (0.0..=1.0).contains(&input.min_p)
        && is_non_neg_zero_finite(input.presence_penalty)
        && (-2.0..=2.0).contains(&input.presence_penalty)
        && is_non_neg_zero_finite(input.frequency_penalty)
        && (-2.0..=2.0).contains(&input.frequency_penalty)
        && is_non_neg_zero_finite(input.repeat_penalty)
        && 0.0 < input.repeat_penalty
        && input.repeat_penalty <= 2.0
        && input
            .reasoning_budget
            .is_none_or(|budget| (1..=262_144).contains(&budget))
        && input
            .dry_multiplier
            .is_none_or(|mult| is_non_neg_zero_finite(mult) && (0.0..=4.0).contains(&mult))
        && input
            .dry_base
            .is_none_or(|base| base.is_finite() && (1.0..=4.0).contains(&base))
        && input.dry_allowed_length.is_none_or(|len| {
            if dry_active {
                (1..=64).contains(&len)
            } else {
                len <= 64
            }
        })
        && input.dry_penalty_last_n.is_none_or(|n| n <= 8192)
        && !(input.top_logprobs.is_some() && input.logprobs != Some(true))
        && input
            .top_logprobs
            .is_none_or(|count| (1..=20).contains(&count)))
    {
        return None;
    }
    Some(ValidatedCognitiveControls {
        temperature: input.temperature,
        top_p: input.top_p,
        top_k: input.top_k,
        min_p: input.min_p,
        presence_penalty: input.presence_penalty,
        frequency_penalty: input.frequency_penalty,
        repeat_penalty: input.repeat_penalty,
        enable_thinking: input.enable_thinking,
        reasoning_effort: effort,
        reasoning_budget: input.reasoning_budget,
        dry_multiplier: input.dry_multiplier,
        dry_base: input.dry_base,
        dry_allowed_length: input.dry_allowed_length,
        dry_penalty_last_n: input.dry_penalty_last_n,
        logprobs: input.logprobs,
        top_logprobs: input.top_logprobs,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        ClientChat, ContentPart, Cow, MAX_BODY_BYTES, MAX_MULTIMODAL_BODY_BYTES, Message,
        MessageContent, Object, Rejection, RequestPolicy, Role, valid_history,
        valid_history_for_vision,
    };

    const TEST_KEY: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    #[test]
    fn every_short_role_history_matches_the_accepted_language() {
        // The complete accepted corpus through six messages, including the
        // optional system preamble. Enumerating other words catches misplaced
        // system messages, repeated speakers and assistant-prefill acceptance.
        let accepted = ["u", "su", "uau", "suau", "uauau", "suauau"];
        for length in 0..=6 {
            for mut word in 0..3_usize.pow(length) {
                let mut spelling = String::new();
                let mut messages = Vec::new();
                for _ in 0..length {
                    let (letter, role) = match word % 3 {
                        0 => ('s', Role::System),
                        1 => ('u', Role::User),
                        _ => ('a', Role::Assistant),
                    };
                    word /= 3;
                    spelling.push(letter);
                    messages.push(Message {
                        role,
                        content: MessageContent::Text(Cow::Borrowed("")),
                        reasoning_content: None,
                    });
                }
                assert_eq!(
                    valid_history(&messages),
                    accepted.contains(&spelling.as_str()),
                    "history {spelling:?}"
                );
            }
        }
    }

    #[test]
    fn message_content_borrows_plain_text_and_decodes_escapes() -> Result<(), serde_json::Error> {
        let Object(chat): Object<ClientChat<'_>> = serde_json::from_slice(
            br#"{"messages":[{"role":"user","content":"plain text"},{"role":"assistant","content":"line\n\u96ea"}]}"#,
        )?;
        assert!(matches!(
            chat.messages[0].content,
            MessageContent::Text(Cow::Borrowed("plain text"))
        ));
        assert!(matches!(
            chat.messages[1].content,
            MessageContent::Text(Cow::Owned(_))
        ));
        if let MessageContent::Text(ref text) = chat.messages[1].content {
            assert_eq!(text, "line\n雪");
        }
        Ok(())
    }

    #[test]
    fn message_reasoning_borrows_plain_text_and_decodes_escapes() -> Result<(), serde_json::Error> {
        let Object(chat): Object<ClientChat<'_>> = serde_json::from_slice(
            br#"{"messages":[{"role":"assistant","content":"","reasoning_content":"plain text"},{"role":"assistant","content":"","reasoning_content":"line\n\u96ea"}]}"#,
        )?;
        assert!(matches!(
            chat.messages[0].reasoning_content,
            Some(Cow::Borrowed("plain text"))
        ));
        assert!(matches!(
            chat.messages[1].reasoning_content,
            Some(Cow::Owned(_))
        ));
        assert_eq!(
            chat.messages[1].reasoning_content.as_deref(),
            Some("line\n雪")
        );
        Ok(())
    }

    #[test]
    fn multimodal_head_and_schema_gate_vision_bounds_and_data_urls() -> Result<(), Rejection> {
        let text_policy = RequestPolicy::new(8080, TEST_KEY)?;
        let vision_policy = RequestPolicy::new(8080, TEST_KEY)?.with_vision(true);
        let oversize = (MAX_BODY_BYTES + 1).to_string();
        let headers = [
            httparse::Header {
                name: "authorization",
                value: format!("Bearer {TEST_KEY}").leak().as_bytes(),
            },
            httparse::Header {
                name: "host",
                value: b"127.0.0.1:8080",
            },
            httparse::Header {
                name: "content-type",
                value: b"application/json",
            },
            httparse::Header {
                name: "content-length",
                value: oversize.as_bytes(),
            },
        ];
        assert_eq!(
            text_policy
                .check_head("POST", "/v1/chat/completions", 1, &headers)
                .err(),
            Some(Rejection::BodySize)
        );
        assert!(
            vision_policy
                .check_head("POST", "/v1/chat/completions", 1, &headers)
                .is_ok()
        );
        let too_big = (MAX_MULTIMODAL_BODY_BYTES + 1).to_string();
        let huge_headers = [
            headers[0],
            headers[1],
            headers[2],
            httparse::Header {
                name: "content-length",
                value: too_big.as_bytes(),
            },
        ];
        assert_eq!(
            vision_policy
                .check_head("POST", "/v1/chat/completions", 1, &huge_headers)
                .err(),
            Some(Rejection::BodySize)
        );

        let mm_body = br#"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}},{"type":"text","text":"Describe."}]}]}"#;
        assert_eq!(
            super::AuthorizedHead {
                length: mm_body.len(),
                vision: false,
            }
            .prepare(mm_body)
            .err(),
            Some(Rejection::ChatSchema)
        );
        let pending = super::AuthorizedHead {
            length: mm_body.len(),
            vision: true,
        }
        .prepare(mm_body)?;
        let parsed: serde_json::Value =
            serde_json::from_slice(&pending.body).map_err(|_| Rejection::ChatSchema)?;
        assert_eq!(parsed["messages"][0]["content"][0]["type"], "image_url");

        for rejected in [
            br#"{"messages":[{"role":"user","content":[]}]}"#.as_slice(),
            br#"{"messages":[{"role":"assistant","content":[{"type":"text","text":"nope"}]}]}"#,
            br#"{"messages":[{"role":"system","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]},{"role":"user","content":"hi"}]}"#,
            br#"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}]}]}"#,
            br#"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"file:///c:/a.png"}}]}]}"#,
            br#"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAA"}}]}]}"#,
            br#"{"messages":[{"role":"user","content":[{"type":"text","text":"hi","image_url":{"url":"data:image/png;base64,AAAA"}}]}]}"#,
        ] {
            assert!(
                super::AuthorizedHead {
                    length: rejected.len(),
                    vision: true,
                }
                .prepare(rejected)
                .is_err()
            );
        }
        assert!(matches!(
            serde_json::from_slice::<Object<ContentPart<'_>>>(br#"{"type":"text","text":"x"}"#),
            Ok(Object(ContentPart::Text { .. }))
        ));
        assert!(!valid_history_for_vision(
            &[Message {
                role: Role::User,
                content: MessageContent::Parts(vec![ContentPart::Text {
                    text: Cow::Borrowed("hi")
                }]),
                reasoning_content: None,
            }],
            false
        ));
        Ok(())
    }
}
