//! Bounded private HTTP service for the independently released Prism runtime.
//!
//! The owning launcher retains process, Job, artifact and memory protections.
#![forbid(unsafe_code)]

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
mod supervisor;
mod wire;

use http::{HeaderMap, HeaderValue, Method, Version, request::Parts, uri::PathAndQuery};
use serde::{Deserialize, Deserializer, Serialize, de};
use std::{borrow::Cow, fmt, marker::PhantomData};
use subtle::ConstantTimeEq;

/// Largest accepted client request body in bytes.
pub const MAX_BODY_BYTES: usize = 1024 * 1024;
/// Largest accepted HTTP header block in bytes.
pub const MAX_HEADER_BYTES: usize = 16 * 1024;
/// Largest accepted number of HTTP header fields, including duplicates.
pub const MAX_HEADERS: usize = 64;
/// Largest accepted number of messages in one chat history.
pub const MAX_MESSAGES: usize = 128;
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
        })
    }

    /// Check the parsed head before the transport accepts a bounded body.
    /// Only HTTP/1.1 origin-form POST /v1/chat/completions is supported.
    /// OPTIONS, readiness, UI, tokenization and administrative routes are closed.
    ///
    /// # Errors
    /// Returns a value-free [`Rejection`] for invalid credentials, headers, route,
    /// content type, framing or body length.
    pub fn check_head(&self, head: &Parts) -> Result<AuthorizedHead, Rejection> {
        let headers = &head.headers;
        if headers.len() > MAX_HEADERS {
            return Err(Rejection::Headers);
        }
        let bytes = headers.iter().try_fold(0usize, |total, (name, value)| {
            total.checked_add(name.as_str().len() + value.as_bytes().len() + 4)
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
        if head.method != Method::POST
            || head.version != Version::HTTP_11
            || head.uri.scheme().is_some()
            || head.uri.authority().is_some()
            || head.uri.path_and_query().map(PathAndQuery::as_str) != Some("/v1/chat/completions")
        {
            return Err(Rejection::Route);
        }
        for name in [
            "transfer-encoding",
            "content-encoding",
            "trailer",
            "expect",
            "upgrade",
        ] {
            if headers.contains_key(name) {
                return Err(Rejection::Framing);
            }
        }
        if headers
            .keys()
            .any(|name| name.as_str() == "forwarded" || name.as_str().starts_with("x-forwarded-"))
        {
            return Err(Rejection::Headers);
        }
        if let Some(connection) = singleton(headers, "connection")?
            && !connection.eq_ignore_ascii_case(b"close")
            && !connection.eq_ignore_ascii_case(b"keep-alive")
        {
            return Err(Rejection::Framing);
        }
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
        if length == 0 || length > MAX_BODY_BYTES {
            return Err(Rejection::BodySize);
        }
        Ok(AuthorizedHead { length })
    }
}

fn singleton<'a>(headers: &'a HeaderMap, name: &str) -> Result<Option<&'a [u8]>, Rejection> {
    let mut values = headers.get_all(name).iter();
    let first = values.next().map(HeaderValue::as_bytes);
    if values.next().is_some() {
        return Err(Rejection::Headers);
    }
    Ok(first)
}

/// Constructed only by successful head validation; does not retain the key.
pub struct AuthorizedHead {
    length: usize,
}

impl AuthorizedHead {
    /// Validate a bounded JSON chat body and construct canonical backend bytes.
    ///
    /// # Errors
    /// Returns [`Rejection`] for a mismatched length, invalid schema, unsupported
    /// message history, completion limit or canonical serialization failure.
    pub fn prepare(self, body: &[u8]) -> Result<PendingChat, Rejection> {
        if body.len() != self.length || body.len() > MAX_BODY_BYTES {
            return Err(Rejection::BodySize);
        }
        // Deserialize directly: maps/Value would silently erase duplicate keys.
        // Parser error strings can include private values; never propagate them.
        let Object(chat): Object<ClientChat<'_>> =
            serde_json::from_slice(body).map_err(|_| Rejection::ChatSchema)?;
        if !valid_history(&chat.messages) {
            return Err(Rejection::ChatSchema);
        }
        if !(1..=MAX_COMPLETION_TOKENS).contains(&chat.max_completion_tokens) {
            return Err(Rejection::CompletionLimit);
        }
        let max_completion_tokens = chat.max_completion_tokens;
        let body = serde_json::to_vec(&BackendChat {
            messages: &chat.messages,
            max_completion_tokens,
            stream: chat.stream,
            n: 1,
            cache_prompt: false,
            n_cache_reuse: 0,
            ignore_eos: false,
            temperature: 1.0,
            top_p: 0.95,
            top_k: 20,
            min_p: 0.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repeat_penalty: 1.0,
            chat_template_kwargs: ChatTemplateKwargs {
                enable_thinking: true,
                reasoning_effort: "xhigh",
                preserve_thinking: true,
            },
        })
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
struct ClientChat<'a> {
    #[serde(borrow, deserialize_with = "deserialize_messages")]
    messages: Vec<Message<'a>>,
    #[serde(default = "default_cap")]
    max_completion_tokens: u32,
    #[serde(default = "default_stream")]
    stream: bool,
}

const fn default_cap() -> u32 {
    MAX_COMPLETION_TOKENS
}
const fn default_stream() -> bool {
    true
}

/// Text only; strings resembling URLs remain plain text and are never fetched.
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Message<'a> {
    role: Role,
    // JSON text without escapes can be borrowed until canonical serialization;
    // escaped text still uses Serde's owned decoding without changing its bytes.
    #[serde(borrow)]
    content: Cow<'a, str>,
    #[serde(
        default,
        borrow,
        deserialize_with = "deserialize_reasoning_content",
        skip_serializing_if = "Option::is_none"
    )]
    reasoning_content: Option<Cow<'a, str>>,
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
        || messages
            .iter()
            .any(|message| message.reasoning_content.is_some() && message.role != Role::Assistant)
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
        let prompt = count_rendered_prompt(&self.body)?;
        if prompt == 0 {
            return Err(Rejection::TokenCountUnavailable);
        }
        let completion =
            usize::try_from(self.max_completion_tokens).map_err(|_| Rejection::ContextLimit)?;
        let total = prompt
            .checked_add(completion)
            .and_then(|n| n.checked_add(TEMPLATE_MARGIN_TOKENS));
        if total.is_none_or(|n| n > CONTEXT_TOKENS) {
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
    top_k: u8,
    min_p: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    repeat_penalty: f32,
    chat_template_kwargs: ChatTemplateKwargs,
}

#[derive(Serialize)]
struct ChatTemplateKwargs {
    enable_thinking: bool,
    reasoning_effort: &'static str,
    preserve_thinking: bool,
}

#[cfg(test)]
mod tests {
    use super::{ClientChat, Cow, Message, Object, Role, valid_history};

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
                        content: Cow::Borrowed(""),
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
            Cow::Borrowed("plain text")
        ));
        assert!(matches!(chat.messages[1].content, Cow::Owned(_)));
        assert_eq!(chat.messages[1].content, "line\n雪");
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
}
