//! Checked synchronous text and raw-ID requests over a token-worker transport.
//!
//! [`TokenClient::connect`] verifies readiness against a caller-pinned artifact,
//! vocabulary and complete worker policy before sending any request. The client
//! owns neither a process nor a socket lifecycle: callers supply `BufRead` and
//! `Write` values, including mutable references to their own streams. Operations
//! are synchronous and have no built-in cancellation or deadlines; the caller
//! must arrange those properties on the underlying transport.
//!
//! Request and response frames have separate explicit byte bounds, excluding LF
//! and including CR in CRLF. Each bound is at most 1 MiB. A valid worker response
//! can exceed its request bound, so callers must budget responses independently.
//! JSON decoding and token storage are bounded by the response frame; token
//! counts, encoded lengths and vocabulary membership are checked before return.
//! Protocol and transport failures permanently disable the client. Rejected
//! input and local request-size rejection leave it reusable. Errors contain only
//! static descriptions, never prompt text, response content or I/O diagnostics.

use std::fmt;
use std::io::{self, BufRead, Write};

use serde::{Deserialize, Serialize};

use crate::special_input::SpecialInputPolicy;
use crate::token_worker::{MAX_WORKER_FRAME_BYTES, WorkerConfig};
use crate::tokenization::{SpecialTokenInsertion, TokenizationResult};
use crate::vocabulary::MAX_VOCABULARY_TOKENS;

/// Pinned readiness identity, complete policy and local response-frame bound.
#[derive(Clone, Copy, Debug)]
pub struct ExpectedWorker<'a> {
    /// Expected lowercase, 64-character artifact SHA-256, supplied independently.
    pub model_sha256: &'a str,
    /// Expected nonzero vocabulary size, at most the tokenizer's vocabulary cap.
    pub vocab_size: usize,
    /// Exact worker request bound, text policy and context/output accounting.
    pub config: WorkerConfig,
    /// Local response byte bound before LF, from 1 through 1 MiB.
    pub max_response_frame_bytes: usize,
}

/// Static, privacy-preserving client failure classifications.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClientError {
    /// The caller supplied invalid identity, policy or resource bounds.
    InvalidConfiguration,
    /// Bounded client transport buffers could not be allocated.
    Allocation,
    /// Readiness did not exactly match the pinned identity and policy.
    ReadinessMismatch,
    /// The peer sent malformed, oversized, incomplete or inconsistent data.
    InvalidResponse,
    /// A transport read, write or flush failed; underlying details are omitted.
    Transport,
    /// The serialized request exceeds the worker bound; nothing was sent.
    RequestTooLarge,
    /// Local raw-ID bounds or the worker rejected input; transport remains usable.
    RejectedInput,
    /// A prior fatal error or exhausted correlation space prevents further use.
    Unavailable,
}

impl fmt::Display for ClientError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidConfiguration => "invalid token client configuration",
            Self::Allocation => "token client allocation failed",
            Self::ReadinessMismatch => "token worker readiness mismatch",
            Self::InvalidResponse => "invalid token worker response",
            Self::Transport => "token client transport failed",
            Self::RequestTooLarge => "token client request exceeds byte limit",
            Self::RejectedInput => "token worker rejected input",
            Self::Unavailable => "token client is unavailable",
        })
    }
}

impl std::error::Error for ClientError {}

// Intentionally no Debug: these buffers and streams can contain private text.
/// Serial checked requests over one verified persistent worker connection.
pub struct TokenClient<R, W> {
    input: R,
    output: W,
    config: WorkerConfig,
    vocab_size: usize,
    prompt_limit: usize,
    response_limit: usize,
    healthy: bool,
    next_id: Option<u64>,
    request: Vec<u8>,
    response: Vec<u8>,
}

impl<R: BufRead, W: Write> TokenClient<R, W> {
    /// Validate configuration and read exactly one bounded readiness record.
    ///
    /// No request bytes are written during connection. Identity must come from
    /// the caller's trusted expectation, rather than being learned from the peer.
    ///
    /// # Errors
    /// Rejects invalid configuration, buffer allocation, transport failures,
    /// malformed readiness and any identity or fixed-policy mismatch.
    pub fn connect(input: R, output: W, expected: ExpectedWorker<'_>) -> Result<Self, ClientError> {
        let config = expected.config;
        let prompt_limit = config
            .text_policy
            .budget
            .prompt_token_limit()
            .map_err(|_| ClientError::InvalidConfiguration)?;
        if config.validate().is_err()
            || expected.model_sha256.len() != 64
            || !expected
                .model_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            || !(1..=MAX_VOCABULARY_TOKENS).contains(&expected.vocab_size)
            || !(1..=MAX_WORKER_FRAME_BYTES).contains(&expected.max_response_frame_bytes)
        {
            return Err(ClientError::InvalidConfiguration);
        }
        let mut client = Self {
            input,
            output,
            config,
            vocab_size: expected.vocab_size,
            prompt_limit,
            response_limit: expected.max_response_frame_bytes,
            healthy: false,
            next_id: Some(0),
            request: Vec::new(),
            response: Vec::new(),
        };
        client
            .request
            .try_reserve_exact(config.max_frame_bytes)
            .map_err(|_| ClientError::Allocation)?;
        client
            .response
            .try_reserve_exact(client.response_limit)
            .map_err(|_| ClientError::Allocation)?;
        match client.read_response()? {
            Response::Ready(ready)
                if ready.protocol == "bonsai-token-worker-v1"
                    && ready.model_sha256 == expected.model_sha256
                    && ready.vocab_size == expected.vocab_size
                    && ready.context == config.text_policy.budget.context
                    && ready.reserved_output_tokens
                        == config.text_policy.budget.reserved_output_tokens
                    && ready.max_frame_bytes == config.max_frame_bytes
                    && ready.special_input
                        == match config.text_policy.special_input {
                            SpecialInputPolicy::Literal => "literal",
                            SpecialInputPolicy::Parse => "parse",
                        }
                    && ready.insert_special
                        == match config.text_policy.insert_special {
                            SpecialTokenInsertion::None => "none",
                            SpecialTokenInsertion::Model => "model",
                        } =>
            {
                client.healthy = true;
                Ok(client)
            }
            _ => Err(ClientError::ReadinessMismatch),
        }
    }

    /// Whether another request can be attempted without replacing the client.
    #[must_use]
    pub const fn is_usable(&self) -> bool {
        self.healthy && self.next_id.is_some()
    }

    /// Count complete text using the exact policy verified at connection.
    ///
    /// # Errors
    /// Returns a static classification for local bounds, worker rejection and
    /// fatal protocol/transport failures. Only local request-size rejection and
    /// worker input rejection permit another attempt on this connection.
    pub fn count(&mut self, text: &str) -> Result<usize, ClientError> {
        match self.exchange(Operation::Count, RequestInput::Text { text })? {
            Response::Count {
                prompt_token_count, ..
            } => Ok(prompt_token_count),
            _ => Err(ClientError::InvalidResponse),
        }
    }

    /// Encode complete text and return checked token IDs and fixed accounting.
    ///
    /// # Errors
    /// Has the same failure and connection-reuse contract as [`Self::count`].
    pub fn encode(&mut self, text: &str) -> Result<TokenizationResult, ClientError> {
        match self.exchange(Operation::Encode, RequestInput::Text { text })? {
            Response::Encoded { token_ids, .. } => Ok(TokenizationResult {
                token_ids,
                context: self.config.text_policy.budget.context,
                reserved_output_tokens: self.config.text_policy.budget.reserved_output_tokens,
            }),
            _ => Err(ClientError::InvalidResponse),
        }
    }

    /// Count checked raw IDs without text parsing or special-token insertion.
    ///
    /// IDs must fit the pinned vocabulary and prompt allowance. Empty input is
    /// valid, and repeated IDs count separately. The response must report the
    /// exact supplied length, including when model insertion is enabled.
    ///
    /// # Errors
    /// Local invalid IDs or context overflow return [`ClientError::RejectedInput`]
    /// before writing or consuming a correlation ID, leaving the client reusable.
    /// Other errors follow the connection-reuse contract of [`Self::count`].
    pub fn count_tokens(&mut self, token_ids: &[u32]) -> Result<usize, ClientError> {
        match self.exchange(Operation::CountTokens, RequestInput::Tokens { token_ids })? {
            Response::Count {
                prompt_token_count, ..
            } => Ok(prompt_token_count),
            _ => Err(ClientError::InvalidResponse),
        }
    }

    /// Return checked raw IDs and fixed accounting without changing their order.
    ///
    /// No text parsing or special-token insertion applies. The response must
    /// preserve the exact supplied sequence, including duplicates and empty input.
    ///
    /// # Errors
    /// Has the same failure and connection-reuse contract as [`Self::count_tokens`].
    pub fn encode_tokens(&mut self, token_ids: &[u32]) -> Result<TokenizationResult, ClientError> {
        match self.exchange(Operation::EncodeTokens, RequestInput::Tokens { token_ids })? {
            Response::Encoded { token_ids, .. } => Ok(TokenizationResult {
                token_ids,
                context: self.config.text_policy.budget.context,
                reserved_output_tokens: self.config.text_policy.budget.reserved_output_tokens,
            }),
            _ => Err(ClientError::InvalidResponse),
        }
    }

    /// Recover the supplied streams for caller-owned shutdown or disposal.
    ///
    /// A fatal error may leave partial frames in either direction. Extracting
    /// streams does not make that connection safe for another protocol client.
    #[must_use]
    pub fn into_parts(self) -> (R, W) {
        (self.input, self.output)
    }

    fn exchange(
        &mut self,
        operation: Operation,
        input: RequestInput<'_>,
    ) -> Result<Response, ClientError> {
        if !self.is_usable() {
            return Err(ClientError::Unavailable);
        }
        let raw_tokens = input.raw_tokens();
        if raw_tokens.is_some_and(|tokens| {
            tokens.len() > self.prompt_limit
                || tokens.iter().any(|&token| {
                    usize::try_from(token).map_or(true, |token| token >= self.vocab_size)
                })
        }) {
            return Err(ClientError::RejectedInput);
        }
        let id = self.next_id.ok_or(ClientError::Unavailable)?;
        self.request.clear();
        serde_json::to_writer(
            BoundedWriter {
                frame: &mut self.request,
                limit: self.config.max_frame_bytes,
            },
            &Request {
                id,
                op: operation,
                input,
            },
        )
        .map_err(|_| ClientError::RequestTooLarge)?;

        // From the first attempted write, every exit is fatal unless a complete
        // checked response (including correlated rejected_input) restores health.
        self.healthy = false;
        self.next_id = id.checked_add(1);
        self.output
            .write_all(&self.request)
            .and_then(|()| self.output.write_all(b"\n"))
            .and_then(|()| self.output.flush())
            .map_err(|_| ClientError::Transport)?;
        let response = self.read_response()?;
        match &response {
            Response::Count {
                id: response_id,
                prompt_token_count,
            } if matches!(operation, Operation::Count | Operation::CountTokens)
                && *response_id == id
                && *prompt_token_count <= self.prompt_limit
                && raw_tokens.is_none_or(|tokens| *prompt_token_count == tokens.len()) => {}
            Response::Encoded {
                id: response_id,
                prompt_token_count,
                token_ids,
            } if matches!(operation, Operation::Encode | Operation::EncodeTokens)
                && *response_id == id
                && *prompt_token_count <= self.prompt_limit
                && *prompt_token_count == token_ids.len()
                && raw_tokens.is_none_or(|tokens| token_ids == tokens)
                && token_ids.iter().all(|&token| {
                    usize::try_from(token).is_ok_and(|token| token < self.vocab_size)
                }) => {}
            Response::Error {
                id: Some(response_id),
                code,
            } if *response_id == id && code == "rejected_input" => {
                self.healthy = true;
                return Err(ClientError::RejectedInput);
            }
            _ => return Err(ClientError::InvalidResponse),
        }
        self.healthy = true;
        Ok(response)
    }

    fn read_response(&mut self) -> Result<Response, ClientError> {
        self.response.clear();
        loop {
            let available = self.input.fill_buf().map_err(|_| ClientError::Transport)?;
            if available.is_empty() {
                return Err(ClientError::InvalidResponse);
            }
            let newline = available.iter().position(|&byte| byte == b'\n');
            let count = newline.unwrap_or(available.len());
            if count > self.response_limit - self.response.len() {
                return Err(ClientError::InvalidResponse);
            }
            self.response.extend_from_slice(&available[..count]);
            self.input.consume(count + usize::from(newline.is_some()));
            if newline.is_some() {
                break;
            }
        }
        if self.response.trim_ascii_start().first() != Some(&b'{') {
            return Err(ClientError::InvalidResponse);
        }
        serde_json::from_slice(&self.response).map_err(|_| ClientError::InvalidResponse)
    }
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum Operation {
    Count,
    Encode,
    CountTokens,
    EncodeTokens,
}

#[derive(Serialize)]
struct Request<'a> {
    id: u64,
    op: Operation,
    #[serde(flatten)]
    input: RequestInput<'a>,
}

#[derive(Clone, Copy, Serialize)]
#[serde(untagged)]
enum RequestInput<'a> {
    Text { text: &'a str },
    Tokens { token_ids: &'a [u32] },
}

impl<'a> RequestInput<'a> {
    const fn raw_tokens(self) -> Option<&'a [u32]> {
        match self {
            Self::Text { .. } => None,
            Self::Tokens { token_ids } => Some(token_ids),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Ready {
    protocol: String,
    model_sha256: String,
    vocab_size: usize,
    context: usize,
    reserved_output_tokens: usize,
    max_frame_bytes: usize,
    special_input: String,
    insert_special: String,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum Response {
    Ready(Ready),
    Count {
        id: u64,
        prompt_token_count: usize,
    },
    Encoded {
        id: u64,
        prompt_token_count: usize,
        token_ids: Vec<u32>,
    },
    Error {
        id: Option<u64>,
        code: String,
    },
}

struct BoundedWriter<'a> {
    frame: &'a mut Vec<u8>,
    limit: usize,
}

impl Write for BoundedWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.limit - self.frame.len() {
            return Err(io::Error::other("token request exceeds byte limit"));
        }
        self.frame.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
