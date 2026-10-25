//! Bounded synchronous client for one pinned Swift worker conversation.
//!
//! Callers supply and own both transports and their process/Job lifetime. Reads
//! and writes may block: transport timeouts or shutdown must make them return.
//! Cancellation/deadlines are checked around those operations and delivery.
//! Deltas remain provisional until a checked terminal completion is returned.
//! Abandoned completion, callback failure and protocol/transport errors poison
//! this client; callers must shut down the owned worker and replace its streams.

use std::{
    fmt,
    io::{self, BufRead, Write},
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    model::swift,
    runtime::{Cancellation, Completion, Finish},
    swift_session::MAX_OUTPUT_BYTES,
};

/// Maximum complete JSONL frame, including its LF, accepted by the Swift worker.
pub const MAX_FRAME_BYTES: usize = 1024 * 1024;
/// Default response bound, sufficient for the worker's 16 KiB UTF-8 deltas.
pub const DEFAULT_RESPONSE_FRAME_BYTES: usize = 128 * 1024;
/// Largest unescaped UTF-8 text piece emitted in one worker delta record.
pub const MAX_DELTA_BYTES: usize = 16 * 1024;

/// Requested CPU dispatch policy reported by worker readiness.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SwiftWorkerBackend {
    /// Eligible baseline NEON operations with scalar fallback.
    Auto,
    /// Scalar reference operations only.
    Scalar,
}

impl SwiftWorkerBackend {
    const fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Scalar => "scalar",
        }
    }
}

/// Caller-pinned effective settings and local response allocation bound.
#[derive(Clone, Copy, Debug)]
pub struct ExpectedSwiftWorker {
    /// Exact explicit non-sentinel sampler seed chosen by the process owner.
    pub seed: u32,
    /// Exact requested scalar/automatic CPU dispatch policy.
    pub backend: SwiftWorkerBackend,
    /// Maximum JSONL response bytes including LF, from 1 through 1 MiB.
    pub max_response_frame_bytes: usize,
}

/// Static failures that never expose prompt, generated text or I/O diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClientError {
    /// Invalid seed, resource bound, output allowance or deadline duration.
    InvalidConfiguration,
    /// Bounded transport storage could not be allocated.
    Allocation,
    /// Readiness did not match pinned model/runtime identity and effective policy.
    ReadinessMismatch,
    /// Malformed, oversized, duplicate-field or inconsistent response.
    InvalidResponse,
    /// Underlying read, write or flush failed.
    Transport,
    /// Serialized request does not fit; no request bytes were sent.
    RequestTooLarge,
    /// Worker rejected the prompt before generation; transport remains reusable.
    RejectedInput,
    /// Generation failed; abandon provisional output and inspect reset state.
    GenerationFailed,
    /// Generation admission is closed until a successful reset.
    ResetRequired,
    /// Worker reset failed; another explicit reset may be attempted.
    ResetFailed,
    /// Caller cancellation was observed; sent completion requires worker shutdown.
    Cancelled,
    /// Absolute deadline expired; sent completion requires worker shutdown.
    Deadline,
    /// Delta delivery failed; worker shutdown is required.
    DeliveryFailed,
    /// Prior fatal failure or correlation exhaustion requires a new transport.
    Unavailable,
}

impl fmt::Display for ClientError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        output.write_str(match self {
            Self::InvalidConfiguration => "invalid Swift client configuration",
            Self::Allocation => "Swift client allocation failed",
            Self::ReadinessMismatch => "Swift worker readiness mismatch",
            Self::InvalidResponse => "invalid Swift worker response",
            Self::Transport => "Swift client transport failed",
            Self::RequestTooLarge => "Swift request exceeds byte limit",
            Self::RejectedInput => "Swift worker rejected input",
            Self::GenerationFailed => "Swift generation failed",
            Self::ResetRequired => "Swift worker requires reset",
            Self::ResetFailed => "Swift worker reset failed",
            Self::Cancelled => "Swift request cancelled",
            Self::Deadline => "Swift request expired",
            Self::DeliveryFailed => "Swift output delivery failed",
            Self::Unavailable => "Swift client is unavailable",
        })
    }
}
impl std::error::Error for ClientError {}

/// One serial, checked pinned-model worker connection with bounded buffers.
///
/// Readiness is a protocol claim checked against independent caller settings;
/// it does not attest executable bytes, process identity or Job ownership.
/// The owner must retain those existing launch/admission protections.
pub struct SwiftClient<R, W> {
    input: R,
    output: W,
    response_limit: usize,
    healthy: bool,
    reset_required: bool,
    next_id: Option<u64>,
    request: Vec<u8>,
    response: Vec<u8>,
}

impl<R: BufRead, W: Write> SwiftClient<R, W> {
    /// Read and verify exactly one readiness frame without sending any request.
    ///
    /// # Errors
    /// Rejects invalid bounds/seed, malformed or mismatched readiness and I/O.
    pub fn connect(
        input: R,
        output: W,
        expected: ExpectedSwiftWorker,
    ) -> Result<Self, ClientError> {
        if expected.seed == u32::MAX
            || !(1..=MAX_FRAME_BYTES).contains(&expected.max_response_frame_bytes)
        {
            return Err(ClientError::InvalidConfiguration);
        }
        let mut client = Self {
            input,
            output,
            response_limit: expected.max_response_frame_bytes,
            healthy: false,
            reset_required: false,
            next_id: Some(0),
            request: Vec::new(),
            response: Vec::new(),
        };
        client
            .request
            .try_reserve_exact(MAX_FRAME_BYTES)
            .map_err(|_| ClientError::Allocation)?;
        client
            .response
            .try_reserve_exact(client.response_limit)
            .map_err(|_| ClientError::Allocation)?;
        match client.read_response()? {
            Response::Ready(ready) if ready.matches(expected) => {
                client.healthy = true;
                Ok(client)
            }
            _ => Err(ClientError::ReadinessMismatch),
        }
    }

    /// Whether prompt counting or completion can start on this connection.
    #[must_use]
    pub const fn is_usable(&self) -> bool {
        self.healthy && self.next_id.is_some() && !self.reset_required
    }

    /// Whether a checked worker failure closed admission pending explicit reset.
    #[must_use]
    pub const fn requires_reset(&self) -> bool {
        self.reset_required
    }

    /// Count the exact rendered chat using the complete supplied output allowance.
    ///
    /// # Errors
    /// Input rejection and local request-size rejection leave the transport
    /// reusable. Protocol/I/O failures permanently disable it.
    pub fn count(&mut self, messages: &Value, max_tokens: usize) -> Result<usize, ClientError> {
        self.admit(max_tokens)?;
        crate::swift_chat::validate_messages(messages).map_err(|_| ClientError::RejectedInput)?;
        let id = self.send(|id| Request::Count {
            id,
            messages,
            max_tokens,
        })?;
        match self.read_response()? {
            Response::Count {
                id: received,
                prompt_tokens,
                reserved_output_tokens,
            } if received == id
                && reserved_output_tokens == max_tokens
                && (1..=swift::CONTEXT_TOKENS - max_tokens).contains(&prompt_tokens) =>
            {
                self.healthy = true;
                Ok(prompt_tokens)
            }
            Response::Error {
                id: Some(received),
                error,
                reset_required: None,
            } if received == id && error == "rejected_input" => {
                self.healthy = true;
                Err(ClientError::RejectedInput)
            }
            _ => Err(ClientError::InvalidResponse),
        }
    }

    /// Count then complete identical messages under one absolute deadline.
    ///
    /// The deadline must be at most ten minutes away. Caller-owned transports
    /// must enforce blocking I/O deadlines/cancellation. Every delivered delta
    /// is provisional until this returns checked terminal accounting. A delivery
    /// callback error, cancellation or expiry after sending completion poisons
    /// the connection; shut down the owned worker rather than attempting reset.
    /// A complete `generation_failed` response is synchronized and may instead
    /// permit reset/retry according to [`Self::requires_reset`].
    ///
    /// # Errors
    /// Rejects invalid settings, count/completion mismatch, output bounds,
    /// cancellation, deadlines, failed delivery and worker/transport failures.
    pub fn complete(
        &mut self,
        messages: &Value,
        max_tokens: usize,
        cancellation: &Cancellation,
        deadline: Instant,
        mut emit: impl FnMut(&str) -> crate::Result<()>,
    ) -> Result<Completion, ClientError> {
        self.admit(max_tokens)?;
        check(cancellation, deadline)?;
        if deadline.saturating_duration_since(Instant::now()) > Duration::from_secs(600) {
            return Err(ClientError::InvalidConfiguration);
        }
        let prompt_tokens = self.count(messages, max_tokens)?;
        check(cancellation, deadline)?;
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .filter(|value| !value.is_zero())
            .ok_or(ClientError::Deadline)?;
        let timeout_seconds = remaining.as_secs() + u64::from(remaining.subsec_nanos() != 0);
        let id = self.send(|id| Request::Complete {
            id,
            messages,
            max_tokens,
            timeout_seconds,
        })?;
        let mut output_bytes = 0usize;
        loop {
            check(cancellation, deadline)?;
            let response = self.read_response()?;
            check(cancellation, deadline)?;
            match response {
                Response::Delta { id: received, text }
                    if received == id && (1..=MAX_DELTA_BYTES).contains(&text.len()) =>
                {
                    output_bytes = output_bytes
                        .checked_add(text.len())
                        .filter(|&n| n <= MAX_OUTPUT_BYTES)
                        .ok_or(ClientError::InvalidResponse)?;
                    emit(&text).map_err(|_| ClientError::DeliveryFailed)?;
                    check(cancellation, deadline)?;
                }
                Response::Complete {
                    id: received,
                    prompt_tokens: reported,
                    generated_tokens,
                    finish_reason,
                } if received == id
                    && reported == prompt_tokens
                    && (generated_tokens == 0) == (output_bytes == 0)
                    && match finish_reason {
                        FinishReason::Stop => generated_tokens < max_tokens,
                        FinishReason::Length => generated_tokens == max_tokens,
                    } =>
                {
                    self.healthy = true;
                    return Ok(Completion {
                        prompt_tokens,
                        reused_tokens: 0,
                        generated_tokens,
                        finish: match finish_reason {
                            FinishReason::Stop => Finish::Stop,
                            FinishReason::Length => Finish::Length,
                        },
                    });
                }
                Response::Error {
                    id: Some(received),
                    error,
                    reset_required: Some(required),
                } if received == id && error == "generation_failed" => {
                    self.healthy = true;
                    self.reset_required = required;
                    return Err(ClientError::GenerationFailed);
                }
                Response::Error {
                    id: Some(received),
                    error,
                    reset_required: None,
                } if received == id && error == "rejected_input" && output_bytes == 0 => {
                    self.healthy = true;
                    return Err(ClientError::RejectedInput);
                }
                _ => return Err(ClientError::InvalidResponse),
            }
        }
    }

    /// Reset a synchronized connection and its model/recurrent/KV/sampler state.
    ///
    /// # Errors
    /// Failed worker reset retains reset-required state; protocol/I/O failure
    /// poisons the connection. A previously abandoned stream cannot be reset.
    pub fn reset(&mut self) -> Result<(), ClientError> {
        let id = self.send(|id| Request::Reset { id })?;
        match self.read_response()? {
            Response::Reset {
                id: received,
                ready,
            } if received == id => {
                self.healthy = true;
                self.reset_required = !ready;
                if ready {
                    Ok(())
                } else {
                    Err(ClientError::ResetFailed)
                }
            }
            _ => Err(ClientError::InvalidResponse),
        }
    }

    /// Recover streams solely for owner-managed shutdown/disposal after failure.
    #[must_use]
    pub fn into_parts(self) -> (R, W) {
        (self.input, self.output)
    }

    fn admit(&self, max_tokens: usize) -> Result<(), ClientError> {
        if !self.healthy || self.next_id.is_none() {
            return Err(ClientError::Unavailable);
        }
        if self.reset_required {
            return Err(ClientError::ResetRequired);
        }
        if !(1..=swift::MAX_OUTPUT_TOKENS).contains(&max_tokens) {
            return Err(ClientError::InvalidConfiguration);
        }
        Ok(())
    }

    fn send<'a>(&mut self, request: impl FnOnce(u64) -> Request<'a>) -> Result<u64, ClientError> {
        if !self.healthy {
            return Err(ClientError::Unavailable);
        }
        let id = self.next_id.ok_or(ClientError::Unavailable)?;
        self.request.clear();
        serde_json::to_writer(
            BoundedWriter {
                bytes: &mut self.request,
            },
            &request(id),
        )
        .map_err(|_| ClientError::RequestTooLarge)?;
        self.healthy = false;
        self.next_id = id.checked_add(1);
        self.output
            .write_all(&self.request)
            .and_then(|()| self.output.write_all(b"\n"))
            .and_then(|()| self.output.flush())
            .map_err(|_| ClientError::Transport)?;
        Ok(id)
    }

    fn read_response(&mut self) -> Result<Response, ClientError> {
        self.response.clear();
        loop {
            let available = self.input.fill_buf().map_err(|_| ClientError::Transport)?;
            if available.is_empty() {
                return Err(ClientError::InvalidResponse);
            }
            let newline = available.iter().position(|&b| b == b'\n');
            let count = newline.unwrap_or(available.len());
            if count + usize::from(newline.is_some()) > self.response_limit - self.response.len() {
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

fn check(cancellation: &Cancellation, deadline: Instant) -> Result<(), ClientError> {
    if cancellation.is_cancelled() {
        return Err(ClientError::Cancelled);
    }
    if Instant::now() >= deadline {
        return Err(ClientError::Deadline);
    }
    Ok(())
}

#[derive(Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum Request<'a> {
    Count {
        id: u64,
        messages: &'a Value,
        max_tokens: usize,
    },
    Complete {
        id: u64,
        messages: &'a Value,
        max_tokens: usize,
        timeout_seconds: u64,
    },
    Reset {
        id: u64,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "Wire readiness exposes independent fixed inference policy flags, not mutable client state"
)]
struct Ready {
    protocol: String,
    model_sha256: String,
    template_sha256: String,
    vocab_size: usize,
    runtime: String,
    prism_commit: String,
    seed: u32,
    backend: String,
    max_frame_bytes: usize,
    max_output_bytes: usize,
    context_tokens: usize,
    max_output_tokens: usize,
    temperature: u32,
    top_p: Value,
    top_k: u32,
    min_p: u32,
    repeat_penalty: u32,
    presence_penalty: u32,
    frequency_penalty: u32,
    thinking: bool,
    reasoning_effort: String,
    preserve_thinking: bool,
    cache_reuse: bool,
    speculation: bool,
    adapters: bool,
    qualification: String,
}
impl Ready {
    fn matches(&self, expected: ExpectedSwiftWorker) -> bool {
        self.protocol == "bonsai-swift-worker-v1"
            && self.model_sha256 == swift::MODEL_SHA256
            && self.template_sha256 == swift::TEMPLATE_SHA256
            && self.vocab_size == swift::VOCAB_SIZE
            && self.runtime == "bonsai-prism-runtime"
            && self.prism_commit == bonsai_gguf::CODEC_COMMIT
            && self.seed == expected.seed
            && self.backend == expected.backend.name()
            && self.max_frame_bytes == MAX_FRAME_BYTES
            && self.max_output_bytes == MAX_OUTPUT_BYTES
            && self.context_tokens == swift::CONTEXT_TOKENS
            && self.max_output_tokens == swift::MAX_OUTPUT_TOKENS
            && self.temperature == 1
            && self.top_p == json!(0.95)
            && self.top_k == 20
            && self.min_p == 0
            && self.repeat_penalty == 1
            && self.presence_penalty == 0
            && self.frequency_penalty == 0
            && self.thinking
            && self.reasoning_effort == "xhigh"
            && self.preserve_thinking
            && !self.cache_reuse
            && !self.speculation
            && !self.adapters
            && self.qualification == "pending-full-model-parity-and-device-evidence"
    }
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum FinishReason {
    Stop,
    Length,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum Response {
    Ready(Box<Ready>),
    Count {
        id: u64,
        prompt_tokens: usize,
        reserved_output_tokens: usize,
    },
    Delta {
        id: u64,
        text: String,
    },
    Complete {
        id: u64,
        prompt_tokens: usize,
        generated_tokens: usize,
        finish_reason: FinishReason,
    },
    Reset {
        id: u64,
        ready: bool,
    },
    Error {
        id: Option<u64>,
        error: String,
        reset_required: Option<bool>,
    },
}

struct BoundedWriter<'a> {
    bytes: &'a mut Vec<u8>,
}
impl Write for BoundedWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > MAX_FRAME_BYTES - 1 - self.bytes.len() {
            return Err(io::Error::other("Swift request exceeds bound"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
