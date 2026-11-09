//! Bounded native Swift HTTP output, with incremental reasoning separation.
//!
//! This supplied-stream component binds no listener and starts no task or model.
//! The owner retains the existing authenticated intake, single-request gate and
//! process-memory policy. Await every operation before acknowledging its native
//! generation callback; failed or abandoned terminal writes require reset.

use bonsai_prism_runtime::runtime::{Completion, Finish};
use serde::Serialize;
use std::io::Write as _;
use tokio::{
    io::{AsyncWrite, AsyncWriteExt},
    time::{Instant, timeout_at},
};

use crate::response::{CLIENT_WRITE_TIMEOUT, MAX_RESPONSE_BYTES, ResponseError};

const INPUT_CHUNK_BYTES: usize = 1024;
const INITIAL_BUFFER_CAPACITY: usize = 512;
const THINK_END: &str = "</think>";
const RESPONSE_ID: &str = "chatcmpl-swift";
const MODEL_ID: &str = "Swift-Bonsai-2-PQ2_0";
const CHUNK_OBJECT: &str = "chat.completion.chunk";
const COMPLETION_OBJECT: &str = "chat.completion";
const COMMON_HEADERS: &str =
    "Connection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n";
const STREAM_HEAD: &[u8] = b"HTTP/1.1 200 OK\r\nConnection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n";

#[derive(Serialize)]
struct StreamChunk<D> {
    choices: [StreamChoice<D>; 1],
    id: &'static str,
    model: &'static str,
    object: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<UsagePayload>,
}

#[derive(Serialize)]
struct StreamChoice<D> {
    delta: D,
    finish_reason: Option<&'static str>,
    index: u8,
}

#[derive(Serialize)]
struct RoleDelta {
    role: &'static str,
}

#[derive(Serialize)]
struct ReasoningDelta<'a> {
    reasoning_content: &'a str,
}

#[derive(Serialize)]
struct ContentDelta<'a> {
    content: &'a str,
}

#[derive(Serialize)]
struct EmptyDelta {}

#[derive(Clone, Copy, Serialize)]
#[expect(
    clippy::struct_field_names,
    reason = "OpenAI-compatible usage payload field names."
)]
struct UsagePayload {
    completion_tokens: usize,
    prompt_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
struct CompletionEnvelope<'a> {
    choices: [CompletionChoice<'a>; 1],
    id: &'static str,
    model: &'static str,
    object: &'static str,
    usage: UsagePayload,
}

#[derive(Serialize)]
struct CompletionChoice<'a> {
    finish_reason: &'static str,
    index: u8,
    message: AssistantMessage<'a>,
}

#[derive(Serialize)]
struct AssistantMessage<'a> {
    content: &'a str,
    reasoning_content: &'a str,
    role: &'static str,
}

const fn stream_chunk<D>(
    delta: D,
    finish_reason: Option<&'static str>,
    usage: Option<UsagePayload>,
) -> StreamChunk<D> {
    StreamChunk {
        choices: [StreamChoice {
            delta,
            finish_reason,
            index: 0,
        }],
        id: RESPONSE_ID,
        model: MODEL_ID,
        object: CHUNK_OBJECT,
        usage,
    }
}

#[derive(Clone, Copy, Default)]
struct Segments<'a> {
    reasoning: &'a str,
    content: &'a str,
}

#[derive(Default)]
struct CollectedSegments {
    reasoning: String,
    content: String,
}

struct Reasoning {
    pending: String,
    scratch: String,
    thinking: bool,
}

impl Reasoning {
    const fn new(thinking: bool) -> Self {
        Self {
            pending: String::new(),
            scratch: String::new(),
            thinking,
        }
    }

    fn push<'a>(&'a mut self, text: &'a str) -> Segments<'a> {
        if !self.thinking {
            return Segments {
                reasoning: "",
                content: text,
            };
        }
        self.pending.push_str(text);
        if let Some(end) = self.pending.find(THINK_END) {
            self.scratch.clear();
            self.scratch
                .push_str(&self.pending[end + THINK_END.len()..]);
            self.pending.truncate(end);
            self.thinking = false;
            return Segments {
                reasoning: &self.pending,
                content: &self.scratch,
            };
        }
        // Only an exact suffix of the ASCII delimiter can cross a UTF-8 chunk.
        // It holds at most seven bytes; no user or model text is trimmed.
        let keep = (1..THINK_END.len())
            .rev()
            .find(|&size| {
                self.pending
                    .as_bytes()
                    .ends_with(&THINK_END.as_bytes()[..size])
            })
            .unwrap_or(0);
        let emit_len = self.pending.len() - keep;
        self.scratch.clear();
        self.scratch.push_str(&self.pending[..emit_len]);
        self.pending.drain(..emit_len);
        Segments {
            reasoning: &self.scratch,
            content: "",
        }
    }

    fn finish(&mut self) -> Segments<'_> {
        if !self.thinking {
            self.pending.clear();
            return Segments::default();
        }
        self.thinking = false;
        Segments {
            reasoning: &self.pending,
            content: "",
        }
    }
}

struct Delivery<W> {
    client: W,
    streaming: bool,
    prompt_tokens: usize,
    deadline: Instant,
    started: bool,
    sent: usize,
    collected: CollectedSegments,
    payload_buffer: Vec<u8>,
    frame_buffer: Vec<u8>,
}

/// A one-response native writer with the broker's existing byte/write bounds.
///
/// With thinking enabled, Swift's prompt already includes its opening marker. Text
/// before the first generated `</think>` is emitted as `reasoning_content`, with
/// subsequent text emitted as `content`. Supplied reasoning can then be replayed
/// through the existing strict chat schema. An incomplete marker at stop/length
/// remains literal reasoning text. With thinking disabled, the prompt has already
/// closed the block and every generated byte is answer content. Model bytes
/// outside the enabled-mode delimiter are exact.
pub struct NativeResponse<W> {
    parser: Reasoning,
    delivery: Delivery<W>,
}

impl<W: AsyncWrite + Unpin> NativeResponse<W> {
    /// Own one supplied client writer after authentication and exact admission.
    ///
    /// No response bytes are sent until a delta or completion is delivered.
    #[must_use]
    pub fn new(client: W, streaming: bool, prompt_tokens: usize, deadline: Instant) -> Self {
        Self::with_thinking(client, streaming, prompt_tokens, deadline, true)
    }

    /// Own a writer with the same thinking mode as the exact prepared prompt.
    ///
    /// With thinking disabled, every generated byte is answer content; the
    /// prompt has already closed its reasoning block before generation starts.
    #[must_use]
    pub fn with_thinking(
        client: W,
        streaming: bool,
        prompt_tokens: usize,
        deadline: Instant,
        enable_thinking: bool,
    ) -> Self {
        let capacity = if streaming {
            INITIAL_BUFFER_CAPACITY
        } else {
            0
        };
        Self {
            parser: Reasoning::new(enable_thinking),
            delivery: Delivery {
                client,
                streaming,
                prompt_tokens,
                deadline,
                started: false,
                sent: 0,
                collected: CollectedSegments::default(),
                payload_buffer: Vec::with_capacity(capacity),
                frame_buffer: Vec::with_capacity(capacity),
            },
        }
    }

    /// Deliver a provisional UTF-8 chunk under bounded client backpressure.
    ///
    /// # Errors
    /// Returns value-free byte, deadline or I/O errors. Never retry on this
    /// writer after an error; the transport may already contain partial bytes.
    pub async fn delta(&mut self, mut text: &str) -> Result<(), ResponseError> {
        while !text.is_empty() {
            let mut end = text.len().min(INPUT_CHUNK_BYTES);
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            let segments = self.parser.push(&text[..end]);
            self.delivery.deliver(segments).await?;
            text = &text[end..];
        }
        ensure_time(self.delivery.deadline)
    }

    /// Deliver complete stop/length accounting, flush and close the response.
    ///
    /// The caller acknowledges the terminal native event only after success.
    /// A response cut off by the output cap reports `length`, including when all
    /// generated text is still reasoning and no final answer was produced.
    ///
    /// # Errors
    /// Returns value-free byte, deadline or I/O errors; the session must remain
    /// closed until reset if its terminal event could not be acknowledged.
    pub async fn complete(mut self, completion: Completion) -> Result<(), ResponseError> {
        if completion.prompt_tokens != self.delivery.prompt_tokens {
            return Err(ResponseError::BackendMetadata);
        }
        let tail = self.parser.finish();
        self.delivery.deliver(tail).await?;
        self.delivery.finish(completion).await
    }
}

impl<W: AsyncWrite + Unpin> Delivery<W> {
    async fn finish(mut self, completion: Completion) -> Result<(), ResponseError> {
        let reason = match completion.finish {
            Finish::Stop => "stop",
            Finish::Length => "length",
        };
        let total = self
            .prompt_tokens
            .checked_add(completion.generated_tokens)
            .ok_or(ResponseError::BodyLimit)?;
        let usage = UsagePayload {
            completion_tokens: completion.generated_tokens,
            prompt_tokens: self.prompt_tokens,
            total_tokens: total,
        };
        if self.streaming {
            self.start_stream().await?;
            self.frame(&stream_chunk(EmptyDelta {}, Some(reason), Some(usage)))
                .await?;
            self.chunk(b"data: [DONE]\n\n").await?;
            write_group(&mut self.client, &[b"0\r\n\r\n"], self.deadline, true).await
        } else {
            let bytes = serde_json::to_vec(&CompletionEnvelope {
                choices: [CompletionChoice {
                    finish_reason: reason,
                    index: 0,
                    message: AssistantMessage {
                        content: &self.collected.content,
                        reasoning_content: &self.collected.reasoning,
                        role: "assistant",
                    },
                }],
                id: RESPONSE_ID,
                model: MODEL_ID,
                object: COMPLETION_OBJECT,
                usage,
            })
            .map_err(|_| ResponseError::JsonSyntax)?;
            charge(&mut self.sent, bytes.len())?;
            let head = format!(
                "HTTP/1.1 200 OK\r\n{COMMON_HEADERS}Content-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                bytes.len()
            );
            write_group(
                &mut self.client,
                &[head.as_bytes(), &bytes],
                self.deadline,
                true,
            )
            .await
        }
    }

    async fn deliver(&mut self, segments: Segments<'_>) -> Result<(), ResponseError> {
        if self.streaming {
            if !segments.reasoning.is_empty() {
                self.start_stream().await?;
                self.frame(&stream_chunk(
                    ReasoningDelta {
                        reasoning_content: segments.reasoning,
                    },
                    None,
                    None,
                ))
                .await?;
            }
            if !segments.content.is_empty() {
                self.start_stream().await?;
                self.frame(&stream_chunk(
                    ContentDelta {
                        content: segments.content,
                    },
                    None,
                    None,
                ))
                .await?;
            }
        } else {
            let retained = self
                .collected
                .reasoning
                .len()
                .checked_add(self.collected.content.len())
                .and_then(|n| n.checked_add(segments.reasoning.len()))
                .and_then(|n| n.checked_add(segments.content.len()))
                .ok_or(ResponseError::BodyLimit)?;
            if retained > MAX_RESPONSE_BYTES {
                return Err(ResponseError::BodyLimit);
            }
            self.collected.reasoning.push_str(segments.reasoning);
            self.collected.content.push_str(segments.content);
        }
        ensure_time(self.deadline)
    }

    async fn start_stream(&mut self) -> Result<(), ResponseError> {
        if !self.started {
            write_group(&mut self.client, &[STREAM_HEAD], self.deadline, false).await?;
            self.started = true;
            self.frame(&stream_chunk(RoleDelta { role: "assistant" }, None, None))
                .await?;
        }
        Ok(())
    }

    async fn frame(&mut self, value: &impl Serialize) -> Result<(), ResponseError> {
        self.payload_buffer.clear();
        self.payload_buffer.extend_from_slice(b"data: ");
        serde_json::to_writer(&mut self.payload_buffer, value)
            .map_err(|_| ResponseError::JsonSyntax)?;
        self.payload_buffer.extend_from_slice(b"\n\n");
        charge(&mut self.sent, self.payload_buffer.len())?;
        self.frame_buffer.clear();
        let _ = write!(self.frame_buffer, "{:X}\r\n", self.payload_buffer.len());
        self.frame_buffer.extend_from_slice(&self.payload_buffer);
        self.frame_buffer.extend_from_slice(b"\r\n");
        write_group(
            &mut self.client,
            &[&self.frame_buffer],
            self.deadline,
            false,
        )
        .await
    }

    async fn chunk(&mut self, bytes: &[u8]) -> Result<(), ResponseError> {
        charge(&mut self.sent, bytes.len())?;
        self.frame_buffer.clear();
        let _ = write!(self.frame_buffer, "{:X}\r\n", bytes.len());
        self.frame_buffer.extend_from_slice(bytes);
        self.frame_buffer.extend_from_slice(b"\r\n");
        write_group(
            &mut self.client,
            &[&self.frame_buffer],
            self.deadline,
            false,
        )
        .await
    }
}

fn charge(sent: &mut usize, bytes: usize) -> Result<(), ResponseError> {
    *sent = sent
        .checked_add(bytes)
        .filter(|&n| n <= MAX_RESPONSE_BYTES)
        .ok_or(ResponseError::BodyLimit)?;
    Ok(())
}

fn ensure_time(deadline: Instant) -> Result<(), ResponseError> {
    if Instant::now() >= deadline {
        Err(ResponseError::Deadline)
    } else {
        Ok(())
    }
}

async fn write_group<W: AsyncWrite + Unpin>(
    client: &mut W,
    pieces: &[&[u8]],
    supervisor_deadline: Instant,
    shutdown: bool,
) -> Result<(), ResponseError> {
    let deadline = supervisor_deadline.min(Instant::now() + CLIENT_WRITE_TIMEOUT);
    ensure_time(deadline)?;
    timeout_at(deadline, async {
        for piece in pieces {
            client.write_all(piece).await?;
        }
        client.flush().await?;
        if shutdown {
            client.shutdown().await?;
        }
        Ok::<_, std::io::Error>(())
    })
    .await
    .map_err(|_| ResponseError::Deadline)?
    .map_err(|_| ResponseError::ClientIo)?;
    ensure_time(deadline)
}
