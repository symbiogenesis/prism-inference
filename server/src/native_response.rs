//! Bounded native Swift HTTP output, with incremental reasoning separation.
//!
//! This supplied-stream component binds no listener and starts no task or model.
//! The owner retains the existing authenticated intake, single-request gate and
//! process-memory policy. Await every operation before acknowledging its native
//! generation callback; failed or abandoned terminal writes require reset.

use bonsai_prism_runtime::runtime::{Completion, Finish};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncWrite, AsyncWriteExt},
    time::{Instant, timeout_at},
};

use crate::response::{CLIENT_WRITE_TIMEOUT, MAX_RESPONSE_BYTES, ResponseError};

const INPUT_CHUNK_BYTES: usize = 1024;
const THINK_END: &str = "</think>";
const COMMON_HEADERS: &str =
    "Connection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n";

#[derive(Default)]
struct Segments {
    reasoning: String,
    content: String,
}

struct Reasoning {
    pending: String,
    thinking: bool,
}

impl Reasoning {
    const fn new() -> Self {
        Self {
            pending: String::new(),
            thinking: true,
        }
    }

    fn push(&mut self, text: &str) -> Segments {
        if !self.thinking {
            return Segments {
                reasoning: String::new(),
                content: text.to_owned(),
            };
        }
        self.pending.push_str(text);
        if let Some(end) = self.pending.find(THINK_END) {
            let content = self.pending[end + THINK_END.len()..].to_owned();
            self.pending.truncate(end);
            self.thinking = false;
            return Segments {
                reasoning: std::mem::take(&mut self.pending),
                content,
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
        let tail = self.pending.split_off(self.pending.len() - keep);
        Segments {
            reasoning: std::mem::replace(&mut self.pending, tail),
            content: String::new(),
        }
    }

    fn finish(&mut self) -> Segments {
        Segments {
            reasoning: std::mem::take(&mut self.pending),
            content: String::new(),
        }
    }
}

/// A one-response native writer with the broker's existing byte/write bounds.
///
/// Swift's rendered thinking prompt already includes its opening marker. Text
/// before the first generated `</think>` is emitted as `reasoning_content`, with
/// subsequent text emitted as `content`. Supplied reasoning can then be replayed
/// through the existing strict chat schema. An incomplete marker at stop/length
/// remains literal reasoning text. Model bytes outside the delimiter are exact.
pub struct NativeResponse<W> {
    client: W,
    streaming: bool,
    prompt_tokens: usize,
    deadline: Instant,
    started: bool,
    sent: usize,
    parser: Reasoning,
    collected: Segments,
}

impl<W: AsyncWrite + Unpin> NativeResponse<W> {
    /// Own one supplied client writer after authentication and exact admission.
    ///
    /// No response bytes are sent until a delta or completion is delivered.
    #[must_use]
    pub fn new(client: W, streaming: bool, prompt_tokens: usize, deadline: Instant) -> Self {
        Self {
            client,
            streaming,
            prompt_tokens,
            deadline,
            started: false,
            sent: 0,
            parser: Reasoning::new(),
            collected: Segments::default(),
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
            self.deliver(segments).await?;
            text = &text[end..];
        }
        ensure_time(self.deadline)
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
        if completion.prompt_tokens != self.prompt_tokens {
            return Err(ResponseError::BackendMetadata);
        }
        let tail = self.parser.finish();
        self.deliver(tail).await?;
        let reason = match completion.finish {
            Finish::Stop => "stop",
            Finish::Length => "length",
        };
        let total = self
            .prompt_tokens
            .checked_add(completion.generated_tokens)
            .ok_or(ResponseError::BodyLimit)?;
        let usage = json!({"prompt_tokens":self.prompt_tokens,"completion_tokens":completion.generated_tokens,"total_tokens":total});
        if self.streaming {
            self.start_stream().await?;
            self.frame(json!({"id":"chatcmpl-swift","object":"chat.completion.chunk","model":"Swift-Bonsai-2-PQ2_0","choices":[{"index":0,"delta":{},"finish_reason":reason}],"usage":usage})).await?;
            self.chunk(b"data: [DONE]\n\n").await?;
            write_group(&mut self.client, &[b"0\r\n\r\n"], self.deadline, true).await
        } else {
            let bytes = serde_json::to_vec(&json!({"id":"chatcmpl-swift","object":"chat.completion","model":"Swift-Bonsai-2-PQ2_0","choices":[{"index":0,"message":{"role":"assistant","content":self.collected.content,"reasoning_content":self.collected.reasoning},"finish_reason":reason}],"usage":usage})).map_err(|_|ResponseError::JsonSyntax)?;
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

    async fn deliver(&mut self, segments: Segments) -> Result<(), ResponseError> {
        if self.streaming {
            for (field, text) in [
                ("reasoning_content", segments.reasoning),
                ("content", segments.content),
            ] {
                if !text.is_empty() {
                    self.start_stream().await?;
                    self.frame(json!({"id":"chatcmpl-swift","object":"chat.completion.chunk","model":"Swift-Bonsai-2-PQ2_0","choices":[{"index":0,"delta":{field:text},"finish_reason":null}]})).await?;
                }
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
            self.collected.reasoning.push_str(&segments.reasoning);
            self.collected.content.push_str(&segments.content);
        }
        ensure_time(self.deadline)
    }

    async fn start_stream(&mut self) -> Result<(), ResponseError> {
        if !self.started {
            let head = format!(
                "HTTP/1.1 200 OK\r\n{COMMON_HEADERS}Content-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
            );
            write_group(&mut self.client, &[head.as_bytes()], self.deadline, false).await?;
            self.started = true;
            self.frame(json!({"id":"chatcmpl-swift","object":"chat.completion.chunk","model":"Swift-Bonsai-2-PQ2_0","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]})).await?;
        }
        Ok(())
    }

    async fn frame(&mut self, value: Value) -> Result<(), ResponseError> {
        let mut bytes = b"data: ".to_vec();
        serde_json::to_writer(&mut bytes, &value).map_err(|_| ResponseError::JsonSyntax)?;
        bytes.extend_from_slice(b"\n\n");
        self.chunk(&bytes).await
    }

    async fn chunk(&mut self, bytes: &[u8]) -> Result<(), ResponseError> {
        charge(&mut self.sent, bytes.len())?;
        let prefix = format!("{:X}\r\n", bytes.len());
        write_group(
            &mut self.client,
            &[prefix.as_bytes(), bytes, b"\r\n"],
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
