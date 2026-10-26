//! Native output keeps supplied reasoning replayable across fragmented UTF-8.

#![forbid(unsafe_code)]

use bonsai_prism_runtime::runtime::{Completion, Finish};
use prism_inference_server::{
    native_response::NativeResponse,
    response::{MAX_RESPONSE_BYTES, ResponseError},
};
use serde_json::Value;
use std::{
    cell::RefCell,
    io,
    pin::Pin,
    rc::Rc,
    task::{Context, Poll},
    time::Duration,
};
use tokio::{io::AsyncWrite, time::Instant};

type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

#[derive(Default)]
struct State {
    bytes: Vec<u8>,
    shutdowns: usize,
}

struct Writer {
    state: Rc<RefCell<State>>,
    stall: bool,
    fail_shutdown: bool,
}

impl AsyncWrite for Writer {
    fn poll_write(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        bytes: &[u8],
    ) -> Poll<io::Result<usize>> {
        let count = bytes.len().min(17);
        self.state
            .borrow_mut()
            .bytes
            .extend_from_slice(&bytes[..count]);
        Poll::Ready(Ok(count))
    }
    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        if self.stall {
            Poll::Pending
        } else {
            Poll::Ready(Ok(()))
        }
    }
    fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        self.state.borrow_mut().shutdowns += 1;
        if self.fail_shutdown {
            Poll::Ready(Err(io::Error::other("abandoned terminal delivery")))
        } else {
            Poll::Ready(Ok(()))
        }
    }
}

fn run(test: impl Future<Output = TestResult>) -> TestResult {
    tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .start_paused(true)
        .build()?
        .block_on(test)
}

fn writer() -> (Writer, Rc<RefCell<State>>) {
    let state = Rc::new(RefCell::new(State::default()));
    (
        Writer {
            state: state.clone(),
            stall: false,
            fail_shutdown: false,
        },
        state,
    )
}

const fn completion(finish: Finish) -> Completion {
    Completion {
        prompt_tokens: 100,
        reused_tokens: 0,
        generated_tokens: 3,
        finish,
    }
}

fn body(
    bytes: &[u8],
    streaming: bool,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    let start = bytes
        .array_windows()
        .position(|part| part == b"\r\n\r\n")
        .ok_or("missing head")?
        + 4;
    if !streaming {
        return Ok(bytes[start..].to_vec());
    }
    let mut input = &bytes[start..];
    let mut decoded = Vec::new();
    loop {
        let end = input
            .array_windows()
            .position(|part| part == b"\r\n")
            .ok_or("missing chunk size")?;
        let count = usize::from_str_radix(std::str::from_utf8(&input[..end])?, 16)?;
        input = &input[end + 2..];
        if count == 0 {
            assert_eq!(input, b"\r\n");
            return Ok(decoded);
        }
        decoded.extend_from_slice(&input[..count]);
        assert_eq!(&input[count..count + 2], b"\r\n");
        input = &input[count + 2..];
    }
}

#[test]
fn native_thinking_off_stream_and_json_keep_all_generated_text_visible() -> TestResult {
    run(async {
        let text = "答え 🌱</think>literal suffix";
        for streaming in [false, true] {
            let (client, state) = writer();
            let mut output = NativeResponse::with_thinking(
                client,
                streaming,
                100,
                Instant::now() + Duration::from_secs(60),
                false,
            );
            for character in text.chars() {
                output
                    .delta(character.encode_utf8(&mut [0; 4]))
                    .await
                    .map_err(|e| format!("{e:?}"))?;
            }
            output
                .complete(completion(Finish::Stop))
                .await
                .map_err(|e| format!("{e:?}"))?;
            let state = state.borrow();
            let payload = body(&state.bytes, streaming)?;
            let mut content = String::new();
            if streaming {
                for frame in std::str::from_utf8(&payload)?
                    .split("\n\n")
                    .filter(|s| !s.is_empty() && *s != "data: [DONE]")
                {
                    let event: Value =
                        serde_json::from_str(frame.strip_prefix("data: ").ok_or("bad frame")?)?;
                    assert!(
                        event["choices"][0]["delta"]
                            .get("reasoning_content")
                            .is_none()
                    );
                    content.push_str(
                        event["choices"][0]["delta"]["content"]
                            .as_str()
                            .unwrap_or(""),
                    );
                }
            } else {
                let value: Value = serde_json::from_slice(&payload)?;
                assert_eq!(value["choices"][0]["message"]["reasoning_content"], "");
                content.push_str(
                    value["choices"][0]["message"]["content"]
                        .as_str()
                        .ok_or("missing answer")?,
                );
            }
            assert_eq!(content, text);
        }
        Ok(())
    })
}

#[test]
fn native_stream_and_json_preserve_reasoning_at_every_delimiter_boundary() -> TestResult {
    run(async {
        let text = "思考</thinX> 🌱</think>\n答え</think>";
        for streaming in [false, true] {
            for boundary in text
                .char_indices()
                .map(|(index, _)| index)
                .chain([text.len()])
            {
                let (client, state) = writer();
                let mut output = NativeResponse::new(
                    client,
                    streaming,
                    100,
                    Instant::now() + Duration::from_secs(60),
                );
                output
                    .delta(&text[..boundary])
                    .await
                    .map_err(|e| format!("{e:?}"))?;
                output
                    .delta(&text[boundary..])
                    .await
                    .map_err(|e| format!("{e:?}"))?;
                output
                    .complete(completion(Finish::Length))
                    .await
                    .map_err(|e| format!("{e:?}"))?;
                let state = state.borrow();
                assert_eq!(state.shutdowns, 1);
                let payload = body(&state.bytes, streaming)?;
                let (mut reasoning, mut content) = (String::new(), String::new());
                if streaming {
                    let text = std::str::from_utf8(&payload)?;
                    assert!(text.ends_with("data: [DONE]\n\n"));
                    for frame in text
                        .split("\n\n")
                        .filter(|s| !s.is_empty() && *s != "data: [DONE]")
                    {
                        let event: Value =
                            serde_json::from_str(frame.strip_prefix("data: ").ok_or("bad frame")?)?;
                        let choice = &event["choices"][0];
                        reasoning
                            .push_str(choice["delta"]["reasoning_content"].as_str().unwrap_or(""));
                        content.push_str(choice["delta"]["content"].as_str().unwrap_or(""));
                        if !choice["finish_reason"].is_null() {
                            assert_eq!(choice["finish_reason"], "length");
                            assert_eq!(event["usage"]["prompt_tokens"], 100);
                        }
                    }
                } else {
                    let event: Value = serde_json::from_slice(&payload)?;
                    reasoning = event["choices"][0]["message"]["reasoning_content"]
                        .as_str()
                        .ok_or("missing reasoning")?
                        .into();
                    content = event["choices"][0]["message"]["content"]
                        .as_str()
                        .ok_or("missing content")?
                        .into();
                    assert_eq!(event["choices"][0]["finish_reason"], "length");
                    assert_eq!(event["usage"]["total_tokens"], 103);
                }
                assert_eq!(reasoning, "思考</thinX> 🌱");
                assert_eq!(content, "\n答え</think>");
            }
        }
        Ok(())
    })
}

#[test]
fn native_output_retains_partial_markers_and_never_completes_failed_terminal_delivery() -> TestResult
{
    run(async {
        let (client, state) = writer();
        let mut output =
            NativeResponse::new(client, false, 100, Instant::now() + Duration::from_secs(60));
        output
            .delta("unfinished </thi")
            .await
            .map_err(|e| format!("{e:?}"))?;
        output
            .complete(completion(Finish::Stop))
            .await
            .map_err(|e| format!("{e:?}"))?;
        let event: Value = serde_json::from_slice(&body(&state.borrow().bytes, false)?)?;
        assert_eq!(
            event["choices"][0]["message"]["reasoning_content"],
            "unfinished </thi"
        );
        assert_eq!(event["choices"][0]["message"]["content"], "");
        assert_eq!(event["choices"][0]["finish_reason"], "stop");
        let (mut client, _) = writer();
        client.fail_shutdown = true;
        let output =
            NativeResponse::new(client, true, 100, Instant::now() + Duration::from_secs(60));
        assert_eq!(
            output.complete(completion(Finish::Stop)).await,
            Err(ResponseError::ClientIo)
        );
        Ok(())
    })
}

#[test]
fn native_output_bounds_framing_expansion_and_client_write_time() -> TestResult {
    run(async {
        for streaming in [false, true] {
            let (client, state) = writer();
            let mut output = NativeResponse::new(
                client,
                streaming,
                100,
                Instant::now() + Duration::from_secs(60),
            );
            let text = "\0".repeat(MAX_RESPONSE_BYTES / 6 + 1);
            let delta = output.delta(&text).await;
            let result = if delta.is_ok() {
                output.complete(completion(Finish::Length)).await
            } else {
                delta
            };
            assert_eq!(result, Err(ResponseError::BodyLimit));
            assert_eq!(state.borrow().shutdowns, 0);
            assert!(!state.borrow().bytes.ends_with(b"data: [DONE]\n\n0\r\n\r\n"));
        }
        let (mut client, _) = writer();
        client.stall = true;
        let mut output =
            NativeResponse::new(client, true, 100, Instant::now() + Duration::from_secs(60));
        let start = Instant::now();
        assert_eq!(
            output.delta("reasoning").await,
            Err(ResponseError::Deadline)
        );
        assert_eq!(Instant::now() - start, Duration::from_secs(5));
        Ok(())
    })
}
