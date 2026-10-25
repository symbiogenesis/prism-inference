//! Private native HTTP child for an already verified, externally owned model.
//!
//! The launcher must keep this process inside its existing memory-limited Job,
//! watchdog and listener-PID ownership boundary. Load the Swift session before
//! binding the supplied exclusive numeric-loopback listener. This module starts
//! one joined model thread, never another process, and admits one request at a
//! time. Output has one event slot and requires a write acknowledgement before
//! the next decoder step. Failed generation closes model admission until reset.

mod worker;

use crate::{
    BackendChat, ChatTemplateKwargs, MAX_BODY_BYTES, MAX_COMPLETION_TOKENS, MAX_HEADER_BYTES,
    MAX_HEADERS, Message, Object, PendingChat, Rejection, RequestPolicy,
    native_response::NativeResponse,
    reception::{HEADER_TIMEOUT, INTAKE_TIMEOUT},
    response::{CLIENT_WRITE_TIMEOUT, ResponseError},
    singleton,
    supervisor::REQUEST_TIMEOUT,
    wire,
};
use bonsai_prism_runtime::{runtime::Cancellation, swift_session::SwiftSession};
use http::{HeaderValue, Method, Request as HttpRequest, Version};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    future::poll_fn,
    pin::pin,
    sync::atomic::{AtomicBool, Ordering},
};
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, split},
    net::TcpListener,
    time::{Instant, sleep_until, timeout_at},
};
use worker::{Event, Input, Output, Worker};

/// Value-free private child errors; keys, prompts and model diagnostics are absent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeBackendError {
    /// Listener, key, framing or request syntax is outside the fixed contract.
    Configuration,
    /// The private credential was absent or invalid.
    Unauthorized,
    /// The supplied absolute deadline expired.
    Deadline,
    /// A bounded socket operation failed or ended prematurely.
    Io,
    /// The canonical body contains unknown, duplicate or overridden fields.
    Schema,
    /// Exact rendered count/context admission rejected the request.
    Admission(Rejection),
    /// Generation did not repeat the successfully counted canonical bytes.
    CountMismatch,
    /// Execution, acknowledgement or reset failed; admission may require reset.
    Engine,
    /// Client input ended, failed or supplied bytes after the complete request.
    ClientAbandoned,
    /// Bounded response delivery failed.
    Response(ResponseError),
}

/// Serve one already verified model on an existing private loopback listener.
///
/// Readiness is reachable only after the caller has loaded the pinned graph.
/// All routes require the private key, including `/health`. `/v1/models` rejects
/// missing credentials with 401 for the existing launcher authentication probe.
/// Counting retains its exact prepared input for one matching completion. The
/// authenticated private `/reset` route clears failed admission and the sampler.
/// There is no frontend administrative route or automatic reset/retry here.
///
/// Dropping this future cancels and joins its worker. Cooperative native work
/// can delay joining; the external owning Job remains the termination boundary.
///
/// # Errors
/// Rejects invalid listener/key configuration, worker startup or accept failure.
pub async fn serve(
    listener: TcpListener,
    session: SwiftSession,
    key: String,
) -> Result<(), NativeBackendError> {
    let address = listener
        .local_addr()
        .map_err(|_| NativeBackendError::Configuration)?;
    if !address.is_ipv4() || address.ip() != std::net::Ipv4Addr::LOCALHOST || address.port() < 1024
    {
        return Err(NativeBackendError::Configuration);
    }
    let policy =
        RequestPolicy::new(address.port(), &key).map_err(|_| NativeBackendError::Configuration)?;
    let seed = session.seed();
    let mut worker = Worker::new(session, seed)?;
    loop {
        let (client, peer) = listener
            .accept()
            .await
            .map_err(|_| NativeBackendError::Io)?;
        if !peer.is_ipv4() || !peer.ip().is_loopback() {
            continue;
        }
        client
            .set_nodelay(true)
            .map_err(|_| NativeBackendError::Io)?;
        let _ = serve_connection(
            client,
            &policy,
            &mut worker,
            Instant::now() + REQUEST_TIMEOUT,
        )
        .await;
    }
}

enum Route {
    Health,
    Models,
    Count,
    Generate,
    Reset,
}
struct Received {
    route: Route,
    pending: Option<PendingChat>,
}

struct CancelOnDrop<'a> {
    cancellation: &'a Cancellation,
    armed: bool,
}

impl Drop for CancelOnDrop<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.cancellation.cancel();
        }
    }
}

async fn serve_connection<C: AsyncRead + AsyncWrite + Unpin + 'static>(
    mut client: C,
    policy: &RequestPolicy,
    worker: &mut Worker,
    deadline: Instant,
) -> Result<(), NativeBackendError> {
    let received = match receive(&mut client, policy, deadline).await {
        Ok(value) => value,
        Err(error) => {
            let status = if error == NativeBackendError::Unauthorized {
                401
            } else {
                400
            };
            write_json(
                &mut client,
                status,
                &json!({"error":"private request rejected"}),
                deadline,
            )
            .await?;
            return Err(error);
        }
    };
    match received.route {
        Route::Health => {
            return write_json(&mut client, 200, &json!({"status":"ok"}), deadline).await;
        }
        Route::Models => {
            return write_json(
                &mut client,
                200,
                &json!({"data":[{"id":"Swift-Bonsai-2-PQ2_0","object":"model","model_sha256":bonsai_prism_runtime::model::swift::MODEL_SHA256,"template_sha256":bonsai_prism_runtime::model::swift::TEMPLATE_SHA256,"runtime":"prism-inference","context_tokens":8192,"seed":worker.seed}]}),
                deadline,
            )
            .await;
        }
        _ => {}
    }
    let input = match received.route {
        Route::Count => Input::Count(received.pending.ok_or(NativeBackendError::Schema)?),
        Route::Generate => Input::Generate(received.pending.ok_or(NativeBackendError::Schema)?),
        Route::Reset => Input::Reset,
        Route::Health | Route::Models => return Err(NativeBackendError::Configuration),
    };
    let cancellation = Cancellation::default();
    let result = exchange(client, worker, input, &cancellation, deadline).await;
    if result.is_err() {
        cancellation.cancel();
    }
    // A following request cannot run until recovery has left the model thread.
    worker.settled().await;
    result
}

async fn exchange<C: AsyncRead + AsyncWrite + Unpin + 'static>(
    client: C,
    worker: &mut Worker,
    input: Input,
    cancellation: &Cancellation,
    deadline: Instant,
) -> Result<(), NativeBackendError> {
    let events = worker.start(input, cancellation.clone(), deadline.into_std())?;
    // Dropping a connection future must stop its model work even when the
    // worker remains owned by the server and no error result is observed.
    let mut cancel_on_drop = CancelOnDrop {
        cancellation,
        armed: true,
    };
    let (mut input, output) = split(client);
    let mut byte = [0; 1];
    let mut abandoned = pin!(input.read(&mut byte));
    let delivered = AtomicBool::new(false);
    let mut pipeline = pin!(deliver(output, events, &delivered, deadline));
    let mut timer = pin!(sleep_until(deadline));
    let result = poll_fn(|cx| {
        if Instant::now() >= deadline || timer.as_mut().poll(cx).is_ready() {
            return std::task::Poll::Ready(Err(NativeBackendError::Deadline));
        }
        if !delivered.load(Ordering::Acquire) && abandoned.as_mut().poll(cx).is_ready() {
            return std::task::Poll::Ready(Err(NativeBackendError::ClientAbandoned));
        }
        pipeline.as_mut().poll(cx)
    })
    .await;
    // Success includes the worker's terminal write acknowledgement and final
    // execution check; provisional output alone must not disarm cancellation.
    if result.is_ok() {
        cancel_on_drop.armed = false;
    }
    result
}

async fn deliver<W: AsyncWrite + Unpin>(
    output: W,
    mut events: tokio::sync::mpsc::Receiver<Event>,
    delivered: &AtomicBool,
    deadline: Instant,
) -> Result<(), NativeBackendError> {
    let mut writer = Some(output);
    let mut response = None;
    while let Some(Event { output, ack }) = events.recv().await {
        match output {
            Output::Count(count) => {
                write_json(
                    writer.as_mut().ok_or(NativeBackendError::Engine)?,
                    200,
                    &json!({"input_tokens":count,"object":"response.input_tokens"}),
                    deadline,
                )
                .await?;
                delivered.store(true, Ordering::Release);
            }
            Output::Reset => {
                write_json(
                    writer.as_mut().ok_or(NativeBackendError::Engine)?,
                    200,
                    &json!({"status":"ok"}),
                    deadline,
                )
                .await?;
                delivered.store(true, Ordering::Release);
            }
            Output::Started { prompt, streaming } => {
                response = Some(NativeResponse::new(
                    writer.take().ok_or(NativeBackendError::Engine)?,
                    streaming,
                    prompt,
                    deadline,
                ));
            }
            Output::Delta(text) => response
                .as_mut()
                .ok_or(NativeBackendError::Engine)?
                .delta(&text)
                .await
                .map_err(NativeBackendError::Response)?,
            Output::Complete(value) => {
                response
                    .take()
                    .ok_or(NativeBackendError::Engine)?
                    .complete(value)
                    .await
                    .map_err(NativeBackendError::Response)?;
                delivered.store(true, Ordering::Release);
            }
            Output::Done(result) => {
                if result.is_err()
                    && !delivered.load(Ordering::Acquire)
                    && let Some(writer) = &mut writer
                {
                    write_json(
                        writer,
                        400,
                        &json!({"error":"native request rejected"}),
                        deadline,
                    )
                    .await?;
                }
                return result.and_then(|()| {
                    if delivered.load(Ordering::Acquire) {
                        Ok(())
                    } else {
                        Err(NativeBackendError::Engine)
                    }
                });
            }
        }
        ack.ok_or(NativeBackendError::Engine)?
            .send(())
            .map_err(|_| NativeBackendError::Engine)?;
    }
    Err(NativeBackendError::Engine)
}

async fn receive<C: AsyncRead + Unpin>(
    client: &mut C,
    policy: &RequestPolicy,
    deadline: Instant,
) -> Result<Received, NativeBackendError> {
    let start = Instant::now();
    let header_deadline = deadline.min(start + HEADER_TIMEOUT);
    let intake_deadline = deadline.min(start + INTAKE_TIMEOUT);
    let mut bytes = vec![0; MAX_HEADER_BYTES];
    let mut used = 0;
    let head = loop {
        if used == bytes.len() {
            return Err(NativeBackendError::Configuration);
        }
        let previous = used.saturating_sub(3);
        used += read_before(client, &mut bytes[used..], header_deadline).await?;
        if let Some(end) = bytes[previous..used]
            .array_windows()
            .position(|part| part == b"\r\n\r\n")
        {
            break previous + end + 4;
        }
    };
    let (route, length) = parse_head(&bytes[..head], policy)?;
    if used - head > length {
        return Err(NativeBackendError::Configuration);
    }
    let mut body = Vec::new();
    body.try_reserve_exact(length)
        .map_err(|_| NativeBackendError::Configuration)?;
    body.extend_from_slice(&bytes[head..used]);
    body.resize(length, 0);
    let mut filled = used - head;
    while filled < length {
        filled += read_before(client, &mut body[filled..], intake_deadline).await?;
    }
    let pending = if matches!(route, Route::Count | Route::Generate) {
        Some(canonical(&body)?)
    } else {
        None
    };
    Ok(Received { route, pending })
}

fn parse_head(bytes: &[u8], policy: &RequestPolicy) -> Result<(Route, usize), NativeBackendError> {
    if bytes.starts_with(b"\r\n") || !wire::valid_line_endings(bytes) {
        return Err(NativeBackendError::Configuration);
    }
    let mut headers = [httparse::EMPTY_HEADER; MAX_HEADERS];
    let mut parsed = httparse::Request::new(&mut headers);
    if !matches!(parsed.parse(bytes),Ok(httparse::Status::Complete(end)) if end == bytes.len())
        || parsed.version != Some(1)
    {
        return Err(NativeBackendError::Configuration);
    }
    let route = match (parsed.method, parsed.path) {
        (Some("GET"), Some("/health")) => Route::Health,
        (Some("GET"), Some("/v1/models" | "/props")) => Route::Models,
        (Some("POST"), Some("/v1/chat/completions/input_tokens")) => Route::Count,
        (Some("POST"), Some("/v1/chat/completions")) => Route::Generate,
        (Some("POST"), Some("/reset")) => Route::Reset,
        _ => return Err(NativeBackendError::Configuration),
    };
    let mut request = HttpRequest::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .version(Version::HTTP_11)
        .body(())
        .map_err(|_| NativeBackendError::Configuration)?;
    wire::append_headers(request.headers_mut(), parsed.headers)
        .map_err(|()| NativeBackendError::Configuration)?;
    let empty = matches!(route, Route::Health | Route::Models | Route::Reset);
    if empty {
        if !matches!(
            singleton(request.headers(), "content-length"),
            Ok(None | Some(b"0"))
        ) || request.headers().contains_key("content-type")
        {
            return Err(NativeBackendError::Configuration);
        }
        request
            .headers_mut()
            .insert("content-length", HeaderValue::from_static("1"));
        request
            .headers_mut()
            .insert("content-type", HeaderValue::from_static("application/json"));
    }
    let authorized = policy
        .check_head(&request.into_parts().0)
        .map_err(|error| {
            if error == Rejection::Unauthorized {
                NativeBackendError::Unauthorized
            } else {
                NativeBackendError::Configuration
            }
        })?;
    Ok((route, if empty { 0 } else { authorized.length }))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FixedBody<'a> {
    #[serde(borrow, deserialize_with = "crate::deserialize_messages")]
    messages: Vec<Message<'a>>,
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
    chat_template_kwargs: Object<FixedTemplate>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FixedTemplate {
    enable_thinking: bool,
    reasoning_effort: String,
    preserve_thinking: bool,
}

fn canonical(bytes: &[u8]) -> Result<PendingChat, NativeBackendError> {
    if bytes.len() > MAX_BODY_BYTES {
        return Err(NativeBackendError::Schema);
    }
    let Object(body): Object<FixedBody<'_>> =
        serde_json::from_slice(bytes).map_err(|_| NativeBackendError::Schema)?;
    if !crate::valid_history(&body.messages)
        || !(1..=MAX_COMPLETION_TOKENS).contains(&body.max_completion_tokens)
        || body.n != 1
        || body.cache_prompt
        || body.n_cache_reuse != 0
        || body.ignore_eos
        || body.top_k != 20
        || !body.chat_template_kwargs.0.enable_thinking
        || body.chat_template_kwargs.0.reasoning_effort != "xhigh"
        || !body.chat_template_kwargs.0.preserve_thinking
        || [
            body.temperature,
            body.top_p,
            body.min_p,
            body.presence_penalty,
            body.frequency_penalty,
            body.repeat_penalty,
        ]
        .map(f32::to_bits)
            != [1.0, 0.95, 0.0, 0.0, 0.0, 1.0].map(f32::to_bits)
    {
        return Err(NativeBackendError::Schema);
    }
    let canonical = serde_json::to_vec(&BackendChat {
        messages: &body.messages,
        max_completion_tokens: body.max_completion_tokens,
        stream: body.stream,
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
    .map_err(|_| NativeBackendError::Schema)?;
    Ok(PendingChat {
        body: canonical,
        max_completion_tokens: body.max_completion_tokens,
        stream: body.stream,
    })
}

async fn read_before<C: AsyncRead + Unpin>(
    client: &mut C,
    bytes: &mut [u8],
    deadline: Instant,
) -> Result<usize, NativeBackendError> {
    ensure_time(deadline)?;
    let count = timeout_at(deadline, client.read(bytes))
        .await
        .map_err(|_| NativeBackendError::Deadline)?
        .map_err(|_| NativeBackendError::Io)?;
    ensure_time(deadline)?;
    if count == 0 {
        return Err(NativeBackendError::Io);
    }
    Ok(count)
}

async fn write_json<W: AsyncWrite + Unpin>(
    client: &mut W,
    status: u16,
    body: &Value,
    deadline: Instant,
) -> Result<(), NativeBackendError> {
    let bytes = serde_json::to_vec(body).map_err(|_| NativeBackendError::Schema)?;
    let reason = if status == 200 {
        "OK"
    } else if status == 401 {
        "Unauthorized"
    } else {
        "Bad Request"
    };
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n\r\n",
        bytes.len()
    );
    let deadline = deadline.min(Instant::now() + CLIENT_WRITE_TIMEOUT);
    ensure_time(deadline)?;
    timeout_at(deadline, async {
        client.write_all(head.as_bytes()).await?;
        client.write_all(&bytes).await?;
        client.flush().await?;
        client.shutdown().await
    })
    .await
    .map_err(|_| NativeBackendError::Deadline)?
    .map_err(|_| NativeBackendError::Io)?;
    ensure_time(deadline)
}

fn ensure_time(deadline: Instant) -> Result<(), NativeBackendError> {
    if Instant::now() >= deadline {
        Err(NativeBackendError::Deadline)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests;
