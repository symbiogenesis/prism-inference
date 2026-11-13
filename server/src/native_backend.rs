//! Private native HTTP child for an already verified, externally owned model.
//!
//! The launcher must keep this process inside its existing memory-limited Job,
//! watchdog and listener-PID ownership boundary. Load the Swift session before
//! binding the supplied exclusive numeric-loopback listener. This module starts
//! one joined model thread, never another process, and admits one request at a
//! time. Output has one event slot and requires a write acknowledgement before
//! the next decoder step. Failed generation closes model admission until reset.

mod profile;
mod worker;

use crate::{
    BackendChat, ChatTemplateKwargs, MAX_BODY_BYTES, MAX_COMPLETION_TOKENS, MAX_HEADER_BYTES,
    MAX_HEADERS, Message, Object, PendingChat, Rejection, RequestPolicy,
    native_response::NativeResponse,
    reception::{HEADER_TIMEOUT, INTAKE_TIMEOUT},
    response::{CLIENT_WRITE_TIMEOUT, ResponseError},
    singleton, wire,
};
use bonsai_prism_runtime::{runtime::Cancellation, swift_session::SwiftSession};

use profile::ModelProfile;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Value, json};
use std::{
    borrow::Cow,
    future::poll_fn,
    pin::pin,
    sync::atomic::{AtomicBool, Ordering},
};
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, split},
    net::TcpListener,
    time::{Instant, timeout_at},
};
#[cfg(windows)]
pub(super) use worker::register_hexagon_model_path;
use worker::{Event, Input, Output, Worker};

/// Immutable projection placement requested by the trusted private child owner.
///
/// This selects execution only; it does not reduce memory admission or qualify
/// a larger context. Requests cannot change placement after model startup.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Placement {
    /// Preserve the existing optional resident Hexagon installation and CPU fallback.
    #[default]
    Auto,
    /// Execute projections on the CPU without discovering or opening accelerators.
    Cpu,
}

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
/// Ordinary requests have no aggregate elapsed-time cutoff. Intake and each
/// response write retain their independent deadlines; cancellation stays active.
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
    serve_with_placement(listener, session, key, Placement::Auto).await
}

/// Serve a verified session with one immutable trusted projection placement.
///
/// CPU placement clears any supplied projection executor before worker creation
/// and bypasses accelerator initialization on the model thread. Its readiness
/// declares both the requested placement and CPU projection backend. Automatic
/// placement preserves the existing readiness schema and initialization policy.
/// Process, context and state memory allowances remain unchanged.
///
/// # Errors
/// Rejects invalid listener/key configuration, worker startup or accept failure.
pub async fn serve_with_placement(
    listener: TcpListener,
    mut session: SwiftSession,
    key: String,
    placement: Placement,
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
    if placement == Placement::Cpu {
        session.set_projection_executor(None);
    }
    let seed = session.seed();
    let mut worker = Worker::with_placement(session, seed, placement)?;
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
        if let Err(error) = serve_connection(client, &policy, &mut worker, None).await {
            let _ = write_request_failure(&mut std::io::stderr().lock(), error);
        }
    }
}

fn write_request_failure(
    writer: &mut impl std::io::Write,
    error: NativeBackendError,
) -> std::io::Result<()> {
    if matches!(
        error,
        NativeBackendError::Unauthorized | NativeBackendError::Admission(Rejection::ContextLimit)
    ) {
        return Ok(());
    }
    writeln!(writer, "private-inference-request-failed error={error:?}")
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
    deadline: impl Into<Option<Instant>>,
) -> Result<(), NativeBackendError> {
    let deadline = deadline.into();
    let received = match receive(&mut client, policy, &worker.profile, deadline).await {
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
                &worker.profile.readiness(worker.seed),
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
    deadline: Option<Instant>,
) -> Result<(), NativeBackendError> {
    let events = worker.start(input, cancellation.clone(), deadline.map(Instant::into_std))?;
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
    let mut timer = pin!(async {
        if let Some(deadline) = deadline {
            tokio::time::sleep_until(deadline).await;
        } else {
            std::future::pending::<()>().await;
        }
    });
    let result = poll_fn(|cx| {
        if deadline.is_some_and(|deadline| Instant::now() >= deadline)
            || timer.as_mut().poll(cx).is_ready()
        {
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
    deadline: Option<Instant>,
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
            Output::Started {
                prompt,
                streaming,
                thinking,
            } => {
                response = Some(NativeResponse::with_thinking(
                    writer.take().ok_or(NativeBackendError::Engine)?,
                    streaming,
                    prompt,
                    deadline,
                    thinking,
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
                    let context =
                        result == Err(NativeBackendError::Admission(Rejection::ContextLimit));
                    let status = if context { 413 } else { 400 };
                    let body = if context {
                        json!({"error":{"code":"context_length_exceeded"}})
                    } else {
                        json!({"error":"native request rejected"})
                    };
                    write_json(writer, status, &body, deadline).await?;
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
    profile: &ModelProfile,
    deadline: Option<Instant>,
) -> Result<Received, NativeBackendError> {
    let start = Instant::now();
    let header_deadline = deadline.map_or(start + HEADER_TIMEOUT, |limit| {
        limit.min(start + HEADER_TIMEOUT)
    });
    let intake_deadline = deadline.map_or(start + INTAKE_TIMEOUT, |limit| {
        limit.min(start + INTAKE_TIMEOUT)
    });
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
    let body = if length <= bytes.len() - head {
        let body_end = head + length;
        while used < body_end {
            used += read_before(client, &mut bytes[used..body_end], intake_deadline).await?;
        }
        &bytes[head..body_end]
    } else {
        let prefix = used - head;
        bytes.copy_within(head..used, 0);
        bytes.truncate(prefix);
        bytes
            .try_reserve_exact(length - prefix)
            .map_err(|_| NativeBackendError::Configuration)?;
        bytes.resize(length, 0);
        let mut filled = prefix;
        while filled < length {
            filled += read_before(client, &mut bytes[filled..], intake_deadline).await?;
        }
        &bytes[..]
    };
    let pending = if matches!(route, Route::Count | Route::Generate) {
        Some(canonical_for_profile(body, profile)?)
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
    if !wire::valid_headers(parsed.headers) {
        return Err(NativeBackendError::Configuration);
    }
    let map_rejection = |error| {
        if error == Rejection::Unauthorized {
            NativeBackendError::Unauthorized
        } else {
            NativeBackendError::Configuration
        }
    };
    if matches!(route, Route::Health | Route::Models | Route::Reset) {
        if !matches!(
            singleton(parsed.headers, "content-length"),
            Ok(None | Some(b"0"))
        ) || crate::contains_header(parsed.headers, "content-type")
        {
            return Err(NativeBackendError::Configuration);
        }
        policy
            .check_common_head("POST", "/v1/chat/completions", 1, parsed.headers)
            .map_err(map_rejection)?;
        Ok((route, 0))
    } else {
        let authorized = policy
            .check_head("POST", "/v1/chat/completions", 1, parsed.headers)
            .map_err(map_rejection)?;
        Ok((route, authorized.length))
    }
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
    #[serde(borrow)]
    chat_template_kwargs: Object<FixedTemplate<'a>>,
    #[serde(default, deserialize_with = "optional_lora")]
    lora: Option<[Object<PinnedLora>; 1]>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FixedTemplate<'a> {
    enable_thinking: bool,
    #[serde(borrow)]
    reasoning_effort: Cow<'a, str>,
    preserve_thinking: bool,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PinnedLora {
    id: u8,
    scale: f64,
}

fn optional_lora<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<[Object<PinnedLora>; 1]>, D::Error> {
    <[Object<PinnedLora>; 1]>::deserialize(deserializer).map(Some)
}

#[derive(Serialize)]
struct PrivateBody<'a, 'b> {
    #[serde(flatten)]
    body: BackendChat<'a, 'b>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lora: Option<[PinnedLora; 1]>,
}

#[cfg(test)]
fn canonical(bytes: &[u8]) -> Result<PendingChat, NativeBackendError> {
    canonical_for_profile(bytes, &ModelProfile::default())
}

fn canonical_for_profile(
    bytes: &[u8],
    profile: &ModelProfile,
) -> Result<PendingChat, NativeBackendError> {
    if bytes.len() > MAX_BODY_BYTES {
        return Err(NativeBackendError::Schema);
    }
    let Object(body): Object<FixedBody<'_>> =
        serde_json::from_slice(bytes).map_err(|_| NativeBackendError::Schema)?;
    if match body.lora.as_ref() {
        Some([Object(lora)]) => {
            !profile.adapter
                || lora.id != 0
                || !lora.scale.is_finite()
                || !(0.0..=2.0).contains(&lora.scale)
        }
        None => profile.adapter,
    } {
        return Err(NativeBackendError::Schema);
    }
    if !crate::valid_history(&body.messages)
        || !(1..=MAX_COMPLETION_TOKENS).contains(&body.max_completion_tokens)
        || body.n != 1
        || body.cache_prompt
        || body.n_cache_reuse != 0
        || body.ignore_eos
        || body.top_k != 20
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
    let canonical = serde_json::to_vec(&PrivateBody {
        body: BackendChat {
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
                enable_thinking: body.chat_template_kwargs.0.enable_thinking,
                reasoning_effort: "xhigh",
                preserve_thinking: true,
            },
        },
        lora: body.lora.map(|[Object(lora)]| [lora]),
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
    deadline: Option<Instant>,
) -> Result<(), NativeBackendError> {
    let bytes = serde_json::to_vec(body).map_err(|_| NativeBackendError::Schema)?;
    let reason = if status == 200 {
        "OK"
    } else if status == 401 {
        "Unauthorized"
    } else if status == 413 {
        "Content Too Large"
    } else {
        "Bad Request"
    };
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n\r\n",
        bytes.len()
    );
    let write_deadline = Instant::now() + CLIENT_WRITE_TIMEOUT;
    let deadline = deadline.map_or(write_deadline, |limit| limit.min(write_deadline));
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

fn ensure_time(deadline: impl Into<Option<Instant>>) -> Result<(), NativeBackendError> {
    if deadline
        .into()
        .is_some_and(|deadline| Instant::now() >= deadline)
    {
        Err(NativeBackendError::Deadline)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests;
