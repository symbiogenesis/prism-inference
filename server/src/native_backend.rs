//! Private native HTTP child for an already verified, externally owned model.
//!
//! The launcher must keep this process inside its existing memory-limited Job,
//! watchdog and listener-PID ownership boundary. Load the Swift session before
//! binding the supplied exclusive numeric-loopback listener. This module starts
//! one joined model thread, never another process, and admits one request at a
//! time. Output has one event slot and requires a write acknowledgement before
//! the next decoder step. Failed generation closes model admission until reset.

mod diagnostics;
mod profile;
mod vulkan_session;
mod worker;

use crate::{
    BackendChat, CognitiveControlInput, MAX_BODY_BYTES, MAX_COMPLETION_TOKENS, MAX_HEADER_BYTES,
    MAX_HEADERS, Message, Object, PendingChat, Rejection, RequestPolicy,
    native_response::NativeResponse,
    reception::{HEADER_TIMEOUT, INTAKE_TIMEOUT},
    response::{CLIENT_WRITE_TIMEOUT, ResponseError},
    singleton, validate_cognitive_controls, wire,
};
use bonsai_prism_runtime::{runtime::Cancellation, swift_session::SwiftSession};

use profile::ModelProfile;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Value, json};
use std::{
    borrow::Cow,
    future::poll_fn,
    path::PathBuf,
    pin::pin,
    sync::atomic::{AtomicBool, Ordering},
};
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, split},
    net::TcpListener,
    time::{Instant, timeout_at},
};
#[cfg(windows)]
pub(crate) use vulkan_session::VulkanSwiftSession;
#[cfg(windows)]
pub(super) use worker::register_hexagon_model_path;
use worker::{Event, Input, Output, Worker};

/// Optional worker initialization controls supplied by the trusted private child CLI.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct WorkerOptions {
    /// Opt-in directory holding verified `"SB2HQ2C1"` Hexagon native Q2 arena caches.
    pub(crate) hexagon_arena_cache: Option<PathBuf>,
}

pub(crate) fn set_swift_prefill_row_parallelism(
    session: &mut SwiftSession,
    parallelism: bonsai_prism_runtime::kernels::RowParallelism,
) {
    worker::Engine::set_prefill_row_parallelism(session, parallelism);
}

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
    /// Require the Adreno Vulkan GPU projection executor with resident prewarmed weights.
    Vulkan,
    /// Require the strict Hexagon NPU prefix + Adreno Vulkan GPU overflow composite executor.
    Composite,
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
    Box::pin(serve_with_placement(
        listener,
        session,
        key,
        Placement::Auto,
    ))
    .await
}

/// Serve a verified session with one immutable trusted projection placement.
///
/// CPU placement clears any supplied projection executor before worker creation
/// and bypasses accelerator initialization on the model thread. Its readiness
/// declares both the requested placement and CPU projection backend. Automatic
/// placement retains the optional resident initialization policy. Both report
/// startup projection facts only after model-thread initialization completes.
/// Process, context and state memory allowances remain unchanged.
///
/// # Errors
/// Rejects invalid listener/key configuration, worker startup or accept failure.
pub async fn serve_with_placement(
    listener: TcpListener,
    session: SwiftSession,
    key: String,
    placement: Placement,
) -> Result<(), NativeBackendError> {
    Box::pin(serve_with_options(
        listener,
        session,
        key,
        placement,
        WorkerOptions::default(),
    ))
    .await
}

pub(crate) async fn serve_with_options(
    listener: TcpListener,
    session: SwiftSession,
    key: String,
    placement: Placement,
    options: WorkerOptions,
) -> Result<(), NativeBackendError> {
    Box::pin(serve_with_prefill_workers(
        listener, session, key, placement, options, None,
    ))
    .await
}

pub(crate) async fn serve_with_prefill_workers(
    listener: TcpListener,
    mut session: SwiftSession,
    key: String,
    placement: Placement,
    options: WorkerOptions,
    prefill_row_workers: Option<usize>,
) -> Result<(), NativeBackendError> {
    if placement == Placement::Cpu {
        session.set_projection_executor(None);
    }
    let seed = session.seed();
    serve_engine_with_options(
        listener,
        session,
        seed,
        key,
        placement,
        options,
        prefill_row_workers,
    )
    .await
}

#[cfg(windows)]
pub(crate) async fn serve_vulkan_with_prefill_workers(
    listener: TcpListener,
    session: VulkanSwiftSession,
    key: String,
    options: WorkerOptions,
    prefill_row_workers: Option<usize>,
) -> Result<(), NativeBackendError> {
    let seed = session.seed();
    serve_engine_with_options(
        listener,
        session,
        seed,
        key,
        Placement::Vulkan,
        options,
        prefill_row_workers,
    )
    .await
}

async fn serve_engine_with_options<E: worker::Engine>(
    listener: TcpListener,
    engine: E,
    seed: u32,
    key: String,
    placement: Placement,
    options: WorkerOptions,
    prefill_row_workers: Option<usize>,
) -> Result<(), NativeBackendError> {
    let address = listener
        .local_addr()
        .map_err(|_| NativeBackendError::Configuration)?;
    if !address.is_ipv4() || address.ip() != std::net::Ipv4Addr::LOCALHOST || address.port() < 1024
    {
        return Err(NativeBackendError::Configuration);
    }
    let base_policy =
        RequestPolicy::new(address.port(), &key).map_err(|_| NativeBackendError::Configuration)?;
    let mut worker =
        Worker::with_prefill_options(engine, seed, placement, &options, prefill_row_workers)?;
    let policy = base_policy.with_vision(worker.profile.vision);
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

pub(crate) async fn serve_broker_pipe_with_prefill_workers(
    endpoint: crate::broker_pipe::ServerEndpoint,
    mut session: SwiftSession,
    key: &str,
    placement: Placement,
    options: WorkerOptions,
    prefill_row_workers: Option<usize>,
) -> Result<(), NativeBackendError> {
    if placement == Placement::Cpu {
        session.set_projection_executor(None);
    }
    let seed = session.seed();
    serve_broker_pipe_engine_with_options(
        endpoint,
        session,
        seed,
        key,
        placement,
        options,
        prefill_row_workers,
    )
    .await
}

#[cfg(windows)]
pub(crate) async fn serve_broker_pipe_vulkan_with_prefill_workers(
    endpoint: crate::broker_pipe::ServerEndpoint,
    session: VulkanSwiftSession,
    key: &str,
    options: WorkerOptions,
    prefill_row_workers: Option<usize>,
) -> Result<(), NativeBackendError> {
    let seed = session.seed();
    serve_broker_pipe_engine_with_options(
        endpoint,
        session,
        seed,
        key,
        Placement::Vulkan,
        options,
        prefill_row_workers,
    )
    .await
}

async fn serve_broker_pipe_engine_with_options<E: worker::Engine>(
    endpoint: crate::broker_pipe::ServerEndpoint,
    engine: E,
    seed: u32,
    key: &str,
    placement: Placement,
    options: WorkerOptions,
    prefill_row_workers: Option<usize>,
) -> Result<(), NativeBackendError> {
    let base_policy = RequestPolicy::new(crate::broker_pipe::BROKER_PIPE_HOST_PORT, key)
        .map_err(|_| NativeBackendError::Configuration)?;
    let worker =
        Worker::with_prefill_options(engine, seed, placement, &options, prefill_row_workers)?;
    let policy = base_policy.with_vision(worker.profile.vision);
    serve_broker_pipe_with_worker(endpoint, &policy, worker).await
}

async fn serve_broker_pipe_with_worker(
    mut endpoint: crate::broker_pipe::ServerEndpoint,
    policy: &RequestPolicy,
    mut worker: Worker,
) -> Result<(), NativeBackendError> {
    endpoint.mark_ready().map_err(|_| NativeBackendError::Io)?;
    loop {
        match endpoint.accept().await {
            Ok(Some(client)) => {
                if let Err(error) = serve_connection(client, policy, &mut worker, None).await {
                    let _ = write_request_failure(&mut std::io::stderr().lock(), error);
                }
            }
            Ok(None) => return Ok(()),
            Err(_) => return Err(NativeBackendError::Io),
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

#[derive(Clone, Copy, Eq, PartialEq)]
enum Route {
    Health,
    Models,
    Count,
    Generate,
    Warm,
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
            return write_json(&mut client, 200, &worker.readiness(), deadline).await;
        }
        Route::Warm if !worker.is_idle() => {
            write_json(&mut client, 409, &json!({"error":"worker busy"}), deadline).await?;
            return Err(NativeBackendError::Engine);
        }
        _ => {}
    }
    let input = match received.route {
        Route::Count => Input::Count(received.pending.ok_or(NativeBackendError::Schema)?),
        Route::Generate => Input::Generate(received.pending.ok_or(NativeBackendError::Schema)?),
        Route::Warm => Input::Warm(received.pending.ok_or(NativeBackendError::Schema)?),
        Route::Reset => Input::Reset,
        Route::Health | Route::Models => return Err(NativeBackendError::Configuration),
    };
    let cancellation = Cancellation::default();
    let result = exchange(client, worker, input, &cancellation, deadline).await;
    if result.is_err() {
        cancellation.cancel();
    }
    // A following request cannot run until recovery has left the model thread.
    let settle_timeout = std::time::Duration::from_secs(10);
    if !worker.settled_within(settle_timeout).await {
        cancellation.cancel();
        return Err(NativeBackendError::Deadline);
    }
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
    let terminal_written = AtomicBool::new(false);
    let mut pipeline = pin!(deliver(
        output,
        events,
        &delivered,
        &terminal_written,
        deadline
    ));
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
        if !delivered.load(Ordering::Acquire)
            && !terminal_written.load(Ordering::Acquire)
            && abandoned.as_mut().poll(cx).is_ready()
        {
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
    terminal_written: &AtomicBool,
    deadline: Option<Instant>,
) -> Result<(), NativeBackendError> {
    let mut writer = Some(output);
    let mut response = None;
    while let Some(Event { output, ack }) = events.recv().await {
        match output {
            Output::Count(count) => {
                write_terminal_json(
                    writer.as_mut().ok_or(NativeBackendError::Engine)?,
                    200,
                    &json!({"input_tokens":count,"object":"response.input_tokens"}),
                    deadline,
                    Some(terminal_written),
                )
                .await?;
                delivered.store(true, Ordering::Release);
            }
            Output::Reset => {
                write_terminal_json(
                    writer.as_mut().ok_or(NativeBackendError::Engine)?,
                    200,
                    &json!({"status":"ok"}),
                    deadline,
                    Some(terminal_written),
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
            Output::Delta { text, confidence } => response
                .as_mut()
                .ok_or(NativeBackendError::Engine)?
                .delta_with_confidence(&text, confidence)
                .await
                .map_err(NativeBackendError::Response)?,
            Output::Complete(value) => {
                response
                    .take()
                    .ok_or(NativeBackendError::Engine)?
                    .complete_marking_terminal(value, terminal_written)
                    .await
                    .map_err(NativeBackendError::Response)?;
                delivered.store(true, Ordering::Release);
            }
            Output::Warmed { prompt } => {
                write_terminal_json(
                    writer.as_mut().ok_or(NativeBackendError::Engine)?,
                    200,
                    &json!({"status":"ok","warmed_tokens":prompt,"object":"chat.warm"}),
                    deadline,
                    Some(terminal_written),
                )
                .await?;
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
    let pending = if matches!(route, Route::Count | Route::Generate | Route::Warm) {
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
        (Some("POST"), Some("/v1/chat/warm")) => Route::Warm,
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
    top_k: u32,
    min_p: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    repeat_penalty: f32,
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
    #[serde(borrow)]
    chat_template_kwargs: Object<FixedTemplate<'a>>,
    #[serde(default, deserialize_with = "optional_lora")]
    lora: Option<Vec<Object<PinnedLora>>>,
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
) -> Result<Option<Vec<Object<PinnedLora>>>, D::Error> {
    Vec::<Object<PinnedLora>>::deserialize(deserializer).map(Some)
}

#[derive(Serialize)]
struct PrivateBody<'a, 'b> {
    #[serde(flatten)]
    body: BackendChat<'a, 'b>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lora: Option<Vec<PinnedLora>>,
}

fn validate_canonical_lora(
    raw: Option<Vec<Object<PinnedLora>>>,
    profile: &ModelProfile,
) -> Result<Option<Vec<PinnedLora>>, NativeBackendError> {
    let entries: Option<Vec<PinnedLora>> =
        raw.map(|items| items.into_iter().map(|Object(item)| item).collect());
    if let Some(items) = &entries {
        if !(1..=2).contains(&items.len()) {
            return Err(NativeBackendError::Schema);
        }
        for item in items {
            if !item.scale.is_finite() || !(0.0..=2.0).contains(&item.scale) {
                return Err(NativeBackendError::Schema);
            }
        }
    }
    match (profile.daisyrose_adapter, entries) {
        (false | true, None) => Ok(None),
        (true, Some(items)) if items.len() == 1 && items[0].id == 0 => Ok(Some(items)),
        _ => Err(NativeBackendError::Schema),
    }
}

#[cfg(test)]
fn canonical(bytes: &[u8]) -> Result<PendingChat, NativeBackendError> {
    canonical_for_profile(bytes, &ModelProfile::default())
}

fn canonical_for_profile(
    bytes: &[u8],
    profile: &ModelProfile,
) -> Result<PendingChat, NativeBackendError> {
    let max_body_bytes = if profile.vision {
        crate::MAX_MULTIMODAL_BODY_BYTES
    } else {
        MAX_BODY_BYTES
    };
    if bytes.len() > max_body_bytes {
        return Err(NativeBackendError::Schema);
    }
    let Object(body): Object<FixedBody<'_>> =
        serde_json::from_slice(bytes).map_err(|_| NativeBackendError::Schema)?;
    let lora = validate_canonical_lora(body.lora, profile)?;
    if !crate::valid_history_for_vision(&body.messages, profile.vision)
        || !(1..=MAX_COMPLETION_TOKENS).contains(&body.max_completion_tokens)
        || body.n != 1
        || body.cache_prompt
        || body.n_cache_reuse != 0
        || body.ignore_eos
    {
        return Err(NativeBackendError::Schema);
    }
    let controls = validate_cognitive_controls(CognitiveControlInput {
        temperature: body.temperature,
        top_p: body.top_p,
        top_k: body.top_k,
        min_p: body.min_p,
        presence_penalty: body.presence_penalty,
        frequency_penalty: body.frequency_penalty,
        repeat_penalty: body.repeat_penalty,
        enable_thinking: body.chat_template_kwargs.0.enable_thinking,
        reasoning_effort: body.chat_template_kwargs.0.reasoning_effort.as_ref(),
        preserve_thinking: body.chat_template_kwargs.0.preserve_thinking,
        reasoning_budget: body.reasoning_budget,
        dry_multiplier: body.dry_multiplier,
        dry_base: body.dry_base,
        dry_allowed_length: body.dry_allowed_length,
        dry_penalty_last_n: body.dry_penalty_last_n,
        logprobs: body.logprobs,
        top_logprobs: body.top_logprobs,
    })
    .ok_or(NativeBackendError::Schema)?;
    let canonical = serde_json::to_vec(&PrivateBody {
        body: controls.to_backend_chat(&body.messages, body.max_completion_tokens, body.stream),
        lora,
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
    write_terminal_json(client, status, body, deadline, None).await
}

async fn write_terminal_json<W: AsyncWrite + Unpin>(
    client: &mut W,
    status: u16,
    body: &Value,
    deadline: Option<Instant>,
    terminal_written: Option<&AtomicBool>,
) -> Result<(), NativeBackendError> {
    let bytes = serde_json::to_vec(body).map_err(|_| NativeBackendError::Schema)?;
    let reason = if status == 200 {
        "OK"
    } else if status == 401 {
        "Unauthorized"
    } else if status == 409 {
        "Conflict"
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
        if let Some(flag) = terminal_written {
            flag.store(true, Ordering::Release);
        }
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

#[cfg(test)]
mod warm_tests {
    use super::*;
    use crate::{MAX_MULTIMODAL_BODY_BYTES, native_backend::worker::Engine};
    use bonsai_prism_runtime::{
        runtime::{Completion, Finish},
        swift_session::TextEvent,
    };
    use std::{
        sync::{Arc, atomic::AtomicUsize},
        task::Poll,
        time::Duration,
    };
    use tokio::io::duplex;

    type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;
    const KEY: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    struct WarmableEngine {
        warmable: bool,
        hold_until_cancel: bool,
        warm_calls: Arc<AtomicUsize>,
        complete_calls: Arc<AtomicUsize>,
        row_workers: usize,
    }

    impl Engine for WarmableEngine {
        type Prepared = PendingChat;

        fn profile(&self) -> Result<ModelProfile, NativeBackendError> {
            Ok(ModelProfile {
                row_workers: self.row_workers,
                ..ModelProfile::default()
            })
        }

        fn prepare(&self, pending: PendingChat) -> Result<Self::Prepared, NativeBackendError> {
            Ok(pending)
        }

        fn properties(prepared: &Self::Prepared) -> (usize, bool, bool, &[u8]) {
            (42, false, false, &prepared.body)
        }

        fn supports_warm(&self) -> bool {
            self.warmable
        }

        fn warm_prefix(
            &mut self,
            _prepared: &Self::Prepared,
            cancel: &Cancellation,
            _deadline: Option<std::time::Instant>,
        ) -> Result<usize, NativeBackendError> {
            self.warm_calls.fetch_add(1, Ordering::Relaxed);
            if self.hold_until_cancel {
                while !cancel.is_cancelled() {
                    std::thread::sleep(Duration::from_millis(1));
                }
                return Err(NativeBackendError::ClientAbandoned);
            }
            Ok(42)
        }

        fn complete(
            &mut self,
            _prepared: &Self::Prepared,
            _cancel: &Cancellation,
            _deadline: Option<std::time::Instant>,
            _emit: impl FnMut(TextEvent<'_>) -> bonsai_prism_runtime::Result<()>,
        ) -> bonsai_prism_runtime::Result<Completion> {
            self.complete_calls.fetch_add(1, Ordering::Relaxed);
            Ok(Completion {
                prompt_tokens: 42,
                reused_tokens: 42,
                generated_tokens: 0,
                finish: Finish::Length,
            })
        }

        fn reset(&mut self) -> bonsai_prism_runtime::Result<()> {
            Ok(())
        }
    }

    fn http_post(
        path: &str,
        body: &[u8],
        authenticated: bool,
        length_override: Option<usize>,
    ) -> Vec<u8> {
        let mut head =
            format!("POST {path} HTTP/1.1\r\nHost: 127.0.0.1:8080\r\nConnection: close\r\n");
        if authenticated {
            head.push_str("Authorization: Bearer ");
            head.push_str(KEY);
            head.push_str("\r\n");
        }
        head.push_str("Content-Type: application/json\r\nContent-Length: ");
        head.push_str(&length_override.unwrap_or(body.len()).to_string());
        head.push_str("\r\n\r\n");
        let mut raw = head.into_bytes();
        raw.extend_from_slice(body);
        raw
    }

    async fn exchange_once(
        worker: &mut Worker,
        raw_request: Vec<u8>,
    ) -> Result<(Result<(), NativeBackendError>, Vec<u8>), std::io::Error> {
        let policy = RequestPolicy::new(8080, KEY).map_err(|_| std::io::Error::other("policy"))?;
        let (server, mut client) = duplex(16 * 1024);
        let mut serve_fut = pin!(serve_connection(server, &policy, worker, None));
        let mut client_fut = pin!(async {
            client.write_all(&raw_request).await?;
            client.flush().await?;
            let mut reply = Vec::new();
            client.read_to_end(&mut reply).await?;
            Ok::<_, std::io::Error>(reply)
        });
        let mut a = None;
        let mut b = None;
        poll_fn(|cx| {
            if a.is_none()
                && let Poll::Ready(res) = serve_fut.as_mut().poll(cx)
            {
                a = Some(res);
            }
            if b.is_none()
                && let Poll::Ready(res) = client_fut.as_mut().poll(cx)
            {
                b = Some(res);
            }
            match (a, b.is_some()) {
                (Some(res), true) => {
                    let reply = b.take().unwrap_or_else(|| Ok(Vec::new()))?;
                    Poll::Ready(Ok((res, reply)))
                }
                _ => Poll::Pending,
            }
        })
        .await
    }

    #[test]
    fn post_warm_enforces_auth_bounds_idle_warm_success_and_cancellation() -> TestResult {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()?;
        runtime.block_on(async {
            let warm_calls = Arc::new(AtomicUsize::new(0));
            let complete_calls = Arc::new(AtomicUsize::new(0));
            let raw_req = serde_json::to_vec(&json!({
                "messages": [{"role": "user", "content": "Hello warm cache"}],
                "max_completion_tokens": 16,
                "stream": false
            }))?;
            let valid_body = crate::AuthorizedHead {
                length: raw_req.len(),
                vision: false,
            }
            .prepare(&raw_req)
            .map_err(|e| format!("{e:?}"))?
            .body;

            let mut worker = Worker::with_prefill_options(
                WarmableEngine {
                    warmable: true,
                    hold_until_cancel: false,
                    warm_calls: warm_calls.clone(),
                    complete_calls: complete_calls.clone(),
                    row_workers: 4,
                },
                42,
                Placement::Cpu,
                &WorkerOptions::default(),
                Some(12),
            )
            .map_err(|e| format!("{e:?}"))?;

            let (res, reply) = exchange_once(
                &mut worker,
                http_post("/v1/chat/warm", &valid_body, false, None),
            )
            .await?;
            assert_eq!(res, Err(NativeBackendError::Unauthorized));
            assert!(reply.starts_with(b"HTTP/1.1 401 "));
            assert_eq!(warm_calls.load(Ordering::Relaxed), 0);

            let (res, reply) = exchange_once(
                &mut worker,
                http_post(
                    "/v1/chat/warm",
                    &valid_body,
                    true,
                    Some(MAX_MULTIMODAL_BODY_BYTES + 1),
                ),
            )
            .await?;
            assert_eq!(res, Err(NativeBackendError::Configuration));
            assert!(reply.starts_with(b"HTTP/1.1 400 "));
            assert_eq!(warm_calls.load(Ordering::Relaxed), 0);

            let (res, reply) = exchange_once(
                &mut worker,
                http_post("/v1/chat/warm", &valid_body, true, None),
            )
            .await?;
            assert_eq!(res, Ok(()));
            assert!(reply.starts_with(b"HTTP/1.1 200 "));
            let end = reply
                .array_windows()
                .position(|w| w == b"\r\n\r\n")
                .ok_or("missing response headers")?
                + 4;
            let body: Value = serde_json::from_slice(&reply[end..])?;
            assert_eq!(
                body,
                json!({"status": "ok", "warmed_tokens": 42, "object": "chat.warm"})
            );
            assert_eq!(warm_calls.load(Ordering::Relaxed), 1);
            assert_eq!(complete_calls.load(Ordering::Relaxed), 0);

            let mut worker_cancel = Worker::with_options(
                WarmableEngine {
                    warmable: true,
                    hold_until_cancel: true,
                    warm_calls,
                    complete_calls,
                    row_workers: 1,
                },
                42,
                Placement::Cpu,
                &WorkerOptions::default(),
            )
            .map_err(|e| format!("{e:?}"))?;
            let policy =
                RequestPolicy::new(8080, KEY).map_err(|_| std::io::Error::other("policy"))?;
            let (server, mut client) = duplex(16 * 1024);
            client
                .write_all(&http_post("/v1/chat/warm", &valid_body, true, None))
                .await?;
            client.flush().await?;
            drop(client);
            let res = serve_connection(server, &policy, &mut worker_cancel, None).await;
            assert!(res.is_err());
            assert!(worker_cancel.settled_within(Duration::from_secs(2)).await);
            Ok(())
        })
    }
}
