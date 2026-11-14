//! Focused private HTTP and acknowledged worker lifecycle contracts.

use super::*;
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

fn pending(stream: bool, max_completion_tokens: u32) -> Result<PendingChat, NativeBackendError> {
    let body = serde_json::to_vec(&json!({"messages":[{"role":"user","content":"A scene."}],"stream":stream,"max_completion_tokens":max_completion_tokens})).map_err(|_|NativeBackendError::Schema)?;
    crate::AuthorizedHead { length: body.len() }
        .prepare(&body)
        .map_err(NativeBackendError::Admission)
}

fn request(route: &str, body: &[u8], authenticated: bool) -> Vec<u8> {
    let method = if route.starts_with("/v1/chat/") || route == "/reset" {
        "POST"
    } else {
        "GET"
    };
    let mut head =
        format!("{method} {route} HTTP/1.1\r\nHost: 127.0.0.1:8080\r\nConnection: close\r\n");
    if authenticated {
        head.push_str("Authorization: Bearer ");
        head.push_str(KEY);
        head.push_str("\r\n");
    }
    if !body.is_empty() {
        head.push_str("Content-Type: application/json\r\nContent-Length: ");
        head.push_str(&body.len().to_string());
        head.push_str("\r\n");
    }
    head.push_str("\r\n");
    let mut bytes = head.into_bytes();
    bytes.extend_from_slice(body);
    bytes
}

fn run(test: impl Future<Output = TestResult>) -> TestResult {
    tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()?
        .block_on(test)
}

async fn together<A: Future, B: Future>(first: A, second: B) -> (A::Output, B::Output) {
    let mut first = pin!(first);
    let mut second = pin!(second);
    let mut a = None;
    let mut b = None;
    poll_fn(|cx| {
        if a.is_none()
            && let Poll::Ready(value) = first.as_mut().poll(cx)
        {
            a = Some(value);
        }
        if b.is_none()
            && let Poll::Ready(value) = second.as_mut().poll(cx)
        {
            b = Some(value);
        }
        match (a.take(), b.take()) {
            (Some(left), Some(right)) => Poll::Ready((left, right)),
            (left, right) => {
                a = left;
                b = right;
                Poll::Pending
            }
        }
    })
    .await
}

struct FakePrepared {
    admitted: crate::AdmittedChat,
    thinking: bool,
}
#[derive(Default, Clone)]
struct Observation {
    worker_initializations: Arc<AtomicUsize>,
    preparations: Arc<AtomicUsize>,
    executions: Arc<AtomicUsize>,
    state: Arc<AtomicUsize>,
    resets: Arc<AtomicUsize>,
    terminated: Arc<AtomicBool>,
    row_workers: Option<usize>,
    prompt_reuse: bonsai_prism_runtime::runtime::PromptReuse,
    kv_cache: Option<bonsai_prism_runtime::graph::KvCachePolicy>,
    cpu_prefill: Option<bonsai_prism_runtime::swift_session::CpuPrefill>,
    cpu_speculation: Option<bonsai_prism_runtime::swift_session::CpuSpeculation>,
    startup_graph_identity: Option<Arc<str>>,
    preparation_rejection: Option<(usize, Rejection)>,
    generation_gate: Option<Arc<AtomicBool>>,
    finite_deadlines: Arc<AtomicUsize>,
    generation_cancelled: Arc<AtomicBool>,
    initialization_gate: Option<Arc<AtomicBool>>,
    initialization_failure: Option<NativeBackendError>,
    startup_projection: Option<profile::ProjectionStartup>,
}
struct FakeEngine(Observation);

impl Drop for FakeEngine {
    fn drop(&mut self) {
        self.0.terminated.store(true, Ordering::Release);
    }
}

impl worker::Engine for FakeEngine {
    type Prepared = FakePrepared;
    fn initialize_worker_thread(
        &mut self,
    ) -> Result<profile::ProjectionStartup, NativeBackendError> {
        self.0
            .worker_initializations
            .fetch_add(1, Ordering::Relaxed);
        while self
            .0
            .initialization_gate
            .as_ref()
            .is_some_and(|gate| !gate.load(Ordering::Acquire))
        {
            std::thread::sleep(Duration::from_millis(1));
        }
        if let Some(error) = self.0.initialization_failure {
            return Err(error);
        }
        Ok(self.0.startup_projection.clone().unwrap_or_default())
    }
    fn profile(&self) -> Result<ModelProfile, NativeBackendError> {
        Ok(ModelProfile {
            row_workers: self.0.row_workers.unwrap_or(1),
            prompt_reuse: self.0.prompt_reuse,
            kv_cache: self
                .0
                .kv_cache
                .unwrap_or(bonsai_prism_runtime::graph::KvCachePolicy::F16),
            startup_graph_identity: self.0.startup_graph_identity.clone(),
            cpu_prefill: self.0.cpu_prefill,
            cpu_speculation: self.0.cpu_speculation,
            ..ModelProfile::default()
        })
    }
    fn prepare(&self, pending: PendingChat) -> Result<Self::Prepared, NativeBackendError> {
        let preparation = self.0.preparations.fetch_add(1, Ordering::Relaxed);
        if let Some((index, error)) = self.0.preparation_rejection
            && preparation == index
        {
            return Err(NativeBackendError::Admission(error));
        }
        let value: Value =
            serde_json::from_slice(&pending.body).map_err(|_| NativeBackendError::Schema)?;
        let thinking = value["chat_template_kwargs"]["enable_thinking"]
            .as_bool()
            .ok_or(NativeBackendError::Schema)?;
        Ok(FakePrepared {
            thinking,
            admitted: pending
                .admit(|_| Ok(100))
                .map_err(NativeBackendError::Admission)?,
        })
    }
    fn properties(value: &Self::Prepared) -> (usize, bool, bool, &[u8]) {
        (
            100,
            value.admitted.streams_response(),
            value.thinking,
            value.admitted.backend_body(),
        )
    }
    fn complete(
        &mut self,
        _prepared: &Self::Prepared,
        cancel: &Cancellation,
        deadline: Option<std::time::Instant>,
        mut emit: impl FnMut(TextEvent<'_>) -> bonsai_prism_runtime::Result<()>,
    ) -> bonsai_prism_runtime::Result<Completion> {
        self.0
            .finite_deadlines
            .fetch_add(usize::from(deadline.is_some()), Ordering::Relaxed);
        self.0.executions.fetch_add(1, Ordering::Release);
        let original = self.0.state.fetch_add(1, Ordering::Relaxed);
        let completion = Completion {
            prompt_tokens: 100,
            reused_tokens: 0,
            generated_tokens: 2,
            finish: Finish::Length,
        };
        let result = (|| {
            while self
                .0
                .generation_gate
                .as_ref()
                .is_some_and(|gate| !gate.load(Ordering::Acquire))
            {
                if cancel.is_cancelled() {
                    self.0.generation_cancelled.store(true, Ordering::Release);
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::Interrupted,
                        "fake generation cancelled",
                    )
                    .into());
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            emit(TextEvent::Delta("draft"))?;
            emit(TextEvent::Delta("</think>answer"))?;
            emit(TextEvent::Complete(completion))?;
            Ok(completion)
        })();
        if result.is_err() {
            self.0.state.store(original, Ordering::Relaxed);
        }
        result
    }
    fn reset(&mut self) -> bonsai_prism_runtime::Result<()> {
        self.0.state.store(0, Ordering::Relaxed);
        self.0.resets.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

#[test]
fn cpu_placement_skips_accelerator_initializer_and_binds_readiness() -> TestResult {
    for placement in [Placement::Auto, Placement::Cpu] {
        let observed = Observation::default();
        let worker = Worker::with_placement(FakeEngine(observed.clone()), 42, placement)
            .map_err(|error| format!("{error:?}"))?;
        let readiness = worker.profile.readiness(worker.seed);
        let model = &readiness["data"][0];
        if placement == Placement::Cpu {
            assert_eq!(model["placement"], "cpu");
            assert_eq!(model["projection_backend"], "cpu");
        } else {
            assert!(model.get("placement").is_none());
            assert!(model.get("projection_backend").is_none());
        }
        assert_eq!(
            model["startup_projection"],
            json!({
                "schema_version": 1,
                "requested": if placement == Placement::Cpu { "cpu" } else { "auto" },
                "effective": "cpu", "identity": null,
                "resident_tensors": 0, "resident_weight_bytes": 0, "host_metadata_bytes": 0,
            })
        );
        drop(worker);
        assert!(observed.terminated.load(Ordering::Acquire));
        assert_eq!(
            observed.worker_initializations.load(Ordering::Relaxed),
            usize::from(placement == Placement::Auto)
        );
    }
    Ok(())
}

#[test]
fn projection_startup_waits_for_owner_and_binds_resident_observations() -> TestResult {
    use profile::ProjectionStartup;
    let gate = Arc::new(AtomicBool::new(false));
    let observed = Observation {
        initialization_gate: Some(gate.clone()),
        startup_projection: Some(ProjectionStartup::NativeQ2 {
            resident_tensors: 168,
            resident_weight_bytes: 2_888_663_040,
            host_metadata_bytes: 524_128,
        }),
        ..Observation::default()
    };
    let engine = FakeEngine(observed.clone());
    let (completed, receive) = std::sync::mpsc::channel();
    let constructor = std::thread::spawn(move || {
        let result = Worker::with_placement(engine, 42, Placement::Auto);
        let _ = completed.send(result);
    });
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while observed.worker_initializations.load(Ordering::Acquire) == 0
        && std::time::Instant::now() < deadline
    {
        std::thread::yield_now();
    }
    let initialized = observed.worker_initializations.load(Ordering::Acquire);
    let premature = receive.try_recv();
    gate.store(true, Ordering::Release);
    assert_eq!(initialized, 1);
    assert!(matches!(
        premature,
        Err(std::sync::mpsc::TryRecvError::Empty)
    ));
    let worker = receive
        .recv_timeout(Duration::from_secs(2))?
        .map_err(|error| format!("{error:?}"))?;
    constructor.join().map_err(|_| "constructor panicked")?;
    let readiness = worker.profile.readiness(worker.seed);
    assert_eq!(
        readiness["data"][0]["startup_projection"],
        json!({
            "schema_version": 1, "requested": "auto", "effective": "hexagon-native-q2",
            "identity": "hexagon-hvx-native-q2-resident-v1-experimental",
            "resident_tensors": 168, "resident_weight_bytes": 2_888_663_040,
            "host_metadata_bytes": 524_128,
        })
    );
    drop(worker);
    assert!(observed.terminated.load(Ordering::Acquire));
    Ok(())
}

#[test]
fn projection_startup_rejects_failure_or_inconsistent_observations_and_joins_owner() {
    use profile::ProjectionStartup;
    for observation in [
        Observation {
            initialization_failure: Some(NativeBackendError::Engine),
            ..Observation::default()
        },
        Observation {
            startup_projection: Some(ProjectionStartup::NativeQ2 {
                resident_tensors: 167,
                resident_weight_bytes: 2_888_663_040,
                host_metadata_bytes: 1,
            }),
            ..Observation::default()
        },
        Observation {
            startup_projection: Some(ProjectionStartup::External(Arc::from(""))),
            ..Observation::default()
        },
        Observation {
            startup_projection: Some(ProjectionStartup::External(Arc::from("bad identity"))),
            ..Observation::default()
        },
    ] {
        assert!(
            Worker::with_placement(FakeEngine(observation.clone()), 42, Placement::Auto).is_err()
        );
        assert!(observation.terminated.load(Ordering::Acquire));
    }
}

#[test]
fn projection_startup_preserves_external_identity_and_cpu_bypasses_failed_initializer() -> TestResult
{
    use profile::ProjectionStartup;
    let observed = Observation {
        startup_projection: Some(ProjectionStartup::External(Arc::from(
            "caller-projection-v1",
        ))),
        ..Observation::default()
    };
    let worker =
        Worker::new(FakeEngine(observed.clone()), 42).map_err(|error| format!("{error:?}"))?;
    let readiness = worker.profile.readiness(worker.seed);
    assert_eq!(
        readiness["data"][0]["startup_projection"]["effective"],
        "external"
    );
    assert_eq!(
        readiness["data"][0]["startup_projection"]["identity"],
        "caller-projection-v1"
    );
    assert_eq!(
        readiness["data"][0]["startup_projection"]["resident_weight_bytes"],
        0
    );
    drop(worker);
    let observed = Observation {
        initialization_failure: Some(NativeBackendError::Engine),
        ..Observation::default()
    };
    let worker = Worker::with_placement(FakeEngine(observed.clone()), 42, Placement::Cpu)
        .map_err(|error| format!("{error:?}"))?;
    assert_eq!(worker.profile.projection_startup, ProjectionStartup::Cpu);
    assert_eq!(observed.worker_initializations.load(Ordering::Acquire), 0);
    drop(worker);
    Ok(())
}

async fn perform(
    worker: &mut Worker,
    bytes: Vec<u8>,
) -> Result<(Result<(), NativeBackendError>, Vec<u8>), Box<dyn std::error::Error + Send + Sync>> {
    let (mut client, server) = duplex(128);
    let policy = RequestPolicy::new(8080, KEY).map_err(|e| format!("{e:?}"))?;
    let serving = serve_connection(
        server,
        &policy,
        worker,
        Instant::now() + Duration::from_secs(10),
    );
    let receive = async {
        client.write_all(&bytes).await?;
        let mut output = Vec::new();
        client.read_to_end(&mut output).await?;
        Ok::<_, std::io::Error>(output)
    };
    let (result, output) = together(serving, receive).await;
    Ok((result, output?))
}

#[test]
fn private_typed_context_rejection_clears_count_seal_without_faulting_owner() -> TestResult {
    run(async {
        for rejection in [
            Rejection::ContextLimit,
            Rejection::TokenCountUnavailable,
            Rejection::ChatSchema,
        ] {
            let observed = Observation {
                preparation_rejection: Some((1, rejection)),
                ..Observation::default()
            };
            let mut worker = Worker::new(FakeEngine(observed.clone()), 42)
                .map_err(|error| format!("{error:?}"))?;
            let body = pending(true, 128)
                .map_err(|error| format!("{error:?}"))?
                .body;
            let (result, _) = perform(
                &mut worker,
                request("/v1/chat/completions/input_tokens", &body, true),
            )
            .await?;
            assert_eq!(result, Ok(()));
            let (result, _) = perform(
                &mut worker,
                // Authentication rejects before reading a body. This helper
                // sends before reading through a small duplex, so retain a
                // header-only negative control rather than requiring a denied
                // peer to consume a complete canonical request body.
                request("/v1/chat/completions/input_tokens", b"", false),
            )
            .await?;
            assert_eq!(result, Err(NativeBackendError::Unauthorized));
            assert_eq!(observed.preparations.load(Ordering::Relaxed), 1);
            let (result, reply) = perform(
                &mut worker,
                request("/v1/chat/completions/input_tokens", &body, true),
            )
            .await?;
            assert_eq!(result, Err(NativeBackendError::Admission(rejection)));
            let boundary = reply
                .array_windows()
                .position(|part| part == b"\r\n\r\n")
                .ok_or("missing HTTP framing")?;
            let head = std::str::from_utf8(&reply[..boundary])?;
            let payload = &reply[boundary + 4..];
            let context = rejection == Rejection::ContextLimit;
            assert_eq!(
                head.lines().next(),
                Some(if context {
                    "HTTP/1.1 413 Content Too Large"
                } else {
                    "HTTP/1.1 400 Bad Request"
                })
            );
            for header in [
                "Content-Type: application/json",
                "Connection: close",
                "Cache-Control: no-store",
                "X-Content-Type-Options: nosniff",
            ] {
                assert!(head.lines().any(|line| line == header));
            }
            let length = head
                .lines()
                .find_map(|line| line.strip_prefix("Content-Length: "))
                .ok_or("missing content length")?
                .parse::<usize>()?;
            assert_eq!(length, payload.len());
            assert_eq!(
                serde_json::from_slice::<Value>(payload)?,
                if context {
                    json!({"error":{"code":"context_length_exceeded"}})
                } else {
                    json!({"error":"native request rejected"})
                }
            );
            assert!(!std::str::from_utf8(&reply)?.contains("A scene."));
            assert_eq!(observed.executions.load(Ordering::Relaxed), 0);
            let (result, _) =
                perform(&mut worker, request("/v1/chat/completions", &body, true)).await?;
            assert_eq!(result, Err(NativeBackendError::CountMismatch));
            assert_eq!(observed.executions.load(Ordering::Relaxed), 0);
            let (result, _) = perform(
                &mut worker,
                request("/v1/chat/completions/input_tokens", &body, true),
            )
            .await?;
            assert_eq!(result, Ok(()));
            let (result, _) =
                perform(&mut worker, request("/v1/chat/completions", &body, true)).await?;
            assert_eq!(result, Ok(()));
            assert_eq!(observed.executions.load(Ordering::Relaxed), 1);
            assert_eq!(observed.resets.load(Ordering::Relaxed), 0);
        }
        Ok(())
    })
}

#[test]
fn private_canonical_body_rejects_overrides_duplicates_and_unknown_fields() -> TestResult {
    let original = pending(true, 128).map_err(|e| format!("{e:?}"))?.body;
    assert_eq!(
        canonical(&original).map_err(|e| format!("{e:?}"))?.body,
        original
    );
    for (name, value) in [
        ("temperature", json!(0.5)),
        ("top_p", json!(0.5)),
        ("top_k", json!(40)),
        ("min_p", json!(0.1)),
        ("repeat_penalty", json!(1.1)),
        ("presence_penalty", json!(1.0)),
        ("frequency_penalty", json!(1.0)),
        ("cache_prompt", json!(true)),
        ("n_cache_reuse", json!(1)),
        ("ignore_eos", json!(true)),
        ("n", json!(2)),
        ("tools", json!([])),
        ("max_completion_tokens", json!(4097)),
    ] {
        let mut value_body: Value = serde_json::from_slice(&original)?;
        value_body[name] = value;
        assert!(
            canonical(&serde_json::to_vec(&value_body)?).is_err(),
            "field={name}"
        );
    }
    for (name, value) in [
        ("enable_thinking", json!(null)),
        ("enable_thinking", json!("false")),
        ("enable_thinking", json!(0)),
        ("reasoning_effort", json!("low")),
        ("preserve_thinking", json!(false)),
        ("other", json!(true)),
    ] {
        let mut value_body: Value = serde_json::from_slice(&original)?;
        value_body["chat_template_kwargs"][name] = value;
        assert!(
            canonical(&serde_json::to_vec(&value_body)?).is_err(),
            "template={name}"
        );
    }
    let mut positional: Value = serde_json::from_slice(&original)?;
    positional["chat_template_kwargs"] = json!([true, "xhigh", true]);
    assert!(canonical(&serde_json::to_vec(&positional)?).is_err());
    let text = std::str::from_utf8(&original)?;
    for changed in [
        text.replacen(
            "\"temperature\":1.0",
            "\"temperature\":1.0,\"temperature\":1.0",
            1,
        ),
        text.replacen(
            "\"enable_thinking\":true",
            "\"enable_thinking\":true,\"enable_thinking\":true",
            1,
        ),
        text.replacen(
            "\"role\":\"user\"",
            "\"role\":\"user\",\"role\":\"user\"",
            1,
        ),
    ] {
        assert_ne!(changed, text);
        assert!(canonical(changed.as_bytes()).is_err());
    }
    Ok(())
}

#[test]
fn private_adapter_scale_requires_selected_profile_and_exact_count_binding() -> TestResult {
    run(async {
        let profile = ModelProfile {
            representation: bonsai_prism_runtime::model::swift::Representation::CurrentRepackedPtq1,
            adapter: true,
            ..ModelProfile::default()
        };
        let original = pending(true, 128).map_err(|e| format!("{e:?}"))?.body;
        let mut value: Value = serde_json::from_slice(&original)?;
        assert!(canonical_for_profile(&original, &profile).is_err());
        for invalid in [
            Value::Null,
            json!([]),
            json!([{"id":1,"scale":1.3}]),
            json!([{"id":0,"scale":"1.3"}]),
            json!([{"id":0,"scale":null}]),
            json!([{"id":0,"scale":-0.01}]),
            json!([{"id":0,"scale":2.000_000_01}]),
            json!([{"id":0,"scale":1.3,"path":"other.gguf"}]),
            json!([{"id":0,"scale":1.3},{"id":0,"scale":1.3}]),
        ] {
            value["lora"] = invalid;
            assert!(canonical_for_profile(&serde_json::to_vec(&value)?, &profile).is_err());
        }
        for scale in [0.0_f64, -0.0, 1.3, 2.0] {
            value["lora"] = json!([{"id":0,"scale":scale}]);
            let bytes = serde_json::to_vec(&value)?;
            assert!(canonical(&bytes).is_err());
            let prepared = canonical_for_profile(&bytes, &profile).map_err(|e| format!("{e:?}"))?;
            let canonical: Value = serde_json::from_slice(&prepared.body)?;
            assert_eq!(
                serde_json::from_value::<f64>(canonical["lora"][0]["scale"].clone())?.to_bits(),
                scale.to_bits()
            );
        }
        value["lora"] = json!([{"id":0,"scale":1.3}]);
        value["abliteration_scale"] = json!(1.3);
        assert!(canonical_for_profile(&serde_json::to_vec(&value)?, &profile).is_err());
        value
            .as_object_mut()
            .ok_or("Missing request object.")?
            .remove("abliteration_scale");
        let body = canonical_for_profile(&serde_json::to_vec(&value)?, &profile)
            .map_err(|e| format!("{e:?}"))?
            .body;
        let duplicate =
            std::str::from_utf8(&body)?.replacen("\"scale\":", "\"scale\":1.3,\"scale\":", 1);
        assert!(canonical_for_profile(duplicate.as_bytes(), &profile).is_err());
        let observed = Observation::default();
        let mut worker =
            Worker::new(FakeEngine(observed.clone()), 42).map_err(|e| format!("{e:?}"))?;
        worker.profile = profile.clone();
        let (result, _) = perform(
            &mut worker,
            request("/v1/chat/completions/input_tokens", &body, true),
        )
        .await?;
        assert_eq!(result, Ok(()));
        value["lora"] = json!([{"id":0,"scale":0.0}]);
        let changed = canonical_for_profile(&serde_json::to_vec(&value)?, &profile)
            .map_err(|e| format!("{e:?}"))?
            .body;
        let (result, _) =
            perform(&mut worker, request("/v1/chat/completions", &changed, true)).await?;
        assert_eq!(result, Err(NativeBackendError::CountMismatch));
        assert_eq!(observed.executions.load(Ordering::Relaxed), 0);
        let (result, _) = perform(
            &mut worker,
            request("/v1/chat/completions/input_tokens", &changed, true),
        )
        .await?;
        assert_eq!(result, Ok(()));
        let (result, _) = perform(&mut worker, request("/reset", &[], true)).await?;
        assert_eq!(result, Ok(()));
        let (result, _) =
            perform(&mut worker, request("/v1/chat/completions", &changed, true)).await?;
        assert_eq!(result, Err(NativeBackendError::CountMismatch));
        for bytes in [&body, &changed] {
            let (result, _) = perform(
                &mut worker,
                request("/v1/chat/completions/input_tokens", bytes, true),
            )
            .await?;
            assert_eq!(result, Ok(()));
            let (result, _) =
                perform(&mut worker, request("/v1/chat/completions", bytes, true)).await?;
            assert_eq!(result, Ok(()));
        }
        assert_eq!(observed.executions.load(Ordering::Relaxed), 2);
        Ok(())
    })
}

#[test]
fn private_context_capacity_controls_admission_and_readiness() -> Result<(), NativeBackendError> {
    let profile = ModelProfile {
        context_tokens: 32_768,
        ..ModelProfile::default()
    };
    assert_eq!(profile.readiness(42)["data"][0]["context_tokens"], 32_768);
    let prompt = 32_768 - 128 - crate::TEMPLATE_MARGIN_TOKENS;
    assert!(
        pending(true, 128)?
            .admit_with_context(profile.context_tokens, |_| Ok(prompt))
            .is_ok()
    );
    assert!(matches!(
        pending(true, 128)?.admit_with_context(profile.context_tokens, |_| Ok(prompt + 1)),
        Err(Rejection::ContextLimit),
    ));
    assert!(matches!(
        pending(true, 128)?.admit(|_| Ok(prompt)),
        Err(Rejection::ContextLimit),
    ));
    assert!(matches!(
        pending(true, 128)?
            .admit_with_context(4_352, |_| Ok(
                4_352 - 128 - crate::TEMPLATE_MARGIN_TOKENS + 1
            )),
        Err(Rejection::ContextLimit),
    ));
    assert!(matches!(
        pending(true, 128)?.admit_with_context(profile.context_tokens, |_| Ok(usize::MAX)),
        Err(Rejection::ContextLimit),
    ));
    Ok(())
}

#[test]
fn private_profile_readiness_binds_exact_representation_adapter_and_dynamic_scale() {
    use bonsai_prism_runtime::model::{
        lora::SWIFT_ADAPTER_SHA256,
        swift::{self, Representation},
    };
    let legacy = ModelProfile::default().readiness(42);
    assert_eq!(
        legacy,
        json!({"data":[{"id":"Swift-Bonsai-2-PQ2_0", "object":"model",
        "model_sha256":swift::MODEL_SHA256, "template_sha256":swift::TEMPLATE_SHA256,
        "runtime":"prism-inference", "context_tokens":8192, "seed":42, "row_workers":1}]})
    );
    let reused = ModelProfile {
        prompt_reuse: bonsai_prism_runtime::runtime::PromptReuse::Latest,
        ..ModelProfile::default()
    }
    .readiness(42);
    let mut expected_reused = legacy;
    expected_reused["data"][0]["prompt_reuse"] = json!("latest");
    assert_eq!(reused, expected_reused);
    for representation in [Representation::Pq2, Representation::CurrentRepackedPtq1] {
        for adapter in [false, true] {
            let reply = ModelProfile {
                representation,
                adapter,
                ..ModelProfile::default()
            }
            .readiness(17);
            let selected = &reply["data"][0];
            assert_eq!(selected["model_sha256"], representation.model_sha256());
            if adapter || representation != Representation::Pq2 {
                assert_eq!(selected["weight_representation"], representation.name());
                assert_eq!(
                    selected["adapter_sha256"],
                    json!(adapter.then_some(SWIFT_ADAPTER_SHA256))
                );
                assert_eq!(selected["supports_abliteration_scale"], adapter);
                assert_eq!(
                    selected.get("abliteration_scale_range"),
                    adapter.then_some(&json!([0, 2]))
                );
            }
        }
    }
}

#[test]
fn private_rotated_cache_readiness_reports_startup_identity_and_rejects_request_overrides()
-> TestResult {
    use bonsai_prism_runtime::graph::KvCachePolicy;

    run(async {
        for kv_cache in [
            KvCachePolicy::Rotated4,
            KvCachePolicy::Rotated4FittedValues,
            KvCachePolicy::Rotated4Keys3Values,
        ] {
            let identity: Arc<str> = Arc::from(format!(
                "verified-model:graph:kv-{}-d256-rht3-v1:scale-1.3",
                kv_cache.name()
            ));
            let observed = Observation {
                kv_cache: Some(kv_cache),
                startup_graph_identity: Some(identity.clone()),
                ..Observation::default()
            };
            let mut worker =
                Worker::new(FakeEngine(observed.clone()), 42).map_err(|e| format!("{e:?}"))?;
            for route in ["/v1/models", "/props"] {
                let (result, reply) = perform(&mut worker, request(route, &[], true)).await?;
                assert_eq!(result, Ok(()));
                let end = reply
                    .array_windows()
                    .position(|part| part == b"\r\n\r\n")
                    .ok_or("missing readiness response head")?
                    + 4;
                let body: Value = serde_json::from_slice(&reply[end..])?;
                let selected = &body["data"][0];
                assert_eq!(selected["context_tokens"], 8192);
                assert_eq!(selected["kv_cache"], kv_cache.name());
                assert_eq!(selected["startup_graph_identity"], identity.as_ref());
                assert!(selected.get("graph_identity").is_none());
            }
            let original = pending(true, 128).map_err(|e| format!("{e:?}"))?.body;
            let mut value: Value = serde_json::from_slice(&original)?;
            value["kv_cache"] = json!("f16");
            assert!(canonical_for_profile(&serde_json::to_vec(&value)?, &worker.profile).is_err());
            assert_eq!(observed.executions.load(Ordering::Relaxed), 0);
            assert_eq!(observed.preparations.load(Ordering::Relaxed), 0);
        }
        Ok(())
    })
}

#[test]
fn private_cache_profile_rejects_missing_unbounded_or_inconsistent_graph_identity() {
    use bonsai_prism_runtime::graph::KvCachePolicy;

    for (kv_cache, identity) in [
        (KvCachePolicy::F16, Some(Arc::<str>::from("unexpected"))),
        (KvCachePolicy::Rotated4, None),
        (KvCachePolicy::Rotated4Keys3Values, Some(Arc::from(""))),
        (KvCachePolicy::Rotated4, Some(Arc::from("x".repeat(4097)))),
    ] {
        let observed = Observation {
            kv_cache: Some(kv_cache),
            startup_graph_identity: identity,
            ..Observation::default()
        };
        assert!(matches!(
            Worker::new(FakeEngine(observed.clone()), 42),
            Err(NativeBackendError::Configuration)
        ));
        assert!(observed.terminated.load(Ordering::Acquire));
    }
}

fn cpu_base_graph_identity(cache: bonsai_prism_runtime::graph::KvCachePolicy) -> String {
    use bonsai_prism_runtime::graph::{KvCachePolicy, SWIFT_GRAPH_CONTRACT};
    let contract = if cache == KvCachePolicy::F16 {
        SWIFT_GRAPH_CONTRACT.to_owned()
    } else {
        SWIFT_GRAPH_CONTRACT.replace("kv-f16", &format!("kv-{}-d256-rht3-v1", cache.name()))
    };
    format!("verified-model:{contract}")
}

#[test]
fn private_cpu_prefill_readiness_binds_tiles_scratch_and_latest_reuse() -> TestResult {
    use bonsai_prism_runtime::{runtime::PromptReuse, swift_session::CpuPrefill};

    run(async {
        for (tokens, cache) in [8, 16, 32].into_iter().flat_map(|tokens| {
            [
                bonsai_prism_runtime::graph::KvCachePolicy::F16,
                bonsai_prism_runtime::graph::KvCachePolicy::Rotated4,
                bonsai_prism_runtime::graph::KvCachePolicy::Rotated4Keys3Values,
            ]
            .map(|cache| (tokens, cache))
        }) {
            let prefill = CpuPrefill::new(tokens, 64 * 1024 * 1024)?;
            let identity: Arc<str> = Arc::from(format!(
                "{}:cpu-tiled-ffn-v1:{tokens}",
                cpu_base_graph_identity(cache),
            ));
            let observed = Observation {
                kv_cache: Some(cache),
                cpu_prefill: Some(prefill),
                prompt_reuse: PromptReuse::Latest,
                startup_graph_identity: Some(identity.clone()),
                ..Observation::default()
            };
            let mut worker = Worker::new(FakeEngine(observed.clone()), 42)
                .map_err(|error| format!("{error:?}"))?;
            for route in ["/v1/models", "/props"] {
                let (result, reply) = perform(&mut worker, request(route, &[], true)).await?;
                assert_eq!(result, Ok(()));
                let end = reply
                    .array_windows()
                    .position(|part| part == b"\r\n\r\n")
                    .ok_or("missing readiness response head")?
                    + 4;
                let body: Value = serde_json::from_slice(&reply[end..])?;
                let profile = &body["data"][0];
                assert_eq!(profile["cpu_prefill_tokens"], tokens);
                assert_eq!(
                    profile["cpu_prefill_scratch_bytes"],
                    prefill.scratch_bytes()
                );
                assert_eq!(profile["prompt_reuse"], "latest");
                assert_eq!(profile["startup_graph_identity"], identity.as_ref());
                if cache == bonsai_prism_runtime::graph::KvCachePolicy::F16 {
                    assert!(profile.get("kv_cache").is_none());
                } else {
                    assert_eq!(profile["kv_cache"], cache.name());
                }
                assert!(profile.get("cpu_speculative_tokens").is_none());
            }
            let original = pending(true, 128)
                .map_err(|error| format!("{error:?}"))?
                .body;
            for field in ["cpu_prefill_tokens", "cpu_prefill_scratch_bytes"] {
                let mut value: Value = serde_json::from_slice(&original)?;
                value[field] = json!(8);
                assert!(
                    canonical_for_profile(&serde_json::to_vec(&value)?, &worker.profile).is_err()
                );
            }
            assert_eq!(observed.executions.load(Ordering::Relaxed), 0);
        }
        Ok(())
    })
}

#[test]
fn private_cpu_prefill_rejects_mismatched_precision_or_graph_selection() -> TestResult {
    use bonsai_prism_runtime::{graph::KvCachePolicy, swift_session::CpuPrefill};

    let prefill = CpuPrefill::new(8, 64 * 1024 * 1024)?;
    for (cache, identity) in [
        (KvCachePolicy::F16, None),
        (
            KvCachePolicy::F16,
            Some(Arc::from("verified-model:kv-f16:cpu-tiled-ffn-v1:16")),
        ),
        (
            KvCachePolicy::F16,
            Some(Arc::from(
                "verified-model:kv-f16:cpu-tiled-ffn-v1:8:gdn-wy-f32-v1",
            )),
        ),
        (
            KvCachePolicy::Rotated4,
            Some(Arc::from("verified-model:kv-f16:cpu-tiled-ffn-v1:8")),
        ),
    ] {
        let observed = Observation {
            kv_cache: Some(cache),
            cpu_prefill: Some(prefill),
            startup_graph_identity: identity,
            ..Observation::default()
        };
        assert!(matches!(
            Worker::new(FakeEngine(observed), 42),
            Err(NativeBackendError::Configuration)
        ));
    }
    Ok(())
}

#[test]
fn private_cpu_speculation_readiness_binds_sampler_profile_and_all_reserves() -> TestResult {
    use bonsai_prism_runtime::{
        graph::KvCachePolicy,
        runtime::PromptReuse,
        swift_session::{CpuPrefill, CpuSpeculation, STATE_MEMORY_BYTES},
    };
    run(async {
        for (tokens, cache) in [1, 4, 8].into_iter().flat_map(|tokens| {
            [
                KvCachePolicy::F16,
                KvCachePolicy::Rotated4,
                KvCachePolicy::Rotated4Keys3Values,
            ]
            .map(|cache| (tokens, cache))
        }) {
            let prefill = CpuPrefill::new(8, 64 * 1024 * 1024)?;
            let speculation = CpuSpeculation::new(tokens, 16 * 1024 * 1024, STATE_MEMORY_BYTES)?;
            let identity: Arc<str> = Arc::from(format!(
                "{}:rank-one-lora-v1:adapter:3fa66666:cpu-tiled-ffn-v1:8:cpu-tiled-verify-v1",
                cpu_base_graph_identity(cache),
            ));
            let observed = Observation {
                kv_cache: Some(cache),
                cpu_prefill: Some(prefill),
                cpu_speculation: Some(speculation),
                prompt_reuse: PromptReuse::Latest,
                startup_graph_identity: Some(identity.clone()),
                ..Observation::default()
            };
            let mut worker = Worker::new(FakeEngine(observed.clone()), 42)
                .map_err(|error| format!("{error:?}"))?;
            let (result, reply) = perform(&mut worker, request("/v1/models", &[], true)).await?;
            assert_eq!(result, Ok(()));
            let end = reply
                .array_windows()
                .position(|part| part == b"\r\n\r\n")
                .ok_or("missing readiness response head")?
                + 4;
            let body: Value = serde_json::from_slice(&reply[end..])?;
            let profile = &body["data"][0];
            assert_eq!(profile["cpu_speculative_tokens"], tokens);
            assert_eq!(
                profile["cpu_verification_scratch_bytes"],
                speculation.verification_scratch_bytes()
            );
            assert_eq!(
                profile["cpu_speculative_checkpoint_bytes"],
                speculation.checkpoint_bytes()
            );
            assert_eq!(profile["startup_graph_identity"], identity.as_ref());
            assert_eq!(profile["prompt_reuse"], "latest");
            let original = pending(true, 128)
                .map_err(|error| format!("{error:?}"))?
                .body;
            for field in [
                "cpu_speculative_tokens",
                "cpu_verification_scratch_bytes",
                "cpu_speculative_checkpoint_bytes",
            ] {
                let mut value: Value = serde_json::from_slice(&original)?;
                value[field] = json!(4);
                assert!(
                    canonical_for_profile(&serde_json::to_vec(&value)?, &worker.profile).is_err()
                );
            }
            assert_eq!(observed.executions.load(Ordering::Relaxed), 0);
        }
        Ok(())
    })
}

#[test]
fn private_cpu_speculation_rejects_missing_or_mismatched_graph_and_recovery_profiles() -> TestResult
{
    use bonsai_prism_runtime::{
        graph::KvCachePolicy,
        swift_session::{CpuPrefill, CpuSpeculation, STATE_MEMORY_BYTES},
    };
    let prefill = CpuPrefill::new(8, 64 * 1024 * 1024)?;
    let speculation = CpuSpeculation::new(4, 16 * 1024 * 1024, STATE_MEMORY_BYTES)?;
    for (cache, tile, identity) in [
        (
            KvCachePolicy::F16,
            None,
            "verified-model:kv-f16:cpu-tiled-ffn-v1:8:cpu-tiled-verify-v1",
        ),
        (
            KvCachePolicy::F16,
            Some(prefill),
            "verified-model:kv-f16:cpu-tiled-ffn-v1:8",
        ),
        (
            KvCachePolicy::F16,
            Some(prefill),
            "verified-model:cpu-tiled-ffn-v1:8:cpu-tiled-verify-v1",
        ),
        (
            KvCachePolicy::F16,
            Some(prefill),
            "verified-model:kv-f16:kv-f16:cpu-tiled-ffn-v1:8:cpu-tiled-verify-v1",
        ),
        (
            KvCachePolicy::F16,
            Some(prefill),
            "verified-model:kv-f16-unknown:cpu-tiled-ffn-v1:8:cpu-tiled-verify-v1",
        ),
        (
            KvCachePolicy::F16,
            Some(prefill),
            "verified-model:kv-f16:cpu-tiled-ffn-v1:16:cpu-tiled-verify-v1",
        ),
        (
            KvCachePolicy::F16,
            Some(prefill),
            "verified-model:kv-f16:cpu-tiled-ffn-v1:8:gdn-wy-f32-v1:cpu-tiled-verify-v1",
        ),
        (
            KvCachePolicy::Rotated4,
            Some(prefill),
            "verified-model:kv-rsq4k3v-d256-rht3-v1:cpu-tiled-ffn-v1:8:cpu-tiled-verify-v1",
        ),
        (
            KvCachePolicy::F16,
            Some(prefill),
            "verified-model:kv-rsq4-d256-rht3-v1:cpu-tiled-ffn-v1:8:cpu-tiled-verify-v1",
        ),
        (
            KvCachePolicy::Rotated4,
            Some(prefill),
            "verified-model:kv-rsq4-d256-rht3-v1:kv-rsq4k3v-d256-rht3-v1:cpu-tiled-ffn-v1:8:cpu-tiled-verify-v1",
        ),
    ] {
        let observed = Observation {
            kv_cache: Some(cache),
            cpu_prefill: tile,
            cpu_speculation: Some(speculation),
            startup_graph_identity: Some(Arc::from(identity)),
            ..Observation::default()
        };
        assert!(matches!(
            Worker::new(FakeEngine(observed), 42),
            Err(NativeBackendError::Configuration)
        ));
    }
    let observed = Observation {
        cpu_prefill: Some(prefill),
        cpu_speculation: Some(CpuSpeculation::new(
            4,
            16 * 1024 * 1024,
            STATE_MEMORY_BYTES - 1,
        )?),
        startup_graph_identity: Some(Arc::from(
            "verified-model:kv-f16:cpu-tiled-ffn-v1:8:cpu-tiled-verify-v1",
        )),
        ..Observation::default()
    };
    assert!(matches!(
        Worker::new(FakeEngine(observed), 42),
        Err(NativeBackendError::Configuration)
    ));
    Ok(())
}

#[test]
fn private_readiness_reports_engine_worker_budget_and_rejects_invalid_counts() -> TestResult {
    use bonsai_prism_runtime::kernels::MAX_ROW_WORKERS;
    run(async {
        for row_workers in [1, 4, MAX_ROW_WORKERS] {
            let observed = Observation {
                row_workers: Some(row_workers),
                ..Observation::default()
            };
            let mut worker =
                Worker::new(FakeEngine(observed.clone()), 42).map_err(|e| format!("{e:?}"))?;
            for route in ["/v1/models", "/props"] {
                let (result, reply) = perform(&mut worker, request(route, &[], true)).await?;
                assert_eq!(result, Ok(()));
                let end = reply
                    .array_windows()
                    .position(|part| part == b"\r\n\r\n")
                    .ok_or("missing readiness response head")?
                    + 4;
                let body: Value = serde_json::from_slice(&reply[end..])?;
                assert_eq!(body["data"][0]["row_workers"], row_workers);
                assert!(body["data"][0].get("parallel_row_operations").is_none());
            }
            assert_eq!(observed.executions.load(Ordering::Relaxed), 0);
            assert_eq!(observed.preparations.load(Ordering::Relaxed), 0);
        }
        for row_workers in [0, MAX_ROW_WORKERS + 1, usize::MAX] {
            let observed = Observation {
                row_workers: Some(row_workers),
                ..Observation::default()
            };
            assert!(matches!(
                Worker::new(FakeEngine(observed.clone()), 42),
                Err(NativeBackendError::Configuration)
            ));
            assert!(observed.terminated.load(Ordering::Acquire));
        }
        Ok(())
    })
}

#[test]
fn private_latest_prefix_policy_preserves_counting_and_transactional_reset_boundaries() -> TestResult
{
    use bonsai_prism_runtime::runtime::PromptReuse;

    run(async {
        let observed = Observation {
            prompt_reuse: PromptReuse::Latest,
            ..Observation::default()
        };
        let mut worker =
            Worker::new(FakeEngine(observed.clone()), 42).map_err(|error| format!("{error:?}"))?;
        let (result, reply) = perform(&mut worker, request("/v1/models", &[], true)).await?;
        assert_eq!(result, Ok(()));
        let end = reply
            .array_windows()
            .position(|part| part == b"\r\n\r\n")
            .ok_or("missing readiness head")?
            + 4;
        let readiness: Value = serde_json::from_slice(&reply[end..])?;
        assert_eq!(readiness["data"][0]["prompt_reuse"], "latest");

        let body = pending(true, 128)
            .map_err(|error| format!("{error:?}"))?
            .body;
        for committed in 1..=2 {
            let (result, _) = perform(
                &mut worker,
                request("/v1/chat/completions/input_tokens", &body, true),
            )
            .await?;
            assert_eq!(result, Ok(()));
            assert_eq!(observed.state.load(Ordering::Relaxed), committed - 1);
            let (result, _) =
                perform(&mut worker, request("/v1/chat/completions", &body, true)).await?;
            assert_eq!(result, Ok(()));
            assert_eq!(observed.state.load(Ordering::Relaxed), committed);
            assert_eq!(observed.resets.load(Ordering::Relaxed), 0);
        }

        let (result, _) = perform(
            &mut worker,
            request("/v1/chat/completions/input_tokens", &body, true),
        )
        .await?;
        assert_eq!(result, Ok(()));
        fail_acknowledgement(&mut worker, &body, false).await?;
        assert_eq!(observed.state.load(Ordering::Relaxed), 2);
        let (result, _) = perform(
            &mut worker,
            request("/v1/chat/completions/input_tokens", &body, true),
        )
        .await?;
        assert_eq!(result, Err(NativeBackendError::Engine));
        let (result, _) = perform(&mut worker, request("/reset", &[], true)).await?;
        assert_eq!(result, Ok(()));
        assert_eq!(observed.state.load(Ordering::Relaxed), 0);
        assert_eq!(observed.resets.load(Ordering::Relaxed), 1);
        // Reset invalidates the one-use count seal as well as all live prefix state.
        let (result, _) =
            perform(&mut worker, request("/v1/chat/completions", &body, true)).await?;
        assert_eq!(result, Err(NativeBackendError::CountMismatch));
        let (result, _) = perform(
            &mut worker,
            request("/v1/chat/completions/input_tokens", &body, true),
        )
        .await?;
        assert_eq!(result, Ok(()));
        let (result, _) =
            perform(&mut worker, request("/v1/chat/completions", &body, true)).await?;
        assert_eq!(result, Ok(()));
        assert_eq!(observed.state.load(Ordering::Relaxed), 1);
        assert_eq!(worker.profile.prompt_reuse, PromptReuse::Latest);
        Ok(())
    })
}

#[test]
fn private_adapter_signed_zero_changes_require_recounting() -> TestResult {
    run(async {
        let observed = Observation::default();
        let mut worker =
            Worker::new(FakeEngine(observed.clone()), 42).map_err(|e| format!("{e:?}"))?;
        worker.profile.adapter = true;
        let mut value: Value =
            serde_json::from_slice(&pending(true, 128).map_err(|e| format!("{e:?}"))?.body)?;
        let mut bodies = Vec::new();
        for scale in [0.0_f64, -0.0] {
            value["lora"] = json!([{"id":0,"scale":scale}]);
            bodies.push(
                canonical_for_profile(&serde_json::to_vec(&value)?, &worker.profile)
                    .map_err(|e| format!("{e:?}"))?
                    .body,
            );
        }
        let (result, _) = perform(
            &mut worker,
            request("/v1/chat/completions/input_tokens", &bodies[0], true),
        )
        .await?;
        assert_eq!(result, Ok(()));
        let (result, _) = perform(
            &mut worker,
            request("/v1/chat/completions", &bodies[1], true),
        )
        .await?;
        assert_eq!(result, Err(NativeBackendError::CountMismatch));
        assert_eq!(observed.executions.load(Ordering::Relaxed), 0);
        Ok(())
    })
}

#[test]
fn private_thinking_mode_controls_visible_output_and_is_bound_to_exact_count() -> TestResult {
    run(async {
        for streaming in [false, true] {
            let observed = Observation::default();
            let mut worker =
                Worker::new(FakeEngine(observed.clone()), 42).map_err(|e| format!("{e:?}"))?;
            let original = pending(streaming, 128).map_err(|e| format!("{e:?}"))?.body;
            let mut value: Value = serde_json::from_slice(&original)?;
            value["chat_template_kwargs"]["enable_thinking"] = json!(false);
            let body = canonical(&serde_json::to_vec(&value)?)
                .map_err(|e| format!("{e:?}"))?
                .body;
            assert_eq!(
                serde_json::from_slice::<Value>(&body)?["chat_template_kwargs"]["enable_thinking"],
                false
            );
            let (result, _) = perform(
                &mut worker,
                request("/v1/chat/completions/input_tokens", &body, true),
            )
            .await?;
            assert_eq!(result, Ok(()));
            let (result, _) = perform(
                &mut worker,
                request("/v1/chat/completions", &original, true),
            )
            .await?;
            assert_eq!(result, Err(NativeBackendError::CountMismatch));
            assert_eq!(observed.executions.load(Ordering::Relaxed), 0);

            let (result, _) = perform(
                &mut worker,
                request("/v1/chat/completions/input_tokens", &body, true),
            )
            .await?;
            assert_eq!(result, Ok(()));
            let (result, reply) =
                perform(&mut worker, request("/v1/chat/completions", &body, true)).await?;
            assert_eq!(result, Ok(()));
            let text = std::str::from_utf8(&reply)?;
            if streaming {
                assert!(!text.contains("reasoning_content"));
            } else {
                assert!(text.contains("\"reasoning_content\":\"\""));
            }
            assert!(text.contains("\"content\":\"draft"));
            assert!(text.contains("</think>answer"));
            assert_eq!(observed.executions.load(Ordering::Relaxed), 1);
        }
        Ok(())
    })
}

#[test]
fn private_health_auth_count_and_completion_use_one_prepared_input() -> TestResult {
    run(async {
        let observed = Observation::default();
        let mut worker =
            Worker::new(FakeEngine(observed.clone()), 42).map_err(|e| format!("{e:?}"))?;
        let (result, denied) = perform(&mut worker, request("/v1/models", &[], false)).await?;
        assert_eq!(result, Err(NativeBackendError::Unauthorized));
        assert!(denied.starts_with(b"HTTP/1.1 401 "));
        for route in ["/health", "/v1/models"] {
            let (result, reply) = perform(&mut worker, request(route, &[], true)).await?;
            assert_eq!(result, Ok(()));
            assert!(reply.starts_with(b"HTTP/1.1 200 "));
        }
        assert_eq!(observed.preparations.load(Ordering::Relaxed), 0);
        for streaming in [true, false] {
            let body = pending(streaming, 128).map_err(|e| format!("{e:?}"))?.body;
            let (result, count) = perform(
                &mut worker,
                request("/v1/chat/completions/input_tokens", &body, true),
            )
            .await?;
            assert_eq!(result, Ok(()));
            assert!(std::str::from_utf8(&count)?.contains("\"input_tokens\":100"));
            let (result, reply) =
                perform(&mut worker, request("/v1/chat/completions", &body, true)).await?;
            assert_eq!(result, Ok(()));
            let text = std::str::from_utf8(&reply)?;
            assert!(text.contains("draft"));
            assert!(text.contains("answer"));
            assert!(text.contains("reasoning_content"));
            assert!(text.contains("\"finish_reason\":\"length\""));
        }
        assert_eq!(observed.preparations.load(Ordering::Relaxed), 2);
        assert_eq!(observed.executions.load(Ordering::Relaxed), 2);
        let body = pending(true, 128).map_err(|e| format!("{e:?}"))?.body;
        let (result, _) = perform(
            &mut worker,
            request("/v1/chat/completions/input_tokens", &body, true),
        )
        .await?;
        assert_eq!(result, Ok(()));
        let changed = pending(true, 127).map_err(|e| format!("{e:?}"))?.body;
        let (result, _) =
            perform(&mut worker, request("/v1/chat/completions", &changed, true)).await?;
        assert_eq!(result, Err(NativeBackendError::CountMismatch));
        assert_eq!(observed.executions.load(Ordering::Relaxed), 2);
        Ok(())
    })
}

#[test]
fn private_disconnect_rolls_back_and_requires_reset_before_subsequent_success() -> TestResult {
    run(async {
        let observed = Observation::default();
        let mut worker =
            Worker::new(FakeEngine(observed.clone()), 42).map_err(|e| format!("{e:?}"))?;
        let body = pending(true, 128).map_err(|e| format!("{e:?}"))?.body;
        let (result, _) = perform(
            &mut worker,
            request("/v1/chat/completions/input_tokens", &body, true),
        )
        .await?;
        assert_eq!(result, Ok(()));
        let (mut client, server) = duplex(64);
        let policy = RequestPolicy::new(8080, KEY).map_err(|e| format!("{e:?}"))?;
        let serving = serve_connection(
            server,
            &policy,
            &mut worker,
            Instant::now() + Duration::from_secs(10),
        );
        let client = async {
            client
                .write_all(&request("/v1/chat/completions", &body, true))
                .await?;
            let mut bytes = Vec::new();
            let mut buffer = [0; 17];
            while !bytes.array_windows().any(|part| part == b"draft") {
                let count = client.read(&mut buffer).await?;
                if count == 0 {
                    return Err(std::io::Error::other("generation ended before abandonment"));
                }
                bytes.extend_from_slice(&buffer[..count]);
            }
            client.shutdown().await?;
            client.read_to_end(&mut bytes).await?;
            Ok::<_, std::io::Error>(())
        };
        let (result, client) = together(serving, client).await;
        client?;
        assert_eq!(result, Err(NativeBackendError::ClientAbandoned));
        assert_eq!(observed.state.load(Ordering::Relaxed), 0);
        let (result, _) = perform(
            &mut worker,
            request("/v1/chat/completions/input_tokens", &body, true),
        )
        .await?;
        assert_eq!(result, Err(NativeBackendError::Engine));
        let (result, _) = perform(&mut worker, request("/reset", &[], true)).await?;
        assert_eq!(result, Ok(()));
        assert_eq!(observed.resets.load(Ordering::Relaxed), 1);
        let (result, _) = perform(
            &mut worker,
            request("/v1/chat/completions/input_tokens", &body, true),
        )
        .await?;
        assert_eq!(result, Ok(()));
        let completed_cancellation = Cancellation::default();
        let (mut client, server) = duplex(128);
        let mut response = Vec::new();
        let (result, read) = together(
            exchange(
                server,
                &mut worker,
                Input::Generate(canonical(&body).map_err(|e| format!("{e:?}"))?),
                &completed_cancellation,
                Instant::now() + Duration::from_secs(10),
            ),
            client.read_to_end(&mut response),
        )
        .await;
        read?;
        assert_eq!(result, Ok(()));
        assert!(!completed_cancellation.is_cancelled());
        assert_eq!(observed.state.load(Ordering::Relaxed), 1);
        Ok(())
    })
}

#[test]
fn private_dropped_exchange_cancels_and_recovers_only_after_reset() -> TestResult {
    run(async {
        let observed = Observation::default();
        let mut worker =
            Worker::new(FakeEngine(observed.clone()), 42).map_err(|e| format!("{e:?}"))?;
        let body = pending(true, 128).map_err(|e| format!("{e:?}"))?.body;
        let (result, _) = perform(
            &mut worker,
            request("/v1/chat/completions/input_tokens", &body, true),
        )
        .await?;
        assert_eq!(result, Ok(()));
        let cancellation = Cancellation::default();
        let (mut client, server) = duplex(1);
        {
            let serving = exchange(
                server,
                &mut worker,
                Input::Generate(canonical(&body).map_err(|e| format!("{e:?}"))?),
                &cancellation,
                Instant::now() + Duration::from_secs(10),
            );
            let receive = async {
                let mut bytes = Vec::new();
                let mut buffer = [0; 17];
                while !bytes.array_windows().any(|part| part == b"draft") {
                    let count = client.read(&mut buffer).await?;
                    if count == 0 {
                        return Err(std::io::Error::other("missing provisional output"));
                    }
                    bytes.extend_from_slice(&buffer[..count]);
                }
                Ok::<_, std::io::Error>(())
            };
            let mut serving = pin!(serving);
            let mut receive = pin!(receive);
            poll_fn(|cx| {
                if serving.as_mut().poll(cx).is_ready() {
                    return Poll::Ready(Err(std::io::Error::other(
                        "exchange completed before abandonment",
                    )));
                }
                receive.as_mut().poll(cx)
            })
            .await?;
            assert!(!cancellation.is_cancelled());
            // Drop the pending future without returning an error through its
            // ordinary path, while retaining its model owner and token.
        }
        assert!(cancellation.is_cancelled());
        worker.settled().await;
        assert_eq!(observed.state.load(Ordering::Relaxed), 0);
        let (result, _) = perform(
            &mut worker,
            request("/v1/chat/completions/input_tokens", &body, true),
        )
        .await?;
        assert_eq!(result, Err(NativeBackendError::Engine));
        let (result, _) = perform(&mut worker, request("/reset", &[], true)).await?;
        assert_eq!(result, Ok(()));
        let (result, _) = perform(
            &mut worker,
            request("/v1/chat/completions/input_tokens", &body, true),
        )
        .await?;
        assert_eq!(result, Ok(()));
        let (result, _) =
            perform(&mut worker, request("/v1/chat/completions", &body, true)).await?;
        assert_eq!(result, Ok(()));
        assert_eq!(observed.state.load(Ordering::Relaxed), 1);
        Ok(())
    })
}

#[test]
fn private_worker_drop_cancels_queued_unacknowledged_output_and_joins() -> TestResult {
    run(async {
        let observed = Observation::default();
        let mut worker =
            Worker::new(FakeEngine(observed.clone()), 42).map_err(|e| format!("{e:?}"))?;
        let mut events = worker
            .start(
                Input::Count(pending(true, 128).map_err(|e| format!("{e:?}"))?),
                Cancellation::default(),
                std::time::Instant::now() + Duration::from_secs(10),
            )
            .map_err(|e| format!("{e:?}"))?;
        while events.is_empty() {
            tokio::task::yield_now().await;
        }
        // Keep the receiver and its unacknowledged queued event live while
        // dropping the owner. Joining must cancel the wait, not detach it or
        // block behind a second send into the already occupied one-event slot.
        drop(worker);
        assert!(observed.terminated.load(Ordering::Acquire));
        let event = events.recv().await.ok_or("missing provisional count")?;
        assert!(matches!(event.output, Output::Count(100)));
        assert!(
            event
                .ack
                .ok_or("missing count acknowledgement")?
                .send(())
                .is_err()
        );
        Ok(())
    })
}

#[test]
fn private_head_reuses_strict_host_auth_origin_and_framing_policy() -> TestResult {
    let policy = RequestPolicy::new(8080, KEY).map_err(|e| format!("{e:?}"))?;
    let bytes = request("/v1/chat/completions/input_tokens", b"{}", true);
    let end = bytes
        .array_windows()
        .position(|part| part == b"\r\n\r\n")
        .ok_or("missing fixture head")?
        + 4;
    let head = std::str::from_utf8(&bytes[..end])?;
    assert!(matches!(
        parse_head(head.as_bytes(), &policy),
        Ok((Route::Count, 2))
    ));
    for changed in [
        head.replace("\r\n", "\n"),
        head.replace("Host: 127.0.0.1:8080", "Host: 127.0.0.1:8081"),
        head.replace(
            "Host: 127.0.0.1:8080",
            "Host: 127.0.0.1:8080\r\nHost: 127.0.0.1:8080",
        ),
        head.replace(
            "Content-Length: 2",
            "Content-Length: 2\r\nContent-Length: 2",
        ),
        head.replace(
            "Content-Length: 2",
            "Content-Length: 2\r\nTransfer-Encoding: chunked",
        ),
        head.replace(
            "Connection: close",
            "Connection: close\r\nOrigin: http://elsewhere.invalid",
        ),
        head.replace(
            "Connection: close",
            "Connection: close\r\nForwarded: for=127.0.0.1",
        ),
        head.replace(
            "/v1/chat/completions/input_tokens",
            "/v1/chat/completions/../input_tokens",
        ),
    ] {
        assert!(parse_head(changed.as_bytes(), &policy).is_err());
    }
    Ok(())
}

async fn fail_acknowledgement(worker: &mut Worker, body: &[u8], terminal: bool) -> TestResult {
    let allowance = if terminal {
        Duration::from_secs(10)
    } else {
        Duration::from_secs(1)
    };
    let mut events = worker
        .start(
            Input::Generate(
                canonical_for_profile(body, &worker.profile).map_err(|e| format!("{e:?}"))?,
            ),
            Cancellation::default(),
            std::time::Instant::now() + allowance,
        )
        .map_err(|e| format!("{e:?}"))?;
    let held = loop {
        let Event { output, ack } = events.recv().await.ok_or("missing worker event")?;
        let target = if terminal {
            matches!(output, Output::Complete(_))
        } else {
            matches!(output, Output::Delta(_))
        };
        if target {
            break ack;
        }
        assert!(!matches!(output, Output::Done(_)));
        ack.ok_or("missing write acknowledgement")?
            .send(())
            .map_err(|_| "worker rejected acknowledgement")?;
    };
    let held = if terminal {
        drop(held);
        None
    } else {
        held
    };
    let event = events.recv().await.ok_or("missing failed completion")?;
    assert!(matches!(
        event.output,
        Output::Done(Err(NativeBackendError::Engine))
    ));
    drop(held);
    drop(events);
    worker.settled().await;
    Ok(())
}

#[test]
fn private_unacknowledged_deadline_and_terminal_failure_restore_then_close_admission() -> TestResult
{
    run(async {
        for (terminal, adapter) in [(false, false), (true, false), (false, true), (true, true)] {
            let observed = Observation::default();
            let mut worker =
                Worker::new(FakeEngine(observed.clone()), 42).map_err(|e| format!("{e:?}"))?;
            let mut value: Value =
                serde_json::from_slice(&pending(true, 128).map_err(|e| format!("{e:?}"))?.body)?;
            if adapter {
                worker.profile = ModelProfile {
                    representation:
                        bonsai_prism_runtime::model::swift::Representation::CurrentRepackedPtq1,
                    adapter: true,
                    ..ModelProfile::default()
                };
                value["lora"] = json!([{"id":0,"scale":1.3}]);
            }
            let body = canonical_for_profile(&serde_json::to_vec(&value)?, &worker.profile)
                .map_err(|e| format!("{e:?}"))?
                .body;
            let (counted, _) = perform(
                &mut worker,
                request("/v1/chat/completions/input_tokens", &body, true),
            )
            .await?;
            assert_eq!(counted, Ok(()));
            fail_acknowledgement(&mut worker, &body, terminal).await?;
            assert_eq!(observed.state.load(Ordering::Relaxed), 0);
            let (result, _) = perform(
                &mut worker,
                request("/v1/chat/completions/input_tokens", &body, true),
            )
            .await?;
            assert_eq!(result, Err(NativeBackendError::Engine));
            let (result, _) = perform(&mut worker, request("/reset", &[], true)).await?;
            assert_eq!(result, Ok(()));
            if adapter {
                value["lora"] = json!([{"id":0,"scale":0.0}]);
            }
            let next = canonical_for_profile(&serde_json::to_vec(&value)?, &worker.profile)
                .map_err(|e| format!("{e:?}"))?
                .body;
            let (result, _) = perform(
                &mut worker,
                request("/v1/chat/completions/input_tokens", &next, true),
            )
            .await?;
            assert_eq!(result, Ok(()));
            let (result, _) =
                perform(&mut worker, request("/v1/chat/completions", &next, true)).await?;
            assert_eq!(result, Ok(()));
        }
        Ok(())
    })
}

#[test]
fn private_uncapped_generation_survives_ten_minutes_and_finite_deadlines_cancel() -> TestResult {
    run(async {
        for finite in [false, true] {
            let gate = Arc::new(AtomicBool::new(false));
            let observed = Observation {
                generation_gate: Some(gate.clone()),
                ..Observation::default()
            };
            let mut worker = Worker::new(FakeEngine(observed.clone()), 42)
                .map_err(|error| format!("{error:?}"))?;
            let body = pending(true, 128)
                .map_err(|error| format!("{error:?}"))?
                .body;
            let (result, _) = perform(
                &mut worker,
                request("/v1/chat/completions/input_tokens", &body, true),
            )
            .await?;
            assert_eq!(result, Ok(()));
            let policy = RequestPolicy::new(8080, KEY).map_err(|error| format!("{error:?}"))?;
            let (mut client, server) = duplex(4096);
            client
                .write_all(&request("/v1/chat/completions", &body, true))
                .await?;
            let deadline = finite.then(|| Instant::now() + Duration::from_mins(10));
            let mut serving = pin!(serve_connection(server, &policy, &mut worker, deadline));
            poll_fn(|cx| {
                assert!(serving.as_mut().poll(cx).is_pending());
                if observed.executions.load(Ordering::Acquire) != 0 {
                    Poll::Ready(())
                } else {
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
            })
            .await;
            assert_eq!(
                observed.finite_deadlines.load(Ordering::Relaxed),
                usize::from(finite)
            );
            tokio::time::pause();
            tokio::time::advance(Duration::from_mins(11)).await;
            tokio::time::resume();
            if !finite {
                poll_fn(|cx| {
                    assert!(serving.as_mut().poll(cx).is_pending());
                    Poll::Ready(())
                })
                .await;
                gate.store(true, Ordering::Release);
            }
            let receiving = async {
                let mut bytes = Vec::new();
                client.read_to_end(&mut bytes).await?;
                Ok::<_, std::io::Error>(bytes)
            };
            let (result, bytes) = together(serving, receiving).await;
            let bytes = bytes?;
            if finite {
                assert_eq!(result, Err(NativeBackendError::Deadline));
                assert!(bytes.is_empty());
                assert!(observed.generation_cancelled.load(Ordering::Acquire));
                assert_eq!(observed.state.load(Ordering::Relaxed), 0);
            } else {
                assert_eq!(result, Ok(()));
                assert!(std::str::from_utf8(&bytes)?.contains("answer"));
                assert!(std::str::from_utf8(&bytes)?.contains("[DONE]"));
            }
        }
        Ok(())
    })
}

#[test]
fn private_request_failure_diagnostic_contains_only_fixed_error_category() -> TestResult {
    for (error, category) in [
        (NativeBackendError::Deadline, "Deadline"),
        (NativeBackendError::Engine, "Engine"),
        (NativeBackendError::CountMismatch, "CountMismatch"),
    ] {
        let mut bytes = Vec::new();
        write_request_failure(&mut bytes, error)?;
        assert_eq!(
            std::str::from_utf8(&bytes)?,
            format!("private-inference-request-failed error={category}\n")
        );
    }
    Ok(())
}

#[test]
fn private_uncapped_requests_preserve_header_intake_and_write_limits() -> TestResult {
    run(async {
        let policy = RequestPolicy::new(8080, KEY).map_err(|error| format!("{error:?}"))?;
        for body_intake in [false, true] {
            let (mut client, mut server) = duplex(4096);
            if body_intake {
                let body = pending(true, 128)
                    .map_err(|error| format!("{error:?}"))?
                    .body;
                let bytes = request("/v1/chat/completions/input_tokens", &body, true);
                client.write_all(&bytes[..bytes.len() - 1]).await?;
            }
            let profile = ModelProfile::default();
            let mut receiving = pin!(receive(&mut server, &policy, &profile, None));
            poll_fn(|cx| {
                assert!(receiving.as_mut().poll(cx).is_pending());
                Poll::Ready(())
            })
            .await;
            tokio::time::pause();
            tokio::time::advance(if body_intake {
                INTAKE_TIMEOUT
            } else {
                HEADER_TIMEOUT
            })
            .await;
            tokio::time::resume();
            assert!(matches!(receiving.await, Err(NativeBackendError::Deadline)));
        }
        let (_client, mut server) = duplex(1);
        let body = json!({"status":"ok"});
        let mut writing = pin!(write_json(&mut server, 200, &body, None));
        poll_fn(|cx| {
            assert!(writing.as_mut().poll(cx).is_pending());
            Poll::Ready(())
        })
        .await;
        tokio::time::pause();
        tokio::time::advance(CLIENT_WRITE_TIMEOUT).await;
        tokio::time::resume();
        assert_eq!(writing.await, Err(NativeBackendError::Deadline));
        Ok(())
    })
}

#[test]
fn private_expected_probe_and_context_rejections_are_not_failure_diagnostics() -> TestResult {
    for error in [
        NativeBackendError::Unauthorized,
        NativeBackendError::Admission(Rejection::ContextLimit),
    ] {
        let mut bytes = Vec::new();
        write_request_failure(&mut bytes, error)?;
        assert!(bytes.is_empty());
    }
    Ok(())
}
