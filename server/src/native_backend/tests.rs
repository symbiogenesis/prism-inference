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
    preparations: Arc<AtomicUsize>,
    executions: Arc<AtomicUsize>,
    state: Arc<AtomicUsize>,
    resets: Arc<AtomicUsize>,
    terminated: Arc<AtomicBool>,
}
struct FakeEngine(Observation);

impl Drop for FakeEngine {
    fn drop(&mut self) {
        self.0.terminated.store(true, Ordering::Release);
    }
}

impl worker::Engine for FakeEngine {
    type Prepared = FakePrepared;
    fn prepare(&self, pending: PendingChat) -> Result<Self::Prepared, NativeBackendError> {
        self.0.preparations.fetch_add(1, Ordering::Relaxed);
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
        _cancel: &Cancellation,
        _deadline: std::time::Instant,
        mut emit: impl FnMut(TextEvent<'_>) -> bonsai_prism_runtime::Result<()>,
    ) -> bonsai_prism_runtime::Result<Completion> {
        self.0.executions.fetch_add(1, Ordering::Relaxed);
        let original = self.0.state.fetch_add(1, Ordering::Relaxed);
        let completion = Completion {
            prompt_tokens: 100,
            reused_tokens: 0,
            generated_tokens: 2,
            finish: Finish::Length,
        };
        let result = (|| {
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
        assert!(canonical_for_profile(&original, profile).is_err());
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
            assert!(canonical_for_profile(&serde_json::to_vec(&value)?, profile).is_err());
        }
        for scale in [0.0_f64, -0.0, 1.3, 2.0] {
            value["lora"] = json!([{"id":0,"scale":scale}]);
            let bytes = serde_json::to_vec(&value)?;
            assert!(canonical(&bytes).is_err());
            let prepared = canonical_for_profile(&bytes, profile).map_err(|e| format!("{e:?}"))?;
            let canonical: Value = serde_json::from_slice(&prepared.body)?;
            assert_eq!(
                serde_json::from_value::<f64>(canonical["lora"][0]["scale"].clone())?.to_bits(),
                scale.to_bits()
            );
        }
        value["lora"] = json!([{"id":0,"scale":1.3}]);
        value["abliteration_scale"] = json!(1.3);
        assert!(canonical_for_profile(&serde_json::to_vec(&value)?, profile).is_err());
        value
            .as_object_mut()
            .ok_or("Missing request object.")?
            .remove("abliteration_scale");
        let body = canonical_for_profile(&serde_json::to_vec(&value)?, profile)
            .map_err(|e| format!("{e:?}"))?
            .body;
        let duplicate =
            std::str::from_utf8(&body)?.replacen("\"scale\":", "\"scale\":1.3,\"scale\":", 1);
        assert!(canonical_for_profile(duplicate.as_bytes(), profile).is_err());
        let observed = Observation::default();
        let mut worker =
            Worker::new(FakeEngine(observed.clone()), 42).map_err(|e| format!("{e:?}"))?;
        worker.profile = profile;
        let (result, _) = perform(
            &mut worker,
            request("/v1/chat/completions/input_tokens", &body, true),
        )
        .await?;
        assert_eq!(result, Ok(()));
        value["lora"] = json!([{"id":0,"scale":0.0}]);
        let changed = canonical_for_profile(&serde_json::to_vec(&value)?, profile)
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
        "runtime":"prism-inference", "context_tokens":8192, "seed":42}]})
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
                canonical_for_profile(&serde_json::to_vec(&value)?, worker.profile)
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
            while !bytes.windows(5).any(|part| part == b"draft") {
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
                while !bytes.windows(5).any(|part| part == b"draft") {
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
                canonical_for_profile(body, worker.profile).map_err(|e| format!("{e:?}"))?,
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
            let body = canonical_for_profile(&serde_json::to_vec(&value)?, worker.profile)
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
            let next = canonical_for_profile(&serde_json::to_vec(&value)?, worker.profile)
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
