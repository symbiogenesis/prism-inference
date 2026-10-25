#![forbid(unsafe_code)]
#![cfg(feature = "cli")]
//! Focused pinned readiness, count/completion and abandoned-stream contracts.

use bonsai_prism_runtime::{
    Result,
    model::swift,
    runtime::{Cancellation, Finish},
    swift_client::{
        ClientError, DEFAULT_RESPONSE_FRAME_BYTES, ExpectedSwiftWorker, MAX_DELTA_BYTES,
        MAX_FRAME_BYTES, SwiftClient, SwiftWorkerBackend,
    },
    swift_session::MAX_OUTPUT_BYTES,
};
use serde_json::{Value, json};
use std::{
    io::{self, BufRead, BufReader, Cursor, Read, Write},
    time::{Duration, Instant},
};

const fn expected() -> ExpectedSwiftWorker {
    ExpectedSwiftWorker {
        seed: 17,
        backend: SwiftWorkerBackend::Scalar,
        max_response_frame_bytes: DEFAULT_RESPONSE_FRAME_BYTES,
    }
}
fn ready() -> Value {
    json!({
        "type":"ready","protocol":"bonsai-swift-worker-v1","model_sha256":swift::MODEL_SHA256,
        "template_sha256":swift::TEMPLATE_SHA256,"vocab_size":248_320,"runtime":"bonsai-prism-runtime",
        "prism_commit":bonsai_gguf::CODEC_COMMIT,"seed":17,"backend":"scalar","max_frame_bytes":MAX_FRAME_BYTES,
        "max_output_bytes":MAX_OUTPUT_BYTES,"context_tokens":8192,"max_output_tokens":4096,
        "temperature":1,"top_p":0.95,"top_k":20,"min_p":0,"repeat_penalty":1,"presence_penalty":0,
        "frequency_penalty":0,"thinking":true,"reasoning_effort":"xhigh","preserve_thinking":true,
        "cache_reuse":false,"speculation":false,"adapters":false,"qualification":"pending-full-model-parity-and-device-evidence"
    })
}
fn messages() -> Value {
    json!([{"role":"user","content":"private \"question\" 🌱"}])
}
fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(60)
}
fn count(id: u64) -> Value {
    json!({"type":"count","id":id,"prompt_tokens":61,"reserved_output_tokens":3})
}
fn terminal(id: u64, generated: usize, reason: &str) -> Value {
    json!({"type":"complete","id":id,"prompt_tokens":61,"generated_tokens":generated,"finish_reason":reason})
}
fn script(responses: &[Value]) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec(&ready())?;
    bytes.push(b'\n');
    for frame in responses {
        serde_json::to_writer(&mut bytes, frame)?;
        bytes.push(b'\n');
    }
    Ok(bytes)
}

#[test]
fn checks_count_complete_identical_messages_accounting_streaming_and_reset() -> Result<()> {
    let responses = script(&[
        count(0),
        count(1),
        json!({"type":"delta","id":2,"text":"café 🌱"}),
        terminal(2, 2, "stop"),
        json!({"type":"reset","id":3,"ready":true}),
        count(4),
        json!({"type":"delta","id":5,"text":"again"}),
        terminal(5, 3, "length"),
    ])?;
    let mut output = Vec::new();
    let mut client = SwiftClient::connect(
        BufReader::with_capacity(1, responses.as_slice()),
        &mut output,
        expected(),
    )?;
    assert_eq!(client.count(&messages(), 3)?, 61);
    let mut text = String::new();
    let first = client.complete(
        &messages(),
        3,
        &Cancellation::default(),
        deadline(),
        |delta| {
            text.push_str(delta);
            Ok(())
        },
    )?;
    assert_eq!(text, "café 🌱");
    assert_eq!(first.finish, Finish::Stop);
    assert_eq!(first.generated_tokens, 2);
    assert_eq!(first.reused_tokens, 0);
    client.reset()?;
    assert_eq!(
        client
            .complete(
                &messages(),
                3,
                &Cancellation::default(),
                deadline(),
                |_| Ok(())
            )?
            .finish,
        Finish::Length
    );
    assert!(client.is_usable());
    let _ = client.into_parts();
    let requests = std::str::from_utf8(&output)?
        .lines()
        .map(serde_json::from_str::<Value>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    for index in [0, 1, 2, 4, 5] {
        assert_eq!(requests[index]["messages"], messages());
        assert_eq!(requests[index]["max_tokens"], 3);
    }
    assert_eq!(requests[1]["op"], "count");
    assert_eq!(requests[2]["op"], "complete");
    assert_eq!(requests[3], json!({"op":"reset","id":3}));
    assert!(
        (1..=600).contains(
            &requests[2]["timeout_seconds"]
                .as_u64()
                .ok_or("missing timeout")?
        )
    );
    Ok(())
}

#[test]
fn readiness_rejects_foreign_identity_effective_settings_and_duplicate_fields() -> Result<()> {
    for (key, value) in [
        ("model_sha256", json!("0".repeat(64))),
        ("template_sha256", json!("0".repeat(64))),
        ("prism_commit", json!("other")),
        ("seed", json!(18)),
        ("backend", json!("auto")),
        ("temperature", json!(2)),
        ("thinking", json!(false)),
        ("preserve_thinking", json!(false)),
        ("cache_reuse", json!(true)),
        ("vocab_size", json!(131_072)),
        ("max_frame_bytes", json!(42)),
        ("unexpected", json!(true)),
    ] {
        let mut frame = ready();
        frame[key] = value;
        let bytes = format!("{}\n", serde_json::to_string(&frame)?);
        assert!(
            SwiftClient::connect(Cursor::new(bytes), Vec::new(), expected()).is_err(),
            "accepted {key}"
        );
    }
    let duplicate = serde_json::to_string(&ready())?.replacen('{', "{\"seed\":17,", 1) + "\n";
    assert!(SwiftClient::connect(Cursor::new(duplicate), Vec::new(), expected()).is_err());
    let mut invalid = expected();
    invalid.seed = u32::MAX;
    assert!(matches!(
        SwiftClient::connect(Cursor::new(Vec::new()), Vec::new(), invalid),
        Err(ClientError::InvalidConfiguration)
    ));
    Ok(())
}

#[test]
fn malformed_wrong_id_count_and_terminal_frames_poison_connection() -> Result<()> {
    for bad in [
        json!({"type":"count","id":9,"prompt_tokens":61,"reserved_output_tokens":3}),
        json!({"type":"count","id":0,"prompt_tokens":8190,"reserved_output_tokens":3}),
        json!({"type":"count","id":0,"prompt_tokens":61,"reserved_output_tokens":4}),
    ] {
        let mut client =
            SwiftClient::connect(Cursor::new(script(&[bad])?), Vec::new(), expected())?;
        assert_eq!(
            client.count(&messages(), 3),
            Err(ClientError::InvalidResponse)
        );
        assert!(!client.is_usable());
        assert_eq!(client.reset(), Err(ClientError::Unavailable));
    }
    for bad in [
        terminal(1, 2, "length"),
        terminal(1, 3, "stop"),
        json!({"type":"complete","id":1,"prompt_tokens":62,"generated_tokens":1,"finish_reason":"stop"}),
        json!({"type":"delta","id":9,"text":"wrong"}),
        json!({"type":"delta","id":1,"text":""}),
        json!({"type":"delta","id":1,"text":"x".repeat(MAX_DELTA_BYTES+1)}),
        json!({"type":"error","id":1,"error":"generation_failed"}),
    ] {
        let mut client = SwiftClient::connect(
            Cursor::new(script(&[count(0), bad])?),
            Vec::new(),
            expected(),
        )?;
        assert_eq!(
            client.complete(
                &messages(),
                3,
                &Cancellation::default(),
                deadline(),
                |_| Ok(())
            ),
            Err(ClientError::InvalidResponse)
        );
        assert!(!client.is_usable());
    }
    for bad in [b"{\"type\":\"count\",\"id\":0,\"id\":0,\"prompt_tokens\":61,\"reserved_output_tokens\":3}\n".as_slice(),b"{\"type\":\"count\",\"id\":0,\"prompt_tokens\":61,\"reserved_output_tokens\":3}",b"[]\n",b"\xff\n"] {
        let mut bytes=script(&[])?;bytes.extend_from_slice(bad);
        let mut client=SwiftClient::connect(Cursor::new(bytes),Vec::new(),expected())?;
        assert_eq!(client.count(&messages(),3),Err(ClientError::InvalidResponse));assert!(!client.is_usable());
    }
    Ok(())
}

#[test]
fn rejected_input_and_checked_generation_failure_allow_reset_and_next_request() -> Result<()> {
    let frames = [
        json!({"type":"error","id":0,"error":"rejected_input"}),
        count(1),
        json!({"type":"delta","id":2,"text":"provisional"}),
        json!({"type":"error","id":2,"error":"generation_failed","reset_required":true}),
        json!({"type":"reset","id":3,"ready":false}),
        json!({"type":"reset","id":4,"ready":true}),
        count(5),
        terminal(6, 0, "stop"),
    ];
    let mut client = SwiftClient::connect(Cursor::new(script(&frames)?), Vec::new(), expected())?;
    assert_eq!(
        client.count(&messages(), 3),
        Err(ClientError::RejectedInput)
    );
    assert!(client.is_usable());
    assert_eq!(
        client.complete(
            &messages(),
            3,
            &Cancellation::default(),
            deadline(),
            |_| Ok(())
        ),
        Err(ClientError::GenerationFailed)
    );
    assert!(client.requires_reset());
    assert!(!client.is_usable());
    assert_eq!(
        client.count(&messages(), 3),
        Err(ClientError::ResetRequired)
    );
    assert_eq!(client.reset(), Err(ClientError::ResetFailed));
    assert!(client.requires_reset());
    client.reset()?;
    assert!(!client.requires_reset());
    assert!(client.is_usable());
    assert_eq!(
        client
            .complete(
                &messages(),
                3,
                &Cancellation::default(),
                deadline(),
                |_| Ok(())
            )?
            .generated_tokens,
        0
    );
    Ok(())
}

#[test]
fn callback_failure_or_cancellation_abandons_unread_stream_and_disables_reset() -> Result<()> {
    for cancel in [false, true] {
        let input = script(&[
            count(0),
            json!({"type":"delta","id":1,"text":"private answer"}),
            terminal(1, 1, "stop"),
        ])?;
        let cancellation = Cancellation::default();
        let mut client = SwiftClient::connect(Cursor::new(input), Vec::new(), expected())?;
        let error = client.complete(&messages(), 3, &cancellation, deadline(), |_| {
            if cancel {
                cancellation.cancel();
                Ok(())
            } else {
                Err(io::Error::other("private diagnostic").into())
            }
        });
        assert_eq!(
            error,
            Err(if cancel {
                ClientError::Cancelled
            } else {
                ClientError::DeliveryFailed
            })
        );
        assert!(!client.is_usable());
        assert_eq!(client.reset(), Err(ClientError::Unavailable));
        let (input, _) = client.into_parts();
        assert!(usize::try_from(input.position())? < input.get_ref().len());
    }
    Ok(())
}

#[test]
fn local_cancellation_deadline_and_request_bounds_do_not_send_or_consume_id() -> Result<()> {
    let mut output = Vec::new();
    let mut client =
        SwiftClient::connect(Cursor::new(script(&[count(0)])?), &mut output, expected())?;
    let cancellation = Cancellation::default();
    cancellation.cancel();
    assert_eq!(
        client.complete(&messages(), 3, &cancellation, deadline(), |_| Ok(())),
        Err(ClientError::Cancelled)
    );
    assert_eq!(
        client.complete(
            &messages(),
            3,
            &Cancellation::default(),
            Instant::now(),
            |_| Ok(())
        ),
        Err(ClientError::Deadline)
    );
    assert_eq!(
        client.complete(
            &messages(),
            3,
            &Cancellation::default(),
            Instant::now() + Duration::from_secs(601),
            |_| Ok(())
        ),
        Err(ClientError::InvalidConfiguration)
    );
    assert_eq!(
        client.count(&messages(), 4097),
        Err(ClientError::InvalidConfiguration)
    );
    let huge = json!([{"role":"user","content":"\u{0000}".repeat(200_000)}]);
    assert_eq!(client.count(&huge, 3), Err(ClientError::RequestTooLarge));
    assert!(client.is_usable());
    assert_eq!(client.count(&messages(), 3)?, 61);
    let _ = client.into_parts();
    assert_eq!(std::str::from_utf8(&output)?.lines().count(), 1);
    assert_eq!(serde_json::from_slice::<Value>(&output)?["id"], 0);
    Ok(())
}

struct FailedWriter;
impl Write for FailedWriter {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> {
        Err(io::Error::other("private transport"))
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn transport_and_total_stream_bounds_fail_closed_without_private_diagnostics() -> Result<()> {
    let mut client = SwiftClient::connect(Cursor::new(script(&[])?), FailedWriter, expected())?;
    assert_eq!(client.count(&messages(), 3), Err(ClientError::Transport));
    assert!(!client.is_usable());
    let mut responses = vec![count(0)];
    for _ in 0..=MAX_OUTPUT_BYTES / MAX_DELTA_BYTES {
        responses.push(json!({"type":"delta","id":1,"text":"x".repeat(MAX_DELTA_BYTES)}));
    }
    let mut client =
        SwiftClient::connect(Cursor::new(script(&responses)?), Vec::new(), expected())?;
    let mut delivered = 0usize;
    assert_eq!(
        client.complete(
            &messages(),
            3,
            &Cancellation::default(),
            deadline(),
            |text| {
                delivered += text.len();
                Ok(())
            }
        ),
        Err(ClientError::InvalidResponse)
    );
    assert_eq!(delivered, MAX_OUTPUT_BYTES);
    assert!(!client.is_usable());
    assert!(!ClientError::Transport.to_string().contains("private"));
    Ok(())
}

struct DelayedReader {
    input: Cursor<Vec<u8>>,
    reads: usize,
}
impl Read for DelayedReader {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        self.input.read(bytes)
    }
}
impl BufRead for DelayedReader {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        self.reads += 1;
        if self.reads == 3 {
            // Simulate an owned transport returning after its request deadline.
            std::thread::sleep(Duration::from_millis(100));
        }
        self.input.fill_buf()
    }
    fn consume(&mut self, amount: usize) {
        self.input.consume(amount);
    }
}

#[test]
fn expiry_after_blocking_response_read_and_oversized_frames_require_worker_shutdown() -> Result<()>
{
    let input = DelayedReader {
        input: Cursor::new(script(&[
            count(0),
            json!({"type":"delta","id":1,"text":"late output"}),
            terminal(1, 1, "stop"),
        ])?),
        reads: 0,
    };
    let mut client = SwiftClient::connect(input, Vec::new(), expected())?;
    let mut delivered = false;
    assert_eq!(
        client.complete(
            &messages(),
            3,
            &Cancellation::default(),
            Instant::now() + Duration::from_millis(50),
            |_| {
                delivered = true;
                Ok(())
            }
        ),
        Err(ClientError::Deadline)
    );
    assert!(!delivered);
    assert!(!client.is_usable());
    assert_eq!(client.reset(), Err(ClientError::Unavailable));

    let oversized = json!({"type":"error","id":0,"error":"x".repeat(DEFAULT_RESPONSE_FRAME_BYTES)});
    let mut client =
        SwiftClient::connect(Cursor::new(script(&[oversized])?), Vec::new(), expected())?;
    assert_eq!(
        client.count(&messages(), 3),
        Err(ClientError::InvalidResponse)
    );
    assert!(!client.is_usable());
    Ok(())
}
