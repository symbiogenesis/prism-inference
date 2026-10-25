//! Explicit pinned Swift JSONL worker with verified readiness and bounded requests.

#![forbid(unsafe_code)]

use std::{
    io::{self, BufRead, BufReader, Write},
    path::PathBuf,
    process::ExitCode,
    time::{Duration, Instant},
};

use bonsai_prism_runtime::{
    Result, kernels::MatvecBackend, model::swift, runtime::Cancellation,
    swift_session::SwiftSession,
};
use serde_json::json;

#[path = "../swift_worker_request.rs"]
mod request;
use request::{MAX_FRAME_BYTES, Request};
#[path = "../swift_worker_output.rs"]
mod delivery;
use delivery::{Delivery, write_record};
const USAGE: &[u8] = b"Usage: bonsai-swift-worker --model <absolute-pinned-swift-gguf> --seed <u32> [--backend <auto|scalar>]\nJSONL: {\"op\":\"count\",\"id\":1,\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":16}\nUse op=complete with timeout_seconds (1..600), or {\"op\":\"reset\",\"id\":2}.\nInput frames require LF and are bounded to 1 MiB. Output is flushed JSONL.\nDeltas are provisional until complete. Close the worker for transport cancellation;\nlibrary owners can also cancel cooperatively through the session Cancellation.\nExisting process admission and ownership policies must wrap this worker.\nFull-model correctness and device qualification require retained comparison evidence.\n";

fn bad() -> bonsai_prism_runtime::Error {
    io::Error::new(io::ErrorKind::InvalidInput, "invalid Swift worker options").into()
}

fn load() -> Result<SwiftSession> {
    let mut args = std::env::args_os().skip(1);
    let mut model = None;
    let mut seed = None;
    let mut backend = None;
    while let Some(key) = args.next() {
        let value = args.next().ok_or_else(bad)?;
        match key.to_str() {
            Some("--model") if model.is_none() => model = Some(PathBuf::from(value)),
            Some("--seed") if seed.is_none() => {
                seed = Some(value.to_str().ok_or_else(bad)?.parse::<u32>()?);
            }
            Some("--backend") if backend.is_none() => {
                backend = Some(match value.to_str() {
                    Some("auto") => MatvecBackend::Auto,
                    Some("scalar") => MatvecBackend::Scalar,
                    _ => return Err(bad()),
                });
            }
            _ => return Err(bad()),
        }
    }
    let model = model.filter(|path| path.is_absolute()).ok_or_else(bad)?;
    SwiftSession::load(
        &model,
        backend.unwrap_or(MatvecBackend::Auto),
        seed.ok_or_else(bad)?,
    )
}

fn frame(input: &mut impl BufRead, bytes: &mut Vec<u8>) -> Result<bool> {
    bytes.clear();
    loop {
        let available = input.fill_buf()?;
        if available.is_empty() {
            return if bytes.is_empty() {
                Ok(false)
            } else {
                Err(bad())
            };
        }
        let count = available
            .iter()
            .position(|&b| b == b'\n')
            .map_or(available.len(), |n| n + 1);
        if bytes
            .len()
            .checked_add(count)
            .is_none_or(|n| n > MAX_FRAME_BYTES)
        {
            return Err(bad());
        }
        bytes.try_reserve(count)?;
        bytes.extend_from_slice(&available[..count]);
        input.consume(count);
        if bytes.last() == Some(&b'\n') {
            return Ok(true);
        }
    }
}

fn write_readiness(session: &SwiftSession, output: &mut impl Write) -> Result<()> {
    let backend = match session.requested_backend() {
        MatvecBackend::Auto => "auto",
        MatvecBackend::Scalar => "scalar",
        MatvecBackend::ExperimentalNeon => "neon",
    };
    write_record(
        output,
        &json!({
            "type":"ready", "model_sha256":swift::MODEL_SHA256,
            "protocol":"bonsai-swift-worker-v1", "template_sha256":swift::TEMPLATE_SHA256,
            "vocab_size":swift::VOCAB_SIZE,
            "prism_commit":"73bfe6cc485554822f80c473a3c0c4f25e80f41b",
            "seed":session.seed(), "backend":backend,
            "max_frame_bytes":MAX_FRAME_BYTES,
            "max_output_bytes":bonsai_prism_runtime::swift_session::MAX_OUTPUT_BYTES,
            "runtime":"bonsai-prism-runtime", "context_tokens":swift::CONTEXT_TOKENS,
            "max_output_tokens":swift::MAX_OUTPUT_TOKENS,
            "temperature":1, "top_p":0.95, "top_k":20, "min_p":0,
            "repeat_penalty":1, "presence_penalty":0, "frequency_penalty":0,
            "thinking":true, "reasoning_effort":"xhigh", "preserve_thinking":true,
            "cache_reuse":false, "speculation":false, "adapters":false,
            "qualification":"pending-full-model-parity-and-device-evidence",
        }),
    )?;
    Ok(())
}

fn serve(
    session: &mut SwiftSession,
    input: &mut impl BufRead,
    output: &mut impl Write,
) -> Result<()> {
    write_readiness(session, output)?;
    let mut bytes = Vec::new();
    while frame(input, &mut bytes)? {
        let Ok(request) = request::parse(&bytes) else {
            write_record(
                output,
                &json!({"type":"error","id":null,"error":"invalid_request"}),
            )?;
            continue;
        };
        match request {
            Request::Reset { id } => {
                let success = session.reset().is_ok();
                write_record(output, &json!({"type":"reset","id":id,"ready":success}))?;
            }
            Request::Count {
                id,
                messages,
                max_tokens,
            } => match session.prepare(&messages, max_tokens) {
                Ok(prompt) => write_record(
                    output,
                    &json!({"type":"count","id":id,
                        "prompt_tokens":prompt.prompt_tokens(),"reserved_output_tokens":max_tokens}),
                )?,
                Err(_) => write_record(
                    output,
                    &json!({"type":"error","id":id,"error":"rejected_input"}),
                )?,
            },
            Request::Complete {
                id,
                messages,
                max_tokens,
                timeout_seconds,
            } => {
                let Some(deadline) = (1..=600)
                    .contains(&timeout_seconds)
                    .then(|| Instant::now().checked_add(Duration::from_secs(timeout_seconds)))
                    .flatten()
                else {
                    write_record(
                        output,
                        &json!({"type":"error","id":id,"error":"invalid_request"}),
                    )?;
                    continue;
                };
                let Ok(prompt) = session.prepare(&messages, max_tokens) else {
                    write_record(
                        output,
                        &json!({"type":"error","id":id,"error":"rejected_input"}),
                    )?;
                    continue;
                };
                let mut delivery = Delivery::new(output, id);
                let result =
                    session.complete(&prompt, &Cancellation::default(), deadline, |event| {
                        delivery.emit(&event)
                    });
                delivery.finish(result.is_ok(), session.is_faulted())?;
            }
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    if std::env::args_os().skip(1).collect::<Vec<_>>() == [std::ffi::OsString::from("--help")] {
        return if io::stdout().lock().write_all(USAGE).is_ok() {
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        };
    }
    let result = load().and_then(|mut session| {
        serve(
            &mut session,
            &mut BufReader::with_capacity(8192, io::stdin().lock()),
            &mut io::stdout().lock(),
        )
    });
    if result.is_ok() {
        ExitCode::SUCCESS
    } else {
        let _ = io::stderr().lock().write_all(b"Swift worker failed\n");
        ExitCode::FAILURE
    }
}
