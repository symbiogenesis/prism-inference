#![cfg(feature = "cli")]
#![forbid(unsafe_code)]
//! Opt-in device correctness exercise; never an inference performance benchmark.

use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

use bonsai_prism_runtime::{
    Result,
    kernels::MatvecBackend,
    model::swift,
    runtime::{Cancellation, Finish},
    swift_session::{PreparedChat, SwiftSession, TextEvent},
};
use serde_json::json;

#[test]
#[ignore = "requires the 7.2 GB pinned Swift artifact and target-device memory admission"]
fn actual_swift_session_stream_cancel_deadline_reset_and_context() -> Result<()> {
    let path = PathBuf::from(
        std::env::var_os("BONSAI_SWIFT_MODEL_PATH").ok_or("BONSAI_SWIFT_MODEL_PATH is required")?,
    );
    let mut session = SwiftSession::load(&path, MatvecBackend::Auto, 42)?;
    assert_eq!(session.receipt().tensor_counts, [353, 96, 402]);
    assert_eq!(
        session.receipt().tensor_payload_bytes,
        swift::TENSOR_PAYLOAD_BYTES
    );
    let messages = json!([{"role":"user","content":"Hi"}]);
    let prompt = session.prepare(&messages, 1)?;
    assert!(prompt.prompt_tokens() > 0);
    assert_eq!(prompt.output_tokens(), 1);
    rejected_requests_leave_session_ready(&mut session, &prompt, &messages);

    let mut first_text = String::new();
    let first = session.complete(
        &prompt,
        &Cancellation::default(),
        Instant::now() + Duration::from_secs(600),
        |event| {
            if let TextEvent::Delta(text) = event {
                first_text.push_str(text);
            }
            Ok(())
        },
    )?;
    assert_eq!(first.reused_tokens, 0);
    assert_eq!(first.prompt_tokens, prompt.prompt_tokens());
    assert!(first.generated_tokens <= 1);
    assert!(first.finish == Finish::Stop || first.generated_tokens == 1);
    session.reset()?;
    let mut replay = String::new();
    let second = session.complete(
        &prompt,
        &Cancellation::default(),
        Instant::now() + Duration::from_secs(600),
        |event| {
            if let TextEvent::Delta(text) = event {
                replay.push_str(text);
            }
            Ok(())
        },
    )?;
    assert_eq!(first, second);
    assert_eq!(first_text, replay);

    // A failed final delivery must latch closed even though model evaluation
    // succeeded; reset is the only recovery before another request is admitted.
    session.reset()?;
    assert!(
        session
            .complete(
                &prompt,
                &Cancellation::default(),
                Instant::now() + Duration::from_secs(600),
                |event| {
                    if matches!(event, TextEvent::Complete(_)) {
                        return Err("abandoned final output".into());
                    }
                    Ok(())
                }
            )
            .is_err()
    );
    assert!(session.is_faulted());
    assert!(session.prepare(&messages, 1).is_err());
    session.reset()?;
    assert!(!session.is_faulted());
    let history = json!([
        {"role":"user","content":"Hi"},
        {"role":"assistant","content":"Hello","reasoning_content":"A greeting is appropriate."},
        {"role":"user","content":"Continue"},
    ]);
    let multi_turn = session.prepare(&history, 1)?;
    assert!(multi_turn.prompt_tokens() > prompt.prompt_tokens());
    let completion = session.complete(
        &multi_turn,
        &Cancellation::default(),
        Instant::now() + Duration::from_secs(600),
        |_| Ok(()),
    )?;
    assert_eq!(completion.reused_tokens, 0);
    assert!(!session.is_faulted());
    Ok(())
}

fn rejected_requests_leave_session_ready(
    session: &mut SwiftSession,
    prompt: &PreparedChat,
    messages: &serde_json::Value,
) {
    let cancellation = Cancellation::default();
    cancellation.cancel();
    assert!(
        session
            .complete(
                prompt,
                &cancellation,
                Instant::now() + Duration::from_secs(10),
                |_| { Err("cancelled request must emit nothing".into()) }
            )
            .is_err()
    );
    assert!(!session.is_faulted());
    assert!(
        session
            .complete(prompt, &Cancellation::default(), Instant::now(), |_| {
                Err("expired request must emit nothing".into())
            })
            .is_err()
    );
    assert!(!session.is_faulted());
    assert!(
        session
            .prepare(messages, swift::MAX_OUTPUT_TOKENS + 1)
            .is_err()
    );
    let overflow = json!([{"role":"user","content":"Hi ".repeat(8192)}]);
    assert!(
        session
            .prepare(&overflow, swift::MAX_OUTPUT_TOKENS)
            .is_err()
    );
}
