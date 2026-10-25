#![forbid(unsafe_code)]
#![cfg(feature = "cli")]
//! Focused actual worker writes, bounded UTF-8 chunks and single-terminal delivery.

#[path = "../src/swift_worker_output.rs"]
mod delivery;

use std::io::{self, Write};

use bonsai_prism_runtime::{
    Result,
    runtime::{Completion, Finish},
    swift_client::DEFAULT_RESPONSE_FRAME_BYTES,
    swift_session::TextEvent,
};
use delivery::Delivery;
use serde_json::{Value, json};

const fn completion() -> Completion {
    Completion {
        prompt_tokens: 61,
        reused_tokens: 0,
        generated_tokens: 2,
        finish: Finish::Stop,
    }
}
fn records(bytes: &[u8]) -> Result<Vec<Value>> {
    std::str::from_utf8(bytes)?
        .lines()
        .map(|line| serde_json::from_str(line).map_err(Into::into))
        .collect()
}

#[test]
fn preserves_utf8_and_json_escape_bounds_then_delivers_exactly_one_terminal() -> Result<()> {
    let unicode = "🦀".repeat(4097);
    let controls = "\u{0000}".repeat(16 * 1024);
    let mut output = Vec::new();
    let mut delivery = Delivery::new(&mut output, 7);
    delivery.emit(&TextEvent::Delta(&unicode))?;
    delivery.emit(&TextEvent::Delta(&controls))?;
    delivery.emit(&TextEvent::Complete(completion()))?;
    delivery.finish(true, false)?;
    let frames = records(&output)?;
    assert_eq!(frames.len(), 4);
    assert_eq!(
        frames[3],
        json!({"type":"complete","id":7,"prompt_tokens":61,"generated_tokens":2,"finish_reason":"stop"})
    );
    let mut displayed = String::new();
    for frame in &frames[..3] {
        let text = frame["text"].as_str().ok_or("missing delta")?;
        assert!(text.len() <= 16 * 1024);
        displayed.push_str(text);
    }
    assert_eq!(displayed, unicode + &controls);
    for line in output
        .split(|&byte| byte == b'\n')
        .filter(|line| !line.is_empty())
    {
        assert!(line.len() < DEFAULT_RESPONSE_FRAME_BYTES);
    }
    Ok(())
}

#[test]
fn late_session_failure_after_flushed_terminal_closes_without_second_record() -> Result<()> {
    let mut output = Vec::new();
    let mut delivery = Delivery::new(&mut output, 8);
    delivery.emit(&TextEvent::Delta("answer"))?;
    delivery.emit(&TextEvent::Complete(completion()))?;
    // The session observed expiry/cancellation after successful terminal delivery.
    assert!(delivery.finish(false, true).is_err());
    let frames = records(&output)?;
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[1]["type"], "complete");
    assert!(frames.iter().all(|frame| frame["type"] != "error"));
    Ok(())
}

#[test]
fn preterminal_failure_reports_reset_state_but_duplicate_terminal_cannot_resume() -> Result<()> {
    let mut output = Vec::new();
    let mut delivery = Delivery::new(&mut output, 9);
    delivery.emit(&TextEvent::Delta("provisional"))?;
    delivery.finish(false, true)?;
    assert_eq!(
        records(&output)?[1],
        json!({"type":"error","id":9,"error":"generation_failed","reset_required":true})
    );
    output.clear();
    let mut delivery = Delivery::new(&mut output, 10);
    delivery.emit(&TextEvent::Complete(completion()))?;
    assert!(delivery.emit(&TextEvent::Complete(completion())).is_err());
    assert!(delivery.finish(false, true).is_err());
    assert_eq!(records(&output)?.len(), 1);
    let delivery = Delivery::new(&mut output, 11);
    assert!(delivery.finish(true, false).is_err());
    assert_eq!(records(&output)?.len(), 1);
    Ok(())
}

struct FailedFlush(Vec<u8>);
impl Write for FailedFlush {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Err(io::Error::other("private transport failure"))
    }
}

#[test]
fn failed_terminal_flush_closes_without_an_attempted_error_record() -> Result<()> {
    let mut output = FailedFlush(Vec::new());
    let mut delivery = Delivery::new(&mut output, 12);
    assert!(delivery.emit(&TextEvent::Complete(completion())).is_err());
    assert!(delivery.finish(false, true).is_err());
    let frames = records(&output.0)?;
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0]["type"], "complete");
    Ok(())
}
