//! Private bounded Swift worker delivery and single-terminal ownership.
#![allow(
    clippy::redundant_pub_crate,
    reason = "Keep shared worker/test delivery helpers explicitly crate scoped"
)]

use std::io::{self, Write};

use bonsai_prism_runtime::{Result, runtime::Finish, swift_session::TextEvent};
use serde_json::{Value, json};

/// Write and flush one complete JSONL protocol record.
pub(crate) fn write_record(output: &mut impl Write, value: &Value) -> Result<()> {
    serde_json::to_writer(&mut *output, value)?;
    output.write_all(b"\n")?;
    output.flush()?;
    Ok(())
}

fn write_delta(output: &mut impl Write, id: u64, mut text: &str) -> Result<()> {
    while !text.is_empty() {
        let mut end = text.len().min(16 * 1024);
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        write_record(output, &json!({"type":"delta","id":id,"text":&text[..end]}))?;
        text = &text[end..];
    }
    Ok(())
}

/// One generation's provisional output and terminal delivery state.
pub(crate) struct Delivery<'a, W> {
    output: &'a mut W,
    id: u64,
    terminal_sent: bool,
    transport_failed: bool,
}

impl<'a, W: Write> Delivery<'a, W> {
    /// Borrow the caller-owned worker output for exactly one generation.
    pub(crate) const fn new(output: &'a mut W, id: u64) -> Self {
        Self {
            output,
            id,
            terminal_sent: false,
            transport_failed: false,
        }
    }

    /// Deliver one bounded provisional delta or the sole terminal completion.
    pub(crate) fn emit(&mut self, event: &TextEvent<'_>) -> Result<()> {
        if self.terminal_sent || self.transport_failed {
            return Err(io::Error::other("Swift generation delivery already ended").into());
        }
        let terminal = matches!(event, TextEvent::Complete(_));
        let result = match event {
            TextEvent::Delta(text) => write_delta(self.output, self.id, text),
            TextEvent::Complete(completion) => write_record(
                self.output,
                &json!({
                    "type":"complete","id":self.id,"prompt_tokens":completion.prompt_tokens,
                    "generated_tokens":completion.generated_tokens,
                    "finish_reason":if completion.finish == Finish::Stop {"stop"} else {"length"},
                }),
            ),
        };
        self.transport_failed = result.is_err();
        self.terminal_sent = terminal && result.is_ok();
        result
    }

    /// Finish delivery after the session returns; an error closes the worker.
    ///
    /// A post-terminal deadline/cancellation failure cannot be reported as a
    /// second terminal record. Close the owned worker, leaving the session's
    /// fault latch intact. Only failures before terminal delivery may produce
    /// a synchronized `generation_failed` record and permit explicit reset.
    pub(crate) fn finish(self, succeeded: bool, reset_required: bool) -> Result<()> {
        if self.transport_failed || self.terminal_sent != succeeded {
            return Err(io::Error::other("Swift generation delivery cannot resume").into());
        }
        if !succeeded {
            write_record(
                self.output,
                &json!({"type":"error","id":self.id,
                "error":"generation_failed","reset_required":reset_required}),
            )?;
        }
        Ok(())
    }
}
