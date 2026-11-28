//! Persistent, metadata-only token counting over a bounded JSONL transport.
//!
//! [`serve`] takes an already verified [`TokenizationSession`], emits and flushes
//! one readiness record, then processes requests serially without reopening the
//! artifact. The session retains no weights, graph or artifact handle. Each
//! text request has exactly `id` (a JSON u64), `op` (`count` or `encode`) and
//! `text` (a UTF-8 string). Raw-ID requests use `count_tokens` or `encode_tokens`
//! and `token_ids` instead of `text`. Raw IDs preserve order and receive no
//! parsing or boundary insertion. Context accounting is fixed at startup.
//!
//! Duplicate and unknown fields, invalid UTF-8/JSON, blank lines and invalid
//! request shapes produce `invalid_request` with a null ID. Tokenizer or context
//! rejection produces `rejected_input` with the parsed ID. Neither error includes
//! input text or underlying diagnostics. Both errors leave the session reusable.
//! IDs are opaque correlation values, not a retained global uniqueness registry.
//! Count responses omit token IDs; encode responses include their exact order.
//!
//! Frames count every byte before LF, including a CR in CRLF. An unterminated
//! final frame is rejected as `invalid_request`, then EOF ends cleanly. A frame
//! beyond the configured bound, frame allocation failure or any transport I/O error
//! terminates serving with an error; no attempt is made to drain or resynchronize
//! an oversized frame. Writes can be partial on I/O failure. Clients must discard
//! an incomplete response and restart the process. Every complete response is
//! flushed, allowing clients to wait for readiness before sending any request.

use std::borrow::Cow;
use std::io::{self, BufRead, Read, Write};
use std::sync::mpsc;

use serde::{Deserialize, Serialize};

use crate::special_input::SpecialInputPolicy;
use crate::tokenization::{
    SpecialTokenInsertion, TextTokenizationPolicy, TokenBudget, TokenizationSession,
};
use crate::{Result, invalid};

/// Maximum admitted JSONL frame length, excluding its LF terminator (1 MiB).
pub const MAX_WORKER_FRAME_BYTES: usize = 1024 * 1024;

/// Immutable transport and complete-text policy for a worker session.
#[derive(Clone, Copy, Debug)]
pub struct WorkerConfig {
    /// Explicit tokenizer parsing, boundary insertion and context budget.
    pub text_policy: TextTokenizationPolicy,
    /// Maximum bytes before LF, from 1 through [`MAX_WORKER_FRAME_BYTES`].
    pub max_frame_bytes: usize,
}

impl WorkerConfig {
    /// Validate all resource bounds without reading or encoding input.
    ///
    /// # Errors
    /// Rejects invalid context/output accounting or frame byte bounds.
    pub fn validate(self) -> Result<()> {
        self.text_policy.budget.prompt_token_limit()?;
        if !(1..=MAX_WORKER_FRAME_BYTES).contains(&self.max_frame_bytes) {
            return Err(invalid("invalid token worker frame bound"));
        }
        Ok(())
    }
}

// Intentionally no Debug: requests contain caller-provided prompt text.
#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Request<'a> {
    Count {
        id: u64,
        #[serde(borrow)]
        text: Cow<'a, str>,
    },
    Encode {
        id: u64,
        #[serde(borrow)]
        text: Cow<'a, str>,
    },
    CountTokens {
        id: u64,
        token_ids: Vec<u32>,
    },
    EncodeTokens {
        id: u64,
        token_ids: Vec<u32>,
    },
}

impl Request<'_> {
    const fn id(&self) -> u64 {
        match *self {
            Self::Count { id, .. }
            | Self::Encode { id, .. }
            | Self::CountTokens { id, .. }
            | Self::EncodeTokens { id, .. } => id,
        }
    }
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Response<'a> {
    Ready {
        protocol: &'static str,
        model_sha256: &'a str,
        vocab_size: usize,
        context: usize,
        reserved_output_tokens: usize,
        max_frame_bytes: usize,
        special_input: &'static str,
        insert_special: &'static str,
    },
    Count {
        id: u64,
        prompt_token_count: usize,
    },
    Encoded {
        id: u64,
        prompt_token_count: usize,
        token_ids: &'a [u32],
    },
    Error {
        id: Option<u64>,
        code: &'static str,
    },
}

/// Serve strict JSONL requests using one immutable, artifact-verified tokenizer.
///
/// Readiness reports `bonsai-token-worker-v1`, the verified lowercase SHA-256,
/// vocabulary size and every fixed transport/text/context option. Input buffering
/// retains at most `max_frame_bytes` bytes plus the caller's bounded reader buffer.
/// Decoded text and raw-ID arrays are bounded by their frame; admitted token
/// output is bounded by the prompt allowance. Raw IDs must each be in the
/// verified vocabulary, including on count-only requests. The text parsing and
/// insertion policies never apply to raw IDs, including empty input.
/// The underlying tokenizer's additional piece/work bounds still apply; these
/// are resource contracts, not a process resident-memory or CPU-time guarantee.
/// The worker never emits prompts, tokenizer diagnostics or generated text.
///
/// # Errors
/// Rejects invalid configuration before readiness; allocation, oversized frames
/// and transport I/O errors terminate serving. Request errors emit safe records
/// and continue. An empty stream cleanly returns after its readiness record.
pub fn serve(
    session: &TokenizationSession,
    config: WorkerConfig,
    input: &mut impl BufRead,
    output: &mut impl Write,
) -> Result<()> {
    config.validate()?;
    let TokenBudget {
        context,
        reserved_output_tokens,
    } = config.text_policy.budget;
    let ready = Response::Ready {
        protocol: "bonsai-token-worker-v1",
        model_sha256: session.identity(),
        vocab_size: session.vocab_size(),
        context,
        reserved_output_tokens,
        max_frame_bytes: config.max_frame_bytes,
        special_input: match config.text_policy.special_input {
            SpecialInputPolicy::Literal => "literal",
            SpecialInputPolicy::Parse => "parse",
        },
        insert_special: match config.text_policy.insert_special {
            SpecialTokenInsertion::None => "none",
            SpecialTokenInsertion::Model => "model",
        },
    };
    let mut frame = Vec::new();
    frame.try_reserve_exact(config.max_frame_bytes)?;
    write_record(output, &ready)?;
    loop {
        match read_frame(input, &mut frame, config.max_frame_bytes)? {
            Frame::Eof => return Ok(()),
            Frame::Unterminated => {
                write_invalid_request(output)?;
                return Ok(());
            }
            Frame::Complete => (),
        }
        if frame.trim_ascii_start().first() != Some(&b'{') {
            write_invalid_request(output)?;
            continue;
        }
        let Ok(request) = serde_json::from_slice::<Request<'_>>(&frame) else {
            write_invalid_request(output)?;
            continue;
        };
        let result = match &request {
            Request::Count { text, .. } | Request::Encode { text, .. } => session
                .encode(text, config.text_policy)
                .map(|result| Cow::Owned(result.token_ids)),
            Request::CountTokens { token_ids, .. } | Request::EncodeTokens { token_ids, .. } => {
                session
                    .validate_tokens(token_ids, config.text_policy.budget)
                    .map(|()| Cow::Borrowed(token_ids.as_slice()))
            }
        };
        let Ok(token_ids) = result else {
            write_record(
                output,
                &Response::Error {
                    id: Some(request.id()),
                    code: "rejected_input",
                },
            )?;
            continue;
        };
        let response = match &request {
            Request::Count { id, .. } | Request::CountTokens { id, .. } => Response::Count {
                id: *id,
                prompt_token_count: token_ids.len(),
            },
            Request::Encode { id, .. } | Request::EncodeTokens { id, .. } => Response::Encoded {
                id: *id,
                prompt_token_count: token_ids.len(),
                token_ids: &token_ids,
            },
        };
        write_record(output, &response)?;
    }
}

enum Frame {
    Eof,
    Complete,
    Unterminated,
}

fn read_frame(input: &mut impl BufRead, frame: &mut Vec<u8>, limit: usize) -> Result<Frame> {
    frame.clear();
    loop {
        let available = match input.fill_buf() {
            Ok(available) => available,
            Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
            Err(err) => return Err(err.into()),
        };
        if available.is_empty() {
            return Ok(if frame.is_empty() {
                Frame::Eof
            } else {
                Frame::Unterminated
            });
        }
        let newline = available.iter().position(|&byte| byte == b'\n');
        let count = newline.unwrap_or(available.len());
        if count > limit - frame.len() {
            return Err(invalid("token worker frame exceeds byte limit"));
        }
        frame.extend_from_slice(&available[..count]);
        input.consume(count + usize::from(newline.is_some()));
        if newline.is_some() {
            return Ok(Frame::Complete);
        }
    }
}

/// Reads one LF-terminated JSONL frame up to `max_frame_bytes` (including LF), retrying `EINTR`.
///
/// Returns `Ok(false)` on clean EOF before any bytes arrive and `Ok(true)` when a full `\n`-ended
/// frame is buffered into `bytes`. Rejects truncated trailing bytes at EOF or frames exceeding
/// `max_frame_bytes` with `err_msg`.
///
/// # Errors
/// Propagates non-interrupt reader I/O errors, allocation failures, oversized frames, and
/// unterminated trailing input.
pub fn read_bounded_jsonl_frame(
    input: &mut impl BufRead,
    bytes: &mut Vec<u8>,
    max_frame_bytes: usize,
    err_msg: &str,
) -> Result<bool> {
    bytes.clear();
    loop {
        let available = match input.fill_buf() {
            Ok(available) => available,
            Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
            Err(err) => return Err(err.into()),
        };
        if available.is_empty() {
            return if bytes.is_empty() {
                Ok(false)
            } else {
                Err(invalid(err_msg))
            };
        }
        let count = available
            .iter()
            .position(|&b| b == b'\n')
            .map_or(available.len(), |n| n + 1);
        if bytes
            .len()
            .checked_add(count)
            .is_none_or(|n| n > max_frame_bytes)
        {
            return Err(invalid(err_msg));
        }
        bytes.try_reserve(count)?;
        bytes.extend_from_slice(&available[..count]);
        input.consume(count);
        if bytes.last() == Some(&b'\n') {
            return Ok(true);
        }
    }
}

/// Channel-backed [`Read`] adapter whose reader thread retries `EINTR` without dropping chunks.
pub struct ChannelReader {
    rx: mpsc::Receiver<io::Result<Vec<u8>>>,
    current: Vec<u8>,
    offset: usize,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl ChannelReader {
    fn pump_reader<R: Read>(mut reader: R, tx: &mpsc::SyncSender<io::Result<Vec<u8>>>) {
        let mut buf = [0_u8; 8192];
        loop {
            match reader.read(&mut buf) {
                Ok(0) => break,
                Ok(read) => {
                    if tx.send(Ok(buf[..read].to_vec())).is_err() {
                        break;
                    }
                }
                Err(err) if err.kind() == io::ErrorKind::Interrupted => {}
                Err(err) => {
                    let _ = tx.send(Err(err));
                    break;
                }
            }
        }
    }

    /// Spawns a background reader thread pumping [`io::stdin`] into a bounded sync channel.
    ///
    /// # Errors
    /// Returns any thread-spawning I/O error from [`std::thread::Builder::spawn`].
    pub fn spawn_stdin() -> io::Result<Self> {
        let (tx, rx) = mpsc::sync_channel::<io::Result<Vec<u8>>>(4);
        let handle = std::thread::Builder::new()
            .name("token-stdin".to_owned())
            .spawn(move || {
                Self::pump_reader(io::stdin().lock(), &tx);
            })?;
        Ok(Self {
            rx,
            current: Vec::new(),
            offset: 0,
            handle: Some(handle),
        })
    }

    /// Spawns a named background thread pumping `reader` while retrying `ErrorKind::Interrupted`.
    ///
    /// # Errors
    /// Returns any thread-spawning I/O error from [`std::thread::Builder::spawn`].
    pub fn spawn_from_reader<R: Read + Send + 'static>(reader: R) -> io::Result<Self> {
        let (tx, rx) = mpsc::sync_channel::<io::Result<Vec<u8>>>(4);
        let handle = std::thread::Builder::new()
            .name("worker-channel-reader".to_owned())
            .spawn(move || {
                Self::pump_reader(reader, &tx);
            })?;
        Ok(Self {
            rx,
            current: Vec::new(),
            offset: 0,
            handle: Some(handle),
        })
    }
}

impl Read for ChannelReader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if out.is_empty() {
            return Ok(0);
        }
        while self.offset >= self.current.len() {
            let Ok(chunk) = self.rx.recv() else {
                return Ok(0);
            };
            self.current = chunk?;
            self.offset = 0;
        }
        let available = &self.current[self.offset..];
        let count = available.len().min(out.len());
        out[..count].copy_from_slice(&available[..count]);
        self.offset += count;
        Ok(count)
    }
}

impl Drop for ChannelReader {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take()
            && handle.is_finished()
        {
            let _ = handle.join();
        }
    }
}

fn write_invalid_request(output: &mut impl Write) -> Result<()> {
    write_record(
        output,
        &Response::Error {
            id: None,
            code: "invalid_request",
        },
    )
}

fn write_record(output: &mut impl Write, response: &Response<'_>) -> Result<()> {
    serde_json::to_writer(&mut *output, response)?;
    output.write_all(b"\n")?;
    output.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Cow, Frame, Request, read_frame};
    use crate::{Result, invalid};
    use std::io::{self, BufRead, Read};

    struct InterruptingReader<'a> {
        bytes: &'a [u8],
        pos: usize,
        interrupted_once: bool,
    }

    impl Read for InterruptingReader<'_> {
        fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
            let avail = &self.bytes[self.pos..];
            let count = out.len().min(avail.len());
            out[..count].copy_from_slice(&avail[..count]);
            self.pos += count;
            Ok(count)
        }
    }

    impl BufRead for InterruptingReader<'_> {
        fn fill_buf(&mut self) -> io::Result<&[u8]> {
            if !self.interrupted_once {
                self.interrupted_once = true;
                return Err(io::Error::new(io::ErrorKind::Interrupted, "eintr"));
            }
            Ok(&self.bytes[self.pos..])
        }

        fn consume(&mut self, amt: usize) {
            self.pos += amt;
        }
    }

    #[test]
    fn read_frame_retries_eintr_before_completing_frame() -> Result<()> {
        let mut reader = InterruptingReader {
            bytes: br#"{"id":1,"op":"count","text":"hi"}
"#,
            pos: 0,
            interrupted_once: false,
        };
        let mut frame = Vec::new();
        assert!(matches!(
            read_frame(&mut reader, &mut frame, 128)?,
            Frame::Complete
        ));
        assert_eq!(frame, br#"{"id":1,"op":"count","text":"hi"}"#);
        Ok(())
    }

    #[test]
    fn text_requests_borrow_plain_frames_and_decode_escaped_frames() -> Result<()> {
        for bytes in [
            br#"{"id":1,"op":"count","text":"ab"}"#.as_slice(),
            br#"{"id":1,"op":"encode","text":"ab"}"#,
        ] {
            let request: Request<'_> = serde_json::from_slice(bytes)?;
            match request {
                Request::Count { text, .. } | Request::Encode { text, .. } => {
                    assert!(matches!(text, Cow::Borrowed("ab")));
                }
                _ => return Err(invalid("expected a text request")),
            }
        }
        for bytes in [
            br#"{"id":1,"op":"count","text":"\u0061b"}"#.as_slice(),
            br#"{"id":1,"op":"encode","text":"a\u0062"}"#,
        ] {
            let request: Request<'_> = serde_json::from_slice(bytes)?;
            match request {
                Request::Count { text, .. } | Request::Encode { text, .. } => {
                    assert!(matches!(&text, Cow::Owned(_)));
                    assert_eq!(text, "ab");
                }
                _ => return Err(invalid("expected a text request")),
            }
        }
        Ok(())
    }

    #[test]
    fn read_bounded_jsonl_frame_retries_eintr_and_rejects_oversized_or_unterminated() -> Result<()>
    {
        struct SplitEintrBuf<'a> {
            chunks: &'a [&'a [u8]],
            idx: usize,
            calls: usize,
        }
        impl Read for SplitEintrBuf<'_> {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                Ok(0)
            }
        }
        impl BufRead for SplitEintrBuf<'_> {
            fn fill_buf(&mut self) -> io::Result<&[u8]> {
                self.calls += 1;
                if self.calls % 2 == 1 {
                    return Err(io::Error::new(io::ErrorKind::Interrupted, "eintr"));
                }
                Ok(self.chunks.get(self.idx).copied().unwrap_or(&[]))
            }
            fn consume(&mut self, amt: usize) {
                if amt > 0 {
                    self.idx += 1;
                }
            }
        }

        let mut frame = Vec::new();
        let mut r = SplitEintrBuf {
            chunks: &[b"{\"op\":", b"\"reset\"}\n"],
            idx: 0,
            calls: 0,
        };
        assert!(super::read_bounded_jsonl_frame(
            &mut r, &mut frame, 64, "bad"
        )?);
        assert_eq!(frame, b"{\"op\":\"reset\"}\n");
        assert!(!super::read_bounded_jsonl_frame(
            &mut r, &mut frame, 64, "bad"
        )?);

        let mut partial = SplitEintrBuf {
            chunks: &[b"unterminated"],
            idx: 0,
            calls: 0,
        };
        assert!(super::read_bounded_jsonl_frame(&mut partial, &mut frame, 64, "bad").is_err());

        let mut oversize = SplitEintrBuf {
            chunks: &[b"{\"op\":\"too_long\"}\n"],
            idx: 0,
            calls: 0,
        };
        assert!(super::read_bounded_jsonl_frame(&mut oversize, &mut frame, 8, "bad").is_err());
        Ok(())
    }

    #[test]
    fn channel_reader_retries_eintr_preserves_chunks_and_propagates_errors() -> io::Result<()> {
        struct EintrSteps(Vec<io::Result<Vec<u8>>>, usize);
        impl Read for EintrSteps {
            fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
                if self.1 >= self.0.len() {
                    return Ok(0);
                }
                let item = std::mem::replace(&mut self.0[self.1], Ok(Vec::new()));
                self.1 += 1;
                match item {
                    Ok(bytes) => {
                        let n = bytes.len().min(out.len());
                        out[..n].copy_from_slice(&bytes[..n]);
                        Ok(n)
                    }
                    Err(e) => Err(e),
                }
            }
        }

        let mut cr = super::ChannelReader::spawn_from_reader(EintrSteps(
            vec![
                Err(io::Error::new(io::ErrorKind::Interrupted, "e1")),
                Ok(b"hello ".to_vec()),
                Err(io::Error::new(io::ErrorKind::Interrupted, "e2")),
                Ok(b"world\n".to_vec()),
            ],
            0,
        ))?;
        assert_eq!(cr.read(&mut [])?, 0);
        let mut out = Vec::new();
        cr.read_to_end(&mut out)?;
        assert_eq!(out, b"hello world\n");

        let mut failing = super::ChannelReader::spawn_from_reader(EintrSteps(
            vec![
                Err(io::Error::new(io::ErrorKind::Interrupted, "e3")),
                Err(io::Error::new(io::ErrorKind::BrokenPipe, "pipe")),
            ],
            0,
        ))?;
        let mut sink = [0_u8; 8];
        let err = failing.read(&mut sink);
        assert!(matches!(err, Err(e) if e.kind() == io::ErrorKind::BrokenPipe));
        Ok(())
    }
}
