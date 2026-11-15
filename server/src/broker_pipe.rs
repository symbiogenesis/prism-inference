//! Framed sequential stream multiplexer over parent-owned anonymous pipes.
//!
//! Used by `--broker-pipe` inference children inside an offline `AppContainer`.
//! Performs an initial in-memory bearer-key handshake before model loading,
//! emits a server-ready control frame once model initialization completes, and
//! multiplexes sequential HTTP/1.1 exchanges with isolated stream reset for
//! in-place generation cancellation.
#![forbid(unsafe_code)]
#![expect(
    clippy::redundant_pub_crate,
    reason = "Explicit crate visibility documents internal module boundary."
)]

use std::{
    collections::VecDeque,
    io::{self, Read, Write},
    pin::Pin,
    sync::{
        Arc, Condvar, Mutex, MutexGuard,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll, Waker},
};
use tokio::{
    io::{AsyncRead, AsyncWrite, ReadBuf},
    sync::mpsc,
};

/// Fixed host-header authority port used when serving requests over broker pipes.
pub(crate) const BROKER_PIPE_HOST_PORT: u16 = 1024;

const HANDSHAKE_REQUEST_MAGIC: &[u8; 8] = b"SB2PIPE1";
const HANDSHAKE_RESPONSE_MAGIC: &[u8; 8] = b"SB2OKAY1";
const KEY_HEX_BYTES: usize = 64;
const HANDSHAKE_BYTES: usize = 8 + KEY_HEX_BYTES;
const FRAME_HEADER_BYTES: usize = 9;
const MAX_FRAME_PAYLOAD: usize = 16_384;
const CHANNEL_CAPACITY: usize = 64;
const MAX_TRACKED_RESETS: usize = 16;

const KIND_OPEN: u8 = 1;
const KIND_DATA: u8 = 2;
const KIND_EOF: u8 = 3;
const KIND_RESET: u8 = 4;
const KIND_SERVER_READY: u8 = 5;

static STDIO_CLAIMED: AtomicBool = AtomicBool::new(false);

fn scrub_bytes(bytes: &mut [u8]) {
    for byte in bytes {
        *byte = std::hint::black_box(0);
    }
    std::sync::atomic::compiler_fence(Ordering::SeqCst);
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn wait_unpoisoned<'a, T>(condvar: &Condvar, guard: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
    match condvar.wait(guard) {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

struct ScrubSlice<'a>(&'a mut [u8]);

impl Drop for ScrubSlice<'_> {
    fn drop(&mut self) {
        scrub_bytes(self.0);
    }
}

/// Heap-owned 64-byte lowercase hexadecimal credential scrubbed on drop.
pub(crate) struct ScrubbedKey(Box<[u8; KEY_HEX_BYTES]>);

impl ScrubbedKey {
    /// Borrows the validated lowercase ASCII hex key text.
    #[must_use]
    pub(crate) fn as_str(&self) -> &str {
        // Every byte was validated as ASCII hex during handshake intake.
        std::str::from_utf8(&*self.0).unwrap_or_default()
    }
}

impl Drop for ScrubbedKey {
    fn drop(&mut self) {
        scrub_bytes(self.0.as_mut_slice());
    }
}

struct OutgoingFrame {
    stream_id: u32,
    kind: u8,
    payload: Vec<u8>,
}

struct TxState {
    queue: VecDeque<OutgoingFrame>,
    queued_data_frames: usize,
    enqueued_seq: u64,
    flushed_seq: u64,
    closed: bool,
    failed: bool,
    reset_streams: VecDeque<u32>,
    dropped_reset_data: VecDeque<u32>,
    write_waker: Option<Waker>,
    flush_waker: Option<Waker>,
}

impl TxState {
    const fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            queued_data_frames: 0,
            enqueued_seq: 0,
            flushed_seq: 0,
            closed: false,
            failed: false,
            reset_streams: VecDeque::new(),
            dropped_reset_data: VecDeque::new(),
            write_waker: None,
            flush_waker: None,
        }
    }

    fn is_reset(&self, stream_id: u32) -> bool {
        self.reset_streams.contains(&stream_id)
    }

    fn has_dropped_data(&self, stream_id: u32) -> bool {
        self.dropped_reset_data.contains(&stream_id)
    }

    fn record_dropped_data(&mut self, stream_id: u32) {
        if !self.has_dropped_data(stream_id) {
            if self.dropped_reset_data.len() >= MAX_TRACKED_RESETS {
                let _ = self.dropped_reset_data.pop_front();
            }
            self.dropped_reset_data.push_back(stream_id);
        }
    }

    fn record_reset(&mut self, stream_id: u32) {
        if !self.is_reset(stream_id) {
            if self.reset_streams.len() >= MAX_TRACKED_RESETS {
                let _ = self.reset_streams.pop_front();
            }
            self.reset_streams.push_back(stream_id);
        }
        let mut retained = VecDeque::with_capacity(self.queue.len());
        let mut removed_data = 0_usize;
        for frame in self.queue.drain(..) {
            if frame.stream_id == stream_id && matches!(frame.kind, KIND_DATA | KIND_EOF) {
                if frame.kind == KIND_DATA {
                    removed_data = removed_data.saturating_add(1);
                }
            } else {
                retained.push_back(frame);
            }
        }
        self.queue = retained;
        self.queued_data_frames = self.queued_data_frames.saturating_sub(removed_data);
        if removed_data > 0 {
            self.record_dropped_data(stream_id);
        }
    }

    fn wake_io(&mut self) {
        if let Some(waker) = self.write_waker.take() {
            waker.wake();
        }
        if let Some(waker) = self.flush_waker.take() {
            waker.wake();
        }
    }
}

struct TxShared {
    state: Mutex<TxState>,
    writer_ready: Condvar,
}

impl TxShared {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(TxState::new()),
            writer_ready: Condvar::new(),
        })
    }
}

struct TxCloser(Arc<TxShared>);

impl Drop for TxCloser {
    fn drop(&mut self) {
        let mut state = lock_unpoisoned(&self.0.state);
        state.closed = true;
        state.wake_io();
        drop(state);
        self.0.writer_ready.notify_one();
    }
}

/// Child-side acceptor for sequential broker-pipe streams.
pub(crate) struct ServerEndpoint {
    accept_rx: mpsc::Receiver<PipeConnection>,
    tx: Arc<TxShared>,
    _closer: Arc<TxCloser>,
}

impl ServerEndpoint {
    /// Claims process `stdin`/`stdout` once and completes the broker-pipe handshake.
    ///
    /// # Errors
    /// Rejects duplicate claiming, truncated/invalid handshake magic or key, or
    /// I/O failures before model loading begins.
    pub(crate) fn accept_stdio() -> io::Result<(Self, ScrubbedKey)> {
        if STDIO_CLAIMED.swap(true, Ordering::AcqRel) {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "broker-pipe stdio already claimed",
            ));
        }
        Self::from_io(io::stdin(), io::stdout())
    }

    /// Completes the broker-pipe handshake and spawns framed I/O threads over the supplied streams.
    ///
    /// # Errors
    /// Rejects malformed handshakes, invalid 64-byte lowercase hex credentials,
    /// thread creation failures, or underlying I/O errors.
    pub(crate) fn from_io<R, W>(mut reader: R, mut writer: W) -> io::Result<(Self, ScrubbedKey)>
    where
        R: Read + Send + 'static,
        W: Write + Send + 'static,
    {
        let mut raw = Box::new([0_u8; HANDSHAKE_BYTES]);
        let key = {
            let guard = ScrubSlice(raw.as_mut_slice());
            reader.read_exact(guard.0)?;
            if &guard.0[..8] != HANDSHAKE_REQUEST_MAGIC {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid broker-pipe handshake magic",
                ));
            }
            let raw_key = &guard.0[8..];
            if !crate::configuration::valid_key(raw_key) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid broker-pipe bearer credential",
                ));
            }
            let mut key = Box::new([0_u8; KEY_HEX_BYTES]);
            key.copy_from_slice(raw_key);
            ScrubbedKey(key)
        };
        writer.write_all(HANDSHAKE_RESPONSE_MAGIC)?;
        writer.flush()?;

        let tx = TxShared::new();
        let closer = Arc::new(TxCloser(Arc::clone(&tx)));
        let (accept_tx, accept_rx) = mpsc::channel(1);

        let rx_tx = Arc::clone(&tx);
        let rx_closer = Arc::clone(&closer);
        std::thread::Builder::new()
            .name("broker-pipe-rx".to_owned())
            .spawn(move || run_reader(reader, &accept_tx, &rx_tx, &rx_closer))
            .map_err(|_| io::Error::other("failed to start broker-pipe reader thread"))?;

        let tx_worker = Arc::clone(&tx);
        std::thread::Builder::new()
            .name("broker-pipe-tx".to_owned())
            .spawn(move || run_writer(writer, &tx_worker))
            .map_err(|_| io::Error::other("failed to start broker-pipe writer thread"))?;

        Ok((
            Self {
                accept_rx,
                tx,
                _closer: closer,
            },
            key,
        ))
    }

    /// Emits `KIND_SERVER_READY` once model loading and worker startup finish.
    ///
    /// # Errors
    /// Returns `BrokenPipe` if the broker pipe has already failed or closed.
    pub(crate) fn mark_ready(&self) -> io::Result<()> {
        let mut state = lock_unpoisoned(&self.tx.state);
        if state.closed || state.failed {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "broker pipe closed before readiness",
            ));
        }
        state.enqueued_seq = state.enqueued_seq.saturating_add(1);
        state.queue.push_back(OutgoingFrame {
            stream_id: 0,
            kind: KIND_SERVER_READY,
            payload: Vec::new(),
        });
        drop(state);
        self.tx.writer_ready.notify_one();
        Ok(())
    }

    /// Waits for the next logical broker stream, returning `None` on clean parent EOF.
    ///
    /// # Errors
    /// Returns `InvalidData` / `BrokenPipe` if the parent protocol stream failed.
    pub(crate) async fn accept(&mut self) -> io::Result<Option<PipeConnection>> {
        if let Some(conn) = self.accept_rx.recv().await {
            return Ok(Some(conn));
        }
        let failed = lock_unpoisoned(&self.tx.state).failed;
        if failed {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "broker pipe protocol terminated with an error",
            ))
        } else {
            Ok(None)
        }
    }
}

/// One sequential logical stream transported over the child's anonymous pipes.
pub(crate) struct PipeConnection {
    stream_id: u32,
    rx: mpsc::Receiver<io::Result<Vec<u8>>>,
    buffered: Vec<u8>,
    offset: usize,
    tx: Arc<TxShared>,
    closer: Arc<TxCloser>,
    eof_enqueued: bool,
    pending_data_seq: u64,
    pending_eof_seq: u64,
}

impl Drop for PipeConnection {
    fn drop(&mut self) {
        if !self.eof_enqueued {
            let mut state = lock_unpoisoned(&self.tx.state);
            if !state.closed && !state.failed && !state.is_reset(self.stream_id) {
                state.record_reset(self.stream_id);
                state.enqueued_seq = state.enqueued_seq.saturating_add(1);
                state.queue.push_back(OutgoingFrame {
                    stream_id: self.stream_id,
                    kind: KIND_RESET,
                    payload: Vec::new(),
                });
                drop(state);
                self.tx.writer_ready.notify_one();
            }
        }
        let _ = &self.closer;
    }
}

impl AsyncRead for PipeConnection {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        if buf.remaining() == 0 {
            return Poll::Ready(Ok(()));
        }
        while self.offset >= self.buffered.len() {
            self.buffered.clear();
            self.offset = 0;
            match self.rx.poll_recv(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => return Poll::Ready(Ok(())),
                Poll::Ready(Some(Err(error))) => return Poll::Ready(Err(error)),
                Poll::Ready(Some(Ok(chunk))) => {
                    if !chunk.is_empty() {
                        self.buffered = chunk;
                    }
                }
            }
        }
        let available = &self.buffered[self.offset..];
        let count = available.len().min(buf.remaining());
        buf.put_slice(&available[..count]);
        self.offset += count;
        if self.offset >= self.buffered.len() {
            self.buffered.clear();
            self.offset = 0;
        }
        Poll::Ready(Ok(()))
    }
}

impl AsyncWrite for PipeConnection {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        if buf.is_empty() {
            return Poll::Ready(Ok(0));
        }
        if self.eof_enqueued {
            return Poll::Ready(Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "write after broker-pipe shutdown",
            )));
        }
        let seq = {
            let mut state = lock_unpoisoned(&self.tx.state);
            if state.closed || state.failed || state.is_reset(self.stream_id) {
                return Poll::Ready(Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "broker-pipe stream closed",
                )));
            }
            if state.queued_data_frames >= CHANNEL_CAPACITY {
                state.write_waker = Some(cx.waker().clone());
                return Poll::Pending;
            }
            let count = buf.len().min(MAX_FRAME_PAYLOAD);
            state.enqueued_seq = state.enqueued_seq.saturating_add(1);
            let seq = state.enqueued_seq;
            state.queued_data_frames = state.queued_data_frames.saturating_add(1);
            state.queue.push_back(OutgoingFrame {
                stream_id: self.stream_id,
                kind: KIND_DATA,
                payload: buf[..count].to_vec(),
            });
            drop(state);
            self.tx.writer_ready.notify_one();
            (count, seq)
        };
        self.pending_data_seq = seq.1;
        Poll::Ready(Ok(seq.0))
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let mut state = lock_unpoisoned(&self.tx.state);
        if state.failed || state.has_dropped_data(self.stream_id) {
            return Poll::Ready(Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "broker-pipe flush failed",
            )));
        }
        let target_seq = if state.is_reset(self.stream_id) {
            self.pending_data_seq
        } else {
            self.pending_data_seq.max(self.pending_eof_seq)
        };
        if state.flushed_seq >= target_seq {
            return Poll::Ready(Ok(()));
        }
        if state.closed {
            return Poll::Ready(Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "broker-pipe closed before flush",
            )));
        }
        state.flush_waker = Some(cx.waker().clone());
        Poll::Pending
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        if !self.eof_enqueued {
            let mut state = lock_unpoisoned(&self.tx.state);
            if state.failed || state.closed || state.has_dropped_data(self.stream_id) {
                drop(state);
                self.eof_enqueued = true;
                return Poll::Ready(Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "broker-pipe shutdown failed",
                )));
            }
            if state.is_reset(self.stream_id) {
                drop(state);
                self.eof_enqueued = true;
                return self.poll_flush(cx);
            }
            state.enqueued_seq = state.enqueued_seq.saturating_add(1);
            let seq = state.enqueued_seq;
            state.queue.push_back(OutgoingFrame {
                stream_id: self.stream_id,
                kind: KIND_EOF,
                payload: Vec::new(),
            });
            drop(state);
            self.pending_eof_seq = seq;
            self.eof_enqueued = true;
            self.tx.writer_ready.notify_one();
        }
        self.poll_flush(cx)
    }
}

fn encode_header(stream_id: u32, kind: u8, payload_len: u32) -> [u8; FRAME_HEADER_BYTES] {
    let mut header = [0_u8; FRAME_HEADER_BYTES];
    header[..4].copy_from_slice(&stream_id.to_be_bytes());
    header[4] = kind;
    header[5..9].copy_from_slice(&payload_len.to_be_bytes());
    header
}

fn decode_header(header: [u8; FRAME_HEADER_BYTES]) -> (u32, u8, usize) {
    let stream_id = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
    let kind = header[4];
    let len = u32::from_be_bytes([header[5], header[6], header[7], header[8]]);
    let len_usize = usize::try_from(len).unwrap_or(usize::MAX);
    (stream_id, kind, len_usize)
}

enum HeaderRead {
    Frame([u8; FRAME_HEADER_BYTES]),
    CleanEof,
}

fn read_header<R: Read>(reader: &mut R) -> io::Result<HeaderRead> {
    let mut header = [0_u8; FRAME_HEADER_BYTES];
    let mut read = 0_usize;
    while read < FRAME_HEADER_BYTES {
        match reader.read(&mut header[read..]) {
            Ok(0) if read == 0 => return Ok(HeaderRead::CleanEof),
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "truncated broker-pipe frame header",
                ));
            }
            Ok(count) => read += count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(HeaderRead::Frame(header))
}

type ActiveStreamTx = Option<(u32, mpsc::Sender<io::Result<Vec<u8>>>)>;

fn fail_reader_protocol(active: &mut ActiveStreamTx, tx: &Arc<TxShared>, error: &io::Error) {
    if let Some((_, sender)) = active.take() {
        let _ = sender.blocking_send(Err(io::Error::new(error.kind(), "broker-pipe failed")));
    }
    let mut state = lock_unpoisoned(&tx.state);
    state.failed = true;
    state.queue.clear();
    state.queued_data_frames = 0;
    state.wake_io();
    drop(state);
    tx.writer_ready.notify_one();
}

fn dispatch_reader_frame<R: Read>(
    reader: &mut R,
    accept_tx: &mpsc::Sender<PipeConnection>,
    tx: &Arc<TxShared>,
    closer: &Arc<TxCloser>,
    last_stream_id: &mut u32,
    active: &mut ActiveStreamTx,
    header: [u8; FRAME_HEADER_BYTES],
) -> io::Result<bool> {
    let (stream_id, kind, len) = decode_header(header);
    match kind {
        KIND_OPEN => {
            if len != 0 || stream_id <= *last_stream_id {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid stream open",
                ));
            }
            *last_stream_id = stream_id;
            drop(active.take());
            let (req_tx, req_rx) = mpsc::channel(CHANNEL_CAPACITY);
            let conn = PipeConnection {
                stream_id,
                rx: req_rx,
                buffered: Vec::new(),
                offset: 0,
                tx: Arc::clone(tx),
                closer: Arc::clone(closer),
                eof_enqueued: false,
                pending_data_seq: 0,
                pending_eof_seq: 0,
            };
            if accept_tx.blocking_send(conn).is_err() {
                return Ok(false);
            }
            *active = Some((stream_id, req_tx));
        }
        KIND_DATA => {
            if stream_id == 0 || !(1..=MAX_FRAME_PAYLOAD).contains(&len) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid data frame",
                ));
            }
            let mut payload = vec![0_u8; len];
            reader.read_exact(&mut payload)?;
            if let Some((active_id, sender)) = active
                && *active_id == stream_id
            {
                let _ = sender.blocking_send(Ok(payload));
            }
        }
        KIND_EOF => {
            if stream_id == 0 || len != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid eof frame",
                ));
            }
            if active
                .as_ref()
                .is_some_and(|(active_id, _)| *active_id == stream_id)
            {
                *active = None;
            }
        }
        KIND_RESET => {
            if stream_id == 0 || len != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid reset frame",
                ));
            }
            let mut state = lock_unpoisoned(&tx.state);
            state.record_reset(stream_id);
            state.wake_io();
            drop(state);
            if active
                .as_ref()
                .is_some_and(|(active_id, _)| *active_id == stream_id)
            {
                *active = None;
            }
        }
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unknown frame kind",
            ));
        }
    }
    Ok(true)
}

fn run_reader<R: Read>(
    mut reader: R,
    accept_tx: &mpsc::Sender<PipeConnection>,
    tx: &Arc<TxShared>,
    closer: &Arc<TxCloser>,
) {
    let mut last_stream_id = 0_u32;
    let mut active: ActiveStreamTx = None;
    loop {
        let header = match read_header(&mut reader) {
            Ok(HeaderRead::CleanEof) => break,
            Ok(HeaderRead::Frame(header)) => header,
            Err(error) => {
                fail_reader_protocol(&mut active, tx, &error);
                break;
            }
        };
        match dispatch_reader_frame(
            &mut reader,
            accept_tx,
            tx,
            closer,
            &mut last_stream_id,
            &mut active,
            header,
        ) {
            Ok(true) => {}
            Ok(false) => break,
            Err(error) => {
                fail_reader_protocol(&mut active, tx, &error);
                break;
            }
        }
    }
}

fn run_writer<W: Write>(mut writer: W, tx: &Arc<TxShared>) {
    let mut batch = Vec::new();
    loop {
        let batch_seq = {
            let mut state = lock_unpoisoned(&tx.state);
            while state.queue.is_empty() && !state.closed && !state.failed {
                state = wait_unpoisoned(&tx.writer_ready, state);
            }
            if state.failed || (state.queue.is_empty() && state.closed) {
                break;
            }
            batch.clear();
            let mut flushed_data = 0_usize;
            while let Some(frame) = state.queue.pop_front() {
                if frame.kind == KIND_DATA {
                    flushed_data = flushed_data.saturating_add(1);
                }
                if matches!(frame.kind, KIND_DATA | KIND_EOF) && state.is_reset(frame.stream_id) {
                    if frame.kind == KIND_DATA {
                        state.record_dropped_data(frame.stream_id);
                    }
                    continue;
                }
                batch.push(frame);
            }
            state.queued_data_frames = state.queued_data_frames.saturating_sub(flushed_data);
            let waker = if state.queued_data_frames < CHANNEL_CAPACITY {
                state.write_waker.take()
            } else {
                None
            };
            let seq = state.enqueued_seq;
            drop(state);
            if let Some(waker) = waker {
                waker.wake();
            }
            seq
        };

        let mut io_result = Ok(());
        for frame in &batch {
            let Ok(payload_len) = u32::try_from(frame.payload.len()) else {
                io_result = Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "broker-pipe frame exceeds u32 bound",
                ));
                break;
            };
            let header = encode_header(frame.stream_id, frame.kind, payload_len);
            if let Err(error) = writer
                .write_all(&header)
                .and_then(|()| writer.write_all(&frame.payload))
            {
                io_result = Err(error);
                break;
            }
        }
        if io_result.is_ok() {
            io_result = writer.flush();
        }

        let mut state = lock_unpoisoned(&tx.state);
        if io_result.is_err() {
            state.failed = true;
            state.queue.clear();
            state.queued_data_frames = 0;
            state.wake_io();
            break;
        }
        state.flushed_seq = state.flushed_seq.max(batch_seq);
        if let Some(waker) = state.flush_waker.take() {
            waker.wake();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque as SyncQueue;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    const VALID_KEY: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    struct PipeQueue {
        bytes: SyncQueue<u8>,
        closed: bool,
    }

    struct MemoryStream {
        shared: Arc<(Mutex<PipeQueue>, Condvar)>,
    }

    fn memory_channel() -> (MemoryStream, MemoryStream) {
        let shared = Arc::new((
            Mutex::new(PipeQueue {
                bytes: SyncQueue::new(),
                closed: false,
            }),
            Condvar::new(),
        ));
        (
            MemoryStream {
                shared: Arc::clone(&shared),
            },
            MemoryStream { shared },
        )
    }

    impl Drop for MemoryStream {
        fn drop(&mut self) {
            let mut guard = lock_unpoisoned(&self.shared.0);
            guard.closed = true;
            drop(guard);
            self.shared.1.notify_all();
        }
    }

    impl Read for MemoryStream {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let mut guard = lock_unpoisoned(&self.shared.0);
            while guard.bytes.is_empty() && !guard.closed {
                guard = wait_unpoisoned(&self.shared.1, guard);
            }
            let count = guard.bytes.len().min(buf.len());
            for slot in &mut buf[..count] {
                *slot = guard.bytes.pop_front().unwrap_or(0);
            }
            drop(guard);
            Ok(count)
        }
    }

    impl Write for MemoryStream {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            let mut guard = lock_unpoisoned(&self.shared.0);
            guard.bytes.extend(buf.iter().copied());
            drop(guard);
            self.shared.1.notify_all();
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn write_frame(
        writer: &mut impl Write,
        stream_id: u32,
        kind: u8,
        payload: &[u8],
    ) -> io::Result<()> {
        let len = u32::try_from(payload.len()).map_err(io::Error::other)?;
        writer.write_all(&encode_header(stream_id, kind, len))?;
        writer.write_all(payload)?;
        writer.flush()
    }

    fn read_frame(reader: &mut impl Read) -> io::Result<(u32, u8, Vec<u8>)> {
        let HeaderRead::Frame(header) = read_header(reader)? else {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "eof"));
        };
        let (stream_id, kind, len) = decode_header(header);
        let mut payload = vec![0_u8; len];
        reader.read_exact(&mut payload)?;
        Ok((stream_id, kind, payload))
    }

    #[test]
    fn handshake_validates_magic_and_hex_key() -> io::Result<()> {
        for bad in [
            [b"WRONGMAG".as_slice(), VALID_KEY.as_bytes()].concat(),
            [
                HANDSHAKE_REQUEST_MAGIC.as_slice(),
                VALID_KEY.to_ascii_uppercase().as_bytes(),
            ]
            .concat(),
            [HANDSHAKE_REQUEST_MAGIC.as_slice(), b"short"].concat(),
        ] {
            let (mut p_to_c_w, p_to_c_r) = memory_channel();
            let (c_to_p_w, _c_to_p_r) = memory_channel();
            p_to_c_w.write_all(&bad)?;
            drop(p_to_c_w);
            assert!(ServerEndpoint::from_io(p_to_c_r, c_to_p_w).is_err());
        }

        let (mut p_to_c_w, p_to_c_r) = memory_channel();
        let (c_to_p_w, mut c_to_p_r) = memory_channel();
        p_to_c_w.write_all(HANDSHAKE_REQUEST_MAGIC)?;
        p_to_c_w.write_all(VALID_KEY.as_bytes())?;
        let (endpoint, key) = ServerEndpoint::from_io(p_to_c_r, c_to_p_w)?;
        assert_eq!(key.as_str(), VALID_KEY);
        let mut ack = [0_u8; 8];
        c_to_p_r.read_exact(&mut ack)?;
        assert_eq!(&ack, HANDSHAKE_RESPONSE_MAGIC);
        endpoint.mark_ready()?;
        assert_eq!(
            read_frame(&mut c_to_p_r)?,
            (0, KIND_SERVER_READY, Vec::new())
        );
        drop(p_to_c_w);
        Ok(())
    }

    #[test]
    fn multiplexer_delivers_large_sequential_exchanges_and_reset_cancellation() -> io::Result<()> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        runtime.block_on(async {
            let (mut p_to_c_w, p_to_c_r) = memory_channel();
            let (c_to_p_w, mut c_to_p_r) = memory_channel();
            p_to_c_w.write_all(HANDSHAKE_REQUEST_MAGIC)?;
            p_to_c_w.write_all(VALID_KEY.as_bytes())?;

            let (mut endpoint, _) = ServerEndpoint::from_io(p_to_c_r, c_to_p_w)?;
            let mut ack = [0_u8; 8];
            c_to_p_r.read_exact(&mut ack)?;
            assert_eq!(&ack, HANDSHAKE_RESPONSE_MAGIC);
            endpoint.mark_ready()?;
            assert_eq!(
                read_frame(&mut c_to_p_r)?,
                (0, KIND_SERVER_READY, Vec::new())
            );

            // Exchange 1: chunked >16 KiB payload + half-close + response.
            let payload: Vec<u8> = (0..=255_u8).cycle().take(MAX_FRAME_PAYLOAD + 777).collect();
            write_frame(&mut p_to_c_w, 1, KIND_OPEN, &[])?;
            write_frame(&mut p_to_c_w, 1, KIND_DATA, &payload[..MAX_FRAME_PAYLOAD])?;
            write_frame(&mut p_to_c_w, 1, KIND_DATA, &payload[MAX_FRAME_PAYLOAD..])?;
            write_frame(&mut p_to_c_w, 1, KIND_EOF, &[])?;

            let mut conn1 = endpoint
                .accept()
                .await?
                .ok_or_else(|| io::Error::other("missing first connection"))?;
            let mut received = Vec::new();
            conn1.read_to_end(&mut received).await?;
            assert_eq!(received, payload);
            conn1.write_all(&payload).await?;
            conn1.shutdown().await?;
            drop(conn1);

            let (s1_a, k1_a, d1_a) = read_frame(&mut c_to_p_r)?;
            let (s1_b, k1_b, d1_b) = read_frame(&mut c_to_p_r)?;
            let (s1_eof, k1_eof, d1_eof) = read_frame(&mut c_to_p_r)?;
            assert_eq!((s1_a, k1_a, d1_a.len()), (1, KIND_DATA, MAX_FRAME_PAYLOAD));
            assert_eq!((s1_b, k1_b, d1_b.len()), (1, KIND_DATA, 777));
            assert_eq!([d1_a, d1_b].concat(), payload);
            assert_eq!((s1_eof, k1_eof, d1_eof), (1, KIND_EOF, Vec::new()));

            // Exchange 2: mid-stream KIND_RESET immediately wakes abandoned read with EOF
            // and suppresses outgoing stale response frames for stream 2.
            write_frame(&mut p_to_c_w, 2, KIND_OPEN, &[])?;
            write_frame(&mut p_to_c_w, 2, KIND_DATA, b"GET /slow")?;
            let mut conn2 = endpoint
                .accept()
                .await?
                .ok_or_else(|| io::Error::other("missing second connection"))?;
            let mut head = [0_u8; 9];
            conn2.read_exact(&mut head).await?;
            assert_eq!(&head, b"GET /slow");

            write_frame(&mut p_to_c_w, 2, KIND_RESET, &[])?;
            let mut abandon = [0_u8; 1];
            assert_eq!(conn2.read(&mut abandon).await?, 0);
            // Writing after reset must fail with BrokenPipe instead of emitting stale frames.
            assert!(conn2.write_all(b"stale").await.is_err());
            drop(conn2);

            // Exchange 3: subsequent stream still works after stream 2 reset.
            write_frame(&mut p_to_c_w, 3, KIND_OPEN, &[])?;
            write_frame(&mut p_to_c_w, 3, KIND_DATA, b"POST /reset")?;
            let mut conn3 = endpoint
                .accept()
                .await?
                .ok_or_else(|| io::Error::other("missing third connection"))?;
            let mut reset_req = [0_u8; 11];
            conn3.read_exact(&mut reset_req).await?;
            assert_eq!(&reset_req, b"POST /reset");
            conn3.write_all(b"ok").await?;
            conn3.shutdown().await?;
            drop(conn3);

            assert_eq!(read_frame(&mut c_to_p_r)?, (3, KIND_DATA, b"ok".to_vec()));
            assert_eq!(read_frame(&mut c_to_p_r)?, (3, KIND_EOF, Vec::new()));

            drop(p_to_c_w);
            assert!(endpoint.accept().await?.is_none());
            Ok(())
        })
    }

    #[test]
    fn protocol_errors_close_endpoint_without_panicking() -> io::Result<()> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        runtime.block_on(async {
            let (mut p_to_c_w, p_to_c_r) = memory_channel();
            let (c_to_p_w, mut c_to_p_r) = memory_channel();
            p_to_c_w.write_all(HANDSHAKE_REQUEST_MAGIC)?;
            p_to_c_w.write_all(VALID_KEY.as_bytes())?;
            let (mut endpoint, _) = ServerEndpoint::from_io(p_to_c_r, c_to_p_w)?;
            let mut ack = [0_u8; 8];
            c_to_p_r.read_exact(&mut ack)?;
            // Oversized DATA frame must fail protocol and reject accept().
            p_to_c_w.write_all(&encode_header(1, KIND_DATA, 16_385))?;
            p_to_c_w.flush()?;
            assert!(endpoint.accept().await.is_err());
            drop(p_to_c_w);
            Ok(())
        })
    }
}
