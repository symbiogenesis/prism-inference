//! Bounds and value-free errors shared by the HTTP response adapter.
use std::time::Duration;
/// Largest accepted decoded backend response body in bytes.
pub const MAX_RESPONSE_BYTES: usize = 4 * 1024 * 1024;
/// Fixed payload buffer size for bounded JSON reads and SSE frames.
pub const RESPONSE_BUFFER_BYTES: usize = 8 * 1024;
/// Maximum elapsed time for writing and flushing one group of response bytes.
pub const CLIENT_WRITE_TIMEOUT: Duration = Duration::from_secs(5);

/// Value-free errors: no upstream header, body, prompt or I/O diagnostic escapes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResponseError {
    /// The supervisor or client-write deadline has expired.
    Deadline,
    /// The backend status, encoding or content type is unsupported.
    BackendMetadata,
    /// The supplied decoded backend body failed while being read.
    BackendIo,
    /// The client stream failed while writing, flushing or shutting down.
    ClientIo,
    /// The decoded backend body exceeds the fixed response cap.
    BodyLimit,
    /// A nonstreaming response is not a UTF-8 JSON object.
    JsonSyntax,
}
