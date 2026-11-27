//! Swift chat, client, sampling, session, and worker integration tests.
#![forbid(unsafe_code)]

#[cfg(feature = "cli")]
#[path = "swift/chat.rs"]
mod chat;
#[cfg(feature = "cli")]
#[path = "swift/client.rs"]
mod client;
#[path = "swift/sampling.rs"]
mod sampling;
#[cfg(feature = "cli")]
#[path = "swift/session.rs"]
mod session;
#[cfg(feature = "cli")]
#[path = "swift/worker_output.rs"]
mod worker_output;
#[cfg(feature = "cli")]
#[path = "swift/worker_request.rs"]
mod worker_request;
