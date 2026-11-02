//! Scoped original-PQ2 projection execution over an owned signed Hexagon session.
//!
//! This unpublished integration crate keeps native-driver dependencies outside
//! the published CPU runtime. It enables no service profile or automatic device
//! selection. Signed device parity, full-model correctness and performance remain
//! independent qualification gates.
#![forbid(unsafe_code)]

mod guarded;
pub use guarded::{GuardedQ2Executor, ProjectionStatistics};
mod hmx;
pub use hmx::HmxQ2Executor;

type Result<T> = bonsai_prism_runtime::Result<T>;

fn invalid(message: &str) -> bonsai_prism_runtime::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message).into()
}
