// Copyright (c) 2023-2026 The ggml/llama.cpp authors
// SPDX-License-Identifier: MIT
//! Rust ownership of the complete backend operation source program.
//!
//! The source program and geometry are immutable inputs. Native operations are
//! production GGML calls; generation, references and every decision belong here.

#![forbid(unsafe_code)]
#![allow(
    clippy::float_cmp,
    clippy::suboptimal_flops,
    reason = "Source comparisons are exact and its separate multiply/add rounding must be preserved."
)]
#![allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "The source contract explicitly uses C++ integer conversions and binary32 arithmetic; the safe interpreter models those operations."
)]

#[path = "backend_ops/metrics.rs"]
mod metrics;
#[path = "backend_ops/number.rs"]
mod number;
#[path = "backend_ops/numeric_loop.rs"]
mod numeric_loop;
#[path = "backend_ops/program.rs"]
mod program;
#[path = "backend_ops/random.rs"]
mod random;
#[path = "backend_ops/tensor.rs"]
mod tensor;
#[path = "backend_ops/transport.rs"]
mod transport;
#[path = "backend_ops/value.rs"]
mod value;

#[path = "backend_ops/graph.rs"]
mod graph;

#[path = "backend_ops/interpreter.rs"]
mod interpreter;
#[path = "backend_ops/standard.rs"]
mod standard;

#[path = "backend_ops/host.rs"]
mod host;
#[path = "backend_ops/inputs.rs"]
mod inputs;

#[path = "backend_ops/planner.rs"]
mod planner;
#[path = "backend_ops/prepare.rs"]
mod prepare;

#[path = "backend_ops/retention.rs"]
mod retention;

#[path = "backend_ops/driver.rs"]
mod driver;
#[path = "backend_ops/evaluate.rs"]
mod evaluate;
#[path = "backend_ops/gradient.rs"]
mod gradient;

#[path = "backend_ops/complete.rs"]
mod complete;
#[path = "backend_ops/performance.rs"]
mod performance;

pub(super) fn run(engine: &super::engine::Engine) -> super::Result<serde_json::Value> {
    complete::run(engine)
}
#[path = "backend_ops/diagnostic.rs"]
mod diagnostic;
#[path = "backend_ops/rust.rs"]
mod rust;

pub(super) fn execute(request: &serde_json::Value) -> super::Result<Option<serde_json::Value>> {
    rust::execute(request)
}
#[path = "backend_ops/reload.rs"]
mod reload;

pub(super) fn diagnostic(
    engine: &super::engine::Engine,
    request: &serde_json::Value,
) -> super::Result<serde_json::Value> {
    diagnostic::dispatch(engine, request)
}
pub(super) fn benchmark(engine: &super::engine::Engine) -> super::Result<serde_json::Value> {
    diagnostic::benchmark(engine)
}
