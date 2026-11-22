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
pub use hmx::{HmxQ2Executor, HmxResidencyStatistics};
mod native_q2;
pub use native_q2::{
    ArenaAssignment, DEFAULT_NATIVE_Q2_PROJECTIONS, DEFAULT_NATIVE_Q2_RESIDENT_METADATA_BYTES,
    DEFAULT_NATIVE_Q2_RESIDENT_WEIGHT_BYTES, NativeQ2ArenaCache, NativeQ2ArenaCacheEntry,
    NativeQ2Executor, NativeQ2ResidencyStatistics, VtcmPlacementPlan,
    plan_native_q2_layers_for_budget,
};
mod q8;
pub use q8::HmxQ8Executor;
mod placement;
pub use placement::{
    AcceleratorTarget, AdrenoQ2KernelKind, HeterogeneousLayerAssignment,
    HeterogeneousPlacementPlan, HybridPrefillSchedule, HybridPrefillStep, PlacedProjectionExecutor,
    X1_V73_MAX_NPU_LAYERS, plan_hybrid_prefill_pipeline, plan_three_processor_placement,
    plan_x1_v73_three_processor_placement,
};
mod composite;
pub use composite::{
    CompositeProjectionExecutor, CompositeTelemetry, HybridPrefillCoordinator,
    OwnedCompositeProjectionExecutor,
};

type Result<T> = bonsai_prism_runtime::Result<T>;

fn invalid(message: &str) -> bonsai_prism_runtime::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message).into()
}

pub(crate) fn finish_sha256_prefixed_hex(
    digest: bonsai_native_hash::Hasher,
    prefix: &str,
) -> Result<String> {
    let mut hex_bytes = [0_u8; 64];
    hex::encode_to_slice(digest.finish()?, &mut hex_bytes)
        .map_err(|_| invalid("sha256 hex digest length mismatch"))?;
    let mut identity = String::new();
    identity.try_reserve_exact(prefix.len() + hex_bytes.len())?;
    identity.push_str(prefix);
    for byte in hex_bytes {
        identity.push(char::from(byte));
    }
    Ok(identity)
}
