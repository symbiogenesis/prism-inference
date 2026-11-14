//! One externally supervised, source-bound fixed CPU comparison arm.
//!
//! The seven-argument FP16 protocol and legacy baseline remain separate from
//! the current-source profiles exposed by `cpu_candidate_check`.

include!("cpu_bundle_check/collector.rs");

const USAGE: &str = "cpu_bundle_check <absolute-original-PQ2> <absolute-approved-adapter> <existing-empty-evidence-directory> <absolute-public-messages.json> <output-1..32> <original-deadline-unix-ms> <source-commit>";

#[derive(Default)]
struct Profile;

fn parse_profile(arguments: &[String]) -> Result<Profile> {
    if !arguments.is_empty() {
        return Err(reject(USAGE));
    }
    Ok(Profile)
}

#[cfg(cpu_bundle_legacy_f16)]
fn load_session(selection: &Selection) -> Result<SwiftSession> {
    // Only the reviewed 0.1.23 baseline uses its intrinsic FP16 graph loader.
    SwiftSession::load_profile_with_row_parallelism(
        &selection.model,
        Representation::Pq2,
        Some(&selection.adapter),
        MatvecBackend::ExperimentalNeon,
        42,
        RowParallelism::new(4)?,
    )
}

#[cfg(not(cpu_bundle_legacy_f16))]
fn load_session(selection: &Selection) -> Result<SwiftSession> {
    SwiftSession::load_profile_with_kv_cache(
        &selection.model,
        Representation::Pq2,
        Some(&selection.adapter),
        MatvecBackend::ExperimentalNeon,
        42,
        RowParallelism::new(4)?,
        bonsai_prism_runtime::graph::KvCachePolicy::F16,
    )
}

const fn profile_matches(selection: &Selection, _session: &SwiftSession) -> bool {
    let Profile = selection.profile;
    true
}

const fn extend_identity(_selection: &Selection, identity: Value) -> Value {
    identity
}

const fn receipt_names() -> (&'static str, &'static str) {
    ("cpu-bundle-started.json", "cpu-bundle.json")
}

#[cfg(test)]
#[path = "cpu_bundle_check/tests.rs"]
mod tests;
