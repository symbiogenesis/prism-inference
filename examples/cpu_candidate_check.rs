//! Current-source CPU profile collection, separate from the frozen bundle lane.
//!
//! External checked builds and native process/resource/power supervision remain
//! mandatory. These receipts alone establish no comparison or promotion verdict.

include!("cpu_bundle_check/collector.rs");

use bonsai_prism_runtime::{graph::KvCachePolicy, swift_session::CpuPrefill};

const USAGE: &str = "cpu_candidate_check <absolute-original-PQ2> <absolute-approved-adapter> <existing-empty-evidence-directory> <absolute-public-messages.json> <output-1..32> <original-deadline-unix-ms> <source-commit> <f16|k4-v4> <1|8|16|32> <prefill-scratch-bytes>";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Profile {
    cache: KvCachePolicy,
    prefill: Option<CpuPrefill>,
}

fn parse_profile(arguments: &[String]) -> Result<Profile> {
    if cfg!(cpu_bundle_legacy_f16) || arguments.len() != 3 {
        return Err(reject(USAGE));
    }
    let cache = match arguments[0].as_str() {
        "f16" => KvCachePolicy::F16,
        "k4-v4" => KvCachePolicy::Rotated4,
        _ => return Err(reject("unsupported source candidate KV policy")),
    };
    let tokens = arguments[1].parse::<usize>()?;
    let scratch = arguments[2].parse::<usize>()?;
    let prefill = match (tokens, scratch) {
        (1, 0) => None,
        (8 | 16 | 32, 1..=268_435_456) => Some(CpuPrefill::new(tokens, scratch)?),
        _ => {
            return Err(reject(
                "ordinary ingestion requires zero scratch; explicit tiles require a bounded nonzero reserve",
            ));
        }
    };
    Ok(Profile { cache, prefill })
}

fn load_session(selection: &Selection) -> Result<SwiftSession> {
    SwiftSession::load_profile_with_prefill(
        &selection.model,
        Representation::Pq2,
        Some(&selection.adapter),
        MatvecBackend::ExperimentalNeon,
        42,
        RowParallelism::new(4)?,
        selection.profile.cache,
        bonsai_prism_runtime::model::swift::CONTEXT_TOKENS,
        selection.profile.prefill,
    )
}

fn profile_matches(selection: &Selection, session: &SwiftSession) -> bool {
    session.kv_cache_policy() == selection.profile.cache
        && session.cpu_prefill() == selection.profile.prefill
        && session.cpu_speculation().is_none()
        && session.context_tokens() == bonsai_prism_runtime::model::swift::CONTEXT_TOKENS
}

fn extend_identity(selection: &Selection, identity: &mut Value) {
    identity["schema_version"] = json!(2);
    identity["kind"] = json!("swift-cpu-source-candidate-arm");
    identity["kv_cache"] = json!(if selection.profile.cache == KvCachePolicy::Rotated4 {
        "k4-v4"
    } else {
        "f16"
    });
    identity["context_tokens"] = json!(bonsai_prism_runtime::model::swift::CONTEXT_TOKENS);
    identity["cpu_prefill_tokens"] = json!(selection.profile.prefill.map_or(1, CpuPrefill::tokens));
    identity["cpu_prefill_scratch_bytes"] = json!(
        selection
            .profile
            .prefill
            .map_or(0, CpuPrefill::scratch_bytes)
    );
    identity["comparison_scope"] = json!(
        "separate current-source profile; frozen CPU bundle evaluator must reject; matched build/profile/power/resource/cleanup and independent complete-state acceptance required"
    );
}

const fn receipt_names() -> (&'static str, &'static str) {
    ("cpu-candidate-started.json", "cpu-candidate.json")
}

#[cfg(test)]
#[path = "cpu_bundle_check/candidate_tests.rs"]
mod tests;
