//! Exact-source intake for frozen schema-2 normalized-input recovery factors.

use std::path::Path;

use bonsai_prism_runtime::graph::weight_patch::{
    FACTOR_APPLICATION_ARITHMETIC, FfnCompensation, GROUP_OMISSION_PROFILE,
};
use serde::Deserialize;

use super::{Hasher, Result, Value, hex, read_bounded, reject};

// Rank sixteen has 163,840 F32 numbers; pretty JSON exceeds rank-one's 4 MiB.
const MAX_RECOVERY_BYTES: u64 = 8 * 1024 * 1024;

#[cfg(test)]
#[path = "group_compensation_tests.rs"]
mod tests;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Factors {
    layout: String,
    a: Vec<Vec<f32>>,
    b: Vec<Vec<f32>>,
    a_sha256: String,
    b_sha256: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "This mirrors the externally frozen research artifact schema and rejects promoted claims."
)]
struct Artifact {
    schema_version: u32,
    kind: String,
    enabled: bool,
    qualification: String,
    source: Value,
    student_execution: Value,
    capture_sha256: String,
    site: Value,
    omission: Value,
    model_bound_sign_sha256: String,
    sign_payload_arithmetic: String,
    sign_binding_recipe: String,
    composition: String,
    teacher_and_retained_targets_include_standard_adapter: bool,
    apply_standard_adapter_again: bool,
    formula: String,
    factor_application_arithmetic: String,
    fit_target_arithmetic: String,
    fit_input_sha256: String,
    controls: Value,
    rank: usize,
    factors: Factors,
    development_screen: Value,
    exported_factor_f64_development_screen: Value,
    source_order_f32_development_screen: Value,
    source_authenticated: bool,
    changed_prefix_drift_recovery_qualified: bool,
    runtime_implemented: bool,
    runtime_implemented_scope: String,
    behavior_qualified: bool,
}

fn canonical_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(super::lower_hex)
}

fn sign_binding(source: &Value, site: &Value, omission: &Value) -> Result<String> {
    let mut hash = Hasher::new()?;
    hash.update(b"bonsai-ffn-model-bound-signs-v1\0")?;
    hash.update(
        source["modelSha256"]
            .as_str()
            .ok_or_else(|| reject("missing model digest"))?
            .as_bytes(),
    )?;
    let layer = u32::try_from(
        site["layer"]
            .as_u64()
            .ok_or_else(|| reject("missing layer"))?,
    )?;
    hash.update(&layer.to_le_bytes())?;
    hash.update(
        omission["signIdentity"]["fullSignsSha256"]
            .as_str()
            .ok_or_else(|| reject("missing signs digest"))?
            .as_bytes(),
    )?;
    hash.update(&11_264_u32.to_le_bytes())?;
    hash.update(&17_408_u32.to_le_bytes())?;
    hex(hash.finish()?)
}

fn matrix_hash(values: &[Vec<f32>]) -> Result<String> {
    let mut hash = Hasher::new()?;
    for value in values.iter().flatten() {
        hash.update(&value.to_le_bytes())?;
    }
    hex(hash.finish()?)
}

fn fixed_screen(screen: &Value) -> bool {
    screen["kind"] == "algebraic-development-screen-only"
        && screen["passed"] == true
        && screen["maximumDevelopmentRelativeSquaredError"] == 0.1
        && screen["maximumDevelopmentMinusFitRelativeSquaredError"] == 0.05
        && screen["requireNoWorseThanUncompensatedDevelopment"] == true
        && screen["rejectionReasons"]
            .as_array()
            .is_some_and(Vec::is_empty)
        && screen["thresholdsAreDesignChoices"] == true
        && screen["answerQualityQualified"] == false
        && screen["changedPrefixDriftRecoveryQualified"] == false
}

fn admit(
    bytes: &[u8],
    source: &Value,
    site: &Value,
    omission: &Value,
    omitted_graph: &str,
) -> Result<FfnCompensation> {
    let artifact: Artifact = serde_json::from_slice(bytes)?;
    if artifact.schema_version != 2
        || artifact.kind != "experimental-structured-ffn-compensation-v2"
        || artifact.enabled
        || artifact.qualification != "unqualified-research-only"
        || artifact.source != *source
        || artifact.site != *site
        || artifact.omission != *omission
        || artifact.student_execution["graphIdentity"] != omitted_graph
        || artifact.student_execution["executionProfile"] != GROUP_OMISSION_PROFILE
        || !artifact.student_execution["prefixPolicySha256"]
            .as_str()
            .is_some_and(canonical_digest)
        || artifact.model_bound_sign_sha256 != sign_binding(source, site, omission)?
        || artifact.sign_payload_arithmetic
            != "complete metadata sign_values array in original order, I32-le +/-1"
        || artifact.sign_binding_recipe
            != "SHA256(domain bonsai-ffn-model-bound-signs-v1 NUL || modelSha256 ASCII || layer U32-le || fullSignsSha256 ASCII || downSignOffset U32-le || downSignCount U32-le)"
        || artifact.composition != "retained-ffn-branch-plus-normalized-input-low-rank-compensation"
        || !artifact.teacher_and_retained_targets_include_standard_adapter
        || artifact.apply_standard_adapter_again
        || artifact.formula
            != "retained_branch + sum_k b[k] * dot(a[k], normalized_original_basis_student_input)"
        || artifact.factor_application_arithmetic != FACTOR_APPLICATION_ARITHMETIC
        || artifact.fit_target_arithmetic != "F64(teacherTarget F32) - F64(retainedBranch F32)"
        || !canonical_digest(&artifact.capture_sha256)
        || !canonical_digest(&artifact.fit_input_sha256)
        || artifact.source_authenticated
        || artifact.changed_prefix_drift_recovery_qualified
        || artifact.runtime_implemented
        || artifact.runtime_implemented_scope != "released-serving-and-selective-loading"
        || artifact.behavior_qualified
        || !fixed_screen(&artifact.development_screen)
        || artifact.development_screen["requireExportedFactorF64Screen"] != true
        || artifact.development_screen["requireSourceOrderF32Screen"] != true
        || artifact.development_screen["factorApplicationArithmetic"]
            != FACTOR_APPLICATION_ARITHMETIC
        || !fixed_screen(&artifact.exported_factor_f64_development_screen)
        || !fixed_screen(&artifact.source_order_f32_development_screen)
        || !(1..=16).contains(&artifact.rank)
        || artifact.factors.layout != "rank-major-f32-le"
        || artifact.factors.a.len() != artifact.rank
        || artifact.factors.b.len() != artifact.rank
        || artifact.controls["rank"]
            .as_u64()
            .is_none_or(|rank| rank < u64::try_from(artifact.rank).unwrap_or(u64::MAX) || rank > 16)
    {
        return Err(reject(
            "recovery artifact does not match exact source, omission, basis or screened arithmetic",
        ));
    }
    if artifact
        .factors
        .a
        .iter()
        .chain(&artifact.factors.b)
        .any(|row| row.len() != 5120)
        || artifact.factors.a_sha256 != matrix_hash(&artifact.factors.a)?
        || artifact.factors.b_sha256 != matrix_hash(&artifact.factors.b)?
    {
        return Err(reject(
            "recovery factors differ from their rank-major F32 identities",
        ));
    }
    FfnCompensation::new(artifact.factors.a, artifact.factors.b)
}

#[expect(
    clippy::redundant_pub_crate,
    reason = "Only the parent capture module admits artifacts."
)]
pub(super) fn read(
    path: &Path,
    source: &Value,
    site: &Value,
    omission: &Value,
    omitted_graph: &str,
) -> Result<(FfnCompensation, String)> {
    let bytes = read_bounded(path, MAX_RECOVERY_BYTES)?;
    let factors = admit(&bytes, source, site, omission, omitted_graph)?;
    let mut hash = Hasher::new()?;
    hash.update(&bytes)?;
    Ok((factors, hex(hash.finish()?)?))
}
