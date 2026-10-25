//! Offline Prism catalog identities, complete source reports and baseline policy.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use serde_json::{Map, Value, json};
use sha2::{Digest as _, Sha256};

use crate::Result;

fn require(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}

fn object<'a>(value: &'a Value, fields: &[&str], context: &str) -> Result<&'a Map<String, Value>> {
    let record = value
        .as_object()
        .ok_or_else(|| format!("{context} must be a JSON object."))?;
    for field in fields {
        require(
            record.contains_key(*field),
            &format!("{context} is missing {field}."),
        )?;
    }
    Ok(record)
}

fn text<'a>(value: &'a Value, field: &str, context: &str) -> Result<&'a str> {
    value[field]
        .as_str()
        .ok_or_else(|| format!("{context}.{field} must be a string."))
}

fn array<'a>(value: &'a Value, field: &str, context: &str) -> Result<&'a [Value]> {
    value[field]
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| format!("{context}.{field} must be an array."))
}

fn integer(value: &Value, field: &str, minimum: u64, context: &str) -> Result<u64> {
    value[field]
        .as_u64()
        .filter(|number| (minimum..=1_000_000).contains(number))
        .ok_or_else(|| {
            format!("{context}.{field} must be an integer from {minimum} through 1000000.")
        })
}

fn hexadecimal(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn identifier(value: &str) -> bool {
    let Some(path) = value.strip_prefix("tests/") else {
        return false;
    };
    let mut parts = path.rsplit('/');
    let Some(filename) = parts.next().and_then(|name| name.strip_prefix("test-")) else {
        return false;
    };
    let Some((stem, extension)) = filename.rsplit_once('.') else {
        return false;
    };
    !stem.is_empty()
        && matches!(extension, "cpp" | "c" | "py" | "sh")
        && stem
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
        && parts.all(|part| {
            !part.is_empty()
                && part
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
        })
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "PascalCase")]
struct Source<'a> {
    id: &'a str,
    source_sha256: &'a str,
}

#[derive(Serialize)]
#[serde(rename_all = "PascalCase")]
struct Catalog<'a> {
    schema_version: u64,
    upstream_commit: &'a str,
    archive_sha256: &'a str,
    #[serde(rename = "TestCMakeSourceSha256")]
    test_cmake_source_sha256: &'a str,
    measure: &'a str,
    adapter_ids: Vec<&'a str>,
    cases: Vec<Source<'a>>,
    #[serde(skip)]
    index: BTreeMap<&'a str, &'a str>,
}

impl<'a> Catalog<'a> {
    fn parse(value: &'a Value) -> Result<Self> {
        object(
            value,
            &[
                "SchemaVersion",
                "UpstreamCommit",
                "ArchiveSha256",
                "TestCMakeSourceSha256",
                "Measure",
                "AdapterIds",
                "Cases",
            ],
            "Catalog",
        )?;
        let schema_version = integer(value, "SchemaVersion", 1, "Catalog")?;
        let upstream_commit = text(value, "UpstreamCommit", "Catalog")?;
        let archive_sha256 = text(value, "ArchiveSha256", "Catalog")?;
        let test_cmake_source_sha256 = text(value, "TestCMakeSourceSha256", "Catalog")?;
        let measure = text(value, "Measure", "Catalog")?;
        let raw_cases = array(value, "Cases", "Catalog")?;
        require(
            schema_version == 1
                && measure == "upstream-source-files"
                && hexadecimal(upstream_commit, 40)
                && hexadecimal(archive_sha256, 64)
                && hexadecimal(test_cmake_source_sha256, 64)
                && !raw_cases.is_empty(),
            "Invalid Prism catalog contract.",
        )?;
        let mut cases = Vec::new();
        let mut index = BTreeMap::new();
        for raw in raw_cases {
            object(raw, &["Id", "SourceSha256"], "Catalog case")?;
            let id = text(raw, "Id", "Catalog case")?;
            let source_sha256 = text(raw, "SourceSha256", "Catalog case")?;
            require(
                identifier(id) && hexadecimal(source_sha256, 64),
                "Invalid Prism catalog case identity or source digest.",
            )?;
            require(
                index.insert(id, source_sha256).is_none(),
                &format!("Duplicate catalog ID: {id}"),
            )?;
            cases.push(Source { id, source_sha256 });
        }
        let mut adapter_ids = Vec::new();
        let mut adapters = BTreeSet::new();
        for value in array(value, "AdapterIds", "Catalog")? {
            let id = value
                .as_str()
                .ok_or("Catalog has an unknown or duplicate adapter ID.")?;
            require(
                index.contains_key(id) && adapters.insert(id),
                "Catalog has an unknown or duplicate adapter ID.",
            )?;
            adapter_ids.push(id);
        }
        Ok(Self {
            schema_version,
            upstream_commit,
            archive_sha256,
            test_cmake_source_sha256,
            measure,
            adapter_ids,
            cases,
            index,
        })
    }

    fn digest(&self) -> Result<String> {
        // Field order matches PowerShell's ordered ConvertTo-Json contract. Every
        // canonical string is validated ASCII, with no escapable characters.
        let canonical = serde_json::to_vec(self).map_err(|error| error.to_string())?;
        Ok(Sha256::digest(canonical)
            .iter()
            .flat_map(|byte| {
                const HEX: &[u8; 16] = b"0123456789abcdef";
                [
                    char::from(HEX[usize::from(byte >> 4)]),
                    char::from(HEX[usize::from(byte & 15)]),
                ]
            })
            .collect())
    }

    fn source_inventory(&self, request: &Value) -> Result<()> {
        if request.get("SourceInventory").is_none()
            && request.get("TestCMakeSourceSha256").is_none()
        {
            return Ok(());
        }
        let mut discovered = BTreeSet::new();
        for source in array(request, "SourceInventory", "Request")? {
            object(source, &["Id", "Sha256"], "Source inventory entry")?;
            let id = text(source, "Id", "Source inventory entry")?;
            let digest = text(source, "Sha256", "Source inventory entry")?;
            let expected = self
                .index
                .get(id)
                .ok_or_else(|| format!("Upstream test is absent from the catalog: {id}"))?;
            require(
                digest == *expected,
                &format!("Upstream source digest differs from the catalog: {id}"),
            )?;
            require(
                discovered.insert(id),
                &format!("Duplicate upstream source inventory ID: {id}"),
            )?;
        }
        require(
            discovered.len() == self.cases.len(),
            "The upstream source inventory is incomplete.",
        )?;
        require(
            text(request, "TestCMakeSourceSha256", "Request")? == self.test_cmake_source_sha256,
            "Upstream tests/CMakeLists.txt differs from its catalog provenance.",
        )
    }
}

fn status(value: &str) -> bool {
    matches!(value, "pass" | "fail" | "skip" | "unsupported")
}

#[derive(Clone, Copy)]
struct Outcome<'a> {
    status: &'a str,
    detail: &'a str,
}

fn result_index<'a>(
    value: &'a Value,
    catalog: &Catalog<'_>,
    context: &str,
) -> Result<BTreeMap<&'a str, Outcome<'a>>> {
    object(
        value,
        &["SchemaVersion", "UpstreamCommit", "Cases"],
        context,
    )?;
    require(
        integer(value, "SchemaVersion", 1, context)? == 1
            && text(value, "UpstreamCommit", context)? == catalog.upstream_commit,
        &format!("{context} schema or upstream revision does not match the catalog."),
    )?;
    let mut index = BTreeMap::new();
    for case in array(value, "Cases", context)? {
        object(
            case,
            &["Id", "Status", "Detail"],
            &format!("{context} case"),
        )?;
        let id = text(case, "Id", context)?;
        require(
            catalog.index.contains_key(id),
            &format!("Unknown {context} case ID: {id}"),
        )?;
        let status_value = text(case, "Status", context)?;
        let detail = text(case, "Detail", context)?;
        require(
            status(status_value),
            &format!("Invalid {context} result for {id}."),
        )?;
        require(
            index
                .insert(
                    id,
                    Outcome {
                        status: status_value,
                        detail,
                    },
                )
                .is_none(),
            &format!("Duplicate {context} case ID: {id}"),
        )?;
    }
    Ok(index)
}

#[derive(Serialize)]
#[serde(rename_all = "PascalCase")]
struct FullCase<'a> {
    id: &'a str,
    source_sha256: &'a str,
    status: &'a str,
    detail: &'a str,
    reference_status: &'a str,
    reference_detail: &'a str,
}

fn summary(cases: &[FullCase<'_>]) -> Result<Value> {
    let passed_ids: Vec<_> = cases
        .iter()
        .filter(|case| case.status == "pass")
        .map(|case| case.id)
        .collect();
    let red_ids: Vec<_> = cases
        .iter()
        .filter(|case| case.status != "pass")
        .map(|case| case.id)
        .collect();
    let reference_failure_ids: Vec<_> = cases
        .iter()
        .filter(|case| {
            matches!(case.reference_status, "fail" | "skip")
                || (case.status != "unsupported" && case.reference_status != "pass")
        })
        .map(|case| case.id)
        .collect();
    let total = u32::try_from(cases.len()).map_err(|error| error.to_string())?;
    let passed = u32::try_from(passed_ids.len()).map_err(|error| error.to_string())?;
    Ok(json!({
        "Total":total, "Passed":passed, "Red":red_ids.len(),
        "Failed":cases.iter().filter(|case| case.status == "fail").count(),
        "Skipped":cases.iter().filter(|case| case.status == "skip").count(),
        "Unsupported":cases.iter().filter(|case| case.status == "unsupported").count(),
        "GreenRatio":f64::from(passed) / f64::from(total),
        "ReferencePassed":reference_failure_ids.is_empty(),
        "ReferencePassedCount":cases.iter().filter(|case| case.reference_status == "pass").count(),
        "ReferenceUnsupportedCount":cases.iter().filter(|case| case.reference_status == "unsupported").count(),
        "PassedIds":passed_ids, "RedIds":red_ids, "ReferenceFailureIds":reference_failure_ids,
    }))
}

fn report(request: &Value, catalog: &Catalog<'_>) -> Result<Value> {
    let targets = result_index(&request["TargetResults"], catalog, "Target")?;
    let references = result_index(&request["ReferenceResults"], catalog, "Reference")?;
    for id in &catalog.adapter_ids {
        let (Some(target), Some(reference)) = (targets.get(id), references.get(id)) else {
            return Err(format!(
                "An implemented adapter is missing its target or C++ reference result: {id}"
            ));
        };
        require(
            target.status != "unsupported" && reference.status != "unsupported",
            &format!("An implemented adapter cannot be reported as unsupported: {id}"),
        )?;
    }
    let cases: Vec<_> = catalog
        .cases
        .iter()
        .map(|source| {
            let target = targets.get(source.id).copied().unwrap_or(Outcome {
                status: "unsupported",
                detail: "No Rust adapter is implemented for this upstream source.",
            });
            let reference = references.get(source.id).copied().unwrap_or(Outcome {
                status: "unsupported",
                detail: "No C++ reference adapter result was supplied.",
            });
            FullCase {
                id: source.id,
                source_sha256: source.source_sha256,
                status: target.status,
                detail: target.detail,
                reference_status: reference.status,
                reference_detail: reference.detail,
            }
        })
        .collect();
    Ok(json!({
        "SchemaVersion":1, "UpstreamCommit":catalog.upstream_commit,
        "CatalogSha256":catalog.digest()?, "Measure":catalog.measure,
        "Summary":summary(&cases)?, "Cases":cases,
    }))
}

fn full_report<'a>(
    value: &'a Value,
    catalog: &Catalog<'_>,
    digest: &str,
) -> Result<Vec<FullCase<'a>>> {
    object(
        value,
        &[
            "SchemaVersion",
            "UpstreamCommit",
            "CatalogSha256",
            "Measure",
            "Cases",
        ],
        "Report",
    )?;
    require(
        integer(value, "SchemaVersion", 1, "Report")? == 1
            && text(value, "UpstreamCommit", "Report")? == catalog.upstream_commit
            && text(value, "CatalogSha256", "Report")? == digest
            && text(value, "Measure", "Report")? == catalog.measure,
        "Report does not match the catalog.",
    )?;
    let mut seen = BTreeSet::new();
    let mut cases = Vec::new();
    for case in array(value, "Cases", "Report")? {
        object(
            case,
            &[
                "Id",
                "SourceSha256",
                "Status",
                "Detail",
                "ReferenceStatus",
                "ReferenceDetail",
            ],
            "Report case",
        )?;
        let id = text(case, "Id", "Report case")?;
        let expected = catalog
            .index
            .get(id)
            .ok_or("Report contains an unknown case ID.")?;
        require(seen.insert(id), &format!("Duplicate report case ID: {id}"))?;
        let source_sha256 = text(case, "SourceSha256", "Report case")?;
        let status_value = text(case, "Status", "Report case")?;
        let reference_status = text(case, "ReferenceStatus", "Report case")?;
        require(
            source_sha256 == *expected && status(status_value) && status(reference_status),
            &format!("Invalid full report case: {id}"),
        )?;
        require(
            !catalog.adapter_ids.contains(&id)
                || (status_value != "unsupported" && reference_status != "unsupported"),
            &format!("Full report lost an implemented adapter: {id}"),
        )?;
        cases.push(FullCase {
            id,
            source_sha256,
            status: status_value,
            detail: text(case, "Detail", "Report case")?,
            reference_status,
            reference_detail: text(case, "ReferenceDetail", "Report case")?,
        });
    }
    require(
        seen.len() == catalog.cases.len(),
        "Full report is missing catalog cases.",
    )?;
    Ok(cases)
}

fn required_ids<'a>(
    value: &'a Value,
    field: &str,
    catalog: &Catalog<'_>,
    message: &str,
) -> Result<Vec<&'a str>> {
    let mut seen = BTreeSet::new();
    let mut ids = Vec::new();
    for value in array(value, field, "Baseline")? {
        let id = value.as_str().ok_or(message)?;
        require(catalog.index.contains_key(id) && seen.insert(id), message)?;
        ids.push(id);
    }
    Ok(ids)
}

fn baseline(request: &Value, catalog: &Catalog<'_>) -> Result<Value> {
    let digest = catalog.digest()?;
    let cases = full_report(&request["Report"], catalog, &digest)?;
    let policy = &request["Baseline"];
    object(
        policy,
        &[
            "SchemaVersion",
            "UpstreamCommit",
            "CatalogSha256",
            "Measure",
            "Total",
            "MinimumPassed",
            "RequiredPassedIds",
            "RequiredReferencePassedIds",
        ],
        "Baseline",
    )?;
    let schema = integer(policy, "SchemaVersion", 1, "Baseline")?;
    let total = integer(policy, "Total", 1, "Baseline")?;
    let minimum = integer(policy, "MinimumPassed", 0, "Baseline")?;
    require(
        schema == 1
            && text(policy, "UpstreamCommit", "Baseline")? == catalog.upstream_commit
            && text(policy, "CatalogSha256", "Baseline")? == digest
            && text(policy, "Measure", "Baseline")? == catalog.measure
            && usize::try_from(total).map_err(|error| error.to_string())? == catalog.cases.len()
            && minimum <= total,
        "Baseline does not match the catalog or has an invalid minimum.",
    )?;
    let required = required_ids(
        policy,
        "RequiredPassedIds",
        catalog,
        "Baseline has an unknown or duplicate green ID.",
    )?;
    require(
        u64::try_from(required.len()).map_err(|error| error.to_string())? == minimum,
        "Baseline minimum must record every measured green ID.",
    )?;
    let references = required_ids(
        policy,
        "RequiredReferencePassedIds",
        catalog,
        "Baseline has an unknown or duplicate C++ reference green ID.",
    )?;
    let index: BTreeMap<_, _> = cases.iter().map(|case| (case.id, case)).collect();
    let mut issues = Vec::new();
    for id in required {
        if let Some(case) = index.get(id).filter(|case| case.status != "pass") {
            issues.push(format!(
                "Previously green case regressed: {id} ({}).",
                case.status
            ));
        }
    }
    for id in references {
        if let Some(case) = index.get(id).filter(|case| case.reference_status != "pass") {
            issues.push(format!(
                "Previously passing C++ reference regressed: {id} ({}).",
                case.reference_status
            ));
        }
    }
    let summary = summary(&cases)?;
    let passed = summary["Passed"]
        .as_u64()
        .ok_or("Invalid recomputed passed count.")?;
    let count = summary["Total"]
        .as_u64()
        .ok_or("Invalid recomputed total count.")?;
    if passed * total < minimum * count {
        issues.push(format!(
            "Green ratio fell below {minimum}/{total}: {passed}/{count}."
        ));
    }
    for id in array(&summary, "ReferenceFailureIds", "Summary")? {
        let id = id
            .as_str()
            .ok_or("Invalid recomputed reference failure ID.")?;
        issues.push(format!(
            "C++ reference did not pass for adapted case: {id}."
        ));
    }
    Ok(json!({
        "Passed":issues.is_empty(), "Issues":issues, "Summary":summary,
        "MinimumPassed":minimum, "BaselineTotal":total,
    }))
}

/// Validate a catalog, expand sparse runner results, or score a complete report.
///
/// # Errors
/// Rejects malformed evidence and catalog provenance mismatches. Measured
/// regressions return a baseline gate with `Passed` set to false.
#[allow(
    clippy::needless_pass_by_value,
    reason = "Maintenance command dispatch transfers ownership through the shared command API."
)]
pub fn run(request: Value) -> Result<Value> {
    object(
        &request,
        &["Operation", "Catalog"],
        "Prism baseline request",
    )?;
    let catalog = Catalog::parse(&request["Catalog"])?;
    match text(&request, "Operation", "Request")? {
        "catalog" => {
            catalog.source_inventory(&request)?;
            let digest = catalog.digest()?;
            let mut result = request["Catalog"].clone();
            result
                .as_object_mut()
                .ok_or("Catalog must be a JSON object.")?
                .insert("CatalogSha256".into(), json!(digest));
            Ok(result)
        }
        "report" => report(&request, &catalog),
        "baseline" => baseline(&request, &catalog),
        _ => Err("Unknown Prism baseline operation.".into()),
    }
}

#[cfg(test)]
#[path = "prism_baseline/tests.rs"]
mod tests;
