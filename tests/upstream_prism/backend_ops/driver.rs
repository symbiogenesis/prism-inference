// SPDX-License-Identifier: MIT
//! Deterministic case execution and complete partition/source accounting.

use super::super::engine::Engine;
use super::super::{Result, invalid};
use super::evaluate;
use super::gradient;
use super::graph::Plan;
use super::host::Production;
use super::performance;
use super::planner;
use super::prepare::{Mode, prepare};
use super::program::{Geometry, SOURCE_SHA256};
use super::transport;
use serde_json::{Value as Json, json};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

const ITEM_FORMAT_VERSION: u32 = 1;
const MAX_ITEM_BYTES: usize = 1024;

fn successful(status: &Json, mode: Mode) -> bool {
    matches!(status.as_str(), Some("pass" | "skip")) || (mode == Mode::Perf && status == "measured")
}

fn sha256_text(value: &Json) -> bool {
    value.as_str().is_some_and(|value| {
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}

fn validate_compact(row: &Json) -> Result<()> {
    const FIELDS: [&str; 8] = [
        "Index",
        "Class",
        "ArgumentsSha256",
        "Status",
        "CrtOffset",
        "CrtDraws",
        "ResultSha256",
        "CaseReceipt",
    ];
    let object = row
        .as_object()
        .ok_or_else(|| invalid("backend compact result is not an object"))?;
    let receipt = row["CaseReceipt"].as_str().and_then(|path| {
        let (directory, file) = path.split_once('/')?;
        (directory.starts_with("backend-")
            && directory.len() <= 128
            && directory
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
            && file == "case-receipt.json")
            .then_some(path)
    });
    if object.len() != FIELDS.len()
        || FIELDS.iter().any(|field| !object.contains_key(*field))
        || row["Index"].as_u64().is_none()
        || row["CrtOffset"].as_u64().is_none()
        || row["CrtDraws"].as_u64().is_none()
        || !row["Class"].as_str().is_some_and(|name| {
            !name.is_empty()
                && name.len() <= 128
                && name
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        })
        || !sha256_text(&row["ArgumentsSha256"])
        || !sha256_text(&row["ResultSha256"])
        || !matches!(
            row["Status"].as_str(),
            Some("pass" | "skip" | "measured" | "fail" | "error")
        )
        || receipt.is_none()
        || serde_json::to_vec(row)?.len() > MAX_ITEM_BYTES
    {
        return Err(invalid(
            "backend compact result schema, digest or bound differs",
        ));
    }
    Ok(())
}

fn compact_result(row: &Json, directory: &Path) -> Result<Json> {
    let directory = directory
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| invalid("backend case directory name is not UTF-8"))?;
    let compact = json!({
        "Index":row["Index"], "Class":row["Class"],
        "ArgumentsSha256":row["ArgumentsSha256"], "Status":row["Status"],
        "CrtOffset":row["CrtOffset"], "CrtDraws":row["CrtDraws"],
        "ResultSha256":planner::hash(&serde_json::to_vec(row)?),
        "CaseReceipt":format!("{directory}/case-receipt.json")
    });
    validate_compact(&compact)?;
    Ok(compact)
}

pub(super) fn cleanup(directory: &Path) -> Result<()> {
    let owned = fs::canonicalize(directory)?;
    if !owned.is_absolute() {
        return Err(invalid("backend owned cleanup directory is not absolute"));
    }
    for entry in fs::read_dir(&owned)? {
        let entry = entry?;
        if entry.file_name() == "case-receipt.json" {
            continue;
        }
        let path = fs::canonicalize(entry.path())?;
        if !path.starts_with(&owned) || !entry.file_type()?.is_file() {
            return Err(invalid(
                "backend cleanup encountered an unowned or non-file artifact",
            ));
        }
        fs::remove_file(path)?;
    }
    Ok(())
}

fn one(
    partition: &transport::Partition,
    index: usize,
    geometry: &Geometry,
    mode: Mode,
    row: &Json,
    plan: &Json,
    directory: &Path,
) -> Result<Option<Json>> {
    let (program, _, _) = planner::extent(mode)?;
    let threads = usize::try_from(
        plan["Threads"]
            .as_u64()
            .ok_or_else(|| invalid("backend plan thread count missing"))?,
    )?;
    if !(1..=256).contains(&threads) {
        return Err(invalid("backend source thread count exceeds native bound"));
    }
    let filter = plan["Filter"].as_str();
    let offset = row["CrtOffset"]
        .as_u64()
        .ok_or_else(|| invalid("backend planned CRT offset missing"))?;
    partition.with_session(directory, |session| {
        let mut host = Production::new(
            Plan::new(session),
            program,
            index as u64,
            threads,
            mode.code(),
        );
        let observed: Result<Json> = (|| {
            let mut prepared = prepare(&mut host, geometry, mode, filter)?;
            let observed = planner::receipt(&host, &prepared, geometry, index)?;
            for name in [
                "Index",
                "Class",
                "ArgumentsSha256",
                "ShouldInitialize",
                "CrtDraws",
                "Supported",
                "SkipReason",
                "Operation",
                "Variables",
                "ReferenceAvailable",
                "MetadataSha256",
            ] {
                if observed[name] != row[name] {
                    return Err(invalid(format!(
                        "backend execution preparation differs from source plan at {name}"
                    )));
                }
            }
            let mut result = if prepared.initialize {
                evaluate::allocate(&mut host, &mut prepared, geometry, mode, offset)?;
                match mode {
                    Mode::Eval => evaluate::evaluate(&mut host, &mut prepared, geometry)?,
                    Mode::Grad => gradient::evaluate(&mut host, &mut prepared, geometry)?,
                    Mode::Perf => performance::evaluate(&mut host, &mut prepared, geometry)?,
                    Mode::Support => {
                        return Err(invalid("source support mode cannot initialize tensors"));
                    }
                }
            } else {
                json!({"Status":if prepared.skip.is_some(){"skip"}else{"pass"},"Supported":prepared.supported,"SkipReason":prepared.skip})
            };
            result["Index"] = json!(index);
            result["Class"] = json!(geometry.class);
            result["ArgumentsSha256"] = json!(planner::arguments_hash(geometry)?);
            result["Operation"] = json!(prepared.operation);
            result["Variables"] = json!(prepared.variables);
            result["CrtOffset"] = json!(offset);
            result["CrtDraws"] = json!(prepared.crt_draws);
            result["Supported"] = json!(prepared.supported);
            result["SourceAssertions"] = json!(host.assertions);
            result["RandomDeviceSeeds"] = json!(host.inputs.ledger.seeds);
            result["InputAndOutputReceipts"] = json!(host.inputs.receipts);
            Ok(result)
        })();
        fs::write(
            directory.join("input-state.json"),
            serde_json::to_vec_pretty(
                &json!({"SchemaVersion":1,"GeometryIndex":index,"RandomDeviceAlgorithm":"SplitMix64 low32; initial state is GeometryIndex","RandomDeviceSeeds":host.inputs.ledger.seeds,"InputAndOutputReceipts":host.inputs.receipts,"CrtOffset":offset,"CrtDraws":host.crt.draws}),
            )?,
        )?;
        observed
    })
}

pub(super) fn run(
    engine: &Engine,
    index: usize,
    count: usize,
    mode: Mode,
    plan: &Json,
) -> Result<Json> {
    if mode == Mode::Perf {
        performance::require_ci()?;
    }
    planner::partition(index, count)?;
    let rows = planner::validate(plan, mode)?;
    let (_, geometries, geometry_hash) = planner::extent(mode)?;
    let mut items = Vec::new();
    let mut retention = super::retention::Budget::default();
    let partition = transport::Partition::open(engine)?;
    for (case_index, geometry) in geometries
        .iter()
        .enumerate()
        .filter(|(case_index, _)| case_index % count == index)
    {
        let directory =
            engine.evidence_path(&format!("backend-{}-run-{case_index:05}", mode.name()))?;
        let observed = one(
            &partition,
            case_index,
            geometry,
            mode,
            &rows[case_index],
            plan,
            &directory,
        );
        let mut observed = match observed {
            Ok(Some(value)) => value,
            Ok(None) => {
                return Ok(
                    json!({"SchemaVersion":1,"ItemFormatVersion":ITEM_FORMAT_VERSION,"SourceId":"test-backend-ops","SourceSha256":SOURCE_SHA256,"GeometrySha256":geometry_hash,"Mode":mode.name(),"CpuSelection":"explicit CPU","Partition":{"Index":index,"Count":count},"Status":"unsupported","Reason":"unsupported production API: backend graph execution, allocation and comparison","PlanSha256":plan["PlanSha256"],"ExpectedGeometries":geometries.len(),"Items":items}),
                );
            }
            Err(error) => {
                json!({"Index":case_index,"Class":geometry.class,"ArgumentsSha256":planner::arguments_hash(geometry)?,"Status":"error","Reason":error.to_string(),"CrtOffset":rows[case_index]["CrtOffset"],"CrtDraws":rows[case_index]["CrtDraws"]})
            }
        };
        fs::write(
            directory.join("case-receipt.json"),
            serde_json::to_vec_pretty(&observed)?,
        )?;
        if matches!(
            observed["Status"].as_str(),
            Some("pass" | "skip" | "measured")
        ) {
            cleanup(&directory)?;
        } else {
            observed["ArtifactRetention"] = retention.failure(&directory)?;
            fs::write(
                directory.join("case-receipt.json"),
                serde_json::to_vec_pretty(&observed)?,
            )?;
        }
        items.push(compact_result(&observed, &directory)?);
        if !partition.healthy() {
            return Err(invalid(
                "backend native transport failed; retained the failed geometry and aborted the partition",
            ));
        }
    }
    partition.finish()?;
    let passed = items.iter().all(|item| successful(&item["Status"], mode));
    Ok(
        json!({"SchemaVersion":1,"ItemFormatVersion":ITEM_FORMAT_VERSION,"SourceId":"test-backend-ops","SourceSha256":SOURCE_SHA256,"GeometrySha256":geometry_hash,"Mode":mode.name(),"CpuSelection":"explicit CPU","Partition":{"Index":index,"Count":count},"Status":if passed{"pass"}else{"fail"},"PlanSha256":plan["PlanSha256"],"ExpectedGeometries":geometries.len(),"Items":items}),
    )
}

#[allow(
    clippy::too_many_lines,
    reason = "The complete source-extent proof checks every partition and ordered geometry against the committed plan in one transaction."
)]
pub(super) fn aggregate(reports: &[Json], plan: &Json) -> Result<Json> {
    let mode = Mode::parse(
        plan["Mode"]
            .as_str()
            .ok_or_else(|| invalid("backend aggregate mode missing"))?,
    )?;
    let rows = planner::validate(plan, mode)?;
    let (_, geometries, geometry_hash) = planner::extent(mode)?;
    let first = reports
        .first()
        .ok_or_else(|| invalid("backend result aggregation has no partitions"))?;
    let count = usize::try_from(
        first["Partition"]["Count"]
            .as_u64()
            .ok_or_else(|| invalid("backend result partition count missing"))?,
    )?;
    if reports.len() != count {
        return Err(invalid("backend result partition count incomplete"));
    }
    let mut partitions = BTreeSet::new();
    let mut items = BTreeMap::new();
    let mut passed = true;
    for report in reports {
        if report["SchemaVersion"] != 1
            || report["ItemFormatVersion"] != ITEM_FORMAT_VERSION
            || report["SourceId"] != "test-backend-ops"
            || report["SourceSha256"] != SOURCE_SHA256
            || report["GeometrySha256"] != geometry_hash
            || report["Mode"] != mode.name()
            || report["CpuSelection"] != "explicit CPU"
            || report["PlanSha256"] != plan["PlanSha256"]
            || report["ExpectedGeometries"] != geometries.len()
            || report["Partition"]["Count"] != count
            || !matches!(report["Status"].as_str(), Some("pass" | "fail"))
        {
            return Err(invalid(
                "backend result provenance, plan or production completeness differs",
            ));
        }
        passed &= report["Status"] == "pass";
        let index = usize::try_from(
            report["Partition"]["Index"]
                .as_u64()
                .ok_or_else(|| invalid("backend result partition index missing"))?,
        )?;
        planner::partition(index, count)?;
        if !partitions.insert(index) {
            return Err(invalid("duplicate backend result partition"));
        }
        let selected = report["Items"]
            .as_array()
            .ok_or_else(|| invalid("backend result rows missing"))?;
        let expected = geometries
            .iter()
            .enumerate()
            .filter(|(case_index, _)| case_index % count == index)
            .map(|(case_index, _)| case_index)
            .collect::<Vec<_>>();
        if selected.len() != expected.len() {
            return Err(invalid(
                "backend result partition geometry extent incomplete",
            ));
        }
        for (row, &expected_index) in selected.iter().zip(&expected) {
            validate_compact(row)?;
            if row["Index"] != expected_index
                || row["Class"] != geometries[expected_index].class
                || row["ArgumentsSha256"] != rows[expected_index]["ArgumentsSha256"]
                || row["CrtOffset"] != rows[expected_index]["CrtOffset"]
                || row["CrtDraws"] != rows[expected_index]["CrtDraws"]
                || !row["CaseReceipt"].as_str().is_some_and(|path| {
                    path.starts_with(&format!("backend-{}-run-{expected_index:05}-", mode.name()))
                })
            {
                return Err(invalid(
                    "backend result geometry identity/order or input-stream receipt differs",
                ));
            }
            if row["Status"] == "measured" && mode != Mode::Perf {
                return Err(invalid(
                    "performance measurement cannot satisfy a correctness geometry",
                ));
            }
            passed &= successful(&row["Status"], mode);
            if items.insert(expected_index, row.clone()).is_some() {
                return Err(invalid("duplicate backend result geometry"));
            }
        }
    }
    if items.len() != geometries.len() {
        return Err(invalid(
            "backend complete source result geometry extent missing",
        ));
    }
    let mut pass_count = 0;
    let mut skip_count = 0;
    let mut fail_count = 0;
    let mut error_count = 0;
    let mut digests = Vec::with_capacity(items.len());
    for (index, row) in items {
        if index != digests.len() {
            return Err(invalid("backend result ordered geometry gap"));
        }
        match row["Status"].as_str() {
            Some("pass") => pass_count += 1,
            Some("measured") if mode == Mode::Perf => pass_count += 1,
            Some("skip") => skip_count += 1,
            Some("fail") => fail_count += 1,
            Some("error") => error_count += 1,
            _ => return Err(invalid("backend result status unknown")),
        }
        digests.push(
            row["ResultSha256"]
                .as_str()
                .ok_or_else(|| invalid("backend full result digest missing"))?
                .to_owned(),
        );
    }
    Ok(
        json!({"SchemaVersion":1,"ItemFormatVersion":ITEM_FORMAT_VERSION,"SourceId":"test-backend-ops","SourceSha256":SOURCE_SHA256,"GeometrySha256":geometry_hash,"Mode":mode.name(),"CpuSelection":"explicit CPU","Status":if passed{"pass"}else{"fail"},"PlanSha256":plan["PlanSha256"],"Filter":plan["Filter"],"Seed":plan["Seed"],"ExpectedGeometries":geometries.len(),"CompletedGeometries":digests.len(),"PassedGeometries":pass_count,"OriginalSkippedGeometries":skip_count,"FailedGeometries":fail_count,"ErrorGeometries":error_count,"OrderedResultSha256":planner::hash(&serde_json::to_vec(&digests)?),"TotalCrtDraws":plan["TotalCrtDraws"]}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_receipt_binds_full_metrics_and_retention_with_bounded_schema() -> Result<()> {
        let mut full = json!({
            "Index":0,"Class":"test_mul_mat","ArgumentsSha256":"a".repeat(64),
            "Status":"measured","CrtOffset":17,"CrtDraws":23,
            "ComparisonPairs":[{"Details":"x".repeat(65_536)}],
            "InputAndOutputReceipts":[{"Sha256":"b".repeat(64)}],
            "MeasuredMicroseconds":1_000_007,
            "ArtifactRetention":{"ReceiptSha256":"c".repeat(64)}
        });
        let directory = Path::new("backend-perf-run-00000-1-1");
        let compact = compact_result(&full, directory)?;
        assert_eq!(
            compact["ResultSha256"],
            planner::hash(&serde_json::to_vec(&full)?)
        );
        let persisted: Json = serde_json::from_slice(&serde_json::to_vec_pretty(&full)?)?;
        assert_eq!(
            compact["ResultSha256"],
            planner::hash(&serde_json::to_vec(&persisted)?)
        );
        assert_eq!(persisted["MeasuredMicroseconds"], 1_000_007);
        assert_eq!(
            persisted["ComparisonPairs"][0]["Details"],
            "x".repeat(65_536)
        );
        assert!(serde_json::to_vec(&compact)?.len() <= MAX_ITEM_BYTES);
        let source_count = planner::extent(Mode::Eval)?.1.len();
        assert!(MAX_ITEM_BYTES * source_count < 32 * 1024 * 1024);
        assert!(6 * source_count.div_ceil(16) * MAX_ITEM_BYTES < 16 * 1024 * 1024);
        full["ArtifactRetention"]["ReceiptSha256"] = json!("d".repeat(64));
        assert_ne!(
            compact_result(&full, directory)?["ResultSha256"],
            compact["ResultSha256"]
        );
        for (field, value) in [
            ("ResultSha256", json!("A".repeat(64))),
            ("ArgumentsSha256", json!("a".repeat(63))),
            ("CaseReceipt", json!("../case-receipt.json")),
            ("Class", json!("x".repeat(129))),
            ("Status", json!("unknown")),
            ("CrtDraws", json!(-1)),
            ("UnexpectedFullMetric", json!("x".repeat(65_536))),
        ] {
            let mut invalid = compact.clone();
            invalid[field] = value;
            assert!(validate_compact(&invalid).is_err());
        }
        Ok(())
    }

    struct Synthetic {
        plan: Json,
        reports: Vec<Json>,
        ordered_sha256: String,
    }

    fn synthetic(mode: Mode) -> Result<Synthetic> {
        let (_, geometries, geometry_hash) = planner::extent(mode)?;
        let mut rows = Vec::with_capacity(geometries.len());
        let mut items = [Vec::new(), Vec::new()];
        let mut digests = Vec::with_capacity(geometries.len());
        let mut offset = 0_u64;
        for (index, geometry) in geometries.iter().enumerate() {
            let arguments = planner::arguments_hash(geometry)?;
            let draws = u64::try_from(index % 7)?;
            rows.push(json!({"Index":index,"Class":geometry.class,"ArgumentsSha256":arguments,"CrtOffset":offset,"CrtDraws":draws}));
            let status = match index {
                0 if mode == Mode::Perf => "measured",
                0 => "pass",
                1 => "fail",
                2 => "error",
                _ => "skip",
            };
            let full = json!({"Index":index,"Class":geometry.class,"ArgumentsSha256":arguments,"CrtOffset":offset,"CrtDraws":draws,"Status":status,"SyntheticMetadataOnly":true,"FullMetrics":{"MeasuredMicroseconds":17,"InputAndOutputReceipts":["input-only synthetic hash"]}});
            digests.push(planner::hash(&serde_json::to_vec(&full)?));
            let directory = format!("backend-{}-run-{index:05}-1-1", mode.name());
            items[index % 2].push(compact_result(&full, Path::new(&directory))?);
            offset += draws;
        }
        let mut plan = json!({"SchemaVersion":1,"SourceId":"test-backend-ops","SourceSha256":SOURCE_SHA256,"GeometrySha256":geometry_hash,"Mode":mode.name(),"CpuSelection":"explicit CPU","Seed":1,"Status":"planned-complete","ExpectedGeometries":geometries.len(),"Filter":null,"Threads":1,"Items":rows,"TotalCrtDraws":offset});
        plan["PlanSha256"] = json!(planner::hash(&serde_json::to_vec(&plan)?));
        let reports = items.into_iter().enumerate().map(|(index, items)| {
            json!({"SchemaVersion":1,"ItemFormatVersion":ITEM_FORMAT_VERSION,"SourceId":"test-backend-ops","SourceSha256":SOURCE_SHA256,"GeometrySha256":geometry_hash,"Mode":mode.name(),"CpuSelection":"explicit CPU","Partition":{"Index":index,"Count":2},"Status":"fail","PlanSha256":plan["PlanSha256"],"ExpectedGeometries":geometries.len(),"Items":items})
        }).collect();
        Ok(Synthetic {
            plan,
            reports,
            ordered_sha256: planner::hash(&serde_json::to_vec(&digests)?),
        })
    }

    #[test]
    fn compact_aggregation_preserves_source_counts_order_and_full_result_digest() -> Result<()> {
        for mode in [Mode::Eval, Mode::Perf] {
            let mut fixture = synthetic(mode)?;
            let count = planner::extent(mode)?.1.len();
            let aggregate = aggregate(&fixture.reports, &fixture.plan)?;
            assert_eq!(aggregate["Status"], "fail");
            assert_eq!(aggregate["CompletedGeometries"], count);
            assert_eq!(aggregate["PassedGeometries"], 1);
            assert_eq!(aggregate["OriginalSkippedGeometries"], count - 3);
            assert_eq!(aggregate["FailedGeometries"], 1);
            assert_eq!(aggregate["ErrorGeometries"], 1);
            assert_eq!(aggregate["OrderedResultSha256"], fixture.ordered_sha256);
            fixture.reports[0]["ItemFormatVersion"] = json!(0);
            assert!(super::aggregate(&fixture.reports, &fixture.plan).is_err());
            fixture.reports[0]["ItemFormatVersion"] = json!(ITEM_FORMAT_VERSION);
            let row = fixture.reports[0]["Items"][0].clone();
            fixture.reports[0]["Items"][0]["Index"] = json!(2);
            assert!(super::aggregate(&fixture.reports, &fixture.plan).is_err());
            fixture.reports[0]["Items"][0] = row;
            let _ = fixture.reports[0]["Items"]
                .as_array_mut()
                .ok_or_else(|| invalid("synthetic compact items missing"))?
                .pop();
            assert!(super::aggregate(&fixture.reports, &fixture.plan).is_err());
        }
        Ok(())
    }
}
