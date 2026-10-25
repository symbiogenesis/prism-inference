// SPDX-License-Identifier: MIT
//! The source's default eval mode requires every geometry, plus the support preflight.
//! Optional gradient receipts retain their actual verdict when explicitly supplied.

use super::super::engine::Engine;
use super::super::{Result, invalid, result};
use super::{driver, planner, prepare::Mode, program};
use serde_json::{Value as Json, json};
use std::collections::BTreeSet;

pub(super) fn aggregate(modes: &[Json]) -> Result<Json> {
    if !matches!(modes.len(), 2 | 3) {
        return Err(invalid(
            "backend source requires complete eval and support mode receipts, with optional grad",
        ));
    }
    let mut seen = BTreeSet::new();
    let mut passed = true;
    let mut receipts = Vec::new();
    for mode in modes {
        let name = mode["Mode"]
            .as_str()
            .ok_or_else(|| invalid("backend mode result name missing"))?;
        if !matches!(name, "eval" | "grad" | "support") || !seen.insert(name) {
            return Err(invalid(
                "backend source correctness modes are missing or duplicated",
            ));
        }
        if mode["SchemaVersion"] != 1
            || mode["SourceId"] != "test-backend-ops"
            || mode["SourceSha256"] != program::SOURCE_SHA256
            || mode["GeometrySha256"]
                != "1724d4a88170deb030f676d81f97ce050ab6ee32c37452626fd834ea1c4f7419"
            || mode["CpuSelection"] != "explicit CPU"
            || mode.get("Filter") != Some(&Json::Null)
            || mode["Seed"] != 1
            || mode["ExpectedGeometries"] != program::EVAL_COUNT
            || mode["CompletedGeometries"] != program::EVAL_COUNT
            || !matches!(mode["Status"].as_str(), Some("pass" | "fail"))
        {
            return Err(invalid(
                "backend source mode provenance/completeness differs",
            ));
        }
        let count = [
            "PassedGeometries",
            "OriginalSkippedGeometries",
            "FailedGeometries",
            "ErrorGeometries",
        ]
        .into_iter()
        .try_fold(0_u64, |count, name| {
            mode[name]
                .as_u64()
                .and_then(|value| count.checked_add(value))
                .ok_or_else(|| invalid("backend mode geometry counts missing or overflow"))
        })?;
        if count != program::EVAL_COUNT as u64 {
            return Err(invalid(
                "backend source mode counts do not cover its complete matrix",
            ));
        }
        if mode["Status"] == "pass"
            && (mode["FailedGeometries"] != 0 || mode["ErrorGeometries"] != 0)
        {
            return Err(invalid(
                "backend passing mode receipt contains failed or errored geometries",
            ));
        }
        passed &= mode["Status"] == "pass";
        receipts.push(mode.clone());
    }
    if !seen.contains("eval") || !seen.contains("support") {
        return Err(invalid(
            "backend source requires both eval and support mode receipts",
        ));
    }
    let gradient_included = seen.contains("grad");
    let names = seen.into_iter().collect::<Vec<_>>().join(", ");
    let mut source = result(
        "tests/test-backend-ops.cpp",
        passed,
        format!(
            "CPU explicitly selected; default upstream mode eval; all {} ordered geometries accounted for in {names}; grad is optional{}; benchmark1530 workloads are separate CI-only measurements",
            program::EVAL_COUNT,
            if gradient_included {
                " and included"
            } else {
                " and not included"
            }
        ),
    );
    source["SourceSha256"] = json!(program::SOURCE_SHA256);
    source["Modes"] = json!(receipts);
    source["SourceAcceptanceProfile"] = json!("default-eval-with-support-v1");
    source["DefaultSourceMode"] = json!("eval");
    source["CorrectnessModeCount"] = json!(modes.len());
    source["OptionalGradientIncluded"] = json!(gradient_included);
    source["SourceCount"] = json!(1);
    source["MetalForcedFaSlice"] = json!({"Status":"original skip","Reason":"CPU backend has no Metal tuning procedures; Apple platforms are outside this target"});
    Ok(source)
}

pub(super) fn run(engine: &Engine) -> Result<Json> {
    super::performance::require_ci()?;
    let mut modes = Vec::new();
    for mode in [Mode::Eval, Mode::Support] {
        let planned = planner::plan(engine, 0, 1, mode, None)?;
        if planned["Status"] != "planned" {
            let mut unsupported=result("tests/test-backend-ops.cpp",false,"unsupported production API: generic backend graph/support/allocation/comparison; no replacement Rust replica is used".into());
            unsupported["Coverage"] = planned;
            return Ok(unsupported);
        }
        let plan = planner::aggregate(&[planned])?;
        let executed = driver::run(engine, 0, 1, mode, &plan)?;
        modes.push(driver::aggregate(&[executed], &plan)?);
    }
    aggregate(&modes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn receipt(mode: &str) -> Json {
        json!({
            "SchemaVersion":1,"SourceId":"test-backend-ops",
            "SourceSha256":program::SOURCE_SHA256,
            "GeometrySha256":"1724d4a88170deb030f676d81f97ce050ab6ee32c37452626fd834ea1c4f7419",
            "Mode":mode,"CpuSelection":"explicit CPU","Filter":null,"Seed":1,
            "Status":"pass","ExpectedGeometries":program::EVAL_COUNT,
            "CompletedGeometries":program::EVAL_COUNT,
            "PassedGeometries":program::EVAL_COUNT,"OriginalSkippedGeometries":0,
            "FailedGeometries":0,"ErrorGeometries":0
        })
    }

    #[test]
    fn source_gate_requires_eval_support_and_preserves_optional_gradient_failure() -> Result<()> {
        let eval = receipt("eval");
        let support = receipt("support");
        let source = aggregate(&[eval.clone(), support.clone()])?;
        assert_eq!(source["Status"], "pass");
        assert_eq!(
            source["SourceAcceptanceProfile"],
            "default-eval-with-support-v1"
        );
        assert_eq!(source["DefaultSourceMode"], "eval");
        assert_eq!(source["CorrectnessModeCount"], 2);
        assert_eq!(source["OptionalGradientIncluded"], false);
        assert_eq!(source["Modes"][0], eval);
        assert_eq!(source["Modes"][1], support);

        let mut grad = receipt("grad");
        grad["Status"] = json!("fail");
        grad["PassedGeometries"] = json!(program::EVAL_COUNT - 1);
        grad["FailedGeometries"] = json!(1);
        let source = aggregate(&[support.clone(), grad.clone(), eval.clone()])?;
        assert_eq!(source["Status"], "fail");
        assert_eq!(source["CorrectnessModeCount"], 3);
        assert_eq!(source["OptionalGradientIncluded"], true);
        assert_eq!(source["Modes"][1], grad);
        assert_eq!(
            aggregate(&[eval.clone(), support.clone(), receipt("grad")])?["Status"],
            "pass"
        );
        for failed_mode in ["eval", "support"] {
            let mut modes = vec![eval.clone(), support.clone()];
            for mode in &mut modes {
                if mode["Mode"] == failed_mode {
                    mode["Status"] = json!("fail");
                    mode["PassedGeometries"] = json!(program::EVAL_COUNT - 1);
                    mode["ErrorGeometries"] = json!(1);
                }
            }
            assert_eq!(aggregate(&modes)?["Status"], "fail");
        }

        for modes in [
            vec![eval.clone()],
            vec![eval.clone(), receipt("grad")],
            vec![eval.clone(), eval.clone()],
            vec![eval.clone(), support.clone(), receipt("perf")],
            vec![eval.clone(), support.clone(), receipt("eval")],
        ] {
            assert!(aggregate(&modes).is_err());
        }
        for (field, value) in [
            ("SourceSha256", json!("0".repeat(64))),
            ("CompletedGeometries", json!(program::EVAL_COUNT - 1)),
            ("PassedGeometries", json!(program::EVAL_COUNT - 1)),
            ("Status", json!("planned")),
        ] {
            let mut incomplete = support.clone();
            incomplete[field] = value;
            assert!(aggregate(&[eval.clone(), incomplete]).is_err());
        }
        for field in ["FailedGeometries", "ErrorGeometries"] {
            let mut inconsistent = support.clone();
            inconsistent["PassedGeometries"] = json!(program::EVAL_COUNT - 1);
            inconsistent[field] = json!(1);
            assert!(aggregate(&[eval.clone(), inconsistent]).is_err());
        }
        Ok(())
    }
}
