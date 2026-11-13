//! Independent mathematical observations for explicitly migrated source routes.
//!
//! This module never calls the production Rust runtime or a native bridge. It
//! supplies no backend, model, tokenizer, codec, ABI or performance observations.

use serde::Deserialize;
use serde_json::{Value, json};

use super::{MTP_ID, Result, invalid};

const SOURCE_SHA256: &str = "c8cad0219bc9c8c11c04574d3020876646835966412cc9e39eaae5aca805d92d";
const TEST_SHA256: &str = "48d136ab03706c888eb393a771a6e1c7ec0c4ac0be1d7581d58a35bb57f1aae0";

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
struct Request {
    schema_version: u32,
    operations: Vec<Operation>,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
struct Operation {
    kind: String,
    batch: i32,
    catchup: i32,
    sequences: i32,
}

// Derive admission by reserving anchors from a nonnegative capacity, then
// comparing the catch-up demand with the remainder. This differs from both
// production implementations' widened signed addition and handles the entire
// i32 input domain without overflow, floating arithmetic or fixture lookup.
fn first_decode_fits(batch: i32, catchup: i32, anchors: i32) -> bool {
    let (Ok(capacity), Ok(demand), Ok(reserved)) = (
        u32::try_from(batch),
        u32::try_from(catchup),
        u32::try_from(anchors),
    ) else {
        return false;
    };
    capacity != 0
        && capacity
            .checked_sub(reserved)
            .is_some_and(|left| demand <= left)
}

pub(super) fn execute(value: &Value) -> Result<Value> {
    if !value["Operations"]
        .as_array()
        .is_some_and(|operations| (1..=128).contains(&operations.len()))
    {
        return Err(invalid(
            "independent MTP operations must number 1 through 128",
        ));
    }
    let request: Request = serde_json::from_value(value.clone())?;
    if request.schema_version != 1
        || request
            .operations
            .iter()
            .any(|operation| operation.kind != "mtp-first-decode-fits")
    {
        return Err(invalid(
            "unsupported independent Rust reference operation or schema",
        ));
    }
    // Validate the complete typed batch above before computing observations.
    let results = request.operations.into_iter().map(|operation| {
        json!({"Fits": first_decode_fits(operation.batch, operation.catchup, operation.sequences)})
    }).collect::<Vec<_>>();
    Ok(json!({"SchemaVersion": 1, "Results": results}))
}

pub(super) fn identity() -> Value {
    json!({"Kind": "independent-rust-mathematical", "SupportedSourceIds": [MTP_ID],
        "UpstreamCommit": super::UPSTREAM,
        "ProductionSource": {"Path": "common/speculative.cpp", "Bytes": 183_232, "Sha256": SOURCE_SHA256},
        "AssertionSource": {"Path": MTP_ID, "Bytes": 812, "Sha256": TEST_SHA256},
        "Arithmetic": "nonnegative-capacity-minus-reserved-anchors-v1",
        "NativeReferenceReplay": false, "ProductionRuntimeCalled": false,
        "CompleteSourceGate": false, "NativeRetirementQualified": false})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, Target};

    #[test]
    fn independent_mtp_provider_preserves_all_six_source_assertions_and_signed_boundaries()
    -> Result<()> {
        let engine = Engine::new(Target::RustReference, None, None, None, None, None)?;
        let report = crate::common::run_mtp(&engine)?;
        assert_eq!(report["Id"], MTP_ID);
        assert_eq!(report["Status"], "pass", "{report}");
        for (batch, demand, anchors, expected) in [
            (1, 0, 0, true),
            (1, 0, 1, true),
            (1, 1, 1, false),
            (17, 11, 6, true),
            (17, 12, 6, false),
            (i32::MAX, i32::MAX, 0, true),
            (i32::MAX, 0, i32::MAX, true),
            (i32::MAX, i32::MAX, i32::MAX, false),
            (i32::MIN, 0, 0, false),
            (1, i32::MIN, 0, false),
            (1, 0, i32::MIN, false),
        ] {
            assert_eq!(first_decode_fits(batch, demand, anchors), expected);
        }
        let provider = identity();
        assert_eq!(provider["NativeReferenceReplay"], false);
        assert_eq!(provider["CompleteSourceGate"], false);
        assert_eq!(provider["NativeRetirementQualified"], false);
        Ok(())
    }

    #[test]
    fn independent_mtp_provider_rejects_unsupported_or_coerced_requests() {
        let valid = json!({"SchemaVersion": 1, "Operations": [{"Kind": "mtp-first-decode-fits",
            "Batch": 32, "Catchup": 31, "Sequences": 1}]});
        for (field, value) in [
            ("Kind", json!("backend-graph")),
            ("Batch", json!(true)),
            ("Batch", json!(i64::from(i32::MAX) + 1)),
            ("Catchup", json!(31.0)),
            ("Sequences", Value::Null),
            ("Unknown", json!(0)),
        ] {
            let mut request = valid.clone();
            request["Operations"][0][field] = value;
            assert!(execute(&request).is_err());
        }
        for operations in [json!([]), json!(vec![valid["Operations"][0].clone(); 129])] {
            let mut request = valid.clone();
            request["Operations"] = operations;
            assert!(execute(&request).is_err());
        }
        for version in [json!(0), json!(2), json!(true)] {
            let mut request = valid.clone();
            request["SchemaVersion"] = version;
            assert!(execute(&request).is_err());
        }
    }

    #[test]
    fn independent_mtp_provider_selection_cannot_replace_native_full_source_acceptance()
    -> Result<()> {
        let mut options = crate::Options {
            target: Target::RustReference,
            case: crate::CASES.iter().find(|case| case.id == MTP_ID),
            benchmark: None,
            diagnostic: None,
            bridge: None,
            common_bridge: None,
            evidence_directory: None,
            source_directory: None,
            asset_directory: None,
            output: std::env::temp_dir().join("independent-mtp-unused-report.json"),
        };
        options.validate()?;
        options.case = None;
        assert!(options.validate().is_err());
        options.case = crate::CASES.iter().find(|case| case.id != MTP_ID);
        assert!(options.validate().is_err());
        options.case = crate::CASES.iter().find(|case| case.id == MTP_ID);
        options.common_bridge = Some(std::env::temp_dir().join("unused-native-bridge"));
        assert!(options.validate().is_err());
        assert!(
            Engine::new(
                Target::RustReference,
                None,
                options.common_bridge,
                None,
                None,
                None
            )
            .is_err()
        );
        Ok(())
    }
}
