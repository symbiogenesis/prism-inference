//! Complete Rust ports of pinned MIT-licensed Unicode and MTP bounds tests.
//!
//! Sources are PrismML-Eng/llama.cpp tests/test-unicode.cpp and
//! tests/test-mtp-catchup-batch.cpp at b87ca7225456b06b0975b8239c66b5aecf07e37c.
//! See tools/prism/upstream/LICENSE. Only production calls cross the bridge.

use serde_json::{Map, Value, json};

use super::engine::Engine;
use super::{MTP_ID, Result, UNICODE_ID, invalid, result};

fn response_results(response: &Value, expected: usize) -> Result<&[Value]> {
    let object = response
        .as_object()
        .ok_or_else(|| invalid("common bridge response must be an object"))?;
    if object.len() != 2 || object.get("SchemaVersion").and_then(Value::as_u64) != Some(1) {
        return Err(invalid("common bridge response schema/version mismatch"));
    }
    let results = object
        .get("Results")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("common bridge response Results must be an array"))?;
    if results.len() != expected {
        return Err(invalid("common bridge response result count mismatch"));
    }
    Ok(results)
}

fn single_field<'a>(value: &'a Value, field: &str) -> Result<&'a Map<String, Value>> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid("common bridge result must be an object"))?;
    if object.len() != 1 || !object.contains_key(field) {
        return Err(invalid("common bridge result field mismatch"));
    }
    Ok(object)
}

fn pieces(value: &Value) -> Result<Vec<&str>> {
    single_field(value, "Pieces")?["Pieces"]
        .as_array()
        .ok_or_else(|| invalid("common bridge Pieces must be an array"))?
        .iter()
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| invalid("common bridge piece must be UTF-8 text"))
        })
        .collect()
}

fn fits(value: &Value) -> Result<bool> {
    single_field(value, "Fits")?["Fits"]
        .as_bool()
        .ok_or_else(|| invalid("common bridge Fits must be boolean"))
}

pub(super) fn run_unicode(engine: &Engine) -> Result<Value> {
    // Exact original regex, input, encoding flag and expected piece vector.
    let request = json!({
        "SchemaVersion": 1,
        "Operations": [{
            "Kind": "unicode-regex-split",
            "Text": " ~foo",
            "Expressions": [r"[~][A-Za-z]+| ?[\p{S}]+|\s+"],
            "ByteEncoding": false,
        }],
    });
    let Some(response) = engine.execute_common("unicode-regex-split", &request)? else {
        return Ok(result(
            UNICODE_ID,
            false,
            "unsupported production API: unicode_regex_split; 1 original assertion required"
                .to_owned(),
        ));
    };
    let actual = pieces(&response_results(&response, 1)?[0])?;
    let expected = [" ~", "foo"];
    Ok(result(
        UNICODE_ID,
        actual == expected,
        format!("1 original regex split assertion; expected={expected:?}; actual={actual:?}"),
    ))
}

pub(super) fn run_mtp(engine: &Engine) -> Result<Value> {
    // All six upstream assertions, including zero batch and negative catch-up.
    let cases = [
        (32, 31, 1, true),
        (32, 32, 1, false),
        (32, 15 + 15, 2, true),
        (32, 16 + 16, 2, false),
        (0, 0, 0, false),
        (32, -1, 1, false),
    ];
    let operations: Vec<Value> = cases
        .iter()
        .map(|&(batch, catchup, sequences, _)| {
            json!({
                "Kind": "mtp-first-decode-fits",
                "Batch": batch,
                "Catchup": catchup,
                "Sequences": sequences,
            })
        })
        .collect();
    let request = json!({"SchemaVersion": 1, "Operations": operations});
    let Some(response) = engine.execute_common("mtp-first-decode-fits", &request)? else {
        return Ok(result(
            MTP_ID,
            false,
            "unsupported production API: common_speculative_mtp_first_decode_fits; 6 original assertions required".to_owned(),
        ));
    };
    let results = response_results(&response, cases.len())?;
    let mut failures = Vec::new();
    for ((batch, catchup, sequences, expected), value) in cases.iter().zip(results) {
        let actual = fits(value)?;
        if actual != *expected {
            failures.push(format!(
                "batch={batch} catchup={catchup} sequences={sequences}: {actual} != {expected}"
            ));
        }
    }
    Ok(result(
        MTP_ID,
        failures.is_empty(),
        format!(
            "6 original first-decode bounds assertions; failed={}; {}",
            failures.len(),
            failures.join("; ")
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rust_engine() -> Result<Engine> {
        Engine::new(
            super::super::engine::Target::Rust,
            None,
            None,
            None,
            None,
            None,
        )
    }

    #[test]
    fn rust_mtp_source_bounds_and_widened_sum() -> Result<()> {
        let report = run_mtp(&rust_engine()?)?;
        assert_eq!(report["Status"], "pass", "{report}");
        // Preserve the additional Rust signed-domain and overflow regressions
        // beside the original six checks, without changing the native schedule.
        for (batch, catchup, anchors, expected) in [
            (32, 0, -1, false),
            (i32::MAX, i32::MAX, 0, true),
            (i32::MAX, i32::MAX, i32::MAX, false),
        ] {
            assert_eq!(
                bonsai_prism_runtime::speculative::mtp_first_decode_fits(batch, catchup, anchors),
                expected,
                "batch={batch} catchup={catchup} anchors={anchors}"
            );
        }
        Ok(())
    }

    #[test]
    fn rust_unicode_source_split_subdivision_encoding_and_errors() -> Result<()> {
        use bonsai_prism_runtime::unicode::regex_split;

        let report = run_unicode(&rust_engine()?)?;
        assert_eq!(report["Status"], "pass", "{report}");
        assert_eq!(
            regex_split("ab12 cd", &[r"\w+", r"[0-9]+"], false)?.concat(),
            "ab12 cd"
        );
        assert_eq!(regex_split(" ", &[r"\s+"], true)?, ["Ġ"]);
        assert!(regex_split("text", &["["], false).is_err());
        Ok(())
    }

    #[test]
    fn rejects_malformed_common_responses() -> Result<()> {
        assert!(response_results(&json!({"SchemaVersion": 2, "Results": []}), 0).is_err());
        assert!(response_results(&json!({"SchemaVersion": 1, "Results": []}), 1).is_err());
        assert!(
            response_results(&json!({"SchemaVersion": 1, "Results": [], "Extra": 1}), 0).is_err()
        );
        assert!(pieces(&json!({"Pieces": ["valid", 1]})).is_err());
        assert!(pieces(&json!({"Pieces": [], "Fits": true})).is_err());
        assert!(fits(&json!({"Fits": "true"})).is_err());
        assert_eq!(pieces(&json!({"Pieces": [" ~", "foo"]}))?, [" ~", "foo"]);
        assert!(fits(&json!({"Fits": true}))?);
        Ok(())
    }
}
