//! Complete safe Rust port of pinned `tests/test-mtmd-impl.cpp`.
//!
//! PrismML-Eng/llama.cpp b87ca7225456b06b0975b8239c66b5aecf07e37c;
//! MIT attribution is retained in tools/prism/upstream/LICENSE.

use serde_json::{Map, Value, json};

use super::engine::Engine;
use super::{MTMD_ID, Result, invalid, result};

const TILING_CASES: [(u32, u32, bool); 5] = [
    (704, 704, false),
    // 720 / (16 * 2) is 22.5: upstream requires ties-to-even rounding.
    (720, 720, false),
    (736, 736, true),
    (1024, 977, true),
    (1056, 384, false),
];

const GROUPING_CASES: [(&str, u32, &str); 12] = [
    ("vv", 2, "2"),
    ("vvv", 2, "21"),
    ("vvvv", 2, "22"),
    ("vvi", 2, "21"),
    ("tvvt", 2, "2"),
    ("vtv", 2, "11"),
    ("vw", 2, "11"),
    ("aa", 2, "11"),
    ("ii", 2, "11"),
    ("iv", 2, "11"),
    ("vi", 2, "11"),
    ("vv", 1, "11"),
];

// The original v/w/a/i/t fixture language is interpreted in safe Rust.
// Only typed parts cross the bridge; it has no test-specific spec parser.
fn parts(specification: &str) -> Result<Vec<Value>> {
    specification
        .chars()
        .map(|part| match part {
            'v' => Ok(json!({"Kind": "bitmap", "Width": 100, "Height": 100, "Mergeable": true})),
            'w' => Ok(json!({"Kind": "bitmap", "Width": 200, "Height": 200, "Mergeable": true})),
            'a' => Ok(json!({"Kind": "audio", "Samples": 100, "Mergeable": true})),
            'i' => Ok(json!({"Kind": "bitmap", "Width": 100, "Height": 100, "Mergeable": false})),
            't' => Ok(json!({"Kind": "text", "Text": "hello"})),
            _ => Err(invalid("unknown original MTMD fixture part")),
        })
        .collect()
}

fn operations() -> Result<Vec<Value>> {
    let mut operations: Vec<Value> = TILING_CASES
        .iter()
        .map(|&(width, height, _)| {
            json!({
                "Kind": "mtmd-image-should-tile",
                "PatchSize": 16,
                "Merge": 2,
                "MinImageTokens": 64,
                "MaxImageTokens": 256,
                "Width": width,
                "Height": height,
            })
        })
        .collect();
    for &(specification, merge, _) in &GROUPING_CASES {
        operations.push(json!({
            "Kind": "mtmd-group-bitmaps",
            "Merge": merge,
            "Parts": parts(specification)?,
        }));
    }
    Ok(operations)
}

fn exact_fields<'a>(value: &'a Value, fields: &[&str]) -> Result<&'a Map<String, Value>> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid("MTMD bridge value must be an object"))?;
    if object.len() != fields.len() || fields.iter().any(|field| !object.contains_key(*field)) {
        return Err(invalid("MTMD bridge object fields mismatch"));
    }
    Ok(object)
}

fn response_results(response: &Value) -> Result<&[Value]> {
    let object = exact_fields(response, &["SchemaVersion", "Results"])?;
    if object["SchemaVersion"].as_u64() != Some(1) {
        return Err(invalid("MTMD bridge response version mismatch"));
    }
    let results = object["Results"]
        .as_array()
        .ok_or_else(|| invalid("MTMD bridge Results must be an array"))?;
    if results.len() != TILING_CASES.len() + GROUPING_CASES.len() {
        return Err(invalid("MTMD bridge result count mismatch"));
    }
    Ok(results)
}

fn should_tile(value: &Value) -> Result<bool> {
    exact_fields(value, &["ShouldTile"])?["ShouldTile"]
        .as_bool()
        .ok_or_else(|| invalid("MTMD ShouldTile must be boolean"))
}

fn grouping(value: &Value) -> Result<(Vec<u64>, u64)> {
    let object = exact_fields(value, &["GroupSizes", "RemainingBitmapParts"])?;
    let sizes = object["GroupSizes"]
        .as_array()
        .ok_or_else(|| invalid("MTMD GroupSizes must be an array"))?;
    if sizes.len() > 64 {
        return Err(invalid("MTMD GroupSizes exceeds part bound"));
    }
    let sizes = sizes
        .iter()
        .map(|size| {
            size.as_u64()
                .filter(|&size| size <= 64)
                .ok_or_else(|| invalid("MTMD group size must be an integer within part bound"))
        })
        .collect::<Result<Vec<_>>>()?;
    let remaining = object["RemainingBitmapParts"]
        .as_u64()
        .filter(|&count| count <= 64)
        .ok_or_else(|| invalid("MTMD remaining bitmap count must be within part bound"))?;
    Ok((sizes, remaining))
}

fn evaluate(results: &[Value]) -> Result<Value> {
    if results.len() != TILING_CASES.len() + GROUPING_CASES.len() {
        return Err(invalid("MTMD assertion result count mismatch"));
    }
    let mut failures = Vec::new();
    for (&(width, height, expected), value) in TILING_CASES.iter().zip(results) {
        let actual = should_tile(value)?;
        if actual != expected {
            failures.push(format!("tiling {width}x{height}: {actual} != {expected}"));
        }
    }
    for (&(specification, merge, expected), value) in
        GROUPING_CASES.iter().zip(&results[TILING_CASES.len()..])
    {
        let (sizes, remaining) = grouping(value)?;
        // Preserve the upstream concatenation assertion, independently of the
        // second assertion about the production-mutated parts vector.
        let actual: String = sizes.iter().map(ToString::to_string).collect();
        if actual != expected {
            failures.push(format!(
                "groups for {specification:?} merge={merge}: {actual:?} != {expected:?}"
            ));
        }
        let group_count = u64::try_from(sizes.len())?;
        if remaining != group_count {
            failures.push(format!(
                "remaining bitmap parts for {specification:?} merge={merge}: \
                 {remaining} != group count {group_count}"
            ));
        }
    }
    Ok(result(
        MTMD_ID,
        failures.is_empty(),
        format!(
            "29 original assertions: 5 LFM2 tiling + 12 group-size + 12 remaining-bitmap-count; \
             failed={}; {}",
            failures.len(),
            failures.join("; ")
        ),
    ))
}

pub(super) fn run(engine: &Engine) -> Result<Value> {
    let request = json!({"SchemaVersion": 1, "Operations": operations()?});
    let Some(response) = engine.execute_common("mtmd-impl", &request)? else {
        return Ok(result(
            MTMD_ID,
            false,
            "unsupported production APIs: MTMD image tiling and temporal bitmap grouping; \
             29 original assertions required"
                .to_owned(),
        ));
    };
    evaluate(response_results(&response)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_and_incomplete_mtmd_responses_cannot_pass() {
        assert!(response_results(&json!({"SchemaVersion": 1, "Results": []})).is_err());
        assert!(should_tile(&json!({"ShouldTile": 0})).is_err());
        assert!(grouping(&json!({"GroupSizes": [2], "RemainingBitmapParts": "1"})).is_err());
        assert!(grouping(&json!({"GroupSizes": [-1], "RemainingBitmapParts": 1})).is_err());
        assert!(parts("?").is_err());
    }

    #[test]
    fn remaining_parts_assertion_is_independent_of_group_sizes() -> Result<()> {
        let mut results: Vec<Value> = TILING_CASES
            .iter()
            .map(|&(_, _, expected)| json!({"ShouldTile": expected}))
            .collect();
        for &(_, _, expected) in &GROUPING_CASES {
            let sizes: Vec<u32> = expected
                .chars()
                .filter_map(|digit| digit.to_digit(10))
                .collect();
            results.push(json!({"GroupSizes": sizes, "RemainingBitmapParts": sizes.len()}));
        }
        assert_eq!(evaluate(&results)?["Status"], "pass");
        results[TILING_CASES.len()]["RemainingBitmapParts"] = json!(2);
        assert_eq!(evaluate(&results)?["Status"], "fail");
        Ok(())
    }
}
