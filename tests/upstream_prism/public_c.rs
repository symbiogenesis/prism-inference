//! Safe Rust ownership of the original empty C public-header compile check.
//!
//! The adapter's production ABI reflection unit is compiled as C17 and includes
//! llama.h. It contains no test main, expected values or assertions. The source
//! contract is compilation only; additional layout guards are labeled below.
//! Pinned upstream MIT attribution: tools/prism/upstream/LICENSE.

use serde_json::{Value, json};

use super::engine::Engine;
use super::{Result, invalid, result};

const ID: &str = "tests/test-c.c";

pub(super) fn run(engine: &Engine) -> Result<Value> {
    let Some(response) = engine.execute_common(
        "public-c-abi",
        &json!({"SchemaVersion":1,"Operations":[{"Kind":"public-c-abi"}]}),
    )?
    else {
        return Ok(result(ID, false, "Rust runtime exposes no C public-header ABI; original source is compile-only with zero behavioral assertions".into()));
    };
    if response["SchemaVersion"] != 1 || response["Results"].as_array().map(Vec::len) != Some(1) {
        return Err(invalid("C ABI response framing mismatch"));
    }
    let value = &response["Results"][0];
    let mut guards = 0;
    let mut passed = true;
    for name in ["ModelParams", "ContextParams", "Batch"] {
        let size = value[name]["Size"]
            .as_u64()
            .ok_or_else(|| invalid("C ABI size missing"))?;
        let alignment = value[name]["Alignment"]
            .as_u64()
            .ok_or_else(|| invalid("C ABI alignment missing"))?;
        passed &= alignment.is_power_of_two() && size >= alignment && size % alignment == 0;
        guards += 1;
    }
    for name in ["TokenSize", "PositionSize", "SequenceIdSize"] {
        passed &= value[name]
            .as_u64()
            .ok_or_else(|| invalid("C ABI scalar size missing"))?
            == 4;
        guards += 1;
    }
    passed &= value["CStandard"]
        .as_u64()
        .ok_or_else(|| invalid("C standard observation missing"))?
        >= 201_710;
    guards += 1;
    Ok(result(
        ID,
        passed,
        json!({
            "SourceSha256":"c973865e5e86263a71a1210003e80cffa6269572de31cfa647cc3d01b4bea99e",
            "Coverage":"C public-header compilation; original empty main retired",
            "SourceBehavioralAssertions":0,"AdditionalAbiGuards":guards,"Observations":value,
        })
        .to_string(),
    ))
}
