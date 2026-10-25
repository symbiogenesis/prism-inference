//! Complete test-mtmd-c-api.c checks expressed in safe Rust.
//!
//! The public upstream fixture factory remains a production-library operation;
//! every traversal, save/load request, comparison and verdict belongs to Rust.
//! Upstream MIT attribution: tools/prism/upstream/LICENSE.

use serde_json::{Value, json};

use super::engine::Engine;
use super::llama_grammar::{array, exchange, uint};
use super::{Result, invalid, result};

const ID: &str = "tests/test-mtmd-c-api.c";

#[path = "mtmd_chunks_runtime.rs"]
mod runtime;

pub(super) fn execute_runtime(request: &Value) -> Result<Option<Value>> {
    runtime::execute(request)
}

#[derive(Default)]
struct Checks {
    count: usize,
    failures: Vec<String>,
}
impl Checks {
    fn check(&mut self, passed: bool, detail: impl Into<String>) {
        self.count += 1;
        if !passed {
            self.failures.push(detail.into());
        }
    }
}

fn factory() -> Value {
    json!({"Kind":"create-test-inputs","Chunks":0})
}
fn get(index: usize) -> Value {
    json!({"Kind":"get","Chunks":0,"Index":index,"Chunk":0})
}
fn snapshot(chunk: usize) -> Value {
    json!({"Kind":"snapshot","Chunk":chunk})
}

fn trace(engine: &Engine, actions: &[Value]) -> Result<Option<Vec<Value>>> {
    let operation = json!({"Kind":"mtmd-chunks-trace","Actions":actions});
    let Some(values) = exchange(engine, &[operation])? else {
        return Ok(None);
    };
    let events = array(&values[0], "Events")?;
    if events.len() != actions.len() {
        return Err(invalid("MTMD chunk action count mismatch"));
    }
    Ok(Some(events.to_vec()))
}

fn required_trace(engine: &Engine, actions: &[Value]) -> Result<Vec<Value>> {
    trace(engine, actions)?.ok_or_else(|| invalid("MTMD production chunk API disappeared"))
}

fn metadata_checks(
    engine: &Engine,
    index: usize,
    metadata: &Value,
    checks: &mut Checks,
) -> Result<()> {
    match uint(metadata, "Type")? {
        0 => {
            checks.check(
                metadata["TokensPresent"] == true,
                format!("chunk {index}: text tokens present"),
            );
            checks.check(
                uint(metadata, "TextCount")? > 0,
                format!("chunk {index}: positive text count"),
            );
            let tokens = array(metadata, "TextTokens")?;
            if u64::try_from(tokens.len())? != uint(metadata, "TextCount")? {
                return Err(invalid("MTMD text observation extent mismatch"));
            }
            for (token, value) in tokens.iter().enumerate() {
                checks.check(
                    value.as_i64().is_some_and(|value| value >= 0),
                    format!("chunk {index}: token {token} nonnegative"),
                );
            }
        }
        1 => {
            let count = uint(metadata, "ImageCount")?;
            checks.check(
                count > 0,
                format!("chunk {index}: positive image token count"),
            );
            let position = required_trace(
                engine,
                &[
                    factory(),
                    get(index),
                    json!({"Kind":"image-position","Chunk":0,"Index":count.checked_sub(1).ok_or_else(||invalid("original image has no last position"))?,"Position0":0}),
                ],
            )?;
            checks.check(
                uint(&position[2], "X")?
                    .checked_add(1)
                    .is_some_and(|value| value > 0),
                format!("chunk {index}: positive nx"),
            );
            checks.check(
                uint(&position[2], "Y")?
                    .checked_add(1)
                    .is_some_and(|value| value > 0),
                format!("chunk {index}: positive ny"),
            );
            checks.check(
                metadata["ImageId"].is_string(),
                format!("chunk {index}: image ID present"),
            );
        }
        _ => {}
    }
    Ok(())
}

fn roundtrip(engine: &Engine, index: usize, original: &Value, checks: &mut Checks) -> Result<()> {
    let query = required_trace(
        engine,
        &[
            factory(),
            get(index),
            json!({"Kind":"query-save","Chunk":0}),
        ],
    )?;
    checks.check(
        query[1]["Present"] == true,
        format!("roundtrip chunk {index}: source present"),
    );
    checks.check(
        query[2]["ReturnCode"] == 0,
        format!("chunk {index}: save query returns zero"),
    );
    let size = usize::try_from(uint(&query[2], "ExpectedLength")?)?;
    checks.check(size > 0, format!("chunk {index}: save length positive"));
    if size == 0 || size > 1024 * 1024 {
        return Err(invalid("MTMD serialization length outside bound"));
    }
    let mut actions = vec![
        factory(),
        get(index),
        json!({"Kind":"save","Chunk":0,"Buffer":0,"Length":1}),
        json!({"Kind":"save","Chunk":0,"Buffer":1,"Length":size}),
    ];
    let truncated = if size > 1 {
        let position = actions.len();
        actions.push(json!({"Kind":"load","Buffer":1,"Length":size-1,"Chunk":1}));
        Some(position)
    } else {
        None
    };
    let loaded = actions.len();
    actions.push(json!({"Kind":"load","Buffer":1,"Length":size,"Chunk":2}));
    let observed = actions.len();
    actions.push(snapshot(2));
    actions.push(json!({"Kind":"free","Chunk":2}));
    if truncated.is_some() {
        actions.push(json!({"Kind":"free","Chunk":1}));
    }
    let values = required_trace(engine, &actions)?;
    checks.check(
        values[2]["ReturnCode"]
            .as_i64()
            .is_some_and(|code| code != 0),
        format!("chunk {index}: tiny save rejected"),
    );
    checks.check(
        values[3]["Allocated"] == true,
        format!("chunk {index}: full save buffer allocated"),
    );
    checks.check(
        values[3]["ReturnCode"] == 0,
        format!("chunk {index}: full save returns zero"),
    );
    if let Some(truncated) = truncated {
        checks.check(
            values[truncated]["Created"] == false,
            format!("chunk {index}: truncated load rejected"),
        );
    }
    checks.check(
        values[loaded]["Created"] == true,
        format!("chunk {index}: full load succeeds"),
    );
    let copy = &values[observed];
    for name in ["Type", "TokenCount", "PositionCount"] {
        checks.check(
            copy[name] == original[name],
            format!("chunk {index}: preserved {name}"),
        );
    }
    if original["Type"] == 0 {
        let before = array(original, "TextTokens")?;
        let after = array(copy, "TextTokens")?;
        checks.check(
            copy["TextCount"] == original["TextCount"],
            format!("chunk {index}: preserved text count"),
        );
        for (token, expected) in before.iter().enumerate() {
            checks.check(
                after.get(token) == Some(expected),
                format!("chunk {index}: preserved token {token}"),
            );
        }
    } else if original["Type"] == 1 || original["Type"] == 2 {
        checks.check(
            original["Id"].is_string() && copy["Id"].is_string(),
            format!("chunk {index}: both IDs present"),
        );
        checks.check(
            original["Id"] == copy["Id"],
            format!("chunk {index}: ID unchanged"),
        );
    }
    Ok(())
}

pub(super) fn run(engine: &Engine) -> Result<Value> {
    let Some(initial) = trace(
        engine,
        &[
            json!({"Kind":"defaults"}),
            factory(),
            json!({"Kind":"collection-info","Chunks":0}),
        ],
    )?
    else {
        return Ok(result(ID,false,"Rust production MTMD chunk APIs are unavailable; all original factory/traversal/serialization checks remain required.".to_owned()));
    };
    let mut checks = Checks::default();
    checks.check(initial[1]["Created"] == true, "fixture collection created");
    checks.check(
        uint(&initial[2], "TotalTokens")? > 0,
        "helper total tokens positive",
    );
    let count = usize::try_from(uint(&initial[2], "Count")?)?;
    checks.check(count > 0, "collection nonempty");
    if count > 4096 {
        return Err(invalid("MTMD collection outside bound"));
    }
    let mut originals = Vec::new();
    for index in 0..count {
        let values = required_trace(engine, &[factory(), get(index), snapshot(0)])?;
        checks.check(
            values[1]["Present"] == true,
            format!("chunk {index}: source present"),
        );
        metadata_checks(engine, index, &values[2], &mut checks)?;
        originals.push(values[2].clone());
    }
    for (index, metadata) in originals.iter().enumerate() {
        roundtrip(engine, index, metadata, &mut checks)?;
    }
    if checks.count != 46 {
        return Err(invalid("pinned MTMD C API source condition count drifted"));
    }
    Ok(result(
        ID,
        checks.failures.is_empty(),
        format!(
            "All46 original creation/traversal/text/image/serialization conditions; chunks={count}; public upstream fixture factory retained as production operation; all requests and assertions in Rust; failed={}: {}",
            checks.failures.len(),
            checks.failures.join("; ")
        ),
    ))
}

#[cfg(test)]
mod tests {
    use bonsai_prism_runtime::media::chunks::{self, Chunk};

    use super::Result;

    #[test]
    fn owned_cache_preserves_full_metadata_and_rejects_corrupt_extents() -> Result<()> {
        for chunk in chunks::test_inputs() {
            let bytes = chunk.save()?;
            // The original source roundtrip checks protocol-visible fields;
            // retain the stronger typed equality for every image/frame field.
            assert_eq!(Chunk::load(&bytes)?, chunk);
            let mut corrupted = bytes;
            corrupted[12..20].copy_from_slice(&u64::MAX.to_le_bytes());
            assert!(Chunk::load(&corrupted).is_err());
        }
        Ok(())
    }
}
