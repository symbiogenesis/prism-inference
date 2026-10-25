//! Safe Rust schedule for the complete Windows test-log.cpp stress source.
//!
//! The source has no output-content assertions. Its contract is successful
//! concurrent logging, joining, flushing and explicit singleton destruction.
//! Source: PrismML-Eng/llama.cpp b87ca7225456b06b0975b8239c66b5aecf07e37c;
//! upstream MIT attribution is in tools/prism/upstream/LICENSE.

use serde_json::{Value, json};

use super::engine::Engine;
use super::{Result, invalid, result};

const ID: &str = "tests/test-log.cpp";

// Windows CRT rand state belongs to each thread and initially equals one.
const fn random(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(214_013).wrapping_add(2_531_011);
    (*state >> 16) & 0x7fff
}

pub(super) fn run(engine: &Engine) -> Result<Value> {
    let mut workers = Vec::with_capacity(8);
    for thread in 0..8 {
        let mut state = 1;
        let mut actions = Vec::new();
        for message in 0..1000 {
            let level =
                ["info", "warning", "error", "debug"][usize::try_from(random(&mut state) % 4)?];
            actions.push(
                json!({"Kind":"emit","Level":level,"Text":format!("Thread {thread}: {message}\n")}),
            );
            if random(&mut state) % 10 < 5 {
                actions.push(json!({"Kind":"timestamps","Value":random(&mut state) % 2 != 0}));
                actions.push(json!({"Kind":"prefix","Value":random(&mut state) % 2 != 0}));
            }
        }
        workers.push(actions);
    }
    let request = json!({"SchemaVersion":1,"Operations":[{"Kind":"concurrent-log","Workers":workers,"Finalize":["flush","free"]}]});
    let Some(response) = engine.execute_common("logging", &request)? else {
        return Ok(result(ID, false, "Rust production concurrent common logger API is unavailable; all eight workers and 8,000 messages remain required.".to_owned()));
    };
    let completed = response["Results"][0]["Completed"]
        .as_array()
        .ok_or_else(|| invalid("logger response must contain worker completion counts"))?;
    if response["SchemaVersion"] != 1
        || response["Results"].as_array().map(Vec::len) != Some(1)
        || completed.len() != workers.len()
    {
        return Err(invalid("logger response shape mismatch"));
    }
    let passed = completed
        .iter()
        .zip(&workers)
        .all(|(count, actions)| count.as_u64() == u64::try_from(actions.len()).ok());
    Ok(result(ID, passed, "All eight original workers, 1,000 messages per worker, Windows CRT random schedules, concurrent timestamp/prefix changes, joins, flush and explicit logger free completed. Source has no output-content assertions.".to_owned()))
}
