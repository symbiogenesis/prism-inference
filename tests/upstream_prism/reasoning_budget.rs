//! Complete safe Rust port of upstream test-reasoning-budget.cpp.
//!
//! Source: PrismML-Eng/llama.cpp b87ca7225456b06b0975b8239c66b5aecf07e37c;
//! see tools/prism/upstream/LICENSE for MIT terms. Rust owns all candidate
//! vectors, action sequences, finite-logit observations and expected conditions.
//! Every sampler uses the original null-vocabulary fallback. No reset coverage
//! is claimed: the original source never calls reset.

use serde_json::{Value, json};

use super::engine::Engine;
use super::{Result, invalid, result};

const ID: &str = "tests/test-reasoning-budget.cpp";
const SOURCE_SHA256: &str = "1499e08757ff9e6f52ddd84134a1454346e92cd78a3dc5a9141f8ba775f57979";
const BASIC_CASES: &str = include_str!("reasoning-budget-cases.json");
const IDLE: u32 = 0;
const COUNTING: u32 = 1;
const FORCING: u32 = 2;
const DONE: u32 = 4;

// Protocol adaptation only: all matching, budget state and logit mutation are
// implemented by the independently usable production runtime sampler.
#[allow(
    clippy::too_many_lines,
    reason = "Keep typed action validation and handle ownership together; behavior is in the production sampler."
)]
pub(super) fn execute_runtime(request: &Value) -> Result<Option<Value>> {
    use bonsai_prism_runtime::sampling::reasoning_budget::{
        ReasoningBudget, State, utf8_is_complete,
    };

    exact_fields(request, &["SchemaVersion", "Operations"])?;
    if uint(request, "SchemaVersion")? != 1 {
        return Err(invalid("reasoning request version mismatch"));
    }
    let operations = array(request, "Operations")?;
    if operations.is_empty() || operations.len() > 128 {
        return Err(invalid("reasoning operation count outside bound"));
    }
    if operations.iter().any(|operation| {
        !matches!(
            operation["Kind"].as_str(),
            Some("reasoning-budget-trace" | "utf8-is-complete")
        )
    }) {
        return Ok(None);
    }
    let tokens = |value: &Value| -> Result<Vec<i32>> {
        let values = value
            .as_array()
            .filter(|values| values.len() <= 1024)
            .ok_or_else(|| invalid("reasoning token sequence outside bound"))?;
        values
            .iter()
            .map(|value| {
                Ok(i32::try_from(value.as_i64().ok_or_else(|| {
                    invalid("reasoning token must be int32")
                })?)?)
            })
            .collect()
    };
    let sequences = |value: &Value| -> Result<Vec<Vec<i32>>> {
        value
            .as_array()
            .filter(|values| values.len() <= 64)
            .ok_or_else(|| invalid("reasoning sequence collection outside bound"))?
            .iter()
            .map(tokens)
            .collect()
    };
    let mut results = Vec::with_capacity(operations.len());
    for operation in operations {
        if text(operation, "Kind")? == "utf8-is-complete" {
            exact_fields(operation, &["Kind", "Bytes"])?;
            let values = array(operation, "Bytes")?;
            if values.len() > 65536 {
                return Err(invalid("UTF-8 bytes exceed bound"));
            }
            let bytes = values
                .iter()
                .map(|value| {
                    Ok(u8::try_from(
                        value
                            .as_u64()
                            .ok_or_else(|| invalid("UTF-8 byte must be unsigned"))?,
                    )?)
                })
                .collect::<Result<Vec<_>>>()?;
            results.push(json!({"Complete":utf8_is_complete(&bytes)}));
            continue;
        }
        exact_fields(operation, &["Kind", "Actions"])?;
        let actions = array(operation, "Actions")?;
        if actions.is_empty() || actions.len() > 4096 {
            return Err(invalid("reasoning trace outside action bound"));
        }
        let mut samplers: [Option<ReasoningBudget>; 32] = std::array::from_fn(|_| None);
        let mut trace = Vec::with_capacity(actions.len());
        let mut total_candidates = 0;
        for action in actions {
            let kind = text(action, "Kind")?;
            let handle = field(action, "Sampler")?;
            let index = if handle.is_null() {
                None
            } else {
                Some(usize::try_from(
                    handle
                        .as_u64()
                        .filter(|index| *index < 32)
                        .ok_or_else(|| invalid("reasoning sampler handle outside bound"))?,
                )?)
            };
            let mut event = json!({});
            if kind == "init" {
                exact_fields(
                    action,
                    &[
                        "Kind",
                        "Sampler",
                        "Vocabulary",
                        "StartSequences",
                        "EndSequences",
                        "ForcedTokens",
                        "Budget",
                        "InitialState",
                    ],
                )?;
                let index = index.ok_or_else(|| invalid("reasoning init requires a handle"))?;
                if samplers[index].is_some() || !field(action, "Vocabulary")?.is_null() {
                    return Err(invalid(
                        "reasoning init requires vacant handle and null vocabulary",
                    ));
                }
                samplers[index] = Some(ReasoningBudget::new(
                    sequences(field(action, "StartSequences")?)?,
                    sequences(field(action, "EndSequences")?)?,
                    tokens(field(action, "ForcedTokens")?)?,
                    integer(action, "Budget")?,
                    State::try_from(u32::try_from(uint(action, "InitialState")?)?)?,
                )?);
            } else {
                if index.is_some_and(|index| samplers[index].is_none()) {
                    return Err(invalid("unknown or freed reasoning sampler"));
                }
                match kind {
                    "apply" => {
                        exact_fields(
                            action,
                            &["Kind", "Sampler", "Candidates", "Selected", "Sorted"],
                        )?;
                        let candidates = array(action, "Candidates")?;
                        total_candidates += candidates.len();
                        if candidates.len() > 4096 || total_candidates > 65536 {
                            return Err(invalid("reasoning candidate bound exceeded"));
                        }
                        let selected = field(action, "Selected")?
                            .as_i64()
                            .ok_or_else(|| invalid("reasoning Selected must be signed"))?;
                        if selected < -1
                            || (selected >= 0 && usize::try_from(selected)? >= candidates.len())
                        {
                            return Err(invalid("reasoning Selected outside candidates"));
                        }
                        let sorted = field(action, "Sorted")?
                            .as_bool()
                            .ok_or_else(|| invalid("reasoning Sorted must be boolean"))?;
                        let mut ids = Vec::with_capacity(candidates.len());
                        let mut logits = Vec::with_capacity(candidates.len());
                        for candidate in candidates {
                            exact_fields(candidate, &["Id", "LogitBits", "ProbabilityBits"])?;
                            ids.push(integer(candidate, "Id")?);
                            logits.push(f32::from_bits(u32::try_from(uint(
                                candidate,
                                "LogitBits",
                            )?)?));
                            u32::try_from(uint(candidate, "ProbabilityBits")?)?;
                        }
                        if let Some(sampler) = index.and_then(|index| samplers[index].as_ref()) {
                            sampler.apply(&ids, &mut logits)?;
                        }
                        let mut values = candidates.to_vec();
                        for (candidate, logit) in values.iter_mut().zip(logits) {
                            candidate["LogitBits"] = json!(logit.to_bits());
                        }
                        event = json!({"Candidates":values,"Selected":selected,"Sorted":sorted});
                    }
                    "accept" => {
                        exact_fields(action, &["Kind", "Sampler", "Token"])?;
                        let token = integer(action, "Token")?;
                        if let Some(sampler) = index.and_then(|index| samplers[index].as_mut()) {
                            sampler.accept(token);
                        }
                    }
                    "clone" => {
                        exact_fields(action, &["Kind", "Sampler", "Target"])?;
                        let target = usize::try_from(uint(action, "Target")?)?;
                        if target >= 32 || samplers[target].is_some() {
                            return Err(invalid("reasoning clone target not vacant"));
                        }
                        samplers[target] = index.and_then(|index| samplers[index].clone());
                    }
                    "state" | "force" | "end-match" | "free" => {
                        exact_fields(action, &["Kind", "Sampler"])?;
                        let sampler = index.and_then(|index| samplers[index].as_mut());
                        event = match kind {
                            "state" => {
                                json!({"State":match sampler.map_or(State::Idle, |sampler| sampler.state()) {
                                    State::Idle => 0, State::Counting => 1, State::Forcing => 2, State::WaitingUtf8 => 3, State::Done => 4,
                                }})
                            }
                            "force" => {
                                json!({"Forced":sampler.is_some_and(ReasoningBudget::force)})
                            }
                            "end-match" => {
                                json!({"EndMatch":sampler.and_then(|sampler| sampler.end_match())})
                            }
                            _ => {
                                if let Some(index) = index {
                                    samplers[index] = None;
                                }
                                json!({})
                            }
                        };
                    }
                    _ => return Err(invalid("unknown reasoning action")),
                }
            }
            trace.push(event);
        }
        results.push(json!({"Trace":trace}));
    }
    Ok(Some(json!({"SchemaVersion":1,"Results":results})))
}

#[derive(Default)]
struct Assertions {
    count: usize,
    failures: Vec<String>,
}

impl Assertions {
    fn check(&mut self, name: impl Into<String>, passed: bool) {
        self.count += 1;
        if !passed {
            self.failures.push(name.into());
        }
    }
}

fn field<'a>(value: &'a Value, name: &str) -> Result<&'a Value> {
    value
        .get(name)
        .ok_or_else(|| invalid(format!("missing reasoning-budget field {name}")))
}

fn array<'a>(value: &'a Value, name: &str) -> Result<&'a [Value]> {
    field(value, name)?
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| invalid(format!("reasoning-budget {name} must be an array")))
}

fn text<'a>(value: &'a Value, name: &str) -> Result<&'a str> {
    field(value, name)?
        .as_str()
        .ok_or_else(|| invalid(format!("reasoning-budget {name} must be text")))
}

fn uint(value: &Value, name: &str) -> Result<u64> {
    field(value, name)?
        .as_u64()
        .ok_or_else(|| invalid(format!("reasoning-budget {name} must be unsigned")))
}

fn integer(value: &Value, name: &str) -> Result<i32> {
    Ok(i32::try_from(field(value, name)?.as_i64().ok_or_else(
        || invalid(format!("reasoning-budget {name} must be int32")),
    )?)?)
}

fn exact_fields(value: &Value, fields: &[&str]) -> Result<()> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid("reasoning-budget result must be an object"))?;
    if object.len() != fields.len() || fields.iter().any(|key| !object.contains_key(*key)) {
        return Err(invalid("reasoning-budget result field mismatch"));
    }
    Ok(())
}

fn exchange(engine: &Engine, operations: &[Value]) -> Result<Option<Vec<Value>>> {
    let mut output = Vec::with_capacity(operations.len());
    for chunk in operations.chunks(8) {
        let Some(response) = engine.execute_common(
            "reasoning-budget",
            &json!({"SchemaVersion":1,"Operations":chunk}),
        )?
        else {
            return Ok(None);
        };
        exact_fields(&response, &["SchemaVersion", "Results"])?;
        if uint(&response, "SchemaVersion")? != 1 {
            return Err(invalid("reasoning-budget common schema version mismatch"));
        }
        let results = array(&response, "Results")?;
        if results.len() != chunk.len() {
            return Err(invalid("reasoning-budget operation count mismatch"));
        }
        output.extend_from_slice(results);
    }
    Ok(Some(output))
}

fn candidates(max_token: i32) -> Result<Vec<Value>> {
    (0..=u16::try_from(max_token)?).map(|token| {
        let value = token.checked_add(1).ok_or_else(|| invalid("reasoning candidate extent overflow"))?;
        Ok(json!({"Id":i32::from(token),"LogitBits":f32::from(value).ln().to_bits(),"ProbabilityBits":0}))
    }).collect()
}

fn apply(sampler: u32, max_token: i32) -> Result<Value> {
    Ok(
        json!({"Kind":"apply", "Sampler":sampler, "Candidates":candidates(max_token)?, "Selected":-1, "Sorted":false}),
    )
}

fn finite_tokens(value: &Value, count: usize) -> Result<Vec<i32>> {
    exact_fields(value, &["Candidates", "Selected", "Sorted"])?;
    if field(value, "Selected")?.as_i64().is_none() || field(value, "Sorted")?.as_bool().is_none() {
        return Err(invalid("reasoning apply metadata type mismatch"));
    }
    let candidates = array(value, "Candidates")?;
    if candidates.len() != count {
        return Err(invalid("reasoning apply candidate extent mismatch"));
    }
    let mut tokens = Vec::new();
    for candidate in candidates {
        exact_fields(candidate, &["Id", "LogitBits", "ProbabilityBits"])?;
        let token = integer(candidate, "Id")?;
        let logit = f32::from_bits(u32::try_from(uint(candidate, "LogitBits")?)?);
        let _ = u32::try_from(uint(candidate, "ProbabilityBits")?)?;
        if logit.is_finite() {
            tokens.push(token);
        }
    }
    Ok(tokens)
}

fn initialize(start: &Value, end: &Value, forced: &Value, budget: i32, state: u32) -> Value {
    json!({"Kind":"init", "Sampler":0, "Vocabulary":null, "StartSequences":start, "EndSequences":end, "ForcedTokens":forced, "Budget":budget, "InitialState":state})
}

fn max_token(case: &Value) -> Result<i32> {
    let mut maximum = 0;
    for field_name in ["Sequence", "ForcedTokens"] {
        for token in array(case, field_name)? {
            maximum = maximum.max(i32::try_from(
                token
                    .as_i64()
                    .ok_or_else(|| invalid("reasoning token must be int32"))?,
            )?);
        }
    }
    for field_name in ["StartSequences", "EndSequences"] {
        for sequence in array(case, field_name)? {
            for token in sequence
                .as_array()
                .ok_or_else(|| invalid("reasoning sequence must be an array"))?
            {
                maximum = maximum.max(i32::try_from(
                    token
                        .as_i64()
                        .ok_or_else(|| invalid("reasoning token must be int32"))?,
                )?);
            }
        }
    }
    Ok(maximum)
}

fn trace<'a>(result: &'a Value, actions: &[Value]) -> Result<&'a [Value]> {
    exact_fields(result, &["Trace"])?;
    let observations = array(result, "Trace")?;
    if observations.len() != actions.len() {
        return Err(invalid("reasoning action trace extent mismatch"));
    }
    for (action, observation) in actions.iter().zip(observations) {
        if matches!(
            text(action, "Kind")?,
            "init" | "accept" | "clone" | "reset" | "free"
        ) {
            exact_fields(observation, &[])?;
        }
    }
    Ok(observations)
}

fn basic_operation(case: &Value) -> Result<Value> {
    let mut actions = vec![initialize(
        field(case, "StartSequences")?,
        field(case, "EndSequences")?,
        field(case, "ForcedTokens")?,
        integer(case, "Budget")?,
        u32::try_from(uint(case, "InitialState")?)?,
    )];
    let maximum = max_token(case)?;
    for token in array(case, "Sequence")? {
        actions.push(apply(0, maximum)?);
        actions.push(json!({"Kind":"accept", "Sampler":0, "Token":token}));
    }
    actions.push(json!({"Kind":"free", "Sampler":0}));
    Ok(json!({"Kind":"reasoning-budget-trace", "Actions":actions}))
}

fn basic_assertions(cases: &[Value], values: &[Value], assertions: &mut Assertions) -> Result<()> {
    for (case, value) in cases.iter().zip(values) {
        let operation = basic_operation(case)?;
        let observations = trace(value, array(&operation, "Actions")?)?;
        let mut force_start = None;
        let mut force_end = None;
        let candidate_count = usize::try_from(max_token(case)?)? + 1;
        for index in 0..array(case, "Sequence")?.len() {
            let finite = finite_tokens(&observations[1 + index * 2], candidate_count)?;
            if finite.len() == 1 {
                force_start.get_or_insert(index);
                force_end = Some(index);
            } else if force_start.is_some() && force_end.is_some() {
                // Preserve the original analysis break. The finite trace can
                // include one later accept on the already-DONE sampler in the
                // budget=0 case; that production accept is a no-op before free.
                break;
            }
        }
        let name = text(case, "Name")?;
        let expected_start = field(case, "ExpectedForceStart")?;
        if expected_start.is_null() {
            assertions.check(format!("{name}: no forcing"), force_start.is_none());
        } else {
            let expected = usize::try_from(
                expected_start
                    .as_u64()
                    .ok_or_else(|| invalid("force start must be unsigned or null"))?,
            )?;
            assertions.check(format!("{name}: forcing occurred"), force_start.is_some());
            assertions.check(
                format!("{name}: force start"),
                force_start == Some(expected),
            );
        }
        let expected_end = field(case, "ExpectedForceEnd")?;
        if !expected_end.is_null() {
            let minimum = usize::try_from(
                expected_end
                    .as_u64()
                    .ok_or_else(|| invalid("force end must be unsigned or null"))?,
            )?;
            // Original comparison is SIZE_MAX < minimum when no forcing
            // occurred. Keep that sentinel behavior and the >= lower bound.
            assertions.check(
                format!("{name}: force end lower bound"),
                force_end.is_none_or(|end| end >= minimum),
            );
        }
    }
    Ok(())
}

enum Expected {
    Forced(i32),
    State(u32),
    Force(bool),
    EndNone,
    EndSome(Vec<i32>),
}

struct Script {
    name: &'static str,
    actions: Vec<Value>,
    expected: Vec<(usize, Expected)>,
}

impl Script {
    const fn empty(name: &'static str) -> Self {
        Self {
            name,
            actions: Vec::new(),
            expected: Vec::new(),
        }
    }

    fn new(name: &'static str, budget: i32, state: u32) -> Self {
        let mut script = Self::empty(name);
        script.actions.push(initialize(
            &json!([[100]]),
            &json!([[101]]),
            &json!([102, 101]),
            budget,
            state,
        ));
        script
    }

    fn end_match(name: &'static str, forced: &[i32], budget: i32, state: u32) -> Self {
        let mut script = Self::empty(name);
        script.actions.push(initialize(
            &json!([[100]]),
            &json!([[101], [103, 104]]),
            &json!(forced),
            budget,
            state,
        ));
        script
    }

    fn observe(&mut self, action: Value, expected: Expected) {
        self.expected.push((self.actions.len(), expected));
        self.actions.push(action);
    }

    fn accept(&mut self, sampler: u32, token: i32) {
        self.actions
            .push(json!({"Kind":"accept", "Sampler":sampler, "Token":token}));
    }

    fn forced(&mut self, sampler: u32, token: i32) -> Result<()> {
        self.observe(apply(sampler, 102)?, Expected::Forced(token));
        Ok(())
    }

    fn state(&mut self, state: u32) {
        self.observe(json!({"Kind":"state", "Sampler":0}), Expected::State(state));
    }

    fn force(&mut self, forced: bool) {
        self.observe(
            json!({"Kind":"force", "Sampler":0}),
            Expected::Force(forced),
        );
    }

    fn end(&mut self, tokens: Option<&[i32]>) {
        self.observe(
            json!({"Kind":"end-match", "Sampler":0}),
            tokens.map_or(Expected::EndNone, |tokens| {
                Expected::EndSome(tokens.to_vec())
            }),
        );
    }

    fn free(&mut self, sampler: u32) {
        self.actions.push(json!({"Kind":"free", "Sampler":sampler}));
    }

    fn operation(&self) -> Value {
        json!({"Kind":"reasoning-budget-trace", "Actions":self.actions})
    }
}

fn clone_scripts() -> Result<Vec<Script>> {
    let mut counting = Script::new("clone mid counting", 2, IDLE);
    counting.accept(0, 100);
    counting.accept(0, 50);
    counting
        .actions
        .push(json!({"Kind":"clone", "Sampler":0, "Target":1}));
    counting.accept(1, 51);
    counting.forced(1, 102)?;
    counting.free(1);
    counting.free(0);
    let mut forcing = Script::new("clone mid forcing", 0, FORCING);
    forcing.forced(0, 102)?;
    forcing.accept(0, 102);
    forcing
        .actions
        .push(json!({"Kind":"clone", "Sampler":0, "Target":1}));
    forcing.forced(1, 101)?;
    forcing.free(1);
    forcing.free(0);
    Ok(vec![counting, forcing])
}

fn manual_scripts() -> Result<Vec<Script>> {
    let mut counting = Script::new("manual force counting", 5, IDLE);
    counting.accept(0, 100);
    counting.accept(0, 50);
    counting.state(COUNTING);
    counting.force(true);
    counting.state(FORCING);
    counting.forced(0, 102)?;
    counting.accept(0, 102);
    counting.forced(0, 101)?;
    counting.accept(0, 101);
    counting.state(DONE);
    counting.free(0);
    let mut idle = Script::new("manual force idle", 5, IDLE);
    idle.force(false);
    idle.state(IDLE);
    idle.free(0);
    let mut done = Script::new("manual force done", 5, IDLE);
    done.accept(0, 100);
    done.accept(0, 101);
    done.state(DONE);
    done.force(false);
    done.state(DONE);
    done.free(0);
    let mut forcing = Script::new("manual force already forcing", 0, FORCING);
    forcing.forced(0, 102)?;
    forcing.accept(0, 102);
    forcing.force(false);
    forcing.state(FORCING);
    forcing.forced(0, 101)?;
    forcing.free(0);
    let mut null = Script::empty("manual force null");
    null.observe(
        json!({"Kind":"force", "Sampler":null}),
        Expected::Force(false),
    );
    Ok(vec![counting, idle, done, forcing, null])
}

fn end_scripts() -> Vec<Script> {
    let mut natural = Script::end_match("natural end and rearm", &[102, 101], 5, IDLE);
    natural.end(None);
    for token in [100, 50, 103, 104] {
        natural.accept(0, token);
    }
    natural.end(Some(&[103, 104]));
    natural.accept(0, 100);
    natural.end(None);
    natural.free(0);
    let mut overlap = Script::empty("longest overlapping end");
    overlap.actions.push(initialize(
        &json!([[100]]),
        &json!([[104], [103, 104]]),
        &json!([102, 104]),
        5,
        IDLE,
    ));
    for token in [100, 103, 104] {
        overlap.accept(0, token);
    }
    overlap.end(Some(&[103, 104]));
    overlap.state(DONE);
    overlap.free(0);
    let mut forced = Script::end_match("forced sequence records end", &[102, 103, 104], 0, FORCING);
    forced.accept(0, 102);
    forced.accept(0, 103);
    forced.end(None);
    forced.accept(0, 104);
    forced.end(Some(&[103, 104]));
    forced.free(0);
    let mut unknown = Script::end_match("forced sequence without known end", &[102], 0, FORCING);
    unknown.accept(0, 102);
    unknown.state(DONE);
    unknown.end(None);
    unknown.free(0);
    let mut null = Script::empty("matched end null");
    null.observe(
        json!({"Kind":"end-match", "Sampler":null}),
        Expected::EndNone,
    );
    vec![natural, overlap, forced, unknown, null]
}

fn validate_script(script: &Script, result: &Value, assertions: &mut Assertions) -> Result<()> {
    let observations = trace(result, &script.actions)?;
    for (index, expected) in &script.expected {
        let value = &observations[*index];
        let label = format!("{} action {index}", script.name);
        match expected {
            Expected::Forced(token) => {
                let finite = finite_tokens(value, 103)?;
                assertions.check(
                    format!("{label}: exactly one finite logit"),
                    finite.len() == 1,
                );
                assertions.check(
                    format!("{label}: forced token"),
                    finite.last() == Some(token),
                );
            }
            Expected::State(state) => {
                exact_fields(value, &["State"])?;
                assertions.check(
                    format!("{label}: state"),
                    uint(value, "State")? == u64::from(*state),
                );
            }
            Expected::Force(expected) => {
                exact_fields(value, &["Forced"])?;
                assertions.check(
                    format!("{label}: manual force"),
                    field(value, "Forced")?
                        .as_bool()
                        .ok_or_else(|| invalid("reasoning Forced must be boolean"))?
                        == *expected,
                );
            }
            Expected::EndNone => {
                exact_fields(value, &["EndMatch"])?;
                assertions.check(
                    format!("{label}: no end match"),
                    field(value, "EndMatch")?.is_null(),
                );
            }
            Expected::EndSome(tokens) => {
                exact_fields(value, &["EndMatch"])?;
                let actual = field(value, "EndMatch")?;
                assertions.check(format!("{label}: end match exists"), !actual.is_null());
                if !actual.is_null() {
                    let actual = actual.as_array().ok_or_else(|| {
                        invalid("reasoning EndMatch must be a token array or null")
                    })?;
                    let actual = actual
                        .iter()
                        .map(|token| {
                            Ok(i32::try_from(
                                token
                                    .as_i64()
                                    .ok_or_else(|| invalid("end token must be int32"))?,
                            )?)
                        })
                        .collect::<Result<Vec<_>>>()?;
                    assertions.check(format!("{label}: matched sequence"), actual == *tokens);
                }
            }
        }
    }
    Ok(())
}

fn utf8_cases() -> Vec<(Vec<u8>, bool)> {
    vec![
        (b"hello".to_vec(), true),
        (vec![], true),
        (vec![0xc2, 0xa0], true),
        (vec![0xe2, 0x80, 0x9c], true),
        (vec![0xf0, 0x9f, 0x98, 0x80], true),
        (vec![b'a', b'b', b'c', 0xc3, 0xa9], true),
        (vec![0xc2], false),
        (vec![0xe2, 0x80], false),
        (vec![0xe2], false),
        (vec![0xf0, 0x9f, 0x98], false),
        (vec![0xf0, 0x9f], false),
        (vec![0xf0], false),
        (vec![0x80], false),
        (vec![b'h', b'e', b'l', b'l', b'o', 0xc3], false),
        (vec![b'h', b'e', b'l', b'l', b'o', 0xc3, 0xa9], true),
    ]
}

pub(super) fn run(engine: &Engine) -> Result<Value> {
    let fixture: Value = serde_json::from_str(BASIC_CASES)?;
    if uint(&fixture, "SchemaVersion")? != 1
        || text(&fixture, "Source")? != ID
        || text(&fixture, "SourceSha256")? != SOURCE_SHA256
    {
        return Err(invalid("reasoning-budget source provenance mismatch"));
    }
    let cases = array(&fixture, "Cases")?;
    if cases.len() != 8 {
        return Err(invalid("all 8 original basic reasoning cases are required"));
    }
    let mut operations = cases
        .iter()
        .map(basic_operation)
        .collect::<Result<Vec<_>>>()?;
    let mut scripts = clone_scripts()?;
    scripts.extend(manual_scripts()?);
    scripts.extend(end_scripts());
    operations.extend(scripts.iter().map(Script::operation));
    let utf8 = utf8_cases();
    operations.extend(
        utf8.iter()
            .map(|(bytes, _)| json!({"Kind":"utf8-is-complete","Bytes":bytes})),
    );
    let Some(values) = exchange(engine, &operations)? else {
        return Ok(result(ID,false,"unsupported production API: reasoning-budget sampler and UTF-8 boundary helper; all 71 original semantic conditions required".to_owned()));
    };
    let mut assertions = Assertions::default();
    basic_assertions(cases, &values[..cases.len()], &mut assertions)?;
    let script_end = cases.len() + scripts.len();
    for (script, value) in scripts.iter().zip(&values[cases.len()..script_end]) {
        validate_script(script, value, &mut assertions)?;
    }
    for (index, ((_, expected), value)) in utf8.iter().zip(&values[script_end..]).enumerate() {
        exact_fields(value, &["Complete"])?;
        let complete = field(value, "Complete")?
            .as_bool()
            .ok_or_else(|| invalid("UTF-8 Complete must be boolean"))?;
        assertions.check(
            format!("UTF-8 boundary case {index}"),
            complete == *expected,
        );
    }
    Ok(result(
        ID,
        assertions.failures.is_empty(),
        format!(
            "8 basic sequences; clone counting/forcing; manual force/null; end-match/rearm/overlap/null; 15 UTF-8 boundaries; 71 required semantic conditions; assertions={}; failed={}; {}",
            assertions.count,
            assertions.failures.len(),
            assertions.failures.join("; ")
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_raw_logit_extent_types_and_nonfinite_observations() -> Result<()> {
        let mut observation = json!({
            "Candidates":[
                {"Id":0,"LogitBits":f32::NEG_INFINITY.to_bits(),"ProbabilityBits":0},
                {"Id":1,"LogitBits":f32::NAN.to_bits(),"ProbabilityBits":0},
                {"Id":2,"LogitBits":1.0_f32.to_bits(),"ProbabilityBits":0},
            ],
            "Selected":-1,"Sorted":false,
        });
        assert_eq!(finite_tokens(&observation, 3)?, vec![2]);
        assert!(finite_tokens(&observation, 2).is_err());
        observation["Candidates"][0]["LogitBits"] = json!(u64::from(u32::MAX) + 1);
        assert!(finite_tokens(&observation, 3).is_err());
        observation["Candidates"][0]["LogitBits"] = json!(0);
        observation["Sorted"] = json!("false");
        assert!(finite_tokens(&observation, 3).is_err());
        Ok(())
    }
}
