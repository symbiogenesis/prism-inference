//! Safe Rust ownership of every default test-sampling.cpp assertion.
//!
//! Source: PrismML-Eng/llama.cpp b87ca7225456b06b0975b8239c66b5aecf07e37c;
//! see tools/prism/upstream/LICENSE for MIT terms. Literal probability inputs,
//! expected probabilities, candidate construction, queue references and RNG
//! checks live here. Native candidate buffers retain all mutations between
//! filters and distribution sampling. The original performance path is a
//! separate CI-only entry and never contributes to correctness results.

use std::ffi::OsStr;

use serde_json::{Value, json};

use super::engine::Engine;
use super::llama_grammar::{array, exact_fields, field, uint};
use super::{Result, invalid, result};

const ID: &str = "tests/test-sampling.cpp";
const SOURCE_SHA256: &str = "817543792852df22e78f705192793ec5797c872708274d572c24c2fb0c95a8ac";
const CASES: &str = include_str!("sampling-cases.json");
const EXPECTED_CONDITIONS: usize = 343;
const MAX_CANDIDATES: usize = 131_072;

use bonsai_prism_runtime::sampling::upstream::{
    Candidate as RuntimeCandidate, Candidates as RuntimeCandidates, Config, State,
};

fn runtime_config(action: &Value) -> Result<Config> {
    let parameters = field(action, "Parameters")?;
    let probability = || float_bits(parameters, "ProbabilityBits");
    let keep = || {
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>(usize::try_from(uint(
            parameters, "MinKeep",
        )?)?)
    };
    Ok(match text(action, "Type")? {
        "temp" => {
            exact_fields(parameters, &["TemperatureBits"])?;
            Config::Temperature(float_bits(parameters, "TemperatureBits")?)
        }
        "temp-ext" => {
            exact_fields(
                parameters,
                &["TemperatureBits", "DeltaBits", "ExponentBits"],
            )?;
            Config::ExtendedTemperature {
                temperature: float_bits(parameters, "TemperatureBits")?,
                delta: float_bits(parameters, "DeltaBits")?,
                exponent: float_bits(parameters, "ExponentBits")?,
            }
        }
        "dist" => {
            exact_fields(parameters, &["Seed"])?;
            Config::Distribution(u32::try_from(uint(parameters, "Seed")?)?)
        }
        "top-k" => {
            exact_fields(parameters, &["K"])?;
            Config::TopK(integer(parameters, "K")?)
        }
        kind @ ("top-p" | "min-p" | "typical") => {
            exact_fields(parameters, &["ProbabilityBits", "MinKeep"])?;
            match kind {
                "top-p" => Config::TopP {
                    probability: probability()?,
                    min_keep: keep()?,
                },
                "min-p" => Config::MinP {
                    probability: probability()?,
                    min_keep: keep()?,
                },
                _ => Config::Typical {
                    probability: probability()?,
                    min_keep: keep()?,
                },
            }
        }
        "xtc" => {
            exact_fields(
                parameters,
                &["ProbabilityBits", "ThresholdBits", "MinKeep", "Seed"],
            )?;
            Config::Xtc {
                probability: probability()?,
                threshold: float_bits(parameters, "ThresholdBits")?,
                min_keep: keep()?,
                seed: u32::try_from(uint(parameters, "Seed")?)?,
            }
        }
        "penalties" => {
            exact_fields(
                parameters,
                &[
                    "VocabularySize",
                    "PenaltyLastN",
                    "RepeatBits",
                    "FrequencyBits",
                    "PresenceBits",
                ],
            )?;
            Config::Penalties {
                vocabulary_size: usize::try_from(uint(parameters, "VocabularySize")?)?,
                last_n: integer(parameters, "PenaltyLastN")?,
                repeat: float_bits(parameters, "RepeatBits")?,
                frequency: float_bits(parameters, "FrequencyBits")?,
                presence: float_bits(parameters, "PresenceBits")?,
            }
        }
        "dry-testing" => runtime_dry_config(parameters)?,
        "top-n-sigma" => {
            exact_fields(parameters, &["NBits"])?;
            Config::TopNSigma(float_bits(parameters, "NBits")?)
        }
        _ => return Err(invalid("unknown production sampler type")),
    })
}

fn runtime_dry_config(parameters: &Value) -> Result<Config> {
    exact_fields(
        parameters,
        &[
            "MultiplierBits",
            "BaseBits",
            "AllowedLength",
            "PenaltyLastN",
            "SequenceBreakers",
        ],
    )?;
    let breakers = array(parameters, "SequenceBreakers")?
        .iter()
        .map(|value| {
            value
                .as_array()
                .ok_or_else(|| invalid("DRY breaker must be an array"))?
                .iter()
                .map(|token| {
                    Ok(i32::try_from(token.as_i64().ok_or_else(|| {
                        invalid("DRY breaker token must be int32")
                    })?)?)
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Config::Dry {
        multiplier: float_bits(parameters, "MultiplierBits")?,
        base: float_bits(parameters, "BaseBits")?,
        allowed_length: usize::try_from(uint(parameters, "AllowedLength")?)?,
        last_n: integer(parameters, "PenaltyLastN")?,
        sequence_breakers: breakers,
    })
}

fn runtime_candidates(action: &Value) -> Result<RuntimeCandidates> {
    let data = array(action, "Data")?
        .iter()
        .map(|entry| {
            exact_fields(entry, &["Id", "LogitBits", "ProbabilityBits"])?;
            Ok(RuntimeCandidate {
                id: integer(entry, "Id")?,
                logit: float_bits(entry, "LogitBits")?,
                probability: float_bits(entry, "ProbabilityBits")?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    RuntimeCandidates::new(
        data,
        field(action, "Selected")?
            .as_i64()
            .ok_or_else(|| invalid("Selected must be int64"))?,
        boolean(action, "Sorted")?,
    )
}

fn runtime_handle(action: &Value, name: &str) -> Result<u32> {
    let handle = u32::try_from(uint(action, name)?)?;
    if handle > 31 {
        return Err(invalid("sampling handle exceeds bound"));
    }
    Ok(handle)
}

fn runtime_snapshot(
    action: &Value,
    candidates: &std::collections::BTreeMap<u32, RuntimeCandidates>,
) -> Result<Value> {
    exact_fields(action, &["Kind", "Candidates"])?;
    let value = candidates
        .get(&runtime_handle(action, "Candidates")?)
        .ok_or_else(|| invalid("candidate handle is missing"))?;
    let data = value.data().iter().map(|entry| json!({"Id":entry.id,"LogitBits":entry.logit.to_bits(),"ProbabilityBits":entry.probability.to_bits()})).collect::<Vec<_>>();
    Ok(json!({"Data":data,"Size":data.len(),"Selected":value.selected(),"Sorted":value.sorted()}))
}

fn runtime_check_retained(
    candidates: &std::collections::BTreeMap<u32, RuntimeCandidates>,
) -> Result<()> {
    if candidates
        .values()
        .map(RuntimeCandidates::backing_len)
        .sum::<usize>()
        > 262_144
    {
        return Err(invalid("retained candidates exceed bound"));
    }
    Ok(())
}

fn runtime_trace(operation: &Value) -> Result<Value> {
    exact_fields(operation, &["Kind", "Actions"])?;
    let actions = array(operation, "Actions")?;
    if actions.is_empty() || actions.len() > 4096 {
        return Err(invalid("sampling action count exceeds bound"));
    }
    let mut samplers = std::collections::BTreeMap::<u32, State>::new();
    let mut candidates = std::collections::BTreeMap::<u32, RuntimeCandidates>::new();
    let mut trace = Vec::with_capacity(actions.len());
    let clock = std::time::Instant::now();
    for action in actions {
        let mut event = json!({});
        match text(action, "Kind")? {
            "clock" => {
                exact_fields(action, &["Kind"])?;
                event = json!({"Microseconds":i64::try_from(clock.elapsed().as_micros())?});
            }
            "init-sampler" => {
                exact_fields(action, &["Kind", "Sampler", "Type", "Parameters"])?;
                let handle = runtime_handle(action, "Sampler")?;
                if samplers.contains_key(&handle) {
                    return Err(invalid("sampler handle already exists"));
                }
                samplers.insert(handle, State::new(runtime_config(action)?)?);
            }
            "init-candidates" => {
                exact_fields(
                    action,
                    &["Kind", "Candidates", "Data", "Selected", "Sorted"],
                )?;
                let handle = runtime_handle(action, "Candidates")?;
                if candidates.contains_key(&handle) {
                    return Err(invalid("candidate handle already exists"));
                }
                candidates.insert(handle, runtime_candidates(action)?);
            }
            "copy-candidates" => {
                exact_fields(action, &["Kind", "Source", "Target"])?;
                let source = runtime_handle(action, "Source")?;
                let target = runtime_handle(action, "Target")?;
                let copied = candidates
                    .get(&source)
                    .ok_or_else(|| invalid("candidate source is missing"))?
                    .clone();
                candidates.insert(target, copied);
            }
            "free-candidates" => {
                exact_fields(action, &["Kind", "Candidates"])?;
                if candidates
                    .remove(&runtime_handle(action, "Candidates")?)
                    .is_none()
                {
                    return Err(invalid("candidate handle is missing"));
                }
            }
            "snapshot-candidates" => {
                event = runtime_snapshot(action, &candidates)?;
            }
            "apply" => {
                exact_fields(action, &["Kind", "Sampler", "Candidates"])?;
                samplers
                    .get_mut(&runtime_handle(action, "Sampler")?)
                    .ok_or_else(|| invalid("sampler handle is missing"))?
                    .apply(
                        candidates
                            .get_mut(&runtime_handle(action, "Candidates")?)
                            .ok_or_else(|| invalid("candidate handle is missing"))?,
                    )?;
            }
            "accept" => {
                exact_fields(action, &["Kind", "Sampler", "Token"])?;
                samplers
                    .get_mut(&runtime_handle(action, "Sampler")?)
                    .ok_or_else(|| invalid("sampler handle is missing"))?
                    .accept(integer(action, "Token")?);
            }
            "reset" => {
                exact_fields(action, &["Kind", "Sampler"])?;
                samplers
                    .get_mut(&runtime_handle(action, "Sampler")?)
                    .ok_or_else(|| invalid("sampler handle is missing"))?
                    .reset();
            }
            "free-sampler" => {
                exact_fields(action, &["Kind", "Sampler"])?;
                if samplers
                    .remove(&runtime_handle(action, "Sampler")?)
                    .is_none()
                {
                    return Err(invalid("sampler handle is missing"));
                }
            }
            _ => return Err(invalid("unknown production sampler action")),
        }
        runtime_check_retained(&candidates)?;
        trace.push(event);
    }
    Ok(json!({"Trace":trace}))
}

pub(super) fn execute_runtime(request: &Value) -> Result<Option<Value>> {
    exact_fields(request, &["SchemaVersion", "Operations"])?;
    if uint(request, "SchemaVersion")? != 1 {
        return Err(invalid("sampling protocol version differs"));
    }
    let operations = array(request, "Operations")?;
    if operations.is_empty() || operations.len() > 128 {
        return Err(invalid("sampling operation count exceeds bound"));
    }
    if operations
        .iter()
        .any(|operation| operation["Kind"] != "sampler-trace")
    {
        return Ok(None);
    }
    let results = operations
        .iter()
        .map(runtime_trace)
        .collect::<Result<Vec<_>>>()?;
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

fn text<'a>(value: &'a Value, name: &str) -> Result<&'a str> {
    field(value, name)?
        .as_str()
        .ok_or_else(|| invalid(format!("sampling {name} must be text")))
}

fn boolean(value: &Value, name: &str) -> Result<bool> {
    field(value, name)?
        .as_bool()
        .ok_or_else(|| invalid(format!("sampling {name} must be boolean")))
}

fn integer(value: &Value, name: &str) -> Result<i32> {
    Ok(i32::try_from(field(value, name)?.as_i64().ok_or_else(
        || invalid(format!("sampling {name} must be int32")),
    )?)?)
}

fn float_bits(value: &Value, name: &str) -> Result<f32> {
    Ok(f32::from_bits(u32::try_from(uint(value, name)?)?))
}

fn float_array(value: &Value, name: &str) -> Result<Vec<f32>> {
    array(value, name)?
        .iter()
        .map(|bits| {
            Ok(f32::from_bits(u32::try_from(bits.as_u64().ok_or_else(
                || invalid("sampling float bits must be unsigned"),
            )?)?))
        })
        .collect()
}

fn fixture() -> Result<Value> {
    let value: Value = serde_json::from_str(CASES)?;
    if text(&value, "Source")? != ID || text(&value, "SourceSha256")? != SOURCE_SHA256 {
        return Err(invalid("sampling fixture source identity mismatch"));
    }
    let cases = array(&value, "LiteralCases")?;
    let queues = array(&value, "Queues")?;
    let literal_conditions = cases.iter().try_fold(0, |sum, case| -> Result<usize> {
        Ok(sum
            + 1
            + array(case, "ExpectedBits")?.len()
            + usize::from(boolean(case, "CheckInputSize")?))
    })?;
    let queue_steps = queues.iter().try_fold(0, |sum, queue| -> Result<usize> {
        Ok(sum + text(queue, "Sequence")?.len())
    })?;
    if cases.len() != 41 || queues.len() != 24 || literal_conditions != 190 || queue_steps != 43 {
        return Err(invalid("sampling source assertion extent mismatch"));
    }
    Ok(value)
}

fn candidate(id: i32, logit: f32, probability: f32) -> Value {
    json!({"Id":id,"LogitBits":logit.to_bits(),"ProbabilityBits":probability.to_bits()})
}

fn init_candidates(handle: u32, data: Vec<Value>) -> Value {
    json!({"Kind":"init-candidates","Candidates":handle,"Data":Value::Array(data),"Selected":-1,"Sorted":false})
}

fn init_sampler(handle: u32, kind: &str, parameters: &Value) -> Value {
    json!({"Kind":"init-sampler","Sampler":handle,"Type":kind,"Parameters":parameters})
}

fn filter_and_free(actions: &mut Vec<Value>, kind: &str, parameters: &Value) {
    actions.push(init_sampler(0, kind, parameters));
    actions.push(json!({"Kind":"apply","Sampler":0,"Candidates":0}));
    actions.push(json!({"Kind":"free-sampler","Sampler":0}));
}

fn snapshot(actions: &mut Vec<Value>, handle: u32) -> usize {
    let index = actions.len();
    actions.push(json!({"Kind":"snapshot-candidates","Candidates":handle}));
    index
}

fn execute(engine: &Engine, actions: &[Value]) -> Result<Option<Vec<Value>>> {
    let Some(response) = engine.execute_common(
        "sampling",
        &json!({"SchemaVersion":1,"Operations":[{"Kind":"sampler-trace","Actions":actions}]}),
    )?
    else {
        return Ok(None);
    };
    exact_fields(&response, &["SchemaVersion", "Results"])?;
    if uint(&response, "SchemaVersion")? != 1 || array(&response, "Results")?.len() != 1 {
        return Err(invalid("sampling common response framing mismatch"));
    }
    let value = &array(&response, "Results")?[0];
    exact_fields(value, &["Trace"])?;
    let trace = array(value, "Trace")?;
    if trace.len() != actions.len() {
        return Err(invalid("sampling trace action count mismatch"));
    }
    for (action, value) in actions.iter().zip(trace) {
        match text(action, "Kind")? {
            "snapshot-candidates" => exact_fields(value, &["Data", "Size", "Selected", "Sorted"]),
            "clock" => exact_fields(value, &["Microseconds"]),
            _ => exact_fields(value, &[]),
        }?;
    }
    Ok(Some(trace.to_vec()))
}

struct Candidate {
    id: i32,
    probability: f32,
}

struct Snapshot {
    data: Vec<Candidate>,
    selected: i64,
    sorted: bool,
}

fn read_snapshot(value: &Value, capacity: usize) -> Result<Snapshot> {
    exact_fields(value, &["Data", "Size", "Selected", "Sorted"])?;
    let data = array(value, "Data")?;
    if uint(value, "Size")? != u64::try_from(data.len())?
        || data.len() > capacity
        || capacity > MAX_CANDIDATES
    {
        return Err(invalid("sampling candidate snapshot extent mismatch"));
    }
    let selected = field(value, "Selected")?
        .as_i64()
        .ok_or_else(|| invalid("sampling Selected must be int64"))?;
    let sorted = boolean(value, "Sorted")?;
    let data = data
        .iter()
        .map(|entry| {
            exact_fields(entry, &["Id", "LogitBits", "ProbabilityBits"])?;
            let _ = float_bits(entry, "LogitBits")?;
            Ok(Candidate {
                id: integer(entry, "Id")?,
                probability: float_bits(entry, "ProbabilityBits")?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Snapshot {
        data,
        selected,
        sorted,
    })
}

fn literal_actions(case: &Value) -> Result<(Vec<Value>, usize)> {
    let probabilities = float_array(case, "ProbabilityBits")?;
    let data = probabilities
        .iter()
        .enumerate()
        .map(|(id, p)| Ok(candidate(i32::try_from(id)?, p.ln(), *p)))
        .collect::<Result<Vec<_>>>()?;
    let mut actions = vec![
        init_candidates(0, data),
        init_sampler(0, text(case, "Type")?, field(case, "Parameters")?),
    ];
    for token in array(case, "AcceptedTokens")? {
        actions.push(json!({"Kind":"accept","Sampler":0,"Token":token}));
    }
    actions.push(json!({"Kind":"apply","Sampler":0,"Candidates":0}));
    actions.push(json!({"Kind":"free-sampler","Sampler":0}));
    if boolean(case, "Normalize")? {
        filter_and_free(&mut actions, "dist", &json!({"Seed":0}));
    }
    let observation = snapshot(&mut actions, 0);
    actions.push(json!({"Kind":"free-candidates","Candidates":0}));
    Ok((actions, observation))
}

fn check_literal(
    case: &Value,
    value: &Value,
    index: usize,
    assertions: &mut Assertions,
) -> Result<()> {
    let input = float_array(case, "ProbabilityBits")?;
    let expected = float_array(case, "ExpectedBits")?;
    let name = format!("literal[{index}] {}", text(case, "Type")?);
    if boolean(case, "CheckInputSize")? {
        assertions.check(format!("{name} input size"), input.len() == expected.len());
    }
    let actual = read_snapshot(value, input.len())?;
    assertions.check(
        format!("{name} active size"),
        actual.data.len() == expected.len(),
    );
    // The original loop is over the actual active extent after the size assertion.
    // Keep every expected position observable even after a failed size assertion.
    for (position, expected) in expected.iter().enumerate() {
        assertions.check(
            format!("{name} probability[{position}]"),
            actual
                .data
                .get(position)
                .is_some_and(|entry| f64::from((entry.probability - expected).abs()) < 1e-5),
        );
    }
    Ok(())
}

fn singleton_actions() -> (Vec<Value>, Vec<usize>) {
    let mut actions = vec![
        init_sampler(0, "dist", &json!({"Seed":4242})),
        init_sampler(1, "dist", &json!({"Seed":4242})),
    ];
    let mut observations = Vec::new();
    for (sampler, count) in [(0, 1), (1, 2)]
        .into_iter()
        .chain((0..4).flat_map(|_| [(0, 256), (1, 256)]))
    {
        let data = (0..count).map(|id| candidate(id, 0.0, 0.0)).collect();
        actions.push(init_candidates(0, data));
        actions.push(json!({"Kind":"apply","Sampler":sampler,"Candidates":0}));
        observations.push(snapshot(&mut actions, 0));
        actions.push(json!({"Kind":"free-candidates","Candidates":0}));
    }
    actions.push(json!({"Kind":"free-sampler","Sampler":0}));
    actions.push(json!({"Kind":"free-sampler","Sampler":1}));
    (actions, observations)
}

fn check_singletons(
    trace: &[Value],
    observations: &[usize],
    assertions: &mut Assertions,
) -> Result<()> {
    let mut ids = Vec::new();
    for (sample, index) in observations.iter().enumerate() {
        let snapshot = read_snapshot(&trace[*index], [1, 2].get(sample).copied().unwrap_or(256))?;
        assertions.check(
            format!("singleton sample[{sample}] nonnegative selection"),
            snapshot.selected >= 0,
        );
        let position = usize::try_from(snapshot.selected).ok();
        assertions.check(
            format!("singleton sample[{sample}] bounded selection"),
            position.is_some_and(|position| position < snapshot.data.len()),
        );
        ids.push(
            position
                .and_then(|position| snapshot.data.get(position))
                .map(|entry| entry.id),
        );
    }
    for iteration in 0..4 {
        let first = ids[2 + iteration * 2];
        let second = ids[3 + iteration * 2];
        assertions.check(
            format!("singleton RNG alignment[{iteration}]"),
            first.is_some() && first == second,
        );
    }
    Ok(())
}

fn queue_parameters(queue: &Value, kind: u8) -> Result<(&'static str, Value)> {
    Ok(match kind {
        b'k' => ("top-k", json!({"K":integer(queue,"TopK")?})),
        b'p' => (
            "top-p",
            json!({"ProbabilityBits":uint(queue,"TopPBits")?,"MinKeep":1}),
        ),
        b'm' => (
            "min-p",
            json!({"ProbabilityBits":uint(queue,"MinPBits")?,"MinKeep":1}),
        ),
        _ => return Err(invalid("original sampling queue has an unsupported filter")),
    })
}

fn queue_actions(queue: &Value) -> Result<(Vec<Value>, Vec<usize>)> {
    let count = u16::try_from(uint(queue, "VocabularySize")?)?;
    if count != 10_000 {
        return Err(invalid("original sampling queue vocabulary mismatch"));
    }
    let data = (0..count)
        .map(|id| candidate(i32::from(id), f32::from(id).ln(), 0.0))
        .collect();
    let mut actions = vec![init_candidates(0, data)];
    let mut observations = Vec::new();
    for kind in text(queue, "Sequence")?.bytes() {
        let (kind, parameters) = queue_parameters(queue, kind)?;
        filter_and_free(&mut actions, kind, &parameters);
        filter_and_free(&mut actions, "dist", &json!({"Seed":0}));
        observations.push(snapshot(&mut actions, 0));
    }
    actions.push(json!({"Kind":"free-candidates","Candidates":0}));
    Ok((actions, observations))
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Mirror the original ceilf/floorf-to-int conversion after checking its finite int32 range."
)]
fn float_integer(value: f32) -> Result<i32> {
    if !value.is_finite()
        || value < -2_147_483_648.0
        || value >= 2_147_483_648.0
        || value.fract() != 0.0
    {
        return Err(invalid("sampling reference integer conversion overflow"));
    }
    Ok(value as i32)
}

fn check_queue(
    queue: &Value,
    trace: &[Value],
    observations: &[usize],
    index: usize,
    assertions: &mut Assertions,
) -> Result<()> {
    let n_vocab = i32::try_from(uint(queue, "VocabularySize")?)?;
    let vocab_float = f32::from(u16::try_from(n_vocab)?);
    let maximum = n_vocab - 1;
    let mut minimum = 0;
    let top_k = integer(queue, "TopK")?;
    let top_p = float_bits(queue, "TopPBits")?;
    let min_p = float_bits(queue, "MinPBits")?;
    for (step, (kind, observation)) in text(queue, "Sequence")?
        .bytes()
        .zip(observations)
        .enumerate()
    {
        let actual = read_snapshot(&trace[*observation], usize::try_from(n_vocab)?)?;
        let size = i32::try_from(actual.data.len())?;
        let expected_size = match kind {
            b'k' => {
                minimum = minimum.max(n_vocab - top_k);
                size.min(top_k)
            }
            b'p' => {
                let divisor = n_vocab * (n_vocab - 1) / 2 - minimum * (minimum - 1) / 2;
                // All original divisors fit exactly in uint32; the conversion to
                // float intentionally has the source's binary32 rounding.
                let divisor_float = round_to_float(f64::from(divisor));
                let target = float_integer((top_p * divisor_float).ceil())?;
                minimum = n_vocab;
                let mut expected_size = 0;
                let mut cumulative = 0;
                loop {
                    minimum -= 1;
                    expected_size += 1;
                    cumulative += minimum;
                    if cumulative >= target {
                        break;
                    }
                }
                if minimum == 1 {
                    minimum -= 1;
                    expected_size += 1;
                }
                expected_size
            }
            b'm' => {
                let expected_size = float_integer(((1.0_f32 - min_p) * vocab_float).ceil())?
                    .max(1)
                    .min(size);
                minimum = float_integer((min_p * vocab_float).floor())?
                    .max(1)
                    .max(n_vocab - size)
                    .min(n_vocab - 1);
                expected_size
            }
            _ => return Err(invalid("sampling reference queue filter mismatch")),
        };
        let last = usize::try_from(expected_size - 1)
            .ok()
            .and_then(|position| actual.data.get(position));
        let check_order = kind == b'k' || actual.sorted;
        assertions.check(
            format!("queue[{index}] step[{step}] size"),
            size == expected_size,
        );
        assertions.check(
            format!("queue[{index}] step[{step}] maximum"),
            !check_order || actual.data.first().is_some_and(|entry| entry.id == maximum),
        );
        assertions.check(
            format!("queue[{index}] step[{step}] minimum"),
            !check_order || last.is_some_and(|entry| entry.id == minimum),
        );
    }
    Ok(())
}

/// Evaluates all 343 original correctness conditions without running performance workloads.
pub(super) fn run(engine: &Engine) -> Result<Value> {
    let fixture = fixture()?;
    let mut assertions = Assertions::default();
    let (actions, observations) = singleton_actions();
    let Some(trace) = execute(engine, &actions)? else {
        return Ok(result(
            ID,
            false,
            "unsupported production API: stateful sampler and candidate-buffer operations"
                .to_owned(),
        ));
    };
    check_singletons(&trace, &observations, &mut assertions)?;
    for (index, case) in array(&fixture, "LiteralCases")?.iter().enumerate() {
        let (actions, observation) = literal_actions(case)?;
        let trace =
            execute(engine, &actions)?.ok_or_else(|| invalid("sampling backend disappeared"))?;
        check_literal(case, &trace[observation], index, &mut assertions)?;
    }
    for (index, queue) in array(&fixture, "Queues")?.iter().enumerate() {
        let (actions, observations) = queue_actions(queue)?;
        let trace =
            execute(engine, &actions)?.ok_or_else(|| invalid("sampling backend disappeared"))?;
        check_queue(queue, &trace, &observations, index, &mut assertions)?;
    }
    if assertions.count != EXPECTED_CONDITIONS {
        return Err(invalid("sampling executed assertion extent mismatch"));
    }
    Ok(result(
        ID,
        assertions.failures.is_empty(),
        format!(
            "41 literal cases/190 conditions; singleton RNG/24; 24 queues/43 steps/129; assertions={}; failures={:?}; five performance workloads dispatched separately in CI",
            assertions.count, assertions.failures
        ),
    ))
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "The original Windows workload rounds its double rand/RAND_MAX expression once to binary32."
)]
const fn round_to_float(value: f64) -> f32 {
    value as f32
}

struct WindowsRandom(u32);
impl WindowsRandom {
    fn next(&mut self) -> i32 {
        self.0 = self.0.wrapping_mul(214_013).wrapping_add(2_531_011);
        let [_, _, low, high] = self.0.to_le_bytes();
        i32::from(u16::from_le_bytes([low, high]) & 0x7fff)
    }
}

fn benchmark_data() -> Result<Vec<Value>> {
    let mut random = WindowsRandom(1);
    (0..MAX_CANDIDATES)
        .map(|id| {
            let logit = round_to_float(2.0 * (f64::from(random.next()) / 32_767.0 - 0.5));
            Ok(candidate(i32::try_from(id)?, logit, 0.0))
        })
        .collect()
}

fn benchmark_actions(
    kind: &str,
    parameters: &Value,
    data: Vec<Value>,
) -> (Vec<Value>, usize, usize) {
    let mut actions = vec![
        init_candidates(1, data),
        json!({"Kind":"copy-candidates","Source":1,"Target":0}),
        init_sampler(0, kind, parameters),
        json!({"Kind":"apply","Sampler":0,"Candidates":0}),
        json!({"Kind":"reset","Sampler":0}),
        json!({"Kind":"clock"}),
    ];
    let start = actions.len() - 1;
    for _ in 0..32 {
        actions.push(json!({"Kind":"copy-candidates","Source":1,"Target":0}));
        actions.push(json!({"Kind":"apply","Sampler":0,"Candidates":0}));
        actions.push(json!({"Kind":"reset","Sampler":0}));
    }
    let end = actions.len();
    actions.push(json!({"Kind":"clock"}));
    actions.push(json!({"Kind":"free-sampler","Sampler":0}));
    actions.push(json!({"Kind":"free-candidates","Candidates":0}));
    actions.push(json!({"Kind":"free-candidates","Candidates":1}));
    (actions, start, end)
}

/// Runs the five original 32-iteration workloads only through the explicit CI benchmark path.
pub(super) fn benchmark(engine: &Engine) -> Result<Value> {
    if std::env::var_os("GITHUB_ACTIONS").as_deref() != Some(OsStr::new("true")) {
        return Err(invalid("sampling benchmarks run only on GitHub Actions"));
    }
    let data = benchmark_data()?;
    let samplers = [
        ("top-k", json!({"K":40})),
        (
            "top-p",
            json!({"ProbabilityBits":0.8_f32.to_bits(),"MinKeep":1}),
        ),
        (
            "min-p",
            json!({"ProbabilityBits":0.2_f32.to_bits(),"MinKeep":1}),
        ),
        (
            "typical",
            json!({"ProbabilityBits":0.5_f32.to_bits(),"MinKeep":1}),
        ),
        (
            "xtc",
            json!({"ProbabilityBits":1.0_f32.to_bits(),"ThresholdBits":0.1_f32.to_bits(),"MinKeep":1,"Seed":1}),
        ),
    ];
    let mut measurements = Vec::new();
    for (kind, parameters) in samplers {
        let (actions, start, end) = benchmark_actions(kind, &parameters, data.clone());
        let Some(trace) = execute(engine, &actions)? else {
            return Ok(
                json!({"Id":"tests/test-sampling.cpp#performance","Status":"unsupported","Iterations":32,"Workloads":5,
                "Detail":"unsupported production API: stateful sampler and candidate-buffer operations"}),
            );
        };
        let start = field(&trace[start], "Microseconds")?
            .as_i64()
            .ok_or_else(|| invalid("sampling clock must be int64"))?;
        let end = field(&trace[end], "Microseconds")?
            .as_i64()
            .ok_or_else(|| invalid("sampling clock must be int64"))?;
        let nanoseconds = u64::try_from(
            end.checked_sub(start)
                .ok_or_else(|| invalid("sampling clock difference overflow"))?,
        )?
        .checked_mul(1000)
        .ok_or_else(|| invalid("sampling timing conversion overflow"))?;
        measurements.push(json!({"Sampler":kind,"Iterations":32,"WarmupIterations":1,"VocabularySize":MAX_CANDIDATES,"WallNanoseconds":nanoseconds}));
    }
    Ok(
        json!({"Id":"tests/test-sampling.cpp#performance","Status":"measured","Measurements":measurements,
        "MeasurementScope":"32 explicit candidate copies, production sampler apply/reset operations, action dispatch and metadata handling; excludes request parsing, sampler construction and warmup",
        "CorrectnessThreshold":null}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_rounding_and_snapshot_bounds_preserve_source_contract() -> Result<()> {
        let fixture = fixture()?;
        assert_eq!(array(&fixture, "LiteralCases")?.len(), 41);
        assert_eq!(array(&fixture, "Queues")?.len(), 24);
        assert_eq!(
            float_integer((0.8_f32 * round_to_float(49_995_000.0)).ceil())?,
            39_996_000
        );
        assert!(float_integer(f32::INFINITY).is_err());
        assert!(float_integer(0.5).is_err());
        let mut random = WindowsRandom(1);
        assert_eq!(
            [random.next(), random.next(), random.next()],
            [41, 18_467, 6_334]
        );
        let bad = json!({"Data":[],"Size":1,"Selected":-1,"Sorted":false});
        assert!(read_snapshot(&bad, 1).is_err());
        let bad = json!({"Data":[],"Size":0,"Selected":null,"Sorted":false});
        assert!(read_snapshot(&bad, 1).is_err());
        Ok(())
    }
}
