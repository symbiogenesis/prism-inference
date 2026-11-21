//! Independent finite-state observations for the null-vocabulary source contract.
//!
//! Delimiters use active prefix positions, rather than the production runtime's
//! suffix buffer. Counting uses accepted-event ordinals rather than a decrementing
//! budget. No production sampler, native library or expected-answer fixture is used.

use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer};
use serde_json::{Value, json};

use super::{Result, invalid};

pub(super) const ID: &str = "tests/test-reasoning-budget.cpp";

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
struct Request {
    schema_version: u32,
    operations: Vec<Operation>,
}

#[derive(Deserialize)]
#[serde(tag = "Kind", deny_unknown_fields)]
enum Operation {
    #[serde(rename = "reasoning-budget-trace")]
    Trace {
        #[serde(rename = "Actions")]
        actions: Vec<Action>,
    },
    #[serde(rename = "utf8-is-complete")]
    Utf8 {
        #[serde(rename = "Bytes")]
        bytes: Vec<u8>,
    },
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
struct Candidate {
    id: i32,
    logit_bits: u32,
    probability_bits: u32,
}

#[derive(Deserialize)]
#[serde(tag = "Kind", rename_all_fields = "PascalCase", deny_unknown_fields)]
enum Action {
    #[serde(rename = "init")]
    Init {
        sampler: u32,
        vocabulary: Value,
        start_sequences: Vec<Vec<i32>>,
        end_sequences: Vec<Vec<i32>>,
        forced_tokens: Vec<i32>,
        budget: i32,
        initial_state: u32,
    },
    #[serde(rename = "apply")]
    Apply {
        #[serde(deserialize_with = "handle")]
        sampler: Option<u32>,
        candidates: Vec<Candidate>,
        selected: i64,
        sorted: bool,
    },
    #[serde(rename = "accept")]
    Accept {
        #[serde(deserialize_with = "handle")]
        sampler: Option<u32>,
        token: i32,
    },
    #[serde(rename = "clone")]
    Clone {
        #[serde(deserialize_with = "handle")]
        sampler: Option<u32>,
        target: u32,
    },
    #[serde(rename = "state")]
    State {
        #[serde(deserialize_with = "handle")]
        sampler: Option<u32>,
    },
    #[serde(rename = "force")]
    Force {
        #[serde(deserialize_with = "handle")]
        sampler: Option<u32>,
    },
    #[serde(rename = "end-match")]
    EndMatch {
        #[serde(deserialize_with = "handle")]
        sampler: Option<u32>,
    },
    #[serde(rename = "free")]
    Free {
        #[serde(deserialize_with = "handle")]
        sampler: Option<u32>,
    },
}

fn handle<'de, D: Deserializer<'de>>(input: D) -> std::result::Result<Option<u32>, D::Error> {
    Option::deserialize(input)
}

#[derive(Clone)]
struct Delimiters {
    words: Vec<Vec<i32>>,
    active: Vec<(usize, usize)>,
}

impl Delimiters {
    fn new(words: Vec<Vec<i32>>) -> Result<Self> {
        if words.len() > 64 || words.iter().any(|word| word.len() > 1024) {
            return Err(invalid("reference delimiter extent exceeds bound"));
        }
        let mut unique = Vec::new();
        for word in words {
            if !word.is_empty() && !unique.contains(&word) {
                unique.push(word);
            }
        }
        Ok(Self {
            words: unique,
            active: Vec::new(),
        })
    }

    fn accept(&mut self, token: i32) -> Option<Vec<i32>> {
        let mut next = Vec::new();
        let mut completed_indices = Vec::new();
        for (index, position) in self
            .active
            .iter()
            .copied()
            .chain((0..self.words.len()).map(|index| (index, 0)))
        {
            let word = &self.words[index];
            if word[position] != token {
                continue;
            }
            if position + 1 == word.len() {
                completed_indices.push(index);
            } else {
                next.push((index, position + 1));
            }
        }
        let matched = completed_indices
            .into_iter()
            .max_by_key(|&index| self.words[index].len());
        self.active = if matched.is_some() { Vec::new() } else { next };
        matched.map(|index| self.words[index].clone())
    }
}

#[derive(Clone)]
struct Sampler {
    start: Delimiters,
    end: Delimiters,
    forced: Vec<i32>,
    budget: i32,
    state: u32,
    accepted: usize,
    forced_accepted: usize,
    end_match: Option<Vec<i32>>,
}

impl Sampler {
    fn forcing(&mut self) {
        self.state = 2;
        self.forced_accepted = 0;
        self.end.active.clear();
    }

    fn accept(&mut self, token: i32) {
        match self.state {
            0 | 4 => {
                if self.start.accept(token).is_some() {
                    if self.state == 4 {
                        self.end.active.clear();
                        self.end_match = None;
                    }
                    self.state = 1;
                    self.accepted = 0;
                    if self.budget <= 0 {
                        self.forcing();
                    }
                }
            }
            1 | 3 => {
                if let Some(word) = self.end.accept(token) {
                    self.end_match = Some(word);
                    self.state = 4;
                } else {
                    self.accepted += 1;
                    // The admitted vocabulary is null: every accepted token is
                    // an immediately complete boundary, including initial WAIT.
                    if self.state == 3
                        || i64::try_from(self.accepted).unwrap_or(i64::MAX)
                            >= i64::from(self.budget)
                    {
                        self.forcing();
                    }
                }
            }
            2 => {
                let matched = self.end.accept(token);
                self.forced_accepted += 1;
                if self.forced_accepted >= self.forced.len() {
                    self.state = 4;
                    self.end_match = matched;
                }
            }
            _ => unreachable!("validated initial state"),
        }
    }
}

fn sampler_mut(
    samplers: &mut BTreeMap<u32, Option<Sampler>>,
    handle: Option<u32>,
) -> Result<Option<&mut Sampler>> {
    handle.map_or(Ok(None), |handle| {
        samplers
            .get_mut(&handle)
            .map(Option::as_mut)
            .ok_or_else(|| invalid("unknown or freed reference sampler"))
    })
}

// A boundary observation deliberately ignores malformed earlier bytes, just as
// the source contract requires. This is not whole-string UTF-8 validation.
fn utf8_boundary(bytes: &[u8]) -> bool {
    let tail = &bytes[bytes.len().saturating_sub(4)..];
    if tail.is_empty() {
        return true;
    }
    let Some(leader) = tail.iter().rposition(|byte| byte & 0xc0 != 0x80) else {
        return false;
    };
    let width = match tail[leader] {
        0..=0xbf => 1,
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        _ => 4,
    };
    tail.len() - leader >= width
}

#[expect(
    clippy::too_many_lines,
    reason = "Keep the complete bounded action protocol beside independent state ownership."
)]
fn trace(actions: Vec<Action>) -> Result<Value> {
    if actions.is_empty() || actions.len() > 4096 {
        return Err(invalid("reference trace action count outside bound"));
    }
    let mut samplers = BTreeMap::new();
    let mut events = Vec::with_capacity(actions.len());
    let mut total_candidates = 0;
    for action in actions {
        let mut event = json!({});
        match action {
            Action::Init {
                sampler,
                vocabulary,
                start_sequences,
                end_sequences,
                forced_tokens,
                budget,
                initial_state,
            } => {
                if sampler >= 32
                    || samplers.contains_key(&sampler)
                    || !vocabulary.is_null()
                    || initial_state > 4
                    || forced_tokens.len() > 1024
                {
                    return Err(invalid("reference sampler initialization outside contract"));
                }
                samplers.insert(
                    sampler,
                    Some(Sampler {
                        start: Delimiters::new(start_sequences)?,
                        end: Delimiters::new(end_sequences)?,
                        forced: forced_tokens,
                        budget,
                        state: if initial_state == 1 && budget <= 0 {
                            2
                        } else {
                            initial_state
                        },
                        accepted: 0,
                        forced_accepted: 0,
                        end_match: None,
                    }),
                );
            }
            Action::Apply {
                sampler,
                candidates,
                selected,
                sorted,
            } => {
                total_candidates += candidates.len();
                if candidates.len() > 4096
                    || total_candidates > 65536
                    || selected < -1
                    || (selected >= 0 && usize::try_from(selected)? >= candidates.len())
                {
                    return Err(invalid("reference candidate extent outside bound"));
                }
                let forced = sampler_mut(&mut samplers, sampler)?.and_then(|sampler| {
                    (sampler.state == 2)
                        .then(|| sampler.forced.get(sampler.forced_accepted).copied())
                        .flatten()
                });
                let values = candidates.into_iter().map(|candidate| {
                    let logit = if forced.is_some_and(|token| token != candidate.id) {
                        0xff80_0000
                    } else { candidate.logit_bits };
                    json!({"Id": candidate.id, "LogitBits": logit, "ProbabilityBits": candidate.probability_bits})
                }).collect::<Vec<_>>();
                event = json!({"Candidates":values,"Selected":selected,"Sorted":sorted});
            }
            Action::Accept { sampler, token } => {
                if let Some(sampler) = sampler_mut(&mut samplers, sampler)? {
                    sampler.accept(token);
                }
            }
            Action::Clone { sampler, target } => {
                let cloned = sampler_mut(&mut samplers, sampler)?.cloned();
                if target >= 32 || samplers.contains_key(&target) {
                    return Err(invalid("reference clone target outside contract"));
                }
                samplers.insert(target, cloned);
            }
            Action::State { sampler } => {
                event = json!({"State":sampler_mut(&mut samplers, sampler)?.map_or(0, |sampler| sampler.state)});
            }
            Action::Force { sampler } => {
                let forced = if let Some(sampler) = sampler_mut(&mut samplers, sampler)?
                    && sampler.state == 1
                {
                    sampler.forcing();
                    true
                } else {
                    false
                };
                event = json!({"Forced":forced});
            }
            Action::EndMatch { sampler } => {
                event = json!({"EndMatch":sampler_mut(&mut samplers, sampler)?.and_then(|sampler| sampler.end_match.as_ref())});
            }
            Action::Free { sampler } => {
                sampler_mut(&mut samplers, sampler)?;
                if let Some(sampler) = sampler {
                    samplers.remove(&sampler);
                }
            }
        }
        events.push(event);
    }
    Ok(json!({"Trace":events}))
}

pub(super) fn handles(value: &Value) -> bool {
    value["Operations"].as_array().is_some_and(|operations| {
        !operations.is_empty()
            && operations.iter().all(|operation| {
                matches!(
                    operation["Kind"].as_str(),
                    Some("reasoning-budget-trace" | "utf8-is-complete")
                )
            })
    })
}

pub(super) fn execute(value: &Value) -> Result<Value> {
    let request: Request = serde_json::from_value(value.clone())?;
    if request.schema_version != 1
        || request.operations.is_empty()
        || request.operations.len() > 128
    {
        return Err(invalid("reference reasoning request framing outside bound"));
    }
    let results = request
        .operations
        .into_iter()
        .map(|operation| match operation {
            Operation::Trace { actions } => trace(actions),
            Operation::Utf8 { bytes } => {
                if bytes.len() > 65536 {
                    return Err(invalid("reference UTF-8 extent outside bound"));
                }
                Ok(json!({"Complete":utf8_boundary(&bytes)}))
            }
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(json!({"SchemaVersion":1,"Results":results}))
}

pub(super) fn identity() -> Value {
    json!({"Kind":"independent-rust-state-machine", "SupportedSourceIds":[ID],
        "UpstreamCommit":super::UPSTREAM,
        "AssertionSource":{"Path":ID,"Sha256":"1499e08757ff9e6f52ddd84134a1454346e92cd78a3dc5a9141f8ba775f57979"},
        "Contract":"null-vocabulary-active-prefix-and-event-ordinal-v1",
        "RequiredSemanticConditions":71,"NativeReferenceReplay":false,
        "ProductionRuntimeCalled":false,"CompleteSourceGate":false,"NativeRetirementQualified":false})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, Target};

    #[test]
    fn independent_reasoning_preserves_complete_source_assertions() -> Result<()> {
        let engine = Engine::new(Target::RustReference, None, None, None, None)?;
        let report = crate::reasoning_budget::run(&engine)?;
        assert_eq!(report["Id"], ID);
        assert_eq!(report["Status"], "pass", "{report}");
        assert!(
            report["Detail"]
                .as_str()
                .is_some_and(|detail| detail.contains("71 required semantic conditions;")
                    && detail.contains("failed=0;"))
        );
        let reference_arm = Engine::new(Target::Cpp, None, None, None, None)?;
        assert_eq!(crate::reasoning_budget::run(&reference_arm)?, report);
        let candidate = Engine::new(Target::Rust, None, None, None, None)?;
        let report = crate::reasoning_budget::run(&candidate)?;
        assert_eq!(report["Status"], "pass", "{report}");
        Ok(())
    }

    #[test]
    fn independent_reasoning_retains_exceptional_bits_and_clone_ownership() -> Result<()> {
        let request = json!({"SchemaVersion":1,"Operations":[{"Kind":"reasoning-budget-trace","Actions":[
            {"Kind":"init","Sampler":0,"Vocabulary":null,"StartSequences":[],"EndSequences":[[9]],"ForcedTokens":[7,9],"Budget":0,"InitialState":2},
            {"Kind":"clone","Sampler":0,"Target":1},
            {"Kind":"accept","Sampler":0,"Token":7},
            {"Kind":"apply","Sampler":1,"Candidates":[{"Id":7,"LogitBits":2_143_289_345_u32,"ProbabilityBits":2_147_483_648_u32},{"Id":9,"LogitBits":0,"ProbabilityBits":1}],"Selected":1,"Sorted":true},
            {"Kind":"free","Sampler":0},{"Kind":"accept","Sampler":1,"Token":7},
            {"Kind":"accept","Sampler":1,"Token":9},{"Kind":"end-match","Sampler":1},
            {"Kind":"free","Sampler":1}
        ]}]});
        let response = execute(&request)?;
        let trace = &response["Results"][0]["Trace"];
        assert_eq!(trace[3]["Candidates"][0]["LogitBits"], 2_143_289_345_u32);
        assert_eq!(
            trace[3]["Candidates"][0]["ProbabilityBits"],
            2_147_483_648_u32
        );
        assert_eq!(trace[3]["Candidates"][1]["LogitBits"], 0xff80_0000u32);
        assert_eq!(trace[3]["Selected"], 1);
        assert_eq!(trace[3]["Sorted"], true);
        assert_eq!(trace[7]["EndMatch"], json!([9]));
        Ok(())
    }

    #[test]
    fn independent_reasoning_rejects_coercions_stale_handles_and_missing_native_routes() {
        let action = json!({"Kind":"init","Sampler":0,"Vocabulary":null,"StartSequences":[],"EndSequences":[],"ForcedTokens":[],"Budget":1,"InitialState":0});
        for (field, value) in [
            ("Sampler", json!(32)),
            ("Budget", json!(true)),
            ("Vocabulary", json!(1)),
            ("InitialState", json!(5)),
            ("Unknown", json!(0)),
        ] {
            let mut changed = action.clone();
            changed[field] = value;
            assert!(execute(&json!({"SchemaVersion":1,"Operations":[{"Kind":"reasoning-budget-trace","Actions":[changed]}]})).is_err());
        }
        assert!(execute(&json!({"SchemaVersion":1,"Operations":[{"Kind":"reasoning-budget-trace","Actions":[action,{"Kind":"free","Sampler":0},{"Kind":"state","Sampler":0}]}]})).is_err());
        assert!(
            execute(&json!({"SchemaVersion":1,"Operations":[{"Kind":"backend-graph"}]})).is_err()
        );
        assert!(execute(&json!({"SchemaVersion":1,"Operations":[{"Kind":"reasoning-budget-trace","Actions":[{"Kind":"free"}]}]})).is_err());
        assert!(!handles(&json!({"Operations":[]})));
        assert!(execute(&json!({"SchemaVersion":1,"Operations":[]})).is_err());
    }
}
