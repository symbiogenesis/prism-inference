//! Focused owned grammar execution, partial UTF-8 and candidate-order checks.
#![forbid(unsafe_code)]

use bonsai_prism_runtime::grammar::{
    Grammar,
    runtime::{Candidate, GrammarState, PartialUtf8},
};

type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

fn grammar(source: &str) -> bonsai_prism_runtime::Result<GrammarState> {
    GrammarState::from_grammar(Grammar::parse(source)?, "root")
}

fn candidate(index: u32, points: Vec<u32>) -> Candidate {
    Candidate {
        index,
        code_points: points,
        partial_utf8: PartialUtf8::default(),
        id: 0,
    }
}

#[test]
fn candidate_rejection_keeps_native_immediate_then_recursive_order() -> TestResult {
    let grammar = grammar("root ::= \"ab\"")?;
    let candidates = [
        candidate(0, vec![97, 100, 0]),
        candidate(1, vec![120, 0]),
        candidate(2, vec![97, 98, 0]),
    ];
    let rejected = grammar.reject_candidates(0, &candidates)?;
    assert_eq!(
        rejected
            .iter()
            .map(|candidate| candidate.index)
            .collect::<Vec<_>>(),
        [1, 0]
    );
    assert_eq!(rejected[1].code_points, [97, 100, 0]);
    Ok(())
}

#[test]
fn split_utf8_tokens_and_clones_own_independent_rule_storage() -> TestResult {
    let mut state = grammar("root ::= \"é\" \"a\"")?;
    let checkpoint = state.save_stacks();
    state.accept_token(0, &[0xc3])?;
    let mut clone = state.clone();
    assert_ne!(state.rules()[0].as_ptr(), clone.rules()[0].as_ptr());
    assert!(clone.restore_stacks(&checkpoint).is_err());
    state.accept_token(0, &[0xa9])?;
    state.accept_token(0, b"a")?;
    assert!(state.stacks().iter().any(Vec::is_empty));
    clone.accept_token(0, &[0xa9])?;
    assert!(clone.accept_token(0, b"b").is_err());
    assert_eq!(clone.stacks().len(), 0);
    assert!(state.stacks().iter().any(Vec::is_empty));
    state.restore_stacks(&checkpoint)?;
    state.accept_token(0, "éa".as_bytes())?;
    assert!(state.stacks().iter().any(Vec::is_empty));
    Ok(())
}

#[test]
fn explicit_token_rules_and_partial_character_ranges_keep_distinct_semantics() -> TestResult {
    let mut tokens = grammar("root ::= <[1]> \"a\"")?;
    tokens.accept_token(1, b"ignored-piece")?;
    tokens.accept_token(0, b"a")?;
    assert!(tokens.stacks().iter().any(Vec::is_empty));
    let range = grammar("root ::= [\\u0080-\\u07ff]")?;
    let mut valid = candidate(0, vec![0]);
    valid.partial_utf8 = PartialUtf8 {
        value: 2,
        remaining: 1,
    };
    let mut overlong = valid.clone();
    overlong.index = 1;
    overlong.partial_utf8.value = 1;
    assert_eq!(
        range.reject_candidates(0, &[valid, overlong.clone()])?,
        [overlong]
    );
    Ok(())
}

#[test]
fn long_candidates_use_heap_work_and_invalid_rules_are_rejected() -> TestResult {
    let state = grammar("root ::= \"a\"*")?;
    let stack = state
        .stacks()
        .iter()
        .position(|stack| !stack.is_empty())
        .ok_or("missing recursive terminal stack")?;
    let mut points = vec![97; 1000];
    points.push(0);
    assert_eq!(state.reject_candidates(stack, &[candidate(0, points)])?, []);
    assert!(grammar("root ::= root \"a\"").is_err());
    assert!(
        GrammarState::from_rules(
            vec![vec![
                bonsai_prism_runtime::grammar::Element { kind: 2, value: 1 },
                bonsai_prism_runtime::grammar::Element { kind: 0, value: 0 }
            ]],
            0
        )
        .is_err()
    );
    assert!(
        state
            .reject_candidates(stack, &[candidate(0, vec![97])])
            .is_err()
    );
    Ok(())
}
