//! Focused state and vocabulary-boundary contracts for production budgets.
#![forbid(unsafe_code)]

use bonsai_prism_runtime::sampling::reasoning_budget::{ReasoningBudget, State, utf8_is_complete};

type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

#[test]
fn incomplete_piece_waits_and_clones_keep_independent_forced_positions() -> TestResult {
    let mut budget = ReasoningBudget::new(
        vec![vec![10]],
        vec![vec![20, 21]],
        vec![20, 21],
        1,
        State::Idle,
    )?;
    budget.accept(10);
    budget.accept_with_utf8(50, false);
    assert_eq!(budget.state(), State::WaitingUtf8);
    assert!(!budget.force());
    budget.accept_with_utf8(51, false);
    assert_eq!(budget.state(), State::WaitingUtf8);
    budget.accept_with_utf8(52, true);
    assert_eq!(budget.forced_token(), Some(20));
    let mut fork = budget.clone();
    budget.accept(20);
    assert_eq!(budget.forced_token(), Some(21));
    assert_eq!(fork.forced_token(), Some(20));
    budget.accept(21);
    assert_eq!(budget.end_match(), Some([20, 21].as_slice()));
    fork.reset();
    assert_eq!(fork.state(), State::Idle);
    assert_eq!(fork.end_match(), None);
    budget.accept(10);
    assert_eq!(budget.state(), State::Counting);
    assert_eq!(budget.end_match(), None);
    Ok(())
}

#[test]
fn normalized_start_delimiters_preserve_forced_candidate_bits() -> TestResult {
    let mut budget = ReasoningBudget::new(
        vec![vec![], vec![1], vec![1]],
        vec![vec![3], vec![2, 3]],
        vec![2, 3],
        8,
        State::Idle,
    )?;
    budget.accept(1);
    assert!(budget.force());
    let preserved = f32::from_bits(0x7fc0_1234);
    let mut logits = [1.0, preserved, -9.0];
    budget.apply(&[1, 2, 3], &mut logits)?;
    assert_eq!(logits[0].to_bits(), f32::NEG_INFINITY.to_bits());
    assert_eq!(logits[1].to_bits(), preserved.to_bits());
    assert_eq!(logits[2].to_bits(), f32::NEG_INFINITY.to_bits());
    let before = logits.map(f32::to_bits);
    assert!(budget.apply(&[2], &mut logits).is_err());
    assert_eq!(logits.map(f32::to_bits), before);
    Ok(())
}

#[test]
fn utf8_helper_checks_native_trailing_boundaries_without_revalidating_prefixes() {
    assert!(utf8_is_complete(&[0xff, b'a']));
    assert!(!utf8_is_complete(&[0x80; 4]));
}
