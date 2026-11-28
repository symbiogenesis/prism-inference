//! Stateful token budgets with native Prism delimiter and forcing semantics.
//!
//! Based on `common/reasoning-budget.cpp` in Prism revision
//! `b87ca7225456b06b0975b8239c66b5aecf07e37c` (MIT, see PRISM-LICENSE.txt).
//! Matchers retain bounded token suffixes; clones own independent state.

use crate::{Result, invalid};

/// Current phase of a reasoning-block token budget.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum State {
    /// Waiting for a start delimiter.
    Idle = 0,
    /// Counting tokens and accepting a natural end delimiter.
    Counting = 1,
    /// Restricting candidates to successive forced tokens.
    Forcing = 2,
    /// Waiting for a complete UTF-8 boundary after exhausting the budget.
    WaitingUtf8 = 3,
    /// Finished, while still watching for a new start delimiter.
    Done = 4,
}

impl TryFrom<u32> for State {
    type Error = crate::Error;

    fn try_from(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::Idle),
            1 => Ok(Self::Counting),
            2 => Ok(Self::Forcing),
            3 => Ok(Self::WaitingUtf8),
            4 => Ok(Self::Done),
            _ => Err(invalid("unknown reasoning budget state")),
        }
    }
}

#[derive(Clone, Debug)]
struct Matcher {
    sequences: Vec<Vec<i32>>,
    suffix: Vec<i32>,
    maximum: usize,
}

impl Matcher {
    fn new(sequences: Vec<Vec<i32>>) -> Result<Self> {
        if sequences.len() > 64 || sequences.iter().any(|sequence| sequence.len() > 1024) {
            return Err(invalid("reasoning delimiter extent exceeds bound"));
        }
        let mut unique = Vec::new();
        for sequence in sequences {
            if !sequence.is_empty() && !unique.contains(&sequence) {
                unique.push(sequence);
            }
        }
        let maximum = unique.iter().map(Vec::len).max().unwrap_or(0);
        let mut suffix = Vec::new();
        suffix.try_reserve_exact(maximum)?;
        Ok(Self {
            sequences: unique,
            suffix,
            maximum,
        })
    }

    fn advance(&mut self, token: i32) -> Option<usize> {
        if self.maximum == 0 {
            return None;
        }
        if self.suffix.len() == self.maximum {
            self.suffix.remove(0);
        }
        self.suffix.push(token);
        let matched = self
            .sequences
            .iter()
            .enumerate()
            .filter(|(_, sequence)| self.suffix.ends_with(sequence))
            .max_by_key(|(_, sequence)| sequence.len())
            .map(|(index, _)| index);
        if matched.is_some() {
            self.suffix.clear();
        }
        matched
    }
}

/// Independently owned reasoning delimiter, countdown and forced-token state.
#[derive(Clone, Debug)]
pub struct ReasoningBudget {
    start: Matcher,
    end: Matcher,
    forced: Vec<i32>,
    budget: i32,
    remaining: i32,
    initial_state: State,
    state: State,
    force_position: usize,
    end_match: Option<usize>,
}

impl ReasoningBudget {
    /// Constructs a budget, discarding empty and duplicate delimiter sequences.
    ///
    /// Initial counting with a nonpositive budget immediately enters forcing.
    ///
    /// # Errors
    /// Rejects more than 64 start/end sequences, sequences or forced-token lists
    /// longer than 1024 tokens, and allocation failures.
    pub fn new(
        start: Vec<Vec<i32>>,
        end: Vec<Vec<i32>>,
        forced: Vec<i32>,
        budget: i32,
        initial: State,
    ) -> Result<Self> {
        if forced.len() > 1024 {
            return Err(invalid("reasoning forced-token extent exceeds bound"));
        }
        let initial_state = if initial == State::Counting && budget <= 0 {
            State::Forcing
        } else {
            initial
        };
        Ok(Self {
            start: Matcher::new(start)?,
            end: Matcher::new(end)?,
            forced,
            budget,
            remaining: budget,
            initial_state,
            state: initial_state,
            force_position: 0,
            end_match: None,
        })
    }

    /// Returns the current budget phase.
    #[must_use]
    pub const fn state(&self) -> State {
        self.state
    }

    /// Returns the end delimiter that completed the latest block, if any.
    #[must_use]
    pub fn end_match(&self) -> Option<&[i32]> {
        self.end_match
            .map(|index| self.end.sequences[index].as_slice())
    }

    /// Returns the candidate token required by the current forcing phase.
    #[must_use]
    pub fn forced_token(&self) -> Option<i32> {
        if self.state == State::Forcing {
            self.forced.get(self.force_position).copied()
        } else {
            None
        }
    }

    /// Sets non-forced candidate logits to negative infinity, retaining all
    /// other bits and the order of the supplied candidates.
    ///
    /// # Errors
    /// Rejects mismatched token/logit lengths before changing any logit.
    pub fn apply(&self, tokens: &[i32], logits: &mut [f32]) -> Result<()> {
        if tokens.len() != logits.len() {
            return Err(invalid("reasoning candidate token/logit extent mismatch"));
        }
        if let Some(forced) = self.forced_token()
            && tokens.contains(&forced)
        {
            for (&token, logit) in tokens.iter().zip(logits) {
                if token != forced {
                    *logit = f32::NEG_INFINITY;
                }
            }
        }
        Ok(())
    }

    const fn begin_forcing(&mut self) {
        self.state = State::Forcing;
        self.force_position = 0;
    }

    /// Accepts a token using the null-vocabulary, complete-boundary behavior.
    pub fn accept(&mut self, token: i32) {
        self.accept_with_utf8(token, true);
    }

    /// Accepts a token with its vocabulary piece's UTF-8 boundary result.
    /// Natural end delimiters take precedence over budget exhaustion. A new
    /// start delimiter after completion re-arms a fresh budget.
    pub fn accept_with_utf8(&mut self, token: i32, complete: bool) {
        match self.state {
            State::Idle | State::Done => {
                if self.start.advance(token).is_some() {
                    if self.state == State::Done {
                        self.end.suffix.clear();
                        self.end_match = None;
                    }
                    self.state = State::Counting;
                    self.remaining = self.budget;
                    if self.remaining <= 0 {
                        self.begin_forcing();
                    }
                }
            }
            State::Counting | State::WaitingUtf8 => {
                if let Some(index) = self.end.advance(token) {
                    self.state = State::Done;
                    self.end_match = Some(index);
                    return;
                }
                if self.state == State::WaitingUtf8 {
                    if complete {
                        self.begin_forcing();
                        self.end.suffix.clear();
                    }
                } else {
                    self.remaining -= 1;
                    if self.remaining <= 0 {
                        if complete {
                            self.begin_forcing();
                        } else {
                            self.state = State::WaitingUtf8;
                        }
                        self.end.suffix.clear();
                    }
                }
            }
            State::Forcing => {
                let matched = self.end.advance(token);
                self.force_position += 1;
                if self.force_position >= self.forced.len() {
                    self.state = State::Done;
                    self.end_match = matched;
                }
            }
        }
    }

    /// Manually enters forcing only when currently counting.
    pub fn force(&mut self) -> bool {
        if self.state != State::Counting {
            return false;
        }
        self.begin_forcing();
        self.end.suffix.clear();
        true
    }

    /// Restores the initial state, the original budget and empty delimiter histories.
    pub fn reset(&mut self) {
        self.state = self.initial_state;
        self.remaining = self.budget;
        self.start.suffix.clear();
        self.end.suffix.clear();
        self.force_position = 0;
        self.end_match = None;
    }
}

/// Tests the trailing byte boundary with the native Prism UTF-8 helper semantics.
///
/// This is a boundary test, not a validator for the entire input. Malformed
/// continuation-only tails remain incomplete; preceding bytes are not scanned.
#[must_use]
pub fn utf8_is_complete(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return true;
    }
    for (index, &byte) in bytes.iter().rev().take(4).enumerate() {
        if byte & 0xc0 != 0x80 {
            let expected = match byte {
                0xf0..=0xff => 4,
                0xe0..=0xef => 3,
                0xc0..=0xdf => 2,
                _ => 1,
            };
            return index + 1 >= expected;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_leaves_logits_untouched_when_forced_token_is_absent() -> Result<()> {
        let budget = ReasoningBudget::new(vec![], vec![], vec![99], 0, State::Counting)?;
        assert_eq!(budget.forced_token(), Some(99));
        let tokens = [1, 2, 3];
        let mut logits = [1.5, -0.5, 2.25];
        budget.apply(&tokens, &mut logits)?;
        assert_eq!(
            logits.iter().copied().map(f32::to_bits).collect::<Vec<_>>(),
            [1.5_f32.to_bits(), (-0.5_f32).to_bits(), 2.25_f32.to_bits()]
        );

        let present_tokens = [1, 99, 3];
        let mut masked_logits = [1.5, -0.5, 2.25];
        budget.apply(&present_tokens, &mut masked_logits)?;
        assert_eq!(
            masked_logits
                .iter()
                .copied()
                .map(f32::to_bits)
                .collect::<Vec<_>>(),
            [
                f32::NEG_INFINITY.to_bits(),
                (-0.5_f32).to_bits(),
                f32::NEG_INFINITY.to_bits(),
            ]
        );
        Ok(())
    }

    #[test]
    fn reset_restores_initial_state_across_counting_forcing_and_done() -> Result<()> {
        let mut counting =
            ReasoningBudget::new(vec![vec![1]], vec![vec![2]], vec![2], 3, State::Counting)?;
        assert_eq!(counting.state(), State::Counting);
        counting.accept(10);
        counting.accept(2);
        assert_eq!(counting.state(), State::Done);
        counting.reset();
        assert_eq!(counting.state(), State::Counting);
        assert_eq!(counting.end_match(), None);

        let mut forcing = ReasoningBudget::new(vec![], vec![vec![2]], vec![2], 0, State::Counting)?;
        assert_eq!(forcing.state(), State::Forcing);
        forcing.accept(2);
        assert_eq!(forcing.state(), State::Done);
        forcing.reset();
        assert_eq!(forcing.state(), State::Forcing);
        assert_eq!(forcing.forced_token(), Some(2));

        let mut done = ReasoningBudget::new(vec![vec![1]], vec![vec![2]], vec![2], 3, State::Done)?;
        assert_eq!(done.state(), State::Done);
        done.accept(1);
        assert_eq!(done.state(), State::Counting);
        done.reset();
        assert_eq!(done.state(), State::Done);
        Ok(())
    }

    #[test]
    fn utf8_is_complete_and_reasoning_budget_bounds_and_forcing_sequence() -> Result<()> {
        assert!(utf8_is_complete(b""));
        assert!(utf8_is_complete("hello 😀".as_bytes()));
        assert!(!utf8_is_complete(&[0xc3]));
        assert!(!utf8_is_complete(&[0xe2, 0x82]));
        assert!(!utf8_is_complete(&[0xf0, 0x9f, 0x98]));
        assert!(!utf8_is_complete(&[0x80, 0x80, 0x80, 0x80]));

        let mut rb =
            ReasoningBudget::new(vec![vec![10]], vec![vec![20]], vec![30, 20], 1, State::Done)?;
        assert_eq!(rb.forced_token(), None);
        rb.accept(10);
        assert_eq!(rb.state(), State::Counting);
        // One counted content token exhausts budget=1 and transitions to Forcing at idx 0 -> 30.
        rb.accept(5);
        assert_eq!(rb.state(), State::Forcing);
        assert_eq!(rb.forced_token(), Some(30));

        // Length mismatch between tokens and logits rejected by apply().
        let mut wrong_logits = [1.0_f32; 2];
        assert!(rb.apply(&[10, 30, 20], &mut wrong_logits).is_err());

        rb.accept(30);
        assert_eq!(rb.forced_token(), Some(20));
        rb.accept(20);
        assert_eq!(rb.state(), State::Done);
        assert_eq!(rb.forced_token(), None);
        Ok(())
    }
}
