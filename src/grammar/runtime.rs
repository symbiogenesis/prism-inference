//! Owned pushdown execution and candidate filtering for parsed GBNF rules.
//!
//! Preserves the Prism `llama-grammar.cpp` stack order, character/token matching,
//! partial UTF-8 and candidate rejection semantics. Rule positions use checked
//! indices, so cloning never retains pointers into another grammar's storage.

use std::collections::BTreeSet;
use std::sync::{Arc, Weak};

pub use super::gss::{GrammarGss, GssCheckpoint};
pub use super::mask::{GrammarMaskCache, VocabularyRadixTrie};

use super::{Element, Grammar};
use crate::{Result, invalid};

/// A checked instruction location in independently owned grammar storage.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Position {
    /// Rule index.
    pub rule: usize,
    /// Element offset within that rule.
    pub element: usize,
}

type Stack = Vec<Position>;

/// Incomplete native UTF-8 state retained between token pieces.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PartialUtf8 {
    /// Accumulated leading bits.
    pub value: u32,
    /// Remaining continuation bytes; minus one denotes an invalid sequence.
    pub remaining: i32,
}

/// A candidate token's code points and native decoder state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Candidate {
    /// Caller-owned candidate identity.
    pub index: u32,
    /// Decoded code points, including a terminal zero sentinel.
    pub code_points: Vec<u32>,
    /// State after decoding the candidate token.
    pub partial_utf8: PartialUtf8,
    /// Token ID for explicit TOKEN and `TOKEN_NOT` rules.
    pub id: i32,
}

/// Stack-only checkpoint valid for its originating execution owner.
#[derive(Clone, Debug)]
pub struct StackCheckpoint {
    owner: Weak<()>,
    stacks: Vec<Stack>,
}

/// A grammar execution owner with independent rule, stack and UTF-8 state.
#[derive(Debug)]
pub struct GrammarState {
    rules: Vec<Vec<Element>>,
    stacks: Vec<Stack>,
    partial: PartialUtf8,
    owner: Arc<()>,
}

impl Clone for GrammarState {
    fn clone(&self) -> Self {
        Self {
            rules: self.rules.clone(),
            stacks: self.stacks.clone(),
            partial: self.partial,
            owner: Arc::new(()),
        }
    }
}

pub(super) const fn end(element: Element) -> bool {
    element.kind == 0 || element.kind == 1
}

pub(super) fn instruction(rules: &[Vec<Element>], position: Position) -> Result<Element> {
    rules
        .get(position.rule)
        .and_then(|rule| rule.get(position.element))
        .copied()
        .ok_or_else(|| invalid("grammar instruction position outside rule storage"))
}

pub(super) const fn next(position: Position) -> Position {
    Position {
        rule: position.rule,
        element: position.element + 1,
    }
}

pub(super) fn alternatives(rule: &[Element]) -> Vec<usize> {
    std::iter::once(0)
        .chain(
            rule.iter()
                .enumerate()
                .filter(|(_, element)| element.kind == 1)
                .map(|(index, _)| index + 1),
        )
        .collect()
}

pub(super) fn reject_left_recursion(rules: &[Vec<Element>]) -> Result<()> {
    // The native detector marks only an explicitly empty alternative. It
    // does not propagate emptiness through wrapper rule references. Preserve
    // that distinction: nested nullable repetitions such as ([x]*)* are valid
    // source inputs, and advance's seen-set safely terminates their expansion.
    let nullable: Vec<_> = rules
        .iter()
        .map(|rule| alternatives(rule).iter().any(|&start| end(rule[start])))
        .collect();
    let mut edges = vec![Vec::new(); rules.len()];
    for (index, rule) in rules.iter().enumerate() {
        for start in alternatives(rule) {
            for element in &rule[start..] {
                if element.kind != 2 {
                    break;
                }
                let reference = usize::try_from(element.value)?;
                edges[index].push(reference);
                if !nullable[reference] {
                    break;
                }
            }
        }
    }
    let mut color = vec![0_u8; rules.len()];
    for start in 0..rules.len() {
        if color[start] != 0 {
            continue;
        }
        let mut todo = vec![(start, 0)];
        color[start] = 1;
        while let Some((node, edge)) = todo.last_mut() {
            if *edge == edges[*node].len() {
                color[*node] = 2;
                todo.pop();
                continue;
            }
            let child = edges[*node][*edge];
            *edge += 1;
            match color[child] {
                1 => return Err(invalid("left-recursive grammar is unsupported")),
                0 => {
                    color[child] = 1;
                    todo.push((child, 0));
                }
                _ => {}
            }
        }
    }
    Ok(())
}

fn advance(rules: &[Vec<Element>], stack: Stack, output: &mut Vec<Stack>) -> Result<()> {
    let mut todo = vec![stack];
    let mut seen = BTreeSet::new();
    while let Some(mut current) = todo.pop() {
        if current.len() > 65536 || seen.len() > 65536 || todo.len() > 65536 || output.len() > 8192
        {
            return Err(invalid("grammar stack expansion exceeds bound"));
        }
        if !seen.insert(current.clone()) {
            continue;
        }
        let Some(&position) = current.last() else {
            if !output.contains(&current) {
                output.push(current);
            }
            continue;
        };
        let element = instruction(rules, position)?;
        match element.kind {
            2 => {
                current.pop();
                let continuation = next(position);
                if !end(instruction(rules, continuation)?) {
                    current.push(continuation);
                }
                let rule = usize::try_from(element.value)?;
                for start in alternatives(&rules[rule]) {
                    let mut branch = current.clone();
                    if !end(rules[rule][start]) {
                        branch.push(Position {
                            rule,
                            element: start,
                        });
                    }
                    todo.push(branch);
                }
            }
            3 | 4 | 7 | 8 | 9 => {
                if !output.contains(&current) {
                    output.push(current);
                }
            }
            _ => {
                return Err(invalid(
                    "grammar stack does not end at a terminal instruction",
                ));
            }
        }
    }
    Ok(())
}

pub(super) fn match_character(
    rules: &[Vec<Element>],
    mut position: Position,
    point: u32,
) -> Result<(bool, Position)> {
    let first = instruction(rules, position)?;
    if !matches!(first.kind, 3 | 4 | 7) {
        return Err(invalid("character matcher needs a character terminal"));
    }
    let positive = first.kind != 4;
    let mut found = false;
    loop {
        let element = instruction(rules, position)?;
        let after = instruction(rules, next(position))?;
        if after.kind == 5 {
            found |= element.value <= point && point <= after.value;
            position = next(next(position));
        } else {
            found |= element.kind == 7 || element.value == point;
            position = next(position);
        }
        if instruction(rules, position)?.kind != 6 {
            break;
        }
    }
    Ok((found == positive, position))
}

pub(super) fn match_partial(
    rules: &[Vec<Element>],
    mut position: Position,
    partial: PartialUtf8,
) -> Result<bool> {
    if partial.remaining < 0 || (partial.remaining == 1 && partial.value < 2) {
        return Ok(false);
    }
    let bits = u32::try_from(partial.remaining)? * 6;
    if bits > 18 {
        return Err(invalid("partial UTF-8 continuation count outside bound"));
    }
    let mut low = partial.value << bits;
    let high = low | ((1_u32 << bits) - 1);
    if low == 0 {
        if partial.remaining == 2 {
            low = 1 << 11;
        } else if partial.remaining == 3 {
            low = 1 << 16;
        }
    }
    let positive = instruction(rules, position)?.kind != 4;
    loop {
        let element = instruction(rules, position)?;
        let after = instruction(rules, next(position))?;
        if after.kind == 5 {
            if element.value <= high && low <= after.value {
                return Ok(positive);
            }
            position = next(next(position));
        } else {
            if element.kind == 7 {
                return Ok(true);
            }
            if low <= element.value && element.value <= high {
                return Ok(positive);
            }
            position = next(position);
        }
        if instruction(rules, position)?.kind != 6 {
            break;
        }
    }
    Ok(!positive)
}

pub(super) const fn match_token(element: Element, token: i32) -> bool {
    let token = u32::from_ne_bytes(token.to_ne_bytes());
    (element.value == token) == (element.kind == 8)
}

fn accept_character(
    rules: &[Vec<Element>],
    stack: &Stack,
    point: u32,
    output: &mut Vec<Stack>,
) -> Result<()> {
    let Some(&position) = stack.last() else {
        return Ok(());
    };
    if matches!(instruction(rules, position)?.kind, 8 | 9) {
        return Ok(());
    }
    let (matches, after) = match_character(rules, position, point)?;
    if matches {
        let mut following = stack[..stack.len() - 1].to_vec();
        if !end(instruction(rules, after)?) {
            following.push(after);
        }
        advance(rules, following, output)?;
    }
    Ok(())
}

const MAX_DIAGNOSTIC_PREVIEW_BYTES: usize = 256;

pub(super) fn bounded_utf8_preview(bytes: &[u8]) -> String {
    let sample = &bytes[..bytes.len().min(MAX_DIAGNOSTIC_PREVIEW_BYTES)];
    let preview = String::from_utf8_lossy(sample);
    if bytes.len() <= MAX_DIAGNOSTIC_PREVIEW_BYTES && preview.len() <= MAX_DIAGNOSTIC_PREVIEW_BYTES
    {
        return preview.into_owned();
    }
    let mut end = preview
        .len()
        .min(MAX_DIAGNOSTIC_PREVIEW_BYTES.saturating_sub(3));
    while end > 0 && !preview.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", &preview[..end])
}

const fn is_valid_utf8_continuation(partial: PartialUtf8, byte: u8) -> bool {
    if (byte & 0xc0) != 0x80 {
        return false;
    }
    match (partial.remaining, partial.value) {
        (1, 0..=1) => false,
        (2, 0) => byte >= 0xa0,
        (2, 0x0d) => byte <= 0x9f,
        (3, 0) => byte >= 0x90,
        (3, 4) => byte <= 0x8f,
        (1..=3, _) => true,
        _ => false,
    }
}

pub(super) fn decode_utf8(bytes: &[u8], mut partial: PartialUtf8) -> (Vec<u32>, PartialUtf8) {
    let mut points = Vec::new();
    let mut position = 0;
    if !(0..=3).contains(&partial.remaining) {
        points.push(0xfffd);
        partial = PartialUtf8::default();
    }
    while position < bytes.len() && bytes[position] != 0 {
        let byte = bytes[position];
        if partial.remaining > 0 {
            if is_valid_utf8_continuation(partial, byte) {
                partial.value = (partial.value << 6) | u32::from(byte & 0x3f);
                partial.remaining -= 1;
                position += 1;
                if partial.remaining == 0 {
                    if partial.value <= 0x10_ffff && !(0xd800..=0xdfff).contains(&partial.value) {
                        points.push(partial.value);
                    } else {
                        points.push(0xfffd);
                    }
                    partial.value = 0;
                }
            } else {
                points.push(0xfffd);
                partial = PartialUtf8::default();
            }
            continue;
        }
        match byte {
            0x01..=0x7f => {
                points.push(u32::from(byte));
            }
            0xc2..=0xdf => {
                partial = PartialUtf8 {
                    value: u32::from(byte & 0x1f),
                    remaining: 1,
                };
            }
            0xe0..=0xef => {
                partial = PartialUtf8 {
                    value: u32::from(byte & 0x0f),
                    remaining: 2,
                };
            }
            0xf0..=0xf4 => {
                partial = PartialUtf8 {
                    value: u32::from(byte & 0x07),
                    remaining: 3,
                };
            }
            _ => {
                points.push(0xfffd);
                partial = PartialUtf8::default();
            }
        }
        position += 1;
    }
    (points, partial)
}

impl GrammarState {
    /// Builds execution state for a named root of a parsed grammar.
    ///
    /// # Errors
    /// Rejects a missing root or any malformed/left-recursive rule structure.
    pub fn from_grammar(grammar: Grammar, root: &str) -> Result<Self> {
        let start = usize::try_from(
            *grammar
                .symbols
                .get(root)
                .ok_or_else(|| invalid("grammar root symbol is missing"))?,
        )?;
        Self::from_rules(grammar.rules, start)
    }

    /// Builds execution state from native flattened rule instructions.
    ///
    /// # Errors
    /// Rejects empty/malformed rules, invalid references, left recursion, more
    /// than 4096 rules/65536 elements, or excessive stack expansion.
    pub fn from_rules(mut rules: Vec<Vec<Element>>, start: usize) -> Result<Self> {
        if rules.is_empty()
            || rules.len() > 4096
            || start >= rules.len()
            || rules.iter().map(Vec::len).sum::<usize>() > 65536
        {
            return Err(invalid("grammar rule extent outside bound"));
        }
        let count = rules.len();
        for rule in &mut rules {
            let last = rule
                .iter()
                .position(|element| element.kind == 0)
                .ok_or_else(|| invalid("grammar rule needs END"))?;
            rule.truncate(last + 1);
            rule[last].value = 0;
            for element in rule {
                if element.kind > 9
                    || (element.kind == 2 && usize::try_from(element.value)? >= count)
                {
                    return Err(invalid("invalid grammar instruction or reference"));
                }
            }
        }
        reject_left_recursion(&rules)?;
        let mut stacks = Vec::new();
        for element in alternatives(&rules[start]) {
            let stack = if end(rules[start][element]) {
                Vec::new()
            } else {
                vec![Position {
                    rule: start,
                    element,
                }]
            };
            advance(&rules, stack, &mut stacks)?;
        }
        Ok(Self {
            rules,
            stacks,
            partial: PartialUtf8::default(),
            owner: Arc::new(()),
        })
    }

    /// Returns this owner's immutable rule storage.
    #[must_use]
    pub fn rules(&self) -> &[Vec<Element>] {
        &self.rules
    }

    /// Returns checked instruction stacks in native expansion order.
    #[must_use]
    pub fn stacks(&self) -> &[Vec<Position>] {
        &self.stacks
    }

    /// Returns the pending partial UTF-8 decoder state across token pieces.
    #[must_use]
    pub const fn partial_utf8(&self) -> PartialUtf8 {
        self.partial
    }

    /// Captures only the pushdown stacks, retaining the originating identity.
    #[must_use]
    pub fn save_stacks(&self) -> StackCheckpoint {
        StackCheckpoint {
            owner: Arc::downgrade(&self.owner),
            stacks: self.stacks.clone(),
        }
    }

    /// Restores a stack-only checkpoint from this exact execution owner.
    ///
    /// # Errors
    /// Rejects checkpoints from clones, other grammars or released owners.
    pub fn restore_stacks(&mut self, checkpoint: &StackCheckpoint) -> Result<()> {
        if !checkpoint
            .owner
            .upgrade()
            .is_some_and(|owner| Arc::ptr_eq(&owner, &self.owner))
        {
            return Err(invalid(
                "saved stacks belong to a different grammar lifetime",
            ));
        }
        self.stacks.clone_from(&checkpoint.stacks);
        Ok(())
    }

    /// Accepts a raw code point; a mismatch leaves no surviving stacks.
    ///
    /// # Errors
    /// Rejects malformed instruction state or resource-bound exhaustion.
    pub fn accept_codepoint(&mut self, point: u32) -> Result<()> {
        let mut output = Vec::new();
        for stack in &self.stacks {
            accept_character(&self.rules, stack, point, &mut output)?;
        }
        self.stacks = output;
        Ok(())
    }

    /// Accepts a token ID and its exact vocabulary piece bytes.
    ///
    /// Explicit token terminals consume the token ID. Character terminals
    /// consume every complete code point while retaining partial UTF-8 bits.
    ///
    /// # Errors
    /// Returns an error when no stacks survive (state remains advanced), for
    /// a piece above 65536 bytes, malformed instructions or resource limits.
    pub fn accept_token(&mut self, token: i32, piece: &[u8]) -> Result<()> {
        if piece.len() > 65536 {
            return Err(invalid("grammar token piece exceeds bound"));
        }
        let (points, partial) = decode_utf8(piece, self.partial);
        let mut output = Vec::new();
        for stack in &self.stacks {
            let Some(&position) = stack.last() else {
                continue;
            };
            let element = instruction(&self.rules, position)?;
            if matches!(element.kind, 8 | 9) {
                if match_token(element, token) {
                    let mut following = stack[..stack.len() - 1].to_vec();
                    let after = next(position);
                    if !end(instruction(&self.rules, after)?) {
                        following.push(after);
                    }
                    advance(&self.rules, following, &mut output)?;
                }
            } else {
                let mut current = vec![stack.clone()];
                for &point in &points {
                    let mut following = Vec::new();
                    for branch in &current {
                        accept_character(&self.rules, branch, point, &mut following)?;
                    }
                    current = following;
                    if current.is_empty() {
                        break;
                    }
                }
                for branch in current {
                    if !output.contains(&branch) {
                        output.push(branch);
                    }
                }
            }
        }
        self.stacks = output;
        self.partial = partial;
        if self.stacks.is_empty() {
            return Err(invalid(format!(
                "Unexpected empty grammar stack after accepting piece: {} ({token})",
                bounded_utf8_preview(piece)
            )));
        }
        Ok(())
    }

    /// Returns candidates rejected by one selected stack in native order.
    ///
    /// # Errors
    /// Rejects an invalid stack, more than 4096 candidates, missing zero
    /// sentinels, code-point vectors above 1024 entries or invalid UTF-8 state.
    pub fn reject_candidates(
        &self,
        stack: usize,
        candidates: &[Candidate],
    ) -> Result<Vec<Candidate>> {
        let stack = self
            .stacks
            .get(stack)
            .ok_or_else(|| invalid("grammar stack index outside extent"))?;
        if candidates.len() > 4096
            || candidates.iter().any(|candidate| {
                candidate.code_points.is_empty()
                    || candidate.code_points.len() > 1024
                    || candidate.code_points.last() != Some(&0)
                    || !(-1..=3).contains(&candidate.partial_utf8.remaining)
            })
        {
            return Err(invalid(
                "grammar candidate extent or partial state outside bound",
            ));
        }
        let cursors = (0..candidates.len()).map(|index| (index, 0)).collect();
        let rejected = reject_for_stack(&self.rules, stack.clone(), candidates, cursors)?;
        Ok(rejected
            .into_iter()
            .map(|(index, _)| candidates[index].clone())
            .collect())
    }
}

type Cursor = (usize, usize);
enum RejectWork {
    Stack(Stack, Vec<Cursor>),
    Continue(Vec<Stack>, usize),
    Finish(Vec<Cursor>),
}

fn reject_for_stack(
    rules: &[Vec<Element>],
    stack: Stack,
    candidates: &[Candidate],
    cursors: Vec<Cursor>,
) -> Result<Vec<Cursor>> {
    let mut todo = vec![RejectWork::Stack(stack, cursors)];
    let mut rejected = Vec::new();
    while let Some(work) = todo.pop() {
        match work {
            RejectWork::Continue(stacks, index) => {
                if index < stacks.len() && !rejected.is_empty() {
                    let stack = stacks[index].clone();
                    todo.push(RejectWork::Continue(stacks, index + 1));
                    todo.push(RejectWork::Stack(stack, std::mem::take(&mut rejected)));
                }
            }
            RejectWork::Finish(mut immediate) => {
                immediate.extend(rejected.iter().map(|&(index, offset)| (index, offset - 1)));
                rejected = immediate;
            }
            RejectWork::Stack(stack, cursors) => {
                rejected.clear();
                let Some(&position) = stack.last() else {
                    rejected = cursors
                        .into_iter()
                        .filter(|&(index, offset)| {
                            candidates[index].code_points[offset] != 0
                                || candidates[index].partial_utf8.remaining != 0
                        })
                        .collect();
                    continue;
                };
                let element = instruction(rules, position)?;
                if matches!(element.kind, 8 | 9) {
                    rejected = cursors
                        .into_iter()
                        .filter(|&(index, offset)| {
                            let candidate = &candidates[index];
                            if candidate.code_points[offset] == 0 {
                                candidate.partial_utf8.remaining != 0
                            } else {
                                !match_token(element, candidate.id)
                            }
                        })
                        .collect();
                    continue;
                }
                let mut following_candidates = Vec::new();
                for (index, offset) in cursors {
                    let candidate = &candidates[index];
                    let point = candidate.code_points[offset];
                    if point == 0 {
                        if candidate.partial_utf8.remaining != 0
                            && !match_partial(rules, position, candidate.partial_utf8)?
                        {
                            rejected.push((index, offset));
                        }
                    } else if match_character(rules, position, point)?.0 {
                        following_candidates.push((index, offset + 1));
                    } else {
                        rejected.push((index, offset));
                    }
                }
                let after = match_character(rules, position, 0)?.1;
                let mut following = stack[..stack.len() - 1].to_vec();
                if !end(instruction(rules, after)?) {
                    following.push(after);
                }
                let mut stacks = Vec::new();
                advance(rules, following, &mut stacks)?;
                todo.push(RejectWork::Finish(std::mem::take(&mut rejected)));
                todo.push(RejectWork::Continue(stacks, 0));
                rejected = following_candidates;
            }
        }
    }
    Ok(rejected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_utf8_enforces_rfc3629_and_reprocesses_interrupting_ascii() {
        let (points, partial) = decode_utf8(&[0xc2, b'a'], PartialUtf8::default());
        assert_eq!(points, vec![0xfffd, u32::from(b'a')]);
        assert_eq!(partial, PartialUtf8::default());

        let (points, partial) = decode_utf8(&[0xc0, 0xaf, b'x'], PartialUtf8::default());
        assert_eq!(points, vec![0xfffd, 0xfffd, u32::from(b'x')]);
        assert_eq!(partial, PartialUtf8::default());

        let (points, partial) = decode_utf8(&[0xed, 0xa0, 0x80, b'y'], PartialUtf8::default());
        assert_eq!(points, vec![0xfffd, 0xfffd, 0xfffd, u32::from(b'y')]);
        assert_eq!(partial, PartialUtf8::default());

        let (points, partial) = decode_utf8(&[0xe0, 0x80, 0x80], PartialUtf8::default());
        assert_eq!(points, vec![0xfffd, 0xfffd, 0xfffd]);
        assert_eq!(partial, PartialUtf8::default());

        let (points, partial) = decode_utf8(&[0xf4, 0x90, 0x80, 0x80], PartialUtf8::default());
        assert_eq!(points, vec![0xfffd, 0xfffd, 0xfffd, 0xfffd]);
        assert_eq!(partial, PartialUtf8::default());

        let (points_first, partial_mid) = decode_utf8(&[0xe2, 0x82], PartialUtf8::default());
        assert_eq!(points_first, Vec::<u32>::new());
        let (points_second, partial_done) = decode_utf8(&[0xac], partial_mid);
        assert_eq!(points_second, vec![0x20ac]);
        assert_eq!(partial_done, PartialUtf8::default());

        let (_, pending) = decode_utf8(&[0xe1, 0x80], PartialUtf8::default());
        let (interrupted, reset) = decode_utf8(b"z", pending);
        assert_eq!(interrupted, vec![0xfffd, u32::from(b'z')]);
        assert_eq!(reset, PartialUtf8::default());
    }

    #[test]
    fn accept_token_error_caps_piece_preview_on_utf8_boundary() -> Result<()> {
        let rules = vec![vec![
            Element {
                kind: 3,
                value: u32::from(b'a'),
            },
            Element { kind: 0, value: 0 },
        ]];
        let mut state = GrammarState::from_rules(rules, 0)?;
        let long_piece = "€".repeat(200);
        let err = state
            .accept_token(1, long_piece.as_bytes())
            .err()
            .ok_or_else(|| invalid("expected stack error"))?;
        let message = err.to_string();
        let rest = message
            .strip_prefix("Unexpected empty grammar stack after accepting piece: ")
            .ok_or_else(|| invalid("unexpected error prefix"))?;
        let preview = rest
            .strip_suffix(" (1)")
            .ok_or_else(|| invalid("unexpected error suffix"))?;
        assert!(preview.len() <= MAX_DIAGNOSTIC_PREVIEW_BYTES);
        assert!(preview.ends_with("..."));
        Ok(())
    }
}
