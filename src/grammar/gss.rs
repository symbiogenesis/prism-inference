//! Arena-backed Graph-Structured Stack (`GrammarGss`) for constrained decoding.
//!
//! Shares common stack prefixes across alternative grammar branches via a
//! hash-consed parent-pointer arena so speculative multi-branch decoding and
//! checkpoint/rollback avoid cloning full `Vec<Position>` stacks.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Weak};

use super::runtime::{
    GrammarState, PartialUtf8, Position, alternatives, bounded_utf8_preview, decode_utf8, end,
    instruction, match_character, match_token, next, reject_left_recursion,
};
use super::{Element, Grammar};
use crate::{Result, invalid};

/// Sentinel stack head representing an empty pushdown stack (`[]`).
const EMPTY_STACK: u32 = u32::MAX;
/// Maximum number of arena nodes retained before rejecting or compacting.
const MAX_GSS_NODES: usize = 1_048_576;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GssNode {
    position: Position,
    parent: u32,
    depth: u32,
}

/// O(1) rollback watermark captured from a single [`GrammarGss`] owner.
#[derive(Clone, Debug)]
pub struct GssCheckpoint {
    owner: Weak<()>,
    node_len: usize,
    active: Vec<u32>,
    partial: PartialUtf8,
    epoch: u64,
}

/// Grammar pushdown execution state backed by an arena Graph-Structured Stack.
///
/// Functionally equivalent to [`GrammarState`] stacks while sharing stack tails
/// across active branches and supporting O(1) speculative [`Self::checkpoint`]
/// and [`Self::rollback`].
#[derive(Debug)]
pub struct GrammarGss {
    rules: Vec<Vec<Element>>,
    nodes: Vec<GssNode>,
    intern: HashMap<(u32, Position), u32>,
    active: Vec<u32>,
    partial: PartialUtf8,
    epoch: u64,
    owner: Arc<()>,
}

impl Clone for GrammarGss {
    fn clone(&self) -> Self {
        Self {
            rules: self.rules.clone(),
            nodes: self.nodes.clone(),
            intern: self.intern.clone(),
            active: self.active.clone(),
            partial: self.partial,
            epoch: self.epoch,
            owner: Arc::new(()),
        }
    }
}

impl GrammarGss {
    /// Builds a GSS execution state for a named root of a parsed grammar.
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

    /// Builds a GSS execution state from native flattened rule instructions.
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
        let mut gss = Self {
            rules,
            nodes: Vec::new(),
            intern: HashMap::new(),
            active: Vec::new(),
            partial: PartialUtf8::default(),
            epoch: 0,
            owner: Arc::new(()),
        };
        let mut output = Vec::new();
        let mut output_seen = HashSet::new();
        for element in alternatives(&gss.rules[start]) {
            let head = if end(gss.rules[start][element]) {
                EMPTY_STACK
            } else {
                gss.intern_push(
                    EMPTY_STACK,
                    Position {
                        rule: start,
                        element,
                    },
                )?
            };
            gss.advance_head(head, &mut output, &mut output_seen)?;
        }
        gss.active = output;
        Ok(gss)
    }

    /// Captures a GSS execution state directly from an existing [`GrammarState`].
    ///
    /// # Errors
    /// Returns an error if materializing stacks exceeds the bounded 1,048,576-node arena limit.
    pub fn from_state(state: &GrammarState) -> Result<Self> {
        let mut gss = Self {
            rules: state.rules().to_vec(),
            nodes: Vec::new(),
            intern: HashMap::new(),
            active: Vec::with_capacity(state.stacks().len()),
            partial: state.partial_utf8(),
            epoch: 0,
            owner: Arc::new(()),
        };
        let mut active_seen = HashSet::with_capacity(state.stacks().len());
        for stack in state.stacks() {
            let mut head = EMPTY_STACK;
            for &pos in stack {
                head = gss.intern_push(head, pos)?;
            }
            if active_seen.insert(head) {
                gss.active.push(head);
            }
        }
        Ok(gss)
    }

    /// Returns this owner's immutable rule storage.
    #[must_use]
    pub fn rules(&self) -> &[Vec<Element>] {
        &self.rules
    }

    /// Returns the number of active grammar execution branches.
    #[must_use]
    pub const fn stack_count(&self) -> usize {
        self.active.len()
    }

    /// Returns the number of allocated arena nodes (for testing compaction/rollback).
    #[must_use]
    pub const fn arena_node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the pending partial UTF-8 decoder state across token pieces.
    #[must_use]
    pub const fn partial_utf8(&self) -> PartialUtf8 {
        self.partial
    }

    /// Captures an O(1) rollback watermark bound to this [`GrammarGss`] lifetime and epoch.
    #[must_use]
    pub fn checkpoint(&self) -> GssCheckpoint {
        GssCheckpoint {
            owner: Arc::downgrade(&self.owner),
            node_len: self.nodes.len(),
            active: self.active.clone(),
            partial: self.partial,
            epoch: self.epoch,
        }
    }

    /// Restores active branches, partial UTF-8 state and arena watermark in O(1) node work.
    ///
    /// # Errors
    /// Rejects checkpoints from clones, other grammars, released owners or compacted epochs.
    pub fn rollback(&mut self, checkpoint: &GssCheckpoint) -> Result<()> {
        if !checkpoint
            .owner
            .upgrade()
            .is_some_and(|owner| Arc::ptr_eq(&owner, &self.owner))
        {
            return Err(invalid(
                "GSS checkpoint belongs to a different grammar lifetime",
            ));
        }
        if checkpoint.epoch != self.epoch || checkpoint.node_len > self.nodes.len() {
            return Err(invalid(
                "GSS checkpoint is invalidated by a prior compaction",
            ));
        }
        self.nodes.truncate(checkpoint.node_len);
        self.active.clone_from(&checkpoint.active);
        self.partial = checkpoint.partial;
        if self.intern.len() > self.nodes.len().saturating_mul(4).saturating_add(1024) {
            self.rebuild_intern()?;
        }
        Ok(())
    }

    /// Garbage-collects unreachable arena nodes outside active stacks and advances the epoch.
    ///
    /// Invalidates all prior [`GssCheckpoint`] handles.
    ///
    /// # Errors
    /// Returns an error if arena indices cannot be converted.
    pub fn compact(&mut self) -> Result<()> {
        let old_len = self.nodes.len();
        let mut reachable = vec![false; old_len];
        for &head in &self.active {
            let mut cursor = head;
            while cursor != EMPTY_STACK {
                let idx = usize::try_from(cursor)?;
                if idx >= old_len || reachable[idx] {
                    break;
                }
                reachable[idx] = true;
                cursor = self.nodes[idx].parent;
            }
        }
        let mut remap = vec![EMPTY_STACK; old_len];
        let mut compacted = Vec::new();
        for (old_idx, node) in self.nodes.iter().enumerate() {
            if !reachable[old_idx] {
                continue;
            }
            let new_parent = if node.parent == EMPTY_STACK {
                EMPTY_STACK
            } else {
                remap[usize::try_from(node.parent)?]
            };
            let new_idx = u32::try_from(compacted.len())?;
            remap[old_idx] = new_idx;
            compacted.push(GssNode {
                position: node.position,
                parent: new_parent,
                depth: node.depth,
            });
        }
        for head in &mut self.active {
            if *head != EMPTY_STACK {
                *head = remap[usize::try_from(*head)?];
            }
        }
        self.nodes = compacted;
        self.rebuild_intern()?;
        self.epoch = self.epoch.wrapping_add(1);
        Ok(())
    }

    /// Accepts a raw code point; a mismatch leaves no surviving stacks.
    ///
    /// # Errors
    /// Rejects malformed instruction state or resource-bound exhaustion.
    pub fn accept_codepoint(&mut self, point: u32) -> Result<()> {
        let current = self.active.clone();
        let mut output = Vec::new();
        let mut output_seen = HashSet::new();
        for head in current {
            self.accept_character_head(head, point, &mut output, &mut output_seen)?;
        }
        self.active = output;
        Ok(())
    }

    /// Accepts a token ID and its exact vocabulary piece bytes.
    ///
    /// Matches [`GrammarState::accept_token`] semantics: explicit token terminals
    /// consume `token`; character terminals consume every decoded code point while
    /// preserving `PartialUtf8` state across pieces.
    ///
    /// # Errors
    /// Returns an error when no stacks survive (state remains advanced), for a
    /// piece above 65536 bytes, malformed instructions or resource limits.
    pub fn accept_token(&mut self, token: i32, piece: &[u8]) -> Result<()> {
        if piece.len() > 65536 {
            return Err(invalid("grammar token piece exceeds bound"));
        }
        let (points, partial) = decode_utf8(piece, self.partial);
        let starting = self.active.clone();
        let mut output = Vec::new();
        let mut output_seen = HashSet::new();
        for head in starting {
            let Some(position) = self.head_position(head) else {
                continue;
            };
            let element = instruction(&self.rules, position)?;
            if matches!(element.kind, 8 | 9) {
                if match_token(element, token) {
                    let base = self.nodes[usize::try_from(head)?].parent;
                    let after = next(position);
                    let next_head = if end(instruction(&self.rules, after)?) {
                        base
                    } else {
                        self.intern_push(base, after)?
                    };
                    self.advance_head(next_head, &mut output, &mut output_seen)?;
                }
            } else {
                let mut current = vec![head];
                for &point in &points {
                    let mut following = Vec::new();
                    let mut following_seen = HashSet::new();
                    for branch in current {
                        self.accept_character_head(
                            branch,
                            point,
                            &mut following,
                            &mut following_seen,
                        )?;
                    }
                    current = following;
                    if current.is_empty() {
                        break;
                    }
                }
                for branch in current {
                    if output_seen.insert(branch) {
                        output.push(branch);
                    }
                }
            }
        }
        self.active = output;
        self.partial = partial;
        if self.active.is_empty() {
            return Err(invalid(format!(
                "Unexpected empty grammar stack after accepting piece: {} ({token})",
                bounded_utf8_preview(piece)
            )));
        }
        Ok(())
    }

    /// Materializes every active pushdown branch into concrete `Vec<Position>` stacks
    /// in native expansion order.
    #[must_use]
    pub fn materialize_stacks(&self) -> Vec<Vec<Position>> {
        self.active
            .iter()
            .map(|&head| self.materialize_stack(head))
            .collect()
    }

    /// Materializes one stack head into bottom-to-top `Vec<Position>` order.
    pub(super) fn materialize_stack(&self, mut head: u32) -> Vec<Position> {
        let mut stack = Vec::new();
        while head != EMPTY_STACK {
            let Some(node) = usize::try_from(head)
                .ok()
                .and_then(|idx| self.nodes.get(idx))
            else {
                break;
            };
            stack.push(node.position);
            head = node.parent;
        }
        stack.reverse();
        stack
    }

    fn head_position(&self, head: u32) -> Option<Position> {
        if head == EMPTY_STACK {
            return None;
        }
        usize::try_from(head)
            .ok()
            .and_then(|idx| self.nodes.get(idx))
            .map(|node| node.position)
    }

    fn node_depth(&self, head: u32) -> Result<u32> {
        if head == EMPTY_STACK {
            return Ok(0);
        }
        let idx = usize::try_from(head)?;
        self.nodes
            .get(idx)
            .map(|node| node.depth)
            .ok_or_else(|| invalid("GSS node index outside arena extent"))
    }

    fn intern_push(&mut self, parent: u32, position: Position) -> Result<u32> {
        if let Some(&existing) = self.intern.get(&(parent, position))
            && let Some(node) = usize::try_from(existing)
                .ok()
                .and_then(|idx| self.nodes.get(idx))
            && node.parent == parent
            && node.position == position
        {
            return Ok(existing);
        }
        if self.nodes.len() >= MAX_GSS_NODES {
            return Err(invalid("grammar GSS arena exceeds node bound"));
        }
        let depth = self.node_depth(parent)?.saturating_add(1);
        let idx = u32::try_from(self.nodes.len())?;
        self.nodes.push(GssNode {
            position,
            parent,
            depth,
        });
        self.intern.insert((parent, position), idx);
        Ok(idx)
    }

    fn rebuild_intern(&mut self) -> Result<()> {
        self.intern.clear();
        self.intern.reserve(self.nodes.len());
        for (idx, node) in self.nodes.iter().enumerate() {
            self.intern
                .insert((node.parent, node.position), u32::try_from(idx)?);
        }
        Ok(())
    }

    fn advance_head(
        &mut self,
        head: u32,
        output: &mut Vec<u32>,
        output_seen: &mut HashSet<u32>,
    ) -> Result<()> {
        let mut todo = vec![head];
        let mut seen = HashSet::new();
        while let Some(current) = todo.pop() {
            if usize::try_from(self.node_depth(current)?)? > 65536
                || seen.len() > 65536
                || todo.len() > 65536
                || output.len() > 8192
            {
                return Err(invalid("grammar stack expansion exceeds bound"));
            }
            if !seen.insert(current) {
                continue;
            }
            let Some(position) = self.head_position(current) else {
                if output_seen.insert(EMPTY_STACK) {
                    output.push(EMPTY_STACK);
                }
                continue;
            };
            let element = instruction(&self.rules, position)?;
            match element.kind {
                2 => {
                    let parent = self.nodes[usize::try_from(current)?].parent;
                    let mut base = parent;
                    let continuation = next(position);
                    if !end(instruction(&self.rules, continuation)?) {
                        base = self.intern_push(base, continuation)?;
                    }
                    let rule = usize::try_from(element.value)?;
                    let starts = alternatives(&self.rules[rule]);
                    for start in starts {
                        let branch = if end(self.rules[rule][start]) {
                            base
                        } else {
                            self.intern_push(
                                base,
                                Position {
                                    rule,
                                    element: start,
                                },
                            )?
                        };
                        todo.push(branch);
                    }
                }
                3 | 4 | 7 | 8 | 9 => {
                    if output_seen.insert(current) {
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

    fn accept_character_head(
        &mut self,
        head: u32,
        point: u32,
        output: &mut Vec<u32>,
        output_seen: &mut HashSet<u32>,
    ) -> Result<()> {
        let Some(position) = self.head_position(head) else {
            return Ok(());
        };
        if matches!(instruction(&self.rules, position)?.kind, 8 | 9) {
            return Ok(());
        }
        let (matches, after) = match_character(&self.rules, position, point)?;
        if matches {
            let parent = self.nodes[usize::try_from(head)?].parent;
            let next_head = if end(instruction(&self.rules, after)?) {
                parent
            } else {
                self.intern_push(parent, after)?
            };
            self.advance_head(next_head, output, output_seen)?;
        }
        Ok(())
    }
}
