//! XGrammar-style compressed vocabulary radix trie and position mask cache.
//!
//! Precomputes context-independent allowed-token bitmasks per grammar rule
//! [`Position`] using DFS pruning over a path-compressed byte [`VocabularyRadixTrie`],
//! matching [`GrammarState::reject_candidates`](super::runtime::GrammarState::reject_candidates)
//! semantics across partial UTF-8 boundaries, `NUL` truncation and explicit token rules.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use super::Element;
use super::gss::GrammarGss;
use super::runtime::{
    GrammarState, PartialUtf8, Position, alternatives, decode_utf8, end, instruction,
    match_character, match_partial, match_token, next,
};
use crate::{Result, invalid};

/// Maximum number of vocabulary tokens accepted by [`VocabularyRadixTrie::build`].
const MAX_VOCAB_TRIE_TOKENS: usize = 524_288;
/// Maximum total NUL-truncated vocabulary byte storage retained by [`VocabularyRadixTrie`].
const MAX_VOCAB_TRIE_BYTES: usize = 16 * 1024 * 1024;
/// Maximum size in bytes permitted for a single vocabulary piece.
const MAX_VOCAB_PIECE_BYTES: usize = 65_536;
/// Maximum number of cached per-position bitmasks in [`GrammarMaskCache`].
const MAX_CACHED_POSITIONS: usize = 4_096;
/// Above this context-dependent count, fall back to a merged radix-trie walk.
const CONTEXT_TRIE_THRESHOLD: usize = 256;

#[derive(Clone, Copy, Debug)]
struct TrieNode {
    edge_start: u32,
    edge_end: u32,
    terminal_start: u32,
    terminal_end: u32,
    subtree_end: u32,
    children_start: u32,
    children_end: u32,
}

/// Path-compressed byte radix trie over a fixed vocabulary.
///
/// Pieces are truncated at the first `NUL` byte to match the UTF-8 grammar
/// decoder, stored contiguously, and indexed in lexicographical DFS order so
/// every trie subtree maps to a contiguous slice of DFS token IDs.
#[derive(Clone, Debug)]
pub struct VocabularyRadixTrie {
    vocab_size: usize,
    piece_bytes: Vec<u8>,
    piece_ranges: Vec<(u32, u32)>,
    edge_bytes: Vec<u8>,
    nodes: Vec<TrieNode>,
    dfs_tokens: Vec<u32>,
    token_dfs_pos: Vec<u32>,
    empty_piece_tokens: Vec<u32>,
    excluded_bitmask: Vec<u64>,
}

impl VocabularyRadixTrie {
    /// Builds a compressed byte radix trie for `pieces`, excluding `excluded` token IDs.
    ///
    /// Excluded tokens (for example padding or control tokens) are omitted from the
    /// radix trie and stored in [`Self::excluded_bitmask`] so grammar bitmasks keep
    /// them disabled unless explicitly constructed outside grammar constraints.
    ///
    /// # Errors
    /// Rejects vocabularies above 524,288 tokens, pieces above 65,536 bytes,
    /// total truncated storage above 16 MiB, or out-of-range excluded IDs.
    pub fn build<P: AsRef<[u8]>>(pieces: &[P], excluded: &[u32]) -> Result<Self> {
        if pieces.len() > MAX_VOCAB_TRIE_TOKENS {
            return Err(invalid("vocabulary size exceeds radix trie bound"));
        }
        let vocab_size = pieces.len();
        let words = vocab_size.div_ceil(64);
        let mut excluded_bitmask = vec![0_u64; words];
        for &tok in excluded {
            let idx = usize::try_from(tok)?;
            if idx >= vocab_size {
                return Err(invalid("excluded token index outside vocabulary extent"));
            }
            set_bit(&mut excluded_bitmask, idx);
        }

        let mut piece_bytes = Vec::new();
        let mut piece_ranges = Vec::with_capacity(vocab_size);
        let mut sorted_entries = Vec::with_capacity(vocab_size);
        for (idx, piece) in pieces.iter().enumerate() {
            let raw = piece.as_ref();
            let trimmed_len = raw.iter().position(|&byte| byte == 0).unwrap_or(raw.len());
            if trimmed_len > MAX_VOCAB_PIECE_BYTES {
                return Err(invalid("vocabulary token piece exceeds 65536-byte bound"));
            }
            let next_len = piece_bytes
                .len()
                .checked_add(trimmed_len)
                .ok_or_else(|| invalid("vocabulary byte extent overflows"))?;
            if next_len > MAX_VOCAB_TRIE_BYTES {
                return Err(invalid("vocabulary byte storage exceeds 16 MiB trie bound"));
            }
            let start = u32::try_from(piece_bytes.len())?;
            let len = u32::try_from(trimmed_len)?;
            piece_bytes.extend_from_slice(&raw[..trimmed_len]);
            piece_ranges.push((start, len));
            if !is_bit_set(&excluded_bitmask, idx) {
                sorted_entries.push((start, len, u32::try_from(idx)?));
            }
        }

        sorted_entries.sort_unstable_by(|a, b| {
            let slice_a = slice_from_range(&piece_bytes, a.0, a.1);
            let slice_b = slice_from_range(&piece_bytes, b.0, b.1);
            slice_a.cmp(slice_b).then_with(|| a.2.cmp(&b.2))
        });

        let mut dfs_tokens = Vec::with_capacity(sorted_entries.len());
        let mut token_dfs_pos = vec![u32::MAX; vocab_size];
        for (pos, &(_, _, token_id)) in sorted_entries.iter().enumerate() {
            let pos_u32 = u32::try_from(pos)?;
            dfs_tokens.push(token_id);
            token_dfs_pos[usize::try_from(token_id)?] = pos_u32;
        }

        let (nodes, edge_bytes) = build_radix_nodes(&piece_bytes, &sorted_entries)?;
        let empty_piece_tokens = if nodes.is_empty() {
            Vec::new()
        } else {
            let root = nodes[0];
            dfs_tokens[usize::try_from(root.terminal_start)?..usize::try_from(root.terminal_end)?]
                .to_vec()
        };

        Ok(Self {
            vocab_size,
            piece_bytes,
            piece_ranges,
            edge_bytes,
            nodes,
            dfs_tokens,
            token_dfs_pos,
            empty_piece_tokens,
            excluded_bitmask,
        })
    }

    /// Returns total vocabulary cardinality covered by this trie.
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Returns total compressed radix trie nodes.
    #[must_use]
    pub const fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the number of `u64` words needed for a full vocabulary bitmask.
    #[must_use]
    pub const fn required_bitmask_words(&self) -> usize {
        self.vocab_size.div_ceil(64)
    }

    /// Returns the `NUL`-truncated byte piece for `token_id`, if within bounds.
    #[must_use]
    pub fn token_piece(&self, token_id: u32) -> Option<&[u8]> {
        let idx = usize::try_from(token_id).ok()?;
        let &(start, len) = self.piece_ranges.get(idx)?;
        Some(slice_from_range(&self.piece_bytes, start, len))
    }

    /// Returns `true` when `token_id` was marked excluded when building this trie.
    #[must_use]
    pub fn is_excluded(&self, token_id: u32) -> bool {
        let Ok(idx) = usize::try_from(token_id) else {
            return true;
        };
        idx >= self.vocab_size || is_bit_set(&self.excluded_bitmask, idx)
    }

    /// Returns the packed bitmask of excluded vocabulary tokens.
    #[must_use]
    pub fn excluded_bitmask(&self) -> &[u64] {
        &self.excluded_bitmask
    }

    fn token_piece_unchecked(&self, token_id: u32) -> &[u8] {
        self.token_piece(token_id).unwrap_or(&[])
    }
}

fn slice_from_range(bytes: &[u8], start: u32, len: u32) -> &[u8] {
    let Ok(lo) = usize::try_from(start) else {
        return &[];
    };
    let Ok(span) = usize::try_from(len) else {
        return &[];
    };
    bytes.get(lo..lo.saturating_add(span)).unwrap_or(&[])
}

fn build_radix_nodes(
    piece_bytes: &[u8],
    sorted_entries: &[(u32, u32, u32)],
) -> Result<(Vec<TrieNode>, Vec<u8>)> {
    let mut nodes = vec![TrieNode {
        edge_start: 0,
        edge_end: 0,
        terminal_start: 0,
        terminal_end: 0,
        subtree_end: u32::try_from(sorted_entries.len())?,
        children_start: 0,
        children_end: 0,
    }];
    let mut edge_bytes = Vec::new();
    let mut todo = vec![(0_usize, 0_usize, sorted_entries.len(), 0_usize)];

    while let Some((node_idx, lo, hi, depth)) = todo.pop() {
        let term_end = lo
            + sorted_entries[lo..hi]
                .partition_point(|entry| usize::try_from(entry.1).is_ok_and(|len| len == depth));
        nodes[node_idx].terminal_start = u32::try_from(lo)?;
        nodes[node_idx].terminal_end = u32::try_from(term_end)?;
        nodes[node_idx].subtree_end = u32::try_from(hi)?;

        let mut cursor = term_end;
        let children_start = u32::try_from(nodes.len())?;
        while cursor < hi {
            let first = slice_from_range(
                piece_bytes,
                sorted_entries[cursor].0,
                sorted_entries[cursor].1,
            );
            let lead_byte = first[depth];
            let group_span = sorted_entries[cursor..hi].partition_point(|entry| {
                slice_from_range(piece_bytes, entry.0, entry.1)[depth] == lead_byte
            });
            let group_end = cursor + group_span;
            let last = slice_from_range(
                piece_bytes,
                sorted_entries[group_end - 1].0,
                sorted_entries[group_end - 1].1,
            );
            let lcp = common_prefix_len(&first[depth..], &last[depth..]);
            let edge_start = u32::try_from(edge_bytes.len())?;
            edge_bytes.extend_from_slice(&first[depth..depth + lcp]);
            let edge_end = u32::try_from(edge_bytes.len())?;

            let child_idx = nodes.len();
            nodes.push(TrieNode {
                edge_start,
                edge_end,
                terminal_start: u32::try_from(cursor)?,
                terminal_end: u32::try_from(cursor)?,
                subtree_end: u32::try_from(group_end)?,
                children_start: 0,
                children_end: 0,
            });
            todo.push((child_idx, cursor, group_end, depth + lcp));
            cursor = group_end;
        }
        let children_end = u32::try_from(nodes.len())?;
        nodes[node_idx].children_start = children_start;
        nodes[node_idx].children_end = children_end;
    }

    Ok((nodes, edge_bytes))
}

fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

#[derive(Clone, Debug)]
struct PositionMask {
    always_accepted: Vec<u64>,
    context_dependent: Vec<u32>,
}

/// XGrammar-inspired cache of context-independent per-[`Position`] allowed bitmasks.
///
/// Evaluates each grammar terminal [`Position`] once against a [`VocabularyRadixTrie`]
/// (assuming clean UTF-8 boundary state), separating tokens unconditionally accepted
/// from tokens whose legality depends on the active caller stack.
#[derive(Clone, Debug)]
pub struct GrammarMaskCache {
    rules: Vec<Vec<Element>>,
    trie: Arc<VocabularyRadixTrie>,
    positions: HashMap<Position, PositionMask>,
}

impl GrammarMaskCache {
    /// Creates an empty mask cache bound to `rules` and `trie`.
    ///
    /// # Errors
    /// Rejects empty rules or more than 4096 rules / 65536 elements.
    pub fn new(rules: Vec<Vec<Element>>, trie: Arc<VocabularyRadixTrie>) -> Result<Self> {
        if rules.is_empty()
            || rules.len() > 4096
            || rules.iter().map(Vec::len).sum::<usize>() > 65536
        {
            return Err(invalid("grammar rule extent outside bound"));
        }
        Ok(Self {
            rules,
            trie,
            positions: HashMap::new(),
        })
    }

    /// Creates a mask cache bound to the rules owned by `state`.
    ///
    /// # Errors
    /// Rejects invalid rule bounds.
    pub fn from_state(state: &GrammarState, trie: Arc<VocabularyRadixTrie>) -> Result<Self> {
        Self::new(state.rules().to_vec(), trie)
    }

    /// Returns the number of cached [`Position`] entries.
    #[must_use]
    pub fn cached_position_count(&self) -> usize {
        self.positions.len()
    }

    /// Populates `bitmask` with every allowed vocabulary token ID for `state`.
    ///
    /// Matches [`GrammarState::reject_candidates`](super::runtime::GrammarState::reject_candidates)
    /// across all active stacks in `state` (excluding tokens marked excluded in `trie`).
    ///
    /// # Errors
    /// Rejects mismatched rule storage, undersized bitmasks, malformed grammar state
    /// or resource-bound exhaustion.
    pub fn fill_allowed_bitmask(
        &mut self,
        state: &GrammarState,
        bitmask: &mut [u64],
    ) -> Result<()> {
        if state.rules() != self.rules.as_slice() {
            return Err(invalid("mask cache rules differ from GrammarState"));
        }
        self.fill_stacks_bitmask(state.stacks(), state.partial_utf8(), bitmask)
    }

    /// Populates `bitmask` with every allowed vocabulary token ID for a [`GrammarGss`].
    ///
    /// # Errors
    /// Rejects mismatched rule storage, undersized bitmasks, malformed grammar state
    /// or resource-bound exhaustion.
    pub fn fill_allowed_bitmask_gss(
        &mut self,
        gss: &GrammarGss,
        bitmask: &mut [u64],
    ) -> Result<()> {
        if gss.rules() != self.rules.as_slice() {
            return Err(invalid("mask cache rules differ from GrammarGss"));
        }
        let stacks = gss.materialize_stacks();
        self.fill_stacks_bitmask(&stacks, gss.partial_utf8(), bitmask)
    }

    /// Returns `true` when bit `token_id` is set in `bitmask`.
    #[must_use]
    pub fn is_token_allowed(bitmask: &[u64], token_id: u32) -> bool {
        let Ok(idx) = usize::try_from(token_id) else {
            return false;
        };
        is_bit_set(bitmask, idx)
    }

    /// Masks disallowed entries in `logits` to `f32::NEG_INFINITY` in place.
    ///
    /// # Errors
    /// Returns an error if `bitmask` has fewer words than `logits.len().div_ceil(64)`.
    pub fn apply_bitmask_to_logits(logits: &mut [f32], bitmask: &[u64]) -> Result<()> {
        let required = logits.len().div_ceil(64);
        if bitmask.len() < required {
            return Err(invalid("allowed bitmask shorter than logits vocabulary"));
        }
        for (word_idx, chunk) in logits.chunks_mut(64).enumerate() {
            let word = bitmask[word_idx];
            if word == u64::MAX {
                continue;
            }
            if word == 0 {
                chunk.fill(f32::NEG_INFINITY);
                continue;
            }
            for (bit_idx, logit) in chunk.iter_mut().enumerate() {
                if ((word >> bit_idx) & 1) == 0 {
                    *logit = f32::NEG_INFINITY;
                }
            }
        }
        Ok(())
    }

    fn fill_stacks_bitmask(
        &mut self,
        stacks: &[Vec<Position>],
        partial: PartialUtf8,
        bitmask: &mut [u64],
    ) -> Result<()> {
        let words = self.trie.required_bitmask_words();
        if bitmask.len() < words {
            return Err(invalid("bitmask slice smaller than vocabulary word count"));
        }
        bitmask[..words].fill(0);
        if self.trie.vocab_size() == 0 || stacks.is_empty() {
            return Ok(());
        }

        if partial.remaining != 0 {
            return walk_trie_exact(&self.trie, &self.rules, stacks, partial, bitmask);
        }

        let mut deferred_stacks = Vec::new();
        for stack in stacks {
            let Some(&position) = stack.last() else {
                for &tok in &self.trie.empty_piece_tokens {
                    set_bit(bitmask, usize::try_from(tok)?);
                }
                continue;
            };
            let mask = if let Some(cached) = self.positions.get(&position) {
                cached
            } else {
                let computed = build_position_mask(&self.trie, &self.rules, position)?;
                if self.positions.len() >= MAX_CACHED_POSITIONS {
                    // Apply uncached position without overflowing bounded cache map.
                    apply_position_mask(
                        &self.trie,
                        &self.rules,
                        &computed,
                        stack,
                        bitmask,
                        &mut deferred_stacks,
                    )?;
                    continue;
                }
                self.positions.insert(position, computed);
                &self.positions[&position]
            };
            apply_position_mask(
                &self.trie,
                &self.rules,
                mask,
                stack,
                bitmask,
                &mut deferred_stacks,
            )?;
        }

        if !deferred_stacks.is_empty() {
            walk_trie_exact(
                &self.trie,
                &self.rules,
                &deferred_stacks,
                PartialUtf8::default(),
                bitmask,
            )?;
        }
        Ok(())
    }
}

fn apply_position_mask(
    trie: &VocabularyRadixTrie,
    rules: &[Vec<Element>],
    mask: &PositionMask,
    stack: &[Position],
    bitmask: &mut [u64],
    deferred_stacks: &mut Vec<Vec<Position>>,
) -> Result<()> {
    for (dst, &src) in bitmask.iter_mut().zip(mask.always_accepted.iter()) {
        *dst |= src;
    }
    if mask.context_dependent.len() > CONTEXT_TRIE_THRESHOLD {
        deferred_stacks.push(stack.to_vec());
        return Ok(());
    }
    for &tok in &mask.context_dependent {
        let idx = usize::try_from(tok)?;
        if is_bit_set(bitmask, idx) {
            continue;
        }
        let piece = trie.token_piece_unchecked(tok);
        if stack_accepts_piece(rules, stack, PartialUtf8::default(), tok, piece)? {
            set_bit(bitmask, idx);
        }
    }
    Ok(())
}

const fn set_bit(bitmask: &mut [u64], idx: usize) {
    bitmask[idx >> 6] |= 1_u64 << (idx & 63);
}

fn is_bit_set(bitmask: &[u64], idx: usize) -> bool {
    let word = idx >> 6;
    bitmask
        .get(word)
        .is_some_and(|bits| ((bits >> (idx & 63)) & 1) != 0)
}

const fn is_valid_continuation(partial: PartialUtf8, byte: u8) -> bool {
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

fn decode_utf8_byte(byte: u8, mut partial: PartialUtf8) -> ([Option<u32>; 2], PartialUtf8) {
    let mut emitted = [None, None];
    let mut emitted_count = 0;

    if partial.remaining > 0 {
        if is_valid_continuation(partial, byte) {
            partial.value = (partial.value << 6) | u32::from(byte & 0x3f);
            partial.remaining -= 1;
            if partial.remaining == 0 {
                let point =
                    if partial.value <= 0x10_ffff && !(0xd800..=0xdfff).contains(&partial.value) {
                        partial.value
                    } else {
                        0xfffd
                    };
                emitted[0] = Some(point);
                partial.value = 0;
            }
            return (emitted, partial);
        }
        emitted[0] = Some(0xfffd);
        emitted_count = 1;
        partial = PartialUtf8::default();
    }

    match byte {
        0x01..=0x7f => {
            emitted[emitted_count] = Some(u32::from(byte));
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
            emitted[emitted_count] = Some(0xfffd);
            partial = PartialUtf8::default();
        }
    }
    (emitted, partial)
}

fn has_next_codepoint(bytes: &[u8], mut partial: PartialUtf8) -> bool {
    if !(0..=3).contains(&partial.remaining) {
        return true;
    }
    for &byte in bytes {
        if byte == 0 {
            break;
        }
        let (emitted, next_partial) = decode_utf8_byte(byte, partial);
        if emitted[0].is_some() {
            return true;
        }
        partial = next_partial;
    }
    false
}

fn advance_stack_vec(
    rules: &[Vec<Element>],
    stack: Vec<Position>,
    output: &mut Vec<Vec<Position>>,
) -> Result<()> {
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

#[derive(Default)]
struct StepResult {
    char_stacks: Vec<Vec<Position>>,
    has_empty: bool,
    token_elems: Vec<Element>,
}

fn step_stacks_on_codepoint(
    rules: &[Vec<Element>],
    char_stacks: &[Vec<Position>],
    point: u32,
) -> Result<StepResult> {
    let mut expanded = Vec::new();
    for stack in char_stacks {
        let Some(&position) = stack.last() else {
            continue;
        };
        if matches!(instruction(rules, position)?.kind, 8 | 9) {
            continue;
        }
        let (matched, after) = match_character(rules, position, point)?;
        if matched {
            let mut following = stack[..stack.len() - 1].to_vec();
            if !end(instruction(rules, after)?) {
                following.push(after);
            }
            advance_stack_vec(rules, following, &mut expanded)?;
        }
    }
    split_expanded_stacks(rules, expanded)
}

fn split_expanded_stacks(rules: &[Vec<Element>], stacks: Vec<Vec<Position>>) -> Result<StepResult> {
    let mut result = StepResult::default();
    for stack in stacks {
        let Some(&position) = stack.last() else {
            result.has_empty = true;
            continue;
        };
        let elem = instruction(rules, position)?;
        if matches!(elem.kind, 8 | 9) {
            if !result.token_elems.contains(&elem) {
                result.token_elems.push(elem);
            }
        } else {
            result.char_stacks.push(stack);
        }
    }
    Ok(result)
}

fn apply_token_elems_to_node(
    trie: &VocabularyRadixTrie,
    node: TrieNode,
    at_node_end: bool,
    depth_bytes: usize,
    partial: PartialUtf8,
    token_elems: &[Element],
    bitmask: &mut [u64],
) -> Result<()> {
    if token_elems.is_empty() {
        return Ok(());
    }
    if at_node_end && partial.remaining == 0 {
        let lo = usize::try_from(node.terminal_start)?;
        let hi = usize::try_from(node.terminal_end)?;
        for &tok in &trie.dfs_tokens[lo..hi] {
            set_bit(bitmask, usize::try_from(tok)?);
        }
    }
    let desc_lo = if at_node_end {
        node.terminal_end
    } else {
        node.terminal_start
    };
    let desc_hi = node.subtree_end;
    if desc_lo >= desc_hi {
        return Ok(());
    }

    for &elem in token_elems {
        if elem.kind == 8 {
            if let Ok(idx) = usize::try_from(elem.value)
                && idx < trie.vocab_size
                && !is_bit_set(&trie.excluded_bitmask, idx)
            {
                let dfs_pos = trie.token_dfs_pos[idx];
                if dfs_pos >= desc_lo && dfs_pos < desc_hi {
                    let piece = trie.token_piece_unchecked(elem.value);
                    if piece
                        .get(depth_bytes..)
                        .is_some_and(|suffix| has_next_codepoint(suffix, partial))
                    {
                        set_bit(bitmask, idx);
                    }
                }
            }
        } else {
            let start = usize::try_from(desc_lo)?;
            let end = usize::try_from(desc_hi)?;
            for &tok in &trie.dfs_tokens[start..end] {
                let tok_idx = usize::try_from(tok)?;
                if is_bit_set(bitmask, tok_idx) {
                    continue;
                }
                let tok_i32 = i32::from_ne_bytes(tok.to_ne_bytes());
                if !match_token(elem, tok_i32) {
                    continue;
                }
                let piece = trie.token_piece_unchecked(tok);
                if piece
                    .get(depth_bytes..)
                    .is_some_and(|suffix| has_next_codepoint(suffix, partial))
                {
                    set_bit(bitmask, tok_idx);
                }
            }
        }
    }
    Ok(())
}

struct BuildFrame {
    node_idx: usize,
    depth_bytes: usize,
    partial: PartialUtf8,
    char_stacks: Vec<Vec<Position>>,
    escaped_earlier: bool,
    escaped_at_cursor: bool,
}

fn build_position_mask(
    trie: &VocabularyRadixTrie,
    rules: &[Vec<Element>],
    position: Position,
) -> Result<PositionMask> {
    let words = trie.required_bitmask_words();
    let mut always_accepted = vec![0_u64; words];
    let mut context_dependent = Vec::new();
    let elem = instruction(rules, position)?;

    if matches!(elem.kind, 8 | 9) {
        if !trie.nodes.is_empty() {
            apply_token_elems_to_node(
                trie,
                trie.nodes[0],
                true,
                0,
                PartialUtf8::default(),
                &[elem],
                &mut always_accepted,
            )?;
        }
        return Ok(PositionMask {
            always_accepted,
            context_dependent,
        });
    }

    if trie.nodes.is_empty() {
        return Ok(PositionMask {
            always_accepted,
            context_dependent,
        });
    }

    let mut todo = vec![BuildFrame {
        node_idx: 0,
        depth_bytes: 0,
        partial: PartialUtf8::default(),
        char_stacks: vec![vec![position]],
        escaped_earlier: false,
        escaped_at_cursor: false,
    }];

    while let Some(frame) = todo.pop() {
        process_build_frame(
            trie,
            rules,
            frame,
            &mut todo,
            &mut always_accepted,
            &mut context_dependent,
        )?;
    }

    Ok(PositionMask {
        always_accepted,
        context_dependent,
    })
}

fn process_build_frame(
    trie: &VocabularyRadixTrie,
    rules: &[Vec<Element>],
    mut frame: BuildFrame,
    todo: &mut Vec<BuildFrame>,
    always_accepted: &mut [u64],
    context_dependent: &mut Vec<u32>,
) -> Result<()> {
    let node = trie.nodes[frame.node_idx];
    let edge = &trie.edge_bytes[usize::try_from(node.edge_start)?..usize::try_from(node.edge_end)?];

    for (offset, &byte) in edge.iter().enumerate() {
        let (emitted, next_partial) = decode_utf8_byte(byte, frame.partial);
        frame.partial = next_partial;
        frame.depth_bytes += 1;
        let at_node_end = offset + 1 == edge.len();

        for maybe_cp in emitted {
            let Some(cp) = maybe_cp else {
                continue;
            };
            frame.escaped_earlier |= frame.escaped_at_cursor;
            let step = step_stacks_on_codepoint(rules, &frame.char_stacks, cp)?;
            frame.char_stacks = step.char_stacks;
            frame.escaped_at_cursor = step.has_empty;
            apply_token_elems_to_node(
                trie,
                node,
                at_node_end,
                frame.depth_bytes,
                frame.partial,
                &step.token_elems,
                always_accepted,
            )?;
            if frame.char_stacks.is_empty() {
                if frame.escaped_earlier || frame.escaped_at_cursor {
                    record_escaped_subtree(
                        trie,
                        node,
                        at_node_end,
                        frame.partial,
                        frame.escaped_at_cursor,
                        always_accepted,
                        context_dependent,
                    )?;
                }
                return Ok(());
            }
        }
    }

    let term_lo = usize::try_from(node.terminal_start)?;
    let term_hi = usize::try_from(node.terminal_end)?;
    if term_lo < term_hi {
        let accepted_locally = if frame.partial.remaining == 0 {
            !frame.char_stacks.is_empty() || frame.escaped_at_cursor
        } else {
            stacks_match_partial(rules, &frame.char_stacks, frame.partial)?
        };
        if accepted_locally {
            for &tok in &trie.dfs_tokens[term_lo..term_hi] {
                set_bit(always_accepted, usize::try_from(tok)?);
            }
        } else if frame.escaped_earlier || frame.escaped_at_cursor {
            context_dependent.extend_from_slice(&trie.dfs_tokens[term_lo..term_hi]);
        }
    }

    for child_idx in usize::try_from(node.children_start)?..usize::try_from(node.children_end)? {
        todo.push(BuildFrame {
            node_idx: child_idx,
            depth_bytes: frame.depth_bytes,
            partial: frame.partial,
            char_stacks: frame.char_stacks.clone(),
            escaped_earlier: frame.escaped_earlier,
            escaped_at_cursor: frame.escaped_at_cursor,
        });
    }
    Ok(())
}

fn record_escaped_subtree(
    trie: &VocabularyRadixTrie,
    node: TrieNode,
    at_node_end: bool,
    partial: PartialUtf8,
    escaped_at_cursor: bool,
    always_accepted: &mut [u64],
    context_dependent: &mut Vec<u32>,
) -> Result<()> {
    let lo = usize::try_from(node.terminal_start)?;
    if at_node_end && partial.remaining == 0 && escaped_at_cursor {
        let hi = usize::try_from(node.terminal_end)?;
        for &tok in &trie.dfs_tokens[lo..hi] {
            set_bit(always_accepted, usize::try_from(tok)?);
        }
        let desc_lo = usize::try_from(node.terminal_end)?;
        let desc_hi = usize::try_from(node.subtree_end)?;
        context_dependent.extend_from_slice(&trie.dfs_tokens[desc_lo..desc_hi]);
    } else {
        let hi = usize::try_from(node.subtree_end)?;
        context_dependent.extend_from_slice(&trie.dfs_tokens[lo..hi]);
    }
    Ok(())
}

fn stacks_match_partial(
    rules: &[Vec<Element>],
    char_stacks: &[Vec<Position>],
    partial: PartialUtf8,
) -> Result<bool> {
    for stack in char_stacks {
        if let Some(&pos) = stack.last()
            && match_partial(rules, pos, partial)?
        {
            return Ok(true);
        }
    }
    Ok(false)
}

struct ExactFrame {
    node_idx: usize,
    depth_bytes: usize,
    partial: PartialUtf8,
    char_stacks: Vec<Vec<Position>>,
    has_empty: bool,
}

fn walk_trie_exact(
    trie: &VocabularyRadixTrie,
    rules: &[Vec<Element>],
    stacks: &[Vec<Position>],
    mut partial: PartialUtf8,
    bitmask: &mut [u64],
) -> Result<()> {
    if trie.nodes.is_empty() || stacks.is_empty() {
        return Ok(());
    }

    let mut split = split_expanded_stacks(rules, stacks.to_vec())?;
    let root = trie.nodes[0];
    if !(0..=3).contains(&partial.remaining) {
        apply_token_elems_to_node(trie, root, false, 0, partial, &split.token_elems, bitmask)?;
        if split.char_stacks.is_empty() {
            return Ok(());
        }
        split = step_stacks_on_codepoint(rules, &split.char_stacks, 0xfffd)?;
        partial = PartialUtf8::default();
    }

    apply_token_elems_to_node(trie, root, true, 0, partial, &split.token_elems, bitmask)?;
    if split.char_stacks.is_empty() && !split.has_empty {
        return Ok(());
    }

    let mut todo = vec![ExactFrame {
        node_idx: 0,
        depth_bytes: 0,
        partial,
        char_stacks: split.char_stacks,
        has_empty: split.has_empty,
    }];

    while let Some(mut frame) = todo.pop() {
        let node = trie.nodes[frame.node_idx];
        let edge =
            &trie.edge_bytes[usize::try_from(node.edge_start)?..usize::try_from(node.edge_end)?];
        let mut alive = true;

        for (offset, &byte) in edge.iter().enumerate() {
            let (emitted, next_partial) = decode_utf8_byte(byte, frame.partial);
            frame.partial = next_partial;
            frame.depth_bytes += 1;
            let at_node_end = offset + 1 == edge.len();

            for maybe_cp in emitted {
                let Some(cp) = maybe_cp else {
                    continue;
                };
                let step = step_stacks_on_codepoint(rules, &frame.char_stacks, cp)?;
                frame.char_stacks = step.char_stacks;
                frame.has_empty = step.has_empty;
                apply_token_elems_to_node(
                    trie,
                    node,
                    at_node_end,
                    frame.depth_bytes,
                    frame.partial,
                    &step.token_elems,
                    bitmask,
                )?;
                if frame.char_stacks.is_empty() && !frame.has_empty {
                    alive = false;
                    break;
                }
            }
            if !alive {
                break;
            }
        }

        if !alive {
            continue;
        }

        let term_lo = usize::try_from(node.terminal_start)?;
        let term_hi = usize::try_from(node.terminal_end)?;
        if term_lo < term_hi {
            let accepted = if frame.partial.remaining == 0 {
                !frame.char_stacks.is_empty() || frame.has_empty
            } else {
                stacks_match_partial(rules, &frame.char_stacks, frame.partial)?
            };
            if accepted {
                for &tok in &trie.dfs_tokens[term_lo..term_hi] {
                    set_bit(bitmask, usize::try_from(tok)?);
                }
            }
        }

        for child_idx in usize::try_from(node.children_start)?..usize::try_from(node.children_end)?
        {
            todo.push(ExactFrame {
                node_idx: child_idx,
                depth_bytes: frame.depth_bytes,
                partial: frame.partial,
                char_stacks: frame.char_stacks.clone(),
                has_empty: frame.has_empty,
            });
        }
    }
    Ok(())
}

fn stack_accepts_piece(
    rules: &[Vec<Element>],
    stack: &[Position],
    partial: PartialUtf8,
    token_id: u32,
    piece: &[u8],
) -> Result<bool> {
    let (points, cand_partial) = decode_utf8(piece, partial);
    let token_i32 = i32::from_ne_bytes(token_id.to_ne_bytes());
    let mut todo = vec![(stack.to_vec(), 0_usize)];

    while let Some((current, offset)) = todo.pop() {
        let Some(&position) = current.last() else {
            if offset == points.len() && cand_partial.remaining == 0 {
                return Ok(true);
            }
            continue;
        };
        let elem = instruction(rules, position)?;
        if matches!(elem.kind, 8 | 9) {
            if offset == points.len() {
                if cand_partial.remaining == 0 {
                    return Ok(true);
                }
            } else if match_token(elem, token_i32) {
                return Ok(true);
            }
            continue;
        }
        if offset == points.len() {
            if cand_partial.remaining == 0 || match_partial(rules, position, cand_partial)? {
                return Ok(true);
            }
            continue;
        }
        let point = points[offset];
        let (matched, after) = match_character(rules, position, point)?;
        if matched {
            let mut following = current[..current.len() - 1].to_vec();
            if !end(instruction(rules, after)?) {
                following.push(after);
            }
            let mut next_stacks = Vec::new();
            advance_stack_vec(rules, following, &mut next_stacks)?;
            for branch in next_stacks {
                todo.push((branch, offset + 1));
            }
        }
    }
    Ok(false)
}
