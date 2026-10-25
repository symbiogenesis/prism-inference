//! Explicit, bounded recognition of declared special tokens in borrowed text.
//!
//! This stage only separates normal text from control/user-defined vocabulary
//! spellings. It does not pretokenize normal text, apply merges or templates, or
//! insert BOS/EOS. Callers must choose whether to recognize special spellings.

use bonsai_gguf::Metadata;

use crate::vocabulary::ValidatedVocabulary;
use crate::{Result, invalid};

/// Maximum number of control and user-defined vocabulary entries.
pub const MAX_SPECIAL_TOKENS: usize = 4096;
/// Maximum aggregate UTF-8 bytes of admitted special spellings.
pub const MAX_SPECIAL_BYTES: usize = 1024 * 1024;
/// Maximum UTF-8 bytes in one special spelling.
pub const MAX_SPECIAL_SPELLING_BYTES: usize = 256;
/// Maximum UTF-8 input bytes admitted by [`SpecialInput::split`].
pub const MAX_SPECIAL_INPUT_BYTES: usize = 1024 * 1024;
/// Maximum returned text and special segments, independently of caller limits.
pub const MAX_SPECIAL_INPUT_SEGMENTS: usize = 64 * 1024;

const NO_MATCH: usize = usize::MAX;
const MAX_BUILD_TRANSITIONS: usize = 16 * 1024 * 1024;

/// Explicit interpretation of special-token spellings in caller-provided text.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpecialInputPolicy {
    /// Preserve every spelling as ordinary text for normal byte encoding.
    Literal,
    /// Recognize vocabulary entries declared control or user-defined.
    Parse,
}

/// One borrowed ordinary-text range or one explicitly recognized special ID.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InputSegment<'a> {
    /// A nonempty contiguous range of the original input, without copying.
    Text(&'a str),
    /// The vocabulary ID of a declared control or user-defined spelling.
    Special(u32),
}

#[derive(Debug)]
struct Node {
    edges: std::ops::Range<usize>,
    failure: usize,
    best: usize,
}

impl Node {
    const fn new() -> Self {
        Self {
            edges: 0..0,
            failure: 0,
            best: NO_MATCH,
        }
    }
}

#[derive(Debug)]
struct Edge {
    byte: u8,
    child: usize,
}

#[derive(Debug)]
struct Special {
    length: usize,
    token: u32,
}

#[derive(Clone, Copy)]
struct BuildRange {
    start: usize,
    end: usize,
    depth: usize,
}

/// Validated special spellings with deterministic, bounded text segmentation.
///
/// Only control (type 3) and user-defined (type 4) entries participate. A normal
/// token with the same spelling remains valid; duplicate special spellings are
/// ambiguous and rejected, including duplicates across those two types.
/// Recognition uses the earliest starting match and the longest match at that
/// start. An overlapping match beginning later never displaces an earlier one.
///
/// A sparse reverse-byte Aho-Corasick trie finds the longest match for each
/// starting byte in one reverse scan. It retains at most `1 + MAX_SPECIAL_BYTES`
/// nodes, `MAX_SPECIAL_BYTES` edges, and `MAX_SPECIAL_TOKENS` length/ID entries;
/// original metadata and spelling strings are not retained. Edge lookup uses
/// binary search over at most 256 children. Failure transitions amortize to at
/// most twice the input length, independently of the number of special tokens.
#[derive(Debug)]
pub struct SpecialInput {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    specials: Vec<Special>,
}

impl SpecialInput {
    /// Validates the pinned `gpt2` / `qwen35` vocabulary and special spellings.
    ///
    /// Uses the complete [`Vocabulary`](crate::vocabulary::Vocabulary)
    /// validation, including exact graph/token counts, token types and the normal
    /// byte alphabet. Merge metadata is neither required nor interpreted. No
    /// special IDs are inferred from token names. An otherwise valid vocabulary
    /// with no special entries is accepted.
    ///
    /// Construction bounds special counts and bytes before trie allocation.
    /// Vocabulary validation borrows metadata without allocating decoded pieces.
    /// Temporary storage comprises one borrowed entry per special and one
    /// three-index range per trie node. Trie construction also caps failure-link
    /// work at 16 Mi transitions.
    ///
    /// # Errors
    /// Rejects invalid vocabulary metadata, empty or duplicate special spellings,
    /// special resource limits, excessive construction work, or allocation errors.
    pub fn from_metadata(metadata: &Metadata, vocab_size: usize) -> Result<Self> {
        Self::from_vocabulary(&ValidatedVocabulary::from_metadata(metadata, vocab_size)?)
    }

    pub(crate) fn from_vocabulary(vocabulary: &ValidatedVocabulary<'_>) -> Result<Self> {
        let mut count = 0_usize;
        let mut bytes = 0_usize;
        for (token, kind) in vocabulary.entries() {
            if !matches!(kind, 3 | 4) {
                continue;
            }
            count += 1;
            if count > MAX_SPECIAL_TOKENS {
                return Err(invalid("special input exceeds special token count limit"));
            }
            if token.is_empty() || token.len() > MAX_SPECIAL_SPELLING_BYTES {
                return Err(invalid("special input spelling exceeds byte bounds"));
            }
            bytes = bytes
                .checked_add(token.len())
                .filter(|&length| length <= MAX_SPECIAL_BYTES)
                .ok_or_else(|| invalid("special input exceeds spelling byte limit"))?;
        }
        let mut entries = Vec::new();
        entries.try_reserve_exact(count)?;
        for (index, (token, kind)) in vocabulary.entries().enumerate() {
            if matches!(kind, 3 | 4) {
                entries.push((token.as_bytes(), u32::try_from(index)?));
            }
        }
        Self::build(entries)
    }

    fn build(mut entries: Vec<(&[u8], u32)>) -> Result<Self> {
        let count = entries.len();
        entries.sort_unstable_by(|left, right| left.0.iter().rev().cmp(right.0.iter().rev()));

        let mut node_count = 1_usize;
        let mut previous: &[u8] = &[];
        for &(spelling, _) in &entries {
            if spelling == previous {
                return Err(invalid("special input has duplicate special spellings"));
            }
            let common = spelling
                .iter()
                .rev()
                .zip(previous.iter().rev())
                .take_while(|(left, right)| left == right)
                .count();
            node_count += spelling.len() - common;
            previous = spelling;
        }
        let mut nodes = Vec::new();
        nodes.try_reserve_exact(node_count)?;
        nodes.push(Node::new());
        let mut edges = Vec::new();
        edges.try_reserve_exact(node_count - 1)?;
        let mut ranges = Vec::new();
        ranges.try_reserve_exact(node_count)?;
        ranges.push(BuildRange {
            start: 0,
            end: count,
            depth: 0,
        });
        let mut cursor = 0;
        while cursor < nodes.len() {
            let range = ranges[cursor];
            let mut start = range.start;
            if start < range.end && entries[start].0.len() == range.depth {
                nodes[cursor].best = start;
                start += 1;
            }
            let edge_start = edges.len();
            while start < range.end {
                let spelling = entries[start].0;
                let byte = spelling[spelling.len() - range.depth - 1];
                let mut end = start + 1;
                while end < range.end {
                    let next = entries[end].0;
                    if next[next.len() - range.depth - 1] != byte {
                        break;
                    }
                    end += 1;
                }
                edges.push(Edge {
                    byte,
                    child: nodes.len(),
                });
                nodes.push(Node::new());
                ranges.push(BuildRange {
                    start,
                    end,
                    depth: range.depth + 1,
                });
                start = end;
            }
            nodes[cursor].edges = edge_start..edges.len();
            cursor += 1;
        }
        drop(ranges);
        let mut specials = Vec::new();
        specials.try_reserve_exact(count)?;
        for (spelling, token) in entries {
            specials.push(Special {
                length: spelling.len(),
                token,
            });
        }
        let mut result = Self {
            nodes,
            edges,
            specials,
        };
        result.build_failure_links()?;
        Ok(result)
    }

    /// Splits text under the caller's explicit recognition policy.
    ///
    /// `Literal` returns one borrowed text segment, or no segments for empty
    /// text. `Parse` returns nonempty ordinary ranges and individual special IDs,
    /// including adjacent specials. Empty input never inserts BOS, EOS or any
    /// other ID. Output is bounded by both `max_segments` and
    /// [`MAX_SPECIAL_INPUT_SEGMENTS`], without returning partial output on error.
    ///
    /// Input is limited to [`MAX_SPECIAL_INPUT_BYTES`]. Parsing temporarily uses
    /// one machine-sized match index per input byte; literal input needs none.
    /// Output reserves at most `min(input bytes, segment limit)` entries, and
    /// borrows text directly from `text`, independently of metadata lifetime.
    ///
    /// # Errors
    /// Rejects oversized input, output above either segment bound, or allocation
    /// errors. Even literal mode enforces input and output bounds.
    pub fn split<'a>(
        &self,
        text: &'a str,
        policy: SpecialInputPolicy,
        max_segments: usize,
    ) -> Result<Vec<InputSegment<'a>>> {
        if text.len() > MAX_SPECIAL_INPUT_BYTES {
            return Err(invalid("special input exceeds input byte limit"));
        }
        let limit = max_segments.min(MAX_SPECIAL_INPUT_SEGMENTS);
        if text.is_empty() {
            return Ok(Vec::new());
        }
        if limit == 0 {
            return Err(invalid("special input exceeds segment limit"));
        }
        if policy == SpecialInputPolicy::Literal || self.specials.is_empty() {
            let mut output = Vec::new();
            output.try_reserve_exact(1)?;
            output.push(InputSegment::Text(text));
            return Ok(output);
        }
        let mut starting_specials = Vec::new();
        starting_specials.try_reserve_exact(text.len())?;
        starting_specials.resize(text.len(), NO_MATCH);
        let mut node = 0;
        for (position, &byte) in text.as_bytes().iter().enumerate().rev() {
            loop {
                if let Some(next) = self.transition(node, byte) {
                    node = next;
                    break;
                }
                if node == 0 {
                    break;
                }
                node = self.nodes[node].failure;
            }
            starting_specials[position] = self.nodes[node].best;
        }
        let mut output = Vec::new();
        output.try_reserve_exact(text.len().min(limit))?;
        let mut position = 0;
        let mut text_start = 0;
        while position < text.len() {
            let index = starting_specials[position];
            if index == NO_MATCH {
                position += 1;
                continue;
            }
            if text_start < position {
                push_segment(
                    &mut output,
                    InputSegment::Text(&text[text_start..position]),
                    limit,
                )?;
            }
            let special = &self.specials[index];
            push_segment(&mut output, InputSegment::Special(special.token), limit)?;
            // UTF-8 spellings can only match complete scalars in valid UTF-8
            // input, so both slice boundaries remain character boundaries.
            position += special.length;
            text_start = position;
        }
        if text_start < text.len() {
            push_segment(&mut output, InputSegment::Text(&text[text_start..]), limit)?;
        }
        Ok(output)
    }

    fn transition(&self, node: usize, byte: u8) -> Option<usize> {
        let edges = &self.edges[self.nodes[node].edges.clone()];
        edges
            .binary_search_by_key(&byte, |edge| edge.byte)
            .ok()
            .map(|index| edges[index].child)
    }

    fn build_failure_links(&mut self) -> Result<()> {
        let mut work = 0;
        // Nodes were created breadth-first: a failure target is always shallower
        // and already has its own failure link and inherited longest output.
        for parent in 0..self.nodes.len() {
            for edge in self.nodes[parent].edges.clone() {
                let child = self.edges[edge].child;
                let byte = self.edges[edge].byte;
                let mut failure = 0;
                if parent != 0 {
                    let mut fallback = self.nodes[parent].failure;
                    loop {
                        work += 1;
                        if work > MAX_BUILD_TRANSITIONS {
                            return Err(invalid("special input exceeds construction work limit"));
                        }
                        if let Some(next) = self.transition(fallback, byte) {
                            failure = next;
                            break;
                        }
                        if fallback == 0 {
                            break;
                        }
                        fallback = self.nodes[fallback].failure;
                    }
                }
                self.nodes[child].failure = failure;
                if self.nodes[child].best == NO_MATCH {
                    self.nodes[child].best = self.nodes[failure].best;
                }
            }
        }
        Ok(())
    }
}

fn push_segment<'a>(
    output: &mut Vec<InputSegment<'a>>,
    segment: InputSegment<'a>,
    limit: usize,
) -> Result<()> {
    if output.len() == limit {
        return Err(invalid("special input exceeds segment limit"));
    }
    output.push(segment);
    Ok(())
}
