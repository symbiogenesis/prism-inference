//! Clean-room independent safe Rust PEG arena, parser, AST, GBNF emitter, and JSON codec.
//!
//! This module implements a standalone PEG combinator builder, backtracking
//! packrat parser with streaming partial-input and UTF-8 boundary support, AST
//! capture tree, GBNF grammar generator, and arena JSON serializer/deserializer
//! directly from `tools/prism/upstream/common/peg.hpp`, `PROTOCOL.md`, and
//! `tests/upstream_prism/peg.rs`. It never imports `bonsai_prism_runtime`, invokes
//! a native bridge, or reads expected-answer fixtures.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use aho_corasick::{AhoCorasick, MatchKind};
use serde_json::{Map, Value, json};

use super::llama_grammar::{array, exact_fields, field};
use super::{Result, invalid};

/// Upstream test case IDs handled by this independent PEG provider.
pub(super) const IDS: &[&str] = &[
    "tests/peg-parser/test-basic.cpp",
    "tests/peg-parser/test-gbnf-generation.cpp",
    "tests/peg-parser/test-json-parser.cpp",
    "tests/peg-parser/test-json-serialization.cpp",
    "tests/peg-parser/test-python-dict-parser.cpp",
    "tests/peg-parser/test-unicode.cpp",
    "tests/test-peg-parser.cpp",
];

const MAX_OPERATIONS: usize = 128;
const MAX_GRAPH_DEPTH: usize = 48;
const MAX_EXPR_NODES: usize = 16_384;
const MAX_STATEMENTS: usize = 4096;
const MAX_LOCAL_NAME: usize = 128;
const MAX_STRING_BYTES: usize = 65_536;
const MAX_ARGS: usize = 64;
const MAX_LIST_LEN: usize = 1024;
const MAX_GRAMMAR_MODES: usize = 8;
const MAX_SERIALIZED_JSON_BYTES: usize = 8 * 1024 * 1024;
const MAX_PARSE_DEPTH: usize = 2048;

/// Identifier of a parser node inside an [`Arena`] or [`ParserBuilder`].
pub(super) type ParserId = usize;

/// Identifier of an [`AstNode`] inside an [`AstArena`].
pub(super) type AstId = usize;

/// Sentinel parser ID representing an unset or invalid parser reference.
pub(super) const INVALID_PARSER_ID: ParserId = usize::MAX;

/// Bit flags controlling PEG parse behavior.
pub(super) mod parse_flags {
    /// Default complete-input parsing mode.
    pub(in super::super) const NONE: u8 = 0;
    /// Partial/streaming input mode (`NEED_MORE_INPUT` allowed at input horizon).
    #[allow(
        dead_code,
        reason = "Exposed for Stage 2-4 chat PEG/auto/chat providers"
    )]
    pub(in super::super) const IS_PARTIAL: u8 = 1 << 0;
}

/// Outcome category of a PEG parse attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ParseResultType {
    /// Input failed to match at the starting offset.
    Fail = 0,
    /// Input matched completely over `[start..end]`.
    Success = 1,
    /// Input matched a valid partial prefix and requires additional bytes.
    NeedMoreInput = 2,
}

impl ParseResultType {
    /// Returns the numeric wire code (`0` = Fail, `1` = Success, `2` = `NeedMoreInput`).
    pub(super) const fn as_u8(self) -> u8 {
        match self {
            Self::Fail => 0,
            Self::Success => 1,
            Self::NeedMoreInput => 2,
        }
    }
}

/// Result of parsing with a PEG node or [`Arena`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ParseResult {
    /// Match outcome (`Fail`, `Success`, or `NeedMoreInput`).
    pub(super) result_type: ParseResultType,
    /// Starting byte offset in the input buffer.
    pub(super) start: usize,
    /// Ending byte offset of the matched/safe prefix in the input buffer.
    pub(super) end: usize,
    /// Top-level captured AST node IDs produced during this match.
    pub(super) nodes: Vec<AstId>,
}

impl ParseResult {
    const fn fail(start: usize) -> Self {
        Self {
            result_type: ParseResultType::Fail,
            start,
            end: start,
            nodes: Vec::new(),
        }
    }

    const fn success(start: usize, end: usize, nodes: Vec<AstId>) -> Self {
        Self {
            result_type: ParseResultType::Success,
            start,
            end,
            nodes,
        }
    }

    const fn need_more_input(start: usize, end: usize, nodes: Vec<AstId>) -> Self {
        Self {
            result_type: ParseResultType::NeedMoreInput,
            start,
            end,
            nodes,
        }
    }

    const fn to_wire(&self) -> (u8, usize, usize) {
        (self.result_type.as_u8(), self.start, self.end)
    }
}

/// Captured AST node produced by `rule` and `tag` combinators.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct AstNode {
    /// Node identifier within its owning [`AstArena`].
    pub(super) id: AstId,
    /// Rule name if captured by `rule`, otherwise empty.
    pub(super) rule: String,
    /// Tag name if captured by `tag`, otherwise empty.
    pub(super) tag: String,
    /// Starting byte offset of the captured span.
    pub(super) start: usize,
    /// Ending byte offset of the captured span.
    pub(super) end: usize,
    /// Raw matched text (lossy-decoded if partial non-UTF-8).
    pub(super) text: String,
    /// Child AST node identifiers in match order.
    pub(super) children: Vec<AstId>,
}

/// Arena storing captured [`AstNode`] entries for one parse invocation.
#[derive(Clone, Debug, Default)]
pub(super) struct AstArena {
    nodes: Vec<AstNode>,
}

impl AstArena {
    fn add_node(
        &mut self,
        rule: String,
        tag: String,
        start: usize,
        end: usize,
        input: &[u8],
        children: Vec<AstId>,
    ) -> AstId {
        let id = self.nodes.len();
        let text = input
            .get(start..end)
            .map_or_default(|slice| String::from_utf8_lossy(slice).into_owned());
        self.nodes.push(AstNode {
            id,
            rule,
            tag,
            start,
            end,
            text,
            children,
        });
        id
    }

    /// Returns a captured AST node by ID.
    #[allow(
        dead_code,
        reason = "Exposed for Stage 2-4 chat PEG/auto/chat providers"
    )]
    pub(super) fn get(&self, id: AstId) -> Option<&AstNode> {
        self.nodes.get(id)
    }

    /// Returns all captured AST nodes in allocation order.
    #[allow(
        dead_code,
        reason = "Exposed for Stage 2-4 chat PEG/auto/chat providers"
    )]
    pub(super) fn nodes(&self) -> &[AstNode] {
        &self.nodes
    }

    /// Traverses captured AST subtrees recursively in pre-order.
    #[allow(
        dead_code,
        reason = "Exposed for Stage 2-4 chat PEG/auto/chat providers"
    )]
    pub(super) fn visit(&self, ids: &[AstId], visitor: &mut impl FnMut(&AstNode)) {
        for &id in ids {
            if let Some(node) = self.nodes.get(id) {
                visitor(node);
                self.visit(&node.children, visitor);
            }
        }
    }
}

/// Mutable parse context holding input bytes, flags, and captured AST nodes.
#[derive(Clone, Debug)]
pub(super) struct ParseContext<'a> {
    /// Input byte slice being parsed.
    pub(super) input: &'a [u8],
    /// Bitmask of [`parse_flags`].
    pub(super) flags: u8,
    /// Arena of AST nodes captured during parsing.
    pub(super) ast: AstArena,
}

impl<'a> ParseContext<'a> {
    /// Creates a fresh parse context for `input` and `flags`.
    pub(super) fn new(input: &'a [u8], flags: u8) -> Self {
        Self {
            input,
            flags,
            ast: AstArena::default(),
        }
    }

    /// Returns true if [`parse_flags::IS_PARTIAL`] is enabled.
    pub(super) const fn is_partial(&self) -> bool {
        (self.flags & parse_flags::IS_PARTIAL) != 0
    }
}

// ---------------------------------------------------------------------------
// UTF-8 decoding and trailing incomplete boundary inspection
// ---------------------------------------------------------------------------

enum Utf8Decode {
    Char(u32, usize),
    Incomplete,
    Invalid,
}

fn decode_utf8(bytes: &[u8]) -> Utf8Decode {
    let Some(&b0) = bytes.first() else {
        return Utf8Decode::Incomplete;
    };
    match b0 {
        0x00..=0x7F => Utf8Decode::Char(u32::from(b0), 1),
        0xC2..=0xDF => {
            let Some(&b1) = bytes.get(1) else {
                return Utf8Decode::Incomplete;
            };
            if (b1 & 0xC0) != 0x80 {
                return Utf8Decode::Invalid;
            }
            let cp = (u32::from(b0 & 0x1F) << 6) | u32::from(b1 & 0x3F);
            Utf8Decode::Char(cp, 2)
        }
        0xE0..=0xEF => {
            if let Some(&b1) = bytes.get(1) {
                if (b1 & 0xC0) != 0x80 || (b0 == 0xE0 && b1 < 0xA0) || (b0 == 0xED && b1 >= 0xA0) {
                    return Utf8Decode::Invalid;
                }
            } else {
                return Utf8Decode::Incomplete;
            }
            let Some(&b2) = bytes.get(2) else {
                return Utf8Decode::Incomplete;
            };
            if (b2 & 0xC0) != 0x80 {
                return Utf8Decode::Invalid;
            }
            let cp = (u32::from(b0 & 0x0F) << 12)
                | (u32::from(bytes[1] & 0x3F) << 6)
                | u32::from(b2 & 0x3F);
            Utf8Decode::Char(cp, 3)
        }
        0xF0..=0xF4 => {
            if let Some(&b1) = bytes.get(1) {
                if (b1 & 0xC0) != 0x80 || (b0 == 0xF0 && b1 < 0x90) || (b0 == 0xF4 && b1 >= 0x90) {
                    return Utf8Decode::Invalid;
                }
            } else {
                return Utf8Decode::Incomplete;
            }
            if let Some(&b2) = bytes.get(2) {
                if (b2 & 0xC0) != 0x80 {
                    return Utf8Decode::Invalid;
                }
            } else {
                return Utf8Decode::Incomplete;
            }
            let Some(&b3) = bytes.get(3) else {
                return Utf8Decode::Incomplete;
            };
            if (b3 & 0xC0) != 0x80 {
                return Utf8Decode::Invalid;
            }
            let cp = (u32::from(b0 & 0x07) << 18)
                | (u32::from(bytes[1] & 0x3F) << 12)
                | (u32::from(bytes[2] & 0x3F) << 6)
                | u32::from(b3 & 0x3F);
            Utf8Decode::Char(cp, 4)
        }
        _ => Utf8Decode::Invalid,
    }
}

fn trailing_incomplete_utf8_len(slice: &[u8]) -> usize {
    let check = slice.len().min(3);
    for back in 1..=check {
        let start = slice.len() - back;
        if matches!(decode_utf8(&slice[start..]), Utf8Decode::Incomplete) {
            return back;
        }
    }
    0
}

// ---------------------------------------------------------------------------
// Character class parser (`chars`)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
struct CharsSpec {
    pattern: String,
    negated: bool,
    ranges: Vec<(u32, u32)>,
    min: usize,
    max: Option<usize>,
}

fn strip_outer_brackets(pattern: &str) -> &str {
    let bytes = pattern.as_bytes();
    if bytes.len() >= 2 && bytes[0] == b'[' && bytes[bytes.len() - 1] == b']' {
        let mut idx = 1;
        while idx < bytes.len() {
            if bytes[idx] == b'\\' {
                idx += 2;
                continue;
            }
            if bytes[idx] == b']' {
                if idx == bytes.len() - 1 {
                    return &pattern[1..pattern.len() - 1];
                }
                return pattern;
            }
            idx += 1;
        }
    }
    pattern
}

fn parse_hex_digits(
    chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
    count: usize,
) -> Result<u32> {
    let mut value = 0_u32;
    for _ in 0..count {
        let digit = chars
            .next()
            .and_then(|c| c.to_digit(16))
            .ok_or_else(|| invalid("invalid hex escape in PEG character class"))?;
        value = value
            .checked_mul(16)
            .and_then(|acc| acc.checked_add(digit))
            .ok_or_else(|| invalid("PEG character class escape overflow"))?;
    }
    Ok(value)
}

fn parse_class_escape(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> Result<u32> {
    let escaped = chars
        .next()
        .ok_or_else(|| invalid("unterminated escape in PEG character class"))?;
    match escaped {
        'n' => Ok(u32::from(b'\n')),
        'r' => Ok(u32::from(b'\r')),
        't' => Ok(u32::from(b'\t')),
        'b' => Ok(8),
        'f' => Ok(12),
        'v' => Ok(11),
        '0' => Ok(0),
        'x' => parse_hex_digits(chars, 2),
        'u' => parse_hex_digits(chars, 4),
        'U' => parse_hex_digits(chars, 8),
        other => Ok(u32::from(other)),
    }
}

impl CharsSpec {
    fn parse(pattern: &str, min: i32, max: i32) -> Result<Self> {
        let min_usize =
            usize::try_from(min).map_err(|_| invalid("PEG chars min count must be nonnegative"))?;
        let max_usize = if max < 0 {
            None
        } else {
            let val = usize::try_from(max)?;
            if val < min_usize {
                return Err(invalid("PEG chars max count cannot be less than min count"));
            }
            Some(val)
        };
        let stripped = strip_outer_brackets(pattern);
        let (negated, body) = stripped
            .strip_prefix('^')
            .map_or((false, stripped), |rest| (true, rest));
        let mut chars = body.chars().peekable();
        let mut ranges = Vec::new();
        while let Some(ch) = chars.next() {
            let start = if ch == '\\' {
                parse_class_escape(&mut chars)?
            } else {
                u32::from(ch)
            };
            if chars.peek() == Some(&'-') {
                let mut lookahead = chars.clone();
                lookahead.next();
                if lookahead.peek().is_some() {
                    chars.next();
                    let next = chars
                        .next()
                        .ok_or_else(|| invalid("missing range end in PEG character class"))?;
                    let end = if next == '\\' {
                        parse_class_escape(&mut chars)?
                    } else {
                        u32::from(next)
                    };
                    if start > end {
                        return Err(invalid("reversed range in PEG character class"));
                    }
                    ranges.push((start, end));
                    continue;
                }
            }
            ranges.push((start, start));
        }
        Ok(Self {
            pattern: pattern.to_owned(),
            negated,
            ranges,
            min: min_usize,
            max: max_usize,
        })
    }

    fn matches(&self, cp: u32) -> bool {
        let contained = self.ranges.iter().any(|&(lo, hi)| lo <= cp && cp <= hi);
        if self.negated { !contained } else { contained }
    }

    fn class_atom_gbnf(&self) -> String {
        let mut out = String::from("[");
        let body = strip_outer_brackets(&self.pattern);
        let mut chars = body.chars();
        while let Some(ch) = chars.next() {
            match ch {
                '\\' => {
                    out.push('\\');
                    if let Some(next) = chars.next() {
                        out.push(next);
                    }
                }
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if u32::from(c) < 0x20 => {
                    let _ = write!(out, "\\x{:02x}", u32::from(c));
                }
                c => out.push(c),
            }
        }
        out.push(']');
        out
    }

    fn to_gbnf(&self) -> String {
        if self.max == Some(0) {
            return String::new();
        }
        let atom = self.class_atom_gbnf();
        match (self.min, self.max) {
            (1, Some(1)) => atom,
            (0, Some(1)) => format!("{atom}?"),
            (0, None) => format!("{atom}*"),
            (1, None) => format!("{atom}+"),
            (lo, Some(hi)) if lo == hi => format!("{atom}{{{lo}}}"),
            (lo, None) => format!("{atom}{{{lo},}}"),
            (lo, Some(hi)) => format!("{atom}{{{lo},{hi}}}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Parser arena nodes and builder
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum ParserNode {
    Eps,
    Start,
    End,
    Any,
    Space,
    Rest,
    Marker,
    Literal(String),
    Chars(CharsSpec),
    StringContent(u8),
    JsonStringContent,
    Sequence(Vec<ParserId>),
    Choice(Vec<ParserId>),
    Repeat {
        child: ParserId,
        min: usize,
        max: Option<usize>,
    },
    Until(Vec<String>),
    Ac {
        child: ParserId,
        delimiters: Vec<String>,
    },
    Atomic(ParserId),
    Peek(ParserId),
    Negate(ParserId),
    Tag {
        tag: String,
        child: ParserId,
    },
    Rule {
        name: String,
        child: ParserId,
        trigger: bool,
    },
    Ref(String),
    Gbnf {
        child: ParserId,
        grammar: String,
    },
}

/// Clean-room PEG combinator graph builder producing an immutable [`Arena`].
#[derive(Clone, Debug, Default)]
pub(super) struct ParserBuilder {
    parsers: Vec<ParserNode>,
    rules: BTreeMap<String, ParserId>,
    root: Option<ParserId>,
}

impl ParserBuilder {
    /// Creates an empty PEG parser builder.
    pub(super) fn new() -> Self {
        Self::default()
    }

    fn push(&mut self, node: ParserNode) -> ParserId {
        let id = self.parsers.len();
        self.parsers.push(node);
        id
    }

    /// Sets the root parser of the arena.
    pub(super) const fn set_root(&mut self, id: ParserId) {
        self.root = Some(id);
    }

    /// Empty match parser (`eps`).
    pub(super) fn eps(&mut self) -> ParserId {
        self.push(ParserNode::Eps)
    }

    /// Start-of-input assertion parser (`start`).
    #[allow(
        dead_code,
        reason = "Exposed for Stage 2-4 chat PEG/auto/chat providers"
    )]
    pub(super) fn start(&mut self) -> ParserId {
        self.push(ParserNode::Start)
    }

    /// End-of-input assertion parser (`end`).
    pub(super) fn end(&mut self) -> ParserId {
        self.push(ParserNode::End)
    }

    /// Single UTF-8 codepoint wildcard parser (`any`).
    pub(super) fn any(&mut self) -> ParserId {
        self.push(ParserNode::Any)
    }

    /// Zero-or-more ASCII whitespace parser (`space`).
    pub(super) fn space(&mut self) -> ParserId {
        self.push(ParserNode::Space)
    }

    /// Consumes all remaining input up to EOF (`rest`).
    #[allow(
        dead_code,
        reason = "Exposed for Stage 2-4 chat PEG/auto/chat providers"
    )]
    pub(super) fn rest(&mut self) -> ParserId {
        self.push(ParserNode::Rest)
    }

    /// Zero-width marker node (`marker`).
    pub(super) fn marker(&mut self) -> ParserId {
        self.push(ParserNode::Marker)
    }

    /// Exact byte-string literal parser (`literal`).
    pub(super) fn literal(&mut self, text: impl Into<String>) -> ParserId {
        self.push(ParserNode::Literal(text.into()))
    }

    /// UTF-8 character class parser (`chars`).
    pub(super) fn chars(&mut self, pattern: &str, min: i32, max: i32) -> Result<ParserId> {
        let spec = CharsSpec::parse(pattern, min, max)?;
        Ok(self.push(ParserNode::Chars(spec)))
    }

    /// String content parser stopping before unescaped `quote` (`string_content`).
    pub(super) fn string_content(&mut self, quote: u8) -> ParserId {
        self.push(ParserNode::StringContent(quote))
    }

    /// Ordered sequence combinator (`sequence`).
    pub(super) fn sequence(&mut self, children: Vec<ParserId>) -> ParserId {
        self.push(ParserNode::Sequence(children))
    }

    /// Ordered PEG choice combinator (`choice`).
    pub(super) fn choice(&mut self, children: Vec<ParserId>) -> ParserId {
        self.push(ParserNode::Choice(children))
    }

    /// Concatenates two parsers (`+` / `add`).
    pub(super) fn add(&mut self, left: ParserId, right: ParserId) -> ParserId {
        self.sequence(vec![left, right])
    }

    /// Alternates two parsers (`|` / `or`).
    pub(super) fn or(&mut self, left: ParserId, right: ParserId) -> ParserId {
        self.choice(vec![left, right])
    }

    /// Appends `right` to `left` as an alternative if `left` is a `Choice`, or wraps both (`|=`).
    #[allow(
        dead_code,
        reason = "Exposed for Stage 2-4 chat PEG/auto/chat providers"
    )]
    pub(super) fn or_assign(&mut self, left: ParserId, right: ParserId) -> ParserId {
        if let Some(ParserNode::Choice(items)) = self.parsers.get_mut(left) {
            items.push(right);
            return left;
        }
        if let Some(existing) = self.parsers.get(left).cloned() {
            let cloned_left = self.push(existing);
            if let Some(slot) = self.parsers.get_mut(left) {
                *slot = ParserNode::Choice(vec![cloned_left, right]);
            }
            return left;
        }
        self.or(left, right)
    }

    /// Concatenates two parsers separated by optional `space` (`<<` / `space-add`).
    pub(super) fn space_add(&mut self, left: ParserId, right: ParserId) -> ParserId {
        let sp = self.space();
        self.sequence(vec![left, sp, right])
    }

    /// One-or-more repetition combinator (`one_or_more`).
    pub(super) fn one_or_more(&mut self, child: ParserId) -> ParserId {
        self.push(ParserNode::Repeat {
            child,
            min: 1,
            max: None,
        })
    }

    /// Zero-or-more repetition combinator (`zero_or_more`).
    pub(super) fn zero_or_more(&mut self, child: ParserId) -> ParserId {
        self.push(ParserNode::Repeat {
            child,
            min: 0,
            max: None,
        })
    }

    /// Optional combinator (`optional`).
    pub(super) fn optional(&mut self, child: ParserId) -> ParserId {
        self.push(ParserNode::Repeat {
            child,
            min: 0,
            max: Some(1),
        })
    }

    /// Bounded repetition combinator (`repeat`).
    #[allow(
        dead_code,
        reason = "Exposed for Stage 2-4 chat PEG/auto/chat providers"
    )]
    pub(super) fn repeat(&mut self, child: ParserId, min: i32, max: i32) -> Result<ParserId> {
        let min_usize = usize::try_from(min)
            .map_err(|_| invalid("PEG repeat min count must be nonnegative"))?;
        let max_usize = if max < 0 {
            None
        } else {
            let val = usize::try_from(max)?;
            if val < min_usize {
                return Err(invalid("PEG repeat max count cannot be less than min"));
            }
            Some(val)
        };
        Ok(self.push(ParserNode::Repeat {
            child,
            min: min_usize,
            max: max_usize,
        }))
    }

    /// Matches characters up to (but not consuming) `delimiter` (`until`).
    pub(super) fn until(&mut self, delimiter: impl Into<String>) -> ParserId {
        self.push(ParserNode::Until(vec![delimiter.into()]))
    }

    /// Matches characters up to (but not consuming) the earliest delimiter (`until_one_of`).
    pub(super) fn until_one_of(&mut self, delimiters: Vec<String>) -> ParserId {
        self.push(ParserNode::Until(delimiters))
    }

    /// Runs `child` bounded before the earliest delimiter (`ac`).
    pub(super) fn ac(&mut self, child: ParserId, delimiters: Vec<String>) -> ParserId {
        self.push(ParserNode::Ac { child, delimiters })
    }

    /// Atomic match wrapper suppressing partial AST captures (`atomic`).
    pub(super) fn atomic(&mut self, child: ParserId) -> ParserId {
        self.push(ParserNode::Atomic(child))
    }

    /// Positive lookahead combinator (`peek`).
    #[allow(
        dead_code,
        reason = "Exposed for Stage 2-4 chat PEG/auto/chat providers"
    )]
    pub(super) fn peek(&mut self, child: ParserId) -> ParserId {
        self.push(ParserNode::Peek(child))
    }

    /// Negative lookahead combinator (`negate`).
    pub(super) fn negate(&mut self, child: ParserId) -> ParserId {
        self.push(ParserNode::Negate(child))
    }

    /// Tags matched span with an AST node (`tag`).
    pub(super) fn tag(&mut self, tag: impl Into<String>, child: ParserId) -> ParserId {
        self.push(ParserNode::Tag {
            tag: tag.into(),
            child,
        })
    }

    /// Defines a named rule (`rule`), optionally marked as a lazy grammar trigger.
    pub(super) fn rule(
        &mut self,
        name: impl Into<String>,
        child: ParserId,
        trigger: bool,
    ) -> ParserId {
        let name = name.into();
        let id = self.push(ParserNode::Rule {
            name: name.clone(),
            child,
            trigger,
        });
        self.rules.insert(name, id);
        id
    }

    /// Reserves a placeholder rule node and registers it before building its body.
    pub(super) fn begin_rule_lazy(&mut self, name: impl Into<String>, trigger: bool) -> ParserId {
        let name = name.into();
        let placeholder = self.push(ParserNode::Rule {
            name: name.clone(),
            child: INVALID_PARSER_ID,
            trigger,
        });
        self.rules.insert(name, placeholder);
        placeholder
    }

    /// Completes a placeholder rule reserved by [`Self::begin_rule_lazy`].
    pub(super) fn finish_rule_lazy(
        &mut self,
        placeholder: ParserId,
        name: impl Into<String>,
        child: ParserId,
        trigger: bool,
    ) {
        if let Some(slot) = self.parsers.get_mut(placeholder) {
            *slot = ParserNode::Rule {
                name: name.into(),
                child,
                trigger,
            };
        }
    }

    /// Defines a named rule lazily via callback (`rule-lazy`).
    pub(super) fn rule_lazy<F>(
        &mut self,
        name: impl Into<String>,
        trigger: bool,
        build_body: F,
    ) -> Result<ParserId>
    where
        F: FnOnce(&mut Self) -> Result<ParserId>,
    {
        let name = name.into();
        let placeholder = self.begin_rule_lazy(name.clone(), trigger);
        let child = build_body(self)?;
        self.finish_rule_lazy(placeholder, name, child, trigger);
        Ok(placeholder)
    }

    /// Forward or recursive reference to a named rule (`ref`).
    pub(super) fn ref_rule(&mut self, name: impl Into<String>) -> ParserId {
        self.push(ParserNode::Ref(name.into()))
    }

    /// Wraps `child` with a custom GBNF grammar string (`gbnf`).
    pub(super) fn gbnf(&mut self, child: ParserId, grammar: impl Into<String>) -> ParserId {
        self.push(ParserNode::Gbnf {
            child,
            grammar: grammar.into(),
        })
    }

    /// Built-in double-quoted string parser (`double_quoted_string`).
    pub(super) fn double_quoted_string(&mut self) -> ParserId {
        let open = self.literal("\"");
        let content = self.string_content(b'"');
        let close = self.literal("\"");
        self.sequence(vec![open, content, close])
    }

    /// Built-in single-quoted string parser (`single_quoted_string`).
    pub(super) fn single_quoted_string(&mut self) -> ParserId {
        let open = self.literal("'");
        let content = self.string_content(b'\'');
        let close = self.literal("'");
        self.sequence(vec![open, content, close])
    }

    /// Built-in JSON member `"<escaped_key>": <value>` parser (`json_member`).
    pub(super) fn json_member(&mut self, key: &str, value_parser: ParserId) -> ParserId {
        let quoted_key = serde_json::to_string(key).unwrap_or_else(|_| format!("\"{key}\""));
        let key_lit = self.literal(quoted_key);
        let colon = self.literal(":");
        let left = self.space_add(key_lit, colon);
        self.space_add(left, value_parser)
    }

    /// Defines a named rule lazily only if `name` has not already been registered.
    fn rule_if_absent<F>(&mut self, name: &str, trigger: bool, build_body: F) -> Result<ParserId>
    where
        F: FnOnce(&mut Self) -> Result<ParserId>,
    {
        if let Some(&existing) = self.rules.get(name) {
            return Ok(existing);
        }
        self.rule_lazy(name, trigger, build_body)
    }

    /// Built-in RFC 8259 JSON parser (`json`).
    pub(super) fn json(&mut self) -> Result<ParserId> {
        self.rule_if_absent("json-value", false, |b| {
            let json_null = b.rule_if_absent("json-null", false, |b| Ok(b.literal("null")))?;
            let json_bool = b.rule_if_absent("json-bool", false, |b| {
                let lit_true = b.literal("true");
                let lit_false = b.literal("false");
                Ok(b.or(lit_true, lit_false))
            })?;
            let json_number = b.rule_if_absent("json-number", false, |b| {
                let minus = b.literal("-");
                let opt_minus = b.optional(minus);
                let zero = b.literal("0");
                let digit = b.chars("[0-9]", 1, 1)?;
                let not_digit = b.negate(digit);
                let zero_alone = b.add(zero, not_digit);
                let nonzero = b.chars("[1-9]", 1, 1)?;
                let more_digits = b.chars("[0-9]", 0, -1)?;
                let pos_int = b.add(nonzero, more_digits);
                let int_part = b.or(zero_alone, pos_int);
                let dot = b.literal(".");
                let frac_digits = b.chars("[0-9]", 1, -1)?;
                let frac_seq = b.add(dot, frac_digits);
                let opt_frac = b.optional(frac_seq);
                let exp_e = b.chars("[eE]", 1, 1)?;
                let exp_sign = b.chars("[+-]", 1, 1)?;
                let opt_sign = b.optional(exp_sign);
                let exp_digits = b.chars("[0-9]", 1, -1)?;
                let exp_seq = b.sequence(vec![exp_e, opt_sign, exp_digits]);
                let opt_exp = b.optional(exp_seq);
                Ok(b.sequence(vec![opt_minus, int_part, opt_frac, opt_exp]))
            })?;
            let json_string = b.rule_if_absent("json-string", false, |b| {
                let str_chars = b.push(ParserNode::JsonStringContent);
                let q1 = b.literal("\"");
                let q2 = b.literal("\"");
                Ok(b.sequence(vec![q1, str_chars, q2]))
            })?;
            let json_array = b.rule_if_absent("json-array", false, |b| {
                let value_ref_1 = b.ref_rule("json-value");
                let comma_1 = b.literal(",");
                let array_tail_step = b.space_add(comma_1, value_ref_1);
                let array_tail = b.zero_or_more(array_tail_step);
                let value_ref_2 = b.ref_rule("json-value");
                let array_items = b.add(value_ref_2, array_tail);
                let opt_array_items = b.optional(array_items);
                let lbrack = b.literal("[");
                let rbrack = b.literal("]");
                let array_inner = b.space_add(lbrack, opt_array_items);
                Ok(b.space_add(array_inner, rbrack))
            })?;
            let json_member = b.rule_if_absent("json-member", false, |b| {
                let key_ref_1 = b.ref_rule("json-string");
                let colon_1 = b.literal(":");
                let val_ref_3 = b.ref_rule("json-value");
                let kv_left_1 = b.space_add(key_ref_1, colon_1);
                Ok(b.space_add(kv_left_1, val_ref_3))
            })?;
            let json_object = b.rule_if_absent("json-object", false, |b| {
                let comma_2 = b.literal(",");
                let member_ref_1 = b.ref_rule("json-member");
                let obj_tail_step = b.space_add(comma_2, member_ref_1);
                let obj_tail = b.zero_or_more(obj_tail_step);
                let obj_items = b.add(json_member, obj_tail);
                let opt_obj_items = b.optional(obj_items);
                let lbrace = b.literal("{");
                let rbrace = b.literal("}");
                let obj_inner = b.space_add(lbrace, opt_obj_items);
                Ok(b.space_add(obj_inner, rbrace))
            })?;
            Ok(b.choice(vec![
                json_object,
                json_array,
                json_string,
                json_number,
                json_bool,
                json_null,
            ]))
        })
    }

    fn build_python_string_rule(&mut self) -> Result<ParserId> {
        let triple_dbl_open = self.literal("\"\"\"");
        let triple_dbl_body = self.build_triple_string_body("\"\"\"");
        let triple_dbl_close = self.literal("\"\"\"");
        let triple_dbl = self.sequence(vec![triple_dbl_open, triple_dbl_body, triple_dbl_close]);

        let triple_sgl_open = self.literal("'''");
        let triple_sgl_body = self.build_triple_string_body("'''");
        let triple_sgl_close = self.literal("'''");
        let triple_sgl = self.sequence(vec![triple_sgl_open, triple_sgl_body, triple_sgl_close]);

        let single_dbl = self.build_python_single_line_string("\"")?;
        let single_sgl = self.build_python_single_line_string("'")?;

        let str_choice = self.choice(vec![triple_dbl, triple_sgl, single_dbl, single_sgl]);
        Ok(self.rule("python-string", str_choice, false))
    }

    fn build_triple_string_body(&mut self, delim: &str) -> ParserId {
        let bs = self.literal("\\");
        let any_esc = self.any();
        let esc = self.add(bs, any_esc);
        let delim_lit = self.literal(delim);
        let not_delim = self.negate(delim_lit);
        let any_ch = self.any();
        let normal = self.add(not_delim, any_ch);
        let item = self.or(esc, normal);
        self.zero_or_more(item)
    }

    fn build_python_single_line_string(&mut self, quote: &str) -> Result<ParserId> {
        let class = if quote == "\"" {
            "[^\"\\\\\\n\\r]"
        } else {
            "[^'\\\\\\n\\r]"
        };
        let normal = self.chars(class, 1, 1)?;
        let bs = self.literal("\\");
        let any_esc = self.chars("[^\\n\\r]", 1, 1)?;
        let esc = self.add(bs, any_esc);
        let item = self.or(normal, esc);
        let body = self.zero_or_more(item);
        let open = self.literal(quote);
        let close = self.literal(quote);
        Ok(self.sequence(vec![open, body, close]))
    }

    fn build_python_exponent(&mut self) -> Result<ParserId> {
        let marker = self.chars("[eE]", 1, 1)?;
        let sign = self.chars("[+-]", 1, 1)?;
        let opt_sign = self.optional(sign);
        let digits = self.chars("[0-9]", 1, -1)?;
        Ok(self.sequence(vec![marker, opt_sign, digits]))
    }

    fn build_python_number_rule(&mut self) -> Result<ParserId> {
        let sign = self.chars("[+-]", 1, 1)?;
        let opt_sign = self.optional(sign);

        // Float with dot: [0-9]+ "." [0-9]* exp? | "." [0-9]+ exp?
        let exp_1 = self.build_python_exponent()?;
        let opt_exp_1 = self.optional(exp_1);
        let d_lead = self.chars("[0-9]", 1, -1)?;
        let dot_1 = self.literal(".");
        let d_frac = self.chars("[0-9]", 0, -1)?;
        let float_1 = self.sequence(vec![d_lead, dot_1, d_frac, opt_exp_1]);

        let dot_2 = self.literal(".");
        let d_frac_2 = self.chars("[0-9]", 1, -1)?;
        let exp_2 = self.build_python_exponent()?;
        let opt_exp_2 = self.optional(exp_2);
        let float_2 = self.sequence(vec![dot_2, d_frac_2, opt_exp_2]);

        let int_digits = self.chars("[0-9]", 1, -1)?;
        let exp_3 = self.build_python_exponent()?;
        let float_3 = self.sequence(vec![int_digits, exp_3]);

        // Hex/octal/binary/decimal integers.
        let hex_pfx = self.chars("[0]", 1, 1)?;
        let hex_x = self.chars("[xX]", 1, 1)?;
        let hex_d = self.chars("[0-9a-fA-F_]", 1, -1)?;
        let hex_int = self.sequence(vec![hex_pfx, hex_x, hex_d]);

        let oct_pfx = self.chars("[0]", 1, 1)?;
        let oct_o = self.chars("[oO]", 1, 1)?;
        let oct_d = self.chars("[0-7_]", 1, -1)?;
        let oct_int = self.sequence(vec![oct_pfx, oct_o, oct_d]);

        let bin_pfx = self.chars("[0]", 1, 1)?;
        let bin_b = self.chars("[bB]", 1, 1)?;
        let bin_d = self.chars("[01_]", 1, -1)?;
        let bin_int = self.sequence(vec![bin_pfx, bin_b, bin_d]);

        let zero = self.literal("0");
        let digit = self.chars("[0-9]", 1, 1)?;
        let not_digit = self.negate(digit);
        let zero_int = self.add(zero, not_digit);
        let nz_head = self.chars("[1-9]", 1, 1)?;
        let nz_tail = self.chars("[0-9_]", 0, -1)?;
        let dec_int = self.add(nz_head, nz_tail);

        let num_choice = self.choice(vec![
            float_1, float_2, float_3, hex_int, oct_int, bin_int, dec_int, zero_int,
        ]);
        let full_num = self.add(opt_sign, num_choice);
        Ok(self.rule("python-number", full_num, false))
    }

    /// Built-in Python literal value parser (`python_value`).
    pub(super) fn python_value(&mut self) -> Result<ParserId> {
        if let Some(&existing) = self.rules.get("python-value") {
            return Ok(existing);
        }
        let py_true = self.literal("True");
        let py_false = self.literal("False");
        let py_bool = self.or(py_true, py_false);
        let py_none = self.literal("None");
        let py_num = self.build_python_number_rule()?;
        let py_str = self.build_python_string_rule()?;

        // Python list: `[` space (val (space `,` space val)* (space `,`)?)? space `]`
        let list_val_1 = self.ref_rule("python-value");
        let list_comma = self.literal(",");
        let list_tail_step = self.space_add(list_comma, list_val_1);
        let list_tail = self.zero_or_more(list_tail_step);
        let list_val_2 = self.ref_rule("python-value");
        let opt_list_comma = self.literal(",");
        let opt_list_trailing = self.optional(opt_list_comma);
        let sp_list = self.space();
        let list_trailing = self.add(sp_list, opt_list_trailing);
        let list_items = self.sequence(vec![list_val_2, list_tail, list_trailing]);
        let opt_list_items = self.optional(list_items);
        let lbrack = self.literal("[");
        let rbrack = self.literal("]");
        let list_left = self.space_add(lbrack, opt_list_items);
        let list_body = self.space_add(list_left, rbrack);
        let py_list = self.rule("python-list", list_body, false);

        // Python tuple: `(` space (val (space `,` space val)* (space `,`)?)? space `)`
        let tup_val_1 = self.ref_rule("python-value");
        let tup_comma = self.literal(",");
        let tup_tail_step = self.space_add(tup_comma, tup_val_1);
        let tup_tail = self.zero_or_more(tup_tail_step);
        let tup_val_2 = self.ref_rule("python-value");
        let tup_trailing_comma = self.literal(",");
        let opt_tup_trailing = self.optional(tup_trailing_comma);
        let sp_tup = self.space();
        let tup_trailing = self.add(sp_tup, opt_tup_trailing);
        let tup_items = self.sequence(vec![tup_val_2, tup_tail, tup_trailing]);
        let opt_tup_items = self.optional(tup_items);
        let lparen = self.literal("(");
        let rparen = self.literal(")");
        let tup_left = self.space_add(lparen, opt_tup_items);
        let tup_body = self.space_add(tup_left, rparen);
        let py_tuple = self.rule("python-tuple", tup_body, false);

        // Python dict: `{` space (kv (space `,` space kv)* (space `,`)?)? space `}`
        let dict_key = self.ref_rule("python-value");
        let dict_colon = self.literal(":");
        let dict_val = self.ref_rule("python-value");
        let dict_key_colon = self.space_add(dict_key, dict_colon);
        let dict_pair = self.space_add(dict_key_colon, dict_val);
        let dict_entry = self.rule("python-dict-entry", dict_pair, false);
        let dict_comma = self.literal(",");
        let entry_ref_1 = self.ref_rule("python-dict-entry");
        let dict_tail_step = self.space_add(dict_comma, entry_ref_1);
        let dict_tail = self.zero_or_more(dict_tail_step);
        let dict_trailing_comma = self.literal(",");
        let opt_dict_trailing = self.optional(dict_trailing_comma);
        let sp_dict = self.space();
        let dict_trailing = self.add(sp_dict, opt_dict_trailing);
        let dict_items = self.sequence(vec![dict_entry, dict_tail, dict_trailing]);
        let opt_dict_items = self.optional(dict_items);
        let open_curly = self.literal("{");
        let close_curly = self.literal("}");
        let dict_left = self.space_add(open_curly, opt_dict_items);
        let dict_body = self.space_add(dict_left, close_curly);
        let py_dict = self.rule("python-dict", dict_body, false);

        // Python set (non-empty): `{` space val (space `,` space val)* (space `,`)? space `}`
        let set_v1 = self.ref_rule("python-value");
        let set_comma = self.literal(",");
        let set_v2 = self.ref_rule("python-value");
        let set_tail_step = self.space_add(set_comma, set_v2);
        let set_tail = self.zero_or_more(set_tail_step);
        let set_trailing_comma = self.literal(",");
        let opt_set_trailing = self.optional(set_trailing_comma);
        let sp_set = self.space();
        let set_trailing = self.add(sp_set, opt_set_trailing);
        let set_lbrace = self.literal("{");
        let set_rbrace = self.literal("}");
        let set_left = self.space_add(set_lbrace, set_v1);
        let set_mid = self.sequence(vec![set_left, set_tail, set_trailing]);
        let set_body = self.space_add(set_mid, set_rbrace);
        let py_set = self.rule("python-set", set_body, false);

        let val_choice = self.choice(vec![
            py_dict, py_set, py_list, py_tuple, py_str, py_num, py_bool, py_none,
        ]);
        Ok(self.rule("python-value", val_choice, false))
    }

    /// Validates all references and builds an immutable [`Arena`].
    pub(super) fn build(self) -> Result<Arena> {
        let root = self
            .root
            .ok_or_else(|| invalid("PEG parser builder has no root parser"))?;
        if root >= self.parsers.len() {
            return Err(invalid("PEG root parser ID out of bounds"));
        }
        for node in &self.parsers {
            match node {
                ParserNode::Ref(name) => {
                    if !self.rules.contains_key(name) {
                        return Err(invalid(format!("undefined PEG rule reference: {name}")));
                    }
                }
                ParserNode::Sequence(items) | ParserNode::Choice(items) => {
                    if items.iter().any(|&id| id >= self.parsers.len()) {
                        return Err(invalid("PEG child parser ID out of bounds"));
                    }
                }
                ParserNode::Repeat { child, .. }
                | ParserNode::Ac { child, .. }
                | ParserNode::Atomic(child)
                | ParserNode::Peek(child)
                | ParserNode::Negate(child)
                | ParserNode::Tag { child, .. }
                | ParserNode::Rule { child, .. }
                | ParserNode::Gbnf { child, .. }
                    if *child >= self.parsers.len() =>
                {
                    return Err(invalid("PEG child parser ID out of bounds"));
                }
                _ => {}
            }
        }
        let ac_matchers = self
            .parsers
            .iter()
            .map(|node| match node {
                ParserNode::Until(delims)
                | ParserNode::Ac {
                    delimiters: delims, ..
                } => compile_ac_matcher(delims),
                _ => Ok(None),
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Arena {
            parsers: self.parsers,
            rules: self.rules,
            ac_matchers,
            root,
        })
    }
}

fn compile_ac_matcher(delims: &[String]) -> Result<Option<AhoCorasick>> {
    if delims.is_empty() || delims.iter().any(String::is_empty) {
        return Ok(None);
    }
    AhoCorasick::builder()
        .match_kind(MatchKind::LeftmostFirst)
        .build(delims)
        .map(Some)
        .map_err(|err| invalid(format!("failed to compile PEG Aho-Corasick matcher: {err}")))
}

// ---------------------------------------------------------------------------
// Compiled immutable Arena, packrat parser, GBNF emitter, JSON codec
// ---------------------------------------------------------------------------

/// Immutable compiled PEG parser arena.
#[derive(Clone, Debug)]
pub(super) struct Arena {
    parsers: Vec<ParserNode>,
    rules: BTreeMap<String, ParserId>,
    ac_matchers: Vec<Option<AhoCorasick>>,
    root: ParserId,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct MemoKey {
    parser_id: ParserId,
    pos: usize,
    slice_end: usize,
    slice_is_partial: bool,
}

#[derive(Clone, Debug)]
struct MemoEntry {
    result: ParseResult,
}

struct ParseState<'arena, 'ctx, 'input> {
    arena: &'arena Arena,
    ctx: &'ctx mut ParseContext<'input>,
    memo: BTreeMap<MemoKey, MemoEntry>,
    active: BTreeSet<MemoKey>,
    depth: usize,
}

impl<'arena, 'ctx, 'input> ParseState<'arena, 'ctx, 'input> {
    fn new(arena: &'arena Arena, ctx: &'ctx mut ParseContext<'input>) -> Self {
        Self {
            arena,
            ctx,
            memo: BTreeMap::new(),
            active: BTreeSet::new(),
            depth: 0,
        }
    }

    fn parse_node(
        &mut self,
        parser_id: ParserId,
        pos: usize,
        slice_end: usize,
        slice_is_partial: bool,
    ) -> Result<ParseResult> {
        if self.depth >= MAX_PARSE_DEPTH {
            return Err(invalid("PEG parse recursion depth exceeded"));
        }
        let key = MemoKey {
            parser_id,
            pos,
            slice_end,
            slice_is_partial,
        };
        if let Some(cached) = self.memo.get(&key) {
            return Ok(cached.result.clone());
        }
        if !self.active.insert(key) {
            return Ok(ParseResult::fail(pos));
        }
        self.depth += 1;
        let result = self.eval_node(parser_id, pos, slice_end, slice_is_partial);
        self.depth -= 1;
        self.active.remove(&key);
        let result = result?;
        self.memo.insert(
            key,
            MemoEntry {
                result: result.clone(),
            },
        );
        Ok(result)
    }

    fn eval_captured(
        &mut self,
        rule: String,
        tag: String,
        child: ParserId,
        pos: usize,
        slice_end: usize,
        slice_is_partial: bool,
    ) -> Result<ParseResult> {
        let res = self.parse_node(child, pos, slice_end, slice_is_partial)?;
        if res.result_type == ParseResultType::Fail {
            return Ok(res);
        }
        let node_id =
            self.ctx
                .ast
                .add_node(rule, tag, res.start, res.end, self.ctx.input, res.nodes);
        Ok(ParseResult {
            result_type: res.result_type,
            start: res.start,
            end: res.end,
            nodes: vec![node_id],
        })
    }

    fn eval_node(
        &mut self,
        parser_id: ParserId,
        pos: usize,
        slice_end: usize,
        slice_is_partial: bool,
    ) -> Result<ParseResult> {
        let node = self
            .arena
            .parsers
            .get(parser_id)
            .ok_or_else(|| invalid("PEG parser ID out of bounds"))?;
        match node {
            ParserNode::Eps | ParserNode::Marker => Ok(ParseResult::success(pos, pos, Vec::new())),
            ParserNode::Start => Ok(if pos == 0 {
                ParseResult::success(pos, pos, Vec::new())
            } else {
                ParseResult::fail(pos)
            }),
            ParserNode::End => Ok(if pos < slice_end {
                ParseResult::fail(pos)
            } else {
                ParseResult::success(pos, pos, Vec::new())
            }),
            ParserNode::Any => Ok(self.eval_any(pos, slice_end, slice_is_partial)),
            ParserNode::Space => Ok(self.eval_space(pos, slice_end, slice_is_partial)),
            ParserNode::Rest => Ok(self.eval_rest(pos, slice_end, slice_is_partial)),
            ParserNode::Literal(lit) => {
                Ok(self.eval_literal(lit, pos, slice_end, slice_is_partial))
            }
            ParserNode::Chars(spec) => Ok(self.eval_chars(spec, pos, slice_end, slice_is_partial)),
            ParserNode::StringContent(quote) => {
                Ok(self.eval_string_content(*quote, false, pos, slice_end, slice_is_partial))
            }
            ParserNode::JsonStringContent => {
                Ok(self.eval_string_content(b'"', true, pos, slice_end, slice_is_partial))
            }
            ParserNode::Sequence(children) => {
                self.eval_sequence(children, pos, slice_end, slice_is_partial)
            }
            ParserNode::Choice(children) => {
                self.eval_choice(children, pos, slice_end, slice_is_partial)
            }
            ParserNode::Repeat { child, min, max } => {
                self.eval_repeat(*child, *min, *max, pos, slice_end, slice_is_partial)
            }
            ParserNode::Until(delims) => {
                Ok(self.eval_until(parser_id, delims, pos, slice_end, slice_is_partial))
            }
            ParserNode::Ac { child, delimiters } => self.eval_ac(
                parser_id,
                *child,
                delimiters,
                pos,
                slice_end,
                slice_is_partial,
            ),
            ParserNode::Atomic(child) => {
                self.eval_wrapper(0, *child, pos, slice_end, slice_is_partial)
            }
            ParserNode::Peek(child) => {
                self.eval_wrapper(1, *child, pos, slice_end, slice_is_partial)
            }
            ParserNode::Negate(child) => {
                self.eval_wrapper(2, *child, pos, slice_end, slice_is_partial)
            }
            ParserNode::Tag { tag, child } => self.eval_captured(
                String::new(),
                tag.clone(),
                *child,
                pos,
                slice_end,
                slice_is_partial,
            ),
            ParserNode::Rule { name, child, .. } => self.eval_captured(
                name.clone(),
                String::new(),
                *child,
                pos,
                slice_end,
                slice_is_partial,
            ),
            ParserNode::Ref(name) => {
                let target = *self
                    .arena
                    .rules
                    .get(name)
                    .ok_or_else(|| invalid(format!("undefined PEG rule reference: {name}")))?;
                self.parse_node(target, pos, slice_end, slice_is_partial)
            }
            ParserNode::Gbnf { child, .. } => {
                self.parse_node(*child, pos, slice_end, slice_is_partial)
            }
        }
    }

    fn eval_wrapper(
        &mut self,
        wrapper_kind: u8,
        child: ParserId,
        pos: usize,
        slice_end: usize,
        slice_is_partial: bool,
    ) -> Result<ParseResult> {
        let res = self.parse_node(child, pos, slice_end, slice_is_partial)?;
        Ok(match wrapper_kind {
            0 => {
                if res.result_type == ParseResultType::NeedMoreInput {
                    ParseResult::need_more_input(pos, pos, Vec::new())
                } else {
                    res
                }
            }
            1 => match res.result_type {
                ParseResultType::Fail => ParseResult::fail(pos),
                ParseResultType::Success => ParseResult::success(pos, pos, Vec::new()),
                ParseResultType::NeedMoreInput => {
                    ParseResult::need_more_input(pos, pos, Vec::new())
                }
            },
            _ => match res.result_type {
                ParseResultType::Fail => ParseResult::success(pos, pos, Vec::new()),
                ParseResultType::Success => ParseResult::fail(pos),
                ParseResultType::NeedMoreInput => {
                    ParseResult::need_more_input(pos, pos, Vec::new())
                }
            },
        })
    }

    fn eval_any(&self, pos: usize, slice_end: usize, slice_is_partial: bool) -> ParseResult {
        if pos >= slice_end {
            return if slice_is_partial {
                ParseResult::need_more_input(pos, pos, Vec::new())
            } else {
                ParseResult::fail(pos)
            };
        }
        match decode_utf8(&self.ctx.input[pos..slice_end]) {
            Utf8Decode::Char(_, len) => ParseResult::success(pos, pos + len, Vec::new()),
            Utf8Decode::Incomplete => {
                if slice_is_partial {
                    ParseResult::need_more_input(pos, pos, Vec::new())
                } else {
                    ParseResult::fail(pos)
                }
            }
            Utf8Decode::Invalid => ParseResult::fail(pos),
        }
    }

    fn eval_space(&self, pos: usize, slice_end: usize, slice_is_partial: bool) -> ParseResult {
        let mut cur = pos;
        while cur < slice_end {
            let b = self.ctx.input[cur];
            if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
                cur += 1;
            } else {
                return ParseResult::success(pos, cur, Vec::new());
            }
        }
        if slice_is_partial {
            ParseResult::need_more_input(pos, cur, Vec::new())
        } else {
            ParseResult::success(pos, cur, Vec::new())
        }
    }

    fn eval_rest(&self, pos: usize, slice_end: usize, slice_is_partial: bool) -> ParseResult {
        if slice_is_partial {
            let safe_end =
                slice_end - trailing_incomplete_utf8_len(&self.ctx.input[pos..slice_end]);
            ParseResult::need_more_input(pos, safe_end, Vec::new())
        } else {
            ParseResult::success(pos, slice_end, Vec::new())
        }
    }

    fn eval_literal(
        &self,
        lit: &str,
        pos: usize,
        slice_end: usize,
        slice_is_partial: bool,
    ) -> ParseResult {
        let lit_bytes = lit.as_bytes();
        let rem = slice_end - pos;
        if rem < lit_bytes.len() {
            if self.ctx.input[pos..slice_end] == lit_bytes[..rem] && slice_is_partial {
                return ParseResult::need_more_input(pos, pos, Vec::new());
            }
            return ParseResult::fail(pos);
        }
        if self.ctx.input[pos..pos + lit_bytes.len()] == *lit_bytes {
            ParseResult::success(pos, pos + lit_bytes.len(), Vec::new())
        } else {
            ParseResult::fail(pos)
        }
    }

    fn eval_chars(
        &self,
        spec: &CharsSpec,
        pos: usize,
        slice_end: usize,
        slice_is_partial: bool,
    ) -> ParseResult {
        let mut cur = pos;
        let mut count = 0_usize;
        while spec.max.is_none_or(|max| count < max) {
            if cur >= slice_end {
                if slice_is_partial && count < spec.min {
                    return ParseResult::need_more_input(pos, cur, Vec::new());
                }
                break;
            }
            match decode_utf8(&self.ctx.input[cur..slice_end]) {
                Utf8Decode::Char(cp, len) => {
                    if spec.matches(cp) {
                        cur += len;
                        count += 1;
                    } else {
                        break;
                    }
                }
                Utf8Decode::Incomplete => {
                    if slice_is_partial {
                        return ParseResult::need_more_input(pos, cur, Vec::new());
                    }
                    break;
                }
                Utf8Decode::Invalid => break,
            }
        }
        if count < spec.min {
            ParseResult::fail(pos)
        } else {
            ParseResult::success(pos, cur, Vec::new())
        }
    }

    fn eval_hex_escape(
        &self,
        pos: usize,
        cur: usize,
        slice_end: usize,
        slice_is_partial: bool,
        digits: usize,
    ) -> std::result::Result<usize, ParseResult> {
        let avail = slice_end - (cur + 2);
        let check = avail.min(digits);
        let hex_ok = self.ctx.input[cur + 2..cur + 2 + check]
            .iter()
            .all(u8::is_ascii_hexdigit);
        if !hex_ok {
            return Err(ParseResult::success(pos, cur, Vec::new()));
        }
        if avail < digits {
            return Err(if slice_is_partial {
                ParseResult::need_more_input(pos, cur, Vec::new())
            } else {
                ParseResult::success(pos, cur, Vec::new())
            });
        }
        Ok(cur + 2 + digits)
    }

    fn eval_string_escape(
        &self,
        json_strict: bool,
        pos: usize,
        cur: usize,
        slice_end: usize,
        slice_is_partial: bool,
    ) -> std::result::Result<usize, ParseResult> {
        let Some(&esc) = self.ctx.input.get(cur + 1).filter(|_| cur + 1 < slice_end) else {
            return Err(if slice_is_partial {
                ParseResult::need_more_input(pos, cur, Vec::new())
            } else {
                ParseResult::success(pos, cur, Vec::new())
            });
        };
        if esc == b'u' {
            return self.eval_hex_escape(pos, cur, slice_end, slice_is_partial, 4);
        }
        if json_strict {
            if matches!(esc, b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't') {
                return Ok(cur + 2);
            }
            return Err(ParseResult::success(pos, cur, Vec::new()));
        }
        if esc == b'x' {
            return self.eval_hex_escape(pos, cur, slice_end, slice_is_partial, 2);
        }
        if esc == b'U' {
            return self.eval_hex_escape(pos, cur, slice_end, slice_is_partial, 8);
        }
        if matches!(
            esc,
            b'"' | b'\'' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' | b'a' | b'v'
        ) {
            return Ok(cur + 2);
        }
        match decode_utf8(&self.ctx.input[cur + 1..slice_end]) {
            Utf8Decode::Char(cp, len) if cp >= 0x20 => Ok(cur + 1 + len),
            Utf8Decode::Incomplete if slice_is_partial => {
                Err(ParseResult::need_more_input(pos, cur, Vec::new()))
            }
            _ => Err(ParseResult::success(pos, cur, Vec::new())),
        }
    }

    fn eval_string_content(
        &self,
        quote: u8,
        json_strict: bool,
        pos: usize,
        slice_end: usize,
        slice_is_partial: bool,
    ) -> ParseResult {
        let mut cur = pos;
        while cur < slice_end {
            let b = self.ctx.input[cur];
            if b == quote || b < 0x20 {
                return ParseResult::success(pos, cur, Vec::new());
            }
            if b == b'\\' {
                match self.eval_string_escape(json_strict, pos, cur, slice_end, slice_is_partial) {
                    Ok(next) => {
                        cur = next;
                        continue;
                    }
                    Err(early) => return early,
                }
            }
            match decode_utf8(&self.ctx.input[cur..slice_end]) {
                Utf8Decode::Char(_, len) => cur += len,
                Utf8Decode::Incomplete => {
                    return if slice_is_partial {
                        ParseResult::need_more_input(pos, cur, Vec::new())
                    } else {
                        ParseResult::fail(pos)
                    };
                }
                Utf8Decode::Invalid => return ParseResult::fail(pos),
            }
        }
        if slice_is_partial {
            ParseResult::need_more_input(pos, cur, Vec::new())
        } else {
            ParseResult::success(pos, cur, Vec::new())
        }
    }

    fn eval_sequence(
        &mut self,
        children: &[ParserId],
        pos: usize,
        slice_end: usize,
        slice_is_partial: bool,
    ) -> Result<ParseResult> {
        let mut cur = pos;
        let mut nodes = Vec::new();
        for &child in children {
            let res = self.parse_node(child, cur, slice_end, slice_is_partial)?;
            match res.result_type {
                ParseResultType::Fail => return Ok(ParseResult::fail(pos)),
                ParseResultType::NeedMoreInput => {
                    nodes.extend(res.nodes);
                    return Ok(ParseResult::need_more_input(pos, res.end, nodes));
                }
                ParseResultType::Success => {
                    nodes.extend(res.nodes);
                    cur = res.end;
                }
            }
        }
        Ok(ParseResult::success(pos, cur, nodes))
    }

    fn eval_choice(
        &mut self,
        children: &[ParserId],
        pos: usize,
        slice_end: usize,
        slice_is_partial: bool,
    ) -> Result<ParseResult> {
        for &child in children {
            let res = self.parse_node(child, pos, slice_end, slice_is_partial)?;
            match res.result_type {
                ParseResultType::Fail => {}
                ParseResultType::Success | ParseResultType::NeedMoreInput => return Ok(res),
            }
        }
        Ok(ParseResult::fail(pos))
    }

    fn eval_repeat(
        &mut self,
        child: ParserId,
        min: usize,
        max: Option<usize>,
        pos: usize,
        slice_end: usize,
        slice_is_partial: bool,
    ) -> Result<ParseResult> {
        let mut cur = pos;
        let mut count = 0_usize;
        let mut nodes = Vec::new();
        while max.is_none_or(|limit| count < limit) {
            let res = self.parse_node(child, cur, slice_end, slice_is_partial)?;
            match res.result_type {
                ParseResultType::Fail => {
                    if count < min {
                        return Ok(ParseResult::fail(pos));
                    }
                    return Ok(ParseResult::success(pos, cur, nodes));
                }
                ParseResultType::NeedMoreInput => {
                    if count >= min && cur == slice_end && res.end == cur {
                        return Ok(ParseResult::success(pos, cur, nodes));
                    }
                    nodes.extend(res.nodes);
                    return Ok(ParseResult::need_more_input(pos, res.end, nodes));
                }
                ParseResultType::Success => {
                    nodes.extend(res.nodes);
                    count += 1;
                    if res.end == cur && count >= min {
                        return Ok(ParseResult::success(pos, cur, nodes));
                    }
                    cur = res.end;
                }
            }
        }
        Ok(ParseResult::success(pos, cur, nodes))
    }

    fn find_delimiter_horizon(
        &self,
        node_idx: usize,
        delims: &[String],
        pos: usize,
        slice_end: usize,
        slice_is_partial: bool,
    ) -> (usize, bool) {
        if delims.iter().any(String::is_empty) {
            return (pos, true);
        }
        if let Some(ac) = self
            .arena
            .ac_matchers
            .get(node_idx)
            .and_then(Option::as_ref)
            && let Some(mat) = ac.find(&self.ctx.input[pos..slice_end])
        {
            return (pos + mat.start(), true);
        }
        if !slice_is_partial {
            return (slice_end, false);
        }
        let safe_end = slice_end - trailing_incomplete_utf8_len(&self.ctx.input[pos..slice_end]);
        let max_partial = delims
            .iter()
            .filter(|d| !d.is_empty())
            .map(|d| {
                let bytes = d.as_bytes();
                let limit = bytes.len().saturating_sub(1).min(safe_end - pos);
                (1..=limit)
                    .rev()
                    .find(|&k| self.ctx.input[safe_end - k..safe_end] == bytes[..k])
                    .unwrap_or(0)
            })
            .max()
            .unwrap_or(0);
        (safe_end - max_partial, false)
    }

    fn eval_until(
        &self,
        node_idx: usize,
        delims: &[String],
        pos: usize,
        slice_end: usize,
        slice_is_partial: bool,
    ) -> ParseResult {
        let (stop, found) =
            self.find_delimiter_horizon(node_idx, delims, pos, slice_end, slice_is_partial);
        let mut cur = pos;
        while cur < stop {
            match decode_utf8(&self.ctx.input[cur..slice_end]) {
                Utf8Decode::Char(_, len) => cur += len,
                Utf8Decode::Incomplete => {
                    return if slice_is_partial {
                        ParseResult::need_more_input(pos, cur, Vec::new())
                    } else {
                        ParseResult::fail(pos)
                    };
                }
                Utf8Decode::Invalid => return ParseResult::fail(pos),
            }
        }
        if found || !slice_is_partial {
            ParseResult::success(pos, stop, Vec::new())
        } else {
            ParseResult::need_more_input(pos, stop, Vec::new())
        }
    }

    fn eval_ac(
        &mut self,
        node_idx: usize,
        child: ParserId,
        delims: &[String],
        pos: usize,
        slice_end: usize,
        slice_is_partial: bool,
    ) -> Result<ParseResult> {
        let (sub_end, found) =
            self.find_delimiter_horizon(node_idx, delims, pos, slice_end, slice_is_partial);
        let sub_partial = !found && slice_is_partial;
        self.parse_node(child, pos, sub_end, sub_partial)
    }
}

impl Arena {
    /// Returns the root parser identifier.
    pub(super) const fn root(&self) -> ParserId {
        self.root
    }

    /// Parses from byte offset 0 using `ctx`.
    pub(super) fn parse_with_context(&self, ctx: &mut ParseContext<'_>) -> Result<ParseResult> {
        self.parse_from(ctx, 0)
    }

    /// Parses starting at byte offset `start` using `ctx`.
    pub(super) fn parse_from(
        &self,
        ctx: &mut ParseContext<'_>,
        start: usize,
    ) -> Result<ParseResult> {
        if start > ctx.input.len() {
            return Err(invalid("PEG parse start offset exceeds input length"));
        }
        let is_partial = ctx.is_partial();
        let mut state = ParseState::new(self, ctx);
        state.parse_node(self.root, start, state.ctx.input.len(), is_partial)
    }

    /// Convenience helper matching `(input, flags)` -> `(Type, Start, End)`.
    pub(super) fn parse(&self, input: &[u8], flags: u8) -> Result<(u8, usize, usize)> {
        let mut ctx = ParseContext::new(input, flags);
        let res = self.parse_with_context(&mut ctx)?;
        Ok(res.to_wire())
    }

    /// Searches forward across `input` for the first non-failing match.
    #[allow(
        dead_code,
        reason = "Exposed for Stage 2-4 chat PEG/auto/chat providers"
    )]
    pub(super) fn parse_anywhere(&self, ctx: &mut ParseContext<'_>) -> Result<ParseResult> {
        if ctx.input.is_empty() {
            return self.parse_from(ctx, 0);
        }
        let mut last = ParseResult::fail(0);
        for pos in 0..ctx.input.len() {
            let res = self.parse_from(ctx, pos)?;
            if res.result_type == ParseResultType::Success {
                return Ok(res);
            }
            last = res;
        }
        Ok(last)
    }

    // -----------------------------------------------------------------------
    // GBNF grammar generation
    // -----------------------------------------------------------------------

    /// Generates a GBNF grammar string from the arena (`lazy` = trigger-only mode).
    pub(super) fn build_grammar(&self, lazy: bool) -> Result<String> {
        let mut emitter = GbnfEmitter::new(self, None);
        Self::finish_grammar(&mut emitter, lazy)
    }

    /// Generates a GBNF grammar string with a custom hook for parser-ID overrides (e.g., `schema` nodes).
    pub(super) fn build_grammar_with_hook<F>(&self, lazy: bool, mut hook: F) -> Result<String>
    where
        F: FnMut(ParserId, &mut BTreeMap<String, String>) -> Result<Option<String>>,
    {
        let mut emitter = GbnfEmitter::new(self, Some(&mut hook));
        Self::finish_grammar(&mut emitter, lazy)
    }

    fn finish_grammar(emitter: &mut GbnfEmitter<'_, '_>, lazy: bool) -> Result<String> {
        if lazy {
            emitter.emit_lazy()?;
        } else {
            emitter.emit_eager()?;
        }
        let mut output = String::new();
        for (name, body) in &emitter.rules {
            let _ = writeln!(output, "{name} ::= {body}");
        }
        Ok(output)
    }

    // -----------------------------------------------------------------------
    // Arena JSON serialization & deserialization
    // -----------------------------------------------------------------------

    /// Serializes the arena into a JSON value.
    pub(super) fn to_json(&self) -> Value {
        let parsers = self
            .parsers
            .iter()
            .map(ParserNode::to_json)
            .collect::<Vec<_>>();
        let mut rules = Map::new();
        for (name, &id) in &self.rules {
            rules.insert(name.clone(), json!(id));
        }
        json!({
            "version": 1,
            "root": self.root,
            "rules": rules,
            "parsers": parsers,
        })
    }

    /// Serializes the arena to a compact JSON string.
    pub(super) fn to_json_string(&self) -> Result<String> {
        serde_json::to_string(&self.to_json()).map_err(Into::into)
    }

    /// Deserializes and validates an [`Arena`] from a JSON string.
    pub(super) fn from_json_str(json_str: &str) -> Result<Self> {
        if json_str.len() > MAX_SERIALIZED_JSON_BYTES {
            return Err(invalid("PEG serialized arena exceeds 8 MiB bound"));
        }
        let value: Value = serde_json::from_str(json_str)?;
        Self::from_json(&value)
    }

    /// Deserializes and validates an [`Arena`] from a JSON value.
    pub(super) fn from_json(value: &Value) -> Result<Self> {
        exact_fields(value, &["version", "root", "rules", "parsers"])?;
        if value.get("version").and_then(Value::as_u64) != Some(1) {
            return Err(invalid("unsupported PEG arena serialization version"));
        }
        let root = usize::try_from(
            value
                .get("root")
                .and_then(Value::as_u64)
                .ok_or_else(|| invalid("PEG serialized root must be unsigned"))?,
        )?;
        let parsers_arr = array(value, "parsers")?;
        if parsers_arr.is_empty() || parsers_arr.len() > 131_072 {
            return Err(invalid("PEG serialized parser count out of bounds"));
        }
        let mut parsers = Vec::with_capacity(parsers_arr.len());
        for entry in parsers_arr {
            parsers.push(ParserNode::from_json(entry)?);
        }
        let rules_obj = value
            .get("rules")
            .and_then(Value::as_object)
            .ok_or_else(|| invalid("PEG serialized rules must be an object"))?;
        let mut rules = BTreeMap::new();
        for (name, id_val) in rules_obj {
            let id = usize::try_from(
                id_val
                    .as_u64()
                    .ok_or_else(|| invalid("PEG serialized rule ID must be unsigned"))?,
            )?;
            if id >= parsers.len() {
                return Err(invalid("PEG serialized rule ID out of bounds"));
            }
            rules.insert(name.clone(), id);
        }
        let builder = ParserBuilder {
            parsers,
            rules,
            root: Some(root),
        };
        builder.build()
    }
}

impl ParserNode {
    fn to_json(&self) -> Value {
        match self {
            Self::Eps => json!({"type": "eps"}),
            Self::Start => json!({"type": "start"}),
            Self::End => json!({"type": "end"}),
            Self::Any => json!({"type": "any"}),
            Self::Space => json!({"type": "space"}),
            Self::Rest => json!({"type": "rest"}),
            Self::Marker => json!({"type": "marker"}),
            Self::Literal(text) => json!({"type": "literal", "literal": text}),
            Self::Chars(spec) => {
                let max = spec
                    .max
                    .map_or(-1_i64, |val| i64::try_from(val).unwrap_or(i64::MAX));
                json!({
                    "type": "chars",
                    "pattern": spec.pattern,
                    "min": spec.min,
                    "max": max,
                })
            }
            Self::StringContent(quote) => json!({"type": "string_content", "quote": quote}),
            Self::JsonStringContent => json!({"type": "json_string_content"}),
            Self::Sequence(children) => json!({"type": "sequence", "children": children}),
            Self::Choice(children) => json!({"type": "choice", "children": children}),
            Self::Repeat { child, min, max } => {
                let max = max.map_or(-1_i64, |val| i64::try_from(val).unwrap_or(i64::MAX));
                json!({"type": "repeat", "child": child, "min": min, "max": max})
            }
            Self::Until(delimiters) => json!({"type": "until", "delimiters": delimiters}),
            Self::Ac { child, delimiters } => {
                json!({"type": "ac", "child": child, "delimiters": delimiters})
            }
            Self::Atomic(child) => json!({"type": "atomic", "child": child}),
            Self::Peek(child) => json!({"type": "peek", "child": child}),
            Self::Negate(child) => json!({"type": "negate", "child": child}),
            Self::Tag { tag, child } => json!({"type": "tag", "tag": tag, "child": child}),
            Self::Rule {
                name,
                child,
                trigger,
            } => json!({
                "type": "rule",
                "name": name,
                "child": child,
                "trigger": trigger,
            }),
            Self::Ref(name) => json!({"type": "ref", "name": name}),
            Self::Gbnf { child, grammar } => {
                json!({"type": "gbnf", "child": child, "grammar": grammar})
            }
        }
    }

    fn from_json(value: &Value) -> Result<Self> {
        let kind = value
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("PEG serialized node missing type"))?;
        match kind {
            "eps"
            | "start"
            | "end"
            | "any"
            | "space"
            | "rest"
            | "marker"
            | "json_string_content" => {
                exact_fields(value, &["type"])?;
                Ok(match kind {
                    "eps" => Self::Eps,
                    "start" => Self::Start,
                    "end" => Self::End,
                    "any" => Self::Any,
                    "space" => Self::Space,
                    "rest" => Self::Rest,
                    "marker" => Self::Marker,
                    _ => Self::JsonStringContent,
                })
            }
            "literal" => {
                exact_fields(value, &["type", "literal"])?;
                let lit = text_field(value, "literal", MAX_STRING_BYTES)?;
                Ok(Self::Literal(lit.to_owned()))
            }
            "chars" => {
                exact_fields(value, &["type", "pattern", "min", "max"])?;
                let pattern = text_field(value, "pattern", MAX_STRING_BYTES)?;
                let min = i32::try_from(int_field(value, "min", 0, 65_536)?)?;
                let max = i32::try_from(int_field(value, "max", -1, 65_536)?)?;
                Ok(Self::Chars(CharsSpec::parse(pattern, min, max)?))
            }
            "string_content" => {
                exact_fields(value, &["type", "quote"])?;
                let quote = u8::try_from(int_field(value, "quote", 0, 255)?)?;
                Ok(Self::StringContent(quote))
            }
            _ => Self::from_json_composite(kind, value),
        }
    }

    fn from_json_composite(kind: &str, value: &Value) -> Result<Self> {
        match kind {
            "sequence" => {
                exact_fields(value, &["type", "children"])?;
                Ok(Self::Sequence(id_array(value, "children")?))
            }
            "choice" => {
                exact_fields(value, &["type", "children"])?;
                Ok(Self::Choice(id_array(value, "children")?))
            }
            "repeat" => {
                exact_fields(value, &["type", "child", "min", "max"])?;
                let child = id_field(value, "child")?;
                let min = usize::try_from(int_field(value, "min", 0, 65_536)?)?;
                let max_raw = int_field(value, "max", -1, 65_536)?;
                let max = if max_raw < 0 {
                    None
                } else {
                    let val = usize::try_from(max_raw)?;
                    if val < min {
                        return Err(invalid("PEG serialized repeat bounds inverted"));
                    }
                    Some(val)
                };
                Ok(Self::Repeat { child, min, max })
            }
            "until" => {
                exact_fields(value, &["type", "delimiters"])?;
                Ok(Self::Until(string_array(value, "delimiters")?))
            }
            "ac" => {
                exact_fields(value, &["type", "child", "delimiters"])?;
                let child = id_field(value, "child")?;
                Ok(Self::Ac {
                    child,
                    delimiters: string_array(value, "delimiters")?,
                })
            }
            "atomic" | "peek" | "negate" => {
                exact_fields(value, &["type", "child"])?;
                let child = id_field(value, "child")?;
                Ok(match kind {
                    "atomic" => Self::Atomic(child),
                    "peek" => Self::Peek(child),
                    _ => Self::Negate(child),
                })
            }
            "tag" => {
                exact_fields(value, &["type", "tag", "child"])?;
                let tag = text_field(value, "tag", MAX_STRING_BYTES)?.to_owned();
                Ok(Self::Tag {
                    tag,
                    child: id_field(value, "child")?,
                })
            }
            "rule" => {
                exact_fields(value, &["type", "name", "child", "trigger"])?;
                let name = text_field(value, "name", MAX_LOCAL_NAME)?.to_owned();
                let child = id_field(value, "child")?;
                let trigger = field(value, "trigger")?
                    .as_bool()
                    .ok_or_else(|| invalid("PEG serialized trigger must be boolean"))?;
                Ok(Self::Rule {
                    name,
                    child,
                    trigger,
                })
            }
            "ref" => {
                exact_fields(value, &["type", "name"])?;
                Ok(Self::Ref(
                    text_field(value, "name", MAX_LOCAL_NAME)?.to_owned(),
                ))
            }
            "gbnf" => {
                exact_fields(value, &["type", "child", "grammar"])?;
                let child = id_field(value, "child")?;
                let grammar = text_field(value, "grammar", MAX_STRING_BYTES)?.to_owned();
                Ok(Self::Gbnf { child, grammar })
            }
            _ => Err(invalid(format!("unknown serialized PEG node type: {kind}"))),
        }
    }
}

fn id_field(value: &Value, name: &str) -> Result<ParserId> {
    let raw = field(value, name)?
        .as_u64()
        .ok_or_else(|| invalid(format!("PEG serialized {name} must be unsigned")))?;
    Ok(usize::try_from(raw)?)
}

fn id_array(value: &Value, name: &str) -> Result<Vec<ParserId>> {
    let arr = array(value, name)?;
    if arr.len() > MAX_LIST_LEN {
        return Err(invalid("PEG serialized child array exceeds bound"));
    }
    arr.iter()
        .map(|entry| {
            let raw = entry
                .as_u64()
                .ok_or_else(|| invalid("PEG serialized child ID must be unsigned"))?;
            Ok(usize::try_from(raw)?)
        })
        .collect()
}

fn string_array(value: &Value, name: &str) -> Result<Vec<String>> {
    let arr = array(value, name)?;
    if arr.len() > MAX_LIST_LEN {
        return Err(invalid("PEG serialized delimiter array exceeds bound"));
    }
    arr.iter()
        .map(|entry| {
            let s = entry
                .as_str()
                .ok_or_else(|| invalid("PEG serialized delimiter must be string"))?;
            if s.len() > MAX_STRING_BYTES {
                return Err(invalid("PEG serialized delimiter exceeds byte bound"));
            }
            Ok(s.to_owned())
        })
        .collect()
}

// ---------------------------------------------------------------------------
// GBNF grammar generation
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum GbnfPrecedence {
    Atom,
    Postfix,
    Sequence,
    Choice,
}

struct GbnfExpr {
    text: String,
    prec: GbnfPrecedence,
}

impl GbnfExpr {
    const fn atom(text: String) -> Self {
        Self {
            text,
            prec: GbnfPrecedence::Atom,
        }
    }

    fn wrap_for_postfix(self) -> String {
        if self.text.is_empty() {
            "()".to_owned()
        } else if self.prec == GbnfPrecedence::Atom {
            self.text
        } else {
            format!("({})", self.text)
        }
    }

    fn wrap_for_sequence(self) -> String {
        if self.text.is_empty() {
            "()".to_owned()
        } else if self.prec == GbnfPrecedence::Choice {
            format!("({})", self.text)
        } else {
            self.text
        }
    }
}

type SchemaHook<'b> =
    &'b mut dyn FnMut(ParserId, &mut BTreeMap<String, String>) -> Result<Option<String>>;

struct GbnfEmitter<'a, 'b> {
    arena: &'a Arena,
    rules: BTreeMap<String, String>,
    rule_names: BTreeMap<ParserId, String>,
    emitted_rules: BTreeSet<String>,
    subrule_counter: usize,
    schema_hook: Option<SchemaHook<'b>>,
}

impl<'a, 'b> GbnfEmitter<'a, 'b> {
    const fn new(arena: &'a Arena, schema_hook: Option<SchemaHook<'b>>) -> Self {
        Self {
            arena,
            rules: BTreeMap::new(),
            rule_names: BTreeMap::new(),
            emitted_rules: BTreeSet::new(),
            subrule_counter: 0,
            schema_hook,
        }
    }

    fn escape_name(name: &str) -> String {
        let mut output = String::new();
        let mut invalid_run = false;
        for ch in name.chars() {
            if ch.is_ascii_alphanumeric() || ch == '-' {
                output.push(ch);
                invalid_run = false;
            } else if !invalid_run {
                output.push('-');
                invalid_run = true;
            }
        }
        if output.is_empty() {
            "rule".to_owned()
        } else {
            output
        }
    }

    fn rule_gbnf_name(&mut self, rule_id: ParserId, raw_name: &str) -> &str {
        if let Some(existing) = self.rule_names.get(&rule_id) {
            return existing;
        }
        let base = Self::escape_name(raw_name);
        let mut candidate = base.clone();
        let mut suffix = 0_usize;
        while self
            .rule_names
            .values()
            .any(|assigned| assigned == &candidate)
        {
            candidate = format!("{base}-{suffix}");
            suffix += 1;
        }
        self.rule_names.entry(rule_id).or_insert(candidate)
    }

    fn add_unique_rule(&mut self, base: &str, body: String) -> String {
        let escaped = Self::escape_name(base);
        if let Some(existing) = self.rules.get(&escaped) {
            if existing == &body {
                return escaped;
            }
        } else {
            self.rules.insert(escaped.clone(), body);
            return escaped;
        }
        loop {
            let key = format!("{escaped}-{}", self.subrule_counter);
            self.subrule_counter += 1;
            if let Some(existing) = self.rules.get(&key) {
                if existing == &body {
                    return key;
                }
            } else {
                self.rules.insert(key.clone(), body);
                return key;
            }
        }
    }

    fn emit_eager(&mut self) -> Result<()> {
        let _ = self.emit_space();
        let root_id = self.arena.root;
        if let Some(ParserNode::Rule { name, child, .. }) = self.arena.parsers.get(root_id) {
            if name == "root" {
                let expr = self.emit_expr(*child)?;
                self.rules.insert("root".to_owned(), expr.text);
                return Ok(());
            }
            let rule_name = self.ensure_rule_emitted(root_id, name, *child)?;
            self.rules.insert("root".to_owned(), rule_name);
            return Ok(());
        }
        let expr = self.emit_expr(root_id)?;
        self.rules.insert("root".to_owned(), expr.text);
        Ok(())
    }

    fn emit_lazy(&mut self) -> Result<()> {
        let _ = self.emit_space();
        let mut triggers = Vec::new();
        let mut visited = BTreeSet::new();
        self.collect_triggers(self.arena.root, &mut visited, &mut triggers);
        if triggers.is_empty() {
            self.rules.insert("root".to_owned(), String::new());
            return Ok(());
        }
        let mut root_alts = Vec::with_capacity(triggers.len());
        for trigger_id in triggers {
            if let Some(ParserNode::Rule { name, child, .. }) = self.arena.parsers.get(trigger_id) {
                if name == "root" {
                    let expr = self.emit_expr(*child)?;
                    root_alts.push(expr.wrap_for_sequence());
                } else {
                    let rule_name = self.ensure_rule_emitted(trigger_id, name, *child)?;
                    root_alts.push(rule_name);
                }
            }
        }
        self.rules.insert("root".to_owned(), root_alts.join(" | "));
        Ok(())
    }

    fn collect_triggers(
        &self,
        id: ParserId,
        visited: &mut BTreeSet<ParserId>,
        triggers: &mut Vec<ParserId>,
    ) {
        if !visited.insert(id) {
            return;
        }
        let Some(node) = self.arena.parsers.get(id) else {
            return;
        };
        match node {
            ParserNode::Rule { child, trigger, .. } => {
                if *trigger {
                    triggers.push(id);
                }
                self.collect_triggers(*child, visited, triggers);
            }
            ParserNode::Repeat { child, .. }
            | ParserNode::Ac { child, .. }
            | ParserNode::Atomic(child)
            | ParserNode::Peek(child)
            | ParserNode::Negate(child)
            | ParserNode::Tag { child, .. }
            | ParserNode::Gbnf { child, .. } => {
                self.collect_triggers(*child, visited, triggers);
            }
            ParserNode::Sequence(items) | ParserNode::Choice(items) => {
                for &item in items {
                    self.collect_triggers(item, visited, triggers);
                }
            }
            ParserNode::Ref(name) => {
                if let Some(&target) = self.arena.rules.get(name) {
                    self.collect_triggers(target, visited, triggers);
                }
            }
            _ => {}
        }
    }

    fn ensure_rule_emitted(
        &mut self,
        rule_id: ParserId,
        raw_name: &str,
        child: ParserId,
    ) -> Result<String> {
        let gbnf_name = self.rule_gbnf_name(rule_id, raw_name).to_owned();
        if !self.emitted_rules.contains(&gbnf_name) {
            self.emitted_rules.insert(gbnf_name.clone());
            let body = self.emit_expr(child)?.text;
            self.rules.insert(gbnf_name.clone(), body);
        }
        Ok(gbnf_name)
    }

    fn format_literal(text: &str) -> String {
        let mut out = String::from("\"");
        for ch in text.chars() {
            match ch {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if u32::from(c) < 0x20 => {
                    let _ = write!(out, "\\x{:02x}", u32::from(c));
                }
                c => out.push(c),
            }
        }
        out.push('"');
        out
    }

    fn emit_space(&mut self) -> GbnfExpr {
        if !self.rules.contains_key("space") {
            self.rules.insert(
                "space".to_owned(),
                "| \" \" | \"\\n\"{1,2} [ \\t]{0,20}".to_owned(),
            );
        }
        GbnfExpr::atom("space".to_owned())
    }

    fn emit_string_content(quote: u8) -> GbnfExpr {
        let q = char::from(quote);
        let escaped_q = match q {
            '"' => "\\\"".to_owned(),
            '\'' => "'".to_owned(),
            '\\' => "\\\\".to_owned(),
            ']' => "\\]".to_owned(),
            '^' => "\\x5E".to_owned(),
            '-' => "\\x2D".to_owned(),
            c if u32::from(c) < 0x20 => format!("\\x{:02x}", u32::from(c)),
            c => c.to_string(),
        };
        let esc_alt = if quote == b'"' {
            r#"["\\/bfnrt]"#.to_owned()
        } else if quote == b'\'' {
            r"['\\/bfnrt]".to_owned()
        } else {
            format!(r"[{escaped_q}\\/bfnrt]")
        };
        let body = format!(
            r#"([^{escaped_q}\\\x7F\x00-\x1F] | "\\" ({esc_alt} | "u" [0-9a-fA-F]{{4}}))*"#
        );
        GbnfExpr::atom(body)
    }

    fn emit_until(&mut self, node_id: ParserId, delims: &[String], is_ac: bool) -> GbnfExpr {
        let non_empty: Vec<&str> = delims
            .iter()
            .map(String::as_str)
            .filter(|d| !d.is_empty())
            .collect();
        if non_empty.is_empty() {
            return GbnfExpr {
                text: ".*".to_owned(),
                prec: GbnfPrecedence::Postfix,
            };
        }
        let prefix = if is_ac {
            format!("ac-{node_id}")
        } else {
            format!("until-{node_id}")
        };
        let rule_name = build_ac_dfa_gbnf(&prefix, &non_empty, is_ac, self);
        GbnfExpr::atom(rule_name)
    }

    fn emit_sequence(&mut self, items: Vec<ParserId>) -> Result<GbnfExpr> {
        let mut exprs = Vec::with_capacity(items.len());
        for item in items {
            let expr = self.emit_expr(item)?;
            if !expr.text.is_empty() {
                exprs.push(expr);
            }
        }
        if exprs.is_empty() {
            Ok(GbnfExpr::atom(String::new()))
        } else if exprs.len() == 1 {
            Ok(exprs.remove(0))
        } else {
            let parts: Vec<String> = exprs.into_iter().map(GbnfExpr::wrap_for_sequence).collect();
            Ok(GbnfExpr {
                text: parts.join(" "),
                prec: GbnfPrecedence::Sequence,
            })
        }
    }

    fn emit_choice(&mut self, items: Vec<ParserId>) -> Result<GbnfExpr> {
        let mut exprs = Vec::with_capacity(items.len());
        for item in items {
            let expr = self.emit_expr(item)?;
            if !expr.text.is_empty() {
                exprs.push(expr);
            }
        }
        if exprs.is_empty() {
            Ok(GbnfExpr::atom(String::new()))
        } else if exprs.len() == 1 {
            Ok(exprs.remove(0))
        } else {
            let parts: Vec<String> = exprs.into_iter().map(|e| e.text).collect();
            Ok(GbnfExpr {
                text: parts.join(" | "),
                prec: GbnfPrecedence::Choice,
            })
        }
    }

    fn emit_repeat(&mut self, child: ParserId, min: usize, max: Option<usize>) -> Result<GbnfExpr> {
        if max == Some(0) {
            return Ok(GbnfExpr::atom(String::new()));
        }
        let inner = self.emit_expr(child)?;
        if inner.text.is_empty() {
            return Ok(GbnfExpr::atom(String::new()));
        }
        let wrapped = inner.wrap_for_postfix();
        let text = match (min, max) {
            (1, Some(1)) => return Ok(GbnfExpr::atom(wrapped)),
            (0, Some(1)) => format!("{wrapped}?"),
            (0, None) => format!("{wrapped}*"),
            (1, None) => format!("{wrapped}+"),
            (lo, Some(hi)) if lo == hi => format!("{wrapped}{{{lo}}}"),
            (lo, None) => format!("{wrapped}{{{lo},}}"),
            (lo, Some(hi)) => format!("{wrapped}{{{lo},{hi}}}"),
        };
        Ok(GbnfExpr {
            text,
            prec: GbnfPrecedence::Postfix,
        })
    }

    fn emit_expr(&mut self, id: ParserId) -> Result<GbnfExpr> {
        if let Some(hook) = self.schema_hook.as_deref_mut()
            && let Some(rule_name) = hook(id, &mut self.rules)?
        {
            return Ok(GbnfExpr::atom(rule_name));
        }
        let node = self
            .arena
            .parsers
            .get(id)
            .ok_or_else(|| invalid("PEG GBNF parser ID out of bounds"))?
            .clone();
        match node {
            ParserNode::Eps
            | ParserNode::Start
            | ParserNode::End
            | ParserNode::Marker
            | ParserNode::Peek(_)
            | ParserNode::Negate(_) => Ok(GbnfExpr::atom(String::new())),
            ParserNode::Any => Ok(GbnfExpr::atom(".".to_owned())),
            ParserNode::Rest => Ok(GbnfExpr {
                text: ".*".to_owned(),
                prec: GbnfPrecedence::Postfix,
            }),
            ParserNode::Space => Ok(self.emit_space()),
            ParserNode::Literal(lit) => Ok(GbnfExpr::atom(Self::format_literal(&lit))),
            ParserNode::Chars(spec) => {
                let text = spec.to_gbnf();
                let prec = if matches!((spec.min, spec.max), (1, Some(1)) | (0, Some(0))) {
                    GbnfPrecedence::Atom
                } else {
                    GbnfPrecedence::Postfix
                };
                Ok(GbnfExpr { text, prec })
            }
            ParserNode::StringContent(quote) => Ok(Self::emit_string_content(quote)),
            ParserNode::JsonStringContent => Ok(Self::emit_string_content(b'"')),
            ParserNode::Sequence(items) => self.emit_sequence(items),
            ParserNode::Choice(items) => self.emit_choice(items),
            ParserNode::Repeat { child, min, max } => self.emit_repeat(child, min, max),
            ParserNode::Until(delims) => Ok(self.emit_until(id, &delims, false)),
            ParserNode::Ac { delimiters, .. } => Ok(self.emit_until(id, &delimiters, true)),
            ParserNode::Atomic(child) | ParserNode::Tag { child, .. } => self.emit_expr(child),
            ParserNode::Rule { name, child, .. } => {
                let gbnf_name = self.ensure_rule_emitted(id, &name, child)?;
                Ok(GbnfExpr::atom(gbnf_name))
            }
            ParserNode::Ref(name) => {
                let target = *self
                    .arena
                    .rules
                    .get(&name)
                    .ok_or_else(|| invalid(format!("undefined PEG rule reference: {name}")))?;
                if let Some(ParserNode::Rule {
                    name: rname, child, ..
                }) = self.arena.parsers.get(target)
                {
                    let gbnf_name = self.ensure_rule_emitted(target, rname, *child)?;
                    Ok(GbnfExpr::atom(gbnf_name))
                } else {
                    Ok(GbnfExpr::atom(Self::escape_name(&name)))
                }
            }
            ParserNode::Gbnf { grammar, .. } => Ok(self.emit_custom_gbnf(&grammar)),
        }
    }

    fn emit_custom_gbnf(&mut self, grammar: &str) -> GbnfExpr {
        if !grammar.contains("::=") {
            return GbnfExpr {
                text: grammar.to_owned(),
                prec: infer_gbnf_prec(grammar),
            };
        }
        let mut parsed_rules = Vec::new();
        for line in grammar.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            if let Some((lhs, rhs)) = trimmed.split_once("::=") {
                parsed_rules.push((lhs.trim().to_owned(), rhs.trim().to_owned()));
            }
        }
        if parsed_rules.is_empty() {
            return GbnfExpr::atom(String::new());
        }
        let mut root_body = None;
        for (name, body) in parsed_rules {
            if name == "root" {
                root_body = Some(body);
            } else {
                self.rules.insert(name, body);
            }
        }
        root_body.map_or_else(
            || GbnfExpr::atom(String::new()),
            |body| {
                let sub = self.add_unique_rule("gbnf", body);
                GbnfExpr::atom(sub)
            },
        )
    }
}

fn infer_gbnf_prec(text: &str) -> GbnfPrecedence {
    let mut paren_depth = 0_usize;
    let mut in_bracket = false;
    let mut in_quote = false;
    let mut escaped = false;
    let mut has_top_space = false;
    for ch in text.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        if in_quote {
            if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_quote = false;
            }
            continue;
        }
        if in_bracket {
            if ch == '\\' {
                escaped = true;
            } else if ch == ']' {
                in_bracket = false;
            }
            continue;
        }
        match ch {
            '"' => in_quote = true,
            '[' => in_bracket = true,
            '(' => paren_depth += 1,
            ')' => paren_depth = paren_depth.saturating_sub(1),
            '|' if paren_depth == 0 => return GbnfPrecedence::Choice,
            ' ' | '\t' if paren_depth == 0 => has_top_space = true,
            _ => {}
        }
    }
    if has_top_space {
        GbnfPrecedence::Sequence
    } else if text.ends_with(['*', '+', '?', '}']) {
        GbnfPrecedence::Postfix
    } else {
        GbnfPrecedence::Atom
    }
}

fn format_ac_char(ch: char) -> String {
    match ch {
        '\n' => "\\n".to_owned(),
        '\r' => "\\r".to_owned(),
        '\t' => "\\t".to_owned(),
        '\\' => "\\\\".to_owned(),
        ']' => "\\]".to_owned(),
        '[' => "\\[".to_owned(),
        '-' => "\\x2D".to_owned(),
        '^' => "\\x5E".to_owned(),
        c if u32::from(c) < 0x20 || u32::from(c) == 0x7F => {
            format!("\\x{:02X}", u32::from(c))
        }
        c if u32::from(c) <= 0x7E => c.to_string(),
        c if u32::from(c) <= 0xFFFF => format!("\\u{:04X}", u32::from(c)),
        c => format!("\\U{:08X}", u32::from(c)),
    }
}

fn format_negated_char_class(excluded: &BTreeSet<char>) -> String {
    if excluded.is_empty() {
        return ".".to_owned();
    }
    let mut out = String::from("[^");
    for &ch in excluded {
        out.push_str(&format_ac_char(ch));
    }
    out.push(']');
    out
}

#[derive(Default)]
struct AcTrieNode {
    children: BTreeMap<char, usize>,
    fail: usize,
    is_end: bool,
}

fn build_ac_trie(delims: &[&str]) -> Vec<AcTrieNode> {
    let mut nodes = vec![AcTrieNode::default()];
    for &delim in delims {
        let mut cur = 0_usize;
        for ch in delim.chars() {
            if let Some(&next) = nodes[cur].children.get(&ch) {
                cur = next;
            } else {
                let next = nodes.len();
                nodes.push(AcTrieNode::default());
                nodes[cur].children.insert(ch, next);
                cur = next;
            }
        }
        nodes[cur].is_end = true;
    }

    let mut queue = std::collections::VecDeque::new();
    let root_children: Vec<usize> = nodes[0].children.values().copied().collect();
    for child in root_children {
        nodes[child].fail = 0;
        queue.push_back(child);
    }
    while let Some(u) = queue.pop_front() {
        let edges: Vec<(char, usize)> = nodes[u].children.iter().map(|(&c, &v)| (c, v)).collect();
        for (ch, v) in edges {
            let mut f = nodes[u].fail;
            let target = loop {
                if let Some(&next) = nodes[f].children.get(&ch) {
                    break next;
                }
                if f == 0 {
                    break 0;
                }
                f = nodes[f].fail;
            };
            nodes[v].fail = target;
            if nodes[target].is_end {
                nodes[v].is_end = true;
            }
            queue.push_back(v);
        }
    }
    nodes
}

fn build_ac_dfa_gbnf(
    prefix: &str,
    delims: &[&str],
    is_ac: bool,
    emitter: &mut GbnfEmitter<'_, '_>,
) -> String {
    let nodes = build_ac_trie(delims);

    let state_name = |idx: usize| -> String {
        if idx == 0 {
            prefix.to_owned()
        } else {
            format!("{prefix}-{idx:02}")
        }
    };

    for idx in 0..nodes.len() {
        if nodes[idx].is_end {
            continue;
        }
        // Collect merged transitions in failure-chain order (outermost ancestor first,
        // overridden in-place if a deeper node defines the same char).
        let mut chain = Vec::new();
        let mut cur = idx;
        loop {
            chain.push(cur);
            if cur == 0 {
                break;
            }
            cur = nodes[cur].fail;
        }
        let mut transitions: Vec<(char, usize)> = Vec::new();
        for node_idx in chain.into_iter().rev() {
            for (&ch, &target) in &nodes[node_idx].children {
                if let Some((_, slot)) = transitions.iter_mut().find(|(c, _)| *c == ch) {
                    *slot = target;
                } else {
                    transitions.push((ch, target));
                }
            }
        }

        let mut terminal_chars = Vec::new();
        let mut next_transitions = Vec::new();
        let mut excluded = BTreeSet::new();
        for (ch, target) in transitions {
            excluded.insert(ch);
            if nodes[target].is_end {
                terminal_chars.push(ch);
            } else {
                next_transitions.push((ch, target));
            }
        }

        let mut alts = Vec::new();
        if is_ac {
            for ch in terminal_chars {
                alts.push(format!("[{}]", format_ac_char(ch)));
            }
            for (ch, target) in next_transitions {
                alts.push(format!("[{}] {}", format_ac_char(ch), state_name(target)));
            }
            alts.push(format!(
                "{} {}",
                format_negated_char_class(&excluded),
                state_name(0)
            ));
            emitter.rules.insert(state_name(idx), alts.join(" | "));
        } else {
            for (ch, target) in next_transitions {
                alts.push(format!("[{}] {}", format_ac_char(ch), state_name(target)));
            }
            alts.push(format!(
                "{} {}",
                format_negated_char_class(&excluded),
                state_name(0)
            ));
            emitter
                .rules
                .insert(state_name(idx), format!("| {}", alts.join(" | ")));
        }
    }

    state_name(0)
}

// ---------------------------------------------------------------------------
// Wire Protocol Validation & Execution (`peg-evaluate`, `peg-serialize`, `peg-deserialize`)
// ---------------------------------------------------------------------------

fn require(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(invalid(message))
    }
}

fn text_field<'a>(value: &'a Value, name: &str, maximum: usize) -> Result<&'a str> {
    let text = field(value, name)?
        .as_str()
        .ok_or_else(|| invalid(format!("PEG {name} must be a string")))?;
    if text.len() > maximum {
        return Err(invalid(format!("PEG {name} exceeds byte bound")));
    }
    Ok(text)
}

fn text_value(value: &Value, maximum: usize) -> Result<&str> {
    let text = value
        .as_str()
        .ok_or_else(|| invalid("PEG argument must be a string"))?;
    if text.len() > maximum {
        return Err(invalid("PEG string exceeds bound"));
    }
    Ok(text)
}

fn int_value(value: &Value, low: i64, high: i64) -> Result<i64> {
    let num = if let Some(u) = value.as_u64() {
        i64::try_from(u).map_err(|_| invalid("PEG integer exceeds signed 64-bit range"))?
    } else if let Some(i) = value.as_i64() {
        i
    } else {
        return Err(invalid("PEG argument must be an integer"));
    };
    if num < low || num > high {
        return Err(invalid("PEG integer outside bound"));
    }
    Ok(num)
}

fn int_field(value: &Value, name: &str, low: i64, high: i64) -> Result<i64> {
    int_value(field(value, name)?, low, high)
}

fn bool_value(value: &Value) -> Result<bool> {
    value
        .as_bool()
        .ok_or_else(|| invalid("PEG argument must be boolean"))
}

fn validate_expr(
    expression: &Value,
    locals: &BTreeSet<String>,
    depth: usize,
    count: &mut usize,
) -> Result<()> {
    *count += 1;
    require(
        depth <= MAX_GRAPH_DEPTH && *count <= MAX_EXPR_NODES && expression.is_object(),
        "PEG expression depth/count exceeds bound",
    )?;
    if expression.get("Ref").is_some() {
        exact_fields(expression, &["Ref"])?;
        let name = text_field(expression, "Ref", MAX_LOCAL_NAME)?;
        require(locals.contains(name), "unknown PEG local Ref")?;
        return Ok(());
    }
    exact_fields(expression, &["Op", "Args"])?;
    let op = text_field(expression, "Op", MAX_LOCAL_NAME)?;
    let args = array(expression, "Args")?;
    require(
        args.len() <= MAX_ARGS,
        "PEG Args must be an array of at most 64 values",
    )?;
    validate_op_expr(op, args, locals, depth, count)
}

fn validate_op_expr(
    op: &str,
    args: &[Value],
    locals: &BTreeSet<String>,
    depth: usize,
    count: &mut usize,
) -> Result<()> {
    let arity = |low: usize, high: usize| -> Result<()> {
        require(
            args.len() >= low && args.len() <= high,
            "PEG argument count mismatch",
        )
    };
    match op {
        "eps"
        | "end"
        | "any"
        | "space"
        | "double_quoted_string"
        | "single_quoted_string"
        | "json"
        | "python_value"
        | "marker" => arity(0, 0),
        "literal" | "ref" | "until" => {
            arity(1, 1)?;
            let _ = text_value(&args[0], MAX_STRING_BYTES)?;
            Ok(())
        }
        "chars" => {
            arity(1, 3)?;
            let _ = text_value(&args[0], MAX_STRING_BYTES)?;
            for (idx, arg) in args.iter().enumerate().skip(1) {
                let low = if idx == 1 { 0 } else { -1 };
                let _ = int_value(arg, low, 65_536)?;
            }
            Ok(())
        }
        "one_or_more" | "zero_or_more" | "optional" | "atomic" => {
            arity(1, 1)?;
            validate_expr(&args[0], locals, depth + 1, count)
        }
        "sequence" | "choice" => {
            arity(1, 1)?;
            let items = args[0]
                .as_array()
                .ok_or_else(|| invalid("PEG parser list must be an array"))?;
            require(items.len() <= MAX_LIST_LEN, "PEG parser list exceeds bound")?;
            for item in items {
                validate_expr(item, locals, depth + 1, count)?;
            }
            Ok(())
        }
        "until_one_of" => {
            arity(1, 1)?;
            let items = args[0]
                .as_array()
                .ok_or_else(|| invalid("PEG delimiter list must be an array"))?;
            require(
                items.len() <= MAX_LIST_LEN,
                "PEG delimiter list exceeds bound",
            )?;
            for item in items {
                let _ = text_value(item, MAX_STRING_BYTES)?;
            }
            Ok(())
        }
        "add" | "or" | "space-add" => {
            arity(2, 2)?;
            require(
                !(args[0].is_string() && args[1].is_string()),
                "PEG operator requires a parser operand",
            )?;
            for arg in args {
                if arg.is_string() {
                    let _ = text_value(arg, MAX_STRING_BYTES)?;
                } else {
                    validate_expr(arg, locals, depth + 1, count)?;
                }
            }
            Ok(())
        }
        _ => validate_op_composite(op, args, locals, depth, count),
    }
}

fn validate_op_composite(
    op: &str,
    args: &[Value],
    locals: &BTreeSet<String>,
    depth: usize,
    count: &mut usize,
) -> Result<()> {
    let arity = |low: usize, high: usize| -> Result<()> {
        require(
            args.len() >= low && args.len() <= high,
            "PEG argument count mismatch",
        )
    };
    match op {
        "rule" | "rule-lazy" => {
            arity(2, 3)?;
            let _ = text_value(&args[0], MAX_LOCAL_NAME)?;
            if op == "rule-lazy" {
                validate_graph(&args[1], locals.clone(), depth + 1, count)?;
            } else {
                validate_expr(&args[1], locals, depth + 1, count)?;
            }
            if args.len() == 3 {
                let _ = bool_value(&args[2])?;
            }
            Ok(())
        }
        "tag" | "json_member" => {
            arity(2, 2)?;
            let _ = text_value(&args[0], MAX_STRING_BYTES)?;
            validate_expr(&args[1], locals, depth + 1, count)
        }
        "gbnf" => {
            arity(2, 2)?;
            validate_expr(&args[0], locals, depth + 1, count)?;
            let _ = text_value(&args[1], MAX_STRING_BYTES)?;
            Ok(())
        }
        "ac" => {
            arity(2, 2)?;
            validate_expr(&args[0], locals, depth + 1, count)?;
            if args[1].is_string() {
                let _ = text_value(&args[1], MAX_STRING_BYTES)?;
            } else {
                let items = args[1]
                    .as_array()
                    .ok_or_else(|| invalid("PEG AC delimiters must be string or array"))?;
                require(
                    items.len() <= MAX_LIST_LEN,
                    "PEG AC delimiter list exceeds bound",
                )?;
                for item in items {
                    let _ = text_value(item, MAX_STRING_BYTES)?;
                }
            }
            Ok(())
        }
        "string_content" => {
            arity(1, 1)?;
            let _ = int_value(&args[0], 0, 255)?;
            Ok(())
        }
        _ => Err(invalid("unknown PEG builder Op")),
    }
}

fn validate_graph(
    graph: &Value,
    mut locals: BTreeSet<String>,
    depth: usize,
    count: &mut usize,
) -> Result<()> {
    require(depth <= MAX_GRAPH_DEPTH, "PEG graph depth exceeds bound")?;
    exact_fields(graph, &["Statements", "Root"])?;
    let statements = array(graph, "Statements")?;
    require(
        statements.len() <= MAX_STATEMENTS,
        "PEG statements exceed bound",
    )?;
    for statement in statements {
        if statement.get("Let").is_some() {
            exact_fields(statement, &["Let", "Expr"])?;
            let _ = text_field(statement, "Let", MAX_LOCAL_NAME)?;
        } else {
            exact_fields(statement, &["Expr"])?;
        }
        validate_expr(field(statement, "Expr")?, &locals, depth + 1, count)?;
        if let Some(let_name) = statement.get("Let").and_then(Value::as_str) {
            locals.insert(let_name.to_owned());
        }
    }
    validate_expr(field(graph, "Root")?, &locals, depth + 1, count)
}

fn parse_input_bytes(bytes_val: &Value) -> Result<Vec<u8>> {
    let arr = bytes_val
        .as_array()
        .ok_or_else(|| invalid("PEG input bytes must be an array"))?;
    require(
        arr.len() <= MAX_STRING_BYTES,
        "PEG input bytes exceed bound",
    )?;
    arr.iter()
        .map(|b| Ok(u8::try_from(int_value(b, 0, 255)?)?))
        .collect()
}

fn validate_operation(operation: &Value) -> Result<()> {
    let kind = text_field(operation, "Kind", MAX_LOCAL_NAME)?;
    match kind {
        "peg-deserialize" => {
            exact_fields(operation, &["Kind", "SerializedJson"])?;
            let json_str = text_field(operation, "SerializedJson", MAX_SERIALIZED_JSON_BYTES)?;
            let _: Value = serde_json::from_str(json_str)?;
            Ok(())
        }
        "peg-serialize" => {
            exact_fields(operation, &["Kind", "Graph"])?;
            let mut count = 0;
            validate_graph(field(operation, "Graph")?, BTreeSet::new(), 0, &mut count)
        }
        "peg-evaluate" => {
            exact_fields(
                operation,
                &["Kind", "Graph", "Inputs", "GenerateGrammar", "RoundTrip"],
            )?;
            let mut count = 0;
            validate_graph(field(operation, "Graph")?, BTreeSet::new(), 0, &mut count)?;
            let inputs = array(operation, "Inputs")?;
            require(inputs.len() <= MAX_LIST_LEN, "PEG inputs exceed bound")?;
            for input in inputs {
                exact_fields(input, &["Bytes", "Flags"])?;
                let _ = parse_input_bytes(field(input, "Bytes")?)?;
                let _ = int_field(input, "Flags", 0, 3)?;
            }
            let modes = array(operation, "GenerateGrammar")?;
            require(
                modes.len() <= MAX_GRAMMAR_MODES,
                "PEG grammar modes exceed bound",
            )?;
            for mode in modes {
                let _ = bool_value(mode)?;
            }
            let _ = bool_value(field(operation, "RoundTrip")?)?;
            Ok(())
        }
        _ => Err(invalid("unknown PEG operation Kind")),
    }
}

fn build_expr(
    builder: &mut ParserBuilder,
    expression: &Value,
    locals: &BTreeMap<String, ParserId>,
) -> Result<ParserId> {
    if let Some(ref_val) = expression.get("Ref") {
        let name = ref_val
            .as_str()
            .ok_or_else(|| invalid("PEG Ref must be string"))?;
        return locals
            .get(name)
            .copied()
            .ok_or_else(|| invalid("unknown PEG local Ref"));
    }
    let op = text_field(expression, "Op", MAX_LOCAL_NAME)?;
    let args = array(expression, "Args")?;
    build_op_expr(builder, op, args, locals)
}

fn build_op_expr(
    builder: &mut ParserBuilder,
    op: &str,
    args: &[Value],
    locals: &BTreeMap<String, ParserId>,
) -> Result<ParserId> {
    match op {
        "eps" => Ok(builder.eps()),
        "end" => Ok(builder.end()),
        "any" => Ok(builder.any()),
        "space" => Ok(builder.space()),
        "double_quoted_string" => Ok(builder.double_quoted_string()),
        "single_quoted_string" => Ok(builder.single_quoted_string()),
        "json" => builder.json(),
        "python_value" => builder.python_value(),
        "marker" => Ok(builder.marker()),
        "literal" => Ok(builder.literal(text_value(&args[0], MAX_STRING_BYTES)?)),
        "ref" => Ok(builder.ref_rule(text_value(&args[0], MAX_STRING_BYTES)?)),
        "until" => Ok(builder.until(text_value(&args[0], MAX_STRING_BYTES)?)),
        "chars" => {
            let pattern = text_value(&args[0], MAX_STRING_BYTES)?;
            let min = if let Some(val) = args.get(1) {
                i32::try_from(int_value(val, 0, 65_536)?)?
            } else {
                1
            };
            let max = if let Some(val) = args.get(2) {
                i32::try_from(int_value(val, -1, 65_536)?)?
            } else {
                -1
            };
            builder.chars(pattern, min, max)
        }
        "string_content" => {
            let quote = u8::try_from(int_value(&args[0], 0, 255)?)?;
            Ok(builder.string_content(quote))
        }
        "until_one_of" => {
            let delims = string_list(&args[0])?;
            Ok(builder.until_one_of(delims))
        }
        "sequence" | "choice" => {
            let items = args[0]
                .as_array()
                .ok_or_else(|| invalid("PEG parser list must be array"))?;
            let mut children = Vec::with_capacity(items.len());
            for item in items {
                children.push(build_expr(builder, item, locals)?);
            }
            if op == "sequence" {
                Ok(builder.sequence(children))
            } else {
                Ok(builder.choice(children))
            }
        }
        "rule-lazy" => {
            let name = text_value(&args[0], MAX_LOCAL_NAME)?.to_owned();
            let trigger = if let Some(val) = args.get(2) {
                bool_value(val)?
            } else {
                false
            };
            let body_graph = &args[1];
            let cloned_locals = locals.clone();
            builder.rule_lazy(name, trigger, |b| build_graph(b, body_graph, cloned_locals))
        }
        "add" | "or" | "space-add" => {
            let left = if args[0].is_string() {
                builder.literal(text_value(&args[0], MAX_STRING_BYTES)?)
            } else {
                build_expr(builder, &args[0], locals)?
            };
            let right = if args[1].is_string() {
                builder.literal(text_value(&args[1], MAX_STRING_BYTES)?)
            } else {
                build_expr(builder, &args[1], locals)?
            };
            match op {
                "add" => Ok(builder.add(left, right)),
                "or" => Ok(builder.or(left, right)),
                _ => Ok(builder.space_add(left, right)),
            }
        }
        _ => build_op_composite(builder, op, args, locals),
    }
}

fn build_op_composite(
    builder: &mut ParserBuilder,
    op: &str,
    args: &[Value],
    locals: &BTreeMap<String, ParserId>,
) -> Result<ParserId> {
    match op {
        "tag" | "json_member" | "rule" => {
            let name = text_value(&args[0], MAX_STRING_BYTES)?;
            let child = build_expr(builder, &args[1], locals)?;
            match op {
                "tag" => Ok(builder.tag(name, child)),
                "json_member" => Ok(builder.json_member(name, child)),
                _ => {
                    let trigger = if let Some(val) = args.get(2) {
                        bool_value(val)?
                    } else {
                        false
                    };
                    Ok(builder.rule(name, child, trigger))
                }
            }
        }
        "one_or_more" => {
            let child = build_expr(builder, &args[0], locals)?;
            Ok(builder.one_or_more(child))
        }
        "zero_or_more" => {
            let child = build_expr(builder, &args[0], locals)?;
            Ok(builder.zero_or_more(child))
        }
        "optional" => {
            let child = build_expr(builder, &args[0], locals)?;
            Ok(builder.optional(child))
        }
        "atomic" => {
            let child = build_expr(builder, &args[0], locals)?;
            Ok(builder.atomic(child))
        }
        "gbnf" => {
            let child = build_expr(builder, &args[0], locals)?;
            let grammar = text_value(&args[1], MAX_STRING_BYTES)?;
            Ok(builder.gbnf(child, grammar))
        }
        "ac" => {
            let child = build_expr(builder, &args[0], locals)?;
            let delims = if args[1].is_string() {
                vec![text_value(&args[1], MAX_STRING_BYTES)?.to_owned()]
            } else {
                string_list(&args[1])?
            };
            Ok(builder.ac(child, delims))
        }
        _ => Err(invalid("unknown PEG builder Op")),
    }
}

fn string_list(value: &Value) -> Result<Vec<String>> {
    let arr = value
        .as_array()
        .ok_or_else(|| invalid("PEG delimiter list must be array"))?;
    arr.iter()
        .map(|entry| Ok(text_value(entry, MAX_STRING_BYTES)?.to_owned()))
        .collect()
}

fn build_graph(
    builder: &mut ParserBuilder,
    graph: &Value,
    mut locals: BTreeMap<String, ParserId>,
) -> Result<ParserId> {
    for statement in array(graph, "Statements")? {
        let value = build_expr(builder, field(statement, "Expr")?, &locals)?;
        if let Some(let_name) = statement.get("Let").and_then(Value::as_str) {
            locals.insert(let_name.to_owned(), value);
        }
    }
    build_expr(builder, field(graph, "Root")?, &locals)
}

fn parse_inputs(arena: &Arena, inputs: &[Value]) -> Result<Vec<Value>> {
    let mut results = Vec::with_capacity(inputs.len());
    for input in inputs {
        let bytes = parse_input_bytes(field(input, "Bytes")?)?;
        let flags = u8::try_from(int_field(input, "Flags", 0, 3)?)?;
        let (result_type, start, end) = arena.parse(&bytes, flags)?;
        results.push(json!({"Type": result_type, "Start": start, "End": end}));
    }
    Ok(results)
}

fn execute_operation(operation: &Value) -> Result<Value> {
    let kind = text_field(operation, "Kind", MAX_LOCAL_NAME)?;
    if kind == "peg-deserialize" {
        let serialized = text_field(operation, "SerializedJson", MAX_SERIALIZED_JSON_BYTES)?;
        let arena = Arena::from_json_str(serialized)?;
        return Ok(json!({"Root": arena.root()}));
    }
    let mut builder = ParserBuilder::new();
    let root = build_graph(&mut builder, field(operation, "Graph")?, BTreeMap::new())?;
    builder.set_root(root);
    let arena = builder.build()?;
    if kind == "peg-serialize" {
        return Ok(json!({"SerializedJson": arena.to_json_string()?}));
    }
    let inputs = array(operation, "Inputs")?;
    let results = parse_inputs(&arena, inputs)?;
    let mut grammars = Vec::new();
    for mode in array(operation, "GenerateGrammar")? {
        let lazy = bool_value(mode)?;
        grammars.push(arena.build_grammar(lazy)?);
    }
    let do_roundtrip = bool_value(field(operation, "RoundTrip")?)?;
    let (serialized, roundtrip) = if do_roundtrip {
        let serialized_str = arena.to_json_string()?;
        let restored = Arena::from_json_str(&serialized_str)?;
        let rt_results = parse_inputs(&restored, inputs)?;
        (Value::String(serialized_str), rt_results)
    } else {
        (Value::Null, Vec::new())
    };
    Ok(json!({
        "Results": results,
        "GrammarTexts": grammars,
        "SerializedJson": serialized,
        "RoundTripResults": roundtrip,
    }))
}

/// Returns true when every operation in `request` is a supported PEG transport kind.
pub(super) fn handles(request: &Value) -> bool {
    request["Operations"].as_array().is_some_and(|operations| {
        !operations.is_empty()
            && operations.iter().all(|operation| {
                matches!(
                    operation["Kind"].as_str(),
                    Some("peg-evaluate" | "peg-serialize" | "peg-deserialize")
                )
            })
    })
}

/// Validates and executes a PEG wire request.
pub(super) fn execute(request: &Value) -> Result<Value> {
    exact_fields(request, &["SchemaVersion", "Operations"])?;
    if int_field(request, "SchemaVersion", 1, 1)? != 1 {
        return Err(invalid("PEG request SchemaVersion must equal 1"));
    }
    let operations = array(request, "Operations")?;
    require(
        !operations.is_empty() && operations.len() <= MAX_OPERATIONS,
        "PEG Operations must contain 1 through 128 operations",
    )?;
    for operation in operations {
        validate_operation(operation)?;
    }
    let results = operations
        .iter()
        .map(execute_operation)
        .collect::<Result<Vec<_>>>()?;
    Ok(json!({"SchemaVersion": 1, "Results": results}))
}

/// Provider identity metadata for the independent PEG reference implementation.
pub(super) fn identity() -> Value {
    json!({
        "Kind": "independent-rust-peg",
        "SupportedSourceIds": IDS,
        "UpstreamCommit": super::UPSTREAM,
        "AssertionSources": [
            {"Path": IDS[0], "Sha256": "24c5c31c65b2f5fc88919a4b5d06506ae644f8f0460c14c0134cd5439c5a7694"},
            {"Path": IDS[1], "Sha256": "d14973bb98faaa87d1e17af30b76d413e2ac6541ab800c20eb55d48c8fa1952a"},
            {"Path": IDS[2], "Sha256": "2731298bf7f28047852e4f420243fc533bdb1408611257f47a5fff2adacb36fc"},
            {"Path": IDS[3], "Sha256": "276f50faa0f1f50da7d5289b3e13110c26cf9120ff23bbdb4a6a8c12c384a028"},
            {"Path": IDS[4], "Sha256": "ab6b20f141fed193bc93e810608c1b6800415d14ff832ad3ff84671594d18d4f"},
            {"Path": IDS[5], "Sha256": "5eda6c6457af1ff1a62081dadac597d7706bb656e9941bb828de6421d9c563cf"},
            {"Path": IDS[6], "Sha256": "52ebb967feae3aec877a1d3cb1c509b2598a9c799e60401a82253a5d4844038b"},
        ],
        "Architecture": "clean-room-peg-arena-backtracking-parser-gbnf-and-json-codec-v1",
        "NativeReferenceReplay": false,
        "ProductionRuntimeCalled": false,
        "CompleteSourceGate": false,
        "NativeRetirementQualified": false
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, Target};

    #[test]
    fn independent_peg_provider_passes_all_seven_suites_on_rust_reference_and_cpp() -> Result<()> {
        type SuiteEntry = (fn(&Engine) -> Result<Value>, &'static str);
        let suites: [SuiteEntry; 7] = [
            (crate::peg::run_basic, IDS[0]),
            (crate::peg::run_gbnf, IDS[1]),
            (crate::peg::run_json, IDS[2]),
            (crate::peg::run_serialization, IDS[3]),
            (crate::peg::run_python_dict, IDS[4]),
            (crate::peg::run_unicode, IDS[5]),
            (crate::peg::run_all, IDS[6]),
        ];
        for target in [Target::RustReference, Target::Cpp] {
            let engine = Engine::new(target, None, None, None, None)?;
            for (runner, expected_id) in suites {
                let report = runner(&engine)?;
                assert_eq!(report["Id"], expected_id);
                assert_eq!(report["Status"], "pass", "{report}");
            }
        }
        for id in IDS {
            assert!(crate::reference_provider::supports(id));
        }
        let meta = identity();
        assert_eq!(meta["NativeReferenceReplay"], false);
        assert_eq!(meta["ProductionRuntimeCalled"], false);
        Ok(())
    }

    #[test]
    fn independent_peg_provider_rejects_invalid_ops_refs_flags_and_malformed_inputs() -> Result<()>
    {
        let valid_eval = json!({
            "SchemaVersion": 1,
            "Operations": [{
                "Kind": "peg-evaluate",
                "Graph": {"Statements": [], "Root": {"Op": "literal", "Args": ["abc"]}},
                "Inputs": [{"Bytes": [97, 98, 99], "Flags": 0}],
                "GenerateGrammar": [false],
                "RoundTrip": true
            }]
        });
        assert!(execute(&valid_eval).is_ok());

        // Unknown Op rejected on peg-evaluate wire protocol.
        let mut bad_op = valid_eval.clone();
        bad_op["Operations"][0]["Graph"]["Root"] = json!({"Op": "nonexistent_op", "Args": []});
        assert!(execute(&bad_op).is_err());

        // Chat-only ops rejected on peg-evaluate wire protocol.
        let mut chat_only = valid_eval.clone();
        chat_only["Operations"][0]["Graph"]["Root"] =
            json!({"Op": "peek", "Args": [{"Op": "eps", "Args": []}]});
        assert!(execute(&chat_only).is_err());

        // Unknown local Ref rejected.
        let mut bad_local = valid_eval.clone();
        bad_local["Operations"][0]["Graph"]["Root"] = json!({"Ref": "missing"});
        assert!(execute(&bad_local).is_err());

        // Undefined rule ref rejected at build time.
        let mut bad_rule_ref = valid_eval.clone();
        bad_rule_ref["Operations"][0]["Graph"]["Root"] =
            json!({"Op": "ref", "Args": ["missing_rule"]});
        assert!(execute(&bad_rule_ref).is_err());

        // Invalid flags (out of 0..=3) rejected.
        let mut bad_flags = valid_eval.clone();
        bad_flags["Operations"][0]["Inputs"][0]["Flags"] = json!(4);
        assert!(execute(&bad_flags).is_err());

        // Invalid byte (> 255) rejected.
        let mut bad_byte = valid_eval;
        bad_byte["Operations"][0]["Inputs"][0]["Bytes"] = json!([256]);
        assert!(execute(&bad_byte).is_err());

        // Malformed serialized JSON rejected on peg-deserialize.
        let bad_deser = json!({
            "SchemaVersion": 1,
            "Operations": [{"Kind": "peg-deserialize", "SerializedJson": "{\"version\":1,\"root\":99,\"rules\":{},\"parsers\":[{\"type\":\"eps\"}]}"}]
        });
        assert!(execute(&bad_deser).is_err());

        // Pure unguarded cycle fails cleanly without stack overflow.
        let cycle = json!({
            "SchemaVersion": 1,
            "Operations": [{
                "Kind": "peg-evaluate",
                "Graph": {
                    "Statements": [],
                    "Root": {"Op": "rule", "Args": ["loop", {"Op": "ref", "Args": ["loop"]}]}
                },
                "Inputs": [{"Bytes": [97], "Flags": 0}],
                "GenerateGrammar": [false, true],
                "RoundTrip": true
            }]
        });
        let cycle_res = execute(&cycle)?;
        assert_eq!(cycle_res["Results"][0]["Results"][0]["Type"], 0);
        Ok(())
    }

    #[test]
    fn independent_peg_builder_ast_and_stage_helpers_work_cleanly() -> Result<()> {
        let mut builder = ParserBuilder::new();
        let kw_a = builder.literal("alpha");
        let kw_b = builder.literal("beta");
        let choice = builder.or(kw_a, kw_b);
        let kw_c = builder.literal("gamma");
        let extended = builder.or_assign(choice, kw_c);
        let rep = builder.repeat(extended, 1, 2)?;
        let st = builder.start();
        let pk_src = builder.literal("!");
        let pk = builder.peek(pk_src);
        let tagged = builder.tag("word", rep);
        let rest_p = builder.rest();
        let rest_tag = builder.tag("tail", rest_p);
        let root_seq = builder.sequence(vec![st, tagged, pk, rest_tag]);
        let root_rule = builder.rule("entry", root_seq, false);
        builder.set_root(root_rule);
        let arena = builder.build()?;

        let mut ctx = ParseContext::new(b"alphabeta!extra", parse_flags::NONE);
        let res = arena.parse_with_context(&mut ctx)?;
        assert_eq!(res.result_type, ParseResultType::Success);
        assert_eq!(res.start, 0);
        assert_eq!(res.end, 15);
        assert_ne!(ctx.ast.nodes(), []);
        let root_node = ctx
            .ast
            .get(res.nodes[0])
            .ok_or_else(|| invalid("missing root AST node"))?;
        assert_eq!(root_node.rule, "entry");
        let mut visited_tags = Vec::new();
        ctx.ast.visit(&res.nodes, &mut |node| {
            if !node.tag.is_empty() {
                visited_tags.push((node.tag.clone(), node.text.clone()));
            }
        });
        assert_eq!(
            visited_tags,
            vec![
                ("word".to_owned(), "alphabeta".to_owned()),
                ("tail".to_owned(), "!extra".to_owned()),
            ]
        );

        let mut anywhere_builder = ParserBuilder::new();
        let target_lit = anywhere_builder.literal("needle");
        let target_tag = anywhere_builder.tag("found", target_lit);
        anywhere_builder.set_root(target_tag);
        let anywhere_arena = anywhere_builder.build()?;
        let mut anywhere_ctx = ParseContext::new(b"haystack needle end", parse_flags::NONE);
        let anywhere_res = anywhere_arena.parse_anywhere(&mut anywhere_ctx)?;
        assert_eq!(anywhere_res.result_type, ParseResultType::Success);
        assert_eq!(anywhere_res.start, 9);
        assert_eq!(anywhere_res.end, 15);

        // Verify peg-serialize and peg-deserialize roundtrip preserves Root.
        let serialize_req = json!({
            "SchemaVersion": 1,
            "Operations": [{
                "Kind": "peg-serialize",
                "Graph": {"Statements": [], "Root": {"Op": "json", "Args": []}}
            }]
        });
        let serialize_reply = execute(&serialize_req)?;
        let serialized = serialize_reply["Results"][0]["SerializedJson"]
            .as_str()
            .ok_or_else(|| invalid("missing SerializedJson"))?;
        let deserialize_req = json!({
            "SchemaVersion": 1,
            "Operations": [{"Kind": "peg-deserialize", "SerializedJson": serialized}]
        });
        let deserialize_reply = execute(&deserialize_req)?;
        assert!(deserialize_reply["Results"][0]["Root"].is_u64());
        Ok(())
    }
}
