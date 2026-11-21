//! Clean-room byte-level Earley CFG engine, UTF-8 regex compiler, and Lark grammar parser.
//!
//! All regexes, Unicode character ranges, and Lark productions lower to byte-level CFG rules
//! processed incrementally across BPE byte tries without executing external parser crates.

use std::collections::{BTreeMap, HashSet};

use super::super::{Result, invalid};
use super::json_ast::parse_ordered_json_prefix;

#[path = "grammar_schema.rs"]
pub(super) mod schema;

const MAX_RULES: usize = 16_384;
const MAX_REPEAT_BOUND: usize = 512;

/// 256-bit terminal byte class.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub(super) struct ByteMask(pub(super) [u64; 4]);

impl ByteMask {
    pub(super) const fn empty() -> Self {
        Self([0; 4])
    }

    pub(super) const fn single(byte: u8) -> Self {
        let mut mask = Self::empty();
        mask.set(byte);
        mask
    }

    pub(super) const fn range(lo: u8, hi: u8) -> Self {
        let mut mask = Self::empty();
        let mut b = lo;
        loop {
            mask.set(b);
            if b == hi {
                break;
            }
            b += 1;
        }
        mask
    }

    const fn word_index(byte: u8) -> usize {
        match byte >> 6 {
            0 => 0,
            1 => 1,
            2 => 2,
            _ => 3,
        }
    }

    pub(super) const fn set(&mut self, byte: u8) {
        self.0[Self::word_index(byte)] |= 1_u64 << (byte & 63);
    }

    pub(super) const fn contains(self, byte: u8) -> bool {
        (self.0[Self::word_index(byte)] & (1_u64 << (byte & 63))) != 0
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Symbol {
    Terminal(ByteMask),
    NonTerm(u32),
}

#[derive(Clone)]
struct Production {
    lhs: u32,
    rhs: Vec<Symbol>,
}

#[derive(Clone)]
pub(super) struct Grammar {
    start: u32,
    prods: Vec<Production>,
    rules_by_lhs: Vec<Vec<u32>>,
    nullable: Vec<bool>,
}

pub(super) struct Builder {
    prods: Vec<Production>,
    rule_count: u32,
    named: BTreeMap<String, u32>,
    bytes_cache: BTreeMap<u8, u32>,
}

impl Builder {
    pub(super) fn new() -> Self {
        Self {
            prods: Vec::new(),
            rule_count: 0,
            named: BTreeMap::new(),
            bytes_cache: BTreeMap::new(),
        }
    }

    pub(super) fn new_rule(&mut self) -> Result<u32> {
        let id = self.rule_count;
        if usize::try_from(id)? >= MAX_RULES {
            return Err(invalid("LLG grammar rule count exceeded bound"));
        }
        self.rule_count += 1;
        Ok(id)
    }

    pub(super) fn intern_named(&mut self, name: &str) -> Result<u32> {
        if let Some(&id) = self.named.get(name) {
            return Ok(id);
        }
        let id = self.new_rule()?;
        self.named.insert(name.to_owned(), id);
        Ok(id)
    }

    pub(super) fn add_prod(&mut self, lhs: u32, rhs: Vec<Symbol>) {
        self.prods.push(Production { lhs, rhs });
    }

    pub(super) fn literal_bytes(&mut self, bytes: &[u8]) -> Result<u32> {
        let id = self.new_rule()?;
        let rhs = bytes
            .iter()
            .map(|&b| Symbol::Terminal(ByteMask::single(b)))
            .collect();
        self.add_prod(id, rhs);
        Ok(id)
    }

    pub(super) fn literal_str(&mut self, text: &str) -> Result<u32> {
        self.literal_bytes(text.as_bytes())
    }

    pub(super) fn single_mask(&mut self, mask: ByteMask) -> Result<u32> {
        let id = self.new_rule()?;
        self.add_prod(id, vec![Symbol::Terminal(mask)]);
        Ok(id)
    }

    pub(super) fn byte_rule(&mut self, byte: u8) -> Result<u32> {
        if let Some(&id) = self.bytes_cache.get(&byte) {
            return Ok(id);
        }
        let id = self.single_mask(ByteMask::single(byte))?;
        self.bytes_cache.insert(byte, id);
        Ok(id)
    }

    /// Left-recursive zero-or-more repetition (`R -> eps | R item`) keeps Earley O(n).
    pub(super) fn zero_or_more(&mut self, item: Symbol) -> Result<u32> {
        let id = self.new_rule()?;
        self.add_prod(id, Vec::new());
        self.add_prod(id, vec![Symbol::NonTerm(id), item]);
        Ok(id)
    }

    /// Left-recursive one-or-more repetition (`R -> item | R item`) keeps Earley O(n).
    pub(super) fn one_or_more(&mut self, item: Symbol) -> Result<u32> {
        let id = self.new_rule()?;
        self.add_prod(id, vec![item]);
        self.add_prod(id, vec![Symbol::NonTerm(id), item]);
        Ok(id)
    }

    pub(super) fn optional(&mut self, item: Symbol) -> Result<u32> {
        let id = self.new_rule()?;
        self.add_prod(id, Vec::new());
        self.add_prod(id, vec![item]);
        Ok(id)
    }

    pub(super) fn utf8_scalar_ranges(&mut self, ranges: &[(u32, u32)]) -> Result<u32> {
        let rule = self.new_rule()?;
        let normalized = normalize_scalar_ranges(ranges);
        for (lo, hi) in normalized {
            emit_utf8_range(self, rule, lo, hi)?;
        }
        Ok(rule)
    }

    pub(super) fn build(self, start: u32) -> Result<Grammar> {
        let count = usize::try_from(self.rule_count)?;
        let mut rules_by_lhs = vec![Vec::new(); count];
        for (idx, prod) in self.prods.iter().enumerate() {
            let pid = u32::try_from(idx)?;
            let lhs = usize::try_from(prod.lhs)?;
            if lhs >= count {
                return Err(invalid(
                    "LLG grammar production references invalid nonterminal",
                ));
            }
            rules_by_lhs[lhs].push(pid);
        }
        let mut nullable = vec![false; count];
        loop {
            let mut changed = false;
            for prod in &self.prods {
                let lhs = usize::try_from(prod.lhs)?;
                if !nullable[lhs]
                    && prod.rhs.iter().all(|sym| match sym {
                        Symbol::Terminal(_) => false,
                        Symbol::NonTerm(child) => {
                            usize::try_from(*child).is_ok_and(|u| nullable[u])
                        }
                    })
                {
                    nullable[lhs] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        Ok(Grammar {
            start,
            prods: self.prods,
            rules_by_lhs,
            nullable,
        })
    }
}

fn normalize_scalar_ranges(ranges: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut parts = Vec::new();
    for &(lo, hi) in ranges {
        if lo > hi {
            continue;
        }
        let lo = lo.min(0x10_ffff);
        let hi = hi.min(0x10_ffff);
        if hi < 0xd800 || lo > 0xdfff {
            parts.push((lo, hi));
        } else {
            if lo < 0xd800 {
                parts.push((lo, 0xd7ff));
            }
            if hi > 0xdfff {
                parts.push((0xe000, hi));
            }
        }
    }
    parts.sort_unstable();
    let mut merged: Vec<(u32, u32)> = Vec::new();
    for (lo, hi) in parts {
        if let Some(last) = merged.last_mut()
            && lo <= last.1.saturating_add(1)
        {
            last.1 = last.1.max(hi);
        } else {
            merged.push((lo, hi));
        }
    }
    merged
}

fn negate_scalar_ranges(ranges: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let excluded = normalize_scalar_ranges(ranges);
    let mut out = Vec::new();
    let mut cursor = 0_u32;
    for (lo, hi) in excluded {
        if cursor < lo {
            out.push((cursor, lo - 1));
        }
        if hi == 0x10_ffff {
            return normalize_scalar_ranges(&out);
        }
        cursor = hi + 1;
    }
    if cursor <= 0x10_ffff {
        out.push((cursor, 0x10_ffff));
    }
    normalize_scalar_ranges(&out)
}

fn emit_utf8_range(builder: &mut Builder, lhs: u32, lo: u32, hi: u32) -> Result<()> {
    const BRACKETS: [(u32, u32); 9] = [
        (0x0000, 0x007f),
        (0x0080, 0x07ff),
        (0x0800, 0x0fff),
        (0x1000, 0xcfff),
        (0xd000, 0xd7ff),
        (0xe000, 0xffff),
        (0x1_0000, 0x3_ffff),
        (0x4_0000, 0xf_ffff),
        (0x10_0000, 0x10_ffff),
    ];
    for (blo, bhi) in BRACKETS {
        let s = lo.max(blo);
        let e = hi.min(bhi);
        if s <= e {
            let mut sb = [0_u8; 4];
            let mut eb = [0_u8; 4];
            let sc =
                char::from_u32(s).ok_or_else(|| invalid("invalid scalar start in UTF-8 range"))?;
            let ec =
                char::from_u32(e).ok_or_else(|| invalid("invalid scalar end in UTF-8 range"))?;
            let sslice = sc.encode_utf8(&mut sb).as_bytes();
            let eslice = ec.encode_utf8(&mut eb).as_bytes();
            let mut prefix = Vec::new();
            emit_utf8_boxes(builder, lhs, &mut prefix, sslice, eslice);
        }
    }
    Ok(())
}

fn emit_utf8_boxes(
    builder: &mut Builder,
    lhs: u32,
    prefix: &mut Vec<Symbol>,
    lo: &[u8],
    hi: &[u8],
) {
    if lo.len() == 1 {
        let mut rhs = prefix.clone();
        rhs.push(Symbol::Terminal(ByteMask::range(lo[0], hi[0])));
        builder.add_prod(lhs, rhs);
        return;
    }
    if lo[0] == hi[0] {
        prefix.push(Symbol::Terminal(ByteMask::single(lo[0])));
        emit_utf8_boxes(builder, lhs, prefix, &lo[1..], &hi[1..]);
        prefix.pop();
        return;
    }
    let tail_len = lo.len() - 1;
    let all_cont_min = lo[1..].iter().all(|&b| b == 0x80);
    let all_cont_max = hi[1..].iter().all(|&b| b == 0xbf);
    let mut mid_start = lo[0];
    if !all_cont_min {
        let hi_fill = [0xbf_u8; 3];
        prefix.push(Symbol::Terminal(ByteMask::single(lo[0])));
        emit_utf8_boxes(builder, lhs, prefix, &lo[1..], &hi_fill[..tail_len]);
        prefix.pop();
        mid_start += 1;
    }
    let mut mid_end = hi[0];
    if !all_cont_max {
        mid_end -= 1;
    }
    if mid_start <= mid_end {
        let mut rhs = prefix.clone();
        rhs.push(Symbol::Terminal(ByteMask::range(mid_start, mid_end)));
        for _ in 0..tail_len {
            rhs.push(Symbol::Terminal(ByteMask::range(0x80, 0xbf)));
        }
        builder.add_prod(lhs, rhs);
    }
    if !all_cont_max {
        let lo_fill = [0x80_u8; 3];
        prefix.push(Symbol::Terminal(ByteMask::single(hi[0])));
        emit_utf8_boxes(builder, lhs, prefix, &lo_fill[..tail_len], &hi[1..]);
        prefix.pop();
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct EarleyItem {
    prod: u32,
    dot: u16,
    origin: u32,
}

#[derive(Clone)]
struct Column {
    items: Vec<EarleyItem>,
    seen: HashSet<EarleyItem>,
    accepts_full: bool,
}

impl Column {
    fn new() -> Self {
        Self {
            items: Vec::new(),
            seen: HashSet::new(),
            accepts_full: false,
        }
    }

    fn push(&mut self, item: EarleyItem) -> bool {
        if self.seen.insert(item) {
            self.items.push(item);
            true
        } else {
            false
        }
    }
}

/// Incremental Earley recognizer supporting O(1) push/pop single-byte exploration.
#[derive(Clone)]
pub(super) struct EarleyState<'a> {
    grammar: &'a Grammar,
    columns: Vec<Column>,
}

impl<'a> EarleyState<'a> {
    pub(super) fn new(grammar: &'a Grammar) -> Result<Self> {
        let mut initial = Column::new();
        let start = usize::try_from(grammar.start)?;
        for &pid in &grammar.rules_by_lhs[start] {
            initial.push(EarleyItem {
                prod: pid,
                dot: 0,
                origin: 0,
            });
        }
        let mut state = Self {
            grammar,
            columns: vec![initial],
        };
        state.close_top()?;
        Ok(state)
    }

    pub(super) fn push_byte(&mut self, byte: u8) -> Result<bool> {
        let depth = u32::try_from(self.columns.len() - 1)?;
        let prev = &self.columns[usize::try_from(depth)?];
        let mut next = Column::new();
        for &item in &prev.items {
            let rhs = &self.grammar.prods[usize::try_from(item.prod)?].rhs;
            if let Some(&Symbol::Terminal(mask)) = rhs.get(usize::from(item.dot))
                && mask.contains(byte)
            {
                next.push(EarleyItem {
                    prod: item.prod,
                    dot: item.dot + 1,
                    origin: item.origin,
                });
            }
        }
        if next.items.is_empty() {
            return Ok(false);
        }
        self.columns.push(next);
        self.close_top()?;
        Ok(true)
    }

    pub(super) fn pop_byte(&mut self) {
        if self.columns.len() > 1 {
            self.columns.pop();
        }
    }

    pub(super) fn accepts_full(&self) -> bool {
        self.columns.last().is_some_and(|col| col.accepts_full)
    }

    fn close_top(&mut self) -> Result<()> {
        let col_idx = self.columns.len() - 1;
        let col_u32 = u32::try_from(col_idx)?;
        let mut cursor = 0_usize;
        while cursor < self.columns[col_idx].items.len() {
            let item = self.columns[col_idx].items[cursor];
            cursor += 1;
            let prod = &self.grammar.prods[usize::try_from(item.prod)?];
            let dot = usize::from(item.dot);
            if dot >= prod.rhs.len() {
                if prod.lhs == self.grammar.start && item.origin == 0 {
                    self.columns[col_idx].accepts_full = true;
                }
                let origin = usize::try_from(item.origin)?;
                let mut to_push = Vec::new();
                for &waiter in &self.columns[origin].items {
                    let wrhs = &self.grammar.prods[usize::try_from(waiter.prod)?].rhs;
                    if wrhs.get(usize::from(waiter.dot)) == Some(&Symbol::NonTerm(prod.lhs)) {
                        to_push.push(EarleyItem {
                            prod: waiter.prod,
                            dot: waiter.dot + 1,
                            origin: waiter.origin,
                        });
                    }
                }
                for next_item in to_push {
                    self.columns[col_idx].push(next_item);
                }
            } else if let Symbol::NonTerm(child) = prod.rhs[dot] {
                let child_idx = usize::try_from(child)?;
                for &pid in &self.grammar.rules_by_lhs[child_idx] {
                    self.columns[col_idx].push(EarleyItem {
                        prod: pid,
                        dot: 0,
                        origin: col_u32,
                    });
                }
                if self.grammar.nullable[child_idx] {
                    self.columns[col_idx].push(EarleyItem {
                        prod: item.prod,
                        dot: item.dot + 1,
                        origin: item.origin,
                    });
                }
            }
        }
        Ok(())
    }
}

/// Regex-to-Grammar compiler supporting character classes, Unicode `.`, groups, and `{m,n}`.
struct RegexCompiler<'a> {
    src: &'a [u8],
    pos: usize,
    builder: &'a mut Builder,
}

impl RegexCompiler<'_> {
    fn compile(builder: &mut Builder, pattern: &str) -> Result<u32> {
        let mut compiler = RegexCompiler {
            src: pattern.as_bytes(),
            pos: 0,
            builder,
        };
        let rule = compiler.parse_alt()?;
        if compiler.pos != compiler.src.len() {
            return Err(invalid("unexpected trailing token in regex pattern"));
        }
        Ok(rule)
    }

    fn parse_alt(&mut self) -> Result<u32> {
        let first = self.parse_seq()?;
        if self.src.get(self.pos) != Some(&b'|') {
            return Ok(first);
        }
        let rule = self.builder.new_rule()?;
        self.builder.add_prod(rule, vec![Symbol::NonTerm(first)]);
        while self.src.get(self.pos) == Some(&b'|') {
            self.pos += 1;
            let next = self.parse_seq()?;
            self.builder.add_prod(rule, vec![Symbol::NonTerm(next)]);
        }
        Ok(rule)
    }

    fn parse_seq(&mut self) -> Result<u32> {
        let mut items = Vec::new();
        while let Some(&b) = self.src.get(self.pos) {
            if b == b')' || b == b'|' {
                break;
            }
            items.push(self.parse_quantified()?);
        }
        if items.len() == 1
            && let Symbol::NonTerm(id) = items[0]
        {
            return Ok(id);
        }
        let rule = self.builder.new_rule()?;
        self.builder.add_prod(rule, items);
        Ok(rule)
    }

    fn parse_quantified(&mut self) -> Result<Symbol> {
        let atom = self.parse_atom()?;
        match self.src.get(self.pos).copied() {
            Some(b'*') => {
                self.pos += 1;
                Ok(Symbol::NonTerm(self.builder.zero_or_more(atom)?))
            }
            Some(b'+') => {
                self.pos += 1;
                Ok(Symbol::NonTerm(self.builder.one_or_more(atom)?))
            }
            Some(b'?') => {
                self.pos += 1;
                Ok(Symbol::NonTerm(self.builder.optional(atom)?))
            }
            Some(b'{') => self.parse_brace_quantifier(atom),
            _ => Ok(atom),
        }
    }

    fn parse_brace_quantifier(&mut self, atom: Symbol) -> Result<Symbol> {
        self.pos += 1;
        let min = self.read_uint()?;
        let max = if self.src.get(self.pos) == Some(&b',') {
            self.pos += 1;
            if self.src.get(self.pos) == Some(&b'}') {
                None
            } else {
                Some(self.read_uint()?)
            }
        } else {
            Some(min)
        };
        if self.src.get(self.pos) != Some(&b'}') {
            return Err(invalid("invalid regex brace quantifier"));
        }
        self.pos += 1;
        if min > MAX_REPEAT_BOUND || max.is_some_and(|m| m > MAX_REPEAT_BOUND || m < min) {
            return Err(invalid("regex quantifier bounds invalid or too large"));
        }
        let rule = self.builder.new_rule()?;
        let mut rhs = vec![atom; min];
        if let Some(m) = max {
            let opt = repeat_optional(self.builder, atom, m - min)?;
            rhs.push(Symbol::NonTerm(opt));
        } else {
            let tail = self.builder.zero_or_more(atom)?;
            rhs.push(Symbol::NonTerm(tail));
        }
        self.builder.add_prod(rule, rhs);
        Ok(Symbol::NonTerm(rule))
    }

    fn read_uint(&mut self) -> Result<usize> {
        let start = self.pos;
        while self.src.get(self.pos).is_some_and(u8::is_ascii_digit) {
            self.pos += 1;
        }
        if self.pos == start {
            return Err(invalid("expected unsigned integer in quantifier"));
        }
        let s = std::str::from_utf8(&self.src[start..self.pos])
            .map_err(|_| invalid("non-UTF8 quantifier"))?;
        Ok(s.parse::<usize>()?)
    }

    fn parse_atom(&mut self) -> Result<Symbol> {
        let b = *self
            .src
            .get(self.pos)
            .ok_or_else(|| invalid("unexpected end of regex atom"))?;
        match b {
            b'(' => {
                self.pos += 1;
                if self.src[self.pos..].starts_with(b"?:") {
                    self.pos += 2;
                }
                let inner = self.parse_alt()?;
                if self.src.get(self.pos) != Some(&b')') {
                    return Err(invalid("unclosed '(' in regex"));
                }
                self.pos += 1;
                Ok(Symbol::NonTerm(inner))
            }
            b'[' => {
                let ranges = self.parse_class()?;
                Ok(Symbol::NonTerm(self.builder.utf8_scalar_ranges(&ranges)?))
            }
            b'.' => {
                self.pos += 1;
                let rule = self
                    .builder
                    .utf8_scalar_ranges(&[(0, 9), (11, 0x10_ffff)])?;
                Ok(Symbol::NonTerm(rule))
            }
            b'\\' => self.parse_escape_atom(),
            _ => {
                let rest = std::str::from_utf8(&self.src[self.pos..])
                    .map_err(|_| invalid("invalid UTF-8 in regex literal"))?;
                let ch = rest
                    .chars()
                    .next()
                    .ok_or_else(|| invalid("empty regex literal"))?;
                self.pos += ch.len_utf8();
                let mut buf = [0_u8; 4];
                let rule = self
                    .builder
                    .literal_bytes(ch.encode_utf8(&mut buf).as_bytes())?;
                Ok(Symbol::NonTerm(rule))
            }
        }
    }

    fn parse_escape_atom(&mut self) -> Result<Symbol> {
        self.pos += 1;
        let ch = self.next_escaped_char()?;
        match ch {
            'd' => Ok(Symbol::NonTerm(
                self.builder
                    .utf8_scalar_ranges(&[(u32::from(b'0'), u32::from(b'9'))])?,
            )),
            'D' => Ok(Symbol::NonTerm(self.builder.utf8_scalar_ranges(
                &negate_scalar_ranges(&[(u32::from(b'0'), u32::from(b'9'))]),
            )?)),
            's' => Ok(Symbol::NonTerm(
                self.builder.utf8_scalar_ranges(&whitespace_ranges())?,
            )),
            'S' => Ok(Symbol::NonTerm(self.builder.utf8_scalar_ranges(
                &negate_scalar_ranges(&whitespace_ranges()),
            )?)),
            'w' => Ok(Symbol::NonTerm(
                self.builder.utf8_scalar_ranges(&word_ranges())?,
            )),
            'W' => Ok(Symbol::NonTerm(
                self.builder
                    .utf8_scalar_ranges(&negate_scalar_ranges(&word_ranges()))?,
            )),
            _ => {
                let mut buf = [0_u8; 4];
                let rule = self
                    .builder
                    .literal_bytes(ch.encode_utf8(&mut buf).as_bytes())?;
                Ok(Symbol::NonTerm(rule))
            }
        }
    }

    fn parse_class(&mut self) -> Result<Vec<(u32, u32)>> {
        self.pos += 1;
        let negated = self.src.get(self.pos) == Some(&b'^');
        if negated {
            self.pos += 1;
        }
        let mut items: Vec<ClassItem> = Vec::new();
        let mut first = true;
        while let Some(&b) = self.src.get(self.pos) {
            if b == b']' && !first {
                self.pos += 1;
                let mut ranges = Vec::new();
                let mut idx = 0_usize;
                while idx < items.len() {
                    if idx + 2 < items.len()
                        && matches!(items[idx + 1], ClassItem::Char('-'))
                        && let (&ClassItem::Char(start), &ClassItem::Char(end)) =
                            (&items[idx], &items[idx + 2])
                    {
                        if start > end {
                            return Err(invalid("inverted character range in regex class"));
                        }
                        ranges.push((u32::from(start), u32::from(end)));
                        idx += 3;
                        continue;
                    }
                    match &items[idx] {
                        ClassItem::Char(c) => ranges.push((u32::from(*c), u32::from(*c))),
                        ClassItem::Multi(list) => ranges.extend_from_slice(list),
                    }
                    idx += 1;
                }
                return Ok(if negated {
                    negate_scalar_ranges(&ranges)
                } else {
                    normalize_scalar_ranges(&ranges)
                });
            }
            first = false;
            if b == b'\\' {
                self.pos += 1;
                let ch = self.next_escaped_char()?;
                match ch {
                    'd' => items.push(ClassItem::Multi(vec![(u32::from(b'0'), u32::from(b'9'))])),
                    'D' => items.push(ClassItem::Multi(negate_scalar_ranges(&[(
                        u32::from(b'0'),
                        u32::from(b'9'),
                    )]))),
                    's' => items.push(ClassItem::Multi(whitespace_ranges())),
                    'S' => items.push(ClassItem::Multi(negate_scalar_ranges(&whitespace_ranges()))),
                    'w' => items.push(ClassItem::Multi(word_ranges())),
                    'W' => items.push(ClassItem::Multi(negate_scalar_ranges(&word_ranges()))),
                    other => items.push(ClassItem::Char(other)),
                }
            } else {
                let rest = std::str::from_utf8(&self.src[self.pos..])
                    .map_err(|_| invalid("invalid UTF-8 in regex class"))?;
                let ch = rest
                    .chars()
                    .next()
                    .ok_or_else(|| invalid("empty char in regex class"))?;
                self.pos += ch.len_utf8();
                items.push(ClassItem::Char(ch));
            }
        }
        Err(invalid("unterminated regex character class"))
    }

    fn next_escaped_char(&mut self) -> Result<char> {
        let byte = *self
            .src
            .get(self.pos)
            .ok_or_else(|| invalid("dangling regex escape"))?;
        self.pos += 1;
        match byte {
            b'n' => Ok('\n'),
            b'r' => Ok('\r'),
            b't' => Ok('\t'),
            b'x' => {
                let hex_bytes = self
                    .src
                    .get(self.pos..self.pos + 2)
                    .ok_or_else(|| invalid("short \\x escape"))?;
                let s = std::str::from_utf8(hex_bytes).map_err(|_| invalid("invalid hex"))?;
                self.pos += 2;
                let val = u8::from_str_radix(s, 16)?;
                Ok(char::from(val))
            }
            b'u' => {
                let hex_bytes = self
                    .src
                    .get(self.pos..self.pos + 4)
                    .ok_or_else(|| invalid("short \\u escape"))?;
                let s = std::str::from_utf8(hex_bytes).map_err(|_| invalid("invalid hex"))?;
                self.pos += 4;
                let val = u32::from_str_radix(s, 16)?;
                char::from_u32(val).ok_or_else(|| invalid("invalid unicode scalar escape"))
            }
            _ => Ok(char::from(byte)),
        }
    }
}

enum ClassItem {
    Char(char),
    Multi(Vec<(u32, u32)>),
}

fn whitespace_ranges() -> Vec<(u32, u32)> {
    vec![(9, 13), (32, 32)]
}

fn word_ranges() -> Vec<(u32, u32)> {
    vec![
        (u32::from(b'0'), u32::from(b'9')),
        (u32::from(b'A'), u32::from(b'Z')),
        (u32::from(b'_'), u32::from(b'_')),
        (u32::from(b'a'), u32::from(b'z')),
    ]
}

pub(super) fn repeat_optional(builder: &mut Builder, atom: Symbol, count: usize) -> Result<u32> {
    let mut next = builder.new_rule()?;
    builder.add_prod(next, Vec::new());
    for _ in 0..count {
        let cur = builder.new_rule()?;
        builder.add_prod(cur, Vec::new());
        builder.add_prod(cur, vec![atom, Symbol::NonTerm(next)]);
        next = cur;
    }
    Ok(next)
}

pub(super) fn compile_regex(builder: &mut Builder, pattern: &str) -> Result<u32> {
    RegexCompiler::compile(builder, pattern)
}

/// Parses an inline Lark grammar (`start: ...`, `%llguidance {}`, `%json ...`) into a `Grammar`.
pub(super) fn compile_lark(source: &str) -> Result<Grammar> {
    let mut builder = Builder::new();
    let entries = split_lark_rules(source)?;
    let mut start_rule = None;
    let mut bodies = Vec::with_capacity(entries.len());
    for (name, body) in entries {
        let id = builder.intern_named(&name)?;
        if start_rule.is_none() || name == "start" {
            start_rule = Some(id);
        }
        bodies.push((id, body));
    }
    let start = start_rule.ok_or_else(|| invalid("Lark grammar defines no start rule"))?;
    for (lhs, body) in bodies {
        let trimmed = body.trim();
        if let Some(json_tail) = trimmed.strip_prefix("%json") {
            let schema_text = json_tail.trim();
            let (val, consumed) = parse_ordered_json_prefix(schema_text)?;
            if !schema_text[consumed..].trim().is_empty() {
                return Err(invalid("unexpected trailing characters after %json schema"));
            }
            let root = schema::compile(&mut builder, &val)?;
            builder.add_prod(lhs, vec![Symbol::NonTerm(root)]);
        } else {
            let mut parser = LarkExprParser {
                src: trimmed.as_bytes(),
                pos: 0,
                builder: &mut builder,
            };
            let parsed = parser.parse_alt()?;
            parser.skip_ws();
            if parser.pos != parser.src.len() {
                return Err(invalid("unexpected trailing token in Lark rule body"));
            }
            builder.add_prod(lhs, vec![Symbol::NonTerm(parsed)]);
        }
    }
    builder.build(start)
}

fn split_lark_rules(source: &str) -> Result<Vec<(String, String)>> {
    let bytes = source.as_bytes();
    let mut pos = 0_usize;
    let mut rules: Vec<(String, String)> = Vec::new();
    let mut cur_name: Option<String> = None;
    let mut cur_start = 0_usize;
    let mut cur_end = 0_usize;
    let mut brace_depth = 0_usize;
    while pos < bytes.len() {
        let line_start = pos;
        while pos < bytes.len() && bytes[pos] != b'\n' {
            pos += 1;
        }
        let line_end = pos;
        if pos < bytes.len() && bytes[pos] == b'\n' {
            pos += 1;
        }
        let line = std::str::from_utf8(&bytes[line_start..line_end])
            .map_err(|_| invalid("Lark source is not UTF-8"))?;
        let trimmed = line.trim();
        if brace_depth == 0 {
            if trimmed.starts_with("%llguidance") || trimmed.starts_with("//") {
                continue;
            }
            if let Some((lhs, body_offset)) = parse_rule_header(line) {
                if let Some(prev) = cur_name.take() {
                    let text = std::str::from_utf8(&bytes[cur_start..cur_end])
                        .map_err(|_| invalid("Lark body not UTF-8"))?;
                    rules.push((prev, text.to_owned()));
                }
                cur_name = Some(lhs);
                cur_start = line_start + body_offset;
                cur_end = line_end;
                brace_depth = count_json_brace_delta(line[body_offset..].trim_start());
                continue;
            }
        } else {
            brace_depth = brace_depth.saturating_add_signed(signed_brace_delta(line));
        }
        if cur_name.is_some() {
            cur_end = line_end;
        } else if !trimmed.is_empty() {
            return Err(invalid("Lark content before initial rule definition"));
        }
    }
    if let Some(prev) = cur_name {
        let text = std::str::from_utf8(&bytes[cur_start..cur_end])
            .map_err(|_| invalid("Lark body not UTF-8"))?;
        rules.push((prev, text.to_owned()));
    }
    Ok(rules)
}

fn parse_rule_header(line: &str) -> Option<(String, usize)> {
    let bytes = line.as_bytes();
    let mut idx = 0_usize;
    while idx < bytes.len() && matches!(bytes[idx], b' ' | b'\t' | b'\r') {
        idx += 1;
    }
    let id_start = idx;
    let first = *bytes.get(idx)?;
    if !(first.is_ascii_alphabetic() || first == b'_') {
        return None;
    }
    idx += 1;
    while idx < bytes.len()
        && (bytes[idx].is_ascii_alphanumeric() || matches!(bytes[idx], b'_' | b'-'))
    {
        idx += 1;
    }
    let name = std::str::from_utf8(&bytes[id_start..idx]).ok()?;
    while idx < bytes.len() && matches!(bytes[idx], b' ' | b'\t') {
        idx += 1;
    }
    if bytes.get(idx) == Some(&b':') {
        Some((name.to_owned(), idx + 1))
    } else {
        None
    }
}

fn count_json_brace_delta(tail: &str) -> usize {
    if !tail.starts_with("%json") {
        return 0;
    }
    usize::try_from(signed_brace_delta(tail).max(0)).unwrap_or(0)
}

fn signed_brace_delta(line: &str) -> isize {
    let mut delta = 0_isize;
    let mut in_str = false;
    let mut escape = false;
    for byte in line.bytes() {
        if escape {
            escape = false;
            continue;
        }
        if in_str {
            match byte {
                b'\\' => escape = true,
                b'"' => in_str = false,
                _ => {}
            }
        } else {
            match byte {
                b'"' => in_str = true,
                b'{' => delta += 1,
                b'}' => delta -= 1,
                _ => {}
            }
        }
    }
    delta
}

struct LarkExprParser<'a> {
    src: &'a [u8],
    pos: usize,
    builder: &'a mut Builder,
}

impl LarkExprParser<'_> {
    fn skip_ws(&mut self) {
        while self
            .src
            .get(self.pos)
            .is_some_and(|&b| matches!(b, b' ' | b'\t' | b'\r' | b'\n'))
        {
            self.pos += 1;
        }
    }

    fn parse_alt(&mut self) -> Result<u32> {
        let first = self.parse_seq()?;
        self.skip_ws();
        if self.src.get(self.pos) != Some(&b'|') {
            return Ok(first);
        }
        let rule = self.builder.new_rule()?;
        self.builder.add_prod(rule, vec![Symbol::NonTerm(first)]);
        while {
            self.skip_ws();
            self.src.get(self.pos) == Some(&b'|')
        } {
            self.pos += 1;
            let next = self.parse_seq()?;
            self.builder.add_prod(rule, vec![Symbol::NonTerm(next)]);
        }
        Ok(rule)
    }

    fn parse_seq(&mut self) -> Result<u32> {
        let mut items = Vec::new();
        loop {
            self.skip_ws();
            match self.src.get(self.pos).copied() {
                None | Some(b')' | b'|') => break,
                _ => items.push(self.parse_postfix()?),
            }
        }
        if items.len() == 1
            && let Symbol::NonTerm(id) = items[0]
        {
            return Ok(id);
        }
        let rule = self.builder.new_rule()?;
        self.builder.add_prod(rule, items);
        Ok(rule)
    }

    fn parse_postfix(&mut self) -> Result<Symbol> {
        let atom = self.parse_atom()?;
        self.skip_ws();
        match self.src.get(self.pos).copied() {
            Some(b'*') => {
                self.pos += 1;
                Ok(Symbol::NonTerm(self.builder.zero_or_more(atom)?))
            }
            Some(b'+') => {
                self.pos += 1;
                Ok(Symbol::NonTerm(self.builder.one_or_more(atom)?))
            }
            Some(b'?') => {
                self.pos += 1;
                Ok(Symbol::NonTerm(self.builder.optional(atom)?))
            }
            _ => Ok(atom),
        }
    }

    fn parse_atom(&mut self) -> Result<Symbol> {
        self.skip_ws();
        let b = *self
            .src
            .get(self.pos)
            .ok_or_else(|| invalid("unexpected end of Lark expression"))?;
        match b {
            b'(' => {
                self.pos += 1;
                let inner = self.parse_alt()?;
                self.skip_ws();
                if self.src.get(self.pos) != Some(&b')') {
                    return Err(invalid("unclosed '(' in Lark expression"));
                }
                self.pos += 1;
                Ok(Symbol::NonTerm(inner))
            }
            b'"' => {
                let bytes = self.read_string_literal()?;
                Ok(Symbol::NonTerm(self.builder.literal_bytes(&bytes)?))
            }
            b'/' => {
                let pattern = self.read_regex_literal()?;
                Ok(Symbol::NonTerm(compile_regex(self.builder, &pattern)?))
            }
            _ if b.is_ascii_alphabetic() || b == b'_' => {
                let start = self.pos;
                self.pos += 1;
                while self
                    .src
                    .get(self.pos)
                    .is_some_and(|&c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-'))
                {
                    self.pos += 1;
                }
                let name = std::str::from_utf8(&self.src[start..self.pos])
                    .map_err(|_| invalid("non-UTF8 rule name in Lark"))?;
                Ok(Symbol::NonTerm(self.builder.intern_named(name)?))
            }
            _ => Err(invalid("unexpected token in Lark expression")),
        }
    }

    fn read_string_literal(&mut self) -> Result<Vec<u8>> {
        self.pos += 1;
        let mut out = Vec::new();
        while self.pos < self.src.len() {
            let b = self.src[self.pos];
            self.pos += 1;
            if b == b'"' {
                return Ok(out);
            }
            if b == b'\\' {
                let esc = *self
                    .src
                    .get(self.pos)
                    .ok_or_else(|| invalid("unterminated Lark string escape"))?;
                self.pos += 1;
                match esc {
                    b'n' => out.push(b'\n'),
                    b'r' => out.push(b'\r'),
                    b't' => out.push(b'\t'),
                    _ => out.push(esc),
                }
            } else {
                out.push(b);
            }
        }
        Err(invalid("unterminated string literal in Lark grammar"))
    }

    fn read_regex_literal(&mut self) -> Result<String> {
        self.pos += 1;
        let mut out = String::new();
        while self.pos < self.src.len() {
            let b = self.src[self.pos];
            if b == b'/' {
                self.pos += 1;
                return Ok(out);
            }
            if b == b'\\' {
                self.pos += 1;
                let esc = *self
                    .src
                    .get(self.pos)
                    .ok_or_else(|| invalid("unterminated regex literal escape"))?;
                self.pos += 1;
                if esc != b'/' {
                    out.push('\\');
                }
                out.push(char::from(esc));
            } else {
                let rest = std::str::from_utf8(&self.src[self.pos..])
                    .map_err(|_| invalid("non-UTF8 regex literal"))?;
                let ch = rest
                    .chars()
                    .next()
                    .ok_or_else(|| invalid("empty char in regex literal"))?;
                out.push(ch);
                self.pos += ch.len_utf8();
            }
        }
        Err(invalid("unterminated /regex/ in Lark grammar"))
    }
}
