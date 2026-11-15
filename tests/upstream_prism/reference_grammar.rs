//! Clean-room independent safe Rust GBNF grammar execution and candidate rejection provider.
//!
//! No production `bonsai_prism_runtime` code, native bridge or precomputed
//! answer fixture is called here. Parses rules via the independent grammar
//! parser reference provider for `init-text`, tracks pushdown stacks using
//! pinned `Box<[Element]>` rule storage so snapshot pointer addresses remain
//! valid across owner clones, validates saved-stack lifetimes, and executes
//! incremental UTF-8 decoding, acceptance, and stack candidate filtering.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Value, json};

use super::{Result, invalid};

pub(super) const IDS: &[&str] = &[
    "tests/test-llama-grammar.cpp",
    "tests/test-gbnf-validator.cpp",
];
const SOURCE_SHA256: &str = "1e0e3ea0f9ec72f8b0f6b6401f2447c3fd8ef1f4a53433772f66066bb5e83a14";
const VALIDATOR_SHA256: &str = "21eb0bac296c0e0dd497cc438ab8f10ba9cad66488c00288321d939afbdf310f";

const MAX_OPS: usize = 128;
const MAX_ACTIONS: usize = 16_384;
const MAX_GRAMMARS: usize = 32;
const MAX_STACK_HANDLES: usize = 64;
const MAX_RULES: usize = 4096;
const MAX_ELEMENTS: usize = 65_536;
const MAX_STACKS: usize = 8192;
const MAX_CANDIDATES: usize = 4096;
const MAX_CANDIDATE_POINTS: usize = 1024;
const MAX_TEXT_BYTES: usize = 262_144;
const MAX_ROOT_BYTES: usize = 4096;
const MAX_PIECE_BYTES: usize = 65_536;
const MAX_PREVIEW_BYTES: usize = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Element {
    kind: u32,
    value: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Position {
    rule: usize,
    element: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PartialUtf8 {
    value: u32,
    remaining: i32,
}

#[derive(Clone)]
struct Candidate {
    index: u32,
    code_points: Vec<u32>,
    partial_utf8: PartialUtf8,
    id: i32,
}

struct GrammarMachine {
    owner_id: u64,
    rules: Vec<Box<[Element]>>,
    stacks: Vec<Vec<Position>>,
    partial: PartialUtf8,
}

struct SavedStacks {
    owner_handle: usize,
    owner_id: u64,
    stacks: Vec<Vec<Position>>,
}

const fn is_end(element: Element) -> bool {
    element.kind == 0 || element.kind == 1
}

fn instruction(rules: &[Box<[Element]>], pos: Position) -> Result<Element> {
    rules
        .get(pos.rule)
        .and_then(|rule| rule.get(pos.element))
        .copied()
        .ok_or_else(|| invalid("grammar instruction position outside rule storage"))
}

const fn next_pos(pos: Position) -> Position {
    Position {
        rule: pos.rule,
        element: pos.element + 1,
    }
}

fn rule_alternatives(rule: &[Element]) -> Vec<usize> {
    std::iter::once(0)
        .chain(
            rule.iter()
                .enumerate()
                .filter(|(_, element)| element.kind == 1)
                .map(|(index, _)| index + 1),
        )
        .collect()
}

fn reject_left_recursion(rules: &[Box<[Element]>]) -> Result<()> {
    let nullable: Vec<bool> = rules
        .iter()
        .map(|rule| {
            rule_alternatives(rule)
                .iter()
                .any(|&start| is_end(rule[start]))
        })
        .collect();
    let mut edges = vec![Vec::new(); rules.len()];
    for (idx, rule) in rules.iter().enumerate() {
        for start in rule_alternatives(rule) {
            for element in &rule[start..] {
                if element.kind != 2 {
                    break;
                }
                let target = usize::try_from(element.value)?;
                edges[idx].push(target);
                if !nullable[target] {
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
        let mut stack = vec![(start, 0_usize)];
        color[start] = 1;
        while let Some((node, edge_idx)) = stack.last_mut() {
            if *edge_idx == edges[*node].len() {
                color[*node] = 2;
                stack.pop();
                continue;
            }
            let child = edges[*node][*edge_idx];
            *edge_idx += 1;
            match color[child] {
                1 => return Err(invalid("left-recursive grammar is unsupported")),
                0 => {
                    color[child] = 1;
                    stack.push((child, 0));
                }
                _ => {}
            }
        }
    }
    Ok(())
}

fn advance_stack(
    rules: &[Box<[Element]>],
    stack: Vec<Position>,
    output: &mut Vec<Vec<Position>>,
) -> Result<()> {
    let mut worklist = vec![stack];
    let mut seen = BTreeSet::new();
    while let Some(mut current) = worklist.pop() {
        if current.len() > MAX_ELEMENTS
            || seen.len() > MAX_ELEMENTS
            || worklist.len() > MAX_ELEMENTS
            || output.len() > MAX_STACKS
        {
            return Err(invalid("grammar stack expansion exceeds bound"));
        }
        if !seen.insert(current.clone()) {
            continue;
        }
        let Some(&pos) = current.last() else {
            if !output.contains(&current) {
                output.push(current);
            }
            continue;
        };
        let elem = instruction(rules, pos)?;
        match elem.kind {
            2 => {
                current.pop();
                let cont = next_pos(pos);
                if !is_end(instruction(rules, cont)?) {
                    current.push(cont);
                }
                let target = usize::try_from(elem.value)?;
                for start in rule_alternatives(&rules[target]) {
                    let mut branch = current.clone();
                    if !is_end(rules[target][start]) {
                        branch.push(Position {
                            rule: target,
                            element: start,
                        });
                    }
                    worklist.push(branch);
                }
            }
            3 | 4 | 7 | 8 | 9 => {
                if !output.contains(&current) {
                    output.push(current);
                }
            }
            _ => return Err(invalid("grammar stack does not end at a terminal")),
        }
    }
    Ok(())
}

fn match_char(rules: &[Box<[Element]>], mut pos: Position, point: u32) -> Result<(bool, Position)> {
    let first = instruction(rules, pos)?;
    if !matches!(first.kind, 3 | 4 | 7) {
        return Err(invalid("character matcher requires a character terminal"));
    }
    let positive = first.kind != 4;
    let mut found = false;
    loop {
        let elem = instruction(rules, pos)?;
        let after = instruction(rules, next_pos(pos))?;
        if after.kind == 5 {
            found |= elem.value <= point && point <= after.value;
            pos = next_pos(next_pos(pos));
        } else {
            found |= elem.kind == 7 || elem.value == point;
            pos = next_pos(pos);
        }
        if instruction(rules, pos)?.kind != 6 {
            break;
        }
    }
    Ok((found == positive, pos))
}

fn match_partial(
    rules: &[Box<[Element]>],
    mut pos: Position,
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
    let positive = instruction(rules, pos)?.kind != 4;
    loop {
        let elem = instruction(rules, pos)?;
        let after = instruction(rules, next_pos(pos))?;
        if after.kind == 5 {
            if elem.value <= high && low <= after.value {
                return Ok(positive);
            }
            pos = next_pos(next_pos(pos));
        } else {
            if elem.kind == 7 {
                return Ok(true);
            }
            if low <= elem.value && elem.value <= high {
                return Ok(positive);
            }
            pos = next_pos(pos);
        }
        if instruction(rules, pos)?.kind != 6 {
            break;
        }
    }
    Ok(!positive)
}

const fn match_token(elem: Element, token: i32) -> bool {
    let token_bits = u32::from_ne_bytes(token.to_ne_bytes());
    (elem.value == token_bits) == (elem.kind == 8)
}

fn accept_char(
    rules: &[Box<[Element]>],
    stack: &[Position],
    point: u32,
    output: &mut Vec<Vec<Position>>,
) -> Result<()> {
    let Some(&pos) = stack.last() else {
        return Ok(());
    };
    if matches!(instruction(rules, pos)?.kind, 8 | 9) {
        return Ok(());
    }
    let (matched, after) = match_char(rules, pos, point)?;
    if matched {
        let mut following = stack[..stack.len() - 1].to_vec();
        if !is_end(instruction(rules, after)?) {
            following.push(after);
        }
        advance_stack(rules, following, output)?;
    }
    Ok(())
}

const fn is_valid_utf8_cont(partial: PartialUtf8, byte: u8) -> bool {
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

fn decode_utf8(bytes: &[u8], mut partial: PartialUtf8) -> (Vec<u32>, PartialUtf8) {
    let mut points = Vec::new();
    let mut idx = 0;
    if !(0..=3).contains(&partial.remaining) {
        points.push(0xfffd);
        partial = PartialUtf8::default();
    }
    while idx < bytes.len() && bytes[idx] != 0 {
        let byte = bytes[idx];
        if partial.remaining > 0 {
            if is_valid_utf8_cont(partial, byte) {
                partial.value = (partial.value << 6) | u32::from(byte & 0x3f);
                partial.remaining -= 1;
                idx += 1;
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
        partial = match byte {
            0x01..=0x7f => {
                points.push(u32::from(byte));
                PartialUtf8::default()
            }
            0xc2..=0xdf => PartialUtf8 {
                value: u32::from(byte & 0x1f),
                remaining: 1,
            },
            0xe0..=0xef => PartialUtf8 {
                value: u32::from(byte & 0x0f),
                remaining: 2,
            },
            0xf0..=0xf4 => PartialUtf8 {
                value: u32::from(byte & 0x07),
                remaining: 3,
            },
            _ => {
                points.push(0xfffd);
                PartialUtf8::default()
            }
        };
        idx += 1;
    }
    (points, partial)
}

fn bounded_preview(bytes: &[u8]) -> String {
    let sample = &bytes[..bytes.len().min(MAX_PREVIEW_BYTES)];
    let preview = String::from_utf8_lossy(sample);
    if bytes.len() <= MAX_PREVIEW_BYTES && preview.len() <= MAX_PREVIEW_BYTES {
        return preview.into_owned();
    }
    let mut end = preview.len().min(MAX_PREVIEW_BYTES.saturating_sub(3));
    while end > 0 && !preview.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", &preview[..end])
}

impl GrammarMachine {
    fn from_rules(owner_id: u64, mut raw_rules: Vec<Vec<Element>>, start: usize) -> Result<Self> {
        if raw_rules.is_empty()
            || raw_rules.len() > MAX_RULES
            || start >= raw_rules.len()
            || raw_rules.iter().map(Vec::len).sum::<usize>() > MAX_ELEMENTS
        {
            return Err(invalid("grammar rule extent outside bound"));
        }
        let count = raw_rules.len();
        let mut boxed_rules = Vec::with_capacity(count);
        for rule in &mut raw_rules {
            let end_idx = rule
                .iter()
                .position(|elem| elem.kind == 0)
                .ok_or_else(|| invalid("grammar rule must end with END"))?;
            rule.truncate(end_idx + 1);
            rule[end_idx].value = 0;
            for elem in rule.iter() {
                if elem.kind > 9 || (elem.kind == 2 && usize::try_from(elem.value)? >= count) {
                    return Err(invalid("invalid grammar instruction or rule reference"));
                }
            }
            boxed_rules.push(rule.clone().into_boxed_slice());
        }
        reject_left_recursion(&boxed_rules)?;
        let mut stacks = Vec::new();
        for start_elem in rule_alternatives(&boxed_rules[start]) {
            let init_stack = if is_end(boxed_rules[start][start_elem]) {
                Vec::new()
            } else {
                vec![Position {
                    rule: start,
                    element: start_elem,
                }]
            };
            advance_stack(&boxed_rules, init_stack, &mut stacks)?;
        }
        Ok(Self {
            owner_id,
            rules: boxed_rules,
            stacks,
            partial: PartialUtf8::default(),
        })
    }

    fn clone_with_owner(&self, owner_id: u64) -> Self {
        Self {
            owner_id,
            rules: self.rules.clone(),
            stacks: self.stacks.clone(),
            partial: self.partial,
        }
    }

    fn stack_sizes(&self) -> Result<Vec<usize>> {
        if self.stacks.len() > MAX_STACKS {
            return Err(invalid("grammar stacks exceed bound"));
        }
        let mut sizes = Vec::with_capacity(self.stacks.len());
        for stack in &self.stacks {
            if stack.len() > MAX_ELEMENTS {
                return Err(invalid("grammar stack length exceeds bound"));
            }
            sizes.push(stack.len());
        }
        Ok(sizes)
    }

    fn snapshot(&self) -> Result<Value> {
        if self.rules.len() > MAX_RULES {
            return Err(invalid("grammar rules exceed bound"));
        }
        let _ = self.stack_sizes()?;
        let mut total_elems = 0_usize;
        let mut encoded_rules = Vec::with_capacity(self.rules.len());
        let mut addresses = Vec::with_capacity(self.rules.len());
        for rule in &self.rules {
            total_elems += rule.len();
            if total_elems > MAX_ELEMENTS {
                return Err(invalid("grammar snapshot elements exceed bound"));
            }
            addresses.push(u64::try_from(rule.as_ptr().addr())?);
            encoded_rules.push(
                rule.iter()
                    .map(|elem| json!({"Type": elem.kind, "Value": elem.value}))
                    .collect::<Vec<_>>(),
            );
        }
        let mut total_ptrs = 0_usize;
        let mut stacks_json = Vec::with_capacity(self.stacks.len());
        for stack in &self.stacks {
            total_ptrs += stack.len();
            if total_ptrs > MAX_ELEMENTS {
                return Err(invalid("grammar snapshot stack pointers exceed bound"));
            }
            let mut ptrs = Vec::with_capacity(stack.len());
            for pos in stack {
                let elem = self
                    .rules
                    .get(pos.rule)
                    .and_then(|rule| rule.get(pos.element))
                    .ok_or_else(|| invalid("grammar stack position out of bounds"))?;
                ptrs.push(u64::try_from(std::ptr::from_ref(elem).addr())?);
            }
            stacks_json.push(ptrs);
        }
        Ok(json!({
            "Rules": encoded_rules,
            "RuleAddresses": addresses,
            "ElementSize": size_of::<Element>(),
            "Stacks": stacks_json
        }))
    }

    fn accept_codepoint(&mut self, point: u32) -> Result<()> {
        let mut next_stacks = Vec::new();
        for stack in &self.stacks {
            accept_char(&self.rules, stack, point, &mut next_stacks)?;
        }
        self.stacks = next_stacks;
        Ok(())
    }

    fn accept_token(&mut self, token: i32, piece: &[u8]) -> Result<()> {
        if piece.len() > MAX_PIECE_BYTES {
            return Err(invalid("grammar token piece exceeds bound"));
        }
        let (points, partial) = decode_utf8(piece, self.partial);
        let mut next_stacks = Vec::new();
        for stack in &self.stacks {
            let Some(&pos) = stack.last() else {
                continue;
            };
            let elem = instruction(&self.rules, pos)?;
            if matches!(elem.kind, 8 | 9) {
                if match_token(elem, token) {
                    let mut following = stack[..stack.len() - 1].to_vec();
                    let cont = next_pos(pos);
                    if !is_end(instruction(&self.rules, cont)?) {
                        following.push(cont);
                    }
                    advance_stack(&self.rules, following, &mut next_stacks)?;
                }
            } else {
                let mut current = vec![stack.clone()];
                for &point in &points {
                    let mut following = Vec::new();
                    for branch in &current {
                        accept_char(&self.rules, branch, point, &mut following)?;
                    }
                    current = following;
                    if current.is_empty() {
                        break;
                    }
                }
                for branch in current {
                    if !next_stacks.contains(&branch) {
                        next_stacks.push(branch);
                    }
                }
            }
        }
        self.stacks = next_stacks;
        self.partial = partial;
        if self.stacks.is_empty() {
            return Err(invalid(format!(
                "Unexpected empty grammar stack after accepting piece: {} ({token})",
                bounded_preview(piece)
            )));
        }
        Ok(())
    }

    fn reject_offset(
        &self,
        stack: &[Position],
        candidate: &Candidate,
        offset: usize,
    ) -> Result<Option<usize>> {
        let point = candidate.code_points[offset];
        let Some(&pos) = stack.last() else {
            return if point == 0 && candidate.partial_utf8.remaining == 0 {
                Ok(None)
            } else {
                Ok(Some(offset))
            };
        };
        let elem = instruction(&self.rules, pos)?;
        if matches!(elem.kind, 8 | 9) {
            return if match_token(elem, candidate.id) {
                Ok(None)
            } else {
                Ok(Some(offset))
            };
        }
        if point == 0 {
            if candidate.partial_utf8.remaining < 0 {
                return Ok(Some(offset));
            }
            if candidate.partial_utf8.remaining > 0
                && !match_partial(&self.rules, pos, candidate.partial_utf8)?
            {
                return Ok(Some(offset));
            }
            return Ok(None);
        }
        let (matched, after) = match_char(&self.rules, pos, point)?;
        if !matched {
            return Ok(Some(offset));
        }
        let mut following = stack[..stack.len() - 1].to_vec();
        if !is_end(instruction(&self.rules, after)?) {
            following.push(after);
        }
        let mut next_stacks = Vec::new();
        advance_stack(&self.rules, following, &mut next_stacks)?;
        for branch in &next_stacks {
            if self.reject_offset(branch, candidate, offset + 1)?.is_none() {
                return Ok(None);
            }
        }
        Ok(Some(offset + 1))
    }

    fn reject_candidates(&self, stack_idx: usize, candidates: &[Candidate]) -> Result<Vec<Value>> {
        let stack = self
            .stacks
            .get(stack_idx)
            .ok_or_else(|| invalid("grammar stack index outside extent"))?;
        let mut rejected = Vec::new();
        for candidate in candidates {
            if let Some(offset) = self.reject_offset(stack, candidate, 0)? {
                rejected.push(json!({
                    "Index": candidate.index,
                    "CodePoints": &candidate.code_points[offset..],
                    "PartialUtf8": {
                        "Value": candidate.partial_utf8.value,
                        "NRemain": candidate.partial_utf8.remaining
                    },
                    "Id": candidate.id
                }));
            }
        }
        Ok(rejected)
    }
}

fn exact_fields<'a>(
    value: &'a Value,
    expected: &[&str],
) -> Result<&'a serde_json::Map<String, Value>> {
    let map = value
        .as_object()
        .ok_or_else(|| invalid("grammar value must be an object"))?;
    if map.len() != expected.len() || expected.iter().any(|key| !map.contains_key(*key)) {
        return Err(invalid("grammar object fields do not match schema"));
    }
    Ok(map)
}

fn field<'a>(value: &'a Value, name: &str) -> Result<&'a Value> {
    value
        .get(name)
        .ok_or_else(|| invalid(format!("missing grammar field {name}")))
}

fn array<'a>(value: &'a Value, name: &str) -> Result<&'a [Value]> {
    field(value, name)?
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| invalid(format!("grammar {name} must be an array")))
}

fn text_bounded<'a>(
    value: &'a Value,
    name: &str,
    max_bytes: usize,
    forbid_nul: bool,
) -> Result<&'a str> {
    let s = value
        .as_str()
        .ok_or_else(|| invalid(format!("grammar {name} must be text")))?;
    if s.len() > max_bytes || (forbid_nul && s.contains('\0')) {
        return Err(invalid(format!(
            "grammar {name} exceeds bound or contains NUL"
        )));
    }
    Ok(s)
}

fn int_range(value: &Value, name: &str, min: i64, max: i64) -> Result<i64> {
    if value.as_f64().is_some() && value.as_i64().is_none() && value.as_u64().is_none() {
        return Err(invalid(format!("grammar {name} must be integer")));
    }
    let v = value
        .as_i64()
        .ok_or_else(|| invalid(format!("grammar {name} must be signed integer")))?;
    if v < min || v > max {
        return Err(invalid(format!(
            "grammar {name} outside bound [{min}, {max}]"
        )));
    }
    Ok(v)
}

fn uint_range(value: &Value, name: &str, max: u64) -> Result<u64> {
    let v = value
        .as_u64()
        .ok_or_else(|| invalid(format!("grammar {name} must be unsigned integer")))?;
    if v > max {
        return Err(invalid(format!("grammar {name} exceeds bound {max}")));
    }
    Ok(v)
}

fn parse_rules_input(rules_value: &Value) -> Result<Vec<Vec<Element>>> {
    let raw = rules_value
        .as_array()
        .ok_or_else(|| invalid("grammar Rules must be an array"))?;
    if raw.is_empty() || raw.len() > MAX_RULES {
        return Err(invalid("grammar rules count outside bound"));
    }
    let rule_count = u64::try_from(raw.len())?;
    let mut total = 0_usize;
    let mut parsed = Vec::with_capacity(raw.len());
    for rule_val in raw {
        let elements = rule_val
            .as_array()
            .ok_or_else(|| invalid("grammar rule must be an array"))?;
        if elements.is_empty() {
            return Err(invalid("grammar rule must not be empty"));
        }
        total += elements.len();
        if total > MAX_ELEMENTS {
            return Err(invalid("grammar rule elements exceed bound"));
        }
        let mut rule = Vec::with_capacity(elements.len());
        for elem_val in elements {
            exact_fields(elem_val, &["Type", "Value"])?;
            let kind = u32::try_from(uint_range(field(elem_val, "Type")?, "Type", 9)?)?;
            let value = u32::try_from(uint_range(
                field(elem_val, "Value")?,
                "Value",
                u64::from(u32::MAX),
            )?)?;
            if kind == 2 && u64::from(value) >= rule_count {
                return Err(invalid("grammar rule reference exceeds extent"));
            }
            rule.push(Element { kind, value });
        }
        if rule.last().is_none_or(|last| last.kind != 0) {
            return Err(invalid("grammar rule must end with END"));
        }
        parsed.push(rule);
    }
    Ok(parsed)
}

fn parse_piece(action: &Value) -> Result<Vec<u8>> {
    let has_piece = action.get("Piece").is_some();
    let has_bytes = action.get("PieceBytes").is_some();
    if has_piece == has_bytes {
        return Err(invalid("accept-token requires exactly Piece or PieceBytes"));
    }
    let piece = if has_piece {
        exact_fields(action, &["Kind", "Grammar", "Token", "Piece"])?;
        text_bounded(field(action, "Piece")?, "Piece", MAX_PIECE_BYTES, false)?
            .as_bytes()
            .to_vec()
    } else {
        exact_fields(action, &["Kind", "Grammar", "Token", "PieceBytes"])?;
        let raw = array(action, "PieceBytes")?;
        if raw.len() > MAX_PIECE_BYTES {
            return Err(invalid("grammar piece bytes exceed bound"));
        }
        raw.iter()
            .map(|b| Ok(u8::try_from(uint_range(b, "PieceByte", 255)?)?))
            .collect::<Result<Vec<u8>>>()?
    };
    Ok(piece)
}

fn parse_candidates(action: &Value) -> Result<Vec<Candidate>> {
    let raw = array(action, "Candidates")?;
    if raw.len() > MAX_CANDIDATES {
        return Err(invalid("grammar candidates count exceeds bound"));
    }
    raw.iter()
        .map(|entry| {
            exact_fields(entry, &["Index", "CodePoints", "PartialUtf8", "Id"])?;
            let index = u32::try_from(uint_range(
                field(entry, "Index")?,
                "Index",
                u64::from(u32::MAX),
            )?)?;
            let id = i32::try_from(int_range(
                field(entry, "Id")?,
                "Id",
                i64::from(i32::MIN),
                i64::from(i32::MAX),
            )?)?;
            let partial = field(entry, "PartialUtf8")?;
            exact_fields(partial, &["Value", "NRemain"])?;
            let pval = u32::try_from(uint_range(
                field(partial, "Value")?,
                "Value",
                u64::from(u32::MAX),
            )?)?;
            let n_remain = i32::try_from(int_range(field(partial, "NRemain")?, "NRemain", -1, 3)?)?;
            let points_raw = array(entry, "CodePoints")?;
            if points_raw.is_empty() || points_raw.len() > MAX_CANDIDATE_POINTS {
                return Err(invalid("candidate code points extent outside bound"));
            }
            let code_points = points_raw
                .iter()
                .map(|p| {
                    Ok(u32::try_from(uint_range(
                        p,
                        "CodePoint",
                        u64::from(u32::MAX),
                    )?)?)
                })
                .collect::<Result<Vec<u32>>>()?;
            if code_points.last() != Some(&0) {
                return Err(invalid("candidate code points must end with zero"));
            }
            Ok(Candidate {
                index,
                code_points,
                partial_utf8: PartialUtf8 {
                    value: pval,
                    remaining: n_remain,
                },
                id,
            })
        })
        .collect()
}

fn validate_trace_actions(actions: &[Value]) -> Result<()> {
    if actions.is_empty() || actions.len() > MAX_ACTIONS {
        return Err(invalid("grammar action count outside bound"));
    }
    let mut live = [false; MAX_GRAMMARS];
    let mut saved: BTreeMap<usize, usize> = BTreeMap::new();
    for action in actions {
        let kind = field(action, "Kind")?
            .as_str()
            .ok_or_else(|| invalid("grammar action Kind must be text"))?;
        let handle = usize::try_from(uint_range(
            field(action, "Grammar")?,
            "Grammar",
            u64::try_from(MAX_GRAMMARS - 1)?,
        )?)?;
        match kind {
            "init-text" => {
                if live[handle] {
                    return Err(invalid("grammar handle already exists"));
                }
                exact_fields(action, &["Kind", "Grammar", "GrammarText", "Root"])?;
                text_bounded(
                    field(action, "GrammarText")?,
                    "GrammarText",
                    MAX_TEXT_BYTES,
                    true,
                )?;
                let root = text_bounded(field(action, "Root")?, "Root", MAX_ROOT_BYTES, true)?;
                if root.is_empty() {
                    return Err(invalid("grammar Root must not be empty"));
                }
                live[handle] = true;
            }
            "init-rules" => {
                if live[handle] {
                    return Err(invalid("grammar handle already exists"));
                }
                exact_fields(action, &["Kind", "Grammar", "Rules", "StartRule"])?;
                let rules = parse_rules_input(field(action, "Rules")?)?;
                uint_range(
                    field(action, "StartRule")?,
                    "StartRule",
                    u64::try_from(rules.len() - 1)?,
                )?;
                live[handle] = true;
            }
            _ => {
                if !live[handle] {
                    return Err(invalid("unknown or freed grammar handle"));
                }
                validate_live_action(action, kind, handle, &mut live, &mut saved)?;
            }
        }
    }
    Ok(())
}

fn validate_live_action(
    action: &Value,
    kind: &str,
    handle: usize,
    live: &mut [bool; MAX_GRAMMARS],
    saved: &mut BTreeMap<usize, usize>,
) -> Result<()> {
    match kind {
        "clone" => {
            exact_fields(action, &["Kind", "Grammar", "Target"])?;
            let target = usize::try_from(uint_range(
                field(action, "Target")?,
                "Target",
                u64::try_from(MAX_GRAMMARS - 1)?,
            )?)?;
            if live[target] {
                return Err(invalid("grammar clone target already exists"));
            }
            live[target] = true;
        }
        "accept-codepoint" => {
            exact_fields(action, &["Kind", "Grammar", "CodePoint"])?;
            uint_range(
                field(action, "CodePoint")?,
                "CodePoint",
                u64::from(u32::MAX),
            )?;
        }
        "accept-token" => {
            let _ = parse_piece(action)?;
            int_range(
                field(action, "Token")?,
                "Token",
                i64::from(i32::MIN),
                i64::from(i32::MAX),
            )?;
        }
        "save-stacks" | "restore-stacks" => {
            exact_fields(action, &["Kind", "Grammar", "StackHandle"])?;
            let stack_handle = usize::try_from(uint_range(
                field(action, "StackHandle")?,
                "StackHandle",
                u64::try_from(MAX_STACK_HANDLES - 1)?,
            )?)?;
            if kind == "save-stacks" {
                saved.insert(stack_handle, handle);
            } else if saved.get(&stack_handle) != Some(&handle) {
                return Err(invalid(
                    "saved stacks belong to a different grammar lifetime",
                ));
            }
        }
        "reject-candidates" => {
            exact_fields(action, &["Kind", "Grammar", "Stack", "Candidates"])?;
            uint_range(
                field(action, "Stack")?,
                "Stack",
                u64::try_from(MAX_STACKS - 1)?,
            )?;
            let _ = parse_candidates(action)?;
        }
        "free" => {
            exact_fields(action, &["Kind", "Grammar"])?;
            live[handle] = false;
            saved.retain(|_, owner| *owner != handle);
        }
        "state" | "snapshot" => {
            exact_fields(action, &["Kind", "Grammar"])?;
        }
        _ => return Err(invalid("unknown grammar action Kind")),
    }
    Ok(())
}

fn init_from_text(owner_id: u64, text: &str, root: &str) -> Option<GrammarMachine> {
    if text.is_empty() {
        return None;
    }
    let req = json!({
        "SchemaVersion": 1,
        "Operations": [{"Kind": "grammar-parser-parse", "Grammar": text}]
    });
    let resp = super::reference_grammar_parser::execute(&req).ok()?;
    let res = resp["Results"].as_array()?.first()?;
    if res["Parsed"].as_bool() != Some(true) {
        return None;
    }
    let start = res["SymbolIds"]
        .as_array()?
        .iter()
        .find(|sym| sym["Name"].as_str() == Some(root))
        .and_then(|sym| sym["Id"].as_u64())
        .and_then(|id| usize::try_from(id).ok())?;
    let parsed_rules = parse_rules_input(&res["Rules"]).ok()?;
    GrammarMachine::from_rules(owner_id, parsed_rules, start).ok()
}

fn execute_trace_action(
    action: &Value,
    next_owner: &mut u64,
    grammars: &mut [Option<GrammarMachine>; MAX_GRAMMARS],
    saved: &mut BTreeMap<usize, SavedStacks>,
) -> Result<Value> {
    let kind = field(action, "Kind")?
        .as_str()
        .ok_or_else(|| invalid("grammar action Kind must be text"))?;
    let handle = usize::try_from(uint_range(
        field(action, "Grammar")?,
        "Grammar",
        u64::try_from(MAX_GRAMMARS - 1)?,
    )?)?;
    match kind {
        "init-text" => {
            let text = field(action, "GrammarText")?.as_str().unwrap_or_default();
            let root = field(action, "Root")?.as_str().unwrap_or_default();
            *next_owner += 1;
            let machine = init_from_text(*next_owner, text, root);
            let created = machine.is_some();
            grammars[handle] = machine;
            Ok(json!({"Created": created}))
        }
        "init-rules" => {
            let rules = parse_rules_input(field(action, "Rules")?)?;
            let start = usize::try_from(uint_range(
                field(action, "StartRule")?,
                "StartRule",
                u64::try_from(rules.len() - 1)?,
            )?)?;
            *next_owner += 1;
            let machine = GrammarMachine::from_rules(*next_owner, rules, start).ok();
            let created = machine.is_some();
            grammars[handle] = machine;
            Ok(json!({"Created": created}))
        }
        "free" => {
            if let Some(prev) = grammars[handle].take() {
                let dead_id = prev.owner_id;
                saved.retain(|_, entry| {
                    !(entry.owner_handle == handle && entry.owner_id == dead_id)
                });
            }
            Ok(json!({}))
        }
        "clone" => {
            let target = usize::try_from(uint_range(
                field(action, "Target")?,
                "Target",
                u64::try_from(MAX_GRAMMARS - 1)?,
            )?)?;
            let source = grammars[handle]
                .as_ref()
                .ok_or_else(|| invalid("grammar clone requires successful initialization"))?;
            *next_owner += 1;
            grammars[target] = Some(source.clone_with_owner(*next_owner));
            Ok(json!({"Created": true}))
        }
        _ => execute_machine_action(action, kind, handle, grammars, saved),
    }
}

fn execute_machine_action(
    action: &Value,
    kind: &str,
    handle: usize,
    grammars: &mut [Option<GrammarMachine>; MAX_GRAMMARS],
    saved: &mut BTreeMap<usize, SavedStacks>,
) -> Result<Value> {
    let machine = grammars[handle]
        .as_mut()
        .ok_or_else(|| invalid("grammar action requires successful initialization"))?;
    match kind {
        "state" => Ok(json!({"StackSizes": machine.stack_sizes()?})),
        "snapshot" => machine.snapshot(),
        "save-stacks" => {
            let stack_handle = usize::try_from(uint_range(
                field(action, "StackHandle")?,
                "StackHandle",
                u64::try_from(MAX_STACK_HANDLES - 1)?,
            )?)?;
            saved.insert(
                stack_handle,
                SavedStacks {
                    owner_handle: handle,
                    owner_id: machine.owner_id,
                    stacks: machine.stacks.clone(),
                },
            );
            Ok(json!({}))
        }
        "restore-stacks" => {
            let stack_handle = usize::try_from(uint_range(
                field(action, "StackHandle")?,
                "StackHandle",
                u64::try_from(MAX_STACK_HANDLES - 1)?,
            )?)?;
            let entry = saved
                .get(&stack_handle)
                .ok_or_else(|| invalid("saved grammar stack handle missing"))?;
            if entry.owner_handle != handle || entry.owner_id != machine.owner_id {
                return Err(invalid(
                    "saved stacks belong to a different grammar lifetime",
                ));
            }
            machine.stacks.clone_from(&entry.stacks);
            Ok(json!({}))
        }
        "accept-codepoint" => {
            let cp = u32::try_from(uint_range(
                field(action, "CodePoint")?,
                "CodePoint",
                u64::from(u32::MAX),
            )?)?;
            let error = machine.accept_codepoint(cp).err().map(|e| e.to_string());
            Ok(json!({
                "ExceptionKind": error.as_ref().map(|_| "runtime"),
                "Message": error.unwrap_or_default(),
                "StackSizes": machine.stack_sizes()?
            }))
        }
        "accept-token" => {
            let token = i32::try_from(int_range(
                field(action, "Token")?,
                "Token",
                i64::from(i32::MIN),
                i64::from(i32::MAX),
            )?)?;
            let piece = parse_piece(action)?;
            let error = machine
                .accept_token(token, &piece)
                .err()
                .map(|e| e.to_string());
            Ok(json!({
                "ExceptionKind": error.as_ref().map(|_| "runtime"),
                "Message": error.unwrap_or_default(),
                "StackSizes": machine.stack_sizes()?
            }))
        }
        "reject-candidates" => {
            let stack_idx = usize::try_from(uint_range(
                field(action, "Stack")?,
                "Stack",
                u64::try_from(MAX_STACKS - 1)?,
            )?)?;
            let candidates = parse_candidates(action)?;
            Ok(json!({"Rejected": machine.reject_candidates(stack_idx, &candidates)?}))
        }
        _ => Err(invalid("unknown grammar action Kind")),
    }
}

fn execute_trace(operation: &Value) -> Result<Value> {
    exact_fields(operation, &["Kind", "Actions"])?;
    let actions = array(operation, "Actions")?;
    validate_trace_actions(actions)?;
    let mut next_owner = 0_u64;
    let mut grammars: [Option<GrammarMachine>; MAX_GRAMMARS] = std::array::from_fn(|_| None);
    let mut saved: BTreeMap<usize, SavedStacks> = BTreeMap::new();
    let mut trace = Vec::with_capacity(actions.len());
    for action in actions {
        trace.push(execute_trace_action(
            action,
            &mut next_owner,
            &mut grammars,
            &mut saved,
        )?);
    }
    Ok(json!({"Trace": trace}))
}

pub(super) fn handles(request: &Value) -> bool {
    request["Operations"].as_array().is_some_and(|operations| {
        !operations.is_empty()
            && operations
                .iter()
                .all(|operation| operation["Kind"] == "grammar-trace")
    })
}

pub(super) fn execute(request: &Value) -> Result<Value> {
    exact_fields(request, &["SchemaVersion", "Operations"])?;
    if uint_range(field(request, "SchemaVersion")?, "SchemaVersion", u64::MAX)? != 1 {
        return Err(invalid("grammar request schema version must equal 1"));
    }
    let operations = array(request, "Operations")?;
    if operations.is_empty() || operations.len() > MAX_OPS {
        return Err(invalid("grammar operation count outside bound"));
    }
    let results = operations
        .iter()
        .map(execute_trace)
        .collect::<Result<Vec<_>>>()?;
    Ok(json!({"SchemaVersion": 1, "Results": results}))
}

pub(super) fn identity() -> Value {
    json!({
        "Kind": "independent-rust-grammar",
        "SupportedSourceIds": IDS,
        "UpstreamCommit": super::UPSTREAM,
        "AssertionSources": [
            {"Path": IDS[0], "Sha256": SOURCE_SHA256},
            {"Path": IDS[1], "Sha256": VALIDATOR_SHA256}
        ],
        "Architecture": "clean-room-boxed-rule-pushdown-and-candidate-filter-v1",
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
    fn independent_grammar_provider_passes_llama_grammar_and_validator_across_all_targets()
    -> Result<()> {
        for target in [Target::RustReference, Target::Cpp, Target::Rust] {
            let engine = Engine::new(target, None, None, None, None, None)?;
            let grammar_report = crate::llama_grammar::run(&engine)?;
            assert_eq!(grammar_report["Id"], IDS[0]);
            assert_eq!(grammar_report["Status"], "pass", "{grammar_report}");

            let validator_report = crate::validator::run(&engine)?;
            assert_eq!(validator_report["Id"], IDS[1]);
            assert_eq!(validator_report["Status"], "pass", "{validator_report}");
        }
        assert!(crate::reference_provider::supports(IDS[0]));
        assert!(crate::reference_provider::supports(IDS[1]));
        let info = identity();
        assert_eq!(info["NativeReferenceReplay"], false);
        assert_eq!(info["ProductionRuntimeCalled"], false);
        assert_eq!(info["CompleteSourceGate"], false);
        assert_eq!(info["NativeRetirementQualified"], false);
        Ok(())
    }

    #[test]
    fn independent_grammar_rejects_invalid_requests_and_tracks_suffix_and_clone_addresses()
    -> Result<()> {
        let trace = execute(&json!({
            "SchemaVersion": 1,
            "Operations": [{
                "Kind": "grammar-trace",
                "Actions": [
                    {"Kind": "init-text", "Grammar": 0, "GrammarText": "root ::= \"ab\"", "Root": "root"},
                    {"Kind": "save-stacks", "Grammar": 0, "StackHandle": 0},
                    {"Kind": "clone", "Grammar": 0, "Target": 1},
                    {"Kind": "free", "Grammar": 0},
                    {"Kind": "snapshot", "Grammar": 1},
                    {
                        "Kind": "reject-candidates",
                        "Grammar": 1,
                        "Stack": 0,
                        "Candidates": [
                            {"Index": 0, "CodePoints": [97, 98, 0], "PartialUtf8": {"Value": 0, "NRemain": 0}, "Id": 10},
                            {"Index": 1, "CodePoints": [97, 99, 0], "PartialUtf8": {"Value": 0, "NRemain": 0}, "Id": 11},
                            {"Index": 2, "CodePoints": [98, 0], "PartialUtf8": {"Value": 0, "NRemain": 0}, "Id": 12}
                        ]
                    },
                    {"Kind": "accept-token", "Grammar": 1, "Token": 10, "Piece": "abc"},
                    {"Kind": "free", "Grammar": 1}
                ]
            }]
        }))?;
        let events = trace["Results"][0]["Trace"]
            .as_array()
            .ok_or_else(|| invalid("missing Trace"))?;
        assert_eq!(events[0]["Created"], true);
        assert_eq!(events[2]["Created"], true);
        assert_eq!(events[5]["Rejected"].as_array().map(Vec::len), Some(2));
        assert_eq!(events[5]["Rejected"][0]["Index"], 1);
        assert_eq!(events[5]["Rejected"][0]["CodePoints"], json!([99, 0]));
        assert_eq!(events[5]["Rejected"][1]["Index"], 2);
        assert_eq!(events[6]["ExceptionKind"], "runtime");
        assert!(
            execute(&json!({
                "SchemaVersion": 1,
                "Operations": [{
                    "Kind": "grammar-trace",
                    "Actions": [
                        {"Kind": "init-text", "Grammar": 0, "GrammarText": "root ::= \"a\"", "Root": "root"},
                        {"Kind": "save-stacks", "Grammar": 0, "StackHandle": 0},
                        {"Kind": "clone", "Grammar": 0, "Target": 1},
                        {"Kind": "restore-stacks", "Grammar": 1, "StackHandle": 0}
                    ]
                }]
            }))
            .is_err()
        );
        Ok(())
    }
}
