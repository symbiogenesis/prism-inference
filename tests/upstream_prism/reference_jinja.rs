//! Clean-room independent safe Rust Jinja2 template, hasher and capability reference provider.
//!
//! This module implements a standalone lexer, parser, AST evaluator, ordered
//! JSON value representation, streaming 64-bit block hasher and capability
//! probe without importing `bonsai_prism_runtime`, spawning any native bridge,
//! or consulting precomputed test fixtures.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::rc::Rc;

use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Value, json};

use super::{Result, invalid};

pub(super) const ID: &str = "tests/test-jinja.cpp";
const SOURCE_SHA: &str = "ef2da16176a8e914cbd54341e4b77f675a8f3b5b15a8bfd0c9a7a2305c30c8ab";

const MAX_OPERATIONS: usize = 128;
const MAX_TEMPLATE_BYTES: usize = 65_536;
const MAX_VARIABLES_BYTES: usize = 256 * 1024;
const MAX_JSON_DEPTH: usize = 32;
const MAX_PATH_SEGMENTS: usize = 32;
const MAX_PATH_STRING_BYTES: usize = 4096;
const MAX_PATH_INDEX: u64 = 65_535;
const MAX_STATS_PATHS: usize = 1024;
const MAX_BYTE_OVERLAYS: usize = 16;
const MAX_BYTE_OVERLAY_LEN: usize = 65_536;
const MAX_HASH_CHUNKS: usize = 1024;
const MAX_HASH_ACTIONS: usize = 2048;
const MAX_HASH_BYTES: usize = 256 * 1024;

const MAX_EXPR_TOKENS: usize = 2048;
const MAX_STMTS_PER_BLOCK: usize = 32_768;
const MAX_PARSE_DEPTH: usize = 128;
const MAX_EXEC_DEPTH: usize = 128;
const MAX_EXEC_STEPS: usize = 1_000_000;
const MAX_OUTPUT_BYTES: usize = 4 * 1024 * 1024;
const MAX_TEXT_BYTES: usize = 1024 * 1024;
const MAX_CONTAINER_LEN: usize = 65_536;
const MAX_DAG_VISITS: usize = 65_536;
const MAX_GRAPH_DEPTH: usize = 64;
const MAX_DIAG_PREVIEW_BYTES: usize = 256;

const HASH_SEED: u64 = 0xcbf2_9ce4_8422_2325;
const HASH_PRIME: u64 = 0x0100_0000_01b3;

const BUILTIN_INVENTORY: &[(&str, &[&str])] = &[
    (
        "global",
        &[
            "namespace",
            "raise_exception",
            "range",
            "strftime_now",
            "test_is_boolean",
            "test_is_callable",
            "test_is_defined",
            "test_is_divisibleby",
            "test_is_eq",
            "test_is_equalto",
            "test_is_escaped",
            "test_is_even",
            "test_is_false",
            "test_is_filter",
            "test_is_float",
            "test_is_ge",
            "test_is_greaterthan",
            "test_is_gt",
            "test_is_in",
            "test_is_integer",
            "test_is_iterable",
            "test_is_lessthan",
            "test_is_lower",
            "test_is_lt",
            "test_is_mapping",
            "test_is_ne",
            "test_is_none",
            "test_is_number",
            "test_is_odd",
            "test_is_sameas",
            "test_is_sequence",
            "test_is_string",
            "test_is_test",
            "test_is_true",
            "test_is_undefined",
            "test_is_upper",
            "tojson",
        ],
    ),
    (
        "int",
        &["abs", "default", "float", "int", "safe", "string", "tojson"],
    ),
    (
        "float",
        &["abs", "default", "float", "int", "safe", "string", "tojson"],
    ),
    (
        "string",
        &[
            "capitalize",
            "default",
            "endswith",
            "float",
            "format",
            "indent",
            "int",
            "join",
            "length",
            "lower",
            "lstrip",
            "replace",
            "rsplit",
            "rstrip",
            "safe",
            "slice",
            "split",
            "startswith",
            "string",
            "strip",
            "title",
            "tojson",
            "upper",
        ],
    ),
    (
        "array",
        &[
            "append",
            "default",
            "first",
            "join",
            "last",
            "length",
            "list",
            "map",
            "max",
            "min",
            "pop",
            "reject",
            "rejectattr",
            "reverse",
            "select",
            "selectattr",
            "slice",
            "sort",
            "string",
            "tojson",
            "unique",
        ],
    ),
    (
        "object",
        &[
            "dictsort", "get", "items", "join", "keys", "length", "string", "tojson", "values",
        ],
    ),
];

// ---------------------------------------------------------------------------
// Error classification (wire errors vs Jinja phase exceptions)
// ---------------------------------------------------------------------------

pub(super) enum JinjaError {
    Standard(String),
    NotImplemented(String),
}

impl JinjaError {
    fn std(message: impl Into<String>) -> Self {
        Self::Standard(message.into())
    }

    fn unimpl(message: impl Into<String>) -> Self {
        Self::NotImplemented(message.into())
    }

    pub(super) fn message(&self) -> &str {
        match self {
            Self::Standard(msg) | Self::NotImplemented(msg) => msg,
        }
    }

    fn into_wire(self, runtime_phase: bool) -> Value {
        match self {
            Self::NotImplemented(msg) if runtime_phase => {
                json!({"ExceptionKind": "not-implemented", "Message": msg})
            }
            Self::NotImplemented(msg) | Self::Standard(msg) => {
                json!({"ExceptionKind": "standard", "Message": msg})
            }
        }
    }
}

pub(super) type JResult<T> = std::result::Result<T, JinjaError>;

// ---------------------------------------------------------------------------
// Streaming 64-bit block hasher
// ---------------------------------------------------------------------------

struct BlockHasher {
    state: u64,
    pending: [u8; 8],
    filled: usize,
}

impl BlockHasher {
    const fn new() -> Self {
        Self {
            state: HASH_SEED,
            pending: [0; 8],
            filled: 0,
        }
    }

    const fn mix_block(&mut self, block: [u8; 8]) {
        self.state = (self.state ^ u64::from_le_bytes(block)).wrapping_mul(HASH_PRIME);
    }

    fn update(&mut self, mut input: &[u8]) {
        if input.is_empty() {
            return;
        }
        if self.filled > 0 {
            let take = (8 - self.filled).min(input.len());
            self.pending[self.filled..self.filled + take].copy_from_slice(&input[..take]);
            self.filled += take;
            input = &input[take..];
            if self.filled == 8 {
                self.mix_block(self.pending);
                self.filled = 0;
            }
        }
        let (full_blocks, tail) = input.as_chunks::<8>();
        for block in full_blocks {
            self.mix_block(*block);
        }
        if !tail.is_empty() {
            self.pending[..tail.len()].copy_from_slice(tail);
            self.filled = tail.len();
        }
    }

    fn digest(&mut self) -> u64 {
        if self.filled > 0 {
            self.pending[self.filled..].fill(0);
            self.mix_block(self.pending);
            self.filled = 0;
        }
        self.state
    }
}

// ---------------------------------------------------------------------------
// Ordered JSON / Jinja value model
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub(super) struct TextSpan {
    pub(super) bytes: Vec<u8>,
    pub(super) is_input: bool,
}

#[derive(Clone, Debug)]
pub(super) struct JVal(Rc<RefCell<ValCell>>);

#[derive(Clone, Debug)]
struct ValCell {
    kind: ValKind,
    used: bool,
    array_access: bool,
    is_namespace: bool,
}

#[derive(Clone, Debug)]
enum ValKind {
    Undefined,
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(Vec<TextSpan>),
    Array(Vec<JVal>),
    Tuple(Vec<JVal>),
    Object(Vec<(JVal, JVal)>),
    Function(String),
    Method(JVal, String),
    Macro(Vec<(String, Option<Expr>)>, Vec<Stmt>),
    Caller(Vec<String>, Vec<Stmt>, Vec<BTreeMap<String, JVal>>),
}

impl JVal {
    fn new(kind: ValKind) -> Self {
        Self(Rc::new(RefCell::new(ValCell {
            kind,
            used: false,
            array_access: false,
            is_namespace: false,
        })))
    }

    pub(super) fn undefined() -> Self {
        Self::new(ValKind::Undefined)
    }

    pub(super) fn none() -> Self {
        Self::new(ValKind::None)
    }

    pub(super) fn boolean(flag: bool) -> Self {
        Self::new(ValKind::Bool(flag))
    }

    pub(super) fn int(num: i64) -> Self {
        Self::new(ValKind::Int(num))
    }

    pub(super) fn float(num: f64) -> Self {
        Self::new(ValKind::Float(num))
    }

    pub(super) fn bytes(raw: Vec<u8>, is_input: bool) -> Self {
        Self::new(ValKind::String(vec![TextSpan {
            bytes: raw,
            is_input,
        }]))
    }

    pub(super) fn text_str(text: impl Into<String>) -> Self {
        Self::bytes(text.into().into_bytes(), false)
    }

    pub(super) fn array(elems: Vec<Self>) -> Self {
        Self::new(ValKind::Array(elems))
    }

    pub(super) fn object(pairs: Vec<(Self, Self)>) -> Self {
        Self::new(ValKind::Object(pairs))
    }

    fn kind_cloned(&self) -> ValKind {
        self.0.borrow().kind.clone()
    }

    fn is_namespace(&self) -> bool {
        self.0.borrow().is_namespace
    }

    fn set_namespace(&self) {
        self.0.borrow_mut().is_namespace = true;
    }

    fn mark_used(&self) {
        self.0.borrow_mut().used = true;
    }

    pub(super) fn is_used(&self) -> bool {
        self.0.borrow().used
    }

    pub(super) fn is_array_access(&self) -> bool {
        self.0.borrow().array_access
    }

    pub(super) fn is_string(&self) -> bool {
        matches!(self.0.borrow().kind, ValKind::String(_))
    }

    fn is_int(&self) -> bool {
        matches!(self.0.borrow().kind, ValKind::Int(_))
    }

    pub(super) fn is_undefined(&self) -> bool {
        matches!(self.0.borrow().kind, ValKind::Undefined)
    }

    pub(super) fn is_none(&self) -> bool {
        matches!(self.0.borrow().kind, ValKind::None)
    }

    pub(super) fn as_array_elems(&self) -> Option<Vec<Self>> {
        match &self.0.borrow().kind {
            ValKind::Array(elems) | ValKind::Tuple(elems) => Some(elems.clone()),
            _ => None,
        }
    }

    pub(super) fn as_object_pairs(&self) -> Option<Vec<(Self, Self)>> {
        match &self.0.borrow().kind {
            ValKind::Object(pairs) => Some(pairs.clone()),
            _ => None,
        }
    }

    fn all_spans_input(&self) -> bool {
        let cell = self.0.borrow();
        matches!(&cell.kind, ValKind::String(spans) if spans.iter().all(|span| span.is_input))
    }

    fn mark_recursive(&self) {
        let mut seen = HashSet::new();
        let mut stack = vec![(self.clone(), 0_usize)];
        let mut steps = 0_usize;
        while let Some((node, depth)) = stack.pop() {
            if depth > MAX_GRAPH_DEPTH || steps >= MAX_DAG_VISITS {
                break;
            }
            if !seen.insert(Rc::as_ptr(&node.0)) {
                continue;
            }
            steps += 1;
            node.mark_used();
            match node.kind_cloned() {
                ValKind::Array(elems) | ValKind::Tuple(elems) => {
                    for elem in elems {
                        stack.push((elem, depth + 1));
                    }
                }
                ValKind::Object(pairs) => {
                    for (entry_key, entry_val) in pairs {
                        stack.push((entry_key, depth + 1));
                        stack.push((entry_val, depth + 1));
                    }
                }
                _ => {}
            }
        }
    }

    fn check_graph_bounds(&self) -> JResult<()> {
        let mut stack = vec![(self.clone(), 0_usize)];
        let mut visited = 0_usize;
        let mut retained = 0_usize;
        while let Some((node, depth)) = stack.pop() {
            visited += 1;
            retained += 32;
            if visited > MAX_DAG_VISITS || depth > MAX_GRAPH_DEPTH || retained > MAX_OUTPUT_BYTES {
                return Err(JinjaError::std("Jinja value graph bound exceeded"));
            }
            match node.kind_cloned() {
                ValKind::String(spans) => {
                    let span_bytes: usize = spans.iter().map(|span| span.bytes.len()).sum();
                    retained += span_bytes;
                    if retained > MAX_OUTPUT_BYTES {
                        return Err(JinjaError::std("Jinja retained string byte bound exceeded"));
                    }
                }
                ValKind::Array(elems) | ValKind::Tuple(elems) => {
                    for elem in elems {
                        stack.push((elem, depth + 1));
                    }
                }
                ValKind::Object(pairs) => {
                    for (entry_key, entry_val) in pairs {
                        stack.push((entry_key, depth + 1));
                        stack.push((entry_val, depth + 1));
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn ensure_acyclic_insert(&self, candidate: &Self) -> JResult<()> {
        candidate.check_graph_bounds()?;
        let mut seen = HashSet::new();
        let mut stack = vec![(candidate.clone(), 0_usize)];
        let mut steps = 0_usize;
        while let Some((node, depth)) = stack.pop() {
            steps += 1;
            if depth > MAX_GRAPH_DEPTH || steps > MAX_DAG_VISITS {
                return Err(JinjaError::std(
                    "Jinja container insertion traversal bound exceeded",
                ));
            }
            if Rc::ptr_eq(&self.0, &node.0) {
                return Err(JinjaError::std("cyclic Jinja container assignment"));
            }
            if !seen.insert(Rc::as_ptr(&node.0)) {
                continue;
            }
            match node.kind_cloned() {
                ValKind::Array(elems) | ValKind::Tuple(elems) => {
                    for elem in elems {
                        stack.push((elem, depth + 1));
                    }
                }
                ValKind::Object(pairs) => {
                    for (entry_key, entry_val) in pairs {
                        stack.push((entry_key, depth + 1));
                        stack.push((entry_val, depth + 1));
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn truthy(&self) -> bool {
        self.mark_used();
        let cell = self.0.borrow();
        match &cell.kind {
            ValKind::Undefined | ValKind::None => false,
            ValKind::Bool(flag) => *flag,
            ValKind::Int(num) => *num != 0,
            ValKind::Float(num) => *num != 0.0,
            ValKind::String(spans) => spans.iter().any(|span| !span.bytes.is_empty()),
            ValKind::Array(elems) | ValKind::Tuple(elems) => !elems.is_empty(),
            ValKind::Object(pairs) => !pairs.is_empty(),
            _ => true,
        }
    }

    #[expect(
        clippy::cast_precision_loss,
        clippy::as_conversions,
        reason = "Jinja floating-point coercions widen i64 operands to f64 while preserving integer-only i64 arithmetic"
    )]
    fn as_f64(&self) -> JResult<f64> {
        self.mark_used();
        let cell = self.0.borrow();
        match cell.kind {
            ValKind::Bool(flag) => Ok(f64::from(flag)),
            ValKind::Int(num) => Ok(num as f64),
            ValKind::Float(num) => Ok(num),
            _ => Err(JinjaError::std("Jinja value is not numeric")),
        }
    }

    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::as_conversions,
        reason = "Jinja float-to-int coercion checks finite range before truncating to i64"
    )]
    fn as_i64(&self) -> JResult<i64> {
        self.mark_used();
        let cell = self.0.borrow();
        match cell.kind {
            ValKind::Bool(flag) => Ok(i64::from(flag)),
            ValKind::Int(num) => Ok(num),
            ValKind::Float(num)
                if num.is_finite() && num >= i64::MIN as f64 && num < i64::MAX as f64 =>
            {
                Ok(num as i64)
            }
            _ => Err(JinjaError::std("Jinja value is not an integer")),
        }
    }

    pub(super) fn raw_bytes(&self) -> Vec<u8> {
        self.mark_recursive();
        let mut out = Vec::new();
        let mut steps = 0_usize;
        let _ = self.write_text(&mut out, 0, &mut steps);
        out
    }

    fn write_text(&self, out: &mut Vec<u8>, depth: usize, steps: &mut usize) -> bool {
        *steps += 1;
        if depth > MAX_GRAPH_DEPTH || *steps > MAX_DAG_VISITS || out.len() >= MAX_TEXT_BYTES {
            return false;
        }
        match self.kind_cloned() {
            ValKind::Undefined => true,
            ValKind::None => append_bounded(out, b"None"),
            ValKind::Bool(flag) => append_bounded(out, if flag { b"True" } else { b"False" }),
            ValKind::Int(num) => append_bounded(out, num.to_string().as_bytes()),
            ValKind::Float(num) => {
                let mut text = num.to_string();
                if !text.contains(['.', 'e', 'E']) {
                    text.push_str(".0");
                }
                append_bounded(out, text.as_bytes())
            }
            ValKind::String(spans) => {
                for span in spans {
                    if !append_bounded(out, &span.bytes) {
                        return false;
                    }
                }
                true
            }
            ValKind::Array(elems) => write_seq(&elems, b"[", b"]", false, out, depth + 1, steps),
            ValKind::Tuple(elems) => write_seq(&elems, b"(", b")", true, out, depth + 1, steps),
            ValKind::Object(pairs) => {
                if !append_bounded(out, b"{") {
                    return false;
                }
                for (idx, (entry_key, entry_val)) in pairs.iter().enumerate() {
                    if idx != 0 && !append_bounded(out, b", ") {
                        return false;
                    }
                    if !entry_key.write_repr(out, depth + 1, steps)
                        || !append_bounded(out, b": ")
                        || !entry_val.write_repr(out, depth + 1, steps)
                    {
                        return false;
                    }
                }
                append_bounded(out, b"}")
            }
            _ => append_bounded(out, b"<function>"),
        }
    }

    fn write_repr(&self, out: &mut Vec<u8>, depth: usize, steps: &mut usize) -> bool {
        *steps += 1;
        if depth > MAX_GRAPH_DEPTH || *steps > MAX_DAG_VISITS || out.len() >= MAX_TEXT_BYTES {
            return false;
        }
        if let ValKind::String(spans) = self.kind_cloned() {
            if !append_bounded(out, b"'") {
                return false;
            }
            for span in spans {
                for byte in span.bytes {
                    let ok = match byte {
                        b'\'' => append_bounded(out, b"\\'"),
                        b'\\' => append_bounded(out, b"\\\\"),
                        b'\n' => append_bounded(out, b"\\n"),
                        b'\r' => append_bounded(out, b"\\r"),
                        b'\t' => append_bounded(out, b"\\t"),
                        _ => append_bounded(out, &[byte]),
                    };
                    if !ok {
                        return false;
                    }
                }
            }
            append_bounded(out, b"'")
        } else {
            self.write_text(out, depth, steps)
        }
    }

    fn to_spans(&self) -> Vec<TextSpan> {
        self.mark_recursive();
        if let ValKind::String(spans) = self.kind_cloned() {
            spans
        } else {
            vec![TextSpan {
                bytes: self.raw_bytes(),
                is_input: false,
            }]
        }
    }

    fn length(&self) -> JResult<usize> {
        self.mark_used();
        let cell = self.0.borrow();
        match &cell.kind {
            ValKind::Undefined => Ok(0),
            ValKind::String(spans) => Ok(spans.iter().map(|span| span.bytes.len()).sum()),
            ValKind::Array(elems) | ValKind::Tuple(elems) => Ok(elems.len()),
            ValKind::Object(pairs) => Ok(pairs.len()),
            _ => Err(JinjaError::std("Jinja value has no length")),
        }
    }

    fn iter_elements(&self) -> JResult<Vec<Self>> {
        self.mark_used();
        match self.kind_cloned() {
            ValKind::Undefined => Ok(Vec::new()),
            ValKind::Array(elems) | ValKind::Tuple(elems) => Ok(elems),
            ValKind::Object(pairs) => Ok(pairs.into_iter().map(|(key, _)| key).collect()),
            ValKind::String(spans) => Ok(spans
                .into_iter()
                .flat_map(|span| {
                    let flag = span.is_input;
                    span.bytes
                        .into_iter()
                        .map(move |byte| Self::bytes(vec![byte], flag))
                })
                .collect()),
            _ => Err(JinjaError::std("Jinja value is not iterable")),
        }
    }

    fn get_member(&self, key: &Self, track_usage: bool) -> Self {
        if track_usage {
            self.mark_used();
            self.0.borrow_mut().array_access = true;
        }
        match self.kind_cloned() {
            ValKind::Object(pairs) => pairs
                .into_iter()
                .find(|(entry_key, _)| vals_equal(entry_key, key))
                .map_or_else(Self::undefined, |(_, entry_val)| entry_val),
            ValKind::Array(elems) | ValKind::Tuple(elems) => key
                .as_i64()
                .ok()
                .and_then(|idx| resolve_index(idx, elems.len()))
                .and_then(|idx| elems.get(idx).cloned())
                .unwrap_or_else(Self::undefined),
            ValKind::String(spans) => {
                let all_input = spans.iter().any(|span| span.is_input);
                let flat: Vec<u8> = spans
                    .into_iter()
                    .flat_map(|span| span.bytes.into_iter())
                    .collect();
                key.as_i64()
                    .ok()
                    .and_then(|idx| resolve_index(idx, flat.len()))
                    .map_or_else(Self::undefined, |idx| {
                        Self::bytes(vec![flat[idx]], all_input)
                    })
            }
            _ => Self::undefined(),
        }
    }

    pub(super) fn set_member(&self, key: Self, value: Self) -> JResult<()> {
        self.ensure_acyclic_insert(&value)?;
        self.ensure_acyclic_insert(&key)?;
        let kind_snapshot = self.kind_cloned();
        match kind_snapshot {
            ValKind::Object(_) => {
                let mut cell = self.0.borrow_mut();
                if let ValKind::Object(pairs) = &mut cell.kind {
                    if let Some((_, existing_val)) =
                        pairs.iter_mut().find(|(k, _)| vals_equal(k, &key))
                    {
                        *existing_val = value;
                    } else {
                        pairs.push((key, value));
                    }
                }
                Ok(())
            }
            ValKind::Array(elems) => {
                let idx = resolve_index(key.as_i64()?, elems.len())
                    .ok_or_else(|| JinjaError::std("Jinja array assignment index out of range"))?;
                let mut cell = self.0.borrow_mut();
                if let ValKind::Array(target) = &mut cell.kind {
                    target[idx] = value;
                }
                Ok(())
            }
            _ => Err(JinjaError::std(
                "Jinja item assignment requires a mutable container",
            )),
        }
    }

    pub(super) fn lookup_exact_key(&self, wanted: &[u8]) -> Option<Self> {
        let cell = self.0.borrow();
        let ValKind::Object(pairs) = &cell.kind else {
            return None;
        };
        for (entry_key, entry_val) in pairs {
            if entry_key.raw_string_bytes_eq(wanted) {
                return Some(entry_val.clone());
            }
        }
        None
    }

    pub(super) fn lookup_exact_index(&self, idx: usize) -> Option<Self> {
        let cell = self.0.borrow();
        let ValKind::Array(elems) = &cell.kind else {
            return None;
        };
        elems.get(idx).cloned()
    }

    fn raw_string_bytes_eq(&self, wanted: &[u8]) -> bool {
        let cell = self.0.borrow();
        let ValKind::String(spans) = &cell.kind else {
            return false;
        };
        let total: usize = spans.iter().map(|span| span.bytes.len()).sum();
        total == wanted.len()
            && spans
                .iter()
                .flat_map(|span| span.bytes.iter().copied())
                .eq(wanted.iter().copied())
    }

    fn replace_with_input_bytes(&self, raw: Vec<u8>) -> Result<()> {
        let mut cell = self.0.borrow_mut();
        if !matches!(cell.kind, ValKind::String(_)) {
            return Err(invalid("ByteStrings must replace an existing string"));
        }
        cell.kind = ValKind::String(vec![TextSpan {
            bytes: raw,
            is_input: true,
        }]);
        Ok(())
    }

    pub(super) fn deep_clone_fresh(&self) -> Self {
        match self.kind_cloned() {
            ValKind::Undefined => Self::undefined(),
            ValKind::None => Self::none(),
            ValKind::Bool(flag) => Self::boolean(flag),
            ValKind::Int(num) => Self::int(num),
            ValKind::Float(num) => Self::float(num),
            ValKind::String(spans) => Self::new(ValKind::String(spans)),
            ValKind::Array(elems) => Self::new(ValKind::Array(
                elems.iter().map(Self::deep_clone_fresh).collect(),
            )),
            ValKind::Tuple(elems) => Self::new(ValKind::Tuple(
                elems.iter().map(Self::deep_clone_fresh).collect(),
            )),
            ValKind::Object(pairs) => {
                let cloned = Self::new(ValKind::Object(
                    pairs
                        .iter()
                        .map(|(k, v)| (k.deep_clone_fresh(), v.deep_clone_fresh()))
                        .collect(),
                ));
                if self.is_namespace() {
                    cloned.set_namespace();
                }
                cloned
            }
            other => Self::new(other),
        }
    }
}

fn append_bounded(out: &mut Vec<u8>, bytes: &[u8]) -> bool {
    let remaining = MAX_TEXT_BYTES.saturating_sub(out.len());
    if remaining == 0 {
        return false;
    }
    let take = bytes.len().min(remaining);
    out.extend_from_slice(&bytes[..take]);
    take == bytes.len()
}

fn write_seq(
    elems: &[JVal],
    open: &[u8],
    close: &[u8],
    is_tuple: bool,
    out: &mut Vec<u8>,
    depth: usize,
    steps: &mut usize,
) -> bool {
    if !append_bounded(out, open) {
        return false;
    }
    for (idx, elem) in elems.iter().enumerate() {
        if idx != 0 && !append_bounded(out, b", ") {
            return false;
        }
        if !elem.write_repr(out, depth, steps) {
            return false;
        }
    }
    if is_tuple && elems.len() == 1 && !append_bounded(out, b",") {
        return false;
    }
    append_bounded(out, close)
}

fn resolve_index(idx: i64, len: usize) -> Option<usize> {
    let adjusted = if idx < 0 {
        i64::try_from(len).ok()?.checked_add(idx)?
    } else {
        idx
    };
    usize::try_from(adjusted).ok().filter(|&pos| pos < len)
}

fn vals_equal(lhs: &JVal, rhs: &JVal) -> bool {
    let mut steps = 0_usize;
    vals_equal_bounded(lhs, rhs, 0, &mut steps)
}

fn vals_equal_bounded(lhs: &JVal, rhs: &JVal, depth: usize, steps: &mut usize) -> bool {
    *steps += 1;
    if depth > MAX_GRAPH_DEPTH || *steps > MAX_DAG_VISITS {
        return false;
    }
    match (lhs.kind_cloned(), rhs.kind_cloned()) {
        (ValKind::Undefined, ValKind::Undefined) | (ValKind::None, ValKind::None) => true,
        (ValKind::Int(a), ValKind::Int(b)) => {
            lhs.mark_used();
            rhs.mark_used();
            a == b
        }
        (ValKind::Bool(a), ValKind::Bool(b)) => {
            lhs.mark_used();
            rhs.mark_used();
            a == b
        }
        (
            ValKind::Bool(_) | ValKind::Int(_) | ValKind::Float(_),
            ValKind::Bool(_) | ValKind::Int(_) | ValKind::Float(_),
        ) => lhs.as_f64().ok() == rhs.as_f64().ok(),
        (ValKind::String(a), ValKind::String(b)) => {
            lhs.mark_used();
            rhs.mark_used();
            a.iter()
                .flat_map(|span| span.bytes.iter().copied())
                .eq(b.iter().flat_map(|span| span.bytes.iter().copied()))
        }
        (ValKind::Array(a), ValKind::Array(b)) | (ValKind::Tuple(a), ValKind::Tuple(b)) => {
            a.len() == b.len()
                && a.iter()
                    .zip(&b)
                    .all(|(x, y)| vals_equal_bounded(x, y, depth + 1, steps))
        }
        (ValKind::Object(a), ValKind::Object(b)) => {
            a.len() == b.len()
                && a.iter().all(|(ka, va)| {
                    b.iter().any(|(kb, vb)| {
                        vals_equal_bounded(ka, kb, depth + 1, steps)
                            && vals_equal_bounded(va, vb, depth + 1, steps)
                    })
                })
        }
        _ => false,
    }
}

fn vals_compare(lhs: &JVal, rhs: &JVal) -> JResult<Ordering> {
    match (lhs.kind_cloned(), rhs.kind_cloned()) {
        (ValKind::String(a), ValKind::String(b)) => {
            lhs.mark_used();
            rhs.mark_used();
            Ok(a.iter()
                .flat_map(|span| span.bytes.iter().copied())
                .cmp(b.iter().flat_map(|span| span.bytes.iter().copied())))
        }
        (ValKind::Int(a), ValKind::Int(b)) => {
            lhs.mark_used();
            rhs.mark_used();
            Ok(a.cmp(&b))
        }
        _ => lhs
            .as_f64()?
            .partial_cmp(&rhs.as_f64()?)
            .ok_or_else(|| JinjaError::std("Jinja unordered numeric comparison")),
    }
}

fn val_contains(container: &JVal, needle: &JVal) -> JResult<bool> {
    container.mark_used();
    match container.kind_cloned() {
        ValKind::String(_) => {
            let sub = needle.raw_bytes();
            let hay = container.raw_bytes();
            Ok(sub.is_empty() || hay.windows(sub.len()).any(|win| win == sub))
        }
        ValKind::Object(pairs) => Ok(pairs.iter().any(|(k, _)| vals_equal(k, needle))),
        _ => Ok(container
            .iter_elements()?
            .iter()
            .any(|item| vals_equal(item, needle))),
    }
}

fn merge_spans(spans: Vec<TextSpan>) -> Vec<TextSpan> {
    let mut merged: Vec<TextSpan> = Vec::new();
    for span in spans {
        if span.bytes.is_empty() {
            continue;
        }
        if let Some(prev) = merged
            .last_mut()
            .filter(|prev| prev.is_input == span.is_input)
        {
            prev.bytes.extend(span.bytes);
        } else {
            merged.push(span);
        }
    }
    merged
}

// ---------------------------------------------------------------------------
// Ordered JSON variables parser (depth & duplicate-key strict)
// ---------------------------------------------------------------------------

struct VarSeed {
    depth: usize,
    max_depth: usize,
}

impl<'de> DeserializeSeed<'de> for VarSeed {
    type Value = JVal;

    fn deserialize<D: de::Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> std::result::Result<JVal, D::Error> {
        if self.depth > self.max_depth {
            return Err(de::Error::custom("variables nesting exceeds limit"));
        }
        deserializer.deserialize_any(VarVisitor {
            depth: self.depth,
            max_depth: self.max_depth,
        })
    }
}

struct VarVisitor {
    depth: usize,
    max_depth: usize,
}

impl<'de> Visitor<'de> for VarVisitor {
    type Value = JVal;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a bounded JSON value")
    }

    fn visit_bool<E: de::Error>(self, val: bool) -> std::result::Result<JVal, E> {
        Ok(JVal::boolean(val))
    }

    fn visit_i64<E: de::Error>(self, val: i64) -> std::result::Result<JVal, E> {
        Ok(JVal::int(val))
    }

    fn visit_u64<E: de::Error>(self, val: u64) -> std::result::Result<JVal, E> {
        i64::try_from(val)
            .map(JVal::int)
            .map_err(|_| de::Error::custom("variables integer exceeds signed 64-bit range"))
    }

    fn visit_f64<E: de::Error>(self, val: f64) -> std::result::Result<JVal, E> {
        Ok(JVal::float(val))
    }

    fn visit_str<E: de::Error>(self, val: &str) -> std::result::Result<JVal, E> {
        Ok(JVal::bytes(val.as_bytes().to_vec(), true))
    }

    fn visit_none<E: de::Error>(self) -> std::result::Result<JVal, E> {
        Ok(JVal::none())
    }

    fn visit_unit<E: de::Error>(self) -> std::result::Result<JVal, E> {
        Ok(JVal::none())
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<JVal, A::Error> {
        let mut elems = Vec::new();
        while let Some(item) = seq.next_element_seed(VarSeed {
            depth: self.depth + 1,
            max_depth: self.max_depth,
        })? {
            elems.push(item);
        }
        Ok(JVal::new(ValKind::Array(elems)))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> std::result::Result<JVal, A::Error> {
        let mut seen = HashSet::new();
        let mut pairs = Vec::new();
        while let Some(key) = map.next_key::<String>()? {
            if !seen.insert(key.clone()) {
                return Err(de::Error::custom("duplicate variables object key"));
            }
            let val = map.next_value_seed(VarSeed {
                depth: self.depth + 1,
                max_depth: self.max_depth,
            })?;
            pairs.push((JVal::text_str(key), val));
        }
        Ok(JVal::new(ValKind::Object(pairs)))
    }
}

pub(super) fn parse_json_jval_bounded(
    text: &str,
    max_bytes: usize,
    max_depth: usize,
) -> Result<JVal> {
    if text.len() > max_bytes {
        return Err(invalid("JSON variables exceed byte limit"));
    }
    let mut de = serde_json::Deserializer::from_str(text);
    let root = VarSeed {
        depth: 0,
        max_depth,
    }
    .deserialize(&mut de)
    .map_err(|err| invalid(format!("invalid variables JSON: {err}")))?;
    de.end()
        .map_err(|err| invalid(format!("trailing variables JSON: {err}")))?;
    Ok(root)
}

pub(super) fn parse_variables_json(text: &str) -> Result<JVal> {
    let root = parse_json_jval_bounded(text, MAX_VARIABLES_BYTES, MAX_JSON_DEPTH)?;
    if !matches!(root.kind_cloned(), ValKind::Object(_)) {
        return Err(invalid("variables JSON must contain an object"));
    }
    Ok(root)
}

// ---------------------------------------------------------------------------
// Lexer & AST Parser
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Literal {
    Int(i64),
    Float(f64),
    Bytes(Vec<u8>),
}

#[derive(Clone, Debug)]
enum CallArg {
    Pos(Expr),
    Kw(String, Expr),
    Spread(Expr),
}

#[derive(Clone, Debug)]
enum Expr {
    Lit(Literal),
    Var(String),
    Array(Vec<Self>),
    Tuple(Vec<Self>),
    Object(Vec<(Self, Self)>),
    Member(Box<Self>, Box<Self>),
    Attr(Box<Self>, String),
    Slice(Option<Box<Self>>, Option<Box<Self>>, Option<Box<Self>>),
    Blank,
    Call(Box<Self>, Vec<CallArg>),
    Filter(Box<Self>, String, Vec<CallArg>),
    Test(Box<Self>, String, Vec<CallArg>, bool),
    Unary(String, Box<Self>),
    Binary(String, Box<Self>, Box<Self>),
    Ternary(Box<Self>, Box<Self>, Box<Self>),
    Select(Box<Self>, Box<Self>),
}

#[derive(Clone, Debug)]
enum Stmt {
    Text(Vec<u8>),
    Output(Expr),
    Set(Expr, Option<Expr>, Vec<Self>),
    If(Vec<(Expr, Vec<Self>)>, Vec<Self>),
    For(Expr, Expr, Vec<Self>, Vec<Self>),
    Macro(String, Vec<(String, Option<Expr>)>, Vec<Self>),
    Call(Vec<String>, Expr, Vec<Self>),
    Filter(Expr, Vec<Self>),
    Break,
    Continue,
}

#[derive(Clone, Debug)]
enum Tok {
    Id(String),
    Num(String),
    Str(Vec<u8>),
    Op(String),
}

fn lex_expr(text: &str) -> JResult<Vec<Tok>> {
    let bytes = text.as_bytes();
    let mut pos = 0_usize;
    let mut tokens = Vec::new();
    while pos < bytes.len() {
        let byte = bytes[pos];
        if byte.is_ascii_whitespace() {
            pos += 1;
            continue;
        }
        if byte.is_ascii_alphabetic() || byte == b'_' {
            let start = pos;
            pos += 1;
            while bytes
                .get(pos)
                .is_some_and(|ch| ch.is_ascii_alphanumeric() || *ch == b'_')
            {
                pos += 1;
            }
            tokens.push(Tok::Id(text[start..pos].to_owned()));
        } else if byte.is_ascii_digit() {
            let start = pos;
            while bytes.get(pos).is_some_and(u8::is_ascii_digit) {
                pos += 1;
            }
            if bytes.get(pos) == Some(&b'.') && bytes.get(pos + 1).is_some_and(u8::is_ascii_digit) {
                pos += 1;
                while bytes.get(pos).is_some_and(u8::is_ascii_digit) {
                    pos += 1;
                }
            }
            tokens.push(Tok::Num(text[start..pos].to_owned()));
        } else if matches!(byte, b'\'' | b'"') {
            let quote = byte;
            pos += 1;
            let mut raw = Vec::new();
            while pos < bytes.len() && bytes[pos] != quote {
                let ch = bytes[pos];
                pos += 1;
                if ch == b'\\' {
                    let esc = *bytes
                        .get(pos)
                        .ok_or_else(|| JinjaError::std("trailing escape in Jinja string"))?;
                    pos += 1;
                    raw.push(match esc {
                        b'n' => b'\n',
                        b't' => b'\t',
                        b'r' => b'\r',
                        b'b' => 8,
                        b'f' => 12,
                        b'v' => 11,
                        b'\\' | b'\'' | b'"' => esc,
                        _ => return Err(JinjaError::std("unknown escape in Jinja string")),
                    });
                } else {
                    raw.push(ch);
                }
            }
            if pos >= bytes.len() {
                return Err(JinjaError::std("unterminated Jinja string literal"));
            }
            pos += 1;
            tokens.push(Tok::Str(raw));
        } else {
            if !byte.is_ascii() {
                return Err(JinjaError::std("non-ASCII character in Jinja expression"));
            }
            let two_char = bytes
                .get(pos..pos + 2)
                .is_some_and(|pair| matches!(pair, b"==" | b"!=" | b"<=" | b">=" | b"//"));
            let width = if two_char { 2 } else { 1 };
            let slice = &text[pos..pos + width];
            if !slice
                .bytes()
                .all(|ch| b"+-~*/%<>=!()[]{}.,:|".contains(&ch))
            {
                return Err(JinjaError::std("unexpected character in Jinja expression"));
            }
            tokens.push(Tok::Op(slice.to_owned()));
            pos += width;
        }
        if tokens.len() > MAX_EXPR_TOKENS {
            return Err(JinjaError::std("Jinja expression token limit exceeded"));
        }
    }
    Ok(tokens)
}

struct ExprParser {
    tokens: Vec<Tok>,
    cursor: usize,
    depth: usize,
}

impl ExprParser {
    fn new(text: &str) -> JResult<Self> {
        Ok(Self {
            tokens: lex_expr(text)?,
            cursor: 0,
            depth: 0,
        })
    }

    fn peek_word(&self, expected: &str) -> bool {
        matches!(
            self.tokens.get(self.cursor),
            Some(Tok::Id(word) | Tok::Op(word)) if word == expected
        )
    }

    fn consume_word(&mut self, expected: &str) -> bool {
        if self.peek_word(expected) {
            self.cursor += 1;
            true
        } else {
            false
        }
    }

    fn expect_word(&mut self, expected: &str) -> JResult<()> {
        if self.consume_word(expected) {
            Ok(())
        } else {
            Err(JinjaError::std(format!(
                "expected Jinja token '{expected}'"
            )))
        }
    }

    fn identifier(&mut self) -> JResult<String> {
        if let Some(Tok::Id(name)) = self.tokens.get(self.cursor) {
            let owned = name.clone();
            self.cursor += 1;
            Ok(owned)
        } else {
            Err(JinjaError::std("expected Jinja identifier"))
        }
    }

    fn ensure_eof(&self) -> JResult<()> {
        if self.cursor == self.tokens.len() {
            Ok(())
        } else {
            Err(JinjaError::std("unexpected trailing Jinja tokens"))
        }
    }

    fn parse_complete(mut self) -> JResult<Expr> {
        let expr = self.parse_sequence()?;
        self.ensure_eof()?;
        Ok(expr)
    }

    fn enter_depth(&mut self) -> JResult<()> {
        self.depth += 1;
        if self.depth > MAX_PARSE_DEPTH {
            return Err(JinjaError::std("Jinja expression nesting depth exceeded"));
        }
        Ok(())
    }

    fn parse_sequence(&mut self) -> JResult<Expr> {
        let first = self.parse_expr()?;
        if !self.consume_word(",") {
            return Ok(first);
        }
        let mut items = vec![first];
        while self.cursor < self.tokens.len() && !self.peek_word(")") && !self.peek_word("=") {
            items.push(self.parse_expr()?);
            if !self.consume_word(",") {
                break;
            }
        }
        Ok(Expr::Tuple(items))
    }

    fn parse_expr(&mut self) -> JResult<Expr> {
        self.enter_depth()?;
        let base = self.parse_binary(0);
        if let Ok(mut value) = base {
            if self.consume_word("if") {
                let cond = self.parse_binary(0);
                value = match cond {
                    Ok(cond) => {
                        if self.consume_word("else") {
                            match self.parse_expr() {
                                Ok(alt) => {
                                    Expr::Ternary(Box::new(cond), Box::new(value), Box::new(alt))
                                }
                                Err(err) => {
                                    self.depth -= 1;
                                    return Err(err);
                                }
                            }
                        } else {
                            Expr::Select(Box::new(value), Box::new(cond))
                        }
                    }
                    Err(err) => {
                        self.depth -= 1;
                        return Err(err);
                    }
                };
            }
            self.depth -= 1;
            Ok(value)
        } else {
            self.depth -= 1;
            base
        }
    }

    fn parse_binary(&mut self, level: usize) -> JResult<Expr> {
        if level == 2 && self.consume_word("not") {
            self.enter_depth()?;
            let inner = self.parse_binary(level);
            self.depth -= 1;
            return Ok(Expr::Unary("not".to_owned(), Box::new(inner?)));
        }
        if level >= 6 {
            return self.parse_test();
        }
        let mut lhs = self.parse_binary(level + 1)?;
        let mut chain = 0_usize;
        while let Some(Tok::Id(tok) | Tok::Op(tok)) = self.tokens.get(self.cursor) {
            let op = if tok == "not"
                && matches!(self.tokens.get(self.cursor + 1), Some(Tok::Id(next)) if next == "in")
            {
                "not in".to_owned()
            } else {
                tok.clone()
            };
            let matches_level = match level {
                0 => op == "or",
                1 => op == "and",
                3 => matches!(
                    op.as_str(),
                    "in" | "not in" | "==" | "!=" | "<" | ">" | "<=" | ">="
                ),
                4 => matches!(op.as_str(), "+" | "-" | "~"),
                5 => matches!(op.as_str(), "*" | "/" | "//" | "%"),
                _ => false,
            };
            if !matches_level {
                break;
            }
            chain += 1;
            if chain > MAX_PARSE_DEPTH {
                return Err(JinjaError::std(
                    "Jinja binary operator chain limit exceeded",
                ));
            }
            self.cursor += if op == "not in" { 2 } else { 1 };
            let rhs = self.parse_binary(level + 1)?;
            lhs = Expr::Binary(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_test(&mut self) -> JResult<Expr> {
        let mut value = self.parse_filter()?;
        let mut chain = 0_usize;
        while self.consume_word("is") {
            chain += 1;
            if self.depth + chain > MAX_PARSE_DEPTH {
                return Err(JinjaError::std("Jinja test chain nesting depth exceeded"));
            }
            let negated = self.consume_word("not");
            let name = self.identifier()?;
            let args = if self.peek_word("(") {
                self.parse_args()?
            } else {
                Vec::new()
            };
            value = Expr::Test(Box::new(value), name, args, negated);
        }
        Ok(value)
    }

    fn parse_filter(&mut self) -> JResult<Expr> {
        let mut value = self.parse_unary()?;
        let mut chain = 0_usize;
        while self.consume_word("|") {
            chain += 1;
            if self.depth + chain > MAX_PARSE_DEPTH {
                return Err(JinjaError::std("Jinja filter chain nesting depth exceeded"));
            }
            let name = self.identifier()?;
            let args = if self.peek_word("(") {
                self.parse_args()?
            } else {
                Vec::new()
            };
            value = Expr::Filter(Box::new(value), name, args);
        }
        Ok(value)
    }

    fn parse_unary(&mut self) -> JResult<Expr> {
        if let Some(Tok::Op(op)) = self.tokens.get(self.cursor)
            && (op == "+" || op == "-")
        {
            let op_owned = op.clone();
            self.cursor += 1;
            self.enter_depth()?;
            let inner = self.parse_unary();
            self.depth -= 1;
            return Ok(Expr::Unary(op_owned, Box::new(inner?)));
        }
        self.parse_postfix()
    }

    fn parse_args(&mut self) -> JResult<Vec<CallArg>> {
        self.expect_word("(")?;
        let mut out = Vec::new();
        while !self.peek_word(")") {
            let arg = if self.consume_word("*") {
                CallArg::Spread(self.parse_expr()?)
            } else {
                let expr = self.parse_expr()?;
                if self.consume_word("=") {
                    let Expr::Var(name) = expr else {
                        return Err(JinjaError::std("invalid Jinja keyword argument name"));
                    };
                    CallArg::Kw(name, self.parse_expr()?)
                } else {
                    CallArg::Pos(expr)
                }
            };
            out.push(arg);
            if !self.consume_word(",") {
                break;
            }
        }
        self.expect_word(")")?;
        Ok(out)
    }

    fn parse_postfix(&mut self) -> JResult<Expr> {
        let mut expr = self.parse_primary()?;
        let mut steps = 0_usize;
        loop {
            steps += 1;
            if steps > MAX_PARSE_DEPTH {
                return Err(JinjaError::std("Jinja postfix chain limit exceeded"));
            }
            if self.consume_word(".") {
                let member = self.identifier()?;
                expr = Expr::Attr(Box::new(expr), member);
            } else if self.consume_word("[") {
                let start = if self.peek_word(":") || self.peek_word("]") {
                    None
                } else {
                    Some(Box::new(self.parse_expr()?))
                };
                let key_expr = if self.consume_word(":") {
                    let stop = if self.peek_word(":") || self.peek_word("]") {
                        None
                    } else {
                        Some(Box::new(self.parse_expr()?))
                    };
                    let stride = if self.consume_word(":") && !self.peek_word("]") {
                        Some(Box::new(self.parse_expr()?))
                    } else {
                        None
                    };
                    Expr::Slice(start, stop, stride)
                } else {
                    start.map_or(Expr::Blank, |inner| *inner)
                };
                self.expect_word("]")?;
                expr = Expr::Member(Box::new(expr), Box::new(key_expr));
            } else if self.peek_word("(") {
                expr = Expr::Call(Box::new(expr), self.parse_args()?);
            } else {
                break;
            }
        }
        Ok(expr)
    }

    fn parse_primary(&mut self) -> JResult<Expr> {
        let token = self
            .tokens
            .get(self.cursor)
            .cloned()
            .ok_or_else(|| JinjaError::std("unexpected end of Jinja expression"))?;
        self.cursor += 1;
        match token {
            Tok::Id(name) => Ok(Expr::Var(name)),
            Tok::Num(text) => {
                let lit = if text.contains('.') {
                    let parsed = text
                        .parse::<f64>()
                        .map_err(|_| JinjaError::std("invalid Jinja float literal"))?;
                    Literal::Float(parsed)
                } else {
                    let parsed = text
                        .parse::<i64>()
                        .map_err(|_| JinjaError::std("Jinja integer literal out of range"))?;
                    Literal::Int(parsed)
                };
                Ok(Expr::Lit(lit))
            }
            Tok::Str(mut bytes) => {
                while let Some(Tok::Str(more)) = self.tokens.get(self.cursor) {
                    bytes.extend(more);
                    self.cursor += 1;
                }
                Ok(Expr::Lit(Literal::Bytes(bytes)))
            }
            Tok::Op(op) if op == "(" => {
                if self.consume_word(")") {
                    return Ok(Expr::Tuple(Vec::new()));
                }
                let inner = self.parse_sequence()?;
                self.expect_word(")")?;
                Ok(inner)
            }
            Tok::Op(op) if op == "[" => {
                let mut elems = Vec::new();
                while !self.peek_word("]") {
                    elems.push(self.parse_expr()?);
                    if !self.consume_word(",") {
                        break;
                    }
                }
                self.expect_word("]")?;
                Ok(Expr::Array(elems))
            }
            Tok::Op(op) if op == "{" => {
                let mut entries = Vec::new();
                while !self.peek_word("}") {
                    let entry_key = self.parse_expr()?;
                    self.expect_word(":")?;
                    let entry_val = self.parse_expr()?;
                    entries.push((entry_key, entry_val));
                    if !self.consume_word(",") {
                        break;
                    }
                }
                self.expect_word("}")?;
                Ok(Expr::Object(entries))
            }
            Tok::Op(_) => Err(JinjaError::std("unexpected Jinja expression token")),
        }
    }
}

enum BlockSegment {
    Text(Vec<u8>),
    Expr(String),
    Tag(String),
}

fn scan_template_segments(source: &str, max_bytes: usize) -> JResult<Vec<BlockSegment>> {
    if source.len() > max_bytes {
        return Err(JinjaError::std("Jinja template exceeds byte bound"));
    }
    let trailing_nl = source.ends_with('\n');
    let mut normalized = source.replace("\r\n", "\n").replace('\r', "\n");
    if trailing_nl {
        normalized.pop();
    }
    let bytes = normalized.as_bytes();
    let mut segments = Vec::new();
    let mut pos = 0_usize;
    let mut strip_next = false;

    while pos < bytes.len() {
        let found_open = (pos..bytes.len().saturating_sub(1))
            .find(|&idx| bytes[idx] == b'{' && matches!(bytes[idx + 1], b'{' | b'%' | b'#'));
        let text_end = found_open.unwrap_or(bytes.len());
        let mut literal = bytes[pos..text_end].to_vec();
        if strip_next {
            let leading_ws = literal
                .iter()
                .take_while(|byte| byte.is_ascii_whitespace())
                .count();
            literal.drain(..leading_ws);
            strip_next = false;
        }
        let Some(open_idx) = found_open else {
            if !literal.is_empty() {
                segments.push(BlockSegment::Text(literal));
            }
            break;
        };

        let tag_kind = bytes[open_idx + 1];
        let trim_left = bytes.get(open_idx + 2) == Some(&b'-');
        if trim_left {
            while literal.pop_if(|byte| byte.is_ascii_whitespace()).is_some() {}
        } else if matches!(tag_kind, b'%' | b'#') {
            let line_start = bytes[..open_idx]
                .iter()
                .rposition(|&byte| byte == b'\n')
                .map_or(0, |idx| idx + 1);
            if bytes[line_start..open_idx]
                .iter()
                .all(|&byte| matches!(byte, b' ' | b'\t'))
            {
                let trimmed_len = literal.len().saturating_sub(open_idx - line_start);
                literal.truncate(trimmed_len);
            }
        }
        if !literal.is_empty() {
            segments.push(BlockSegment::Text(literal));
        }

        let body_start = open_idx + if trim_left { 3 } else { 2 };
        let close_idx = find_closing_delimiter(bytes, body_start, tag_kind)?;
        let trim_right = close_idx > body_start && bytes[close_idx - 1] == b'-';
        let body_end = close_idx - usize::from(trim_right);
        let body = normalized[body_start..body_end].to_owned();
        pos = close_idx + 2;

        if tag_kind != b'#' {
            segments.push(if tag_kind == b'{' {
                BlockSegment::Expr(body)
            } else {
                BlockSegment::Tag(body)
            });
        }
        if trim_right {
            strip_next = true;
        } else if tag_kind != b'{' && bytes.get(pos) == Some(&b'\n') {
            pos += 1;
        }
    }
    Ok(segments)
}

fn find_closing_delimiter(bytes: &[u8], start: usize, tag_kind: u8) -> JResult<usize> {
    let close_first = if tag_kind == b'{' { b'}' } else { tag_kind };
    let mut cursor = start;
    let mut active_quote: Option<u8> = None;
    let mut brace_depth = 0_usize;
    while cursor + 1 < bytes.len() {
        let ch = bytes[cursor];
        if tag_kind == b'#' {
            if ch == b'#' && bytes[cursor + 1] == b'}' {
                return Ok(cursor);
            }
        } else if let Some(quote) = active_quote {
            if ch == b'\\' {
                cursor += 2;
                continue;
            }
            if ch == quote {
                active_quote = None;
            }
        } else if matches!(ch, b'\'' | b'"') {
            active_quote = Some(ch);
        } else if brace_depth == 0 && ch == close_first && bytes[cursor + 1] == b'}' {
            return Ok(cursor);
        } else if ch == b'{' {
            brace_depth += 1;
        } else if ch == b'}' && brace_depth > 0 {
            brace_depth -= 1;
        }
        cursor += 1;
    }
    Err(JinjaError::std("unterminated Jinja delimiter block"))
}

struct StmtParser {
    segments: Vec<BlockSegment>,
    cursor: usize,
    depth: usize,
}

impl StmtParser {
    fn current_tag_trimmed(&self) -> Option<&str> {
        if let Some(BlockSegment::Tag(text)) = self.segments.get(self.cursor) {
            Some(text.trim())
        } else {
            None
        }
    }

    fn at_stop(&self, names: &[&str]) -> bool {
        self.current_tag_trimmed().is_some_and(|tag| {
            let keyword = tag.split_ascii_whitespace().next().unwrap_or("");
            names.contains(&keyword)
        })
    }

    fn consume_tag(&mut self, expected: &str) -> JResult<()> {
        if !self.at_stop(&[expected]) {
            return Err(JinjaError::std(format!(
                "expected closing Jinja tag '{expected}'"
            )));
        }
        self.cursor += 1;
        Ok(())
    }

    fn parse_block(&mut self, stop_tags: &[&str]) -> JResult<Vec<Stmt>> {
        self.depth += 1;
        if self.depth > MAX_PARSE_DEPTH {
            self.depth -= 1;
            return Err(JinjaError::std("Jinja statement nesting depth exceeded"));
        }
        let parsed = self.parse_block_inner(stop_tags);
        self.depth -= 1;
        parsed
    }

    fn parse_block_inner(&mut self, stop_tags: &[&str]) -> JResult<Vec<Stmt>> {
        let mut stmts = Vec::new();
        while self.cursor < self.segments.len() && !self.at_stop(stop_tags) {
            let idx = self.cursor;
            self.cursor += 1;
            match &self.segments[idx] {
                BlockSegment::Text(bytes) => stmts.push(Stmt::Text(bytes.clone())),
                BlockSegment::Expr(expr_text) => {
                    let expr = ExprParser::new(expr_text)?.parse_complete()?;
                    stmts.push(Stmt::Output(expr));
                }
                BlockSegment::Tag(raw_tag) => {
                    let trimmed = raw_tag.trim().to_owned();
                    let split_at = trimmed
                        .find(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_')
                        .unwrap_or(trimmed.len());
                    let (keyword, rest) = trimmed.split_at(split_at);
                    let stmt = self.parse_tag_stmt(keyword, rest.trim_start())?;
                    if let Some(node) = stmt {
                        stmts.push(node);
                    }
                }
            }
            if stmts.len() > MAX_STMTS_PER_BLOCK {
                return Err(JinjaError::std("Jinja statement count bound exceeded"));
            }
        }
        Ok(stmts)
    }

    fn parse_tag_stmt(&mut self, keyword: &str, rest: &str) -> JResult<Option<Stmt>> {
        match keyword {
            "if" => self.parse_if(rest).map(Some),
            "for" => self.parse_for(rest).map(Some),
            "set" => self.parse_set(rest).map(Some),
            "macro" => self.parse_macro(rest).map(Some),
            "call" => self.parse_call(rest).map(Some),
            "filter" => self.parse_filter_stmt(rest).map(Some),
            "break" => Ok(Some(Stmt::Break)),
            "continue" => Ok(Some(Stmt::Continue)),
            "generation" | "endgeneration" => Ok(None),
            _ => Err(JinjaError::std(format!(
                "unknown Jinja statement '{keyword}'"
            ))),
        }
    }

    fn parse_if(&mut self, rest: &str) -> JResult<Stmt> {
        let mut branches = Vec::new();
        let mut cond = ExprParser::new(rest)?.parse_complete()?;
        loop {
            let body = self.parse_block(&["elif", "else", "endif"])?;
            branches.push((cond, body));
            if self.at_stop(&["elif"]) {
                let tag_text = self.current_tag_trimmed().unwrap_or("").to_owned();
                self.cursor += 1;
                let elif_rest = tag_text.strip_prefix("elif").unwrap_or("").trim_start();
                cond = ExprParser::new(elif_rest)?.parse_complete()?;
            } else {
                break;
            }
        }
        let alternate = if self.at_stop(&["else"]) {
            self.cursor += 1;
            self.parse_block(&["endif"])?
        } else {
            Vec::new()
        };
        self.consume_tag("endif")?;
        Ok(Stmt::If(branches, alternate))
    }

    fn parse_for(&mut self, rest: &str) -> JResult<Stmt> {
        let mut parser = ExprParser::new(rest)?;
        let mut targets = vec![Expr::Var(parser.identifier()?)];
        while parser.consume_word(",") {
            targets.push(Expr::Var(parser.identifier()?));
        }
        parser.expect_word("in")?;
        let iterable = parser.parse_expr()?;
        parser.ensure_eof()?;
        let target = if targets.len() == 1 {
            targets.remove(0)
        } else {
            Expr::Tuple(targets)
        };
        let body = self.parse_block(&["else", "endfor"])?;
        let alternate = if self.at_stop(&["else"]) {
            self.cursor += 1;
            self.parse_block(&["endfor"])?
        } else {
            Vec::new()
        };
        self.consume_tag("endfor")?;
        Ok(Stmt::For(target, iterable, body, alternate))
    }

    fn parse_set(&mut self, rest: &str) -> JResult<Stmt> {
        let mut parser = ExprParser::new(rest)?;
        let target = parser.parse_sequence()?;
        if parser.consume_word("=") {
            let val_expr = parser.parse_sequence()?;
            parser.ensure_eof()?;
            Ok(Stmt::Set(target, Some(val_expr), Vec::new()))
        } else {
            parser.ensure_eof()?;
            let body = self.parse_block(&["endset"])?;
            self.consume_tag("endset")?;
            Ok(Stmt::Set(target, None, body))
        }
    }

    fn parse_macro(&mut self, rest: &str) -> JResult<Stmt> {
        let mut parser = ExprParser::new(rest)?;
        let name = parser.identifier()?;
        let mut params = Vec::new();
        for arg in parser.parse_args()? {
            match arg {
                CallArg::Pos(Expr::Var(param_name)) => params.push((param_name, None)),
                CallArg::Kw(param_name, default_expr) => {
                    params.push((param_name, Some(default_expr)));
                }
                _ => return Err(JinjaError::std("invalid Jinja macro parameter")),
            }
        }
        parser.ensure_eof()?;
        let body = self.parse_block(&["endmacro"])?;
        self.consume_tag("endmacro")?;
        Ok(Stmt::Macro(name, params, body))
    }

    fn parse_call(&mut self, rest: &str) -> JResult<Stmt> {
        let mut parser = ExprParser::new(rest)?;
        let params = if parser.peek_word("(") {
            parser
                .parse_args()?
                .into_iter()
                .map(|arg| match arg {
                    CallArg::Pos(Expr::Var(name)) => Ok(name),
                    _ => Err(JinjaError::std("invalid Jinja caller parameter")),
                })
                .collect::<JResult<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let call_expr = parser.parse_expr()?;
        parser.ensure_eof()?;
        let body = self.parse_block(&["endcall"])?;
        self.consume_tag("endcall")?;
        Ok(Stmt::Call(params, call_expr, body))
    }

    fn parse_filter_stmt(&mut self, rest: &str) -> JResult<Stmt> {
        let filter_expr = ExprParser::new(rest)?.parse_complete()?;
        let body = self.parse_block(&["endfilter"])?;
        self.consume_tag("endfilter")?;
        Ok(Stmt::Filter(filter_expr, body))
    }
}

/// Parsed Jinja template program.
#[derive(Clone, Debug)]
pub(super) struct Program(Vec<Stmt>);

pub(super) fn parse_template_with_limit(source: &str, max_bytes: usize) -> JResult<Program> {
    let segments = scan_template_segments(source, max_bytes)?;
    let mut parser = StmtParser {
        segments,
        cursor: 0,
        depth: 0,
    };
    let program = parser.parse_block(&[])?;
    if parser.cursor != parser.segments.len() {
        return Err(JinjaError::std("unexpected Jinja closing tag"));
    }
    Ok(Program(program))
}

pub(super) fn parse_template(source: &str) -> JResult<Program> {
    parse_template_with_limit(source, MAX_TEMPLATE_BYTES)
}

// ---------------------------------------------------------------------------
// Evaluator & Builtins
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum LoopFlow {
    Normal,
    Break,
    Continue,
}

fn builtin_from_json(
    target: &JVal,
    pos_args: &[JVal],
    kw_args: &BTreeMap<String, JVal>,
) -> JResult<JVal> {
    if pos_args.len() != 1 || !kw_args.is_empty() {
        return Err(JinjaError::std("Jinja from_json takes no extra arguments"));
    }
    if !target.is_string() {
        return Err(JinjaError::std("Jinja from_json requires a JSON string"));
    }
    let raw = target.raw_bytes();
    let text = std::str::from_utf8(&raw)
        .map_err(|_| JinjaError::std("Jinja from_json input must be valid UTF-8"))?;
    parse_json_jval_bounded(text, MAX_TEXT_BYTES, 64)
        .map_err(|err| JinjaError::std(format!("Jinja from_json error: {err}")))
}

fn builtin_strftime_now(
    now_unix_seconds: Option<i64>,
    pos_args: &[JVal],
    kw_args: &BTreeMap<String, JVal>,
) -> JResult<JVal> {
    let Some(unix_secs) = now_unix_seconds else {
        return Err(JinjaError::unimpl(
            "Jinja builtin 'strftime_now' is not implemented",
        ));
    };
    if pos_args.len() > 1 {
        return Err(JinjaError::std("strftime_now accepts one format argument"));
    }
    let fmt_val = if let Some(val) = kw_args.get("format") {
        val.clone()
    } else if let Some(val) = pos_args.first() {
        val.clone()
    } else {
        return Err(JinjaError::std("strftime_now requires a format string"));
    };
    let fmt_bytes = fmt_val.raw_bytes();
    let fmt_str = std::str::from_utf8(&fmt_bytes)
        .map_err(|_| JinjaError::std("strftime_now format must be valid UTF-8"))?;
    let formatted = format_strftime_utc(unix_secs, fmt_str)?;
    Ok(JVal::bytes(formatted.into_bytes(), false))
}

struct Evaluator {
    scopes: Vec<BTreeMap<String, JVal>>,
    steps: usize,
    stmt_depth: usize,
    expr_depth: usize,
    output_bytes: usize,
    now_unix_seconds: Option<i64>,
    allow_undef_str_add: bool,
}

impl Evaluator {
    fn new_with_now(globals: &JVal, now_unix_seconds: Option<i64>) -> Self {
        let root_scope = if let ValKind::Object(pairs) = globals.kind_cloned() {
            pairs
                .into_iter()
                .map(|(k, v)| (String::from_utf8_lossy(&k.raw_bytes()).into_owned(), v))
                .collect()
        } else {
            BTreeMap::new()
        };
        Self {
            scopes: vec![root_scope],
            steps: 0,
            stmt_depth: 0,
            expr_depth: 0,
            output_bytes: 0,
            now_unix_seconds,
            allow_undef_str_add: false,
        }
    }

    fn tick(&mut self) -> JResult<()> {
        self.steps += 1;
        if self.steps > MAX_EXEC_STEPS {
            return Err(JinjaError::std("Jinja execution step limit exceeded"));
        }
        Ok(())
    }

    fn lookup_var(&self, name: &str) -> JVal {
        for scope in self.scopes.iter().rev() {
            if let Some(val) = scope.get(name) {
                return val.clone();
            }
        }
        match name {
            "true" | "True" => JVal::boolean(true),
            "false" | "False" => JVal::boolean(false),
            "none" | "None" => JVal::none(),
            _ if is_global_builtin(name) => JVal::new(ValKind::Function(name.to_owned())),
            _ => JVal::undefined(),
        }
    }

    fn bind_var(&mut self, name: String, value: JVal) {
        if let Some(top) = self.scopes.last_mut() {
            top.insert(name, value);
        }
    }

    fn emit_spans(&mut self, out: &mut Vec<TextSpan>, spans: Vec<TextSpan>) -> JResult<()> {
        let added: usize = spans.iter().map(|span| span.bytes.len()).sum();
        self.output_bytes = self
            .output_bytes
            .checked_add(added)
            .ok_or_else(|| JinjaError::std("Jinja output size overflow"))?;
        if self.output_bytes > MAX_OUTPUT_BYTES {
            return Err(JinjaError::std("Jinja output byte bound exceeded"));
        }
        out.extend(spans);
        Ok(())
    }

    fn assign_target(&mut self, target: &Expr, value: JVal) -> JResult<()> {
        match target {
            Expr::Var(name) => self.bind_var(name.clone(), value),
            Expr::Tuple(targets) => {
                let elems = value.iter_elements()?;
                if targets.len() != elems.len() {
                    return Err(JinjaError::std("Jinja tuple unpacking length mismatch"));
                }
                for (sub_target, sub_val) in targets.iter().zip(elems) {
                    self.assign_target(sub_target, sub_val)?;
                }
            }
            Expr::Attr(container_expr, attr_name) => {
                let container = self.eval(container_expr)?;
                container.set_member(JVal::text_str(attr_name.clone()), value)?;
            }
            Expr::Member(container_expr, key_expr) => {
                let container = self.eval(container_expr)?;
                let key = self.eval(key_expr)?;
                container.set_member(key, value)?;
            }
            _ => return Err(JinjaError::std("invalid Jinja assignment target")),
        }
        Ok(())
    }

    fn exec_block(&mut self, stmts: &[Stmt], out: &mut Vec<TextSpan>) -> JResult<LoopFlow> {
        self.stmt_depth += 1;
        if self.stmt_depth > MAX_EXEC_DEPTH {
            self.stmt_depth -= 1;
            return Err(JinjaError::std("Jinja execution depth exceeded"));
        }
        let flow = self.exec_block_inner(stmts, out);
        self.stmt_depth -= 1;
        flow
    }

    fn exec_block_inner(&mut self, stmts: &[Stmt], out: &mut Vec<TextSpan>) -> JResult<LoopFlow> {
        for stmt in stmts {
            self.tick()?;
            match stmt {
                Stmt::Text(bytes) => self.emit_spans(
                    out,
                    vec![TextSpan {
                        bytes: bytes.clone(),
                        is_input: false,
                    }],
                )?,
                Stmt::Output(expr) => {
                    let val = self.eval(expr)?;
                    self.emit_spans(out, val.to_spans())?;
                }
                Stmt::Set(target, expr_opt, body) => {
                    let assigned = if let Some(expr) = expr_opt {
                        self.eval(expr)?
                    } else {
                        let mut captured = Vec::new();
                        self.exec_block(body, &mut captured)?;
                        JVal::new(ValKind::String(captured))
                    };
                    self.assign_target(target, assigned)?;
                }
                Stmt::If(branches, alternate) => {
                    let mut chosen = alternate;
                    for (cond, body) in branches {
                        if self.eval(cond)?.truthy() {
                            chosen = body;
                            break;
                        }
                    }
                    let flow = self.exec_block(chosen, out)?;
                    if flow != LoopFlow::Normal {
                        return Ok(flow);
                    }
                }
                Stmt::For(target, iterable, body, alternate) => {
                    self.exec_for(target, iterable, body, alternate, out)?;
                }
                Stmt::Macro(name, params, body) => {
                    self.bind_var(
                        name.clone(),
                        JVal::new(ValKind::Macro(params.clone(), body.clone())),
                    );
                }
                Stmt::Call(params, call_expr, body) => {
                    self.exec_call_stmt(params, call_expr, body, out)?;
                }
                Stmt::Filter(filter_expr, body) => {
                    let mut captured = Vec::new();
                    self.exec_block(body, &mut captured)?;
                    let input_str = JVal::new(ValKind::String(captured));
                    let filtered = self.eval_block_filter(filter_expr, input_str)?;
                    self.emit_spans(out, filtered.to_spans())?;
                }
                Stmt::Break => return Ok(LoopFlow::Break),
                Stmt::Continue => return Ok(LoopFlow::Continue),
            }
        }
        Ok(LoopFlow::Normal)
    }

    fn exec_for(
        &mut self,
        target: &Expr,
        iterable: &Expr,
        body: &[Stmt],
        alternate: &[Stmt],
        out: &mut Vec<TextSpan>,
    ) -> JResult<()> {
        let (source_expr, filter_cond) = if let Expr::Select(inner, cond) = iterable {
            (&**inner, Some(&**cond))
        } else {
            (iterable, None)
        };
        let raw_items = self.eval(source_expr)?.iter_elements()?;
        if raw_items.len() > MAX_CONTAINER_LEN {
            return Err(JinjaError::std("Jinja loop iteration bound exceeded"));
        }
        let mut selected_items = Vec::new();
        self.scopes.push(BTreeMap::new());
        let prep_res: JResult<()> = (|| {
            for item in raw_items {
                self.assign_target(target, item.clone())?;
                if let Some(cond) = filter_cond {
                    if self.eval(cond)?.truthy() {
                        selected_items.push(item);
                    }
                } else {
                    selected_items.push(item);
                }
            }
            Ok(())
        })();
        if let Err(err) = prep_res {
            self.scopes.pop();
            return Err(err);
        }

        let exec_res: JResult<()> = (|| {
            if selected_items.is_empty() {
                self.exec_block(alternate, out)?;
            } else {
                let total = selected_items.len();
                for (idx, item) in selected_items.iter().enumerate() {
                    let previtem = idx
                        .checked_sub(1)
                        .and_then(|prev| selected_items.get(prev))
                        .cloned()
                        .unwrap_or_else(JVal::undefined);
                    let nextitem = selected_items
                        .get(idx + 1)
                        .cloned()
                        .unwrap_or_else(JVal::undefined);
                    self.assign_target(target, item.clone())?;
                    let idx_i64 = i64::try_from(idx)
                        .map_err(|_| JinjaError::std("Jinja loop index overflow"))?;
                    let total_i64 = i64::try_from(total)
                        .map_err(|_| JinjaError::std("Jinja loop length overflow"))?;
                    let rev_idx = total - idx;
                    let rev_idx_i64 = i64::try_from(rev_idx)
                        .map_err(|_| JinjaError::std("Jinja loop revindex overflow"))?;
                    let loop_fields = vec![
                        (JVal::text_str("index"), JVal::int(idx_i64 + 1)),
                        (JVal::text_str("index0"), JVal::int(idx_i64)),
                        (JVal::text_str("revindex"), JVal::int(rev_idx_i64)),
                        (JVal::text_str("revindex0"), JVal::int(rev_idx_i64 - 1)),
                        (JVal::text_str("first"), JVal::boolean(idx == 0)),
                        (JVal::text_str("last"), JVal::boolean(idx + 1 == total)),
                        (JVal::text_str("length"), JVal::int(total_i64)),
                        (JVal::text_str("previtem"), previtem),
                        (JVal::text_str("nextitem"), nextitem),
                    ];
                    let loop_obj = JVal::new(ValKind::Object(loop_fields));
                    loop_obj.set_namespace();
                    self.bind_var("loop".to_owned(), loop_obj);
                    if self.exec_block(body, out)? == LoopFlow::Break {
                        break;
                    }
                }
            }
            Ok(())
        })();
        self.scopes.pop();
        exec_res
    }

    fn exec_call_stmt(
        &mut self,
        params: &[String],
        call_expr: &Expr,
        body: &[Stmt],
        out: &mut Vec<TextSpan>,
    ) -> JResult<()> {
        let captured = self.scopes.clone();
        let caller_val = JVal::new(ValKind::Caller(params.to_vec(), body.to_vec(), captured));
        let previous = self
            .scopes
            .last_mut()
            .and_then(|scope| scope.insert("caller".to_owned(), caller_val));
        let call_result = self.eval(call_expr);
        if let Some(scope) = self.scopes.last_mut() {
            if let Some(prev) = previous {
                scope.insert("caller".to_owned(), prev);
            } else {
                scope.remove("caller");
            }
        }
        let val = call_result?;
        self.emit_spans(out, val.to_spans())
    }

    fn eval_block_filter(&mut self, filter_expr: &Expr, value: JVal) -> JResult<JVal> {
        match filter_expr {
            Expr::Var(name) => self.invoke_builtin(name, &[value], &BTreeMap::new()),
            Expr::Call(callee, args) => {
                let Expr::Var(name) = &**callee else {
                    return Err(JinjaError::std("invalid Jinja filter block expression"));
                };
                let (mut pos_args, kw_args) = self.eval_args(args)?;
                pos_args.insert(0, value);
                self.invoke_builtin(name, &pos_args, &kw_args)
            }
            Expr::Filter(inner, name, args) => {
                let piped = self.eval_block_filter(inner, value)?;
                let (mut pos_args, kw_args) = self.eval_args(args)?;
                pos_args.insert(0, piped);
                self.invoke_builtin(name, &pos_args, &kw_args)
            }
            _ => Err(JinjaError::std("invalid Jinja filter block expression")),
        }
    }

    fn eval_args(&mut self, args: &[CallArg]) -> JResult<(Vec<JVal>, BTreeMap<String, JVal>)> {
        let mut pos_args = Vec::new();
        let mut kw_args = BTreeMap::new();
        for arg in args {
            match arg {
                CallArg::Pos(expr) => pos_args.push(self.eval(expr)?),
                CallArg::Kw(name, expr) => {
                    let evaluated = self.eval(expr)?;
                    if kw_args.insert(name.clone(), evaluated).is_some() {
                        return Err(JinjaError::std("duplicate Jinja keyword argument"));
                    }
                }
                CallArg::Spread(expr) => {
                    pos_args.extend(self.eval(expr)?.iter_elements()?);
                }
            }
        }
        Ok((pos_args, kw_args))
    }

    fn invoke_callable(
        &mut self,
        callee: &JVal,
        pos_args: &[JVal],
        kw_args: &BTreeMap<String, JVal>,
    ) -> JResult<JVal> {
        match callee.kind_cloned() {
            ValKind::Function(name) => self.invoke_builtin(&name, pos_args, kw_args),
            ValKind::Method(receiver, name) => {
                let mut all_pos = Vec::with_capacity(pos_args.len() + 1);
                all_pos.push(receiver);
                all_pos.extend(pos_args.iter().cloned());
                self.invoke_builtin(&name, &all_pos, kw_args)
            }
            ValKind::Macro(params, body) => {
                if pos_args.len() > params.len() {
                    return Err(JinjaError::std(
                        "too many positional arguments for Jinja macro",
                    ));
                }
                if kw_args
                    .keys()
                    .any(|key| !params.iter().any(|(param_name, _)| param_name == key))
                {
                    return Err(JinjaError::std(
                        "unexpected keyword argument for Jinja macro",
                    ));
                }
                let mut frame = BTreeMap::new();
                for (idx, (param_name, default_opt)) in params.iter().enumerate() {
                    let bound = if let Some(val) = pos_args.get(idx) {
                        if kw_args.contains_key(param_name) {
                            return Err(JinjaError::std(
                                "multiple values for Jinja macro parameter",
                            ));
                        }
                        val.clone()
                    } else if let Some(val) = kw_args.get(param_name) {
                        val.clone()
                    } else if let Some(default_expr) = default_opt {
                        self.eval(default_expr)?
                    } else {
                        JVal::undefined()
                    };
                    frame.insert(param_name.clone(), bound);
                }
                self.scopes.push(frame);
                let mut out = Vec::new();
                let res = self.exec_block(&body, &mut out);
                self.scopes.pop();
                res?;
                Ok(JVal::new(ValKind::String(out)))
            }
            ValKind::Caller(params, body, captured) => {
                if !kw_args.is_empty() {
                    return Err(JinjaError::std("Jinja caller does not accept keyword args"));
                }
                let mut frame = BTreeMap::new();
                for (idx, param_name) in params.iter().enumerate() {
                    frame.insert(
                        param_name.clone(),
                        pos_args.get(idx).cloned().unwrap_or_else(JVal::undefined),
                    );
                }
                let saved = std::mem::replace(&mut self.scopes, captured);
                self.scopes.push(frame);
                let mut out = Vec::new();
                let res = self.exec_block(&body, &mut out);
                self.scopes = saved;
                res?;
                Ok(JVal::new(ValKind::String(out)))
            }
            _ => Err(JinjaError::std("Jinja value is not callable")),
        }
    }

    fn eval(&mut self, expr: &Expr) -> JResult<JVal> {
        self.expr_depth += 1;
        if self.expr_depth > MAX_EXEC_DEPTH {
            self.expr_depth -= 1;
            return Err(JinjaError::std(
                "Jinja expression evaluation depth exceeded",
            ));
        }
        let outcome = self.eval_inner(expr);
        self.expr_depth -= 1;
        let val = outcome?;
        val.check_graph_bounds()?;
        Ok(val)
    }

    fn eval_inner(&mut self, expr: &Expr) -> JResult<JVal> {
        self.tick()?;
        match expr {
            Expr::Lit(Literal::Int(num)) => Ok(JVal::int(*num)),
            Expr::Lit(Literal::Float(num)) => Ok(JVal::float(*num)),
            Expr::Lit(Literal::Bytes(bytes)) => Ok(JVal::bytes(bytes.clone(), false)),
            Expr::Var(name) => Ok(self.lookup_var(name)),
            Expr::Blank => Ok(JVal::undefined()),
            Expr::Array(_) | Expr::Tuple(_) | Expr::Object(_) => self.eval_container(expr),
            Expr::Attr(base_expr, attr_name) => {
                let base = self.eval(base_expr)?;
                if has_value_method(&base, attr_name) {
                    return Ok(JVal::new(ValKind::Method(base, attr_name.clone())));
                }
                Ok(base.get_member(&JVal::text_str(attr_name.clone()), true))
            }
            Expr::Member(base_expr, key_expr) => {
                let base = self.eval(base_expr)?;
                if let Expr::Slice(start, stop, delta) = &**key_expr {
                    return self.eval_slice(
                        &base,
                        start.as_deref(),
                        stop.as_deref(),
                        delta.as_deref(),
                    );
                }
                let key = self.eval(key_expr)?;
                Ok(base.get_member(&key, true))
            }
            Expr::Call(callee_expr, args) => {
                let callee = self.eval(callee_expr)?;
                let (pos_args, kw_args) = self.eval_args(args)?;
                self.invoke_callable(&callee, &pos_args, &kw_args)
            }
            Expr::Filter(val_expr, filter_name, args) => {
                let target = self.eval(val_expr)?;
                let (mut pos_args, kw_args) = self.eval_args(args)?;
                pos_args.insert(0, target);
                self.invoke_builtin(filter_name, &pos_args, &kw_args)
            }
            Expr::Test(val_expr, test_name, args, negated) => {
                let target = self.eval(val_expr)?;
                let (pos_args, kw_args) = self.eval_args(args)?;
                if !kw_args.is_empty() {
                    return Err(JinjaError::std("Jinja tests do not accept keyword args"));
                }
                let matched = self.eval_test(&target, test_name, &pos_args)?;
                Ok(JVal::boolean(matched != *negated))
            }
            Expr::Unary(op, inner) => self.eval_unary(op, inner),
            Expr::Binary(op, lhs_expr, rhs_expr) => self.eval_binary_expr(op, lhs_expr, rhs_expr),
            Expr::Ternary(cond, yes, no) => {
                if self.eval(cond)?.truthy() {
                    self.eval(yes)
                } else {
                    self.eval(no)
                }
            }
            Expr::Select(yes, cond) => {
                if self.eval(cond)?.truthy() {
                    self.eval(yes)
                } else {
                    Ok(JVal::undefined())
                }
            }
            Expr::Slice(..) => Err(JinjaError::std("Jinja slice used outside indexing")),
        }
    }

    fn eval_container(&mut self, expr: &Expr) -> JResult<JVal> {
        match expr {
            Expr::Array(items) | Expr::Tuple(items) => {
                let evaluated = items
                    .iter()
                    .map(|item| self.eval(item))
                    .collect::<JResult<Vec<_>>>()?;
                Ok(JVal::new(if matches!(expr, Expr::Array(_)) {
                    ValKind::Array(evaluated)
                } else {
                    ValKind::Tuple(evaluated)
                }))
            }
            Expr::Object(entries) => {
                let obj = JVal::new(ValKind::Object(Vec::new()));
                for (key_expr, val_expr) in entries {
                    let entry_key = self.eval(key_expr)?;
                    let entry_val = self.eval(val_expr)?;
                    obj.set_member(entry_key, entry_val)?;
                }
                Ok(obj)
            }
            _ => Err(JinjaError::std("invalid Jinja container expression")),
        }
    }

    fn eval_binary_expr(&mut self, op: &str, lhs_expr: &Expr, rhs_expr: &Expr) -> JResult<JVal> {
        let lhs = self.eval(lhs_expr)?;
        if op == "or" {
            return if lhs.truthy() {
                Ok(lhs)
            } else {
                self.eval(rhs_expr)
            };
        }
        if op == "and" {
            return if lhs.truthy() {
                self.eval(rhs_expr)
            } else {
                Ok(lhs)
            };
        }
        let rhs = self.eval(rhs_expr)?;
        self.eval_binary(op, lhs, rhs)
    }

    fn eval_unary(&mut self, op: &str, inner: &Expr) -> JResult<JVal> {
        let val = self.eval(inner)?;
        match op {
            "not" => Ok(JVal::boolean(!val.truthy())),
            "+" => {
                val.as_f64()?;
                Ok(val)
            }
            "-" => {
                if let ValKind::Int(num) = val.kind_cloned() {
                    Ok(JVal::int(num.checked_neg().ok_or_else(|| {
                        JinjaError::std("Jinja integer negation overflow")
                    })?))
                } else {
                    Ok(JVal::float(-val.as_f64()?))
                }
            }
            _ => Err(JinjaError::std("unknown Jinja unary operator")),
        }
    }

    fn eval_binary(&mut self, op: &str, lhs: JVal, rhs: JVal) -> JResult<JVal> {
        self.tick()?;
        if matches!(op, "+" | "-" | "*" | "//" | "%")
            && matches!(lhs.kind_cloned(), ValKind::Int(_) | ValKind::Bool(_))
            && matches!(rhs.kind_cloned(), ValKind::Int(_) | ValKind::Bool(_))
        {
            return eval_int_binary(op, &lhs, &rhs);
        }

        match op {
            "==" => Ok(JVal::boolean(vals_equal(&lhs, &rhs))),
            "!=" => Ok(JVal::boolean(!vals_equal(&lhs, &rhs))),
            "in" | "not in" => Ok(JVal::boolean(val_contains(&rhs, &lhs)? == (op == "in"))),
            "<" | ">" | "<=" | ">=" => {
                let ord = vals_compare(&lhs, &rhs)?;
                Ok(JVal::boolean(match op {
                    "<" => ord.is_lt(),
                    ">" => ord.is_gt(),
                    "<=" => !ord.is_gt(),
                    _ => !ord.is_lt(),
                }))
            }
            "~" => {
                let mut spans = lhs.to_spans();
                spans.extend(rhs.to_spans());
                Ok(JVal::new(ValKind::String(spans)))
            }
            "+" if lhs.is_string() && rhs.is_string() => {
                let mut spans = lhs.to_spans();
                spans.extend(rhs.to_spans());
                Ok(JVal::new(ValKind::String(spans)))
            }
            "+" if self.allow_undef_str_add
                && ((lhs.is_string() && rhs.is_undefined())
                    || (lhs.is_undefined() && rhs.is_string())) =>
            {
                let spans = if lhs.is_string() {
                    lhs.to_spans()
                } else {
                    rhs.to_spans()
                };
                Ok(JVal::new(ValKind::String(spans)))
            }
            "+" if matches!(lhs.kind_cloned(), ValKind::Array(_) | ValKind::Tuple(_)) => {
                let is_tuple = matches!(lhs.kind_cloned(), ValKind::Tuple(_));
                let mut combined = lhs.iter_elements()?;
                combined.extend(rhs.iter_elements()?);
                if combined.len() > MAX_CONTAINER_LEN {
                    return Err(JinjaError::std(
                        "Jinja sequence concatenation bound exceeded",
                    ));
                }
                Ok(JVal::new(if is_tuple {
                    ValKind::Tuple(combined)
                } else {
                    ValKind::Array(combined)
                }))
            }
            "*" if is_seq_repeat_pair(&lhs, &rhs) => eval_seq_repeat(lhs, rhs),
            "+" | "-" | "*" | "/" | "//" | "%" => eval_float_binary(op, &lhs, &rhs),
            _ => Err(JinjaError::std("unknown Jinja binary operator")),
        }
    }

    fn eval_slice(
        &mut self,
        container: &JVal,
        start: Option<&Expr>,
        stop: Option<&Expr>,
        delta_expr: Option<&Expr>,
    ) -> JResult<JVal> {
        let elems = container.iter_elements()?;
        let len = i64::try_from(elems.len())
            .map_err(|_| JinjaError::std("Jinja slice container length overflow"))?;
        let delta = if let Some(expr) = delta_expr {
            self.eval(expr)?.as_i64()?
        } else {
            1
        };
        if delta == 0 {
            return Err(JinjaError::std("Jinja slice step cannot be zero"));
        }
        let lower = if delta < 0 { -1 } else { 0 };
        let upper = if delta < 0 { len - 1 } else { len };
        let clamp_bound = |raw: i64| -> i64 {
            let shifted = if raw < 0 {
                len.saturating_add(raw)
            } else {
                raw
            };
            shifted.clamp(lower, upper)
        };
        let mut cursor = if let Some(start_expr) = start {
            clamp_bound(self.eval(start_expr)?.as_i64()?)
        } else if delta < 0 {
            len - 1
        } else {
            0
        };
        let end = if let Some(stop_expr) = stop {
            clamp_bound(self.eval(stop_expr)?.as_i64()?)
        } else if delta < 0 {
            -1
        } else {
            len
        };
        let mut sliced = Vec::new();
        while if delta < 0 {
            cursor > end
        } else {
            cursor < end
        } {
            if let Some(elem) = usize::try_from(cursor).ok().and_then(|pos| elems.get(pos)) {
                sliced.push(elem.clone());
            }
            let Some(next) = cursor.checked_add(delta) else {
                break;
            };
            cursor = next;
        }
        Ok(match container.kind_cloned() {
            ValKind::String(_) => JVal::new(ValKind::String(
                sliced.into_iter().flat_map(|v| v.to_spans()).collect(),
            )),
            ValKind::Tuple(_) => JVal::new(ValKind::Tuple(sliced)),
            _ => JVal::new(ValKind::Array(sliced)),
        })
    }

    fn invoke_builtin(
        &mut self,
        name: &str,
        pos_args: &[JVal],
        kw_args: &BTreeMap<String, JVal>,
    ) -> JResult<JVal> {
        self.tick()?;
        if let Some(test_name) = name.strip_prefix("test_is_") {
            let target = pos_args.first().cloned().unwrap_or_else(JVal::undefined);
            let extra = pos_args.get(1..).unwrap_or_default();
            return Ok(JVal::boolean(self.eval_test(&target, test_name, extra)?));
        }
        if name == "strftime_now" {
            return builtin_strftime_now(self.now_unix_seconds, pos_args, kw_args);
        }
        if name == "slice" {
            return Err(JinjaError::unimpl(format!(
                "Jinja builtin '{name}' is not implemented"
            )));
        }
        let target = pos_args.first().cloned().unwrap_or_else(JVal::undefined);
        match name {
            "raise_exception" => Err(JinjaError::std(format!(
                "Jinja Exception: {}",
                preview_utf8(&target.raw_bytes())
            ))),
            "namespace" => {
                if !pos_args.is_empty() {
                    return Err(JinjaError::std("namespace accepts only keyword arguments"));
                }
                let pairs = kw_args
                    .iter()
                    .map(|(k, v)| (JVal::text_str(k.clone()), v.clone()))
                    .collect();
                let ns = JVal::new(ValKind::Object(pairs));
                ns.set_namespace();
                Ok(ns)
            }
            "range" => builtin_range(pos_args, kw_args),
            "default" | "d" => {
                let fallback = pick_arg(pos_args, kw_args, "default_value", 1, JVal::text_str(""));
                let check_falsy =
                    pick_arg(pos_args, kw_args, "boolean", 2, JVal::boolean(false)).truthy();
                Ok(
                    if target.is_undefined() || (check_falsy && !target.truthy()) {
                        fallback
                    } else {
                        target
                    },
                )
            }
            "length" | "count" => {
                let len = i64::try_from(target.length()?)
                    .map_err(|_| JinjaError::std("Jinja length overflow"))?;
                Ok(JVal::int(len))
            }
            "string" | "safe" => Ok(if target.is_string() {
                target
            } else {
                JVal::bytes(target.raw_bytes(), false)
            }),
            "int" => builtin_int(&target, pos_args, kw_args),
            "float" => builtin_float(&target, pos_args, kw_args),
            "abs" => builtin_abs(&target),
            "tojson" => {
                let opts = JsonDumpOpts::from_args(pos_args, kw_args)?;
                let encoded = encode_json_bytes(&target, &opts, 0)?;
                Ok(JVal::bytes(encoded, false))
            }
            "from_json" => builtin_from_json(&target, pos_args, kw_args),
            "upper" | "lower" | "capitalize" | "title" | "strip" | "trim" | "lstrip" | "rstrip"
            | "startswith" | "endswith" | "replace" | "split" | "rsplit" | "format" | "indent"
            | "wordcount" | "truncate" => invoke_string_builtin(&target, name, pos_args, kw_args),
            "list" | "first" | "last" | "reverse" | "join" | "append" | "pop" | "sort" | "min"
            | "max" | "map" | "select" | "reject" | "selectattr" | "rejectattr" | "unique"
            | "sum" | "get" | "items" | "keys" | "values" | "dictsort" => {
                self.invoke_collection_builtin(&target, name, pos_args, kw_args)
            }
            _ => Err(JinjaError::std(format!(
                "unsupported Jinja builtin '{name}'"
            ))),
        }
    }

    fn invoke_collection_builtin(
        &mut self,
        target: &JVal,
        name: &str,
        pos_args: &[JVal],
        kw_args: &BTreeMap<String, JVal>,
    ) -> JResult<JVal> {
        match name {
            "list" => Ok(JVal::new(ValKind::Array(target.iter_elements()?))),
            "first" | "last" => {
                let elems = target.iter_elements()?;
                let picked = if name == "first" {
                    elems.first()
                } else {
                    elems.last()
                };
                Ok(picked.cloned().unwrap_or_else(JVal::undefined))
            }
            "reverse" => {
                let mut elems = target.iter_elements()?;
                elems.reverse();
                Ok(JVal::new(ValKind::Array(elems)))
            }
            "join" => builtin_join(target, pos_args, kw_args),
            "append" => builtin_append(target, pos_args),
            "pop" => builtin_pop(target, pos_args, kw_args),
            "sort" | "min" | "max" => builtin_order(target, name, pos_args, kw_args),
            "map" => self.builtin_map(target, pos_args, kw_args),
            "select" | "reject" | "selectattr" | "rejectattr" => {
                self.builtin_filter_seq(target, name, pos_args)
            }
            "unique" => {
                let mut distinct = Vec::new();
                for item in target.iter_elements()? {
                    if !distinct.iter().any(|prev| vals_equal(prev, &item)) {
                        distinct.push(item);
                    }
                }
                Ok(JVal::new(ValKind::Array(distinct)))
            }
            "sum" => self.builtin_sum(target, pos_args, kw_args),
            "get" => {
                let key = pos_args
                    .get(1)
                    .ok_or_else(|| JinjaError::std("Jinja dict.get requires a key"))?;
                let found = target.get_member(key, true);
                if found.is_undefined() {
                    Ok(pick_arg(pos_args, kw_args, "default", 2, JVal::none()))
                } else {
                    Ok(found)
                }
            }
            _ => builtin_dict_view(target, name, pos_args, kw_args),
        }
    }

    fn builtin_map(
        &mut self,
        target: &JVal,
        pos_args: &[JVal],
        kw_args: &BTreeMap<String, JVal>,
    ) -> JResult<JVal> {
        let attr_key = pick_arg(
            pos_args,
            kw_args,
            "attribute",
            usize::MAX,
            JVal::undefined(),
        );
        let default_val = pick_arg(pos_args, kw_args, "default", usize::MAX, JVal::undefined());
        let mut mapped = Vec::new();
        for item in target.iter_elements()? {
            let val = if attr_key.is_undefined() {
                let filter_name = String::from_utf8(
                    pos_args
                        .get(1)
                        .ok_or_else(|| JinjaError::std("Jinja map requires filter or attribute"))?
                        .raw_bytes(),
                )
                .map_err(|_| JinjaError::std("Jinja map filter name must be UTF-8"))?;
                let mut args = vec![item];
                args.extend(pos_args.iter().skip(2).cloned());
                self.invoke_builtin(&filter_name, &args, &BTreeMap::new())?
            } else {
                lookup_attr_path(&item, &attr_key)
            };
            mapped.push(if val.is_undefined() {
                default_val.clone()
            } else {
                val
            });
        }
        Ok(JVal::new(ValKind::Array(mapped)))
    }

    fn builtin_filter_seq(
        &mut self,
        target: &JVal,
        name: &str,
        pos_args: &[JVal],
    ) -> JResult<JVal> {
        if target.is_none() {
            return Ok(JVal::new(ValKind::Array(Vec::new())));
        }
        let by_attr = name.ends_with("attr");
        if by_attr {
            target.0.borrow_mut().array_access = true;
        }
        let test_idx = if by_attr { 2 } else { 1 };
        let test_name = pos_args
            .get(test_idx)
            .map(|val| {
                String::from_utf8(val.raw_bytes())
                    .map_err(|_| JinjaError::std("Jinja test name must be UTF-8"))
            })
            .transpose()?;
        let reject_mode = name.starts_with("reject");
        let mut kept = Vec::new();
        for item in target.iter_elements()? {
            let subject = if by_attr {
                let attr_key = pos_args
                    .get(1)
                    .ok_or_else(|| JinjaError::std("Jinja selectattr requires attribute"))?;
                lookup_attr_path(&item, attr_key)
            } else {
                item.clone()
            };
            let matched = if let Some(ref pred) = test_name {
                self.eval_test(
                    &subject,
                    pred,
                    pos_args.get(test_idx + 1..).unwrap_or_default(),
                )?
            } else {
                subject.truthy()
            };
            if matched != reject_mode {
                kept.push(item);
            }
        }
        Ok(JVal::new(ValKind::Array(kept)))
    }

    fn builtin_sum(
        &mut self,
        target: &JVal,
        pos_args: &[JVal],
        kw_args: &BTreeMap<String, JVal>,
    ) -> JResult<JVal> {
        let attr_key = pick_arg(pos_args, kw_args, "attribute", 1, JVal::undefined());
        let mut total = pick_arg(pos_args, kw_args, "start", 2, JVal::int(0));
        for item in target.iter_elements()? {
            let operand = if attr_key.is_undefined() {
                item
            } else {
                lookup_attr_path(&item, &attr_key)
            };
            total = self.eval_binary("+", total, operand)?;
        }
        Ok(total)
    }

    fn eval_test(&mut self, value: &JVal, name: &str, args: &[JVal]) -> JResult<bool> {
        self.tick()?;
        value.mark_used();
        let other = args.first().cloned().unwrap_or_else(JVal::undefined);
        match name {
            "defined" => Ok(!value.is_undefined()),
            "undefined" => Ok(value.is_undefined()),
            "none" => Ok(matches!(value.kind_cloned(), ValKind::None)),
            "boolean" => Ok(matches!(value.kind_cloned(), ValKind::Bool(_))),
            "true" => Ok(matches!(value.kind_cloned(), ValKind::Bool(true))),
            "false" => Ok(matches!(value.kind_cloned(), ValKind::Bool(false))),
            "integer" => Ok(matches!(value.kind_cloned(), ValKind::Int(_))),
            "float" => Ok(matches!(value.kind_cloned(), ValKind::Float(_))),
            "number" => Ok(matches!(
                value.kind_cloned(),
                ValKind::Int(_) | ValKind::Float(_)
            )),
            "string" => Ok(value.is_string()),
            "mapping" => Ok(matches!(value.kind_cloned(), ValKind::Object(_))),
            "sequence" | "iterable" => Ok(matches!(
                value.kind_cloned(),
                ValKind::Undefined | ValKind::Array(_) | ValKind::Tuple(_) | ValKind::String(_)
            )),
            "callable" => Ok(matches!(
                value.kind_cloned(),
                ValKind::Function(_)
                    | ValKind::Method(..)
                    | ValKind::Macro(..)
                    | ValKind::Caller(..)
            )),
            "odd" => Ok(value.as_i64()? % 2 != 0),
            "even" => Ok(value.as_i64()? % 2 == 0),
            "divisibleby" => {
                let divisor = other.as_i64()?;
                if divisor == 0 {
                    return Err(JinjaError::std("Jinja divisibleby zero"));
                }
                let rem = value
                    .as_i64()?
                    .checked_rem(divisor)
                    .ok_or_else(|| JinjaError::std("Jinja divisibleby overflow"))?;
                Ok(rem == 0)
            }
            "eq" | "equalto" | "sameas" => Ok(vals_equal(value, &other)),
            "ne" => Ok(!vals_equal(value, &other)),
            "ge" => Ok(!vals_compare(value, &other)?.is_lt()),
            "gt" | "greaterthan" => Ok(vals_compare(value, &other)?.is_gt()),
            "lt" | "lessthan" => Ok(vals_compare(value, &other)?.is_lt()),
            "le" => Ok(!vals_compare(value, &other)?.is_gt()),
            "in" => val_contains(&other, value),
            "lower" => {
                let raw = value.raw_bytes();
                Ok(raw.iter().any(u8::is_ascii_lowercase)
                    && !raw.iter().any(u8::is_ascii_uppercase))
            }
            "upper" => {
                let raw = value.raw_bytes();
                Ok(raw.iter().any(u8::is_ascii_uppercase)
                    && !raw.iter().any(u8::is_ascii_lowercase))
            }
            "escaped" => Ok(true),
            "filter" => {
                let candidate = String::from_utf8(value.raw_bytes())
                    .map_err(|_| JinjaError::std("Jinja filter name must be UTF-8"))?;
                Ok(is_known_filter(&candidate))
            }
            "test" => {
                let candidate = String::from_utf8(value.raw_bytes())
                    .map_err(|_| JinjaError::std("Jinja test name must be UTF-8"))?;
                Ok(is_known_test(&candidate))
            }
            _ => Err(JinjaError::std(format!("unknown Jinja test '{name}'"))),
        }
    }
}

fn eval_int_binary(op: &str, lhs: &JVal, rhs: &JVal) -> JResult<JVal> {
    let a = lhs.as_i64()?;
    let b = rhs.as_i64()?;
    let val = match op {
        "+" => a.checked_add(b),
        "-" => a.checked_sub(b),
        "*" => a.checked_mul(b),
        "//" => python_floor_div(a, b),
        _ => a.checked_rem(b),
    }
    .ok_or_else(|| JinjaError::std("Jinja integer overflow or division by zero"))?;
    Ok(JVal::int(val))
}

fn eval_float_binary(op: &str, lhs: &JVal, rhs: &JVal) -> JResult<JVal> {
    let a = lhs.as_f64()?;
    let b = rhs.as_f64()?;
    if matches!(op, "/" | "//" | "%") && b == 0.0 {
        return Err(JinjaError::std("Jinja floating division by zero"));
    }
    let result = match op {
        "+" => a + b,
        "-" => a - b,
        "*" => a * b,
        "/" => a / b,
        "//" => (a / b).floor(),
        _ => a.rem_euclid(b),
    };
    if !result.is_finite() {
        return Err(JinjaError::std("Jinja non-finite arithmetic result"));
    }
    Ok(JVal::float(result))
}

fn is_seq_repeat_pair(lhs: &JVal, rhs: &JVal) -> bool {
    matches!(
        lhs.kind_cloned(),
        ValKind::String(_) | ValKind::Array(_) | ValKind::Tuple(_)
    ) || matches!(
        rhs.kind_cloned(),
        ValKind::String(_) | ValKind::Array(_) | ValKind::Tuple(_)
    )
}

fn eval_seq_repeat(lhs: JVal, rhs: JVal) -> JResult<JVal> {
    let (container, count) = if lhs.is_int() {
        (rhs, lhs.as_i64()?)
    } else {
        let n = rhs.as_i64()?;
        (lhs, n)
    };
    let repeat = usize::try_from(count.max(0))
        .map_err(|_| JinjaError::std("Jinja repetition count bound exceeded"))?;
    if repeat.saturating_mul(container.length()?) > MAX_CONTAINER_LEN {
        return Err(JinjaError::std("Jinja repetition output bound exceeded"));
    }
    if container.is_string() {
        let base = container.to_spans();
        let repeated = (0..repeat).flat_map(|_| base.clone()).collect();
        Ok(JVal::new(ValKind::String(repeated)))
    } else {
        let is_tuple = matches!(container.kind_cloned(), ValKind::Tuple(_));
        let base = container.iter_elements()?;
        let repeated = (0..repeat).flat_map(|_| base.clone()).collect();
        Ok(JVal::new(if is_tuple {
            ValKind::Tuple(repeated)
        } else {
            ValKind::Array(repeated)
        }))
    }
}

fn invoke_string_builtin(
    target: &JVal,
    name: &str,
    pos_args: &[JVal],
    kw_args: &BTreeMap<String, JVal>,
) -> JResult<JVal> {
    match name {
        "upper" | "lower" | "capitalize" | "title" => Ok(builtin_case(target, name)),
        "strip" | "trim" | "lstrip" | "rstrip" => {
            let chars = pick_bytes_arg(pos_args, kw_args, "chars", 1, b" \t\n\r\x0b\x0c");
            Ok(builtin_trim(target, name, &chars))
        }
        "startswith" | "endswith" => {
            let needle = pick_bytes_arg(pos_args, kw_args, "prefix", 1, b"");
            let text = target.raw_bytes();
            let matched = if name == "startswith" {
                text.starts_with(&needle)
            } else {
                text.ends_with(&needle)
            };
            Ok(JVal::boolean(matched))
        }
        "replace" => builtin_replace(target, pos_args, kw_args),
        "split" | "rsplit" => builtin_split(target, name, pos_args, kw_args),
        "format" => builtin_format(target, pos_args, kw_args),
        "indent" => builtin_indent(target, pos_args, kw_args),
        "wordcount" => {
            let count = target
                .raw_bytes()
                .split(u8::is_ascii_whitespace)
                .filter(|word| !word.is_empty())
                .count();
            let count_i64 =
                i64::try_from(count).map_err(|_| JinjaError::std("Jinja wordcount overflow"))?;
            Ok(JVal::int(count_i64))
        }
        "truncate" => {
            let mut raw = target.raw_bytes();
            let limit_i64 = pick_arg(pos_args, kw_args, "length", 1, JVal::int(255)).as_i64()?;
            let limit = usize::try_from(limit_i64.max(0)).unwrap_or(usize::MAX);
            let suffix = pick_bytes_arg(pos_args, kw_args, "end", 3, b"...");
            if raw.len() > limit {
                raw.truncate(limit.saturating_sub(suffix.len()));
                raw.extend(suffix);
            }
            Ok(preserving_provenance(target, raw))
        }
        _ => Err(JinjaError::std(format!(
            "unsupported Jinja string builtin '{name}'"
        ))),
    }
}

const fn python_floor_div(a: i64, b: i64) -> Option<i64> {
    if b == 0 {
        return None;
    }
    let Some(q) = a.checked_div(b) else {
        return None;
    };
    let Some(r) = a.checked_rem(b) else {
        return None;
    };
    if r != 0 && ((r < 0) != (b < 0)) {
        q.checked_sub(1)
    } else {
        Some(q)
    }
}

fn is_global_builtin(name: &str) -> bool {
    matches!(
        name,
        "range" | "namespace" | "raise_exception" | "tojson" | "strftime_now"
    ) || name.starts_with("test_is_")
}

fn has_value_method(value: &JVal, name: &str) -> bool {
    if value.is_namespace() {
        return false;
    }
    match value.kind_cloned() {
        ValKind::String(_) => matches!(
            name,
            "default"
                | "upper"
                | "lower"
                | "strip"
                | "rstrip"
                | "lstrip"
                | "title"
                | "capitalize"
                | "length"
                | "startswith"
                | "endswith"
                | "split"
                | "rsplit"
                | "replace"
                | "format"
                | "int"
                | "float"
                | "string"
                | "slice"
                | "safe"
                | "tojson"
                | "indent"
                | "join"
        ),
        ValKind::Array(_) | ValKind::Tuple(_) => matches!(
            name,
            "default"
                | "list"
                | "first"
                | "last"
                | "length"
                | "count"
                | "reverse"
                | "sort"
                | "join"
                | "pop"
                | "append"
                | "map"
                | "select"
                | "reject"
                | "selectattr"
                | "rejectattr"
                | "min"
                | "max"
                | "sum"
                | "unique"
                | "safe"
                | "string"
                | "tojson"
        ),
        ValKind::Object(_) => matches!(
            name,
            "get"
                | "items"
                | "keys"
                | "values"
                | "dictsort"
                | "length"
                | "join"
                | "string"
                | "tojson"
        ),
        ValKind::Int(_) | ValKind::Float(_) | ValKind::Bool(_) => matches!(
            name,
            "default" | "abs" | "int" | "float" | "safe" | "string" | "tojson"
        ),
        _ => false,
    }
}

fn is_known_filter(name: &str) -> bool {
    matches!(
        name,
        "abs"
            | "capitalize"
            | "count"
            | "d"
            | "default"
            | "dictsort"
            | "first"
            | "float"
            | "format"
            | "from_json"
            | "indent"
            | "int"
            | "items"
            | "join"
            | "keys"
            | "last"
            | "length"
            | "list"
            | "lower"
            | "lstrip"
            | "map"
            | "max"
            | "min"
            | "reject"
            | "rejectattr"
            | "replace"
            | "reverse"
            | "rsplit"
            | "rstrip"
            | "safe"
            | "select"
            | "selectattr"
            | "slice"
            | "sort"
            | "split"
            | "string"
            | "strip"
            | "sum"
            | "title"
            | "tojson"
            | "trim"
            | "truncate"
            | "unique"
            | "upper"
            | "values"
            | "wordcount"
    )
}

fn is_known_test(name: &str) -> bool {
    matches!(
        name,
        "boolean"
            | "callable"
            | "odd"
            | "even"
            | "false"
            | "true"
            | "divisibleby"
            | "string"
            | "integer"
            | "float"
            | "number"
            | "iterable"
            | "sequence"
            | "mapping"
            | "lower"
            | "upper"
            | "none"
            | "defined"
            | "undefined"
            | "eq"
            | "equalto"
            | "ge"
            | "gt"
            | "greaterthan"
            | "lt"
            | "lessthan"
            | "ne"
            | "in"
            | "test"
            | "sameas"
            | "escaped"
            | "filter"
    )
}

fn preview_utf8(raw: &[u8]) -> String {
    let slice = &raw[..raw.len().min(MAX_DIAG_PREVIEW_BYTES)];
    let preview = String::from_utf8_lossy(slice);
    if raw.len() <= MAX_DIAG_PREVIEW_BYTES && preview.len() <= MAX_DIAG_PREVIEW_BYTES {
        return preview.into_owned();
    }
    let mut cut = preview.len().min(MAX_DIAG_PREVIEW_BYTES.saturating_sub(3));
    while cut > 0 && !preview.is_char_boundary(cut) {
        cut -= 1;
    }
    format!("{}...", &preview[..cut])
}

fn pick_arg(
    pos_args: &[JVal],
    kw_args: &BTreeMap<String, JVal>,
    name: &str,
    index: usize,
    fallback: JVal,
) -> JVal {
    kw_args
        .get(name)
        .or_else(|| pos_args.get(index))
        .cloned()
        .unwrap_or(fallback)
}

fn pick_bytes_arg(
    pos_args: &[JVal],
    kw_args: &BTreeMap<String, JVal>,
    name: &str,
    index: usize,
    fallback: &[u8],
) -> Vec<u8> {
    pick_arg(
        pos_args,
        kw_args,
        name,
        index,
        JVal::bytes(fallback.to_vec(), false),
    )
    .raw_bytes()
}

fn preserving_provenance(original: &JVal, bytes: Vec<u8>) -> JVal {
    JVal::bytes(bytes, original.all_spans_input())
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        Some(0)
    } else {
        haystack.windows(needle.len()).position(|win| win == needle)
    }
}

fn lookup_attr_path(item: &JVal, key: &JVal) -> JVal {
    if key.is_string() {
        let mut cursor = item.clone();
        for part in key.raw_bytes().split(|&byte| byte == b'.') {
            let segment = String::from_utf8_lossy(part);
            let seg_key = segment
                .parse::<i64>()
                .map_or_else(|_| JVal::text_str(segment.into_owned()), JVal::int);
            cursor = cursor.get_member(&seg_key, true);
        }
        cursor
    } else {
        item.get_member(key, true)
    }
}

fn builtin_range(pos_args: &[JVal], kw_args: &BTreeMap<String, JVal>) -> JResult<JVal> {
    if pos_args.is_empty() || pos_args.len() > 3 || !kw_args.is_empty() {
        return Err(JinjaError::std(
            "Jinja range requires 1 to 3 positional args",
        ));
    }
    let (begin, limit) = if pos_args.len() == 1 {
        (0, pos_args[0].as_i64()?)
    } else {
        (pos_args[0].as_i64()?, pos_args[1].as_i64()?)
    };
    let stride = if let Some(step_val) = pos_args.get(2) {
        step_val.as_i64()?
    } else {
        1
    };
    if stride == 0 {
        return Err(JinjaError::std("Jinja range step cannot be zero"));
    }
    let mut seq = Vec::new();
    let mut cur = begin;
    while if stride > 0 { cur < limit } else { cur > limit } {
        if seq.len() >= MAX_CONTAINER_LEN {
            return Err(JinjaError::std("Jinja range length bound exceeded"));
        }
        seq.push(JVal::int(cur));
        let Some(next) = cur.checked_add(stride) else {
            break;
        };
        cur = next;
    }
    Ok(JVal::new(ValKind::Array(seq)))
}

fn builtin_case(target: &JVal, mode: &str) -> JVal {
    let mut raw = target.raw_bytes();
    match mode {
        "upper" => raw.make_ascii_uppercase(),
        "lower" => raw.make_ascii_lowercase(),
        "capitalize" => {
            raw.make_ascii_lowercase();
            if let Some(first) = raw.first_mut() {
                first.make_ascii_uppercase();
            }
        }
        _ => {
            let mut in_word = false;
            for byte in &mut raw {
                let alpha = byte.is_ascii_alphabetic();
                if alpha {
                    if in_word {
                        byte.make_ascii_lowercase();
                    } else {
                        byte.make_ascii_uppercase();
                    }
                }
                in_word = alpha;
            }
        }
    }
    preserving_provenance(target, raw)
}

fn builtin_trim(target: &JVal, mode: &str, chars: &[u8]) -> JVal {
    let raw = target.raw_bytes();
    let mut head = 0_usize;
    let mut tail = raw.len();
    if mode != "rstrip" {
        while head < tail && chars.contains(&raw[head]) {
            head += 1;
        }
    }
    if mode != "lstrip" {
        while tail > head && chars.contains(&raw[tail - 1]) {
            tail -= 1;
        }
    }
    preserving_provenance(target, raw[head..tail].to_vec())
}

fn builtin_replace(
    target: &JVal,
    pos_args: &[JVal],
    kw_args: &BTreeMap<String, JVal>,
) -> JResult<JVal> {
    let raw = target.raw_bytes();
    let old = pick_bytes_arg(pos_args, kw_args, "old", 1, b"");
    let new = pick_bytes_arg(pos_args, kw_args, "new", 2, b"");
    let count = pick_arg(pos_args, kw_args, "count", 3, JVal::int(-1)).as_i64()?;
    let mut left = if count < 0 {
        usize::MAX
    } else {
        usize::try_from(count).unwrap_or(usize::MAX)
    };
    let mut out = Vec::new();
    if old.is_empty() {
        if left != 0 {
            out.extend(&new);
            left -= 1;
        }
        for byte in raw {
            out.push(byte);
            if left != 0 {
                out.extend(&new);
                left -= 1;
            }
            if out.len() > MAX_OUTPUT_BYTES {
                return Err(JinjaError::std("Jinja replace output bound exceeded"));
            }
        }
    } else {
        let mut pos = 0_usize;
        while left != 0 {
            let Some(found) = find_subslice(&raw[pos..], &old) else {
                break;
            };
            out.extend(&raw[pos..pos + found]);
            out.extend(&new);
            pos += found + old.len();
            left -= 1;
            if out.len() > MAX_OUTPUT_BYTES {
                return Err(JinjaError::std("Jinja replace output bound exceeded"));
            }
        }
        out.extend(&raw[pos..]);
    }
    Ok(preserving_provenance(target, out))
}

fn builtin_split(
    target: &JVal,
    mode: &str,
    pos_args: &[JVal],
    kw_args: &BTreeMap<String, JVal>,
) -> JResult<JVal> {
    let raw = target.raw_bytes();
    let sep = pick_bytes_arg(pos_args, kw_args, "sep", 1, b" ");
    if sep.is_empty() {
        return Err(JinjaError::std("Jinja split separator cannot be empty"));
    }
    let maxsplit = pick_arg(pos_args, kw_args, "maxsplit", 2, JVal::int(-1)).as_i64()?;
    let mut left = if maxsplit < 0 {
        usize::MAX
    } else {
        usize::try_from(maxsplit).unwrap_or(usize::MAX)
    };
    let mut pieces = Vec::new();
    if mode == "split" {
        let mut pos = 0_usize;
        while left != 0 {
            let Some(found) = find_subslice(&raw[pos..], &sep) else {
                break;
            };
            pieces.push(JVal::bytes(raw[pos..pos + found].to_vec(), false));
            pos += found + sep.len();
            left -= 1;
        }
        pieces.push(preserving_provenance(target, raw[pos..].to_vec()));
    } else {
        let mut end = raw.len();
        while left != 0 {
            let Some(found) = raw[..end].windows(sep.len()).rposition(|win| win == sep) else {
                break;
            };
            pieces.push(JVal::bytes(raw[found + sep.len()..end].to_vec(), false));
            end = found;
            left -= 1;
        }
        pieces.push(preserving_provenance(target, raw[..end].to_vec()));
        pieces.reverse();
    }
    Ok(JVal::new(ValKind::Array(pieces)))
}

fn builtin_format(
    target: &JVal,
    pos_args: &[JVal],
    kw_args: &BTreeMap<String, JVal>,
) -> JResult<JVal> {
    let template = target.raw_bytes();
    let is_input = target.all_spans_input();
    let mut spans = Vec::new();
    let mut literal = Vec::new();
    let mut pos = 0_usize;
    let mut auto_idx = 1_usize;
    let mut total_bytes = 0_usize;

    while pos < template.len() {
        if template[pos] == b'{' {
            if template.get(pos + 1) == Some(&b'{') {
                literal.push(b'{');
                pos += 2;
                continue;
            }
            if !literal.is_empty() {
                total_bytes += literal.len();
                spans.push(TextSpan {
                    bytes: std::mem::take(&mut literal),
                    is_input,
                });
            }
            let close = template[pos + 1..]
                .iter()
                .position(|&byte| byte == b'}')
                .map(|offset| pos + 1 + offset)
                .ok_or_else(|| JinjaError::std("Jinja unclosed format placeholder"))?;
            let field_name = String::from_utf8(template[pos + 1..close].to_vec())
                .map_err(|_| JinjaError::std("Jinja format placeholder must be UTF-8"))?;
            let arg_val = if field_name.is_empty() {
                let picked = pos_args.get(auto_idx).cloned();
                auto_idx += 1;
                picked
            } else if let Ok(explicit_idx) = field_name.parse::<usize>() {
                pos_args.get(explicit_idx + 1).cloned()
            } else {
                kw_args.get(&field_name).cloned()
            }
            .ok_or_else(|| JinjaError::std("Jinja missing format argument"))?;
            let arg_spans = arg_val.to_spans();
            total_bytes += arg_spans.iter().map(|span| span.bytes.len()).sum::<usize>();
            if total_bytes > MAX_OUTPUT_BYTES {
                return Err(JinjaError::std("Jinja format output bound exceeded"));
            }
            spans.extend(arg_spans);
            pos = close + 1;
        } else if template[pos] == b'}' {
            if template.get(pos + 1) != Some(&b'}') {
                return Err(JinjaError::std("Jinja unmatched format brace"));
            }
            literal.push(b'}');
            pos += 2;
        } else {
            literal.push(template[pos]);
            pos += 1;
        }
    }
    if !literal.is_empty() {
        spans.push(TextSpan {
            bytes: literal,
            is_input,
        });
    }
    Ok(JVal::new(ValKind::String(spans)))
}

fn builtin_int(
    target: &JVal,
    pos_args: &[JVal],
    kw_args: &BTreeMap<String, JVal>,
) -> JResult<JVal> {
    if target.is_undefined() {
        return Ok(JVal::int(0));
    }
    if !target.is_string() {
        return Ok(JVal::int(target.as_i64()?));
    }
    let fallback = pick_arg(pos_args, kw_args, "default", 1, JVal::int(0));
    let requested_base = pick_arg(pos_args, kw_args, "base", 2, JVal::int(10)).as_i64()?;
    if requested_base != 0 && !(2..=36).contains(&requested_base) {
        return Err(JinjaError::std("Jinja int base must be 0 or 2..=36"));
    }
    let raw = target.raw_bytes();
    let lossy = String::from_utf8_lossy(&raw);
    let mut slice = lossy.trim_start();
    let negative = slice.starts_with('-');
    if slice.starts_with(['-', '+']) {
        slice = &slice[1..];
    }
    let radix = if requested_base == 0 {
        if slice.starts_with("0x") || slice.starts_with("0X") {
            16_u32
        } else if slice.starts_with('0') {
            8_u32
        } else {
            10_u32
        }
    } else {
        u32::try_from(requested_base).map_err(|_| JinjaError::std("invalid Jinja int radix"))?
    };
    if radix == 16 && (slice.starts_with("0x") || slice.starts_with("0X")) {
        slice = &slice[2..];
    }
    let digits_len = slice
        .char_indices()
        .take_while(|&(_, ch)| ch.is_digit(radix))
        .last()
        .map_or(0, |(idx, ch)| idx + ch.len_utf8());
    let parsed = i64::from_str_radix(&slice[..digits_len], radix)
        .ok()
        .and_then(|num| {
            if negative {
                num.checked_neg()
            } else {
                Some(num)
            }
        });
    Ok(parsed.map_or(fallback, JVal::int))
}

fn builtin_float(
    target: &JVal,
    pos_args: &[JVal],
    kw_args: &BTreeMap<String, JVal>,
) -> JResult<JVal> {
    if !target.is_string() {
        return Ok(JVal::float(target.as_f64()?));
    }
    let text = String::from_utf8(target.raw_bytes())
        .map_err(|_| JinjaError::std("Jinja float input must be UTF-8"))?;
    let parsed = text
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|num| num.is_finite());
    Ok(parsed.map_or_else(
        || pick_arg(pos_args, kw_args, "default", 1, JVal::float(0.0)),
        JVal::float,
    ))
}

fn builtin_abs(target: &JVal) -> JResult<JVal> {
    if let ValKind::Int(num) = target.kind_cloned() {
        let abs_val = num
            .checked_abs()
            .ok_or_else(|| JinjaError::std("Jinja abs integer overflow"))?;
        Ok(JVal::int(abs_val))
    } else {
        Ok(JVal::float(target.as_f64()?.abs()))
    }
}

fn builtin_join(
    target: &JVal,
    pos_args: &[JVal],
    kw_args: &BTreeMap<String, JVal>,
) -> JResult<JVal> {
    let delim = pick_arg(pos_args, kw_args, "d", 1, JVal::text_str(""));
    let attr_key = pick_arg(pos_args, kw_args, "attribute", 2, JVal::undefined());
    let mut spans = Vec::new();
    for (idx, item) in target.iter_elements()?.iter().enumerate() {
        if idx != 0 {
            spans.extend(delim.to_spans());
        }
        let extracted = if attr_key.is_undefined() {
            item.clone()
        } else {
            lookup_attr_path(item, &attr_key)
        };
        spans.extend(extracted.to_spans());
    }
    Ok(JVal::new(ValKind::String(spans)))
}

fn builtin_append(target: &JVal, pos_args: &[JVal]) -> JResult<JVal> {
    let elem = pos_args
        .get(1)
        .cloned()
        .ok_or_else(|| JinjaError::std("Jinja array.append requires an item"))?;
    target.ensure_acyclic_insert(&elem)?;
    let mut cell = target.0.borrow_mut();
    let ValKind::Array(elems) = &mut cell.kind else {
        return Err(JinjaError::std("Jinja append requires an array"));
    };
    if elems.len() >= MAX_CONTAINER_LEN {
        return Err(JinjaError::std("Jinja array size bound exceeded"));
    }
    elems.push(elem);
    Ok(JVal::none())
}

fn builtin_pop(
    target: &JVal,
    pos_args: &[JVal],
    kw_args: &BTreeMap<String, JVal>,
) -> JResult<JVal> {
    let raw_idx = pick_arg(pos_args, kw_args, "index", 1, JVal::int(-1)).as_i64()?;
    let mut cell = target.0.borrow_mut();
    let ValKind::Array(elems) = &mut cell.kind else {
        return Err(JinjaError::std("Jinja pop requires an array"));
    };
    let pos = resolve_index(raw_idx, elems.len())
        .ok_or_else(|| JinjaError::std("Jinja pop index out of range"))?;
    Ok(elems.remove(pos))
}

fn builtin_order(
    target: &JVal,
    mode: &str,
    pos_args: &[JVal],
    kw_args: &BTreeMap<String, JVal>,
) -> JResult<JVal> {
    let elems = target.iter_elements()?;
    let descending = mode == "max"
        || (mode == "sort"
            && pick_arg(pos_args, kw_args, "reverse", 1, JVal::boolean(false)).truthy());
    let case_sensitive =
        pick_arg(pos_args, kw_args, "case_sensitive", 2, JVal::boolean(false)).truthy();
    let attr_pos = if mode == "sort" { 3 } else { 2 };
    let attr_key = pick_arg(pos_args, kw_args, "attribute", attr_pos, JVal::undefined());
    let mut keyed = Vec::with_capacity(elems.len());
    for item in elems {
        let mut sort_key = if attr_key.is_undefined() {
            item.clone()
        } else {
            lookup_attr_path(&item, &attr_key)
        };
        if !case_sensitive && sort_key.is_string() {
            let mut lower = sort_key.raw_bytes();
            lower.make_ascii_lowercase();
            sort_key = JVal::bytes(lower, false);
        }
        keyed.push((sort_key, item));
    }
    let mut sort_err: Option<JinjaError> = None;
    keyed.sort_by(|(ka, _), (kb, _)| match vals_compare(ka, kb) {
        Ok(ord) => {
            if descending {
                ord.reverse()
            } else {
                ord
            }
        }
        Err(err) => {
            if sort_err.is_none() {
                sort_err = Some(err);
            }
            Ordering::Equal
        }
    });
    if let Some(err) = sort_err {
        return Err(err);
    }
    let mut sorted: Vec<JVal> = keyed.into_iter().map(|(_, item)| item).collect();
    if mode == "sort" {
        Ok(JVal::new(ValKind::Array(sorted)))
    } else if sorted.is_empty() {
        Ok(JVal::undefined())
    } else {
        Ok(sorted.remove(0))
    }
}

fn builtin_dict_view(
    target: &JVal,
    mode: &str,
    pos_args: &[JVal],
    kw_args: &BTreeMap<String, JVal>,
) -> JResult<JVal> {
    if target.is_undefined() {
        return Ok(JVal::new(ValKind::Array(Vec::new())));
    }
    target.mark_used();
    let ValKind::Object(mut pairs) = target.kind_cloned() else {
        return Err(JinjaError::std("Jinja object method requires an object"));
    };
    if mode == "dictsort" {
        let case_sensitive =
            pick_arg(pos_args, kw_args, "case_sensitive", 1, JVal::boolean(false)).truthy();
        let by_field = pick_bytes_arg(pos_args, kw_args, "by", 2, b"key");
        if by_field != b"key" && by_field != b"value" {
            return Err(JinjaError::std("Jinja dictsort 'by' must be key or value"));
        }
        let by_key = by_field == b"key";
        let descending = pick_arg(pos_args, kw_args, "reverse", 3, JVal::boolean(false)).truthy();
        let mut sort_err: Option<JinjaError> = None;
        let normalize = |val: &JVal| -> JVal {
            if !case_sensitive && val.is_string() {
                let mut lower = val.raw_bytes();
                lower.make_ascii_lowercase();
                JVal::bytes(lower, false)
            } else {
                val.clone()
            }
        };
        pairs.sort_by(|pair_a, pair_b| {
            let lhs = normalize(if by_key { &pair_a.0 } else { &pair_a.1 });
            let rhs = normalize(if by_key { &pair_b.0 } else { &pair_b.1 });
            match vals_compare(&lhs, &rhs) {
                Ok(ord) => {
                    if descending {
                        ord.reverse()
                    } else {
                        ord
                    }
                }
                Err(err) => {
                    if sort_err.is_none() {
                        sort_err = Some(err);
                    }
                    Ordering::Equal
                }
            }
        });
        if let Some(err) = sort_err {
            return Err(err);
        }
    }
    let projected = pairs
        .into_iter()
        .map(|(k, v)| match mode {
            "keys" => k,
            "values" => v,
            _ => JVal::new(ValKind::Tuple(vec![k, v])),
        })
        .collect();
    Ok(JVal::new(ValKind::Array(projected)))
}

fn builtin_indent(
    target: &JVal,
    pos_args: &[JVal],
    kw_args: &BTreeMap<String, JVal>,
) -> JResult<JVal> {
    let width_val = pick_arg(pos_args, kw_args, "width", 1, JVal::int(4));
    let prefix = if width_val.is_string() {
        width_val.raw_bytes()
    } else {
        let count = width_val.as_i64()?;
        if !(0..=65_536).contains(&count) {
            return Err(JinjaError::std("Jinja indent width bound exceeded"));
        }
        vec![b' '; usize::try_from(count).unwrap_or(0)]
    };
    let indent_first = pick_arg(pos_args, kw_args, "first", 2, JVal::boolean(false)).truthy();
    let indent_blank = pick_arg(pos_args, kw_args, "blank", 3, JVal::boolean(false)).truthy();
    let raw = target.raw_bytes();
    let mut out = Vec::new();
    for (idx, line) in raw.split(|&byte| byte == b'\n').enumerate() {
        if idx != 0 {
            out.push(b'\n');
        }
        let should_indent = if idx == 0 {
            indent_first
        } else {
            !line.is_empty() || indent_blank
        };
        if should_indent {
            out.extend(&prefix);
        }
        out.extend(line);
        if out.len() > MAX_OUTPUT_BYTES {
            return Err(JinjaError::std("Jinja indent output bound exceeded"));
        }
    }
    Ok(preserving_provenance(target, out))
}

struct JsonDumpOpts {
    ensure_ascii: bool,
    sort_keys: bool,
    indent: Option<usize>,
    item_sep: Vec<u8>,
    colon_sep: Vec<u8>,
}

impl JsonDumpOpts {
    fn from_args(pos_args: &[JVal], kw_args: &BTreeMap<String, JVal>) -> JResult<Self> {
        let indent_val = pick_arg(pos_args, kw_args, "indent", 1, JVal::undefined());
        let indent = if matches!(indent_val.kind_cloned(), ValKind::Undefined | ValKind::None) {
            None
        } else {
            let spaces = indent_val.as_i64()?;
            if !(0..=256).contains(&spaces) {
                return Err(JinjaError::std("Jinja JSON indent bound exceeded"));
            }
            Some(
                usize::try_from(spaces)
                    .map_err(|_| JinjaError::std("Jinja JSON indent overflow"))?,
            )
        };
        let sep_val = pick_arg(
            pos_args,
            kw_args,
            "separators",
            usize::MAX,
            JVal::undefined(),
        );
        let (item_sep, colon_sep) = if sep_val.is_undefined() {
            (
                if indent.is_some() {
                    b",".to_vec()
                } else {
                    b", ".to_vec()
                },
                b": ".to_vec(),
            )
        } else {
            let pair = sep_val.iter_elements()?;
            if pair.len() != 2 {
                return Err(JinjaError::std("Jinja JSON separators must be a pair"));
            }
            (pair[0].raw_bytes(), pair[1].raw_bytes())
        };
        Ok(Self {
            ensure_ascii: pick_arg(
                pos_args,
                kw_args,
                "ensure_ascii",
                usize::MAX,
                JVal::boolean(false),
            )
            .truthy(),
            sort_keys: pick_arg(
                pos_args,
                kw_args,
                "sort_keys",
                usize::MAX,
                JVal::boolean(false),
            )
            .truthy(),
            indent,
            item_sep,
            colon_sep,
        })
    }
}

fn encode_json_str(raw: &[u8], ensure_ascii: bool) -> JResult<Vec<u8>> {
    let text = std::str::from_utf8(raw)
        .map_err(|_| JinjaError::std("Jinja tojson string is not valid UTF-8"))?;
    let mut out = vec![b'"'];
    for ch in text.chars() {
        match ch {
            '"' => out.extend(b"\\\""),
            '\\' => out.extend(b"\\\\"),
            '\n' => out.extend(b"\\n"),
            '\r' => out.extend(b"\\r"),
            '\t' => out.extend(b"\\t"),
            '\u{8}' => out.extend(b"\\b"),
            '\u{c}' => out.extend(b"\\f"),
            ch if ch < ' ' || (ensure_ascii && !ch.is_ascii()) => {
                let mut utf16 = [0_u16; 2];
                for unit in ch.encode_utf16(&mut utf16) {
                    out.extend(format!("\\u{unit:04x}").bytes());
                }
            }
            _ => {
                let mut utf8 = [0_u8; 4];
                out.extend(ch.encode_utf8(&mut utf8).as_bytes());
            }
        }
    }
    out.push(b'"');
    Ok(out)
}

fn encode_json_bytes(value: &JVal, opts: &JsonDumpOpts, depth: usize) -> JResult<Vec<u8>> {
    if depth > MAX_GRAPH_DEPTH {
        return Err(JinjaError::std("Jinja JSON serialization depth exceeded"));
    }
    value.mark_used();
    let mut out = Vec::new();
    match value.kind_cloned() {
        ValKind::Undefined => {
            return Err(JinjaError::std("Jinja undefined is not JSON serializable"));
        }
        ValKind::None => out.extend(b"null"),
        ValKind::Bool(flag) => out.extend(if flag { &b"true"[..] } else { b"false" }),
        ValKind::Int(_) | ValKind::Float(_) => out.extend(value.raw_bytes()),
        ValKind::String(_) => out.extend(encode_json_str(&value.raw_bytes(), opts.ensure_ascii)?),
        ValKind::Array(elems) | ValKind::Tuple(elems) => {
            out.push(b'[');
            for (idx, elem) in elems.iter().enumerate() {
                if idx != 0 {
                    out.extend(&opts.item_sep);
                }
                if let Some(spaces) = opts.indent {
                    out.push(b'\n');
                    out.resize(out.len() + spaces * (depth + 1), b' ');
                }
                out.extend(encode_json_bytes(elem, opts, depth + 1)?);
            }
            if !elems.is_empty()
                && let Some(spaces) = opts.indent
            {
                out.push(b'\n');
                out.resize(out.len() + spaces * depth, b' ');
            }
            out.push(b']');
        }
        ValKind::Object(mut pairs) => {
            if opts.sort_keys {
                pairs.sort_by_key(|(k, _)| k.raw_bytes());
            }
            out.push(b'{');
            for (idx, (entry_key, entry_val)) in pairs.iter().enumerate() {
                if idx != 0 {
                    out.extend(&opts.item_sep);
                }
                if let Some(spaces) = opts.indent {
                    out.push(b'\n');
                    out.resize(out.len() + spaces * (depth + 1), b' ');
                }
                let key_bytes = match entry_key.kind_cloned() {
                    ValKind::String(_) | ValKind::Int(_) | ValKind::Float(_) => {
                        entry_key.raw_bytes()
                    }
                    ValKind::None => b"null".to_vec(),
                    ValKind::Bool(flag) => {
                        if flag {
                            b"true".to_vec()
                        } else {
                            b"false".to_vec()
                        }
                    }
                    _ => return Err(JinjaError::std("Jinja object key is not JSON serializable")),
                };
                out.extend(encode_json_str(&key_bytes, opts.ensure_ascii)?);
                out.extend(&opts.colon_sep);
                out.extend(encode_json_bytes(entry_val, opts, depth + 1)?);
            }
            if !pairs.is_empty()
                && let Some(spaces) = opts.indent
            {
                out.push(b'\n');
                out.resize(out.len() + spaces * depth, b' ');
            }
            out.push(b'}');
        }
        _ => return Err(JinjaError::std("Jinja callable is not JSON serializable")),
    }
    if out.len() > MAX_OUTPUT_BYTES {
        return Err(JinjaError::std("Jinja JSON output byte bound exceeded"));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Wire protocol validation & dispatch
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum PathSeg {
    Key(String),
    Index(usize),
}

enum HashTraceAction {
    Update(Vec<u8>),
    Digest,
}

enum ValidatedOp {
    HashTrace(Vec<HashTraceAction>),
    Hash(Vec<Vec<u8>>),
    Builtins,
    Caps {
        template: String,
    },
    Render {
        template: String,
        variables: JVal,
    },
    Stats {
        template: String,
        variables: JVal,
        paths: Vec<Vec<PathSeg>>,
    },
    StringParts {
        template: String,
        variables: JVal,
    },
}

fn require_exact_fields(
    obj: &Map<String, Value>,
    required: &[&str],
    optional: &[&str],
) -> Result<()> {
    for &field in required {
        if !obj.contains_key(field) {
            return Err(invalid(format!("missing Jinja operation field '{field}'")));
        }
    }
    for key in obj.keys() {
        let key_str = key.as_str();
        if !required.contains(&key_str) && !optional.contains(&key_str) {
            return Err(invalid(format!("unknown Jinja operation field '{key}'")));
        }
    }
    Ok(())
}

fn nonneg_int(val: &Value, max: u64) -> Result<u64> {
    let num = if let Some(u) = val.as_u64() {
        u
    } else if let Some(i) = val.as_i64() {
        u64::try_from(i).map_err(|_| invalid("negative integer is not permitted"))?
    } else {
        return Err(invalid("expected nonnegative integer"));
    };
    if num > max {
        return Err(invalid("integer exceeds Jinja protocol limit"));
    }
    Ok(num)
}

fn byte_vec(val: &Value, max_len: Option<usize>) -> Result<Vec<u8>> {
    let items = val
        .as_array()
        .ok_or_else(|| invalid("expected byte array"))?;
    if let Some(limit) = max_len
        && items.len() > limit
    {
        return Err(invalid("byte array exceeds protocol limit"));
    }
    items
        .iter()
        .map(|entry| {
            let n = nonneg_int(entry, 255)?;
            u8::try_from(n).map_err(Into::into)
        })
        .collect()
}

fn validate_path(val: &Value) -> Result<Vec<PathSeg>> {
    let items = val
        .as_array()
        .ok_or_else(|| invalid("Jinja path must be an array"))?;
    if items.is_empty() || items.len() > MAX_PATH_SEGMENTS {
        return Err(invalid("Jinja path must contain 1 through 32 segments"));
    }
    if !items[0].is_string() {
        return Err(invalid("Jinja path must begin with a global name"));
    }
    items
        .iter()
        .map(|seg| {
            if let Some(key) = seg.as_str() {
                if key.len() > MAX_PATH_STRING_BYTES {
                    return Err(invalid("Jinja path key exceeds 4096 bytes"));
                }
                Ok(PathSeg::Key(key.to_owned()))
            } else {
                let idx = usize::try_from(nonneg_int(seg, MAX_PATH_INDEX)?)?;
                Ok(PathSeg::Index(idx))
            }
        })
        .collect()
}

fn validate_template(obj: &Map<String, Value>) -> Result<String> {
    let text = obj
        .get("Template")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("Jinja Template must be a string"))?;
    if text.len() > MAX_TEMPLATE_BYTES {
        return Err(invalid("Jinja Template exceeds 65536 UTF-8 bytes"));
    }
    Ok(text.to_owned())
}

fn validate_variables(obj: &Map<String, Value>) -> Result<JVal> {
    let has_obj = obj.contains_key("Variables");
    let has_json = obj.contains_key("VariablesJson");
    if has_obj == has_json {
        return Err(invalid(
            "exactly one of Variables and VariablesJson is required",
        ));
    }
    let raw_json = if let Some(text_val) = obj.get("VariablesJson") {
        text_val
            .as_str()
            .ok_or_else(|| invalid("VariablesJson must be a string"))?
            .to_owned()
    } else {
        let var_obj = obj
            .get("Variables")
            .filter(|val| val.is_object())
            .ok_or_else(|| invalid("Variables must be a JSON object"))?;
        serde_json::to_string(var_obj)?
    };
    parse_variables_json(&raw_json)
}

fn apply_byte_overlays(obj: &Map<String, Value>, root: &JVal) -> Result<()> {
    let Some(overlays_val) = obj.get("ByteStrings") else {
        return Ok(());
    };
    let overlays = overlays_val
        .as_array()
        .ok_or_else(|| invalid("ByteStrings must be an array"))?;
    if overlays.len() > MAX_BYTE_OVERLAYS {
        return Err(invalid("ByteStrings exceeds 16 overlays"));
    }
    for overlay_val in overlays {
        let overlay = overlay_val
            .as_object()
            .ok_or_else(|| invalid("ByteStrings entry must be an object"))?;
        require_exact_fields(overlay, &["Path", "Bytes"], &[])?;
        let path = validate_path(
            overlay
                .get("Path")
                .ok_or_else(|| invalid("missing ByteStrings Path"))?,
        )?;
        let bytes = byte_vec(
            overlay
                .get("Bytes")
                .ok_or_else(|| invalid("missing ByteStrings Bytes"))?,
            Some(MAX_BYTE_OVERLAY_LEN),
        )?;
        let mut cursor = root.clone();
        for seg in &path {
            let next = match seg {
                PathSeg::Key(key) => cursor.lookup_exact_key(key.as_bytes()),
                PathSeg::Index(idx) => cursor.lookup_exact_index(*idx),
            };
            cursor = next.ok_or_else(|| invalid("ByteStrings overlay path target missing"))?;
        }
        cursor.replace_with_input_bytes(bytes)?;
    }
    Ok(())
}

fn validate_hash_trace(obj: &Map<String, Value>) -> Result<ValidatedOp> {
    require_exact_fields(obj, &["Kind", "Actions"], &[])?;
    let actions_arr = obj
        .get("Actions")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("jinja-hash-trace Actions must be an array"))?;
    if actions_arr.len() > MAX_HASH_ACTIONS {
        return Err(invalid("jinja-hash-trace Actions exceeds 2048"));
    }
    let mut total_bytes = 0_usize;
    let mut actions = Vec::with_capacity(actions_arr.len());
    for action_val in actions_arr {
        let action_obj = action_val
            .as_object()
            .filter(|map| map.len() == 1)
            .ok_or_else(|| invalid("hash action must be a single-field object"))?;
        if let Some(update_val) = action_obj.get("Update") {
            let bytes = byte_vec(update_val, None)?;
            total_bytes += bytes.len();
            if total_bytes > MAX_HASH_BYTES {
                return Err(invalid("jinja-hash-trace input exceeds 256 KiB"));
            }
            actions.push(HashTraceAction::Update(bytes));
        } else if action_obj.get("Digest") == Some(&Value::Bool(true)) {
            actions.push(HashTraceAction::Digest);
        } else {
            return Err(invalid("hash action must contain Update or Digest: true"));
        }
    }
    Ok(ValidatedOp::HashTrace(actions))
}

fn validate_hash_chunks(obj: &Map<String, Value>) -> Result<ValidatedOp> {
    require_exact_fields(obj, &["Kind", "Chunks"], &[])?;
    let chunks_arr = obj
        .get("Chunks")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("jinja-hash Chunks must be an array"))?;
    if chunks_arr.len() > MAX_HASH_CHUNKS {
        return Err(invalid("jinja-hash Chunks exceeds 1024"));
    }
    let mut total_bytes = 0_usize;
    let mut chunks = Vec::with_capacity(chunks_arr.len());
    for chunk_val in chunks_arr {
        let bytes = byte_vec(chunk_val, None)?;
        total_bytes += bytes.len();
        if total_bytes > MAX_HASH_BYTES {
            return Err(invalid("jinja-hash input exceeds 256 KiB"));
        }
        chunks.push(bytes);
    }
    Ok(ValidatedOp::Hash(chunks))
}

fn validate_operation(op_val: &Value) -> Result<ValidatedOp> {
    let obj = op_val
        .as_object()
        .ok_or_else(|| invalid("Jinja operation must be an object"))?;
    let kind = obj
        .get("Kind")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("Jinja operation Kind must be a string"))?;
    match kind {
        "jinja-hash-trace" => validate_hash_trace(obj),
        "jinja-hash" => validate_hash_chunks(obj),
        "jinja-builtins" => {
            require_exact_fields(obj, &["Kind"], &[])?;
            Ok(ValidatedOp::Builtins)
        }
        "jinja-caps" => {
            require_exact_fields(obj, &["Kind", "Template"], &[])?;
            Ok(ValidatedOp::Caps {
                template: validate_template(obj)?,
            })
        }
        "jinja-render" => {
            require_exact_fields(
                obj,
                &["Kind", "Template"],
                &["Variables", "VariablesJson", "ByteStrings"],
            )?;
            let template = validate_template(obj)?;
            let variables = validate_variables(obj)?;
            apply_byte_overlays(obj, &variables)?;
            Ok(ValidatedOp::Render {
                template,
                variables,
            })
        }
        "jinja-stats" => {
            require_exact_fields(
                obj,
                &["Kind", "Template", "Paths"],
                &["Variables", "VariablesJson"],
            )?;
            let template = validate_template(obj)?;
            let paths_arr = obj
                .get("Paths")
                .and_then(Value::as_array)
                .ok_or_else(|| invalid("jinja-stats Paths must be an array"))?;
            if paths_arr.len() > MAX_STATS_PATHS {
                return Err(invalid("jinja-stats Paths exceeds 1024"));
            }
            let paths = paths_arr
                .iter()
                .map(validate_path)
                .collect::<Result<Vec<_>>>()?;
            let variables = validate_variables(obj)?;
            Ok(ValidatedOp::Stats {
                template,
                variables,
                paths,
            })
        }
        "jinja-string-parts" => {
            require_exact_fields(obj, &["Kind", "Template"], &["Variables", "VariablesJson"])?;
            let template = validate_template(obj)?;
            let variables = validate_variables(obj)?;
            Ok(ValidatedOp::StringParts {
                template,
                variables,
            })
        }
        _ => Err(invalid("unknown Jinja operation Kind")),
    }
}

struct UtcCalendar {
    year: i64,
    month: i64,
    day: i64,
    hour: i64,
    minute: i64,
    second: i64,
    wday: i64,
    month_idx: usize,
    wday_idx: usize,
    yday: i64,
}

fn utc_calendar_from_unix(unix_secs: i64) -> JResult<UtcCalendar> {
    const CUM_DAYS: [i64; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    if !(-62_167_219_200..=253_402_300_799).contains(&unix_secs) {
        return Err(JinjaError::std("strftime_now timestamp outside UTC range"));
    }
    let days = unix_secs.div_euclid(86_400);
    let secs_of_day = unix_secs.rem_euclid(86_400);
    let hour = secs_of_day / 3600;
    let minute = (secs_of_day % 3600) / 60;
    let second = secs_of_day % 60;
    let wday = (days + 4).rem_euclid(7);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    let year = y + i64::from(month <= 2);
    let is_leap = (year % 4 == 0) && (year % 100 != 0 || year % 400 == 0);
    let month_idx = usize::try_from(month - 1).map_err(|_| JinjaError::std("invalid UTC month"))?;
    let wday_idx = usize::try_from(wday).map_err(|_| JinjaError::std("invalid UTC weekday"))?;
    let yday = CUM_DAYS[month_idx] + i64::from(is_leap && month > 2) + day;
    Ok(UtcCalendar {
        year,
        month,
        day,
        hour,
        minute,
        second,
        wday,
        month_idx,
        wday_idx,
        yday,
    })
}

fn format_strftime_utc(unix_secs: i64, fmt: &str) -> JResult<String> {
    use std::fmt::Write as _;

    const MONTH_ABBR: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    const MONTH_FULL: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    const WDAY_ABBR: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    const WDAY_FULL: [&str; 7] = [
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
    ];

    let cal = utc_calendar_from_unix(unix_secs)?;
    let (year, month, day) = (cal.year, cal.month, cal.day);
    let (hour, minute, second) = (cal.hour, cal.minute, cal.second);
    let (wday, month_idx, wday_idx, yday) = (cal.wday, cal.month_idx, cal.wday_idx, cal.yday);

    let mut out = String::with_capacity(fmt.len());
    let mut chars = fmt.chars();
    while let Some(ch) = chars.next() {
        if ch != '%' {
            out.push(ch);
            continue;
        }
        let Some(spec) = chars.next() else {
            out.push('%');
            break;
        };
        let write_res = match spec {
            'Y' => write!(out, "{year:04}"),
            'y' => write!(out, "{:02}", year.rem_euclid(100)),
            'm' => write!(out, "{month:02}"),
            'd' => write!(out, "{day:02}"),
            'e' => write!(out, "{day:2}"),
            'H' => write!(out, "{hour:02}"),
            'I' => write!(
                out,
                "{:02}",
                match hour % 12 {
                    0 => 12,
                    val => val,
                }
            ),
            'M' => write!(out, "{minute:02}"),
            'S' => write!(out, "{second:02}"),
            'p' => out.write_str(if hour < 12 { "AM" } else { "PM" }),
            'P' => out.write_str(if hour < 12 { "am" } else { "pm" }),
            'b' | 'h' => out.write_str(MONTH_ABBR[month_idx]),
            'B' => out.write_str(MONTH_FULL[month_idx]),
            'a' => out.write_str(WDAY_ABBR[wday_idx]),
            'A' => out.write_str(WDAY_FULL[wday_idx]),
            'w' => write!(out, "{wday}"),
            'u' => write!(out, "{}", if wday == 0 { 7 } else { wday }),
            'j' => write!(out, "{yday:03}"),
            'F' => write!(out, "{year:04}-{month:02}-{day:02}"),
            'T' => write!(out, "{hour:02}:{minute:02}:{second:02}"),
            'R' => write!(out, "{hour:02}:{minute:02}"),
            'D' => write!(out, "{month:02}/{day:02}/{:02}", year.rem_euclid(100)),
            'z' => out.write_str("+0000"),
            'Z' => out.write_str("UTC"),
            'n' => out.write_str("\n"),
            't' => out.write_str("\t"),
            '%' => out.write_str("%"),
            's' => write!(out, "{unix_secs}"),
            other => {
                out.push('%');
                out.push(other);
                Ok(())
            }
        };
        write_res.map_err(|_| JinjaError::std("strftime_now formatting failed"))?;
    }
    Ok(out)
}

pub(super) fn eval_program_with_now(
    program: &Program,
    globals: &JVal,
    now_unix_seconds: Option<i64>,
) -> JResult<Vec<TextSpan>> {
    let mut evaluator = Evaluator::new_with_now(globals, now_unix_seconds);
    let mut out = Vec::new();
    evaluator.exec_block(&program.0, &mut out)?;
    Ok(out)
}

pub(super) fn eval_program_chat_fallback(
    program: &Program,
    globals: &JVal,
    now_unix_seconds: Option<i64>,
) -> JResult<Vec<TextSpan>> {
    let mut evaluator = Evaluator::new_with_now(globals, now_unix_seconds);
    evaluator.allow_undef_str_add = true;
    let mut out = Vec::new();
    evaluator.exec_block(&program.0, &mut out)?;
    Ok(out)
}

pub(super) fn eval_program_with_context(
    program: &Program,
    globals: &JVal,
    now_unix_seconds: Option<i64>,
) -> (JResult<Vec<TextSpan>>, BTreeMap<String, JVal>) {
    let mut evaluator = Evaluator::new_with_now(globals, now_unix_seconds);
    let mut out = Vec::new();
    let res = evaluator.exec_block(&program.0, &mut out).map(|_| out);
    let root = evaluator.scopes.into_iter().next().unwrap_or_default();
    (res, root)
}

pub(super) fn eval_program(program: &Program, globals: &JVal) -> JResult<Vec<TextSpan>> {
    eval_program_with_now(program, globals, None)
}

pub(super) fn execute_caps_flags(
    program: &Program,
    now_unix_seconds: Option<i64>,
) -> JResult<(bool, bool)> {
    const CAPS_PROBE_JSON: &str = r#"{"messages":[{"role":"user","content":"STRING_MARKER"}],"tools":[],"bos_token":"","eos_token":"","add_generation_prompt":true}"#;
    let probe_vars =
        parse_variables_json(CAPS_PROBE_JSON).map_err(|err| JinjaError::std(err.to_string()))?;
    let mut rendered_bytes = Vec::new();
    let succeeded =
        eval_program_with_now(program, &probe_vars, now_unix_seconds).is_ok_and(|spans| {
            for span in spans {
                rendered_bytes.extend(span.bytes);
            }
            true
        });
    let typed = probe_vars
        .lookup_exact_key(b"messages")
        .and_then(|msgs| msgs.lookup_exact_index(0))
        .and_then(|msg| msg.lookup_exact_key(b"content"))
        .is_some_and(|content| content.is_array_access());
    let has_marker = rendered_bytes
        .windows(b"STRING_MARKER".len())
        .any(|win| win == b"STRING_MARKER");
    let string_supported = succeeded && (!typed || has_marker);
    Ok((string_supported, typed))
}

/// Full 9-field capabilities extracted from a parsed Jinja chat template.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "Mirrors the nine boolean capability flags of upstream jinja::caps."
)]
pub(super) struct JinjaCaps {
    pub(super) supports_tools: bool,
    pub(super) supports_tool_calls: bool,
    pub(super) supports_system_role: bool,
    pub(super) supports_parallel_tool_calls: bool,
    pub(super) supports_preserve_reasoning: bool,
    pub(super) supports_reasoning_effort: bool,
    pub(super) supports_string_content: bool,
    pub(super) supports_typed_content: bool,
    pub(super) supports_object_arguments: bool,
}

fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && haystack.windows(needle.len()).any(|win| win == needle)
}

fn eval_probe_bytes(program: &Program, globals: &JVal, now_unix_seconds: i64) -> Option<Vec<u8>> {
    let spans = eval_program_with_now(program, globals, Some(now_unix_seconds)).ok()?;
    Some(spans.into_iter().flat_map(|span| span.bytes).collect())
}

fn make_probe_msg(role: &str, content: JVal) -> JVal {
    JVal::object(vec![
        (
            JVal::text_str("role"),
            JVal::bytes(role.as_bytes().to_vec(), true),
        ),
        (JVal::text_str("content"), content),
    ])
}

fn make_typed_text(text: &str) -> JVal {
    if text.is_empty() {
        JVal::array(Vec::new())
    } else {
        JVal::array(vec![JVal::object(vec![
            (JVal::text_str("type"), JVal::bytes(b"text".to_vec(), true)),
            (
                JVal::text_str("text"),
                JVal::bytes(text.as_bytes().to_vec(), true),
            ),
        ])])
    }
}

fn make_probe_globals(
    messages: JVal,
    tools: Option<JVal>,
    add_generation_prompt: bool,
    enable_thinking: bool,
    extra: &[(&str, JVal)],
) -> JVal {
    let mut pairs = vec![
        (JVal::text_str("messages"), messages),
        (JVal::text_str("bos_token"), JVal::bytes(Vec::new(), true)),
        (JVal::text_str("eos_token"), JVal::bytes(Vec::new(), true)),
        (
            JVal::text_str("add_generation_prompt"),
            JVal::boolean(add_generation_prompt),
        ),
        (
            JVal::text_str("enable_thinking"),
            JVal::boolean(enable_thinking),
        ),
    ];
    if let Some(tools_val) = tools {
        pairs.push((JVal::text_str("tools"), tools_val));
    }
    for (k, v) in extra {
        pairs.push((JVal::text_str(*k), v.clone()));
    }
    JVal::object(pairs)
}

fn make_tool_calls_probe(calls: &[(&str, &str, JVal)], assistant_content: JVal) -> JVal {
    let call_vals: Vec<JVal> = calls
        .iter()
        .map(|(id, name, args)| {
            JVal::object(vec![
                (
                    JVal::text_str("id"),
                    JVal::bytes(id.as_bytes().to_vec(), true),
                ),
                (
                    JVal::text_str("type"),
                    JVal::bytes(b"function".to_vec(), true),
                ),
                (
                    JVal::text_str("function"),
                    JVal::object(vec![
                        (
                            JVal::text_str("name"),
                            JVal::bytes(name.as_bytes().to_vec(), true),
                        ),
                        (JVal::text_str("arguments"), args.clone()),
                    ]),
                ),
            ])
        })
        .collect();
    JVal::object(vec![
        (
            JVal::text_str("role"),
            JVal::bytes(b"assistant".to_vec(), true),
        ),
        (JVal::text_str("content"), assistant_content),
        (JVal::text_str("tool_calls"), JVal::array(call_vals)),
    ])
}

fn probe_tool_caps(
    program: &Program,
    now_unix_seconds: i64,
    msg_content: &dyn Fn(&str) -> JVal,
) -> (bool, bool, bool, bool) {
    let tools_json = r#"[{"type":"function","function":{"name":"caps_tool_alpha","description":"Alpha tool","parameters":{"type":"object","properties":{"arg":{"type":"string"}}}}}]"#;
    let tools_val =
        parse_json_jval_bounded(tools_json, 65_536, 16).unwrap_or_else(|_| JVal::array(Vec::new()));
    let tools_globals = make_probe_globals(
        JVal::array(vec![make_probe_msg("user", msg_content("hello"))]),
        Some(tools_val.clone()),
        true,
        true,
        &[],
    );
    let supports_tools = eval_probe_bytes(program, &tools_globals, now_unix_seconds)
        .is_some_and(|bytes| contains_bytes(&bytes, b"caps_tool_alpha"));

    let obj_args = JVal::object(vec![(
        JVal::text_str("arg_k"),
        JVal::bytes(b"arg_v".to_vec(), true),
    )]);
    let str_args = JVal::bytes(br#"{"arg_k":"arg_v"}"#.to_vec(), true);

    let mut supports_tool_calls = false;
    let mut supports_object_arguments = false;
    let mut working_asst_empty = JVal::none();
    let mut working_args_is_obj = true;

    let empty_candidates = [JVal::none(), msg_content("")];
    'probe_tc: for (is_obj, args) in [(true, &obj_args), (false, &str_args)] {
        for empty_content in &empty_candidates {
            let msgs = JVal::array(vec![
                make_probe_msg("user", msg_content("hi")),
                make_tool_calls_probe(
                    &[("call00001", "tc_probe_one", args.clone())],
                    empty_content.clone(),
                ),
            ]);
            let globals = make_probe_globals(msgs, Some(tools_val.clone()), false, true, &[]);
            if let Some(bytes) = eval_probe_bytes(program, &globals, now_unix_seconds)
                && contains_bytes(&bytes, b"tc_probe_one")
            {
                supports_tool_calls = true;
                working_asst_empty = empty_content.clone();
                working_args_is_obj = is_obj;
                if is_obj {
                    supports_object_arguments = contains_bytes(&bytes, b"arg_k")
                        && contains_bytes(&bytes, b"arg_v")
                        && !contains_bytes(&bytes, br#"\"arg_k\""#);
                }
                break 'probe_tc;
            }
        }
    }

    let supports_parallel_tool_calls = if supports_tool_calls {
        let arg = if working_args_is_obj {
            obj_args
        } else {
            str_args
        };
        let msgs = JVal::array(vec![
            make_probe_msg("user", msg_content("hi")),
            make_tool_calls_probe(
                &[
                    ("call00001", "tc_probe_one", arg.clone()),
                    ("call00002", "tc_probe_two", arg),
                ],
                working_asst_empty,
            ),
        ]);
        let globals = make_probe_globals(msgs, Some(tools_val), false, true, &[]);
        eval_probe_bytes(program, &globals, now_unix_seconds).is_some_and(|bytes| {
            contains_bytes(&bytes, b"tc_probe_one") && contains_bytes(&bytes, b"tc_probe_two")
        })
    } else {
        false
    };

    (
        supports_tools,
        supports_tool_calls,
        supports_parallel_tool_calls,
        supports_object_arguments,
    )
}

fn probe_reasoning_caps(
    program: &Program,
    now_unix_seconds: i64,
    msg_content: &dyn Fn(&str) -> JVal,
) -> (bool, bool) {
    let reason_asst = JVal::object(vec![
        (
            JVal::text_str("role"),
            JVal::bytes(b"assistant".to_vec(), true),
        ),
        (JVal::text_str("content"), msg_content("answer")),
        (
            JVal::text_str("reasoning_content"),
            JVal::bytes(b"REASON_CAPS_MARKER".to_vec(), true),
        ),
    ]);
    let preserve_globals = make_probe_globals(
        JVal::array(vec![
            make_probe_msg("user", msg_content("hi")),
            reason_asst,
            make_probe_msg("user", msg_content("next")),
        ]),
        None,
        false,
        true,
        &[],
    );
    let supports_preserve_reasoning =
        eval_probe_bytes(program, &preserve_globals, now_unix_seconds)
            .is_some_and(|bytes| contains_bytes(&bytes, b"REASON_CAPS_MARKER"));

    let effort_high_globals = make_probe_globals(
        JVal::array(vec![make_probe_msg("user", msg_content("hi"))]),
        None,
        true,
        true,
        &[("reasoning_effort", JVal::bytes(b"high".to_vec(), true))],
    );
    let effort_low_globals = make_probe_globals(
        JVal::array(vec![make_probe_msg("user", msg_content("hi"))]),
        None,
        true,
        true,
        &[("reasoning_effort", JVal::bytes(b"low".to_vec(), true))],
    );
    let (high_res, high_scope) =
        eval_program_with_context(program, &effort_high_globals, Some(now_unix_seconds));
    let low_res = eval_probe_bytes(program, &effort_low_globals, now_unix_seconds);
    let high_bytes = high_res
        .ok()
        .map(|spans| spans.into_iter().flat_map(|s| s.bytes).collect::<Vec<u8>>());
    let supports_reasoning_effort = high_scope
        .get("reasoning_effort")
        .is_some_and(JVal::is_used)
        || (high_bytes.is_some() && low_res.is_some() && high_bytes != low_res);
    (supports_preserve_reasoning, supports_reasoning_effort)
}

/// Probes all nine capabilities of a parsed Jinja chat template.
pub(super) fn detect_jinja_caps(program: &Program, now_unix_seconds: i64) -> JinjaCaps {
    let (flag_str, flag_typed) =
        execute_caps_flags(program, Some(now_unix_seconds)).unwrap_or((false, false));

    let str_msg = make_probe_msg("user", JVal::bytes(b"STRING_MARKER".to_vec(), true));
    let str_globals = make_probe_globals(
        JVal::array(vec![str_msg]),
        Some(JVal::array(Vec::new())),
        true,
        true,
        &[],
    );
    let str_bytes = eval_probe_bytes(program, &str_globals, now_unix_seconds);
    let str_has_marker = str_bytes
        .as_deref()
        .is_some_and(|bytes| contains_bytes(bytes, b"STRING_MARKER"));
    let supports_string_content = flag_str && str_has_marker;

    let typed_msg = make_probe_msg("user", make_typed_text("TYPED_MARKER"));
    let typed_globals = make_probe_globals(
        JVal::array(vec![typed_msg]),
        Some(JVal::array(Vec::new())),
        true,
        true,
        &[],
    );
    let typed_bytes = eval_probe_bytes(program, &typed_globals, now_unix_seconds);
    let typed_rendered = typed_bytes.as_deref().is_some_and(|bytes| {
        contains_bytes(bytes, b"TYPED_MARKER") && !contains_bytes(bytes, b"'type': 'text'")
    });
    let supports_typed_content = flag_typed || typed_rendered;
    let use_typed = !supports_string_content && supports_typed_content;
    let msg_content = |text: &str| -> JVal {
        if use_typed {
            make_typed_text(text)
        } else {
            JVal::bytes(text.as_bytes().to_vec(), true)
        }
    };

    let sys_globals = make_probe_globals(
        JVal::array(vec![
            make_probe_msg("system", msg_content("SYS_CAPS_MARKER")),
            make_probe_msg("user", msg_content("USR_CAPS_MARKER")),
        ]),
        None,
        true,
        true,
        &[],
    );
    let supports_system_role = eval_probe_bytes(program, &sys_globals, now_unix_seconds)
        .is_some_and(|bytes| contains_bytes(&bytes, b"SYS_CAPS_MARKER"));

    let (
        supports_tools,
        supports_tool_calls,
        supports_parallel_tool_calls,
        supports_object_arguments,
    ) = probe_tool_caps(program, now_unix_seconds, &msg_content);

    let (supports_preserve_reasoning, supports_reasoning_effort) =
        probe_reasoning_caps(program, now_unix_seconds, &msg_content);

    JinjaCaps {
        supports_tools,
        supports_tool_calls,
        supports_system_role,
        supports_parallel_tool_calls,
        supports_preserve_reasoning,
        supports_reasoning_effort,
        supports_string_content,
        supports_typed_content,
        supports_object_arguments,
    }
}

fn execute_caps(template: &str) -> JResult<Value> {
    let program = parse_template(template)?;
    let (string_supported, typed) = execute_caps_flags(&program, None)?;
    Ok(json!({
        "SupportsStringContent": string_supported,
        "SupportsTypedContent": typed,
    }))
}

fn execute_op_unchecked(op: ValidatedOp, runtime_phase: &mut bool) -> JResult<Value> {
    match op {
        ValidatedOp::HashTrace(actions) => {
            let mut hasher = BlockHasher::new();
            let mut digests = Vec::new();
            for action in actions {
                match action {
                    HashTraceAction::Update(bytes) => hasher.update(&bytes),
                    HashTraceAction::Digest => digests.push(hasher.digest()),
                }
            }
            Ok(json!({"Digests": digests}))
        }
        ValidatedOp::Hash(chunks) => {
            let mut hasher = BlockHasher::new();
            for chunk in chunks {
                hasher.update(&chunk);
            }
            Ok(json!({"Digest": hasher.digest()}))
        }
        ValidatedOp::Builtins => {
            let mut entries = Vec::new();
            for &(group, names) in BUILTIN_INVENTORY {
                for &name in names {
                    entries.push(json!({"Type": group, "Name": name}));
                }
            }
            Ok(json!({"Builtins": entries}))
        }
        ValidatedOp::Caps { template } => execute_caps(&template),
        ValidatedOp::Render {
            template,
            variables,
        } => {
            let program = parse_template(&template)?;
            let fresh_vars = variables.deep_clone_fresh();
            *runtime_phase = true;
            let spans = eval_program(&program, &fresh_vars)?;
            let raw: Vec<u8> = spans.into_iter().flat_map(|span| span.bytes).collect();
            let rendered = String::from_utf8(raw).map_err(|_| {
                JinjaError::std("rendered Jinja output cannot be JSON-encoded as UTF-8")
            })?;
            Ok(json!({"Rendered": rendered}))
        }
        ValidatedOp::Stats {
            template,
            variables,
            paths,
        } => {
            let program = parse_template(&template)?;
            let fresh_vars = variables.deep_clone_fresh();
            *runtime_phase = true;
            let _ = eval_program(&program, &fresh_vars)?;
            let used = paths
                .into_iter()
                .map(|path| {
                    let mut cursor = fresh_vars.clone();
                    for seg in path {
                        let key = match seg {
                            PathSeg::Key(name) => JVal::text_str(name),
                            PathSeg::Index(idx) => {
                                i64::try_from(idx).map_or_else(|_| JVal::undefined(), JVal::int)
                            }
                        };
                        cursor = cursor.get_member(&key, false);
                    }
                    cursor.is_used()
                })
                .collect::<Vec<_>>();
            Ok(json!({"Used": used}))
        }
        ValidatedOp::StringParts {
            template,
            variables,
        } => {
            let program = parse_template(&template)?;
            let fresh_vars = variables.deep_clone_fresh();
            *runtime_phase = true;
            let spans = merge_spans(eval_program(&program, &fresh_vars)?);
            let mut parts = Vec::with_capacity(spans.len());
            for span in spans {
                let val_str = String::from_utf8(span.bytes).map_err(|_| {
                    JinjaError::std("Jinja string part cannot be JSON-encoded as UTF-8")
                })?;
                parts.push(json!({"Value": val_str, "IsInput": span.is_input}));
            }
            Ok(json!({"Parts": parts}))
        }
    }
}

fn execute_op(op: ValidatedOp) -> Value {
    let mut runtime_phase = false;
    match execute_op_unchecked(op, &mut runtime_phase) {
        Ok(value) => value,
        Err(err) => err.into_wire(runtime_phase),
    }
}

pub(super) fn handles(request: &Value) -> bool {
    request["Operations"].as_array().is_some_and(|operations| {
        !operations.is_empty()
            && operations.iter().all(|operation| {
                matches!(
                    operation["Kind"].as_str(),
                    Some(
                        "jinja-render"
                            | "jinja-hash"
                            | "jinja-hash-trace"
                            | "jinja-stats"
                            | "jinja-caps"
                            | "jinja-string-parts"
                            | "jinja-builtins"
                    )
                )
            })
    })
}

pub(super) fn execute(request: &Value) -> Result<Value> {
    let obj = request
        .as_object()
        .ok_or_else(|| invalid("Jinja request must be a JSON object"))?;
    require_exact_fields(obj, &["SchemaVersion", "Operations"], &[])?;
    let version = obj
        .get("SchemaVersion")
        .ok_or_else(|| invalid("missing SchemaVersion"))?;
    if nonneg_int(version, u64::MAX)? != 1 {
        return Err(invalid("unsupported Jinja SchemaVersion"));
    }
    let ops_arr = obj
        .get("Operations")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("Jinja Operations must be an array"))?;
    if ops_arr.is_empty() || ops_arr.len() > MAX_OPERATIONS {
        return Err(invalid(
            "Jinja Operations must contain 1 through 128 operations",
        ));
    }
    let validated = ops_arr
        .iter()
        .map(validate_operation)
        .collect::<Result<Vec<_>>>()?;
    let results = validated.into_iter().map(execute_op).collect::<Vec<_>>();
    Ok(json!({"SchemaVersion": 1, "Results": results}))
}

pub(super) fn identity() -> Value {
    json!({
        "Kind": "independent-rust-jinja",
        "SupportedSourceIds": [ID],
        "UpstreamCommit": super::UPSTREAM,
        "AssertionSource": {"Path": ID, "Sha256": SOURCE_SHA},
        "Architecture": "clean-room-jinja-lexer-parser-ast-evaluator-hasher-and-caps-v1",
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

    fn wrap(op: &Value) -> Value {
        json!({"SchemaVersion": 1, "Operations": [op]})
    }

    #[test]
    fn independent_jinja_provider_passes_complete_test_jinja_suite_across_reference_arms()
    -> Result<()> {
        for target in [Target::RustReference, Target::Cpp] {
            let engine = Engine::new(target, None, None, None, None, None)?;
            let report = crate::jinja::run(&engine)?;
            assert_eq!(report["Id"], ID);
            assert_eq!(report["Status"], "pass", "{report}");
            assert!(
                report["Detail"]
                    .as_str()
                    .is_some_and(|detail| detail.contains("278 literal renders")
                        && detail.contains("original_not_implemented_skips=0;")
                        && detail.contains("failed=0;")),
                "{report}"
            );
        }
        assert!(crate::reference_provider::supports(ID));
        let info = identity();
        assert_eq!(info["SupportedSourceIds"], json!([ID]));
        assert_eq!(info["AssertionSource"]["Sha256"], SOURCE_SHA);
        assert_eq!(info["NativeReferenceReplay"], false);
        assert_eq!(info["ProductionRuntimeCalled"], false);
        assert_eq!(info["CompleteSourceGate"], false);
        assert_eq!(info["NativeRetirementQualified"], false);
        Ok(())
    }

    #[test]
    fn independent_jinja_exception_phase_and_byte_strings_behave_per_protocol() -> Result<()> {
        for bad_tpl in [
            "{{ unclosed",
            "{{ é }}",
            "{% if true %}",
            "{% unknown_tag %}",
            &format!("{{{{ {}1 }}}}", "-".repeat(200)),
        ] {
            let resp = execute(&wrap(&json!({
                "Kind": "jinja-render",
                "Template": bad_tpl,
                "VariablesJson": "{}"
            })))?;
            assert_eq!(resp["Results"][0]["ExceptionKind"], "standard", "{bad_tpl}");
        }
        for err_tpl in [
            "{{ 1 / 0 }}",
            "{{ 1 // 0 }}",
            "{{ raise_exception('boom') }}",
            "{% macro f() %}{{ f() }}{% endmacro %}{{ f() }}",
            "{% set x = [] %}{% set _ = x.append(x) %}",
            "{{ range(0, 100000) }}",
        ] {
            let resp = execute(&wrap(&json!({
                "Kind": "jinja-render",
                "Template": err_tpl,
                "VariablesJson": "{}"
            })))?;
            assert_eq!(resp["Results"][0]["ExceptionKind"], "standard", "{err_tpl}");
        }
        let unimpl_resp = execute(&wrap(&json!({
            "Kind": "jinja-render",
            "Template": "{{ strftime_now('%Y') }}",
            "VariablesJson": "{}"
        })))?;
        assert_eq!(
            unimpl_resp["Results"][0]["ExceptionKind"],
            "not-implemented"
        );
        let non_utf8 = execute(&wrap(&json!({
            "Kind": "jinja-render",
            "Template": "{{ data }}",
            "VariablesJson": r#"{"data":""}"#,
            "ByteStrings": [{"Path": ["data"], "Bytes": [255, 128]}]
        })))?;
        assert_eq!(non_utf8["Results"][0]["ExceptionKind"], "standard");
        let valid_overlay = execute(&wrap(&json!({
            "Kind": "jinja-render",
            "Template": "{{ nested[0] }}:{{ extra }}",
            "Variables": {"nested": ["old"], "extra": 7},
            "ByteStrings": [{"Path": ["nested", 0], "Bytes": [111, 107]}]
        })))?;
        assert_eq!(valid_overlay["Results"][0]["Rendered"], "ok:7");
        Ok(())
    }

    #[test]
    fn independent_jinja_control_flow_stats_caps_and_hash_trace_behave_per_protocol() -> Result<()>
    {
        let control_flow = execute(&wrap(&json!({
            "Kind": "jinja-render",
            "Template": "{% generation %}{% set block_v %}{% filter upper %}hi{% endfilter %}{% endset %}{{ block_v }}:{% for i in range(5) %}{% if i == 1 %}{% continue %}{% endif %}{% if i == 3 %}{% break %}{% endif %}{{ i }}{% endfor %}:{{ -7 // 2 }}{% endgeneration %}",
            "Variables": {}
        })))?;
        assert_eq!(control_flow["Results"][0]["Rendered"], "HI:02:-4");

        let ordered = execute(&wrap(&json!({
            "Kind": "jinja-render",
            "Template": "{{ obj|string }}:{{ a + b }}:{{ f }}",
            "VariablesJson": r#"{"obj":{"z":1,"a":2},"a":9007199254740992,"b":1,"f":1.0}"#
        })))?;
        assert_eq!(
            ordered["Results"][0]["Rendered"],
            "{'z': 1, 'a': 2}:9007199254740993:1.0"
        );

        let stats_missing = execute(&wrap(&json!({
            "Kind": "jinja-stats",
            "Template": "{{ known }}",
            "Variables": {"known": 1},
            "Paths": [["known"], ["missing"], ["known", 0]]
        })))?;
        assert_eq!(
            stats_missing["Results"][0]["Used"],
            json!([true, false, false])
        );

        let caps_err = execute(&wrap(&json!({
            "Kind": "jinja-caps",
            "Template": "{{ raise_exception('unsupported') }}"
        })))?;
        assert_eq!(
            caps_err["Results"][0],
            json!({"SupportsStringContent": false, "SupportsTypedContent": false})
        );

        let caps_parse_err = execute(&wrap(&json!({
            "Kind": "jinja-caps",
            "Template": "{{ unclosed"
        })))?;
        assert_eq!(caps_parse_err["Results"][0]["ExceptionKind"], "standard");

        let hash_trace = execute(&wrap(&json!({
            "Kind": "jinja-hash-trace",
            "Actions": [
                {"Update": [97, 98, 99]},
                {"Digest": true},
                {"Update": []},
                {"Digest": true},
                {"Update": [100]},
                {"Digest": true}
            ]
        })))?;
        let digests = hash_trace["Results"][0]["Digests"]
            .as_array()
            .ok_or_else(|| invalid("missing hash trace digests"))?;
        assert_eq!(digests.len(), 3);
        assert_eq!(digests[0], digests[1]);
        assert_ne!(digests[1], digests[2]);

        let exact_hash = execute(&wrap(&json!({
            "Kind": "jinja-hash",
            "Chunks": [vec![1_u8; 256 * 1024]]
        })))?;
        assert!(exact_hash["Results"][0]["Digest"].is_u64());

        let loop_neighbors = execute(&wrap(&json!({
            "Kind": "jinja-render",
            "Template": "{% for x in arr %}[{{ loop.previtem if loop.previtem is defined else 'NONE' }},{{ x }},{{ loop.nextitem if loop.nextitem is defined else 'NONE' }}]{% endfor %}",
            "Variables": {"arr": ["a", "b", "c"]}
        })))?;
        assert_eq!(
            loop_neighbors["Results"][0]["Rendered"],
            "[NONE,a,b][a,b,c][b,c,NONE]"
        );

        let parsed_json = execute(&wrap(&json!({
            "Kind": "jinja-render",
            "Template": "{% set d = raw | from_json %}{{ d.k }}:{{ d.nums[1] }}",
            "Variables": {"raw": r#"{"k":"ok","nums":[10,20]}"#}
        })))?;
        assert_eq!(parsed_json["Results"][0]["Rendered"], "ok:20");

        let bad_from_json = execute(&wrap(&json!({
            "Kind": "jinja-render",
            "Template": "{{ '{bad' | from_json }}",
            "Variables": {}
        })))?;
        assert_eq!(bad_from_json["Results"][0]["ExceptionKind"], "standard");

        let caps_prog = parse_template(
            "{% if reasoning_effort is defined %}[{{ reasoning_effort }}]{% endif %}{% for m in messages %}{{ m.role }}:{% if m.content is string %}{{ m.content }}{% elif m.content is iterable %}{% for p in m.content %}{{ p.text }}{% endfor %}{% endif %}{% if m.reasoning_content %}<think>{{ m.reasoning_content }}</think>{% endif %}{% if m.tool_calls %}{% for tc in m.tool_calls %}{{ tc.function.name }}({% for k, v in tc.function.arguments|items %}{{ k }}={{ v }}{% endfor %}){% endfor %}{% endif %}{% endfor %}{% if tools %}{{ tools[0].function.name }}{% endif %}",
        )
        .map_err(|err| invalid(err.message()))?;
        let caps = detect_jinja_caps(&caps_prog, 0);
        assert!(caps.supports_tools);
        assert!(caps.supports_tool_calls);
        assert!(caps.supports_system_role);
        assert!(caps.supports_parallel_tool_calls);
        assert!(caps.supports_preserve_reasoning);
        assert!(caps.supports_reasoning_effort);
        assert!(caps.supports_string_content);
        assert!(caps.supports_typed_content);
        assert!(caps.supports_object_arguments);
        Ok(())
    }

    #[test]
    fn independent_jinja_rejects_invalid_wire_framing_and_variables_bounds() {
        assert!(!handles(&json!({"Operations": []})));
        assert!(!handles(&wrap(&json!({"Kind": "gguf-inventory"}))));
        assert!(!handles(&json!({
            "SchemaVersion": 1,
            "Operations": [{"Kind": "jinja-builtins"}, {"Kind": "gguf-inventory"}]
        })));
        let too_deep_json = format!("{}1{}", "{\"a\":".repeat(34), "}".repeat(34));
        for bad_req in [
            json!({"SchemaVersion": 2, "Operations": [{"Kind": "jinja-builtins"}]}),
            json!({"SchemaVersion": 1.0, "Operations": [{"Kind": "jinja-builtins"}]}),
            json!({"SchemaVersion": true, "Operations": [{"Kind": "jinja-builtins"}]}),
            json!({"SchemaVersion": 1, "Operations": []}),
            json!({"SchemaVersion": 1, "Operations": vec![json!({"Kind": "jinja-builtins"}); 129]}),
            wrap(&json!({"Kind": "jinja-unknown"})),
            wrap(&json!({"Kind": "jinja-builtins", "Extra": 1})),
            wrap(&json!({"Kind": "jinja-caps", "Template": "x".repeat(65_537)})),
            wrap(&json!({"Kind": "jinja-caps", "Template": 123})),
            wrap(&json!({"Kind": "jinja-render", "Template": "x"})),
            wrap(&json!(
                {"Kind": "jinja-render", "Template": "x", "Variables": {}, "VariablesJson": "{}"}
            )),
            wrap(&json!({"Kind": "jinja-render", "Template": "x", "Variables": []})),
            wrap(&json!({"Kind": "jinja-render", "Template": "x", "VariablesJson": "[]"})),
            wrap(&json!(
                {"Kind": "jinja-render", "Template": "x", "VariablesJson": r#"{"a":1,"a":2}"#}
            )),
            wrap(&json!({"Kind": "jinja-render", "Template": "x", "VariablesJson": too_deep_json})),
            wrap(&json!({
                "Kind": "jinja-render",
                "Template": "x",
                "VariablesJson": format!(r#"{{"s":"{}"}}"#, "a".repeat(256 * 1024))
            })),
        ] {
            assert!(execute(&bad_req).is_err(), "{bad_req}");
        }
    }

    #[test]
    fn independent_jinja_rejects_invalid_byte_strings_hash_and_stats_bounds() {
        let too_deep_path: Vec<Value> = std::iter::once(json!("root"))
            .chain((0..32).map(|_| json!("sub")))
            .collect();
        for bad_req in [
            wrap(&json!({
                "Kind": "jinja-render",
                "Template": "{{ a }}",
                "VariablesJson": r#"{"a":1}"#,
                "ByteStrings": [{"Path": ["a"], "Bytes": [65]}]
            })),
            wrap(&json!({
                "Kind": "jinja-render",
                "Template": "{{ a }}",
                "VariablesJson": r#"{"a":"ok"}"#,
                "ByteStrings": [{"Path": ["missing"], "Bytes": [65]}]
            })),
            wrap(&json!({
                "Kind": "jinja-render",
                "Template": "{{ a }}",
                "VariablesJson": r#"{"a":"ok"}"#,
                "ByteStrings": [{"Path": ["a"], "Bytes": [256]}]
            })),
            wrap(&json!({
                "Kind": "jinja-render",
                "Template": "{{ a }}",
                "VariablesJson": r#"{"a":"ok"}"#,
                "ByteStrings": vec![json!({"Path": ["a"], "Bytes": [65]}); 17]
            })),
            wrap(&json!({"Kind": "jinja-hash", "Chunks": [[256]]})),
            wrap(&json!({"Kind": "jinja-hash", "Chunks": [[-1]]})),
            wrap(&json!({"Kind": "jinja-hash", "Chunks": vec![json!([]); 1025]})),
            wrap(&json!({"Kind": "jinja-hash", "Chunks": [vec![1_u8; 256 * 1024 + 1]]})),
            wrap(&json!({"Kind": "jinja-hash-trace", "Actions": [{"Digest": false}]})),
            wrap(&json!({"Kind": "jinja-hash-trace", "Actions": [{"Update": [], "Digest": true}]})),
            wrap(&json!(
                {"Kind": "jinja-hash-trace", "Actions": vec![json!({"Digest": true}); 2049]}
            )),
            wrap(&json!({
                "Kind": "jinja-stats",
                "Template": "x",
                "VariablesJson": "{}",
                "Paths": [[0, "a"]]
            })),
            wrap(&json!({
                "Kind": "jinja-stats",
                "Template": "x",
                "VariablesJson": "{}",
                "Paths": [too_deep_path]
            })),
            wrap(&json!({
                "Kind": "jinja-stats",
                "Template": "x",
                "VariablesJson": "{}",
                "Paths": [["a", 65_536]]
            })),
            wrap(&json!({
                "Kind": "jinja-stats",
                "Template": "x",
                "VariablesJson": "{}",
                "Paths": vec![json!(["a"]); 1025]
            })),
            wrap(&json!({
                "Kind": "jinja-string-parts",
                "Template": "x",
                "VariablesJson": "{}",
                "ByteStrings": []
            })),
        ] {
            assert!(execute(&bad_req).is_err(), "{bad_req}");
        }
    }
}
