//! Clean-room independent safe Rust JSON Schema to GBNF converter and schema info provider.
//!
//! No production `bonsai_prism_runtime` code, native bridge or precomputed
//! answer fixture is called here. Parses schemas with an insertion-order
//! preserving JSON parser, resolves local JSON Pointer `$ref` references,
//! converts JSON Schema constructs to GBNF grammars, and answers
//! `json-schema-info`, `json-schema-to-grammar`, and `json-schema-convert` queries.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use serde_json::{Value, json};

use super::llama_grammar::{array, exact_fields, field};
use super::{Result, invalid};

pub(super) const IDS: &[&str] = &[
    "tests/test-json-schema-to-grammar.cpp",
    "tests/test-grammar-integration.cpp",
];

const SCHEMA_SOURCE_SHA256: &str =
    "409361847bab0707ee02749c4e6978a63f9cc9f119ffff40de8c69b0c948e99d";
const INTEGRATION_SOURCE_SHA256: &str =
    "6acd7735c8f6db1d3869bd2de5ea7a585accc75e3f8627eb163be11bc123716a";

const MAX_OPS: usize = 256;
const MAX_INFO_ACTIONS: usize = 128;
const MAX_SCHEMA_BYTES: usize = 262_144;
const MAX_JSON_DEPTH: usize = 128;
const MAX_SCHEMA_DEPTH: usize = 128;
const MAX_RULES: usize = 65_536;
const MAX_GRAMMAR_BYTES: usize = 4 * 1024 * 1024;
const MAX_PREVIEW_BYTES: usize = 256;

fn bounded_preview(text: &str) -> String {
    if text.len() <= MAX_PREVIEW_BYTES {
        return text.to_owned();
    }
    let mut end = MAX_PREVIEW_BYTES.saturating_sub(3);
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", &text[..end])
}

// Insertion-order preserving JSON value & parser.

/// Insertion-order-preserving JSON DOM value for JSON Schema inspection and conversion.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum JsonVal {
    /// JSON `null`.
    Null,
    /// JSON boolean.
    Bool(bool),
    /// JSON number retaining original lexeme and integer/float representations.
    Number {
        /// Original JSON number lexeme.
        raw: String,
        /// Exact signed 64-bit integer value when integral and in range.
        int_value: Option<i64>,
        /// Finite IEEE 754 `f64` value.
        float_value: f64,
    },
    /// JSON string.
    Str(String),
    /// JSON array.
    Array(Vec<Self>),
    /// JSON object preserving key insertion order.
    Object(Vec<(String, Self)>),
}

impl JsonVal {
    /// Returns the property value for `key`, or `JsonVal::Null` if absent.
    pub(super) fn get(&self, key: &str) -> &Self {
        if let Self::Object(entries) = self {
            entries
                .iter()
                .find(|(name, _)| name == key)
                .map_or(&Self::Null, |(_, value)| value)
        } else {
            &Self::Null
        }
    }

    /// Returns `true` if `self` is an object containing `key`.
    pub(super) fn has(&self, key: &str) -> bool {
        matches!(self, Self::Object(entries) if entries.iter().any(|(name, _)| name == key))
    }

    /// Returns the inner string slice if `self` is `JsonVal::Str`.
    pub(super) fn text(&self) -> Option<&str> {
        if let Self::Str(value) = self {
            Some(value.as_str())
        } else {
            None
        }
    }

    /// Returns the inner slice if `self` is `JsonVal::Array`, otherwise empty.
    pub(super) fn array(&self) -> &[Self] {
        if let Self::Array(values) = self {
            values.as_slice()
        } else {
            &[]
        }
    }

    /// Returns the key-value entries if `self` is `JsonVal::Object`, otherwise empty.
    pub(super) fn object(&self) -> &[(String, Self)] {
        if let Self::Object(entries) = self {
            entries.as_slice()
        } else {
            &[]
        }
    }

    /// Returns `true` if `self` is `JsonVal::Str(expected)`.
    pub(super) fn is(&self, expected: &str) -> bool {
        self.text() == Some(expected)
    }

    /// Returns the exact integer value if `self` is an integral `JsonVal::Number`.
    pub(super) const fn number(&self) -> Option<i64> {
        if let Self::Number { int_value, .. } = self {
            *int_value
        } else {
            None
        }
    }

    /// Inserts or updates `key` on a `JsonVal::Object`.
    pub(super) fn set(&mut self, key: &str, value: Self) {
        if let Self::Object(entries) = self {
            if let Some((_, existing)) = entries.iter_mut().find(|(name, _)| name == key) {
                *existing = value;
            } else {
                entries.push((key.to_owned(), value));
            }
        }
    }

    /// Removes `key` from a `JsonVal::Object`.
    pub(super) fn remove(&mut self, key: &str) {
        if let Self::Object(entries) = self {
            entries.retain(|(name, _)| name != key);
        }
    }

    /// Returns `true` for `Null`, empty `Array`, or empty `Object`.
    pub(super) const fn empty(&self) -> bool {
        matches!(self, Self::Null)
            || matches!(self, Self::Array(values) if values.is_empty())
            || matches!(self, Self::Object(entries) if entries.is_empty())
    }

    /// Serializes `self` to compact JSON preserving key order and number lexemes.
    pub(super) fn dump(&self) -> String {
        let mut out = String::new();
        self.write_dump(&mut out);
        out
    }

    fn write_dump(&self, out: &mut String) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(true) => out.push_str("true"),
            Self::Bool(false) => out.push_str("false"),
            Self::Number { raw, .. } => out.push_str(raw),
            Self::Str(s) => write_json_string(s, out),
            Self::Array(items) => {
                out.push('[');
                for (idx, item) in items.iter().enumerate() {
                    if idx > 0 {
                        out.push(',');
                    }
                    item.write_dump(out);
                }
                out.push(']');
            }
            Self::Object(entries) => {
                out.push('{');
                for (idx, (k, v)) in entries.iter().enumerate() {
                    if idx > 0 {
                        out.push(',');
                    }
                    write_json_string(k, out);
                    out.push(':');
                    v.write_dump(out);
                }
                out.push('}');
            }
        }
    }
}

fn write_json_string(text: &str, out: &mut String) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str(r#"\""#),
            '\\' => out.push_str(r"\\"),
            '\u{08}' => out.push_str(r"\b"),
            '\u{0C}' => out.push_str(r"\f"),
            '\n' => out.push_str(r"\n"),
            '\r' => out.push_str(r"\r"),
            '\t' => out.push_str(r"\t"),
            c if u32::from(c) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Bounded, insertion-order-preserving JSON parser.
pub(super) struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    const fn new(input: &'a str) -> Self {
        Self {
            bytes: input.as_bytes(),
            pos: 0,
        }
    }

    fn parse_root(input: &'a str) -> std::result::Result<JsonVal, String> {
        Self::parse_with_limit(input, MAX_SCHEMA_BYTES)
    }

    /// Parses a JSON value enforcing `max_bytes` and nesting/duplicate-key bounds.
    pub(super) fn parse_with_limit(
        input: &'a str,
        max_bytes: usize,
    ) -> std::result::Result<JsonVal, String> {
        if input.len() > max_bytes {
            return Err("schema JSON exceeds source byte bound".to_owned());
        }
        let mut parser = Self::new(input);
        parser.skip_ws();
        let value = parser.parse_value(0)?;
        parser.skip_ws();
        if parser.pos != parser.bytes.len() {
            return Err("trailing characters after JSON value".to_owned());
        }
        Ok(value)
    }

    fn skip_ws(&mut self) {
        while let Some(&b) = self.bytes.get(self.pos) {
            if matches!(b, b' ' | b'\t' | b'\r' | b'\n') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn expect_byte(&mut self, expected: u8) -> std::result::Result<(), String> {
        if self.peek() == Some(expected) {
            self.pos += 1;
            Ok(())
        } else {
            Err(format!(
                "expected '{}' at offset {}",
                char::from(expected),
                self.pos
            ))
        }
    }

    fn parse_value(&mut self, depth: usize) -> std::result::Result<JsonVal, String> {
        if depth > MAX_JSON_DEPTH {
            return Err("JSON nesting depth exceeds bound".to_owned());
        }
        self.skip_ws();
        match self.peek() {
            Some(b'n') => self.parse_lit(b"null", JsonVal::Null),
            Some(b't') => self.parse_lit(b"true", JsonVal::Bool(true)),
            Some(b'f') => self.parse_lit(b"false", JsonVal::Bool(false)),
            Some(b'"') => self.parse_string().map(JsonVal::Str),
            Some(b'[') => self.parse_array(depth + 1),
            Some(b'{') => self.parse_object(depth + 1),
            Some(b'-' | b'0'..=b'9') => self.parse_number(),
            Some(_) => Err(format!("unexpected byte at offset {}", self.pos)),
            None => Err("unexpected end of JSON input".to_owned()),
        }
    }

    fn parse_lit(&mut self, lit: &[u8], val: JsonVal) -> std::result::Result<JsonVal, String> {
        if self.bytes.get(self.pos..self.pos + lit.len()) == Some(lit) {
            self.pos += lit.len();
            Ok(val)
        } else {
            Err(format!("invalid literal at offset {}", self.pos))
        }
    }

    fn parse_hex4(&mut self) -> std::result::Result<u16, String> {
        let slice = self
            .bytes
            .get(self.pos..self.pos + 4)
            .ok_or_else(|| "truncated unicode escape".to_owned())?;
        let text = std::str::from_utf8(slice).map_err(|_| "invalid hex escape".to_owned())?;
        let val = u16::from_str_radix(text, 16).map_err(|_| "invalid hex escape".to_owned())?;
        self.pos += 4;
        Ok(val)
    }

    fn parse_string(&mut self) -> std::result::Result<String, String> {
        self.expect_byte(b'"')?;
        let mut out = String::new();
        loop {
            let b = self
                .bytes
                .get(self.pos)
                .copied()
                .ok_or_else(|| "unterminated string literal".to_owned())?;
            if b == b'"' {
                self.pos += 1;
                return Ok(out);
            }
            if b == b'\\' {
                self.parse_escape(&mut out)?;
            } else if b < 0x20 {
                return Err("unescaped control character in JSON string".to_owned());
            } else if b < 0x80 {
                out.push(char::from(b));
                self.pos += 1;
            } else {
                let rest = std::str::from_utf8(&self.bytes[self.pos..])
                    .map_err(|_| "invalid UTF-8 in JSON string".to_owned())?;
                let ch = rest
                    .chars()
                    .next()
                    .ok_or_else(|| "invalid UTF-8 char".to_owned())?;
                out.push(ch);
                self.pos += ch.len_utf8();
            }
        }
    }

    fn parse_escape(&mut self, out: &mut String) -> std::result::Result<(), String> {
        self.pos += 1;
        let esc = self
            .bytes
            .get(self.pos)
            .copied()
            .ok_or_else(|| "truncated string escape".to_owned())?;
        self.pos += 1;
        match esc {
            b'"' => out.push('"'),
            b'\\' => out.push('\\'),
            b'/' => out.push('/'),
            b'b' => out.push('\u{08}'),
            b'f' => out.push('\u{0C}'),
            b'n' => out.push('\n'),
            b'r' => out.push('\r'),
            b't' => out.push('\t'),
            b'u' => {
                let code = self.parse_hex4()?;
                if (0xD800..=0xDBFF).contains(&code) {
                    if self.bytes.get(self.pos..self.pos + 2) != Some(b"\\u") {
                        return Err("unpaired high surrogate in JSON string".to_owned());
                    }
                    self.pos += 2;
                    let low = self.parse_hex4()?;
                    if !(0xDC00..=0xDFFF).contains(&low) {
                        return Err("invalid low surrogate in JSON string".to_owned());
                    }
                    let cp =
                        0x1_0000_u32 + (u32::from(code - 0xD800) << 10) + u32::from(low - 0xDC00);
                    let ch =
                        char::from_u32(cp).ok_or_else(|| "invalid surrogate scalar".to_owned())?;
                    out.push(ch);
                } else if (0xDC00..=0xDFFF).contains(&code) {
                    return Err("unpaired low surrogate in JSON string".to_owned());
                } else {
                    let ch = char::from_u32(u32::from(code))
                        .ok_or_else(|| "invalid unicode scalar".to_owned())?;
                    out.push(ch);
                }
            }
            _ => return Err("invalid JSON escape sequence".to_owned()),
        }
        Ok(())
    }

    fn parse_number(&mut self) -> std::result::Result<JsonVal, String> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => self.pos += 1,
            Some(b'1'..=b'9') => {
                while self.peek().is_some_and(|b| b.is_ascii_digit()) {
                    self.pos += 1;
                }
            }
            _ => return Err("invalid number integer part".to_owned()),
        }
        let mut is_int = true;
        if self.peek() == Some(b'.') {
            is_int = false;
            self.pos += 1;
            if !self.peek().is_some_and(|b| b.is_ascii_digit()) {
                return Err("missing fractional digits in number".to_owned());
            }
            while self.peek().is_some_and(|b| b.is_ascii_digit()) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            is_int = false;
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if !self.peek().is_some_and(|b| b.is_ascii_digit()) {
                return Err("missing exponent digits in number".to_owned());
            }
            while self.peek().is_some_and(|b| b.is_ascii_digit()) {
                self.pos += 1;
            }
        }
        let raw = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| "invalid number UTF-8".to_owned())?
            .to_owned();
        let float_value = raw
            .parse::<f64>()
            .map_err(|_| "invalid floating-point number".to_owned())?;
        if !float_value.is_finite() {
            return Err("non-finite JSON number".to_owned());
        }
        let int_value = if is_int {
            raw.parse::<i64>().ok()
        } else {
            None
        };
        Ok(JsonVal::Number {
            raw,
            int_value,
            float_value,
        })
    }

    fn parse_array(&mut self, depth: usize) -> std::result::Result<JsonVal, String> {
        self.expect_byte(b'[')?;
        self.skip_ws();
        let mut items = Vec::new();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(JsonVal::Array(items));
        }
        loop {
            items.push(self.parse_value(depth)?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(JsonVal::Array(items));
                }
                _ => return Err("expected ',' or ']' in array".to_owned()),
            }
        }
    }

    fn parse_object(&mut self, depth: usize) -> std::result::Result<JsonVal, String> {
        self.expect_byte(b'{')?;
        self.skip_ws();
        let mut entries: Vec<(String, JsonVal)> = Vec::new();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonVal::Object(entries));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err("expected string object key".to_owned());
            }
            let key = self.parse_string()?;
            if entries.iter().any(|(existing, _)| existing == &key) {
                return Err(format!("duplicate object key: {key}"));
            }
            self.skip_ws();
            self.expect_byte(b':')?;
            let val = self.parse_value(depth)?;
            entries.push((key, val));
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(JsonVal::Object(entries));
                }
                _ => return Err("expected ',' or '}' in object".to_owned()),
            }
        }
    }
}

// Local `$ref` resolution and `resolves_to_string` inspection.

fn lookup_json_pointer<'a>(root: &'a JsonVal, reference: &str) -> Option<&'a JsonVal> {
    let rest = reference.strip_prefix("#/")?;
    let mut target = root;
    for key in rest.split('/') {
        target = match target {
            JsonVal::Object(values) => values
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value),
            JsonVal::Array(values) => key
                .parse::<usize>()
                .ok()
                .and_then(|index| values.get(index)),
            _ => None,
        }?;
    }
    Some(target)
}

fn resolve_refs_walk_with_roots(
    node: &JsonVal,
    root: &JsonVal,
    extra_roots: &[JsonVal],
    refs: &mut BTreeMap<String, JsonVal>,
    depth: usize,
) -> std::result::Result<(), String> {
    if depth > MAX_SCHEMA_DEPTH {
        return Err("schema reference nesting exceeds bound".to_owned());
    }
    if let JsonVal::Array(values) = node {
        for value in values {
            resolve_refs_walk_with_roots(value, root, extra_roots, refs, depth + 1)?;
        }
        return Ok(());
    }
    if node.has("$ref") {
        let reference = node
            .get("$ref")
            .text()
            .ok_or_else(|| "schema reference must be text".to_owned())?;
        if refs.contains_key(reference) {
            return Ok(());
        }
        if !reference.starts_with("#/") {
            return Err(format!("unsupported schema ref {reference}"));
        }
        let target = lookup_json_pointer(root, reference)
            .or_else(|| {
                extra_roots
                    .iter()
                    .find_map(|extra| lookup_json_pointer(extra, reference))
            })
            .ok_or_else(|| format!("unresolved schema ref {reference}"))?
            .clone();
        refs.insert(reference.to_owned(), target.clone());
        resolve_refs_walk_with_roots(&target, root, extra_roots, refs, depth + 1)?;
        return Ok(());
    }
    for (_, value) in node.object() {
        resolve_refs_walk_with_roots(value, root, extra_roots, refs, depth + 1)?;
    }
    Ok(())
}

fn resolve_refs_walk(
    node: &JsonVal,
    root: &JsonVal,
    refs: &mut BTreeMap<String, JsonVal>,
    depth: usize,
) -> std::result::Result<(), String> {
    resolve_refs_walk_with_roots(node, root, &[], refs, depth)
}

/// Returns `true` when `node` resolves to a JSON string schema under `refs`.
pub(super) fn check_resolves_to_string(
    node: &JsonVal,
    refs: &BTreeMap<String, JsonVal>,
    visited: &mut BTreeSet<String>,
    depth: usize,
) -> bool {
    if depth > MAX_SCHEMA_DEPTH || !matches!(node, JsonVal::Object(_)) {
        return false;
    }
    if node.has("$ref") {
        let reference = node.get("$ref").text().unwrap_or("");
        return visited.insert(reference.to_owned())
            && refs
                .get(reference)
                .is_some_and(|target| check_resolves_to_string(target, refs, visited, depth + 1));
    }
    if node.get("type").is("string")
        || node
            .get("type")
            .array()
            .iter()
            .any(|kind| kind.is("string"))
    {
        return true;
    }
    for key in ["oneOf", "anyOf"] {
        if node
            .get(key)
            .array()
            .iter()
            .any(|value| check_resolves_to_string(value, refs, visited, depth + 1))
        {
            return true;
        }
    }
    if node.has("allOf")
        && node
            .get("allOf")
            .array()
            .iter()
            .all(|value| check_resolves_to_string(value, refs, visited, depth + 1))
    {
        return true;
    }
    if node.get("const").text().is_some()
        || node
            .get("enum")
            .array()
            .iter()
            .any(|value| value.text().is_some())
    {
        return true;
    }
    if ["pattern", "minLength", "maxLength"]
        .iter()
        .any(|key| node.has(key))
    {
        return true;
    }
    node.get("format").text().is_some_and(|format| {
        matches!(
            format,
            "date" | "time" | "date-time" | "uri" | "email" | "hostname" | "ipv4" | "ipv6"
        ) || format.starts_with("uuid")
    })
}

// Decimal integer range GBNF builder.

fn digit_range_class(from: u8, to: u8) -> String {
    if from == to {
        format!("[{}]", char::from(from))
    } else {
        format!("[{}-{}]", char::from(from), char::from(to))
    }
}

fn digits_rep(minimum: i32, maximum: i32) -> String {
    if minimum == 1 && maximum == 1 {
        return "[0-9]".to_owned();
    }
    if minimum == maximum {
        format!("[0-9]{{{minimum}}}")
    } else if maximum == i32::MAX {
        format!("[0-9]{{{minimum},}}")
    } else {
        format!("[0-9]{{{minimum},{maximum}}}")
    }
}

fn uniform_digits(from: &str, to: &str) -> std::result::Result<String, String> {
    if from.len() != to.len()
        || !from
            .bytes()
            .chain(to.bytes())
            .all(|byte| byte.is_ascii_digit())
    {
        return Err("integer grammar range requires equally sized digit strings".to_owned());
    }
    let shared = from
        .bytes()
        .zip(to.bytes())
        .take_while(|(left, right)| left == right)
        .count();
    let mut output = String::new();
    if shared > 0 {
        let _ = write!(output, "\"{}\"", &from[..shared]);
    }
    if shared < from.len() {
        if shared > 0 {
            output.push(' ');
        }
        let trailing = from.len() - shared - 1;
        let first = from.as_bytes()[shared];
        let last = to.as_bytes()[shared];
        if trailing == 0 {
            let _ = write!(output, "[{}-{}]", char::from(first), char::from(last));
        } else {
            let lower = &from[shared + 1..];
            let upper = &to[shared + 1..];
            let zeros = "0".repeat(trailing);
            let nines = "9".repeat(trailing);
            let mut reached = false;
            output.push('(');
            let trail_i32 =
                i32::try_from(trailing).map_err(|_| "digit length overflow".to_owned())?;
            if lower == zeros {
                let last_dec = last
                    .checked_sub(1)
                    .ok_or_else(|| "invalid integer digit bound".to_owned())?;
                output.push_str(&digit_range_class(first, last_dec));
                output.push(' ');
                output.push_str(&digits_rep(trail_i32, trail_i32));
            } else {
                let _ = write!(
                    output,
                    "[{}] ({})",
                    char::from(first),
                    uniform_digits(lower, &nines)?
                );
                if first + 1 < last {
                    output.push_str(" | ");
                    if upper == nines {
                        output.push_str(&digit_range_class(first + 1, last));
                        reached = true;
                    } else {
                        output.push_str(&digit_range_class(first + 1, last - 1));
                    }
                    output.push(' ');
                    output.push_str(&digits_rep(trail_i32, trail_i32));
                }
            }
            if !reached {
                output.push_str(" | ");
                output.push_str(&digit_range_class(last, last));
                output.push(' ');
                output.push_str(&uniform_digits(&zeros, upper)?);
            }
            output.push(')');
        }
    }
    Ok(output)
}

#[expect(
    clippy::too_many_lines,
    reason = "Preserve exact signed decimal transition rules"
)]
fn build_int_range(
    mut minimum: i64,
    maximum: i64,
    decimals: i32,
    top: bool,
) -> std::result::Result<String, String> {
    if !(1..=32).contains(&decimals) {
        return Err("integer grammar decimal budget outside 1..32".to_owned());
    }
    let has_min = minimum != i64::MIN;
    let has_max = maximum != i64::MAX;
    let less = (decimals - 1).max(1);
    if has_min && has_max {
        if minimum > maximum {
            return Err("invalid integer range: minimum exceeds maximum".to_owned());
        }
        if minimum < 0 && maximum < 0 {
            let max_neg = maximum
                .checked_neg()
                .ok_or_else(|| "integer grammar negation overflow".to_owned())?;
            let min_neg = minimum
                .checked_neg()
                .ok_or_else(|| "integer grammar negation overflow".to_owned())?;
            return Ok(format!(
                "\"-\" ({})",
                build_int_range(max_neg, min_neg, decimals, true)?
            ));
        }
        let mut prefix = String::new();
        if minimum < 0 {
            let min_neg = minimum
                .checked_neg()
                .ok_or_else(|| "integer grammar negation overflow".to_owned())?;
            prefix = format!(
                "\"-\" ({}) | ",
                build_int_range(0, min_neg, decimals, true)?
            );
            minimum = 0;
        }
        let mut lower = minimum.to_string();
        let upper = maximum.to_string();
        for length in lower.len()..upper.len() {
            prefix.push_str(&uniform_digits(&lower, &"9".repeat(length))?);
            prefix.push_str(" | ");
            lower = format!("1{}", "0".repeat(length));
        }
        prefix.push_str(&uniform_digits(&lower, &upper)?);
        return Ok(prefix);
    }
    if has_min {
        if minimum < 0 {
            let min_neg = minimum
                .checked_neg()
                .ok_or_else(|| "integer grammar negation overflow".to_owned())?;
            return Ok(format!(
                "\"-\" ({}) | [0] | [1-9] {}",
                build_int_range(i64::MIN, min_neg, decimals, false)?,
                digits_rep(0, decimals - 1)
            ));
        }
        if minimum == 0 {
            return Ok(if top {
                format!("[0] | [1-9] {}", digits_rep(0, less))
            } else {
                digits_rep(1, decimals)
            });
        }
        if minimum <= 9 {
            let digit =
                b'0' + u8::try_from(minimum).map_err(|_| "digit conversion overflow".to_owned())?;
            let first = if top { b'1' } else { b'0' };
            let prefix = if digit > first {
                format!(
                    "{} {} | ",
                    digit_range_class(first, digit - 1),
                    digits_rep(1, less)
                )
            } else {
                String::new()
            };
            return Ok(format!(
                "{prefix}{} {}",
                digit_range_class(digit, b'9'),
                digits_rep(0, less)
            ));
        }
        let lower = minimum.to_string();
        let length =
            i32::try_from(lower.len()).map_err(|_| "integer length overflow".to_owned())?;
        let digit = lower.as_bytes()[0];
        let mut output = if digit > b'1' {
            format!(
                "{} {} | ",
                digit_range_class(if top { b'1' } else { b'0' }, digit - 1),
                digits_rep(length, less)
            )
        } else {
            String::new()
        };
        let rest_min: i64 = lower[1..]
            .parse()
            .map_err(|_| "invalid lower tail integer".to_owned())?;
        let _ = write!(
            output,
            "{} ({})",
            digit_range_class(digit, digit),
            build_int_range(rest_min, i64::MAX, less, false)?
        );
        if digit < b'9' {
            let _ = write!(
                output,
                " | {} {}",
                digit_range_class(digit + 1, b'9'),
                digits_rep(length - 1, less)
            );
        }
        return Ok(output);
    }
    if has_max {
        if maximum >= 0 {
            let prefix = if top {
                format!("\"-\" [1-9] {} | ", digits_rep(0, less))
            } else {
                String::new()
            };
            return Ok(format!(
                "{prefix}{}",
                build_int_range(0, maximum, decimals, true)?
            ));
        }
        let max_neg = maximum
            .checked_neg()
            .ok_or_else(|| "integer grammar negation overflow".to_owned())?;
        return Ok(format!(
            "\"-\" ({})",
            build_int_range(max_neg, i64::MAX, decimals, false)?
        ));
    }
    Err("at least one integer grammar bound is required".to_owned())
}

// Schema keyword formatting, anchored regex transform, and GBNF converter.

fn builtin(name: &str) -> Option<(&'static str, &'static [&'static str])> {
    Some(match name {
        "boolean" => (r#"("true" | "false")"#, &[]),
        "decimal-part" => ("[0-9]{1,16}", &[]),
        "integral-part" => ("[0] | [1-9] [0-9]{0,15}", &[]),
        "number" => (
            r#"("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?"#,
            &["integral-part", "decimal-part"],
        ),
        "integer" => (r#"("-"? integral-part)"#, &["integral-part"]),
        "value" => (
            "object | array | string | number | boolean | null",
            &["object", "array", "string", "number", "boolean", "null"],
        ),
        "object" => (
            r#""{" space ( string ":" space value ("," space string ":" space value)* )? space "}""#,
            &["string", "value"],
        ),
        "array" => (
            r#""[" space ( value ("," space value)* )? space "]""#,
            &["value"],
        ),
        "uuid" => (
            r#""\"" [0-9a-fA-F]{8} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{12} "\"""#,
            &[],
        ),
        "char" => (
            r#"[^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})"#,
            &[],
        ),
        "string" => (r#""\"" char* "\"""#, &["char"]),
        "null" => (r#""null""#, &[]),
        "date" => (
            r#"[0-9]{4} "-" ( "0" [1-9] | "1" [0-2] ) "-" ( "0" [1-9] | [1-2] [0-9] | "3" [0-1] )"#,
            &[],
        ),
        "time" => (
            r#"([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9]{3} )? ( "Z" | ( "+" | "-" ) ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] )"#,
            &[],
        ),
        "date-time" => (r#"date "T" time"#, &["date", "time"]),
        "date-string" => (r#""\"" date "\"""#, &["date"]),
        "time-string" => (r#""\"" time "\"""#, &["time"]),
        "date-time-string" => (r#""\"" date-time "\"""#, &["date-time"]),
        _ => return None,
    })
}

fn escape_name(name: &str) -> String {
    let mut output = String::new();
    let mut invalid_run = false;
    for value in name.chars() {
        if value.is_ascii_alphanumeric() || value == '-' {
            output.push(value);
            invalid_run = false;
        } else if !invalid_run {
            output.push('-');
            invalid_run = true;
        }
    }
    output
}

fn format_literal(text: &str) -> String {
    let mut result = String::from("\"");
    for value in text.chars() {
        match value {
            '\r' => result.push_str("\\r"),
            '\n' => result.push_str("\\n"),
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            _ => result.push(value),
        }
    }
    result.push('"');
    result
}

fn repetition(item: &str, minimum: i32, maximum: i32, separator: &str) -> String {
    if maximum == 0 {
        return String::new();
    }
    if minimum == 0 && maximum == 1 {
        return format!("{item}?");
    }
    if separator.is_empty() {
        return match (minimum, maximum) {
            (1, i32::MAX) => format!("{item}+"),
            (0, i32::MAX) => format!("{item}*"),
            _ => format!(
                "{item}{{{minimum},{}}}",
                if maximum == i32::MAX {
                    String::new()
                } else {
                    maximum.to_string()
                }
            ),
        };
    }
    let rest = repetition(
        &format!("({separator} {item})"),
        minimum.saturating_sub(1).max(0),
        if maximum == i32::MAX {
            maximum
        } else {
            maximum.saturating_sub(1)
        },
        "",
    );
    let result = format!("{item} {rest}");
    if minimum == 0 {
        format!("({result})?")
    } else {
        result
    }
}

enum PatternError {
    Unsupported,
    Invalid,
}
type PatternResult<T> = std::result::Result<T, PatternError>;
type Piece = (String, bool);

fn piece_to_rule(piece: &Piece) -> String {
    if piece.1 {
        format!("\"{}\"", piece.0)
    } else {
        piece.0.clone()
    }
}

fn join_pieces(sequence: &[Piece]) -> Piece {
    let mut rules = Vec::new();
    let mut literal = String::new();
    for piece in sequence {
        if piece.1 {
            literal.push_str(&piece.0);
        } else {
            if !literal.is_empty() {
                rules.push(format!("\"{literal}\""));
                literal.clear();
            }
            rules.push(piece.0.clone());
        }
    }
    if !literal.is_empty() {
        rules.push(format!("\"{literal}\""));
    }
    (rules.join(" "), false)
}

const fn is_regex_nonliteral(value: char) -> bool {
    matches!(
        value,
        '|' | '.' | '(' | ')' | '[' | ']' | '{' | '}' | '*' | '+' | '?' | '^' | '$'
    )
}

fn pattern_escape_length(values: &[char], offset: usize) -> usize {
    if values.get(offset) != Some(&'\\') {
        return 0;
    }
    let length = match values.get(offset + 1) {
        Some('x') => 2,
        Some('u') => 4,
        Some('U') => 8,
        Some('t' | 'r' | 'n' | '\\' | '"' | '[' | ']') => return 2,
        _ => return 0,
    };
    if values
        .get(offset + 2..offset + 2 + length)
        .is_some_and(|digits| digits.iter().all(char::is_ascii_hexdigit))
    {
        length + 2
    } else {
        0
    }
}

struct PatternParser<'a> {
    converter: &'a mut SchemaConverter,
    name: &'a str,
    values: Vec<char>,
    offset: usize,
    depth: usize,
    subrules: BTreeMap<String, String>,
}

impl PatternParser<'_> {
    fn subrule(&mut self, rule: &str) -> &str {
        if let Some(existing) = self.subrules.get(rule) {
            return existing;
        }
        let assigned = self
            .converter
            .add(&format!("{}-{}", self.name, self.subrules.len() + 1), rule);
        self.subrules.entry(rule.to_owned()).or_insert(assigned)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Preserve exact ordered regex token and quantifier semantics"
    )]
    fn transform(&mut self) -> PatternResult<Piece> {
        let mut sequence: Vec<Piece> = Vec::new();
        while self.offset < self.values.len() {
            let value = self.values[self.offset];
            match value {
                '.' => {
                    let rule = if self.converter.dotall {
                        r"[\U00000000-\U0010FFFF]"
                    } else {
                        r"[^\x0A\x0D]"
                    };
                    let dot_name = self.converter.add("dot", rule);
                    sequence.push((dot_name, false));
                    self.offset += 1;
                }
                '(' => {
                    self.offset += 1;
                    if self.values.get(self.offset) == Some(&'?') {
                        if self.values.get(self.offset + 1) == Some(&':') {
                            self.offset += 2;
                        } else {
                            return Err(PatternError::Unsupported);
                        }
                    }
                    self.depth += 1;
                    if self.depth > 100 {
                        return Err(PatternError::Unsupported);
                    }
                    let piece = self.transform()?;
                    sequence.push((format!("({})", piece_to_rule(&piece)), false));
                }
                ')' => {
                    self.offset += 1;
                    if self.depth == 0 {
                        return Err(PatternError::Invalid);
                    }
                    self.depth -= 1;
                    return Ok(join_pieces(&sequence));
                }
                '^' | '$' => return Err(PatternError::Unsupported),
                '[' => {
                    let mut range = String::from("[");
                    self.offset += 1;
                    while self.offset < self.values.len() && self.values[self.offset] != ']' {
                        if self.values[self.offset] == '\\' {
                            let length = pattern_escape_length(&self.values, self.offset);
                            if length == 0 {
                                return Err(PatternError::Unsupported);
                            }
                            range.extend(&self.values[self.offset..self.offset + length]);
                            self.offset += length;
                        } else {
                            range.push(self.values[self.offset]);
                            self.offset += 1;
                        }
                    }
                    if self.offset == self.values.len() {
                        return Err(PatternError::Invalid);
                    }
                    range.push(']');
                    self.offset += 1;
                    sequence.push((range, false));
                }
                '|' => {
                    sequence.push(("|".into(), false));
                    self.offset += 1;
                }
                '*' | '+' | '?' => {
                    let piece = sequence.last_mut().ok_or(PatternError::Invalid)?;
                    piece.0 = format!("{}{value}", piece_to_rule(piece));
                    piece.1 = false;
                    self.offset += 1;
                }
                '{' => {
                    self.offset += 1;
                    let start = self.offset;
                    while self.offset < self.values.len() && self.values[self.offset] != '}' {
                        self.offset += 1;
                    }
                    if self.offset == self.values.len() {
                        return Err(PatternError::Unsupported);
                    }
                    let count: String = self.values[start..self.offset].iter().collect();
                    self.offset += 1;
                    let counts: Vec<_> = count.split(',').collect();
                    let parse =
                        |val: &str| val.parse::<i32>().map_err(|_| PatternError::Unsupported);
                    let (minimum, maximum) = match counts.as_slice() {
                        [exact] => {
                            let n = parse(exact)?;
                            (n, n)
                        }
                        [lo, hi] => (
                            if lo.is_empty() { 0 } else { parse(lo)? },
                            if hi.is_empty() { i32::MAX } else { parse(hi)? },
                        ),
                        _ => return Err(PatternError::Unsupported),
                    };
                    let piece = sequence.last_mut().ok_or(PatternError::Invalid)?;
                    piece.0 = if piece.1 {
                        repetition(&piece_to_rule(piece), minimum, maximum, "")
                    } else {
                        repetition(self.subrule(&piece.0), minimum, maximum, "")
                    };
                    piece.1 = false;
                }
                _ => {
                    let mut literal = String::new();
                    while self.offset < self.values.len() {
                        let ch = self.values[self.offset];
                        if ch == '\\' {
                            let next = *self
                                .values
                                .get(self.offset + 1)
                                .ok_or(PatternError::Invalid)?;
                            if matches!(
                                next,
                                '^' | '$'
                                    | '.'
                                    | '['
                                    | ']'
                                    | '('
                                    | ')'
                                    | '|'
                                    | '{'
                                    | '}'
                                    | '*'
                                    | '+'
                                    | '?'
                            ) {
                                literal.push(next);
                                self.offset += 2;
                            } else {
                                let length = pattern_escape_length(&self.values, self.offset);
                                if length == 0 {
                                    return Err(PatternError::Unsupported);
                                }
                                literal.extend(&self.values[self.offset..self.offset + length]);
                                self.offset += length;
                            }
                        } else if ch == '"' {
                            literal.push_str("\\\"");
                            self.offset += 1;
                        } else if !is_regex_nonliteral(ch)
                            && (self.offset + 1 == self.values.len()
                                || literal.is_empty()
                                || self.values[self.offset + 1] == '.'
                                || !is_regex_nonliteral(self.values[self.offset + 1]))
                        {
                            literal.push(ch);
                            self.offset += 1;
                        } else {
                            break;
                        }
                    }
                    if literal.is_empty() {
                        return Err(PatternError::Unsupported);
                    }
                    sequence.push((literal, true));
                }
            }
        }
        Ok(join_pieces(&sequence))
    }
}

struct SchemaConverter {
    rules: BTreeMap<String, String>,
    refs: BTreeMap<String, JsonVal>,
    extra_roots: Vec<JsonVal>,
    resolving: BTreeSet<String>,
    depth: usize,
    dotall: bool,
    raw: bool,
}

impl SchemaConverter {
    fn new() -> Self {
        Self {
            rules: BTreeMap::from([("space".into(), "| \" \" | \"\\n\"{1,2} [ \\t]{0,20}".into())]),
            refs: BTreeMap::new(),
            extra_roots: Vec::new(),
            resolving: BTreeSet::new(),
            depth: 0,
            dotall: false,
            raw: false,
        }
    }

    fn add(&mut self, name: &str, rule: &str) -> String {
        let escaped = escape_name(name);
        match self.rules.get(&escaped) {
            Some(existing) if existing == rule => return escaped,
            None => {
                self.rules.insert(escaped.clone(), rule.to_owned());
                return escaped;
            }
            Some(_) => {}
        }
        let mut index = 0_usize;
        loop {
            let key = format!("{escaped}{index}");
            match self.rules.get(&key) {
                Some(existing) if existing == rule => return key,
                None => {
                    self.rules.insert(key.clone(), rule.to_owned());
                    return key;
                }
                Some(_) => index += 1,
            }
        }
    }

    fn primitive(&mut self, name: &str, kind: &str) -> std::result::Result<String, String> {
        let (rule, dependencies) =
            builtin(kind).ok_or_else(|| format!("unknown schema rule {kind}"))?;
        let assigned = self.add(name, rule);
        for dependency in dependencies {
            if !self.rules.contains_key(*dependency) {
                self.primitive(dependency, dependency)?;
            }
        }
        Ok(assigned)
    }

    fn union(
        &mut self,
        name: &str,
        alternatives: &[JsonVal],
    ) -> std::result::Result<String, String> {
        let mut alt_rules = Vec::new();
        for (index, schema) in alternatives.iter().enumerate() {
            let sep = if name.is_empty() { "alternative-" } else { "-" };
            alt_rules.push(self.visit(schema, &format!("{name}{sep}{index}"))?);
        }
        Ok(alt_rules.join(" | "))
    }

    fn resolve(&mut self, schema: &JsonVal) -> std::result::Result<(), String> {
        resolve_refs_walk_with_roots(schema, schema, &self.extra_roots, &mut self.refs, 0)
    }

    fn reference(&mut self, reference: &str) -> std::result::Result<String, String> {
        let fragment = reference
            .split_once('#')
            .map_or(reference, |(_, frag)| frag);
        let name = format!("ref{}", escape_name(fragment));
        if self.rules.contains_key(&name) || !self.resolving.insert(reference.to_owned()) {
            return Ok(name);
        }
        let resolved = if let Some(found) = self.refs.get(reference).cloned() {
            found
        } else {
            let found = self
                .extra_roots
                .iter()
                .find_map(|extra| lookup_json_pointer(extra, reference))
                .cloned()
                .ok_or_else(|| format!("unresolved schema ref {reference}"))?;
            self.refs.insert(reference.to_owned(), found.clone());
            for extra in &self.extra_roots {
                let _ = resolve_refs_walk_with_roots(
                    &found,
                    extra,
                    &self.extra_roots,
                    &mut self.refs,
                    0,
                );
            }
            found
        };
        let result = self.visit(&resolved, &name);
        self.resolving.remove(reference);
        result
    }

    fn pattern(&mut self, pat: &str, name: &str) -> std::result::Result<String, String> {
        let snapshot = self.rules.clone();
        let outcome = (|| -> PatternResult<String> {
            if pat.len() < 2 || !pat.starts_with('^') || !pat.ends_with('$') {
                return Err(PatternError::Unsupported);
            }
            let mut parser = PatternParser {
                converter: self,
                name,
                values: pat[1..pat.len() - 1].chars().collect(),
                offset: 0,
                depth: 0,
                subrules: BTreeMap::new(),
            };
            let rule = piece_to_rule(&parser.transform()?);
            if parser.depth != 0 {
                return Err(PatternError::Invalid);
            }
            if self.raw {
                Ok(format!("({rule})"))
            } else {
                Ok(format!("\"\\\"\" ({rule}) \"\\\"\""))
            }
        })();
        match outcome {
            Ok(rule) => Ok(self.add(name, &rule)),
            Err(PatternError::Unsupported) => {
                self.rules = snapshot;
                let rule = self.primitive("string", "string")?;
                Ok(self.add(name, &rule))
            }
            Err(PatternError::Invalid) => {
                self.rules = snapshot;
                Err(format!("invalid schema pattern {pat}"))
            }
        }
    }

    fn not_strings(&mut self, strings: &[String]) -> std::result::Result<String, String> {
        #[derive(Default)]
        struct ByteTrie {
            children: BTreeMap<u8, Self>,
            end: bool,
        }
        fn render(node: &ByteTrie) -> String {
            let mut result = Vec::new();
            let rejects: String = node.children.keys().map(|&b| char::from(b)).collect();
            for (&byte, child) in &node.children {
                let mut rule = format!("[{}]", char::from(byte));
                if !child.children.is_empty() {
                    rule.push_str(" (");
                    rule.push_str(&render(child));
                    rule.push(')');
                } else if child.end {
                    rule.push_str(" char+");
                }
                result.push(rule);
            }
            if !node.children.is_empty() {
                result.push(format!("[^\"{rejects}] char*"));
            }
            result.join(" | ")
        }
        let mut trie = ByteTrie::default();
        for text in strings {
            let mut node = &mut trie;
            for byte in text.bytes() {
                node = node.children.entry(byte).or_default();
            }
            node.end = true;
        }
        self.primitive("char", "char")?;
        Ok(format!(
            "[\"] ( {} ){} [\"]",
            render(&trie),
            if trie.end { "" } else { "?" }
        ))
    }

    fn optional_rest(
        &mut self,
        keys: &[String],
        optional: bool,
        kv_rules: &BTreeMap<String, String>,
        prefix: &str,
    ) -> String {
        let Some(key) = keys.first() else {
            return String::new();
        };
        let kv = &kv_rules[key];
        let comma = format!("( \",\" space {kv} )");
        let mut result = if optional {
            format!("{comma}{}", if key == "*" { "*" } else { "?" })
        } else if key == "*" {
            format!("{kv} {comma}*")
        } else {
            kv.clone()
        };
        if keys.len() > 1 {
            let rest = self.optional_rest(&keys[1..], true, kv_rules, prefix);
            let rest_ref = self.add(&format!("{prefix}{key}-rest"), &rest);
            result.push(' ');
            result.push_str(&rest_ref);
        }
        result
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Preserves object property/additionalProperties grammar construction cleanly"
    )]
    fn object_rule(
        &mut self,
        properties: &[(String, JsonVal)],
        required: &BTreeSet<String>,
        name: &str,
        additional: &JsonVal,
    ) -> std::result::Result<String, String> {
        if properties.len() > 256 || properties.iter().any(|(key, _)| key.len() > 256) {
            return Err("schema object property resource limit exceeded".to_owned());
        }
        let prefix = if name.is_empty() {
            String::new()
        } else {
            format!("{name}-")
        };
        let saved_raw = self.raw;
        self.raw = false;
        let res = (|| -> std::result::Result<String, String> {
            let mut names = Vec::new();
            let mut mandatory = Vec::new();
            let mut optional = Vec::new();
            let mut kv_rules = BTreeMap::new();
            for (key, prop_schema) in properties {
                let val_rule = self.visit(prop_schema, &format!("{prefix}{key}"))?;
                let kv_body = format!(
                    "{} space \":\" space {val_rule}",
                    format_literal(&JsonVal::Str(key.clone()).dump())
                );
                let rule = self.add(&format!("{prefix}{key}-kv"), &kv_body);
                kv_rules.insert(key.clone(), rule);
                names.push(key.clone());
                if required.contains(key) {
                    mandatory.push(key.clone());
                } else {
                    optional.push(key.clone());
                }
            }
            if matches!(additional, JsonVal::Bool(true) | JsonVal::Object(_)) {
                let subname = format!("{prefix}additional");
                let val_rule = if matches!(additional, JsonVal::Object(_)) {
                    self.visit(additional, &format!("{subname}-value"))?
                } else {
                    self.primitive("value", "value")?
                };
                let key_rule = if names.is_empty() {
                    self.primitive("string", "string")?
                } else {
                    let expr = self.not_strings(&names)?;
                    self.add(&format!("{subname}-k"), &expr)
                };
                let rule = self.add(
                    &format!("{subname}-kv"),
                    &format!("{key_rule} \":\" space {val_rule}"),
                );
                kv_rules.insert("*".into(), rule);
                optional.push("*".into());
            }
            let mut seen_optional = false;
            let mut interleaved = false;
            for key in &names {
                if required.contains(key) {
                    interleaved |= seen_optional;
                } else {
                    seen_optional = true;
                }
            }
            let declared = if interleaved
                && names.first().is_some_and(|key| required.contains(key))
                && !kv_rules.contains_key("*")
            {
                let mut out = format!("\"{{\" space {}", kv_rules[&names[0]]);
                for key in &names[1..] {
                    let kv = format!(" \",\" space {}", kv_rules[key]);
                    if required.contains(key) {
                        out.push_str(&kv);
                    } else {
                        let _ = write!(out, " ({kv} )?");
                    }
                }
                out.push_str(" space \"}\"");
                Some(out)
            } else {
                None
            };
            let mut out = format!(
                "\"{{\" space {}",
                mandatory
                    .iter()
                    .map(|key| kv_rules[key].clone())
                    .collect::<Vec<_>>()
                    .join(" \",\" space ")
            );
            if !optional.is_empty() {
                out.push_str(" (");
                if !mandatory.is_empty() {
                    out.push_str(" \",\" space ( ");
                }
                let mut alts = Vec::new();
                for index in 0..optional.len() {
                    alts.push(self.optional_rest(&optional[index..], false, &kv_rules, &prefix));
                }
                out.push_str(&alts.join(" | "));
                if !mandatory.is_empty() {
                    out.push_str(" )");
                }
                out.push_str(" )?");
            }
            out.push_str(" space \"}\"");
            Ok(declared.map_or_else(|| out.clone(), |decl| format!("{decl} | {out}")))
        })();
        self.raw = saved_raw;
        res
    }

    fn visit(&mut self, schema: &JsonVal, name: &str) -> std::result::Result<String, String> {
        self.depth += 1;
        if self.depth > MAX_SCHEMA_DEPTH || self.rules.len() > MAX_RULES {
            self.depth -= 1;
            return Err("schema grammar nesting/rule bound exceeded".to_owned());
        }
        let res = self.visit_inner(schema, name);
        self.depth -= 1;
        res
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Preserve ordered schema keyword precedence matching source specification"
    )]
    fn visit_inner(&mut self, schema: &JsonVal, name: &str) -> std::result::Result<String, String> {
        for key in ["oneOf", "anyOf", "type"] {
            if matches!(schema.get(key), JsonVal::Array(values) if values.is_empty()) {
                let mut stripped = schema.clone();
                stripped.remove(key);
                return self.visit(&stripped, name);
            }
        }
        let kind = schema.get("type");
        let fmt_str = schema.get("format").text().unwrap_or("");
        let rule_name = if name == "root" || builtin(name).is_some() {
            format!("{name}-")
        } else if name.is_empty() {
            "root".to_owned()
        } else {
            name.to_owned()
        };
        if schema.has("$ref") {
            let ptr = schema
                .get("$ref")
                .text()
                .ok_or_else(|| "schema ref must be text".to_owned())?;
            let target_rule = self.reference(ptr)?;
            return Ok(self.add(&rule_name, &target_rule));
        }
        if schema.has("oneOf") || schema.has("anyOf") {
            let alts = schema
                .get(if schema.has("oneOf") {
                    "oneOf"
                } else {
                    "anyOf"
                })
                .array();
            let rule = self.union(name, alts)?;
            return Ok(self.add(&rule_name, &rule));
        }
        if let JsonVal::Array(types) = kind {
            let alts: Vec<_> = types
                .iter()
                .map(|t| {
                    let mut copy = schema.clone();
                    copy.set("type", t.clone());
                    copy
                })
                .collect();
            let rule = self.union(name, &alts)?;
            return Ok(self.add(&rule_name, &rule));
        }
        if schema.has("const") {
            let val = schema.get("const");
            let body = if self.raw {
                val.text()
                    .map_or_else(|| format_literal(&val.dump()), format_literal)
            } else {
                format_literal(&val.dump())
            };
            return Ok(self.add(&rule_name, &body));
        }
        if schema.has("enum") {
            let vals: Vec<_> = schema
                .get("enum")
                .array()
                .iter()
                .map(|v| {
                    if self.raw {
                        v.text()
                            .map_or_else(|| format_literal(&v.dump()), format_literal)
                    } else {
                        format_literal(&v.dump())
                    }
                })
                .collect();
            return Ok(self.add(&rule_name, &format!("({})", vals.join(" | "))));
        }
        if (matches!(kind, JsonVal::Null) || kind.is("object"))
            && (schema.has("properties")
                || schema.has("additionalProperties")
                    && schema.get("additionalProperties") != &JsonVal::Bool(true))
        {
            let required: BTreeSet<_> = schema
                .get("required")
                .array()
                .iter()
                .filter_map(JsonVal::text)
                .map(str::to_owned)
                .collect();
            let rule = self.object_rule(
                schema.get("properties").object(),
                &required,
                name,
                schema.get("additionalProperties"),
            )?;
            return Ok(self.add(&rule_name, &rule));
        }
        if (matches!(kind, JsonVal::Null) || kind.is("object") || kind.is("string"))
            && schema.has("allOf")
        {
            let mut properties = Vec::new();
            let mut required = BTreeSet::new();
            let mut enums = BTreeMap::new();
            let all_of_items = schema.get("allOf").array();
            for component in all_of_items {
                let (components, mandatory) = if component.has("anyOf") {
                    (component.get("anyOf").array().to_vec(), false)
                } else {
                    (vec![component.clone()], true)
                };
                for sub in components {
                    self.collect_component(
                        &sub,
                        mandatory,
                        &mut properties,
                        &mut required,
                        &mut enums,
                        0,
                    )?;
                }
            }
            if !enums.is_empty() {
                let intersection: Vec<_> = enums
                    .into_iter()
                    .filter_map(|(k, count)| (count == all_of_items.len()).then_some(k))
                    .collect();
                if !intersection.is_empty() {
                    return Ok(self.add(&rule_name, &format!("({})", intersection.join(" | "))));
                }
            }
            let rule = self.object_rule(&properties, &required, name, &JsonVal::Null)?;
            return Ok(self.add(&rule_name, &rule));
        }
        let prefix = if name.is_empty() {
            String::new()
        } else {
            format!("{name}-")
        };
        if (matches!(kind, JsonVal::Null) || kind.is("array"))
            && (schema.has("items") || schema.has("prefixItems"))
        {
            let items = schema.get(if schema.has("items") {
                "items"
            } else {
                "prefixItems"
            });
            let saved_raw = self.raw;
            self.raw = false;
            let rule_res = if let JsonVal::Array(tuple) = items {
                let mut rules = Vec::new();
                for (index, item) in tuple.iter().enumerate() {
                    rules.push(self.visit(item, &format!("{prefix}tuple-{index}"))?);
                }
                Ok(format!(
                    "\"[\" space {} space \"]\"",
                    rules.join(" \",\" space ")
                ))
            } else {
                self.visit(items, &format!("{prefix}item"))
                    .and_then(|item| {
                        Ok(format!(
                            "\"[\" space {} space \"]\"",
                            repetition(
                                &item,
                                int_bound(schema, "minItems", 0)?,
                                int_bound(schema, "maxItems", i32::MAX)?,
                                "\",\" space"
                            )
                        ))
                    })
            };
            self.raw = saved_raw;
            return Ok(self.add(&rule_name, &rule_res?));
        }
        if (matches!(kind, JsonVal::Null) || kind.is("string")) && schema.has("pattern") {
            let pat = schema
                .get("pattern")
                .text()
                .ok_or_else(|| "schema pattern must be text".to_owned())?;
            return self.pattern(pat, &rule_name);
        }
        if matches!(kind, JsonVal::Null) || kind.is("string") {
            if fmt_str == "uuid"
                || fmt_str.strip_prefix("uuid").is_some_and(|suffix| {
                    suffix.len() == 1 && matches!(suffix.as_bytes()[0], b'1'..=b'5')
                })
            {
                if self.raw {
                    return Ok(self.add(
                        if rule_name == "root" { "root" } else { fmt_str },
                        r#"[0-9a-fA-F]{8} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{12}"#,
                    ));
                }
                return self.primitive(if rule_name == "root" { "root" } else { fmt_str }, "uuid");
            }
            let prim = format!("{fmt_str}-string");
            if builtin(&prim).is_some() {
                let target_kind = if self.raw && builtin(fmt_str).is_some() {
                    fmt_str
                } else {
                    &prim
                };
                let value = self.primitive(target_kind, target_kind)?;
                return Ok(self.add(&rule_name, &value));
            }
        }
        if kind.is("string") && (schema.has("minLength") || schema.has("maxLength")) {
            let ch_rule = self.primitive("char", "char")?;
            let min_len = int_bound(schema, "minLength", 0)?;
            let max_len = int_bound(schema, "maxLength", i32::MAX)?;
            let rep = repetition(&ch_rule, min_len, max_len, "");
            let body = if self.raw {
                rep
            } else {
                format!("\"\\\"\" {rep} \"\\\"\"")
            };
            return Ok(self.add(&rule_name, &body));
        }
        if kind.is("integer")
            && ["minimum", "exclusiveMinimum", "maximum", "exclusiveMaximum"]
                .iter()
                .any(|key| schema.has(key))
        {
            let minimum = match schema.get("minimum").number() {
                Some(n) => n,
                None => match schema.get("exclusiveMinimum").number() {
                    Some(v) => v
                        .checked_add(1)
                        .ok_or_else(|| "schema integer lower bound overflow".to_owned())?,
                    None => i64::MIN,
                },
            };
            let maximum = match schema.get("maximum").number() {
                Some(n) => n,
                None => match schema.get("exclusiveMaximum").number() {
                    Some(v) => v
                        .checked_sub(1)
                        .ok_or_else(|| "schema integer upper bound overflow".to_owned())?,
                    None => i64::MAX,
                },
            };
            let range_body = build_int_range(minimum, maximum, 16, true)?;
            return Ok(self.add(&rule_name, &format!("({range_body})")));
        }
        if schema.empty() || kind.is("object") {
            let value = self.primitive("object", "object")?;
            return Ok(self.add(&rule_name, &value));
        }
        if matches!(kind, JsonVal::Null) && matches!(schema, JsonVal::Object(_)) {
            let value = self.primitive("value", "value")?;
            return Ok(self.add(&rule_name, &value));
        }
        let kind_name = kind
            .text()
            .filter(|k| builtin(k).is_some())
            .ok_or_else(|| format!("unrecognized schema {}", bounded_preview(&schema.dump())))?;
        if self.raw && kind_name == "string" {
            let ch_rule = self.primitive("char", "char")?;
            return Ok(self.add(&rule_name, &format!("{ch_rule}*")));
        }
        self.primitive(
            if rule_name == "root" {
                "root"
            } else {
                kind_name
            },
            kind_name,
        )
    }

    fn collect_component(
        &self,
        component: &JsonVal,
        mandatory: bool,
        properties: &mut Vec<(String, JsonVal)>,
        required: &mut BTreeSet<String>,
        enums: &mut BTreeMap<String, usize>,
        depth: usize,
    ) -> std::result::Result<(), String> {
        if depth > MAX_SCHEMA_DEPTH {
            return Err("schema intersection ref depth exceeded".to_owned());
        }
        if component.has("$ref") {
            let ptr = component.get("$ref").text().unwrap_or("");
            let target = self
                .refs
                .get(ptr)
                .ok_or_else(|| "unresolved intersection ref".to_owned())?;
            return self.collect_component(
                target,
                mandatory,
                properties,
                required,
                enums,
                depth + 1,
            );
        }
        if component.has("properties") {
            for (key, value) in component.get("properties").object() {
                properties.push((key.clone(), value.clone()));
                if mandatory {
                    required.insert(key.clone());
                }
            }
        } else if component.has("enum") {
            for value in component.get("enum").array() {
                let lit = if self.raw {
                    value
                        .text()
                        .map_or_else(|| format_literal(&value.dump()), format_literal)
                } else {
                    format_literal(&value.dump())
                };
                *enums.entry(lit).or_default() += 1;
            }
        }
        Ok(())
    }
}

fn int_bound(schema: &JsonVal, name: &str, default: i32) -> std::result::Result<i32, String> {
    schema.get(name).number().map_or(Ok(default), |value| {
        i32::try_from(value).map_err(|e| e.to_string())
    })
}

/// Stateful JSON Schema to GBNF converter preserving shared `$ref` definitions and rule tables.
pub(super) struct SchemaGrammarContext {
    converter: SchemaConverter,
}

impl SchemaGrammarContext {
    /// Creates an empty schema grammar context.
    pub(super) fn new() -> Self {
        Self {
            converter: SchemaConverter::new(),
        }
    }

    /// Merges externally emitted GBNF rules before generating schema rules.
    pub(super) fn merge_rules(&mut self, rules: &BTreeMap<String, String>) {
        for (name, body) in rules {
            if !self.converter.rules.contains_key(name) {
                self.converter.rules.insert(name.clone(), body.clone());
            }
        }
    }

    /// Registers a schema document for external `$ref` resolution.
    pub(super) fn resolve_refs(&mut self, schema: &JsonVal) -> std::result::Result<(), String> {
        self.converter.extra_roots.push(schema.clone());
        resolve_refs_walk_with_roots(
            schema,
            schema,
            &self.converter.extra_roots,
            &mut self.converter.refs,
            0,
        )
    }

    /// Resolves refs and returns `true` when `schema` resolves to a JSON string.
    pub(super) fn resolves_to_string(&self, schema: &JsonVal) -> bool {
        let mut refs = self.converter.refs.clone();
        let _ =
            resolve_refs_walk_with_roots(schema, schema, &self.converter.extra_roots, &mut refs, 0);
        let mut visited = BTreeSet::new();
        check_resolves_to_string(schema, &refs, &mut visited, 0)
    }

    /// Converts `schema` into GBNF rules named under `name` and returns the root rule symbol.
    pub(super) fn add_schema(
        &mut self,
        name: &str,
        schema: &JsonVal,
        raw: bool,
    ) -> std::result::Result<String, String> {
        self.converter.resolve(schema)?;
        self.converter.raw = raw;
        let res = self.converter.visit(schema, name);
        self.converter.raw = false;
        res
    }

    /// Returns the accumulated GBNF rules map.
    pub(super) const fn rules(&self) -> &BTreeMap<String, String> {
        &self.converter.rules
    }
}

fn convert_to_gbnf(schema_json: &str, force_gbnf: bool) -> std::result::Result<String, String> {
    if !force_gbnf {
        return Err(
            "direct llguidance schema mode is unavailable; request GBNF conversion explicitly"
                .to_owned(),
        );
    }
    let schema = JsonParser::parse_root(schema_json)?;
    let mut converter = SchemaConverter::new();
    converter.resolve(&schema)?;
    converter.visit(&schema, "")?;
    let mut grammar = String::new();
    for (name, rule) in converter.rules {
        let _ = writeln!(grammar, "{name} ::= {rule}");
        if grammar.len() > MAX_GRAMMAR_BYTES {
            return Err("schema grammar exceeds output bound".to_owned());
        }
    }
    Ok(grammar)
}

// Protocol request validation and execution.

fn check_schema_text(operation: &Value) -> Result<&str> {
    let source = field(operation, "SchemaJson")?
        .as_str()
        .ok_or_else(|| invalid("schema field must be text"))?;
    if source.len() > MAX_SCHEMA_BYTES {
        return Err(invalid("schema JSON exceeds source transport bound"));
    }
    Ok(source)
}

fn execute_info(operation: &Value, source: &str) -> Result<Value> {
    exact_fields(operation, &["Kind", "SchemaJson", "Actions"])?;
    let actions = array(operation, "Actions")?;
    if actions.len() > MAX_INFO_ACTIONS {
        return Err(invalid("schema info action count exceeds bound"));
    }
    let schema = JsonParser::parse_root(source).map_err(invalid)?;
    let mut refs = BTreeMap::new();
    let mut trace = Vec::with_capacity(actions.len());
    for action in actions {
        match action.as_str() {
            Some("resolve-refs") => {
                resolve_refs_walk(&schema, &schema, &mut refs, 0).map_err(invalid)?;
                trace.push(json!({"SchemaJson": schema.dump()}));
            }
            Some("resolves-to-string") => {
                let mut visited = BTreeSet::new();
                let rts = check_resolves_to_string(&schema, &refs, &mut visited, 0);
                trace.push(json!({"ResolvesToString": rts}));
            }
            _ => return Err(invalid("unknown schema info action")),
        }
    }
    Ok(json!({"Trace": trace}))
}

fn execute_operation(operation: &Value) -> Result<Value> {
    let source = check_schema_text(operation)?;
    let kind = field(operation, "Kind")?
        .as_str()
        .ok_or_else(|| invalid("schema Kind must be text"))?;
    if kind == "json-schema-info" {
        return execute_info(operation, source);
    }
    exact_fields(operation, &["Kind", "SchemaJson", "ForceGbnf"])?;
    let force_gbnf = field(operation, "ForceGbnf")?
        .as_bool()
        .ok_or_else(|| invalid("ForceGbnf must be boolean"))?;
    if kind == "json-schema-to-grammar" {
        let grammar = convert_to_gbnf(source, force_gbnf).map_err(invalid)?;
        return Ok(json!({"GrammarText": grammar}));
    }
    match convert_to_gbnf(source, force_gbnf) {
        Ok(grammar) => Ok(json!({
            "Grammar": grammar,
            "ExceptionKind": null,
            "Message": "",
        })),
        Err(error) => Ok(json!({
            "Grammar": null,
            "ExceptionKind": "invalid-argument",
            "Message": error,
        })),
    }
}

pub(super) fn handles(request: &Value) -> bool {
    request["Operations"].as_array().is_some_and(|operations| {
        !operations.is_empty()
            && operations.iter().all(|operation| {
                matches!(
                    operation["Kind"].as_str(),
                    Some("json-schema-to-grammar" | "json-schema-convert" | "json-schema-info")
                )
            })
    })
}

pub(super) fn execute(request: &Value) -> Result<Value> {
    exact_fields(request, &["SchemaVersion", "Operations"])?;
    if request["SchemaVersion"].as_u64() != Some(1) {
        return Err(invalid("schema request SchemaVersion must equal 1"));
    }
    let operations = array(request, "Operations")?;
    if operations.is_empty() || operations.len() > MAX_OPS {
        return Err(invalid("schema operation count outside bound"));
    }
    let results = operations
        .iter()
        .map(execute_operation)
        .collect::<Result<Vec<_>>>()?;
    Ok(json!({ "SchemaVersion": 1, "Results": results }))
}

pub(super) fn identity() -> Value {
    json!({
        "Kind": "independent-rust-json-schema",
        "SupportedSourceIds": IDS,
        "UpstreamCommit": super::UPSTREAM,
        "AssertionSources": [
            {"Path": IDS[0], "Sha256": SCHEMA_SOURCE_SHA256},
            {"Path": IDS[1], "Sha256": INTEGRATION_SOURCE_SHA256}
        ],
        "Architecture": "clean-room-ordered-json-pointer-and-schema-gbnf-converter-v1",
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

    const fn target_label(target: Target) -> &'static str {
        match target {
            Target::RustReference => "rust-reference",
            Target::Cpp => "cpp",
            Target::Rust => "rust",
        }
    }

    #[test]
    fn independent_json_schema_passes_grammar_integration_across_all_targets() -> Result<()> {
        for target in [Target::RustReference, Target::Cpp, Target::Rust] {
            let engine = Engine::new(target, None, None, None, None)?;
            let report = crate::grammar_integration::run(&engine)?;
            assert_eq!(report["Id"], IDS[1]);
            assert_eq!(
                report["Status"],
                "pass",
                "{}: {report}",
                target_label(target)
            );
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
    fn independent_json_schema_matches_cases_across_all_targets() -> Result<()> {
        let cases: Value = serde_json::from_str(include_str!("json-schema-cases.json"))
            .map_err(|e| invalid(format!("failed to parse json-schema-cases.json: {e}")))?;
        let shared = cases["Shared"]
            .as_array()
            .ok_or_else(|| invalid("missing Shared cases"))?;
        let cpp_only = cases["CppOnly"]
            .as_array()
            .ok_or_else(|| invalid("missing CppOnly cases"))?;
        let rts_cases = cases["ResolvesToString"]
            .as_array()
            .ok_or_else(|| invalid("missing ResolvesToString cases"))?;
        assert_eq!(shared.len(), 78);
        assert_eq!(cpp_only.len(), 6);
        assert_eq!(rts_cases.len(), 28);

        let rust_ref = Engine::new(Target::RustReference, None, None, None, None)?;
        let cpp = Engine::new(Target::Cpp, None, None, None, None)?;
        let rust = Engine::new(Target::Rust, None, None, None, None)?;

        for case in shared.iter().chain(cpp_only.iter()) {
            let schema_str = case["SchemaJson"]
                .as_str()
                .ok_or_else(|| invalid(format!("missing SchemaJson in case {case}")))?;
            let req = json!({
                "SchemaVersion": 1,
                "Operations": [{
                    "Kind": "json-schema-convert",
                    "SchemaJson": schema_str,
                    "ForceGbnf": true
                }]
            });
            let got_ref = rust_ref.execute_common("json-schema", &req)?;
            let got_cpp = cpp.execute_common("json-schema", &req)?;
            let got_rust = rust.execute_common("json-schema", &req)?;
            assert_eq!(got_ref, got_cpp, "mismatch vs Cpp on case: {case}");
            assert_eq!(got_ref, got_rust, "mismatch vs Rust on case: {case}");
        }

        for case in rts_cases {
            let schema_str = case["SchemaJson"]
                .as_str()
                .ok_or_else(|| invalid(format!("missing SchemaJson in probe {case}")))?;
            let req = json!({
                "SchemaVersion": 1,
                "Operations": [{
                    "Kind": "json-schema-info",
                    "SchemaJson": schema_str,
                    "Actions": ["resolve-refs", "resolves-to-string"]
                }]
            });
            let got_ref = rust_ref.execute_common("json-schema", &req)?;
            let got_cpp = cpp.execute_common("json-schema", &req)?;
            let got_rust = rust.execute_common("json-schema", &req)?;
            assert_eq!(got_ref, got_cpp, "info mismatch vs Cpp on case: {case}");
            assert_eq!(got_ref, got_rust, "info mismatch vs Rust on case: {case}");
        }
        Ok(())
    }

    #[test]
    fn independent_json_schema_rejects_duplicate_keys_oversize_and_false_force_gbnf() -> Result<()>
    {
        let dup = execute(&json!({
            "SchemaVersion": 1,
            "Operations": [{
                "Kind": "json-schema-to-grammar",
                "SchemaJson": "{\"type\":\"object\",\"type\":\"string\"}",
                "ForceGbnf": true
            }]
        }));
        assert!(dup.is_err());

        let dup_convert = execute(&json!({
            "SchemaVersion": 1,
            "Operations": [{
                "Kind": "json-schema-convert",
                "SchemaJson": "{\"a\":1,\"a\":2}",
                "ForceGbnf": true
            }]
        }))?;
        assert_eq!(
            dup_convert["Results"][0]["ExceptionKind"],
            "invalid-argument"
        );

        let huge = format!(
            "{{\"type\":\"string\",\"description\":\"{}\"}}",
            "a".repeat(MAX_SCHEMA_BYTES + 1)
        );
        assert!(
            execute(&json!({
                "SchemaVersion": 1,
                "Operations": [{
                    "Kind": "json-schema-convert",
                    "SchemaJson": huge,
                    "ForceGbnf": true
                }]
            }))
            .is_err()
        );

        let false_gbnf_grammar = execute(&json!({
            "SchemaVersion": 1,
            "Operations": [{
                "Kind": "json-schema-to-grammar",
                "SchemaJson": "{\"type\":\"string\"}",
                "ForceGbnf": false
            }]
        }));
        assert!(false_gbnf_grammar.is_err());

        let false_gbnf_convert = execute(&json!({
            "SchemaVersion": 1,
            "Operations": [{
                "Kind": "json-schema-convert",
                "SchemaJson": "{\"type\":\"string\"}",
                "ForceGbnf": false
            }]
        }))?;
        assert_eq!(
            false_gbnf_convert["Results"][0]["ExceptionKind"],
            "invalid-argument"
        );
        Ok(())
    }
}
