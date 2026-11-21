//! Order-preserving JSON AST parser and arbitrary-precision decimal integers.
//!
//! JSON Schema property declaration order is semantically significant in `LLGuidance`
//! schemas, so object members must preserve their exact source order rather than sorting
//! keys alphabetically.

use std::cmp::Ordering;
use std::fmt::Write as _;

use super::super::{Result, invalid};

const MAX_JSON_DEPTH: usize = 64;

#[derive(Clone, Debug, PartialEq)]
pub(super) enum JsonVal {
    Null,
    Bool(bool),
    Int(BigInt),
    Float(f64),
    String(String),
    Array(Vec<Self>),
    Object(Vec<(String, Self)>),
}

impl JsonVal {
    pub(super) fn get(&self, key: &str) -> Option<&Self> {
        let Self::Object(entries) = self else {
            return None;
        };
        entries.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub(super) fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub(super) const fn as_array(&self) -> Option<&[Self]> {
        match self {
            Self::Array(items) => Some(items.as_slice()),
            _ => None,
        }
    }

    pub(super) const fn as_int(&self) -> Option<&BigInt> {
        match self {
            Self::Int(n) => Some(n),
            _ => None,
        }
    }
}

pub(super) fn escape_json_string(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len() + 2);
    out.push('"');
    for ch in raw.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\x08' => out.push_str("\\b"),
            '\x0c' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Arbitrary-precision signed base-10 integer stored as normalized digits (no leading zeros).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct BigInt {
    pub(super) negative: bool,
    pub(super) digits: Vec<u8>,
}

impl BigInt {
    pub(super) fn zero() -> Self {
        Self {
            negative: false,
            digits: vec![b'0'],
        }
    }

    pub(super) fn one() -> Self {
        Self {
            negative: false,
            digits: vec![b'1'],
        }
    }

    pub(super) fn from_digits(negative: bool, raw: &[u8]) -> Result<Self> {
        let trimmed = raw
            .iter()
            .position(|&b| b != b'0')
            .map_or(&b"0"[..], |idx| &raw[idx..]);
        if !trimmed.iter().all(u8::is_ascii_digit) {
            return Err(invalid("invalid decimal digit in schema integer bound"));
        }
        let is_zero = trimmed == b"0";
        Ok(Self {
            negative: negative && !is_zero,
            digits: trimmed.to_vec(),
        })
    }

    pub(super) fn is_zero(&self) -> bool {
        self.digits.len() == 1 && self.digits[0] == b'0'
    }

    pub(super) fn abs_digits(&self) -> &[u8] {
        &self.digits
    }

    pub(super) fn abs_cmp(&self, other: &Self) -> Ordering {
        self.digits
            .len()
            .cmp(&other.digits.len())
            .then_with(|| self.digits.cmp(&other.digits))
    }

    pub(super) fn increment(&self) -> Self {
        if self.is_zero() {
            return Self::one();
        }
        if self.negative {
            let dec = sub_one_digits(&self.digits);
            let is_zero = dec == b"0";
            Self {
                negative: !is_zero,
                digits: dec,
            }
        } else {
            Self {
                negative: false,
                digits: add_one_digits(&self.digits),
            }
        }
    }

    pub(super) fn decrement(&self) -> Self {
        if self.is_zero() {
            return Self {
                negative: true,
                digits: vec![b'1'],
            };
        }
        if self.negative {
            Self {
                negative: true,
                digits: add_one_digits(&self.digits),
            }
        } else {
            Self {
                negative: false,
                digits: sub_one_digits(&self.digits),
            }
        }
    }

    pub(super) fn negate(&self) -> Self {
        if self.is_zero() {
            Self::zero()
        } else {
            Self {
                negative: !self.negative,
                digits: self.digits.clone(),
            }
        }
    }
}

impl Ord for BigInt {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.negative, other.negative) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (false, false) => self.abs_cmp(other),
            (true, true) => other.abs_cmp(self),
        }
    }
}

impl PartialOrd for BigInt {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl std::fmt::Display for BigInt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.negative {
            f.write_str("-")?;
        }
        for &b in &self.digits {
            write!(f, "{}", char::from(b))?;
        }
        Ok(())
    }
}

pub(super) fn add_one_digits(digits: &[u8]) -> Vec<u8> {
    let mut out = digits.to_vec();
    let mut carry = true;
    for byte in out.iter_mut().rev() {
        if !carry {
            break;
        }
        if *byte < b'9' {
            *byte += 1;
            carry = false;
        } else {
            *byte = b'0';
        }
    }
    if carry {
        out.insert(0, b'1');
    }
    out
}

fn sub_one_digits(digits: &[u8]) -> Vec<u8> {
    if digits == b"0" {
        return vec![b'0'];
    }
    let mut out = digits.to_vec();
    let mut borrow = true;
    for byte in out.iter_mut().rev() {
        if !borrow {
            break;
        }
        if *byte > b'0' {
            *byte -= 1;
            borrow = false;
        } else {
            *byte = b'9';
        }
    }
    out.iter()
        .position(|&b| b != b'0')
        .map_or_else(|| vec![b'0'], |idx| out[idx..].to_vec())
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn skip_ws(&mut self) {
        while self
            .bytes
            .get(self.pos)
            .is_some_and(|b| matches!(*b, b' ' | b'\t' | b'\n' | b'\r'))
        {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_ws();
        self.bytes.get(self.pos).copied()
    }

    fn parse_value(&mut self, depth: usize) -> Result<JsonVal> {
        if depth > MAX_JSON_DEPTH {
            return Err(invalid("JSON AST depth exceeded"));
        }
        let byte = self
            .peek()
            .ok_or_else(|| invalid("unexpected end of JSON input"))?;
        match byte {
            b'{' => self.parse_object(depth),
            b'[' => self.parse_array(depth),
            b'"' => self.parse_string().map(JsonVal::String),
            b't' => self.expect_literal(b"true", JsonVal::Bool(true)),
            b'f' => self.expect_literal(b"false", JsonVal::Bool(false)),
            b'n' => self.expect_literal(b"null", JsonVal::Null),
            b'-' | b'0'..=b'9' => self.parse_number(),
            _ => Err(invalid("unexpected JSON token")),
        }
    }

    fn expect_literal(&mut self, lit: &[u8], val: JsonVal) -> Result<JsonVal> {
        if self.bytes[self.pos..].starts_with(lit) {
            self.pos += lit.len();
            Ok(val)
        } else {
            Err(invalid("invalid JSON keyword"))
        }
    }

    fn parse_object(&mut self, depth: usize) -> Result<JsonVal> {
        self.pos += 1;
        let mut pairs = Vec::new();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonVal::Object(pairs));
        }
        loop {
            if self.peek() != Some(b'"') {
                return Err(invalid("expected JSON object key string"));
            }
            let key = self.parse_string()?;
            if self.peek() != Some(b':') {
                return Err(invalid("expected ':' after JSON object key"));
            }
            self.pos += 1;
            let value = self.parse_value(depth + 1)?;
            pairs.push((key, value));
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(invalid("expected ',' or '}' in JSON object")),
            }
        }
        Ok(JsonVal::Object(pairs))
    }

    fn parse_array(&mut self, depth: usize) -> Result<JsonVal> {
        self.pos += 1;
        let mut items = Vec::new();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(JsonVal::Array(items));
        }
        loop {
            items.push(self.parse_value(depth + 1)?);
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(invalid("expected ',' or ']' in JSON array")),
            }
        }
        Ok(JsonVal::Array(items))
    }

    fn parse_string(&mut self) -> Result<String> {
        self.pos += 1;
        let mut out = String::new();
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            if b == b'"' {
                self.pos += 1;
                return Ok(out);
            }
            if b == b'\\' {
                self.pos += 1;
                let esc = *self
                    .bytes
                    .get(self.pos)
                    .ok_or_else(|| invalid("unterminated JSON string escape"))?;
                self.pos += 1;
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\x08'),
                    b'f' => out.push('\x0c'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => self.push_unicode_escape(&mut out)?,
                    _ => return Err(invalid("invalid JSON escape character")),
                }
            } else {
                let s = std::str::from_utf8(&self.bytes[self.pos..])
                    .map_err(|_| invalid("invalid UTF-8 inside JSON string"))?;
                let ch = s
                    .chars()
                    .next()
                    .ok_or_else(|| invalid("empty UTF-8 slice in JSON string"))?;
                out.push(ch);
                self.pos += ch.len_utf8();
            }
        }
        Err(invalid("unterminated JSON string"))
    }

    fn push_unicode_escape(&mut self, out: &mut String) -> Result<()> {
        let hex_bytes = self
            .bytes
            .get(self.pos..self.pos + 4)
            .ok_or_else(|| invalid("short JSON unicode escape"))?;
        let hex_str =
            std::str::from_utf8(hex_bytes).map_err(|_| invalid("non-UTF8 unicode escape"))?;
        let unit =
            u16::from_str_radix(hex_str, 16).map_err(|_| invalid("invalid hex unicode escape"))?;
        self.pos += 4;
        if (0xd800..=0xdbff).contains(&unit)
            && self.bytes.get(self.pos..self.pos + 2) == Some(b"\\u")
        {
            self.pos += 2;
            let low_bytes = self
                .bytes
                .get(self.pos..self.pos + 4)
                .ok_or_else(|| invalid("short JSON surrogate escape"))?;
            let low_str =
                std::str::from_utf8(low_bytes).map_err(|_| invalid("non-UTF8 unicode escape"))?;
            let low = u16::from_str_radix(low_str, 16)
                .map_err(|_| invalid("invalid low surrogate hex"))?;
            self.pos += 4;
            if !(0xdc00..=0xdfff).contains(&low) {
                return Err(invalid("invalid UTF-16 surrogate pair in JSON"));
            }
            let cp = 0x10000 + ((u32::from(unit) - 0xd800) << 10) + (u32::from(low) - 0xdc00);
            let ch = char::from_u32(cp).ok_or_else(|| invalid("invalid surrogate scalar value"))?;
            out.push(ch);
            return Ok(());
        }
        let ch = char::from_u32(u32::from(unit))
            .ok_or_else(|| invalid("unpaired UTF-16 surrogate in JSON"))?;
        out.push(ch);
        Ok(())
    }

    fn parse_number(&mut self) -> Result<JsonVal> {
        let start = self.pos;
        if self.bytes.get(self.pos) == Some(&b'-') {
            self.pos += 1;
        }
        let digits_start = self.pos;
        while self.bytes.get(self.pos).is_some_and(u8::is_ascii_digit) {
            self.pos += 1;
        }
        if self.pos == digits_start {
            return Err(invalid("expected digit in JSON number"));
        }
        let mut is_float = false;
        if self.bytes.get(self.pos) == Some(&b'.') {
            is_float = true;
            self.pos += 1;
            while self.bytes.get(self.pos).is_some_and(u8::is_ascii_digit) {
                self.pos += 1;
            }
        }
        if self
            .bytes
            .get(self.pos)
            .is_some_and(|&b| b == b'e' || b == b'E')
        {
            is_float = true;
            self.pos += 1;
            if self
                .bytes
                .get(self.pos)
                .is_some_and(|&b| b == b'+' || b == b'-')
            {
                self.pos += 1;
            }
            while self.bytes.get(self.pos).is_some_and(u8::is_ascii_digit) {
                self.pos += 1;
            }
        }
        let slice = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| invalid("invalid UTF-8 JSON number"))?;
        if is_float {
            let value = slice
                .parse::<f64>()
                .map_err(|_| invalid("invalid JSON float"))?;
            Ok(JsonVal::Float(value))
        } else {
            let negative = self.bytes[start] == b'-';
            let raw_digits = if negative {
                &self.bytes[start + 1..self.pos]
            } else {
                &self.bytes[start..self.pos]
            };
            Ok(JsonVal::Int(BigInt::from_digits(negative, raw_digits)?))
        }
    }
}

pub(super) fn parse_ordered_json_prefix(input: &str) -> Result<(JsonVal, usize)> {
    let mut parser = Parser {
        bytes: input.as_bytes(),
        pos: 0,
    };
    let value = parser.parse_value(0)?;
    Ok((value, parser.pos))
}
