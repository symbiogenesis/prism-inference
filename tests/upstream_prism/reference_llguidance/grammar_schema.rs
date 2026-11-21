//! Order-preserving JSON Schema compiler lowering schemas to byte-level CFG productions.
//!
//! Enforces strict declared property ordering, prevents `additionalProperties` from overriding
//! declared keys via a finite-set exclusion DFA, and lowers integer bounds, `format`,
//! `pattern`, tuple `items`, `minLength`/`maxLength`, `minItems`/`maxItems`, `const`, and `enum`.

use std::collections::{BTreeMap, BTreeSet};

use super::super::super::{Result, invalid};
use super::super::json_ast::{BigInt, JsonVal, escape_json_string};
use super::{Builder, ByteMask, Symbol, compile_regex, repeat_optional};

const MAX_LENGTH_BOUND: usize = 512;

struct Helpers {
    ws: u32,
    any_json: u32,
    any_str: u32,
    any_num: u32,
    any_int: u32,
    str_char: u32,
    date_body: u32,
    time_body: u32,
}

impl Helpers {
    fn new(builder: &mut Builder) -> Result<Self> {
        let ws = compile_regex(builder, "[ \t\n\r]*")?;
        let str_char = build_json_char_rule(builder)?;
        let str_body = builder.zero_or_more(Symbol::NonTerm(str_char))?;
        let any_str = builder.new_rule()?;
        let quote = builder.byte_rule(b'"')?;
        builder.add_prod(
            any_str,
            vec![
                Symbol::NonTerm(quote),
                Symbol::NonTerm(str_body),
                Symbol::NonTerm(quote),
            ],
        );

        let any_num = compile_regex(builder, "-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?")?;
        let any_int = compile_regex(builder, "0|-?[1-9][0-9]*")?;

        let any_json = builder.new_rule()?;
        let true_kw = builder.literal_str("true")?;
        let false_kw = builder.literal_str("false")?;
        let null_kw = builder.literal_str("null")?;
        let any_arr = build_any_array(builder, ws, any_json)?;
        let any_obj = build_any_object(builder, ws, any_str, any_json)?;
        for sym in [
            any_str, any_num, true_kw, false_kw, null_kw, any_arr, any_obj,
        ] {
            builder.add_prod(any_json, vec![Symbol::NonTerm(sym)]);
        }

        let date_body =
            compile_regex(builder, "[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])")?;
        let time_body = compile_regex(
            builder,
            "([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9](\\.[0-9]+)?([Zz]|[+-]([01][0-9]|2[0-3]):[0-5][0-9])",
        )?;
        Ok(Self {
            ws,
            any_json,
            any_str,
            any_num,
            any_int,
            str_char,
            date_body,
            time_body,
        })
    }
}

fn build_json_char_rule(builder: &mut Builder) -> Result<u32> {
    let char_rule = builder.new_rule()?;
    let safe = builder.utf8_scalar_ranges(&[
        (0x20, 0x21),
        (0x23, 0x5b),
        (0x5d, 0xd7ff),
        (0xe000, 0x10_ffff),
    ])?;
    builder.add_prod(char_rule, vec![Symbol::NonTerm(safe)]);
    let simple_esc = compile_regex(builder, "\\\\[\"\\\\/bfnrt]")?;
    builder.add_prod(char_rule, vec![Symbol::NonTerm(simple_esc)]);
    let uni_esc = compile_regex(builder, "\\\\u[0-9a-fA-F]{4}")?;
    builder.add_prod(char_rule, vec![Symbol::NonTerm(uni_esc)]);
    Ok(char_rule)
}

fn build_any_array(builder: &mut Builder, ws: u32, item: u32) -> Result<u32> {
    let arr = builder.new_rule()?;
    let lb = builder.byte_rule(b'[')?;
    let rb = builder.byte_rule(b']')?;
    let comma = builder.byte_rule(b',')?;
    let tail = builder.new_rule()?;
    builder.add_prod(tail, Vec::new());
    builder.add_prod(
        tail,
        vec![
            Symbol::NonTerm(tail),
            Symbol::NonTerm(ws),
            Symbol::NonTerm(comma),
            Symbol::NonTerm(ws),
            Symbol::NonTerm(item),
        ],
    );
    builder.add_prod(
        arr,
        vec![
            Symbol::NonTerm(lb),
            Symbol::NonTerm(ws),
            Symbol::NonTerm(rb),
        ],
    );
    builder.add_prod(
        arr,
        vec![
            Symbol::NonTerm(lb),
            Symbol::NonTerm(ws),
            Symbol::NonTerm(item),
            Symbol::NonTerm(tail),
            Symbol::NonTerm(ws),
            Symbol::NonTerm(rb),
        ],
    );
    Ok(arr)
}

fn build_any_object(builder: &mut Builder, ws: u32, key: u32, val: u32) -> Result<u32> {
    let obj = builder.new_rule()?;
    let lc = builder.byte_rule(b'{')?;
    let rc = builder.byte_rule(b'}')?;
    let colon = builder.byte_rule(b':')?;
    let comma = builder.byte_rule(b',')?;
    let pair = builder.new_rule()?;
    builder.add_prod(
        pair,
        vec![
            Symbol::NonTerm(key),
            Symbol::NonTerm(ws),
            Symbol::NonTerm(colon),
            Symbol::NonTerm(ws),
            Symbol::NonTerm(val),
        ],
    );
    let tail = builder.new_rule()?;
    builder.add_prod(tail, Vec::new());
    builder.add_prod(
        tail,
        vec![
            Symbol::NonTerm(tail),
            Symbol::NonTerm(ws),
            Symbol::NonTerm(comma),
            Symbol::NonTerm(ws),
            Symbol::NonTerm(pair),
        ],
    );
    builder.add_prod(
        obj,
        vec![
            Symbol::NonTerm(lc),
            Symbol::NonTerm(ws),
            Symbol::NonTerm(rc),
        ],
    );
    builder.add_prod(
        obj,
        vec![
            Symbol::NonTerm(lc),
            Symbol::NonTerm(ws),
            Symbol::NonTerm(pair),
            Symbol::NonTerm(tail),
            Symbol::NonTerm(ws),
            Symbol::NonTerm(rc),
        ],
    );
    Ok(obj)
}

pub(super) fn compile(builder: &mut Builder, schema: &JsonVal) -> Result<u32> {
    let helpers = Helpers::new(builder)?;
    compile_node(builder, &helpers, schema)
}

fn compile_node(builder: &mut Builder, helpers: &Helpers, schema: &JsonVal) -> Result<u32> {
    match schema {
        JsonVal::Bool(true) => Ok(helpers.any_json),
        JsonVal::Bool(false) => builder.new_rule(),
        JsonVal::Object(_) => compile_object_schema(builder, helpers, schema),
        _ => Err(invalid("JSON Schema must be a boolean or object")),
    }
}

fn compile_object_schema(
    builder: &mut Builder,
    helpers: &Helpers,
    schema: &JsonVal,
) -> Result<u32> {
    if let Some(val) = schema.get("const") {
        return compile_literal_json(builder, helpers, val);
    }
    if let Some(enum_list) = schema.get("enum").and_then(JsonVal::as_array) {
        let out = builder.new_rule()?;
        for val in enum_list {
            let item = compile_literal_json(builder, helpers, val)?;
            builder.add_prod(out, vec![Symbol::NonTerm(item)]);
        }
        return Ok(out);
    }
    let types = allowed_types(schema)?;
    let out = builder.new_rule()?;
    if types.contains(&"null") {
        let kw = builder.literal_str("null")?;
        builder.add_prod(out, vec![Symbol::NonTerm(kw)]);
    }
    if types.contains(&"boolean") {
        let t = builder.literal_str("true")?;
        let f = builder.literal_str("false")?;
        builder.add_prod(out, vec![Symbol::NonTerm(t)]);
        builder.add_prod(out, vec![Symbol::NonTerm(f)]);
    }
    if types.contains(&"string") {
        let r = compile_string_schema(builder, helpers, schema)?;
        builder.add_prod(out, vec![Symbol::NonTerm(r)]);
    }
    if types.contains(&"integer") {
        let r = compile_integer_schema(builder, helpers, schema)?;
        builder.add_prod(out, vec![Symbol::NonTerm(r)]);
    }
    if types.contains(&"number") {
        let r = compile_number_schema(builder, helpers, schema)?;
        builder.add_prod(out, vec![Symbol::NonTerm(r)]);
    }
    if types.contains(&"array") {
        let r = compile_array_schema(builder, helpers, schema)?;
        builder.add_prod(out, vec![Symbol::NonTerm(r)]);
    }
    if types.contains(&"object") {
        let r = compile_obj_value_schema(builder, helpers, schema)?;
        builder.add_prod(out, vec![Symbol::NonTerm(r)]);
    }
    Ok(out)
}

fn allowed_types(schema: &JsonVal) -> Result<Vec<&'static str>> {
    let Some(type_val) = schema.get("type") else {
        return Ok(vec![
            "null", "boolean", "integer", "number", "string", "array", "object",
        ]);
    };
    let mut out = Vec::new();
    match type_val {
        JsonVal::String(s) => out.push(parse_type_name(s)?),
        JsonVal::Array(items) => {
            for item in items {
                let s = item
                    .as_str()
                    .ok_or_else(|| invalid("schema type array element must be string"))?;
                out.push(parse_type_name(s)?);
            }
        }
        _ => return Err(invalid("schema 'type' must be string or array")),
    }
    Ok(out)
}

fn parse_type_name(name: &str) -> Result<&'static str> {
    match name {
        "null" => Ok("null"),
        "boolean" => Ok("boolean"),
        "integer" => Ok("integer"),
        "number" => Ok("number"),
        "string" => Ok("string"),
        "array" => Ok("array"),
        "object" => Ok("object"),
        _ => Err(invalid(format!("unsupported JSON Schema type {name}"))),
    }
}

fn compile_literal_json(builder: &mut Builder, helpers: &Helpers, val: &JsonVal) -> Result<u32> {
    match val {
        JsonVal::Null => builder.literal_str("null"),
        JsonVal::Bool(b) => builder.literal_str(if *b { "true" } else { "false" }),
        JsonVal::Int(n) => builder.literal_str(&n.to_string()),
        JsonVal::Float(f) => builder.literal_str(&f.to_string()),
        JsonVal::String(s) => builder.literal_str(&escape_json_string(s)),
        JsonVal::Array(items) => {
            let rule = builder.new_rule()?;
            let mut rhs = vec![
                Symbol::NonTerm(builder.byte_rule(b'[')?),
                Symbol::NonTerm(helpers.ws),
            ];
            for (idx, item) in items.iter().enumerate() {
                if idx > 0 {
                    rhs.push(Symbol::NonTerm(builder.byte_rule(b',')?));
                    rhs.push(Symbol::NonTerm(helpers.ws));
                }
                let child = compile_literal_json(builder, helpers, item)?;
                rhs.push(Symbol::NonTerm(child));
                rhs.push(Symbol::NonTerm(helpers.ws));
            }
            rhs.push(Symbol::NonTerm(builder.byte_rule(b']')?));
            builder.add_prod(rule, rhs);
            Ok(rule)
        }
        JsonVal::Object(pairs) => {
            let rule = builder.new_rule()?;
            let mut rhs = vec![
                Symbol::NonTerm(builder.byte_rule(b'{')?),
                Symbol::NonTerm(helpers.ws),
            ];
            for (idx, (k, v)) in pairs.iter().enumerate() {
                if idx > 0 {
                    rhs.push(Symbol::NonTerm(builder.byte_rule(b',')?));
                    rhs.push(Symbol::NonTerm(helpers.ws));
                }
                rhs.push(Symbol::NonTerm(
                    builder.literal_str(&escape_json_string(k))?,
                ));
                rhs.push(Symbol::NonTerm(helpers.ws));
                rhs.push(Symbol::NonTerm(builder.byte_rule(b':')?));
                rhs.push(Symbol::NonTerm(helpers.ws));
                rhs.push(Symbol::NonTerm(compile_literal_json(builder, helpers, v)?));
                rhs.push(Symbol::NonTerm(helpers.ws));
            }
            rhs.push(Symbol::NonTerm(builder.byte_rule(b'}')?));
            builder.add_prod(rule, rhs);
            Ok(rule)
        }
    }
}

fn compile_string_schema(
    builder: &mut Builder,
    helpers: &Helpers,
    schema: &JsonVal,
) -> Result<u32> {
    if let Some(fmt) = schema.get("format").and_then(JsonVal::as_str) {
        return compile_string_format(builder, helpers, fmt);
    }
    if let Some(pat) = schema.get("pattern").and_then(JsonVal::as_str) {
        let trimmed_start = pat.strip_prefix('^');
        let trimmed_both = trimmed_start.unwrap_or(pat).strip_suffix('$');
        let core_pat = trimmed_both.unwrap_or_else(|| trimmed_start.unwrap_or(pat));
        let core = compile_regex(builder, core_pat)?;
        let rule = builder.new_rule()?;
        let quote = builder.byte_rule(b'"')?;
        let mut rhs = vec![Symbol::NonTerm(quote)];
        if trimmed_start.is_none() {
            rhs.push(Symbol::NonTerm(
                builder.zero_or_more(Symbol::NonTerm(helpers.str_char))?,
            ));
        }
        rhs.push(Symbol::NonTerm(core));
        if trimmed_both.is_none() {
            rhs.push(Symbol::NonTerm(
                builder.zero_or_more(Symbol::NonTerm(helpers.str_char))?,
            ));
        }
        rhs.push(Symbol::NonTerm(quote));
        builder.add_prod(rule, rhs);
        return Ok(rule);
    }
    let min_len = schema
        .get("minLength")
        .and_then(JsonVal::as_int)
        .map(bigint_to_usize)
        .transpose()?
        .unwrap_or(0);
    let max_len = schema
        .get("maxLength")
        .and_then(JsonVal::as_int)
        .map(bigint_to_usize)
        .transpose()?;
    if min_len == 0 && max_len.is_none() {
        return Ok(helpers.any_str);
    }
    if min_len > MAX_LENGTH_BOUND || max_len.is_some_and(|m| m > MAX_LENGTH_BOUND) {
        return Err(invalid("string length bound exceeds limit"));
    }
    if max_len.is_some_and(|m| m < min_len) {
        return builder.new_rule();
    }
    let quote = builder.byte_rule(b'"')?;
    let rule = builder.new_rule()?;
    let ch = Symbol::NonTerm(helpers.str_char);
    let mut rhs = Vec::with_capacity(min_len + 3);
    rhs.push(Symbol::NonTerm(quote));
    rhs.resize(1 + min_len, ch);
    if let Some(m) = max_len {
        let opt = repeat_optional(builder, ch, m - min_len)?;
        rhs.push(Symbol::NonTerm(opt));
    } else {
        rhs.push(Symbol::NonTerm(builder.zero_or_more(ch)?));
    }
    rhs.push(Symbol::NonTerm(quote));
    builder.add_prod(rule, rhs);
    Ok(rule)
}

fn compile_string_format(builder: &mut Builder, helpers: &Helpers, fmt: &str) -> Result<u32> {
    let quote = builder.byte_rule(b'"')?;
    let body = match fmt {
        "date" => helpers.date_body,
        "time" => helpers.time_body,
        "date-time" => {
            let t = compile_regex(builder, "[Tt]")?;
            let dt = builder.new_rule()?;
            builder.add_prod(
                dt,
                vec![
                    Symbol::NonTerm(helpers.date_body),
                    Symbol::NonTerm(t),
                    Symbol::NonTerm(helpers.time_body),
                ],
            );
            dt
        }
        "uuid" => compile_regex(
            builder,
            "[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        )?,
        _ => return Ok(helpers.any_str),
    };
    let rule = builder.new_rule()?;
    builder.add_prod(
        rule,
        vec![
            Symbol::NonTerm(quote),
            Symbol::NonTerm(body),
            Symbol::NonTerm(quote),
        ],
    );
    Ok(rule)
}

fn bigint_to_usize(n: &BigInt) -> Result<usize> {
    if n.negative {
        return Err(invalid("expected non-negative integer length in schema"));
    }
    Ok(n.to_string().parse::<usize>()?)
}

fn compile_number_schema(
    builder: &mut Builder,
    helpers: &Helpers,
    schema: &JsonVal,
) -> Result<u32> {
    let min_val = json_bound_f64(schema.get("minimum"));
    let exc_min = json_bound_f64(schema.get("exclusiveMinimum"));
    if exc_min.is_some_and(|v| v >= 0.0) || min_val.is_some_and(|v| v > 0.0) {
        return compile_regex(
            builder,
            "([1-9][0-9]*|0\\.[0-9]*[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?",
        );
    }
    Ok(helpers.any_num)
}

fn json_bound_f64(val: Option<&JsonVal>) -> Option<f64> {
    match val? {
        JsonVal::Int(n) => n.to_string().parse::<f64>().ok(),
        JsonVal::Float(f) => Some(*f),
        _ => None,
    }
}

fn compile_integer_schema(
    builder: &mut Builder,
    helpers: &Helpers,
    schema: &JsonVal,
) -> Result<u32> {
    let mut lo: Option<BigInt> = None;
    let mut hi: Option<BigInt> = None;
    if let Some(v) = schema.get("minimum").and_then(JsonVal::as_int) {
        lo = Some(lo.map_or_else(|| v.clone(), |cur| cur.max(v.clone())));
    }
    if let Some(v) = schema.get("exclusiveMinimum").and_then(JsonVal::as_int) {
        let next = v.increment();
        lo = Some(if let Some(cur) = lo {
            cur.max(next)
        } else {
            next
        });
    }
    if let Some(v) = schema.get("maximum").and_then(JsonVal::as_int) {
        hi = Some(hi.map_or_else(|| v.clone(), |cur| cur.min(v.clone())));
    }
    if let Some(v) = schema.get("exclusiveMaximum").and_then(JsonVal::as_int) {
        let prev = v.decrement();
        hi = Some(if let Some(cur) = hi {
            cur.min(prev)
        } else {
            prev
        });
    }
    if lo.is_none() && hi.is_none() {
        return Ok(helpers.any_int);
    }
    if let (Some(l), Some(h)) = (&lo, &hi)
        && l > h
    {
        return builder.new_rule();
    }
    let out = builder.new_rule()?;
    let zero = BigInt::zero();
    let one = BigInt::one();
    if lo.as_ref().is_none_or(|l| *l <= zero) && hi.as_ref().is_none_or(|h| *h >= zero) {
        let z = builder.byte_rule(b'0')?;
        builder.add_prod(out, vec![Symbol::NonTerm(z)]);
    }
    if lo.as_ref().is_none_or(|l| *l < zero) {
        let neg_lo = if hi.as_ref().is_some_and(|h| *h < zero) {
            hi.as_ref().map_or_else(BigInt::one, BigInt::negate)
        } else {
            one.clone()
        };
        let neg_hi = lo.as_ref().map(BigInt::negate);
        if neg_hi.as_ref().is_none_or(|nh| neg_lo <= *nh) {
            let pos = compile_pos_uint_range(
                builder,
                neg_lo.abs_digits(),
                neg_hi.as_ref().map(BigInt::abs_digits),
            )?;
            let minus = builder.byte_rule(b'-')?;
            builder.add_prod(out, vec![Symbol::NonTerm(minus), Symbol::NonTerm(pos)]);
        }
    }
    if hi.as_ref().is_none_or(|h| *h >= one) {
        let pos_lo = lo.as_ref().filter(|l| **l >= one).unwrap_or(&one);
        let pos_hi = hi.as_ref();
        if pos_hi.is_none_or(|ph| pos_lo <= ph) {
            let pos = compile_pos_uint_range(
                builder,
                pos_lo.abs_digits(),
                pos_hi.map(BigInt::abs_digits),
            )?;
            builder.add_prod(out, vec![Symbol::NonTerm(pos)]);
        }
    }
    Ok(out)
}

fn compile_pos_uint_range(
    builder: &mut Builder,
    lo_digits: &[u8],
    hi_digits: Option<&[u8]>,
) -> Result<u32> {
    let any_digit = Symbol::Terminal(ByteMask::range(b'0', b'9'));
    let rule = builder.new_rule()?;
    let la = lo_digits.len();
    match hi_digits {
        None => {
            let upper = vec![b'9'; la];
            let fixed = fixed_len_digits(builder, lo_digits, &upper, 0, true, false)?;
            builder.add_prod(rule, vec![Symbol::NonTerm(fixed)]);
            let first = Symbol::Terminal(ByteMask::range(b'1', b'9'));
            let mut rhs = vec![first];
            for _ in 0..la {
                rhs.push(any_digit);
            }
            rhs.push(Symbol::NonTerm(builder.zero_or_more(any_digit)?));
            builder.add_prod(rule, rhs);
        }
        Some(hi_d) => {
            let lb = hi_d.len();
            if la == lb {
                let fixed = fixed_len_digits(builder, lo_digits, hi_d, 0, true, true)?;
                builder.add_prod(rule, vec![Symbol::NonTerm(fixed)]);
            } else {
                let upper_a = vec![b'9'; la];
                let r_a = fixed_len_digits(builder, lo_digits, &upper_a, 0, true, false)?;
                builder.add_prod(rule, vec![Symbol::NonTerm(r_a)]);
                for len in (la + 1)..lb {
                    let mut rhs = vec![Symbol::Terminal(ByteMask::range(b'1', b'9'))];
                    for _ in 1..len {
                        rhs.push(any_digit);
                    }
                    builder.add_prod(rule, rhs);
                }
                let mut lower_b = vec![b'0'; lb];
                lower_b[0] = b'1';
                let r_b = fixed_len_digits(builder, &lower_b, hi_d, 0, false, true)?;
                builder.add_prod(rule, vec![Symbol::NonTerm(r_b)]);
            }
        }
    }
    Ok(rule)
}

fn fixed_len_digits(
    builder: &mut Builder,
    lo: &[u8],
    hi: &[u8],
    pos: usize,
    tight_lo: bool,
    tight_hi: bool,
) -> Result<u32> {
    let any_digit = Symbol::Terminal(ByteMask::range(b'0', b'9'));
    let rule = builder.new_rule()?;
    if pos >= lo.len() {
        builder.add_prod(rule, Vec::new());
        return Ok(rule);
    }
    let min_d = if tight_lo {
        lo[pos]
    } else if pos == 0 {
        b'1'
    } else {
        b'0'
    };
    let max_d = if tight_hi { hi[pos] } else { b'9' };
    if min_d > max_d {
        return Ok(rule);
    }
    if !tight_lo && !tight_hi {
        let mut rhs = vec![Symbol::Terminal(ByteMask::range(min_d, max_d))];
        for _ in (pos + 1)..lo.len() {
            rhs.push(any_digit);
        }
        builder.add_prod(rule, rhs);
        return Ok(rule);
    }
    let mut mid_start = min_d;
    if tight_lo && min_d == max_d {
        let next = fixed_len_digits(builder, lo, hi, pos + 1, true, tight_hi)?;
        builder.add_prod(
            rule,
            vec![
                Symbol::Terminal(ByteMask::single(min_d)),
                Symbol::NonTerm(next),
            ],
        );
        return Ok(rule);
    }
    if tight_lo {
        let next = fixed_len_digits(builder, lo, hi, pos + 1, true, false)?;
        builder.add_prod(
            rule,
            vec![
                Symbol::Terminal(ByteMask::single(min_d)),
                Symbol::NonTerm(next),
            ],
        );
        mid_start += 1;
    }
    let mut mid_end = max_d;
    if tight_hi {
        let next = fixed_len_digits(builder, lo, hi, pos + 1, false, true)?;
        builder.add_prod(
            rule,
            vec![
                Symbol::Terminal(ByteMask::single(max_d)),
                Symbol::NonTerm(next),
            ],
        );
        mid_end -= 1;
    }
    if mid_start <= mid_end {
        let next = fixed_len_digits(builder, lo, hi, pos + 1, false, false)?;
        builder.add_prod(
            rule,
            vec![
                Symbol::Terminal(ByteMask::range(mid_start, mid_end)),
                Symbol::NonTerm(next),
            ],
        );
    }
    Ok(rule)
}

fn compile_array_schema(builder: &mut Builder, helpers: &Helpers, schema: &JsonVal) -> Result<u32> {
    let min_items = schema
        .get("minItems")
        .and_then(JsonVal::as_int)
        .map(bigint_to_usize)
        .transpose()?
        .unwrap_or(0);
    let max_items = schema
        .get("maxItems")
        .and_then(JsonVal::as_int)
        .map(bigint_to_usize)
        .transpose()?;
    if min_items > MAX_LENGTH_BOUND || max_items.is_some_and(|m| m > MAX_LENGTH_BOUND) {
        return Err(invalid("array item bound exceeds limit"));
    }
    if max_items.is_some_and(|m| m < min_items) {
        return builder.new_rule();
    }
    let lb = builder.byte_rule(b'[')?;
    let rb = builder.byte_rule(b']')?;
    let comma = builder.byte_rule(b',')?;
    if let Some(tuple) = schema.get("items").and_then(JsonVal::as_array) {
        let out = builder.new_rule()?;
        let mut rhs = vec![Symbol::NonTerm(lb), Symbol::NonTerm(helpers.ws)];
        for (idx, sub) in tuple.iter().enumerate() {
            if idx > 0 {
                rhs.push(Symbol::NonTerm(comma));
                rhs.push(Symbol::NonTerm(helpers.ws));
            }
            rhs.push(Symbol::NonTerm(compile_node(builder, helpers, sub)?));
            rhs.push(Symbol::NonTerm(helpers.ws));
        }
        rhs.push(Symbol::NonTerm(rb));
        builder.add_prod(out, rhs);
        return Ok(out);
    }
    let item = match schema.get("items") {
        Some(sub) => compile_node(builder, helpers, sub)?,
        None => helpers.any_json,
    };
    let out = builder.new_rule()?;
    if min_items == 0 && max_items.is_none() {
        return build_any_array(builder, helpers.ws, item);
    }
    let sep_item = builder.new_rule()?;
    builder.add_prod(
        sep_item,
        vec![
            Symbol::NonTerm(helpers.ws),
            Symbol::NonTerm(comma),
            Symbol::NonTerm(helpers.ws),
            Symbol::NonTerm(item),
        ],
    );
    if min_items == 0 {
        builder.add_prod(
            out,
            vec![
                Symbol::NonTerm(lb),
                Symbol::NonTerm(helpers.ws),
                Symbol::NonTerm(rb),
            ],
        );
    }
    let mut rhs = vec![
        Symbol::NonTerm(lb),
        Symbol::NonTerm(helpers.ws),
        Symbol::NonTerm(item),
    ];
    let needed_extra = min_items.saturating_sub(1);
    rhs.resize(rhs.len() + needed_extra, Symbol::NonTerm(sep_item));
    if let Some(max_n) = max_items {
        let extra = max_n.saturating_sub(min_items.max(1));
        let opt = repeat_optional(builder, Symbol::NonTerm(sep_item), extra)?;
        rhs.push(Symbol::NonTerm(opt));
    } else {
        rhs.push(Symbol::NonTerm(
            builder.zero_or_more(Symbol::NonTerm(sep_item))?,
        ));
    }
    rhs.push(Symbol::NonTerm(helpers.ws));
    rhs.push(Symbol::NonTerm(rb));
    builder.add_prod(out, rhs);
    Ok(out)
}

fn build_additional_props_tails(
    builder: &mut Builder,
    helpers: &Helpers,
    additional_schema: Option<u32>,
    declared_keys: &[&str],
) -> Result<(u32, u32)> {
    let add_tail_no_comma = builder.new_rule()?;
    let add_tail_with_comma = builder.new_rule()?;
    let Some(add_val) = additional_schema else {
        builder.add_prod(add_tail_no_comma, Vec::new());
        builder.add_prod(add_tail_with_comma, Vec::new());
        return Ok((add_tail_no_comma, add_tail_with_comma));
    };
    let add_key = build_excluded_json_key(builder, helpers, declared_keys)?;
    let colon = builder.byte_rule(b':')?;
    let comma = builder.byte_rule(b',')?;
    let pair = builder.new_rule()?;
    builder.add_prod(
        pair,
        vec![
            Symbol::NonTerm(add_key),
            Symbol::NonTerm(helpers.ws),
            Symbol::NonTerm(colon),
            Symbol::NonTerm(helpers.ws),
            Symbol::NonTerm(add_val),
        ],
    );
    let more = builder.new_rule()?;
    builder.add_prod(more, Vec::new());
    builder.add_prod(
        more,
        vec![
            Symbol::NonTerm(more),
            Symbol::NonTerm(helpers.ws),
            Symbol::NonTerm(comma),
            Symbol::NonTerm(helpers.ws),
            Symbol::NonTerm(pair),
        ],
    );
    builder.add_prod(add_tail_no_comma, Vec::new());
    builder.add_prod(
        add_tail_no_comma,
        vec![Symbol::NonTerm(pair), Symbol::NonTerm(more)],
    );
    builder.add_prod(add_tail_with_comma, Vec::new());
    builder.add_prod(
        add_tail_with_comma,
        vec![
            Symbol::NonTerm(helpers.ws),
            Symbol::NonTerm(comma),
            Symbol::NonTerm(helpers.ws),
            Symbol::NonTerm(pair),
            Symbol::NonTerm(more),
        ],
    );
    Ok((add_tail_no_comma, add_tail_with_comma))
}

fn compile_obj_value_schema(
    builder: &mut Builder,
    helpers: &Helpers,
    schema: &JsonVal,
) -> Result<u32> {
    let props = match schema.get("properties") {
        Some(JsonVal::Object(pairs)) => pairs.as_slice(),
        None => return build_any_object(builder, helpers.ws, helpers.any_str, helpers.any_json),
        Some(_) => return Err(invalid("schema 'properties' must be an object")),
    };
    let mut required = BTreeSet::new();
    if let Some(req_arr) = schema.get("required").and_then(JsonVal::as_array) {
        for item in req_arr {
            let name = item
                .as_str()
                .ok_or_else(|| invalid("schema 'required' entry must be string"))?;
            required.insert(name.to_owned());
        }
    }
    let additional_schema = match schema.get("additionalProperties") {
        Some(JsonVal::Bool(false)) => None,
        Some(JsonVal::Bool(true)) | None => Some(helpers.any_json),
        Some(other) => Some(compile_node(builder, helpers, other)?),
    };
    let declared_keys: Vec<&str> = props.iter().map(|(k, _)| k.as_str()).collect();
    let (mut next_no_comma, mut next_with_comma) =
        build_additional_props_tails(builder, helpers, additional_schema, &declared_keys)?;
    let colon = builder.byte_rule(b':')?;
    let comma = builder.byte_rule(b',')?;
    for (key, sub) in props.iter().rev() {
        let cur_no_comma = builder.new_rule()?;
        let cur_with_comma = builder.new_rule()?;
        let val_rule = compile_node(builder, helpers, sub)?;
        let key_lit = builder.literal_str(&escape_json_string(key))?;
        let emit_no_comma = vec![
            Symbol::NonTerm(key_lit),
            Symbol::NonTerm(helpers.ws),
            Symbol::NonTerm(colon),
            Symbol::NonTerm(helpers.ws),
            Symbol::NonTerm(val_rule),
            Symbol::NonTerm(next_with_comma),
        ];
        builder.add_prod(cur_no_comma, emit_no_comma);
        let emit_with_comma = vec![
            Symbol::NonTerm(helpers.ws),
            Symbol::NonTerm(comma),
            Symbol::NonTerm(helpers.ws),
            Symbol::NonTerm(key_lit),
            Symbol::NonTerm(helpers.ws),
            Symbol::NonTerm(colon),
            Symbol::NonTerm(helpers.ws),
            Symbol::NonTerm(val_rule),
            Symbol::NonTerm(next_with_comma),
        ];
        builder.add_prod(cur_with_comma, emit_with_comma);
        if !required.contains(key) {
            builder.add_prod(cur_no_comma, vec![Symbol::NonTerm(next_no_comma)]);
            builder.add_prod(cur_with_comma, vec![Symbol::NonTerm(next_with_comma)]);
        }
        next_no_comma = cur_no_comma;
        next_with_comma = cur_with_comma;
    }
    let obj = builder.new_rule()?;
    let lc = builder.byte_rule(b'{')?;
    let rc = builder.byte_rule(b'}')?;
    builder.add_prod(
        obj,
        vec![
            Symbol::NonTerm(lc),
            Symbol::NonTerm(helpers.ws),
            Symbol::NonTerm(next_no_comma),
            Symbol::NonTerm(helpers.ws),
            Symbol::NonTerm(rc),
        ],
    );
    Ok(obj)
}

#[derive(Default)]
struct Trie {
    edges: BTreeMap<u8, usize>,
    terminal: bool,
}

fn build_excluded_json_key(
    builder: &mut Builder,
    helpers: &Helpers,
    excluded: &[&str],
) -> Result<u32> {
    if excluded.is_empty() {
        return Ok(helpers.any_str);
    }
    let mut nodes = vec![Trie::default()];
    for key in excluded {
        let mut cur = 0_usize;
        for &b in key.as_bytes() {
            if let Some(&next) = nodes[cur].edges.get(&b) {
                cur = next;
            } else {
                let id = nodes.len();
                nodes.push(Trie::default());
                nodes[cur].edges.insert(b, id);
                cur = id;
            }
        }
        nodes[cur].terminal = true;
    }
    let quote = builder.byte_rule(b'"')?;
    let free_tail = builder.new_rule()?;
    let free_chars = builder.zero_or_more(Symbol::NonTerm(helpers.str_char))?;
    builder.add_prod(
        free_tail,
        vec![Symbol::NonTerm(free_chars), Symbol::NonTerm(quote)],
    );
    let mut state_rules = Vec::with_capacity(nodes.len());
    for _ in 0..nodes.len() {
        state_rules.push(builder.new_rule()?);
    }
    for (idx, node) in nodes.iter().enumerate() {
        let rule = state_rules[idx];
        if !node.terminal {
            builder.add_prod(rule, vec![Symbol::NonTerm(quote)]);
        }
        for (&byte, &next) in &node.edges {
            if !node_byte_needs_escape(byte) {
                let b_rule = builder.byte_rule(byte)?;
                builder.add_prod(
                    rule,
                    vec![Symbol::NonTerm(b_rule), Symbol::NonTerm(state_rules[next])],
                );
            }
        }
        let diverge_first = build_diverge_first_byte(builder, &node.edges)?;
        builder.add_prod(
            rule,
            vec![Symbol::NonTerm(diverge_first), Symbol::NonTerm(free_tail)],
        );
    }
    let out = builder.new_rule()?;
    builder.add_prod(
        out,
        vec![Symbol::NonTerm(quote), Symbol::NonTerm(state_rules[0])],
    );
    Ok(out)
}

const fn node_byte_needs_escape(b: u8) -> bool {
    b < 0x20 || b == b'"' || b == b'\\'
}

fn build_diverge_first_byte(builder: &mut Builder, edges: &BTreeMap<u8, usize>) -> Result<u32> {
    let rule = builder.new_rule()?;
    let mut mask = ByteMask::empty();
    for byte in 0x20..=0x7f_u8 {
        if byte != b'"' && byte != b'\\' && !edges.contains_key(&byte) {
            mask.set(byte);
        }
    }
    if mask != ByteMask::empty() {
        builder.add_prod(rule, vec![Symbol::Terminal(mask)]);
    }
    let multi = builder.utf8_scalar_ranges(&[(0x80, 0xd7ff), (0xe000, 0x10_ffff)])?;
    builder.add_prod(rule, vec![Symbol::NonTerm(multi)]);
    let simple_esc = compile_regex(builder, "\\\\[\"\\\\/bfnrt]")?;
    builder.add_prod(rule, vec![Symbol::NonTerm(simple_esc)]);
    let uni_esc = compile_regex(builder, "\\\\u[0-9a-fA-F]{4}")?;
    builder.add_prod(rule, vec![Symbol::NonTerm(uni_esc)]);
    Ok(rule)
}
