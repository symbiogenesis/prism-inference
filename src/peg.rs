//! Bounded production PEG construction, streaming parsing and arena persistence.
//!
//! Port of the MIT-licensed Prism `common/peg-parser.cpp` contract at
//! `b87ca7225456b06b0975b8239c66b5aecf07e37c`. Attribution is retained in
//! `tools/prism/upstream/LICENSE`. Declarative graphs are caller input; this
//! module contains no source-test schedules or expected parse observations.

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{Result, invalid};

#[path = "peg_capture.rs"]
mod capture;
#[path = "peg_grammar.rs"]
mod grammar;
#[path = "peg_parse.rs"]
mod parse;
#[path = "peg_tools.rs"]
mod tools;
pub use capture::{Capture, CapturedParse};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Range {
    start: u32,
    end: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Parser {
    Epsilon,
    Start,
    End,
    Any,
    Space,
    Literal {
        literal: String,
    },
    Sequence {
        children: Vec<usize>,
    },
    Choice {
        children: Vec<usize>,
    },
    Repetition {
        child: usize,
        min_count: i32,
        max_count: i32,
    },
    And {
        child: usize,
    },
    Not {
        child: usize,
    },
    Chars {
        pattern: String,
        ranges: Vec<Range>,
        negated: bool,
        min_count: i32,
        max_count: i32,
    },
    String {
        delimiter: String,
    },
    Until {
        delimiters: Vec<String>,
    },
    Schema {
        child: usize,
        name: String,
        schema: Value,
        #[serde(skip)]
        schema_source: Option<String>,
        raw: bool,
    },
    Rule {
        name: String,
        child: usize,
        trigger: bool,
    },
    Ref {
        name: String,
    },
    Atomic {
        child: usize,
    },
    Tag {
        child: usize,
        tag: String,
    },
    Gbnf {
        child: usize,
        grammar: String,
    },
    Ac {
        child: usize,
        delimiters: Vec<String>,
    },
}

/// Immutable parser arena with explicit rule references and serialized IDs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Arena {
    parsers: Vec<Parser>,
    rules: BTreeMap<String, usize>,
    root: usize,
    #[serde(skip)]
    retained_bytes: usize,
}

/// Streaming parse status and byte extent, independent of UTF-8 scalar length.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ParseResult {
    /// Zero means failure, one success, and two an incomplete input prefix.
    #[serde(rename = "Type")]
    pub result_type: u8,
    /// Initial input byte offset.
    #[serde(rename = "Start")]
    pub start: usize,
    /// End byte offset reached by this result.
    #[serde(rename = "End")]
    pub end: usize,
}
impl ParseResult {
    const fn new(result_type: u8, start: usize, end: usize) -> Self {
        Self {
            result_type,
            start,
            end,
        }
    }
}

fn text(value: &Value) -> Result<&str> {
    let text = value
        .as_str()
        .ok_or_else(|| invalid("PEG string argument missing"))?;
    if text.len() > 65536 {
        return Err(invalid("PEG string argument exceeds bound"));
    }
    Ok(text)
}
fn array(value: &Value) -> Result<&[Value]> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| invalid("PEG array argument missing"))
}
fn number(value: &Value) -> Result<usize> {
    usize::try_from(
        value
            .as_u64()
            .ok_or_else(|| invalid("PEG ID must be unsigned"))?,
    )
    .map_err(Into::into)
}
fn name(value: &str) -> String {
    let mut out = String::new();
    let mut invalid_run = false;
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' {
            out.push(ch);
            invalid_run = false;
        } else if !invalid_run {
            out.push('-');
            invalid_run = true;
        }
    }
    out
}

impl Arena {
    /// Builds a generic declarative PEG graph with scoped locals and named rules.
    ///
    /// # Errors
    /// Rejects malformed operations, unknown references and graph resource limits.
    pub fn from_graph(graph: &Value) -> Result<Self> {
        let mut arena = Self {
            parsers: Vec::new(),
            rules: BTreeMap::new(),
            root: usize::MAX,
            retained_bytes: 0,
        };
        arena.root = arena.graph(graph, &HashMap::new(), 0)?;
        arena.resolve()?;
        Ok(arena)
    }
    /// Restores a bounded production arena and validates all parser IDs.
    /// # Errors
    /// Rejects malformed serialization, unsupported variants and invalid IDs.
    pub fn load(serialized: &str) -> Result<Self> {
        if serialized.len() > 8 * 1024 * 1024 {
            return Err(invalid("PEG serialized arena exceeds bound"));
        }
        let arena: Self = serde_json::from_str(serialized)?;
        arena.validate()?;
        Ok(arena)
    }
    /// Serializes all parser variants, named rules and the root ID.
    /// # Errors
    /// Returns JSON serialization or output-resource-limit errors.
    pub fn save(&self) -> Result<String> {
        let value = serde_json::to_string(self)?;
        if value.len() > 8 * 1024 * 1024 {
            return Err(invalid("PEG serialized arena exceeds bound"));
        }
        Ok(value)
    }
    /// Returns the validated root parser ID.
    #[must_use]
    pub const fn root(&self) -> usize {
        self.root
    }
    fn add(&mut self, parser: Parser) -> Result<usize> {
        if self.parsers.len() >= 16384 {
            return Err(invalid("PEG node count exceeds bound"));
        }
        let retained_bytes = serde_json::to_vec(&parser)?.len();
        if self.retained_bytes + retained_bytes > 8 * 1024 * 1024 {
            return Err(invalid("PEG retained arena exceeds byte bound"));
        }
        self.retained_bytes += retained_bytes;
        let id = self.parsers.len();
        self.parsers.push(parser);
        Ok(id)
    }
    fn literal(&mut self, value: &str) -> Result<usize> {
        if value.len() > 65536 {
            return Err(invalid("PEG literal exceeds byte bound"));
        }
        self.add(Parser::Literal {
            literal: value.to_owned(),
        })
    }
    fn combine(&mut self, children: Vec<usize>, choice: bool) -> Result<usize> {
        let mut flat = Vec::new();
        for child in children {
            match (&self.parsers[child], choice) {
                (Parser::Sequence { children }, false) | (Parser::Choice { children }, true) => {
                    flat.extend(children);
                }
                _ => flat.push(child),
            }
        }
        if flat.len() > 16384 {
            return Err(invalid("PEG child count exceeds bound"));
        }
        self.add(if choice {
            Parser::Choice { children: flat }
        } else {
            Parser::Sequence { children: flat }
        })
    }
    fn repeat(&mut self, child: usize, min_count: i32, max_count: i32) -> Result<usize> {
        if !(0..=65536).contains(&min_count) || !(-1..=65536).contains(&max_count) {
            return Err(invalid("PEG repetition extent exceeds bound"));
        }
        self.add(Parser::Repetition {
            child,
            min_count,
            max_count,
        })
    }
    fn chars(&mut self, pattern: &str, min_count: i32, max_count: i32) -> Result<usize> {
        if pattern.len() > 65536
            || !(0..=65536).contains(&min_count)
            || !(-1..=65536).contains(&max_count)
        {
            return Err(invalid("PEG class extent exceeds bound"));
        }
        let (ranges, negated) = parse::classes(pattern)?;
        self.add(Parser::Chars {
            pattern: pattern.to_owned(),
            ranges,
            negated,
            min_count,
            max_count,
        })
    }
    fn rule(&mut self, name: &str, child: usize, trigger: bool) -> Result<usize> {
        let name = name.to_owned();
        let id = self.add(Parser::Rule {
            name: name.clone(),
            child,
            trigger,
        })?;
        self.rules.insert(name.clone(), id);
        self.add(Parser::Ref { name })
    }
    fn begin_rule(&mut self, name: &str, trigger: bool) -> Result<Option<usize>> {
        if self.rules.contains_key(name) {
            return Ok(None);
        }
        let child = self.add(Parser::Any)?;
        let id = self.add(Parser::Rule {
            name: name.to_owned(),
            child,
            trigger,
        })?;
        self.rules.insert(name.to_owned(), id);
        Ok(Some(id))
    }
    fn graph(
        &mut self,
        graph: &Value,
        outer: &HashMap<String, usize>,
        depth: usize,
    ) -> Result<usize> {
        if depth > 48 {
            return Err(invalid("PEG graph depth exceeds bound"));
        }
        let mut locals = outer.clone();
        for statement in array(&graph["Statements"])? {
            let id = self.expression(&statement["Expr"], &locals, depth + 1)?;
            if let Some(local) = statement.get("Let") {
                locals.insert(text(local)?.to_owned(), id);
            }
        }
        self.expression(&graph["Root"], &locals, depth + 1)
    }
    #[allow(
        clippy::too_many_lines,
        reason = "Keep declarative builder operation ordering identical to the production bridge"
    )]
    fn expression(
        &mut self,
        expression: &Value,
        locals: &HashMap<String, usize>,
        depth: usize,
    ) -> Result<usize> {
        if depth > 48 {
            return Err(invalid("PEG expression depth exceeds bound"));
        }
        if let Some(reference) = expression.get("Ref") {
            return locals
                .get(text(reference)?)
                .copied()
                .ok_or_else(|| invalid("unknown PEG local"));
        }
        let op = text(&expression["Op"])?;
        let args = array(&expression["Args"])?;
        let arg = |index| {
            args.get(index)
                .ok_or_else(|| invalid("PEG operation argument missing"))
        };
        match op {
            "standard_json_tools" | "standard_constructed_tools" => self.tools(op, args),
            "reasoning_block"
            | "reasoning"
            | "content"
            | "tool"
            | "tool_open"
            | "tool_close"
            | "tool_id"
            | "tool_name"
            | "tool_args"
            | "tool_arg"
            | "tool_arg_open"
            | "tool_arg_close"
            | "tool_arg_name"
            | "tool_arg_string_value"
            | "tool_arg_json_value" => {
                let child = self.expression(arg(0)?, locals, depth + 1)?;
                let tag = match op {
                    "tool_arg_json_value" => "tool-arg-value".to_owned(),
                    _ => op.replace('_', "-"),
                };
                let child = self.add(Parser::Tag { child, tag })?;
                if matches!(
                    op,
                    "tool_open"
                        | "tool_close"
                        | "tool_id"
                        | "tool_name"
                        | "tool_arg_open"
                        | "tool_arg_close"
                        | "tool_arg_name"
                ) {
                    self.add(Parser::Atomic { child })
                } else {
                    Ok(child)
                }
            }
            "start" => self.add(Parser::Start),
            "rest" => self.add(Parser::Until {
                delimiters: Vec::new(),
            }),
            "permute" => {
                let children = array(arg(1)?)?
                    .iter()
                    .map(|child| self.expression(child, locals, depth + 1))
                    .collect::<Result<Vec<_>>>()?;
                if children.is_empty() {
                    self.add(Parser::Epsilon)
                } else if children.len() == 1 || children.len() > 6 {
                    self.combine(children, false)
                } else {
                    let mask = (1_u8 << children.len()) - 1;
                    self.permute(text(arg(0)?)?, &children, mask, &mut HashMap::new())
                }
            }
            "schema" => {
                let child = self.expression(arg(0)?, locals, depth + 1)?;
                let schema = serde_json::from_str(text(&arg(2)?["Json"])?)?;
                self.add(Parser::Schema {
                    child,
                    name: text(arg(1)?)?.into(),
                    schema,
                    schema_source: Some(text(&arg(2)?["Json"])?.to_owned()),
                    raw: args.get(3).and_then(Value::as_bool).unwrap_or(false),
                })
            }
            "choice-empty" => self.combine(Vec::new(), true),
            "or-assign" => {
                let left = self.expression(arg(0)?, locals, depth + 1)?;
                let right = self.expression(arg(1)?, locals, depth + 1)?;
                self.combine(vec![left, right], true)
            }
            "peek" | "negate" => {
                let child = self.expression(arg(0)?, locals, depth + 1)?;
                self.add(if op == "peek" {
                    Parser::And { child }
                } else {
                    Parser::Not { child }
                })
            }
            "repeat" => {
                let child = self.expression(arg(0)?, locals, depth + 1)?;
                let minimum = i32::try_from(
                    arg(1)?
                        .as_i64()
                        .ok_or_else(|| invalid("PEG repeat minimum missing"))?,
                )?;
                let maximum = args.get(2).map_or(Ok(minimum), |arg| {
                    i32::try_from(
                        arg.as_i64()
                            .ok_or_else(|| invalid("PEG repeat maximum missing"))?,
                    )
                    .map_err(crate::Error::from)
                })?;
                self.repeat(child, minimum, maximum)
            }
            "trigger_rule" => {
                let child = self.expression(arg(1)?, locals, depth + 1)?;
                self.rule(text(arg(0)?)?, child, true)
            }
            "eps" => self.add(Parser::Epsilon),
            "end" => self.add(Parser::End),
            "any" => self.add(Parser::Any),
            "space" => self.add(Parser::Space),
            "literal" => self.literal(text(arg(0)?)?),
            "ref" => self.add(Parser::Ref {
                name: text(arg(0)?)?.to_owned(),
            }),
            "chars" => self.chars(
                text(arg(0)?)?,
                args.get(1).map_or(Ok(1), |v| {
                    i32::try_from(v.as_i64().ok_or_else(|| invalid("PEG minimum missing"))?)
                        .map_err(crate::Error::from)
                })?,
                args.get(2).map_or(Ok(-1), |v| {
                    i32::try_from(v.as_i64().ok_or_else(|| invalid("PEG maximum missing"))?)
                        .map_err(crate::Error::from)
                })?,
            ),
            "until" => self.add(Parser::Until {
                delimiters: vec![text(arg(0)?)?.to_owned()],
            }),
            "until_one_of" => self.add(Parser::Until {
                delimiters: array(arg(0)?)?
                    .iter()
                    .map(|v| text(v).map(str::to_owned))
                    .collect::<Result<_>>()?,
            }),
            "string_content" => self.add(Parser::String {
                delimiter: char::from(u8::try_from(number(arg(0)?)?)?).to_string(),
            }),
            "sequence" | "choice" => {
                let mut children = Vec::new();
                for child in array(arg(0)?)? {
                    children.push(self.expression(child, locals, depth + 1)?);
                }
                self.combine(children, op == "choice")
            }
            "add" | "or" | "space-add" => {
                let (left, right) = if arg(0)?.is_string() {
                    let right = self.expression(arg(1)?, locals, depth + 1)?;
                    let left = self.literal(text(arg(0)?)?)?;
                    (left, right)
                } else {
                    let left = self.expression(arg(0)?, locals, depth + 1)?;
                    let right = if arg(1)?.is_string() {
                        self.literal(text(arg(1)?)?)?
                    } else {
                        self.expression(arg(1)?, locals, depth + 1)?
                    };
                    (left, right)
                };
                let children = if op == "space-add" {
                    let space = self.add(Parser::Space)?;
                    vec![left, space, right]
                } else {
                    vec![left, right]
                };
                self.combine(children, op == "or")
            }
            "rule-lazy" => {
                let rule = name(text(arg(0)?)?);
                let trigger = args.get(2).and_then(Value::as_bool).unwrap_or(false);
                if self.begin_rule(&rule, trigger)?.is_none() {
                    return self.add(Parser::Ref { name: rule });
                }
                let child = self.graph(arg(1)?, locals, depth + 1)?;
                self.rule(&rule, child, trigger)
            }
            "rule" => {
                let rule = name(text(arg(0)?)?);
                let child = self.expression(arg(1)?, locals, depth + 1)?;
                self.rule(
                    &rule,
                    child,
                    args.get(2).and_then(Value::as_bool).unwrap_or(false),
                )
            }
            "tag" => {
                let child = self.expression(arg(1)?, locals, depth + 1)?;
                self.add(Parser::Tag {
                    child,
                    tag: text(arg(0)?)?.to_owned(),
                })
            }
            "json_member" => {
                let child = self.expression(arg(1)?, locals, depth + 1)?;
                let ws = self.add(Parser::Space)?;
                let key = self.literal(&format!("\"{}\"", text(arg(0)?)?))?;
                let colon = self.literal(":")?;
                self.combine(vec![key, ws, colon, ws, child], false)
            }
            "one_or_more" | "zero_or_more" | "optional" | "atomic" | "gbnf" | "ac" => {
                let child = self.expression(arg(0)?, locals, depth + 1)?;
                match op {
                    "one_or_more" => self.repeat(child, 1, -1),
                    "zero_or_more" => self.repeat(child, 0, -1),
                    "optional" => self.repeat(child, 0, 1),
                    "atomic" => self.add(Parser::Atomic { child }),
                    "gbnf" => self.add(Parser::Gbnf {
                        child,
                        grammar: text(arg(1)?)?.to_owned(),
                    }),
                    _ => {
                        let delimiters = if arg(1)?.is_string() {
                            vec![text(arg(1)?)?.to_owned()]
                        } else {
                            array(arg(1)?)?
                                .iter()
                                .map(|v| text(v).map(str::to_owned))
                                .collect::<Result<_>>()?
                        };
                        if delimiters.is_empty() {
                            return Err(invalid("PEG ac delimiters empty"));
                        }
                        self.add(Parser::Ac { child, delimiters })
                    }
                }
            }
            "double_quoted_string" => self.quoted(false),
            "single_quoted_string" => self.quoted(true),
            "json" => self.value(false),
            "python_value" => self.value(true),
            "marker" => {
                let mut alternatives = Vec::new();
                for (open, close) in [("<", ">"), ("[", "]")] {
                    let left = self.literal(open)?;
                    let until = self.add(Parser::Until {
                        delimiters: vec![close.to_owned()],
                    })?;
                    let first = self.combine(vec![left, until], false)?;
                    let right = self.literal(close)?;
                    alternatives.push(self.combine(vec![first, right], false)?);
                }
                self.combine(alternatives, true)
            }
            _ => Err(invalid(format!("unsupported PEG builder operation {op}"))),
        }
    }
    fn permute(
        &mut self,
        prefix: &str,
        children: &[usize],
        mask: u8,
        cache: &mut HashMap<u8, usize>,
    ) -> Result<usize> {
        if mask == 0 {
            return self.add(Parser::Epsilon);
        }
        if let Some(&id) = cache.get(&mask) {
            return Ok(id);
        }
        let mut alternatives = Vec::new();
        for (index, &child) in children.iter().enumerate() {
            let bit = 1_u8 << index;
            if mask & bit != 0 {
                let rest = self.permute(prefix, children, mask & !bit, cache)?;
                alternatives.push(self.combine(vec![child, rest], false)?);
            }
        }
        let choice = self.combine(alternatives, true)?;
        let id = self.rule(&format!("{prefix}-{mask}"), choice, false)?;
        cache.insert(mask, id);
        Ok(id)
    }
    fn quoted(&mut self, single: bool) -> Result<usize> {
        let rule = if single {
            "single-quoted-string"
        } else {
            "double-quoted-string"
        };
        if self.begin_rule(rule, false)?.is_none() {
            return self.add(Parser::Ref { name: rule.into() });
        }
        let quote = if single { "'" } else { "\"" };
        let left = self.literal(quote)?;
        let content = self.add(Parser::String {
            delimiter: quote.into(),
        })?;
        let right = self.literal(quote)?;
        let child = self.combine(vec![left, content, right], false)?;
        self.rule(rule, child, false)
    }
    fn value(&mut self, python: bool) -> Result<usize> {
        let rule = if python { "python-value" } else { "json-value" };
        if self.begin_rule(rule, false)?.is_none() {
            return self.add(Parser::Ref { name: rule.into() });
        }
        let alternatives = vec![
            self.container(python, false)?,
            self.container(python, true)?,
            self.string(python)?,
            self.json_number()?,
            self.boolean(python)?,
            self.null(python)?,
        ];
        let child = self.combine(alternatives, true)?;
        self.rule(rule, child, false)
    }
    fn string(&mut self, python: bool) -> Result<usize> {
        let rule = if python {
            "python-string"
        } else {
            "json-string"
        };
        if self.begin_rule(rule, false)?.is_none() {
            return self.add(Parser::Ref { name: rule.into() });
        }
        let child = if python {
            let left = self.quoted(false)?;
            let right = self.quoted(true)?;
            self.combine(vec![left, right], true)?
        } else {
            let left = self.literal("\"")?;
            let content = self.add(Parser::String {
                delimiter: "\"".into(),
            })?;
            let right = self.literal("\"")?;
            self.combine(vec![left, content, right], false)?
        };
        self.rule(rule, child, false)
    }
    fn boolean(&mut self, python: bool) -> Result<usize> {
        let rule = if python { "python-bool" } else { "json-bool" };
        if self.begin_rule(rule, false)?.is_none() {
            return self.add(Parser::Ref { name: rule.into() });
        }
        let yes = self.literal(if python { "True" } else { "true" })?;
        let no = self.literal(if python { "False" } else { "false" })?;
        let child = self.combine(vec![yes, no], true)?;
        self.rule(rule, child, false)
    }
    fn null(&mut self, python: bool) -> Result<usize> {
        let rule = if python { "python-none" } else { "json-null" };
        if self.begin_rule(rule, false)?.is_none() {
            return self.add(Parser::Ref { name: rule.into() });
        }
        let child = self.literal(if python { "None" } else { "null" })?;
        self.rule(rule, child, false)
    }
    fn json_number(&mut self) -> Result<usize> {
        let rule = "json-number";
        if self.begin_rule(rule, false)?.is_none() {
            return self.add(Parser::Ref { name: rule.into() });
        }
        let digit = self.chars("[1-9]", 1, 1)?;
        let digits = self.chars("[0-9]", 1, -1)?;
        let zero = self.literal("0")?;
        let tail = self.chars("[0-9]", 0, -1)?;
        let nonzero = self.combine(vec![digit, tail], false)?;
        let integer = self.combine(vec![zero, nonzero], true)?;
        let dot = self.literal(".")?;
        let fraction = self.combine(vec![dot, digits], false)?;
        let lower = self.literal("e")?;
        let upper = self.literal("E")?;
        let e = self.combine(vec![lower, upper], true)?;
        let sign = self.chars("[+-]", 1, 1)?;
        let sign = self.repeat(sign, 0, 1)?;
        let exponent = self.combine(vec![e, sign, digits], false)?;
        let continuation = self.chars("[0-9.eE+-]", 1, 1)?;
        let lookahead = self.add(Parser::Not {
            child: continuation,
        })?;
        let negative = self.literal("-")?;
        let negative = self.repeat(negative, 0, 1)?;
        let fraction = self.repeat(fraction, 0, 1)?;
        let exponent = self.repeat(exponent, 0, 1)?;
        let child = self.combine(
            vec![negative, integer, fraction, exponent, lookahead],
            false,
        )?;
        self.rule(rule, child, false)
    }
    fn container(&mut self, python: bool, list: bool) -> Result<usize> {
        let rule = match (python, list) {
            (false, false) => "json-object",
            (false, true) => "json-array",
            (true, false) => "python-dict",
            (true, true) => "python-array",
        };
        if self.begin_rule(rule, false)?.is_none() {
            return self.add(Parser::Ref { name: rule.into() });
        }
        let ws = self.add(Parser::Space)?;
        let element = if list {
            self.value(python)?
        } else {
            let key = self.string(python)?;
            let colon = self.literal(":")?;
            let value = self.value(python)?;
            self.combine(vec![key, ws, colon, ws, value], false)?
        };
        let comma = self.literal(",")?;
        let rest = if list && python {
            let value = self.value(python)?;
            self.combine(vec![comma, ws, value], false)?
        } else {
            self.combine(vec![ws, comma, ws, element], false)?
        };
        let repeated = self.repeat(rest, 0, -1)?;
        let elements = self.combine(vec![element, repeated], false)?;
        let open = self.literal(if list { "[" } else { "{" })?;
        let close = self.literal(if list { "]" } else { "}" })?;
        let closing = self.literal(if list { "]" } else { "}" })?;
        let filled = self.combine(vec![elements, ws, closing], false)?;
        let choice = self.combine(vec![close, filled], true)?;
        let mut children = vec![open, ws, choice];
        if python {
            children.push(ws);
        }
        let child = self.combine(children, false)?;
        self.rule(rule, child, false)
    }
    fn resolve(&mut self) -> Result<()> {
        let refs: HashMap<usize, usize> = self
            .parsers
            .iter()
            .enumerate()
            .filter_map(|(id, p)| {
                if let Parser::Ref { name } = p {
                    Some((id, self.rules.get(name).copied()))
                } else {
                    None
                }
            })
            .map(|(id, target)| {
                target
                    .map(|target| (id, target))
                    .ok_or_else(|| invalid("unresolved PEG rule"))
            })
            .collect::<Result<_>>()?;
        for parser in &mut self.parsers {
            match parser {
                Parser::Sequence { children } | Parser::Choice { children } => {
                    for child in children {
                        if let Some(id) = refs.get(child) {
                            *child = *id;
                        }
                    }
                }
                Parser::Repetition { child, .. }
                | Parser::And { child }
                | Parser::Not { child }
                | Parser::Rule { child, .. }
                | Parser::Atomic { child }
                | Parser::Tag { child, .. }
                | Parser::Gbnf { child, .. }
                | Parser::Ac { child, .. }
                | Parser::Schema { child, .. } => {
                    if let Some(id) = refs.get(child) {
                        *child = *id;
                    }
                }
                _ => {}
            }
        }
        if let Some(id) = refs.get(&self.root) {
            self.root = *id;
        }
        self.validate()
    }
    fn validate(&self) -> Result<()> {
        if self.parsers.len() > 16384
            || self.root >= self.parsers.len()
            || self.rules.values().any(|&id| id >= self.parsers.len())
        {
            return Err(invalid("PEG arena IDs/count outside bound"));
        }
        for parser in &self.parsers {
            let children: Vec<_> = match parser {
                Parser::Sequence { children } | Parser::Choice { children } => children.clone(),
                Parser::Repetition { child, .. }
                | Parser::And { child }
                | Parser::Not { child }
                | Parser::Rule { child, .. }
                | Parser::Atomic { child }
                | Parser::Tag { child, .. }
                | Parser::Gbnf { child, .. }
                | Parser::Ac { child, .. }
                | Parser::Schema { child, .. } => vec![*child],
                _ => Vec::new(),
            };
            if children.iter().any(|&id| id >= self.parsers.len()) {
                return Err(invalid("PEG child ID outside arena"));
            }
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn captured_tags_preserve_partial_nodes_and_atomic_suppression() -> Result<()> {
        use serde_json::json;
        let graph = json!({"Statements":[],"Root":{"Op":"add","Args":[{"Op":"tag","Args":["first",{"Op":"literal","Args":["a"]}]},{"Op":"atomic","Args":[{"Op":"tag","Args":["second",{"Op":"literal","Args":["bc"]}]}]}]}});
        let arena = Arena::from_graph(&graph)?;
        let partial = arena.parse_captured(b"ab", 1, 0)?;
        assert_eq!(partial.result.result_type, 2);
        assert_eq!(partial.nodes.len(), 1);
        assert_eq!(partial.nodes[0].tag, "first");
        let full = arena.parse_captured(b"abc", 0, 0)?;
        assert_eq!(full.result.result_type, 1);
        assert_eq!(full.nodes.len(), 2);
        assert_eq!(full.nodes[1].tag, "second");
        let failed = arena.parse_captured(b"ax", 0, 0)?;
        assert_eq!(failed.result.result_type, 0);
        assert!(failed.nodes.is_empty());
        Ok(())
    }
    #[test]
    fn restored_arenas_reject_invalid_ids_and_bound_recursive_parsing() -> Result<()> {
        assert!(
            Arena::load(r#"{"parsers":[{"type":"literal","literal":"x"}],"rules":{},"root":1}"#)
                .is_err()
        );
        assert!(
            Arena::load(r#"{"parsers":[{"type":"sequence","children":[1]}],"rules":{},"root":0}"#)
                .is_err()
        );
        let recursive = Arena::load(
            r#"{"parsers":[{"type":"rule","name":"cycle","child":0,"trigger":false}],"rules":{"cycle":0},"root":0}"#,
        )?;
        assert!(recursive.parse(b"x", 0).is_err());
        Ok(())
    }
}
