//! Deterministic GBNF lowering, including Aho-Corasick delimiter grammars.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt::Write as _;

use super::{Arena, Parser, Result, invalid};
fn schema_delegates(schema: &serde_json::Value, raw: bool) -> bool {
    if schema.is_null() {
        return true;
    }
    if !raw {
        return false;
    }
    if let Some(kind) = schema.get("type") {
        if kind == "string" {
            return true;
        }
        if let Some(types) = kind.as_array() {
            return types
                .iter()
                .find(|kind| kind.as_str().is_some_and(|value| value != "null"))
                .is_some_and(|kind| kind == "string");
        }
        false
    } else {
        schema.get("enum").is_some()
    }
}

fn literal(value: &str) -> String {
    let mut result = String::from("\"");
    for value in value.chars() {
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
fn repeated(value: &str, minimum: i32, maximum: i32) -> String {
    match (minimum, maximum) {
        (0, 1) => format!("{value}?"),
        (0, -1) => format!("{value}*"),
        (1, -1) => format!("{value}+"),
        (_, -1) => format!("{value}{{{minimum},}}"),
        (1, 1) => value.to_owned(),
        _ if minimum == maximum => format!("{value}{{{minimum}}}"),
        _ => format!("{value}{{{minimum},{maximum}}}"),
    }
}

#[derive(Default)]
struct State {
    children: BTreeMap<u32, usize>,
    terminal: bool,
    fail: usize,
}
fn character(value: u32) -> String {
    match value {
        45 | 91 | 93 | 92 => format!("\\{}", char::from_u32(value).unwrap_or('\0')),
        10 => "\\n".into(),
        9 => "\\t".into(),
        13 => "\\r".into(),
        32..=126 => char::from_u32(value).unwrap_or('\0').to_string(),
        0..=255 => format!("\\x{value:02X}"),
        256..=65535 => format!("\\u{value:04X}"),
        _ => format!("\\U{value:08X}"),
    }
}
fn class(values: &[u32], negated: bool) -> String {
    let mut result = if negated {
        String::from("[^")
    } else {
        String::from("[")
    };
    for &value in values {
        result.push_str(&character(value));
    }
    result.push(']');
    result
}
#[allow(
    clippy::too_many_lines,
    reason = "Keep trie construction, suffix links and deterministic DFA grammar emission together"
)]
fn automaton(
    strings: &[String],
    prefix: &str,
    including: bool,
    rules: &mut BTreeMap<String, String>,
) -> Result<String> {
    let mut states = vec![State::default()];
    let mut alphabet = BTreeSet::new();
    for string in strings {
        let mut current = 0;
        for value in string.chars().map(u32::from) {
            alphabet.insert(value);
            current = if let Some(&next) = states[current].children.get(&value) {
                next
            } else {
                let next = states.len();
                states.push(State::default());
                states[current].children.insert(value, next);
                next
            };
        }
        states[current].terminal = true;
    }
    if states.len() > 65536 {
        return Err(invalid("PEG delimiter automaton state bound exceeded"));
    }
    let mut queue = VecDeque::from([0]);
    let mut order = Vec::new();
    while let Some(current) = queue.pop_front() {
        order.push(current);
        let children: Vec<_> = states[current]
            .children
            .iter()
            .map(|(&value, &next)| (value, next))
            .collect();
        for (value, next) in children {
            if current != 0 {
                let mut fail = states[current].fail;
                while fail != 0 && !states[fail].children.contains_key(&value) {
                    fail = states[fail].fail;
                }
                states[next].fail = states[fail]
                    .children
                    .get(&value)
                    .copied()
                    .filter(|&id| id != next)
                    .unwrap_or(0);
            }
            queue.push_back(next);
        }
    }
    for current in order {
        if current != 0 {
            states[current].terminal |= states[states[current].fail].terminal;
        }
    }
    let state_name = |state| {
        if state == 0 {
            prefix.to_owned()
        } else {
            format!("{prefix}-{state:02}")
        }
    };
    for current in 0..states.len() {
        if states[current].terminal {
            continue;
        }
        let mut buckets: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
        let mut completing = Vec::new();
        let mut specific = Vec::new();
        for &value in &alphabet {
            let mut state = current;
            while state != 0 && !states[state].children.contains_key(&value) {
                state = states[state].fail;
            }
            let next = states[state].children.get(&value).copied().unwrap_or(0);
            if states[next].terminal {
                completing.push(value);
                specific.push(value);
            } else if next != 0 {
                buckets.entry(next).or_default().push(value);
                specific.push(value);
            }
        }
        let rule = if including {
            let mut choices = Vec::new();
            if !completing.is_empty() {
                choices.push(class(&completing, false));
            }
            for (next, values) in buckets {
                choices.push(format!("{} {}", class(&values, false), state_name(next)));
            }
            choices.push(format!("{} {}", class(&specific, true), state_name(0)));
            choices.join(" | ")
        } else {
            let mut rule = String::from("|");
            for (next, values) in buckets {
                write!(rule, " {} {} |", class(&values, false), state_name(next))?;
            }
            write!(rule, " {} {}", class(&specific, true), state_name(0))?;
            rule
        };
        rules.insert(state_name(current), rule);
    }
    if states[0].terminal {
        rules.insert(prefix.to_owned(), "|".into());
    }
    Ok(prefix.to_owned())
}

impl Arena {
    fn effective(&self, mut id: usize) -> Result<&Parser> {
        for _ in 0..256 {
            match &self.parsers[id] {
                Parser::Tag { child, .. } | Parser::Atomic { child } => id = *child,
                Parser::Schema {
                    child, schema, raw, ..
                } if schema_delegates(schema, *raw) => id = *child,
                _ => return Ok(&self.parsers[id]),
            }
        }
        Err(invalid("PEG effective parser cycle"))
    }
    fn reachable(&self, id: usize, visited: &mut BTreeSet<String>, depth: usize) -> Result<()> {
        if depth > 256 {
            return Err(invalid("PEG rule graph depth exceeded"));
        }
        match &self.parsers[id] {
            Parser::Sequence { children } | Parser::Choice { children } => {
                for &child in children {
                    self.reachable(child, visited, depth + 1)?;
                }
            }
            Parser::Rule { name, child, .. } => {
                if visited.insert(name.clone()) {
                    self.reachable(*child, visited, depth + 1)?;
                }
            }
            Parser::Ref { name } => self.reachable(
                *self
                    .rules
                    .get(name)
                    .ok_or_else(|| invalid("PEG rule missing"))?,
                visited,
                depth + 1,
            )?,
            Parser::Repetition { child, .. }
            | Parser::And { child }
            | Parser::Not { child }
            | Parser::Tag { child, .. }
            | Parser::Atomic { child }
            | Parser::Gbnf { child, .. }
            | Parser::Ac { child, .. }
            | Parser::Schema { child, .. } => self.reachable(*child, visited, depth + 1)?,
            _ => {}
        }
        Ok(())
    }
    #[allow(
        clippy::too_many_lines,
        reason = "One exhaustive lowering match preserves all PEG variant grouping and quantifier rules"
    )]
    fn gbnf(
        &self,
        id: usize,
        rules: &mut BTreeMap<String, String>,
        schemas: &mut crate::schema::RuleBuilder,
        depth: usize,
    ) -> Result<String> {
        if depth > 256 {
            return Err(invalid("PEG grammar expansion depth exceeded"));
        }
        let result = match &self.parsers[id] {
            Parser::Epsilon
            | Parser::Start
            | Parser::End
            | Parser::And { .. }
            | Parser::Not { .. } => String::new(),
            Parser::Any => ".".into(),
            Parser::Space => "space".into(),
            Parser::Literal { literal: value } => literal(value),
            Parser::Sequence { children } => {
                let mut values = Vec::new();
                let mut bytes = 0;
                for &child in children {
                    let value = self.gbnf(child, rules, schemas, depth + 1)?;
                    bytes += value.len() + 3;
                    if bytes > 4 * 1024 * 1024 {
                        return Err(invalid("PEG grammar expansion exceeds byte bound"));
                    }
                    if value.is_empty() {
                        continue;
                    }
                    values.push(
                        if matches!(
                            self.effective(child)?,
                            Parser::Sequence { .. } | Parser::Choice { .. }
                        ) {
                            format!("({value})")
                        } else {
                            value
                        },
                    );
                }
                values.join(" ")
            }
            Parser::Choice { children } => {
                let mut result = String::new();
                for &child in children {
                    if !result.is_empty() {
                        result.push_str(" | ");
                    }
                    let value = self.gbnf(child, rules, schemas, depth + 1)?;
                    if matches!(self.effective(child)?, Parser::Choice { .. }) {
                        write!(result, "({value})")?;
                    } else {
                        result.push_str(&value);
                    }
                    if result.len() > 4 * 1024 * 1024 {
                        return Err(invalid("PEG grammar expansion exceeds byte bound"));
                    }
                }
                result
            }
            Parser::Repetition {
                child,
                min_count,
                max_count,
            } => {
                let value = self.gbnf(*child, rules, schemas, depth + 1)?;
                repeated(
                    &if matches!(
                        self.effective(*child)?,
                        Parser::Sequence { .. } | Parser::Choice { .. }
                    ) {
                        format!("({value})")
                    } else {
                        value
                    },
                    *min_count,
                    *max_count,
                )
            }
            Parser::Chars {
                pattern,
                min_count,
                max_count,
                ..
            } => repeated(pattern, *min_count, *max_count),
            Parser::String { delimiter } => format!(
                r#"( [^{delimiter}\\] | "\\" ( [{delimiter}\\/ bfnrt] | "u" [0-9a-fA-F]{{4}} ) )*"#
            ),
            Parser::Until { delimiters } => {
                if delimiters.is_empty() {
                    ".*".into()
                } else {
                    automaton(delimiters, &format!("until-{id}"), false, rules)?
                }
            }
            Parser::Ac { delimiters, .. } => {
                automaton(delimiters, &format!("ac-{id}"), true, rules)?
            }
            Parser::Rule { name, .. } | Parser::Ref { name } => name.clone(),
            Parser::Tag { child, .. } | Parser::Atomic { child } => {
                self.gbnf(*child, rules, schemas, depth + 1)?
            }
            Parser::Gbnf { grammar, .. } => grammar.clone(),
            Parser::Schema {
                child,
                name,
                schema,
                schema_source,
                raw,
            } => {
                if schema_delegates(schema, *raw) {
                    self.gbnf(*child, rules, schemas, depth + 1)?
                } else {
                    let source = schema_source.as_ref().map_or_else(
                        || serde_json::to_string(schema),
                        |source| Ok(source.clone()),
                    )?;
                    schemas.add(name, &source)?
                }
            }
        };
        if result.len() > 4 * 1024 * 1024 {
            return Err(invalid("PEG grammar expansion exceeds byte bound"));
        }
        Ok(result)
    }
    /// Generates deterministic GBNF from reachable rules or lazy trigger rules.
    /// # Errors
    /// Rejects cyclic expansion, missing rules and grammar resource limits.
    pub fn grammar(&self, lazy: bool) -> Result<String> {
        self.grammar_with_refs(lazy, &[])
    }
    /// Generates a composed grammar after resolving shared source schema documents.
    ///
    /// # Errors
    /// Rejects malformed references, unsupported schemas and grammar resource limits.
    pub fn grammar_with_refs(&self, lazy: bool, references: &[String]) -> Result<String> {
        let mut schemas = crate::schema::RuleBuilder::new();
        if references.len() > 128 {
            return Err(invalid("PEG schema reference count bound exceeded"));
        }
        for reference in references {
            schemas.resolve(reference)?;
        }
        let mut reachable = BTreeSet::new();
        let mut triggers = Vec::new();
        for (name, &id) in &self.rules {
            if matches!(&self.parsers[id], Parser::Rule { trigger: true, .. }) {
                triggers.push(name.clone());
                if lazy {
                    self.reachable(id, &mut reachable, 0)?;
                }
            }
        }
        if !lazy {
            self.reachable(self.root, &mut reachable, 0)?;
        }
        let mut rules = BTreeMap::from([(
            "space".to_owned(),
            "| \" \" | \"\\n\"{1,2} [ \\t]{0,20}".to_owned(),
        )]);
        for (name, &id) in &self.rules {
            if !reachable.contains(name) {
                continue;
            }
            if let Parser::Rule { child, .. } = &self.parsers[id] {
                let rule = self.gbnf(*child, &mut rules, &mut schemas, 0)?;
                rules.insert(name.clone(), rule);
            }
        }
        let root = if lazy {
            triggers.join(" | ")
        } else {
            self.gbnf(self.root, &mut rules, &mut schemas, 0)?
        };
        rules.insert("root".into(), root);
        for (name, rule) in schemas.into_rules() {
            rules.entry(name).or_insert(rule);
        }
        let mut result = String::new();
        for (name, rule) in rules {
            writeln!(result, "{name} ::= {rule}")?;
            if result.len() > 4 * 1024 * 1024 {
                return Err(invalid("PEG grammar output limit exceeded"));
            }
        }
        Ok(result)
    }
}
