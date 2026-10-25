//! Safe GBNF parsing with the pinned Prism rule-element layout.
//!
//! The parser follows the MIT-licensed `llama_grammar_parser` in
//! PrismML-Eng/llama.cpp revision b87ca7225456b06b0975b8239c66b5aecf07e37c.
//! Attribution is retained in the runtime's PRISM-LICENSE.txt.

use std::collections::BTreeMap;

use crate::{Result, invalid};

pub mod runtime;

/// One flattened grammar instruction in the pinned production representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Element {
    /// END=0, ALT=1, REF=2, CHAR=3, NOT=4, RANGE=5, `CHAR_ALT=6`,
    /// ANY=7, TOKEN=8 and `TOKEN_NOT=9`.
    pub kind: u32,
    /// Code point, token ID or referenced rule ID; zero for structural markers.
    pub value: u32,
}

const fn element(kind: u32, value: u32) -> Element {
    Element { kind, value }
}

/// An independently owned parsed grammar with deterministic generated rule IDs.
#[derive(Clone, Debug)]
pub struct Grammar {
    /// Symbol names in lexical order, mapped to source encounter IDs.
    pub symbols: BTreeMap<String, u32>,
    /// Rules in ID order, each terminated by END.
    pub rules: Vec<Vec<Element>>,
}

impl Grammar {
    /// Parses GBNF literals, character classes, groups, alternation, repetition,
    /// rule references, any-character instructions and explicit `<[token-id]>`.
    /// Named tokenizer spellings require a vocabulary and are rejected here.
    ///
    /// # Errors
    /// Rejects invalid grammar, missing rules, expansion above the original
    /// 2,000 repetition threshold, or bounded input/stack/result limits.
    pub fn parse(source: &str) -> Result<Self> {
        if source.is_empty() || source.len() > 65536 || source.contains('\0') {
            return Err(invalid("grammar must contain 1..65536 non-NUL bytes"));
        }
        let mut parser = Parser {
            source,
            pos: 0,
            symbols: BTreeMap::new(),
            rules: Vec::new(),
            depth: 0,
        };
        parser.space(true);
        while parser.byte() != 0 {
            let name = parser.name()?.to_owned();
            parser.space(false);
            let id = parser.symbol(&name)?;
            parser.expect(b"::=")?;
            parser.space(true);
            parser.alternates(&name, id, false)?;
            match parser.byte() {
                b'\r' => {
                    parser.pos += 1;
                    if parser.byte() == b'\n' {
                        parser.pos += 1;
                    }
                }
                b'\n' => parser.pos += 1,
                0 => (),
                _ => return Err(invalid("expected newline after grammar rule")),
            }
            parser.space(true);
        }
        let count: usize = parser.rules.iter().map(Vec::len).sum();
        if count > 65536 || parser.rules.iter().any(Vec::is_empty) {
            return Err(invalid("undefined grammar rule or excessive expansion"));
        }
        for instruction in parser
            .rules
            .iter()
            .flatten()
            .filter(|instruction| instruction.kind == 2)
        {
            if parser
                .rules
                .get(usize::try_from(instruction.value)?)
                .is_none_or(Vec::is_empty)
            {
                return Err(invalid("undefined grammar rule reference"));
            }
        }
        Ok(Self {
            symbols: parser.symbols,
            rules: parser.rules,
        })
    }
}

struct Parser<'a> {
    source: &'a str,
    pos: usize,
    symbols: BTreeMap<String, u32>,
    rules: Vec<Vec<Element>>,
    depth: usize,
}

const fn word(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'-'
}

impl<'a> Parser<'a> {
    fn byte(&self) -> u8 {
        self.source.as_bytes().get(self.pos).copied().unwrap_or(0)
    }

    fn space(&mut self, newline: bool) {
        loop {
            match self.byte() {
                b' ' | b'\t' => self.pos += 1,
                b'\r' | b'\n' if newline => self.pos += 1,
                b'#' => {
                    while !matches!(self.byte(), 0 | b'\r' | b'\n') {
                        self.pos += 1;
                    }
                }
                _ => break,
            }
        }
    }

    fn expect(&mut self, bytes: &[u8]) -> Result<()> {
        if !self.source.as_bytes()[self.pos..].starts_with(bytes) {
            return Err(invalid("unexpected grammar delimiter"));
        }
        self.pos += bytes.len();
        Ok(())
    }

    fn name(&mut self) -> Result<&'a str> {
        let start = self.pos;
        while word(self.byte()) {
            self.pos += 1;
        }
        if self.pos == start {
            return Err(invalid("expected grammar rule name"));
        }
        Ok(&self.source[start..self.pos])
    }

    fn integer(&mut self) -> Result<u64> {
        let start = self.pos;
        while self.byte().is_ascii_digit() {
            self.pos += 1;
        }
        if self.pos == start {
            return Err(invalid("expected grammar integer"));
        }
        Ok(self.source[start..self.pos].parse()?)
    }

    fn symbol(&mut self, name: &str) -> Result<u32> {
        if let Some(id) = self.symbols.get(name) {
            return Ok(*id);
        }
        if self.symbols.len() >= 4096 {
            return Err(invalid("grammar symbol limit exceeded"));
        }
        let id = u32::try_from(self.symbols.len())?;
        self.symbols.insert(name.to_owned(), id);
        Ok(id)
    }

    fn generated(&mut self, name: &str) -> Result<u32> {
        if self.symbols.len() >= 4096 {
            return Err(invalid("grammar symbol limit exceeded"));
        }
        let id = u32::try_from(self.symbols.len())?;
        self.symbols.insert(format!("{name}_{id}"), id);
        Ok(id)
    }

    fn add_rule(&mut self, id: u32, rule: Vec<Element>) -> Result<()> {
        let index = usize::try_from(id)?;
        if index >= 4096 || rule.len() > 65536 {
            return Err(invalid("grammar rule limit exceeded"));
        }
        if self.rules.len() <= index {
            self.rules.resize(index + 1, Vec::new());
        }
        self.rules[index] = rule;
        if self.rules.iter().map(Vec::len).sum::<usize>() > 65536 {
            return Err(invalid("grammar element limit exceeded"));
        }
        Ok(())
    }

    fn character(&mut self) -> Result<u32> {
        if self.byte() == b'\\' {
            self.pos += 1;
            let code = self.byte();
            self.pos += usize::from(code != 0);
            match code {
                b'x' | b'u' | b'U' => {
                    let count = match code {
                        b'x' => 2,
                        b'u' => 4,
                        _ => 8,
                    };
                    let end = self.pos + count;
                    let digits = self
                        .source
                        .as_bytes()
                        .get(self.pos..end)
                        .ok_or_else(|| invalid("truncated grammar escape"))?;
                    if !digits.iter().all(u8::is_ascii_hexdigit) {
                        return Err(invalid("invalid grammar hex escape"));
                    }
                    let value = u32::from_str_radix(&self.source[self.pos..end], 16)?;
                    self.pos = end;
                    Ok(value)
                }
                b't' => Ok(u32::from(b'\t')),
                b'r' => Ok(u32::from(b'\r')),
                b'n' => Ok(u32::from(b'\n')),
                b'\\' | b'"' | b'[' | b']' => Ok(u32::from(code)),
                _ => Err(invalid("unknown grammar escape")),
            }
        } else {
            let character = self.source[self.pos..]
                .chars()
                .next()
                .ok_or_else(|| invalid("truncated grammar character"))?;
            self.pos += character.len_utf8();
            Ok(u32::from(character))
        }
    }

    fn token(&mut self) -> Result<u32> {
        self.expect(b"<[")?;
        let token = u32::try_from(self.integer()?)?;
        self.expect(b"]>")?;
        Ok(token)
    }

    fn alternates(&mut self, name: &str, id: u32, nested: bool) -> Result<()> {
        self.depth += 1;
        if self.depth > 256 {
            return Err(invalid("grammar nesting limit exceeded"));
        }
        let mut rule = Vec::new();
        self.sequence(name, &mut rule, nested)?;
        while self.byte() == b'|' {
            rule.push(element(1, 0));
            self.pos += 1;
            self.space(true);
            self.sequence(name, &mut rule, nested)?;
        }
        rule.push(element(0, 0));
        self.add_rule(id, rule)?;
        self.depth -= 1;
        Ok(())
    }

    fn repeat(
        &mut self,
        name: &str,
        rule: &mut Vec<Element>,
        start: usize,
        previous: &mut u64,
        minimum: u64,
        maximum: Option<u64>,
    ) -> Result<()> {
        if start == rule.len() {
            return Err(invalid("repetition requires a preceding grammar item"));
        }
        let total = maximum
            .filter(|value| *value > 0)
            .unwrap_or_else(|| minimum.max(1));
        let expansion = previous
            .checked_mul(total)
            .ok_or_else(|| invalid("grammar repetition overflow"))?;
        if expansion >= 2000 {
            return Err(invalid(
                "grammar repetition complexity exceeds pinned threshold",
            ));
        }
        let prior = rule[start..].to_vec();
        if minimum == 0 {
            rule.truncate(start);
        } else {
            for _ in 1..minimum {
                if rule.len() + prior.len() > 65536 {
                    return Err(invalid("grammar expansion exceeds element limit"));
                }
                rule.extend_from_slice(&prior);
            }
        }
        let optional = maximum.map_or(Ok(1), |value| {
            value
                .checked_sub(minimum)
                .ok_or_else(|| invalid("reversed grammar repetition bounds"))
        })?;
        let mut last = 0;
        for index in 0..optional {
            let id = self.generated(name)?;
            let mut recursive = prior.clone();
            if index > 0 || maximum.is_none() {
                recursive.push(element(2, if maximum.is_none() { id } else { last }));
            }
            recursive.extend([element(1, 0), element(0, 0)]);
            self.add_rule(id, recursive)?;
            last = id;
        }
        if optional > 0 {
            rule.push(element(2, last));
        }
        *previous = expansion;
        Ok(())
    }

    #[allow(
        clippy::too_many_lines,
        reason = "One source-ordered grammar sequence state machine retains the pinned lowering rules."
    )]
    fn sequence(&mut self, name: &str, rule: &mut Vec<Element>, nested: bool) -> Result<()> {
        let mut last_start = rule.len();
        let mut previous = 1;
        loop {
            match self.byte() {
                b'"' => {
                    self.pos += 1;
                    last_start = rule.len();
                    previous = 1;
                    while self.byte() != b'"' {
                        rule.push(element(3, self.character()?));
                    }
                    self.pos += 1;
                    self.space(nested);
                }
                b'[' => {
                    self.pos += 1;
                    let first_type = if self.byte() == b'^' {
                        self.pos += 1;
                        4
                    } else {
                        3
                    };
                    last_start = rule.len();
                    previous = 1;
                    while self.byte() != b']' {
                        let kind = if last_start < rule.len() {
                            6
                        } else {
                            first_type
                        };
                        rule.push(element(kind, self.character()?));
                        if self.byte() == b'-'
                            && self.source.as_bytes().get(self.pos + 1) != Some(&b']')
                        {
                            self.pos += 1;
                            rule.push(element(5, self.character()?));
                        }
                    }
                    self.pos += 1;
                    self.space(nested);
                }
                b'<' | b'!' => {
                    let kind = if self.byte() == b'!' {
                        self.pos += 1;
                        9
                    } else {
                        8
                    };
                    last_start = rule.len();
                    previous = 1;
                    rule.push(element(kind, self.token()?));
                    self.space(nested);
                }
                byte if word(byte) => {
                    let reference = self.name()?.to_owned();
                    let id = self.symbol(&reference)?;
                    last_start = rule.len();
                    previous = 1;
                    rule.push(element(2, id));
                    self.space(nested);
                }
                b'(' => {
                    self.pos += 1;
                    self.space(true);
                    let before = self.symbols.len();
                    let id = self.generated(name)?;
                    self.alternates(name, id, true)?;
                    previous = u64::try_from((self.symbols.len() - before).max(1))?;
                    last_start = rule.len();
                    rule.push(element(2, id));
                    self.expect(b")")?;
                    self.space(nested);
                }
                b'.' => {
                    last_start = rule.len();
                    previous = 1;
                    rule.push(element(7, 0));
                    self.pos += 1;
                    self.space(nested);
                }
                b'*' | b'+' | b'?' => {
                    let kind = self.byte();
                    self.pos += 1;
                    self.space(nested);
                    self.repeat(
                        name,
                        rule,
                        last_start,
                        &mut previous,
                        u64::from(kind == b'+'),
                        (kind == b'?').then_some(1),
                    )?;
                }
                b'{' => {
                    self.pos += 1;
                    self.space(nested);
                    let minimum = self.integer()?;
                    self.space(nested);
                    let mut maximum;
                    if self.byte() == b'}' {
                        maximum = Some(minimum);
                    } else {
                        self.expect(b",")?;
                        self.space(nested);
                        maximum = if self.byte().is_ascii_digit() {
                            Some(self.integer()?)
                        } else {
                            None
                        };
                        self.space(nested);
                    }
                    self.expect(b"}")?;
                    self.space(nested);
                    if minimum > 2000 {
                        return Err(invalid("grammar repetition exceeds pinned threshold"));
                    }
                    if maximum.is_some_and(|value| value > 2000) {
                        maximum = None;
                    }
                    self.repeat(name, rule, last_start, &mut previous, minimum, maximum)?;
                }
                _ => break,
            }
            if rule.len() > 65536 {
                return Err(invalid("grammar rule element limit exceeded"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grammar_parses_forward_refs_tokens_and_bounded_repetitions() -> Result<()> {
        let grammar = Grammar::parse("root ::= item{1,2} !<[42]>\nitem ::= [a-z] | \"é\"\n")?;
        assert_eq!(grammar.symbols["root"], 0);
        assert!(grammar.rules[0].contains(&element(9, 42)));
        assert!(Grammar::parse("root ::= undefined").is_err());
        assert!(Grammar::parse("root ::= \"unterminated").is_err());
        assert!(Grammar::parse("root ::= [unterminated").is_err());
        Ok(())
    }
}
