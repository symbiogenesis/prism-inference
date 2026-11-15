//! Independent GBNF observations from a token stream, syntax tree and lowering pass.
//!
//! The production parser emits instructions while scanning bytes. This reference
//! first lexes characters, then builds owned alternatives/terms, then allocates
//! symbols and lowers that tree. It imports neither production grammar code nor
//! the literal assertion corpus. The pinned MIT source contract owns instruction
//! kinds, encounter-order IDs, generated names and repetition expansion order.

use std::{collections::BTreeMap, iter::Peekable, str::Chars};

use serde::Deserialize;
use serde_json::{Value, json};

use super::{Result, invalid};

pub(super) const ID: &str = "tests/test-grammar-parser.cpp";
type Instruction = (u32, u32);

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
struct Request {
    schema_version: u32,
    operations: Vec<Operation>,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
struct Operation {
    kind: String,
    grammar: String,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mark {
    Assign,
    Line,
    Bar,
    Open,
    Close,
    Star,
    Plus,
    Optional,
    Brace,
    EndBrace,
    Comma,
}

enum Token {
    Name(String),
    Units(Vec<Instruction>),
    Mark(Mark),
}

fn require(condition: bool) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(invalid("invalid independent GBNF syntax or expansion"))
    }
}

fn expect(chars: &mut Peekable<Chars<'_>>, wanted: char) -> Result<()> {
    require(chars.next() == Some(wanted))
}

fn character(chars: &mut Peekable<Chars<'_>>) -> Result<u32> {
    let next = chars
        .next()
        .ok_or_else(|| invalid("missing independent grammar character"))?;
    if next != '\\' {
        return Ok(u32::from(next));
    }
    let escaped = chars
        .next()
        .ok_or_else(|| invalid("missing independent grammar escape"))?;
    let digits = match escaped {
        'x' => 2,
        'u' => 4,
        'U' => 8,
        _ => 0,
    };
    if digits != 0 {
        let mut code = 0_u32;
        for _ in 0..digits {
            let digit = chars
                .next()
                .and_then(|digit| digit.to_digit(16))
                .ok_or_else(|| invalid("invalid independent grammar hex escape"))?;
            code = code
                .checked_mul(16)
                .and_then(|value| value.checked_add(digit))
                .ok_or_else(|| invalid("independent grammar escape overflow"))?;
        }
        return Ok(code);
    }
    match escaped {
        't' => Ok(9),
        'r' => Ok(13),
        'n' => Ok(10),
        '\\' | '"' | '[' | ']' => Ok(u32::from(escaped)),
        _ => Err(invalid("unknown independent grammar escape")),
    }
}

fn units(chars: &mut Peekable<Chars<'_>>, quote: bool) -> Result<Vec<Instruction>> {
    let ending = if quote { '"' } else { ']' };
    let excluded = !quote && chars.peek() == Some(&'^');
    if excluded {
        chars.next();
    }
    let mut output = Vec::new();
    while chars.peek() != Some(&ending) {
        let kind = if quote || output.is_empty() {
            if excluded { 4 } else { 3 }
        } else {
            6
        };
        output.push((kind, character(chars)?));
        if !quote && chars.peek() == Some(&'-') {
            let mut lookahead = chars.clone();
            lookahead.next();
            if lookahead.peek() != Some(&']') {
                chars.next();
                output.push((5, character(chars)?));
            }
        }
    }
    chars.next();
    Ok(output)
}

fn token_id(chars: &mut Peekable<Chars<'_>>, excluded: bool) -> Result<Token> {
    if excluded {
        expect(chars, '<')?;
    }
    expect(chars, '[')?;
    let mut digits = String::new();
    while chars.peek().is_some_and(char::is_ascii_digit) {
        digits.push(chars.next().ok_or_else(|| invalid("missing token digit"))?);
    }
    let value = digits.parse::<u32>()?;
    expect(chars, ']')?;
    expect(chars, '>')?;
    Ok(Token::Units(vec![(if excluded { 9 } else { 8 }, value)]))
}

fn word(character: char) -> bool {
    character.is_ascii_alphanumeric() || character == '-'
}

fn lex(source: &str) -> Result<Vec<Token>> {
    let mut chars = source.chars().peekable();
    let mut tokens = Vec::new();
    while let Some(character) = chars.next() {
        let marker = match character {
            ' ' | '\t' => continue,
            '#' => {
                while chars
                    .peek()
                    .is_some_and(|next| !matches!(next, '\r' | '\n'))
                {
                    chars.next();
                }
                continue;
            }
            '\r' | '\n' => {
                if character == '\r' && chars.peek() == Some(&'\n') {
                    chars.next();
                }
                Mark::Line
            }
            ':' => {
                expect(&mut chars, ':')?;
                expect(&mut chars, '=')?;
                Mark::Assign
            }
            '|' => Mark::Bar,
            '(' => Mark::Open,
            ')' => Mark::Close,
            '*' => Mark::Star,
            '+' => Mark::Plus,
            '?' => Mark::Optional,
            '{' => Mark::Brace,
            '}' => Mark::EndBrace,
            ',' => Mark::Comma,
            '"' | '[' => {
                tokens.push(Token::Units(units(&mut chars, character == '"')?));
                continue;
            }
            '<' | '!' => {
                tokens.push(token_id(&mut chars, character == '!')?);
                continue;
            }
            '.' => {
                tokens.push(Token::Units(vec![(7, 0)]));
                continue;
            }
            first if word(first) => {
                let mut name = String::from(first);
                while chars.peek().is_some_and(|next| word(*next)) {
                    name.push(
                        chars
                            .next()
                            .ok_or_else(|| invalid("missing grammar name"))?,
                    );
                }
                tokens.push(Token::Name(name));
                continue;
            }
            _ => return Err(invalid("unknown independent grammar token")),
        };
        tokens.push(Token::Mark(marker));
    }
    Ok(tokens)
}

struct Alternatives(Vec<Vec<Term>>);
struct Term {
    atom: Atom,
    repetitions: Vec<(u64, Option<u64>)>,
}
enum Atom {
    Units(Vec<Instruction>),
    Reference(String),
    Group(Alternatives),
}

struct Syntax {
    tokens: std::vec::IntoIter<Token>,
    next: Option<Token>,
}

impl Syntax {
    fn new(tokens: Vec<Token>) -> Self {
        let mut tokens = tokens.into_iter();
        let next = tokens.next();
        Self { tokens, next }
    }

    fn advance(&mut self) -> Option<Token> {
        std::mem::replace(&mut self.next, self.tokens.next())
    }

    fn marked(&self, mark: Mark) -> bool {
        matches!(&self.next, Some(Token::Mark(actual)) if *actual == mark)
    }

    fn marker(&mut self, mark: Mark) -> Result<()> {
        require(self.marked(mark))?;
        self.advance();
        Ok(())
    }

    fn lines(&mut self, allowed: bool) {
        while allowed && self.marked(Mark::Line) {
            self.advance();
        }
    }

    fn name(&mut self) -> Result<String> {
        match self.advance() {
            Some(Token::Name(name)) => Ok(name),
            _ => Err(invalid("expected independent grammar name")),
        }
    }

    fn integer(&mut self) -> Result<u64> {
        let number = self.name()?;
        require(number.bytes().all(|byte| byte.is_ascii_digit()))?;
        Ok(number.parse()?)
    }

    fn repetition(&mut self, nested: bool) -> Result<Option<(u64, Option<u64>)>> {
        let result = if self.marked(Mark::Star) {
            self.advance();
            Some((0, None))
        } else if self.marked(Mark::Plus) {
            self.advance();
            Some((1, None))
        } else if self.marked(Mark::Optional) {
            self.advance();
            Some((0, Some(1)))
        } else if self.marked(Mark::Brace) {
            self.advance();
            self.lines(nested);
            let minimum = self.integer()?;
            self.lines(nested);
            let maximum = if self.marked(Mark::EndBrace) {
                Some(minimum)
            } else {
                self.marker(Mark::Comma)?;
                self.lines(nested);
                let maximum = if matches!(&self.next, Some(Token::Name(_))) {
                    Some(self.integer()?)
                } else {
                    None
                };
                self.lines(nested);
                maximum
            };
            self.marker(Mark::EndBrace)?;
            Some((minimum, maximum))
        } else {
            None
        };
        if result.is_some() {
            self.lines(nested);
        }
        Ok(result)
    }

    fn alternatives(&mut self, nested: bool, depth: usize) -> Result<Alternatives> {
        require(depth <= 256)?;
        let mut alternatives = Vec::new();
        loop {
            let mut sequence = Vec::new();
            loop {
                self.lines(nested);
                let atom = match &self.next {
                    Some(Token::Name(_)) => Atom::Reference(self.name()?),
                    Some(Token::Units(_)) => match self.advance() {
                        Some(Token::Units(units)) => Atom::Units(units),
                        _ => return Err(invalid("independent grammar token disappeared")),
                    },
                    Some(Token::Mark(Mark::Open)) => {
                        self.advance();
                        self.lines(true);
                        let body = self.alternatives(true, depth + 1)?;
                        self.marker(Mark::Close)?;
                        Atom::Group(body)
                    }
                    _ => break,
                };
                self.lines(nested);
                let mut repetitions = Vec::new();
                while let Some(bounds) = self.repetition(nested)? {
                    repetitions.push(bounds);
                }
                sequence.push(Term { atom, repetitions });
            }
            alternatives.push(sequence);
            if !self.marked(Mark::Bar) {
                break;
            }
            self.advance();
            self.lines(true);
        }
        Ok(Alternatives(alternatives))
    }

    fn rules(mut self) -> Result<Vec<(String, Alternatives)>> {
        let mut rules = Vec::new();
        self.lines(true);
        while self.next.is_some() {
            let name = self.name()?;
            self.marker(Mark::Assign)?;
            self.lines(true);
            let alternatives = self.alternatives(false, 1)?;
            require(self.next.is_none() || self.marked(Mark::Line))?;
            rules.push((name, alternatives));
            self.lines(true);
        }
        Ok(rules)
    }
}

#[derive(Default)]
struct Lowering {
    symbols: BTreeMap<String, u32>,
    rules: Vec<Vec<Instruction>>,
    elements: usize,
}

impl Lowering {
    fn symbol(&mut self, name: &str) -> Result<u32> {
        if let Some(id) = self.symbols.get(name) {
            return Ok(*id);
        }
        require(self.symbols.len() < 4096)?;
        let id = u32::try_from(self.symbols.len())?;
        self.symbols.insert(name.to_owned(), id);
        Ok(id)
    }

    fn generated(&mut self, owner: &str) -> Result<u32> {
        require(self.symbols.len() < 4096)?;
        let id = u32::try_from(self.symbols.len())?;
        self.symbols.insert(format!("{owner}_{id}"), id);
        Ok(id)
    }

    fn publish(&mut self, id: u32, body: Vec<Instruction>) -> Result<()> {
        let index = usize::try_from(id)?;
        require(index < 4096 && body.len() <= 65_536)?;
        if self.rules.len() <= index {
            self.rules.resize_with(index + 1, Vec::new);
        }
        let total = self
            .elements
            .checked_sub(self.rules[index].len())
            .and_then(|count| count.checked_add(body.len()))
            .ok_or_else(|| invalid("independent rule size overflow"))?;
        require(total <= 65_536)?;
        self.rules[index] = body;
        self.elements = total;
        Ok(())
    }

    fn alternates(&mut self, owner: &str, id: u32, alternatives: Alternatives) -> Result<()> {
        let mut body = Vec::new();
        for (index, sequence) in alternatives.0.into_iter().enumerate() {
            if index != 0 {
                body.push((1, 0));
            }
            for term in sequence {
                self.term(owner, term, &mut body)?;
            }
        }
        body.push((0, 0));
        self.publish(id, body)
    }

    fn term(&mut self, owner: &str, term: Term, output: &mut Vec<Instruction>) -> Result<()> {
        let start = output.len();
        let mut complexity = match term.atom {
            Atom::Units(units) => {
                output.extend(units);
                1
            }
            Atom::Reference(name) => {
                output.push((2, self.symbol(&name)?));
                1
            }
            Atom::Group(alternatives) => {
                let before = self.symbols.len();
                let id = self.generated(owner)?;
                self.alternates(owner, id, alternatives)?;
                output.push((2, id));
                u64::try_from(self.symbols.len().saturating_sub(before).max(1))?
            }
        };
        for (minimum, mut maximum) in term.repetitions {
            require(minimum <= 2000 && start < output.len())?;
            if maximum.is_some_and(|maximum| maximum > 2000) {
                maximum = None;
            }
            let multiplier = maximum
                .filter(|maximum| *maximum != 0)
                .unwrap_or_else(|| minimum.max(1));
            complexity = complexity
                .checked_mul(multiplier)
                .ok_or_else(|| invalid("independent repetition complexity overflow"))?;
            require(complexity < 2000)?;
            let atom = output[start..].to_vec();
            if minimum == 0 {
                output.truncate(start);
            }
            for _ in 1..minimum {
                require(
                    output
                        .len()
                        .checked_add(atom.len())
                        .is_some_and(|count| count <= 65_536),
                )?;
                output.extend_from_slice(&atom);
            }
            let optional = maximum
                .map(|maximum| {
                    maximum
                        .checked_sub(minimum)
                        .ok_or_else(|| invalid("reversed independent repetition"))
                })
                .transpose()?
                .unwrap_or(1);
            let mut previous = None;
            for _ in 0..optional {
                let id = self.generated(owner)?;
                let mut branch = atom.clone();
                if maximum.is_none() {
                    branch.push((2, id));
                } else if let Some(id) = previous {
                    branch.push((2, id));
                }
                branch.extend([(1, 0), (0, 0)]);
                self.publish(id, branch)?;
                previous = Some(id);
            }
            if let Some(id) = previous {
                output.push((2, id));
            }
        }
        require(output.len() <= 65_536)
    }

    fn compile(source: &str) -> Result<Self> {
        let tree = Syntax::new(lex(source)?).rules()?;
        let mut lowering = Self::default();
        for (name, body) in tree {
            let id = lowering.symbol(&name)?;
            lowering.alternates(&name, id, body)?;
        }
        require(lowering.rules.iter().all(|rule| !rule.is_empty()))?;
        for (kind, reference) in lowering.rules.iter().flatten() {
            if *kind == 2 {
                require(
                    lowering
                        .rules
                        .get(usize::try_from(*reference)?)
                        .is_some_and(|rule| !rule.is_empty()),
                )?;
            }
        }
        Ok(lowering)
    }
}

pub(super) fn handles(request: &Value) -> bool {
    request["Operations"].as_array().is_some_and(|operations| {
        !operations.is_empty()
            && operations
                .iter()
                .all(|operation| operation["Kind"] == "grammar-parser-parse")
    })
}

pub(super) fn execute(value: &Value) -> Result<Value> {
    let request: Request = serde_json::from_value(value.clone())?;
    require(request.schema_version == 1 && (1..=128).contains(&request.operations.len()))?;
    for operation in &request.operations {
        require(
            operation.kind == "grammar-parser-parse"
                && !operation.grammar.is_empty()
                && operation.grammar.len() <= 65_536
                && !operation.grammar.contains('\0'),
        )?;
    }
    let results = request.operations.into_iter().map(|operation| {
        match Lowering::compile(&operation.grammar) {
            Ok(grammar) => json!({"Parsed":true,
                "SymbolIds":grammar.symbols.iter().map(|(name,id)| json!({"Name":name,"Id":id})).collect::<Vec<_>>(),
                "Rules":grammar.rules.iter().map(|rule| rule.iter().map(|(kind,value)| json!({"Type":kind,"Value":value})).collect::<Vec<_>>()).collect::<Vec<_>>()}),
            Err(_) => json!({"Parsed":false,"SymbolIds":[],"Rules":[]}),
        }
    }).collect::<Vec<_>>();
    Ok(json!({"SchemaVersion":1,"Results":results}))
}

pub(super) fn identity() -> Value {
    json!({"Kind":"independent-rust-grammar-parser","SupportedSourceIds":[ID],
        "UpstreamCommit":super::UPSTREAM,
        "AssertionSource":{"Path":ID,"Sha256":"4c3f3b8ba3f4aa426b750815c7420c5af88672037c98f324168038dedd913ece"},
        "Architecture":"character-lexer-owned-syntax-tree-and-separate-lowering-v1",
        "OriginalVariants":23,"OriginalAssertions":274,
        "NativeReferenceReplay":false,"ProductionRuntimeCalled":false,
        "CompleteSourceGate":false,"NativeRetirementQualified":false})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, Target};

    fn request(grammar: &str) -> Value {
        json!({"SchemaVersion":1,"Operations":[{"Kind":"grammar-parser-parse","Grammar":grammar}]})
    }

    fn observe(grammar: &str) -> Result<Value> {
        Ok(execute(&request(grammar))?["Results"][0].clone())
    }

    #[test]
    fn independent_grammar_parser_preserves_all_274_source_assertions_and_both_arms() -> Result<()>
    {
        for target in [Target::RustReference, Target::Cpp, Target::Rust] {
            let engine = Engine::new(target, None, None, None, None, None)?;
            let report = crate::grammar_parser::run(&engine)?;
            assert_eq!(report["Id"], ID);
            assert_eq!(report["Status"], "pass", "{report}");
            assert!(
                report["Detail"]
                    .as_str()
                    .is_some_and(|detail| detail.contains("274 source assertions")
                        && detail.contains("provider="))
            );
        }
        assert!(crate::reference_provider::supports(ID));
        assert_eq!(identity()["NativeReferenceReplay"], false);
        assert_eq!(identity()["ProductionRuntimeCalled"], false);
        Ok(())
    }

    #[test]
    fn independent_grammar_parser_generic_unicode_groups_and_lowering_have_literal_answers()
    -> Result<()> {
        let observed = observe("root ::= item{1,2} !<[42]>\nitem ::= \"雪\" | [a-z]\n")?;
        assert_eq!(
            observed,
            json!({"Parsed":true,
            "SymbolIds":[{"Name":"item","Id":1},{"Name":"root","Id":0},{"Name":"root_2","Id":2}],
            "Rules":[
                [{"Type":2,"Value":1},{"Type":2,"Value":2},{"Type":9,"Value":42},{"Type":0,"Value":0}],
                [{"Type":3,"Value":38634},{"Type":1,"Value":0},{"Type":3,"Value":97},{"Type":5,"Value":122},{"Type":0,"Value":0}],
                [{"Type":2,"Value":1},{"Type":1,"Value":0},{"Type":0,"Value":0}]]})
        );
        assert_eq!(
            observe("# comment\r\nroot ::= (\"a\" |\n \"b\")? <[7]>\r\n")?,
            json!({"Parsed":true,
            "SymbolIds":[{"Name":"root","Id":0},{"Name":"root_1","Id":1},{"Name":"root_2","Id":2}],
            "Rules":[
                [{"Type":2,"Value":2},{"Type":8,"Value":7},{"Type":0,"Value":0}],
                [{"Type":3,"Value":97},{"Type":1,"Value":0},{"Type":3,"Value":98},{"Type":0,"Value":0}],
                [{"Type":2,"Value":1},{"Type":1,"Value":0},{"Type":0,"Value":0}]]})
        );
        assert_eq!(
            observe("root ::= \"\\u96ea\\x41\\n\"")?["Rules"],
            json!([[
                {"Type":3,"Value":38634},{"Type":3,"Value":65},{"Type":3,"Value":10},{"Type":0,"Value":0}
            ]])
        );
        Ok(())
    }

    #[test]
    fn independent_grammar_parser_rejects_syntax_expansion_and_never_reuses_failed_rules()
    -> Result<()> {
        for grammar in [
            "root ::= missing",
            "root ::= [unterminated",
            "root ::= \"unterminated",
            "root ::= \"x\"{2,1}",
            "root ::= \"x\"{2000}",
            "root ::= *",
        ] {
            assert_eq!(
                observe(grammar)?,
                json!({"Parsed":false,"SymbolIds":[],"Rules":[]})
            );
        }
        let valid = observe("root ::= \"x\"{1999}")?;
        assert_eq!(valid["Parsed"], true);
        assert_eq!(valid["Rules"][0].as_array().map(Vec::len), Some(2000));
        let nested = format!("root ::= {}\"x\"{}", "(".repeat(257), ")".repeat(257));
        assert_eq!(observe(&nested)?["Parsed"], false);
        let batch = json!({"SchemaVersion":1,"Operations":[
            {"Kind":"grammar-parser-parse","Grammar":"root ::= \"old\""},
            {"Kind":"grammar-parser-parse","Grammar":"root ::= missing"},
            {"Kind":"grammar-parser-parse","Grammar":"fresh ::= ."}]});
        let results = execute(&batch)?;
        assert_eq!(results["Results"][1]["Rules"], json!([]));
        assert_eq!(
            results["Results"][2]["SymbolIds"],
            json!([{"Name":"fresh","Id":0}])
        );
        Ok(())
    }

    #[test]
    fn independent_grammar_parser_typed_protocol_refuses_unsupported_batches() {
        let valid = request("root ::= \"x\"");
        for grammar in [
            json!(""),
            json!("x\0"),
            json!("x".repeat(65_537)),
            json!(null),
            json!([]),
        ] {
            let mut changed = valid.clone();
            changed["Operations"][0]["Grammar"] = grammar;
            assert!(execute(&changed).is_err());
        }
        for (field, value) in [("Kind", json!("grammar-runtime")), ("Unknown", json!(0))] {
            let mut changed = valid.clone();
            changed["Operations"][0][field] = value;
            assert!(execute(&changed).is_err());
        }
        for operations in [json!([]), json!(vec![valid["Operations"][0].clone(); 129])] {
            let mut changed = valid.clone();
            changed["Operations"] = operations;
            assert!(execute(&changed).is_err());
        }
        for version in [json!(2), json!(true)] {
            let mut changed = valid.clone();
            changed["SchemaVersion"] = version;
            assert!(execute(&changed).is_err());
        }
    }
}
