//! Bounded Jinja lexer and expression/statement parser.

use crate::{Result, invalid};

#[derive(Clone, Debug)]
pub(super) enum Literal {
    Int(i64),
    Float(f64),
    Bytes(Vec<u8>),
}
#[derive(Clone, Debug)]
pub(super) enum Arg {
    Pos(Expr),
    Keyword(String, Expr),
    Spread(Expr),
}
#[derive(Clone, Debug)]
pub(super) enum Expr {
    Literal(Literal),
    Var(String),
    Array(Vec<Self>),
    Tuple(Vec<Self>),
    Object(Vec<(Self, Self)>),
    Member(Box<Self>, Box<Self>),
    Attribute(Box<Self>, String),
    Slice(Option<Box<Self>>, Option<Box<Self>>, Option<Box<Self>>),
    Blank,
    Call(Box<Self>, Vec<Arg>),
    Filter(Box<Self>, String, Vec<Arg>),
    Test(Box<Self>, String, Vec<Arg>, bool),
    Unary(String, Box<Self>),
    Binary(String, Box<Self>, Box<Self>),
    Ternary(Box<Self>, Box<Self>, Box<Self>),
    Select(Box<Self>, Box<Self>),
}
#[derive(Clone, Debug)]
pub(super) enum Stmt {
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
enum Token {
    Id(String),
    Num(String),
    Str(Vec<u8>),
    Op(String),
}
fn lex(text: &str) -> Result<Vec<Token>> {
    let bytes = text.as_bytes();
    let mut position = 0;
    let mut tokens = Vec::new();
    while position < bytes.len() {
        let byte = bytes[position];
        if byte.is_ascii_whitespace() {
            position += 1;
            continue;
        }
        if byte.is_ascii_alphabetic() || byte == b'_' {
            let start = position;
            position += 1;
            while bytes
                .get(position)
                .is_some_and(|value| value.is_ascii_alphanumeric() || *value == b'_')
            {
                position += 1;
            }
            tokens.push(Token::Id(text[start..position].into()));
        } else if byte.is_ascii_digit() {
            let start = position;
            while bytes.get(position).is_some_and(u8::is_ascii_digit) {
                position += 1;
            }
            if bytes.get(position) == Some(&b'.')
                && bytes.get(position + 1).is_some_and(u8::is_ascii_digit)
            {
                position += 1;
                while bytes.get(position).is_some_and(u8::is_ascii_digit) {
                    position += 1;
                }
            }
            tokens.push(Token::Num(text[start..position].into()));
        } else if matches!(byte, b'\'' | b'"') {
            let quote = byte;
            position += 1;
            let mut value = Vec::new();
            while position < bytes.len() && bytes[position] != quote {
                let byte = bytes[position];
                position += 1;
                if byte == b'\\' {
                    let next = *bytes
                        .get(position)
                        .ok_or_else(|| invalid("Jinja trailing escape"))?;
                    position += 1;
                    value.push(match next {
                        b'n' => b'\n',
                        b't' => b'\t',
                        b'r' => b'\r',
                        b'b' => 8,
                        b'f' => 12,
                        b'v' => 11,
                        b'\\' | b'\'' | b'"' => next,
                        _ => return Err(invalid("unknown Jinja string escape")),
                    });
                } else {
                    value.push(byte);
                }
            }
            if position == bytes.len() {
                return Err(invalid("unterminated Jinja string"));
            }
            position += 1;
            tokens.push(Token::Str(value));
        } else {
            if !byte.is_ascii() {
                return Err(invalid("unsupported non-ASCII Jinja expression token"));
            }
            let length = if bytes
                .get(position..position + 2)
                .is_some_and(|pair| matches!(pair, b"==" | b"!=" | b"<=" | b">="))
            {
                2
            } else {
                1
            };
            let op = &text[position..position + length];
            if !op
                .bytes()
                .all(|value| b"+-~*/%<>=!()[]{}.,:|".contains(&value))
            {
                return Err(invalid("unknown Jinja expression character"));
            }
            tokens.push(Token::Op(op.into()));
            position += length;
        }
        if tokens.len() > 2048 {
            return Err(invalid("Jinja expression token limit exceeded"));
        }
    }
    Ok(tokens)
}

struct Expression {
    tokens: Vec<Token>,
    position: usize,
    depth: usize,
}
impl Expression {
    fn new(text: &str) -> Result<Self> {
        Ok(Self {
            tokens: lex(text)?,
            position: 0,
            depth: 0,
        })
    }
    fn word(&self, value: &str) -> bool {
        matches!(self.tokens.get(self.position),Some(Token::Id(text)|Token::Op(text)) if text==value)
    }
    fn eat(&mut self, value: &str) -> bool {
        if self.word(value) {
            self.position += 1;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, value: &str) -> Result<()> {
        if self.eat(value) {
            Ok(())
        } else {
            Err(invalid(format!("expected Jinja token {value}")))
        }
    }
    fn identifier(&mut self) -> Result<String> {
        if let Some(Token::Id(value)) = self.tokens.get(self.position) {
            let value = value.clone();
            self.position += 1;
            Ok(value)
        } else {
            Err(invalid("Jinja identifier expected"))
        }
    }
    fn complete(mut self) -> Result<Expr> {
        let result = self.sequence()?;
        if self.position != self.tokens.len() {
            return Err(invalid("trailing Jinja expression tokens"));
        }
        Ok(result)
    }
    fn sequence(&mut self) -> Result<Expr> {
        let first = self.expression()?;
        if !self.eat(",") {
            return Ok(first);
        }
        let mut values = vec![first];
        while self.position < self.tokens.len() && !self.word(")") && !self.word("=") {
            values.push(self.expression()?);
            if !self.eat(",") {
                break;
            }
        }
        Ok(Expr::Tuple(values))
    }
    fn expression(&mut self) -> Result<Expr> {
        self.depth += 1;
        if self.depth > 128 {
            return Err(invalid("Jinja expression depth exceeded"));
        }
        let mut value = self.binary(0)?;
        if self.eat("if") {
            let condition = self.binary(0)?;
            value = if self.eat("else") {
                Expr::Ternary(
                    Box::new(condition),
                    Box::new(value),
                    Box::new(self.expression()?),
                )
            } else {
                Expr::Select(Box::new(value), Box::new(condition))
            };
        }
        self.depth -= 1;
        Ok(value)
    }
    fn binary(&mut self, level: usize) -> Result<Expr> {
        if level == 2 && self.eat("not") {
            self.depth += 1;
            if self.depth > 128 {
                return Err(invalid("Jinja unary nesting bound exceeded"));
            }
            let expression = self.binary(level)?;
            self.depth -= 1;
            return Ok(Expr::Unary("not".into(), Box::new(expression)));
        }
        if level >= 6 {
            return self.test();
        }
        let mut value = self.binary(level + 1)?;
        let mut count = 0;
        while let Some(Token::Id(token) | Token::Op(token)) = self.tokens.get(self.position) {
            let operator = token.clone();
            let operator = if operator == "not"
                && matches!(self.tokens.get(self.position+1),Some(Token::Id(value)) if value=="in")
            {
                "not in".to_owned()
            } else {
                operator
            };
            let accepted = match level {
                0 => operator == "or",
                1 => operator == "and",
                3 => matches!(
                    operator.as_str(),
                    "in" | "not in" | "==" | "!=" | "<" | ">" | "<=" | ">="
                ),
                4 => matches!(operator.as_str(), "+" | "-" | "~"),
                5 => matches!(operator.as_str(), "*" | "/" | "%"),
                _ => false,
            };
            if !accepted {
                break;
            }
            count += 1;
            if count > 128 {
                return Err(invalid("Jinja operator chain bound exceeded"));
            }
            self.position += if operator == "not in" { 2 } else { 1 };
            let right = self.binary(level + 1)?;
            value = Expr::Binary(operator, Box::new(value), Box::new(right));
        }
        Ok(value)
    }
    fn test(&mut self) -> Result<Expr> {
        let mut value = self.filter()?;
        while self.eat("is") {
            let negative = self.eat("not");
            let name = self.identifier()?;
            let args = if self.word("(") {
                self.args()?
            } else {
                Vec::new()
            };
            value = Expr::Test(Box::new(value), name, args, negative);
        }
        Ok(value)
    }
    fn filter(&mut self) -> Result<Expr> {
        let mut value = self.postfix()?;
        while self.eat("|") {
            let name = self.identifier()?;
            let args = if self.word("(") {
                self.args()?
            } else {
                Vec::new()
            };
            value = Expr::Filter(Box::new(value), name, args);
        }
        Ok(value)
    }
    fn args(&mut self) -> Result<Vec<Arg>> {
        self.expect("(")?;
        let mut values = Vec::new();
        while !self.word(")") {
            let value = if self.eat("*") {
                Arg::Spread(self.expression()?)
            } else {
                let value = self.expression()?;
                if self.eat("=") {
                    let Expr::Var(name) = value else {
                        return Err(invalid("Jinja keyword argument name invalid"));
                    };
                    Arg::Keyword(name, self.expression()?)
                } else {
                    Arg::Pos(value)
                }
            };
            values.push(value);
            if !self.eat(",") {
                break;
            }
        }
        self.expect(")")?;
        Ok(values)
    }
    fn postfix(&mut self) -> Result<Expr> {
        let mut value = self.primary()?;
        let mut count = 0;
        loop {
            count += 1;
            if count > 128 {
                return Err(invalid("Jinja member chain bound exceeded"));
            }
            if self.eat(".") {
                let property = self.identifier()?;
                value = Expr::Attribute(Box::new(value), property);
            } else if self.eat("[") {
                let first = if self.word(":") || self.word("]") {
                    None
                } else {
                    Some(Box::new(self.expression()?))
                };
                let property = if self.eat(":") {
                    let stop = if self.word(":") || self.word("]") {
                        None
                    } else {
                        Some(Box::new(self.expression()?))
                    };
                    let step = if self.eat(":") && !self.word("]") {
                        Some(Box::new(self.expression()?))
                    } else {
                        None
                    };
                    Expr::Slice(first, stop, step)
                } else {
                    first.map_or(Expr::Blank, |value| *value)
                };
                self.expect("]")?;
                value = Expr::Member(Box::new(value), Box::new(property));
            } else if self.word("(") {
                value = Expr::Call(Box::new(value), self.args()?);
            } else {
                break;
            }
        }
        Ok(value)
    }
    fn primary(&mut self) -> Result<Expr> {
        let token = self
            .tokens
            .get(self.position)
            .cloned()
            .ok_or_else(|| invalid("unexpected Jinja expression end"))?;
        self.position += 1;
        Ok(match token {
            Token::Id(value) => Expr::Var(value),
            Token::Num(value) => Expr::Literal(if value.contains('.') {
                Literal::Float(value.parse()?)
            } else {
                Literal::Int(value.parse()?)
            }),
            Token::Str(mut value) => {
                while let Some(Token::Str(next)) = self.tokens.get(self.position) {
                    value.extend(next);
                    self.position += 1;
                }
                Expr::Literal(Literal::Bytes(value))
            }
            Token::Op(value) if value == "+" || value == "-" => {
                self.depth += 1;
                if self.depth > 128 {
                    return Err(invalid("Jinja unary nesting bound exceeded"));
                }
                let expression = self.primary()?;
                self.depth -= 1;
                Expr::Unary(value, Box::new(expression))
            }
            Token::Op(value) if value == "(" => {
                let value = self.sequence()?;
                self.expect(")")?;
                value
            }
            Token::Op(value) if value == "[" => {
                let mut values = Vec::new();
                while !self.word("]") {
                    values.push(self.expression()?);
                    if !self.eat(",") {
                        break;
                    }
                }
                self.expect("]")?;
                Expr::Array(values)
            }
            Token::Op(value) if value == "{" => {
                let mut values = Vec::new();
                while !self.word("}") {
                    let key = self.expression()?;
                    self.expect(":")?;
                    let value = self.expression()?;
                    values.push((key, value));
                    if !self.eat(",") {
                        break;
                    }
                }
                self.expect("}")?;
                Expr::Object(values)
            }
            Token::Op(_) => return Err(invalid("unexpected Jinja primary token")),
        })
    }
}

enum Segment {
    Text(Vec<u8>),
    Expression(String),
    Statement(String),
}
#[allow(
    clippy::too_many_lines,
    reason = "Keep delimiter scanning and adjacent whitespace ownership in a single pass"
)]
fn segments(source: &str) -> Result<Vec<Segment>> {
    if source.len() > 65536 {
        return Err(invalid("Jinja template exceeds byte bound"));
    }
    let remove_final_newline = source.ends_with('\n');
    let mut source = source.replace("\r\n", "\n").replace('\r', "\n");
    if remove_final_newline {
        source.pop();
    }
    let bytes = source.as_bytes();
    let mut result = Vec::new();
    let mut position = 0;
    let mut strip = false;
    while position < bytes.len() {
        let next = (position..bytes.len().saturating_sub(1))
            .find(|&index| bytes[index] == b'{' && matches!(bytes[index + 1], b'{' | b'%' | b'#'));
        let end = next.unwrap_or(bytes.len());
        let mut literal = bytes[position..end].to_vec();
        if strip {
            let count = literal
                .iter()
                .take_while(|value| value.is_ascii_whitespace())
                .count();
            literal.drain(..count);
            strip = false;
        }
        if let Some(start) = next {
            let kind = bytes[start + 1];
            let trim = bytes.get(start + 2) == Some(&b'-');
            if trim {
                while literal.last().is_some_and(u8::is_ascii_whitespace) {
                    literal.pop();
                }
            } else if matches!(kind, b'%' | b'#') {
                let line = bytes[..start]
                    .iter()
                    .rposition(|&value| value == b'\n')
                    .map_or(0, |index| index + 1);
                if bytes[line..start]
                    .iter()
                    .all(|value| matches!(value, b' ' | b'\t'))
                {
                    let keep = literal.len().saturating_sub(start - line);
                    literal.truncate(keep);
                }
            }
            if !literal.is_empty() {
                result.push(Segment::Text(literal));
            }
            let begin = start + if trim { 3 } else { 2 };
            let mut cursor = begin;
            let mut quote = None;
            let mut curly = 0;
            let closing = if kind == b'{' { b'}' } else { kind };
            let mut found = None;
            while cursor + 1 < bytes.len() {
                let value = bytes[cursor];
                if kind == b'#' {
                    if value == b'#' && bytes[cursor + 1] == b'}' {
                        found = Some(cursor);
                        break;
                    }
                } else if let Some(delimiter) = quote {
                    if value == b'\\' {
                        cursor += 2;
                        continue;
                    }
                    if value == delimiter {
                        quote = None;
                    }
                } else if matches!(value, b'\'' | b'"') {
                    quote = Some(value);
                } else if curly == 0 && value == closing && bytes[cursor + 1] == b'}' {
                    if value == closing && bytes[cursor + 1] == b'}' {
                        found = Some(cursor);
                        break;
                    }
                } else if value == b'{' {
                    curly += 1;
                } else if value == b'}' && curly > 0 {
                    curly -= 1;
                }
                cursor += 1;
            }
            let close = found.ok_or_else(|| invalid("unterminated Jinja template block"))?;
            let trim_right = close > begin && bytes[close - 1] == b'-';
            let content = source[begin..close - usize::from(trim_right)].to_owned();
            position = close + 2;
            if kind != b'#' {
                result.push(if kind == b'{' {
                    Segment::Expression(content)
                } else {
                    Segment::Statement(content)
                });
            }
            if trim_right {
                strip = true;
            } else if kind != b'{' && bytes.get(position) == Some(&b'\n') {
                position += 1;
            }
        } else {
            if !literal.is_empty() {
                result.push(Segment::Text(literal));
            }
            break;
        }
    }
    Ok(result)
}

struct Statements {
    segments: Vec<Segment>,
    position: usize,
    depth: usize,
}
impl Statements {
    fn header(&self) -> Option<&str> {
        if let Some(Segment::Statement(value)) = self.segments.get(self.position) {
            Some(value.trim())
        } else {
            None
        }
    }
    fn stop(&self, names: &[&str]) -> bool {
        self.header().is_some_and(|header| {
            names.contains(&header.split_ascii_whitespace().next().unwrap_or(""))
        })
    }
    fn consume(&mut self, name: &str) -> Result<()> {
        if !self.stop(&[name]) {
            return Err(invalid(format!("expected Jinja closing block {name}")));
        }
        self.position += 1;
        Ok(())
    }
    #[allow(
        clippy::too_many_lines,
        reason = "Keep source statement block nesting and expression ownership together"
    )]
    fn body(&mut self, stops: &[&str]) -> Result<Vec<Stmt>> {
        self.depth += 1;
        if self.depth > 128 {
            return Err(invalid("Jinja statement depth exceeded"));
        }
        let mut result = Vec::new();
        while self.position < self.segments.len() && !self.stop(stops) {
            let index = self.position;
            self.position += 1;
            match &self.segments[index] {
                Segment::Text(bytes) => result.push(Stmt::Text(bytes.clone())),
                Segment::Expression(expression) => {
                    result.push(Stmt::Output(Expression::new(expression)?.complete()?));
                }
                Segment::Statement(header) => {
                    let header = header.trim().to_owned();
                    let end = header
                        .find(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_')
                        .unwrap_or(header.len());
                    let (name, rest) = header.split_at(end);
                    let rest = rest.trim_start();
                    match name {
                        "if" => {
                            let mut branches = Vec::new();
                            let mut condition = Expression::new(rest)?.complete()?;
                            loop {
                                let body = self.body(&["elif", "else", "endif"])?;
                                branches.push((condition, body));
                                if self.stop(&["elif"]) {
                                    let header = self.header().unwrap_or("").to_owned();
                                    self.position += 1;
                                    condition =
                                        Expression::new(header.strip_prefix("elif").unwrap_or(""))?
                                            .complete()?;
                                } else {
                                    break;
                                }
                            }
                            let alternate = if self.stop(&["else"]) {
                                self.position += 1;
                                self.body(&["endif"])?
                            } else {
                                Vec::new()
                            };
                            self.consume("endif")?;
                            result.push(Stmt::If(branches, alternate));
                        }
                        "for" => {
                            let mut parser = Expression::new(rest)?;
                            let mut targets = vec![Expr::Var(parser.identifier()?)];
                            while parser.eat(",") {
                                targets.push(Expr::Var(parser.identifier()?));
                            }
                            parser.expect("in")?;
                            let iterable = parser.expression()?;
                            if parser.position != parser.tokens.len() {
                                return Err(invalid("trailing Jinja loop header"));
                            }
                            let target = if targets.len() == 1 {
                                targets.remove(0)
                            } else {
                                Expr::Tuple(targets)
                            };
                            let body = self.body(&["else", "endfor"])?;
                            let alternate = if self.stop(&["else"]) {
                                self.position += 1;
                                self.body(&["endfor"])?
                            } else {
                                Vec::new()
                            };
                            self.consume("endfor")?;
                            result.push(Stmt::For(target, iterable, body, alternate));
                        }
                        "set" => {
                            let mut parser = Expression::new(rest)?;
                            let target = parser.sequence()?;
                            let (value, body) = if parser.eat("=") {
                                (Some(parser.sequence()?), Vec::new())
                            } else {
                                let body = self.body(&["endset"])?;
                                self.consume("endset")?;
                                (None, body)
                            };
                            result.push(Stmt::Set(target, value, body));
                        }
                        "macro" => {
                            let mut parser = Expression::new(rest)?;
                            let name = parser.identifier()?;
                            let mut arguments = Vec::new();
                            for arg in parser.args()? {
                                match arg {
                                    Arg::Pos(Expr::Var(name)) => arguments.push((name, None)),
                                    Arg::Keyword(name, value) => {
                                        arguments.push((name, Some(value)));
                                    }
                                    _ => return Err(invalid("invalid Jinja macro parameter")),
                                }
                            }
                            let body = self.body(&["endmacro"])?;
                            self.consume("endmacro")?;
                            result.push(Stmt::Macro(name, arguments, body));
                        }
                        "call" => {
                            let mut parser = Expression::new(rest)?;
                            let arguments = if parser.word("(") {
                                parser
                                    .args()?
                                    .into_iter()
                                    .map(|arg| {
                                        if let Arg::Pos(Expr::Var(name)) = arg {
                                            Ok(name)
                                        } else {
                                            Err(invalid("invalid Jinja caller parameter"))
                                        }
                                    })
                                    .collect::<Result<Vec<_>>>()?
                            } else {
                                Vec::new()
                            };
                            let call = parser.expression()?;
                            let body = self.body(&["endcall"])?;
                            self.consume("endcall")?;
                            result.push(Stmt::Call(arguments, call, body));
                        }
                        "filter" => {
                            let filter = Expression::new(rest)?.complete()?;
                            let body = self.body(&["endfilter"])?;
                            self.consume("endfilter")?;
                            result.push(Stmt::Filter(filter, body));
                        }
                        "break" => result.push(Stmt::Break),
                        "continue" => result.push(Stmt::Continue),
                        "generation" | "endgeneration" => {}
                        _ => return Err(invalid(format!("unknown Jinja statement {name}"))),
                    }
                }
            }
            if result.len() > 32768 {
                return Err(invalid("Jinja statement count exceeded"));
            }
        }
        self.depth -= 1;
        Ok(result)
    }
}

pub(super) fn parse(source: &str) -> Result<Vec<Stmt>> {
    let mut parser = Statements {
        segments: segments(source)?,
        position: 0,
        depth: 0,
    };
    parser.body(&[])
}
