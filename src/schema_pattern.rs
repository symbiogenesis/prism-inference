//! Pinned anchored-regex to GBNF conversion, with explicit fallback semantics.

use std::collections::HashMap;

use super::{Converter, Result, invalid, repetition};

enum PatternError {
    Unsupported,
    Invalid,
}
type PatternResult<T> = std::result::Result<T, PatternError>;
type Piece = (String, bool);

fn to_rule(piece: &Piece) -> String {
    if piece.1 {
        format!("\"{}\"", piece.0)
    } else {
        piece.0.clone()
    }
}
fn join(sequence: &[Piece]) -> Piece {
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

const fn nonliteral(value: char) -> bool {
    matches!(
        value,
        '|' | '.' | '(' | ')' | '[' | ']' | '{' | '}' | '*' | '+' | '?' | '^' | '$'
    )
}
fn escape_length(values: &[char], offset: usize) -> usize {
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

struct Parser<'a> {
    converter: &'a mut Converter,
    name: &'a str,
    values: Vec<char>,
    offset: usize,
    depth: usize,
    subrules: HashMap<String, String>,
}
impl Parser<'_> {
    #[allow(
        clippy::too_many_lines,
        reason = "Preserve the source regex transform's ordered token/quantifier behavior"
    )]
    fn transform(&mut self) -> PatternResult<Piece> {
        let mut sequence: Vec<Piece> = Vec::new();
        while self.offset < self.values.len() {
            let value = self.values[self.offset];
            match value {
                '.' => {
                    let rule = if self.converter.dotall {
                        "[\\U00000000-\\U0010FFFF]"
                    } else {
                        "[^\\x0A\\x0D]"
                    };
                    let name = self.converter.add("dot", rule);
                    sequence.push((name, false));
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
                    sequence.push((format!("({})", to_rule(&piece)), false));
                }
                ')' => {
                    self.offset += 1;
                    if self.depth == 0 {
                        return Err(PatternError::Invalid);
                    }
                    self.depth -= 1;
                    return Ok(join(&sequence));
                }
                '^' | '$' => return Err(PatternError::Unsupported),
                '[' => {
                    let mut range = String::from("[");
                    self.offset += 1;
                    while self.offset < self.values.len() && self.values[self.offset] != ']' {
                        if self.values[self.offset] == '\\' {
                            let length = escape_length(&self.values, self.offset);
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
                    piece.0 = format!("{}{value}", to_rule(piece));
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
                        |value: &str| value.parse::<i32>().map_err(|_| PatternError::Unsupported);
                    let (minimum, maximum) = match counts.as_slice() {
                        [count] => {
                            let count = parse(count)?;
                            (count, count)
                        }
                        [min, max] => (
                            if min.is_empty() { 0 } else { parse(min)? },
                            if max.is_empty() {
                                i32::MAX
                            } else {
                                parse(max)?
                            },
                        ),
                        _ => return Err(PatternError::Unsupported),
                    };
                    let piece = sequence.last_mut().ok_or(PatternError::Invalid)?;
                    if !piece.1 {
                        let rule = piece.0.clone();
                        let name = if let Some(name) = self.subrules.get(&rule) {
                            name.clone()
                        } else {
                            let name = self
                                .converter
                                .add(&format!("{}-{}", self.name, self.subrules.len() + 1), &rule);
                            self.subrules.insert(rule, name.clone());
                            name
                        };
                        piece.0 = name;
                    }
                    piece.0 = repetition(&to_rule(piece), minimum, maximum, "");
                    piece.1 = false;
                }
                _ => {
                    let mut literal = String::new();
                    while self.offset < self.values.len() {
                        let value = self.values[self.offset];
                        if value == '\\' {
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
                                let length = escape_length(&self.values, self.offset);
                                if length == 0 {
                                    return Err(PatternError::Unsupported);
                                }
                                literal.extend(&self.values[self.offset..self.offset + length]);
                                self.offset += length;
                            }
                        } else if value == '"' {
                            literal.push_str("\\\"");
                            self.offset += 1;
                        } else if !nonliteral(value)
                            && (self.offset + 1 == self.values.len()
                                || literal.is_empty()
                                || self.values[self.offset + 1] == '.'
                                || !nonliteral(self.values[self.offset + 1]))
                        {
                            literal.push(value);
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
        Ok(join(&sequence))
    }
}

impl Converter {
    pub(super) fn pattern(&mut self, pattern: &str, name: &str) -> Result<String> {
        let snapshot = self.rules.clone();
        let result = (|| -> PatternResult<String> {
            if pattern.len() < 2 || !pattern.starts_with('^') || !pattern.ends_with('$') {
                return Err(PatternError::Unsupported);
            }
            let mut parser = Parser {
                converter: self,
                name,
                values: pattern[1..pattern.len() - 1].chars().collect(),
                offset: 0,
                depth: 0,
                subrules: HashMap::new(),
            };
            let rule = to_rule(&parser.transform()?);
            if parser.depth != 0 {
                return Err(PatternError::Invalid);
            }
            Ok(format!("\"\\\"\" ({rule}) \"\\\"\""))
        })();
        match result {
            Ok(rule) => Ok(self.add(name, &rule)),
            Err(PatternError::Unsupported) => {
                self.rules = snapshot;
                let rule = self.primitive("string", "string")?;
                Ok(self.add(name, &rule))
            }
            Err(PatternError::Invalid) => {
                self.rules = snapshot;
                Err(invalid(format!("invalid schema pattern {pattern}")))
            }
        }
    }
}
