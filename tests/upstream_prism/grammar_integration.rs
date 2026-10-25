//! Safe Rust ownership of all 11 test-grammar-integration.cpp groups.
//!
//! Source: PrismML-Eng/llama.cpp b87ca7225456b06b0975b8239c66b5aecf07e37c;
//! see tools/prism/upstream/LICENSE for MIT terms. Literal grammar/schema text,
//! encoded input bytes, pass/fail expectations, token parsing and completion
//! decisions belong to Rust. Native code supplies production state observations.

use std::ops::Range;

use serde_json::{Value, json};

use super::engine::Engine;
use super::llama_grammar::{array, exact_fields, exchange, field, pointed_element, uint};
use super::{Result, invalid, result};

const ID: &str = "tests/test-grammar-integration.cpp";
const SOURCE_SHA256: &str = "6acd7735c8f6db1d3869bd2de5ea7a585accc75e3f8627eb163be11bc123716a";
const CASES: &str = include_str!("grammar-integration-cases.json");

#[derive(Default)]
struct Assertions {
    count: usize,
    failures: Vec<String>,
}

impl Assertions {
    fn check(&mut self, name: impl Into<String>, passed: bool) {
        self.count += 1;
        if !passed {
            self.failures.push(name.into());
        }
    }
}

fn text<'a>(value: &'a Value, name: &str) -> Result<&'a str> {
    field(value, name)?
        .as_str()
        .ok_or_else(|| invalid(format!("grammar integration {name} must be text")))
}

fn bytes(value: &Value) -> Result<Vec<u8>> {
    value
        .as_array()
        .ok_or_else(|| invalid("grammar input must be bytes"))?
        .iter()
        .map(|byte| {
            Ok(u8::try_from(byte.as_u64().ok_or_else(|| {
                invalid("grammar input byte must be unsigned")
            })?)?)
        })
        .collect()
}

#[derive(Debug, PartialEq)]
struct Token {
    id: i32,
    piece: Vec<u8>,
}

fn decode_utf8(input: &[u8]) -> Option<(u32, usize)> {
    let first = *input.first()?;
    if first & 0x80 == 0 {
        return Some((u32::from(first), 1));
    }
    if first & 0x40 == 0 {
        return None;
    }
    let (width, mask) = if first & 0x20 == 0 {
        (2, 0x1f)
    } else if first & 0x10 == 0 {
        (3, 0x0f)
    } else if first & 0x08 == 0 {
        (4, 0x07)
    } else {
        return None;
    };
    if input.len() < width || input[1..width].iter().any(|byte| byte & 0xc0 != 0x80) {
        return None;
    }
    let mut point = u32::from(first & mask);
    for byte in &input[1..width] {
        point = (point << 6) | u32::from(byte & 0x3f);
    }
    Some((point, width))
}

fn encode_utf8(point: u32) -> Option<Vec<u8>> {
    let low = |value: u32| value.to_le_bytes()[0];
    Some(if point <= 0x7f {
        vec![low(point)]
    } else if point <= 0x7ff {
        vec![0xc0 | low((point >> 6) & 0x1f), 0x80 | low(point & 0x3f)]
    } else if point <= 0xffff {
        vec![
            0xe0 | low((point >> 12) & 0x0f),
            0x80 | low((point >> 6) & 0x3f),
            0x80 | low(point & 0x3f),
        ]
    } else if point <= 0x10_ffff {
        vec![
            0xf0 | low((point >> 18) & 0x07),
            0x80 | low((point >> 12) & 0x3f),
            0x80 | low((point >> 6) & 0x3f),
            0x80 | low(point & 0x3f),
        ]
    } else {
        return None;
    })
}

// Mirror the original helper, including its legacy UTF-8 decoder: malformed
// leading/continuation bytes consume one byte and become U+FFFD; overlong
// sequences are decoded and re-encoded rather than rejected by Rust's str API.
fn parse_tokens(input: &[u8]) -> Result<Vec<Token>> {
    let mut output = Vec::new();
    let mut offset = 0;
    while offset < input.len() {
        if input[offset] == 0xff {
            let encoded = input
                .get(offset + 1..offset + 5)
                .ok_or_else(|| invalid("not enough bytes for token id"))?;
            let encoded = <[u8; 4]>::try_from(encoded)?;
            output.push(Token {
                id: i32::from_be_bytes(encoded),
                piece: format!("<[{}]>", u32::from_be_bytes(encoded)).into_bytes(),
            });
            offset += 5;
        } else if let Some((point, width)) = decode_utf8(&input[offset..]) {
            offset += width;
            if let Some(piece) = encode_utf8(point) {
                output.push(Token { id: 0, piece });
            } else {
                // unicode_cpt_to_utf8 throws after the decoder has advanced;
                // the source catch then advances one additional byte.
                offset += 1;
                output.push(Token {
                    id: 0,
                    piece: vec![0xef, 0xbf, 0xbd],
                });
            }
        } else {
            offset += 1;
            output.push(Token {
                id: 0,
                piece: vec![0xef, 0xbf, 0xbd],
            });
        }
    }
    Ok(output)
}

fn accept(grammar: u32, token: &Token) -> Value {
    json!({"Kind":"accept-token","Grammar":grammar,"Token":token.id,"PieceBytes":token.piece})
}

fn initialize(grammar: &str, root: &str) -> Value {
    json!({"Kind":"init-text","Grammar":0,"GrammarText":grammar,"Root":root})
}

fn native_trace<'a>(value: &'a Value, actions: &[Value]) -> Result<&'a [Value]> {
    exact_fields(value, &["Trace"])?;
    let trace = array(value, "Trace")?;
    if trace.len() != actions.len() {
        return Err(invalid("grammar integration trace extent mismatch"));
    }
    for (action, value) in actions.iter().zip(trace) {
        if matches!(
            text(action, "Kind")?,
            "free" | "save-stacks" | "restore-stacks"
        ) {
            exact_fields(value, &[])?;
        }
    }
    Ok(trace)
}

fn created(value: &Value) -> Result<bool> {
    exact_fields(value, &["Created"])?;
    field(value, "Created")?
        .as_bool()
        .ok_or_else(|| invalid("grammar Created must be boolean"))
}

fn stack_sizes(value: &Value) -> Result<Vec<u64>> {
    array(value, "StackSizes")?
        .iter()
        .map(|size| {
            size.as_u64()
                .ok_or_else(|| invalid("grammar stack size must be unsigned"))
        })
        .collect()
}

fn accepted(value: &Value) -> Result<Option<Vec<u64>>> {
    exact_fields(value, &["ExceptionKind", "Message", "StackSizes"])?;
    let _ = text(value, "Message")?;
    let sizes = stack_sizes(value)?;
    match field(value, "ExceptionKind")? {
        Value::Null => Ok(Some(sizes)),
        Value::String(kind) if kind == "runtime" => Ok(None),
        _ => Err(invalid(
            "unexpected exception during production grammar acceptance",
        )),
    }
}

fn matches(initial: &Value, observations: &[Value]) -> Result<bool> {
    exact_fields(initial, &["StackSizes"])?;
    let mut sizes = stack_sizes(initial)?;
    for value in observations {
        let Some(next) = accepted(value)? else {
            return Ok(false);
        };
        if next.is_empty() {
            return Ok(false);
        }
        sizes = next;
    }
    Ok(sizes.contains(&0))
}

struct InputCheck {
    expected: bool,
    initial: usize,
    accepts: Range<usize>,
}

fn literal_operation(case: &Value, grammar: &str) -> Result<(Value, Vec<InputCheck>)> {
    let mut actions = vec![
        initialize(grammar, "root"),
        json!({"Kind":"save-stacks","Grammar":0,"StackHandle":0}),
    ];
    let mut checks = Vec::new();
    for (field_name, expected) in [("PassingBytes", true), ("FailingBytes", false)] {
        for input in array(case, field_name)? {
            let initial = actions.len();
            actions.push(json!({"Kind":"state","Grammar":0}));
            let start = actions.len();
            actions.extend(
                parse_tokens(&bytes(input)?)?
                    .iter()
                    .map(|token| accept(0, token)),
            );
            checks.push(InputCheck {
                expected,
                initial,
                accepts: start..actions.len(),
            });
            actions.push(json!({"Kind":"restore-stacks","Grammar":0,"StackHandle":0}));
        }
    }
    actions.push(json!({"Kind":"free","Grammar":0}));
    Ok((json!({"Kind":"grammar-trace","Actions":actions}), checks))
}

fn convert_schema(engine: &Engine, schema: &str) -> Result<Option<String>> {
    let operation = json!({"Kind":"json-schema-to-grammar","SchemaJson":schema,"ForceGbnf":true});
    let Some(results) = exchange(engine, &[operation])? else {
        return Ok(None);
    };
    exact_fields(&results[0], &["GrammarText"])?;
    Ok(Some(text(&results[0], "GrammarText")?.to_owned()))
}

fn literals(engine: &Engine, cases: &[Value], assertions: &mut Assertions) -> Result<bool> {
    for case in cases {
        let definition = text(case, "Definition")?;
        let grammar = match text(case, "Kind")? {
            "grammar" => definition.to_owned(),
            "schema" => {
                let Some(grammar) = convert_schema(engine, definition)? else {
                    return Ok(false);
                };
                grammar
            }
            _ => return Err(invalid("unknown grammar integration literal kind")),
        };
        let (operation, checks) = literal_operation(case, &grammar)?;
        let Some(values) = exchange(engine, std::slice::from_ref(&operation))? else {
            return Ok(false);
        };
        let actions = array(&operation, "Actions")?;
        let trace = native_trace(&values[0], actions)?;
        if !created(&trace[0])? {
            return Err(invalid(format!(
                "{}.{}: original integration grammar failed to initialize\n{grammar}",
                text(case, "Group")?,
                text(case, "Name")?
            )));
        }
        for (index, check) in checks.iter().enumerate() {
            // The source stops at the first rejection. A finite action vector
            // may include subsequent, unobserved runtime errors on empty stacks;
            // only this Rust reference decides the original match result.
            let actual = matches(&trace[check.initial], &trace[check.accepts.clone()])?;
            assertions.check(
                format!(
                    "{}.{} input {index}",
                    text(case, "Group")?,
                    text(case, "Name")?
                ),
                actual == check.expected,
            );
        }
    }
    Ok(true)
}

fn parser_failures(engine: &Engine, fixture: &Value, assertions: &mut Assertions) -> Result<()> {
    for case in array(fixture, "ParserCases")? {
        let operation = json!({"Kind":"grammar-parser-parse","Grammar":text(case,"Definition")?});
        let values = exchange(engine, &[operation])?
            .ok_or_else(|| invalid("grammar production backend disappeared"))?;
        let value = &values[0];
        exact_fields(value, &["Parsed", "SymbolIds", "Rules"])?;
        let _ = field(value, "Parsed")?
            .as_bool()
            .ok_or_else(|| invalid("grammar Parsed must be boolean"))?;
        let expected = field(case, "ExpectedRulesEmpty")?
            .as_bool()
            .ok_or_else(|| invalid("expected RulesEmpty must be boolean"))?;
        assertions.check(
            format!("{}: rules emptiness", text(case, "Group")?),
            array(value, "Rules")?.is_empty() == expected,
        );
        if field(case, "ExpectedRootAbsent").is_ok_and(|value| value.as_bool() == Some(true)) {
            let symbols = array(value, "SymbolIds")?;
            let mut root = false;
            for symbol in symbols {
                if text(symbol, "Name")? == "root" {
                    root = true;
                }
            }
            assertions.check("missing root node: root symbol absent", !root);
        }
    }
    Ok(())
}

fn build_failures(engine: &Engine, fixture: &Value, assertions: &mut Assertions) -> Result<()> {
    for case in array(fixture, "BuildCases")? {
        let actions = [
            initialize(text(case, "Definition")?, text(case, "Root")?),
            json!({"Kind":"free","Grammar":0}),
        ];
        let operation = json!({"Kind":"grammar-trace","Actions":actions});
        let values = exchange(engine, &[operation])?
            .ok_or_else(|| invalid("grammar production backend disappeared"))?;
        let trace = native_trace(&values[0], &actions)?;
        let expected = field(case, "ExpectedCreated")?
            .as_bool()
            .ok_or_else(|| invalid("expected Created must be boolean"))?;
        assertions.check(
            format!("{}.{}", text(case, "Group")?, text(case, "Name")?),
            created(&trace[0])? == expected,
        );
    }
    Ok(())
}

fn pointers_belong_to_clone(snapshot: &Value) -> Result<bool> {
    exact_fields(
        snapshot,
        &["Rules", "RuleAddresses", "ElementSize", "Stacks"],
    )?;
    for stack in array(snapshot, "Stacks")? {
        for address in stack
            .as_array()
            .ok_or_else(|| invalid("cloned stack must be an array"))?
        {
            let address = address
                .as_u64()
                .ok_or_else(|| invalid("cloned pointer must be unsigned"))?;
            if pointed_element(snapshot, address)?.is_none() {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn clone_operation(
    grammar: &str,
    input: &[u8],
    split: usize,
) -> Result<(Value, usize, usize, Range<usize>)> {
    let mut actions = vec![initialize(grammar, "root")];
    actions.extend(
        parse_tokens(&input[..split])?
            .iter()
            .map(|token| accept(0, token)),
    );
    let cloned = actions.len();
    actions.push(json!({"Kind":"clone","Grammar":0,"Target":1}));
    actions.push(json!({"Kind":"free","Grammar":0}));
    let snapshot = actions.len();
    actions.push(json!({"Kind":"snapshot","Grammar":1}));
    actions.push(json!({"Kind":"state","Grammar":1}));
    let suffix = actions.len();
    actions.extend(
        parse_tokens(&input[split..])?
            .iter()
            .map(|token| accept(1, token)),
    );
    let range = suffix..actions.len();
    actions.push(json!({"Kind":"free","Grammar":1}));
    Ok((
        json!({"Kind":"grammar-trace","Actions":actions}),
        cloned,
        snapshot,
        range,
    ))
}

fn clone_assertions(engine: &Engine, fixture: &Value, assertions: &mut Assertions) -> Result<()> {
    let fixture = field(fixture, "Clone")?;
    let input = bytes(field(fixture, "InputBytes")?)?;
    if input.len() != 42 {
        return Err(invalid("all 43 original clone split points are required"));
    }
    let grammar = convert_schema(engine, text(fixture, "SchemaJson")?)?
        .ok_or_else(|| invalid("grammar production backend disappeared"))?;
    for split in 0..=input.len() {
        let (operation, cloned, snapshot, suffix) = clone_operation(&grammar, &input, split)?;
        let values = exchange(engine, std::slice::from_ref(&operation))?
            .ok_or_else(|| invalid("grammar production backend disappeared"))?;
        let trace = native_trace(&values[0], array(&operation, "Actions")?)?;
        if !created(&trace[0])? || !created(&trace[cloned])? {
            return Err(invalid(
                "original clone fixture failed to initialize or clone",
            ));
        }
        for value in &trace[1..cloned] {
            if accepted(value)?.is_none() {
                return Err(invalid("original clone prefix was rejected"));
            }
        }
        assertions.check(
            format!("clone split {split}: pointers into cloned rules"),
            pointers_belong_to_clone(&trace[snapshot])?,
        );
        assertions.check(
            format!("clone split {split}: suffix matches"),
            matches(&trace[snapshot + 1], &trace[suffix])?,
        );
    }
    Ok(())
}

pub(super) fn run(engine: &Engine) -> Result<Value> {
    let fixture: Value = serde_json::from_str(CASES)?;
    if uint(&fixture, "SchemaVersion")? != 1
        || text(&fixture, "Source")? != ID
        || text(&fixture, "SourceSha256")? != SOURCE_SHA256
    {
        return Err(invalid("grammar integration source provenance mismatch"));
    }
    let cases = array(&fixture, "Cases")?;
    let passing = cases
        .iter()
        .map(|case| array(case, "PassingBytes").map(<[Value]>::len))
        .collect::<Result<Vec<_>>>()?
        .iter()
        .sum::<usize>();
    let failing = cases
        .iter()
        .map(|case| array(case, "FailingBytes").map(<[Value]>::len))
        .collect::<Result<Vec<_>>>()?
        .iter()
        .sum::<usize>();
    if cases.len() != 51
        || passing != 207
        || failing != 201
        || array(&fixture, "ParserCases")?.len() != 2
        || array(&fixture, "BuildCases")?.len() != 7
    {
        return Err(invalid("grammar integration literal coverage mismatch"));
    }
    let mut assertions = Assertions::default();
    if !literals(engine, cases, &mut assertions)? {
        return Ok(result(ID,false,"unsupported production API: grammar/schema/clone state; all 11 original groups and 504 conditions required".to_owned()));
    }
    parser_failures(engine, &fixture, &mut assertions)?;
    build_failures(engine, &fixture, &mut assertions)?;
    clone_assertions(engine, &fixture, &mut assertions)?;
    if assertions.count != 504 {
        return Err(invalid(
            "grammar integration omitted original assertion positions",
        ));
    }
    Ok(result(
        ID,
        assertions.failures.is_empty(),
        format!(
            "11 original groups; 51 literal cases; 207 passing + 201 failing inputs; parser/build conditions=10; 43 clone splits x 2; assertions={}; failed={}; {}",
            assertions.count,
            assertions.failures.len(),
            assertions.failures.join("; ")
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_parser_preserves_big_endian_ids_unicode_and_malformed_boundaries() -> Result<()> {
        assert_eq!(
            parse_tokens(&[0xff, 0, 0, 0, 10])?,
            vec![Token {
                id: 10,
                piece: b"<[10]>".to_vec()
            }]
        );
        assert_eq!(
            parse_tokens(&[0xff, 0xff, 0xff, 0xff, 0xff])?,
            vec![Token {
                id: -1,
                piece: b"<[4294967295]>".to_vec()
            }]
        );
        assert!(parse_tokens(&[0xff, 0, 0, 10]).is_err());
        assert_eq!(
            parse_tokens("✅".as_bytes())?,
            vec![Token {
                id: 0,
                piece: "✅".as_bytes().to_vec()
            }]
        );
        assert_eq!(
            parse_tokens(&[0x80])?,
            vec![Token {
                id: 0,
                piece: vec![0xef, 0xbf, 0xbd]
            }]
        );
        assert_eq!(
            parse_tokens(&[0xc0, 0x80])?,
            vec![Token {
                id: 0,
                piece: vec![0]
            }]
        );
        Ok(())
    }
}
