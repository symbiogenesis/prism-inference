//! Safe Rust port of the complete manual GBNF validator's control flow.
//!
//! The source requires two file arguments and defines no fixed tests. Default
//! regression checks reuse all119 valid UTF-8 inputs from the twelve ordinary
//! text grammars in the independently ported upstream integration corpus. The
//! two token-ID grammars are outside this character-only diagnostic's domain.
//! These are additional regressions, never described as original assertions.
//! Upstream MIT attribution: tools/prism/upstream/LICENSE.

use std::fmt::Write as _;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use serde_json::{Value, json};

use super::engine::Engine;
use super::llama_grammar::{array, exact_fields, exchange, field};
use super::{Result, invalid, result};

const ID: &str = "tests/test-gbnf-validator.cpp";
const CORPUS: &str = include_str!("grammar-integration-cases.json");
const MAX_FILE: u64 = 1024 * 1024;

#[derive(Debug, PartialEq)]
struct Validation {
    valid: bool,
    error_position: usize,
    error_message: String,
}

fn text<'a>(value: &'a Value, name: &str) -> Result<&'a str> {
    field(value, name)?
        .as_str()
        .ok_or_else(|| invalid(format!("validator {name} must be text")))
}

fn hex(bytes: &[u8]) -> Result<String> {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(output, "{byte:02x}")?;
    }
    Ok(output)
}

fn sizes(value: &Value) -> Result<Vec<u64>> {
    array(value, "StackSizes")?
        .iter()
        .map(|size| {
            size.as_u64()
                .ok_or_else(|| invalid("grammar stack size must be unsigned"))
        })
        .collect()
}

fn inspect(points: &[u32], initial: &Value, events: &[Value]) -> Result<Validation> {
    exact_fields(initial, &["StackSizes"])?;
    if points.len() != events.len() {
        return Err(invalid("validator observation extent mismatch"));
    }
    let mut stacks = sizes(initial)?;
    for (position, (&point, event)) in points.iter().zip(events).enumerate() {
        exact_fields(event, &["ExceptionKind", "Message", "StackSizes"])?;
        if !field(event, "ExceptionKind")?.is_null() {
            return Err(invalid(format!(
                "production grammar acceptance threw: {}",
                text(event, "Message")?
            )));
        }
        stacks = sizes(event)?;
        if stacks.is_empty() {
            let character = char::from_u32(point)
                .ok_or_else(|| invalid("validator error character is not a Unicode scalar"))?;
            return Ok(Validation {
                valid: false,
                error_position: position,
                error_message: format!("Unexpected character '{character}'"),
            });
        }
    }
    let valid = stacks.contains(&0);
    Ok(Validation {
        valid,
        error_position: points.len(),
        error_message: if valid {
            String::new()
        } else {
            "Unexpected end of input".into()
        },
    })
}

fn decode(engine: &Engine, inputs: &[Vec<u8>]) -> Result<Option<Vec<Vec<u32>>>> {
    let encoded = inputs
        .iter()
        .map(|input| hex(input))
        .collect::<Result<Vec<_>>>()?;
    let operation = json!({"Kind":"unicode-decode","InputsHex":encoded});
    let Some(values) = exchange(engine, &[operation])? else {
        return Ok(None);
    };
    exact_fields(&values[0], &["Values", "Errors"])?;
    let points = array(&values[0], "Values")?;
    let errors = array(&values[0], "Errors")?;
    if points.len() != inputs.len() || errors.len() != inputs.len() {
        return Err(invalid("validator Unicode framing mismatch"));
    }
    points
        .iter()
        .zip(errors)
        .map(|(points, error)| {
            if !error.is_null() {
                return Err(invalid(format!(
                    "production Unicode decoding threw: {error}"
                )));
            }
            points
                .as_array()
                .ok_or_else(|| invalid("Unicode points must be an array"))?
                .iter()
                .map(|point| {
                    Ok(u32::try_from(point.as_u64().ok_or_else(|| {
                        invalid("Unicode point must be unsigned")
                    })?)?)
                })
                .collect()
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

fn validate(
    engine: &Engine,
    grammar: &str,
    points: &[Vec<u32>],
) -> Result<Option<Vec<Validation>>> {
    let mut actions = Vec::new();
    let mut ranges = Vec::new();
    for input in points {
        let start = actions.len();
        actions.push(json!({"Kind":"init-text","Grammar":0,"GrammarText":grammar,"Root":"root"}));
        actions.push(json!({"Kind":"state","Grammar":0}));
        actions.extend(
            input
                .iter()
                .map(|point| json!({"Kind":"accept-codepoint","Grammar":0,"CodePoint":point})),
        );
        actions.push(json!({"Kind":"free","Grammar":0}));
        ranges.push(start..actions.len());
    }
    let operation = json!({"Kind":"grammar-trace","Actions":actions});
    if actions.len() > 16384 {
        return Err(invalid(
            "validator diagnostic exceeds the bounded16384-action transport",
        ));
    }
    let Some(values) = exchange(engine, &[operation])? else {
        return Ok(None);
    };
    exact_fields(&values[0], &["Trace"])?;
    let trace = array(&values[0], "Trace")?;
    if trace.len() != actions.len() {
        return Err(invalid("validator grammar trace extent mismatch"));
    }
    ranges
        .iter()
        .zip(points)
        .map(|(range, input)| {
            exact_fields(&trace[range.start], &["Created"])?;
            if trace[range.start]["Created"] != true {
                return Err(invalid(
                    "production validator grammar initialization failed",
                ));
            }
            inspect(
                input,
                &trace[range.start + 1],
                &trace[range.start + 2..range.end - 1],
            )
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

fn reference_inputs(case: &Value) -> Result<(Vec<Vec<u8>>, Vec<bool>)> {
    let mut inputs = Vec::new();
    let mut expected = Vec::new();
    for (name, valid) in [("PassingBytes", true), ("FailingBytes", false)] {
        for bytes in array(case, name)? {
            let bytes = bytes
                .as_array()
                .ok_or_else(|| invalid("validator reference must be bytes"))?
                .iter()
                .map(|byte| {
                    Ok(u8::try_from(byte.as_u64().ok_or_else(|| {
                        invalid("validator input byte must be unsigned")
                    })?)?)
                })
                .collect::<Result<Vec<_>>>()?;
            if std::str::from_utf8(&bytes).is_ok() {
                inputs.push(bytes);
                expected.push(valid);
            }
        }
    }
    Ok((inputs, expected))
}

pub(super) fn run(engine: &Engine) -> Result<Value> {
    let corpus: Value = serde_json::from_str(CORPUS)?;
    if text(&corpus, "SourceSha256")?
        != "6acd7735c8f6db1d3869bd2de5ea7a585accc75e3f8627eb163be11bc123716a"
    {
        return Err(invalid(
            "validator independent regression corpus identity mismatch",
        ));
    }
    let mut groups = 0;
    let mut checks = 0;
    let mut failures = Vec::new();
    for case in array(&corpus, "Cases")? {
        if text(case, "Kind")? != "grammar" || text(case, "Definition")?.contains("<[") {
            continue;
        }
        groups += 1;
        let (inputs, expected) = reference_inputs(case)?;
        let Some(points) = decode(engine, &inputs)? else {
            return Ok(result(
                ID,
                false,
                "production Unicode/grammar APIs unavailable".into(),
            ));
        };
        for (input, points) in inputs.iter().zip(&points) {
            let reference: Vec<_> = std::str::from_utf8(input)?.chars().map(u32::from).collect();
            if *points != reference {
                return Err(invalid(
                    "production Unicode decoding differs on valid UTF-8 regression input",
                ));
            }
        }
        let Some(observations) = validate(engine, text(case, "Definition")?, &points)? else {
            return Ok(result(
                ID,
                false,
                "production grammar API unavailable".into(),
            ));
        };
        for (index, (observation, expected)) in observations.iter().zip(expected).enumerate() {
            checks += 1;
            if observation.valid != expected {
                failures.push(format!(
                    "{} input{index}: {observation:?}",
                    text(case, "Name")?
                ));
            }
        }
    }
    Ok(result(ID,groups==12&&checks==119&&failures.is_empty(),json!({
        "SourceBehavioralAssertions":0,"SourceMode":"manual two-file validator; complete algorithm and output formatting ported",
        "SourceSha256":"21eb0bac296c0e0dd497cc438ab8f10ba9cad66488c00288321d939afbdf310f",
        "AdditionalRegressionCorpus":"all validUTF8 input vectors from all12 ordinary-text grammar-integration grammars; token-ID grammars excluded from this character-only diagnostic",
        "RegressionGrammars":groups,"RegressionChecks":checks,"Failures":failures,
        "ExitSemantics":"original returns0 for both valid and invalid input; infrastructure/init failures remain errors",
    }).to_string()))
}

fn windows_text(bytes: &[u8]) -> Vec<u8> {
    let bytes = bytes.split(|byte| *byte == 0x1a).next().unwrap_or_default();
    let mut output = Vec::new();
    let mut input = bytes.iter().copied().peekable();
    while let Some(byte) = input.next() {
        if byte == b'\r' && input.peek() == Some(&b'\n') {
            continue;
        }
        output.push(byte);
    }
    output
}

fn read_file(path: &str) -> Result<Vec<u8>> {
    if !Path::new(path).is_absolute() {
        return Err(invalid("validator input paths must be absolute"));
    }
    let mut bytes = Vec::new();
    File::open(path)?
        .take(MAX_FILE + 1)
        .read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len())? > MAX_FILE {
        return Err(invalid("validator input file exceeds1MiB bound"));
    }
    Ok(windows_text(&bytes))
}

fn output(input: &[u8], validation: &Validation) -> Vec<u8> {
    if validation.valid {
        return b"Input string is valid according to the grammar.\n".to_vec();
    }
    let position = validation.error_position;
    let mut bytes = format!("Input string is invalid according to the grammar.\nError: {} at position {position}\n\nInput string:\n",validation.error_message).into_bytes();
    // The original display mistakenly uses its codepoint index as a byte
    // offset. Preserve that observable diagnostic, including split UTF-8 bytes.
    bytes.extend_from_slice(&input[..position.min(input.len())]);
    if position < input.len() {
        bytes.extend_from_slice(b"\x1b[1;31m");
        bytes.push(input[position]);
        if position + 1 < input.len() {
            bytes.extend_from_slice(b"\x1b[0;31m");
            bytes.extend_from_slice(&input[position + 1..]);
        }
        bytes.extend_from_slice(b"\x1b[0m\n");
    }
    bytes
}

pub(super) fn diagnostic(engine: &Engine, config: &Value) -> Result<Value> {
    exact_fields(config, &["Kind", "GrammarPath", "InputPath"])?;
    let grammar = String::from_utf8(read_file(text(config, "GrammarPath")?)?)?;
    let input = read_file(text(config, "InputPath")?)?;
    let points = decode(engine, std::slice::from_ref(&input))?
        .ok_or_else(|| invalid("production Unicode API unavailable"))?;
    let values = validate(engine, &grammar, &points)?
        .ok_or_else(|| invalid("production grammar API unavailable"))?;
    Ok(
        json!({"Source":ID,"Diagnostic":"two-file-gbnf-validator","Valid":values[0].valid,
        "ErrorPosition":values[0].error_position,"ErrorMessage":values[0].error_message,
        "StdoutHex":hex(&output(&input,&values[0]))?,"OriginalExitCode":0}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codepoint_position_and_legacy_byte_highlight_are_distinct() -> Result<()> {
        let failure = inspect(
            &[0xe9, 0x78],
            &json!({"StackSizes":[2]}),
            &[
                json!({"ExceptionKind":null,"Message":"","StackSizes":[1]}),
                json!({"ExceptionKind":null,"Message":"","StackSizes":[]}),
            ],
        )?;
        assert_eq!(
            failure,
            Validation {
                valid: false,
                error_position: 1,
                error_message: "Unexpected character 'x'".into()
            }
        );
        let rendered = output("éx".as_bytes(), &failure);
        assert!(
            rendered
                .windows(9)
                .any(|bytes| bytes == b"\xc3\x1b[1;31m\xa9")
        );
        assert_eq!(windows_text(b"a\r\nb\rc\x1aignored"), b"a\nb\rc");
        assert!(inspect(&[], &json!({"StackSizes":[0]}), &[])?.valid);
        Ok(())
    }
}
