//! All default automated test-chat-template.cpp cases in safe Rust.
//!
//! Inputs and expected byte strings are source data; both legacy and common
//! production renderers are observed through generic native operations.
//! Upstream MIT attribution: tools/prism/upstream/LICENSE.

use serde_json::{Value, json};

use super::engine::Engine;
use super::llama_grammar::{array, exchange, field, uint};
use super::{Result, invalid, result};

const ID: &str = "tests/test-chat-template.cpp";
const SOURCE_SHA: &str = "7a790605222858ca5d5a2de94de27115d693e2c293638c392b7726a44ad3451f";

fn text<'a>(value: &'a Value, name: &str) -> Result<&'a str> {
    field(value, name)?
        .as_str()
        .ok_or_else(|| invalid(format!("chat template {name} must be text")))
}
fn decoded(value: &Value) -> Result<Vec<u8>> {
    let raw = text(value, "BufferHex")?.as_bytes();
    if raw.len() > 262_144 || !raw.len().is_multiple_of(2) {
        return Err(invalid("chat buffer byte extent exceeds bound"));
    }
    raw.as_chunks::<2>()
        .0
        .iter()
        .map(|pair| {
            let digit = |value| match value {
                b'0'..=b'9' => Ok(value - b'0'),
                b'a'..=b'f' => Ok(value - b'a' + 10),
                _ => Err(invalid("chat buffer must be lowercase hex")),
            };
            Ok(16 * digit(pair[0])? + digit(pair[1])?)
        })
        .collect()
}
fn legacy(value: &Value, capacity: usize) -> Result<Option<Vec<u8>>> {
    let count = value["ReturnCode"]
        .as_i64()
        .ok_or_else(|| invalid("legacy renderer ReturnCode must be integer"))?;
    let mut bytes = decoded(value)?;
    if bytes.len() != capacity {
        return Err(invalid("legacy renderer buffer capacity mismatch"));
    }
    if !(0..=131_072).contains(&count) {
        return Ok(None);
    }
    // This preserves the original formatted_chat.resize(res), including zero
    // extension if a renderer reports a size greater than the supplied buffer.
    bytes.resize(usize::try_from(count)?, 0);
    Ok(Some(bytes))
}
fn messages(fixture: &Value, case: &Value) -> Result<Vec<Value>> {
    let mut messages = array(fixture, "Conversation")?.to_vec();
    messages.extend_from_slice(array(case, "ExtraMessages")?);
    Ok(messages)
}

pub(super) fn execute(request: &Value) -> Result<Option<Value>> {
    let Some(operations) = request.get("Operations").and_then(Value::as_array) else {
        return Ok(None);
    };
    if !operations.iter().all(|operation| {
        operation
            .get("Kind")
            .and_then(Value::as_str)
            .is_some_and(|kind| {
                matches!(
                    kind,
                    "common-chat-render" | "legacy-chat-render" | "chat-builtins"
                )
            })
    }) {
        return Ok(None);
    }
    let mut results = Vec::new();
    for operation in operations {
        let kind = text(operation, "Kind")?;
        if kind == "chat-builtins" {
            let capacity = usize::try_from(uint(operation, "Capacity")?)?;
            if capacity > 4096 {
                return Err(invalid("chat builtin capacity exceeds bound"));
            }
            let names = bonsai_prism_runtime::chat::builtin_template_names();
            results.push(json!({"Count":names.len(),"Names":(0..capacity).map(|index|names.get(index).copied()).collect::<Vec<_>>()}));
            continue;
        }
        let render = (|| -> Result<String> {
            use bonsai_prism_runtime::chat::{ChatTemplate, Message};
            let messages = array(operation, "Messages")?
                .iter()
                .map(|message| {
                    Ok(Message {
                        role: text(message, "Role")?.into(),
                        content: text(message, "Content")?.into(),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let add = field(operation, "AddGeneration")?
                .as_bool()
                .ok_or("chat generation flag must be boolean")?;
            if kind == "legacy-chat-render" || operation.get("UseJinja") != Some(&Value::Bool(true))
            {
                return bonsai_prism_runtime::chat::render_legacy(
                    text(operation, "Template")?,
                    &messages,
                    add,
                );
            }
            ChatTemplate::new(
                text(operation, "Template")?,
                text(operation, "Bos")?,
                text(operation, "Eos")?,
            )?
            .render(&messages, add)
        })();
        if kind == "legacy-chat-render" {
            let capacity = usize::try_from(uint(operation, "Capacity")?)?;
            if capacity > 131_072 {
                return Err(invalid("legacy chat buffer exceeds bound"));
            }
            let mut bytes = vec![0_u8; capacity];
            let code = match render {
                Ok(prompt) => {
                    let count = prompt.len().min(capacity);
                    bytes[..count].copy_from_slice(&prompt.as_bytes()[..count]);
                    i64::try_from(prompt.len())?
                }
                Err(_) => -1,
            };
            let mut buffer_hex = String::with_capacity(2 * capacity);
            let digits = b"0123456789abcdef";
            for byte in bytes {
                buffer_hex.push(char::from(digits[usize::from(byte >> 4)]));
                buffer_hex.push(char::from(digits[usize::from(byte & 15)]));
            }
            results.push(json!({"ReturnCode":code,"BufferHex":buffer_hex}));
            continue;
        }
        results.push(match render {
            Ok(prompt) => json!({"Success":true,"Prompt":prompt,"Error":null}),
            Err(error) => json!({"Success":false,"Prompt":null,"Error":error.to_string()}),
        });
    }
    Ok(Some(json!({"SchemaVersion":1,"Results":results})))
}

pub(super) fn run(engine: &Engine) -> Result<Value> {
    let fixture: Value = serde_json::from_str(include_str!("chat-template-cases.json"))?;
    let cases = array(&fixture, "Cases")?;
    if uint(&fixture, "SchemaVersion")? != 1
        || text(&fixture, "Source")? != ID
        || text(&fixture, "SourceSha256")? != SOURCE_SHA
        || cases.len() != 36
        || array(&fixture, "Conversation")?.len() != 6
    {
        return Err(invalid(
            "chat template fixture identity or complete source extent mismatch",
        ));
    }
    let Some(query) = exchange(engine, &[json!({"Kind":"chat-builtins","Capacity":0})])? else {
        return Ok(result(ID,false,"Rust production built-in/common chat template APIs are unavailable; all 36 legacy cases and 35 Jinja cases remain required.".to_owned()));
    };
    let mut checked = 0;
    let mut failures = Vec::new();
    let mut check = |passed: bool, detail: String| {
        checked += 1;
        if !passed {
            failures.push(detail);
        }
    };
    let count = query[0]["Count"]
        .as_i64()
        .ok_or_else(|| invalid("built-in template Count must be integer"))?;
    check(count > 0, "built-in template count positive".to_owned());
    if !(1..=4096).contains(&count) {
        return Err(invalid("built-in template count outside bound"));
    }
    let invalid_request = json!({"Kind":"legacy-chat-render","Template":"INVALID TEMPLATE","Messages":array(&fixture,"Conversation")?,"AddGeneration":true,"Capacity":1024});
    let setup = exchange(
        engine,
        &[
            json!({"Kind":"chat-builtins","Capacity":count}),
            invalid_request,
        ],
    )?
    .ok_or_else(|| invalid("chat template production API disappeared"))?;
    if array(&setup[0], "Names")?.len() != usize::try_from(count)? {
        return Err(invalid("built-in template names extent mismatch"));
    }
    check(
        setup[1]["ReturnCode"].as_i64().is_some_and(|code| code < 0),
        "invalid template rejected".to_owned(),
    );
    let operations=cases.iter().map(|case| Ok(json!({"Kind":"legacy-chat-render","Template":text(case,"Template")?,"Messages":messages(&fixture,case)?,"AddGeneration":true,"Capacity":2048}))).collect::<Result<Vec<_>>>()?;
    let values = exchange(engine, &operations)?
        .ok_or_else(|| invalid("legacy chat production API disappeared"))?;
    for (case, value) in cases.iter().zip(values) {
        check(
            legacy(&value, 2048)?.as_deref() == Some(text(case, "Expected")?.as_bytes()),
            format!("{}: legacy render", text(case, "Name")?),
        );
    }
    let jinja_cases = cases
        .iter()
        .filter(|case| case["SupportedWithJinja"] == true)
        .collect::<Vec<_>>();
    if jinja_cases.len() != 35 {
        return Err(invalid("chat template Jinja eligibility extent mismatch"));
    }
    let operations=jinja_cases.iter().map(|case| Ok(json!({"Kind":"common-chat-render","Template":text(case,"Template")?,"Bos":text(case,"Bos")?,"Eos":text(case,"Eos")?,"UseJinja":true,"AddGeneration":true,"Messages":messages(&fixture,case)?}))).collect::<Result<Vec<_>>>()?;
    let values = exchange(engine, &operations)?
        .ok_or_else(|| invalid("common chat production API disappeared"))?;
    for (case, value) in jinja_cases.iter().zip(values) {
        let alternate = text(case, "ExpectedJinja")?;
        let expected = if alternate.is_empty() {
            text(case, "Expected")?
        } else {
            alternate
        }
        .replace("\r\n", "\n");
        let actual = value["Prompt"]
            .as_str()
            .map(|prompt| prompt.replace("\r\n", "\n"));
        check(
            value["Success"] == true && actual.as_deref() == Some(expected.as_str()),
            format!(
                "{}: common Jinja render: {}",
                text(case, "Name")?,
                value["Error"]
            ),
        );
    }
    if checked != 73 {
        return Err(invalid("chat template original condition count drifted"));
    }
    Ok(result(
        ID,
        failures.is_empty(),
        format!(
            "73 original conditions: built-in count, invalid template, all36 legacy renders and35 common Jinja renders; Windows newline normalization and extra conversations retained; failed={}: {}",
            failures.len(),
            failures.join("; ")
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn production_original_chat_templates() -> Result<()> {
        let engine = Engine::new(
            super::super::engine::Target::Rust,
            None,
            None,
            None,
            None,
            None,
        )?;
        let report = run(&engine)?;
        assert_eq!(report["Status"], "pass", "{report:?}");
        Ok(())
    }
    #[test]
    fn production_common_chat_templates() -> Result<()> {
        let fixture: Value = serde_json::from_str(include_str!("chat-template-cases.json"))?;
        let mut failures = Vec::new();
        for case in array(&fixture, "Cases")?
            .iter()
            .filter(|case| case["SupportedWithJinja"] == true)
        {
            let operation = json!({"Kind":"common-chat-render","Template":text(case,"Template")?,"Bos":text(case,"Bos")?,"Eos":text(case,"Eos")?,"UseJinja":true,"AddGeneration":true,"Messages":messages(&fixture,case)?});
            let result =
                execute(&json!({"Operations":[operation]}))?.ok_or("chat result missing")?;
            let actual = &result["Results"][0];
            let alternate = text(case, "ExpectedJinja")?;
            let expected = if alternate.is_empty() {
                text(case, "Expected")?
            } else {
                alternate
            }
            .replace("\r\n", "\n");
            if actual["Prompt"]
                .as_str()
                .map(|value| value.replace("\r\n", "\n"))
                .as_deref()
                != Some(&expected)
            {
                failures.push(format!(
                    "{}: {actual:?} expected {expected:?}",
                    text(case, "Name")?
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "{} failures:\n{}",
            failures.len(),
            failures.join("\n")
        );
        Ok(())
    }

    #[test]
    fn legacy_buffer_capacity_and_resize_preserve_original_byte_semantics() -> Result<()> {
        let response = json!({"ReturnCode":2,"BufferHex":"ff010203"});
        assert_eq!(legacy(&response, 4)?, Some(vec![0xff, 1]));
        let response = json!({"ReturnCode":6,"BufferHex":"ff010203"});
        assert_eq!(legacy(&response, 4)?, Some(vec![0xff, 1, 2, 3, 0, 0]));
        assert!(legacy(&response, 3).is_err());
        assert!(decoded(&json!({"BufferHex":"0g"})).is_err());
        assert_eq!(
            legacy(&json!({"ReturnCode":-1,"BufferHex":"0000"}), 2)?,
            None
        );
        Ok(())
    }
}
