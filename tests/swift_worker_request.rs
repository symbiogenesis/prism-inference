#![forbid(unsafe_code)]
#![cfg(feature = "cli")]
//! Direct tests of the production Swift worker's private strict request schema.

#[path = "../src/swift_worker_request.rs"]
mod request;

use bonsai_prism_runtime::{Result, swift_chat::MAX_MESSAGES};
use request::{MAX_FRAME_BYTES, Request};
use serde_json::{Value, json};

fn count(messages: &str) -> String {
    format!("{{\"op\":\"count\",\"id\":7,\"messages\":{messages},\"max_tokens\":16}}\n")
}

fn snapshot(value: Request) -> Value {
    match value {
        Request::Count {
            id,
            messages,
            max_tokens,
        } => json!({"op":"count","id":id,"messages":messages,"max_tokens":max_tokens}),
        Request::Complete {
            id,
            messages,
            max_tokens,
            timeout_seconds,
        } => {
            json!({"op":"complete","id":id,"messages":messages,"max_tokens":max_tokens,"timeout_seconds":timeout_seconds})
        }
        Request::Reset { id } => json!({"op":"reset","id":id}),
    }
}

#[test]
fn preserves_valid_text_reasoning_and_field_order_before_json_conversion() -> Result<()> {
    let messages = json!([
        {"role":"system","content":"Keep Unicode: café 🌱.\n"},
        {"role":"user","content":"first"},
        {"role":"assistant","content":" answer ","reasoning_content":" thought\r\n "},
        {"role":"user","content":" next "}
    ]);
    let encoded = format!(
        "{{\"op\":\"complete\",\"messages\":{},\"id\":8,\"timeout_seconds\":600,\"max_tokens\":16}}\r\n",
        serde_json::to_string(&messages)?
    );
    assert_eq!(
        snapshot(request::parse(encoded.as_bytes())?),
        json!({"op":"complete","messages":messages,"id":8,"timeout_seconds":600,"max_tokens":16})
    );
    let reversed = count(r#"[{"content":"q","role":"user"}]"#);
    assert_eq!(
        snapshot(request::parse(reversed.as_bytes())?)["messages"],
        json!([{"role":"user","content":"q"}])
    );
    assert_eq!(
        snapshot(request::parse(b"{\"op\":\"reset\",\"id\":9}\n")?),
        json!({"op":"reset","id":9})
    );
    Ok(())
}

#[test]
fn rejects_duplicate_nested_message_fields_even_when_values_are_identical() {
    for messages in [
        r#"[{"role":"user","role":"user","content":"q"}]"#,
        r#"[{"role":"user","content":"q","content":"q"}]"#,
        r#"[{"role":"assistant","content":"a","reasoning_content":"","reasoning_content":""},{"role":"user","content":"q"}]"#,
        r#"[{"role":"assistant","content":"a","reasoning_content":null,"reasoning_content":"r"}]"#,
    ] {
        assert!(
            request::parse(count(messages).as_bytes()).is_err(),
            "accepted duplicate: {messages}"
        );
    }
}

#[test]
fn rejects_positional_messages_unknown_fields_and_nontext_or_missing_values() {
    for messages in [
        r#"[["user","q"]]"#,
        r#"[["assistant","a","r"],["user","q"]]"#,
        r#"[{"role":"user","content":"q","tools":[]}]"#,
        r#"[{"role":"user","content":[{"type":"text","text":"q"}]}]"#,
        r#"[{"role":"user","content":{"text":"q"}}]"#,
        r#"[{"role":null,"content":"q"}]"#,
        r#"[{"role":"user","content":null}]"#,
        r#"[{"role":"assistant","content":"a","reasoning_content":null}]"#,
        r#"[{"role":"user"}]"#,
        r#"[{"content":"q"}]"#,
        "[null]",
        "{}",
        "[]",
    ] {
        assert!(
            request::parse(count(messages).as_bytes()).is_err(),
            "accepted invalid message shape: {messages}"
        );
    }
}

#[test]
fn bounds_message_count_and_exact_complete_frame_without_losing_top_level_strictness() -> Result<()>
{
    let messages = vec![json!({"role":"user","content":"q"}); MAX_MESSAGES];
    assert!(request::parse(count(&serde_json::to_string(&messages)?).as_bytes()).is_ok());
    let too_many = vec![json!({"role":"user","content":"q"}); MAX_MESSAGES + 1];
    assert!(request::parse(count(&serde_json::to_string(&too_many)?).as_bytes()).is_err());
    let mut exact = count(r#"[{"role":"user","content":"q"}]"#).into_bytes();
    exact.pop();
    exact.resize(MAX_FRAME_BYTES - 1, b' ');
    exact.push(b'\n');
    assert!(request::parse(&exact).is_ok());
    exact.push(b'\n');
    assert!(request::parse(&exact).is_err());
    for frame in [
        b"{\"op\":\"reset\",\"id\":1,\"id\":1}\n".as_slice(),
        b"{\"op\":\"reset\",\"id\":1,\"extra\":0}\n",
        b"{\"op\":\"reset\",\"id\":1}",
        b"[\"reset\",1]\n",
    ] {
        assert!(request::parse(frame).is_err());
    }
    Ok(())
}
