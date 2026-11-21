//! Independent concurrent logger observations for the original smoke contract.
//!
//! Producers send records/settings to a bounded channel; a separate consumer
//! owns formatting, flushing and destruction. The production Rust logger uses
//! shared writer locking instead. No native logger or production Rust API is
//! called, and no output-content or native execution claim is made.

use std::io::Write;
use std::sync::mpsc::{self, Receiver};

use serde::Deserialize;
use serde_json::{Value, json};

use super::{Result, invalid};

pub(super) const ID: &str = "tests/test-log.cpp";

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
    workers: Vec<Vec<Action>>,
    finalize: Vec<String>,
}

#[derive(Deserialize)]
#[serde(tag = "Kind", deny_unknown_fields)]
enum Action {
    #[serde(rename = "emit")]
    Emit {
        #[serde(rename = "Level")]
        level: Level,
        #[serde(rename = "Text")]
        text: String,
    },
    #[serde(rename = "timestamps")]
    Timestamps {
        #[serde(rename = "Value")]
        enabled: bool,
    },
    #[serde(rename = "prefix")]
    Prefix {
        #[serde(rename = "Value")]
        enabled: bool,
    },
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum Level {
    Info,
    Warning,
    Error,
    Debug,
}

enum Message<'a> {
    Action(&'a Action),
    Flush,
    Free,
}

fn validate(operation: &Operation) -> Result<()> {
    if operation.kind != "concurrent-log"
        || !(1..=32).contains(&operation.workers.len())
        || operation.finalize.len() > 8
        || operation.finalize.last().map(String::as_str) != Some("free")
        || !operation.finalize[..operation.finalize.len().saturating_sub(1)]
            .iter()
            .all(|action| action == "flush")
    {
        return Err(invalid(
            "invalid independent logger schedule or finalization",
        ));
    }
    let mut total = 0;
    for worker in &operation.workers {
        total += worker.len();
        if worker.len() > 16_384 || total > 65_536 {
            return Err(invalid("independent logger action bound exceeded"));
        }
        for action in worker {
            if let Action::Emit { text, .. } = action
                && (text.len() > 1024 || text.contains('\0'))
            {
                return Err(invalid("independent logger text bound or NUL violation"));
            }
        }
    }
    Ok(())
}

fn consume<W: Write>(receiver: Receiver<Message<'_>>, mut writer: W) -> std::io::Result<()> {
    let mut prefix = false;
    let mut timestamps = false;
    // Keep draining after I/O failure so a bounded-channel producer cannot be
    // stranded. The failure remains sticky through flush, free and joining.
    let mut failure = None;
    let mut freed = false;
    for message in receiver {
        let result = match message {
            Message::Action(Action::Timestamps { enabled }) => {
                timestamps = *enabled;
                Ok(())
            }
            Message::Action(Action::Prefix { enabled }) => {
                prefix = *enabled;
                Ok(())
            }
            Message::Action(Action::Emit { level, text }) => (|| {
                if timestamps {
                    // Content is intentionally unspecified by this source.
                    writer.write_all(b"[timestamp] ")?;
                }
                if prefix {
                    writer.write_all(match level {
                        Level::Info => b"info: ",
                        Level::Warning => b"warning: ",
                        Level::Error => b"error: ",
                        Level::Debug => b"debug: ",
                    })?;
                }
                writer.write_all(text.as_bytes())
            })(),
            Message::Flush => writer.flush(),
            Message::Free => {
                freed = true;
                let result = writer.flush();
                if let Err(error) = result {
                    failure.get_or_insert(error);
                }
                break;
            }
        };
        if let Err(error) = result {
            failure.get_or_insert(error);
        }
    }
    if let Some(error) = failure {
        return Err(error);
    }
    if !freed {
        return Err(std::io::Error::other("independent logger was not freed"));
    }
    Ok(())
}

fn run<W: Write + Send>(operation: &Operation, writer: W) -> Result<Value> {
    validate(operation)?;
    let completed = std::thread::scope(|scope| -> Result<Vec<usize>> {
        let (sender, receiver) = mpsc::sync_channel(64);
        let consumer = scope.spawn(move || consume(receiver, writer));
        let mut workers = Vec::new();
        for actions in &operation.workers {
            let sender = sender.clone();
            workers.push(scope.spawn(move || -> Result<usize> {
                let mut completed = 0;
                for action in actions {
                    sender
                        .send(Message::Action(action))
                        .map_err(|_| invalid("independent logger consumer exited"))?;
                    completed += 1;
                }
                Ok(completed)
            }));
        }
        let mut completed = Vec::new();
        let mut failure = None;
        for worker in workers {
            match worker.join() {
                Ok(Ok(count)) => completed.push(count),
                Ok(Err(error)) => {
                    failure.get_or_insert(error);
                }
                Err(_) => {
                    failure.get_or_insert_with(|| invalid("independent logger producer panicked"));
                }
            }
        }
        for action in &operation.finalize {
            if sender
                .send(if action == "flush" {
                    Message::Flush
                } else {
                    Message::Free
                })
                .is_err()
            {
                failure.get_or_insert_with(|| invalid("independent logger finalization failed"));
                break;
            }
        }
        drop(sender);
        match consumer.join() {
            Ok(Ok(())) => (),
            Ok(Err(error)) => {
                failure.get_or_insert_with(|| error.into());
            }
            Err(_) => {
                failure.get_or_insert_with(|| invalid("independent logger consumer panicked"));
            }
        }
        failure.map_or(Ok(completed), Err)
    })?;
    Ok(json!({"Completed": completed}))
}

pub(super) fn handles(value: &Value) -> bool {
    value["Operations"].as_array().is_some_and(|operations| {
        !operations.is_empty()
            && operations
                .iter()
                .all(|operation| operation["Kind"] == "concurrent-log")
    })
}

pub(super) fn execute(value: &Value) -> Result<Value> {
    let request: Request = serde_json::from_value(value.clone())?;
    // Each request owns one logger and explicitly frees it. A second operation
    // would follow singleton destruction and was rejected by the native bridge.
    if request.schema_version != 1 || request.operations.len() != 1 {
        return Err(invalid(
            "independent logger requires schema 1 and one terminal operation",
        ));
    }
    let result = run(&request.operations[0], std::io::sink())?;
    Ok(json!({"SchemaVersion": 1, "Results": [result]}))
}

pub(super) fn identity() -> Value {
    json!({"Kind": "independent-rust-concurrent-logger", "SupportedSourceIds": [ID],
        "UpstreamCommit": super::UPSTREAM,
        "AssertionSource": {"Path": ID, "Sha256": "f668c6d6987b0e8888a3823c0425971356c16ca77b44819236fb00c30e4b60c1"},
        "Architecture": "bounded-channel-producers-and-owned-consumer-v1",
        "NativeReferenceReplay": false, "ProductionRuntimeCalled": false,
        "OutputContentParityClaimed": false, "CompleteSourceGate": false,
        "NativeRetirementQualified": false})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, Target};

    fn valid() -> Value {
        json!({"SchemaVersion": 1, "Operations": [{"Kind": "concurrent-log",
            "Workers": [[{"Kind": "emit", "Level": "info", "Text": "hello\n"},
                {"Kind": "timestamps", "Value": true}, {"Kind": "prefix", "Value": true}]],
            "Finalize": ["flush", "free"]}]})
    }

    #[test]
    fn independent_logger_preserves_source_schedule_and_reference_arm() -> Result<()> {
        for target in [Target::RustReference, Target::Cpp, Target::Rust] {
            let engine = Engine::new(target, None, None, None, None)?;
            let report = crate::logging::run(&engine)?;
            assert_eq!(report["Id"], ID);
            assert_eq!(report["Status"], "pass", "{report}");
        }
        assert_eq!(execute(&valid())?["Results"][0]["Completed"], json!([3]));
        assert_eq!(identity()["NativeReferenceReplay"], false);
        Ok(())
    }

    #[test]
    fn independent_logger_rejects_invalid_actions_and_teardown_before_execution() -> Result<()> {
        let request = valid();
        for finalization in [
            json!([]),
            json!(["flush"]),
            json!(["free", "flush"]),
            json!(["free", "free"]),
            json!(["unknown", "free"]),
            json!([true, "free"]),
        ] {
            let mut changed = request.clone();
            changed["Operations"][0]["Finalize"] = finalization;
            assert!(execute(&changed).is_err());
        }
        for action in [
            json!({"Kind": "emit", "Level": "fatal", "Text": "x"}),
            json!({"Kind": "emit", "Level": "info", "Text": "x\0"}),
            json!({"Kind": "emit", "Level": "info", "Text": "x".repeat(1025)}),
            json!({"Kind": "timestamps", "Value": 1}),
            json!({"Kind": "prefix", "Value": false, "Extra": 0}),
            json!({"Kind": "unknown"}),
        ] {
            let mut changed = request.clone();
            changed["Operations"][0]["Workers"][0][0] = action;
            assert!(execute(&changed).is_err());
        }
        for workers in [
            json!([]),
            json!(vec![json!([]); 33]),
            json!([vec![json!({"Kind":"prefix","Value":false}); 16_385]]),
        ] {
            let mut changed = request.clone();
            changed["Operations"][0]["Workers"] = workers;
            assert!(execute(&changed).is_err());
        }
        let mut changed = request;
        let duplicate = changed["Operations"][0].clone();
        changed["Operations"]
            .as_array_mut()
            .ok_or_else(|| invalid("test operation array missing"))?
            .push(duplicate);
        assert!(execute(&changed).is_err());
        Ok(())
    }

    struct FailedWriter;

    impl Write for FailedWriter {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("injected write failure"))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Err(std::io::Error::other("injected flush failure"))
        }
    }

    #[test]
    fn independent_logger_drains_bounded_channel_and_joins_after_writer_failure() -> Result<()> {
        let mut request: Request = serde_json::from_value(valid())?;
        request.operations[0].workers = (0..8)
            .map(|_| {
                (0..1000)
                    .map(|_| Action::Emit {
                        level: Level::Info,
                        text: "bounded\n".to_owned(),
                    })
                    .collect()
            })
            .collect();
        assert!(run(&request.operations[0], FailedWriter).is_err());
        assert_eq!(
            run(&request.operations[0], std::io::sink())?["Completed"],
            json!(vec![1000; 8])
        );
        Ok(())
    }
}
