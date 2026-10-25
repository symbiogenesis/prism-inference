//! Serialization adapters to public production runtime operations.

use bonsai_prism_runtime::{logging, media, speculative, unicode};
use serde_json::{Value, json};

use super::llama_grammar::{array, exact_fields, field};
use super::{Result, invalid};

fn number(operation: &Value, name: &str) -> Result<u32> {
    Ok(u32::try_from(
        field(operation, name)?
            .as_u64()
            .ok_or_else(|| invalid("expected unsigned integer"))?,
    )?)
}

fn signed(operation: &Value, name: &str) -> Result<i32> {
    Ok(i32::try_from(
        field(operation, name)?
            .as_i64()
            .ok_or_else(|| invalid("expected signed integer"))?,
    )?)
}

fn boolean(operation: &Value, name: &str) -> Result<bool> {
    field(operation, name)?
        .as_bool()
        .ok_or_else(|| invalid("expected boolean"))
}

fn text<'a>(operation: &'a Value, name: &str) -> Result<&'a str> {
    field(operation, name)?
        .as_str()
        .ok_or_else(|| invalid("expected text"))
}

fn part(value: &Value) -> Result<media::Part> {
    match text(value, "Kind")? {
        "text" => {
            exact_fields(value, &["Kind", "Text"])?;
            Ok(media::Part::Text(text(value, "Text")?.to_owned()))
        }
        "bitmap" => {
            exact_fields(value, &["Kind", "Width", "Height", "Mergeable"])?;
            Ok(media::Part::Image {
                width: number(value, "Width")?,
                height: number(value, "Height")?,
                mergeable: boolean(value, "Mergeable")?,
            })
        }
        "audio" => {
            exact_fields(value, &["Kind", "Samples", "Mergeable"])?;
            boolean(value, "Mergeable")?;
            Ok(media::Part::Audio {
                samples: number(value, "Samples")?,
            })
        }
        _ => Err(invalid("unknown multimodal part")),
    }
}

fn logging(operation: &Value) -> Result<Value> {
    exact_fields(operation, &["Kind", "Workers", "Finalize"])?;
    let workers = array(operation, "Workers")?;
    let finalize = array(operation, "Finalize")?;
    if workers.is_empty() || workers.len() > 32 || finalize.len() > 8 {
        return Err(invalid("logger schedule bound exceeded"));
    }
    let mut count = 0;
    for worker in workers {
        let actions = worker
            .as_array()
            .ok_or_else(|| invalid("logger actions must be an array"))?;
        count += actions.len();
        if actions.len() > 16384 || count > 65536 {
            return Err(invalid("logger action bound exceeded"));
        }
        for action in actions {
            match text(action, "Kind")? {
                "emit" => {
                    exact_fields(action, &["Kind", "Level", "Text"])?;
                    level(text(action, "Level")?)?;
                    let message = text(action, "Text")?;
                    if message.len() > 1024 || message.contains('\0') {
                        return Err(invalid("invalid log text"));
                    }
                }
                "timestamps" | "prefix" => {
                    exact_fields(action, &["Kind", "Value"])?;
                    boolean(action, "Value")?;
                }
                _ => return Err(invalid("unknown log action")),
            }
        }
    }
    let mut freed = false;
    for action in finalize {
        if freed {
            return Err(invalid("logger action after close"));
        }
        match action.as_str() {
            Some("free") => freed = true,
            Some("flush") => (),
            _ => return Err(invalid("unknown logger finalization")),
        }
    }
    if !freed {
        return Err(invalid("logger must be explicitly closed"));
    }
    let log = logging::Logger::new(std::io::sink());
    let completed = std::thread::scope(|scope| -> Result<Vec<usize>> {
        let mut threads = Vec::new();
        for worker in workers {
            let actions = worker.as_array().ok_or_else(|| invalid("invalid worker"))?;
            let log = &log;
            threads.push(scope.spawn(move || -> Result<usize> {
                for action in actions {
                    match text(action, "Kind")? {
                        "emit" => {
                            log.emit(level(text(action, "Level")?)?, text(action, "Text")?)?;
                        }
                        "timestamps" => log.set_timestamps(boolean(action, "Value")?)?,
                        "prefix" => log.set_prefix(boolean(action, "Value")?)?,
                        _ => return Err(invalid("invalid validated logger action")),
                    }
                }
                Ok(actions.len())
            }));
        }
        threads
            .into_iter()
            .map(|thread| {
                thread
                    .join()
                    .map_err(|_| invalid("logger worker panicked"))?
            })
            .collect()
    })?;
    for action in finalize {
        if action == "flush" {
            log.flush()?;
        } else {
            log.close()?;
        }
    }
    Ok(json!({"Completed": completed}))
}

fn level(name: &str) -> Result<logging::Level> {
    match name {
        "info" => Ok(logging::Level::Info),
        "warning" => Ok(logging::Level::Warning),
        "error" => Ok(logging::Level::Error),
        "debug" => Ok(logging::Level::Debug),
        _ => Err(invalid("unknown logger severity")),
    }
}

pub(super) fn execute(request: &Value) -> Result<Option<Value>> {
    let Some(operations) = request["Operations"].as_array() else {
        return Ok(None);
    };
    if !operations.iter().all(|op| {
        matches!(
            op["Kind"].as_str(),
            Some(
                "mtp-first-decode-fits"
                    | "unicode-regex-split"
                    | "mtmd-image-should-tile"
                    | "mtmd-group-bitmaps"
                    | "concurrent-log"
                    | "grammar-parser-parse"
            )
        )
    }) {
        return Ok(None);
    }
    exact_fields(request, &["SchemaVersion", "Operations"])?;
    if request["SchemaVersion"] != 1 || operations.is_empty() || operations.len() > 128 {
        return Err(invalid("invalid runtime request schema or operation count"));
    }
    let results = operations.iter().map(|operation| -> Result<Value> {
        match text(operation, "Kind")? {
            "mtp-first-decode-fits" => {
                exact_fields(operation, &["Kind", "Batch", "Catchup", "Sequences"])?;
                Ok(json!({"Fits": speculative::mtp_first_decode_fits(signed(operation, "Batch")?, signed(operation, "Catchup")?, signed(operation, "Sequences")?)}))
            }
            "unicode-regex-split" => {
                exact_fields(operation, &["Kind", "Text", "Expressions", "ByteEncoding"])?;
                let expressions = array(operation, "Expressions")?.iter().map(|value| value.as_str().ok_or_else(|| invalid("regex must be text"))).collect::<Result<Vec<_>>>()?;
                Ok(json!({"Pieces": unicode::regex_split(text(operation, "Text")?, &expressions, boolean(operation, "ByteEncoding")?)?}))
            }
            "mtmd-image-should-tile" => {
                exact_fields(operation, &["Kind", "PatchSize", "Merge", "MinImageTokens", "MaxImageTokens", "Width", "Height"])?;
                Ok(json!({"ShouldTile": media::should_tile(number(operation, "PatchSize")?, number(operation, "Merge")?, number(operation, "MinImageTokens")?, number(operation, "MaxImageTokens")?, number(operation, "Width")?, number(operation, "Height")?)?}))
            }
            "mtmd-group-bitmaps" => {
                exact_fields(operation, &["Kind", "Merge", "Parts"])?;
                let mut parts = array(operation, "Parts")?.iter().map(part).collect::<Result<Vec<_>>>()?;
                let groups = media::group_bitmaps(&mut parts, number(operation, "Merge")?)?;
                Ok(json!({"GroupSizes": groups.iter().map(Vec::len).collect::<Vec<_>>(), "RemainingBitmapParts": parts.iter().filter(|part| !matches!(part, media::Part::Text(_))).count()}))
            }
            "concurrent-log" => logging(operation),
            "grammar-parser-parse" => {
                exact_fields(operation, &["Kind", "Grammar"])?;
                match bonsai_prism_runtime::grammar::Grammar::parse(text(operation, "Grammar")?) {
                    Ok(grammar) => Ok(json!({"Parsed": true,
                        "SymbolIds": grammar.symbols.iter().map(|(name,id)| json!({"Name": name, "Id": id})).collect::<Vec<_>>(),
                        "Rules": grammar.rules.iter().map(|rule| rule.iter().map(|element| json!({"Type":element.kind,"Value":element.value})).collect::<Vec<_>>()).collect::<Vec<_>>()
                    })),
                    Err(_) => Ok(json!({"Parsed": false, "SymbolIds": [], "Rules": []})),
                }
            }
            _ => Err(invalid("unknown validated runtime operation")),
        }
    }).collect::<Result<Vec<_>>>()?;
    Ok(Some(json!({"SchemaVersion": 1, "Results": results})))
}
