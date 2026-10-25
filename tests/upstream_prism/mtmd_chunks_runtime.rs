//! Typed serialization/lifetime dispatch to the production media chunk API.

use std::collections::BTreeMap;

use bonsai_prism_runtime::media::chunks::{self, Chunk};
use serde_json::{Value, json};

use super::super::llama_grammar::{array, exact_fields, field};
use super::super::{Result, invalid};

enum Handle {
    Borrowed(usize, usize),
    Owned(Option<Chunk>),
}

fn integer(value: &Value, name: &str, maximum: usize) -> Result<usize> {
    let number = usize::try_from(
        field(value, name)?
            .as_u64()
            .ok_or_else(|| invalid("chunk index must be unsigned"))?,
    )?;
    if number > maximum {
        return Err(invalid("chunk index exceeds source bound"));
    }
    Ok(number)
}

fn get<'a>(
    handle: &'a Handle,
    collections: &'a BTreeMap<usize, Vec<Chunk>>,
) -> Result<Option<&'a Chunk>> {
    match handle {
        Handle::Borrowed(collection, index) => Ok(Some(
            collections
                .get(collection)
                .and_then(|chunks| chunks.get(*index))
                .ok_or_else(|| invalid("borrowed chunk outside collection"))?,
        )),
        Handle::Owned(chunk) => Ok(chunk.as_ref()),
    }
}

fn snapshot(chunk: Option<&Chunk>) -> Result<Value> {
    let Some(chunk) = chunk else {
        return Ok(json!({"Present":false}));
    };
    let kind = match chunk {
        Chunk::Text(_) => 0,
        Chunk::Image(_) => 1,
        Chunk::Audio(_) => 2,
    };
    let mut value = json!({"Present":true,"Type":kind,"TokenCount":chunk.token_count()?,"PositionCount":chunk.position_count()?,"Id":chunk.id()});
    match chunk {
        Chunk::Text(tokens) => {
            value["TokensPresent"] = json!(true);
            value["TextCount"] = json!(tokens.len());
            value["TextTokens"] = json!(tokens);
        }
        Chunk::Image(image) => {
            value["ImagePresent"] = json!(true);
            value["ImageCount"] = json!(image.token_count()?);
            value["ImageId"] = json!(image.id);
        }
        Chunk::Audio(_) => (),
    }
    Ok(value)
}

#[allow(
    clippy::too_many_lines,
    reason = "Source-ordered chunk handle actions preserve borrowed and owned lifetime rules."
)]
fn operation(operation: &Value) -> Result<Value> {
    exact_fields(operation, &["Kind", "Actions"])?;
    let actions = array(operation, "Actions")?;
    if actions.is_empty() || actions.len() > 256 {
        return Err(invalid("chunk action count outside source bound"));
    }
    let mut collections = BTreeMap::new();
    let mut handles = BTreeMap::new();
    let mut buffers = BTreeMap::<usize, Vec<u8>>::new();
    let mut allocated = 0;
    let mut events = Vec::new();
    for action in actions {
        let kind = action["Kind"]
            .as_str()
            .ok_or_else(|| invalid("chunk action kind must be text"))?;
        let event = match kind {
            "defaults" => {
                exact_fields(action, &["Kind"])?;
                json!({"ImageMarker":null})
            }
            "create-test-inputs" => {
                exact_fields(action, &["Kind", "Chunks"])?;
                let id = integer(action, "Chunks", 31)?;
                if collections.insert(id, chunks::test_inputs()).is_some() {
                    return Err(invalid("chunk collection handle reused"));
                }
                json!({"Created":true})
            }
            "collection-info" | "get" => {
                let id = integer(action, "Chunks", 31)?;
                let collection: &Vec<Chunk> = collections
                    .get(&id)
                    .ok_or_else(|| invalid("unknown chunk collection"))?;
                if kind == "collection-info" {
                    exact_fields(action, &["Kind", "Chunks"])?;
                    json!({"Count":collection.len(),"TotalTokens":collection.iter().try_fold(0_usize, |total,chunk| Ok::<_,Box<dyn std::error::Error+Send+Sync>>(total+chunk.token_count()?))?})
                } else {
                    exact_fields(action, &["Kind", "Chunks", "Index", "Chunk"])?;
                    let index = integer(action, "Index", 4095)?;
                    if index >= collection.len() {
                        return Err(invalid("chunk collection index outside extent"));
                    }
                    let handle = integer(action, "Chunk", 63)?;
                    if handles
                        .insert(handle, Handle::Borrowed(id, index))
                        .is_some()
                    {
                        return Err(invalid("chunk handle reused"));
                    }
                    json!({"Present":true})
                }
            }
            "load" => {
                exact_fields(action, &["Kind", "Buffer", "Length", "Chunk"])?;
                let buffer = buffers
                    .get(&integer(action, "Buffer", 63)?)
                    .ok_or_else(|| invalid("unknown chunk cache buffer"))?;
                let length = integer(action, "Length", 1024 * 1024)?;
                let bytes = buffer
                    .get(..length)
                    .ok_or_else(|| invalid("chunk cache length exceeds buffer"))?;
                let chunk = Chunk::load(bytes).ok();
                let created = chunk.is_some();
                if handles
                    .insert(integer(action, "Chunk", 63)?, Handle::Owned(chunk))
                    .is_some()
                {
                    return Err(invalid("loaded chunk handle reused"));
                }
                json!({"Created":created})
            }
            "free" => {
                exact_fields(action, &["Kind", "Chunk"])?;
                match handles.remove(&integer(action, "Chunk", 63)?) {
                    Some(Handle::Owned(_)) => json!({}),
                    _ => return Err(invalid("cannot free unknown or borrowed chunk")),
                }
            }
            _ => {
                let handle = handles
                    .get(&integer(action, "Chunk", 63)?)
                    .ok_or_else(|| invalid("unknown chunk handle"))?;
                let chunk = get(handle, &collections)?;
                if kind == "snapshot" {
                    exact_fields(action, &["Kind", "Chunk"])?;
                    snapshot(chunk)?
                } else {
                    let chunk =
                        chunk.ok_or_else(|| invalid("chunk operation requires live payload"))?;
                    match kind {
                        "image-position" => {
                            exact_fields(action, &["Kind", "Chunk", "Index", "Position0"])?;
                            let Chunk::Image(image) = chunk else {
                                return Err(invalid("image position requires image chunk"));
                            };
                            let [time, x, y, z] = image.position(
                                u32::try_from(integer(action, "Position0", 1_048_576)?)?,
                                integer(action, "Index", 1_048_575)?,
                            )?;
                            json!({"T":time,"X":x,"Y":y,"Z":z})
                        }
                        "query-save" => {
                            exact_fields(action, &["Kind", "Chunk"])?;
                            json!({"ReturnCode":0,"ExpectedLength":chunk.save()?.len()})
                        }
                        "save" => {
                            exact_fields(action, &["Kind", "Chunk", "Buffer", "Length"])?;
                            let id = integer(action, "Buffer", 63)?;
                            let length = integer(action, "Length", 1024 * 1024)?;
                            allocated += length;
                            if length == 0
                                || allocated > 4 * 1024 * 1024
                                || buffers.contains_key(&id)
                            {
                                return Err(invalid("invalid or reused chunk cache buffer"));
                            }
                            let bytes = chunk.save()?;
                            let mut output = vec![0; length];
                            let code = if bytes.len() <= length {
                                output[..bytes.len()].copy_from_slice(&bytes);
                                0
                            } else {
                                -1
                            };
                            buffers.insert(id, output);
                            json!({"Allocated":true,"ReturnCode":code})
                        }
                        _ => return Err(invalid("unknown chunk action")),
                    }
                }
            }
        };
        events.push(event);
    }
    Ok(json!({"Events":events}))
}

pub(super) fn execute(request: &Value) -> Result<Option<Value>> {
    let Some(operations) = request["Operations"].as_array() else {
        return Ok(None);
    };
    if !operations
        .iter()
        .all(|operation| operation["Kind"] == "mtmd-chunks-trace")
    {
        return Ok(None);
    }
    exact_fields(request, &["SchemaVersion", "Operations"])?;
    if request["SchemaVersion"] != 1 || operations.is_empty() || operations.len() > 128 {
        return Err(invalid("invalid chunk transport envelope"));
    }
    Ok(Some(
        json!({"SchemaVersion":1,"Results":operations.iter().map(operation).collect::<Result<Vec<_>>>()?}),
    ))
}
