//! Independent source-defined MTMD factory, wire records and owned reloads.
//!
//! The reference owns serialized packets and uses a bounded stream decoder;
//! it never calls the production Rust Chunk enum or a native library. Only the
//! original factory and states reachable through the retired transport are
//! supplied. This is source-derived behavior, not captured native execution.
//! Upstream MIT attribution remains in tools/prism/upstream/LICENSE.

use std::{
    collections::BTreeMap,
    io::{Cursor, Read as _},
};

use serde::Deserialize;
use serde_json::{Value, json};

use super::{Result, invalid};

pub(super) const ID: &str = "tests/test-mtmd-c-api.c";
const LIMIT: usize = 1024 * 1024;

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
    actions: Vec<Action>,
}

#[derive(Deserialize)]
#[serde(tag = "Kind", rename_all_fields = "PascalCase", deny_unknown_fields)]
enum Action {
    #[serde(rename = "defaults")]
    Defaults,
    #[serde(rename = "create-test-inputs")]
    Create { chunks: usize },
    #[serde(rename = "collection-info")]
    Info { chunks: usize },
    #[serde(rename = "get")]
    Get {
        chunks: usize,
        index: usize,
        chunk: usize,
    },
    #[serde(rename = "snapshot")]
    Snapshot { chunk: usize },
    #[serde(rename = "image-position")]
    Position {
        chunk: usize,
        index: usize,
        position0: usize,
    },
    #[serde(rename = "query-save")]
    Query { chunk: usize },
    #[serde(rename = "save")]
    Save {
        chunk: usize,
        buffer: usize,
        length: usize,
    },
    #[serde(rename = "load")]
    Load {
        buffer: usize,
        length: usize,
        chunk: usize,
    },
    #[serde(rename = "free")]
    Free { chunk: usize },
}

struct WireChunk {
    bytes: Vec<u8>,
    observation: Value,
    image: Option<[u64; 5]>, // width, height, layout, image ordinal, token count
}

fn word<const N: usize>(input: &mut Cursor<&[u8]>) -> Result<[u8; N]> {
    let mut bytes = [0; N];
    input.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn uint(input: &mut Cursor<&[u8]>) -> Result<u64> {
    Ok(u64::from(u32::from_le_bytes(word(input)?)))
}

fn length(input: &mut Cursor<&[u8]>, element_size: usize) -> Result<usize> {
    let count = usize::try_from(u64::from_le_bytes(word(input)?))?;
    let remaining = input
        .get_ref()
        .len()
        .saturating_sub(usize::try_from(input.position())?);
    if count > remaining / element_size {
        return Err(invalid("independent chunk extent exceeds original buffer"));
    }
    Ok(count)
}

fn boolean(input: &mut Cursor<&[u8]>) -> Result<bool> {
    Ok(word::<1>(input)?[0] != 0)
}

fn identifier(input: &mut Cursor<&[u8]>) -> Result<String> {
    let count = length(input, 1)?;
    let mut bytes = vec![0; count];
    input.read_exact(&mut bytes)?;
    Ok(String::from_utf8(bytes)?)
}

fn frames(input: &mut Cursor<&[u8]>) -> Result<u64> {
    boolean(input)?; // the original batch's is_audio field
    let count = length(input, 10)?;
    // Each placeholder is two flag bytes followed by signed width/height.
    // Preserve the bytes without allocating a second frame/pixel graph.
    let next = input
        .position()
        .checked_add(u64::try_from(count)? * 10)
        .ok_or_else(|| invalid("independent frame cursor overflow"))?;
    input.set_position(next);
    Ok(u64::try_from(count)?)
}

#[expect(
    clippy::too_many_lines,
    reason = "Read the independent bounded v1 record in its original wire-field order."
)]
fn decode(bytes: &[u8]) -> Result<WireChunk> {
    if bytes.len() > LIMIT {
        return Err(invalid("independent chunk buffer exceeds 1 MiB"));
    }
    let mut input = Cursor::new(bytes);
    if u64::from_le_bytes(word(&mut input)?) != 1 {
        return Err(invalid("independent chunk version mismatch"));
    }
    let kind = uint(&mut input)?;
    if kind > 2 {
        return Err(invalid("independent chunk type outside source enum"));
    }
    let text_count = length(&mut input, 4)?;
    let mut text = Vec::with_capacity(text_count);
    for _ in 0..text_count {
        text.push(i32::from_le_bytes(word(&mut input)?));
    }
    let image = if boolean(&mut input)? {
        let width = uint(&mut input)?;
        let height = uint(&mut input)?;
        let layout = uint(&mut input)?;
        let ordinal = uint(&mut input)?;
        let merge = uint(&mut input)?;
        if layout > 2 {
            return Err(invalid(
                "independent image position type outside source enum",
            ));
        }
        let id = identifier(&mut input)?;
        let count = frames(&mut input)?;
        let count = if layout == 2 {
            width
                .checked_add(1)
                .and_then(|width| width.checked_mul(height))
                .and_then(|count| count.checked_add(2))
        } else {
            let temporal = if merge > 1 {
                count.div_ceil(merge)
            } else {
                count
            };
            width
                .checked_mul(height)
                .and_then(|area| area.checked_mul(temporal))
        }
        .filter(|count| *count <= 1_048_576)
        .ok_or_else(|| invalid("independent image token bound exceeded"))?;
        Some(([width, height, layout, ordinal, count], id))
    } else {
        None
    };
    let audio = if boolean(&mut input)? {
        let tokens = uint(&mut input)?;
        let id = identifier(&mut input)?;
        frames(&mut input)?;
        Some((tokens, id))
    } else {
        None
    };
    let (tokens, positions, id) = match kind {
        0 => (
            u64::try_from(text_count)?,
            u64::try_from(text_count)?,
            Value::Null,
        ),
        1 => {
            let (image, id) = image
                .as_ref()
                .ok_or_else(|| invalid("independent image payload missing"))?;
            (
                image[4],
                if image[2] == 1 {
                    image[0].max(image[1])
                } else {
                    image[4]
                },
                json!(id),
            )
        }
        _ => {
            let (count, id) = audio.ok_or_else(|| invalid("independent audio payload missing"))?;
            (count, count, json!(id))
        }
    };
    let mut observation =
        json!({"Present":true,"Type":kind,"TokenCount":tokens,"PositionCount":positions,"Id":id});
    if kind == 0 {
        observation["TokensPresent"] = json!(!text.is_empty());
        observation["TextCount"] = json!(text_count);
        observation["TextTokens"] = json!(text);
    } else if kind == 1 {
        let (image, id) = image
            .as_ref()
            .ok_or_else(|| invalid("independent image payload missing"))?;
        observation["ImagePresent"] = json!(true);
        observation["ImageCount"] = json!(image[4]);
        observation["ImageId"] = json!(id);
    }
    // Native loading ignores trailing buffer capacity; saving returns only the
    // complete source record. Reachable factory records use canonical booleans.
    let end = usize::try_from(input.position())?;
    Ok(WireChunk {
        bytes: bytes[..end].to_vec(),
        observation,
        image: image.map(|(image, _)| image),
    })
}

fn factory() -> Result<[WireChunk; 2]> {
    // Independent flat field recipes from mtmd.cpp/clip-impl.h. This is neither
    // a captured expected-output file nor a call to production test_inputs().
    let mut text = 1_u64.to_le_bytes().to_vec();
    text.extend(0_u32.to_le_bytes());
    text.extend(5_u64.to_le_bytes());
    for token in 1_i32..=5 {
        text.extend(token.to_le_bytes());
    }
    text.extend([0, 0]); // no image or audio payload
    let mut image = 1_u64.to_le_bytes().to_vec();
    image.extend(1_u32.to_le_bytes());
    image.extend(0_u64.to_le_bytes());
    image.push(1);
    for value in [4_u32, 4, 0, 0, 1] {
        image.extend(value.to_le_bytes());
    }
    image.extend(7_u64.to_le_bytes());
    image.extend(b"image_1");
    image.push(0); // image batch, not audio
    image.extend(16_u64.to_le_bytes());
    image.extend([0; 160]); // sixteen 10-byte default placeholder frames
    image.push(0); // no audio payload
    Ok([decode(&text)?, decode(&image)?])
}

enum Handle {
    Borrowed(usize, usize),
    Owned(Option<WireChunk>),
}

fn handle(value: usize, maximum: usize) -> Result<usize> {
    if value > maximum {
        return Err(invalid(
            "independent chunk handle/index exceeds source bound",
        ));
    }
    Ok(value)
}

fn chunk<'a>(
    id: usize,
    handles: &'a BTreeMap<usize, Handle>,
    collections: &'a BTreeMap<usize, [WireChunk; 2]>,
) -> Result<Option<&'a WireChunk>> {
    match handles
        .get(&handle(id, 63)?)
        .ok_or_else(|| invalid("unknown independent chunk handle"))?
    {
        Handle::Borrowed(collection, index) => Ok(Some(&collections[collection][*index])),
        Handle::Owned(value) => Ok(value.as_ref()),
    }
}

fn live<'a>(
    id: usize,
    handles: &'a BTreeMap<usize, Handle>,
    collections: &'a BTreeMap<usize, [WireChunk; 2]>,
) -> Result<&'a WireChunk> {
    chunk(id, handles, collections)?
        .ok_or_else(|| invalid("independent chunk operation requires a live payload"))
}

#[expect(
    clippy::too_many_lines,
    reason = "Preserve the source-ordered handle and buffer lifetime dispatcher in one auditable scope."
)]
fn operation(operation: &Operation) -> Result<Value> {
    if operation.kind != "mtmd-chunks-trace" || !(1..=256).contains(&operation.actions.len()) {
        return Err(invalid(
            "independent chunk operation/action count outside source bound",
        ));
    }
    let mut collections = BTreeMap::new();
    let mut handles = BTreeMap::new();
    let mut buffers = BTreeMap::<usize, Vec<u8>>::new();
    let mut allocated = 0;
    let mut events = Vec::new();
    for action in &operation.actions {
        let event = match *action {
            Action::Defaults => json!({"ImageMarker":null}),
            Action::Create { chunks } => {
                if collections
                    .insert(handle(chunks, 31)?, factory()?)
                    .is_some()
                {
                    return Err(invalid("independent chunk collection handle reused"));
                }
                json!({"Created":true})
            }
            Action::Info { chunks } | Action::Get { chunks, .. } => {
                let collection = collections
                    .get(&handle(chunks, 31)?)
                    .ok_or_else(|| invalid("unknown independent chunk collection"))?;
                if let Action::Get { index, chunk, .. } = *action {
                    handle(index, 4095)?;
                    if index >= collection.len()
                        || handles
                            .insert(handle(chunk, 63)?, Handle::Borrowed(chunks, index))
                            .is_some()
                    {
                        return Err(invalid("independent chunk index or handle reuse"));
                    }
                    json!({"Present":true})
                } else {
                    let total = collection.iter().try_fold(0_u64, |sum, chunk| {
                        sum.checked_add(
                            chunk.observation["TokenCount"]
                                .as_u64()
                                .ok_or_else(|| invalid("independent chunk count missing"))?,
                        )
                        .ok_or_else(|| invalid("independent collection count overflow"))
                    })?;
                    json!({"Count":collection.len(),"TotalTokens":total})
                }
            }
            Action::Snapshot { chunk: id } => chunk(id, &handles, &collections)?.map_or_else(
                || json!({"Present":false}),
                |chunk| chunk.observation.clone(),
            ),
            Action::Position {
                chunk,
                index,
                position0,
            } => {
                handle(index, 1_048_575)?;
                handle(position0, 1_048_576)?;
                let [width, _, layout, ordinal, count] = live(chunk, &handles, &collections)?
                    .image
                    .ok_or_else(|| invalid("independent position requires image chunk"))?;
                let index = u64::try_from(index)?;
                let start = u64::try_from(position0)?;
                if width == 0 || index >= count {
                    return Err(invalid("independent image position outside extent"));
                }
                let sequential = start + index;
                let [time, x, y, z] =
                    if layout == 0 || (layout == 2 && (index == 0 || index + 1 == count)) {
                        [sequential; 4]
                    } else if layout == 1 {
                        [start, start + index % width, start + index / width, 0]
                    } else {
                        [
                            sequential,
                            (index - 1) / (width + 1),
                            (index - 1) % (width + 1),
                            ordinal,
                        ]
                    };
                json!({"T":time,"X":x,"Y":y,"Z":z})
            }
            Action::Query { chunk } => {
                json!({"ReturnCode":0,"ExpectedLength":live(chunk, &handles, &collections)?.bytes.len()})
            }
            Action::Save {
                chunk,
                buffer,
                length,
            } => {
                handle(buffer, 63)?;
                handle(length, LIMIT)?;
                allocated += length;
                if length == 0 || allocated > 4 * LIMIT || buffers.contains_key(&buffer) {
                    return Err(invalid(
                        "independent chunk buffer reused/empty or aggregate bound exceeded",
                    ));
                }
                let payload = &live(chunk, &handles, &collections)?.bytes;
                let mut output = vec![0; length];
                let code = if length < payload.len() {
                    -1
                } else {
                    output[..payload.len()].copy_from_slice(payload);
                    0
                };
                buffers.insert(buffer, output);
                json!({"Allocated":true,"ReturnCode":code})
            }
            Action::Load {
                buffer,
                length,
                chunk,
            } => {
                handle(length, LIMIT)?;
                let bytes = buffers
                    .get(&handle(buffer, 63)?)
                    .and_then(|bytes| bytes.get(..length))
                    .ok_or_else(|| invalid("independent load exceeds its original buffer"))?;
                let loaded = decode(bytes).ok();
                let created = loaded.is_some();
                if handles
                    .insert(handle(chunk, 63)?, Handle::Owned(loaded))
                    .is_some()
                {
                    return Err(invalid("independent loaded chunk handle reused"));
                }
                json!({"Created":created})
            }
            Action::Free { chunk } => match handles.remove(&handle(chunk, 63)?) {
                Some(Handle::Owned(_)) => json!({}),
                _ => return Err(invalid("cannot free unknown or borrowed independent chunk")),
            },
        };
        events.push(event);
    }
    Ok(json!({"Events":events}))
}

pub(super) fn handles(value: &Value) -> bool {
    value["Operations"].as_array().is_some_and(|operations| {
        !operations.is_empty()
            && operations
                .iter()
                .all(|operation| operation["Kind"] == "mtmd-chunks-trace")
    })
}

pub(super) fn execute(value: &Value) -> Result<Value> {
    let request: Request = serde_json::from_value(value.clone())?;
    if request.schema_version != 1 || !(1..=128).contains(&request.operations.len()) {
        return Err(invalid("independent chunk envelope outside source bound"));
    }
    let results = request
        .operations
        .iter()
        .map(operation)
        .collect::<Result<Vec<_>>>()?;
    Ok(json!({"SchemaVersion":1,"Results":results}))
}

pub(super) fn identity() -> Value {
    json!({"Kind":"independent-rust-mtmd-chunk-wire", "SupportedSourceIds":[ID], "UpstreamCommit":super::UPSTREAM,
        "AssertionSource":{"Path":ID,"Bytes":5_735,"Sha256":"bdeb47199c1bc878566c7b47dd4905e6b82fb42586ad92aabc6ee27e1647d884"},
        "ProductionSources":[
            {"Path":"tools/mtmd/mtmd.cpp","Bytes":103_968,"Sha256":"9300de0676f5a3100ab67b0194e4999a57b217d12ef5c0c933069d7bf9592925"},
            {"Path":"tools/mtmd/mtmd-helper.cpp","Bytes":34_595,"Sha256":"de3eb3ae5d261635051bc50d3dae89f5ff8420e06ab616dedfca2d7348ea4eaf"},
            {"Path":"tools/mtmd/clip-impl.h","Bytes":43_760,"Sha256":"23060c1dea9dec19a722d1ff7f5639996826a49be6ab5446aec5b0a7e0d6c335"}],
        "Architecture":"source-defined-flat-wire-recipes-and-bounded-cursor-decoder-v1",
        "Scope":"original-two-chunk-factory-and-reachable-transport-states",
        "NativeReferenceReplay":false,"ProductionRuntimeCalled":false,"CompleteSourceGate":false,"NativeRetirementQualified":false})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, Target};

    fn request(actions: Value) -> Value {
        let mut request = json!({"SchemaVersion":1,"Operations":[{"Kind":"mtmd-chunks-trace"}]});
        request["Operations"][0]["Actions"] = actions;
        request
    }

    #[test]
    fn independent_mtmd_chunks_preserves_all_46_source_assertions_and_required_reference_id()
    -> Result<()> {
        for target in [Target::RustReference, Target::Cpp, Target::Rust] {
            let engine = Engine::new(target, None, None, None, None)?;
            let report = crate::mtmd_chunks::run(&engine)?;
            assert_eq!(report["Id"], ID);
            assert_eq!(report["Status"], "pass", "{report}");
            assert!(
                report["Detail"]
                    .as_str()
                    .is_some_and(|detail| detail.contains("All46"))
            );
        }
        for field in [
            "NativeReferenceReplay",
            "ProductionRuntimeCalled",
            "CompleteSourceGate",
            "NativeRetirementQualified",
        ] {
            assert_eq!(identity()[field], false);
        }
        Ok(())
    }

    #[test]
    fn independent_mtmd_chunks_has_exact_source_wire_lengths_metadata_and_positions() -> Result<()>
    {
        let [text, image] = factory()?;
        assert_eq!(text.bytes.len(), 42);
        assert_eq!(
            text.observation,
            json!({"Present":true,"Type":0,"TokenCount":5,"PositionCount":5,"Id":null,"TokensPresent":true,"TextCount":5,"TextTokens":[1,2,3,4,5]})
        );
        assert_eq!(image.bytes.len(), 226);
        assert_eq!(
            image.observation,
            json!({"Present":true,"Type":1,"TokenCount":256,"PositionCount":256,"Id":"image_1","ImagePresent":true,"ImageCount":256,"ImageId":"image_1"})
        );
        assert_eq!(&image.bytes[65..225], &[0; 160]);
        let result = execute(&request(json!([
            {"Kind":"defaults"},{"Kind":"create-test-inputs","Chunks":0},
            {"Kind":"collection-info","Chunks":0},{"Kind":"get","Chunks":0,"Index":1,"Chunk":0},
            {"Kind":"image-position","Chunk":0,"Index":255,"Position0":13},
            {"Kind":"query-save","Chunk":0}
        ])))?;
        assert_eq!(
            result["Results"][0]["Events"][0],
            json!({"ImageMarker":null})
        );
        assert_eq!(
            result["Results"][0]["Events"][2],
            json!({"Count":2,"TotalTokens":261})
        );
        assert_eq!(
            result["Results"][0]["Events"][4],
            json!({"T":268,"X":268,"Y":268,"Z":268})
        );
        assert_eq!(
            result["Results"][0]["Events"][5],
            json!({"ReturnCode":0,"ExpectedLength":226})
        );
        Ok(())
    }

    #[test]
    fn independent_mtmd_chunks_decoder_rejects_every_truncation_and_bounded_malformed_fields()
    -> Result<()> {
        for chunk in factory()? {
            for end in 0..chunk.bytes.len() {
                assert!(decode(&chunk.bytes[..end]).is_err(), "truncated at {end}");
            }
            let mut extended = chunk.bytes.clone();
            extended.extend([7, 8, 9]);
            assert_eq!(decode(&extended)?.bytes, chunk.bytes);
            for (offset, value) in [(0, 2_u64), (12, u64::MAX)] {
                let mut changed = chunk.bytes.clone();
                changed[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
                assert!(decode(&changed).is_err());
            }
            let mut changed = chunk.bytes;
            changed[8..12].copy_from_slice(&3_u32.to_le_bytes());
            assert!(decode(&changed).is_err());
        }
        let [_, image] = factory()?;
        for (offset, value) in [(41, u64::MAX), (57, u64::MAX)] {
            let mut changed = image.bytes.clone();
            changed[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
            assert!(decode(&changed).is_err());
        }
        let mut changed = image.bytes;
        changed[29..33].copy_from_slice(&3_u32.to_le_bytes());
        assert!(decode(&changed).is_err());
        assert!(decode(&vec![0; LIMIT + 1]).is_err());
        Ok(())
    }

    #[test]
    fn independent_mtmd_chunks_preserves_owned_null_reload_lifecycle_and_rejects_bad_schedules()
    -> Result<()> {
        let prefix = vec![
            json!({"Kind":"create-test-inputs","Chunks":0}),
            json!({"Kind":"get","Chunks":0,"Index":0,"Chunk":0}),
        ];
        let mut actions = prefix.clone();
        actions.extend([
            json!({"Kind":"save","Chunk":0,"Buffer":0,"Length":42}),
            json!({"Kind":"load","Buffer":0,"Length":41,"Chunk":1}),
            json!({"Kind":"snapshot","Chunk":1}),
            json!({"Kind":"free","Chunk":1}),
            json!({"Kind":"load","Buffer":0,"Length":42,"Chunk":1}),
            json!({"Kind":"snapshot","Chunk":1}),
            json!({"Kind":"free","Chunk":1}),
            json!({"Kind":"snapshot","Chunk":0}),
        ]);
        let events = execute(&request(json!(actions)))?["Results"][0]["Events"].clone();
        assert_eq!(events[4], json!({"Present":false}));
        assert_eq!(events[7], events[9]);
        for bad in [
            json!({"Kind":"free","Chunk":0}),
            json!({"Kind":"get","Chunks":0,"Index":0,"Chunk":0}),
            json!({"Kind":"get","Chunks":0,"Index":2,"Chunk":1}),
            json!({"Kind":"get","Chunks":0,"Index":0,"Chunk":64}),
            json!({"Kind":"save","Chunk":0,"Buffer":0,"Length":0}),
            json!({"Kind":"save","Chunk":0,"Buffer":0,"Length":LIMIT+1}),
            json!({"Kind":"snapshot","Chunk":true}),
            json!({"Kind":"snapshot","Chunk":0,"Extra":true}),
            json!({"Kind":"image-position","Chunk":0,"Index":0,"Position0":0}),
            json!({"Kind":"unknown","Chunk":0}),
        ] {
            let mut actions = prefix.clone();
            actions.push(bad);
            assert!(execute(&request(json!(actions))).is_err());
        }
        let mut actions = prefix;
        for buffer in 0..5 {
            actions.push(json!({"Kind":"save","Chunk":0,"Buffer":buffer,"Length":LIMIT}));
        }
        assert!(execute(&request(json!(actions))).is_err());
        assert!(execute(&request(json!([]))).is_err());
        assert!(execute(&request(json!(vec![json!({"Kind":"defaults"}); 257]))).is_err());
        Ok(())
    }
}
