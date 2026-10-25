//! Complete safe Rust GGUF source contracts, with Windows input generation.
//!
//! Pinned upstream source: b87ca7225456b06b0975b8239c66b5aecf07e37c.
//! MIT attribution is retained in tools/prism/upstream/LICENSE.

use super::engine::Engine;
use super::{Result, invalid, result};
use serde_json::{Value, json};

const ID: &str = "tests/test-gguf.cpp";
const SOURCE_SHA: &str = "38515758a58762f3084db5cd0ce8ddf28d84c78774598a0bbae0fd4c612ea023";
const SEED: u32 = 12345;

use bonsai_prism_runtime::model::gguf_context::{Context as RuntimeContext, Entry as RuntimeEntry};

struct RuntimeScratch {
    directory: std::path::PathBuf,
    path: std::path::PathBuf,
}
impl RuntimeScratch {
    fn new() -> Result<Self> {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let sequence = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "SwiftBonsai-GgufContext-{}-{nonce}-{sequence}",
            std::process::id()
        ));
        if !directory.is_absolute() {
            return Err(invalid("GGUF scratch directory must be absolute"));
        }
        std::fs::create_dir(&directory)?;
        let path = directory.join("context.gguf");
        Ok(Self { directory, path })
    }
}
impl Drop for RuntimeScratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
        let _ = std::fs::remove_dir(&self.directory);
    }
}

fn runtime_unhex(value: &Value) -> Result<Vec<u8>> {
    let source = value
        .as_str()
        .ok_or_else(|| invalid("GGUF DataHex must be text"))?;
    if !source.len().is_multiple_of(2) || source.len() > 16 * 1024 * 1024 {
        return Err(invalid("GGUF byte extent outside bound"));
    }
    source
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| {
            if pair
                .iter()
                .any(|byte| !byte.is_ascii_digit() && !(b'a'..=b'f').contains(byte))
            {
                return Err(invalid("GGUF hex requires lowercase digits"));
            }
            Ok(u8::from_str_radix(std::str::from_utf8(pair)?, 16)?)
        })
        .collect()
}

fn runtime_read(bytes: &[u8], mode: &str, load: bool) -> Result<RuntimeContext> {
    match mode {
        "file" => {
            let file = RuntimeScratch::new()?;
            std::fs::write(&file.path, bytes)?;
            RuntimeContext::read_file(&file.path, load)
        }
        "buffer" => RuntimeContext::read_buffer(bytes, load),
        "callback" => {
            RuntimeContext::read_callback(u64::try_from(bytes.len())?, load, |offset, output| {
                let offset = usize::try_from(offset)?;
                let source = bytes
                    .get(offset..)
                    .ok_or_else(|| invalid("GGUF callback offset outside extent"))?;
                let count = source.len().min(output.len());
                output[..count].copy_from_slice(&source[..count]);
                Ok(count)
            })
        }
        _ => Err(invalid("unknown GGUF ReadMode")),
    }
}

fn runtime_context(descriptor: &Value) -> Result<RuntimeContext> {
    fields(descriptor, &["Entries", "Tensors"])?;
    let entries = array(&descriptor["Entries"])?;
    let tensors = array(&descriptor["Tensors"])?;
    if entries.len() > 256 || tensors.len() > 256 {
        return Err(invalid("GGUF descriptor collection exceeds bound"));
    }
    let mut context = RuntimeContext::new();
    let mut total = 0;
    for entry in entries {
        fields(
            entry,
            &["Key", "Type", "ArrayType", "Count", "DataHex", "StringsHex"],
        )?;
        let key = text(&entry["Key"])?;
        if key.len() > 4096 || key.contains('\0') {
            return Err(invalid("GGUF descriptor key outside bound"));
        }
        let strings = array(&entry["StringsHex"])?;
        let strings = strings
            .iter()
            .map(runtime_unhex)
            .collect::<Result<Vec<_>>>()?;
        let data = runtime_unhex(&entry["DataHex"])?;
        total += data.len() + strings.iter().map(Vec::len).sum::<usize>();
        if total > 8 * 1024 * 1024 {
            return Err(invalid("GGUF descriptor bytes outside bound"));
        }
        context.set_entry(RuntimeEntry {
            key: key.to_owned(),
            kind: u32::try_from(number(&entry["Type"])?)?,
            array_kind: if entry["ArrayType"].is_null() {
                None
            } else {
                Some(u32::try_from(number(&entry["ArrayType"])?)?)
            },
            count: usize::try_from(number(&entry["Count"])?)?,
            data,
            strings,
        })?;
    }
    for tensor in tensors {
        fields(tensor, &["Name", "Type", "Dimensions", "DataHex"])?;
        let shape = array(&tensor["Dimensions"])?;
        if shape.len() != 4 {
            return Err(invalid("GGUF descriptor needs four dimensions"));
        }
        let dimensions = shape.iter().map(number).collect::<Result<Vec<_>>>()?;
        let data = runtime_unhex(&tensor["DataHex"])?;
        total += data.len();
        if total > 8 * 1024 * 1024 {
            return Err(invalid("GGUF descriptor data outside bound"));
        }
        context.add_tensor(
            text(&tensor["Name"])?.to_owned(),
            u32::try_from(number(&tensor["Type"])?)?,
            dimensions
                .try_into()
                .map_err(|_| invalid("GGUF shape requires four dimensions"))?,
            data,
        )?;
    }
    Ok(context)
}

fn runtime_snapshot(context: Option<&RuntimeContext>, selection: &Value) -> Result<Value> {
    let indices = array(selection)?;
    if indices.len() > 256 {
        return Err(invalid("GGUF data selection exceeds bound"));
    }
    let Some(context) = context else {
        return Ok(
            json!({"ContextExists":false,"GgmlContextExists":false,"Header":null,"ContextTensors":null}),
        );
    };
    let mut selected = std::collections::BTreeSet::new();
    for index in indices {
        let index = usize::try_from(number(index)?)?;
        if index >= context.tensors().len() || !selected.insert(index) {
            return Err(invalid(
                "GGUF tensor selection outside extent or duplicated",
            ));
        }
    }
    let keys=context.entries().iter().enumerate().map(|(index,entry)|json!({"Index":index,"LookupIndex":index,"Key":entry.key,"Type":entry.kind,"ArrayType":entry.array_kind,"Count":entry.count,"DataHex":hex(&entry.data),"StringsHex":entry.strings.iter().map(|string|hex(string.split(|&byte|byte==0).next().unwrap_or_default())).collect::<Vec<_>>()})).collect::<Vec<_>>();
    let tensors=context.tensors().iter().enumerate().map(|(index,tensor)|json!({"Index":index,"LookupIndex":index,"Name":tensor.name,"Type":tensor.kind,"Dimensions":tensor.dimensions,"Offset":tensor.offset,"Bytes":tensor.bytes})).collect::<Vec<_>>();
    let context_tensors = if context.has_tensor_context() {
        let mut values = Vec::new();
        if let Some(blob) = context.blob() {
            values.push(
                json!({"Name":"GGUF tensor data binary blob","Bytes":blob.len(),"DataHex":null}),
            );
        }
        let mut captured = 0;
        for (index, tensor) in context.tensors().iter().enumerate() {
            let data = if selected.contains(&index) {
                let data = tensor
                    .data
                    .as_ref()
                    .ok_or_else(|| invalid("GGUF selected tensor data is unavailable"))?;
                captured += data.len();
                if captured > 2 * 1024 * 1024 {
                    return Err(invalid("GGUF selected tensor bytes exceed bound"));
                }
                Some(hex(data))
            } else {
                None
            };
            values.push(json!({"Name":tensor.name,"Bytes":tensor.bytes,"DataHex":data}));
        }
        Some(values)
    } else {
        None
    };
    Ok(
        json!({"ContextExists":true,"GgmlContextExists":context.has_tensor_context(),"Header":{"Version":context.version(),"Alignment":context.alignment(),"DataOffset":context.data_offset(),"NKeys":keys.len(),"NTensors":tensors.len(),"Keys":keys,"Tensors":tensors},"ContextTensors":context_tensors}),
    )
}

fn runtime_operation(operation: &Value) -> Result<Value> {
    match text(&operation["Kind"])? {
        "gguf-inventory" => {
            fields(operation, &["Kind"])?;
            Ok(json!({"Devices":[{"Index":0,"Name":"Rust CPU","Type":0}]}))
        }
        "gguf-read" => {
            fields(
                operation,
                &[
                    "Kind",
                    "DataHex",
                    "ReadMode",
                    "LoadTensors",
                    "TensorDataIndices",
                ],
            )?;
            let bytes = runtime_unhex(&operation["DataHex"])?;
            let mode = text(&operation["ReadMode"])?;
            if !matches!(mode, "file" | "buffer" | "callback") {
                return Err(invalid("unknown GGUF read mode"));
            }
            let context = runtime_read(&bytes, mode, boolean(&operation["LoadTensors"])?).ok();
            runtime_snapshot(context.as_ref(), &operation["TensorDataIndices"])
        }
        "gguf-context-trace" => runtime_trace(operation),
        _ => Err(invalid("unknown GGUF runtime operation")),
    }
}

fn runtime_trace(operation: &Value) -> Result<Value> {
    fields(operation, &["Kind", "Backend", "Contexts", "Actions"])?;
    if number(&operation["Backend"])? != 0 {
        return Err(invalid("GGUF CPU device index outside inventory"));
    }
    let descriptors = array(&operation["Contexts"])?;
    if descriptors.is_empty() || descriptors.len() > 8 {
        return Err(invalid("GGUF context count outside bound"));
    }
    let mut contexts = descriptors
        .iter()
        .map(|descriptor| {
            if descriptor.is_null() {
                Ok(None)
            } else {
                Ok(Some(runtime_context(descriptor)?))
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let actions = array(&operation["Actions"])?;
    if actions.is_empty() || actions.len() > 32 {
        return Err(invalid("GGUF action count outside bound"));
    }
    let context_count = contexts.len();
    let index = |value: &Value| -> Result<usize> {
        let index = usize::try_from(number(value)?)?;
        if index >= context_count {
            return Err(invalid("GGUF context index outside extent"));
        }
        Ok(index)
    };
    let mut events = Vec::with_capacity(actions.len());
    for action in actions {
        let mut event = json!({"Snapshot":null,"WriteResult":null});
        match text(&action["Op"])? {
            "snapshot" => {
                fields(action, &["Op", "Context", "TensorDataIndices"])?;
                event["Snapshot"] = runtime_snapshot(
                    contexts[index(&action["Context"])?].as_ref(),
                    &action["TensorDataIndices"],
                )?;
            }
            "copy-kv" => {
                fields(action, &["Op", "Source", "Destination"])?;
                let source = index(&action["Source"])?;
                let destination = index(&action["Destination"])?;
                if source == destination {
                    return Err(invalid("GGUF metadata copy requires distinct contexts"));
                }
                let copied = contexts[source]
                    .as_ref()
                    .ok_or_else(|| invalid("GGUF metadata source is missing"))?
                    .clone();
                contexts[destination]
                    .as_mut()
                    .ok_or_else(|| invalid("GGUF metadata destination is missing"))?
                    .copy_metadata(&copied)?;
            }
            "write-read" => {
                fields(
                    action,
                    &["Op", "Context", "Destination", "OnlyMetadata", "ReadMode"],
                )?;
                let source = index(&action["Context"])?;
                let destination = index(&action["Destination"])?;
                if source == destination {
                    return Err(invalid("GGUF roundtrip needs distinct contexts"));
                }
                let metadata = boolean(&action["OnlyMetadata"])?;
                let scratch = RuntimeScratch::new()?;
                {
                    let mut file = std::fs::File::create(&scratch.path)?;
                    contexts[source]
                        .as_ref()
                        .ok_or_else(|| invalid("GGUF write source is missing"))?
                        .write_to(&mut file, metadata)?;
                }
                let mode = text(&action["ReadMode"])?;
                if !matches!(mode, "file" | "buffer" | "callback") {
                    return Err(invalid("unknown GGUF read mode"));
                }
                contexts[destination] = if mode == "file" {
                    RuntimeContext::read_file(&scratch.path, !metadata).ok()
                } else {
                    runtime_read(&std::fs::read(&scratch.path)?, mode, !metadata).ok()
                };
                event["WriteResult"] = json!(true);
            }
            _ => return Err(invalid("unknown GGUF context action")),
        }
        events.push(event);
    }
    Ok(json!({"Actions":events}))
}

pub(super) fn execute_runtime(request: &Value) -> Result<Option<Value>> {
    fields(request, &["SchemaVersion", "Operations"])?;
    if request["SchemaVersion"] != 1 {
        return Err(invalid("GGUF protocol version differs"));
    }
    let operations = array(&request["Operations"])?;
    if operations.is_empty() || operations.len() > 128 {
        return Err(invalid("GGUF operation count outside bound"));
    }
    if operations.iter().any(|operation| {
        !matches!(
            operation["Kind"].as_str(),
            Some("gguf-inventory" | "gguf-read" | "gguf-context-trace")
        )
    }) {
        return Ok(None);
    }
    Ok(Some(
        json!({"SchemaVersion":1,"Results":operations.iter().map(runtime_operation).collect::<Result<Vec<_>>>()?}),
    ))
}

const VARIANTS: [u32; 34] = [
    10, 15, 20, 30, 50, 40, 800, 1010, 1015, 1020, 1040, 1050, 1055, 1800, 2010, 2020, 2030, 2035,
    2040, 2045, 2050, 2060, 2070, 2075, 2080, 2800, 2810, 2820, 3010, 3015, 3020, 3030, 3800, 3810,
];

#[derive(Clone)]
struct Random {
    state: [u32; 624],
    index: usize,
}
impl Random {
    fn new(seed: u32) -> Self {
        let mut state = [0; 624];
        state[0] = seed;
        for (index, increment) in (1..624).zip(1_u32..) {
            state[index] = 1_812_433_253_u32
                .wrapping_mul(state[index - 1] ^ (state[index - 1] >> 30))
                .wrapping_add(increment);
        }
        Self { state, index: 624 }
    }
    fn next(&mut self) -> u32 {
        if self.index == 624 {
            for index in 0..624 {
                let value = (self.state[index] & 0x8000_0000)
                    | (self.state[(index + 1) % 624] & 0x7fff_ffff);
                self.state[index] = self.state[(index + 397) % 624]
                    ^ (value >> 1)
                    ^ if value & 1 == 0 { 0 } else { 0x9908_b0df };
            }
            self.index = 0;
        }
        let mut value = self.state[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^ (value >> 18)
    }
    fn bounded(&mut self, bound: u32) -> Result<usize> {
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let product = u64::from(self.next()) * u64::from(bound);
            let low = u32::from_le_bytes(product.to_le_bytes()[..4].try_into()?);
            if low >= threshold {
                return usize::try_from(product >> 32).map_err(Into::into);
            }
        }
    }
    fn shuffle<T>(&mut self, values: &mut [T]) -> Result<()> {
        for index in 1..values.len() {
            let selected = self.bounded(u32::try_from(index + 1)?)?;
            values.swap(index, selected);
        }
        Ok(())
    }
}

// Pinned public block layouts from ggml.c/ggml-common.h. Holes and removed
// formats remain absent, exactly as ggml_type_size(type)==0 in the source.
const fn layout(kind: u32) -> Option<(u64, u64)> {
    Some(match kind {
        0 | 26 => (1, 4),
        1 | 25 | 30 => (1, 2),
        2 | 20 => (32, 18),
        3 => (32, 20),
        6 => (32, 22),
        7 => (32, 24),
        8 => (32, 34),
        9 => (32, 36),
        10 => (256, 84),
        11 | 21 => (256, 110),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        15 => (256, 292),
        16 | 35 => (256, 66),
        17 => (256, 74),
        18 => (256, 98),
        19 => (256, 50),
        22 => (256, 82),
        23 => (256, 136),
        24 => (1, 1),
        27 | 28 => (1, 8),
        29 => (256, 56),
        34 => (256, 54),
        39 => (32, 17),
        40 => (64, 36),
        41 => (128, 18),
        42 => (64, 18),
        142 => (128, 34),
        143 => (128, 28),
        _ => return None,
    })
}
const fn kv_size(kind: u32) -> Option<usize> {
    match kind {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}
fn pad(value: u64, alignment: u64) -> Result<u64> {
    value
        .checked_add(alignment - 1)
        .map(|n| n & !(alignment - 1))
        .ok_or_else(|| invalid("GGUF padding overflow"))
}
fn row_bytes(kind: u32, shape: &[u64; 4]) -> Result<u64> {
    let (block, bytes) = layout(kind).ok_or_else(|| invalid("removed GGUF tensor type"))?;
    let count = shape.iter().try_fold(1_u64, |n, d| {
        n.checked_mul(*d)
            .ok_or_else(|| invalid("GGUF tensor element overflow"))
    })?;
    if !count.is_multiple_of(block) {
        return Err(invalid("GGUF tensor block alignment mismatch"));
    }
    count
        .checked_div(block)
        .and_then(|n| n.checked_mul(bytes))
        .ok_or_else(|| invalid("GGUF tensor byte overflow"))
}
#[derive(Clone)]
struct TensorInput {
    name: String,
    kind: u32,
    shape: [u64; 4],
    data: Vec<u8>,
}
impl TensorInput {
    fn descriptor(&self) -> Value {
        json!({"Name":self.name,"Type":self.kind,"Dimensions":self.shape,"DataHex":hex(&self.data)})
    }
}
fn tensor_configs(rng: &mut Random, variant: u32) -> Result<Vec<TensorInput>> {
    let mut tensors = Vec::new();
    for _ in 0..100 {
        let kind = rng.next() % 144;
        let Some((block, _)) = layout(kind) else {
            continue;
        };
        let mut shape = [1; 4];
        shape[0] = u64::from(1 + rng.next() % 10) * block;
        let dimensions = 1 + rng.next() % 4;
        for dim in &mut shape[1..usize::try_from(dimensions)?] {
            *dim = u64::from(1 + rng.next() % 10);
        }
        tensors.push(TensorInput {
            name: format!("my_tensor_{}", tensors.len()),
            kind,
            shape,
            data: Vec::new(),
        });
    }
    if variant == 2820 && !tensors.is_empty() {
        tensors[0].kind = 42;
        tensors[0].shape = [64, 1, 1, 1];
    }
    Ok(tensors)
}
// Deliberately take a clone: upstream get_kv_types(std::mt19937 rng) takes its
// argument by value, including the Windows ascending Fisher-Yates shuffle.
fn kv_types(mut rng: Random) -> Result<Vec<(u32, Option<u32>)>> {
    let mut types = Vec::new();
    for _ in 0..100 {
        let kind = rng.next() % 13;
        if kind == 9 {
            let array = rng.next() % 13;
            if array != 9 {
                types.push((kind, Some(array)));
            }
        } else {
            types.push((kind, None));
        }
    }
    rng.shuffle(&mut types)?;
    Ok(types)
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct Entry {
    key: String,
    kind: u32,
    array: Option<u32>,
    count: usize,
    data: Vec<u8>,
    strings: Vec<Vec<u8>>,
}
impl Entry {
    fn descriptor(&self) -> Value {
        json!({"Key":self.key,"Type":self.kind,"ArrayType":self.array,"Count":self.count,"DataHex":hex(&self.data),"StringsHex":self.strings.iter().map(|s|hex(s)).collect::<Vec<_>>()})
    }
    fn equivalent(&self, other: &Self) -> bool {
        if self.key != other.key
            || self.kind != other.kind
            || self.array != other.array
            || self.count != other.count
            || self.strings != other.strings
        {
            return false;
        }
        if self.kind == 7 || self.array == Some(7) {
            self.data.len() == other.data.len()
                && self
                    .data
                    .iter()
                    .zip(&other.data)
                    .all(|(a, b)| (*a != 0) == (*b != 0))
        } else {
            self.data == other.data
        }
    }
}
fn generated_kv(rng: &mut Random, types: &[(u32, Option<u32>)]) -> Result<Vec<Entry>> {
    let mut entries = Vec::new();
    for (index, (kind, array)) in types.iter().enumerate() {
        let mut bytes = Vec::with_capacity(64);
        for _ in 0..16 {
            let mut word = rng.next();
            if *kind == 8 || *array == Some(8) {
                word |= 0x0101_0101;
            }
            bytes.extend(word.to_le_bytes());
        }
        let mut entry = Entry {
            key: format!("my_key_{index}"),
            kind: *kind,
            array: *array,
            count: 1,
            data: Vec::new(),
            strings: Vec::new(),
        };
        if *kind == 8 {
            let count = usize::try_from(rng.next() % 64)?;
            entry.strings.push(bytes[..count].to_vec());
        } else if *kind == 9 {
            let array = array.ok_or_else(|| invalid("GGUF array subtype missing"))?;
            if array == 8 {
                entry.count = usize::try_from(rng.next() % 17)?;
                for i in 0..entry.count {
                    let count = usize::try_from(rng.next() % 5)?;
                    entry.strings.push(bytes[i * 4..i * 4 + count].to_vec());
                }
            } else {
                let size =
                    kv_size(array).ok_or_else(|| invalid("GGUF nested array is unsupported"))?;
                entry.count = usize::try_from(rng.next() % 64)? / size;
                entry.data = bytes[..entry.count * size].to_vec();
            }
        } else {
            let size = kv_size(*kind).ok_or_else(|| invalid("unknown GGUF scalar type"))?;
            entry.data = bytes[..size].to_vec();
        }
        entries.push(entry);
    }
    Ok(entries)
}
fn put_string(bytes: &mut Vec<u8>, value: &[u8]) -> Result<()> {
    bytes.extend(u64::try_from(value.len())?.to_le_bytes());
    bytes.extend(value);
    Ok(())
}
fn put_entry(bytes: &mut Vec<u8>, entry: &Entry, bad_key: bool, bad_type: bool) -> Result<()> {
    if bad_key {
        bytes.extend(u64::MAX.to_le_bytes());
        bytes.extend(entry.key.as_bytes());
    } else {
        put_string(bytes, entry.key.as_bytes())?;
    }
    bytes.extend(if bad_type { 13_u32 } else { entry.kind }.to_le_bytes());
    if bad_type {
        bytes.push(
            entry
                .data
                .first()
                .copied()
                .or_else(|| entry.strings.first().and_then(|s| s.first().copied()))
                .unwrap_or(0),
        );
        return Ok(());
    }
    if entry.kind == 8 {
        put_string(bytes, &entry.strings[0])?;
    } else if entry.kind == 9 {
        bytes.extend(
            entry
                .array
                .ok_or_else(|| invalid("GGUF array subtype missing"))?
                .to_le_bytes(),
        );
        bytes.extend(u64::try_from(entry.count)?.to_le_bytes());
        if entry.array == Some(8) {
            for string in &entry.strings {
                put_string(bytes, string)?;
            }
        } else {
            bytes.extend(&entry.data);
        }
    } else {
        bytes.extend(&entry.data);
    }
    Ok(())
}
const fn expected_context(variant: u32) -> bool {
    if variant == 2035 {
        return true;
    }
    if variant < 1000 {
        variant >= 800
    } else if variant < 2000 {
        variant >= 1800
    } else if variant < 3000 {
        variant >= 2800
    } else {
        variant >= 3800
    }
}
const fn custom_alignment(variant: u32) -> bool {
    matches!(variant, 2810 | 3810)
}
fn original_dimensions(shape: &[u64; 4], variant: u32) -> usize {
    if variant == 2035 {
        return 2;
    }
    for i in (1..4).rev() {
        if shape[i] != 1 {
            return i + 1;
        }
    }
    if matches!(variant, 2040 | 2045) { 2 } else { 1 }
}
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::as_conversions,
    reason = "Preserve the original std::pow double expression and explicit int64 narrowing for malformed overflow-only inputs."
)]
fn oversized_dimension(dimensions: usize) -> Result<u64> {
    let exponent = 1.0_f32 / f32::from(u16::try_from(dimensions)?);
    Ok((f64::from(1.01_f32) * ((u64::MAX / 8) as f64).powf(f64::from(exponent))) as u64 + 1)
}
struct Handcrafted {
    variant: u32,
    bytes: Vec<u8>,
    entries: Vec<Entry>,
    tensors: Vec<TensorInput>,
    data_offset: usize,
}
#[allow(
    clippy::too_many_lines,
    reason = "Keep all original malformed input branches together for exact source generator parity review."
)]
fn handcrafted(variant: u32) -> Result<Handcrafted> {
    let mut rng = Random::new(SEED);
    let mut bytes = if variant == 10 {
        b"FUGG".to_vec()
    } else {
        b"GGUF".to_vec()
    };
    bytes.extend(
        match variant {
            15 => 0_u32,
            20 => 1,
            30 => 4,
            _ => 3,
        }
        .to_le_bytes(),
    );
    let mut tensors = if variant >= 2000 {
        tensor_configs(&mut rng, variant)?
    } else {
        Vec::new()
    };
    if variant == 3030 {
        tensors.truncate(2);
        for t in &mut tensors {
            t.kind = 24;
            t.shape = [0x7fff_ffff_ffff_ffc0, 1, 1, 1];
        }
    }
    bytes.extend(
        if variant == 40 {
            u64::MAX
        } else {
            u64::try_from(tensors.len())?
        }
        .to_le_bytes(),
    );
    let types = if variant >= 1000 {
        kv_types(rng.clone())?
    } else {
        Vec::new()
    };
    let has_alignment = matches!(variant, 1050 | 1055 | 2075 | 2810 | 3015 | 3810);
    bytes.extend(
        if variant == 50 {
            u64::MAX
        } else {
            u64::try_from(types.len() + usize::from(has_alignment))?
        }
        .to_le_bytes(),
    );
    let mut entries = if variant == 1020 {
        let mut entries = Vec::new();
        for index in 0..types.len() {
            let first = rng.next().to_le_bytes()[0];
            for _ in 1..16 {
                rng.next();
            }
            entries.push(Entry {
                key: format!("my_key_{index}"),
                kind: 13,
                array: None,
                count: 1,
                data: vec![first],
                strings: Vec::new(),
            });
        }
        entries
    } else if variant >= 1000 {
        generated_kv(&mut rng, &types)?
    } else {
        Vec::new()
    };
    for (index, entry) in entries.iter().enumerate() {
        let mut modified = entry.clone();
        if variant == 1015 {
            modified.key.clear();
        }
        if variant == 1040 {
            modified.key = format!("my_key_{}", index / 2);
        }
        put_entry(&mut bytes, &modified, variant == 1010, variant == 1020)?;
    }
    let mut alignment = 32_u64;
    if has_alignment {
        alignment = if expected_context(variant) { 1 } else { 13 };
        let entry = Entry {
            key: "general.alignment".to_owned(),
            kind: if variant == 1055 { 5 } else { 4 },
            array: None,
            count: 1,
            data: u32::try_from(alignment)?.to_le_bytes().to_vec(),
            strings: Vec::new(),
        };
        put_entry(&mut bytes, &entry, false, false)?;
        entries.push(entry);
    }
    if matches!(variant, 2080 | 3020) {
        alignment = 1;
    }
    let mut offset = 0_u64;
    for tensor in &tensors {
        let kind = if variant == 2045 { 27 } else { tensor.kind };
        let mut name = if variant == 2070 {
            "my_tensor".to_owned()
        } else {
            tensor.name.clone()
        };
        if variant == 2010 {
            name.push_str(
                "_with_a_very_long_name_which_is_longer_than_what_is_allowed_for_ggml_tensors",
            );
        }
        put_string(&mut bytes, name.as_bytes())?;
        let dimensions = original_dimensions(&tensor.shape, variant);
        bytes.extend(u32::try_from(if variant == 2020 { 5 } else { dimensions })?.to_le_bytes());
        for i in 0..dimensions {
            let dimension = match variant {
                2030 => u64::MAX,
                2035 => {
                    if i == 0 {
                        tensor.shape[0]
                    } else {
                        0
                    }
                }
                2040 => 4 * u64::try_from(i32::MAX)?,
                2045 => oversized_dimension(dimensions)?,
                _ => tensor.shape[i],
            };
            bytes.extend(dimension.to_le_bytes());
        }
        bytes.extend(if variant == 2050 { 144 } else { kind }.to_le_bytes());
        bytes.extend(if variant == 2060 { u64::MAX } else { offset }.to_le_bytes());
        let size = if variant == 2035 {
            0
        } else {
            row_bytes(kind, &tensor.shape)?
        };
        offset = offset
            .checked_add(pad(size, alignment)?)
            .ok_or_else(|| invalid("GGUF input offset overflow"))?;
    }
    while !u64::try_from(bytes.len())?.is_multiple_of(alignment) {
        bytes.push(0);
    }
    let data_offset = bytes.len();
    if variant >= 3000 {
        let count = if variant == 3030 {
            32
        } else {
            offset - u64::from(variant == 3010)
        };
        if count > 8 * 1024 * 1024 {
            return Err(invalid("GGUF crafted payload exceeds bound"));
        }
        for i in 0..count {
            bytes.push(i.to_le_bytes()[0]);
        }
    }
    Ok(Handcrafted {
        variant,
        bytes,
        entries,
        tensors,
        data_offset,
    })
}

#[derive(Clone)]
struct ContextInput {
    entries: Vec<Entry>,
    tensors: Vec<TensorInput>,
}
impl ContextInput {
    fn descriptor(&self) -> Value {
        json!({"Entries":self.entries.iter().map(Entry::descriptor).collect::<Vec<_>>(),"Tensors":self.tensors.iter().map(TensorInput::descriptor).collect::<Vec<_>>()})
    }
}
#[allow(
    clippy::cast_precision_loss,
    clippy::as_conversions,
    reason = "Windows mt19937 unsigned arithmetic is converted to float by the original setter expression."
)]
const fn unsigned_float(value: u32) -> f32 {
    value as f32
}
fn random_context(seed: u32) -> Result<ContextInput> {
    let mut rng = Random::new(seed);
    let mut entries = Vec::new();
    for _ in 0..256 {
        let key = format!("my_key_{}", rng.next() % 1024);
        let mut kind = rng.next() % 13;
        let mut array = None;
        let mut count = 1;
        let mut data = Vec::new();
        let mut strings = Vec::new();
        match kind {
            0 => data.push((rng.next() % 128).to_le_bytes()[0]),
            1 => data.push((rng.next() % 128).wrapping_sub(64).to_le_bytes()[0]),
            2 => data.extend(&(rng.next() % 32768).to_le_bytes()[..2]),
            3 => data.extend(&(rng.next() % 32768).wrapping_sub(16384).to_le_bytes()[..2]),
            4 => data.extend(rng.next().to_le_bytes()),
            5 => data.extend(rng.next().wrapping_sub(1 << 30).to_le_bytes()),
            6 | 12 => {
                kind = 6;
                data.extend(unsigned_float((rng.next() % 1024).wrapping_sub(512)).to_le_bytes());
            }
            7 => data.push(u8::from(rng.next() % 2 == 0)),
            8 => strings.push(rng.next().to_string().into_bytes()),
            10 => data.extend(u64::from(rng.next()).to_le_bytes()),
            11 => data.extend(i64::from(rng.next().wrapping_sub(1 << 30)).to_le_bytes()),
            9 => {
                let subtype = rng.next() % 13;
                array = Some(subtype);
                count = usize::try_from(rng.next() % 1024)?;
                if subtype == 9 {
                    continue;
                }
                if subtype == 8 {
                    for _ in 0..count {
                        strings.push(rng.next().to_string().into_bytes());
                    }
                } else {
                    let size = kv_size(subtype)
                        .ok_or_else(|| invalid("unknown random GGUF array type"))?;
                    for _ in 0..(count * size).div_ceil(4) {
                        let mut word = rng.next();
                        if subtype == 7 {
                            word &= 0x0101_0101;
                        }
                        data.extend(word.to_le_bytes());
                    }
                    data.truncate(count * size);
                }
            }
            _ => return Err(invalid("unknown random GGUF KV type")),
        }
        entries.push(Entry {
            key,
            kind,
            array,
            count,
            data,
            strings,
        });
    }
    let mut tensors = Vec::new();
    for index in 0..256 {
        let kind = rng.next() % 144;
        let Some((block, _)) = layout(kind) else {
            continue;
        };
        let dimensions = usize::try_from(1 + rng.next() % 4)?;
        let mut shape = [1; 4];
        shape[0] = u64::from(1 + rng.next() % 10) * block;
        for dimension in &mut shape[1..dimensions] {
            *dimension = u64::from(1 + rng.next() % 10);
        }
        tensors.push(TensorInput {
            name: format!("my_tensor_{index}"),
            kind,
            shape,
            data: Vec::new(),
        });
    }
    for tensor in &mut tensors {
        let size = usize::try_from(row_bytes(tensor.kind, &tensor.shape)?)?;
        for _ in 0..size.div_ceil(4) {
            tensor.data.extend(rng.next().to_le_bytes());
        }
        tensor.data.truncate(size);
    }
    Ok(ContextInput { entries, tensors })
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8] = b"0123456789abcdef";
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        text.push(char::from(DIGITS[usize::from(byte >> 4)]));
        text.push(char::from(DIGITS[usize::from(byte & 15)]));
    }
    text
}

fn fields(value: &Value, names: &[&str]) -> Result<()> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid("GGUF observation must be object"))?;
    if object.len() != names.len() || names.iter().any(|name| !object.contains_key(*name)) {
        return Err(invalid("GGUF observation fields mismatch"));
    }
    Ok(())
}
fn array(value: &Value) -> Result<&[Value]> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| invalid("GGUF array required"))
}
fn number(value: &Value) -> Result<u64> {
    value
        .as_u64()
        .ok_or_else(|| invalid("GGUF unsigned integer required"))
}
fn boolean(value: &Value) -> Result<bool> {
    value
        .as_bool()
        .ok_or_else(|| invalid("GGUF boolean required"))
}
fn text(value: &Value) -> Result<&str> {
    value.as_str().ok_or_else(|| invalid("GGUF text required"))
}
fn unhex(value: &Value) -> Result<Vec<u8>> {
    let text = text(value)?;
    if text.len() > 16 * 1024 * 1024
        || !text.len().is_multiple_of(2)
        || !text
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(invalid("GGUF byte extent/digits invalid"));
    }
    text.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| Ok(u8::from_str_radix(std::str::from_utf8(pair)?, 16)?))
        .collect()
}
fn entry(value: &Value, index: usize) -> Result<Entry> {
    fields(
        value,
        &[
            "Index",
            "LookupIndex",
            "Key",
            "Type",
            "ArrayType",
            "Count",
            "DataHex",
            "StringsHex",
        ],
    )?;
    if number(&value["Index"])? != u64::try_from(index)?
        || value["LookupIndex"].as_i64() != Some(i64::try_from(index)?)
    {
        return Err(invalid("GGUF key lookup/index mismatch"));
    }
    let kind = u32::try_from(number(&value["Type"])?)?;
    let subtype = if value["ArrayType"].is_null() {
        None
    } else {
        Some(u32::try_from(number(&value["ArrayType"])?)?)
    };
    let count = usize::try_from(number(&value["Count"])?)?;
    if count > 1024 * 1024 {
        return Err(invalid("GGUF array count exceeds bound"));
    }
    let data = unhex(&value["DataHex"])?;
    let strings = array(&value["StringsHex"])?
        .iter()
        .map(unhex)
        .collect::<Result<Vec<_>>>()?;
    if kind > 12 || (kind == 9) == subtype.is_none() || subtype == Some(9) {
        return Err(invalid("GGUF KV type/subtype invalid"));
    }
    let element = subtype.unwrap_or(kind);
    if element == 8 {
        if !data.is_empty() || strings.len() != count {
            return Err(invalid("GGUF string extent mismatch"));
        }
    } else {
        let width = kv_size(element).ok_or_else(|| invalid("GGUF scalar width unavailable"))?;
        if !strings.is_empty() || data.len() != count * width {
            return Err(invalid("GGUF numeric extent mismatch"));
        }
    }
    if kind != 9 && count != 1 {
        return Err(invalid("GGUF scalar count mismatch"));
    }
    Ok(Entry {
        key: text(&value["Key"])?.to_owned(),
        kind,
        array: subtype,
        count,
        data,
        strings,
    })
}
struct TensorMeta {
    name: String,
    kind: u32,
    shape: [u64; 4],
    offset: u64,
    bytes: u64,
}
fn tensor_meta(value: &Value, index: usize) -> Result<TensorMeta> {
    fields(
        value,
        &[
            "Index",
            "LookupIndex",
            "Name",
            "Type",
            "Dimensions",
            "Offset",
            "Bytes",
        ],
    )?;
    if number(&value["Index"])? != u64::try_from(index)?
        || value["LookupIndex"].as_i64() != Some(i64::try_from(index)?)
    {
        return Err(invalid("GGUF tensor lookup/index mismatch"));
    }
    let dimensions = array(&value["Dimensions"])?;
    if dimensions.len() != 4 {
        return Err(invalid("GGUF shape requires four dimensions"));
    }
    let mut shape = [0; 4];
    for (target, value) in shape.iter_mut().zip(dimensions) {
        *target = number(value)?;
    }
    Ok(TensorMeta {
        name: text(&value["Name"])?.to_owned(),
        kind: u32::try_from(number(&value["Type"])?)?,
        shape,
        offset: number(&value["Offset"])?,
        bytes: number(&value["Bytes"])?,
    })
}
struct Header {
    version: u64,
    alignment: u64,
    data_offset: usize,
    entries: Vec<Entry>,
    tensors: Vec<TensorMeta>,
}
fn header(value: &Value) -> Result<Header> {
    fields(
        value,
        &[
            "Version",
            "Alignment",
            "DataOffset",
            "NKeys",
            "NTensors",
            "Keys",
            "Tensors",
        ],
    )?;
    let keys = array(&value["Keys"])?;
    let tensors = array(&value["Tensors"])?;
    if keys.len() > 1024
        || tensors.len() > 256
        || number(&value["NKeys"])? != u64::try_from(keys.len())?
        || number(&value["NTensors"])? != u64::try_from(tensors.len())?
    {
        return Err(invalid("GGUF header count mismatch"));
    }
    Ok(Header {
        version: number(&value["Version"])?,
        alignment: number(&value["Alignment"])?,
        data_offset: usize::try_from(number(&value["DataOffset"])?)?,
        entries: keys
            .iter()
            .enumerate()
            .map(|(i, value)| entry(value, i))
            .collect::<Result<Vec<_>>>()?,
        tensors: tensors
            .iter()
            .enumerate()
            .map(|(i, value)| tensor_meta(value, i))
            .collect::<Result<Vec<_>>>()?,
    })
}
struct ContextTensor {
    name: String,
    bytes: u64,
    data: Option<Vec<u8>>,
}
struct Snapshot {
    exists: bool,
    ggml_exists: bool,
    header: Option<Header>,
    context_tensors: Option<Vec<ContextTensor>>,
}
fn snapshot(value: &Value) -> Result<Snapshot> {
    fields(
        value,
        &[
            "ContextExists",
            "GgmlContextExists",
            "Header",
            "ContextTensors",
        ],
    )?;
    let exists = boolean(&value["ContextExists"])?;
    let ggml_exists = boolean(&value["GgmlContextExists"])?;
    let header = if value["Header"].is_null() {
        None
    } else {
        Some(header(&value["Header"])?)
    };
    if exists != header.is_some() {
        return Err(invalid("GGUF context/header presence mismatch"));
    }
    let context_tensors = if value["ContextTensors"].is_null() {
        None
    } else {
        let values = array(&value["ContextTensors"])?;
        if values.len() > 257 {
            return Err(invalid("GGUF context tensor count exceeds bound"));
        }
        Some(
            values
                .iter()
                .map(|value| {
                    fields(value, &["Name", "Bytes", "DataHex"])?;
                    let bytes = number(&value["Bytes"])?;
                    let data = if value["DataHex"].is_null() {
                        None
                    } else {
                        let data = unhex(&value["DataHex"])?;
                        if u64::try_from(data.len())? != bytes {
                            return Err(invalid("GGUF context tensor byte count mismatch"));
                        }
                        Some(data)
                    };
                    Ok(ContextTensor {
                        name: text(&value["Name"])?.to_owned(),
                        bytes,
                        data,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        )
    };
    if !ggml_exists && context_tensors.is_some() {
        return Err(invalid("absent GGML context returned tensors"));
    }
    Ok(Snapshot {
        exists,
        ggml_exists,
        header,
        context_tensors,
    })
}
fn live(value: &Snapshot) -> Result<&Header> {
    value
        .header
        .as_ref()
        .ok_or_else(|| invalid("expected live production GGUF context"))
}
fn exchange(engine: &Engine, name: &str, operations: &[Value]) -> Result<Option<Vec<Value>>> {
    let request = json!({"SchemaVersion":1,"Operations":operations});
    let Some(response) = engine.execute_common(name, &request)? else {
        return Ok(None);
    };
    fields(&response, &["SchemaVersion", "Results"])?;
    if response["SchemaVersion"] != 1 {
        return Err(invalid("GGUF response version mismatch"));
    }
    let values = array(&response["Results"])?;
    if values.len() != operations.len() {
        return Err(invalid("GGUF response operation count mismatch"));
    }
    Ok(Some(values.to_vec()))
}
#[derive(Default)]
struct Checks {
    original: usize,
    failed: usize,
    extra: usize,
    examples: Vec<String>,
}
impl Checks {
    fn check(&mut self, passed: bool, detail: String, extra: bool) {
        if extra {
            self.extra += 1;
        } else {
            self.original += 1;
        }
        if !passed {
            self.failed += 1;
            if self.examples.len() < 12 {
                self.examples.push(detail);
            }
        }
    }
}
fn all_kv(left: &[Entry], right: &[Entry]) -> bool {
    left.iter().all(|entry| {
        right
            .iter()
            .find(|value| value.key == entry.key)
            .is_some_and(|value| entry.equivalent(value))
    })
}
fn all_tensors(left: &[TensorMeta], right: &[TensorMeta]) -> bool {
    left.iter().enumerate().all(|(i, tensor)| {
        right.get(i).is_some_and(|other| {
            tensor.name == other.name && tensor.kind == other.kind && tensor.offset == other.offset
        })
    })
}
fn tensor_geometry(expected: &[TensorInput], actual: &Header) -> Result<bool> {
    if actual.tensors.len() != expected.len() || actual.alignment == 0 {
        return Ok(false);
    }
    let mut offset = 0;
    for (tensor, other) in expected.iter().zip(&actual.tensors) {
        if tensor.name != other.name
            || tensor.kind != other.kind
            || tensor.shape != other.shape
            || offset != other.offset
            || other.bytes != row_bytes(tensor.kind, &tensor.shape)?
        {
            return Ok(false);
        }
        offset = offset
            .checked_add(pad(
                row_bytes(tensor.kind, &tensor.shape)?,
                actual.alignment,
            )?)
            .ok_or_else(|| invalid("GGUF expected offset overflow"))?;
    }
    Ok(true)
}
fn crafted_data(case: &Handcrafted, header: &Header) -> Result<bool> {
    for tensor in &case.tensors {
        let Some(actual) = header
            .tensors
            .iter()
            .find(|value| value.name == tensor.name)
        else {
            return Ok(false);
        };
        let size = usize::try_from(row_bytes(tensor.kind, &tensor.shape)?)?;
        let offset = usize::try_from(actual.offset)?;
        let Some(start) = header.data_offset.checked_add(offset) else {
            return Ok(false);
        };
        let Some(end) = start.checked_add(size) else {
            return Ok(false);
        };
        let Some(bytes) = case.bytes.get(start..end) else {
            return Ok(false);
        };
        for (index, byte) in bytes.iter().enumerate() {
            if *byte != u64::try_from(index + offset)?.to_le_bytes()[0] {
                return Ok(false);
            }
        }
    }
    Ok(true)
}
fn handcrafted_checks(case: &Handcrafted, actual: &Snapshot, checks: &mut Checks) -> Result<()> {
    let expected = expected_context(case.variant);
    checks.check(
        actual.exists == expected,
        format!("variant{}: context presence mismatch", case.variant),
        false,
    );
    if case.variant >= 3000 && !expected {
        checks.check(
            !actual.ggml_exists,
            format!(
                "variant{}: dangling GGML context after parse rejection",
                case.variant
            ),
            false,
        );
    }
    if expected {
        let header = live(actual)?;
        checks.check(
            header.version == 3
                && header.tensors.len() == case.tensors.len()
                && header.entries.len() == case.entries.len(),
            format!("variant{}: header values mismatch", case.variant),
            false,
        );
        if case.variant >= 1000 {
            checks.check(
                all_kv(&case.entries, &header.entries)
                    && header.alignment
                        == if custom_alignment(case.variant) {
                            1
                        } else {
                            32
                        },
                format!(
                    "variant{}: complete KV data/alignment mismatch",
                    case.variant
                ),
                false,
            );
        }
        if case.variant >= 2000 && case.variant != 2035 {
            checks.check(
                tensor_geometry(&case.tensors, header)?,
                format!(
                    "variant{}: tensor names/types/shapes/offsets mismatch",
                    case.variant
                ),
                false,
            );
        }
        if case.variant >= 3000 {
            checks.check(
                crafted_data(case, header)?,
                format!("variant{}: file tensor bytes mismatch", case.variant),
                false,
            );
        }
        checks.check(
            actual.ggml_exists == (case.variant >= 3000),
            format!(
                "variant{}: loaded GGML context presence mismatch",
                case.variant
            ),
            true,
        );
        // DataOffset is returned by production, never inferred by the adapter.
        if case.variant >= 3000 {
            checks.check(
                header.data_offset == case.data_offset,
                format!(
                    "variant{}: production data offset differs from supplied bytes",
                    case.variant
                ),
                true,
            );
        }
    }
    Ok(())
}
fn effective_entries(entries: &[Entry]) -> Vec<Entry> {
    let mut values = Vec::<Entry>::new();
    for entry in entries {
        values.retain(|value| value.key != entry.key);
        values.push(entry.clone());
    }
    values
}
fn input_fidelity(input: &ContextInput, actual: &Snapshot) -> Result<bool> {
    let header = live(actual)?;
    let expected = effective_entries(&input.entries);
    Ok(header.version == 3
        && header.entries.len() == expected.len()
        && all_kv(&expected, &header.entries)
        && tensor_geometry(&input.tensors, header)?)
}
fn same_data(before: &Snapshot, after: &Snapshot) -> bool {
    let (Some(left), Some(right)) = (&before.context_tensors, &after.context_tensors) else {
        return false;
    };
    if right.len() != left.len() + 1
        || right
            .first()
            .is_none_or(|tensor| tensor.name != "GGUF tensor data binary blob")
    {
        return false;
    }
    left.iter()
        .zip(&right[1..])
        .all(|(a, b)| a.bytes == b.bytes && a.data.is_some() && a.data == b.data)
}
fn input_data(input: &ContextInput, snapshot: &Snapshot) -> bool {
    let Some(actual) = &snapshot.context_tensors else {
        return false;
    };
    actual.len() == input.tensors.len()
        && input
            .tensors
            .iter()
            .zip(actual)
            .all(|(a, b)| a.name == b.name && b.data.as_ref() == Some(&a.data))
}
fn trace_events(value: &Value, count: usize) -> Result<&[Value]> {
    fields(value, &["Actions"])?;
    let events = array(&value["Actions"])?;
    if events.len() != count {
        return Err(invalid("GGUF action result count mismatch"));
    }
    for event in events {
        fields(event, &["Snapshot", "WriteResult"])?;
    }
    Ok(events)
}
fn snap_action(context: usize, indices: &[usize]) -> Value {
    json!({"Op":"snapshot","Context":context,"TensorDataIndices":indices})
}
fn roundtrip(
    engine: &Engine,
    device: usize,
    input: &ContextInput,
    only_meta: bool,
    mode: &str,
    checks: &mut Checks,
) -> Result<bool> {
    let indices: Vec<usize> = (0..input.tensors.len()).collect();
    let operation = json!({"Kind":"gguf-context-trace","Backend":device,"Contexts":[input.descriptor(),Value::Null],"Actions":[snap_action(0,if only_meta{&[]}else{&indices}),{"Op":"write-read","Context":0,"Destination":1,"OnlyMetadata":only_meta,"ReadMode":mode},snap_action(1,if only_meta{&[]}else{&indices})]});
    let Some(values) = exchange(
        engine,
        &format!("gguf-roundtrip-{device}-{only_meta}-{mode}"),
        &[operation],
    )?
    else {
        return Ok(false);
    };
    let events = trace_events(&values[0], 3)?;
    let before = snapshot(&events[0]["Snapshot"])?;
    let after = snapshot(&events[2]["Snapshot"])?;
    let left = live(&before)?;
    let right = live(&after)?;
    checks.check(
        left.version == right.version,
        format!("device{device} {mode} meta={only_meta}: version mismatch"),
        false,
    );
    checks.check(
        left.entries.len() == right.entries.len(),
        format!("device{device} {mode}: KV count mismatch"),
        false,
    );
    checks.check(
        left.tensors.len() == right.tensors.len(),
        format!("device{device} {mode}: tensor count mismatch"),
        false,
    );
    checks.check(
        all_kv(&left.entries, &right.entries),
        format!("device{device} {mode}: original KV not in read"),
        false,
    );
    checks.check(
        all_kv(&right.entries, &left.entries),
        format!("device{device} {mode}: read KV not in original"),
        false,
    );
    checks.check(
        all_tensors(&left.tensors, &right.tensors),
        format!("device{device} {mode}: original tensors not in read"),
        false,
    );
    checks.check(
        all_tensors(&right.tensors, &left.tensors),
        format!("device{device} {mode}: read tensors not in original"),
        false,
    );
    if !only_meta {
        checks.check(
            same_data(&before, &after),
            format!("device{device} {mode}: blob name/order/size/full tensor data mismatch"),
            false,
        );
    }
    checks.check(
        boolean(&events[1]["WriteResult"])?,
        format!("device{device} {mode}: production write failed"),
        true,
    );
    checks.check(
        input_fidelity(input, &before)?,
        format!("device{device} {mode}: constructor changed supplied input"),
        true,
    );
    checks.check(
        before.ggml_exists && after.ggml_exists != only_meta,
        format!("device{device} {mode}: GGML context presence mismatch"),
        true,
    );
    if !only_meta {
        checks.check(
            input_data(input, &before),
            format!(
                "device{device} {mode}: production backend tensor data differs from supplied input"
            ),
            true,
        );
    }
    Ok(true)
}
fn copy_kv(
    engine: &Engine,
    device: usize,
    inputs: &[ContextInput; 2],
    checks: &mut Checks,
) -> Result<bool> {
    let empty = ContextInput {
        entries: Vec::new(),
        tensors: Vec::new(),
    };
    let operation = json!({"Kind":"gguf-context-trace","Backend":device,"Contexts":[inputs[0].descriptor(),inputs[1].descriptor(),empty.descriptor()],"Actions":[{"Op":"copy-kv","Source":0,"Destination":1},{"Op":"copy-kv","Source":0,"Destination":2},snap_action(0,&[]),snap_action(1,&[]),snap_action(2,&[]),{"Op":"copy-kv","Source":1,"Destination":0},snap_action(0,&[]),snap_action(1,&[])]});
    let Some(values) = exchange(engine, &format!("gguf-kv-copy-{device}"), &[operation])? else {
        return Ok(false);
    };
    let events = trace_events(&values[0], 8)?;
    let source = snapshot(&events[2]["Snapshot"])?;
    let copied = snapshot(&events[3]["Snapshot"])?;
    let empty = snapshot(&events[4]["Snapshot"])?;
    let merged0 = snapshot(&events[6]["Snapshot"])?;
    let merged1 = snapshot(&events[7]["Snapshot"])?;
    let source_header = live(&source)?;
    let copied_header = live(&copied)?;
    let empty_header = live(&empty)?;
    let merged0_header = live(&merged0)?;
    let merged1_header = live(&merged1)?;
    checks.check(
        source_header.entries.len() == empty_header.entries.len(),
        format!("device{device}: KV copy count mismatch"),
        false,
    );
    checks.check(
        all_kv(&source_header.entries, &copied_header.entries),
        format!("device{device}: source KV missing after copy into nonempty context"),
        false,
    );
    checks.check(
        all_kv(&source_header.entries, &empty_header.entries),
        format!("device{device}: source KV missing after copy into empty context"),
        false,
    );
    checks.check(
        merged0_header.entries.len() == merged1_header.entries.len(),
        format!("device{device}: merged KV count mismatch"),
        false,
    );
    checks.check(
        all_kv(&merged1_header.entries, &merged0_header.entries),
        format!("device{device}: merged KV differs after double copy"),
        false,
    );
    checks.check(
        input_fidelity(&inputs[0], &source)?,
        format!("device{device}: KV source constructor changed supplied input"),
        true,
    );
    checks.check(
        !empty.ggml_exists,
        format!("device{device}: initially empty KV-only context gained a GGML context"),
        true,
    );
    Ok(true)
}
pub(super) fn run(engine: &Engine) -> Result<Value> {
    let Some(metadata) = exchange(
        engine,
        "gguf-inventory",
        &[json!({"Kind":"gguf-inventory"})],
    )?
    else {
        return Ok(result(
            ID,
            false,
            "unsupported: production GGUF APIs are missing".to_owned(),
        ));
    };
    fields(&metadata[0], &["Devices"])?;
    let devices = array(&metadata[0]["Devices"])?;
    if devices.is_empty() || devices.len() > 64 {
        return Err(invalid(
            "GGUF requires a nonempty production backend inventory",
        ));
    }
    for (index, device) in devices.iter().enumerate() {
        fields(device, &["Index", "Name", "Type"])?;
        text(&device["Name"])?;
        if number(&device["Index"])? != u64::try_from(index)? || number(&device["Type"])? != 0 {
            return Err(invalid(
                "GGUF CPU-only acceptance encountered non-CPU or misindexed device",
            ));
        }
    }
    let crafted = VARIANTS
        .into_iter()
        .map(handcrafted)
        .collect::<Result<Vec<_>>>()?;
    let operations:Vec<Value>=crafted.iter().map(|case|json!({"Kind":"gguf-read","DataHex":hex(&case.bytes),"ReadMode":"file","LoadTensors":case.variant>=3000,"TensorDataIndices":[]})).collect();
    let Some(values) = exchange(engine, "gguf-handcrafted", &operations)? else {
        return Ok(result(
            ID,
            false,
            "unsupported: production GGUF reader API is missing".to_owned(),
        ));
    };
    let mut checks = Checks::default();
    for (case, value) in crafted.iter().zip(&values) {
        handcrafted_checks(case, &snapshot(value)?, &mut checks)?;
    }
    if checks.original != 60 {
        return Err(invalid("GGUF handcrafted source assertion count mismatch"));
    }
    let inputs = [random_context(SEED)?, random_context(SEED + 1)?];
    for device in 0..devices.len() {
        for (only_meta, mode) in [
            (true, "file"),
            (false, "file"),
            (false, "buffer"),
            (false, "callback"),
        ] {
            if !roundtrip(engine, device, &inputs[0], only_meta, mode, &mut checks)? {
                return Ok(result(
                    ID,
                    false,
                    "unsupported: production GGUF roundtrip API is missing".to_owned(),
                ));
            }
        }
        if !copy_kv(engine, device, &inputs, &mut checks)? {
            return Ok(result(
                ID,
                false,
                "unsupported: production GGUF KV-copy API is missing".to_owned(),
            ));
        }
    }
    if checks.original != 60 + 36 * devices.len() {
        return Err(invalid("GGUF full-source assertion count mismatch"));
    }
    Ok(result(
        ID,
        checks.failed == 0,
        format!(
            "source={SOURCE_SHA}; seed12345 Windows MT19937/shuffle; all34 handcrafted variants,4 read-mode roundtrips and KV copy/merge per CPU device ({} devices); original contracts={}, extra construction/data guards={}; failures={}; examples={:?}",
            devices.len(),
            checks.original,
            checks.extra,
            checks.failed,
            checks.examples
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_malformed_snapshot_and_detects_binary_value_change() -> Result<()> {
        assert!(snapshot(&json!({"ContextExists":true,"GgmlContextExists":false,"Header":null,"ContextTensors":null})).is_err());
        assert!(unhex(&json!("0g")).is_err());
        let before = Entry {
            key: "raw".to_owned(),
            kind: 12,
            array: None,
            count: 1,
            data: 0x7ff8_0000_0000_0001_u64.to_le_bytes().to_vec(),
            strings: Vec::new(),
        };
        let mut after = before.clone();
        assert!(before.equivalent(&after));
        after.data[0] ^= 1;
        assert!(!before.equivalent(&after));
        assert_eq!(VARIANTS.len(), 34);
        let mut rng = Random::new(SEED);
        let next = rng.clone().next();
        let _ = kv_types(rng.clone())?;
        assert_eq!(rng.next(), next);
        assert_eq!(pad(18, 13)?, 18);
        Ok(())
    }
}
