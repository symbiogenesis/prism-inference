//! Clean-room independent safe Rust GGUF v2/v3 container reference provider.
//!
//! No production `bonsai_prism_runtime` code, native bridge or precomputed
//! answer fixture is called here. Parses and writes little-endian GGUF headers,
//! ordered metadata entries and contiguous aligned tensor payloads directly.

use std::collections::BTreeSet;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

use super::{Result, invalid};

pub(super) const ID: &str = "tests/test-gguf.cpp";
const SOURCE_SHA: &str = "38515758a58762f3084db5cd0ce8ddf28d84c78774598a0bbae0fd4c612ea023";
const MAX_WIRE_BYTES: usize = 8 * 1024 * 1024;
const MAX_ITEMS: usize = 65_536;
const MAX_PARSE_ALIGNMENT: u32 = 1_048_576;
const MAX_SETTER_ALIGNMENT: u32 = 4096;
const MAX_SELECTED_BYTES: usize = 2 * 1024 * 1024;
const CALLBACK_WINDOW: usize = 4096;

#[derive(Clone)]
struct MetadataEntry {
    key: String,
    kind: u32,
    array_kind: Option<u32>,
    count: usize,
    data: Vec<u8>,
    strings: Vec<Vec<u8>>,
}

#[derive(Clone)]
struct TensorRecord {
    name: String,
    kind: u32,
    dimensions: [u64; 4],
    offset: u64,
    bytes: u64,
    data: Option<Vec<u8>>,
}

#[derive(Clone)]
struct Container {
    version: u32,
    alignment: u64,
    data_offset: u64,
    entries: Vec<MetadataEntry>,
    entry_keys: BTreeSet<String>,
    tensors: Vec<TensorRecord>,
    tensor_names: BTreeSet<String>,
    loaded: bool,
    blob: Option<Vec<u8>>,
}

impl Container {
    const fn new() -> Self {
        Self {
            version: 3,
            alignment: 32,
            data_offset: 0,
            entries: Vec::new(),
            entry_keys: BTreeSet::new(),
            tensors: Vec::new(),
            tensor_names: BTreeSet::new(),
            loaded: false,
            blob: None,
        }
    }

    fn set_entry(&mut self, entry: MetadataEntry) -> Result<()> {
        validate_entry(&entry, true)?;
        if self.entry_keys.contains(&entry.key) {
            self.entries.retain(|existing| existing.key != entry.key);
        } else {
            if self.entries.len() >= MAX_ITEMS {
                return Err(invalid("GGUF metadata entry count exceeds bound"));
            }
            self.entry_keys.insert(entry.key.clone());
        }
        self.entries.push(entry);
        Ok(())
    }

    fn copy_kv_from(&mut self, source: &Self) -> Result<()> {
        for entry in &source.entries {
            let mut copied = entry.clone();
            if copied.kind == 8 || copied.array_kind == Some(8) {
                for raw in &mut copied.strings {
                    if let Some(end) = raw.iter().position(|&byte| byte == 0) {
                        raw.truncate(end);
                    }
                }
            }
            if copied.kind == 7 || copied.array_kind == Some(7) {
                for byte in &mut copied.data {
                    *byte = u8::from(*byte != 0);
                }
            }
            self.set_entry(copied)?;
        }
        Ok(())
    }

    fn add_tensor(
        &mut self,
        name: String,
        kind: u32,
        dimensions: [u64; 4],
        data: Vec<u8>,
    ) -> Result<()> {
        let bytes = tensor_byte_size(kind, dimensions)?;
        if name.len() >= 64
            || name.contains('\0')
            || self.tensor_names.contains(&name)
            || self.tensors.len() >= MAX_ITEMS
            || usize::try_from(bytes)? != data.len()
            || data.len() > MAX_WIRE_BYTES
        {
            return Err(invalid("GGUF tensor descriptor or payload is invalid"));
        }
        let offset = match self.tensors.last() {
            Some(prev) => prev
                .offset
                .checked_add(align_up(prev.bytes, self.alignment)?)
                .ok_or_else(|| invalid("GGUF tensor offset overflow"))?,
            None => 0,
        };
        self.tensor_names.insert(name.clone());
        self.tensors.push(TensorRecord {
            name,
            kind,
            dimensions,
            offset,
            bytes,
            data: Some(data),
        });
        self.loaded = true;
        Ok(())
    }

    fn encode(&self, only_metadata: bool) -> Result<Vec<u8>> {
        let mut out = b"GGUF".to_vec();
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&u64::try_from(self.tensors.len())?.to_le_bytes());
        out.extend_from_slice(&u64::try_from(self.entries.len())?.to_le_bytes());
        for entry in &self.entries {
            encode_string(&mut out, entry.key.as_bytes())?;
            out.extend_from_slice(&entry.kind.to_le_bytes());
            if let Some(array_kind) = entry.array_kind {
                out.extend_from_slice(&array_kind.to_le_bytes());
                out.extend_from_slice(&u64::try_from(entry.count)?.to_le_bytes());
            }
            if entry.kind == 8 || entry.array_kind == Some(8) {
                for raw in &entry.strings {
                    encode_string(&mut out, raw)?;
                }
            } else {
                out.extend_from_slice(&entry.data);
            }
        }
        for tensor in &self.tensors {
            encode_string(&mut out, tensor.name.as_bytes())?;
            let rank = tensor
                .dimensions
                .iter()
                .rposition(|&dim| dim != 1)
                .map_or(1, |idx| idx + 1);
            out.extend_from_slice(&u32::try_from(rank)?.to_le_bytes());
            for dim in &tensor.dimensions[..rank] {
                out.extend_from_slice(&dim.to_le_bytes());
            }
            out.extend_from_slice(&tensor.kind.to_le_bytes());
            out.extend_from_slice(&tensor.offset.to_le_bytes());
        }
        let header_end = usize::try_from(align_up(u64::try_from(out.len())?, self.alignment)?)?;
        if header_end > MAX_WIRE_BYTES {
            return Err(invalid("GGUF encoded metadata exceeds wire bound"));
        }
        out.resize(header_end, 0);
        if !only_metadata {
            for tensor in &self.tensors {
                let data = tensor
                    .data
                    .as_ref()
                    .ok_or_else(|| invalid("GGUF write requires loaded tensor bytes"))?;
                if out.len().checked_sub(header_end) != Some(usize::try_from(tensor.offset)?)
                    || usize::try_from(tensor.bytes)? != data.len()
                {
                    return Err(invalid("GGUF tensor write offset or extent mismatch"));
                }
                let next_end = out
                    .len()
                    .checked_add(usize::try_from(align_up(tensor.bytes, self.alignment)?)?)
                    .filter(|&end| end <= MAX_WIRE_BYTES)
                    .ok_or_else(|| invalid("GGUF encoded payload exceeds wire bound"))?;
                out.extend_from_slice(data);
                out.resize(next_end, 0);
            }
        }
        Ok(out)
    }
}

struct ScratchFile {
    directory: PathBuf,
    path: PathBuf,
}

impl ScratchFile {
    fn new() -> Result<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let seq = NEXT.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "SwiftBonsai-RefGguf-{}-{nonce}-{seq}",
            std::process::id()
        ));
        if !directory.is_absolute() {
            return Err(invalid("GGUF reference scratch directory must be absolute"));
        }
        std::fs::create_dir(&directory)?;
        let path = directory.join("payload.gguf");
        Ok(Self { directory, path })
    }
}

impl Drop for ScratchFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
        let _ = std::fs::remove_dir(&self.directory);
    }
}

struct ByteReader<F> {
    offset: u64,
    length: u64,
    read: F,
}

impl<F: FnMut(u64, &mut [u8]) -> Result<usize>> ByteReader<F> {
    fn take(&mut self, count: usize) -> Result<Vec<u8>> {
        if count > MAX_WIRE_BYTES
            || self
                .offset
                .checked_add(u64::try_from(count)?)
                .is_none_or(|end| end > self.length)
        {
            return Err(invalid("GGUF read request exceeds source extent"));
        }
        let mut buf = vec![0_u8; count];
        let mut done = 0;
        while done < count {
            let chunk_end = (done + CALLBACK_WINDOW).min(count);
            let got = (self.read)(self.offset, &mut buf[done..chunk_end])?;
            if got != chunk_end - done {
                return Err(invalid("GGUF reader returned short chunk"));
            }
            self.offset += u64::try_from(got)?;
            done += got;
        }
        Ok(buf)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.as_slice().try_into()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.as_slice().try_into()?))
    }

    fn bytes_string(&mut self) -> Result<Vec<u8>> {
        let len = usize::try_from(self.u64()?)?;
        self.take(len)
    }
}

const fn tensor_layout(kind: u32) -> Option<(u64, u64)> {
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

fn scalar_width(kind: u32) -> Result<usize> {
    match kind {
        0 | 1 | 7 => Ok(1),
        2 | 3 => Ok(2),
        4..=6 => Ok(4),
        10..=12 => Ok(8),
        _ => Err(invalid("GGUF scalar type has no numeric width")),
    }
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    value
        .checked_add(alignment - 1)
        .map(|aligned| aligned & !(alignment - 1))
        .ok_or_else(|| invalid("GGUF alignment arithmetic overflow"))
}

fn tensor_byte_size(kind: u32, dimensions: [u64; 4]) -> Result<u64> {
    let (block, width) =
        tensor_layout(kind).ok_or_else(|| invalid("GGUF tensor type unsupported"))?;
    if dimensions.iter().any(|&dim| dim > i64::MAX.unsigned_abs())
        || !dimensions[0].is_multiple_of(block)
    {
        return Err(invalid("GGUF tensor dimensions violate layout"));
    }
    if dimensions.contains(&0) {
        return Ok(0);
    }
    let elements = dimensions
        .iter()
        .try_fold(1_u64, |acc, &dim| acc.checked_mul(dim))
        .filter(|&count| count < i64::MAX.unsigned_abs())
        .ok_or_else(|| invalid("GGUF tensor element count overflow"))?;
    (elements / block)
        .checked_mul(width)
        .ok_or_else(|| invalid("GGUF tensor byte size overflow"))
}

fn encode_string(out: &mut Vec<u8>, bytes: &[u8]) -> Result<()> {
    if out
        .len()
        .checked_add(bytes.len())
        .is_none_or(|end| end > MAX_WIRE_BYTES)
    {
        return Err(invalid("GGUF serialized string exceeds bound"));
    }
    out.extend_from_slice(&u64::try_from(bytes.len())?.to_le_bytes());
    out.extend_from_slice(bytes);
    Ok(())
}

fn validate_entry(entry: &MetadataEntry, setter: bool) -> Result<()> {
    let key_limit = if setter { 4096 } else { MAX_WIRE_BYTES };
    let align_limit = if setter {
        MAX_SETTER_ALIGNMENT
    } else {
        MAX_PARSE_ALIGNMENT
    };
    if entry.key.is_empty()
        || entry.key.len() > key_limit
        || entry.key.contains('\0')
        || entry.count > MAX_WIRE_BYTES
    {
        return Err(invalid("GGUF metadata key or count outside bound"));
    }
    let element = if entry.kind == 9 {
        entry
            .array_kind
            .filter(|&kind| kind != 9)
            .ok_or_else(|| invalid("GGUF array requires non-array element type"))?
    } else {
        if entry.array_kind.is_some() || entry.count != 1 {
            return Err(invalid("GGUF scalar metadata descriptor mismatch"));
        }
        entry.kind
    };
    if element == 8 {
        if !entry.data.is_empty()
            || entry.strings.len() != entry.count
            || entry
                .strings
                .iter()
                .any(|raw| raw.len() > MAX_WIRE_BYTES || (setter && raw.contains(&0)))
        {
            return Err(invalid("GGUF string metadata descriptor invalid"));
        }
    } else {
        let width = scalar_width(element)?;
        if !entry.strings.is_empty()
            || entry.count.checked_mul(width) != Some(entry.data.len())
            || entry.data.len() > MAX_WIRE_BYTES
        {
            return Err(invalid("GGUF numeric metadata extent mismatch"));
        }
        if setter && element == 7 && entry.data.iter().any(|&b| b > 1) {
            return Err(invalid("GGUF bool setter requires zero or one"));
        }
    }
    if entry.key == "general.alignment" {
        if entry.kind != 4 || entry.data.len() != 4 {
            return Err(invalid("GGUF alignment entry must be UINT32"));
        }
        let align = u32::from_le_bytes(entry.data[..4].try_into()?);
        if !align.is_power_of_two() || align > align_limit {
            return Err(invalid("GGUF alignment must be power of two within bound"));
        }
    }
    Ok(())
}

fn parse_container(
    length: u64,
    load_tensors: bool,
    read: impl FnMut(u64, &mut [u8]) -> Result<usize>,
) -> Result<Container> {
    let mut reader = ByteReader {
        offset: 0,
        length,
        read,
    };
    if reader.take(4)? != b"GGUF" {
        return Err(invalid("GGUF magic header mismatch"));
    }
    let version = reader.u32()?;
    if !matches!(version, 2 | 3) {
        return Err(invalid("unsupported GGUF format version"));
    }
    let tensor_count = usize::try_from(reader.u64()?)?;
    let key_count = usize::try_from(reader.u64()?)?;
    if tensor_count > MAX_ITEMS || key_count > MAX_ITEMS {
        return Err(invalid("GGUF header collection count exceeds bound"));
    }
    let mut container = Container {
        version,
        ..Container::new()
    };
    for _ in 0..key_count {
        let entry = read_metadata_entry(&mut reader, &mut container)?;
        container.entries.push(entry);
    }
    let total_tensor_bytes = read_tensor_infos(&mut reader, &mut container, tensor_count)?;
    if tensor_count > 0 {
        let start = align_up(reader.offset, container.alignment)?;
        if start > reader.length {
            return Err(invalid("GGUF aligned data offset exceeds extent"));
        }
        reader.offset = start;
    }
    container.data_offset = reader.offset;
    if load_tensors {
        let blob = reader.take(usize::try_from(total_tensor_bytes)?)?;
        for tensor in &mut container.tensors {
            let start = usize::try_from(tensor.offset)?;
            let end = start
                .checked_add(usize::try_from(tensor.bytes)?)
                .ok_or_else(|| invalid("GGUF tensor payload span overflow"))?;
            tensor.data = Some(
                blob.get(start..end)
                    .ok_or_else(|| invalid("GGUF tensor payload exceeds blob"))?
                    .to_vec(),
            );
        }
        container.loaded = true;
        container.blob = Some(blob);
    }
    Ok(container)
}

fn read_metadata_entry<F: FnMut(u64, &mut [u8]) -> Result<usize>>(
    reader: &mut ByteReader<F>,
    container: &mut Container,
) -> Result<MetadataEntry> {
    let key = String::from_utf8(reader.bytes_string()?)?;
    if !container.entry_keys.insert(key.clone()) {
        return Err(invalid("duplicate GGUF metadata key"));
    }
    let kind = reader.u32()?;
    let array_kind = if kind == 9 { Some(reader.u32()?) } else { None };
    let count = if kind == 9 {
        usize::try_from(reader.u64()?)?
    } else {
        1
    };
    if count > MAX_WIRE_BYTES {
        return Err(invalid("GGUF metadata array count exceeds bound"));
    }
    let element = array_kind.unwrap_or(kind);
    let mut data = Vec::new();
    let mut strings = Vec::new();
    if element == 8 {
        if count > MAX_ITEMS {
            return Err(invalid("GGUF string array count exceeds bound"));
        }
        for _ in 0..count {
            strings.push(reader.bytes_string()?);
        }
    } else {
        data = reader.take(
            count
                .checked_mul(scalar_width(element)?)
                .ok_or_else(|| invalid("GGUF metadata numeric byte count overflow"))?,
        )?;
        if element == 7 {
            for byte in &mut data {
                *byte = u8::from(*byte != 0);
            }
        }
    }
    let entry = MetadataEntry {
        key,
        kind,
        array_kind,
        count,
        data,
        strings,
    };
    validate_entry(&entry, false)?;
    if entry.key == "general.alignment" {
        container.alignment = u64::from(u32::from_le_bytes(entry.data[..4].try_into()?));
    }
    Ok(entry)
}

fn read_tensor_infos<F: FnMut(u64, &mut [u8]) -> Result<usize>>(
    reader: &mut ByteReader<F>,
    container: &mut Container,
    tensor_count: usize,
) -> Result<u64> {
    let mut total = 0_u64;
    for _ in 0..tensor_count {
        let raw_name = reader.bytes_string()?;
        if raw_name.len() >= 64 || raw_name.contains(&0) {
            return Err(invalid("GGUF tensor name overlong or contains NUL"));
        }
        let name = String::from_utf8(raw_name)?;
        if !container.tensor_names.insert(name.clone()) {
            return Err(invalid("duplicate GGUF tensor name"));
        }
        let rank = usize::try_from(reader.u32()?)?;
        if !(1..=4).contains(&rank) {
            return Err(invalid("GGUF tensor rank must be in 1..=4"));
        }
        let mut dimensions = [1_u64; 4];
        for dim in &mut dimensions[..rank] {
            *dim = reader.u64()?;
        }
        let kind = reader.u32()?;
        let bytes = tensor_byte_size(kind, dimensions)?;
        let offset = reader.u64()?;
        if offset != total {
            return Err(invalid("GGUF tensor offset deviates from aligned layout"));
        }
        total = total
            .checked_add(align_up(bytes, container.alignment)?)
            .ok_or_else(|| invalid("GGUF total tensor extent overflow"))?;
        container.tensors.push(TensorRecord {
            name,
            kind,
            dimensions,
            offset,
            bytes,
            data: None,
        });
    }
    Ok(total)
}

fn read_with_mode(bytes: &[u8], mode: &str, load_tensors: bool) -> Result<Container> {
    match mode {
        "file" => {
            let scratch = ScratchFile::new()?;
            std::fs::write(&scratch.path, bytes)?;
            read_from_file(&scratch.path, load_tensors)
        }
        "buffer" | "callback" => {
            parse_container(u64::try_from(bytes.len())?, load_tensors, |offset, dst| {
                copy_slice(bytes, offset, dst)
            })
        }
        _ => Err(invalid("unknown GGUF ReadMode")),
    }
}

fn read_from_file(path: &std::path::Path, load_tensors: bool) -> Result<Container> {
    let mut file = File::open(path)?;
    let length = file.metadata()?.len();
    parse_container(length, load_tensors, |offset, dst| {
        file.seek(SeekFrom::Start(offset))?;
        Ok(file.read(dst)?)
    })
}

fn copy_slice(source: &[u8], offset: u64, dst: &mut [u8]) -> Result<usize> {
    let offset = usize::try_from(offset)?;
    let available = source
        .get(offset..)
        .ok_or_else(|| invalid("GGUF reader offset outside buffer"))?;
    let count = available.len().min(dst.len());
    dst[..count].copy_from_slice(&available[..count]);
    Ok(count)
}

fn build_container(descriptor: &Value) -> Result<Container> {
    fields(descriptor, &["Entries", "Tensors"])?;
    let entries = array(&descriptor["Entries"])?;
    let tensors = array(&descriptor["Tensors"])?;
    if entries.len() > 256 || tensors.len() > 256 {
        return Err(invalid("GGUF descriptor count exceeds bound"));
    }
    let mut container = Container::new();
    let mut total_bytes = 0_usize;
    for entry in entries {
        fields(
            entry,
            &["Key", "Type", "ArrayType", "Count", "DataHex", "StringsHex"],
        )?;
        let key = text_no_nul(&entry["Key"], 4096)?;
        let strings_json = array(&entry["StringsHex"])?;
        let mut strings = Vec::with_capacity(strings_json.len());
        for item in strings_json {
            let decoded = unhex(item, 1024 * 1024)?;
            total_bytes += decoded.len();
            strings.push(decoded);
        }
        let data = unhex(&entry["DataHex"], MAX_WIRE_BYTES)?;
        total_bytes += data.len();
        if total_bytes > MAX_WIRE_BYTES {
            return Err(invalid("GGUF descriptor entry bytes exceed bound"));
        }
        let kind = u32::try_from(number(&entry["Type"], 12)?)?;
        let array_kind = if entry["ArrayType"].is_null() {
            None
        } else {
            Some(u32::try_from(number(&entry["ArrayType"], 12)?)?)
        };
        let count = usize::try_from(number(&entry["Count"], 1_048_576)?)?;
        container.set_entry(MetadataEntry {
            key: key.to_owned(),
            kind,
            array_kind,
            count,
            data,
            strings,
        })?;
    }
    for tensor in tensors {
        fields(tensor, &["Name", "Type", "Dimensions", "DataHex"])?;
        let name = text_no_nul(&tensor["Name"], 63)?;
        let dims = array(&tensor["Dimensions"])?;
        if dims.len() != 4 {
            return Err(invalid("GGUF tensor descriptor needs 4 dimensions"));
        }
        let shape = [
            number(&dims[0], 1_048_576)?,
            number(&dims[1], 1_048_576)?,
            number(&dims[2], 1_048_576)?,
            number(&dims[3], 1_048_576)?,
        ];
        if shape.contains(&0) {
            return Err(invalid(
                "GGUF descriptor tensor dimensions must be positive",
            ));
        }
        let data = unhex(&tensor["DataHex"], MAX_WIRE_BYTES)?;
        total_bytes += data.len();
        if total_bytes > MAX_WIRE_BYTES {
            return Err(invalid("GGUF descriptor tensor payload exceeds bound"));
        }
        let kind = u32::try_from(number(&tensor["Type"], 143)?)?;
        container.add_tensor(name.to_owned(), kind, shape, data)?;
    }
    Ok(container)
}

fn snapshot_keys(container: &Container) -> Vec<Value> {
    container
        .entries
        .iter()
        .enumerate()
        .map(|(idx, entry)| {
            let strings = entry
                .strings
                .iter()
                .map(|raw| {
                    let prefix = raw.split(|&byte| byte == 0).next().unwrap_or_default();
                    hex(prefix)
                })
                .collect::<Vec<_>>();
            json!({
                "Index": idx,
                "LookupIndex": idx,
                "Key": entry.key,
                "Type": entry.kind,
                "ArrayType": entry.array_kind,
                "Count": entry.count,
                "DataHex": hex(&entry.data),
                "StringsHex": strings,
            })
        })
        .collect()
}

fn snapshot_context_tensors(
    container: &Container,
    selected: &BTreeSet<usize>,
) -> Result<Option<Vec<Value>>> {
    if !container.loaded {
        return Ok(None);
    }
    let mut out = Vec::new();
    if let Some(blob) = &container.blob {
        out.push(json!({
            "Name": "GGUF tensor data binary blob",
            "Bytes": blob.len(),
            "DataHex": null,
        }));
    }
    let mut captured = 0_usize;
    for (idx, tensor) in container.tensors.iter().enumerate() {
        let data_hex = if selected.contains(&idx) {
            let raw = tensor
                .data
                .as_ref()
                .ok_or_else(|| invalid("GGUF tensor payload unavailable"))?;
            captured += raw.len();
            if captured > MAX_SELECTED_BYTES {
                return Err(invalid("GGUF snapshot selected tensor bytes exceed 2 MiB"));
            }
            Some(hex(raw))
        } else {
            None
        };
        out.push(json!({
            "Name": tensor.name,
            "Bytes": tensor.bytes,
            "DataHex": data_hex,
        }));
    }
    Ok(Some(out))
}

fn snapshot_container(container: Option<&Container>, selection: &Value) -> Result<Value> {
    let selected_values = array(selection)?;
    if selected_values.len() > 256 {
        return Err(invalid("GGUF tensor data selection exceeds bound"));
    }
    let Some(container) = container else {
        if !selected_values.is_empty() {
            for item in selected_values {
                let _ = number(item, 255)?;
            }
            return Err(invalid("GGUF selected tensor index on absent context"));
        }
        return Ok(json!({
            "ContextExists": false,
            "GgmlContextExists": false,
            "Header": null,
            "ContextTensors": null
        }));
    };
    let mut selected = BTreeSet::new();
    for item in selected_values {
        let index = usize::try_from(number(item, 255)?)?;
        if index >= container.tensors.len() || !selected.insert(index) {
            return Err(invalid("GGUF selected tensor index invalid or duplicated"));
        }
    }
    if container.entries.len() > 1024 || container.tensors.len() > 256 {
        return Err(invalid("GGUF snapshot collection exceeds bound"));
    }
    let keys = snapshot_keys(container);
    let infos = container
        .tensors
        .iter()
        .enumerate()
        .map(|(idx, tensor)| {
            json!({
                "Index": idx,
                "LookupIndex": idx,
                "Name": tensor.name,
                "Type": tensor.kind,
                "Dimensions": tensor.dimensions,
                "Offset": tensor.offset,
                "Bytes": tensor.bytes,
            })
        })
        .collect::<Vec<_>>();
    let context_tensors = snapshot_context_tensors(container, &selected)?;
    Ok(json!({
        "ContextExists": true,
        "GgmlContextExists": container.loaded,
        "Header": {
            "Version": container.version,
            "Alignment": container.alignment,
            "DataOffset": container.data_offset,
            "NKeys": keys.len(),
            "NTensors": infos.len(),
            "Keys": keys,
            "Tensors": infos,
        },
        "ContextTensors": context_tensors,
    }))
}

fn execute_trace(operation: &Value) -> Result<Value> {
    fields(operation, &["Kind", "Backend", "Contexts", "Actions"])?;
    if number(&operation["Backend"], 63)? != 0 {
        return Err(invalid("GGUF CPU backend index outside inventory"));
    }
    let descriptors = array(&operation["Contexts"])?;
    if descriptors.is_empty() || descriptors.len() > 8 {
        return Err(invalid("GGUF context count outside bound"));
    }
    let mut contexts = descriptors
        .iter()
        .map(|desc| {
            if desc.is_null() {
                Ok(None)
            } else {
                Ok(Some(build_container(desc)?))
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let actions = array(&operation["Actions"])?;
    if actions.is_empty() || actions.len() > 32 {
        return Err(invalid("GGUF action count outside bound"));
    }
    let limit = contexts.len() - 1;
    let mut results = Vec::with_capacity(actions.len());
    for action in actions {
        let mut event = json!({"Snapshot": null, "WriteResult": null});
        match text(&action["Op"])? {
            "snapshot" => {
                fields(action, &["Op", "Context", "TensorDataIndices"])?;
                let idx = usize::try_from(number(&action["Context"], u64::try_from(limit)?)?)?;
                event["Snapshot"] =
                    snapshot_container(contexts[idx].as_ref(), &action["TensorDataIndices"])?;
            }
            "copy-kv" => {
                fields(action, &["Op", "Source", "Destination"])?;
                let src = usize::try_from(number(&action["Source"], u64::try_from(limit)?)?)?;
                let dst = usize::try_from(number(&action["Destination"], u64::try_from(limit)?)?)?;
                if src == dst {
                    return Err(invalid("GGUF copy requires distinct contexts"));
                }
                let copied = contexts[src]
                    .as_ref()
                    .ok_or_else(|| invalid("GGUF copy source does not exist"))?
                    .clone();
                contexts[dst]
                    .as_mut()
                    .ok_or_else(|| invalid("GGUF copy destination does not exist"))?
                    .copy_kv_from(&copied)?;
            }
            "write-read" => {
                fields(
                    action,
                    &["Op", "Context", "Destination", "OnlyMetadata", "ReadMode"],
                )?;
                let src = usize::try_from(number(&action["Context"], u64::try_from(limit)?)?)?;
                let dst = usize::try_from(number(&action["Destination"], u64::try_from(limit)?)?)?;
                if src == dst {
                    return Err(invalid("GGUF write-read requires distinct contexts"));
                }
                let only_metadata = boolean(&action["OnlyMetadata"])?;
                let mode = check_mode(&action["ReadMode"])?;
                let scratch = ScratchFile::new()?;
                let encoded = contexts[src]
                    .as_ref()
                    .ok_or_else(|| invalid("GGUF write source does not exist"))?
                    .encode(only_metadata)?;
                {
                    let mut file = File::create(&scratch.path)?;
                    file.write_all(&encoded)?;
                }
                contexts[dst] = if mode == "file" {
                    read_from_file(&scratch.path, !only_metadata).ok()
                } else {
                    read_with_mode(&encoded, mode, !only_metadata).ok()
                };
                event["WriteResult"] = json!(true);
            }
            _ => return Err(invalid("unknown GGUF action Op")),
        }
        results.push(event);
    }
    Ok(json!({"Actions": results}))
}

fn execute_operation(operation: &Value) -> Result<Value> {
    match text(&operation["Kind"])? {
        "gguf-inventory" => {
            fields(operation, &["Kind"])?;
            Ok(
                json!({"Devices": [{"Index": 0, "Name": "Independent Rust Reference CPU", "Type": 0}]}),
            )
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
            let bytes = unhex(&operation["DataHex"], MAX_WIRE_BYTES)?;
            let mode = check_mode(&operation["ReadMode"])?;
            let load = boolean(&operation["LoadTensors"])?;
            let container = read_with_mode(&bytes, mode, load).ok();
            snapshot_container(container.as_ref(), &operation["TensorDataIndices"])
        }
        "gguf-context-trace" => execute_trace(operation),
        _ => Err(invalid("unknown GGUF reference operation")),
    }
}

fn fields(value: &Value, names: &[&str]) -> Result<()> {
    let obj = value
        .as_object()
        .ok_or_else(|| invalid("GGUF message must be object"))?;
    if obj.len() != names.len() || names.iter().any(|name| !obj.contains_key(*name)) {
        return Err(invalid("GGUF object fields mismatch"));
    }
    Ok(())
}

fn array(value: &Value) -> Result<&[Value]> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| invalid("GGUF array required"))
}

fn number(value: &Value, maximum: u64) -> Result<u64> {
    let n = value
        .as_u64()
        .ok_or_else(|| invalid("GGUF nonnegative integer required"))?;
    if n > maximum {
        return Err(invalid("GGUF integer exceeds bound"));
    }
    Ok(n)
}

fn boolean(value: &Value) -> Result<bool> {
    value
        .as_bool()
        .ok_or_else(|| invalid("GGUF boolean required"))
}

fn text(value: &Value) -> Result<&str> {
    value.as_str().ok_or_else(|| invalid("GGUF text required"))
}

fn text_no_nul(value: &Value, maximum: usize) -> Result<&str> {
    let s = text(value)?;
    if s.len() > maximum || s.contains('\0') {
        return Err(invalid("GGUF text exceeds bound or contains NUL"));
    }
    Ok(s)
}

fn check_mode(value: &Value) -> Result<&str> {
    let mode = text(value)?;
    if !matches!(mode, "file" | "buffer" | "callback") {
        return Err(invalid("unknown GGUF ReadMode"));
    }
    Ok(mode)
}

fn unhex(value: &Value, max_bytes: usize) -> Result<Vec<u8>> {
    let s = text(value)?;
    if !s.len().is_multiple_of(2) || s.len() / 2 > max_bytes {
        return Err(invalid("GGUF hex extent outside bound"));
    }
    if !s
        .bytes()
        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(invalid("GGUF hex requires lowercase hexadecimal digits"));
    }
    s.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| Ok(u8::from_str_radix(std::str::from_utf8(pair)?, 16)?))
        .collect()
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(char::from(DIGITS[usize::from(byte >> 4)]));
        out.push(char::from(DIGITS[usize::from(byte & 15)]));
    }
    out
}

pub(super) fn handles(request: &Value) -> bool {
    request["Operations"].as_array().is_some_and(|operations| {
        !operations.is_empty()
            && operations.iter().all(|operation| {
                matches!(
                    operation["Kind"].as_str(),
                    Some("gguf-inventory" | "gguf-read" | "gguf-context-trace")
                )
            })
    })
}

pub(super) fn execute(request: &Value) -> Result<Value> {
    fields(request, &["SchemaVersion", "Operations"])?;
    if request["SchemaVersion"] != 1 {
        return Err(invalid("GGUF request schema version must equal 1"));
    }
    let operations = array(&request["Operations"])?;
    if operations.is_empty() || operations.len() > 128 {
        return Err(invalid("GGUF operation count outside bound"));
    }
    let results = operations
        .iter()
        .map(execute_operation)
        .collect::<Result<Vec<_>>>()?;
    Ok(json!({"SchemaVersion": 1, "Results": results}))
}

pub(super) fn identity() -> Value {
    json!({
        "Kind": "independent-rust-gguf",
        "SupportedSourceIds": [ID],
        "UpstreamCommit": super::UPSTREAM,
        "AssertionSource": {"Path": ID, "Sha256": SOURCE_SHA},
        "Architecture": "clean-room-gguf-v2-v3-container-parser-writer-and-kv-trace-v1",
        "NativeReferenceReplay": false,
        "ProductionRuntimeCalled": false,
        "CompleteSourceGate": false,
        "NativeRetirementQualified": false
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, Target};

    #[test]
    fn independent_gguf_provider_passes_complete_test_gguf_suite_across_all_targets() -> Result<()>
    {
        for target in [Target::RustReference, Target::Cpp, Target::Rust] {
            let engine = Engine::new(target, None, None, None, None, None)?;
            let report = crate::gguf::run(&engine)?;
            assert_eq!(report["Id"], ID);
            assert_eq!(report["Status"], "pass", "{report}");
        }
        assert!(crate::reference_provider::supports(ID));
        let info = identity();
        assert_eq!(info["SupportedSourceIds"], json!([ID]));
        assert_eq!(info["NativeReferenceReplay"], false);
        assert_eq!(info["ProductionRuntimeCalled"], false);
        assert_eq!(info["CompleteSourceGate"], false);
        assert_eq!(info["NativeRetirementQualified"], false);
        Ok(())
    }

    #[test]
    fn independent_gguf_supports_v2_tensor_data_selection_and_rejects_malformed_requests()
    -> Result<()> {
        let mut header = b"GGUF".to_vec();
        header.extend_from_slice(&2_u32.to_le_bytes());
        header.extend_from_slice(&1_u64.to_le_bytes());
        header.extend_from_slice(&1_u64.to_le_bytes());
        encode_string(&mut header, b"flag")?;
        header.extend_from_slice(&7_u32.to_le_bytes());
        header.push(9);
        encode_string(&mut header, b"w0")?;
        header.extend_from_slice(&1_u32.to_le_bytes());
        header.extend_from_slice(&2_u64.to_le_bytes());
        header.extend_from_slice(&0_u32.to_le_bytes());
        header.extend_from_slice(&0_u64.to_le_bytes());
        let data_start = usize::try_from(align_up(u64::try_from(header.len())?, 32)?)?;
        header.resize(data_start, 0);
        header.extend_from_slice(&1.5_f32.to_le_bytes());
        header.extend_from_slice(&(-2.0_f32).to_le_bytes());
        header.resize(data_start + 32, 0);

        let response = execute(&json!({
            "SchemaVersion": 1,
            "Operations": [{
                "Kind": "gguf-read",
                "DataHex": hex(&header),
                "ReadMode": "callback",
                "LoadTensors": true,
                "TensorDataIndices": [0]
            }]
        }))?;
        let snap = &response["Results"][0];
        assert_eq!(snap["ContextExists"], true);
        assert_eq!(snap["GgmlContextExists"], true);
        assert_eq!(snap["Header"]["Version"], 2);
        assert_eq!(snap["Header"]["Keys"][0]["DataHex"], "01");
        assert_eq!(
            snap["ContextTensors"][0]["Name"],
            "GGUF tensor data binary blob"
        );
        assert_eq!(snap["ContextTensors"][1]["Name"], "w0");
        assert_eq!(
            snap["ContextTensors"][1]["DataHex"],
            hex(&[1.5_f32.to_le_bytes(), (-2.0_f32).to_le_bytes()].concat())
        );

        assert!(
            execute(&json!({
                "SchemaVersion": 2,
                "Operations": [{"Kind": "gguf-inventory"}]
            }))
            .is_err()
        );
        let truncated = execute(&json!({
            "SchemaVersion": 1,
            "Operations": [{
                "Kind": "gguf-read",
                "DataHex": "00",
                "ReadMode": "file",
                "LoadTensors": false,
                "TensorDataIndices": []
            }]
        }))?;
        assert_eq!(truncated["Results"][0]["ContextExists"], false);
        assert!(
            execute(&json!({
                "SchemaVersion": 1,
                "Operations": [{
                    "Kind": "gguf-read",
                    "DataHex": "0g",
                    "ReadMode": "file",
                    "LoadTensors": false,
                    "TensorDataIndices": []
                }]
            }))
            .is_err()
        );
        assert!(
            execute(&json!({
                "SchemaVersion": 1,
                "Operations": [{
                    "Kind": "gguf-context-trace",
                    "Backend": 1,
                    "Contexts": [null],
                    "Actions": [{"Op": "snapshot", "Context": 0, "TensorDataIndices": []}]
                }]
            }))
            .is_err()
        );
        Ok(())
    }
}
