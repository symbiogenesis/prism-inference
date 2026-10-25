//! Ordered, mutable GGUF contexts with exact scalar and tensor byte ownership.
//!
//! Implements the pinned Prism GGUF v2/v3 format and context setter/copy/write
//! semantics from `ggml/src/gguf.cpp`. Metadata strings remain byte strings;
//! this container API does not reinterpret model weights or require a tokenizer.

use std::collections::BTreeSet;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::{Result, invalid};

const MAX_BYTES: usize = 64 * 1024 * 1024;
const MAX_ITEMS: usize = 65536;

/// Exact ordered GGUF metadata value, retaining its scalar/array wire type.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Entry {
    /// Metadata key, with no embedded zero byte.
    pub key: String,
    /// GGUF scalar type, or ARRAY (9).
    pub kind: u32,
    /// Scalar element type for arrays.
    pub array_kind: Option<u32>,
    /// One for scalars, otherwise the number of array elements.
    pub count: usize,
    /// Exact little-endian numeric bytes; empty for strings.
    pub data: Vec<u8>,
    /// Exact string bytes; one for a scalar, otherwise one per array element.
    pub strings: Vec<Vec<u8>>,
}

/// Immutable tensor descriptor and optionally retained serialized payload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Tensor {
    /// Native tensor name, shorter than 64 bytes.
    pub name: String,
    /// Pinned native GGML type ID.
    pub kind: u32,
    /// Four native dimensions, contiguous dimension first.
    pub dimensions: [u64; 4],
    /// Byte offset relative to the start of the data section.
    pub offset: u64,
    /// Number of serialized tensor bytes, excluding alignment padding.
    pub bytes: u64,
    /// Independent tensor payload, when loaded or constructed.
    pub data: Option<Vec<u8>>,
}

/// Ordered GGUF metadata and independently owned tensor storage.
#[derive(Clone, Debug)]
pub struct Context {
    version: u32,
    alignment: usize,
    data_offset: u64,
    entries: Vec<Entry>,
    tensors: Vec<Tensor>,
    loaded: bool,
    blob: Option<Vec<u8>>,
}

/// Pinned active GGML block dimensions and serialized byte width.
#[must_use]
pub const fn tensor_layout(kind: u32) -> Option<(u64, u64)> {
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
        _ => Err(invalid("GGUF scalar type has no numeric byte width")),
    }
}

fn padded(value: u64, alignment: usize) -> Result<u64> {
    let alignment = u64::try_from(alignment)?;
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| invalid("GGUF alignment overflow"))
}

/// Computes a native tensor's serialized extent with checked shape arithmetic.
///
/// # Errors
/// Rejects removed types, negative-as-unsigned dimensions, incomplete block
/// rows and element/byte counts that exceed native signed/unsigned capacities.
pub fn tensor_bytes(kind: u32, dimensions: [u64; 4]) -> Result<u64> {
    let (block, width) =
        tensor_layout(kind).ok_or_else(|| invalid("GGUF tensor type is removed or unknown"))?;
    if dimensions
        .iter()
        .any(|&dimension| dimension > i64::MAX.unsigned_abs())
        || !dimensions[0].is_multiple_of(block)
    {
        return Err(invalid(
            "GGUF tensor dimensions or block alignment are invalid",
        ));
    }
    if dimensions.contains(&0) {
        return Ok(0);
    }
    let elements = dimensions
        .iter()
        .try_fold(1_u64, |count, &dimension| count.checked_mul(dimension))
        .filter(|&count| count < i64::MAX.unsigned_abs())
        .ok_or_else(|| invalid("GGUF tensor element count overflow"))?;
    (elements / block)
        .checked_mul(width)
        .ok_or_else(|| invalid("GGUF tensor byte count overflow"))
}

impl Entry {
    fn validate(&self, setter: bool) -> Result<()> {
        if self.key.is_empty()
            || self.key.len() > MAX_BYTES
            || self.key.contains('\0')
            || self.count > MAX_BYTES
        {
            return Err(invalid("GGUF metadata key or count outside bound"));
        }
        let element = if self.kind == 9 {
            self.array_kind
                .filter(|kind| *kind != 9)
                .ok_or_else(|| invalid("GGUF array requires a scalar element type"))?
        } else {
            if self.array_kind.is_some() || self.count != 1 {
                return Err(invalid("GGUF scalar descriptor mismatch"));
            }
            self.kind
        };
        if element == 8 {
            if !self.data.is_empty()
                || self.strings.len() != self.count
                || self
                    .strings
                    .iter()
                    .any(|string| string.len() > MAX_BYTES || (setter && string.contains(&0)))
            {
                return Err(invalid("GGUF string descriptor extent mismatch"));
            }
        } else {
            let width = scalar_width(element)?;
            if !self.strings.is_empty()
                || self.count.checked_mul(width) != Some(self.data.len())
                || self.data.len() > MAX_BYTES
            {
                return Err(invalid("GGUF numeric descriptor extent mismatch"));
            }
            if setter && element == 7 && self.data.iter().any(|&byte| byte > 1) {
                return Err(invalid("GGUF boolean setter needs zero or one"));
            }
        }
        if self.key == "general.alignment" {
            if self.kind != 4 || self.data.len() != 4 {
                return Err(invalid("GGUF alignment must be UINT32"));
            }
            let value = u32::from_le_bytes(self.data[..4].try_into()?);
            if !value.is_power_of_two() {
                return Err(invalid("GGUF alignment must be a positive power of two"));
            }
        }
        Ok(())
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl Context {
    /// Creates an empty native v3 context with 32-byte tensor alignment.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            version: 3,
            alignment: 32,
            data_offset: 0,
            entries: Vec::new(),
            tensors: Vec::new(),
            loaded: false,
            blob: None,
        }
    }
    /// Parsed or constructed GGUF version.
    #[must_use]
    pub const fn version(&self) -> u32 {
        self.version
    }
    /// Active native data-section alignment.
    #[must_use]
    pub const fn alignment(&self) -> usize {
        self.alignment
    }
    /// Parsed data-section file offset; zero for a newly constructed context.
    #[must_use]
    pub const fn data_offset(&self) -> u64 {
        self.data_offset
    }
    /// Metadata in native encounter/setter order.
    #[must_use]
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }
    /// Tensor descriptors in encounter order.
    #[must_use]
    pub fn tensors(&self) -> &[Tensor] {
        &self.tensors
    }
    /// Whether this owner contains a constructed or loaded tensor context.
    #[must_use]
    pub const fn has_tensor_context(&self) -> bool {
        self.loaded
    }
    /// Complete retained binary blob for a parsed context loaded with tensors.
    #[must_use]
    pub fn blob(&self) -> Option<&[u8]> {
        self.blob.as_deref()
    }

    /// Replaces a key by removing its old entry and appending the new value.
    ///
    /// # Errors
    /// Rejects invalid scalar/array extents, strings containing zero, invalid
    /// reserved alignment metadata, collection bounds or allocation failures.
    pub fn set_entry(&mut self, entry: Entry) -> Result<()> {
        entry.validate(true)?;
        if self.entries.len() >= MAX_ITEMS
            && !self.entries.iter().any(|value| value.key == entry.key)
        {
            return Err(invalid("GGUF metadata entry count exceeds bound"));
        }
        self.entries.retain(|value| value.key != entry.key);
        self.entries.push(entry);
        Ok(())
    }

    /// Copies all source metadata through native replacement/append semantics.
    ///
    /// # Errors
    /// Returns setter validation or allocation errors. Tensor storage is retained.
    pub fn copy_metadata(&mut self, source: &Self) -> Result<()> {
        for entry in &source.entries {
            let mut entry = entry.clone();
            if entry.kind == 8 || entry.array_kind == Some(8) {
                for string in &mut entry.strings {
                    if let Some(end) = string.iter().position(|&byte| byte == 0) {
                        string.truncate(end);
                    }
                }
            }
            if entry.kind == 7 {
                entry
                    .data
                    .iter_mut()
                    .for_each(|byte| *byte = u8::from(*byte != 0));
            }
            self.set_entry(entry)?;
        }
        Ok(())
    }

    /// Adds a contiguous tensor with exact independently owned payload bytes.
    ///
    /// # Errors
    /// Rejects duplicate/overlong names, malformed shapes, unknown types,
    /// byte-length mismatches and data/descriptor resource bounds.
    pub fn add_tensor(
        &mut self,
        name: String,
        kind: u32,
        dimensions: [u64; 4],
        data: Vec<u8>,
    ) -> Result<()> {
        let bytes = tensor_bytes(kind, dimensions)?;
        if name.len() >= 64
            || name.contains('\0')
            || self.tensors.iter().any(|tensor| tensor.name == name)
            || self.tensors.len() >= MAX_ITEMS
            || usize::try_from(bytes)? != data.len()
            || data.len() > MAX_BYTES
        {
            return Err(invalid("GGUF tensor descriptor or payload is invalid"));
        }
        let offset = if let Some(previous) = self.tensors.last() {
            previous
                .offset
                .checked_add(padded(previous.bytes, self.alignment)?)
                .ok_or_else(|| invalid("GGUF tensor offset overflow"))?
        } else {
            0
        };
        self.tensors.push(Tensor {
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

    /// Reads GGUF header and optionally complete tensor storage from a buffer.
    ///
    /// # Errors
    /// Rejects malformed headers, duplicate keys/names, invalid alignment,
    /// shapes, types, offsets, truncation, overflow and bounded resource excess.
    pub fn read_buffer(bytes: &[u8], load_tensors: bool) -> Result<Self> {
        Self::read_callback(
            u64::try_from(bytes.len())?,
            load_tensors,
            |offset, output| {
                let offset = usize::try_from(offset)?;
                let available = bytes
                    .get(offset..)
                    .ok_or_else(|| invalid("GGUF buffer offset exceeds extent"))?;
                let count = available.len().min(output.len());
                output[..count].copy_from_slice(&available[..count]);
                Ok(count)
            },
        )
    }

    /// Reads through a file-backed offset source without changing process state.
    ///
    /// # Errors
    /// Returns file I/O errors and the same format/resource errors as `read_buffer`.
    pub fn read_file(path: &Path, load_tensors: bool) -> Result<Self> {
        let mut file = File::open(path)?;
        let length = file.metadata()?.len();
        Self::read_callback(length, load_tensors, |offset, output| {
            file.seek(SeekFrom::Start(offset))?;
            Ok(file.read(output)?)
        })
    }

    /// Reads from an offset callback, retaining metadata strings as raw bytes.
    ///
    /// A short callback read denotes truncation, matching the native reader.
    /// The callback is asked for at most 4096 bytes, including tensor data.
    ///
    /// # Errors
    /// Returns callback errors and the same format/resource errors as `read_buffer`.
    pub fn read_callback(
        length: u64,
        load_tensors: bool,
        read: impl FnMut(u64, &mut [u8]) -> Result<usize>,
    ) -> Result<Self> {
        parse(
            Reader {
                offset: 0,
                length,
                read,
            },
            load_tensors,
        )
    }

    /// Serializes ordered metadata/descriptors and optionally all tensor bytes.
    ///
    /// # Errors
    /// Rejects missing tensor storage, malformed extents and outputs above 64 MiB.
    pub fn encode(&self, metadata_only: bool) -> Result<Vec<u8>> {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(self.version.to_le_bytes());
        bytes.extend(u64::try_from(self.tensors.len())?.to_le_bytes());
        bytes.extend(u64::try_from(self.entries.len())?.to_le_bytes());
        for entry in &self.entries {
            put_string(&mut bytes, entry.key.as_bytes())?;
            bytes.extend(entry.kind.to_le_bytes());
            if let Some(kind) = entry.array_kind {
                bytes.extend(kind.to_le_bytes());
                bytes.extend(u64::try_from(entry.count)?.to_le_bytes());
            }
            if entry.kind == 8 || entry.array_kind == Some(8) {
                for string in &entry.strings {
                    put_string(&mut bytes, string)?;
                }
            } else {
                bytes.extend(&entry.data);
            }
        }
        for tensor in &self.tensors {
            put_string(&mut bytes, tensor.name.as_bytes())?;
            let dimensions = tensor
                .dimensions
                .iter()
                .rposition(|&dimension| dimension != 1)
                .map_or(1, |index| index + 1);
            bytes.extend(u32::try_from(dimensions)?.to_le_bytes());
            for dimension in &tensor.dimensions[..dimensions] {
                bytes.extend(dimension.to_le_bytes());
            }
            bytes.extend(tensor.kind.to_le_bytes());
            bytes.extend(tensor.offset.to_le_bytes());
        }
        let header = usize::try_from(padded(u64::try_from(bytes.len())?, self.alignment)?)?;
        if header > MAX_BYTES {
            return Err(invalid("GGUF metadata exceeds byte bound"));
        }
        bytes.resize(header, 0);
        if !metadata_only {
            for tensor in &self.tensors {
                let data = tensor
                    .data
                    .as_ref()
                    .ok_or_else(|| invalid("GGUF write requires loaded tensor bytes"))?;
                if bytes.len().checked_sub(header) != Some(usize::try_from(tensor.offset)?)
                    || usize::try_from(tensor.bytes)? != data.len()
                {
                    return Err(invalid("GGUF write tensor offsets or bytes differ"));
                }
                let end = bytes
                    .len()
                    .checked_add(usize::try_from(padded(tensor.bytes, self.alignment)?)?)
                    .filter(|&end| end <= MAX_BYTES)
                    .ok_or_else(|| invalid("GGUF encoded data exceeds byte bound"))?;
                bytes.extend(data);
                bytes.resize(end, 0);
            }
        }
        Ok(bytes)
    }

    /// Writes serialized GGUF bytes to an already owned output stream.
    ///
    /// # Errors
    /// Returns encoding and output stream I/O errors.
    pub fn write_to(&self, output: &mut impl Write, metadata_only: bool) -> Result<()> {
        output.write_all(&self.encode(metadata_only)?)?;
        Ok(())
    }
}

fn put_string(output: &mut Vec<u8>, string: &[u8]) -> Result<()> {
    if output
        .len()
        .checked_add(string.len())
        .is_none_or(|end| end > MAX_BYTES)
    {
        return Err(invalid("GGUF serialized metadata exceeds bound"));
    }
    output.extend(u64::try_from(string.len())?.to_le_bytes());
    output.extend(string);
    Ok(())
}

struct Reader<F> {
    offset: u64,
    length: u64,
    read: F,
}
impl<F: FnMut(u64, &mut [u8]) -> Result<usize>> Reader<F> {
    fn take(&mut self, count: usize) -> Result<Vec<u8>> {
        if count > MAX_BYTES
            || self
                .offset
                .checked_add(u64::try_from(count)?)
                .is_none_or(|end| end > self.length)
        {
            return Err(invalid("GGUF input extent exceeds source"));
        }
        let mut bytes = vec![0; count];
        let mut consumed = 0;
        while consumed < count {
            let end = (consumed + 4096).min(count);
            let amount = (self.read)(self.offset, &mut bytes[consumed..end])?;
            if amount != end - consumed {
                return Err(invalid(
                    "GGUF callback returned a truncated or invalid read",
                ));
            }
            self.offset += u64::try_from(amount)?;
            consumed += amount;
        }
        Ok(bytes)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.as_slice().try_into()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.as_slice().try_into()?))
    }
    fn string(&mut self) -> Result<Vec<u8>> {
        let count = usize::try_from(self.u64()?)?;
        self.take(count)
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "Keep the native GGUF header, key, tensor and blob transaction together so failures cannot expose partial contexts."
)]
fn parse<F: FnMut(u64, &mut [u8]) -> Result<usize>>(
    mut reader: Reader<F>,
    load: bool,
) -> Result<Context> {
    if reader.take(4)? != b"GGUF" {
        return Err(invalid("GGUF magic mismatch"));
    }
    let version = reader.u32()?;
    if !matches!(version, 2 | 3) {
        return Err(invalid("unsupported GGUF version"));
    }
    let tensor_count = usize::try_from(reader.u64()?)?;
    let key_count = usize::try_from(reader.u64()?)?;
    if tensor_count > MAX_ITEMS || key_count > MAX_ITEMS {
        return Err(invalid("GGUF header collection count exceeds bound"));
    }
    let mut context = Context {
        version,
        ..Context::new()
    };
    let mut keys = BTreeSet::new();
    for _ in 0..key_count {
        let key = String::from_utf8(reader.string()?)?;
        if !keys.insert(key.clone()) {
            return Err(invalid("duplicate GGUF metadata key"));
        }
        let kind = reader.u32()?;
        let array_kind = if kind == 9 { Some(reader.u32()?) } else { None };
        let count = if kind == 9 {
            usize::try_from(reader.u64()?)?
        } else {
            1
        };
        if count > MAX_BYTES {
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
                strings.push(reader.string()?);
            }
        } else {
            data = reader.take(
                count
                    .checked_mul(scalar_width(element)?)
                    .ok_or_else(|| invalid("GGUF metadata array byte count overflow"))?,
            )?;
            if element == 7 {
                for byte in &mut data {
                    *byte = u8::from(*byte != 0);
                }
            }
        }
        let entry = Entry {
            key,
            kind,
            array_kind,
            count,
            data,
            strings,
        };
        entry.validate(false)?;
        if entry.key == "general.alignment" {
            context.alignment = usize::try_from(u32::from_le_bytes(entry.data[..4].try_into()?))?;
        }
        context.entries.push(entry);
    }
    let mut names = BTreeSet::new();
    let mut total = 0_u64;
    for _ in 0..tensor_count {
        let raw_name = reader.string()?;
        if raw_name.len() >= 64 {
            return Err(invalid("GGUF tensor name is too long"));
        }
        let end = raw_name
            .iter()
            .position(|&byte| byte == 0)
            .unwrap_or(raw_name.len());
        let name = String::from_utf8(raw_name[..end].to_vec())?;
        if !names.insert(name.clone()) {
            return Err(invalid("duplicate GGUF tensor name"));
        }
        let rank = usize::try_from(reader.u32()?)?;
        if rank > 4 {
            return Err(invalid("GGUF tensor rank exceeds four"));
        }
        let mut dimensions = [1; 4];
        for dimension in &mut dimensions[..rank] {
            *dimension = reader.u64()?;
        }
        let kind = reader.u32()?;
        let bytes = tensor_bytes(kind, dimensions)?;
        let offset = reader.u64()?;
        if offset != total {
            return Err(invalid(
                "GGUF tensor offsets do not form the native contiguous layout",
            ));
        }
        total = total
            .checked_add(padded(bytes, context.alignment)?)
            .ok_or_else(|| invalid("GGUF tensor data size overflow"))?;
        context.tensors.push(Tensor {
            name,
            kind,
            dimensions,
            offset,
            bytes,
            data: None,
        });
    }
    if tensor_count > 0 {
        let start = padded(reader.offset, context.alignment)?;
        if start > reader.length {
            return Err(invalid("GGUF data alignment exceeds source extent"));
        }
        reader.offset = start;
    }
    context.data_offset = reader.offset;
    if load {
        let blob = reader.take(usize::try_from(total)?)?;
        for tensor in &mut context.tensors {
            let start = usize::try_from(tensor.offset)?;
            let end = start
                .checked_add(usize::try_from(tensor.bytes)?)
                .ok_or_else(|| invalid("GGUF tensor data extent overflow"))?;
            tensor.data = Some(
                blob.get(start..end)
                    .ok_or_else(|| invalid("GGUF tensor data exceeds blob"))?
                    .to_vec(),
            );
        }
        context.loaded = true;
        context.blob = Some(blob);
    }
    Ok(context)
}
