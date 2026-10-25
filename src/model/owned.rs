//! Generated model storage and streaming, lossless GGUF v3 serialization.
//!
//! The callback and file-progress split follows pinned Prism `b87ca722`,
//! `src/llama-model-loader.cpp`: generated tensors receive caller data directly;
//! file tensors use byte-weighted progress and cancellation. This module owns
//! real tensor payloads, but does not claim architecture graph execution.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::sync::Arc;

use bonsai_gguf::{Metadata, MetadataArray, MetadataValue};
use sha2::{Digest, Sha256};

use super::{Model, Tensor, gguf_context, validate_payload};
use crate::{Result, invalid};

const HEADER_LIMIT: u64 = 256 * 1024 * 1024;
const STRING_LIMIT: usize = 8 * 1024 * 1024;
const ARRAY_LIMIT: usize = 2 * 1024 * 1024;
const ELEMENT_LIMIT: usize = 8 * 1024 * 1024;

/// One required generated tensor, in caller-specified callback order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorSpec {
    /// Unique, nonempty GGUF tensor name, at most 1024 UTF-8 bytes.
    pub name: String,
    /// Pinned type ID with an implemented production row decoder.
    pub kind: u32,
    /// One to four nonzero extents, with contiguous dimension first.
    pub dimensions: Vec<usize>,
}

pub(super) const fn supported_kind(kind: u32) -> bool {
    matches!(kind, 0..=3 | 6..=23 | 29 | 30 | 34 | 35 | 39..=42 | 142 | 143)
}

fn overflow() -> crate::Error {
    invalid("owned model size overflow")
}

fn add(total: &mut usize, bytes: usize) -> Result<()> {
    *total = total.checked_add(bytes).ok_or_else(overflow)?;
    Ok(())
}

fn tensor_bytes(name: &str, kind: u32, dimensions: &[usize]) -> Result<usize> {
    if name.is_empty()
        || name.len() > 1024
        || !supported_kind(kind)
        || !(1..=4).contains(&dimensions.len())
        || dimensions.contains(&0)
    {
        return Err(invalid("invalid generated model tensor descriptor"));
    }
    let mut shape = [1_u64; 4];
    for (out, &size) in shape.iter_mut().zip(dimensions) {
        *out = u64::try_from(size)?;
    }
    Ok(usize::try_from(gguf_context::tensor_bytes(kind, shape)?)?)
}

fn alignment(metadata: &Metadata) -> Result<u64> {
    match metadata.get("general.alignment") {
        None => Ok(32),
        Some(MetadataValue::U32(value)) if value.is_power_of_two() && *value <= 1_048_576 => {
            Ok(u64::from(*value))
        }
        _ => Err(invalid("invalid owned model GGUF alignment")),
    }
}

fn padded(value: u64, alignment: u64) -> Result<u64> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(overflow)
}

impl Model {
    /// Builds independently owned tensors using real caller-supplied bytes.
    ///
    /// Every descriptor, metadata value and cumulative retained-storage budget
    /// is validated before the first callback. Callbacks run exactly once in
    /// descriptor order and fill a zero-initialized, exact-sized byte slice.
    /// Any callback or payload-validation error drops the entire partial model.
    /// Cancellation can be returned as a callback error. As in the pinned
    /// generated loader, this path does not invoke file-progress callbacks.
    ///
    /// The identity is the SHA-256 of this model's canonical GGUF serialization.
    /// `memory_limit` bounds conservative retained storage, excluding the
    /// caller's descriptor list, fixed writer/decoder scratch and one transient
    /// tensor copy during conversion to shared ownership. Metadata is moved in.
    /// No architecture-specific tensor inventory or graph validation is implied.
    ///
    /// # Errors
    /// Rejects malformed metadata/descriptors, duplicate tensor names, missing
    /// production decoders, over-budget storage, invalid/nonfinite payloads,
    /// allocation failures and callback errors.
    pub fn generate(
        metadata: Metadata,
        specifications: &[TensorSpec],
        memory_limit: usize,
        mut tensor_data: impl FnMut(&TensorSpec, &mut [u8]) -> Result<()>,
    ) -> Result<Self> {
        if specifications.len() > 100_000 {
            return Err(invalid("generated tensor count exceeds model bound"));
        }
        let mut model = Self {
            metadata,
            tensors: BTreeMap::new(),
            identity: String::new(),
        };
        let mut required = metadata_charge(&model.metadata)?;
        let mut header = CountWriter(0);
        model.write_header(&mut header)?;
        let mut names = BTreeSet::new();
        for spec in specifications {
            if !names.insert(spec.name.as_str()) {
                return Err(invalid("duplicate generated model tensor name"));
            }
            let bytes = tensor_bytes(&spec.name, spec.kind, &spec.dimensions)?;
            header.0 = header
                .0
                .checked_add(u64::try_from(
                    24 + spec.name.len() + 8 * spec.dimensions.len(),
                )?)
                .ok_or_else(overflow)?;
            add(&mut required, bytes)?;
            add(
                &mut required,
                spec.name.len().checked_mul(2).ok_or_else(overflow)?,
            )?;
            add(
                &mut required,
                256 + spec.dimensions.len() * size_of::<usize>(),
            )?;
        }
        add(&mut required, 64)?;
        if padded(header.0, alignment(&model.metadata)?)? > HEADER_LIMIT {
            return Err(invalid("generated model header exceeds byte bound"));
        }
        if required > memory_limit {
            return Err(invalid("generated model exceeds retained-storage budget"));
        }
        for spec in specifications {
            let bytes = tensor_bytes(&spec.name, spec.kind, &spec.dimensions)?;
            let mut data = Vec::new();
            data.try_reserve_exact(bytes)?;
            data.resize(bytes, 0);
            tensor_data(spec, &mut data)?;
            validate_payload(spec.kind, &data)?;
            model.tensors.insert(
                spec.name.clone(),
                Tensor {
                    name: spec.name.clone(),
                    dimensions: spec.dimensions.clone(),
                    kind: spec.kind,
                    data: Arc::from(data),
                },
            );
        }
        let mut output = HashWriter(Sha256::new());
        model.write_to(&mut output)?;
        model.identity = output
            .0
            .finalize()
            .into_iter()
            .flat_map(|byte| [byte >> 4, byte & 15])
            .map(|nibble| char::from(b"0123456789abcdef"[usize::from(nibble)]))
            .collect();
        Ok(model)
    }

    /// Streams complete GGUF v3 metadata and tensor bytes to an owned output.
    ///
    /// All metadata scalar/array types, nested arrays, tensor dimensions and
    /// payload bytes are retained. Keys/tensors use deterministic map order;
    /// alignment padding is zero-filled. The output starts at the stream's
    /// current position, supporting an already owned file or file pointer.
    /// Serialization does not materialize another complete model or payload.
    /// Metadata and payload validation finish before the first output write.
    /// An I/O error can leave a partial output; the caller owns its disposition.
    ///
    /// # Errors
    /// Rejects malformed public model fields, unsupported encodings, nonfinite
    /// metadata/payloads, overflow, header bounds and output I/O errors.
    pub fn write_to(&self, output: &mut impl Write) -> Result<u64> {
        metadata_charge(&self.metadata)?;
        if self.tensors.len() > 100_000 {
            return Err(invalid("owned model tensor count exceeds bound"));
        }
        for (key, tensor) in &self.tensors {
            if *key != tensor.name
                || tensor_bytes(&tensor.name, tensor.kind, &tensor.dimensions)? != tensor.data.len()
            {
                return Err(invalid("owned model tensor bytes or map identity differ"));
            }
            validate_payload(tensor.kind, &tensor.data)?;
        }
        let align = alignment(&self.metadata)?;
        let mut count = CountWriter(0);
        self.write_header(&mut count)?;
        let header = padded(count.0, align)?;
        if header > HEADER_LIMIT {
            return Err(invalid("owned model header exceeds byte bound"));
        }
        let mut total = header;
        for tensor in self.tensors.values() {
            total = total
                .checked_add(padded(u64::try_from(tensor.data.len())?, align)?)
                .ok_or_else(overflow)?;
        }
        self.write_header(output)?;
        padding(output, header - count.0)?;
        for tensor in self.tensors.values() {
            output.write_all(&tensor.data)?;
            let bytes = u64::try_from(tensor.data.len())?;
            padding(output, padded(bytes, align)? - bytes)?;
        }
        Ok(total)
    }

    fn write_header(&self, output: &mut impl Write) -> Result<()> {
        output.write_all(b"GGUF")?;
        output.write_all(&3_u32.to_le_bytes())?;
        output.write_all(&u64::try_from(self.tensors.len())?.to_le_bytes())?;
        output.write_all(&u64::try_from(self.metadata.len())?.to_le_bytes())?;
        let mut writer = MetadataWriter {
            output,
            elements: 0,
            charge: 0,
        };
        for (key, value) in &self.metadata {
            writer.string(key)?;
            writer.output.write_all(&value_kind(value).to_le_bytes())?;
            writer.value(value, 0)?;
        }
        let align = alignment(&self.metadata)?;
        let mut offset = 0_u64;
        for tensor in self.tensors.values() {
            writer.string(&tensor.name)?;
            writer
                .output
                .write_all(&u32::try_from(tensor.dimensions.len())?.to_le_bytes())?;
            for &dimension in &tensor.dimensions {
                writer
                    .output
                    .write_all(&u64::try_from(dimension)?.to_le_bytes())?;
            }
            writer.output.write_all(&tensor.kind.to_le_bytes())?;
            writer.output.write_all(&offset.to_le_bytes())?;
            offset = offset
                .checked_add(padded(u64::try_from(tensor.data.len())?, align)?)
                .ok_or_else(overflow)?;
        }
        Ok(())
    }
}

fn padding(output: &mut impl Write, mut bytes: u64) -> Result<()> {
    let zeros = [0_u8; 4096];
    while bytes != 0 {
        let count = usize::try_from(bytes.min(4096))?;
        output.write_all(&zeros[..count])?;
        bytes -= u64::try_from(count)?;
    }
    Ok(())
}

struct CountWriter(u64);

impl Write for CountWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0 = self
            .0
            .checked_add(u64::try_from(bytes.len()).map_err(std::io::Error::other)?)
            .ok_or_else(|| std::io::Error::other("GGUF output length overflow"))?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

struct HashWriter(Sha256);

impl Write for HashWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

const fn value_kind(value: &MetadataValue) -> u32 {
    match value {
        MetadataValue::U8(_) => 0,
        MetadataValue::I8(_) => 1,
        MetadataValue::U16(_) => 2,
        MetadataValue::I16(_) => 3,
        MetadataValue::U32(_) => 4,
        MetadataValue::I32(_) => 5,
        MetadataValue::F32(_) => 6,
        MetadataValue::Bool(_) => 7,
        MetadataValue::String(_) => 8,
        MetadataValue::Array(_) => 9,
        MetadataValue::U64(_) => 10,
        MetadataValue::I64(_) => 11,
        MetadataValue::F64(_) => 12,
    }
}

fn metadata_charge(metadata: &Metadata) -> Result<usize> {
    if metadata.len() > 10_000
        || metadata
            .get("general.architecture")
            .and_then(MetadataValue::as_str)
            .is_none_or(str::is_empty)
    {
        return Err(invalid(
            "owned model architecture or metadata count is invalid",
        ));
    }
    alignment(metadata)?;
    let mut count = CountWriter(0);
    let mut writer = MetadataWriter {
        output: &mut count,
        elements: 0,
        charge: 0,
    };
    for (key, value) in metadata {
        if key.is_empty() || key.len() > 1024 {
            return Err(invalid("owned model metadata key is invalid"));
        }
        if key.starts_with("tokenizer.")
            && key.ends_with("_token_id")
            && !matches!(value_kind(value), 4 | 5 | 10 | 11)
        {
            return Err(invalid("owned model special token ID must be an integer"));
        }
        add(
            &mut writer.charge,
            key.capacity().checked_add(256).ok_or_else(overflow)?,
        )?;
        writer.string(key)?;
        writer.value(value, 0)?;
    }
    let charge = writer.charge;
    if count.0 > HEADER_LIMIT {
        return Err(invalid("owned model metadata exceeds header bound"));
    }
    Ok(charge)
}

struct MetadataWriter<'a, W> {
    output: &'a mut W,
    elements: usize,
    charge: usize,
}

impl<W: Write> MetadataWriter<'_, W> {
    fn string(&mut self, value: &str) -> Result<()> {
        if value.len() > STRING_LIMIT {
            return Err(invalid("owned model metadata string exceeds byte bound"));
        }
        self.output
            .write_all(&u64::try_from(value.len())?.to_le_bytes())?;
        self.output.write_all(value.as_bytes())?;
        Ok(())
    }

    fn value(&mut self, value: &MetadataValue, depth: u32) -> Result<()> {
        macro_rules! scalar {
            ($value:expr) => {
                self.output.write_all(&$value.to_le_bytes())?
            };
        }
        match value {
            MetadataValue::U8(value) => scalar!(value),
            MetadataValue::I8(value) => scalar!(value),
            MetadataValue::U16(value) => scalar!(value),
            MetadataValue::I16(value) => scalar!(value),
            MetadataValue::U32(value) => scalar!(value),
            MetadataValue::I32(value) => scalar!(value),
            MetadataValue::F32(value) => {
                if !value.is_finite() {
                    return Err(invalid("nonfinite owned model metadata"));
                }
                scalar!(value);
            }
            MetadataValue::Bool(value) => self.output.write_all(&[u8::from(*value)])?,
            MetadataValue::String(value) => {
                add(&mut self.charge, value.capacity())?;
                self.string(value)?;
            }
            MetadataValue::U64(value) => scalar!(value),
            MetadataValue::I64(value) => scalar!(value),
            MetadataValue::F64(value) => {
                if !value.is_finite() {
                    return Err(invalid("nonfinite owned model metadata"));
                }
                scalar!(value);
            }
            MetadataValue::Array(value) => self.array(value, depth)?,
        }
        Ok(())
    }

    fn array(&mut self, array: &MetadataArray, depth: u32) -> Result<()> {
        let count = array.len();
        if count > ARRAY_LIMIT || (count > 0 && depth >= 4) {
            return Err(invalid(
                "owned model metadata array count or depth exceeds bound",
            ));
        }
        add(&mut self.elements, count)?;
        if self.elements > ELEMENT_LIMIT {
            return Err(invalid(
                "owned model metadata total element count exceeds bound",
            ));
        }
        macro_rules! values {
            ($values:expr, $kind:expr, $variant:ident) => {{
                add(
                    &mut self.charge,
                    $values
                        .capacity()
                        .checked_mul(size_of_val_type($values))
                        .ok_or_else(overflow)?,
                )?;
                self.output.write_all(&$kind.to_le_bytes())?;
                self.output
                    .write_all(&u64::try_from(count)?.to_le_bytes())?;
                for value in $values {
                    self.value(&MetadataValue::$variant(*value), depth + 1)?;
                }
            }};
        }
        match array {
            MetadataArray::U8(values) => values!(values, 0_u32, U8),
            MetadataArray::I8(values) => values!(values, 1_u32, I8),
            MetadataArray::U16(values) => values!(values, 2_u32, U16),
            MetadataArray::I16(values) => values!(values, 3_u32, I16),
            MetadataArray::U32(values) => values!(values, 4_u32, U32),
            MetadataArray::I32(values) => values!(values, 5_u32, I32),
            MetadataArray::F32(values) => values!(values, 6_u32, F32),
            MetadataArray::Bool(values) => values!(values, 7_u32, Bool),
            MetadataArray::U64(values) => values!(values, 10_u32, U64),
            MetadataArray::I64(values) => values!(values, 11_u32, I64),
            MetadataArray::F64(values) => values!(values, 12_u32, F64),
            MetadataArray::String(values) => {
                add(
                    &mut self.charge,
                    values
                        .capacity()
                        .checked_mul(size_of::<String>())
                        .ok_or_else(overflow)?,
                )?;
                self.output.write_all(&8_u32.to_le_bytes())?;
                self.output
                    .write_all(&u64::try_from(count)?.to_le_bytes())?;
                for value in values {
                    add(&mut self.charge, value.capacity())?;
                    self.string(value)?;
                }
            }
            MetadataArray::Array(values) => {
                add(
                    &mut self.charge,
                    values
                        .capacity()
                        .checked_mul(size_of::<MetadataArray>())
                        .ok_or_else(overflow)?,
                )?;
                self.output.write_all(&9_u32.to_le_bytes())?;
                self.output
                    .write_all(&u64::try_from(count)?.to_le_bytes())?;
                for value in values {
                    self.array(value, depth + 1)?;
                }
            }
        }
        Ok(())
    }
}

const fn size_of_val_type<T>(_: &Vec<T>) -> usize {
    size_of::<T>()
}
