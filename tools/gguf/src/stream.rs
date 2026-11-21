//! Bounded, policy-driven, cancellable `Read + Seek` GGUF v2/v3 header and tensor-index parser.
//!
//! Intended for callers that already own and verify their stream or mapped handle,
//! and need configurable bounds, selective array retention, and cooperative cancellation.

use std::collections::BTreeMap;
use std::io::{self, BufReader, Read, Seek, SeekFrom};

use crate::inspect::{scalar_width, stock_tensor_layout};
use crate::{MetadataArray, MetadataValue};

const MIN_HEADER_BYTES: u64 = 24;
const DEFAULT_MAX_FILE_BYTES: u64 = 16 * 1024 * 1024 * 1024;
const DEFAULT_MAX_ITEMS: u64 = 131_072;
const DEFAULT_METADATA_BUDGET: usize = 128 * 1024 * 1024;
const DEFAULT_MAX_KEY_BYTES: usize = 65_536;
const DEFAULT_MAX_STRING_BYTES: usize = 16 * 1024 * 1024;
const DEFAULT_MAX_RETAINED_STRING_ARRAY_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_MAX_ARRAY_ITEMS: u64 = 1_000_000;
const DEFAULT_MAX_ALIGNMENT: u64 = 1_048_576;
const BUFFER_BYTES: usize = 256 * 1024;

fn invalid() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, "Invalid GGUF header or index.")
}

fn check_cancel(cancel: &mut (impl FnMut() -> bool + ?Sized)) -> io::Result<()> {
    if cancel() {
        Err(io::Error::from(io::ErrorKind::Interrupted))
    } else {
        Ok(())
    }
}

const fn retain_all_strings(_key: &str) -> bool {
    true
}

const fn skip_all_scalar_arrays(_key: &str, _element_type_id: u32) -> bool {
    false
}

/// Bounds and retention policy for [`read_index`] and [`read_index_cancellable`].
#[derive(Clone, Copy, Debug)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "Explicit policy booleans mirror downstream intake and weights validation flags."
)]
pub struct IndexPolicy<'a> {
    /// Admit legacy GGUF version 2 headers in addition to version 3.
    pub allow_v2: bool,
    /// Admit GGUF files with zero tensor descriptors.
    pub allow_zero_tensors: bool,
    /// Admit GGUF files with zero metadata entries.
    pub allow_zero_metadata: bool,
    /// Maximum accepted input stream byte length.
    pub max_file_bytes: u64,
    /// Maximum accepted tensor descriptor count.
    pub max_tensors: u64,
    /// Maximum accepted metadata KV count.
    pub max_metadata: u64,
    /// Total byte budget charged across keys, strings, overhead, and retained arrays.
    pub metadata_budget: usize,
    /// Maximum accepted metadata key or tensor name UTF-8 byte length.
    pub max_key_bytes: usize,
    /// Maximum accepted scalar string UTF-8 byte length.
    pub max_string_bytes: usize,
    /// Maximum accepted array-element string UTF-8 byte length.
    pub max_array_string_bytes: usize,
    /// Optional cumulative byte cap on retained string-array elements.
    pub max_retained_string_array_bytes: Option<usize>,
    /// Maximum accepted element count in any single GGUF array.
    pub max_array_items: u64,
    /// Maximum accepted `general.alignment` value (must remain a nonzero power of two).
    pub max_alignment: u64,
    /// Accept positive signed `INT32` `general.alignment` values in addition to `UINT32`.
    pub allow_positive_i32_alignment: bool,
    /// Reject `NaN` and infinite `F32`/`F64` metadata values.
    pub reject_nonfinite_floats: bool,
    /// Reject interior NUL (`'\0'`) bytes inside metadata keys and tensor names.
    pub reject_interior_nul_keys: bool,
    /// Reject interior NUL (`'\0'`) bytes inside scalar and array string values.
    pub reject_interior_nul_strings: bool,
    /// Fixed per-KV overhead bytes charged against `metadata_budget`.
    pub kv_overhead_bytes: usize,
    /// Fixed per-string overhead bytes charged against `metadata_budget`.
    pub string_overhead_bytes: usize,
    /// Fixed per-tensor overhead bytes charged against `metadata_budget`.
    pub tensor_overhead_bytes: usize,
    /// Charge `size_of::<String>()` per slot when retaining string arrays.
    pub charge_string_array_slots: bool,
    /// Charge skipped scalar array payload bytes against `metadata_budget`.
    pub charge_skipped_scalar_arrays: bool,
    /// Enable private Prism tensor encodings (`PQ2_0` / `PTQ1_0`).
    pub prism_dialect: bool,
    /// Optional explicit allowlist of accepted GGML tensor type IDs.
    pub allowed_tensor_types: Option<&'a [u32]>,
    /// Predicate deciding whether a string array (`type_id = 8`) for `key` is retained.
    pub retain_string_array: fn(&str) -> bool,
    /// Predicate deciding whether a scalar array (`element_type_id != 8`) for `key` is retained.
    pub retain_scalar_array: fn(&str, u32) -> bool,
}

impl IndexPolicy<'_> {
    /// Returns conservative bounded defaults suitable for GGUF v3 index validation.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            allow_v2: false,
            allow_zero_tensors: false,
            allow_zero_metadata: true,
            max_file_bytes: DEFAULT_MAX_FILE_BYTES,
            max_tensors: DEFAULT_MAX_ITEMS,
            max_metadata: DEFAULT_MAX_ITEMS,
            metadata_budget: DEFAULT_METADATA_BUDGET,
            max_key_bytes: DEFAULT_MAX_KEY_BYTES,
            max_string_bytes: DEFAULT_MAX_STRING_BYTES,
            max_array_string_bytes: DEFAULT_MAX_STRING_BYTES,
            max_retained_string_array_bytes: Some(DEFAULT_MAX_RETAINED_STRING_ARRAY_BYTES),
            max_array_items: DEFAULT_MAX_ARRAY_ITEMS,
            max_alignment: DEFAULT_MAX_ALIGNMENT,
            allow_positive_i32_alignment: false,
            reject_nonfinite_floats: true,
            reject_interior_nul_keys: true,
            reject_interior_nul_strings: true,
            kv_overhead_bytes: 0,
            string_overhead_bytes: 0,
            tensor_overhead_bytes: 0,
            charge_string_array_slots: false,
            charge_skipped_scalar_arrays: false,
            prism_dialect: false,
            allowed_tensor_types: None,
            retain_string_array: retain_all_strings,
            retain_scalar_array: skip_all_scalar_arrays,
        }
    }
}

impl Default for IndexPolicy<'_> {
    fn default() -> Self {
        Self::new()
    }
}

/// A parsed metadata entry, either retained losslessly or recorded as a validated skipped array.
#[derive(Clone, Debug, PartialEq)]
pub enum IndexValue {
    /// Retained scalar, string, or homogeneous array value.
    Metadata(MetadataValue),
    /// Validated array whose elements were skipped rather than retained in memory.
    SkippedArray {
        /// GGML array element type ID (`0..=8` or `10..=12`).
        element_type_id: u32,
        /// Number of validated array elements.
        count: usize,
    },
}

impl IndexValue {
    /// Borrows the retained [`MetadataValue`] when this entry was not skipped.
    #[must_use]
    pub const fn as_metadata(&self) -> Option<&MetadataValue> {
        match self {
            Self::Metadata(value) => Some(value),
            Self::SkippedArray { .. } => None,
        }
    }

    /// Consumes this entry into its retained [`MetadataValue`], if present.
    #[must_use]
    pub fn into_metadata(self) -> Option<MetadataValue> {
        match self {
            Self::Metadata(value) => Some(value),
            Self::SkippedArray { .. } => None,
        }
    }

    /// Returns `(element_type_id, count)` when this entry represents a validated skipped array.
    #[must_use]
    pub const fn skipped_array(&self) -> Option<(u32, usize)> {
        match self {
            Self::Metadata(_) => None,
            Self::SkippedArray {
                element_type_id,
                count,
            } => Some((*element_type_id, *count)),
        }
    }
}

/// Validated tensor descriptor produced by [`read_index`] / [`read_index_cancellable`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndexedTensor {
    /// Nonzero dimensions in GGML order.
    pub dimensions: Vec<u64>,
    /// GGML tensor encoding ID.
    pub type_id: u32,
    /// Relative byte offset from `data_offset` as stored in the GGUF descriptor.
    pub relative_offset: u64,
    /// Absolute byte offset (`data_offset + relative_offset`) within the input stream.
    pub offset: u64,
    /// Exact serialized payload byte length.
    pub bytes: u64,
    /// Total element count across all dimensions.
    pub elements: u64,
}

/// Complete validated GGUF header, metadata map, and non-overlapping tensor index.
#[derive(Clone, Debug)]
pub struct ParsedIndex {
    /// GGUF format version (`2` or `3`).
    pub version: u32,
    /// Resolved power-of-two tensor data alignment in bytes.
    pub alignment: u64,
    /// Absolute byte offset where tensor payloads begin.
    pub data_offset: u64,
    /// Remaining uncharged bytes from [`IndexPolicy::metadata_budget`].
    pub remaining_budget: usize,
    /// Validated metadata keyed by unique GGUF key name.
    pub metadata: BTreeMap<String, IndexValue>,
    /// Validated tensor descriptors keyed by unique tensor name.
    pub tensors: BTreeMap<String, IndexedTensor>,
}

struct StreamReader<'a, R, C: ?Sized> {
    reader: BufReader<&'a mut R>,
    position: u64,
    file_bytes: u64,
    remaining_budget: usize,
    retained_string_array_bytes: usize,
    policy: &'a IndexPolicy<'a>,
    cancel: &'a mut C,
    scratch: Vec<u8>,
}

impl<R: Read + Seek, C: FnMut() -> bool + ?Sized> StreamReader<'_, R, C> {
    fn charge(&mut self, bytes: usize) -> io::Result<()> {
        self.remaining_budget = self
            .remaining_budget
            .checked_sub(bytes)
            .ok_or_else(invalid)?;
        Ok(())
    }

    fn charge_retained_string_bytes(&mut self, bytes: usize) -> io::Result<()> {
        if let Some(limit) = self.policy.max_retained_string_array_bytes {
            let next = self
                .retained_string_array_bytes
                .checked_add(bytes)
                .ok_or_else(invalid)?;
            if next > limit {
                return Err(invalid());
            }
            self.retained_string_array_bytes = next;
        }
        Ok(())
    }

    fn need(&self, bytes: u64) -> io::Result<()> {
        if bytes > self.file_bytes.saturating_sub(self.position) {
            return Err(invalid());
        }
        Ok(())
    }

    fn read_bytes<const N: usize>(&mut self) -> io::Result<[u8; N]> {
        check_cancel(self.cancel)?;
        let len = u64::try_from(N).map_err(|_| invalid())?;
        self.need(len)?;
        let mut buf = [0_u8; N];
        self.reader.read_exact(&mut buf)?;
        self.position = self.position.checked_add(len).ok_or_else(invalid)?;
        Ok(buf)
    }

    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.read_bytes()?))
    }

    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.read_bytes()?))
    }

    fn f32(&mut self) -> io::Result<f32> {
        let value = f32::from_le_bytes(self.read_bytes()?);
        if self.policy.reject_nonfinite_floats && !value.is_finite() {
            return Err(invalid());
        }
        Ok(value)
    }

    fn f64(&mut self) -> io::Result<f64> {
        let value = f64::from_le_bytes(self.read_bytes()?);
        if self.policy.reject_nonfinite_floats && !value.is_finite() {
            return Err(invalid());
        }
        Ok(value)
    }

    fn boolean(&mut self) -> io::Result<bool> {
        match self.read_bytes::<1>()?[0] {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(invalid()),
        }
    }

    fn skip(&mut self, bytes: u64) -> io::Result<()> {
        check_cancel(self.cancel)?;
        self.need(bytes)?;
        let delta = i64::try_from(bytes).map_err(|_| invalid())?;
        let next = self.position.checked_add(bytes).ok_or_else(invalid)?;
        self.reader.seek_relative(delta)?;
        self.position = next;
        Ok(())
    }

    fn validate_bool_bytes(&mut self, mut count: u64) -> io::Result<()> {
        self.need(count)?;
        let mut buf = [0_u8; 1024];
        while count > 0 {
            check_cancel(self.cancel)?;
            let step = usize::try_from(count.min(1024)).map_err(|_| invalid())?;
            self.reader.read_exact(&mut buf[..step])?;
            if buf[..step].iter().any(|&byte| byte > 1) {
                return Err(invalid());
            }
            let step_u64 = u64::try_from(step).map_err(|_| invalid())?;
            self.position = self.position.checked_add(step_u64).ok_or_else(invalid)?;
            count -= step_u64;
        }
        Ok(())
    }

    fn text_into_scratch(&mut self, limit: usize, reject_nul: bool) -> io::Result<()> {
        let count_u64 = self.u64()?;
        let count = usize::try_from(count_u64).map_err(|_| invalid())?;
        if count > limit {
            return Err(invalid());
        }
        self.need(count_u64)?;
        let charge = count
            .checked_add(self.policy.string_overhead_bytes)
            .ok_or_else(invalid)?;
        self.charge(charge)?;
        self.scratch.clear();
        if count > self.scratch.capacity() {
            self.scratch
                .try_reserve_exact(count - self.scratch.capacity())
                .map_err(|_| invalid())?;
        }
        self.scratch.resize(count, 0);
        check_cancel(self.cancel)?;
        self.reader.read_exact(&mut self.scratch)?;
        self.position = self.position.checked_add(count_u64).ok_or_else(invalid)?;
        if reject_nul && self.scratch.contains(&0) {
            return Err(invalid());
        }
        let _ = std::str::from_utf8(&self.scratch).map_err(|_| invalid())?;
        Ok(())
    }

    fn text(&mut self, limit: usize, reject_nul: bool) -> io::Result<String> {
        self.text_into_scratch(limit, reject_nul)?;
        let mut value = String::new();
        value
            .try_reserve_exact(self.scratch.len())
            .map_err(|_| invalid())?;
        let text = std::str::from_utf8(&self.scratch).map_err(|_| invalid())?;
        value.push_str(text);
        Ok(value)
    }

    fn read_scalar(&mut self, type_id: u32) -> io::Result<MetadataValue> {
        Ok(match type_id {
            0 => MetadataValue::U8(self.read_bytes::<1>()?[0]),
            1 => MetadataValue::I8(i8::from_le_bytes(self.read_bytes()?)),
            2 => MetadataValue::U16(u16::from_le_bytes(self.read_bytes()?)),
            3 => MetadataValue::I16(i16::from_le_bytes(self.read_bytes()?)),
            4 => MetadataValue::U32(self.u32()?),
            5 => MetadataValue::I32(i32::from_le_bytes(self.read_bytes()?)),
            6 => MetadataValue::F32(self.f32()?),
            7 => MetadataValue::Bool(self.boolean()?),
            10 => MetadataValue::U64(self.u64()?),
            11 => MetadataValue::I64(i64::from_le_bytes(self.read_bytes()?)),
            12 => MetadataValue::F64(self.f64()?),
            _ => return Err(invalid()),
        })
    }

    fn read_scalar_array<T>(
        &mut self,
        count: usize,
        mut read: impl FnMut(&mut Self) -> io::Result<T>,
    ) -> io::Result<Vec<T>> {
        let bytes = count.checked_mul(size_of::<T>()).ok_or_else(invalid)?;
        self.charge(bytes)?;
        let mut values = Vec::new();
        values.try_reserve_exact(count).map_err(|_| invalid())?;
        for _ in 0..count {
            values.push(read(self)?);
        }
        Ok(values)
    }

    fn read_array(&mut self, key: &str) -> io::Result<IndexValue> {
        let element_type_id = self.u32()?;
        let count_u64 = self.u64()?;
        if count_u64 > self.policy.max_array_items {
            return Err(invalid());
        }
        let count = usize::try_from(count_u64).map_err(|_| invalid())?;
        if element_type_id == 8 {
            let min_bytes = count_u64.checked_mul(8).ok_or_else(invalid)?;
            self.need(min_bytes)?;
            let retain = (self.policy.retain_string_array)(key);
            if retain {
                if self.policy.charge_string_array_slots {
                    self.charge(count.checked_mul(size_of::<String>()).ok_or_else(invalid)?)?;
                }
                let mut strings = Vec::new();
                strings.try_reserve_exact(count).map_err(|_| invalid())?;
                for _ in 0..count {
                    let s = self.text(
                        self.policy.max_array_string_bytes,
                        self.policy.reject_interior_nul_strings,
                    )?;
                    self.charge_retained_string_bytes(s.len())?;
                    strings.push(s);
                }
                return Ok(IndexValue::Metadata(MetadataValue::Array(
                    MetadataArray::String(strings),
                )));
            }
            for _ in 0..count {
                self.text_into_scratch(
                    self.policy.max_array_string_bytes,
                    self.policy.reject_interior_nul_strings,
                )?;
            }
            return Ok(IndexValue::SkippedArray {
                element_type_id,
                count,
            });
        }
        // GGUF prohibits nested arrays (`type_id = 9` inside an array).
        let width = scalar_width(element_type_id).ok_or_else(invalid)?;
        let payload_u64 = count_u64.checked_mul(width).ok_or_else(invalid)?;
        self.need(payload_u64)?;
        if (self.policy.retain_scalar_array)(key, element_type_id) {
            let array = match element_type_id {
                0 => MetadataArray::U8(
                    self.read_scalar_array(count, |reader| Ok(reader.read_bytes::<1>()?[0]))?,
                ),
                1 => MetadataArray::I8(self.read_scalar_array(count, |reader| {
                    Ok(i8::from_le_bytes(reader.read_bytes()?))
                })?),
                2 => MetadataArray::U16(self.read_scalar_array(count, |reader| {
                    Ok(u16::from_le_bytes(reader.read_bytes()?))
                })?),
                3 => MetadataArray::I16(self.read_scalar_array(count, |reader| {
                    Ok(i16::from_le_bytes(reader.read_bytes()?))
                })?),
                4 => MetadataArray::U32(self.read_scalar_array(count, Self::u32)?),
                5 => MetadataArray::I32(self.read_scalar_array(count, |reader| {
                    Ok(i32::from_le_bytes(reader.read_bytes()?))
                })?),
                6 => MetadataArray::F32(self.read_scalar_array(count, Self::f32)?),
                7 => MetadataArray::Bool(self.read_scalar_array(count, Self::boolean)?),
                10 => MetadataArray::U64(self.read_scalar_array(count, Self::u64)?),
                11 => MetadataArray::I64(self.read_scalar_array(count, |reader| {
                    Ok(i64::from_le_bytes(reader.read_bytes()?))
                })?),
                12 => MetadataArray::F64(self.read_scalar_array(count, Self::f64)?),
                _ => return Err(invalid()),
            };
            return Ok(IndexValue::Metadata(MetadataValue::Array(array)));
        }
        if self.policy.charge_skipped_scalar_arrays {
            let payload_usize = usize::try_from(payload_u64).map_err(|_| invalid())?;
            self.charge(payload_usize)?;
        }
        if element_type_id == 7 {
            self.validate_bool_bytes(count_u64)?;
        } else {
            self.skip(payload_u64)?;
        }
        Ok(IndexValue::SkippedArray {
            element_type_id,
            count,
        })
    }

    fn read_value(&mut self, key: &str) -> io::Result<IndexValue> {
        let type_id = self.u32()?;
        match type_id {
            8 => {
                let s = self.text(
                    self.policy.max_string_bytes,
                    self.policy.reject_interior_nul_strings,
                )?;
                Ok(IndexValue::Metadata(MetadataValue::String(s)))
            }
            9 => self.read_array(key),
            _ => Ok(IndexValue::Metadata(self.read_scalar(type_id)?)),
        }
    }

    fn read_header(&mut self) -> io::Result<(u32, u64, u64)> {
        if &self.read_bytes::<4>()? != b"GGUF" {
            return Err(invalid());
        }
        let version = self.u32()?;
        if version != 3 && !(self.policy.allow_v2 && version == 2) {
            return Err(invalid());
        }
        let tensor_count = self.u64()?;
        let metadata_count = self.u64()?;
        if (!self.policy.allow_zero_tensors && tensor_count == 0)
            || (!self.policy.allow_zero_metadata && metadata_count == 0)
            || tensor_count > self.policy.max_tensors
            || metadata_count > self.policy.max_metadata
        {
            return Err(invalid());
        }
        Ok((version, tensor_count, metadata_count))
    }

    fn read_metadata_table(
        &mut self,
        metadata_count: u64,
    ) -> io::Result<(BTreeMap<String, IndexValue>, u64)> {
        let mut metadata = BTreeMap::new();
        for _ in 0..metadata_count {
            self.charge(self.policy.kv_overhead_bytes)?;
            let key = self.text(
                self.policy.max_key_bytes,
                self.policy.reject_interior_nul_keys,
            )?;
            if key.is_empty() || metadata.contains_key(&key) {
                return Err(invalid());
            }
            let value = self.read_value(&key)?;
            metadata.insert(key, value);
        }

        let alignment = match metadata
            .get("general.alignment")
            .and_then(IndexValue::as_metadata)
        {
            None => 32_u64,
            Some(MetadataValue::U32(value))
                if *value > 0
                    && value.is_power_of_two()
                    && u64::from(*value) <= self.policy.max_alignment =>
            {
                u64::from(*value)
            }
            Some(MetadataValue::I32(value))
                if self.policy.allow_positive_i32_alignment
                    && *value > 0
                    && value.unsigned_abs().is_power_of_two()
                    && u64::from(value.unsigned_abs()) <= self.policy.max_alignment =>
            {
                u64::from(value.unsigned_abs())
            }
            Some(_) => return Err(invalid()),
        };

        Ok((metadata, alignment))
    }

    fn read_tensor_table(
        &mut self,
        tensor_count: u64,
        alignment: u64,
    ) -> io::Result<BTreeMap<String, IndexedTensor>> {
        let mut tensors = BTreeMap::new();
        for _ in 0..tensor_count {
            self.charge(self.policy.tensor_overhead_bytes)?;
            let name = self.text(
                self.policy.max_key_bytes,
                self.policy.reject_interior_nul_keys,
            )?;
            let dimension_count = self.u32()?;
            if name.is_empty() || tensors.contains_key(&name) || !(1..=4).contains(&dimension_count)
            {
                return Err(invalid());
            }
            let mut dimensions =
                Vec::with_capacity(usize::try_from(dimension_count).map_err(|_| invalid())?);
            let mut elements = 1_u64;
            for _ in 0..dimension_count {
                let size = self.u64()?;
                if size == 0 || i64::try_from(size).is_err() {
                    return Err(invalid());
                }
                elements = elements.checked_mul(size).ok_or_else(invalid)?;
                dimensions.push(size);
            }
            let type_id = self.u32()?;
            let relative_offset = self.u64()?;
            if let Some(allowed) = self.policy.allowed_tensor_types
                && !allowed.contains(&type_id)
            {
                return Err(invalid());
            }
            let (block, block_bytes) =
                stock_tensor_layout(type_id, self.policy.prism_dialect).ok_or_else(invalid)?;
            if !dimensions[0].is_multiple_of(block) || !relative_offset.is_multiple_of(alignment) {
                return Err(invalid());
            }
            let bytes = (elements / block)
                .checked_mul(block_bytes)
                .ok_or_else(invalid)?;
            tensors.insert(
                name,
                IndexedTensor {
                    dimensions,
                    type_id,
                    relative_offset,
                    offset: 0,
                    bytes,
                    elements,
                },
            );
        }
        Ok(tensors)
    }

    fn resolve_tensor_offsets(
        &mut self,
        tensors: &mut BTreeMap<String, IndexedTensor>,
        alignment: u64,
    ) -> io::Result<u64> {
        if tensors.is_empty() {
            return Ok(self.position);
        }
        let aligned = self
            .position
            .checked_next_multiple_of(alignment)
            .ok_or_else(invalid)?;
        if aligned > self.file_bytes {
            return Err(invalid());
        }
        let mut ranges = Vec::new();
        ranges
            .try_reserve_exact(tensors.len())
            .map_err(|_| invalid())?;
        for tensor in tensors.values_mut() {
            check_cancel(self.cancel)?;
            tensor.offset = aligned
                .checked_add(tensor.relative_offset)
                .ok_or_else(invalid)?;
            let end = tensor
                .offset
                .checked_add(tensor.bytes)
                .ok_or_else(invalid)?;
            if end > self.file_bytes {
                return Err(invalid());
            }
            ranges.push((tensor.offset, end));
        }
        ranges.sort_unstable();
        if ranges.array_windows().any(|[left, right]| left.1 > right.0) {
            return Err(invalid());
        }
        Ok(aligned)
    }

    fn parse(&mut self) -> io::Result<ParsedIndex> {
        let (version, tensor_count, metadata_count) = self.read_header()?;
        let (metadata, alignment) = self.read_metadata_table(metadata_count)?;
        let mut tensors = self.read_tensor_table(tensor_count, alignment)?;
        let data_offset = self.resolve_tensor_offsets(&mut tensors, alignment)?;
        check_cancel(self.cancel)?;
        Ok(ParsedIndex {
            version,
            alignment,
            data_offset,
            remaining_budget: self.remaining_budget,
            metadata,
            tensors,
        })
    }
}

/// Parses and validates a GGUF v2/v3 header and tensor index from `source` under `policy`.
///
/// Rewinds `source` to offset `0` before reading. Does not read or validate tensor payloads.
///
/// # Errors
/// Returns [`io::ErrorKind::InvalidData`] on malformed headers, unsupported versions/types,
/// out-of-range offsets, overlapping tensor extents, or policy budget violations.
pub fn read_index<R: Read + Seek>(
    source: &mut R,
    file_bytes: u64,
    policy: &IndexPolicy<'_>,
) -> io::Result<ParsedIndex> {
    read_index_cancellable(source, file_bytes, policy, &mut || false)
}

/// Cancellable variant of [`read_index`].
///
/// Polls `cancel` before performing any seek or read, and periodically throughout buffered
/// metadata, string-array, and descriptor parsing.
///
/// # Errors
/// Returns [`io::ErrorKind::Interrupted`] if `cancel` returns `true` (leaving `source` untouched
/// when pre-cancelled), [`io::ErrorKind::InvalidData`] on malformed headers or policy violations,
/// or underlying stream I/O errors.
pub fn read_index_cancellable<R: Read + Seek>(
    source: &mut R,
    file_bytes: u64,
    policy: &IndexPolicy<'_>,
    cancel: &mut (impl FnMut() -> bool + ?Sized),
) -> io::Result<ParsedIndex> {
    check_cancel(cancel)?;
    if !(MIN_HEADER_BYTES..=policy.max_file_bytes).contains(&file_bytes) {
        return Err(invalid());
    }
    source.seek(SeekFrom::Start(0))?;
    let mut reader = StreamReader {
        reader: BufReader::with_capacity(BUFFER_BYTES, source),
        position: 0,
        file_bytes,
        remaining_budget: policy.metadata_budget,
        retained_string_array_bytes: 0,
        policy,
        cancel,
        scratch: Vec::new(),
    };
    reader.parse()
}
