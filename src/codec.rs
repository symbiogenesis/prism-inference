//! Bounded persistent codec buffers and pinned GGML trait metadata.
//!
//! Native metadata follows Prism `b87ca7225456b06b0975b8239c66b5aecf07e37c`,
//! `ggml/src/ggml.c` and `ggml/src/ggml-cpu/ggml-cpu.c` (MIT; see NOTICE).
//! Native function availability and implemented Rust operations are separate:
//! a reference quantizer never substitutes for an unverified CPU quantizer.
//! All packed data is accessed as bytes, including deliberately unaligned rows.

use crate::kernels::{decode_row_into, packed_dot, quantize_row};
use crate::model::{gguf_context::tensor_layout, quantization_policy::type_name};
use crate::{Result, invalid};

/// Pinned type enum extent, including deprecated and reserved IDs.
pub const TYPE_COUNT: u32 = crate::model::quantization_policy::TYPE_COUNT;
/// Maximum logical bytes in one persistent buffer or codec row.
pub const MAX_BUFFER_BYTES: usize = 512 * 1024 * 1024;
/// Maximum logical bytes retained by one buffer owner.
pub const MAX_ALLOCATED_BYTES: usize = 2 * 1024 * 1024 * 1024;
/// Maximum simultaneously live persistent buffers.
pub const MAX_BUFFERS: usize = 32;
const HANDLE_LIMIT: u32 = 1_000_000;

/// Independently available codec operations, never inferred from the layout.
#[allow(
    clippy::struct_excessive_bools,
    reason = "Each field represents an independent native function pointer or Rust implementation."
)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CodecCapabilities {
    /// CPU quantization with its own verified semantics.
    pub from_float_cpu: bool,
    /// Deterministic reference quantization.
    pub from_float_reference: bool,
    /// Packed bytes to scalar float decoding.
    pub to_float: bool,
    /// Dot product using the declared packed activation type.
    pub vec_dot: bool,
}

/// One exact enum slot, retaining removed names and unnamed reserved gaps.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CodecTraits {
    /// Stable on-disk GGML type ID.
    pub type_id: u32,
    /// Pinned native name, or none for an unnamed reserved slot.
    pub name: Option<&'static str>,
    /// Number of elements per block, zero for an inactive type.
    pub block_size: usize,
    /// Serialized bytes per block, zero for an inactive type.
    pub type_size: usize,
    /// Original native pointer availability, independent of Rust support.
    pub native: CodecCapabilities,
    /// Available safe Rust kernels, including scalar and auxiliary conversions.
    /// Encoding calls also require the corresponding native entry point so an
    /// absent native reference/CPU operation cannot be silently invented.
    pub rust: CodecCapabilities,
    /// Native activation type ID; meaningful for dots and retained for all slots.
    pub vec_dot_type: u32,
}

impl CodecTraits {
    /// Computes the extent of a complete row without allocating it.
    ///
    /// Empty rows have zero bytes for an active layout. Executable codec calls
    /// require a nonempty row, as do the underlying checked Rust kernels.
    ///
    /// # Errors
    /// Rejects inactive layouts, partial blocks, overflow and rows above 512 MiB.
    pub fn row_bytes(self, count: usize) -> Result<usize> {
        if self.block_size == 0 || !count.is_multiple_of(self.block_size) {
            return Err(invalid("codec element count violates block precondition"));
        }
        (count / self.block_size)
            .checked_mul(self.type_size)
            .filter(|&bytes| bytes <= MAX_BUFFER_BYTES)
            .ok_or_else(|| invalid("codec row exceeds buffer bound"))
    }
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Pinned tensor_layout block and byte widths are all at most 292, representable by every Rust usize."
)]
fn descriptor(type_id: u32) -> CodecTraits {
    // Layouts and names share the production GGUF/policy identity rather than
    // introducing a second table with subtly different enum numbering.
    let (block_size, type_size) =
        tensor_layout(type_id).map_or((0, 0), |(block, bytes)| (block as usize, bytes as usize));
    let reference = matches!(
        type_id,
        1..=3 | 6..=14 | 18 | 20..=23 | 30 | 34 | 35 | 39..=42 | 142 | 143
    );
    let cpu = matches!(
        type_id,
        0..=3 | 6..=15 | 20 | 23 | 26 | 30 | 34 | 35 | 39..=42 | 142 | 143
    );
    let decode = matches!(
        type_id,
        1..=3 | 6..=8 | 10..=14 | 16..=23 | 29 | 30 | 34 | 35 | 39..=42 | 142 | 143
    );
    let dot = type_id == 0 || decode;
    let vec_dot_type = match type_id {
        1 => 1,
        2 | 6 | 8 | 20 | 39..=42 | 143 => 8,
        3 | 7 | 9 => 9,
        10..=14 | 16..=19 | 21..=23 | 29 | 34 | 35 | 142 => 15,
        30 => 30,
        _ => 0,
    };
    CodecTraits {
        type_id,
        name: type_name(type_id),
        block_size,
        type_size,
        native: CodecCapabilities {
            from_float_cpu: cpu,
            from_float_reference: reference,
            to_float: decode,
            vec_dot: dot,
        },
        rust: CodecCapabilities {
            // Only the CPU F32 identity conversion is architecture independent
            // and verified here. CPU SIMD quantizers can differ from reference.
            from_float_cpu: type_id == 0,
            from_float_reference: matches!(
                type_id,
                0..=3 | 6..=15 | 18 | 20..=23 | 30 | 34 | 35 | 39..=42 | 142 | 143
            ),
            // Safe kernels also define scalar/auxiliary reads unavailable as
            // native to_float pointers; this does not change native metadata.
            to_float: decode || matches!(type_id, 0 | 9 | 15),
            vec_dot: dot,
        },
        vec_dot_type,
    }
}

/// Enumerates every pinned type ID in order, including inactive slots.
pub fn inventory() -> impl ExactSizeIterator<Item = CodecTraits> {
    (0..TYPE_COUNT).map(descriptor)
}

/// Returns traits for a checked type ID.
///
/// # Errors
/// Rejects IDs outside the pinned enum extent.
pub fn traits(type_id: u32) -> Result<CodecTraits> {
    if type_id >= TYPE_COUNT {
        return Err(invalid("codec type ID exceeds enum extent"));
    }
    Ok(descriptor(type_id))
}

/// Codec operation with an explicit CPU/reference distinction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CodecFunction {
    /// Execute only an implemented CPU conversion; never use a reference fallback.
    FromFloatCpu,
    /// Encode deterministic reference-format bytes from little-endian floats.
    FromFloatReference,
    /// Decode packed input into little-endian scalar floats.
    ToFloat,
    /// Compute one float using the exact declared packed activation format.
    VecDot,
}

/// Checked buffer identity and byte offset, with no pointer alignment requirement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BufferRegion {
    /// Live handle returned by [`CodecBuffers::new_buffer`].
    pub buffer: u32,
    /// Byte offset within that buffer's logical extent.
    pub offset: usize,
}

struct Buffer {
    storage: Vec<u8>,
    start: usize,
    bytes: usize,
}

impl Buffer {
    fn range(&self, offset: usize, bytes: usize) -> Result<std::ops::Range<usize>> {
        if offset > self.bytes || bytes > self.bytes - offset {
            return Err(invalid("codec buffer access exceeds extent"));
        }
        let start = self.start + offset;
        Ok(start..start + bytes)
    }
}

/// Exclusive owner of persistent codec buffers and monotonically unique handles.
///
/// Logical extents are limited to 512 MiB each and 2 GiB total, with at most 32
/// live handles. Alignment padding is bounded separately by 8192 bytes per buffer.
/// Callers synchronize ownership if sharing between workers; mutation requires
/// an exclusive borrow. Released handles never identify a later allocation.
pub struct CodecBuffers {
    buffers: Vec<(u32, Buffer)>,
    next: u32,
    allocated: usize,
}

impl Default for CodecBuffers {
    fn default() -> Self {
        Self {
            buffers: Vec::new(),
            next: 1,
            allocated: 0,
        }
    }
}

impl CodecBuffers {
    /// Returns the total logical bytes held by live buffers.
    #[must_use]
    pub const fn allocated_bytes(&self) -> usize {
        self.allocated
    }

    /// Returns the number of live buffer handles.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.buffers.len()
    }

    /// Reports whether no live buffers remain.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.buffers.is_empty()
    }

    /// Allocates zeroed bytes after an aligned base and a deliberate byte offset.
    ///
    /// Alignment is a power of two up to 4096; offset may be from zero through
    /// alignment. Zero-byte buffers are valid. Failure leaves the owner unchanged.
    ///
    /// # Errors
    /// Rejects invalid alignment/offset, resource limits and allocation failures.
    pub fn new_buffer(&mut self, bytes: usize, alignment: usize, offset: usize) -> Result<u32> {
        if alignment == 0 || !alignment.is_power_of_two() || alignment > 4096 || offset > alignment
        {
            return Err(invalid("invalid codec buffer alignment/offset"));
        }
        if bytes > MAX_BUFFER_BYTES
            || self.buffers.len() >= MAX_BUFFERS
            || self.next >= HANDLE_LIMIT
            || bytes > MAX_ALLOCATED_BYTES - self.allocated
        {
            return Err(invalid("codec aggregate buffer bound"));
        }
        self.buffers.try_reserve(1)?;
        let mut storage = Vec::new();
        let size = bytes + alignment * 2;
        storage.try_reserve_exact(size)?;
        storage.resize(size, 0);
        let start = (alignment - storage.as_ptr().addr() % alignment) % alignment + offset;
        let handle = self.next;
        self.buffers.push((
            handle,
            Buffer {
                storage,
                start,
                bytes,
            },
        ));
        self.next += 1;
        self.allocated += bytes;
        Ok(handle)
    }

    /// Releases one live buffer and restores its logical allocation allowance.
    ///
    /// # Errors
    /// Rejects unknown or previously released handles without changing state.
    pub fn free_buffer(&mut self, handle: u32) -> Result<()> {
        let index = self
            .buffers
            .iter()
            .position(|(id, _)| *id == handle)
            .ok_or_else(|| invalid("unknown codec release handle"))?;
        let (_, buffer) = self.buffers.swap_remove(index);
        self.allocated -= buffer.bytes;
        Ok(())
    }

    fn buffer(&self, handle: u32) -> Result<&Buffer> {
        self.buffers
            .iter()
            .find_map(|(id, buffer)| (*id == handle).then_some(buffer))
            .ok_or_else(|| invalid("unknown codec buffer handle"))
    }

    /// Borrows a checked region without copying it.
    ///
    /// # Errors
    /// Rejects unknown handles, out-of-range offsets and extent overflow.
    pub fn read(&self, region: BufferRegion, bytes: usize) -> Result<&[u8]> {
        let buffer = self.buffer(region.buffer)?;
        Ok(&buffer.storage[buffer.range(region.offset, bytes)?])
    }

    /// Replaces exactly one checked region without changing allocation size.
    ///
    /// # Errors
    /// Rejects unknown handles and out-of-range extents before writing any bytes.
    pub fn write(&mut self, region: BufferRegion, bytes: &[u8]) -> Result<()> {
        let buffer = self
            .buffers
            .iter_mut()
            .find_map(|(id, buffer)| (*id == region.buffer).then_some(buffer))
            .ok_or_else(|| invalid("unknown codec buffer handle"))?;
        let range = buffer.range(region.offset, bytes.len())?;
        buffer.storage[range].copy_from_slice(bytes);
        Ok(())
    }

    /// Executes one codec call from persistent bytes into persistent bytes.
    ///
    /// All ranges and capabilities are checked before kernel execution. Input,
    /// right-hand side and output may overlap; inputs are consumed completely
    /// before the result is written. Failure leaves all persistent bytes intact.
    /// Scalar floats always use little-endian encoding. Temporary row storage
    /// is bounded by the same 512 MiB limit and allocated fallibly.
    ///
    /// # Errors
    /// Rejects unsupported operations, empty/partial/oversized rows, malformed
    /// regions, unexpected/missing right-hand side, numerical and allocation errors.
    #[allow(
        clippy::too_many_arguments,
        reason = "One typed primitive mirrors the native codec call and its three distinct regions."
    )]
    pub fn call(
        &mut self,
        function: CodecFunction,
        kind: u32,
        count: usize,
        input: BufferRegion,
        rhs: Option<BufferRegion>,
        output: BufferRegion,
    ) -> Result<()> {
        let codec = traits(kind)?;
        let supported = match function {
            CodecFunction::FromFloatCpu => codec.native.from_float_cpu && codec.rust.from_float_cpu,
            CodecFunction::FromFloatReference => {
                codec.native.from_float_reference && codec.rust.from_float_reference
            }
            CodecFunction::ToFloat => codec.rust.to_float,
            CodecFunction::VecDot => codec.rust.vec_dot,
        };
        if !supported {
            return Err(invalid("requested Rust codec operation is unavailable"));
        }
        if count == 0 || count > MAX_BUFFER_BYTES / 4 {
            return Err(invalid("codec element count exceeds nonempty float bound"));
        }
        let packed_bytes = codec.row_bytes(count)?;
        let float_bytes = count * 4;
        if (function == CodecFunction::VecDot) != rhs.is_some() {
            return Err(invalid("codec right-hand side does not match function"));
        }
        let output_bytes = match function {
            CodecFunction::FromFloatCpu | CodecFunction::FromFloatReference => packed_bytes,
            CodecFunction::ToFloat => float_bytes,
            CodecFunction::VecDot => 4,
        };
        self.read(output, output_bytes)?;
        let encoded = match function {
            CodecFunction::FromFloatCpu | CodecFunction::FromFloatReference => {
                let raw = self.read(input, float_bytes)?;
                let mut values = Vec::new();
                values.try_reserve_exact(count)?;
                values.extend(
                    raw.as_chunks::<4>()
                        .0
                        .iter()
                        .map(|bytes| f32::from_le_bytes(*bytes)),
                );
                quantize_row(kind, &values)?
            }
            CodecFunction::ToFloat => {
                let mut values = Vec::new();
                decode_row_into(kind, count, self.read(input, packed_bytes)?, &mut values)?;
                encode_floats(&values)?
            }
            CodecFunction::VecDot => {
                let rhs = rhs.ok_or_else(|| invalid("codec vector dot needs right-hand side"))?;
                let rhs_bytes = traits(codec.vec_dot_type)?.row_bytes(count)?;
                let value = packed_dot(
                    kind,
                    count,
                    self.read(input, packed_bytes)?,
                    self.read(rhs, rhs_bytes)?,
                )?;
                encode_floats(&[value])?
            }
        };
        if encoded.len() != output_bytes {
            return Err(invalid("codec result extent differs"));
        }
        self.write(output, &encoded)
    }
}

fn encode_floats(values: &[f32]) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(values.len() * 4)?;
    bytes.extend(values.iter().flat_map(|value| value.to_le_bytes()));
    Ok(bytes)
}
