//! Safe, strided CPU tensor graphs using the pinned GGML operation contract.
//!
//! Storage is owned by the graph. Views retain byte strides and offsets, and
//! graph execution writes through aliases. Unsupported operations are errors;
//! support queries never advertise an operation without a numerical kernel.

#![allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::suboptimal_flops,
    reason = "The pinned GGML ABI specifies integer casts and binary32 operation ordering."
)]

use crate::{Result, invalid};
use bonsai_ternary::half_to_float;

pub mod allocator;
mod attention;
mod backward;
mod backward_image;
mod call;
mod gdn;
mod image;
mod ops;
mod recurrent;
mod rope;
mod spatial;
pub mod threadpool;

pub use call::{operation_name, unary_name};

/// An owned tensor identity within one [`CpuBackend`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct TensorId(pub usize);

/// An owned computation graph identity within one [`CpuBackend`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphId(pub usize);

/// A typed argument to the pinned production graph operation API.
#[derive(Clone, Debug)]
pub enum Argument {
    /// A tensor owned by this backend.
    Tensor(TensorId),
    /// A graph owned by this backend.
    Graph(GraphId),
    /// A signed integer or enum discriminant.
    Integer(i64),
    /// A binary32 scalar.
    Float(f32),
    /// A boolean option.
    Bool(bool),
    /// An immutable dimension, stride, or axis list.
    Integers(Vec<i64>),
    /// A nullable byte string, validated by the caller as UTF-8.
    String(Option<String>),
    /// An execution-context identity. Contexts own metadata, not native memory.
    Context(u32),
    /// An absent optional input.
    Null,
}

/// The typed result of a graph API call.
#[derive(Clone, Debug)]
pub enum Observation {
    /// A tensor, or absence for a failed name lookup.
    Tensor(Option<TensorId>),
    /// A newly allocated graph.
    Graph(GraphId),
    /// A signed integer result.
    Integer(i64),
    /// A boolean result.
    Bool(bool),
    /// A nullable string result.
    String(Option<String>),
    /// No returned value.
    Void,
}

/// The storage geometry of a pinned scalar or quantized element type.
#[derive(Clone, Copy, Debug)]
pub struct TypeLayout {
    /// Number of logical values in one storage block.
    pub block: usize,
    /// Bytes in one storage block.
    pub bytes: usize,
    /// Pinned type name.
    pub name: &'static str,
}

/// Resolve supported production tensor storage types.
///
/// # Errors
/// Rejects types without an implemented production decoder.
pub fn type_layout(kind: i32) -> Result<TypeLayout> {
    let (block, bytes, name) = match kind {
        0 => (1, 4, "f32"),
        1 => (1, 2, "f16"),
        2 => (32, 18, "q4_0"),
        3 => (32, 20, "q4_1"),
        6 => (32, 22, "q5_0"),
        7 => (32, 24, "q5_1"),
        8 => (32, 34, "q8_0"),
        9 => (32, 36, "q8_1"),
        10 => (256, 84, "q2_K"),
        11 => (256, 110, "q3_K"),
        12 => (256, 144, "q4_K"),
        13 => (256, 176, "q5_K"),
        14 => (256, 210, "q6_K"),
        15 => (256, 292, "q8_K"),
        16 => (256, 66, "iq2_xxs"),
        17 => (256, 74, "iq2_xs"),
        18 => (256, 98, "iq3_xxs"),
        19 => (256, 50, "iq1_s"),
        21 => (256, 110, "iq3_s"),
        22 => (256, 82, "iq2_s"),
        29 => (256, 56, "iq1_m"),
        20 => (32, 18, "iq4_nl"),
        23 => (256, 136, "iq4_xs"),
        34 => (256, 54, "tq1_0"),
        35 => (256, 66, "tq2_0"),
        39 => (32, 17, "mxfp4"),
        40 => (64, 36, "nvfp4"),
        41 => (128, 18, "q1_0"),
        42 => (64, 18, "q2_0"),
        24 => (1, 1, "i8"),
        25 => (1, 2, "i16"),
        26 => (1, 4, "i32"),
        27 => (1, 8, "i64"),
        28 => (1, 8, "f64"),
        30 => (1, 2, "bf16"),
        142 => (128, 34, "pq2_0"),
        143 => (128, 28, "ptq1_0"),
        _ => return Err(invalid(format!("unsupported CPU tensor type {kind}"))),
    };
    Ok(TypeLayout { block, bytes, name })
}

/// Public tensor metadata; dimensions are ordered with the contiguous axis first.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    /// Storage type discriminant.
    pub kind: i32,
    /// Logical dimensions.
    pub shape: [usize; 4],
    /// Byte strides, including packed block stride on axis zero.
    pub strides: [usize; 4],
    /// Pinned operation discriminant.
    pub operation: i32,
    /// Input tensor identities, in operation order.
    pub sources: Vec<TensorId>,
    /// Original nullable GGML source slots, retaining optional-input positions.
    pub source_slots: [Option<TensorId>; 10],
    /// Underlying tensor for a storage view.
    pub view: Option<TensorId>,
    /// Byte offset relative to the underlying tensor.
    pub offset: usize,
    /// Operation parameters, in the pinned little-endian ABI.
    pub parameters: [u8; 64],
    /// Caller assigned tensor name.
    pub name: String,
    /// Pinned tensor flags.
    pub flags: i32,
    /// The metadata context that owns this tensor.
    pub context: u32,
}

impl TensorInfo {
    /// Number of logical scalar elements.
    #[must_use]
    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Number of logical rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.shape[1..].iter().product()
    }

    /// Storage byte extent, including gaps between strided rows.
    ///
    /// # Errors
    /// Rejects unsupported storage types or overflowing byte extents.
    pub fn bytes(&self) -> Result<usize> {
        let layout = type_layout(self.kind)?;
        if self.shape.contains(&0) {
            return Ok(0);
        }
        let mut bytes = layout.bytes;
        for (axis, (&dimension, &stride)) in self.shape.iter().zip(&self.strides).enumerate() {
            let count = if axis == 0 {
                dimension / layout.block
            } else {
                dimension
            };
            bytes = bytes
                .checked_add(
                    (count - 1)
                        .checked_mul(stride)
                        .ok_or_else(|| invalid("tensor byte extent overflow"))?,
                )
                .ok_or_else(|| invalid("tensor byte extent overflow"))?;
        }
        Ok(bytes)
    }

    /// Whether all dimensions use the canonical contiguous byte strides.
    #[must_use]
    pub fn contiguous(&self) -> bool {
        contiguous_strides(self.kind, self.shape).is_ok_and(|strides| strides == self.strides)
    }
}

struct Node {
    info: TensorInfo,
    data: Vec<u8>,
    allocation: Option<(TensorId, usize)>,
}

#[derive(Clone, Default)]
struct Graph {
    nodes: Vec<TensorId>,
    leaves: Vec<TensorId>,
    capacity: usize,
    gradients_enabled: bool,
    gradients: std::collections::BTreeMap<TensorId, TensorId>,
    gradient_accumulators: std::collections::BTreeMap<TensorId, TensorId>,
}

/// Safe reference CPU backend with owned graph metadata and checked storage.
#[derive(Default)]
pub struct CpuBackend {
    tensors: Vec<Node>,
    graphs: Vec<Graph>,
}

fn product(shape: [usize; 4]) -> Result<usize> {
    shape.into_iter().try_fold(1_usize, |count, dim| {
        count
            .checked_mul(dim)
            .ok_or_else(|| invalid("tensor element count overflow"))
    })
}

fn contiguous_strides(kind: i32, shape: [usize; 4]) -> Result<[usize; 4]> {
    product(shape)?;
    let layout = type_layout(kind)?;
    if !shape[0].is_multiple_of(layout.block) {
        return Err(invalid("tensor width is not block aligned"));
    }
    let mut strides = [layout.bytes; 4];
    strides[1] = (shape[0] / layout.block)
        .checked_mul(layout.bytes)
        .ok_or_else(|| invalid("tensor row extent overflow"))?;
    for axis in 2..4 {
        strides[axis] = strides[axis - 1]
            .checked_mul(shape[axis - 1])
            .ok_or_else(|| invalid("tensor stride overflow"))?;
    }
    Ok(strides)
}

fn coordinates(mut index: usize, shape: [usize; 4]) -> [usize; 4] {
    std::array::from_fn(|axis| {
        let coordinate = index % shape[axis];
        index /= shape[axis];
        coordinate
    })
}

const fn linear(index: [usize; 4], shape: [usize; 4]) -> usize {
    index[0] + shape[0] * (index[1] + shape[1] * (index[2] + shape[2] * index[3]))
}

impl CpuBackend {
    /// Construct an empty CPU backend.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Read tensor metadata by owned identity.
    ///
    /// # Errors
    /// Rejects an unknown identity.
    pub fn info(&self, tensor: TensorId) -> Result<&TensorInfo> {
        self.tensors
            .get(tensor.0)
            .map(|node| &node.info)
            .ok_or_else(|| invalid("unknown CPU tensor identity"))
    }

    /// Change tensor metadata before graph execution.
    ///
    /// # Errors
    /// Rejects an unknown identity. Storage and numerical bounds are rechecked
    /// at allocation and I/O, including after callers change strides.
    pub fn info_mut(&mut self, tensor: TensorId) -> Result<&mut TensorInfo> {
        self.tensors
            .get_mut(tensor.0)
            .map(|node| &mut node.info)
            .ok_or_else(|| invalid("unknown CPU tensor identity"))
    }

    /// List tensors owned by a metadata context in creation order.
    #[must_use]
    pub fn context_tensors(&self, context: u32) -> Vec<TensorId> {
        self.tensors
            .iter()
            .enumerate()
            .filter_map(|(index, node)| (node.info.context == context).then_some(TensorId(index)))
            .collect()
    }

    /// Allocate a scalar or supported packed tensor's metadata, without data.
    ///
    /// # Errors
    /// Rejects invalid shapes, types, overflowing extents, and excess tensors.
    pub fn tensor(&mut self, context: u32, kind: i32, shape: [usize; 4]) -> Result<TensorId> {
        let strides = contiguous_strides(kind, shape)?;
        if self.tensors.len() >= 32768 || product(shape)? > 1 << 30 {
            return Err(invalid("CPU tensor extent exceeds resource bound"));
        }
        let info = TensorInfo {
            kind,
            shape,
            strides,
            operation: 0,
            sources: Vec::new(),
            source_slots: [None; 10],
            view: None,
            offset: 0,
            parameters: [0; 64],
            name: String::new(),
            flags: 0,
            context,
        };
        if info.bytes()? > 4 * 1024 * 1024 * 1024_usize {
            return Err(invalid("CPU tensor storage exceeds 4 GiB"));
        }
        let id = TensorId(self.tensors.len());
        self.tensors.push(Node {
            info,
            data: Vec::new(),
            allocation: None,
        });
        Ok(id)
    }

    fn view_root(&self, tensor: TensorId) -> Result<(TensorId, usize)> {
        let mut current = tensor;
        let mut offset = 0_usize;
        for _ in 0..=self.tensors.len() {
            let info = self.info(current)?;
            offset = offset
                .checked_add(info.offset)
                .ok_or_else(|| invalid("CPU view offset overflow"))?;
            if let Some(source) = info.view {
                current = source;
            } else {
                return Ok((current, offset));
            }
        }
        Err(invalid("cyclic CPU storage views"))
    }

    fn storage(&self, tensor: TensorId) -> Result<(TensorId, usize)> {
        let mut current = tensor;
        let mut offset = 0_usize;
        for _ in 0..=self.tensors.len() {
            let node = self
                .tensors
                .get(current.0)
                .ok_or_else(|| invalid("unknown CPU storage identity"))?;
            if let Some((arena, relative)) = node.allocation {
                offset = offset
                    .checked_add(relative)
                    .ok_or_else(|| invalid("CPU allocation offset overflow"))?;
                current = arena;
                continue;
            }
            let info = self.info(current)?;
            offset = offset
                .checked_add(info.offset)
                .ok_or_else(|| invalid("CPU view offset overflow"))?;
            if let Some(source) = info.view {
                current = source;
            } else {
                return Ok((current, offset));
            }
        }
        Err(invalid("cyclic CPU storage views"))
    }

    /// Allocate all non-view tensors owned by a context, preserving existing data.
    ///
    /// # Errors
    /// Rejects allocation failures, malformed views, and excessive storage.
    pub fn allocate(&mut self, context: u32) -> Result<()> {
        for id in self.context_tensors(context) {
            self.ensure_storage(id)?;
        }
        Ok(())
    }

    /// Place non-view tensors in one shared byte arena at explicit offsets.
    ///
    /// Placements may overlap. Numerical operations snapshot their inputs before
    /// writing their outputs, so legal input/output aliases remain well defined.
    /// Existing tensor payloads are discarded; initialize inputs after placement.
    ///
    /// # Errors
    /// Rejects unknown tensors, placements of existing views, overflowing bounds,
    /// and allocation failures.
    pub fn allocate_arena(&mut self, bytes: usize, placements: &[(TensorId, usize)]) -> Result<()> {
        for &(id, offset) in placements {
            let info = self.info(id)?;
            if info.view.is_some()
                || offset
                    .checked_add(info.bytes()?)
                    .is_none_or(|end| end > bytes)
            {
                return Err(invalid(
                    "CPU arena placement exceeds bounds or replaces a view",
                ));
            }
        }
        let arena = self.tensor(u32::MAX, 24, [bytes, 1, 1, 1])?;
        self.ensure_storage(arena)?;
        for &(id, offset) in placements {
            self.tensors[id.0].data.clear();
            self.tensors[id.0].allocation = Some((arena, offset));
        }
        Ok(())
    }

    /// Execute two independently owned copies and observe each numerical node.
    ///
    /// Both copies use the reference CPU kernel. This checks graph/alias behavior
    /// across backend instances; it does not provide an independent numerical
    /// implementation. Independent source references remain necessary for parity.
    ///
    /// # Errors
    /// Rejects graph/kernel/storage failures and propagates observer errors.
    pub fn compare(
        &mut self,
        graph: GraphId,
        mut observer: impl FnMut(usize, TensorId, &Self, &Self) -> Result<()>,
    ) -> Result<()> {
        let mut reference = Self::new();
        for node in &self.tensors {
            let mut data = Vec::new();
            data.try_reserve_exact(node.data.len())?;
            data.extend_from_slice(&node.data);
            reference.tensors.push(Node {
                info: node.info.clone(),
                data,
                allocation: node.allocation,
            });
        }
        let nodes = self.graph_nodes(graph)?.0.to_vec();
        for (index, tensor) in nodes.into_iter().enumerate() {
            if !self.supports(tensor)? {
                return Err(invalid("unsupported CPU comparison operation"));
            }
            self.evaluate(tensor)?;
            reference.evaluate(tensor)?;
            observer(index, tensor, self, &reference)?;
        }
        Ok(())
    }

    fn ensure_storage(&mut self, tensor: TensorId) -> Result<()> {
        let bytes = self.info(tensor)?.bytes()?;
        let (root, offset) = self.storage(tensor)?;
        let extent = offset
            .checked_add(bytes)
            .ok_or_else(|| invalid("CPU storage extent overflow"))?;
        let root_bytes = self.info(root)?.bytes()?;
        if extent > root_bytes {
            return Err(invalid("CPU view exceeds its underlying tensor"));
        }
        let data = &mut self.tensors[root.0].data;
        if data.len() < root_bytes {
            data.try_reserve_exact(root_bytes - data.len())?;
            data.resize(root_bytes, 0);
        }
        Ok(())
    }

    /// Copy raw bytes into tensor storage, including a view's underlying storage.
    ///
    /// # Errors
    /// Rejects allocation failures or an out-of-range byte extent.
    pub fn set(&mut self, tensor: TensorId, offset: usize, data: &[u8]) -> Result<()> {
        self.ensure_storage(tensor)?;
        let bytes = self.info(tensor)?.bytes()?;
        if offset > bytes || data.len() > bytes - offset {
            return Err(invalid("CPU tensor write exceeds extent"));
        }
        let (root, base) = self.storage(tensor)?;
        self.tensors[root.0].data[base + offset..base + offset + data.len()].copy_from_slice(data);
        Ok(())
    }

    /// Read raw bytes from allocated tensor storage.
    ///
    /// # Errors
    /// Rejects unallocated tensors and out-of-range byte extents.
    pub fn get(&self, tensor: TensorId, offset: usize, bytes: usize) -> Result<&[u8]> {
        let extent = self.info(tensor)?.bytes()?;
        if offset > extent || bytes > extent - offset {
            return Err(invalid("CPU tensor read exceeds extent"));
        }
        let (root, base) = self.storage(tensor)?;
        self.tensors[root.0]
            .data
            .get(base + offset..base + offset + bytes)
            .ok_or_else(|| invalid("CPU tensor has no allocated storage"))
    }

    /// Decode all logical values in contiguous logical order, respecting strides.
    ///
    /// # Errors
    /// Rejects malformed storage, unsupported types, or allocation failure.
    pub fn values(&self, tensor: TensorId) -> Result<Vec<f32>> {
        let info = self.info(tensor)?;
        let layout = type_layout(info.kind)?;
        let mut values = Vec::new();
        values.try_reserve_exact(info.elements())?;
        for row in 0..info.rows() {
            let index = coordinates(row, [info.shape[1], info.shape[2], info.shape[3], 1]);
            let row_offset = index[0] * info.strides[1]
                + index[1] * info.strides[2]
                + index[2] * info.strides[3];
            if layout.block != 1 {
                let mut packed = Vec::new();
                packed.try_reserve_exact(info.shape[0] / layout.block * layout.bytes)?;
                for block in 0..info.shape[0] / layout.block {
                    packed.extend_from_slice(self.get(
                        tensor,
                        row_offset + block * info.strides[0],
                        layout.bytes,
                    )?);
                }
                let packed = crate::model::Tensor {
                    name: info.name.clone(),
                    kind: info.kind as u32,
                    dimensions: vec![info.shape[0]],
                    data: std::sync::Arc::from(packed),
                };
                values.extend(crate::kernels::row(&packed, 0)?);
                continue;
            }
            for column in 0..info.shape[0] {
                let bytes =
                    self.get(tensor, row_offset + column * info.strides[0], layout.bytes)?;
                values.push(match info.kind {
                    0 => f32::from_le_bytes(bytes.try_into()?),
                    1 => half_to_float(u16::from_le_bytes(bytes.try_into()?)),
                    24 => f32::from(i8::from_le_bytes(bytes.try_into()?)),
                    25 => f32::from(i16::from_le_bytes(bytes.try_into()?)),
                    26 => i32::from_le_bytes(bytes.try_into()?) as f32,
                    27 => i64::from_le_bytes(bytes.try_into()?) as f32,
                    28 => f64::from_le_bytes(bytes.try_into()?) as f32,
                    30 => f32::from_bits(u32::from(u16::from_le_bytes(bytes.try_into()?)) << 16),
                    _ => return Err(invalid("unsupported CPU scalar decoding")),
                });
            }
        }
        Ok(values)
    }

    /// Decode integer tensors without passing through floating point.
    ///
    /// # Errors
    /// Rejects noninteger types, unknown identities, malformed storage, and
    /// allocation failures. All signed 64-bit values retain their exact bits.
    pub fn integer_values(&self, tensor: TensorId) -> Result<Vec<i64>> {
        let info = self.info(tensor)?;
        if !(24..=27).contains(&info.kind) {
            return Err(invalid("CPU integer access requires an integer tensor"));
        }
        let width = type_layout(info.kind)?.bytes;
        let mut values = Vec::new();
        values.try_reserve_exact(info.elements())?;
        for index in 0..info.elements() {
            let coordinate = coordinates(index, info.shape);
            let offset: usize = coordinate
                .iter()
                .zip(info.strides)
                .map(|(&index, stride)| index * stride)
                .sum();
            let bytes = self.get(tensor, offset, width)?;
            values.push(match info.kind {
                24 => i64::from(i8::from_le_bytes(bytes.try_into()?)),
                25 => i64::from(i16::from_le_bytes(bytes.try_into()?)),
                26 => i64::from(i32::from_le_bytes(bytes.try_into()?)),
                27 => i64::from_le_bytes(bytes.try_into()?),
                _ => return Err(invalid("CPU integer kind differs")),
            });
        }
        Ok(values)
    }

    fn copy_storage(&mut self, destination: TensorId, source: TensorId) -> Result<()> {
        let input = self.info(source)?.clone();
        let output = self.info(destination)?.clone();
        if input.kind != output.kind || input.elements() != output.elements() {
            return Err(invalid("CPU bit-preserving copy shapes or types differ"));
        }
        let layout = type_layout(input.kind)?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(input.elements() / layout.block * layout.bytes)?;
        let mut shape = input.shape;
        shape[0] /= layout.block;
        for index in 0..input.elements() / layout.block {
            let coordinate = coordinates(index, shape);
            let offset: usize = coordinate
                .iter()
                .zip(input.strides)
                .map(|(&index, stride)| index * stride)
                .sum();
            bytes.extend_from_slice(self.get(source, offset, layout.bytes)?);
        }
        shape = output.shape;
        shape[0] /= layout.block;
        for (index, bytes) in bytes.chunks_exact(layout.bytes).enumerate() {
            let coordinate = coordinates(index, shape);
            let offset: usize = coordinate
                .iter()
                .zip(output.strides)
                .map(|(&index, stride)| index * stride)
                .sum();
            self.set(destination, offset, bytes)?;
        }
        Ok(())
    }

    fn matrix(
        &self,
        tensor: TensorId,
        batch2: usize,
        batch3: usize,
    ) -> Result<crate::model::Tensor> {
        let info = self.info(tensor)?;
        let layout = type_layout(info.kind)?;
        if batch2 >= info.shape[2] || batch3 >= info.shape[3] {
            return Err(invalid("CPU matrix batch exceeds tensor extent"));
        }
        let mut packed = Vec::new();
        packed.try_reserve_exact(info.shape[0] / layout.block * info.shape[1] * layout.bytes)?;
        for row in 0..info.shape[1] {
            for block in 0..info.shape[0] / layout.block {
                let offset = block * info.strides[0]
                    + row * info.strides[1]
                    + batch2 * info.strides[2]
                    + batch3 * info.strides[3];
                packed.extend_from_slice(self.get(tensor, offset, layout.bytes)?);
            }
        }
        Ok(crate::model::Tensor {
            name: info.name.clone(),
            kind: u32::try_from(info.kind)?,
            dimensions: vec![info.shape[0], info.shape[1]],
            data: std::sync::Arc::from(packed),
        })
    }

    /// Replace tensor values in logical order, applying its scalar or packed format.
    ///
    /// # Errors
    /// Rejects wrong element counts, unavailable encoders and invalid storage.
    pub fn set_values(&mut self, tensor: TensorId, values: &[f32]) -> Result<()> {
        let info = self.info(tensor)?.clone();
        let layout = type_layout(info.kind)?;
        if values.len() != info.elements() {
            return Err(invalid("CPU scalar output extent or type differs"));
        }
        self.ensure_storage(tensor)?;
        if info.elements() == 0 {
            return Ok(());
        }
        if layout.block != 1 {
            for (row, values) in values.chunks_exact(info.shape[0]).enumerate() {
                let row_indices =
                    coordinates(row, [info.shape[1], info.shape[2], info.shape[3], 1]);
                let row_offset = row_indices[0] * info.strides[1]
                    + row_indices[1] * info.strides[2]
                    + row_indices[2] * info.strides[3];
                let bytes = crate::kernels::quantize_row(u32::try_from(info.kind)?, values)?;
                for (block, bytes) in bytes.chunks_exact(layout.bytes).enumerate() {
                    self.set(tensor, row_offset + block * info.strides[0], bytes)?;
                }
            }
            return Ok(());
        }
        for (index, &value) in values.iter().enumerate() {
            let coordinate = coordinates(index, info.shape);
            let offset: usize = coordinate
                .iter()
                .zip(info.strides)
                .map(|(&index, stride)| index * stride)
                .sum();
            let bytes = match info.kind {
                0 => value.to_le_bytes().to_vec(),
                1 => crate::kernels::half_bits(value).to_le_bytes().to_vec(),
                24 => (value as i8).to_le_bytes().to_vec(),
                25 => (value as i16).to_le_bytes().to_vec(),
                26 => (value as i32).to_le_bytes().to_vec(),
                27 => (value as i64).to_le_bytes().to_vec(),
                28 => f64::from(value).to_le_bytes().to_vec(),
                30 => {
                    let bits = value.to_bits();
                    let rounded = if value.is_nan() {
                        (bits >> 16) | 0x40
                    } else {
                        bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16
                    };
                    (rounded as u16).to_le_bytes().to_vec()
                }
                _ => return Err(invalid("unsupported CPU scalar output type")),
            };
            self.set(tensor, offset, &bytes)?;
        }
        Ok(())
    }

    /// Return the topologically ordered numerical nodes and leaves of a graph.
    ///
    /// # Errors
    /// Rejects an unknown graph identity.
    pub fn graph_nodes(&self, graph: GraphId) -> Result<(&[TensorId], &[TensorId])> {
        let graph = self
            .graphs
            .get(graph.0)
            .ok_or_else(|| invalid("unknown CPU graph"))?;
        Ok((&graph.nodes, &graph.leaves))
    }

    fn expand(&mut self, graph: GraphId, tensor: TensorId) -> Result<()> {
        let graph_ref = self
            .graphs
            .get(graph.0)
            .ok_or_else(|| invalid("unknown CPU graph"))?;
        // Tensor identities are dense and bounded. Track unseen/active/complete
        // states directly, rather than repeatedly searching nodes and the path.
        let mut states = vec![0_u8; self.tensors.len()];
        for id in graph_ref.nodes.iter().chain(&graph_ref.leaves) {
            states[id.0] = 2;
        }
        // Explicit DFS frames retain source-slot order and postorder insertion
        // without a call-stack frame (or cloned TensorInfo) per dependency.
        let mut pending = vec![(tensor, 0)];
        while let Some(frame) = pending.last_mut() {
            let (tensor, next_source) = *frame;
            if states.get(tensor.0) == Some(&2) {
                pending.pop();
                continue;
            }
            let info = self.info(tensor)?;
            states[tensor.0] = 1;
            if let Some(&source) = info.sources.get(next_source) {
                frame.1 += 1;
                match states.get(source.0) {
                    Some(1) => return Err(invalid("cyclic or excessive CPU graph")),
                    Some(2) => {}
                    _ => pending.push((source, 0)),
                }
                continue;
            }
            let leaf = info.operation == 0 && info.flags & 16 == 0;
            let graph = &mut self.graphs[graph.0];
            if leaf {
                graph.leaves.push(tensor);
            } else {
                if graph.nodes.len() >= graph.capacity {
                    return Err(invalid("CPU graph exceeds requested node capacity"));
                }
                graph.nodes.push(tensor);
            }
            states[tensor.0] = 2;
            pending.pop();
        }
        Ok(())
    }

    /// Check whether a tensor's operation and storage formats have CPU kernels.
    ///
    /// # Errors
    /// Rejects an unknown tensor.
    pub fn supports(&self, tensor: TensorId) -> Result<bool> {
        let info = self.info(tensor)?;
        if matches!(info.kind, 16..=19 | 21 | 22 | 29)
            && matches!(info.operation, 1 | 34 | 35)
            && !info.sources.first().is_some_and(|&source| {
                self.info(source)
                    .is_ok_and(|source| source.kind == info.kind)
            })
        {
            // These formats can copy packed bytes, but have no encoder for a
            // numeric conversion from another storage type.
            return Ok(false);
        }
        if matches!(info.operation, 29 | 30 | 74)
            && !info
                .sources
                .get(usize::from(info.operation == 74))
                .is_some_and(|&source| {
                    self.info(source).is_ok_and(
                        |source| matches!(source.kind, 0..=3 | 6..=8 | 10..=23 | 29 | 30 | 34 | 35 | 39..=42 | 142 | 143),
                    )
                })
        {
            return Ok(false);
        }
        if info.kind == 28 && !matches!(info.operation, 0 | 1 | 34..=39) {
            return Ok(false);
        }
        if matches!(info.kind, 24..=27)
            && !matches!(info.operation, 0 | 1 | 18 | 19 | 34..=40 | 69 | 70)
        {
            return Ok(false);
        }
        Ok(ops::supports(info.operation)
            && (type_layout(info.kind)?.block == 1
                || info.operation == 4 && ops::packed_add1(info.kind)
                || matches!(info.operation, 0 | 36..=39)
                || matches!(info.operation, 1 | 34 | 35)))
    }

    /// Evaluate the graph in topological order using production CPU kernels.
    ///
    /// # Errors
    /// Rejects unsupported operations, malformed input geometry, and allocation
    /// errors. Existing input storage remains owned by this backend.
    pub fn compute(&mut self, graph: GraphId) -> Result<()> {
        let nodes = self.graph_nodes(graph)?.0.to_vec();
        for tensor in nodes {
            if !self.supports(tensor)? {
                return Err(invalid(format!(
                    "unsupported CPU operation {}",
                    self.info(tensor)?.operation
                )));
            }
            self.evaluate(tensor)?;
        }
        Ok(())
    }
}
