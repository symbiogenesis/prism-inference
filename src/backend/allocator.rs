//! Reusable graph memory planning with pinned GGML lifetime and chunk semantics.
//!
//! Adapted from ggml-alloc.c (MIT, upstream attribution in tools/prism/upstream/LICENSE).
//! Plans preserve tensor metadata, reuse dead parents only for allowed operations,
//! and retain buffer reservations across graphs without retaining graph identities.

use super::{CpuBackend, GraphId, TensorId, TensorInfo};
use crate::{Result, invalid};

/// Alignment and preferred maximum size of a physical buffer.
#[derive(Clone, Copy, Debug)]
pub struct BufferType {
    /// Byte alignment; must be a nonzero power of two.
    pub alignment: usize,
    /// Preferred chunk limit. Oversized tensors and the final chunk may exceed it.
    pub maximum: Option<usize>,
}

/// One physical storage buffer in an allocation result.
#[derive(Clone, Copy, Debug)]
pub struct Buffer {
    /// Index into the allocator's buffer types.
    pub kind: usize,
    /// Allocated byte extent, including retained capacity.
    pub bytes: usize,
}

/// A tensor address relative to one owned storage buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Placement {
    /// Buffer index in [`Allocation::buffers`].
    pub buffer: usize,
    /// Byte offset from that buffer's base.
    pub offset: usize,
}

/// A complete graph allocation, including external context buffers.
#[derive(Debug)]
pub struct Allocation {
    /// Physical buffers, in buffer-type and chunk order.
    pub buffers: Vec<Buffer>,
    /// Tensor addresses indexed by owned tensor identity. Unused tensors are absent.
    pub tensors: Vec<Option<Placement>>,
}

/// A graph allocator retaining physical buffer capacity across reservations.
pub struct GraphAllocator {
    types: Vec<BufferType>,
    retained: Vec<Vec<usize>>,
}

fn align(bytes: usize, alignment: usize) -> Result<usize> {
    bytes
        .checked_add(alignment - 1)
        .map(|n| n & !(alignment - 1))
        .ok_or_else(|| invalid("graph allocation alignment overflow"))
}

/// Whether the pinned operation permits an output to overwrite a dead input.
#[must_use]
pub const fn can_inplace(operation: i32) -> bool {
    matches!(operation, 2..=4 | 6..=11 | 23 | 25 | 26 | 32 | 44..=50 | 73 | 91)
}

#[derive(Clone, Copy, Default)]
struct Address {
    kind: usize,
    chunk: usize,
    offset: usize,
}
#[derive(Clone, Default)]
struct State {
    address: Option<Address>,
    owned: bool,
    children: usize,
    views: usize,
}
struct Chunk {
    free: Vec<(usize, usize)>,
    high: usize,
}
struct Dynamic {
    descriptor: BufferType,
    chunks: Vec<Chunk>,
}

impl Dynamic {
    fn allocate(&mut self, bytes: usize, kind: usize) -> Result<Address> {
        let size = align(bytes, self.descriptor.alignment)?;
        let mut best = None;
        // The native allocator searches every interior block before tails.
        for (chunk_id, chunk) in self.chunks.iter().enumerate() {
            let mut best_size = usize::MAX;
            for (block_id, &(_, available)) in chunk
                .free
                .iter()
                .enumerate()
                .take(chunk.free.len().saturating_sub(1))
            {
                if available >= size && available <= best_size {
                    best = Some((chunk_id, block_id));
                    best_size = available;
                }
            }
        }
        if best.is_none() {
            let mut reuse = i128::MIN;
            for (chunk_id, chunk) in self.chunks.iter().enumerate() {
                if let Some(&(offset, available)) = chunk.free.last() {
                    let factor = chunk.high as i128 - offset as i128 - size as i128;
                    if available >= size
                        && ((reuse < 0 && factor > reuse) || (factor >= 0 && factor < reuse))
                    {
                        best = Some((chunk_id, chunk.free.len() - 1));
                        reuse = factor;
                    }
                }
            }
        }
        let (chunk_id, block_id) = if let Some(best) = best {
            best
        } else {
            if self.chunks.len() >= 16 {
                return Err(invalid("graph allocator exhausted chunks"));
            }
            let capacity = if self.chunks.len() == 15 {
                usize::MAX / 2
            } else {
                self.descriptor.maximum.unwrap_or(usize::MAX / 2).max(size)
            };
            self.chunks.push(Chunk {
                free: vec![(0, capacity)],
                high: 0,
            });
            (self.chunks.len() - 1, 0)
        };
        let chunk = &mut self.chunks[chunk_id];
        let (offset, available) = chunk.free[block_id];
        let end = offset
            .checked_add(size)
            .ok_or_else(|| invalid("graph allocation extent overflow"))?;
        chunk.free[block_id] = (end, available - size);
        if available == size {
            chunk.free.remove(block_id);
        }
        chunk.high = chunk.high.max(end);
        Ok(Address {
            kind,
            chunk: chunk_id,
            offset,
        })
    }

    fn free(&mut self, address: Address, bytes: usize) -> Result<()> {
        let size = align(bytes, self.descriptor.alignment)?;
        if size == 0 {
            return Ok(());
        }
        let chunk = &mut self.chunks[address.chunk];
        chunk.free.push((address.offset, size));
        chunk.free.sort_unstable_by_key(|block| block.0);
        let mut merged: Vec<(usize, usize)> = Vec::new();
        for &(offset, extent) in &chunk.free {
            if let Some((previous, bytes)) = merged.last_mut() {
                let end = previous
                    .checked_add(*bytes)
                    .ok_or_else(|| invalid("graph free extent overflow"))?;
                if end == offset {
                    *bytes = bytes
                        .checked_add(extent)
                        .ok_or_else(|| invalid("graph free merge overflow"))?;
                    continue;
                }
                if end > offset {
                    return Err(invalid("graph allocator double free"));
                }
            }
            merged.push((offset, extent));
        }
        chunk.free = merged;
        Ok(())
    }
}

struct Planner<'a> {
    backend: &'a CpuBackend,
    states: Vec<State>,
    dynamic: Vec<Dynamic>,
}

impl Planner<'_> {
    fn allocate(&mut self, id: TensorId, kind: usize) -> Result<()> {
        let info = self.backend.info(id)?;
        if self.states[id.0].address.is_some() || info.view.is_some() {
            return Ok(());
        }
        self.states[id.0].owned = true;
        if can_inplace(info.operation) {
            for &parent in &info.sources {
                let parent_info = self.backend.info(parent)?;
                let parent_state = self.states[parent.0].clone();
                let root = parent_info.view.unwrap_or(parent);
                let root_info = self.backend.info(root)?;
                if !self.states[root.0].owned
                    || parent_info.flags & 2 != 0
                    || root_info.flags & 2 != 0
                    || !same_layout(info, parent_info)
                    || parent_state.children != 1
                    || parent_state.views != 0
                {
                    continue;
                }
                if parent_info.view.is_some()
                    && (self.states[root.0].views != 1
                        || self.states[root.0].children != 0
                        || parent_info.offset != 0)
                {
                    continue;
                }
                let address = self.states[root.0]
                    .address
                    .ok_or_else(|| invalid("graph reuse parent has no address"))?;
                self.states[id.0].address = Some(address);
                self.states[parent.0].owned = false;
                self.states[root.0].owned = false;
                let alignment = self.dynamic[address.kind].descriptor.alignment;
                let parent_bytes = align(root_info.bytes()?, alignment)?;
                let node_bytes = align(info.bytes()?, alignment)?;
                if parent_bytes > node_bytes {
                    self.dynamic[address.kind].free(
                        Address {
                            offset: address.offset + node_bytes,
                            ..address
                        },
                        parent_bytes - node_bytes,
                    )?;
                }
                return Ok(());
            }
        }
        self.states[id.0].address = Some(self.dynamic[kind].allocate(info.bytes()?, kind)?);
        Ok(())
    }

    fn release(&mut self, id: TensorId) -> Result<()> {
        let info = self.backend.info(id)?;
        if info.flags & 2 != 0 || !self.states[id.0].owned {
            return Ok(());
        }
        let address = self.states[id.0]
            .address
            .ok_or_else(|| invalid("graph free tensor has no address"))?;
        self.dynamic[address.kind].free(address, info.bytes()?)?;
        self.states[id.0].owned = false;
        Ok(())
    }
}

fn same_layout(left: &TensorInfo, right: &TensorInfo) -> bool {
    left.kind == right.kind && left.shape == right.shape && left.strides == right.strides
}

impl GraphAllocator {
    /// Create an allocator with one or more distinct buffer types.
    ///
    /// # Errors
    /// Rejects invalid alignment, zero maximum sizes, or an empty type list.
    pub fn new(types: Vec<BufferType>) -> Result<Self> {
        if types.is_empty()
            || types.len() > 256
            || types
                .iter()
                .any(|kind| !kind.alignment.is_power_of_two() || kind.maximum == Some(0))
        {
            return Err(invalid("invalid graph allocator buffer type"));
        }
        Ok(Self {
            retained: vec![Vec::new(); types.len()],
            types,
        })
    }

    /// Reserve graph storage with per-tensor buffer types and externally allocated contexts.
    ///
    /// External contexts remain live and are never reused by the graph allocator.
    /// The mapping is indexed by tensor identity; omitted entries select type zero.
    ///
    /// # Errors
    /// Rejects invalid graphs, buffer references, cyclic views, overflow, or resource failures.
    #[allow(
        clippy::too_many_lines,
        reason = "Keep the pinned allocate/count/execute/release stages in reviewable source order."
    )]
    pub fn reserve(
        &mut self,
        backend: &CpuBackend,
        graph: GraphId,
        kinds: &[usize],
        external: &[(u32, usize)],
    ) -> Result<Allocation> {
        let (nodes, leaves) = backend.graph_nodes(graph)?;
        let count = backend.tensors.len();
        let kind_for = |id: TensorId| -> Result<usize> {
            let kind = kinds.get(id.0).copied().unwrap_or(0);
            if kind >= self.types.len() {
                return Err(invalid("graph tensor buffer type is unknown"));
            }
            Ok(kind)
        };
        let mut planner = Planner {
            backend,
            states: vec![State::default(); count],
            dynamic: self
                .types
                .iter()
                .map(|&descriptor| Dynamic {
                    descriptor,
                    chunks: Vec::new(),
                })
                .collect(),
        };
        let mut allocation = Allocation {
            buffers: Vec::new(),
            tensors: vec![None; count],
        };
        let mut external_roots = vec![false; count];
        for &(context, kind) in external {
            let descriptor = *self
                .types
                .get(kind)
                .ok_or_else(|| invalid("external graph buffer type is unknown"))?;
            let mut dynamic = Dynamic {
                descriptor,
                chunks: Vec::new(),
            };
            let mut entries = Vec::new();
            for id in backend.context_tensors(context) {
                if backend.info(id)?.view.is_none() {
                    entries.push((id, dynamic.allocate(backend.info(id)?.bytes()?, kind)?));
                    external_roots[id.0] = true;
                }
            }
            let first = allocation.buffers.len();
            allocation
                .buffers
                .extend(dynamic.chunks.iter().map(|chunk| Buffer {
                    kind,
                    bytes: chunk.high,
                }));
            for (id, address) in entries {
                allocation.tensors[id.0] = Some(Placement {
                    buffer: first + address.chunk,
                    offset: address.offset,
                });
            }
        }
        // External storage must be recognizable as allocated but never owned.
        for (index, external) in external_roots.iter().copied().enumerate() {
            if external {
                planner.states[index].address = Some(Address::default());
            }
        }
        for &leaf in leaves {
            planner.allocate(leaf, kind_for(leaf)?)?;
        }
        for &node in nodes {
            let info = backend.info(node)?;
            if let Some(root) = info.view {
                planner.states[root.0].views += 1;
            }
            if info.flags & 1 != 0 {
                planner.allocate(node, kind_for(node)?)?;
            }
            for &source in &info.sources {
                planner.states[source.0].children += 1;
                if backend.info(source)?.flags & 1 != 0 {
                    planner.allocate(source, kind_for(node)?)?;
                }
            }
        }
        for &node in nodes {
            let info = backend.info(node)?;
            let kind = kind_for(node)?;
            for &source in &info.sources {
                planner.allocate(source, kind)?;
            }
            planner.allocate(node, kind)?;
            for &source in &info.sources {
                let state = &mut planner.states[source.0];
                state.children = state
                    .children
                    .checked_sub(1)
                    .ok_or_else(|| invalid("graph lifetime count underflow"))?;
                if state.children != 0 || state.views != 0 {
                    continue;
                }
                if let Some(root) = backend.info(source)?.view {
                    planner.states[root.0].views = planner.states[root.0]
                        .views
                        .checked_sub(1)
                        .ok_or_else(|| invalid("graph view lifetime underflow"))?;
                    if planner.states[root.0].children == 0 && planner.states[root.0].views == 0 {
                        planner.release(root)?;
                    }
                } else {
                    planner.release(source)?;
                }
            }
        }
        let mut buffer_ids = Vec::new();
        for (kind, dynamic) in planner.dynamic.iter().enumerate() {
            let needed: Vec<usize> = dynamic.chunks.iter().map(|chunk| chunk.high).collect();
            let retained = &mut self.retained[kind];
            if needed
                .iter()
                .enumerate()
                .any(|(index, &bytes)| bytes > retained.get(index).copied().unwrap_or(0))
            {
                *retained = needed;
            }
            // GGML creates one zero-size buffer for an unused type as well.
            if retained.is_empty() {
                retained.push(0);
            }
            let first = allocation.buffers.len();
            buffer_ids.push(first);
            allocation
                .buffers
                .extend(retained.iter().map(|&bytes| Buffer { kind, bytes }));
        }
        for (index, state) in planner.states.iter().enumerate() {
            if external_roots[index] || backend.info(TensorId(index))?.view.is_some() {
                continue;
            }
            if let Some(address) = state.address {
                allocation.tensors[index] = Some(Placement {
                    buffer: buffer_ids[address.kind] + address.chunk,
                    offset: address.offset,
                });
            }
        }
        for &id in leaves.iter().chain(nodes) {
            if backend.info(id)?.view.is_some() {
                let (root, offset) = backend.view_root(id)?;
                let parent = allocation.tensors[root.0]
                    .ok_or_else(|| invalid("graph view root has no storage"))?;
                let offset = parent
                    .offset
                    .checked_add(offset)
                    .ok_or_else(|| invalid("graph view placement overflow"))?;
                if offset
                    .checked_add(backend.info(id)?.bytes()?)
                    .is_none_or(|end| end > allocation.buffers[parent.buffer].bytes)
                {
                    return Err(invalid("graph view exceeds allocation"));
                }
                allocation.tensors[id.0] = Some(Placement {
                    buffer: parent.buffer,
                    offset,
                });
            }
        }
        Ok(allocation)
    }

    /// Bind a reserved graph to actual owned byte arenas without changing tensor views.
    /// Initialize graph inputs after this operation, since previous data is discarded.
    ///
    /// # Errors
    /// Rejects malformed placements, excessive storage, and allocation failures.
    pub fn allocate(backend: &mut CpuBackend, allocation: &Allocation) -> Result<()> {
        for (buffer, description) in allocation.buffers.iter().enumerate() {
            let placements = allocation
                .tensors
                .iter()
                .enumerate()
                .filter_map(|(index, placement)| {
                    placement
                        .filter(|placement| {
                            placement.buffer == buffer && backend.tensors[index].info.view.is_none()
                        })
                        .map(|placement| (TensorId(index), placement.offset))
                })
                .collect::<Vec<_>>();
            backend.allocate_arena(description.bytes, &placements)?;
        }
        Ok(())
    }
}
