//! Independent graph-allocator, chunked-buffer and liveness observations.
//!
//! No production Rust or native allocator API or expected-answer fixture is
//! used. Immutable upstream MIT source identity is disclosed by `identity`.

use serde::{Deserialize, Deserializer};
use serde_json::{Value, json};

use super::{Result, invalid};

/// Supported upstream assertion source identifier for `tests/test-alloc.cpp`.
pub(super) const ID: &str = "tests/test-alloc.cpp";

const MAX_OPERATIONS: usize = 128;
const MAX_BACKENDS: usize = 3;
const MAX_STEPS: usize = 4;
const MAX_NODES: usize = 40;
const MAX_ALIGNMENT: u64 = 64;
const MAX_BUFFER_SIZE: u64 = 1_048_576;
const MAX_EXTENT: usize = 1024;
const MAX_ELEMENTS: usize = 16_384;
const MAX_OFFSET_BYTES: u64 = 65_536;

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
    backends: Vec<BackendInput>,
    steps: Vec<StepInput>,
}

fn nullable_u64<'de, D: Deserializer<'de>>(input: D) -> std::result::Result<Option<u64>, D::Error> {
    Option::deserialize(input)
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
struct BackendInput {
    #[serde(deserialize_with = "nullable_u64")]
    max_buffer_size: Option<u64>,
    alignment: u64,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
struct ExternalInput {
    context: usize,
    buffer_type: usize,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
struct StepInput {
    nodes: Vec<NodeInput>,
    output: usize,
    buffer_types: Vec<usize>,
    external_allocations: Vec<ExternalInput>,
    reserve: bool,
}

#[derive(Deserialize)]
#[serde(tag = "Op", rename_all_fields = "PascalCase", deny_unknown_fields)]
enum NodeInput {
    #[serde(rename = "input")]
    Input { context: usize, elements: usize },
    #[serde(rename = "add")]
    Add { context: usize, inputs: Vec<usize> },
    #[serde(rename = "mul")]
    Mul { context: usize, inputs: Vec<usize> },
    #[serde(rename = "concat")]
    Concat { context: usize, inputs: Vec<usize> },
    #[serde(rename = "mean")]
    Mean { context: usize, inputs: Vec<usize> },
    #[serde(rename = "scale")]
    Scale {
        context: usize,
        inputs: Vec<usize>,
        factor: f32,
    },
    #[serde(rename = "pad")]
    Pad {
        context: usize,
        inputs: Vec<usize>,
        amount: usize,
    },
    #[serde(rename = "reshape")]
    Reshape {
        context: usize,
        inputs: Vec<usize>,
        dimensions: Vec<usize>,
    },
    #[serde(rename = "permute")]
    Permute {
        context: usize,
        inputs: Vec<usize>,
        axes: Vec<usize>,
    },
    #[serde(rename = "view")]
    View {
        context: usize,
        inputs: Vec<usize>,
        elements: usize,
        offset_bytes: u64,
    },
}

impl NodeInput {
    const fn context(&self) -> usize {
        match *self {
            Self::Input { context, .. }
            | Self::Add { context, .. }
            | Self::Mul { context, .. }
            | Self::Concat { context, .. }
            | Self::Mean { context, .. }
            | Self::Scale { context, .. }
            | Self::Pad { context, .. }
            | Self::Reshape { context, .. }
            | Self::Permute { context, .. }
            | Self::View { context, .. } => context,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Op {
    Input,
    Add,
    Mul,
    Concat,
    Mean,
    Scale,
    Pad,
    Reshape,
    Permute,
    View,
}

impl Op {
    const fn code(self) -> u32 {
        match self {
            Self::Input => 0,
            Self::Add => 2,
            Self::Mul => 7,
            Self::Mean => 17,
            Self::Concat => 22,
            Self::Scale => 35,
            Self::Reshape => 56,
            Self::View => 57,
            Self::Permute => 58,
            Self::Pad => 64,
        }
    }

    const fn can_inplace(self) -> bool {
        matches!(self, Self::Add | Self::Mul | Self::Scale)
    }
}

#[derive(Clone, Copy)]
struct BackendSpec {
    alignment: u64,
    max_size: Option<u64>,
}

impl BackendSpec {
    const fn max_limit(self) -> u64 {
        match self.max_size {
            Some(limit) => limit,
            None => u64::MAX,
        }
    }
}

struct TensorDesc {
    op: Op,
    context: usize,
    bytes: u64,
    view_src: Option<usize>,
    view_offs: u64,
    sources: [Option<usize>; 10],
    flags: u32,
}

struct ValidatedStep {
    tensors: Vec<TensorDesc>,
    output: usize,
    buffer_types: Vec<usize>,
    external: Vec<(usize, usize)>,
    reserve: bool,
}

struct ValidatedOperation {
    backends: Vec<BackendSpec>,
    steps: Vec<ValidatedStep>,
}

fn require(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(invalid(message))
    }
}

fn elements_count(dimensions: &[usize; 4]) -> Result<usize> {
    let mut product = 1_usize;
    for &dimension in dimensions {
        require(
            dimension > 0 && dimension <= MAX_EXTENT && product <= MAX_ELEMENTS / dimension,
            "allocation tensor extent exceeds bound",
        )?;
        product *= dimension;
    }
    Ok(product)
}

fn validate_backends(backends: &[BackendInput]) -> Result<Vec<BackendSpec>> {
    require(
        !backends.is_empty() && backends.len() <= MAX_BACKENDS,
        "allocation backend count outside bound",
    )?;
    backends
        .iter()
        .map(|backend| {
            require(
                backend.alignment > 0
                    && backend.alignment <= MAX_ALIGNMENT
                    && backend.alignment.is_power_of_two(),
                "allocation alignment must be a power of two within bound",
            )?;
            if let Some(maximum) = backend.max_buffer_size {
                require(
                    maximum > 0 && maximum <= MAX_BUFFER_SIZE,
                    "allocation maximum buffer size outside bound",
                )?;
            }
            Ok(BackendSpec {
                alignment: backend.alignment,
                max_size: backend.max_buffer_size,
            })
        })
        .collect()
}

type ExternalAllocations = (Vec<(usize, usize)>, [bool; 3]);
type UnaryValidation = (Op, [usize; 4], Option<usize>, u64, [Option<usize>; 10]);
type ExternalPlacements = (Vec<(usize, u64)>, Vec<Option<(usize, u64)>>);

fn validate_external(
    external: &[ExternalInput],
    backend_count: usize,
) -> Result<ExternalAllocations> {
    require(
        external.len() <= 2,
        "allocation external context count outside bound",
    )?;
    let mut seen = [false; 3];
    let mut mapped = Vec::with_capacity(external.len());
    for item in external {
        require(
            (1..=2).contains(&item.context) && !seen[item.context],
            "external allocation context must be unique and nonzero",
        )?;
        require(
            item.buffer_type < backend_count,
            "external allocation buffer type out of range",
        )?;
        seen[item.context] = true;
        mapped.push((item.context, item.buffer_type));
    }
    Ok((mapped, seen))
}

fn preceding_shape(shapes: &[[usize; 4]], index: usize, input: usize) -> Result<&[usize; 4]> {
    require(input < index, "allocation node input must precede node")?;
    shapes
        .get(input)
        .ok_or_else(|| invalid("allocation input shape missing"))
}

fn validate_binary(
    op: Op,
    inputs: &[usize],
    index: usize,
    shapes: &[[usize; 4]],
) -> Result<([usize; 4], [Option<usize>; 10])> {
    require(inputs.len() == 2, "allocation binary input count mismatch")?;
    let left = *preceding_shape(shapes, index, inputs[0])?;
    let right = *preceding_shape(shapes, index, inputs[1])?;
    let mut dims = left;
    if op == Op::Concat {
        require(
            left[1..] == right[1..],
            "allocation concat dimensions mismatch",
        )?;
        dims[0] = left[0]
            .checked_add(right[0])
            .ok_or_else(|| invalid("allocation concat extent overflow"))?;
    } else {
        require(left == right, "allocation elementwise dimensions mismatch")?;
    }
    let mut sources = [None; 10];
    sources[0] = Some(inputs[0]);
    sources[1] = Some(inputs[1]);
    Ok((dims, sources))
}

fn validate_reshape_permute_view(
    node: &NodeInput,
    source: usize,
    orig: [usize; 4],
    descs: &[TensorDesc],
) -> Result<(Op, [usize; 4], usize, u64)> {
    let parent = &descs[source];
    let root = parent.view_src.unwrap_or(source);
    match node {
        NodeInput::Reshape { dimensions, .. } => {
            require(
                dimensions.len() == 2,
                "allocation reshape requires two dimensions",
            )?;
            let count = elements_count(&orig)?;
            let dims = [dimensions[0], dimensions[1], 1, 1];
            require(
                elements_count(&dims)? == count,
                "allocation reshape element count mismatch",
            )?;
            Ok((Op::Reshape, dims, root, parent.view_offs))
        }
        NodeInput::Permute { axes, .. } => {
            require(axes.len() == 4, "allocation permutation requires four axes")?;
            let mut seen = [false; 4];
            let mut dims = [0_usize; 4];
            for (&dest, &extent) in axes.iter().zip(&orig) {
                require(
                    dest < 4 && !seen[dest],
                    "allocation permutation axes must be distinct",
                )?;
                seen[dest] = true;
                dims[dest] = extent;
            }
            Ok((Op::Permute, dims, root, parent.view_offs))
        }
        NodeInput::View {
            elements,
            offset_bytes,
            ..
        } => {
            let source_bytes = u64::try_from(elements_count(&orig)?)?
                .checked_mul(4)
                .ok_or_else(|| invalid("allocation source byte extent overflow"))?;
            let view_bytes = u64::try_from(*elements)?
                .checked_mul(4)
                .ok_or_else(|| invalid("allocation view byte extent overflow"))?;
            require(
                *elements > 0
                    && *elements <= MAX_EXTENT
                    && *offset_bytes <= MAX_OFFSET_BYTES
                    && offset_bytes.is_multiple_of(4)
                    && offset_bytes
                        .checked_add(view_bytes)
                        .is_some_and(|end| end <= source_bytes),
                "allocation view exceeds source extent",
            )?;
            let view_offs = parent
                .view_offs
                .checked_add(*offset_bytes)
                .ok_or_else(|| invalid("allocation view offset overflow"))?;
            Ok((Op::View, [*elements, 1, 1, 1], root, view_offs))
        }
        _ => Err(invalid("unexpected allocation view variant")),
    }
}

fn validate_unary(
    node: &NodeInput,
    index: usize,
    shapes: &[[usize; 4]],
    descs: &[TensorDesc],
) -> Result<UnaryValidation> {
    let (NodeInput::Mean { inputs, .. }
    | NodeInput::Scale { inputs, .. }
    | NodeInput::Pad { inputs, .. }
    | NodeInput::Reshape { inputs, .. }
    | NodeInput::Permute { inputs, .. }
    | NodeInput::View { inputs, .. }) = node
    else {
        return Err(invalid("unexpected allocation unary node"));
    };
    require(inputs.len() == 1, "allocation unary input count mismatch")?;
    let source = inputs[0];
    let orig = *preceding_shape(shapes, index, source)?;
    let mut sources = [None; 10];
    sources[0] = Some(source);
    match node {
        NodeInput::Mean { .. } => {
            let mut dims = orig;
            dims[0] = 1;
            Ok((Op::Mean, dims, None, 0, sources))
        }
        NodeInput::Scale { factor, .. } => {
            require(factor.is_finite(), "allocation scale Factor must be finite")?;
            Ok((Op::Scale, orig, None, 0, sources))
        }
        NodeInput::Pad { amount, .. } => {
            require(*amount <= MAX_EXTENT, "allocation pad amount outside bound")?;
            let mut dims = orig;
            dims[0] = dims[0]
                .checked_add(*amount)
                .ok_or_else(|| invalid("allocation pad extent overflow"))?;
            Ok((Op::Pad, dims, None, 0, sources))
        }
        _ => {
            let (op, dims, root, offs) = validate_reshape_permute_view(node, source, orig, descs)?;
            Ok((op, dims, Some(root), offs, sources))
        }
    }
}

fn validate_node(
    node: &NodeInput,
    index: usize,
    external_present: [bool; 3],
    shapes: &mut Vec<[usize; 4]>,
    descs: &mut Vec<TensorDesc>,
) -> Result<()> {
    let context = node.context();
    require(
        context <= 2 && (context == 0 || external_present[context]),
        "allocation node belongs to unallocated external context",
    )?;
    let (op, dims, view_src, view_offs, sources, flags) = match node {
        NodeInput::Input { elements, .. } => {
            (Op::Input, [*elements, 1, 1, 1], None, 0, [None; 10], 1)
        }
        NodeInput::Add { inputs, .. } => {
            require(context == 0, "allocation operations require context zero")?;
            let (dims, sources) = validate_binary(Op::Add, inputs, index, shapes)?;
            (Op::Add, dims, None, 0, sources, 0)
        }
        NodeInput::Mul { inputs, .. } => {
            require(context == 0, "allocation operations require context zero")?;
            let (dims, sources) = validate_binary(Op::Mul, inputs, index, shapes)?;
            (Op::Mul, dims, None, 0, sources, 0)
        }
        NodeInput::Concat { inputs, .. } => {
            require(context == 0, "allocation operations require context zero")?;
            let (dims, sources) = validate_binary(Op::Concat, inputs, index, shapes)?;
            (Op::Concat, dims, None, 0, sources, 0)
        }
        _ => {
            require(context == 0, "allocation operations require context zero")?;
            let (op, dims, view_src, view_offs, sources) =
                validate_unary(node, index, shapes, descs)?;
            (op, dims, view_src, view_offs, sources, 0)
        }
    };
    let count = elements_count(&dims)?;
    let bytes = u64::try_from(count)?
        .checked_mul(4)
        .ok_or_else(|| invalid("allocation tensor byte count overflow"))?;
    shapes.push(dims);
    descs.push(TensorDesc {
        op,
        context,
        bytes,
        view_src,
        view_offs,
        sources,
        flags,
    });
    Ok(())
}

fn validate_step(step: &StepInput, backend_count: usize) -> Result<ValidatedStep> {
    require(
        !step.nodes.is_empty() && step.nodes.len() <= MAX_NODES,
        "allocation graph node count outside bound",
    )?;
    require(
        step.output < step.nodes.len(),
        "allocation step output index out of range",
    )?;
    require(
        step.buffer_types.len() == step.nodes.len(),
        "allocation buffer type mapping extent mismatch",
    )?;
    for &kind in &step.buffer_types {
        require(
            kind < backend_count,
            "allocation buffer type index out of range",
        )?;
    }
    let (external, external_present) =
        validate_external(&step.external_allocations, backend_count)?;
    let mut shapes = Vec::with_capacity(step.nodes.len());
    let mut tensors = Vec::with_capacity(step.nodes.len());
    for (index, node) in step.nodes.iter().enumerate() {
        validate_node(node, index, external_present, &mut shapes, &mut tensors)?;
    }
    tensors[step.output].flags |= 2;
    Ok(ValidatedStep {
        tensors,
        output: step.output,
        buffer_types: step.buffer_types.clone(),
        external,
        reserve: step.reserve,
    })
}

fn validate_operation(operation: &Operation) -> Result<ValidatedOperation> {
    require(
        operation.kind == "allocation-graph",
        "unknown allocation Kind",
    )?;
    let backends = validate_backends(&operation.backends)?;
    require(
        !operation.steps.is_empty() && operation.steps.len() <= MAX_STEPS,
        "allocation step count outside bound",
    )?;
    let steps = operation
        .steps
        .iter()
        .map(|step| validate_step(step, backends.len()))
        .collect::<Result<Vec<_>>>()?;
    Ok(ValidatedOperation { backends, steps })
}

struct Graph {
    nodes: Vec<usize>,
    leaves: Vec<usize>,
    visited: Vec<bool>,
}

fn visit(tensors: &[TensorDesc], index: usize, graph: &mut Graph) -> Result<()> {
    let slot = graph
        .visited
        .get_mut(index)
        .ok_or_else(|| invalid("allocation graph index out of bounds"))?;
    if *slot {
        return Ok(());
    }
    *slot = true;
    let tensor = tensors
        .get(index)
        .ok_or_else(|| invalid("allocation tensor descriptor missing"))?;
    if let Some(view_src) = tensor.view_src {
        visit(tensors, view_src, graph)?;
    }
    for source in tensor.sources.into_iter().flatten() {
        visit(tensors, source, graph)?;
    }
    if tensor.op == Op::Input {
        graph.leaves.push(index);
    } else {
        graph.nodes.push(index);
    }
    Ok(())
}

fn build_graph(tensors: &[TensorDesc], output: usize) -> Result<Graph> {
    let mut graph = Graph {
        nodes: Vec::new(),
        leaves: Vec::new(),
        visited: vec![false; tensors.len()],
    };
    visit(tensors, output, &mut graph)?;
    Ok(graph)
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    value
        .div_ceil(alignment)
        .checked_mul(alignment)
        .ok_or_else(|| invalid("allocation alignment overflow"))
}

fn allocate_external(step: &ValidatedStep, backends: &[BackendSpec]) -> Result<ExternalPlacements> {
    let mut buffers = Vec::new();
    let mut placements = vec![None; step.tensors.len()];
    for &(context, buft) in &step.external {
        let spec = backends[buft];
        let max_size = spec.max_limit();
        let mut current_size = 0_u64;
        let mut entries: Vec<(usize, u64)> = Vec::new();
        for (index, tensor) in step.tensors.iter().enumerate() {
            if tensor.context != context {
                continue;
            }
            let offset = align_up(current_size, spec.alignment)?;
            let end = offset
                .checked_add(tensor.bytes)
                .ok_or_else(|| invalid("external buffer extent overflow"))?;
            if entries.is_empty() || end <= max_size {
                entries.push((index, offset));
                current_size = end;
            } else {
                let id = buffers.len();
                buffers.push((buft, current_size));
                for (tensor_idx, placed_off) in std::mem::take(&mut entries) {
                    placements[tensor_idx] = Some((id, placed_off));
                }
                entries.push((index, 0));
                current_size = tensor.bytes;
            }
        }
        if entries.is_empty() {
            return Err(invalid("production external context allocation failed"));
        }
        let id = buffers.len();
        buffers.push((buft, current_size));
        for (tensor_idx, placed_off) in entries {
            placements[tensor_idx] = Some((id, placed_off));
        }
    }
    Ok((buffers, placements))
}

#[derive(Clone, Copy)]
struct FreeBlock {
    offset: u64,
    size: u64,
}

struct Chunk {
    free_blocks: Vec<FreeBlock>,
    max_size: u64,
}

struct DynAlloc {
    alignment: u64,
    max_size: u64,
    chunks: Vec<Chunk>,
}

impl DynAlloc {
    const fn new(spec: BackendSpec) -> Self {
        Self {
            alignment: spec.alignment,
            max_size: spec.max_limit(),
            chunks: Vec::new(),
        }
    }

    fn find_within_limit(&self, bytes: u64) -> Option<(usize, usize)> {
        let mut best: Option<(u64, usize, usize)> = None;
        for (ci, chunk) in self.chunks.iter().enumerate() {
            for (bi, block) in chunk.free_blocks.iter().enumerate() {
                let fits = block.size >= bytes
                    && block
                        .offset
                        .checked_add(bytes)
                        .is_some_and(|end| end <= chunk.max_size);
                if fits && best.is_none_or(|(best_size, _, _)| block.size < best_size) {
                    best = Some((block.size, ci, bi));
                }
            }
        }
        best.map(|(_, ci, bi)| (ci, bi))
    }

    fn find_with_growth(&self, bytes: u64) -> Option<(usize, usize)> {
        let mut best: Option<(u64, u64, usize, usize)> = None;
        for (ci, chunk) in self.chunks.iter().enumerate() {
            for (bi, block) in chunk.free_blocks.iter().enumerate() {
                if block.size < bytes {
                    continue;
                }
                let added = block
                    .offset
                    .saturating_add(bytes)
                    .saturating_sub(chunk.max_size);
                let key = (added, block.size, ci, bi);
                if best.is_none_or(|current| key < current) {
                    best = Some(key);
                }
            }
        }
        best.map(|(_, _, ci, bi)| (ci, bi))
    }

    fn alloc(&mut self, bytes: u64) -> Result<(usize, u64, u64)> {
        let size_aligned = align_up(bytes, self.alignment)?;
        let candidate = self
            .find_within_limit(bytes)
            .or_else(|| self.find_with_growth(bytes));
        if let Some((ci, bi)) = candidate {
            let chunk = &mut self.chunks[ci];
            let block = chunk.free_blocks[bi];
            let consumed = block.size.min(size_aligned);
            if block.size > consumed {
                chunk.free_blocks[bi] = FreeBlock {
                    offset: block
                        .offset
                        .checked_add(consumed)
                        .ok_or_else(|| invalid("free block offset overflow"))?,
                    size: block.size - consumed,
                };
            } else {
                chunk.free_blocks.remove(bi);
            }
            let end = block
                .offset
                .checked_add(bytes)
                .ok_or_else(|| invalid("chunk high-water overflow"))?;
            chunk.max_size = chunk.max_size.max(end);
            return Ok((ci, block.offset, consumed));
        }
        let capacity = self.max_size.max(size_aligned);
        let consumed = capacity.min(size_aligned);
        let remaining = capacity - consumed;
        let free_blocks = if remaining > 0 {
            vec![FreeBlock {
                offset: consumed,
                size: remaining,
            }]
        } else {
            Vec::new()
        };
        let ci = self.chunks.len();
        self.chunks.push(Chunk {
            free_blocks,
            max_size: bytes,
        });
        Ok((ci, 0, consumed))
    }

    fn free(&mut self, chunk_idx: usize, offset: u64, size: u64) -> Result<()> {
        if size == 0 {
            return Ok(());
        }
        let chunk = self
            .chunks
            .get_mut(chunk_idx)
            .ok_or_else(|| invalid("free block chunk index out of bounds"))?;
        let blocks = &mut chunk.free_blocks;
        let pos = blocks.partition_point(|block| block.offset < offset);
        let index = if pos > 0
            && blocks[pos - 1].offset.checked_add(blocks[pos - 1].size) == Some(offset)
        {
            blocks[pos - 1].size = blocks[pos - 1]
                .size
                .checked_add(size)
                .ok_or_else(|| invalid("free block size overflow"))?;
            pos - 1
        } else {
            blocks.insert(pos, FreeBlock { offset, size });
            pos
        };
        if index + 1 < blocks.len()
            && blocks[index].offset.checked_add(blocks[index].size)
                == Some(blocks[index + 1].offset)
        {
            blocks[index].size = blocks[index]
                .size
                .checked_add(blocks[index + 1].size)
                .ok_or_else(|| invalid("merged free block size overflow"))?;
            blocks.remove(index + 1);
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct BlockAllocation {
    buft: usize,
    chunk: usize,
    offset: u64,
    consumed: u64,
}

struct PlanState {
    dyn_allocs: Vec<DynAlloc>,
    allocs: Vec<Option<BlockAllocation>>,
    n_children: Vec<usize>,
    n_views: Vec<usize>,
}

impl PlanState {
    fn new(backends: &[BackendSpec], tensors: &[TensorDesc], graph: &Graph) -> Self {
        let dyn_allocs = backends.iter().copied().map(DynAlloc::new).collect();
        let mut n_children = vec![0_usize; tensors.len()];
        let mut n_views = vec![0_usize; tensors.len()];
        for &node in &graph.nodes {
            if let Some(root) = tensors[node].view_src {
                n_views[root] += 1;
            }
            for source in tensors[node].sources.into_iter().flatten() {
                n_children[source] += 1;
            }
        }
        Self {
            dyn_allocs,
            allocs: vec![None; tensors.len()],
            n_children,
            n_views,
        }
    }

    fn release_tensor(
        &mut self,
        id: usize,
        tensors: &[TensorDesc],
        ext_placements: &[Option<(usize, u64)>],
    ) -> Result<()> {
        let tensor = &tensors[id];
        if (tensor.flags & 2) != 0 || ext_placements[id].is_some() {
            return Ok(());
        }
        if let Some(root) = tensor.view_src {
            self.n_views[root] = self.n_views[root]
                .checked_sub(1)
                .ok_or_else(|| invalid("view reference underflow"))?;
            if self.n_children[root] == 0 && self.n_views[root] == 0 {
                return self.release_tensor(root, tensors, ext_placements);
            }
            return Ok(());
        }
        if let Some(allocation) = self.allocs[id] {
            self.dyn_allocs[allocation.buft].free(
                allocation.chunk,
                allocation.offset,
                allocation.consumed,
            )?;
        }
        Ok(())
    }

    fn try_inplace(
        &self,
        node: usize,
        buft: usize,
        tensors: &[TensorDesc],
        ext_placements: &[Option<(usize, u64)>],
    ) -> Option<(usize, BlockAllocation)> {
        let tensor = &tensors[node];
        if !tensor.op.can_inplace() {
            return None;
        }
        for parent in tensor.sources.into_iter().flatten() {
            let parent_desc = &tensors[parent];
            if ext_placements[parent].is_none()
                && parent_desc.view_src.is_none()
                && (parent_desc.flags & 2) == 0
                && self.n_children[parent] == 1
                && self.n_views[parent] == 0
                && parent_desc.bytes >= tensor.bytes
                && let Some(parent_alloc) = self.allocs[parent]
                && parent_alloc.buft == buft
            {
                return Some((parent, parent_alloc));
            }
        }
        None
    }

    fn alloc_or_reuse_node(
        &mut self,
        node: usize,
        buft: usize,
        tensors: &[TensorDesc],
        ext_placements: &[Option<(usize, u64)>],
    ) -> Result<Option<usize>> {
        if let Some((parent, allocation)) = self.try_inplace(node, buft, tensors, ext_placements) {
            self.allocs[node] = Some(allocation);
            return Ok(Some(parent));
        }
        let (chunk, offset, consumed) = self.dyn_allocs[buft].alloc(tensors[node].bytes)?;
        self.allocs[node] = Some(BlockAllocation {
            buft,
            chunk,
            offset,
            consumed,
        });
        Ok(None)
    }

    fn process_node(
        &mut self,
        node: usize,
        tensors: &[TensorDesc],
        bufts: &[usize],
        ext_placements: &[Option<(usize, u64)>],
    ) -> Result<()> {
        let tensor = &tensors[node];
        let reused_parent = if ext_placements[node].is_none() && tensor.view_src.is_none() {
            self.alloc_or_reuse_node(node, bufts[node], tensors, ext_placements)?
        } else {
            None
        };
        for parent in tensor.sources.into_iter().flatten() {
            self.n_children[parent] -= 1;
            if self.n_children[parent] == 0
                && self.n_views[parent] == 0
                && reused_parent != Some(parent)
            {
                self.release_tensor(parent, tensors, ext_placements)?;
            }
        }
        Ok(())
    }
}

struct GraphAllocator {
    backends: Vec<BackendSpec>,
    live_chunks: Vec<Vec<u64>>,
}

impl GraphAllocator {
    fn new(backends: Vec<BackendSpec>) -> Self {
        let count = backends.len();
        Self {
            backends,
            live_chunks: vec![Vec::new(); count],
        }
    }

    fn plan_graph(
        &mut self,
        tensors: &[TensorDesc],
        graph: &Graph,
        bufts: &[usize],
        ext_placements: &[Option<(usize, u64)>],
    ) -> Result<Vec<Option<BlockAllocation>>> {
        let mut state = PlanState::new(&self.backends, tensors, graph);
        for &leaf in &graph.leaves {
            if ext_placements[leaf].is_some() || tensors[leaf].view_src.is_some() {
                continue;
            }
            let buft = bufts[leaf];
            let (chunk, offset, consumed) = state.dyn_allocs[buft].alloc(tensors[leaf].bytes)?;
            state.allocs[leaf] = Some(BlockAllocation {
                buft,
                chunk,
                offset,
                consumed,
            });
        }
        for &node in &graph.nodes {
            state.process_node(node, tensors, bufts, ext_placements)?;
        }
        self.update_live_chunks(&state.dyn_allocs);
        Ok(state.allocs)
    }

    fn update_live_chunks(&mut self, dyn_allocs: &[DynAlloc]) {
        for (live, alloc) in self.live_chunks.iter_mut().zip(dyn_allocs) {
            let planned = alloc
                .chunks
                .iter()
                .map(|chunk| chunk.max_size)
                .collect::<Vec<_>>();
            let reuse = live.len() == planned.len()
                && live
                    .iter()
                    .zip(&planned)
                    .all(|(existing, needed)| existing >= needed);
            if !reuse {
                *live = planned;
            }
        }
    }
}

struct BufferTable {
    entries: Vec<Value>,
    ext_ids: Vec<usize>,
    chunk_ids: Vec<Vec<usize>>,
}

fn build_buffer_table(
    backends: &[BackendSpec],
    ext_buffers: &[(usize, u64)],
    live_chunks: &[Vec<u64>],
) -> BufferTable {
    let mut entries = Vec::new();
    let mut ext_ids = vec![0_usize; ext_buffers.len()];
    let mut chunk_ids = live_chunks
        .iter()
        .map(|chunks| vec![0_usize; chunks.len()])
        .collect::<Vec<_>>();
    for (buft, spec) in backends.iter().enumerate() {
        let maximum = spec.max_size.map_or(Value::Null, Value::from);
        for (ext_idx, &(ext_buft, size)) in ext_buffers.iter().enumerate() {
            if ext_buft != buft {
                continue;
            }
            let id = entries.len();
            ext_ids[ext_idx] = id;
            entries.push(json!({"Id": id, "Type": buft, "Size": size, "Maximum": maximum}));
        }
        for (chunk_idx, &size) in live_chunks[buft].iter().enumerate() {
            let id = entries.len();
            chunk_ids[buft][chunk_idx] = id;
            entries.push(json!({"Id": id, "Type": buft, "Size": size, "Maximum": maximum}));
        }
    }
    BufferTable {
        entries,
        ext_ids,
        chunk_ids,
    }
}

fn resolve_placement(
    index: usize,
    tensors: &[TensorDesc],
    visited: &[bool],
    ext_placements: &[Option<(usize, u64)>],
    allocs: &[Option<BlockAllocation>],
    table: &BufferTable,
) -> Result<Option<(usize, u64)>> {
    if let Some((ext_idx, offset)) = ext_placements[index] {
        return Ok(Some((table.ext_ids[ext_idx], offset)));
    }
    if !visited[index] {
        return Ok(None);
    }
    if let Some(root) = tensors[index].view_src {
        let Some((buf_id, root_offset)) =
            resolve_placement(root, tensors, visited, ext_placements, allocs, table)?
        else {
            return Ok(None);
        };
        let offset = root_offset
            .checked_add(tensors[index].view_offs)
            .ok_or_else(|| invalid("view placement offset overflow"))?;
        return Ok(Some((buf_id, offset)));
    }
    Ok(allocs[index].map(|alloc| (table.chunk_ids[alloc.buft][alloc.chunk], alloc.offset)))
}

fn tensor_snapshot(index: usize, tensor: &TensorDesc, placement: Option<(usize, u64)>) -> Value {
    let buffer = placement.map(|(buf_id, _)| buf_id);
    let offset = placement.map(|(_, offset)| offset);
    let sources = tensor
        .sources
        .iter()
        .map(|slot| slot.map_or(Value::Null, Value::from))
        .collect::<Vec<_>>();
    json!({
        "Index": index,
        "Context": tensor.context,
        "Buffer": buffer,
        "DataPresent": placement.is_some(),
        "Offset": offset,
        "Bytes": tensor.bytes,
        "ViewSource": tensor.view_src,
        "Sources": sources,
        "Op": tensor.op.code(),
        "Flags": tensor.flags,
        "CanInplace": tensor.op.can_inplace(),
        "Output": (tensor.flags & 2) != 0
    })
}

fn execute_operation(operation: &ValidatedOperation) -> Result<Value> {
    let mut allocator = GraphAllocator::new(operation.backends.clone());
    let mut observations = Vec::with_capacity(operation.steps.len());
    for step in &operation.steps {
        let (ext_buffers, ext_placements) = allocate_external(step, &operation.backends)?;
        let graph = build_graph(&step.tensors, step.output)?;
        let default_bufts = vec![0_usize; step.tensors.len()];
        let bufts = if step.reserve {
            &step.buffer_types
        } else {
            &default_bufts
        };
        let allocs = allocator.plan_graph(&step.tensors, &graph, bufts, &ext_placements)?;
        let table = build_buffer_table(&operation.backends, &ext_buffers, &allocator.live_chunks);
        let mut tensor_values = Vec::with_capacity(step.tensors.len());
        for (index, tensor) in step.tensors.iter().enumerate() {
            let placement = resolve_placement(
                index,
                &step.tensors,
                &graph.visited,
                &ext_placements,
                &allocs,
                &table,
            )?;
            tensor_values.push(tensor_snapshot(index, tensor, placement));
        }
        observations.push(json!({
            "Allocated": true,
            "Reserved": step.reserve.then_some(true),
            "Tensors": tensor_values,
            "Buffers": table.entries,
            "GraphNodes": graph.nodes,
            "GraphLeaves": graph.leaves
        }));
    }
    Ok(json!({"Steps": observations}))
}

/// Report whether `value` contains only `allocation-graph` operations.
pub(super) fn handles(value: &Value) -> bool {
    value["Operations"].as_array().is_some_and(|operations| {
        !operations.is_empty()
            && operations
                .iter()
                .all(|operation| operation["Kind"] == "allocation-graph")
    })
}

/// Validate and evaluate a batch of independent `allocation-graph` requests.
///
/// # Errors
///
/// Returns an error when `value` fails schema, shape, alignment, or context
/// allocation validation.
pub(super) fn execute(value: &Value) -> Result<Value> {
    let request: Request = serde_json::from_value(value.clone())?;
    require(
        request.schema_version == 1 && (1..=MAX_OPERATIONS).contains(&request.operations.len()),
        "independent allocation requires schema 1 and 1 through 128 operations",
    )?;
    let validated = request
        .operations
        .iter()
        .map(validate_operation)
        .collect::<Result<Vec<_>>>()?;
    let results = validated
        .iter()
        .map(execute_operation)
        .collect::<Result<Vec<_>>>()?;
    Ok(json!({"SchemaVersion": 1, "Results": results}))
}

/// Disclose the independent Rust graph-allocator provider identity.
pub(super) fn identity() -> Value {
    json!({
        "Kind": "independent-rust-graph-allocator",
        "SupportedSourceIds": [ID],
        "UpstreamCommit": super::UPSTREAM,
        "AssertionSource": {
            "Path": ID,
            "Sha256": "89de1c9549888a37b388474aca89cad3052c1751fbdace09c81e79d5844633db"
        },
        "Architecture": "liveness-aware-chunked-free-block-graph-allocator-v1",
        "OriginalCases": 13,
        "OriginalGraphAllocations": 14,
        "NativeReferenceReplay": false,
        "ProductionRuntimeCalled": false,
        "CompleteSourceGate": false,
        "NativeRetirementQualified": false
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn envelope(operations: Value) -> Value {
        let mut object = serde_json::Map::new();
        object.insert("SchemaVersion".into(), json!(1));
        object.insert("Operations".into(), operations);
        Value::Object(object)
    }

    fn single_op(backends: Value, steps: Value) -> Value {
        let mut operation = serde_json::Map::new();
        operation.insert("Kind".into(), Value::String("allocation-graph".into()));
        operation.insert("Backends".into(), backends);
        operation.insert("Steps".into(), steps);
        envelope(Value::Array(vec![Value::Object(operation)]))
    }

    fn step(
        nodes: Value,
        output: usize,
        buffer_types: Vec<usize>,
        external: Value,
        reserve: bool,
    ) -> Value {
        let mut object = serde_json::Map::new();
        object.insert("Nodes".into(), nodes);
        object.insert("Output".into(), json!(output));
        object.insert(
            "BufferTypes".into(),
            Value::Array(buffer_types.into_iter().map(Value::from).collect()),
        );
        object.insert("ExternalAllocations".into(), external);
        object.insert("Reserve".into(), Value::Bool(reserve));
        Value::Object(object)
    }

    fn backend_total(observation: &Value, backend: usize) -> Result<u64> {
        let buffers = observation["Buffers"]
            .as_array()
            .ok_or_else(|| invalid("missing test Buffers array"))?;
        let mut total = 0_u64;
        for buffer in buffers {
            if buffer["Type"] == backend {
                let size = buffer["Size"]
                    .as_u64()
                    .ok_or_else(|| invalid("missing test buffer Size"))?;
                total = total
                    .checked_add(size)
                    .ok_or_else(|| invalid("test buffer sum overflow"))?;
            }
        }
        Ok(total)
    }

    fn read_span(tensor: &Value) -> Result<(usize, u64, u64)> {
        let buf = usize::try_from(
            tensor["Buffer"]
                .as_u64()
                .ok_or_else(|| invalid("buffer missing"))?,
        )?;
        let off = tensor["Offset"]
            .as_u64()
            .ok_or_else(|| invalid("offset missing"))?;
        let bytes = tensor["Bytes"]
            .as_u64()
            .ok_or_else(|| invalid("bytes missing"))?;
        Ok((buf, off, bytes))
    }

    fn check_invariants(observation: &Value, alignments: &[u64]) -> Result<()> {
        let buffers = observation["Buffers"]
            .as_array()
            .ok_or_else(|| invalid("missing Buffers"))?;
        let tensors = observation["Tensors"]
            .as_array()
            .ok_or_else(|| invalid("missing Tensors"))?;
        let nodes = observation["GraphNodes"]
            .as_array()
            .ok_or_else(|| invalid("missing GraphNodes"))?;
        let leaves = observation["GraphLeaves"]
            .as_array()
            .ok_or_else(|| invalid("missing GraphLeaves"))?;
        for tensor in tensors {
            if tensor["DataPresent"] == json!(true) {
                let (buf, off, bytes) = read_span(tensor)?;
                let size = buffers[buf]["Size"]
                    .as_u64()
                    .ok_or_else(|| invalid("buffer size missing"))?;
                assert!(off + bytes <= size);
                let buft = usize::try_from(
                    buffers[buf]["Type"]
                        .as_u64()
                        .ok_or_else(|| invalid("buffer type missing"))?,
                )?;
                if tensor["ViewSource"].is_null() {
                    assert!(off.is_multiple_of(alignments[buft]));
                }
                if tensor["Context"] == 0
                    && let Some(max) = buffers[buf]["Maximum"].as_u64()
                    && bytes <= max
                {
                    assert!(off + bytes <= max);
                }
            }
        }
        check_unique_present(tensors, nodes)?;
        check_unique_present(tensors, leaves)?;
        check_leaves_disjoint(tensors, leaves)?;
        Ok(())
    }

    fn check_unique_present(tensors: &[Value], indices: &[Value]) -> Result<()> {
        for (i, item) in indices.iter().enumerate() {
            let id = usize::try_from(item.as_u64().ok_or_else(|| invalid("index missing"))?)?;
            assert_eq!(tensors[id]["DataPresent"], json!(true));
            for prev_val in &indices[..i] {
                let prev =
                    usize::try_from(prev_val.as_u64().ok_or_else(|| invalid("prev index"))?)?;
                assert_ne!(id, prev);
            }
        }
        Ok(())
    }

    fn check_leaves_disjoint(tensors: &[Value], leaves: &[Value]) -> Result<()> {
        let mut ranges = Vec::new();
        for item in leaves {
            let id = usize::try_from(item.as_u64().ok_or_else(|| invalid("leaf index"))?)?;
            if tensors[id]["Context"] == 0 && tensors[id]["ViewSource"].is_null() {
                ranges.push(read_span(&tensors[id])?);
            }
        }
        for (i, &(buf_a, off_a, bytes_a)) in ranges.iter().enumerate() {
            for &(buf_b, off_b, bytes_b) in &ranges[..i] {
                if buf_a == buf_b {
                    assert!(off_a + bytes_a <= off_b || off_b + bytes_b <= off_a);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn independent_allocation_handles_and_identity_contract() {
        assert!(handles(&single_op(
            json!([{"MaxBufferSize": null, "Alignment": 8}]),
            json!([])
        )));
        assert!(!handles(&json!({})));
        assert!(!handles(&envelope(json!([]))));
        assert!(!handles(&envelope(json!([{"Kind": "other"}]))));
        assert!(!handles(&envelope(json!([
            {"Kind": "allocation-graph"},
            {"Kind": "other"}
        ]))));
        let id = identity();
        assert_eq!(id["SupportedSourceIds"], json!([ID]));
        assert_eq!(id["UpstreamCommit"], super::super::UPSTREAM);
        assert_eq!(
            id["AssertionSource"]["Sha256"],
            "89de1c9549888a37b388474aca89cad3052c1751fbdace09c81e79d5844633db"
        );
        for field in [
            "NativeReferenceReplay",
            "ProductionRuntimeCalled",
            "CompleteSourceGate",
            "NativeRetirementQualified",
        ] {
            assert_eq!(id[field], json!(false));
        }
    }

    #[test]
    fn independent_allocation_handles_max_sizes_chunks_and_view_inplace() -> Result<()> {
        let too_large = execute(&single_op(
            json!([{"MaxBufferSize": 16, "Alignment": 8}]),
            json!([step(
                json!([
                    {"Op": "input", "Context": 0, "Elements": 6},
                    {"Op": "scale", "Context": 0, "Inputs": [0], "Factor": 2.0}
                ]),
                1,
                vec![0, 0],
                json!([]),
                false
            )]),
        ))?;
        let obs = &too_large["Results"][0]["Steps"][0];
        assert_eq!(obs["Allocated"], json!(true));
        assert_eq!(obs["Reserved"], Value::Null);
        assert_eq!(backend_total(obs, 0)?, 24);
        assert_eq!(obs["Tensors"][1]["CanInplace"], json!(true));
        assert_eq!(obs["Tensors"][1]["Output"], json!(true));
        assert_eq!(obs["Tensors"][1]["Offset"], 0);

        let mut chunk_nodes = vec![json!({"Op": "input", "Context": 0, "Elements": 2}); 17];
        for i in 1..17 {
            let left = if i == 1 { 0 } else { 15 + i };
            chunk_nodes.push(json!({"Op": "add", "Context": 0, "Inputs": [left, i]}));
        }
        let chunk_types = vec![0; chunk_nodes.len()];
        let chunks_obs = &execute(&single_op(
            json!([{"MaxBufferSize": 8, "Alignment": 8}]),
            json!([step(json!(chunk_nodes), 32, chunk_types, json!([]), false)]),
        ))?["Results"][0]["Steps"][0];
        assert_eq!(chunks_obs["Allocated"], json!(true));
        assert!(backend_total(chunks_obs, 0)? > 128);

        let view_obs = &execute(&single_op(
            json!([{"MaxBufferSize": 32, "Alignment": 8}]),
            json!([step(
                json!([
                    {"Op": "input", "Context": 0, "Elements": 4},
                    {"Op": "reshape", "Context": 0, "Inputs": [0], "Dimensions": [2, 2]},
                    {"Op": "permute", "Context": 0, "Inputs": [1], "Axes": [1, 0, 2, 3]},
                    {"Op": "view", "Context": 0, "Inputs": [2], "Elements": 2, "OffsetBytes": 4},
                    {"Op": "input", "Context": 0, "Elements": 2},
                    {"Op": "add", "Context": 0, "Inputs": [3, 4]}
                ]),
                5,
                vec![0; 6],
                json!([]),
                false
            )]),
        ))?["Results"][0]["Steps"][0];
        check_invariants(view_obs, &[8])?;
        assert!(backend_total(view_obs, 0)? <= 24);
        assert_eq!(view_obs["Tensors"][3]["ViewSource"], 0);
        assert_eq!(view_obs["Tensors"][3]["Offset"], 4);
        assert_eq!(
            view_obs["Tensors"][5]["Buffer"],
            view_obs["Tensors"][4]["Buffer"]
        );
        assert_eq!(
            view_obs["Tensors"][5]["Offset"],
            view_obs["Tensors"][4]["Offset"]
        );
        Ok(())
    }

    #[test]
    fn independent_allocation_merges_free_blocks_and_prefers_allocated_memory() -> Result<()> {
        for max_buf in [json!(32), Value::Null] {
            let obs = &execute(&single_op(
                json!([{"MaxBufferSize": max_buf, "Alignment": 8}]),
                json!([step(
                    json!([
                        {"Op": "input", "Context": 0, "Elements": 4},
                        {"Op": "input", "Context": 0, "Elements": 4},
                        {"Op": "input", "Context": 0, "Elements": 4},
                        {"Op": "mean", "Context": 0, "Inputs": [0]},
                        {"Op": "mean", "Context": 0, "Inputs": [1]},
                        {"Op": "pad", "Context": 0, "Inputs": [2], "Amount": 2},
                        {"Op": "add", "Context": 0, "Inputs": [3, 4]},
                        {"Op": "pad", "Context": 0, "Inputs": [6], "Amount": 5},
                        {"Op": "add", "Context": 0, "Inputs": [5, 7]}
                    ]),
                    8,
                    vec![0; 9],
                    json!([]),
                    false
                )]),
            ))?["Results"][0]["Steps"][0];
            check_invariants(obs, &[8])?;
            assert!(backend_total(obs, 0)? <= 88);
        }

        let prefer_obs = &execute(&single_op(
            json!([{"MaxBufferSize": 32, "Alignment": 4}]),
            json!([step(
                json!([
                    {"Op": "input", "Context": 0, "Elements": 6},
                    {"Op": "mean", "Context": 0, "Inputs": [0]},
                    {"Op": "mean", "Context": 0, "Inputs": [1]}
                ]),
                2,
                vec![0; 3],
                json!([]),
                false
            )]),
        ))?["Results"][0]["Steps"][0];
        assert_eq!(backend_total(prefer_obs, 0)?, 28);
        assert_eq!(prefer_obs["Tensors"][2]["Offset"], 0);

        let leftover_obs = &execute(&single_op(
            json!([{"MaxBufferSize": 16, "Alignment": 8}]),
            json!([step(
                json!([
                    {"Op": "input", "Context": 0, "Elements": 2},
                    {"Op": "pad", "Context": 0, "Inputs": [0], "Amount": 2},
                    {"Op": "mean", "Context": 0, "Inputs": [1]}
                ]),
                2,
                vec![0; 3],
                json!([]),
                false
            )]),
        ))?["Results"][0]["Steps"][0];
        check_invariants(leftover_obs, &[8])?;
        assert!(backend_total(leftover_obs, 0)? <= 28);
        Ok(())
    }

    #[test]
    fn independent_allocation_supports_multi_backend_and_zero_byte_backends() -> Result<()> {
        let multi_obs = &execute(&single_op(
            json!([
                {"MaxBufferSize": 32, "Alignment": 8},
                {"MaxBufferSize": null, "Alignment": 8}
            ]),
            json!([step(
                json!([
                    {"Op": "input", "Context": 1, "Elements": 4},
                    {"Op": "input", "Context": 1, "Elements": 4},
                    {"Op": "input", "Context": 2, "Elements": 6},
                    {"Op": "input", "Context": 2, "Elements": 1},
                    {"Op": "input", "Context": 0, "Elements": 4},
                    {"Op": "mul", "Context": 0, "Inputs": [4, 0]},
                    {"Op": "pad", "Context": 0, "Inputs": [5], "Amount": 2},
                    {"Op": "mul", "Context": 0, "Inputs": [6, 2]},
                    {"Op": "mean", "Context": 0, "Inputs": [7]},
                    {"Op": "add", "Context": 0, "Inputs": [8, 3]},
                    {"Op": "pad", "Context": 0, "Inputs": [9], "Amount": 3},
                    {"Op": "add", "Context": 0, "Inputs": [10, 1]},
                    {"Op": "scale", "Context": 0, "Inputs": [11], "Factor": 2.0}
                ]),
                12,
                vec![0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                json!([{"Context": 1, "BufferType": 0}, {"Context": 2, "BufferType": 1}]),
                true
            )]),
        ))?["Results"][0]["Steps"][0];
        assert_eq!(multi_obs["Reserved"], json!(true));
        assert_eq!(multi_obs["GraphLeaves"].as_array().map(Vec::len), Some(5));
        assert_eq!(multi_obs["GraphNodes"].as_array().map(Vec::len), Some(8));
        check_invariants(multi_obs, &[8, 8])?;
        assert!(backend_total(multi_obs, 0)? <= 88);
        assert!(backend_total(multi_obs, 1)? <= 56);

        let zero_obs = &execute(&single_op(
            json!([
                {"MaxBufferSize": null, "Alignment": 8},
                {"MaxBufferSize": null, "Alignment": 8}
            ]),
            json!([step(
                json!([
                    {"Op": "input", "Context": 0, "Elements": 4},
                    {"Op": "scale", "Context": 0, "Inputs": [0], "Factor": 2.0}
                ]),
                1,
                vec![0, 0],
                json!([]),
                true
            )]),
        ))?["Results"][0]["Steps"][0];
        assert_eq!(zero_obs["Reserved"], json!(true));
        assert_eq!(backend_total(zero_obs, 0)?, 16);
        assert_eq!(backend_total(zero_obs, 1)?, 0);
        Ok(())
    }

    #[test]
    fn independent_allocation_reuses_and_reallocates_chunks_across_steps() -> Result<()> {
        let realloc = execute(&single_op(
            json!([{"MaxBufferSize": 32, "Alignment": 4}]),
            json!([
                step(
                    json!([
                        {"Op": "input", "Context": 0, "Elements": 6},
                        {"Op": "input", "Context": 0, "Elements": 4},
                        {"Op": "view", "Context": 0, "Inputs": [0], "Elements": 4, "OffsetBytes": 0},
                        {"Op": "add", "Context": 0, "Inputs": [2, 1]}
                    ]),
                    3,
                    vec![0; 4],
                    json!([]),
                    false
                ),
                step(
                    json!([
                        {"Op": "input", "Context": 0, "Elements": 5},
                        {"Op": "input", "Context": 0, "Elements": 5},
                        {"Op": "add", "Context": 0, "Inputs": [0, 1]}
                    ]),
                    2,
                    vec![0; 3],
                    json!([]),
                    false
                ),
                step(
                    json!([
                        {"Op": "input", "Context": 0, "Elements": 2},
                        {"Op": "scale", "Context": 0, "Inputs": [0], "Factor": 1.0}
                    ]),
                    1,
                    vec![0; 2],
                    json!([]),
                    false
                )
            ]),
        ))?;
        let steps = &realloc["Results"][0]["Steps"];
        check_invariants(&steps[0], &[4])?;
        check_invariants(&steps[1], &[4])?;
        assert_eq!(backend_total(&steps[0], 0)?, 40);
        assert_eq!(backend_total(&steps[1], 0)?, 40);
        assert_eq!(backend_total(&steps[2], 0)?, 8);
        Ok(())
    }

    #[test]
    fn independent_allocation_handles_unreachable_nodes_alignment_and_chains() -> Result<()> {
        let obs = &execute(&single_op(
            json!([{"MaxBufferSize": null, "Alignment": 64}]),
            json!([step(
                json!([
                    {"Op": "input", "Context": 0, "Elements": 4},
                    {"Op": "input", "Context": 1, "Elements": 2},
                    {"Op": "view", "Context": 0, "Inputs": [0], "Elements": 2, "OffsetBytes": 4},
                    {"Op": "view", "Context": 0, "Inputs": [2], "Elements": 1, "OffsetBytes": 4},
                    {"Op": "concat", "Context": 0, "Inputs": [0, 1]}
                ]),
                4,
                vec![0; 5],
                json!([{"Context": 1, "BufferType": 0}]),
                true
            )]),
        ))?["Results"][0]["Steps"][0];
        check_invariants(obs, &[64])?;
        assert_eq!(obs["Tensors"][2]["DataPresent"], json!(false));
        assert_eq!(obs["Tensors"][2]["Buffer"], Value::Null);
        assert_eq!(obs["Tensors"][3]["ViewSource"], 0);
        assert_eq!(obs["Tensors"][1]["DataPresent"], json!(true));
        assert_eq!(obs["Tensors"][4]["Bytes"], 24);
        assert_eq!(
            obs["Tensors"][4]["Sources"].as_array().map(Vec::len),
            Some(10)
        );
        assert_eq!(obs["GraphLeaves"], json!([0, 1]));
        assert_eq!(obs["GraphNodes"], json!([4]));

        let leaf_out = &execute(&single_op(
            json!([{"MaxBufferSize": null, "Alignment": 1}]),
            json!([step(
                json!([{"Op": "input", "Context": 0, "Elements": 3}]),
                0,
                vec![0],
                json!([]),
                false
            )]),
        ))?["Results"][0]["Steps"][0];
        check_invariants(leaf_out, &[1])?;
        assert_eq!(leaf_out["Tensors"][0]["Flags"], 3);
        assert_eq!(leaf_out["Tensors"][0]["Output"], json!(true));
        assert_eq!(leaf_out["GraphLeaves"], json!([0]));
        assert_eq!(leaf_out["GraphNodes"], json!([]));
        Ok(())
    }

    #[test]
    fn independent_allocation_rejects_invalid_envelopes_and_backends() {
        let valid_step = step(
            json!([
                {"Op": "input", "Context": 0, "Elements": 4},
                {"Op": "scale", "Context": 0, "Inputs": [0], "Factor": 1.0}
            ]),
            1,
            vec![0, 0],
            json!([]),
            false,
        );
        let valid = single_op(
            json!([{"MaxBufferSize": null, "Alignment": 8}]),
            json!([&valid_step]),
        );
        assert!(execute(&valid).is_ok());

        for bad in [
            json!({"SchemaVersion": 2, "Operations": &valid["Operations"]}),
            envelope(json!([])),
            envelope(json!(vec![&valid["Operations"][0]; 129])),
            envelope(json!([{
                "Kind": "other",
                "Backends": [{"MaxBufferSize": null, "Alignment": 8}],
                "Steps": [&valid_step]
            }])),
            single_op(json!([]), json!([&valid_step])),
            single_op(
                json!(vec![json!({"MaxBufferSize": null, "Alignment": 8}); 4]),
                json!([&valid_step]),
            ),
            single_op(json!([{"Alignment": 8}]), json!([&valid_step])),
            single_op(
                json!([{"MaxBufferSize": 0, "Alignment": 8}]),
                json!([&valid_step]),
            ),
            single_op(
                json!([{"MaxBufferSize": 1_048_577, "Alignment": 8}]),
                json!([&valid_step]),
            ),
            single_op(
                json!([{"MaxBufferSize": null, "Alignment": 0}]),
                json!([&valid_step]),
            ),
            single_op(
                json!([{"MaxBufferSize": null, "Alignment": 6}]),
                json!([&valid_step]),
            ),
            single_op(
                json!([{"MaxBufferSize": null, "Alignment": 128}]),
                json!([&valid_step]),
            ),
            single_op(json!([{"MaxBufferSize": null, "Alignment": 8}]), json!([])),
            single_op(
                json!([{"MaxBufferSize": null, "Alignment": 8}]),
                json!(vec![&valid_step; 5]),
            ),
        ] {
            assert!(execute(&bad).is_err());
        }
    }

    #[test]
    fn independent_allocation_rejects_invalid_steps() {
        for bad_step in [
            step(json!([]), 0, vec![], json!([]), false),
            step(
                json!(vec![
                    json!({"Op": "input", "Context": 0, "Elements": 1});
                    41
                ]),
                0,
                vec![0; 41],
                json!([]),
                false,
            ),
            step(
                json!([{"Op": "input", "Context": 0, "Elements": 0}]),
                0,
                vec![0],
                json!([]),
                false,
            ),
            step(
                json!([{"Op": "input", "Context": 0, "Elements": 1025}]),
                0,
                vec![0],
                json!([]),
                false,
            ),
            step(
                json!([{"Op": "input", "Context": 1, "Elements": 4}]),
                0,
                vec![0],
                json!([]),
                false,
            ),
            step(
                json!([{"Op": "input", "Context": 0, "Elements": 4}]),
                1,
                vec![0],
                json!([]),
                false,
            ),
            step(
                json!([{"Op": "input", "Context": 0, "Elements": 4}]),
                0,
                vec![0, 0],
                json!([]),
                false,
            ),
            step(
                json!([{"Op": "input", "Context": 0, "Elements": 4}]),
                0,
                vec![1],
                json!([]),
                false,
            ),
            step(
                json!([{"Op": "input", "Context": 0, "Elements": 4}]),
                0,
                vec![0],
                json!([{"Context": 0, "BufferType": 0}]),
                false,
            ),
            step(
                json!([{"Op": "input", "Context": 1, "Elements": 4}]),
                0,
                vec![0],
                json!([{"Context": 1, "BufferType": 0}, {"Context": 1, "BufferType": 0}]),
                false,
            ),
            step(
                json!([{"Op": "input", "Context": 0, "Elements": 4}]),
                0,
                vec![0],
                json!([{"Context": 1, "BufferType": 0}]),
                false,
            ),
        ] {
            assert!(
                execute(&single_op(
                    json!([{"MaxBufferSize": null, "Alignment": 8}]),
                    json!([bad_step]),
                ))
                .is_err()
            );
        }
    }

    #[test]
    fn independent_allocation_rejects_invalid_shapes() {
        for bad_step in [
            step(
                json!([{"Op": "input", "Context": 0, "Elements": 4}, {"Op": "add", "Context": 0, "Inputs": [0, 1]}]),
                1,
                vec![0, 0],
                json!([]),
                false,
            ),
            step(
                json!([{"Op": "input", "Context": 0, "Elements": 4}, {"Op": "input", "Context": 0, "Elements": 2}, {"Op": "add", "Context": 0, "Inputs": [0, 1]}]),
                2,
                vec![0, 0, 0],
                json!([]),
                false,
            ),
            step(
                json!([{"Op": "input", "Context": 0, "Elements": 4}, {"Op": "reshape", "Context": 0, "Inputs": [0], "Dimensions": [2, 2]}, {"Op": "concat", "Context": 0, "Inputs": [0, 1]}]),
                2,
                vec![0, 0, 0],
                json!([]),
                false,
            ),
            step(
                json!([{"Op": "input", "Context": 0, "Elements": 4}, {"Op": "scale", "Context": 0, "Inputs": [0], "Factor": 1e40_f64}]),
                1,
                vec![0, 0],
                json!([]),
                false,
            ),
            step(
                json!([{"Op": "input", "Context": 0, "Elements": 4}, {"Op": "reshape", "Context": 0, "Inputs": [0], "Dimensions": [2, 3]}]),
                1,
                vec![0, 0],
                json!([]),
                false,
            ),
            step(
                json!([{"Op": "input", "Context": 0, "Elements": 4}, {"Op": "permute", "Context": 0, "Inputs": [0], "Axes": [0, 0, 2, 3]}]),
                1,
                vec![0, 0],
                json!([]),
                false,
            ),
            step(
                json!([{"Op": "input", "Context": 0, "Elements": 4}, {"Op": "view", "Context": 0, "Inputs": [0], "Elements": 2, "OffsetBytes": 2}]),
                1,
                vec![0, 0],
                json!([]),
                false,
            ),
            step(
                json!([{"Op": "input", "Context": 0, "Elements": 4}, {"Op": "view", "Context": 0, "Inputs": [0], "Elements": 4, "OffsetBytes": 4}]),
                1,
                vec![0, 0],
                json!([]),
                false,
            ),
            step(
                json!([{"Op": "input", "Context": 0, "Elements": 1000}, {"Op": "pad", "Context": 0, "Inputs": [0], "Amount": 25}]),
                1,
                vec![0, 0],
                json!([]),
                false,
            ),
            step(
                json!([{"Op": "input", "Context": 0, "Elements": 4, "Extra": 1}]),
                0,
                vec![0],
                json!([]),
                false,
            ),
        ] {
            assert!(
                execute(&single_op(
                    json!([{"MaxBufferSize": null, "Alignment": 8}]),
                    json!([bad_step]),
                ))
                .is_err()
            );
        }
    }
}
