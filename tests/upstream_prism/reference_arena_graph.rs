//! Independent safe Rust arena-placement and graph reference for `MoE` weighted reduction.
//!
//! No production `bonsai_prism_runtime` code, native `CPU` backend or expected-answer
//! fixture is called here. Immutable upstream MIT source identity is disclosed by
//! `identity`.

use serde::{Deserialize, Deserializer};
use serde_json::{Value, json};

use super::{Result, invalid};

pub(super) const ID: &str = "tests/test-moe-weighted-reduce.cpp";

const ALIGNMENT: usize = 64;
const MAX_ELEMENTS: usize = 2 * 1024 * 1024;
const MAX_STORAGE: usize = 96 * 1024 * 1024;
const MAX_HEX_CHARS: usize = 16 * 1024 * 1024;
const MAX_OUTPUT_HEX_CHARS: usize = 15 * 1024 * 1024;

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
struct Request {
    schema_version: u32,
    operations: Vec<Operation>,
}

#[derive(Deserialize)]
#[serde(tag = "Kind", rename_all_fields = "PascalCase", deny_unknown_fields)]
enum Operation {
    #[serde(rename = "cpu-buffer-info")]
    BufferInfo,
    #[serde(rename = "cpu-arena-graph")]
    ArenaGraph {
        graph: GraphSpec,
        forward: Vec<u64>,
        #[serde(deserialize_with = "nullable")]
        allocation: Option<AllocationSpec>,
    },
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
struct GraphSpec {
    kind: String,
    threads: u64,
    nodes: Vec<ArenaNode>,
    outputs: Vec<u64>,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
struct AllocationSpec {
    buffer_bytes: u64,
    offsets: Vec<Option<u64>>,
}

#[derive(Deserialize)]
#[serde(tag = "Op", rename_all_fields = "PascalCase", deny_unknown_fields)]
enum ArenaNode {
    #[serde(rename = "input")]
    Input {
        #[serde(rename = "Type")]
        kind: String,
        dimensions: Vec<u64>,
        data_hex: String,
    },
    #[serde(rename = "mul")]
    Mul { inputs: Vec<u64> },
    #[serde(rename = "add")]
    Add { inputs: Vec<u64> },
    #[serde(rename = "view_2d")]
    View2d {
        inputs: Vec<u64>,
        dimensions: Vec<u64>,
        row_stride_bytes: u64,
        offset_bytes: u64,
    },
}

fn nullable<'de, T: Deserialize<'de>, D: Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Option<T>, D::Error> {
    Option::deserialize(deserializer)
}

#[derive(Clone, Copy)]
struct Descriptor {
    ne: [usize; 4],
    strides: [usize; 4],
    contiguous: bool,
}

impl Descriptor {
    const fn contiguous(ne: [usize; 4]) -> Self {
        let s1 = 4 * ne[0];
        let s2 = s1 * ne[1];
        let s3 = s2 * ne[2];
        Self {
            ne,
            strides: [4, s1, s2, s3],
            contiguous: true,
        }
    }

    const fn count(&self) -> usize {
        self.ne[0] * self.ne[1] * self.ne[2] * self.ne[3]
    }

    const fn bytes(&self) -> usize {
        self.count() * 4
    }
}

enum PlannedOp {
    Input(Vec<u8>),
    Mul { a: usize, b: usize },
    Add { a: usize, b: usize },
    View2d { src: usize, offset_bytes: usize },
}

struct PlannedNode {
    shape: Descriptor,
    op: PlannedOp,
}

struct PlannedAllocation {
    buffer_bytes: usize,
    offsets: Vec<Option<usize>>,
}

struct ArenaPlan {
    nodes: Vec<PlannedNode>,
    outputs: Vec<usize>,
    forward: Vec<usize>,
    allocation: Option<PlannedAllocation>,
}

#[derive(Clone, Copy)]
struct Operand<'a> {
    desc: &'a Descriptor,
    offset: usize,
}

fn require(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(invalid(message))
    }
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(char::from(DIGITS[usize::from(byte >> 4)]));
        output.push(char::from(DIGITS[usize::from(byte & 15)]));
    }
    output
}

const fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn unhex(text: &str) -> Result<Vec<u8>> {
    require(
        text.len() <= MAX_HEX_CHARS && text.len().is_multiple_of(2),
        "arena hex extent outside bound",
    )?;
    text.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&[hi, lo]| {
            let high = hex_nibble(hi).ok_or_else(|| invalid("invalid arena hex digit"))?;
            let low = hex_nibble(lo).ok_or_else(|| invalid("invalid arena hex digit"))?;
            Ok((high << 4) | low)
        })
        .collect()
}

fn parse_dimensions(raw: &[u64], expected_rank: Option<usize>) -> Result<Descriptor> {
    require(
        !raw.is_empty() && raw.len() <= 4,
        "arena tensor requires 1..4 dimensions",
    )?;
    if let Some(rank) = expected_rank {
        require(raw.len() == rank, "arena tensor dimension rank mismatch")?;
    }
    let mut ne = [1_usize; 4];
    let mut count = 1_usize;
    for (target, &dim) in ne.iter_mut().zip(raw) {
        let extent = usize::try_from(dim)?;
        require((1..=8192).contains(&extent), "arena tensor dimension outside bound")?;
        *target = extent;
        count = count
            .checked_mul(extent)
            .ok_or_else(|| invalid("arena tensor element count overflow"))?;
        require(count <= MAX_ELEMENTS, "arena tensor exceeds element bound")?;
    }
    Ok(Descriptor::contiguous(ne))
}

fn unary_input(inputs: &[u64], available: usize) -> Result<usize> {
    require(inputs.len() == 1 && available > 0, "arena unary input count mismatch")?;
    let idx = usize::try_from(inputs[0])?;
    require(idx < available, "arena input reference out of range")?;
    Ok(idx)
}

fn binary_inputs(inputs: &[u64], available: usize) -> Result<(usize, usize)> {
    require(inputs.len() == 2 && available > 0, "arena binary input count mismatch")?;
    let a = usize::try_from(inputs[0])?;
    let b = usize::try_from(inputs[1])?;
    require(a < available && b < available, "arena input reference out of range")?;
    Ok((a, b))
}

fn plan_node(node: &ArenaNode, shapes: &[Descriptor]) -> Result<PlannedNode> {
    match node {
        ArenaNode::Input {
            kind,
            dimensions,
            data_hex,
        } => {
            require(kind == "F32", "arena inputs must be F32")?;
            let shape = parse_dimensions(dimensions, None)?;
            let bytes = unhex(data_hex)?;
            require(bytes.len() == shape.bytes(), "arena input bytes mismatch")?;
            for chunk in bytes.as_chunks::<4>().0 {
                require(
                    f32::from_le_bytes(*chunk).is_finite(),
                    "nonfinite arena graph input",
                )?;
            }
            Ok(PlannedNode {
                shape,
                op: PlannedOp::Input(bytes),
            })
        }
        ArenaNode::Mul { inputs } | ArenaNode::Add { inputs } => {
            let (a_idx, b_idx) = binary_inputs(inputs, shapes.len())?;
            let (a, b) = (&shapes[a_idx], &shapes[b_idx]);
            for (da, db) in a.ne.into_iter().zip(b.ne) {
                require(da.is_multiple_of(db), "incompatible arena broadcast shapes")?;
            }
            let op = if matches!(node, ArenaNode::Mul { .. }) {
                PlannedOp::Mul { a: a_idx, b: b_idx }
            } else {
                PlannedOp::Add { a: a_idx, b: b_idx }
            };
            Ok(PlannedNode {
                shape: Descriptor::contiguous(a.ne),
                op,
            })
        }
        ArenaNode::View2d {
            inputs,
            dimensions,
            row_stride_bytes,
            offset_bytes,
        } => plan_view_2d(inputs, dimensions, *row_stride_bytes, *offset_bytes, shapes),
    }
}

fn plan_view_2d(
    inputs: &[u64],
    dimensions: &[u64],
    row_stride_bytes: u64,
    offset_bytes: u64,
    shapes: &[Descriptor],
) -> Result<PlannedNode> {
    let src_idx = unary_input(inputs, shapes.len())?;
    let src = &shapes[src_idx];
    require(src.contiguous, "arena view_2d requires contiguous source")?;
    let mut shape = parse_dimensions(dimensions, Some(2))?;
    let stride = usize::try_from(row_stride_bytes)?;
    let offset = usize::try_from(offset_bytes)?;
    require(
        (1..=MAX_STORAGE).contains(&stride)
            && stride.is_multiple_of(4)
            && stride >= shape.ne[0] * 4,
        "arena view_2d invalid row stride",
    )?;
    require(
        offset <= MAX_STORAGE && offset.is_multiple_of(4),
        "arena view_2d invalid byte offset",
    )?;
    let extent = (shape.ne[1] - 1)
        .checked_mul(stride)
        .and_then(|rows| rows.checked_add(shape.ne[0] * 4))
        .and_then(|span| offset.checked_add(span))
        .ok_or_else(|| invalid("arena view_2d extent overflow"))?;
    require(extent <= src.bytes(), "arena view_2d exceeds source extent")?;
    shape.strides = [4, stride, stride * shape.ne[1], stride * shape.ne[1]];
    shape.contiguous = stride == shape.ne[0] * 4;
    Ok(PlannedNode {
        shape,
        op: PlannedOp::View2d {
            src: src_idx,
            offset_bytes: offset,
        },
    })
}

fn plan_allocation(
    allocation: Option<&AllocationSpec>,
    nodes: &[PlannedNode],
) -> Result<Option<PlannedAllocation>> {
    let Some(alloc) = allocation else {
        return Ok(None);
    };
    let buffer_bytes = usize::try_from(alloc.buffer_bytes)?;
    require(
        (1..=MAX_STORAGE).contains(&buffer_bytes),
        "arena BufferBytes outside bound",
    )?;
    require(
        alloc.offsets.len() == nodes.len(),
        "arena placement count mismatch",
    )?;
    let mut offsets = Vec::with_capacity(nodes.len());
    for (node, &maybe_offset) in nodes.iter().zip(&alloc.offsets) {
        if matches!(node.op, PlannedOp::View2d { .. }) {
            require(
                maybe_offset.is_none(),
                "arena view storage must derive from its source",
            )?;
            offsets.push(None);
        } else {
            let raw = maybe_offset.ok_or_else(|| invalid("arena placement offset missing"))?;
            let offset = usize::try_from(raw)?;
            require(
                offset.is_multiple_of(ALIGNMENT),
                "arena placement is misaligned",
            )?;
            require(
                offset
                    .checked_add(node.shape.bytes())
                    .is_some_and(|end| end <= buffer_bytes),
                "arena placement exceeds buffer",
            )?;
            offsets.push(Some(offset));
        }
    }
    Ok(Some(PlannedAllocation {
        buffer_bytes,
        offsets,
    }))
}

fn plan_arena(
    graph: &GraphSpec,
    forward_raw: &[u64],
    allocation: Option<&AllocationSpec>,
) -> Result<ArenaPlan> {
    require(graph.kind == "tensor-graph", "unknown arena graph Kind")?;
    require((1..=16).contains(&graph.threads), "arena Threads outside bound")?;
    require(
        (1..=96).contains(&graph.nodes.len()),
        "arena graph requires 1..96 nodes",
    )?;
    let mut nodes = Vec::with_capacity(graph.nodes.len());
    let mut shapes = Vec::with_capacity(graph.nodes.len());
    let mut storage = 0_usize;
    for raw_node in &graph.nodes {
        let planned = plan_node(raw_node, &shapes)?;
        storage = storage
            .checked_add(planned.shape.bytes())
            .ok_or_else(|| invalid("arena cumulative storage overflow"))?;
        require(storage <= MAX_STORAGE, "arena cumulative storage exceeds bound")?;
        shapes.push(planned.shape);
        nodes.push(planned);
    }
    let outputs = plan_outputs(&graph.outputs, &nodes)?;
    require(
        !forward_raw.is_empty() && forward_raw.len() <= nodes.len(),
        "arena forward list outside bound",
    )?;
    let mut forward = Vec::with_capacity(forward_raw.len());
    for &raw_idx in forward_raw {
        let idx = usize::try_from(raw_idx)?;
        require(idx < nodes.len(), "arena forward index out of range")?;
        forward.push(idx);
    }
    let planned_alloc = plan_allocation(allocation, &nodes)?;
    Ok(ArenaPlan {
        nodes,
        outputs,
        forward,
        allocation: planned_alloc,
    })
}

fn plan_outputs(raw_outputs: &[u64], nodes: &[PlannedNode]) -> Result<Vec<usize>> {
    require(
        (1..=16).contains(&raw_outputs.len()),
        "arena graph needs 1..16 outputs",
    )?;
    let mut outputs = Vec::with_capacity(raw_outputs.len());
    let mut output_bytes = 0_usize;
    for &raw_idx in raw_outputs {
        let idx = usize::try_from(raw_idx)?;
        let node = nodes
            .get(idx)
            .ok_or_else(|| invalid("arena output index out of range"))?;
        require(node.shape.contiguous, "arena outputs must be contiguous")?;
        output_bytes = output_bytes
            .checked_add(node.shape.bytes())
            .ok_or_else(|| invalid("arena output byte count overflow"))?;
        outputs.push(idx);
    }
    require(
        output_bytes
            .checked_mul(2)
            .is_some_and(|chars| chars <= MAX_OUTPUT_HEX_CHARS),
        "arena response hex exceeds bound",
    )?;
    Ok(outputs)
}

fn visit_forward(idx: usize, nodes: &[PlannedNode], visited: &mut [bool], order: &mut Vec<usize>) {
    if visited[idx] {
        return;
    }
    visited[idx] = true;
    match nodes[idx].op {
        PlannedOp::Input(_) => return,
        PlannedOp::View2d { src, .. } => visit_forward(src, nodes, visited, order),
        PlannedOp::Mul { a, b } | PlannedOp::Add { a, b } => {
            visit_forward(a, nodes, visited, order);
            visit_forward(b, nodes, visited, order);
        }
    }
    order.push(idx);
}

fn build_order(nodes: &[PlannedNode], forward: &[usize]) -> Vec<usize> {
    let mut visited = vec![false; nodes.len()];
    let mut order = Vec::new();
    for &idx in forward {
        visit_forward(idx, nodes, &mut visited, &mut order);
    }
    order
}

fn read_f32(arena: &[u8], offset: usize) -> Result<f32> {
    let slice = arena
        .get(offset..offset + 4)
        .ok_or_else(|| invalid("arena f32 read out of bounds"))?;
    Ok(f32::from_le_bytes(slice.try_into()?))
}

fn write_f32(arena: &mut [u8], offset: usize, value: f32) -> Result<()> {
    let slice = arena
        .get_mut(offset..offset + 4)
        .ok_or_else(|| invalid("arena f32 write out of bounds"))?;
    slice.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn eval_binary(
    arena: &mut [u8],
    left: Operand<'_>,
    right: Operand<'_>,
    dst: Operand<'_>,
    is_mul: bool,
) -> Result<()> {
    let mut dst_pos = dst.offset;
    for i3 in 0..dst.desc.ne[3] {
        let a3 = i3 * left.desc.strides[3];
        let b3 = (i3 % right.desc.ne[3]) * right.desc.strides[3];
        for i2 in 0..dst.desc.ne[2] {
            let a2 = a3 + i2 * left.desc.strides[2];
            let b2 = b3 + (i2 % right.desc.ne[2]) * right.desc.strides[2];
            for i1 in 0..dst.desc.ne[1] {
                let a1 = a2 + i1 * left.desc.strides[1];
                let b1 = b2 + (i1 % right.desc.ne[1]) * right.desc.strides[1];
                for i0 in 0..dst.desc.ne[0] {
                    let va = read_f32(arena, left.offset + a1 + i0 * left.desc.strides[0])?;
                    let vb = read_f32(
                        arena,
                        right.offset + b1 + (i0 % right.desc.ne[0]) * right.desc.strides[0],
                    )?;
                    let value = if is_mul { va * vb } else { va + vb };
                    write_f32(arena, dst_pos, value)?;
                    dst_pos += 4;
                }
            }
        }
    }
    Ok(())
}

fn apply_allocation(alloc: &PlannedAllocation, offsets: &mut [usize]) -> usize {
    for (idx, offset) in alloc
        .offsets
        .iter()
        .enumerate()
        .filter_map(|(idx, maybe)| maybe.map(|offset| (idx, offset)))
    {
        offsets[idx] = offset;
    }
    alloc.buffer_bytes
}

fn pack_offsets(nodes: &[PlannedNode], offsets: &mut [usize]) -> Result<usize> {
    let mut cursor = 0_usize;
    for (idx, node) in nodes.iter().enumerate() {
        if !matches!(node.op, PlannedOp::View2d { .. }) {
            offsets[idx] = cursor;
            cursor = cursor
                .checked_add(node.shape.bytes())
                .ok_or_else(|| invalid("arena layout byte overflow"))?;
        }
    }
    Ok(cursor)
}

fn resolve_arena_layout(plan: &ArenaPlan) -> Result<(usize, Vec<usize>)> {
    let mut offsets = vec![0_usize; plan.nodes.len()];
    let buffer_bytes = plan.allocation.as_ref().map_or_else(
        || pack_offsets(&plan.nodes, &mut offsets),
        |alloc| Ok(apply_allocation(alloc, &mut offsets)),
    )?;
    for (idx, node) in plan.nodes.iter().enumerate() {
        if let PlannedOp::View2d { src, offset_bytes } = node.op {
            offsets[idx] = offsets[src]
                .checked_add(offset_bytes)
                .ok_or_else(|| invalid("arena view offset overflow"))?;
        }
    }
    Ok((buffer_bytes, offsets))
}

fn run_arena(plan: &ArenaPlan) -> Result<Value> {
    let (buffer_bytes, offsets) = resolve_arena_layout(plan)?;
    let mut arena = vec![0_u8; buffer_bytes];
    for (idx, node) in plan.nodes.iter().enumerate() {
        if let PlannedOp::Input(bytes) = &node.op {
            let off = offsets[idx];
            let dst = arena
                .get_mut(off..off + bytes.len())
                .ok_or_else(|| invalid("arena input slice out of bounds"))?;
            dst.copy_from_slice(bytes);
        }
    }
    for idx in build_order(&plan.nodes, &plan.forward) {
        match plan.nodes[idx].op {
            PlannedOp::Input(_) | PlannedOp::View2d { .. } => {}
            PlannedOp::Mul { a, b } | PlannedOp::Add { a, b } => {
                let is_mul = matches!(plan.nodes[idx].op, PlannedOp::Mul { .. });
                eval_binary(
                    &mut arena,
                    Operand {
                        desc: &plan.nodes[a].shape,
                        offset: offsets[a],
                    },
                    Operand {
                        desc: &plan.nodes[b].shape,
                        offset: offsets[b],
                    },
                    Operand {
                        desc: &plan.nodes[idx].shape,
                        offset: offsets[idx],
                    },
                    is_mul,
                )?;
            }
        }
    }
    let mut outputs = Vec::with_capacity(plan.outputs.len());
    for &idx in &plan.outputs {
        let off = offsets[idx];
        let len = plan.nodes[idx].shape.bytes();
        let slice = arena
            .get(off..off + len)
            .ok_or_else(|| invalid("arena output slice out of bounds"))?;
        outputs.push(json!({"DataHex": hex(slice)}));
    }
    Ok(json!({"Status": 0, "Outputs": outputs}))
}

pub(super) fn handles(value: &Value) -> bool {
    value["Operations"].as_array().is_some_and(|operations| {
        !operations.is_empty()
            && operations.iter().all(|operation| {
                matches!(
                    operation["Kind"].as_str(),
                    Some("cpu-buffer-info" | "cpu-arena-graph")
                )
            })
    })
}

pub(super) fn execute(value: &Value) -> Result<Value> {
    let request: Request = serde_json::from_value(value.clone())?;
    require(
        request.schema_version == 1 && (1..=128).contains(&request.operations.len()),
        "arena request schema or operation count outside bound",
    )?;
    let mut plans = Vec::with_capacity(request.operations.len());
    for operation in &request.operations {
        match operation {
            Operation::BufferInfo => plans.push(None),
            Operation::ArenaGraph {
                graph,
                forward,
                allocation,
            } => plans.push(Some(plan_arena(graph, forward, allocation.as_ref())?)),
        }
    }
    let results = plans
        .iter()
        .map(|plan| {
            plan.as_ref().map_or_else(
                || Ok(json!({"Alignment": ALIGNMENT})),
                run_arena,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(json!({"SchemaVersion": 1, "Results": results}))
}

pub(super) fn identity() -> Value {
    json!({
        "Kind": "independent-rust-arena-graph",
        "SupportedSourceIds": [ID],
        "UpstreamCommit": super::UPSTREAM,
        "AssertionSource": {
            "Path": ID,
            "Sha256": "099680dac156e3289db96ad1143973271205d27b878ea07661097fa9eba699eb"
        },
        "Architecture": "flat-arena-byte-buffer-topological-evaluator-v1",
        "OriginalCombinations": 48,
        "NativeReferenceReplay": false,
        "ProductionRuntimeCalled": false,
        "CompleteSourceGate": false,
        "NativeRetirementQualified": false
    })
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::super::{Engine, Target};
    use super::{ALIGNMENT, ID, Result, execute, handles, hex, identity, invalid, unhex};

    fn f32_hex(values: &[f32]) -> String {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        hex(&bytes)
    }

    fn decode_f32_hex(hex_str: &str) -> Result<Vec<f32>> {
        let bytes = unhex(hex_str)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect())
    }

    fn sample_inputs(n_embd: usize, n_exp: usize, n_tok: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let x = (0..n_embd * n_exp * n_tok)
            .map(|i| Ok((f32::from(u16::try_from((i % 29) + 1)?) - 12.0) * 0.125))
            .collect::<Result<Vec<_>>>()?;
        let w = (0..n_exp * n_tok)
            .map(|i| Ok((f32::from(u16::try_from((i % 13) + 1)?) - 4.0) * 0.25))
            .collect::<Result<Vec<_>>>()?;
        Ok((x, w))
    }

    fn moe_sample_graph(n_embd: usize, n_exp: usize, n_tok: usize) -> Result<(Value, Vec<usize>, Vec<usize>)> {
        let (x, w) = sample_inputs(n_embd, n_exp, n_tok)?;
        let mut nodes = vec![
            json!({"Op": "input", "Type": "F32", "Dimensions": [n_embd, n_exp, n_tok], "DataHex": f32_hex(&x)}),
            json!({"Op": "input", "Type": "F32", "Dimensions": [1, n_exp, n_tok], "DataHex": f32_hex(&w)}),
            json!({"Op": "mul", "Inputs": [0, 1]}),
        ];
        let mut sizes = vec![x.len() * 4, w.len() * 4, x.len() * 4];
        let mut views = Vec::new();
        for expert in 0..n_exp {
            views.push(nodes.len());
            nodes.push(json!({
                "Op": "view_2d",
                "Inputs": [2],
                "Dimensions": [n_embd, n_tok],
                "RowStrideBytes": n_embd * n_exp * 4,
                "OffsetBytes": expert * n_embd * 4
            }));
            sizes.push(((n_tok - 1) * n_embd * n_exp + n_embd) * 4);
        }
        let mut adds = Vec::new();
        let mut out = views[0];
        for &view in &views[1..] {
            let prev = out;
            out = nodes.len();
            nodes.push(json!({"Op": "add", "Inputs": [prev, view]}));
            adds.push(out);
            sizes.push(n_embd * n_tok * 4);
        }
        let mut forward = views;
        forward.push(out);
        let op = json!({
            "Kind": "cpu-arena-graph",
            "Graph": {"Kind": "tensor-graph", "Threads": 4, "Nodes": nodes, "Outputs": [out]},
            "Forward": forward,
            "Allocation": Value::Null
        });
        Ok((op, sizes, adds))
    }

    fn layout_allocation(sizes: &[usize], adds: &[usize], layout: usize, alignment: usize) -> Result<Value> {
        let total = sizes.iter().sum::<usize>() + 4096 + sizes.len() * alignment;
        let mut offsets = vec![Value::Null; sizes.len()];
        let mut cursor = 0_usize;
        let mut alloc = |idx: usize| {
            let at = cursor;
            cursor = (cursor + sizes[idx]).div_ceil(alignment) * alignment;
            at
        };
        let x_off = alloc(0);
        offsets[0] = json!(x_off);
        offsets[2] = json!(alloc(2));
        for &add in &adds[..adds.len() - 1] {
            offsets[add] = json!(alloc(add));
        }
        let dst = *adds.last().ok_or_else(|| invalid("missing dst node"))?;
        match layout {
            0 => {
                offsets[dst] = json!(alloc(dst));
                offsets[1] = json!(alloc(1));
            }
            1 => {
                let at = alloc(dst);
                offsets[dst] = json!(at);
                offsets[1] = json!(at + alignment);
            }
            2 => {
                offsets[dst] = json!(x_off + alignment);
                offsets[1] = json!(alloc(1));
            }
            _ => return Err(invalid("unknown layout")),
        }
        Ok(json!({"BufferBytes": total, "Offsets": offsets}))
    }

    #[test]
    fn independent_arena_graph_preserves_all_48_source_cases_and_both_arms() -> Result<()> {
        for target in [Target::RustReference, Target::Cpp, Target::Rust] {
            let engine = Engine::new(target, None, None, None, None, None)?;
            let report = crate::moe_reduce::run(&engine)?;
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
    fn independent_arena_graph_executes_buffer_info_and_all_three_layouts() -> Result<()> {
        let info_req = json!({"SchemaVersion": 1, "Operations": [{"Kind": "cpu-buffer-info"}]});
        assert!(handles(&info_req));
        let info_res = execute(&info_req)?;
        let alignment = usize::try_from(
            info_res["Results"][0]["Alignment"]
                .as_u64()
                .ok_or_else(|| invalid("missing Alignment"))?,
        )?;
        assert!(alignment > 0 && alignment.is_power_of_two() && alignment <= 4096);

        let (n_embd, n_exp, n_tok) = (64, 4, 3);
        let (x, w) = sample_inputs(n_embd, n_exp, n_tok)?;
        let (plain, sizes, adds) = moe_sample_graph(n_embd, n_exp, n_tok)?;
        let mut none_op = plain.clone();
        none_op["Allocation"] = layout_allocation(&sizes, &adds, 0, alignment)?;
        let mut w_in_dst_op = plain.clone();
        w_in_dst_op["Allocation"] = layout_allocation(&sizes, &adds, 1, alignment)?;
        let mut x_in_dst_op = plain.clone();
        x_in_dst_op["Allocation"] = layout_allocation(&sizes, &adds, 2, alignment)?;

        let batch = json!({
            "SchemaVersion": 1,
            "Operations": [
                {"Kind": "cpu-buffer-info"},
                plain,
                none_op,
                w_in_dst_op,
                x_in_dst_op
            ]
        });
        assert!(handles(&batch));
        let response = execute(&batch)?;
        let results = response["Results"]
            .as_array()
            .ok_or_else(|| invalid("Results missing"))?;
        assert_eq!(results.len(), 5);
        let expected_hex = results[1]["Outputs"][0]["DataHex"]
            .as_str()
            .ok_or_else(|| invalid("missing DataHex"))?;
        for result in &results[2..=4] {
            assert_eq!(result["Status"], 0);
            assert_eq!(result["Outputs"][0]["DataHex"], expected_hex);
        }
        let actual = decode_f32_hex(expected_hex)?;
        assert_eq!(actual.len(), n_embd * n_tok);
        for tok in 0..n_tok {
            for col in 0..n_embd {
                let mut acc = 0.0_f32;
                for exp in 0..n_exp {
                    let prod = x[(tok * n_exp + exp) * n_embd + col] * w[tok * n_exp + exp];
                    acc += prod;
                }
                assert_eq!(actual[tok * n_embd + col].to_bits(), acc.to_bits());
            }
        }
        assert!(!handles(&json!({"Operations": []})));
        assert!(!handles(&json!({"Operations": [{"Kind": "tensor-graph"}]})));
        Ok(())
    }

    #[test]
    fn independent_arena_graph_refuses_bounds_nonfinite_and_views() -> Result<()> {
        let (base, _, _) = moe_sample_graph(64, 2, 2)?;
        let wrap = |op: Value| json!({"SchemaVersion": 1, "Operations": [op]});

        let mut bad_nan = base.clone();
        bad_nan["Graph"]["Nodes"][1]["DataHex"] = json!(f32_hex(&[1.0, f32::NAN, 1.0, 1.0]));
        assert!(execute(&wrap(bad_nan)).is_err());

        let mut bad_inf = base.clone();
        bad_inf["Graph"]["Nodes"][1]["DataHex"] = json!(f32_hex(&[1.0, f32::INFINITY, 1.0, 1.0]));
        assert!(execute(&wrap(bad_inf)).is_err());

        let mut bad_i32 = base.clone();
        bad_i32["Graph"]["Nodes"][1]["Type"] = json!("I32");
        assert!(execute(&wrap(bad_i32)).is_err());

        let mut bad_op = base.clone();
        bad_op["Graph"]["Nodes"][2]["Op"] = json!("mul_mat");
        assert!(execute(&wrap(bad_op)).is_err());

        let mut bad_cycle = base.clone();
        bad_cycle["Graph"]["Nodes"][2]["Inputs"] = json!([0, 2]);
        assert!(execute(&wrap(bad_cycle)).is_err());

        let mut bad_broadcast = base.clone();
        bad_broadcast["Graph"]["Nodes"][1]["Dimensions"] = json!([3, 1]);
        bad_broadcast["Graph"]["Nodes"][1]["DataHex"] = json!(f32_hex(&[1.0, 2.0, 3.0]));
        assert!(execute(&wrap(bad_broadcast)).is_err());

        let mut bad_view_out = base.clone();
        bad_view_out["Graph"]["Outputs"] = json!([3]);
        assert!(execute(&wrap(bad_view_out)).is_err());

        let mut bad_view_stride = base.clone();
        bad_view_stride["Graph"]["Nodes"][3]["RowStrideBytes"] = json!(63 * 4);
        assert!(execute(&wrap(bad_view_stride)).is_err());

        let mut bad_view_offset = base.clone();
        bad_view_offset["Graph"]["Nodes"][3]["OffsetBytes"] = json!(2);
        assert!(execute(&wrap(bad_view_offset)).is_err());

        let mut bad_view_oob = base;
        bad_view_oob["Graph"]["Nodes"][3]["OffsetBytes"] = json!(100 * 4);
        assert!(execute(&wrap(bad_view_oob)).is_err());

        assert!(execute(&json!({"SchemaVersion": 2, "Operations": [{"Kind": "cpu-buffer-info"}]})).is_err());
        assert!(execute(&json!({"SchemaVersion": 1, "Operations": [{"Kind": "cpu-buffer-info", "Extra": 1}]})).is_err());
        assert!(execute(&json!({"SchemaVersion": 1, "Operations": []})).is_err());
        Ok(())
    }

    #[test]
    fn independent_arena_graph_refuses_invalid_placements_and_forward() -> Result<()> {
        let (base, sizes, adds) = moe_sample_graph(64, 2, 2)?;
        let wrap = |op: Value| json!({"SchemaVersion": 1, "Operations": [op]});

        let mut bad_forward = base.clone();
        bad_forward["Forward"] = json!([]);
        assert!(execute(&wrap(bad_forward)).is_err());

        let mut bad_forward_oob = base.clone();
        bad_forward_oob["Forward"] = json!([99]);
        assert!(execute(&wrap(bad_forward_oob)).is_err());

        let mut missing_alloc_key = base.clone();
        missing_alloc_key
            .as_object_mut()
            .ok_or_else(|| invalid("not an object"))?
            .remove("Allocation");
        assert!(execute(&wrap(missing_alloc_key)).is_err());

        let alloc = layout_allocation(&sizes, &adds, 0, ALIGNMENT)?;
        let mut misaligned = base.clone();
        misaligned["Allocation"] = alloc.clone();
        misaligned["Allocation"]["Offsets"][1] = json!(4);
        assert!(execute(&wrap(misaligned)).is_err());

        let mut exceeds_buf = base.clone();
        exceeds_buf["Allocation"] = alloc.clone();
        let buf_bytes = exceeds_buf["Allocation"]["BufferBytes"]
            .as_u64()
            .ok_or_else(|| invalid("missing BufferBytes"))?;
        exceeds_buf["Allocation"]["Offsets"][0] = json!(buf_bytes);
        assert!(execute(&wrap(exceeds_buf)).is_err());

        let mut view_with_offset = base.clone();
        view_with_offset["Allocation"] = alloc.clone();
        view_with_offset["Allocation"]["Offsets"][3] = json!(0);
        assert!(execute(&wrap(view_with_offset)).is_err());

        let mut null_non_view = base;
        null_non_view["Allocation"] = alloc;
        null_non_view["Allocation"]["Offsets"][0] = Value::Null;
        assert!(execute(&wrap(null_non_view)).is_err());
        Ok(())
    }
}
