//! All 13 default graph-allocation scenarios from pinned test-alloc.cpp.
//!
//! The generic adapter constructs supplied graphs and configurable buffer inputs.
//! Rust owns every source allocation, buffer-size, view-root and overlap/liveness
//! assertion. Upstream MIT attribution: tools/prism/upstream/LICENSE.

use super::engine::Engine;
use super::{Result, invalid, result};
use serde_json::{Value, json};

const ID: &str = "tests/test-alloc.cpp";
const SOURCE_SHA: &str = "89de1c9549888a37b388474aca89cad3052c1751fbdace09c81e79d5844633db";

/// Construct caller-supplied tensors and serialize production allocator observations.
#[allow(
    clippy::too_many_lines,
    reason = "The adapter only marshals graph calls and raw allocation metadata."
)]
pub(super) fn execute(operation: &Value) -> Result<Value> {
    use bonsai_prism_runtime::backend::allocator::{BufferType, GraphAllocator, can_inplace};
    use bonsai_prism_runtime::backend::{Argument, CpuBackend, Observation, TensorId};
    let descriptors = array(&operation["Backends"])?;
    if descriptors.is_empty() || descriptors.len() > 3 {
        return Err(invalid("allocation backend count outside bound"));
    }
    let mut types = Vec::new();
    for descriptor in descriptors {
        let alignment = number(&descriptor["Alignment"])?;
        if alignment > 64 {
            return Err(invalid("allocation alignment outside bound"));
        }
        types.push(BufferType {
            alignment,
            maximum: if descriptor["MaxBufferSize"].is_null() {
                None
            } else {
                Some(number(&descriptor["MaxBufferSize"])?)
            },
        });
    }
    let mut allocator = GraphAllocator::new(types)?;
    let steps = array(&operation["Steps"])?;
    if steps.is_empty() || steps.len() > 4 {
        return Err(invalid("allocation step count outside bound"));
    }
    let mut observations = Vec::new();
    for step in steps {
        let descriptions = array(&step["Nodes"])?;
        if descriptions.is_empty() || descriptions.len() > 40 {
            return Err(invalid("allocation node count outside bound"));
        }
        let mut backend = CpuBackend::new();
        let mut tensors: Vec<TensorId> = Vec::new();
        for node in descriptions {
            let context = u32::try_from(number(&node["Context"])?)?;
            let op = node["Op"]
                .as_str()
                .ok_or_else(|| invalid("allocation node Op required"))?;
            let tensor = if op == "input" {
                let id = backend.tensor(context, 0, [number(&node["Elements"])?, 1, 1, 1])?;
                backend.info_mut(id)?.flags |= 1;
                id
            } else {
                let mut arguments = vec![Argument::Context(context)];
                for input in array(&node["Inputs"])? {
                    arguments.push(Argument::Tensor(
                        *tensors
                            .get(number(input)?)
                            .ok_or_else(|| invalid("allocation input must precede node"))?,
                    ));
                }
                let function = match op {
                    "add" => "ggml_add",
                    "mul" => "ggml_mul",
                    "concat" => {
                        arguments.push(Argument::Integer(0));
                        "ggml_concat"
                    }
                    "mean" => "ggml_mean",
                    "scale" => {
                        let factor: f32 = serde_json::from_value(node["Factor"].clone())?;
                        if !factor.is_finite() {
                            return Err(invalid("allocation scale factor must be finite"));
                        }
                        arguments.push(Argument::Float(factor));
                        "ggml_scale"
                    }
                    "pad" => {
                        arguments.push(Argument::Integer(i64::try_from(number(&node["Amount"])?)?));
                        arguments.extend([
                            Argument::Integer(0),
                            Argument::Integer(0),
                            Argument::Integer(0),
                        ]);
                        "ggml_pad"
                    }
                    "reshape" => {
                        for value in array(&node["Dimensions"])? {
                            arguments.push(Argument::Integer(i64::try_from(number(value)?)?));
                        }
                        "ggml_reshape_2d"
                    }
                    "permute" => {
                        for value in array(&node["Axes"])? {
                            arguments.push(Argument::Integer(i64::try_from(number(value)?)?));
                        }
                        "ggml_permute"
                    }
                    "view" => {
                        arguments.push(Argument::Integer(i64::try_from(number(
                            &node["Elements"],
                        )?)?));
                        arguments.push(Argument::Integer(i64::try_from(number(
                            &node["OffsetBytes"],
                        )?)?));
                        "ggml_view_1d"
                    }
                    _ => return Err(invalid("unknown allocation graph operation")),
                };
                let Observation::Tensor(Some(id)) = backend.call(function, &arguments)? else {
                    return Err(invalid("allocation operation did not return tensor"));
                };
                id
            };
            tensors.push(tensor);
        }
        let output = *tensors
            .get(number(&step["Output"])?)
            .ok_or_else(|| invalid("allocation output is unknown"))?;
        backend.info_mut(output)?.flags |= 2;
        let Observation::Graph(graph) = backend.call("ggml_new_graph", &[Argument::Context(0)])?
        else {
            return Err(invalid("allocation graph creation failed"));
        };
        backend.call(
            "ggml_build_forward_expand",
            &[Argument::Graph(graph), Argument::Tensor(output)],
        )?;
        let kinds = array(&step["BufferTypes"])?
            .iter()
            .map(number)
            .collect::<Result<Vec<_>>>()?;
        if kinds.len() != tensors.len() {
            return Err(invalid("allocation buffer mapping extent differs"));
        }
        let external = array(&step["ExternalAllocations"])?
            .iter()
            .map(|value| {
                Ok((
                    u32::try_from(number(&value["Context"])?)?,
                    number(&value["BufferType"])?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let reserve = boolean(&step["Reserve"])?;
        let allocation = allocator.reserve(
            &backend,
            graph,
            if reserve { &kinds } else { &[] },
            &external,
        )?;
        GraphAllocator::allocate(&mut backend, &allocation)?;
        let buffers = allocation.buffers.iter().enumerate().map(|(index, buffer)| json!({"Id":index,"Type":buffer.kind,"Size":buffer.bytes,"Maximum":descriptors[buffer.kind]["MaxBufferSize"]})).collect::<Vec<_>>();
        let tensor_values = tensors.iter().enumerate().map(|(index, &id)| -> Result<Value> {
            let info = backend.info(id)?;
            let placement = allocation.tensors[id.0];
            Ok(json!({"Index":index,"Context":info.context,"Buffer":placement.map(|p|p.buffer),"DataPresent":placement.is_some(),"Offset":placement.map(|p|p.offset),"Bytes":info.bytes()?,"ViewSource":info.view.map(|id|id.0),"Sources":info.source_slots.map(|source|source.map(|id|id.0)),"Op":info.operation,"Flags":info.flags,"CanInplace":can_inplace(info.operation),"Output":info.flags&2!=0}))
        }).collect::<Result<Vec<_>>>()?;
        let (nodes, leaves) = backend.graph_nodes(graph)?;
        observations.push(json!({"Allocated":true,"Reserved":reserve.then_some(true),"Tensors":tensor_values,"Buffers":buffers,"GraphNodes":nodes.iter().map(|id|id.0).collect::<Vec<_>>(),"GraphLeaves":leaves.iter().map(|id|id.0).collect::<Vec<_>>() }));
    }
    Ok(json!({"Steps":observations}))
}

#[derive(Default)]
struct Graph {
    nodes: Vec<Value>,
}
impl Graph {
    fn node(&mut self, value: Value) -> usize {
        let id = self.nodes.len();
        self.nodes.push(value);
        id
    }
    fn input(&mut self, bytes: usize, context: usize) -> usize {
        self.node(json!({"Op":"input","Context":context,"Elements":bytes/4}))
    }
    fn binary(&mut self, op: &str, left: usize, right: usize) -> usize {
        self.node(json!({"Op":op,"Context":0,"Inputs":[left,right]}))
    }
    fn mean(&mut self, input: usize) -> usize {
        self.node(json!({"Op":"mean","Context":0,"Inputs":[input]}))
    }
    fn scale(&mut self, input: usize) -> usize {
        self.node(json!({"Op":"scale","Context":0,"Inputs":[input],"Factor":2.0}))
    }
    fn pad(&mut self, input: usize, amount: usize) -> usize {
        self.node(json!({"Op":"pad","Context":0,"Inputs":[input],"Amount":amount}))
    }
    fn view(&mut self, input: usize, elements: usize, offset: usize) -> usize {
        self.node(json!({"Op":"view","Context":0,"Inputs":[input],"Elements":elements,"OffsetBytes":offset}))
    }
}
enum Status {
    Allocate,
    ReserveAndAllocate,
    Ignore,
}
enum Comparison {
    AtMost,
    Exactly,
    Greater,
}
struct Total {
    backend: usize,
    comparison: Comparison,
    bytes: u64,
}
struct Step {
    graph: Graph,
    output: usize,
    buffer_types: Vec<usize>,
    external: Vec<Value>,
    reserve: bool,
    rules: Rules,
}
struct Rules {
    status: Status,
    overlap: bool,
    maximum: bool,
    totals: Vec<Total>,
    graph_counts: Option<(usize, usize)>,
}
impl Rules {
    fn standard(maximum: bool, bytes: u64) -> Self {
        Self {
            status: Status::Allocate,
            overlap: true,
            maximum,
            totals: vec![Total {
                backend: 0,
                comparison: Comparison::AtMost,
                bytes,
            }],
            graph_counts: None,
        }
    }
}
impl Step {
    fn new(graph: Graph, output: usize, rules: Rules) -> Self {
        let buffer_types = vec![0; graph.nodes.len()];
        Self {
            graph,
            output,
            buffer_types,
            external: Vec::new(),
            reserve: false,
            rules,
        }
    }
    fn request(&self) -> Value {
        json!({"Nodes":self.graph.nodes,"Output":self.output,"BufferTypes":self.buffer_types,"ExternalAllocations":self.external,"Reserve":self.reserve})
    }
}
struct Case {
    name: &'static str,
    backends: Vec<Value>,
    steps: Vec<Step>,
}
impl Case {
    fn new(name: &'static str, maximum: Option<usize>, alignment: usize, step: Step) -> Self {
        Self {
            name,
            backends: vec![json!({"MaxBufferSize":maximum,"Alignment":alignment})],
            steps: vec![step],
        }
    }
    fn request(&self) -> Value {
        json!({"Kind":"allocation-graph","Backends":self.backends,"Steps":self.steps.iter().map(Step::request).collect::<Vec<_>>()})
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "Keep all literal source graph scenarios and bounds in source order for parity review."
)]
fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    let mut g = Graph::default();
    let x0 = g.input(8, 0);
    let x1 = g.input(8, 0);
    let x2 = g.input(8, 0);
    let x3 = g.binary("mul", x0, x1);
    let x4 = g.binary("add", x1, x2);
    let x5 = g.binary("add", x3, x0);
    let x6 = g.binary("add", x4, x5);
    cases.push(Case::new(
        "max_size_too_many_tensors",
        Some(16),
        8,
        Step::new(g, x6, Rules::standard(true, 32)),
    ));
    let mut g = Graph::default();
    let x0 = g.input(16, 0);
    let x1 = g.input(8, 0);
    let x2 = g.binary("concat", x0, x1);
    cases.push(Case::new(
        "max_size_tensor_too_large",
        Some(32),
        8,
        Step::new(g, x2, Rules::standard(true, 56)),
    ));
    let mut g = Graph::default();
    let x0 = g.input(24, 0);
    let x1 = g.scale(x0);
    let mut r = Rules::standard(false, 24);
    r.totals[0].comparison = Comparison::Exactly;
    cases.push(Case::new(
        "tensor_larger_than_max_size",
        Some(16),
        8,
        Step::new(g, x1, r),
    ));
    let mut g = Graph::default();
    let x0 = g.input(8, 0);
    let mut inputs = vec![x0];
    for _ in 0..16 {
        inputs.push(g.input(8, 0));
    }
    let mut acc = x0;
    for input in &inputs[1..] {
        acc = g.binary("add", acc, *input);
    }
    let mut r = Rules::standard(false, 128);
    r.totals[0].comparison = Comparison::Greater;
    cases.push(Case::new(
        "not_enough_chunks",
        Some(8),
        8,
        Step::new(g, acc, r),
    ));
    let mut g = Graph::default();
    let x0 = g.input(8, 0);
    let x1 = g.pad(x0, 2);
    let x3 = g.mean(x1);
    cases.push(Case::new(
        "fill_leftover_space",
        Some(16),
        8,
        Step::new(g, x3, Rules::standard(true, 28)),
    ));
    let mut g = Graph::default();
    let x0 = g.input(16, 0);
    let x1 = g.node(json!({"Op":"reshape","Context":0,"Inputs":[x0],"Dimensions":[2,2]}));
    let x2 = g.node(json!({"Op":"permute","Context":0,"Inputs":[x1],"Axes":[1,0,2,3]}));
    let x3 = g.view(x2, 2, 4);
    let x4 = g.input(8, 0);
    let x5 = g.binary("add", x3, x4);
    cases.push(Case::new(
        "view_inplace",
        Some(32),
        8,
        Step::new(g, x5, Rules::standard(true, 24)),
    ));
    let mut g = Graph::default();
    let x0 = g.input(24, 0);
    let x1 = g.input(8, 0);
    let x2 = g.input(8, 0);
    let x3 = g.binary("add", x1, x2);
    let x4 = g.pad(x0, 2);
    let x5 = g.scale(x4);
    let x6 = g.binary("add", x4, x5);
    let x7 = g.view(x6, 2, 8);
    let x8 = g.binary("add", x3, x7);
    cases.push(Case::new(
        "reuse_and_free",
        Some(40),
        8,
        Step::new(g, x8, Rules::standard(true, 104)),
    ));
    for (name, maximum) in [
        ("merge_free_block_32", Some(32)),
        ("merge_free_block_size_max", None),
    ] {
        let mut g = Graph::default();
        let x0 = g.input(16, 0);
        let x1 = g.input(16, 0);
        let x2 = g.input(16, 0);
        let x3 = g.mean(x0);
        let x4 = g.mean(x1);
        let x5 = g.pad(x2, 2);
        let x6 = g.binary("add", x3, x4);
        let x7 = g.pad(x6, 5);
        let x8 = g.binary("add", x5, x7);
        cases.push(Case::new(
            name,
            maximum,
            8,
            Step::new(g, x8, Rules::standard(true, 88)),
        ));
    }
    let mut g = Graph::default();
    let x0 = g.input(24, 0);
    let x1 = g.mean(x0);
    let x2 = g.mean(x1);
    cases.push(Case::new(
        "prefer_already_allocated_memory",
        Some(32),
        4,
        Step::new(g, x2, Rules::standard(false, 28)),
    ));
    let mut g = Graph::default();
    let a0 = g.input(16, 1);
    let a1 = g.input(16, 1);
    let b0 = g.input(24, 2);
    let b1 = g.input(4, 2);
    let x0 = g.input(16, 0);
    let x1 = g.binary("mul", x0, a0);
    let x2 = g.pad(x1, 2);
    let x3 = g.binary("mul", x2, b0);
    let x4 = g.mean(x3);
    let x5 = g.binary("add", x4, b1);
    let x6 = g.pad(x5, 3);
    let x7 = g.binary("add", x6, a1);
    let x8 = g.scale(x7);
    let mut rules = Rules::standard(true, 88);
    rules.status = Status::Ignore;
    rules.totals.push(Total {
        backend: 1,
        comparison: Comparison::AtMost,
        bytes: 56,
    });
    rules.graph_counts = Some((5, 8));
    let mut step = Step::new(g, x8, rules);
    step.reserve = true;
    step.external = vec![
        json!({"Context":1,"BufferType":0}),
        json!({"Context":2,"BufferType":1}),
    ];
    step.buffer_types = vec![0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0];
    let mut case = Case::new("multiple_buffer_types", Some(32), 8, step);
    case.backends
        .push(json!({"MaxBufferSize":Value::Null,"Alignment":8}));
    cases.push(case);
    let mut g = Graph::default();
    let x0 = g.input(16, 0);
    let x1 = g.scale(x0);
    let mut rules = Rules::standard(false, 16);
    rules.status = Status::ReserveAndAllocate;
    rules.overlap = false;
    rules.totals[0].comparison = Comparison::Exactly;
    rules.totals.push(Total {
        backend: 1,
        comparison: Comparison::Exactly,
        bytes: 0,
    });
    let mut step = Step::new(g, x1, rules);
    step.reserve = true;
    let mut case = Case::new("buffer_size_zero", None, 8, step);
    case.backends
        .push(json!({"MaxBufferSize":Value::Null,"Alignment":8}));
    cases.push(case);
    let mut g = Graph::default();
    let x0 = g.input(24, 0);
    let x1 = g.input(16, 0);
    let x2 = g.view(x0, 4, 0);
    let x3 = g.binary("add", x2, x1);
    let mut rules = Rules::standard(false, 40);
    rules.overlap = false;
    rules.totals[0].comparison = Comparison::Exactly;
    let mut case = Case::new("reallocation", Some(32), 4, Step::new(g, x3, rules));
    let mut g = Graph::default();
    let x0 = g.input(20, 0);
    let x1 = g.input(20, 0);
    let x2 = g.binary("add", x0, x1);
    let mut rules = Rules::standard(false, 40);
    rules.overlap = false;
    rules.totals[0].comparison = Comparison::Exactly;
    case.steps.push(Step::new(g, x2, rules));
    cases.push(case);
    cases
}

fn fields(value: &Value, names: &[&str]) -> Result<()> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid("allocation observation must be object"))?;
    if object.len() != names.len() || names.iter().any(|name| !object.contains_key(*name)) {
        return Err(invalid("allocation observation fields mismatch"));
    }
    Ok(())
}
fn array(value: &Value) -> Result<&[Value]> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| invalid("allocation array required"))
}
fn number(value: &Value) -> Result<usize> {
    usize::try_from(
        value
            .as_u64()
            .ok_or_else(|| invalid("allocation unsigned integer required"))?,
    )
    .map_err(Into::into)
}
fn boolean(value: &Value) -> Result<bool> {
    value
        .as_bool()
        .ok_or_else(|| invalid("allocation boolean required"))
}
fn optional_id(value: &Value, bound: usize) -> Result<Option<usize>> {
    if value.is_null() {
        return Ok(None);
    }
    let id = number(value)?;
    if id >= bound {
        return Err(invalid("allocation tensor/buffer reference out of bounds"));
    }
    Ok(Some(id))
}
struct Tensor {
    context: usize,
    buffer: Option<usize>,
    data: bool,
    offset: Option<i64>,
    bytes: u64,
    view: Option<usize>,
    sources: Vec<Option<usize>>,
    inplace: bool,
    output: bool,
}
struct Buffer {
    kind: usize,
    size: u64,
    maximum: u64,
}
struct Observation {
    allocated: bool,
    reserved: Option<bool>,
    tensors: Vec<Tensor>,
    buffers: Vec<Buffer>,
    nodes: Vec<usize>,
    leaves: Vec<usize>,
}
fn read_buffers(value: &Value, backends: usize) -> Result<Vec<Buffer>> {
    let buffer_values = array(value)?;
    if buffer_values.len() > 64 {
        return Err(invalid("allocation buffer count exceeds bound"));
    }
    buffer_values
        .iter()
        .enumerate()
        .map(|(id, value)| {
            fields(value, &["Id", "Type", "Size", "Maximum"])?;
            let kind = number(&value["Type"])?;
            if number(&value["Id"])? != id || kind >= backends {
                return Err(invalid("allocation buffer ID/type mismatch"));
            }
            Ok(Buffer {
                kind,
                size: value["Size"]
                    .as_u64()
                    .ok_or_else(|| invalid("allocation buffer size must be unsigned"))?,
                maximum: if value["Maximum"].is_null() {
                    u64::MAX
                } else {
                    value["Maximum"]
                        .as_u64()
                        .ok_or_else(|| invalid("allocation maximum must be unsigned"))?
                },
            })
        })
        .collect()
}
fn read_tensors(value: &Value, step: &Step, buffer_count: usize) -> Result<Vec<Tensor>> {
    let tensor_values = array(value)?;
    if tensor_values.len() != step.graph.nodes.len() {
        return Err(invalid("allocation tensor count mismatch"));
    }
    let count = tensor_values.len();
    tensor_values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            fields(
                value,
                &[
                    "Index",
                    "Context",
                    "Buffer",
                    "DataPresent",
                    "Offset",
                    "Bytes",
                    "ViewSource",
                    "Sources",
                    "Op",
                    "Flags",
                    "CanInplace",
                    "Output",
                ],
            )?;
            if number(&value["Index"])? != index
                || value["Context"] != step.graph.nodes[index]["Context"]
            {
                return Err(invalid("allocation tensor index/context mismatch"));
            }
            number(&value["Op"])?;
            number(&value["Flags"])?;
            let sources = array(&value["Sources"])?;
            if sources.len() != 10 {
                return Err(invalid("allocation source slot count mismatch"));
            }
            Ok(Tensor {
                context: number(&value["Context"])?,
                buffer: optional_id(&value["Buffer"], buffer_count)?,
                data: boolean(&value["DataPresent"])?,
                offset: if value["Offset"].is_null() {
                    None
                } else {
                    Some(
                        value["Offset"]
                            .as_i64()
                            .ok_or_else(|| invalid("allocation offset must fit signed64"))?,
                    )
                },
                bytes: value["Bytes"]
                    .as_u64()
                    .ok_or_else(|| invalid("allocation tensor size required"))?,
                view: optional_id(&value["ViewSource"], count)?,
                sources: sources
                    .iter()
                    .map(|id| optional_id(id, count))
                    .collect::<Result<Vec<_>>>()?,
                inplace: boolean(&value["CanInplace"])?,
                output: boolean(&value["Output"])?,
            })
        })
        .collect()
}
fn observation(value: &Value, step: &Step, backends: usize) -> Result<Observation> {
    fields(
        value,
        &[
            "Allocated",
            "Reserved",
            "Tensors",
            "Buffers",
            "GraphNodes",
            "GraphLeaves",
        ],
    )?;
    let buffers = read_buffers(&value["Buffers"], backends)?;
    let tensors = read_tensors(&value["Tensors"], step, buffers.len())?;
    let count = tensors.len();
    let ids = |values: &Value| -> Result<Vec<usize>> {
        array(values)?
            .iter()
            .map(|value| {
                optional_id(value, count)?
                    .ok_or_else(|| invalid("allocation graph IDs cannot be null"))
            })
            .collect()
    };
    let nodes = ids(&value["GraphNodes"])?;
    let leaves = ids(&value["GraphLeaves"])?;
    // Added structural guard: no source graph operation/leaf may disappear and
    // reduce the number of assertions in the original graph-driven checks.
    let mut expected_nodes = Vec::new();
    let mut expected_leaves = Vec::new();
    for (index, node) in step.graph.nodes.iter().enumerate() {
        if node["Op"] == "input" {
            expected_leaves.push(index);
        } else {
            expected_nodes.push(index);
        }
    }
    let mut sorted_nodes = nodes.clone();
    sorted_nodes.sort_unstable();
    let mut sorted_leaves = leaves.clone();
    sorted_leaves.sort_unstable();
    if sorted_nodes != expected_nodes || sorted_leaves != expected_leaves {
        return Err(invalid(
            "production graph omitted/duplicated a supplied operation or leaf",
        ));
    }
    Ok(Observation {
        allocated: boolean(&value["Allocated"])?,
        reserved: if value["Reserved"].is_null() {
            None
        } else {
            Some(boolean(&value["Reserved"])?)
        },
        tensors,
        buffers,
        nodes,
        leaves,
    })
}

#[derive(Default)]
struct Checks {
    original: usize,
    conditional: usize,
    failed: usize,
    examples: Vec<String>,
}
impl Checks {
    fn check(&mut self, passed: bool, label: String, conditional: bool) {
        self.original += 1;
        if conditional {
            self.conditional += 1;
        }
        if !passed {
            self.failed += 1;
            if self.examples.len() < 12 {
                self.examples.push(label);
            }
        }
    }
}
fn view_root(tensors: &[Tensor], mut id: usize) -> Result<usize> {
    for _ in 0..=tensors.len() {
        if let Some(source) = tensors[id].view {
            id = source;
        } else {
            return Ok(id);
        }
    }
    Err(invalid("production allocation view cycle"))
}
fn overlaps(left: &Tensor, right: &Tensor) -> bool {
    if left.buffer.is_none() || left.buffer != right.buffer {
        return false;
    }
    let (Some(a), Some(b)) = (left.offset, right.offset) else {
        return false;
    };
    let a = i128::from(a);
    let b = i128::from(b);
    a + i128::from(left.bytes) > b && b + i128::from(right.bytes) > a
}
fn reusable(observation: &Observation, current: usize, current_id: usize, other_id: usize) -> bool {
    if observation.tensors[other_id].output {
        return false;
    }
    for id in &observation.nodes[current..] {
        let tensor = &observation.tensors[*id];
        for source in &tensor.sources {
            if *id == current_id && tensor.inplace {
                continue;
            }
            if *source == Some(other_id)
                || source.is_some_and(|id| observation.tensors[id].view == Some(other_id))
            {
                return false;
            }
        }
    }
    true
}
fn check_step(name: &str, step: &Step, actual: &Observation, checks: &mut Checks) -> Result<()> {
    match step.rules.status {
        Status::Allocate => checks.check(
            actual.allocated,
            format!("{name}: graph allocation failed"),
            false,
        ),
        Status::ReserveAndAllocate => checks.check(
            actual.reserved == Some(true) && actual.allocated,
            format!("{name}: reserve/allocate failed"),
            false,
        ),
        Status::Ignore => {}
    }
    for id in &actual.nodes {
        let tensor = &actual.tensors[*id];
        checks.check(
            tensor.buffer.is_some(),
            format!("{name}: node{id} has no buffer"),
            false,
        );
        checks.check(tensor.data, format!("{name}: node{id} has no data"), false);
    }
    if step.rules.maximum {
        for (index, tensor) in actual
            .tensors
            .iter()
            .enumerate()
            .filter(|(_, t)| t.context == 0)
        {
            let offset = tensor.offset.and_then(|offset| u64::try_from(offset).ok());
            checks.check(
                offset.is_some(),
                format!("{name}: tensor{index} data precedes buffer base"),
                false,
            );
            let bound = tensor.buffer.map(|buffer| actual.buffers[buffer].maximum);
            checks.check(
                offset.zip(bound).is_some_and(|(offset, bound)| {
                    offset
                        .checked_add(tensor.bytes)
                        .is_some_and(|end| end <= bound)
                }),
                format!("{name}: tensor{index} exceeds maximum buffer size"),
                false,
            );
        }
    }
    if step.rules.overlap {
        for (i, id) in actual.nodes.iter().enumerate() {
            for other in &actual.nodes[..i] {
                checks.check(id != other, format!("{name}: repeated graph node"), false);
                if view_root(&actual.tensors, *id)? == view_root(&actual.tensors, *other)? {
                    continue;
                }
                if overlaps(&actual.tensors[*id], &actual.tensors[*other]) {
                    checks.check(
                        reusable(actual, i, *id, *other),
                        format!("{name}: node{id} illegally overlaps live node{other}"),
                        true,
                    );
                }
            }
        }
    }
    if let Some((leaves, nodes)) = step.rules.graph_counts {
        checks.check(
            actual.leaves.len() == leaves,
            format!("{name}: graph leaf count mismatch"),
            false,
        );
        checks.check(
            actual.nodes.len() == nodes,
            format!("{name}: graph node count mismatch"),
            false,
        );
    }
    for total in &step.rules.totals {
        let bytes = actual
            .buffers
            .iter()
            .filter(|b| b.kind == total.backend)
            .try_fold(0_u64, |sum, b| {
                sum.checked_add(b.size)
                    .ok_or_else(|| invalid("allocation total overflow"))
            })?;
        let passed = match total.comparison {
            Comparison::AtMost => bytes <= total.bytes,
            Comparison::Exactly => bytes == total.bytes,
            Comparison::Greater => bytes > total.bytes,
        };
        checks.check(
            passed,
            format!(
                "{name}: backend{} allocated {bytes} bytes, expected bound{}",
                total.backend, total.bytes
            ),
            false,
        );
    }
    Ok(())
}

pub(super) fn run(engine: &Engine) -> Result<Value> {
    let cases = cases();
    if cases.len() != 13 || cases.iter().map(|c| c.steps.len()).sum::<usize>() != 14 {
        return Err(invalid("allocation source case count mismatch"));
    }
    let request =
        json!({"SchemaVersion":1,"Operations":cases.iter().map(Case::request).collect::<Vec<_>>()});
    let Some(response) = engine.execute_common("allocation-graphs", &request)? else {
        return Ok(result(
            ID,
            false,
            "unsupported: production graph allocator API is missing".to_owned(),
        ));
    };
    fields(&response, &["SchemaVersion", "Results"])?;
    if response["SchemaVersion"] != 1 {
        return Err(invalid("allocation response version mismatch"));
    }
    let values = array(&response["Results"])?;
    if values.len() != cases.len() {
        return Err(invalid("allocation result count mismatch"));
    }
    let mut checks = Checks::default();
    for (case, value) in cases.iter().zip(values) {
        fields(value, &["Steps"])?;
        let steps = array(&value["Steps"])?;
        if steps.len() != case.steps.len() {
            return Err(invalid("allocation step count mismatch"));
        }
        for (step, value) in case.steps.iter().zip(steps) {
            check_step(
                case.name,
                step,
                &observation(value, step, case.backends.len())?,
                &mut checks,
            )?;
        }
    }
    if checks.original - checks.conditional != 468 {
        return Err(invalid(
            "allocation unconditional source assertion count mismatch",
        ));
    }
    Ok(result(
        ID,
        checks.failed == 0,
        format!(
            "source={SOURCE_SHA}; all13 scenarios,14 graph allocations; original assertions={} (468 unconditional+{} overlap-dependent); failures={}; examples={:?}",
            checks.original, checks.conditional, checks.failed, checks.examples
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_view_cycle_and_live_overlap() {
        let tensor = |view, sources, inplace| Tensor {
            context: 0,
            buffer: Some(0),
            data: true,
            offset: Some(0),
            bytes: 8,
            view,
            sources,
            inplace,
            output: false,
        };
        let mut actual = Observation {
            allocated: true,
            reserved: None,
            tensors: vec![
                tensor(None, vec![], false),
                tensor(None, vec![Some(0)], false),
            ],
            buffers: vec![Buffer {
                kind: 0,
                size: 8,
                maximum: 8,
            }],
            nodes: vec![0, 1],
            leaves: vec![],
        };
        assert!(overlaps(&actual.tensors[0], &actual.tensors[1]));
        assert!(!reusable(&actual, 1, 1, 0));
        actual.tensors[1].inplace = true;
        assert!(reusable(&actual, 1, 1, 0));
        actual.tensors[0].output = true;
        assert!(!reusable(&actual, 1, 1, 0));
        actual.tensors[0].view = Some(1);
        actual.tensors[1].view = Some(0);
        assert!(view_root(&actual.tensors, 0).is_err());
        // Only count source fixture topology and assertion descriptors. This
        // does not execute any production source scenario locally.
        let inventory = cases();
        assert_eq!(inventory.len(), 13);
        let unconditional: usize = inventory
            .iter()
            .flat_map(|case| &case.steps)
            .map(|step| {
                let nodes = step
                    .graph
                    .nodes
                    .iter()
                    .filter(|node| node["Op"] != "input")
                    .count();
                let maximum = if step.rules.maximum {
                    2 * step
                        .graph
                        .nodes
                        .iter()
                        .filter(|node| node["Context"] == 0)
                        .count()
                } else {
                    0
                };
                let pairs = if step.rules.overlap {
                    nodes * (nodes - 1) / 2
                } else {
                    0
                };
                usize::from(!matches!(step.rules.status, Status::Ignore))
                    + 2 * nodes
                    + maximum
                    + pairs
                    + step.rules.totals.len()
                    + usize::from(step.rules.graph_counts.is_some()) * 2
            })
            .sum();
        assert_eq!(unconditional, 468);
    }
}
