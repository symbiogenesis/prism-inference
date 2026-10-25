// Copyright (c) 2023-2026 The ggml/llama.cpp authors
// SPDX-License-Identifier: MIT
//! Exact source graph preparation, support/skip decisions and CRT input planning.

use super::super::{Result, invalid};
use super::host::Production;
use super::interpreter::{Host, Interpreter};
use super::number::Number;
use super::program::{Geometry, Program};
use super::value::{Handle, Scope, Value, variable};
use serde::{Deserialize, Serialize};
use serde_json::{Value as Json, json};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum Mode {
    Eval,
    Grad,
    Support,
    Perf,
}
impl Mode {
    pub(super) const fn code(self) -> i32 {
        match self {
            Self::Eval => 0,
            Self::Perf => 1,
            Self::Grad => 2,
            Self::Support => 3,
        }
    }
    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::Eval => "eval",
            Self::Grad => "grad",
            Self::Support => "support",
            Self::Perf => "perf",
        }
    }
    pub(super) fn parse(text: &str) -> Result<Self> {
        match text {
            "eval" => Ok(Self::Eval),
            "grad" => Ok(Self::Grad),
            "support" => Ok(Self::Support),
            "perf" => Ok(Self::Perf),
            _ => Err(invalid("backend mode must be eval, grad, support or perf")),
        }
    }
}

pub(super) struct Prepared {
    pub(super) fields: Scope,
    pub(super) context: u32,
    pub(super) weights: Option<u32>,
    pub(super) forward: Option<u32>,
    pub(super) backward: Option<u32>,
    pub(super) output: Option<u32>,
    pub(super) backend: u32,
    pub(super) reference: Option<u32>,
    pub(super) reference_available: Option<bool>,
    pub(super) operation: String,
    pub(super) variables: String,
    pub(super) supported: bool,
    pub(super) skip: Option<String>,
    pub(super) initialize: bool,
    pub(super) crt_draws: u64,
    pub(super) expected: Vec<f32>,
}

pub(super) fn tensor(value: &Value) -> Result<u32> {
    if let Value::Handle(Handle::Tensor(id)) = value {
        Ok(*id)
    } else {
        Err(invalid("source graph builder did not return a tensor"))
    }
}
fn graph(value: &Value) -> Result<u32> {
    if let Value::Handle(Handle::Graph(id)) = value {
        Ok(*id)
    } else {
        Err(invalid(
            "production graph constructor did not return a graph",
        ))
    }
}
fn context(value: Handle) -> Result<u32> {
    if let Handle::Context(id) = value {
        Ok(id)
    } else {
        Err(invalid("production context handle differs"))
    }
}
fn backend(value: Handle) -> Result<u32> {
    if let Handle::Backend(id) = value {
        Ok(id)
    } else {
        Err(invalid("production backend handle differs"))
    }
}
fn string(value: Value) -> Result<String> {
    match value {
        Value::Bytes(bytes) => String::from_utf8(bytes).map_err(Into::into),
        Value::Pointer { value, index } => {
            let text = string(*value)?;
            Ok(text
                .get(index..)
                .ok_or_else(|| invalid("source diagnostic pointer exceeds string"))?
                .into())
        }
        _ => Err(invalid("source diagnostic result is not string")),
    }
}
fn numbers(value: &Value) -> Result<Vec<f32>> {
    (0..value.len().map_err(invalid)?)
        .map(|index| {
            Ok(value
                .index(index)
                .map_err(invalid)?
                .number()
                .map_err(invalid)?
                .as_f32())
        })
        .collect()
}

pub(super) fn invoke(
    host: &mut Production<'_, '_>,
    class: &str,
    fields: &mut Scope,
    name: &str,
    arguments: Vec<Value>,
) -> Result<Value> {
    let program = host.program;
    let mut runtime = Interpreter::new(program, class, host)?;
    runtime.fields = std::mem::take(fields);
    let result = runtime
        .call(name, arguments)
        .and_then(|value| runtime.forced(value));
    *fields = std::mem::take(&mut runtime.fields);
    result
}
fn number(
    host: &mut Production<'_, '_>,
    class: &str,
    fields: &mut Scope,
    name: &str,
    arguments: Vec<Value>,
) -> Result<Number> {
    invoke(host, class, fields, name, arguments)?
        .number()
        .map_err(invalid)
}
fn truth(
    host: &mut Production<'_, '_>,
    class: &str,
    fields: &mut Scope,
    name: &str,
) -> Result<bool> {
    invoke(host, class, fields, name, Vec::new())?
        .truth()
        .map_err(invalid)
}

fn initial_fields() -> Scope {
    Scope::from([
        ("gf".into(), variable("ggml_cgraph*".into(), Value::Null)),
        ("gb".into(), variable("ggml_cgraph*".into(), Value::Null)),
        (
            "sentinels".into(),
            variable("std::vector<ggml_tensor*>".into(), Value::array(Vec::new())),
        ),
        (
            "current_op_name".into(),
            variable("std::string".into(), Value::Bytes(Vec::new())),
        ),
        (
            "contains_f16".into(),
            variable("bool".into(), Value::Bool(false)),
        ),
    ])
}

fn set(fields: &Scope, name: &str, value: Value) -> Result<()> {
    fields
        .get(name)
        .ok_or_else(|| invalid("source base field missing"))?
        .borrow_mut()
        .value = value;
    Ok(())
}
fn constant(program: &Program, name: &str) -> Result<i64> {
    program
        .constants
        .get(name)
        .and_then(Json::as_i64)
        .ok_or_else(|| invalid(format!("source constant {name} missing")))
}
const fn handle_tensor(id: u32) -> Value {
    Value::Handle(Handle::Tensor(id))
}
const fn handle_context(id: u32) -> Value {
    Value::Handle(Handle::Context(id))
}
const fn handle_graph(id: u32) -> Value {
    Value::Handle(Handle::Graph(id))
}
const fn integer(value: i64) -> Value {
    Value::Number(Number::I64(value))
}

fn supports(
    host: &mut Production<'_, '_>,
    backend: u32,
    tensors: &[u32],
    stop: bool,
) -> Result<bool> {
    let mut supported = true;
    for &id in tensors {
        host.graph
            .push(json!({"Op":"supports","Backend":backend,"Tensor":id}));
        if stop {
            let values = host.graph.flush()?;
            if !values
                .last()
                .and_then(|value| value["Supported"].as_bool())
                .ok_or_else(|| invalid("production support observation differs"))?
            {
                return Ok(false);
            }
        }
    }
    if !stop {
        for value in host.graph.flush()? {
            if let Some(value) = value.get("Supported") {
                supported &= value
                    .as_bool()
                    .ok_or_else(|| invalid("production support observation differs"))?;
            }
        }
    }
    Ok(supported)
}

#[allow(
    clippy::too_many_lines,
    reason = "The original gradient preparation includes ordered support checks, graph construction and parameter assertions which are kept together for review."
)]
fn gradients(host: &mut Production<'_, '_>, prepared: &mut Prepared, class: &str) -> Result<()> {
    let output = prepared
        .output
        .ok_or_else(|| invalid("gradient output missing"))?;
    let metadata = host.graph.tensor(output)?;
    if i64::from(metadata.op) == constant(host.program, "GGML_OP_OPT_STEP_ADAMW")? {
        prepared.skip = Some("original optimizer gradient skip".into());
        return Ok(());
    }
    if i64::from(metadata.kind) != constant(host.program, "GGML_TYPE_F32")? {
        prepared.skip = Some("output type != FP32".into());
        prepared.supported = false;
        return Ok(());
    }
    let order = host
        .graph
        .contexts
        .get(&prepared.context)
        .cloned()
        .ok_or_else(|| invalid("gradient context traversal missing"))?;
    let param_flag = constant(host.program, "GGML_TENSOR_FLAG_PARAM")? as i32;
    let f32_kind = constant(host.program, "GGML_TYPE_F32")? as i32;
    let mut parameters = Vec::new();
    for &id in &order {
        if !supports(host, prepared.backend, &[id], true)? {
            prepared.supported = false;
            prepared.skip =
                Some("backend does not support a gradient input graph operation".into());
            return Ok(());
        }
        let tensor = host.graph.tensor(id)?;
        if tensor.flags & param_flag != 0 {
            if tensor.kind != f32_kind {
                prepared.supported = false;
                prepared.skip = Some(format!("{}->type != FP32", tensor.name));
                return Ok(());
            }
            parameters.push(tensor);
        }
    }
    if parameters.is_empty() {
        prepared.supported = false;
        prepared.skip = Some("no parameter tensors".into());
        return Ok(());
    }
    let count = parameters.iter().map(|tensor| tensor.elements).sum::<i64>();
    if count > number(host, class, &mut prepared.fields, "grad_nmax", Vec::new())?.as_i64() {
        prepared.skip = Some("original large-gradient tensor skip".into());
        return Ok(());
    }
    let output = if metadata.elements == 1 {
        output
    } else {
        let output = tensor(&host.graph.call(
            "ggml_sum",
            &[handle_context(prepared.context), handle_tensor(output)],
        )?)?;
        let _ = host.graph.call(
            "ggml_set_name",
            &[handle_tensor(output), Value::Bytes(b"sum_of_out".to_vec())],
        )?;
        output
    };
    prepared.output = Some(output);
    let forward = prepared
        .forward
        .ok_or_else(|| invalid("gradient forward graph missing"))?;
    let backward = prepared
        .backward
        .ok_or_else(|| invalid("gradient backward graph missing"))?;
    let _ = host.graph.call("ggml_set_loss", &[handle_tensor(output)])?;
    let _ = host.graph.call(
        "ggml_build_forward_expand",
        &[handle_graph(forward), handle_tensor(output)],
    )?;
    let _ = host.graph.call(
        "ggml_graph_cpy",
        &[handle_graph(forward), handle_graph(backward)],
    )?;
    let _ = host.graph.call(
        "ggml_build_backward_expand",
        &[
            handle_context(prepared.context),
            handle_graph(backward),
            Value::Null,
        ],
    )?;
    host.graph.snapshot_contexts()?;
    host.graph
        .push(json!({"Op":"graph-snapshot","Graph":forward}));
    host.graph
        .push(json!({"Op":"graph-snapshot","Graph":backward}));
    let _ = host.graph.flush()?;
    if prepared.expected.len() != 1 || prepared.expected[0] != 0.0 {
        host.assertions += 1;
        let f = host.graph.graphs[&forward]["Nodes"]
            .as_array()
            .ok_or_else(|| invalid("forward graph nodes missing"))?
            .len();
        let b = host.graph.graphs[&backward]["Nodes"]
            .as_array()
            .ok_or_else(|| invalid("backward graph nodes missing"))?
            .len();
        if b <= f {
            return Err(invalid(
                "original backward graph node-growth assertion failed",
            ));
        }
        let gradients = host.graph.graphs[&backward]["Gradients"]
            .as_array()
            .ok_or_else(|| invalid("backward graph gradients missing"))?
            .clone();
        for parameter in &parameters {
            host.assertions += 1;
            let gradient = gradients
                .iter()
                .find(|gradient| gradient["Tensor"].as_u64() == Some(u64::from(parameter.id)))
                .and_then(|gradient| gradient["Gradient"].as_u64())
                .ok_or_else(|| {
                    invalid("original parameter-gradient assertion: gradient missing")
                })?;
            if i64::from(host.graph.tensor(u32::try_from(gradient)?)?.op)
                == constant(host.program, "GGML_OP_NONE")?
            {
                return Err(invalid(
                    "original parameter-gradient non-NONE assertion failed",
                ));
            }
        }
    }
    let order = host
        .graph
        .contexts
        .get(&prepared.context)
        .cloned()
        .ok_or_else(|| invalid("post-backward context missing"))?;
    for id in order {
        let tensor = host.graph.tensor(id)?;
        if !supports(host, prepared.backend, &[id], true)?
            || (tensor.flags & param_flag != 0 && tensor.kind != f32_kind)
        {
            prepared.supported = false;
            prepared.skip = Some("post-backward operation/type not supported".into());
            return Ok(());
        }
    }
    prepared.initialize = true;
    Ok(())
}

#[allow(
    clippy::too_many_lines,
    reason = "Preparation follows the four original lifecycle branches in source order; no numerical work is performed here."
)]
pub(super) fn prepare(
    host: &mut Production<'_, '_>,
    geometry: &Geometry,
    mode: Mode,
    filter: Option<&str>,
) -> Result<Prepared> {
    let mut fields = initial_fields();
    {
        let program = host.program;
        let mut runtime = Interpreter::new(program, &geometry.class, host)?;
        runtime.fields = fields;
        let result = runtime.instantiate(geometry);
        fields = runtime.fields;
        result?;
    }
    fields
        .entry("mode".into())
        .or_insert_with(|| variable("test_mode".into(), integer(i64::from(mode.code()))));
    let threads = host.inputs.threads;
    let optimized = backend(host.graph.backend(Some(threads), false))?;
    let reference = if mode == Mode::Eval {
        Some(backend(host.graph.backend(None, true))?)
    } else {
        None
    };
    let startup = host.graph.flush()?;
    let reference_available = if reference.is_some() {
        startup
            .last()
            .and_then(|value| value["UseRefAvailable"].as_bool())
    } else {
        None
    };
    let mut prepared = Prepared {
        fields,
        context: 0,
        weights: None,
        forward: None,
        backward: None,
        output: None,
        backend: optimized,
        reference,
        reference_available,
        operation: String::new(),
        variables: String::new(),
        supported: true,
        skip: None,
        initialize: false,
        crt_draws: 0,
        expected: Vec::new(),
    };
    if matches!(mode, Mode::Grad | Mode::Support)
        && truth(
            host,
            &geometry.class,
            &mut prepared.fields,
            "run_whole_graph",
        )?
    {
        prepared.skip = Some("original whole-graph mode skip".into());
        return Ok(prepared);
    }
    if mode == Mode::Grad {
        prepared.expected = numbers(&invoke(
            host,
            &geometry.class,
            &mut prepared.fields,
            "grad_expect",
            Vec::new(),
        )?)?;
    }
    let capacity = number(
        host,
        &geometry.class,
        &mut prepared.fields,
        "ctx_tensors",
        Vec::new(),
    )?
    .as_u64();
    let tensor_overhead = host.graph.call("ggml_tensor_overhead", &[])?;
    let tensor_overhead = host
        .force(tensor_overhead)?
        .number()
        .map_err(invalid)?
        .as_u64();
    let nodes = if matches!(mode, Mode::Perf | Mode::Support) {
        8192
    } else {
        constant(host.program, "GGML_DEFAULT_GRAPH_SIZE")?
    };
    let graph_overhead = if mode == Mode::Eval {
        host.graph.call("ggml_graph_overhead", &[])?
    } else {
        host.graph.call(
            "ggml_graph_overhead_custom",
            &[integer(nodes), Value::Bool(mode == Mode::Grad)],
        )?
    };
    let graph_overhead = host
        .force(graph_overhead)?
        .number()
        .map_err(invalid)?
        .as_u64();
    let memory = tensor_overhead
        .checked_mul(capacity)
        .and_then(|bytes| {
            bytes.checked_add(graph_overhead * if mode == Mode::Grad { 2 } else { 1 })
        })
        .ok_or_else(|| invalid("source context memory extent overflow"))?;
    prepared.context = context(host.graph.context(memory))?;
    if matches!(mode, Mode::Eval | Mode::Perf)
        && truth(
            host,
            &geometry.class,
            &mut prepared.fields,
            "use_weight_context",
        )?
    {
        prepared.weights = Some(context(host.graph.context(memory))?);
    }
    if mode != Mode::Perf {
        let value = if mode == Mode::Eval {
            host.graph
                .call("ggml_new_graph", &[handle_context(prepared.context)])?
        } else {
            host.graph.call(
                "ggml_new_graph_custom",
                &[
                    handle_context(prepared.context),
                    integer(nodes),
                    Value::Bool(mode == Mode::Grad),
                ],
            )?
        };
        let forward = graph(&value)?;
        prepared.forward = Some(forward);
        set(&prepared.fields, "gf", handle_graph(forward))?;
    }
    if mode == Mode::Grad {
        let backward = graph(&host.graph.call(
            "ggml_new_graph_custom",
            &[
                handle_context(prepared.context),
                integer(nodes),
                Value::Bool(true),
            ],
        )?)?;
        prepared.backward = Some(backward);
        set(&prepared.fields, "gb", handle_graph(backward))?;
    }
    if mode == Mode::Eval {
        host.add_sentinel(prepared.context)?;
        if let Some(weights) = prepared.weights {
            host.add_sentinel(weights)?;
        }
    }
    let arguments = if matches!(mode, Mode::Eval | Mode::Perf) {
        vec![
            handle_context(prepared.context),
            prepared.weights.map_or(Value::Null, handle_context),
        ]
    } else {
        vec![handle_context(prepared.context)]
    };
    let output = tensor(&invoke(
        host,
        &geometry.class,
        &mut prepared.fields,
        "build_graph",
        arguments,
    )?)?;
    prepared.output = Some(output);
    host.graph.snapshot_contexts()?;
    prepared.operation = string(invoke(
        host,
        &geometry.class,
        &mut prepared.fields,
        "op_desc",
        vec![handle_tensor(output)],
    )?)?;
    prepared.variables = string(invoke(
        host,
        &geometry.class,
        &mut prepared.fields,
        "vars",
        Vec::new(),
    )?)?;
    set(
        &prepared.fields,
        "current_op_name",
        Value::Bytes(prepared.operation.as_bytes().to_vec()),
    )?;
    let f16 = constant(host.program, "GGML_TYPE_F16")? as i32;
    let contains_f16 = host.graph.contexts[&prepared.context].iter().any(|id| {
        host.graph
            .tensors
            .get(id)
            .is_some_and(|tensor| tensor.kind == f16)
    });
    set(&prepared.fields, "contains_f16", Value::Bool(contains_f16))?;
    if filter
        .is_some_and(|filter| !matches_filter(&prepared.operation, &prepared.variables, filter))
    {
        prepared.skip = Some("original operation filter skip".into());
        return Ok(prepared);
    }
    match mode {
        Mode::Eval => {
            let order = host.graph.contexts[&prepared.context].clone();
            prepared.supported = supports(host, optimized, &order, false)?;
            if let Some(reference) = reference {
                prepared.supported &= supports(host, reference, &order, false)?;
            }
            if prepared.supported {
                host.add_sentinel(prepared.context)?;
                if let Some(weights) = prepared.weights {
                    host.add_sentinel(weights)?;
                }
                host.graph.snapshot_contexts()?;
                prepared.initialize = true;
            } else {
                prepared.skip = Some("operation not supported by a compared CPU backend".into());
            }
        }
        Mode::Grad => gradients(host, &mut prepared, &geometry.class)?,
        Mode::Perf | Mode::Support => {
            prepared.supported = supports(host, optimized, &[output], true)?;
            if !prepared.supported {
                prepared.skip = Some("output operation not supported by CPU backend".into());
            }
            prepared.initialize = prepared.supported && mode == Mode::Perf;
        }
    }
    prepared.crt_draws = if prepared.initialize {
        crt_draws(
            host,
            &geometry.class,
            &prepared.fields,
            prepared.context,
            prepared.weights,
        )?
    } else {
        0
    };
    Ok(prepared)
}

fn field_number(fields: &Scope, name: &str) -> Result<Number> {
    fields
        .get(name)
        .ok_or_else(|| invalid("CRT source field missing"))?
        .borrow()
        .value
        .number()
        .map_err(invalid)
}
fn field_index(fields: &Scope, name: &str, index: usize) -> Result<Number> {
    fields
        .get(name)
        .ok_or_else(|| invalid("CRT source array field missing"))?
        .borrow()
        .value
        .index(index)
        .map_err(invalid)?
        .number()
        .map_err(invalid)
}
fn crt_draws(
    host: &mut Production<'_, '_>,
    class: &str,
    fields: &Scope,
    context: u32,
    weights: Option<u32>,
) -> Result<u64> {
    let mut draws = 0_u64;
    let i32_kind = constant(host.program, "GGML_TYPE_I32")? as i32;
    let view_ops = [
        "GGML_OP_VIEW",
        "GGML_OP_RESHAPE",
        "GGML_OP_PERMUTE",
        "GGML_OP_TRANSPOSE",
    ]
    .map(|name| constant(host.program, name))
    .into_iter()
    .collect::<Result<Vec<_>>>()?;
    for context in [Some(context), weights].into_iter().flatten() {
        for id in host
            .graph
            .contexts
            .get(&context)
            .cloned()
            .ok_or_else(|| invalid("CRT context traversal missing"))?
        {
            let tensor = host.graph.tensor(id)?;
            if tensor.kind != i32_kind {
                continue;
            }
            let count = match class {
                "test_get_rows" if !view_ops.contains(&i64::from(tensor.op)) => {
                    field_number(fields, "r")?
                        .binary("*", field_number(fields, "be1")?)
                        .map_err(invalid)?
                        .binary("*", field_number(fields, "be2")?)
                        .map_err(invalid)?
                        .as_u64()
                }
                "test_get_rows_back" if !view_ops.contains(&i64::from(tensor.op)) => {
                    field_number(fields, "r")?
                        .binary("*", field_number(fields, "b")?)
                        .map_err(invalid)?
                        .as_u64()
                }
                "test_rope" | "test_rope_set_rows" if tensor.name != "row_idxs" => {
                    let mode = field_number(fields, "mode")?.as_i64();
                    let rows = field_index(fields, "ne_a", 2)?.as_i64();
                    (if mode & constant(host.program, "GGML_ROPE_TYPE_MROPE")? != 0 {
                        rows * 4
                    } else {
                        rows
                    }) as u64
                }
                "test_argsort" => tensor.elements as u64,
                _ => 0,
            };
            draws = draws
                .checked_add(count)
                .ok_or_else(|| invalid("CRT draw count overflow"))?;
        }
    }
    Ok(draws)
}

pub(super) fn matches_filter(operation: &str, variables: &str, mut filter: &str) -> bool {
    let full = format!("{operation}({variables})");
    while !filter.is_empty() {
        let comma = filter.find(',').unwrap_or(filter.len());
        let paren = filter.find('(').unwrap_or(filter.len());
        let end = if paren < comma {
            let right = filter.find(')').unwrap_or(filter.len());
            filter
                .get(right..)
                .and_then(|text| text.find(',').map(|position| right + position))
                .unwrap_or(filter.len())
        } else {
            comma
        };
        let selected = &filter[..end];
        if selected
            == if paren < comma {
                full.as_str()
            } else {
                operation
            }
        {
            return true;
        }
        filter = filter
            .get(end + usize::from(end < filter.len())..)
            .unwrap_or("");
    }
    false
}
