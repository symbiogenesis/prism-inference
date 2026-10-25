// Copyright (c) 2023-2026 The ggml/llama.cpp authors
// SPDX-License-Identifier: MIT
//! Source allocation, initialization and optimized/reference comparison contracts.

use super::super::{Result, invalid};
use super::host::Production;
use super::inputs::Inputs;
use super::interpreter::Host;
use super::metrics;
use super::number::Number;
use super::prepare::{self, Mode, Prepared, invoke};
use super::program::Geometry;
use super::tensor::{self, Tensor};
use super::value::{Handle, Scope, Value};
use serde_json::{Value as Json, json};
use std::fs;
use std::io::Read as _;
use std::path::{Path, PathBuf};

pub(super) const fn tensor_value(id: u32) -> Value {
    Value::Handle(Handle::Tensor(id))
}
pub(super) const fn graph_value(id: u32) -> Value {
    Value::Handle(Handle::Graph(id))
}
pub(super) const fn context_value(id: u32) -> Value {
    Value::Handle(Handle::Context(id))
}
pub(super) const fn backend_value(id: u32) -> Value {
    Value::Handle(Handle::Backend(id))
}
pub(super) const fn integer(value: u64) -> Value {
    Value::Number(Number::U64(value))
}
pub(super) fn field(fields: &Scope, name: &str) -> Result<Value> {
    Ok(fields
        .get(name)
        .ok_or_else(|| invalid(format!("source field {name} missing")))?
        .borrow()
        .value
        .clone())
}
fn truth(fields: &Scope, name: &str) -> Result<bool> {
    field(fields, name)?.truth().map_err(invalid)
}

pub(super) fn allocate(
    host: &mut Production<'_, '_>,
    prepared: &mut Prepared,
    geometry: &Geometry,
    mode: Mode,
    offset: u64,
) -> Result<()> {
    if let Some(weights) = prepared.weights {
        let buffer = host.graph.allocate(weights, prepared.backend);
        let usage = host.program.constants["GGML_BACKEND_BUFFER_USAGE_WEIGHTS"]
            .as_i64()
            .ok_or_else(|| invalid("weight usage constant missing"))?;
        let _ = host.graph.call(
            "ggml_backend_buffer_set_usage",
            &[
                Value::Handle(buffer),
                Value::Number(Number::I32(usage as i32)),
            ],
        )?;
    }
    let _ = host.graph.allocate(prepared.context, prepared.backend);
    let _ = host.graph.flush()?;
    if mode == Mode::Eval {
        let forward = prepared
            .forward
            .ok_or_else(|| invalid("comparison forward graph missing"))?;
        let output = prepared
            .output
            .ok_or_else(|| invalid("comparison output missing"))?;
        let _ = host.graph.call(
            "ggml_build_forward_expand",
            &[graph_value(forward), tensor_value(output)],
        )?;
        for id in host.sentinels.clone() {
            let _ = host.graph.call(
                "ggml_graph_add_node",
                &[graph_value(forward), tensor_value(id)],
            )?;
        }
        let _ = host.graph.flush()?;
    }
    host.crt.skip(offset);
    let start = host.crt.draws;
    let _ = invoke(
        host,
        &geometry.class,
        &mut prepared.fields,
        "initialize_tensors",
        vec![context_value(prepared.context)],
    )?;
    if let Some(weights) = prepared.weights {
        let _ = invoke(
            host,
            &geometry.class,
            &mut prepared.fields,
            "initialize_tensors",
            vec![context_value(weights)],
        )?;
    }
    let _ = host.graph.flush()?;
    if host.crt.draws - start != prepared.crt_draws {
        return Err(invalid(
            "actual source initializer CRT draw count differs from complete ordered plan",
        ));
    }
    Ok(())
}

pub(super) fn compute(host: &mut Production<'_, '_>, backend: u32, graph: u32) -> Result<()> {
    let status = host.graph.call(
        "ggml_backend_graph_compute",
        &[backend_value(backend), graph_value(graph)],
    )?;
    let status = host.force(status)?.number().map_err(invalid)?.as_i64();
    let success = host.program.constants["GGML_STATUS_SUCCESS"]
        .as_i64()
        .ok_or_else(|| invalid("source compute-success constant missing"))?;
    if status != success {
        return Err(invalid(format!(
            "original ggml_backend_graph_compute success check failed: status {status}"
        )));
    }
    Ok(())
}

pub(super) fn capture(
    host: &mut Production<'_, '_>,
    id: u32,
    purpose: &str,
) -> Result<(Tensor, PathBuf)> {
    let tensor = host.graph.tensor(id)?;
    let path = host.inputs.name(&host.graph, purpose)?;
    host.graph.push(json!({"Op":"tensor-get","Tensor":id,"File":path,"FileOffset":0,"TensorOffset":0,"Bytes":tensor.bytes}));
    let _ = host.graph.flush()?;
    let path = host.graph.session.checked_file(&path)?;
    host.inputs.record(&path, purpose)?;
    Ok((tensor, path))
}

pub(super) fn floats(host: &mut Production<'_, '_>, id: u32, purpose: &str) -> Result<Vec<f32>> {
    let (tensor, path) = capture(host, id, purpose)?;
    tensor::tensor_to_float(host.graph.session, host.program, &tensor, &path, purpose)
}

fn metric(
    host: &mut Production<'_, '_>,
    prepared: &Prepared,
    geometry: &Geometry,
    a: &[f32],
    b: &[f32],
    ordinal: usize,
) -> Result<f64> {
    match geometry.class.as_str() {
        "test_get_rows" if truth(&prepared.fields, "exact")? => Ok(metrics::get_rows_exact(a, b)),
        "test_ssm_scan_rollback" => Ok(metrics::maximum_absolute(a, b)),
        "test_gdn_replay_chain" if truth(&prepared.fields, "exact")? => {
            Ok(metrics::maximum_absolute(a, b))
        }
        "test_topk_moe" => Ok(metrics::sorted_nmse(a, b)),
        "test_top_k" => {
            if truth(&prepared.fields, "ties")? {
                host.assertions += 1;
                let input = prepare::tensor(&field(&prepared.fields, "input")?)?;
                let metadata = host.graph.tensor(input)?;
                let (_, source_path) = capture(host, input, &format!("top-k-source-{ordinal}"))?;
                let mut source_bytes = vec![0_u8; usize::try_from(metadata.elements)? * 4];
                let raw = fs::read(source_path)?;
                let copied = usize::try_from(metadata.elements)?
                    .checked_mul(usize::try_from(metadata.type_size)?)
                    .ok_or_else(|| invalid("source top-k raw-copy extent overflow"))?;
                source_bytes
                    .get_mut(..copied)
                    .ok_or_else(|| invalid("source top-k raw-copy exceeds float storage"))?
                    .copy_from_slice(
                        raw.get(..copied)
                            .ok_or_else(|| invalid("source top-k raw-copy exceeds tensor"))?,
                    );
                let source = source_bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|bytes| f32::from_le_bytes(*bytes))
                    .collect::<Vec<_>>();
                let k = field(&prepared.fields, "k")?
                    .number()
                    .map_err(invalid)?
                    .index()
                    .map_err(invalid)?;
                metrics::top_k(a, b, Some((&source, usize::try_from(metadata.ne[0])?, k)))
                    .map_err(invalid)
            } else {
                metrics::top_k(a, b, None).map_err(invalid)
            }
        }
        _ => Ok(metrics::nmse(a, b)),
    }
}

fn raw_equal(host: &Production<'_, '_>, a: &Path, b: &Path, bytes: u64) -> Result<bool> {
    let a = host.graph.session.checked_file(a)?;
    let b = host.graph.session.checked_file(b)?;
    if fs::metadata(&a)?.len() != bytes || fs::metadata(&b)?.len() != bytes {
        return Err(invalid("compared sentinel raw byte extent differs"));
    }
    let mut a = fs::File::open(a)?;
    let mut b = fs::File::open(b)?;
    let mut av = vec![0_u8; 64 * 1024];
    let mut bv = vec![0_u8; 64 * 1024];
    let mut remaining = bytes;
    while remaining != 0 {
        let count = usize::try_from(remaining.min(av.len() as u64))?;
        a.read_exact(&mut av[..count])?;
        b.read_exact(&mut bv[..count])?;
        if av[..count] != bv[..count] {
            return Ok(false);
        }
        remaining -= count as u64;
    }
    Ok(true)
}

#[allow(
    clippy::too_many_lines,
    reason = "Every original callback branch and early-return condition is kept in source order in one comparison loop."
)]
pub(super) fn evaluate(
    host: &mut Production<'_, '_>,
    prepared: &mut Prepared,
    geometry: &Geometry,
) -> Result<Json> {
    let forward = prepared
        .forward
        .ok_or_else(|| invalid("comparison forward graph missing"))?;
    let output = prepared
        .output
        .ok_or_else(|| invalid("comparison output missing"))?;
    let reference = prepared
        .reference
        .ok_or_else(|| invalid("comparison reference backend missing"))?;
    let nodes = invoke(
        host,
        &geometry.class,
        &mut prepared.fields,
        "fusion_test_nodes",
        Vec::new(),
    )?;
    let mut selected = (0..nodes.len().map_err(invalid)?)
        .map(|index| prepare::tensor(&nodes.index(index).map_err(invalid)?))
        .collect::<Result<Vec<_>>>()?;
    let whole = invoke(
        host,
        &geometry.class,
        &mut prepared.fields,
        "run_whole_graph",
        Vec::new(),
    )?
    .truth()
    .map_err(invalid)?;
    if selected.is_empty() && whole {
        selected.push(output);
    }
    let prefix = host.graph.session.file("compare")?;
    host.graph.push(json!({"Op":"compare","Backend1":prepared.backend,"Backend2":reference,"Graph":forward,"Nodes":if whole{json!(selected)}else{Json::Null},"Prefix":prefix,"CaptureTensors":[],"MaxOutputBytes":8_u64*1024*1024*1024}));
    let observed = host.graph.flush()?;
    let observed = observed
        .last()
        .ok_or_else(|| invalid("production comparison observation missing"))?;
    let compared = observed["Compared"]
        .as_bool()
        .ok_or_else(|| invalid("production comparison status missing"))?;
    let pairs = observed["Pairs"]
        .as_array()
        .ok_or_else(|| invalid("production comparison pairs missing"))?
        .clone();
    let mut summaries = Vec::with_capacity(pairs.len());
    let mut passed = compared;
    let mut element_checks = 0_u64;
    for (ordinal, pair) in pairs.iter().enumerate() {
        let left: Tensor = serde_json::from_value(pair["Left"].clone())?;
        let right: Tensor = serde_json::from_value(pair["Right"].clone())?;
        let left_path = Path::new(
            pair["LeftFile"]
                .as_str()
                .ok_or_else(|| invalid("comparison left file missing"))?,
        );
        let right_path = Path::new(
            pair["RightFile"]
                .as_str()
                .ok_or_else(|| invalid("comparison right file missing"))?,
        );
        let left_path = host.graph.session.checked_file(left_path)?;
        let right_path = host.graph.session.checked_file(right_path)?;
        host.inputs
            .record(&left_path, "optimized comparison observation")?;
        host.inputs
            .record(&right_path, "reference comparison observation")?;
        let mut summary = json!({"Ordinal":ordinal,"Index":pair["Index"],"Name":left.name,"Op":left.op_description,"Elements":left.elements,"LeftBytes":left.bytes,"RightBytes":right.bytes});
        if left.bytes != right.bytes || left.elements != right.elements {
            passed = false;
            summary["Status"] = json!("fail");
            summary["Reason"] = json!("production comparison tensor extents differ");
            summaries.push(summary);
            continue;
        }
        if i64::from(left.op)
            == host.program.constants["GGML_OP_NONE"]
                .as_i64()
                .ok_or_else(|| invalid("source NONE constant missing"))?
            && !raw_equal(host, &left_path, &right_path, left.bytes)?
        {
            passed = false;
            summary["Status"] = json!("fail");
            summary["Reason"] = json!("original sentinel raw memcmp failed");
            summaries.push(summary);
            continue;
        }
        let a = tensor::tensor_to_float(
            host.graph.session,
            host.program,
            &left,
            &left_path,
            &format!("pair-{ordinal}-left"),
        )?;
        let b = tensor::tensor_to_float(
            host.graph.session,
            host.program,
            &right,
            &right_path,
            &format!("pair-{ordinal}-right"),
        )?;
        let failure = metrics::element_failure(&a, &b);
        element_checks += failure.map_or(a.len() as u64, |(index, _)| index as u64 + 1);
        if let Some((index, reason)) = failure {
            passed = false;
            summary["Status"] = json!("fail");
            summary["Reason"] = json!(reason);
            summary["Element"] = json!(index);
            summaries.push(summary);
            continue;
        }
        let error = metric(host, prepared, geometry, &a, &b, ordinal)?;
        let threshold = invoke(
            host,
            &geometry.class,
            &mut prepared.fields,
            "max_err",
            vec![backend_value(prepared.backend)],
        )?
        .number()
        .map_err(invalid)?
        .as_f64();
        let success = !matches!(
            error.partial_cmp(&threshold),
            Some(std::cmp::Ordering::Greater)
        );
        passed &= success;
        summary["Status"] = json!(if success { "pass" } else { "fail" });
        summary["ErrorF64Bits"] = json!(error.to_bits());
        summary["ThresholdF64Bits"] = json!(threshold.to_bits());
        summaries.push(summary);
    }
    Ok(
        json!({"Status":if passed{"pass"}else{"fail"},"Compared":compared,"ComparisonPairs":summaries,"ElementChecks":element_checks,"SourceAssertions":host.assertions}),
    )
}

pub(super) fn restore(
    host: &mut Production<'_, '_>,
    tensor: u32,
    path: &Path,
    bytes: u64,
) -> Result<()> {
    Inputs::set_file(&mut host.graph, tensor, path, 0, bytes)
}
