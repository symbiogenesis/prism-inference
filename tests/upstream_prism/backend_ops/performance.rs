// Copyright (c) 2023-2026 The ggml/llama.cpp authors
// SPDX-License-Identifier: MIT
//! CI-only original performance workload. Timing has no correctness threshold.

use super::super::{Result, invalid};
use super::evaluate::{self, backend_value, context_value, graph_value, integer, tensor_value};
use super::host::Production;
use super::interpreter::Host;
use super::prepare::{Prepared, invoke};
use super::program::Geometry;
use super::value::{Handle, Value};
use serde_json::{Value as Json, json};

pub(super) fn require_ci() -> Result<()> {
    if std::env::var("GITHUB_ACTIONS").as_deref() != Ok("true")
        || std::env::var("GITHUB_REF").as_deref() != Ok("refs/heads/main")
    {
        return Err(invalid(
            "original backend performance workloads run only on main CI",
        ));
    }
    Ok(())
}

fn numeric(host: &mut Production<'_, '_>, value: Value) -> Result<u64> {
    host.force(value)?
        .number()
        .map_err(invalid)
        .map(super::number::Number::as_u64)
}
fn property(
    host: &mut Production<'_, '_>,
    prepared: &mut Prepared,
    geometry: &Geometry,
    name: &str,
    output: u32,
) -> Result<u64> {
    invoke(
        host,
        &geometry.class,
        &mut prepared.fields,
        name,
        vec![tensor_value(output)],
    )?
    .number()
    .map_err(invalid)
    .map(super::number::Number::as_u64)
}

fn interval(host: &mut Production<'_, '_>, backend: u32, graph: u32) -> Result<i64> {
    let _ = host.graph.flush()?;
    host.graph.push(json!({"Op":"clock"}));
    let status = host.graph.call(
        "ggml_backend_graph_compute",
        &[backend_value(backend), graph_value(graph)],
    )?;
    host.graph.push(json!({"Op":"clock"}));
    let Value::NativeInteger { argument, .. } = &status else {
        return Err(invalid("production compute status did not return a scalar"));
    };
    let scalar = argument["Scalar"]
        .as_u64()
        .ok_or_else(|| invalid("production compute scalar handle missing"))?;
    host.graph
        .push(json!({"Op":"scalar","Scalar":scalar,"Release":true}));
    let observed = host.graph.flush()?;
    if observed.len() != 4 {
        return Err(invalid(
            "native performance interval observation extent differs",
        ));
    }
    let start = observed[0]["Microseconds"]
        .as_i64()
        .ok_or_else(|| invalid("native performance start clock missing"))?;
    let end = observed[2]["Microseconds"]
        .as_i64()
        .ok_or_else(|| invalid("native performance end clock missing"))?;
    let code = host.force(status)?.number().map_err(invalid)?.as_i64();
    if Some(code) != host.program.constants["GGML_STATUS_SUCCESS"].as_i64() {
        return Err(invalid(format!(
            "original timed compute-success check failed: {code}"
        )));
    }
    if end < start {
        return Err(invalid("production monotonic clock moved backwards"));
    }
    Ok(end - start)
}

#[allow(
    clippy::too_many_lines,
    reason = "The original warmup, graph duplication, memory accounting and minimum-duration loop are preserved in source order."
)]
pub(super) fn evaluate(
    host: &mut Production<'_, '_>,
    prepared: &mut Prepared,
    geometry: &Geometry,
) -> Result<Json> {
    require_ci()?;
    let output = prepared
        .output
        .ok_or_else(|| invalid("performance output missing"))?;
    let value = host.graph.call(
        "ggml_new_graph_custom",
        &[
            context_value(prepared.context),
            integer(8192),
            Value::Bool(false),
        ],
    )?;
    let Value::Handle(Handle::Graph(forward)) = value else {
        return Err(invalid(
            "performance graph constructor did not return a graph",
        ));
    };
    let _ = host.graph.call(
        "ggml_build_forward_expand",
        &[graph_value(forward), tensor_value(output)],
    )?;
    // The original warmup is excluded from every reported timing interval.
    evaluate::compute(host, prepared.backend, forward)?;
    let size = host
        .graph
        .call("ggml_graph_size", &[graph_value(forward)])?;
    let size = numeric(host, size)?;
    let nodes = host
        .graph
        .call("ggml_graph_n_nodes", &[graph_value(forward)])?;
    let nodes = numeric(host, nodes)?;
    let available = size
        .checked_sub(nodes)
        .ok_or_else(|| invalid("performance graph capacity smaller than its nodes"))?;
    let flops = property(host, prepared, geometry, "op_flops", output)?;
    let operation_size = property(host, prepared, geometry, "op_size", output)?;
    let limit = if let Some(limit) = 8_000_000_000_u64.checked_div(flops) {
        limit
    } else {
        (8_u64 << 30)
            .checked_div(operation_size)
            .ok_or_else(|| invalid("original performance op-size division is zero"))?
    };
    let runs = i32::try_from(available.min(limit) + 1)?;
    for _ in 1..runs {
        let _ = host.graph.call(
            "ggml_graph_add_node",
            &[graph_value(forward), tensor_value(output)],
        )?;
    }
    host.graph.snapshot_contexts()?;
    host.graph
        .push(json!({"Op":"graph-snapshot","Graph":forward}));
    let _ = host.graph.flush()?;
    let nodes = host.graph.graphs[&forward]["Nodes"]
        .as_array()
        .ok_or_else(|| invalid("performance graph nodes missing"))?
        .clone();
    let output_identity = host.graph.tensor(output)?.id;
    let views = [
        "GGML_OP_VIEW",
        "GGML_OP_RESHAPE",
        "GGML_OP_PERMUTE",
        "GGML_OP_TRANSPOSE",
    ]
    .map(|name| {
        host.program.constants[name]
            .as_i64()
            .ok_or_else(|| invalid("source view-op constant missing"))
    })
    .into_iter()
    .collect::<Result<Vec<_>>>()?;
    let mut memory = (runs as u64).wrapping_mul(operation_size);
    for node in nodes {
        let id = u32::try_from(
            node.as_u64()
                .ok_or_else(|| invalid("performance graph tensor ID missing"))?,
        )?;
        let metadata = host.graph.tensor(id)?;
        if views.contains(&i64::from(metadata.op)) || metadata.id == output_identity {
            continue;
        }
        let mut bytes = metadata.bytes;
        for source in metadata.sources.into_iter().flatten() {
            bytes = bytes.wrapping_add(host.graph.tensor(source)?.bytes);
        }
        memory = memory.wrapping_add(bytes);
    }
    let mut total_time = 0_i64;
    let mut total_memory = 0_i64;
    let mut total_runs = 0_i32;
    let mut iterations = 0_u64;
    loop {
        total_time = total_time
            .checked_add(interval(host, prepared.backend, forward)?)
            .ok_or_else(|| invalid("performance elapsed clock accumulation overflow"))?;
        total_memory = total_memory.wrapping_add(memory as i64);
        total_runs = total_runs.wrapping_add(runs);
        iterations += 1;
        if total_time >= 1_000_000 {
            break;
        }
    }
    let average = total_time as f64 / f64::from(total_runs);
    let seconds = total_time as f64 / 1e6;
    let calculated_flops = if flops > 0 {
        flops.wrapping_mul(total_runs as u64) as f64 / seconds
    } else {
        0.0
    };
    let bandwidth = if flops == 0 {
        total_memory as f64 / seconds / 1024.0 / 1024.0 / 1024.0
    } else {
        0.0
    };
    Ok(
        json!({"Status":"measured","SourceMinimumMicroseconds":1_000_000,"MeasuredMicroseconds":total_time,"TotalRuns":total_runs,"RunsPerGraph":runs,"Iterations":iterations,"OperationBytes":operation_size,"MemoryKilobytes":operation_size/1024,"TotalMemoryBytes":total_memory,"AverageMicrosecondsF64Bits":average.to_bits(),"FlopsPerSecondF64Bits":calculated_flops.to_bits(),"GiBPerSecondF64Bits":bandwidth.to_bits(),"NativeActionDispatchIncluded":true,"ExcludedFromTiming":["request parsing","graph construction","input generation","quantization","allocation","warmup","Rust continuation waits"],"CorrectnessThreshold":null}),
    )
}
