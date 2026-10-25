// Copyright (c) 2023-2026 The ggml/llama.cpp authors
// SPDX-License-Identifier: MIT
//! Source numerical/algebraic gradient comparisons; every coordinate and stop decision is Rust-owned.

use super::super::{Result, invalid};
use super::evaluate::{self, graph_value};
use super::host::Production;
use super::metrics;
use super::prepare::{Prepared, invoke};
use super::program::Geometry;
use serde_json::{Value as Json, json};
use std::fs;

fn scalar_read(host: &mut Production<'_, '_>, output: u32, path: &std::path::Path, index: usize) {
    host.graph.push(json!({"Op":"tensor-get","Tensor":output,"File":path,"FileOffset":index*4,"TensorOffset":0,"Bytes":4,"Truncate":index==0}));
}

fn perturb(
    host: &mut Production<'_, '_>,
    parameter: u32,
    input: &std::path::Path,
    input_index: usize,
    coordinate: usize,
) {
    host.graph.push(json!({"Op":"tensor-set","Tensor":parameter,"File":input,"FileOffset":input_index*4,"TensorOffset":coordinate*4,"Bytes":4}));
}

#[allow(
    clippy::too_many_lines,
    reason = "The ordered per-parameter and per-coordinate loop preserves the source's early returns, restores and two/four-point arithmetic."
)]
pub(super) fn evaluate(
    host: &mut Production<'_, '_>,
    prepared: &mut Prepared,
    geometry: &Geometry,
) -> Result<Json> {
    let forward = prepared
        .forward
        .ok_or_else(|| invalid("gradient forward graph missing"))?;
    let backward = prepared
        .backward
        .ok_or_else(|| invalid("gradient backward graph missing"))?;
    let output = prepared
        .output
        .ok_or_else(|| invalid("gradient output tensor missing"))?;
    let _ = host
        .graph
        .call("ggml_graph_reset", &[graph_value(backward)])?;
    evaluate::compute(host, prepared.backend, forward)?;
    evaluate::compute(host, prepared.backend, backward)?;
    host.graph.snapshot_contexts()?;
    host.graph
        .push(json!({"Op":"graph-snapshot","Graph":backward}));
    let _ = host.graph.flush()?;
    let gradients = host.graph.graphs[&backward]["Gradients"]
        .as_array()
        .ok_or_else(|| invalid("gradient map observation missing"))?
        .clone();
    let order = host
        .graph
        .contexts
        .get(&prepared.context)
        .cloned()
        .ok_or_else(|| invalid("gradient context order missing"))?;
    let parameter_flag = host.program.constants["GGML_TENSOR_FLAG_PARAM"]
        .as_i64()
        .ok_or_else(|| invalid("source parameter flag missing"))? as i32;
    let mut parameters = Vec::new();
    let mut passed = true;
    let mut finite_checks = 0_u64;
    let mut coordinates = 0_u64;
    for id in order {
        let metadata = host.graph.tensor(id)?;
        if metadata.flags & parameter_flag == 0 {
            continue;
        }
        let count = usize::try_from(metadata.elements)?;
        let gradient = gradients
            .iter()
            .find(|gradient| gradient["Tensor"].as_u64() == Some(u64::from(metadata.id)))
            .and_then(|gradient| gradient["Gradient"].as_u64());
        let algebraic = if let Some(gradient) = gradient {
            evaluate::floats(host, u32::try_from(gradient)?, &format!("algebraic-{id}"))?
        } else {
            vec![0.0_f32; count]
        };
        let mut failure = None;
        for (index, value) in algebraic.iter().enumerate() {
            finite_checks += 1;
            if !value.is_finite() {
                failure = Some(index);
                break;
            }
        }
        if let Some(index) = failure {
            passed = false;
            parameters.push(json!({"Tensor":id,"Name":metadata.name,"Status":"fail","Reason":"original algebraic gradient finite check failed","Coordinate":index}));
            break;
        }
        host.assertions += 1;
        if algebraic.len() != count {
            return Err(invalid("original ga.size == gn.size assertion failed"));
        }
        // This capture deliberately follows BOTH baseline computations, preserving in-place effects.
        let original = evaluate::floats(host, id, &format!("original-parameter-{id}"))?;
        let original_path =
            host.inputs
                .stage_floats(&host.graph, &original, "gradient-original")?;
        if original.len() != count || metadata.bytes != count as u64 * 4 {
            return Err(invalid(
                "source parameter float reset buffer does not cover its raw extent",
            ));
        }
        host.assertions += 2;
        let out = host.graph.tensor(output)?;
        let f32_kind = host.program.constants["GGML_TYPE_F32"]
            .as_i64()
            .ok_or_else(|| invalid("source F32 kind missing"))?;
        if out.elements != 1 || i64::from(out.kind) != f32_kind {
            return Err(invalid(
                "original gradient output scalar/F32 assertions failed",
            ));
        }
        let epsilon = invoke(
            host,
            &geometry.class,
            &mut prepared.fields,
            "grad_eps",
            Vec::new(),
        )?
        .number()
        .map_err(invalid)?
        .as_f32();
        let precise = invoke(
            host,
            &geometry.class,
            &mut prepared.fields,
            "grad_precise",
            Vec::new(),
        )?
        .truth()
        .map_err(invalid)?;
        let width = if precise { 4 } else { 2 };
        let mut perturbations = Vec::with_capacity(count * width);
        for &value in &original {
            perturbations.push(value + 1.0_f32 * epsilon);
            perturbations.push(value - 1.0_f32 * epsilon);
            if precise {
                perturbations.push(value + 0.5_f32 * epsilon);
                perturbations.push(value - 0.5_f32 * epsilon);
            }
        }
        let perturbation_path =
            host.inputs
                .stage_floats(&host.graph, &perturbations, "gradient-perturbations")?;
        let output_path = host.inputs.name(&host.graph, "gradient-observations")?;
        for coordinate in 0..count {
            for position in 0..width {
                let index = coordinate * width + position;
                perturb(host, id, &perturbation_path, index, coordinate);
                evaluate::compute(host, prepared.backend, forward)?;
                scalar_read(host, output, &output_path, index);
            }
            evaluate::restore(host, id, &original_path, metadata.bytes)?;
            coordinates += 1;
        }
        let _ = host.graph.flush()?;
        let output_path = host.graph.session.checked_file(&output_path)?;
        host.inputs
            .record(&output_path, "numerical gradient raw outputs")?;
        let raw = fs::read(output_path)?;
        if raw.len() != count * width * 4 {
            return Err(invalid("numerical gradient raw output extent differs"));
        }
        let outputs = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect::<Vec<_>>();
        let mut numerical = Vec::with_capacity(count);
        for values in outputs.chunks_exact(width) {
            numerical.push(if precise {
                metrics::four_point(values[0], values[2], values[3], values[1], epsilon)
            } else {
                metrics::two_point(values[0], values[1], epsilon)
            });
        }
        let error = metrics::mean_abs_asymm(&numerical, &algebraic, &prepared.expected);
        let threshold = invoke(
            host,
            &geometry.class,
            &mut prepared.fields,
            "max_maa_err",
            Vec::new(),
        )?
        .number()
        .map_err(invalid)?
        .as_f64();
        let success = !matches!(
            error.partial_cmp(&threshold),
            Some(std::cmp::Ordering::Greater)
        );
        parameters.push(json!({"Tensor":id,"Name":metadata.name,"Status":if success{"pass"}else{"fail"},"Coordinates":count,"Precise":precise,"EpsilonF32Bits":epsilon.to_bits(),"ErrorF64Bits":error.to_bits(),"ThresholdF64Bits":threshold.to_bits()}));
        if !success {
            passed = false;
            break;
        }
    }
    Ok(
        json!({"Status":if passed{"pass"}else{"fail"},"Parameters":parameters,"GradientCoordinates":coordinates,"AlgebraicFiniteChecks":finite_checks,"SourceAssertions":host.assertions}),
    )
}
