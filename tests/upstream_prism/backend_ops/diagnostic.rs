// SPDX-License-Identifier: MIT
//! Explicit backend diagnostic/CI dispatch; inline input avoids arbitrary filesystem reads.

use super::super::engine::Engine;
use super::super::{Result, invalid};
use super::value::Scope;
use super::{
    complete, driver, evaluate, gradient,
    graph::Plan,
    host::Production,
    interpreter::Interpreter,
    performance, planner,
    prepare::{self, Mode},
    program, reload, transport,
};
use serde_json::{Value as Json, json};

const KINDS: [&str; 7] = [
    "backend-plan",
    "backend-aggregate-plans",
    "backend-run",
    "backend-aggregate",
    "backend-source-aggregate",
    "backend-options",
    "backend-benchmark",
];
fn keys(request: &Json, required: &[&str], optional: &[&str]) -> Result<()> {
    let object = request
        .as_object()
        .ok_or_else(|| invalid("backend diagnostic must be a JSON object"))?;
    if required.iter().any(|key| !object.contains_key(*key))
        || object
            .keys()
            .any(|key| !required.contains(&key.as_str()) && !optional.contains(&key.as_str()))
    {
        return Err(invalid("backend diagnostic required/unknown fields differ"));
    }
    Ok(())
}
fn text<'a>(request: &'a Json, name: &str) -> Result<&'a str> {
    request[name]
        .as_str()
        .ok_or_else(|| invalid(format!("backend diagnostic {name} must be a string")))
}
fn optional<'a>(request: &'a Json, name: &str) -> Result<Option<&'a str>> {
    if request.get(name).is_none_or(Json::is_null) {
        Ok(None)
    } else {
        text(request, name).map(Some)
    }
}
fn index(request: &Json, name: &str) -> Result<usize> {
    usize::try_from(request[name].as_u64().ok_or_else(|| {
        invalid(format!(
            "backend diagnostic {name} must be an unsigned integer"
        ))
    })?)
    .map_err(Into::into)
}
fn array<'a>(request: &'a Json, name: &str) -> Result<&'a [Json]> {
    request[name]
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| invalid(format!("backend diagnostic {name} must be an array")))
}

pub(super) fn dispatch(engine: &Engine, request: &Json) -> Result<Json> {
    let kind = text(request, "Kind")?;
    if !KINDS.contains(&kind) {
        return Err(invalid("unknown backend diagnostic Kind"));
    }
    if kind != "backend-options" {
        performance::require_ci()?;
    }
    match kind {
        "backend-plan" => {
            keys(request, &["Kind", "Index", "Count", "Mode"], &[])?;
            planner::plan(
                engine,
                index(request, "Index")?,
                index(request, "Count")?,
                Mode::parse(text(request, "Mode")?)?,
                None,
            )
        }
        "backend-aggregate-plans" => {
            keys(request, &["Kind", "Plans"], &[])?;
            planner::aggregate(array(request, "Plans")?)
        }
        "backend-run" => {
            keys(request, &["Kind", "Index", "Count", "Mode", "Plan"], &[])?;
            driver::run(
                engine,
                index(request, "Index")?,
                index(request, "Count")?,
                Mode::parse(text(request, "Mode")?)?,
                &request["Plan"],
            )
        }
        "backend-aggregate" => {
            keys(request, &["Kind", "Reports", "Plan"], &[])?;
            driver::aggregate(array(request, "Reports")?, &request["Plan"])
        }
        "backend-source-aggregate" => {
            keys(request, &["Kind", "Modes"], &[])?;
            complete::aggregate(array(request, "Modes")?)
        }
        "backend-benchmark" => match text(request, "Phase")? {
            "full" => {
                keys(request, &["Kind", "Phase"], &[])?;
                benchmark(engine)
            }
            "aggregate" => {
                keys(request, &["Kind", "Phase", "Reports", "Plan"], &[])?;
                aggregate_benchmarks(array(request, "Reports")?, &request["Plan"])
            }
            _ => Err(invalid("backend benchmark Phase must be full or aggregate")),
        },
        "backend-options" => options(engine, request),
        _ => Err(invalid("unknown backend diagnostic Kind")),
    }
}

pub(super) fn aggregate_benchmarks(reports: &[Json], plan: &Json) -> Result<Json> {
    if plan["Mode"] != "perf" {
        return Err(invalid(
            "backend benchmark aggregation requires a performance plan",
        ));
    }
    let aggregate = driver::aggregate(reports, plan)?;
    let measured = aggregate["PassedGeometries"]
        .as_u64()
        .ok_or_else(|| invalid("backend benchmark measured count missing"))?;
    Ok(
        json!({"SchemaVersion":1,"Benchmark":"backend-ops-performance","Status":if aggregate["Status"]!="pass"{"fail"}else if measured>0{"measured"}else{"unsupported"},"ExpectedWorkloads":program::PERF_COUNT,"MeasuredWorkloads":measured,"CorrectnessSourceCount":0,"Aggregate":aggregate}),
    )
}

pub(super) fn benchmark(engine: &Engine) -> Result<Json> {
    performance::require_ci()?;
    let partition = planner::plan(engine, 0, 1, Mode::Perf, None)?;
    if partition["Status"] != "planned" {
        return Ok(
            json!({"SchemaVersion":1,"Benchmark":"backend-ops-performance","Status":"unsupported","Reason":"unsupported production API: backend graph performance workload","CorrectnessSourceCount":0,"Coverage":partition}),
        );
    }
    let plan = planner::aggregate(&[partition])?;
    let start = std::time::Instant::now();
    let report = driver::run(engine, 0, 1, Mode::Perf, &plan)?;
    let elapsed = start.elapsed().as_nanos();
    if report["Status"] == "unsupported" {
        return Ok(report);
    }
    let mut aggregate = aggregate_benchmarks(std::slice::from_ref(&report), &plan)?;
    aggregate["WallNanoseconds"] = json!(u64::try_from(elapsed)?);
    aggregate["WallScope"] = json!(
        "all case controllers including allocation/input/warmup; per-workload native measured intervals exclude them"
    );
    aggregate["PartitionReport"] = report;
    Ok(aggregate)
}

/// Boolean searches in this explicitly checked ASCII regular-language core have the same language
/// as default ECMAScript `regex_search`. Captures are never observed; backreferences/lookaround and
/// syntax with differing class/Unicode semantics are rejected before executing a case.
fn pattern(pattern: &str) -> Result<regex::Regex> {
    if !pattern.is_ascii()
        || pattern.len() > 64 * 1024
        || pattern.contains("(?")
        || pattern.contains("[:")
        || pattern.contains("&&")
        || pattern.contains("--")
        || pattern.contains("~~")
    {
        return Err(invalid(
            "unsupported diagnostic pattern: outside the shared ASCII ECMAScript regular-language core",
        ));
    }
    let bytes = pattern.as_bytes();
    let mut escaped = false;
    let mut class = false;
    let mut quantifier_end = None;
    for (index, &byte) in bytes.iter().enumerate() {
        if escaped {
            if !b"dDwWsSbBfnrtv.^$|?*+()[]{}\\/-".contains(&byte) {
                return Err(invalid(
                    "unsupported diagnostic escape: backreferences, octal, control/Unicode/property escapes are outside the shared core",
                ));
            }
            escaped = false;
            continue;
        }
        if byte == b'\\' {
            escaped = true;
            continue;
        }
        if byte == 0 || byte == b'\r' || byte == b'\n' {
            return Err(invalid("unsupported diagnostic pattern line/control byte"));
        }
        if byte == b'[' {
            if class {
                return Err(invalid("unsupported diagnostic nested character class"));
            }
            class = true;
        } else if byte == b']' {
            if !class {
                return Err(invalid("unsupported diagnostic unescaped closing bracket"));
            }
            class = false;
        } else if !class && byte == b'{' {
            let end = bytes[index + 1..]
                .iter()
                .position(|&byte| byte == b'}')
                .map(|end| index + 1 + end)
                .ok_or_else(|| invalid("unsupported diagnostic repetition brace"))?;
            let count = &pattern[index + 1..end];
            let fields = count.split(',').collect::<Vec<_>>();
            if fields.is_empty()
                || fields.len() > 2
                || fields[0].is_empty()
                || !fields[0].bytes().all(|byte| byte.is_ascii_digit())
                || fields
                    .get(1)
                    .is_some_and(|field| !field.bytes().all(|byte| byte.is_ascii_digit()))
            {
                return Err(invalid("unsupported diagnostic repetition form"));
            }
            quantifier_end = Some(end);
        } else if !class && byte == b'}' && quantifier_end != Some(index) {
            return Err(invalid(
                "unsupported diagnostic unescaped closing repetition brace",
            ));
        }
    }
    if escaped {
        return Err(invalid("invalid diagnostic regex: final escape"));
    }
    regex::RegexBuilder::new(pattern)
        .unicode(false)
        .size_limit(8 * 1024 * 1024)
        .dfa_size_limit(8 * 1024 * 1024)
        .build()
        .map_err(|error| invalid(format!("invalid or unsupported diagnostic regex: {error}")))
}

fn diagnostic_vars(host: &mut Production<'_, '_>, geometry: &program::Geometry) -> Result<String> {
    let source = host.program;
    let mut runtime = Interpreter::new(source, &geometry.class, host)?;
    runtime.fields = Scope::new();
    runtime.instantiate(geometry)?;
    let value = runtime.call("vars", Vec::new())?;
    let value = runtime.forced(value)?;
    if let super::value::Value::Bytes(bytes) = value {
        String::from_utf8(bytes).map_err(Into::into)
    } else {
        Err(invalid("diagnostic vars() did not return a string"))
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "The diagnostic preserves constructor/parameter filter, graph/operation filter and source CRT sequencing in one ordered transaction."
)]
fn options(engine: &Engine, request: &Json) -> Result<Json> {
    keys(
        request,
        &["Kind", "Mode", "Index", "Count"],
        &["OperationFilter", "ParametersFilter", "TestFileText"],
    )?;
    let mode = Mode::parse(text(request, "Mode")?)?;
    if mode == Mode::Perf {
        performance::require_ci()?;
    }
    let count = index(request, "Count")?;
    let partition_index = index(request, "Index")?;
    planner::partition(partition_index, count)?;
    if count != 1 {
        return Ok(
            json!({"Status":"unsupported-diagnostic","Reason":"filtered/test-file diagnostics preserve one source-order CRT stream; use the planner and complete prefix manifests for full default matrix sharding","CorrectnessSourceCount":0}),
        );
    }
    let operation_filter = optional(request, "OperationFilter")?;
    let parameter_filter = optional(request, "ParametersFilter")?;
    let matcher = if let Some(pattern_text) = parameter_filter {
        match pattern(pattern_text) {
            Ok(matcher) => Some(matcher),
            Err(error) => {
                return Ok(
                    json!({"Status":"unsupported-diagnostic","Reason":error.to_string(),"CorrectnessSourceCount":0}),
                );
            }
        }
    } else {
        None
    };
    let (program, defaults, _) = planner::extent(mode)?;
    let text = optional(request, "TestFileText")?;
    let loaded = text.map(|text| reload::parse(program, text)).transpose()?;
    let geometries = loaded.as_deref().unwrap_or(defaults);
    let threads = std::thread::available_parallelism()?.get();
    let mut offset = 0_u64;
    let mut items = Vec::new();
    let mut retention = super::retention::Budget::default();
    let mut passed = true;
    for (case_index, geometry) in geometries.iter().enumerate() {
        let directory = engine.evidence_path(&format!(
            "backend-diagnostic-{}-{case_index:05}",
            mode.name()
        ))?;
        let observed = transport::with_session(engine, &directory, |session| {
            let mut host = Production::new(
                Plan::new(session),
                program,
                case_index as u64,
                threads,
                mode.code(),
            );
            if let Some(matcher) = &matcher {
                let variables = diagnostic_vars(&mut host, geometry)?;
                if !variables.is_ascii() || variables.bytes().any(|byte| matches!(byte, 10 | 13)) {
                    return Ok(
                        json!({"Index":case_index,"Status":"unsupported-diagnostic","Reason":"non-ASCII/line-containing vars() is outside shared regex semantics"}),
                    );
                }
                if !matcher.is_match(&variables) {
                    return Ok(
                        json!({"Index":case_index,"Status":"skip","Reason":"original parameter regex filter skip","CrtDraws":0}),
                    );
                }
            }
            let mut prepared = prepare::prepare(&mut host, geometry, mode, operation_filter)?;
            let mut result = if prepared.initialize {
                evaluate::allocate(&mut host, &mut prepared, geometry, mode, offset)?;
                match mode {
                    Mode::Eval => evaluate::evaluate(&mut host, &mut prepared, geometry)?,
                    Mode::Grad => gradient::evaluate(&mut host, &mut prepared, geometry)?,
                    Mode::Perf => performance::evaluate(&mut host, &mut prepared, geometry)?,
                    Mode::Support => {
                        return Err(invalid("source support mode cannot initialize tensors"));
                    }
                }
            } else {
                json!({"Status":if prepared.skip.is_some(){"skip"}else{"pass"},"SkipReason":prepared.skip,"Supported":prepared.supported})
            };
            result["Index"] = json!(case_index);
            result["Class"] = json!(geometry.class);
            result["ArgumentsSha256"] = json!(planner::arguments_hash(geometry)?);
            result["Operation"] = json!(prepared.operation);
            result["Variables"] = json!(prepared.variables);
            result["CrtOffset"] = json!(offset);
            result["CrtDraws"] = json!(prepared.crt_draws);
            result["RandomDeviceSeeds"] = json!(host.inputs.ledger.seeds);
            result["InputAndOutputReceipts"] = json!(host.inputs.receipts);
            Ok(result)
        })?;
        let Some(mut observed) = observed else {
            return Ok(
                json!({"Status":"unsupported","Reason":"unsupported production API: generic backend diagnostic graph execution","CorrectnessSourceCount":0,"Items":items}),
            );
        };
        let draws = observed["CrtDraws"].as_u64().unwrap_or(0);
        offset = offset
            .checked_add(draws)
            .ok_or_else(|| invalid("diagnostic CRT offset overflow"))?;
        passed &= matches!(
            observed["Status"].as_str(),
            Some("pass" | "skip" | "measured")
        );
        std::fs::write(
            directory.join("case-receipt.json"),
            serde_json::to_vec_pretty(&observed)?,
        )?;
        if matches!(
            observed["Status"].as_str(),
            Some("pass" | "skip" | "measured")
        ) {
            driver::cleanup(&directory)?;
        } else {
            observed["ArtifactRetention"] = retention.failure(&directory)?;
            std::fs::write(
                directory.join("case-receipt.json"),
                serde_json::to_vec_pretty(&observed)?,
            )?;
        }
        if observed["Status"] == "unsupported-diagnostic" {
            items.push(observed);
            return Ok(
                json!({"Status":"unsupported-diagnostic","CorrectnessSourceCount":0,"Items":items}),
            );
        }
        items.push(observed);
    }
    Ok(
        json!({"SchemaVersion":1,"Status":if passed{"pass"}else{"fail"},"Diagnostic":true,"CorrectnessSourceCount":0,"SourceSha256":program::SOURCE_SHA256,"Mode":mode.name(),"CpuSelection":"explicit CPU","OperationFilter":operation_filter,"ParametersFilter":parameter_filter,"PatternLanguage":"checked ASCII ECMAScript regular-language core; unsupported syntax reported explicitly","TestFileSha256":text.map(|text|planner::hash(text.as_bytes())),"InputGeometries":geometries.len(),"TotalCrtDraws":offset,"Items":items}),
    )
}
