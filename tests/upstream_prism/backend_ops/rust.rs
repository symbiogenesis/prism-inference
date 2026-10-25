//! Wire adapters for the production safe Rust CPU graph API.

use super::super::{Result, invalid};
use bonsai_prism_runtime::backend::{Argument, CpuBackend, Observation, TensorId};
use serde_json::{Value, json};

#[path = "rust_graph.rs"]
mod graph;
pub(super) use graph::GraphSession;

fn array(value: &Value) -> Result<&[Value]> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| invalid("Rust graph requires an array"))
}
fn text(value: &Value) -> Result<&str> {
    value
        .as_str()
        .ok_or_else(|| invalid("Rust graph requires text"))
}
fn integer(value: &Value) -> Result<i64> {
    value
        .as_i64()
        .ok_or_else(|| invalid("Rust graph requires a signed integer"))
}
fn unsigned(value: &Value) -> Result<usize> {
    Ok(usize::try_from(value.as_u64().ok_or_else(|| {
        invalid("Rust graph requires an unsigned integer")
    })?)?)
}
fn real(value: &Value) -> Result<f32> {
    let value = value
        .as_f64()
        .ok_or_else(|| invalid("Rust graph requires a real"))? as f32;
    if !value.is_finite() {
        return Err(invalid("Rust graph requires a finite real"));
    }
    Ok(value)
}
fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8] = b"0123456789abcdef";
    bytes
        .iter()
        .flat_map(|&byte| {
            [
                char::from(DIGITS[usize::from(byte >> 4)]),
                char::from(DIGITS[usize::from(byte & 15)]),
            ]
        })
        .collect()
}
fn unhex(value: &Value) -> Result<Vec<u8>> {
    let text = text(value)?;
    if text.len() > 128 * 1024 * 1024 || !text.len().is_multiple_of(2) {
        return Err(invalid("Rust graph hexadecimal extent differs"));
    }
    text.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| Ok(u8::from_str_radix(std::str::from_utf8(pair)?, 16)?))
        .collect()
}
fn kind(value: &Value) -> Result<i32> {
    match text(value)? {
        "F32" => Ok(0),
        "F16" => Ok(1),
        "BF16" => Ok(30),
        "I32" => Ok(26),
        _ => Err(invalid("Rust graph transport type is unsupported")),
    }
}
fn type_name(kind: i32) -> Result<&'static str> {
    match kind {
        0 => Ok("F32"),
        1 => Ok("F16"),
        30 => Ok("BF16"),
        26 => Ok("I32"),
        _ => Err(invalid("Rust graph response type is unsupported")),
    }
}
fn result_tensor(observation: &Observation) -> Result<TensorId> {
    if let Observation::Tensor(Some(id)) = observation {
        Ok(*id)
    } else {
        Err(invalid("Rust graph call did not return a tensor"))
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "Transport dispatch only constructs production calls and serializes their results."
)]
fn tensor_graph(
    operation: &Value,
    arena: Option<&Value>,
    forward: Option<&Value>,
) -> Result<Value> {
    let nodes = array(&operation["Nodes"])?;
    if nodes.is_empty() || nodes.len() > 96 {
        return Err(invalid("Rust tensor graph node count outside bound"));
    }
    let mut backend = CpuBackend::new();
    let mut tensors: Vec<TensorId> = Vec::new();
    let mut inputs = Vec::new();
    for node in nodes {
        let name = text(&node["Op"])?;
        let id = if name == "input" {
            let mut shape = [1; 4];
            let dims = array(&node["Dimensions"])?;
            if dims.is_empty() || dims.len() > 4 {
                return Err(invalid("Rust tensor graph rank differs"));
            }
            for (destination, dimension) in shape.iter_mut().zip(dims) {
                *destination = unsigned(dimension)?;
            }
            let id = backend.tensor(1, kind(&node["Type"])?, shape)?;
            let bytes = unhex(&node["DataHex"])?;
            if bytes.len() != backend.info(id)?.bytes()? {
                return Err(invalid("Rust tensor input byte extent differs"));
            }
            inputs.push((id, bytes));
            id
        } else {
            let refs = array(&node["Inputs"])?;
            let mut args = vec![Argument::Context(1)];
            for reference in refs {
                args.push(Argument::Tensor(
                    *tensors
                        .get(unsigned(reference)?)
                        .ok_or_else(|| invalid("Rust graph reference exceeds preceding nodes"))?,
                ));
            }
            let production = match name {
                "cast" => {
                    let a = *tensors
                        .get(unsigned(&refs[0])?)
                        .ok_or_else(|| invalid("Rust graph cast source missing"))?;
                    let output = backend.tensor(1, kind(&node["Type"])?, backend.info(a)?.shape)?;
                    args.push(Argument::Tensor(output));
                    "ggml_cpy"
                }
                "reshape" => {
                    let dims = array(&node["Dimensions"])?;
                    for axis in 0..4 {
                        args.push(Argument::Integer(dims.get(axis).map_or(Ok(1), integer)?));
                    }
                    "ggml_reshape_4d"
                }
                "permute" => {
                    for axis in array(&node["Axes"])? {
                        args.push(Argument::Integer(integer(axis)?));
                    }
                    "ggml_permute"
                }
                "rms_norm" => {
                    args.push(Argument::Float(real(&node["Epsilon"])?));
                    "ggml_rms_norm"
                }
                "view_2d" => {
                    for dim in array(&node["Dimensions"])? {
                        args.push(Argument::Integer(integer(dim)?));
                    }
                    args.push(Argument::Integer(integer(&node["RowStrideBytes"])?));
                    args.push(Argument::Integer(integer(&node["OffsetBytes"])?));
                    "ggml_view_2d"
                }
                "conv_transpose_1d" => {
                    for field in ["Stride", "Padding", "Dilation"] {
                        args.push(Argument::Integer(integer(&node[field])?));
                    }
                    "ggml_conv_transpose_1d"
                }
                "col2im_1d" => {
                    for field in ["Stride", "Channels", "Padding"] {
                        args.push(Argument::Integer(integer(&node[field])?));
                    }
                    "ggml_col2im_1d"
                }
                "rope" => {
                    args.push(Argument::Integer(integer(&node["Rotations"])?));
                    args.push(Argument::Integer(integer(&node["Mode"])?));
                    "ggml_rope"
                }
                "rope_multi" => {
                    args.push(Argument::Null);
                    args.push(Argument::Integer(integer(&node["Rotations"])?));
                    args.push(Argument::Integers(
                        array(&node["Sections"])?
                            .iter()
                            .map(integer)
                            .collect::<Result<Vec<_>>>()?,
                    ));
                    args.push(Argument::Integer(integer(&node["Mode"])?));
                    args.push(Argument::Integer(integer(&node["Context"])?));
                    for field in [
                        "FrequencyBase",
                        "FrequencyScale",
                        "ExtensionFactor",
                        "AttentionFactor",
                        "BetaFast",
                        "BetaSlow",
                    ] {
                        args.push(Argument::Float(real(&node[field])?));
                    }
                    "ggml_rope_multi"
                }
                "cont" => "ggml_cont",
                "transpose" => "ggml_transpose",
                "soft_max" => "ggml_soft_max",
                "mul_mat" => "ggml_mul_mat",
                "add" => "ggml_add",
                "mul" => "ggml_mul",
                _ => {
                    return Err(invalid(format!(
                        "unsupported Rust tensor graph operation {name}"
                    )));
                }
            };
            result_tensor(&backend.call(production, &args)?)?
        };
        tensors.push(id);
    }
    if let Some(arena) = arena.filter(|value| !value.is_null()) {
        let offsets = array(&arena["Offsets"])?;
        if offsets.len() != tensors.len() {
            return Err(invalid("Rust arena placement count differs"));
        }
        let placements = offsets
            .iter()
            .zip(&tensors)
            .filter(|(offset, _)| !offset.is_null())
            .map(|(offset, &id)| Ok((id, unsigned(offset)?)))
            .collect::<Result<Vec<_>>>()?;
        backend.allocate_arena(unsigned(&arena["BufferBytes"])?, &placements)?;
    }
    backend.allocate(1)?;
    for (id, bytes) in inputs {
        backend.set(id, 0, &bytes)?;
    }
    let Observation::Graph(graph) = backend.call("ggml_new_graph", &[Argument::Context(1)])? else {
        return Err(invalid("Rust graph allocation result differs"));
    };
    let outputs = array(&operation["Outputs"])?;
    if let Some(forward) = forward {
        for reference in array(forward)? {
            let id = *tensors
                .get(unsigned(reference)?)
                .ok_or_else(|| invalid("Rust forward reference exceeds graph"))?;
            backend.call(
                "ggml_build_forward_expand",
                &[Argument::Graph(graph), Argument::Tensor(id)],
            )?;
        }
    }
    for reference in outputs {
        let id = *tensors
            .get(unsigned(reference)?)
            .ok_or_else(|| invalid("Rust output reference exceeds graph"))?;
        backend.call(
            "ggml_build_forward_expand",
            &[Argument::Graph(graph), Argument::Tensor(id)],
        )?;
    }
    backend.compute(graph)?;
    let outputs=outputs.iter().map(|reference|{let id=*tensors.get(unsigned(reference)?).ok_or_else(||invalid("Rust output missing"))?;let info=backend.info(id)?;if !info.contiguous(){return Err(invalid("Rust transport output must be contiguous"));}Ok(json!({"Type":type_name(info.kind)?,"Dimensions":info.shape,"DataHex":hex(backend.get(id,0,info.bytes()?)?)}))}).collect::<Result<Vec<_>>>()?;
    if arena.is_some() {
        Ok(json!({"Status":0,"Outputs":outputs}))
    } else {
        Ok(json!({"Outputs":outputs}))
    }
}

pub(super) fn execute(request: &Value) -> Result<Option<Value>> {
    if request["SchemaVersion"] != 1 {
        return Err(invalid("Rust graph request version differs"));
    }
    let operations = array(&request["Operations"])?;
    if operations.iter().any(|op| {
        !matches!(
            op["Kind"].as_str(),
            Some(
                "backend-graph"
                    | "tensor-graph"
                    | "cpu-buffer-info"
                    | "cpu-arena-graph"
                    | "allocation-graph"
                    | "cpu-threadpool"
            )
        )
    }) {
        return Ok(None);
    }
    let results = operations
        .iter()
        .map(|operation| match operation["Kind"].as_str() {
            Some("backend-graph") => graph::execute(operation),
            Some("allocation-graph") => super::super::allocation::execute(operation),
            Some("cpu-threadpool") => super::super::barrier::execute(operation),
            Some("tensor-graph") => tensor_graph(operation, None, None),
            Some("cpu-buffer-info") => Ok(json!({"Alignment":64})),
            Some("cpu-arena-graph") => tensor_graph(
                &operation["Graph"],
                Some(&operation["Allocation"]),
                Some(&operation["Forward"]),
            ),
            _ => Err(invalid("unknown Rust graph operation kind")),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(json!({"SchemaVersion":1,"Results":results})))
}

#[cfg(test)]
mod tests {
    use super::super::super::engine::{Engine, Target};
    use super::super::transport;
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn rust_backend_continuation_executes_owned_numerical_graph() -> Result<()> {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let directory =
            std::env::temp_dir().join(format!("prism-rust-backend-{}-{nonce}", std::process::id()));
        fs::create_dir(&directory)?;
        let input = directory.join("input.bin");
        let output = directory.join("output.bin");
        fs::write(
            &input,
            [1.0_f32, 2.0, 3.0, 4.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let engine = Engine::new(Target::Rust, None, None, None, None, None)?;
        let outcome = transport::with_session(&engine, &directory, |session| {
            let observed=session.exchange(vec![
                json!({"Op":"context-init","Id":1,"MemoryBytes":1_048_576,"NoAlloc":true}),
                json!({"Op":"backend-init","Id":1,"Device":"CPU","Threads":1,"UseRef":false}),
                json!({"Op":"call","Function":"ggml_new_tensor_1d","Arguments":[{"Context":1},{"I32":0},{"I64":4}],"Result":{"Tensor":1}}),
                json!({"Op":"call","Function":"ggml_scale_bias","Arguments":[{"Context":1},{"Tensor":1},{"F32Bits":2.0_f32.to_bits()},{"F32Bits":1.0_f32.to_bits()}],"Result":{"Tensor":2}}),
                json!({"Op":"allocate","Context":1,"Backend":1,"Result":{"Buffer":1}}),
                json!({"Op":"tensor-set","Tensor":1,"File":input,"FileOffset":0,"TensorOffset":0,"Bytes":16}),
                json!({"Op":"call","Function":"ggml_new_graph","Arguments":[{"Context":1}],"Result":{"Graph":1}}),
                json!({"Op":"call","Function":"ggml_build_forward_expand","Arguments":[{"Graph":1},{"Tensor":2}],"Result":Value::Null}),
                json!({"Op":"call","Function":"ggml_backend_graph_compute","Arguments":[{"Backend":1},{"Graph":1}],"Result":{"Scalar":1}}),
                json!({"Op":"scalar","Scalar":1}),
                json!({"Op":"tensor-get","Tensor":2,"File":output,"FileOffset":0,"TensorOffset":0,"Bytes":16}),
            ])?;
            assert_eq!(observed[9], json!({"Value":{"I32":0}}));
            let bytes = fs::read(&output)?;
            let actual = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| f32::from_le_bytes(*bytes))
                .collect::<Vec<_>>();
            assert_eq!(actual, [3.0, 5.0, 7.0, 9.0]);
            Ok(())
        });
        for entry in fs::read_dir(&directory)? {
            fs::remove_file(entry?.path())?;
        }
        fs::remove_dir(&directory)?;
        assert!(outcome?.is_some());
        Ok(())
    }

    #[test]
    fn rust_backend_first_original_geometry_preserves_source_controller() -> Result<()> {
        use super::super::{
            evaluate,
            graph::Plan,
            host::Production,
            prepare::{self, Mode},
            program,
        };
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "prism-rust-original-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&directory)?;
        let engine = Engine::new(Target::Rust, None, None, None, None, None)?;
        let (program, geometries) = program::load().map_err(invalid)?;
        let geometry = &geometries.eval[0];
        let outcome = transport::with_session(&engine, &directory, |session| {
            let mut host = Production::new(Plan::new(session), program, 0, 1, Mode::Eval.code());
            let mut prepared = prepare::prepare(&mut host, geometry, Mode::Eval, None)?;
            assert!(prepared.supported);
            assert!(prepared.initialize);
            evaluate::allocate(&mut host, &mut prepared, geometry, Mode::Eval, 0)?;
            let result = evaluate::evaluate(&mut host, &mut prepared, geometry)?;
            assert_eq!(result["Status"], "pass");
            Ok(())
        });
        for entry in fs::read_dir(&directory)? {
            fs::remove_file(entry?.path())?;
        }
        fs::remove_dir(&directory)?;
        assert!(outcome?.is_some());
        Ok(())
    }

    #[test]
    fn rust_backend_original_quantized_cpy_has_zero_crt_draws() -> Result<()> {
        use super::super::{
            evaluate,
            graph::Plan,
            host::Production,
            prepare::{self, Mode},
            program,
        };
        // Exact catalog entry: Q4_0 -> Q4_0, original 32x2x3x4 geometry.
        // Its source initializer consumes no global CRT draws, so the global
        // offset cannot affect this focused check. Geometry-index seeds remain
        // exactly those used by complete ordered execution.
        const INDEX: usize = 7527;
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let directory =
            std::env::temp_dir().join(format!("prism-rust-q4-cpy-{}-{nonce}", std::process::id()));
        fs::create_dir(&directory)?;
        let engine = Engine::new(Target::Rust, None, None, None, None, None)?;
        let (program, geometries) = program::load().map_err(invalid)?;
        let geometry = &geometries.eval[INDEX];
        assert_eq!(geometry.class, "test_cpy");
        assert_eq!(geometry.arguments[0], 2);
        assert_eq!(geometry.arguments[1], 2);
        let outcome = transport::with_session(&engine, &directory, |session| {
            let mut host = Production::new(
                Plan::new(session),
                program,
                INDEX as u64,
                1,
                Mode::Eval.code(),
            );
            let mut prepared = prepare::prepare(&mut host, geometry, Mode::Eval, None)?;
            assert!(prepared.supported);
            assert!(prepared.initialize);
            assert_eq!(prepared.crt_draws, 0);
            evaluate::allocate(&mut host, &mut prepared, geometry, Mode::Eval, 0)?;
            let result = evaluate::evaluate(&mut host, &mut prepared, geometry)?;
            assert_eq!(result["Status"], "pass");
            Ok(())
        });
        for entry in fs::read_dir(&directory)? {
            fs::remove_file(entry?.path())?;
        }
        fs::remove_dir(&directory)?;
        assert!(outcome?.is_some());
        Ok(())
    }

    #[test]
    fn rust_backend_hosted_argsort_geometry_preserves_original_crt_stream() -> Result<()> {
        use super::super::{
            evaluate,
            graph::Plan,
            host::Production,
            prepare::{self, Mode},
            program,
        };
        // Hosted run40153890429 partition10 exhausted the bounded action
        // transport while its original loop queried nelements262,400times.
        // Preserve the exact geometry, index seed and joined globalCRT offset.
        const INDEX: usize = 15450;
        const OFFSET: u64 = 7_748_238;
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "prism-hosted-argsort-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&directory)?;
        // An explicit immutable production bridge enables the same focused
        // reproduction against C++; ordinary CI runs the Rust backend here.
        let bridge = std::env::var_os("PRISM_FOCUSED_COMMON_BRIDGE").map(std::path::PathBuf::from);
        let retain_native_evidence = bridge.is_some();
        let target = if bridge.is_some() {
            Target::Cpp
        } else {
            Target::Rust
        };
        let engine = Engine::new(target, None, bridge, Some(directory.clone()), None, None)?;
        let (program, geometries) = program::load().map_err(invalid)?;
        let geometry = &geometries.eval[INDEX];
        assert_eq!(geometry.class, "test_argsort");
        let outcome = transport::with_session(&engine, &directory, |session| {
            let mut host = Production::new(
                Plan::new(session),
                program,
                INDEX as u64,
                1,
                Mode::Eval.code(),
            );
            let mut prepared = prepare::prepare(&mut host, geometry, Mode::Eval, None)?;
            assert!(prepared.supported && prepared.initialize);
            assert_eq!(prepared.crt_draws, 262_400);
            evaluate::allocate(&mut host, &mut prepared, geometry, Mode::Eval, OFFSET)?;
            let result = evaluate::evaluate(&mut host, &mut prepared, geometry)?;
            assert_eq!(result["Status"], "pass");
            assert_eq!(host.crt.draws, OFFSET + prepared.crt_draws);
            Ok(())
        });
        assert!(outcome?.is_some());
        if retain_native_evidence {
            return Ok(());
        }
        for entry in fs::read_dir(&directory)? {
            fs::remove_file(entry?.path())?;
        }
        fs::remove_dir(&directory)?;
        Ok(())
    }
}
