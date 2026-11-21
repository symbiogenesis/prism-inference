//! Independent clean-room Rust reference backend-ops provider.
//!
//! No `bonsai_prism_runtime`, `bonsai_gguf`, or native GGML libraries are linked or called.

#[path = "reference_backend_ops/eval_ops.rs"]
mod eval_ops;
#[path = "reference_backend_ops/eval_ops_extra.rs"]
mod eval_ops_extra;
#[path = "reference_backend_ops/ops_builders.rs"]
mod ops_builders;
#[path = "reference_backend_ops/quants.rs"]
mod quants;
#[path = "reference_backend_ops/state.rs"]
mod state;
#[path = "reference_backend_ops/types.rs"]
mod types;

use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::{Result, invalid};
use state::BackendState;

pub(super) const IDS: &[&str] = &[
    "tests/test-backend-ops.cpp",
    "tests/test-backend-ops.cpp#performance",
];

const MAX_FRAME_LEN: usize = 128 * 1024 * 1024;
const MAX_OBSERVATION_LEN: usize = 16 * 1024 * 1024;
const DEFAULT_MAX_FILE_BYTES: u64 = 4 * 1024 * 1024 * 1024;

pub(super) fn identity() -> Value {
    json!({
        "SourceIds": IDS,
        "Provider": "independent-rust-reference-backend-ops",
        "NativeReferenceReplay": false,
        "ProductionRuntimeCalled": false,
        "CompleteSourceGate": false,
        "NativeRetirementQualified": false,
    })
}

pub(super) fn handles(value: &Value) -> bool {
    value.get("SchemaVersion").and_then(Value::as_u64) == Some(1)
        && value
            .get("Operations")
            .and_then(Value::as_array)
            .is_some_and(|ops| {
                ops.len() == 1
                    && ops[0].get("Kind").and_then(Value::as_str) == Some("backend-graph")
            })
}

fn atomic_write_checkpoint(path: &Path, trace: &[Value]) -> Result<()> {
    let bytes = serde_json::to_vec(&json!({"Trace": trace}))?;
    if bytes.len() > MAX_OBSERVATION_LEN {
        return Err(invalid("backend checkpoint trace exceeds 16 MiB"));
    }
    let pending = path.with_extension("pending");
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&pending)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    drop(file);
    fs::rename(&pending, path)?;
    Ok(())
}

pub(super) fn execute(value: &Value) -> Result<Value> {
    let ops = value
        .get("Operations")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("backend-graph request missing Operations"))?;
    if ops.len() != 1 || ops[0].get("Kind").and_then(Value::as_str) != Some("backend-graph") {
        return Err(invalid(
            "unexpected operation kind for reference_backend_ops",
        ));
    }
    let op = &ops[0];
    let dir = op
        .get("Directory")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("backend-graph missing Directory"))?;
    let max_file_bytes = op
        .get("MaxFileBytes")
        .and_then(Value::as_u64)
        .unwrap_or(DEFAULT_MAX_FILE_BYTES);
    let mut actions = op
        .get("Actions")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("backend-graph missing Actions"))?
        .clone();

    let mut state = BackendState::new(Path::new(dir), max_file_bytes)?;
    let final_trace = loop {
        let mut trace = Vec::with_capacity(actions.len());
        let mut next_actions: Option<Vec<Value>> = None;
        for action in &actions {
            if action.get("Op").and_then(Value::as_str) == Some("checkpoint") {
                let output_raw = action
                    .get("OutputFile")
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid("checkpoint missing OutputFile"))?;
                let next_raw = action
                    .get("NextFile")
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid("checkpoint missing NextFile"))?;
                let timeout_ms = action
                    .get("TimeoutMilliseconds")
                    .and_then(Value::as_u64)
                    .unwrap_or(60_000);
                let out_path = state.checked_path(output_raw, false)?;
                let next_path = state.checked_path(next_raw, false)?;
                trace.push(json!({}));
                atomic_write_checkpoint(&out_path, &trace)?;
                let deadline = Instant::now() + Duration::from_millis(timeout_ms);
                loop {
                    if fs::exists(&next_path)? {
                        let raw = fs::read(&next_path)?;
                        if raw.len() > MAX_FRAME_LEN {
                            return Err(invalid("next action file exceeds 128 MiB"));
                        }
                        let parsed: Vec<Value> = serde_json::from_slice(&raw)?;
                        next_actions = Some(parsed);
                        break;
                    }
                    if Instant::now() >= deadline {
                        return Err(invalid("backend checkpoint wait timed out"));
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
                break;
            }
            trace.push(state.run_action(action)?);
        }
        match next_actions {
            Some(batch) if !batch.is_empty() => actions = batch,
            Some(_) => break Vec::new(),
            None => break trace,
        }
    };
    Ok(json!({"SchemaVersion": 1, "Results": [{"Trace": final_trace}]}))
}

fn read_request_frame(input: &mut impl Read) -> Result<Value> {
    let mut header = [0_u8; 4];
    input.read_exact(&mut header)?;
    let len = usize::try_from(u32::from_le_bytes(header))?;
    if len == 0 || len > MAX_FRAME_LEN {
        return Err(invalid("framed backend request length out of bounds"));
    }
    let mut payload = vec![0_u8; len];
    input.read_exact(&mut payload)?;
    Ok(serde_json::from_slice(&payload)?)
}

fn write_response_frame(output: &mut impl Write, seq: u64, result: &Value) -> Result<()> {
    let bytes = serde_json::to_vec(&json!({"Sequence": seq, "Result": result}))?;
    if bytes.is_empty() || bytes.len() > MAX_OBSERVATION_LEN {
        return Err(invalid("framed backend response exceeds 16 MiB"));
    }
    let header = u32::try_from(bytes.len())?.to_le_bytes();
    output.write_all(&header)?;
    output.write_all(&bytes)?;
    output.flush()?;
    Ok(())
}

pub(super) fn run_framed_session(input: &mut impl Read, output: &mut impl Write) -> Result<()> {
    let mut state: Option<BackendState> = None;
    let mut expected_seq = 1_u64;
    loop {
        let frame = read_request_frame(input)?;
        let obj = frame
            .as_object()
            .ok_or_else(|| invalid("framed request must be a JSON object"))?;
        if obj.len() != 2 || frame.get("Sequence").and_then(Value::as_u64) != Some(expected_seq) {
            return Err(invalid("framed request identity or sequence mismatch"));
        }
        let op_obj = frame
            .get("Operation")
            .and_then(Value::as_object)
            .ok_or_else(|| invalid("framed request missing Operation object"))?;
        let op_name = op_obj
            .get("Op")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("framed request missing Op string"))?;
        let seq = expected_seq;
        expected_seq += 1;
        match op_name {
            "open" => {
                if state.is_some() {
                    return Err(invalid(
                        "backend pipe open called while session already open",
                    ));
                }
                let dir = op_obj
                    .get("Directory")
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid("backend pipe open missing Directory"))?;
                let max_bytes = op_obj
                    .get("MaxFileBytes")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| invalid("backend pipe open missing MaxFileBytes"))?;
                state = Some(BackendState::new(Path::new(dir), max_bytes)?);
                write_response_frame(output, seq, &json!({}))?;
            }
            "actions" => {
                let st = state
                    .as_mut()
                    .ok_or_else(|| invalid("backend pipe actions called with no open session"))?;
                let actions = op_obj
                    .get("Actions")
                    .and_then(Value::as_array)
                    .ok_or_else(|| invalid("backend pipe actions missing Actions array"))?;
                if actions.is_empty() {
                    return Err(invalid("backend pipe actions received empty batch"));
                }
                let mut trace = Vec::with_capacity(actions.len());
                for action in actions {
                    trace.push(st.run_action(action)?);
                }
                write_response_frame(output, seq, &json!({"Trace": trace}))?;
            }
            "close" => {
                if state.take().is_none() {
                    return Err(invalid("backend pipe close called with no open session"));
                }
                write_response_frame(output, seq, &json!({}))?;
            }
            "shutdown" => {
                if state.is_some() {
                    return Err(invalid("backend pipe shutdown called before close"));
                }
                write_response_frame(output, seq, &json!({}))?;
                return Ok(());
            }
            _ => {
                return Err(invalid(format!("unknown framed backend op {op_name}")));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn frame_request(val: &Value) -> Vec<u8> {
        let payload = serde_json::to_vec(val).unwrap_or_default();
        let mut bytes = u32::try_from(payload.len())
            .unwrap_or(0)
            .to_le_bytes()
            .to_vec();
        bytes.extend_from_slice(&payload);
        bytes
    }

    #[test]
    fn reference_backend_ops_handles_match_only_backend_graph_requests() {
        assert!(handles(&json!({
            "SchemaVersion": 1,
            "Operations": [{"Kind": "backend-graph"}]
        })));
        assert!(!handles(&json!({
            "SchemaVersion": 1,
            "Operations": [{"Kind": "vocab-tokenize"}]
        })));
    }

    #[test]
    fn reference_backend_ops_framed_session_runs_add_and_rejects_malformed_frames() -> Result<()> {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "prism-ref-backend-ops-{}-{stamp}",
            std::process::id()
        ));
        fs::create_dir(&root)?;
        let a_path = root.join("a.bin");
        let out_path = root.join("out.bin");
        let a_vals: Vec<u8> = [1.0_f32, 2.5, -3.0, 4.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        fs::write(&a_path, &a_vals)?;

        let mut req_bytes = Vec::new();
        req_bytes.extend(frame_request(&json!({
            "Sequence": 1,
            "Operation": {"Op": "open", "Directory": root, "MaxFileBytes": DEFAULT_MAX_FILE_BYTES}
        })));
        req_bytes.extend(frame_request(&json!({
            "Sequence": 2,
            "Operation": {"Op": "actions", "Actions": [
                {"Op": "context-init", "Id": 0, "MemoryBytes": 65536, "NoAlloc": true},
                {"Op": "backend-init", "Id": 0, "Device": "CPU", "Threads": 1, "UseRef": false},
                {"Op": "call", "Function": "ggml_new_tensor_1d", "Arguments": [{"Context": 0}, {"I32": 0}, {"I64": 4}], "Result": {"Tensor": 0}},
                {"Op": "call", "Function": "ggml_add", "Arguments": [{"Context": 0}, {"Tensor": 0}, {"Tensor": 0}], "Result": {"Tensor": 1}},
                {"Op": "call", "Function": "ggml_new_graph", "Arguments": [{"Context": 0}], "Result": {"Graph": 0}},
                {"Op": "call", "Function": "ggml_build_forward_expand", "Arguments": [{"Graph": 0}, {"Tensor": 1}], "Result": null},
                {"Op": "allocate", "Context": 0, "Backend": 0, "Result": {"Buffer": 0}},
                {"Op": "tensor-set", "Tensor": 0, "File": a_path, "FileOffset": 0, "TensorOffset": 0, "Bytes": 16},
                {"Op": "call", "Function": "ggml_backend_graph_compute", "Arguments": [{"Backend": 0}, {"Graph": 0}], "Result": {"Scalar": 0}},
                {"Op": "scalar", "Scalar": 0, "Release": true},
                {"Op": "tensor-get", "Tensor": 1, "File": out_path, "FileOffset": 0, "TensorOffset": 0, "Bytes": 16}
            ]}
        })));
        req_bytes.extend(frame_request(
            &json!({"Sequence": 3, "Operation": {"Op": "close"}}),
        ));
        req_bytes.extend(frame_request(
            &json!({"Sequence": 4, "Operation": {"Op": "shutdown"}}),
        ));

        let mut resp = Vec::new();
        run_framed_session(&mut Cursor::new(req_bytes), &mut resp)?;
        let raw_out = fs::read(&out_path)?;
        let out_floats: Vec<f32> = raw_out
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        assert_eq!(out_floats, vec![2.0, 5.0, -6.0, 8.0]);

        assert!(run_framed_session(&mut Cursor::new(Vec::<u8>::new()), &mut Vec::new()).is_err());
        assert!(run_framed_session(&mut Cursor::new([0_u8, 0, 0, 0]), &mut Vec::new()).is_err());
        fs::remove_dir_all(root)?;
        Ok(())
    }
}
