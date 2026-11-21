//! Framed JSON-lines session dispatcher replacing native `common/bridge.cpp` IPC.
//!
//! No `bonsai_prism_runtime`, `bonsai_gguf`, or external native child process is used.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::thread;

use serde_json::{Value, json};

use super::architectures::{
    execute_architecture_inventory, execute_devices, execute_model_chat_render,
};
use super::dspark::{DsparkEnv, DsparkState, logits_rows_slice, observe_production};
use super::forward::{
    BatchData, BatchRow, ContextState, RuntimeEnv, build_context, check_batch_limits,
    check_context_limits, context_graph_reserve, context_limits, load_bias_gguf, run_decode,
};
use super::gguf_model::{
    GgufValue, ModelData, ModelVocab, TensorEntry, checked_session_file, vocab_type_from_model,
};
use super::llg_vocab::{execute_llg, execute_vocab, handles_llg, handles_vocab};
use super::params::{
    SessionParams, checked_model_path, hardware_concurrency, parse_params, update_params,
    validate_state_name,
};
use super::sampler::{BackendSamplerRegistry, SamplerEventCb, SamplerNode, bs_piece, bs_tokenize};
use crate::reference_llguidance::LlgRegistry;
use crate::{Result, invalid};

const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;
const MAX_JSON_DEPTH: usize = 16;
const MAX_REQUESTS: usize = 65_536;
const MAX_BLOBS: usize = 64;
const MAX_PARAMS: usize = 64;
const MAX_MODELS: usize = 64;

struct SessionTables {
    params: BTreeMap<u64, SessionParams>,
    models: BTreeMap<u64, Arc<ModelData>>,
    contexts: BTreeMap<u64, ContextState>,
    batches: BTreeMap<u64, BatchData>,
    samplers: BTreeMap<u64, SamplerNode>,
    blobs: BTreeMap<u64, Vec<u8>>,
    llg: LlgRegistry,
    backend_samplers: BackendSamplerRegistry,
    dspark: DsparkState,
    env: RuntimeEnv,
    next_id: u64,
    backend_initialized: bool,
}

impl SessionTables {
    fn new() -> Self {
        Self {
            params: BTreeMap::new(),
            models: BTreeMap::new(),
            contexts: BTreeMap::new(),
            batches: BTreeMap::new(),
            samplers: BTreeMap::new(),
            blobs: BTreeMap::new(),
            llg: LlgRegistry::new(),
            backend_samplers: BackendSamplerRegistry::new(),
            dspark: DsparkState::new(),
            env: RuntimeEnv::default(),
            next_id: 1,
            backend_initialized: false,
        }
    }

    const fn alloc_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }
}

struct WireWriter<W> {
    out: Mutex<W>,
    pending_callbacks: Mutex<BTreeMap<u64, Value>>,
    cv: Condvar,
    next_cb: Mutex<u64>,
}

impl<W: Write> WireWriter<W> {
    const fn new(out: W) -> Self {
        Self {
            out: Mutex::new(out),
            pending_callbacks: Mutex::new(BTreeMap::new()),
            cv: Condvar::new(),
            next_cb: Mutex::new(1),
        }
    }

    fn write_frame(&self, frame: &Value) -> Result<()> {
        let line = serde_json::to_vec(frame)?;
        if line.len() > MAX_FRAME_BYTES {
            return Err(invalid("session reply exceeds 16 MiB frame limit"));
        }
        let mut guard = self
            .out
            .lock()
            .map_err(|_| invalid("writer mutex poisoned"))?;
        guard.write_all(&line)?;
        guard.write_all(b"\n")?;
        guard.flush()?;
        drop(guard);
        Ok(())
    }

    fn call_client(&self, mut frame: Value) -> Result<Value> {
        let cb_id = {
            let mut next = self
                .next_cb
                .lock()
                .map_err(|_| invalid("callback counter poisoned"))?;
            let id = *next;
            *next += 1;
            id
        };
        if let Some(obj) = frame.as_object_mut() {
            obj.insert("CallbackId".into(), json!(cb_id));
        }
        self.write_frame(&frame)?;
        let mut pending = self
            .pending_callbacks
            .lock()
            .map_err(|_| invalid("callback map poisoned"))?;
        loop {
            if let Some(reply) = pending.remove(&cb_id) {
                return Ok(reply);
            }
            pending = self
                .cv
                .wait(pending)
                .map_err(|_| invalid("callback condvar poisoned"))?;
        }
    }

    fn deliver_callback(&self, cb_id: u64, msg: Value) -> Result<()> {
        self.pending_callbacks
            .lock()
            .map_err(|_| invalid("callback map poisoned"))?
            .insert(cb_id, msg);
        self.cv.notify_all();
        Ok(())
    }
}

struct WorkerJob {
    req_id: u64,
    op_val: Value,
}

pub(super) fn run_framed_session_with_env<R: Read, W: Write + Send>(
    reader: R,
    writer: W,
    session_dir: &Path,
    initial_env: &[(String, String)],
) -> Result<()> {
    let mut initial_tables = SessionTables::new();
    for (k, v) in initial_env {
        initial_tables
            .env
            .vars
            .insert(k.clone(), Value::String(v.clone()));
    }
    let tables = Arc::new(Mutex::new(initial_tables));
    let wire = Arc::new(WireWriter::new(BufWriter::new(writer)));
    let mut lines = BufReader::new(reader);
    let mut buf = Vec::new();
    let mut request_count = 0_usize;

    let res: Result<()> = thread::scope(|scope| {
        let mut workers: BTreeMap<u64, mpsc::Sender<WorkerJob>> = BTreeMap::new();
        let mut handles = Vec::new();
        loop {
            buf.clear();
            let n = lines.read_until(b'\n', &mut buf)?;
            if n == 0 {
                break;
            }
            if buf.len() > MAX_FRAME_BYTES {
                return Err(invalid("request frame exceeds 16 MiB limit"));
            }
            let line = std::str::from_utf8(&buf)?.trim();
            if line.is_empty() {
                continue;
            }
            let msg = parse_strict_json(line)?;
            if let Some(cb_id) = msg.get("CallbackId").and_then(Value::as_u64) {
                wire.deliver_callback(cb_id, msg)?;
                continue;
            }
            if msg.get("Shutdown").and_then(Value::as_bool) == Some(true) {
                break;
            }
            request_count += 1;
            if request_count > MAX_REQUESTS {
                return Err(invalid("session request limit 65536 exceeded"));
            }
            let req_id = msg
                .get("Id")
                .and_then(Value::as_u64)
                .filter(|&id| (1..=1_000_000).contains(&id))
                .ok_or_else(|| invalid("request Id must be in 1..=1_000_000"))?;
            let worker_idx = msg.get("Worker").and_then(Value::as_u64).unwrap_or(0);
            if worker_idx > 15 {
                return Err(invalid("Worker index outside 0..=15"));
            }
            let op_val = msg
                .get("Operation")
                .cloned()
                .ok_or_else(|| invalid("missing Operation field"))?;
            let tx = match workers.entry(worker_idx) {
                std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::btree_map::Entry::Vacant(slot) => {
                    let (sender, receiver) = mpsc::channel::<WorkerJob>();
                    let tables_c = Arc::clone(&tables);
                    let wire_c = Arc::clone(&wire);
                    handles.push(scope.spawn(move || {
                        for job in receiver {
                            let reply = match dispatch_operation(
                                job.req_id,
                                &job.op_val,
                                &tables_c,
                                &wire_c,
                                session_dir,
                            ) {
                                Ok(v) => json!({ "Id": job.req_id, "Result": v }),
                                Err(err) => json!({ "Id": job.req_id, "Error": err.to_string() }),
                            };
                            let _ = wire_c.write_frame(&reply);
                        }
                    }));
                    slot.insert(sender)
                }
            };
            tx.send(WorkerJob { req_id, op_val })
                .map_err(|_| invalid("worker thread stopped unexpectedly"))?;
        }
        drop(workers);
        for handle in handles {
            if handle.join().is_err() {
                return Err(invalid("session worker thread panicked"));
            }
        }
        wire.write_frame(&json!({ "Shutdown": true }))?;
        Ok(())
    });
    res
}

fn parse_strict_json(raw: &str) -> Result<Value> {
    let val: Value = serde_json::from_str(raw)?;
    check_depth(&val, 0)?;
    Ok(val)
}

fn check_depth(val: &Value, depth: usize) -> Result<()> {
    if depth > MAX_JSON_DEPTH {
        return Err(invalid("JSON nesting depth exceeds 16"));
    }
    match val {
        Value::Array(items) => {
            for item in items {
                check_depth(item, depth + 1)?;
            }
        }
        Value::Object(map) => {
            for v in map.values() {
                check_depth(v, depth + 1)?;
            }
        }
        _ => {}
    }
    Ok(())
}

#[expect(
    clippy::too_many_lines,
    reason = "single top-level operation router cleanly delegating to domain handlers"
)]
fn dispatch_operation<W: Write>(
    req_id: u64,
    op_val: &Value,
    tables: &Arc<Mutex<SessionTables>>,
    wire: &Arc<WireWriter<W>>,
    session_dir: &Path,
) -> Result<Value> {
    let op = op_val["Op"]
        .as_str()
        .ok_or_else(|| invalid("Operation.Op must be string"))?;
    match op {
        "model-load" => return exec_model_load(req_id, op_val, tables, wire),
        "model-user" | "model-init-from-user" => {
            return exec_model_init_from_user(req_id, op_val, tables, wire, session_dir);
        }
        "bs-sampler-backend-init" | "bs-candidates-apply" => {
            return exec_backend_sampler_cb(req_id, op, op_val, tables, wire);
        }
        _ => {}
    }

    let mut guard = tables
        .lock()
        .map_err(|_| invalid("session tables mutex poisoned"))?;

    if handles_llg(op) {
        let sess = &mut *guard;
        return execute_llg(&mut sess.llg, op_val, |m_val| {
            let mid = m_val
                .as_u64()
                .ok_or_else(|| invalid("LLG Model handle invalid"))?;
            sess.models
                .get(&mid)
                .cloned()
                .ok_or_else(|| invalid("unknown Model handle"))
        });
    }
    if handles_vocab(op) {
        let vocab_ref = if let Some(cid) = op_val.get("Context").and_then(Value::as_u64) {
            let ctx = guard
                .contexts
                .get(&cid)
                .ok_or_else(|| invalid("unknown Context handle"))?;
            Some(Arc::clone(ctx.model.vocab.full_ref()?))
        } else if let Some(mid) = op_val.get("Model").and_then(Value::as_u64) {
            let m = guard
                .models
                .get(&mid)
                .ok_or_else(|| invalid("unknown Model handle"))?;
            Some(Arc::clone(m.vocab.full_ref()?))
        } else {
            None
        };
        return execute_vocab(op_val, vocab_ref.as_ref());
    }
    if op == "observe-production" {
        return exec_with_dspark(&mut guard, session_dir, |d_env| {
            observe_production(&op_val["Operation"], d_env)
        });
    }
    if super::dspark::handles(op) {
        return exec_with_dspark(&mut guard, session_dir, |d_env| {
            super::dspark::execute(op, op_val, d_env)
        });
    }

    match op {
        "devices" => execute_devices(op_val),
        "architecture-inventory" => execute_architecture_inventory(op_val),
        "model-chat-render" => {
            let mid = req_handle(op_val, "Model")?;
            let m = get_model(&guard.models, mid)?;
            execute_model_chat_render(op_val, &m)
        }
        "backend-load-all" | "backend-init" | "common-init" | "numa-init" => {
            guard.backend_initialized = true;
            Ok(Value::Null)
        }
        "backend-free" => {
            guard.backend_initialized = false;
            Ok(Value::Null)
        }
        "hardware-concurrency" => Ok(hardware_concurrency()),
        "params" => exec_params_new(&mut guard, op_val),
        "params-update" => exec_params_update(&mut guard, op_val),
        "params-unified" => exec_params_unified(&mut guard, op_val),
        "model-user-metadata" => exec_model_user_metadata(op_val, &mut guard, session_dir),
        "common-model" => exec_common_model(&mut guard, op_val, session_dir),
        "common-model-sampler" => exec_common_model_sampler(&mut guard, op_val),
        "vocab" => exec_vocab_meta(&guard.models, op_val),
        "tokenize" => exec_tokenize_op(&guard.contexts, op_val),
        "token-info" => exec_token_info(&guard.contexts, op_val),
        "context" => exec_context_new(&mut guard, op_val, session_dir),
        "context-limits" => {
            let ctx = guard
                .contexts
                .get(&req_handle(op_val, "Context")?)
                .ok_or_else(|| invalid("unknown Context handle"))?;
            Ok(context_limits(ctx))
        }
        "context-graph-reserve" => {
            let ctx = guard
                .contexts
                .get(&req_handle(op_val, "Context")?)
                .ok_or_else(|| invalid("unknown Context handle"))?;
            let tokens = op_val["Tokens"].as_u64().unwrap_or(1);
            Ok(context_graph_reserve(ctx, tokens))
        }
        "context-logits" => {
            let ctx = guard
                .contexts
                .get(&req_handle(op_val, "Context")?)
                .ok_or_else(|| invalid("unknown Context handle"))?;
            let indices = op_val["Indices"]
                .as_array()
                .ok_or_else(|| invalid("context-logits Indices must be array"))?;
            logits_rows_slice(ctx, indices)
        }
        "synchronize" => Ok(Value::Null),
        "batch-new" => exec_batch_new(&mut guard, op_val),
        "batch-one" => exec_batch_one(&mut guard, op_val),
        "batch-add" => exec_batch_add(&mut guard.batches, op_val),
        "batch-clear" => exec_batch_clear(&mut guard.batches, op_val),
        "batch-logit" => exec_batch_logit(&mut guard.batches, op_val),
        "decode" => exec_decode_op(op_val, &mut guard),
        "sampler" => exec_sampler_new(&mut guard, op_val),
        "sampler-accept" => exec_sampler_accept(&mut guard.samplers, op_val),
        "sample" => exec_sample_op(&mut guard, op_val),
        "state-get" => exec_state_get(&mut guard, op_val),
        "state-set" => exec_state_set(&mut guard, op_val),
        "state-save-file" => exec_state_save_file(&guard.contexts, op_val, session_dir),
        "state-load-file" => exec_state_load_file(&mut guard.contexts, op_val, session_dir),
        "memory-clear" => {
            let ctx = guard
                .contexts
                .get_mut(&req_handle(op_val, "Context")?)
                .ok_or_else(|| invalid("unknown Context handle"))?;
            ctx.memory_clear();
            Ok(Value::Null)
        }
        "memory-remove" => {
            let cid = req_handle(op_val, "Context")?;
            let seq = i32::try_from(op_val["Sequence"].as_i64().unwrap_or(0))?;
            let p0 = i32::try_from(op_val["From"].as_i64().unwrap_or(-1))?;
            let p1 = i32::try_from(op_val["To"].as_i64().unwrap_or(-1))?;
            let ctx = guard
                .contexts
                .get_mut(&cid)
                .ok_or_else(|| invalid("unknown Context handle"))?;
            Ok(json!({ "Removed": ctx.memory_remove(seq, p0, p1) }))
        }
        "memory-cp" | "memory-seq-cp" => {
            let cid = req_handle(op_val, "Context")?;
            let src = i32::try_from(op_val["Source"].as_i64().unwrap_or(0))?;
            let dst = i32::try_from(op_val["Target"].as_i64().unwrap_or(1))?;
            let p0 = i32::try_from(op_val["From"].as_i64().unwrap_or(-1))?;
            let p1 = i32::try_from(op_val["To"].as_i64().unwrap_or(-1))?;
            let ctx = guard
                .contexts
                .get_mut(&cid)
                .ok_or_else(|| invalid("unknown Context handle"))?;
            ctx.memory_cp(src, dst, p0, p1);
            Ok(Value::Null)
        }
        "memory-keep" | "memory-seq-keep" => {
            let cid = req_handle(op_val, "Context")?;
            let seq = i32::try_from(op_val["Sequence"].as_i64().unwrap_or(0))?;
            let ctx = guard
                .contexts
                .get_mut(&cid)
                .ok_or_else(|| invalid("unknown Context handle"))?;
            ctx.memory_keep(seq);
            Ok(Value::Null)
        }
        "memory-add" | "memory-seq-pos-add" => {
            let cid = req_handle(op_val, "Context")?;
            let seq = i32::try_from(op_val["Sequence"].as_i64().unwrap_or(0))?;
            let p0 = i32::try_from(op_val["From"].as_i64().unwrap_or(-1))?;
            let p1 = i32::try_from(op_val["To"].as_i64().unwrap_or(-1))?;
            let delta = i32::try_from(op_val["Delta"].as_i64().unwrap_or(0))?;
            let ctx = guard
                .contexts
                .get_mut(&cid)
                .ok_or_else(|| invalid("unknown Context handle"))?;
            ctx.memory_add(seq, p0, p1, delta);
            Ok(Value::Null)
        }
        "memory-pos-min" => {
            let ctx = guard
                .contexts
                .get(&req_handle(op_val, "Context")?)
                .ok_or_else(|| invalid("unknown Context handle"))?;
            let seq = i32::try_from(op_val["Sequence"].as_i64().unwrap_or(0))?;
            Ok(json!({ "Pos": ctx.kv_seq_min(seq) }))
        }
        "memory-pos-max" => {
            let ctx = guard
                .contexts
                .get(&req_handle(op_val, "Context")?)
                .ok_or_else(|| invalid("unknown Context handle"))?;
            let seq = i32::try_from(op_val["Sequence"].as_i64().unwrap_or(0))?;
            Ok(json!({ "Pos": ctx.kv_seq_max(seq) }))
        }
        "checkpoint-update" => {
            let cid = req_handle(op_val, "Context")?;
            let slot = i32::try_from(op_val["Slot"].as_i64().unwrap_or(0))?;
            let seq = i32::try_from(op_val["Sequence"].as_i64().unwrap_or(0))?;
            let ctx = guard
                .contexts
                .get_mut(&cid)
                .ok_or_else(|| invalid("unknown Context handle"))?;
            ctx.checkpoint_update(slot, seq);
            Ok(Value::Null)
        }
        "checkpoint-load" => {
            let cid = req_handle(op_val, "Context")?;
            let slot = i32::try_from(op_val["Slot"].as_i64().unwrap_or(0))?;
            let seq = i32::try_from(op_val["Sequence"].as_i64().unwrap_or(0))?;
            let ctx = guard
                .contexts
                .get_mut(&cid)
                .ok_or_else(|| invalid("unknown Context handle"))?;
            ctx.checkpoint_load(slot, seq)?;
            Ok(Value::Null)
        }
        "kv-center-write" => exec_kv_center_write(op_val, session_dir),
        "kv-center-load" => exec_kv_center_load(op_val, &mut guard.contexts, session_dir),
        "release" => exec_release(&mut guard, op_val),
        _ if op.starts_with("bs-") => {
            exec_backend_sampler_sync(op, op_val, &mut guard, session_dir)
        }
        _ => Err(invalid(format!("unknown session operation {op}"))),
    }
}

fn exec_with_dspark<F>(guard: &mut SessionTables, session_dir: &Path, f: F) -> Result<Value>
where
    F: FnOnce(&mut DsparkEnv<'_>) -> Result<Value>,
{
    let next = &mut guard.next_id;
    let mut alloc = || {
        let id = *next;
        *next += 1;
        id
    };
    let mut env = DsparkEnv {
        models: &mut guard.models,
        contexts: &mut guard.contexts,
        batches: &mut guard.batches,
        samplers: &mut guard.samplers,
        state: &mut guard.dspark,
        env: &mut guard.env,
        next_handle: &mut alloc,
        session_dir,
    };
    f(&mut env)
}

fn req_handle(op: &Value, key: &str) -> Result<u64> {
    op[key]
        .as_u64()
        .ok_or_else(|| invalid(format!("missing handle field {key}")))
}

fn get_model(models: &BTreeMap<u64, Arc<ModelData>>, id: u64) -> Result<Arc<ModelData>> {
    models
        .get(&id)
        .cloned()
        .ok_or_else(|| invalid("unknown Model handle"))
}

fn exec_params_new(guard: &mut SessionTables, op_val: &Value) -> Result<Value> {
    if guard.params.len() >= MAX_PARAMS {
        return Err(invalid("params handle bound exceeded"));
    }
    let parsed = if op_val.get("Initial").is_some() && op_val.get("Arguments").is_some() {
        parse_params(op_val)?
    } else {
        SessionParams::default()
    };
    let id = guard.alloc_id();
    guard.params.insert(id, parsed);
    Ok(json!({ "Params": id }))
}

fn exec_params_update(guard: &mut SessionTables, op_val: &Value) -> Result<Value> {
    let pid = req_handle(op_val, "Params")?;
    let params = guard
        .params
        .get_mut(&pid)
        .ok_or_else(|| invalid("unknown Params handle"))?;
    update_params(params, &op_val["Fields"])?;
    Ok(Value::Null)
}

fn exec_params_unified(guard: &mut SessionTables, op_val: &Value) -> Result<Value> {
    let pid = req_handle(op_val, "Params")?;
    let val = op_val["Value"]
        .as_bool()
        .ok_or_else(|| invalid("params-unified Value must be bool"))?;
    let params = guard
        .params
        .get_mut(&pid)
        .ok_or_else(|| invalid("unknown Params handle"))?;
    params.kv_unified = val;
    Ok(Value::Null)
}

fn exec_model_load<W: Write>(
    req_id: u64,
    op_val: &Value,
    tables: &Arc<Mutex<SessionTables>>,
    wire: &Arc<WireWriter<W>>,
) -> Result<Value> {
    let path = checked_model_path(&op_val["Path"])?;
    let ov = op_val.get("Overrides").and_then(Value::as_object);
    let load_weights = ov.and_then(|o| o.get("LoadMode")).and_then(Value::as_str) != Some("none");
    let head_id = ov.and_then(|o| o.get("HeadSource")).and_then(Value::as_u64);
    let has_progress = op_val["Progress"].as_bool() == Some(true);
    let head_source = {
        let g = tables
            .lock()
            .map_err(|_| invalid("tables mutex poisoned"))?;
        if g.models.len() >= MAX_MODELS {
            return Err(invalid("model handle bound exceeded"));
        }
        let res = head_id.map(|id| get_model(&g.models, id)).transpose()?;
        drop(g);
        res
    };
    let loaded = if has_progress {
        let mut cb = |progress: f32| -> bool {
            let frame = json!({
                "RequestId": req_id,
                "ProgressBits": progress.to_bits(),
            });
            let Ok(reply) = wire.call_client(frame) else {
                return false;
            };
            reply
                .get("Return")
                .and_then(Value::as_bool)
                .unwrap_or(false)
        };
        ModelData::load_from_file(&path, load_weights, head_source, Some(&mut cb))?
    } else {
        ModelData::load_from_file(&path, load_weights, head_source, None)?
    };
    let mut g = tables
        .lock()
        .map_err(|_| invalid("tables mutex poisoned"))?;
    let out = loaded.map_or_else(
        || json!({ "Model": Value::Null, "Cancelled": true }),
        |model| {
            let id = g.alloc_id();
            g.models.insert(id, model);
            json!({ "Model": id, "Cancelled": false })
        },
    );
    drop(g);
    Ok(out)
}

fn exec_model_user_metadata(
    op_val: &Value,
    guard: &mut SessionTables,
    session_dir: &Path,
) -> Result<Value> {
    let raw_file = op_val["MetadataFile"]
        .as_str()
        .ok_or_else(|| invalid("MetadataFile must be string"))?;
    let path = checked_session_file(raw_file, session_dir, true)?;
    let model = ModelData::init_from_user(&path, session_dir, false, None)?;
    let id = guard.alloc_id();
    guard.models.insert(id, model);
    Ok(json!({ "Model": id }))
}

fn exec_model_init_from_user<W: Write>(
    req_id: u64,
    op_val: &Value,
    tables: &Arc<Mutex<SessionTables>>,
    wire: &Arc<WireWriter<W>>,
    session_dir: &Path,
) -> Result<Value> {
    let raw_file = op_val["MetadataFile"]
        .as_str()
        .ok_or_else(|| invalid("MetadataFile missing"))?;
    let path = checked_session_file(raw_file, session_dir, true)?;
    let load_weights = op_val["Weights"].as_bool().unwrap_or(true);
    let mut cb = |spec: &Value| -> Result<Value> {
        let reply = wire.call_client(json!({ "RequestId": req_id, "TensorSpec": spec }))?;
        if let Some(err) = reply.get("Error").and_then(Value::as_str) {
            return Err(invalid(err));
        }
        Ok(reply.get("Result").cloned().unwrap_or(Value::Null))
    };
    let model = ModelData::init_from_user(&path, session_dir, load_weights, Some(&mut cb))?;
    let mut g = tables
        .lock()
        .map_err(|_| invalid("tables mutex poisoned"))?;
    let id = g.alloc_id();
    g.models.insert(id, model);
    drop(g);
    Ok(json!({ "Model": id }))
}

fn exec_common_model(
    guard: &mut SessionTables,
    op_val: &Value,
    session_dir: &Path,
) -> Result<Value> {
    let pid = req_handle(op_val, "Params")?;
    let params = guard
        .params
        .get(&pid)
        .ok_or_else(|| invalid("unknown Params handle"))?
        .clone();
    let model_only = op_val["ModelOnly"].as_bool().unwrap_or(false);
    let path = params
        .model_path
        .as_ref()
        .ok_or_else(|| invalid("Params has no model path"))?;
    let model = ModelData::load_from_file(path, true, None, None)?
        .ok_or_else(|| invalid("unexpected cancellation in common-model"))?;
    let mid = guard.alloc_id();
    guard.models.insert(mid, Arc::clone(&model));
    let cid = if model_only {
        Value::Null
    } else {
        check_context_limits(&guard.contexts)?;
        let mut ctx = build_context(model, &json!({}), session_dir, Some(&params))?
            .ok_or_else(|| invalid("common-model context rejected"))?;
        if let Some(lora_path) = &params.lora_path {
            let adapter = ModelData::load_from_file(lora_path, true, None, None)?
                .ok_or_else(|| invalid("LoRA load cancelled"))?;
            ctx.lora_adapters.push((adapter, 1.0));
        }
        let cid_num = guard.alloc_id();
        guard.contexts.insert(cid_num, ctx);
        json!(cid_num)
    };
    Ok(json!({ "Model": mid, "Context": cid }))
}

fn exec_common_model_sampler(guard: &mut SessionTables, op_val: &Value) -> Result<Value> {
    let _mid = req_handle(op_val, "Model")?;
    let id = guard.alloc_id();
    guard.samplers.insert(id, SamplerNode::Greedy);
    Ok(json!({ "Sampler": id }))
}

fn exec_vocab_meta(models: &BTreeMap<u64, Arc<ModelData>>, op_val: &Value) -> Result<Value> {
    let model = get_model(models, req_handle(op_val, "Model")?)?;
    Ok(json!({
        "Count": model.vocab.count(),
        "Type": vocab_type_from_model(&model.vocab),
        "Bos": model.vocab.bos(),
        "Eos": model.vocab.eos(),
    }))
}

fn exec_tokenize_op(contexts: &BTreeMap<u64, ContextState>, op_val: &Value) -> Result<Value> {
    let ctx = contexts
        .get(&req_handle(op_val, "Context")?)
        .ok_or_else(|| invalid("unknown Context handle"))?;
    let text = op_val["Text"]
        .as_str()
        .ok_or_else(|| invalid("tokenize Text must be string"))?;
    let special = op_val["Special"].as_bool().unwrap_or(false);
    let tokens = ctx.model.vocab.tokenize(text, true, special)?;
    Ok(json!({ "Tokens": tokens }))
}

fn exec_token_info(contexts: &BTreeMap<u64, ContextState>, op_val: &Value) -> Result<Value> {
    let ctx = contexts
        .get(&req_handle(op_val, "Context")?)
        .ok_or_else(|| invalid("unknown Context handle"))?;
    let tok = i32::try_from(
        op_val["Token"]
            .as_i64()
            .ok_or_else(|| invalid("token-info Token must be int"))?,
    )?;
    let special = op_val["Special"].as_bool().unwrap_or(false);
    let piece = ctx.model.vocab.token_to_piece(tok, 0, special)?;
    let eog = ctx.model.vocab.is_eog(tok);
    Ok(json!({
        "PieceHex": hex::encode(&piece),
        "Eog": eog,
        "Piece": String::from_utf8_lossy(&piece),
    }))
}

fn exec_context_new(
    guard: &mut SessionTables,
    op_val: &Value,
    session_dir: &Path,
) -> Result<Value> {
    check_context_limits(&guard.contexts)?;
    let model = get_model(&guard.models, req_handle(op_val, "Model")?)?;
    let params_opt = op_val["Params"]
        .as_u64()
        .and_then(|pid| guard.params.get(&pid));
    let Some(ctx) = build_context(model, op_val, session_dir, params_opt)? else {
        return Ok(json!({ "Context": Value::Null }));
    };
    let id = guard.alloc_id();
    guard.contexts.insert(id, ctx);
    Ok(json!({ "Context": id }))
}

fn exec_batch_new(guard: &mut SessionTables, op_val: &Value) -> Result<Value> {
    check_batch_limits(&guard.batches)?;
    let cap = usize::try_from(op_val["Capacity"].as_u64().unwrap_or(512))?;
    let seqs = usize::try_from(op_val["Sequences"].as_u64().unwrap_or(1))?;
    let id = guard.alloc_id();
    guard.batches.insert(id, BatchData::new(cap, seqs));
    Ok(json!({ "Batch": id }))
}

fn exec_batch_one(guard: &mut SessionTables, op_val: &Value) -> Result<Value> {
    let tok_arr = op_val["Tokens"]
        .as_array()
        .ok_or_else(|| invalid("batch-one Tokens must be array"))?;
    let pos_arr = op_val.get("Positions").and_then(Value::as_array);
    let batch_id = if let Some(existing) = op_val["Batch"].as_u64() {
        let batch = guard
            .batches
            .get_mut(&existing)
            .ok_or_else(|| invalid("unknown Batch handle"))?;
        batch.rows.clear();
        existing
    } else {
        check_batch_limits(&guard.batches)?;
        let id = guard.alloc_id();
        guard
            .batches
            .insert(id, BatchData::new(tok_arr.len().max(1), 1));
        id
    };
    let batch = guard
        .batches
        .get_mut(&batch_id)
        .ok_or_else(|| invalid("batch missing"))?;
    for (i, t) in tok_arr.iter().enumerate() {
        let token = i32::try_from(t.as_i64().ok_or_else(|| invalid("batch token int"))?)?;
        let pos = if let Some(arr) = pos_arr {
            i32::try_from(
                arr.get(i)
                    .and_then(Value::as_i64)
                    .ok_or_else(|| invalid("batch position int"))?,
            )?
        } else {
            i32::try_from(i)?
        };
        batch.rows.push(BatchRow {
            token,
            pos,
            seq_ids: vec![0],
            logits: i + 1 == tok_arr.len(),
        });
    }
    Ok(json!({ "Batch": batch_id }))
}

fn exec_batch_add(batches: &mut BTreeMap<u64, BatchData>, op_val: &Value) -> Result<Value> {
    let bid = req_handle(op_val, "Batch")?;
    let rows = op_val["Rows"]
        .as_array()
        .ok_or_else(|| invalid("batch-add Rows must be array"))?;
    let batch = batches
        .get_mut(&bid)
        .ok_or_else(|| invalid("unknown Batch handle"))?;
    if batch.rows.len() + rows.len() > batch.capacity {
        return Err(invalid("batch-add exceeds batch capacity"));
    }
    for row in rows {
        let token = i32::try_from(row["Token"].as_i64().ok_or_else(|| invalid("Token int"))?)?;
        let pos = i32::try_from(
            row["Position"]
                .as_i64()
                .ok_or_else(|| invalid("Position int"))?,
        )?;
        let logits = row["Logits"].as_bool().unwrap_or(false);
        let seq_arr = row["Sequences"]
            .as_array()
            .ok_or_else(|| invalid("Sequences array"))?;
        let mut seq_ids = Vec::with_capacity(seq_arr.len());
        for s in seq_arr {
            seq_ids.push(i32::try_from(
                s.as_i64().ok_or_else(|| invalid("Seq int"))?,
            )?);
        }
        batch.rows.push(BatchRow {
            token,
            pos,
            seq_ids,
            logits,
        });
    }
    Ok(Value::Null)
}

fn exec_batch_clear(batches: &mut BTreeMap<u64, BatchData>, op_val: &Value) -> Result<Value> {
    let batch = batches
        .get_mut(&req_handle(op_val, "Batch")?)
        .ok_or_else(|| invalid("unknown Batch handle"))?;
    batch.rows.clear();
    Ok(Value::Null)
}

fn exec_batch_logit(batches: &mut BTreeMap<u64, BatchData>, op_val: &Value) -> Result<Value> {
    let batch = batches
        .get_mut(&req_handle(op_val, "Batch")?)
        .ok_or_else(|| invalid("unknown Batch handle"))?;
    let idx = usize::try_from(req_handle(op_val, "Index")?)?;
    let val = op_val["Value"].as_bool().unwrap_or(true);
    let row = batch
        .rows
        .get_mut(idx)
        .ok_or_else(|| invalid("batch-logit index out of bounds"))?;
    row.logits = val;
    Ok(Value::Null)
}

fn exec_decode_op(op_val: &Value, guard: &mut SessionTables) -> Result<Value> {
    let cid = req_handle(op_val, "Context")?;
    let bid = req_handle(op_val, "Batch")?;
    let batch = guard
        .batches
        .get(&bid)
        .ok_or_else(|| invalid("unknown Batch handle"))?
        .clone();
    let ctx = guard
        .contexts
        .get_mut(&cid)
        .ok_or_else(|| invalid("unknown Context handle"))?;
    run_decode(ctx, &batch, &mut guard.dspark.traces, &guard.env)?;
    Ok(json!({ "Code": 0 }))
}

fn exec_sampler_new(guard: &mut SessionTables, op_val: &Value) -> Result<Value> {
    let kind = op_val["Kind"]
        .as_str()
        .ok_or_else(|| invalid("sampler Kind missing"))?;
    let node = match kind {
        "dist" => {
            let seed = u32::try_from(op_val["Seed"].as_u64().unwrap_or(0))?;
            SamplerNode::Dist(super::sampler::Mt19937::new(seed))
        }
        "greedy" | "common" => {
            if let Some(pid) = op_val["Params"].as_u64()
                && let Some(p) = guard.params.get(&pid)
                && p.temp > 0.0
            {
                SamplerNode::Chain(vec![
                    SamplerNode::Temp(p.temp),
                    SamplerNode::Dist(super::sampler::Mt19937::new(p.seed)),
                ])
            } else {
                SamplerNode::Greedy
            }
        }
        _ => return Err(invalid(format!("unknown sampler kind {kind}"))),
    };
    let id = guard.alloc_id();
    guard.samplers.insert(id, node);
    Ok(json!({ "Sampler": id }))
}

fn exec_sampler_accept(samplers: &mut BTreeMap<u64, SamplerNode>, op_val: &Value) -> Result<Value> {
    let sid = req_handle(op_val, "Sampler")?;
    let tok = i32::try_from(op_val["Token"].as_i64().unwrap_or(0))?;
    if let Some(s) = samplers.get_mut(&sid) {
        s.accept(tok);
    }
    Ok(Value::Null)
}

fn exec_sample_op(guard: &mut SessionTables, op_val: &Value) -> Result<Value> {
    let sid = req_handle(op_val, "Sampler")?;
    let cid = req_handle(op_val, "Context")?;
    let idx = op_val["Index"].as_i64().unwrap_or(-1);
    let ctx = guard
        .contexts
        .get(&cid)
        .ok_or_else(|| invalid("unknown Context handle"))?;
    let logits = super::dspark::resolve_logit_row(&ctx.output_logits, idx)?;
    let mut sampler = guard
        .samplers
        .get(&sid)
        .ok_or_else(|| invalid("unknown Sampler handle"))?
        .clone();
    let (tok, _) = sampler.sample_logits(logits, &mut None)?;
    guard.samplers.insert(sid, sampler);
    Ok(json!({ "Token": tok }))
}

fn exec_state_get(guard: &mut SessionTables, op_val: &Value) -> Result<Value> {
    let cid = req_handle(op_val, "Context")?;
    let seq = i32::try_from(op_val["Sequence"].as_i64().unwrap_or(0))?;
    let ret_data = op_val["ReturnData"].as_bool().unwrap_or(false);
    let ctx = guard
        .contexts
        .get(&cid)
        .ok_or_else(|| invalid("unknown Context handle"))?;
    let bytes = ctx.encode_state_bytes(seq)?;
    if guard.blobs.len() >= MAX_BLOBS {
        return Err(invalid("state blob handle bound exceeded"));
    }
    let bid = guard.alloc_id();
    let size = bytes.len();
    let hex_str = if ret_data {
        Value::String(hex::encode(&bytes))
    } else {
        Value::Null
    };
    guard.blobs.insert(bid, bytes);
    Ok(json!({
        "Blob": bid,
        "Size": size,
        "Copied": size,
        "DataHex": hex_str,
    }))
}

fn exec_state_set(guard: &mut SessionTables, op_val: &Value) -> Result<Value> {
    let cid = req_handle(op_val, "Context")?;
    let bid = req_handle(op_val, "Blob")?;
    let seq = i32::try_from(op_val["Sequence"].as_i64().unwrap_or(0))?;
    let data = guard
        .blobs
        .get(&bid)
        .cloned()
        .ok_or_else(|| invalid("unknown Blob handle"))?;
    let ctx = guard
        .contexts
        .get_mut(&cid)
        .ok_or_else(|| invalid("unknown Context handle"))?;
    ctx.decode_state_bytes(seq, &data)?;
    Ok(json!({ "Read": data.len() }))
}

fn exec_state_save_file(
    contexts: &BTreeMap<u64, ContextState>,
    op_val: &Value,
    session_dir: &Path,
) -> Result<Value> {
    let cid = req_handle(op_val, "Context")?;
    let name = validate_state_name(&op_val["Name"])?;
    let toks_arr = op_val["Tokens"]
        .as_array()
        .ok_or_else(|| invalid("state-save-file Tokens array"))?;
    let ctx = contexts
        .get(&cid)
        .ok_or_else(|| invalid("unknown Context handle"))?;
    let state_bytes = ctx.encode_state_bytes(0)?;
    let mut file_bytes = Vec::new();
    file_bytes.extend_from_slice(&u32::try_from(toks_arr.len())?.to_le_bytes());
    for t in toks_arr {
        let tok = i32::try_from(t.as_i64().ok_or_else(|| invalid("token int"))?)?;
        file_bytes.extend_from_slice(&tok.to_le_bytes());
    }
    file_bytes.extend_from_slice(&state_bytes);
    let path = session_dir.join(name);
    fs::write(path, file_bytes)?;
    Ok(json!({ "Saved": true }))
}

fn exec_state_load_file(
    contexts: &mut BTreeMap<u64, ContextState>,
    op_val: &Value,
    session_dir: &Path,
) -> Result<Value> {
    let cid = req_handle(op_val, "Context")?;
    let name = validate_state_name(&op_val["Name"])?;
    let cap = usize::try_from(op_val["Capacity"].as_u64().unwrap_or(4096))?;
    let raw = fs::read(session_dir.join(name))?;
    if raw.len() < 4 {
        return Err(invalid("state file truncated"));
    }
    let count = usize::try_from(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))?;
    if count > cap || raw.len() < 4 + count * 4 {
        return Err(invalid("state file token count exceeds capacity"));
    }
    let mut tokens = Vec::with_capacity(count);
    for chunk in raw[4..4 + count * 4].as_chunks::<4>().0 {
        tokens.push(i32::from_le_bytes(*chunk));
    }
    let ctx = contexts
        .get_mut(&cid)
        .ok_or_else(|| invalid("unknown Context handle"))?;
    ctx.decode_state_bytes(0, &raw[4 + count * 4..])?;
    Ok(json!({
        "Loaded": true,
        "Count": tokens.len(),
        "Tokens": tokens,
    }))
}

fn exec_kv_center_write(op_val: &Value, session_dir: &Path) -> Result<Value> {
    let name = validate_state_name(&op_val["Name"])?;
    let rotated = op_val["Rotated"]
        .as_bool()
        .ok_or_else(|| invalid("kv-center-write Rotated bool"))?;
    let layers = op_val["Layers"]
        .as_array()
        .ok_or_else(|| invalid("kv-center-write Layers array"))?;
    let mut kvs = BTreeMap::new();
    kvs.insert("kv_mean_center.rotated".into(), GgufValue::Bool(rotated));
    let kv_order = vec!["kv_mean_center.rotated".into()];
    let mut tensors = BTreeMap::new();
    let mut tensor_order = Vec::with_capacity(layers.len());
    for layer in layers {
        let idx = layer["Index"]
            .as_u64()
            .ok_or_else(|| invalid("layer index"))?;
        let bytes = hex::decode(
            layer["BiasHex"]
                .as_str()
                .ok_or_else(|| invalid("BiasHex missing"))?,
        )?;
        let n_elems = i64::try_from(bytes.len() / 4)?;
        let tname = format!("blk.{idx}.attn_k_bias");
        tensor_order.push(tname.clone());
        tensors.insert(
            tname.clone(),
            TensorEntry {
                name: tname,
                kind: 0,
                ne: [n_elems, 1, 1, 1],
                raw: bytes,
            },
        );
    }
    let model = ModelData {
        kvs,
        kv_order,
        tensors,
        tensor_order,
        vocab: Arc::new(ModelVocab::Synthetic {
            count: 32,
            bos: 1,
            eos: 2,
        }),
        vocab_size: 32,
        n_ctx_train: 512,
        n_embd: 64,
        n_layer: layers.len(),
        n_ff: 128,
        n_head: 2,
        n_head_kv: 2,
        rms_eps: 1e-5,
        rope_freq_base: 10_000.0,
        is_recurrent: false,
        is_hybrid: false,
        is_dspark: false,
        n_embd_cap: 0,
        block_size: 1,
        mask_token_id: 0,
        markov_rank: 0,
        graph_corrected: false,
        dspark_target_layers: Vec::new(),
        output_identity: 0,
        output_buffer: Arc::new(Vec::new()),
        head_source: None,
    };
    let path = session_dir.join(name);
    let mut writer = BufWriter::new(File::create(path)?);
    model.save_to_writer(&mut writer)?;
    Ok(json!({ "Written": true }))
}

fn exec_kv_center_load(
    op_val: &Value,
    contexts: &mut BTreeMap<u64, ContextState>,
    session_dir: &Path,
) -> Result<Value> {
    let cid = req_handle(op_val, "Context")?;
    let name = validate_state_name(&op_val["Name"])?;
    let req_q4 = op_val["RequireQ4"].as_bool().unwrap_or(false);
    let path = session_dir.join(name);
    if !path.is_file() {
        return Ok(json!({ "Loaded": false, "CacheExists": false }));
    }
    let ctx = contexts
        .get_mut(&cid)
        .ok_or_else(|| invalid("unknown Context handle"))?;
    if req_q4 && ctx.type_k != 2 {
        return Ok(json!({ "Loaded": false, "CacheExists": true }));
    }
    let Some(layers) = load_bias_gguf(&path, ctx.model.n_layer, req_q4)? else {
        return Ok(json!({ "Loaded": false, "CacheExists": true }));
    };
    ctx.kv_bias = Some(layers);
    Ok(json!({ "Loaded": true, "CacheExists": true }))
}

fn exec_release(guard: &mut SessionTables, op_val: &Value) -> Result<Value> {
    let kind = op_val["Kind"]
        .as_str()
        .ok_or_else(|| invalid("release Kind missing"))?;
    let id = req_handle(op_val, "Handle")?;
    match kind {
        "params" => {
            guard.params.remove(&id);
        }
        "model" => {
            guard.models.remove(&id);
        }
        "context" => {
            guard.contexts.remove(&id);
        }
        "batch" => {
            guard.batches.remove(&id);
        }
        "sampler" => {
            guard.samplers.remove(&id);
        }
        "blob" => {
            guard.blobs.remove(&id);
        }
        "trace" => {
            guard.dspark.traces.remove(&id);
        }
        "speculative" => {
            guard.dspark.specs.remove(&id);
        }
        _ => return Err(invalid(format!("unknown release kind {kind}"))),
    }
    Ok(Value::Null)
}

fn exec_backend_sampler_cb<W: Write>(
    req_id: u64,
    op: &str,
    op_val: &Value,
    tables: &Arc<Mutex<SessionTables>>,
    wire: &Arc<WireWriter<W>>,
) -> Result<Value> {
    let sid = req_handle(op_val, "Sampler")?;
    let mut cb = |payload: &Value| -> Result<Value> {
        let reply = wire.call_client(json!({ "RequestId": req_id, "SamplerEvent": payload }))?;
        if let Some(err) = reply.get("Error").and_then(Value::as_str) {
            return Err(invalid(err));
        }
        Ok(reply.get("Result").cloned().unwrap_or(Value::Null))
    };
    match op {
        "bs-sampler-backend-init" => {
            let outputs = u32::try_from(op_val["OutputsMaxPerSequence"].as_u64().unwrap_or(1))?;
            let sampler = {
                let g = tables
                    .lock()
                    .map_err(|_| invalid("tables mutex poisoned"))?;
                g.backend_samplers.get_sampler(sid)?.clone()
            };
            let accepted = sampler.notify_backend_init(outputs, &mut cb)?;
            Ok(json!({ "Accepted": accepted }))
        }
        "bs-candidates-apply" => {
            let cid = req_handle(op_val, "Candidates")?;
            let mut g = tables
                .lock()
                .map_err(|_| invalid("tables mutex poisoned"))?;
            let has_obs = g.backend_samplers.get_sampler(sid)?.has_observer();
            if has_obs {
                let mut cb_opt: Option<SamplerEventCb<'_>> = Some(&mut cb);
                g.backend_samplers.candidates_apply(sid, cid, &mut cb_opt)
            } else {
                g.backend_samplers.candidates_apply(sid, cid, &mut None)
            }
        }
        _ => Err(invalid(format!("unexpected callback bs op {op}"))),
    }
}

fn exec_backend_sampler_sync(
    op: &str,
    op_val: &Value,
    guard: &mut SessionTables,
    session_dir: &Path,
) -> Result<Value> {
    match op {
        "bs-init-sampler" => {
            let typ = op_val["Type"]
                .as_str()
                .ok_or_else(|| invalid("bs-init-sampler Type missing"))?;
            guard.backend_samplers.init_sampler(typ, &op_val["Params"])
        }
        "bs-chain-add" => guard
            .backend_samplers
            .chain_add(req_handle(op_val, "Chain")?, req_handle(op_val, "Child")?),
        "bs-sampler-clone" => guard
            .backend_samplers
            .clone_sampler(req_handle(op_val, "Sampler")?),
        "bs-sampler-copy" => guard.backend_samplers.copy_sampler(
            req_handle(op_val, "Destination")?,
            req_handle(op_val, "Source")?,
        ),
        "bs-sampler-accept" => {
            let sid = req_handle(op_val, "Sampler")?;
            let tok = i32::try_from(op_val["Token"].as_i64().unwrap_or(0))?;
            guard.backend_samplers.get_sampler_mut(sid)?.accept(tok);
            Ok(Value::Null)
        }
        "bs-sampler-free" => {
            let _ = req_handle(op_val, "Sampler")?;
            Ok(Value::Null)
        }
        "bs-candidates-new" => guard.backend_samplers.candidates_new(op_val, session_dir),
        "bs-candidates-export" => guard
            .backend_samplers
            .candidates_export(op_val, session_dir),
        "bs-candidates-free" => guard
            .backend_samplers
            .candidates_free(req_handle(op_val, "Candidates")?),
        "bs-tokenize" => {
            let m = get_model(&guard.models, req_handle(op_val, "Model")?)?;
            bs_tokenize(op_val, &m)
        }
        "bs-piece" => {
            let m = get_model(&guard.models, req_handle(op_val, "Model")?)?;
            bs_piece(op_val, &m)
        }
        _ => Err(invalid(format!("unknown backend sampler op {op}"))),
    }
}
