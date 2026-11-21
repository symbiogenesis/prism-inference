//! Clean-room safe Rust reference provider for persistent codec buffers and GGUF model tensor sessions.
//!
//! Independently implements `qb-*` buffer management, thread-safe concurrent
//! `qb-trace` / `qb-checkpoint` execution, GGUF v2/v3 tensor metadata & F32/F16
//! reading, and codec dispatch over `reference_codec` primitives without calling
//! `bonsai_prism_runtime`, any native bridge, or reading expected-answer fixtures.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{OsStr, OsString};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

use super::reference_codec::{
    TYPE_COUNT, Traits, decode_row, f16_to_f32, quantize_row, traits_of, vec_dot_row,
};
use super::{Result, invalid};

/// Upstream conformance route identifiers backed by this persistent codec buffer provider.
pub(super) const IDS: [&str; 2] = [
    "tests/test-quantize-perf.cpp",
    "tests/test-quantize-stats.cpp",
];

const PERF_SHA256: &str = "09de1812b11836bd899a63dfa6b06927a2d0a012edc1e702a33fb51c95d63563";
const STATS_SHA256: &str = "1866b300a8966c3797740a5b4543236d92d1c7b8f2ca67829f453ffcd5b9707b";

const MAX_BUFFER: usize = 512 * 1024 * 1024;
const MAX_TOTAL_ALLOCATED: usize = 2 * 1024 * 1024 * 1024;
const MAX_BUFFERS: usize = 32;
const MAX_HANDLES: u64 = 1_000_000;
const MAX_MODEL_HANDLES: usize = 512;
const MAX_ALIGNMENT: usize = 4096;
const MAX_CALL_COUNT: usize = 134_217_728;
const MAX_TEXT_LEN: usize = 32_768;
const MAX_FILE_OFFSET: u64 = 4 * 1024 * 1024 * 1024;
const MAX_ACTION_FILE_BYTES: u64 = 128 * 1024 * 1024;
const MAX_TRACE_ACTIONS: usize = 524_288;
const MAX_STATELESS_OPS: usize = 128;
const MAX_JSON_DEPTH: usize = 160;
const MAX_CHECKPOINT_BYTES: usize = 16 * 1024 * 1024;
const MAX_CHECKPOINT_TIMEOUT_MS: usize = 3_600_000;
const MAX_TENSORS: usize = 65_536;
const MAX_KV_ENTRIES: u64 = 1_048_576;
const MAX_GGUF_STRING: u64 = 1_048_576;
const MAX_GGUF_ALIGNMENT: u32 = 1_048_576;
const MAX_WORKERS: u32 = 16;

fn require(condition: bool, message: &'static str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(invalid(message))
    }
}

fn exact_fields<'a>(
    value: &'a Value,
    names: &[&str],
) -> Result<&'a serde_json::Map<String, Value>> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid("codec object fields mismatch"))?;
    require(
        object.len() == names.len() && names.iter().all(|name| object.contains_key(*name)),
        "codec object fields mismatch",
    )?;
    Ok(object)
}

fn integer(value: &Value, maximum: usize) -> Result<usize> {
    let number = value
        .as_u64()
        .ok_or_else(|| invalid("codec integer must be nonnegative"))?;
    let converted = usize::try_from(number)?;
    require(converted <= maximum, "codec integer exceeds bound")?;
    Ok(converted)
}

fn integer_u64(value: &Value, maximum: u64) -> Result<u64> {
    let number = value
        .as_u64()
        .ok_or_else(|| invalid("codec integer must be nonnegative"))?;
    require(number <= maximum, "codec integer exceeds bound")?;
    Ok(number)
}

fn text_bound(value: &Value, maximum: usize) -> Result<&str> {
    let s = value
        .as_str()
        .ok_or_else(|| invalid("codec buffer text required"))?;
    require(
        s.len() <= maximum && !s.contains('\0'),
        "codec buffer text exceeds bound",
    )?;
    Ok(s)
}

fn text(value: &Value) -> Result<&str> {
    text_bound(value, MAX_TEXT_LEN)
}

fn boolean(value: &Value, message: &'static str) -> Result<bool> {
    value.as_bool().ok_or_else(|| invalid(message))
}

fn check_json_depth(value: &Value, max_depth: usize) -> Result<()> {
    let mut stack = vec![(value, 0_usize)];
    while let Some((current, depth)) = stack.pop() {
        require(depth <= max_depth, "codec action JSON depth exceeds bound")?;
        match current {
            Value::Array(items) => {
                for item in items {
                    stack.push((item, depth + 1));
                }
            }
            Value::Object(map) => {
                for item in map.values() {
                    stack.push((item, depth + 1));
                }
            }
            Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
        }
    }
    Ok(())
}

fn row_bytes(traits: Traits, count: usize) -> Result<usize> {
    require(
        traits.block_size > 0 && count.is_multiple_of(traits.block_size),
        "codec element count violates production block precondition",
    )?;
    let bytes = (count / traits.block_size)
        .checked_mul(traits.type_size)
        .ok_or_else(|| invalid("codec row exceeds buffer bound"))?;
    require(bytes <= MAX_BUFFER, "codec row exceeds buffer bound")?;
    Ok(bytes)
}

fn buffer_ref(value: &Value) -> Result<(u64, usize)> {
    exact_fields(value, &["Buffer", "Offset"])?;
    let id = integer_u64(&value["Buffer"], MAX_HANDLES)?;
    let offset = integer(&value["Offset"], MAX_BUFFER)?;
    Ok((id, offset))
}

struct BufferEntry {
    bytes: usize,
    data: Arc<RwLock<Vec<u8>>>,
}

struct BufferTable {
    entries: BTreeMap<u64, BufferEntry>,
    next_id: u64,
    allocated: usize,
}

struct TensorRecord {
    name: String,
    type_id: u32,
    ne: [u64; 4],
    nb: [u64; 4],
    elements: u64,
    bytes: u64,
    type_size: u64,
    block_size: u64,
    file_offset: u64,
}

struct LoadedModel {
    owner_worker: u32,
    file: Mutex<File>,
    tensors: Vec<TensorRecord>,
    by_name: BTreeMap<String, usize>,
}

struct ContextRecord {
    owner_worker: u32,
    _model_id: u64,
}

struct ModelTable {
    models: BTreeMap<u64, Arc<LoadedModel>>,
    contexts: BTreeMap<u64, ContextRecord>,
    next_id: u64,
}

/// Thread-safe persistent codec buffer, GGUF tensor model, and checkpoint trace session.
pub(super) struct Session {
    canonical_session_dir: PathBuf,
    started: Instant,
    buffers: Mutex<BufferTable>,
    models: Mutex<ModelTable>,
    requests: AtomicU64,
    closed: Mutex<bool>,
}

impl Session {
    /// Creates a new isolated session scoped to `session_dir`.
    pub(super) fn new(session_dir: &Path) -> Result<Self> {
        let canonical_session_dir = fs::canonicalize(session_dir)?;
        require(
            canonical_session_dir.is_dir(),
            "codec session directory must exist",
        )?;
        Ok(Self {
            canonical_session_dir,
            started: Instant::now(),
            buffers: Mutex::new(BufferTable {
                entries: BTreeMap::new(),
                next_id: 1,
                allocated: 0,
            }),
            models: Mutex::new(ModelTable {
                models: BTreeMap::new(),
                contexts: BTreeMap::new(),
                next_id: 1,
            }),
            requests: AtomicU64::new(0),
            closed: Mutex::new(false),
        })
    }

    /// Marks the session closed and verifies no unreleased handles remain.
    pub(super) fn finish(&self) -> Result<u64> {
        {
            let mut closed = self
                .closed
                .lock()
                .map_err(|_| invalid("codec session closed lock poisoned"))?;
            require(!*closed, "codec session already finished")?;
            *closed = true;
        }
        let open_buffers = {
            let buffers = self
                .buffers
                .lock()
                .map_err(|_| invalid("codec buffer lock poisoned"))?;
            !buffers.entries.is_empty()
        };
        let open_models = {
            let models = self
                .models
                .lock()
                .map_err(|_| invalid("codec model lock poisoned"))?;
            !models.models.is_empty() || !models.contexts.is_empty()
        };
        require(
            !open_buffers && !open_models,
            "codec session finished with live buffer or model handles",
        )?;
        Ok(self.requests.load(Ordering::Relaxed))
    }

    /// Validates and executes one session operation on `worker` (`0..16`).
    pub(super) fn execute(&self, worker: u32, operation: &Value) -> Result<Value> {
        require(worker < MAX_WORKERS, "codec worker is outside range")?;
        {
            let closed = self
                .closed
                .lock()
                .map_err(|_| invalid("codec session closed lock poisoned"))?;
            require(!*closed, "codec session is closed")?;
        }
        self.requests.fetch_add(1, Ordering::Relaxed);
        self.dispatch(worker, operation, true)
    }

    fn dispatch(&self, worker: u32, operation: &Value, top_level: bool) -> Result<Value> {
        let object = operation
            .as_object()
            .ok_or_else(|| invalid("codec operation must be an object"))?;
        let op = object
            .get("Op")
            .ok_or_else(|| invalid("codec operation Op required"))?;
        let op_name = text(op)?;

        if !top_level {
            let allowed = matches!(
                op_name,
                "qb-call" | "qb-clock" | "qb-init-type" | "qb-time-init" | "qb-model-copy-f32"
            ) || op_name.starts_with("qb-buffer-");
            require(allowed, "codec trace action outside primitive allowlist")?;
        }

        match op_name {
            "qb-inventory" => {
                exact_fields(operation, &["Op"])?;
                Ok(inventory_json())
            }
            "qb-time-init" => {
                exact_fields(operation, &["Op"])?;
                Ok(Value::Null)
            }
            "qb-init-type" => {
                exact_fields(operation, &["Op", "Type"])?;
                let _ = integer(&operation["Type"], TYPE_COUNT - 1)?;
                Ok(Value::Null)
            }
            "qb-buffer-new" => self.op_buffer_new(worker, operation),
            "qb-buffer-free" => self.op_buffer_free(worker, operation),
            "qb-buffer-load" | "qb-buffer-store" => self.op_buffer_io(op_name, operation),
            "qb-call" => self.op_call(operation),
            "qb-clock" => self.op_clock(operation),
            "qb-trace" => self.op_trace(worker, operation),
            "qb-model-tensors" => self.op_model_tensors(operation),
            "qb-model-copy-f32" => self.op_model_copy_f32(operation),
            "model-load" => self.op_model_load(worker, operation),
            "context" => self.op_context(worker, operation),
            "hardware-concurrency" => {
                exact_fields(operation, &["Op"])?;
                let count = thread::available_parallelism()
                    .map_or(1_usize, std::num::NonZeroUsize::get)
                    .min(1_048_576);
                Ok(json!({"Count": u32::try_from(count)?}))
            }
            "release" => self.op_release(worker, operation),
            _ => Err(invalid("unknown codec buffer operation")),
        }
    }

    fn session_path(&self, value: &Value) -> Result<PathBuf> {
        let raw = text(value)?;
        let path = Path::new(raw);
        require(path.is_absolute(), "codec buffer path must be absolute")?;
        let resolved = weakly_canonical(path)?;
        let relative = resolved
            .strip_prefix(&self.canonical_session_dir)
            .map_err(|_| invalid("codec buffer path outside session"))?;
        require(
            !relative.as_os_str().is_empty(),
            "codec buffer path outside session",
        )?;
        Ok(resolved)
    }

    fn get_buffer(&self, id: u64) -> Result<Arc<RwLock<Vec<u8>>>> {
        let table = self
            .buffers
            .lock()
            .map_err(|_| invalid("codec buffer lock poisoned"))?;
        table
            .entries
            .get(&id)
            .map(|entry| Arc::clone(&entry.data))
            .ok_or_else(|| invalid("unknown codec buffer handle"))
    }

    fn get_model(&self, id: u64) -> Result<Arc<LoadedModel>> {
        let table = self
            .models
            .lock()
            .map_err(|_| invalid("codec model lock poisoned"))?;
        table
            .models
            .get(&id)
            .map(Arc::clone)
            .ok_or_else(|| invalid("unknown model session handle"))
    }

    fn op_buffer_new(&self, worker: u32, operation: &Value) -> Result<Value> {
        exact_fields(operation, &["Op", "Bytes", "Alignment", "Offset"])?;
        require(worker == 0, "codec allocation requires worker zero")?;
        let bytes = integer(&operation["Bytes"], MAX_BUFFER)?;
        let alignment = integer(&operation["Alignment"], MAX_ALIGNMENT)?;
        let offset = integer(&operation["Offset"], MAX_ALIGNMENT)?;
        require(
            alignment > 0 && alignment.is_power_of_two() && offset <= alignment,
            "invalid codec buffer alignment/offset",
        )?;
        let allocation = Arc::new(RwLock::new(vec![0_u8; bytes]));
        let id = {
            let mut table = self
                .buffers
                .lock()
                .map_err(|_| invalid("codec buffer lock poisoned"))?;
            require(
                table.entries.len() < MAX_BUFFERS
                    && table.next_id < MAX_HANDLES
                    && bytes <= MAX_TOTAL_ALLOCATED - table.allocated,
                "codec aggregate buffer bound",
            )?;
            let id = table.next_id;
            table.next_id += 1;
            table.allocated += bytes;
            table.entries.insert(
                id,
                BufferEntry {
                    bytes,
                    data: allocation,
                },
            );
            id
        };
        Ok(json!({"Buffer": id}))
    }

    fn op_buffer_free(&self, worker: u32, operation: &Value) -> Result<Value> {
        exact_fields(operation, &["Op", "Buffer"])?;
        require(worker == 0, "codec release requires worker zero")?;
        let id = integer_u64(&operation["Buffer"], MAX_HANDLES)?;
        let removed = {
            let mut table = self
                .buffers
                .lock()
                .map_err(|_| invalid("codec buffer lock poisoned"))?;
            let entry = table
                .entries
                .remove(&id)
                .ok_or_else(|| invalid("unknown codec release handle"))?;
            table.allocated -= entry.bytes;
            entry
        };
        drop(removed);
        Ok(Value::Null)
    }

    fn op_buffer_io(&self, op_name: &str, operation: &Value) -> Result<Value> {
        exact_fields(
            operation,
            &[
                "Op",
                "Buffer",
                "Offset",
                "File",
                "FileOffset",
                "Bytes",
                "Truncate",
            ],
        )?;
        let truncate = boolean(
            &operation["Truncate"],
            "codec file Truncate must be boolean",
        )?;
        let buf_id = integer_u64(&operation["Buffer"], MAX_HANDLES)?;
        let buf = self.get_buffer(buf_id)?;
        let count = integer(&operation["Bytes"], MAX_BUFFER)?;
        let offset = integer(&operation["Offset"], MAX_BUFFER)?;
        let filename = self.session_path(&operation["File"])?;
        let file_offset = integer_u64(&operation["FileOffset"], MAX_FILE_OFFSET)?;

        if op_name == "qb-buffer-load" {
            require(!truncate, "codec input file extent differs")?;
            {
                let guard = buf
                    .read()
                    .map_err(|_| invalid("codec buffer rwlock poisoned"))?;
                require(
                    offset <= guard.len() && count <= guard.len() - offset,
                    "codec buffer access exceeds extent",
                )?;
            }
            let meta = fs::metadata(&filename)?;
            require(meta.is_file(), "codec input file extent differs")?;
            let file_len = meta.len();
            let count_u64 = u64::try_from(count)?;
            require(
                file_offset <= file_len && count_u64 <= file_len - file_offset,
                "codec input file extent differs",
            )?;
            let mut staged = vec![0_u8; count];
            if count > 0 {
                let mut file = File::open(&filename)?;
                file.seek(SeekFrom::Start(file_offset))?;
                file.read_exact(&mut staged)?;
            }
            let mut guard = buf
                .write()
                .map_err(|_| invalid("codec buffer rwlock poisoned"))?;
            require(
                offset <= guard.len() && count <= guard.len() - offset,
                "codec buffer access exceeds extent",
            )?;
            guard[offset..offset + count].copy_from_slice(&staged);
        } else {
            require(
                !truncate || file_offset == 0,
                "codec truncate requires file offset zero",
            )?;
            let staged = {
                let guard = buf
                    .read()
                    .map_err(|_| invalid("codec buffer rwlock poisoned"))?;
                require(
                    offset <= guard.len() && count <= guard.len() - offset,
                    "codec buffer access exceeds extent",
                )?;
                guard[offset..offset + count].to_vec()
            };
            let mut file = if truncate {
                OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(&filename)?
            } else {
                OpenOptions::new().read(true).write(true).open(&filename)?
            };
            file.seek(SeekFrom::Start(file_offset))?;
            if !staged.is_empty() {
                file.write_all(&staged)?;
            }
            file.flush()?;
        }
        Ok(Value::Null)
    }

    fn op_call(&self, operation: &Value) -> Result<Value> {
        let function = text(&operation["Function"])?;
        if function == "vec-dot" {
            exact_fields(
                operation,
                &["Op", "Function", "Type", "Count", "Input", "Rhs", "Output"],
            )?;
        } else {
            exact_fields(
                operation,
                &["Op", "Function", "Type", "Count", "Input", "Output"],
            )?;
        }

        let type_id = integer(&operation["Type"], TYPE_COUNT - 1)?;
        let traits = traits_of(type_id);
        let count = integer(&operation["Count"], MAX_CALL_COUNT)?;
        let (in_id, in_offset) = buffer_ref(&operation["Input"])?;
        let (out_id, out_offset) = buffer_ref(&operation["Output"])?;
        let in_buf = self.get_buffer(in_id)?;
        let out_buf = self.get_buffer(out_id)?;

        match function {
            "from-float-cpu" | "from-float-reference" => {
                if function == "from-float-cpu" {
                    require(
                        traits.from_float_cpu,
                        "production CPU quantizer unavailable",
                    )?;
                } else {
                    require(
                        traits.from_float_ref,
                        "production reference quantizer unavailable",
                    )?;
                }
                let in_bytes = count
                    .checked_mul(4)
                    .ok_or_else(|| invalid("codec float extent overflow"))?;
                require(in_bytes <= MAX_BUFFER, "codec input exceeds buffer bound")?;
                let out_bytes = row_bytes(traits, count)?;
                let floats = read_buffer_f32(&in_buf, in_offset, in_bytes)?;
                let packed = quantize_row(type_id, &floats)?;
                require(packed.len() == out_bytes, "codec quantized extent differs")?;
                write_buffer_bytes(&out_buf, out_offset, &packed)?;
                Ok(Value::Null)
            }
            "to-float" => {
                require(traits.to_float, "production decoder unavailable")?;
                let in_bytes = row_bytes(traits, count)?;
                let out_bytes = count
                    .checked_mul(4)
                    .ok_or_else(|| invalid("codec float extent overflow"))?;
                require(out_bytes <= MAX_BUFFER, "codec output exceeds buffer bound")?;
                let packed = read_buffer_bytes(&in_buf, in_offset, in_bytes)?;
                let decoded = decode_row(type_id, count, &packed)?;
                require(decoded.len() == count, "codec decoded count differs")?;
                let mut raw = Vec::with_capacity(out_bytes);
                for value in decoded {
                    raw.extend_from_slice(&value.to_le_bytes());
                }
                write_buffer_bytes(&out_buf, out_offset, &raw)?;
                Ok(Value::Null)
            }
            "vec-dot" => {
                require(
                    traits.vec_dot && traits.vec_dot_type < TYPE_COUNT,
                    "production vector dot unavailable",
                )?;
                let (rhs_id, rhs_offset) = buffer_ref(&operation["Rhs"])?;
                let rhs_buf = self.get_buffer(rhs_id)?;
                let lhs_bytes = row_bytes(traits, count)?;
                let rhs_traits = traits_of(traits.vec_dot_type);
                let rhs_bytes = row_bytes(rhs_traits, count)?;
                let lhs = read_buffer_bytes(&in_buf, in_offset, lhs_bytes)?;
                let rhs = read_buffer_bytes(&rhs_buf, rhs_offset, rhs_bytes)?;
                check_buffer_extent(&out_buf, out_offset, 4)?;
                let dot_bytes = match vec_dot_row(type_id, count, &lhs, &rhs) {
                    Ok(value) => value.to_le_bytes(),
                    // Upstream test-quantize-perf prepares rhs with `t.id` rather than
                    // `vec_dot_type`, so random nibbles can hit Inf/NaN scale bit patterns.
                    // Native `ggml_vec_dot` writes IEEE NaN/Inf without throwing.
                    Err(err) if err.to_string().contains("nonfinite") => f32::NAN.to_le_bytes(),
                    Err(err) => return Err(err),
                };
                write_buffer_bytes(&out_buf, out_offset, &dot_bytes)?;
                Ok(Value::Null)
            }
            _ => Err(invalid("production vector dot unavailable")),
        }
    }

    fn op_clock(&self, operation: &Value) -> Result<Value> {
        exact_fields(operation, &["Op", "Counter"])?;
        let counter = text(&operation["Counter"])?;
        if counter == "microseconds" {
            let micros = i64::try_from(self.started.elapsed().as_micros())?;
            return Ok(json!({"Value": micros}));
        }
        require(counter == "cycles", "unknown codec clock counter")?;
        Ok(json!({"Value": 0_i64}))
    }

    fn op_trace(&self, worker: u32, operation: &Value) -> Result<Value> {
        let mut actions = if operation.get("ActionsFile").is_some() {
            exact_fields(operation, &["Op", "ActionsFile"])?;
            let filename = self.session_path(&operation["ActionsFile"])?;
            read_actions_json(&filename)?
        } else {
            exact_fields(operation, &["Op", "Actions"])?;
            check_json_depth(&operation["Actions"], MAX_JSON_DEPTH)?;
            operation["Actions"].clone()
        };

        let mut trace = Vec::new();
        let mut total = 0_usize;
        loop {
            let batch = actions
                .as_array()
                .ok_or_else(|| invalid("codec explicit action count exceeds bound"))?;
            require(
                batch.len() <= MAX_TRACE_ACTIONS - total,
                "codec explicit action count exceeds bound",
            )?;
            total += batch.len();

            let mut continuation: Option<PathBuf> = None;
            for (index, action) in batch.iter().enumerate() {
                let obj = action
                    .as_object()
                    .ok_or_else(|| invalid("codec trace action required"))?;
                let op_val = obj
                    .get("Op")
                    .ok_or_else(|| invalid("codec trace action required"))?;
                let name = text(op_val)?;

                if name == "qb-checkpoint" {
                    exact_fields(
                        action,
                        &["Op", "OutputFile", "NextFile", "TimeoutMilliseconds"],
                    )?;
                    require(
                        index + 1 == batch.len(),
                        "codec checkpoint must end the supplied action batch",
                    )?;
                    let output = self.session_path(&action["OutputFile"])?;
                    let next_file = self.session_path(&action["NextFile"])?;
                    let mut temp_os = OsString::from(output.as_os_str());
                    temp_os.push(".partial");
                    let temporary = PathBuf::from(temp_os);
                    require(
                        output != next_file
                            && !output.exists()
                            && !next_file.exists()
                            && !temporary.exists(),
                        "codec checkpoint files must be fresh and distinct",
                    )?;
                    let timeout_ms =
                        integer(&action["TimeoutMilliseconds"], MAX_CHECKPOINT_TIMEOUT_MS)?;
                    require(timeout_ms > 0, "codec checkpoint timeout empty")?;

                    trace.push(Value::Null);
                    let encoded = serde_json::to_vec(&json!({"Trace": trace}))?;
                    require(
                        encoded.len() <= MAX_CHECKPOINT_BYTES,
                        "codec checkpoint output exceeds 16MiB",
                    )?;
                    {
                        let mut stream = OpenOptions::new()
                            .create_new(true)
                            .write(true)
                            .open(&temporary)?;
                        stream.write_all(&encoded)?;
                        stream.flush()?;
                    }
                    fs::rename(&temporary, &output)?;

                    let deadline =
                        Instant::now() + Duration::from_millis(u64::try_from(timeout_ms)?);
                    while !next_file.exists() {
                        require(Instant::now() < deadline, "codec continuation timed out")?;
                        thread::sleep(Duration::from_millis(1));
                    }
                    continuation = Some(next_file);
                    break;
                }

                trace.push(self.dispatch(worker, action, false)?);
            }

            if let Some(next_file) = continuation {
                actions = read_actions_json(&next_file)?;
                trace.clear();
            } else {
                break;
            }
        }
        Ok(json!({"Trace": trace}))
    }

    fn op_model_load(&self, worker: u32, operation: &Value) -> Result<Value> {
        exact_fields(
            operation,
            &["Op", "Path", "Base", "Params", "Overrides", "Progress"],
        )?;
        let raw_path = text(&operation["Path"])?;
        let path = Path::new(raw_path);
        require(path.is_absolute(), "model path must be absolute")?;
        let base = text_bound(&operation["Base"], 16)?;
        require(
            matches!(base, "default" | "common" | "zero"),
            "unknown model parameter base",
        )?;
        let overrides = operation["Overrides"]
            .as_object()
            .ok_or_else(|| invalid("model overrides must be an object"))?;
        require(overrides.len() <= 7, "model overrides exceed bound")?;
        for (key, value) in overrides {
            match key.as_str() {
                "GpuLayers" => {
                    let _ = integer(value, 999)?;
                }
                "HeadSource" => {
                    let _ = integer_u64(value, MAX_HANDLES)?;
                }
                "VocabOnly" => {
                    let _ = boolean(value, "VocabOnly must be boolean")?;
                }
                "LoadMode" => {
                    require(value == "none", "unsupported load mode")?;
                }
                "SplitMode" => {
                    let _ = integer(value, 2)?;
                }
                "MainGpu" => {
                    let n = value
                        .as_i64()
                        .ok_or_else(|| invalid("MainGpu must be integer"))?;
                    require((-1..=15).contains(&n), "MainGpu outside bound")?;
                }
                "Devices" => {
                    let list = value
                        .as_array()
                        .ok_or_else(|| invalid("model device list must be array"))?;
                    require(list.len() <= 16, "model device list exceeds bound")?;
                }
                _ => return Err(invalid("unknown model override")),
            }
        }
        let progress = boolean(&operation["Progress"], "Progress must be boolean")?;
        require(!progress, "codec model session does not support progress")?;

        let Ok(model) = load_gguf_model(path, worker) else {
            return Ok(json!({"Model": Value::Null}));
        };
        let id = {
            let mut table = self
                .models
                .lock()
                .map_err(|_| invalid("codec model lock poisoned"))?;
            require(
                table.models.len() < MAX_MODEL_HANDLES && table.next_id < MAX_HANDLES,
                "model handle count exceeded",
            )?;
            let id = table.next_id;
            table.next_id += 1;
            table.models.insert(id, Arc::new(model));
            id
        };
        Ok(json!({"Model": id}))
    }

    fn op_context(&self, worker: u32, operation: &Value) -> Result<Value> {
        exact_fields(operation, &["Op", "Model", "Params", "Overrides"])?;
        let model_id = integer_u64(&operation["Model"], MAX_HANDLES)?;
        let _ = self.get_model(model_id)?;
        let overrides = operation["Overrides"]
            .as_object()
            .ok_or_else(|| invalid("context overrides must be an object"))?;
        require(overrides.len() <= 16, "context overrides exceed bound")?;
        if let Some(ctx) = overrides.get("Context") {
            let _ = integer(ctx, 4096)?;
        }
        let id = {
            let mut table = self
                .models
                .lock()
                .map_err(|_| invalid("codec model lock poisoned"))?;
            require(
                table.contexts.len() < MAX_MODEL_HANDLES && table.next_id < MAX_HANDLES,
                "model handle count exceeded",
            )?;
            let id = table.next_id;
            table.next_id += 1;
            table.contexts.insert(
                id,
                ContextRecord {
                    owner_worker: worker,
                    _model_id: model_id,
                },
            );
            id
        };
        Ok(json!({"Context": id}))
    }

    fn op_release(&self, worker: u32, operation: &Value) -> Result<Value> {
        exact_fields(operation, &["Op", "Kind", "Handle"])?;
        let kind = text_bound(&operation["Kind"], 16)?;
        let id = integer_u64(&operation["Handle"], MAX_HANDLES)?;
        match kind {
            "model" => {
                let removed = {
                    let mut table = self
                        .models
                        .lock()
                        .map_err(|_| invalid("codec model lock poisoned"))?;
                    let found = table
                        .models
                        .get(&id)
                        .ok_or_else(|| invalid("unknown model session handle"))?;
                    require(
                        found.owner_worker == worker,
                        "release must use owning native worker",
                    )?;
                    table.models.remove(&id)
                };
                drop(removed);
            }
            "context" => {
                let mut table = self
                    .models
                    .lock()
                    .map_err(|_| invalid("codec model lock poisoned"))?;
                let found = table
                    .contexts
                    .get(&id)
                    .ok_or_else(|| invalid("unknown model session handle"))?;
                require(
                    found.owner_worker == worker,
                    "release must use owning native worker",
                )?;
                table.contexts.remove(&id);
            }
            _ => return Err(invalid("unknown handle release kind")),
        }
        Ok(Value::Null)
    }

    fn op_model_tensors(&self, operation: &Value) -> Result<Value> {
        exact_fields(operation, &["Op", "Model"])?;
        let model_id = integer_u64(&operation["Model"], MAX_HANDLES)?;
        let model = self.get_model(model_id)?;
        require(
            model.tensors.len() <= MAX_TENSORS,
            "codec model tensor count exceeds bound",
        )?;
        let values = model
            .tensors
            .iter()
            .map(|t| {
                json!({
                    "Name": t.name,
                    "Type": t.type_id,
                    "NE": t.ne,
                    "NB": t.nb,
                    "Elements": t.elements,
                    "Bytes": t.bytes,
                    "TypeSize": t.type_size,
                    "BlockSize": t.block_size,
                })
            })
            .collect::<Vec<_>>();
        Ok(json!({"Tensors": values}))
    }

    fn op_model_copy_f32(&self, operation: &Value) -> Result<Value> {
        exact_fields(
            operation,
            &[
                "Op",
                "Model",
                "Name",
                "ElementOffset",
                "Count",
                "Buffer",
                "BufferOffset",
            ],
        )?;
        let model_id = integer_u64(&operation["Model"], MAX_HANDLES)?;
        let model = self.get_model(model_id)?;
        let name = text(&operation["Name"])?;
        let index = *model
            .by_name
            .get(name)
            .ok_or_else(|| invalid("codec model tensor missing"))?;
        let tensor = model
            .tensors
            .get(index)
            .ok_or_else(|| invalid("codec model tensor missing"))?;
        require(
            tensor.type_id == 0 || tensor.type_id == 1,
            "codec source tensor must be contiguous CPU float",
        )?;
        let elem_offset = integer(&operation["ElementOffset"], MAX_CALL_COUNT)?;
        let count = integer(&operation["Count"], MAX_CALL_COUNT)?;
        let total_elements = usize::try_from(tensor.elements)?;
        require(
            elem_offset <= total_elements && count <= total_elements - elem_offset,
            "codec model element range exceeds tensor",
        )?;
        let buf_id = integer_u64(&operation["Buffer"], MAX_HANDLES)?;
        let buf = self.get_buffer(buf_id)?;
        let buf_offset = integer(&operation["BufferOffset"], MAX_BUFFER)?;
        let out_bytes = count
            .checked_mul(4)
            .ok_or_else(|| invalid("codec model copy overflow"))?;
        check_buffer_extent(&buf, buf_offset, out_bytes)?;

        let elem_width = if tensor.type_id == 0 { 4_u64 } else { 2_u64 };
        let byte_start = tensor
            .file_offset
            .checked_add(
                u64::try_from(elem_offset)?
                    .checked_mul(elem_width)
                    .ok_or_else(|| invalid("codec model offset overflow"))?,
            )
            .ok_or_else(|| invalid("codec model offset overflow"))?;
        let read_bytes = count
            .checked_mul(usize::try_from(elem_width)?)
            .ok_or_else(|| invalid("codec model byte extent overflow"))?;
        let mut raw = vec![0_u8; read_bytes];
        if read_bytes > 0 {
            let mut file = model
                .file
                .lock()
                .map_err(|_| invalid("codec model file lock poisoned"))?;
            file.seek(SeekFrom::Start(byte_start))?;
            file.read_exact(&mut raw)?;
        }

        let converted = if tensor.type_id == 0 {
            raw
        } else {
            let mut f32_bytes = Vec::with_capacity(out_bytes);
            for chunk in raw.as_chunks::<2>().0 {
                let value = f16_to_f32(u16::from_le_bytes(*chunk));
                f32_bytes.extend_from_slice(&value.to_le_bytes());
            }
            f32_bytes
        };
        write_buffer_bytes(&buf, buf_offset, &converted)?;
        Ok(Value::Null)
    }
}

fn inventory_json() -> Value {
    let types = (0..TYPE_COUNT)
        .map(|id| {
            let info = traits_of(id);
            json!({
                "Type": id,
                "Name": info.name,
                "BlockSize": info.block_size,
                "TypeSize": info.type_size,
                "FromFloatReference": info.from_float_ref,
                "FromFloatCpu": info.from_float_cpu,
                "ToFloat": info.to_float,
                "VecDot": info.vec_dot,
                "VecDotType": info.vec_dot_type,
            })
        })
        .collect::<Vec<_>>();
    json!({"Types": types})
}

fn read_buffer_bytes(buf: &RwLock<Vec<u8>>, offset: usize, len: usize) -> Result<Vec<u8>> {
    let guard = buf
        .read()
        .map_err(|_| invalid("codec buffer rwlock poisoned"))?;
    require(
        offset <= guard.len() && len <= guard.len() - offset,
        "codec buffer access exceeds extent",
    )?;
    Ok(guard[offset..offset + len].to_vec())
}

fn read_buffer_f32(buf: &RwLock<Vec<u8>>, offset: usize, byte_len: usize) -> Result<Vec<f32>> {
    let raw = read_buffer_bytes(buf, offset, byte_len)?;
    Ok(raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}

fn check_buffer_extent(buf: &RwLock<Vec<u8>>, offset: usize, len: usize) -> Result<()> {
    let guard = buf
        .read()
        .map_err(|_| invalid("codec buffer rwlock poisoned"))?;
    require(
        offset <= guard.len() && len <= guard.len() - offset,
        "codec buffer access exceeds extent",
    )
}

fn write_buffer_bytes(buf: &RwLock<Vec<u8>>, offset: usize, src: &[u8]) -> Result<()> {
    let mut guard = buf
        .write()
        .map_err(|_| invalid("codec buffer rwlock poisoned"))?;
    require(
        offset <= guard.len() && src.len() <= guard.len() - offset,
        "codec buffer access exceeds extent",
    )?;
    guard[offset..offset + src.len()].copy_from_slice(src);
    drop(guard);
    Ok(())
}

fn weakly_canonical(path: &Path) -> Result<PathBuf> {
    let mut current = path.to_path_buf();
    let mut tail = Vec::new();
    let base = loop {
        match fs::canonicalize(&current) {
            Ok(canon) => break canon,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                let Some(name) = current.file_name().map(OsStr::to_os_string) else {
                    return Err(invalid("codec buffer path outside session"));
                };
                tail.push(name);
                if !current.pop() {
                    return Err(invalid("codec buffer path outside session"));
                }
            }
            Err(err) => return Err(err.into()),
        }
    };
    let mut resolved = base;
    for part in tail.into_iter().rev() {
        require(
            part != ".." && part != ".",
            "codec buffer path outside session",
        )?;
        resolved.push(part);
    }
    Ok(resolved)
}

fn read_action_file_bytes(path: &Path) -> Result<Vec<u8>> {
    let meta = fs::metadata(path)?;
    require(meta.is_file(), "codec action file must be regular")?;
    require(
        meta.len() <= MAX_ACTION_FILE_BYTES,
        "codec action file exceeds bound",
    )?;
    let file = File::open(path)?;
    let expected = usize::try_from(meta.len())?;
    let mut bytes = Vec::with_capacity(expected);
    file.take(MAX_ACTION_FILE_BYTES + 1)
        .read_to_end(&mut bytes)?;
    require(
        bytes.len() == expected,
        "codec action file size changed during read",
    )?;
    Ok(bytes)
}

fn read_actions_json(path: &Path) -> Result<Value> {
    let bytes = read_action_file_bytes(path)?;
    let parsed: Value = serde_json::from_slice(&bytes)?;
    check_json_depth(&parsed, MAX_JSON_DEPTH)?;
    Ok(parsed)
}

// ---------------------------------------------------------------------------
// Clean-room GGUF v2/v3 tensor reader (F32 & F16 support)
// ---------------------------------------------------------------------------

struct GgufReader<'a> {
    reader: BufReader<&'a File>,
    file_len: u64,
}

impl<'a> GgufReader<'a> {
    fn new(file: &'a File, file_len: u64) -> Self {
        Self {
            reader: BufReader::new(file),
            file_len,
        }
    }

    fn read_bytes(&mut self, len: usize) -> Result<Vec<u8>> {
        let pos = self.reader.stream_position()?;
        let len_u64 = u64::try_from(len)?;
        require(
            pos <= self.file_len && len_u64 <= self.file_len - pos,
            "GGUF file truncated",
        )?;
        let mut buf = vec![0_u8; len];
        self.reader.read_exact(&mut buf)?;
        Ok(buf)
    }

    fn skip_bytes(&mut self, len: u64) -> Result<()> {
        let pos = self.reader.stream_position()?;
        require(
            pos <= self.file_len && len <= self.file_len - pos,
            "GGUF file truncated",
        )?;
        let next = pos + len;
        self.reader.seek(SeekFrom::Start(next))?;
        Ok(())
    }

    fn read_u32(&mut self) -> Result<u32> {
        let mut buf = [0_u8; 4];
        self.reader.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let mut buf = [0_u8; 8];
        self.reader.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    fn read_string(&mut self, max_len: u64) -> Result<String> {
        let len = self.read_u64()?;
        require(len <= max_len, "GGUF string exceeds bound")?;
        let raw = self.read_bytes(usize::try_from(len)?)?;
        let s = String::from_utf8(raw)?;
        require(!s.contains('\0'), "GGUF string contains NUL")?;
        Ok(s)
    }

    fn skip_value(&mut self, kind: u32, depth: u32) -> Result<()> {
        require(depth < 8, "GGUF metadata array nesting exceeds bound")?;
        match kind {
            0 | 1 | 7 => self.skip_bytes(1),
            2 | 3 => self.skip_bytes(2),
            4..=6 => self.skip_bytes(4),
            10..=12 => self.skip_bytes(8),
            8 => {
                let len = self.read_u64()?;
                require(len <= MAX_GGUF_STRING, "GGUF metadata string exceeds bound")?;
                self.skip_bytes(len)
            }
            9 => {
                let elem_type = self.read_u32()?;
                let count = self.read_u64()?;
                if let Some(width) = scalar_width(elem_type) {
                    let total = count
                        .checked_mul(width)
                        .ok_or_else(|| invalid("GGUF metadata array overflow"))?;
                    return self.skip_bytes(total);
                }
                let pos = self.reader.stream_position()?;
                require(
                    count <= self.file_len - pos,
                    "GGUF metadata array count exceeds file extent",
                )?;
                for _ in 0..count {
                    self.skip_value(elem_type, depth + 1)?;
                }
                Ok(())
            }
            _ => Err(invalid("unknown GGUF metadata type")),
        }
    }
}

const fn scalar_width(kind: u32) -> Option<u64> {
    match kind {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    value
        .checked_add(alignment - 1)
        .map(|aligned| aligned & !(alignment - 1))
        .ok_or_else(|| invalid("GGUF alignment arithmetic overflow"))
}

fn read_gguf_alignment(reader: &mut GgufReader<'_>, kv_count: u64) -> Result<u64> {
    let mut alignment = 32_u64;
    let mut seen_keys = BTreeSet::new();
    for _ in 0..kv_count {
        let key = reader.read_string(MAX_GGUF_STRING)?;
        require(seen_keys.insert(key.clone()), "duplicate GGUF metadata key")?;
        let kind = reader.read_u32()?;
        if key == "general.alignment" {
            require(kind == 4, "GGUF general.alignment must be UINT32")?;
            let align = reader.read_u32()?;
            require(
                align > 0 && align.is_power_of_two() && align <= MAX_GGUF_ALIGNMENT,
                "invalid GGUF alignment",
            )?;
            alignment = u64::from(align);
        } else {
            reader.skip_value(kind, 0)?;
        }
    }
    Ok(alignment)
}

fn read_gguf_tensors(
    reader: &mut GgufReader<'_>,
    tensor_count: usize,
) -> Result<(Vec<TensorRecord>, BTreeMap<String, usize>)> {
    let mut tensors = Vec::with_capacity(tensor_count);
    let mut by_name = BTreeMap::new();
    for idx in 0..tensor_count {
        let name = reader.read_string(4096)?;
        require(
            by_name.insert(name.clone(), idx).is_none(),
            "duplicate GGUF tensor name",
        )?;
        let n_dims = usize::try_from(reader.read_u32()?)?;
        require((1..=4).contains(&n_dims), "GGUF tensor rank outside 1..=4")?;
        let mut ne = [1_u64; 4];
        for dim in &mut ne[..n_dims] {
            *dim = reader.read_u64()?;
        }
        let elements = ne
            .iter()
            .try_fold(1_u64, |acc, &d| acc.checked_mul(d))
            .filter(|&e| e > 0 && e <= i64::MAX.unsigned_abs())
            .ok_or_else(|| invalid("GGUF tensor element count overflow"))?;
        let type_id = reader.read_u32()?;
        let type_idx = usize::try_from(type_id)?;
        require(type_idx < TYPE_COUNT, "unknown GGUF tensor type")?;
        let traits = traits_of(type_idx);
        let block_size = u64::try_from(traits.block_size)?;
        let type_size = u64::try_from(traits.type_size)?;
        require(
            block_size > 0 && type_size > 0 && ne[0].is_multiple_of(block_size),
            "GGUF tensor dimensions violate block layout",
        )?;
        let bytes = (elements / block_size)
            .checked_mul(type_size)
            .ok_or_else(|| invalid("GGUF tensor bytes overflow"))?;
        let nb0 = type_size;
        let nb1 = (nb0
            .checked_mul(ne[0])
            .ok_or_else(|| invalid("GGUF stride overflow"))?)
            / block_size;
        let nb2 = nb1
            .checked_mul(ne[1])
            .ok_or_else(|| invalid("GGUF stride overflow"))?;
        let nb3 = nb2
            .checked_mul(ne[2])
            .ok_or_else(|| invalid("GGUF stride overflow"))?;
        let rel_offset = reader.read_u64()?;
        tensors.push(TensorRecord {
            name,
            type_id,
            ne,
            nb: [nb0, nb1, nb2, nb3],
            elements,
            bytes,
            type_size,
            block_size,
            file_offset: rel_offset,
        });
    }
    Ok((tensors, by_name))
}

fn load_gguf_model(path: &Path, owner_worker: u32) -> Result<LoadedModel> {
    let file = File::open(path)?;
    let meta = file.metadata()?;
    require(meta.is_file(), "GGUF model path must be a regular file")?;
    let file_len = meta.len();

    let (tensors, by_name) = {
        let mut reader = GgufReader::new(&file, file_len);
        let magic = reader.read_bytes(4)?;
        require(magic == b"GGUF", "GGUF magic mismatch")?;
        let version = reader.read_u32()?;
        require(matches!(version, 2 | 3), "unsupported GGUF version")?;
        let tensor_count = usize::try_from(reader.read_u64()?)?;
        require(
            tensor_count <= MAX_TENSORS,
            "GGUF tensor count exceeds bound",
        )?;
        let kv_count = reader.read_u64()?;
        require(kv_count <= MAX_KV_ENTRIES, "GGUF kv count exceeds bound")?;

        let alignment = read_gguf_alignment(&mut reader, kv_count)?;
        let (mut tensors, by_name) = read_gguf_tensors(&mut reader, tensor_count)?;

        let header_end = reader.reader.stream_position()?;
        let data_start = if tensor_count > 0 {
            align_up(header_end, alignment)?
        } else {
            header_end
        };
        require(data_start <= file_len, "GGUF tensor section exceeds file")?;
        for tensor in &mut tensors {
            let abs_offset = data_start
                .checked_add(tensor.file_offset)
                .ok_or_else(|| invalid("GGUF tensor file offset overflow"))?;
            require(
                abs_offset <= file_len && tensor.bytes <= file_len - abs_offset,
                "GGUF tensor payload exceeds file",
            )?;
            tensor.file_offset = abs_offset;
        }
        (tensors, by_name)
    };

    Ok(LoadedModel {
        owner_worker,
        file: Mutex::new(file),
        tensors,
        by_name,
    })
}

/// Reports whether `request` contains stateless `qb-*` operations.
pub(super) fn handles(request: &Value) -> bool {
    if request
        .get("Op")
        .and_then(Value::as_str)
        .is_some_and(|op| op.starts_with("qb-"))
    {
        return true;
    }
    request["Operations"].as_array().is_some_and(|operations| {
        !operations.is_empty()
            && operations.iter().all(|operation| {
                operation
                    .get("Op")
                    .and_then(Value::as_str)
                    .or_else(|| operation.get("Kind").and_then(Value::as_str))
                    .is_some_and(|op| op.starts_with("qb-"))
            })
    })
}

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(prefix: &str) -> Result<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let seq = NEXT.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("{prefix}-{}-{nonce}-{seq}", std::process::id()));
        fs::create_dir(&path)?;
        Ok(Self { path })
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

/// Executes stateless `qb-*` requests routed via `reference_provider`.
pub(super) fn execute(request: &Value) -> Result<Value> {
    let dir = TempDir::new("SwiftBonsai-RefCodecBuffers")?;
    let session = Session::new(&dir.path)?;
    if request.get("Op").is_some() {
        return session.execute(0, request);
    }
    exact_fields(request, &["SchemaVersion", "Operations"])?;
    require(
        integer_u64(&request["SchemaVersion"], 1)? == 1,
        "codec buffer request SchemaVersion must equal 1",
    )?;
    let operations = request["Operations"]
        .as_array()
        .ok_or_else(|| invalid("codec buffer Operations must be an array"))?;
    require(
        !operations.is_empty() && operations.len() <= MAX_STATELESS_OPS,
        "codec buffer Operations must contain 1 through 128 operations",
    )?;
    let mut results = Vec::with_capacity(operations.len());
    for op in operations {
        let normalized = if op.get("Op").is_none() && op.get("Kind").is_some() {
            let mut mapped = op.clone();
            if let Some(obj) = mapped.as_object_mut()
                && let Some(kind) = obj.remove("Kind")
            {
                obj.insert("Op".into(), kind);
            }
            mapped
        } else {
            op.clone()
        };
        results.push(session.execute(0, &normalized)?);
    }
    Ok(json!({"SchemaVersion": 1, "Results": results}))
}

/// Provider identity metadata for clean-room persistent codec buffer conformance reporting.
pub(super) fn identity() -> Value {
    json!({
        "Kind": "independent-rust-codec-buffers",
        "SupportedSourceIds": IDS,
        "UpstreamCommit": super::UPSTREAM,
        "AssertionSources": [
            {"Path": IDS[0], "Sha256": PERF_SHA256},
            {"Path": IDS[1], "Sha256": STATS_SHA256},
        ],
        "Architecture": "clean-room-gguf-tensor-session-codec-buffers-and-checkpoint-trace-v1",
        "NativeReferenceReplay": false,
        "ProductionRuntimeCalled": false,
        "CompleteSourceGate": false,
        "NativeRetirementQualified": false
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, Target};
    use crate::quant_buffers::{Driver, QuantSession, call, pointer};
    use std::sync::mpsc;

    fn f32_to_f16_bits(value: f32) -> u16 {
        let bits = value.to_bits();
        let sign = u16::try_from((bits >> 16) & 0x8000).unwrap_or(0);
        let exp = (bits >> 23) & 0xff;
        let mant = bits & 0x007f_ffff;
        if exp == 255 {
            return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
        }
        if exp > 142 {
            return sign | 0x7c00;
        }
        if exp <= 112 {
            return sign;
        }
        let he = exp - 112;
        let hm = mant >> 13;
        let rem = mant & 0x1fff;
        let round_up = rem > 0x1000 || (rem == 0x1000 && (hm & 1) != 0);
        let rounded = (he << 10) + hm + u32::from(round_up);
        sign | u16::try_from(rounded).unwrap_or(0)
    }

    fn encode_gguf_str(out: &mut Vec<u8>, s: &str) -> Result<()> {
        out.extend_from_slice(&u64::try_from(s.len())?.to_le_bytes());
        out.extend_from_slice(s.as_bytes());
        Ok(())
    }

    fn write_synthetic_gguf(
        path: &Path,
        version: u32,
        alignment: Option<u32>,
        f32_values: &[f32],
        f16_values: &[f32],
    ) -> Result<()> {
        let mut buf = b"GGUF".to_vec();
        buf.extend_from_slice(&version.to_le_bytes());
        buf.extend_from_slice(&2_u64.to_le_bytes());
        let kv_count = u64::from(alignment.is_some()) + 1;
        buf.extend_from_slice(&kv_count.to_le_bytes());

        encode_gguf_str(&mut buf, "general.architecture")?;
        buf.extend_from_slice(&8_u32.to_le_bytes());
        encode_gguf_str(&mut buf, "llama")?;

        if let Some(align) = alignment {
            encode_gguf_str(&mut buf, "general.alignment")?;
            buf.extend_from_slice(&4_u32.to_le_bytes());
            buf.extend_from_slice(&align.to_le_bytes());
        }

        let align_u64 = u64::from(alignment.unwrap_or(32));
        let f32_bytes_len = u64::try_from(f32_values.len() * 4)?;
        let f16_offset = align_up(f32_bytes_len, align_u64)?;

        encode_gguf_str(&mut buf, "blk.0.attn_q.weight")?;
        buf.extend_from_slice(&2_u32.to_le_bytes());
        buf.extend_from_slice(&u64::try_from(f32_values.len())?.to_le_bytes());
        buf.extend_from_slice(&1_u64.to_le_bytes());
        buf.extend_from_slice(&0_u32.to_le_bytes());
        buf.extend_from_slice(&0_u64.to_le_bytes());

        encode_gguf_str(&mut buf, "blk.0.attn_k.weight")?;
        buf.extend_from_slice(&1_u32.to_le_bytes());
        buf.extend_from_slice(&u64::try_from(f16_values.len())?.to_le_bytes());
        buf.extend_from_slice(&1_u32.to_le_bytes());
        buf.extend_from_slice(&f16_offset.to_le_bytes());

        let header_aligned = usize::try_from(align_up(u64::try_from(buf.len())?, align_u64)?)?;
        buf.resize(header_aligned, 0);
        for &v in f32_values {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf.resize(header_aligned + usize::try_from(f16_offset)?, 0);
        for &v in f16_values {
            buf.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
        }
        fs::write(path, buf)?;
        Ok(())
    }

    #[test]
    fn provider_identity_and_registration_match_catalog() -> Result<()> {
        for id in IDS {
            assert!(crate::reference_provider::supports(id));
        }
        let info = identity();
        assert_eq!(info["SupportedSourceIds"], json!(IDS));
        assert_eq!(info["NativeReferenceReplay"], false);
        assert_eq!(info["ProductionRuntimeCalled"], false);
        assert_eq!(info["CompleteSourceGate"], false);
        assert_eq!(info["NativeRetirementQualified"], false);

        let inv = crate::reference_provider::execute(&json!({"Op": "qb-inventory"}))?;
        let types = inv["Types"]
            .as_array()
            .ok_or_else(|| invalid("missing Types array"))?;
        assert_eq!(types.len(), TYPE_COUNT);
        assert_eq!(types[0]["Name"], "f32");
        assert_eq!(types[8]["Name"], "q8_0");

        let wrapped = crate::reference_provider::execute(&json!({
            "SchemaVersion": 1,
            "Operations": [{"Op": "qb-time-init"}, {"Op": "qb-clock", "Counter": "microseconds"}]
        }))?;
        assert_eq!(wrapped["SchemaVersion"], 1);
        assert_eq!(wrapped["Results"][0], Value::Null);
        assert!(wrapped["Results"][1]["Value"].as_i64().is_some());
        Ok(())
    }

    fn alloc_buf(session: &Session, bytes: usize, alignment: usize, offset: usize) -> Result<u64> {
        session.execute(
            0,
            &json!({"Op": "qb-buffer-new", "Bytes": bytes, "Alignment": alignment, "Offset": offset}),
        )?["Buffer"]
            .as_u64()
            .ok_or_else(|| invalid("missing buffer id"))
    }

    #[test]
    fn direct_qb_ops_buffers_codecs_and_worker_rules() -> Result<()> {
        let dir = TempDir::new("SwiftBonsai-CodecBuffers-Direct")?;
        let session = Session::new(&dir.path)?;

        assert_eq!(
            session.execute(0, &json!({"Op": "qb-time-init"}))?,
            Value::Null
        );
        assert_eq!(
            session.execute(0, &json!({"Op": "qb-init-type", "Type": 2}))?,
            Value::Null
        );
        let t0 = session.execute(0, &json!({"Op": "qb-clock", "Counter": "microseconds"}))?;
        let t1 = session.execute(0, &json!({"Op": "qb-clock", "Counter": "cycles"}))?;
        assert!(t0["Value"].as_i64().is_some_and(|v| v >= 0));
        assert!(t1["Value"].as_i64().is_some());

        // Worker != 0 must be rejected on buffer allocation & release.
        assert!(
            session
                .execute(
                    1,
                    &json!({"Op": "qb-buffer-new", "Bytes": 256, "Alignment": 64, "Offset": 0})
                )
                .is_err()
        );

        let count = 256_usize;
        let in_buf = alloc_buf(&session, count * 4, 64, 16)?;
        let q_buf = alloc_buf(&session, count * 4, 64, 0)?;
        let act_buf = alloc_buf(&session, count * 4, 64, 0)?;
        let dec_buf = alloc_buf(&session, count * 4, 64, 0)?;
        let dot_buf = alloc_buf(&session, 16, 16, 0)?;
        assert!(
            session
                .execute(2, &json!({"Op": "qb-buffer-free", "Buffer": in_buf}))
                .is_err()
        );

        let in_file = dir.path.join("input.bin");
        let mut raw_floats = Vec::with_capacity(count * 4);
        for i in 0..count {
            let v = f32::mul_add(f32::from(u16::try_from((i % 17) + 1)?), 0.25, -1.5);
            raw_floats.extend_from_slice(&v.to_le_bytes());
        }
        fs::write(&in_file, &raw_floats)?;
        session.execute(
            0,
            &json!({"Op": "qb-buffer-load", "Buffer": in_buf, "Offset": 0, "File": in_file, "FileOffset": 0, "Bytes": count * 4, "Truncate": false}),
        )?;

        for op in [
            call(
                "from-float-reference",
                2,
                count,
                pointer(in_buf, 0),
                pointer(q_buf, 0),
            ),
            call("to-float", 2, count, pointer(q_buf, 0), pointer(dec_buf, 0)),
            call(
                "from-float-cpu",
                2,
                count,
                pointer(in_buf, 0),
                pointer(q_buf, 0),
            ),
            call(
                "from-float-cpu",
                8,
                count,
                pointer(in_buf, 0),
                pointer(act_buf, 0),
            ),
        ] {
            session.execute(1, &op)?;
        }
        let mut dot_op = call("vec-dot", 2, count, pointer(q_buf, 0), pointer(dot_buf, 0));
        dot_op["Rhs"] = pointer(act_buf, 0);
        session.execute(1, &dot_op)?;

        let dot_out = dir.path.join("dot.bin");
        session.execute(
            1,
            &json!({"Op": "qb-buffer-store", "Buffer": dot_buf, "Offset": 0, "File": dot_out, "FileOffset": 0, "Bytes": 4, "Truncate": true}),
        )?;
        let dot_val = f32::from_le_bytes(
            fs::read(&dot_out)?
                .as_slice()
                .try_into()
                .map_err(|_| invalid("dot bytes len"))?,
        );
        assert!(dot_val.is_finite() && dot_val > 100.0);

        for id in [in_buf, q_buf, act_buf, dec_buf, dot_buf] {
            session.execute(0, &json!({"Op": "qb-buffer-free", "Buffer": id}))?;
        }
        assert!(session.finish()? > 0);
        Ok(())
    }

    fn publish_next_file(path: &Path, bytes: &[u8]) -> Result<()> {
        let mut temp_os = OsString::from(path.as_os_str());
        temp_os.push(".partial");
        let temporary = PathBuf::from(temp_os);
        {
            let mut file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&temporary)?;
            file.write_all(bytes)?;
            file.flush()?;
        }
        fs::rename(&temporary, path)?;
        Ok(())
    }

    fn run_concurrent_checkpoints(
        session: &Arc<Session>,
        dir: &Path,
        version: u32,
        model: u64,
        jobs: [(u32, &str, u64, &[f32]); 2],
    ) -> Result<()> {
        thread::scope(|scope| -> Result<()> {
            let mut handles = Vec::new();
            for (lane, tensor_name, buf_id, expected_vals) in jobs {
                let s = Arc::clone(session);
                let actions_file = dir.join(format!("actions-v{version}-lane{lane}.json"));
                let out_file = dir.join(format!("stored-v{version}-lane{lane}.bin"));
                let ckpt_file = dir.join(format!("ckpt-v{version}-lane{lane}.json"));
                let next_file = dir.join(format!("next-v{version}-lane{lane}.json"));
                let actions = json!([
                    {"Op": "qb-model-copy-f32", "Model": model, "Name": tensor_name, "ElementOffset": 0, "Count": 512, "Buffer": buf_id, "BufferOffset": 0},
                    {"Op": "qb-buffer-store", "Buffer": buf_id, "Offset": 0, "File": out_file, "FileOffset": 0, "Bytes": 512 * 4, "Truncate": true},
                    {"Op": "qb-checkpoint", "OutputFile": ckpt_file, "NextFile": next_file, "TimeoutMilliseconds": 5000}
                ]);
                fs::write(&actions_file, serde_json::to_vec(&actions)?)?;
                let (tx, rx) = mpsc::channel();
                let join = scope.spawn(move || {
                    let res = s.execute(
                        lane,
                        &json!({"Op": "qb-trace", "ActionsFile": actions_file}),
                    );
                    let _ = tx.send(res);
                });
                handles.push((join, rx, ckpt_file, next_file, out_file, expected_vals));
            }
            for (join, rx, ckpt_file, next_file, out_file, expected_vals) in handles {
                let started = Instant::now();
                while !ckpt_file.is_file() {
                    require(
                        started.elapsed() < Duration::from_secs(5),
                        "checkpoint wait timed out",
                    )?;
                    thread::sleep(Duration::from_millis(1));
                }
                let ckpt: Value = serde_json::from_slice(&fs::read(&ckpt_file)?)?;
                assert_eq!(ckpt["Trace"], json!([null, null, null]));
                let stored = fs::read(&out_file)?;
                assert_eq!(stored.len(), 512 * 4);
                for (got, &exp) in stored.as_chunks::<4>().0.iter().zip(expected_vals) {
                    assert_eq!(f32::from_le_bytes(*got).to_bits(), exp.to_bits());
                }
                publish_next_file(&next_file, b"[]")?;
                let final_res = rx.recv().map_err(|_| invalid("worker disconnected"))??;
                assert_eq!(final_res, json!({"Trace": []}));
                join.join().map_err(|_| invalid("worker panicked"))?;
            }
            Ok(())
        })
    }

    #[test]
    fn gguf_v2_and_v3_f32_f16_load_copy_and_concurrent_trace_checkpoints() -> Result<()> {
        let dir = TempDir::new("SwiftBonsai-CodecBuffers-Gguf")?;
        let session = Arc::new(Session::new(&dir.path)?);

        let f32_vals: Vec<f32> = (0..512)
            .map(|i| f32::mul_add(f32::from(u16::try_from(i % 29).unwrap_or(0)), 0.125, -1.5))
            .collect();
        let f16_vals: Vec<f32> = (0..512)
            .map(|i| f32::mul_add(f32::from(u16::try_from(i % 19).unwrap_or(0)), 0.5, -4.0))
            .collect();

        for (version, align) in [(2_u32, None), (3_u32, Some(64_u32))] {
            let gguf_path = dir.path.join(format!("synthetic-v{version}.gguf"));
            write_synthetic_gguf(&gguf_path, version, align, &f32_vals, &f16_vals)?;

            let model = session.execute(
                0,
                &json!({"Op": "model-load", "Path": gguf_path, "Base": "default", "Params": null, "Overrides": {"LoadMode": "none"}, "Progress": false}),
            )?["Model"]
                .as_u64()
                .ok_or_else(|| invalid("missing model id"))?;
            let ctx = session.execute(
                0,
                &json!({"Op": "context", "Model": model, "Params": null, "Overrides": {"Context": 256}}),
            )?["Context"]
                .as_u64()
                .ok_or_else(|| invalid("missing context id"))?;

            let tensors = session.execute(0, &json!({"Op": "qb-model-tensors", "Model": model}))?;
            let list = tensors["Tensors"]
                .as_array()
                .ok_or_else(|| invalid("missing Tensors"))?;
            assert_eq!(list.len(), 2);
            assert_eq!(list[0]["Name"], "blk.0.attn_q.weight");
            assert_eq!(list[0]["Type"], 0);
            assert_eq!(list[1]["Name"], "blk.0.attn_k.weight");
            assert_eq!(list[1]["Type"], 1);

            let buf_f32 = alloc_buf(&session, 512 * 4, 16, 0)?;
            let buf_f16 = alloc_buf(&session, 512 * 4, 16, 0)?;
            run_concurrent_checkpoints(
                &session,
                &dir.path,
                version,
                model,
                [
                    (1, "blk.0.attn_q.weight", buf_f32, &f32_vals),
                    (2, "blk.0.attn_k.weight", buf_f16, &f16_vals),
                ],
            )?;

            for id in [buf_f32, buf_f16] {
                session.execute(0, &json!({"Op": "qb-buffer-free", "Buffer": id}))?;
            }
            session.execute(
                0,
                &json!({"Op": "release", "Kind": "context", "Handle": ctx}),
            )?;
            session.execute(
                0,
                &json!({"Op": "release", "Kind": "model", "Handle": model}),
            )?;
        }
        session.finish()?;
        Ok(())
    }

    #[test]
    fn rejects_malformed_requests_and_bounds_violations() -> Result<()> {
        let dir = TempDir::new("SwiftBonsai-CodecBuffers-Reject")?;
        let session = Session::new(&dir.path)?;

        assert!(session.execute(0, &json!({"Op": "qb-unknown"})).is_err());
        assert!(session.execute(16, &json!({"Op": "qb-time-init"})).is_err());
        assert!(
            session
                .execute(
                    0,
                    &json!({"Op": "qb-buffer-new", "Bytes": 16, "Alignment": 3, "Offset": 0})
                )
                .is_err()
        );
        assert!(
            session
                .execute(
                    0,
                    &json!({"Op": "qb-buffer-new", "Bytes": 16, "Alignment": 16, "Offset": 32})
                )
                .is_err()
        );
        assert!(
            session
                .execute(0, &json!({"Op": "qb-buffer-free", "Buffer": 999}))
                .is_err()
        );

        let buf = alloc_buf(&session, 64, 16, 0)?;
        for bad_path in [
            PathBuf::from("relative.bin"),
            dir.path.join("..").join("escape.bin"),
        ] {
            assert!(
                session
                    .execute(
                        0,
                        &json!({"Op": "qb-buffer-store", "Buffer": buf, "Offset": 0, "File": bad_path, "FileOffset": 0, "Bytes": 4, "Truncate": true})
                    )
                    .is_err()
            );
        }
        assert!(
            session
                .execute(
                    0,
                    &json!({"Op": "qb-trace", "Actions": [{"Op": "qb-inventory"}]})
                )
                .is_err()
        );
        assert!(
            session
                .execute(
                    0,
                    &json!({"Op": "qb-trace", "Actions": [
                        {"Op": "qb-checkpoint", "OutputFile": dir.path.join("o.json"), "NextFile": dir.path.join("n.json"), "TimeoutMilliseconds": 100},
                        {"Op": "qb-time-init"}
                    ]})
                )
                .is_err()
        );

        let bad_gguf = dir.path.join("bad.gguf");
        fs::write(&bad_gguf, b"NOTG")?;
        let bad_res = session.execute(
            0,
            &json!({"Op": "model-load", "Path": bad_gguf, "Base": "default", "Params": null, "Overrides": {"LoadMode": "none"}, "Progress": false}),
        )?;
        assert_eq!(bad_res, json!({"Model": null}));
        assert!(
            session
                .execute(
                    0,
                    &call("to-float", 0, 16, pointer(buf, 0), pointer(buf, 0))
                )
                .is_err()
        );
        assert!(
            session
                .execute(
                    0,
                    &call("from-float-cpu", 2, 16, pointer(buf, 0), pointer(buf, 0))
                )
                .is_err()
        );

        session.execute(0, &json!({"Op": "qb-buffer-free", "Buffer": buf}))?;
        session.finish()?;
        Ok(())
    }

    #[test]
    fn quant_session_supports_reference_and_cpp_and_preserves_rust_unsupported() -> Result<()> {
        let rust_engine = Engine::new(Target::Rust, None, None, None, None)?;
        assert!(QuantSession::open(&rust_engine, "quantize-perf")?.is_none());
        assert!(QuantSession::open_quantization(&rust_engine)?.is_none());

        for target in [Target::RustReference, Target::Cpp] {
            let engine = Engine::new(target, None, None, None, None)?;
            let session = QuantSession::open(&engine, "quantize-perf")?
                .ok_or_else(|| invalid("expected QuantSession for reference target"))?;
            let driver = Driver::new(session.client(), session.evidence_directory());
            let types = driver.inventory()?;
            assert_eq!(types.len(), TYPE_COUNT);

            // Exercise all 22 source-eligible (cpu && decoder) types through perf operations,
            // including vec_dot_q where q2 is prepared with `t.id` as in test-quantize-perf.
            let count = 256_usize;
            let cap = count * 4 + 128;
            let data1 = driver.buffer(cap, 64, 0)?;
            let data2 = driver.buffer(cap, 64, 0)?;
            let q1 = driver.buffer(cap, 64, 0)?;
            let q2 = driver.buffer(cap, 64, 0)?;
            let out = driver.buffer(cap, 64, 0)?;

            let f1_path = driver.path("d1");
            let f2_path = driver.path("d2");
            let mut b1 = Vec::with_capacity(count * 4);
            let mut b2 = Vec::with_capacity(count * 4);
            for idx in 0..count {
                let x = f32::from(u16::try_from(idx)?) * 0.05;
                b1.extend_from_slice(&2.0_f32.mul_add(x.cos(), 0.1).to_le_bytes());
                b2.extend_from_slice(&2.0_f32.mul_add((x + 1.0).cos(), 0.1).to_le_bytes());
            }
            fs::write(&f1_path, &b1)?;
            fs::write(&f2_path, &b2)?;
            driver.load(data1, &f1_path, count * 4)?;
            driver.load(data2, &f2_path, count * 4)?;

            for t in types.iter().filter(|t| t.cpu && t.decoder) {
                driver.null(0, json!({"Op": "qb-init-type", "Type": t.id}))?;
                let mut actions = vec![
                    call(
                        "from-float-cpu",
                        t.id,
                        count,
                        pointer(data1, 0),
                        pointer(q1, 0),
                    ),
                    call(
                        "from-float-cpu",
                        t.id,
                        count,
                        pointer(data2, 0),
                        pointer(q2, 0),
                    ),
                    call("to-float", t.id, count, pointer(q1, 0), pointer(out, 0)),
                    call(
                        "from-float-cpu",
                        t.dot_operand,
                        count,
                        pointer(data1, 0),
                        pointer(q1, 0),
                    ),
                ];
                if t.reference {
                    actions.push(call(
                        "from-float-reference",
                        t.id,
                        count,
                        pointer(data1, 0),
                        pointer(q1, 0),
                    ));
                }
                let mut dot = call("vec-dot", t.id, count, pointer(q1, 0), pointer(out, 0));
                dot["Rhs"] = pointer(q2, 0);
                actions.push(dot);
                let trace = driver.trace(&actions)?;
                assert!(trace.iter().all(Value::is_null));
            }
            for b in [data1, data2, q1, q2, out] {
                driver.free(b)?;
            }
            session.finish()?;

            if std::env::var("GITHUB_ACTIONS").as_deref() == Ok("true")
                && std::env::var("GITHUB_REF").as_deref() == Ok("refs/heads/main")
            {
                let perf_report = crate::quantize_perf::run(&engine)?;
                assert_eq!(perf_report["Id"], IDS[0]);
                assert_eq!(perf_report["Status"], "pass", "{perf_report}");
            } else {
                assert!(crate::quantize_perf::run(&engine).is_err());
                assert!(crate::quantize_stats::run(&engine).is_err());
            }
        }
        Ok(())
    }

    #[test]
    fn multi_batch_checkpoint_continuation_hardware_concurrency_and_cli_options() -> Result<()> {
        let dir = TempDir::new("SwiftBonsai-CodecBuffers-Cont")?;
        let session = Arc::new(Session::new(&dir.path)?);
        let hw = session.execute(0, &json!({"Op": "hardware-concurrency"}))?;
        assert!(hw["Count"].as_u64().is_some_and(|n| n >= 1));

        let f32_vals: Vec<f32> = (0..512)
            .map(|i| f32::mul_add(f32::from(u16::try_from(i % 13).unwrap_or(0)), 0.25, -1.0))
            .collect();
        let gguf_path = dir.path.join("cont.gguf");
        write_synthetic_gguf(&gguf_path, 3, None, &f32_vals, &f32_vals)?;
        let model = session.execute(
            0,
            &json!({"Op": "model-load", "Path": gguf_path, "Base": "default", "Params": null, "Overrides": {"LoadMode": "none"}, "Progress": false}),
        )?["Model"]
            .as_u64()
            .ok_or_else(|| invalid("missing model id"))?;
        let buf = alloc_buf(&session, 512 * 4, 16, 0)?;

        let out_file = dir.path.join("combined.bin");
        let ckpt1 = dir.path.join("ckpt1.json");
        let next1 = dir.path.join("next1.json");
        let ckpt2 = dir.path.join("ckpt2.json");
        let next2 = dir.path.join("next2.json");
        let batch1 = json!([
            {"Op": "qb-model-copy-f32", "Model": model, "Name": "blk.0.attn_q.weight", "ElementOffset": 0, "Count": 256, "Buffer": buf, "BufferOffset": 0},
            {"Op": "qb-buffer-store", "Buffer": buf, "Offset": 0, "File": out_file, "FileOffset": 0, "Bytes": 256 * 4, "Truncate": true},
            {"Op": "qb-checkpoint", "OutputFile": ckpt1, "NextFile": next1, "TimeoutMilliseconds": 5000}
        ]);
        let batch2 = json!([
            {"Op": "qb-model-copy-f32", "Model": model, "Name": "blk.0.attn_q.weight", "ElementOffset": 256, "Count": 256, "Buffer": buf, "BufferOffset": 256 * 4},
            {"Op": "qb-buffer-store", "Buffer": buf, "Offset": 256 * 4, "File": out_file, "FileOffset": 256 * 4, "Bytes": 256 * 4, "Truncate": false},
            {"Op": "qb-checkpoint", "OutputFile": ckpt2, "NextFile": next2, "TimeoutMilliseconds": 5000}
        ]);

        let s = Arc::clone(&session);
        let (tx, rx) = mpsc::channel();
        let worker = thread::spawn(move || {
            let _ = tx.send(s.execute(3, &json!({"Op": "qb-trace", "Actions": batch1})));
        });
        while !ckpt1.is_file() {
            thread::sleep(Duration::from_millis(1));
        }
        publish_next_file(&next1, &serde_json::to_vec(&batch2)?)?;
        while !ckpt2.is_file() {
            thread::sleep(Duration::from_millis(1));
        }
        publish_next_file(&next2, b"[]")?;
        let final_trace = rx.recv().map_err(|_| invalid("worker disconnected"))??;
        assert_eq!(final_trace, json!({"Trace": []}));
        worker.join().map_err(|_| invalid("worker panicked"))?;

        let stored = fs::read(&out_file)?;
        assert_eq!(stored.len(), 512 * 4);
        for (got, &exp) in stored.as_chunks::<4>().0.iter().zip(&f32_vals) {
            assert_eq!(f32::from_le_bytes(*got).to_bits(), exp.to_bits());
        }
        session.execute(0, &json!({"Op": "qb-buffer-free", "Buffer": buf}))?;
        session.execute(
            0,
            &json!({"Op": "release", "Kind": "model", "Handle": model}),
        )?;
        session.finish()?;

        for id in IDS {
            let case = crate::CASES
                .iter()
                .find(|case| case.id == id)
                .ok_or_else(|| invalid("missing quantize case"))?;
            for (target, evidence) in [
                (Target::RustReference, None),
                (Target::Cpp, Some(dir.path.clone())),
            ] {
                crate::Options {
                    target,
                    case: Some(case),
                    benchmark: None,
                    diagnostic: None,
                    bridge: None,
                    evidence_directory: evidence,
                    source_directory: None,
                    asset_directory: None,
                    output: dir.path.join("unused-report.json"),
                }
                .validate()?;
            }
        }
        Ok(())
    }
}
