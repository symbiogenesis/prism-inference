//! Safe process/file transport shared by Rust-owned quantization source ports.
use super::engine::Engine;
use super::llama_grammar::{array, exact_fields, field, uint};
use super::reference_codec_buffers::Session;
use super::{Result, invalid};
use serde_json::{Value, json};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
#[derive(Clone)]
pub(super) struct QuantClient {
    session: Arc<Session>,
}
impl QuantClient {
    pub(super) fn call(&self, worker: u32, operation: &Value) -> Result<Value> {
        self.session.execute(worker, operation)
    }
}
pub(super) struct QuantSession {
    session: Arc<Session>,
    directory: PathBuf,
    cleanup: bool,
    finished: bool,
}
impl QuantSession {
    pub(super) fn open(engine: &Engine, name: &str) -> Result<Option<Self>> {
        if engine.is_rust() {
            return Ok(None);
        }
        let (directory, cleanup) = if engine.has_evidence_directory() {
            (engine.evidence_path(name)?, false)
        } else {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
            let seq = NEXT.fetch_add(1, Ordering::Relaxed);
            let sanitized: String = name
                .chars()
                .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
                .collect();
            let dir = std::env::temp_dir().join(format!(
                "SwiftBonsai-Quant-{sanitized}-{}-{nonce}-{seq}",
                std::process::id()
            ));
            fs::create_dir(&dir)?;
            (dir, true)
        };
        let session = Arc::new(Session::new(&directory)?);
        Ok(Some(Self {
            session,
            directory,
            cleanup,
            finished: false,
        }))
    }
    pub(super) fn open_quantization(engine: &Engine) -> Result<Option<Self>> {
        Self::open(engine, "quantize-stats")
    }
    pub(super) fn evidence_directory(&self) -> &Path {
        &self.directory
    }
    pub(super) fn client(&self) -> QuantClient {
        QuantClient {
            session: Arc::clone(&self.session),
        }
    }
    pub(super) fn finish(mut self) -> Result<()> {
        let requests = self.session.finish()?;
        if !self.cleanup {
            fs::write(
                self.directory.join("session.json"),
                serde_json::to_vec_pretty(&json!({
                    "SchemaVersion": 1,
                    "Provider": "independent-rust-codec-buffers",
                    "Requests": requests,
                    "Completed": true,
                }))?,
            )?;
        }
        self.finished = true;
        if self.cleanup {
            let _ = fs::remove_dir_all(&self.directory);
        }
        Ok(())
    }
}
impl Drop for QuantSession {
    fn drop(&mut self) {
        if self.cleanup && !self.finished {
            let _ = fs::remove_dir_all(&self.directory);
        }
    }
}
#[derive(Clone)]
pub(super) struct Driver {
    pub(super) client: QuantClient,
    pub(super) directory: PathBuf,
    serial: Arc<AtomicU64>,
}
#[derive(Clone)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "Independent production trait function availability is copied without inferred capabilities."
)]
pub(super) struct Type {
    pub(super) id: u32,
    pub(super) name: Option<String>,
    pub(super) block: usize,
    pub(super) size: usize,
    pub(super) reference: bool,
    pub(super) cpu: bool,
    pub(super) decoder: bool,
    pub(super) dot: bool,
    pub(super) dot_operand: u32,
}
pub(super) fn flag(v: &Value, name: &str) -> Result<bool> {
    field(v, name)?
        .as_bool()
        .ok_or_else(|| invalid(format!("quantization {name} boolean missing")))
}
pub(super) fn integer(v: &Value, name: &str) -> Result<i64> {
    field(v, name)?
        .as_i64()
        .ok_or_else(|| invalid(format!("quantization {name} integer missing")))
}
pub(super) fn verify(value: bool, message: impl Into<String>) -> Result<()> {
    if value { Ok(()) } else { Err(invalid(message)) }
}
impl Driver {
    pub(super) fn new(client: QuantClient, directory: &Path) -> Self {
        Self {
            client,
            directory: directory.into(),
            serial: Arc::new(AtomicU64::new(0)),
        }
    }
    #[expect(
        clippy::needless_pass_by_value,
        reason = "Retains the existing Driver::call signature for quantize_perf and quantize_stats."
    )]
    pub(super) fn call(&self, worker: u32, operation: Value) -> Result<Value> {
        self.client.call(worker, &operation)
    }
    pub(super) fn null(&self, worker: u32, operation: Value) -> Result<()> {
        verify(
            self.call(worker, operation)?.is_null(),
            "quantization void reply differs",
        )
    }
    pub(super) fn handle(&self, operation: Value, name: &str) -> Result<u64> {
        let value = self.call(0, operation)?;
        exact_fields(&value, &[name])?;
        let id = uint(&value, name)?;
        verify(id > 0, "null quantization production handle")?;
        Ok(id)
    }
    pub(super) fn path(&self, label: &str) -> PathBuf {
        self.directory.join(format!(
            "{label}-{}.bin",
            self.serial.fetch_add(1, Ordering::Relaxed)
        ))
    }
    pub(super) fn inventory(&self) -> Result<Vec<Type>> {
        let value = self.call(0, json!({"Op":"qb-inventory"}))?;
        exact_fields(&value, &["Types"])?;
        let mut types = Vec::new();
        for t in array(&value, "Types")? {
            exact_fields(
                t,
                &[
                    "Type",
                    "Name",
                    "BlockSize",
                    "TypeSize",
                    "FromFloatReference",
                    "FromFloatCpu",
                    "ToFloat",
                    "VecDot",
                    "VecDotType",
                ],
            )?;
            types.push(Type {
                id: u32::try_from(uint(t, "Type")?)?,
                name: if t["Name"].is_null() {
                    None
                } else {
                    Some(
                        field(t, "Name")?
                            .as_str()
                            .ok_or_else(|| invalid("quantization name text missing"))?
                            .into(),
                    )
                },
                block: usize::try_from(uint(t, "BlockSize")?)?,
                size: usize::try_from(uint(t, "TypeSize")?)?,
                reference: flag(t, "FromFloatReference")?,
                cpu: flag(t, "FromFloatCpu")?,
                decoder: flag(t, "ToFloat")?,
                dot: flag(t, "VecDot")?,
                dot_operand: u32::try_from(uint(t, "VecDotType")?)?,
            });
        }
        verify(
            !types.is_empty() && types.len() <= 1024,
            "quantization type extent differs",
        )?;
        for (index, t) in types.iter().enumerate() {
            verify(
                usize::try_from(t.id)? == index,
                "quantization inventory enum order differs",
            )?;
        }
        Ok(types)
    }
    pub(super) fn buffer(&self, bytes: usize, alignment: usize, offset: usize) -> Result<u64> {
        self.handle(
            json!({"Op":"qb-buffer-new","Bytes":bytes,"Alignment":alignment,"Offset":offset}),
            "Buffer",
        )
    }
    pub(super) fn free(&self, buffer: u64) -> Result<()> {
        self.null(0, json!({"Op":"qb-buffer-free","Buffer":buffer}))
    }
    pub(super) fn load(&self, buffer: u64, file: &Path, bytes: usize) -> Result<()> {
        self.null(0,json!({"Op":"qb-buffer-load","Buffer":buffer,"Offset":0,"File":file,"FileOffset":0,"Bytes":bytes,"Truncate":false}))
    }
    pub(super) fn trace(&self, actions: &[Value]) -> Result<Vec<Value>> {
        let path = self.path("quantize-actions");
        let encoded = serde_json::to_vec(actions)?;
        verify(
            encoded.len() <= 128 * 1024 * 1024,
            "quantization explicit trace exceeds file bound",
        )?;
        // File is unbuffered: streaming JSON into it emits a native write for
        // almost every key, delimiter and digit. Publish the bounded batch with
        // one write instead of thousands of tiny writes.
        fs::write(&path, encoded)?;
        let value = self.call(0, json!({"Op":"qb-trace","ActionsFile":path}))?;
        exact_fields(&value, &["Trace"])?;
        let trace = array(&value, "Trace")?;
        verify(
            trace.len() == actions.len(),
            "quantization trace length differs",
        )?;
        Ok(trace.to_vec())
    }
    pub(super) fn release(&self, kind: &str, id: u64) -> Result<()> {
        self.null(0, json!({"Op":"release","Kind":kind,"Handle":id}))
    }
}
pub(super) fn pointer(buffer: u64, offset: usize) -> Value {
    json!({"Buffer":buffer,"Offset":offset})
}
pub(super) fn call(
    function: &str,
    type_id: u32,
    count: usize,
    input: Value,
    output: Value,
) -> Value {
    let mut value = json!({"Op":"qb-call","Function":function,"Type":type_id,"Count":count});
    value["Input"] = input;
    value["Output"] = output;
    value
}
pub(super) fn row_size(t: &Type, count: usize) -> Result<usize> {
    verify(
        t.block > 0 && count.is_multiple_of(t.block),
        "source size violates selected quantizer block precondition",
    )?;
    count
        .checked_div(t.block)
        .and_then(|n| n.checked_mul(t.size))
        .ok_or_else(|| invalid("quantization row byte overflow"))
}
