//! Persistent vocabulary transport over the selected production runtime.
//! All source schedules and predicates remain in the original adapters.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write as _;
use std::path::Path;
use std::sync::{Arc, Mutex};

use bonsai_gguf::{MetadataArray, MetadataValue};
use bonsai_prism_runtime::tokenizer::gguf::GgufTokenizer;
use bonsai_prism_runtime::unicode::{codepoint_to_utf8, codepoints_from_utf8};
use serde_json::{Value, json};

use super::super::engine::Engine;
use super::super::llama_grammar::{array, exact_fields, field, uint};
use super::super::model_session::{self as native, Reply};
use super::super::{Result, invalid};

pub(in super::super) struct RustState {
    models: HashMap<u64, Arc<GgufTokenizer>>,
    contexts: HashMap<u64, Arc<GgufTokenizer>>,
    next: u64,
    requests: u64,
    request_bytes: usize,
    response_bytes: usize,
    input: File,
    output: File,
    directory: std::path::PathBuf,
    closed: bool,
}

#[derive(Clone)]
pub(in super::super) enum Client {
    Native(native::Client),
    Rust(Arc<Mutex<RustState>>),
}

pub(in super::super) enum ModelSession {
    Native(native::ModelSession),
    Rust(Arc<Mutex<RustState>>),
}

fn lock(state: &Mutex<RustState>) -> Result<std::sync::MutexGuard<'_, RustState>> {
    state
        .lock()
        .map_err(|_| invalid("Rust vocabulary session lock poisoned"))
}

fn boolean(value: &Value, key: &str) -> Result<bool> {
    field(value, key)?
        .as_bool()
        .ok_or_else(|| invalid(format!("vocabulary {key} must be BOOL")))
}

fn hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .flat_map(|byte| [byte >> 4, byte & 15])
        .map(|value| char::from(b"0123456789abcdef"[usize::from(value)]))
        .collect()
}

fn unhex(value: &Value) -> Result<Vec<u8>> {
    let input = value
        .as_str()
        .ok_or_else(|| invalid("vocabulary input must be hex text"))?;
    if input.len() > 2 * 1024 * 1024
        || input.len() % 2 != 0
        || !input
            .bytes()
            .all(|value| value.is_ascii_digit() || (b'a'..=b'f').contains(&value))
    {
        return Err(invalid("vocabulary hex outside bounds"));
    }
    input
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| Ok(u8::from_str_radix(std::str::from_utf8(pair)?, 16)?))
        .collect()
}

impl ModelSession {
    pub(in super::super) fn open(engine: &Engine, name: &str) -> Result<Option<Self>> {
        if let Some(session) = native::ModelSession::open(engine, name)? {
            return Ok(Some(Self::Native(session)));
        }
        let directory = engine.evidence_path(name)?;
        Ok(Some(Self::Rust(Arc::new(Mutex::new(RustState {
            models: HashMap::new(),
            contexts: HashMap::new(),
            next: 1,
            requests: 0,
            request_bytes: 0,
            response_bytes: 0,
            input: File::create(directory.join("requests.jsonl"))?,
            output: File::create(directory.join("responses.jsonl"))?,
            directory,
            closed: false,
        })))))
    }

    pub(in super::super) fn client(&self) -> Client {
        match self {
            Self::Native(session) => Client::Native(session.client()),
            Self::Rust(state) => Client::Rust(Arc::clone(state)),
        }
    }

    pub(in super::super) fn finish(self) -> Result<()> {
        match self {
            Self::Native(session) => session.finish(),
            Self::Rust(state) => {
                let mut state = lock(&state)?;
                if !state.models.is_empty() || !state.contexts.is_empty() || state.closed {
                    return Err(invalid(
                        "Rust vocabulary session has live handles or is closed",
                    ));
                }
                state.closed = true;
                let path = state.directory.join("session.json");
                let record = serde_json::to_vec_pretty(
                    &json!({"SchemaVersion":1,"Runtime":"safe-rust-production-vocabulary","Requests":state.requests,"Completed":true}),
                )?;
                drop(state);
                fs::write(path, record)?;
                Ok(())
            }
        }
    }
}

impl Client {
    pub(in super::super) fn call(
        &self,
        worker: u32,
        operation: Value,
        progress: Option<fn(f32) -> bool>,
    ) -> Result<Reply> {
        match self {
            Self::Native(client) => client.call(worker, operation, progress),
            Self::Rust(state) => {
                if worker >= 16 || progress.is_some() {
                    return Err(invalid(
                        "Rust vocabulary session worker/callback unsupported",
                    ));
                }
                let mut state = lock(state)?;
                if state.closed || state.requests >= 65536 {
                    return Err(invalid("Rust vocabulary session is closed or exhausted"));
                }
                state.requests += 1;
                let request = state.requests;
                let frame = serde_json::to_vec(
                    &json!({"Id":request,"Worker":worker,"Operation":operation}),
                )?;
                if frame.len() > 16 * 1024 * 1024
                    || state.request_bytes + frame.len() > 64 * 1024 * 1024
                {
                    return Err(invalid(
                        "Rust vocabulary request frame/stream bound exceeded",
                    ));
                }
                state.request_bytes += frame.len();
                state.input.write_all(&frame)?;
                state.input.write_all(b"\n")?;
                let value = state.execute(&operation)?;
                let frame = serde_json::to_vec(&json!({"Id":request,"Result":value}))?;
                if frame.len() > 16 * 1024 * 1024
                    || state.response_bytes + frame.len() > 64 * 1024 * 1024
                {
                    return Err(invalid(
                        "Rust vocabulary response frame/stream bound exceeded",
                    ));
                }
                state.response_bytes += frame.len();
                state.output.write_all(&frame)?;
                state.output.write_all(b"\n")?;
                drop(state);
                Ok(Reply {
                    value,
                    progress: Vec::new(),
                })
            }
        }
    }
}

impl RustState {
    fn handle(&mut self) -> Result<u64> {
        if self.next > 1_000_000 || self.models.len() + self.contexts.len() >= 256 {
            return Err(invalid("Rust vocabulary handles exceed bound"));
        }
        let value = self.next;
        self.next += 1;
        Ok(value)
    }

    fn execute(&mut self, operation: &Value) -> Result<Value> {
        let op = field(operation, "Op")?
            .as_str()
            .ok_or_else(|| invalid("vocabulary operation name missing"))?;
        match op {
            "backend-init" | "backend-free" => {
                exact_fields(operation, &["Op"])?;
                Ok(Value::Null)
            }
            "hardware-concurrency" => {
                exact_fields(operation, &["Op"])?;
                Ok(json!({"Count":std::thread::available_parallelism()?.get()}))
            }
            "model-load" => {
                exact_fields(
                    operation,
                    &["Op", "Path", "Base", "Params", "Overrides", "Progress"],
                )?;
                exact_fields(&operation["Overrides"], &["VocabOnly"])?;
                if operation["Overrides"]["VocabOnly"] != true
                    || operation["Base"] != "default"
                    || !operation["Params"].is_null()
                    || operation["Progress"] != false
                {
                    return Err(invalid(
                        "Rust vocabulary session requires vocabulary-only model load",
                    ));
                }
                let path = Path::new(
                    field(operation, "Path")?
                        .as_str()
                        .ok_or_else(|| invalid("vocabulary model path missing"))?,
                );
                if !path.is_absolute() {
                    return Err(invalid("vocabulary model path must be absolute"));
                }
                let metadata = super::super::engine_vocab::metadata(path)?;
                let Some(MetadataArray::String(tokens)) = metadata
                    .get("tokenizer.ggml.tokens")
                    .and_then(MetadataValue::as_array)
                else {
                    return Err(invalid("vocabulary token metadata missing"));
                };
                let tokenizer = Arc::new(GgufTokenizer::from_metadata(&metadata, tokens.len())?);
                let handle = self.handle()?;
                self.models.insert(handle, tokenizer);
                Ok(json!({"Model":handle}))
            }
            "context" => {
                exact_fields(operation, &["Op", "Model", "Params", "Overrides"])?;
                if !operation["Params"].is_null()
                    || !operation["Overrides"].is_object()
                    || operation["Overrides"]
                        .as_object()
                        .is_some_and(|v| v.keys().any(|key| key != "Context"))
                {
                    return Err(invalid("Rust vocabulary context parameters unsupported"));
                }
                let model = self
                    .models
                    .get(&uint(operation, "Model")?)
                    .cloned()
                    .ok_or_else(|| invalid("unknown Rust vocabulary model handle"))?;
                let handle = self.handle()?;
                self.contexts.insert(handle, model);
                Ok(json!({"Context":handle}))
            }
            "vocab" => {
                exact_fields(operation, &["Op", "Model"])?;
                let model = self
                    .models
                    .get(&uint(operation, "Model")?)
                    .ok_or_else(|| invalid("unknown vocabulary handle"))?;
                Ok(
                    json!({"Count":model.vocab_size(),"Bos":model.bos_token().map_or(-1, i64::from),"Type":model.vocabulary_type()}),
                )
            }
            "release" => {
                exact_fields(operation, &["Op", "Kind", "Handle"])?;
                let table = match field(operation, "Kind")?.as_str() {
                    Some("model") => &mut self.models,
                    Some("context") => &mut self.contexts,
                    _ => return Err(invalid("Rust vocabulary release kind unsupported")),
                };
                table
                    .remove(&uint(operation, "Handle")?)
                    .ok_or_else(|| invalid("Rust vocabulary release handle missing"))?;
                Ok(Value::Null)
            }
            "vocab-tokenize"
            | "vocab-detokenize"
            | "raw-vocab-detokenize"
            | "unicode-encode"
            | "unicode-decode" => self.batch(operation, op),
            other => Err(invalid(format!(
                "Rust vocabulary session operation {other} unsupported"
            ))),
        }
    }

    #[allow(
        clippy::too_many_lines,
        reason = "Keep primitive request validation and per-item error framing together"
    )]
    fn batch(&self, operation: &Value, op: &str) -> Result<Value> {
        let key = match op {
            "vocab-tokenize" => {
                exact_fields(
                    operation,
                    &["Op", "Context", "InputsHex", "AddSpecial", "ParseSpecial"],
                )?;
                "InputsHex"
            }
            "vocab-detokenize" => {
                exact_fields(operation, &["Op", "Context", "TokenLists", "Special"])?;
                "TokenLists"
            }
            "raw-vocab-detokenize" => {
                exact_fields(
                    operation,
                    &[
                        "Op",
                        "Context",
                        "TokenLists",
                        "RemoveSpecial",
                        "UnparseSpecial",
                    ],
                )?;
                "TokenLists"
            }
            "unicode-encode" => {
                exact_fields(operation, &["Op", "CodePoints"])?;
                "CodePoints"
            }
            _ => {
                exact_fields(operation, &["Op", "InputsHex"])?;
                "InputsHex"
            }
        };
        let tokenizer = if op.starts_with("vocab-") || op == "raw-vocab-detokenize" {
            Some(
                self.contexts
                    .get(&uint(operation, "Context")?)
                    .ok_or_else(|| invalid("Rust vocabulary context missing"))?,
            )
        } else {
            None
        };
        let items = array(operation, key)?;
        if items.len() > 512 {
            return Err(invalid("Rust vocabulary batch exceeds512"));
        }
        let mut values = Vec::new();
        let mut errors = Vec::new();
        let mut supplied = 0;
        let mut returned = 0;
        for item in items {
            let result = (|| -> Result<Value> {
                let value = match op {
                    "unicode-encode" => json!(hex(&codepoint_to_utf8(u32::try_from(
                        item.as_u64()
                            .ok_or_else(|| invalid("Unicode point must be unsigned"))?
                    )?)?)),
                    "unicode-decode" => {
                        let bytes = unhex(item)?;
                        supplied += bytes.len();
                        json!(codepoints_from_utf8(&bytes)?)
                    }
                    "vocab-tokenize" => {
                        let bytes = unhex(item)?;
                        supplied += bytes.len();
                        json!(
                            tokenizer
                                .ok_or_else(|| invalid("tokenizer missing"))?
                                .encode(
                                    &bytes,
                                    boolean(operation, "AddSpecial")?,
                                    boolean(operation, "ParseSpecial")?
                                )?
                        )
                    }
                    _ => {
                        let tokens = item
                            .as_array()
                            .ok_or_else(|| invalid("vocabulary token list missing"))?;
                        if tokens.len() > 65536 {
                            return Err(invalid("vocabulary token list exceeds bound"));
                        }
                        supplied += tokens.len() * 4;
                        let tokens =
                            tokens
                                .iter()
                                .map(|id| {
                                    u32::try_from(id.as_u64().ok_or_else(|| {
                                        invalid("vocabulary token must be unsigned")
                                    })?)
                                    .map_err(Into::into)
                                })
                                .collect::<Result<Vec<_>>>()?;
                        let (remove, special) = if op == "vocab-detokenize" {
                            (false, boolean(operation, "Special")?)
                        } else {
                            (
                                boolean(operation, "RemoveSpecial")?,
                                boolean(operation, "UnparseSpecial")?,
                            )
                        };
                        json!(hex(&tokenizer
                            .ok_or_else(|| invalid("tokenizer missing"))?
                            .decode(&tokens, remove, special)?))
                    }
                };
                if supplied > 1024 * 1024 {
                    return Err(invalid("vocabulary supplied payload exceeds bound"));
                }
                returned += serde_json::to_vec(&value)?.len();
                if returned > 8 * 1024 * 1024 {
                    return Err(invalid("vocabulary returned payload exceeds bound"));
                }
                Ok(value)
            })();
            match result {
                Ok(value) => {
                    values.push(value);
                    errors.push(Value::Null);
                }
                Err(error) => {
                    values.push(Value::Null);
                    errors.push(json!({"Kind":"standard","Message":error.to_string()}));
                }
            }
        }
        Ok(json!({"Values":values,"Errors":errors}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires immutable original vocabulary fixtures in PRISM_TOKENIZER_SOURCE"]
    fn focused_production_vocabulary_session() -> Result<()> {
        let source = std::env::var("PRISM_TOKENIZER_SOURCE")?;
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let evidence = std::env::temp_dir().join(format!(
            "Prism-tokenizer-session-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&evidence)?;
        let engine = Engine::new(
            super::super::super::engine::Target::Rust,
            None,
            None,
            Some(evidence),
            Some(source.clone().into()),
            None,
        )?;
        for name in ["gpt-2", "llama-spm", "aquila"] {
            let session =
                ModelSession::open(&engine, name)?.ok_or("Rust vocabulary session missing")?;
            let client = session.client();
            let call = |operation| -> Result<Value> { Ok(client.call(0, operation, None)?.value) };
            call(json!({"Op":"backend-init"}))?;
            let path = Path::new(&source).join(format!("models/ggml-vocab-{name}.gguf"));
            let model = call(json!({"Op":"model-load","Path":path,"Base":"default","Params":null,"Overrides":{"VocabOnly":true},"Progress":false}))?["Model"].as_u64().ok_or("model missing")?;
            let context = call(json!({"Op":"context","Model":model,"Params":null,"Overrides":{}}))?
                ["Context"]
                .as_u64()
                .ok_or("context missing")?;
            let encoded = call(
                json!({"Op":"vocab-tokenize","Context":context,"InputsHex":[hex(b"Hello world")],"AddSpecial":false,"ParseSpecial":false}),
            )?;
            assert_eq!(encoded["Errors"], json!([null]));
            let decoded = call(
                json!({"Op":"vocab-detokenize","Context":context,"TokenLists":encoded["Values"],"Special":true}),
            )?;
            assert_eq!(decoded["Values"], json!([hex(b"Hello world")]));
            if name == "llama-spm" {
                let encoded = call(
                    json!({"Op":"vocab-tokenize","Context":context,"InputsHex":["80"],"AddSpecial":false,"ParseSpecial":true}),
                )?;
                assert_eq!(encoded["Errors"], json!([null]));
                let decoded = call(
                    json!({"Op":"vocab-detokenize","Context":context,"TokenLists":encoded["Values"],"Special":true}),
                )?;
                assert_eq!(decoded["Values"], json!(["80"]));
            }
            assert_eq!(
                call(json!({"Op":"unicode-decode","InputsHex":["c080","edb080"]}))?["Values"],
                json!([[0], [0xdc00]])
            );
            call(json!({"Op":"release","Kind":"context","Handle":context}))?;
            call(json!({"Op":"release","Kind":"model","Handle":model}))?;
            call(json!({"Op":"backend-free"}))?;
            session.finish()?;
        }
        Ok(())
    }
}
