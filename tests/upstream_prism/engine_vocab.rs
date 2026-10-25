//! Adapters to the runtime's existing production vocabulary APIs.
//!
//! A Rust vocabulary context means successful construction of the production
//! `GgufTokenizer`, including family-specific insertion declarations. It makes
//! no inference-context claim. Metadata is never relabeled;
//! unsupported declarations and API limits return failed operation metadata.

use std::fmt::Write as _;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::thread;
use std::time::Instant;

use bonsai_gguf::{Metadata, MetadataArray};

use bonsai_prism_runtime::tokenizer::gguf::GgufTokenizer;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::llama_grammar::{array, exact_fields, field, uint};
use super::{Result, invalid};

fn boolean(value: &Value, name: &str) -> Result<bool> {
    field(value, name)?
        .as_bool()
        .ok_or_else(|| invalid(format!("vocabulary {name} must be boolean")))
}

fn kind(value: &Value) -> Result<&str> {
    field(value, "Kind")?
        .as_str()
        .ok_or_else(|| invalid("vocabulary Kind must be text"))
}

fn validate(operation: &Value) -> Result<()> {
    if kind(operation)? == "vocab-runtime-info" {
        return exact_fields(operation, &["Kind"]);
    }
    exact_fields(
        operation,
        &["Kind", "Path", "CreateContext", "Inputs", "Workers"],
    )?;
    let path = field(operation, "Path")?
        .as_str()
        .ok_or_else(|| invalid("vocabulary Path must be text"))?;
    if !Path::new(path).is_absolute() || path.len() > 32768 || path.contains('\0') {
        return Err(invalid("vocabulary Path must be absolute and bounded"));
    }
    boolean(operation, "CreateContext")?;
    let inputs = array(operation, "Inputs")?;
    let workers = array(operation, "Workers")?;
    if inputs.len() > 1024 || workers.len() > 256 {
        return Err(invalid("vocabulary request extent exceeds bound"));
    }
    let mut supplied = 0;
    for input in inputs {
        exact_fields(input, &["Bytes", "AddSpecial", "ParseSpecial"])?;
        boolean(input, "AddSpecial")?;
        boolean(input, "ParseSpecial")?;
        let bytes = array(input, "Bytes")?;
        supplied += bytes.len();
        if bytes.len() > 65536 || supplied > 1024 * 1024 {
            return Err(invalid("vocabulary input bytes exceed bound"));
        }
        for byte in bytes {
            u8::try_from(
                byte.as_u64()
                    .ok_or_else(|| invalid("vocabulary byte must be unsigned"))?,
            )?;
        }
    }
    let mut actions = 0;
    let mut captured = 0;
    for worker in workers {
        exact_fields(worker, &["Actions"])?;
        let schedule = array(worker, "Actions")?;
        actions += schedule.len();
        if schedule.len() > 1024 || actions > 262_144 {
            return Err(invalid("vocabulary worker schedule exceeds bound"));
        }
        for action in schedule {
            match kind(action)? {
                "clock" => exact_fields(action, &["Kind"])?,
                "tokenize" => {
                    exact_fields(action, &["Kind", "Input", "Capture", "Detokenize"])?;
                    boolean(action, "Detokenize")?;
                    if boolean(action, "Capture")? {
                        captured += 1;
                    }
                    if usize::try_from(uint(action, "Input")?)? >= inputs.len() || captured > 4096 {
                        return Err(invalid("vocabulary input/capture outside extent"));
                    }
                }
                _ => return Err(invalid("unknown vocabulary action")),
            }
        }
    }
    Ok(())
}

pub(super) fn metadata(path: &Path) -> Result<Metadata> {
    let mut file = File::open(path)?;
    let bytes = file.metadata()?.len();
    if !file.metadata()?.is_file() || bytes > 64 * 1024 * 1024 {
        return Err(invalid("vocabulary artifact is not a bounded regular file"));
    }
    let mut hash = Sha256::new();
    let mut buffer = [0; 16384];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    let mut digest = String::with_capacity(64);
    for byte in hash.finalize() {
        write!(digest, "{byte:02x}")?;
    }
    bonsai_gguf::read_vocabulary(path, bytes, &digest, 128 * 1024 * 1024)
}

struct Objects {
    tokenizer: GgufTokenizer,
}

fn objects(metadata: &Metadata, _insert_specials: bool) -> Result<Objects> {
    let Some(MetadataArray::String(tokens)) = metadata
        .get("tokenizer.ggml.tokens")
        .and_then(bonsai_gguf::MetadataValue::as_array)
    else {
        return Err(invalid(
            "production vocabulary requires a STRING token array",
        ));
    };
    Ok(Objects {
        tokenizer: GgufTokenizer::from_metadata(metadata, tokens.len())?,
    })
}
fn input_bytes(input: &Value) -> Result<Vec<u8>> {
    array(input, "Bytes")?
        .iter()
        .map(|byte| {
            Ok(u8::try_from(byte.as_u64().ok_or_else(|| {
                invalid("vocabulary byte must be unsigned")
            })?)?)
        })
        .collect()
}

fn encode(objects: &Objects, input: &Value) -> Result<Vec<u32>> {
    objects.tokenizer.encode(
        &input_bytes(input)?,
        boolean(input, "AddSpecial")?,
        boolean(input, "ParseSpecial")?,
    )
}

fn detokenize(objects: &Objects, tokens: &[u32]) -> Result<String> {
    let bytes = objects.tokenizer.decode(tokens, false, true)?;
    Ok(bytes
        .into_iter()
        .flat_map(|byte| [byte >> 4, byte & 15])
        .map(|nibble| char::from(b"0123456789abcdef"[usize::from(nibble)]))
        .collect())
}
fn worker(
    objects: &Objects,
    inputs: &[Value],
    schedule: &Value,
    started: Instant,
) -> Result<Value> {
    let actions = array(schedule, "Actions")?;
    let mut results = Vec::new();
    let mut captured = 0;
    for action in actions {
        if kind(action)? == "clock" {
            results.push(json!({"Microseconds":i64::try_from(started.elapsed().as_micros())?}));
            continue;
        }
        let input = &inputs[usize::try_from(uint(action, "Input")?)?];
        let tokens = encode(objects, input)?;
        let text = if boolean(action, "Detokenize")? {
            Some(detokenize(objects, &tokens)?)
        } else {
            None
        };
        if boolean(action, "Capture")? {
            captured += tokens.len();
            if captured > 1_048_576 {
                return Err(invalid("captured production token extent exceeds bound"));
            }
            results.push(json!({"Tokens":tokens,"TextHex":text}));
        } else {
            results.push(json!({}));
        }
    }
    Ok(json!({"Completed":actions.len(),"Results":results}))
}

fn evaluate(operation: &Value) -> Result<Value> {
    let requested_context = boolean(operation, "CreateContext")?;
    let mut response = json!({"ModelLoaded":false,"ContextLoaded":if requested_context {json!(false)} else {Value::Null},"Workers":[],"Error":null});
    let path = Path::new(
        field(operation, "Path")?
            .as_str()
            .ok_or_else(|| invalid("vocabulary path must be text"))?,
    );
    let metadata = match metadata(path) {
        Ok(value) => value,
        Err(error) => {
            response["Error"] = json!(error.to_string());
            return Ok(response);
        }
    };
    response["ModelLoaded"] = json!(true);
    let inputs = array(operation, "Inputs")?;
    let insert_specials = inputs.iter().any(|input| input["AddSpecial"] == true);
    let objects = match objects(&metadata, insert_specials) {
        Ok(value) => value,
        Err(error) => {
            response["Error"] = json!(error.to_string());
            return Ok(response);
        }
    };
    if requested_context {
        response["ContextLoaded"] = json!(true);
    }
    let started = Instant::now();
    let workers = array(operation, "Workers")?;
    let observations = thread::scope(|scope| {
        let mut handles = Vec::new();
        for schedule in workers {
            let objects = &objects;
            handles.push(scope.spawn(move || worker(objects, inputs, schedule, started)));
        }
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .map_err(|_| invalid("production Rust tokenizer worker panicked"))?
            })
            .collect::<Result<Vec<_>>>()
    });
    match observations {
        Ok(values) => response["Workers"] = json!(values),
        Err(error) => response["Error"] = json!(error.to_string()),
    }
    Ok(response)
}

pub(super) fn execute(request: &Value) -> Result<Option<Value>> {
    exact_fields(request, &["SchemaVersion", "Operations"])?;
    let operations = array(request, "Operations")?;
    if uint(request, "SchemaVersion")? != 1 || operations.is_empty() || operations.len() > 128 {
        return Err(invalid("Rust vocabulary request framing invalid"));
    }
    for operation in operations {
        if !matches!(kind(operation)?, "vocab-runtime-info" | "vocab-evaluate") {
            return Ok(None);
        }
    }
    for operation in operations {
        validate(operation)?;
    }
    let results = operations
        .iter()
        .map(|operation| {
            if kind(operation)? == "vocab-runtime-info" {
                Ok(json!({"HardwareConcurrency":thread::available_parallelism()?.get()}))
            } else {
                evaluate(operation)
            }
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(json!({"SchemaVersion":1,"Results":results})))
}
