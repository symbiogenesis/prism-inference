//! Rust-owned schedules for all five pinned model lifecycle source tests.
//!
//! Production operations use native sessions or the owned Rust model loader.
//! Pinned source: b87ca7225456b06b0975b8239c66b5aecf07e37c (MIT).

use super::engine::Engine;
use super::model_session::{Client, ModelSession, Reply};
use super::{Result, invalid, result};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::io::Read;
use std::path::Path;

const ASSET: &str = "tinyllamas/stories15M-q4_0.gguf";
const ASSET_LENGTH: u64 = 19_077_344;
const ASSET_SHA: &str = "66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739";
const STATE: &str = "dump_state.bin";
const PROMPT: &str = "The meaning of life is";
const AUTORELEASE: &str = "tests/test-autorelease.cpp";
const CANCEL: &str = "tests/test-model-load-cancel.cpp";
const SAVE: &str = "tests/test-save-load-state.cpp";
const FRAGMENTED: &str = "tests/test-state-restore-fragmented.cpp";
const THREADS: &str = "tests/test-thread-safety.cpp";

mod storage;

type ProgressPolicy = fn(f32) -> bool;

#[derive(Default)]
struct Checks {
    source: u64,
    guards: u64,
}
impl Checks {
    fn source(&mut self, passed: bool, message: &str) -> Result<()> {
        self.source += 1;
        verify(passed, message)
    }
    fn guard(&mut self, passed: bool, message: &str) -> Result<()> {
        self.guards += 1;
        verify(passed, message)
    }
}
fn verify(passed: bool, message: &str) -> Result<()> {
    if passed {
        Ok(())
    } else {
        Err(invalid(message))
    }
}
fn verify_artifact(path: &Path, length: u64, sha256: &str) -> Result<()> {
    let mut file = std::fs::File::open(path)?;
    verify(
        file.metadata()?.len() == length,
        "model artifact length differs from its lock",
    )?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 8192];
    let mut total = 0_u64;
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        total = total
            .checked_add(u64::try_from(count)?)
            .ok_or_else(|| invalid("model artifact length overflow"))?;
        verify(total <= length, "model artifact grew while hashing")?;
        hash.update(&buffer[..count]);
    }
    let mut actual = String::with_capacity(64);
    for byte in hash.finalize() {
        write!(&mut actual, "{byte:02x}")?;
    }
    verify(
        total == length && actual == sha256,
        "model artifact SHA-256 differs from its lock",
    )
}
fn invoke(client: &Client, worker: u32, operation: Value) -> Result<Value> {
    Ok(client.call(worker, operation, None)?.value)
}
fn number(value: &Value, field: &str) -> Result<u64> {
    value[field]
        .as_u64()
        .ok_or_else(|| invalid(format!("missing unsigned {field}")))
}
fn handle(value: &Value, field: &str) -> Result<u64> {
    let id = number(value, field)?;
    verify(id > 0, "null/zero production handle")?;
    Ok(id)
}
fn boolean(value: &Value, field: &str) -> Result<bool> {
    value[field]
        .as_bool()
        .ok_or_else(|| invalid(format!("missing boolean {field}")))
}
fn token(value: &Value) -> Result<i32> {
    i32::try_from(
        value["Token"]
            .as_i64()
            .ok_or_else(|| invalid("missing token"))?,
    )
    .map_err(Into::into)
}
fn tokens(value: &Value) -> Result<Vec<i32>> {
    value["Tokens"]
        .as_array()
        .ok_or_else(|| invalid("missing token list"))?
        .iter()
        .map(|v| {
            i32::try_from(v.as_i64().ok_or_else(|| invalid("invalid token"))?).map_err(Into::into)
        })
        .collect()
}
fn release(client: &Client, worker: u32, kind: &str, id: u64) -> Result<()> {
    invoke(
        client,
        worker,
        json!({"Op":"release","Kind":kind,"Handle":id}),
    )?;
    Ok(())
}
fn plain(client: &Client, worker: u32, op: &str) -> Result<()> {
    invoke(client, worker, json!({"Op":op}))?;
    Ok(())
}
fn params(client: &Client, initial: Value, arguments: Value, checks: &mut Checks) -> Result<u64> {
    let mut operation = json!({"Op":"params"});
    operation["Initial"] = initial;
    operation["Arguments"] = arguments;
    let value = invoke(client, 0, operation)?;
    checks.source(
        boolean(&value, "Parsed")?,
        "common_params_parse rejected original arguments",
    )?;
    handle(&value, "Handle")
}
fn context(
    client: &Client,
    worker: u32,
    model: u64,
    params: Option<u64>,
    overrides: Value,
) -> Result<u64> {
    let mut operation = json!({"Op":"context","Model":model,"Params":params});
    operation["Overrides"] = overrides;
    handle(&invoke(client, worker, operation)?, "Context")
}
fn dist(client: &Client, worker: u32, model: u64) -> Result<u64> {
    handle(
        &invoke(
            client,
            worker,
            json!({"Op":"sampler","Model":model,"Kind":"dist","Seed":1234,"Params":null}),
        )?,
        "Sampler",
    )
}
fn batch_new(client: &Client, worker: u32, capacity: usize) -> Result<u64> {
    handle(
        &invoke(
            client,
            worker,
            json!({"Op":"batch-new","Capacity":capacity,"Sequences":1}),
        )?,
        "Batch",
    )
}
fn batch_one(
    client: &Client,
    worker: u32,
    batch: Option<u64>,
    input: &[i32],
    positions: Option<&[i32]>,
) -> Result<u64> {
    handle(
        &invoke(
            client,
            worker,
            json!({"Op":"batch-one","Batch":batch,"Tokens":input,"Positions":positions}),
        )?,
        "Batch",
    )
}
fn decode(client: &Client, worker: u32, context: u64, batch: u64) -> Result<i64> {
    invoke(
        client,
        worker,
        json!({"Op":"decode","Context":context,"Batch":batch}),
    )?["Code"]
        .as_i64()
        .ok_or_else(|| invalid("missing decode result"))
}
fn decode_zero(
    client: &Client,
    worker: u32,
    context: u64,
    batch: u64,
    checks: &mut Checks,
) -> Result<()> {
    checks.source(
        decode(client, worker, context, batch)? == 0,
        "production llama_decode failed",
    )
}
fn row(token: i32, position: usize, sequence: i32, logits: bool) -> Value {
    json!({"Token":token,"Position":position,"Sequences":[sequence],"Logits":logits})
}
fn add_rows(client: &Client, worker: u32, batch: u64, rows: Vec<Value>) -> Result<()> {
    invoke(
        client,
        worker,
        json!({"Op":"batch-add","Batch":batch,"Rows":Value::Array(rows)}),
    )?;
    Ok(())
}
fn sample(client: &Client, worker: u32, sampler: u64, context: u64, index: i64) -> Result<i32> {
    token(&invoke(
        client,
        worker,
        json!({"Op":"sample","Sampler":sampler,"Context":context,"Index":index}),
    )?)
}
fn state_get(
    client: &Client,
    context: u64,
    sequence: i32,
    device: bool,
    raw: bool,
    checks: &mut Checks,
) -> Result<Value> {
    let value = invoke(
        client,
        0,
        json!({"Op":"state-get","Context":context,"Sequence":sequence,"Extended":device,"Flags":if device {2} else {0},"ReturnData":raw}),
    )?;
    checks.source(
        number(&value, "Copied")? == number(&value, "Size")?,
        "sequence-state get copied length differs",
    )?;
    Ok(value)
}
fn state_set(
    client: &Client,
    context: u64,
    blob: &Value,
    sequence: i32,
    device: bool,
    checks: &mut Checks,
) -> Result<()> {
    let value = invoke(
        client,
        0,
        json!({"Op":"state-set","Context":context,"Blob":handle(blob,"Blob")?,"Sequence":sequence,"Extended":device,"Flags":if device {2} else {0}}),
    )?;
    checks.source(
        number(&value, "Copied")? == number(blob, "Size")?,
        "sequence-state set copied length differs",
    )
}

// MSVC's 32-bit uniform_int_distribution uses multiply-high with rejection.
struct Random {
    state: [u32; 624],
    index: usize,
}
impl Random {
    fn new(seed: u32) -> Self {
        let mut state = [0; 624];
        state[0] = seed;
        for (index, increment) in (1..624).zip(1_u32..) {
            state[index] = 1_812_433_253_u32
                .wrapping_mul(state[index - 1] ^ (state[index - 1] >> 30))
                .wrapping_add(increment);
        }
        Self { state, index: 624 }
    }
    fn next(&mut self) -> u32 {
        if self.index == 624 {
            for index in 0..624 {
                let value = (self.state[index] & 0x8000_0000)
                    | (self.state[(index + 1) % 624] & 0x7fff_ffff);
                self.state[index] = self.state[(index + 397) % 624]
                    ^ (value >> 1)
                    ^ if value & 1 == 0 { 0 } else { 0x9908_b0df };
            }
            self.index = 0;
        }
        let mut value = self.state[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^ (value >> 18)
    }
    fn bounded(&mut self, bound: u32) -> Result<i32> {
        verify(bound > 0, "empty model vocabulary")?;
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let product = u64::from(self.next()) * u64::from(bound);
            let low = u32::from_le_bytes(product.to_le_bytes()[..4].try_into()?);
            if low >= threshold {
                return i32::try_from(product >> 32).map_err(Into::into);
            }
        }
    }
}

fn autorelease(client: &Client, path: &str, checks: &mut Checks) -> Result<Value> {
    // Every API, including both backend calls, executes on this one native thread.
    const WORKER: u32 = 1;
    plain(client, WORKER, "backend-init")?;
    let model = handle(
        &invoke(
            client,
            WORKER,
            json!({"Op":"model-load","Path":path,"Base":"default","Params":null,"Overrides":{},"Progress":false}),
        )?,
        "Model",
    )?;
    checks.guard(model > 0, "autorelease model is null")?;
    let ctx = context(client, WORKER, model, None, json!({}))?;
    checks.guard(ctx > 0, "autorelease context is null")?;
    release(client, WORKER, "context", ctx)?;
    release(client, WORKER, "model", model)?;
    plain(client, WORKER, "backend-free")?;
    Ok(json!({"NativeWorker":WORKER,"SameThreadCreationAndDestruction":true}))
}
fn cancel_policy(progress: f32) -> bool {
    progress > 0.50
}
fn load_cancel(
    mut call: impl FnMut(u32, Value, Option<ProgressPolicy>) -> Result<Reply>,
    path: &str,
    checks: &mut Checks,
) -> Result<Value> {
    checks.source(
        std::fs::File::open(path).is_ok(),
        "original model file-open predicate failed",
    )?;
    call(0, json!({"Op":"backend-init"}), None)?;
    let reply = call(
        0,
        json!({"Op":"model-load","Path":path,"Base":"zero","Params":null,"Overrides":{"LoadMode":"none"},"Progress":true}),
        Some(cancel_policy),
    )?;
    call(0, json!({"Op":"backend-free"}), None)?;
    checks.source(
        reply.value["Model"].is_null(),
        "load-cancel returned a model",
    )?;
    checks.guard(
        !reply.progress.is_empty(),
        "load-cancel never invoked the Rust progress policy",
    )?;
    for &(bits, decision) in &reply.progress {
        let progress = f32::from_bits(bits);
        checks.guard(
            progress.is_finite() && (0.0..=1.0).contains(&progress),
            "invalid model-load progress",
        )?;
        checks.guard(
            decision == cancel_policy(progress),
            "progress callback did not use the original >0.50 policy",
        )?;
    }
    checks.guard(
        reply.progress.iter().any(|(_, decision)| !decision),
        "null model had no cancellation decision",
    )?;
    Ok(
        json!({"ProgressCallbacks":reply.progress,"Policy":"progress > 0.50","ZeroInitializedModelParams":true}),
    )
}
fn generate(
    client: &Client,
    ctx: u64,
    sampler: u64,
    mut past: usize,
    sequence: i32,
    checks: &mut Checks,
) -> Result<Vec<i32>> {
    let batch = batch_new(client, 0, 1)?;
    let mut output = Vec::with_capacity(16);
    for _ in 0..16 {
        let next = sample(client, 0, sampler, ctx, -1)?;
        output.push(next);
        invoke(client, 0, json!({"Op":"batch-clear","Batch":batch}))?;
        add_rows(client, 0, batch, vec![row(next, past, sequence, true)])?;
        decode_zero(client, 0, ctx, batch, checks)?;
        past += 1;
    }
    release(client, 0, "batch", batch)?;
    checks.source(!output.is_empty(), "generated token list is empty")?;
    Ok(output)
}
fn replay(
    client: &Client,
    ctx: u64,
    last: i32,
    position: usize,
    checks: &mut Checks,
) -> Result<()> {
    let positions = [i32::try_from(position)?];
    let batch = batch_one(client, 0, None, &[last], Some(&positions))?;
    decode_zero(client, 0, ctx, batch, checks)?;
    release(client, 0, "batch", batch)
}
fn load_state(client: &Client, ctx: u64, prompt: &[i32], checks: &mut Checks) -> Result<usize> {
    let value = invoke(
        client,
        0,
        json!({"Op":"state-load-file","Context":ctx,"Name":STATE,"Capacity":prompt.len()}),
    )?;
    checks.source(boolean(&value, "Loaded")?, "llama_state_load_file failed")?;
    let count = usize::try_from(number(&value, "Count")?)?;
    checks.guard(
        count == prompt.len() && tokens(&value)? == prompt,
        "state-file token metadata changed",
    )?;
    replay(
        client,
        ctx,
        *prompt.last().ok_or_else(|| invalid("empty prompt"))?,
        count
            .checked_sub(1)
            .ok_or_else(|| invalid("empty restored state"))?,
        checks,
    )?;
    Ok(count)
}
fn removal_isolation(
    client: &Client,
    model: u64,
    p: u64,
    prompt: &[i32],
    checks: &mut Checks,
) -> Result<()> {
    // Removal isolation: the same 100-token prefix, separate 0/1 batches.
    let ctx = context(
        client,
        0,
        model,
        Some(p),
        json!({"Context":256,"Sequences":2,"Unified":true}),
    )?;
    checks.source(ctx > 0, "removal-isolation context is null")?;
    for sequence in 0..2 {
        let batch = batch_new(client, 0, prompt.len())?;
        add_rows(
            client,
            0,
            batch,
            prompt
                .iter()
                .enumerate()
                .map(|(i, &t)| row(t, i, sequence, false))
                .collect(),
        )?;
        decode_zero(client, 0, ctx, batch, checks)?;
        release(client, 0, "batch", batch)?;
    }
    let before = state_get(client, ctx, 1, false, true, checks)?;
    checks.source(
        number(&before, "Size")? > 0,
        "isolation sequence state before removal is empty",
    )?;
    let removed = invoke(
        client,
        0,
        json!({"Op":"memory-remove","Context":ctx,"Sequence":0,"From":-1,"To":-1}),
    )?;
    checks.source(
        boolean(&removed, "Removed")?,
        "sequence zero removal failed",
    )?;
    let after = state_get(client, ctx, 1, false, true, checks)?;
    checks.source(
        number(&after, "Size")? > 0,
        "isolation sequence state after removal is empty",
    )?;
    let before_hex = before["DataHex"]
        .as_str()
        .ok_or_else(|| invalid("isolation bytes absent"))?;
    let after_hex = after["DataHex"]
        .as_str()
        .ok_or_else(|| invalid("isolation bytes absent"))?;
    checks.source(
        before_hex == after_hex,
        "removing sequence zero changed sequence one",
    )?;
    for blob in [&before, &after] {
        release(client, 0, "blob", handle(blob, "Blob")?)?;
    }
    release(client, 0, "context", ctx)?;

    Ok(())
}
fn save_load(client: &Client, path: &str, checks: &mut Checks) -> Result<Value> {
    plain(client, 0, "common-init")?;
    let p = params(
        client,
        json!({"Prompt":"","Batch":100,"OutputFile":STATE,"Seed":1234}),
        json!(["test-save-load-state", "-m", path]),
        checks,
    )?;
    invoke(
        client,
        0,
        json!({"Op":"params-unified","Params":p,"Value":true}),
    )?;
    plain(client, 0, "backend-load-all")?;
    let loaded = invoke(
        client,
        0,
        json!({"Op":"common-model","Params":p,"ModelOnly":true}),
    )?;
    checks.source(
        !loaded["Model"].is_null(),
        "common model-only initialization failed",
    )?;
    checks.source(
        loaded["Context"].is_null(),
        "model-only initialization created a context",
    )?;
    let model = handle(&loaded, "Model")?;
    let vocab = invoke(client, 0, json!({"Op":"vocab","Model":model}))?;
    let count = u32::try_from(number(&vocab, "Count")?)?;
    let mut random = Random::new(1234);
    let prompt = (0..100)
        .map(|_| random.bounded(count))
        .collect::<Result<Vec<_>>>()?;
    let ctx = context(client, 0, model, Some(p), json!({}))?;
    let sampler = dist(client, 0, model)?;
    let batch = batch_one(client, 0, None, &prompt[..99], None)?;
    decode_zero(client, 0, ctx, batch, checks)?;
    release(client, 0, "batch", batch)?;
    let saved = invoke(
        client,
        0,
        json!({"Op":"state-save-file","Context":ctx,"Name":STATE,"Tokens":prompt}),
    )?;
    checks.guard(
        boolean(&saved, "Saved")?,
        "production failed to save the baseline state",
    )?;
    replay(client, ctx, prompt[99], 99, checks)?;
    let baseline = generate(client, ctx, sampler, 100, 0, checks)?;
    release(client, 0, "sampler", sampler)?;
    release(client, 0, "context", ctx)?;

    removal_isolation(client, model, p, &prompt, checks)?;

    let mut variants = Vec::new();
    for mode in ["state-load", "seq-copy-host", "seq-copy-device"] {
        let copy = mode != "state-load";
        let device = mode == "seq-copy-device";
        let ctx = context(
            client,
            0,
            model,
            Some(p),
            if copy {
                json!({"Sequences":2})
            } else {
                json!({})
            },
        )?;
        let sampler = dist(client, 0, model)?;
        let past = load_state(client, ctx, &prompt, checks)?;
        if copy {
            let blob = state_get(client, ctx, 0, device, false, checks)?;
            invoke(
                client,
                0,
                json!({"Op":"memory-clear","Context":ctx,"Data":true}),
            )?;
            state_set(client, ctx, &blob, 1, device, checks)?;
            release(client, 0, "blob", handle(&blob, "Blob")?)?;
        }
        let output = generate(client, ctx, sampler, past, i32::from(copy), checks)?;
        checks.source(
            output == baseline,
            "restored generation differs from baseline token IDs",
        )?;
        variants.push(json!({"Mode":mode,"Tokens":output}));
        release(client, 0, "sampler", sampler)?;
        release(client, 0, "context", ctx)?;
    }
    release(client, 0, "model", model)?;
    release(client, 0, "params", p)?;
    Ok(
        json!({"RandomSeed":1234,"PromptTokens":prompt,"Baseline":baseline,"Variants":variants,"IsolationSequences":[0,1],"BaselineKvTokens":99,"StateFileTokenCount":100}),
    )
}
fn fragmented(client: &Client, path: &str, checks: &mut Checks) -> Result<Value> {
    plain(client, 0, "common-init")?;
    let p = params(
        client,
        json!({"Seed":1234,"Unified":true,"Parallel":3,"Context":256}),
        json!(["test-state-restore-fragmented", "-m", path]),
        checks,
    )?;
    plain(client, 0, "backend-load-all")?;
    let loaded = invoke(
        client,
        0,
        json!({"Op":"common-model","Params":p,"ModelOnly":false}),
    )?;
    checks.source(
        !loaded["Model"].is_null() && !loaded["Context"].is_null(),
        "fragmented model/context initialization failed",
    )?;
    let model = handle(&loaded, "Model")?;
    let ctx = handle(&loaded, "Context")?;
    let batch = batch_new(client, 0, 210)?;
    let mut rows = Vec::with_capacity(210);
    for position in 0..70 {
        for sequence in 0..3 {
            rows.push(row(1, position, sequence, false));
        }
    }
    add_rows(client, 0, batch, rows)?;
    invoke(
        client,
        0,
        json!({"Op":"batch-logit","Batch":batch,"Index":209,"Value":true}),
    )?;
    decode_zero(client, 0, ctx, batch, checks)?;
    let blob = state_get(client, ctx, 1, false, false, checks)?;
    invoke(
        client,
        0,
        json!({"Op":"memory-remove","Context":ctx,"Sequence":1,"From":-1,"To":-1}),
    )?;
    state_set(client, ctx, &blob, 1, false, checks)?;
    let sampler = dist(client, 0, model)?;
    let next = sample(client, 0, sampler, ctx, -1)?;
    invoke(
        client,
        0,
        json!({"Op":"token-info","Context":ctx,"Token":next}),
    )?;
    invoke(client, 0, json!({"Op":"batch-clear","Batch":batch}))?;
    add_rows(client, 0, batch, vec![row(next, 70, 1, true)])?;
    decode_zero(client, 0, ctx, batch, checks)?;
    release(client, 0, "sampler", sampler)?;
    release(client, 0, "batch", batch)?;
    release(client, 0, "blob", handle(&blob, "Blob")?)?;
    release(client, 0, "context", ctx)?;
    release(client, 0, "model", model)?;
    release(client, 0, "params", p)?;
    Ok(
        json!({"Interleave":"position-major sequences 0,1,2","TokensPerSequence":70,"RestoredSequence":1,"NextToken":next}),
    )
}

fn thread_worker(client: &Client, worker: u32, model: u64, p: u64) -> Result<(Checks, Value)> {
    let mut checks = Checks::default();
    let ctx = context(client, worker, model, Some(p), json!({"Sequences":1}))?;
    checks.source(ctx > 0, "concurrent context initialization failed")?;
    let value = invoke(
        client,
        worker,
        json!({"Op":"sampler","Model":model,"Kind":"common","Seed":null,"Params":p}),
    )?;
    checks.source(
        !value["Sampler"].is_null(),
        "common sampler initialization failed",
    )?;
    let sampler = handle(&value, "Sampler")?;
    let prompt = tokens(&invoke(
        client,
        worker,
        json!({"Op":"tokenize","Context":ctx,"Text":PROMPT,"Special":true}),
    )?)?;
    checks.source(
        !prompt.is_empty(),
        "concurrent prompt tokenization returned no tokens",
    )?;
    let batch = batch_one(client, worker, None, &prompt, None)?;
    decode_zero(client, worker, ctx, batch, &mut checks)?;
    let mut previous_count = prompt.len();
    let mut generated = Vec::new();
    let mut reason = "prediction-limit";
    for index in 0..128 {
        // The source's BOS branch is unreachable after successful nonempty decode.
        let next = sample(
            client,
            worker,
            sampler,
            ctx,
            i64::try_from(previous_count - 1)?,
        )?;
        generated.push(next);
        let info = invoke(
            client,
            worker,
            json!({"Op":"token-info","Context":ctx,"Token":next}),
        )?;
        checks.guard(
            info["PieceHex"].is_string(),
            "token-to-piece result is missing",
        )?;
        if boolean(&info, "Eog")? {
            reason = "eog";
            break;
        }
        batch_one(client, worker, Some(batch), &[next], None)?;
        previous_count = 1;
        let code = decode(client, worker, ctx, batch)?;
        if code == 1 && index > 0 {
            reason = "context-full";
            break;
        }
        checks.source(code == 0, "concurrent generation decode failed")?;
    }
    invoke(client, worker, json!({"Op":"synchronize","Context":ctx}))?;
    release(client, worker, "sampler", sampler)?;
    release(client, worker, "batch", batch)?;
    release(client, worker, "context", ctx)?;
    Ok((
        checks,
        json!({"Worker":worker,"PromptTokens":prompt,"GeneratedTokens":generated,"Stop":reason}),
    ))
}
fn thread_safety(client: &Client, path: &str, checks: &mut Checks) -> Result<Value> {
    plain(client, 0, "common-init")?;
    let arguments = json!([
        "test-thread-safety",
        "-m",
        path,
        "-ngl",
        "99",
        "-p",
        PROMPT,
        "-n",
        "128",
        "-c",
        "256",
        "-ub",
        "32",
        "-np",
        "4",
        "-t",
        "2"
    ]);
    let p = params(client, json!({}), arguments, checks)?;
    plain(client, 0, "backend-init")?;
    invoke(client, 0, json!({"Op":"numa-init","Params":p}))?;
    let inventory = invoke(client, 0, json!({"Op":"devices"}))?;
    let devices = inventory["Devices"]
        .as_array()
        .ok_or_else(|| invalid("missing device inventory"))?;
    let gpu_ids = devices
        .iter()
        .filter(|d| d["Type"].as_i64() == Some(1))
        .map(|d| number(d, "Index"))
        .collect::<Result<Vec<_>>>()?;
    checks.guard(
        (gpu_ids.len() + 2) * 4 < 16,
        "all original concurrent workers exceed the bounded native lanes",
    )?;
    let mut models = Vec::new();
    for index in 0..gpu_ids.len() + 2 {
        let overrides = match index.cmp(&gpu_ids.len()) {
            std::cmp::Ordering::Less => json!({"SplitMode":0,"Devices":[gpu_ids[index]]}),
            std::cmp::Ordering::Equal => json!({"SplitMode":0,"MainGpu":-1}),
            std::cmp::Ordering::Greater => json!({"SplitMode":1}),
        };
        let loaded = invoke(
            client,
            0,
            json!({"Op":"model-load","Path":path,"Base":"common","Params":p,"Overrides":overrides,"Progress":false}),
        )?;
        checks.source(
            !loaded["Model"].is_null(),
            "thread-safety model initialization failed",
        )?;
        models.push(handle(&loaded, "Model")?);
    }
    // Rust threads submit concurrently to distinct persistent native OS workers.
    // There is no central sequential inference loop and no expected-text fixture.
    let outcomes = std::thread::scope(|scope| {
        let mut jobs = Vec::new();
        for (index, &model) in models.iter().enumerate() {
            for context in 0..4 {
                let worker = u32::try_from(1 + index * 4 + context)?;
                jobs.push(scope.spawn(move || thread_worker(client, worker, model, p)));
            }
        }
        jobs.into_iter()
            .map(|job| {
                job.join()
                    .map_err(|_| invalid("Rust model worker panicked"))?
            })
            .collect::<Result<Vec<_>>>()
    })?;
    let mut workers = Vec::new();
    for (worker_checks, detail) in outcomes {
        checks.source += worker_checks.source;
        checks.guards += worker_checks.guards;
        workers.push(detail);
    }
    checks.source(
        workers.len() == models.len() * 4,
        "not all concurrent context workers finished",
    )?;
    for model in &models {
        release(client, 0, "model", *model)?;
    }
    release(client, 0, "params", p)?;
    Ok(
        json!({"GpuDevices":gpu_ids,"Models":models.len(),"ContextsPerModel":4,"Workers":workers,"CpuModel":true,"LayerSplitModel":true}),
    )
}

fn source_sha(id: &str) -> Result<&'static str> {
    match id {
        AUTORELEASE => Ok("71d51cea2eefa5810189156e76ac1d87a46e244081cedc16eae384f2d19dc02d"),
        CANCEL => Ok("92b561c68b058db20722873c5f9dbced7c240fba37de5ffbda4f4296b111e2b6"),
        SAVE => Ok("126af9ad38b3540e0cea3052af5987a0fc8d4d05e0680f8faf3380bda50c6cc1"),
        FRAGMENTED => Ok("623ac05b4ea99556d741f0c4c8016967e3260e057ce073d1080830237cf5a9bd"),
        THREADS => Ok("7fb63816a7033566ff1dfb76001b7bdf812a475de2cfe16e544ab14550f56039"),
        _ => Err(invalid("unknown model lifecycle source")),
    }
}
pub(super) fn run(engine: &Engine, id: &str) -> Result<Value> {
    let sha = source_sha(id)?;
    if id == CANCEL && engine.is_rust() {
        return Ok(run_rust_cancel(engine, sha));
    }
    let Some(session) = ModelSession::open(engine, id)? else {
        return Ok(result(
            id,
            false,
            "production model/context/state API unavailable".into(),
        ));
    };
    let client = session.client();
    let mut checks = Checks::default();
    let outcome = (|| {
        let asset = engine.asset_path(ASSET)?;
        verify_artifact(&asset, ASSET_LENGTH, ASSET_SHA)?;
        let path = asset
            .to_str()
            .ok_or_else(|| invalid("model path is not UTF-8"))?;
        match id {
            AUTORELEASE => autorelease(&client, path, &mut checks),
            CANCEL => load_cancel(
                |worker, operation, progress| client.call(worker, operation, progress),
                path,
                &mut checks,
            ),
            SAVE => save_load(&client, path, &mut checks),
            FRAGMENTED => fragmented(&client, path, &mut checks),
            THREADS => thread_safety(&client, path, &mut checks),
            _ => Err(invalid("unknown model lifecycle source")),
        }
    })();
    let finished = session.finish();
    let (passed, detail) = match (outcome, finished) {
        (Ok(schedule), Ok(())) => (true, json!({"Schedule":schedule})),
        (Err(error), _) | (_, Err(error)) => (false, json!({"Error":error.to_string()})),
    };
    Ok(result(id,passed,json!({"SourceSHA256":sha,"SourceChecks":checks.source,"AdditionalGuards":checks.guards,"Asset":ASSET,"Observation":detail}).to_string()))
}

fn run_rust_cancel(engine: &Engine, sha: &str) -> Value {
    let mut client = storage::Client::default();
    let mut checks = Checks::default();
    let outcome = (|| {
        let asset = engine.asset_path(ASSET)?;
        verify_artifact(&asset, ASSET_LENGTH, ASSET_SHA)?;
        let path = asset
            .to_str()
            .ok_or_else(|| invalid("model path is not UTF-8"))?;
        load_cancel(
            |worker, operation, progress| client.call(worker, &operation, progress),
            path,
            &mut checks,
        )
    })();
    let (passed, detail) = match (outcome, client.finish()) {
        (Ok(schedule), Ok(())) => (true, json!({"Schedule":schedule})),
        (Err(error), _) | (_, Err(error)) => (false, json!({"Error":error.to_string()})),
    };
    result(CANCEL,passed,json!({"SourceSHA256":sha,"SourceChecks":checks.source,"AdditionalGuards":checks.guards,"Asset":ASSET,"Observation":detail}).to_string())
}

#[cfg(test)]
mod focused_guards {
    use super::*;

    #[test]
    fn original_cancellation_boundary_and_invalid_observations() {
        assert!(!cancel_policy(0.0));
        assert!(!cancel_policy(0.5));
        assert!(cancel_policy(f32::from_bits(0.5_f32.to_bits() + 1)));
        assert!(handle(&json!({"Model":null}), "Model").is_err());
        assert!(handle(&json!({"Model":0}), "Model").is_err());
        assert!(token(&json!({"Token":null})).is_err());
        assert!(boolean(&json!({"Loaded":"true"}), "Loaded").is_err());
        let mut checks = Checks::default();
        assert!(checks.source(false, "injected production failure").is_err());
        assert_eq!(checks.source, 1);
    }

    #[test]
    fn seed_1234_matches_standard_mt19937_words() {
        let mut random = Random::new(1234);
        let words: Vec<_> = (0..5).map(|_| random.next()).collect();
        assert_eq!(
            words,
            [
                822_569_775,
                2_137_449_171,
                2_671_936_806,
                3_512_589_365,
                1_880_026_316
            ]
        );
    }

    #[test]
    fn artifact_guard_rejects_size_and_same_length_content_changes() -> Result<()> {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "prism-model-lock-{}-{stamp}.bin",
            std::process::id()
        ));
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)?;
        std::io::Write::write_all(&mut file, b"abc")?;
        drop(file);
        let sha = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        let checked = (|| {
            verify_artifact(&path, 3, sha)?;
            assert!(verify_artifact(&path, 4, sha).is_err());
            std::fs::write(&path, b"abd")?;
            assert!(verify_artifact(&path, 3, sha).is_err());
            Ok(())
        })();
        std::fs::remove_file(&path)?;
        checked
    }
}
