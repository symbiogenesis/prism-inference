//! Complete generated-fixture recurrent rollback schedules, owned by safe Rust.
use super::architectures::{export_selected, handle, invoke, release};
use super::engine::Engine;
use super::generated_inputs;
use super::model_session::{Client, ModelSession};
use super::{Result, invalid, result};
use serde_json::{Value, json};
use std::path::Path;

const ID: &str = "tests/test-recurrent-state-rollback.cpp";
const SOURCE_SHA: &str = "51c1ab7b2acfcbad88651ae8f4d4ad055ab5196c4794949255545bcb82abc299";
const FIXTURES: [(&str, bool); 3] = [
    ("qwen35", false),
    ("nemotron_h", false),
    ("deepseek4", true),
];
#[derive(Default)]
struct Counts {
    predicates: usize,
    decodes: usize,
    logit_pointers: usize,
    float_comparisons: usize,
}
impl Counts {
    fn check(&mut self, ok: bool, message: &str) -> Result<()> {
        self.predicates += 1;
        if ok { Ok(()) } else { Err(invalid(message)) }
    }
}
fn number(v: &Value, key: &str) -> Result<u32> {
    u32::try_from(
        v[key]
            .as_u64()
            .ok_or_else(|| invalid(format!("missing recurrent number {key}")))?,
    )
    .map_err(Into::into)
}
fn ctx(c: &Client, model: u64, params: u64, overrides: Value, counts: &mut Counts) -> Result<u64> {
    let mut operation = json!({"Op":"context","Model":model,"Params":params});
    operation["Overrides"] = overrides;
    let v = invoke(c, operation)?;
    counts.check(!v["Context"].is_null(), "source recurrent context is null")?;
    handle(&v, "Context")
}
fn batch(c: &Client, capacity: usize) -> Result<u64> {
    handle(
        &invoke(
            c,
            json!({"Op":"batch-new","Capacity":capacity,"Sequences":1}),
        )?,
        "Batch",
    )
}
fn add(c: &Client, b: u64, rows: Vec<Value>) -> Result<()> {
    let mut operation = json!({"Op":"batch-add","Batch":b});
    operation["Rows"] = Value::Array(rows);
    invoke(c, operation)?;
    Ok(())
}
fn row(token: u32, position: u32, sequence: u32, logits: bool) -> Value {
    json!({"Token":token,"Position":position,"Sequences":[sequence],"Logits":logits})
}
fn decode(c: &Client, ctx: u64, b: u64, counts: &mut Counts) -> Result<()> {
    counts.decodes += 1;
    counts.check(
        invoke(c, json!({"Op":"decode","Context":ctx,"Batch":b}))?["Code"] == 0,
        "source recurrent decode failed",
    )
}
fn decode_rows(c: &Client, ctx: u64, rows: Vec<Value>, counts: &mut Counts) -> Result<()> {
    let b = batch(c, rows.len())?;
    add(c, b, rows)?;
    decode(c, ctx, b, counts)?;
    release(c, "batch", b)
}
fn clear(c: &Client, b: u64) -> Result<()> {
    invoke(c, json!({"Op":"batch-clear","Batch":b}))?;
    Ok(())
}
fn remove(
    c: &Client,
    ctx: u64,
    sequence: u32,
    from: u32,
    expected: bool,
    counts: &mut Counts,
) -> Result<()> {
    let value = invoke(
        c,
        json!({"Op":"memory-remove","Context":ctx,"Sequence":sequence,"From":from,"To":-1}),
    )?;
    counts.check(
        value["Removed"].as_bool() == Some(expected),
        "source rollback/remove result differs",
    )
}
fn logits(
    c: &Client,
    ctx: u64,
    indices: &[u32],
    vocab: u32,
    counts: &mut Counts,
) -> Result<Vec<Vec<f32>>> {
    let value = invoke(
        c,
        json!({"Op":"context-logits","Context":ctx,"Indices":indices}),
    )?;
    let values = value["Values"]
        .as_array()
        .ok_or_else(|| invalid("missing recurrent logits"))?;
    if values.len() != indices.len() {
        return Err(invalid("recurrent logit result count differs"));
    }
    values
        .iter()
        .map(|v| {
            counts.logit_pointers += 1;
            counts.check(
                v["Present"] == true,
                "source recurrent logit pointer is null",
            )?;
            let floats = generated_inputs::floats(&v["DataHex"])?;
            if floats.len() != usize::try_from(vocab)? {
                return Err(invalid("recurrent logit vocabulary extent differs"));
            }
            Ok(floats)
        })
        .collect()
}
// The source checks `fabs(a-b) > eps`; NaNs therefore do not trigger that predicate.
fn compare(a: &[f32], b: &[f32], eps: f32, counts: &mut Counts) -> Result<()> {
    if a.len() != b.len() {
        return Err(invalid("recurrent comparison shape differs"));
    }
    for (&x, &y) in a.iter().zip(b) {
        counts.float_comparisons += 1;
        counts.check(
            (x - y).abs().partial_cmp(&eps) != Some(std::cmp::Ordering::Greater),
            "source recurrent logits mismatch",
        )?;
    }
    Ok(())
}
fn checkpoint(c: &Client, ctx: u64, flags: u32) -> Result<u64> {
    handle(
        &invoke(
            c,
            json!({"Op":"checkpoint-update","Checkpoint":null,"Context":ctx,"Sequence":0,"Flags":flags}),
        )?,
        "Checkpoint",
    )
}
fn restore(c: &Client, ctx: u64, checkpoint: u64, flags: u32) -> Result<()> {
    invoke(
        c,
        json!({"Op":"checkpoint-load","Checkpoint":checkpoint,"Context":ctx,"Sequence":0,"Flags":flags}),
    )?;
    Ok(())
}
fn prompt_rows(tokens: &[u32]) -> Result<Vec<Value>> {
    tokens
        .iter()
        .enumerate()
        .map(|(p, &t)| Ok(row(t, u32::try_from(p)?, 0, p + 1 == tokens.len())))
        .collect()
}
fn replay(
    c: &Client,
    src: u64,
    dst: u64,
    tokens: &[u32],
    start: u32,
    vocab: u32,
    counts: &mut Counts,
) -> Result<Vec<Vec<f32>>> {
    let mut saved = Vec::new();
    for position in start..u32::try_from(tokens.len())? {
        let t = tokens[usize::try_from(position)?];
        decode_rows(c, src, vec![row(t, position, 0, true)], counts)?;
        decode_rows(c, dst, vec![row(t, position, 0, true)], counts)?;
        let a = logits(c, src, &[0], vocab, counts)?.remove(0);
        let b = logits(c, dst, &[0], vocab, counts)?.remove(0);
        compare(&a, &b, 1e-5, counts)?;
        saved.push(a);
    }
    Ok(saved)
}
fn single_sequence(
    c: &Client,
    model: u64,
    params: u64,
    vocab: u32,
    vocab_type: u32,
    counts: &mut Counts,
) -> Result<Value> {
    let defaults = invoke(c, json!({"Op":"context-params","Params":params}))?;
    let overrides = json!({"Sequences":1,"RecurrentSnapshots":8,"Batch":number(&defaults,"Batch")?.max(9),"MicroBatch":number(&defaults,"MicroBatch")?.max(9)});
    let src = ctx(c, model, params, overrides.clone(), counts)?;
    let dst = ctx(c, model, params, overrides.clone(), counts)?;
    let snapshots = number(
        &invoke(c, json!({"Op":"context-metadata","Context":src}))?,
        "RecurrentSnapshots",
    )?;
    if snapshots < 3 {
        release(c, "context", src)?;
        release(c, "context", dst)?;
        return Ok(
            json!({"Skipped":if snapshots==0{"source recurrent snapshots disabled"}else{"source recurrent snapshot count below3"}}),
        );
    }
    let mut tokens = if vocab_type == 0 {
        (1..=9).collect::<Vec<u32>>()
    } else {
        invoke(c,json!({"Op":"tokenize","Context":src,"Text":"The quick brown fox jumps over the lazy dog","Special":true}))?["Tokens"].as_array().ok_or_else(||invalid("prompt token list absent"))?.iter().map(|v|u32::try_from(v.as_u64().ok_or_else(||invalid("negative prompt token"))?).map_err(Into::into)).collect::<Result<Vec<_>>>()?
    };
    counts.check(!tokens.is_empty(), "source recurrent prompt is empty")?;
    let last = *tokens
        .last()
        .ok_or_else(|| invalid("recurrent prompt absent"))?;
    tokens.resize(usize::try_from(snapshots + 1)?, last);
    let start = u32::try_from(tokens.len())? - 3;
    decode_rows(c, src, prompt_rows(&tokens)?, counts)?;
    remove(c, src, 0, start, true, counts)?;
    let full = checkpoint(c, src, 0)?;
    restore(c, dst, full, 0)?;
    replay(c, src, dst, &tokens, start, vocab, counts)?;
    remove(c, src, 0, start, true, counts)?;
    remove(c, dst, 0, start, true, counts)?;
    let partial = checkpoint(c, src, 1)?;
    restore(c, dst, partial, 1)?;
    let saved = replay(c, src, dst, &tokens, start, vocab, counts)?;
    let dirty = ctx(c, model, params, overrides, counts)?;
    let noise: Vec<_> = tokens.iter().map(|t| (t + 1) % vocab).collect();
    decode_rows(c, dirty, prompt_rows(&noise)?, counts)?;
    remove(c, dirty, 0, start, true, counts)?;
    restore(c, dirty, full, 0)?;
    for (position, expected) in (start..u32::try_from(tokens.len())?).zip(&saved) {
        decode_rows(
            c,
            dirty,
            vec![row(tokens[usize::try_from(position)?], position, 0, true)],
            counts,
        )?;
        let actual = logits(c, dirty, &[0], vocab, counts)?.remove(0);
        compare(expected, &actual, 1e-5, counts)?;
    }
    for id in [src, dst, dirty] {
        release(c, "context", id)?;
    }
    for id in [full, partial] {
        release(c, "checkpoint", id)?;
    }
    Ok(
        json!({"Snapshots":snapshots,"Prompt":tokens,"RollbackPosition":start,"FullReplay":3,"PartialReplay":3,"DirtyReplay":3,"Tolerance":1e-5}),
    )
}
fn multi_overrides(micro: u32) -> Value {
    json!({"Sequences":2,"RecurrentSnapshots":8,"Context":256,"Batch":256,"MicroBatch":micro,"Unified":false})
}
const fn multi_token(sequence: u32, position: u32, vocab: u32) -> u32 {
    (7 * position + 31 * sequence + 1) % vocab
}
fn max_difference(a: &[f32], b: &[f32], counts: &mut Counts) -> Result<f32> {
    if a.len() != b.len() {
        return Err(invalid("multi-sequence comparison shape differs"));
    }
    let mut max = 0.0;
    for (&x, &y) in a.iter().zip(b) {
        counts.float_comparisons += 1;
        let difference = (x - y).abs();
        if difference > max {
            max = difference;
        }
    }
    Ok(max)
}
fn multi_sequence(
    c: &Client,
    model: u64,
    params: u64,
    vocab: u32,
    counts: &mut Counts,
) -> Result<Value> {
    let roll = ctx(c, model, params, multi_overrides(16), counts)?;
    let reference = ctx(c, model, params, multi_overrides(16), counts)?;
    if number(
        &invoke(c, json!({"Op":"context-metadata","Context":roll}))?,
        "RecurrentSnapshots",
    )? < 3
    {
        release(c, "context", roll)?;
        release(c, "context", reference)?;
        return Ok(json!({"Skipped":"source multi-sequence snapshot count below3"}));
    }
    for sequence in 0..2 {
        let b = batch(c, 19)?;
        add(
            c,
            b,
            (0..16)
                .map(|p| row(multi_token(sequence, p, vocab), p, sequence, false))
                .collect(),
        )?;
        decode(c, roll, b, counts)?;
        decode(c, reference, b, counts)?;
        clear(c, b)?;
        add(
            c,
            b,
            (16..19)
                .map(|p| row(multi_token(sequence, p, vocab), p, sequence, false))
                .collect(),
        )?;
        decode(c, roll, b, counts)?;
        release(c, "batch", b)?;
        remove(c, roll, sequence, 16, true, counts)?;
        remove(c, roll, sequence, 15, false, counts)?;
    }
    let b = batch(c, 80)?;
    let rows = (0..2)
        .flat_map(|s| (16..56).map(move |p| row(multi_token(s, p, vocab), p, s, true)))
        .collect();
    add(c, b, rows)?;
    decode(c, roll, b, counts)?;
    decode(c, reference, b, counts)?;
    release(c, "batch", b)?;
    let indices: Vec<_> = (0..80).collect();
    let a = logits(c, roll, &indices, vocab, counts)?;
    let b = logits(c, reference, &indices, vocab, counts)?;
    let mut maximum = 0.0_f32;
    for (x, y) in a.iter().zip(&b) {
        maximum = maximum.max(max_difference(x, y, counts)?);
    }
    counts.check(
        maximum.partial_cmp(&1e-7) != Some(std::cmp::Ordering::Greater),
        "source multi-sequence split replay exceeds tolerance",
    )?;
    decode_rows(
        c,
        reference,
        (56..60)
            .map(|p| row(multi_token(0, p + 7, vocab), p, 0, false))
            .collect(),
        counts,
    )?;
    let mut tail = 0.0_f32;
    for p in 56..60 {
        let b = batch(c, 1)?;
        add(c, b, vec![row(multi_token(1, p, vocab), p, 1, true)])?;
        decode(c, roll, b, counts)?;
        decode(c, reference, b, counts)?;
        release(c, "batch", b)?;
        let a = logits(c, roll, &[0], vocab, counts)?.remove(0);
        let b = logits(c, reference, &[0], vocab, counts)?.remove(0);
        tail = tail.max(max_difference(&a, &b, counts)?);
    }
    counts.check(
        tail.partial_cmp(&1e-7) != Some(std::cmp::Ordering::Greater),
        "source sequence1 continuation leaked sequence0 state",
    )?;
    release(c, "context", roll)?;
    release(c, "context", reference)?;
    Ok(
        json!({"Sequences":2,"MicroBatch":16,"ReplayPerSequence":40,"ReplayMaximumDifference":maximum,"IndependentTailMaximumDifference":tail,"Tolerance":1e-7}),
    )
}
fn pending_fork(
    c: &Client,
    model: u64,
    params: u64,
    vocab: u32,
    counts: &mut Counts,
) -> Result<Value> {
    let mut states = vec![String::new(); 2];
    for fork in [1_usize, 0] {
        let ctx = ctx(c, model, params, multi_overrides(256), counts)?;
        let source = u32::from(fork == 0);
        let b = batch(c, 12)?;
        add(
            c,
            b,
            (0..12)
                .map(|p| row((5 * p + 3) % vocab, p, source, false))
                .collect(),
        )?;
        decode(c, ctx, b, counts)?;
        remove(c, ctx, source, 9, true, counts)?;
        if fork == 1 {
            invoke(
                c,
                json!({"Op":"memory-copy","Context":ctx,"Source":0,"Destination":1,"From":-1,"To":-1}),
            )?;
        }
        clear(c, b)?;
        add(c, b, vec![row(9 % vocab, 9, 1, true)])?;
        decode(c, ctx, b, counts)?;
        release(c, "batch", b)?;
        let state = invoke(
            c,
            json!({"Op":"state-get","Context":ctx,"Sequence":1,"Extended":true,"Flags":1,"ReturnData":true}),
        )?;
        if state["Copied"] != state["Size"] {
            return Err(invalid("production recurrent state copy is incomplete"));
        }
        states[fork] = state["DataHex"]
            .as_str()
            .ok_or_else(|| invalid("recurrent state bytes absent"))?
            .into();
        release(c, "blob", handle(&state, "Blob")?)?;
        release(c, "context", ctx)?;
    }
    counts.check(
        !states[0].is_empty(),
        "source reference partial state is empty",
    )?;
    counts.check(
        states[0] == states[1],
        "source forked partial rollback state bytes differ",
    )?;
    Ok(
        json!({"ForkThenRollback":false,"RollbackThenFork":true,"ComparedStateBytes":states[0].len()/2}),
    )
}
fn fixture_run(engine: &Engine, path: &Path) -> Result<Value> {
    let label = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| invalid("fixture filename absent"))?;
    let session = ModelSession::open(engine, &format!("recurrent-{label}"))?
        .ok_or_else(|| invalid("missing production recurrent/context API"))?;
    let c = session.client();
    let run: Result<Value> = (|| {
        let mut counts = Counts::default();
        invoke(&c, json!({"Op":"common-init"}))?;
        let parsed = invoke(
            &c,
            json!({"Op":"params","Initial":{"Seed":1234,"Predict":1},"Arguments":["test-recurrent-state-rollback","-m",path]}),
        )?;
        counts.check(
            parsed["Parsed"] == true,
            "source recurrent model arguments rejected",
        )?;
        let params = handle(&parsed, "Handle")?;
        invoke(&c, json!({"Op":"backend-load-all"}))?;
        let loaded = invoke(
            &c,
            json!({"Op":"common-model","Params":params,"ModelOnly":false}),
        )?;
        counts.check(!loaded["Model"].is_null(), "source recurrent model is null")?;
        let model = handle(&loaded, "Model")?;
        let warmup = handle(&loaded, "Context")?;
        let meta = invoke(&c, json!({"Op":"context-metadata","Context":warmup}))?;
        if meta["Recurrent"] != true && meta["Hybrid"] != true {
            release(&c, "context", warmup)?;
            release(&c, "model", model)?;
            release(&c, "params", params)?;
            return Ok(
                json!({"Fixture":label,"Skipped":"source non-recurrent/non-hybrid model gate"}),
            );
        }
        let vocab = invoke(&c, json!({"Op":"vocab","Model":model}))?;
        let size = number(&vocab, "Count")?;
        if size == 0 {
            return Err(invalid("recurrent vocabulary is empty"));
        }
        let single = single_sequence(
            &c,
            model,
            params,
            size,
            number(&vocab, "Type")?,
            &mut counts,
        )?;
        // Source returns immediately when its first context cannot retain three snapshots.
        let (multi, fork) = if single.get("Skipped").is_some() {
            (Value::Null, Value::Null)
        } else {
            (
                multi_sequence(&c, model, params, size, &mut counts)?,
                pending_fork(&c, model, params, size, &mut counts)?,
            )
        };
        release(&c, "context", warmup)?;
        release(&c, "model", model)?;
        release(&c, "params", params)?;
        Ok(
            json!({"Fixture":label,"Vocabulary":size,"SingleSequence":single,"MultiSequence":multi,"PendingFork":fork,"SourcePredicates":counts.predicates,"DecodeCalls":counts.decodes,"LogitPointerChecks":counts.logit_pointers,"FloatComparisons":counts.float_comparisons}),
        )
    })();
    let finished = session.finish();
    let value = run?;
    finished?;
    Ok(value)
}
#[allow(
    clippy::unnecessary_wraps,
    reason = "source adapter returns explicit failed contract evidence"
)]
pub(super) fn run(engine: &Engine) -> Result<Value> {
    let execute: Result<Value> = (|| {
        let output = engine.evidence_path("recurrent-fixtures")?;
        let paths = export_selected(engine, &output, &FIXTURES, 1234)?;
        let mut results = Vec::new();
        for path in &paths {
            results.push(fixture_run(engine, path)?);
        }
        Ok(
            json!({"SourceSHA256":SOURCE_SHA,"FixtureGeneratorSource":"tests/test-llama-archs.cpp","FixtureSeed":1234,"Paths":paths,"Results":results}),
        )
    })();
    match execute {
        Ok(v) => Ok(result(ID, true, v.to_string())),
        Err(e) => Ok(result(
            ID,
            false,
            format!("complete recurrent-state source contract: {e}"),
        )),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn focused_recurrent_schedule_and_mutation_guard() -> Result<()> {
        let mut counts = Counts::default();
        compare(&[1.0], &[1.0 + 1e-6], 1e-5, &mut counts)?;
        if compare(&[1.0], &[1.1], 1e-5, &mut counts).is_ok() {
            return Err(invalid("recurrent mutation accepted"));
        }
        if compare(&[1.0], &[], 1e-5, &mut counts).is_ok() {
            return Err(invalid("recurrent shape mismatch accepted"));
        }
        if FIXTURES.len() != 3
            || multi_token(1, 16, 128) != 16
            || prompt_rows(&[1, 2, 3])?[2]["Logits"] != true
        {
            return Err(invalid("source recurrent schedule inputs differ"));
        }
        Ok(())
    }
}
