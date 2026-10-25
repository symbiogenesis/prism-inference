//! Complete Windows CPU loop contract from the pinned `DSpark` source.
//! The independent reference is computed before opening the production session.
use super::engine::Engine;
use super::generated_inputs::hex;
use super::model_session::{Client, ModelSession};
use super::{Result, invalid, result};
use serde_json::{Value, json};
use std::path::PathBuf;
const ID: &str = "tests/test-dspark-loop.cpp";
const SOURCE_SHA: &str = "8ad2915c406b296439d0d1474cb147ea35e4543f2b4d479120e67b2151f8a80f";
const ACCEPT: [i32; 7] = [7, 3, 0, 7, 5, 1, 4];
const PROMPT: [i32; 5] = [1, 2, 3, 4, 5];
#[derive(Default)]
pub(super) struct LoopOptions {
    pub(super) model: Option<PathBuf>,
    pub(super) reference: Option<PathBuf>,
}
pub(super) const fn hash_u32(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}
pub(super) fn features(begin: i32, rows: usize, width: usize) -> Result<Vec<f32>> {
    let mut values = Vec::with_capacity(rows * width);
    for row in 0..rows {
        let position = u32::try_from(
            begin
                .checked_add(i32::try_from(row)?)
                .ok_or_else(|| invalid("feature position overflow"))?,
        )?;
        for d in 0..width {
            let value = hash_u32(
                position
                    .wrapping_mul(131_071)
                    .wrapping_add(u32::try_from(d)?.wrapping_mul(97))
                    .wrapping_add(12_345),
            );
            let numerator = i16::try_from(value % 2000)? - 1000;
            values.push(f32::from(numerator) / 500.0);
        }
    }
    Ok(values)
}
pub(super) fn bonus(round: i32, vocabulary: i32, mask: i32) -> Result<i32> {
    let raw_round = u32::from_le_bytes(round.to_le_bytes());
    let h = hash_u32(raw_round.wrapping_mul(2_654_435_761).wrapping_add(999_983));
    let mut token = i32::try_from(h % u32::try_from(vocabulary - 1)?)?;
    if token == mask {
        token = (token + 1) % vocabulary;
    }
    Ok(token)
}
fn invoke(client: &Client, operation: Value) -> Result<Value> {
    Ok(client.call(0, operation, None)?.value)
}
fn handle(value: &Value, key: &str) -> Result<u64> {
    value[key]
        .as_u64()
        .filter(|&v| v > 0)
        .ok_or_else(|| invalid(format!("missing production {key}")))
}
fn integer(value: &Value, key: &str) -> Result<i32> {
    Ok(i32::try_from(value[key].as_i64().ok_or_else(|| {
        invalid(format!("missing loop reference integer {key}"))
    })?)?)
}
fn check(count: &mut usize, condition: bool, message: &str) -> Result<()> {
    *count += 1;
    if condition {
        Ok(())
    } else {
        Err(invalid(message))
    }
}
fn positions(begin: i32, rows: usize) -> Result<Vec<i32>> {
    (0..rows)
        .map(|i| {
            begin
                .checked_add(i32::try_from(i)?)
                .ok_or_else(|| invalid("loop position overflow"))
        })
        .collect()
}
fn stage(client: &Client, speculative: u64, begin: i32, rows: usize, width: usize) -> Result<bool> {
    let data = features(begin, rows, width)?;
    let raw = data
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect::<Vec<_>>();
    Ok(invoke(
        client,
        json!({"Op":"speculative-stage","Speculative":speculative,"Sequence":0,"Rows":rows,"Width":width,"FeaturesHex":hex(&raw)?,"Positions":positions(begin,rows)?}),
    )?["Staged"]
        == true)
}
fn independent_reference() -> Result<Value> {
    let mut reference = super::dspark_reference::Reference::new()?;
    let (mut start, mut context_begin, mut context_rows) = (5, 0, 5);
    let mut anchor = bonus(-1, 256, 255)?;
    let mut rounds = Vec::new();
    for (round, &accepted) in ACCEPT.iter().enumerate() {
        let mut tokens = vec![255; 7];
        tokens[0] = anchor;
        let logits = reference.forward(
            &features(context_begin, context_rows, 128)?,
            &positions(context_begin, context_rows)?,
            &tokens,
            &positions(start, 7)?,
            Some(usize::try_from(start)?),
        )?;
        let sampled = reference.sample(&logits, anchor)?;
        let next_bonus = bonus(i32::try_from(round)?, 256, 255)?;
        rounds.push(json!({"n_accepted":accepted,"ctx_len":context_rows,"start":start,"sampled":sampled,"bonus":next_bonus}));
        context_begin = start;
        context_rows = usize::try_from(accepted + 1)?;
        start += accepted + 1;
        anchor = next_bonus;
    }
    Ok(
        json!({"n_embd_cap":128,"block_size":7,"vocab_size":256,"mask_token_id":255,"prefill_bonus":bonus(-1,256,255)?,"rounds":rounds,"ReferenceProvenance":{"Implementation":"safe scalar Rust; no SUT observations","DeepSpecRevision":"005e03b81cec38b7da6399833d609ee89a2587f2","Equations":["qwen3/modeling.py _forward_backbone","markov_head.py VanillaMarkov","eval/dspark/draft_ops.py cache crop"],"MissingOriginalGenerator":"dspark_phase2_py_ref.py is not present in pinned source or its introduction commit","Coverage":"one entire explicit seven-entry ACCEPT_SCHEDULE cycle","PublishedSources":super::dspark_reference::Reference::provenance()?}}),
    )
}
#[allow(
    clippy::too_many_lines,
    clippy::unnecessary_wraps,
    reason = "all source loop predicates and stage/accept ordering remain together"
)]
pub(super) fn run_with_options(engine: &Engine, options: &LoopOptions) -> Result<Value> {
    let execute: Result<Value> = (|| {
        let (model_path, reference, pair_receipt) = match (&options.model, &options.reference) {
            (None, None) => {
                let pair = super::dspark_pair::create(engine, "reference")?;
                let reference = independent_reference()?;
                std::fs::write(
                    pair.drafter
                        .parent()
                        .ok_or_else(|| invalid("pair input parent absent"))?
                        .join("independent-loop-reference.json"),
                    serde_json::to_vec_pretty(&reference)?,
                )?;
                (pair.drafter, reference, pair.receipt)
            }
            (Some(model), Some(path)) => {
                if !model.is_absolute()
                    || !model.is_file()
                    || !path.is_absolute()
                    || !path.is_file()
                    || std::fs::metadata(path)?.len() > 1024 * 1024
                {
                    return Err(invalid(
                        "loop model/reference paths must be bounded absolute regular files",
                    ));
                }
                (
                    model.clone(),
                    serde_json::from_slice(&std::fs::read(path)?)?,
                    json!({"ExternalPair":true,"Reference":path}),
                )
            }
            _ => {
                return Err(invalid(
                    "model and independent loop reference must be supplied together",
                ));
            }
        };
        let (width, block, vocabulary, mask) = (
            integer(&reference, "n_embd_cap")?,
            integer(&reference, "block_size")?,
            integer(&reference, "vocab_size")?,
            integer(&reference, "mask_token_id")?,
        );
        if !(1..=1_048_576).contains(&width)
            || !(7..=64).contains(&block)
            || !(2..=1_048_576).contains(&vocabulary)
            || !(0..vocabulary).contains(&mask)
            || i64::from(block + 1) * i64::from(width) * 4 > 4 * 1024 * 1024
        {
            return Err(invalid(
                "loop reference metadata outside bounded source domain",
            ));
        }
        let rounds = reference["rounds"]
            .as_array()
            .filter(|r| !r.is_empty() && r.len() <= 64)
            .ok_or_else(|| invalid("loop reference must contain1..64 rounds"))?;
        for (r, row) in rounds.iter().enumerate() {
            let sampled = row["sampled"]
                .as_array()
                .filter(|v| v.len() == usize::try_from(block).unwrap_or(0))
                .ok_or_else(|| invalid("loop reference sampled extent mismatch"))?;
            if sampled.iter().any(|v| {
                v.as_i64()
                    .is_none_or(|v| v < 0 || v >= i64::from(vocabulary))
            }) || integer(row, "n_accepted")? != ACCEPT[r % ACCEPT.len()]
                || integer(row, "start")? < 0
                || integer(row, "ctx_len")? < 0
            {
                return Err(invalid("loop reference row invalid"));
            }
        }
        let session = ModelSession::open(engine, "dspark-loop")?
            .ok_or_else(|| invalid("missing production DSpark model/speculative API"))?;
        let client = session.client();
        let outcome: Result<Value> = (|| {
            let mut assertions = 0;
            invoke(&client, json!({"Op":"backend-init"}))?;
            let model = handle(
                &invoke(
                    &client,
                    json!({"Op":"model-load","Path":model_path,"Base":"default","Params":null,"Overrides":{"GpuLayers":0},"Progress":false}),
                )?,
                "Model",
            )?;
            assertions += 1;
            let meta = invoke(&client, json!({"Op":"dspark-meta","Model":model}))?;
            check(
                &mut assertions,
                meta["Found"] == true,
                "llama_model_dspark_get_meta rejected model",
            )?;
            for (key, expected) in [
                ("CaptureEmbedding", width),
                ("BlockSize", block),
                ("Vocabulary", vocabulary),
                ("MaskToken", mask),
            ] {
                check(
                    &mut assertions,
                    meta[key] == expected,
                    "DSpark metadata differs from independent reference",
                )?;
            }
            let maximum = 2 * block + 9;
            let context = handle(
                &invoke(
                    &client,
                    json!({"Op":"context","Model":model,"Params":null,"Overrides":{"Context":maximum,"Batch":maximum,"MicroBatch":maximum,"Sequences":1,"NoPerf":true}}),
                )?,
                "Context",
            )?;
            assertions += 1;
            let spec = handle(
                &invoke(
                    &client,
                    json!({"Op":"speculative-init","DraftContext":context,"TargetContext":context,"Max":block,"Min":0,"Sequences":1}),
                )?,
                "Speculative",
            )?;
            assertions += 1;
            invoke(
                &client,
                json!({"Op":"speculative-begin","Speculative":spec,"Sequence":0,"Prompt":PROMPT}),
            )?;
            check(
                &mut assertions,
                stage(&client, spec, 0, PROMPT.len(), usize::try_from(width)?)?,
                "prefill context stage failed",
            )?;
            let mut start = i32::try_from(PROMPT.len())?;
            let mut last = bonus(-1, vocabulary, mask)?;
            check(
                &mut assertions,
                last == integer(&reference, "prefill_bonus")?,
                "prefill bonus drift",
            )?;
            let mut mismatches = 0;
            let mut receipts = Vec::new();
            for (r, row) in rounds.iter().enumerate() {
                let accepted = integer(row, "n_accepted")?;
                check(
                    &mut assertions,
                    accepted == ACCEPT[r % ACCEPT.len()],
                    "accept schedule drift",
                )?;
                check(
                    &mut assertions,
                    start == integer(row, "start")?,
                    "loop start bookkeeping differs",
                )?;
                let actual = invoke(
                    &client,
                    json!({"Op":"speculative-draft","Speculative":spec,"Sequence":0,"Drafting":true,"Max":-1,"Past":start,"Last":last,"Prompt":null}),
                )?;
                let values = actual["Tokens"]
                    .as_array()
                    .ok_or_else(|| invalid("production draft tokens absent"))?;
                check(
                    &mut assertions,
                    values.len() == usize::try_from(block)?,
                    "production draft block length differs",
                )?;
                let expected = row["sampled"]
                    .as_array()
                    .ok_or_else(|| invalid("reference sampled tokens absent"))?;
                let mut same = true;
                for (actual, expected) in values.iter().zip(expected) {
                    assertions += 1;
                    same &= actual == expected;
                }
                if !same {
                    mismatches += 1;
                }
                check(
                    &mut assertions,
                    stage(
                        &client,
                        spec,
                        start,
                        usize::try_from(block + 1)?,
                        usize::try_from(width)?,
                    )?,
                    "verify context stage failed",
                )?;
                invoke(
                    &client,
                    json!({"Op":"speculative-accept","Speculative":spec,"Sequence":0,"Accepted":accepted}),
                )?;
                let next = bonus(i32::try_from(r)?, vocabulary, mask)?;
                check(
                    &mut assertions,
                    next == integer(row, "bonus")?,
                    "round bonus drift",
                )?;
                receipts.push(json!({"Round":r,"Start":start,"Accepted":accepted,"ReferenceContextRowsReadButNotCompared":row["ctx_len"],"Actual":values,"Expected":expected,"Matched":same}));
                last = next;
                start += accepted + 1;
            }
            invoke(
                &client,
                json!({"Op":"speculative-stats","Speculative":spec}),
            )?;
            invoke(
                &client,
                json!({"Op":"speculative-release","Speculative":spec}),
            )?;
            invoke(
                &client,
                json!({"Op":"release","Kind":"context","Handle":context}),
            )?;
            invoke(
                &client,
                json!({"Op":"release","Kind":"model","Handle":model}),
            )?;
            invoke(&client, json!({"Op":"backend-free"}))?;
            check(
                &mut assertions,
                mismatches == 0,
                "loop tokens differ from independent DeepSpec reference",
            )?;
            Ok(
                json!({"SourceSHA256":SOURCE_SHA,"SourcePredicates":assertions,"Rounds":receipts,"Pair":pair_receipt,"ReferenceProvenance":reference["ReferenceProvenance"],"PlatformExclusions":["Metal Markov self-test is not compiled for native Windows ARM64 CPU","CUDA D-cut self-test is not compiled for native Windows ARM64 CPU"]}),
            )
        })();
        let finish = session.finish();
        let value = outcome?;
        finish?;
        Ok(value)
    })();
    match execute {
        Ok(value) => Ok(result(ID, true, value.to_string())),
        Err(error) => Ok(result(
            ID,
            false,
            format!("complete DSpark loop source contract: {error}"),
        )),
    }
}
pub(super) fn run(engine: &Engine) -> Result<Value> {
    run_with_options(engine, &LoopOptions::default())
}
pub(super) fn diagnostic(engine: &Engine, value: &Value) -> Result<Value> {
    super::llama_grammar::exact_fields(value, &["Kind", "Model", "Reference"])?;
    let path = |name: &str| -> Result<Option<PathBuf>> {
        if value[name].is_null() {
            Ok(None)
        } else {
            Ok(Some(PathBuf::from(value[name].as_str().ok_or_else(
                || invalid("loop input must be null or absolute path text"),
            )?)))
        }
    };
    run_with_options(
        engine,
        &LoopOptions {
            model: path("Model")?,
            reference: path("Reference")?,
        },
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn focused_loop_input_formula_and_reference_shape_guard() -> Result<()> {
        if features(0, 2, 128)?.len() != 256
            || features(-1, 1, 1).is_ok()
            || bonus(-1, 256, 255)? == 255
        {
            return Err(invalid("loop closed-form input bounds differ"));
        }
        let mut reference = super::super::dspark_reference::Reference::new()?;
        if reference.forward(&[], &[], &[], &[], None).is_ok()
            || reference.sample(&[0.0], 1).is_ok()
        {
            return Err(invalid("independent reference accepted malformed shapes"));
        }
        Ok(())
    }
}
