//! Full generated-model source contracts for K-cache mean-centering.
//!
//! Metadata, original Windows random inputs, schedules and verdicts are Rust.
//! Production APIs initialize the model, write/load biases and produce logits.
//! Pinned upstream b87ca7225456b06b0975b8239c66b5aecf07e37c (MIT).

use super::engine::Engine;
use super::generated_inputs::{self as inputs, WeightEvidence};
use super::model_session::{Client, ModelSession};
use super::windows_random::Random;
use super::{Result, invalid, result};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const ID: &str = "tests/test-kv-mean-center.cpp";
const SOURCE_SHA: &str = "376c3c7b5b00811e06d30c2dc275f76c6b434cc87464f90ee8eac25de01c01e9";
const VOCAB: u32 = 128;
#[derive(Default)]
struct Checks {
    source: usize,
}
impl Checks {
    fn check(&mut self, passed: bool, message: &str) -> Result<()> {
        self.source += 1;
        if passed {
            Ok(())
        } else {
            Err(invalid(message))
        }
    }
}
fn invoke(client: &Client, operation: Value) -> Result<Value> {
    Ok(client.call(0, operation, None)?.value)
}
fn handle(value: &Value, key: &str) -> Result<u64> {
    value[key]
        .as_u64()
        .filter(|&v| v > 0)
        .ok_or_else(|| invalid(format!("missing generated model handle {key}")))
}
fn release(client: &Client, kind: &str, id: u64) -> Result<()> {
    invoke(client, json!({"Op":"release","Kind":kind,"Handle":id}))?;
    Ok(())
}
fn metadata(directory: &Path) -> Result<PathBuf> {
    let entries = vec![
        inputs::string("general.architecture", "llama")?,
        inputs::u32_value("llama.vocab_size", VOCAB),
        inputs::u32_value("llama.context_length", 128),
        inputs::u32_value("llama.embedding_length", 256),
        inputs::u32_value("llama.block_count", 2),
        inputs::u32_value("llama.feed_forward_length", 384),
        inputs::boolean("llama.use_parallel_residual", false),
        inputs::u32_value("llama.attention.head_count", 2),
        inputs::u32_value("llama.attention.head_count_kv", 2),
        inputs::f32_value("llama.attention.layer_norm_rms_epsilon", 1e-5),
        inputs::u32_array("llama.rope.dimension_sections", &[32, 32, 32, 32])?,
        inputs::string("tokenizer.ggml.model", "no_vocab")?,
    ];
    let path = directory.join("kv-model-metadata.gguf");
    inputs::metadata(&path, &entries, &[])?;
    Ok(path)
}
fn model(
    session: &ModelSession,
    client: &Client,
    path: &Path,
    seed: u64,
    checks: &mut Checks,
) -> Result<(u64, Arc<Mutex<WeightEvidence>>)> {
    let (weights, evidence) =
        inputs::weights(session.evidence_directory().to_owned(), seed, false, true);
    let value = client
        .call_with_handlers(
            0,
            json!({"Op":"model-user","MetadataFile":path,"Devices":null,"SplitMode":1}),
            None,
            Some(weights),
        )?
        .value;
    checks.check(
        !value["Model"].is_null(),
        "failed to create original tiny llama model",
    )?;
    checks.source += evidence
        .lock()
        .map_err(|_| invalid("weight receipt poisoned"))?
        .source_type_checks;
    Ok((handle(&value, "Model")?, evidence))
}
fn context(client: &Client, model: u64, kind: u32, path: Option<&str>) -> Result<Value> {
    let mut overrides =
        json!({"Context":0,"Batch":32,"MicroBatch":32,"Threads":2,"ThreadsBatch":2,"TypeK":kind});
    if kind == 2 {
        overrides["FlashAttention"] = json!(1);
    }
    if let Some(path) = path {
        overrides["MeanCenterPath"] = json!(path);
    }
    invoke(
        client,
        json!({"Op":"context","Model":model,"Params":null,"Overrides":overrides}),
    )
}
fn require_context(value: &Value, checks: &mut Checks) -> Result<u64> {
    checks.check(
        !value["Context"].is_null(),
        "original context must be nonnull",
    )?;
    handle(value, "Context")
}
fn tokens(seed: u32) -> Result<Vec<u32>> {
    let mut random = Random::new(seed);
    (0..16).map(|_| random.bounded32(VOCAB)).collect()
}
fn logits(client: &Client, ctx: u64, tokens: &[u32], checks: &mut Checks) -> Result<Vec<f32>> {
    let batch = handle(
        &invoke(
            client,
            json!({"Op":"batch-new","Capacity":tokens.len(),"Sequences":1}),
        )?,
        "Batch",
    )?;
    let rows:Vec<_>=tokens.iter().enumerate().map(|(position,&token)|json!({"Token":token,"Position":position,"Sequences":[0],"Logits":true})).collect();
    invoke(client, json!({"Op":"batch-add","Batch":batch,"Rows":rows}))?;
    let decoded = invoke(client, json!({"Op":"decode","Context":ctx,"Batch":batch}))?;
    checks.check(
        decoded["Code"].as_i64() == Some(0),
        "original generated-model decode failed",
    )?;
    let indices: Vec<_> = (0..tokens.len()).collect();
    let value = invoke(
        client,
        json!({"Op":"context-logits","Context":ctx,"Indices":indices}),
    )?;
    if value["Vocabulary"] != VOCAB {
        return Err(invalid(
            "generated model vocabulary differs from source metadata",
        ));
    }
    let values = value["Values"]
        .as_array()
        .ok_or_else(|| invalid("missing production logits rows"))?;
    if values.len() != tokens.len() {
        return Err(invalid("generated logits row count differs"));
    }
    let mut logits = Vec::with_capacity(tokens.len() * usize::try_from(VOCAB)?);
    for row in values {
        if row["Present"].as_bool() != Some(true) {
            return Err(invalid("production logits pointer is null"));
        }
        let data = inputs::floats(&row["DataHex"])?;
        if data.len() != usize::try_from(VOCAB)? {
            return Err(invalid("generated logit column count differs"));
        }
        logits.extend(data);
    }
    release(client, "batch", batch)?;
    Ok(logits)
}
fn write_bias(
    client: &Client,
    name: &str,
    layers: &[Vec<f32>],
    rotated: bool,
    checks: &mut Checks,
) -> Result<()> {
    let values = layers
        .iter()
        .enumerate()
        .map(|(index, bias)| {
            let bytes: Vec<_> = bias
                .iter()
                .flat_map(|value| value.to_bits().to_le_bytes())
                .collect();
            Ok(json!({"Index":index,"BiasHex":inputs::hex(&bytes)?}))
        })
        .collect::<Result<Vec<_>>>()?;
    let value = invoke(
        client,
        json!({"Op":"kv-center-write","Name":name,"Rotated":rotated,"Layers":values}),
    )?;
    checks.check(
        value["Written"].as_bool() == Some(true),
        "production KV mean-center writer failed",
    )
}
fn evidence(value: &Arc<Mutex<WeightEvidence>>) -> Result<Value> {
    let state = value
        .lock()
        .map_err(|_| invalid("weight receipt poisoned"))?;
    Ok(
        json!({"TensorInitializers":state.tensors,"FloatInputBytes":state.bytes,"SourceF32TypeChecks":state.source_type_checks}),
    )
}
#[allow(
    clippy::float_cmp,
    reason = "The original disabled-centering regression requires exact float equality for every logit."
)]
fn regression(
    session: &ModelSession,
    client: &Client,
    path: &Path,
    checks: &mut Checks,
) -> Result<Value> {
    let (model, weights) = model(session, client, path, 1234, checks)?;
    let tokens = tokens(42)?;
    let ctx1 = require_context(&context(client, model, 1, None)?, checks)?;
    let first = logits(client, ctx1, &tokens, checks)?;
    let ctx2 = require_context(&context(client, model, 1, None)?, checks)?;
    let second = logits(client, ctx2, &tokens, checks)?;
    checks.check(
        first.len() == second.len(),
        "regression logits lengths differ",
    )?;
    for (&left, &right) in first.iter().zip(&second) {
        checks.check(left == right, "disabled centering changed F16 logits")?;
    }
    let ctx3 = require_context(&context(client, model, 2, None)?, checks)?;
    let quantized = logits(client, ctx3, &tokens, checks)?;
    for value in quantized {
        checks.check(
            value.is_finite(),
            "Q4 cache without bias produced nonfinite logits",
        )?;
    }
    for ctx in [ctx1, ctx2, ctx3] {
        release(client, "context", ctx)?;
    }
    release(client, "model", model)?;
    Ok(json!({"Seed":1234,"TokenSeed":42,"Tokens":tokens,"Weights":evidence(&weights)?}))
}
fn gate(
    session: &ModelSession,
    client: &Client,
    path: &Path,
    checks: &mut Checks,
) -> Result<Value> {
    let (model, weights) = model(session, client, path, 2024, checks)?;
    let layers = vec![vec![0.25; 256]; 2];
    let rotated = "test-kv-mean-center-gate.gguf";
    let unrotated = "test-kv-mean-center-gate-unrot.gguf";
    write_bias(client, rotated, &layers, true, checks)?;
    write_bias(client, unrotated, &layers, false, checks)?;
    let wrong_type = context(client, model, 1, Some(unrotated))?;
    checks.check(
        wrong_type["Context"].is_null(),
        "F16 cache accepted public KV mean-center gate",
    )?;
    let wrong_basis = context(client, model, 2, Some(unrotated))?;
    checks.check(
        wrong_basis["Context"].is_null(),
        "rotated Q4 cache accepted unrotated bias",
    )?;
    let good = require_context(&context(client, model, 2, Some(rotated))?, checks)?;
    let tokens = tokens(3)?;
    for value in logits(client, good, &tokens, checks)? {
        checks.check(value.is_finite(), "Q4 centered logits are nonfinite")?;
    }
    release(client, "context", good)?;
    release(client, "model", model)?;
    Ok(json!({"Seed":2024,"TokenSeed":3,"Tokens":tokens,"Weights":evidence(&weights)?}))
}
#[allow(
    clippy::suboptimal_flops,
    reason = "Separate fp64 products and sums preserve the original numerical reference and tolerance."
)]
fn nmse(a: &[f32], b: &[f32], checks: &mut Checks) -> Result<f64> {
    checks.check(a.len() == b.len(), "NMSE source input lengths differ")?;
    let (mut numerator, mut denominator) = (0.0, 0.0);
    for (&a, &b) in a.iter().zip(b) {
        let a = f64::from(a);
        let difference = a - f64::from(b);
        numerator += difference * difference;
        denominator += a * a;
    }
    Ok(numerator / denominator)
}
fn softmax(
    session: &ModelSession,
    client: &Client,
    path: &Path,
    checks: &mut Checks,
) -> Result<Value> {
    let (model, weights) = model(session, client, path, 5678, checks)?;
    let tokens = tokens(7)?;
    let base = require_context(&context(client, model, 0, None)?, checks)?;
    let baseline = logits(client, base, &tokens, checks)?;
    let mut random = Random::new(99);
    let layers: Vec<Vec<f32>> = (0..2)
        .map(|_| (0..256).map(|_| random.uniform_f32(-0.5, 0.5)).collect())
        .collect();
    let bias = "test-kv-mean-center-bias.gguf";
    write_bias(client, bias, &layers, false, checks)?;
    let centered = require_context(&context(client, model, 0, None)?, checks)?;
    let loaded = invoke(
        client,
        json!({"Op":"kv-center-load","Context":centered,"Name":bias,"RequireQ4":false}),
    )?;
    checks.check(
        loaded["CacheExists"].as_bool() == Some(true),
        "production memory is not a llama_kv_cache",
    )?;
    checks.check(
        loaded["Loaded"].as_bool() == Some(true),
        "production F32 bias test-seam load failed",
    )?;
    let actual = logits(client, centered, &tokens, checks)?;
    checks.check(
        baseline.len() == actual.len(),
        "softmax-invariance logits lengths differ",
    )?;
    let error = nmse(&baseline, &actual, checks)?;
    checks.check(error < 1e-8, "F32 softmax-invariance NMSE exceeded1e-8")?;
    release(client, "context", centered)?;
    release(client, "context", base)?;
    release(client, "model", model)?;
    Ok(
        json!({"Seed":5678,"TokenSeed":7,"BiasSeed":99,"Tokens":tokens,"NMSE":error,"Weights":evidence(&weights)?}),
    )
}
pub(super) fn run(engine: &Engine) -> Result<Value> {
    let Some(session) = ModelSession::open(engine, "kv-mean-center")? else {
        return Ok(result(
            ID,
            false,
            "production generated-model/K-cache API unavailable".into(),
        ));
    };
    let client = session.client();
    let mut checks = Checks::default();
    let outcome = (|| {
        let path = metadata(session.evidence_directory())?;
        let first = regression(&session, &client, &path, &mut checks)?;
        let second = gate(&session, &client, &path, &mut checks)?;
        let third = softmax(&session, &client, &path, &mut checks)?;
        Ok(json!({"Regression":first,"PublicGate":second,"SoftmaxInvariance":third}))
    })();
    let finished = session.finish();
    let (passed, observation) = match (outcome, finished) {
        (Ok(value), Ok(())) => (true, value),
        (Err(error), _) | (_, Err(error)) => (false, json!({"Error":error.to_string()})),
    };
    Ok(result(ID,passed,json!({"SourceSHA256":SOURCE_SHA,"SourceChecks":checks.source,"FixedDirectAssertions":6161,"DecodeSuccessPredicates":6,"ModelNonnullPredicates":3,"MetadataAndAllWeightsGeneratedByRust":true,"Observation":observation}).to_string()))
}

#[cfg(test)]
mod focused_guards {
    use super::*;
    #[test]
    fn source_nmse_double_arithmetic_and_changed_logits_fail() -> Result<()> {
        let mut checks = Checks::default();
        assert_eq!(nmse(&[1.0, 2.0], &[1.0, 2.0], &mut checks)?, 0.0);
        assert!(nmse(&[1.0, 2.0], &[1.0, 3.0], &mut checks)? > 1e-8);
        assert!(nmse(&[1.0], &[], &mut checks).is_err());
        assert!(inputs::floats(&json!("0000000g")).is_err());
        Ok(())
    }
}
