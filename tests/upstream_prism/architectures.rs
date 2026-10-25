//! Full architecture/backend and model-export schedules from the pinned source.
//! Metadata literals, Windows random inputs, comparisons and skips belong to Rust.
use super::engine::Engine;
use super::generated_inputs::{self as inputs, Entry};
use super::model_session::{Client, ModelSession};
use super::windows_random::Random;
use super::{Result, invalid, result};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const ID: &str = "tests/test-llama-archs.cpp";
const SOURCE_SHA: &str = "7d542af7b20f6e43d2b174f1dddf9b9480a3130db36c005c3f17c164c032b44d";
#[derive(Clone, Debug)]
pub(super) struct ArchitectureOptions {
    pub(super) architecture: Option<String>,
    pub(super) seed: u64,
    pub(super) verbose: bool,
    pub(super) output: Option<PathBuf>,
}
impl Default for ArchitectureOptions {
    fn default() -> Self {
        Self {
            architecture: None,
            seed: 1234,
            verbose: false,
            output: None,
        }
    }
}
fn fixture() -> Result<Value> {
    Ok(serde_json::from_str(include_str!(
        "architecture-inputs.json"
    ))?)
}
fn array<'a>(v: &'a Value, key: &str) -> Result<&'a [Value]> {
    v[key]
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| invalid(format!("missing architecture array {key}")))
}
fn text<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v[key]
        .as_str()
        .ok_or_else(|| invalid(format!("missing architecture string {key}")))
}
pub(super) fn invoke(client: &Client, operation: Value) -> Result<Value> {
    Ok(client.call(0, operation, None)?.value)
}
pub(super) fn handle(v: &Value, key: &str) -> Result<u64> {
    v[key]
        .as_u64()
        .filter(|&n| n > 0)
        .ok_or_else(|| invalid(format!("missing architecture handle {key}")))
}
pub(super) fn release(client: &Client, kind: &str, id: u64) -> Result<()> {
    invoke(client, json!({"Op":"release","Kind":kind,"Handle":id}))?;
    Ok(())
}
fn check(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(invalid(message)) }
}
struct Metadata<'a> {
    arch: &'a str,
    keys: &'a Value,
    entries: Vec<(String, Entry)>,
}
impl Metadata<'_> {
    fn name(&self, key: &str) -> Result<String> {
        Ok(self.keys[key]
            .as_str()
            .ok_or_else(|| invalid(format!("missing metadata key {key}")))?
            .replace("%s", self.arch))
    }
    fn put(&mut self, key: &str, value: Entry) {
        if let Some(slot) = self.entries.iter_mut().find(|(k, _)| k == key) {
            slot.1 = value;
        } else {
            self.entries.push((key.into(), value));
        }
    }
    fn u(&mut self, key: &str, value: u32) -> Result<()> {
        let e = inputs::u32_value(&self.name(key)?, value);
        self.put(key, e);
        Ok(())
    }
    fn f(&mut self, key: &str, value: f32) -> Result<()> {
        let e = inputs::f32_value(&self.name(key)?, value);
        self.put(key, e);
        Ok(())
    }
    fn b(&mut self, key: &str, value: bool) -> Result<()> {
        let e = inputs::boolean(&self.name(key)?, value);
        self.put(key, e);
        Ok(())
    }
    fn s(&mut self, key: &str, value: &str) -> Result<()> {
        let e = inputs::string(&self.name(key)?, value)?;
        self.put(key, e);
        Ok(())
    }
    fn ua(&mut self, key: &str, value: &[u32]) -> Result<()> {
        let e = inputs::u32_array(&self.name(key)?, value)?;
        self.put(key, e);
        Ok(())
    }
    fn fa(&mut self, key: &str, value: &[f32]) -> Result<()> {
        let e = inputs::f32_array(&self.name(key)?, value)?;
        self.put(key, e);
        Ok(())
    }
}
fn among(arch: &str, values: &[&str]) -> bool {
    values.contains(&arch)
}
#[allow(
    clippy::too_many_lines,
    reason = "literal source metadata branches stay together for parity review"
)]
fn metadata(
    directory: &Path,
    row: &Value,
    moe: bool,
    backend: bool,
    fixture: &Value,
) -> Result<(PathBuf, usize)> {
    let arch = text(row, "Name")?;
    let en = text(row, "Enum")?;
    let (mut vocab, mut embd, mut heads, mut ff, mut layers) = (128, 256, 2, 384, 2);
    match en {
        "LLAMA4" => layers = 4,
        "GEMMA4" => {
            embd = 128;
            heads = 2;
            ff = 192;
            layers = 5;
        }
        "GEMMA3N" => {
            embd = 64;
            heads = 1;
            ff = 96;
            layers = 22;
        }
        "DEEPSEEK4" => {
            embd = 512;
            heads = 8;
            ff = 1024;
            layers = 4;
        }
        "STEP35" | "LAGUNA" => embd = 160,
        "QWEN3" | "MUSE_GLIMMER" | "AFMOE" => heads = 4,
        "DEEPSEEK2" | "DEEPSEEK32" | "GLM_DSA" | "DOTS3NOTE" | "KIMI_LINEAR" | "BAILINGMOE3"
        | "KIMI_K3" | "MISTRAL4" => {
            embd = 128;
            heads = 1;
            ff = 192;
        }
        "NEMOTRON_H" | "NEMOTRON_H_MOE" => layers = 3,
        "CHAMELEON" => vocab = 10240,
        "QWEN3TTS" => vocab = 4096,
        _ => {}
    }
    let head_kv = match en {
        "QWEN3" => 1,
        "MUSE_GLIMMER" | "AFMOE" => 2,
        _ => heads,
    };
    let dim = embd / heads;
    let mut m = Metadata {
        arch,
        keys: &fixture["KeyMap"],
        entries: Vec::new(),
    };
    m.s("GENERAL_ARCHITECTURE", arch)?;
    for (k, v) in [
        ("VOCAB_SIZE", vocab),
        ("CONTEXT_LENGTH", 128),
        ("EMBEDDING_LENGTH", embd),
        ("FEATURES_LENGTH", embd),
        ("BLOCK_COUNT", layers),
        ("LEADING_DENSE_BLOCK_COUNT", 1),
    ] {
        m.u(k, v)?;
    }
    if among(en, &["NEMOTRON_H", "NEMOTRON_H_MOE"]) {
        m.ua(
            "FEED_FORWARD_LENGTH",
            &(0..layers)
                .map(|il| if il <= 1 { 0 } else { ff })
                .collect::<Vec<_>>(),
        )?;
    } else {
        m.u("FEED_FORWARD_LENGTH", ff)?;
    }
    m.b("USE_PARALLEL_RESIDUAL", false)?;
    m.f("LOGIT_SCALE", 1.0)?;
    for (k, v) in [
        ("TIME_MIX_EXTRA_DIM", 64),
        ("TIME_DECAY_EXTRA_DIM", 128),
        ("FULL_ATTENTION_INTERVAL", 2),
    ] {
        m.u(k, v)?;
    }
    let mut checks = 0;
    if among(
        en,
        &[
            "PLAMO2",
            "JAMBA",
            "NEMOTRON_H",
            "NEMOTRON_H_MOE",
            "GRANITE_HYBRID",
            "LFM2",
            "LFM2MOE",
            "KIMI_LINEAR",
            "BAILINGMOE3",
            "KIMI_K3",
        ],
    ) {
        check(layers >= 2, "source hybrid layer assertion failed")?;
        checks += 1;
        let values: Vec<_> = (0..layers)
            .map(|il| if il == 1 { 0 } else { heads })
            .collect();
        m.ua("ATTENTION_HEAD_COUNT", &values)?;
        m.ua("ATTENTION_HEAD_COUNT_KV", &values)?;
    } else {
        m.u("ATTENTION_HEAD_COUNT", heads)?;
        m.u(
            "ATTENTION_HEAD_COUNT_KV",
            if en == "DEEPSEEK4" { 1 } else { head_kv },
        )?;
    }
    m.f("ATTENTION_MAX_ALIBI_BIAS", 8.0)?;
    if en == "DEEPSEEK4" {
        m.u("ATTENTION_KEY_LENGTH", dim)?;
        m.u("ATTENTION_VALUE_LENGTH", dim)?;
        m.u("ROPE_DIMENSION_COUNT", dim / 2)?;
    } else if among(
        en,
        &[
            "DEEPSEEK2",
            "DEEPSEEK32",
            "GLM_DSA",
            "DOTS3NOTE",
            "KIMI_LINEAR",
            "BAILINGMOE3",
            "KIMI_K3",
            "MISTRAL4",
        ],
    ) {
        for (k, v) in [
            ("ATTENTION_KEY_LENGTH", 576),
            ("ATTENTION_VALUE_LENGTH", 512),
            ("ROPE_DIMENSION_COUNT", 64),
            ("ATTENTION_KEY_LENGTH_MLA", 192),
            ("ATTENTION_VALUE_LENGTH_MLA", 128),
        ] {
            m.u(k, v)?;
        }
        if en == "DOTS3NOTE" {
            for (k, v) in [
                ("ATTENTION_KV_LORA_RANK_SWA", 512),
                ("ATTENTION_KEY_LENGTH_SWA", 576),
                ("ATTENTION_VALUE_LENGTH_SWA", 512),
                ("ATTENTION_KEY_LENGTH_MLA_SWA", 192),
                ("ATTENTION_VALUE_LENGTH_MLA_SWA", 128),
            ] {
                m.u(k, v)?;
            }
            m.f("ROPE_FREQ_BASE_SWA", 10000.0)?;
            m.ua(
                "ATTENTION_INDEXER_TYPES",
                &(0..layers)
                    .map(|i| u32::from(i % 2 == 0))
                    .collect::<Vec<_>>(),
            )?;
        }
    } else if en == "MINIMAX_M3" {
        m.u("ROPE_DIMENSION_COUNT", 64)?;
    }
    for (k, v) in [
        ("ATTENTION_CLAMP_KQV", 1.0),
        ("ATTENTION_LAYERNORM_EPS", 1e-5),
        ("ATTENTION_LAYERNORM_RMS_EPS", 1e-5),
        ("ATTENTION_GROUPNORM_EPS", 1e-5),
    ] {
        m.f(k, v)?;
    }
    for (k, v) in [
        ("ATTENTION_GROUPNORM_GROUPS", 8),
        (
            "ATTENTION_Q_LORA_RANK",
            if en == "DEEPSEEK4" { 64 } else { 512 },
        ),
        ("ATTENTION_KV_LORA_RANK", 512),
        ("ATTENTION_RELATIVE_BUCKETS_COUNT", 8),
        ("ATTENTION_SLIDING_WINDOW", 16),
    ] {
        m.u(k, v)?;
    }
    if en == "GEMMA4" {
        for (k, v) in [
            ("EMBEDDING_LENGTH_PER_LAYER", embd / 2),
            ("ATTENTION_SHARED_KV_LAYERS", 0),
            ("ATTENTION_KEY_LENGTH_SWA", dim),
            ("ATTENTION_VALUE_LENGTH_SWA", dim),
            ("ATTENTION_SLIDING_WINDOW_PATTERN", 5),
        ] {
            m.u(k, v)?;
        }
        m.f("ROPE_FREQ_BASE_SWA", 10000.0)?;
    } else if among(
        en,
        &[
            "COHERE2MOE",
            "MIMO2",
            "STEP35",
            "MUSE_GLIMMER",
            "GRANITE_SWA",
            "DOTS3NOTE",
        ],
    ) {
        m.ua(
            "ATTENTION_SLIDING_WINDOW_PATTERN",
            &(0..layers).map(|i| i % 2).collect::<Vec<_>>(),
        )?;
    } else {
        m.u("ATTENTION_SLIDING_WINDOW_PATTERN", 2)?;
    }
    for (k, v) in [
        (
            "ATTENTION_INDEXER_HEAD_COUNT",
            if among(en, &["MINIMAX_M3", "DEEPSEEK4"]) {
                heads
            } else {
                1
            },
        ),
        ("ATTENTION_INDEXER_KEY_LENGTH", 64),
        ("ATTENTION_INDEXER_TOP_K", 8),
        ("ATTENTION_INDEXER_BLOCK_SIZE", 4),
        ("ATTENTION_INDEXER_LOCAL_BLOCKS", 1),
    ] {
        m.u(k, v)?;
    }
    m.ua("ROPE_DIMENSION_SECTIONS", &[dim / 4; 4])?;
    if en == "DEEPSEEK4" {
        for (k, v) in [
            ("ATTENTION_OUTPUT_GROUP_COUNT", 8),
            ("ATTENTION_OUTPUT_LORA_RANK", 32),
            ("HYPER_CONNECTION_COUNT", 4),
            ("HYPER_CONNECTION_SINKHORN_ITERATIONS", 2),
            ("HASH_LAYER_COUNT", 0),
        ] {
            m.u(k, v)?;
        }
        m.ua("ATTENTION_COMPRESS_RATIOS", &[0, 0, 4, 128])?;
        for (k, v) in [
            ("ATTENTION_COMPRESS_ROPE_FREQ_BASE", 160_000.0),
            ("HYPER_CONNECTION_EPSILON", 1e-6),
            ("SWIGLU_CLAMP_EXP", 10.0),
            ("EXPERT_WEIGHTS_SCALE", 1.0),
        ] {
            m.f(k, v)?;
        }
        m.b("EXPERT_WEIGHTS_NORM", true)?;
    }
    m.s("TOKENIZER_MODEL", "no_vocab")?;
    if moe {
        for (k, v) in [
            ("EXPERT_FEED_FORWARD_LENGTH", ff),
            ("EXPERT_SHARED_FEED_FORWARD_LENGTH", ff / 2),
            ("EXPERT_LATENT_LENGTH", ff),
            ("INTERLEAVE_MOE_LAYER_STEP", 2),
            ("EXPERT_COUNT", 2),
            ("EXPERT_USED_COUNT", 1),
            ("EXPERT_SHARED_COUNT", 1),
            ("EXPERT_GATING_FUNC", if en == "DEEPSEEK4" { 4 } else { 2 }),
            ("EXPERTS_PER_GROUP", 1),
        ] {
            m.u(k, v)?;
        }
        m.f("EXPERT_GROUP_SCALE", 1.0)?;
    }
    for (k, v) in [
        ("POSNET_EMBEDDING_LENGTH", embd),
        ("POSNET_BLOCK_COUNT", layers),
        ("CONVNEXT_EMBEDDING_LENGTH", embd),
        ("CONVNEXT_BLOCK_COUNT", layers),
    ] {
        m.u(k, v)?;
    }
    for (k, v) in [
        ("XIELU_ALPHA_N", 1.0),
        ("XIELU_ALPHA_P", 1.0),
        ("XIELU_BETA", 1.0),
        ("XIELU_EPS", 1e-7),
    ] {
        m.f(k, v)?;
    }
    for (k, v) in [
        (
            "SSM_INNER_SIZE",
            if among(en, &["QWEN3NEXT", "QWEN35", "QWEN35MOE"]) {
                256
            } else {
                2 * embd
            },
        ),
        ("SSM_CONV_KERNEL", 4),
        ("SSM_STATE_SIZE", 128),
        ("SSM_TIME_STEP_RANK", heads),
        ("SSM_GROUP_COUNT", if en == "PLAMO2" { 0 } else { 2 }),
        ("KDA_HEAD_DIM", 128),
    ] {
        m.u(k, v)?;
    }
    m.b("KDA_SAFE_GATE", true)?;
    m.f("KDA_GATE_LOWER_BOUND", -5.0)?;
    if en == "BAILINGMOE3" {
        m.fa("SWIGLU_CLAMP_EXP", &[0.0, 4.0])?;
        m.fa("SWIGLU_CLAMP_SHEXP", &[0.0, 5.0])?;
    }
    m.u("WKV_HEAD_SIZE", dim)?;
    m.u("SHORTCONV_L_CACHE", 3)?;
    #[allow(
        clippy::excessive_precision,
        reason = "preserve the source floating literal rounded to f32"
    )]
    let residual = 3.556_558_820_077_845_5_f32;
    m.f("RESIDUAL_SCALE", residual)?;
    m.u("ATTN_RES_BLOCK_SIZE", 12)?;
    m.f("ACTIVATION_SITU_BETA", 4.0)?;
    m.f("ACTIVATION_SITU_LINEAR_BETA", 25.0)?;
    m.f("KDA_GATE_LOWER_BOUND", -5.0)?;
    if en == "DSPARK" {
        // The pinned generic fixture predates these required loader keys.
        // Span all source tokens so every requested row remains a draft output.
        m.u("DSPARK_BLOCK_SIZE", 128)?;
        m.u("DSPARK_MASK_TOKEN_ID", 127)?;
        m.ua("DSPARK_TARGET_LAYERS", &[1, 16])?;
    }
    if backend && en == "BAILINGMOE3" {
        let old = m.entries.len();
        m.entries.retain(|(k, _)| k != "KDA_SAFE_GATE");
        check(
            m.entries.len() + 1 == old,
            "source safe-gate removal assertion failed",
        )?;
        checks += 1;
    }
    let hints: Vec<_> = (0..layers)
        .flat_map(|i| {
            [
                format!("conv{i}d.weight"),
                format!("posnet.{i}.conv1.weight"),
                format!("posnet.{i}.conv2.weight"),
                format!("convnext.{i}.dw.weight"),
            ]
        })
        .map(|n| (n, 1))
        .collect();
    let path = directory.join("architecture-metadata.gguf");
    inputs::metadata(
        &path,
        &m.entries.into_iter().map(|(_, e)| e).collect::<Vec<_>>(),
        &hints,
    )?;
    Ok((path, checks))
}
fn supported(row: &Value, inventory: &Value) -> bool {
    row["Supported"] == true
        && !(inventory["WebGpu"] == true
            && among(
                row["Enum"].as_str().unwrap_or(""),
                &["DEEPSEEK32", "GLM_DSA", "DOTS3NOTE"],
            ))
        && !(inventory["Hip"] == true && row["Enum"] == "JAMBA")
}
fn variants(fixture: &Value, filter: Option<&str>) -> Result<Vec<(Value, bool)>> {
    if let Some(name) = filter {
        check(
            array(fixture, "Architectures")?
                .iter()
                .any(|r| r["Name"] == name && r["Enum"] != "UNKNOWN"),
            "unknown LLM architecture",
        )?;
    }
    let mut ret = Vec::new();
    for row in array(fixture, "Architectures")? {
        if row["LoopExcluded"] == true || filter.is_some_and(|name| row["Name"] != name) {
            continue;
        }
        for moe in [false, true] {
            if (!moe && row["MandatoryMoe"] == true)
                || (moe && row["MandatoryMoe"] != true && row["OptionalMoe"] != true)
            {
                continue;
            }
            ret.push((row.clone(), moe));
        }
    }
    Ok(ret)
}
fn inventory(engine: &Engine) -> Result<Value> {
    let session = ModelSession::open(engine, "architecture-inventory")?
        .ok_or_else(|| invalid("missing production architecture/model API"))?;
    let c = session.client();
    let out: Result<Value> = (|| {
        invoke(&c, json!({"Op":"common-init"}))?;
        let mut value = invoke(&c, json!({"Op":"architecture-inventory"}))?;
        value["Devices"] = invoke(&c, json!({"Op":"devices"}))?["Devices"].clone();
        Ok(value)
    })();
    let finish = session.finish();
    let value = out?;
    finish?;
    Ok(value)
}
fn capability<'a>(inventory: &'a Value, name: &str) -> Result<&'a Value> {
    array(inventory, "Architectures")?
        .iter()
        .find(|v| v["Name"] == name)
        .ok_or_else(|| invalid(format!("production architecture missing {name}")))
}
fn init_model(
    session: &ModelSession,
    c: &Client,
    path: &Path,
    devices: &[Value],
    split: u32,
    weights: &inputs::TensorFunction,
) -> Result<u64> {
    let _ = session;
    handle(
        &c.call_with_handlers(
            0,
            json!({"Op":"model-user","MetadataFile":path,"Devices":devices,"SplitMode":split}),
            None,
            Some(Arc::clone(weights)),
        )?
        .value,
        "Model",
    )
}
fn context(c: &Client, model: u64, encode: bool) -> Result<u64> {
    let mut overrides = json!({"Context":0,"Threads":4,"ThreadsBatch":4});
    if !encode {
        overrides["MicroBatch"] = json!(64);
    }
    handle(
        &invoke(
            c,
            json!({"Op":"context","Model":model,"Params":null,"Overrides":overrides}),
        )?,
        "Context",
    )
}
fn get_logits(c: &Client, ctx: u64, tokens: &[u32], encode: bool) -> Result<Vec<f32>> {
    let size = invoke(c, json!({"Op":"context-metadata","Context":ctx}))?["ContextSize"]
        .as_u64()
        .ok_or_else(|| invalid("context size absent"))?;
    check(
        u64::try_from(tokens.len())? <= size,
        "source token count exceeds context",
    )?;
    let b = handle(
        &invoke(c, json!({"Op":"batch-new","Capacity":size,"Sequences":1}))?,
        "Batch",
    )?;
    let rows: Vec<_> = tokens
        .iter()
        .enumerate()
        .map(|(p, t)| json!({"Token":t,"Position":p,"Sequences":[0],"Logits":true}))
        .collect();
    invoke(c, json!({"Op":"batch-add","Batch":b,"Rows":rows}))?;
    if encode {
        check(
            invoke(c, json!({"Op":"encode","Context":ctx,"Batch":b}))?["Code"] == 0,
            "failed to encode original batch",
        )?;
    }
    check(
        invoke(c, json!({"Op":"decode","Context":ctx,"Batch":b}))?["Code"] == 0,
        "failed to decode original batch",
    )?;
    let values = invoke(
        c,
        json!({"Op":"context-logits","Context":ctx,"Indices":(0..tokens.len()).collect::<Vec<_>>()}),
    )?;
    let mut out = Vec::new();
    for v in array(&values, "Values")? {
        check(v["Present"] == true, "production logit pointer absent")?;
        out.extend(inputs::floats(&v["DataHex"])?);
    }
    release(c, "batch", b)?;
    Ok(out)
}
#[allow(
    clippy::suboptimal_flops,
    reason = "source calculates float products before accumulating doubles"
)]
fn nmse(a: &[f32], b: &[f32]) -> Result<f64> {
    check(a.len() == b.len(), "source NMSE shape assertion failed")?;
    let (mut delta, mut norm) = (0.0_f64, 0.0_f64);
    for (&x, &y) in a.iter().zip(b) {
        delta += f64::from((x - y) * (x - y));
        norm += f64::from(x * x);
    }
    Ok(delta / norm)
}
#[allow(
    clippy::too_many_lines,
    reason = "preserve the source backend and roundtrip schedule in order"
)]
fn backend_variant(
    engine: &Engine,
    row: &Value,
    moe: bool,
    options: &ArchitectureOptions,
    inventory: &Value,
    fixture: &Value,
) -> Result<Value> {
    let name = text(row, "Name")?;
    let encode = among(
        text(row, "Enum")?,
        &["T5", "DREAM", "LLADA", "LLADA_MOE", "RND1"],
    );
    let session = ModelSession::open(engine, &format!("arch-{name}-{moe}"))?
        .ok_or_else(|| invalid("missing production architecture model API"))?;
    let c = session.client();
    let run: Result<Value> = (|| {
        invoke(&c, json!({"Op":"common-init"}))?;
        invoke(
            &c,
            json!({"Op":"log-minimum","Level":if options.verbose{2}else{4}}),
        )?;
        let (path, metadata_checks) =
            metadata(session.evidence_directory(), row, moe, true, fixture)?;
        let (weights, evidence) = inputs::weights(
            session.evidence_directory().to_owned(),
            options.seed,
            true,
            false,
        );
        let seed = u32::from_le_bytes(options.seed.to_le_bytes()[..4].try_into()?);
        let mut random = Random::new(seed);
        let tokens = (0..128)
            .map(|_| random.bounded32(128))
            .collect::<Result<Vec<_>>>()?;
        let devices = array(inventory, "Devices")?;
        let mut configs: Vec<_> = devices
            .iter()
            .map(|d| (vec![d["Index"].clone()], d["Description"].clone(), 1_u32))
            .collect();
        let meta: Vec<_> = devices
            .iter()
            .filter(|d| d["CpuBuffer"] != true)
            .map(|d| d["Index"].clone())
            .collect();
        configs.push((meta, json!("Meta"), 2));
        let cap = capability(inventory, name)?;
        let mut baseline = None;
        let mut baseline_handles = None;
        let mut rows = Vec::new();
        let mut assertions = metadata_checks;
        for (devs, label, split) in configs {
            if !supported(row, inventory) || (split == 2 && devs.is_empty()) {
                rows.push(json!({"Device":label,"Skipped":"source architecture/backend gate"}));
                continue;
            }
            if baseline.is_none() {
                let m = init_model(&session, &c, &path, &[], 1, &weights)?;
                let ctx = context(&c, m, encode)?;
                let logits = get_logits(&c, ctx, &tokens, encode)?;
                baseline_handles = Some((m, ctx));
                baseline = Some(logits);
            }
            if split == 2 && cap["TensorSplit"] != true {
                rows.push(json!({"Device":label,"Skipped":"production tensor-split capability"}));
                continue;
            }
            let m = init_model(&session, &c, &path, &devs, split, &weights)?;
            let ctx = context(&c, m, encode)?;
            let logits = get_logits(&c, ctx, &tokens, encode)?;
            let error = nmse(
                baseline
                    .as_deref()
                    .ok_or_else(|| invalid("CPU reference missing"))?,
                &logits,
            )?;
            assertions += 2;
            // Preserve source's `error > 1e-4` predicate, including its NaN behavior.
            check(
                error.partial_cmp(&1e-4) != Some(std::cmp::Ordering::Greater),
                "architecture NMSE exceeds source tolerance",
            )?;
            let f = invoke(&c, json!({"Op":"temporary-file"}))?;
            let mut roundtrip = json!("source saver/split gate");
            if let Some(file) = f["File"].as_u64() {
                if cap["Saver"] == true && split != 2 {
                    assertions += 2;
                    invoke(
                        &c,
                        json!({"Op":"model-save-file-pointer","Model":m,"File":file}),
                    )?;
                    let rm = handle(
                        &invoke(
                            &c,
                            json!({"Op":"model-load-file-pointer","File":file,"Devices":devs,"SplitMode":split}),
                        )?,
                        "Model",
                    )?;
                    let rc = context(&c, rm, encode)?;
                    let restored = get_logits(&c, rc, &tokens, encode)?;
                    assertions += 1;
                    check(
                        restored.len() == logits.len(),
                        "roundtrip output shape differs",
                    )?;
                    #[allow(
                        clippy::float_cmp,
                        reason = "source roundtrip requires exact float equality"
                    )]
                    let same = restored.iter().zip(&logits).all(|(a, b)| a == b);
                    assertions += 1;
                    check(same, "roundtrip logits differ from original model")?;
                    roundtrip = json!({"Compared":logits.len()});
                    release(&c, "context", rc)?;
                    release(&c, "model", rm)?;
                }
                release(&c, "file", file)?;
            } else {
                roundtrip = json!("source tmpfile unavailable on Windows");
            }
            rows.push(json!({"Device":label,"SplitMode":split,"Nmse":if error.is_finite(){json!(error)}else{json!(format!("{error}"))},"Roundtrip":roundtrip}));
            release(&c, "context", ctx)?;
            release(&c, "model", m)?;
        }
        if let Some((model, context)) = baseline_handles {
            release(&c, "context", context)?;
            release(&c, "model", model)?;
        }
        let weights = evidence
            .lock()
            .map_err(|_| invalid("architecture weight receipt poisoned"))?;
        let value = json!({"Architecture":name,"Moe":moe,"Encode":encode,"SourceAssertions":assertions,"BackendRows":rows,"FloatInputBytes":weights.bytes,"TensorInputs":weights.tensors,"FixtureRepairs":if name=="dspark"{fixture["FixtureRepairs"].clone()}else{json!([])}});
        drop(weights);
        Ok(value)
    })();
    let finish = session.finish();
    let v = run?;
    finish?;
    Ok(v)
}
fn export_variant(
    engine: &Engine,
    row: &Value,
    moe: bool,
    seed: u64,
    output: &Path,
    fixture: &Value,
    verbose: bool,
) -> Result<Value> {
    let name = text(row, "Name")?;
    let session = ModelSession::open(engine, &format!("arch-export-{name}-{moe}"))?
        .ok_or_else(|| invalid("missing production model saver API"))?;
    let c = session.client();
    let run: Result<Value> = (|| {
        invoke(&c, json!({"Op":"common-init"}))?;
        invoke(&c, json!({"Op":"log-minimum","Level":if verbose{2}else{4}}))?;
        let (path, checks) = metadata(session.evidence_directory(), row, moe, false, fixture)?;
        let (weights, evidence) =
            inputs::weights(session.evidence_directory().to_owned(), seed, true, false);
        let m = init_model(&session, &c, &path, &[], 1, &weights)?;
        let ctx = context(&c, m, false)?;
        let filename = format!("{name}-{}.gguf", if moe { "moe" } else { "dense" });
        let saved = invoke(&c, json!({"Op":"model-save","Model":m,"Name":filename}))?;
        check(
            saved["Exists"] == true && saved["Bytes"].as_u64().is_some_and(|n| n > 0),
            "production model export absent",
        )?;
        let destination = output.join(&filename);
        check(
            !destination.exists(),
            "model export destination already exists",
        )?;
        std::fs::copy(session.evidence_directory().join(&filename), &destination)?;
        release(&c, "context", ctx)?;
        release(&c, "model", m)?;
        let e = evidence
            .lock()
            .map_err(|_| invalid("export weight receipt poisoned"))?;
        Ok(
            json!({"Architecture":name,"Moe":moe,"Path":destination,"Bytes":saved["Bytes"],"MetadataAssertions":checks,"FloatInputBytes":e.bytes,"TensorInputs":e.tensors}),
        )
    })();
    let finish = session.finish();
    let v = run?;
    finish?;
    Ok(v)
}
pub(super) fn export_selected(
    engine: &Engine,
    output: &Path,
    names: &[(&str, bool)],
    seed: u64,
) -> Result<Vec<PathBuf>> {
    check(
        output.is_absolute() && output.is_dir(),
        "generated fixture output must be an external absolute directory",
    )?;
    let f = fixture()?;
    let inv = inventory(engine)?;
    let mut paths = Vec::new();
    for &(name, moe) in names {
        let row = array(&f, "Architectures")?
            .iter()
            .find(|r| r["Name"] == name)
            .ok_or_else(|| invalid("unknown generated fixture architecture"))?;
        check(
            supported(row, &inv) && capability(&inv, name)?["Saver"] == true,
            "original fixture architecture cannot be exported by production",
        )?;
        let value = export_variant(engine, row, moe, seed, output, &f, false)?;
        let path = PathBuf::from(text(&value, "Path")?);
        paths.push(path);
    }
    Ok(paths)
}
#[allow(
    clippy::unnecessary_wraps,
    reason = "shared source adapter entrypoint returns a measured failure result"
)]
pub(super) fn run_with_options(engine: &Engine, options: &ArchitectureOptions) -> Result<Value> {
    let execute: Result<Value> = (|| {
        let f = fixture()?;
        let inv = inventory(engine)?;
        let selected = variants(&f, options.architecture.as_deref())?;
        // llm_arch_all follows enum/map order, which differs from the source name-table order.
        let variants: Vec<_> = array(&inv, "Architectures")?
            .iter()
            .flat_map(|cap| {
                selected
                    .iter()
                    .filter(|(row, _)| row["Name"] == cap["Name"])
                    .cloned()
            })
            .collect();
        check(
            variants.len() == selected.len(),
            "production architecture inventory lost a selected source variant",
        )?;
        for row in array(&f, "Architectures")? {
            let cap = capability(&inv, text(row, "Name")?)?;
            check(
                cap["Id"] == cap["ParsedId"],
                "production architecture name roundtrip differs",
            )?;
        }
        let mut results = Vec::new();
        if let Some(output) = &options.output {
            check(
                output.is_absolute() && output.is_dir(),
                "architecture export output must be an absolute directory",
            )?;
            for (row, moe) in variants {
                if supported(&row, &inv) && capability(&inv, text(&row, "Name")?)?["Saver"] == true
                {
                    results.push(export_variant(
                        engine,
                        &row,
                        moe,
                        options.seed,
                        output,
                        &f,
                        options.verbose,
                    )?);
                } else {
                    results.push(json!({"Architecture":row["Name"],"Moe":moe,"Skipped":"source export capability gate"}));
                }
            }
        } else {
            for (row, moe) in variants {
                results.push(backend_variant(engine, &row, moe, options, &inv, &f)?);
            }
        }
        Ok(
            json!({"SourceSHA256":SOURCE_SHA,"Seed":options.seed,"DefaultSeedPolicy":"replayable choice; original random_device was unpinned","Verbose":options.verbose,"Export":options.output,"Results":results}),
        )
    })();
    match execute {
        Ok(detail) => Ok(result(ID, true, detail.to_string())),
        Err(e) => Ok(result(
            ID,
            false,
            format!("full architecture source contract: {e}"),
        )),
    }
}
pub(super) fn run(engine: &Engine) -> Result<Value> {
    let options = ArchitectureOptions::default();
    let backend = run_with_options(engine, &options)?;
    if backend["Status"] == "fail" {
        return Ok(backend);
    }
    let directory = engine.evidence_path("architecture-model-export")?;
    let export = run_with_options(
        engine,
        &ArchitectureOptions {
            output: Some(directory),
            ..options
        },
    )?;
    Ok(result(
        ID,
        backend["Status"] == "pass" && export["Status"] == "pass",
        json!({"BackendPhase":backend,"ExportPhase":export}).to_string(),
    ))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn focused_architecture_input_and_comparison_guard() -> Result<()> {
        let f = fixture()?;
        check(
            array(&f, "Architectures")?.len() == 149,
            "architecture fixture inventory differs",
        )?;
        let v = variants(&f, None)?;
        check(
            v.len() == 151 && v.iter().filter(|(r, _)| r["Supported"] == true).count() == 128,
            "source variant inventory differs",
        )?;
        check(
            variants(&f, Some("missing")).is_err(),
            "invalid architecture accepted",
        )?;
        check(
            nmse(&[1.0, 2.0], &[1.0, 2.0])? == 0.0,
            "NMSE identity differs",
        )?;
        check(nmse(&[1.0], &[2.0])? > 1e-4, "NMSE mutation not detected")?;
        check(nmse(&[1.0], &[]).is_err(), "NMSE shape accepted")?;
        check(
            nmse(&[0.0], &[0.0])?.is_nan(),
            "source zero norm semantics changed",
        )?;
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "prism-architecture-input-{}-{stamp}",
            std::process::id()
        ));
        std::fs::create_dir(&directory)?;
        let checked: Result<()> = (|| {
            for name in [
                "qwen35",
                "nemotron_h",
                "deepseek4",
                "bailingmoe3",
                "gemma3n",
                "dspark",
            ] {
                let row = array(&f, "Architectures")?
                    .iter()
                    .find(|r| r["Name"] == name)
                    .ok_or_else(|| invalid("focused architecture fixture absent"))?;
                let (path, _) = metadata(&directory, row, row["MandatoryMoe"] == true, true, &f)?;
                let bytes = std::fs::read(&path)?;
                check(
                    bytes.starts_with(b"GGUF\x03\0\0\0") && bytes.len() % 32 == 0,
                    "independent input metadata framing differs",
                )?;
                std::fs::remove_file(path)?;
            }
            Ok(())
        })();
        std::fs::remove_dir(directory)?;
        checked?;
        Ok(())
    }
}
