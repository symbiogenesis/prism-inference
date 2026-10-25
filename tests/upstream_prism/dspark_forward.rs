//! Complete native Windows CPU forward contracts from the pinned `DSpark` source.
//! Inputs, schedules, scalar correction oracle and all verdicts belong to Rust.
use super::engine::Engine;
use super::generated_inputs::{floats, hex};
use super::model_session::{Client, ModelSession};
use super::windows_random::{Normal, Random};
use super::{Result, invalid, result};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
const ID: &str = "tests/test-dspark-forward.cpp";
const SOURCE_SHA: &str = "b47fb55a0911ad7b8ac6a6cce370d8cce8d6c23e45a732b82c0ec5a0f35fe94e";
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
pub(super) enum ForwardMode {
    #[serde(rename = "tier1")]
    Tier1,
    #[serde(rename = "snr-ab-cpu")]
    SnrAb,
    #[serde(rename = "output-ab-cpu")]
    OutputAb,
    #[serde(rename = "runtime-ab-cpu")]
    RuntimeAb,
    #[serde(rename = "prefix-ab-cpu")]
    PrefixAb,
    #[serde(rename = "tier2")]
    Tier2,
}
const MODES: [ForwardMode; 6] = [
    ForwardMode::Tier1,
    ForwardMode::SnrAb,
    ForwardMode::OutputAb,
    ForwardMode::RuntimeAb,
    ForwardMode::PrefixAb,
    ForwardMode::Tier2,
];
#[derive(Default, Deserialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
pub(super) struct ForwardOptions {
    pub(super) mode: Option<ForwardMode>,
    pub(super) model: Option<PathBuf>,
    pub(super) reference: Option<PathBuf>,
    pub(super) shared_head_source: Option<PathBuf>,
    pub(super) no_flash_attention: bool,
    pub(super) trace: bool,
    pub(super) oracle: bool,
    pub(super) logits_output: Option<PathBuf>,
    pub(super) head_top_k: Option<i32>,
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
fn integer(value: &Value, key: &str) -> Result<usize> {
    Ok(usize::try_from(value[key].as_u64().ok_or_else(|| {
        invalid(format!("missing production {key}"))
    })?)?)
}
fn release(client: &Client, kind: &str, id: u64) -> Result<()> {
    invoke(client, json!({"Op":"release","Kind":kind,"Handle":id}))?;
    Ok(())
}
fn positions(begin: i32, count: usize) -> Result<Vec<i32>> {
    (0..count)
        .map(|i| {
            begin
                .checked_add(i32::try_from(i)?)
                .ok_or_else(|| invalid("forward position overflow"))
        })
        .collect()
}
fn bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect()
}
#[allow(
    clippy::cast_possible_truncation,
    clippy::as_conversions,
    reason = "source scalar oracle explicitly rounds F64 dot sums to F32"
)]
const fn round(value: f64) -> f32 {
    value as f32
}
fn f64_size(value: usize) -> Result<f64> {
    Ok(f64::from(u32::try_from(value)?))
}
#[derive(Clone)]
struct Inputs {
    features: Vec<f32>,
    context_positions: Vec<i32>,
    tokens: Vec<i32>,
    draft_positions: Vec<i32>,
    width: usize,
}
impl Inputs {
    fn sinusoidal(width: usize, depth: usize, mask: i32) -> Result<Self> {
        let features = (0..6 * width)
            .map(|i| Ok(round(f64_size(i)?).sin()))
            .collect::<Result<Vec<_>>>()?;
        let mut tokens = vec![mask; depth];
        if let Some(first) = tokens.first_mut() {
            *first = 0;
        }
        Ok(Self {
            features,
            context_positions: positions(0, 6)?,
            tokens,
            draft_positions: positions(6, depth)?,
            width,
        })
    }
}
struct Forward {
    logits: Vec<f32>,
    ids: Vec<i32>,
    exception: Value,
}
struct Harness {
    session: ModelSession,
    client: Client,
    assertions: usize,
    forwards: usize,
}
impl Harness {
    fn open(engine: &Engine, name: &str) -> Result<Self> {
        let session = ModelSession::open(engine, name)?
            .ok_or_else(|| invalid("missing production DSpark forward/graph API"))?;
        let client = session.client();
        invoke(&client, json!({"Op":"backend-load-all"}))?;
        invoke(&client, json!({"Op":"backend-init"}))?;
        Ok(Self {
            session,
            client,
            assertions: 0,
            forwards: 0,
        })
    }
    fn check(&mut self, condition: bool, message: &str) -> Result<()> {
        self.assertions += 1;
        if condition {
            Ok(())
        } else {
            Err(invalid(message))
        }
    }
    #[allow(
        clippy::needless_pass_by_value,
        reason = "wire composition accepts owned caller operations like the session client"
    )]
    fn environment(&self, changes: Value, save: bool) -> Result<Value> {
        invoke(
            &self.client,
            json!({"Op":"process-environment","Changes":changes,"Save":save}),
        )
    }
    fn load(&mut self, path: &Path, head: Option<u64>) -> Result<u64> {
        let mut overrides = json!({"GpuLayers":0});
        if let Some(head) = head {
            overrides["HeadSource"] = json!(head);
        }
        let model = handle(
            &invoke(
                &self.client,
                json!({"Op":"model-load","Path":path,"Base":"default","Params":null,"Overrides":overrides,"Progress":false}),
            )?,
            "Model",
        )?;
        self.assertions += 1;
        Ok(model)
    }
    fn metadata(&mut self, model: u64) -> Result<Value> {
        let meta = invoke(&self.client, json!({"Op":"dspark-meta","Model":model}))?;
        self.check(meta["Found"] == true, "model is not DSpark")?;
        let (embedding, capture, width) = (
            integer(&meta, "Embedding")?,
            integer(&meta, "CaptureCount")?,
            integer(&meta, "CaptureEmbedding")?,
        );
        self.check(
            embedding > 0 && capture > 0 && width == embedding * capture,
            "invalid target-feature projection width",
        )?;
        self.check(
            integer(&meta, "BlockSize")? > 0,
            "DSpark block metadata missing",
        )?;
        Ok(meta)
    }
    fn trace(&self) -> Result<u64> {
        handle(
            &invoke(
                &self.client,
                json!({"Op":"trace-new","Names":["dspark_fc","dspark_hidden_norm","dspark_log_snr_fc1","dspark_log_snr_fc2","dspark_draft_embd_snr","dspark_log_snr_expanded","dspark_balanced_output","result_norm"],"Prefixes":["l_out-","dspark_corr_"]}),
            )?,
            "Trace",
        )
    }
    fn context(&mut self, model: u64, rows: usize, trace: Option<u64>, no_fa: bool) -> Result<u64> {
        let mut overrides = json!({"Context":rows,"Batch":rows,"MicroBatch":rows,"Sequences":1,"OutputsPerSequence":0,"NoPerf":true});
        if let Some(trace) = trace {
            overrides["TraceId"] = json!(trace);
        }
        if no_fa {
            overrides["FlashAttention"] = json!(0);
        }
        let context = handle(
            &invoke(
                &self.client,
                json!({"Op":"context","Model":model,"Params":null,"Overrides":overrides}),
            )?,
            "Context",
        )?;
        self.assertions += 1;
        invoke(
            &self.client,
            json!({"Op":"causal-attention","Context":context,"Value":false}),
        )?;
        Ok(context)
    }
    fn forward(
        &mut self,
        context: u64,
        input: &Inputs,
        output_rows: usize,
        ids: bool,
    ) -> Result<Forward> {
        let context_rows = input.context_positions.len();
        let depth = input.tokens.len();
        self.check(
            input.features.len() == context_rows * input.width,
            "ctx_feat size mismatch",
        )?;
        self.check(
            input.draft_positions.len() == depth,
            "draft position size mismatch",
        )?;
        self.check(
            output_rows > 0 && output_rows <= depth,
            "invalid output row count",
        )?;
        invoke(
            &self.client,
            json!({"Op":"dspark-stage","Context":context,"Rows":context_rows,"Width":input.width,"FeaturesHex":hex(&bytes(&input.features))?,"Positions":input.context_positions}),
        )?;
        let batch = handle(
            &invoke(
                &self.client,
                json!({"Op":"batch-new","Capacity":context_rows+depth,"Sequences":1}),
            )?,
            "Batch",
        )?;
        let mut rows = Vec::new();
        for &position in &input.context_positions {
            rows.push(json!({"Token":0,"Position":position,"Sequences":[0],"Logits":false}));
        }
        for (i, (&token, &position)) in input.tokens.iter().zip(&input.draft_positions).enumerate()
        {
            rows.push(
                json!({"Token":token,"Position":position,"Sequences":[0],"Logits":i<output_rows}),
            );
        }
        invoke(
            &self.client,
            json!({"Op":"batch-add","Batch":batch,"Rows":rows}),
        )?;
        let observation = invoke(
            &self.client,
            json!({"Op":"observe-decode","Context":context,"Batch":batch}),
        );
        release(&self.client, "batch", batch)?;
        invoke(
            &self.client,
            json!({"Op":"dspark-stage-clear","Context":context}),
        )?;
        let observation = observation?;
        self.forwards += 1;
        if !observation["Exception"].is_null() {
            return Ok(Forward {
                logits: Vec::new(),
                ids: Vec::new(),
                exception: observation["Exception"].clone(),
            });
        }
        self.check(observation["Code"] == 0, "llama_decode returned nonzero")?;
        if ids {
            let indices = (0..output_rows)
                .map(|i| Ok(i32::try_from(i)? - i32::try_from(output_rows)?))
                .collect::<Result<Vec<_>>>()?;
            let output = invoke(
                &self.client,
                json!({"Op":"sampled-tokens","Context":context,"Indices":indices}),
            )?;
            let ids = output["Tokens"]
                .as_array()
                .ok_or_else(|| invalid("sampled IDs absent"))?
                .iter()
                .map(|v| {
                    Ok(i32::try_from(
                        v.as_i64()
                            .ok_or_else(|| invalid("sampled ID not integer"))?,
                    )?)
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Forward {
                logits: Vec::new(),
                ids,
                exception: Value::Null,
            })
        } else {
            let output = invoke(
                &self.client,
                json!({"Op":"decode-output-logits","Context":context}),
            )?;
            self.check(output["Present"] == true, "llama_get_logits returned null")?;
            self.check(
                output["Rows"] == output_rows,
                "production output count differs",
            )?;
            Ok(Forward {
                logits: floats(&output["DataHex"])?,
                ids: Vec::new(),
                exception: Value::Null,
            })
        }
    }
    fn successful(&mut self, value: &Forward) -> Result<()> {
        self.check(
            value.exception.is_null(),
            "forward threw an unexpected production exception",
        )
    }
    fn exact(&mut self, reference: &[f32], actual: &[f32], label: &str) -> Result<()> {
        self.check(
            !reference.is_empty() && reference.len() == actual.len(),
            "exact A/B shape mismatch",
        )?;
        for (&reference, &actual) in reference.iter().zip(actual) {
            self.check(
                reference.is_finite() && actual.is_finite(),
                "exact A/B nonfinite value",
            )?;
            self.check(reference.to_bits() == actual.to_bits(), label)?;
        }
        Ok(())
    }
    fn traces(&mut self, trace: u64) -> Result<BTreeMap<String, Vec<f32>>> {
        let output = invoke(&self.client, json!({"Op":"context-traces","Trace":trace}))?;
        self.check(output["Failed"] == false, "production trace capture failed")?;
        let mut values = BTreeMap::new();
        for row in output["Values"]
            .as_array()
            .ok_or_else(|| invalid("trace values absent"))?
        {
            let name = row["Name"]
                .as_str()
                .ok_or_else(|| invalid("trace name absent"))?;
            values.insert(name.into(), floats(&row["DataHex"])?);
        }
        Ok(values)
    }
    fn tensor(&mut self, model: u64, name: &str, index: usize) -> Result<Option<Vec<f32>>> {
        let info = invoke(
            &self.client,
            json!({"Op":"model-tensor-info","Model":model,"Names":[name]}),
        )?;
        let row = info["Tensors"]
            .as_array()
            .and_then(|v| v.first())
            .ok_or_else(|| invalid("production tensor metadata missing"))?;
        if row["Found"] != true {
            return Ok(None);
        }
        let kind = integer(row, "Type")?;
        self.check(
            kind == 0 || kind == 30,
            "correction oracle requires F32 or BF16 tensor",
        )?;
        let filename = format!("oracle-weight-{}-{index}.bin", self.forwards);
        let output = invoke(
            &self.client,
            json!({"Op":"model-tensor-file","Model":model,"Name":name,"Output":filename}),
        )?;
        let row = output["Tensors"]
            .as_array()
            .and_then(|v| v.first())
            .ok_or_else(|| invalid("production tensor file absent"))?;
        let path = PathBuf::from(
            row["DataFile"]
                .as_str()
                .ok_or_else(|| invalid("production tensor path absent"))?,
        );
        if path.parent() != Some(self.session.evidence_directory())
            || !path.is_file()
            || std::fs::metadata(&path)?.len() > 32 * 1024 * 1024
        {
            return Err(invalid("oracle weight observation outside session bounds"));
        }
        let raw = std::fs::read(&path)?;
        let values = if kind == 0 {
            if raw.len() % 4 != 0 {
                return Err(invalid("F32 oracle weight byte mismatch"));
            }
            raw.as_chunks::<4>()
                .0
                .iter()
                .map(|&v| f32::from_le_bytes(v))
                .collect()
        } else {
            if raw.len() % 2 != 0 {
                return Err(invalid("BF16 oracle weight byte mismatch"));
            }
            raw.as_chunks::<2>()
                .0
                .iter()
                .map(|&v| f32::from_bits(u32::from(u16::from_le_bytes(v)) << 16))
                .collect()
        };
        Ok(Some(values))
    }
}
fn argmax(row: &[f32]) -> Result<i32> {
    let mut winner = 0;
    for i in 1..row.len() {
        if row[i] > row[winner] {
            winner = i;
        }
    }
    Ok(i32::try_from(winner)?)
}
fn mv(weight: &[f32], input: &[f32]) -> Result<Vec<f32>> {
    if input.is_empty() || !weight.len().is_multiple_of(input.len()) {
        return Err(invalid("scalar correction matrix shape mismatch"));
    }
    Ok(weight
        .chunks_exact(input.len())
        .map(|w| {
            round(
                w.iter()
                    .zip(input)
                    .map(|(&w, &x)| f64::from(w) * f64::from(x))
                    .sum(),
            )
        })
        .collect())
}
fn norm(input: &[f32], weight: &[f32]) -> Result<Vec<f32>> {
    if input.is_empty() || input.len() != weight.len() {
        return Err(invalid("scalar correction norm shape mismatch"));
    }
    let square = input
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum::<f64>();
    let scale = (round(square / f64_size(input.len())?) + 1e-6_f32)
        .sqrt()
        .recip();
    Ok(input
        .iter()
        .zip(weight)
        .map(|(&v, &w)| v * (scale * w))
        .collect())
}
fn correction_oracle(
    h: &mut Harness,
    model: u64,
    hidden: &[f32],
    ids: &[i32],
    meta: &Value,
) -> Result<Vec<f32>> {
    let mut tensors = BTreeMap::new();
    for (i, name) in [
        "token_embd.weight",
        "dspark.correction_hidden_norm.weight",
        "dspark.correction_embed_norm.weight",
        "dspark.correction_gate.weight",
        "dspark.correction_up.weight",
        "dspark.correction_down.weight",
        "output.weight",
        "dspark.markov_head_a.weight",
        "dspark.markov_head_b.weight",
    ]
    .into_iter()
    .enumerate()
    {
        if let Some(value) = h.tensor(model, name, i)? {
            tensors.insert(name, value);
        }
    }
    let get = |name: &str| -> Result<&[f32]> {
        tensors
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| invalid(format!("correction weight absent {name}")))
    };
    let (embedding, vocabulary, mask) = (
        integer(meta, "Embedding")?,
        integer(meta, "Vocabulary")?,
        integer(meta, "MaskToken")?,
    );
    let (embeddings, a, b) = (
        get("token_embd.weight")?,
        get("dspark.markov_head_a.weight")?,
        get("dspark.markov_head_b.weight")?,
    );
    let rank = a.len() / vocabulary;
    let head = tensors
        .get("output.weight")
        .map_or(embeddings, Vec::as_slice);
    let mut previous = usize::try_from(ids[0])?;
    let mut output = Vec::new();
    for (slot, &id) in ids.iter().enumerate() {
        let mut hidden = hidden
            .get(slot * embedding..(slot + 1) * embedding)
            .ok_or_else(|| invalid("trunk oracle embedding shape mismatch"))?
            .to_vec();
        let mut input = norm(&hidden, get("dspark.correction_hidden_norm.weight")?)?;
        input.extend(norm(
            &embeddings[previous * embedding..(previous + 1) * embedding],
            get("dspark.correction_embed_norm.weight")?,
        )?);
        let mut gate = mv(get("dspark.correction_gate.weight")?, &input)?;
        let up = mv(get("dspark.correction_up.weight")?, &input)?;
        for (gate, up) in gate.iter_mut().zip(up) {
            *gate = *gate / (1.0 + (-*gate).exp()) * up;
        }
        let delta = mv(get("dspark.correction_down.weight")?, &gate)?;
        for (hidden, delta) in hidden.iter_mut().zip(delta) {
            *hidden += delta;
        }
        let mut logits = mv(head, &hidden)?;
        let id = usize::try_from(id)?;
        let bias = mv(b, &a[id * rank..(id + 1) * rank])?;
        for (logit, bias) in logits.iter_mut().zip(bias) {
            *logit += bias;
        }
        logits[mask] = -1e6;
        previous = usize::try_from(argmax(&logits)?)?;
        output.extend(logits);
    }
    Ok(output)
}
struct Tier1 {
    logits: Vec<f32>,
    traces: BTreeMap<String, Vec<f32>>,
}
fn trace_value<'a>(traces: &'a BTreeMap<String, Vec<f32>>, name: &str) -> Result<&'a [f32]> {
    traces
        .get(name)
        .map(Vec::as_slice)
        .ok_or_else(|| invalid(format!("required source trace absent {name}")))
}
#[allow(
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::suboptimal_flops,
    reason = "tier1 optional scalar oracle and trace predicates remain with the source input schedule"
)]
fn tier1(
    h: &mut Harness,
    path: &Path,
    head_path: Option<&Path>,
    options: &ForwardOptions,
    trace_enabled: bool,
    oracle: bool,
) -> Result<Tier1> {
    let head = head_path.map(|path| h.load(path, None)).transpose()?;
    let model = h.load(path, head)?;
    if let Some(head) = head {
        let identity = invoke(
            &h.client,
            json!({"Op":"model-output-identity","Model":model,"Other":head}),
        )?;
        h.check(
            identity["TensorEqual"] == true && identity["BufferEqual"] == true,
            "shared output tensor/buffer identity differs",
        )?;
    }
    let meta = h.metadata(model)?;
    let (width, depth, vocabulary, mask) = (
        integer(&meta, "CaptureEmbedding")?,
        integer(&meta, "BlockSize")?,
        integer(&meta, "Vocabulary")?,
        i32::try_from(integer(&meta, "MaskToken")?)?,
    );
    let trace = if trace_enabled || oracle {
        Some(h.trace()?)
    } else {
        None
    };
    let context = h.context(model, 6 + depth, trace, options.no_flash_attention)?;
    let mut random = Random::new(1234);
    let mut normal = Normal::new(0.0, 2.0);
    let features = (0..6 * width).map(|_| normal.sample(&mut random)).collect();
    let mut tokens = vec![mask; depth];
    tokens[0] = i32::try_from(1000.min(vocabulary - 1))?;
    if tokens[0] == mask {
        tokens[0] = 0;
    }
    let input = Inputs {
        features,
        context_positions: positions(0, 6)?,
        tokens,
        draft_positions: positions(6, depth)?,
        width,
    };
    let actual = h.forward(context, &input, depth, false)?;
    h.successful(&actual)?;
    let traces = trace
        .map(|id| h.traces(id))
        .transpose()?
        .unwrap_or_default();
    if meta["GraphCorrected"] == true && trace_enabled {
        for slot in 0..depth {
            let before = trace_value(&traces, &format!("dspark_corr_unmasked-{slot}"))?;
            let after = trace_value(&traces, &format!("dspark_corr_masked-{slot}"))?;
            h.check(
                before.len() == vocabulary && after.len() == before.len(),
                "mask suppression trace shape differs",
            )?;
            for (token, &after) in after.iter().enumerate() {
                let expected = if token == usize::try_from(mask)? {
                    -1e6
                } else {
                    before[token]
                };
                h.check(after == expected, "mask suppression changed non-mask logit")?;
            }
        }
    }
    if meta["GraphCorrected"] == true && oracle {
        let expected = correction_oracle(
            h,
            model,
            trace_value(&traces, "result_norm")?,
            &input.tokens,
            &meta,
        )?;
        h.check(
            expected.len() == actual.logits.len(),
            "correction oracle shape differs",
        )?;
        for (&actual, &expected) in actual.logits.iter().zip(&expected) {
            h.check(
                (actual - expected)
                    .abs()
                    .partial_cmp(&(1e-3 + 1e-2 * expected.abs()))
                    != Some(std::cmp::Ordering::Greater),
                "hidden correction differs from scalar CPU oracle",
            )?;
        }
    }
    h.check(
        actual.logits.iter().all(|v| v.is_finite()),
        "Tier1 nonfinite logits",
    )?;
    h.check(
        actual.logits.len() == depth * vocabulary,
        "Tier1 logits shape differs",
    )?;
    if let Some(path) = &options.logits_output {
        if !path.is_absolute() {
            return Err(invalid("logits output must be an absolute external path"));
        }
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        std::io::Write::write_all(&mut file, &bytes(&actual.logits))?;
    }
    release(&h.client, "context", context)?;
    release(&h.client, "model", model)?;
    if let Some(head) = head {
        release(&h.client, "model", head)?;
    }
    Ok(Tier1 {
        logits: actual.logits,
        traces,
    })
}
fn ab(h: &mut Harness, path: &Path, options: &ForwardOptions, snr: bool) -> Result<Value> {
    let key = if snr {
        "LLAMA_DSPARK_SNR_TWO_ROWS"
    } else {
        "LLAMA_DSPARK_BALANCED_OUTPUT"
    };
    let mut changes = json!({"DSPARK_TEST_TRACE":true});
    changes[key] = json!(false);
    h.environment(changes, false)?;
    let baseline = tier1(
        h,
        path,
        options.shared_head_source.as_deref(),
        options,
        true,
        options.oracle,
    )?;
    if snr {
        h.check(
            baseline.traces.contains_key("dspark_draft_embd_snr")
                && !baseline.traces.contains_key("dspark_log_snr_expanded"),
            "baseline did not run unfused conditioning",
        )?;
    } else {
        h.check(
            !baseline.traces.contains_key("dspark_balanced_output"),
            "baseline used balanced output",
        )?;
    }
    let mut changes = json!({});
    changes[key] = json!(true);
    h.environment(changes, false)?;
    let candidate = tier1(
        h,
        path,
        options.shared_head_source.as_deref(),
        options,
        true,
        options.oracle,
    )?;
    if snr {
        h.check(
            candidate.traces.contains_key("dspark_log_snr_expanded")
                && candidate.traces.contains_key("dspark_draft_embd_snr")
                && candidate.traces.contains_key("dspark_log_snr_fc2"),
            "candidate did not exercise two-row conditioning",
        )?;
        h.check(
            trace_value(&candidate.traces, "dspark_log_snr_fc2")?.len()
                < trace_value(&baseline.traces, "dspark_log_snr_fc2")?.len(),
            "candidate conditioning rows were not reduced",
        )?;
        h.exact(
            trace_value(&baseline.traces, "dspark_draft_embd_snr")?,
            trace_value(&candidate.traces, "dspark_draft_embd_snr")?,
            "conditioned embeddings differ",
        )?;
        h.exact(
            trace_value(&baseline.traces, "dspark_log_snr_fc2")?,
            trace_value(&candidate.traces, "dspark_log_snr_expanded")?,
            "expanded conditioning differs",
        )?;
    } else {
        h.check(
            candidate.traces.contains_key("dspark_balanced_output"),
            "candidate balanced-output branch not exercised",
        )?;
    }
    h.exact(
        &baseline.logits,
        &candidate.logits,
        "full A/B logits differ",
    )?;
    Ok(
        json!({"Mode":if snr{"snr-ab-cpu"}else{"output-ab-cpu"},"Elements":baseline.logits.len(),"Exact":true}),
    )
}
fn clear(client: &Client, context: u64) -> Result<()> {
    invoke(
        client,
        json!({"Op":"memory-clear","Context":context,"Data":true}),
    )?;
    Ok(())
}
fn greedy(h: &mut Harness, context: u64) -> Result<u64> {
    let sampler = handle(
        &invoke(&h.client, json!({"Op":"greedy-sampler-new"}))?,
        "Sampler",
    )?;
    h.check(
        invoke(
            &h.client,
            json!({"Op":"context-set-sampler","Context":context,"Sequence":0,"Sampler":sampler}),
        )?["Set"]
            == true,
        "sampler offload failed",
    )?;
    Ok(sampler)
}
fn free_greedy(client: &Client, context: u64, sampler: u64) -> Result<()> {
    invoke(
        client,
        json!({"Op":"context-set-sampler","Context":context,"Sequence":0,"Sampler":null}),
    )?;
    invoke(
        client,
        json!({"Op":"common-sampler-free","Sampler":sampler}),
    )?;
    Ok(())
}
fn spec(h: &mut Harness, context: u64, maximum: usize, sequences: usize) -> Result<u64> {
    let value = handle(
        &invoke(
            &h.client,
            json!({"Op":"speculative-init","DraftContext":context,"TargetContext":context,"Max":maximum,"Min":0,"Sequences":sequences}),
        )?,
        "Speculative",
    )?;
    h.assertions += 1;
    Ok(value)
}
fn stage_spec(h: &mut Harness, id: u64, input: &Inputs) -> Result<()> {
    h.check(invoke(&h.client,json!({"Op":"speculative-stage","Speculative":id,"Sequence":0,"Rows":input.context_positions.len(),"Width":input.width,"FeaturesHex":hex(&bytes(&input.features))?,"Positions":input.context_positions}))?["Staged"]==true,"common speculative staging failed")
}
fn draft_operation(id: u64, maximum: i32) -> Value {
    json!({"Op":"speculative-draft","Speculative":id,"Sequence":0,"Drafting":true,"Max":maximum,"Past":6,"Last":0,"Prompt":null})
}
fn draft(h: &mut Harness, id: u64, input: &Inputs, maximum: i32) -> Result<Vec<i32>> {
    invoke(
        &h.client,
        json!({"Op":"speculative-begin","Speculative":id,"Sequence":0,"Prompt":[]}),
    )?;
    stage_spec(h, id, input)?;
    let output = invoke(&h.client, draft_operation(id, maximum))?;
    h.forwards += 1;
    output["Tokens"]
        .as_array()
        .ok_or_else(|| invalid("common draft tokens absent"))?
        .iter()
        .map(|v| {
            Ok(i32::try_from(v.as_i64().ok_or_else(|| {
                invalid("common draft token not integer")
            })?)?)
        })
        .collect()
}
fn free_spec(client: &Client, id: u64) -> Result<()> {
    invoke(client, json!({"Op":"speculative-release","Speculative":id}))?;
    Ok(())
}
fn best(logits: &[f32], vocabulary: usize) -> Result<Vec<i32>> {
    if vocabulary == 0 || !logits.len().is_multiple_of(vocabulary) {
        return Err(invalid("argmax logit shape differs"));
    }
    logits.chunks_exact(vocabulary).map(argmax).collect()
}
#[allow(
    clippy::too_many_lines,
    reason = "preserve the source top-k/depth/variant/repetition nesting and negative-mask schedule"
)]
fn runtime_ab(h: &mut Harness, path: &Path, options: &ForwardOptions) -> Result<Value> {
    let model = h.load(path, None)?;
    let meta = h.metadata(model)?;
    h.check(
        meta["GraphCorrected"] == true,
        "runtime A/B requires graph-corrected drafter",
    )?;
    let (width, block, vocabulary, mask) = (
        integer(&meta, "CaptureEmbedding")?,
        integer(&meta, "BlockSize")?,
        integer(&meta, "Vocabulary")?,
        i32::try_from(integer(&meta, "MaskToken")?)?,
    );
    let mut checks = 0;
    let mut negative_checks = 0;
    let mut rows = Vec::new();
    for top_k in [0, 32] {
        if top_k > vocabulary {
            rows.push(json!({"SkippedTopK":top_k,"Reason":"original top-k exceeds vocabulary"}));
            continue;
        }
        h.environment(json!({"DSPARK_HEAD_TOP_K":top_k}), false)?;
        for depth in [1, 2, 4, 8] {
            if depth > block {
                rows.push(
                    json!({"SkippedDepth":depth,"Reason":"original depth exceeds block size"}),
                );
                continue;
            }
            let mut expected = Vec::new();
            for variant in [0, 4, 1, 2, 3] {
                if variant == 3 && depth != block {
                    continue;
                }
                h.environment(
                    json!({"LLAMA_DSPARK_REUSE_MASK":variant!=0&&variant!=4}),
                    false,
                )?;
                let context = h.context(model, 6 + depth, None, options.no_flash_attention)?;
                let ids_only = variant == 2 || variant == 4;
                // The source constructs and frees the chain even when unbound.
                let sampler = if ids_only {
                    greedy(h, context)?
                } else {
                    handle(
                        &invoke(&h.client, json!({"Op":"greedy-sampler-new"}))?,
                        "Sampler",
                    )?
                };
                let runner = if variant == 3 {
                    h.environment(json!({"LLAMA_DSPARK_GREEDY_IDS":true,"DSPARK_FORWARD_ROWS":null,"DSPARK_CORRECTION_ROWS":null}),false)?;
                    Some(spec(h, context, depth, 1)?)
                } else {
                    None
                };
                for _rep in 0..3 {
                    clear(&h.client, context)?;
                    let input = Inputs::sinusoidal(width, depth, mask)?;
                    let (logits, ids) = if let Some(spec) = runner {
                        let ids = draft(h, spec, &input, -1)?;
                        h.check(ids.len() == depth, "common draft length differs")?;
                        (Vec::new(), ids)
                    } else {
                        let value = h.forward(context, &input, depth, ids_only)?;
                        h.successful(&value)?;
                        let ids = if ids_only {
                            value.ids
                        } else {
                            best(&value.logits, vocabulary)?
                        };
                        (value.logits, ids)
                    };
                    if variant == 0 {
                        expected.clone_from(&logits);
                    }
                    if variant == 1 {
                        h.exact(&expected, &logits, "mask reuse changed logits")?;
                    }
                    if variant >= 2 {
                        let expected_ids = best(&expected, vocabulary)?;
                        h.check(ids.len() >= depth, "graph greedy IDs are truncated")?;
                        for (&actual, &expected) in ids.iter().zip(&expected_ids).take(depth) {
                            h.check(actual == expected, "graph greedy ID differs")?;
                        }
                    }
                    checks += 1;
                }
                if variant == 1 && depth > 1 {
                    clear(&h.client, context)?;
                    let mut input = Inputs::sinusoidal(width, depth, mask)?;
                    input.features.fill(0.0);
                    input.tokens[0] = 0;
                    input.tokens[1] = 0;
                    let rejected = h.forward(context, &input, depth, false)?;
                    h.check(
                        rejected.exception["Kind"] == "runtime-error"
                            && rejected.exception["Message"]
                                .as_str()
                                .is_some_and(|s| s.contains("anchor followed by MASK")),
                        "mask reuse accepted non-MASK input",
                    )?;
                    negative_checks += 1;
                }
                if let Some(spec) = runner {
                    free_spec(&h.client, spec)?;
                }
                free_greedy(&h.client, context, sampler)?;
                release(&h.client, "context", context)?;
            }
            rows.push(json!({"TopK":top_k,"Depth":depth}));
        }
    }
    release(&h.client, "model", model)?;
    Ok(
        json!({"Mode":"runtime-ab-cpu","Forwards":checks,"NegativeMaskCases":negative_checks,"Rows":rows,"Repetitions":3,"VariantOrder":[0,4,1,2,3]}),
    )
}
#[allow(
    clippy::needless_pass_by_value,
    reason = "wire composition accepts owned caller operations like the session client"
)]
fn observed(h: &mut Harness, operation: Value, kind: &str, label: &str) -> Result<()> {
    let value = invoke(
        &h.client,
        json!({"Op":"observe-production","Operation":operation}),
    )?;
    h.check(value["Exception"]["Kind"] == kind, label)
}
#[allow(
    clippy::too_many_lines,
    reason = "full prefix shape/capture, positive replay and eight negative contracts preserve source order"
)]
fn prefix_ab(h: &mut Harness, path: &Path, options: &ForwardOptions) -> Result<Value> {
    let previous = json!({"HasContext":true,"Rows":6,"Width":32});
    for (next, live_rows, expected) in [
        (previous.clone(), 6, true),
        (json!({"HasContext":true,"Rows":7,"Width":32}), 7, false),
        (json!({"HasContext":false,"Rows":6,"Width":32}), 7, false),
        (json!({"HasContext":true,"Rows":6,"Width":64}), 7, false),
    ] {
        let value = invoke(
            &h.client,
            json!({"Op":"graph-allow-reuse","Previous":previous,"Next":next,"LiveRows":live_rows,"LiveWidth":32}),
        )?;
        h.check(
            value["AllowReuse"] == expected,
            "graph reuse shape contract differs",
        )?;
    }
    let saved=handle(&h.environment(json!({"LLAMA_DSPARK_CORRECTION_PREFIX":null,"DSPARK_FORWARD_ROWS":null,"DSPARK_CORRECTION_ROWS":null,"LLAMA_DSPARK_GREEDY_IDS":false,"LLAMA_DSPARK_REUSE_MASK":false,"DSPARK_HEAD_TOP_K":null}),true)?,"Environment")?;
    let model = h.load(path, None)?;
    let meta = h.metadata(model)?;
    let capture_context = h.context(model, 16, None, options.no_flash_attention)?;
    invoke(
        &h.client,
        json!({"Op":"capture-layers","Context":capture_context,"Layers":[0]}),
    )?;
    for invalid in [
        json!([0, -1]),
        json!([0, integer(&meta, "LayerCount")?]),
        json!(vec![0; integer(&meta, "MaxCaptureLayers")? + 1]),
    ] {
        observed(
            h,
            json!({"Op":"capture-layers","Context":capture_context,"Layers":invalid}),
            "invalid-argument",
            "invalid capture list accepted",
        )?;
        h.check(
            invoke(
                &h.client,
                json!({"Op":"dspark-context-info","Context":capture_context}),
            )?["CaptureCount"]
                == 1,
            "invalid capture registration changed previous registration",
        )?;
    }
    release(&h.client, "context", capture_context)?;
    h.check(
        meta["GraphCorrected"] == true,
        "prefix A/B requires graph-corrected drafter",
    )?;
    let (width, block, vocabulary, mask) = (
        integer(&meta, "CaptureEmbedding")?,
        integer(&meta, "BlockSize")?,
        integer(&meta, "Vocabulary")?,
        i32::try_from(integer(&meta, "MaskToken")?)?,
    );
    let input = Inputs::sinusoidal(width, block, mask)?;
    let restricted = options
        .head_top_k
        .unwrap_or(i32::try_from(32.min(vocabulary))?);
    let mut checks = 0;
    for top_k in [0, restricted] {
        h.check(
            top_k >= 0 && usize::try_from(top_k)? <= vocabulary,
            "invalid test head top-k",
        )?;
        h.environment(
            json!({"DSPARK_HEAD_TOP_K":top_k,"LLAMA_DSPARK_CORRECTION_PREFIX":null}),
            false,
        )?;
        let baseline = h.context(model, 6 + block, None, options.no_flash_attention)?;
        let mut reference = Vec::new();
        for rep in 0..3 {
            clear(&h.client, baseline)?;
            let value = h.forward(baseline, &input, block, false)?;
            h.successful(&value)?;
            if rep == 0 {
                reference = value.logits;
            } else {
                h.exact(&reference, &value.logits, "baseline replay changed logits")?;
            }
        }
        release(&h.client, "context", baseline)?;
        for keep in [1, 2, 3, 4, 6, 8] {
            if keep > block {
                continue;
            }
            h.environment(json!({"LLAMA_DSPARK_CORRECTION_PREFIX":keep}), false)?;
            let expected = reference
                .get(..keep * vocabulary)
                .ok_or_else(|| invalid("prefix expected extent absent"))?;
            let expected_ids = best(expected, vocabulary)?;
            for variant in [0, 1, 2] {
                h.environment(json!({"LLAMA_DSPARK_GREEDY_IDS":variant==2}), false)?;
                let context = h.context(model, 6 + block, None, options.no_flash_attention)?;
                let sampler = if variant == 1 {
                    Some(greedy(h, context)?)
                } else {
                    None
                };
                let runner = if variant == 2 {
                    Some(spec(h, context, keep, 1)?)
                } else {
                    None
                };
                for _rep in 0..3 {
                    clear(&h.client, context)?;
                    let ids = if let Some(spec) = runner {
                        draft(h, spec, &input, i32::try_from(keep)?)?
                    } else {
                        let value = h.forward(context, &input, keep, variant == 1)?;
                        h.successful(&value)?;
                        if variant == 0 {
                            h.exact(expected, &value.logits, "prefix changed retained logits")?;
                        }
                        value.ids
                    };
                    if variant != 0 {
                        h.check(ids == expected_ids, "prefix changed retained IDs or length")?;
                    }
                    checks += 1;
                }
                if let Some(spec) = runner {
                    free_spec(&h.client, spec)?;
                }
                if let Some(sampler) = sampler {
                    free_greedy(&h.client, context, sampler)?;
                }
                release(&h.client, "context", context)?;
            }
        }
    }
    let mut negatives = 0;
    h.environment(
        json!({"DSPARK_HEAD_TOP_K":0,"LLAMA_DSPARK_GREEDY_IDS":false}),
        false,
    )?;
    for invalid in [json!("0"), json!((block + 1).to_string()), json!("nope")] {
        h.environment(json!({"LLAMA_DSPARK_CORRECTION_PREFIX":null}), false)?;
        let context = h.context(model, 6 + block, None, options.no_flash_attention)?;
        h.environment(json!({"LLAMA_DSPARK_CORRECTION_PREFIX":invalid}), false)?;
        let actual = h.forward(context, &input, block, false)?;
        h.check(
            actual.exception["Kind"] == "runtime-error",
            "invalid correction prefix accepted",
        )?;
        negatives += 1;
        release(&h.client, "context", context)?;
    }
    let keep = 4.min(block);
    for (maximum, sequences) in [(keep + 1, 1), (keep, 2)] {
        h.environment(json!({"LLAMA_DSPARK_CORRECTION_PREFIX":null}), false)?;
        let context = h.context(model, 6 + block, None, options.no_flash_attention)?;
        h.environment(json!({"LLAMA_DSPARK_CORRECTION_PREFIX":keep}), false)?;
        observed(
            h,
            json!({"Op":"speculative-init","DraftContext":context,"TargetContext":context,"Max":maximum,"Min":0,"Sequences":sequences}),
            "runtime-error",
            "invalid prefix runner contract accepted",
        )?;
        negatives += 1;
        release(&h.client, "context", context)?;
    }
    for changed in [true, false] {
        h.environment(json!({"LLAMA_DSPARK_CORRECTION_PREFIX":null}), false)?;
        let context = h.context(model, 6 + block, None, options.no_flash_attention)?;
        h.environment(json!({"LLAMA_DSPARK_CORRECTION_PREFIX":keep}), false)?;
        let runner = spec(h, context, keep, 1)?;
        invoke(
            &h.client,
            json!({"Op":"speculative-begin","Speculative":runner,"Sequence":0,"Prompt":[]}),
        )?;
        stage_spec(h, runner, &input)?;
        h.environment(json!({"LLAMA_DSPARK_CORRECTION_PREFIX":if changed{json!(if keep==1{2}else{1})}else{Value::Null}}),false)?;
        observed(
            h,
            draft_operation(runner, i32::try_from(keep)?),
            "runtime-error",
            "changed prefix policy accepted after init",
        )?;
        negatives += 1;
        free_spec(&h.client, runner)?;
        release(&h.client, "context", context)?;
    }
    if keep < block {
        h.environment(json!({"LLAMA_DSPARK_CORRECTION_PREFIX":null}), false)?;
        let context = h.context(model, 6 + block, None, options.no_flash_attention)?;
        h.environment(json!({"LLAMA_DSPARK_CORRECTION_PREFIX":keep}), false)?;
        let value = h.forward(context, &input, block, false)?;
        h.check(
            value.exception["Kind"] == "runtime-error",
            "full-output mask accepted with prefix",
        )?;
        negatives += 1;
        release(&h.client, "context", context)?;
    }
    release(&h.client, "model", model)?;
    invoke(
        &h.client,
        json!({"Op":"process-environment-restore","Environment":saved}),
    )?;
    Ok(
        json!({"Mode":"prefix-ab-cpu","Forwards":checks,"NegativeCases":negatives,"GraphReuseChecks":4,"InvalidCaptureLists":3,"Repetitions":3,"KeepOrder":[1,2,3,4,6,8]}),
    )
}
struct Tier2Input {
    input: Inputs,
    vocabulary: usize,
    reference: Vec<f32>,
}
fn word(cursor: &mut std::io::Cursor<&[u8]>) -> Result<u32> {
    let mut bytes = [0; 4];
    std::io::Read::read_exact(cursor, &mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}
fn read_tier2(path: &Path) -> Result<Tier2Input> {
    if !path.is_absolute() || !path.is_file() || std::fs::metadata(path)?.len() > 16 * 1024 * 1024 {
        return Err(invalid(
            "tier2 reference must be bounded absolute regular file",
        ));
    }
    let raw = std::fs::read(path)?;
    let mut cursor = std::io::Cursor::new(raw.as_slice());
    let (rows, width, block, vocabulary) = (
        usize::try_from(word(&mut cursor)?)?,
        usize::try_from(word(&mut cursor)?)?,
        usize::try_from(word(&mut cursor)?)?,
        usize::try_from(word(&mut cursor)?)?,
    );
    if !(1..=2048).contains(&rows)
        || !(1..=1_048_576).contains(&width)
        || rows.checked_mul(width).is_none_or(|v| v > 1024 * 1024)
        || !(1..=64).contains(&block)
        || !(5..=1_048_576).contains(&vocabulary)
        || block
            .checked_mul(vocabulary)
            .is_none_or(|v| v > 2 * 1024 * 1024)
    {
        return Err(invalid(
            "tier2 reference dimensions outside bounded source domain",
        ));
    }
    let context_positions = (0..rows)
        .map(|_| Ok(i32::from_le_bytes(word(&mut cursor)?.to_le_bytes())))
        .collect::<Result<Vec<_>>>()?;
    let features = (0..rows * width)
        .map(|_| Ok(f32::from_bits(word(&mut cursor)?)))
        .collect::<Result<Vec<_>>>()?;
    let tokens = (0..block)
        .map(|_| Ok(i32::from_le_bytes(word(&mut cursor)?.to_le_bytes())))
        .collect::<Result<Vec<_>>>()?;
    let draft_positions = (0..block)
        .map(|_| Ok(i32::from_le_bytes(word(&mut cursor)?.to_le_bytes())))
        .collect::<Result<Vec<_>>>()?;
    let reference = (0..block * vocabulary)
        .map(|_| Ok(f32::from_bits(word(&mut cursor)?)))
        .collect::<Result<Vec<_>>>()?;
    Ok(Tier2Input {
        input: Inputs {
            features,
            context_positions,
            tokens,
            draft_positions,
            width,
        },
        vocabulary,
        reference,
    })
}
fn generated_tier2(directory: &Path) -> Result<PathBuf> {
    let mut random = Random::new(1234);
    let mut normal = Normal::new(0.0, 2.0);
    let features = (0..6 * 128)
        .map(|_| normal.sample(&mut random))
        .collect::<Vec<_>>();
    let context_positions = positions(0, 6)?;
    let draft_positions = positions(6, 7)?;
    let mut tokens = vec![255; 7];
    tokens[0] = 0;
    let mut reference = super::dspark_reference::Reference::new()?;
    let logits = reference.forward(
        &features,
        &context_positions,
        &tokens,
        &draft_positions,
        None,
    )?;
    let path = directory.join("independent-forward-reference.bin");
    let mut output = Vec::new();
    for v in [6_i32, 128, 7, 256] {
        output.extend_from_slice(&v.to_le_bytes());
    }
    for value in &context_positions {
        output.extend_from_slice(&value.to_le_bytes());
    }
    output.extend(bytes(&features));
    for value in tokens.iter().chain(&draft_positions) {
        output.extend_from_slice(&value.to_le_bytes());
    }
    output.extend(bytes(&logits));
    std::fs::write(&path, output)?;
    std::fs::write(
        directory.join("independent-forward-reference.json"),
        serde_json::to_vec_pretty(
            &json!({"Independent":true,"Implementation":"safe scalar Rust; published DeepSpec equations","Revision":"005e03b81cec38b7da6399833d609ee89a2587f2","ProductionObservations":false,"NumericalDifferencesAreDiagnosticOnly":true,"PublishedSources":super::dspark_reference::Reference::provenance()?}),
        )?,
    )?;
    Ok(path)
}
fn top5(row: &[f32]) -> Vec<usize> {
    let mut indices = (0..row.len()).collect::<Vec<_>>();
    indices.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
    indices.truncate(5);
    indices
}
fn tier2(
    h: &mut Harness,
    path: &Path,
    reference: &Path,
    options: &ForwardOptions,
) -> Result<Value> {
    let data = read_tier2(reference)?;
    let model = h.load(path, None)?;
    let meta = h.metadata(model)?;
    h.check(
        integer(&meta, "Vocabulary")? == data.vocabulary,
        "tier2 model/reference vocabulary mismatch",
    )?;
    let context = h.context(
        model,
        data.input.context_positions.len() + data.input.tokens.len(),
        None,
        options.no_flash_attention,
    )?;
    let actual = h.forward(context, &data.input, data.input.tokens.len(), false)?;
    h.successful(&actual)?;
    release(&h.client, "context", context)?;
    release(&h.client, "model", model)?;
    if actual.logits.len() != data.reference.len() {
        return Err(invalid("tier2 production logit extent mismatch"));
    }
    let (mut sum, mut maximum, mut nonfinite, mut matches, mut overlap) =
        (0.0_f64, 0.0_f64, 0_usize, 0_usize, 0_usize);
    for (actual, reference) in actual
        .logits
        .chunks_exact(data.vocabulary)
        .zip(data.reference.chunks_exact(data.vocabulary))
    {
        for (&a, &b) in actual.iter().zip(reference).skip(1) {
            let difference = (f64::from(a) - f64::from(b)).abs();
            sum += difference;
            maximum = maximum.max(difference);
            nonfinite += usize::from(!a.is_finite());
        }
        matches += usize::from(argmax(actual)? == argmax(reference)?);
        let a = top5(actual);
        let b = top5(reference);
        overlap += a.iter().filter(|id| b.contains(id)).count();
    }
    h.check(nonfinite == 0, "tier2 nonfinite logits")?;
    Ok(
        json!({"Mode":"tier2","Rows":data.input.tokens.len(),"MeanAbsoluteDifference":sum/f64_size(data.reference.len())?,"MaximumAbsoluteDifference":maximum,"ArgmaxMatchRate":f64_size(matches)?/f64_size(data.input.tokens.len())?,"MeanTop5Overlap":f64_size(overlap)?/(5.0*f64_size(data.input.tokens.len())?),"Nonfinite":nonfinite,"OriginalColumnZeroOmissionPreserved":true,"NumericalMetricsAreDiagnosticOnly":true}),
    )
}
const fn phase_name(mode: ForwardMode) -> &'static str {
    match mode {
        ForwardMode::Tier1 => "tier1",
        ForwardMode::SnrAb => "snr-ab-cpu",
        ForwardMode::OutputAb => "output-ab-cpu",
        ForwardMode::RuntimeAb => "runtime-ab-cpu",
        ForwardMode::PrefixAb => "prefix-ab-cpu",
        ForwardMode::Tier2 => "tier2",
    }
}
#[allow(
    clippy::too_many_lines,
    clippy::unnecessary_wraps,
    reason = "bounded named source phases preserve all CPU modes and report failures through the shared case result API"
)]
pub(super) fn run_with_options(engine: &Engine, options: &ForwardOptions) -> Result<Value> {
    let execute: Result<Value> = (|| {
        let mut receipts = Vec::new();
        for mode in MODES {
            if options.mode.is_some_and(|selected| selected != mode) {
                continue;
            }
            let (path, default_head, pair_receipt, reference) = if let Some(path) = &options.model {
                if !path.is_absolute() || !path.is_file() {
                    return Err(invalid("forward model must be an absolute regular file"));
                }
                (
                    path.clone(),
                    None,
                    json!({"ExternalModel":path}),
                    options.reference.clone(),
                )
            } else {
                let pair = super::dspark_pair::create(
                    engine,
                    if mode == ForwardMode::Tier2 {
                        "reference"
                    } else {
                        "corrected"
                    },
                )?;
                let reference = if mode == ForwardMode::Tier2 {
                    Some(generated_tier2(
                        pair.drafter
                            .parent()
                            .ok_or_else(|| invalid("pair input directory absent"))?,
                    )?)
                } else {
                    None
                };
                (pair.drafter, Some(pair.target), pair.receipt, reference)
            };
            let mut harness =
                Harness::open(engine, &format!("dspark-forward-{}", phase_name(mode)))?;
            let outcome: Result<Value> = (|| match mode {
                ForwardMode::Tier1 => {
                    let full = options.mode.is_none();
                    let head = options.shared_head_source.as_deref().or(if full {
                        default_head.as_deref()
                    } else {
                        None
                    });
                    let actual = tier1(
                        &mut harness,
                        &path,
                        head,
                        options,
                        full || options.trace,
                        full || options.oracle,
                    )?;
                    Ok(
                        json!({"Mode":"tier1","Elements":actual.logits.len(),"Traces":actual.traces.keys().collect::<Vec<_>>(),"SharedHead":head.is_some(),"ScalarOracle":full||options.oracle}),
                    )
                }
                ForwardMode::SnrAb => ab(&mut harness, &path, options, true),
                ForwardMode::OutputAb => ab(&mut harness, &path, options, false),
                ForwardMode::RuntimeAb => runtime_ab(&mut harness, &path, options),
                ForwardMode::PrefixAb => prefix_ab(&mut harness, &path, options),
                ForwardMode::Tier2 => tier2(
                    &mut harness,
                    &path,
                    reference
                        .as_deref()
                        .ok_or_else(|| invalid("tier2 requires an independent reference file"))?,
                    options,
                ),
            })();
            let assertions = harness.assertions;
            let forwards = harness.forwards;
            let backend_finish = if outcome.is_ok() {
                invoke(&harness.client, json!({"Op":"backend-free"})).map(|_| ())
            } else {
                Ok(())
            };
            let finish = harness.session.finish();
            let value = outcome?;
            backend_finish?;
            finish?;
            receipts.push(json!({"Mode":phase_name(mode),"SourcePredicates":assertions,"DecodeCalls":forwards,"Pair":pair_receipt,"Contract":value}));
        }
        Ok(
            json!({"SourceSHA256":SOURCE_SHA,"Modes":receipts,"PlatformExclusions":["Metal modes require Apple Metal and are outside native Windows ARM64 scope","CUDA modes and CUDA/Metal benchmarks require GPU backends absent on Snapdragon X Elite"],"BenchmarksExecuted":false}),
        )
    })();
    match execute {
        Ok(value) => Ok(result(ID, true, value.to_string())),
        Err(error) => Ok(result(
            ID,
            false,
            format!("complete Windows CPU DSpark forward source contract: {error}"),
        )),
    }
}
pub(super) fn run(engine: &Engine) -> Result<Value> {
    run_with_options(engine, &ForwardOptions::default())
}
pub(super) fn diagnostic(engine: &Engine, value: &Value) -> Result<Value> {
    super::llama_grammar::exact_fields(value, &["Kind", "Options"])?;
    run_with_options(engine, &serde_json::from_value(value["Options"].clone())?)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn focused_forward_input_and_oracle_shape_guard() -> Result<()> {
        let input = Inputs::sinusoidal(128, 8, 255)?;
        if input.features.len() != 768
            || input.tokens != [0, 255, 255, 255, 255, 255, 255, 255]
            || mv(&[1.0], &[]).is_ok()
            || norm(&[1.0], &[]).is_ok()
        {
            return Err(invalid("forward input/oracle shape guard differs"));
        }
        if best(&[1.0, 3.0, 2.0, 4.0], 2)? != [1, 1] || best(&[1.0], 2).is_ok() {
            return Err(invalid("forward row comparison guard differs"));
        }
        let mut cursor = std::io::Cursor::new([0_u8; 3].as_slice());
        if word(&mut cursor).is_ok() {
            return Err(invalid("truncated reference header accepted"));
        }
        Ok(())
    }
}
