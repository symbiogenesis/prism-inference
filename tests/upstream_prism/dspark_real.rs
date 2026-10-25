//! The real-eval source schedule, parity refusals, teacher forcing and rollups.
//! All model operations are production primitives; inputs and decisions are Rust.
use super::dspark_real_inputs::{self as inputs, Options};
use super::engine::Engine;
use super::model_session::{Client, ModelSession};
use super::{Result, invalid, result};
use serde_json::{Value, json};
use std::time::Instant;

const ID: &str = "tests/test-dspark-real-eval.cpp";
fn require(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(invalid(message))
    }
}
fn unsigned(value: &Value, name: &str) -> Result<usize> {
    usize::try_from(
        value[name]
            .as_u64()
            .ok_or_else(|| invalid(format!("missing unsigned {name}")))?,
    )
    .map_err(Into::into)
}
fn tokens(value: &Value, field: &str) -> Result<Vec<i32>> {
    value[field]
        .as_array()
        .ok_or_else(|| invalid(format!("missing {field}")))?
        .iter()
        .map(|v| {
            i32::try_from(v.as_i64().ok_or_else(|| invalid("invalid token"))?).map_err(Into::into)
        })
        .collect()
}
struct Runtime {
    client: Client,
    batch: usize,
    batch_size: usize,
    target: usize,
    drafter: usize,
    ar: usize,
    spec: usize,
}
impl Runtime {
    fn call(&self, op: Value) -> Result<Value> {
        Ok(self.client.call(0, op, None)?.value)
    }
    fn sampler(&self, model: usize) -> Result<usize> {
        let value = self
            .call(json!({"Op":"common-sampler-new","Model":model,"Seed":42,"TemperatureBits":0}))?;
        let id = unsigned(&value, "Sampler")?;
        require(id > 0, "common_sampler_init returned null")?;
        Ok(id)
    }
    fn free_sampler(&self, id: usize) -> Result<()> {
        self.call(json!({"Op":"common-sampler-free","Sampler":id}))?;
        Ok(())
    }
    fn reset(&self, context: usize, required: bool) -> Result<()> {
        let value = self
            .call(json!({"Op":"memory-remove","Context":context,"Sequence":0,"From":0,"To":-1}))?;
        if required {
            require(value["Removed"] == true, "oracle reset failed")?;
        }
        Ok(())
    }
    fn capture(&self, context: usize, layers: &[i32]) -> Result<()> {
        self.call(json!({"Op":"capture-layers","Context":context,"Layers":layers}))?;
        Ok(())
    }
    fn decode(
        &self,
        context: usize,
        values: &[i32],
        past: usize,
        all_logits: bool,
        final_chunk: bool,
    ) -> Result<()> {
        require(
            !values.is_empty() && values.len() <= self.batch_size,
            "batch extent exceeds configured source batch",
        )?;
        self.call(json!({"Op":"batch-clear","Batch":self.batch}))?;
        let rows: Vec<_>=values.iter().enumerate().map(|(i,token)|json!({"Token":token,"Position":past+i,"Sequences":[0],"Logits":all_logits||(final_chunk&&i+1==values.len())})).collect();
        self.call(json!({"Op":"batch-add","Batch":self.batch,"Rows":rows}))?;
        let value = self.call(json!({"Op":"decode","Context":context,"Batch":self.batch}))?;
        require(value["Code"] == 0, "source decode failed")
    }
    fn process(&self) -> Result<()> {
        let value = self
            .call(json!({"Op":"speculative-process","Speculative":self.spec,"Batch":self.batch}))?;
        require(
            value["Processed"] == true,
            "common_speculative_process failed",
        )
    }
    fn prefill(&self, context: usize, values: &[i32], capture: bool) -> Result<()> {
        for (index, chunk) in values.chunks(self.batch_size).enumerate() {
            self.decode(
                context,
                chunk,
                index * self.batch_size,
                capture,
                (index + 1) * self.batch_size >= values.len(),
            )?;
            if capture {
                self.process()?;
            }
        }
        Ok(())
    }
    fn sample(&self, sampler: usize, context: usize) -> Result<i32> {
        let value=self.call(json!({"Op":"common-sampler-sample","Sampler":sampler,"Context":context,"Index":-1,"GrammarFirst":false}))?;
        let token = i32::try_from(
            value["Token"]
                .as_i64()
                .ok_or_else(|| invalid("missing sampled token"))?,
        )?;
        self.call(
            json!({"Op":"common-sampler-accept","Sampler":sampler,"Token":token,"Generated":true}),
        )?;
        Ok(token)
    }
    fn logits(&self, context: usize, row: i32) -> Result<Vec<f32>> {
        let value = self.call(json!({"Op":"context-logits","Context":context,"Indices":[row]}))?;
        let rows = value["Values"]
            .as_array()
            .ok_or_else(|| invalid("missing logits rows"))?;
        require(
            rows.len() == 1 && rows[0]["Present"] == true,
            "source logits row missing",
        )?;
        let floats = super::generated_inputs::floats(&rows[0]["DataHex"])?;
        require(
            floats.len() == unsigned(&value, "Vocabulary")?,
            "logit vocabulary extent differs",
        )?;
        Ok(floats)
    }
}
struct Baseline {
    sampler: usize,
    tokens: Vec<i32>,
    logits: Vec<Vec<f32>>,
    seconds: f64,
    decode_seconds: f64,
}
fn baseline(
    rt: &Runtime,
    model: usize,
    prompt: &[i32],
    eog: &[i32],
    options: &Options,
) -> Result<Baseline> {
    rt.reset(rt.ar, false)?;
    rt.capture(rt.ar, &[])?;
    let start = Instant::now();
    rt.prefill(rt.ar, prompt, false)?;
    let sampler = rt.sampler(model)?;
    let mut logits = Vec::new();
    if options.oracle_rows.is_some() {
        logits.push(rt.logits(rt.ar, -1)?);
    }
    let mut current = rt.sample(sampler, rt.ar)?;
    let mut output = vec![current];
    let decode_start = Instant::now();
    while output.len() < options.prediction_limit && !eog.contains(&current) {
        rt.decode(
            rt.ar,
            &[current],
            prompt.len() + output.len() - 1,
            true,
            true,
        )?;
        if options.oracle_rows.is_some() {
            logits.push(rt.logits(rt.ar, -1)?);
        }
        current = rt.sample(sampler, rt.ar)?;
        output.push(current);
    }
    let seconds = start.elapsed().as_secs_f64();
    let decode_seconds = decode_start.elapsed().as_secs_f64();
    Ok(Baseline {
        sampler,
        tokens: output,
        logits,
        seconds,
        decode_seconds,
    })
}
fn oracle(
    rt: &Runtime,
    prompt: &[i32],
    ar: &Baseline,
    options: &Options,
) -> Result<(Value, usize)> {
    let context = if options.same_path { rt.ar } else { rt.target };
    rt.capture(context, &[])?;
    rt.reset(context, true)?;
    rt.prefill(
        context,
        if options.split_prefill {
            &prompt[..prompt.len() - 1]
        } else {
            prompt
        },
        false,
    )?;
    let mut count = 0_usize;
    let mut mismatches = Vec::new();
    let mut max_abs = 0_f32;
    let (mut kld_sum, mut kld_max) = (0_f64, 0_f64);
    let mut compare = |output_index: usize, row: i32| -> Result<()> {
        let actual = rt.logits(context, row)?;
        let reference = &ar.logits[output_index];
        let metrics = inputs::oracle_compare(reference, &actual, options.measure_kld)?;
        max_abs = max_abs.max(metrics.max_abs);
        count += 1;
        if let Some(kld) = metrics.kld {
            kld_sum += kld;
            kld_max = kld_max.max(kld);
        }
        let expected = usize::try_from(ar.tokens[output_index])?;
        require(
            expected < actual.len(),
            "sampled token outside oracle vocabulary",
        )?;
        if metrics.top != expected {
            let probabilities = if options.measure_kld {
                let ref_max =
                    f64::from(reference.iter().copied().fold(f32::NEG_INFINITY, f32::max));
                let test_max = f64::from(actual[metrics.top]);
                let ref_sum: f64 = reference
                    .iter()
                    .map(|&v| (f64::from(v) - ref_max).exp())
                    .sum();
                let test_sum: f64 = actual
                    .iter()
                    .map(|&v| (f64::from(v) - test_max).exp())
                    .sum();
                json!({"ReferenceAr":(f64::from(reference[expected])-ref_max).exp()/ref_sum,
                    "ReferenceOther":(f64::from(reference[metrics.top])-ref_max).exp()/ref_sum,
                    "TestAr":(f64::from(actual[expected])-test_max).exp()/test_sum,"TestOther":1.0/test_sum})
            } else {
                Value::Null
            };
            mismatches.push(json!({"TokenIndex":output_index,"Ar":ar.tokens[output_index],"Block":metrics.top,
                "ArMarginBits":(reference[expected]-reference[metrics.top]).to_bits(),
                "BlockMarginBits":(actual[metrics.top]-actual[expected]).to_bits(),"Kld":metrics.kld,"Probabilities":probabilities}));
        }
        Ok(())
    };
    let rows = options
        .oracle_rows
        .ok_or_else(|| invalid("oracle row count missing"))?;
    if options.split_prefill {
        for start in (0..ar.tokens.len()).step_by(rows) {
            let end = (start + rows).min(ar.tokens.len());
            let values: Vec<_> = (start..end)
                .map(|i| {
                    if i == 0 {
                        prompt[prompt.len() - 1]
                    } else {
                        ar.tokens[i - 1]
                    }
                })
                .collect();
            rt.decode(context, &values, prompt.len() - 1 + start, true, true)?;
            for i in start..end {
                compare(i, i32::try_from(i - start)?)?;
            }
        }
    } else {
        compare(0, -1)?;
        for start in (0..ar.tokens.len() - 1).step_by(rows) {
            let end = (start + rows).min(ar.tokens.len() - 1);
            rt.decode(
                context,
                &ar.tokens[start..end],
                prompt.len() + start,
                true,
                true,
            )?;
            for i in start..end {
                compare(i + 1, i32::try_from(i - start)?)?;
            }
        }
    }
    let mismatch_count = mismatches.len();
    Ok((
        json!({"Tokens":count,"Mismatches":mismatches,"MaxAbsoluteLogitBits":max_abs.to_bits(),
        "KldCount":if options.measure_kld {count}else{0},"KldMean":if options.measure_kld {Some(kld_sum/count_f64(count))}else{None},
        "KldMax":if options.measure_kld {Some(kld_max)}else{None}}),
        mismatch_count,
    ))
}
#[derive(Default)]
struct Totals {
    drafted: usize,
    accepted: usize,
    rounds: usize,
    predicted: usize,
    ar_predicted: usize,
    ar_seconds: f64,
    spec_seconds: f64,
}
impl Totals {
    fn report(&self) -> Value {
        let ar_rate = ratio(self.ar_predicted, self.ar_seconds);
        let sp_rate = ratio(self.predicted, self.spec_seconds);
        json!({"Drafted":self.drafted,"Accepted":self.accepted,"Rounds":self.rounds,"Predicted":self.predicted,
            "ArPredicted":self.ar_predicted,"ArSeconds":self.ar_seconds,"SpecSeconds":self.spec_seconds,
            "AcceptRate":ratio(self.accepted,count_f64(self.drafted)),"Tau":if self.rounds>0 {count_f64(self.accepted)/count_f64(self.rounds)+1.0}else{0.0},
            "ArTokensPerSecond":ar_rate,"SpecTokensPerSecond":sp_rate,"Speedup":if ar_rate>0.0 {sp_rate/ar_rate}else{0.0}})
    }
    fn add(&mut self, other: &Self) {
        self.drafted += other.drafted;
        self.accepted += other.accepted;
        self.rounds += other.rounds;
        self.predicted += other.predicted;
        self.ar_predicted += other.ar_predicted;
        self.ar_seconds += other.ar_seconds;
        self.spec_seconds += other.spec_seconds;
    }
}
#[allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Original double ratios of counters bounded by256 prompts and2048 tokens; all counts are exactly representable."
)]
const fn count_f64(n: usize) -> f64 {
    n as f64
}
fn ratio(n: usize, d: f64) -> f64 {
    if d > 0.0 { count_f64(n) / d } else { 0.0 }
}
struct SpecOutput {
    sampler: usize,
    tokens: Vec<i32>,
    totals: Totals,
    timing: Value,
    warning: bool,
}
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "Retain the source's single speculative round schedule and shared depth counters."
)]
fn speculative(
    rt: &Runtime,
    model: usize,
    prompt: &[i32],
    layers: &[i32],
    eog: &[i32],
    options: &Options,
    max_draft: usize,
    reached: &mut [usize],
    accepted: &mut [usize],
) -> Result<SpecOutput> {
    rt.reset(rt.target, false)?;
    rt.capture(rt.target, layers)?;
    let start = Instant::now();
    let prefix = &prompt[..prompt.len() - 1];
    rt.call(json!({"Op":"speculative-begin","Speculative":rt.spec,"Sequence":0,"Prompt":prefix}))?;
    let sampler = rt.sampler(model)?;
    rt.prefill(rt.target, prefix, true)?;
    let decode_start = Instant::now();
    let mut output = Vec::new();
    let mut past = prefix.len();
    let mut last = prompt[prompt.len() - 1];
    let mut totals = Totals::default();
    let mut retained = vec![0_usize; max_draft + 1];
    let (mut draft_us, mut verify_us, mut steady_us, mut steady_rounds) =
        (0_u128, 0_u128, 0_u128, 0_usize);
    let mut warning = false;
    while output.len() < options.prediction_limit
        && !output.last().is_some_and(|token| eog.contains(token))
    {
        let round_start = Instant::now();
        let draft_start = Instant::now();
        let response = rt.call(
            json!({"Op":"speculative-draft","Speculative":rt.spec,"Sequence":0,"Drafting":true,
            "Max":max_draft,"Past":past,"Last":last,"Prompt":null}),
        )?;
        draft_us += draft_start.elapsed().as_micros();
        let draft = tokens(&response, "Tokens")?;
        if draft.is_empty() {
            warning = true;
            break;
        }
        require(
            draft.len() < retained.len(),
            "retained depth exceeds configured cap",
        )?;
        retained[draft.len()] += 1;
        let mut verify = vec![last];
        verify.extend_from_slice(&draft);
        // Batch construction precedes the source's verify timer.
        rt.call(json!({"Op":"batch-clear","Batch":rt.batch}))?;
        let rows: Vec<_> = verify
            .iter()
            .enumerate()
            .map(
                |(i, token)| json!({"Token":token,"Position":past+i,"Sequences":[0],"Logits":true}),
            )
            .collect();
        rt.call(json!({"Op":"batch-add","Batch":rt.batch,"Rows":rows}))?;
        let verify_start = Instant::now();
        let decoded = rt.call(json!({"Op":"decode","Context":rt.target,"Batch":rt.batch}))?;
        require(decoded["Code"] == 0, "verify decode failed")?;
        rt.call(json!({"Op":"synchronize","Context":rt.target}))?;
        verify_us += verify_start.elapsed().as_micros();
        rt.process()?;
        let response=rt.call(json!({"Op":"common-sampler-sample-and-accept-n","Sampler":sampler,"Context":rt.target,"Indices":null,"Draft":draft,"GrammarFirst":false}))?;
        let ids = tokens(&response, "Tokens")?;
        require(!ids.is_empty(), "sample_and_accept_n returned empty")?;
        let n_accepted = usize::from(u16::try_from(ids.len() - 1)?);
        rt.call(json!({"Op":"speculative-accept","Speculative":rt.spec,"Sequence":0,"Accepted":n_accepted}))?;
        for i in 1..=draft.len() {
            if i <= n_accepted + 1 {
                reached[i] += 1;
            }
            if i <= n_accepted {
                accepted[i] += 1;
            }
        }
        totals.drafted += draft.len();
        totals.accepted += n_accepted;
        totals.rounds += 1;
        past += ids.len();
        for token in ids {
            if output.len() == options.prediction_limit {
                break;
            }
            last = token;
            output.push(token);
            if eog.contains(&token) {
                break;
            }
        }
        let crop = rt.call(
            json!({"Op":"memory-remove","Context":rt.target,"Sequence":0,"From":past,"To":-1}),
        )?;
        require(
            crop["Removed"] == true,
            "target recurrent-state rollback failed",
        )?;
        if totals.rounds > 2 {
            steady_us += round_start.elapsed().as_micros();
            steady_rounds += 1;
        }
    }
    totals.spec_seconds = start.elapsed().as_secs_f64();
    let decode_seconds = decode_start.elapsed().as_secs_f64();
    totals.predicted = output.len();
    Ok(SpecOutput {
        sampler,
        tokens: output,
        totals,
        warning,
        timing: json!({"DecodeSeconds":decode_seconds,"DraftMicroseconds":draft_us,
        "VerifyMicroseconds":verify_us,"SteadyMicroseconds":steady_us,"SteadyRounds":steady_rounds,"RetainedDepths":&retained[1..]}),
    })
}

pub(super) fn run(engine: &Engine) -> Result<Value> {
    run_with_options(engine, &Options::default())
}
#[allow(
    clippy::too_many_lines,
    reason = "Retain the source's model ownership and ordered prompt loop in one scope."
)]
pub(super) fn run_with_options(engine: &Engine, options: &Options) -> Result<Value> {
    require(
        std::env::var("GITHUB_ACTIONS").as_deref() == Ok("true")
            && std::env::var("GITHUB_REF").as_deref() == Ok("refs/heads/main"),
        "DSpark real evaluation includes benchmark timings and runs only on main-branch CI",
    )?;
    options.validate()?;
    let (target, drafter, input_receipt) =
        if let (Some(target), Some(drafter)) = (&options.target_path, &options.drafter_path) {
            (
                target.clone(),
                drafter.clone(),
                json!({"Kind":"CallerSuppliedPair","Target":target,"Drafter":drafter}),
            )
        } else {
            let pair = super::dspark_pair::create(engine, "corrected")?;
            (pair.target, pair.drafter, pair.receipt)
        };
    let layers = inputs::target_layers(&drafter)?;
    let prompts = inputs::prompts(options)?;
    let Some(session) = ModelSession::open(engine, "dspark-real-eval")? else {
        return Ok(result(
            ID,
            false,
            "production DSpark model/context/speculative/sampler APIs unavailable".into(),
        ));
    };
    let client = session.client();
    let call = |value| -> Result<Value> { Ok(client.call(0, value, None)?.value) };
    call(json!({"Op":"backend-init"}))?;
    let load = call(
        json!({"Op":"model-load","Path":target,"Base":"default","Params":null,"Overrides":{"GpuLayers":options.gpu_layers},"Progress":false}),
    )?;
    let target_model = unsigned(&load, "Model")?;
    require(target_model > 0, "failed to load target model")?;
    let mut overrides = json!({"GpuLayers":options.gpu_layers});
    if options.shared_head {
        overrides["HeadSource"] = json!(target_model);
    }
    let load = call(
        json!({"Op":"model-load","Path":drafter,"Base":"default","Params":null,"Overrides":overrides,"Progress":false}),
    )?;
    let draft_model = unsigned(&load, "Model")?;
    require(draft_model > 0, "failed to load drafter model")?;
    let meta = call(json!({"Op":"dspark-meta","Model":draft_model}))?;
    require(meta["Found"] == true, "drafter missing DSpark metadata")?;
    require(
        layers.len() == unsigned(&meta, "CaptureCount")?,
        "target_layers count != meta.n_capture",
    )?;
    let block = unsigned(&meta, "BlockSize")?;
    let max_draft = options.max_draft.unwrap_or(block);
    require(
        (1..=block).contains(&max_draft),
        "n_max_override must be in [1,block_size]",
    )?;
    let context = |model, snapshots, outputs| -> Result<usize> {
        let mut overrides = json!({"Context":options.context,"Batch":options.batch,"MicroBatch":options.batch,"Sequences":1,"NoPerf":true,"RecurrentSnapshots":snapshots});
        if outputs > 0 {
            overrides["OutputsMax"] = json!(outputs);
            overrides["OutputsPerSequence"] = json!(outputs);
        }
        let value =
            call(json!({"Op":"context","Model":model,"Params":null,"Overrides":overrides}))?;
        let handle = unsigned(&value, "Context")?;
        require(handle > 0, "failed to create source context")?;
        Ok(handle)
    };
    let target_context = context(target_model, block, 0)?;
    let info = call(json!({"Op":"dspark-context-info","Context":target_context}))?;
    require(
        info["Hybrid"] != true || unsigned(&info, "RsSequences")? > 0,
        "hybrid target requires recurrent rollback snapshots",
    )?;
    let ar_context = context(target_model, 0, 0)?;
    let draft_context = context(draft_model, 0, block)?;
    call(json!({"Op":"capture-layers","Context":target_context,"Layers":layers}))?;
    let value = call(
        json!({"Op":"speculative-init","DraftContext":draft_context,"TargetContext":target_context,"Max":max_draft,"Min":0,"Sequences":1}),
    )?;
    let spec = unsigned(&value, "Speculative")?;
    require(spec > 0, "common_speculative_init returned null")?;
    require(
        call(json!({"Op":"speculative-needs-capture","Speculative":spec}))?["Needed"] == true,
        "DSpark must need embedding capture",
    )?;
    let value =
        call(json!({"Op":"batch-new","Capacity":unsigned(&info,"BatchSize")?,"Sequences":1}))?;
    let batch = unsigned(&value, "Batch")?;
    let vocabulary = call(json!({"Op":"model-vocabulary","Model":target_model}))?;
    let eog = tokens(&vocabulary, "EogTokenIds")?;
    let rt = Runtime {
        client: client.clone(),
        batch,
        batch_size: options.batch,
        target: target_context,
        drafter: draft_context,
        ar: ar_context,
        spec,
    };
    let mut total = Totals::default();
    let mut categories: Vec<(String, Totals)> = Vec::new();
    let mut reached = vec![0; block + 1];
    let mut accepted = vec![0; block + 1];
    let mut failures = Vec::new();
    let mut reports = Vec::new();
    let mut oracle_mismatches = 0;
    for (index, prompt) in prompts.iter().enumerate() {
        rt.reset(rt.target, false)?;
        let text = if options.dataset_path.is_some() {
            let messages = if prompt.messages.is_empty() {
                vec![json!({"role":"user","content":prompt.text})]
            } else {
                prompt.messages.clone()
            };
            let value=rt.call(json!({"Op":"model-chat-render","Model":target_model,"Messages":messages,"EnableThinking":false,"AddGeneration":true}))?;
            value["Prompt"]
                .as_str()
                .ok_or_else(|| invalid("missing model chat prompt"))?
                .to_owned()
        } else {
            prompt.text.clone()
        };
        let value=rt.call(json!({"Op":"vocab-tokenize","Context":rt.target,"InputsHex":[super::generated_inputs::hex(text.as_bytes())?],"AddSpecial":true,"ParseSpecial":true}))?;
        require(value["Errors"][0].is_null(), "prompt tokenization failed")?;
        let inp = tokens(&json!({"Tokens":value["Values"][0]}), "Tokens")?;
        if inp.len() < 2 {
            reports.push(json!({"Index":index,"Skipped":"too short after tokenization"}));
            continue;
        }
        require(
            inp.len() + options.prediction_limit + block + 8 <= unsigned(&info, "ContextSize")?,
            "prompt plus output exceeds context capacity",
        )?;
        let ar = baseline(&rt, target_model, &inp, &eog, options)?;
        if options.oracle_rows.is_some() {
            let (metrics, mismatches) = oracle(&rt, &inp, &ar, options)?;
            oracle_mismatches += mismatches;
            reports.push(json!({"Index":index,"Category":prompt.category,"InputTokens":inp.len(),"Oracle":metrics}));
            rt.free_sampler(ar.sampler)?;
            continue;
        }
        let mut sp = speculative(
            &rt,
            target_model,
            &inp,
            &layers,
            &eog,
            options,
            max_draft,
            &mut reached,
            &mut accepted,
        )?;
        sp.totals.ar_predicted = ar.tokens.len();
        sp.totals.ar_seconds = ar.seconds;
        let parity = sp.tokens == ar.tokens;
        let first = sp
            .tokens
            .iter()
            .zip(&ar.tokens)
            .position(|(a, b)| a != b)
            .unwrap_or_else(|| sp.tokens.len().min(ar.tokens.len()));
        reports.push(json!({"Index":index,"Category":prompt.category,"InputTokens":inp.len(),"ArTokens":ar.tokens,"SpecTokens":sp.tokens,
            "Parity":parity,"FirstDifference":if parity {None}else{Some(first)},"EmptyDraftWarning":sp.warning,"Timing":sp.timing,
            "ArDecodeSeconds":ar.decode_seconds,"ArDecodeTokens":ar.tokens.len()-1,"Totals":sp.totals.report(),"ExcludedFromTotals":!parity}));
        if !parity {
            require(
                options.parity_continue,
                "refusing performance claim for non-equivalent output",
            )?;
            failures.push(index);
            rt.free_sampler(sp.sampler)?;
            rt.free_sampler(ar.sampler)?;
            continue;
        }
        total.add(&sp.totals);
        if let Some((_, category)) = categories
            .iter_mut()
            .find(|(name, _)| name == &prompt.category)
        {
            category.add(&sp.totals);
        } else {
            let mut stats = Totals::default();
            stats.add(&sp.totals);
            categories.push((prompt.category.clone(), stats));
        }
        rt.free_sampler(sp.sampler)?;
        rt.free_sampler(ar.sampler)?;
    }
    if options.oracle_rows.is_none() {
        rt.call(json!({"Op":"speculative-stats","Speculative":spec}))?;
    }
    rt.call(json!({"Op":"release","Kind":"batch","Handle":batch}))?;
    rt.call(json!({"Op":"speculative-release","Speculative":spec}))?;
    for id in [rt.drafter, rt.target, rt.ar] {
        rt.call(json!({"Op":"release","Kind":"context","Handle":id}))?;
    }
    for id in [draft_model, target_model] {
        rt.call(json!({"Op":"release","Kind":"model","Handle":id}))?;
    }
    rt.call(json!({"Op":"backend-free"}))?;
    session.finish()?;
    let depth:Vec<_>=(1..=max_draft).map(|i|json!({"Depth":i,"Reached":reached[i],"Accepted":accepted[i],"ConditionalAcceptance":ratio(accepted[i],count_f64(reached[i]))})).collect();
    let category_reports: Vec<_> = categories
        .iter()
        .map(|(name, stats)| json!({"Category":name,"Totals":stats.report()}))
        .collect();
    let mut output = result(
        ID,
        oracle_mismatches == 0,
        format!(
            "{} source prompts; {} parity failures; {} oracle mismatches",
            prompts.len(),
            failures.len(),
            oracle_mismatches
        ),
    );
    output["SourceSha256"] = json!(inputs::SOURCE_SHA);
    output["Options"] = serde_json::to_value(options)?;
    output["InputPair"] = input_receipt;
    output["Metadata"] = meta;
    output["TargetContext"] = info;
    output["Prompts"] = json!(reports);
    output["Totals"] = total.report();
    output["Categories"] = json!(category_reports);
    output["Depth"] = json!(depth);
    output["ParityFailedPrompts"] = json!(failures);
    output["ReportedPromptCount"] = json!(prompts.len() - failures.len());
    output["TimingScope"] = json!(
        "source wall-clock schedule including process dispatch; oracle timings are diagnostics only; generated models do not qualify publisher throughput"
    );
    output["SourceParityContinuePolicy"] = json!(
        "when requested, mismatches are recorded and excluded from aggregate totals; source retains their depth counters and exits successfully"
    );
    Ok(output)
}
