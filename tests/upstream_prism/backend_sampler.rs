//! Complete Rust-owned backend-sampler source schedules and comparisons.
//! Pinned upstream b87ca7225456b06b0975b8239c66b5aecf07e37c, MIT.
#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    reason = "Source arithmetic and sampler/reference identifiers remain aligned with the original C++ comparisons."
)]
use super::engine::Engine;
use super::llama_grammar::{array, exact_fields, field, uint};
use super::model_session::{Client, ModelSession, SamplerEvent};
use super::windows_random::Random;
use super::{Result, invalid, result};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[path = "backend_sampler_input.rs"]
mod input;

const ID: &str = "tests/test-backend-sampler.cpp";
const SOURCE_SHA: &str = "93711ee1d33e6a8a7b30973732c5dc1faee5a4cb742177eb0133bba1800271d8";
const NAMES: [&str; 19] = [
    "greedy",
    "logit_bias",
    "penalties",
    "temp",
    "temp_ext",
    "top_k",
    "multi_sequence",
    "dist",
    "dist_and_cpu",
    "set_sampler",
    "multi_output_limit",
    "multi_sequence_multi_output_dist",
    "multi_output_dist_transaction",
    "multi_output_sampling_chain",
    "multi_output_cpu",
    "mixed",
    "min_p",
    "cpu_mixed",
    "top_p",
];
fn verify(v: bool, message: impl Into<String>) -> Result<()> {
    if v { Ok(()) } else { Err(invalid(message)) }
}
fn signed(v: &Value, k: &str) -> Result<i32> {
    i32::try_from(
        field(v, k)?
            .as_i64()
            .ok_or_else(|| invalid(format!("backend sampler {k} integer required")))?,
    )
    .map_err(Into::into)
}
fn flag(v: &Value, k: &str) -> Result<bool> {
    field(v, k)?
        .as_bool()
        .ok_or_else(|| invalid(format!("backend sampler {k} boolean required")))
}
fn hex(v: &[u8]) -> String {
    v.iter()
        .flat_map(|b| [b >> 4, b & 15])
        .map(|n| char::from(b"0123456789abcdef"[usize::from(n)]))
        .collect()
}
fn f32s(bytes: &[u8]) -> Result<Vec<f32>> {
    verify(
        bytes.len() % 4 == 0,
        "backend sampler float byte extent differs",
    )?;
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| Ok(f32::from_le_bytes(*b)))
        .collect()
}
fn i32s(bytes: &[u8]) -> Result<Vec<i32>> {
    verify(
        bytes.len() % 4 == 0,
        "backend sampler token byte extent differs",
    )?;
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| Ok(i32::from_le_bytes(*b)))
        .collect()
}
#[derive(Default, Clone)]
struct Observer {
    initialized: bool,
    outputs: u32,
    backend_applies: u64,
    applies: u64,
}
#[derive(Default)]
struct Context {
    id: u64,
    positions: BTreeMap<i32, i32>,
    last: BTreeMap<i32, i32>,
}
#[derive(Clone, Copy)]
struct Candidate {
    id: i32,
    logit: f32,
    p: f32,
}
struct Output {
    token: Option<i32>,
    raw: Option<Vec<f32>>,
    logits: Option<Vec<f32>>,
    probs: Option<Vec<f32>>,
    candidates: Option<Vec<i32>>,
}
struct Run {
    client: Client,
    events: Arc<SamplerEvent>,
    observers: Arc<Mutex<BTreeMap<u32, Observer>>>,
    directory: PathBuf,
    serial: u64,
    model: u64,
    n: usize,
    bos: i32,
    eos: i32,
    checks: u64,
    diagnostics: Vec<String>,
}
impl Run {
    fn call(&self, v: Value) -> Result<Value> {
        let reply = self.client.call_with_event_handlers(
            0,
            v,
            None,
            None,
            Some(Arc::clone(&self.events)),
        )?;
        verify(
            reply.progress.is_empty(),
            "unexpected backend sampler progress event",
        )?;
        Ok(reply.value)
    }
    fn null(&self, v: Value) -> Result<()> {
        verify(
            self.call(v)?.is_null(),
            "backend sampler void acknowledgement differs",
        )
    }
    fn handle(&self, v: Value, name: &str) -> Result<u64> {
        let v = self.call(v)?;
        exact_fields(&v, &[name])?;
        let id = uint(&v, name)?;
        verify(id > 0, "null backend sampler production handle")?;
        Ok(id)
    }
    fn check(&mut self, v: bool, message: impl Into<String>) -> Result<()> {
        self.checks += 1;
        verify(v, message)
    }
    fn path(&mut self, label: &str) -> PathBuf {
        self.serial += 1;
        self.directory.join(format!("{label}-{}.bin", self.serial))
    }
    #[allow(
        clippy::needless_pass_by_value,
        reason = "Temporary JSON constructor parameters are consumed semantically by the owned wire request."
    )]
    fn init(&self, kind: &str, p: Value) -> Result<u64> {
        self.handle(
            json!({"Op":"bs-init","Type":kind,"Parameters":p}),
            "Sampler",
        )
    }
    fn chain(&self, items: &[(&str, Value)]) -> Result<u64> {
        let c = self.init("chain", json!({"NoPerf":null}))?;
        for (kind, p) in items {
            let s = self.init(kind, p.clone())?;
            self.null(json!({"Op":"bs-chain-add","Chain":c,"Sampler":s}))?;
        }
        Ok(c)
    }
    fn free_sampler(&self, s: u64) -> Result<()> {
        self.null(json!({"Op":"bs-free","Sampler":s}))
    }
    fn accept(&self, s: u64, t: i32) -> Result<()> {
        self.null(json!({"Op":"bs-accept","Sampler":s,"Token":t}))
    }
    fn context(
        &self,
        configs: &[(i32, u64)],
        sequences: Option<i32>,
        outputs: u32,
        micro: u32,
        per_seq: u32,
    ) -> Result<Context> {
        let seq =
            sequences.unwrap_or_else(|| configs.iter().map(|(id, _)| *id).max().unwrap_or(0) + 1);
        let mut overrides = json!({"Context":512,"Batch":512,"Sequences":seq,"OutputsMax":outputs,"OutputsPerSequence":per_seq,"Unified":true,"Samplers":configs.iter().map(|(id,s)|json!({"Sequence":id,"Sampler":s})).collect::<Vec<_>>()});
        if micro > 0 {
            overrides["MicroBatch"] = json!(micro);
        }
        let id = self.handle(
            json!({"Op":"context","Model":self.model,"Params":null,"Overrides":overrides}),
            "Context",
        )?;
        Ok(Context {
            id,
            ..Context::default()
        })
    }
    fn release(&self, kind: &str, id: u64) -> Result<()> {
        self.null(json!({"Op":"release","Kind":kind,"Handle":id}))
    }
    fn tokenize_capacity(&self, text: &str, capacity: usize) -> Result<(i32, Vec<i32>)> {
        let v=self.call(json!({"Op":"bs-tokenize","Model":self.model,"InputHex":hex(text.as_bytes()),"Capacity":capacity,"AddSpecial":false,"ParseSpecial":false}))?;
        exact_fields(&v, &["ReturnCode", "Backing"])?;
        let t = array(&v, "Backing")?
            .iter()
            .map(|v| {
                i32::try_from(
                    v.as_i64()
                        .ok_or_else(|| invalid("token backing integer required"))?,
                )
                .map_err(Into::into)
            })
            .collect::<Result<Vec<_>>>()?;
        verify(t.len() == capacity, "token backing fixed capacity differs")?;
        Ok((signed(&v, "ReturnCode")?, t))
    }
    fn tokenize_retry(&self, text: &str, capacity: usize) -> Result<Vec<i32>> {
        let (mut count, mut tokens) = self.tokenize_capacity(text, capacity)?;
        if count < 0 {
            (count, tokens) = self.tokenize_capacity(text, usize::try_from(-count)?)?;
        }
        verify(
            count >= 0 && usize::try_from(count)? <= tokens.len(),
            "token retry extent invalid",
        )?;
        tokens.truncate(usize::try_from(count)?);
        Ok(tokens)
    }
    fn piece(&mut self, token: i32) -> Result<()> {
        let mut v=self.call(json!({"Op":"bs-piece","Model":self.model,"Token":token,"Capacity":15,"LStrip":0,"Special":false}))?;
        let count = signed(&v, "ReturnCode")?;
        if count < 0 {
            v=self.call(json!({"Op":"bs-piece","Model":self.model,"Token":token,"Capacity":-count,"LStrip":0,"Special":false}))?;
            self.check(
                signed(&v, "ReturnCode")? == -count,
                "piece retry source assertion",
            )?;
        }
        Ok(())
    }
    fn rows(&self, c: &Context, rows: &[Value], capacity: usize, sequences: usize) -> Result<i32> {
        let b = self.handle(
            json!({"Op":"batch-new","Capacity":capacity,"Sequences":sequences}),
            "Batch",
        )?;
        self.null(json!({"Op":"batch-add","Batch":b,"Rows":rows}))?;
        let value = self.call(json!({"Op":"decode","Context":c.id,"Batch":b}))?;
        exact_fields(&value, &["Code"])?;
        self.release("batch", b)?;
        signed(&value, "Code")
    }
    fn prompts(&self, c: &mut Context, prompts: &[(i32, &str)]) -> Result<bool> {
        c.last.clear();
        let mut rows = Vec::new();
        for (seq, prompt) in prompts {
            let (count, t) = self.tokenize_capacity(prompt, 32)?;
            if count < 0 {
                return Ok(false);
            }
            let mut tokens = vec![self.bos];
            tokens.extend_from_slice(&t[..usize::try_from(count)?]);
            let start = *c.positions.entry(*seq).or_default();
            for (i, token) in tokens.iter().enumerate() {
                rows.push(json!({"Token":token,"Position":start+i32::try_from(i)?,"Sequences":[seq],"Logits":i+1==tokens.len()}));
            }
            c.positions
                .insert(*seq, start + i32::try_from(tokens.len())?);
        }
        let success = self.rows(c, &rows, 512, prompts.len())? == 0;
        if success {
            for (i, row) in rows.iter().enumerate() {
                if flag(row, "Logits")? {
                    c.last.insert(
                        i32::try_from(
                            row["Sequences"][0]
                                .as_i64()
                                .ok_or_else(|| invalid("sequence integer"))?,
                        )?,
                        i32::try_from(i)?,
                    );
                }
            }
        }
        Ok(success)
    }
    fn decode_tokens(&self, c: &mut Context, tokens: &[(i32, i32)]) -> Result<bool> {
        let mut rows = Vec::new();
        for (seq, t) in tokens {
            let pos = *c.positions.entry(*seq).or_default();
            rows.push(json!({"Token":t,"Position":pos,"Sequences":[seq],"Logits":true}));
        }
        let success = self.rows(c, &rows, tokens.len(), tokens.len())? == 0;
        if success {
            c.last.clear();
            for (i, (seq, _)) in tokens.iter().enumerate() {
                *c.positions.entry(*seq).or_default() += 1;
                c.last.insert(*seq, i32::try_from(i)?);
            }
        }
        Ok(success)
    }
    fn index(c: &Context, seq: i32) -> i32 {
        c.last.get(&seq).copied().unwrap_or(-1)
    }
    #[allow(
        clippy::too_many_arguments,
        clippy::fn_params_excessive_bools,
        reason = "Independent flags request exactly the source production getters without extra observations."
    )]
    fn observe(
        &mut self,
        c: &Context,
        index: i32,
        raw: bool,
        logits: bool,
        probs: bool,
        candidates: bool,
        token: bool,
    ) -> Result<Output> {
        let prefix = self.path("observation");
        let v=self.call(json!({"Op":"bs-output","Context":c.id,"Index":index,"Prefix":prefix,"Raw":raw,"Logits":logits,"Probabilities":probs,"Candidates":candidates,"Token":token}))?;
        exact_fields(
            &v,
            &["Raw", "Logits", "Probabilities", "Candidates", "Token"],
        )?;
        let load = |name: &str, enabled: bool| -> Result<Option<Vec<u8>>> {
            let x = field(&v, name)?;
            if !enabled {
                verify(x.is_null(), "unsolicited sampler output")?;
                return Ok(None);
            }
            exact_fields(x, &["Count", "Present", "File"])?;
            let count = usize::try_from(uint(x, "Count")?)?;
            verify(count <= 1_048_576, "sampler observation count bound")?;
            if !flag(x, "Present")? {
                verify(x["File"].is_null(), "absent output has file")?;
                return Ok(None);
            }
            let p = Path::new(
                x["File"]
                    .as_str()
                    .ok_or_else(|| invalid("output file required"))?,
            );
            verify(
                p.parent() == Some(self.directory.as_path()),
                "sampler output file outside evidence directory",
            )?;
            let bytes = fs::read(p)?;
            verify(
                bytes.len() == count * 4,
                "sampler output exact byte count differs",
            )?;
            Ok(Some(bytes))
        };
        Ok(Output {
            token: if token {
                Some(signed(&v, "Token")?)
            } else {
                verify(v["Token"].is_null(), "unsolicited sampled token")?;
                None
            },
            raw: load("Raw", raw)?.as_deref().map(f32s).transpose()?,
            logits: load("Logits", logits)?.as_deref().map(f32s).transpose()?,
            probs: load("Probabilities", probs)?
                .as_deref()
                .map(f32s)
                .transpose()?,
            candidates: load("Candidates", candidates)?
                .as_deref()
                .map(i32s)
                .transpose()?,
        })
    }
    fn sampled(&mut self, c: &Context, seq: i32) -> Result<i32> {
        self.observe(c, Self::index(c, seq), false, false, false, false, true)?
            .token
            .ok_or_else(|| invalid("sampled token missing"))
    }
    fn valid(&mut self, t: i32) -> Result<()> {
        self.check(
            t >= 0 && usize::try_from(t).is_ok_and(|x| x < self.n),
            "sampled token outside vocabulary",
        )
    }
    fn sample(&self, s: u64, c: &Context, index: i32) -> Result<i32> {
        let v = self.call(json!({"Op":"bs-sample","Sampler":s,"Context":c.id,"Index":index}))?;
        exact_fields(&v, &["Token"])?;
        signed(&v, "Token")
    }
    fn set(&self, c: &Context, seq: i32, s: Option<u64>) -> Result<()> {
        let v =
            self.call(json!({"Op":"bs-context-set","Context":c.id,"Sequence":seq,"Sampler":s}))?;
        exact_fields(&v, &["Set"])?;
        flag(&v, "Set")?;
        Ok(())
    }
    fn raw(&mut self, prompt: &str) -> Result<Vec<f32>> {
        let mut c = self.context(&[], None, 0, 0, 1)?;
        let ok = self.prompts(&mut c, &[(0, prompt)])?;
        self.check(ok, "raw reference decode failed")?;
        let out = self.observe(&c, Self::index(&c, 0), true, false, false, false, false)?;
        self.check(out.raw.is_some(), "raw reference logits absent")?;
        self.release("context", c.id)?;
        out.raw.ok_or_else(|| invalid("raw reference absent"))
    }
    fn candidates(&mut self, raw: &[f32]) -> Result<u64> {
        let file = self.path("candidate-input");
        let mut bytes = Vec::with_capacity(raw.len() * 12);
        for (i, x) in raw.iter().enumerate() {
            bytes.extend_from_slice(&i32::try_from(i)?.to_le_bytes());
            bytes.extend_from_slice(&x.to_le_bytes());
            bytes.extend_from_slice(&0_f32.to_le_bytes());
        }
        fs::write(&file, bytes)?;
        self.handle(json!({"Op":"bs-candidates-new","Count":raw.len(),"File":file,"Selected":-1,"Sorted":false}),"Candidates")
    }
    fn export(
        &mut self,
        candidates: u64,
        selection: &str,
        count: Option<usize>,
    ) -> Result<Vec<Candidate>> {
        let file = self.path("candidate-output");
        let v=self.call(json!({"Op":"bs-candidates-export","Candidates":candidates,"File":file,"Selection":selection,"Count":count}))?;
        exact_fields(
            &v,
            &[
                "BufferSize",
                "Size",
                "Offset",
                "Selected",
                "Sorted",
                "Rows",
                "Bytes",
            ],
        )?;
        let n = usize::try_from(uint(&v, "Rows")?)?;
        let bytes = fs::read(file)?;
        verify(
            bytes.len() == n * 12 && uint(&v, "Bytes")? == u64::try_from(bytes.len())?,
            "candidate serialization extent differs",
        )?;
        bytes
            .as_chunks::<12>()
            .0
            .iter()
            .map(|b| {
                Ok(Candidate {
                    id: i32::from_le_bytes(b[..4].try_into()?),
                    logit: f32::from_le_bytes(b[4..8].try_into()?),
                    p: f32::from_le_bytes(b[8..12].try_into()?),
                })
            })
            .collect()
    }
    fn apply(&self, s: u64, c: u64) -> Result<usize> {
        let v = self.call(json!({"Op":"bs-apply","Sampler":s,"Candidates":c}))?;
        exact_fields(&v, &["BufferSize", "Size", "Offset", "Selected", "Sorted"])?;
        usize::try_from(uint(&v, "Size")?).map_err(Into::into)
    }
    fn cpu(&mut self, raw: &[f32], s: u64) -> Result<Vec<Candidate>> {
        let c = self.candidates(raw)?;
        let n = self.apply(s, c)?;
        let out = self.export(c, "prefix", Some(n))?;
        self.null(json!({"Op":"bs-candidates-free","Candidates":c}))?;
        Ok(out)
    }
    fn backend(&mut self, prompt: &str, s: u64) -> Result<(Vec<f32>, Vec<i32>)> {
        let mut c = self.context(&[(0, s)], None, 0, 0, 1)?;
        let ok = self.prompts(&mut c, &[(0, prompt)])?;
        self.check(ok, "backend comparison decode failed")?;
        self.null(json!({"Op":"synchronize","Context":c.id}))?;
        let o = self.observe(&c, Self::index(&c, 0), false, true, false, true, false)?;
        self.check(
            o.logits.is_some(),
            "backend comparison sampled logits absent",
        )?;
        let logits = o
            .logits
            .ok_or_else(|| invalid("comparison logits absent"))?;
        let ids = match o.candidates {
            Some(v) if !v.is_empty() => {
                self.check(
                    v.len() == logits.len(),
                    "backend candidates/logits count differs",
                )?;
                v
            }
            _ => (0..logits.len())
                .map(|i| i32::try_from(i).map_err(Into::into))
                .collect::<Result<_>>()?,
        };
        self.release("context", c.id)?;
        Ok((logits, ids))
    }
}
fn temp(t: f32) -> (&'static str, Value) {
    ("temp", json!({"TemperatureBits":t.to_bits()}))
}
fn dist(seed: u32) -> (&'static str, Value) {
    ("dist", json!({"Seed":seed}))
}
fn topk(k: i32) -> (&'static str, Value) {
    ("top-k", json!({"K":k}))
}
fn topp(p: f32, keep: u32) -> (&'static str, Value) {
    (
        "top-p",
        json!({"ProbabilityBits":p.to_bits(),"MinKeep":keep}),
    )
}
fn minp(p: f32, keep: u32) -> (&'static str, Value) {
    (
        "min-p",
        json!({"ProbabilityBits":p.to_bits(),"MinKeep":keep}),
    )
}
fn penalties(n: usize, last: i32, repeat: f32, freq: f32, present: f32) -> (&'static str, Value) {
    (
        "penalties",
        json!({"VocabularySize":n,"PenaltyLastN":last,"RepeatBits":repeat.to_bits(),"FrequencyBits":freq.to_bits(),"PresenceBits":present.to_bits()}),
    )
}
fn bias(n: usize, token: i32, value: f32) -> (&'static str, Value) {
    (
        "logit-bias",
        json!({"VocabularySize":n,"Biases":[{"Token":token,"BiasBits":value.to_bits()}]}),
    )
}
impl Run {
    #[allow(
        clippy::too_many_lines,
        reason = "Keeping each source scenario in one schedule makes its original ordering and checks reviewable."
    )]
    fn basic(&mut self, name: &str) -> Result<()> {
        match name {
            "greedy" | "dist" | "dist_and_cpu" => {
                let s = self.chain(&[if name == "greedy" {
                    ("greedy", json!({}))
                } else {
                    dist(88)
                }])?;
                let mut c = self.context(&[(0, s)], None, 0, 0, 1)?;
                let ok = self.prompts(&mut c, &[(0, "Some")])?;
                self.check(ok, "prompt decode failed")?;
                let t = self
                    .observe(&c, Self::index(&c, 0), false, false, false, false, true)?
                    .token
                    .ok_or_else(|| invalid("single-output token absent"))?;
                self.piece(t)?;
                if name == "dist_and_cpu" {
                    let cpu = self.chain(&[dist(18)])?;
                    let sampled = self.sample(cpu, &c, Self::index(&c, 0))?;
                    self.piece(sampled)?;
                    self.check(t == sampled, "backend/CPU sampled tokens differ")?;
                    self.free_sampler(cpu)?;
                } else {
                    self.valid(t)?;
                    let last = self
                        .observe(&c, -1, false, false, false, false, true)?
                        .token
                        .ok_or_else(|| invalid("last token missing"))?;
                    self.piece(last)?;
                    self.valid(last)?;
                }
                if name == "greedy" {
                    for _ in 0..10 {
                        let t = self.sampled(&c, 0)?;
                        self.piece(t)?;
                        let ok = self.decode_tokens(&mut c, &[(0, t)])?;
                        self.check(ok, "greedy generation decode failed")?;
                    }
                }
                self.release("context", c.id)?;
                self.free_sampler(s)?;
            }
            "logit_bias" => {
                let (_, t) = self.tokenize_capacity("World", 16)?;
                let b = t[0];
                let s = self.chain(&[bias(self.n, b, 10.0), dist(88)])?;
                let mut c = self.context(&[(0, s)], None, 0, 0, 1)?;
                let ok = self.prompts(&mut c, &[(0, "Hello")])?;
                self.check(ok, "bias prompt decode failed")?;
                let t = self.sampled(&c, 0)?;
                self.check(t == b, "biased token differs")?;
                self.release("context", c.id)?;
                self.free_sampler(s)?;
            }
            "top_k" => {
                let s = self.chain(&[topk(8)])?;
                let mut c = self.context(&[(0, s)], None, 0, 0, 1)?;
                let ok = self.prompts(&mut c, &[(0, "Hello")])?;
                self.check(ok, "top-k prompt decode failed")?;
                let o = self.observe(&c, Self::index(&c, 0), false, true, false, true, false)?;
                if let Some(ids) = o.candidates {
                    for t in ids {
                        self.piece(t)?;
                    }
                }
                let cpu = self.chain(&[])?;
                let info = self.call(json!({"Op":"bs-info","Sampler":cpu}))?;
                self.check(
                    flag(&info, "HasBackendApply")?,
                    "empty chain backend apply missing",
                )?;
                let d = self.init("dist", dist(18).1)?;
                self.null(json!({"Op":"bs-chain-add","Chain":cpu,"Sampler":d}))?;
                let t = self.sample(cpu, &c, Self::index(&c, 0))?;
                self.valid(t)?;
                self.release("context", c.id)?;
                self.free_sampler(s)?;
                self.free_sampler(cpu)?;
            }
            "temp" => {
                let a = self.chain(&[temp(0.8)])?;
                let b = self.chain(&[temp(0.1)])?;
                let mut c = self.context(&[(0, a), (1, b)], None, 0, 0, 1)?;
                let ok = self.prompts(&mut c, &[(0, "Some where over the"), (1, "Once upon a")])?;
                self.check(ok, "temperature decode failed")?;
                let o = self.observe(&c, Self::index(&c, 0), false, true, false, false, false)?;
                self.check(
                    o.logits.as_ref().map(Vec::len) == Some(self.n),
                    "temperature vocabulary extent differs",
                )?;
                for seq in 0..2 {
                    let cpu = self.chain(&[dist(18)])?;
                    let t = self.sample(cpu, &c, Self::index(&c, seq))?;
                    self.piece(t)?;
                    self.valid(t)?;
                    self.free_sampler(cpu)?;
                }
                self.release("context", c.id)?;
                self.free_sampler(a)?;
                self.free_sampler(b)?;
                for t in [0.0, -1.0] {
                    let s = self.chain(&[temp(t)])?;
                    let mut c = self.context(&[(0, s)], None, 0, 0, 1)?;
                    let ok = self.prompts(&mut c, &[(0, "Once")])?;
                    self.check(ok, "nonpositive temperature decode failed")?;
                    let o =
                        self.observe(&c, Self::index(&c, 0), false, true, false, false, false)?;
                    self.check(
                        o.logits.as_ref().map(Vec::len) == Some(1),
                        "nonpositive temperature does not return one logit",
                    )?;
                    self.release("context", c.id)?;
                    self.free_sampler(s)?;
                }
            }
            "temp_ext" => {
                for (t, d, e, prompt) in [
                    (0.8_f32, 0.5_f32, 1.5_f32, "Once upon a"),
                    (0.0, 0.3, 1.0, "Once"),
                    (-1.0, 0.3, 2.0, "Once"),
                    (0.8, 0.0, 2.0, "Once"),
                ] {
                    let s=self.chain(&[("temp-ext",json!({"TemperatureBits":t.to_bits(),"DeltaBits":d.to_bits(),"ExponentBits":e.to_bits()}))])?;
                    let mut c = self.context(&[(0, s)], None, 0, 0, 1)?;
                    let ok = self.prompts(&mut c, &[(0, prompt)])?;
                    self.check(ok, "extended temperature decode failed")?;
                    let o =
                        self.observe(&c, Self::index(&c, 0), false, true, false, false, false)?;
                    self.check(
                        o.logits.as_ref().map(Vec::len)
                            == Some(if t <= 0.0 && d >= 0.0 { 1 } else { self.n }),
                        "extended temperature extent differs",
                    )?;
                    self.release("context", c.id)?;
                    self.free_sampler(s)?;
                }
            }
            "min_p" | "top_p" => {
                let s = self.chain(&[if name == "min_p" {
                    minp(0.1, 0)
                } else {
                    topp(0.9, 0)
                }])?;
                let mut c = self.context(&[(0, s)], None, 0, 0, 1)?;
                let ok = self.prompts(&mut c, &[(0, "Hello")])?;
                if name == "top_p" && !ok {
                    self.diagnostics.push("top_p returned early after failed prompt decode, exactly as source; its remaining predicates were not executed".into());
                    self.release("context", c.id)?;
                    self.free_sampler(s)?;
                    return Ok(());
                }
                if name == "min_p" {
                    self.check(ok, "min-p decode failed")?;
                }
                let o = self.observe(&c, Self::index(&c, 0), false, true, false, false, false)?;
                let count = o
                    .logits
                    .ok_or_else(|| invalid("filter diagnostics logits absent"))?
                    .iter()
                    .filter(|x| **x > -1e9)
                    .count();
                self.check(count < self.n, "filter retained full vocabulary")?;
                if name == "top_p" {
                    self.check(count > 0, "top-p retained no logits")?;
                }
                let cpu = self.chain(&[dist(88)])?;
                let t = self.sample(cpu, &c, Self::index(&c, 0))?;
                self.piece(t)?;
                self.valid(t)?;
                for _ in 0..10 {
                    let t = self.sample(cpu, &c, Self::index(&c, 0))?;
                    self.piece(t)?;
                    let ok = self.decode_tokens(&mut c, &[(0, t)])?;
                    if name == "min_p" {
                        self.check(ok, "min-p generation decode failed")?;
                    } else if !ok {
                        self.diagnostics
                            .push("top_p source ignores generation decode failure".into());
                    }
                }
                self.release("context", c.id)?;
                self.free_sampler(s)?;
                self.free_sampler(cpu)?;
            }
            "multi_sequence" | "mixed" | "cpu_mixed" => self.multi_basic(name)?,
            "set_sampler" => {
                let s = self.chain(&[dist(88)])?;
                let mut c = self.context(&[(0, s)], None, 0, 0, 1)?;
                let ok = self.prompts(&mut c, &[(0, "Hello")])?;
                self.check(ok, "set-sampler initial decode failed")?;
                let t = self.sampled(&c, 0)?;
                self.piece(t)?;
                self.set(&c, 0, None)?;
                let cpu = self.chain(&[dist(18)])?;
                let ok = self.decode_tokens(&mut c, &[(0, t)])?;
                self.check(ok, "cleared sampler decode failed")?;
                let o = self.observe(&c, Self::index(&c, 0), false, false, true, false, true)?;
                self.check(o.token == Some(-1), "cleared sampler still returns token")?;
                self.check(
                    o.probs.is_none(),
                    "cleared sampler still returns probabilities",
                )?;
                let t = self.sample(cpu, &c, 0)?;
                self.piece(t)?;
                let next = self.chain(&[topk(20), dist(88)])?;
                self.set(&c, 0, Some(next))?;
                let ok = self.decode_tokens(&mut c, &[(0, t)])?;
                self.check(ok, "replacement sampler decode failed")?;
                let t = self.sampled(&c, 0)?;
                self.piece(t)?;
                self.release("context", c.id)?;
                for s in [s, cpu, next] {
                    self.free_sampler(s)?;
                }
            }
            _ => return Err(invalid("unknown basic backend sampler case")),
        }
        Ok(())
    }
    fn multi_basic(&mut self, name: &str) -> Result<()> {
        let a = self.chain(&[if name == "multi_sequence" {
            ("greedy", json!({}))
        } else {
            dist(88)
        }])?;
        let b = if name == "cpu_mixed" {
            None
        } else {
            Some(self.chain(&if name == "multi_sequence" {
                vec![temp(0.8), ("greedy", json!({}))]
            } else {
                vec![topk(40)]
            })?)
        };
        let configs = b.map_or_else(|| vec![(0, a)], |b| vec![(0, a), (1, b)]);
        let mut c = self.context(
            &configs,
            if name == "cpu_mixed" { Some(2) } else { None },
            0,
            0,
            1,
        )?;
        let ok = self.prompts(&mut c, &[(0, "Hello"), (1, "Some")])?;
        self.check(ok, "multi-sequence prompt decode failed")?;
        let t = self.sampled(&c, 0)?;
        self.piece(t)?;
        self.valid(t)?;
        match name {
            "multi_sequence" => {
                let t = self.sampled(&c, 1)?;
                self.piece(t)?;
                self.valid(t)?;
                for _ in 0..4 {
                    let mut tokens = Vec::new();
                    for seq in 0..2 {
                        let t = self.sampled(&c, seq)?;
                        self.piece(t)?;
                        tokens.push((seq, t));
                    }
                    let ok = self.decode_tokens(&mut c, &tokens)?;
                    self.check(ok, "multi-sequence generation decode failed")?;
                }
            }
            "mixed" => {
                let o = self.observe(&c, Self::index(&c, 1), false, true, false, false, true)?;
                self.check(o.logits.is_some(), "mixed top-k logits absent")?;
                self.check(
                    o.logits.as_ref().map(Vec::len) == Some(40),
                    "mixed top-k count differs",
                )?;
                self.check(o.token == Some(-1), "top-k-only backend returned token")?;
            }
            "cpu_mixed" => {
                let t = self.sampled(&c, 1)?;
                self.check(t == -1, "CPU sequence unexpectedly backend sampled")?;
                let cpu = self.chain(&[("greedy", json!({}))])?;
                let t = self.sample(cpu, &c, Self::index(&c, 1))?;
                self.piece(t)?;
                self.valid(t)?;
                self.free_sampler(cpu)?;
                self.set(&c, 0, None)?;
                let cpu = self.chain(&[("greedy", json!({}))])?;
                let t = self.sample(cpu, &c, Self::index(&c, 1))?;
                let ok = self.decode_tokens(&mut c, &[(1, t)])?;
                self.check(ok, "CPU mixed decode failed")?;
                self.free_sampler(cpu)?;
                let next = self.chain(&[dist(88)])?;
                self.set(&c, 0, Some(next))?;
                let ok = self.decode_tokens(&mut c, &[(0, 3834)])?;
                self.check(ok, "re-added backend decode failed")?;
                let t = self.sampled(&c, 0)?;
                self.piece(t)?;
                self.valid(t)?;
                self.free_sampler(next)?;
            }
            _ => unreachable!(),
        }
        self.release("context", c.id)?;
        self.free_sampler(a)?;
        if let Some(b) = b {
            self.free_sampler(b)?;
        }
        Ok(())
    }
}
#[derive(Default)]
struct Stats {
    mismatch: usize,
    masked: usize,
}
fn compare(
    expected: &BTreeMap<i32, f32>,
    logits: &[f32],
    ids: &[i32],
    allow_extra: bool,
) -> Result<Stats> {
    verify(
        logits.len() == ids.len(),
        "comparison output vector sizes differ",
    )?;
    let mut out = Stats::default();
    let mut seen = BTreeSet::new();
    for (&token, &logit) in ids.iter().zip(logits) {
        if !seen.insert(token) || logit.is_nan() {
            out.mismatch += 1;
            continue;
        }
        if let Some(&e) = expected.get(&token) {
            if !logit.is_finite() || (e - logit).abs() > 1e-3 {
                out.mismatch += 1;
            }
        } else if logit.is_infinite() && logit < 0.0 {
            out.masked += 1;
        } else if !allow_extra {
            out.mismatch += 1;
        }
    }
    out.mismatch += expected
        .keys()
        .filter(|token| !seen.contains(token))
        .count();
    Ok(out)
}
fn mapped(rows: &[Candidate]) -> BTreeMap<i32, f32> {
    rows.iter().map(|c| (c.id, c.logit)).collect()
}
impl Run {
    fn comparison(
        &mut self,
        prompt: &str,
        raw: &[f32],
        items: &[(&str, Value)],
        history: &[i32],
    ) -> Result<(Vec<Candidate>, Vec<f32>, Vec<i32>)> {
        let cpu = self.chain(items)?;
        let backend = self.chain(items)?;
        for s in [cpu, backend] {
            for &t in history {
                self.accept(s, t)?;
            }
        }
        let expected = self.cpu(raw, cpu)?;
        let (logits, ids) = self.backend(prompt, backend)?;
        self.free_sampler(cpu)?;
        self.free_sampler(backend)?;
        Ok((expected, logits, ids))
    }
    fn history(&self, prompt: &str) -> Result<Vec<i32>> {
        let mut h = Vec::new();
        if self.bos != -1 {
            h.push(self.bos);
        }
        h.extend(self.tokenize_retry(prompt, 64)?);
        Ok(h)
    }
    #[allow(
        clippy::too_many_arguments,
        reason = "Arguments mirror the original penalty comparison helper and its explicit schedule."
    )]
    fn penalty_compare(
        &mut self,
        last: i32,
        repeat: f32,
        freq: f32,
        present: f32,
        prompt: &str,
        extra: &[i32],
        strict_mask: bool,
    ) -> Result<bool> {
        let raw = self.raw(prompt)?;
        let mut h = self.history(prompt)?;
        h.extend_from_slice(extra);
        let (e, l, ids) = self.comparison(
            prompt,
            &raw,
            &[penalties(self.n, last, repeat, freq, present)],
            &h,
        )?;
        self.check(
            e.len() == l.len(),
            "penalties expected/backend size differs",
        )?;
        let stats = compare(&mapped(&e), &l, &ids, false)?;
        if strict_mask {
            self.check(
                stats.masked == 0,
                "unfiltered penalties unexpectedly masks tokens",
            )?;
            self.check(stats.mismatch == 0, "penalty CPU/backend logits differ")?;
        }
        Ok(stats.mismatch == 0)
    }
    fn topk_penalty(&mut self, before: bool) -> Result<()> {
        let raw = self.raw("Hello")?;
        self.check(self.n > 8, "top-k penalty vocabulary too small")?;
        let filter = self.init("top-k", topk(8).1)?;
        let filtered = self.cpu(&raw, filter)?;
        self.free_sampler(filter)?;
        self.check(filtered.len() == 8, "CPU top-k extent differs")?;
        let retained = filtered[0].id;
        let excluded = (0..self.n)
            .filter_map(|x| i32::try_from(x).ok())
            .find(|t| !filtered.iter().any(|x| x.id == *t))
            .unwrap_or(-1);
        self.check(
            excluded != -1,
            "top-k penalty has no excluded history token",
        )?;
        let p = penalties(self.n, 64, 1.1, 0.5, 0.25);
        let items = if before {
            vec![p, topk(8)]
        } else {
            vec![topk(8), p]
        };
        let mut history = self.history("Hello")?;
        history.extend([excluded, excluded, retained, retained]);
        let (e, l, ids) = self.comparison("Hello", &raw, &items, &history)?;
        self.check(e.len() == 8, "expected penalties/top-k extent differs")?;
        self.check(l.len() == 8, "backend penalties/top-k extent differs")?;
        let expected = mapped(&e);
        if before {
            let original = mapped(&filtered);
            self.check(
                expected
                    .iter()
                    .any(|(t, l)| original.get(t).is_none_or(|v| (v - l).abs() > 1e-6)),
                "penalties before top-k did not change output",
            )?;
        } else {
            self.check(
                expected.contains_key(&retained),
                "retained history token absent",
            )?;
            self.check(
                (expected[&retained] - raw[usize::try_from(retained)?]).abs() > 1e-6,
                "retained token not penalized",
            )?;
            self.check(
                !expected.contains_key(&excluded),
                "CPU excluded token remains",
            )?;
            self.check(!ids.contains(&excluded), "backend excluded token remains")?;
        }
        let stats = compare(&expected, &l, &ids, false)?;
        self.check(
            stats.masked == 0,
            "top-k penalties unexpectedly masks tokens",
        )?;
        self.check(stats.mismatch == 0, "top-k penalties CPU/backend mismatch")
    }
    #[allow(
        clippy::too_many_arguments,
        reason = "Parameters mirror the source's masking-penalties matrix without hidden defaults."
    )]
    fn masking_penalty(
        &mut self,
        filter: (&str, Value),
        before: bool,
        last: i32,
        repeat: f32,
        freq: f32,
        present: f32,
        allow_extra: bool,
        add_history: bool,
    ) -> Result<()> {
        let raw = self.raw("Hello")?;
        let s = self.init(filter.0, filter.1.clone())?;
        let filtered = self.cpu(&raw, s)?;
        self.free_sampler(s)?;
        self.check(!filtered.is_empty(), "CPU masking filter empty")?;
        self.check(
            filtered.len() < self.n,
            "CPU masking filter retained vocabulary",
        )?;
        let penalized = filtered[0].id;
        let retained = filtered.iter().map(|x| x.id).collect::<BTreeSet<_>>();
        let masked = (0..self.n)
            .filter_map(|x| i32::try_from(x).ok())
            .find(|t| !retained.contains(t))
            .unwrap_or(-1);
        self.check(masked != -1, "masking filter has no excluded token")?;
        let p = penalties(self.n, last, repeat, freq, present);
        let items = if before {
            vec![p.clone(), filter]
        } else {
            vec![filter, p.clone()]
        };
        let mut history = if add_history {
            self.history("Hello")?
        } else {
            Vec::new()
        };
        if add_history {
            history.extend([penalized, penalized, masked, masked]);
        }
        let (e, l, ids) = self.comparison("Hello", &raw, &items, &history)?;
        self.check(
            l.len() == self.n,
            "backend masking penalties extent differs",
        )?;
        let expected = mapped(&e);
        self.check(!expected.contains_key(&masked), "CPU masked token remains")?;
        if add_history {
            if before {
                let sampler = self.init(p.0, p.1)?;
                for t in history {
                    self.accept(sampler, t)?;
                }
                let rows = self.cpu(&raw, sampler)?;
                self.free_sampler(sampler)?;
                self.check(
                    (mapped(&rows)[&penalized] - raw[usize::try_from(penalized)?]).abs() > 1e-6,
                    "pre-filter history penalty absent",
                )?;
            } else {
                self.check(
                    expected.contains_key(&penalized),
                    "retained penalized token absent",
                )?;
                self.check(
                    (expected[&penalized] - raw[usize::try_from(penalized)?]).abs() > 1e-6,
                    "post-filter history penalty absent",
                )?;
            }
        }
        let stats = compare(&expected, &l, &ids, allow_extra)?;
        let value = ids
            .iter()
            .position(|t| *t == masked)
            .map(|i| l[i])
            .ok_or_else(|| invalid("backend masked token absent"))?;
        self.check(stats.masked > 0, "backend filter did not mask tokens")?;
        self.check(
            value.is_infinite() && value < 0.0,
            "backend masked logit is not negative infinity",
        )?;
        self.check(
            stats.mismatch == 0,
            "masking penalties CPU/backend mismatch",
        )
    }
    fn penalty_cases(&mut self) -> Result<()> {
        self.penalty_compare(64, 1.1, 0.5, 0.25, "Hello Hello world", &[], true)?;
        let tokens = self.tokenize_retry("a", 8)?;
        self.check(!tokens.is_empty(), "penalties history tokenization empty")?;
        self.penalty_compare(80, 1.15, 0.1, 0.05, "a", &vec![tokens[0]; 78], true)?;
        self.penalty_compare(
            64,
            1.1,
            0.5,
            0.25,
            "Hello",
            &(0..64).collect::<Vec<_>>(),
            true,
        )?;
        self.topk_penalty(false)?;
        self.topk_penalty(true)?;
        self.masking_penalty(topp(0.9, 0), false, 64, 1.1, 0.5, 0.25, true, true)?;
        self.masking_penalty(topp(0.9, 0), false, 4096, 1.1, 0.5, 0.25, true, true)?;
        self.masking_penalty(topp(0.9, 0), true, 64, 1.1, 0.5, 0.25, true, true)?;
        self.masking_penalty(minp(0.1, 0), false, 64, 1.1, 0.5, 0.25, false, true)?;
        self.masking_penalty(minp(0.1, 0), true, 64, 1.1, 0.5, 0.25, false, true)?;
        self.masking_penalty(topp(0.9, 0), false, 64, 1.1, 0.5, 0.25, true, false)?;
        for (r, f, p) in [(1.1, 0.0, 0.0), (1.0, 0.5, 0.0), (1.0, 0.0, 0.25)] {
            self.masking_penalty(topp(0.9, 0), false, 64, r, f, p, true, true)?;
        }
        let mut failed = 0;
        for (r, f, p) in [
            (1.0, -1.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (1.0, 0.0, -1.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
        ] {
            failed +=
                usize::from(!self.penalty_compare(64, r, f, p, "Hello Hello world", &[], false)?);
        }
        self.check(failed == 0, "penalty parameter value comparison failures")
    }
}
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "The original double distribution is assigned to a float before each CDF comparison."
)]
fn random_float(rng: &mut Random) -> f32 {
    rng.uniform_f64(0.0, 1.0) as f32
}
impl Run {
    fn raw_batch(&self, c: &Context, entries: &[(i32, i32, i32)], capacity: usize) -> Result<i32> {
        let rows=entries.iter().map(|(seq,token,pos)|json!({"Token":token,"Position":pos,"Sequences":[seq],"Logits":true})).collect::<Vec<_>>();
        self.rows(c, &rows, capacity, 1)
    }
    fn verify_random(
        &mut self,
        c: &Context,
        chain: u64,
        row: i32,
        rnd: f32,
        accept: bool,
    ) -> Result<()> {
        let token = if accept {
            self.sample(chain, c, row)?
        } else {
            self.observe(c, row, false, false, false, false, true)?
                .token
                .ok_or_else(|| invalid("transaction token absent"))?
        };
        let out = self.observe(c, row, false, false, true, false, false)?;
        self.valid(token)?;
        self.check(out.probs.is_some(), "transaction probabilities absent")?;
        let p = out
            .probs
            .ok_or_else(|| invalid("transaction probabilities absent"))?;
        verify(
            p.len() > usize::try_from(token)?,
            "transaction sampled index out of probability buffer",
        )?;
        let mut before = 0_f32;
        for &x in &p[..usize::try_from(token)?] {
            before += x;
        }
        let through = before + p[usize::try_from(token)?];
        self.check(rnd >= before - 1e-4, "transaction random below sampled CDF")?;
        self.check(
            rnd <= through + 1e-4,
            "transaction random above sampled CDF",
        )
    }
    fn multi_dist(&mut self) -> Result<()> {
        let a = self.chain(&[temp(10.0), dist(88)])?;
        let b = self.chain(&[temp(10.0), dist(1337)])?;
        let c = self.context(&[(0, a), (1, b)], Some(2), 4, 0, 2)?;
        let reference = self.context(&[], Some(2), 4, 0, 1)?;
        let rows = [
            (0, self.bos, 0),
            (1, self.eos, 0),
            (0, self.eos, 1),
            (1, self.bos, 1),
        ];
        let code = self.raw_batch(&c, &rows, 4)?;
        self.check(code == 0, "multi-output backend decode failed")?;
        let code = self.raw_batch(&reference, &rows, 4)?;
        self.check(code == 0, "multi-output raw reference decode failed")?;
        let mut randoms = [Random::new(88), Random::new(1337)];
        for (i, (seq, _, _)) in rows.iter().enumerate() {
            self.check(*seq == 0 || *seq == 1, "multi-output sequence not 0/1")?;
            let token = self.sample(if *seq == 0 { a } else { b }, &c, i32::try_from(i)?)?;
            let o = self.observe(&c, i32::try_from(i)?, false, true, true, false, false)?;
            let r = self.observe(
                &reference,
                i32::try_from(i)?,
                true,
                false,
                false,
                false,
                false,
            )?;
            self.valid(token)?;
            self.check(o.logits.is_some(), "multi-output logits absent")?;
            self.check(o.probs.is_some(), "multi-output probabilities absent")?;
            self.check(r.raw.is_some(), "multi-output raw reference absent")?;
            let l = o.logits.ok_or_else(|| invalid("sampled logits absent"))?;
            let p = o
                .probs
                .ok_or_else(|| invalid("sampled probabilities absent"))?;
            let raw = r.raw.ok_or_else(|| invalid("reference logits absent"))?;
            self.check(l.len() == self.n, "multi-output logits extent differs")?;
            self.check(p.len() == self.n, "multi-output probability extent differs")?;
            verify(raw.len() == self.n, "reference vocabulary extent differs")?;
            let mut total = 0_f32;
            let mut before = 0_f32;
            for (j, ((&x, &prob), &original)) in l.iter().zip(&p).zip(&raw).enumerate() {
                let expected = original / 10.0;
                let tolerance = 1e-4 * expected.abs().max(1.0);
                self.check(
                    (x - expected).abs() <= tolerance,
                    "multi-output temperature logit tolerance exceeded",
                )?;
                self.check(prob.is_finite(), "multi-output probability is not finite")?;
                self.check(prob >= 0.0, "multi-output negative probability")?;
                total += prob;
                if i32::try_from(j)? < token {
                    before += prob;
                }
            }
            self.check(
                (total - 1.0).abs() <= 1e-3,
                "multi-output probabilities do not sum to one",
            )?;
            let rnd = random_float(&mut randoms[usize::try_from(*seq)?]);
            let through = before + p[usize::try_from(token)?];
            self.check(
                rnd >= before - 1e-4,
                "multi-output random below sampled CDF",
            )?;
            self.check(
                rnd <= through + 1e-4,
                "multi-output random above sampled CDF",
            )?;
        }
        self.release("context", c.id)?;
        self.release("context", reference.id)?;
        self.free_sampler(a)?;
        self.free_sampler(b)
    }
    fn transaction(&mut self) -> Result<()> {
        let s = self.chain(&[temp(10.0), dist(95)])?;
        let c = self.context(&[(0, s)], Some(1), 3, 2, 3)?;
        let mut rng = Random::new(95);
        let randoms = [
            random_float(&mut rng),
            random_float(&mut rng),
            random_float(&mut rng),
        ];
        let mut pos = 0;
        for stage in 0..4 {
            let rows = (0..3).map(|i| (0, self.bos, pos + i)).collect::<Vec<_>>();
            pos += 3;
            let code = self.raw_batch(&c, &rows, 3)?;
            self.check(code == 0, "transaction decode failed")?;
            match stage {
                0 => self.verify_random(&c, s, 0, randoms[0], false)?,
                1 => {
                    self.verify_random(&c, s, 0, randoms[0], true)?;
                    self.verify_random(&c, s, 1, randoms[1], true)?;
                }
                2 => {
                    let saved = self.handle(json!({"Op":"bs-clone","Sampler":s}), "Sampler")?;
                    self.verify_random(&c, s, 0, randoms[2], true)?;
                    self.null(json!({"Op":"bs-copy","Source":saved,"Destination":s}))?;
                    self.free_sampler(saved)?;
                }
                3 => self.verify_random(&c, s, 0, randoms[2], true)?,
                _ => unreachable!(),
            }
        }
        self.release("context", c.id)?;
        self.free_sampler(s)
    }
    fn output_limit(&mut self) -> Result<()> {
        let s = self.chain(&[dist(88)])?;
        let c = self.context(&[(0, s)], Some(1), 3, 0, 2)?;
        let code = self.raw_batch(
            &c,
            &[(0, self.bos, 0), (0, self.bos, 1), (0, self.bos, 2)],
            3,
        )?;
        self.check(
            code != 0,
            "output limit should reject three outputs above per-sequence two",
        )?;
        self.release("context", c.id)?;
        self.free_sampler(s)
    }
    #[allow(
        clippy::too_many_lines,
        reason = "Keeping each source scenario in one schedule makes its original ordering and checks reviewable."
    )]
    fn sampling_chain(&mut self) -> Result<()> {
        let k = self.n.min(512);
        let bias = bias(self.n, self.bos, -0.1);
        let filters = [
            bias.clone(),
            topk(i32::try_from(k)?),
            topp(0.9, 1),
            minp(0.01, 1),
            temp(0.8),
        ];
        let mut items = filters.to_vec();
        items.push(dist(88));
        let s = self.chain(&items)?;
        let c = self.context(&[(0, s)], Some(1), 2, 2, 2)?;
        let reference = self.context(&[], Some(1), 2, 2, 1)?;
        let mut cpu = Vec::new();
        for (kind, p) in &filters {
            cpu.push(self.init(kind, p.clone())?);
        }
        let rows = [(0, self.bos, 0), (0, self.bos, 1)];
        let code = self.raw_batch(&c, &rows, 2)?;
        self.check(code == 0, "sampling-chain backend decode failed")?;
        let code = self.raw_batch(&reference, &rows, 2)?;
        self.check(code == 0, "sampling-chain reference decode failed")?;
        for row in 0..2 {
            let token = self.sample(s, &c, row)?;
            let o = self.observe(&c, row, false, true, true, true, false)?;
            let r = self.observe(&reference, row, true, false, false, false, false)?;
            self.valid(token)?;
            self.check(o.logits.is_some(), "sampling-chain logits absent")?;
            self.check(o.probs.is_some(), "sampling-chain probabilities absent")?;
            self.check(o.candidates.is_some(), "sampling-chain candidates absent")?;
            self.check(r.raw.is_some(), "sampling-chain reference absent")?;
            let l = o.logits.ok_or_else(|| invalid("chain logits absent"))?;
            let p = o
                .probs
                .ok_or_else(|| invalid("chain probabilities absent"))?;
            let ids = o
                .candidates
                .ok_or_else(|| invalid("chain candidates absent"))?;
            self.check(l.len() == k, "sampling-chain logits count differs")?;
            self.check(
                p.len() == l.len(),
                "sampling-chain probability count differs",
            )?;
            self.check(
                ids.len() == l.len(),
                "sampling-chain candidate count differs",
            )?;
            let candidates =
                self.candidates(&r.raw.ok_or_else(|| invalid("chain reference absent"))?)?;
            for &s in &cpu[..3] {
                self.apply(s, candidates)?;
            }
            let preliminary = self.export(candidates, "active", None)?;
            self.check(!preliminary.is_empty(), "sampling-chain CPU top-p empty")?;
            let mut cdf = 0_f32;
            for v in &preliminary {
                cdf += v.p;
            }
            let before = cdf
                - preliminary
                    .last()
                    .ok_or_else(|| invalid("empty preliminary"))?
                    .p;
            let boundary = (before - 0.9).abs().min((cdf - 0.9).abs());
            for &s in &cpu[3..] {
                self.apply(s, candidates)?;
            }
            let expected = self.export(candidates, "active", None)?;
            self.null(json!({"Op":"bs-candidates-free","Candidates":candidates}))?;
            let mut by_id = mapped(&expected);
            let mut extra = 0_usize;
            let mut sampled = None;
            let mut total = 0_f32;
            for (i, ((&id, &logit), &prob)) in ids.iter().zip(&l).zip(&p).enumerate() {
                self.check(
                    id >= 0 && usize::try_from(id).is_ok_and(|i| i < self.n),
                    "sampling-chain candidate outside vocabulary",
                )?;
                self.check(prob.is_finite(), "sampling-chain nonfinite probability")?;
                self.check(prob >= 0.0, "sampling-chain negative probability")?;
                total += prob;
                if id == token {
                    sampled = Some(i);
                }
                if !logit.is_finite() {
                    self.check(
                        logit.is_infinite() && logit < 0.0,
                        "sampling-chain masked logit invalid",
                    )?;
                    self.check(prob == 0.0, "sampling-chain masked probability nonzero")?;
                    continue;
                }
                if let Some(expected) = by_id.remove(&id) {
                    let tolerance = 1e-4 * expected.abs().max(1.0);
                    self.check(
                        (logit - expected).abs() <= tolerance,
                        "sampling-chain logit tolerance exceeded",
                    )?;
                } else {
                    extra += 1;
                }
            }
            if extra != 0 || !by_id.is_empty() {
                self.check(
                    extra <= 1,
                    "sampling-chain too many backend-only candidates",
                )?;
                self.check(
                    by_id.len() <= 1,
                    "sampling-chain too many reference-only candidates",
                )?;
                self.check(
                    boundary <= 1e-4,
                    "sampling-chain boundary candidate difference outside epsilon",
                )?;
            }
            self.check(sampled.is_some(), "sampled token not in chain candidates")?;
            let i = sampled.ok_or_else(|| invalid("sampled candidate absent"))?;
            self.check(l[i].is_finite(), "sampled chain logit nonfinite")?;
            self.check(p[i] > 0.0, "sampled chain probability not positive")?;
            self.check(
                (total - 1.0).abs() <= 1e-3,
                "sampling-chain probabilities do not sum to one",
            )?;
        }
        let code = self.raw_batch(&c, &[(0, self.bos, 2), (0, self.bos, 3)], 2)?;
        self.check(code == 0, "sampling-chain second batch decode failed")?;
        self.release("context", c.id)?;
        self.release("context", reference.id)?;
        self.free_sampler(s)?;
        for s in cpu {
            self.free_sampler(s)?;
        }
        Ok(())
    }
    fn cpu_suffix(&mut self) -> Result<()> {
        for label in 0..2_u32 {
            let observer = self.init(
                "observer",
                json!({"Name":"single-output-backend","Label":label}),
            )?;
            let chain = self.chain(&[topk(8)])?;
            self.null(json!({"Op":"bs-chain-add","Chain":chain,"Sampler":observer}))?;
            let dist = self.init("dist", dist(88).1)?;
            self.null(json!({"Op":"bs-chain-add","Chain":chain,"Sampler":dist}))?;
            let rows = if label == 0 { 1 } else { 2 };
            let c = self.context(
                &[(0, chain)],
                Some(1),
                rows,
                0,
                if label == 0 { 4 } else { 0 },
            )?;
            let entries = (0..rows)
                .map(|p| Ok((0, self.bos, i32::try_from(p)?)))
                .collect::<Result<Vec<_>>>()?;
            let code = self.raw_batch(&c, &entries, usize::try_from(rows)?)?;
            self.check(code == 0, "CPU suffix decode failed")?;
            let observed = self
                .observers
                .lock()
                .map_err(|_| invalid("observer lock poisoned"))?
                .get(&label)
                .cloned()
                .unwrap_or_default();
            if label == 0 {
                self.check(
                    observed.initialized,
                    "single-output observer not backend initialized",
                )?;
                self.check(
                    observed.outputs == 1,
                    "single-output observer max output differs",
                )?;
                self.check(
                    observed.backend_applies > 0,
                    "single-output observer backend apply absent",
                )?;
                self.check(
                    observed.applies == 0,
                    "single-output observer unexpectedly CPU applied",
                )?;
                let t = self.sampled(&c, 0)?;
                self.check(t != -1, "single-output backend token absent")?;
            } else {
                self.check(
                    !observed.initialized,
                    "multi-output observer unexpectedly backend initialized",
                )?;
                self.check(
                    observed.outputs == 2,
                    "multi-output observer max output differs",
                )?;
                self.check(
                    observed.backend_applies == 0,
                    "multi-output observer backend applied",
                )?;
                for row in 0..2 {
                    let o = self.observe(&c, row, false, true, false, true, true)?;
                    self.check(o.token == Some(-1), "CPU suffix sampled prematurely")?;
                    self.check(
                        o.logits.as_ref().map(Vec::len) == Some(8),
                        "CPU suffix logits count differs",
                    )?;
                    self.check(
                        o.candidates.as_ref().map(Vec::len) == Some(8),
                        "CPU suffix candidate count differs",
                    )?;
                    let t = self.sample(chain, &c, row)?;
                    self.valid(t)?;
                }
                let count = self
                    .observers
                    .lock()
                    .map_err(|_| invalid("observer lock poisoned"))?
                    .get(&label)
                    .cloned()
                    .unwrap_or_default()
                    .applies;
                self.check(
                    count == 2,
                    "CPU observer apply count differs from batch length",
                )?;
            }
            self.release("context", c.id)?;
            self.free_sampler(chain)?;
        }
        Ok(())
    }
}
/// Original --model/--test/--device controls; the runner supplies external assets.
#[derive(Default, Deserialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
pub(super) struct Options {
    pub(super) model: Option<PathBuf>,
    pub(super) test: Option<String>,
    pub(super) device: Option<String>,
}
fn callback_state(observers: Arc<Mutex<BTreeMap<u32, Observer>>>) -> Arc<SamplerEvent> {
    Arc::new(move |event: &Value| -> Result<Value> {
        let kind = field(event, "Kind")?
            .as_str()
            .ok_or_else(|| invalid("observer kind text required"))?;
        let label = u32::try_from(uint(event, "Sampler")?)?;
        let mut values = observers
            .lock()
            .map_err(|_| invalid("observer state lock poisoned"))?;
        let value = values.entry(label).or_default();
        let result = match kind {
            "backend-init" => {
                exact_fields(
                    event,
                    &["Kind", "Sampler", "OutputsMaxPerSequence", "BufferTypeName"],
                )?;
                field(event, "BufferTypeName")?
                    .as_str()
                    .ok_or_else(|| invalid("buffer type name text required"))?;
                value.outputs = u32::try_from(uint(event, "OutputsMaxPerSequence")?)?;
                let accept = value.outputs <= 1;
                if accept {
                    value.initialized = true;
                }
                Ok(json!({"Accept":accept}))
            }
            "apply" | "backend-apply" => {
                exact_fields(event, &["Kind", "Sampler"])?;
                if kind == "apply" {
                    value.applies += 1;
                } else {
                    value.backend_applies += 1;
                }
                Ok(json!({}))
            }
            _ => Err(invalid("unknown sampler observer event")),
        };
        drop(values);
        result
    })
}
pub(super) fn run(engine: &Engine) -> Result<Value> {
    run_with_options(engine, &Options::default())
}
pub(super) fn diagnostic(engine: &Engine, value: &Value) -> Result<Value> {
    exact_fields(value, &["Kind", "Options"])?;
    run_with_options(engine, &serde_json::from_value(value["Options"].clone())?)
}
#[allow(
    clippy::too_many_lines,
    reason = "The bounded session initialization and full source-case dispatch share one explicit cleanup scope."
)]
pub(super) fn run_with_options(engine: &Engine, options: &Options) -> Result<Value> {
    let device = options.device.as_deref().unwrap_or("auto");
    verify(
        matches!(device, "auto" | "cpu" | "gpu"),
        "backend sampler device must be auto, cpu or gpu",
    )?;
    if let Some(name) = &options.test {
        verify(
            NAMES.contains(&name.as_str()),
            "unknown backend sampler test selection",
        )?;
    }
    let (path, model_input) = if let Some(path) = &options.model {
        verify(
            path.is_absolute() && path.is_file(),
            "backend sampler model must be an absolute regular file",
        )?;
        (path.clone(), json!({"Kind":"CallerSelectedModel"}))
    } else {
        input::create(
            &engine.asset_path("tinyllamas/stories15M-q4_0.gguf")?,
            &engine.evidence_path("backend-sampler-input")?,
        )?
    };
    let Some(session) = ModelSession::open(engine, "backend-sampler")? else {
        let mut output = result(
            ID,
            false,
            "production model/context/backend sampling APIs are unavailable".into(),
        );
        output["ModelInput"] = model_input;
        output["Model"] = json!(path);
        return Ok(output);
    };
    let observers = Arc::new(Mutex::new(BTreeMap::new()));
    let events = callback_state(Arc::clone(&observers));
    let client = session.client();
    let mut run = Run {
        client,
        events,
        observers,
        directory: session.evidence_directory().to_path_buf(),
        serial: 0,
        model: 0,
        n: 0,
        bos: -1,
        eos: -1,
        checks: 0,
        diagnostics: Vec::new(),
    };
    run.null(json!({"Op":"backend-init"}))?;
    let mut overrides = json!({});
    if device != "auto" {
        let v = run.call(json!({"Op":"devices"}))?;
        exact_fields(&v, &["Devices"])?;
        let want = i64::from(device != "cpu");
        let selected = array(&v, "Devices")?
            .iter()
            .find(|v| v["Type"].as_i64() == Some(want))
            .ok_or_else(|| invalid(format!("requested {device} backend is unavailable")))?;
        overrides["Devices"] = json!([uint(selected, "Index")?]);
        overrides["GpuLayers"] = json!(if device == "gpu" { 999 } else { 0 });
    }
    run.model=run.handle(json!({"Op":"model-load","Path":path,"Base":"default","Params":null,"Overrides":overrides,"Progress":false}),"Model")?;
    let v = run.call(json!({"Op":"model-vocabulary","Model":run.model}))?;
    exact_fields(&v, &["Count", "Bos", "Eos", "Type", "EogTokenIds"])?;
    run.n = usize::try_from(uint(&v, "Count")?)?;
    run.bos = signed(&v, "Bos")?;
    run.eos = signed(&v, "Eos")?;
    verify(
        run.n > 0 && run.n <= 1_048_576,
        "backend sampler vocabulary out of bounds",
    )?;
    let mut completed = Vec::new();
    let mut error = None;
    for name in NAMES {
        if options
            .test
            .as_deref()
            .is_some_and(|selected| selected != name)
        {
            continue;
        }
        let before = run.checks;
        let outcome = match name {
            "penalties" => run.penalty_cases(),
            "multi_output_limit" => run.output_limit(),
            "multi_sequence_multi_output_dist" => run.multi_dist(),
            "multi_output_dist_transaction" => run.transaction(),
            "multi_output_sampling_chain" => run.sampling_chain(),
            "multi_output_cpu" => run.cpu_suffix(),
            _ => run.basic(name),
        };
        match outcome {
            Ok(()) => completed.push(json!({"Name":name,"SourcePredicates":run.checks-before})),
            Err(e) => {
                error = Some(format!("{name}: {e}"));
                break;
            }
        }
    }
    // All explicit source lifetimes end before its model is released. An error
    // leaves scoped native ownership to the bounded session's normal teardown.
    if error.is_none() {
        run.release("model", run.model)?;
        run.null(json!({"Op":"backend-free"}))?;
    }
    let checks = run.checks;
    let diagnostics = run.diagnostics;
    session.finish()?;
    let expected = if options.test.is_some() { 1 } else { 19 };
    let passed = error.is_none() && completed.len() == expected;
    let mut output = result(
        ID,
        passed,
        format!(
            "{}/{} original scenarios, {checks} dynamic source predicates; error={error:?}",
            completed.len(),
            expected
        ),
    );
    output["Cases"] = json!(completed);
    output["Diagnostics"] = json!(diagnostics);
    output["Device"] = json!(device);
    output["SourcePredicates"] = json!(checks);
    output["SourceSha256"] = json!(SOURCE_SHA);
    output["DefaultScenarios"] = json!(19);
    output["Model"] = json!(path);
    output["ModelInput"] = model_input;
    Ok(output)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn observer_and_comparison_guards_are_rust_owned() -> Result<()> {
        let values = Arc::new(Mutex::new(BTreeMap::new()));
        let callback = callback_state(Arc::clone(&values));
        let v = callback(
            &json!({"Kind":"backend-init","Sampler":7,"OutputsMaxPerSequence":2,"BufferTypeName":"CPU"}),
        )?;
        assert_eq!(v, json!({"Accept":false}));
        callback(&json!({"Kind":"apply","Sampler":7}))?;
        assert_eq!(
            values.lock().map_err(|_| invalid("lock poisoned"))?[&7].applies,
            1
        );
        assert!(callback(&json!({"Kind":"apply","Sampler":7,"Unexpected":true})).is_err());
        let expected = BTreeMap::from([(3, 1.0_f32)]);
        assert_eq!(compare(&expected, &[1.0], &[3], false)?.mismatch, 0);
        assert_eq!(compare(&expected, &[f32::NAN], &[3], false)?.mismatch, 1);
        assert_eq!(compare(&expected, &[1.0, 1.0], &[3, 3], false)?.mismatch, 1);
        assert!(compare(&expected, &[1.0], &[], false).is_err());
        Ok(())
    }
}
