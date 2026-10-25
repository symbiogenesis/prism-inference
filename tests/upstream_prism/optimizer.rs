//! Complete Windows CPU source contracts from pinned `tests/test-opt.cpp`.
//!
//! Rust owns all dataset bytes, graph descriptors, training/reset schedules,
//! regression RNG, parameter schedules and comparisons. Native code calls only
//! production optimizer primitives. Upstream MIT attribution: tools/prism/upstream/LICENSE.

#![allow(
    clippy::float_cmp,
    clippy::suboptimal_flops,
    clippy::too_many_lines,
    reason = "Exact comparisons, separate float operations and complete source scenarios preserve the upstream contracts."
)]

use super::engine::Engine;
use super::llama_grammar::{array, exact_fields, field, uint};
use super::windows_random::{Normal, Random};
use super::{Result, invalid, result};
use serde_json::{Value, json};

#[path = "optimizer_runtime.rs"]
mod runtime;
pub(super) use runtime::execute_runtime;

const ID: &str = "tests/test-opt.cpp";
const SOURCE_SHA: &str = "24af6c26293d328ad42820cc817ab891869c93ec744950e6505018cfd33d04bb";
const NDATA: usize = 6;
const SOURCE_SUBTESTS: [usize; 2] = [118, 46];

fn boolean(value: &Value, name: &str) -> Result<bool> {
    field(value, name)?
        .as_bool()
        .ok_or_else(|| invalid(format!("optimizer {name} must be boolean")))
}
fn f32_of(value: usize) -> Result<f32> {
    Ok(f32::from(u16::try_from(value)?))
}
fn f64_of(value: usize) -> Result<f64> {
    Ok(f64::from(u32::try_from(value)?))
}
fn almost(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() < tolerance
}
fn float_hex(values: &[f32]) -> String {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .flat_map(|byte| [byte >> 4, byte & 15])
        .map(|nibble| char::from(b"0123456789abcdef"[usize::from(nibble)]))
        .collect()
}
fn floats(value: &Value) -> Result<Vec<f32>> {
    let hex = field(value, "DataHex")?
        .as_str()
        .ok_or_else(|| invalid("optimizer tensor data must be hex"))?;
    if hex.len() % 8 != 0
        || hex.len() > 2 * 1024 * 1024
        || !hex
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(invalid("optimizer tensor byte encoding invalid"));
    }
    let bytes = hex
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| {
            let digit = |byte: u8| {
                if byte <= b'9' {
                    byte - b'0'
                } else {
                    byte - b'a' + 10
                }
            };
            (digit(pair[0]) << 4) | digit(pair[1])
        })
        .collect::<Vec<_>>();
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| Ok(f32::from_le_bytes(*bytes)))
        .collect()
}
fn scalar(value: &Value) -> Result<f64> {
    let values = floats(value)?;
    if values.len() != 1 {
        return Err(invalid("optimizer scalar extent mismatch"));
    }
    Ok(f64::from(values[0]))
}
fn double(value: &Value, name: &str) -> Result<f64> {
    Ok(f64::from_bits(uint(value, name)?))
}

fn exchange(engine: &Engine, operation: &Value) -> Result<Option<Value>> {
    let Some(response) = engine.execute_common(
        "optimizer",
        &json!({"SchemaVersion":1,"Operations":[operation]}),
    )?
    else {
        return Ok(None);
    };
    exact_fields(&response, &["SchemaVersion", "Results"])?;
    let values = array(&response, "Results")?;
    if uint(&response, "SchemaVersion")? != 1 || values.len() != 1 {
        return Err(invalid("optimizer response framing invalid"));
    }
    Ok(Some(values[0].clone()))
}

struct Trace {
    actions: Vec<Value>,
}
impl Trace {
    const fn new() -> Self {
        Self {
            actions: Vec::new(),
        }
    }
    fn add(&mut self, action: Value) -> usize {
        let index = self.actions.len();
        self.actions.push(action);
        index
    }
    fn tensor(&mut self, id: usize, context: usize, dims: &[usize], parameter: bool, name: &str) {
        self.add(json!({"Op":"tensor","Handle":id,"Context":context,"Dimensions":dims,"Parameter":parameter,"Name":name}));
    }
    fn node(&mut self, id: usize, context: usize, op: &str, inputs: &[usize], name: &str) {
        self.add(json!({"Op":"node","Handle":id,"Context":context,"Operation":op,"Inputs":inputs,"FactorBits":if op=="scale"{json!(1_f32.to_bits())}else{Value::Null},"Name":name}));
    }
    fn set(&mut self, tensor: usize, values: &[f32]) {
        self.add(json!({"Op":"set","Tensor":tensor,"DataHex":float_hex(values)}));
    }
    fn get(&mut self, tensor: usize) -> usize {
        self.add(json!({"Op":"get","Tensor":tensor}))
    }
    fn alloc(&mut self, backward: bool) {
        self.add(json!({"Op":"alloc","Optimizer":0,"Value":backward}));
    }
    fn eval(&mut self, result: usize) {
        self.add(json!({"Op":"eval","Optimizer":0,"Result":result}));
    }
    fn snapshot(&mut self, result: usize, uncertainty: bool) -> usize {
        self.add(json!({"Op":"result-snapshot","Result":result,"LossUncertainty":uncertainty,"AccuracyUncertainty":true}))
    }
    fn epoch(&mut self, train: Option<usize>, evaluate: Option<usize>, split: usize) {
        self.add(json!({"Op":"epoch","Optimizer":0,"Dataset":6,"Train":train,"Evaluate":evaluate,"Split":split}));
    }
    fn shuffle(&mut self, dataset: usize) {
        self.add(json!({"Op":"shuffle","Optimizer":0,"Dataset":dataset,"Count":-1}));
    }
    fn reset_result(&mut self, result: usize) {
        self.add(json!({"Op":"result-reset","Result":result}));
    }
    fn execute(&self, engine: &Engine, info: &Value, backend: usize) -> Result<Vec<Value>> {
        let devices = array(info, "Devices")?;
        let ids = devices
            .iter()
            .map(|v| uint(v, "Index"))
            .collect::<Result<Vec<_>>>()?;
        let order = std::iter::once(backend)
            .chain(0..devices.len())
            .collect::<Vec<_>>();
        let response=exchange(engine,&json!({"Kind":"optimizer-trace","Devices":ids,"Scheduler":order,"Backend":backend,"Threads":uint(info,"HardwareConcurrency")?/2,"Actions":self.actions}))?.ok_or_else(||invalid("optimizer API disappeared"))?;
        exact_fields(&response, &["Trace"])?;
        let observations = array(&response, "Trace")?;
        if observations.len() != self.actions.len() {
            return Err(invalid("optimizer action/result extent mismatch"));
        }
        for (action, event) in self.actions.iter().zip(observations) {
            let op = action["Op"]
                .as_str()
                .ok_or_else(|| invalid("optimizer action Op missing"))?;
            match op {
                "context" => {
                    exact_fields(event, &["Created"])?;
                    if !boolean(event, "Created")? {
                        return Err(invalid("production optimizer context creation failed"));
                    }
                }
                "dataset" => {
                    exact_fields(event, &["Count"])?;
                    if event["Count"] != action["Count"] {
                        return Err(invalid("production dataset count mismatch"));
                    }
                }
                "params" => {
                    exact_fields(event, &["HasCallback"])?;
                    if !boolean(event, "HasCallback")? {
                        return Err(invalid("source optimizer parameter callback guard failed"));
                    }
                }
                "init" => {
                    exact_fields(event, &["Created", "Optimizer", "HasCallback"])?;
                    if !boolean(event, "Created")?
                        || !boolean(event, "HasCallback")?
                        || event["Optimizer"] != action["Optimizer"]
                    {
                        return Err(invalid("source optimizer initialization/type guard failed"));
                    }
                }
                "get" => {
                    exact_fields(event, &["DataHex"])?;
                    floats(event)?;
                }
                "supports" => {
                    exact_fields(event, &["Supported"])?;
                    boolean(event, "Supported")?;
                }
                "result-snapshot" => {
                    exact_fields(
                        event,
                        &[
                            "Count",
                            "LossBits",
                            "LossUncertaintyBits",
                            "AccuracyBits",
                            "AccuracyUncertaintyBits",
                        ],
                    )?;
                    uint(event, "Count")?;
                    double(event, "LossBits")?;
                    double(event, "AccuracyBits")?;
                    double(event, "AccuracyUncertaintyBits")?;
                    if action["LossUncertainty"] == true {
                        double(event, "LossUncertaintyBits")?;
                    } else if !event["LossUncertaintyBits"].is_null() {
                        return Err(invalid("unexpected optimizer uncertainty observation"));
                    }
                }
                _ => exact_fields(event, &[])?,
            }
        }
        Ok(observations.to_vec())
    }
}

fn custom_parameters(info: &Value) -> Result<Value> {
    let mut p = field(info, "Defaults")?.clone();
    exact_fields(&p, &["AdamW", "Sgd"])?;
    if array(&p, "AdamW")?.len() != 5 || array(&p, "Sgd")?.len() != 2 {
        return Err(invalid("optimizer defaults extent mismatch"));
    }
    p["AdamW"] = json!([1_f32.to_bits(), 0, 0, 0, 0]);
    p["Sgd"] = json!([1_f32.to_bits(), 0]);
    Ok(p)
}

struct Helper {
    trace: Trace,
    initialized: bool,
}
impl Helper {
    fn new(
        info: &Value,
        optim: usize,
        initialize: bool,
        defaults: bool,
        logical: usize,
        physical: usize,
        loss: usize,
    ) -> Result<Self> {
        if !logical.is_multiple_of(physical) {
            return Err(invalid("source logical batch divisibility guard failed"));
        }
        let mut t = Trace::new();
        for shard in 1..=NDATA {
            let data = (0..NDATA)
                .flat_map(|row| (0..2).map(move |i| u16::try_from(16 * row + i).map(f32::from)))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let labels = (0..NDATA)
                .map(|row| u16::try_from(16 * 16 * row).map(f32::from))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            t.add(json!({"Op":"dataset","Handle":shard-1,"DatapointElements":2,"LabelElements":1,"Count":NDATA,"Shard":shard,"DataHex":float_hex(&data),"LabelsHex":float_hex(&labels)}));
        }
        let data = (0..NDATA).map(f32_of).collect::<Result<Vec<_>>>()?;
        t.add(json!({"Op":"dataset","Handle":6,"DatapointElements":1,"LabelElements":0,"Count":NDATA,"Shard":1,"DataHex":float_hex(&data),"LabelsHex":""}));
        t.add(json!({"Op":"context","Handle":0,"Bytes":(2*NDATA+2)*usize::try_from(uint(info,"TensorOverhead")?)?}));
        t.add(json!({"Op":"context","Handle":1,"Bytes":uint(info,"DefaultGraphSize")?*uint(info,"TensorOverhead")?+3*uint(info,"GraphOverhead")?}));
        for batch in 1..=NDATA {
            t.tensor(batch - 1, 0, &[batch * 2], false, "");
            t.tensor(6 + batch - 1, 0, &[batch], false, "");
        }
        t.tensor(12, 0, &[physical], false, "inputs");
        t.tensor(13, 0, &[1], true, "weights");
        t.node(14, 1, "add", &[12, 13], "");
        t.node(15, 1, "scale", &[14], "outputs");
        t.add(json!({"Op":"allocate","Handle":0,"Context":0}));
        t.set(13, &[3.0]);
        t.add(json!({"Op":"params","Loss":loss}));
        if initialize {
            t.add(json!({"Op":"init","Handle":0,"Context":1,"Inputs":12,"Outputs":15,"Loss":loss,"Optimizer":optim,"Period":logical/physical,"Parameters":if defaults{Value::Null}else{custom_parameters(info)?}}));
        }
        t.add(json!({"Op":"result","Handle":0}));
        t.add(json!({"Op":"result","Handle":1}));
        Ok(Self {
            trace: t,
            initialized: initialize,
        })
    }
    fn loss(&mut self) {
        self.trace
            .add(json!({"Op":"loss","Optimizer":0,"Handle":16}));
    }
    fn gradient(&mut self) {
        self.trace
            .add(json!({"Op":"gradient","Optimizer":0,"Handle":17,"Tensor":13}));
    }
    fn finish(&mut self) {
        for id in 0..2 {
            self.trace
                .add(json!({"Op":"free","Type":"result","Handle":id}));
        }
        if self.initialized {
            self.trace
                .add(json!({"Op":"free","Type":"optimizer","Handle":0}));
        }
        self.trace
            .add(json!({"Op":"free","Type":"buffer","Handle":0}));
        for id in 0..2 {
            self.trace
                .add(json!({"Op":"free","Type":"context","Handle":id}));
        }
        for id in 0..7 {
            self.trace
                .add(json!({"Op":"free","Type":"dataset","Handle":id}));
        }
    }
}

#[derive(Default)]
struct Checks {
    count: usize,
    failures: Vec<String>,
}
impl Checks {
    fn check(&mut self, name: impl Into<String>, passed: bool) {
        self.count += 1;
        if !passed {
            self.failures.push(name.into());
        }
    }
}
fn statistics(
    value: &Value,
    count: usize,
    loss: f64,
    uncertainty: Option<f64>,
    tolerance: f64,
) -> Result<bool> {
    Ok(uint(value, "Count")? == u64::try_from(count)?
        && almost(double(value, "LossBits")?, loss, tolerance)
        && uncertainty.is_none_or(|expected| {
            double(value, "LossUncertaintyBits")
                .is_ok_and(|actual| almost(actual, expected, tolerance))
        })
        && double(value, "AccuracyBits")?.is_nan()
        && double(value, "AccuracyUncertaintyBits")?.is_nan())
}

fn dataset(
    engine: &Engine,
    info: &Value,
    backend: usize,
    optim: usize,
    shuffle: bool,
    checks: &mut Checks,
) -> Result<()> {
    let mut h = Helper::new(info, optim, true, true, 1, 1, 1)?;
    let mut cases = Vec::new();
    for shard in 1..=NDATA {
        if shuffle {
            h.trace.shuffle(shard - 1);
        }
        for batch in 1..=NDATA {
            if batch % shard != 0 {
                continue;
            }
            let mut outputs = Vec::new();
            for index in 0..NDATA / batch {
                h.trace.add(json!({"Op":"batch","Dataset":shard-1,"Data":batch-1,"Labels":6+batch-1,"Index":index}));
                outputs.push((h.trace.get(batch - 1), h.trace.get(6 + batch - 1)));
            }
            cases.push((shard, batch, outputs));
        }
    }
    h.finish();
    let observations = h.trace.execute(engine, info, backend)?;
    for (shard, batch, outputs) in cases {
        let mut okay = true;
        let mut shuffled = Vec::new();
        for (index, (data, labels)) in outputs.into_iter().enumerate() {
            let data = floats(&observations[data])?;
            let labels = floats(&observations[labels])?;
            if data.len() != 2 * batch || labels.len() != batch {
                return Err(invalid("dataset batch extent mismatch"));
            }
            for row in 0..batch {
                let found = (0..NDATA).find(|found| {
                    f32_of(16 * found).is_ok_and(|expected| data[row * 2] == expected)
                });
                let Some(found) = found else {
                    okay = false;
                    continue;
                };
                okay &= shuffle || found == index * batch + row;
                shuffled.push(found);
                for i in 0..2 {
                    okay &= data[row * 2 + i] == f32_of(16 * found + i)?;
                }
                okay &= labels[row] == f32_of(16 * 16 * found)?;
            }
        }
        if !shuffle || NDATA.is_multiple_of(batch) {
            for id in 0..(NDATA / batch) * batch {
                if !okay {
                    break;
                }
                okay &= shuffled.iter().filter(|&&value| value == id).count() == 1;
            }
        }
        checks.check(
            format!("optimizer{optim} dataset shuffle{shuffle} shard{shard} batch{batch}"),
            okay,
        );
    }
    Ok(())
}

fn gradient(
    engine: &Engine,
    info: &Value,
    backend: usize,
    optim: usize,
    checks: &mut Checks,
) -> Result<()> {
    let mut h = Helper::new(info, optim, true, false, 999_999, 1, 1)?;
    let mut reads = Vec::new();
    for input in 0..NDATA {
        h.trace.alloc(true);
        h.trace.set(12, &[f32_of(input)?]);
        h.trace.eval(0);
        h.gradient();
        reads.push(h.trace.get(17));
    }
    h.finish();
    let values = h.trace.execute(engine, info, backend)?;
    let mut okay = true;
    for (index, read) in reads.into_iter().enumerate() {
        okay &= scalar(&values[read])? == f64_of(index + 1)?;
    }
    checks.check(format!("optimizer{optim} accumulated gradient"), okay);
    Ok(())
}

fn forward_backward(
    engine: &Engine,
    info: &Value,
    backend: usize,
    optim: usize,
    high: bool,
    shuffle: bool,
    checks: &mut Checks,
) -> Result<()> {
    let mut h = Helper::new(info, optim, true, false, 1, 1, 1)?;
    h.loss();
    let initial = h.trace.snapshot(0, true);
    if high {
        if shuffle {
            h.trace.shuffle(6);
        }
        h.trace.epoch(None, Some(0), 0);
    } else {
        for input in 0..NDATA {
            h.trace.alloc(false);
            h.trace.set(12, &[f32_of(input)?]);
            h.trace.eval(0);
            h.trace.get(16);
        }
    }
    let forward_weights = h.trace.get(13);
    let forward_result = h.trace.snapshot(0, true);
    h.trace.add(json!({"Op":"save","Tensor":13,"Handle":0}));
    for _ in 0..10 {
        h.trace.alloc(true);
        h.trace.eval(0);
    }
    h.trace.add(json!({"Op":"restore","Tensor":13,"Handle":0}));
    h.trace
        .add(json!({"Op":"reset","Optimizer":0,"Value":false}));
    h.trace.reset_result(0);
    if high {
        if shuffle {
            h.trace.shuffle(6);
        }
        h.trace.epoch(Some(0), None, NDATA);
    } else {
        for input in 0..NDATA {
            h.trace.alloc(true);
            h.trace.set(12, &[f32_of(input)?]);
            h.trace.eval(0);
            h.trace.get(16);
        }
    }
    let backward_weights = h.trace.get(13);
    let backward_result = h.trace.snapshot(0, true);
    h.finish();
    let v = h.trace.execute(engine, info, backend)?;
    let prefix = format!("optimizer{optim} forward/backward high{high} shuffle{shuffle}");
    let initial = &v[initial];
    checks.check(
        format!("{prefix} initial"),
        uint(initial, "Count")? == 0
            && almost(double(initial, "LossBits")?, 0.0, 1e-6)
            && double(initial, "LossUncertaintyBits")?.is_nan()
            && double(initial, "AccuracyBits")?.is_nan()
            && double(initial, "AccuracyUncertaintyBits")?.is_nan(),
    );
    checks.check(
        format!("{prefix} forward weights"),
        almost(scalar(&v[forward_weights])?, 3.0, 1e-10),
    );
    checks.check(
        format!("{prefix} forward statistics"),
        statistics(&v[forward_result], 6, 33.0, Some(3.5_f64.sqrt()), 1e-10)?,
    );
    checks.check(
        format!("{prefix} backward weights"),
        almost(scalar(&v[backward_weights])?, -3.0, 1e-10),
    );
    checks.check(
        format!("{prefix} backward statistics"),
        statistics(&v[backward_result], 6, 18.0, None, 1e-10)?
            && (shuffle || double(&v[backward_result], "LossUncertaintyBits")? == 0.0),
    );
    Ok(())
}

fn epoch_fit(
    engine: &Engine,
    info: &Value,
    backend: usize,
    optim: usize,
    checks: &mut Checks,
) -> Result<()> {
    let mut epoch = Helper::new(info, optim, true, true, 1, 1, 1)?;
    epoch.trace.shuffle(6);
    epoch.trace.epoch(Some(0), None, NDATA);
    let weight_epoch = epoch.trace.get(13);
    epoch.finish();
    let epoch = epoch.trace.execute(engine, info, backend)?;
    let mut fit = Helper::new(info, optim, false, true, 1, 1, 1)?;
    fit.trace.add(json!({"Op":"fit","Context":1,"Inputs":12,"Outputs":15,"Dataset":6,"Loss":1,"Optimizer":optim,"Epochs":1,"LogicalBatch":1,"ValidationBits":0_f32.to_bits(),"Silent":true,"Parameters":null}));
    let weight_fit = fit.trace.get(13);
    fit.finish();
    let fit = fit.trace.execute(engine, info, backend)?;
    checks.check(
        format!("optimizer{optim} epoch-vs-fit"),
        scalar(&epoch[weight_epoch])? == scalar(&fit[weight_fit])?,
    );
    Ok(())
}

fn split(
    engine: &Engine,
    info: &Value,
    backend: usize,
    optim: usize,
    high: bool,
    checks: &mut Checks,
) -> Result<()> {
    let mut h = Helper::new(info, optim, true, false, 1, 1, 1)?;
    h.loss();
    let mut epochs = Vec::new();
    for _ in 1..=4 {
        if high {
            h.trace.epoch(Some(0), Some(1), 4);
        } else {
            for input in 0..NDATA {
                h.trace.alloc(input < 4);
                h.trace.set(12, &[f32_of(input)?]);
                h.trace.eval(usize::from(input >= 4));
                h.trace.get(16);
            }
        }
        if optim == 0 {
            epochs.push((
                h.trace.get(13),
                h.trace.snapshot(0, true),
                h.trace.snapshot(1, true),
            ));
        }
        h.trace.reset_result(0);
        h.trace.reset_result(1);
    }
    h.finish();
    let values = h.trace.execute(engine, info, backend)?;
    for (epoch, (weight, train, evaluate)) in (1..).zip(epochs) {
        let prefix = format!("optimizer{optim} split high{high} epoch{epoch}");
        let epoch = f64_of(epoch)?;
        checks.check(
            format!("{prefix} weights"),
            almost(scalar(&values[weight])?, 3.0 - epoch * 4.0, 1e-10),
        );
        checks.check(
            format!("{prefix} backward results"),
            statistics(&values[train], 4, 28.0 - epoch * 16.0, Some(0.0), 1e-10)?,
        );
        checks.check(
            format!("{prefix} forward results"),
            statistics(
                &values[evaluate],
                2,
                15.0 - epoch * 8.0,
                Some(0.5_f64.sqrt()),
                1e-10,
            )?,
        );
    }
    Ok(())
}

fn accumulation(
    engine: &Engine,
    info: &Value,
    backend: usize,
    physical: usize,
    loss: usize,
    checks: &mut Checks,
) -> Result<()> {
    let mut h = Helper::new(info, 0, true, false, 6, physical, loss)?;
    let mut epochs = Vec::new();
    for _ in 1..=4 {
        let mut gradients = Vec::new();
        for input in (0..NDATA).step_by(physical) {
            h.trace.alloc(true);
            h.trace.set(
                12,
                &(input..input + physical)
                    .map(f32_of)
                    .collect::<Result<Vec<_>>>()?,
            );
            h.trace.eval(0);
            h.gradient();
            gradients.push(h.trace.get(17));
        }
        epochs.push((gradients, h.trace.get(13), h.trace.snapshot(0, false)));
        h.trace.reset_result(0);
    }
    h.finish();
    let values = h.trace.execute(engine, info, backend)?;
    for (epoch, (gradients, weight, stats)) in (1..).zip(epochs) {
        let prefix = format!("accumulation physical{physical} loss{loss} epoch{epoch}");
        let mut history = Vec::new();
        for index in gradients {
            if physical == 2 {
                history.push(0.0);
            }
            history.push(scalar(&values[index])?);
        }
        if history.len() != NDATA {
            return Err(invalid("gradient history extent mismatch"));
        }
        let mut okay = true;
        for (index, &actual) in history.iter().enumerate() {
            let expected = if physical == 2 && index % 2 == 0 {
                0.0
            } else {
                f64_of(index + 1)?
            };
            let expected = if loss == 0 { expected / 6.0 } else { expected };
            okay &= almost(actual, expected, 1e-6);
        }
        checks.check(format!("{prefix} gradients"), okay);
        checks.check(
            format!("{prefix} weight"),
            almost(scalar(&values[weight])?, 3.0 - f64_of(epoch)?, 1e-6),
        );
        let expected = 39.0 - f64_of(epoch)? * 6.0;
        let expected = if loss == 0 { expected / 6.0 } else { expected };
        checks.check(
            format!("{prefix} statistics"),
            statistics(&values[stats], NDATA / physical, expected, None, 1e-6)?,
        );
    }
    Ok(())
}

fn regression(
    engine: &Engine,
    info: &Value,
    backend: usize,
    optim: usize,
    checks: &mut Checks,
) -> Result<Value> {
    let mut random = Random::new(12345);
    let mut normal = Normal::new(0.0, 0.1);
    let mut data = Vec::new();
    let mut labels = Vec::new();
    for id in 0..201 {
        let x = -100.0 + 200.0 * f32_of(id)? / 200.0;
        data.push(x);
        labels.push(1.2 * x + 3.4 + normal.sample(&mut random));
    }
    let mut t = Trace::new();
    t.add(json!({"Op":"dataset","Handle":0,"DatapointElements":1,"LabelElements":1,"Count":201,"Shard":201,"DataHex":float_hex(&data),"LabelsHex":float_hex(&labels)}));
    t.add(json!({"Op":"context","Handle":0,"Bytes":3*uint(info,"TensorOverhead")?}));
    t.add(json!({"Op":"context","Handle":1,"Bytes":uint(info,"DefaultGraphSize")?*uint(info,"TensorOverhead")?+3*uint(info,"GraphOverhead")?}));
    t.tensor(0, 0, &[1, 201], false, "x");
    t.tensor(1, 0, &[1], true, "a");
    t.tensor(2, 0, &[1], true, "b");
    t.node(3, 1, "mul", &[0, 1], "");
    t.node(4, 1, "add", &[3, 2], "f");
    t.add(json!({"Op":"allocate","Handle":0,"Context":0}));
    t.set(1, &[1.0]);
    t.set(2, &[3.0]);
    let epochs = if optim == 0 { 100 } else { 900 };
    let mut schedule = Vec::new();
    for epoch in 1..=epochs {
        let mut p = field(info, "Defaults")?.clone();
        p["AdamW"][0] = json!(0.1_f32.to_bits());
        p["Sgd"][0] = json!(regression_rate(epoch)?.to_bits());
        p["Sgd"][1] = json!(1e-10_f32.to_bits());
        schedule.push(p);
    }
    t.add(json!({"Op":"fit","Context":1,"Inputs":0,"Outputs":4,"Dataset":0,"Loss":3,"Optimizer":optim,"Epochs":epochs,"LogicalBatch":201,"ValidationBits":0_f32.to_bits(),"Silent":true,"Parameters":schedule}));
    let a = t.get(1);
    let b = t.get(2);
    t.add(json!({"Op":"free","Type":"buffer","Handle":0}));
    t.add(json!({"Op":"free","Type":"context","Handle":0}));
    t.add(json!({"Op":"free","Type":"context","Handle":1}));
    t.add(json!({"Op":"free","Type":"dataset","Handle":0}));
    let values = t.execute(engine, info, backend)?;
    let a = scalar(&values[a])?;
    let b = scalar(&values[b])?;
    let tolerance = if optim == 0 {
        f64::from(1e-2_f32)
    } else {
        f64::from(5e-2_f32)
    };
    let tolerance_met =
        almost(a, f64::from(1.2_f32), tolerance) && almost(b, f64::from(3.4_f32), tolerance);
    // The pinned source explicitly prints true for SGD, even when aok/bok fail.
    checks.check(
        format!("optimizer{optim} regression"),
        optim == 1 || tolerance_met,
    );
    Ok(
        json!({"Optimizer":optim,"Epochs":epochs,"AFitBits":a.to_bits(),"BFitBits":b.to_bits(),"ToleranceMet":tolerance_met,"SourceEnforcesTolerance":optim==0}),
    )
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "The source computes the scheduled learning rate in double precision then assigns its float32 parameter."
)]
fn regression_rate(epoch: usize) -> Result<f32> {
    Ok((f64::from(1e-4_f32) * 0.99_f64.powf(1000.0 * f64_of(epoch)? / 900.0)) as f32)
}

fn supported(engine: &Engine, info: &Value, backend: usize, optim: usize) -> Result<bool> {
    let mut t = Trace::new();
    t.add(json!({"Op":"context","Handle":0,"Bytes":6*uint(info,"TensorOverhead")?}));
    for id in 0..4 {
        t.tensor(id, 0, &[1], id == 0, "");
    }
    t.tensor(4, 0, &[if optim == 0 { 7 } else { 2 }], false, "");
    t.node(
        5,
        0,
        if optim == 0 { "adamw-step" } else { "sgd-step" },
        if optim == 0 {
            &[0, 1, 2, 3, 4]
        } else {
            &[0, 1, 4]
        },
        "",
    );
    let index = t.add(json!({"Op":"supports","Tensor":5}));
    t.add(json!({"Op":"free","Type":"context","Handle":0}));
    let values = t.execute(engine, info, backend)?;
    boolean(&values[index], "Supported")
}

pub(super) fn run(engine: &Engine) -> Result<Value> {
    let Some(info) = exchange(engine, &json!({"Kind":"optimizer-info"}))? else {
        return Ok(result(
            ID,
            false,
            "unsupported production optimizer/dataset APIs; all164 original CPU subtests required"
                .into(),
        ));
    };
    exact_fields(
        &info,
        &[
            "Devices",
            "HardwareConcurrency",
            "Defaults",
            "TensorOverhead",
            "GraphOverhead",
            "DefaultGraphSize",
        ],
    )?;
    let devices = array(&info, "Devices")?;
    if devices.is_empty() || devices.len() > 8 || uint(&info, "HardwareConcurrency")? > 256 {
        return Err(invalid("optimizer device/thread inventory outside bound"));
    }
    for device in devices {
        exact_fields(device, &["Index", "Name", "Type"])?;
        if uint(device, "Type")? != 0 {
            return Err(invalid(
                "Windows CPU-only optimizer fixture unexpectedly exposes non-CPU device",
            ));
        }
    }
    custom_parameters(&info)?;
    let mut checks = Checks::default();
    let mut diagnostics = Vec::new();
    let mut skipped = Vec::new();
    let mut expected = 0;
    for (optim, &source_subtests) in SOURCE_SUBTESTS.iter().enumerate() {
        for backend in 0..devices.len() {
            if !supported(engine, &info, backend, optim)? {
                skipped.push(json!({"Backend":backend,"Optimizer":optim}));
                continue;
            }
            expected += source_subtests;
            for shuffle in [false, true] {
                dataset(engine, &info, backend, optim, shuffle, &mut checks)?;
            }
            gradient(engine, &info, backend, optim, &mut checks)?;
            for (high, shuffle) in [(false, false), (true, false), (true, true)] {
                forward_backward(engine, &info, backend, optim, high, shuffle, &mut checks)?;
            }
            epoch_fit(engine, &info, backend, optim, &mut checks)?;
            for high in [false, true] {
                split(engine, &info, backend, optim, high, &mut checks)?;
            }
            if optim == 0 {
                for physical in [2, 1] {
                    for loss in [1, 0] {
                        accumulation(engine, &info, backend, physical, loss, &mut checks)?;
                    }
                }
            }
            diagnostics.push(regression(engine, &info, backend, optim, &mut checks)?);
        }
    }
    let passed = checks.count == expected && checks.failures.is_empty();
    let mut output = result(
        ID,
        passed,
        format!(
            "{}/{} original result subtests; {} device(s); source initialization/callback/type guards retained; SGD regression diagnostic is not enforced by source; failures={:?}",
            checks.count,
            expected,
            devices.len(),
            checks.failures
        ),
    );
    output["SourceSha256"] = json!(SOURCE_SHA);
    output["Regression"] = json!(diagnostics);
    output["UnsupportedOptimizerSkips"] = json!(skipped);
    Ok(output)
}
