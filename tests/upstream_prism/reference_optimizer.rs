//! Clean-room independent safe Rust optimizer, autodiff, and dataset reference provider.
//!
//! No production `bonsai_prism_runtime` code, native bridge or precomputed
//! answer fixture is called here. Implements bounded tensor/graph descriptors,
//! forward evaluation, reverse-mode autodiff, `AdamW`/`SGD` parameter steps,
//! Bessel-corrected result metrics, and an MSVC-compatible MT19937 dataset
//! shuffler directly from the protocol specification.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Value, json};

use super::{Result, invalid};

pub(super) const ID: &str = "tests/test-opt.cpp";
const SOURCE_SHA: &str = "24af6c26293d328ad42820cc817ab891869c93ec744950e6505018cfd33d04bb";

const TENSOR_OVERHEAD: usize = 336;
const GRAPH_OVERHEAD: usize = 4096;
const DEFAULT_GRAPH_SIZE: usize = 2048;

const MAX_OPERATIONS: usize = 128;
const MAX_DEVICES: usize = 8;
const MAX_SCHEDULER: usize = 16;
const MAX_THREADS: u64 = 256;
const MAX_ACTIONS: usize = 8192;
const MAX_HANDLE: u64 = 4095;
const MAX_CONTEXT_BYTES: u64 = 8 * 1024 * 1024;
const MAX_CONTEXTS: usize = 16;
const MAX_TENSORS: usize = 256;
const MAX_BUFFERS: usize = 16;
const MAX_SAVED: usize = 16;
const MAX_DATASETS: usize = 16;
const MAX_OPTIMIZERS: usize = 16;
const MAX_RESULTS: usize = 16;
const MAX_TENSOR_DIM: u64 = 65_536;
const MAX_TENSOR_ELEMENTS: usize = 262_144;
const MAX_PAYLOAD_BYTES: usize = 1024 * 1024;
const MAX_HEX_CHARS: usize = 2 * MAX_PAYLOAD_BYTES;
const MAX_NAME_LEN: usize = 128;
const MAX_DATASET_COUNT: usize = 4096;
const MAX_DATASET_COUNT_U64: u64 = 4096;
const MAX_DATASET_COUNT_I64: i64 = 4096;
const MAX_DATAPOINT_ELEMS: usize = 64;
const MAX_DATAPOINT_ELEMS_U64: u64 = 64;
const MAX_PERIOD: usize = 1_000_000;
const MAX_PERIOD_U64: u64 = 1_000_000;
const MAX_EPOCHS_U64: u64 = 4096;
const MT_SEED: u32 = 5489;

fn require(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(invalid(message))
    }
}

fn exact_fields(value: &Value, names: &[&str]) -> Result<()> {
    let obj = value
        .as_object()
        .ok_or_else(|| invalid("optimizer object expected"))?;
    require(
        obj.len() == names.len() && names.iter().all(|name| obj.contains_key(*name)),
        "optimizer fields mismatch",
    )
}

fn field<'a>(value: &'a Value, name: &str) -> Result<&'a Value> {
    value
        .as_object()
        .and_then(|obj| obj.get(name))
        .ok_or_else(|| invalid(format!("missing optimizer field {name}")))
}

fn array<'a>(value: &'a Value, name: &str) -> Result<&'a [Value]> {
    field(value, name)?
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| invalid(format!("optimizer {name} must be an array")))
}

fn text<'a>(value: &'a Value, name: &str) -> Result<&'a str> {
    field(value, name)?
        .as_str()
        .ok_or_else(|| invalid(format!("optimizer {name} must be a string")))
}

fn boolean(value: &Value, name: &str) -> Result<bool> {
    field(value, name)?
        .as_bool()
        .ok_or_else(|| invalid(format!("optimizer {name} must be boolean")))
}

fn uint_value(value: &Value, low: u64, high: u64) -> Result<u64> {
    let number = value
        .as_u64()
        .ok_or_else(|| invalid("optimizer nonnegative integer required"))?;
    require(
        number >= low && number <= high,
        "optimizer integer outside bound",
    )?;
    Ok(number)
}

fn uint_field(value: &Value, name: &str, low: u64, high: u64) -> Result<u64> {
    uint_value(field(value, name)?, low, high)
}

fn int_field(value: &Value, name: &str, low: i64, high: i64) -> Result<i64> {
    let number = field(value, name)?
        .as_i64()
        .ok_or_else(|| invalid("optimizer signed integer required"))?;
    require(
        number >= low && number <= high,
        "optimizer integer outside bound",
    )?;
    Ok(number)
}

fn handle_value(value: &Value) -> Result<u32> {
    Ok(u32::try_from(uint_value(value, 0, MAX_HANDLE)?)?)
}

fn handle(value: &Value, name: &str) -> Result<u32> {
    handle_value(field(value, name)?)
}

fn finite_f32_value(value: &Value) -> Result<f32> {
    let bits = u32::try_from(uint_value(value, 0, u64::from(u32::MAX))?)?;
    let number = f32::from_bits(bits);
    require(number.is_finite(), "optimizer float must be finite")?;
    Ok(number)
}

fn finite_f32_field(value: &Value, name: &str) -> Result<f32> {
    finite_f32_value(field(value, name)?)
}

fn unhex_bytes(value: &Value) -> Result<Vec<u8>> {
    let text = value
        .as_str()
        .ok_or_else(|| invalid("optimizer data must be hex"))?;
    require(
        text.len().is_multiple_of(2)
            && text.len() <= MAX_HEX_CHARS
            && text
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "optimizer hex payload is invalid",
    )?;
    Ok(hex::decode(text)?)
}

fn unhex_f32(value: &Value) -> Result<Vec<f32>> {
    let bytes = unhex_bytes(value)?;
    require(
        bytes.len().is_multiple_of(4),
        "optimizer F32 byte payload must align to 4 bytes",
    )?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}

fn hex_f32(values: &[f32]) -> Result<String> {
    let byte_len = values
        .len()
        .checked_mul(4)
        .filter(|&len| len <= MAX_PAYLOAD_BYTES)
        .ok_or_else(|| invalid("optimizer output bytes exceed bound"))?;
    let mut bytes = Vec::with_capacity(byte_len);
    for &value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(hex::encode(bytes))
}

#[derive(Clone)]
struct Mt19937 {
    state: [u32; 624],
    index: usize,
}

impl Mt19937 {
    fn new(seed: u32) -> Self {
        let mut state = [0_u32; 624];
        state[0] = seed;
        for (idx, inc) in (1..624).zip(1_u32..) {
            state[idx] = 1_812_433_253_u32
                .wrapping_mul(state[idx - 1] ^ (state[idx - 1] >> 30))
                .wrapping_add(inc);
        }
        Self { state, index: 624 }
    }

    fn next_u32(&mut self) -> u32 {
        if self.index == 624 {
            for idx in 0..624 {
                let y =
                    (self.state[idx] & 0x8000_0000) | (self.state[(idx + 1) % 624] & 0x7fff_ffff);
                self.state[idx] = self.state[(idx + 397) % 624]
                    ^ (y >> 1)
                    ^ if y & 1 == 0 { 0 } else { 0x9908_b0df };
            }
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    fn bounded(&mut self, bound: usize) -> Result<usize> {
        let bound_u32 = u32::try_from(bound)?;
        require(bound_u32 > 0, "optimizer random bound must be positive")?;
        let threshold = bound_u32.wrapping_neg() % bound_u32;
        loop {
            let product = u64::from(self.next_u32()) * u64::from(bound_u32);
            let low = u32::from_le_bytes(product.to_le_bytes()[..4].try_into()?);
            if low >= threshold {
                return Ok(usize::try_from(product >> 32)?);
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LossKind {
    Mean,
    Sum,
    CrossEntropy,
    MeanSquaredError,
}

impl LossKind {
    fn from_code(code: u64) -> Result<Self> {
        match code {
            0 => Ok(Self::Mean),
            1 => Ok(Self::Sum),
            2 => Ok(Self::CrossEntropy),
            3 => Ok(Self::MeanSquaredError),
            _ => Err(invalid("unknown optimizer loss kind")),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OptimKind {
    AdamW,
    Sgd,
}

impl OptimKind {
    fn from_code(code: u64) -> Result<Self> {
        match code {
            0 => Ok(Self::AdamW),
            1 => Ok(Self::Sgd),
            _ => Err(invalid("unknown optimizer algorithm")),
        }
    }
}

#[derive(Clone, Copy)]
struct OptimParams {
    adamw: [f32; 5],
    sgd: [f32; 2],
}

impl Default for OptimParams {
    fn default() -> Self {
        Self {
            adamw: [0.001, 0.9, 0.999, 1e-8, 0.0],
            sgd: [0.001, 0.0],
        }
    }
}

fn parse_parameters(value: &Value) -> Result<OptimParams> {
    exact_fields(value, &["AdamW", "Sgd"])?;
    let adam_raw = array(value, "AdamW")?;
    let sgd_raw = array(value, "Sgd")?;
    require(
        adam_raw.len() == 5 && sgd_raw.len() == 2,
        "optimizer parameter extent mismatch",
    )?;
    let mut adamw = [0.0_f32; 5];
    for (slot, raw) in adamw.iter_mut().zip(adam_raw) {
        *slot = finite_f32_value(raw)?;
    }
    let mut sgd = [0.0_f32; 2];
    for (slot, raw) in sgd.iter_mut().zip(sgd_raw) {
        *slot = finite_f32_value(raw)?;
    }
    Ok(OptimParams { adamw, sgd })
}

fn validate_optim_params(params: OptimParams, kind: OptimKind) -> Result<()> {
    if params
        .adamw
        .iter()
        .chain(&params.sgd)
        .any(|v| !v.is_finite())
    {
        return Err(invalid("optimizer parameters must be finite"));
    }
    let valid = match kind {
        OptimKind::AdamW => {
            let [alpha, beta1, beta2, eps, wd] = params.adamw;
            alpha > 0.0
                && (0.0..=1.0).contains(&beta1)
                && (0.0..=1.0).contains(&beta2)
                && eps >= 0.0
                && (0.0..=1.0).contains(&wd)
        }
        OptimKind::Sgd => params.sgd[0] > 0.0 && (0.0..=1.0).contains(&params.sgd[1]),
    };
    require(valid, "optimizer parameters violate bounds")
}

#[derive(Clone, Copy)]
enum NodeOp {
    Add,
    Multiply,
    Scale(f32),
    AdamWStep,
    SgdStep,
}

struct TensorObj {
    shape: [usize; 4],
    values: Vec<f32>,
    parameter: bool,
    op: Option<NodeOp>,
    inputs: Vec<usize>,
}

impl TensorObj {
    fn elements(&self) -> usize {
        self.shape.iter().product()
    }
}

struct TensorHandle {
    obj: usize,
    context: u32,
    optimizer: Option<u32>,
}

#[derive(Clone)]
struct Dataset {
    width: usize,
    label_width: usize,
    count: usize,
    shard: usize,
    data: Vec<f32>,
    labels: Vec<f32>,
    permutation: Vec<usize>,
}

impl Dataset {
    fn new(
        width: usize,
        label_width: usize,
        count: usize,
        shard: usize,
        data: Vec<f32>,
        labels: Vec<f32>,
    ) -> Result<Self> {
        require(
            (1..=MAX_DATAPOINT_ELEMS).contains(&width)
                && label_width <= MAX_DATAPOINT_ELEMS
                && (1..=MAX_DATASET_COUNT).contains(&count)
                && (1..=count).contains(&shard)
                && count.checked_mul(width) == Some(data.len())
                && count.checked_mul(label_width) == Some(labels.len()),
            "optimizer dataset geometry or payload mismatch",
        )?;
        Ok(Self {
            width,
            label_width,
            count,
            shard,
            data,
            labels,
            permutation: (0..count / shard).collect(),
        })
    }

    fn shuffle(&mut self, rng: &mut Mt19937, count: Option<usize>) -> Result<()> {
        let shards = if let Some(count) = count {
            require(
                count <= self.count && count.is_multiple_of(self.shard),
                "optimizer shuffle count invalid for shard size",
            )?;
            count / self.shard
        } else {
            self.permutation.len()
        };
        for idx in 1..shards {
            let selected = rng.bounded(idx + 1)?;
            self.permutation.swap(idx, selected);
        }
        Ok(())
    }

    fn batch(&self, count: usize, index: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        require(
            count > 0 && count.is_multiple_of(self.shard),
            "optimizer batch count must be a positive multiple of shard",
        )?;
        let shards = count / self.shard;
        let start = index
            .checked_mul(shards)
            .ok_or_else(|| invalid("optimizer batch start overflow"))?;
        let end = start
            .checked_add(shards)
            .filter(|&end| end <= self.permutation.len())
            .ok_or_else(|| invalid("optimizer batch exceeds dataset shards"))?;
        let mut data = Vec::with_capacity(count * self.width);
        let mut labels = Vec::with_capacity(count * self.label_width);
        for &src in &self.permutation[start..end] {
            let begin = src * self.shard;
            data.extend_from_slice(
                &self.data[begin * self.width..(begin + self.shard) * self.width],
            );
            labels.extend_from_slice(
                &self.labels[begin * self.label_width..(begin + self.shard) * self.label_width],
            );
        }
        Ok((data, labels))
    }
}

#[derive(Clone, Default)]
struct Metrics {
    count: usize,
    losses: Vec<f32>,
    correct: Option<usize>,
    period: usize,
    per_datapoint: bool,
}

impl Metrics {
    fn reset(&mut self) {
        *self = Self::default();
    }

    fn record(
        &mut self,
        count: usize,
        loss: f32,
        correct: Option<usize>,
        period: usize,
        per_datapoint: bool,
    ) -> Result<()> {
        if self.count == 0 {
            self.period = period;
            self.per_datapoint = per_datapoint;
            self.correct = Some(0);
        }
        require(
            self.period == period
                && self.per_datapoint == per_datapoint
                && self.count == count * self.losses.len(),
            "optimizer result mixes incompatible batch conventions",
        )?;
        self.count += count;
        self.losses.push(loss);
        self.correct = self.correct.zip(correct).map(|(acc, next)| acc + next);
        Ok(())
    }

    fn append(&mut self, other: &Self) -> Result<()> {
        if other.count == 0 {
            return Ok(());
        }
        if self.count == 0 {
            *self = other.clone();
            return Ok(());
        }
        require(
            self.period == other.period
                && self.per_datapoint == other.per_datapoint
                && self.count / self.losses.len() == other.count / other.losses.len(),
            "optimizer result append mixes incompatible batch conventions",
        )?;
        self.count += other.count;
        self.losses.extend_from_slice(&other.losses);
        self.correct = self
            .correct
            .zip(other.correct)
            .map(|(left, right)| left + right);
        Ok(())
    }

    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "Metrics replicate native binary32 period scaling and binary64 Bessel-corrected variance."
    )]
    fn loss(&self) -> (f64, f64) {
        if self.losses.is_empty() {
            return (0.0, f64::NAN);
        }
        let mut sum = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        for &loss in &self.losses {
            let scaled = if self.per_datapoint {
                loss * self.period as f32
            } else {
                loss
            };
            sum += f64::from(scaled);
            sum_sq += f64::from(scaled * scaled);
        }
        let batches = self.losses.len() as f64;
        let mean = sum / batches;
        let reported_loss = if self.per_datapoint { mean } else { sum };
        let mean_sq = mean * mean;
        let variance = (sum_sq / batches - mean_sq).max(0.0);
        let uncertainty = if self.losses.len() < 2 {
            f64::NAN
        } else if self.per_datapoint {
            (variance / (batches - 1.0)).sqrt()
        } else {
            (variance * batches / (batches - 1.0)).sqrt()
        };
        (reported_loss, uncertainty)
    }

    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "Accuracy ratios use IEEE binary64 division over bounded element counts."
    )]
    fn accuracy(&self) -> (f64, f64) {
        let accuracy = self
            .correct
            .map_or(f64::NAN, |correct| correct as f64 / self.count as f64);
        let uncertainty = if self.correct.is_some() && self.count >= 2 {
            (accuracy * (1.0 - accuracy) / (self.count - 1) as f64).sqrt()
        } else {
            f64::NAN
        };
        (accuracy, uncertainty)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum EvalMode {
    Forward,
    Gradient,
    Update,
}

struct TrainableSlot {
    param: usize,
    gradient: usize,
    moment: Option<usize>,
    variance: Option<usize>,
}

#[derive(Clone, Copy)]
struct TrainingSpec {
    context: u32,
    inputs: u32,
    outputs: u32,
    loss: LossKind,
    optim: OptimKind,
    period: usize,
    params: OptimParams,
}

struct OptimizerState {
    context: u32,
    inputs_obj: usize,
    inputs_ctx: u32,
    outputs_obj: usize,
    loss_kind: LossKind,
    optim_kind: OptimKind,
    period: usize,
    params: OptimParams,
    topology: Vec<usize>,
    trainable: Vec<TrainableSlot>,
    loss_obj: usize,
    labels_obj: Option<usize>,
    mode: EvalMode,
    ready: bool,
    iteration: usize,
    index: usize,
    rng: Mt19937,
}

const fn can_repeat(small: [usize; 4], large: [usize; 4]) -> bool {
    let mut i = 0;
    while i < 4 {
        if small[i] == 0 || !large[i].is_multiple_of(small[i]) {
            return false;
        }
        i += 1;
    }
    true
}

const fn repeated_index(mut index: usize, large: [usize; 4], small: [usize; 4]) -> usize {
    let mut stride = 1;
    let mut result = 0;
    let mut axis = 0;
    while axis < 4 {
        result += ((index % large[axis]) % small[axis]) * stride;
        index /= large[axis];
        stride *= small[axis];
        axis += 1;
    }
    result
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for idx in 1..values.len() {
        if values[idx] > values[best] {
            best = idx;
        }
    }
    best
}

struct TraceState {
    contexts: BTreeSet<u32>,
    unavailable_storage: BTreeSet<u32>,
    objects: Vec<TensorObj>,
    handles: BTreeMap<u32, TensorHandle>,
    buffers: BTreeMap<u32, u32>,
    datasets: BTreeMap<u32, Dataset>,
    optimizers: BTreeMap<u32, OptimizerState>,
    results: BTreeMap<u32, Metrics>,
    saved: BTreeMap<u32, Vec<f32>>,
}

impl TraceState {
    const fn new() -> Self {
        Self {
            contexts: BTreeSet::new(),
            unavailable_storage: BTreeSet::new(),
            objects: Vec::new(),
            handles: BTreeMap::new(),
            buffers: BTreeMap::new(),
            datasets: BTreeMap::new(),
            optimizers: BTreeMap::new(),
            results: BTreeMap::new(),
            saved: BTreeMap::new(),
        }
    }

    fn alloc_obj(
        &mut self,
        shape: [usize; 4],
        parameter: bool,
        op: Option<NodeOp>,
        inputs: Vec<usize>,
    ) -> usize {
        let len = shape.iter().product();
        let id = self.objects.len();
        self.objects.push(TensorObj {
            shape,
            values: vec![0.0_f32; len],
            parameter,
            op,
            inputs,
        });
        id
    }

    fn resolve_tensor(&self, id: u32, require_storage: bool) -> Result<&TensorHandle> {
        let entry = self
            .handles
            .get(&id)
            .ok_or_else(|| invalid("unknown optimizer tensor handle"))?;
        if require_storage && self.unavailable_storage.contains(&entry.context) {
            return Err(invalid(
                "optimizer tensor storage lifetime ended or was not allocated",
            ));
        }
        Ok(entry)
    }

    fn topology(&self, root: usize) -> Result<Vec<usize>> {
        fn dfs(
            objects: &[TensorObj],
            node: usize,
            visited: &mut BTreeSet<usize>,
            order: &mut Vec<usize>,
        ) -> Result<()> {
            if !visited.insert(node) {
                return Ok(());
            }
            let obj = objects
                .get(node)
                .ok_or_else(|| invalid("unknown graph tensor"))?;
            for &input in &obj.inputs {
                dfs(objects, input, visited, order)?;
            }
            order.push(node);
            Ok(())
        }
        let mut order = Vec::new();
        dfs(&self.objects, root, &mut BTreeSet::new(), &mut order)?;
        Ok(order)
    }

    fn build_optimizer(&mut self, spec: TrainingSpec) -> Result<OptimizerState> {
        require(
            (1..=MAX_PERIOD).contains(&spec.period),
            "invalid optimizer accumulation period",
        )?;
        validate_optim_params(spec.params, spec.optim)?;
        let inputs = self.resolve_tensor(spec.inputs, false)?;
        let inputs_obj = inputs.obj;
        let inputs_ctx = inputs.context;
        let outputs_obj = self.resolve_tensor(spec.outputs, false)?.obj;
        let topology = self.topology(outputs_obj)?;
        let out_shape = self.objects[outputs_obj].shape;
        let labels_obj = if matches!(
            spec.loss,
            LossKind::CrossEntropy | LossKind::MeanSquaredError
        ) {
            Some(self.alloc_obj(out_shape, false, None, Vec::new()))
        } else {
            None
        };
        let loss_obj = self.alloc_obj([1, 1, 1, 1], false, None, Vec::new());

        let mut param_nodes = Vec::new();
        for &node_id in topology.iter().rev() {
            let obj = &self.objects[node_id];
            require(
                !matches!(obj.op, Some(NodeOp::AdamWStep | NodeOp::SgdStep)),
                "optimizer step nodes have no backward operator",
            )?;
            if obj.parameter {
                param_nodes.push((node_id, obj.shape));
            }
        }

        let mut trainable = Vec::with_capacity(param_nodes.len());
        for (param, shape) in param_nodes {
            let gradient = self.alloc_obj(shape, false, None, Vec::new());
            let (moment, variance) = if spec.optim == OptimKind::AdamW {
                (
                    Some(self.alloc_obj(shape, false, None, Vec::new())),
                    Some(self.alloc_obj(shape, false, None, Vec::new())),
                )
            } else {
                (None, None)
            };
            trainable.push(TrainableSlot {
                param,
                gradient,
                moment,
                variance,
            });
        }

        Ok(OptimizerState {
            context: spec.context,
            inputs_obj,
            inputs_ctx,
            outputs_obj,
            loss_kind: spec.loss,
            optim_kind: spec.optim,
            period: spec.period,
            params: spec.params,
            topology,
            trainable,
            loss_obj,
            labels_obj,
            mode: EvalMode::Update,
            ready: false,
            iteration: 1,
            index: 0,
            rng: Mt19937::new(MT_SEED),
        })
    }

    fn clear_gradients(&mut self, opt: &OptimizerState) {
        for slot in &opt.trainable {
            self.objects[slot.gradient].values.fill(0.0_f32);
        }
    }

    fn alloc_optimizer(&mut self, opt_id: u32, backward: bool) -> Result<()> {
        let mut opt = self
            .optimizers
            .remove(&opt_id)
            .ok_or_else(|| invalid("unknown optimizer handle"))?;
        let res = self.alloc_optimizer_inner(&mut opt, backward);
        self.optimizers.insert(opt_id, opt);
        res
    }

    fn alloc_optimizer_inner(&mut self, opt: &mut OptimizerState, backward: bool) -> Result<()> {
        require(!opt.ready, "optimizer graph is already awaiting evaluation")?;
        if opt.mode == EvalMode::Update && opt.period > 1 && opt.index == 0 {
            self.clear_gradients(opt);
        }
        opt.mode = if !backward {
            EvalMode::Forward
        } else if (opt.index + 1).is_multiple_of(opt.period) {
            EvalMode::Update
        } else {
            EvalMode::Gradient
        };
        opt.ready = true;
        self.unavailable_storage.remove(&opt.context);
        Ok(())
    }

    fn reset_optimizer(&mut self, opt_id: u32, reset_state: bool) -> Result<()> {
        let mut opt = self
            .optimizers
            .remove(&opt_id)
            .ok_or_else(|| invalid("unknown optimizer handle"))?;
        self.clear_gradients(&opt);
        if reset_state {
            opt.iteration = 1;
            for slot in &opt.trainable {
                if let Some(m) = slot.moment {
                    self.objects[m].values.fill(0.0_f32);
                }
                if let Some(v) = slot.variance {
                    self.objects[v].values.fill(0.0_f32);
                }
            }
        }
        self.optimizers.insert(opt_id, opt);
        Ok(())
    }

    fn eval_forward(&mut self, opt: &OptimizerState) -> Result<()> {
        for &node_id in &opt.topology {
            let Some(op) = self.objects[node_id].op else {
                continue;
            };
            let shape = self.objects[node_id].shape;
            let len = self.objects[node_id].elements();
            let inputs = self.objects[node_id].inputs.clone();
            let mut next = vec![0.0_f32; len];
            match op {
                NodeOp::Add | NodeOp::Multiply => {
                    let a = &self.objects[inputs[0]].values;
                    let b_obj = &self.objects[inputs[1]];
                    let b_shape = b_obj.shape;
                    let b = &b_obj.values;
                    for (idx, dst) in next.iter_mut().enumerate() {
                        let bi = repeated_index(idx, shape, b_shape);
                        *dst = match op {
                            NodeOp::Add => a[idx] + b[bi],
                            _ => a[idx] * b[bi],
                        };
                    }
                }
                NodeOp::Scale(factor) => {
                    for (&val, dst) in self.objects[inputs[0]].values.iter().zip(&mut next) {
                        *dst = val * factor;
                    }
                }
                NodeOp::AdamWStep | NodeOp::SgdStep => {
                    return Err(invalid(
                        "optimizer step node cannot execute in forward graph",
                    ));
                }
            }
            self.objects[node_id].values = next;
        }
        self.compute_loss(opt)
    }

    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::suboptimal_flops,
        reason = "Loss computation preserves native f64 accumulation narrowed to binary32 and period/element divisors."
    )]
    fn compute_loss(&mut self, opt: &OptimizerState) -> Result<()> {
        let out_obj = &self.objects[opt.outputs_obj];
        let outputs = &out_obj.values;
        let out_shape = out_obj.shape;
        let loss_val = match opt.loss_kind {
            LossKind::Sum => outputs.iter().map(|&x| f64::from(x)).sum::<f64>() as f32,
            LossKind::Mean => {
                let sum = outputs.iter().map(|&x| f64::from(x)).sum::<f64>() as f32;
                sum * (1.0_f32 / (opt.period * outputs.len()) as f32)
            }
            LossKind::MeanSquaredError => {
                let labels_id = opt
                    .labels_obj
                    .ok_or_else(|| invalid("missing MSE labels tensor"))?;
                let labels = &self.objects[labels_id].values;
                let sum_sq = outputs
                    .iter()
                    .zip(labels)
                    .map(|(&y, &t)| {
                        let d = y - t;
                        f64::from(d * d)
                    })
                    .sum::<f64>() as f32;
                sum_sq * (1.0_f32 / (opt.period * outputs.len()) as f32)
            }
            LossKind::CrossEntropy => {
                let labels_id = opt
                    .labels_obj
                    .ok_or_else(|| invalid("missing cross-entropy labels tensor"))?;
                let labels = &self.objects[labels_id].values;
                let width = out_shape[0];
                let rows = outputs.len() / width;
                let mut total = 0.0_f64;
                for (out_row, lbl_row) in
                    outputs.chunks_exact(width).zip(labels.chunks_exact(width))
                {
                    let max_val = out_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let sum_exp = out_row
                        .iter()
                        .map(|&x| f64::from((x - max_val).exp()))
                        .sum::<f64>();
                    let log_sum = sum_exp.ln();
                    for (&x, &t) in out_row.iter().zip(lbl_row) {
                        total -= f64::from(t) * (f64::from(x - max_val) - log_sum);
                    }
                }
                let ce = (total / rows as f64) as f32;
                ce * (1.0_f32 / opt.period as f32)
            }
        };
        self.objects[opt.loss_obj].values[0] = loss_val;
        Ok(())
    }

    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "Loss gradient preserves native binary32 scaling, subtraction and softmax rounding."
    )]
    fn output_gradient(&self, opt: &OptimizerState) -> Result<Vec<f32>> {
        let out_obj = &self.objects[opt.outputs_obj];
        let outputs = &out_obj.values;
        let out_shape = out_obj.shape;
        match opt.loss_kind {
            LossKind::Sum => Ok(vec![1.0_f32; outputs.len()]),
            LossKind::Mean => {
                let scale = 1.0_f32 / (opt.period * outputs.len()) as f32;
                Ok(vec![scale; outputs.len()])
            }
            LossKind::MeanSquaredError => {
                let labels_id = opt
                    .labels_obj
                    .ok_or_else(|| invalid("missing MSE labels tensor"))?;
                let labels = &self.objects[labels_id].values;
                let scale = 1.0_f32 / (opt.period * outputs.len()) as f32;
                Ok(outputs
                    .iter()
                    .zip(labels)
                    .map(|(&y, &t)| (y - t) * scale * 2.0_f32)
                    .collect())
            }
            LossKind::CrossEntropy => {
                let labels_id = opt
                    .labels_obj
                    .ok_or_else(|| invalid("missing cross-entropy labels tensor"))?;
                let labels = &self.objects[labels_id].values;
                let width = out_shape[0];
                let rows = outputs.len() / width;
                let scale = 1.0_f32 / (opt.period * rows) as f32;
                let mut grad = vec![0.0_f32; outputs.len()];
                for ((out_row, lbl_row), grad_row) in outputs
                    .chunks_exact(width)
                    .zip(labels.chunks_exact(width))
                    .zip(grad.chunks_exact_mut(width))
                {
                    let max_val = out_row[argmax(out_row)];
                    let sum_exp = out_row
                        .iter()
                        .map(|&x| f64::from((x - max_val).exp()))
                        .sum::<f64>() as f32;
                    for ((&x, &t), dst) in out_row.iter().zip(lbl_row).zip(grad_row) {
                        *dst = ((x - max_val).exp() / sum_exp - t) * scale;
                    }
                }
                Ok(grad)
            }
        }
    }

    fn backward_gradients(
        &self,
        opt: &OptimizerState,
        seed: Vec<f32>,
    ) -> Result<BTreeMap<usize, Vec<f32>>> {
        let mut grads = BTreeMap::from([(opt.outputs_obj, seed)]);
        for &node_id in opt.topology.iter().rev() {
            let obj = &self.objects[node_id];
            let Some(op) = obj.op else {
                continue;
            };
            let Some(grad_out) = grads.get(&node_id).cloned() else {
                continue;
            };
            let out_shape = obj.shape;
            for (pos, &in_id) in obj.inputs.iter().enumerate() {
                let in_shape = self.objects[in_id].shape;
                let in_len = self.objects[in_id].elements();
                let other = if matches!(op, NodeOp::Multiply) {
                    let other_id = obj.inputs[1 - pos];
                    Some((&self.objects[other_id].values, self.objects[other_id].shape))
                } else {
                    None
                };
                let target = grads.entry(in_id).or_insert_with(|| vec![0.0_f32; in_len]);
                for (idx, &g) in grad_out.iter().enumerate() {
                    let val = match op {
                        NodeOp::Add => g,
                        NodeOp::Scale(factor) => g * factor,
                        NodeOp::Multiply => {
                            let (other_vals, other_shape) = other
                                .as_ref()
                                .ok_or_else(|| invalid("missing multiply partner"))?;
                            g * other_vals[repeated_index(idx, out_shape, *other_shape)]
                        }
                        NodeOp::AdamWStep | NodeOp::SgdStep => {
                            return Err(invalid("optimizer step node has no backward operator"));
                        }
                    };
                    target[repeated_index(idx, out_shape, in_shape)] += val;
                }
            }
        }
        Ok(grads)
    }

    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        clippy::suboptimal_flops,
        reason = "AdamW/SGD parameter updates preserve native binary32 powf bias correction and operation order."
    )]
    fn step_parameters(&mut self, opt: &OptimizerState) -> Result<()> {
        validate_optim_params(opt.params, opt.optim_kind)?;
        match opt.optim_kind {
            OptimKind::AdamW => {
                let [alpha, beta1, beta2, eps, wd] = opt.params.adamw;
                let beta1_corr = 1.0_f32 / (1.0_f32 - beta1.powf(opt.iteration as f32));
                let beta2_corr = 1.0_f32 / (1.0_f32 - beta2.powf(opt.iteration as f32));
                let keep = 1.0_f32 - alpha * wd;
                for slot in &opt.trainable {
                    let grad = self.objects[slot.gradient].values.clone();
                    let moment_id = slot
                        .moment
                        .ok_or_else(|| invalid("missing AdamW moment tensor"))?;
                    let variance_id = slot
                        .variance
                        .ok_or_else(|| invalid("missing AdamW variance tensor"))?;
                    let mut moment = self.objects[moment_id].values.clone();
                    let mut variance = self.objects[variance_id].values.clone();
                    let weights = &mut self.objects[slot.param].values;
                    for idx in 0..weights.len() {
                        let g = grad[idx];
                        moment[idx] = moment[idx] * beta1 + g * (1.0_f32 - beta1);
                        variance[idx] = variance[idx] * beta2 + g * g * (1.0_f32 - beta2);
                        let m_hat = moment[idx] * beta1_corr;
                        let v_hat = (variance[idx] * beta2_corr).sqrt() + eps;
                        weights[idx] = weights[idx] * keep - alpha * m_hat / v_hat;
                    }
                    self.objects[moment_id].values = moment;
                    self.objects[variance_id].values = variance;
                }
            }
            OptimKind::Sgd => {
                let [alpha, wd] = opt.params.sgd;
                let keep = 1.0_f32 - alpha * wd;
                for slot in &opt.trainable {
                    let grad = self.objects[slot.gradient].values.clone();
                    let weights = &mut self.objects[slot.param].values;
                    for (w, g) in weights.iter_mut().zip(grad) {
                        *w = *w * keep - alpha * g;
                    }
                }
            }
        }
        Ok(())
    }

    fn eval_optimizer_inner(
        &mut self,
        opt: &mut OptimizerState,
        result: Option<&mut Metrics>,
    ) -> Result<()> {
        require(opt.ready, "optimizer evaluation requires graph allocation")?;
        require(
            !self.unavailable_storage.contains(&opt.inputs_ctx)
                && !self.unavailable_storage.contains(&opt.context),
            "optimizer evaluation requires allocated input and compute storage",
        )?;
        self.eval_forward(opt)?;

        let out_shape = self.objects[opt.outputs_obj].shape;
        let correct = if result.is_some() && opt.loss_kind == LossKind::CrossEntropy {
            let labels_id = opt
                .labels_obj
                .ok_or_else(|| invalid("missing cross-entropy labels tensor"))?;
            let outputs = &self.objects[opt.outputs_obj].values;
            let labels = &self.objects[labels_id].values;
            Some(
                outputs
                    .chunks_exact(out_shape[0])
                    .zip(labels.chunks_exact(out_shape[0]))
                    .filter(|(pred, target)| argmax(pred) == argmax(target))
                    .count(),
            )
        } else {
            None
        };

        if opt.mode != EvalMode::Forward {
            let seed = self.output_gradient(opt)?;
            let grads = self.backward_gradients(opt, seed)?;
            for slot in &opt.trainable {
                let next = grads
                    .get(&slot.param)
                    .ok_or_else(|| invalid("missing trainable parameter gradient"))?;
                let acc = &mut self.objects[slot.gradient].values;
                for (dst, &g) in acc.iter_mut().zip(next) {
                    if opt.period > 1 {
                        *dst += g;
                    } else {
                        *dst = g;
                    }
                }
            }
            if opt.mode == EvalMode::Update {
                self.step_parameters(opt)?;
                opt.iteration += 1;
            }
        }

        if let Some(metrics) = result {
            metrics.record(
                out_shape[1],
                self.objects[opt.loss_obj].values[0],
                correct,
                opt.period,
                opt.loss_kind != LossKind::Sum,
            )?;
        }

        opt.index = (opt.index + 1) % opt.period;
        opt.ready = false;
        Ok(())
    }

    fn eval_action(&mut self, opt_id: u32, result_id: Option<u32>) -> Result<()> {
        let mut opt = self
            .optimizers
            .remove(&opt_id)
            .ok_or_else(|| invalid("unknown optimizer handle"))?;
        let mut metrics = result_id
            .map(|id| {
                self.results
                    .remove(&id)
                    .ok_or_else(|| invalid("unknown optimizer result"))
            })
            .transpose()?;
        let res = self.eval_optimizer_inner(&mut opt, metrics.as_mut());
        if let (Some(id), Some(m)) = (result_id, metrics) {
            self.results.insert(id, m);
        }
        self.optimizers.insert(opt_id, opt);
        res
    }

    fn run_epoch(
        &mut self,
        opt: &mut OptimizerState,
        dataset: &Dataset,
        mut train: Option<&mut Metrics>,
        mut evaluate: Option<&mut Metrics>,
        split: usize,
    ) -> Result<()> {
        require(
            !opt.ready,
            "optimizer epoch cannot start while awaiting eval",
        )?;
        require(
            !self.unavailable_storage.contains(&opt.inputs_ctx),
            "optimizer epoch requires allocated input tensor storage",
        )?;
        let in_shape = self.objects[opt.inputs_obj].shape;
        let physical = in_shape[1];
        require(
            dataset.width == in_shape[0]
                && in_shape[2..] == [1, 1]
                && dataset.count.is_multiple_of(physical)
                && split <= dataset.count
                && split.is_multiple_of(physical),
            "optimizer epoch has incompatible dataset or split shape",
        )?;
        for idx in 0..dataset.count / physical {
            let backward = idx < split / physical;
            self.alloc_optimizer_inner(opt, backward)?;
            let (data, labels) = dataset.batch(physical, idx)?;
            self.objects[opt.inputs_obj].values = data;
            if let Some(labels_obj) = opt.labels_obj {
                require(
                    labels.len() == self.objects[labels_obj].elements(),
                    "optimizer dataset label extent mismatches outputs",
                )?;
                self.objects[labels_obj].values = labels;
            } else {
                require(labels.is_empty(), "optimizer has no label input")?;
            }
            self.eval_optimizer_inner(
                opt,
                if backward {
                    train.as_deref_mut()
                } else {
                    evaluate.as_deref_mut()
                },
            )?;
        }
        self.unavailable_storage.remove(&opt.context);
        Ok(())
    }

    fn epoch_action(&mut self, action: &Value) -> Result<()> {
        let opt_id = handle(action, "Optimizer")?;
        let ds_id = handle(action, "Dataset")?;
        let split = usize::try_from(uint_field(action, "Split", 0, 4096)?)?;
        let train_id = if field(action, "Train")?.is_null() {
            None
        } else {
            Some(handle(action, "Train")?)
        };
        let eval_id = if field(action, "Evaluate")?.is_null() {
            None
        } else {
            Some(handle(action, "Evaluate")?)
        };
        let dataset = self
            .datasets
            .get(&ds_id)
            .cloned()
            .ok_or_else(|| invalid("unknown optimizer dataset"))?;
        require(
            split <= dataset.count,
            "optimizer epoch split exceeds dataset",
        )?;

        let mut train_metrics = train_id
            .map(|id| {
                self.results
                    .remove(&id)
                    .ok_or_else(|| invalid("unknown optimizer result"))
            })
            .transpose()?;
        let shared = train_id.is_some() && train_id == eval_id;
        let mut eval_metrics = if shared {
            Some(Metrics::default())
        } else {
            eval_id
                .map(|id| {
                    self.results
                        .remove(&id)
                        .ok_or_else(|| invalid("unknown optimizer result"))
                })
                .transpose()?
        };

        let mut opt = self
            .optimizers
            .remove(&opt_id)
            .ok_or_else(|| invalid("unknown optimizer handle"))?;
        let run_res = self.run_epoch(
            &mut opt,
            &dataset,
            train_metrics.as_mut(),
            eval_metrics.as_mut(),
            split,
        );
        self.optimizers.insert(opt_id, opt);

        if shared && let (Some(train), Some(eval)) = (train_metrics.as_mut(), eval_metrics.as_ref())
        {
            train.append(eval)?;
        }
        if let (Some(id), Some(m)) = (train_id, train_metrics) {
            self.results.insert(id, m);
        }
        if !shared && let (Some(id), Some(m)) = (eval_id, eval_metrics) {
            self.results.insert(id, m);
        }
        run_res
    }

    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        reason = "Fitting reproduces native binary32 validation-split batch truncation."
    )]
    fn fit_split(count: usize, logical_batch: usize, val_split: f32) -> usize {
        (((1.0_f32 - val_split) * (count / logical_batch) as f32) as usize) * logical_batch
    }

    fn fit_action(&mut self, action: &Value) -> Result<()> {
        let context = handle(action, "Context")?;
        require(
            self.contexts.contains(&context),
            "unknown optimizer context",
        )?;
        let inputs_handle = handle(action, "Inputs")?;
        let outputs_handle = handle(action, "Outputs")?;
        let ds_id = handle(action, "Dataset")?;
        let loss_kind = LossKind::from_code(uint_field(action, "Loss", 0, 3)?)?;
        let optim_kind = OptimKind::from_code(uint_field(action, "Optimizer", 0, 1)?)?;
        let epochs = usize::try_from(uint_field(action, "Epochs", 1, MAX_EPOCHS_U64)?)?;
        let logical_batch = usize::try_from(uint_field(
            action,
            "LogicalBatch",
            1,
            MAX_DATASET_COUNT_U64,
        )?)?;
        let val_split = finite_f32_field(action, "ValidationBits")?;
        require(
            (0.0_f32..1.0_f32).contains(&val_split),
            "optimizer validation split invalid",
        )?;
        boolean(action, "Silent")?;
        let schedule = if field(action, "Parameters")?.is_null() {
            None
        } else {
            let items = array(action, "Parameters")?;
            require(
                items.len() == epochs,
                "optimizer fit parameter schedule mismatch",
            )?;
            Some(
                items
                    .iter()
                    .map(parse_parameters)
                    .collect::<Result<Vec<_>>>()?,
            )
        };

        let inputs_entry = self.resolve_tensor(inputs_handle, true)?;
        let physical = self.objects[inputs_entry.obj].shape[1];
        let mut dataset = self
            .datasets
            .remove(&ds_id)
            .ok_or_else(|| invalid("unknown optimizer dataset"))?;

        let prep = (|| -> Result<()> {
            require(
                dataset.count.is_multiple_of(logical_batch)
                    && logical_batch.is_multiple_of(physical),
                "optimizer fit batch extent mismatch",
            )?;
            let period = logical_batch / physical;
            let split = Self::fit_split(dataset.count, logical_batch, val_split);
            let mut opt = self.build_optimizer(TrainingSpec {
                context,
                inputs: inputs_handle,
                outputs: outputs_handle,
                loss: loss_kind,
                optim: optim_kind,
                period,
                params: OptimParams::default(),
            })?;
            if logical_batch < dataset.count {
                dataset.shuffle(&mut opt.rng, None)?;
            }
            for epoch in 0..epochs {
                if let Some(ref sched) = schedule {
                    opt.params = sched[epoch];
                }
                if logical_batch < split {
                    dataset.shuffle(&mut opt.rng, Some(split))?;
                }
                self.run_epoch(&mut opt, &dataset, None, None, split)?;
            }
            Ok(())
        })();

        self.datasets.insert(ds_id, dataset);
        self.unavailable_storage.insert(context);
        prep
    }

    fn tensor_action(&mut self, action: &Value) -> Result<()> {
        let id = handle(action, "Handle")?;
        let ctx = handle(action, "Context")?;
        require(self.contexts.contains(&ctx), "unknown optimizer context")?;
        require(
            self.handles.len() < MAX_TENSORS && !self.handles.contains_key(&id),
            "optimizer tensor exists or count exceeds bound",
        )?;
        let dims = array(action, "Dimensions")?;
        require(
            !dims.is_empty() && dims.len() <= 4,
            "optimizer dimensions invalid",
        )?;
        let mut shape = [1_usize; 4];
        let mut elements = 1_usize;
        for (slot, raw) in shape.iter_mut().zip(dims) {
            let dim = usize::try_from(uint_value(raw, 1, MAX_TENSOR_DIM)?)?;
            elements = elements
                .checked_mul(dim)
                .filter(|&count| count <= MAX_TENSOR_ELEMENTS)
                .ok_or_else(|| invalid("optimizer tensor elements exceed bound"))?;
            *slot = dim;
        }
        let parameter = boolean(action, "Parameter")?;
        let name = text(action, "Name")?;
        require(
            name.len() <= MAX_NAME_LEN && !name.contains('\0'),
            "optimizer tensor name invalid",
        )?;
        let obj = self.alloc_obj(shape, parameter, None, Vec::new());
        self.handles.insert(
            id,
            TensorHandle {
                obj,
                context: ctx,
                optimizer: None,
            },
        );
        Ok(())
    }

    fn node_action(&mut self, action: &Value) -> Result<()> {
        let id = handle(action, "Handle")?;
        let ctx = handle(action, "Context")?;
        require(self.contexts.contains(&ctx), "unknown optimizer context")?;
        require(
            self.handles.len() < MAX_TENSORS && !self.handles.contains_key(&id),
            "optimizer node exists or count exceeds bound",
        )?;
        let op_name = text(action, "Operation")?;
        let input_vals = array(action, "Inputs")?;
        let mut input_objs = Vec::with_capacity(input_vals.len());
        for raw in input_vals {
            let h = handle_value(raw)?;
            input_objs.push(self.resolve_tensor(h, false)?.obj);
        }
        let shape0 = self.objects[input_objs[0]].shape;
        let op = match op_name {
            "add" | "mul" => {
                require(
                    input_objs.len() == 2,
                    "optimizer binary input count invalid",
                )?;
                require(
                    field(action, "FactorBits")?.is_null(),
                    "non-scale factor must be null",
                )?;
                let shape1 = self.objects[input_objs[1]].shape;
                require(
                    can_repeat(shape1, shape0),
                    "optimizer broadcast shape mismatch",
                )?;
                if op_name == "add" {
                    NodeOp::Add
                } else {
                    NodeOp::Multiply
                }
            }
            "scale" => {
                require(input_objs.len() == 1, "optimizer scale input count invalid")?;
                NodeOp::Scale(finite_f32_field(action, "FactorBits")?)
            }
            "adamw-step" => {
                require(input_objs.len() == 5, "optimizer AdamW input count invalid")?;
                require(
                    field(action, "FactorBits")?.is_null(),
                    "non-scale factor must be null",
                )?;
                require(
                    self.objects[input_objs[1]].shape == shape0
                        && self.objects[input_objs[2]].shape == shape0
                        && self.objects[input_objs[3]].shape == shape0
                        && self.objects[input_objs[4]].elements() == 7,
                    "optimizer AdamW step shape mismatch",
                )?;
                NodeOp::AdamWStep
            }
            "sgd-step" => {
                require(input_objs.len() == 3, "optimizer SGD input count invalid")?;
                require(
                    field(action, "FactorBits")?.is_null(),
                    "non-scale factor must be null",
                )?;
                require(
                    self.objects[input_objs[1]].shape == shape0
                        && self.objects[input_objs[2]].elements() == 2,
                    "optimizer SGD step shape mismatch",
                )?;
                NodeOp::SgdStep
            }
            _ => return Err(invalid("unknown optimizer node operation")),
        };
        let name = text(action, "Name")?;
        require(
            name.len() <= MAX_NAME_LEN && !name.contains('\0'),
            "optimizer node name invalid",
        )?;
        let obj = self.alloc_obj(shape0, false, Some(op), input_objs);
        self.handles.insert(
            id,
            TensorHandle {
                obj,
                context: ctx,
                optimizer: None,
            },
        );
        Ok(())
    }

    fn save_restore_action(&mut self, action: &Value, save: bool) -> Result<()> {
        let obj = self.resolve_tensor(handle(action, "Tensor")?, true)?.obj;
        let id = handle(action, "Handle")?;
        let elems = self.objects[obj].elements();
        require(
            elems * 4 <= MAX_PAYLOAD_BYTES,
            "optimizer saved tensor extent invalid",
        )?;
        if save {
            require(
                self.saved.len() < MAX_SAVED || self.saved.contains_key(&id),
                "optimizer saved tensor count exceeds bound",
            )?;
            self.saved.insert(id, self.objects[obj].values.clone());
        } else {
            let saved = self
                .saved
                .get(&id)
                .ok_or_else(|| invalid("unknown saved optimizer tensor"))?;
            require(
                saved.len() == elems,
                "optimizer restore tensor extent mismatch",
            )?;
            self.objects[obj].values.clone_from(saved);
        }
        Ok(())
    }

    fn dataset_action(&mut self, action: &Value) -> Result<Value> {
        let id = handle(action, "Handle")?;
        require(
            self.datasets.len() < MAX_DATASETS && !self.datasets.contains_key(&id),
            "optimizer dataset exists or count exceeds bound",
        )?;
        let count = usize::try_from(uint_field(action, "Count", 1, MAX_DATASET_COUNT_U64)?)?;
        let width = usize::try_from(uint_field(
            action,
            "DatapointElements",
            1,
            MAX_DATAPOINT_ELEMS_U64,
        )?)?;
        let label_width = usize::try_from(uint_field(
            action,
            "LabelElements",
            0,
            MAX_DATAPOINT_ELEMS_U64,
        )?)?;
        let shard = usize::try_from(uint_field(action, "Shard", 1, u64::try_from(count)?)?)?;
        let data = unhex_f32(field(action, "DataHex")?)?;
        let labels = unhex_f32(field(action, "LabelsHex")?)?;
        let dataset = Dataset::new(width, label_width, count, shard, data, labels)?;
        self.datasets.insert(id, dataset);
        Ok(json!({"Count": count}))
    }

    fn init_action(&mut self, action: &Value) -> Result<Value> {
        let id = handle(action, "Handle")?;
        let ctx = handle(action, "Context")?;
        require(self.contexts.contains(&ctx), "unknown optimizer context")?;
        require(
            self.optimizers.len() < MAX_OPTIMIZERS && !self.optimizers.contains_key(&id),
            "optimizer handle exists or count exceeds bound",
        )?;
        let params = if field(action, "Parameters")?.is_null() {
            OptimParams::default()
        } else {
            parse_parameters(field(action, "Parameters")?)?
        };
        let state = self.build_optimizer(TrainingSpec {
            context: ctx,
            inputs: handle(action, "Inputs")?,
            outputs: handle(action, "Outputs")?,
            loss: LossKind::from_code(uint_field(action, "Loss", 0, 3)?)?,
            optim: OptimKind::from_code(uint_field(action, "Optimizer", 0, 1)?)?,
            period: usize::try_from(uint_field(action, "Period", 1, MAX_PERIOD_U64)?)?,
            params,
        })?;
        self.optimizers.insert(id, state);
        Ok(json!({
            "Created": true,
            "Optimizer": field(action, "Optimizer")?,
            "HasCallback": true
        }))
    }

    fn observation_action(&mut self, action: &Value, is_gradient: bool) -> Result<()> {
        let opt_id = handle(action, "Optimizer")?;
        let handle_id = handle(action, "Handle")?;
        require(
            self.handles.len() < MAX_TENSORS || self.handles.contains_key(&handle_id),
            "optimizer observation handle count exceeds bound",
        )?;
        let target_param = if is_gradient {
            Some(self.resolve_tensor(handle(action, "Tensor")?, false)?.obj)
        } else {
            None
        };
        let opt = self
            .optimizers
            .get(&opt_id)
            .ok_or_else(|| invalid("unknown optimizer handle"))?;
        let obj = if let Some(param_obj) = target_param {
            require(
                opt.period > 1,
                "period-one optimizer has no persistent gradient accumulator",
            )?;
            opt.trainable
                .iter()
                .find(|slot| slot.param == param_obj)
                .map(|slot| slot.gradient)
                .ok_or_else(|| invalid("tensor has no optimizer gradient accumulator"))?
        } else {
            opt.loss_obj
        };
        let context = opt.context;
        self.handles.insert(
            handle_id,
            TensorHandle {
                obj,
                context,
                optimizer: Some(opt_id),
            },
        );
        Ok(())
    }

    fn batch_action(&mut self, action: &Value) -> Result<()> {
        let ds_id = handle(action, "Dataset")?;
        let data_obj = self.resolve_tensor(handle(action, "Data")?, true)?.obj;
        let labels_obj = self.resolve_tensor(handle(action, "Labels")?, true)?.obj;
        let index = usize::try_from(uint_field(action, "Index", 0, MAX_HANDLE)?)?;
        let dataset = self
            .datasets
            .get(&ds_id)
            .ok_or_else(|| invalid("unknown optimizer dataset"))?;
        let data_elems = self.objects[data_obj].elements();
        let labels_elems = self.objects[labels_obj].elements();
        require(
            data_elems.is_multiple_of(dataset.width),
            "optimizer batch data extent mismatch",
        )?;
        let count = data_elems / dataset.width;
        require(
            labels_elems == count * dataset.label_width,
            "optimizer batch labels extent mismatch",
        )?;
        let (data, labels) = dataset.batch(count, index)?;
        self.objects[data_obj].values = data;
        if dataset.label_width > 0 {
            self.objects[labels_obj].values = labels;
        }
        Ok(())
    }

    fn free_action(&mut self, action: &Value) -> Result<()> {
        let kind = text(action, "Type")?;
        let id = handle(action, "Handle")?;
        match kind {
            "optimizer" => {
                let opt = self
                    .optimizers
                    .remove(&id)
                    .ok_or_else(|| invalid("unknown optimizer handle"))?;
                self.unavailable_storage.insert(opt.context);
                self.handles.retain(|_, entry| entry.optimizer != Some(id));
            }
            "result" => {
                self.results
                    .remove(&id)
                    .ok_or_else(|| invalid("unknown optimizer result"))?;
            }
            "dataset" => {
                self.datasets
                    .remove(&id)
                    .ok_or_else(|| invalid("unknown optimizer dataset"))?;
            }
            "buffer" => {
                require(
                    self.optimizers.is_empty(),
                    "optimizer still retains tensor buffers",
                )?;
                let ctx = self
                    .buffers
                    .remove(&id)
                    .ok_or_else(|| invalid("unknown optimizer buffer"))?;
                self.unavailable_storage.insert(ctx);
            }
            "context" => {
                require(
                    !self.optimizers.values().any(|opt| opt.context == id)
                        && !self.buffers.values().any(|&ctx| ctx == id)
                        && self.contexts.remove(&id),
                    "optimizer context missing or still retained",
                )?;
                self.unavailable_storage.remove(&id);
                self.handles.retain(|_, entry| entry.context != id);
            }
            _ => return Err(invalid("unknown optimizer free type")),
        }
        Ok(())
    }

    fn result_snapshot_action(&self, action: &Value) -> Result<Value> {
        let id = handle(action, "Result")?;
        let metrics = self
            .results
            .get(&id)
            .ok_or_else(|| invalid("unknown optimizer result"))?;
        let loss_unc_flag = boolean(action, "LossUncertainty")?;
        let acc_unc_flag = boolean(action, "AccuracyUncertainty")?;
        let (loss, loss_unc) = metrics.loss();
        let (acc, acc_unc) = metrics.accuracy();
        Ok(json!({
            "Count": metrics.count,
            "LossBits": loss.to_bits(),
            "LossUncertaintyBits": loss_unc_flag.then(|| loss_unc.to_bits()),
            "AccuracyBits": acc.to_bits(),
            "AccuracyUncertaintyBits": acc_unc_flag.then(|| acc_unc.to_bits()),
        }))
    }

    fn shuffle_action(&mut self, action: &Value) -> Result<()> {
        let opt_id = handle(action, "Optimizer")?;
        let ds_id = handle(action, "Dataset")?;
        let count = int_field(action, "Count", -1, MAX_DATASET_COUNT_I64)?;
        let opt = self
            .optimizers
            .get_mut(&opt_id)
            .ok_or_else(|| invalid("unknown optimizer handle"))?;
        let dataset = self
            .datasets
            .get_mut(&ds_id)
            .ok_or_else(|| invalid("unknown optimizer dataset"))?;
        dataset.shuffle(&mut opt.rng, usize::try_from(count).ok())
    }

    fn execute_action(&mut self, action: &Value) -> Result<Value> {
        let mut event = json!({});
        match text(action, "Op")? {
            "context" => {
                let id = handle(action, "Handle")?;
                uint_field(action, "Bytes", 1, MAX_CONTEXT_BYTES)?;
                require(
                    self.contexts.len() < MAX_CONTEXTS && self.contexts.insert(id),
                    "optimizer context exists or count exceeds bound",
                )?;
                self.unavailable_storage.insert(id);
                event = json!({"Created": true});
            }
            "tensor" => self.tensor_action(action)?,
            "node" => self.node_action(action)?,
            "allocate" => {
                let id = handle(action, "Handle")?;
                let ctx = handle(action, "Context")?;
                require(self.contexts.contains(&ctx), "unknown optimizer context")?;
                require(
                    self.buffers.len() < MAX_BUFFERS && !self.buffers.contains_key(&id),
                    "optimizer buffer exists or count exceeds bound",
                )?;
                self.buffers.insert(id, ctx);
                self.unavailable_storage.remove(&ctx);
            }
            "set" => {
                let obj = self.resolve_tensor(handle(action, "Tensor")?, true)?.obj;
                let values = unhex_f32(field(action, "DataHex")?)?;
                require(
                    values.len() == self.objects[obj].elements(),
                    "optimizer tensor set extent mismatch",
                )?;
                self.objects[obj].values = values;
            }
            "save" => self.save_restore_action(action, true)?,
            "restore" => self.save_restore_action(action, false)?,
            "get" => {
                let obj = self.resolve_tensor(handle(action, "Tensor")?, true)?.obj;
                event = json!({"DataHex": hex_f32(&self.objects[obj].values)?});
            }
            "supports" => {
                self.resolve_tensor(handle(action, "Tensor")?, false)?;
                event = json!({"Supported": true});
            }
            "dataset" => event = self.dataset_action(action)?,
            "params" => {
                LossKind::from_code(uint_field(action, "Loss", 0, 3)?)?;
                event = json!({"HasCallback": true});
            }
            "init" => event = self.init_action(action)?,
            "result" => {
                let id = handle(action, "Handle")?;
                require(
                    self.results.len() < MAX_RESULTS && !self.results.contains_key(&id),
                    "optimizer result exists or count exceeds bound",
                )?;
                self.results.insert(id, Metrics::default());
            }
            "result-reset" => {
                let id = handle(action, "Result")?;
                self.results
                    .get_mut(&id)
                    .ok_or_else(|| invalid("unknown optimizer result"))?
                    .reset();
            }
            "result-snapshot" => event = self.result_snapshot_action(action)?,
            "alloc" => {
                self.alloc_optimizer(handle(action, "Optimizer")?, boolean(action, "Value")?)?;
            }
            "reset" => {
                self.reset_optimizer(handle(action, "Optimizer")?, boolean(action, "Value")?)?;
            }
            "eval" => {
                let opt_id = handle(action, "Optimizer")?;
                let res_id = if field(action, "Result")?.is_null() {
                    None
                } else {
                    Some(handle(action, "Result")?)
                };
                self.eval_action(opt_id, res_id)?;
            }
            "loss" => self.observation_action(action, false)?,
            "gradient" => self.observation_action(action, true)?,
            "shuffle" => self.shuffle_action(action)?,
            "batch" => self.batch_action(action)?,
            "epoch" => self.epoch_action(action)?,
            "fit" => self.fit_action(action)?,
            "free" => self.free_action(action)?,
            _ => return Err(invalid("unknown optimizer action")),
        }
        Ok(event)
    }
}

fn validate_graph_action(op: &str, action: &Value) -> Result<()> {
    match op {
        "context" => {
            exact_fields(action, &["Op", "Handle", "Bytes"])?;
            handle(action, "Handle")?;
            uint_field(action, "Bytes", 1, MAX_CONTEXT_BYTES)?;
        }
        "tensor" => {
            exact_fields(
                action,
                &["Op", "Handle", "Context", "Dimensions", "Parameter", "Name"],
            )?;
            handle(action, "Handle")?;
            handle(action, "Context")?;
            boolean(action, "Parameter")?;
            let name = text(action, "Name")?;
            require(
                name.len() <= MAX_NAME_LEN && !name.contains('\0'),
                "optimizer tensor name invalid",
            )?;
            let dims = array(action, "Dimensions")?;
            require(
                !dims.is_empty() && dims.len() <= 4,
                "optimizer dimensions invalid",
            )?;
            let mut elements = 1_usize;
            for dim in dims {
                let d = usize::try_from(uint_value(dim, 1, MAX_TENSOR_DIM)?)?;
                elements = elements
                    .checked_mul(d)
                    .filter(|&count| count <= MAX_TENSOR_ELEMENTS)
                    .ok_or_else(|| invalid("optimizer tensor elements exceed bound"))?;
            }
        }
        "node" => {
            exact_fields(
                action,
                &[
                    "Op",
                    "Handle",
                    "Context",
                    "Operation",
                    "Inputs",
                    "FactorBits",
                    "Name",
                ],
            )?;
            handle(action, "Handle")?;
            handle(action, "Context")?;
            let op_name = text(action, "Operation")?;
            let expected_inputs = match op_name {
                "add" | "mul" => 2,
                "scale" => 1,
                "adamw-step" => 5,
                "sgd-step" => 3,
                _ => return Err(invalid("unknown optimizer node operation")),
            };
            let inputs = array(action, "Inputs")?;
            require(
                inputs.len() == expected_inputs,
                "optimizer node input count invalid",
            )?;
            for input in inputs {
                handle_value(input)?;
            }
            if op_name == "scale" {
                finite_f32_field(action, "FactorBits")?;
            } else {
                require(
                    field(action, "FactorBits")?.is_null(),
                    "non-scale factor must be null",
                )?;
            }
            let name = text(action, "Name")?;
            require(
                name.len() <= MAX_NAME_LEN && !name.contains('\0'),
                "optimizer node name invalid",
            )?;
        }
        "allocate" => {
            exact_fields(action, &["Op", "Handle", "Context"])?;
            handle(action, "Handle")?;
            handle(action, "Context")?;
        }
        _ => return Err(invalid("unknown optimizer graph action")),
    }
    Ok(())
}

fn validate_storage_action(op: &str, action: &Value) -> Result<()> {
    match op {
        "set" => {
            exact_fields(action, &["Op", "Tensor", "DataHex"])?;
            handle(action, "Tensor")?;
            unhex_bytes(field(action, "DataHex")?)?;
        }
        "save" | "restore" => {
            exact_fields(action, &["Op", "Tensor", "Handle"])?;
            handle(action, "Tensor")?;
            handle(action, "Handle")?;
        }
        "get" | "supports" => {
            exact_fields(action, &["Op", "Tensor"])?;
            handle(action, "Tensor")?;
        }
        "dataset" => {
            exact_fields(
                action,
                &[
                    "Op",
                    "Handle",
                    "DatapointElements",
                    "LabelElements",
                    "Count",
                    "Shard",
                    "DataHex",
                    "LabelsHex",
                ],
            )?;
            handle(action, "Handle")?;
            let count = usize::try_from(uint_field(action, "Count", 1, MAX_DATASET_COUNT_U64)?)?;
            let data_elems = usize::try_from(uint_field(
                action,
                "DatapointElements",
                1,
                MAX_DATAPOINT_ELEMS_U64,
            )?)?;
            let label_elems = usize::try_from(uint_field(
                action,
                "LabelElements",
                0,
                MAX_DATAPOINT_ELEMS_U64,
            )?)?;
            uint_field(action, "Shard", 1, u64::try_from(count)?)?;
            let data = unhex_bytes(field(action, "DataHex")?)?;
            let labels = unhex_bytes(field(action, "LabelsHex")?)?;
            require(
                data.len() == count * data_elems * 4 && labels.len() == count * label_elems * 4,
                "optimizer dataset bytes mismatch",
            )?;
        }
        "batch" => {
            exact_fields(action, &["Op", "Dataset", "Data", "Labels", "Index"])?;
            for key in ["Dataset", "Data", "Labels"] {
                handle(action, key)?;
            }
            uint_field(action, "Index", 0, MAX_HANDLE)?;
        }
        _ => return Err(invalid("unknown optimizer storage action")),
    }
    Ok(())
}

fn validate_fit_action(action: &Value) -> Result<()> {
    exact_fields(
        action,
        &[
            "Op",
            "Context",
            "Inputs",
            "Outputs",
            "Dataset",
            "Loss",
            "Optimizer",
            "Epochs",
            "LogicalBatch",
            "ValidationBits",
            "Silent",
            "Parameters",
        ],
    )?;
    for key in ["Context", "Inputs", "Outputs", "Dataset"] {
        handle(action, key)?;
    }
    uint_field(action, "Loss", 0, 3)?;
    uint_field(action, "Optimizer", 0, 1)?;
    let epochs = usize::try_from(uint_field(action, "Epochs", 1, MAX_EPOCHS_U64)?)?;
    uint_field(action, "LogicalBatch", 1, MAX_DATASET_COUNT_U64)?;
    let split = finite_f32_field(action, "ValidationBits")?;
    require(
        (0.0_f32..1.0_f32).contains(&split),
        "optimizer validation split invalid",
    )?;
    boolean(action, "Silent")?;
    if !field(action, "Parameters")?.is_null() {
        let schedule = array(action, "Parameters")?;
        require(
            schedule.len() == epochs,
            "optimizer fit parameter schedule mismatch",
        )?;
        for entry in schedule {
            parse_parameters(entry)?;
        }
    }
    Ok(())
}

fn validate_optim_action(op: &str, action: &Value) -> Result<()> {
    match op {
        "params" => {
            exact_fields(action, &["Op", "Loss"])?;
            uint_field(action, "Loss", 0, 3)?;
        }
        "init" => {
            exact_fields(
                action,
                &[
                    "Op",
                    "Handle",
                    "Context",
                    "Inputs",
                    "Outputs",
                    "Loss",
                    "Optimizer",
                    "Period",
                    "Parameters",
                ],
            )?;
            for key in ["Handle", "Context", "Inputs", "Outputs"] {
                handle(action, key)?;
            }
            uint_field(action, "Loss", 0, 3)?;
            uint_field(action, "Optimizer", 0, 1)?;
            uint_field(action, "Period", 1, MAX_PERIOD_U64)?;
            if !field(action, "Parameters")?.is_null() {
                parse_parameters(field(action, "Parameters")?)?;
            }
        }
        "result" => {
            exact_fields(action, &["Op", "Handle"])?;
            handle(action, "Handle")?;
        }
        "result-reset" => {
            exact_fields(action, &["Op", "Result"])?;
            handle(action, "Result")?;
        }
        "result-snapshot" => {
            exact_fields(
                action,
                &["Op", "Result", "LossUncertainty", "AccuracyUncertainty"],
            )?;
            handle(action, "Result")?;
            boolean(action, "LossUncertainty")?;
            boolean(action, "AccuracyUncertainty")?;
        }
        "alloc" | "reset" => {
            exact_fields(action, &["Op", "Optimizer", "Value"])?;
            handle(action, "Optimizer")?;
            boolean(action, "Value")?;
        }
        "eval" => {
            exact_fields(action, &["Op", "Optimizer", "Result"])?;
            handle(action, "Optimizer")?;
            if !field(action, "Result")?.is_null() {
                handle(action, "Result")?;
            }
        }
        "loss" => {
            exact_fields(action, &["Op", "Optimizer", "Handle"])?;
            handle(action, "Optimizer")?;
            handle(action, "Handle")?;
        }
        "gradient" => {
            exact_fields(action, &["Op", "Optimizer", "Handle", "Tensor"])?;
            handle(action, "Optimizer")?;
            handle(action, "Handle")?;
            handle(action, "Tensor")?;
        }
        "shuffle" => {
            exact_fields(action, &["Op", "Optimizer", "Dataset", "Count"])?;
            handle(action, "Optimizer")?;
            handle(action, "Dataset")?;
            int_field(action, "Count", -1, MAX_DATASET_COUNT_I64)?;
        }
        "epoch" => {
            exact_fields(
                action,
                &["Op", "Optimizer", "Dataset", "Train", "Evaluate", "Split"],
            )?;
            handle(action, "Optimizer")?;
            handle(action, "Dataset")?;
            for key in ["Train", "Evaluate"] {
                if !field(action, key)?.is_null() {
                    handle(action, key)?;
                }
            }
            uint_field(action, "Split", 0, MAX_DATASET_COUNT_U64)?;
        }
        "free" => {
            exact_fields(action, &["Op", "Type", "Handle"])?;
            handle(action, "Handle")?;
            text(action, "Type")?;
        }
        _ => return Err(invalid("unknown optimizer action")),
    }
    Ok(())
}

fn validate_action(action: &Value) -> Result<()> {
    let op = text(action, "Op")?;
    match op {
        "context" | "tensor" | "node" | "allocate" => validate_graph_action(op, action),
        "set" | "save" | "restore" | "get" | "supports" | "dataset" | "batch" => {
            validate_storage_action(op, action)
        }
        "fit" => validate_fit_action(action),
        _ => validate_optim_action(op, action),
    }
}

fn validate_trace_operation(operation: &Value) -> Result<()> {
    exact_fields(
        operation,
        &[
            "Kind",
            "Devices",
            "Scheduler",
            "Backend",
            "Threads",
            "Actions",
        ],
    )?;
    let devices = array(operation, "Devices")?;
    require(
        !devices.is_empty() && devices.len() <= MAX_DEVICES,
        "optimizer devices exceed bound",
    )?;
    for device in devices {
        uint_value(device, 0, 0)?;
    }
    let dev_bound = u64::try_from(devices.len() - 1)?;
    let scheduler = array(operation, "Scheduler")?;
    require(
        !scheduler.is_empty() && scheduler.len() <= MAX_SCHEDULER,
        "optimizer scheduler exceeds bound",
    )?;
    for entry in scheduler {
        uint_value(entry, 0, dev_bound)?;
    }
    uint_field(operation, "Backend", 0, dev_bound)?;
    uint_field(operation, "Threads", 0, MAX_THREADS)?;
    let actions = array(operation, "Actions")?;
    require(
        actions.len() <= MAX_ACTIONS,
        "optimizer actions exceed bound",
    )?;
    for action in actions {
        validate_action(action)?;
    }
    Ok(())
}

fn execute_operation(operation: &Value) -> Result<Value> {
    match text(operation, "Kind")? {
        "optimizer-info" => {
            exact_fields(operation, &["Kind"])?;
            let defaults = OptimParams::default();
            let concurrency = std::thread::available_parallelism()?.get().min(256);
            Ok(json!({
                "Devices": [{"Index": 0, "Name": "Rust Reference CPU", "Type": 0}],
                "HardwareConcurrency": concurrency,
                "Defaults": {
                    "AdamW": defaults.adamw.map(f32::to_bits),
                    "Sgd": defaults.sgd.map(f32::to_bits),
                },
                "TensorOverhead": TENSOR_OVERHEAD,
                "GraphOverhead": GRAPH_OVERHEAD,
                "DefaultGraphSize": DEFAULT_GRAPH_SIZE,
            }))
        }
        "optimizer-trace" => {
            validate_trace_operation(operation)?;
            let mut state = TraceState::new();
            let mut trace = Vec::with_capacity(array(operation, "Actions")?.len());
            for (idx, action) in array(operation, "Actions")?.iter().enumerate() {
                let event = state.execute_action(action).map_err(|err| {
                    invalid(format!(
                        "optimizer action {idx} ({}): {err}",
                        action["Op"].as_str().unwrap_or("unknown")
                    ))
                })?;
                trace.push(event);
            }
            Ok(json!({"Trace": trace}))
        }
        _ => Err(invalid("unknown optimizer operation Kind")),
    }
}

pub(super) fn handles(request: &Value) -> bool {
    request["Operations"].as_array().is_some_and(|operations| {
        !operations.is_empty()
            && operations.iter().all(|operation| {
                matches!(
                    operation["Kind"].as_str(),
                    Some("optimizer-info" | "optimizer-trace")
                )
            })
    })
}

pub(super) fn execute(request: &Value) -> Result<Value> {
    exact_fields(request, &["SchemaVersion", "Operations"])?;
    require(
        uint_field(request, "SchemaVersion", 1, 1)? == 1,
        "optimizer schema version must equal 1",
    )?;
    let operations = array(request, "Operations")?;
    require(
        !operations.is_empty() && operations.len() <= MAX_OPERATIONS,
        "optimizer operation count exceeds bound",
    )?;
    let results = operations
        .iter()
        .map(execute_operation)
        .collect::<Result<Vec<_>>>()?;
    Ok(json!({"SchemaVersion": 1, "Results": results}))
}

pub(super) fn identity() -> Value {
    json!({
        "Kind": "independent-rust-optimizer",
        "SupportedSourceIds": [ID],
        "UpstreamCommit": super::UPSTREAM,
        "AssertionSource": {"Path": ID, "Sha256": SOURCE_SHA},
        "Architecture": "clean-room-static-graph-autodiff-adamw-sgd-and-sharded-dataset-v1",
        "NativeReferenceReplay": false,
        "ProductionRuntimeCalled": false,
        "CompleteSourceGate": false,
        "NativeRetirementQualified": false
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, Target};

    fn wrap_trace(actions: &[Value]) -> Value {
        json!({
            "SchemaVersion": 1,
            "Operations": [{
                "Kind": "optimizer-trace",
                "Devices": [0],
                "Scheduler": [0],
                "Backend": 0,
                "Threads": 1,
                "Actions": actions
            }]
        })
    }

    fn f32_hex(values: &[f32]) -> String {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for &val in values {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        hex::encode(bytes)
    }

    #[test]
    fn independent_optimizer_provider_passes_all_164_subtests_across_targets() -> Result<()> {
        for target in [Target::RustReference, Target::Cpp] {
            let engine = Engine::new(target, None, None, None, None, None)?;
            let report = crate::optimizer::run(&engine)?;
            assert_eq!(report["Id"], ID);
            assert_eq!(report["Status"], "pass", "{report}");
            let detail = report["Detail"]
                .as_str()
                .ok_or_else(|| invalid("missing optimizer Detail"))?;
            assert!(
                detail.starts_with("164/164 original result subtests;"),
                "unexpected subtest summary: {detail}"
            );
        }
        assert!(crate::reference_provider::supports(ID));
        let info = identity();
        assert_eq!(info["SupportedSourceIds"], json!([ID]));
        assert_eq!(info["NativeReferenceReplay"], false);
        assert_eq!(info["ProductionRuntimeCalled"], false);
        assert_eq!(info["CompleteSourceGate"], false);
        assert_eq!(info["NativeRetirementQualified"], false);
        Ok(())
    }

    #[test]
    fn independent_optimizer_cross_entropy_mse_and_storage_lifetimes() -> Result<()> {
        // CrossEntropy classification with shared Train == Evaluate result and AccuracyUncertainty.
        let logits_x = [1.0_f32, 0.0, 0.0, 1.0, 2.0, -1.0, -1.0, 2.0];
        let labels = [1.0_f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let response = execute(&wrap_trace(&[
            json!({"Op": "dataset", "Handle": 0, "DatapointElements": 2, "LabelElements": 2, "Count": 4, "Shard": 1, "DataHex": f32_hex(&logits_x), "LabelsHex": f32_hex(&labels)}),
            json!({"Op": "context", "Handle": 0, "Bytes": 4096}),
            json!({"Op": "context", "Handle": 1, "Bytes": 65536}),
            json!({"Op": "tensor", "Handle": 0, "Context": 0, "Dimensions": [2, 1], "Parameter": false, "Name": "x"}),
            json!({"Op": "tensor", "Handle": 1, "Context": 0, "Dimensions": [2, 1], "Parameter": true, "Name": "w"}),
            json!({"Op": "node", "Handle": 2, "Context": 1, "Operation": "mul", "Inputs": [0, 1], "FactorBits": null, "Name": "y"}),
            json!({"Op": "allocate", "Handle": 0, "Context": 0}),
            json!({"Op": "set", "Tensor": 1, "DataHex": f32_hex(&[1.0, 1.0])}),
            json!({"Op": "init", "Handle": 0, "Context": 1, "Inputs": 0, "Outputs": 2, "Loss": 2, "Optimizer": 0, "Period": 1, "Parameters": null}),
            json!({"Op": "result", "Handle": 0}),
            json!({"Op": "epoch", "Optimizer": 0, "Dataset": 0, "Train": 0, "Evaluate": 0, "Split": 2}),
            json!({"Op": "result-snapshot", "Result": 0, "LossUncertainty": true, "AccuracyUncertainty": true}),
            json!({"Op": "get", "Tensor": 2}),
            json!({"Op": "free", "Type": "result", "Handle": 0}),
            json!({"Op": "free", "Type": "optimizer", "Handle": 0}),
            json!({"Op": "free", "Type": "buffer", "Handle": 0}),
            json!({"Op": "free", "Type": "context", "Handle": 0}),
            json!({"Op": "free", "Type": "context", "Handle": 1}),
            json!({"Op": "free", "Type": "dataset", "Handle": 0}),
        ]))?;
        let trace = response["Results"][0]["Trace"]
            .as_array()
            .ok_or_else(|| invalid("missing Trace array"))?;
        let snap = &trace[11];
        assert_eq!(snap["Count"], 4);
        let acc = f64::from_bits(
            snap["AccuracyBits"]
                .as_u64()
                .ok_or_else(|| invalid("missing AccuracyBits"))?,
        );
        assert!((acc - 1.0).abs() < 1e-12);
        let acc_unc = f64::from_bits(
            snap["AccuracyUncertaintyBits"]
                .as_u64()
                .ok_or_else(|| invalid("missing AccuracyUncertaintyBits"))?,
        );
        assert!(acc_unc.abs() < 1e-12);

        // UnavailableStorage: reading node 2 before alloc/epoch must fail.
        assert!(
            execute(&wrap_trace(&[
                json!({"Op": "context", "Handle": 0, "Bytes": 4096}),
                json!({"Op": "tensor", "Handle": 0, "Context": 0, "Dimensions": [1], "Parameter": false, "Name": "x"}),
                json!({"Op": "get", "Tensor": 0}),
            ]))
            .is_err()
        );

        // UnavailableStorage: reading compute output after fit must fail.
        assert!(
            execute(&wrap_trace(&[
                json!({"Op": "dataset", "Handle": 0, "DatapointElements": 1, "LabelElements": 1, "Count": 2, "Shard": 1, "DataHex": f32_hex(&[1.0, 2.0]), "LabelsHex": f32_hex(&[2.0, 4.0])}),
                json!({"Op": "context", "Handle": 0, "Bytes": 4096}),
                json!({"Op": "context", "Handle": 1, "Bytes": 65536}),
                json!({"Op": "tensor", "Handle": 0, "Context": 0, "Dimensions": [1, 2], "Parameter": false, "Name": "x"}),
                json!({"Op": "tensor", "Handle": 1, "Context": 0, "Dimensions": [1], "Parameter": true, "Name": "w"}),
                json!({"Op": "node", "Handle": 2, "Context": 1, "Operation": "mul", "Inputs": [0, 1], "FactorBits": null, "Name": "y"}),
                json!({"Op": "allocate", "Handle": 0, "Context": 0}),
                json!({"Op": "set", "Tensor": 1, "DataHex": f32_hex(&[1.0])}),
                json!({"Op": "fit", "Context": 1, "Inputs": 0, "Outputs": 2, "Dataset": 0, "Loss": 3, "Optimizer": 1, "Epochs": 2, "LogicalBatch": 2, "ValidationBits": 0_f32.to_bits(), "Silent": true, "Parameters": null}),
                json!({"Op": "get", "Tensor": 2}),
            ]))
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn independent_optimizer_rejects_malformed_requests_and_lifecycle_violations() {
        let valid_info = json!({"SchemaVersion": 1, "Operations": [{"Kind": "optimizer-info"}]});
        assert!(handles(&valid_info));
        assert!(execute(&valid_info).is_ok());

        for bad in [
            json!({"SchemaVersion": 2, "Operations": [{"Kind": "optimizer-info"}]}),
            json!({"SchemaVersion": 1, "Operations": []}),
            json!({"SchemaVersion": 1, "Operations": [{"Kind": "optimizer-info", "Extra": 1}]}),
            json!({"SchemaVersion": 1, "Operations": [{"Kind": "unknown"}]}),
            json!({
                "SchemaVersion": 1,
                "Operations": [{
                    "Kind": "optimizer-trace",
                    "Devices": [1],
                    "Scheduler": [0],
                    "Backend": 0,
                    "Threads": 1,
                    "Actions": []
                }]
            }),
            json!({
                "SchemaVersion": 1,
                "Operations": [{
                    "Kind": "optimizer-trace",
                    "Devices": [0],
                    "Scheduler": [1],
                    "Backend": 0,
                    "Threads": 1,
                    "Actions": []
                }]
            }),
            json!({
                "SchemaVersion": 1,
                "Operations": [{
                    "Kind": "optimizer-trace",
                    "Devices": [0],
                    "Scheduler": [0],
                    "Backend": 1,
                    "Threads": 1,
                    "Actions": []
                }]
            }),
            json!({
                "SchemaVersion": 1,
                "Operations": [{
                    "Kind": "optimizer-trace",
                    "Devices": [0],
                    "Scheduler": [0],
                    "Backend": 0,
                    "Threads": 257,
                    "Actions": []
                }]
            }),
            wrap_trace(&[json!({"Op": "context", "Handle": 0, "Bytes": 0})]),
            wrap_trace(&[
                json!({"Op": "context", "Handle": 0, "Bytes": 4096}),
                json!({"Op": "tensor", "Handle": 0, "Context": 0, "Dimensions": [1], "Parameter": false, "Name": "a\0b"}),
            ]),
            wrap_trace(&[
                json!({"Op": "context", "Handle": 0, "Bytes": 4096}),
                json!({"Op": "tensor", "Handle": 0, "Context": 0, "Dimensions": [2], "Parameter": false, "Name": "a"}),
                json!({"Op": "node", "Handle": 1, "Context": 0, "Operation": "scale", "Inputs": [0], "FactorBits": f32::NAN.to_bits(), "Name": "s"}),
            ]),
            wrap_trace(&[
                json!({"Op": "context", "Handle": 0, "Bytes": 4096}),
                json!({"Op": "tensor", "Handle": 0, "Context": 0, "Dimensions": [2], "Parameter": false, "Name": "a"}),
                json!({"Op": "tensor", "Handle": 1, "Context": 0, "Dimensions": [3], "Parameter": false, "Name": "b"}),
                json!({"Op": "node", "Handle": 2, "Context": 0, "Operation": "add", "Inputs": [0, 1], "FactorBits": null, "Name": "bad"}),
            ]),
            wrap_trace(&[
                json!({"Op": "context", "Handle": 0, "Bytes": 4096}),
                json!({"Op": "tensor", "Handle": 0, "Context": 0, "Dimensions": [1], "Parameter": false, "Name": "a"}),
                json!({"Op": "allocate", "Handle": 0, "Context": 0}),
                json!({"Op": "set", "Tensor": 0, "DataHex": "0000803F"}),
            ]),
            wrap_trace(&[
                json!({"Op": "context", "Handle": 0, "Bytes": 4096}),
                json!({"Op": "tensor", "Handle": 0, "Context": 0, "Dimensions": [1], "Parameter": false, "Name": "x"}),
                json!({"Op": "tensor", "Handle": 1, "Context": 0, "Dimensions": [1], "Parameter": true, "Name": "w"}),
                json!({"Op": "node", "Handle": 2, "Context": 0, "Operation": "add", "Inputs": [0, 1], "FactorBits": null, "Name": "y"}),
                json!({"Op": "allocate", "Handle": 0, "Context": 0}),
                json!({"Op": "init", "Handle": 0, "Context": 0, "Inputs": 0, "Outputs": 2, "Loss": 1, "Optimizer": 0, "Period": 1, "Parameters": null}),
                json!({"Op": "free", "Type": "buffer", "Handle": 0}),
            ]),
            wrap_trace(&[
                json!({"Op": "context", "Handle": 0, "Bytes": 4096}),
                json!({"Op": "tensor", "Handle": 0, "Context": 0, "Dimensions": [1], "Parameter": false, "Name": "x"}),
                json!({"Op": "tensor", "Handle": 1, "Context": 0, "Dimensions": [1], "Parameter": true, "Name": "w"}),
                json!({"Op": "node", "Handle": 2, "Context": 0, "Operation": "add", "Inputs": [0, 1], "FactorBits": null, "Name": "y"}),
                json!({"Op": "allocate", "Handle": 0, "Context": 0}),
                json!({"Op": "init", "Handle": 0, "Context": 0, "Inputs": 0, "Outputs": 2, "Loss": 1, "Optimizer": 0, "Period": 1, "Parameters": null}),
                json!({"Op": "gradient", "Optimizer": 0, "Handle": 3, "Tensor": 1}),
            ]),
        ] {
            assert!(execute(&bad).is_err(), "expected error for {bad}");
        }
    }
}
