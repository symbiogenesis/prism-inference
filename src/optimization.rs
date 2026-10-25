//! Owned CPU training graphs, sharded datasets and optimizer result statistics.
//!
//! Follows pinned `ggml-opt.cpp` and delegates tensor execution and optimizer
//! primitives to [`crate::backend::CpuBackend`]. Training graphs are ordinary
//! differentiable tensor expressions, independent of source-test workloads.

#![allow(
    clippy::suboptimal_flops,
    clippy::float_cmp,
    reason = "Preserve native optimization operation ordering and exact identity parameters."
)]

use crate::backend::{Argument, CpuBackend, GraphId, Observation, TensorId};
use crate::sampling::upstream::Random;
use crate::{Result, invalid};
use std::collections::{BTreeMap, BTreeSet};

/// Differentiable expressions and direct optimizer step nodes.
#[derive(Clone, Copy, Debug)]
pub enum Operator {
    /// Broadcast addition.
    Add,
    /// Broadcast multiplication.
    Multiply,
    /// Scalar multiplication.
    Scale(f32),
    /// In-place `AdamW` primitive with weight, gradient, moment, variance and parameters.
    AdamW,
    /// In-place SGD primitive with weight, gradient and parameters.
    Sgd,
}

#[derive(Clone)]
struct Node {
    operator: Option<Operator>,
    inputs: Vec<TensorId>,
    parameter: bool,
}

/// An owning CPU tensor graph, including metadata, leaf values and aliases.
#[derive(Default)]
pub struct Graph {
    backend: CpuBackend,
    nodes: BTreeMap<TensorId, Node>,
}

impl Graph {
    /// Creates an empty training graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an owned bounded F32 tensor, optionally trainable.
    ///
    /// # Errors
    /// Rejects invalid names/shapes or allocation failures.
    pub fn tensor(
        &mut self,
        context: u32,
        dimensions: &[usize],
        parameter: bool,
        name: &str,
    ) -> Result<TensorId> {
        if dimensions.is_empty()
            || dimensions.len() > 4
            || dimensions.iter().any(|&value| value == 0 || value > 65536)
        {
            return Err(invalid("invalid training tensor shape"));
        }
        let count = dimensions
            .iter()
            .try_fold(1_usize, |count, &value| count.checked_mul(value))
            .filter(|&count| count <= 262_144)
            .ok_or_else(|| invalid("training tensor exceeds element bound"))?;
        let mut shape = [1; 4];
        shape[..dimensions.len()].copy_from_slice(dimensions);
        let id = self.backend.tensor(context, 0, shape)?;
        let info = self.backend.info_mut(id)?;
        info.name = tensor_name(name)?;
        if parameter {
            info.flags |= 16;
        }
        self.nodes.insert(
            id,
            Node {
                operator: None,
                inputs: vec![],
                parameter,
            },
        );
        self.backend.set_values(id, &vec![0.0; count])?;
        Ok(id)
    }

    /// Adds an expression using the production backend's shape and alias rules.
    ///
    /// # Errors
    /// Rejects missing inputs, malformed arity, invalid broadcast shapes or names.
    pub fn operation(
        &mut self,
        context: u32,
        operator: Operator,
        inputs: &[TensorId],
        name: &str,
    ) -> Result<TensorId> {
        let (function, count) = match operator {
            Operator::Add => ("ggml_add", 2),
            Operator::Multiply => ("ggml_mul", 2),
            Operator::Scale(_) => ("ggml_scale", 1),
            Operator::AdamW => ("ggml_opt_step_adamw", 5),
            Operator::Sgd => ("ggml_opt_step_sgd", 3),
        };
        if inputs.len() != count || inputs.iter().any(|input| !self.nodes.contains_key(input)) {
            return Err(invalid(
                "training operation has unknown inputs or invalid arity",
            ));
        }
        let mut arguments = vec![Argument::Context(context)];
        arguments.extend(inputs.iter().copied().map(Argument::Tensor));
        if let Operator::Scale(factor) = operator {
            if !factor.is_finite() {
                return Err(invalid("training scale must be finite"));
            }
            arguments.push(Argument::Float(factor));
        }
        let id = self.call_tensor(function, &arguments)?;
        self.backend.info_mut(id)?.name = tensor_name(name)?;
        self.nodes.insert(
            id,
            Node {
                operator: Some(operator),
                inputs: inputs.to_vec(),
                parameter: false,
            },
        );
        Ok(id)
    }

    /// Reads an independent copy of tensor values.
    ///
    /// # Errors
    /// Rejects an unknown tensor identity.
    pub fn values(&self, tensor: TensorId) -> Result<Vec<f32>> {
        self.backend.values(tensor)
    }

    /// Replaces tensor values without changing shape or aliases.
    ///
    /// # Errors
    /// Rejects an unknown tensor or a mismatched value extent.
    pub fn set_values(&mut self, tensor: TensorId, values: &[f32]) -> Result<()> {
        self.backend.set_values(tensor, values)
    }

    /// Returns four native logical dimensions.
    ///
    /// # Errors
    /// Rejects an unknown tensor identity.
    pub fn shape(&self, tensor: TensorId) -> Result<[usize; 4]> {
        Ok(self.backend.info(tensor)?.shape)
    }

    /// Queries implemented production support for this tensor operation.
    ///
    /// # Errors
    /// Rejects an unknown tensor identity.
    pub fn supports(&self, tensor: TensorId) -> Result<bool> {
        self.backend.supports(tensor)
    }

    fn call_tensor(&mut self, name: &str, arguments: &[Argument]) -> Result<TensorId> {
        match self.backend.call(name, arguments)? {
            Observation::Tensor(Some(tensor)) => Ok(tensor),
            _ => Err(invalid("training backend did not produce a tensor")),
        }
    }
    fn execution(&mut self, context: u32, outputs: &[TensorId]) -> Result<GraphId> {
        let Observation::Graph(graph) = self
            .backend
            .call("ggml_new_graph", &[Argument::Context(context)])?
        else {
            return Err(invalid("training backend did not produce a graph"));
        };
        for &output in outputs {
            self.backend.call(
                "ggml_build_forward_expand",
                &[Argument::Graph(graph), Argument::Tensor(output)],
            )?;
        }
        Ok(graph)
    }
    fn topology(&self, output: TensorId) -> Result<Vec<TensorId>> {
        fn visit(
            graph: &Graph,
            tensor: TensorId,
            visited: &mut BTreeSet<TensorId>,
            order: &mut Vec<TensorId>,
        ) -> Result<()> {
            if !visited.insert(tensor) {
                return Ok(());
            }
            let node = graph
                .nodes
                .get(&tensor)
                .ok_or_else(|| invalid("unknown differentiable training tensor"))?;
            for &input in &node.inputs {
                visit(graph, input, visited, order)?;
            }
            order.push(tensor);
            Ok(())
        }
        let mut order = Vec::new();
        visit(self, output, &mut BTreeSet::new(), &mut order)?;
        Ok(order)
    }
}

fn tensor_name(name: &str) -> Result<String> {
    if name.len() > 128 || name.contains('\0') {
        return Err(invalid("invalid training tensor name"));
    }
    Ok(String::from_utf8(
        name.as_bytes()[..name.len().min(63)].to_vec(),
    )?)
}

/// Sharded dataset with independently owned input and label bytes.
#[derive(Clone, Debug)]
pub struct Dataset {
    width: usize,
    label_width: usize,
    count: usize,
    shard: usize,
    data: Vec<f32>,
    labels: Vec<f32>,
    permutation: Vec<usize>,
}
impl Dataset {
    /// Constructs a bounded dataset, retaining only complete shards in its permutation.
    ///
    /// # Errors
    /// Rejects invalid dimensions or mismatched data/label extents.
    pub fn new(
        width: usize,
        label_width: usize,
        count: usize,
        shard: usize,
        data: Vec<f32>,
        labels: Vec<f32>,
    ) -> Result<Self> {
        if width == 0
            || width > 64
            || label_width > 64
            || count == 0
            || count > 4096
            || shard == 0
            || shard > count
            || count.checked_mul(width) != Some(data.len())
            || count.checked_mul(label_width) != Some(labels.len())
        {
            return Err(invalid("invalid optimizer dataset dimensions or payload"));
        }
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
    /// Number of source datapoints, including any incomplete trailing shard.
    #[must_use]
    pub const fn count(&self) -> usize {
        self.count
    }
    /// Data elements per datapoint.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }
    /// Label elements per datapoint; zero means no labels.
    #[must_use]
    pub const fn label_width(&self) -> usize {
        self.label_width
    }
    /// Copies one physical batch in current shard permutation order.
    ///
    /// # Errors
    /// Rejects incomplete shard batches and indices outside retained complete shards.
    pub fn batch(&self, count: usize, index: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        if count == 0
            || !count.is_multiple_of(self.shard)
            || index
                .checked_add(1)
                .and_then(|index| index.checked_mul(count / self.shard))
                .is_none_or(|end| end > self.permutation.len())
        {
            return Err(invalid("optimizer batch exceeds complete dataset shards"));
        }
        let shards = count / self.shard;
        let mut data = Vec::with_capacity(count * self.width);
        let mut labels = Vec::with_capacity(count * self.label_width);
        for &source in &self.permutation[index * shards..(index + 1) * shards] {
            let begin = source * self.shard;
            data.extend(&self.data[begin * self.width..(begin + self.shard) * self.width]);
            labels.extend(
                &self.labels[begin * self.label_width..(begin + self.shard) * self.label_width],
            );
        }
        Ok((data, labels))
    }
    fn shuffle(&mut self, random: &mut Random, count: Option<usize>) -> Result<()> {
        let shards = if let Some(count) = count {
            if count > self.count || !count.is_multiple_of(self.shard) {
                return Err(invalid("dataset shuffle count must cover complete shards"));
            }
            count / self.shard
        } else {
            self.permutation.len()
        };
        for index in 1..shards {
            let selected = random.bounded(index + 1)?;
            self.permutation.swap(index, selected);
        }
        Ok(())
    }
}

/// Production optimizer loss convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Loss {
    /// Mean of all output elements.
    Mean,
    /// Sum of all output elements.
    Sum,
    /// Mean per-row cross entropy.
    CrossEntropy,
    /// Mean of all squared output/label errors.
    MeanSquaredError,
}

/// Implemented CPU optimizer algorithm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Adam with decoupled weight decay.
    AdamW,
    /// Stochastic gradient descent with decoupled weight decay.
    Sgd,
}

/// Native step parameter arrays, with both algorithms retained independently.
#[derive(Clone, Copy, Debug)]
pub struct Parameters {
    /// Alpha, beta1, beta2, epsilon and weight decay.
    pub adamw: [f32; 5],
    /// Alpha and weight decay.
    pub sgd: [f32; 2],
}
impl Default for Parameters {
    fn default() -> Self {
        Self {
            adamw: [0.001, 0.9, 0.999, 1e-8, 0.0],
            sgd: [0.001, 0.0],
        }
    }
}

/// Static differentiable graph and optimizer configuration.
#[derive(Clone, Copy, Debug)]
pub struct Training {
    /// Compute metadata context.
    pub context: u32,
    /// Physical batch input tensor.
    pub inputs: TensorId,
    /// Differentiable output tensor.
    pub outputs: TensorId,
    /// Loss convention.
    pub loss: Loss,
    /// Optimizer algorithm.
    pub kind: Kind,
    /// Physical backward evaluations per parameter update.
    pub period: usize,
    /// Current constant parameters.
    pub parameters: Parameters,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Forward,
    Gradient,
    Update,
}

struct Trainable {
    tensor: TensorId,
    gradient: TensorId,
    moment: Option<TensorId>,
    variance: Option<TensorId>,
}

/// Stateful static-graph optimizer with owned momenta and gradient accumulators.
pub struct Optimizer {
    training: Training,
    topology: Vec<TensorId>,
    trainable: Vec<Trainable>,
    loss: TensorId,
    labels: Option<TensorId>,
    forward: GraphId,
    update: GraphId,
    step_parameters: TensorId,
    mode: Mode,
    ready: bool,
    iteration: usize,
    index: usize,
    random: Random,
}

impl Optimizer {
    /// Builds production forward/step graphs and independent gradient/momentum storage.
    ///
    /// # Errors
    /// Rejects invalid identities, unsupported differentiable operations, periods,
    /// parameters, shapes or backend graph construction failures.
    pub fn new(graph: &mut Graph, training: Training) -> Result<Self> {
        if training.period == 0 || training.period > 1_000_000 {
            return Err(invalid("invalid optimizer accumulation period"));
        }
        validate_parameters(training.parameters, training.kind)?;
        let topology = graph.topology(training.outputs)?;
        graph.shape(training.inputs)?;
        let shape = graph.shape(training.outputs)?;
        let labels = if matches!(training.loss, Loss::CrossEntropy | Loss::MeanSquaredError) {
            Some(graph.tensor(training.context, &shape, false, "labels")?)
        } else {
            None
        };
        let loss = build_loss(graph, training, labels)?;
        let forward = graph.execution(training.context, &[loss])?;
        let step_parameters = graph.tensor(
            training.context,
            &[if training.kind == Kind::AdamW { 7 } else { 2 }],
            false,
            "optimizer parameters",
        )?;
        let mut trainable = Vec::new();
        let mut steps = Vec::new();
        for &tensor in topology.iter().rev() {
            let node = graph
                .nodes
                .get(&tensor)
                .ok_or_else(|| invalid("missing differentiable node"))?;
            if matches!(node.operator, Some(Operator::AdamW | Operator::Sgd)) {
                return Err(invalid("optimizer step nodes have no backward operator"));
            }
            if !node.parameter {
                continue;
            }
            let shape = graph.shape(tensor)?;
            let gradient = graph.tensor(training.context, &shape, false, "gradient accumulator")?;
            let (moment, variance, step) = if training.kind == Kind::AdamW {
                let moment = graph.tensor(training.context, &shape, false, "AdamW moment")?;
                let variance = graph.tensor(training.context, &shape, false, "AdamW variance")?;
                let step = graph.operation(
                    training.context,
                    Operator::AdamW,
                    &[tensor, gradient, moment, variance, step_parameters],
                    "AdamW step",
                )?;
                (Some(moment), Some(variance), step)
            } else {
                (
                    None,
                    None,
                    graph.operation(
                        training.context,
                        Operator::Sgd,
                        &[tensor, gradient, step_parameters],
                        "SGD step",
                    )?,
                )
            };
            steps.push(step);
            trainable.push(Trainable {
                tensor,
                gradient,
                moment,
                variance,
            });
        }
        let update = graph.execution(training.context, &steps)?;
        Ok(Self {
            training,
            topology,
            trainable,
            loss,
            labels,
            forward,
            update,
            step_parameters,
            mode: Mode::Update,
            ready: false,
            iteration: 1,
            index: 0,
            random: Random::new(5489),
        })
    }

    /// Loss observation tensor owned by the training graph.
    #[must_use]
    pub const fn loss_tensor(&self) -> TensorId {
        self.loss
    }
    /// Optional label input tensor owned by the training graph.
    #[must_use]
    pub const fn labels_tensor(&self) -> Option<TensorId> {
        self.labels
    }
    /// Current algorithm.
    #[must_use]
    pub const fn kind(&self) -> Kind {
        self.training.kind
    }

    /// Returns a native persistent gradient accumulator when accumulation is enabled.
    ///
    /// # Errors
    /// Rejects nonparameters and period-one optimizers, which have no accumulator.
    pub fn gradient_tensor(&self, tensor: TensorId) -> Result<TensorId> {
        if self.training.period <= 1 {
            return Err(invalid(
                "period-one optimizer has no persistent gradient accumulator",
            ));
        }
        self.trainable
            .iter()
            .find(|parameter| parameter.tensor == tensor)
            .map(|parameter| parameter.gradient)
            .ok_or_else(|| invalid("tensor has no optimizer gradient accumulator"))
    }

    /// Selects and prepares the next forward, gradient or parameter-update graph.
    ///
    /// # Errors
    /// Rejects a second allocation before evaluation and invalid retained tensors.
    pub fn allocate(&mut self, graph: &mut Graph, backward: bool) -> Result<()> {
        if self.ready {
            return Err(invalid("optimizer graph is already awaiting evaluation"));
        }
        if self.mode == Mode::Update && self.training.period > 1 && self.index == 0 {
            self.clear_gradients(graph)?;
        }
        self.mode = if !backward {
            Mode::Forward
        } else if (self.index + 1).is_multiple_of(self.training.period) {
            Mode::Update
        } else {
            Mode::Gradient
        };
        self.ready = true;
        Ok(())
    }

    /// Clears accumulated gradients and optionally momenta/update iteration.
    ///
    /// # Errors
    /// Returns retained tensor storage errors.
    pub fn reset(&mut self, graph: &mut Graph, optimizer: bool) -> Result<()> {
        self.clear_gradients(graph)?;
        if optimizer {
            self.iteration = 1;
            for parameter in &self.trainable {
                for tensor in [parameter.moment, parameter.variance].into_iter().flatten() {
                    graph.set_values(tensor, &vec![0.0; graph.values(tensor)?.len()])?;
                }
            }
        }
        Ok(())
    }

    /// Evaluates forward outputs, optional gradients/updates, and result observations.
    ///
    /// # Errors
    /// Rejects evaluation without allocation, invalid derivative shapes, incompatible
    /// result conventions and backend execution or parameter errors.
    pub fn evaluate(&mut self, graph: &mut Graph, result: Option<&mut Metrics>) -> Result<()> {
        if !self.ready {
            return Err(invalid("optimizer evaluation requires graph allocation"));
        }
        graph.backend.compute(self.forward)?;
        let correct = if result.is_some() && self.training.loss == Loss::CrossEntropy {
            let shape = graph.shape(self.training.outputs)?;
            let values = graph.values(self.training.outputs)?;
            let labels = graph.values(
                self.labels
                    .ok_or_else(|| invalid("missing classification labels"))?,
            )?;
            Some(
                values
                    .chunks_exact(shape[0])
                    .zip(labels.chunks_exact(shape[0]))
                    .filter(|(values, labels)| maximum_index(values) == maximum_index(labels))
                    .count(),
            )
        } else {
            None
        };
        if self.mode != Mode::Forward {
            let gradient = output_gradient(graph, self.training, self.labels)?;
            let gradients = backward(graph, &self.topology, self.training.outputs, gradient)?;
            for parameter in &self.trainable {
                let next = gradients
                    .get(&parameter.tensor)
                    .ok_or_else(|| invalid("missing trainable gradient"))?;
                let mut accumulated = graph.values(parameter.gradient)?;
                for (value, &gradient) in accumulated.iter_mut().zip(next) {
                    if self.training.period > 1 {
                        *value += gradient;
                    } else {
                        *value = gradient;
                    }
                }
                graph.set_values(parameter.gradient, &accumulated)?;
            }
            if self.mode == Mode::Update {
                self.prepare_step(graph)?;
                graph.backend.compute(self.update)?;
                self.iteration += 1;
            }
        }
        if let Some(result) = result {
            let shape = graph.shape(self.training.outputs)?;
            result.record(
                shape[1],
                graph.values(self.loss)?[0],
                correct,
                self.training.period,
                self.training.loss != Loss::Sum,
            )?;
        }
        self.index = (self.index + 1) % self.training.period;
        self.ready = false;
        Ok(())
    }

    /// Shuffles complete dataset shards with the optimizer's persistent native RNG.
    ///
    /// # Errors
    /// Rejects counts that exceed the dataset or split a shard.
    pub fn shuffle(&mut self, dataset: &mut Dataset, count: Option<usize>) -> Result<()> {
        dataset.shuffle(&mut self.random, count)
    }

    /// Runs the training prefix and evaluation suffix of one physical-batch epoch.
    ///
    /// # Errors
    /// Rejects incompatible batch, dataset, label and split extents.
    pub fn epoch(
        &mut self,
        graph: &mut Graph,
        dataset: &Dataset,
        mut train: Option<&mut Metrics>,
        mut evaluate: Option<&mut Metrics>,
        split: usize,
    ) -> Result<()> {
        let shape = graph.shape(self.training.inputs)?;
        let physical = shape[1];
        if dataset.width != shape[0]
            || shape[2..] != [1, 1]
            || !dataset.count.is_multiple_of(physical)
            || split > dataset.count
            || !split.is_multiple_of(physical)
        {
            return Err(invalid(
                "optimizer epoch has incompatible data or split shape",
            ));
        }
        for index in 0..dataset.count / physical {
            let backward = index < split / physical;
            self.allocate(graph, backward)?;
            let (data, labels) = dataset.batch(physical, index)?;
            graph.set_values(self.training.inputs, &data)?;
            if let Some(tensor) = self.labels {
                graph.set_values(tensor, &labels)?;
            } else if !labels.is_empty() {
                return Err(invalid("optimizer has no label input"));
            }
            self.evaluate(
                graph,
                if backward {
                    train.as_deref_mut()
                } else {
                    evaluate.as_deref_mut()
                },
            )?;
        }
        Ok(())
    }

    fn clear_gradients(&self, graph: &mut Graph) -> Result<()> {
        for parameter in &self.trainable {
            graph.set_values(
                parameter.gradient,
                &vec![0.0; graph.values(parameter.gradient)?.len()],
            )?;
        }
        Ok(())
    }
    #[allow(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "Native AdamW powf converts the integer update iteration to binary32."
    )]
    fn prepare_step(&self, graph: &mut Graph) -> Result<()> {
        validate_parameters(self.training.parameters, self.training.kind)?;
        if self.training.kind == Kind::AdamW {
            let [alpha, beta1, beta2, epsilon, decay] = self.training.parameters.adamw;
            graph.set_values(
                self.step_parameters,
                &[
                    alpha,
                    beta1,
                    beta2,
                    epsilon,
                    decay,
                    1.0 / (1.0 - beta1.powf(self.iteration as f32)),
                    1.0 / (1.0 - beta2.powf(self.iteration as f32)),
                ],
            )
        } else {
            graph.set_values(self.step_parameters, &self.training.parameters.sgd)
        }
    }
}

fn maximum_index(values: &[f32]) -> usize {
    let mut maximum = 0;
    for index in 1..values.len() {
        if values[index] > values[maximum] {
            maximum = index;
        }
    }
    maximum
}

fn validate_parameters(parameters: Parameters, kind: Kind) -> Result<()> {
    if parameters
        .adamw
        .iter()
        .chain(&parameters.sgd)
        .any(|value| !value.is_finite())
    {
        return Err(invalid("optimizer parameters must be finite"));
    }
    let valid = if kind == Kind::AdamW {
        let [alpha, beta1, beta2, epsilon, decay] = parameters.adamw;
        alpha > 0.0
            && (0.0..=1.0).contains(&beta1)
            && (0.0..=1.0).contains(&beta2)
            && epsilon >= 0.0
            && (0.0..=1.0).contains(&decay)
    } else {
        parameters.sgd[0] > 0.0 && (0.0..=1.0).contains(&parameters.sgd[1])
    };
    if !valid {
        return Err(invalid("optimizer step parameters violate native bounds"));
    }
    Ok(())
}

#[allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "Native loss scaling converts period and output element count to binary32."
)]
fn build_loss(graph: &mut Graph, training: Training, labels: Option<TensorId>) -> Result<TensorId> {
    let mut input = training.outputs;
    let call = |graph: &mut Graph, name: &str, inputs: &[TensorId]| {
        let mut arguments = vec![Argument::Context(training.context)];
        arguments.extend(inputs.iter().copied().map(Argument::Tensor));
        graph.call_tensor(name, &arguments)
    };
    if let Some(labels) = labels {
        input = call(
            graph,
            if training.loss == Loss::CrossEntropy {
                "ggml_cross_entropy_loss"
            } else {
                "ggml_sub"
            },
            &[input, labels],
        )?;
        if training.loss == Loss::MeanSquaredError {
            input = call(graph, "ggml_sqr", &[input])?;
        }
    }
    if training.loss != Loss::CrossEntropy {
        input = call(graph, "ggml_sum", &[input])?;
    }
    let scale = match training.loss {
        Loss::Sum => return Ok(input),
        Loss::CrossEntropy => 1.0_f32 / training.period as f32,
        Loss::Mean | Loss::MeanSquaredError => {
            1.0_f32
                / (training.period * graph.shape(training.outputs)?.iter().product::<usize>())
                    as f32
        }
    };
    graph.call_tensor(
        "ggml_scale",
        &[
            Argument::Context(training.context),
            Argument::Tensor(input),
            Argument::Float(scale),
        ],
    )
}

#[allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "Native loss gradients use binary32 period/shape scaling and rounded softmax sums."
)]
fn output_gradient(
    graph: &Graph,
    training: Training,
    labels: Option<TensorId>,
) -> Result<Vec<f32>> {
    let output = graph.values(training.outputs)?;
    let scale = match training.loss {
        Loss::Sum => 1.0,
        Loss::CrossEntropy => 1.0 / (training.period * graph.shape(training.outputs)?[1]) as f32,
        _ => 1.0 / (training.period * output.len()) as f32,
    };
    match training.loss {
        Loss::Mean | Loss::Sum => Ok(vec![scale; output.len()]),
        Loss::MeanSquaredError => Ok(output
            .iter()
            .zip(graph.values(labels.ok_or_else(|| invalid("missing MSE labels"))?)?)
            .map(|(&value, label)| (value - label) * scale * 2.0)
            .collect()),
        Loss::CrossEntropy => {
            let labels =
                graph.values(labels.ok_or_else(|| invalid("missing cross entropy labels"))?)?;
            let width = graph.shape(training.outputs)?[0];
            let mut gradient = vec![0.0; output.len()];
            for ((values, labels), gradient) in output
                .chunks_exact(width)
                .zip(labels.chunks_exact(width))
                .zip(gradient.chunks_exact_mut(width))
            {
                let maximum = values[maximum_index(values)];
                let sum = values
                    .iter()
                    .map(|value| f64::from((*value - maximum).exp()))
                    .sum::<f64>() as f32;
                for ((&value, &label), gradient) in values.iter().zip(labels).zip(gradient) {
                    *gradient = ((value - maximum).exp() / sum - label) * scale;
                }
            }
            Ok(gradient)
        }
    }
}

fn repeated_index(index: usize, output: [usize; 4], input: [usize; 4]) -> usize {
    let mut remaining = index;
    let mut stride = 1;
    let mut result = 0;
    for axis in 0..4 {
        result += (remaining % output[axis] % input[axis]) * stride;
        remaining /= output[axis];
        stride *= input[axis];
    }
    result
}

fn backward(
    graph: &Graph,
    topology: &[TensorId],
    output: TensorId,
    seed: Vec<f32>,
) -> Result<BTreeMap<TensorId, Vec<f32>>> {
    let mut gradients = BTreeMap::from([(output, seed)]);
    for &tensor in topology.iter().rev() {
        let node = graph
            .nodes
            .get(&tensor)
            .ok_or_else(|| invalid("unknown backward node"))?;
        let Some(operator) = node.operator else {
            continue;
        };
        let Some(gradient) = gradients.get(&tensor).cloned() else {
            continue;
        };
        let shape = graph.shape(tensor)?;
        for (position, &input) in node.inputs.iter().enumerate() {
            let input_shape = graph.shape(input)?;
            let values = graph.values(input)?;
            let target = gradients
                .entry(input)
                .or_insert_with(|| vec![0.0; values.len()]);
            let other = if matches!(operator, Operator::Multiply) {
                Some((
                    graph.values(node.inputs[1 - position])?,
                    graph.shape(node.inputs[1 - position])?,
                ))
            } else {
                None
            };
            for (index, &gradient) in gradient.iter().enumerate() {
                let value = match operator {
                    Operator::Add => gradient,
                    Operator::Scale(factor) => gradient * factor,
                    Operator::Multiply => {
                        let (values, shape_other) = other
                            .as_ref()
                            .ok_or_else(|| invalid("missing multiply derivative"))?;
                        gradient * values[repeated_index(index, shape, *shape_other)]
                    }
                    Operator::AdamW | Operator::Sgd => {
                        return Err(invalid("optimizer step has no backward operator"));
                    }
                };
                target[repeated_index(index, shape, input_shape)] += value;
            }
        }
    }
    Ok(gradients)
}

/// High-level fitting schedule.
#[derive(Clone, Debug)]
pub struct Fit {
    /// Number of epochs.
    pub epochs: usize,
    /// Datapoints per logical optimizer update.
    pub logical_batch: usize,
    /// Validation fraction, between zero inclusive and one exclusive.
    pub validation_split: f32,
    /// Optional constant optimizer parameters for each epoch.
    pub parameters: Option<Vec<Parameters>>,
}

/// Fits an arbitrary supported static graph with a native epoch/accumulation schedule.
///
/// # Errors
/// Rejects invalid epochs, physical/logical batch geometry, validation splits,
/// parameter schedules and graph/data execution failures.
#[allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "Native fitting truncates its finite binary32 logical-batch split expression."
)]
pub fn fit(
    graph: &mut Graph,
    mut training: Training,
    dataset: &mut Dataset,
    options: &Fit,
) -> Result<()> {
    let physical = graph.shape(training.inputs)?[1];
    if options.epochs == 0
        || options.epochs > 4096
        || options.logical_batch == 0
        || options.logical_batch > 4096
        || !dataset.count.is_multiple_of(options.logical_batch)
        || !options.logical_batch.is_multiple_of(physical)
        || !options.validation_split.is_finite()
        || !(0.0..1.0).contains(&options.validation_split)
        || options
            .parameters
            .as_ref()
            .is_some_and(|parameters| parameters.len() != options.epochs)
    {
        return Err(invalid("invalid optimizer fitting schedule"));
    }
    training.period = options.logical_batch / physical;
    let split = (((1.0_f32 - options.validation_split)
        * (dataset.count / options.logical_batch) as f32) as usize)
        * options.logical_batch;
    let mut optimizer = Optimizer::new(graph, training)?;
    if options.logical_batch < dataset.count {
        optimizer.shuffle(dataset, None)?;
    }
    for epoch in 0..options.epochs {
        if let Some(parameters) = &options.parameters {
            optimizer.training.parameters = parameters[epoch];
        }
        if options.logical_batch < split {
            optimizer.shuffle(dataset, Some(split))?;
        }
        optimizer.epoch(graph, dataset, None, None, split)?;
    }
    Ok(())
}

/// Aggregated native optimizer statistics, retaining each physical batch loss.
#[derive(Clone, Debug, Default)]
pub struct Metrics {
    count: usize,
    losses: Vec<f32>,
    correct: Option<usize>,
    period: usize,
    per_datapoint: bool,
}
impl Metrics {
    /// Appends compatible physical-batch observations in their original order.
    ///
    /// # Errors
    /// Rejects differing loss conventions, periods or physical batch sizes.
    pub fn append(&mut self, other: &Self) -> Result<()> {
        if other.count == 0 {
            return Ok(());
        }
        if self.count == 0 {
            *self = other.clone();
            return Ok(());
        }
        if self.period != other.period
            || self.per_datapoint != other.per_datapoint
            || self.count / self.losses.len() != other.count / other.losses.len()
        {
            return Err(invalid(
                "optimizer statistics have incompatible batch conventions",
            ));
        }
        self.count += other.count;
        self.losses.extend(&other.losses);
        self.correct = self
            .correct
            .zip(other.correct)
            .map(|(left, right)| left + right);
        Ok(())
    }
    /// Clears accumulated observations.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
    /// Number of observed output rows.
    #[must_use]
    pub const fn count(&self) -> usize {
        self.count
    }
    /// Native aggregated loss and standard uncertainty.
    #[must_use]
    #[allow(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "Native statistics convert bounded counts and periods to floating point."
    )]
    pub fn loss(&self) -> (f64, f64) {
        if self.losses.is_empty() {
            return (0.0, f64::NAN);
        }
        let mut sum = 0.0_f64;
        let mut sum_squared = 0.0_f64;
        for &loss in &self.losses {
            let scaled = if self.per_datapoint {
                loss * self.period as f32
            } else {
                loss
            };
            sum += f64::from(scaled);
            sum_squared += f64::from(scaled * scaled);
        }
        let batches = self.losses.len() as f64;
        let mean = sum / batches;
        let loss = if self.per_datapoint { mean } else { sum };
        let mean_squared = mean * mean;
        let variance = sum_squared / batches - mean_squared;
        let uncertainty = if self.losses.len() < 2 {
            f64::NAN
        } else if self.per_datapoint {
            (variance / (batches - 1.0)).sqrt()
        } else {
            (variance * batches / (batches - 1.0)).sqrt()
        };
        (loss, uncertainty)
    }
    /// Classification accuracy and binomial uncertainty, otherwise NaN.
    #[must_use]
    #[allow(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "Native accuracy uses binary64 count ratios."
    )]
    pub fn accuracy(&self) -> (f64, f64) {
        let accuracy = self
            .correct
            .map_or(f64::NAN, |count| count as f64 / self.count as f64);
        let uncertainty = if self.correct.is_some() && self.count >= 2 {
            (accuracy * (1.0 - accuracy) / (self.count - 1) as f64).sqrt()
        } else {
            f64::NAN
        };
        (accuracy, uncertainty)
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
        if self.period != period
            || self.per_datapoint != per_datapoint
            || self.count != count * self.losses.len()
        {
            return Err(invalid(
                "optimizer result mixes incompatible batch conventions",
            ));
        }
        self.count += count;
        self.losses.push(loss);
        self.correct = self.correct.zip(correct).map(|(total, next)| total + next);
        Ok(())
    }
}
