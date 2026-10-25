//! Thin descriptor/lifetime transport for the production training API.

use super::{
    Result, Value, array, boolean, exact_fields, field, float_hex, floats, invalid, json, uint,
};
use bonsai_prism_runtime::backend::{TensorId, TensorInfo};
use bonsai_prism_runtime::optimization::{
    Dataset, Fit, Graph, Kind, Loss, Metrics, Operator, Optimizer, Parameters, Training, fit,
};
use std::collections::{BTreeMap, BTreeSet};

fn number(value: &Value, name: &str, maximum: usize) -> Result<usize> {
    let number = usize::try_from(uint(value, name)?)?;
    if number > maximum {
        return Err(invalid("optimizer integer exceeds bound"));
    }
    Ok(number)
}
fn handle(value: &Value, name: &str) -> Result<u32> {
    Ok(u32::try_from(number(value, name, 4095)?)?)
}
fn text<'a>(value: &'a Value, name: &str) -> Result<&'a str> {
    field(value, name)?
        .as_str()
        .ok_or_else(|| invalid("optimizer text required"))
}
fn float(value: &Value) -> Result<f32> {
    let value = f32::from_bits(u32::try_from(
        value
            .as_u64()
            .ok_or_else(|| invalid("optimizer float bits must be unsigned"))?,
    )?);
    if !value.is_finite() {
        return Err(invalid("optimizer parameter must be finite"));
    }
    Ok(value)
}
fn parameters(value: &Value) -> Result<Parameters> {
    if value.is_null() {
        return Ok(Parameters::default());
    }
    exact_fields(value, &["AdamW", "Sgd"])?;
    let adam = array(value, "AdamW")?
        .iter()
        .map(float)
        .collect::<Result<Vec<_>>>()?;
    let sgd = array(value, "Sgd")?
        .iter()
        .map(float)
        .collect::<Result<Vec<_>>>()?;
    Ok(Parameters {
        adamw: adam
            .try_into()
            .map_err(|_| invalid("AdamW parameters require five values"))?,
        sgd: sgd
            .try_into()
            .map_err(|_| invalid("SGD parameters require two values"))?,
    })
}
fn loss(value: &Value) -> Result<Loss> {
    Ok(match number(value, "Loss", 3)? {
        0 => Loss::Mean,
        1 => Loss::Sum,
        2 => Loss::CrossEntropy,
        _ => Loss::MeanSquaredError,
    })
}
fn kind(value: &Value) -> Result<Kind> {
    Ok(if number(value, "Optimizer", 1)? == 0 {
        Kind::AdamW
    } else {
        Kind::Sgd
    })
}

struct Tensor {
    id: TensorId,
    context: u32,
    optimizer: Option<u32>,
}
#[derive(Default)]
struct Transport {
    graph: Graph,
    contexts: BTreeSet<u32>,
    available: BTreeSet<u32>,
    tensors: BTreeMap<u32, Tensor>,
    buffers: BTreeMap<u32, u32>,
    datasets: BTreeMap<u32, Dataset>,
    optimizers: BTreeMap<u32, (Optimizer, u32)>,
    results: BTreeMap<u32, Metrics>,
    saved: BTreeMap<u32, Vec<f32>>,
}
impl Transport {
    fn tensor(&self, action: &Value, name: &str, storage: bool) -> Result<TensorId> {
        let tensor = self
            .tensors
            .get(&handle(action, name)?)
            .ok_or_else(|| invalid("unknown optimizer tensor handle"))?;
        if storage && !self.available.contains(&tensor.context) {
            return Err(invalid(
                "optimizer tensor storage lifetime ended or was not allocated",
            ));
        }
        Ok(tensor.id)
    }
    fn context(&self, action: &Value) -> Result<u32> {
        let context = handle(action, "Context")?;
        if !self.contexts.contains(&context) {
            return Err(invalid("unknown optimizer context"));
        }
        Ok(context)
    }
    fn training(&self, action: &Value, period: usize) -> Result<Training> {
        Ok(Training {
            context: self.context(action)?,
            inputs: self.tensor(action, "Inputs", true)?,
            outputs: self.tensor(action, "Outputs", false)?,
            loss: loss(action)?,
            kind: kind(action)?,
            period,
            parameters: parameters(field(action, "Parameters")?)?,
        })
    }
    fn create_tensor(&mut self, action: &Value) -> Result<()> {
        exact_fields(
            action,
            &["Op", "Handle", "Context", "Dimensions", "Parameter", "Name"],
        )?;
        let handle = handle(action, "Handle")?;
        if self.tensors.len() >= 256 || self.tensors.contains_key(&handle) {
            return Err(invalid("optimizer tensor handle exists or exceeds bound"));
        }
        let dimensions = array(action, "Dimensions")?
            .iter()
            .map(|value| {
                Ok(usize::try_from(value.as_u64().ok_or_else(|| {
                    invalid("optimizer dimensions must be unsigned")
                })?)?)
            })
            .collect::<Result<Vec<_>>>()?;
        let context = self.context(action)?;
        let id = self.graph.tensor(
            context,
            &dimensions,
            boolean(action, "Parameter")?,
            text(action, "Name")?,
        )?;
        self.tensors.insert(
            handle,
            Tensor {
                id,
                context,
                optimizer: None,
            },
        );
        Ok(())
    }
    fn create_node(&mut self, action: &Value) -> Result<()> {
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
        let id = handle(action, "Handle")?;
        if self.tensors.len() >= 256 || self.tensors.contains_key(&id) {
            return Err(invalid("optimizer node handle exists or exceeds bound"));
        }
        let context = self.context(action)?;
        let operator = match text(action, "Operation")? {
            "add" => Operator::Add,
            "mul" => Operator::Multiply,
            "scale" => Operator::Scale(float(field(action, "FactorBits")?)?),
            "adamw-step" => Operator::AdamW,
            "sgd-step" => Operator::Sgd,
            _ => return Err(invalid("unknown optimizer graph operation")),
        };
        if !matches!(operator, Operator::Scale(_)) && !action["FactorBits"].is_null() {
            return Err(invalid("non-scale factor must be null"));
        }
        let inputs = array(action, "Inputs")?
            .iter()
            .map(|value| {
                let handle = u32::try_from(
                    value
                        .as_u64()
                        .ok_or_else(|| invalid("optimizer input handle must be unsigned"))?,
                )?;
                self.tensors
                    .get(&handle)
                    .map(|tensor| tensor.id)
                    .ok_or_else(|| invalid("unknown optimizer input tensor"))
            })
            .collect::<Result<Vec<_>>>()?;
        let tensor = self
            .graph
            .operation(context, operator, &inputs, text(action, "Name")?)?;
        self.tensors.insert(
            id,
            Tensor {
                id: tensor,
                context,
                optimizer: None,
            },
        );
        Ok(())
    }
    fn create_dataset(&mut self, action: &Value) -> Result<Value> {
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
        let id = handle(action, "Handle")?;
        if self.datasets.len() >= 16 || self.datasets.contains_key(&id) {
            return Err(invalid("optimizer dataset handle exists or exceeds bound"));
        }
        let labels = floats(&json!({"DataHex":field(action, "LabelsHex")?}))?;
        let count = number(action, "Count", 4096)?;
        let dataset = Dataset::new(
            number(action, "DatapointElements", 64)?,
            number(action, "LabelElements", 64)?,
            count,
            number(action, "Shard", count)?,
            floats(action)?,
            labels,
        )?;
        self.datasets.insert(id, dataset);
        Ok(json!({"Count":count}))
    }
    fn epoch(&mut self, action: &Value) -> Result<()> {
        exact_fields(
            action,
            &["Op", "Optimizer", "Dataset", "Train", "Evaluate", "Split"],
        )?;
        let target = |name| -> Result<Option<u32>> {
            if field(action, name)?.is_null() {
                Ok(None)
            } else {
                Ok(Some(handle(action, name)?))
            }
        };
        let train = target("Train")?;
        let evaluate = target("Evaluate")?;
        let mut train_metrics = train
            .map(|id| {
                self.results
                    .remove(&id)
                    .ok_or_else(|| invalid("unknown optimizer result"))
            })
            .transpose()?;
        let shared = train.is_some() && train == evaluate;
        let mut evaluate_metrics = if shared {
            Some(Metrics::default())
        } else {
            evaluate
                .map(|id| {
                    self.results
                        .remove(&id)
                        .ok_or_else(|| invalid("unknown optimizer result"))
                })
                .transpose()?
        };
        let (optimizer, context) = self
            .optimizers
            .get_mut(&handle(action, "Optimizer")?)
            .ok_or_else(|| invalid("unknown optimizer handle"))?;
        let dataset = self
            .datasets
            .get(&handle(action, "Dataset")?)
            .ok_or_else(|| invalid("unknown optimizer dataset"))?;
        optimizer.epoch(
            &mut self.graph,
            dataset,
            train_metrics.as_mut(),
            evaluate_metrics.as_mut(),
            number(action, "Split", 4096)?,
        )?;
        self.available.insert(*context);
        if shared {
            train_metrics
                .as_mut()
                .ok_or_else(|| invalid("missing shared result"))?
                .append(
                    evaluate_metrics
                        .as_ref()
                        .ok_or_else(|| invalid("missing shared result"))?,
                )?;
        }
        if let (Some(id), Some(metrics)) = (train, train_metrics) {
            self.results.insert(id, metrics);
        }
        if !shared && let (Some(id), Some(metrics)) = (evaluate, evaluate_metrics) {
            self.results.insert(id, metrics);
        }
        Ok(())
    }
    fn free(&mut self, action: &Value) -> Result<()> {
        exact_fields(action, &["Op", "Type", "Handle"])?;
        let id = handle(action, "Handle")?;
        match text(action, "Type")? {
            "optimizer" => {
                let (_, context) = self
                    .optimizers
                    .remove(&id)
                    .ok_or_else(|| invalid("unknown optimizer handle"))?;
                self.available.remove(&context);
                self.tensors
                    .retain(|_, tensor| tensor.optimizer != Some(id));
            }
            "dataset" => {
                self.datasets
                    .remove(&id)
                    .ok_or_else(|| invalid("unknown dataset handle"))?;
            }
            "result" => {
                self.results
                    .remove(&id)
                    .ok_or_else(|| invalid("unknown result handle"))?;
            }
            "buffer" => {
                if !self.optimizers.is_empty() {
                    return Err(invalid("optimizer still retains tensor buffers"));
                }
                let context = self
                    .buffers
                    .remove(&id)
                    .ok_or_else(|| invalid("unknown buffer handle"))?;
                self.available.remove(&context);
            }
            "context" => {
                if self.optimizers.values().any(|(_, context)| *context == id)
                    || self.buffers.values().any(|&context| context == id)
                    || !self.contexts.remove(&id)
                {
                    return Err(invalid("optimizer context is missing or still retained"));
                }
                self.available.remove(&id);
                self.tensors.retain(|_, tensor| tensor.context != id);
            }
            _ => return Err(invalid("unknown optimizer free type")),
        }
        Ok(())
    }
    fn fit(&mut self, action: &Value) -> Result<()> {
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
        let training = Training {
            context: self.context(action)?,
            inputs: self.tensor(action, "Inputs", true)?,
            outputs: self.tensor(action, "Outputs", false)?,
            loss: loss(action)?,
            kind: kind(action)?,
            period: 1,
            parameters: Parameters::default(),
        };
        boolean(action, "Silent")?;
        let schedule = if action["Parameters"].is_null() {
            None
        } else {
            Some(
                array(action, "Parameters")?
                    .iter()
                    .map(parameters)
                    .collect::<Result<Vec<_>>>()?,
            )
        };
        let options = Fit {
            epochs: number(action, "Epochs", 4096)?,
            logical_batch: number(action, "LogicalBatch", 4096)?,
            validation_split: float(field(action, "ValidationBits")?)?,
            parameters: schedule,
        };
        fit(
            &mut self.graph,
            training,
            self.datasets
                .get_mut(&handle(action, "Dataset")?)
                .ok_or_else(|| invalid("unknown optimizer dataset"))?,
            &options,
        )?;
        self.available.remove(&training.context);
        Ok(())
    }

    fn action(&mut self, action: &Value) -> Result<Value> {
        let mut event = json!({});
        match text(action, "Op")? {
            "context" => {
                exact_fields(action, &["Op", "Handle", "Bytes"])?;
                let id = handle(action, "Handle")?;
                if number(action, "Bytes", 8 * 1024 * 1024)? == 0
                    || self.contexts.len() >= 16
                    || !self.contexts.insert(id)
                {
                    return Err(invalid("optimizer context exists or exceeds bound"));
                }
                event = json!({"Created":true});
            }
            "tensor" => self.create_tensor(action)?,
            "node" => self.create_node(action)?,
            "allocate" => {
                exact_fields(action, &["Op", "Handle", "Context"])?;
                let id = handle(action, "Handle")?;
                let context = self.context(action)?;
                if self.buffers.len() >= 16 || self.buffers.contains_key(&id) {
                    return Err(invalid("optimizer buffer exists or exceeds bound"));
                }
                self.buffers.insert(id, context);
                self.available.insert(context);
            }
            "set" => {
                exact_fields(action, &["Op", "Tensor", "DataHex"])?;
                self.graph
                    .set_values(self.tensor(action, "Tensor", true)?, &floats(action)?)?;
            }
            "get" => {
                exact_fields(action, &["Op", "Tensor"])?;
                event = json!({"DataHex":float_hex(&self.graph.values(self.tensor(action, "Tensor", true)?)?)});
            }
            "save" | "restore" => {
                exact_fields(action, &["Op", "Tensor", "Handle"])?;
                let id = handle(action, "Handle")?;
                let tensor = self.tensor(action, "Tensor", true)?;
                if text(action, "Op")? == "save" {
                    if self.saved.len() >= 16 && !self.saved.contains_key(&id) {
                        return Err(invalid("optimizer saved tensor count exceeds bound"));
                    }
                    self.saved.insert(id, self.graph.values(tensor)?);
                } else {
                    self.graph.set_values(
                        tensor,
                        self.saved
                            .get(&id)
                            .ok_or_else(|| invalid("unknown saved tensor"))?,
                    )?;
                }
            }
            "supports" => {
                exact_fields(action, &["Op", "Tensor"])?;
                event = json!({"Supported":self.graph.supports(self.tensor(action, "Tensor", false)?)?});
            }
            "dataset" => event = self.create_dataset(action)?,
            "params" => {
                exact_fields(action, &["Op", "Loss"])?;
                loss(action)?;
                event = json!({"HasCallback":true});
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
                let id = handle(action, "Handle")?;
                if self.optimizers.len() >= 16 || self.optimizers.contains_key(&id) {
                    return Err(invalid("optimizer handle exists or exceeds bound"));
                }
                let training = self.training(action, number(action, "Period", 1_000_000)?)?;
                let optimizer = Optimizer::new(&mut self.graph, training)?;
                self.optimizers.insert(id, (optimizer, training.context));
                event = json!({"Created":true,"Optimizer":action["Optimizer"],"HasCallback":true});
            }
            "result" => {
                exact_fields(action, &["Op", "Handle"])?;
                let id = handle(action, "Handle")?;
                if self.results.len() >= 16 || self.results.contains_key(&id) {
                    return Err(invalid("optimizer result handle exists or exceeds bound"));
                }
                self.results.insert(id, Metrics::default());
            }
            "result-reset" => {
                exact_fields(action, &["Op", "Result"])?;
                self.results
                    .get_mut(&handle(action, "Result")?)
                    .ok_or_else(|| invalid("unknown optimizer result"))?
                    .reset();
            }
            "result-snapshot" => {
                exact_fields(
                    action,
                    &["Op", "Result", "LossUncertainty", "AccuracyUncertainty"],
                )?;
                let result = self
                    .results
                    .get(&handle(action, "Result")?)
                    .ok_or_else(|| invalid("unknown optimizer result"))?;
                let (loss, loss_uncertainty) = result.loss();
                let (accuracy, accuracy_uncertainty) = result.accuracy();
                event = json!({"Count":result.count(),"LossBits":loss.to_bits(),"LossUncertaintyBits":boolean(action,"LossUncertainty")?.then(||loss_uncertainty.to_bits()),"AccuracyBits":accuracy.to_bits(),"AccuracyUncertaintyBits":boolean(action,"AccuracyUncertainty")?.then(||accuracy_uncertainty.to_bits())});
            }
            "alloc" | "reset" => {
                exact_fields(action, &["Op", "Optimizer", "Value"])?;
                let (optimizer, context) = self
                    .optimizers
                    .get_mut(&handle(action, "Optimizer")?)
                    .ok_or_else(|| invalid("unknown optimizer handle"))?;
                if text(action, "Op")? == "alloc" {
                    optimizer.allocate(&mut self.graph, boolean(action, "Value")?)?;
                    self.available.insert(*context);
                } else {
                    optimizer.reset(&mut self.graph, boolean(action, "Value")?)?;
                }
            }
            "eval" => {
                exact_fields(action, &["Op", "Optimizer", "Result"])?;
                let result = if action["Result"].is_null() {
                    None
                } else {
                    Some(
                        self.results
                            .get_mut(&handle(action, "Result")?)
                            .ok_or_else(|| invalid("unknown optimizer result"))?,
                    )
                };
                self.optimizers
                    .get_mut(&handle(action, "Optimizer")?)
                    .ok_or_else(|| invalid("unknown optimizer handle"))?
                    .0
                    .evaluate(&mut self.graph, result)?;
            }
            "loss" | "gradient" => {
                let gradient = text(action, "Op")? == "gradient";
                exact_fields(
                    action,
                    if gradient {
                        &["Op", "Optimizer", "Handle", "Tensor"]
                    } else {
                        &["Op", "Optimizer", "Handle"]
                    },
                )?;
                let id = handle(action, "Optimizer")?;
                let (optimizer, context) = self
                    .optimizers
                    .get(&id)
                    .ok_or_else(|| invalid("unknown optimizer handle"))?;
                let tensor = if gradient {
                    optimizer.gradient_tensor(self.tensor(action, "Tensor", false)?)?
                } else {
                    optimizer.loss_tensor()
                };
                self.tensors.insert(
                    handle(action, "Handle")?,
                    Tensor {
                        id: tensor,
                        context: *context,
                        optimizer: Some(id),
                    },
                );
            }
            "shuffle" => {
                exact_fields(action, &["Op", "Optimizer", "Dataset", "Count"])?;
                let count = field(action, "Count")?
                    .as_i64()
                    .ok_or_else(|| invalid("shuffle count must be signed"))?;
                if !(-1..=4096).contains(&count) {
                    return Err(invalid("shuffle count exceeds bound"));
                }
                self.optimizers
                    .get_mut(&handle(action, "Optimizer")?)
                    .ok_or_else(|| invalid("unknown optimizer handle"))?
                    .0
                    .shuffle(
                        self.datasets
                            .get_mut(&handle(action, "Dataset")?)
                            .ok_or_else(|| invalid("unknown optimizer dataset"))?,
                        usize::try_from(count).ok(),
                    )?;
            }
            "batch" => {
                exact_fields(action, &["Op", "Dataset", "Data", "Labels", "Index"])?;
                let dataset = self
                    .datasets
                    .get(&handle(action, "Dataset")?)
                    .ok_or_else(|| invalid("unknown dataset"))?;
                let data = self.tensor(action, "Data", true)?;
                let labels = self.tensor(action, "Labels", true)?;
                let elements = self.graph.values(data)?.len();
                let count = elements / dataset.width();
                if !elements.is_multiple_of(dataset.width())
                    || self.graph.values(labels)?.len() != count * dataset.label_width()
                {
                    return Err(invalid("dataset batch tensor extents differ"));
                }
                let (values, label_values) =
                    dataset.batch(count, number(action, "Index", 4095)?)?;
                self.graph.set_values(data, &values)?;
                self.graph.set_values(labels, &label_values)?;
            }
            "epoch" => self.epoch(action)?,
            "fit" => self.fit(action)?,
            "free" => self.free(action)?,
            _ => return Err(invalid("unknown optimizer action")),
        }
        Ok(event)
    }
}

fn operation(operation: &Value) -> Result<Value> {
    if text(operation, "Kind")? == "optimizer-info" {
        exact_fields(operation, &["Kind"])?;
        let parameters = Parameters::default();
        return Ok(
            json!({"Devices":[{"Index":0,"Name":"Rust CPU","Type":0}],"HardwareConcurrency":std::thread::available_parallelism()?.get(),"Defaults":{"AdamW":parameters.adamw.map(f32::to_bits),"Sgd":parameters.sgd.map(f32::to_bits)},"TensorOverhead":size_of::<TensorInfo>(),"GraphOverhead":size_of::<Graph>(),"DefaultGraphSize":2048}),
        );
    }
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
    let scheduler = array(operation, "Scheduler")?;
    if devices.is_empty()
        || devices.len() > 8
        || devices.iter().any(|device| device != 0)
        || scheduler.is_empty()
        || scheduler.len() > 16
        || scheduler.iter().any(|index| {
            index
                .as_u64()
                .is_none_or(|index| index >= u64::try_from(devices.len()).unwrap_or(0))
        })
    {
        return Err(invalid(
            "optimizer device or scheduler configuration exceeds inventory",
        ));
    }
    number(operation, "Backend", devices.len() - 1)?;
    number(operation, "Threads", 256)?;
    let actions = array(operation, "Actions")?;
    if actions.len() > 8192 {
        return Err(invalid("optimizer action count exceeds bound"));
    }
    let mut state = Transport::default();
    let trace = actions
        .iter()
        .enumerate()
        .map(|(index, action)| {
            state.action(action).map_err(|error| {
                invalid(format!(
                    "optimizer action {index} ({}): {error}",
                    action["Op"].as_str().unwrap_or("unknown")
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(json!({"Trace":trace}))
}

pub(crate) fn execute_runtime(request: &Value) -> Result<Option<Value>> {
    exact_fields(request, &["SchemaVersion", "Operations"])?;
    if uint(request, "SchemaVersion")? != 1 {
        return Err(invalid("optimizer protocol version differs"));
    }
    let operations = array(request, "Operations")?;
    if operations.is_empty() || operations.len() > 128 {
        return Err(invalid("optimizer operation count exceeds bound"));
    }
    if operations.iter().any(|operation| {
        !matches!(
            operation["Kind"].as_str(),
            Some("optimizer-info" | "optimizer-trace")
        )
    }) {
        return Ok(None);
    }
    let results = operations
        .iter()
        .map(operation)
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(json!({"SchemaVersion":1,"Results":results})))
}
