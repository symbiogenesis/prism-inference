//! Externally supervised, terminal original-Swift graph qualification worker.
//!
//! The Runner owner verifies source/package/driver identities and creates this
//! worker suspended in its deadline-owned, kill-on-close Job. This executable
//! grants no service admission. Up to two real graph tokens offload only the original
//! first FFN up projection; every native integer limb/plane and complete CPU logits
//! and logical state must match before the final result is published.
#![forbid(unsafe_code)]
#![cfg_attr(
    not(windows),
    allow(
        dead_code,
        unused_imports,
        reason = "The real driver graph worker is Windows-only; hosted non-Windows targets retain parser and oracle controls."
    )
)]

use std::{
    collections::BTreeMap,
    env,
    fs::{self, File},
    io::{self, Read as _, Write as _},
    path::{Component, Path, PathBuf},
    process::ExitCode,
    time::Duration,
};

use bonsai_hexagon_runtime::{GuardedQ2Executor, HmxQ2Executor, PlacedProjectionExecutor};
use bonsai_prism_runtime::{
    Decoder, Result,
    graph::{Qwen35, StateComponent},
    kernels::{MatvecBackend, Pq2ProjectionExecutor},
    model::{Model, swift},
};
use hexagon_native_rpc::{
    Driver, DriverLocation, Error, GuardedQ2Submitter, HmxQ2ProjectionLimits, HmxQ2Submitter,
    Q2_PROJECTION_MAX_OPERATIONS, Q2ProjectionLimits,
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};

const PROJECTION: &str = "blk.0.ffn_up.weight";
const MAX_REQUEST_BYTES: u64 = 8192;
const MODEL_MEMORY_LIMIT: usize = 12 * 1024 * 1024 * 1024;
const MAX_HOST_PAYLOAD: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
enum ExecutionBackend {
    #[default]
    GuardedHvx,
    HmxBitplane,
}

impl ExecutionBackend {
    const fn name(self) -> &'static str {
        match self {
            Self::GuardedHvx => "guarded-hvx",
            Self::HmxBitplane => "hmx-bitplane",
        }
    }

    const fn values_per_token(self) -> u64 {
        match self {
            Self::GuardedHvx => 1_392_640,
            Self::HmxBitplane => 5_570_560,
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Request {
    schema_version: u32,
    model_path: PathBuf,
    package_directory: PathBuf,
    token_id: u32,
    #[serde(default)]
    token_ids: Option<Vec<u32>>,
    #[serde(default)]
    execution_backend: ExecutionBackend,
    context_limit: usize,
    model_memory_limit_bytes: usize,
    state_budget_bytes: usize,
    session_seconds: u64,
    maximum_operations: u32,
    maximum_owned_payload_bytes: usize,
    expected_parent_pid: u32,
    expected_parent_creation_time: u64,
}

fn reject() -> bonsai_prism_runtime::Error {
    io::Error::other("hexagon-model-probe-contract-rejected").into()
}

fn require(value: bool) -> Result<()> {
    if value { Ok(()) } else { Err(reject()) }
}

impl Request {
    fn tokens(&self) -> &[u32] {
        self.token_ids
            .as_deref()
            .unwrap_or_else(|| std::slice::from_ref(&self.token_id))
    }

    fn limits(&self) -> Result<Q2ProjectionLimits> {
        let tokens = self.tokens();
        require(
            self.schema_version == 1
                && (1..=2).contains(&tokens.len())
                && tokens.first() == Some(&self.token_id)
                && tokens.iter().all(|&token| {
                    usize::try_from(token).is_ok_and(|token| token < swift::VOCAB_SIZE)
                })
                && (1..=16).contains(&self.context_limit)
                && self.context_limit >= tokens.len()
                && (swift::TENSOR_PAYLOAD_BYTES..=MODEL_MEMORY_LIMIT)
                    .contains(&self.model_memory_limit_bytes)
                && (1..=bonsai_prism_runtime::graph::MAX_STATE_BYTES)
                    .contains(&self.state_budget_bytes)
                && (1..=MAX_HOST_PAYLOAD).contains(&self.maximum_owned_payload_bytes)
                && self.maximum_operations >= u32::try_from(tokens.len())? * 40
                && self.expected_parent_pid != 0
                && self.expected_parent_creation_time != 0,
        )?;
        Ok(Q2ProjectionLimits::new(
            17_408,
            self.maximum_operations,
            Duration::from_secs(self.session_seconds),
        )?)
    }

    fn validate(&self) -> Result<Q2ProjectionLimits> {
        let limits = self.limits()?;
        external(&self.model_path)?;
        external(&self.package_directory)?;
        Ok(limits)
    }
}

fn external(path: &Path) -> Result<()> {
    require(path.is_absolute())?;
    for part in path.components() {
        match part {
            Component::CurDir | Component::ParentDir => return Err(reject()),
            Component::Normal(name) => require(
                !name.to_string_lossy().contains(':')
                    && !name.to_string_lossy().chars().any(char::is_control),
            )?,
            _ => {}
        }
    }
    #[cfg(windows)]
    require(
        matches!(path.components().next(), Some(Component::Prefix(prefix))
        if matches!(prefix.kind(), std::path::Prefix::Disk(_))),
    )?;
    for ancestor in path.ancestors() {
        let metadata = fs::symlink_metadata(ancestor)?;
        require(!metadata.file_type().is_symlink())?;
        #[cfg(windows)]
        {
            use std::os::windows::fs::MetadataExt as _;
            require(metadata.file_attributes() & 0x400 == 0)?;
        }
        if metadata.is_dir() {
            require(!ancestor.join(".git").try_exists()?)?;
        }
    }
    Ok(())
}

fn emit(value: &Value) -> Result<()> {
    let mut output = io::stdout().lock();
    serde_json::to_writer(&mut output, value)?;
    output.write_all(b"\n")?;
    output.flush()?;
    Ok(())
}

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .flat_map(|byte| [byte >> 4, byte & 15])
        .map(|value| char::from(b"0123456789abcdef"[usize::from(value)]))
        .collect()
}

fn hash_values(hash: &mut Sha256, values: &[f32]) -> Result<()> {
    require(values.iter().all(|value| value.is_finite()))?;
    hash.update(u64::try_from(values.len())?.to_le_bytes());
    for value in values {
        hash.update(value.to_bits().to_le_bytes());
    }
    Ok(())
}

fn state_digest(graph: &Qwen35) -> Result<String> {
    let mut hash = Sha256::new();
    hash.update(b"hexagon-live-model-state-v1\0");
    graph.visit_state(&mut |block, component, values| {
        hash.update(u64::try_from(block)?.to_le_bytes());
        let (tag, position) = match component {
            StateComponent::Key(position) => (0_u8, position),
            StateComponent::Value(position) => (1, position),
            StateComponent::Convolution => (2, 0),
            StateComponent::Recurrent => (3, 0),
        };
        hash.update([tag]);
        hash.update(u64::try_from(position)?.to_le_bytes());
        hash_values(&mut hash, values)
    })?;
    Ok(hex_digest(hash))
}

fn hex_digest(hash: Sha256) -> String {
    hash.finalize()
        .iter()
        .flat_map(|byte| [byte >> 4, byte & 15])
        .map(|value| char::from(b"0123456789abcdef"[usize::from(value)]))
        .collect()
}

fn logits_digest(logits: &[f32]) -> Result<String> {
    let mut hash = Sha256::new();
    hash_values(&mut hash, logits)?;
    Ok(hex_digest(hash))
}

fn shared_model(model: &Model) -> Model {
    Model {
        metadata: model.metadata.clone(),
        tensors: model.tensors.clone(),
        identity: model.identity.clone(),
    }
}

fn power_value() -> Value {
    hexagon_native_rpc::power_status().map_or_else(
        |_| json!({"available":false,"acConnected":null,"batterySaver":null,"batteryPercent":null}),
        |status| {
            json!({"available":true,"acConnected":status.ac_connected,
            "batterySaver":status.battery_saver,"batteryPercent":status.battery_percent})
        },
    )
}

struct StateRow {
    block: usize,
    component: StateComponent,
    bits: Vec<u32>,
}

fn capture_state(graph: &Qwen35, maximum_bytes: usize) -> Result<Vec<StateRow>> {
    let mut retained = 0_usize;
    let mut rows = Vec::new();
    graph.visit_state(&mut |block, component, values| {
        retained = values
            .len()
            .checked_mul(4)
            .and_then(|bytes| retained.checked_add(bytes))
            .filter(|&bytes| bytes <= maximum_bytes)
            .ok_or_else(reject)?;
        let mut bits = Vec::new();
        bits.try_reserve_exact(values.len())?;
        bits.extend(values.iter().map(|value| value.to_bits()));
        rows.try_reserve(1)?;
        rows.push(StateRow {
            block,
            component,
            bits,
        });
        Ok(())
    })?;
    Ok(rows)
}

fn compare_state(graph: &Qwen35, expected: &[StateRow]) -> Result<()> {
    let mut index = 0;
    graph.visit_state(&mut |block, component, values| {
        let row = expected.get(index).ok_or_else(reject)?;
        require(
            block == row.block
                && component == row.component
                && values.len() == row.bits.len()
                && values
                    .iter()
                    .zip(&row.bits)
                    .all(|(value, &bits)| value.to_bits() == bits),
        )?;
        index += 1;
        Ok(())
    })?;
    require(index == expected.len())
}

struct Control {
    logits: Vec<f32>,
    state_sha256: String,
    rows: Vec<StateRow>,
}

fn cpu_control(model: Model, request: &Request) -> Result<Vec<Control>> {
    let mut graph = Qwen35::with_state_budget(
        model,
        request.context_limit,
        request.state_budget_bytes,
        MatvecBackend::Scalar,
    )?;
    let mut controls = Vec::new();
    let mut remaining_state_bytes = request.state_budget_bytes;
    controls.try_reserve_exact(request.tokens().len())?;
    for &token in request.tokens() {
        let logits = graph.evaluate(token)?;
        let rows = capture_state(&graph, remaining_state_bytes)?;
        for row in &rows {
            remaining_state_bytes = remaining_state_bytes
                .checked_sub(row.bits.len().checked_mul(4).ok_or_else(reject)?)
                .ok_or_else(reject)?;
        }
        controls.push(Control {
            logits,
            state_sha256: state_digest(&graph)?,
            rows,
        });
    }
    Ok(controls)
}

/// Independent scalar oracle over each signed operation's serialized operands.
struct CheckedSubmitter<'a> {
    native: &'a mut dyn GuardedQ2Submitter,
    compared: u64,
    failed: bool,
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    reason = "Validated integer activations and bounded integer dots are exactly representable in i32 and F32."
)]
fn compare_limbs(weights: &[u8], activation: &[[f32; 256]; 2], output: &[f32]) -> Result<u64> {
    let rows = output.len() / 2;
    require(rows != 0 && output.len() == rows * 2 && weights.len() == rows * 72)?;
    require(rows.is_multiple_of(32) && rows <= 17_408)?;
    for tile in weights.as_chunks::<576>().0 {
        require(
            tile[512..]
                .as_chunks::<2>()
                .0
                .iter()
                .all(|scale| *scale == [0, 0x3c]),
        )?;
    }
    for (limb, values) in activation.iter().enumerate() {
        require(
            values
                .iter()
                .all(|value| value.is_finite() && value.fract() == 0.0 && value.abs() <= 127.0),
        )?;
        for row in 0..rows {
            let mut dot = 0_i32;
            for (lane, &value) in values.iter().enumerate() {
                let local = lane % 64;
                let tile = (row / 32 * 4 + lane / 64) * 576;
                let byte = weights[tile + local / 16 * 128 + row % 32 * 4 + local % 4];
                let symbol = (byte >> (local / 4 % 4 * 2)) & 3;
                dot += (i32::from(symbol) - 1) * value as i32;
            }
            // Integer zero has one mathematical value; native -0 is accepted
            // here and subsequent complete graph checks still compare bits.
            require(output[limb * rows + row] == dot as f32)?;
        }
    }
    Ok(u64::try_from(output.len())?)
}

impl GuardedQ2Submitter for CheckedSubmitter<'_> {
    fn remaining_operations(&self) -> Option<u64> {
        if self.failed {
            Some(0)
        } else {
            self.native.remaining_operations()
        }
    }

    fn submit(
        &mut self,
        weights: &[u8],
        activation: &[[f32; 256]; 2],
        output: &mut [f32],
    ) -> std::result::Result<(), Error> {
        if self.failed {
            return Err(Error::Contract);
        }
        // Mark terminal before calling the native owner so any failure, including
        // an oracle mismatch, makes later submission impossible without replay.
        self.failed = true;
        self.native.submit(weights, activation, output)?;
        let count = compare_limbs(weights, activation, output).map_err(|_| Error::Contract)?;
        self.compared = self.compared.checked_add(count).ok_or(Error::Contract)?;
        self.failed = false;
        Ok(())
    }
}

/// Independent decoding of the serialized signed-kernel Q2 tiles and all eight
/// binary activation planes. This does not reuse the production HMX packer or
/// reconstruction helpers.
#[allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    reason = "Binary activation planes and bounded integer sums are exactly representable in F32."
)]
fn compare_planes(weights: &[u8], activation: &[[f32; 128]; 8], output: &[f32]) -> Result<u64> {
    let rows = output.len() / 8;
    require(
        rows != 0
            && output.len() == rows * 8
            && rows.is_multiple_of(32)
            && rows <= 17_408
            && weights.len() == rows * 36,
    )?;
    for tile in weights.as_chunks::<576>().0 {
        require(
            tile[512..]
                .as_chunks::<2>()
                .0
                .iter()
                .all(|scale| *scale == [0, 0x3c]),
        )?;
    }
    for (plane, values) in activation.iter().enumerate() {
        require(values.iter().all(|&value| value == 0.0 || value == 1.0))?;
        for row in 0..rows {
            let mut dot = 0_i32;
            for (lane, &value) in values.iter().enumerate() {
                if value == 1.0 {
                    let local = lane % 64;
                    let tile = (row / 32 * 2 + lane / 64) * 576;
                    let byte = weights[tile + local / 16 * 128 + row % 32 * 4 + local % 4];
                    let symbol = (byte >> (local / 4 % 4 * 2)) & 3;
                    dot += i32::from(symbol) - 1;
                }
            }
            require(output[plane * rows + row] == dot as f32)?;
        }
    }
    Ok(u64::try_from(output.len())?)
}

struct CheckedHmxSubmitter<'a> {
    native: &'a mut dyn HmxQ2Submitter,
    compared: u64,
    failed: bool,
}

impl HmxQ2Submitter for CheckedHmxSubmitter<'_> {
    fn remaining_operations(&self) -> Option<u64> {
        if self.failed {
            Some(0)
        } else {
            self.native.remaining_operations()
        }
    }

    fn submit(
        &mut self,
        weights: &[u8],
        activation: &[[f32; 128]; 8],
        output: &mut [f32],
    ) -> std::result::Result<(), Error> {
        if self.failed {
            output.fill(f32::NAN);
            return Err(Error::Contract);
        }
        self.failed = true;
        let result = (|| {
            self.native.submit(weights, activation, output)?;
            let count = compare_planes(weights, activation, output).map_err(|_| Error::Contract)?;
            self.compared = self.compared.checked_add(count).ok_or(Error::Contract)?;
            Ok(())
        })();
        if result.is_ok() {
            self.failed = false;
        } else {
            output.fill(f32::NAN);
        }
        result
    }
}

#[cfg(windows)]
fn compare_graph_sequence(
    graph: Qwen35,
    executor: &mut dyn Pq2ProjectionExecutor,
    placement_model: &Model,
    controls: &[Control],
    request: &Request,
    parent: &bonsai_windows_process::ParentProcess,
) -> Result<Value> {
    require(controls.len() == request.tokens().len())?;
    let mut placed = PlacedProjectionExecutor::new(executor, placement_model, &[PROJECTION])?;
    let mut projected = graph.with_projection_executor(&mut placed)?;
    // Reserve the complete sequence before the first submitted projection.
    projected.admit_work(0, request.tokens().len())?;
    let mut comparisons = Vec::new();
    comparisons.try_reserve_exact(controls.len())?;
    for (index, (&token, control)) in request.tokens().iter().zip(controls).enumerate() {
        parent.ensure_alive()?;
        let actual = projected.evaluate(token)?;
        require(
            actual.len() == control.logits.len()
                && actual
                    .iter()
                    .zip(&control.logits)
                    .all(|(actual, expected)| actual.to_bits() == expected.to_bits()),
        )?;
        compare_state(projected.decoder(), &control.rows)?;
        let actual_state = state_digest(projected.decoder())?;
        require(actual_state == control.state_sha256)?;
        comparisons.push(json!({"index":index,"tokenId":token,
            "logitsBitwiseEqual":true,"stateBitwiseEqual":true,
            "logitsSha256":logits_digest(&actual)?,"stateSha256":actual_state}));
    }
    let last = comparisons.last().ok_or_else(reject)?;
    Ok(
        json!({"placementIdentity":projected.identity(),"tokenComparisons":comparisons,
        "logitsBitwiseEqual":true,"stateBitwiseEqual":true,
        "logitsSha256":last["logitsSha256"],"stateSha256":last["stateSha256"]}),
    )
}

fn check_statistics(
    statistics: bonsai_hexagon_runtime::ProjectionStatistics,
    compared: u64,
    failed: bool,
    request: &Request,
) -> Result<()> {
    let tokens = u64::try_from(request.tokens().len())?;
    require(
        !failed
            && compared == request.execution_backend.values_per_token() * tokens
            && statistics.projections_completed == tokens
            && statistics.operations_completed == tokens * 40
            && statistics.operations_attempted == tokens * 40,
    )
}

#[cfg(windows)]
#[allow(
    clippy::too_many_lines,
    reason = "Keep the complete terminal CPU/native comparison and post-cleanup publication sequence auditable together."
)]
fn run_request(request: &Request, request_sha256: &str) -> Result<()> {
    let limits = request.validate()?;
    let parent = bonsai_windows_process::ParentProcess::open(
        request.expected_parent_pid,
        request.expected_parent_creation_time,
    )?;
    parent.ensure_alive()?;
    emit(
        &json!({"event":"started", "schemaVersion":1, "requestSha256":request_sha256,
        "processId":std::process::id(), "modelSha256":swift::MODEL_SHA256,
        "modelBytes":swift::MODEL_BYTES, "projection":PROJECTION}),
    )?;
    let (model, _) = swift::load(&request.model_path, request.model_memory_limit_bytes)?;
    let selected = model.tensors.get(PROJECTION).ok_or_else(reject)?;
    require(selected.kind == 142 && selected.dimensions == [5120, 17_408])?;
    let shadow = shared_model(&model);
    let placement_model = Model {
        metadata: BTreeMap::default(),
        tensors: [(PROJECTION.to_owned(), selected.clone())].into(),
        identity: model.identity.clone(),
    };
    // Shared serialized weights are never copied. The shadow graph is dropped
    // before native setup; bounded logical-state bits and CPU logits survive.
    let cpu_power_before = power_value();
    let control = cpu_control(shadow, request);
    let cpu_power_after = power_value();
    emit(
        &json!({"event":"cpu-arm-finished", "completed":control.is_ok(),
        "powerBefore":cpu_power_before,"powerAfter":cpu_power_after}),
    )?;
    let controls = control?;
    let last_control = controls.last().ok_or_else(reject)?;
    parent.ensure_alive()?;
    emit(
        &json!({"event":"cpu-control-complete", "logitsSha256":logits_digest(&last_control.logits)?,
        "stateSha256":last_control.state_sha256,"tokenIds":request.tokens()}),
    )?;
    let graph = Qwen35::with_state_budget(
        model,
        request.context_limit,
        request.state_budget_bytes,
        MatvecBackend::Scalar,
    )?;
    parent.ensure_alive()?;
    let npu_power_before = power_value();
    let mut result = None;
    let mut progress_failed = false;
    let report_result = (|| {
        let driver = Driver::open_system()?;
        let driver_dll = driver.path().to_path_buf();
        let progress = |report: &hexagon_native_rpc::Q2ProjectionReport| {
            if emit(
                &json!({"event":"native-progress", "stage":report.queue.stage,
                "operationsSubmitted":report.operations_submitted,
                "operationsCompleted":report.operations_completed}),
            )
            .is_err()
            {
                progress_failed = true;
            }
        };
        let mut accept = |mut value: Value,
                          statistics: bonsai_hexagon_runtime::ProjectionStatistics,
                          compared: u64| {
            parent.ensure_alive()?;
            value["driverDll"] = json!(driver_dll);
            value["projectionsCompleted"] = json!(statistics.projections_completed);
            value["integerValuesCompared"] = json!(compared);
            result = Some(value);
            Ok::<(), bonsai_prism_runtime::Error>(())
        };
        match request.execution_backend {
            ExecutionBackend::GuardedHvx => driver.reference_q2_projection(
                &request.package_directory,
                limits,
                progress,
                |native| {
                    let work = (|| -> Result<()> {
                        let mut checked = CheckedSubmitter {
                            native,
                            compared: 0,
                            failed: false,
                        };
                        let (value, statistics) = {
                            let mut executor = GuardedQ2Executor::new(
                                &mut checked,
                                limits,
                                request.maximum_owned_payload_bytes,
                            );
                            let value = compare_graph_sequence(
                                graph,
                                &mut executor,
                                &placement_model,
                                &controls,
                                request,
                                &parent,
                            )?;
                            (value, executor.statistics())
                        };
                        check_statistics(statistics, checked.compared, checked.failed, request)?;
                        accept(value, statistics, checked.compared)
                    })();
                    work.map_err(|_| Error::Contract)
                },
            ),
            ExecutionBackend::HmxBitplane => {
                let hmx_limits = HmxQ2ProjectionLimits::new(
                    17_408,
                    request.maximum_operations,
                    Duration::from_secs(request.session_seconds),
                )?;
                driver.reference_hmx_q2_projection(
                    &request.package_directory,
                    hmx_limits,
                    progress,
                    |native| {
                        let work = (|| -> Result<()> {
                            let mut checked = CheckedHmxSubmitter {
                                native,
                                compared: 0,
                                failed: false,
                            };
                            let (value, statistics) = {
                                let mut executor = HmxQ2Executor::new(
                                    &mut checked,
                                    hmx_limits,
                                    request.maximum_owned_payload_bytes,
                                );
                                let value = compare_graph_sequence(
                                    graph,
                                    &mut executor,
                                    &placement_model,
                                    &controls,
                                    request,
                                    &parent,
                                )?;
                                (value, executor.statistics())
                            };
                            check_statistics(
                                statistics,
                                checked.compared,
                                checked.failed,
                                request,
                            )?;
                            accept(value, statistics, checked.compared)
                        })();
                        work.map_err(|_| Error::Contract)
                    },
                )
            }
        }
    })();
    let npu_power_after = power_value();
    emit(
        &json!({"event":"native-arm-finished", "returnedReport":report_result.is_ok(),
        "powerBefore":npu_power_before,"powerAfter":npu_power_after}),
    )?;
    let report = report_result?;
    require(!progress_failed && report.completed_cycle())?;
    require(
        report.is_hmx_bitplane_control()
            == (request.execution_backend == ExecutionBackend::HmxBitplane),
    )?;
    let mut value = result.ok_or_else(reject)?;
    value["event"] = json!("result");
    value["schemaVersion"] = json!(1);
    value["completed"] = json!(true);
    value["requestSha256"] = json!(request_sha256);
    value["modelSha256"] = json!(swift::MODEL_SHA256);
    value["modelBytes"] = json!(swift::MODEL_BYTES);
    value["projection"] = json!(PROJECTION);
    value["tokenId"] = json!(request.token_id);
    value["tokenIds"] = json!(request.tokens());
    value["executionBackend"] = json!(request.execution_backend.name());
    value["maximumOperations"] = json!(request.maximum_operations);
    value["sessionSeconds"] = json!(request.session_seconds);
    value["nativeHmxBitplaneControl"] = json!(report.is_hmx_bitplane_control());
    value["profileEligible"] = json!(false);
    value["performanceMeasured"] = json!(false);
    value["cpuPowerBefore"] = cpu_power_before;
    value["cpuPowerAfter"] = cpu_power_after;
    value["npuPowerBefore"] = npu_power_before;
    value["nativeCompletedCycle"] = json!(true);
    value["nativeOperationsSubmitted"] = json!(report.operations_submitted);
    value["nativeOperationsCompleted"] = json!(report.operations_completed);
    value["nativeStage"] = json!(report.queue.stage);
    value["nativeCleanup"] = json!({
        "queueStopResult":report.queue.queue_stop_result,
        "queueCloseResult":report.queue.queue_close_result,
        "unmappingResult":report.queue.unmapping_result,
        "moduleCloseResult":report.queue.module_close_result,
        "freeCompleted":report.queue.free_completed,
        "poisoned":report.queue.poisoned,
        "ownerPinnedUntilExit":report.queue.owner_pinned_until_exit});
    value["nativeResources"] = json!({
        "mappedAllocationBytes":report.mapped_allocation_bytes,
        "architecture":report.queue.metadata.architecture,
        "effectiveDomain":report.queue.metadata.effective_domain,
        "residentWeightBytesPrepared":report.resident_weight_bytes_prepared,
        "effectiveHmxPlacementVerified":false,"effectiveVtcmBytes":null});
    value["npuPowerAfter"] = npu_power_after;
    parent.ensure_alive()?;
    emit(&value)
}

fn main_work() -> Result<()> {
    let arguments: Vec<_> = env::args_os().skip(1).collect();
    if arguments.len() == 1 && arguments[0] == "--self-test" {
        require(Q2_PROJECTION_MAX_OPERATIONS >= 24_000)?;
        Q2ProjectionLimits::new(17_408, 24_000, Duration::from_secs(120))?;
        require(Q2ProjectionLimits::new(17_408, 24_001, Duration::from_secs(120)).is_err())?;
        return emit(
            &json!({"event":"self-test", "schemaVersion":1,"valid":true,"driverLoaded":false,"modelLoaded":false}),
        );
    }
    if arguments.len() == 1 && arguments[0] == "--inspect" {
        let location = DriverLocation::inspect_system()?;
        return emit(
            &json!({"event":"inspection","kind":"native-fastrpc-inspection",
            "driverDll":location.path(),"driverLoaded":false,"dspCodeLoaded":false,"profileEligible":false}),
        );
    }
    require(arguments.len() == 2 && arguments[0] == "--request")?;
    let path = PathBuf::from(&arguments[1]);
    external(&path)?;
    let mut bytes = Vec::new();
    File::open(path)?
        .take(MAX_REQUEST_BYTES + 1)
        .read_to_end(&mut bytes)?;
    require(bytes.len() <= usize::try_from(MAX_REQUEST_BYTES)?)?;
    let request: Request = serde_json::from_slice(&bytes)?;
    request.validate()?;
    #[cfg(windows)]
    {
        run_request(&request, &digest(&bytes))
    }
    #[cfg(not(windows))]
    {
        let _ = (request, digest(&bytes));
        Err(Error::UnsupportedPlatform.into())
    }
}

fn main() -> ExitCode {
    main_work().map_or_else(
        |_| {
            let _ = emit(&json!({"event":"error","schemaVersion":1,"completed":false,
                "reason":"hexagon-model-probe-rejected","profileEligible":false}));
            ExitCode::FAILURE
        },
        |()| ExitCode::SUCCESS,
    )
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, rc::Rc};

    use super::*;

    fn request() -> Value {
        json!({"schemaVersion":1,"modelPath":"C:\\model.gguf","packageDirectory":"C:\\runtime",
            "tokenId":0,"contextLimit":1,"modelMemoryLimitBytes":MODEL_MEMORY_LIMIT,
            "stateBudgetBytes":512*1024*1024,"sessionSeconds":120,"maximumOperations":24_000,
            "maximumOwnedPayloadBytes":MAX_HOST_PAYLOAD,"expectedParentPid":1,
            "expectedParentCreationTime":1})
    }

    #[test]
    fn model_probe_request_rejects_unknown_fields_and_infeasible_limits() -> Result<()> {
        let value = request();
        serde_json::from_value::<Request>(value.clone())?.limits()?;
        let mut unknown = value.clone();
        unknown["modelSha256"] = json!("caller-supplied-model-is-not-admitted");
        assert!(serde_json::from_value::<Request>(unknown).is_err());
        for (name, invalid) in [
            ("maximumOperations", json!(24_001)),
            ("maximumOperations", json!(39)),
            ("contextLimit", json!(17)),
            ("sessionSeconds", json!(121)),
            ("tokenId", json!(swift::VOCAB_SIZE)),
            ("expectedParentPid", json!(0)),
            ("maximumOwnedPayloadBytes", json!(MAX_HOST_PAYLOAD + 1)),
        ] {
            let mut changed = value.clone();
            changed[name] = invalid;
            assert!(
                serde_json::from_value::<Request>(changed)?
                    .limits()
                    .is_err()
            );
        }
        Ok(())
    }

    #[test]
    fn model_probe_sequence_requires_bounded_tokens_context_and_complete_work() -> Result<()> {
        let mut two = request();
        two["tokenIds"] = json!([0, 42]);
        two["contextLimit"] = json!(2);
        two["maximumOperations"] = json!(80);
        two["executionBackend"] = json!("hmx-bitplane");
        let parsed: Request = serde_json::from_value(two.clone())?;
        parsed.limits()?;
        assert_eq!(parsed.tokens(), &[0, 42]);
        assert_eq!(parsed.execution_backend.name(), "hmx-bitplane");
        for (name, invalid) in [
            ("tokenIds", json!([])),
            ("tokenIds", json!([1, 42])),
            ("tokenIds", json!([0, 1, 2])),
            ("tokenIds", json!([0, swift::VOCAB_SIZE])),
            ("contextLimit", json!(1)),
            ("maximumOperations", json!(79)),
        ] {
            let mut changed = two.clone();
            changed[name] = invalid;
            assert!(
                serde_json::from_value::<Request>(changed)?
                    .limits()
                    .is_err()
            );
        }
        two["executionBackend"] = json!("automatic");
        assert!(serde_json::from_value::<Request>(two).is_err());
        Ok(())
    }

    fn zero_hmx_weights(rows: usize) -> Vec<u8> {
        let mut weights = vec![0x55; rows * 36];
        for tile in weights.as_chunks_mut::<576>().0 {
            for scale in tile[512..].as_chunks_mut::<2>().0 {
                *scale = [0, 0x3c];
            }
        }
        weights
    }

    #[test]
    fn model_probe_hmx_oracle_checks_all_planes_row_tiles_signs_and_output_bytes() -> Result<()> {
        let mut weights = zero_hmx_weights(64);
        weights[0] = (weights[0] & !3) | 2; // Row0, lane0: +1.
        weights[133] &= !3; // Row1, lane17: -1.
        weights[1728] = (weights[1728] & !3) | 3; // Row32, lane64: +2.
        weights[2239] = (weights[2239] & 0x3f) | (2 << 6); // Row63, lane127: +1.
        let mut planes = [[0.0; 128]; 8];
        planes[0][0] = 1.0;
        planes[0][17] = 1.0;
        planes[1][64] = 1.0;
        planes[7][127] = 1.0;
        let mut output = [0.0; 512];
        output[0] = 1.0;
        output[1] = -1.0;
        output[96] = 2.0;
        output[511] = 1.0;
        assert_eq!(compare_planes(&weights, &planes, &output)?, 512);
        let mut wrong = output;
        wrong[511] = 0.0;
        assert!(compare_planes(&weights, &planes, &wrong).is_err());
        wrong[511] = f32::NAN;
        assert!(compare_planes(&weights, &planes, &wrong).is_err());
        assert!(compare_planes(&weights, &planes, &output[..504]).is_err());
        planes[3][5] = 0.5;
        assert!(compare_planes(&weights, &planes, &output).is_err());
        planes[3][5] = 0.0;
        weights[512] = 1;
        assert!(compare_planes(&weights, &planes, &output).is_err());
        Ok(())
    }

    struct HmxSubmitter {
        calls: Rc<Cell<u32>>,
        fail: bool,
        wrong: bool,
    }

    impl HmxQ2Submitter for HmxSubmitter {
        fn remaining_operations(&self) -> Option<u64> {
            Some(80)
        }

        fn submit(
            &mut self,
            _: &[u8],
            _: &[[f32; 128]; 8],
            output: &mut [f32],
        ) -> std::result::Result<(), Error> {
            self.calls.set(self.calls.get() + 1);
            output.fill(if self.wrong { 1.0 } else { 0.0 });
            if self.fail {
                Err(Error::Contract)
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn model_probe_hmx_native_failure_and_plane_mismatch_close_without_replay() {
        for (fail, wrong) in [(true, false), (false, true)] {
            let calls = Rc::new(Cell::new(0));
            let mut native = HmxSubmitter {
                calls: calls.clone(),
                fail,
                wrong,
            };
            let mut checked = CheckedHmxSubmitter {
                native: &mut native,
                compared: 0,
                failed: false,
            };
            let weights = zero_hmx_weights(32);
            let mut output = [f32::NAN; 256];
            for _ in 0..2 {
                assert!(
                    checked
                        .submit(&weights, &[[1.0; 128]; 8], &mut output)
                        .is_err()
                );
                assert!(output.iter().all(|value| value.is_nan()));
            }
            assert_eq!(checked.remaining_operations(), Some(0));
            assert_eq!(checked.compared, 0);
            assert_eq!(calls.get(), 1);
        }
    }

    #[test]
    fn model_probe_hmx_success_counts_every_plane_across_submissions() -> Result<()> {
        let mut native = HmxSubmitter {
            calls: Rc::new(Cell::new(0)),
            fail: false,
            wrong: false,
        };
        let mut checked = CheckedHmxSubmitter {
            native: &mut native,
            compared: 0,
            failed: false,
        };
        for _ in 0..2 {
            checked.submit(
                &zero_hmx_weights(32),
                &[[1.0; 128]; 8],
                &mut [f32::NAN; 256],
            )?;
        }
        assert_eq!(checked.compared, 512);
        assert!(!checked.failed);
        Ok(())
    }

    fn zero_weights_rows(rows: usize) -> Vec<u8> {
        let mut weights = vec![0x55; rows * 72];
        for tile in weights.as_chunks_mut::<576>().0 {
            for scale in tile[512..].as_chunks_mut::<2>().0 {
                *scale = [0, 0x3c];
            }
        }
        weights
    }

    fn zero_weights() -> Vec<u8> {
        zero_weights_rows(32)
    }

    #[test]
    fn model_probe_oracle_decodes_nonzero_rows_lanes_and_signed_limbs() -> Result<()> {
        let mut weights = zero_weights();
        weights[0] = (weights[0] & !3) | 2; // Row zero, lane zero, symbol +1.
        weights[133] &= !3; // Row one, lane seventeen, symbol -1.
        let mut activation = [[0.0; 256]; 2];
        activation[0][0] = 2.0;
        activation[1][0] = -3.0;
        activation[0][17] = 4.0;
        activation[1][17] = 5.0;
        let mut output = [0.0; 64];
        output[0] = 2.0;
        output[1] = -4.0;
        output[32] = -3.0;
        output[33] = -5.0;
        assert_eq!(compare_limbs(&weights, &activation, &output)?, 64);
        let mut weights = zero_weights_rows(64);
        // Second row tile: row 32 exercises each of the four K=64 tiles.
        for (offset, symbol) in [(2304, 2_u8), (2880, 0), (3456, 3), (4032, 2)] {
            weights[offset] = (weights[offset] & !3) | symbol;
        }
        // Row 63, lane 255 reaches the last byte and highest two-bit field.
        weights[4543] = (weights[4543] & 0x3f) | (2 << 6);
        let mut activation = [[0.0; 256]; 2];
        for (lane, first, second) in [
            (0, 2.0, -2.0),
            (64, 3.0, 1.0),
            (128, 4.0, 3.0),
            (192, 5.0, -5.0),
            (255, 7.0, -8.0),
        ] {
            activation[0][lane] = first;
            activation[1][lane] = second;
        }
        let mut output = [0.0; 128];
        output[32] = 12.0;
        output[63] = 7.0;
        output[96] = -2.0;
        output[127] = -8.0;
        assert_eq!(compare_limbs(&weights, &activation, &output)?, 128);
        Ok(())
    }

    #[test]
    fn model_probe_oracle_rejects_plausible_wrong_unwritten_and_partial_results() -> Result<()> {
        let weights = zero_weights();
        let activation = [[1.0; 256]; 2];
        let output = [0.0; 64];
        assert_eq!(compare_limbs(&weights, &activation, &output)?, 64);
        let mut wrong = output;
        wrong[63] = 1.0;
        assert!(compare_limbs(&weights, &activation, &wrong).is_err());
        wrong[63] = f32::NAN;
        assert!(compare_limbs(&weights, &activation, &wrong).is_err());
        assert!(compare_limbs(&weights, &activation, &output[..62]).is_err());
        let mut changed = weights;
        changed[512] = 1;
        assert!(compare_limbs(&changed, &activation, &output).is_err());
        Ok(())
    }

    struct Submitter {
        calls: Rc<Cell<u32>>,
        fail: bool,
        wrong: bool,
    }

    impl GuardedQ2Submitter for Submitter {
        fn remaining_operations(&self) -> Option<u64> {
            Some(40)
        }

        fn submit(
            &mut self,
            _: &[u8],
            _: &[[f32; 256]; 2],
            output: &mut [f32],
        ) -> std::result::Result<(), Error> {
            self.calls.set(self.calls.get() + 1);
            output.fill(if self.wrong { 1.0 } else { 0.0 });
            if self.fail {
                Err(Error::Contract)
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn model_probe_native_failure_and_oracle_mismatch_are_terminal_without_replay() {
        for (fail, wrong) in [(true, false), (false, true)] {
            let calls = Rc::new(Cell::new(0));
            let mut native = Submitter {
                calls: calls.clone(),
                fail,
                wrong,
            };
            let mut checked = CheckedSubmitter {
                native: &mut native,
                compared: 0,
                failed: false,
            };
            let weights = zero_weights();
            let mut output = [f32::NAN; 64];
            assert!(
                checked
                    .submit(&weights, &[[1.0; 256]; 2], &mut output)
                    .is_err()
            );
            assert!(
                checked
                    .submit(&weights, &[[1.0; 256]; 2], &mut output)
                    .is_err()
            );
            assert_eq!(checked.remaining_operations(), Some(0));
            assert_eq!(checked.compared, 0);
            assert_eq!(calls.get(), 1);
        }
    }

    #[test]
    fn model_probe_success_counts_every_independently_compared_limb() -> Result<()> {
        let mut native = Submitter {
            calls: Rc::new(Cell::new(0)),
            fail: false,
            wrong: false,
        };
        let mut checked = CheckedSubmitter {
            native: &mut native,
            compared: 0,
            failed: false,
        };
        for _ in 0..2 {
            checked.submit(&zero_weights(), &[[1.0; 256]; 2], &mut [f32::NAN; 64])?;
        }
        assert_eq!(checked.compared, 128);
        assert!(!checked.failed);
        Ok(())
    }
}
